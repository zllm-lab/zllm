//! Kimi-K3 单层平台无关编排。

use crate::{
    attention::{
        attn_res::{self, AttnResBackend, AttnResState},
        kda::{KdaInputs, KdaKernel, KdaSpec, KdaState, KdaWeightsRef, kda},
        mla::GatedMlaSpec,
    },
    backend::{BackendError, DecodeBackend, MlaPrefillBackend},
};

/// projection → depthwise short-conv 是一条串行数据路径。
pub struct KdaConvPath<W> {
    pub projection: W,
    pub convolution: W,
}

/// hidden → f_a → f_b → decay activation 是一条串行数据路径。
pub struct KdaDecayPath<W> {
    pub first_projection: W,
    pub second_projection: W,
    pub a_log: W,
    pub dt_bias: W,
}

/// recurrent output 与独立 output gate 汇合后，再进入 output projection。
pub struct KdaOutputPath<W> {
    pub gate_projection: W,
    pub norm: W,
    pub projection: W,
}

pub struct KdaWeights<W> {
    pub query: KdaConvPath<W>,
    pub key: KdaConvPath<W>,
    pub value: KdaConvPath<W>,
    pub decay: KdaDecayPath<W>,
    pub beta_projection: W,
    pub output: KdaOutputPath<W>,
}

pub struct AttnResWeights<W> {
    pub norm: W,
    pub projection: W,
}

pub struct KdaLayerWeights<W> {
    pub input_norm: W,
    pub attention_residual: AttnResWeights<W>,
    pub attention: KdaWeights<W>,
    pub post_attention_norm: W,
    pub mlp_residual: AttnResWeights<W>,
}

/// hidden → q_a → RMSNorm → q_b 是一条串行 query 路径。
pub struct MlaQueryPath<W> {
    pub first_projection: W,
    pub norm: W,
    pub second_projection: W,
}

/// hidden → kv_a 后拆分 latent/rope；latent → RMSNorm → kv_b。
pub struct MlaKvPath<W> {
    pub first_projection: W,
    pub norm: W,
    pub second_projection: W,
}

pub struct GatedMlaOutputPath<W> {
    pub gate_projection: W,
    pub projection: W,
}

pub struct GatedMlaWeights<W> {
    pub query: MlaQueryPath<W>,
    pub kv: MlaKvPath<W>,
    pub output: GatedMlaOutputPath<W>,
}

pub struct GatedMlaLayerWeights<W> {
    pub input_norm: W,
    pub attention_residual: AttnResWeights<W>,
    pub attention: GatedMlaWeights<W>,
    pub post_attention_norm: W,
    pub mlp_residual: AttnResWeights<W>,
}

#[allow(clippy::too_many_arguments)]
pub fn kda_attention<B: KdaKernel>(backend: &B, state: &mut KdaState<B::KdaStorage>, layer: usize, position: usize, input: &B::Tensor, weights: &KdaWeights<B::Weight>, spec: &KdaSpec) -> Result<B::Tensor, BackendError> {
    // 六条首级投影互不依赖，按 dual capability 成对提交，让设备后端保留并行空间。
    let (query, key) = backend.dual_linear(input, &weights.query.projection, &weights.key.projection)?;
    let (value, beta) = backend.dual_linear(input, &weights.value.projection, &weights.beta_projection)?;
    let (decay_hidden, output_gate) = backend.dual_linear(input, &weights.decay.first_projection, &weights.output.gate_projection)?;
    let decay = backend.linear(&decay_hidden, &weights.decay.second_projection)?;
    let inputs = KdaInputs { query: &query, key: &key, value: &value, decay: &decay, beta: &beta, output_gate: &output_gate };
    let recurrent_weights =
        KdaWeightsRef { query_conv: &weights.query.convolution, key_conv: &weights.key.convolution, value_conv: &weights.value.convolution, a_log: &weights.decay.a_log, dt_bias: &weights.decay.dt_bias, output_norm: &weights.output.norm };
    let recurrent = kda(backend, state, layer, position, inputs, recurrent_weights, spec)?;
    backend.linear(&recurrent, &weights.output.projection)
}

#[allow(clippy::too_many_arguments)]
pub fn gated_mla_prefill<B: MlaPrefillBackend>(
    backend: &B,
    cache: Option<&mut B::Cache>,
    layer: usize,
    position: usize,
    input: &B::Tensor,
    weights: &GatedMlaWeights<B::Weight>,
    spec: &GatedMlaSpec,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
) -> Result<B::Tensor, BackendError> {
    if !spec.output_gate {
        return Err(BackendError::Compute { msg: "Gated MLA runtime 需要 output_gate=true".to_owned() });
    }
    let mla = &spec.mla;
    let (query_hidden, compressed_kv) = backend.dual_linear(input, &weights.query.first_projection, &weights.kv.first_projection)?;
    let output_gate = backend.linear(input, &weights.output.gate_projection)?;
    let query_hidden = backend.rmsnorm(&query_hidden, &weights.query.norm, rms_eps)?;
    let query = backend.linear(&query_hidden, &weights.query.second_projection)?;
    let (latent, key_rope) = backend.split_columns(&compressed_kv, mla.kv_lora_rank)?;
    let latent = backend.rmsnorm(&latent, &weights.kv.norm, rms_eps)?;
    let (query, key_rope) = if spec.use_rope {
        (backend.rope(&query, mla.num_heads, mla.qk_rope_head_dim, mla.rotary_layout, position, cos, sin)?, backend.rope(&key_rope, 1, mla.qk_rope_head_dim, mla.rotary_layout, position, cos, sin)?)
    } else {
        (query, key_rope)
    };
    let attention = backend.mla_prefill_attention(&query, &latent, &key_rope, &weights.kv.second_projection, cache, layer, mla)?;
    let attention = backend.sigmoid_gate(&attention, &output_gate)?;
    backend.linear(&attention, &weights.output.projection)
}

#[allow(clippy::too_many_arguments)]
pub fn gated_mla_decode<B: DecodeBackend>(
    backend: &B,
    cache: &mut B::Cache,
    layer: usize,
    position: usize,
    input: &B::Tensor,
    weights: &GatedMlaWeights<B::Weight>,
    spec: &GatedMlaSpec,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
) -> Result<B::Tensor, BackendError> {
    if !spec.output_gate {
        return Err(BackendError::Compute { msg: "Gated MLA runtime 需要 output_gate=true".to_owned() });
    }
    let mla = &spec.mla;
    let (query_hidden, compressed_kv) = backend.dual_linear(input, &weights.query.first_projection, &weights.kv.first_projection)?;
    let output_gate = backend.linear(input, &weights.output.gate_projection)?;
    let query_hidden = backend.rmsnorm(&query_hidden, &weights.query.norm, rms_eps)?;
    let query = backend.linear(&query_hidden, &weights.query.second_projection)?;
    let (latent, key_rope) = backend.split_columns(&compressed_kv, mla.kv_lora_rank)?;
    let latent = backend.rmsnorm(&latent, &weights.kv.norm, rms_eps)?;
    let (query, key_rope) = if spec.use_rope {
        (backend.rope(&query, mla.num_heads, mla.qk_rope_head_dim, mla.rotary_layout, position, cos, sin)?, backend.rope(&key_rope, 1, mla.qk_rope_head_dim, mla.rotary_layout, position, cos, sin)?)
    } else {
        (query, key_rope)
    };
    backend.append_mla(cache, layer, &latent, &key_rope)?;
    let end = position.checked_add(backend.token_rows(input)).ok_or_else(|| BackendError::Compute { msg: "Gated MLA position + rows 溢出".to_owned() })?;
    let attention = backend.mla_decode_attention(&query, cache, &weights.kv.second_projection, layer, end, mla)?;
    let attention = backend.sigmoid_gate(&attention, &output_gate)?;
    backend.linear(&attention, &weights.output.projection)
}

#[allow(clippy::too_many_arguments)]
fn attn_res_layer<B, A, F>(
    backend: &B,
    attn_res_state: &mut AttnResState<B::Tensor>,
    layer: usize,
    hidden: B::Tensor,
    input_norm: &B::Weight,
    attention_residual: &AttnResWeights<B::Weight>,
    post_attention_norm: &B::Weight,
    mlp_residual: &AttnResWeights<B::Weight>,
    rms_eps: f32,
    attention: A,
    feedforward: F,
) -> Result<B::Tensor, BackendError>
where
    B: AttnResBackend,
    A: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    attn_res_state.begin_layer(layer)?;
    let attention_mixed = if attn_res_state.block_count() == 0 { None } else { Some(attn_res::mix(backend, attn_res_state, &hidden, &attention_residual.norm, &attention_residual.projection, rms_eps)?) };
    let attention_input = attention_mixed.as_ref().unwrap_or(&hidden);
    let attention_input = backend.rmsnorm(attention_input, input_norm, rms_eps)?;
    let attention = attention(&attention_input)?;

    let prefix = if attn_res_state.is_block_start(layer) {
        attn_res_state.push(hidden);
        attention
    } else {
        backend.add(&hidden, &attention)?
    };
    let mlp_input = attn_res::mix(backend, attn_res_state, &prefix, &mlp_residual.norm, &mlp_residual.projection, rms_eps)?;
    let mlp_input = backend.rmsnorm_f32(&mlp_input, post_attention_norm, rms_eps)?;
    let feedforward = feedforward(&mlp_input)?;
    let output = backend.add(&prefix, &feedforward)?;
    attn_res_state.finish_layer();
    Ok(output)
}

/// 执行一个 KDA 层；FFN 由调用方注入，因此 dense 与 LatentMoE 不复制 AttnRes 流程。
#[allow(clippy::too_many_arguments)]
pub fn kda_layer<B, F>(
    backend: &B,
    kda_state: &mut KdaState<B::KdaStorage>,
    attn_res_state: &mut AttnResState<B::Tensor>,
    layer: usize,
    position: usize,
    hidden: B::Tensor,
    weights: &KdaLayerWeights<B::Weight>,
    spec: &KdaSpec,
    rms_eps: f32,
    feedforward: F,
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + AttnResBackend,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    attn_res_layer(
        backend,
        attn_res_state,
        layer,
        hidden,
        &weights.input_norm,
        &weights.attention_residual,
        &weights.post_attention_norm,
        &weights.mlp_residual,
        rms_eps,
        |input| kda_attention(backend, kda_state, layer, position, input, &weights.attention, spec),
        feedforward,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_mla_prefill_layer<B, F>(
    backend: &B,
    cache: Option<&mut B::Cache>,
    attn_res_state: &mut AttnResState<B::Tensor>,
    layer: usize,
    position: usize,
    hidden: B::Tensor,
    weights: &GatedMlaLayerWeights<B::Weight>,
    spec: &GatedMlaSpec,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
    feedforward: F,
) -> Result<B::Tensor, BackendError>
where
    B: MlaPrefillBackend + AttnResBackend,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    attn_res_layer(
        backend,
        attn_res_state,
        layer,
        hidden,
        &weights.input_norm,
        &weights.attention_residual,
        &weights.post_attention_norm,
        &weights.mlp_residual,
        rms_eps,
        |input| gated_mla_prefill(backend, cache, layer, position, input, &weights.attention, spec, rms_eps, cos, sin),
        feedforward,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_mla_decode_layer<B, F>(
    backend: &B,
    cache: &mut B::Cache,
    attn_res_state: &mut AttnResState<B::Tensor>,
    layer: usize,
    position: usize,
    hidden: B::Tensor,
    weights: &GatedMlaLayerWeights<B::Weight>,
    spec: &GatedMlaSpec,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
    feedforward: F,
) -> Result<B::Tensor, BackendError>
where
    B: DecodeBackend + AttnResBackend,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    attn_res_layer(
        backend,
        attn_res_state,
        layer,
        hidden,
        &weights.input_norm,
        &weights.attention_residual,
        &weights.post_attention_norm,
        &weights.mlp_residual,
        rms_eps,
        |input| gated_mla_decode(backend, cache, layer, position, input, &weights.attention, spec, rms_eps, cos, sin),
        feedforward,
    )
}

/// 93 层结束后，官方模型还会用独立权重做一次 AttnRes 混合，再进入 output RMSNorm。
pub fn output_attn_res<B: AttnResBackend>(backend: &B, state: &AttnResState<B::Tensor>, hidden: &B::Tensor, weights: &AttnResWeights<B::Weight>, eps: f32) -> Result<B::Tensor, BackendError> {
    attn_res::mix(backend, state, hidden, &weights.norm, &weights.projection, eps)
}
