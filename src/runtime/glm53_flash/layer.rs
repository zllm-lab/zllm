//! GLM-5.3-Flash 单层平台无关编排。
//!
//! 每层两个 mHC 子层(attention / FFN)包裹注意力与 FFN;注意力为 KDA(线性)
//! 或 nope MLA + DSA 索引两者之一。FFN 由调用方注入,dense 与 MoE 不复制流程。

use crate::{
    attention::{
        dsa::{DsaSpec, DsaWeightsRef},
        hyper_connection::{HyperConnectionKernel, HyperConnectionSpec},
        kda::{KdaInputs, KdaKernel, KdaSpec, KdaState, KdaWeightsRef, kda},
        mla::{MlaSpec, mla_causal_prefill},
        rope::RotaryLayout,
    },
    backend::{Backend, BackendError, DsaPrefillBackend},
};

/// mHC 投影(prepared)。checkpoint 只有 fn/base/scale 三个张量;
/// `input_norm` 是 copies*hidden 宽的单位 RMSNorm,由平台 prepare 构造。
pub struct Glm53HyperConnection<W> {
    pub input_norm: W,
    pub function: W,
    pub base: W,
    pub scale: W,
}

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

/// 输出门 g_a → g_b 两步因子化(glm5_next 特有,K3 是单 g_proj 全秩门)。
pub struct KdaFactoredGatePath<W> {
    pub input_projection: W,
    pub output_projection: W,
}

pub struct Glm53KdaWeights<W> {
    pub query: KdaConvPath<W>,
    pub key: KdaConvPath<W>,
    pub value: KdaConvPath<W>,
    pub decay: KdaDecayPath<W>,
    pub beta_projection: W,
    pub gate: KdaFactoredGatePath<W>,
    pub output_norm: W,
    pub output_projection: W,
}

/// hidden → q_a → RMSNorm → q_b 是一条串行 query 路径。
pub struct MlaQueryPath<W> {
    pub first_projection: W,
    pub norm: W,
    pub second_projection: W,
}

/// hidden → kv_a(nope-only:输出即纯 latent)→ RMSNorm → kv_b。
pub struct MlaKvPath<W> {
    pub first_projection: W,
    pub norm: W,
    pub second_projection: W,
}

/// DSA 索引器:index query 从 q_lora 投影,index key/head weights 从层输入投影。
/// kpool>0 时 ape/gate 参与池内 softmax 加权(打包进 backend 的 kpool 状态)。
pub struct Glm53IndexerWeights<W> {
    pub query_projection: W,
    pub key_projection: W,
    pub head_weights_projection: W,
    pub key_norm_weight: W,
    pub key_norm_bias: W,
    /// 池内绝对位置嵌入 [kpool, head_dim]。
    pub kpool_ape: W,
    /// 池化 gate 投影 [head_dim, hidden]。
    pub kpool_gate: W,
}

pub struct Glm53DsaMlaWeights<W> {
    pub query: MlaQueryPath<W>,
    pub key_value: MlaKvPath<W>,
    pub indexer: Glm53IndexerWeights<W>,
    pub output_projection: W,
}

pub enum Glm53AttentionWeights<W> {
    Kda(Glm53KdaWeights<W>),
    DsaMla(Glm53DsaMlaWeights<W>),
}

pub struct Glm53LayerWeights<W> {
    pub input_norm: W,
    pub post_attention_norm: W,
    pub attention_hyper_connection: Glm53HyperConnection<W>,
    pub attention: Glm53AttentionWeights<W>,
    pub feedforward_hyper_connection: Glm53HyperConnection<W>,
}

/// MTP 层(layers.{layer_count})无 mHC,普通残差;注意力为 DSA-MLA。
pub struct Glm53MtpLayerWeights<W> {
    pub input_norm: W,
    pub post_attention_norm: W,
    pub attention: Glm53DsaMlaWeights<W>,
}

/// 官方 MTP 头:hnorm/enorm/eh_proj + 单层 transformer + shared_head.norm。
pub struct Glm53Mtp<W> {
    pub embedding_norm: W,
    pub hidden_norm: W,
    pub input_projection: W,
    pub layer: Glm53MtpLayerWeights<W>,
    pub output_norm: W,
}

/// mHC 子层:单位 norm → mixes 投影 → Sinkhorn split → 层 norm → 子层 → 展开合并。
/// `norm` 是层自身的 input/post_attention layernorm,作用在 reduced(hidden 宽)上。
#[allow(clippy::too_many_arguments)]
pub fn mhc_sublayer<B, F>(backend: &B, hidden: &B::Tensor, hyper_connection: &Glm53HyperConnection<B::Weight>, spec: &HyperConnectionSpec, rms_eps: f32, norm: &B::Weight, sublayer: F) -> Result<B::Tensor, BackendError>
where
    B: HyperConnectionKernel,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    let normalized_hidden = backend.rmsnorm(hidden, &hyper_connection.input_norm, rms_eps)?;
    let mixes = backend.linear(&normalized_hidden, &hyper_connection.function)?;
    let prepared = backend.hyper_connection_prepare_sublayer(hidden, &mixes, &hyper_connection.base, &hyper_connection.scale, spec)?;
    let normalized = backend.rmsnorm(&prepared.reduced, norm, rms_eps)?;
    let output = sublayer(&normalized)?;
    backend.hyper_connection_expand_scaled_add(&output, &prepared.post, &prepared.residual, spec.copies)
}

/// KDA 注意力:六条首级投影互不依赖,成对提交;decay 与输出门各含一步串行投影。
#[allow(clippy::too_many_arguments)]
pub fn glm53_kda_attention<B: KdaKernel>(backend: &B, state: &mut KdaState<B::KdaStorage>, layer: usize, position: usize, input: &B::Tensor, weights: &Glm53KdaWeights<B::Weight>, spec: &KdaSpec) -> Result<B::Tensor, BackendError> {
    let (query, key) = backend.dual_linear(input, &weights.query.projection, &weights.key.projection)?;
    let (value, beta) = backend.dual_linear(input, &weights.value.projection, &weights.beta_projection)?;
    // decay(f_a→f_b)与输出门(g_a→g_b)互不依赖,第一步并行、各自完成第二步。
    let (decay_hidden, gate_hidden) = backend.dual_linear(input, &weights.decay.first_projection, &weights.gate.input_projection)?;
    let decay = backend.linear(&decay_hidden, &weights.decay.second_projection)?;
    let output_gate = backend.linear(&gate_hidden, &weights.gate.output_projection)?;
    let inputs = KdaInputs { query: &query, key: &key, value: &value, decay: &decay, beta: &beta, output_gate: &output_gate };
    let recurrent_weights =
        KdaWeightsRef { query_conv: &weights.query.convolution, key_conv: &weights.key.convolution, value_conv: &weights.value.convolution, a_log: &weights.decay.a_log, dt_bias: &weights.decay.dt_bias, output_norm: &weights.output_norm };
    let recurrent = kda(backend, state, layer, position, inputs, recurrent_weights, spec)?;
    backend.linear(&recurrent, &weights.output_projection)
}

/// DSA 索引:key 路(norm→rope→追加)与 query 路(投影→rope)、head weights 三路独立。
///
/// glm5_next 特有:index key 走 kpool 压缩(每 `index_kpool` token 压成 1 条,
/// APE 位置嵌入 + gate 门控,权重 `indexer.index_kpool_compress_{ape,gate}`),
/// 语义由 backend DSA kernel 在 append/select 内实现;此处编排与 glm52 同构。
#[allow(clippy::too_many_arguments)]
fn glm53_dsa_select<B: DsaPrefillBackend>(
    backend: &B,
    state: &mut B::DsaState,
    hidden: &B::Tensor,
    query_lora: &B::Tensor,
    weights: &Glm53IndexerWeights<B::Weight>,
    layer: usize,
    position: usize,
    spec: &DsaSpec,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), BackendError> {
    let reference = DsaWeightsRef { wq_b: &weights.query_projection, wk: &weights.key_projection, weights_proj: &weights.head_weights_projection, k_norm_weight: &weights.key_norm_weight, k_norm_bias: &weights.key_norm_bias };
    let key = backend.linear(hidden, reference.wk)?;
    let key = backend.layernorm_bias(&key, reference.k_norm_weight, reference.k_norm_bias, 1.0e-6)?;
    // indexer 无 rope(dim=0)时直通,空表会被 ROCm RoPE kernel 拒绝。
    let key = if spec.rope_dim > 0 { backend.rope_prefix(&key, 1, spec.rope_dim, spec.rotary_layout, position, cos, sin)? } else { key };
    if spec.kpool > 0 {
        // 池化 gate 与 key 同步打包进 DSA 状态;gate 投影与 head weights 投影独立。
        let gate = backend.linear(hidden, &weights.kpool_gate)?;
        backend.append_dsa_keys_gated(state, layer, position, &key, &gate, spec)?;
    } else {
        backend.append_dsa_keys(state, layer, position, &key, spec)?;
    }
    let query = backend.linear(query_lora, reference.wq_b)?;
    let query = if spec.rope_dim > 0 { backend.rope_prefix(&query, spec.num_heads, spec.rope_dim, spec.rotary_layout, position, cos, sin)? } else { query };
    let head_weights = backend.linear(hidden, reference.weights_proj)?;
    backend.dsa_select_prefill(state, layer, &query, &head_weights, spec)
}

/// nope MLA + DSA:query 低秩路径与 kv 压缩并行;indexer 与主注意力共享它们的中间结果。
#[allow(clippy::too_many_arguments)]
pub fn glm53_dsa_mla_attention<B: DsaPrefillBackend>(
    backend: &B,
    cache: Option<&mut B::Cache>,
    dsa_state: &mut B::DsaState,
    layer: usize,
    position: usize,
    input: &B::Tensor,
    weights: &Glm53DsaMlaWeights<B::Weight>,
    mla: &MlaSpec,
    dsa: &DsaSpec,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
) -> Result<B::Tensor, BackendError> {
    let (query_lora, compressed_kv) = backend.dual_linear(input, &weights.query.first_projection, &weights.key_value.first_projection)?;
    let query_lora = backend.rmsnorm(&query_lora, &weights.query.norm, rms_eps)?;
    let query = backend.linear(&query_lora, &weights.query.second_projection)?;
    // nope-only:kv_a 输出即纯 latent 无 rope 段;qk_rope_head_dim=0 时
    // attention kernel 的 rope 循环零次,rope 参数传 latent 占位即可。
    if mla.qk_rope_head_dim > 0 {
        let (latent_raw, key_rope) = backend.split_columns(&compressed_kv, mla.kv_lora_rank)?;
        let latent = backend.rmsnorm(&latent_raw, &weights.key_value.norm, rms_eps)?;
        glm53_dsa_select(backend, dsa_state, input, &query_lora, &weights.indexer, layer, position, dsa, cos, sin)?;
        let attention = mla_causal_prefill(backend, &query, &latent, &key_rope, &weights.key_value.second_projection, cache, layer, mla, dsa, Some(dsa_state))?;
        backend.linear(&attention, &weights.output_projection)
    } else {
        let latent = backend.rmsnorm(&compressed_kv, &weights.key_value.norm, rms_eps)?;
        glm53_dsa_select(backend, dsa_state, input, &query_lora, &weights.indexer, layer, position, dsa, cos, sin)?;
        let attention = mla_causal_prefill(backend, &query, &latent, &latent, &weights.key_value.second_projection, cache, layer, mla, dsa, Some(dsa_state))?;
        backend.linear(&attention, &weights.output_projection)
    }
}

/// 执行一个 KDA 层:attention 与 FFN 各包一层 mHC;FFN 由调用方注入。
/// `observe` 在 attention 子层入口收到 (layer, 输入),供 speculative verify
/// 保存 anchor 行(行 0)用于拒绝后的递归状态重放。
#[allow(clippy::too_many_arguments)]
pub fn glm53_kda_layer<B, F>(
    backend: &B,
    kda_state: &mut KdaState<B::KdaStorage>,
    hyper_connection: &HyperConnectionSpec,
    weights: &Glm53LayerWeights<B::Weight>,
    kda_spec: &KdaSpec,
    layer: usize,
    position: usize,
    hidden: B::Tensor,
    rms_eps: f32,
    feedforward: F,
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + HyperConnectionKernel,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    glm53_kda_layer_with_observer(backend, kda_state, hyper_connection, weights, kda_spec, layer, position, hidden, rms_eps, feedforward, &mut |_| {})
}

#[allow(clippy::too_many_arguments)]
pub fn glm53_kda_layer_with_observer<B, F, O>(
    backend: &B,
    kda_state: &mut KdaState<B::KdaStorage>,
    hyper_connection: &HyperConnectionSpec,
    weights: &Glm53LayerWeights<B::Weight>,
    kda_spec: &KdaSpec,
    layer: usize,
    position: usize,
    hidden: B::Tensor,
    rms_eps: f32,
    feedforward: F,
    observe: &mut O,
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + HyperConnectionKernel,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
    O: FnMut(&B::Tensor),
{
    let attention_weights = match &weights.attention {
        Glm53AttentionWeights::Kda(weights) => weights,
        Glm53AttentionWeights::DsaMla(_) => return Err(BackendError::Compute { msg: "GLM-5.3-Flash KDA 层收到 DSA-MLA 权重".to_owned() }),
    };
    let hidden = mhc_sublayer(backend, &hidden, &weights.attention_hyper_connection, hyper_connection, rms_eps, &weights.input_norm, |input| {
        observe(input);
        glm53_kda_attention(backend, kda_state, layer, position, input, attention_weights, kda_spec)
    })?;
    mhc_sublayer(backend, &hidden, &weights.feedforward_hyper_connection, hyper_connection, rms_eps, &weights.post_attention_norm, feedforward)
}

/// 执行一个 DSA-MLA 层:mHC 包裹与 KDA 层一致,attention 子层换为稀疏 MLA。
#[allow(clippy::too_many_arguments)]
pub fn glm53_dsa_mla_layer<B, F>(
    backend: &B,
    cache: Option<&mut B::Cache>,
    dsa_state: &mut B::DsaState,
    hyper_connection: &HyperConnectionSpec,
    weights: &Glm53LayerWeights<B::Weight>,
    mla: &MlaSpec,
    dsa: &DsaSpec,
    layer: usize,
    position: usize,
    hidden: B::Tensor,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
    feedforward: F,
) -> Result<B::Tensor, BackendError>
where
    B: DsaPrefillBackend + HyperConnectionKernel,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    let attention_weights = match &weights.attention {
        Glm53AttentionWeights::DsaMla(weights) => weights,
        Glm53AttentionWeights::Kda(_) => return Err(BackendError::Compute { msg: "GLM-5.3-Flash DSA 层收到 KDA 权重".to_owned() }),
    };
    let hidden = mhc_sublayer(backend, &hidden, &weights.attention_hyper_connection, hyper_connection, rms_eps, &weights.input_norm, |input| {
        glm53_dsa_mla_attention(backend, cache, dsa_state, layer, position, input, attention_weights, mla, dsa, rms_eps, cos, sin)
    })?;
    mhc_sublayer(backend, &hidden, &weights.feedforward_hyper_connection, hyper_connection, rms_eps, &weights.post_attention_norm, feedforward)
}

/// 多模态包裹下第 0 层前的入口展开:把 hidden(1×hidden)扩展为 copies 份残差流。
pub fn glm53_expand_hidden<B: HyperConnectionKernel>(backend: &B, input: &B::Tensor, copies: usize) -> Result<B::Tensor, BackendError> {
    backend.hyper_connection_expand(input, copies)
}

/// 45 层收尾:copies 残差流折叠回 hidden 宽(官方 `hc_head = mean`)。
/// final RMSNorm 由 OutputHead 权重承担;RMSNorm 尺度不变,
/// 全 1 系数 reduce 与 mean 在 norm 后等价,省去除法。
pub fn glm53_collapse_hidden<B: HyperConnectionKernel>(backend: &B, hidden: &B::Tensor, copies: usize) -> Result<B::Tensor, BackendError> {
    backend.hyper_connection_collapse(hidden, copies)
}

/// MTP 层:普通残差(无 mHC),attention 子层复用 DSA-MLA;FFN 由调用方注入。
#[allow(clippy::too_many_arguments)]
pub fn glm53_mtp_layer<B, F>(
    backend: &B,
    cache: Option<&mut B::Cache>,
    dsa_state: &mut B::DsaState,
    weights: &Glm53MtpLayerWeights<B::Weight>,
    mla: &MlaSpec,
    dsa: &DsaSpec,
    layer: usize,
    position: usize,
    input: B::Tensor,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
    feedforward: F,
) -> Result<B::Tensor, BackendError>
where
    B: DsaPrefillBackend,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    let normed = backend.rmsnorm(&input, &weights.input_norm, rms_eps)?;
    let attention = glm53_dsa_mla_attention(backend, cache, dsa_state, layer, position, &normed, &weights.attention, mla, dsa, rms_eps, cos, sin)?;
    let hidden = backend.add(&input, &attention)?;
    let normed = backend.rmsnorm(&hidden, &weights.post_attention_norm, rms_eps)?;
    let feedforward = feedforward(&normed)?;
    backend.add(&hidden, &feedforward)
}

/// MTP 追赶只写入下一次草稿需要的 MLA/DSA 状态；当前行输出不会再被消费，
/// 因此不执行 query、attention、输出投影与 MoE。
#[allow(clippy::too_many_arguments)]
pub fn glm53_mtp_cache<B: DsaPrefillBackend>(
    backend: &B,
    cache: &mut B::Cache,
    dsa_state: &mut B::DsaState,
    mtp: &Glm53Mtp<B::Weight>,
    mla: &MlaSpec,
    dsa: &DsaSpec,
    layer: usize,
    position: usize,
    embedding: &B::Tensor,
    previous_hidden: &B::Tensor,
    hidden_size: usize,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), BackendError> {
    let input = glm53_mtp_project(backend, mtp, embedding, previous_hidden, hidden_size, rms_eps)?;
    let normalized = backend.rmsnorm(&input, &mtp.layer.input_norm, rms_eps)?;
    let compressed_kv = backend.linear(&normalized, &mtp.layer.attention.key_value.first_projection)?;
    if mla.qk_rope_head_dim > 0 {
        let (latent, key_rope) = backend.split_columns(&compressed_kv, mla.kv_lora_rank)?;
        let latent = backend.rmsnorm(&latent, &mtp.layer.attention.key_value.norm, rms_eps)?;
        backend.append_mla(cache, layer, &latent, &key_rope)?;
    } else {
        let latent = backend.rmsnorm(&compressed_kv, &mtp.layer.attention.key_value.norm, rms_eps)?;
        backend.append_mla(cache, layer, &latent, &latent)?;
    }

    let indexer = &mtp.layer.attention.indexer;
    let key = backend.linear(&normalized, &indexer.key_projection)?;
    let key = backend.layernorm_bias(&key, &indexer.key_norm_weight, &indexer.key_norm_bias, 1.0e-6)?;
    let key = if dsa.rope_dim > 0 { backend.rope_prefix(&key, 1, dsa.rope_dim, dsa.rotary_layout, position, cos, sin)? } else { key };
    if dsa.kpool > 0 {
        let gate = backend.linear(&normalized, &indexer.kpool_gate)?;
        backend.append_dsa_keys_gated(dsa_state, layer, position, &key, &gate, dsa)
    } else {
        backend.append_dsa_keys(dsa_state, layer, position, &key, dsa)
    }
}

/// MTP 入口:归一化 embedding/主干 hidden 拼接后直接投影到 MTP block。
pub fn glm53_mtp_project<B: Backend>(backend: &B, mtp: &Glm53Mtp<B::Weight>, embedding: &B::Tensor, hidden: &B::Tensor, hidden_size: usize, rms_eps: f32) -> Result<B::Tensor, BackendError> {
    crate::runtime::mtp_project(backend, embedding, hidden, &mtp.embedding_norm, &mtp.hidden_norm, &mtp.input_projection, hidden_size, crate::norm::NormSpec::Rms { eps: rms_eps }, false)
}

/// 保持 RotaryLayout 引用可被发现;glm5_next 的 MLA 不用 rope,
/// 仅 indexer 在 rope_dim > 0 时按 Interleaved 前缀旋转。
#[allow(dead_code)]
const INDEXER_ROTARY_LAYOUT: RotaryLayout = RotaryLayout::Interleaved;
