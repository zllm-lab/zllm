//! Gemma 4 dense text runtime。模型层顺序只在这里出现，backend 只提供既有算子能力。

#[cfg(feature = "with-cuda")]
pub mod cuda_mtp;
#[cfg(feature = "with-cuda")]
pub mod cuda_node;
pub mod engine;
#[cfg(target_os = "macos")]
pub(crate) mod metal_mtp;
#[cfg(target_os = "macos")]
mod metal_session;
pub mod multimodal;
pub mod protocol;

/// MTP draft 头的 RoPE 表上限：ensure_mtp 按它封顶构建，请求跨度（prompt +
/// completion）越过它的请求由 engine 级禁用 MTP 回落普通重放 decode——draft
/// kernel 按表索引，越界读会楔死 GPU 命令队列。
pub const MTP_MAX_POSITIONS: usize = 8192;

use half::{bf16, f16};

use crate::{
    attention::{
        gqa::{CausalWindow, GqaSpec, HybridGqaLayerSpec},
        rope::{RopeSpec, RopeTable},
    },
    backend::{Backend, BackendError, GqaPrefillBackend, LinearWeight},
    moe::{
        Activation,
        dense_mlp::{DenseMlpSpec, DenseMlpWeightsRef, forward_observed},
    },
    weight::{
        container::safetensor::TensorData,
        model::gemma4::{Gemma4LayerWeights, Gemma4Matrix, Gemma4Weights},
    },
};

pub struct Gemma4Attention<W> {
    query: W,
    key: Option<W>,
    value: Option<W>,
    output: W,
    query_norm: W,
    key_norm: W,
    value_norm: W,
}

pub struct Gemma4Mlp<W> {
    gate: W,
    up: W,
    down: W,
}

pub struct Gemma4PerLayerInput<W> {
    gate: W,
    projection: W,
    post_norm: W,
}

pub struct Gemma4PerLayerModel<W> {
    projection: W,
    projection_norm: W,
}

pub struct Gemma4Layer<W> {
    input_norm: W,
    post_attention_norm: W,
    pre_feedforward_norm: W,
    post_feedforward_norm: W,
    attention: Gemma4Attention<W>,
    mlp: Gemma4Mlp<W>,
    per_layer_input: Option<Gemma4PerLayerInput<W>>,
    layer_scalar: f32,
}

pub struct Gemma4RopeTables {
    sliding: RopeTable,
    full: RopeTable,
}

impl Gemma4RopeTables {
    pub fn new(config: &Gemma4Config, sequence_len: usize) -> Result<Self, BackendError> {
        let sliding = RopeTable::from_spec(sequence_len, RopeSpec::Default { rotary_dim: config.local_head_dim, theta: config.local_rope_theta }).map_err(crate::runtime::compute_error)?;
        let full = RopeTable::from_spec(sequence_len, RopeSpec::Proportional { head_dim: config.global_head_dim, theta: config.global_rope_theta, active_fraction: config.global_rope_fraction }).map_err(crate::runtime::compute_error)?;
        Ok(Self { sliding, full })
    }

    pub fn layer(&self, attention: &HybridGqaLayerSpec) -> &RopeTable {
        match attention.window {
            CausalWindow::Full => &self.full,
            CausalWindow::Sliding { .. } => &self.sliding,
        }
    }
}

/// 文本 embedding 的模型语义只与权重和配置有关，Metal/CUDA 必须共享同一份
/// BF16 边界与缩放规则。
pub fn gemma4_embedding_rows(weights: &Gemma4Weights, tokens: &[u32], hidden_size: usize, scale: f32) -> Result<Vec<f32>, String> {
    let mut values = weights.embedding_rows_f32(tokens)?;
    let expected = tokens.len().checked_mul(hidden_size).ok_or("Gemma4 embedding 大小溢出")?;
    if values.len() != expected {
        return Err(format!("Gemma4 embedding elements={}，期望 {expected}", values.len()));
    }
    values.iter_mut().for_each(|value| *value = bf16::from_f32(bf16::from_f32(*value).to_f32() * scale).to_f32());
    Ok(values)
}

/// E4B per-layer embedding 使用自身列宽的 sqrt scale，不能误用主干 hidden
/// embedding scale。设备上传留给具体 backend 组合模块。
pub fn gemma4_per_layer_embedding_rows(config: &Gemma4Config, weights: &Gemma4Weights, tokens: &[u32]) -> Result<Vec<f32>, String> {
    if config.per_layer_input_size == 0 {
        return Err("Gemma4 当前模型没有 per-layer embedding".to_owned());
    }
    let columns = config.layer_count.checked_mul(config.per_layer_input_size).ok_or("Gemma4 per-layer embedding 列数溢出")?;
    let expected = tokens.len().checked_mul(columns).ok_or("Gemma4 per-layer embedding 大小溢出")?;
    let scale = bf16::from_f32(config.per_layer_embedding_scale().expect("已检查 per_layer_input_size 非零")).to_f32();
    let mut values = weights.per_layer_embedding_rows_f32(tokens)?;
    if values.len() != expected {
        return Err(format!("Gemma4 per-layer embedding elements={}，期望 {expected}", values.len()));
    }
    values.iter_mut().for_each(|value| *value = bf16::from_f32(bf16::from_f32(*value).to_f32() * scale).to_f32());
    Ok(values)
}

pub fn prepare_gemma4_layers<B>(backend: &B, model: &Gemma4, source: &Gemma4Weights) -> Result<Vec<Gemma4Layer<B::Weight>>, BackendError>
where
    B: Backend,
{
    (0..model.layer_count())
        .map(|layer| {
            let weights = source.load_layer(layer).map_err(crate::runtime::compute_error)?;
            prepare_gemma4_layer(backend, model.config(), model.layer_spec(layer).map_err(|error| crate::runtime::compute_error(error.to_string()))?, &weights)
        })
        .collect()
}

pub fn prepare_gemma4_per_layer_model<B>(backend: &B, config: &Gemma4Config, source: &Gemma4Weights) -> Result<Option<Gemma4PerLayerModel<B::Weight>>, BackendError>
where
    B: Backend,
{
    source
        .load_per_layer_model()
        .map_err(crate::runtime::compute_error)?
        .map(|weights| {
            let output_size = config.layer_count * config.per_layer_input_size;
            Ok(Gemma4PerLayerModel { projection: prepare_matrix(backend, &weights.projection, output_size, config.hidden_size)?, projection_norm: prepare_vector(backend, &weights.projection_norm)? })
        })
        .transpose()
}

pub fn gemma4_per_layer_inputs<B>(backend: &B, config: &Gemma4Config, model: Option<&Gemma4PerLayerModel<B::Weight>>, hidden: &B::Tensor, token_inputs: Option<B::Tensor>) -> Result<Option<Vec<B::Tensor>>, BackendError>
where
    B: Backend,
{
    if config.per_layer_input_size == 0 {
        return Ok(None);
    }
    let model = model.ok_or_else(|| crate::runtime::compute_error("Gemma 4 缺少 per-layer model projection"))?;
    let token_inputs = token_inputs.ok_or_else(|| crate::runtime::compute_error("Gemma 4 缺少 token per-layer embedding"))?;
    let expected_columns = config.layer_count * config.per_layer_input_size;
    if backend.token_rows(hidden) != backend.token_rows(&token_inputs) || backend.token_cols(&token_inputs) != expected_columns {
        return Err(crate::runtime::compute_error(format!(
            "Gemma 4 per-layer input shape 异常: hidden=[{},{}], token=[{},{}]",
            backend.token_rows(hidden),
            backend.token_cols(hidden),
            backend.token_rows(&token_inputs),
            backend.token_cols(&token_inputs),
        )));
    }
    let projected = backend.linear(hidden, &model.projection)?;
    let norm_eps = config.rms_eps * config.hidden_size as f32;
    backend.segmented_rmsnorm_add_scaled(&token_inputs, &projected, &model.projection_norm, config.layer_count, config.per_layer_input_size, norm_eps, std::f32::consts::FRAC_1_SQRT_2).map(Some)
}

pub fn prepare_gemma4_layer<B>(backend: &B, config: &Gemma4Config, spec: &Gemma4LayerSpec, weights: &Gemma4LayerWeights) -> Result<Gemma4Layer<B::Weight>, BackendError>
where
    B: Backend,
{
    if !weights.layer_scalar.is_finite() {
        return Err(crate::runtime::compute_error(format!("Gemma 4 layer_scalar={} 非法", weights.layer_scalar)));
    }
    let geometry = spec.attention.hybrid.geometry;
    let (mlp_gate, mlp_up) = prepare_matrix_pair(backend, &weights.mlp.gate_proj, &weights.mlp.up_proj, config.intermediate_size, config.hidden_size)?;
    // add-scaled norm 的 Metal kernel 只接受 F16 权重;prepare 时直接存 F16,
    // 避免每次前向都把 F32 权重现转一遍
    let f16_norm = |values: &[f32]| -> Result<B::Weight, BackendError> {
        let packed: Vec<f16> = values.iter().map(|&value| f16::from_f32(value)).collect();
        backend.prepare_weight(LinearWeight::F16(&packed), 1, values.len())
    };
    Ok(Gemma4Layer {
        // 全部 norm 权重按 F16 准备:plain rmsnorm 的 F16-weight kernel 原生直通,
        // 避免 F32 权重路径每层两次 [rows,2560] 的 f16↔f32 cast
        input_norm: f16_norm(&weights.input_norm)?,
        post_attention_norm: f16_norm(&weights.post_attention_norm)?,
        pre_feedforward_norm: f16_norm(&weights.pre_feedforward_norm)?,
        post_feedforward_norm: f16_norm(&weights.post_feedforward_norm)?,
        attention: Gemma4Attention {
            query: prepare_matrix(backend, &weights.attention.q_proj, geometry.num_heads * geometry.head_dim, config.hidden_size)?,
            key: weights.attention.k_proj.as_ref().map(|weight| prepare_matrix(backend, weight, geometry.num_kv_heads * geometry.head_dim, config.hidden_size)).transpose()?,
            value: weights.attention.v_proj.as_ref().map(|weight| prepare_matrix(backend, weight, geometry.num_kv_heads * geometry.head_dim, config.hidden_size)).transpose()?,
            output: prepare_matrix(backend, &weights.attention.o_proj, config.hidden_size, geometry.num_heads * geometry.head_dim)?,
            query_norm: prepare_zero_centered_vector(backend, &weights.attention.q_norm)?,
            // 共享 KV 层没有 k_norm；前向只走 query 分支，零向量仅作占位
            key_norm: if weights.attention.k_norm.is_empty() { prepare_zero_vector(backend, geometry.head_dim)? } else { prepare_zero_centered_vector(backend, &weights.attention.k_norm)? },
            value_norm: prepare_zero_vector(backend, geometry.head_dim)?,
        },
        mlp: Gemma4Mlp { gate: mlp_gate, up: mlp_up, down: prepare_matrix(backend, &weights.mlp.down_proj, config.hidden_size, config.intermediate_size)? },
        per_layer_input: weights
            .per_layer_input
            .as_ref()
            .map(|weights| {
                Ok(Gemma4PerLayerInput {
                    gate: prepare_matrix(backend, &weights.gate, config.per_layer_input_size, config.hidden_size)?,
                    projection: prepare_matrix(backend, &weights.projection, config.hidden_size, config.per_layer_input_size)?,
                    post_norm: f16_norm(&weights.post_norm)?,
                })
            })
            .transpose()?,
        layer_scalar: weights.layer_scalar,
    })
}

#[allow(clippy::too_many_arguments)]
fn gemma4_prefill_layer<B>(
    backend: &B,
    cache: &mut B::Cache,
    layer: usize,
    config: &Gemma4Config,
    spec: &Gemma4LayerSpec,
    weights: &Gemma4Layer<B::Weight>,
    hidden: &B::Tensor,
    per_layer_input: Option<&B::Tensor>,
    rope: &RopeTable,
    position: usize,
    visible_ends: Option<&[u32]>,
) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    let attention = spec.attention.hybrid;
    let geometry = attention.geometry;
    let attention_output = if let Some(source_layer) = spec.kv_source_layer {
        let normed = backend.rmsnorm(hidden, &weights.input_norm, config.rms_eps)?;
        if weights.attention.key.is_some() || weights.attention.value.is_some() {
            return Err(crate::runtime::compute_error(format!("Gemma 4 shared-KV L{layer} 不应持有 K/V 权重")));
        }
        let query = backend.linear(&normed, &weights.attention.query)?;
        let query = backend.gemma_rmsnorm_heads(&query, &weights.attention.query_norm, geometry.num_heads, geometry.head_dim, config.rms_eps)?;
        let query = backend.rope_prefix(&query, geometry.num_heads, attention.rope.rotary_dim(), crate::attention::rope::RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
        // 图像 soft-token 块在所有层都必须双向可见；官方 non-causal image
        // chunk 同时覆盖 full 与 sliding attention，不能只给滑窗层传 mask。
        match visible_ends {
            Some(ends) => backend.gqa_prefill_attention_cached_from_visible(cache, source_layer, position, &query, &gqa_spec(attention), ends)?,
            None => backend.gqa_prefill_attention_cached_from(cache, source_layer, position, &query, &gqa_spec(attention))?,
        }
    } else {
        let key_weight = weights.attention.key.as_ref().ok_or_else(|| crate::runtime::compute_error(format!("Gemma 4 L{layer} 缺少 K 权重")))?;
        // 融合 QKV+RoPE kernel 单算子数值与顺序路径一致(F16 边界对齐),
        // 但端到端生成质量仍会劣化,根因未明;默认关闭,显式设置
        // ZLLM_GEMMA4_FUSED 才启用。
        let fused = (visible_ends.is_none() && backend.token_rows(hidden) == 1 && std::env::var_os("ZLLM_GEMMA4_FUSED").is_some() && weights.attention.value.is_some()).then(|| {
            let rope_spec = attention.rope.rotary_dim();
            backend.fused_rmsnorm_qkv_head_norm_rope(
                hidden,
                &weights.input_norm,
                &weights.attention.query,
                key_weight,
                weights.attention.value.as_ref().unwrap(),
                &weights.attention.query_norm,
                &weights.attention.key_norm,
                &rope.cos,
                &rope.sin,
                geometry.num_heads,
                geometry.num_kv_heads,
                geometry.head_dim,
                rope_spec,
                position,
                config.rms_eps,
            )
        });
        let (query, key, value) = match fused {
            Some(result) => result?,
            None => {
                let (query, key_source, value_source) = match &weights.attention.value {
                    Some(value) if backend.token_rows(hidden) == 1 => {
                        // rmsnorm + QKV 三路投影共享一个 encoder(省 1 个边界)
                        let (query, key, val) = backend.rmsnorm_triple_linear(hidden, &weights.input_norm, config.rms_eps, &weights.attention.query, key_weight, value)?;
                        (query, key, Some(val))
                    }
                    Some(value) => {
                        let normed = backend.rmsnorm(hidden, &weights.input_norm, config.rms_eps)?;
                        let (query, key, val) = backend.triple_linear(&normed, &weights.attention.query, key_weight, value)?;
                        (query, key, Some(val))
                    }
                    None => {
                        let normed = backend.rmsnorm(hidden, &weights.input_norm, config.rms_eps)?;
                        let (query, key) = backend.dual_linear(&normed, &weights.attention.query, key_weight)?;
                        (query, key, None)
                    }
                };
                // Q/K/V 三路逐头 norm 合并为单 encoder(3 dispatch 省 2 个边界)
                if let (Some(value), 1) = (value_source.as_ref(), backend.token_rows(&query)) {
                    let (query, key, value) = backend.qkv_head_norms(
                        &query,
                        &key_source,
                        value,
                        &weights.attention.query_norm,
                        &weights.attention.key_norm,
                        &weights.attention.value_norm,
                        geometry.num_heads,
                        geometry.num_kv_heads,
                        geometry.head_dim,
                        config.rms_eps,
                    )?;
                    let query = backend.rope_prefix(&query, geometry.num_heads, attention.rope.rotary_dim(), crate::attention::rope::RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
                    let key = backend.rope_prefix(&key, geometry.num_kv_heads, attention.rope.rotary_dim(), crate::attention::rope::RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
                    (query, key, value)
                } else {
                    let query = backend.gemma_rmsnorm_heads(&query, &weights.attention.query_norm, geometry.num_heads, geometry.head_dim, config.rms_eps)?;
                    let key = backend.gemma_rmsnorm_heads(&key_source, &weights.attention.key_norm, geometry.num_kv_heads, geometry.head_dim, config.rms_eps)?;
                    let value_source = value_source.as_ref().unwrap_or(&key_source);
                    let value = backend.gemma_rmsnorm_heads(value_source, &weights.attention.value_norm, geometry.num_kv_heads, geometry.head_dim, config.rms_eps)?;
                    let query = backend.rope_prefix(&query, geometry.num_heads, attention.rope.rotary_dim(), crate::attention::rope::RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
                    let key = backend.rope_prefix(&key, geometry.num_kv_heads, attention.rope.rotary_dim(), crate::attention::rope::RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
                    (query, key, value)
                }
            }
        };
        match visible_ends {
            Some(ends) => backend.gqa_prefill_attention_cached_visible(cache, layer, position, &query, &key, &value, &gqa_spec(attention), ends, spec.retain_full_kv)?,
            None => backend.gqa_prefill_attention_cached(cache, layer, position, &query, &key, &value, &gqa_spec(attention), spec.retain_full_kv)?,
        }
    };
    let attention_output = backend.linear(&attention_output, &weights.attention.output)?;
    let hidden = backend.rmsnorm_add_scaled(hidden, &attention_output, &weights.post_attention_norm, config.rms_eps, 1.0)?;

    // rmsnorm + gate/up gemv 共享一个 encoder(省 1 个边界),down 单独
    let mlp = if backend.token_rows(&hidden) == 1 {
        let activated = backend.rmsnorm_gated_linear(&hidden, &weights.pre_feedforward_norm, config.rms_eps, &weights.mlp.gate, &weights.mlp.up, &Activation::GeluTanh)?;
        backend.linear(&activated, &weights.mlp.down)?
    } else {
        let mlp_input = backend.rmsnorm(&hidden, &weights.pre_feedforward_norm, config.rms_eps)?;
        forward_observed(
            backend,
            &DenseMlpSpec { intermediate_size: config.intermediate_size, activation: Activation::GeluTanh },
            DenseMlpWeightsRef { gate: &weights.mlp.gate, up: &weights.mlp.up, down: &weights.mlp.down },
            &mlp_input,
            |_| {},
        )?
    };
    match (&weights.per_layer_input, per_layer_input) {
        (None, None) => backend.rmsnorm_add_scaled(&hidden, &mlp, &weights.post_feedforward_norm, config.rms_eps, weights.layer_scalar),
        (Some(per_layer_weights), Some(per_layer_input)) => {
            let hidden = backend.rmsnorm_add_scaled(&hidden, &mlp, &weights.post_feedforward_norm, config.rms_eps, 1.0)?;
            let gated = backend.linear_gated_activation(&hidden, &per_layer_weights.gate, per_layer_input, &Activation::GeluTanh)?;
            let projected = backend.linear(&gated, &per_layer_weights.projection)?;
            backend.rmsnorm_add_scaled(&hidden, &projected, &per_layer_weights.post_norm, config.rms_eps, weights.layer_scalar)
        }
        _ => Err(crate::runtime::compute_error("Gemma 4 layer weights 与 per-layer input 不匹配")),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gemma4_prefill_hidden<B>(
    backend: &B,
    cache: &mut B::Cache,
    model: &Gemma4,
    layers: &[Gemma4Layer<B::Weight>],
    rope: &Gemma4RopeTables,
    hidden: B::Tensor,
    per_layer_inputs: Option<&[B::Tensor]>,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    gemma4_prefill_hidden_inner(backend, cache, model, layers, rope, hidden, per_layer_inputs, position, None)
}

#[allow(clippy::too_many_arguments)]
pub fn gemma4_prefill_hidden_with_visibility<B>(
    backend: &B,
    cache: &mut B::Cache,
    model: &Gemma4,
    layers: &[Gemma4Layer<B::Weight>],
    rope: &Gemma4RopeTables,
    hidden: B::Tensor,
    per_layer_inputs: Option<&[B::Tensor]>,
    position: usize,
    visible_ends: &[u32],
) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    gemma4_prefill_hidden_inner(backend, cache, model, layers, rope, hidden, per_layer_inputs, position, Some(visible_ends))
}

#[allow(clippy::too_many_arguments)]
fn gemma4_prefill_hidden_inner<B>(
    backend: &B,
    cache: &mut B::Cache,
    model: &Gemma4,
    layers: &[Gemma4Layer<B::Weight>],
    rope: &Gemma4RopeTables,
    hidden: B::Tensor,
    per_layer_inputs: Option<&[B::Tensor]>,
    position: usize,
    visible_ends: Option<&[u32]>,
) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    let rows = backend.token_rows(&hidden);
    if layers.len() != model.layer_count() || rows == 0 || backend.token_cols(&hidden) != model.config().hidden_size {
        return Err(crate::runtime::compute_error(format!("Gemma 4 prefill 输入不完整: layers={}/{}, hidden=[{},{}]", layers.len(), model.layer_count(), rows, backend.token_cols(&hidden),)));
    }
    validate_per_layer_inputs(model.config(), per_layer_inputs, "prefill")?;
    if visible_ends.is_some_and(|ends| ends.len() != rows) {
        return Err(crate::runtime::compute_error("Gemma 4 visible_ends 与 prefill rows 不一致"));
    }
    let end = position.checked_add(rows).ok_or_else(|| crate::runtime::compute_error("Gemma 4 position 溢出"))?;
    if end > model.config().max_position_embeddings {
        return Err(crate::runtime::compute_error(format!("Gemma 4 prefill end={end} 超过 max_position_embeddings={}", model.config().max_position_embeddings)));
    }

    let result = (|| {
        let mut hidden = hidden;
        for (layer, weights) in layers.iter().enumerate() {
            let spec = model.layer_spec(layer).map_err(|error| crate::runtime::compute_error(error.to_string()))?;
            let _scope = backend.layer_scope();
            backend.begin_batch();
            hidden = gemma4_prefill_layer(backend, cache, layer, model.config(), spec, weights, &hidden, per_layer_inputs.map(|inputs| &inputs[layer]), rope.layer(&spec.attention.hybrid), position, visible_ends)?;
            backend.submit_batch();
        }
        Ok(hidden)
    })();
    backend.finish_batch();
    result
}

#[allow(clippy::too_many_arguments)]
pub fn gemma4_decode_round<B>(
    backend: &B,
    cache: &mut B::Cache,
    model: &Gemma4,
    layers: &[Gemma4Layer<B::Weight>],
    rope: &Gemma4RopeTables,
    hidden: B::Tensor,
    per_layer_inputs: Option<&[B::Tensor]>,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    gemma4_decode_round_impl(backend, cache, model, layers, rope, hidden, per_layer_inputs, position, false)
}

/// decode round 的推迟同步变体:结束批次但不等待 GPU,等待点由 decode 流水线的
/// argmax CB 句柄管理(见 gemma4/node.rs 的 submit_step/wait_token)。
#[allow(clippy::too_many_arguments)]
pub fn gemma4_decode_round_deferred<B>(
    backend: &B,
    cache: &mut B::Cache,
    model: &Gemma4,
    layers: &[Gemma4Layer<B::Weight>],
    rope: &Gemma4RopeTables,
    hidden: B::Tensor,
    per_layer_inputs: Option<&[B::Tensor]>,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    gemma4_decode_round_impl(backend, cache, model, layers, rope, hidden, per_layer_inputs, position, true)
}

#[allow(clippy::too_many_arguments)]
fn gemma4_decode_round_impl<B>(
    backend: &B,
    cache: &mut B::Cache,
    model: &Gemma4,
    layers: &[Gemma4Layer<B::Weight>],
    rope: &Gemma4RopeTables,
    mut hidden: B::Tensor,
    per_layer_inputs: Option<&[B::Tensor]>,
    position: usize,
    deferred: bool,
) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    if layers.len() != model.layer_count() || backend.token_rows(&hidden) != 1 || backend.token_cols(&hidden) != model.config().hidden_size {
        return Err(crate::runtime::compute_error("Gemma 4 decode 输入必须是完整层权重和一个 hidden token"));
    }
    validate_per_layer_inputs(model.config(), per_layer_inputs, "decode")?;
    if position >= model.config().max_position_embeddings {
        return Err(crate::runtime::compute_error(format!("Gemma 4 decode position={position} 超过上下文上限")));
    }
    let result = (|| {
        // 整个 decode round 共用一个 batch：逐层 begin/submit 会把一个 token
        // 拆成数百个 command buffer，flush 开销远超 kernel 本身。
        backend.begin_decode_batch();
        for (layer, weights) in layers.iter().enumerate() {
            let spec = model.layer_spec(layer).map_err(|error| crate::runtime::compute_error(error.to_string()))?;
            let _scope = backend.layer_scope();
            hidden = gemma4_prefill_layer(backend, cache, layer, model.config(), spec, weights, &hidden, per_layer_inputs.map(|inputs| &inputs[layer]), rope.layer(&spec.attention.hybrid), position, None)?;
        }
        backend.submit_batch();
        Ok(hidden)
    })();
    if deferred {
        backend.finish_batch_deferred();
    } else {
        backend.finish_batch();
    }
    result
}

fn validate_per_layer_inputs<T>(config: &Gemma4Config, inputs: Option<&[T]>, phase: &str) -> Result<(), BackendError> {
    let expected = if config.per_layer_input_size == 0 { 0 } else { config.layer_count };
    let actual = inputs.map_or(0, <[T]>::len);
    if actual != expected {
        return Err(crate::runtime::compute_error(format!("Gemma 4 {phase} per-layer inputs={actual}，期望 {expected}")));
    }
    Ok(())
}

pub type Gemma4OutputHead<W> = super::output::OutputHead<W>;

pub fn prepare_gemma4_output_head<B>(backend: &B, config: &Gemma4Config, final_norm: &[f32], tied_embedding: LinearWeight<'_>) -> Result<Gemma4OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    prepare_gemma4_output_head_quantized(backend, config, final_norm, tied_embedding, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_gemma4_output_head_quantized<B>(backend: &B, config: &Gemma4Config, final_norm: &[f32], tied_embedding: LinearWeight<'_>, quantization: crate::weight::LmHeadQuantization) -> Result<Gemma4OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    let zero_centered: Vec<f32> = final_norm.iter().map(|value| value - 1.0).collect();
    super::output::prepare_output_head_gemma_quantized(backend, &zero_centered, tied_embedding, config.vocab_size, config.hidden_size, quantization)
}

pub fn gemma4_last_token_output<B>(backend: &B, config: &Gemma4Config, head: &Gemma4OutputHead<B::Weight>, hidden: &B::Tensor, row: usize) -> Result<super::output::OutputResult<B::Tensor>, BackendError>
where
    B: Backend,
{
    super::output::last_token_output(backend, head, hidden, row, &super::output::OutputPlan { eps: config.rms_eps, norm: super::output::OutputNorm::GemmaRms, excluded_tokens: vec![config.end_image_token_id, config.end_audio_token_id] })
}

pub fn gemma4_token_output<B>(backend: &B, config: &Gemma4Config, head: &Gemma4OutputHead<B::Weight>, hidden: &B::Tensor) -> Result<super::output::OutputResult<B::Tensor>, BackendError>
where
    B: Backend,
{
    super::output::token_output(backend, head, hidden, &super::output::OutputPlan { eps: config.rms_eps, norm: super::output::OutputNorm::GemmaRms, excluded_tokens: vec![config.end_image_token_id, config.end_audio_token_id] })
}

fn gqa_spec(spec: HybridGqaLayerSpec) -> GqaSpec {
    GqaSpec {
        num_heads: spec.geometry.num_heads,
        num_kv_heads: spec.geometry.num_kv_heads,
        head_dim: spec.geometry.head_dim,
        rope_dim: spec.rope.rotary_dim(),
        rope_theta: spec.rope.theta(),
        use_qk_norm: true,
        window: spec.window,
        score_scale: spec.score_scale,
        output_gate: false,
    }
}

fn prepare_vector<B: Backend>(backend: &B, values: &[f32]) -> Result<B::Weight, BackendError> {
    backend.prepare_f32(values, 1, values.len())
}

/// 现有逐头 GemmaRMSNorm capability 使用 `1 + weight`，这里把官方 gamma 转成零中心表示。
fn prepare_zero_centered_vector<B: Backend>(backend: &B, values: &[f32]) -> Result<B::Weight, BackendError> {
    let values = values.iter().map(|&value| value - 1.0).collect::<Vec<_>>();
    backend.prepare_gemma_f32(&values, 1, values.len())
}

fn prepare_zero_vector<B: Backend>(backend: &B, len: usize) -> Result<B::Weight, BackendError> {
    backend.prepare_f32(&vec![0.0; len], 1, len)
}

fn prepare_matrix<B: Backend>(backend: &B, matrix: &Gemma4Matrix, rows: usize, cols: usize) -> Result<B::Weight, BackendError> {
    if matrix.rows() != rows || matrix.cols() != cols {
        return Err(crate::runtime::compute_error(format!("Gemma 4 matrix shape=[{},{}]，期望 [{rows},{cols}]", matrix.rows(), matrix.cols(),)));
    }
    match matrix {
        Gemma4Matrix::Quantized(matrix) => backend.prepare_weight(LinearWeight::Quantized(matrix.as_ref()), rows, cols),
        Gemma4Matrix::Dense(tensor) => prepare_dense_tensor(backend, tensor, rows, cols),
    }
}

fn prepare_matrix_pair<B: Backend>(backend: &B, first: &Gemma4Matrix, second: &Gemma4Matrix, rows: usize, cols: usize) -> Result<(B::Weight, B::Weight), BackendError> {
    if first.rows() != rows || first.cols() != cols || second.rows() != rows || second.cols() != cols {
        return Err(crate::runtime::compute_error(format!("Gemma 4 paired matrix shape first=[{},{}], second=[{},{}]，期望 [{rows},{cols}]", first.rows(), first.cols(), second.rows(), second.cols(),)));
    }
    match (first, second) {
        (Gemma4Matrix::Quantized(first), Gemma4Matrix::Quantized(second)) => backend.prepare_weight_pair(LinearWeight::Quantized(first.as_ref()), LinearWeight::Quantized(second.as_ref()), rows, cols),
        _ => Ok((prepare_matrix(backend, first, rows, cols)?, prepare_matrix(backend, second, rows, cols)?)),
    }
}

fn prepare_dense_tensor<B: Backend>(backend: &B, tensor: &TensorData, rows: usize, cols: usize) -> Result<B::Weight, BackendError> {
    match tensor.dtype.as_str() {
        "BF16" => backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, cols),
        "F16" => {
            let values: Vec<f16> = tensor.data.chunks_exact(2).map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]])).collect();
            backend.prepare_weight(LinearWeight::F16(&values), rows, cols)
        }
        "F32" => {
            let values: Vec<f32> = tensor.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 dense chunk"))).collect();
            backend.prepare_weight(LinearWeight::F32(&values), rows, cols)
        }
        dtype => Err(crate::runtime::compute_error(format!("Gemma 4 dense tensor {} dtype={dtype} 不受支持", tensor.name))),
    }
}

// Gemma 4 unified 多模态模型规格。
//
// Gemma 4 的 local/global attention 在 head_dim、KV heads、RoPE 和窗口上都不同，
// 每层还包含四段 RMSNorm 与 layer scalar。这里保留精确规格，不把差异压进现有
// [`crate::runtime::LayerSpec`]，避免影响 GLM/MiniMax 已优化执行路径。

use crate::attention::gqa::{GqaGeometry, GqaKvProjection, HybridGqaSpec};
use crate::runtime::{LayerId, ModelError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4RmsNormSpec {
    pub eps: f32,
    pub learned_scale: bool,
}

pub use crate::model_spec::gemma4::{Gemma4AudioConfig, Gemma4Config, Gemma4VisionConfig, Gemma4VisionEncoderConfig};

impl Gemma4VisionConfig {
    pub fn model_patch_size(&self) -> Result<usize, ModelError> {
        self.patch_size.checked_mul(self.pooling_kernel_size).ok_or_else(|| ModelError::InvalidArchitecture("Gemma 4 vision model_patch_size 溢出".into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4AttentionSpec {
    pub hybrid: HybridGqaLayerSpec,
    pub q_norm: Gemma4RmsNormSpec,
    pub k_norm: Gemma4RmsNormSpec,
    pub v_norm: Gemma4RmsNormSpec,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4MlpSpec {
    pub intermediate_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4LayerSpec {
    pub attention: Gemma4AttentionSpec,
    pub mlp: Gemma4MlpSpec,
    pub input_norm: Gemma4RmsNormSpec,
    pub post_attention_norm: Gemma4RmsNormSpec,
    pub pre_feedforward_norm: Gemma4RmsNormSpec,
    pub post_feedforward_norm: Gemma4RmsNormSpec,
    pub has_layer_scalar: bool,
    pub kv_source_layer: Option<LayerId>,
    pub retain_full_kv: bool,
}

impl Gemma4Config {
    fn is_full_attention(&self, layer: LayerId) -> bool {
        (layer + 1).is_multiple_of(6) || layer + 1 == self.layer_count
    }

    pub fn kv_source_layer(&self, layer: LayerId) -> Result<Option<LayerId>, ModelError> {
        if layer >= self.layer_count {
            return Err(ModelError::LayerOutOfRange { layer, layer_count: self.layer_count });
        }
        if self.num_kv_shared_layers == 0 || layer < self.layer_count - self.num_kv_shared_layers {
            return Ok(None);
        }
        let shared_start = self.layer_count - self.num_kv_shared_layers;
        let full = self.is_full_attention(layer);
        (0..shared_start).rev().find(|&source| self.is_full_attention(source) == full).map(Some).ok_or_else(|| ModelError::InvalidArchitecture(format!("Gemma 4 L{layer} 找不到同类型 KV 来源层")))
    }

    pub fn standard_12b() -> Self {
        Self {
            vocab_size: 262_144,
            hidden_size: 3_840,
            intermediate_size: 15_360,
            layer_count: 48,
            num_heads: 16,
            local_num_kv_heads: 8,
            local_head_dim: 256,
            global_num_kv_heads: 1,
            global_head_dim: 512,
            sliding_window: 1_024,
            local_rope_theta: 10_000.0,
            global_rope_theta: 1_000_000.0,
            global_rope_fraction: 0.25,
            max_position_embeddings: 262_144,
            rms_eps: 1.0e-6,
            final_logit_softcap: 30.0,
            per_layer_input_size: 0,
            num_kv_shared_layers: 0,
            attention_k_eq_v: true,
            bos_token_id: 2,
            eos_token_ids: vec![1, 106, 50],
            pad_token_id: 0,
            image_token_id: 258_880,
            video_token_id: 258_884,
            audio_token_id: 258_881,
            begin_image_token_id: 255_999,
            end_image_token_id: 258_882,
            begin_audio_token_id: 256_000,
            end_audio_token_id: 258_883,
            vision: Some(Gemma4VisionConfig { patch_size: 16, pooling_kernel_size: 3, embedding_size: 3_840, position_embedding_size: 1_120, default_soft_tokens: 280, encoder: None }),
            audio: Some(Gemma4AudioConfig { sample_rate: 16_000, samples_per_token: 640, embedding_size: 640 }),
        }
    }

    pub fn e4b() -> Self {
        Self {
            vocab_size: 262_144,
            hidden_size: 2_560,
            intermediate_size: 10_240,
            layer_count: 42,
            num_heads: 8,
            local_num_kv_heads: 2,
            local_head_dim: 256,
            global_num_kv_heads: 2,
            global_head_dim: 512,
            sliding_window: 512,
            local_rope_theta: 10_000.0,
            global_rope_theta: 1_000_000.0,
            global_rope_fraction: 0.25,
            max_position_embeddings: 131_072,
            rms_eps: 1.0e-6,
            final_logit_softcap: 30.0,
            per_layer_input_size: 256,
            num_kv_shared_layers: 18,
            attention_k_eq_v: false,
            bos_token_id: 2,
            eos_token_ids: vec![1, 106, 50],
            pad_token_id: 0,
            image_token_id: 258_880,
            video_token_id: 258_884,
            audio_token_id: 258_881,
            begin_image_token_id: 255_999,
            end_image_token_id: 258_882,
            begin_audio_token_id: 256_000,
            end_audio_token_id: 258_883,
            vision: Some(Gemma4VisionConfig {
                patch_size: 16,
                pooling_kernel_size: 3,
                embedding_size: 768,
                position_embedding_size: 10_240,
                default_soft_tokens: 280,
                encoder: Some(Gemma4VisionEncoderConfig { layer_count: 16, num_heads: 12, intermediate_size: 3_072, rope_theta: 100.0, rms_eps: 1.0e-6 }),
            }),
            audio: None,
        }
    }

    pub fn embedding_scale(&self) -> f32 {
        (self.hidden_size as f32).sqrt()
    }

    pub fn per_layer_embedding_scale(&self) -> Option<f32> {
        (self.per_layer_input_size != 0).then(|| (self.per_layer_input_size as f32).sqrt())
    }

    pub fn attention_spec(&self, layer: LayerId) -> Result<Gemma4AttentionSpec, ModelError> {
        if layer >= self.layer_count {
            return Err(ModelError::LayerOutOfRange { layer, layer_count: self.layer_count });
        }
        let norm = |learned_scale| Gemma4RmsNormSpec { eps: self.rms_eps, learned_scale };
        let full = self.is_full_attention(layer);
        Ok(if full {
            Gemma4AttentionSpec {
                hybrid: HybridGqaLayerSpec {
                    geometry: GqaGeometry { num_heads: self.num_heads, num_kv_heads: self.global_num_kv_heads, head_dim: self.global_head_dim },
                    rope: RopeSpec::Proportional { head_dim: self.global_head_dim, theta: self.global_rope_theta, active_fraction: self.global_rope_fraction },
                    window: CausalWindow::Full,
                    score_scale: 1.0,
                    kv_projection: if self.attention_k_eq_v { GqaKvProjection::KeyAsValue } else { GqaKvProjection::Separate },
                },
                q_norm: norm(true),
                k_norm: norm(true),
                v_norm: norm(false),
            }
        } else {
            Gemma4AttentionSpec {
                hybrid: HybridGqaLayerSpec {
                    geometry: GqaGeometry { num_heads: self.num_heads, num_kv_heads: self.local_num_kv_heads, head_dim: self.local_head_dim },
                    rope: RopeSpec::Default { rotary_dim: self.local_head_dim, theta: self.local_rope_theta },
                    window: CausalWindow::Sliding { size: self.sliding_window },
                    score_scale: 1.0,
                    kv_projection: GqaKvProjection::Separate,
                },
                q_norm: norm(true),
                k_norm: norm(true),
                v_norm: norm(false),
            }
        })
    }
}

pub struct Gemma4 {
    config: Gemma4Config,
    layer_specs: Vec<Gemma4LayerSpec>,
    hybrid_gqa: HybridGqaSpec,
}

impl Gemma4 {
    pub fn new(config: Gemma4Config) -> Result<Self, ModelError> {
        if config.layer_count == 0 || config.hidden_size == 0 || config.intermediate_size == 0 {
            return Err(ModelError::InvalidArchitecture("Gemma 4 维度与层数必须非零".into()));
        }
        if config.num_kv_shared_layers > config.layer_count {
            return Err(ModelError::InvalidArchitecture("Gemma 4 KV 共享层数不能超过总层数".into()));
        }
        if config.per_layer_input_size != 0 && !config.hidden_size.is_multiple_of(config.per_layer_input_size) {
            return Err(ModelError::InvalidArchitecture("Gemma 4 per-layer input 维度必须整除 hidden_size".into()));
        }
        if config.num_heads == 0 || config.local_num_kv_heads == 0 || config.global_num_kv_heads == 0 || !config.num_heads.is_multiple_of(config.local_num_kv_heads) || !config.num_heads.is_multiple_of(config.global_num_kv_heads) {
            return Err(ModelError::InvalidArchitecture("Gemma 4 query heads 必须能被 local/global KV heads 整除".into()));
        }
        if config.local_head_dim == 0
            || config.global_head_dim == 0
            || !config.local_head_dim.is_multiple_of(2)
            || !config.global_head_dim.is_multiple_of(2)
            || !config.global_rope_fraction.is_finite()
            || !(0.0..=1.0).contains(&config.global_rope_fraction)
        {
            return Err(ModelError::InvalidArchitecture("Gemma 4 RoPE/head 维度无效".into()));
        }
        if let Some(vision) = config.vision {
            if vision.patch_size == 0 || vision.pooling_kernel_size == 0 || vision.position_embedding_size == 0 || !matches!(vision.default_soft_tokens, 70 | 140 | 280 | 560 | 1120) {
                return Err(ModelError::InvalidArchitecture("Gemma 4 unified vision 规格无效".into()));
            }
            if let Some(encoder) = vision.encoder {
                if encoder.layer_count == 0
                    || encoder.num_heads == 0
                    || !vision.embedding_size.is_multiple_of(encoder.num_heads)
                    || encoder.intermediate_size == 0
                    || !encoder.rope_theta.is_finite()
                    || encoder.rope_theta <= 0.0
                    || !encoder.rms_eps.is_finite()
                    || encoder.rms_eps <= 0.0
                {
                    return Err(ModelError::InvalidArchitecture("Gemma 4 vision encoder 规格无效".into()));
                }
            } else if vision.embedding_size != config.hidden_size {
                return Err(ModelError::InvalidArchitecture("Gemma 4 unified vision embedding 必须等于文本 hidden".into()));
            }
            vision.model_patch_size()?;
        }
        if let Some(audio) = config.audio
            && (audio.sample_rate == 0 || audio.samples_per_token == 0 || audio.embedding_size == 0)
        {
            return Err(ModelError::InvalidArchitecture("Gemma 4 unified audio 规格无效".into()));
        }
        let learned_norm = Gemma4RmsNormSpec { eps: config.rms_eps, learned_scale: true };
        let kv_sources: Vec<Option<LayerId>> = (0..config.layer_count).map(|layer| config.kv_source_layer(layer)).collect::<Result<_, _>>()?;
        let mut layer_specs = Vec::with_capacity(config.layer_count);
        for layer in 0..config.layer_count {
            layer_specs.push(Gemma4LayerSpec {
                attention: config.attention_spec(layer)?,
                mlp: Gemma4MlpSpec { intermediate_size: config.intermediate_size },
                input_norm: learned_norm,
                post_attention_norm: learned_norm,
                pre_feedforward_norm: learned_norm,
                post_feedforward_norm: learned_norm,
                has_layer_scalar: true,
                kv_source_layer: kv_sources[layer],
                retain_full_kv: kv_sources.contains(&Some(layer)),
            });
        }
        let hybrid_gqa = HybridGqaSpec::new(
            layer_specs
                .iter()
                .map(|layer| {
                    let mut cache_spec = layer.attention.hybrid;
                    if layer.retain_full_kv {
                        cache_spec.window = CausalWindow::Full;
                    }
                    cache_spec
                })
                .collect(),
        )
        .map_err(ModelError::InvalidArchitecture)?;
        Ok(Self { config, layer_specs, hybrid_gqa })
    }

    pub fn standard_12b() -> Self {
        Self::new(Gemma4Config::standard_12b()).expect("Gemma 4 12B 标准配置必须有效")
    }

    pub fn e4b() -> Self {
        Self::new(Gemma4Config::e4b()).expect("Gemma 4 E4B 标准配置必须有效")
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    pub fn layer_count(&self) -> usize {
        self.config.layer_count
    }

    pub fn layer_spec(&self, layer: LayerId) -> Result<&Gemma4LayerSpec, ModelError> {
        self.layer_specs.get(layer).ok_or(ModelError::LayerOutOfRange { layer, layer_count: self.config.layer_count })
    }

    pub fn hybrid_gqa(&self) -> &HybridGqaSpec {
        &self.hybrid_gqa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4b_shared_kv_uses_last_matching_layer() {
        let model = Gemma4::e4b();
        let vision = model.config().vision.expect("E4B vision");
        let encoder = vision.encoder.expect("E4B Gemma4V encoder");
        assert_eq!((vision.embedding_size, encoder.layer_count, encoder.num_heads), (768, 16, 12));
        assert_eq!(model.layer_spec(23).unwrap().kv_source_layer, None);
        assert!(model.layer_spec(22).unwrap().retain_full_kv);
        assert!(model.layer_spec(23).unwrap().retain_full_kv);
        assert_eq!(model.layer_spec(24).unwrap().kv_source_layer, Some(22));
        assert_eq!(model.layer_spec(29).unwrap().kv_source_layer, Some(23));
        assert_eq!(model.layer_spec(41).unwrap().kv_source_layer, Some(23));
    }

    #[test]
    fn e4b的per_layer_embedding使用独立scale() {
        let config = Gemma4Config::e4b();
        assert_eq!(config.per_layer_embedding_scale(), Some(16.0));
        assert_ne!(config.per_layer_embedding_scale(), Some(config.embedding_scale()));
        assert_eq!(Gemma4Config::standard_12b().per_layer_embedding_scale(), None);
    }
}
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
pub mod metal_replay;
pub mod node;
