//! Gemma 4 MTP 头在 CUDA 上的 draft 前向。结构对齐 `metal_mtp.rs` 的同步路径
//! (无命令重放):4 层 Q-only decoder,K/V 运行时读主干 KV cache 对应层
//! (vLLM 规则:同 attention 类型的最后一个非 KV-shared 主干层)。
//!
//! 一步 draft:concat(主干 embedding 行, 主干 normed hidden) → pre_projection
//! → 4 层(attn_norm → q → q_norm → rope → attention(读主干 KV) → o →
//! post_norm+res → ffn → post_norm+res → ×layer_output_scale) → output_norm
//! → logits(draft token_embd) + h_next(post_projection 回主干空间)。
//! 与 llama.cpp gemma4-assistant 同语义:draft 链所有步用同一 position。

use crate::{
    attention::{
        gqa::{CausalWindow, GqaSpec},
        rope::{RopeTable, RotaryLayout},
    },
    backend::{
        Backend, BackendError, BackendResources, GqaPrefillBackend,
        cuda::{CudaContext, CudaKvCache, CudaWeight},
    },
    moe::{
        Activation,
        dense_mlp::{DenseMlpSpec, DenseMlpWeightsRef},
    },
    weight::model::gemma4::{Gemma4MtpLayerWeights, Gemma4MtpWeights},
};

fn compute_error(message: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: message.into() }
}

struct PreparedLayer {
    attention_norm: CudaWeight,
    query: CudaWeight,
    query_norm: CudaWeight,
    attention_output: CudaWeight,
    attention_post_norm: CudaWeight,
    ffn_norm: CudaWeight,
    ffn_gate: CudaWeight,
    ffn_up: CudaWeight,
    ffn_down: CudaWeight,
    ffn_post_norm: CudaWeight,
    layer_output_scale: f32,
    /// local/global 与逐层 GQA 参数(与主干同型层对齐后直接读主干 KV)。
    spec: GqaSpec,
    /// 本 draft 层读主干 KV cache 的层。
    backbone_layer: usize,
}

pub struct Gemma4CudaMtp {
    layers: Vec<PreparedLayer>,
    token_embedding: CudaWeight,
    output_norm: CudaWeight,
    pre_projection: CudaWeight,
    post_projection: CudaWeight,
    hidden_size: usize,
    backbone_hidden_size: usize,
    intermediate_size: usize,
    rms_eps: f32,
    /// draft 自有 RoPE 表(与 Metal 同款生成;global 应用 rope_freqs 因子)。
    /// 表长期持有,经 context 的 (地址,长度) 窗口缓存设备驻留。
    rope_local: RopeTable,
    rope_global: RopeTable,
}

/// 全维 RoPE 表(frequency = theta^(-2i/d)),global 版再按 factors 重生成角度。
/// 与 metal_mtp.rs 的同名逻辑一致;CUDA 侧持 host 表,上传走窗口缓存。
fn mtp_rope_table(rotary_dim: usize, theta: f32, seq_len: usize, factors: Option<&[f32]>) -> Result<RopeTable, BackendError> {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0f32; seq_len * half];
    let mut sin = vec![0.0f32; seq_len * half];
    for position in 0..seq_len {
        for index in 0..half {
            let mut frequency = theta.powf(-2.0 * index as f32 / rotary_dim as f32);
            if let Some(factors) = factors {
                if factors.len() != half {
                    return Err(compute_error("MTP RoPE 频率因子维度不匹配"));
                }
                let factor = factors[index];
                if !factor.is_finite() || factor <= 0.0 {
                    return Err(compute_error("MTP RoPE 频率因子非法"));
                }
                frequency /= factor;
            }
            let angle = position as f32 * frequency;
            cos[position * half + index] = angle.cos();
            sin[position * half + index] = angle.sin();
        }
    }
    Ok(RopeTable { cos, sin, seq_len, rotary_dim })
}

fn prepare_layer(ctx: &CudaContext, config: &crate::weight::model::gemma4::Gemma4MtpConfig, layer: &Gemma4MtpLayerWeights, layer_index: usize, backbone_layer: usize) -> Result<PreparedLayer, BackendError> {
    let norm = |values: &[f32]| -> Result<CudaWeight, BackendError> { ctx.prepare_f32(values, 1, values.len()) };
    // q_norm 走 gemma_rmsnorm_heads(+1 语义)而 GGUF 已是 1+γ 编码,预减 1 精确等价
    // 直乘;其余 norm 走普通 rmsnorm 直乘(与 metal_mtp 同款变换)。
    let qnorm = |values: &[f32]| -> Result<CudaWeight, BackendError> { ctx.prepare_f32(&values.iter().map(|value| value - 1.0).collect::<Vec<_>>(), 1, values.len()) };
    let matrix = |matrix: &crate::weight::container::gguf::GgufMatrix| -> Result<CudaWeight, BackendError> { ctx.prepare_weight(crate::backend::LinearWeight::gguf(matrix), matrix.rows, matrix.columns) };
    let global = layer_index + 1 == config.layer_count;
    let (num_kv_heads, head_dim, window) =
        if global { (config.global_num_kv_heads, config.global_head_dim, CausalWindow::Full) } else { (config.local_num_kv_heads, config.local_head_dim, CausalWindow::Sliding { size: config.sliding_window }) };
    let (rope_dim, rope_theta) = if global { (config.global_head_dim, config.global_rope_theta) } else { (config.local_head_dim, config.local_rope_theta) };
    Ok(PreparedLayer {
        attention_norm: norm(&layer.attention_norm)?,
        query: matrix(&layer.query)?,
        query_norm: qnorm(&layer.query_norm)?,
        attention_output: matrix(&layer.attention_output)?,
        attention_post_norm: norm(&layer.attention_post_norm)?,
        ffn_norm: norm(&layer.ffn_norm)?,
        ffn_gate: matrix(&layer.ffn_gate)?,
        ffn_up: matrix(&layer.ffn_up)?,
        ffn_down: matrix(&layer.ffn_down)?,
        ffn_post_norm: norm(&layer.ffn_post_norm)?,
        layer_output_scale: layer.layer_output_scale,
        spec: GqaSpec { num_heads: config.num_heads, num_kv_heads, head_dim, rope_dim, rope_theta, use_qk_norm: true, window, score_scale: 1.0, output_gate: false },
        backbone_layer,
    })
}

impl Gemma4CudaMtp {
    /// 常驻 MTP 权重并生成 RoPE 表。`backbone_layer_types` 是主干逐层 is_full,
    /// `backbone_num_kv_shared` 是主干尾部共享 KV 层数(与 metal_mtp 同款映射规则)。
    pub fn prepare(ctx: &CudaContext, weights: &Gemma4MtpWeights, backbone_layer_types: &[bool], backbone_num_kv_shared: usize, max_positions: usize) -> Result<Self, BackendError> {
        let config = &weights.config;
        let usable = backbone_layer_types.len().saturating_sub(backbone_num_kv_shared);
        let last_full = (0..usable).rev().find(|index| backbone_layer_types[*index]).ok_or_else(|| compute_error("主干没有 full attention 层"))?;
        let last_sliding = (0..usable).rev().find(|index| !backbone_layer_types[*index]).ok_or_else(|| compute_error("主干没有 sliding attention 层"))?;
        let layers = weights
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let global = index + 1 == config.layer_count;
                let backbone_layer = if global { last_full } else { last_sliding };
                prepare_layer(ctx, config, layer, index, backbone_layer)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rope_local = mtp_rope_table(config.local_head_dim, config.local_rope_theta, max_positions, None)?;
        let rope_global = mtp_rope_table(config.global_head_dim, config.global_rope_theta, max_positions, weights.rope_freqs.as_deref())?;
        let matrix = |matrix: &crate::weight::container::gguf::GgufMatrix| -> Result<CudaWeight, BackendError> { ctx.prepare_weight(crate::backend::LinearWeight::gguf(matrix), matrix.rows, matrix.columns) };
        Ok(Self {
            layers,
            token_embedding: matrix(&weights.token_embedding)?,
            output_norm: ctx.prepare_f32(&weights.output_norm, 1, weights.output_norm.len())?,
            pre_projection: matrix(&weights.pre_projection)?,
            post_projection: matrix(&weights.post_projection)?,
            hidden_size: config.hidden_size,
            backbone_hidden_size: config.backbone_hidden_size,
            intermediate_size: config.intermediate_size,
            rms_eps: config.rms_eps,
            rope_local,
            rope_global,
        })
    }

    pub fn backbone_hidden_size(&self) -> usize {
        self.backbone_hidden_size
    }

    pub fn concat_columns(&self) -> usize {
        2 * self.backbone_hidden_size
    }

    /// 一步 draft。`input` 是 CPU 组装的 concat(主干 embedding 行, 主干 normed
    /// hidden);`position` 为本 draft 步在主干序列中的位置(链内共享);返回
    /// (draft token, h_next f32)。
    pub fn draft_step(&self, ctx: &CudaContext, backbone_cache: &CudaKvCache, input: &[f32], position: usize) -> Result<(u32, Vec<f32>), BackendError> {
        if input.len() != self.concat_columns() {
            return Err(compute_error(format!("MTP draft 输入长度 {}，期望 {}", input.len(), self.concat_columns())));
        }
        let input_tensor = ctx.tensor_from_f32(input, 1, self.concat_columns()).map_err(compute_error)?;
        let mut hidden = ctx.linear(&input_tensor, &self.pre_projection)?;
        for layer in &self.layers {
            let normed = ctx.rmsnorm(&hidden, &layer.attention_norm, self.rms_eps)?;
            let query = ctx.linear(&normed, &layer.query)?;
            let query = ctx.gemma_rmsnorm_heads(&query, &layer.query_norm, layer.spec.num_heads, layer.spec.head_dim, self.rms_eps)?;
            let (rope, rotary_dim) = if matches!(layer.spec.window, CausalWindow::Full) { (&self.rope_global, self.rope_global.rotary_dim) } else { (&self.rope_local, self.rope_local.rotary_dim) };
            let query = ctx.rope_prefix(&query, layer.spec.num_heads, rotary_dim, RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
            let attention = crate::backend::GqaPrefillBackend::gqa_prefill_attention_cached_from(ctx, backbone_cache, layer.backbone_layer, position, &query, &layer.spec)?;
            let attention = ctx.linear(&attention, &layer.attention_output)?;
            let attention_normed = ctx.rmsnorm(&attention, &layer.attention_post_norm, self.rms_eps)?;
            let residual = ctx.add(&attention_normed, &hidden)?;
            let ffn_input = ctx.rmsnorm(&residual, &layer.ffn_norm, self.rms_eps)?;
            let ffn = crate::moe::dense_mlp::forward_observed(
                ctx,
                &DenseMlpSpec { intermediate_size: self.intermediate_size, activation: Activation::GeluTanh },
                DenseMlpWeightsRef { gate: &layer.ffn_gate, up: &layer.ffn_up, down: &layer.ffn_down },
                &ffn_input,
                |_| {},
            )?;
            let ffn_normed = ctx.rmsnorm(&ffn, &layer.ffn_post_norm, self.rms_eps)?;
            let added = ctx.add(&ffn_normed, &residual)?;
            // ×layer_output_scale = zeros + x*scale(add_scaled 复用)。
            let zeros = ctx.tensor_zeros(added.rows, added.cols).map_err(compute_error)?;
            hidden = ctx.add_scaled(&zeros, &added, layer.layer_output_scale)?;
        }
        let normed = ctx.rmsnorm(&hidden, &self.output_norm, self.rms_eps)?;
        let logits = ctx.linear(&normed, &self.token_embedding)?;
        let token = ctx.argmax(&logits)?;
        let h_next = ctx.linear(&normed, &self.post_projection)?;
        let h_next = ctx.tensor_to_f32(&h_next).map_err(compute_error)?;
        Ok((token, h_next))
    }
}
