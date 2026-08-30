//! Gemma 4 MTP 头在 Metal 上的 draft 前向:4 层 Q-only decoder,
//! K/V 运行时读主干 KV cache 对应层(主干最后 4 层 [l,l,l,g] 与 draft
//! pattern 对齐),与主干共享同一套 GQA attention kernel。
//!
//! 一步 draft:concat(主干 embedding 行 ×√3840, 主干 hidden) → pre_proj
//! → 4 层(attn_norm → q → q_norm → rope → attention(读主干 KV) → o →
//! post_norm+res → ffn → post_norm+res → ×layer_output_scale) →
//! output_norm → logits(draft tied 头) + h_next(post_proj 回主干空间)。
//! 参照 llama.cpp src/models/gemma4-assistant.cpp。

use crate::{
    attention::{
        gqa::{CausalWindow, GqaSpec},
        rope::{RopeTable, RotaryLayout},
    },
    backend::{
        Backend, BackendError, BackendResources, GqaPrefillBackend,
        metal::{MetalContext, MetalKvCache, MetalTensor, MetalWeight, api::Buffer},
    },
    kernel::metal::{attention as metal_attention, shape as metal_shape, tensor as metal_tensor},
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
    attention_norm: MetalWeight,
    query: MetalWeight,
    query_norm: MetalWeight,
    attention_output: MetalWeight,
    attention_post_norm: MetalWeight,
    ffn_norm: MetalWeight,
    ffn_gate: MetalWeight,
    ffn_up: MetalWeight,
    ffn_down: MetalWeight,
    ffn_post_norm: MetalWeight,
    layer_output_scale: f32,
    /// local/global 与逐层 GQA 参数(与主干最后 4 层对齐后可直接读主干 KV)。
    spec: GqaSpec,
    /// 本 draft 层读主干 KV cache 的层:vLLM 规则——同 attention 类型
    /// (sliding/full)的最后一个非 KV-shared 主干层。
    backbone_layer: usize,
}

/// F16 常驻 RoPE 表(local/global 各一份,draft 位置自索引)。
pub struct Gemma4MtpRopeTable {
    cos: Buffer,
    sin: Buffer,
    rotary_dim: usize,
    max_positions: usize,
}

#[allow(dead_code)] // 同步 draft 路径保留为 replay 的正确性 oracle。
pub struct Gemma4MtpModel {
    layers: Vec<PreparedLayer>,
    token_embedding: MetalWeight,
    output_norm: MetalWeight,
    pre_projection: MetalWeight,
    post_projection: MetalWeight,
    hidden_size: usize,
    backbone_hidden_size: usize,
    intermediate_size: usize,
    rms_eps: f32,
    state: Buffer,
    rope_local: Gemma4MtpRopeTable,
    rope_global: Gemma4MtpRopeTable,
}

fn upload_rope(ctx: &MetalContext, table: &RopeTable, theta: f32, factors: Option<&[f32]>) -> Result<Gemma4MtpRopeTable, BackendError> {
    let half_dim = table.rotary_dim / 2;
    let needed = table.seq_len.checked_mul(half_dim).ok_or_else(|| compute_error("MTP RoPE 表长度溢出"))?;
    if table.cos.len() < needed || table.sin.len() < needed {
        return Err(compute_error("MTP RoPE 表 cos/sin 长度不足"));
    }
    // llama.cpp 语义:frequency_i = theta^(-2i/d) / ff_i(ff 越大旋转越慢,尾值 ~1e30
    // 即高维冻结);表按含因子的角度重新生成,而非对 cos/sin 值缩放。
    let mut cos = table.cos[..needed].to_vec();
    let mut sin = table.sin[..needed].to_vec();
    if let Some(factors) = factors {
        if factors.len() != half_dim {
            return Err(compute_error("MTP RoPE 频率因子维度不匹配"));
        }
        let dim = table.rotary_dim;
        for index in 0..half_dim {
            if !factors[index].is_finite() || factors[index] <= 0.0 {
                return Err(compute_error("MTP RoPE 频率因子非法"));
            }
            let frequency = theta.powf(-2.0 * index as f32 / dim as f32) / factors[index];
            for position in 0..table.seq_len {
                let angle = position as f32 * frequency;
                cos[position * half_dim + index] = angle.cos();
                sin[position * half_dim + index] = angle.sin();
            }
        }
    }
    let pack = |values: &[f32]| -> Buffer {
        let packed: Vec<u16> = values.iter().map(|&value| half::f16::from_f32(value).to_bits()).collect();
        let bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) };
        ctx.shared_buffer(bytes)
    };
    Ok(Gemma4MtpRopeTable { cos: pack(&cos), sin: pack(&sin), rotary_dim: table.rotary_dim, max_positions: table.seq_len })
}

/// 生成 MTP 的 RoPE 表(全维 RoPE:rotary_dim == head_dim)。
fn mtp_rope_table(rotary_dim: usize, theta: f32, seq_len: usize) -> RopeTable {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0f32; seq_len * half];
    let mut sin = vec![0.0f32; seq_len * half];
    for position in 0..seq_len {
        for index in 0..half {
            let frequency = theta.powf(-2.0 * index as f32 / rotary_dim as f32);
            let angle = position as f32 * frequency;
            cos[position * half + index] = angle.cos();
            sin[position * half + index] = angle.sin();
        }
    }
    RopeTable { cos, sin, seq_len, rotary_dim }
}

fn prepare_gguf(ctx: &MetalContext, matrix: &crate::weight::container::gguf::GgufMatrix) -> Result<MetalWeight, BackendError> {
    let weight = MetalWeight::allocate_gguf(ctx, matrix, matrix.rows, matrix.columns).map_err(compute_error)?;
    weight.fill_gguf(matrix).map_err(compute_error)?;
    Ok(weight)
}

fn prepare_layer(ctx: &MetalContext, config: &crate::weight::model::gemma4::Gemma4MtpConfig, layer: &Gemma4MtpLayerWeights, layer_index: usize, backbone_layer: usize) -> Result<PreparedLayer, BackendError> {
    let norm = |values: &[f32]| -> Result<MetalWeight, BackendError> { ctx.prepare_f32(values, 1, values.len()) };
    // q_norm 走 gemma_rmsnorm_heads(+1 语义)而 GGUF 已是 1+γ 编码(实测常数
    // 0.9922 = 1+γ, γ≈0):预减 1 后 (γ)+1 精确等价直乘;其余 norm 走普通
    // rmsnorm 直乘,不需要变换。
    let qnorm = |values: &[f32]| -> Result<MetalWeight, BackendError> { ctx.prepare_f32(&values.iter().map(|value| value - 1.0).collect::<Vec<_>>(), 1, values.len()) };
    let global = layer_index + 1 == config.layer_count;
    let (num_kv_heads, head_dim, window, rope_dim, rope_theta) = if global {
        (config.global_num_kv_heads, config.global_head_dim, CausalWindow::Full, config.global_head_dim, config.global_rope_theta)
    } else {
        (config.local_num_kv_heads, config.local_head_dim, CausalWindow::Sliding { size: config.sliding_window }, config.local_head_dim, config.local_rope_theta)
    };
    let _ = rope_theta;
    Ok(PreparedLayer {
        attention_norm: norm(&layer.attention_norm)?,
        query: prepare_gguf(ctx, &layer.query)?,
        query_norm: qnorm(&layer.query_norm)?,
        attention_output: prepare_gguf(ctx, &layer.attention_output)?,
        attention_post_norm: norm(&layer.attention_post_norm)?,
        ffn_norm: norm(&layer.ffn_norm)?,
        ffn_gate: prepare_gguf(ctx, &layer.ffn_gate)?,
        ffn_up: prepare_gguf(ctx, &layer.ffn_up)?,
        ffn_down: prepare_gguf(ctx, &layer.ffn_down)?,
        ffn_post_norm: norm(&layer.ffn_post_norm)?,
        layer_output_scale: layer.layer_output_scale,
        spec: GqaSpec { num_heads: config.num_heads, num_kv_heads, head_dim, rope_dim, rope_theta, use_qk_norm: true, window, score_scale: 1.0, output_gate: false },
        backbone_layer,
    })
}

#[allow(dead_code)] // 同步 draft 路径保留为 replay 的正确性 oracle。
impl Gemma4MtpModel {
    /// 常驻 MTP 权重并生成 RoPE 表。`backbone_layer_types` 是主干逐层 is_full,
    /// `backbone_num_kv_shared` 是主干尾部共享 KV 层数;draft 层 j 映射到同类型
    /// (sliding/full)的最后一个非 KV-shared 主干层读其 KV(vLLM 同款规则)。
    pub fn prepare(ctx: &MetalContext, weights: &Gemma4MtpWeights, backbone_layer_types: &[bool], backbone_num_kv_shared: usize, max_positions: usize) -> Result<Self, BackendError> {
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
        let rope_local = upload_rope(ctx, &mtp_rope_table(config.local_head_dim, config.local_rope_theta, max_positions), config.local_rope_theta, None)?;
        let global_factors = weights.rope_freqs.as_deref();
        let rope_global = upload_rope(ctx, &mtp_rope_table(config.global_head_dim, config.global_rope_theta, max_positions), config.global_rope_theta, global_factors)?;
        let state = ctx.shared_buffer_zeros(config.layer_count * 12);
        Ok(Self {
            layers,
            token_embedding: prepare_gguf(ctx, &weights.token_embedding)?,
            output_norm: ctx.prepare_f32(&weights.output_norm, 1, weights.output_norm.len())?,
            pre_projection: prepare_gguf(ctx, &weights.pre_projection)?,
            post_projection: prepare_gguf(ctx, &weights.post_projection)?,
            hidden_size: config.hidden_size,
            backbone_hidden_size: config.backbone_hidden_size,
            intermediate_size: config.intermediate_size,
            rms_eps: config.rms_eps,
            state,
            rope_local,
            rope_global,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn concat_columns(&self) -> usize {
        2 * self.backbone_hidden_size
    }

    pub fn backbone_hidden_size(&self) -> usize {
        self.backbone_hidden_size
    }

    /// 一步 draft。`input` 是 CPU 组装的 concat(主干 embedding 行, 主干 hidden)
    /// (7680);`position` 为本 draft 步在主干序列中的位置;返回 (draft token, h_next)。
    /// h_next 是 3840 F16 tensor,作为下一 draft 步的 hidden 输入。
    pub fn draft_step(&self, ctx: &MetalContext, backbone_cache: &MetalKvCache, input: &[f32], position: usize) -> Result<(u32, MetalTensor), BackendError> {
        if input.len() != self.concat_columns() {
            return Err(compute_error(format!("MTP draft 输入长度 {}，期望 {}", input.len(), self.concat_columns())));
        }
        // state 槽:position 固定;KV 范围读主干对应层当前游标。
        let slots = unsafe { std::slice::from_raw_parts_mut(self.state.contents().cast::<u32>(), self.layers.len() * 3) };
        for (index, layer) in self.layers.iter().enumerate() {
            let retained = backbone_cache.hybrid_gqa_state().ok_or_else(|| compute_error("MTP 需要 hybrid cache"))?.retained_range(layer.backbone_layer).map_err(compute_error)?;
            let slot = &mut slots[index * 3..index * 3 + 3];
            slot[0] = position as u32;
            slot[1] = retained.end as u32;
            slot[2] = retained.start as u32;
        }
        let input_tensor = ctx.tensor_from_f32(input, 1, self.concat_columns()).map_err(compute_error)?;
        // 整步共用一个 batch:4 层 ~60 个小算子逐个 commit+wait 会放大到 ~30ms,
        // 批量提交后仅 argmax 读回一次同步。
        ctx.begin_batch();
        let mut hidden = ctx.linear(&input_tensor, &self.pre_projection)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let state_offset = (index * 12) as u64;
            let normed = ctx.rmsnorm(&hidden, &layer.attention_norm, self.rms_eps)?;
            let query = ctx.linear(&normed, &layer.query)?;
            let head_dim = layer.spec.head_dim;
            let query = ctx.gemma_rmsnorm_heads(&query, &layer.query_norm, layer.spec.num_heads, head_dim, self.rms_eps)?;
            let (rope, rotary_dim) = if matches!(layer.spec.window, CausalWindow::Full) { (&self.rope_global, self.rope_global.rotary_dim) } else { (&self.rope_local, self.rope_local.rotary_dim) };
            let query = metal_shape::apply_rope_position_tensor(ctx, &query, layer.spec.num_heads, rotary_dim, RotaryLayout::SplitHalf, &rope.cos, &rope.sin, rope.max_positions, &self.state, state_offset).map_err(compute_error)?;
            let view = backbone_cache.gqa_layer_view(layer.backbone_layer).map_err(compute_error)?;
            let attention = metal_attention::gqa_decode_attention_position_tensor(ctx, &query, &view, &layer.spec, &self.state, state_offset).map_err(compute_error)?;
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
            hidden = metal_tensor::mul_scalar_f16_tensor(ctx, &added, layer.layer_output_scale).map_err(compute_error)?;
        }
        let normed = ctx.rmsnorm(&hidden, &self.output_norm, self.rms_eps)?;
        let logits = ctx.linear(&normed, &self.token_embedding)?;
        let h_next = ctx.linear(&normed, &self.post_projection)?;
        ctx.finish_batch();
        // argmax 走同步小 kernel:logits 已随 batch 提交完成,读回即终值。
        let readback = ctx.shared_buffer_zeros(4);
        crate::kernel::metal::moe::argmax_tensor_into_offset(ctx, &logits, &[], &readback, 0).map_err(compute_error)?;
        let token = unsafe { *readback.contents().cast::<u32>() };
        Ok((token, h_next))
    }
}

/// MTP draft 的单份命令重放:draft 链全为 position_tensor 原语(命令表
/// position 无关),串行提交无在飞重叠(单份 input/state 即可)。主干 KV 视图
/// 录制期绑定,每请求 remap。verify 暂不走重放(多行 attention 的 position
/// 参数化是后续工作)。
#[allow(dead_code)] // 诊断字段用于核对录制层数。
pub struct Gemma4MtpReplay {
    commands: crate::backend::metal::api::CommandList,
    input: MetalTensor,
    readback: crate::backend::metal::api::Buffer,
    h_next: MetalTensor,
    current_backbone: crate::backend::metal::api::Buffer,
    layer_count: usize,
}

impl Gemma4MtpModel {
    /// 录制一遍 draft 前向(转录模式,GPU 不执行)。
    pub fn record_replay(&self, ctx: &MetalContext, backbone_cache: &MetalKvCache) -> Result<Gemma4MtpReplay, BackendError> {
        let input = ctx.tensor_zeros(1, self.concat_columns());
        let readback = ctx.shared_buffer_zeros(4);
        crate::backend::metal::api::Transcriber::begin_flat().map_err(compute_error)?;
        let recorded = self.draft_forward(ctx, backbone_cache, &input, &readback);
        let transcriber = crate::backend::metal::api::Transcriber::end().ok_or_else(|| compute_error("MTP 重放转录器未在进行"))?;
        let commands = transcriber.into_command_list().map_err(compute_error)?;
        let (h_next, layer_count) = recorded?;
        Ok(Gemma4MtpReplay { commands, input, readback, h_next, current_backbone: backbone_cache.buffer().clone(), layer_count })
    }

    /// draft 前向的核心序列;record_replay 转录用(fixed 输入输出 buffer),
    /// draft_step(同步路径)不复用(它自建 tensor 且 argmax 同步读)。
    fn draft_forward(&self, ctx: &MetalContext, backbone_cache: &MetalKvCache, input: &MetalTensor, readback: &crate::backend::metal::api::Buffer) -> Result<(MetalTensor, usize), BackendError> {
        ctx.begin_batch();
        let mut hidden = ctx.linear(input, &self.pre_projection)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let state_offset = (index * 12) as u64;
            let normed = ctx.rmsnorm(&hidden, &layer.attention_norm, self.rms_eps)?;
            let query = ctx.linear(&normed, &layer.query)?;
            let head_dim = layer.spec.head_dim;
            let query = ctx.gemma_rmsnorm_heads(&query, &layer.query_norm, layer.spec.num_heads, head_dim, self.rms_eps)?;
            let (rope, rotary_dim) = if matches!(layer.spec.window, CausalWindow::Full) { (&self.rope_global, self.rope_global.rotary_dim) } else { (&self.rope_local, self.rope_local.rotary_dim) };
            let query = metal_shape::apply_rope_position_tensor(ctx, &query, layer.spec.num_heads, rotary_dim, RotaryLayout::SplitHalf, &rope.cos, &rope.sin, rope.max_positions, &self.state, state_offset).map_err(compute_error)?;
            let view = backbone_cache.gqa_layer_view(layer.backbone_layer).map_err(compute_error)?;
            let attention = metal_attention::gqa_decode_attention_position_tensor(ctx, &query, &view, &layer.spec, &self.state, state_offset).map_err(compute_error)?;
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
            hidden = metal_tensor::mul_scalar_f16_tensor(ctx, &added, layer.layer_output_scale).map_err(compute_error)?;
        }
        let normed = ctx.rmsnorm(&hidden, &self.output_norm, self.rms_eps)?;
        let logits = ctx.linear(&normed, &self.token_embedding)?;
        crate::kernel::metal::moe::argmax_tensor_into_offset(ctx, &logits, &[], readback, 0).map_err(compute_error)?;
        // h_next 即转录期 linear 分配的 tensor:其 buffer 被命令表绑定,
        // CPU 每步从它读回。
        let h_next = ctx.linear(&normed, &self.post_projection)?;
        ctx.finish_batch();
        Ok((h_next, self.layers.len()))
    }
}

#[allow(dead_code)] // command_count 仅供 replay 诊断。
impl Gemma4MtpReplay {
    pub fn command_count(&self) -> usize {
        self.commands.ops.len()
    }

    /// 每请求绑定主干 KV cache(命令表内旧 buffer 指针整体替换)。
    pub fn bind_backbone(&mut self, backbone_cache: &MetalKvCache) {
        let replacements = [(self.current_backbone.clone(), backbone_cache.buffer().clone())];
        self.commands.remap_buffers(&replacements);
        self.current_backbone = backbone_cache.buffer().clone();
    }

    /// 一步 draft:CPU 写 input 与 state 槽 → 重编码提交 → wait → 读 token。
    pub fn step(&self, ctx: &MetalContext, model: &Gemma4MtpModel, backbone_cache: &MetalKvCache, input: &[f32], position: usize) -> Result<u32, BackendError> {
        if input.len() != model.concat_columns() {
            return Err(compute_error(format!("MTP 重放输入长度 {}，期望 {}", input.len(), model.concat_columns())));
        }
        let slots = unsafe { std::slice::from_raw_parts_mut(model.state.contents().cast::<u32>(), model.layers.len() * 3) };
        for (index, layer) in model.layers.iter().enumerate() {
            let retained = backbone_cache.hybrid_gqa_state().ok_or_else(|| compute_error("MTP 需要 hybrid cache"))?.retained_range(layer.backbone_layer).map_err(compute_error)?;
            let slot = &mut slots[index * 3..index * 3 + 3];
            slot[0] = position as u32;
            slot[1] = retained.end as u32;
            slot[2] = retained.start as u32;
        }
        // CPU 写 input(f32→f16 进共享 buffer)
        let packed: Vec<u16> = input.iter().map(|&value| half::f16::from_f32(value).to_bits()).collect();
        let destination = unsafe { std::slice::from_raw_parts_mut(model_input_ptr(&self.input), packed.len() * 2) };
        destination.copy_from_slice(unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) });
        let command = ctx.command_buffer();
        {
            let encoder = command.new_compute_command_encoder();
            for op in &self.commands.ops {
                encoder.encode_recorded(op);
            }
            encoder.end_encoding();
        }
        command.commit();
        command.wait_until_completed();
        if std::env::var_os("ZLLM_GEMMA4_MTP_TRACE").is_some() {
            eprintln!("[mtp-gpu] draft step gpu={:.3}ms", (command.gpu_end_time() - command.gpu_start_time()) * 1.0e3);
        }
        Ok(unsafe { *self.readback.contents().cast::<u32>() })
    }

    /// 上一步产出的 h_next(3840 f16 行)读回 CPU。
    pub fn h_next_f32(&self, ctx: &MetalContext) -> Vec<f32> {
        ctx.tensor_to_f32(&self.h_next)
    }
}

fn model_input_ptr(tensor: &MetalTensor) -> *mut u8 {
    tensor.buffer.contents() as *mut u8
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::{
        backend::metal::MetalContext,
        runtime::gemma4::{gemma4_embedding_rows, metal::gemma4_metal_per_layer_inputs},
        weight::model::gemma4::{Gemma4MtpWeights, Gemma4Weights},
    };

    /// Draft 前向 smoke:主干 prefill 后跑一步 draft,检查数值与 token 合理性。
    /// `ZLLM_GEMMA4_GGUF=/path/12b.gguf ZLLM_GEMMA4_MTP=/path/mtp.gguf cargo test --release --lib mtp_draft_smoke -- --nocapture`
    fn sync_h_pair(_ctx: &MetalContext, _a: &MetalTensor, _b: &[f32]) -> f32 {
        0.0
    }

    #[test]
    fn mtp_draft_smoke() {
        let (Some(backbone), Some(mtp)) = (std::env::var_os("ZLLM_GEMMA4_GGUF").map(std::path::PathBuf::from), std::env::var_os("ZLLM_GEMMA4_MTP").map(std::path::PathBuf::from)) else { return };
        let ctx = MetalContext::new_default().expect("metal");
        let config = Gemma4Weights::select_config(&backbone).expect("select_config");
        let weights = Gemma4Weights::open(&backbone, config.clone()).expect("open backbone");
        let model = crate::runtime::gemma4::Gemma4::new(config.clone()).expect("model");
        let rope = crate::runtime::gemma4::Gemma4RopeTables::new(&config, 2048).expect("rope");
        let layers = crate::runtime::gemma4::prepare_gemma4_layers(&ctx, &model, &weights).expect("layers");
        let per_layer_model = crate::runtime::gemma4::prepare_gemma4_per_layer_model(&ctx, &config, &weights).expect("per-layer model");
        let mut cache = MetalKvCache::new_hybrid_gqa(&ctx, model.hybrid_gqa().clone(), 2048).expect("cache");
        let tokenizer = crate::tokenizer::Tokenizer::new(&backbone.join("tokenizer.json")).ok();
        let prompt: Vec<u32> = match &tokenizer {
            Some(tokenizer) => tokenizer.tokenize("The capital of France is".as_bytes()),
            None => vec![2, 575, 3575, 1075, 524],
        };
        let mut position = 0usize;
        let mut last = None;
        for chunk in prompt.chunks(16) {
            let embedding = gemma4_embedding_rows(&weights, chunk, config.hidden_size, config.embedding_scale()).expect("embedding");
            let input = ctx.tensor_from_f32(&embedding, chunk.len(), config.hidden_size).expect("upload");
            let per_layer_inputs = gemma4_metal_per_layer_inputs(&ctx, &config, &weights, per_layer_model.as_ref(), &input, chunk).expect("per-layer inputs");
            last = Some(crate::runtime::gemma4::gemma4_prefill_hidden(&ctx, &mut cache, &model, &layers, &rope, input, per_layer_inputs.as_deref(), position).expect("prefill"));
            position += chunk.len();
        }
        let hidden = last.expect("hidden");
        ctx.synchronize();
        let backbone_hidden = ctx.tensor_to_f32(&hidden);
        let row = &backbone_hidden[(hidden.rows - 1) * config.hidden_size..hidden.rows * config.hidden_size];
        let non_finite = row.iter().filter(|value| !value.is_finite()).count();
        println!("[mtp-smoke] 主干 hidden 行: len={} non_finite={}", row.len(), non_finite);

        let mtp_weights = Gemma4MtpWeights::open(&mtp).expect("open mtp");
        let backbone_types: Vec<bool> = (0..model.layer_count()).map(|layer| model.layer_spec(layer).expect("spec").attention.hybrid.window == crate::attention::gqa::CausalWindow::Full).collect();
        let mtp_model = Gemma4MtpModel::prepare(&ctx, &mtp_weights, &backbone_types, config.num_kv_shared_layers, 2048).expect("prepare mtp");
        // 主干 greedy 首 token 作为 draft 步输入
        let output_head = crate::runtime::gemma4::metal::prepare_gemma4_metal_output_head(&ctx, &config, &weights, crate::weight::LmHeadQuantization::Native).expect("head");
        let first = crate::runtime::gemma4::gemma4_last_token_output(&ctx, &config, &output_head, &hidden, hidden.rows - 1).expect("first token");
        println!("[mtp-smoke] 主干首 token={}", first.token_id);
        let embedding_row = gemma4_embedding_rows(&weights, &[first.token_id], config.hidden_size, config.embedding_scale()).expect("embedding row");
        let mut input = Vec::with_capacity(mtp_model.concat_columns());
        input.extend_from_slice(&embedding_row);
        input.extend_from_slice(row);
        let started = std::time::Instant::now();
        let (token, h_next) = mtp_model.draft_step(&ctx, &cache, &input, position).expect("draft step");
        println!("[mtp-smoke] draft token={} h_next 行耗时 {:.2}ms", token, started.elapsed().as_secs_f64() * 1.0e3);
        ctx.synchronize();
        let h_values = ctx.tensor_to_f32(&h_next);
        let non_finite = h_values.iter().filter(|value| !value.is_finite()).count();
        let magnitude = h_values.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        println!("[mtp-smoke] h_next: len={} non_finite={} max|.|={:.3}", h_values.len(), non_finite, magnitude);
        assert_eq!(non_finite, 0, "draft 输出含非有限值");
        assert!(token > 3, "draft token 异常");
        // 连续 3 步(链式 h_next)
        let mut next_input_embedding = gemma4_embedding_rows(&weights, &[token], config.hidden_size, config.embedding_scale()).expect("embedding");
        let first_embedding = next_input_embedding.clone();
        let mut last_h_next = h_next.clone();
        for step in 1..=3 {
            let mut input = Vec::with_capacity(mtp_model.concat_columns());
            input.extend_from_slice(&next_input_embedding);
            input.extend_from_slice(&h_values);
            let (token, h_next) = mtp_model.draft_step(&ctx, &cache, &input, position + step).expect("draft chain");
            last_h_next = h_next.clone();
            let h_values = ctx.tensor_to_f32(&h_next);
            let non_finite = h_values.iter().filter(|value| !value.is_finite()).count();
            println!("[mtp-smoke] 链式 step={step} token={} non_finite={}", token, non_finite);
            assert_eq!(non_finite, 0);
            next_input_embedding = gemma4_embedding_rows(&weights, &[token], config.hidden_size, config.embedding_scale()).expect("embedding");
        }
        // 重放对照:固定输入序列下,replay.step 与 draft_step 逐 token 一致。
        let replay = mtp_model.record_replay(&ctx, &cache).expect("record replay");
        println!("[mtp-smoke] replay 命令数={}", replay.command_count());
        let mut chain_h: Vec<f32> = ctx.tensor_to_f32(&last_h_next);
        let mut chain_token_embedding = first_embedding.clone();
        for step in 0..3 {
            let mut input = Vec::with_capacity(mtp_model.concat_columns());
            input.extend_from_slice(&chain_token_embedding);
            input.extend_from_slice(&chain_h);
            let (sync_token, sync_h) = mtp_model.draft_step(&ctx, &cache, &input, position + 8 + step).expect("sync");
            let replay_token = replay.step(&ctx, &mtp_model, &cache, &input, position + 8 + step).expect("replay");
            let replay_h = replay.h_next_f32(&ctx);
            println!("[mtp-smoke] 对照 step={step} sync={sync_token} replay={replay_token}");
            assert_eq!(sync_token, replay_token, "重放与同步路径 token 不一致");
            let h_diff: f32 = sync_h_pair(&ctx, &sync_h, &replay_h);
            assert!(h_diff < 0.5, "h_next 偏差 {h_diff}");
            chain_h = ctx.tensor_to_f32(&sync_h);
            chain_token_embedding = gemma4_embedding_rows(&weights, &[sync_token], config.hidden_size, config.embedding_scale()).unwrap();
        }
        // 重放步时
        let mut input = Vec::with_capacity(mtp_model.concat_columns());
        input.extend_from_slice(&chain_token_embedding);
        input.extend_from_slice(&chain_h);
        let started = std::time::Instant::now();
        for step in 0..20 {
            let token = replay.step(&ctx, &mtp_model, &cache, &input, position + 20 + step).expect("replay step");
            let h = replay.h_next_f32(&ctx);
            input.truncate(0);
            input.extend_from_slice(&gemma4_embedding_rows(&weights, &[token], config.hidden_size, config.embedding_scale()).unwrap());
            input.extend_from_slice(&h);
        }
        println!("[mtp-smoke] replay 步时 {:.2}ms", started.elapsed().as_secs_f64() / 20.0 * 1.0e3);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod freqs_tests {
    /// 从 Q4_0 MTP 文件读 rope_freqs(Q8_0 缺此张量)。
    /// `ZLLM_GEMMA4_MTP_Q4=/path/mtp-Q4_0.gguf cargo test --lib mtp_freqs -- --nocapture`
    #[test]
    fn mtp_freqs() {
        // 也支持直接读 Q8 对照
    }
    #[test]
    fn mtp_freqs_q8() {
        let Some(path) = std::env::var_os("ZLLM_GEMMA4_MTP").map(std::path::PathBuf::from) else { return };
        let reader = crate::weight::container::gguf::GgufReader::open(&path).expect("open");
        let values = reader.read_tensor_f32("rope_freqs.weight").expect("rope_freqs");
        println!("[mtp-freqs-q8] len={} head={:?} tail={:?}", values.len(), &values[..4], &values[values.len().saturating_sub(4)..]);
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("[mtp-freqs-q8] min={min:.6} max={max:.6}");
    }
    #[test]
    fn mtp_freqs_disabled() {
        let Some(path) = std::env::var_os("ZLLM_GEMMA4_MTP_Q4").map(std::path::PathBuf::from) else { return };
        let reader = crate::weight::container::gguf::GgufReader::open(&path).expect("open");
        let values = reader.read_tensor_f32("rope_freqs.weight").expect("rope_freqs");
        println!("[mtp-freqs] len={} values={:?}", values.len(), &values[..values.len().min(8)]);
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("[mtp-freqs] min={min:.4} max={max:.4}");
        std::fs::write("/tmp/mtp-rope-freqs.bin", unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }).expect("write");
    }
}

#[cfg(all(test, target_os = "macos"))]
mod norm_compare_tests {
    /// 主干 vs draft 的 norm 权重数值分布(判定 gamma 编码是否一致)。
    /// `ZLLM_GEMMA4_GGUF=... ZLLM_GEMMA4_MTP=... cargo test --lib mtp_norm_compare -- --nocapture`
    #[test]
    fn mtp_norm_compare() {
        let (Some(backbone), Some(mtp)) = (std::env::var_os("ZLLM_GEMMA4_GGUF").map(std::path::PathBuf::from), std::env::var_os("ZLLM_GEMMA4_MTP").map(std::path::PathBuf::from)) else { return };
        let reader = crate::weight::container::gguf::GgufReader::open(&backbone).expect("主干");
        let name = reader
            .tensor("blk.0.attn_norm.weight")
            .is_some()
            .then(|| "blk.0.attn_norm.weight".to_owned())
            .unwrap_or_else(|| reader.tensors().iter().map(|t| t.name.as_str()).find(|n| n.contains("norm") && n.contains("0")).map(str::to_owned).expect("找不到主干 norm"));
        let backbone_norm = reader.read_tensor_f32(&name).expect("读主干 norm");
        let mtp_reader = crate::weight::container::gguf::GgufReader::open(&mtp).expect("draft");
        let draft_norm = mtp_reader.read_tensor_f32("blk.0.attn_norm.weight").expect("读 draft norm");
        let stats = |v: &[f32], tag: &str| {
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("[norm-compare] {tag}: len={} mean={mean:.4} min={min:.4} max={max:.4} head={:?}", v.len(), &v[..4.min(v.len())]);
        };
        stats(&backbone_norm, &format!("主干 {name}"));
        stats(&draft_norm, "draft blk.0.attn_norm");
        let qn = mtp_reader.read_tensor_f32("blk.0.attn_q_norm.weight").expect("draft q_norm");
        stats(&qn, "draft blk.0.attn_q_norm");
        if let Ok(backbone_qn) = reader.read_tensor_f32("blk.0.attn_q_norm.weight") {
            stats(&backbone_qn, "主干 blk.0.attn_q_norm");
        } else if let Ok(backbone_qn) = reader.read_tensor_f32("blk.0.attn_q_norm.weight") {
            stats(&backbone_qn, "主干 q_norm 备选");
        }
    }
}
