//! MiniCPM5 × Metal decode 平铺重放：命令表一次转录，逐 token 重编码一个 CB。
//!
//! 背景（2026-09-09 profile）：2B Q4_K_M 的异步流水线 wall 14.7ms/tok 中 GPU 仅
//! 11.1ms（权重带宽已打满），CPU 编码 14.7ms 是瓶颈——其中 11.2ms 阻塞在
//! `new_command_buffer`（16 ops/CB × 2 轮深度的在飞 CB 积压触发驱动节流）。
//! 重放把每 token 的 CPU 工作缩到「推进 KV 游标 + 写 state 槽 + 重编码一个 CB」，
//! CB 数从 24/token 降到 1/token。层内全部用 position 化原语（rope/attention 从
//! decode_state 槽读位置），命令表与 position 无关；A/B 双表深度-2 流水与现有
//! 异步路径语义对齐（首 token 由普通输出步产出后 prime 进对侧 readback）。

use super::metal_session::{EmbeddingSource, MiniCpm5MetalSequence};
use crate::runtime::minicpm5::{MiniCpm5Config, MiniCpm5OutputHead, MiniCpm5TextLayer};
use crate::attention::gqa::{CausalWindow, GqaSpec};
use crate::attention::rope::RopeTable;
use crate::backend::metal::api::Buffer;
use crate::backend::metal::replay::{DualReplay, ReplayStep};
use crate::backend::metal::{MetalContext, MetalKvCache, MetalTensor, MetalWeight};
use crate::backend::{Backend, BackendError};
use crate::moe::Activation;
use crate::runtime::output::{OutputNorm, OutputPlan};
use half::f16;

fn compute_error(message: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: message.into() }
}

/// Q8 direct-append kernel 的可见行上限（source_rows ≤ 512），超过走均分 split。
const DIRECT_MAX_ROWS: usize = 512;
/// split 录制的块粒度（每块 ≤128 行，与现有动态路径的调优一致）。
const SPLIT_BLOCK_TOKENS: usize = 128;

pub(super) struct Minicpm5ReplayEngine {
    replay: DualReplay,
    states: [Buffer; 2],
    readbacks: [Buffer; 2],
    hidden_outputs: [MetalTensor; 2],
    q8_pipelines: crate::kernel::metal::attention::Q8PositionReplayPipelines,
    current_cache: Buffer,
    layer_count: usize,
    max_seq_len: usize,
}

/// 一步在飞重放：CB 句柄即精确等待点，readback 槽按 parity 区分（无覆写竞态）。
pub(super) struct Minicpm5ReplayPending {
    pub(super) step: ReplayStep,
    placeholder: usize,
}

impl Minicpm5ReplayEngine {
    /// 用首个请求的真实 KV cache 录制（布局 offset 由 capacity 决定，后续请求
    /// 经 [`Self::bind_cache`] 整体替换 buffer 指针）。只支持 Q8 cache 与
    /// Full attention；embedding 需要设备端 gather 来源（untied F16 或 tied Q6_K）。
    pub(super) fn record(
        ctx: &MetalContext,
        cache: &MetalKvCache,
        config: &MiniCpm5Config,
        layers: &[MiniCpm5TextLayer<MetalWeight>],
        rope: &RopeTable,
        output_head: &MiniCpm5OutputHead<MetalWeight>,
        embedding: &EmbeddingSource,
        max_seq_len: usize,
    ) -> Result<Self, BackendError> {
        if layers.len() != config.layer_count {
            return Err(compute_error(format!("MiniCPM5 replay 层数={}，期望 {}", layers.len(), config.layer_count)));
        }
        if cache.format() != crate::kv_cache::KvCacheFormat::Int8 {
            return Err(compute_error("MiniCPM5 replay 需要 Q8 KV cache"));
        }
        let state_bytes = config.layer_count.checked_mul(12).ok_or_else(|| compute_error("MiniCPM5 replay state 大小溢出"))?;
        let slots = [ctx.shared_buffer_zeros(state_bytes), ctx.shared_buffer_zeros(state_bytes)];
        let readbacks = [ctx.shared_buffer_zeros(4), ctx.shared_buffer_zeros(4)];
        let rope_f16 = upload_rope_table(ctx, rope).map_err(compute_error)?;
        let split_blocks = max_seq_len.div_ceil(SPLIT_BLOCK_TOKENS);
        let spec = minicpm5_replay_spec(config);
        let mut hidden_outputs: [Option<MetalTensor>; 2] = [None, None];
        let replay = DualReplay::record(|parity| {
            let hidden = record_round(ctx, cache, config, layers, output_head, embedding, &spec, &rope_f16, &slots[parity], &readbacks[parity], &readbacks[1 - parity], split_blocks)?;
            hidden_outputs[parity] = Some(hidden);
            Ok(())
        })?;
        let hidden_outputs = hidden_outputs.map(|slot| slot.expect("MiniCPM5 replay 两个 parity 都必须录制 hidden"));
        let q8_pipelines = crate::kernel::metal::attention::Q8PositionReplayPipelines::new(ctx).map_err(compute_error)?;
        Ok(Self { replay, states: slots, readbacks, hidden_outputs, q8_pipelines, current_cache: cache.buffer().clone(), layer_count: config.layer_count, max_seq_len })
    }

    /// 仅测试诊断使用(engine.rs 录制耗时打点)。
    #[cfg(test)]
    pub(super) fn command_count(&self) -> usize {
        self.replay.command_count()
    }

    /// 请求级 KV cache 重绑定：录制期钉住的 cache buffer 换成当前请求实例。
    pub(super) fn bind_cache(&mut self, cache: &MetalKvCache) {
        if !self.current_cache.same_handle(cache.buffer()) {
            self.replay.remap_buffers(&[(self.current_cache.clone(), cache.buffer().clone())]);
            self.current_cache = cache.buffer().clone();
        }
    }

    /// 首步 prime：prefill 后的第一个 token 由普通输出步产出，写进对侧 readback
    /// 供第一轮 gather 读取（`parity` 是即将提交的步的槽位）。
    pub(super) fn prime(&self, parity: usize, token: u32) {
        unsafe { *self.readbacks[1 - parity].contents().cast::<u32>() = token };
    }

    /// 提交一步：推进 KV 游标、写 state 槽、按当前 KV 长度选 direct/split 路径，
    /// 重编码一个 CB 并 commit（不等待）。`sequence.tokens` 先占位，wait 时回填。
    pub(super) fn step(&self, ctx: &MetalContext, sequence: &mut MiniCpm5MetalSequence, parity: usize) -> Result<Minicpm5ReplayPending, BackendError> {
        let position = sequence.tokens.len();
        if position >= self.max_seq_len {
            return Err(compute_error(format!("MiniCPM5 replay position={position} 超过 max_seq_len {}", self.max_seq_len)));
        }
        for layer in 0..self.layer_count {
            sequence.cache.reserve_layer_gqa_row(layer, position).map_err(compute_error)?;
        }
        let slots = unsafe { std::slice::from_raw_parts_mut(self.states[parity].contents().cast::<u32>(), self.layer_count * 3) };
        for slot in slots.chunks_exact_mut(3) {
            slot[0] = position as u32;
            slot[1] = (position + 1) as u32;
            slot[2] = 0;
        }
        sequence.tokens.push(0);
        let split = position + 1 > DIRECT_MAX_ROWS;
        let pipelines = &self.q8_pipelines;
        let step = self.replay.submit_filtered(ctx, parity, &|op| pipelines.accepts(op, split))?;
        sequence.hidden = self.hidden_outputs[parity].clone();
        Ok(Minicpm5ReplayPending { step, placeholder: position })
    }

    /// 精确等待一步并读回 token id；已提交的后续步继续在 GPU 上执行。
    pub(super) fn wait_token(&self, sequence: &mut MiniCpm5MetalSequence, pending: &Minicpm5ReplayPending) -> u32 {
        pending.step.command.wait_until_completed();
        let token = unsafe { *self.readbacks[pending.step.parity].contents().cast::<u32>() };
        if let Some(slot) = sequence.tokens.get_mut(pending.placeholder) {
            *slot = token;
        }
        token
    }
}

fn minicpm5_replay_spec(config: &MiniCpm5Config) -> GqaSpec {
    GqaSpec {
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        rope_dim: config.head_dim,
        rope_theta: config.rope_theta,
        use_qk_norm: false,
        window: CausalWindow::Full,
        score_scale: 1.0 / (config.head_dim as f32).sqrt(),
        output_gate: false,
    }
}

struct RopeF16Table {
    cos: Buffer,
    sin: Buffer,
    max_positions: usize,
}

fn upload_rope_table(ctx: &MetalContext, table: &RopeTable) -> Result<RopeF16Table, String> {
    // MiniCPM5 rotary_dim == head_dim（全维度 Interleaved）。
    let half_dim = table.rotary_dim / 2;
    let needed = table.seq_len.checked_mul(half_dim).ok_or("MiniCPM5 replay RoPE 表长度溢出")?;
    if table.cos.len() < needed || table.sin.len() < needed {
        return Err("MiniCPM5 replay RoPE cos/sin 长度不足".to_owned());
    }
    let upload = |values: &[f32]| -> Buffer {
        let packed: Vec<u16> = values[..needed].iter().map(|&value| f16::from_f32(value).to_bits()).collect();
        let bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) };
        ctx.shared_buffer(bytes)
    };
    Ok(RopeF16Table { cos: upload(&table.cos), sin: upload(&table.sin), max_positions: table.seq_len })
}

/// 录制一份 parity 的完整 decode 步：gather 上一 token → 42 层 → 输出步。
/// 层内 projection/norm/MLP 走常规 dispatch（静态参数），rope 与 attention
/// 用 position 化原语，全部经 Transcriber 固化。
#[allow(clippy::too_many_arguments)]
fn record_round(
    ctx: &MetalContext,
    cache: &MetalKvCache,
    config: &MiniCpm5Config,
    layers: &[MiniCpm5TextLayer<MetalWeight>],
    output_head: &MiniCpm5OutputHead<MetalWeight>,
    embedding: &EmbeddingSource,
    spec: &GqaSpec,
    rope: &RopeF16Table,
    state: &Buffer,
    readback: &Buffer,
    previous_readback: &Buffer,
    split_blocks: usize,
) -> Result<MetalTensor, BackendError> {
    let mut hidden = match embedding {
        EmbeddingSource::F16(matrix) => crate::kernel::metal::shape::gather_row_f16_tensor_offset(ctx, previous_readback, 0, matrix).map_err(compute_error)?,
        EmbeddingSource::Q6k { blob, row_bytes } => crate::kernel::metal::gguf::gguf_gather_row_q6k_tensor_offset(ctx, previous_readback, 0, blob, config.vocab_size, config.hidden_size, *row_bytes).map_err(compute_error)?,
    };
    for (layer, weights) in layers.iter().enumerate() {
        let state_offset = (layer * 12) as u64;
        let normed = ctx.rmsnorm(&hidden, &weights.input_norm, config.rms_eps)?;
        let (query, key) = ctx.dual_linear(&normed, &weights.query, &weights.key)?;
        let value = ctx.linear(&normed, &weights.value)?;
        let query = crate::kernel::metal::shape::apply_rope_interleaved_position_tensor(ctx, &query, config.num_heads, config.head_dim, &rope.cos, &rope.sin, rope.max_positions, state, state_offset).map_err(compute_error)?;
        let key = crate::kernel::metal::shape::apply_rope_interleaved_position_tensor(ctx, &key, config.num_kv_heads, config.head_dim, &rope.cos, &rope.sin, rope.max_positions, state, state_offset).map_err(compute_error)?;
        let view = cache.gqa_layer_view(layer).map_err(compute_error)?;
        let attended = crate::kernel::metal::attention::gqa_decode_attention_append_adaptive_q8_position_tensor(ctx, &query, &key, &value, &view, spec, state, state_offset, split_blocks).map_err(compute_error)?;
        let attention_residual = ctx.linear_add(&attended, &weights.output, &hidden)?;
        let normed = ctx.rmsnorm(&attention_residual, &weights.post_attention_norm, config.rms_eps)?;
        hidden = ctx.gated_mlp_add_residual(&normed, &weights.gate, &weights.up, &weights.down, &Activation::Silu, &attention_residual)?;
    }
    // 输出步：final norm + lm head + argmax 写本 parity readback[0]。
    let plan = OutputPlan { eps: config.rms_eps, norm: OutputNorm::Rms, excluded_tokens: Vec::new() };
    let (hidden, logits) = crate::runtime::output::norm_and_lm_head(ctx, output_head, &hidden, &plan)?;
    crate::kernel::metal::moe::argmax_tensor_into_offset(ctx, &logits, &plan.excluded_tokens, readback, 0).map_err(compute_error)?;
    Ok(hidden)
}
