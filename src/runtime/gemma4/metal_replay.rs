//! Gemma 4 × Metal decode ICB 重放:position 化层前向转录一次,逐 token 重放。
//!
//! 转录器把常规 dispatch 函数的绑定镜像成 ICB 命令,之后每 token CPU 只更新三样
//! 东西:embedding 行、decode_state 槽、重放序列。
//!
//! M5(AGXG17G)实测(2026-08):ICB 形态(录制 436s/重放 337s/驱动疑似丢弃上下文)
//! 不可用——两处病态均为 ICB 驱动开销,非捕获-重放概念本身。2026-08-25 起改为
//! 平铺转录(Transcriber::begin_flat):dispatch 记为调用表,重放用普通 encoder
//! 重编码,完全绕开 ICB。瓶颈背景:decode 为 GPU-bound,重放的收益在 CPU 编码
//! 61ms -> ~3ms 与静态操作表,为深融合/投机解码留空间,wall 提升有限。

use crate::{
    attention::{
        gqa::{CausalWindow, GqaSpec, HybridGqaLayerSpec},
        rope::RopeTable,
    },
    backend::{
        Backend, BackendError, GqaPrefillBackend,
        metal::{
            MetalContext, MetalKvCache, MetalTensor, MetalWeight,
            api::{Buffer, Transcriber},
            replay::{DualReplay, ReplayStep},
        },
    },
    kernel::metal::{attention as metal_attention, shape as metal_shape},
    moe::{
        Activation,
        dense_mlp::{DenseMlpSpec, DenseMlpWeightsRef},
    },
    runtime::gemma4::{Gemma4, Gemma4Layer, Gemma4OutputHead, Gemma4PerLayerModel, Gemma4RopeTables, Gemma4Weights, gemma4_embedding_rows, gemma4_per_layer_embedding_rows, gemma4_per_layer_inputs, gemma4_token_output},
};
use half::{bf16, f16};

/// 与模型层规格一致的 GQA attention 参数(gemma4/mod.rs 的私有同名函数的镜像,
/// 重放组合模块独立持有,避免跨模块泄漏私有实现)。
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

fn compute_error(message: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: message.into() }
}

/// F16 常驻 RoPE 表(sliding/full 各一份),kernel 按全局 position 自行索引。
struct RopeF16Tables {
    cos: Buffer,
    sin: Buffer,
    rotary_dim: usize,
    max_positions: usize,
}

fn upload_rope_table(ctx: &MetalContext, table: &RopeTable) -> Result<RopeF16Tables, String> {
    let half_dim = table.rotary_dim / 2;
    let needed = table.seq_len.checked_mul(half_dim).ok_or("RoPE 表长度溢出")?;
    if table.cos.len() < needed || table.sin.len() < needed {
        return Err("RoPE 表 cos/sin 长度不足".to_owned());
    }
    let upload = |values: &[f32]| -> Buffer {
        let packed: Vec<u16> = values[..needed].iter().map(|&value| f16::from_f32(value).to_bits()).collect();
        let bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) };
        ctx.shared_buffer(bytes)
    };
    Ok(RopeF16Tables { cos: upload(&table.cos), sin: upload(&table.sin), rotary_dim: table.rotary_dim, max_positions: table.seq_len })
}

pub struct Gemma4DecodeReplay<'a> {
    ctx: &'a MetalContext,
    model: &'a Gemma4,
    weights: &'a Gemma4Weights,
    cache: &'a mut MetalKvCache,
    commands: crate::backend::metal::api::CommandList,
    decode_state: Buffer,
    input: MetalTensor,
    token_inputs: Option<MetalTensor>,
    embedding_scale: f32,
}

impl<'a> Gemma4DecodeReplay<'a> {
    /// 录制一遍 decode 步:输入内容与 kv 状态是占位(GPU 不执行),只有命令序列被固化。
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        ctx: &'a MetalContext,
        cache: &'a mut MetalKvCache,
        model: &'a Gemma4,
        layers: &'a [Gemma4Layer<MetalWeight>],
        rope: &'a Gemma4RopeTables,
        weights: &'a Gemma4Weights,
        per_layer_model: Option<&'a Gemma4PerLayerModel<MetalWeight>>,
        output_head: &'a Gemma4OutputHead<MetalWeight>,
    ) -> Result<Self, BackendError> {
        let cfg = model.config();
        let input = ctx.tensor_zeros(1, cfg.hidden_size);
        let token_inputs = (cfg.per_layer_input_size > 0).then(|| ctx.tensor_zeros(1, cfg.layer_count * cfg.per_layer_input_size));
        let state_bytes = cfg.layer_count.checked_mul(12).ok_or_else(|| compute_error("decode state 大小溢出"))?;
        let decode_state = ctx.shared_buffer_zeros(state_bytes);
        let rope_sliding = upload_rope_table(ctx, &rope.sliding).map_err(compute_error)?;
        let rope_full = upload_rope_table(ctx, &rope.full).map_err(compute_error)?;

        // 平铺转录:整层序列记为普通调用表(无 ICB 依赖,记录成本 ~µs/op)
        Transcriber::begin_flat().map_err(compute_error)?;
        let recorded = record_round(ctx, cache, model, layers, per_layer_model, output_head, &input, token_inputs.as_ref(), &decode_state, &rope_sliding, &rope_full);
        let transcriber = Transcriber::end().expect("转录器应在进行");
        let commands = transcriber.into_command_list().map_err(compute_error)?;
        recorded?;
        let embedding_scale = bf16::from_f32(cfg.embedding_scale()).to_f32();
        Ok(Self { ctx, model, weights, cache, commands, decode_state, input, token_inputs, embedding_scale })
    }

    pub fn command_count(&self) -> usize {
        self.commands.ops.len()
    }

    /// 静态命令表(消融/诊断用)。
    pub fn ops(&self) -> &[crate::backend::metal::api::RecordedComputeOp] {
        &self.commands.ops
    }

    /// 一个 decode token:写输入与 kv 状态 → 重放 → 读回 argmax 结果。
    pub fn step(&mut self, token: u32, position: usize) -> Result<u32, BackendError> {
        self.step_filtered(token, position, &|_| true)
    }

    /// 算子消融重放:keep 为 false 的 kernel 被跳过(输出无效,仅用于计时分解)。
    pub fn step_filtered(&mut self, token: u32, position: usize, keep: &dyn Fn(&crate::backend::metal::api::RecordedComputeOp) -> bool) -> Result<u32, BackendError> {
        let cfg = self.model.config();
        if position >= cfg.max_position_embeddings {
            return Err(compute_error(format!("Gemma4 重放 position={position} 超过上下文上限")));
        }
        let row = gemma4_embedding_rows(self.weights, &[token], cfg.hidden_size, self.embedding_scale).map_err(compute_error)?;
        write_f16_row(&self.input, &row);
        if let Some(target) = &self.token_inputs {
            let values = gemma4_per_layer_embedding_rows(cfg, self.weights, &[token]).map_err(compute_error)?;
            write_f16_row(target, &values);
        }
        {
            let state = self.cache.hybrid_gqa_state_mut().ok_or_else(|| compute_error("重放需要 hybrid GQA cache"))?;
            for layer in 0..cfg.layer_count {
                if self.model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?.kv_source_layer.is_none() {
                    let plan = state.plan_append(layer, position, 1).map_err(compute_error)?;
                    state.commit_append(&plan).map_err(compute_error)?;
                }
            }
        }
        let state = self.cache.hybrid_gqa_state().ok_or_else(|| compute_error("重放需要 hybrid GQA cache"))?;
        let slots = unsafe { std::slice::from_raw_parts_mut(self.decode_state.contents().cast::<u32>(), cfg.layer_count * 3) };
        for layer in 0..cfg.layer_count {
            let spec = self.model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?;
            let read_layer = spec.kv_source_layer.unwrap_or(layer);
            let retained = state.retained_range(read_layer).map_err(compute_error)?;
            let slot = &mut slots[layer * 3..layer * 3 + 3];
            slot[0] = position as u32;
            slot[1] = retained.end as u32;
            slot[2] = retained.start as u32;
        }
        // 重放:普通 encoder 按录制顺序重编码(encoder 序即执行序,无 ICB 依赖)
        let command = self.ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        for op in &self.commands.ops {
            if keep(op) {
                encoder.encode_recorded(op);
            }
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let readback = self.ctx.token_readback_buffer();
        Ok(unsafe { *readback.contents().cast::<u32>() })
    }
}

fn write_f16_row(target: &MetalTensor, values: &[f32]) {
    let packed: Vec<u16> = values.iter().map(|&value| f16::from_f32(value).to_bits()).collect();
    let destination = unsafe { std::slice::from_raw_parts_mut(target.buffer.contents().cast::<u8>(), packed.len() * 2) };
    destination.copy_from_slice(unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) });
}

/// 生产版 decode 重放:双份命令表,设备端闭环。
///
/// parity p 的一步:步首 gather kernel 从 slots[1-p].readback 读上一步产出的
/// token,在常驻 Q4_K tied embedding 上抠行(×scale)作为 input；E4B 同时从
/// 常驻 Q5_K per-layer embedding 抠行 → 模型层前向
/// (position 全部经 state 槽间接寻址,命令表 position 无关)→ norm+lm head+
/// argmax 写 slots[p].readback。queue FIFO 保证步 n+1 的 gather 读到步 n 的
/// readback,因此 CPU 可以在步 n 在飞时直接提交步 n+1(深度-2 流水),只在
/// wait 点读 token;首步由 prime() 预写对侧 readback。
pub struct Gemma4ReplayEngine {
    replay: DualReplay,
    slots: [ReplaySlot; 2],
    current_cache: Buffer,
    layer_count: usize,
    /// 持有 K/V(append 目标)的层;共享 KV 层读源层 cache,不 append。
    kv_layers: Vec<bool>,
    kv_source: Vec<Option<usize>>,
}

struct ReplaySlot {
    state: Buffer,
    readback: Buffer,
}

impl Gemma4ReplayEngine {
    /// 录制 A/B 两份命令表。要求 GGUF Q4_K tied lm_head；E4B 还要求常驻
    /// Q5_K per-layer embedding，二者都由上一步设备 token id 直接 gather。
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        ctx: &MetalContext,
        cache: &MetalKvCache,
        model: &Gemma4,
        layers: &[Gemma4Layer<MetalWeight>],
        per_layer_model: Option<&Gemma4PerLayerModel<MetalWeight>>,
        rope: &Gemma4RopeTables,
        output_head: &Gemma4OutputHead<MetalWeight>,
        embedding_blob: &Buffer,
        embedding_row_bytes: usize,
        per_layer_embedding_blob: Option<&Buffer>,
        per_layer_embedding_row_bytes: Option<usize>,
        embedding_scale: f32,
    ) -> Result<Self, BackendError> {
        let cfg = model.config();
        if cfg.per_layer_input_size != 0 && (per_layer_model.is_none() || per_layer_embedding_blob.is_none() || per_layer_embedding_row_bytes.is_none()) {
            return Err(compute_error("Gemma4 E4B 重放缺少 per-layer model 或 Q5_K embedding"));
        }
        let state_bytes = cfg.layer_count.checked_mul(12).ok_or_else(|| compute_error("decode state 大小溢出"))?;
        let slots = [ReplaySlot { state: ctx.shared_buffer_zeros(state_bytes), readback: ctx.shared_buffer_zeros(4) }, ReplaySlot { state: ctx.shared_buffer_zeros(state_bytes), readback: ctx.shared_buffer_zeros(4) }];
        let rope_sliding = upload_rope_table(ctx, &rope.sliding).map_err(compute_error)?;
        let rope_full = upload_rope_table(ctx, &rope.full).map_err(compute_error)?;
        let layer_count = cfg.layer_count;
        let kv_layers: Vec<bool> = (0..layer_count).map(|layer| model.layer_spec(layer).expect("层规格").kv_source_layer.is_none()).collect();
        let kv_source: Vec<Option<usize>> = (0..layer_count).map(|layer| model.layer_spec(layer).expect("层规格").kv_source_layer).collect();

        let replay = DualReplay::record(|parity| {
            let slot = &slots[parity];
            // 步首 gather:从对侧 readback 读上一步 token,设备端抠 embedding 行
            let mut hidden =
                crate::kernel::metal::gguf::gguf_gather_row_q4k_tensor_offset(ctx, &slots[1 - parity].readback, 0, embedding_blob, cfg.vocab_size, cfg.hidden_size, embedding_row_bytes, embedding_scale).map_err(compute_error)?;
            let per_layer_inputs = if let (Some(blob), Some(row_bytes)) = (per_layer_embedding_blob, per_layer_embedding_row_bytes) {
                let columns = cfg.layer_count * cfg.per_layer_input_size;
                let scale = bf16::from_f32(cfg.per_layer_embedding_scale().expect("存在 per-layer embedding blob")).to_f32();
                let token_inputs = crate::kernel::metal::gguf::gguf_gather_row_q5k_tensor_offset(ctx, &slots[1 - parity].readback, 0, blob, cfg.vocab_size, columns, row_bytes, scale).map_err(compute_error)?;
                gemma4_per_layer_inputs(ctx, cfg, per_layer_model, &hidden, Some(token_inputs))?
            } else {
                None
            };
            for (layer, weights) in layers.iter().enumerate() {
                let spec = model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?.attention.hybrid;
                let rope = match spec.window {
                    CausalWindow::Full => &rope_full,
                    CausalWindow::Sliding { .. } => &rope_sliding,
                };
                hidden = record_layer(ctx, cache, model, layer, spec, weights, &hidden, per_layer_inputs.as_ref().map(|inputs| &inputs[layer]), rope, &slot.state)?;
            }
            // 输出步:final norm + lm head + argmax 写本 parity 的 readback[0]。
            let plan = crate::runtime::output::OutputPlan { eps: cfg.rms_eps, norm: crate::runtime::output::OutputNorm::GemmaRms, excluded_tokens: vec![cfg.end_image_token_id, cfg.end_audio_token_id] };
            let (_normed, logits) = crate::runtime::output::norm_and_lm_head(ctx, output_head, &hidden, &plan)?;
            crate::kernel::metal::moe::argmax_tensor_into_offset(ctx, &logits, &plan.excluded_tokens, &slot.readback, 0).map_err(compute_error)?;
            Ok(())
        })?;
        Ok(Self { replay, slots, current_cache: cache.buffer().clone(), layer_count, kv_layers, kv_source })
    }

    pub fn command_count(&self) -> usize {
        self.replay.command_count()
    }

    /// 请求级 KV cache 绑定:命令表内旧 cache 指针整体替换为本请求实例。
    pub fn bind_cache(&mut self, cache: &MetalKvCache) {
        let replacements = [(self.current_cache.clone(), cache.buffer().clone())];
        self.replay.remap_buffers(&replacements);
        self.current_cache = cache.buffer().clone();
    }

    /// 预写某 parity 的 readback(首轮:对侧 readback 没有上一步产出,首步的
    /// gather 以此取得 prefill 首 token)。
    pub fn prime(&self, parity: usize, token: u32) {
        unsafe { *self.slots[parity].readback.contents().cast::<u32>() = token };
    }

    /// 提交一个 decode 步:推进 KV 游标、写 state 槽(position/retained 区间),
    /// 重编码本 parity 命令表为一个 CB 并 commit,不等待。
    pub fn step_async(&self, ctx: &MetalContext, cache: &mut MetalKvCache, parity: usize, position: usize) -> Result<ReplayStep, BackendError> {
        {
            let state = cache.hybrid_gqa_state_mut().ok_or_else(|| compute_error("重放需要 hybrid GQA cache"))?;
            for layer in 0..self.layer_count {
                if self.kv_layers[layer] {
                    let plan = state.plan_append(layer, position, 1).map_err(compute_error)?;
                    state.commit_append(&plan).map_err(compute_error)?;
                }
            }
        }
        let state = cache.hybrid_gqa_state().ok_or_else(|| compute_error("重放需要 hybrid GQA cache"))?;
        let slots = unsafe { std::slice::from_raw_parts_mut(self.slots[parity].state.contents().cast::<u32>(), self.layer_count * 3) };
        for layer in 0..self.layer_count {
            let read_layer = self.kv_source[layer].unwrap_or(layer);
            let retained = state.retained_range(read_layer).map_err(compute_error)?;
            let slot = &mut slots[layer * 3..layer * 3 + 3];
            slot[0] = position as u32;
            slot[1] = retained.end as u32;
            slot[2] = retained.start as u32;
        }
        self.replay.submit(ctx, parity)
    }

    /// 精确等待一步并读回 token id(CB 完成 ⇔ 此前全部在飞工作完成)。
    pub fn wait_token(&self, step: &ReplayStep) -> u32 {
        step.command.wait_until_completed();
        unsafe { *self.slots[step.parity].readback.contents().cast::<u32>() }
    }
}

/// 录制一遍 decode round(转录模式):层前向全部走常规 dispatch,rope/append/attention
/// 使用 position 化原语,命令序列与数值语义与重放完全一致。
#[allow(clippy::too_many_arguments)]
fn record_round(
    ctx: &MetalContext,
    cache: &MetalKvCache,
    model: &Gemma4,
    layers: &[Gemma4Layer<MetalWeight>],
    per_layer_model: Option<&Gemma4PerLayerModel<MetalWeight>>,
    output_head: &Gemma4OutputHead<MetalWeight>,
    input: &MetalTensor,
    token_inputs: Option<&MetalTensor>,
    decode_state: &Buffer,
    rope_sliding: &RopeF16Tables,
    rope_full: &RopeF16Tables,
) -> Result<(), BackendError> {
    let cfg = model.config();
    if layers.len() != model.layer_count() {
        return Err(compute_error("重放层权重与模型层数不一致"));
    }
    let per_layer_inputs = gemma4_per_layer_inputs(ctx, cfg, per_layer_model, input, token_inputs.cloned())?;
    let per_layer_inputs = per_layer_inputs.as_deref();
    let mut hidden = input.clone();
    for (layer, weights) in layers.iter().enumerate() {
        let spec = model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?;
        let rope = match spec.attention.hybrid.window {
            CausalWindow::Full => rope_full,
            CausalWindow::Sliding { .. } => rope_sliding,
        };
        hidden = record_layer(ctx, cache, model, layer, spec.attention.hybrid, weights, &hidden, per_layer_inputs.map(|inputs| &inputs[layer]), rope, decode_state)?;
    }
    // head(norm → lm head gemv → argmax)同样转录;录制步的 token_id 是占位,重放后由引擎读 readback
    let _ = gemma4_token_output(ctx, cfg, output_head, &hidden)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_layer(
    ctx: &MetalContext,
    cache: &MetalKvCache,
    model: &Gemma4,
    layer: usize,
    attention: HybridGqaLayerSpec,
    weights: &Gemma4Layer<MetalWeight>,
    hidden: &MetalTensor,
    per_layer_input: Option<&MetalTensor>,
    rope: &RopeF16Tables,
    decode_state: &Buffer,
) -> Result<MetalTensor, BackendError> {
    let cfg = model.config();
    let geometry = attention.geometry;
    let layout = crate::attention::rope::RotaryLayout::SplitHalf;
    let state_offset = (layer * 12) as u64;
    let attention_output = if let Some(source_layer) = model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?.kv_source_layer {
        // 共享 KV 层:只算 query,attention 读源层 cache(源层槽已含本步 append 后状态)
        let normed = ctx.rmsnorm(hidden, &weights.input_norm, cfg.rms_eps)?;
        if weights.attention.key.is_some() || weights.attention.value.is_some() {
            return Err(compute_error(format!("Gemma4 shared-KV L{layer} 不应持有 K/V 权重")));
        }
        let query = ctx.linear(&normed, &weights.attention.query)?;
        let query = ctx.gemma_rmsnorm_heads(&query, &weights.attention.query_norm, geometry.num_heads, geometry.head_dim, cfg.rms_eps)?;
        let query = metal_shape::apply_rope_position_tensor(ctx, &query, geometry.num_heads, rope.rotary_dim, layout, &rope.cos, &rope.sin, rope.max_positions, decode_state, state_offset).map_err(compute_error)?;
        let view = cache.gqa_layer_view(source_layer).map_err(compute_error)?;
        metal_attention::gqa_decode_attention_position_tensor(ctx, &query, &view, &gqa_spec(attention), decode_state, (source_layer * 12) as u64).map_err(compute_error)?
    } else {
        let key_weight = weights.attention.key.as_ref().ok_or_else(|| compute_error(format!("Gemma4 L{layer} 缺少 K 权重")))?;
        let normed = ctx.rmsnorm(hidden, &weights.input_norm, cfg.rms_eps)?;
        let (query, key_source, value_source) = match &weights.attention.value {
            Some(value) => {
                let (query, key, val) = ctx.triple_linear(&normed, &weights.attention.query, key_weight, value)?;
                (query, key, Some(val))
            }
            None => {
                let (query, key) = ctx.dual_linear(&normed, &weights.attention.query, key_weight)?;
                (query, key, None)
            }
        };
        let query = ctx.gemma_rmsnorm_heads(&query, &weights.attention.query_norm, geometry.num_heads, geometry.head_dim, cfg.rms_eps)?;
        let key = ctx.gemma_rmsnorm_heads(&key_source, &weights.attention.key_norm, geometry.num_kv_heads, geometry.head_dim, cfg.rms_eps)?;
        let value_source = value_source.as_ref().unwrap_or(&key_source);
        let value = ctx.gemma_rmsnorm_heads(value_source, &weights.attention.value_norm, geometry.num_kv_heads, geometry.head_dim, cfg.rms_eps)?;
        let query = metal_shape::apply_rope_position_tensor(ctx, &query, geometry.num_heads, rope.rotary_dim, layout, &rope.cos, &rope.sin, rope.max_positions, decode_state, state_offset).map_err(compute_error)?;
        let key = metal_shape::apply_rope_position_tensor(ctx, &key, geometry.num_kv_heads, rope.rotary_dim, layout, &rope.cos, &rope.sin, rope.max_positions, decode_state, state_offset).map_err(compute_error)?;
        let view = cache.gqa_layer_view(layer).map_err(compute_error)?;
        metal_attention::gqa_kv_append_position_tensor(ctx, &view, &key, &value, decode_state, state_offset).map_err(compute_error)?;
        metal_attention::gqa_decode_attention_position_tensor(ctx, &query, &view, &gqa_spec(attention), decode_state, state_offset).map_err(compute_error)?
    };
    let attention_output = ctx.linear(&attention_output, &weights.attention.output)?;
    let hidden = ctx.rmsnorm_add_scaled(hidden, &attention_output, &weights.post_attention_norm, cfg.rms_eps, 1.0)?;

    let mlp_input = ctx.rmsnorm(&hidden, &weights.pre_feedforward_norm, cfg.rms_eps)?;
    let mlp = crate::moe::dense_mlp::forward_observed(
        ctx,
        &DenseMlpSpec { intermediate_size: cfg.intermediate_size, activation: Activation::GeluTanh },
        DenseMlpWeightsRef { gate: &weights.mlp.gate, up: &weights.mlp.up, down: &weights.mlp.down },
        &mlp_input,
        |_| {},
    )?;
    match (&weights.per_layer_input, per_layer_input) {
        (None, None) => ctx.rmsnorm_add_scaled(&hidden, &mlp, &weights.post_feedforward_norm, cfg.rms_eps, weights.layer_scalar),
        (Some(per_layer_weights), Some(per_layer_input)) => {
            let hidden = ctx.rmsnorm_add_scaled(&hidden, &mlp, &weights.post_feedforward_norm, cfg.rms_eps, 1.0)?;
            let gated = ctx.linear_gated_activation(&hidden, &per_layer_weights.gate, per_layer_input, &Activation::GeluTanh)?;
            let projected = ctx.linear(&gated, &per_layer_weights.projection)?;
            ctx.rmsnorm_add_scaled(&hidden, &projected, &per_layer_weights.post_norm, cfg.rms_eps, weights.layer_scalar)
        }
        _ => Err(compute_error("Gemma4 layer weights 与 per-layer input 不匹配")),
    }
}

/// MTP verify 的命令重放(K+1 行恒定):多行 position 原语(rope/append/
/// attention 的 rows 版)使命令表 position 无关。CPU 每轮写 4 行 embedding
/// 与 48 层 state 槽,重编码一个 CB 提交;verify 138ms → 预期 ~65ms
/// (消 48 个 CB 边界与逐算子 Rust 准备,GPU 贴带宽)。
pub struct Gemma4VerifyReplay {
    commands: crate::backend::metal::api::CommandList,
    input: MetalTensor,
    token_inputs: Option<MetalTensor>,
    state: Buffer,
    readback: Buffer,
    normed: MetalTensor,
    current_cache: Buffer,
    layer_count: usize,
    rows: usize,
    kv_layers: Vec<bool>,
    kv_source: Vec<Option<usize>>,
}

impl Gemma4VerifyReplay {
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        ctx: &MetalContext,
        cache: &MetalKvCache,
        model: &Gemma4,
        layers: &[Gemma4Layer<MetalWeight>],
        per_layer_model: Option<&Gemma4PerLayerModel<MetalWeight>>,
        rope: &Gemma4RopeTables,
        output_head: &Gemma4OutputHead<MetalWeight>,
        rows: usize,
        final_norm_f32: &[f32],
    ) -> Result<Self, BackendError> {
        let cfg = model.config();
        if cfg.per_layer_input_size != 0 && per_layer_model.is_none() {
            return Err(compute_error("verify 重放缺少 per-layer model"));
        }
        // draft 的 inp_h 用 F32(draft 位置 1 接受率 54.6% vs llama.cpp 79.5%,
        // F16 hidden 精度损失翻转 argmax;llama.cpp verify_h 全程 F32)
        let f32_norm = <MetalContext as crate::backend::BackendResources>::prepare_f32(ctx, &final_norm_f32.iter().map(|value| value - 1.0).collect::<Vec<_>>(), 1, final_norm_f32.len())
            .map_err(|error| compute_error(format!("F32 final norm: {error:?}")))?;
        let state_bytes = cfg.layer_count.checked_mul(12).ok_or_else(|| compute_error("verify state 大小溢出"))?;
        let state = ctx.shared_buffer_zeros(state_bytes);
        let input = ctx.tensor_zeros(rows, cfg.hidden_size);
        let token_inputs = (cfg.per_layer_input_size != 0).then(|| ctx.tensor_zeros(rows, cfg.layer_count * cfg.per_layer_input_size));
        let readback = ctx.shared_buffer_zeros(rows * 4);
        let layer_count = cfg.layer_count;
        let kv_layers: Vec<bool> = (0..layer_count).map(|layer| model.layer_spec(layer).expect("层规格").kv_source_layer.is_none()).collect();
        let kv_source: Vec<Option<usize>> = (0..layer_count).map(|layer| model.layer_spec(layer).expect("层规格").kv_source_layer).collect();
        let rope_sliding = upload_rope_table(ctx, &rope.sliding).map_err(compute_error)?;
        let rope_full = upload_rope_table(ctx, &rope.full).map_err(compute_error)?;
        let excluded = vec![cfg.end_image_token_id, cfg.end_audio_token_id];

        crate::backend::metal::api::Transcriber::begin_flat().map_err(compute_error)?;
        let recorded = (|| -> Result<MetalTensor, BackendError> {
            let per_layer_inputs = gemma4_per_layer_inputs(ctx, cfg, per_layer_model, &input, token_inputs.clone())?;
            let per_layer_inputs = per_layer_inputs.as_deref();
            let mut hidden = input.clone();
            for (layer, weights) in layers.iter().enumerate() {
                let spec = model.layer_spec(layer).map_err(|error| compute_error(error.to_string()))?.attention.hybrid;
                let rope_table = match spec.window {
                    CausalWindow::Full => &rope_full,
                    CausalWindow::Sliding { .. } => &rope_sliding,
                };
                let offset = (layer * 12) as u64;
                hidden = record_verify_layer(ctx, cache, layer, spec, kv_source[layer], cfg, weights, &hidden, per_layer_inputs.map(|inputs| &inputs[layer]), rope_table, &state, offset)?;
            }
            // 输出步:F32 norm(draft inp_h 精度)+ lm_head + 逐行 argmax。
            let plan = crate::runtime::output::OutputPlan { eps: cfg.rms_eps, norm: crate::runtime::output::OutputNorm::GemmaRms, excluded_tokens: excluded };
            let hidden_f32 = crate::kernel::metal::to_f32_tensor(ctx, &hidden).map_err(|error| compute_error(error.to_string()))?;
            let normed = crate::backend::Backend::gemma_rmsnorm_f32(ctx, &hidden_f32, &f32_norm, cfg.rms_eps)?;
            let normed_f16 = crate::kernel::metal::to_f16_tensor(ctx, &normed).map_err(|error| compute_error(error.to_string()))?;
            let logits = crate::backend::Backend::linear(ctx, &normed_f16, output_head.lm_head())?;
            // argmax_tensor_into_offset 是单行全量:多行 logits 必须先按行切
            // (argmax_rows 的默认组合同款),否则返回全局扁平 idx(超词表)。
            for row in 0..rows {
                let row_logits = crate::backend::Backend::select_row(ctx, &logits, row).map_err(|error| compute_error(format!("{error:?}")))?;
                crate::kernel::metal::moe::argmax_tensor_into_offset(ctx, &row_logits, &plan.excluded_tokens, &readback, (row * 4) as u64).map_err(compute_error)?;
            }
            Ok(normed)
        })();
        let transcriber = crate::backend::metal::api::Transcriber::end().ok_or_else(|| compute_error("verify 重放转录器未在进行"))?;
        let commands = transcriber.into_command_list().map_err(compute_error)?;
        let normed = recorded?;
        Ok(Self { commands, input, token_inputs, state, readback, normed, current_cache: cache.buffer().clone(), layer_count, rows, kv_layers, kv_source })
    }

    pub fn command_count(&self) -> usize {
        self.commands.ops.len()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn bind_cache(&mut self, cache: &MetalKvCache) {
        let replacements = [(self.current_cache.clone(), cache.buffer().clone())];
        self.commands.remap_buffers(&replacements);
        self.current_cache = cache.buffer().clone();
    }

    /// CPU 写 K+1 行 embedding(f32 → F16)。
    pub fn write_input(&self, embedding: &[f32]) -> Result<(), BackendError> {
        let expected = self.rows * self.input.cols;
        if embedding.len() != expected {
            return Err(compute_error(format!("verify 重放输入 {} 期望 {expected}", embedding.len())));
        }
        let packed: Vec<u16> = embedding.iter().map(|&value| f16::from_f32(value).to_bits()).collect();
        let destination = unsafe { std::slice::from_raw_parts_mut(self.input.buffer.contents() as *mut u8, packed.len() * 2) };
        destination.copy_from_slice(unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) });
        Ok(())
    }

    pub fn write_token_inputs(&self, values: &[f32]) -> Result<(), BackendError> {
        let Some(token_inputs) = &self.token_inputs else {
            if values.is_empty() {
                return Ok(());
            }
            return Err(compute_error("verify 重放没有 per-layer input buffer"));
        };
        if values.len() != token_inputs.len() {
            return Err(compute_error(format!("verify per-layer input {} 期望 {}", values.len(), token_inputs.len())));
        }
        let packed: Vec<u16> = values.iter().map(|&value| f16::from_f32(value).to_bits()).collect();
        let destination = unsafe { std::slice::from_raw_parts_mut(token_inputs.buffer.contents().cast::<u8>(), packed.len() * 2) };
        destination.copy_from_slice(unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) });
        Ok(())
    }

    /// 一步 verify:推进 KV 游标 rows 行、写 state 槽、重编码提交、wait、
    /// 返回 (每行 argmax, normed hidden 的 CPU 行集)。
    #[allow(clippy::too_many_arguments)]
    pub fn step(&self, ctx: &MetalContext, cache: &mut MetalKvCache, position: usize) -> Result<(Vec<u32>, Vec<f32>), BackendError> {
        {
            let state = cache.hybrid_gqa_state_mut().ok_or_else(|| compute_error("verify 重放需要 hybrid cache"))?;
            for layer in 0..self.layer_count {
                if self.kv_layers[layer] {
                    let plan = state.plan_append(layer, position, self.rows).map_err(compute_error)?;
                    state.commit_append(&plan).map_err(compute_error)?;
                }
            }
        }
        let cache_state = cache.hybrid_gqa_state().ok_or_else(|| compute_error("verify 重放需要 hybrid cache"))?;
        let slots = unsafe { std::slice::from_raw_parts_mut(self.state.contents().cast::<u32>(), self.layer_count * 3) };
        for layer in 0..self.layer_count {
            let read_layer = self.kv_source[layer].unwrap_or(layer);
            let retained = cache_state.retained_range(read_layer).map_err(compute_error)?;
            let slot = &mut slots[layer * 3..layer * 3 + 3];
            slot[0] = position as u32;
            slot[1] = retained.end as u32;
            slot[2] = retained.start as u32;
        }
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
            eprintln!("[mtp-gpu] verify step rows={} gpu={:.3}ms", self.rows, (command.gpu_end_time() - command.gpu_start_time()) * 1.0e3);
        }
        let tokens = unsafe { std::slice::from_raw_parts(self.readback.contents().cast::<u32>(), self.rows) }.to_vec();
        let normed = ctx.tensor_to_f32(&self.normed);
        Ok((tokens, normed))
    }
}

/// verify 重放的单层转录序列(多行 position 原语版)。
#[allow(clippy::too_many_arguments)]
fn record_verify_layer(
    ctx: &MetalContext,
    cache: &MetalKvCache,
    layer: usize,
    attention: HybridGqaLayerSpec,
    kv_source: Option<usize>,
    config: &crate::runtime::gemma4::Gemma4Config,
    weights: &Gemma4Layer<MetalWeight>,
    hidden: &MetalTensor,
    per_layer_input: Option<&MetalTensor>,
    rope: &RopeF16Tables,
    state: &Buffer,
    state_offset: u64,
) -> Result<MetalTensor, BackendError> {
    let cfg_rope = |tensor: &MetalTensor, heads: usize| -> Result<MetalTensor, BackendError> {
        crate::kernel::metal::shape::apply_rope_rows_position_tensor(ctx, tensor, heads, rope.rotary_dim, crate::attention::rope::RotaryLayout::SplitHalf, &rope.cos, &rope.sin, rope.max_positions, state, state_offset).map_err(compute_error)
    };
    let geometry = attention.geometry;
    let attention_output = if let Some(source_layer) = kv_source {
        let normed = ctx.rmsnorm(hidden, &weights.input_norm, config.rms_eps)?;
        let query = ctx.linear(&normed, &weights.attention.query)?;
        let query = ctx.gemma_rmsnorm_heads(&query, &weights.attention.query_norm, geometry.num_heads, geometry.head_dim, config.rms_eps)?;
        let query = cfg_rope(&query, geometry.num_heads)?;
        let view = cache.gqa_layer_view(source_layer).map_err(compute_error)?;
        crate::kernel::metal::attention::gqa_decode_attention_headmajor_position_tensor(ctx, &query, &view, &gqa_spec(attention), state, state_offset).map_err(compute_error)?
    } else {
        let key_weight = weights.attention.key.as_ref().ok_or_else(|| compute_error(format!("verify 重放 L{layer} 缺少 K 权重")))?;
        let normed = ctx.rmsnorm(hidden, &weights.input_norm, config.rms_eps)?;
        let (query, key_source, value_source) = match &weights.attention.value {
            Some(value) => {
                let (q, k, v) = ctx.triple_linear(&normed, &weights.attention.query, key_weight, value)?;
                (q, k, Some(v))
            }
            None => {
                let (q, k) = ctx.dual_linear(&normed, &weights.attention.query, key_weight)?;
                (q, k, None)
            }
        };
        let query = ctx.gemma_rmsnorm_heads(&query, &weights.attention.query_norm, geometry.num_heads, geometry.head_dim, config.rms_eps)?;
        let key = ctx.gemma_rmsnorm_heads(&key_source, &weights.attention.key_norm, geometry.num_kv_heads, geometry.head_dim, config.rms_eps)?;
        let value_source = value_source.as_ref().unwrap_or(&key_source);
        let value = ctx.gemma_rmsnorm_heads(value_source, &weights.attention.value_norm, geometry.num_kv_heads, geometry.head_dim, config.rms_eps)?;
        let query = cfg_rope(&query, geometry.num_heads)?;
        let key = cfg_rope(&key, geometry.num_kv_heads)?;
        let view = cache.gqa_layer_view(layer).map_err(compute_error)?;
        crate::kernel::metal::attention::gqa_kv_append_rows_position_tensor(ctx, &view, &key, &value, state, state_offset).map_err(compute_error)?;
        crate::kernel::metal::attention::gqa_decode_attention_headmajor_position_tensor(ctx, &query, &view, &gqa_spec(attention), state, state_offset).map_err(compute_error)?
    };
    let attention_output = ctx.linear(&attention_output, &weights.attention.output)?;
    let hidden = ctx.rmsnorm_add_scaled(hidden, &attention_output, &weights.post_attention_norm, config.rms_eps, 1.0)?;
    let mlp_input = ctx.rmsnorm(&hidden, &weights.pre_feedforward_norm, config.rms_eps)?;
    let mlp = crate::moe::dense_mlp::forward_observed(
        ctx,
        &DenseMlpSpec { intermediate_size: config.intermediate_size, activation: Activation::GeluTanh },
        DenseMlpWeightsRef { gate: &weights.mlp.gate, up: &weights.mlp.up, down: &weights.mlp.down },
        &mlp_input,
        |_| {},
    )?;
    match (&weights.per_layer_input, per_layer_input) {
        (None, None) => ctx.rmsnorm_add_scaled(&hidden, &mlp, &weights.post_feedforward_norm, config.rms_eps, weights.layer_scalar),
        (Some(per_layer_weights), Some(per_layer_input)) => {
            let hidden = ctx.rmsnorm_add_scaled(&hidden, &mlp, &weights.post_feedforward_norm, config.rms_eps, 1.0)?;
            let gated = ctx.linear_gated_activation(&hidden, &per_layer_weights.gate, per_layer_input, &Activation::GeluTanh)?;
            let projected = ctx.linear(&gated, &per_layer_weights.projection)?;
            ctx.rmsnorm_add_scaled(&hidden, &projected, &per_layer_weights.post_norm, config.rms_eps, weights.layer_scalar)
        }
        _ => Err(compute_error("Gemma4 verify layer weights 与 per-layer input 不匹配")),
    }
}
