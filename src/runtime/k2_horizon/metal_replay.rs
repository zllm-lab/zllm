//! K2-Horizon × Metal 的单 token decode 平铺重放。

use super::{K2HorizonConfig, K2HorizonGguf};
use crate::{
    attention::rope::{RopeTable, RotaryLayout},
    backend::metal::{MetalContext, MetalKvCache, MetalMoeDecodeState, MetalTensor, MetalWeight, api::Buffer, replay::ReplayPlan},
    backend::{Backend, BackendError},
    kernel::metal::{attention as metal_attention, shape as metal_shape},
    moe::topk_moe::{MoeFfnRef, RoutedMoeInputs, SharedExpertRef},
    runtime::{
        expert_pipeline::{ExpertDecodePipeline, ExpertDecodeRequest},
        k2_horizon::metal::{K2Layer, K2Mlp, K2OutputHead, K2Value},
        output::{OutputNorm, OutputPlan},
    },
    weight::expert_source::ExpertSourceProvider,
};
use half::f16;

fn compute_error(message: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: message.into() }
}

struct RopeF16Tables {
    cos: Buffer,
    sin: Buffer,
    rotary_dim: usize,
    max_positions: usize,
}

fn upload_rope_table(ctx: &MetalContext, table: &RopeTable) -> Result<RopeF16Tables, BackendError> {
    let half_dim = table.rotary_dim / 2;
    let needed = table.seq_len.checked_mul(half_dim).ok_or_else(|| compute_error("K2-Horizon replay RoPE 表长度溢出"))?;
    if table.cos.len() < needed || table.sin.len() < needed {
        return Err(compute_error("K2-Horizon replay RoPE cos/sin 长度不足"));
    }
    let upload = |values: &[f32]| -> Buffer {
        let packed: Vec<u16> = values[..needed].iter().map(|&value| f16::from_f32(value).to_bits()).collect();
        let bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 2) };
        ctx.shared_buffer(bytes)
    };
    Ok(RopeF16Tables { cos: upload(&table.cos), sin: upload(&table.sin), rotary_dim: table.rotary_dim, max_positions: table.seq_len })
}

pub struct K2DecodeReplay {
    commands: ReplayPlan,
    q8_pipelines: Option<metal_attention::Q8PositionReplayPipelines>,
    decode_state: Buffer,
    input: MetalTensor,
    readback: Buffer,
    current_cache: Buffer,
    layer_count: usize,
}

impl K2DecodeReplay {
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        ctx: &MetalContext,
        cache: &MetalKvCache,
        config: &K2HorizonConfig,
        layers: &[K2Layer<MetalWeight>],
        rope: &RopeTable,
        output_head: &K2OutputHead,
        source: &K2HorizonGguf,
        experts: &mut ExpertDecodePipeline<MetalMoeDecodeState>,
        position: usize,
    ) -> Result<Self, BackendError> {
        if layers.len() != config.layer_count {
            return Err(compute_error(format!("K2-Horizon replay layers={}，期望 {}", layers.len(), config.layer_count)));
        }
        let input = ctx.tensor_zeros_f32(1, config.hidden_size);
        let decode_state = ctx.shared_buffer_zeros(config.layer_count.checked_mul(12).ok_or_else(|| compute_error("K2-Horizon replay state 大小溢出"))?);
        let readback = ctx.shared_buffer_zeros(std::mem::size_of::<u32>());
        let rope = upload_rope_table(ctx, rope)?;
        let (commands, ()) = ReplayPlan::record(|| record_round(ctx, cache, config, layers, &rope, output_head, source, experts, &input, &decode_state, &readback, position))?;
        let q8_pipelines = (cache.format() == crate::kv_cache::KvCacheFormat::Int8).then(|| metal_attention::Q8PositionReplayPipelines::new(ctx).map_err(compute_error)).transpose()?;
        Ok(Self { commands, q8_pipelines, decode_state, input, readback, current_cache: cache.buffer().clone(), layer_count: config.layer_count })
    }

    pub fn command_count(&self) -> usize {
        self.commands.command_count()
    }

    pub fn bind_cache(&mut self, cache: &MetalKvCache) {
        if !self.current_cache.same_handle(cache.buffer()) {
            self.commands.remap_buffers(&[(self.current_cache.clone(), cache.buffer().clone())]);
            self.current_cache = cache.buffer().clone();
        }
    }

    pub fn step(&mut self, ctx: &MetalContext, cache: &mut MetalKvCache, embedding: &[f32], position: usize) -> Result<u32, BackendError> {
        if embedding.len() != self.input.cols {
            return Err(compute_error(format!("K2-Horizon replay embedding={}，期望 {}", embedding.len(), self.input.cols)));
        }
        let destination = unsafe { std::slice::from_raw_parts_mut(self.input.buffer.contents().cast::<f32>(), embedding.len()) };
        destination.copy_from_slice(embedding);
        let slots = unsafe { std::slice::from_raw_parts_mut(self.decode_state.contents().cast::<u32>(), self.layer_count * 3) };
        for layer in 0..self.layer_count {
            cache.reserve_layer_gqa_row(layer, position).map_err(compute_error)?;
            let slot = &mut slots[layer * 3..layer * 3 + 3];
            slot[0] = position as u32;
            slot[1] = (position + 1) as u32;
            slot[2] = 0;
        }
        let submit_started = std::time::Instant::now();
        let command = if let Some(pipelines) = &self.q8_pipelines {
            let split = position + 1 > 256;
            self.commands.submit_filtered(ctx, &|op| pipelines.accepts(op, split))
        } else {
            self.commands.submit(ctx)
        };
        let submit_seconds = submit_started.elapsed().as_secs_f64();
        command.wait_until_completed();
        if std::env::var_os("ZLLM_K2_REPLAY_PROFILE").is_some() {
            static PROFILE_STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let step = PROFILE_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if step < 4 || matches!(step, 63 | 127 | 191 | 254) {
                let gpu_seconds = (command.gpu_end_time() - command.gpu_start_time()).max(0.0);
                eprintln!("[k2-horizon-replay-step] step={step} submit={:.3}ms gpu={:.3}ms wall={:.3}ms", submit_seconds * 1.0e3, gpu_seconds * 1.0e3, submit_started.elapsed().as_secs_f64() * 1.0e3);
            }
        }
        Ok(unsafe { *self.readback.contents().cast::<u32>() })
    }
}

#[allow(clippy::too_many_arguments)]
fn record_round(
    ctx: &MetalContext,
    cache: &MetalKvCache,
    config: &K2HorizonConfig,
    layers: &[K2Layer<MetalWeight>],
    rope: &RopeF16Tables,
    output_head: &K2OutputHead,
    source: &K2HorizonGguf,
    experts: &mut ExpertDecodePipeline<MetalMoeDecodeState>,
    input: &MetalTensor,
    decode_state: &Buffer,
    readback: &Buffer,
    position: usize,
) -> Result<(), BackendError> {
    let mut hidden = input.clone();
    let mut normed = ctx.grouped_rmsnorm(&hidden, &layers[0].input_norm, config.rms_eps, config.norm_groups)?;
    for (layer, weights) in layers.iter().enumerate() {
        let attention = attention_projection(ctx, cache, config, layer, weights, &normed, rope, decode_state)?;
        let (residual, input) = add_grouped_norm(ctx, config, &hidden, &attention, &weights.ffn_norm)?;
        let output = match &weights.mlp {
            K2Mlp::Dense { gate, up, down } => {
                let activated = ctx.gated_linear(&input, gate, up, &crate::moe::Activation::Silu)?;
                ctx.linear(&activated, down)?
            }
            K2Mlp::Sparse { router, bias, shared_gate, shared_up, shared_down } => {
                let shared = [SharedExpertRef { gate: shared_gate, up: shared_up, down: shared_down, output_gate: None }];
                let expert_source = source.source(layer).map_err(BackendError::ExpertLoad)?;
                let next = (layer + 1 < config.layer_count).then(|| source.source(layer + 1).map(|next| (layer + 1, next))).transpose().map_err(BackendError::ExpertLoad)?;
                experts.decode_inputs(
                    ctx,
                    &config.moe_spec(),
                    &MoeFfnRef { router_weight: router, router_bias: bias, shared_experts: &shared, selected_experts: None },
                    ExpertDecodeRequest { layer, source: expert_source, position, next },
                    RoutedMoeInputs { route: &input, expert: &input },
                )?
            }
        };
        if let Some(next_weights) = layers.get(layer + 1) {
            (hidden, normed) = add_grouped_norm(ctx, config, &residual, &output, &next_weights.input_norm)?;
        } else {
            hidden = ctx.add(&residual, &output)?;
        }
    }
    let plan = OutputPlan { eps: config.rms_eps, norm: OutputNorm::GroupedRms { groups: config.norm_groups }, excluded_tokens: Vec::new() };
    let (_, logits) = crate::runtime::output::norm_and_lm_head(ctx, output_head, &hidden, &plan)?;
    crate::kernel::metal::moe::argmax_tensor_into(ctx, &logits, &[], readback).map_err(compute_error)
}

fn attention_projection(
    ctx: &MetalContext,
    cache: &MetalKvCache,
    config: &K2HorizonConfig,
    layer: usize,
    weights: &K2Layer<MetalWeight>,
    normed: &MetalTensor,
    rope: &RopeF16Tables,
    decode_state: &Buffer,
) -> Result<MetalTensor, BackendError> {
    let state_offset = (layer * 12) as u64;
    let (query, gate) = ctx.dual_linear(normed, &weights.query, &weights.attention_gate)?;
    let key = ctx.linear(normed, &weights.key)?;
    let value = match &weights.value {
        K2Value::Dense(value) => ctx.linear(normed, value)?,
        K2Value::Routed { router, bias, experts } => routed_value(ctx, config, normed, router, bias, experts)?,
    };
    let query = metal_shape::apply_rope_position_tensor(ctx, &query, config.num_heads, rope.rotary_dim, RotaryLayout::SplitHalf, &rope.cos, &rope.sin, rope.max_positions, decode_state, state_offset).map_err(compute_error)?;
    let key = metal_shape::apply_rope_position_tensor(ctx, &key, config.num_kv_heads, rope.rotary_dim, RotaryLayout::SplitHalf, &rope.cos, &rope.sin, rope.max_positions, decode_state, state_offset).map_err(compute_error)?;
    let view = cache.gqa_layer_view(layer).map_err(compute_error)?;
    let attended = match cache.format() {
        crate::kv_cache::KvCacheFormat::F16 => {
            metal_attention::gqa_kv_append_position_tensor(ctx, &view, &key, &value, decode_state, state_offset).map_err(compute_error)?;
            metal_attention::gqa_decode_attention_position_tensor(ctx, &query, &view, &config.gqa_spec(), decode_state, state_offset).map_err(compute_error)?
        }
        crate::kv_cache::KvCacheFormat::Int8 => metal_attention::gqa_decode_attention_append_adaptive_q8_position_tensor(ctx, &query, &key, &value, &view, &config.gqa_spec(), decode_state, state_offset).map_err(compute_error)?,
    };
    let gated = crate::kernel::metal::tensor::softplus_gate_scaled_tensor(ctx, &attended, &gate, std::f32::consts::LN_2, std::f32::consts::LOG2_E).map_err(compute_error)?;
    ctx.linear(&gated, &weights.output)
}

fn routed_value(ctx: &MetalContext, config: &K2HorizonConfig, input: &MetalTensor, router: &MetalWeight, bias: &MetalWeight, experts: &MetalWeight) -> Result<MetalTensor, BackendError> {
    let (MetalWeight::F32 { buffer: router, len: router_len }, MetalWeight::F32 { buffer: bias, len: bias_len }) = (router, bias) else {
        return Err(compute_error("K2-Horizon replay MoVA router 必须是 F32 resident 权重"));
    };
    let (expert_ids, route_weights) =
        crate::kernel::metal::moe::moe_router_sigmoid_decode_resident_f32(ctx, input, router, *router_len, bias, *bias_len, config.value_expert_count, config.value_expert_top_k, config.routed_scaling_factor).map_err(compute_error)?;
    crate::kernel::metal::gguf::gguf_routed_value_iq3s_tensor_resident(ctx, input, experts, &expert_ids, &route_weights, config.value_expert_count, config.value_expert_top_k, config.num_kv_heads * config.head_dim).map_err(compute_error)
}

fn add_grouped_norm(ctx: &MetalContext, config: &K2HorizonConfig, residual: &MetalTensor, projected: &MetalTensor, weight: &MetalWeight) -> Result<(MetalTensor, MetalTensor), BackendError> {
    let MetalWeight::F32 { buffer, len } = weight else {
        return Err(compute_error("K2-Horizon replay grouped RMSNorm 权重必须是 F32 resident"));
    };
    crate::kernel::metal::tensor::add_f32_f16_grouped_rmsnorm_tensor_resident(ctx, residual, projected, buffer, *len, config.rms_eps, config.norm_groups).map_err(compute_error)
}
