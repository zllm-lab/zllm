//! H3 与 ROCm Ulysses 的设备组合。

use std::collections::HashSet;

use crate::backend::{
    Backend, BackendError, BackendResources, DiffusionBackend, SegmentedTensorBackend,
    rocm::{RocmContext, RocmTensor, RocmWeight},
};
use crate::diffusion::ModulationSegment;
use crate::weight::model::h3::{H3AttentionWeights, H3DitBlockWeights};

use super::{
    H3ConditionInput, H3Config, H3DenoiseLatents, H3PackedLayout, H3PreparedGlobal, H3UlyssesPlan, H3VelocityOutput, block_adaln, build_modulation_plan, dit_final_layer, h3_euler_step, h3_sigma_schedule, h3_time_state, mlp, mm_rope_tables,
    pack_dit_inputs_with_prefix, prepare_dit_static_prefix_conditioned, time_embedding,
};

/// 同一 H3 请求使用的 ROCm rank 组。
pub struct H3RocmUlyssesGroup {
    contexts: Vec<RocmContext>,
}

/// First-block cache 的数值策略。缓存只活在一次 denoise 请求内，不能跨请求复用。
#[derive(Clone, Copy, Debug)]
pub struct H3BlockCacheSpec {
    pub threshold: f32,
    pub start_percent: f32,
    pub end_percent: f32,
    pub max_consecutive_hits: usize,
}

impl H3BlockCacheSpec {
    pub fn validate(self) -> Result<Self, String> {
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            return Err(format!("H3 block cache threshold={} 必须是非负有限数", self.threshold));
        }
        if !self.start_percent.is_finite() || !self.end_percent.is_finite() || self.start_percent < 0.0 || self.start_percent >= self.end_percent || self.end_percent > 1.0 {
            return Err(format!("H3 block cache window=[{},{}] 必须满足 0 <= start < end <= 1", self.start_percent, self.end_percent));
        }
        if self.max_consecutive_hits == 0 {
            return Err("H3 block cache max_consecutive_hits 必须 >= 1".to_owned());
        }
        Ok(self)
    }
}

impl H3RocmUlyssesGroup {
    pub fn new(devices: &[i32], allow_cpu_reference_fallback: bool) -> Result<Self, String> {
        if !matches!(devices.len(), 1 | 2 | 4 | 8) {
            return Err(format!("H3 ROCm Ulysses devices={} 不受支持，必须是 1/2/4/8", devices.len()));
        }
        if devices.iter().copied().collect::<HashSet<_>>().len() != devices.len() {
            return Err("H3 ROCm Ulysses devices 不能重复".to_owned());
        }
        let contexts = devices
            .iter()
            .map(|&device| RocmContext::configured(device, allow_cpu_reference_fallback).and_then(|context| context.with_independent_stream()).map_err(|error| format!("初始化 H3 ROCm rank device {device}: {error}")))
            .collect::<Result<Vec<_>, _>>()?;
        for destination in &contexts {
            for source in &contexts {
                if destination.device_id() != source.device_id() {
                    destination.enable_peer_access_from(source.device_id()).map_err(|error| format!("H3 ROCm P2P {} <- {} 不可用: {error}", destination.device_id(), source.device_id()))?;
                }
            }
        }
        Ok(Self { contexts })
    }

    pub fn contexts(&self) -> &[RocmContext] {
        &self.contexts
    }

    pub fn primary(&self) -> &RocmContext {
        &self.contexts[0]
    }

    pub fn plan(&self, layout: &H3PackedLayout, config: &H3Config) -> Result<H3UlyssesPlan, String> {
        H3UlyssesPlan::new(layout.seq_len, config.num_attention_heads, self.contexts.len())
    }

    /// ordered P2P 依赖当前提交线程能看到源、目标两端的 compute stream。
    /// 每个请求线程只需注册一次；重复激活不会创建新 stream。
    pub(crate) fn activate_all(&self) -> Result<(), BackendError> {
        for context in &self.contexts {
            context.activate().map_err(|msg| BackendError::Compute { msg })?;
        }
        Ok(())
    }

    /// 严格基线按逻辑 phase 等待全部 rank；确认可重复正确后，再逐段替换为 event。
    fn synchronize_all(&self) -> Result<(), BackendError> {
        for context in &self.contexts {
            context.synchronize_compute_stream()?;
        }
        // 所有目标 stream 都已完成 ordered peer-copy，此时才能同时释放各目标
        // completion 持有的源 Arc；逐卡同步后立刻释放会让后续目标仍读已回收地址。
        for context in &self.contexts {
            context.retire_ordered_p2p_sources();
        }
        Ok(())
    }

    /// 主卡完整 sequence resident tensor 切分到各 rank；当前实现是正确性底座，
    /// 后续以 ring scatter 和计算重叠替换同步 P2P。
    pub fn scatter_sequence(&self, tensor: &RocmTensor, plan: &H3UlyssesPlan) -> Result<Vec<RocmTensor>, BackendError> {
        self.activate_all()?;
        if plan.shards.len() != self.contexts.len() || tensor.rows != plan.sequence_len {
            return Err(BackendError::Compute { msg: format!("H3 Ulysses scatter tensor rows={} plan rows={} shards={} ranks={}", tensor.rows, plan.sequence_len, plan.shards.len(), self.contexts.len()) });
        }
        let shards = plan
            .shards
            .iter()
            .zip(&self.contexts)
            .map(|(shard, context)| {
                let local = self.primary().slice_token_rows(tensor, shard.sequence.start, shard.sequence.len())?;
                let local = self.primary().tensor_to_stable_deferred(local)?;
                self.primary().synchronize()?;
                let shard = context.tensor_on_device_ordered(local)?;
                Ok(shard)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.synchronize_all()?;
        Ok(shards)
    }

    /// 把各 rank 的连续 sequence shard 按原顺序聚合回主卡。
    pub fn gather_sequence(&self, shards: &[RocmTensor], plan: &H3UlyssesPlan) -> Result<RocmTensor, BackendError> {
        self.activate_all()?;
        if shards.len() != self.contexts.len() || plan.shards.len() != self.contexts.len() {
            return Err(BackendError::Compute { msg: format!("H3 Ulysses gather shards={} plan={} ranks={}", shards.len(), plan.shards.len(), self.contexts.len()) });
        }
        let mut primary_shards = Vec::with_capacity(shards.len());
        for ((tensor, expected), context) in shards.iter().zip(&plan.shards).zip(&self.contexts) {
            if tensor.rows != expected.sequence.len() || tensor.device.as_deref().is_none_or(|buffer| buffer.device_id() != context.device_id()) {
                return Err(BackendError::Compute { msg: format!("H3 Ulysses rank device={} shard=[{},{}]，期望 rows={}", context.device_id(), tensor.rows, tensor.cols, expected.sequence.len()) });
            }
            let tensor = context.tensor_to_stable_deferred(tensor.clone())?;
            context.synchronize()?;
            primary_shards.push(self.primary().tensor_on_device_ordered(tensor)?);
        }
        self.synchronize_all()?;
        let refs = primary_shards.iter().collect::<Vec<_>>();
        let output = self.primary().concat_token_rows(&refs)?;
        self.synchronize_all()?;
        Ok(output)
    }

    /// Ulysses attention：本地 sequence shard 投影 QKV，第一次交换得到完整
    /// sequence 的 head shard；attention 后第二次交换回本地 sequence shard。
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, inputs: &[RocmTensor], weights: &[&H3AttentionWeights<RocmWeight>], config: &H3Config, plan: &H3UlyssesPlan, cosine: &[f32], sine: &[f32]) -> Result<Vec<RocmTensor>, BackendError> {
        self.activate_all()?;
        if inputs.len() != self.contexts.len() || weights.len() != self.contexts.len() || plan.shards.len() != self.contexts.len() {
            return Err(BackendError::Compute { msg: format!("H3 Ulysses attention inputs={} weights={} shards={} ranks={}", inputs.len(), weights.len(), plan.shards.len(), self.contexts.len()) });
        }
        let mut qkv = Vec::with_capacity(inputs.len());
        for (((context, input), weights), shard) in self.contexts.iter().zip(inputs).zip(weights).zip(&plan.shards) {
            if input.rows != shard.sequence.len() || input.cols != config.hidden_size {
                return Err(BackendError::Compute { msg: format!("H3 Ulysses attention device={} input=[{},{}]，期望 [{},{}]", context.device_id(), input.rows, input.cols, shard.sequence.len(), config.hidden_size) });
            }
            qkv.push(context.linear(input, &weights.qkv)?);
        }

        // 每个 source rank 先一次性产出所有目标 head shard。ordered P2P 会在
        // source stream 的 compact/stable-copy 之后录 event，目标 stream 等待该
        // event；这里不需要额外的 host/global barrier。
        let compact_qkv = self
            .contexts
            .iter()
            .enumerate()
            .map(|(source_rank, source)| {
                let shards = plan
                    .shards
                    .iter()
                    .map(|head_shard| {
                        let compact = source.compact_qkv_heads(&qkv[source_rank], config.num_attention_heads, head_shard.heads.clone(), config.attention_head_dim)?;
                        source.tensor_to_stable_deferred(compact)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(shards)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let head_rows = self
            .contexts
            .iter()
            .enumerate()
            .map(|(head_rank, destination)| {
                let rows = compact_qkv.iter().map(|source_shards| destination.tensor_on_device_ordered(source_shards[head_rank].clone())).collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.synchronize_all()?;
        let mut head_outputs = Vec::with_capacity(self.contexts.len());
        for (head_rank, ((destination, weights), rows)) in self.contexts.iter().zip(weights).zip(head_rows).enumerate() {
            let refs = rows.iter().collect::<Vec<_>>();
            let qkv = destination.concat_token_rows(&refs)?;
            let output = destination.full_attention_qkv(
                qkv,
                &weights.q_norm,
                &weights.k_norm,
                plan.shards[head_rank].heads.len(),
                config.attention_head_dim,
                config.rope_inv_freq_len * 6,
                config.qk_norm_eps,
                cosine,
                sine,
                (config.attention_head_dim as f32).sqrt().recip(),
            )?;
            head_outputs.push(destination.tensor_to_stable_deferred(output)?);
        }

        let source_columns = plan
            .shards
            .iter()
            .map(|shard| {
                let sequence = &shard.sequence;
                self.contexts
                    .iter()
                    .enumerate()
                    .map(|(head_rank, source)| {
                        let local = source.slice_token_rows(&head_outputs[head_rank], sequence.start, sequence.len())?;
                        let local = source.tensor_to_stable_deferred(local)?;
                        Ok(local)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let destination_columns = self
            .contexts
            .iter()
            .zip(source_columns)
            .map(|(destination, columns)| {
                let columns = columns.into_iter().map(|column| destination.tensor_on_device_ordered(column)).collect::<Result<Vec<_>, _>>()?;
                Ok(columns)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.synchronize_all()?;
        let mut outputs = Vec::with_capacity(self.contexts.len());
        for ((destination, weights), columns) in self.contexts.iter().zip(weights).zip(destination_columns) {
            let mut columns = columns.into_iter();
            let mut attended = columns.next().expect("H3 Ulysses rank 非空");
            for column in columns {
                attended = destination.concat_columns(&attended, &column)?;
            }
            outputs.push(destination.linear(&attended, &weights.output)?);
        }
        Ok(outputs)
    }

    /// 单个 H3 DiT block 的 sequence-parallel 执行。权重在每个 rank 完整常驻，
    /// 只有 self-attention 做两次 sequence/head 交换，其余算子保持 rank-local。
    #[allow(clippy::too_many_arguments)]
    pub fn dit_block(
        &self,
        inputs: &[RocmTensor],
        time_embeddings: &[RocmTensor],
        segments: &[ModulationSegment],
        weights: &[&H3DitBlockWeights<RocmWeight>],
        config: &H3Config,
        plan: &H3UlyssesPlan,
        cosine: &[f32],
        sine: &[f32],
    ) -> Result<Vec<RocmTensor>, BackendError> {
        let ranks = self.contexts.len();
        if inputs.len() != ranks || time_embeddings.len() != ranks || weights.len() != ranks || plan.shards.len() != ranks {
            return Err(BackendError::Compute { msg: format!("H3 Ulysses block inputs={} time={} weights={} shards={} ranks={ranks}", inputs.len(), time_embeddings.len(), weights.len(), plan.shards.len()) });
        }
        let local_segments = (0..ranks).map(|rank| plan.local_modulation_segments(rank, segments).map_err(|msg| BackendError::Compute { msg })).collect::<Result<Vec<_>, _>>()?;
        let modulation = self
            .contexts
            .iter()
            .zip(time_embeddings)
            .zip(weights)
            .map(|((context, time), weights)| {
                let modulation = block_adaln(context, config, time, weights)?;
                Ok(modulation)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let normalized = self
            .contexts
            .iter()
            .zip(inputs)
            .zip(weights)
            .zip(&modulation)
            .zip(&local_segments)
            .map(|((((context, hidden), weights), modulation), segments)| {
                let normalized = context.rmsnorm_adaln_modulate_segmented(hidden, &weights.norm1, config.norm_eps, &modulation.shift_msa, &modulation.scale_msa, segments)?;
                Ok(normalized)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let attention_weights = weights.iter().map(|weights| &weights.attention).collect::<Vec<_>>();
        let updates = self.attention(&normalized, &attention_weights, config, plan, cosine, sine)?;

        let hidden = self
            .contexts
            .iter()
            .zip(inputs)
            .zip(&updates)
            .zip(&modulation)
            .zip(&local_segments)
            .map(|((((context, residual), update), modulation), segments)| {
                let hidden = context.gated_residual_segmented(residual, update, &modulation.gate_msa, segments)?;
                Ok(hidden)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let normalized = self
            .contexts
            .iter()
            .zip(&hidden)
            .zip(weights)
            .zip(&modulation)
            .zip(&local_segments)
            .map(|((((context, hidden), weights), modulation), segments)| {
                let normalized = context.rmsnorm_adaln_modulate_segmented(hidden, &weights.norm2, config.norm_eps, &modulation.shift_mlp, &modulation.scale_mlp, segments)?;
                Ok(normalized)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mlp_updates = self
            .contexts
            .iter()
            .zip(&normalized)
            .zip(weights)
            .map(|((context, normalized), weights)| {
                let update = mlp(context, config, normalized, &weights.mlp)?;
                Ok(update)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = self
            .contexts
            .iter()
            .zip(&hidden)
            .zip(&mlp_updates)
            .zip(&modulation)
            .zip(&local_segments)
            .map(|((((context, hidden), update), modulation), segments)| {
                let output = context.gated_residual_segmented(hidden, update, &modulation.gate_mlp, segments)?;
                Ok(output)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(outputs)
    }
}

/// 完整 H3 Ulysses 采样。conditioning、prefix、final layer 与 latent 更新留在
/// 主卡；50 个 DiT block 在所有 rank 上按 sequence shard 执行。
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn denoise_ulysses_conditioned(
    group: &H3RocmUlyssesGroup,
    source: &crate::weight::model::h3::H3DitSource,
    globals: &[H3PreparedGlobal<RocmWeight>],
    blocks: &[Vec<H3DitBlockWeights<RocmWeight>>],
    final_weights: &crate::weight::model::h3::H3DitFinalWeights<RocmWeight>,
    text: &RocmTensor,
    conditions: &[H3ConditionInput<'_, RocmTensor>],
    mut latents: H3DenoiseLatents<RocmTensor>,
    layout: &H3PackedLayout,
    num_inference_steps: usize,
    start_step: usize,
    block_cache: Option<H3BlockCacheSpec>,
    on_step: &mut dyn FnMut(usize, usize, &H3DenoiseLatents<RocmTensor>) -> Result<bool, BackendError>,
) -> Result<H3DenoiseLatents<RocmTensor>, BackendError> {
    let ranks = group.contexts.len();
    let config = source.config();
    if globals.len() != ranks || blocks.len() != ranks || blocks.iter().any(|layers| layers.len() != config.num_layers) {
        return Err(BackendError::Compute {
            msg: format!("H3 Ulysses resident weights globals={} block_ranks={} layers={:?}，期望 ranks={ranks} layers={}", globals.len(), blocks.len(), blocks.iter().map(Vec::len).collect::<Vec<_>>(), config.num_layers),
        });
    }
    // conditioning 与 resident 权重准备发生在 Ulysses stream 激活之前；独立
    // stream 带 NON_BLOCKING 标志，不能依赖 legacy default stream 的隐式顺序。
    // 请求边界先清空准备期默认流；当前正确性基线在每个 Ulysses phase 使用
    // 严格 barrier，确认重复请求稳定后再逐段收紧为 ordered event DAG。
    for context in &group.contexts {
        context.synchronize()?;
    }
    group.activate_all()?;
    let primary = group.primary();
    let plan = group.plan(layout, config).map_err(|msg| BackendError::Compute { msg })?;
    let prefix = prepare_dit_static_prefix_conditioned(primary, source, &globals[0], text, conditions).map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses 准备静态 prefix: {error:?}") })?;
    let video_sigmas = h3_sigma_schedule(num_inference_steps, config.sigma_shift_video).map_err(|msg| BackendError::Compute { msg })?;
    let audio_sigmas = h3_sigma_schedule(num_inference_steps, config.sigma_shift_audio).map_err(|msg| BackendError::Compute { msg })?;
    if audio_sigmas.len() != video_sigmas.len() {
        return Err(BackendError::Compute { msg: format!("H3 video/audio sigma 网格长度不同：{}/{}", video_sigmas.len(), audio_sigmas.len()) });
    }
    let total = video_sigmas.len() - 1;
    if start_step > total {
        return Err(BackendError::Compute { msg: format!("H3 Ulysses resume step={start_step} 超过采样步数 {total}") });
    }
    let (cosine, sine) = mm_rope_tables(&layout.position_ids, &globals[0].rope_inv_freq).map_err(|msg| BackendError::Compute { msg })?;
    let block_cache = block_cache.map(H3BlockCacheSpec::validate).transpose().map_err(|msg| BackendError::Compute { msg })?;
    let mut previous_first_residual: Option<Vec<RocmTensor>> = None;
    let mut remaining_blocks_residual: Option<Vec<RocmTensor>> = None;
    let mut consecutive_hits = 0usize;
    let mut cached_steps = 0usize;
    let mut full_steps = 0usize;

    for index in start_step..total {
        let time = h3_time_state(video_sigmas[index], config).map_err(|msg| BackendError::Compute { msg })?;
        let modulation = build_modulation_plan(layout, time);
        let time_embeddings = group
            .contexts
            .iter()
            .zip(globals)
            .map(|(context, global)| time_embedding(context, config, &modulation.unique_timesteps, &global.weights))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} time embedding: {error:?}") })?;
        group.synchronize_all().map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} time embedding barrier: {error:?}") })?;
        let hidden = pack_dit_inputs_with_prefix(primary, source, &globals[0], &prefix, &latents.audio, &latents.video, layout).map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} pack DiT input: {error:?}") })?;
        primary.synchronize().map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} pack DiT input barrier: {error:?}") })?;
        let mut shards = group.scatter_sequence(&hidden, &plan).map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} scatter: {error:?}") })?;
        if let Some(cache) = block_cache {
            let first_input = shards.clone();
            let weights = blocks.iter().map(|rank| &rank[0]).collect::<Vec<_>>();
            let first_output = group.dit_block(&shards, &time_embeddings, &modulation.segments, &weights, config, &plan, &cosine, &sine).map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses block 0: {error:?}") })?;
            let progress = index as f32 / total as f32;
            let in_window = progress >= cache.start_percent && progress <= cache.end_percent;
            let diff = if in_window {
                previous_first_residual.as_ref().zip(remaining_blocks_residual.as_ref()).map(|(previous, _)| {
                    group
                        .contexts
                        .iter()
                        .zip(&first_output)
                        .zip(&first_input)
                        .zip(previous)
                        .try_fold((0.0_f64, 0.0_f64), |(numerator, denominator), (((context, output), input), previous)| {
                            let (local_numerator, local_denominator) = context.relative_l1_delta_partial(output, input, previous)?;
                            Ok::<_, BackendError>((numerator + local_numerator, denominator + local_denominator))
                        })
                        .map(|(numerator, denominator)| numerator / denominator.max(1.0e-8))
                })
            } else {
                None
            }
            .transpose()?;
            let use_cache = diff.is_some_and(|value| value.is_finite() && value <= f64::from(cache.threshold)) && consecutive_hits < cache.max_consecutive_hits;
            if use_cache {
                let tail = remaining_blocks_residual.as_ref().expect("H3 block cache 命中前已检查 tail");
                shards = group.contexts.iter().zip(&first_output).zip(tail).map(|((context, first), tail)| context.add(first, tail)).collect::<Result<Vec<_>, _>>()?;
                consecutive_hits += 1;
                cached_steps += 1;
                eprintln!("[h3-block-cache] step={}/{} hit diff={:.6} consecutive={}/{}", index + 1, total, diff.expect("命中必有 diff"), consecutive_hits, cache.max_consecutive_hits);
            } else {
                shards = first_output;
                let first_output = shards.clone();
                for layer in 1..config.num_layers {
                    let weights = blocks.iter().map(|rank| &rank[layer]).collect::<Vec<_>>();
                    shards = group.dit_block(&shards, &time_embeddings, &modulation.segments, &weights, config, &plan, &cosine, &sine).map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses block {layer}: {error:?}") })?;
                }
                remaining_blocks_residual = Some(group.contexts.iter().zip(&shards).zip(&first_output).map(|((context, output), first)| context.subtract_resident(output, first)).collect::<Result<Vec<_>, _>>()?);
                previous_first_residual = Some(group.contexts.iter().zip(&first_output).zip(&first_input).map(|((context, output), input)| context.subtract_resident(output, input)).collect::<Result<Vec<_>, _>>()?);
                consecutive_hits = 0;
                full_steps += 1;
                eprintln!("[h3-block-cache] step={}/{} full diff={}", index + 1, total, diff.map_or_else(|| "n/a".to_owned(), |value| format!("{value:.6}")));
            }
        } else {
            for layer in 0..config.num_layers {
                let weights = blocks.iter().map(|rank| &rank[layer]).collect::<Vec<_>>();
                shards = group.dit_block(&shards, &time_embeddings, &modulation.segments, &weights, config, &plan, &cosine, &sine).map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses block {layer}: {error:?}") })?;
            }
        }
        let hidden = group.gather_sequence(&shards, &plan).map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} gather: {error:?}") })?;
        let H3VelocityOutput { video, audio, .. } = dit_final_layer(primary, config, &hidden, &time_embeddings[0], &modulation.segments, layout.audio_rows.clone(), layout.video_rows.clone(), final_weights, time.audio_velocity_scale)
            .map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} final layer: {error:?}") })?;
        primary.synchronize().map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} final layer barrier: {error:?}") })?;
        latents.video = h3_euler_step(primary, &latents.video, &video, video_sigmas[index], video_sigmas[index + 1]).map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} video update: {error:?}") })?;
        latents.audio = h3_euler_step(primary, &latents.audio, &audio, audio_sigmas[index], audio_sigmas[index + 1]).map_err(|error| BackendError::Compute { msg: format!("H3 Ulysses step {index} audio update: {error:?}") })?;
        for context in &group.contexts {
            context.synchronize()?;
        }
        if !on_step(index + 1, total, &latents)? {
            return Err(BackendError::Compute { msg: format!("H3 Ulysses denoise 在完整 step {}/{} 后中断", index + 1, total) });
        }
    }
    if block_cache.is_some() {
        let executed_blocks = full_steps * config.num_layers + cached_steps;
        let nominal_blocks = (full_steps + cached_steps) * config.num_layers;
        eprintln!("[h3-block-cache] cached={cached_steps}/{} executed_blocks={executed_blocks}/{nominal_blocks} estimated_block_speedup={:.3}x", full_steps + cached_steps, nominal_blocks as f64 / executed_blocks.max(1) as f64);
    }
    Ok(latents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "需要显式设置 H3_ROCM_TEST_DEVICES 并占用对应 ROCm 卡"]
    fn ulysses_scatter_gather_round_trip() {
        let devices = std::env::var("H3_ROCM_TEST_DEVICES").expect("设置 H3_ROCM_TEST_DEVICES，例如 4,5,6,7").split(',').map(|value| value.parse::<i32>().expect("device id 必须是 i32")).collect::<Vec<_>>();
        let group = H3RocmUlyssesGroup::new(&devices, false).unwrap();
        let plan = H3UlyssesPlan::new(11, 56, devices.len()).unwrap();
        let values = (0..44).map(|value| value as f32 - 17.0).collect::<Vec<_>>();
        let input = group.primary().tensor_from_f32(values.clone(), 11, 4).unwrap();
        let shards = group.scatter_sequence(&input, &plan).unwrap();
        assert_eq!(shards.iter().map(|tensor| tensor.rows).collect::<Vec<_>>(), plan.shards.iter().map(|shard| shard.sequence.len()).collect::<Vec<_>>());
        let output = group.gather_sequence(&shards, &plan).unwrap();
        assert_eq!(group.primary().tensor_to_f32(&output).unwrap(), values);
    }

    #[test]
    #[ignore = "需要显式设置 H3_ROCM_TEST_DEVICES 并占用对应 ROCm 卡"]
    fn ulysses_qkv_and_attention_output_exchanges_are_bitwise() {
        use crate::backend::Backend;

        let devices = std::env::var("H3_ROCM_TEST_DEVICES").expect("设置 H3_ROCM_TEST_DEVICES，例如 0,1,2,3").split(',').map(|value| value.parse::<i32>().expect("device id 必须是 i32")).collect::<Vec<_>>();
        let group = H3RocmUlyssesGroup::new(&devices, false).unwrap();
        let rows = 11;
        let heads = 8;
        let head_dim = 8;
        let plan = H3UlyssesPlan::new(rows, heads, devices.len()).unwrap();

        let qkv_cols = heads * head_dim * 3;
        let qkv_values = (0..rows * qkv_cols).map(|index| index as f32 - 999.0).collect::<Vec<_>>();
        let qkv = group.primary().tensor_from_f32(qkv_values.clone(), rows, qkv_cols).unwrap();
        let sequence_shards = group.scatter_sequence(&qkv, &plan).unwrap();
        let compact = group
            .contexts
            .iter()
            .zip(&sequence_shards)
            .map(|(source, tensor)| plan.shards.iter().map(|head_shard| source.compact_qkv_heads(tensor, heads, head_shard.heads.clone(), head_dim).and_then(|tensor| source.tensor_to_stable_deferred(tensor))).collect::<Result<Vec<_>, _>>())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        group.synchronize_all().unwrap();
        for (head_rank, destination) in group.contexts.iter().enumerate() {
            let chunks = compact.iter().map(|source| destination.tensor_on_device_ordered(source[head_rank].clone())).collect::<Result<Vec<_>, _>>().unwrap();
            group.synchronize_all().unwrap();
            for (source_rank, chunk) in chunks.iter().enumerate() {
                let values = destination.tensor_to_f32(chunk).unwrap();
                assert!(values.iter().any(|value| *value != 0.0), "第一次 QKV exchange source_rank={source_rank} head_rank={head_rank} 全 0");
            }
            let refs = chunks.iter().collect::<Vec<_>>();
            let actual = destination.concat_token_rows(&refs).unwrap();
            let actual = destination.tensor_to_f32(&actual).unwrap();
            let head_range = &plan.shards[head_rank].heads;
            let local_cols = head_range.len() * head_dim;
            let mut expected = Vec::with_capacity(rows * local_cols * 3);
            for row in 0..rows {
                for section in 0..3 {
                    let start = row * qkv_cols + section * heads * head_dim + head_range.start * head_dim;
                    expected.extend_from_slice(&qkv_values[start..start + local_cols]);
                }
            }
            assert_eq!(actual, expected, "第一次 QKV exchange head_rank={head_rank}");
        }

        let output_cols = heads * head_dim;
        let output_values = (0..rows * output_cols).map(|index| index as f32 + 17.0).collect::<Vec<_>>();
        let head_outputs = plan
            .shards
            .iter()
            .zip(&group.contexts)
            .map(|(head_shard, context)| {
                let mut values = Vec::with_capacity(rows * head_shard.heads.len() * head_dim);
                for row in 0..rows {
                    let start = row * output_cols + head_shard.heads.start * head_dim;
                    values.extend_from_slice(&output_values[start..start + head_shard.heads.len() * head_dim]);
                }
                context.tensor_from_f32(values, rows, head_shard.heads.len() * head_dim).unwrap()
            })
            .collect::<Vec<_>>();
        let source_columns = plan
            .shards
            .iter()
            .map(|shard| {
                group
                    .contexts
                    .iter()
                    .enumerate()
                    .map(|(head_rank, source)| source.slice_token_rows(&head_outputs[head_rank], shard.sequence.start, shard.sequence.len()).and_then(|tensor| source.tensor_to_stable_deferred(tensor)))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        group.synchronize_all().unwrap();
        let mut sequence_outputs = Vec::new();
        for (destination, columns) in group.contexts.iter().zip(source_columns) {
            let columns = columns.into_iter().map(|column| destination.tensor_on_device_ordered(column)).collect::<Result<Vec<_>, _>>().unwrap();
            group.synchronize_all().unwrap();
            let mut columns = columns.into_iter();
            let mut joined = columns.next().unwrap();
            for column in columns {
                joined = destination.concat_columns(&joined, &column).unwrap();
            }
            sequence_outputs.push(joined);
        }
        let gathered = group.gather_sequence(&sequence_outputs, &plan).unwrap();
        assert_eq!(group.primary().tensor_to_f32(&gathered).unwrap(), output_values, "第二次 attention output exchange");
    }
}
