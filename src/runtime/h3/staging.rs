//! H3 DiT 权重常驻、分层执行与低内存流式执行。

use std::time::Instant;

use super::{
    H3ConditionInput, H3DenoiseLatents, H3PackedInputs, H3PackedLayout, H3PreparedGlobal, H3VelocityOutput, HostLayerCache, HostStreamedLayers, build_modulation_plan, dit_block, dit_final_layer, h3_euler_step, h3_sigma_schedule,
    h3_time_state, mm_rope_tables, pack_dit_inputs, pack_dit_inputs_with_prefix, prepare_block_weights, prepare_dit_static_prefix_conditioned, prepare_final_weights, time_embedding,
};
use crate::{
    backend::{Backend, BackendError, DiffusionBackend},
    weight::model::h3::{H3DitBlockWeights, H3DitFinalWeights, H3DitSource},
};

/// 把连续 DiT block 准备到同一个 backend，供多个采样步复用。
pub fn prepare_dit_blocks<B: Backend>(backend: &B, source: &H3DitSource, layers: std::ops::Range<usize>) -> Result<Vec<H3DitBlockWeights<B::Weight>>, BackendError> {
    if layers.start >= layers.end || layers.end > source.config().num_layers {
        return Err(BackendError::Compute { msg: format!("H3 resident block range={layers:?}，num_layers={} 非法", source.config().num_layers,) });
    }
    // 注:曾尝试 rayon into_par_iter 并行 prepare,但 MetalContext 持 MTLCommandBuffer
    // (!Send + !Sync) 无法跨线程,ROCm 走 streamed 不经此函数。串行 prepare 仅一次性
    // 发生(staged path 跨步复用),不构成瓶颈,故保留串行。
    layers
        .map(|layer| {
            let source_weights = source.load_block(layer).map_err(|error| BackendError::ExpertLoad(format!("读取 H3 resident block {layer} 失败: {error}",)))?;
            prepare_block_weights(backend, &source_weights)
        })
        .collect()
}

pub fn prepare_cached_dit_blocks<B: Backend>(backend: &B, block_sources: &HostLayerCache<H3DitBlockWeights>) -> Result<Vec<H3DitBlockWeights<B::Weight>>, BackendError> {
    block_sources.iter().map(|weights| prepare_block_weights(backend, weights)).collect()
}

/// final layer 很小，固定放到主 backend 并跨采样步复用。
pub fn prepare_dit_final<B: Backend>(backend: &B, source: &H3DitSource) -> Result<H3DitFinalWeights<B::Weight>, BackendError> {
    let source_weights = source.load_final().map_err(|error| BackendError::ExpertLoad(format!("读取 H3 resident final layer 失败: {error}",)))?;
    prepare_final_weights(backend, &source_weights)
}

/// 一个连续层 stage；backend 和权重必须属于同一个设备。
pub struct H3PreparedStage<'a, B: Backend> {
    pub backend: &'a B,
    pub global: &'a H3PreparedGlobal<B::Weight>,
    pub first_layer: usize,
    pub blocks: &'a [H3DitBlockWeights<B::Weight>],
}

#[allow(clippy::too_many_arguments)]
fn dit_backbone_forward_staged<B: DiffusionBackend>(
    primary: &B,
    source: &H3DitSource,
    primary_global: &H3PreparedGlobal<B::Weight>,
    stages: &[H3PreparedStage<'_, B>],
    final_weights: &H3DitFinalWeights<B::Weight>,
    hidden: B::Tensor,
    layout: &H3PackedLayout,
    sigma_video: f32,
) -> Result<H3VelocityOutput<B::Tensor>, BackendError> {
    let config = source.config();
    if primary.token_rows(&hidden) != layout.seq_len || primary.token_cols(&hidden) != config.hidden_size {
        return Err(BackendError::Compute { msg: format!("H3 staged hidden=[{},{}]，期望 [{},{}]", primary.token_rows(&hidden), primary.token_cols(&hidden), layout.seq_len, config.hidden_size,) });
    }
    let time = h3_time_state(sigma_video, config).map_err(|msg| BackendError::Compute { msg })?;
    let modulation = build_modulation_plan(layout, time);
    let (cosine, sine) = mm_rope_tables(&layout.position_ids, &primary_global.rope_inv_freq).map_err(|msg| BackendError::Compute { msg })?;
    let mut hidden = hidden;
    let mut next_layer = 0;
    for stage in stages {
        if stage.first_layer != next_layer || stage.blocks.is_empty() {
            return Err(BackendError::Compute { msg: format!("H3 stage first={} blocks={}，期望从 {next_layer} 开始", stage.first_layer, stage.blocks.len(),) });
        }
        hidden = stage.backend.transfer_tensor(hidden)?;
        let time_embedding = time_embedding(stage.backend, config, &modulation.unique_timesteps, &stage.global.weights)?;
        let result = (|| {
            for weights in stage.blocks {
                let _scope = stage.backend.layer_scope();
                stage.backend.begin_batch();
                hidden = dit_block(stage.backend, config, &hidden, &time_embedding, &modulation.segments, weights, &cosine, &sine).map_err(|error| BackendError::Compute { msg: format!("H3 staged block {next_layer}: {error:?}") })?;
                stage.backend.submit_batch();
                next_layer += 1;
            }
            Ok(hidden)
        })();
        stage.backend.finish_batch();
        hidden = result?;
    }
    if next_layer != config.num_layers {
        return Err(BackendError::Compute { msg: format!("H3 staged layers={next_layer}，期望 {}", config.num_layers,) });
    }
    let hidden = primary.transfer_tensor(hidden)?;
    let time_embedding = time_embedding(primary, config, &modulation.unique_timesteps, &primary_global.weights)?;
    primary.begin_batch();
    let result = dit_final_layer(primary, config, &hidden, &time_embedding, &modulation.segments, layout.audio_rows.clone(), layout.video_rows.clone(), final_weights, time.audio_velocity_scale);
    primary.finish_batch();
    result
}

/// 单 backend 流式执行;host 按 chunk 加载用完即释放,设备只保留当前 chunk。
/// 24GB unified memory 设备走 [`HostStreamedLayers`] 路径;ROCm 80GB HBM 仍走
/// [`HostLayerCache`] 常驻路径(在 `denoise_staged` 中)。
#[allow(clippy::too_many_arguments)]
fn dit_backbone_forward_streamed<'a, B: DiffusionBackend + 'a>(
    backend: &B,
    source: &H3DitSource,
    global: &H3PreparedGlobal<B::Weight>,
    block_sources: &mut HostStreamedLayers<'a, H3DitBlockWeights>,
    chunk_size: usize,
    final_weights: &H3DitFinalWeights<B::Weight>,
    hidden: B::Tensor,
    layout: &H3PackedLayout,
    sigma_video: f32,
) -> Result<H3VelocityOutput<B::Tensor>, BackendError> {
    let config = source.config();
    if block_sources.layer_count() != config.num_layers {
        return Err(BackendError::Compute { msg: format!("H3 host streamed layers={}，期望 {}", block_sources.layer_count(), config.num_layers) });
    }
    if backend.token_rows(&hidden) != layout.seq_len || backend.token_cols(&hidden) != config.hidden_size {
        return Err(BackendError::Compute { msg: format!("H3 streamed hidden=[{},{}]，期望 [{},{}]", backend.token_rows(&hidden), backend.token_cols(&hidden), layout.seq_len, config.hidden_size,) });
    }
    let time = h3_time_state(sigma_video, config).map_err(|msg| BackendError::Compute { msg })?;
    let modulation = build_modulation_plan(layout, time);
    let (cosine, sine) = mm_rope_tables(&layout.position_ids, &global.rope_inv_freq).map_err(|msg| BackendError::Compute { msg })?;
    let time_embedding = time_embedding(backend, config, &modulation.unique_timesteps, &global.weights)?;
    let mut hidden = hidden;
    block_sources.rewind();
    let mut next_layer = 0;
    let mut prepare_wall = 0.0;
    let mut compute_wall = 0.0;
    let denoise_started = Instant::now();
    while let Some(source_weights) = block_sources.next_chunk(chunk_size).map_err(|msg| BackendError::Compute { msg })? {
        let started = Instant::now();
        let weights = source_weights.iter().map(|weights| prepare_block_weights(backend, weights)).collect::<Result<Vec<_>, _>>()?;
        prepare_wall += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let result = (|| {
            for weights in &weights {
                let _scope = backend.layer_scope();
                backend.begin_batch();
                hidden = dit_block(backend, config, &hidden, &time_embedding, &modulation.segments, weights, &cosine, &sine).map_err(|error| BackendError::Compute { msg: format!("H3 streamed block {next_layer}: {error:?}") })?;
                backend.submit_batch();
                next_layer += 1;
            }
            Ok(hidden)
        })();
        backend.finish_stream_chunk();
        hidden = result?;
        compute_wall += started.elapsed().as_secs_f64();
        let loaded = next_layer;
        eprintln!("[h3-stream] layer {loaded}/{total_layers} wall={wall:.1}s prepare={pw:.2}s compute={cw:.2}s", total_layers = config.num_layers, wall = denoise_started.elapsed().as_secs_f64(), pw = prepare_wall, cw = compute_wall,);
        drop(weights);
    }
    if next_layer != config.num_layers {
        return Err(BackendError::Compute { msg: format!("H3 streamed layers={next_layer}，期望 {}", config.num_layers) });
    }
    eprintln!("[h3-stream] layers={next_layer} chunk={chunk_size} prepare={prepare_wall:.3}s compute={compute_wall:.3}s");
    backend.begin_batch();
    let result = dit_final_layer(backend, config, &hidden, &time_embedding, &modulation.segments, layout.audio_rows.clone(), layout.video_rows.clone(), final_weights, time.audio_velocity_scale);
    backend.finish_batch();
    result
}

/// 多 backend 分层常驻采样；每个采样步只搬运 stage 边界 activation。
#[allow(clippy::too_many_arguments)]
pub fn denoise_staged<B: DiffusionBackend>(
    primary: &B,
    source: &H3DitSource,
    primary_global: &H3PreparedGlobal<B::Weight>,
    stages: &[H3PreparedStage<'_, B>],
    final_weights: &H3DitFinalWeights<B::Weight>,
    text: &B::Tensor,
    video_conditions: &[B::Tensor],
    mut latents: H3DenoiseLatents<B::Tensor>,
    layout: &H3PackedLayout,
    num_inference_steps: usize,
) -> Result<H3DenoiseLatents<B::Tensor>, BackendError> {
    let video_sigmas = h3_sigma_schedule(num_inference_steps, source.config().sigma_shift_video).map_err(|msg| BackendError::Compute { msg })?;
    let audio_sigmas = h3_sigma_schedule(num_inference_steps, source.config().sigma_shift_audio).map_err(|msg| BackendError::Compute { msg })?;
    if audio_sigmas.len() != video_sigmas.len() {
        return Err(BackendError::Compute { msg: format!("H3 video/audio sigma 网格长度不同：{}/{}", video_sigmas.len(), audio_sigmas.len()) });
    }
    for index in 0..video_sigmas.len() - 1 {
        let hidden = pack_dit_inputs(primary, source, primary_global, H3PackedInputs { text, video_conditions, audio: &latents.audio, video: &latents.video }, layout)?;
        let output = dit_backbone_forward_staged(primary, source, primary_global, stages, final_weights, hidden, layout, video_sigmas[index])?;
        latents.video = h3_euler_step(primary, &latents.video, &output.video, video_sigmas[index], video_sigmas[index + 1])?;
        latents.audio = h3_euler_step(primary, &latents.audio, &output.audio, audio_sigmas[index], audio_sigmas[index + 1])?;
    }
    Ok(latents)
}

/// 多 backend 分层常驻的 reference-aware 采样；静态文本与参考条件只投影一次。
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn denoise_staged_conditioned<B: DiffusionBackend>(
    primary: &B,
    source: &H3DitSource,
    primary_global: &H3PreparedGlobal<B::Weight>,
    stages: &[H3PreparedStage<'_, B>],
    final_weights: &H3DitFinalWeights<B::Weight>,
    text: &B::Tensor,
    conditions: &[H3ConditionInput<'_, B::Tensor>],
    mut latents: H3DenoiseLatents<B::Tensor>,
    layout: &H3PackedLayout,
    num_inference_steps: usize,
    start_step: usize,
    on_step: &mut dyn FnMut(usize, usize, &H3DenoiseLatents<B::Tensor>) -> Result<bool, BackendError>,
) -> Result<H3DenoiseLatents<B::Tensor>, BackendError> {
    let video_sigmas = h3_sigma_schedule(num_inference_steps, source.config().sigma_shift_video).map_err(|msg| BackendError::Compute { msg })?;
    let audio_sigmas = h3_sigma_schedule(num_inference_steps, source.config().sigma_shift_audio).map_err(|msg| BackendError::Compute { msg })?;
    if audio_sigmas.len() != video_sigmas.len() {
        return Err(BackendError::Compute { msg: format!("H3 video/audio sigma 网格长度不同：{}/{}", video_sigmas.len(), audio_sigmas.len()) });
    }
    let total = video_sigmas.len() - 1;
    if start_step > total {
        return Err(BackendError::Compute { msg: format!("H3 resume step={start_step} 超过采样步数 {total}") });
    }
    let prefix = prepare_dit_static_prefix_conditioned(primary, source, primary_global, text, conditions)?;
    for index in start_step..total {
        let hidden = pack_dit_inputs_with_prefix(primary, source, primary_global, &prefix, &latents.audio, &latents.video, layout)?;
        let output = dit_backbone_forward_staged(primary, source, primary_global, stages, final_weights, hidden, layout, video_sigmas[index])?;
        latents.video = h3_euler_step(primary, &latents.video, &output.video, video_sigmas[index], video_sigmas[index + 1])?;
        latents.audio = h3_euler_step(primary, &latents.audio, &output.audio, audio_sigmas[index], audio_sigmas[index + 1])?;
        primary.synchronize()?;
        if !on_step(index + 1, total, &latents)? {
            return Err(BackendError::Compute { msg: format!("H3 denoise 在完整 step {}/{} 后中断", index + 1, total) });
        }
    }
    Ok(latents)
}

/// 单设备流式 reference-aware 采样；完整 host 权重常驻，设备权重按 chunk 生存。
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn denoise_streamed_conditioned<'a, B: DiffusionBackend + 'a>(
    backend: &B,
    source: &H3DitSource,
    global: &H3PreparedGlobal<B::Weight>,
    block_sources: &mut HostStreamedLayers<'a, H3DitBlockWeights>,
    chunk_size: usize,
    final_weights: &H3DitFinalWeights<B::Weight>,
    text: &B::Tensor,
    conditions: &[H3ConditionInput<'_, B::Tensor>],
    mut latents: H3DenoiseLatents<B::Tensor>,
    layout: &H3PackedLayout,
    num_inference_steps: usize,
    start_step: usize,
    on_step: &mut dyn FnMut(usize, usize, &H3DenoiseLatents<B::Tensor>) -> Result<bool, BackendError>,
) -> Result<H3DenoiseLatents<B::Tensor>, BackendError> {
    let video_sigmas = h3_sigma_schedule(num_inference_steps, source.config().sigma_shift_video).map_err(|msg| BackendError::Compute { msg })?;
    let audio_sigmas = h3_sigma_schedule(num_inference_steps, source.config().sigma_shift_audio).map_err(|msg| BackendError::Compute { msg })?;
    if audio_sigmas.len() != video_sigmas.len() {
        return Err(BackendError::Compute { msg: format!("H3 video/audio sigma 网格长度不同：{}/{}", video_sigmas.len(), audio_sigmas.len()) });
    }
    let prefix = prepare_dit_static_prefix_conditioned(backend, source, global, text, conditions)?;
    let total = video_sigmas.len() - 1;
    if start_step > total {
        return Err(BackendError::Compute { msg: format!("H3 resume step={start_step} 超过采样步数 {total}") });
    }
    for index in start_step..total {
        let hidden = pack_dit_inputs_with_prefix(backend, source, global, &prefix, &latents.audio, &latents.video, layout)?;
        let output = dit_backbone_forward_streamed(backend, source, global, block_sources, chunk_size, final_weights, hidden, layout, video_sigmas[index])?;
        latents.video = h3_euler_step(backend, &latents.video, &output.video, video_sigmas[index], video_sigmas[index + 1])?;
        latents.audio = h3_euler_step(backend, &latents.audio, &output.audio, audio_sigmas[index], audio_sigmas[index + 1])?;
        backend.synchronize()?;
        if !on_step(index + 1, total, &latents)? {
            return Err(BackendError::Compute { msg: format!("H3 denoise 在完整 step {}/{} 后中断", index + 1, total) });
        }
    }
    Ok(latents)
}

/// 对已经按 `[text | condition | audio | video]` 投影到 hidden 的序列执行完整 50 层 DiT。
/// Global 权重跨层常驻；block 从 safetensors 逐层读取、准备并在 layer scope 结束时释放。
pub fn dit_backbone_forward<B: DiffusionBackend>(backend: &B, source: &H3DitSource, global: &H3PreparedGlobal<B::Weight>, hidden: B::Tensor, layout: &H3PackedLayout, sigma_video: f32) -> Result<H3VelocityOutput<B::Tensor>, BackendError> {
    let config = source.config();
    if backend.token_rows(&hidden) != layout.seq_len || backend.token_cols(&hidden) != config.hidden_size {
        return Err(BackendError::Compute { msg: format!("H3 packed hidden=[{},{}]，期望 [{},{}]", backend.token_rows(&hidden), backend.token_cols(&hidden), layout.seq_len, config.hidden_size,) });
    }
    let time = h3_time_state(sigma_video, config).map_err(|msg| BackendError::Compute { msg })?;
    let modulation = build_modulation_plan(layout, time);
    let (cosine, sine) = mm_rope_tables(&layout.position_ids, &global.rope_inv_freq).map_err(|msg| BackendError::Compute { msg })?;
    let time_embedding = time_embedding(backend, config, &modulation.unique_timesteps, &global.weights)?;
    let mut hidden = hidden;
    for layer in 0..config.num_layers {
        let _scope = backend.layer_scope();
        let source_weights = source.load_block(layer).map_err(|error| BackendError::ExpertLoad(format!("读取 H3 block {layer} 失败: {error}",)))?;
        let weights = prepare_block_weights(backend, &source_weights)?;
        backend.begin_batch();
        let result = dit_block(backend, config, &hidden, &time_embedding, &modulation.segments, &weights, &cosine, &sine);
        backend.finish_stream_chunk();
        hidden = result?;
    }
    let _scope = backend.layer_scope();
    let source_weights = source.load_final().map_err(|error| BackendError::ExpertLoad(format!("读取 H3 final layer 失败: {error}",)))?;
    let weights = prepare_final_weights(backend, &source_weights)?;
    backend.begin_batch();
    let result = dit_final_layer(backend, config, &hidden, &time_embedding, &modulation.segments, layout.audio_rows.clone(), layout.video_rows.clone(), &weights, time.audio_velocity_scale);
    backend.finish_batch();
    result
}
