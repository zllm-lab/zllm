//! H3 视频 VAE 权重准备、解码与时空分块。

use super::{linear_bias, patchify_video, prepare_h3_video_vae_weight};
use crate::{
    backend::{Backend, BackendError, DiffusionBackend, VaeBackend},
    moe::Activation,
    vae::H3VideoVaeSpec,
    weight::model::h3_vae::{H3VideoVaeBlockWeights, H3VideoVaeSource},
};

pub struct H3PreparedVideoVae<W> {
    pub latent_scale: W,
    pub latent_bias: W,
    pub post_quant_weight: W,
    pub post_quant_bias: W,
    pub input_weight: W,
    pub input_bias: W,
    pub register_tokens: W,
    pub zero_suffix: W,
    pub norm_weight: W,
    pub norm_bias: W,
    pub output_weight: W,
    pub output_bias: W,
    pub blocks: Vec<H3VideoVaeBlockWeights<W>>,
}

pub fn prepare_video_vae<B: VaeBackend>(backend: &B, source: &H3VideoVaeSource) -> Result<H3PreparedVideoVae<B::Weight>, BackendError> {
    let weights = source.load_decoder_global().map_err(BackendError::ExpertLoad)?;
    let spec = H3VideoVaeSpec::standard();
    let blocks = (0..spec.decoder_layers)
        .map(|layer| {
            let weights = source.load_decoder_block(layer).map_err(|error| BackendError::ExpertLoad(format!("读取 H3 resident video VAE block {layer} 失败: {error}",)))?;
            prepare_video_vae_block(backend, &weights)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(H3PreparedVideoVae {
        latent_scale: prepare_h3_video_vae_weight(backend, &weights.latent_scale)?,
        latent_bias: prepare_h3_video_vae_weight(backend, &weights.latent_bias)?,
        post_quant_weight: prepare_h3_video_vae_weight(backend, &weights.post_quant_weight)?,
        post_quant_bias: prepare_h3_video_vae_weight(backend, &weights.post_quant_bias)?,
        input_weight: prepare_h3_video_vae_weight(backend, &weights.input_weight)?,
        input_bias: prepare_h3_video_vae_weight(backend, &weights.input_bias)?,
        register_tokens: prepare_h3_video_vae_weight(backend, &weights.register_tokens)?,
        zero_suffix: prepare_h3_video_vae_weight(backend, &weights.zero_suffix)?,
        norm_weight: prepare_h3_video_vae_weight(backend, &weights.norm_weight)?,
        norm_bias: prepare_h3_video_vae_weight(backend, &weights.norm_bias)?,
        output_weight: prepare_h3_video_vae_weight(backend, &weights.output_weight)?,
        output_bias: prepare_h3_video_vae_weight(backend, &weights.output_bias)?,
        blocks,
    })
}

fn prepare_video_vae_block<B: Backend>(backend: &B, source: &H3VideoVaeBlockWeights) -> Result<H3VideoVaeBlockWeights<B::Weight>, BackendError> {
    Ok(H3VideoVaeBlockWeights {
        norm1: prepare_h3_video_vae_weight(backend, &source.norm1)?,
        qkv_weight: prepare_h3_video_vae_weight(backend, &source.qkv_weight)?,
        qkv_bias: prepare_h3_video_vae_weight(backend, &source.qkv_bias)?,
        attention_output_weight: prepare_h3_video_vae_weight(backend, &source.attention_output_weight)?,
        attention_output_bias: prepare_h3_video_vae_weight(backend, &source.attention_output_bias)?,
        scale1: prepare_h3_video_vae_weight(backend, &source.scale1)?,
        norm2: prepare_h3_video_vae_weight(backend, &source.norm2)?,
        gate_up_weight: prepare_h3_video_vae_weight(backend, &source.gate_up_weight)?,
        gate_up_bias: prepare_h3_video_vae_weight(backend, &source.gate_up_bias)?,
        down_weight: prepare_h3_video_vae_weight(backend, &source.down_weight)?,
        down_bias: prepare_h3_video_vae_weight(backend, &source.down_bias)?,
        scale2: prepare_h3_video_vae_weight(backend, &source.scale2)?,
    })
}

fn video_vae_rope_tables(shape: [usize; 3], spec: &H3VideoVaeSpec) -> Result<(Vec<f32>, Vec<f32>), String> {
    if shape.contains(&0) || !spec.decoder_rope_dim.is_multiple_of(6) {
        return Err(format!("H3 video VAE RoPE shape={shape:?} dim={} 非法", spec.decoder_rope_dim));
    }
    let frequencies = spec.decoder_rope_dim / 6;
    let suffix = spec.decoder_register_tokens + 1;
    let rows = shape.iter().product::<usize>() + suffix;
    let mut cosine = Vec::with_capacity(rows * spec.decoder_rope_dim / 2);
    let mut sine = Vec::with_capacity(rows * spec.decoder_rope_dim / 2);
    for time in 0..shape[0] {
        for height in 0..shape[1] {
            for width in 0..shape[2] {
                for (coordinate, size) in [(time, shape[0]), (height, shape[1]), (width, shape[2])] {
                    let position = 2.0 * ((coordinate as f32 + 0.5) / size as f32) - 1.0;
                    for index in 0..frequencies {
                        let frequency = spec.decoder_rope_theta.powf(-(index as f32) / frequencies as f32);
                        let angle = std::f32::consts::TAU * position * frequency;
                        cosine.push(angle.cos());
                        sine.push(angle.sin());
                    }
                }
            }
        }
    }
    cosine.extend(std::iter::repeat_n(1.0, suffix * spec.decoder_rope_dim / 2));
    sine.extend(std::iter::repeat_n(0.0, suffix * spec.decoder_rope_dim / 2));
    Ok((cosine, sine))
}

fn video_vae_block<B: DiffusionBackend + VaeBackend>(backend: &B, hidden: &B::Tensor, weights: &H3VideoVaeBlockWeights<B::Weight>, cosine: &[f32], sine: &[f32], batch: usize) -> Result<B::Tensor, BackendError> {
    let spec = H3VideoVaeSpec::standard();
    let normalized = backend.rmsnorm(hidden, &weights.norm1, spec.decoder_norm_eps)?;
    let qkv = backend.linear(&normalized, &weights.qkv_weight)?;
    let (query, key, value) = backend.split_three_columns_bias(&qkv, &weights.qkv_bias, spec.decoder_hidden_size)?;
    let (query, key) = backend.rms_norm_rope_pair_unit(&query, &key, spec.decoder_heads, spec.decoder_head_dim, spec.decoder_rope_dim, spec.decoder_norm_eps, cosine, sine)?;
    let rows = backend.token_rows(hidden) / batch;
    let attended = backend.full_attention_batched(query, key, value, batch, rows, spec.decoder_heads, spec.decoder_head_dim, (spec.decoder_head_dim as f32).sqrt().recip())?;
    let update = backend.linear(&attended, &weights.attention_output_weight)?;
    let hidden = backend.scaled_residual_bias(hidden, &update, &weights.attention_output_bias, &weights.scale1)?;
    let normalized = backend.rmsnorm(&hidden, &weights.norm2, spec.decoder_norm_eps)?;
    let gate_up = backend.linear(&normalized, &weights.gate_up_weight)?;
    let activated = backend.split_gated_bias_activation(gate_up, &weights.gate_up_bias, spec.decoder_ffn_size, &Activation::Silu)?;
    let update = backend.linear(&activated, &weights.down_weight)?;
    backend.scaled_residual_bias(&hidden, &update, &weights.down_bias, &weights.scale2)
}

pub(crate) fn decode_video_vae_batched<B: DiffusionBackend + VaeBackend>(
    backend: &B,
    _source: &H3VideoVaeSource,
    global: &H3PreparedVideoVae<B::Weight>,
    latents: &[B::Tensor],
    shape: [usize; 3],
    patch: [usize; 3],
) -> Result<B::Tensor, BackendError> {
    let spec = H3VideoVaeSpec::standard();
    let batch = latents.len();
    let rows = shape.iter().product::<usize>();
    if batch == 0 || shape.contains(&0) || patch.contains(&0) || (0..3).any(|axis| !shape[axis].is_multiple_of(patch[axis])) {
        return Err(BackendError::Compute { msg: format!("H3 video VAE batch={batch} shape={shape:?} patch={patch:?} 不兼容") });
    }
    let patch_rows = (0..3).map(|axis| shape[axis] / patch[axis]).product::<usize>();
    let patch_columns = spec.latent_channels * patch.into_iter().product::<usize>();
    let mut combined = None;
    for latent in latents {
        if backend.token_rows(latent) != patch_rows || backend.token_cols(latent) != patch_columns {
            return Err(BackendError::Compute { msg: format!("H3 video VAE latent=[{},{}]，期望 [{patch_rows},{patch_columns}]", backend.token_rows(latent), backend.token_cols(latent)) });
        }
        let latent = backend.unpatch_affine(latent, &global.latent_scale, &global.latent_bias, shape, patch, spec.latent_channels)?;
        let hidden = linear_bias(backend, &latent, &global.post_quant_weight, &global.post_quant_bias)?;
        let hidden = linear_bias(backend, &hidden, &global.input_weight, &global.input_bias)?;
        combined = Some(match combined.take() {
            Some(previous) => backend.concat_rows(&previous, &hidden)?,
            None => hidden,
        });
    }
    let hidden = combined.ok_or_else(|| BackendError::Compute { msg: "H3 video VAE batch 为空".to_owned() })?;
    let hidden = backend.concat_weight_rows_batched(&hidden, &global.register_tokens, spec.decoder_register_tokens, batch)?;
    let mut hidden = backend.concat_weight_rows_batched(&hidden, &global.zero_suffix, 1, batch)?;
    let (cosine, sine) = video_vae_rope_tables(shape, &spec).map_err(|msg| BackendError::Compute { msg })?;
    let cosine = cosine.repeat(batch);
    let sine = sine.repeat(batch);
    if global.blocks.len() != spec.decoder_layers {
        return Err(BackendError::Compute { msg: format!("H3 resident video VAE blocks={}，期望 {}", global.blocks.len(), spec.decoder_layers) });
    }
    let result = (|| {
        for weights in &global.blocks {
            backend.begin_batch();
            hidden = video_vae_block(backend, &hidden, weights, &cosine, &sine, batch)?;
            backend.submit_batch();
        }
        let hidden = backend.take_rows_batched(&hidden, rows, batch)?;
        let hidden = backend.layer_norm(&hidden, &global.norm_weight, &global.norm_bias, spec.decoder_norm_eps)?;
        linear_bias(backend, &hidden, &global.output_weight, &global.output_bias)
    })();
    backend.finish_batch();
    result
}

pub fn video_vae_latent_frames(frames: usize, spec: &H3VideoVaeSpec) -> Result<usize, String> {
    if frames < 5 || !(frames - 5).is_multiple_of(spec.temporal_clip_length) {
        return Err(format!("H3 frames={frames} 与 clip_length={} 不兼容", spec.temporal_clip_length));
    }
    let tokens_per_chunk = spec.temporal_clip_length.div_ceil(spec.temporal_compression);
    Ok((frames - 5) / spec.temporal_clip_length * tokens_per_chunk + tokens_per_chunk - spec.temporal_token_drop)
}

#[allow(clippy::type_complexity)]
fn split_video_tiles(input: usize, ratio: usize) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), String> {
    const TILE_SIZE: usize = 256;
    const OVERLAP_MIN: usize = 64;
    if TILE_SIZE >= input {
        return Ok((vec![0], vec![input], Vec::new()));
    }
    let mut count = input.div_ceil(TILE_SIZE);
    let mut overlaps;
    let remaining;
    loop {
        overlaps = vec![OVERLAP_MIN; count - 1];
        let covered = TILE_SIZE * count - overlaps.iter().sum::<usize>();
        if covered >= input {
            remaining = covered - input;
            break;
        }
        count += 1;
    }
    if remaining % ratio != 0 {
        return Err(format!("H3 VAE tile remaining={remaining} 不能被 ratio={ratio} 整除"));
    }
    for index in 0..remaining / ratio {
        let overlap = index % overlaps.len();
        overlaps[overlap] += ratio;
    }
    let mut starts = vec![0];
    for index in 0..count - 1 {
        starts.push(starts[index] + TILE_SIZE - overlaps[index]);
    }
    Ok((starts, vec![TILE_SIZE; count], overlaps))
}

#[allow(clippy::too_many_arguments)]
fn prepare_video_spatial_tiles(clip: &[f32], clip_t: usize, latent_h: usize, latent_w: usize, patch: [usize; 3], spec: &H3VideoVaeSpec) -> Result<(Vec<Vec<f32>>, [usize; 3]), String> {
    let ratio = spec.spatial_compression;
    let pixel_h = latent_h * ratio;
    let pixel_w = latent_w * ratio;
    let (y_starts, y_lengths, _) = split_video_tiles(pixel_h, ratio)?;
    let (x_starts, x_lengths, _) = split_video_tiles(pixel_w, ratio)?;
    let batch_tile_h = y_lengths[0];
    let batch_tile_w = x_lengths[0];
    if y_lengths.iter().any(|&length| length != batch_tile_h) || x_lengths.iter().any(|&length| length != batch_tile_w) {
        return Err("H3 VAE batched spatial tiles 必须等尺寸".to_owned());
    }
    let tile_latent_h = batch_tile_h / ratio;
    let tile_latent_w = batch_tile_w / ratio;
    let tile_count = y_starts.len() * x_starts.len();
    let mut inputs = Vec::with_capacity(tile_count);
    for &pixel_y in &y_starts {
        let latent_y = pixel_y / ratio;
        for &pixel_x in &x_starts {
            let latent_x = pixel_x / ratio;
            let mut tile_latent = vec![0.0f32; spec.latent_channels * clip_t * tile_latent_h * tile_latent_w];
            for channel in 0..spec.latent_channels {
                for time in 0..clip_t {
                    for row in 0..tile_latent_h {
                        let source_offset = ((channel * clip_t + time) * latent_h + latent_y + row) * latent_w + latent_x;
                        let target_offset = ((channel * clip_t + time) * tile_latent_h + row) * tile_latent_w;
                        tile_latent[target_offset..target_offset + tile_latent_w].copy_from_slice(&clip[source_offset..source_offset + tile_latent_w]);
                    }
                }
            }
            let rows = patchify_video(&tile_latent, [1, spec.latent_channels, clip_t, tile_latent_h, tile_latent_w], patch)?;
            inputs.push(rows);
        }
    }
    Ok((inputs, [clip_t, tile_latent_h, tile_latent_w]))
}

fn blend_video_spatial_tiles(blocks: &[Vec<f32>], clip_t: usize, latent_h: usize, latent_w: usize, spec: &H3VideoVaeSpec) -> Result<Vec<f32>, String> {
    let ratio = spec.spatial_compression;
    let pixel_h = latent_h * ratio;
    let pixel_w = latent_w * ratio;
    let frames = clip_t * spec.temporal_compression;
    let (y_starts, y_lengths, y_overlaps) = split_video_tiles(pixel_h, ratio)?;
    let (x_starts, x_lengths, x_overlaps) = split_video_tiles(pixel_w, ratio)?;
    let batch_tile_h = y_lengths[0];
    let batch_tile_w = x_lengths[0];
    let tile_count = y_starts.len() * x_starts.len();
    let tile_block_elements = 3 * frames * batch_tile_h * batch_tile_w;
    let output_elements = blocks.iter().map(Vec::len).sum::<usize>();
    if output_elements != tile_count * tile_block_elements || blocks.iter().any(|batch| !batch.len().is_multiple_of(tile_block_elements)) {
        return Err(format!("H3 batched spatial VAE output={output_elements}，期望 {tile_count} tiles × {tile_block_elements} elements"));
    }
    // 各卡回读保持独立分配，避免每个时间块再复制整个空间 batch。
    let block_tiles = blocks.iter().flat_map(|batch| batch.chunks_exact(tile_block_elements)).collect::<Vec<_>>();
    let mut canvas = vec![0.0f32; 3 * frames * pixel_h * pixel_w];
    // 各 plane 的空间融合互不依赖；每个 plane 内仍先做上下融合，再做左右融合。
    let blend = |first_plane: usize, canvas: &mut [f32]| -> Result<(), String> {
        let planes = canvas.len() / (pixel_h * pixel_w);
        let mut row_tails = Vec::<Vec<f32>>::new();
        let mut out_y = 0;

        for (tile_y, (&_pixel_y, &tile_pixel_h)) in y_starts.iter().zip(&y_lengths).enumerate() {
            let mut new_tails = Vec::<Vec<f32>>::new();
            let mut left_tail: Option<Vec<f32>> = None;
            let mut out_x = 0;
            let mut visible_h = 0;

            for (tile_x, (&_pixel_x, &tile_pixel_w)) in x_starts.iter().zip(&x_lengths).enumerate() {
                let tile_blocks = block_tiles[tile_y * x_lengths.len() + tile_x];
                let mut tile = vec![0.0f32; planes * tile_pixel_h * tile_pixel_w];
                let pt = spec.temporal_compression;
                let patch_columns = 3 * pt * ratio * ratio;
                for plane in 0..planes {
                    let original_plane = first_plane + plane;
                    let channel = original_plane / frames;
                    let frame = original_plane % frames;
                    for row in 0..tile_pixel_h {
                        for x in 0..tile_pixel_w / ratio {
                            let token = ((frame / pt * (tile_pixel_h / ratio) + row / ratio) * (tile_pixel_w / ratio)) + x;
                            let column = ((channel * pt + frame % pt) * ratio + row % ratio) * ratio;
                            let source = token * patch_columns + column;
                            let target = (plane * tile_pixel_h + row) * tile_pixel_w + x * ratio;
                            tile[target..target + ratio].copy_from_slice(&tile_blocks[source..source + ratio]);
                        }
                    }
                }

                let next_bottom = if tile_y + 1 < y_starts.len() {
                    let extent = y_overlaps[tile_y];
                    let mut tail = vec![0.0f32; planes * extent * tile_pixel_w];
                    for plane in 0..planes {
                        for row in 0..extent {
                            let source_offset = (plane * tile_pixel_h + tile_pixel_h - extent + row) * tile_pixel_w;
                            let target_offset = (plane * extent + row) * tile_pixel_w;
                            tail[target_offset..target_offset + tile_pixel_w].copy_from_slice(&tile[source_offset..source_offset + tile_pixel_w]);
                        }
                    }
                    Some(tail)
                } else {
                    None
                };
                let next_right = if tile_x + 1 < x_starts.len() {
                    let extent = x_overlaps[tile_x];
                    let mut tail = vec![0.0f32; planes * tile_pixel_h * extent];
                    for plane in 0..planes {
                        for row in 0..tile_pixel_h {
                            let source_offset = (plane * tile_pixel_h + row) * tile_pixel_w + tile_pixel_w - extent;
                            let target_offset = (plane * tile_pixel_h + row) * extent;
                            tail[target_offset..target_offset + extent].copy_from_slice(&tile[source_offset..source_offset + extent]);
                        }
                    }
                    Some(tail)
                } else {
                    None
                };

                if tile_y > 0 {
                    let extent = y_overlaps[tile_y - 1];
                    let previous = &row_tails[tile_x];
                    for plane in 0..planes {
                        for row in 0..extent {
                            let weight_b = row as f32 / extent as f32;
                            let weight_a = 1.0 - weight_b;
                            for column in 0..tile_pixel_w {
                                let tile_index = (plane * tile_pixel_h + row) * tile_pixel_w + column;
                                let tail_index = (plane * extent + row) * tile_pixel_w + column;
                                tile[tile_index] = previous[tail_index] * weight_a + tile[tile_index] * weight_b;
                            }
                        }
                    }
                }
                if tile_x > 0 {
                    let extent = x_overlaps[tile_x - 1];
                    let previous = left_tail.as_ref().ok_or("H3 VAE 缺少左侧 overlap")?;
                    for plane in 0..planes {
                        for row in 0..tile_pixel_h {
                            for column in 0..extent {
                                let weight_b = column as f32 / extent as f32;
                                let weight_a = 1.0 - weight_b;
                                let tile_index = (plane * tile_pixel_h + row) * tile_pixel_w + column;
                                let tail_index = (plane * tile_pixel_h + row) * extent + column;
                                tile[tile_index] = previous[tail_index] * weight_a + tile[tile_index] * weight_b;
                            }
                        }
                    }
                }

                visible_h = tile_pixel_h - if tile_y + 1 < y_starts.len() { y_overlaps[tile_y] } else { 0 };
                let visible_w = tile_pixel_w - if tile_x + 1 < x_starts.len() { x_overlaps[tile_x] } else { 0 };
                for plane in 0..planes {
                    for row in 0..visible_h {
                        let source_offset = (plane * tile_pixel_h + row) * tile_pixel_w;
                        let target_offset = (plane * pixel_h + out_y + row) * pixel_w + out_x;
                        canvas[target_offset..target_offset + visible_w].copy_from_slice(&tile[source_offset..source_offset + visible_w]);
                    }
                }
                if let Some(tail) = next_bottom {
                    new_tails.push(tail);
                }
                left_tail = next_right;
                out_x += visible_w;
            }
            row_tails = new_tails;
            out_y += visible_h;
        }

        Ok(())
    };
    let workers = std::thread::available_parallelism().map_or(1, |count| count.get()).min(4).min(3 * frames).min((canvas.len() / 65536).max(1));
    if workers == 1 {
        blend(0, &mut canvas)?;
    } else {
        let planes_per_worker = (3 * frames).div_ceil(workers);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for (part, canvas) in canvas.chunks_mut(planes_per_worker * pixel_h * pixel_w).enumerate() {
                let blend = &blend;
                handles.push(scope.spawn(move || blend(part * planes_per_worker, canvas)));
            }
            for handle in handles {
                handle.join().map_err(|_| "H3 VAE spatial blend 线程 panic".to_owned())??;
            }
            Ok::<_, String>(())
        })?;
    }
    Ok(canvas)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_video_vae_tiled_temporal<B: DiffusionBackend + VaeBackend>(
    backend: &B,
    source: &H3VideoVaeSource,
    global: &H3PreparedVideoVae<B::Weight>,
    latent: &[f32],
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    output_frames: usize,
    patch: [usize; 3],
) -> Result<Vec<f32>, String> {
    decode_video_vae_tiled_temporal_with_tiles(
        &mut |tiles, shape, patch| {
            let spec = H3VideoVaeSpec::standard();
            let rows = (0..3).map(|axis| shape[axis] / patch[axis]).product::<usize>();
            let cols = spec.latent_channels * patch.into_iter().product::<usize>();
            let inputs = tiles.iter().map(|tile| backend.vae_tensor_from_f32(tile.clone(), rows, cols)).collect::<Result<Vec<_>, _>>().map_err(|error| format!("上传 H3 spatial VAE tiles: {error:?}"))?;
            let blocks = decode_video_vae_batched(backend, source, global, &inputs, shape, patch).map_err(|error| format!("H3 batched spatial VAE: {error:?}"))?;
            backend.vae_tensor_to_f32(&blocks).map(|output| vec![output]).map_err(|error| format!("读取 H3 batched spatial VAE: {error:?}"))
        },
        latent,
        latent_t,
        latent_h,
        latent_w,
        output_frames,
        patch,
    )
}

/// 分块及融合保持唯一实现；设备组合只负责独立 tile batch 的执行与回读。
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_video_vae_tiled_temporal_with_tiles(
    decode_tiles: &mut impl FnMut(&[Vec<f32>], [usize; 3], [usize; 3]) -> Result<Vec<Vec<f32>>, String>,
    latent: &[f32],
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    output_frames: usize,
    patch: [usize; 3],
) -> Result<Vec<f32>, String> {
    let spec = H3VideoVaeSpec::standard();
    let clips = video_temporal_clips(latent, [latent_t, latent_h, latent_w])?;
    let decoded = clips.enumerate().map(|(chunk, (clip_t, clip))| {
        let result = (|| {
            let (inputs, shape) = prepare_video_spatial_tiles(&clip, clip_t, latent_h, latent_w, patch, &spec)?;
            let blocks = decode_tiles(&inputs, shape, patch)?;
            let pixels = blend_video_spatial_tiles(&blocks, clip_t, latent_h, latent_w, &spec)?;
            Ok((clip_t * spec.temporal_compression, pixels))
        })();
        result.map_err(|error: String| format!("H3 temporal VAE chunk {chunk}: {error}"))
    });
    blend_video_temporal_chunks(decoded, latent_h, latent_w, output_frames, &mut |_, _| Ok(()))
}

/// 零容量握手：consumer 融合当前块时，producer 最多准备并解码下一块。
/// 队列只交接已回读的主存数据；错误退出必须先关 receiver，再等待 producer。
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_video_vae_tiled_temporal_pipelined_with_tiles(
    decode_tiles: &mut (impl FnMut(&[Vec<f32>], [usize; 3], [usize; 3]) -> Result<Vec<Vec<f32>>, String> + Send),
    latent: &[f32],
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    output_frames: usize,
    patch: [usize; 3],
    cancellation: &std::sync::atomic::AtomicBool,
    on_decoded: &mut dyn FnMut(&[f32], usize) -> Result<(), String>,
) -> Result<Vec<f32>, String> {
    use std::sync::{atomic::Ordering, mpsc::sync_channel};
    if cancellation.load(Ordering::Acquire) {
        return Err("H3 VAE 任务已取消".to_owned());
    }
    let clips = video_temporal_clips(latent, [latent_t, latent_h, latent_w])?;
    let spec = H3VideoVaeSpec::standard();
    std::thread::scope(|scope| {
        let (sender, receiver) = sync_channel(0);
        let producer = scope.spawn(move || {
            for (chunk, (clip_t, clip)) in clips.enumerate() {
                let decoded = (|| {
                    if cancellation.load(Ordering::Acquire) {
                        return Err("H3 VAE 任务已取消".to_owned());
                    }
                    let (inputs, shape) = prepare_video_spatial_tiles(&clip, clip_t, latent_h, latent_w, patch, &H3VideoVaeSpec::standard())?;
                    decode_tiles(&inputs, shape, patch).map(|blocks| (clip_t, blocks))
                })()
                .map_err(|error| format!("H3 temporal VAE chunk {chunk}: {error}"));
                drop(clip);
                let failed = decoded.is_err();
                if sender.send(decoded).is_err() || failed {
                    break;
                }
            }
        });
        let decoded = receiver.iter().enumerate().map(|(chunk, result)| {
            let (clip_t, blocks) = result?;
            if cancellation.load(Ordering::Acquire) {
                return Err("H3 VAE 任务已取消".to_owned());
            }
            blend_video_spatial_tiles(&blocks, clip_t, latent_h, latent_w, &spec).map(|pixels| (clip_t * spec.temporal_compression, pixels)).map_err(|error| format!("H3 temporal VAE chunk {chunk}: {error}"))
        });
        let output = blend_video_temporal_chunks(decoded, latent_h, latent_w, output_frames, on_decoded);
        drop(receiver);
        producer.join().map_err(|_| "H3 VAE producer 线程 panic".to_owned())?;
        output
    })
}

fn video_temporal_clips(latent: &[f32], shape: [usize; 3]) -> Result<impl Iterator<Item = (usize, Vec<f32>)> + use<>, String> {
    let [latent_t, latent_h, latent_w] = shape;
    if shape.contains(&0) {
        return Err(format!("H3 temporal VAE latent shape={shape:?} 必须非零"));
    }
    let spec = H3VideoVaeSpec::standard();
    let frame_elements = latent_h.checked_mul(latent_w).ok_or("H3 latent frame 大小溢出")?;
    let expected = spec.latent_channels.checked_mul(latent_t).and_then(|value| value.checked_mul(frame_elements)).ok_or("H3 latent 大小溢出")?;
    if latent.len() != expected {
        return Err(format!("H3 temporal VAE latent={}，期望 {expected}", latent.len()));
    }

    let tokens_per_chunk = spec.temporal_clip_length.div_ceil(spec.temporal_compression);
    let token_overlap = (tokens_per_chunk - spec.temporal_token_drop % tokens_per_chunk) % tokens_per_chunk;
    let mut pseudo_tokens = latent_t + spec.temporal_token_drop;
    let mut pad_tokens = (tokens_per_chunk - pseudo_tokens % tokens_per_chunk) % tokens_per_chunk;
    pseudo_tokens += pad_tokens;
    let mut chunks = pseudo_tokens / tokens_per_chunk - usize::from(spec.temporal_token_drop > 0);
    if chunks == 0 {
        pad_tokens += tokens_per_chunk;
        chunks = 1;
    }
    let padded_t = latent_t + pad_tokens;
    let mut padded = vec![0.0f32; spec.latent_channels * padded_t * frame_elements];
    for channel in 0..spec.latent_channels {
        for time in 0..padded_t {
            let source_time = time.min(latent_t - 1);
            let source_offset = (channel * latent_t + source_time) * frame_elements;
            let target_offset = (channel * padded_t + time) * frame_elements;
            padded[target_offset..target_offset + frame_elements].copy_from_slice(&latent[source_offset..source_offset + frame_elements]);
        }
    }

    Ok((0..chunks).map(move |chunk| {
        let start = chunk * tokens_per_chunk;
        let end = (start + tokens_per_chunk + token_overlap).min(padded_t);
        let clip_t = end - start;
        let mut clip = vec![0.0f32; spec.latent_channels * clip_t * frame_elements];
        for channel in 0..spec.latent_channels {
            let source = (channel * padded_t + start) * frame_elements;
            let target = channel * clip_t * frame_elements;
            clip[target..target + clip_t * frame_elements].copy_from_slice(&padded[source..source + clip_t * frame_elements]);
        }
        (clip_t, clip)
    }))
}

fn blend_video_temporal_chunks(
    decoded: impl Iterator<Item = Result<(usize, Vec<f32>), String>>,
    latent_h: usize,
    latent_w: usize,
    output_frames: usize,
    on_decoded: &mut dyn FnMut(&[f32], usize) -> Result<(), String>,
) -> Result<Vec<f32>, String> {
    let spec = H3VideoVaeSpec::standard();
    let tokens_per_chunk = spec.temporal_clip_length.div_ceil(spec.temporal_compression);
    let token_overlap = (tokens_per_chunk - spec.temporal_token_drop % tokens_per_chunk) % tokens_per_chunk;
    let frame_pre_padding = (spec.temporal_compression - spec.temporal_clip_length % spec.temporal_compression) % spec.temporal_compression;
    let frame_overlap = token_overlap * spec.temporal_compression - frame_pre_padding;
    let chunk_decoded_frames = tokens_per_chunk * spec.temporal_compression;
    let pixel_height = latent_h * spec.spatial_compression;
    let pixel_width = latent_w * spec.spatial_compression;
    let pixel_area = pixel_height * pixel_width;
    // 直接写入最终 [C,T,H,W]，不再先累积逐帧 Vec 后复制整段视频。
    let mut output = vec![0.0f32; 3 * output_frames * pixel_area];
    let mut decoded_frames = 0;
    let mut overlap: Option<(Vec<f32>, usize, usize)> = None;
    for chunk in decoded {
        let (clip_frames, pixels) = chunk?;
        let first_end = chunk_decoded_frames.min(clip_frames);
        let first_frames = first_end.saturating_sub(frame_pre_padding);
        for frame in 0..first_frames.min(output_frames.saturating_sub(decoded_frames)) {
            for channel in 0..3 {
                let source = (channel * clip_frames + frame_pre_padding + frame) * pixel_area;
                let target = (channel * output_frames + decoded_frames + frame) * pixel_area;
                let current = &pixels[source..source + pixel_area];
                let destination = &mut output[target..target + pixel_area];
                let previous = overlap.as_ref().filter(|(_, frames, start)| frame < frame_overlap.min(frames - start).min(first_frames));
                if let Some((previous, previous_frames, previous_start)) = previous {
                    let extent = frame_overlap.min(previous_frames - previous_start).min(first_frames);
                    let previous_source = (channel * previous_frames + previous_frames - extent + frame) * pixel_area;
                    let weight_b = frame as f32 / extent as f32;
                    let weight_a = 1.0 - weight_b;
                    for ((value, &old), &current) in destination.iter_mut().zip(&previous[previous_source..previous_source + pixel_area]).zip(current) {
                        *value = old * weight_a + current * weight_b;
                    }
                } else {
                    destination.copy_from_slice(current);
                }
            }
        }
        decoded_frames += first_frames;
        on_decoded(&output, decoded_frames.min(output_frames))?;
        let second_start = (chunk_decoded_frames + frame_pre_padding).min(clip_frames);
        // 下一块只读取尾部重叠帧；保留当前 canvas 可省去单独提取尾帧的复制。
        overlap = Some((pixels, clip_frames, second_start));
    }
    if let Some((pixels, frames, start)) = overlap {
        for frame in 0..(frames - start).min(output_frames.saturating_sub(decoded_frames)) {
            for channel in 0..3 {
                let source = (channel * frames + start + frame) * pixel_area;
                let target = (channel * output_frames + decoded_frames + frame) * pixel_area;
                output[target..target + pixel_area].copy_from_slice(&pixels[source..source + pixel_area]);
            }
        }
        decoded_frames += frames - start;
        on_decoded(&output, decoded_frames.min(output_frames))?;
    }
    if decoded_frames < output_frames {
        return Err(format!("H3 temporal VAE decoded frames={decoded_frames}，期望至少 {output_frames}"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_unpatch_preserves_channel_and_frame_layout() {
        let blocks = (0..3 * 28 * 64 * 64).map(|index| ((index % 997) as f32 - 498.0) / 497.0).collect::<Vec<_>>();
        let expected = super::super::unpatchify_video(&blocks, [1, 3, 28, 64, 64], [4, 16, 16]).unwrap();
        let actual = blend_video_spatial_tiles(&[blocks], 7, 4, 4, &H3VideoVaeSpec::standard()).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn temporal_output_preserves_spatial_and_temporal_overlaps() {
        let latent = vec![0.0; 24 * 12 * 4 * 20];
        for output_frames in [22, 39] {
            let mut chunk = 0;
            let pixels = decode_video_vae_tiled_temporal_with_tiles(
                &mut |tiles, shape, _| {
                    assert_eq!(tiles.len(), 2);
                    let elements = 3 * shape[0] * 4 * 64 * 256;
                    let outputs = (0..2).map(|tile| vec![(1 + tile + chunk * 10) as f32; elements]).collect();
                    chunk += 1;
                    Ok(outputs)
                },
                &latent,
                12,
                4,
                20,
                output_frames,
                [1, 2, 2],
            )
            .unwrap();
            assert_eq!(pixels.len(), 3 * output_frames * 64 * 320);
            for channel in 0..3 {
                let pixel = |frame, x| pixels[(channel * output_frames + frame) * 64 * 320 + x];
                assert_eq!(pixel(0, 0), 1.0);
                assert_eq!(pixel(0, 160), 1.5);
                assert_eq!(pixel(0, 319), 2.0);
                assert_eq!(pixel(17, 0), 1.0);
                assert_eq!(pixel(18, 0), 1.0 * 0.8 + 11.0 * 0.2);
                assert_eq!(pixel(21, 0), 1.0 * (1.0 - 0.8) + 11.0 * 0.8);
                if output_frames == 39 {
                    assert_eq!(pixel(22, 0), 11.0);
                    assert_eq!(pixel(38, 319), 12.0);
                }
            }
        }
    }

    #[test]
    fn spatial_output_rejects_rank_buffers_that_split_a_tile() {
        let error = blend_video_spatial_tiles(&[vec![0.0; 3 * 4 * 32 * 32 - 1], vec![0.0]], 1, 2, 2, &H3VideoVaeSpec::standard()).unwrap_err();
        assert!(error.contains("1 tiles × 12288 elements"), "{error}");
    }

    fn within_deadline<T: Send + 'static>(run: impl FnOnce() -> T + Send + 'static) -> T {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || sender.send(run()).is_ok());
        let result = receiver.recv_timeout(std::time::Duration::from_secs(10)).expect("H3 VAE pipeline 未在超时内结束");
        assert!(worker.join().unwrap());
        result
    }

    fn dummy_tile_outputs(tiles: &[Vec<f32>], shape: [usize; 3], chunk: usize) -> Vec<Vec<f32>> {
        let elements = 3 * shape[0] * 4 * shape[1] * 16 * shape[2] * 16;
        tiles.iter().enumerate().map(|(tile, input)| (0..elements).map(|index| input[index % input.len()] + ((index * 7 + tile * 31 + chunk * 11) % 251) as f32 / 64.0).collect()).collect()
    }

    #[test]
    fn pipeline_matches_serial_with_overlaps_and_truncated_tail() {
        within_deadline(|| {
            let latent = (0..24 * 12 * 4 * 20).map(|index| (index % 127) as f32 / 128.0).collect::<Vec<_>>();
            for frames in [22, 39] {
                let mut serial_chunk = 0;
                let serial = decode_video_vae_tiled_temporal_with_tiles(
                    &mut |tiles, shape, _| {
                        let output = dummy_tile_outputs(tiles, shape, serial_chunk);
                        serial_chunk += 1;
                        Ok(output)
                    },
                    &latent,
                    12,
                    4,
                    20,
                    frames,
                    [1, 2, 2],
                )
                .unwrap();
                let mut pipeline_chunk = 0;
                let mut decoded_progress = Vec::new();
                let pipeline = decode_video_vae_tiled_temporal_pipelined_with_tiles(
                    &mut |tiles, shape, _| {
                        let output = dummy_tile_outputs(tiles, shape, pipeline_chunk);
                        pipeline_chunk += 1;
                        Ok(output)
                    },
                    &latent,
                    12,
                    4,
                    20,
                    frames,
                    [1, 2, 2],
                    &std::sync::atomic::AtomicBool::new(false),
                    &mut |_, decoded_frames| {
                        decoded_progress.push(decoded_frames);
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(pipeline_chunk, serial_chunk);
                assert_eq!(decoded_progress.last(), Some(&frames));
                assert!(decoded_progress.windows(2).all(|pair| pair[0] <= pair[1]));
                assert!(pipeline.iter().zip(&serial).all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
                assert_eq!(pipeline.len(), serial.len());
            }
        });
    }

    #[test]
    fn pipeline_errors_and_producer_panic_release_the_sender() {
        within_deadline(|| {
            let latent = vec![0.0; 24 * 12 * 2 * 2];
            let cancellation = std::sync::atomic::AtomicBool::new(false);
            for mode in 0..3 {
                let mut calls = 0;
                let error = decode_video_vae_tiled_temporal_pipelined_with_tiles(
                    &mut |tiles, shape, _| {
                        calls += 1;
                        match mode {
                            0 if calls == 2 => Err("injected producer failure".to_owned()),
                            1 => Ok(vec![vec![0.0]]),
                            2 => panic!("injected producer panic"),
                            _ => Ok(dummy_tile_outputs(tiles, shape, calls)),
                        }
                    },
                    &latent,
                    12,
                    2,
                    2,
                    39,
                    [1, 2, 2],
                    &cancellation,
                    &mut |_, _| Ok(()),
                )
                .unwrap_err();
                let expected = ["injected producer failure", "batched spatial VAE output=1", "producer 线程 panic"][mode];
                assert!(error.contains(expected), "{error}");
                assert!((1..=2).contains(&calls), "零容量交接不允许积压更多未来块");
            }
        });
    }

    #[test]
    fn pipeline_cancellation_during_decode_allows_another_call() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc::sync_channel,
        };
        let cancellation = Arc::new(AtomicBool::new(false));
        let worker_cancellation = Arc::clone(&cancellation);
        let (entered_sender, entered_receiver) = sync_channel(0);
        let (release_sender, release_receiver) = sync_channel(0);
        let (result_sender, result_receiver) = sync_channel(1);
        let worker = std::thread::spawn(move || {
            let mut calls = 0;
            let result = decode_video_vae_tiled_temporal_pipelined_with_tiles(
                &mut move |tiles, shape, _| {
                    calls += 1;
                    if calls == 1 {
                        entered_sender.send(()).unwrap();
                        release_receiver.recv().unwrap();
                    }
                    Ok(dummy_tile_outputs(tiles, shape, calls))
                },
                &vec![0.0; 24 * 12 * 2 * 2],
                12,
                2,
                2,
                39,
                [1, 2, 2],
                &worker_cancellation,
                &mut |_, _| Ok(()),
            );
            result_sender.send(result).unwrap();
        });
        let timeout = std::time::Duration::from_secs(10);
        entered_receiver.recv_timeout(timeout).expect("producer 未进入受控 decode");
        cancellation.store(true, Ordering::Release);
        release_sender.send(()).unwrap();
        assert!(result_receiver.recv_timeout(timeout).expect("取消后 pipeline 未退出").unwrap_err().contains("已取消"));
        worker.join().unwrap();
        cancellation.store(false, Ordering::Release);
        within_deadline(move || {
            let result = decode_video_vae_tiled_temporal_pipelined_with_tiles(&mut |tiles, shape, _| Ok(dummy_tile_outputs(tiles, shape, 0)), &vec![0.0; 24 * 12 * 2 * 2], 12, 2, 2, 39, [1, 2, 2], &cancellation, &mut |_, _| Ok(())).unwrap();
            assert_eq!(result.len(), 3 * 39 * 32 * 32);
        });
    }
}
