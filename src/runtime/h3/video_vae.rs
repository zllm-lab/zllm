//! H3 视频 VAE 权重准备、解码与时空分块。

use super::{linear_bias, patchify_video, prepare_h3_video_vae_weight, unpatchify_video};
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

fn decode_video_vae_batched<B: DiffusionBackend + VaeBackend>(backend: &B, _source: &H3VideoVaeSource, global: &H3PreparedVideoVae<B::Weight>, latents: &[B::Tensor], shape: [usize; 3], patch: [usize; 3]) -> Result<B::Tensor, BackendError> {
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
fn decode_video_spatial_tiled<B: DiffusionBackend + VaeBackend>(
    backend: &B,
    source: &H3VideoVaeSource,
    global: &H3PreparedVideoVae<B::Weight>,
    clip: &[f32],
    clip_t: usize,
    latent_h: usize,
    latent_w: usize,
    patch: [usize; 3],
    spec: &H3VideoVaeSpec,
) -> Result<Vec<f32>, String> {
    let ratio = spec.spatial_compression;
    let pixel_h = latent_h * ratio;
    let pixel_w = latent_w * ratio;
    let frames = clip_t * spec.temporal_compression;
    let (y_starts, y_lengths, y_overlaps) = split_video_tiles(pixel_h, ratio)?;
    let (x_starts, x_lengths, x_overlaps) = split_video_tiles(pixel_w, ratio)?;
    let batch_tile_h = y_lengths[0];
    let batch_tile_w = x_lengths[0];
    if y_lengths.iter().any(|&length| length != batch_tile_h) || x_lengths.iter().any(|&length| length != batch_tile_w) {
        return Err("H3 VAE batched spatial tiles 必须等尺寸".to_owned());
    }
    let tile_latent_h = batch_tile_h / ratio;
    let tile_latent_w = batch_tile_w / ratio;
    let tile_count = y_starts.len() * x_starts.len();
    let mut inputs = Vec::with_capacity(tile_count);
    for (tile_y, &pixel_y) in y_starts.iter().enumerate() {
        let latent_y = pixel_y / ratio;
        for (tile_x, &pixel_x) in x_starts.iter().enumerate() {
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
            inputs.push(
                backend
                    .vae_tensor_from_f32(rows, clip_t * (tile_latent_h / patch[1]) * (tile_latent_w / patch[2]), spec.latent_channels * patch.into_iter().product::<usize>())
                    .map_err(|error| format!("上传 H3 spatial VAE tile ({tile_y},{tile_x}): {error:?}"))?,
            );
        }
    }
    let blocks = decode_video_vae_batched(backend, source, global, &inputs, [clip_t, tile_latent_h, tile_latent_w], patch).map_err(|error| format!("H3 batched spatial VAE: {error:?}"))?;
    let blocks = backend.vae_tensor_to_f32(&blocks).map_err(|error| format!("读取 H3 batched spatial VAE: {error:?}"))?;
    if blocks.len() % tile_count != 0 {
        return Err(format!("H3 batched spatial VAE output={} 不能按 {tile_count} tiles 拆分", blocks.len()));
    }
    let tile_block_elements = blocks.len() / tile_count;
    let mut block_tiles = blocks.chunks_exact(tile_block_elements);
    let mut canvas = vec![0.0f32; 3 * frames * pixel_h * pixel_w];
    let mut row_tails = Vec::<Vec<f32>>::new();
    let mut out_y = 0;

    for (tile_y, (&_pixel_y, &tile_pixel_h)) in y_starts.iter().zip(&y_lengths).enumerate() {
        let mut new_tails = Vec::<Vec<f32>>::new();
        let mut left_tail: Option<Vec<f32>> = None;
        let mut out_x = 0;
        let mut visible_h = 0;

        for (tile_x, (&_pixel_x, &tile_pixel_w)) in x_starts.iter().zip(&x_lengths).enumerate() {
            let tile_blocks = block_tiles.next().ok_or("H3 batched spatial VAE 缺少 tile output")?;
            let mut tile = unpatchify_video(tile_blocks, [1, 3, frames, tile_pixel_h, tile_pixel_w], [spec.temporal_compression, ratio, ratio])?;

            let next_bottom = if tile_y + 1 < y_starts.len() {
                let extent = y_overlaps[tile_y];
                let mut tail = vec![0.0f32; 3 * frames * extent * tile_pixel_w];
                for channel in 0..3 {
                    for frame in 0..frames {
                        for row in 0..extent {
                            let source_offset = ((channel * frames + frame) * tile_pixel_h + tile_pixel_h - extent + row) * tile_pixel_w;
                            let target_offset = ((channel * frames + frame) * extent + row) * tile_pixel_w;
                            tail[target_offset..target_offset + tile_pixel_w].copy_from_slice(&tile[source_offset..source_offset + tile_pixel_w]);
                        }
                    }
                }
                Some(tail)
            } else {
                None
            };
            let next_right = if tile_x + 1 < x_starts.len() {
                let extent = x_overlaps[tile_x];
                let mut tail = vec![0.0f32; 3 * frames * tile_pixel_h * extent];
                for channel in 0..3 {
                    for frame in 0..frames {
                        for row in 0..tile_pixel_h {
                            let source_offset = ((channel * frames + frame) * tile_pixel_h + row) * tile_pixel_w + tile_pixel_w - extent;
                            let target_offset = ((channel * frames + frame) * tile_pixel_h + row) * extent;
                            tail[target_offset..target_offset + extent].copy_from_slice(&tile[source_offset..source_offset + extent]);
                        }
                    }
                }
                Some(tail)
            } else {
                None
            };

            if tile_y > 0 {
                let extent = y_overlaps[tile_y - 1];
                let previous = &row_tails[tile_x];
                for channel in 0..3 {
                    for frame in 0..frames {
                        for row in 0..extent {
                            let weight_b = row as f32 / extent as f32;
                            let weight_a = 1.0 - weight_b;
                            for column in 0..tile_pixel_w {
                                let tile_index = ((channel * frames + frame) * tile_pixel_h + row) * tile_pixel_w + column;
                                let tail_index = ((channel * frames + frame) * extent + row) * tile_pixel_w + column;
                                tile[tile_index] = previous[tail_index] * weight_a + tile[tile_index] * weight_b;
                            }
                        }
                    }
                }
            }
            if tile_x > 0 {
                let extent = x_overlaps[tile_x - 1];
                let previous = left_tail.as_ref().ok_or("H3 VAE 缺少左侧 overlap")?;
                for channel in 0..3 {
                    for frame in 0..frames {
                        for row in 0..tile_pixel_h {
                            for column in 0..extent {
                                let weight_b = column as f32 / extent as f32;
                                let weight_a = 1.0 - weight_b;
                                let tile_index = ((channel * frames + frame) * tile_pixel_h + row) * tile_pixel_w + column;
                                let tail_index = ((channel * frames + frame) * tile_pixel_h + row) * extent + column;
                                tile[tile_index] = previous[tail_index] * weight_a + tile[tile_index] * weight_b;
                            }
                        }
                    }
                }
            }

            visible_h = tile_pixel_h - if tile_y + 1 < y_starts.len() { y_overlaps[tile_y] } else { 0 };
            let visible_w = tile_pixel_w - if tile_x + 1 < x_starts.len() { x_overlaps[tile_x] } else { 0 };
            for channel in 0..3 {
                for frame in 0..frames {
                    for row in 0..visible_h {
                        let source_offset = ((channel * frames + frame) * tile_pixel_h + row) * tile_pixel_w;
                        let target_offset = ((channel * frames + frame) * pixel_h + out_y + row) * pixel_w + out_x;
                        canvas[target_offset..target_offset + visible_w].copy_from_slice(&tile[source_offset..source_offset + visible_w]);
                    }
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
    let spec = H3VideoVaeSpec::standard();
    let frame_elements = latent_h.checked_mul(latent_w).ok_or("H3 latent frame 大小溢出")?;
    let expected = spec.latent_channels.checked_mul(latent_t).and_then(|value| value.checked_mul(frame_elements)).ok_or("H3 latent 大小溢出")?;
    if latent.len() != expected {
        return Err(format!("H3 temporal VAE latent={}，期望 {expected}", latent.len()));
    }

    let tokens_per_chunk = spec.temporal_clip_length.div_ceil(spec.temporal_compression);
    let token_overlap = (tokens_per_chunk - spec.temporal_token_drop % tokens_per_chunk) % tokens_per_chunk;
    let frame_pre_padding = (spec.temporal_compression - spec.temporal_clip_length % spec.temporal_compression) % spec.temporal_compression;
    let frame_overlap = token_overlap * spec.temporal_compression - frame_pre_padding;
    let chunk_decoded_frames = tokens_per_chunk * spec.temporal_compression;
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

    let pixel_height = latent_h * spec.spatial_compression;
    let pixel_width = latent_w * spec.spatial_compression;
    let pixel_area = pixel_height * pixel_width;
    let extract_frame = |pixels: &[f32], frames: usize, frame: usize| {
        let mut output = vec![0.0f32; 3 * pixel_area];
        for channel in 0..3 {
            let source = (channel * frames + frame) * pixel_area;
            output[channel * pixel_area..(channel + 1) * pixel_area].copy_from_slice(&pixels[source..source + pixel_area]);
        }
        output
    };

    let mut decoded = Vec::<Vec<f32>>::new();
    let mut overlap: Option<Vec<Vec<f32>>> = None;
    for chunk in 0..chunks {
        let start = chunk * tokens_per_chunk;
        let end = (start + tokens_per_chunk + token_overlap).min(padded_t);
        let clip_t = end - start;
        let mut clip = vec![0.0f32; spec.latent_channels * clip_t * frame_elements];
        for channel in 0..spec.latent_channels {
            let source = (channel * padded_t + start) * frame_elements;
            let target = channel * clip_t * frame_elements;
            clip[target..target + clip_t * frame_elements].copy_from_slice(&padded[source..source + clip_t * frame_elements]);
        }
        let clip_frames = clip_t * spec.temporal_compression;
        let pixels = decode_video_spatial_tiled(backend, source, global, &clip, clip_t, latent_h, latent_w, patch, &spec).map_err(|error| format!("H3 temporal VAE chunk {chunk}: {error}"))?;
        let first_end = chunk_decoded_frames.min(clip_frames);
        let mut first = (frame_pre_padding..first_end).map(|frame| extract_frame(&pixels, clip_frames, frame)).collect::<Vec<_>>();
        if let Some(previous) = overlap.take() {
            let extent = frame_overlap.min(previous.len()).min(first.len());
            for frame in 0..extent {
                let weight_b = frame as f32 / extent as f32;
                let weight_a = 1.0 - weight_b;
                for (value, &old) in first[frame].iter_mut().zip(&previous[previous.len() - extent + frame]) {
                    *value = old * weight_a + *value * weight_b;
                }
            }
        }
        decoded.extend(first);
        let second_start = (chunk_decoded_frames + frame_pre_padding).min(clip_frames);
        overlap = Some((second_start..clip_frames).map(|frame| extract_frame(&pixels, clip_frames, frame)).collect());
    }
    if let Some(overlap) = overlap {
        decoded.extend(overlap);
    }
    if decoded.len() < output_frames {
        return Err(format!("H3 temporal VAE decoded frames={}，期望至少 {output_frames}", decoded.len()));
    }
    decoded.truncate(output_frames);
    let mut output = vec![0.0f32; 3 * output_frames * pixel_area];
    for (frame, pixels) in decoded.iter().enumerate() {
        for channel in 0..3 {
            let target = (channel * output_frames + frame) * pixel_area;
            output[target..target + pixel_area].copy_from_slice(&pixels[channel * pixel_area..(channel + 1) * pixel_area]);
        }
    }
    Ok(output)
}
