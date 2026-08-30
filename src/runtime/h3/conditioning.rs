//! H3 参考媒体条件编码；这里只编排算法，设备存储与 kernel 由 backend capability 提供。

use crate::{
    backend::{Backend, BackendError, DiffusionBackend, GqaPrefillBackend, LinearWeight, VaeBackend, VisionBackend},
    runtime::qwen3_vl::{Qwen3VlConfig, Qwen3VlH3VisualInput, Qwen3VlTextLayer, Qwen3VlTextVisual, Qwen3VlVisualKind, qwen3_vl_h3_input, qwen3_vl_mrope_table, qwen3_vl_multimodal_text_hidden_resident},
    tokenizer::Tokenizer,
    vae::{Conv1dSpec, Conv3dSpec, H3AudioVaeSpec, H3VideoVaeSpec},
    vision::{RgbImage, pillow_bicubic_resize},
    weight::{
        container::safetensor::TensorData,
        model::h3_vae::{H3AudioConvWeights, H3AudioEncoderUnitWeights, H3AudioPreBlockWeights, H3AudioSnakeWeights, H3AudioVaeSource, H3VideoConvWeights, H3VideoEncoderResnetWeights, H3VideoNormWeights, H3VideoVaeSource},
        model::qwen3_vl::Qwen3VlWeights,
    },
};

pub const H3_REFERENCE_SHORT_EDGE: usize = 2048;
pub const H3_REFERENCE_ALIGNMENT: usize = 32;
pub const H3_VIDEO_CLIP_FRAMES: usize = 17;
pub const H3_VIDEO_CLIP_TOKEN_DROP: usize = 3;

pub struct H3VisualVaeInput {
    /// `[channels, time, height, width]` channel-major F32。
    pub values: Vec<f32>,
    pub shape: [usize; 3],
}

pub struct H3VisualVaeClip {
    pub frame_range: std::ops::Range<usize>,
    pub drop_latent_tokens: usize,
    pub input: H3VisualVaeInput,
}

pub fn h3_preprocess_reference_image(image: &RgbImage) -> Result<H3VisualVaeInput, BackendError> {
    h3_preprocess_reference_frames(std::slice::from_ref(image))
}

pub fn h3_preprocess_reference_frames(frames: &[RgbImage]) -> Result<H3VisualVaeInput, BackendError> {
    let first = frames.first().ok_or_else(|| crate::runtime::compute_error("H3 reference video 没有帧"))?;
    if first.width == 0 || first.height == 0 {
        return Err(crate::runtime::compute_error("H3 reference frame 尺寸为 0"));
    }
    let (width, height) = h3_reference_size(first.width, first.height)?;
    let frame_pixels = height.checked_mul(width).ok_or_else(|| crate::runtime::compute_error("H3 reference frame pixels 溢出"))?;
    let mut values = vec![0.0; 3usize.checked_mul(frames.len()).and_then(|value| value.checked_mul(frame_pixels)).ok_or_else(|| crate::runtime::compute_error("H3 reference video values 溢出"))?];
    for (time, frame) in frames.iter().enumerate() {
        if frame.width == 0 || frame.height == 0 || frame.pixels.len() != frame.width.checked_mul(frame.height).and_then(|value| value.checked_mul(3)).ok_or_else(|| crate::runtime::compute_error("H3 reference source pixels 溢出"))? {
            return Err(crate::runtime::compute_error(format!("H3 reference frame {time} pixels={} 与 {}x{} 不匹配", frame.pixels.len(), frame.width, frame.height)));
        }
        let source = image::RgbImage::from_raw(frame.width as u32, frame.height as u32, frame.pixels.clone()).ok_or_else(|| crate::runtime::compute_error(format!("H3 reference frame {time} 构造 RGB 失败")))?;
        let resized = pillow_bicubic_resize(&source, width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let pixel = resized.get_pixel(x as u32, y as u32).0;
                for channel in 0..3 {
                    let destination = (channel * frames.len() + time) * frame_pixels + y * width + x;
                    values[destination] = pixel[channel] as f32 / 127.5 - 1.0;
                }
            }
        }
    }
    Ok(H3VisualVaeInput { values, shape: [frames.len(), height, width] })
}

pub fn h3_preprocess_reference_video(frames: &[RgbImage]) -> Result<Vec<H3VisualVaeClip>, BackendError> {
    if frames.is_empty() {
        return Err(crate::runtime::compute_error("H3 reference video 没有帧"));
    }
    let mut clips = Vec::new();
    let mut start = 0usize;
    while start < frames.len() {
        let end = start.saturating_add(H3_VIDEO_CLIP_FRAMES).min(frames.len());
        let input = h3_preprocess_reference_frames(&frames[start..end])?;
        clips.push(H3VisualVaeClip { frame_range: start..end, drop_latent_tokens: usize::from(start != 0) * H3_VIDEO_CLIP_TOKEN_DROP, input });
        start = end;
    }
    Ok(clips)
}

fn h3_reference_size(width: usize, height: usize) -> Result<(usize, usize), BackendError> {
    let short = width.min(height);
    if short == 0 {
        return Err(crate::runtime::compute_error("H3 reference image short edge 为 0"));
    }
    let scale = H3_REFERENCE_SHORT_EDGE as f64 / short as f64;
    let aligned = |value: usize| -> Result<usize, BackendError> {
        let scaled = value as f64 * scale;
        if !scaled.is_finite() || scaled > usize::MAX as f64 {
            return Err(crate::runtime::compute_error("H3 reference resize 尺寸溢出"));
        }
        Ok((((scaled / H3_REFERENCE_ALIGNMENT as f64).round() as usize).max(1)) * H3_REFERENCE_ALIGNMENT)
    };
    Ok((aligned(width)?, aligned(height)?))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3ReferenceKind {
    Image,
    Video,
    Audio,
}

pub struct H3QwenVisual<T> {
    pub embedding: T,
    pub deepstack: Vec<T>,
    pub kind: H3ReferenceKind,
    pub grid: crate::vision::VisionGrid,
    pub merge_size: usize,
}

impl<T> H3QwenVisual<T> {
    pub fn new(output: crate::runtime::qwen3_vl::Qwen3VlVisionOutput<T>, kind: H3ReferenceKind, grid: crate::vision::VisionGrid, merge_size: usize) -> Result<Self, BackendError> {
        if matches!(kind, H3ReferenceKind::Audio) || grid.temporal == 0 || grid.height == 0 || grid.width == 0 || merge_size == 0 {
            return Err(crate::runtime::compute_error(format!("H3 Qwen visual kind={kind:?} grid={grid:?} merge={merge_size} 非法")));
        }
        Ok(Self { embedding: output.embedding, deepstack: output.deepstack, kind, grid, merge_size })
    }
}

pub struct H3QwenTextCondition<T> {
    pub hidden: T,
    pub token_ids: Vec<u32>,
}

pub struct H3ReferenceEncoding<T> {
    pub source_index: usize,
    pub kind: H3ReferenceKind,
    /// Qwen 视觉编码由调用方注入文本 encoder；纯音频为 `None`。
    pub qwen_visual: Option<H3QwenVisual<T>>,
    pub video_condition: Option<H3VideoCondition<T>>,
    pub audio_condition: Option<H3AudioCondition<T>>,
}

pub enum H3PackedReferenceCondition<T> {
    Audio { source_index: usize, condition: H3AudioCondition<T> },
    Video { source_index: usize, condition: H3VideoCondition<T> },
}

pub struct H3OrderedReferences<T> {
    pub conditions: Vec<H3PackedReferenceCondition<T>>,
}

impl<T> H3ReferenceEncoding<T> {
    /// `deepstack_features` 由调用方从 `Qwen3VlVisionConfig::deepstack_visual_indexes` 长度推导。
    pub fn validate(&self, deepstack_features: usize) -> Result<(), BackendError> {
        if self.qwen_visual.as_ref().is_some_and(|visual| visual.deepstack.len() != deepstack_features) {
            return Err(crate::runtime::compute_error(format!("H3 Qwen visual DeepStack={}，期望 {deepstack_features}", self.qwen_visual.as_ref().map_or(0, |visual| visual.deepstack.len()))));
        }
        match self.kind {
            H3ReferenceKind::Image if self.qwen_visual.is_none() || self.video_condition.is_none() || self.audio_condition.is_some() => {
                Err(crate::runtime::compute_error("H3 image reference 必须同时包含 Qwen visual 与 VisualVAE condition"))
            }
            H3ReferenceKind::Video if self.qwen_visual.is_none() || self.video_condition.is_none() => Err(crate::runtime::compute_error("H3 video reference 必须同时包含 Qwen visual 与 VisualVAE condition")),
            H3ReferenceKind::Audio if self.qwen_visual.is_some() || self.video_condition.is_some() || self.audio_condition.is_none() => Err(crate::runtime::compute_error("H3 audio reference 只能包含 AudioVAE condition")),
            _ => Ok(()),
        }
    }
}

/// 使用常驻文本层生成 H3 conditioning；Qwen visual 与 Ref2VA condition 可位于不同设备。
#[allow(clippy::too_many_arguments)]
pub fn h3_qwen_text_condition_resident<B>(
    backend: &B,
    config: &Qwen3VlConfig,
    weights: &Qwen3VlWeights,
    tokenizer: &Tokenizer,
    prompt: &str,
    qwen_visuals: &[(usize, H3QwenVisual<B::Tensor>)],
    layers: usize,
    resident: &[Qwen3VlTextLayer<B::Weight>],
) -> Result<H3QwenTextCondition<B::Tensor>, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    for (source_index, visual) in qwen_visuals {
        if visual.merge_size == 0 || !visual.grid.height.is_multiple_of(visual.merge_size) || !visual.grid.width.is_multiple_of(visual.merge_size) {
            return Err(crate::runtime::compute_error(format!("H3 Qwen visual {source_index} grid/merge 非法")));
        }
        let expected_rows =
            visual.grid.temporal.checked_mul(visual.grid.height / visual.merge_size).and_then(|value| value.checked_mul(visual.grid.width / visual.merge_size)).ok_or_else(|| crate::runtime::compute_error("H3 Qwen visual rows 溢出"))?;
        if backend.token_rows(&visual.embedding) != expected_rows || backend.token_cols(&visual.embedding) != config.hidden_size {
            return Err(crate::runtime::compute_error(format!("H3 Qwen visual {source_index} embedding shape 不匹配，期望 [{expected_rows},{}]", config.hidden_size)));
        }
        if visual.deepstack.iter().any(|feature| backend.token_rows(feature) != expected_rows || backend.token_cols(feature) != config.hidden_size) {
            return Err(crate::runtime::compute_error(format!("H3 Qwen visual {source_index} DeepStack shape 不匹配")));
        }
    }
    let visual_inputs = qwen_visuals
        .iter()
        .map(|(_, visual)| Qwen3VlH3VisualInput {
            kind: match visual.kind {
                H3ReferenceKind::Image => Qwen3VlVisualKind::Image,
                H3ReferenceKind::Video => Qwen3VlVisualKind::Video,
                H3ReferenceKind::Audio => unreachable!("纯音频没有 Qwen visual"),
            },
            grid: visual.grid,
            merge_size: visual.merge_size,
        })
        .collect::<Vec<_>>();
    let input = qwen3_vl_h3_input(tokenizer, config, prompt, &visual_inputs).map_err(crate::runtime::compute_error)?;
    if input.visual_tokens.len() != qwen_visuals.len() {
        return Err(crate::runtime::compute_error("H3 Qwen visual ranges 数量不匹配"));
    }
    let embedding = weights.embedding_rows_f32(&input.token_ids).map_err(BackendError::ExpertLoad)?;
    let hidden = backend.vision_tensor_from_f32(&embedding, input.token_ids.len(), config.hidden_size)?;
    let rope = qwen3_vl_mrope_table(config, &input.position_ids).map_err(crate::runtime::compute_error)?;
    let visuals = qwen_visuals.iter().zip(&input.visual_tokens).map(|((_, visual), tokens)| Qwen3VlTextVisual { tokens: tokens.clone(), embedding: &visual.embedding, deepstack: &visual.deepstack }).collect::<Vec<_>>();
    let hidden = qwen3_vl_multimodal_text_hidden_resident(backend, config, weights, resident, hidden, &rope, layers, &visuals)?;
    Ok(H3QwenTextCondition { hidden, token_ids: input.token_ids })
}

pub fn build_h3_reference_layout<T>(text_len: usize, latent_t: usize, latent_h: usize, latent_w: usize, audio_t: usize, patch_size: [usize; 3], references: &H3OrderedReferences<T>) -> Result<super::H3PackedLayout, String> {
    let mut layout = super::build_packed_layout(text_len, latent_t, latent_h, latent_w, audio_t, patch_size, &[])?;
    if references.conditions.is_empty() {
        return Ok(layout);
    }
    let (_, target_width_grid) = super::spatial_frame_grid(latent_h, latent_w, patch_size[1], patch_size[2]);
    let mut positions = Vec::new();
    let mut segments = Vec::with_capacity(references.conditions.len());
    for reference in &references.conditions {
        let start = text_len + positions.len();
        let kind = match reference {
            H3PackedReferenceCondition::Video { condition, .. } => {
                if condition.patch_shape.contains(&0) || condition.latent_shape[0] % condition.patch_shape[0] != 0 || condition.latent_shape[1] % condition.patch_shape[1] != 0 || condition.latent_shape[2] % condition.patch_shape[2] != 0 {
                    return Err(format!("H3 reference video latent={:?} 不能被 patch={:?} 整除", condition.latent_shape, condition.patch_shape));
                }
                let grid_t = condition.latent_shape[0] / condition.patch_shape[0];
                let (frame_grid, _) = super::spatial_frame_grid(condition.latent_shape[1], condition.latent_shape[2], condition.patch_shape[1], condition.patch_shape[2]);
                let mut time = text_len as f32;
                for span in super::video_time_spans(grid_t) {
                    positions.extend(frame_grid.iter().map(|&[height, width]| [time, height, width]));
                    time += span;
                }
                super::H3SegmentKind::VideoCondition
            }
            H3PackedReferenceCondition::Audio { condition, .. } => {
                if condition.batch == 0 || condition.time == 0 {
                    return Err("H3 reference audio batch/time 必须非零".to_owned());
                }
                for channel in 0..condition.batch {
                    let width = if channel == 0 { target_width_grid[0] } else { target_width_grid[target_width_grid.len() - 1] };
                    positions.extend((0..condition.time).map(|frame| [text_len as f32 + frame as f32, 0.0, width]));
                }
                super::H3SegmentKind::AudioCondition
            }
        };
        segments.push(super::H3PackedSegment { rows: start..text_len + positions.len(), kind });
    }
    let shift = positions.len();
    let mut position_ids = Vec::with_capacity(layout.position_ids.len() + shift);
    position_ids.extend_from_slice(&layout.position_ids[..text_len]);
    position_ids.extend(positions);
    position_ids.extend_from_slice(&layout.position_ids[text_len..]);
    let mut packed_segments = Vec::with_capacity(layout.segments.len() + segments.len());
    packed_segments.push(layout.segments[0].clone());
    packed_segments.extend(segments);
    packed_segments.extend(layout.segments.iter().skip(1).map(|segment| super::H3PackedSegment { rows: segment.rows.start + shift..segment.rows.end + shift, kind: segment.kind }));
    layout.seq_len += shift;
    layout.position_ids = position_ids;
    layout.segments = packed_segments;
    layout.audio_rows = layout.audio_rows.start + shift..layout.audio_rows.end + shift;
    layout.video_rows = layout.video_rows.start + shift..layout.video_rows.end + shift;
    Ok(layout)
}

/// 将 safetensors tensor 直接准备成 backend resident 权重；卷积也按 `[out, flattened input]` 处理。
pub fn prepare_h3_tensor<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let rows = tensor.shape.first().copied().ok_or_else(|| crate::runtime::compute_error(format!("H3 weight {} shape 为空", tensor.name)))?;
    let elements = tensor.shape.iter().try_fold(1usize, |count, value| count.checked_mul(*value)).ok_or_else(|| crate::runtime::compute_error(format!("H3 weight {} shape 溢出", tensor.name)))?;
    let cols = elements.checked_div(rows).filter(|value| *value > 0).ok_or_else(|| crate::runtime::compute_error(format!("H3 weight {} shape={:?} 非法", tensor.name, tensor.shape)))?;
    match tensor.dtype.as_str() {
        "BF16" => {
            if tensor.data.len() != elements.checked_mul(2).ok_or_else(|| crate::runtime::compute_error("H3 BF16 bytes 溢出"))? {
                return Err(crate::runtime::compute_error(format!("H3 BF16 weight {} bytes={}，期望 {}", tensor.name, tensor.data.len(), elements * 2)));
            }
            backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, cols)
        }
        "F16" => {
            if tensor.data.len() != elements.checked_mul(2).ok_or_else(|| crate::runtime::compute_error("H3 F16 bytes 溢出"))? {
                return Err(crate::runtime::compute_error(format!("H3 F16 weight {} bytes={}，期望 {}", tensor.name, tensor.data.len(), elements * 2)));
            }
            let values = tensor.data.chunks_exact(2).map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            backend.prepare_weight(LinearWeight::F16(&values), rows, cols)
        }
        "F32" => {
            if tensor.data.len() != elements.checked_mul(4).ok_or_else(|| crate::runtime::compute_error("H3 F32 bytes 溢出"))? {
                return Err(crate::runtime::compute_error(format!("H3 F32 weight {} bytes={}，期望 {}", tensor.name, tensor.data.len(), elements * 4)));
            }
            let values = tensor.data.chunks_exact(4).map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])).collect::<Vec<_>>();
            backend.prepare_weight(LinearWeight::F32(&values), rows, cols)
        }
        dtype => Err(crate::runtime::compute_error(format!("H3 weight {} dtype={dtype} 暂不支持", tensor.name))),
    }
}

pub fn prepare_h3_video_encoder<B>(backend: &B, source: &H3VideoVaeSource, spec: &H3VideoVaeSpec) -> Result<PreparedH3VideoEncoder<B::Weight>, BackendError>
where
    B: Backend,
{
    let global = source.load_encoder_global().map_err(BackendError::ExpertLoad)?;
    let mut levels = Vec::with_capacity(spec.encoder_channels.len());
    for level in 0..spec.encoder_channels.len() {
        let weights = source.load_encoder_level(level).map_err(BackendError::ExpertLoad)?;
        let blocks = weights.blocks.iter().map(|block| prepare_video_residual(backend, block)).collect::<Result<Vec<_>, _>>()?;
        let stride = [spec.encoder_time_strides[level], spec.encoder_space_strides[level], spec.encoder_space_strides[level]];
        let downsample = weights.downsample.as_ref().map(|conv| prepare_video_conv(backend, conv, stride, [usize::from(stride[1] > 1), usize::from(stride[2] > 1)])).transpose()?;
        levels.push(PreparedH3VideoLevel { blocks, downsample });
    }
    let latent_mean = global.latent_mean.to_f32().map_err(crate::runtime::compute_error)?;
    let latent_std = global.latent_std.to_f32().map_err(crate::runtime::compute_error)?;
    let latent_channels = latent_mean.len();
    if latent_std.len() != latent_channels {
        return Err(crate::runtime::compute_error(format!("H3 video latent mean/std={latent_channels}/{} 不一致", latent_std.len())));
    }
    Ok(PreparedH3VideoEncoder {
        conv_in: prepare_video_conv(backend, &global.conv_in, [1, 1, 1], [0, 0])?,
        levels,
        middle: Vec::new(),
        norm_out: prepare_video_norm(backend, &global.norm_out)?,
        conv_out: prepare_video_conv(backend, &global.conv_out, [1, 1, 1], [0, 0])?,
        quant_conv: prepare_video_conv(backend, &global.quant_conv, [1, 1, 1], [0, 0])?,
        latent_channels,
        latent_mean,
        latent_std,
        patch: spec.condition_patch,
        posterior_seed: 42,
    })
}

fn prepare_video_conv<B: Backend>(backend: &B, conv: &H3VideoConvWeights, stride: [usize; 3], spatial_pad_after: [usize; 2]) -> Result<PreparedH3VideoConv<B::Weight>, BackendError> {
    if conv.weight.shape.len() != 5 {
        return Err(crate::runtime::compute_error(format!("H3 video conv {} shape={:?} 不是 rank-5", conv.weight.name, conv.weight.shape)));
    }
    let output_channels = conv.weight.shape[0];
    let input_channels = conv.weight.shape[1];
    let kernel = [conv.weight.shape[2], conv.weight.shape[3], conv.weight.shape[4]];
    Ok(PreparedH3VideoConv {
        weight: prepare_h3_tensor(backend, &conv.weight)?,
        bias: Some(prepare_h3_tensor(backend, &conv.bias)?),
        input_channels,
        output_channels,
        kernel,
        stride,
        padding: [kernel[0].saturating_sub(1), usize::from(spatial_pad_after[0] == 0) * (kernel[1] / 2), usize::from(spatial_pad_after[1] == 0) * (kernel[2] / 2)],
        causal: kernel[0] > 1,
        spatial_pad_after,
    })
}

fn prepare_video_norm<B: Backend>(backend: &B, norm: &H3VideoNormWeights) -> Result<PreparedH3VideoNorm<B::Weight>, BackendError> {
    let channels = norm.weight.shape.iter().product::<usize>();
    Ok(PreparedH3VideoNorm { weight: prepare_h3_tensor(backend, &norm.weight)?, bias: prepare_h3_tensor(backend, &norm.bias)?, groups: 32.min(channels), eps: 1.0e-6 })
}

fn prepare_video_residual<B: Backend>(backend: &B, block: &H3VideoEncoderResnetWeights) -> Result<PreparedH3VideoResidual<B::Weight>, BackendError> {
    Ok(PreparedH3VideoResidual {
        norm1: prepare_video_norm(backend, &block.norm1)?,
        conv1: prepare_video_conv(backend, &block.conv1, [1, 1, 1], [0, 0])?,
        norm2: prepare_video_norm(backend, &block.norm2)?,
        conv2: prepare_video_conv(backend, &block.conv2, [1, 1, 1], [0, 0])?,
        shortcut: block.shortcut.as_ref().map(|conv| prepare_video_conv(backend, conv, [1, 1, 1], [0, 0])).transpose()?,
    })
}

pub fn prepare_h3_audio_encoder<B>(backend: &B, source: &H3AudioVaeSource, spec: &H3AudioVaeSpec) -> Result<PreparedH3AudioEncoder<B::Weight>, BackendError>
where
    B: Backend,
{
    let global = source.load_encoder_global().map_err(BackendError::ExpertLoad)?;
    let mut stages = Vec::with_capacity(spec.encoder_rates.len());
    for (stage, &rate) in spec.encoder_rates.iter().enumerate() {
        let weights = source.load_encoder_stage(stage).map_err(BackendError::ExpertLoad)?;
        let residuals = weights
            .units
            .iter()
            .enumerate()
            .map(|(unit, weights)| {
                // unit 数量超过 dilation 表属于权重/规格不匹配，不能静默回退 1。
                let dilation = [1, 3, 9].get(unit).copied().ok_or_else(|| crate::runtime::compute_error(format!("H3 audio encoder stage {stage} unit={unit} 超出 dilation 表 [1,3,9]")))?;
                prepare_audio_unit(backend, weights, dilation)
            })
            .collect::<Result<Vec<_>, _>>()?;
        stages.push(PreparedH3AudioStage { residuals, snake: prepare_audio_snake(backend, &weights.activation)?, downsample: prepare_audio_conv(backend, &weights.downsample, rate, 1, rate.div_ceil(2))? });
    }
    let latent_mean = global.latent_mean.to_f32().map_err(crate::runtime::compute_error)?;
    let latent_std = global.latent_std.to_f32().map_err(crate::runtime::compute_error)?;
    let latent_channels = latent_mean.len();
    if latent_channels == 0 || latent_std.len() != latent_channels {
        return Err(crate::runtime::compute_error(format!("H3 audio latent mean/std={latent_channels}/{} 不一致", latent_std.len())));
    }
    let attention = prepare_audio_pre_block(backend, &global.pre_block)?;
    Ok(PreparedH3AudioEncoder { conv_in: prepare_audio_conv(backend, &global.input, 1, 1, 3)?, stages, final_snake: prepare_audio_snake(backend, &global.final_activation)?, attention, latent_channels, latent_mean, latent_std })
}

fn prepare_audio_conv<B: Backend>(backend: &B, conv: &H3AudioConvWeights, stride: usize, dilation: usize, padding: usize) -> Result<PreparedH3AudioConv<B::Weight>, BackendError> {
    if conv.weight_v.shape.len() != 3 {
        return Err(crate::runtime::compute_error(format!("H3 audio conv {} shape={:?} 不是 rank-3", conv.weight_v.name, conv.weight_v.shape)));
    }
    Ok(PreparedH3AudioConv {
        weight_g: conv.weight_g.as_ref().map(|weight| prepare_h3_tensor(backend, weight)).transpose()?,
        weight_v: prepare_h3_tensor(backend, &conv.weight_v)?,
        bias: conv.bias.as_ref().map(|bias| prepare_h3_tensor(backend, bias)).transpose()?,
        input_channels: conv.weight_v.shape[1],
        output_channels: conv.weight_v.shape[0],
        kernel: conv.weight_v.shape[2],
        stride,
        dilation,
        padding,
    })
}

fn prepare_audio_snake<B: Backend>(backend: &B, snake: &H3AudioSnakeWeights) -> Result<B::Weight, BackendError> {
    prepare_h3_tensor(backend, &snake.alpha)
}

fn prepare_audio_unit<B: Backend>(backend: &B, unit: &H3AudioEncoderUnitWeights, dilation: usize) -> Result<PreparedH3AudioResidual<B::Weight>, BackendError> {
    Ok(PreparedH3AudioResidual {
        snake1: prepare_audio_snake(backend, &unit.activation1)?,
        conv1: prepare_audio_conv(backend, &unit.conv1, 1, dilation, 3 * dilation)?,
        snake2: prepare_audio_snake(backend, &unit.activation2)?,
        conv2: prepare_audio_conv(backend, &unit.conv2, 1, 1, 0)?,
    })
}

fn prepare_audio_pre_block<B: Backend>(backend: &B, block: &H3AudioPreBlockWeights) -> Result<PreparedH3AudioAttention<B::Weight>, BackendError> {
    if block.qkv_weight.shape.len() != 2 || block.qkv_weight.shape[0] % 3 != 0 {
        return Err(crate::runtime::compute_error(format!("H3 audio pre-block qkv shape={:?} 非法", block.qkv_weight.shape)));
    }
    let attention_columns = block.qkv_weight.shape[0] / 3;
    Ok(PreparedH3AudioAttention {
        qkv: PreparedH3Linear { weight: prepare_h3_tensor(backend, &block.qkv_weight)?, bias: Some(prepare_h3_tensor(backend, &block.qkv_bias)?) },
        output: PreparedH3Linear { weight: prepare_h3_tensor(backend, &block.attention_output_weight)?, bias: Some(prepare_h3_tensor(backend, &block.attention_output_bias)?) },
        heads: 1,
        head_dim: attention_columns,
    })
}

pub struct PreparedH3Linear<W> {
    pub weight: W,
    pub bias: Option<W>,
}

pub struct PreparedH3VideoConv<W> {
    pub weight: W,
    pub bias: Option<W>,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: [usize; 3],
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub causal: bool,
    pub spatial_pad_after: [usize; 2],
}

pub struct PreparedH3VideoNorm<W> {
    pub weight: W,
    pub bias: W,
    pub groups: usize,
    pub eps: f32,
}

pub struct PreparedH3VideoResidual<W> {
    pub norm1: PreparedH3VideoNorm<W>,
    pub conv1: PreparedH3VideoConv<W>,
    pub norm2: PreparedH3VideoNorm<W>,
    pub conv2: PreparedH3VideoConv<W>,
    pub shortcut: Option<PreparedH3VideoConv<W>>,
}

pub struct PreparedH3VideoLevel<W> {
    pub blocks: Vec<PreparedH3VideoResidual<W>>,
    pub downsample: Option<PreparedH3VideoConv<W>>,
}

pub struct PreparedH3VideoEncoder<W> {
    pub conv_in: PreparedH3VideoConv<W>,
    pub levels: Vec<PreparedH3VideoLevel<W>>,
    pub middle: Vec<PreparedH3VideoResidual<W>>,
    pub norm_out: PreparedH3VideoNorm<W>,
    pub conv_out: PreparedH3VideoConv<W>,
    pub quant_conv: PreparedH3VideoConv<W>,
    pub latent_channels: usize,
    pub latent_mean: Vec<f32>,
    pub latent_std: Vec<f32>,
    pub patch: [usize; 3],
    pub posterior_seed: u64,
}

pub struct H3VideoCondition<T> {
    pub tensor: T,
    pub latent_shape: [usize; 3],
    pub patch_shape: [usize; 3],
}

pub struct PreparedH3AudioConv<W> {
    pub weight_g: Option<W>,
    pub weight_v: W,
    pub bias: Option<W>,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
}

pub struct PreparedH3AudioResidual<W> {
    pub snake1: W,
    pub conv1: PreparedH3AudioConv<W>,
    pub snake2: W,
    pub conv2: PreparedH3AudioConv<W>,
}

pub struct PreparedH3AudioStage<W> {
    pub residuals: Vec<PreparedH3AudioResidual<W>>,
    pub snake: W,
    pub downsample: PreparedH3AudioConv<W>,
}

pub struct PreparedH3AudioAttention<W> {
    pub qkv: PreparedH3Linear<W>,
    pub output: PreparedH3Linear<W>,
    pub heads: usize,
    pub head_dim: usize,
}

pub struct PreparedH3AudioEncoder<W> {
    pub conv_in: PreparedH3AudioConv<W>,
    pub stages: Vec<PreparedH3AudioStage<W>>,
    pub final_snake: W,
    pub attention: PreparedH3AudioAttention<W>,
    pub latent_channels: usize,
    pub latent_mean: Vec<f32>,
    pub latent_std: Vec<f32>,
}

pub struct H3AudioCondition<T> {
    pub tensor: T,
    pub batch: usize,
    pub time: usize,
    pub channels: usize,
}

pub fn h3_encode_video_condition<B>(backend: &B, encoder: &PreparedH3VideoEncoder<B::Weight>, pixels: Vec<f32>, shape: [usize; 3]) -> Result<H3VideoCondition<B::Tensor>, BackendError>
where
    B: VaeBackend + DiffusionBackend,
{
    let expected = encoder.conv_in.input_channels.checked_mul(shape.into_iter().product::<usize>()).ok_or_else(|| crate::runtime::compute_error("H3 video input 大小溢出"))?;
    if pixels.len() != expected {
        return Err(crate::runtime::compute_error(format!("H3 video input values={}，期望 channels={} shape={shape:?}", pixels.len(), encoder.conv_in.input_channels)));
    }
    validate_video_encoder(encoder)?;
    let mut hidden = backend.vae_tensor_from_f32(pixels, encoder.conv_in.input_channels, shape.into_iter().product())?;
    let (next, mut current_shape) = video_conv(backend, &hidden, shape, &encoder.conv_in)?;
    hidden = next;
    for level in &encoder.levels {
        for block in &level.blocks {
            hidden = video_residual(backend, &hidden, current_shape, block)?;
        }
        if let Some(downsample) = &level.downsample {
            (hidden, current_shape) = video_conv(backend, &hidden, current_shape, downsample)?;
        }
    }
    for block in &encoder.middle {
        hidden = video_residual(backend, &hidden, current_shape, block)?;
    }
    hidden = backend.group_norm_time_isolated(&hidden, &encoder.norm_out.weight, &encoder.norm_out.bias, current_shape[0], encoder.norm_out.groups, encoder.norm_out.eps)?;
    hidden = backend.silu(&hidden)?;
    (hidden, current_shape) = video_conv(backend, &hidden, current_shape, &encoder.conv_out)?;
    (hidden, current_shape) = video_conv(backend, &hidden, current_shape, &encoder.quant_conv)?;

    let moments = backend.vae_tensor_to_f32(&hidden)?;
    let latent = sample_video_posterior(&moments, current_shape, &encoder.latent_mean, &encoder.latent_std, encoder.posterior_seed)?;
    let (patches, rows, columns) = patchify_video(&latent, current_shape, encoder.patch)?;
    let tensor = backend.vae_tensor_from_f32(patches, rows, columns)?;
    Ok(H3VideoCondition { tensor, latent_shape: current_shape, patch_shape: encoder.patch })
}

/// 合并 17 帧分片；除首片外按官方规则丢弃开头 3 个 latent time token。
pub fn h3_join_video_conditions<B>(backend: &B, clips: Vec<(H3VideoCondition<B::Tensor>, usize)>) -> Result<H3VideoCondition<B::Tensor>, BackendError>
where
    B: VaeBackend,
{
    let mut clips = clips.into_iter();
    let (first, first_drop) = clips.next().ok_or_else(|| crate::runtime::compute_error("H3 video condition clips 为空"))?;
    if first_drop != 0 {
        return Err(crate::runtime::compute_error("H3 video condition 首片不能丢弃 latent token"));
    }
    let columns = backend.token_cols(&first.tensor);
    let patch_shape = first.patch_shape;
    let mut latent_shape = first.latent_shape;
    let mut values = backend.vae_tensor_to_f32(&first.tensor)?;
    if values.len() != backend.token_rows(&first.tensor).checked_mul(columns).ok_or_else(|| crate::runtime::compute_error("H3 first video condition 大小溢出"))? {
        return Err(crate::runtime::compute_error("H3 first video condition tensor shape 不匹配"));
    }
    for (clip, drop_tokens) in clips {
        if clip.patch_shape != patch_shape || backend.token_cols(&clip.tensor) != columns {
            return Err(crate::runtime::compute_error("H3 video condition clip patch/columns 不一致"));
        }
        let grid_height = clip.latent_shape[1] / patch_shape[1];
        let grid_width = clip.latent_shape[2] / patch_shape[2];
        let drop_rows = drop_tokens.checked_mul(grid_height).and_then(|value| value.checked_mul(grid_width)).ok_or_else(|| crate::runtime::compute_error("H3 video condition drop rows 溢出"))?;
        let rows = backend.token_rows(&clip.tensor);
        if drop_rows >= rows || clip.latent_shape[0] < drop_tokens.checked_mul(patch_shape[0]).ok_or_else(|| crate::runtime::compute_error("H3 video condition drop time 溢出"))? {
            return Err(crate::runtime::compute_error(format!("H3 video condition drop={drop_tokens} 超过 clip rows={rows}")));
        }
        let clip_values = backend.vae_tensor_to_f32(&clip.tensor)?;
        let start = drop_rows.checked_mul(columns).ok_or_else(|| crate::runtime::compute_error("H3 video condition drop offset 溢出"))?;
        values.extend_from_slice(clip_values.get(start..).ok_or_else(|| crate::runtime::compute_error("H3 video condition drop offset 越界"))?);
        latent_shape[0] = latent_shape[0].checked_add(clip.latent_shape[0] - drop_tokens * patch_shape[0]).ok_or_else(|| crate::runtime::compute_error("H3 video condition latent time 溢出"))?;
    }
    let rows = values.len().checked_div(columns).ok_or_else(|| crate::runtime::compute_error("H3 video condition columns 为 0"))?;
    let tensor = backend.vae_tensor_from_f32(values, rows, columns)?;
    Ok(H3VideoCondition { tensor, latent_shape, patch_shape })
}

pub fn h3_noise_video_condition<B>(backend: &B, mut condition: H3VideoCondition<B::Tensor>, seed: u64, timestep: f32) -> Result<H3VideoCondition<B::Tensor>, BackendError>
where
    B: VaeBackend,
{
    condition.tensor = h3_noise_condition_tensor(backend, &condition.tensor, seed, timestep)?;
    Ok(condition)
}

pub fn h3_noise_audio_condition<B>(backend: &B, mut condition: H3AudioCondition<B::Tensor>, seed: u64, timestep: f32) -> Result<H3AudioCondition<B::Tensor>, BackendError>
where
    B: VaeBackend,
{
    condition.tensor = h3_noise_condition_tensor(backend, &condition.tensor, seed.wrapping_add(1), timestep)?;
    Ok(condition)
}

fn h3_noise_condition_tensor<B>(backend: &B, input: &B::Tensor, seed: u64, timestep: f32) -> Result<B::Tensor, BackendError>
where
    B: VaeBackend,
{
    if !timestep.is_finite() || !(0.0..=1.0).contains(&timestep) {
        return Err(crate::runtime::compute_error(format!("H3 condition timestep={timestep} 非法")));
    }
    let rows = backend.token_rows(input);
    let columns = backend.token_cols(input);
    let mut values = backend.vae_tensor_to_f32(input)?;
    if values.len() != rows.checked_mul(columns).ok_or_else(|| crate::runtime::compute_error("H3 condition tensor 大小溢出"))? {
        return Err(crate::runtime::compute_error("H3 condition tensor shape 不匹配"));
    }
    let mut random = ConditioningGaussian::new(seed);
    let clean_scale = 1.0 - timestep;
    for value in &mut values {
        *value = *value * clean_scale + random.next() * timestep;
    }
    backend.vae_tensor_from_f32(values, rows, columns)
}

pub fn h3_encode_audio_condition<B>(backend: &B, encoder: &PreparedH3AudioEncoder<B::Weight>, samples: Vec<f32>, batch: usize) -> Result<H3AudioCondition<B::Tensor>, BackendError>
where
    B: VaeBackend + DiffusionBackend,
{
    if batch == 0 || samples.is_empty() || !samples.len().is_multiple_of(batch) || encoder.conv_in.input_channels == 0 {
        return Err(crate::runtime::compute_error(format!("H3 audio samples={} batch={batch} input_channels={} 非法", samples.len(), encoder.conv_in.input_channels)));
    }
    if encoder.latent_channels == 0 || encoder.latent_mean.len() != encoder.latent_channels || encoder.latent_std.len() != encoder.latent_channels || encoder.latent_std.iter().any(|value| !value.is_finite() || *value == 0.0) {
        return Err(crate::runtime::compute_error("H3 audio latent normalization 参数非法"));
    }
    let input_rows = batch.checked_mul(encoder.conv_in.input_channels).ok_or_else(|| crate::runtime::compute_error("H3 audio input rows 溢出"))?;
    if !samples.len().is_multiple_of(input_rows) {
        return Err(crate::runtime::compute_error(format!("H3 audio samples={} 不能被 batch={batch} x input_channels={} 整除", samples.len(), encoder.conv_in.input_channels)));
    }
    let input_time = samples.len() / input_rows;
    let mut hidden = backend.vae_tensor_from_f32(samples, input_rows, input_time)?;
    hidden = audio_conv(backend, &hidden, batch, &encoder.conv_in)?;
    let mut channels = encoder.conv_in.output_channels;
    for stage in &encoder.stages {
        for residual in &stage.residuals {
            if residual.conv1.input_channels != channels || residual.conv2.output_channels != channels {
                return Err(crate::runtime::compute_error("H3 audio residual channel 不连续"));
            }
            let update = backend.snake(&hidden, &residual.snake1, channels)?;
            let update = audio_conv(backend, &update, batch, &residual.conv1)?;
            let update = backend.snake(&update, &residual.snake2, residual.conv1.output_channels)?;
            let update = audio_conv(backend, &update, batch, &residual.conv2)?;
            hidden = backend.add(&hidden, &update)?;
        }
        hidden = backend.snake(&hidden, &stage.snake, channels)?;
        hidden = audio_conv(backend, &hidden, batch, &stage.downsample)?;
        channels = stage.downsample.output_channels;
    }
    hidden = backend.snake(&hidden, &encoder.final_snake, channels)?;
    hidden = backend.channels_to_time(&hidden, channels)?;

    let qkv = linear_with_bias(backend, &hidden, &encoder.attention.qkv)?;
    let attention_columns = encoder.attention.heads.checked_mul(encoder.attention.head_dim).ok_or_else(|| crate::runtime::compute_error("H3 audio attention columns 溢出"))?;
    let (query, key_value) = backend.split_columns(&qkv, attention_columns)?;
    let (key, value) = backend.split_columns(&key_value, attention_columns)?;
    let time = backend.token_rows(&hidden) / batch;
    let attention = backend.causal_attention(&query, &key, &value, time, encoder.attention.heads, encoder.attention.head_dim, (encoder.attention.head_dim as f32).sqrt().recip())?;
    let moments = linear_with_bias(backend, &attention, &encoder.attention.output)?;
    let (mean, _) = backend.split_columns(&moments, encoder.latent_channels)?;
    let mut values = backend.vae_tensor_to_f32(&mean)?;
    for row in values.chunks_exact_mut(encoder.latent_channels) {
        for channel in 0..encoder.latent_channels {
            row[channel] = (row[channel] - encoder.latent_mean[channel]) / encoder.latent_std[channel];
        }
    }
    let tensor = backend.vae_tensor_from_f32(values, batch * time, encoder.latent_channels)?;
    Ok(H3AudioCondition { tensor, batch, time, channels: encoder.latent_channels })
}

fn video_residual<B>(backend: &B, input: &B::Tensor, shape: [usize; 3], block: &PreparedH3VideoResidual<B::Weight>) -> Result<B::Tensor, BackendError>
where
    B: VaeBackend + DiffusionBackend,
{
    let mut hidden = backend.group_norm_time_isolated(input, &block.norm1.weight, &block.norm1.bias, shape[0], block.norm1.groups, block.norm1.eps)?;
    hidden = backend.silu(&hidden)?;
    let (next, next_shape) = video_conv(backend, &hidden, shape, &block.conv1)?;
    if next_shape != shape {
        return Err(crate::runtime::compute_error("H3 video residual conv1 改变了空间 shape"));
    }
    hidden = backend.group_norm_time_isolated(&next, &block.norm2.weight, &block.norm2.bias, shape[0], block.norm2.groups, block.norm2.eps)?;
    hidden = backend.silu(&hidden)?;
    let (hidden, next_shape) = video_conv(backend, &hidden, shape, &block.conv2)?;
    if next_shape != shape {
        return Err(crate::runtime::compute_error("H3 video residual conv2 改变了空间 shape"));
    }
    match &block.shortcut {
        Some(shortcut) => {
            let residual = video_conv(backend, input, shape, shortcut)?.0;
            backend.add(&residual, &hidden)
        }
        None => backend.add(input, &hidden),
    }
}

fn video_conv<B: VaeBackend>(backend: &B, input: &B::Tensor, input_shape: [usize; 3], conv: &PreparedH3VideoConv<B::Weight>) -> Result<(B::Tensor, [usize; 3]), BackendError> {
    let spec = Conv3dSpec { input_channels: conv.input_channels, output_channels: conv.output_channels, input_shape, kernel: conv.kernel, stride: conv.stride, padding: conv.padding, causal: conv.causal };
    let mut output_shape = spec.output_shape().map_err(crate::runtime::compute_error)?;
    for axis in 1..3 {
        let padded = input_shape[axis]
            .checked_add(conv.padding[axis].checked_mul(2).ok_or_else(|| crate::runtime::compute_error("H3 video padding 溢出"))?)
            .and_then(|value| value.checked_add(conv.spatial_pad_after[axis - 1]))
            .ok_or_else(|| crate::runtime::compute_error("H3 video padded shape 溢出"))?;
        output_shape[axis] = padded.checked_sub(conv.kernel[axis]).ok_or_else(|| crate::runtime::compute_error("H3 video kernel 超过输入"))? / conv.stride[axis] + 1;
    }
    let output = backend.encoder_conv3d(input, &conv.weight, conv.bias.as_ref(), &spec, conv.spatial_pad_after)?;
    Ok((output, output_shape))
}

fn audio_conv<B: VaeBackend>(backend: &B, input: &B::Tensor, batch: usize, conv: &PreparedH3AudioConv<B::Weight>) -> Result<B::Tensor, BackendError> {
    let spec = Conv1dSpec { batch, input_channels: conv.input_channels, output_channels: conv.output_channels, kernel: conv.kernel, stride: conv.stride, dilation: conv.dilation, padding: conv.padding };
    backend.conv1d_strided(input, conv.weight_g.as_ref(), &conv.weight_v, conv.bias.as_ref(), &spec)
}

fn linear_with_bias<B>(backend: &B, input: &B::Tensor, linear: &PreparedH3Linear<B::Weight>) -> Result<B::Tensor, BackendError>
where
    B: VaeBackend + DiffusionBackend,
{
    let output = backend.linear(input, &linear.weight)?;
    match &linear.bias {
        Some(bias) => backend.add_row_bias(&output, bias),
        None => Ok(output),
    }
}

fn validate_video_encoder<W>(encoder: &PreparedH3VideoEncoder<W>) -> Result<(), BackendError> {
    if encoder.latent_channels == 0 || encoder.latent_mean.len() != encoder.latent_channels || encoder.latent_std.len() != encoder.latent_channels || encoder.latent_std.iter().any(|value| !value.is_finite() || *value == 0.0) {
        return Err(crate::runtime::compute_error("H3 video latent normalization 参数非法"));
    }
    if encoder.patch.into_iter().any(|value| value == 0) {
        return Err(crate::runtime::compute_error("H3 video patch shape 包含 0"));
    }
    Ok(())
}

fn sample_video_posterior(moments: &[f32], shape: [usize; 3], mean: &[f32], std: &[f32], seed: u64) -> Result<Vec<f32>, BackendError> {
    let spatial = shape.into_iter().product::<usize>();
    if spatial == 0 || !moments.len().is_multiple_of(2 * spatial) {
        return Err(crate::runtime::compute_error(format!("H3 video moments={}，期望 2x{spatial} = channels*2*spatial", moments.len())));
    }
    let channels = moments.len() / (2 * spatial);
    if mean.len() != channels || std.len() != channels {
        return Err(crate::runtime::compute_error(format!("H3 video mean/std={}/{} ≠ channels={channels}", mean.len(), std.len())));
    }
    let mut random = ConditioningGaussian::new(seed);
    let mut latent = vec![0.0; channels * spatial];
    for channel in 0..channels {
        for position in 0..spatial {
            let index = channel * spatial + position;
            let log_variance = moments[(channel + channels) * spatial + position].clamp(-30.0, 20.0);
            let value = moments[index] + (0.5 * log_variance).exp() * random.next();
            latent[index] = (value - mean[channel]) / std[channel];
        }
    }
    Ok(latent)
}

fn patchify_video(latent: &[f32], shape: [usize; 3], patch: [usize; 3]) -> Result<(Vec<f32>, usize, usize), BackendError> {
    if !shape[0].is_multiple_of(patch[0]) || !shape[1].is_multiple_of(patch[1]) || !shape[2].is_multiple_of(patch[2]) {
        return Err(crate::runtime::compute_error(format!("H3 video latent shape={shape:?} 不能按 patch={patch:?} 分块")));
    }
    let spatial = shape.into_iter().product::<usize>();
    if spatial == 0 || !latent.len().is_multiple_of(spatial) {
        return Err(crate::runtime::compute_error(format!("H3 video latent len={} 不能按 spatial={spatial} 拆分", latent.len())));
    }
    let channels = latent.len() / spatial;
    let grid = [shape[0] / patch[0], shape[1] / patch[1], shape[2] / patch[2]];
    let rows = grid.into_iter().product();
    let columns = channels.checked_mul(patch.into_iter().product::<usize>()).ok_or_else(|| crate::runtime::compute_error("H3 video patch columns 溢出"))?;
    let mut output = Vec::with_capacity(rows * columns);
    let spatial = shape.into_iter().product::<usize>();
    for t in 0..grid[0] {
        for h in 0..grid[1] {
            for w in 0..grid[2] {
                for channel in 0..channels {
                    for pt in 0..patch[0] {
                        for ph in 0..patch[1] {
                            for pw in 0..patch[2] {
                                let source_t = t * patch[0] + pt;
                                let source_h = h * patch[1] + ph;
                                let source_w = w * patch[2] + pw;
                                output.push(latent[channel * spatial + (source_t * shape[1] + source_h) * shape[2] + source_w]);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((output, rows, columns))
}

/// Reference conditioning 使用 xorshift 序列，不能替换为 DiT latent 的 SplitMix64 序列。
struct ConditioningGaussian {
    state: u64,
    spare: Option<f32>,
}

impl ConditioningGaussian {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1), spare: None }
    }

    fn next(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let first = self.uniform().max(f32::MIN_POSITIVE);
        let second = self.uniform();
        let radius = (-2.0 * first.ln()).sqrt();
        let angle = std::f32::consts::TAU * second;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    fn uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) as f32 + 0.5) / 16_777_216.0
    }
}
