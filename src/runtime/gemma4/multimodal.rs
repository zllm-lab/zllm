//! Gemma 4 12B unified / E4B Gemma4V 多模态输入与 soft-token 编排。

use std::ops::Range;

use half::f16;

use crate::{
    backend::{Backend, BackendError, GqaPrefillBackend, LinearWeight, VisionBackend},
    runtime::gemma4::{Gemma4AudioConfig, Gemma4Config, Gemma4VisionConfig},
    tokenizer::Tokenizer,
    vision::{ContentPart, RgbImage, pillow_bicubic_resize},
    weight::{
        container::safetensor::TensorData,
        model::gemma4::{Gemma4MultimodalWeights, Gemma4VisionClippedLinearWeights, Gemma4VisionEncoderLayerWeights, Gemma4VisionEncoderWeights},
    },
};

#[derive(Debug)]
pub struct Gemma4ImageInput {
    data: Vec<f32>,
    positions: Vec<[usize; 2]>,
    grid_height: usize,
    grid_width: usize,
    /// 投影后替换语言 soft token 的行数；encoder 输入行数是 `positions.len()`。
    rows: usize,
    cols: usize,
}

#[derive(Debug)]
pub struct Gemma4AudioInput {
    data: Vec<f32>,
    rows: usize,
}

#[derive(Debug)]
pub struct Gemma4SoftTokenRange {
    pub input_index: usize,
    pub tokens: Range<usize>,
}

pub struct Gemma4MultimodalInput {
    pub token_ids: Vec<u32>,
    pub embedding_token_ids: Vec<u32>,
    pub visible_ends: Vec<u32>,
    images: Vec<Gemma4ImageInput>,
    image_ranges: Vec<Gemma4SoftTokenRange>,
    audio: Vec<Gemma4AudioInput>,
    audio_ranges: Vec<Gemma4SoftTokenRange>,
}

impl Gemma4MultimodalInput {
    pub fn chunk_end(&self, start: usize, preferred_end: usize) -> usize {
        let mut end = preferred_end.min(self.token_ids.len());
        for range in self.image_ranges.iter().chain(&self.audio_ranges) {
            if range.tokens.start < end && range.tokens.end > end {
                end = range.tokens.end;
            }
        }
        end.max(start + 1).min(self.token_ids.len())
    }

    pub fn chunk_has_visual_visibility(&self, range: Range<usize>) -> bool {
        self.image_ranges.iter().any(|image| image.tokens.start < range.end && image.tokens.end > range.start)
    }
}

pub fn gemma4_multimodal_input(tokenizer: &Tokenizer, config: &Gemma4Config, prompt: &str, images: &[RgbImage], video_frames: &[RgbImage], audio_samples: &[Vec<f32>]) -> Result<Gemma4MultimodalInput, String> {
    let vision = config.vision.ok_or("Gemma 4 checkpoint 不包含 unified vision")?;
    let audio_config = config.audio.ok_or("Gemma 4 checkpoint 不包含 unified audio")?;
    let mut processed_images = Vec::with_capacity(images.len() + video_frames.len());
    for image in images {
        processed_images.push(preprocess_image(image, &vision, vision.default_soft_tokens)?);
    }
    for frame in video_frames {
        processed_images.push(preprocess_image(frame, &vision, 70)?);
    }
    let processed_audio: Vec<_> = audio_samples.iter().map(|samples| preprocess_audio(samples, &audio_config)).collect::<Result<_, _>>()?;

    let mut text = String::from("<bos><|turn>user\n");
    for image in &processed_images[..images.len()] {
        text.push_str("<|image>");
        text.push_str(&"<|image|>".repeat(image.rows));
        text.push_str("<image|>");
    }
    for (index, frame) in processed_images[images.len()..].iter().enumerate() {
        text.push_str(&format!("{}:00 ", index / 60));
        text.push_str("<|image>");
        text.push_str(&"<|video|>".repeat(frame.rows));
        text.push_str("<image|>");
    }
    text.push_str(prompt);
    for audio in &processed_audio {
        text.push_str("<|audio>");
        text.push_str(&"<|audio|>".repeat(audio.rows));
        text.push_str("<audio|>");
    }
    text.push_str("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");

    finalize_multimodal_input(tokenizer, config, &text, processed_images, images.len(), processed_audio)
}

/// 节点图文混排入口:parts 保序展开,图像就地展开为 `<|image>` soft-token 段
/// (无视频/音频;server 协议只有 text/image part)。与 `gemma4_multimodal_input`
/// 共享 finalize(分词/soft-token 区间/可见性)。
pub fn gemma4_multimodal_input_from_parts(tokenizer: &Tokenizer, config: &Gemma4Config, parts: &[ContentPart<'_>]) -> Result<Gemma4MultimodalInput, String> {
    let vision = config.vision.ok_or("Gemma 4 checkpoint 不包含 unified vision")?;
    let mut text = String::new();
    let mut images = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text(segment) => text.push_str(segment),
            ContentPart::Image(image) => {
                let processed = preprocess_image(image, &vision, vision.default_soft_tokens)?;
                text.push_str("<|image>");
                text.push_str(&"<|image|>".repeat(processed.rows));
                text.push_str("<image|>");
                images.push(processed);
            }
        }
    }
    let still_images = images.len();
    finalize_multimodal_input(tokenizer, config, &text, images, still_images, Vec::new())
}

/// 分词并定位 soft-token 区间;`still_images` 是 images 前缀中用 image_token_id
/// 的数量(其余为视频帧,用 video_token_id)。图像区间整段双向可见。
fn finalize_multimodal_input(tokenizer: &Tokenizer, config: &Gemma4Config, text: &str, images: Vec<Gemma4ImageInput>, still_images: usize, audio: Vec<Gemma4AudioInput>) -> Result<Gemma4MultimodalInput, String> {
    let token_ids = tokenizer.tokenize(text.as_bytes());
    let mut image_ranges = token_ranges(&token_ids, config.image_token_id, images[..still_images].iter().map(|image| image.rows), 0)?;
    let mut video_ranges = token_ranges(&token_ids, config.video_token_id, images[still_images..].iter().map(|image| image.rows), still_images)?;
    image_ranges.append(&mut video_ranges);
    image_ranges.sort_by_key(|range| range.tokens.start);
    let audio_ranges = token_ranges(&token_ids, config.audio_token_id, audio.iter().map(|audio| audio.rows), 0)?;

    let embedding_token_ids = token_ids.iter().map(|&token| if matches!(token, value if value == config.image_token_id || value == config.video_token_id || value == config.audio_token_id) { config.pad_token_id } else { token }).collect();
    let mut visible_ends: Vec<u32> = (1..=token_ids.len()).map(|end| u32::try_from(end).map_err(|_| "Gemma 4 prompt 超过 u32 position")).collect::<Result<_, _>>()?;
    for range in &image_ranges {
        let end = u32::try_from(range.tokens.end).map_err(|_| "Gemma 4 visual range 超过 u32")?;
        visible_ends[range.tokens.clone()].fill(end);
    }

    Ok(Gemma4MultimodalInput { token_ids, embedding_token_ids, visible_ends, images, image_ranges, audio, audio_ranges })
}

fn token_ranges(tokens: &[u32], token_id: u32, expected_rows: impl Iterator<Item = usize>, input_offset: usize) -> Result<Vec<Gemma4SoftTokenRange>, String> {
    let mut cursor = 0;
    expected_rows
        .enumerate()
        .map(|(input_index, rows)| {
            let start = tokens[cursor..].iter().position(|token| *token == token_id).map(|position| cursor + position).ok_or_else(|| format!("Gemma 4 缺少 soft token {token_id}"))?;
            let end = start.checked_add(rows).ok_or("Gemma 4 soft token range 溢出")?;
            if tokens.get(start..end).is_none_or(|values| values.iter().any(|token| *token != token_id)) {
                return Err(format!("Gemma 4 soft token {token_id} 数量不匹配，期望 {rows}"));
            }
            cursor = end;
            Ok(Gemma4SoftTokenRange { input_index: input_offset + input_index, tokens: start..end })
        })
        .collect()
}

fn preprocess_image(image: &RgbImage, config: &Gemma4VisionConfig, max_soft_tokens: usize) -> Result<Gemma4ImageInput, String> {
    if !matches!(max_soft_tokens, 70 | 140 | 280 | 560 | 1120) {
        return Err(format!("Gemma 4 vision soft tokens={max_soft_tokens} 不受支持"));
    }
    let side = config.model_patch_size().map_err(|error| error.to_string())?;
    let target_pixels = max_soft_tokens.checked_mul(side).and_then(|value| value.checked_mul(side)).ok_or("Gemma 4 image pixel budget 溢出")?;
    let factor = (target_pixels as f64 / (image.height * image.width) as f64).sqrt();
    let mut height = ((factor * image.height as f64) / side as f64).floor() as usize * side;
    let mut width = ((factor * image.width as f64) / side as f64).floor() as usize * side;
    let max_side = max_soft_tokens.checked_mul(side).ok_or("Gemma 4 max image side 溢出")?;
    if height == 0 {
        height = side;
        width = ((image.width / image.height).max(1) * side).min(max_side);
    }
    if width == 0 {
        width = side;
        height = ((image.height / image.width).max(1) * side).min(max_side);
    }
    let source = image::RgbImage::from_raw(image.width as u32, image.height as u32, image.pixels.clone()).ok_or("构造 Gemma 4 RGB 图像失败")?;
    let resized = pillow_bicubic_resize(&source, width as u32, height as u32);
    let patch = if config.encoder.is_some() { config.patch_size } else { side };
    let grid_height = height / patch;
    let grid_width = width / patch;
    let input_rows = grid_height.checked_mul(grid_width).ok_or("Gemma 4 image rows 溢出")?;
    let rows = if config.encoder.is_some() {
        let merge = config.pooling_kernel_size;
        if !grid_height.is_multiple_of(merge) || !grid_width.is_multiple_of(merge) {
            return Err(format!("Gemma 4 image grid {grid_height}x{grid_width} 不能按 {merge} 池化"));
        }
        (grid_height / merge) * (grid_width / merge)
    } else {
        input_rows
    };
    let cols = patch.checked_mul(patch).and_then(|value| value.checked_mul(3)).ok_or("Gemma 4 image cols 溢出")?;
    let mut data = Vec::with_capacity(input_rows * cols);
    let mut positions = Vec::with_capacity(input_rows);
    for y_block in 0..grid_height {
        for x_block in 0..grid_width {
            // 像素平面布局(CHW):每个 patch 是 R/G/B 三个行主序平面拼接,
            // 对应 llama.cpp gemma4uv 从 planar inp_raw 的 im2col 提取顺序
            for channel in 0..3 {
                for y in 0..patch {
                    for x in 0..patch {
                        let pixel = resized.get_pixel((x_block * patch + x) as u32, (y_block * patch + y) as u32);
                        let value = pixel.0[channel] as f32 / 255.0;
                        data.push(if config.encoder.is_some() { value.mul_add(2.0, -1.0) } else { value });
                    }
                }
            }
            positions.push([x_block, y_block]);
        }
    }
    Ok(Gemma4ImageInput { data, positions, grid_height, grid_width, rows, cols })
}

fn preprocess_audio(samples: &[f32], config: &Gemma4AudioConfig) -> Result<Gemma4AudioInput, String> {
    if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
        return Err("Gemma 4 audio samples 必须是非空有限值".to_owned());
    }
    let rows = samples.len().div_ceil(config.samples_per_token);
    let mut data = vec![0.0; rows * config.samples_per_token];
    data[..samples.len()].copy_from_slice(samples);
    Ok(Gemma4AudioInput { data, rows })
}

struct PreparedNorm<W> {
    weight: W,
    bias: W,
}

struct PreparedLinear<W> {
    weight: W,
    bias: Option<W>,
}

struct PreparedClippedLinear<W> {
    weight: W,
    input_min: f32,
    input_max: f32,
    output_min: f32,
    output_max: f32,
}

struct PreparedVisionEncoderLayer<W> {
    input_norm: W,
    query: PreparedClippedLinear<W>,
    query_norm: W,
    key: PreparedClippedLinear<W>,
    key_norm: W,
    value: PreparedClippedLinear<W>,
    output: PreparedClippedLinear<W>,
    attention_post_norm: W,
    ffn_norm: W,
    gate: PreparedClippedLinear<W>,
    up: PreparedClippedLinear<W>,
    down: PreparedClippedLinear<W>,
    ffn_post_norm: W,
}

struct PreparedVisionEncoder<W> {
    patch_embedding: W,
    position_embedding: Vec<f32>,
    layers: Vec<PreparedVisionEncoderLayer<W>>,
    head_unit_norm: W,
    projection_unit_norm: W,
    projection: W,
}

struct PreparedVision<W> {
    patch_ln1: PreparedNorm<W>,
    patch_dense: PreparedLinear<W>,
    patch_ln2: PreparedNorm<W>,
    position_embedding: Vec<f32>,
    position_norm: PreparedNorm<W>,
    unit_norm: W,
    projection: W,
}

struct PreparedAudio<W> {
    unit_norm: W,
    projection: W,
}

pub struct Gemma4MultimodalModel<W> {
    vision: Option<PreparedVision<W>>,
    vision_encoder: Option<PreparedVisionEncoder<W>>,
    audio: Option<PreparedAudio<W>>,
}

pub fn prepare_gemma4_multimodal_model<B>(backend: &B, config: &Gemma4Config, weights: &Gemma4MultimodalWeights) -> Result<Gemma4MultimodalModel<B::Weight>, BackendError>
where
    B: Backend,
{
    let vision = config
        .vision
        .zip(weights.vision.as_ref())
        .map(|(config, weights)| {
            Ok(PreparedVision {
                patch_ln1: prepare_norm(backend, &weights.patch_ln1_weight, &weights.patch_ln1_bias)?,
                patch_dense: PreparedLinear { weight: prepare_dense(backend, &weights.patch_dense)?, bias: Some(backend.prepare_weight(LinearWeight::F32(&weights.patch_dense_bias), 1, config.embedding_size)?) },
                patch_ln2: prepare_norm(backend, &weights.patch_ln2_weight, &weights.patch_ln2_bias)?,
                position_embedding: weights.position_embedding.clone(),
                position_norm: prepare_norm(backend, &weights.position_norm_weight, &weights.position_norm_bias)?,
                unit_norm: backend.prepare_f32(&vec![1.0; config.embedding_size], 1, config.embedding_size)?,
                projection: prepare_dense(backend, &weights.projection)?,
            })
        })
        .transpose()?;
    let audio = config
        .audio
        .zip(weights.audio.as_ref())
        .map(|(config, weights)| Ok(PreparedAudio { unit_norm: backend.prepare_f32(&vec![1.0; config.embedding_size], 1, config.embedding_size)?, projection: prepare_dense(backend, &weights.projection)? }))
        .transpose()?;
    let vision_encoder = config.vision.zip(weights.vision_encoder.as_ref()).map(|(vision, weights)| prepare_vision_encoder(backend, &vision, weights)).transpose()?;
    Ok(Gemma4MultimodalModel { vision, vision_encoder, audio })
}

#[allow(clippy::too_many_arguments)]
pub fn gemma4_multimodal_embedding<B>(backend: &B, config: &Gemma4Config, model: &Gemma4MultimodalModel<B::Weight>, input: &Gemma4MultimodalInput, text_embedding: &[f32], token_range: Range<usize>) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    let rows = token_range.len();
    if text_embedding.len() != rows.checked_mul(config.hidden_size).ok_or_else(|| crate::runtime::compute_error("Gemma 4 multimodal embedding 溢出"))? {
        return Err(crate::runtime::compute_error("Gemma 4 multimodal text embedding shape 不匹配"));
    }
    let mut hidden = backend.vision_tensor_from_f32(text_embedding, rows, config.hidden_size)?;
    for range in &input.image_ranges {
        if range.tokens.end <= token_range.start || range.tokens.start >= token_range.end {
            continue;
        }
        if range.tokens.start < token_range.start || range.tokens.end > token_range.end {
            return Err(crate::runtime::compute_error(format!("Gemma 4 image range {:?} 被 prefill chunk 截断", range.tokens)));
        }
        let image = &input.images[range.input_index];
        let encoded = match (model.vision.as_ref(), model.vision_encoder.as_ref()) {
            (Some(weights), None) => encode_unified_image(backend, config, weights, image)?,
            (None, Some(weights)) => encode_encoder_image(backend, config, weights, image)?,
            _ => return Err(crate::runtime::compute_error("Gemma 4 缺少或同时装载了两种 vision weights")),
        };
        backend.scatter_rows(&mut hidden, range.tokens.start - token_range.start, &encoded)?;
    }
    for range in &input.audio_ranges {
        if range.tokens.end <= token_range.start || range.tokens.start >= token_range.end {
            continue;
        }
        if range.tokens.start < token_range.start || range.tokens.end > token_range.end {
            return Err(crate::runtime::compute_error(format!("Gemma 4 audio range {:?} 被 prefill chunk 截断", range.tokens)));
        }
        let audio = &input.audio[range.input_index];
        let encoded = encode_audio(backend, config, model.audio.as_ref().ok_or_else(|| crate::runtime::compute_error("Gemma 4 缺少 audio weights"))?, audio)?;
        backend.scatter_rows(&mut hidden, range.tokens.start - token_range.start, &encoded)?;
    }
    Ok(hidden)
}

fn encode_unified_image<B: VisionBackend>(backend: &B, config: &Gemma4Config, weights: &PreparedVision<B::Weight>, image: &Gemma4ImageInput) -> Result<B::Tensor, BackendError> {
    let vision = config.vision.ok_or_else(|| crate::runtime::compute_error("Gemma 4 vision config 缺失"))?;
    // unified vision 的 LayerNorm 是 PyTorch 默认 eps(1e-5),与语言侧 rms_eps 无关
    // (llama.cpp gemma4uv 同样硬编码该值)
    const VISION_LN_EPS: f32 = 1e-5;
    let input = backend.vision_tensor_from_f32(&image.data, image.rows, image.cols)?;
    let hidden = backend.layernorm_bias(&input, &weights.patch_ln1.weight, &weights.patch_ln1.bias, VISION_LN_EPS)?;
    let hidden = backend.linear(&hidden, &weights.patch_dense.weight)?;
    let hidden = match weights.patch_dense.bias.as_ref() {
        Some(bias) => backend.add_bias(&hidden, bias)?,
        None => hidden,
    };
    let hidden = backend.layernorm_bias(&hidden, &weights.patch_ln2.weight, &weights.patch_ln2.bias, VISION_LN_EPS)?;
    let mut positions = Vec::with_capacity(image.rows * vision.embedding_size);
    for &[x, y] in &image.positions {
        if x >= vision.position_embedding_size || y >= vision.position_embedding_size {
            return Err(crate::runtime::compute_error(format!("Gemma 4 image position [{x},{y}] 越界")));
        }
        // 官方布局 [2, position_embedding_size, hidden]:axis 0 是 x 编码、axis 1 是 y 编码,
        // 各自分块连续存储;x_emb + y_emb 相加(见 transformers Gemma4VisionPatchEmbedder)。
        let x_start = x * vision.embedding_size;
        let y_start = vision.position_embedding_size * vision.embedding_size + y * vision.embedding_size;
        for column in 0..vision.embedding_size {
            positions.push(weights.position_embedding[x_start + column] + weights.position_embedding[y_start + column]);
        }
    }
    let positions = backend.vision_tensor_from_f32(&positions, image.rows, vision.embedding_size)?;
    let hidden = backend.add(&hidden, &positions)?;
    let hidden = backend.layernorm_bias(&hidden, &weights.position_norm.weight, &weights.position_norm.bias, VISION_LN_EPS)?;
    let hidden = backend.rmsnorm(&hidden, &weights.unit_norm, config.rms_eps)?;
    backend.linear(&hidden, &weights.projection)
}

fn encode_encoder_image<B: VisionBackend + GqaPrefillBackend>(backend: &B, config: &Gemma4Config, weights: &PreparedVisionEncoder<B::Weight>, image: &Gemma4ImageInput) -> Result<B::Tensor, BackendError> {
    let vision = config.vision.ok_or_else(|| crate::runtime::compute_error("Gemma 4 vision config 缺失"))?;
    let encoder = vision.encoder.ok_or_else(|| crate::runtime::compute_error("Gemma 4 vision encoder config 缺失"))?;
    let input_rows = image.positions.len();
    if image.data.len() != input_rows * image.cols || image.cols != 3 * vision.patch_size * vision.patch_size {
        return Err(crate::runtime::compute_error(format!("Gemma4V patch input=[{input_rows},{}] 非法", image.cols)));
    }
    let input = backend.vision_tensor_from_f32(&image.data, input_rows, image.cols)?;
    let mut hidden = backend.linear(&input, &weights.patch_embedding)?;
    let mut positions = Vec::with_capacity(input_rows * vision.embedding_size);
    for &[x, y] in &image.positions {
        if x >= vision.position_embedding_size || y >= vision.position_embedding_size {
            return Err(crate::runtime::compute_error(format!("Gemma4V image position [{x},{y}] 越界")));
        }
        let x_start = x * vision.embedding_size;
        let y_start = (vision.position_embedding_size + y) * vision.embedding_size;
        for column in 0..vision.embedding_size {
            positions.push(weights.position_embedding[x_start + column] + weights.position_embedding[y_start + column]);
        }
    }
    hidden = backend.add(&hidden, &backend.vision_tensor_from_f32(&positions, input_rows, vision.embedding_size)?)?;
    let head_dim = vision.embedding_size / encoder.num_heads;
    let (cos, sin) = gemma4v_rope(&image.positions, head_dim, encoder.rope_theta)?;
    let cos = backend.vision_tensor_from_f32(&cos, input_rows, head_dim)?;
    let sin = backend.vision_tensor_from_f32(&sin, input_rows, head_dim)?;
    for layer in &weights.layers {
        let normed = backend.rmsnorm(&hidden, &layer.input_norm, encoder.rms_eps)?;
        let query = clipped_linear(backend, &normed, &layer.query)?;
        let key = clipped_linear(backend, &normed, &layer.key)?;
        let value = clipped_linear(backend, &normed, &layer.value)?;
        let query = backend.rmsnorm_heads(&query, &layer.query_norm, encoder.num_heads, head_dim, encoder.rms_eps)?;
        let key = backend.rmsnorm_heads(&key, &layer.key_norm, encoder.num_heads, head_dim, encoder.rms_eps)?;
        let value = backend.rmsnorm_heads(&value, &weights.head_unit_norm, encoder.num_heads, head_dim, encoder.rms_eps)?;
        let attention = backend.vision_attention_2d(&query, &key, &value, &cos, &sin, encoder.num_heads, 1.0)?;
        let attention = clipped_linear(backend, &attention, &layer.output)?;
        let attention = backend.rmsnorm(&attention, &layer.attention_post_norm, encoder.rms_eps)?;
        let residual = backend.add(&hidden, &attention)?;
        let normed = backend.rmsnorm(&residual, &layer.ffn_norm, encoder.rms_eps)?;
        let gate = clipped_linear(backend, &normed, &layer.gate)?;
        let up = clipped_linear(backend, &normed, &layer.up)?;
        let activated = backend.vision_quick_gelu_gated(&gate, &up)?;
        let output = clipped_linear(backend, &activated, &layer.down)?;
        let output = backend.rmsnorm(&output, &layer.ffn_post_norm, encoder.rms_eps)?;
        hidden = backend.add(&residual, &output)?;
        backend.submit_batch();
    }
    let hidden = backend.vision_average_pool(&hidden, image.grid_height, image.grid_width, vision.pooling_kernel_size, (vision.embedding_size as f32).sqrt())?;
    if backend.token_rows(&hidden) != image.rows {
        return Err(crate::runtime::compute_error(format!("Gemma4V pooled rows={}，期望 {}", backend.token_rows(&hidden), image.rows)));
    }
    let hidden = backend.rmsnorm(&hidden, &weights.projection_unit_norm, encoder.rms_eps)?;
    backend.linear(&hidden, &weights.projection)
}

fn gemma4v_rope(positions: &[[usize; 2]], head_dim: usize, theta: f32) -> Result<(Vec<f32>, Vec<f32>), BackendError> {
    if !head_dim.is_multiple_of(4) || !theta.is_finite() || theta <= 0.0 {
        return Err(crate::runtime::compute_error(format!("Gemma4V RoPE head_dim={head_dim} theta={theta} 非法")));
    }
    let axis_dim = head_dim / 2;
    let axis_pairs = axis_dim / 2;
    let mut cosine = Vec::with_capacity(positions.len() * head_dim);
    let mut sine = Vec::with_capacity(positions.len() * head_dim);
    for &[x, y] in positions {
        for position in [x, y] {
            let angles = (0..axis_pairs).map(|pair| position as f32 * theta.powf(-2.0 * pair as f32 / axis_dim as f32)).collect::<Vec<_>>();
            for _ in 0..2 {
                cosine.extend(angles.iter().map(|angle| angle.cos()));
                sine.extend(angles.iter().map(|angle| angle.sin()));
            }
        }
    }
    Ok((cosine, sine))
}

fn clipped_linear<B: VisionBackend>(backend: &B, input: &B::Tensor, linear: &PreparedClippedLinear<B::Weight>) -> Result<B::Tensor, BackendError> {
    let input = if linear.input_min == f32::MIN && linear.input_max == f32::MAX { backend.linear(input, &linear.weight)? } else { backend.linear(&backend.vision_clamp(input, linear.input_min, linear.input_max)?, &linear.weight)? };
    if linear.output_min == f32::MIN && linear.output_max == f32::MAX { Ok(input) } else { backend.vision_clamp(&input, linear.output_min, linear.output_max) }
}

fn prepare_vision_encoder<B: Backend>(backend: &B, config: &Gemma4VisionConfig, weights: &Gemma4VisionEncoderWeights) -> Result<PreparedVisionEncoder<B::Weight>, BackendError> {
    let encoder = config.encoder.ok_or_else(|| crate::runtime::compute_error("Gemma4V encoder config 缺失"))?;
    let head_dim = config.embedding_size / encoder.num_heads;
    let layers = weights.layers.iter().map(|layer| prepare_vision_encoder_layer(backend, layer)).collect::<Result<Vec<_>, _>>()?;
    Ok(PreparedVisionEncoder {
        patch_embedding: prepare_dense(backend, &weights.patch_embedding)?,
        position_embedding: weights.position_embedding.clone(),
        layers,
        head_unit_norm: backend.prepare_f32(&vec![1.0; head_dim], 1, head_dim)?,
        projection_unit_norm: backend.prepare_f32(&vec![1.0; config.embedding_size], 1, config.embedding_size)?,
        projection: prepare_dense(backend, &weights.projection)?,
    })
}

fn prepare_vision_encoder_layer<B: Backend>(backend: &B, layer: &Gemma4VisionEncoderLayerWeights) -> Result<PreparedVisionEncoderLayer<B::Weight>, BackendError> {
    let vector = |values: &[f32]| backend.prepare_f32(values, 1, values.len());
    Ok(PreparedVisionEncoderLayer {
        input_norm: vector(&layer.input_norm)?,
        query: prepare_clipped_linear(backend, &layer.query)?,
        query_norm: vector(&layer.query_norm)?,
        key: prepare_clipped_linear(backend, &layer.key)?,
        key_norm: vector(&layer.key_norm)?,
        value: prepare_clipped_linear(backend, &layer.value)?,
        output: prepare_clipped_linear(backend, &layer.output)?,
        attention_post_norm: vector(&layer.attention_post_norm)?,
        ffn_norm: vector(&layer.ffn_norm)?,
        gate: prepare_clipped_linear(backend, &layer.gate)?,
        up: prepare_clipped_linear(backend, &layer.up)?,
        down: prepare_clipped_linear(backend, &layer.down)?,
        ffn_post_norm: vector(&layer.ffn_post_norm)?,
    })
}

fn prepare_clipped_linear<B: Backend>(backend: &B, weights: &Gemma4VisionClippedLinearWeights) -> Result<PreparedClippedLinear<B::Weight>, BackendError> {
    Ok(PreparedClippedLinear { weight: prepare_dense(backend, &weights.weight)?, input_min: weights.input_min, input_max: weights.input_max, output_min: weights.output_min, output_max: weights.output_max })
}

fn encode_audio<B: VisionBackend>(backend: &B, config: &Gemma4Config, weights: &PreparedAudio<B::Weight>, audio: &Gemma4AudioInput) -> Result<B::Tensor, BackendError> {
    let audio_config = config.audio.ok_or_else(|| crate::runtime::compute_error("Gemma 4 audio config 缺失"))?;
    let input = backend.vision_tensor_from_f32(&audio.data, audio.rows, audio_config.embedding_size)?;
    let hidden = backend.rmsnorm(&input, &weights.unit_norm, config.rms_eps)?;
    backend.linear(&hidden, &weights.projection)
}

fn prepare_norm<B: Backend>(backend: &B, weight: &[f32], bias: &[f32]) -> Result<PreparedNorm<B::Weight>, BackendError> {
    if weight.len() != bias.len() {
        return Err(crate::runtime::compute_error("Gemma 4 multimodal norm weight/bias 长度不一致"));
    }
    Ok(PreparedNorm { weight: backend.prepare_f32(weight, 1, weight.len())?, bias: backend.prepare_f32(bias, 1, bias.len())? })
}

fn prepare_dense<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let rows = tensor.shape.first().copied().ok_or_else(|| crate::runtime::compute_error(format!("{} shape 为空", tensor.name)))?;
    let cols = tensor.shape.get(1).copied().ok_or_else(|| crate::runtime::compute_error(format!("{} 不是 rank-2", tensor.name)))?;
    match tensor.dtype.as_str() {
        "BF16" => backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, cols),
        "F16" => {
            let values: Vec<f16> = tensor.data.chunks_exact(2).map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]])).collect();
            backend.prepare_weight(LinearWeight::F16(&values), rows, cols)
        }
        "F32" => {
            let values: Vec<f32> = tensor.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 multimodal chunk"))).collect();
            backend.prepare_weight(LinearWeight::F32(&values), rows, cols)
        }
        dtype => Err(crate::runtime::compute_error(format!("{} dtype={dtype} 不受支持", tensor.name))),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use std::path::PathBuf;

    use super::*;

    use crate::{backend::cpu::CpuContext, kernel::cpu::CpuTensor};

    #[cfg(target_os = "macos")]
    use crate::{
        backend::metal::{MetalContext, MetalWeight},
        weight::model::gemma4::Gemma4Weights,
    };

    #[test]
    fn square_image_uses_requested_soft_token_budget() {
        let config = Gemma4Config::standard_12b().vision.unwrap();
        let image = RgbImage::new(96, 96, vec![255; 96 * 96 * 3]).unwrap();
        let input = preprocess_image(&image, &config, 280).unwrap();
        assert!(input.rows <= 280);
        assert_eq!(input.data.len(), input.rows * 48 * 48 * 3);
        assert_eq!(input.positions.len(), input.rows);
    }

    #[test]
    fn e4b_image_keeps_encoder_patches_and_soft_token_grid_separate() {
        let config = Gemma4Config::e4b().vision.unwrap();
        let image = RgbImage::new(96, 96, vec![255; 96 * 96 * 3]).unwrap();
        let input = preprocess_image(&image, &config, 280).unwrap();
        assert!(input.grid_height.is_multiple_of(3) && input.grid_width.is_multiple_of(3));
        assert_eq!(input.positions.len(), input.grid_height * input.grid_width);
        assert_eq!(input.rows, input.positions.len() / 9);
        assert_eq!(input.cols, 16 * 16 * 3);
        assert_eq!(input.data.len(), input.positions.len() * input.cols);
        assert!(input.data.iter().all(|value| *value == 1.0));
    }

    #[test]
    fn gemma4v_rope_keeps_x_y_axes_independent() {
        let (cos, sin) = gemma4v_rope(&[[1, 2]], 8, 100.0).unwrap();
        assert_eq!(cos.len(), 8);
        assert_eq!(sin.len(), 8);
        assert_eq!(&cos[0..2], &cos[2..4]);
        assert_eq!(&cos[4..6], &cos[6..8]);
        assert_ne!(cos[1], cos[5]);
        assert_ne!(sin[0], sin[4]);
    }

    #[test]
    fn e4b_average_pool_preserves_spatial_windows() {
        use crate::backend::VisionBackend;
        let cpu = CpuContext;
        let input = CpuTensor { data: (0..36).map(|value| value as f32).collect(), rows: 36, cols: 1 };
        let pooled = cpu.vision_average_pool(&input, 6, 6, 3, 1.0).unwrap();
        for (actual, expected) in pooled.data.iter().zip([7.0, 10.0, 25.0, 28.0]) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gemma4v_visual_primitives_cpu_match_metal() {
        use crate::backend::VisionBackend;
        let cpu = CpuContext;
        let metal = MetalContext::new_default().unwrap();
        let values = (0..36 * 4).map(|value| value as f32 / 17.0 - 2.0).collect::<Vec<_>>();
        let cpu_input = CpuTensor { data: values.clone(), rows: 36, cols: 4 };
        let metal_input = metal.vision_tensor_from_f32(&values, 36, 4).unwrap();
        let cpu_pool = cpu.vision_average_pool(&cpu_input, 6, 6, 3, 4.0).unwrap();
        let metal_pool = metal.vision_average_pool(&metal_input, 6, 6, 3, 4.0).unwrap();
        let metal_pool = metal.tensor_to_f32(&metal_pool);
        for (actual, expected) in metal_pool.iter().zip(&cpu_pool.data) {
            assert!((actual - expected).abs() < 2.0e-2 * (1.0 + expected.abs()), "pool {actual} vs {expected}");
        }

        let gate = CpuTensor { data: vec![-2.0, -0.5, 0.5, 2.0], rows: 1, cols: 4 };
        let up = CpuTensor { data: vec![0.5, 1.0, 1.5, 2.0], rows: 1, cols: 4 };
        let expected = cpu.vision_quick_gelu_gated(&gate, &up).unwrap();
        let gate_metal = metal.vision_tensor_from_f32(&gate.data, 1, 4).unwrap();
        let up_metal = metal.vision_tensor_from_f32(&up.data, 1, 4).unwrap();
        let actual = metal.vision_quick_gelu_gated(&gate_metal, &up_metal).unwrap();
        let actual = metal.tensor_to_f32(&actual);
        for (actual, expected) in actual.iter().zip(&expected.data) {
            assert!((actual - expected).abs() < 2.0e-3, "quick gelu {actual} vs {expected}");
        }

        let query = CpuTensor { data: vec![0.2, -0.3, 0.5, 0.7, -0.4, 0.8, 0.1, -0.2], rows: 2, cols: 4 };
        let key = CpuTensor { data: vec![0.6, 0.1, -0.5, 0.3, 0.7, -0.2, 0.4, 0.9], rows: 2, cols: 4 };
        let value = CpuTensor { data: vec![0.5, -0.1, 0.2, 0.4, -0.3, 0.6, 0.8, -0.7], rows: 2, cols: 4 };
        let (cos, sin) = gemma4v_rope(&[[0, 0], [1, 1]], 4, 100.0).unwrap();
        let cos_cpu = CpuTensor { data: cos.clone(), rows: 2, cols: 4 };
        let sin_cpu = CpuTensor { data: sin.clone(), rows: 2, cols: 4 };
        let expected = cpu.vision_attention_2d(&query, &key, &value, &cos_cpu, &sin_cpu, 1, 1.0).unwrap();
        let upload = |tensor: &CpuTensor| metal.vision_tensor_from_f32(&tensor.data, tensor.rows, tensor.cols).unwrap();
        let actual = metal.vision_attention_2d(&upload(&query), &upload(&key), &upload(&value), &upload(&cos_cpu), &upload(&sin_cpu), 1, 1.0).unwrap();
        let actual = metal.tensor_to_f32(&actual);
        for (actual, expected) in actual.iter().zip(&expected.data) {
            assert!((actual - expected).abs() < 3.0e-3, "2d attention {actual} vs {expected}");
        }
    }

    /// mmproj 位置编码布局诊断:GGUF 路径没有 safetensors 路径的长度校验,
    /// v.position_embd 元素数与 unified vision 期望不符时位置编码错位(图像感知乱码)。
    /// `ZLLM_GEMMA4_GGUF=/path/to/gemma4.gguf cargo test --lib mmproj_layout -- --nocapture`
    #[cfg(target_os = "macos")]
    #[test]
    fn mmproj_position_embedding_layout() {
        let Some(root) = std::env::var_os("ZLLM_GEMMA4_GGUF").map(PathBuf::from) else { return };
        let config = Gemma4Weights::select_config(&root).expect("select_config");
        let weights = Gemma4Weights::open(&root, config.clone()).expect("open");
        let mm = weights.load_multimodal_weights().expect("multimodal");
        let Some(vision) = &mm.vision else {
            if let Some(encoder) = &mm.vision_encoder {
                let vision_cfg = config.vision.expect("vision config");
                assert_eq!(encoder.layers.len(), vision_cfg.encoder.expect("encoder config").layer_count);
                assert_eq!(encoder.projection.shape, vec![config.hidden_size, vision_cfg.embedding_size]);
                println!("Gemma4V mmproj encoder 加载成功");
            }
            println!("无 mmproj vision");
            return;
        };
        let vision_cfg = config.vision.expect("vision config");
        let patch = vision_cfg.model_patch_size().expect("patch size");
        println!("vision cfg: embedding_size={} position_embedding_size={} default_soft_tokens={} patch={patch}", vision_cfg.embedding_size, vision_cfg.position_embedding_size, vision_cfg.default_soft_tokens);
        println!("position_embedding.len()={} expected={}", vision.position_embedding.len(), vision_cfg.position_embedding_size * 2 * vision_cfg.embedding_size);
        if root.is_file() {
            if let Some(directory) = root.parent() {
                let mmproj = std::fs::read_dir(directory)
                    .expect("读目录")
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .find(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("mmproj") && name.ends_with(".gguf")));
                if let Some(path) = mmproj {
                    let reader = crate::weight::container::gguf::GgufReader::open(&path).expect("open mmproj");
                    for name in ["v.position_embd.weight", "v.patch_embd.weight", "v.patch_norm.1.weight", "mm.input_projection.weight"] {
                        if let Some(info) = reader.tensor(name) {
                            println!("mmproj {name} dims={:?} type={}", info.dims, info.tensor_type.name());
                        }
                    }
                    let mut names: Vec<&str> = reader.tensors().iter().map(|tensor| tensor.name.as_str()).collect();
                    names.sort();
                    println!("mmproj 张量总数={} 全部名:", names.len());
                    for name in names {
                        println!("  {name}");
                    }
                }
            }
        }
        assert_eq!(vision.position_embedding.len(), vision_cfg.position_embedding_size * 2 * vision_cfg.embedding_size, "mmproj v.position_embd 长度与 unified vision 期望不符(布局错位)");
    }

    /// 纯色图位置编码诊断:纯色下所有 patch 内容向量相同,输出行差异完全来自
    /// 位置编码。正确的 2D 分解位置编码应满足:同排相邻(x+1)与同列相邻(y+1)
    /// 的差分各自构成稳定模式,且整体呈 base+g(x)+h(y) 加性结构。
    /// `ZLLM_GEMMA4_GGUF=... ZLLM_GEMMA4_IMAGE=... cargo test --lib flat_image_position -- --nocapture`
    #[cfg(target_os = "macos")]
    #[test]
    fn flat_image_position_encoding_diagnostics() {
        let Some(root) = std::env::var_os("ZLLM_GEMMA4_GGUF").map(PathBuf::from) else { return };
        let config = Gemma4Weights::select_config(&root).expect("select_config");
        let weights = Gemma4Weights::open(&root, config.clone()).expect("open");
        let mm = weights.load_multimodal_weights().expect("multimodal");
        if mm.vision.is_none() {
            return;
        }
        let vision_cfg = config.vision.expect("vision config");
        let cpu = CpuContext;
        let model = prepare_gemma4_multimodal_model(&cpu, &config, &mm).unwrap();
        let vision_model = model.vision.as_ref().unwrap();

        // 256×256 纯色图 → 5×5 patch 网格(256 soft tokens 上限内)
        let flat = RgbImage::new(256, 256, vec![137; 256 * 256 * 3]).unwrap();
        let image = preprocess_image(&flat, &vision_cfg, vision_cfg.default_soft_tokens).unwrap();
        println!("纯色图 grid: rows={}", image.rows);
        let side = (image.rows as f64).sqrt().round() as usize;

        let encode = |image: &Gemma4ImageInput| -> Vec<f32> { encode_unified_image(&cpu, &config, vision_model, image).unwrap().data };
        let output = encode(&image);
        let emb = vision_cfg.embedding_size;
        let row = |i: usize| -> &[f32] { &output[i * emb..(i + 1) * emb] };

        // 差分统计:相邻 patch(x 方向)与隔排 patch(y 方向)的 L2 距离
        fn l2(a: &[f32], b: &[f32]) -> f64 {
            a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt()
        }
        let base = row(0);
        let norms: Vec<f64> = (0..image.rows).map(|i| l2(base, row(i))).collect();
        println!("row(i) 相对 row(0) 的 L2:");
        for y in 0..side {
            let line: Vec<String> = (0..side).map(|x| format!("{:7.2}", norms[y * side + x])).collect();
            println!("  y={y}: {}", line.join(" "));
        }
        // 加性结构检验:dist(i,j) 应满足 |g(x_i,x_j) + h(y_i,y_j)| 的可分解性;
        // 简化检验:同排差分 dist((x,y),(x+1,y)) 与同列差分 dist((x,y),(x,y+1)) 分布
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        for y in 0..side {
            for x in 0..side {
                if x + 1 < side {
                    dx.push(l2(row(y * side + x), row(y * side + x + 1)));
                }
                if y + 1 < side {
                    dy.push(l2(row(y * side + x), row((y + 1) * side + x)));
                }
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let std = |v: &[f64], m: f64| (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        let (mx, my) = (mean(&dx), mean(&dy));
        println!("x 相邻差分: mean={mx:.3} std={:.3} | y 相邻差分: mean={my:.3} std={:.3}", std(&dx, mx), std(&dy, my));
        // 对角差分(位置编码完全失效时 dx≈dy≈0;乱码时模式混乱)
        assert!(mx > 0.0 && my > 0.0, "纯色图下位置编码无差异:视觉塔输出与位置无关");
    }

    #[test]
    fn panorama_uses_official_short_side_fallback() {
        let config = Gemma4Config::standard_12b().vision.unwrap();
        let image = RgbImage::new(10_000, 1, vec![255; 10_000 * 3]).unwrap();
        let input = preprocess_image(&image, &config, 280).unwrap();
        assert_eq!(input.rows, 280);
        assert_eq!(input.positions.last(), Some(&[279, 0]));
    }

    #[test]
    fn audio_is_padded_to_640_sample_tokens() {
        let config = Gemma4Config::standard_12b().audio.unwrap();
        let input = preprocess_audio(&vec![0.5; 641], &config).unwrap();
        assert_eq!(input.rows, 2);
        assert_eq!(input.data.len(), 1280);
        assert_eq!(input.data[641], 0.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn unified_vision_cpu_matches_metal() {
        let Some(root) = std::env::var_os("ZLLM_GEMMA4_ROOT").map(PathBuf::from) else {
            return;
        };
        let config = Gemma4Config::standard_12b();
        let weights = Gemma4Weights::open(&root, config.clone()).unwrap();
        let source = weights.load_multimodal_weights().unwrap();
        let cpu = CpuContext;
        let metal = MetalContext::new_default().unwrap();
        let cpu_model = prepare_gemma4_multimodal_model(&cpu, &config, &source).unwrap();
        let metal_model = prepare_gemma4_multimodal_model(&metal, &config, &source).unwrap();
        let image = Gemma4ImageInput { data: (0..48 * 48 * 3).map(|index| (index % 256) as f32 / 255.0).collect(), positions: vec![[0, 0]], grid_height: 1, grid_width: 1, rows: 1, cols: 48 * 48 * 3 };

        let cpu_weights = cpu_model.vision.as_ref().unwrap();
        let metal_weights = metal_model.vision.as_ref().unwrap();
        let compare = |name: &str, expected: &CpuTensor, actual: &crate::backend::metal::MetalTensor| {
            let actual = metal.tensor_to_f32(actual);
            let squared_error: f64 = expected.data.iter().zip(&actual).map(|(left, right)| f64::from(left - right).powi(2)).sum();
            let squared_expected: f64 = expected.data.iter().map(|value| f64::from(*value).powi(2)).sum();
            let relative_l2 = (squared_error / squared_expected.max(f64::EPSILON)).sqrt();
            let expected_range = expected.data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| (min.min(value), max.max(value)));
            let actual_range = actual.iter().filter(|value| value.is_finite()).fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| (min.min(value), max.max(value)));
            let non_finite = actual.iter().filter(|value| !value.is_finite()).count();
            println!("[gemma4-vision] {name} relative_l2={relative_l2:.6} expected={expected_range:?} actual={actual_range:?} non_finite={non_finite}");
            relative_l2
        };

        let MetalWeight::F16(metal_patch_weight) = &metal_weights.patch_dense.weight else {
            panic!("Unified vision patch dense 不是 F16 resident weight");
        };
        let cpu_patch_weight = CpuTensor { data: cpu_weights.patch_dense.weight.data()[..image.cols].to_vec(), rows: 1, cols: image.cols };
        let metal_patch_weight = metal.select_row(metal_patch_weight, 0).unwrap();
        compare("patch_weight_row0", &cpu_patch_weight, &metal_patch_weight);

        let cpu_input = CpuTensor { data: image.data.clone(), rows: 1, cols: image.cols };
        let metal_input = metal.vision_tensor_from_f32(&image.data, 1, image.cols).unwrap();
        compare("input", &cpu_input, &metal_input);
        let cpu_ln1 = cpu.layernorm_bias(&cpu_input, &cpu_weights.patch_ln1.weight, &cpu_weights.patch_ln1.bias, config.rms_eps).unwrap();
        let metal_ln1 = metal.layernorm_bias(&metal_input, &metal_weights.patch_ln1.weight, &metal_weights.patch_ln1.bias, config.rms_eps).unwrap();
        compare("ln1", &cpu_ln1, &metal_ln1);
        let mut cpu_dense = cpu.linear(&cpu_ln1, &cpu_weights.patch_dense.weight).unwrap();
        let bias = cpu_weights.patch_dense.bias.as_ref().unwrap().data();
        cpu_dense.data.iter_mut().zip(bias).for_each(|(value, bias)| *value += bias);
        let metal_dense = metal.linear(&metal_ln1, &metal_weights.patch_dense.weight).unwrap();
        let metal_dense = metal.add_bias(&metal_dense, metal_weights.patch_dense.bias.as_ref().unwrap()).unwrap();
        compare("dense", &cpu_dense, &metal_dense);
        let cpu_ln2 = cpu.layernorm_bias(&cpu_dense, &cpu_weights.patch_ln2.weight, &cpu_weights.patch_ln2.bias, config.rms_eps).unwrap();
        let metal_ln2 = metal.layernorm_bias(&metal_dense, &metal_weights.patch_ln2.weight, &metal_weights.patch_ln2.bias, config.rms_eps).unwrap();
        compare("ln2", &cpu_ln2, &metal_ln2);
        let position_embedding = &cpu_weights.position_embedding;
        let cpu_positions = CpuTensor { data: (0..config.hidden_size).map(|column| position_embedding[column] + position_embedding[config.hidden_size + column]).collect(), rows: 1, cols: config.hidden_size };
        let metal_positions = metal.vision_tensor_from_f32(&cpu_positions.data, 1, config.hidden_size).unwrap();
        let cpu_positioned = cpu.add(&cpu_ln2, &cpu_positions).unwrap();
        let metal_positioned = metal.add(&metal_ln2, &metal_positions).unwrap();
        compare("position", &cpu_positioned, &metal_positioned);
        let cpu_position_norm = cpu.layernorm_bias(&cpu_positioned, &cpu_weights.position_norm.weight, &cpu_weights.position_norm.bias, config.rms_eps).unwrap();
        let metal_position_norm = metal.layernorm_bias(&metal_positioned, &metal_weights.position_norm.weight, &metal_weights.position_norm.bias, config.rms_eps).unwrap();
        compare("position_norm", &cpu_position_norm, &metal_position_norm);
        let cpu_unit_norm = cpu.rmsnorm(&cpu_position_norm, &cpu_weights.unit_norm, config.rms_eps).unwrap();
        let metal_unit_norm = metal.rmsnorm(&metal_position_norm, &metal_weights.unit_norm, config.rms_eps).unwrap();
        compare("unit_norm", &cpu_unit_norm, &metal_unit_norm);
        let expected = cpu.linear(&cpu_unit_norm, &cpu_weights.projection).unwrap();
        let actual = metal.linear(&metal_unit_norm, &metal_weights.projection).unwrap();
        let relative_l2 = compare("projection", &expected, &actual);
        assert!(relative_l2 < 0.05, "Unified vision CPU/Metal relative_l2={relative_l2}");
    }
}
