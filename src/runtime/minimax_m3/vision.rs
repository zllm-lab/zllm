//! MiniMax-M3 图文输入与视觉塔编排。
//!
//! 图像只在首次 prefill 编码；输出直接写入文本 embedding 的 image token 行。

use crate::{
    backend::{BackendError, LinearWeight, VisionBackend},
    runtime::minimax_m3::{MiniMaxM3Config, MiniMaxM3VisionConfig},
    tokenizer::Tokenizer,
    vision::{ContentPart, ImageProcessor, ImageTensor, ImageTokenRange, MultimodalInput, MultimodalProcessor, PatchImageConfig, PatchImageProcessor, RgbImage, VisionGrid},
    weight::{
        container::safetensor::TensorData,
        model::minimax_m3::MiniMaxM3Weights,
        model::vision::{LayerNormWeights, LinearWeights, VisionEncoderLayerWeights, VisionMlpWeights},
    },
};

const IMAGE_TOKEN: &str = "]<]image[>[";
const VISION_START_TOKEN: &str = "]<]start of image[>[";
const VISION_END_TOKEN: &str = "]<]end of image[>[";

pub struct MiniMaxM3ImageProcessor {
    patch: PatchImageProcessor,
}

impl MiniMaxM3ImageProcessor {
    pub fn new(config: &MiniMaxM3VisionConfig) -> Result<Self, String> {
        let patch = PatchImageProcessor::new(PatchImageConfig {
            patch_size: config.patch_size,
            temporal_patch_size: config.temporal_patch_size,
            merge_size: config.spatial_merge_size,
            min_pixels: config.min_pixels,
            max_pixels: config.max_pixels,
            max_aspect_ratio: config.max_aspect_ratio,
            mean: config.image_mean,
            std: config.image_std,
        })?;
        Ok(Self { patch })
    }
}

impl ImageProcessor for MiniMaxM3ImageProcessor {
    fn preprocess(&self, image: &RgbImage) -> Result<ImageTensor, String> {
        self.patch.preprocess(image)
    }
}

pub struct MiniMaxM3MultimodalProcessor {
    tokenizer: Tokenizer,
    image_processor: MiniMaxM3ImageProcessor,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
}

impl MiniMaxM3MultimodalProcessor {
    pub fn new(tokenizer: Tokenizer, config: &MiniMaxM3VisionConfig) -> Result<Self, String> {
        let image_token_id = special_token_id(&tokenizer, IMAGE_TOKEN)?;
        let vision_start_token_id = special_token_id(&tokenizer, VISION_START_TOKEN)?;
        let vision_end_token_id = special_token_id(&tokenizer, VISION_END_TOKEN)?;
        Ok(Self { tokenizer, image_processor: MiniMaxM3ImageProcessor::new(config)?, image_token_id, vision_start_token_id, vision_end_token_id })
    }
}

impl MultimodalProcessor for MiniMaxM3MultimodalProcessor {
    fn process(&self, parts: &[ContentPart<'_>]) -> Result<MultimodalInput, String> {
        let mut prompt = String::new();
        let mut images = Vec::new();
        let mut expected_image_tokens = Vec::new();

        for part in parts {
            match part {
                ContentPart::Text(text) => {
                    if text.contains(IMAGE_TOKEN) || text.contains(VISION_START_TOKEN) || text.contains(VISION_END_TOKEN) {
                        return Err("文本包含 MiniMax-M3 保留的视觉特殊 token".to_owned());
                    }
                    prompt.push_str(text);
                }
                ContentPart::Image(image) => {
                    let tensor = self.image_processor.preprocess(image)?;
                    let token_count = tensor.visual_token_count()?;
                    prompt.push_str(VISION_START_TOKEN);
                    for _ in 0..token_count {
                        prompt.push_str(IMAGE_TOKEN);
                    }
                    prompt.push_str(VISION_END_TOKEN);
                    expected_image_tokens.push(token_count);
                    images.push(tensor);
                }
            }
        }

        let token_ids = self.tokenizer.tokenize(prompt.as_bytes());
        let mut image_token_ranges = Vec::with_capacity(images.len());
        let mut cursor = 0;
        for (image_index, &token_count) in expected_image_tokens.iter().enumerate() {
            let start = token_ids[cursor..].iter().position(|token| *token == self.vision_start_token_id).map(|position| cursor + position).ok_or_else(|| format!("图像 {image_index} 缺少 vision start token"))?;
            let image_start = start + 1;
            let image_end = image_start.checked_add(token_count).ok_or_else(|| "图像 token range 溢出".to_owned())?;
            if token_ids.get(image_start..image_end).is_none_or(|tokens| tokens.iter().any(|token| *token != self.image_token_id)) {
                return Err(format!("图像 {image_index} 的 image token 数量或顺序异常"));
            }
            if token_ids.get(image_end) != Some(&self.vision_end_token_id) {
                return Err(format!("图像 {image_index} 缺少 vision end token"));
            }
            image_token_ranges.push(ImageTokenRange { image_index, tokens: image_start..image_end });
            cursor = image_end + 1;
        }

        Ok(MultimodalInput { token_ids, images, image_token_ranges })
    }
}

fn special_token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, String> {
    let ids = tokenizer.tokenize(token.as_bytes());
    match ids.as_slice() {
        [id] => Ok(*id),
        _ => Err(format!("MiniMax-M3 特殊 token {token:?} 未映射到单一 token ID: {ids:?}")),
    }
}

struct PreparedLinear<W> {
    weight: W,
    bias: Option<W>,
}

struct PreparedNorm<W> {
    weight: W,
    bias: W,
}

struct PreparedVisionLayer<W> {
    input_norm: PreparedNorm<W>,
    query: PreparedLinear<W>,
    key: PreparedLinear<W>,
    value: PreparedLinear<W>,
    output: PreparedLinear<W>,
    post_attention_norm: PreparedNorm<W>,
    mlp_input: PreparedLinear<W>,
    mlp_output: PreparedLinear<W>,
}

/// 生成文本 embedding，并在设备内用视觉输出替换 image placeholder。
pub fn minimax_m3_multimodal_embedding<B>(backend: &B, config: &MiniMaxM3Config, weights: &MiniMaxM3Weights, text_embedding: &[f32], token_count: usize, images: &[ImageTensor], ranges: &[ImageTokenRange]) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend,
{
    if text_embedding.len() != token_count.checked_mul(config.hidden_size).ok_or_else(|| crate::runtime::compute_error("多模态 embedding 大小溢出"))? {
        return Err(crate::runtime::compute_error(format!("多模态文本 embedding values={}，期望 {}x{}", text_embedding.len(), token_count, config.hidden_size,)));
    }
    let mut hidden = backend.vision_tensor_from_f32(text_embedding, token_count, config.hidden_size)?;
    let mut previous_end = 0;
    for range in ranges {
        if range.tokens.start < previous_end || range.tokens.end > token_count || range.tokens.start > range.tokens.end {
            return Err(crate::runtime::compute_error(format!("图像 {} token range {:?} 无效，token_count={token_count}", range.image_index, range.tokens)));
        }
        let image = images.get(range.image_index).ok_or_else(|| crate::runtime::compute_error(format!("缺少图像 {}", range.image_index)))?;
        let image_hidden = minimax_m3_vision_encode(backend, &config.vision, weights, image)?;
        let rows = range.tokens.len();
        let expected_rows = image.visual_token_count().map_err(crate::runtime::compute_error)?;
        if rows != expected_rows {
            return Err(crate::runtime::compute_error(format!("图像 {} placeholder rows={rows}，视觉输出 rows={expected_rows}", range.image_index)));
        }
        backend.scatter_rows(&mut hidden, range.tokens.start, &image_hidden)?;
        previous_end = range.tokens.end;
    }
    Ok(hidden)
}

/// 完整 Vision Tower：patch embedding -> 32 层 ViT -> projector -> 2x2 merge projector。
pub fn minimax_m3_vision_encode<B>(backend: &B, config: &MiniMaxM3VisionConfig, weights: &MiniMaxM3Weights, image: &ImageTensor) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend,
{
    let expected_cols =
        3usize.checked_mul(config.temporal_patch_size).and_then(|value| value.checked_mul(config.patch_size)).and_then(|value| value.checked_mul(config.patch_size)).ok_or_else(|| crate::runtime::compute_error("视觉 patch columns 溢出"))?;
    if image.rows == 0 || image.cols != expected_cols || image.data.len() != image.rows * image.cols {
        return Err(crate::runtime::compute_error(format!("视觉 patch tensor shape 非法: rows={} cols={} values={}，期望 cols={expected_cols}", image.rows, image.cols, image.data.len(),)));
    }

    let mut hidden = {
        let _scope = backend.layer_scope();
        let input = backend.vision_tensor_from_f32(&image.data, image.rows, image.cols)?;
        let patch = prepare_linear(backend, &weights.load_vision_patch_embedding().map_err(BackendError::ExpertLoad)?)?;
        let norm = prepare_norm(backend, &weights.load_vision_pre_norm().map_err(BackendError::ExpertLoad)?)?;
        backend.begin_batch();
        let result = (|| {
            let embedded = linear(backend, &input, &patch)?;
            backend.layernorm_bias(&embedded, &norm.weight, &norm.bias, config.layer_norm_eps)
        })();
        backend.finish_batch();
        result?
    };
    let head_dim = config.hidden_size / config.num_heads;
    let (cos, sin) = vision_rope(image.grid, image.merge_size, head_dim, config.rope_theta).map_err(crate::runtime::compute_error)?;
    let rotary_dim = cos.len() / image.rows;
    let cos = backend.vision_tensor_from_f32(&cos, image.rows, rotary_dim)?;
    let sin = backend.vision_tensor_from_f32(&sin, image.rows, rotary_dim)?;

    for layer in 0..config.layer_count {
        let _scope = backend.layer_scope();
        let source = weights.load_vision_layer(layer).map_err(BackendError::ExpertLoad)?;
        let prepared = prepare_layer(backend, &source)?;
        backend.begin_batch();
        let result = encoder_layer(backend, config, &prepared, &hidden, &cos, &sin);
        backend.finish_batch();
        hidden = result?;
    }

    let projected = {
        let _scope = backend.layer_scope();
        let projector = prepare_mlp(backend, &weights.load_vision_patch_projector().map_err(BackendError::ExpertLoad)?)?;
        backend.begin_batch();
        let result = mlp(backend, &hidden, &projector);
        backend.finish_batch();
        result?
    };
    let output = {
        let _scope = backend.layer_scope();
        let projector = prepare_mlp(backend, &weights.load_vision_merge_projector().map_err(BackendError::ExpertLoad)?)?;
        backend.begin_batch();
        let result = (|| {
            let merged = backend.merge_spatial(&projected, config.spatial_merge_size)?;
            mlp(backend, &merged, &projector)
        })();
        backend.finish_batch();
        result?
    };
    Ok(output)
}

fn encoder_layer<B: VisionBackend>(backend: &B, config: &MiniMaxM3VisionConfig, layer: &PreparedVisionLayer<B::Weight>, hidden: &B::Tensor, cos: &B::Tensor, sin: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let normed = backend.layernorm_bias(hidden, &layer.input_norm.weight, &layer.input_norm.bias, config.layer_norm_eps)?;
    let (query, key) = backend.dual_linear(&normed, &layer.query.weight, &layer.key.weight)?;
    let value = backend.linear(&normed, &layer.value.weight)?;
    let query = add_optional_bias(backend, query, layer.query.bias.as_ref())?;
    let key = add_optional_bias(backend, key, layer.key.bias.as_ref())?;
    let value = add_optional_bias(backend, value, layer.value.bias.as_ref())?;
    let attention = backend.vision_attention(&query, &key, &value, cos, sin, config.num_heads)?;
    let attention = linear(backend, &attention, &layer.output)?;
    let residual = backend.add(hidden, &attention)?;

    let normed = backend.layernorm_bias(&residual, &layer.post_attention_norm.weight, &layer.post_attention_norm.bias, config.layer_norm_eps)?;
    let output = linear(backend, &normed, &layer.mlp_input)?;
    let output = backend.gelu(&output)?;
    let output = linear(backend, &output, &layer.mlp_output)?;
    backend.add(&residual, &output)
}

fn mlp<B: VisionBackend>(backend: &B, input: &B::Tensor, mlp: &(PreparedLinear<B::Weight>, PreparedLinear<B::Weight>)) -> Result<B::Tensor, BackendError> {
    let hidden = linear(backend, input, &mlp.0)?;
    let hidden = backend.gelu(&hidden)?;
    linear(backend, &hidden, &mlp.1)
}

fn linear<B: VisionBackend>(backend: &B, input: &B::Tensor, linear: &PreparedLinear<B::Weight>) -> Result<B::Tensor, BackendError> {
    let output = backend.linear(input, &linear.weight)?;
    add_optional_bias(backend, output, linear.bias.as_ref())
}

fn add_optional_bias<B: VisionBackend>(backend: &B, input: B::Tensor, bias: Option<&B::Weight>) -> Result<B::Tensor, BackendError> {
    match bias {
        Some(bias) => backend.add_bias(&input, bias),
        None => Ok(input),
    }
}

fn prepare_layer<B: VisionBackend>(backend: &B, weights: &VisionEncoderLayerWeights) -> Result<PreparedVisionLayer<B::Weight>, BackendError> {
    Ok(PreparedVisionLayer {
        input_norm: prepare_norm(backend, &weights.input_norm)?,
        query: prepare_linear(backend, &weights.attention.query)?,
        key: prepare_linear(backend, &weights.attention.key)?,
        value: prepare_linear(backend, &weights.attention.value)?,
        output: prepare_linear(backend, &weights.attention.output)?,
        post_attention_norm: prepare_norm(backend, &weights.post_attention_norm)?,
        mlp_input: prepare_linear(backend, &weights.mlp.input)?,
        mlp_output: prepare_linear(backend, &weights.mlp.output)?,
    })
}

#[allow(clippy::type_complexity)]
fn prepare_mlp<B: VisionBackend>(backend: &B, weights: &VisionMlpWeights) -> Result<(PreparedLinear<B::Weight>, PreparedLinear<B::Weight>), BackendError> {
    Ok((prepare_linear(backend, &weights.input)?, prepare_linear(backend, &weights.output)?))
}

fn prepare_linear<B: VisionBackend>(backend: &B, weights: &LinearWeights) -> Result<PreparedLinear<B::Weight>, BackendError> {
    if weights.weight.shape.len() != 2 {
        return Err(crate::runtime::compute_error(format!("视觉 linear {} shape={:?} 不是 rank-2", weights.weight.name, weights.weight.shape)));
    }
    let rows = weights.weight.shape[0];
    let cols = weights.weight.shape[1];
    let weight = if weights.weight.dtype == "BF16" {
        backend.prepare_weight(LinearWeight::Bf16Bytes(&weights.weight.data), rows, cols)?
    } else {
        let values = weights.weight.to_f32().map_err(crate::runtime::compute_error)?;
        backend.prepare_weight(LinearWeight::F32(&values), rows, cols)?
    };
    let bias = weights.bias.as_ref().map(|bias| prepare_vector(backend, bias, rows)).transpose()?;
    Ok(PreparedLinear { weight, bias })
}

fn prepare_norm<B: VisionBackend>(backend: &B, weights: &LayerNormWeights) -> Result<PreparedNorm<B::Weight>, BackendError> {
    let columns = weights.weight.shape.first().copied().ok_or_else(|| crate::runtime::compute_error(format!("视觉 norm {} shape 为空", weights.weight.name)))?;
    Ok(PreparedNorm { weight: prepare_vector(backend, &weights.weight, columns)?, bias: prepare_vector(backend, &weights.bias, columns)? })
}

fn prepare_vector<B: VisionBackend>(backend: &B, tensor: &TensorData, columns: usize) -> Result<B::Weight, BackendError> {
    let elements = tensor.shape.iter().try_fold(1usize, |count, value| count.checked_mul(*value)).ok_or_else(|| crate::runtime::compute_error(format!("{} shape 大小溢出", tensor.name)))?;
    if elements != columns {
        return Err(crate::runtime::compute_error(format!("视觉向量 {} shape={:?}，期望 {columns} 个元素", tensor.name, tensor.shape)));
    }
    if tensor.dtype == "BF16" {
        backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), 1, columns)
    } else {
        let values = tensor.to_f32().map_err(crate::runtime::compute_error)?;
        backend.prepare_weight(LinearWeight::F32(&values), 1, columns)
    }
}

/// M3 的 3D RoPE 为 temporal/height/width 等分旋转维度，不能整除的尾维原样通过。
/// image processor 已按 merge group 排列 patch，这里生成同序 position。
fn vision_rope(grid: VisionGrid, merge_size: usize, head_dim: usize, theta: f32) -> Result<(Vec<f32>, Vec<f32>), String> {
    if grid.temporal == 0 || grid.height == 0 || grid.width == 0 || merge_size == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(format!("视觉 RoPE 参数非法: grid={grid:?}, merge={merge_size}, head_dim={head_dim}"));
    }
    if !grid.height.is_multiple_of(merge_size) || !grid.width.is_multiple_of(merge_size) || !theta.is_finite() || theta <= 0.0 {
        return Err(format!("视觉 RoPE grid/merge/theta 非法: grid={grid:?}, merge={merge_size}, theta={theta}"));
    }
    let rows = grid.temporal.checked_mul(grid.height).and_then(|value| value.checked_mul(grid.width)).ok_or("视觉 RoPE rows 溢出")?;
    let axis_dim = 2 * ((head_dim / 3) / 2);
    let axis_pairs = axis_dim / 2;
    let rotary_dim = axis_dim * 3;
    if rotary_dim == 0 {
        return Err(format!("视觉 RoPE head_dim={head_dim} 无法分配 3D 旋转维度"));
    }
    let mut cosine = Vec::with_capacity(rows * rotary_dim);
    let mut sine = Vec::with_capacity(rows * rotary_dim);
    for temporal in 0..grid.temporal {
        for block_y in 0..grid.height / merge_size {
            for block_x in 0..grid.width / merge_size {
                for merge_y in 0..merge_size {
                    for merge_x in 0..merge_size {
                        let positions = [temporal, block_y * merge_size + merge_y, block_x * merge_size + merge_x];
                        let mut angles = Vec::with_capacity(rotary_dim / 2);
                        for position in positions {
                            for pair in 0..axis_pairs {
                                let inverse_frequency = theta.powf(-((2 * pair) as f32) / axis_dim as f32);
                                angles.push(position as f32 * inverse_frequency);
                            }
                        }
                        for &angle in &angles {
                            cosine.push(angle.cos());
                            sine.push(angle.sin());
                        }
                        for &angle in &angles {
                            cosine.push(angle.cos());
                            sine.push(angle.sin());
                        }
                    }
                }
            }
        }
    }
    if cosine.len() != rows * rotary_dim {
        return Err(format!("视觉 RoPE values={}，期望 {}", cosine.len(), rows * rotary_dim));
    }
    Ok((cosine, sine))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_rope_matches_merge_order_and_unit_circle() {
        let grid = VisionGrid { temporal: 1, height: 2, width: 2 };
        let (cos, sin) = vision_rope(grid, 2, 80, 10_000.0).unwrap();
        assert_eq!(cos.len(), 4 * 78);
        assert_eq!(sin.len(), cos.len());
        for (cos, sin) in cos.iter().zip(&sin) {
            assert!((cos * cos + sin * sin - 1.0).abs() < 1.0e-5);
        }
    }
}
