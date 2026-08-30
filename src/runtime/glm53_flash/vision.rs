//! GLM-5.3-Flash 视觉塔(`glm5_next_vision`)编排与图文输入。
//!
//! 图像只在首次 prefill 编码:patch embed(conv3d 无重叠窗口,等价线性)
//! → 24 层 ViT(RMSNorm、fused qkv、per-head q/k RMSNorm、2D RoPE、
//! 非因果整图 attention、SwiGLU 限幅 MLP)→ post RMSNorm → 2×2 downsample
//! (权重已置换为线性)→ merger(proj → LayerNorm → GELU → SwiGLU)。
//! 输出行宽等于语言模型 hidden_size,直接替换 image token 的 embedding 行。

use crate::{
    backend::{BackendError, GqaPrefillBackend, LinearWeight, VisionBackend},
    moe::Activation,
    tokenizer::Tokenizer,
    vision::{ContentPart, ImageProcessor, ImageTensor, ImageTokenRange, MultimodalInput, MultimodalProcessor, PatchImageConfig, PatchImageProcessor, VisionGrid},
    weight::{
        container::safetensor::TensorData,
        model::glm53_flash::{Glm53FlashWeights, Glm53VisionBlockWeights, Glm53VisionDims, Glm53VisionPatchEmbedWeights, Glm53VisionTailWeights},
    },
};

const IMAGE_TOKEN: &str = "<|image|>";
const VISION_START_TOKEN: &str = "<|begin_of_image|>";
const VISION_END_TOKEN: &str = "<|end_of_image|>";

pub use crate::model_spec::glm53_flash::Glm53FlashVisionConfig;

impl Glm53FlashVisionConfig {
    /// 装配尺寸单一来源:runtime 配置到 weight 层校验副本的转换。
    pub fn dims(&self) -> Glm53VisionDims {
        Glm53VisionDims {
            hidden_size: self.hidden_size,
            depth: self.layer_count,
            num_heads: self.num_heads,
            patch_size: self.patch_size,
            temporal_patch_size: self.temporal_patch_size,
            spatial_merge_size: self.spatial_merge_size,
            intermediate_size: self.intermediate_size,
            projection_intermediate_size: self.projection_intermediate_size,
            out_hidden_size: self.out_hidden_size,
        }
    }
}

/// patch embed:conv3d kernel 与 stride 相同,无重叠,等价一次线性 + bias。
pub struct Glm53VisionPatchEmbed<W> {
    pub projection: W,
    pub bias: W,
}

/// qkv 融合投影 → per-head q/k RMSNorm(rope 前)→ 非因果 attention → 输出投影。
pub struct Glm53VisionAttention<W> {
    pub query_key_value: W,
    pub query_key_value_bias: W,
    pub query_norm: W,
    pub key_norm: W,
    pub output: W,
    pub output_bias: W,
}

/// gate/up 输入独立可并行;down 收敛。clamp 语义在激活 kernel 内。
pub struct Glm53VisionMlp<W> {
    pub gate: W,
    pub gate_bias: W,
    pub up: W,
    pub up_bias: W,
    pub down: W,
    pub down_bias: W,
}

pub struct Glm53VisionBlock<W> {
    pub input_norm: W,
    pub post_attention_norm: W,
    pub attention: Glm53VisionAttention<W>,
    pub mlp: Glm53VisionMlp<W>,
}

/// 2×2 downsample:权重已在 weight 层置换为 [out, merge²*hidden] 线性布局。
pub struct Glm53VisionDownsample<W> {
    pub projection: W,
    pub bias: W,
}

/// merger 串行链:proj → LayerNorm → GELU → SwiGLU(线性均无 bias)。
pub struct Glm53VisionMerger<W> {
    pub projection: W,
    pub norm_weight: W,
    pub norm_bias: W,
    pub gate: W,
    pub up: W,
    pub down: W,
}

pub struct Glm53VisionTail<W> {
    pub post_layernorm: W,
    pub downsample: Glm53VisionDownsample<W>,
    pub merger: Glm53VisionMerger<W>,
}

/// 图文输入处理:图像展开为 vision start/end 包裹的连续 image token。
pub struct Glm53FlashMultimodalProcessor {
    tokenizer: Tokenizer,
    image_processor: PatchImageProcessor,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
}

impl Glm53FlashMultimodalProcessor {
    pub fn new(tokenizer: Tokenizer, config: &Glm53FlashVisionConfig) -> Result<Self, String> {
        let image_processor = PatchImageProcessor::new(PatchImageConfig {
            patch_size: config.patch_size,
            temporal_patch_size: config.temporal_patch_size,
            merge_size: config.spatial_merge_size,
            min_pixels: config.min_pixels,
            max_pixels: config.max_pixels,
            max_aspect_ratio: config.max_aspect_ratio,
            mean: config.image_mean,
            std: config.image_std,
        })?;
        Ok(Self {
            image_token_id: special_token_id(&tokenizer, IMAGE_TOKEN)?,
            vision_start_token_id: special_token_id(&tokenizer, VISION_START_TOKEN)?,
            vision_end_token_id: special_token_id(&tokenizer, VISION_END_TOKEN)?,
            tokenizer,
            image_processor,
        })
    }

    /// 采样与输出侧需要拦截的视觉特殊 token(image / begin / end)。
    pub fn vision_special_token_ids(&self) -> [u32; 3] {
        [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
    }
}

impl MultimodalProcessor for Glm53FlashMultimodalProcessor {
    fn process(&self, parts: &[ContentPart<'_>]) -> Result<MultimodalInput, String> {
        let mut prompt = String::new();
        let mut images = Vec::new();
        let mut expected_image_tokens = Vec::new();

        for part in parts {
            match part {
                ContentPart::Text(text) => {
                    if text.contains(IMAGE_TOKEN) || text.contains(VISION_START_TOKEN) || text.contains(VISION_END_TOKEN) {
                        return Err("文本包含 GLM-5.3-Flash 保留的视觉特殊 token".to_owned());
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
        _ => Err(format!("GLM-5.3-Flash 特殊 token {token:?} 未映射到单一 token ID: {ids:?}")),
    }
}

/// 生成文本 embedding,并在设备内用视觉塔输出替换 image token 行。
/// 图像按 range 逐张独立编码(与官方 per-image attention 语义一致)。
#[allow(clippy::too_many_arguments)]
pub fn glm53_multimodal_embedding<B>(
    backend: &B,
    vision_config: &Glm53FlashVisionConfig,
    weights: &Glm53FlashWeights,
    text_embedding: &[f32],
    token_count: usize,
    hidden_size: usize,
    images: &[ImageTensor],
    ranges: &[ImageTokenRange],
) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    if hidden_size != vision_config.out_hidden_size {
        return Err(crate::runtime::compute_error(format!("GLM-5.3-Flash 视觉输出宽 {} 与文本 hidden_size {hidden_size} 不一致", vision_config.out_hidden_size)));
    }
    let expected = token_count.checked_mul(hidden_size).ok_or_else(|| crate::runtime::compute_error("多模态 embedding 大小溢出"))?;
    if text_embedding.len() != expected {
        return Err(crate::runtime::compute_error(format!("多模态文本 embedding values={}，期望 {expected}", text_embedding.len())));
    }
    let mut hidden = backend.vision_tensor_from_f32(text_embedding, token_count, hidden_size)?;
    let mut previous_end = 0;
    for range in ranges {
        if range.tokens.start < previous_end || range.tokens.end > token_count || range.tokens.start > range.tokens.end {
            return Err(crate::runtime::compute_error(format!("图像 {} token range {:?} 无效，token_count={token_count}", range.image_index, range.tokens)));
        }
        let image = images.get(range.image_index).ok_or_else(|| crate::runtime::compute_error(format!("缺少图像 {}", range.image_index)))?;
        let image_hidden = glm53_vision_encode(backend, vision_config, weights, image)?;
        let expected_rows = image.visual_token_count().map_err(crate::runtime::compute_error)?;
        let encoded_rows = backend.token_rows(&image_hidden);
        if encoded_rows != range.tokens.len() || encoded_rows != expected_rows {
            return Err(crate::runtime::compute_error(format!("图像 {} placeholder rows={}，视觉输出 rows={encoded_rows}", range.image_index, range.tokens.len())));
        }
        backend.scatter_rows(&mut hidden, range.tokens.start, &image_hidden)?;
        previous_end = range.tokens.end;
    }
    Ok(hidden)
}

/// 完整 Vision Tower:patch embed → ViT 层 → post norm → downsample → merger。
/// 输出 [patch_rows/merge², out_hidden_size],行序与 image token 展开一致。
pub fn glm53_vision_encode<B>(backend: &B, config: &Glm53FlashVisionConfig, weights: &Glm53FlashWeights, image: &ImageTensor) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    let expected_cols =
        3usize.checked_mul(config.temporal_patch_size).and_then(|value| value.checked_mul(config.patch_size)).and_then(|value| value.checked_mul(config.patch_size)).ok_or_else(|| crate::runtime::compute_error("视觉 patch columns 溢出"))?;
    if image.rows == 0 || image.cols != expected_cols || image.data.len() != image.rows * image.cols {
        return Err(crate::runtime::compute_error(format!("视觉 patch tensor=[{},{}] values={}，期望 cols={expected_cols}", image.rows, image.cols, image.data.len())));
    }

    let mut hidden = {
        let _scope = backend.layer_scope();
        let source = weights.load_vision_patch_embedding().map_err(BackendError::ExpertLoad)?;
        let patch = prepare_patch_embed(backend, &source)?;
        let input = backend.vision_tensor_from_f32(&image.data, image.rows, image.cols)?;
        backend.begin_batch();
        let result = (|| {
            let embedded = backend.linear(&input, &patch.projection)?;
            backend.add_bias(&embedded, &patch.bias)
        })();
        backend.finish_batch();
        result?
    };

    let head_dim = config.hidden_size / config.num_heads;
    let (cos, sin) = vision_rope(image.grid, image.merge_size, head_dim, config.rope_theta).map_err(crate::runtime::compute_error)?;
    let cos = backend.vision_tensor_from_f32(&cos, image.rows, head_dim)?;
    let sin = backend.vision_tensor_from_f32(&sin, image.rows, head_dim)?;

    for layer in 0..config.layer_count {
        let _scope = backend.layer_scope();
        let source = weights.load_vision_layer(layer).map_err(BackendError::ExpertLoad)?;
        let prepared = prepare_vision_block(backend, &source)?;
        backend.begin_batch();
        let result = vision_block(backend, config, &prepared, &hidden, &cos, &sin);
        backend.finish_batch();
        hidden = result?;
    }

    let output = {
        let _scope = backend.layer_scope();
        let source = weights.load_vision_tail().map_err(BackendError::ExpertLoad)?;
        let prepared = prepare_vision_tail(backend, &source)?;
        backend.begin_batch();
        let result = vision_tail(backend, config, &prepared, &hidden);
        backend.finish_batch();
        result?
    };

    let expected_rows = image.rows / (config.spatial_merge_size * config.spatial_merge_size);
    if backend.token_rows(&output) != expected_rows {
        return Err(crate::runtime::compute_error(format!("视觉输出 rows={}，期望 {expected_rows}", backend.token_rows(&output))));
    }
    Ok(output)
}

/// 一个 ViT 块:RMSNorm → attention → 残差;RMSNorm → SwiGLU MLP → 残差。
fn vision_block<B>(backend: &B, config: &Glm53FlashVisionConfig, layer: &Glm53VisionBlock<B::Weight>, hidden: &B::Tensor, cos: &B::Tensor, sin: &B::Tensor) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    let normed = backend.rmsnorm(hidden, &layer.input_norm, config.rms_norm_eps)?;
    let attention = vision_attention_block(backend, config, &layer.attention, &normed, cos, sin)?;
    let residual = backend.add(hidden, &attention)?;
    let normed = backend.rmsnorm(&residual, &layer.post_attention_norm, config.rms_norm_eps)?;
    let mlp = vision_mlp_block(backend, config, &layer.mlp, &normed)?;
    backend.add(&residual, &mlp)
}

fn vision_attention_block<B>(backend: &B, config: &Glm53FlashVisionConfig, attention: &Glm53VisionAttention<B::Weight>, normed: &B::Tensor, cos: &B::Tensor, sin: &B::Tensor) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    let head_dim = config.hidden_size / config.num_heads;
    let qkv = backend.linear(normed, &attention.query_key_value)?;
    let qkv = backend.add_bias(&qkv, &attention.query_key_value_bias)?;
    let (query, key_value) = backend.split_columns(&qkv, config.hidden_size)?;
    let (key, value) = backend.split_columns(&key_value, config.hidden_size)?;
    // 官方顺序:q/k per-head RMSNorm 在 RoPE 之前(与 GQA 文本层同款能力)。
    let query = backend.rmsnorm_heads(&query, &attention.query_norm, config.num_heads, head_dim, config.rms_norm_eps)?;
    let key = backend.rmsnorm_heads(&key, &attention.key_norm, config.num_heads, head_dim, config.rms_norm_eps)?;
    let attended = backend.vision_attention(&query, &key, &value, cos, sin, config.num_heads)?;
    let projected = backend.linear(&attended, &attention.output)?;
    backend.add_bias(&projected, &attention.output_bias)
}

fn vision_mlp_block<B: VisionBackend>(backend: &B, config: &Glm53FlashVisionConfig, mlp: &Glm53VisionMlp<B::Weight>, normed: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let (gate, up) = backend.dual_linear(normed, &mlp.gate, &mlp.up)?;
    let gate = backend.add_bias(&gate, &mlp.gate_bias)?;
    let up = backend.add_bias(&up, &mlp.up_bias)?;
    let activated = backend.gated_activation(&gate, &up, &Activation::SiluClamped { limit: config.swiglu_limit })?;
    let output = backend.linear(&activated, &mlp.down)?;
    backend.add_bias(&output, &mlp.down_bias)
}

/// post RMSNorm → 2×2 downsample(置换后的线性)→ merger(proj→LN→GELU→SwiGLU)。
fn vision_tail<B: VisionBackend>(backend: &B, config: &Glm53FlashVisionConfig, tail: &Glm53VisionTail<B::Weight>, hidden: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let normed = backend.rmsnorm(hidden, &tail.post_layernorm, config.rms_norm_eps)?;
    let merged = backend.merge_spatial(&normed, config.spatial_merge_size)?;
    let downsampled = backend.linear(&merged, &tail.downsample.projection)?;
    let downsampled = backend.add_bias(&downsampled, &tail.downsample.bias)?;
    let projected = backend.linear(&downsampled, &tail.merger.projection)?;
    let projected = backend.layernorm_bias(&projected, &tail.merger.norm_weight, &tail.merger.norm_bias, config.layer_norm_eps)?;
    let activated = backend.gelu(&projected)?;
    let (gate, up) = backend.dual_linear(&activated, &tail.merger.gate, &tail.merger.up)?;
    let gated = backend.gated_activation(&gate, &up, &Activation::SiluClamped { limit: config.swiglu_limit })?;
    backend.linear(&gated, &tail.merger.down)
}

fn prepare_patch_embed<B: VisionBackend>(backend: &B, source: &Glm53VisionPatchEmbedWeights) -> Result<Glm53VisionPatchEmbed<B::Weight>, BackendError> {
    Ok(Glm53VisionPatchEmbed { projection: prepare_matrix(backend, &source.projection)?, bias: prepare_vector(backend, &source.bias)? })
}

fn prepare_vision_block<B: VisionBackend>(backend: &B, source: &Glm53VisionBlockWeights) -> Result<Glm53VisionBlock<B::Weight>, BackendError> {
    Ok(Glm53VisionBlock {
        input_norm: prepare_vector(backend, &source.input_norm)?,
        post_attention_norm: prepare_vector(backend, &source.post_attention_norm)?,
        attention: Glm53VisionAttention {
            query_key_value: prepare_matrix(backend, &source.attention.query_key_value)?,
            query_key_value_bias: prepare_vector(backend, &source.attention.query_key_value_bias)?,
            query_norm: prepare_vector(backend, &source.attention.query_norm)?,
            key_norm: prepare_vector(backend, &source.attention.key_norm)?,
            output: prepare_matrix(backend, &source.attention.output)?,
            output_bias: prepare_vector(backend, &source.attention.output_bias)?,
        },
        mlp: Glm53VisionMlp {
            gate: prepare_matrix(backend, &source.mlp.gate)?,
            gate_bias: prepare_vector(backend, &source.mlp.gate_bias)?,
            up: prepare_matrix(backend, &source.mlp.up)?,
            up_bias: prepare_vector(backend, &source.mlp.up_bias)?,
            down: prepare_matrix(backend, &source.mlp.down)?,
            down_bias: prepare_vector(backend, &source.mlp.down_bias)?,
        },
    })
}

fn prepare_vision_tail<B: VisionBackend>(backend: &B, source: &Glm53VisionTailWeights) -> Result<Glm53VisionTail<B::Weight>, BackendError> {
    Ok(Glm53VisionTail {
        post_layernorm: prepare_vector(backend, &source.post_layernorm)?,
        downsample: Glm53VisionDownsample { projection: prepare_matrix(backend, &source.downsample)?, bias: prepare_vector(backend, &source.downsample_bias)? },
        merger: Glm53VisionMerger {
            projection: prepare_matrix(backend, &source.merger_projection)?,
            norm_weight: prepare_vector(backend, &source.merger_norm_weight)?,
            norm_bias: prepare_vector(backend, &source.merger_norm_bias)?,
            gate: prepare_matrix(backend, &source.merger_gate)?,
            up: prepare_matrix(backend, &source.merger_up)?,
            down: prepare_matrix(backend, &source.merger_down)?,
        },
    })
}

/// BF16 rank-2 线性权重(含已展平/已置换的 conv 权重);shape 由 weight 层校验。
fn prepare_matrix<B: VisionBackend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let [rows, cols] = tensor.shape.as_slice() else {
        return Err(crate::runtime::compute_error(format!("视觉线性权重 {} shape={:?} 不是 rank-2", tensor.name, tensor.shape)));
    };
    backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), *rows, *cols)
}

/// BF16 一维向量(bias / norm 权重)。
fn prepare_vector<B: VisionBackend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let elements = tensor.shape.iter().try_fold(1usize, |count, &dimension| count.checked_mul(dimension)).ok_or_else(|| crate::runtime::compute_error(format!("视觉向量 {} shape 溢出", tensor.name)))?;
    backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), 1, elements)
}

/// 2D 视觉 RoPE:h/w 各占 head_dim/2 旋转维(theta^(-2i/(head_dim/2))),
/// patch 行序已按 merge group 排列;角度重复两遍构成 SplitHalf 布局,
/// 与 backend `vision_attention` 内部的 rope kernel 约定一致。
fn vision_rope(grid: VisionGrid, merge_size: usize, head_dim: usize, theta: f32) -> Result<(Vec<f32>, Vec<f32>), String> {
    if grid.temporal == 0
        || grid.height == 0
        || grid.width == 0
        || merge_size == 0
        || !grid.height.is_multiple_of(merge_size)
        || !grid.width.is_multiple_of(merge_size)
        || head_dim == 0
        || !head_dim.is_multiple_of(4)
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(format!("视觉 RoPE 参数非法: grid={grid:?}, merge={merge_size}, head_dim={head_dim}"));
    }
    let axis_dim = head_dim / 2;
    let axis_pairs = axis_dim / 2;
    let rows = grid.temporal.checked_mul(grid.height).and_then(|value| value.checked_mul(grid.width)).ok_or("视觉 RoPE rows 溢出")?;
    let frequencies = (0..axis_pairs).map(|pair| theta.powf(-2.0 * pair as f32 / axis_dim as f32)).collect::<Vec<_>>();
    let mut cos = Vec::with_capacity(rows * head_dim);
    let mut sin = Vec::with_capacity(rows * head_dim);
    for _ in 0..grid.temporal {
        for block_y in 0..grid.height / merge_size {
            for block_x in 0..grid.width / merge_size {
                for merge_y in 0..merge_size {
                    for merge_x in 0..merge_size {
                        let positions = [block_y * merge_size + merge_y, block_x * merge_size + merge_x];
                        let angles = positions.iter().flat_map(|&position| frequencies.iter().map(move |&frequency| position as f32 * frequency)).collect::<Vec<_>>();
                        for _ in 0..2 {
                            cos.extend(angles.iter().map(|angle| angle.cos()));
                            sin.extend(angles.iter().map(|angle| angle.sin()));
                        }
                    }
                }
            }
        }
    }
    if cos.len() != rows * head_dim {
        return Err(format!("视觉 RoPE values={}，期望 {}", cos.len(), rows * head_dim));
    }
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 标准视觉配置自洽() {
        let config = Glm53FlashVisionConfig::standard();
        config.validate().unwrap();
        assert_eq!(config.hidden_size / config.num_heads, 64);
        let dims = config.dims();
        assert_eq!(dims.depth, 24);
        assert_eq!(dims.out_hidden_size, 4_096);
    }

    #[test]
    fn 头数不整除被拒绝() {
        let mut config = Glm53FlashVisionConfig::standard();
        config.num_heads = 15;
        assert!(config.validate().is_err());
    }

    #[test]
    fn vision_rope匹配merge顺序与单位圆() {
        // 4×4 patch 网格、merge 2:同 window 内 4 个 patch 位置连续,
        // 每行 head_dim=64 个角度,前半与后半重复(SplitHalf)。
        let grid = VisionGrid { temporal: 1, height: 4, width: 4 };
        let (cos, sin) = vision_rope(grid, 2, 64, 10_000.0).unwrap();
        assert_eq!(cos.len(), 16 * 64);
        for row in 0..16 {
            for offset in 0..32 {
                assert_eq!(cos[row * 64 + offset], cos[row * 64 + 32 + offset]);
            }
            let energy: f32 = (0..32).map(|offset| cos[row * 64 + offset] * cos[row * 64 + offset] + sin[row * 64 + offset] * sin[row * 64 + offset]).sum();
            assert!((energy - 32.0).abs() < 1.0e-3, "row {row} 能量和 {energy}");
        }
        // 位置 (0,0) 的第一个角度恒为 0;第 1 行是 merge window 内 (0,1) patch:
        // h=0(w 频率段从 16 起),第 2 行是 (1,0) patch:h=1。
        assert_eq!(cos[0], 1.0);
        assert_eq!(cos[64], 1.0);
        assert!(cos[80] < 1.0 && cos[80] > 0.0);
        assert!(cos[128] < 1.0 && cos[128] > 0.0);
        // merge window 内 (1,1) patch(第 3 行)与 (1,0)(第 2 行)w 位置不同。
        assert_ne!(cos[2 * 64 + axis_w_offset()], cos[3 * 64 + axis_w_offset()]);
    }

    fn axis_w_offset() -> usize {
        // h 占 head_dim/4=16 个频率,w 紧随其后。
        16
    }
}
