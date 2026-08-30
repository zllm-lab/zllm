//! Qwen3 家族 runtime:经典 dense Qwen3 文本模型 + Qwen3-VL 单图识别、DeepStack
//! 与 dense GQA 编排。两者共享同一份文本层(`text_layer`)——dense 即去掉 vision/M-RoPE
//! 的 VL 文本路径;三轴等位的 M-RoPE 与标准 RoPE 数值等价(`qwen3_vl_mrope_table`
//! 的 frequency 只依赖全局 pair 下标)。

pub mod cpu;
#[cfg(feature = "with-cuda")]
pub mod cuda;
pub mod protocol;

use std::ops::Range;

use half::f16;

use crate::{
    attention::{
        gqa::{CausalWindow, GqaSpec},
        rope::RotaryLayout,
    },
    backend::{Backend, BackendError, GqaPrefillBackend, LinearWeight, VisionBackend},
    moe::{Activation, FeedforwardSpec, dense_mlp::DenseMlpSpec},
    runtime::{LayerId, LayerSpec, Model, ModelError},
    tokenizer::Tokenizer,
    vision::{ImageProcessor, ImageTensor, PatchImageConfig, PatchImageProcessor, RgbImage, VisionGrid},
    weight::{
        container::safetensor::TensorData,
        model::qwen3_vl::{Qwen3TextSource, Qwen3VlLayerWeights, Qwen3VlMatrix, Qwen3VlVisionLayerWeights, Qwen3VlVisionMergerWeights, Qwen3VlWeights},
        model::vision::{LayerNormWeights, LinearWeights},
    },
};

pub(crate) const IMAGE_TOKEN: &str = "<|image_pad|>";
pub(crate) const VIDEO_TOKEN: &str = "<|video_pad|>";
pub(crate) const VISION_START_TOKEN: &str = "<|vision_start|>";
pub(crate) const VISION_END_TOKEN: &str = "<|vision_end|>";

pub use protocol::chat_prompt;

pub struct Qwen3VlImageProcessor {
    patch: PatchImageProcessor,
}

impl Qwen3VlImageProcessor {
    pub fn new(config: &Qwen3VlVisionConfig) -> Result<Self, String> {
        Ok(Self {
            patch: PatchImageProcessor::new(PatchImageConfig {
                patch_size: config.patch_size,
                temporal_patch_size: config.temporal_patch_size,
                merge_size: config.spatial_merge_size,
                min_pixels: config.min_pixels,
                max_pixels: config.max_pixels,
                max_aspect_ratio: config.max_aspect_ratio,
                mean: config.image_mean,
                std: config.image_std,
            })?,
        })
    }

    /// 视频帧必须已按 2 FPS 采样，并按 temporal patch=2 补成偶数帧。
    pub fn preprocess_video(&self, frames: &[RgbImage]) -> Result<ImageTensor, String> {
        if frames.is_empty() || !frames.len().is_multiple_of(2) {
            return Err(format!("Qwen3-VL video frames={}，必须是非零偶数", frames.len()));
        }
        let mut data = Vec::new();
        let mut grid = None;
        let mut merge_size = 0usize;
        let mut rows_per_pair = 0usize;
        let mut columns = 0usize;
        for (pair_index, pair) in frames.chunks_exact(2).enumerate() {
            let first = self.patch.preprocess(&pair[0])?;
            let second = self.patch.preprocess(&pair[1])?;
            if first.grid.temporal != 1
                || second.grid.temporal != 1
                || first.grid.height != second.grid.height
                || first.grid.width != second.grid.width
                || first.rows != second.rows
                || first.cols != second.cols
                || first.merge_size != second.merge_size
            {
                return Err(format!("Qwen3-VL video pair {pair_index} 两帧 patch grid 不一致"));
            }
            if let Some(expected) = grid {
                if expected != first.grid || rows_per_pair != first.rows || columns != first.cols || merge_size != first.merge_size {
                    return Err(format!("Qwen3-VL video pair {pair_index} 与首个 pair grid 不一致"));
                }
            } else {
                grid = Some(first.grid);
                rows_per_pair = first.rows;
                columns = first.cols;
                merge_size = first.merge_size;
                data.reserve(frames.len() / 2 * rows_per_pair * columns);
            }
            let patch_pixels = columns.checked_div(3 * 2).ok_or("Qwen3-VL video patch columns 非法")?;
            for row in 0..first.rows {
                let first_row = &first.data[row * columns..(row + 1) * columns];
                let second_row = &second.data[row * columns..(row + 1) * columns];
                for channel in 0..3 {
                    let base = channel * 2 * patch_pixels;
                    data.extend_from_slice(&first_row[base..base + patch_pixels]);
                    data.extend_from_slice(&second_row[base..base + patch_pixels]);
                }
            }
        }
        let spatial = grid.ok_or("Qwen3-VL video grid 为空")?;
        let temporal = frames.len() / 2;
        let rows = temporal.checked_mul(rows_per_pair).ok_or("Qwen3-VL video rows 溢出")?;
        Ok(ImageTensor { data, rows, cols: columns, grid: VisionGrid { temporal, height: spatial.height, width: spatial.width }, merge_size })
    }
}

impl ImageProcessor for Qwen3VlImageProcessor {
    fn preprocess(&self, image: &RgbImage) -> Result<ImageTensor, String> {
        self.patch.preprocess(image)
    }
}

pub struct Qwen3VlMultimodalInput {
    pub token_ids: Vec<u32>,
    pub image: ImageTensor,
    pub image_tokens: Range<usize>,
    pub position_ids: [Vec<usize>; 3],
    pub rope_delta: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3VlVisualKind {
    Image,
    Video,
}

pub struct Qwen3VlH3VisualInput {
    pub kind: Qwen3VlVisualKind,
    pub grid: VisionGrid,
    pub merge_size: usize,
}

pub struct Qwen3VlH3Input {
    pub token_ids: Vec<u32>,
    pub visual_tokens: Vec<Range<usize>>,
    pub position_ids: [Vec<usize>; 3],
    pub rope_delta: i64,
}

pub fn qwen3_vl_h3_input(tokenizer: &Tokenizer, config: &Qwen3VlConfig, prompt: &str, visuals: &[Qwen3VlH3VisualInput]) -> Result<Qwen3VlH3Input, String> {
    if prompt.contains(IMAGE_TOKEN) || prompt.contains(VIDEO_TOKEN) || prompt.contains(VISION_START_TOKEN) || prompt.contains(VISION_END_TOKEN) {
        return Err("用户文本包含 Qwen3-VL 保留视觉 token".to_owned());
    }
    let mut token_ids = Vec::new();
    let mut visual_tokens = Vec::with_capacity(visuals.len());
    let mut image_index = 0usize;
    let mut video_index = 0usize;
    for visual in visuals {
        if visual.merge_size == 0 || !visual.grid.height.is_multiple_of(visual.merge_size) || !visual.grid.width.is_multiple_of(visual.merge_size) {
            return Err(format!("Qwen3-VL H3 visual grid={:?} merge={} 非法", visual.grid, visual.merge_size));
        }
        let rows = visual.grid.temporal.checked_mul(visual.grid.height / visual.merge_size).and_then(|value| value.checked_mul(visual.grid.width / visual.merge_size)).ok_or("Qwen3-VL H3 visual token 数溢出")?;
        if rows == 0 {
            return Err("Qwen3-VL H3 visual token 数为 0".to_owned());
        }
        let (label, token_id) = match visual.kind {
            Qwen3VlVisualKind::Image => {
                image_index += 1;
                (format!("<Picture {image_index}>: "), config.image_token_id)
            }
            Qwen3VlVisualKind::Video => {
                video_index += 1;
                (format!("<Video {video_index}>: "), config.video_token_id)
            }
        };
        token_ids.extend(tokenizer.tokenize_with_special(label.as_bytes(), false));
        token_ids.push(config.vision_start_token_id);
        let start = token_ids.len();
        token_ids.resize(start + rows, token_id);
        visual_tokens.push(start..start + rows);
        token_ids.push(config.vision_end_token_id);
    }
    // H3 直接编码原始提示词，不使用 Qwen chat template，也不自动加入 BOS/EOS。
    token_ids.extend(tokenizer.tokenize_with_special(prompt.as_bytes(), false));
    if token_ids.is_empty() {
        return Err("Qwen3-VL H3 输入 token 数为 0".to_owned());
    }
    let descriptors = visual_tokens.iter().cloned().zip(visuals.iter().map(|visual| (visual.grid, visual.merge_size))).collect::<Vec<_>>();
    let (position_ids, rope_delta) = multimodal_positions_many(token_ids.len(), &descriptors)?;
    Ok(Qwen3VlH3Input { token_ids, visual_tokens, position_ids, rope_delta })
}

pub fn multimodal_positions_many(token_count: usize, visuals: &[(Range<usize>, (VisionGrid, usize))]) -> Result<([Vec<usize>; 3], i64), String> {
    let mut positions: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::with_capacity(token_count));
    let mut token_cursor = 0usize;
    let mut position_cursor = 0usize;
    for (index, (range, (grid, merge))) in visuals.iter().enumerate() {
        if range.start < token_cursor || range.start > range.end || range.end > token_count || *merge == 0 || !grid.height.is_multiple_of(*merge) || !grid.width.is_multiple_of(*merge) {
            return Err(format!("Qwen3-VL H3 visual {index} range/grid 非法"));
        }
        for _ in token_cursor..range.start {
            for axis in &mut positions {
                axis.push(position_cursor);
            }
            position_cursor += 1;
        }
        let grid_h = grid.height / *merge;
        let grid_w = grid.width / *merge;
        let expected = grid.temporal.checked_mul(grid_h).and_then(|value| value.checked_mul(grid_w)).ok_or("Qwen3-VL H3 position 数溢出")?;
        if expected != range.len() {
            return Err(format!("Qwen3-VL H3 visual {index} positions={expected} range={}", range.len()));
        }
        for temporal in 0..grid.temporal {
            for height in 0..grid_h {
                for width in 0..grid_w {
                    positions[0].push(position_cursor + temporal);
                    positions[1].push(position_cursor + height);
                    positions[2].push(position_cursor + width);
                }
            }
        }
        position_cursor += grid.temporal.max(grid_h).max(grid_w);
        token_cursor = range.end;
    }
    for _ in token_cursor..token_count {
        for axis in &mut positions {
            axis.push(position_cursor);
        }
        position_cursor += 1;
    }
    let max_position = positions.iter().flat_map(|axis| axis.iter()).copied().max().unwrap_or(0);
    Ok((positions, max_position as i64 + 1 - token_count as i64))
}

pub fn qwen3_vl_single_image_input(tokenizer: &Tokenizer, config: &Qwen3VlConfig, prompt: &str, image: &RgbImage) -> Result<Qwen3VlMultimodalInput, String> {
    if prompt.contains(IMAGE_TOKEN) || prompt.contains(VISION_START_TOKEN) || prompt.contains(VISION_END_TOKEN) {
        return Err("用户文本包含 Qwen3-VL 保留视觉 token".to_owned());
    }
    let image = Qwen3VlImageProcessor::new(config.vision.as_ref().ok_or("dense Qwen3 文本模型不支持图像输入".to_owned())?)?.preprocess(image)?;
    let visual_tokens = image.visual_token_count()?;
    let rendered = format!("<|im_start|>user\n{VISION_START_TOKEN}{}{VISION_END_TOKEN}{prompt}<|im_end|>\n<|im_start|>assistant\n", IMAGE_TOKEN.repeat(visual_tokens),);
    let token_ids = tokenizer.tokenize(rendered.as_bytes());
    let vision_start = token_ids.iter().position(|&token| token == config.vision_start_token_id).ok_or("Qwen3-VL prompt 缺少 vision_start token")?;
    let image_start = vision_start + 1;
    let image_end = image_start.checked_add(visual_tokens).ok_or("Qwen3-VL image token range 溢出")?;
    if token_ids.get(image_start..image_end).is_none_or(|tokens| tokens.iter().any(|&token| token != config.image_token_id)) {
        return Err(format!("Qwen3-VL image token 数量或顺序异常，期望 {visual_tokens}"));
    }
    if token_ids.get(image_end) != Some(&config.vision_end_token_id) {
        return Err("Qwen3-VL prompt 缺少 vision_end token".to_owned());
    }
    let image_tokens = image_start..image_end;
    let (position_ids, rope_delta) = multimodal_positions(token_ids.len(), &image_tokens, image.grid, image.merge_size)?;
    Ok(Qwen3VlMultimodalInput { token_ids, image, image_tokens, position_ids, rope_delta })
}

fn multimodal_positions(token_count: usize, image_tokens: &Range<usize>, grid: VisionGrid, merge_size: usize) -> Result<([Vec<usize>; 3], i64), String> {
    if image_tokens.start > image_tokens.end || image_tokens.end > token_count || merge_size == 0 {
        return Err(format!("Qwen3-VL image token range {image_tokens:?} 无效，tokens={token_count}"));
    }
    if !grid.height.is_multiple_of(merge_size) || !grid.width.is_multiple_of(merge_size) {
        return Err(format!("Qwen3-VL grid={grid:?} 不能按 merge={merge_size} 合并"));
    }
    let grid_h = grid.height / merge_size;
    let grid_w = grid.width / merge_size;
    let expected = grid.temporal.checked_mul(grid_h).and_then(|value| value.checked_mul(grid_w)).ok_or("Qwen3-VL visual token 数溢出")?;
    if expected != image_tokens.len() {
        return Err(format!("Qwen3-VL visual positions={expected}，image tokens={}", image_tokens.len()));
    }

    let mut positions: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::with_capacity(token_count));
    for position in 0..image_tokens.start {
        for axis in &mut positions {
            axis.push(position);
        }
    }
    let visual_start = image_tokens.start;
    for temporal in 0..grid.temporal {
        for height in 0..grid_h {
            for width in 0..grid_w {
                positions[0].push(visual_start + temporal);
                positions[1].push(visual_start + height);
                positions[2].push(visual_start + width);
            }
        }
    }
    let tail_start = visual_start + grid.temporal.max(grid_h).max(grid_w);
    for (current, _) in (tail_start..).zip(image_tokens.end..token_count) {
        for axis in &mut positions {
            axis.push(current);
        }
    }
    if positions.iter().any(|axis| axis.len() != token_count) {
        return Err("Qwen3-VL M-RoPE position 长度异常".to_owned());
    }
    let max_position = positions.iter().flat_map(|axis| axis.iter()).copied().max().unwrap_or(0);
    let rope_delta = max_position as i64 + 1 - token_count as i64;
    Ok((positions, rope_delta))
}

pub struct Qwen3VlRopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

pub fn qwen3_vl_mrope_table(config: &Qwen3VlConfig, position_ids: &[Vec<usize>; 3]) -> Result<Qwen3VlRopeTable, String> {
    let rows = position_ids[0].len();
    if rows == 0 || position_ids.iter().any(|axis| axis.len() != rows) {
        return Err("Qwen3-VL M-RoPE position shape 无效".to_owned());
    }
    let half = config.head_dim / 2;
    let mut cos = Vec::with_capacity(rows * half);
    let mut sin = Vec::with_capacity(rows * half);
    for row in 0..rows {
        for pair in 0..half {
            let axis = if pair < config.mrope_section[1] * 3 && pair % 3 == 1 {
                1
            } else if pair < config.mrope_section[2] * 3 && pair % 3 == 2 {
                2
            } else {
                0
            };
            let frequency = config.rope_theta.powf(-2.0 * pair as f32 / config.head_dim as f32);
            let angle = position_ids[axis][row] as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok(Qwen3VlRopeTable { cos, sin })
}

pub fn qwen3_vl_decode_rope_table(config: &Qwen3VlConfig, sequence_position: usize, rope_delta: i64) -> Result<Qwen3VlRopeTable, String> {
    let position = i64::try_from(sequence_position)
        .ok()
        .and_then(|position| position.checked_add(rope_delta))
        .and_then(|position| usize::try_from(position).ok())
        .ok_or_else(|| format!("Qwen3-VL decode position={sequence_position} delta={rope_delta} 无效"))?;
    qwen3_vl_mrope_table(config, &[vec![position], vec![position], vec![position]])
}

pub struct PreparedLinear<W> {
    pub weight: W,
    pub bias: Option<W>,
}

pub struct PreparedNorm<W> {
    pub weight: W,
    pub bias: W,
}

pub struct PreparedVisionLayer<W> {
    pub input_norm: PreparedNorm<W>,
    pub qkv: PreparedLinear<W>,
    pub output: PreparedLinear<W>,
    pub post_attention_norm: PreparedNorm<W>,
    pub mlp_input: PreparedLinear<W>,
    pub mlp_output: PreparedLinear<W>,
}

pub struct PreparedVisionMerger<W> {
    pub norm: PreparedNorm<W>,
    pub input: PreparedLinear<W>,
    pub output: PreparedLinear<W>,
    pub norm_after_merge: bool,
}

pub struct Qwen3VlVisionOutput<T> {
    pub embedding: T,
    pub deepstack: Vec<T>,
}

pub fn qwen3_vl_encode_image<B>(backend: &B, config: &Qwen3VlVisionConfig, weights: &Qwen3VlWeights, image: &ImageTensor) -> Result<Qwen3VlVisionOutput<B::Tensor>, BackendError>
where
    B: VisionBackend,
{
    let expected_columns = 3usize
        .checked_mul(config.temporal_patch_size)
        .and_then(|value| value.checked_mul(config.patch_size))
        .and_then(|value| value.checked_mul(config.patch_size))
        .ok_or_else(|| crate::runtime::compute_error("Qwen3-VL patch columns 溢出"))?;
    if image.rows == 0 || image.cols != expected_columns || image.data.len() != image.rows * image.cols {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL patch tensor=[{},{}] values={}，期望 cols={expected_columns}", image.rows, image.cols, image.data.len(),)));
    }

    let mut hidden = {
        let _scope = backend.layer_scope();
        let source = weights.load_vision_patch_embedding().map_err(BackendError::ExpertLoad)?;
        let patch = prepare_linear(backend, &source)?;
        let position = weights.load_vision_position_embedding().map_err(BackendError::ExpertLoad)?;
        let position = interpolate_position_embedding(config, image.grid, &position).map_err(crate::runtime::compute_error)?;
        let input = backend.vision_tensor_from_f32(&image.data, image.rows, image.cols)?;
        let position = backend.vision_tensor_from_f32(&position, image.rows, config.hidden_size)?;
        backend.begin_batch();
        let result = (|| {
            let embedded = linear(backend, &input, &patch)?;
            backend.add(&embedded, &position)
        })();
        backend.finish_batch();
        result?
    };
    let (cos, sin) = vision_rope(image.grid, image.merge_size, config.hidden_size / config.num_heads, config.rope_theta).map_err(crate::runtime::compute_error)?;
    let cos = backend.vision_tensor_from_f32(&cos, image.rows, config.hidden_size / config.num_heads)?;
    let sin = backend.vision_tensor_from_f32(&sin, image.rows, config.hidden_size / config.num_heads)?;
    let mut deepstack = Vec::with_capacity(config.deepstack_visual_indexes.len());

    for layer in 0..config.depth {
        let _scope = backend.layer_scope();
        let source = weights.load_vision_layer(layer).map_err(BackendError::ExpertLoad)?;
        let prepared = prepare_vision_layer(backend, &source)?;
        backend.begin_batch();
        let result = (|| {
            let hidden = vision_layer(backend, config, &prepared, &hidden, &cos, &sin)?;
            if let Some(index) = config.deepstack_visual_indexes.iter().position(|&candidate| candidate == layer) {
                let source = weights.load_vision_deepstack_merger(index).map_err(BackendError::ExpertLoad)?;
                let merger = prepare_vision_merger(backend, &source)?;
                deepstack.push(vision_merger(backend, config, &merger, &hidden)?);
            }
            Ok(hidden)
        })();
        backend.finish_batch();
        hidden = result?;
    }
    let embedding = {
        let _scope = backend.layer_scope();
        let source = weights.load_vision_merger().map_err(BackendError::ExpertLoad)?;
        let merger = prepare_vision_merger(backend, &source)?;
        backend.begin_batch();
        let result = vision_merger(backend, config, &merger, &hidden);
        backend.finish_batch();
        result?
    };
    if deepstack.len() != config.deepstack_visual_indexes.len() {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL DeepStack features={}，期望 {}", deepstack.len(), config.deepstack_visual_indexes.len(),)));
    }
    Ok(Qwen3VlVisionOutput { embedding, deepstack })
}

pub fn qwen3_vl_encode_video<B>(backend: &B, config: &Qwen3VlVisionConfig, weights: &Qwen3VlWeights, video: &ImageTensor) -> Result<Qwen3VlVisionOutput<B::Tensor>, BackendError>
where
    B: VisionBackend,
{
    if video.grid.temporal == 0 {
        return Err(crate::runtime::compute_error("Qwen3-VL video temporal grid 为 0"));
    }
    qwen3_vl_encode_image(backend, config, weights, video)
}

pub(crate) fn vision_layer<B: VisionBackend>(backend: &B, config: &Qwen3VlVisionConfig, layer: &PreparedVisionLayer<B::Weight>, hidden: &B::Tensor, cos: &B::Tensor, sin: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let normed = backend.layernorm_bias(hidden, &layer.input_norm.weight, &layer.input_norm.bias, 1.0e-6)?;
    let qkv = linear(backend, &normed, &layer.qkv)?;
    let (query, kv) = backend.split_columns(&qkv, config.hidden_size)?;
    let (key, value) = backend.split_columns(&kv, config.hidden_size)?;
    let attention = backend.vision_attention(&query, &key, &value, cos, sin, config.num_heads)?;
    let attention = linear(backend, &attention, &layer.output)?;
    let residual = backend.add(hidden, &attention)?;
    let normed = backend.layernorm_bias(&residual, &layer.post_attention_norm.weight, &layer.post_attention_norm.bias, 1.0e-6)?;
    let output = linear(backend, &normed, &layer.mlp_input)?;
    let output = backend.gelu(&output)?;
    let output = linear(backend, &output, &layer.mlp_output)?;
    backend.add(&residual, &output)
}

pub(crate) fn vision_merger<B: VisionBackend>(backend: &B, config: &Qwen3VlVisionConfig, merger: &PreparedVisionMerger<B::Weight>, hidden: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let merged = if merger.norm_after_merge {
        let merged = backend.merge_spatial(hidden, config.spatial_merge_size)?;
        backend.layernorm_bias(&merged, &merger.norm.weight, &merger.norm.bias, 1.0e-6)?
    } else {
        let normalized = backend.layernorm_bias(hidden, &merger.norm.weight, &merger.norm.bias, 1.0e-6)?;
        backend.merge_spatial(&normalized, config.spatial_merge_size)?
    };
    let output = linear(backend, &merged, &merger.input)?;
    let output = backend.gelu(&output)?;
    linear(backend, &output, &merger.output)
}

pub(crate) fn interpolate_position_embedding(config: &Qwen3VlVisionConfig, grid: VisionGrid, tensor: &TensorData) -> Result<Vec<f32>, String> {
    let table = tensor.to_f32()?;
    let side = config.position_embeddings.isqrt();
    if side * side != config.position_embeddings || grid.temporal == 0 {
        return Err(format!("Qwen3-VL learned position grid={} image grid={grid:?} 不受支持", config.position_embeddings));
    }
    let coordinate = |index: usize, size: usize| {
        if size <= 1 { 0.0 } else { index as f32 * (side - 1) as f32 / (size - 1) as f32 }
    };
    let merge = config.spatial_merge_size;
    let mut output = Vec::with_capacity(grid.temporal * grid.height * grid.width * config.hidden_size);
    for _ in 0..grid.temporal {
        for block_y in 0..grid.height / merge {
            for block_x in 0..grid.width / merge {
                for merge_y in 0..merge {
                    for merge_x in 0..merge {
                        let y = coordinate(block_y * merge + merge_y, grid.height);
                        let x = coordinate(block_x * merge + merge_x, grid.width);
                        let y0 = y.floor() as usize;
                        let x0 = x.floor() as usize;
                        let y1 = (y0 + 1).min(side - 1);
                        let x1 = (x0 + 1).min(side - 1);
                        let wy = y - y0 as f32;
                        let wx = x - x0 as f32;
                        let taps = [(y0 * side + x0, (1.0 - wy) * (1.0 - wx)), (y0 * side + x1, (1.0 - wy) * wx), (y1 * side + x0, wy * (1.0 - wx)), (y1 * side + x1, wy * wx)];
                        for column in 0..config.hidden_size {
                            output.push(taps.iter().map(|&(row, weight)| table[row * config.hidden_size + column] * weight).sum());
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

pub(crate) fn vision_rope(grid: VisionGrid, merge_size: usize, head_dim: usize, theta: f32) -> Result<(Vec<f32>, Vec<f32>), String> {
    if grid.temporal == 0 || grid.height == 0 || grid.width == 0 || merge_size == 0 || !grid.height.is_multiple_of(merge_size) || !grid.width.is_multiple_of(merge_size) || head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(format!("Qwen3-VL vision RoPE 参数无效: grid={grid:?} merge={merge_size} head_dim={head_dim}"));
    }
    let axis_dim = head_dim / 2;
    let axis_pairs = axis_dim / 2;
    let rows = grid.temporal * grid.height * grid.width;
    let mut cos = Vec::with_capacity(rows * head_dim);
    let mut sin = Vec::with_capacity(rows * head_dim);
    for _ in 0..grid.temporal {
        for block_y in 0..grid.height / merge_size {
            for block_x in 0..grid.width / merge_size {
                for merge_y in 0..merge_size {
                    for merge_x in 0..merge_size {
                        let positions = [block_y * merge_size + merge_y, block_x * merge_size + merge_x];
                        let mut angles = Vec::with_capacity(axis_dim);
                        for position in positions {
                            for pair in 0..axis_pairs {
                                let frequency = theta.powf(-2.0 * pair as f32 / axis_dim as f32);
                                angles.push(position as f32 * frequency);
                            }
                        }
                        for _ in 0..2 {
                            cos.extend(angles.iter().map(|angle| angle.cos()));
                            sin.extend(angles.iter().map(|angle| angle.sin()));
                        }
                    }
                }
            }
        }
    }
    Ok((cos, sin))
}

pub struct Qwen3VlTextLayer<W> {
    input_norm: W,
    query: W,
    query_norm: W,
    key: W,
    key_norm: W,
    value: W,
    output: W,
    post_attention_norm: W,
    gate: W,
    up: W,
    down: W,
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_vl_prefill_resident<B, W: Qwen3TextSource>(
    backend: &B,
    config: &Qwen3VlConfig,
    weights: &W,
    resident: &[Qwen3VlTextLayer<B::Weight>],
    cache: &mut B::Cache,
    mut hidden: B::Tensor,
    rope: &Qwen3VlRopeTable,
    image_tokens: &Range<usize>,
    deepstack: &[B::Tensor],
) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    let rows = backend.token_rows(&hidden);
    if image_tokens.start > image_tokens.end || image_tokens.end > rows {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL image token range {image_tokens:?} 超出 hidden rows={rows}")));
    }
    let Some(vision) = &config.vision else {
        return Err(crate::runtime::compute_error("dense Qwen3 文本模型不支持 DeepStack 视觉特征注入".to_string()));
    };
    if deepstack.len() != vision.deepstack_visual_indexes.len() {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL DeepStack features={}，期望 {}", deepstack.len(), vision.deepstack_visual_indexes.len())));
    }
    if resident.len() > config.layer_count {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL resident layers={}，模型共 {} 层", resident.len(), config.layer_count)));
    }
    for layer in 0..config.layer_count {
        let _scope = backend.layer_scope();
        let streamed_source;
        let streamed;
        let prepared = if let Some(prepared) = resident.get(layer) {
            prepared
        } else {
            streamed_source = weights.text_layer(layer).map_err(BackendError::ExpertLoad)?;
            streamed = prepare_text_layer(backend, &streamed_source)?;
            &streamed
        };
        let result = (|| {
            let mut hidden = text_layer(backend, config, Some(cache), layer, prepared, &hidden, rope, 0)?;
            if let Some(feature) = deepstack.get(layer) {
                let rows = backend.token_rows(&hidden);
                let mut addition = backend.vision_tensor_zeros(rows, config.hidden_size)?;
                backend.scatter_rows(&mut addition, image_tokens.start, feature)?;
                hidden = backend.add(&hidden, &addition)?;
            }
            Ok(hidden)
        })();
        backend.finish_batch();
        hidden = result?;
    }
    Ok(hidden)
}

pub struct Qwen3VlTextVisual<'a, T> {
    pub tokens: Range<usize>,
    pub embedding: &'a T,
    pub deepstack: &'a [T],
}

/// H3 prompt encoder 常驻路径：优先直接使用 CPU 内存中已准备的文本层。
#[allow(clippy::too_many_arguments)]
pub fn qwen3_vl_multimodal_text_hidden_resident<B, W: Qwen3TextSource>(
    backend: &B,
    config: &Qwen3VlConfig,
    weights: &W,
    resident: &[Qwen3VlTextLayer<B::Weight>],
    mut hidden: B::Tensor,
    rope: &Qwen3VlRopeTable,
    layers: usize,
    visuals: &[Qwen3VlTextVisual<'_, B::Tensor>],
) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    let rows = backend.token_rows(&hidden);
    if layers == 0 || layers > config.layer_count {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL H3 text layers={layers}，模型共有 {} 层", config.layer_count)));
    }
    if resident.len() > layers {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL H3 resident layers={}，本次只执行 {layers} 层", resident.len())));
    }
    let Some(vision) = &config.vision else {
        return Err(crate::runtime::compute_error("dense Qwen3 文本模型不支持 H3 视觉特征注入".to_string()));
    };
    let mut previous_end = 0usize;
    for (index, visual) in visuals.iter().enumerate() {
        if visual.tokens.start < previous_end || visual.tokens.start > visual.tokens.end || visual.tokens.end > rows {
            return Err(crate::runtime::compute_error(format!("Qwen3-VL H3 visual {index} token range {:?} 非法，rows={rows}", visual.tokens)));
        }
        if visual.deepstack.len() != vision.deepstack_visual_indexes.len() {
            return Err(crate::runtime::compute_error(format!("Qwen3-VL H3 visual {index} deepstack={}，期望 {}", visual.deepstack.len(), vision.deepstack_visual_indexes.len())));
        }
        if backend.token_rows(visual.embedding) != visual.tokens.len() || backend.token_cols(visual.embedding) != config.hidden_size {
            return Err(crate::runtime::compute_error(format!("Qwen3-VL H3 visual {index} embedding shape 与 token range 不匹配")));
        }
        backend.scatter_rows(&mut hidden, visual.tokens.start, visual.embedding)?;
        previous_end = visual.tokens.end;
    }

    for layer in 0..layers {
        let _scope = backend.layer_scope();
        let streamed_source;
        let streamed;
        let prepared = if let Some(prepared) = resident.get(layer) {
            prepared
        } else {
            streamed_source = weights.text_layer(layer).map_err(BackendError::ExpertLoad)?;
            streamed = prepare_text_layer(backend, &streamed_source)?;
            &streamed
        };
        backend.begin_batch();
        let result = (|| {
            let mut hidden = text_layer(backend, config, None, layer, prepared, &hidden, rope, 0)?;
            if layer < vision.deepstack_visual_indexes.len() && !visuals.is_empty() {
                let mut addition = backend.vision_tensor_zeros(rows, config.hidden_size)?;
                for visual in visuals {
                    let feature = &visual.deepstack[layer];
                    if backend.token_rows(feature) != visual.tokens.len() || backend.token_cols(feature) != config.hidden_size {
                        return Err(crate::runtime::compute_error(format!("Qwen3-VL H3 DeepStack layer={layer} shape 与 token range 不匹配")));
                    }
                    backend.scatter_rows(&mut addition, visual.tokens.start, feature)?;
                }
                hidden = backend.add(&hidden, &addition)?;
            }
            Ok(hidden)
        })();
        backend.finish_batch();
        hidden = result?;
    }
    Ok(hidden)
}

/// 纯文本 causal prefill(无视觉特征注入):逐层执行 [`text_layer`] 并填充 KV cache。
///
/// 等价于 [`qwen3_vl_prefill_resident`] 去掉 DeepStack 注入,供不带 [`VisionBackend`]
/// 的后端(如 CUDA 文本路径)使用。`resident` 语义与 [`qwen3_vl_decode_round_resident`]
/// 一致:前 `resident.len()` 层用常驻权重,其余层从 host 流式加载(12GB 显存传 `&[]` 全流式)。
/// `position` 是序列起始位置(首段 prompt 为 0)。
#[allow(clippy::too_many_arguments)]
pub fn qwen3_vl_text_prefill<B, W: Qwen3TextSource>(
    backend: &B,
    config: &Qwen3VlConfig,
    weights: &W,
    resident: &[Qwen3VlTextLayer<B::Weight>],
    cache: &mut B::Cache,
    mut hidden: B::Tensor,
    rope: &Qwen3VlRopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    let rows = backend.token_rows(&hidden);
    if rows == 0 || backend.token_cols(&hidden) != config.hidden_size {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL text prefill hidden=[{rows},{}]，期望非空且 columns={}", backend.token_cols(&hidden), config.hidden_size)));
    }
    if resident.len() > config.layer_count {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL resident layers={}，模型共 {} 层", resident.len(), config.layer_count)));
    }
    let result = (|| {
        for layer in 0..config.layer_count {
            let _scope = backend.layer_scope();
            let streamed_source;
            let streamed;
            let prepared = if let Some(prepared) = resident.get(layer) {
                prepared
            } else {
                streamed_source = weights.text_layer(layer).map_err(BackendError::ExpertLoad)?;
                streamed = prepare_text_layer(backend, &streamed_source)?;
                &streamed
            };
            let result = text_layer(backend, config, Some(cache), layer, prepared, &hidden, rope, position);
            if layer >= resident.len() || result.is_err() {
                backend.finish_batch();
            } else {
                backend.submit_batch();
            }
            hidden = result?;
        }
        Ok(hidden)
    })();
    backend.finish_batch();
    result
}

/// 文本编码路径只执行前 `layers` 层，返回未经 final norm 的逐 token hidden。
pub fn qwen3_vl_text_hidden<B, W: Qwen3TextSource>(backend: &B, config: &Qwen3VlConfig, weights: &W, mut hidden: B::Tensor, rope: &Qwen3VlRopeTable, layers: usize) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    if layers == 0 || layers > config.layer_count {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL text encoder layers={layers}，模型共 {} 层", config.layer_count)));
    }
    if backend.token_rows(&hidden) == 0 || backend.token_cols(&hidden) != config.hidden_size {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL text hidden=[{},{}]，期望非空且 columns={}", backend.token_rows(&hidden), backend.token_cols(&hidden), config.hidden_size,)));
    }
    for layer in 0..layers {
        let _scope = backend.layer_scope();
        let source = weights.text_layer(layer).map_err(BackendError::ExpertLoad)?;
        let prepared = prepare_text_layer(backend, &source)?;
        let result = text_layer(backend, config, None, layer, &prepared, &hidden, rope, 0);
        backend.finish_batch();
        hidden = result?;
    }
    Ok(hidden)
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_vl_decode_round_resident<B, W: Qwen3TextSource>(
    backend: &B,
    config: &Qwen3VlConfig,
    weights: &W,
    resident: &[Qwen3VlTextLayer<B::Weight>],
    cache: &mut B::Cache,
    mut hidden: B::Tensor,
    rope: &Qwen3VlRopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    if backend.token_rows(&hidden) != 1 {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL decode hidden rows={}，期望 1", backend.token_rows(&hidden))));
    }
    if resident.len() > config.layer_count {
        return Err(crate::runtime::compute_error(format!("Qwen3-VL resident layers={}，模型共 {} 层", resident.len(), config.layer_count)));
    }
    let result = (|| {
        for layer in 0..config.layer_count {
            let _scope = backend.layer_scope();
            let streamed_source;
            let streamed;
            let prepared = if let Some(prepared) = resident.get(layer) {
                prepared
            } else {
                streamed_source = weights.text_layer(layer).map_err(BackendError::ExpertLoad)?;
                streamed = prepare_text_layer(backend, &streamed_source)?;
                &streamed
            };
            let result = text_layer(backend, config, Some(cache), layer, prepared, &hidden, rope, position);
            if layer >= resident.len() || result.is_err() {
                backend.finish_batch();
            } else {
                backend.submit_batch();
            }
            hidden = result?;
        }
        Ok(hidden)
    })();
    backend.finish_batch();
    result
}

#[allow(clippy::too_many_arguments)]
fn text_layer<B: GqaPrefillBackend>(
    backend: &B,
    config: &Qwen3VlConfig,
    cache: Option<&mut B::Cache>,
    layer: usize,
    weights: &Qwen3VlTextLayer<B::Weight>,
    hidden: &B::Tensor,
    rope: &Qwen3VlRopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    if backend.token_rows(hidden) == 1 {
        backend.begin_decode_batch();
    } else {
        backend.begin_batch();
    }
    let normed = backend.rmsnorm(hidden, &weights.input_norm, config.rms_eps)?;
    let (query, key) = backend.dual_linear(&normed, &weights.query, &weights.key)?;
    let value = backend.linear(&normed, &weights.value)?;
    let query = backend.gemma_rmsnorm_heads(&query, &weights.query_norm, config.num_heads, config.head_dim, config.rms_eps)?;
    let key = backend.gemma_rmsnorm_heads(&key, &weights.key_norm, config.num_kv_heads, config.head_dim, config.rms_eps)?;
    let query = backend.rope_prefix(&query, config.num_heads, config.head_dim, RotaryLayout::SplitHalf, 0, &rope.cos, &rope.sin)?;
    let key = backend.rope_prefix(&key, config.num_kv_heads, config.head_dim, RotaryLayout::SplitHalf, 0, &rope.cos, &rope.sin)?;
    let spec = GqaSpec {
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        rope_dim: config.head_dim,
        rope_theta: config.rope_theta,
        use_qk_norm: true,
        window: CausalWindow::Full,
        score_scale: 1.0 / (config.head_dim as f32).sqrt(),
        output_gate: false,
    };
    let attention = match cache {
        Some(cache) => backend.gqa_prefill_attention_cached(cache, layer, position, &query, &key, &value, &spec, false)?,
        None => backend.gqa_prefill_attention(&query, &key, &value, &spec)?,
    };
    let attention_residual = backend.linear_add(&attention, &weights.output, hidden)?;
    let normed = backend.rmsnorm(&attention_residual, &weights.post_attention_norm, config.rms_eps)?;
    backend.gated_mlp_add_residual(&normed, &weights.gate, &weights.up, &weights.down, &Activation::Silu, &attention_residual)
}

pub type Qwen3VlOutputHead<W> = super::output::OutputHead<W>;

pub fn prepare_qwen3_vl_output_head<B: Backend, W: Qwen3TextSource>(backend: &B, config: &Qwen3VlConfig, weights: &W) -> Result<Qwen3VlOutputHead<B::Weight>, BackendError> {
    prepare_qwen3_vl_output_head_quantized(backend, config, weights, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_qwen3_vl_output_head_quantized<B: Backend, W: Qwen3TextSource>(backend: &B, config: &Qwen3VlConfig, weights: &W, quantization: crate::weight::LmHeadQuantization) -> Result<Qwen3VlOutputHead<B::Weight>, BackendError> {
    let norm = weights.final_norm().map_err(BackendError::ExpertLoad)?;
    let lm_head = weights.lm_head().map_err(BackendError::ExpertLoad)?;
    match lm_head.dtype.as_str() {
        "BF16" => super::output::prepare_output_head_quantized(backend, &norm, LinearWeight::Bf16Bytes(&lm_head.data), config.vocab_size, config.hidden_size, quantization),
        "F16" => {
            let values = lm_head.data.chunks_exact(2).map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            super::output::prepare_output_head_quantized(backend, &norm, LinearWeight::F16(&values), config.vocab_size, config.hidden_size, quantization)
        }
        "F32" => {
            let values = lm_head.to_f32().map_err(crate::runtime::compute_error)?;
            super::output::prepare_output_head_quantized(backend, &norm, LinearWeight::F32(&values), config.vocab_size, config.hidden_size, quantization)
        }
        dtype => Err(crate::runtime::compute_error(format!("Qwen3-VL lm_head dtype={dtype} 不受支持"))),
    }
}

pub fn qwen3_vl_last_token_output<B: Backend>(backend: &B, config: &Qwen3VlConfig, head: &Qwen3VlOutputHead<B::Weight>, hidden: &B::Tensor) -> Result<super::output::OutputResult<B::Tensor>, BackendError> {
    super::output::last_token_output(
        backend,
        head,
        hidden,
        backend.token_rows(hidden) - 1,
        &super::output::OutputPlan { eps: config.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: vec![config.image_token_id, config.video_token_id] },
    )
}

pub fn qwen3_vl_token_output<B: Backend>(backend: &B, config: &Qwen3VlConfig, head: &Qwen3VlOutputHead<B::Weight>, hidden: &B::Tensor) -> Result<super::output::OutputResult<B::Tensor>, BackendError> {
    super::output::token_output(backend, head, hidden, &super::output::OutputPlan { eps: config.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: vec![config.image_token_id, config.video_token_id] })
}

pub fn prepare_qwen3_vl_text_layer<B: Backend, W: Qwen3TextSource>(backend: &B, weights: &W, layer: usize) -> Result<Qwen3VlTextLayer<B::Weight>, BackendError> {
    let source = weights.text_layer(layer).map_err(BackendError::ExpertLoad)?;
    prepare_text_layer(backend, &source)
}

fn prepare_text_layer<B: Backend>(backend: &B, weights: &Qwen3VlLayerWeights) -> Result<Qwen3VlTextLayer<B::Weight>, BackendError> {
    let query_norm = weights.query_norm.iter().map(|value| value - 1.0).collect::<Vec<_>>();
    let key_norm = weights.key_norm.iter().map(|value| value - 1.0).collect::<Vec<_>>();
    Ok(Qwen3VlTextLayer {
        input_norm: backend.prepare_f32(&weights.input_norm, 1, weights.input_norm.len())?,
        query: prepare_matrix(backend, &weights.query)?,
        query_norm: backend.prepare_f32(&query_norm, 1, query_norm.len())?,
        key: prepare_matrix(backend, &weights.key)?,
        key_norm: backend.prepare_f32(&key_norm, 1, key_norm.len())?,
        value: prepare_matrix(backend, &weights.value)?,
        output: prepare_matrix(backend, &weights.output)?,
        post_attention_norm: backend.prepare_f32(&weights.post_attention_norm, 1, weights.post_attention_norm.len())?,
        gate: prepare_matrix(backend, &weights.gate)?,
        up: prepare_matrix(backend, &weights.up)?,
        down: prepare_matrix(backend, &weights.down)?,
    })
}

fn prepare_matrix<B: Backend>(backend: &B, matrix: &Qwen3VlMatrix) -> Result<B::Weight, BackendError> {
    match matrix {
        Qwen3VlMatrix::Quantized(matrix) => backend.prepare_weight(LinearWeight::w4a16(matrix), matrix.rows, matrix.cols),
        Qwen3VlMatrix::Nvfp4(matrix) => backend.prepare_weight(LinearWeight::nvfp4(matrix), matrix.rows, matrix.cols),
        Qwen3VlMatrix::Dense(tensor) => prepare_dense_tensor(backend, tensor, tensor.shape[0], tensor.shape[1]),
    }
}

fn prepare_vision_layer<B: Backend>(backend: &B, source: &Qwen3VlVisionLayerWeights) -> Result<PreparedVisionLayer<B::Weight>, BackendError> {
    Ok(PreparedVisionLayer {
        input_norm: prepare_norm(backend, &source.input_norm)?,
        qkv: prepare_linear(backend, &source.qkv)?,
        output: prepare_linear(backend, &source.output)?,
        post_attention_norm: prepare_norm(backend, &source.post_attention_norm)?,
        mlp_input: prepare_linear(backend, &source.mlp_input)?,
        mlp_output: prepare_linear(backend, &source.mlp_output)?,
    })
}

fn prepare_vision_merger<B: Backend>(backend: &B, source: &Qwen3VlVisionMergerWeights) -> Result<PreparedVisionMerger<B::Weight>, BackendError> {
    Ok(PreparedVisionMerger { norm: prepare_norm(backend, &source.norm)?, input: prepare_linear(backend, &source.input)?, output: prepare_linear(backend, &source.output)?, norm_after_merge: source.norm_after_merge })
}

pub(crate) fn prepare_linear<B: Backend>(backend: &B, source: &LinearWeights) -> Result<PreparedLinear<B::Weight>, BackendError> {
    let rows = *source.weight.shape.first().ok_or_else(|| crate::runtime::compute_error("视觉 linear weight 缺少 rows"))?;
    let columns = source.weight.shape.iter().skip(1).try_fold(1usize, |value, &dimension| value.checked_mul(dimension)).ok_or_else(|| crate::runtime::compute_error("视觉 linear weight columns 溢出"))?;
    Ok(PreparedLinear { weight: prepare_dense_tensor(backend, &source.weight, rows, columns)?, bias: source.bias.as_ref().map(|bias| prepare_dense_tensor(backend, bias, 1, rows)).transpose()? })
}

pub(crate) fn prepare_norm<B: Backend>(backend: &B, source: &LayerNormWeights) -> Result<PreparedNorm<B::Weight>, BackendError> {
    let columns = source.weight.shape.iter().product::<usize>();
    Ok(PreparedNorm { weight: prepare_dense_tensor(backend, &source.weight, 1, columns)?, bias: prepare_dense_tensor(backend, &source.bias, 1, columns)? })
}

fn prepare_dense_tensor<B: Backend>(backend: &B, tensor: &TensorData, rows: usize, columns: usize) -> Result<B::Weight, BackendError> {
    let expected = rows.checked_mul(columns).ok_or_else(|| crate::runtime::compute_error("dense tensor elements 溢出"))?;
    if tensor.shape.iter().product::<usize>() != expected {
        return Err(crate::runtime::compute_error(format!("{} elements={}，期望 {expected}", tensor.name, tensor.shape.iter().product::<usize>())));
    }
    match tensor.dtype.as_str() {
        "BF16" => backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, columns),
        "F16" => {
            let values = tensor.data.chunks_exact(2).map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            backend.prepare_weight(LinearWeight::F16(&values), rows, columns)
        }
        "F32" => {
            let values = tensor.to_f32().map_err(crate::runtime::compute_error)?;
            backend.prepare_weight(LinearWeight::F32(&values), rows, columns)
        }
        dtype => Err(crate::runtime::compute_error(format!("{} dense dtype={dtype} 不受支持", tensor.name))),
    }
}

pub(crate) fn linear<B: VisionBackend>(backend: &B, input: &B::Tensor, linear: &PreparedLinear<B::Weight>) -> Result<B::Tensor, BackendError> {
    let output = backend.linear(input, &linear.weight)?;
    match &linear.bias {
        Some(bias) => backend.add_bias(&output, bias),
        None => Ok(output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_positions_compress_visual_grid() {
        let (positions, delta) = multimodal_positions(9, &(2..6), VisionGrid { temporal: 1, height: 4, width: 4 }, 2).unwrap();
        assert_eq!(positions[0], vec![0, 1, 2, 2, 2, 2, 4, 5, 6]);
        assert_eq!(positions[1], vec![0, 1, 2, 2, 3, 3, 4, 5, 6]);
        assert_eq!(positions[2], vec![0, 1, 2, 3, 2, 3, 4, 5, 6]);
        assert_eq!(delta, -2);
    }

    #[test]
    fn vision_rope_fills_full_head() {
        let (cos, sin) = vision_rope(VisionGrid { temporal: 1, height: 2, width: 2 }, 2, 72, 10_000.0).unwrap();
        assert_eq!(cos.len(), 4 * 72);
        assert_eq!(sin.len(), 4 * 72);
        assert!(cos.iter().chain(&sin).all(|value| value.is_finite()));
    }

    #[test]
    fn dense_14b_validates_without_vision() {
        let config = Qwen3VlConfig::dense_14b();
        config.validate().unwrap();
        assert!(config.vision.is_none());
        assert_eq!(config.layer_count, 40);
        assert_eq!(config.num_heads * config.head_dim, 5120);
        assert_eq!(config.num_kv_heads * config.head_dim, 1024);
    }

    /// dense Qwen3 复用 M-RoPE 表的依据:三轴等位时输出与标准 RoPE 逐位一致
    /// (frequency 只依赖全局 pair 下标,与 section 划分无关)。
    #[test]
    fn dense_mrope_table_equals_standard_rope() {
        let config = Qwen3VlConfig::dense_14b();
        let rows = 8;
        let positions: [Vec<usize>; 3] = std::array::from_fn(|_| (0..rows).collect());
        let table = qwen3_vl_mrope_table(&config, &positions).unwrap();
        let half = config.head_dim / 2;
        for row in 0..rows {
            for pair in 0..half {
                let frequency = config.rope_theta.powf(-2.0 * pair as f32 / config.head_dim as f32);
                let angle = row as f32 * frequency;
                let index = row * half + pair;
                assert!((table.cos[index] - angle.cos()).abs() < 1e-6, "cos[{row},{pair}]");
                assert!((table.sin[index] - angle.sin()).abs() < 1e-6, "sin[{row},{pair}]");
            }
        }
    }

    #[test]
    fn chat_prompt_wraps_chatml() {
        assert_eq!(chat_prompt("你好"), "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n");
    }
}

// Qwen3-VL-32B 模型规格。

use crate::{
    attention::{
        AttentionSpec,
        gqa::{GqaGeometry, GqaKvProjection, HybridGqaLayerSpec, HybridGqaSpec},
        rope::RopeSpec,
    },
    norm::NormSpec,
};

pub use crate::model_spec::qwen3_vl::{Qwen3VlConfig, Qwen3VlVisionConfig};

impl Qwen3VlConfig {
    pub fn instruct_32b() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 5_120,
            intermediate_size: 25_600,
            layer_count: 64,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
            rms_eps: 1.0e-6,
            max_position_embeddings: 262_144,
            bos_token_id: 151_643,
            eos_token_ids: vec![151_645, 151_643],
            image_token_id: 151_655,
            video_token_id: 151_656,
            vision_start_token_id: 151_652,
            vision_end_token_id: 151_653,
            vision: Some(Qwen3VlVisionConfig {
                depth: 27,
                hidden_size: 1_152,
                intermediate_size: 4_304,
                num_heads: 16,
                position_embeddings: 2_304,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                output_hidden_size: 5_120,
                rope_theta: 10_000.0,
                deepstack_visual_indexes: vec![8, 16, 24],
                min_pixels: 65_536,
                max_pixels: 16_777_216,
                max_aspect_ratio: 200.0,
                image_mean: [0.5; 3],
                image_std: [0.5; 3],
            }),
        }
    }

    /// 经典 dense Qwen3-14B(`Qwen3ForCausalLM`,nvidia/Qwen3-14B-NVFP4 超参)。
    ///
    /// 纯文本模型:无视觉塔,M-RoPE 三轴等位退化为标准 RoPE(mrope_section 取和为
    /// head_dim/2 的任意划分,等位下不影响数值)。tie_word_embeddings=false,lm_head
    /// 独立于 embed_tokens。
    pub fn dense_14b() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 5_120,
            intermediate_size: 17_408,
            layer_count: 40,
            num_heads: 40,
            num_kv_heads: 8,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            mrope_section: [32, 32, 0],
            rms_eps: 1.0e-6,
            max_position_embeddings: 40_960,
            bos_token_id: 151_643,
            eos_token_ids: vec![151_645, 151_643],
            image_token_id: 151_655,
            video_token_id: 151_656,
            vision_start_token_id: 151_652,
            vision_end_token_id: 151_653,
            vision: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.layer_count == 0 || self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err("Qwen3-VL 文本层维度必须大于 0".to_owned());
        }
        if self.num_heads == 0 || self.num_kv_heads == 0 || !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err("Qwen3-VL attention heads 规格无效".to_owned());
        }
        if self.mrope_section.iter().sum::<usize>() != self.head_dim / 2 {
            return Err(format!("Qwen3-VL M-RoPE section {:?} 与 head_dim {} 不一致", self.mrope_section, self.head_dim,));
        }
        if let Some(vision) = &self.vision {
            if vision.depth == 0
                || vision.num_heads == 0
                || !vision.hidden_size.is_multiple_of(vision.num_heads)
                || vision.output_hidden_size != self.hidden_size
                || vision.position_embeddings.isqrt().pow(2) != vision.position_embeddings
                || vision.deepstack_visual_indexes.iter().any(|&layer| layer >= vision.depth)
            {
                return Err("Qwen3-VL vision 规格无效".to_owned());
            }
        }
        Ok(())
    }
}

pub struct Qwen3Vl {
    config: Qwen3VlConfig,
    layer_specs: Vec<LayerSpec>,
    hybrid_gqa: HybridGqaSpec,
}

impl Qwen3Vl {
    pub fn new(config: Qwen3VlConfig) -> Result<Self, ModelError> {
        config.validate().map_err(ModelError::InvalidArchitecture)?;
        let gqa = GqaSpec {
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            rope_dim: config.head_dim,
            rope_theta: config.rope_theta,
            use_qk_norm: true,
            window: CausalWindow::Full,
            score_scale: 1.0 / (config.head_dim as f32).sqrt(),
            output_gate: false,
        };
        let layer_specs = (0..config.layer_count)
            .map(|_| LayerSpec {
                attention: AttentionSpec::Gqa(gqa),
                feedforward: FeedforwardSpec::Dense(DenseMlpSpec { intermediate_size: config.intermediate_size, activation: Activation::Silu }),
                input_norm: NormSpec::Rms { eps: config.rms_eps },
                post_attention_norm: NormSpec::Rms { eps: config.rms_eps },
                post_norm: None,
            })
            .collect();
        let hybrid = HybridGqaLayerSpec {
            geometry: GqaGeometry { num_heads: config.num_heads, num_kv_heads: config.num_kv_heads, head_dim: config.head_dim },
            rope: RopeSpec::Default { rotary_dim: config.head_dim, theta: config.rope_theta },
            window: CausalWindow::Full,
            score_scale: gqa.score_scale,
            kv_projection: GqaKvProjection::Separate,
        };
        let hybrid_gqa = HybridGqaSpec::new(vec![hybrid; config.layer_count]).map_err(ModelError::InvalidArchitecture)?;
        Ok(Self { config, layer_specs, hybrid_gqa })
    }

    pub fn instruct_32b() -> Self {
        Self::new(Qwen3VlConfig::instruct_32b()).expect("内置 Qwen3-VL-32B 配置必须有效")
    }

    pub fn hybrid_gqa(&self) -> &HybridGqaSpec {
        &self.hybrid_gqa
    }
}

impl Model for Qwen3Vl {
    type Config = Qwen3VlConfig;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn layer_count(&self) -> usize {
        self.config.layer_count
    }

    fn layer_spec(&self, layer: LayerId) -> Result<&LayerSpec, ModelError> {
        self.layer_specs.get(layer).ok_or(ModelError::LayerOutOfRange { layer, layer_count: self.config.layer_count })
    }
}
