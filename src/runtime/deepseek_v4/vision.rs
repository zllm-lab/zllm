//! DeepSeek-V4.1-Flash 视觉塔编排与图文输入。
//!
//! 语义对照官方 inference/vision.py 与 image_processor.py:
//! - 预处理:像素尺寸先按 token 预算等比收缩(保持 patch 对齐),再 contain
//!   缩放 + 灰(127)补边;patch 按行主序展开,(x-0.5)/0.5 归一化。
//! - ViT:patch 线性嵌入 → 32 层块(pre-RMSNorm、fused wqkv 带 bias、
//!   h/w 交错 2D RoPE、非因果整图 attention、SwiGLU 无 bias MLP)→ post RMSNorm。
//! - aligner:ViT 输出右侧/下侧零填充到 downsample 倍数后 3×3 空间合并,
//!   双线性投影(BF16 带 bias)+ GELU + 输出投影,行宽等于 LLM hidden_size。
//! - token 合成:`[IMAGE_START] + ([IMAGE]*w + [IMAGE_NEWLINE])*h + [IMAGE_END]`,
//!   span 内所有位置共用 image_token_id,仅靠类型区分;embedding 阶段把
//!   IMAGE 槽位替换为 aligner 行、首尾/换行槽位替换为学习向量。

use crate::{
    backend::{BackendError, GqaPrefillBackend, LinearWeight, VisionBackend},
    model_spec::deepseek_v4::DeepseekV41VisionConfig,
    moe::Activation,
    vision::{ImageProcessor, ImageTensor, ImageTokenRange, RgbImage, VisionGrid, pillow_bicubic_resize},
    weight::{
        container::safetensor::TensorData,
        model::deepseek_v4::{DeepSeekV4Weights, DeepseekV41AlignerWeights, DeepseekV41ImageSpanEmbeddings, DeepseekV41VisionBlockWeights, DeepseekV41VisionPatchEmbedWeights},
    },
};

/// 一次多模态请求的展开输入(node 侧组装图像,engine 侧做视觉编码)。
#[derive(Debug)]
pub struct DeepseekV41VisionInput {
    /// 占位符已展开为完整 span 的 prompt tokens。
    pub tokens: Vec<u32>,
    pub images: Vec<ImageTensor>,
    pub spans: Vec<DeepseekV41ImageSpan>,
}

/// engine 视觉编码后的 span 行覆盖:每张图首尾/换行/aligner 混排的
/// hidden 行与其 token 区间,供 chunked prefill 的 embedding 上传拼接。
#[derive(Debug)]
pub struct DeepseekV41SpanOverlays {
    pub ranges: Vec<ImageTokenRange>,
    pub rows: Vec<Vec<f32>>,
}

/// 官方 padding 颜色(ImageOps.pad color=(127,127,127))。
const PAD_GRAY: [u8; 3] = [127, 127, 127];
/// 官方 vision RMSNorm eps(RMSNorm(dim, eps=1e-6))。
const VISION_RMS_EPS: f32 = 1.0e-6;

/// PIL `ImageOps.contain + pad(centering=0.5, color=127)` 的精确语义:
/// 按宽高比分支 round 出 contain 尺寸(不是 ceil),居中偏移 int(差×0.5)。
/// round 半值取整与 Python banker's rounding 在 .5 处可能差 1px,极罕见。
fn pad_contain_pil(source: &image::RgbImage, width: usize, height: usize) -> image::RgbImage {
    let image_ratio = source.width() as f64 / source.height() as f64;
    let destination_ratio = width as f64 / height as f64;
    let (resized_width, resized_height) = if image_ratio == destination_ratio {
        (width, height)
    } else if image_ratio > destination_ratio {
        (width, ((source.height() as f64 * width as f64 / source.width() as f64).round() as usize).clamp(1, height))
    } else {
        (((source.width() as f64 * height as f64 / source.height() as f64).round() as usize).clamp(1, width), height)
    };
    let resized = pillow_bicubic_resize(source, resized_width as u32, resized_height as u32);
    let offset_x = ((width - resized_width) as f64 * 0.5) as u32;
    let offset_y = ((height - resized_height) as f64 * 0.5) as u32;
    let mut output = image::RgbImage::from_pixel(width as u32, height as u32, image::Rgb(PAD_GRAY));
    image::imageops::replace(&mut output, &resized, i64::from(offset_x), i64::from(offset_y));
    output
}

/// span 布局:LLM 侧 token 网格与展开后的区间。
#[derive(Clone, Debug)]
pub struct DeepseekV41ImageSpan {
    /// 展开 span 前一个 token 的位置(官方 ImageInput.start 语义)。
    pub start: usize,
    pub llm_rows: usize,
    pub llm_cols: usize,
}

impl DeepseekV41ImageSpan {
    /// span 总 token 数(含首尾与每行换行)。
    pub fn token_count(&self) -> usize {
        self.llm_rows * (self.llm_cols + 1) + 2
    }
}

fn llm_grid(vit_rows: usize, vit_cols: usize, patch: usize, downsample: usize) -> (usize, usize) {
    (vit_rows.div_ceil(patch) / downsample + usize::from(vit_rows.div_ceil(patch) % downsample > 0), vit_cols.div_ceil(patch) / downsample + usize::from(vit_cols.div_ceil(patch) % downsample > 0))
}

fn num_image_tokens(rows: usize, cols: usize) -> usize {
    rows * (cols + 1) + 2
}

/// 官方 safe_resize 像素规划:先 ceil 到 patch 倍数,token 超预算时按
/// solve_resize_ratio 等比收缩(地板到 cell=patch*downsample 倍数)。
/// 返回 (n_llm_h, n_llm_w, best_height, best_width)。
fn plan_image_grid(height: usize, width: usize, vision: &DeepseekV41VisionConfig) -> (usize, usize, usize, usize) {
    let patch = vision.patch_size;
    let downsample = vision.downsample_ratio;
    let mut best = (height.div_ceil(patch) * patch, width.div_ceil(patch) * patch);
    let mut grid = llm_grid(best.0, best.1, patch, downsample);
    if num_image_tokens(grid.0, grid.1) > vision.max_image_tokens {
        let cell = patch * downsample;
        let ratio = height as f64 / width as f64;
        let max_w_float = ((vision.max_image_tokens as f64 - 2.0) / ratio + 0.25).sqrt() - 0.5;
        let max_h_float = max_w_float * ratio;
        best = if max_w_float < 1.0 {
            ((vision.max_image_tokens - 2) / 2 * cell, cell)
        } else if max_h_float < 1.0 {
            (cell, (vision.max_image_tokens - 3) * cell)
        } else {
            let beta = ((max_w_float.floor() * cell as f64) / width as f64).min((max_h_float.floor() * cell as f64) / height as f64);
            (((height as f64 * beta / patch as f64).floor() as usize) * patch, ((width as f64 * beta / patch as f64).floor() as usize) * patch)
        };
        grid = llm_grid(best.0, best.1, patch, downsample);
    }
    (grid.0, grid.1, best.0, best.1)
}

/// DeepSeek-V4.1 图像预处理器:contain 缩放 + 灰补边 + 行主序 patch 展开。
pub struct DeepseekV41ImageProcessor {
    config: DeepseekV41VisionConfig,
}

impl DeepseekV41ImageProcessor {
    pub fn new(config: DeepseekV41VisionConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self { config })
    }

    /// 规划后的 LLM token 网格(官方 llm_grid,patch 对齐像素之外的独立输出)。
    pub fn plan_grid(&self, image: &RgbImage) -> (usize, usize) {
        let (rows, cols, _, _) = self.plan(image);
        (rows, cols)
    }

    fn plan(&self, image: &RgbImage) -> (usize, usize, usize, usize) {
        let mut width = image.width;
        let mut height = image.height;
        // min_pixels 等比放大(官方对 w/h 各自 int 截断,基于原始尺寸)。
        if width * height < self.config.min_pixels {
            let ratio = (self.config.min_pixels as f64 / (width as f64 * height as f64)).sqrt();
            width = (width as f64 * ratio) as usize;
            height = (height as f64 * ratio) as usize;
        }
        plan_image_grid(height, width, &self.config)
    }
}

impl ImageProcessor for DeepseekV41ImageProcessor {
    fn preprocess(&self, image: &RgbImage) -> Result<ImageTensor, String> {
        let patch = self.config.patch_size;
        let (_, _, best_height, best_width) = self.plan(image);
        let source = image::RgbImage::from_raw(image.width as u32, image.height as u32, image.pixels.clone()).ok_or("构造 DeepSeek-V4.1 RGB 图像失败")?;
        let resized = pad_contain_pil(&source, best_width, best_height);
        let grid_height = best_height / patch;
        let grid_width = best_width / patch;
        let rows = grid_height.checked_mul(grid_width).ok_or("DeepSeek-V4.1 视觉 patch 行数溢出")?;
        let cols = 3 * patch * patch;
        let mut data = Vec::with_capacity(rows * cols);
        // 行主序 (gy, gx) + 通道平面 (c, y, x),与官方 permute(1,3,0,2,4) 一致。
        for gy in 0..grid_height {
            for gx in 0..grid_width {
                for channel in 0..3 {
                    for y in 0..patch {
                        for x in 0..patch {
                            let pixel = resized.get_pixel((gx * patch + x) as u32, (gy * patch + y) as u32)[channel];
                            data.push(pixel as f32 / 255.0 * 2.0 - 1.0);
                        }
                    }
                }
            }
        }
        Ok(ImageTensor { data, rows, cols, grid: VisionGrid { temporal: 1, height: grid_height, width: grid_width }, merge_size: self.config.downsample_ratio })
    }
}

/// 把 prompt token 中的 image 占位符逐张展开为完整 span。
/// 官方要求占位符数量与图像一致;这里返回展开后的 tokens 与每张图的布局。
pub fn expand_image_spans(tokens: &[u32], image_token_id: u32, images: &[ImageTensor], downsample: usize) -> Result<(Vec<u32>, Vec<DeepseekV41ImageSpan>), String> {
    let mut expanded = Vec::with_capacity(tokens.len() + images.len() * 32);
    let mut spans = Vec::with_capacity(images.len());
    let mut next_image = 0usize;
    for &token in tokens {
        if token != image_token_id {
            expanded.push(token);
            continue;
        }
        let image = images.get(next_image).ok_or_else(|| format!("image 占位符多于图像({})", images.len()))?;
        let llm_rows = image.grid.height.div_ceil(downsample);
        let llm_cols = image.grid.width.div_ceil(downsample);
        spans.push(DeepseekV41ImageSpan { start: expanded.len(), llm_rows, llm_cols });
        expanded.extend(std::iter::repeat_n(image_token_id, spans.last().expect("刚压入 span").token_count()));
        next_image += 1;
    }
    if next_image != images.len() {
        return Err(format!("图像 {} 张多于 image 占位符 {next_image} 个", images.len()));
    }
    Ok((expanded, spans))
}

/// 组装一张图的 span 行(BF16 权重转 F32):首/换行/尾学习向量 + aligner 行。
pub fn span_rows_f32(span: &DeepseekV41ImageSpan, aligner_rows: &[f32], hidden_size: usize, embeddings: &DeepseekV41ImageSpanEmbeddings) -> Result<Vec<f32>, String> {
    let expected = span.llm_rows * span.llm_cols;
    if aligner_rows.len() != expected * hidden_size {
        return Err(format!("aligner 行 values={}，期望 {}x{}", aligner_rows.len(), expected, hidden_size));
    }
    let vector = |tensor: &TensorData| -> Result<Vec<f32>, String> {
        let elements = tensor.shape.iter().product::<usize>();
        if elements != hidden_size || tensor.data.len() != elements * 2 {
            return Err(format!("image span 向量 {} shape={:?} 与 hidden_size={hidden_size} 不一致", tensor.name, tensor.shape));
        }
        Ok(tensor.data.chunks_exact(2).map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect())
    };
    let start = vector(&embeddings.start)?;
    let newline = vector(&embeddings.newline)?;
    let end = vector(&embeddings.end)?;
    let mut rows = Vec::with_capacity(span.token_count() * hidden_size);
    rows.extend_from_slice(&start);
    for row in 0..span.llm_rows {
        for column in 0..span.llm_cols {
            let offset = (row * span.llm_cols + column) * hidden_size;
            rows.extend_from_slice(&aligner_rows[offset..offset + hidden_size]);
        }
        rows.extend_from_slice(&newline);
    }
    rows.extend_from_slice(&end);
    Ok(rows)
}

/// h/w 分块 2D RoPE 表:每位置前半 [h*f0..h*f(n-1)]、后半 [w*f0..w*f(n-1)]
/// (官方 stack([h,w],-1).reshape(-1,2,1)*inv_freq 后 flatten 的块状布局),
/// 按官方 chunk-2 旋转约定重复两半构成 head_dim 宽的 cos/sin。
fn vision_rope_2d(grid: VisionGrid, head_dim: usize, theta: f32) -> Result<(Vec<f32>, Vec<f32>), String> {
    if grid.temporal != 1 || grid.height == 0 || grid.width == 0 || head_dim % 4 != 0 {
        return Err(format!("DeepSeek-V4.1 视觉 RoPE 参数非法: grid={grid:?} head_dim={head_dim}"));
    }
    let rope_dim = head_dim / 2;
    let frequencies = (0..rope_dim / 2).map(|index| theta.powf(-2.0 * index as f32 / rope_dim as f32)).collect::<Vec<_>>();
    let rows = grid.height * grid.width;
    let mut cos = Vec::with_capacity(rows * head_dim);
    let mut sin = Vec::with_capacity(rows * head_dim);
    for position_y in 0..grid.height {
        for position_x in 0..grid.width {
            for _ in 0..2 {
                cos.extend(frequencies.iter().map(|&frequency| (position_y as f32 * frequency).cos()));
                sin.extend(frequencies.iter().map(|&frequency| (position_y as f32 * frequency).sin()));
                cos.extend(frequencies.iter().map(|&frequency| (position_x as f32 * frequency).cos()));
                sin.extend(frequencies.iter().map(|&frequency| (position_x as f32 * frequency).sin()));
            }
        }
    }
    Ok((cos, sin))
}

/// 完整视觉塔:patch embed → ViT 层 → post norm → aligner。
/// 输出 [ceil 网格行 × ceil 网格列, hidden_size],行序与 IMAGE 槽位一致。
pub fn deepseek_v41_vision_encode<B>(backend: &B, vision: &DeepseekV41VisionConfig, weights: &DeepSeekV4Weights, image: &ImageTensor) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    let expected_cols = 3 * vision.patch_size * vision.patch_size;
    if image.rows == 0 || image.cols != expected_cols || image.data.len() != image.rows * image.cols {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4.1 视觉 patch tensor=[{},{}] values={}，期望 cols={expected_cols}", image.rows, image.cols, image.data.len())));
    }
    let debug_dir = std::env::var_os("ZLLM_VISION_DEBUG_DIR").map(|p| p.to_string_lossy().to_string());
    eprintln!("[vision-debug] ZLLM_VISION_DEBUG_DIR={:?}", debug_dir);
    let debug_save = |name: &str, tensor: &B::Tensor, rows: usize, cols: usize| {
        let Some(dir) = debug_dir.as_deref() else {
            eprintln!("[vision-debug] hook {name} skipped (no debug_dir)");
            return;
        };
        eprintln!("[vision-debug] hook {name} dir={dir} rows={rows} cols={cols}");
        match backend.vision_download(tensor) {
            Ok(data) => debug_dump_tensor(dir, name, &data, rows, cols),
            Err(error) => eprintln!("[vision-debug] {name} download failed: {error:?}"),
        }
    };
    let hidden = {
        let _scope = backend.layer_scope();
        let source = weights.load_vision_patch_embed(vision).map_err(BackendError::ExpertLoad)?;
        // patch 列 3p²(=588)不是 16 的倍数;权重与输入同步零填充到 16 倍数,
        // 零列不改变线性输出(数学等价)。
        let patch = {
            let [output_rows, columns] = [source.projection.shape[0], source.projection.shape[1]];
            let padded_columns = columns.div_ceil(16) * 16;
            let projection = if padded_columns == columns {
                source.projection
            } else {
                let mut data = Vec::with_capacity(output_rows * padded_columns * 2);
                for row in 0..output_rows {
                    data.extend_from_slice(&source.projection.data[row * columns * 2..(row + 1) * columns * 2]);
                    data.extend(std::iter::repeat_n(0u8, (padded_columns - columns) * 2));
                }
                TensorData { name: source.projection.name, dtype: "BF16".to_owned(), shape: vec![output_rows, padded_columns], data }
            };
            let mut values = Vec::with_capacity(image.rows * padded_columns);
            for row in 0..image.rows {
                values.extend_from_slice(&image.data[row * image.cols..(row + 1) * image.cols]);
                values.extend(std::iter::repeat_n(0.0, padded_columns - columns));
            }
            let input = backend.vision_tensor_from_f32(&values, image.rows, padded_columns)?;
            DeepseekV41PreparedPatchEmbed { projection: prepare_matrix(backend, &projection)?, bias: prepare_vector(backend, &source.bias)?, input }
        };
        backend.begin_batch();
        let result = (|| {
            let embedded = backend.linear(&patch.input, &patch.projection)?;
            backend.add_bias(&embedded, &patch.bias)
        })();
        backend.finish_batch();
        result?
    };
    debug_save("patch_embed_out", &hidden, image.rows, vision.hidden_size);

    let head_dim = vision.hidden_size / vision.num_heads;
    let (cos, sin) = vision_rope_2d(image.grid, head_dim, vision.rope_theta).map_err(crate::runtime::compute_error)?;
    let cos = backend.vision_tensor_from_f32(&cos, image.rows, head_dim)?;
    let sin = backend.vision_tensor_from_f32(&sin, image.rows, head_dim)?;

    let mut hidden = hidden;
    let debug_all_blocks = std::env::var_os("ZLLM_VISION_DEBUG_BLOCKS").is_some();
    for layer in 0..vision.layer_count {
        let _scope = backend.layer_scope();
        let source = weights.load_vision_block(layer, vision).map_err(BackendError::ExpertLoad)?;
        let prepared = prepare_vision_block(backend, &source)?;
        backend.begin_batch();
        let result = vision_block(backend, vision, &prepared, &hidden, &cos, &sin);
        backend.finish_batch();
        hidden = result?;
        if debug_all_blocks || layer == 0 || layer == vision.layer_count / 2 || layer + 1 == vision.layer_count {
            debug_save(&format!("block_{layer}_out"), &hidden, image.rows, vision.hidden_size);
        }
    }

    let normed = {
        let _scope = backend.layer_scope();
        let norm = weights.load_vision_final_norm(vision).map_err(BackendError::ExpertLoad)?;
        let norm = prepare_vector(backend, &norm)?;
        backend.begin_batch();
        backend.rmsnorm(&hidden, &norm, VISION_RMS_EPS)?
    };
    debug_save("final_norm_out", &normed, image.rows, vision.hidden_size);

    let merged = {
        let _scope = backend.layer_scope();
        let source = weights.load_aligner(vision).map_err(BackendError::ExpertLoad)?;
        let llm_dim = source.output.shape[0];
        let aligner = DeepseekV41PreparedAligner {
            input: prepare_matrix(backend, &source.input)?,
            input_bias: prepare_vector(backend, &source.input_bias)?,
            output: prepare_matrix(backend, &source.output)?,
            output_bias: prepare_vector(backend, &source.output_bias)?,
        };
        let merge = vision.downsample_ratio;
        // 官方 aligner 对 ViT 输出右侧/下侧零填充后做 merge×merge unfold。
        // w 方向补零使有效行在填充网格中不再连续,scatter 无法表达;
        // 下载到 host 按网格拼装 [ceil 网格行×列, merge²*dim] 再上传(≤15MB/图)。
        let vit_rows = backend.vision_download(&normed)?;
        let merged_rows = image.grid.height.div_ceil(merge) * image.grid.width.div_ceil(merge);
        let window = merge * merge;
        // 官方 unfold 行内布局是通道主序:每 merged 行 = [c0 的 r² 窗口, c1 的 r² 窗口, ...]。
        let mut merged = vec![0.0_f32; merged_rows * window * vision.hidden_size];
        for source_y in 0..image.grid.height {
            for source_x in 0..image.grid.width {
                let merged_row = (source_y / merge) * image.grid.width.div_ceil(merge) + source_x / merge;
                let window_slot = (source_y % merge) * merge + source_x % merge;
                let input_offset = (source_y * image.grid.width + source_x) * vision.hidden_size;
                for channel in 0..vision.hidden_size {
                    merged[(merged_row * vision.hidden_size + channel) * window + window_slot] = vit_rows[input_offset + channel];
                }
            }
        }
        debug_save("aligner_unfolded_in", &normed, image.rows, vision.hidden_size);
        let merged = backend.vision_tensor_from_f32(&merged, merged_rows, window * vision.hidden_size)?;
        debug_save("aligner_unfolded_out", &merged, merged_rows, window * vision.hidden_size);
        backend.begin_batch();
        let result = (|| {
            let projected = backend.linear(&merged, &aligner.input)?;
            let projected = backend.add_bias(&projected, &aligner.input_bias)?;
            debug_save("aligner_w1_out", &projected, merged_rows, llm_dim);
            let activated = backend.gelu(&projected)?;
            debug_save("aligner_gelu_out", &activated, merged_rows, llm_dim);
            let output = backend.linear(&activated, &aligner.output)?;
            backend.add_bias(&output, &aligner.output_bias)
        })();
        backend.finish_batch();
        result?
    };
    let aligner_rows = image.grid.height.div_ceil(vision.downsample_ratio) * image.grid.width.div_ceil(vision.downsample_ratio);
    let llm_dim = weights.load_aligner(vision).map_err(BackendError::ExpertLoad)?.output.shape[0];
    debug_save("aligner_out", &merged, aligner_rows, llm_dim);

    let expected_rows = image.grid.height.div_ceil(vision.downsample_ratio) * image.grid.width.div_ceil(vision.downsample_ratio);
    if backend.token_rows(&merged) != expected_rows {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4.1 视觉输出 rows={}，期望 {expected_rows}", backend.token_rows(&merged))));
    }
    Ok(merged)
}

struct DeepseekV41PreparedPatchEmbed<W, T> {
    projection: W,
    bias: W,
    input: T,
}

struct DeepseekV41PreparedAttention<W> {
    query_key_value: W,
    query_key_value_bias: W,
    output: W,
    output_bias: W,
}

struct DeepseekV41PreparedMlp<W> {
    gate: W,
    up: W,
    down: W,
}

struct DeepseekV41PreparedBlock<W> {
    input_norm: W,
    post_attention_norm: W,
    attention: DeepseekV41PreparedAttention<W>,
    mlp: DeepseekV41PreparedMlp<W>,
}

struct DeepseekV41PreparedAligner<W> {
    input: W,
    input_bias: W,
    output: W,
    output_bias: W,
}

/// 把中间量落盘到 `ZLLM_VISION_DEBUG_DIR`:每 tensor 一对文件,
/// `<name>.f32`(原始 little-endian f32 bytes)+ `<name>.shape`(text "rows cols")。
/// 任何 IO 错误都吞掉——debug hook 永远不阻塞主流程。
fn debug_dump_tensor(dir: &str, name: &str, data: &[f32], rows: usize, cols: usize) {
    use std::io::Write;
    let raw_path = format!("{dir}/{name}.f32");
    let shape_path = format!("{dir}/{name}.shape");
    match std::fs::create_dir_all(dir) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("[vision-debug] create_dir_all({dir}) failed: {e}");
            return;
        }
    }
    match std::fs::File::create(&raw_path) {
        Ok(mut file) => {
            let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
            match file.write_all(bytes) {
                Ok(_) => eprintln!("[vision-debug] wrote {} bytes to {raw_path}", bytes.len()),
                Err(e) => eprintln!("[vision-debug] write_all({raw_path}) failed: {e}"),
            }
        }
        Err(e) => eprintln!("[vision-debug] File::create({raw_path}) failed: {e}"),
    }
    let _ = std::fs::write(&shape_path, format!("{rows} {cols}\n"));
}

/// 一个 ViT 块:RMSNorm → attention → 残差;RMSNorm → SwiGLU MLP → 残差。
fn vision_block<B>(backend: &B, vision: &DeepseekV41VisionConfig, layer: &DeepseekV41PreparedBlock<B::Weight>, hidden: &B::Tensor, cos: &B::Tensor, sin: &B::Tensor) -> Result<B::Tensor, BackendError>
where
    B: VisionBackend + GqaPrefillBackend,
{
    let normed = backend.rmsnorm(hidden, &layer.input_norm, VISION_RMS_EPS)?;
    let attention = {
        let qkv = backend.linear(&normed, &layer.attention.query_key_value)?;
        let qkv = backend.add_bias(&qkv, &layer.attention.query_key_value_bias)?;
        let (query, key_value) = backend.split_columns(&qkv, vision.hidden_size)?;
        let (key, value) = backend.split_columns(&key_value, vision.hidden_size)?;
        let attended = backend.vision_attention(&query, &key, &value, cos, sin, vision.num_heads)?;
        let projected = backend.linear(&attended, &layer.attention.output)?;
        backend.add_bias(&projected, &layer.attention.output_bias)?
    };
    let residual = backend.add(hidden, &attention)?;
    let normed = backend.rmsnorm(&residual, &layer.post_attention_norm, VISION_RMS_EPS)?;
    let (gate, up) = backend.dual_linear(&normed, &layer.mlp.gate, &layer.mlp.up)?;
    let activated = backend.gated_activation(&gate, &up, &Activation::Silu)?;
    let mlp = backend.linear(&activated, &layer.mlp.down)?;
    backend.add(&residual, &mlp)
}

fn prepare_vision_block<B: VisionBackend>(backend: &B, source: &DeepseekV41VisionBlockWeights) -> Result<DeepseekV41PreparedBlock<B::Weight>, BackendError> {
    // 官方 w1 是 [2*inter, dim] 融合矩阵,前半 gate 后半 up;行切片成两份权重。
    let gate_up_rows = source.mlp_gate_up.shape.first().copied().unwrap_or(0);
    let gate_up = split_rows(&source.mlp_gate_up, gate_up_rows / 2).map_err(crate::runtime::compute_error)?;
    Ok(DeepseekV41PreparedBlock {
        input_norm: prepare_vector(backend, &source.input_norm)?,
        post_attention_norm: prepare_vector(backend, &source.post_attention_norm)?,
        attention: DeepseekV41PreparedAttention {
            query_key_value: prepare_matrix(backend, &source.query_key_value)?,
            query_key_value_bias: prepare_vector(backend, &source.query_key_value_bias)?,
            output: prepare_matrix(backend, &source.output)?,
            output_bias: prepare_vector(backend, &source.output_bias)?,
        },
        mlp: DeepseekV41PreparedMlp { gate: prepare_matrix(backend, &gate_up.0)?, up: prepare_matrix(backend, &gate_up.1)?, down: prepare_matrix(backend, &source.mlp_down)? },
    })
}

/// 按行把 BF16 rank-2 TensorData 切成两份(gate/up 融合矩阵拆分)。
fn split_rows(source: &TensorData, rows: usize) -> Result<(TensorData, TensorData), String> {
    let [total, columns] = [source.shape.first().copied().unwrap_or(0), source.shape.get(1).copied().unwrap_or(0)];
    if rows == 0 || rows >= total {
        return Err(format!("DeepSeek-V4.1 视觉融合矩阵 {} 行切分 {rows}/{total} 非法", source.name));
    }
    let bytes = columns * 2;
    let make = |rows: usize, offset: usize| TensorData { name: format!("{}[{offset}]", source.name), dtype: "BF16".to_owned(), shape: vec![rows, columns], data: source.data[offset * bytes..(offset + rows) * bytes].to_vec() };
    Ok((make(rows, 0), make(total - rows, rows)))
}

/// BF16 rank-2 线性权重;shape 由 weight 层校验。
fn prepare_matrix<B: VisionBackend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let [rows, cols] = tensor.shape.as_slice() else {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4.1 视觉线性权重 {} shape={:?} 不是 rank-2", tensor.name, tensor.shape)));
    };
    backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), *rows, *cols)
}

/// BF16 一维向量(bias / norm 权重)。
fn prepare_vector<B: VisionBackend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let elements = tensor.shape.iter().product::<usize>();
    backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), 1, elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 真实官方 checkpoint 的视觉塔单卡对拍:不起服务,直接在 GPU 0 上
    /// 跑完整 encode,中间量经 debug hook 落盘(/tmp/zllm_*.f32)。
    /// 运行:`cargo test --release --features with-rocm 视觉塔真权重 -- --ignored --nocapture`
    /// 依赖 DeepSeek-V4.1-Flash 官方 safetensors。
    #[cfg(all(target_os = "linux", feature = "with-rocm"))]
    #[test]
    #[ignore]
    fn 视觉塔真权重对拍() {
        let root = "/path/to/DeepSeek-V4.1-Flash";
        let config = DeepseekV41VisionConfig::standard();
        let weights = crate::weight::model::deepseek_v4::DeepSeekV4Weights::open(root, crate::model_spec::deepseek_v4::DeepSeekV4Config::flash_v41()).unwrap_or_else(|error| panic!("open: {error}"));
        assert!(weights.has_vision_tower(), "checkpoint 缺视觉塔");
        let image = RgbImage::open(format!("{root}/inference/examples/images/carrots.jpeg")).unwrap();
        let tensor = DeepseekV41ImageProcessor::new(config.clone()).unwrap().preprocess(&image).unwrap();
        eprintln!("[vision-real] patches [{},{}] grid={:?}", tensor.rows, tensor.cols, tensor.grid);
        let context = crate::backend::rocm::RocmContext::new(0).unwrap();
        context.activate().unwrap();
        let encoded = deepseek_v41_vision_encode(&context, &config, &weights, &tensor).unwrap_or_else(|error| panic!("encode: {error:?}"));
        let rows = context.tensor_to_f32(&encoded).unwrap();
        let expected = tensor.grid.height.div_ceil(3) * tensor.grid.width.div_ceil(3);
        assert_eq!(rows.len(), expected * 5120);
        let mean = rows.iter().sum::<f32>() / rows.len() as f32;
        let std = (rows.iter().map(|v| v * v).sum::<f32>() / rows.len() as f32).sqrt();
        eprintln!("[vision-real] aligner rows={} mean={mean:.6} std={std:.6}", rows.len() / 5120);
        assert!(rows.iter().all(|v| v.is_finite()));
        assert!(std > 0.01 && std < 0.2, "aligner 输出统计异常 std={std}");
    }

    fn config(patch: usize, downsample: usize, max_tokens: usize, min_pixels: usize) -> DeepseekV41VisionConfig {
        DeepseekV41VisionConfig {
            layer_count: 2,
            hidden_size: 64,
            num_heads: 4,
            intermediate_size: 32,
            patch_size: patch,
            rope_theta: 10_000.0,
            downsample_ratio: downsample,
            max_image_tokens: max_tokens,
            min_pixels,
            image_token_id: 129_264,
        }
    }

    #[test]
    fn 标准视觉配置自洽() {
        let config = DeepseekV41VisionConfig::standard();
        config.validate().unwrap();
        assert_eq!(config.hidden_size / config.num_heads, 64);
        assert_eq!(config.rope_theta, 10_000.0);
    }

    #[test]
    fn 最小像素下限会放大画布() {
        // 28×28=784 < min_pixels 3136(=28²×4):放大后画布到 56×56,网格 4×4。
        let vision = config(14, 3, 1024, 3_136);
        let processor = DeepseekV41ImageProcessor::new(vision).unwrap();
        let image = RgbImage::new(28, 28, vec![200; 28 * 28 * 3]).unwrap();
        let (rows, cols) = processor.plan_grid(&image);
        assert_eq!((rows, cols), (2, 2));
    }

    #[test]
    fn token_预算收缩保持patch对齐() {
        // patch 14 / downsample 3 / 上限 184 token:182px → ViT 13×13 → LLM 5×5=32 token。
        // 更大的画布被等比收缩,恰好用满 184 token 预算。
        let vision = config(14, 3, 184, 1);
        let small = plan_image_grid(182, 182, &vision);
        assert_eq!((small.0, small.1, small.2, small.3), (5, 5, 182, 182));
        let large = plan_image_grid(1120, 1120, &vision);
        assert_eq!(large.0 * (large.1 + 1) + 2, 184);
        assert_eq!((large.2, large.3), (546, 546));
        assert_eq!(large.3 % 14, 0);
    }

    #[test]
    fn patch_展开是行主序且归一化到负一到一() {
        let vision = config(2, 2, 64, 1);
        let processor = DeepseekV41ImageProcessor::new(vision).unwrap();
        // 4×4 白图:patch 2 → 网格 2×2,全部值 = 1.0(255 → +1)。
        let image = RgbImage::new(4, 4, vec![255; 4 * 4 * 3]).unwrap();
        let tensor = processor.preprocess(&image).unwrap();
        assert_eq!((tensor.rows, tensor.cols), (4, 12));
        assert_eq!(tensor.grid, VisionGrid { temporal: 1, height: 2, width: 2 });
        assert!(tensor.data.iter().all(|value| (*value - 1.0).abs() < 1e-6));
    }

    #[test]
    fn span_展开数量与布局一致() {
        let vision = config(14, 3, 1024, 1);
        let processor = DeepseekV41ImageProcessor::new(vision).unwrap();
        let image = RgbImage::new(182, 182, vec![0; 182 * 182 * 3]).unwrap();
        let tensor = processor.preprocess(&image).unwrap();
        let (tokens, spans) = expand_image_spans(&[7, 129_264, 9, 129_264], 129_264, &[tensor.clone(), tensor.clone()], 3).unwrap();
        assert_eq!(spans.len(), 2);
        // ViT 13×13 → LLM 5×5:span = 5*(5+1)+2 = 32 token。
        assert_eq!(spans[0].token_count(), 32);
        assert_eq!(spans[0].start, 1);
        assert_eq!(spans[1].start, 1 + spans[0].token_count() + 1);
        assert_eq!(tokens.len(), 2 + spans.iter().map(DeepseekV41ImageSpan::token_count).sum::<usize>());
        assert!(tokens.iter().all(|token| *token == 129_264 || *token == 7 || *token == 9));
        // 占位符与图像数量不一致报错。
        assert!(expand_image_spans(&[129_264], 129_264, &[tensor.clone(), tensor], 3).is_err());
    }

    #[test]
    fn rope_表h_w分块且两半重复() {
        // head_dim=16:rope_dim=8,前 4 个 h 频率、后 4 个 w 频率;两半重复;首位置恒 1。
        let grid = VisionGrid { temporal: 1, height: 2, width: 3 };
        let (cos, sin) = vision_rope_2d(grid, 16, 10_000.0).unwrap();
        assert_eq!(cos.len(), 6 * 16);
        for row in 0..6 {
            for offset in 0..8 {
                assert_eq!(cos[row * 16 + offset], cos[row * 16 + 8 + offset]);
            }
            let energy: f32 = (0..8).map(|offset| cos[row * 16 + offset] * cos[row * 16 + offset] + sin[row * 16 + offset] * sin[row * 16 + offset]).sum();
            assert!((energy - 8.0).abs() < 1e-3);
        }
        // (0,0) 全部角度 0;(0,1) 只有 w 段(下标 4..8)非平凡;(1,0) 只有 h 段。
        assert!(cos.iter().take(16).all(|value| (*value - 1.0).abs() < 1e-6));
        let row_w1 = 16;
        assert!(cos[row_w1..row_w1 + 4].iter().all(|value| (*value - 1.0).abs() < 1e-6), "h=0 段恒 1");
        assert!(cos[row_w1 + 4] < 1.0 && cos[row_w1 + 4] > 0.0, "w 频率角度应非平凡");
        let row_h1 = 3 * 16;
        assert!(cos[row_h1] < 1.0 && cos[row_h1] > 0.0, "h 频率角度应非平凡");
        assert!(cos[row_h1 + 4..row_h1 + 8].iter().all(|value| (*value - 1.0).abs() < 1e-6), "w=0 段恒 1");
    }

    #[test]
    fn span_行组装混排aligner与学习向量() {
        let hidden = 4;
        let span = DeepseekV41ImageSpan { start: 0, llm_rows: 2, llm_cols: 2 };
        let aligner = (0..span.llm_rows * span.llm_cols * hidden).map(|index| index as f32).collect::<Vec<_>>();
        let vector = |name: &str, bits: [u8; 2]| TensorData { name: name.to_owned(), dtype: "BF16".to_owned(), shape: vec![hidden], data: bits.iter().copied().cycle().take(hidden * 2).collect() };
        let embeddings = DeepseekV41ImageSpanEmbeddings { start: vector("image_start", [0x00, 0x3F]), newline: vector("image_newline", [0x00, 0xBF]), end: vector("image_end", [0x00, 0x40]) };
        let rows = span_rows_f32(&span, &aligner, hidden, &embeddings).unwrap();
        assert_eq!(rows.len(), span.token_count() * hidden);
        let start_value = half::bf16::from_le_bytes([0x00, 0x3F]).to_f32();
        let newline_value = half::bf16::from_le_bytes([0x00, 0xBF]).to_f32();
        let end_value = half::bf16::from_le_bytes([0x00, 0x40]).to_f32();
        assert!(rows[..hidden].iter().all(|value| (*value - start_value).abs() < 1e-6));
        // 第一行末尾是换行向量(第 4 个槽位),其后是第二行 aligner 行(网格行 1 列 0)。
        let newline_offset = hidden * 3;
        assert!(rows[newline_offset..newline_offset + hidden].iter().all(|value| (*value - newline_value).abs() < 1e-6));
        assert_eq!(rows[newline_offset + hidden], aligner[hidden * 2]);
        let tail = rows.len() - hidden;
        assert!(rows[tail..].iter().all(|value| (*value - end_value).abs() < 1e-6));
    }
}
