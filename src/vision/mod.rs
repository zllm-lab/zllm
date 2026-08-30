//! 多模态视觉输入的公共边界。
//!
//! 图像 codec、像素前处理与模型专有 Vision Encoder 分离。Kimi 等模型可以
//! 复用 `PatchImageProcessor`，或在相同输入/输出边界上提供自己的 processor。

use image::ImageReader;
use std::{ops::Range, path::Path};

pub mod video;

#[derive(Clone, Debug)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self, String> {
        let expected = width.checked_mul(height).and_then(|pixels| pixels.checked_mul(3)).ok_or_else(|| "RGB 图像尺寸溢出".to_owned())?;
        if width == 0 || height == 0 {
            return Err("RGB 图像宽高必须大于 0".to_owned());
        }
        if pixels.len() != expected {
            return Err(format!("RGB 图像字节数 {}，期望 {expected}", pixels.len()));
        }
        Ok(Self { width, height, pixels })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let decoded = ImageReader::open(path).map_err(|error| format!("打开图像 {} 失败: {error}", path.display()))?.decode().map_err(|error| format!("解码图像 {} 失败: {error}", path.display()))?.to_rgb8();
        Self::new(decoded.width() as usize, decoded.height() as usize, decoded.into_raw())
    }

    pub fn from_memory(bytes: &[u8]) -> Result<Self, String> {
        let decoded = image::load_from_memory(bytes).map_err(|error| format!("解码图像 bytes={} 失败: {error}", bytes.len()))?.to_rgb8();
        Self::new(decoded.width() as usize, decoded.height() as usize, decoded.into_raw())
    }
}

/// 把 OpenAI 风格 `image_url.url` 物化并解码为 RGB 图像。
/// 依次支持 `mm_file://`/`file://` 本地路径、`data:...;base64,` URI、
/// http(s) curl 下载、已存在的本地路径与裸 base64；格式由解码器按魔数识别。
pub fn image_from_url(url: &str) -> Result<RgbImage, String> {
    for scheme in ["mm_file://", "file://"] {
        if let Some(path) = url.strip_prefix(scheme) {
            return RgbImage::open(path);
        }
    }
    if let Some(data) = url.strip_prefix("data:") {
        let (metadata, payload) = data.split_once(',').ok_or("image data URI 缺少逗号")?;
        if !metadata.split(';').skip(1).any(|parameter| parameter.eq_ignore_ascii_case("base64")) {
            return Err("image data URI 必须使用 base64".to_owned());
        }
        let bytes = decode_base64(payload)?;
        return RgbImage::from_memory(&bytes);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let path = temporary_image_path();
        let output = std::process::Command::new("curl")
            .args(["--fail", "--location", "--silent", "--show-error", "--connect-timeout", "20", "--max-time", "300", "--output"])
            .arg(&path)
            .arg(url)
            .output()
            .map_err(|error| format!("启动 curl 下载图像失败: {error}"))?;
        if !output.status.success() {
            let _ = std::fs::remove_file(&path);
            return Err(format!("下载图像 {url} 失败: {}", String::from_utf8_lossy(&output.stderr).trim()));
        }
        let result = RgbImage::open(&path);
        let _ = std::fs::remove_file(&path);
        return result;
    }
    let path = Path::new(url);
    if path.is_file() {
        return RgbImage::open(path);
    }
    // 无 scheme 且不是本地文件时按裸 base64 尝试；解码器按魔数识别格式。
    let bytes = decode_base64(url)?;
    RgbImage::from_memory(&bytes)
}

static NEXT_TEMPORARY_IMAGE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn temporary_image_path() -> std::path::PathBuf {
    let sequence = NEXT_TEMPORARY_IMAGE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("zllm-image-{}-{sequence}.download", std::process::id()))
}

/// 手写 base64 解码：兼容标准与 URL-safe 字母表，忽略空白，校验 padding。
fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let mut output = Vec::with_capacity(input.len().saturating_mul(3) / 4);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    let mut sextets = 0usize;
    let mut padded = false;
    for byte in input.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if byte == b'=' {
            padded = true;
            continue;
        }
        if padded {
            return Err("base64 padding 后仍有数据".to_owned());
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return Err(format!("base64 包含非法字符 0x{byte:02x}")),
        };
        buffer = (buffer << 6) | u32::from(value);
        bits += 6;
        sextets += 1;
        if bits >= 8 {
            bits -= 8;
            output.push((buffer >> bits) as u8);
            buffer &= if bits == 0 { 0 } else { (1u32 << bits) - 1 };
        }
    }
    if sextets % 4 == 1 || (bits != 0 && buffer != 0) {
        return Err("base64 末尾位无效".to_owned());
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionGrid {
    pub temporal: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Debug)]
pub struct ImageTensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub grid: VisionGrid,
    pub merge_size: usize,
}

impl ImageTensor {
    pub fn visual_token_count(&self) -> Result<usize, String> {
        let merge = self.merge_size.checked_mul(self.merge_size).ok_or_else(|| "视觉 merge_size 溢出".to_owned())?;
        if merge == 0 || !self.rows.is_multiple_of(merge) {
            return Err(format!("视觉 patch 行数 {} 不能按 merge_size={} 合并", self.rows, self.merge_size));
        }
        Ok(self.rows / merge)
    }
}

pub trait ImageProcessor {
    fn preprocess(&self, image: &RgbImage) -> Result<ImageTensor, String>;
}

pub enum ContentPart<'a> {
    Text(&'a str),
    Image(&'a RgbImage),
}

#[derive(Clone, Debug)]
pub struct ImageTokenRange {
    pub image_index: usize,
    pub tokens: Range<usize>,
}

#[derive(Debug)]
pub struct MultimodalInput {
    pub token_ids: Vec<u32>,
    pub images: Vec<ImageTensor>,
    pub image_token_ranges: Vec<ImageTokenRange>,
}

pub trait MultimodalProcessor {
    fn process(&self, parts: &[ContentPart<'_>]) -> Result<MultimodalInput, String>;
}

/// chunked prefill 的 BF16 行覆盖:把落在图像 token 区间内的文本 embedding 行
/// 替换为 Vision Encoder 输出(BF16 bits)。区间允许跨越 chunk 边界,
/// 只覆盖与 `[chunk_position, chunk_position + rows)` 相交的部分。
pub fn splice_image_embeddings_bf16(hidden: &mut [u16], hidden_size: usize, chunk_position: usize, ranges: &[ImageTokenRange], image_embeddings: &[Vec<u16>]) -> Result<(), String> {
    if hidden_size == 0 || !hidden.len().is_multiple_of(hidden_size) {
        return Err(format!("BF16 文本 embedding shape 无效: values={} hidden_size={hidden_size}", hidden.len()));
    }
    let rows = hidden.len() / hidden_size;
    let chunk_end = chunk_position.checked_add(rows).ok_or("BF16 视觉覆盖 chunk 边界溢出")?;
    let mut previous_end = 0usize;
    for range in ranges {
        if range.tokens.start < previous_end || range.tokens.start > range.tokens.end {
            return Err(format!("图像 {} token range {:?} 无序或非法", range.image_index, range.tokens));
        }
        let image = image_embeddings.get(range.image_index).ok_or_else(|| format!("缺少图像 {} 的 BF16 视觉 embedding", range.image_index))?;
        let expected = range.tokens.len().checked_mul(hidden_size).ok_or("BF16 视觉 embedding 大小溢出")?;
        if image.len() != expected {
            return Err(format!("图像 {} BF16 embedding values={}，期望 {expected}", range.image_index, image.len()));
        }
        let start = range.tokens.start.max(chunk_position);
        let end = range.tokens.end.min(chunk_end);
        if start < end {
            let source = (start - range.tokens.start) * hidden_size;
            let target = (start - chunk_position) * hidden_size;
            let length = (end - start) * hidden_size;
            hidden[target..target + length].copy_from_slice(&image[source..source + length]);
        }
        previous_end = range.tokens.end;
    }
    Ok(())
}

/// CPU reference：把 Vision Encoder 输出替换到文本 embedding 的 image token 行。
pub fn scatter_image_embeddings_f32(hidden: &mut [f32], hidden_size: usize, ranges: &[ImageTokenRange], image_embeddings: &[Vec<f32>]) -> Result<(), String> {
    if hidden_size == 0 || !hidden.len().is_multiple_of(hidden_size) {
        return Err(format!("文本 embedding shape 无效: values={} hidden_size={hidden_size}", hidden.len()));
    }
    let rows = hidden.len() / hidden_size;
    let mut previous_end = 0;
    for range in ranges {
        if range.tokens.start < previous_end || range.tokens.end > rows || range.tokens.start > range.tokens.end {
            return Err(format!("图像 {} token range {:?} 无效，文本行数={rows}", range.image_index, range.tokens));
        }
        let image = image_embeddings.get(range.image_index).ok_or_else(|| format!("缺少图像 {} 的视觉 embedding", range.image_index))?;
        let expected = range.tokens.len().checked_mul(hidden_size).ok_or_else(|| "视觉 embedding 大小溢出".to_owned())?;
        if image.len() != expected {
            return Err(format!("图像 {} embedding values={}，期望 {expected}", range.image_index, image.len()));
        }
        hidden[range.tokens.start * hidden_size..range.tokens.end * hidden_size].copy_from_slice(image);
        previous_end = range.tokens.end;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct PatchImageConfig {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub max_aspect_ratio: f64,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

pub struct PatchImageProcessor {
    config: PatchImageConfig,
}

impl PatchImageProcessor {
    pub fn new(config: PatchImageConfig) -> Result<Self, String> {
        if config.patch_size == 0 || config.temporal_patch_size == 0 || config.merge_size == 0 {
            return Err("patch_size、temporal_patch_size、merge_size 必须大于 0".to_owned());
        }
        if config.min_pixels == 0 || config.min_pixels > config.max_pixels {
            return Err("视觉 min_pixels/max_pixels 无效".to_owned());
        }
        if !config.max_aspect_ratio.is_finite() || config.max_aspect_ratio < 1.0 {
            return Err("视觉 max_aspect_ratio 无效".to_owned());
        }
        if config.std.iter().any(|value| !value.is_finite() || *value == 0.0) {
            return Err("视觉归一化 std 必须是非零有限值".to_owned());
        }
        Ok(Self { config })
    }

    pub fn target_size(&self, image: &RgbImage) -> Result<(usize, usize), String> {
        let factor = self.config.patch_size.checked_mul(self.config.merge_size).ok_or_else(|| "视觉 resize factor 溢出".to_owned())?;
        smart_resize(image.height, image.width, factor, self.config.min_pixels, self.config.max_pixels, self.config.max_aspect_ratio)
    }
}

impl ImageProcessor for PatchImageProcessor {
    fn preprocess(&self, image: &RgbImage) -> Result<ImageTensor, String> {
        let (height, width) = self.target_size(image)?;
        let source = image::RgbImage::from_raw(image.width as u32, image.height as u32, image.pixels.clone()).ok_or_else(|| "构造 RGB 图像失败".to_owned())?;
        let resized = pillow_bicubic_resize(&source, width as u32, height as u32);

        let patch = self.config.patch_size;
        let merge = self.config.merge_size;
        let grid_height = height / patch;
        let grid_width = width / patch;
        let rows = grid_height.checked_mul(grid_width).ok_or_else(|| "视觉 patch 行数溢出".to_owned())?;
        let cols = 3usize.checked_mul(self.config.temporal_patch_size).and_then(|value| value.checked_mul(patch)).and_then(|value| value.checked_mul(patch)).ok_or_else(|| "视觉 patch 列数溢出".to_owned())?;
        let capacity = rows.checked_mul(cols).ok_or_else(|| "视觉 patch tensor 过大".to_owned())?;
        let mut data = Vec::with_capacity(capacity);

        // merge group 内的 patch 连续排列，Vision Encoder 后可直接做 2×2 patch merge。
        for block_y in 0..grid_height / merge {
            for block_x in 0..grid_width / merge {
                for merge_y in 0..merge {
                    for merge_x in 0..merge {
                        let patch_y = (block_y * merge + merge_y) * patch;
                        let patch_x = (block_x * merge + merge_x) * patch;
                        for channel in 0..3 {
                            for _ in 0..self.config.temporal_patch_size {
                                for y in 0..patch {
                                    for x in 0..patch {
                                        let pixel = resized.get_pixel((patch_x + x) as u32, (patch_y + y) as u32)[channel];
                                        data.push((pixel as f32 / 255.0 - self.config.mean[channel]) / self.config.std[channel]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ImageTensor { data, rows, cols, grid: VisionGrid { temporal: 1, height: grid_height, width: grid_width }, merge_size: merge })
    }
}

/// 官方 Torchvision 的 antialiased BICUBIC 与 Pillow 很接近。`image` crate 的
/// CatmullRom 使用不同的 pass 顺序和取整时机，细小像素差会被多层 ViT 放大。
pub(crate) fn pillow_bicubic_resize(source: &image::RgbImage, width: u32, height: u32) -> image::RgbImage {
    const PRECISION_BITS: u32 = 22;

    struct Coefficients {
        start: usize,
        weights: Vec<i32>,
    }

    fn cubic(value: f64) -> f64 {
        let value = value.abs();
        if value < 1.0 {
            ((1.5 * value - 2.5) * value) * value + 1.0
        } else if value < 2.0 {
            ((-0.5 * value + 2.5) * value - 4.0) * value + 2.0
        } else {
            0.0
        }
    }

    fn coefficients(input: usize, output: usize) -> Vec<Coefficients> {
        let scale = input as f64 / output as f64;
        let filter_scale = scale.max(1.0);
        let support = 2.0 * filter_scale;
        let fixed_scale = (1u64 << PRECISION_BITS) as f64;
        (0..output)
            .map(|index| {
                let center = (index as f64 + 0.5) * scale;
                let start = ((center - support + 0.5) as isize).clamp(0, input as isize) as usize;
                let end = ((center + support + 0.5) as isize).clamp(start as isize, input as isize) as usize;
                let mut weights: Vec<f64> = (start..end).map(|source| cubic((source as f64 - center + 0.5) / filter_scale)).collect();
                let sum: f64 = weights.iter().sum();
                if sum != 0.0 {
                    for weight in &mut weights {
                        *weight /= sum;
                    }
                }
                let weights = weights
                    .into_iter()
                    .map(|weight| {
                        let scaled = weight * fixed_scale;
                        if scaled < 0.0 { (scaled - 0.5) as i32 } else { (scaled + 0.5) as i32 }
                    })
                    .collect();
                Coefficients { start, weights }
            })
            .collect()
    }

    let source_width = source.width() as usize;
    let source_height = source.height() as usize;
    let width = width as usize;
    let height = height as usize;
    let horizontal = coefficients(source_width, width);
    let vertical = coefficients(source_height, height);
    let rounding = 1i64 << (PRECISION_BITS - 1);
    let mut temporary = vec![0u8; source_height * width * 3];
    for y in 0..source_height {
        for (x, coefficients) in horizontal.iter().enumerate() {
            for channel in 0..3 {
                let sum = coefficients.weights.iter().enumerate().fold(rounding, |sum, (offset, &weight)| sum + i64::from(source.get_pixel((coefficients.start + offset) as u32, y as u32)[channel]) * i64::from(weight));
                temporary[(y * width + x) * 3 + channel] = (sum >> PRECISION_BITS).clamp(0, 255) as u8;
            }
        }
    }

    let mut output = vec![0u8; height * width * 3];
    for (y, coefficients) in vertical.iter().enumerate() {
        for x in 0..width {
            for channel in 0..3 {
                let sum = coefficients.weights.iter().enumerate().fold(rounding, |sum, (offset, &weight)| sum + i64::from(temporary[((coefficients.start + offset) * width + x) * 3 + channel]) * i64::from(weight));
                output[(y * width + x) * 3 + channel] = (sum >> PRECISION_BITS).clamp(0, 255) as u8;
            }
        }
    }
    image::RgbImage::from_raw(width as u32, height as u32, output).expect("BICUBIC 输出大小已验证")
}

/// 等比缩放到目标框后居中补黑。Gemma 4 等动态分辨率视觉预处理器先按
/// patch 预算确定目标框，再用这种 PAD_CEIL 语义避免拉伸原图。
pub(crate) fn pillow_bicubic_resize_contain(source: &image::RgbImage, width: u32, height: u32) -> image::RgbImage {
    let scale = (width as f32 / source.width() as f32).min(height as f32 / source.height() as f32);
    let resized_width = ((source.width() as f32 * scale).ceil() as u32).min(width);
    let resized_height = ((source.height() as f32 * scale).ceil() as u32).min(height);
    let resized = pillow_bicubic_resize(source, resized_width, resized_height);
    let offset_x = (width - resized_width) / 2;
    let offset_y = (height - resized_height) / 2;
    let mut output = image::RgbImage::new(width, height);
    image::imageops::replace(&mut output, &resized, i64::from(offset_x), i64::from(offset_y));
    output
}

fn smart_resize(height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize, max_aspect_ratio: f64) -> Result<(usize, usize), String> {
    let ratio = height.max(width) as f64 / height.min(width) as f64;
    if ratio > max_aspect_ratio {
        return Err(format!("图像宽高比 {ratio:.3} 超过上限 {max_aspect_ratio}"));
    }
    let mut target_height = round_by_factor(height as f64, factor).max(factor);
    let mut target_width = round_by_factor(width as f64, factor).max(factor);
    let area = target_height.checked_mul(target_width).ok_or_else(|| "图像 resize 面积溢出".to_owned())?;
    if area > max_pixels {
        let scale = ((height as f64 * width as f64) / max_pixels as f64).sqrt();
        target_height = floor_by_factor(height as f64 / scale, factor).max(factor);
        target_width = floor_by_factor(width as f64 / scale, factor).max(factor);
    } else if area < min_pixels {
        let scale = (min_pixels as f64 / (height as f64 * width as f64)).sqrt();
        target_height = ceil_by_factor(height as f64 * scale, factor).max(factor);
        target_width = ceil_by_factor(width as f64 * scale, factor).max(factor);
    }
    Ok((target_height, target_width))
}

fn round_by_factor(value: f64, factor: usize) -> usize {
    (value / factor as f64).round() as usize * factor
}

fn floor_by_factor(value: f64, factor: usize) -> usize {
    (value / factor as f64).floor() as usize * factor
}

fn ceil_by_factor(value: f64, factor: usize) -> usize {
    (value / factor as f64).ceil() as usize * factor
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PatchImageConfig {
        PatchImageConfig { patch_size: 14, temporal_patch_size: 2, merge_size: 2, min_pixels: 28 * 28, max_pixels: 672 * 672, max_aspect_ratio: 200.0, mean: [0.0; 3], std: [1.0; 3] }
    }

    #[test]
    fn bicubic_matches_pillow_rounding() {
        let image = image::RgbImage::from_raw(4, 3, (0..36).collect()).unwrap();
        let resized = pillow_bicubic_resize(&image, 2, 2);
        assert_eq!(resized.into_raw(), vec![6, 7, 8, 11, 12, 13, 22, 23, 24, 27, 28, 29]);
    }

    #[test]
    fn bicubic_contain_preserves_ratio_and_centers_black_padding() {
        let image = image::RgbImage::from_pixel(4, 8, image::Rgb([255, 128, 64]));
        let resized = pillow_bicubic_resize_contain(&image, 6, 6);
        assert_eq!(resized.dimensions(), (6, 6));
        assert_eq!(resized.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(resized.get_pixel(1, 0).0, [255, 128, 64]);
        assert_eq!(resized.get_pixel(3, 5).0, [255, 128, 64]);
        assert_eq!(resized.get_pixel(5, 5).0, [0, 0, 0]);
    }

    #[test]
    fn patch_tensor_shape_and_merge_count() {
        let image = RgbImage::new(28, 28, vec![255; 28 * 28 * 3]).unwrap();
        let tensor = PatchImageProcessor::new(config()).unwrap().preprocess(&image).unwrap();
        assert_eq!(tensor.grid, VisionGrid { temporal: 1, height: 2, width: 2 });
        assert_eq!((tensor.rows, tensor.cols), (4, 3 * 2 * 14 * 14));
        assert_eq!(tensor.visual_token_count().unwrap(), 1);
        assert_eq!(tensor.data.len(), tensor.rows * tensor.cols);
        assert!(tensor.data.iter().all(|value| *value == 1.0));
    }

    #[test]
    fn resize_respects_factor_and_pixel_budget() {
        let image = RgbImage::new(1_000, 500, vec![0; 1_000 * 500 * 3]).unwrap();
        let processor = PatchImageProcessor::new(config()).unwrap();
        let (height, width) = processor.target_size(&image).unwrap();
        assert_eq!(height % 28, 0);
        assert_eq!(width % 28, 0);
        assert!(height * width <= 672 * 672);
    }

    #[test]
    fn scatter_replaces_only_image_token_rows() {
        let mut hidden = vec![0.0; 5 * 2];
        let ranges = vec![ImageTokenRange { image_index: 0, tokens: 1..3 }];
        scatter_image_embeddings_f32(&mut hidden, 2, &ranges, &[vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
        assert_eq!(hidden, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn splice_covers_only_chunk_overlap_and_spans_boundaries() {
        // 全 prompt 8 行,图像 0 占 1..5(跨 chunk 边界),图像 1 占 6..8。
        let ranges = vec![ImageTokenRange { image_index: 0, tokens: 1..5 }, ImageTokenRange { image_index: 1, tokens: 6..8 }];
        let images = vec![(10u16..18).collect::<Vec<_>>(), (20u16..24).collect::<Vec<_>>()];
        // chunk 覆盖 3..6:行 3、4 是图像 0 的第 2、3 行,行 5 是文本;
        // 图像 1 完整落在 chunk 外。
        let mut chunk = vec![0u16; 3 * 2];
        splice_image_embeddings_bf16(&mut chunk, 2, 3, &ranges, &images).unwrap();
        assert_eq!(chunk, vec![14, 15, 16, 17, 0, 0]);
        // 第二个 chunk 覆盖 6..8:图像 1 完整替换。
        let mut chunk = vec![0u16; 2 * 2];
        splice_image_embeddings_bf16(&mut chunk, 2, 6, &ranges, &images).unwrap();
        assert_eq!(chunk, vec![20, 21, 22, 23]);
    }

    #[test]
    fn splice_rejects_bad_shapes_and_ordering() {
        let mut chunk = vec![0u16; 4];
        let ranges = vec![ImageTokenRange { image_index: 0, tokens: 0..2 }];
        // 图像行数与区间不一致。
        assert!(splice_image_embeddings_bf16(&mut chunk, 2, 0, &ranges, &[vec![1u16; 3]]).is_err());
        // 区间倒序。
        let reversed = vec![ImageTokenRange { image_index: 0, tokens: std::ops::Range { start: 3, end: 2 } }];
        assert!(splice_image_embeddings_bf16(&mut chunk, 2, 0, &reversed, &[vec![1u16; 2]]).is_err());
        // hidden_size 不整除。
        assert!(splice_image_embeddings_bf16(&mut chunk, 3, 0, &ranges, &[vec![1u16; 6]]).is_err());
    }

    fn encode_base64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
        for chunk in bytes.chunks(3) {
            let group = (chunk[0] as u32) << 16 | (chunk.get(1).copied().unwrap_or(0) as u32) << 8 | chunk.get(2).copied().unwrap_or(0) as u32;
            output.push(ALPHABET[(group >> 18) as usize & 63] as char);
            output.push(ALPHABET[(group >> 12) as usize & 63] as char);
            output.push(if chunk.len() > 1 { ALPHABET[(group >> 6) as usize & 63] as char } else { '=' });
            output.push(if chunk.len() > 2 { ALPHABET[group as usize & 63] as char } else { '=' });
        }
        output
    }

    fn tiny_png() -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(3, 2, image::Rgb([200u8, 100, 50]))).write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        bytes.into_inner()
    }

    #[test]
    fn image_from_data_uri_and_raw_base64() {
        let png = tiny_png();
        let encoded = encode_base64(&png);
        let from_uri = image_from_url(&format!("data:image/png;base64,{encoded}")).unwrap();
        assert_eq!((from_uri.width, from_uri.height), (3, 2));
        let from_raw = image_from_url(&encoded).unwrap();
        assert_eq!(from_raw.pixels, from_uri.pixels);
    }

    #[test]
    fn image_from_url_rejects_bad_input() {
        assert!(image_from_url("data:image/png;base64,###").is_err());
        assert!(image_from_url("not a path or base64 !!!").is_err());
        assert!(image_from_url("/nonexistent/zllm-image-test.png").is_err());
        // 非 base64 参数的 data URI 拒绝
        assert!(image_from_url("data:image/png,raw-bytes").is_err());
    }
}
