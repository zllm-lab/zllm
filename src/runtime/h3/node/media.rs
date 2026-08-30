//! H3 node 本地参考媒体解码；只负责文件到原始像素/采样，不包含模型算法。

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    backend::{DiffusionBackend, VaeBackend},
    runtime::h3::{
        H3AudioCondition, H3OrderedReferences, H3PackedReferenceCondition, H3QwenVisual, H3VisualVaeClip, H3VisualVaeInput, PreparedH3AudioEncoder, PreparedH3VideoEncoder, h3_encode_audio_condition, h3_encode_video_condition,
        h3_join_video_conditions, h3_noise_audio_condition, h3_noise_video_condition, h3_preprocess_reference_image, h3_preprocess_reference_video,
    },
    vision::{
        RgbImage,
        video::{decode_video, qwen_video_frames, video_dimensions},
    },
};

pub enum DecodedReference {
    Image(RgbImage),
    Video { frames: Vec<RgbImage>, audio: Option<Vec<f32>> },
    Audio(Vec<f32>),
}

pub struct LocalReference<'a> {
    pub source_index: usize,
    pub path: &'a Path,
    pub media_type: &'a str,
}

pub struct IndexedReference {
    pub source_index: usize,
    pub decoded: DecodedReference,
}

pub enum ConditioningMedia {
    Image { source_index: usize, qwen_image: RgbImage, visual_vae: H3VisualVaeInput },
    Video { source_index: usize, qwen_frames: Vec<RgbImage>, qwen_timestamps: Vec<f32>, visual_vae: Vec<H3VisualVaeClip>, audio: Option<Vec<f32>> },
    Audio { source_index: usize, samples: Vec<f32> },
}

static NEXT_TEMP_REFERENCE: AtomicU64 = AtomicU64::new(0);

pub fn decode_references(references: &[LocalReference<'_>]) -> Result<Vec<IndexedReference>, String> {
    let mut ordered = references.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|reference| reference.source_index);
    let mut previous = None;
    ordered
        .into_iter()
        .map(|reference| {
            if previous == Some(reference.source_index) {
                return Err(format!("H3 reference source_index={} 重复", reference.source_index));
            }
            previous = Some(reference.source_index);
            Ok(IndexedReference { source_index: reference.source_index, decoded: decode_reference(reference.path, reference.media_type)? })
        })
        .collect()
}

pub fn decode_control_references<T: serde::Serialize>(control: &T) -> Result<Vec<IndexedReference>, String> {
    let value = serde_json::to_value(control).map_err(|error| format!("序列化 H3 control message 失败: {error}"))?;
    let root = value.as_object().ok_or("H3 control message 不是 object")?;
    let materialized = ["references", "reference_media", "media"].into_iter().find_map(|field| root.get(field).and_then(serde_json::Value::as_array).filter(|items| !items.is_empty()));
    let mut owned = Vec::new();
    let mut temporary = Vec::new();
    if let Some(references) = materialized {
        owned.reserve(references.len());
        for (position, reference) in references.iter().enumerate() {
            let reference = reference.as_object().ok_or_else(|| format!("H3 reference {position} 不是 object"))?;
            let source_index = reference
                .get("source_index")
                .or_else(|| reference.get("index"))
                .and_then(serde_json::Value::as_u64)
                .map(|value| usize::try_from(value).map_err(|_| format!("H3 reference source_index={value} 超过 usize")))
                .transpose()?
                .unwrap_or(position);
            let path = ["local_path", "path", "file"].into_iter().find_map(|field| reference.get(field).and_then(serde_json::Value::as_str)).ok_or_else(|| format!("H3 reference {source_index} 缺少 node 本地路径"))?;
            let media_type = ["media_type", "mime_type", "content_type"].into_iter().find_map(|field| reference.get(field).and_then(serde_json::Value::as_str)).ok_or_else(|| format!("H3 reference {source_index} 缺少 MIME type"))?;
            owned.push((source_index, std::path::PathBuf::from(path), media_type.to_owned()));
        }
    } else if let Some(content) = root.get("content").and_then(serde_json::Value::as_array) {
        for (source_index, item) in content.iter().enumerate() {
            let item = item.as_object().ok_or_else(|| format!("H3 content {source_index} 不是 object"))?;
            let kind = item.get("type").and_then(serde_json::Value::as_str);
            let role = item.get("role").and_then(serde_json::Value::as_str);
            let (field, media_type) = match (kind, role) {
                (Some("image_url"), None | Some("first_frame" | "last_frame" | "reference_image")) => ("image_url", "image/*"),
                (Some("video_url"), Some("reference_video")) => ("video_url", "video/*"),
                (Some("audio_url"), Some("reference_audio")) => ("audio_url", "audio/*"),
                _ => continue,
            };
            let url = item.get(field).and_then(serde_json::Value::as_object).and_then(|url| url.get("url")).and_then(serde_json::Value::as_str).ok_or_else(|| format!("H3 content {source_index} 缺少 {field}.url"))?;
            let (path, media_type, is_temporary) = materialize_reference_url(url, media_type, source_index)?;
            if is_temporary {
                temporary.push(path.clone());
            }
            owned.push((source_index, path, media_type));
        }
    }
    let borrowed = owned.iter().map(|(source_index, path, media_type)| LocalReference { source_index: *source_index, path: path.as_path(), media_type }).collect::<Vec<_>>();
    let decoded = decode_references(&borrowed);
    for path in temporary {
        let _ = fs::remove_file(path);
    }
    decoded
}

pub fn reference_url_dimensions(url: &str, media_type: &str) -> Result<Option<(usize, usize)>, String> {
    let (path, media_type, temporary) = materialize_reference_url(url, media_type, 0)?;
    let dimensions = reference_dimensions(&path, &media_type);
    if temporary {
        let _ = fs::remove_file(path);
    }
    dimensions
}

fn materialize_reference_url(url: &str, media_type: &str, source_index: usize) -> Result<(PathBuf, String, bool), String> {
    if let Some(path) = url.strip_prefix("mm_file://") {
        return Ok((PathBuf::from(path), media_type.to_owned(), false));
    }
    if let Some(data) = url.strip_prefix("data:") {
        let (metadata, payload) = data.split_once(',').ok_or_else(|| format!("H3 reference {source_index} data URI 缺少逗号"))?;
        let mut metadata = metadata.split(';');
        let actual_media_type = metadata.next().filter(|value| !value.is_empty()).unwrap_or(media_type);
        if !metadata.any(|value| value.eq_ignore_ascii_case("base64")) {
            return Err(format!("H3 reference {source_index} data URI 必须使用 base64"));
        }
        return materialize_base64(payload, actual_media_type, source_index);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let path = temporary_reference_path(source_index, media_type);
        let output = Command::new("curl")
            .args(["--fail", "--location", "--silent", "--show-error", "--connect-timeout", "20", "--max-time", "300", "--output"])
            .arg(&path)
            .arg(url)
            .output()
            .map_err(|error| format!("启动 curl 下载 H3 reference {source_index}: {error}"))?;
        if !output.status.success() {
            let _ = fs::remove_file(&path);
            return Err(format!("下载 H3 reference {source_index} 失败: {}", String::from_utf8_lossy(&output.stderr).trim()));
        }
        let bytes = fs::read(&path).map_err(|error| format!("读取 H3 reference {}: {error}", path.display()))?;
        let actual_media_type = sniff_media_type(&bytes).unwrap_or(media_type);
        if actual_media_type == media_type {
            return Ok((path, media_type.to_owned(), true));
        }
        let actual_path = temporary_reference_path(source_index, actual_media_type);
        fs::rename(&path, &actual_path).map_err(|error| format!("修正 H3 reference 格式 {}: {error}", path.display()))?;
        return Ok((actual_path, actual_media_type.to_owned(), true));
    }
    materialize_base64(url, media_type, source_index)
}

fn materialize_base64(payload: &str, media_type: &str, source_index: usize) -> Result<(PathBuf, String, bool), String> {
    if payload.len() > 512 * 1024 * 1024 {
        return Err(format!("H3 reference {source_index} base64 超过 512 MiB"));
    }
    let bytes = decode_base64(payload).map_err(|error| format!("H3 reference {source_index} base64: {error}"))?;
    let actual_media_type = sniff_media_type(&bytes).unwrap_or(media_type);
    let path = temporary_reference_path(source_index, actual_media_type);
    fs::write(&path, bytes).map_err(|error| format!("写入 H3 reference {}: {error}", path.display()))?;
    Ok((path, actual_media_type.to_owned(), true))
}

fn temporary_reference_path(source_index: usize, media_type: &str) -> PathBuf {
    let extension = match media_type {
        "image/png" => "png",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/jpeg" | "image/jpg" => "jpg",
        value if value.starts_with("video/") => "video",
        value if value.starts_with("audio/") => "audio",
        _ => "media",
    };
    let sequence = NEXT_TEMP_REFERENCE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("zllm-h3-reference-{}-{source_index}-{sequence}.{extension}", std::process::id()))
}

fn sniff_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else {
        None
    }
}

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
            return Err("padding 后仍有数据".to_owned());
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return Err(format!("包含非法字符 0x{byte:02x}")),
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
        return Err("末尾位无效".to_owned());
    }
    Ok(output)
}

pub fn prepare_conditioning_media(references: Vec<IndexedReference>) -> Result<Vec<ConditioningMedia>, String> {
    references
        .into_iter()
        .map(|reference| match reference.decoded {
            DecodedReference::Image(image) => {
                let visual_vae = h3_preprocess_reference_image(&image).map_err(|error| format!("H3 image {} VisualVAE preprocess: {error:?}", reference.source_index))?;
                Ok(ConditioningMedia::Image { source_index: reference.source_index, qwen_image: image, visual_vae })
            }
            DecodedReference::Video { frames, audio } => {
                let visual_vae = h3_preprocess_reference_video(&frames).map_err(|error| format!("H3 video {} VisualVAE preprocess: {error:?}", reference.source_index))?;
                let (qwen_frames, qwen_timestamps) = qwen_video_frames(&frames)?;
                Ok(ConditioningMedia::Video { source_index: reference.source_index, qwen_frames, qwen_timestamps, visual_vae, audio })
            }
            DecodedReference::Audio(samples) => Ok(ConditioningMedia::Audio { source_index: reference.source_index, samples }),
        })
        .collect()
}

pub fn encode_conditioning_media<B, Q, ImageEncoder, VideoEncoder>(
    backend: &B,
    video_encoder: Option<&PreparedH3VideoEncoder<B::Weight>>,
    audio_encoder: Option<&PreparedH3AudioEncoder<B::Weight>>,
    media: Vec<ConditioningMedia>,
    seed: u64,
    timestep: f32,
    mut encode_qwen_image: ImageEncoder,
    mut encode_qwen_video: VideoEncoder,
) -> Result<(Vec<(usize, H3QwenVisual<Q>)>, H3OrderedReferences<B::Tensor>), String>
where
    B: VaeBackend + DiffusionBackend,
    ImageEncoder: FnMut(&RgbImage) -> Result<H3QwenVisual<Q>, String>,
    VideoEncoder: FnMut(&[RgbImage], &[f32]) -> Result<H3QwenVisual<Q>, String>,
{
    let mut qwen_visuals = Vec::with_capacity(media.len());
    let mut conditions = Vec::with_capacity(media.len() * 2);
    for reference in media {
        match reference {
            ConditioningMedia::Image { source_index, qwen_image, visual_vae } => {
                let video_encoder = video_encoder.ok_or("H3 图片参考缺少 video encoder")?;
                let qwen_visual = encode_qwen_image(&qwen_image)?;
                let ref2va_started = std::time::Instant::now();
                let video_condition = h3_encode_video_condition(backend, video_encoder, visual_vae.values, visual_vae.shape).map_err(backend_error)?;
                eprintln!("[h3-node] Ref2VA image encode wall={:.3}s", ref2va_started.elapsed().as_secs_f64());
                let video_condition = h3_noise_video_condition(backend, video_condition, seed, timestep).map_err(backend_error)?;
                qwen_visuals.push((source_index, qwen_visual));
                conditions.push(H3PackedReferenceCondition::Video { source_index, condition: video_condition });
            }
            ConditioningMedia::Video { source_index, qwen_frames, qwen_timestamps, visual_vae, audio } => {
                let video_encoder = video_encoder.ok_or("H3 视频参考缺少 video encoder")?;
                let qwen_visual = encode_qwen_video(&qwen_frames, &qwen_timestamps)?;
                let mut clips = Vec::with_capacity(visual_vae.len());
                for clip in visual_vae {
                    let condition = h3_encode_video_condition(backend, video_encoder, clip.input.values, clip.input.shape).map_err(backend_error)?;
                    clips.push((condition, clip.drop_latent_tokens));
                }
                let video_condition = h3_join_video_conditions(backend, clips).map_err(backend_error)?;
                let video_condition = h3_noise_video_condition(backend, video_condition, seed, timestep).map_err(backend_error)?;
                let audio_condition = if let Some(samples) = audio {
                    let audio_encoder = audio_encoder.ok_or("H3 视频参考音轨缺少 audio encoder")?;
                    let condition = h3_encode_audio_condition(backend, audio_encoder, samples, 1).map_err(backend_error)?;
                    Some(h3_noise_audio_condition(backend, condition, seed, timestep).map_err(backend_error)?)
                } else {
                    None
                };
                qwen_visuals.push((source_index, qwen_visual));
                if let Some(condition) = audio_condition {
                    conditions.push(H3PackedReferenceCondition::Audio { source_index, condition });
                }
                conditions.push(H3PackedReferenceCondition::Video { source_index, condition: video_condition });
            }
            ConditioningMedia::Audio { source_index, samples } => {
                let audio_encoder = audio_encoder.ok_or("H3 音频参考缺少 audio encoder")?;
                let audio_condition: H3AudioCondition<B::Tensor> = h3_encode_audio_condition(backend, audio_encoder, samples, 1).map_err(backend_error)?;
                let audio_condition = h3_noise_audio_condition(backend, audio_condition, seed, timestep).map_err(backend_error)?;
                conditions.push(H3PackedReferenceCondition::Audio { source_index, condition: audio_condition });
            }
        }
    }
    if !conditions.is_empty() && qwen_visuals.is_empty() {
        return Err("H3 Ref2VA 不允许仅提供音频参考，至少需要一张图片或一段视频".to_owned());
    }
    Ok((qwen_visuals, H3OrderedReferences { conditions }))
}

fn backend_error(error: crate::backend::BackendError) -> String {
    format!("{error:?}")
}

pub fn decode_reference(path: &Path, media_type: &str) -> Result<DecodedReference, String> {
    if media_type.starts_with("image/") {
        return RgbImage::open(path).map(DecodedReference::Image);
    }
    if media_type.starts_with("video/") {
        let frames = decode_video(path)?;
        let audio = decode_audio(path)?.filter(|samples| !samples.is_empty());
        return Ok(DecodedReference::Video { frames, audio });
    }
    if media_type.starts_with("audio/") {
        let samples = decode_audio(path)?.ok_or_else(|| format!("参考音频 {} 没有可解码的音轨", path.display()))?;
        if samples.is_empty() {
            return Err(format!("参考音频 {} 解码结果为空", path.display()));
        }
        return Ok(DecodedReference::Audio(samples));
    }
    Err(format!("H3 不支持参考媒体类型 {media_type:?}"))
}

pub fn reference_dimensions(path: &Path, media_type: &str) -> Result<Option<(usize, usize)>, String> {
    if media_type.starts_with("image/") {
        let image = RgbImage::open(path)?;
        return Ok(Some((image.width, image.height)));
    }
    if media_type.starts_with("video/") {
        return video_dimensions(path).map(Some);
    }
    Ok(None)
}

fn decode_audio(path: &Path) -> Result<Option<Vec<f32>>, String> {
    let decoded = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(path).args(["-map", "0:a:0?", "-vn", "-ac", "1", "-ar", "24000", "-f", "f32le", "pipe:1"]).output().map_err(|error| format!("启动 ffmpeg 音频解码失败: {error}"))?;
    if !decoded.status.success() {
        return Err(format!("ffmpeg 音频解码 {} 失败: {}", path.display(), String::from_utf8_lossy(&decoded.stderr)));
    }
    if decoded.stdout.is_empty() {
        return Ok(None);
    }
    if !decoded.stdout.len().is_multiple_of(4) {
        return Err(format!("ffmpeg 音频 bytes={} 不是 F32LE", decoded.stdout.len()));
    }
    let samples = decoded.stdout.chunks_exact(4).map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])).collect();
    Ok(Some(samples))
}
