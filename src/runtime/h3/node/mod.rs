#[cfg(all(target_os = "linux", feature = "with-rocm"))]
use std::{
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::Path,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
use crate::artifact::{write_wav, write_y4m};

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
use serde_json::Value;

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
use crate::{
    backend::{BackendError, BackendResources, rocm::RocmTensor},
    config::{H3NodeExecutionConfig, RocmBackendConfig},
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::{
        Model,
        h3::{
            H3BlockCacheSpec, H3ConditionInput, H3Config, H3DenoiseLatents, H3GaussianNoise, H3PackedReferenceCondition, H3PreparedStage, H3QwenVisual, H3ReferenceKind, H3RocmUlyssesGroup, VISUAL_CONDITION_TIMESTEP,
            build_h3_reference_layout, decode_audio_vae, decode_video_vae_tiled_temporal, denoise_staged_conditioned, denoise_streamed_conditioned, denoise_ulysses_conditioned, h3_qwen_text_condition_resident, pack_audio, patchify_video,
            prepare_audio_vae, prepare_cached_dit_blocks, prepare_dit_final, prepare_dit_global, prepare_h3_audio_encoder, prepare_h3_video_encoder, prepare_video_vae, unpatchify_video, video_vae_latent_frames,
        },
        qwen3_vl::{Qwen3Vl, Qwen3VlImageProcessor, prepare_qwen3_vl_text_layer, qwen3_vl_encode_image, qwen3_vl_encode_video},
        session::NodeCapabilities,
    },
    server::node::{DynError, GeneratedArtifact, NodeEngine},
    server::scheduler::TaskProgress,
    tokenizer::Tokenizer,
    vae::{H3AudioVaeSpec, H3VideoVaeSpec},
    vision::ImageProcessor,
    weight::{
        model::h3::H3DitSource,
        model::h3_vae::{H3AudioVaeSource, H3VideoVaeSource},
        model::qwen3_vl::Qwen3VlWeights,
    },
};

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
type H3TaskRunner = Box<dyn FnMut(&H3ControlMessage, usize, usize, usize, &Path, &AtomicBool, &mut dyn FnMut(TaskProgress)) -> Result<Vec<GeneratedArtifact>, String>>;

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub struct H3Engine {
    capabilities: NodeCapabilities,
    run_task: H3TaskRunner,
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
#[derive(Deserialize, Serialize)]
struct H3ControlMessage {
    model: String,
    content: Vec<H3Content>,
    resolution: String,
    duration: u64,
    #[serde(default = "adaptive_ratio")]
    ratio: String,
    seed: Option<u64>,
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum H3Content {
    Text { text: String },
    ImageUrl { image_url: H3Url, role: Option<String> },
    VideoUrl { video_url: H3Url, role: Option<String> },
    AudioUrl { audio_url: H3Url, role: Option<String> },
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
#[derive(Deserialize, Serialize)]
struct H3Url {
    url: String,
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn adaptive_ratio() -> String {
    "adaptive".to_owned()
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
impl H3ControlMessage {
    fn validate(&self) -> Result<(), String> {
        if self.model != "MiniMax-H3" {
            return Err(format!("H3 model 必须是 MiniMax-H3，实际 {}", self.model));
        }
        if !(4..=15).contains(&self.duration) {
            return Err(format!("H3 duration={} 不在 4..=15", self.duration));
        }
        if !matches!(self.resolution.as_str(), "512P" | "768P" | "1080P" | "2K") {
            return Err(format!("H3 resolution 不支持 {}", self.resolution));
        }
        if !matches!(self.ratio.as_str(), "adaptive" | "21:9" | "16:9" | "4:3" | "1:1" | "3:4" | "9:16") {
            return Err(format!("H3 ratio 不支持 {}", self.ratio));
        }
        let mut text = 0usize;
        let mut keyframes = 0usize;
        let mut references = 0usize;
        let mut first = 0usize;
        let mut last = 0usize;
        let mut reference_images = 0usize;
        let mut reference_videos = 0usize;
        let mut reference_audio = 0usize;
        for item in &self.content {
            match item {
                H3Content::Text { text: value } => {
                    if value.trim().is_empty() || value.chars().count() > 7000 {
                        return Err("H3 text 必须为 1..=7000 字符".to_owned());
                    }
                    text += 1;
                }
                H3Content::ImageUrl { image_url, role } => {
                    validate_media_url(&image_url.url)?;
                    match role.as_deref().unwrap_or("first_frame") {
                        "first_frame" => {
                            first += 1;
                            keyframes += 1;
                        }
                        "last_frame" => {
                            last += 1;
                            keyframes += 1;
                        }
                        "reference_image" => {
                            reference_images += 1;
                            references += 1;
                        }
                        role => return Err(format!("H3 image role 不支持 {role}")),
                    }
                }
                H3Content::VideoUrl { video_url, role } => {
                    validate_media_url(&video_url.url)?;
                    if role.as_deref() != Some("reference_video") {
                        return Err("H3 video role 必须是 reference_video".to_owned());
                    }
                    reference_videos += 1;
                    references += 1;
                }
                H3Content::AudioUrl { audio_url, role } => {
                    validate_media_url(&audio_url.url)?;
                    if role.as_deref() != Some("reference_audio") {
                        return Err("H3 audio role 必须是 reference_audio".to_owned());
                    }
                    reference_audio += 1;
                    references += 1;
                }
            }
        }
        if text == 0 {
            return Err("H3 content 必须包含非空 text".to_owned());
        }
        if keyframes > 0 && references > 0 {
            return Err("H3 首尾帧与 reference 输入不能混用".to_owned());
        }
        if first > 1 || last > 1 || (last == 1 && first == 0) {
            return Err("H3 first_frame/last_frame 组合无效".to_owned());
        }
        if reference_images > 9 || reference_videos > 3 || reference_audio > 3 {
            return Err("H3 reference 数量超过限制".to_owned());
        }
        if reference_audio > 0 && reference_images + reference_videos == 0 {
            return Err("H3 reference_audio 不能单独使用".to_owned());
        }
        if !self.has_media() && self.ratio == "adaptive" {
            return Err("H3 纯文本生成必须指定非 adaptive ratio".to_owned());
        }
        Ok(())
    }

    fn prompt(&self) -> String {
        self.content
            .iter()
            .filter_map(|item| match item {
                H3Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn has_media(&self) -> bool {
        self.content.iter().any(|item| !matches!(item, H3Content::Text { .. }))
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn validate_media_url(url: &str) -> Result<(), String> {
    let raw_base64 = url.len() >= 4 && url.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte.is_ascii_whitespace() || matches!(byte, b'+' | b'/' | b'-' | b'_' | b'='));
    if url.starts_with("https://") || url.starts_with("http://") || url.starts_with("data:") || url.starts_with("mm_file://") || raw_base64 {
        Ok(())
    } else {
        Err("H3 媒体必须是 http(s) URL、data URI、原始 base64 或 mm_file://".to_owned())
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
impl H3Engine {
    pub fn load(model_root: &Path, qwen_root: &Path, qwen_tokenizer_root: &Path, backend: &RocmBackendConfig, execution: H3NodeExecutionConfig) -> Result<Self, DynError> {
        if !model_root.is_dir() {
            return Err(format!("H3 model 目录不存在: {}", model_root.display()).into());
        }
        if !qwen_root.is_dir() {
            return Err(format!("H3 Qwen 编码器目录不存在: {}", qwen_root.display()).into());
        }
        if !qwen_tokenizer_root.is_dir() {
            return Err(format!("H3 Qwen tokenizer 目录不存在: {}", qwen_tokenizer_root.display()).into());
        }
        let devices = backend.devices.clone();
        let run_task = build_runner(model_root, qwen_root, qwen_tokenizer_root, devices.clone(), backend.allow_cpu_reference_fallback, execution).map_err(|error| -> DynError { error.into() })?;
        Ok(Self {
            capabilities: NodeCapabilities {
                backend: "rocm".to_owned(),
                platform: std::env::consts::OS.to_owned(),
                architecture: std::env::consts::ARCH.to_owned(),
                accelerator: backend.accelerator_name.clone().unwrap_or_else(|| format!("ROCm Ulysses devices {devices:?}")),
                compute_units: backend.compute_units,
                compute_unit_kind: "compute_unit".to_owned(),
                memory_kind: "dedicated".to_owned(),
                unified_memory: false,
                system_memory_bytes: None,
                accelerator_memory_bytes: backend.accelerator_memory_bytes,
                recommended_working_set_bytes: backend.recommended_working_set_bytes,
                model_format: "safetensors".to_owned(),
                model_bytes: directory_bytes(model_root).saturating_add(directory_bytes(qwen_root)),
                max_seq_len: 0,
                kv_cache_format: "none".to_owned(),
                kv_cache_devices: Vec::new(),
                kv_reservation_page_tokens: 0,
                task_kinds: vec!["video_generation".to_owned()],
                input_modalities: vec!["text".to_owned(), "image".to_owned(), "video".to_owned(), "audio".to_owned()],
                output_modalities: vec!["video".to_owned(), "audio".to_owned()],
                artifact_streaming: true,
            },
            run_task,
        })
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
impl NodeEngine for H3Engine {
    fn model_key(&self) -> &'static str {
        "MiniMax-H3"
    }

    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        (self.capabilities.clone(), Arc::new(|| 0))
    }

    fn refresh_runtime(&self) {}

    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        Vec::new()
    }

    fn max_concurrency(&self) -> usize {
        1
    }

    fn generate_batch(
        &mut self,
        requests: Vec<crate::server::node::NodeBatchRequest>,
        _intake: &mut dyn FnMut(usize) -> Vec<crate::server::node::NodeBatchRequest>,
        _on_token: &mut dyn FnMut(&str, u32, String) -> bool,
        _on_tool_call_delta: &mut dyn FnMut(&str, crate::runtime::session::ToolCallDelta) -> bool,
        _on_result: &mut dyn FnMut(crate::server::node::NodeBatchResult),
    ) -> Vec<crate::server::node::NodeBatchResult> {
        requests
            .into_iter()
            .map(|request| {
                let request_id = request.request_id;
                let result = Err("MiniMax-H3 节点不执行 text_generation".to_owned());
                crate::server::node::NodeBatchResult { request_id, result }
            })
            .collect()
    }

    fn execute_task(&mut self, task_kind: &str, request: &Value, output_dir: &Path, cancellation: &AtomicBool, on_progress: &mut dyn FnMut(TaskProgress)) -> Result<Vec<GeneratedArtifact>, String> {
        if task_kind != "video_generation" {
            return Err(format!("MiniMax-H3 不支持任务 {task_kind}"));
        }
        let control: H3ControlMessage = serde_json::from_value(request.clone()).map_err(|error| format!("解析 H3 control message: {error}"))?;
        control.validate()?;
        let duration = control.duration;
        let resolution = control.resolution.as_str();
        let ratio = if control.ratio == "adaptive" { adaptive_ratio_for_media(&control)? } else { control.ratio.clone() };
        let (width, height) = canvas(resolution, &ratio)?;
        let target_frames = duration.checked_mul(24).ok_or("H3 duration 溢出")?;
        // 帧数必须满足 video_vae_latent_frames 的 (frames-5) % temporal_clip_length == 0 约束；
        // 17 取自 spec（参照 video_vae.rs），5 是官方因果 VAE 的首 clip 基准帧数，spec 无对应字段。
        let clip_length = u64::try_from(H3VideoVaeSpec::standard().temporal_clip_length).map_err(|_| "H3 temporal clip length 超过 u64")?;
        let frames = target_frames.saturating_sub(5).div_ceil(clip_length).saturating_mul(clip_length).saturating_add(5);
        let frames = usize::try_from(frames).map_err(|_| "H3 frames 超过 usize")?;
        (self.run_task)(&control, width, height, frames, output_dir, cancellation, on_progress)
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn adaptive_ratio_for_media(control: &H3ControlMessage) -> Result<String, String> {
    let mut dimensions = None;
    for item in &control.content {
        let (url, media_type) = match item {
            H3Content::ImageUrl { image_url, .. } => (&image_url.url, "image/*"),
            H3Content::VideoUrl { video_url, .. } => (&video_url.url, "video/*"),
            _ => continue,
        };
        dimensions = media::reference_url_dimensions(url, media_type)?;
        break;
    }
    let (width, height) = dimensions.ok_or("H3 adaptive ratio 需要图片或视频参考")?;
    let ratio = width as f64 / height as f64;
    let candidates: [(&str, f64); 6] = [("21:9", 21.0 / 9.0), ("16:9", 16.0 / 9.0), ("4:3", 4.0 / 3.0), ("1:1", 1.0), ("3:4", 3.0 / 4.0), ("9:16", 9.0 / 16.0)];
    Ok(candidates.into_iter().min_by(|(_, left), (_, right)| (ratio.ln() - left.ln()).abs().total_cmp(&(ratio.ln() - right.ln()).abs())).expect("adaptive ratio candidates 非空").0.to_owned())
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
struct H3DenoiseCheckpoint {
    completed_steps: usize,
    video: Vec<f32>,
    audio: Vec<f32>,
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn write_h3_denoise_checkpoint(path: &Path, completed_steps: usize, total_steps: usize, video_shape: (usize, usize), audio_shape: (usize, usize), video: &[f32], audio: &[f32]) -> Result<(), String> {
    const MAGIC: &[u8; 8] = b"ZLH3CP01";
    let video_elements = video_shape.0.checked_mul(video_shape.1).ok_or("H3 checkpoint video shape 溢出")?;
    let audio_elements = audio_shape.0.checked_mul(audio_shape.1).ok_or("H3 checkpoint audio shape 溢出")?;
    if video.len() != video_elements || audio.len() != audio_elements {
        return Err(format!("H3 checkpoint latent 大小无效: video={}/{} audio={}/{}", video.len(), video_elements, audio.len(), audio_elements));
    }
    let temporary = path.with_extension("bin.tmp");
    let mut output = BufWriter::new(File::create(&temporary).map_err(|error| format!("创建 {}: {error}", temporary.display()))?);
    output.write_all(MAGIC).map_err(|error| format!("写入 {}: {error}", temporary.display()))?;
    for value in [completed_steps, total_steps, video_shape.0, video_shape.1, audio_shape.0, audio_shape.1] {
        let value = u64::try_from(value).map_err(|_| "H3 checkpoint usize 超过 u64")?;
        output.write_all(&value.to_le_bytes()).map_err(|error| format!("写入 {}: {error}", temporary.display()))?;
    }
    for value in video.iter().chain(audio) {
        output.write_all(&value.to_le_bytes()).map_err(|error| format!("写入 {}: {error}", temporary.display()))?;
    }
    output.flush().map_err(|error| format!("刷新 {}: {error}", temporary.display()))?;
    output.get_ref().sync_all().map_err(|error| format!("同步 {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|error| format!("提交 {}: {error}", path.display()))
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn load_h3_denoise_checkpoint(path: &Path, total_steps: usize, video_shape: (usize, usize), audio_shape: (usize, usize)) -> Result<Option<H3DenoiseCheckpoint>, String> {
    const MAGIC: &[u8; 8] = b"ZLH3CP01";
    let mut input = match File::open(path) {
        Ok(input) => input,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("打开 {}: {error}", path.display())),
    };
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic).map_err(|error| format!("读取 {} magic: {error}", path.display()))?;
    if &magic != MAGIC {
        return Err(format!("{} 不是 H3 denoise checkpoint", path.display()));
    }
    let mut read_usize = || -> Result<usize, String> {
        let mut bytes = [0u8; 8];
        input.read_exact(&mut bytes).map_err(|error| format!("读取 {} header: {error}", path.display()))?;
        usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| format!("{} header 超过 usize", path.display()))
    };
    let completed_steps = read_usize()?;
    let stored_total = read_usize()?;
    let stored_video_shape = (read_usize()?, read_usize()?);
    let stored_audio_shape = (read_usize()?, read_usize()?);
    if stored_total != total_steps || completed_steps > total_steps || stored_video_shape != video_shape || stored_audio_shape != audio_shape {
        return Err(format!("H3 checkpoint 不匹配: step={completed_steps}/{stored_total} video={stored_video_shape:?} audio={stored_audio_shape:?}，期望 total={total_steps} video={video_shape:?} audio={audio_shape:?}",));
    }
    let video_elements = video_shape.0.checked_mul(video_shape.1).ok_or("H3 checkpoint video shape 溢出")?;
    let audio_elements = audio_shape.0.checked_mul(audio_shape.1).ok_or("H3 checkpoint audio shape 溢出")?;
    let payload_elements = video_elements.checked_add(audio_elements).ok_or("H3 checkpoint payload 溢出")?;
    let expected_bytes = 56usize.checked_add(payload_elements.checked_mul(4).ok_or("H3 checkpoint payload bytes 溢出")?).ok_or("H3 checkpoint file bytes 溢出")?;
    let actual_bytes = usize::try_from(input.metadata().map_err(|error| format!("读取 {} metadata: {error}", path.display()))?.len()).map_err(|_| "H3 checkpoint 文件过大")?;
    if actual_bytes != expected_bytes {
        return Err(format!("H3 checkpoint bytes={actual_bytes}，期望 {expected_bytes}"));
    }
    let mut bytes = vec![0u8; payload_elements * 4];
    input.read_exact(&mut bytes).map_err(|error| format!("读取 {} payload: {error}", path.display()))?;
    let mut values = bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().expect("checkpoint chunk 固定 4 bytes")));
    let video = values.by_ref().take(video_elements).collect();
    let audio = values.collect();
    Ok(Some(H3DenoiseCheckpoint { completed_steps, video, audio }))
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn build_runner(model_root: &Path, qwen_root: &Path, qwen_tokenizer_root: &Path, devices: Vec<i32>, allow_cpu_reference_fallback: bool, execution: H3NodeExecutionConfig) -> Result<H3TaskRunner, String> {
    // ggml/BLAS 后端在进程启动时读取这些变量决定线程数；节点进程与调度器同进程，
    // 多线程 BLAS 会与 ROCm 单流执行争用 CPU，必须在创建任何 backend 之前做进程级设置。
    // 这是有意的进程级副作用：该 runner 独占整个 node 进程。
    unsafe {
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("OMP_DYNAMIC", "FALSE");
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    }
    let config = H3Config::standard();
    let group = H3RocmUlyssesGroup::new(&devices, allow_cpu_reference_fallback)?;
    let qwen_model = Qwen3Vl::instruct_32b();
    let qwen_config = qwen_model.config().clone();
    // MiniMax-H3 使用 hidden_states[50]（embedding 后执行 50 层），且不经过最终 RMSNorm。
    const H3_TEXT_ENCODER_LAYERS: usize = 50;
    let text_encoder_layers = H3_TEXT_ENCODER_LAYERS;
    if qwen_config.layer_count < text_encoder_layers {
        return Err(format!("H3 文本编码器只有 {} 层，无法取得 hidden_states[{text_encoder_layers}]", qwen_config.layer_count));
    }
    let qwen_vision = qwen_config.vision.clone().ok_or_else(|| "H3 node 需要 Qwen3-VL vision 配置".to_owned())?;
    let qwen_processor = Qwen3VlImageProcessor::new(&qwen_vision)?;
    let tokenizer = Tokenizer::new(qwen_tokenizer_root.join("tokenizer.json")).map_err(|error| format!("打开 H3 Qwen tokenizer: {error}"))?;
    let qwen_weights = Qwen3VlWeights::open(qwen_root, qwen_config.clone())?;
    let source = H3DitSource::open(model_root, config.clone())?;
    let started = Instant::now();
    let block_sources = crate::runtime::h3::HostLayerCache::load(config.num_layers, |layer| source.load_block(layer))?;
    eprintln!("[h3-node] DiT host layer cache resident layers={} wall={:.3}s", block_sources.len(), started.elapsed().as_secs_f64());
    let global_source = source.load_global()?;
    let video_spec = H3VideoVaeSpec::standard();
    let video_source = H3VideoVaeSource::open(model_root)?;
    let audio_spec = H3AudioVaeSpec::standard();
    let audio_source = H3AudioVaeSource::open(model_root)?;
    let steps = execution.steps;
    if steps < 2 {
        return Err("H3 execution.steps 必须 >= 2".to_owned());
    }
    let stream_chunk_layers = execution.stream_chunk_layers;
    if stream_chunk_layers == 0 || stream_chunk_layers > config.num_layers {
        return Err(format!("H3 execution.stream_chunk_layers={stream_chunk_layers} 必须在 1..={} 范围内", config.num_layers));
    }
    if devices.len() > 1 && stream_chunk_layers != config.num_layers {
        return Err(format!("H3 Ulysses 要求 stream_chunk_layers={} 使 50 层在每个 rank 全驻留，实际 {stream_chunk_layers}", config.num_layers));
    }
    let block_cache =
        execution.block_cache.map(|cache| H3BlockCacheSpec { threshold: cache.threshold, start_percent: cache.start_percent, end_percent: cache.end_percent, max_consecutive_hits: cache.max_consecutive_hits }.validate()).transpose()?;
    if block_cache.is_some() && devices.len() == 1 {
        return Err("H3 block cache 当前只接入 ROCm Ulysses 多卡路径，单卡配置必须关闭".to_owned());
    }
    let mut reference_cache = None;
    Ok(Box::new(move |control, width, height, frames, output_dir, cancellation, on_progress| {
        group.activate_all().map_err(|error| format!("激活 H3 ROCm streams: {error:?}"))?;
        let context = group.primary();
        let task_started = Instant::now();
        let mut report = |phase: &str, completed: usize, total: usize, phase_eta_seconds: Option<f64>| {
            on_progress(TaskProgress { phase: phase.to_owned(), completed, total, elapsed_seconds: task_started.elapsed().as_secs_f64(), phase_eta_seconds });
        };
        report("conditioning", 0, 1, None);
        if cancellation.load(Ordering::Acquire) {
            return Err("H3 任务已取消".to_owned());
        }
        let seed = control.seed.unwrap_or(42);
        // 媒体 content 内联完整 base64，可能上百 MB，不能直接作为 cache key 持有；
        // key 只用于本进程单槽命中判断，对序列化内容取 blake3 摘要（已是仓库依赖）。
        let reference_material = serde_json::to_vec(&(seed, control.content.iter().filter(|item| !matches!(item, H3Content::Text { .. })).collect::<Vec<_>>())).map_err(|error| format!("序列化 H3 reference cache key: {error}"))?;
        let reference_key = blake3::hash(&reference_material);
        let reference_count = control.content.iter().filter(|item| !matches!(item, H3Content::Text { .. })).count();
        let started = Instant::now();
        let cache_hit = reference_cache.as_ref().is_some_and(|(key, _, _)| key == &reference_key);
        if !cache_hit {
            reference_cache = None;
            let media = media::prepare_conditioning_media(media::decode_control_references(control)?)?;
            let needs_video_encoder = media.iter().any(|item| matches!(item, media::ConditioningMedia::Image { .. } | media::ConditioningMedia::Video { .. }));
            let needs_audio_encoder = media.iter().any(|item| matches!(item, media::ConditioningMedia::Audio { .. } | media::ConditioningMedia::Video { audio: Some(_), .. }));
            let (qwen_visuals, references) = {
                let encoder_started = Instant::now();
                let video_encoder = if needs_video_encoder {
                    let item_started = Instant::now();
                    let encoder = prepare_h3_video_encoder(context, &video_source, &video_spec).map_err(|error| format!("准备 GPU H3 video encoder: {error:?}"))?;
                    eprintln!("[h3-node] video encoder prepare wall={:.3}s", item_started.elapsed().as_secs_f64());
                    Some(encoder)
                } else {
                    None
                };
                let audio_encoder = if needs_audio_encoder {
                    let item_started = Instant::now();
                    let encoder = prepare_h3_audio_encoder(context, &audio_source, &audio_spec).map_err(|error| format!("准备 GPU H3 audio encoder: {error:?}"))?;
                    eprintln!("[h3-node] audio encoder prepare wall={:.3}s", item_started.elapsed().as_secs_f64());
                    Some(encoder)
                } else {
                    None
                };
                eprintln!("[h3-node] reference encoders video={needs_video_encoder} audio={needs_audio_encoder} wall={:.3}s", encoder_started.elapsed().as_secs_f64());
                let encoded = media::encode_conditioning_media(
                    context,
                    video_encoder.as_ref(),
                    audio_encoder.as_ref(),
                    media,
                    seed,
                    VISUAL_CONDITION_TIMESTEP,
                    |image| {
                        let image = qwen_processor.preprocess(image)?;
                        let grid = image.grid;
                        let merge_size = image.merge_size;
                        let qwen_started = Instant::now();
                        let output = qwen3_vl_encode_image(context, &qwen_vision, &qwen_weights, &image).map_err(|error| format!("Qwen GPU reference image: {error:?}"))?;
                        eprintln!("[h3-node] Qwen reference image wall={:.3}s", qwen_started.elapsed().as_secs_f64());
                        H3QwenVisual::new(output, H3ReferenceKind::Image, grid, merge_size).map_err(|error| format!("Qwen reference image output: {error:?}"))
                    },
                    |frames, _timestamps| {
                        let video = qwen_processor.preprocess_video(frames)?;
                        let grid = video.grid;
                        let merge_size = video.merge_size;
                        let output = qwen3_vl_encode_video(context, &qwen_vision, &qwen_weights, &video).map_err(|error| format!("Qwen GPU reference video: {error:?}"))?;
                        H3QwenVisual::new(output, H3ReferenceKind::Video, grid, merge_size).map_err(|error| format!("Qwen reference video output: {error:?}"))
                    },
                )?;
                context.finish_batch();
                drop(video_encoder);
                drop(audio_encoder);
                context.finish_batch();
                encoded
            };
            reference_cache = Some((reference_key, qwen_visuals, references));
        }
        let (_, qwen_visuals, references) = reference_cache.as_ref().ok_or("H3 reference cache 未初始化")?;
        let reference_condition_count = references.conditions.len();
        eprintln!(
            "[h3-node] references={reference_count} cache={} qwen_gpu_visuals={} ref2va_gpu_conditions={reference_condition_count} wall={:.3}s",
            if cache_hit { "hit" } else { "miss" },
            qwen_visuals.len(),
            started.elapsed().as_secs_f64()
        );
        let prompt = control.prompt();
        let started = Instant::now();
        let prepare_started = Instant::now();
        let qwen_text_layers = (0..text_encoder_layers).map(|layer| prepare_qwen3_vl_text_layer(context, &qwen_weights, layer).map_err(|error| format!("准备 Qwen GPU L{layer}: {error:?}"))).collect::<Result<Vec<_>, _>>()?;
        eprintln!("[h3-node] Qwen text weights prepare wall={:.3}s", prepare_started.elapsed().as_secs_f64());
        let execute_started = Instant::now();
        let text_condition =
            h3_qwen_text_condition_resident(context, &qwen_config, &qwen_weights, &tokenizer, &prompt, qwen_visuals, text_encoder_layers, &qwen_text_layers).map_err(|error| format!("Qwen GPU H3 multimodal hidden: {error:?}"))?;
        eprintln!("[h3-node] Qwen text execute wall={:.3}s", execute_started.elapsed().as_secs_f64());
        let token_count = text_condition.token_ids.len();
        let text = text_condition.hidden;
        context.finish_batch();
        drop(qwen_text_layers);
        context.finish_batch();
        eprintln!("[h3-node] Qwen GPU tokens={token_count} layers={} weights_released wall={:.3}s", text_encoder_layers, started.elapsed().as_secs_f64());
        report("conditioning", 1, 1, Some(0.0));
        if cancellation.load(Ordering::Acquire) {
            return Err("H3 任务已取消".to_owned());
        }

        let started = Instant::now();
        let globals = group.contexts().iter().map(|rank| prepare_dit_global(rank, &global_source)).collect::<Result<Vec<_>, _>>().map_err(|error| format!("准备 H3 rank globals: {error:?}"))?;
        let final_weights = prepare_dit_final(context, &source).map_err(|error| format!("准备 H3 final layer: {error:?}"))?;
        let resident_blocks = if stream_chunk_layers == config.num_layers {
            Some(group.contexts().iter().map(|rank| prepare_cached_dit_blocks(rank, &block_sources)).collect::<Result<Vec<_>, _>>().map_err(|error| format!("准备 H3 rank resident blocks: {error:?}"))?)
        } else {
            None
        };
        eprintln!(
            "[h3-node] DiT prepared ranks={} resident_layers_per_rank={} chunk_layers={stream_chunk_layers} wall={:.3}s",
            group.contexts().len(),
            resident_blocks.as_ref().and_then(|ranks| ranks.first()).map_or(0, Vec::len),
            started.elapsed().as_secs_f64()
        );

        let latent_t = video_vae_latent_frames(frames, &video_spec)?;
        let latent_h = height / 16;
        let latent_w = width / 16;
        let audio_t = (frames * 40).div_ceil(24);
        let patch = config.patch_size;
        let layout = build_h3_reference_layout(context.token_rows(&text), latent_t, latent_h, latent_w, audio_t, patch, &references)?;
        let conditions = references
            .conditions
            .iter()
            .map(|condition| match condition {
                H3PackedReferenceCondition::Video { condition, .. } => H3ConditionInput::Video(&condition.tensor),
                H3PackedReferenceCondition::Audio { condition, .. } => H3ConditionInput::Audio(&condition.tensor),
            })
            .collect::<Vec<_>>();
        let video_shape = [1, config.video_latent_channels, latent_t, latent_h, latent_w];
        let checkpoint_path = output_dir.join("denoise-checkpoint.bin");
        let total_steps = steps;
        let checkpoint = load_h3_denoise_checkpoint(&checkpoint_path, total_steps, (layout.video_rows.len(), config.video_patch_dim()), (layout.audio_rows.len(), config.audio_latent_channels))?;
        let (video, audio, start_step) = if let Some(checkpoint) = checkpoint {
            let video = context.tensor_from_f32(checkpoint.video, layout.video_rows.len(), config.video_patch_dim()).map_err(|error| format!("恢复 H3 video latent: {error:?}"))?;
            let audio = context.tensor_from_f32(checkpoint.audio, layout.audio_rows.len(), config.audio_latent_channels).map_err(|error| format!("恢复 H3 audio latent: {error:?}"))?;
            eprintln!("[h3-node] denoise checkpoint 恢复 step={}/{} path={}", checkpoint.completed_steps, total_steps, checkpoint_path.display());
            (video, audio, checkpoint.completed_steps)
        } else {
            let mut random = H3GaussianNoise::new(seed);
            let video_rows = patchify_video(&random.values(video_shape.iter().product()), video_shape, patch)?;
            let audio_rows = pack_audio(&random.values(2 * config.audio_latent_channels * audio_t), config.audio_latent_channels, 2, audio_t)?;
            let video = context.tensor_from_f32(video_rows, layout.video_rows.len(), config.video_patch_dim()).map_err(|error| format!("上传 H3 video latent: {error:?}"))?;
            let audio = context.tensor_from_f32(audio_rows, layout.audio_rows.len(), config.audio_latent_channels).map_err(|error| format!("上传 H3 audio latent: {error:?}"))?;
            (video, audio, 0)
        };
        let denoise_started = Instant::now();
        report("denoise", start_step, total_steps, None);
        let mut report_step = |completed: usize, total: usize, latents: &H3DenoiseLatents<RocmTensor>| -> Result<bool, BackendError> {
            let checkpoint_started = Instant::now();
            let video = context.tensor_to_f32(&latents.video)?;
            let audio = context.tensor_to_f32(&latents.audio)?;
            write_h3_denoise_checkpoint(&checkpoint_path, completed, total, (latents.video.rows, latents.video.cols), (latents.audio.rows, latents.audio.cols), &video, &audio).map_err(|msg| BackendError::Compute { msg })?;
            eprintln!("[h3-node] denoise checkpoint step={completed}/{total} wall={:.3}s", checkpoint_started.elapsed().as_secs_f64());
            let completed_this_run = completed - start_step;
            let average = denoise_started.elapsed().as_secs_f64() / completed_this_run as f64;
            report("denoise", completed, total, Some(average * (total - completed) as f64));
            Ok(!cancellation.load(Ordering::Acquire))
        };
        let latents = if group.contexts().len() > 1 {
            denoise_ulysses_conditioned(
                &group,
                &source,
                &globals,
                resident_blocks.as_ref().expect("H3 Ulysses resident blocks 已在配置期强制"),
                &final_weights,
                &text,
                &conditions,
                H3DenoiseLatents { video, audio },
                &layout,
                steps,
                start_step,
                block_cache,
                &mut report_step,
            )
        } else if let Some(blocks) = resident_blocks.as_ref() {
            let stages = [H3PreparedStage { backend: context, global: &globals[0], first_layer: 0, blocks: &blocks[0] }];
            denoise_staged_conditioned(context, &source, &globals[0], &stages, &final_weights, &text, &conditions, H3DenoiseLatents { video, audio }, &layout, steps, start_step, &mut report_step)
        } else {
            let mut streamed_blocks = crate::runtime::h3::HostStreamedLayers::new(config.num_layers, |layer| source.load_block(layer));
            denoise_streamed_conditioned(context, &source, &globals[0], &mut streamed_blocks, stream_chunk_layers, &final_weights, &text, &conditions, H3DenoiseLatents { video, audio }, &layout, steps, start_step, &mut report_step)
        }
        .map_err(|error| format!("H3 denoise: {error:?}"))?;
        drop(report_step);
        eprintln!("[h3-node] denoise wall={:.3}s", denoise_started.elapsed().as_secs_f64());
        drop(conditions);
        drop(text);
        context.finish_batch();
        drop(resident_blocks);
        drop(final_weights);
        drop(globals);
        context.finish_batch();
        eprintln!("[h3-node] denoise temporary buffers and streamed DiT weights released");
        if cancellation.load(Ordering::Acquire) {
            return Err("H3 任务已取消".to_owned());
        }

        let H3DenoiseLatents { video, audio } = latents;
        let video_rows = context.tensor_to_f32(&video).map_err(|error| format!("读取 H3 video latent: {error:?}"))?;
        validate_finite("去噪 video latent", &video_rows, Some(config.video_patch_dim()))?;
        let audio_shape = (audio.rows, audio.cols);
        let audio_rows = context.tensor_to_f32(&audio).map_err(|error| format!("读取 H3 audio latent: {error:?}"))?;
        drop(video);
        drop(audio);
        context.finish_batch();
        if execution.save_latent {
            let latent_path = output_dir.join("video-latent.f32");
            let mut output = BufWriter::new(File::create(&latent_path).map_err(|error| format!("创建 {}: {error}", latent_path.display()))?);
            for value in &video_rows {
                output.write_all(&value.to_le_bytes()).map_err(|error| format!("写入 {}: {error}", latent_path.display()))?;
            }
            output.flush().map_err(|error| format!("刷新 {}: {error}", latent_path.display()))?;
        }
        let video_latent = unpatchify_video(&video_rows, [1, config.video_latent_channels, latent_t, latent_h, latent_w], patch)?;
        report("video_vae", 0, 1, None);
        let decoder_started = Instant::now();
        let video_global = prepare_video_vae(context, &video_source).map_err(|error| format!("准备 H3 video VAE: {error:?}"))?;
        let audio_global = prepare_audio_vae(context, &audio_source).map_err(|error| format!("准备 H3 audio VAE: {error:?}"))?;
        let audio = context.tensor_from_f32(audio_rows, audio_shape.0, audio_shape.1).map_err(|error| format!("恢复 H3 audio latent: {error}"))?;
        eprintln!("[h3-node] decoders prepared wall={:.3}s", decoder_started.elapsed().as_secs_f64());
        let started = Instant::now();
        let decode_started = Instant::now();
        let video_values = decode_video_vae_tiled_temporal(context, &video_source, &video_global, &video_latent, latent_t, latent_h, latent_w, frames, patch)?;
        validate_finite("Video VAE 输出", &video_values, None)?;
        eprintln!("[h3-node] Video VAE decode wall={:.3}s", decode_started.elapsed().as_secs_f64());
        report("video_vae", 1, 1, Some(0.0));
        let decoded_height = latent_h * video_spec.spatial_compression;
        let decoded_width = latent_w * video_spec.spatial_compression;
        let y4m = output_dir.join("video.y4m");
        let y4m_started = Instant::now();
        write_y4m(&y4m, &video_values, frames, frames, decoded_height, decoded_width)?;
        eprintln!("[h3-node] Y4M encode wall={:.3}s", y4m_started.elapsed().as_secs_f64());
        eprintln!("[h3-node] video VAE+Y4M wall={:.3}s", started.elapsed().as_secs_f64());

        report("audio_vae", 0, 1, None);
        let started = Instant::now();
        let samples = decode_audio_vae(context, &audio_source, &audio_global, &audio, audio_t).map_err(|error| format!("H3 audio VAE: {error:?}"))?;
        let samples = context.tensor_to_f32(&samples).map_err(|error| format!("读取 H3 audio samples: {error:?}"))?;
        drop(audio);
        let available_samples = audio_t * audio_spec.sample_rate / audio_spec.latent_rate;
        let output_samples = frames.checked_mul(audio_spec.sample_rate).and_then(|value| value.checked_add(12)).map(|value| value / 24).ok_or("H3 audio sample 数溢出")?;
        let wav = output_dir.join("audio.wav");
        write_wav(&wav, &samples, audio_spec.output_channels, available_samples, output_samples, audio_spec.sample_rate)?;
        eprintln!("[h3-node] audio VAE+WAV wall={:.3}s", started.elapsed().as_secs_f64());
        report("audio_vae", 1, 1, Some(0.0));

        report("muxing", 0, 1, None);
        let mp4 = output_dir.join("output.mp4");
        let status = Command::new(&execution.ffmpeg)
            .args(["-y", "-i", y4m.to_string_lossy().as_ref(), "-i", wav.to_string_lossy().as_ref(), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", mp4.to_string_lossy().as_ref()])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .status()
            .map_err(|error| format!("启动 ffmpeg: {error}"))?;
        if !status.success() {
            return Err(format!("ffmpeg 退出状态 {status}"));
        }
        match fs::remove_file(&checkpoint_path) {
            Ok(()) => eprintln!("[h3-node] denoise checkpoint 已清理 path={}", checkpoint_path.display()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("清理 {}: {error}", checkpoint_path.display())),
        }
        report("completed", 1, 1, Some(0.0));
        drop(video_global);
        drop(audio_global);
        context.finish_batch();
        Ok(vec![GeneratedArtifact { id: "video".to_owned(), file_name: "output.mp4".to_owned(), content_type: "video/mp4".to_owned(), path: mp4 }])
    }))
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn validate_finite(label: &str, values: &[f32], columns: Option<usize>) -> Result<(), String> {
    let mut finite = 0usize;
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for &value in values {
        if value.is_finite() {
            finite += 1;
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
    }
    eprintln!("[h3-node] {label} finite={finite}/{} range=[{minimum:.6}, {maximum:.6}]", values.len());
    if let Some(columns) = columns {
        let mut bad_rows = 0usize;
        let mut full_rows = 0usize;
        let mut first = None;
        let mut bad_columns = vec![0usize; columns];
        for (row, row_values) in values.chunks(columns).enumerate() {
            let mut bad = 0usize;
            for (column, value) in row_values.iter().enumerate() {
                if !value.is_finite() {
                    first.get_or_insert((row, column));
                    bad_columns[column] += 1;
                    bad += 1;
                }
            }
            bad_rows += usize::from(bad != 0);
            full_rows += usize::from(bad == row_values.len());
        }
        let bad_columns = bad_columns.into_iter().enumerate().filter(|(_, count)| *count != 0).collect::<Vec<_>>();
        eprintln!("[h3-node] {label} first={first:?} bad_rows={bad_rows} full_rows={full_rows} bad_columns={bad_columns:?}");
    }

    if finite != values.len() {
        return Err(format!("{label} 包含 {} 个非有限值", values.len() - finite));
    }
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn canvas(resolution: &str, ratio: &str) -> Result<(usize, usize), String> {
    let short: usize = match resolution {
        "512P" => 544,
        "768P" => 768,
        "1080P" => 1088,
        "2K" => 1152,
        _ => return Err(format!("H3 resolution 不支持 {resolution}")),
    };
    let aligned_long = |numerator: usize, denominator: usize| short * numerator / denominator / 32 * 32;
    match ratio {
        "21:9" => Ok((aligned_long(7, 3), short)),
        "16:9" => Ok((aligned_long(16, 9), short)),
        "4:3" => Ok((aligned_long(4, 3), short)),
        "1:1" => Ok((short, short)),
        "3:4" => Ok((short, aligned_long(4, 3))),
        "9:16" => Ok((short, aligned_long(16, 9))),
        "adaptive" => Err("纯文本 H3 请求的 ratio 不能为 adaptive".to_owned()),
        _ => Err(format!("H3 ratio 不支持 {ratio}")),
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn directory_bytes(root: &Path) -> u64 {
    let mut total = 0u64;
    let mut pending = vec![root.to_owned()];
    while let Some(path) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(path) else { continue };
        for entry in entries.filter_map(Result::ok) {
            let Ok(metadata) = entry.metadata() else { continue };
            if metadata.is_dir() {
                pending.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}

mod media;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub async fn run(model: crate::config::H3NodeModelConfig, backend: crate::config::RocmBackendConfig, config: crate::server::node::NodeConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = model.weights_directory;
    let qwen_model = model.qwen_weights_directory;
    let qwen_tokenizer = model.qwen_tokenizer_directory.unwrap_or_else(|| qwen_model.clone());
    let execution = model.execution;
    let factory = Box::new(move |_runtime, _terminal_state| H3Engine::load(&model_path, &qwen_model, &qwen_tokenizer, &backend, execution.clone()).map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>));
    crate::server::node::run_node(config, factory).await
}
