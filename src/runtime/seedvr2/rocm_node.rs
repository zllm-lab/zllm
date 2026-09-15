//! SeedVR2 正式 ROCm 节点：原生 VAE/八卡 DiT、CPU 流式 Y4M→H.264。

use super::{
    SeedVr2Request, gaussian_noise,
    rocm_pipeline::SeedVr2RocmPipeline,
    rocm_vae,
    vae::{PreparedVae, VaeVideo},
};
use crate::{
    artifact::Y4mWriter,
    backend::{BackendError, DiffusionBackend, rocm::RocmWeight},
    config::{RocmBackendConfig, SeedVr2NodeModelConfig},
    runtime::session::NodeCapabilities,
    server::{
        node::{DynError, GeneratedArtifact, NodeEngine},
        scheduler::TaskProgress,
    },
    weight::model::seedvr2::SeedVr2VaeSource,
};
use serde_json::Value;
use std::{
    fs,
    io::Read,
    path::Path,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

pub struct SeedVr2Engine {
    pipeline: SeedVr2RocmPipeline,
    vae: Vec<PreparedVae<RocmWeight>>,
    batch_frames: usize,
    capabilities: NodeCapabilities,
}

impl SeedVr2Engine {
    pub fn load(model: &SeedVr2NodeModelConfig, backend: &RocmBackendConfig) -> Result<Self, String> {
        if backend.allow_cpu_reference_fallback {
            return Err("SeedVR2 禁止 CPU reference fallback".to_owned());
        }
        if backend.devices.len() != 8 {
            return Err("SeedVR2 正式视频路径要求 8 张卡：DiT 序列并行、VAE 八段流水线".to_owned());
        }
        let pipeline = SeedVr2RocmPipeline::load(&model.weights_directory, &backend.devices)?;
        pipeline.group.primary().activate()?;
        let source = SeedVr2VaeSource::open(model.weights_directory.join("ema_vae_fp16.safetensors"))?;
        let vae = rocm_vae::prepare(&pipeline.group, &source).map_err(|e| e.to_string())?;
        let capabilities = NodeCapabilities {
            backend: "rocm".to_owned(),
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            accelerator: format!("ROCm {} GPUs / {} heads per rank", backend.devices.len(), pipeline.config.num_heads / backend.devices.len()),
            compute_units: backend.compute_units,
            compute_unit_kind: "compute_unit".to_owned(),
            memory_kind: "dedicated".to_owned(),
            accelerator_memory_bytes: backend.accelerator_memory_bytes,
            recommended_working_set_bytes: backend.recommended_working_set_bytes,
            model_format: "safetensors".to_owned(),
            kv_cache_format: "none".to_owned(),
            task_kinds: vec!["video_super_resolution".to_owned()],
            input_modalities: vec!["video".to_owned()],
            output_modalities: vec!["video".to_owned()],
            ..NodeCapabilities::default()
        };
        Ok(Self { pipeline, vae, batch_frames: model.batch_frames, capabilities })
    }

    pub fn upscale(&self, request: &SeedVr2Request, output: &Path, cancellation: &AtomicBool, on_progress: &mut dyn FnMut(TaskProgress)) -> Result<(), String> {
        request.validate()?;
        if !request.input_video.is_file() {
            return Err(format!("SeedVR2 输入视频不存在: {}", request.input_video.display()));
        }
        if self.batch_frames < 5 || !(self.batch_frames - 1).is_multiple_of(4) {
            return Err("SeedVR2 batch_frames 必须为至少5的4n+1".to_owned());
        }
        let started = Instant::now();
        let mut reader = Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(&request.input_video)
            .args(["-an", "-vf"])
            .arg(format!("fps=24,scale={}:{}:flags=bicubic", request.width, request.height))
            .args(["-frames:v", &request.frames.to_string(), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("SeedVR2 ffmpeg读取: {e}"))?;
        let mut encoder = match Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-nostdin", "-f", "yuv4mpegpipe", "-i", "pipe:0", "-i"])
            .arg(&request.input_video)
            .args(["-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t"])
            .arg((request.frames as f64 / 24.0).to_string())
            .args(["-movflags", "+faststart"])
            .arg(output)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                let _ = reader.kill();
                let _ = reader.wait();
                return Err(format!("SeedVR2 ffmpeg编码: {e}"));
            }
        };
        let result = (|| {
            let mut input = reader.stdout.take().ok_or("SeedVR2 reader stdout缺失")?;
            let stdin = encoder.stdin.take().ok_or("SeedVR2 encoder stdin缺失")?;
            let mut writer = Y4mWriter::with_rgb_transform(stdin, request.height, request.width, [0.5; 3], [0.5; 3])?;
            let pixels = request.width * request.height;
            let frame_bytes = pixels * 3;
            let mut read_frames = 0;
            let mut written_frames = 0;
            let mut previous: Option<Vec<u8>> = None;
            let batches = (request.frames.saturating_sub(1)).div_ceil(self.batch_frames - 1).max(1);
            let context = self.pipeline.group.primary();
            for batch in 0..batches {
                if cancellation.load(Ordering::Relaxed) {
                    return Err("SeedVR2 任务已取消".to_owned());
                }
                let carry = usize::from(previous.is_some());
                let new_frames = (self.batch_frames - carry).min(request.frames - read_frames);
                if new_frames == 0 {
                    break;
                }
                let actual = carry + new_frames;
                let padded = (actual - 1).div_ceil(4) * 4 + 1;
                let mut rgb = Vec::with_capacity(actual * frame_bytes);
                if let Some(previous) = previous.take() {
                    rgb.extend_from_slice(&previous);
                }
                rgb.resize(actual * frame_bytes, 0);
                input.read_exact(&mut rgb[carry * frame_bytes..]).map_err(|e| format!("SeedVR2 读取帧 {read_frames}..{}: {e}", read_frames + new_frames))?;
                previous = Some(rgb[(actual - 1) * frame_bytes..].to_vec());
                read_frames += new_frames;
                // 最后一批复制尾帧对齐 VAE 时间压缩，编码时只写实际请求的帧。
                let mut values = vec![0.0; 3 * padded * pixels];
                for t in 0..padded {
                    for p in 0..pixels {
                        for c in 0..3 {
                            values[(c * padded + t) * pixels + p] = rgb[(t.min(actual - 1) * pixels + p) * 3 + c] as f32 * (2.0 / 255.0) - 1.0;
                        }
                    }
                }
                drop(rgb);
                on_progress(progress("vae_encode", written_frames, request.frames, &started));
                let phase = Instant::now();
                context.activate()?;
                let tensor = context.diffusion_tensor_from_f32(&values, 3, padded * pixels).map_err(|e| e.to_string())?;
                drop(values);
                let encoded = rocm_vae::encode(&self.pipeline.group, VaeVideo { tensor, shape: [padded, request.height, request.width] }, &self.vae).map_err(|e| format!("SeedVR2 VAE encode batch={batch}: {e}"))?;
                let latent = context.tensor_to_f32(&encoded.tensor).map_err(|e| e.to_string())?;
                if latent.iter().any(|v| !v.is_finite()) {
                    return Err(format!("SeedVR2 VAE encode batch={batch} 产生非有限 latent"));
                }
                let latent_shape = encoded.shape;
                drop(encoded);
                eprintln!("[seedvr2-native] batch={batch} vae_encode={:.3}s latent={latent_shape:?}", phase.elapsed().as_secs_f64());
                let noise = gaussian_noise(latent.len(), request.seed.wrapping_add(batch as u64));
                let phase = Instant::now();
                let denoised = self.pipeline.denoise_with_noise(&latent, &noise, latent_shape, &mut |completed, total| {
                    if cancellation.load(Ordering::Relaxed) {
                        return Err("SeedVR2 任务已取消".to_owned());
                    }
                    let _ = (completed, total);
                    on_progress(progress("denoise", written_frames, request.frames, &started));
                    Ok(())
                })?;
                drop(latent);
                drop(noise);
                if denoised.iter().any(|v| !v.is_finite()) {
                    return Err(format!("SeedVR2 DiT batch={batch} 产生非有限输出"));
                }
                eprintln!("[seedvr2-native] batch={batch} denoise={:.3}s ranks={}", phase.elapsed().as_secs_f64(), self.pipeline.group.contexts().len());
                context.activate()?;
                let tensor = context.diffusion_tensor_from_f32(&denoised, latent_shape.into_iter().product(), self.pipeline.config.output_channels).map_err(|e| e.to_string())?;
                drop(denoised);
                let phase = Instant::now();
                rocm_vae::decode_stream(&self.pipeline.group, VaeVideo { tensor, shape: latent_shape }, &self.vae, |offset, video| {
                    if cancellation.load(Ordering::Relaxed) {
                        return Err(BackendError::Compute { msg: "SeedVR2 任务已取消".to_owned() });
                    }
                    let start = offset.max(carry);
                    let end = (offset + video.shape[0]).min(actual);
                    if end <= start {
                        return Ok(());
                    }
                    // 最后一段直接输出本卡 RGB；回读时沿用该卡的 stream，避免跨回主卡。
                    let source = self.pipeline.group.contexts()[7].tensor_to_f32(&video.tensor)?;
                    if source.iter().any(|v| !v.is_finite()) {
                        return Err(BackendError::Compute { msg: format!("SeedVR2 VAE decode batch={batch} offset={offset} 产生非有限 RGB") });
                    }
                    let count = end - start;
                    if count == video.shape[0] {
                        writer.write_frames(&source, count, count).map_err(|msg| BackendError::Compute { msg })?;
                    } else {
                        let mut frames = Vec::with_capacity(3 * count * pixels);
                        for c in 0..3 {
                            let begin = (c * video.shape[0] + start - offset) * pixels;
                            frames.extend_from_slice(&source[begin..begin + count * pixels]);
                        }
                        writer.write_frames(&frames, count, count).map_err(|msg| BackendError::Compute { msg })?;
                    }
                    written_frames += count;
                    on_progress(progress("vae_decode", written_frames, request.frames, &started));
                    Ok(())
                })
                .map_err(|e| format!("SeedVR2 VAE decode batch={batch}: {e}"))?;
                eprintln!("[seedvr2-native] batch={batch} vae_decode_y4m={:.3}s frames={written_frames}/{}", phase.elapsed().as_secs_f64(), request.frames);
            }
            if written_frames != request.frames {
                return Err(format!("SeedVR2 输出帧数={written_frames}，期望 {}", request.frames));
            }
            writer.flush()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = reader.kill();
            let _ = encoder.kill();
        }
        let reader_status = reader.wait().map_err(|e| format!("等待SeedVR2 reader: {e}"));
        let encoder_status = encoder.wait().map_err(|e| format!("等待SeedVR2 encoder: {e}"));
        result?;
        if !reader_status?.success() || !encoder_status?.success() {
            return Err("SeedVR2 ffmpeg未正常结束".to_owned());
        }
        on_progress(progress("completed", request.frames, request.frames, &started));
        Ok(())
    }
}

fn progress(phase: &str, completed: usize, total: usize, started: &Instant) -> TaskProgress {
    TaskProgress { phase: phase.to_owned(), completed, total, elapsed_seconds: started.elapsed().as_secs_f64(), phase_eta_seconds: None, preview: None }
}

impl NodeEngine for SeedVr2Engine {
    fn model_key(&self) -> &'static str {
        "SeedVR2-7B"
    }
    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        (self.capabilities.clone(), Arc::new(|| 0))
    }
    fn terminal_cache_infos(&self) -> Vec<crate::kv_cache::terminal_cache::TerminalInfo> {
        Vec::new()
    }
    fn max_concurrency(&self) -> usize {
        1
    }
    fn execute_task(&mut self, kind: &str, request: &Value, output_dir: &Path, cancellation: &AtomicBool, on_progress: &mut dyn FnMut(TaskProgress)) -> Result<Vec<GeneratedArtifact>, String> {
        if kind != "video_super_resolution" {
            return Err(format!("SeedVR2不支持任务{kind}"));
        }
        let mut request: SeedVr2Request = serde_json::from_value(request.clone()).map_err(|e| format!("SeedVR2请求: {e}"))?;
        request.validate()?;
        fs::create_dir_all(output_dir).map_err(|e| format!("创建SeedVR2输出目录: {e}"))?;
        let transferred_input = if let Some(data) = request.input_video_data.take() {
            use base64::Engine;
            let encoded = data.strip_prefix("data:video/mp4;base64,").ok_or("SeedVR2 输入不是 MP4 data URL")?;
            let bytes = base64::engine::general_purpose::STANDARD.decode(encoded).map_err(|e| format!("SeedVR2 输入解码: {e}"))?;
            if bytes.is_empty() || bytes.len() > 50 * 1024 * 1024 {
                return Err("SeedVR2 输入视频必须为 1..50MB".to_owned());
            }
            let path = output_dir.join("input.mp4");
            fs::write(&path, bytes).map_err(|e| format!("SeedVR2 写入输入视频: {e}"))?;
            request.input_video = path.clone();
            Some(path)
        } else {
            None
        };
        let output = output_dir.join("output.mp4");
        let result = self.upscale(&request, &output, cancellation, on_progress);
        if let Some(path) = transferred_input {
            let _ = fs::remove_file(path);
        }
        result?;
        Ok(vec![GeneratedArtifact { id: "output".to_owned(), file_name: "output.mp4".to_owned(), path: output, content_type: "video/mp4".to_owned() }])
    }
}

pub async fn run(model: SeedVr2NodeModelConfig, backend: RocmBackendConfig, config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    let factory = Box::new(move |_, _| -> Result<Box<dyn NodeEngine>, DynError> { Ok(Box::new(SeedVr2Engine::load(&model, &backend)?)) });
    crate::server::node::run_node(config, factory).await
}
