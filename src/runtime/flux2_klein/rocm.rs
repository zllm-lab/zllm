//! FLUX.2 Klein 4B 与 ROCm backend 的单设备组合。

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use serde::Deserialize;
use serde_json::Value;

use crate::{
    backend::{DiffusionBackend, VaeBackend, rocm::RocmContext},
    config::{Flux2KleinNodeModelConfig, RocmBackendConfig},
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    model_spec::{flux2_klein::Flux2KleinConfig, qwen3_vl::Qwen3VlConfig},
    runtime::{
        flux2_klein::{Flux2PreparedGlobal, Flux2PreparedVae, decode_vae, denoise, denoise_conditioned, encode_vae, position_ids, position_tables, position_tables_conditioned, prepare_global, prepare_vae},
        qwen3_vl::{qwen3_text_hidden_concat_padded, qwen3_vl_mrope_table},
        session::NodeCapabilities,
    },
    server::{
        node::{DynError, GeneratedArtifact, NodeEngine},
        scheduler::TaskProgress,
    },
    tokenizer::Tokenizer,
    vision::{RgbImage, image_from_url, pillow_bicubic_resize},
    weight::model::{
        flux2_klein::{Flux2KleinSource, Flux2VaeSource},
        qwen3::Qwen3DenseWeights,
    },
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Flux2ImageRequest {
    model: String,
    prompt: String,
    #[serde(default = "default_image_size")]
    width: usize,
    #[serde(default = "default_image_size")]
    height: usize,
    #[serde(default = "default_steps")]
    steps: usize,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default)]
    input_image: Option<String>,
    #[serde(default)]
    input_images: Vec<String>,
}

const fn default_image_size() -> usize {
    512
}

const fn default_steps() -> usize {
    4
}

const fn default_seed() -> u64 {
    42
}

pub struct Flux2RocmPipeline {
    context: RocmContext,
    tokenizer: Tokenizer,
    text_config: Qwen3VlConfig,
    text_weights: Qwen3DenseWeights,
    transformer: Flux2KleinSource,
    transformer_global: Flux2PreparedGlobal<<RocmContext as crate::backend::BackendResources>::Weight>,
    vae: Flux2PreparedVae<<RocmContext as crate::backend::BackendResources>::Weight>,
}

pub struct Flux2KleinEngine {
    pipeline: Flux2RocmPipeline,
    capabilities: NodeCapabilities,
}

impl Flux2KleinEngine {
    pub fn load(model: &Flux2KleinNodeModelConfig, backend: &RocmBackendConfig) -> Result<Self, DynError> {
        let device = *backend.devices.first().ok_or("FLUX.2 Klein ROCm device 不能为空")?;
        let pipeline = Flux2RocmPipeline::load(&model.weights_directory, device, backend.allow_cpu_reference_fallback)?;
        let capabilities = NodeCapabilities {
            backend: "rocm".to_owned(),
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            accelerator: backend.accelerator_name.clone().unwrap_or_else(|| format!("ROCm device {device}")),
            compute_units: backend.compute_units,
            compute_unit_kind: "compute_unit".to_owned(),
            memory_kind: "dedicated".to_owned(),
            unified_memory: false,
            system_memory_bytes: None,
            accelerator_memory_bytes: backend.accelerator_memory_bytes,
            recommended_working_set_bytes: backend.recommended_working_set_bytes,
            model_format: "safetensors".to_owned(),
            model_bytes: directory_bytes(&model.weights_directory),
            max_seq_len: 512,
            kv_cache_format: "none".to_owned(),
            kv_cache_devices: Vec::new(),
            kv_reservation_page_tokens: 0,
            task_kinds: vec!["image_generation".to_owned()],
            task_models: Default::default(),
            input_modalities: vec!["text".to_owned(), "image".to_owned()],
            output_modalities: vec!["image".to_owned()],
            artifact_streaming: false,
            terminal_resume_delta: false,
        };
        Ok(Self { pipeline, capabilities })
    }
}

impl NodeEngine for Flux2KleinEngine {
    fn model_key(&self) -> &'static str {
        "FLUX.2-klein-4B"
    }

    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        (self.capabilities.clone(), Arc::new(|| 0))
    }

    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        Vec::new()
    }

    fn max_concurrency(&self) -> usize {
        1
    }

    fn execute_task(&mut self, task_kind: &str, request: &Value, output_dir: &Path, cancellation: &AtomicBool, on_progress: &mut dyn FnMut(TaskProgress)) -> Result<Vec<GeneratedArtifact>, String> {
        if task_kind != "image_generation" {
            return Err(format!("FLUX.2 Klein 不支持任务 {task_kind}"));
        }
        let request: Flux2ImageRequest = serde_json::from_value(request.clone()).map_err(|error| format!("解析 FLUX.2 image request: {error}"))?;
        if request.model != self.model_key() {
            return Err(format!("FLUX.2 model 必须为 {}", self.model_key()));
        }
        if cancellation.load(Ordering::Acquire) {
            return Err("FLUX.2 image_generation 已取消".to_owned());
        }
        std::fs::create_dir_all(output_dir).map_err(|error| format!("创建 FLUX.2 输出目录 {}: {error}", output_dir.display()))?;
        let started = Instant::now();
        on_progress(TaskProgress { phase: "generation".to_owned(), completed: 0, total: 1, elapsed_seconds: 0.0, phase_eta_seconds: None, preview: None });
        let path = output_dir.join("image.png");
        let references = request.input_image.iter().chain(&request.input_images).map(|url| image_from_url(url)).collect::<Result<Vec<_>, _>>()?;
        self.pipeline.generate_png_with_references(&request.prompt, &references, request.width, request.height, request.steps, request.seed, &path)?;
        if cancellation.load(Ordering::Acquire) {
            return Err("FLUX.2 image_generation 已取消".to_owned());
        }
        on_progress(TaskProgress { phase: "generation".to_owned(), completed: 1, total: 1, elapsed_seconds: started.elapsed().as_secs_f64(), phase_eta_seconds: Some(0.0), preview: None });
        Ok(vec![GeneratedArtifact { id: "image".to_owned(), file_name: "image.png".to_owned(), content_type: "image/png".to_owned(), path }])
    }
}

pub async fn run(model: Flux2KleinNodeModelConfig, backend: RocmBackendConfig, config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    let factory = Box::new(move |_runtime, _compute_steps| Flux2KleinEngine::load(&model, &backend).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

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

impl Flux2RocmPipeline {
    pub fn load(root: &Path, device: i32, allow_cpu_reference_fallback: bool) -> Result<Self, String> {
        if !root.is_dir() {
            return Err(format!("FLUX.2 Klein 模型目录不存在: {}", root.display()));
        }
        let context = RocmContext::configured(device, allow_cpu_reference_fallback).map_err(|error| format!("初始化 FLUX.2 ROCm device={device}: {error:?}"))?;
        let tokenizer = Tokenizer::new(root.join("tokenizer/tokenizer.json")).map_err(|error| format!("打开 FLUX.2 tokenizer: {error}"))?;
        let text_config = Qwen3VlConfig::dense_4b_flux2();
        text_config.validate()?;
        let text_weights = Qwen3DenseWeights::open(root.join("text_encoder"), text_config.clone())?;
        let transformer = Flux2KleinSource::open(root, Flux2KleinConfig::klein_4b())?;
        let transformer_global = prepare_global(&context, &transformer.load_global()?).map_err(backend_error)?;
        let vae_source = Flux2VaeSource::open(root)?;
        let vae = prepare_vae(&context, &vae_source).map_err(backend_error)?;
        Ok(Self { context, tokenizer, text_config, text_weights, transformer, transformer_global, vae })
    }

    pub fn generate_png(&self, prompt: &str, width: usize, height: usize, steps: usize, seed: u64, output: &Path) -> Result<(), String> {
        self.generate_png_with_references(prompt, &[], width, height, steps, seed, output)
    }

    pub fn generate_png_with_references(&self, prompt: &str, references: &[RgbImage], width: usize, height: usize, steps: usize, seed: u64, output: &Path) -> Result<(), String> {
        if prompt.trim().is_empty() {
            return Err("FLUX.2 prompt 不能为空".to_owned());
        }
        if width == 0 || height == 0 || !width.is_multiple_of(16) || !height.is_multiple_of(16) || width > 2048 || height > 2048 {
            return Err(format!("FLUX.2 width={width} height={height} 必须是 16 的倍数且位于 16..=2048"));
        }
        if !(1..=50).contains(&steps) {
            return Err(format!("FLUX.2 steps={steps} 必须位于 1..=50"));
        }
        if references.len() > 4 {
            return Err(format!("FLUX.2 Klein reference images={}，最多支持 4 张", references.len()));
        }
        let template = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let mut tokens = self.tokenizer.tokenize(template.as_bytes());
        tokens.truncate(512);
        let valid_tokens = tokens.len();
        if valid_tokens == 0 {
            return Err("FLUX.2 prompt tokenize 后为空".to_owned());
        }
        tokens.resize(512, self.text_config.bos_token_id);
        let embeddings = self.text_weights.embedding_rows_f32(&tokens)?;
        let hidden = self.context.diffusion_tensor_from_f32(&embeddings, 512, self.text_config.hidden_size).map_err(backend_error)?;
        let positions = (0..512).collect::<Vec<_>>();
        let rope = qwen3_vl_mrope_table(&self.text_config, &[positions.clone(), positions.clone(), positions]).map_err(|error| format!("FLUX.2 text RoPE: {error}"))?;
        let text = qwen3_text_hidden_concat_padded(&self.context, &self.text_config, &self.text_weights, hidden, &rope, &[9, 18, 27], valid_tokens).map_err(backend_error)?;

        let latent_height = height / 16;
        let latent_width = width / 16;
        let (text_ids, mut image_ids) = position_ids(512, latent_height, latent_width)?;
        let mut reference_latents = Vec::with_capacity(references.len());
        for (index, reference) in references.iter().enumerate() {
            let (reference, reference_height, reference_width) = prepare_reference_image(reference)?;
            let reference = self.context.vae_tensor_from_f32(reference, 3, reference_height * reference_width).map_err(backend_error)?;
            let latent = encode_vae(&self.context, &self.vae, &reference, reference_height, reference_width).map_err(backend_error)?;
            let packed_height = reference_height / 16;
            let packed_width = reference_width / 16;
            let time = (index + 1) as f32 * 10.0;
            image_ids.extend((0..packed_height).flat_map(|y| (0..packed_width).map(move |x| [time, y as f32, x as f32, 0.0])));
            reference_latents.push(latent);
        }
        let output_rows = latent_height * latent_width;
        let positions = if reference_latents.is_empty() { position_tables(self.transformer.config(), &text_ids, &image_ids)? } else { position_tables_conditioned(self.transformer.config(), &text_ids, &image_ids, output_rows)? };
        let noise = normal_noise(latent_height * latent_width * self.transformer.config().input_channels, seed);
        let image = self.context.diffusion_tensor_from_f32(&noise, latent_height * latent_width, self.transformer.config().input_channels).map_err(backend_error)?;
        let latent = if reference_latents.is_empty() {
            denoise(&self.context, &self.transformer, &self.transformer_global, &text, image, &positions, steps).map_err(backend_error)?
        } else {
            let mut reference = reference_latents.remove(0);
            for latent in reference_latents {
                reference = self.context.concat_rows(&reference, &latent).map_err(backend_error)?;
            }
            denoise_conditioned(&self.context, &self.transformer, &self.transformer_global, &text, &reference, image, &positions, steps).map_err(backend_error)?
        };
        let image = decode_vae(&self.context, &self.vae, &latent, latent_height, latent_width).map_err(backend_error)?;
        let values = self.context.vae_tensor_to_f32(&image).map_err(backend_error)?;
        write_png(&values, width, height, output)
    }
}

fn prepare_reference_image(reference: &RgbImage) -> Result<(Vec<f32>, usize, usize), String> {
    let area = reference.width.checked_mul(reference.height).ok_or("FLUX.2 reference image 面积溢出")?;
    let scale = if area > 1024 * 1024 { ((1024 * 1024) as f64 / area as f64).sqrt() } else { 1.0 };
    let width = ((reference.width as f64 * scale) as usize / 16 * 16).max(16);
    let height = ((reference.height as f64 * scale) as usize / 16 * 16).max(16);
    let source = image::RgbImage::from_raw(reference.width as u32, reference.height as u32, reference.pixels.clone()).ok_or("FLUX.2 reference RGB buffer 非法")?;
    let resized = pillow_bicubic_resize(&source, width as u32, height as u32);
    let mut values = vec![0.0f32; width * height * 3];
    for (pixel, rgb) in resized.pixels().enumerate() {
        for channel in 0..3 {
            values[channel * width * height + pixel] = rgb[channel] as f32 / 127.5 - 1.0;
        }
    }
    Ok((values, height, width))
}

fn normal_noise(elements: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut values = Vec::with_capacity(elements);
    while values.len() < elements {
        let u1 = uniform(&mut state).max(f64::MIN_POSITIVE);
        let u2 = uniform(&mut state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        values.push((radius * angle.cos()) as f32);
        if values.len() < elements {
            values.push((radius * angle.sin()) as f32);
        }
    }
    values
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    let value = value ^ (value >> 31);
    ((value >> 11) as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64))
}

fn write_png(values: &[f32], width: usize, height: usize, path: &Path) -> Result<(), String> {
    let spatial = width.checked_mul(height).ok_or("FLUX.2 PNG shape 溢出")?;
    if values.len() != spatial * 3 {
        return Err(format!("FLUX.2 VAE 输出元素={}，期望 {}", values.len(), spatial * 3));
    }
    let mut pixels = vec![0u8; spatial * 3];
    for pixel in 0..spatial {
        for channel in 0..3 {
            pixels[pixel * 3 + channel] = ((values[channel * spatial + pixel] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    let image = image::RgbImage::from_raw(width as u32, height as u32, pixels).ok_or("FLUX.2 PNG buffer shape 非法")?;
    image.save(path).map_err(|error| format!("保存 {}: {error}", path.display()))
}

fn backend_error(error: crate::backend::BackendError) -> String {
    format!("{error:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_noise_is_deterministic_and_finite() {
        let first = normal_noise(17, 42);
        assert_eq!(first, normal_noise(17, 42));
        assert_ne!(first, normal_noise(17, 43));
        assert!(first.iter().all(|value| value.is_finite()));
    }
}
