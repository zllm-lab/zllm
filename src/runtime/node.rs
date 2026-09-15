//! Node 模型运行时分发与服务启动。
//!
//! 平台入口完成设备初始化后，把已校验配置交给这里。
use crate::config::{NodeBackendConfig, NodeModelConfig, NodeProcessConfig, StandaloneProcessConfig};

/// Session 只读能力描述；不包含 prefill/decode 行为，也不拥有设备资源。
pub struct SessionDescriptor<'a> {
    pub model_format: &'a str,
    pub model_bytes: u64,
    pub max_seq_len: usize,
    pub kv_cache_format: &'a str,
    pub input_modalities: &'a [&'a str],
}

/// backend 只读设备事实；不包含模型格式、上下文或 KV 几何。
pub struct DeviceDescriptor<'a> {
    pub backend: &'a str,
    pub accelerator: String,
    pub compute_units: Option<usize>,
    pub compute_unit_kind: &'a str,
    pub memory_kind: &'a str,
    pub unified_memory: bool,
    pub system_memory_bytes: Option<u64>,
    pub accelerator_memory_bytes: Option<u64>,
    pub recommended_working_set_bytes: Option<u64>,
}

pub fn text_capabilities(device: DeviceDescriptor<'_>, session: SessionDescriptor<'_>) -> crate::runtime::session::NodeCapabilities {
    crate::runtime::session::NodeCapabilities {
        backend: device.backend.to_owned(),
        platform: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        accelerator: device.accelerator,
        compute_units: device.compute_units,
        compute_unit_kind: device.compute_unit_kind.to_owned(),
        memory_kind: device.memory_kind.to_owned(),
        unified_memory: device.unified_memory,
        system_memory_bytes: device.system_memory_bytes,
        accelerator_memory_bytes: device.accelerator_memory_bytes,
        recommended_working_set_bytes: device.recommended_working_set_bytes,
        model_format: session.model_format.to_owned(),
        model_bytes: session.model_bytes,
        max_seq_len: session.max_seq_len,
        kv_cache_format: session.kv_cache_format.to_owned(),
        kv_cache_devices: Vec::new(),
        kv_reservation_page_tokens: 0,
        task_kinds: vec!["text_generation".to_owned()],
        task_models: Default::default(),
        input_modalities: session.input_modalities.iter().map(|value| (*value).to_owned()).collect(),
        output_modalities: vec!["text".to_owned()],
        artifact_streaming: false,
        terminal_resume_delta: false,
    }
}

/// backend 无关的单设备 KV 容量描述；设备枚举与可用预算策略由 backend 负责。
pub fn kv_device_capacity(device: String, available_bytes: usize, bytes_per_token: usize) -> crate::runtime::session::KvCacheDeviceCapacity {
    let bytes_per_token = bytes_per_token.max(1);
    crate::runtime::session::KvCacheDeviceCapacity { device, available_bytes: available_bytes as u64, bytes_per_token: bytes_per_token as u64, token_capacity: available_bytes / bytes_per_token }
}

/// 不启动网络宿主，直接装载一个模型 engine。console 与 Node 共用同一个
/// `NodeEngine` 实现，避免 prompt、session 和生成语义分叉。
pub fn load_direct_engine(
    model: NodeModelConfig,
    backend: NodeBackendConfig,
    cache_directory: std::path::PathBuf,
    persist_kv_cache: bool,
    resident_cache_entries: usize,
) -> Result<Box<dyn crate::server::node::NodeEngine>, crate::server::node::DynError> {
    let runtime = std::sync::Arc::new(std::sync::Mutex::new(crate::runtime::session::RuntimeStatus::default()));
    let compute_steps = std::sync::Arc::new(crate::runtime::session::AtomicCounterU64::new(0));
    match (model, backend) {
        (NodeModelConfig::Gemma4(model), NodeBackendConfig::Cuda(cuda)) => {
            #[cfg(feature = "with-cuda")]
            {
                crate::runtime::gemma4::cuda_node::Gemma4CudaEngine::load(&model, &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(feature = "with-cuda"))]
            {
                let _ = (model, cuda, runtime, compute_steps);
                Err("Gemma4 CUDA console 需要 --features with-cuda".into())
            }
        }
        (NodeModelConfig::Gemma4(model), NodeBackendConfig::Metal(metal)) => {
            #[cfg(target_os = "macos")]
            {
                crate::runtime::gemma4::engine::Gemma4Engine::load(
                    &model.weights_directory,
                    model.max_sequence_length,
                    model.execution,
                    metal.replay,
                    model.lm_head_quantization,
                    cache_directory,
                    persist_kv_cache,
                    resident_cache_entries,
                    runtime,
                    compute_steps,
                )
                .map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (model, metal, cache_directory, persist_kv_cache, resident_cache_entries, runtime, compute_steps);
                Err("Gemma4 Metal console 只支持 macOS".into())
            }
        }
        (NodeModelConfig::Qwen36(model), NodeBackendConfig::Metal(metal)) => {
            #[cfg(target_os = "macos")]
            {
                crate::runtime::qwen36::engine::Qwen36Engine::load(
                    &model.weights_directory,
                    model.max_sequence_length,
                    model.variant,
                    model.execution,
                    metal.replay,
                    model.lm_head_quantization,
                    cache_directory,
                    persist_kv_cache,
                    resident_cache_entries,
                    runtime,
                    compute_steps,
                )
                .map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (model, metal, cache_directory, persist_kv_cache, resident_cache_entries, runtime, compute_steps);
                Err("Qwen Metal console 只支持 macOS".into())
            }
        }
        (NodeModelConfig::Ornith(model), NodeBackendConfig::Metal(metal)) => {
            #[cfg(target_os = "macos")]
            {
                crate::runtime::ornith::node::OrnithEngine::load(
                    &model.weights_directory,
                    model.max_sequence_length,
                    crate::runtime::ornith::options::OrnithOptions::from(model.execution),
                    metal.replay,
                    model.lm_head_quantization,
                    runtime,
                    compute_steps,
                )
                .map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (model, metal, runtime, compute_steps);
                Err("Ornith Metal console 只支持 macOS".into())
            }
        }
        (NodeModelConfig::Mistral(model), NodeBackendConfig::Metal(metal)) => {
            #[cfg(target_os = "macos")]
            {
                crate::runtime::mistral::node::MistralEngine::load(
                    &model.weights_directory,
                    model.max_sequence_length,
                    model.execution.kv_cache_format == crate::config::KvCacheFormat::F16,
                    metal.replay,
                    model.lm_head_quantization,
                    runtime,
                    compute_steps,
                )
                .map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (model, metal, runtime, compute_steps);
                Err("Mistral Metal console 只支持 macOS".into())
            }
        }
        (NodeModelConfig::K2Horizon(model), NodeBackendConfig::Metal(metal)) => {
            #[cfg(target_os = "macos")]
            {
                crate::runtime::k2_horizon::node::K2Engine::load(
                    &model.weights_directory,
                    model.max_sequence_length,
                    model.execution.kv_cache_format == crate::config::KvCacheFormat::F16,
                    metal.replay,
                    model.execution.expert_cache_gib,
                    model.lm_head_quantization,
                    runtime,
                    compute_steps,
                )
                .map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (model, metal, runtime, compute_steps);
                Err("K2-Horizon Metal console 只支持 macOS".into())
            }
        }
        (NodeModelConfig::MiniCpm5(model), NodeBackendConfig::Metal(metal)) => {
            #[cfg(target_os = "macos")]
            {
                crate::runtime::minicpm5::engine::MiniCpm5Engine::load(
                    &model.weights_directory,
                    model.max_sequence_length,
                    model.execution.kv_cache_format == crate::config::KvCacheFormat::F16,
                    metal.replay,
                    model.lm_head_quantization,
                    runtime,
                    compute_steps,
                )
                .map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (model, metal, runtime, compute_steps);
                Err("MiniCPM5 Metal console 只支持 macOS".into())
            }
        }
        (NodeModelConfig::Qwen36(model), NodeBackendConfig::Cuda(cuda)) => {
            #[cfg(feature = "with-cuda")]
            {
                crate::runtime::qwen36::cuda_node::Qwen36CudaEngine::load(model, &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(feature = "with-cuda"))]
            {
                let _ = (model, cuda, runtime, compute_steps);
                Err("Qwen CUDA console 需要 --features with-cuda".into())
            }
        }
        (NodeModelConfig::Ornith(model), NodeBackendConfig::Cuda(cuda)) => {
            #[cfg(feature = "with-cuda")]
            {
                crate::runtime::ornith::cuda_node::OrnithCudaEngine::load(model, &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(feature = "with-cuda"))]
            {
                let _ = (model, cuda, runtime, compute_steps);
                Err("Ornith CUDA console 需要 --features with-cuda".into())
            }
        }
        (NodeModelConfig::Qwen4Exp(model), NodeBackendConfig::Cuda(cuda)) => {
            #[cfg(feature = "with-cuda")]
            {
                crate::runtime::qwen4exp::cuda_node::Qwen4ExpCudaEngine::load(model, &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(feature = "with-cuda"))]
            {
                let _ = (model, cuda, runtime, compute_steps);
                Err("Qwen4-Exp CUDA console 需要 --features with-cuda".into())
            }
        }
        (NodeModelConfig::Laguna(model), NodeBackendConfig::Cuda(cuda)) => {
            #[cfg(feature = "with-cuda")]
            {
                crate::runtime::laguna::cuda_node::LagunaCudaEngine::load(model, &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(feature = "with-cuda"))]
            {
                let _ = (model, cuda, runtime, compute_steps);
                Err("Laguna CUDA console 需要 --features with-cuda".into())
            }
        }
        (NodeModelConfig::Mistral(model), NodeBackendConfig::Cuda(cuda)) => {
            #[cfg(feature = "with-cuda")]
            {
                crate::runtime::mistral::cuda_node::MistralCudaEngine::load(model, &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(feature = "with-cuda"))]
            {
                let _ = (model, cuda, runtime, compute_steps);
                Err("Mistral CUDA console 需要 --features with-cuda".into())
            }
        }
        (NodeModelConfig::Ornith(model), NodeBackendConfig::Rocm(rocm)) => {
            #[cfg(all(target_os = "linux", feature = "with-rocm"))]
            {
                crate::runtime::ornith::rocm_node::OrnithRocmEngine::load(
                    &model.weights_directory,
                    rocm.devices.clone(),
                    rocm.allow_cpu_reference_fallback,
                    model.max_sequence_length,
                    model.layer_ends.clone(),
                    crate::runtime::ornith::options::OrnithOptions::from(model.execution),
                    model.lm_head_quantization,
                    runtime,
                    compute_steps,
                )
                .map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
            }
            #[cfg(not(all(target_os = "linux", feature = "with-rocm")))]
            {
                let _ = (model, rocm, runtime, compute_steps);
                Err("Ornith ROCm console 需要 Linux + --features with-rocm".into())
            }
        }
        (model, backend) => Err(format!("console 尚未接入 model={model:?} backend={backend:?}").into()),
    }
}

/// 单个宿主持有一个模型，加载和执行都在同一工作线程，避免两种任务争抢八卡。
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
struct SwitchingTaskEngine {
    models: Vec<NodeModelConfig>,
    backend: crate::config::RocmBackendConfig,
    engine: Option<Box<dyn crate::server::node::NodeEngine>>,
    primary: &'static str,
    capabilities: crate::runtime::session::NodeCapabilities,
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
impl SwitchingTaskEngine {
    fn load_model(model: &NodeModelConfig, backend: &crate::config::RocmBackendConfig) -> Result<Box<dyn crate::server::node::NodeEngine>, crate::server::node::DynError> {
        match model {
            NodeModelConfig::MinimaxH3(model) => Ok(Box::new(crate::runtime::h3::node::H3Engine::load(
                &model.weights_directory,
                &model.qwen_weights_directory,
                model.qwen_tokenizer_directory.as_deref().unwrap_or(&model.qwen_weights_directory),
                backend,
                model.execution.clone(),
            )?)),
            NodeModelConfig::Seedvr2(model) => Ok(Box::new(crate::runtime::seedvr2::rocm_node::SeedVr2Engine::load(model, backend)?)),
            _ => Err("该模型不支持任务切换".into()),
        }
    }

    fn switch_model(&mut self, model_key: &str) -> Result<(), String> {
        if self.engine.as_ref().is_some_and(|engine| engine.model_key() == model_key) {
            return Ok(());
        }
        let model = self
            .models
            .iter()
            .find(|model| match model {
                NodeModelConfig::MinimaxH3(_) => model_key == "MiniMax-H3",
                NodeModelConfig::Seedvr2(_) => model_key == "SeedVR2-7B",
                _ => false,
            })
            .ok_or_else(|| format!("切换节点未配置 {model_key}"))?;
        if let Some(mut old) = self.engine.take() {
            old.shutdown()?;
            drop(old);
        }
        for &device in &self.backend.devices {
            crate::kernel::rocm::hip::release_tensor_workspace(device as i32)?;
        }
        // 分配策略属于当前模型，不能由前一个模型的全局开关泄漏到下一次执行。
        crate::kernel::rocm::hip::set_device_buffer_reuse(model_key == "SeedVR2-7B");
        self.engine = Some(Self::load_model(model, &self.backend).map_err(|error| format!("加载 {model_key} 失败: {error}"))?);
        Ok(())
    }

    fn load(models: Vec<NodeModelConfig>, backend: crate::config::RocmBackendConfig) -> Result<Self, crate::server::node::DynError> {
        let engine = Self::load_model(models.first().ok_or("切换模型列表为空")?, &backend)?;
        let primary = engine.model_key();
        let (mut capabilities, _) = engine.startup_info();
        capabilities.task_models.insert("MiniMax-H3".to_owned(), vec!["video_generation".to_owned()]);
        capabilities.task_models.insert("SeedVR2-7B".to_owned(), vec!["video_super_resolution".to_owned()]);
        Ok(Self { models, backend, engine: Some(engine), primary, capabilities })
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
impl crate::server::node::NodeEngine for SwitchingTaskEngine {
    fn model_key(&self) -> &'static str {
        self.primary
    }
    fn startup_info(&self) -> (crate::runtime::session::NodeCapabilities, std::sync::Arc<dyn Fn() -> u64 + Send + Sync>) {
        // 原模型的统计闭包可能持有设备资源，不能跨模型切换保存。
        (self.capabilities.clone(), std::sync::Arc::new(|| 0))
    }
    fn terminal_cache_infos(&self) -> Vec<crate::server::scheduler::CacheInfo> {
        Vec::new()
    }
    fn max_concurrency(&self) -> usize {
        1
    }
    fn generate_one(&mut self, request_id: &str, request: &serde_json::Value, cancellation: &std::sync::atomic::AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<crate::server::node::GenerationSummary, String> {
        if cancellation.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("任务已取消".to_owned());
        }
        // H3 内置的 Qwen 提示词生成也使用同一执行槽；超分结束后先切回 H3。
        self.switch_model("MiniMax-H3")?;
        if cancellation.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("任务已取消".to_owned());
        }
        self.engine.as_mut().ok_or("模型加载后为空")?.generate_one(request_id, request, cancellation, on_token)
    }
    fn execute_task(
        &mut self,
        kind: &str,
        request: &serde_json::Value,
        output: &std::path::Path,
        cancellation: &std::sync::atomic::AtomicBool,
        progress: &mut dyn FnMut(crate::server::scheduler::TaskProgress),
    ) -> Result<Vec<crate::server::node::GeneratedArtifact>, String> {
        let model_key = request.get("model").and_then(serde_json::Value::as_str).ok_or("任务缺少 model")?;
        self.models
            .iter()
            .find(|model| match model {
                NodeModelConfig::MinimaxH3(_) => model_key == "MiniMax-H3" && kind == "video_generation",
                NodeModelConfig::Seedvr2(_) => model_key == "SeedVR2-7B" && kind == "video_super_resolution",
                _ => false,
            })
            .ok_or_else(|| format!("切换节点不支持 {model_key}/{kind}"))?;
        if cancellation.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("任务已取消".to_owned());
        }
        if self.engine.as_ref().is_none_or(|engine| engine.model_key() != model_key) {
            progress(crate::server::scheduler::TaskProgress { phase: "model_loading".to_owned(), completed: 0, total: 1, elapsed_seconds: 0.0, phase_eta_seconds: None, preview: None });
            self.switch_model(model_key)?;
        }
        if cancellation.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("任务已取消".to_owned());
        }
        self.engine.as_mut().ok_or("模型加载后为空")?.execute_task(kind, request, output, cancellation, progress)
    }
}

#[cfg(all(test, target_os = "linux", feature = "with-rocm"))]
mod switching_task_tests {
    use super::*;
    use crate::server::node::{GenerationSummary, NodeEngine};
    use std::sync::{Arc, atomic::AtomicBool};

    struct PromptEngine;
    impl NodeEngine for PromptEngine {
        fn model_key(&self) -> &'static str {
            "MiniMax-H3"
        }
        fn startup_info(&self) -> (crate::runtime::session::NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
            (Default::default(), Arc::new(|| 0))
        }
        fn terminal_cache_infos(&self) -> Vec<crate::server::scheduler::CacheInfo> {
            Vec::new()
        }
        fn max_concurrency(&self) -> usize {
            1
        }
        fn generate_one(&mut self, id: &str, request: &serde_json::Value, _: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
            assert_eq!(id, "prompt-id");
            assert_eq!(request["model"], "MiniMax-H3");
            assert!(on_token(7, "镜头缓慢推进".to_owned()));
            Ok(GenerationSummary { finish_reason: "stop".to_owned(), prompt_tokens: 3, completion_tokens: 1, cache: None, tool_calls: Vec::new() })
        }
    }

    #[test]
    fn shared_slot_preserves_h3_prompt_and_cancellation() {
        let mut engine =
            SwitchingTaskEngine { models: Vec::new(), backend: serde_json::from_value(serde_json::json!({"devices": []})).unwrap(), engine: Some(Box::new(PromptEngine)), primary: "MiniMax-H3", capabilities: Default::default() };
        let request = serde_json::json!({"model": "MiniMax-H3"});
        let mut tokens = Vec::new();
        let result = engine
            .generate_one("prompt-id", &request, &AtomicBool::new(false), &mut |id, text| {
                tokens.push((id, text));
                true
            })
            .unwrap();
        assert_eq!(result.completion_tokens, 1);
        assert_eq!(tokens, vec![(7, "镜头缓慢推进".to_owned())]);
        assert_eq!(engine.max_concurrency(), 1);
        let result = engine.generate_one("cancelled", &request, &AtomicBool::new(true), &mut |_, _| panic!("取消请求不能进入模型"));
        assert_eq!(result.err().as_deref(), Some("任务已取消"));
    }
}

pub async fn run(config: NodeProcessConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let node_config = config.runtime()?;
    run_parts(config.model, config.node.alternate_models, config.backend, node_config).await
}

async fn run_parts(model: NodeModelConfig, alternate_models: Vec<NodeModelConfig>, backend: NodeBackendConfig, node_config: crate::server::node::NodeConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if !alternate_models.is_empty() {
        #[cfg(all(target_os = "linux", feature = "with-rocm"))]
        {
            let NodeBackendConfig::Rocm(backend) = backend else { return Err("切换模型节点需要 ROCm".into()) };
            let models = std::iter::once(model).chain(alternate_models).collect();
            let factory = Box::new(move |_, _| -> Result<Box<dyn crate::server::node::NodeEngine>, crate::server::node::DynError> { Ok(Box::new(SwitchingTaskEngine::load(models, backend)?)) });
            return crate::server::node::run_node(node_config, factory).await;
        }
        #[cfg(not(all(target_os = "linux", feature = "with-rocm")))]
        return Err("切换模型节点需要 Linux ROCm 与 --features with-rocm".into());
    }
    match model {
        NodeModelConfig::Laguna(model) => match backend {
            NodeBackendConfig::Cuda(cuda) => {
                #[cfg(feature = "with-cuda")]
                {
                    crate::runtime::laguna::cuda_node::run(model, cuda, node_config).await
                }
                #[cfg(not(feature = "with-cuda"))]
                {
                    let _ = (model, cuda, node_config);
                    Err("Laguna CUDA Node 需要 --features with-cuda".into())
                }
            }
            _ => unreachable!("配置校验已保证 Laguna 只使用 CUDA"),
        },
        NodeModelConfig::Qwen4Exp(model) => match backend {
            NodeBackendConfig::Cuda(cuda) => {
                #[cfg(feature = "with-cuda")]
                {
                    crate::runtime::qwen4exp::cuda_node::run(model, cuda, node_config).await
                }
                #[cfg(not(feature = "with-cuda"))]
                {
                    let _ = (model, cuda, node_config);
                    Err("Qwen4-Exp CUDA Node 需要 --features with-cuda".into())
                }
            }
            _ => unreachable!("配置校验已保证 Qwen4-Exp 只使用 CUDA"),
        },
        NodeModelConfig::Ornith(model) => match backend {
            NodeBackendConfig::Metal(metal) => crate::runtime::ornith::node::run(model, metal, node_config).await,
            NodeBackendConfig::Cpu(_) => unreachable!("配置校验已保证 Ornith 不使用 CPU"),
            NodeBackendConfig::Cuda(cuda) => {
                #[cfg(feature = "with-cuda")]
                {
                    crate::runtime::ornith::cuda_node::run(model, cuda, node_config).await
                }
                #[cfg(not(feature = "with-cuda"))]
                {
                    let _ = (model, cuda, node_config);
                    Err("Ornith CUDA Node 需要 --features with-cuda".into())
                }
            }
            NodeBackendConfig::Rocm(rocm) => {
                #[cfg(all(target_os = "linux", feature = "with-rocm"))]
                {
                    crate::runtime::ornith::rocm_node::run(model, rocm, node_config).await
                }
                #[cfg(not(all(target_os = "linux", feature = "with-rocm")))]
                {
                    let _ = (model, rocm, node_config);
                    Err("Ornith ROCm Node 需要 Linux ROCm 与 --features with-rocm".into())
                }
            }
        },
        NodeModelConfig::Gemma4(model) => match backend {
            NodeBackendConfig::Metal(metal) => {
                #[cfg(target_os = "macos")]
                {
                    crate::runtime::gemma4::node::run(model, metal, node_config).await
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (model, metal, node_config);
                    Err("Gemma 4 Metal Node 需要 macOS".into())
                }
            }
            NodeBackendConfig::Cuda(cuda) => {
                #[cfg(feature = "with-cuda")]
                {
                    crate::runtime::gemma4::cuda_node::run(model, cuda, node_config).await
                }
                #[cfg(not(feature = "with-cuda"))]
                {
                    let _ = (model, cuda, node_config);
                    Err("Gemma 4 CUDA Node 需要 --features with-cuda".into())
                }
            }
            _ => unreachable!("配置校验已保证 Gemma 4 使用 Metal/CUDA"),
        },
        NodeModelConfig::Qwen36(model) => match backend {
            NodeBackendConfig::Metal(metal) => {
                #[cfg(target_os = "macos")]
                {
                    crate::runtime::qwen36::node::run(model, metal, node_config).await
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (model, metal, node_config);
                    Err("Qwen3.6/Qwen3.8 Metal Node 需要 macOS".into())
                }
            }
            NodeBackendConfig::Cpu(_) => unreachable!("配置校验已保证 Qwen3.6/Qwen3.8 不使用 CPU"),
            NodeBackendConfig::Cuda(cuda) => {
                #[cfg(feature = "with-cuda")]
                {
                    crate::runtime::qwen36::cuda_node::run(model, cuda, node_config).await
                }
                #[cfg(not(feature = "with-cuda"))]
                {
                    let _ = (model, cuda, node_config);
                    Err("Qwen3.6/Qwen3.8 CUDA Node 需要 --features with-cuda".into())
                }
            }
            NodeBackendConfig::Rocm(_) => unreachable!("配置校验已保证 Qwen3.6/Qwen3.8 不使用 ROCm"),
        },
        NodeModelConfig::DeepseekV4(model) => {
            #[cfg(all(target_os = "linux", feature = "with-rocm"))]
            {
                let NodeBackendConfig::Rocm(backend) = backend else { unreachable!("配置校验已保证 DeepSeek-V4 使用 ROCm") };
                crate::runtime::deepseek_v4::rocm_node::run(model, backend, node_config).await
            }
            #[cfg(not(all(target_os = "linux", feature = "with-rocm")))]
            {
                let _ = (model, backend, node_config);
                Err("DeepSeek-V4 Node 需要 Linux ROCm 与 --features with-rocm".into())
            }
        }
        NodeModelConfig::MinimaxH3(model) => {
            #[cfg(all(target_os = "linux", feature = "with-rocm"))]
            {
                let NodeBackendConfig::Rocm(backend) = backend else { unreachable!("配置校验已保证 H3 使用 ROCm") };
                crate::runtime::h3::node::run(model, backend, node_config).await
            }
            #[cfg(not(all(target_os = "linux", feature = "with-rocm")))]
            {
                let _ = model;
                Err("MiniMax-H3 Node 需要 Linux ROCm 与 --features with-rocm".into())
            }
        }
        NodeModelConfig::Flux2Klein(model) => {
            #[cfg(all(target_os = "linux", feature = "with-rocm"))]
            {
                let NodeBackendConfig::Rocm(backend) = backend else { unreachable!("配置校验已保证 FLUX.2 Klein 使用 ROCm") };
                crate::runtime::flux2_klein::rocm::run(model, backend, node_config).await
            }
            #[cfg(not(all(target_os = "linux", feature = "with-rocm")))]
            {
                let _ = (model, backend, node_config);
                Err("FLUX.2 Klein Node 需要 Linux ROCm 与 --features with-rocm".into())
            }
        }
        NodeModelConfig::Seedvr2(model) => {
            #[cfg(all(target_os = "linux", feature = "with-rocm"))]
            {
                let NodeBackendConfig::Rocm(backend) = backend else { unreachable!("配置校验已保证 SeedVR2 使用 ROCm") };
                crate::runtime::seedvr2::rocm_node::run(model, backend, node_config).await
            }
            #[cfg(not(all(target_os = "linux", feature = "with-rocm")))]
            {
                let _ = (model, backend, node_config);
                Err("SeedVR2 Node 需要 Linux ROCm 与 --features with-rocm".into())
            }
        }
        NodeModelConfig::Glm53Flash(model) => {
            #[cfg(all(target_os = "linux", feature = "with-rocm"))]
            {
                let NodeBackendConfig::Rocm(backend) = backend else { unreachable!("配置校验已保证 GLM-5.3-Flash 使用 ROCm") };
                crate::runtime::glm53_flash::rocm_node::run(model, backend, node_config).await
            }
            #[cfg(not(all(target_os = "linux", feature = "with-rocm")))]
            {
                let _ = (model, backend, node_config);
                Err("GLM-5.3-Flash Node 需要 Linux ROCm 与 --features with-rocm".into())
            }
        }
        NodeModelConfig::Glm52(model) => {
            #[cfg(all(target_os = "linux", feature = "with-rocm"))]
            {
                let NodeBackendConfig::Rocm(backend) = backend else { unreachable!("配置校验已保证 GLM-5.2 使用 ROCm") };
                crate::runtime::glm52::rocm_node::run(model, backend, node_config).await
            }
            #[cfg(not(all(target_os = "linux", feature = "with-rocm")))]
            {
                let _ = model;
                Err("GLM-5.2 Node 需要 Linux ROCm 与 --features with-rocm".into())
            }
        }
        NodeModelConfig::Mistral(model) => match backend {
            NodeBackendConfig::Metal(metal) => crate::runtime::mistral::node::run(model, metal, node_config).await,
            NodeBackendConfig::Cuda(cuda) => {
                #[cfg(feature = "with-cuda")]
                {
                    crate::runtime::mistral::cuda_node::run(model, cuda, node_config).await
                }
                #[cfg(not(feature = "with-cuda"))]
                {
                    let _ = (model, cuda, node_config);
                    Err("Mistral CUDA Node 需要 --features with-cuda".into())
                }
            }
            _ => unreachable!("配置校验已保证 Mistral 使用 Metal/CUDA"),
        },
        NodeModelConfig::K2Horizon(model) => match backend {
            NodeBackendConfig::Metal(metal) => crate::runtime::k2_horizon::node::run(model, metal, node_config).await,
            _ => unreachable!("配置校验已保证 K2-Horizon 使用 Metal"),
        },
        NodeModelConfig::MiniCpm5(model) => match backend {
            NodeBackendConfig::Metal(metal) => crate::runtime::minicpm5::node::run(model, metal, node_config).await,
            NodeBackendConfig::Cpu(_) => crate::runtime::minicpm5::cpu_node::run(model, node_config).await,
            _ => unreachable!("配置校验已保证 MiniCPM5 使用 Metal/CPU"),
        },
    }
}

pub async fn run_standalone(config: StandaloneProcessConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (address, server) = config.server()?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    let model = config.model;
    let alternate_models = config.node.alternate_models;
    let backend = config.backend;
    let cache_dir = config.node.cache_directory;
    let persist_kv_cache = config.node.persist_kv_cache;
    let max_concurrency = config.node.max_concurrency;
    let model_alias = config.node.model_alias;
    let terminal_cache_global_entries = config.node.terminal_cache_global_entries;
    let terminal_cache_prefix_rounds = config.node.terminal_cache_prefix_rounds;
    let iroh = config.iroh.runtime()?;
    eprintln!("zLLM standalone HTTP listening on http://{address}");
    crate::server::serve_standalone(listener, server, move |scheduler| async move {
        let node = crate::server::node::NodeConfig {
            upstream: crate::server::node::NodeUpstream::Local(scheduler),
            api_key: None,
            cache_dir,
            persist_kv_cache,
            iroh,
            max_concurrency,
            model_alias,
            terminal_cache_global_entries,
            terminal_cache_prefix_rounds,
        };
        run_parts(model, alternate_models, backend, node).await
    })
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_capabilities_composes_device_and_session_facts() {
        let capabilities = text_capabilities(
            DeviceDescriptor {
                backend: "test",
                accelerator: "device".to_owned(),
                compute_units: Some(4),
                compute_unit_kind: "unit",
                memory_kind: "memory",
                unified_memory: false,
                system_memory_bytes: Some(10),
                accelerator_memory_bytes: Some(20),
                recommended_working_set_bytes: Some(15),
            },
            SessionDescriptor { model_format: "format", model_bytes: 30, max_seq_len: 40, kv_cache_format: "kv", input_modalities: &["text", "image"] },
        );
        assert_eq!(capabilities.backend, "test");
        assert_eq!(capabilities.model_format, "format");
        assert_eq!(capabilities.input_modalities, ["text", "image"]);
        assert_eq!(capabilities.task_kinds, ["text_generation"]);
        assert!(capabilities.kv_cache_devices.is_empty());
    }
}
