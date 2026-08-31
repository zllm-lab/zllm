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
        input_modalities: session.input_modalities.iter().map(|value| (*value).to_owned()).collect(),
        output_modalities: vec!["text".to_owned()],
        artifact_streaming: false,
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
        (model, backend) => Err(format!("console 尚未接入 model={model:?} backend={backend:?}").into()),
    }
}

pub async fn run(config: NodeProcessConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let node_config = config.runtime()?;
    run_parts(config.model, config.backend, node_config).await
}

async fn run_parts(model: NodeModelConfig, backend: NodeBackendConfig, node_config: crate::server::node::NodeConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match model {
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
        run_parts(model, backend, node).await
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
