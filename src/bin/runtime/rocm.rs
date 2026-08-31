//! ROCm 进程入口：配置加载、平台初始化与模型分发。

#[cfg(not(target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("zllm-rt-rocm 目前仅支持 Linux".into())
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use zllm::config::{BackendConfig, ConfigCommand, RuntimeProcessConfig, parse_config_command};

    let ConfigCommand::Load { path, check_only, print_effective } = parse_config_command(std::env::args()).map_err(std::io::Error::other)? else {
        println!("用法: zllm-rt-rocm --config stage-or-standalone.yaml [--check-config] [--print-effective-config]");
        return Ok(());
    };
    let config = RuntimeProcessConfig::load(&path)?;
    if print_effective {
        print!("{}", serde_yaml::to_string(&config)?);
    }
    if let RuntimeProcessConfig::Node(node_config) = &config {
        configure_node_backend(node_config)?;
        if check_only {
            println!("ROCm node 配置有效: {}", path.display());
            return Ok(());
        }
        #[cfg(feature = "with-rocm")]
        {
            let RuntimeProcessConfig::Node(node_config) = config else { unreachable!() };
            return tokio::runtime::Runtime::new()?.block_on(zllm::runtime::node::run(node_config)).map_err(|error| -> Box<dyn std::error::Error> { format!("node 角色: {error}").into() });
        }
        #[cfg(not(feature = "with-rocm"))]
        return Err("ROCm node 需要 --features with-rocm".into());
    }
    if let RuntimeProcessConfig::Standalone(standalone) = &config {
        configure_standalone_backend(standalone)?;
        if check_only {
            println!("ROCm standalone HTTP 配置有效: {}", path.display());
            return Ok(());
        }
        #[cfg(feature = "with-rocm")]
        {
            let RuntimeProcessConfig::Standalone(standalone) = config else { unreachable!() };
            return tokio::runtime::Runtime::new()?.block_on(zllm::runtime::node::run_standalone(standalone)).map_err(|error| -> Box<dyn std::error::Error> { error.to_string().into() });
        }
        #[cfg(not(feature = "with-rocm"))]
        return Err("ROCm standalone 需要 --features with-rocm".into());
    }
    if !matches!(config.backend(), BackendConfig::Rocm(_)) {
        return Err("zllm-rt-rocm 只接受 kind: rocm backend 或 kind: node".into());
    }
    if check_only {
        println!("ROCm runtime 配置有效: {}", path.display());
        return Ok(());
    }

    #[cfg(feature = "with-rocm")]
    {
        let BackendConfig::Rocm(backend) = config.backend().clone() else { unreachable!() };
        let kv_f16 = rocm_kv_f16(&config);
        configure_backend(&backend, kv_f16)?;
        let ctx = zllm::backend::rocm::RocmContext::configured(backend.devices[0], backend.allow_cpu_reference_fallback).map_err(|error| format!("ROCm 初始化失败: {error}"))?;
        println!("ROCm backend 就绪: {}", zllm::backend::rocm::device_name(&ctx));
        println!("kernel: {}", zllm::kernel::rocm::ROCM_STATUS);
        match config {
            RuntimeProcessConfig::Stage(stage) => match &stage.model {
                zllm::config::StageModelConfig::Glm52(_) => zllm::runtime::glm52::rocm::run(&ctx, RuntimeProcessConfig::Stage(stage)),
                zllm::config::StageModelConfig::Glm53Flash(_) => zllm::runtime::glm53_flash::rocm_stage::run(stage),
            },
            RuntimeProcessConfig::Standalone(_) => unreachable!(),
            RuntimeProcessConfig::Node(_) => unreachable!(),
        }
    }
    #[cfg(not(feature = "with-rocm"))]
    Err("ROCm runtime 需要 --features with-rocm".into())
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn rocm_kv_f16(config: &zllm::config::RuntimeProcessConfig) -> bool {
    use zllm::config::{KvCacheFormat, NodeModelConfig, RuntimeProcessConfig};
    match config {
        RuntimeProcessConfig::Node(config) => match &config.model {
            NodeModelConfig::Ornith(model) => model.execution.kv_cache_format == KvCacheFormat::F16,
            NodeModelConfig::Glm52(model) => model.execution.kv_cache_format == KvCacheFormat::F16,
            NodeModelConfig::MinimaxH3(_) => false,
            NodeModelConfig::DeepseekV4(_) => false,
            // 尚无 ROCm node engine,保持默认 KV 格式;真正拒绝运行由入口层负责。
            NodeModelConfig::Gemma4(_) | NodeModelConfig::Qwen36(_) | NodeModelConfig::Glm53Flash(_) | NodeModelConfig::Mistral(_) | NodeModelConfig::MiniCpm5(_) => false,
        },
        RuntimeProcessConfig::Stage(config) => match &config.model {
            zllm::config::StageModelConfig::Glm52(model) => model.execution.kv_cache_format == KvCacheFormat::F16,
            zllm::config::StageModelConfig::Glm53Flash(_) => false,
        },
        RuntimeProcessConfig::Standalone(config) => match &config.model {
            NodeModelConfig::Glm52(model) => model.execution.kv_cache_format == KvCacheFormat::F16,
            NodeModelConfig::Ornith(model) => model.execution.kv_cache_format == KvCacheFormat::F16,
            NodeModelConfig::MinimaxH3(_) => false,
            NodeModelConfig::DeepseekV4(_) => false,
            NodeModelConfig::Gemma4(_) | NodeModelConfig::Qwen36(_) | NodeModelConfig::Glm53Flash(_) | NodeModelConfig::Mistral(_) | NodeModelConfig::MiniCpm5(_) => false,
        },
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn configure_node_backend(config: &zllm::config::NodeProcessConfig) -> Result<(), Box<dyn std::error::Error>> {
    let zllm::config::NodeBackendConfig::Rocm(backend) = &config.backend else {
        return Err("zllm-rt-rocm node 角色只接受 ROCm backend".into());
    };
    configure_backend(backend, rocm_kv_f16(&zllm::config::RuntimeProcessConfig::Node(config.clone())))
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn configure_standalone_backend(config: &zllm::config::StandaloneProcessConfig) -> Result<(), Box<dyn std::error::Error>> {
    let zllm::config::NodeBackendConfig::Rocm(backend) = &config.backend else { return Err("zllm-rt-rocm standalone 只接受 ROCm backend".into()) };
    configure_backend(backend, rocm_kv_f16(&zllm::config::RuntimeProcessConfig::Standalone(config.clone())))
}

#[cfg(all(target_os = "linux", not(feature = "with-rocm")))]
fn configure_standalone_backend(_config: &zllm::config::StandaloneProcessConfig) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

#[cfg(all(target_os = "linux", not(feature = "with-rocm")))]
fn configure_node_backend(_config: &zllm::config::NodeProcessConfig) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn configure_backend(backend: &zllm::config::RocmBackendConfig, kv_f16: bool) -> Result<(), Box<dyn std::error::Error>> {
    zllm::runtime::glm52::dspark_cpu::set_profile_enabled(backend.kernel_profile);
    let options = zllm::kernel::rocm::hip::RocmOptions::configured(
        backend.kernel_sync,
        backend.kernel_profile,
        backend.decode_graph,
        backend.memory_pool,
        backend.grouped_down_route_buffer,
        backend.root.to_string_lossy().into_owned(),
        backend.hiprtc_cache_directory.clone(),
        kv_f16,
        backend.mla_decode_split_threshold,
        backend.dsa_hadamard_i8,
        backend.dsa_hadamard_shadow_samples,
        backend.dsa_hisa_shadow_samples,
        backend.dsa_cpu_select,
        backend.mla_cpu_hot_rows,
        backend.precise_router,
    );
    zllm::kernel::rocm::hip::configure(options)?;
    Ok(())
}
