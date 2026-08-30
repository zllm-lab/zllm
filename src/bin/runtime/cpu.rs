//! CPU 平台 runtime 入口：配置加载与模型分发。

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use zllm::config::{ConfigCommand, NodeBackendConfig, RuntimeProcessConfig, parse_config_command};

    let ConfigCommand::Load { path, check_only, print_effective } = parse_config_command(std::env::args()).map_err(std::io::Error::other)? else {
        println!("用法: zllm-rt-cpu --config standalone-or-node.yaml [--check-config] [--print-effective-config]");
        return Ok(());
    };
    let config = RuntimeProcessConfig::load(&path)?;
    if print_effective {
        print!("{}", serde_yaml::to_string(&config)?);
    }
    let backend = match &config {
        RuntimeProcessConfig::Node(node_config) => &node_config.backend,
        RuntimeProcessConfig::Standalone(standalone) => &standalone.backend,
        RuntimeProcessConfig::Stage(_) => return Err("zllm-rt-cpu 只接受 kind: standalone 或 kind: node；Stage 当前只支持 ROCm backend".into()),
    };
    let NodeBackendConfig::Cpu(cpu) = backend else {
        return Err("zllm-rt-cpu 只接受 kind: cpu backend".into());
    };
    if let Some(threads) = cpu.threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }
    if check_only {
        println!("CPU runtime 配置有效: {}", path.display());
        return Ok(());
    }
    match config {
        RuntimeProcessConfig::Node(node_config) => tokio::runtime::Runtime::new()?.block_on(zllm::runtime::node::run(node_config)).map_err(|error| -> Box<dyn std::error::Error> { format!("node 角色: {error}").into() }),
        RuntimeProcessConfig::Standalone(standalone) => tokio::runtime::Runtime::new()?.block_on(zllm::runtime::node::run_standalone(standalone)).map_err(|error| -> Box<dyn std::error::Error> { error.to_string().into() }),
        RuntimeProcessConfig::Stage(_) => unreachable!(),
    }
}
