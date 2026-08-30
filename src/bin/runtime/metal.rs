//! Mac Metal standalone 进程组合：完整 prefill + decode。

#[cfg(target_os = "macos")]
use zllm::config::NodeBackendConfig;

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("zllm-rt 是 Mac Metal 推理入口，当前平台不受支持".into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use zllm::config::{ConfigCommand, RuntimeProcessConfig, parse_config_command};
    let ConfigCommand::Load { path, check_only, print_effective } = parse_config_command(std::env::args()).map_err(std::io::Error::other)? else {
        println!("用法: zllm-rt-metal --config standalone-or-node.yaml [--check-config] [--print-effective-config]");
        return Ok(());
    };
    let config = RuntimeProcessConfig::load(&path)?;
    if print_effective {
        print!("{}", serde_yaml::to_string(&config)?);
    }
    if let RuntimeProcessConfig::Node(node_config) = &config {
        use zllm::config::NodeBackendConfig;
        if !matches!(node_config.backend, NodeBackendConfig::Metal(_)) {
            return Err("zllm-rt-metal node 角色只接受 metal backend".into());
        }
        if check_only {
            println!("Metal node 配置有效: {}", path.display());
            return Ok(());
        }
        let RuntimeProcessConfig::Node(node_config) = config else { unreachable!() };
        return tokio::runtime::Runtime::new()?.block_on(zllm::runtime::node::run(node_config)).map_err(|error| -> Box<dyn std::error::Error> { format!("node 角色: {error}").into() });
    }
    if check_only {
        println!("Metal standalone 配置有效: {}", path.display());
        return Ok(());
    }
    let RuntimeProcessConfig::Standalone(config) = config else {
        return Err("zllm-rt-metal 只接受 kind: standalone 或 kind: node".into());
    };
    if !matches!(&config.backend, NodeBackendConfig::Metal(_)) {
        return Err("zllm-rt-metal 只接受 kind: metal backend".into());
    }
    tokio::runtime::Runtime::new()?.block_on(zllm::runtime::node::run_standalone(config)).map_err(|error| -> Box<dyn std::error::Error> { error.to_string().into() })
}
