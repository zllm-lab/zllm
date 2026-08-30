//! zllm-server:纯 HTTP 调度入口,只跑 axum + iroh SchedulerService,不在本进程内加载模型。
//! 多个 zllm-rt-<platform> 节点通过 iroh ticket 连接,所有推理请求经 HTTP 进来后由
//! 调度器按 cache 命中 + 负载均衡分配到对应节点。

use zllm::config::{ConfigCommand, SchedulerProcessConfig, parse_config_command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ConfigCommand::Load { path, check_only, print_effective } = parse_config_command(std::env::args()).map_err(std::io::Error::other)? else {
        println!("用法: zllm-server --config server.yaml [--check-config] [--print-effective-config]");
        return Ok(());
    };
    let config = SchedulerProcessConfig::load(&path)?;
    if print_effective {
        print!("{}", serde_yaml::to_string(&config)?);
    }
    if check_only {
        println!("zllm-server 配置有效: {}", path.display());
        return Ok(());
    }
    let (address, server_config) = config.runtime()?;
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(address).await?;
        eprintln!("zLLM scheduler HTTP listening on http://{address}");
        zllm::server::serve(listener, server_config).await
    })?;
    Ok(())
}
