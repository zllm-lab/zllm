//! Android QNN HTP runtime 入口。

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os();
    let _program = args.next();
    let Some(flag) = args.next() else {
        println!("用法: zllm-rt-qnn --config minicpm5-qnn.yaml [--check-config]");
        return Ok(());
    };
    if flag != "--config" {
        return Err("zllm-rt-qnn 只接受 --config <yaml>".into());
    }
    let path = std::path::PathBuf::from(args.next().ok_or("--config 缺少 YAML 路径")?);
    let check_only = match args.next() {
        Some(value) if value == "--check-config" => true,
        Some(value) => return Err(format!("未知参数 {}", value.to_string_lossy()).into()),
        None => false,
    };
    if args.next().is_some() {
        return Err("zllm-rt-qnn 参数过多".into());
    }
    let config = zllm::runtime::minicpm5::qnn::QnnRuntimeConfig::load(&path)?;
    if check_only {
        config.check()?;
        println!("MiniCPM5 QNN runtime 配置有效: {}", path.display());
        return Ok(());
    }
    config.run()
}
