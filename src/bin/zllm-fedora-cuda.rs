// zllm-<platform>-cuda:不启动网络服务的 CUDA 终端对话入口。

use std::{
    io::{self, BufRead, IsTerminal, Write},
    sync::atomic::AtomicBool,
};

use serde_json::{Value, json};
use zllm::{config::RuntimeProcessConfig, runtime::node::load_direct_engine, server::node::NodeEngine};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let program = std::env::current_exe().ok().and_then(|path| path.file_stem().map(|name| name.to_string_lossy().into_owned())).unwrap_or_else(|| "zllm-cuda".to_owned());
    let (config_path, prompt, max_tokens) = parse_args(std::env::args().skip(1)).map_err(io::Error::other)?;
    let config = RuntimeProcessConfig::load(&config_path)?;
    let RuntimeProcessConfig::Standalone(config) = config else {
        return Err(format!("{program} 只接受 kind: standalone 配置，但不会启动 standalone HTTP 服务").into());
    };
    if !matches!(config.backend, zllm::config::NodeBackendConfig::Cuda(_)) {
        return Err(format!("{program} 要求 backend.kind: cuda").into());
    }
    let mut engine = load_direct_engine(config.model, config.backend, config.node.cache_directory, config.node.persist_kv_cache, config.node.terminal_cache_global_entries)?;
    let cancelled = AtomicBool::new(false);
    let mut history = Vec::<Value>::new();
    if let Some(prompt) = prompt {
        run_turn(&mut engine, &cancelled, &mut history, prompt, max_tokens).map_err(io::Error::other)?;
        return Ok(());
    }
    if !io::stdin().is_terminal() {
        let prompt = io::stdin().lock().lines().collect::<Result<Vec<_>, _>>()?.join("\n");
        if !prompt.trim().is_empty() {
            run_turn(&mut engine, &cancelled, &mut history, prompt, max_tokens).map_err(io::Error::other)?;
        }
        return Ok(());
    }
    eprintln!("{program} 已就绪；输入 /exit 退出，/clear 清空对话。");
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        if io::stdin().read_line(&mut prompt)? == 0 {
            break;
        }
        let prompt = prompt.trim();
        if prompt.is_empty() {
            continue;
        }
        match prompt {
            "/exit" | "/quit" => break,
            "/clear" => {
                history.clear();
                eprintln!("对话已清空。");
            }
            _ => run_turn(&mut engine, &cancelled, &mut history, prompt.to_owned(), max_tokens).map_err(io::Error::other)?,
        }
    }
    Ok(())
}

fn run_turn(engine: &mut Box<dyn NodeEngine>, cancelled: &AtomicBool, history: &mut Vec<Value>, prompt: String, max_tokens: usize) -> Result<(), String> {
    history.push(json!({"role": "user", "content": prompt}));
    let request = json!({"model": engine.model_key(), "messages": history, "max_tokens": max_tokens, "temperature": 0});
    let mut assistant = String::new();
    let result = engine.generate_one("console", &request, cancelled, &mut |_, text| {
        print!("{text}");
        let _ = io::stdout().flush();
        assistant.push_str(&text);
        true
    });
    println!();
    match result {
        Ok(summary) => {
            history.push(json!({"role": "assistant", "content": assistant}));
            eprintln!("[{} prompt={} completion={}]", summary.finish_reason, summary.prompt_tokens, summary.completion_tokens);
            Ok(())
        }
        Err(error) => {
            history.pop();
            Err(error)
        }
    }
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<(std::path::PathBuf, Option<String>, usize), String> {
    let mut config = None;
    let mut prompt = None;
    let mut max_tokens = 512usize;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--config" => config = Some(args.next().ok_or("--config 缺少 YAML 路径")?.into()),
            "--prompt" => prompt = Some(args.next().ok_or("--prompt 缺少文本")?),
            "--max-tokens" => {
                let value = args.next().ok_or("--max-tokens 缺少数值")?;
                max_tokens = value.parse().map_err(|_| format!("--max-tokens={value} 不是正整数"))?;
                if max_tokens == 0 {
                    return Err("--max-tokens 必须大于 0".to_owned());
                }
            }
            "-h" | "--help" => return Err("用法: zllm-<platform>-cuda --config standalone-<model>-cuda.yaml [--prompt TEXT] [--max-tokens N]".to_owned()),
            _ => return Err(format!("未知参数 {argument}")),
        }
    }
    Ok((config.ok_or("缺少 --config YAML_PATH")?, prompt, max_tokens))
}
