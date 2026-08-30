//! Qwen3.6 / Qwen3.8 Metal 库模式最小示例。

#[cfg(target_os = "macos")]
use std::{io::Write, time::Instant};

#[cfg(target_os = "macos")]
use serde_json::json;
#[cfg(target_os = "macos")]
use zllm::Engine;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let config = arguments.next().ok_or("用法: cargo run --example qwen36_embedded -- <embedded.yaml> <prompt> [max_tokens]")?;
    let prompt = arguments.next().ok_or("缺少 prompt")?;
    let max_tokens = arguments.next().map(|value| value.parse::<usize>()).transpose()?.unwrap_or(256);
    let load_started = Instant::now();
    let mut engine = Engine::from_config(config)?;
    let load_elapsed = load_started.elapsed();
    let cancellation = engine.cancellation();
    let generation_started = Instant::now();
    let result = engine.generate(
        &json!({
            "model": "qwen38",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens
        }),
        &cancellation,
        |_token, text| {
            print!("{text}");
            std::io::stdout().flush().is_ok()
        },
    )?;
    let generation_elapsed = generation_started.elapsed();
    println!();
    eprintln!(
        "finish={} prompt_tokens={} completion_tokens={} load={:.3}s generation={:.3}s throughput={:.3} tok/s cache_id={:?}",
        result.finish_reason,
        result.prompt_tokens,
        result.completion_tokens,
        load_elapsed.as_secs_f64(),
        generation_elapsed.as_secs_f64(),
        result.completion_tokens as f64 / generation_elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
        result.cache_id
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen36_embedded 仅支持 macOS Metal");
}
