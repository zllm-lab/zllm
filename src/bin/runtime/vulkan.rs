#[cfg(target_os = "android")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or("用法: zllm-rt-vulkan MODEL.gguf [PROMPT] [DECODE_STEPS] [MAX_SEQ_LEN] [stream|resident|resident-profile]")?;
    let prompt = args.next().unwrap_or_else(|| "你好，请简短介绍你自己。".to_owned());
    let decode_steps = args.next().map(|value| value.parse()).transpose()?.unwrap_or(8);
    let max_seq_len = args.next().map(|value| value.parse()).transpose()?.unwrap_or(256);
    let (resident, profile) = match args.next().as_deref() {
        None | Some("stream") => (false, false),
        Some("resident") => (true, false),
        Some("resident-profile") => (true, true),
        Some(value) => return Err(format!("未知权重模式 {value}，期望 stream、resident 或 resident-profile").into()),
    };
    zllm::runtime::qwen36::vulkan::run(std::path::Path::new(&model), &prompt, max_seq_len, decode_steps, resident, profile)
}

#[cfg(not(target_os = "android"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("zllm-rt-vulkan 只能在 Android 上运行".into())
}
