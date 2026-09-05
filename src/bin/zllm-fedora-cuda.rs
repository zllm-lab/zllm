// zllm-<platform>-cuda:零配置的 CUDA 终端对话入口。
//
// 用法:
//   zllm-<platform>-cuda <模型路径> [--max-seq-len N] [--max-tokens N] [--prompt TEXT]
//   zllm-<platform>-cuda --config <standalone yaml> [--max-tokens N] [--prompt TEXT]
//
// 零配置路径不写 YAML:自动识别权重架构,自动探测 CUDA toolkit 头文件目录。
// Qwen3.6/3.8 由引擎按空闲显存自动 CPU+CUDA 分层,其余架构全 GPU 驻留;
// 上下文 8192 起步、加载失败减半重试。需要多卡、MTP、专家缓存等更多控制
// 请用 zllm-rt-cuda --config <yaml>。

use std::{
    io::{self, BufRead, IsTerminal, Write},
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Instant,
};

use serde_json::{Value, json};
use zllm::{
    config::{CudaBackendConfig, Gemma4NodeModelConfig, MistralNodeModelConfig, NodeBackendConfig, NodeModelConfig, OrnithNodeExecutionConfig, OrnithNodeModelConfig, Qwen36NodeModelConfig, Qwen36Variant, RuntimeProcessConfig},
    model_spec::qwen36::Qwen36Config,
    runtime::node::load_direct_engine,
    server::node::NodeEngine,
    weight::{container::gguf::GgufReader, model::gemma4::Gemma4Weights},
};

const GIB: u64 = 1024 * 1024 * 1024;
/// 零配置默认上下文:小显存卡的安全起点;加载失败会继续减半重试。
const DEFAULT_MAX_SEQ_LEN: usize = 8192;

#[derive(Debug)]
enum Invocation {
    Config { config_path: PathBuf, prompt: Option<String>, max_tokens: usize },
    Model { path: PathBuf, max_seq_len: Option<usize>, prompt: Option<String>, max_tokens: usize },
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let program = std::env::current_exe().ok().and_then(|path| path.file_stem().map(|name| name.to_string_lossy().into_owned())).unwrap_or_else(|| "zllm-cuda".to_owned());
    let invocation = parse_args(std::env::args().skip(1)).map_err(io::Error::other)?;
    let cancelled = AtomicBool::new(false);
    let (mut engine, prompt, max_tokens) = match invocation {
        Invocation::Config { config_path, prompt, max_tokens } => {
            let config = RuntimeProcessConfig::load(&config_path)?;
            let RuntimeProcessConfig::Standalone(config) = config else {
                return Err(format!("{program} 只接受 kind: standalone 配置，但不会启动 standalone HTTP 服务").into());
            };
            if !matches!(config.backend, NodeBackendConfig::Cuda(_)) {
                return Err(format!("{program} 要求 backend.kind: cuda").into());
            }
            let engine = load_direct_engine(config.model, config.backend, config.node.cache_directory, config.node.persist_kv_cache, config.node.terminal_cache_global_entries)?;
            (engine, prompt, max_tokens)
        }
        Invocation::Model { path, max_seq_len, prompt, max_tokens } => {
            let engine = run_zero_config(&program, &path, max_seq_len).map_err(io::Error::other)?;
            (engine, prompt, max_tokens)
        }
    };
    console(&program, &mut engine, &cancelled, prompt, max_tokens)?;
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Invocation, String> {
    let mut path = None;
    let mut config = None;
    let mut max_seq_len = None;
    let mut prompt = None;
    let mut max_tokens = 512usize;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--config" => {
                let value = args.next().ok_or("--config 缺少 YAML 路径")?;
                if config.replace(value.into()).is_some() {
                    return Err("--config 不能重复指定".to_owned());
                }
            }
            "--prompt" => prompt = Some(args.next().ok_or("--prompt 缺少文本")?),
            "--max-seq-len" => {
                let value = args.next().ok_or("--max-seq-len 缺少数值")?;
                let value = value.parse::<usize>().map_err(|_| format!("--max-seq-len={value} 不是正整数"))?;
                if value == 0 {
                    return Err("--max-seq-len 必须大于 0".to_owned());
                }
                if max_seq_len.replace(value).is_some() {
                    return Err("--max-seq-len 不能重复指定".to_owned());
                }
            }
            "--max-tokens" => {
                let value = args.next().ok_or("--max-tokens 缺少数值")?;
                let value = value.parse::<usize>().map_err(|_| format!("--max-tokens={value} 不是正整数"))?;
                if value == 0 {
                    return Err("--max-tokens 必须大于 0".to_owned());
                }
                max_tokens = value;
            }
            "-h" | "--help" => return Err(USAGE.to_owned()),
            argument if argument.starts_with('-') => return Err(format!("未知参数 {argument}")),
            argument => {
                if path.replace(PathBuf::from(argument)).is_some() {
                    return Err(format!("只能指定一个模型路径，多余参数 {argument}"));
                }
            }
        }
    }
    match (path, config) {
        (Some(path), None) => Ok(Invocation::Model { path, max_seq_len, prompt, max_tokens }),
        (None, Some(config_path)) => {
            if max_seq_len.is_some() {
                return Err("--max-seq-len 只适用于零配置模型路径模式".to_owned());
            }
            Ok(Invocation::Config { config_path, prompt, max_tokens })
        }
        (Some(_), Some(_)) => Err("模型路径与 --config 不能同时指定".to_owned()),
        (None, None) => Err(USAGE.to_owned()),
    }
}

const USAGE: &str = "用法: zllm-<platform>-cuda <模型路径> [--max-seq-len N] [--max-tokens N] [--prompt TEXT]\n      zllm-<platform>-cuda --config standalone-<model>-cuda.yaml [--max-tokens N] [--prompt TEXT]";

// ---------- 零配置:模型识别 ----------

/// 已识别的模型形态。CUDA 执行路径只覆盖这四个架构;MiniCPM5 等无 CUDA 引擎,
/// 在 detect 阶段直接拒绝并引导用户改用对应平台的客户端。
enum DetectedModel {
    Gemma4 { context_limit: usize },
    Qwen36 { variant: Qwen36Variant, context_limit: usize },
    Ornith { context_limit: usize },
    Mistral { context_limit: usize },
}

impl DetectedModel {
    fn context_limit(&self) -> usize {
        match self {
            Self::Gemma4 { context_limit } | Self::Qwen36 { context_limit, .. } | Self::Ornith { context_limit } | Self::Mistral { context_limit } => *context_limit,
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Gemma4 { .. } => "gemma4 (全 GPU 驻留, KV f16)".to_owned(),
            Self::Qwen36 { variant, .. } => format!("{} (KV Q8g64, hybrid 显存自动分层)", variant.model_key()),
            Self::Ornith { .. } => "ornith (lazy experts, KV Q8g64)".to_owned(),
            Self::Mistral { .. } => "mistral (全 GPU 驻留, KV Q8g64)".to_owned(),
        }
    }
}

fn detect(path: &Path) -> Result<DetectedModel, String> {
    // GGUF 优先:单文件或目录内主权重(locate 会排除 mmproj)。
    if let Ok(main) = GgufReader::locate(path) {
        let reader = GgufReader::open(&main)?;
        let architecture = reader.metadata("general.architecture").and_then(|value| value.as_str()).unwrap_or_default().to_owned();
        return match architecture.as_str() {
            "gemma4" => Ok(DetectedModel::Gemma4 { context_limit: reader.metadata_u64("gemma4.context_length").unwrap_or(32_768) as usize }),
            "qwen35" => Ok(DetectedModel::Qwen36 { variant: infer_qwen_variant(path), context_limit: reader.metadata_u64("qwen35.context_length").unwrap_or(Qwen36Config::standard_27b().max_position_embeddings as u64) as usize }),
            "qwen35moe" => Ok(DetectedModel::Ornith { context_limit: reader.metadata_u64("qwen35moe.context_length")? as usize }),
            // llama 架构同时被 Mistral 等使用;ChatML 模板的是 MiniCPM5,只有 Metal/CPU 引擎。
            "llama" => {
                let chat_template = reader.metadata("tokenizer.chat_template").and_then(|value| value.as_str()).unwrap_or_default();
                if chat_template.contains("<|im_start|>") {
                    return Err("MiniCPM5 无 CUDA 执行路径;请在本机(Metal/CPU)使用 zllm-metal".to_owned());
                }
                Ok(DetectedModel::Mistral { context_limit: reader.metadata_u64("llama.context_length")? as usize })
            }
            other => Err(format!("GGUF 架构 {other} 尚未接入 CUDA 客户端;请使用 zllm-rt-cuda --config <yaml>")),
        };
    }
    // safetensors 目录:Gemma4 MLX affine 与 Qwen3.6/3.8 MLX packed 都是 CUDA 原生路径。
    let has_safetensors = path.is_dir() && std::fs::read_dir(path).map(|entries| entries.filter_map(Result::ok).any(|entry| entry.path().extension().is_some_and(|extension| extension == "safetensors"))).unwrap_or(false);
    if has_safetensors {
        // 无 config.json 的目录(如 Qwen MLX 导出)不是 Gemma4 MLX;读取失败
        // 不在识别层报错,由后续分支给出统一的"布局不支持"提示。
        if Gemma4Weights::is_mlx_affine(path).unwrap_or(false) {
            let config = Gemma4Weights::select_config(path).map_err(|error| format!("Gemma4 MLX 规格无效: {error:?}"))?;
            return Ok(DetectedModel::Gemma4 { context_limit: config.max_position_embeddings });
        }
        if path.to_string_lossy().to_lowercase().contains("qwen") {
            return Ok(DetectedModel::Qwen36 { variant: infer_qwen_variant(path), context_limit: Qwen36Config::standard_27b().max_position_embeddings });
        }
        return Err(format!("{} 是 safetensors 权重目录但不是 Gemma4 MLX 或 Qwen3.6/3.8 布局", path.display()));
    }
    Err(format!("{} 既不是 GGUF 也不是 safetensors 权重目录", path.display()))
}

/// Qwen3.8 与 3.6 的 GGUF 文本主干超参一致,variant 只影响 model_key 上报,
/// 因此从路径名启发式判断即可(见 config.rs 中 Qwen36Variant 注释)。
fn infer_qwen_variant(path: &Path) -> Qwen36Variant {
    let name = path.to_string_lossy().to_lowercase();
    if name.contains("qwen38") || name.contains("3.8") || name.contains("-38") { Qwen36Variant::Qwen38 } else { Qwen36Variant::Qwen36 }
}

fn weights_bytes(path: &Path) -> u64 {
    if path.is_file() {
        return std::fs::metadata(path).map(|metadata| metadata.len()).unwrap_or(0);
    }
    // mmproj 视觉塔与主权重同驻留,一并计入;macOS 的 ._ 元数据文件排除。
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|file| file.extension().is_some_and(|extension| extension == "gguf" || extension == "safetensors"))
                .filter(|file| file.file_name().and_then(|name| name.to_str()).is_some_and(|name| !name.starts_with("._")))
                .filter_map(|file| std::fs::metadata(&file).ok())
                .map(|metadata| metadata.len())
                .sum()
        })
        .unwrap_or(0)
}

// ---------- 零配置:CUDA toolkit 头文件探测 ----------

/// NVRTC 编译 kernel 需要 CUDA toolkit 头文件目录(-I)。按标准安装位置依次
/// 探测,候选必须真的含 cuda_runtime.h;全部失败时引导 --config 显式指定。
fn cuda_include_directory() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            candidates.push(PathBuf::from(root).join("include"));
        }
    }
    candidates.extend(["/usr/local/cuda/include", "/opt/cuda/include", "/usr/include"].map(PathBuf::from));
    if let Some(include) = nvcc_include_candidate() {
        candidates.push(include);
    }
    candidates.extend(windows_toolkit_includes());
    pick_include_directory(&candidates).ok_or_else(|| {
        let tried = candidates.iter().map(|directory| directory.display().to_string()).collect::<Vec<_>>().join(", ");
        format!("未找到含 cuda_runtime.h 的 CUDA 头文件目录(尝试过 {tried});请用 --config 在 backend.include_directory 显式指定")
    })
}

/// PATH 里的 nvcc 通常位于 <toolkit>/bin,头文件在同级 include。
fn nvcc_include_candidate() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let nvcc_name = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    std::env::split_paths(&path).map(|directory| directory.join(nvcc_name)).find(|nvcc| nvcc.is_file()).and_then(|nvcc| nvcc.parent().and_then(Path::parent).map(|root| root.join("include")))
}

/// Windows toolkit 目录按版本号降序排候选,优先最新安装。
fn windows_toolkit_includes() -> Vec<PathBuf> {
    let root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    let mut versions: Vec<String> =
        std::fs::read_dir(&root).into_iter().flatten().filter_map(Result::ok).filter(|entry| entry.path().is_dir()).filter_map(|entry| entry.file_name().into_string().ok()).filter(|name| name.starts_with('v')).collect();
    versions.sort_by(|left, right| right.cmp(left));
    versions.into_iter().map(|name| root.join(name).join("include")).collect()
}

fn pick_include_directory(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|directory| directory.join("cuda_runtime.h").is_file()).cloned()
}

// ---------- 零配置:资源规划与加载 ----------

struct ZeroConfigPlan {
    weights_directory: PathBuf,
    model: DetectedModel,
    weights_bytes: u64,
    max_seq_len: usize,
}

fn node_model_config(plan: &ZeroConfigPlan) -> NodeModelConfig {
    match &plan.model {
        DetectedModel::Gemma4 { .. } => {
            NodeModelConfig::Gemma4(Gemma4NodeModelConfig { weights_directory: plan.weights_directory.clone(), lm_head_quantization: Default::default(), max_sequence_length: plan.max_seq_len, execution: Default::default() })
        }
        DetectedModel::Qwen36 { variant, .. } => NodeModelConfig::Qwen36(Qwen36NodeModelConfig {
            weights_directory: plan.weights_directory.clone(),
            lm_head_quantization: Default::default(),
            max_sequence_length: plan.max_seq_len,
            variant: *variant,
            // cuda_gpu_layers: None —— 引擎按空闲显存自动 CPU+CUDA 分层。
            execution: Default::default(),
        }),
        DetectedModel::Ornith { .. } => NodeModelConfig::Ornith(OrnithNodeModelConfig {
            weights_directory: plan.weights_directory.clone(),
            lm_head_quantization: Default::default(),
            max_sequence_length: plan.max_seq_len,
            layer_ends: None,
            // 35B MoE 在单张消费级卡放不下全量 expert,保守走 lazy + 小缓存
            // (对齐 standalone-ornith-fedora-cuda.yaml 的 12GB 卡实测值)。
            execution: OrnithNodeExecutionConfig { lazy_experts: true, expert_cache_gib: 1, expert_prefetch_count: Some(2), ..Default::default() },
        }),
        DetectedModel::Mistral { .. } => {
            NodeModelConfig::Mistral(MistralNodeModelConfig { weights_directory: plan.weights_directory.clone(), lm_head_quantization: Default::default(), max_sequence_length: plan.max_seq_len, execution: Default::default() })
        }
    }
}

fn run_zero_config(program: &str, path: &Path, requested_max_seq_len: Option<usize>) -> Result<Box<dyn NodeEngine>, String> {
    let model = detect(path)?;
    let max_seq_len = requested_max_seq_len.unwrap_or(DEFAULT_MAX_SEQ_LEN).min(model.context_limit()).max(1024);
    let mut plan = ZeroConfigPlan { weights_directory: path.to_owned(), weights_bytes: weights_bytes(path), model, max_seq_len };
    let include_directory = cuda_include_directory()?;
    eprintln!("[{program}] {} | 权重 {:.2} GiB | 上下文 {} (模型上限 {}) | CUDA include {}", plan.model.label(), plan.weights_bytes as f64 / GIB as f64, plan.max_seq_len, plan.model.context_limit(), include_directory.display(),);
    load_with_retry(program, &mut plan, include_directory)
}

fn load_with_retry(program: &str, plan: &mut ZeroConfigPlan, include_directory: PathBuf) -> Result<Box<dyn NodeEngine>, String> {
    let backend = NodeBackendConfig::Cuda(CudaBackendConfig { device: 0, include_directory, architecture: None });
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let cache_directory = home.join(".cache").join("zllm").join("zllm-cuda");
    loop {
        let load_started = Instant::now();
        match load_direct_engine(node_model_config(plan), backend.clone(), cache_directory.clone(), false, 1) {
            Ok(engine) => {
                eprintln!("[{program}] 模型加载完成({:.1}s),直接输入消息开始对话;/exit 退出,/clear 清空", load_started.elapsed().as_secs_f32());
                return Ok(engine);
            }
            // 显存不足以容纳 KV/workspace 时降上下文重试;权重本身放不下时
            // 减半也无济于事,由最后一次错误带出。
            Err(error) if plan.max_seq_len > 4096 => {
                plan.max_seq_len = (plan.max_seq_len / 2).max(4096);
                eprintln!("[{program}] 加载失败({error}),上下文降至 {} 重试", plan.max_seq_len);
            }
            Err(error) => return Err(format!("加载失败: {error}")),
        }
    }
}

// ---------- 对话循环 ----------

fn console(program: &str, engine: &mut Box<dyn NodeEngine>, cancelled: &AtomicBool, prompt: Option<String>, max_tokens: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut history = Vec::<Value>::new();
    if let Some(prompt) = prompt {
        run_turn(engine, cancelled, &mut history, prompt, max_tokens).map_err(io::Error::other)?;
        return Ok(());
    }
    if !io::stdin().is_terminal() {
        let prompt = io::stdin().lock().lines().collect::<Result<Vec<_>, _>>()?.join("\n");
        if !prompt.trim().is_empty() {
            run_turn(engine, cancelled, &mut history, prompt, max_tokens).map_err(io::Error::other)?;
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
            _ => run_turn(engine, cancelled, &mut history, prompt.to_owned(), max_tokens).map_err(io::Error::other)?,
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

#[cfg(test)]
mod tests {
    use super::{DetectedModel, detect, parse_args, pick_include_directory};
    use zllm::model_spec::qwen36::Qwen36Config;

    #[test]
    fn 参数解析_模型路径与config互斥() {
        assert!(parse_args(["模型".to_owned(), "--config".to_owned(), "a.yaml".to_owned()]).is_err());
        assert!(parse_args(["--config".to_owned(), "a.yaml".to_owned(), "模型".to_owned()]).is_err());
        assert!(parse_args(Vec::<String>::new()).is_err());
        match parse_args(["模型".to_owned(), "--max-seq-len".to_owned(), "4096".to_owned(), "--max-tokens".to_owned(), "64".to_owned(), "--prompt".to_owned(), "你好".to_owned()]).unwrap() {
            super::Invocation::Model { path, max_seq_len, prompt, max_tokens } => {
                assert_eq!(path.to_string_lossy(), "模型");
                assert_eq!(max_seq_len, Some(4096));
                assert_eq!(prompt.as_deref(), Some("你好"));
                assert_eq!(max_tokens, 64);
            }
            other => panic!("应为模型路径模式:{other:?}"),
        }
        match parse_args(["--config".to_owned(), "a.yaml".to_owned()]).unwrap() {
            super::Invocation::Config { config_path, prompt, max_tokens } => {
                assert_eq!(config_path.to_string_lossy(), "a.yaml");
                assert!(prompt.is_none());
                assert_eq!(max_tokens, 512);
            }
            other => panic!("应为 config 模式:{other:?}"),
        }
    }

    #[test]
    fn 参数解析_数值参数拒绝零值与重复() {
        assert!(parse_args(["模型".to_owned(), "--max-seq-len".to_owned(), "0".to_owned()]).is_err());
        assert!(parse_args(["模型".to_owned(), "--max-tokens".to_owned(), "0".to_owned()]).is_err());
        assert!(parse_args(["--config".to_owned(), "a.yaml".to_owned(), "--max-seq-len".to_owned(), "4096".to_owned()]).is_err());
        assert!(parse_args(["模型".to_owned(), "--max-seq-len".to_owned(), "1".to_owned(), "--max-seq-len".to_owned(), "2".to_owned()]).is_err());
    }

    #[test]
    fn include候选过滤_含cuda_runtime_h的目录生效() {
        let directory = std::env::temp_dir().join(format!("zllm-cuda-include-test-{}", std::process::id()));
        let with_header = directory.join("with");
        let without_header = directory.join("without");
        std::fs::create_dir_all(&with_header).unwrap();
        std::fs::create_dir_all(&without_header).unwrap();
        std::fs::write(with_header.join("cuda_runtime.h"), b"// stub").unwrap();

        let picked = pick_include_directory(&[without_header.clone(), with_header.clone()]).unwrap();
        assert_eq!(picked, with_header);
        assert!(pick_include_directory(&[without_header]).is_none());

        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn detect_mlx_gemma4_按checkpoint识别() {
        let directory = std::env::temp_dir().join(format!("zllm-cuda-gemma4-mlx-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "quantization": { "mode": "affine", "bits": 4, "group_size": 64 },
                "text_config": { "hidden_size": 2560, "num_hidden_layers": 42 }
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(directory.join("model.safetensors"), []).unwrap();

        let detected = detect(&directory).expect("Gemma4 MLX 应按 config.json 识别");
        assert!(matches!(detected, DetectedModel::Gemma4 { context_limit: 131_072 }));
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn detect_目录名qwen识别为qwen36() {
        let directory = std::env::temp_dir().join(format!("Qwen3.6-27B-MLX-4bit-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), []).unwrap();

        let detected = detect(&directory).expect("Qwen safetensors 目录应按路径名识别");
        assert!(matches!(detected, DetectedModel::Qwen36 { variant: zllm::config::Qwen36Variant::Qwen36, context_limit } if context_limit == Qwen36Config::standard_27b().max_position_embeddings));
        let _ = std::fs::remove_dir_all(&directory);
    }
}
