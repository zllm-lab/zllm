//! zllm-metal:零配置的本机 Metal 对话入口。
//!
//! 用法:zllm-metal <模型路径>(GGUF 文件或权重目录) [--image-max-tokens N]
//!
//! 不写 YAML:自动识别权重架构,按本机 GPU 工作集扣除权重与安全余量后推导
//! max_seq_len(Qwen3.6 用 Q8g64 KV,精算;Gemma4 是 hybrid GQA f16,按层型估算),
//! 进入多轮 prompt 模式;上下文接近上限时按 codex 风格 compact(旧轮次摘要 +
//! 保留近期原文)。需要更多控制(多卡、MTP、持久化 KV)请用 zllm-rt-metal --config。

#[cfg(target_os = "macos")]
mod run {
    use std::io::{BufRead, IsTerminal, Write};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use serde_json::{Value, json};
    use zllm::config::{
        Gemma4ExecutionConfig, Gemma4NodeModelConfig, MistralNodeModelConfig, NodeBackendConfig, NodeMetalBackendConfig, NodeModelConfig, OrnithNodeModelConfig, Qwen36NodeExecutionConfig, Qwen36NodeModelConfig, Qwen36Variant,
    };
    use zllm::embedded::{Cancellation, Engine, GenerationResult, SessionConfig};
    use zllm::kv_cache::{DEFAULT_GROUP_SIZE, KvCacheSpec};
    use zllm::model_spec::qwen36::Qwen36Config;
    use zllm::runtime::gemma4::Gemma4;
    use zllm::server::node::split_local_images;
    use zllm::weight::container::gguf::GgufReader;
    use zllm::weight::model::gemma4::Gemma4Weights;

    const GIB: u64 = 1024 * 1024 * 1024;
    const DEFAULT_QWEN_VISION_MAX_TOKENS: usize = 1024;
    /// compact 触发水位:留 20% 给新回复与 tokenizer 计数误差,避免 prefill 越界。
    const COMPACT_NUMERATOR: usize = 4;
    const COMPACT_DENOMINATOR: usize = 5;
    /// compact 后保留的近期完整轮数(user/assistant 对);更早的轮次交给摘要。
    const KEEP_TURNS: usize = 2;

    /// 已识别的嵌入式模型形态与推导 max_seq_len 所需的静态参数。
    pub enum DetectedModel {
        Gemma4 { layers: usize, kv_heads: usize, head_dim: usize, sliding_window: usize, sliding_layers: usize, context_limit: usize },
        Qwen36 { variant: Qwen36Variant, context_limit: usize },
        Ornith { layers: usize, kv_heads: usize, head_dim: usize, context_limit: usize },
        Mistral { layers: usize, kv_heads: usize, head_dim: usize, context_limit: usize },
        MiniCpm5 { layers: usize, kv_heads: usize, head_dim: usize, context_limit: usize },
    }

    impl DetectedModel {
        fn label(&self) -> String {
            match self {
                Self::Gemma4 { .. } => "gemma4 (hybrid GQA, KV f16)".to_owned(),
                Self::Qwen36 { variant, .. } => format!("{} (KV Q8g64)", variant.model_key()),
                Self::Ornith { .. } => "ornith (hybrid attention, KV Q8g64)".to_owned(),
                Self::Mistral { .. } => "mistral (dense GQA, KV Q8g64)".to_owned(),
                Self::MiniCpm5 { .. } => "minicpm5 (KV Q8g64)".to_owned(),
            }
        }
    }

    struct Plan {
        weights_directory: PathBuf,
        model: DetectedModel,
        weights_bytes: u64,
        kv_budget_bytes: u64,
        kv_bytes_per_token: usize,
        max_seq_len: usize,
        vision_max_tokens: Option<usize>,
    }

    pub fn main() -> Result<(), String> {
        let (path, requested_vision_max_tokens) = parse_args(std::env::args().skip(1))?;
        let model = detect(&path)?;
        let vision_max_tokens = match &model {
            DetectedModel::Qwen36 { .. } => Some(requested_vision_max_tokens.unwrap_or(DEFAULT_QWEN_VISION_MAX_TOKENS)),
            _ if requested_vision_max_tokens.is_some() => return Err("--image-max-tokens 当前只适用于 Qwen3.6/Qwen3.8".to_owned()),
            _ => None,
        };
        let mut plan = plan_resources(&path, model, vision_max_tokens)?;
        print_plan(&plan);
        let model_key = match &plan.model {
            DetectedModel::Gemma4 { .. } => "gemma4",
            DetectedModel::Qwen36 { variant, .. } => variant.model_key(),
            DetectedModel::Ornith { .. } => "ornith",
            DetectedModel::Mistral { .. } => "mistral",
            DetectedModel::MiniCpm5 { .. } => "minicpm5",
        };
        let mut engine = load_with_retry(&mut plan)?;
        let cancellation = engine.cancellation();
        repl(&mut engine, &cancellation, model_key, plan.max_seq_len);
        Ok(())
    }

    pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<(PathBuf, Option<usize>), String> {
        let mut path = None;
        let mut vision_max_tokens = None;
        let mut args = args.into_iter();
        while let Some(argument) = args.next() {
            if argument == "--image-max-tokens" {
                let value = args.next().ok_or("--image-max-tokens 缺少数值")?;
                let value = value.parse::<usize>().map_err(|_| format!("--image-max-tokens={value} 不是正整数"))?;
                if value == 0 {
                    return Err("--image-max-tokens 必须大于 0".to_owned());
                }
                if vision_max_tokens.replace(value).is_some() {
                    return Err("--image-max-tokens 不能重复指定".to_owned());
                }
            } else if argument.starts_with('-') {
                return Err(format!("未知参数 {argument}"));
            } else if path.replace(PathBuf::from(&argument)).is_some() {
                return Err(format!("只能指定一个模型路径，多余参数 {argument}"));
            }
        }
        let path = path.ok_or_else(|| "用法: zllm-metal <模型路径(GGUF 文件或权重目录)> [--image-max-tokens N]".to_owned())?;
        Ok((path, vision_max_tokens))
    }

    // ---------- 权重识别 ----------

    pub(super) fn detect(path: &Path) -> Result<DetectedModel, String> {
        // GGUF 优先:单文件或目录内主权重(locate 会排除 mmproj)。
        if let Ok(main) = GgufReader::locate(path) {
            let reader = GgufReader::open(&main)?;
            let architecture = reader.metadata("general.architecture").and_then(|value| value.as_str()).unwrap_or_default().to_owned();
            return match architecture.as_str() {
                "gemma4" => detect_gemma4(&reader),
                "qwen35" => Ok(DetectedModel::Qwen36 { variant: infer_qwen_variant(path), context_limit: reader.metadata_u64("qwen35.context_length").unwrap_or(Qwen36Config::standard_27b().max_position_embeddings as u64) as usize }),
                "qwen35moe" => Ok(DetectedModel::Ornith {
                    layers: reader.metadata_u64("qwen35moe.block_count")? as usize,
                    kv_heads: reader.metadata_u64("qwen35moe.attention.head_count_kv")? as usize,
                    head_dim: reader.metadata_u64("qwen35moe.attention.key_length")? as usize,
                    context_limit: reader.metadata_u64("qwen35moe.context_length")? as usize,
                }),
                // llama 架构同时被 Mistral 等使用;MiniCPM5 用 ChatML 模板,据此区分。
                "llama" => detect_llama_family(&reader),
                other => Err(format!("GGUF 架构 {other} 尚未接入嵌入式引擎;请使用 zllm-rt-metal --config <yaml>")),
            };
        }
        // 非 GGUF 目录按 checkpoint metadata 识别 Gemma4 MLX affine；Qwen
        // 暂无架构字段解析器，保留已有目录名启发式。
        let has_safetensors = path.is_dir() && std::fs::read_dir(path).map(|entries| entries.filter_map(Result::ok).any(|entry| entry.path().extension().is_some_and(|extension| extension == "safetensors"))).unwrap_or(false);
        if has_safetensors {
            if Gemma4Weights::is_mlx_affine(path)? {
                return detect_gemma4_mlx(path);
            }
            if path.to_string_lossy().to_lowercase().contains("qwen") {
                return Ok(DetectedModel::Qwen36 { variant: infer_qwen_variant(path), context_limit: Qwen36Config::standard_27b().max_position_embeddings });
            }
            return Err(format!("{} 是 safetensors 权重目录但不是 Gemma4 MLX 或 Qwen3.6/3.8 布局", path.display()));
        }
        Err(format!("{} 既不是 GGUF 也不是 MLX safetensors 权重目录", path.display()))
    }

    fn detect_gemma4(reader: &GgufReader) -> Result<DetectedModel, String> {
        let layers = reader.metadata_u64("gemma4.block_count")? as usize;
        // head_count_kv 可能是标量(gemma3 风格)或数组(混合层型逐层给出),取最大值保守估算。
        let kv_heads = reader
            .metadata("gemma4.attention.head_count_kv")
            .and_then(|value| value.as_u64().map(|scalar| scalar as usize).or_else(|| value.as_i64_array().map(|array| array.iter().map(|&head| head.max(0) as usize).max().unwrap_or(0))))
            .filter(|heads| *heads > 0)
            .ok_or("GGUF metadata 缺 gemma4.attention.head_count_kv")?;
        let head_dim = reader.metadata_u64("gemma4.attention.key_length")? as usize;
        let sliding_window = reader.metadata_u64("gemma4.attention.sliding_window").unwrap_or(0) as usize;
        let sliding_layers = reader.metadata("gemma4.attention.layer_types").and_then(|value| value.as_string_array()).map(|types| types.iter().filter(|kind| kind.contains("sliding")).count()).filter(|count| *count > 0).unwrap_or(0);
        let context_limit = reader.metadata_u64("gemma4.context_length").unwrap_or(32_768) as usize;
        Ok(DetectedModel::Gemma4 { layers, kv_heads, head_dim, sliding_window, sliding_layers, context_limit })
    }

    fn detect_gemma4_mlx(path: &Path) -> Result<DetectedModel, String> {
        let model = Gemma4::new(Gemma4Weights::select_config(path)?).map_err(|error| format!("Gemma4 MLX 规格无效: {error:?}"))?;
        let config = model.config();
        let sliding_layers = model.hybrid_gqa().layers().iter().filter(|layer| matches!(layer.window, zllm::attention::gqa::CausalWindow::Sliding { .. })).count();
        Ok(DetectedModel::Gemma4 {
            layers: config.layer_count,
            kv_heads: config.local_num_kv_heads.max(config.global_num_kv_heads),
            head_dim: config.local_head_dim.max(config.global_head_dim),
            sliding_window: config.sliding_window,
            sliding_layers,
            context_limit: config.max_position_embeddings,
        })
    }

    fn detect_llama_family(reader: &GgufReader) -> Result<DetectedModel, String> {
        let chat_template = reader.metadata("tokenizer.chat_template").and_then(|value| value.as_str()).unwrap_or_default();
        if !chat_template.contains("<|im_start|>") {
            return Ok(DetectedModel::Mistral {
                layers: reader.metadata_u64("llama.block_count")? as usize,
                kv_heads: reader.metadata_u64("llama.attention.head_count_kv")? as usize,
                head_dim: reader.metadata_u64("llama.attention.key_length")? as usize,
                context_limit: reader.metadata_u64("llama.context_length")? as usize,
            });
        }
        Ok(DetectedModel::MiniCpm5 {
            layers: reader.metadata_u64("llama.block_count")? as usize,
            kv_heads: reader.metadata_u64("llama.attention.head_count_kv")? as usize,
            head_dim: reader.metadata_u64("llama.attention.key_length")? as usize,
            context_limit: reader.metadata_u64("llama.context_length")? as usize,
        })
    }

    /// Qwen3.8 与 3.6 的 GGUF 文本主干超参一致,variant 只影响 model_key 上报,
    /// 因此从路径名启发式判断即可(见 config.rs 中 Qwen36Variant 注释)。
    fn infer_qwen_variant(path: &Path) -> Qwen36Variant {
        let name = path.to_string_lossy().to_lowercase();
        if name.contains("qwen38") || name.contains("3.8") || name.contains("-38") { Qwen36Variant::Qwen38 } else { Qwen36Variant::Qwen36 }
    }

    // ---------- 资源规划 ----------

    fn sysctl_u64(name: &str) -> Option<u64> {
        let output = std::process::Command::new("sysctl").args(["-n", name]).output().ok()?;
        String::from_utf8_lossy(&output.stdout).trim().parse().ok()
    }

    /// GPU 可用工作集:显式抬过 wired limit 时以它为准,否则按统一内存的 2/3 保守估计
    /// (低于 recommendedMaxWorkingSetSize 的典型值,宁可少算)。
    fn gpu_working_set_bytes() -> u64 {
        if let Some(mb) = sysctl_u64("iogpu.wired_limit_mb") {
            if mb > 0 {
                return mb.saturating_mul(1024 * 1024);
            }
        }
        sysctl_u64("hw.memsize").map_or(0, |ram| ram * 2 / 3)
    }

    fn weights_bytes(path: &Path) -> u64 {
        let sum_directory = |predicate: &dyn Fn(&str) -> bool| -> u64 {
            std::fs::read_dir(path)
                .map(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|file| file.extension().is_some_and(|extension| extension == "gguf" || extension == "safetensors"))
                        .filter(|file| file.file_name().and_then(|name| name.to_str()).is_some_and(|name| !name.starts_with("._") && predicate(name)))
                        .filter_map(|file| std::fs::metadata(&file).ok())
                        .map(|metadata| metadata.len())
                        .sum()
                })
                .unwrap_or(0)
        };
        if path.is_file() {
            return std::fs::metadata(path).map(|metadata| metadata.len()).unwrap_or(0);
        }
        // mmproj 视觉塔与主权重同驻留,一并计入。
        sum_directory(&|name| !name.starts_with("mmproj")) + sum_directory(&|name| name.starts_with("mmproj"))
    }

    /// 由 KV 预算反解上下文长度,向下对齐 1024。Qwen3.6 用 model_spec 精算
    /// (full-attention 层线性 + gated delta net 固定状态);Gemma4 的滑窗层只占
    /// 窗口容量,分两段解。估算误差由 load 失败减半重试兜底。
    pub fn solve_max_seq_len(model: &DetectedModel, kv_budget: u64) -> Result<(usize, usize), String> {
        let max_seq_len = match model {
            DetectedModel::MiniCpm5 { layers, kv_heads, head_dim, .. } => {
                // 24 层稠密 GQA,Q8g64 每 token 每层 = codes + scales。
                let spec = KvCacheSpec::Gqa { num_kv_heads: *kv_heads, head_dim: *head_dim };
                let per_layer = spec.bytes_per_token(DEFAULT_GROUP_SIZE)? as u64;
                kv_budget / per_layer / (*layers as u64).max(1)
            }
            DetectedModel::Mistral { layers, kv_heads, head_dim, .. } | DetectedModel::Ornith { layers, kv_heads, head_dim, .. } => {
                let spec = KvCacheSpec::Gqa { num_kv_heads: *kv_heads, head_dim: *head_dim };
                let per_layer = spec.bytes_per_token(DEFAULT_GROUP_SIZE)? as u64;
                kv_budget / per_layer / (*layers as u64).max(1)
            }
            DetectedModel::Qwen36 { .. } => {
                let config = Qwen36Config::standard_27b();
                let spec = KvCacheSpec::Gqa { num_kv_heads: config.num_kv_heads, head_dim: config.head_dim };
                let per_layer = spec.bytes_per_token(DEFAULT_GROUP_SIZE)? as u64;
                let full_layers = (config.num_layers / config.full_attention_interval) as u64;
                let recurrent_layers = (config.num_layers - config.num_layers / config.full_attention_interval) as u64;
                // gated delta net 的 conv/recurrent 状态是每会话固定驻留,先扣除再反解。
                let recurrent_fixed = recurrent_layers * ((config.gated_delta_net_spec().conv_state_elements() + config.gated_delta_net_spec().recurrent_elements()) as u64 * std::mem::size_of::<f32>() as u64);
                kv_budget.saturating_sub(recurrent_fixed) / full_layers.max(1) / per_layer.max(1)
            }
            DetectedModel::Gemma4 { layers, kv_heads, head_dim, sliding_window, sliding_layers, .. } => {
                // hybrid 布局按 f16:每 token 每 K 与 V 各 heads×head_dim×2 字节。
                let per_layer = (*kv_heads as u64) * (*head_dim as u64) * 4;
                let global_layers = (*layers as u64).saturating_sub(*sliding_layers as u64);
                if *sliding_layers == 0 || *sliding_window == 0 {
                    kv_budget / per_layer / (*layers as u64).max(1)
                } else {
                    // 先假设 seq 超过滑窗:滑窗层容量封顶在 window,只有全局层线性增长。
                    let window = (*sliding_window as u64).min(kv_budget / per_layer / (*sliding_layers as u64).max(1));
                    let seq = (kv_budget / per_layer).saturating_sub((*sliding_layers as u64) * window) / global_layers.max(1);
                    if seq > window { seq } else { kv_budget / per_layer / (*layers as u64).max(1) }
                }
            }
        };
        let context_limit = match model {
            DetectedModel::Gemma4 { context_limit, .. } => *context_limit,
            DetectedModel::Qwen36 { context_limit, .. } => *context_limit,
            DetectedModel::MiniCpm5 { context_limit, .. } => *context_limit,
            DetectedModel::Ornith { context_limit, .. } | DetectedModel::Mistral { context_limit, .. } => *context_limit,
        };
        let max_seq_len = max_seq_len.min(context_limit as u64) as usize;
        let max_seq_len = max_seq_len - max_seq_len % 1024;
        if max_seq_len < 1024 {
            return Err("KV 预算不足以支撑 1024 token 上下文;请关闭其他大内存应用后重试".to_owned());
        }
        let per_token_report = match model {
            DetectedModel::MiniCpm5 { layers, kv_heads, head_dim, .. } => {
                let spec = KvCacheSpec::Gqa { num_kv_heads: *kv_heads, head_dim: *head_dim };
                layers * spec.bytes_per_token(DEFAULT_GROUP_SIZE)?
            }
            DetectedModel::Mistral { layers, kv_heads, head_dim, .. } | DetectedModel::Ornith { layers, kv_heads, head_dim, .. } => {
                let spec = KvCacheSpec::Gqa { num_kv_heads: *kv_heads, head_dim: *head_dim };
                layers * spec.bytes_per_token(DEFAULT_GROUP_SIZE)?
            }
            DetectedModel::Qwen36 { .. } => {
                let config = Qwen36Config::standard_27b();
                let spec = KvCacheSpec::Gqa { num_kv_heads: config.num_kv_heads, head_dim: config.head_dim };
                (config.num_layers / config.full_attention_interval) * spec.bytes_per_token(DEFAULT_GROUP_SIZE)?
            }
            // 满上下文口径的保守每 token 字节数,只用于展示。
            DetectedModel::Gemma4 { layers, kv_heads, head_dim, .. } => layers * kv_heads * head_dim * 4,
        };
        Ok((max_seq_len, per_token_report))
    }

    fn plan_resources(path: &Path, model: DetectedModel, vision_max_tokens: Option<usize>) -> Result<Plan, String> {
        let working_set = gpu_working_set_bytes();
        if working_set == 0 {
            return Err("无法探测本机内存(sysctl hw.memsize 失败)".to_owned());
        }
        // 系统保留与 backend::metal::resource 同口径;再留 15% 覆盖 terminal cache
        // 与 kernel 工作区等加载前不可见的开销。
        let reserve = (working_set / 8).max(2 * GIB).min(working_set / 2);
        let weights = weights_bytes(path);
        let kv_budget = working_set.saturating_sub(reserve).saturating_sub(weights) / 100 * 85;
        let (max_seq_len, kv_bytes_per_token) = solve_max_seq_len(&model, kv_budget)?;
        Ok(Plan { weights_directory: path.to_owned(), model, weights_bytes: weights, kv_budget_bytes: kv_budget, kv_bytes_per_token, max_seq_len, vision_max_tokens })
    }

    fn print_plan(plan: &Plan) {
        let context_limit = match &plan.model {
            DetectedModel::Gemma4 { context_limit, .. } => *context_limit,
            DetectedModel::Qwen36 { context_limit, .. } => *context_limit,
            DetectedModel::MiniCpm5 { context_limit, .. } => *context_limit,
            DetectedModel::Ornith { context_limit, .. } | DetectedModel::Mistral { context_limit, .. } => *context_limit,
        };
        eprintln!(
            "[zllm-metal] {} | 权重 {:.2} GiB | KV 预算 {:.2} GiB | ≈{} B/token | 上下文 {} (模型上限 {})",
            plan.model.label(),
            plan.weights_bytes as f64 / GIB as f64,
            plan.kv_budget_bytes as f64 / GIB as f64,
            plan.kv_bytes_per_token,
            plan.max_seq_len,
            context_limit,
        );
        if let Some(tokens) = plan.vision_max_tokens {
            eprintln!("[zllm-metal] 单图视觉预算 ≤{tokens} tokens（可用 --image-max-tokens 调整）");
        }
    }

    // ---------- 引擎加载 ----------

    fn node_model_config(plan: &Plan) -> NodeModelConfig {
        match &plan.model {
            DetectedModel::Gemma4 { .. } => {
                let mtp_weights = if has_mmproj(&plan.weights_directory) {
                    eprintln!("[zllm-metal] 检测到视觉权重，Gemma4 MTP 已禁用");
                    None
                } else {
                    find_gemma_mtp(&plan.weights_directory)
                };
                if let Some(path) = &mtp_weights {
                    eprintln!("[zllm-metal] 自动启用 Gemma4 MTP: {}", path.display());
                }
                NodeModelConfig::Gemma4(Gemma4NodeModelConfig {
                    weights_directory: plan.weights_directory.clone(),
                    lm_head_quantization: Default::default(),
                    max_sequence_length: plan.max_seq_len,
                    execution: Gemma4ExecutionConfig { mtp_weights, replay: true, ..Default::default() },
                })
            }
            DetectedModel::Qwen36 { variant, .. } => NodeModelConfig::Qwen36(Qwen36NodeModelConfig {
                weights_directory: plan.weights_directory.clone(),
                lm_head_quantization: Default::default(),
                max_sequence_length: plan.max_seq_len,
                variant: *variant,
                execution: Qwen36NodeExecutionConfig { vision_max_tokens: plan.vision_max_tokens, ..Default::default() },
            }),
            DetectedModel::MiniCpm5 { .. } => NodeModelConfig::MiniCpm5(zllm::config::MiniCpm5NodeModelConfig {
                weights_directory: plan.weights_directory.clone(),
                lm_head_quantization: Default::default(),
                max_sequence_length: plan.max_seq_len,
                execution: Default::default(),
            }),
            DetectedModel::Ornith { .. } => NodeModelConfig::Ornith(OrnithNodeModelConfig {
                weights_directory: plan.weights_directory.clone(),
                lm_head_quantization: Default::default(),
                max_sequence_length: plan.max_seq_len,
                layer_ends: None,
                execution: Default::default(),
            }),
            DetectedModel::Mistral { .. } => {
                NodeModelConfig::Mistral(MistralNodeModelConfig { weights_directory: plan.weights_directory.clone(), lm_head_quantization: Default::default(), max_sequence_length: plan.max_seq_len, execution: Default::default() })
            }
        }
    }

    /// zllm-metal 的零配置入口只认同目录正式 `mtp-*.gguf`；主模型为 Q4 时
    /// 优先 Q4 draft，避免无意选中 BF16/Q8 增加常驻带宽。
    fn find_gemma_mtp(model_path: &Path) -> Option<PathBuf> {
        let directory = if model_path.is_dir() { model_path } else { model_path.parent()? };
        let main_q4 = model_path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.to_ascii_lowercase().contains("q4"));
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(directory)
            .ok()?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file() && path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("gguf")) && path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.to_ascii_lowercase().starts_with("mtp-"))
            })
            .collect();
        candidates.sort_by_key(|path| {
            let name = path.file_name().and_then(|name| name.to_str()).unwrap_or_default().to_ascii_lowercase();
            let preferred = main_q4 && name.contains("q4");
            (!preferred, name)
        });
        candidates.into_iter().next()
    }

    fn has_mmproj(model_path: &Path) -> bool {
        let Some(directory) = (if model_path.is_dir() { Some(model_path) } else { model_path.parent() }) else {
            return false;
        };
        std::fs::read_dir(directory).is_ok_and(|entries| {
            entries.filter_map(Result::ok).any(|entry| {
                let path = entry.path();
                path.is_file() && path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("mmproj") && name.ends_with(".gguf"))
            })
        })
    }

    fn load_with_retry(plan: &mut Plan) -> Result<Engine, String> {
        let backend = NodeBackendConfig::Metal(NodeMetalBackendConfig { device: "default".to_owned() });
        let cache_directory = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join(".cache").join("zllm").join("zllm-metal");
        let session = SessionConfig { cache_directory, persist_kv_cache: false, resident_cache_entries: 1 };
        loop {
            let load_started = Instant::now();
            match Engine::load(node_model_config(plan), backend.clone(), session.clone()) {
                Ok(engine) => {
                    eprintln!("[zllm-metal] 模型加载完成({:.1}s),直接输入消息开始对话;消息中的本地图片路径自动作为图像输入(需模型带视觉权重),行尾 \\ 续行;/exit 退出,/compact 手动压缩,/stats 查看资源", load_started.elapsed().as_secs_f32());
                    return Ok(engine);
                }
                // 预估偏乐观(工作集口径/滑窗近似误差)时降上下文重试,而不是直接失败。
                Err(error) if plan.max_seq_len > 4096 => {
                    plan.max_seq_len = (plan.max_seq_len / 2).max(4096);
                    eprintln!("[zllm-metal] 加载失败({error}),上下文降至 {} 重试", plan.max_seq_len);
                }
                Err(error) => return Err(format!("加载失败: {error}")),
            }
        }
    }

    // ---------- REPL ----------

    struct Turn {
        content: String,
        result: GenerationResult,
    }

    /// 历史消息的 content 在 push 时一次性固化为多模态 Value(文件路径展开/
    /// 文本内联只发生一次),后续轮次直接重放——否则历史里的图片每轮都会被
    /// 引擎重新视觉编码,文本内联每轮重复膨胀 token。
    fn messages_json(history: &[(String, Value)]) -> Vec<Value> {
        history.iter().map(|(role, content)| json!({ "role": role, "content": content })).collect()
    }

    /// 已固化 content 的纯文本视图(compact 的摘要转录用)。
    fn content_text(content: &Value) -> String {
        match content {
            Value::String(text) => text.clone(),
            Value::Array(pieces) => pieces.iter().filter_map(|piece| piece.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"),
            _ => String::new(),
        }
    }

    /// 单个文本文件的内联上限;超出截断并注明,防止误引用巨型文件撑爆上下文。
    const INLINE_TEXT_LIMIT: usize = 16 * 1024;

    /// 文本扩展名 → fence 语言标注;非文本扩展返回 None。
    fn text_fence(lowered: &str) -> Option<&'static str> {
        let extension = lowered.rsplit_once('.')?.1;
        let language = match extension {
            "rs" => "rust",
            "py" => "python",
            "js" | "ts" | "jsx" | "tsx" => "javascript",
            "go" => "go",
            "java" => "java",
            "c" | "h" => "c",
            "cpp" | "hpp" | "cc" => "cpp",
            "sh" | "zsh" => "shell",
            "sql" => "sql",
            "json" => "json",
            "yaml" | "yml" => "yaml",
            "toml" => "toml",
            "xml" => "xml",
            "html" => "html",
            "css" => "css",
            "md" => "markdown",
            "csv" | "log" | "txt" => "",
            _ => return None,
        };
        Some(language)
    }

    /// 读取文本文件内联体:前 4KiB 含 NUL 视为二进制不内联,超上限截断注明。
    fn read_text_inline(path: &str) -> Option<String> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() > 4096 && bytes[..4096].contains(&0) {
            return None;
        }
        let mut body = String::from_utf8_lossy(&bytes).into_owned();
        if body.len() > INLINE_TEXT_LIMIT {
            let truncated: String = body.drain(..INLINE_TEXT_LIMIT).collect();
            body = format!("{truncated}\n…(截断,全文 {} 字节)", bytes.len());
        }
        Some(body)
    }

    /// 消息内出现的本地文件路径自动展开:图片转 image_url 进视觉,文本类
    /// (源码/配置/日志)内容以 fenced block 内联,其余保留原路径。终端粘贴
    /// 文件/图片即粘贴路径,这一层让两种粘贴都直接可用。
    pub fn message_content(content: &str) -> Value {
        let (mut text, images) = split_local_images(content);
        if !images.is_empty() && text.trim_matches(|character: char| character.is_whitespace() || character == '>').is_empty() {
            text = if images.len() == 1 { "请描述这张图片".to_owned() } else { "请描述这些图片".to_owned() };
        }
        let mut inlined = false;
        for word in content.split_whitespace() {
            let lowered = word.to_lowercase();
            if !std::path::Path::new(word).is_file() {
                continue;
            }
            if let Some(language) = text_fence(&lowered) {
                if let Some(body) = read_text_inline(word) {
                    text.push_str(&format!("\n\n[{word}]\n```{language}\n{body}\n```"));
                    inlined = true;
                }
            }
        }
        if images.is_empty() && !inlined {
            return json!(content);
        }
        let mut pieces = images.into_iter().map(|path| json!({ "type": "image_url", "image_url": { "url": path } })).collect::<Vec<_>>();
        pieces.push(json!({ "type": "text", "text": text }));
        json!(pieces)
    }

    /// 把系统剪贴板中的图片(PNG)落盘并返回路径。终端 Cmd+V 不会把图片位数据
    /// 送进 stdin,真正的"粘贴图片"只能从剪贴板取;macOS 经 osascript 读
    /// «class PNGf»,非图片剪贴板返回明确错误。
    fn paste_clipboard_png() -> Result<PathBuf, String> {
        let directory = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join(".cache").join("zllm").join("zllm-metal");
        std::fs::create_dir_all(&directory).map_err(|error| format!("创建缓存目录: {error}"))?;
        let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
        let path = directory.join(format!("paste-{timestamp}.png"));
        let script =
            format!("set the png to (the clipboard as \u{ab}class PNGf\u{bb})\nset outFile to open for access (POSIX file \"{}\") with write permission\nset eof of outFile to 0\nwrite png to outFile\nclose access outFile", path.display());
        let status = std::process::Command::new("osascript").arg("-e").arg(script).status().map_err(|error| format!("执行 osascript: {error}"))?;
        if !status.success() {
            return Err("剪贴板里没有图片(仅支持 PNG 形态的截图/复制图)".to_owned());
        }
        let bytes = std::fs::read(&path).map_err(|error| format!("读取粘贴图: {error}"))?;
        if bytes.len() < 8 || bytes[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
            return Err("剪贴板内容不是有效的 PNG 图片".to_owned());
        }
        Ok(path)
    }

    /// 流式 <think>…</think> 显示过滤:跨 chunk 匹配标记,思考段静默、正文照打。
    /// 只影响显示,不改动累积文本(历史与 terminal cache 需要完整输出)。
    struct ThinkFilter {
        in_think: bool,
        /// 尾部可能构成标记前缀的未决字节。
        pending: String,
    }

    impl ThinkFilter {
        fn new(in_think: bool) -> Self {
            Self { in_think, pending: String::new() }
        }

        fn push(&mut self, chunk: &str) -> Option<String> {
            self.pending.push_str(chunk);
            let mut visible = String::new();
            loop {
                let open = self.pending.find("<think>");
                let close = self.pending.find("</think>");
                match (self.in_think, close, open) {
                    (false, _, Some(at)) => {
                        visible.push_str(&self.pending[..at]);
                        self.pending.drain(..at + "<think>".len());
                        self.in_think = true;
                    }
                    (true, Some(at), _) => {
                        self.pending.drain(..at + "</think>".len());
                        self.in_think = false;
                    }
                    (false, _, None) => break,
                    (true, None, _) => {
                        self.pending.clear();
                        break;
                    }
                }
            }
            // 尾部可能是半个标记,留住待下个 chunk;其余即为可见文本。标记全
            // ASCII,潜在前缀必是 pending 的 ASCII 后缀,其起点天然落在字符边界
            // (直接按字节数回退会切进多字节中文导致 char boundary panic)。
            if !self.in_think {
                let ascii_suffix = self.pending.bytes().rev().take_while(|byte| byte.is_ascii()).take("<think>".len() - 1).count();
                let keep = self.pending.len() - ascii_suffix;
                visible.push_str(&self.pending[..keep]);
                self.pending.drain(..keep);
            }
            (!visible.is_empty()).then_some(visible)
        }
    }

    /// 生成请求。多轮携带 cache_id:引擎侧是 token 级对齐 resume——全量分词与
    /// 缓存终态前缀完全一致才增量续写,漂移(assistant 重分词差异)时回退
    /// LCP 前缀复用或全量 prefill,不会有模板级 suffix 的 BPE 错位问题。
    fn generate(engine: &mut Engine, cancellation: &Cancellation, model_key: &str, messages: Vec<Value>, max_completion_tokens: usize, cache_id: Option<&str>, echo: bool) -> Result<Turn, String> {
        let mut content = String::new();
        // temperature/top_p 只有 MiniCPM5 路径读取:纯贪心在 MiniCPM5-1B 上会复读
        // 循环(node.rs 已注明)。思考保持默认开启——空围栏 no-think 下该模型多轮
        // 召回明显变弱(引擎级对照实测);REPL 只在显示层过滤 <think> 段,历史与
        // terminal cache 仍持有完整文本。
        // 注意:MiniCPM5/Qwen 的 <think>/</think> 是 special token,引擎解码时
        // skip_special_tokens 输出为空,文本流里没有边界标记,显示层无从过滤——
        // 思考文本按原样显示(信息不丢);filter 只对输出包含字面标记的形态生效。
        let mut filter = ThinkFilter::new(false);
        let result = engine.generate(
            &json!({ "model": model_key, "messages": messages, "max_completion_tokens": max_completion_tokens, "cache_id": cache_id, "temperature": 0.7, "top_p": 0.8, "enable_thinking": model_key == "minicpm5" }),
            cancellation,
            |_token, text| {
                content.push_str(text);
                if echo {
                    if let Some(visible) = filter.push(text) {
                        print!("{visible}");
                        let _ = std::io::stdout().flush();
                    }
                }
                true
            },
        )?;
        if echo {
            println!();
        }
        Ok(Turn { content, result })
    }

    /// codex 风格 compact:首个 system 与最近 KEEP_TURNS 轮保留原文,其余轮次
    /// 交给模型压成要点摘要后重建历史。摘要请求不携带 cache_id(前缀已变),
    /// 重建后的下一轮会全量 prefill 新前缀。
    fn compact(engine: &mut Engine, cancellation: &Cancellation, model_key: &str, history: &mut Vec<(String, Value)>, max_seq_len: usize) -> Result<Option<usize>, String> {
        let system_slots = usize::from(history.first().is_some_and(|(role, _)| role == "system"));
        let tail_start = history.len().saturating_sub(KEEP_TURNS * 2).max(system_slots);
        if tail_start <= system_slots {
            return Ok(None);
        }
        let transcript = history[system_slots..tail_start].iter().map(|(role, content)| format!("[{role}]\n{}", content_text(content))).collect::<Vec<_>>().join("\n\n");
        eprintln!("[compact] 压缩 {} 条旧消息…", tail_start - system_slots);
        let summary_budget = (max_seq_len / 8).clamp(256, 2048);
        let summary = generate(
            engine,
            cancellation,
            model_key,
            vec![json!({ "role": "user", "content": format!(
                "请将下面的对话历史压缩为一份简洁摘要,供后续对话参考。必须保留:用户的核心目标与约束、已确认的重要事实与数据、已做出的决定及其理由、涉及代码/文件/命令的关键信息、尚未完成或待办的事项。用紧凑的要点形式,不要复述寒暄,直接输出摘要。\n\n----- 对话开始 -----\n{transcript}\n----- 对话结束 -----"
            ) })],
            summary_budget,
            None,
            false,
        )?;
        let mut rebuilt: Vec<(String, Value)> = history[..system_slots].to_vec();
        rebuilt.push(("user".to_owned(), json!(format!("[此前对话的自动摘要]\n{}", summary.content))));
        rebuilt.extend(history[tail_start..].iter().cloned());
        *history = rebuilt;
        Ok(Some(summary.result.prompt_tokens + summary.result.completion_tokens))
    }

    /// 行读取:交互终端走 rustyline(历史/行内编辑/中文 IME 完整支持),
    /// 管道(自动化测试)走原始字节 + lossy 解码,坏字节只替换不丢行。
    enum LineReader {
        Editor { editor: rustyline::DefaultEditor, history_path: PathBuf },
        Piped { stdin: std::io::Stdin },
    }

    impl LineReader {
        fn new() -> Self {
            if std::io::stdin().is_terminal() {
                let mut editor = rustyline::DefaultEditor::new().expect("rustyline 初始化");
                let history_path = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join(".cache").join("zllm").join("zllm-metal").join("history");
                if let Some(parent) = history_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = editor.load_history(&history_path);
                Self::Editor { editor, history_path }
            } else {
                Self::Piped { stdin: std::io::stdin() }
            }
        }

        /// 读一行;返回 None 表示输入结束。Ctrl-C 清行继续,Ctrl-D 结束;
        /// Editor 失败(非 tty 等)自动降级管道模式。
        fn read_line(&mut self, prompt: &str) -> Option<String> {
            loop {
                let failure = match self {
                    Self::Editor { editor, history_path } => match editor.readline(prompt) {
                        Ok(line) => {
                            let _ = editor.add_history_entry(&line);
                            let _ = editor.save_history(history_path);
                            return Some(line);
                        }
                        Err(rustyline::error::ReadlineError::Interrupted) => continue,
                        Err(rustyline::error::ReadlineError::Eof) => return None,
                        Err(error) => Some(error),
                    },
                    Self::Piped { stdin } => {
                        let mut handle = stdin.lock();
                        let mut raw: Vec<u8> = Vec::new();
                        if handle.read_until(b'\n', &mut raw).unwrap_or(0) == 0 {
                            return None;
                        }
                        return Some(String::from_utf8_lossy(&raw).into_owned());
                    }
                };
                if let Some(error) = failure {
                    eprintln!("[zllm-metal] 行读取失败: {error};切换管道模式");
                    *self = Self::Piped { stdin: std::io::stdin() };
                }
            }
        }
    }

    fn repl(engine: &mut Engine, cancellation: &Cancellation, model_key: &str, max_seq_len: usize) {
        let compact_threshold = max_seq_len / COMPACT_DENOMINATOR * COMPACT_NUMERATOR;
        let max_completion_tokens = (max_seq_len / 4).clamp(512, 4096);
        let mut history: Vec<(String, Value)> = Vec::new();
        let mut cache_id: Option<String> = None;
        let mut context_tokens = 0usize;
        let mut reader = LineReader::new();
        loop {
            // 行尾反斜杠续行:多行粘贴(日志/代码片段/长中文)拼成一条消息发送。
            let mut input = String::new();
            loop {
                let Some(line) = reader.read_line(if input.is_empty() { "> " } else { "… " }) else {
                    return;
                };
                let trimmed = line.trim_end();
                let continued = trimmed.ends_with('\\') && !trimmed.ends_with("\\\\");
                if continued {
                    input.push_str(&trimmed[..trimmed.len() - 1]);
                    input.push('\n');
                } else {
                    input.push_str(&line);
                    break;
                }
            }
            let input = input.trim().to_owned();
            match input.as_str() {
                "" => continue,
                "/exit" | "/quit" => break,
                "/reset" => {
                    history.clear();
                    cache_id = None;
                    context_tokens = 0;
                    eprintln!("[zllm-metal] 已清空对话");
                    continue;
                }
                "/stats" => {
                    let report = engine.kv_resources();
                    eprintln!(
                        "[stats] 上下文 {context_tokens}/{max_seq_len} token | engine {:.0} MiB | KV capacity {:.0} MiB resident {:.0} MiB available {:.0} MiB / session {:.0} MiB",
                        report.engine_resident_bytes as f64 / 1048576.0,
                        report.capacity_bytes as f64 / 1048576.0,
                        report.resident_bytes as f64 / 1048576.0,
                        report.available_bytes as f64 / 1048576.0,
                        report.session_resident_bytes as f64 / 1048576.0
                    );
                    continue;
                }
                // /paste [问题]:剪贴板图片落盘后与问题一起发送;qwen36/38 的
                // mmproj 视觉权重直接消费 image_url。
                command if command == "/paste" || command.starts_with("/paste ") => {
                    match paste_clipboard_png() {
                        Ok(path) => {
                            let question = input.strip_prefix("/paste").unwrap_or_default().trim();
                            let question = if question.is_empty() { "请描述这张图片" } else { question };
                            let message = format!("{question}\n{}", path.display());
                            history.push(("user".to_owned(), message_content(&message)));
                            match generate(engine, cancellation, model_key, messages_json(&history), max_completion_tokens, cache_id.as_deref(), true) {
                                Ok(turn) => {
                                    cache_id = turn.result.cache_id;
                                    context_tokens = turn.result.prompt_tokens + turn.result.completion_tokens;
                                    history.push(("assistant".to_owned(), json!(turn.content)));
                                    eprintln!("[zllm-metal] {context_tokens}/{max_seq_len} token | finish={}", turn.result.finish_reason);
                                }
                                Err(error) => {
                                    eprintln!("[zllm-metal] 生成失败: {error}");
                                    history.pop();
                                }
                            }
                        }
                        Err(error) => eprintln!("[paste] {error}"),
                    }
                    continue;
                }
                "/compact" => {
                    match compact(engine, cancellation, model_key, &mut history, max_seq_len) {
                        Ok(Some(tokens)) => {
                            context_tokens = tokens;
                            cache_id = None;
                            eprintln!("[compact] 完成,当前约 {context_tokens} token");
                        }
                        _ => eprintln!("[compact] 没有可压缩的旧消息"),
                    }
                    continue;
                }
                _ => {}
            }
            history.push(("user".to_owned(), message_content(&input)));
            match generate(engine, cancellation, model_key, messages_json(&history), max_completion_tokens, cache_id.as_deref(), true) {
                Ok(turn) => {
                    cache_id = turn.result.cache_id;
                    context_tokens = turn.result.prompt_tokens + turn.result.completion_tokens;
                    history.push(("assistant".to_owned(), json!(turn.content)));
                    eprintln!("[zllm-metal] {context_tokens}/{max_seq_len} token | finish={}", turn.result.finish_reason);
                }
                Err(error) => {
                    eprintln!("[zllm-metal] 生成失败: {error}");
                    // 失败轮不进入历史;上一轮成功的 cache_id 前缀仍然有效,保留复用。
                    history.pop();
                }
            }
            if context_tokens >= compact_threshold && history.len() > KEEP_TURNS * 2 {
                match compact(engine, cancellation, model_key, &mut history, max_seq_len) {
                    Ok(Some(tokens)) => {
                        context_tokens = tokens;
                        cache_id = None;
                        eprintln!("[compact] 上下文达到 {} token,已压缩至约 {context_tokens} token", compact_threshold);
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    run::main()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("zllm-metal 仅支持 macOS(Metal);其他平台请使用对应的 zllm-rt-<platform> --config 入口");
    std::process::exit(1);
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::run::{DetectedModel, detect, message_content, parse_args, solve_max_seq_len};
    use serde_json::json;
    use zllm::config::Qwen36Variant;
    use zllm::model_spec::qwen36::Qwen36Config;

    fn qwen36() -> DetectedModel {
        DetectedModel::Qwen36 { variant: Qwen36Variant::Qwen36, context_limit: Qwen36Config::standard_27b().max_position_embeddings }
    }

    #[test]
    fn qwen36_预算反解对齐1024且受模型上限约束() {
        let (small, _) = solve_max_seq_len(&qwen36(), 512 * 1024 * 1024).expect("512MiB 必须能反解");
        let (large, _) = solve_max_seq_len(&qwen36(), 64 << 30).expect("64GiB 必须能反解");
        assert_eq!(small % 1024, 0);
        assert!(small >= 1024);
        assert!(small < large, "上下文应随预算单调增长");
        assert!(large <= Qwen36Config::standard_27b().max_position_embeddings, "不能超过模型上限");
        assert_eq!(large, Qwen36Config::standard_27b().max_position_embeddings, "64GiB 预算应顶到模型上限");
    }

    #[test]
    fn 图像预算参数支持前后顺序并拒绝零值() {
        let (path, budget) = parse_args(["模型".to_owned(), "--image-max-tokens".to_owned(), "768".to_owned()]).unwrap();
        assert_eq!(path.to_string_lossy(), "模型");
        assert_eq!(budget, Some(768));

        let (path, budget) = parse_args(["--image-max-tokens".to_owned(), "1024".to_owned(), "模型".to_owned()]).unwrap();
        assert_eq!(path.to_string_lossy(), "模型");
        assert_eq!(budget, Some(1024));
        assert!(parse_args(["模型".to_owned(), "--image-max-tokens".to_owned(), "0".to_owned()]).is_err());
    }

    #[test]
    fn 文件路径展开_文本内联图片多模态二进制跳过() {
        let directory = std::env::temp_dir().join(format!("zllm-metal-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("sample.rs");
        std::fs::write(&source, b"fn main() { println!(\"hi\"); }").unwrap();
        let binary = directory.join("blob.dat");
        std::fs::write(&binary, [0u8; 8192]).unwrap();
        let image = directory.join("demo image.png");
        std::fs::write(&image, b"png placeholder").unwrap();
        let missing = directory.join("missing.rs");

        // 纯文本(无文件):原样字符串。
        assert_eq!(message_content("普通消息"), json!("普通消息"));
        // 文本文件:内容以 fenced block 内联进 text。
        let inlined = message_content(&format!("看看这个文件 {}", source.display()));
        let pieces = inlined.as_array().unwrap();
        assert_eq!(pieces[0]["type"], "text");
        let text = pieces[0]["text"].as_str().unwrap();
        assert!(text.contains("```rust"), "应有 rust fence:{text}");
        assert!(text.contains("fn main()"));
        // 图片路径由公共消息层识别；反引号保留带空格路径并编码成标准 image_url part。
        let multimodal = message_content(&format!("请描述 `{}`。", image.display()));
        let pieces = multimodal.as_array().unwrap();
        assert_eq!(pieces[0]["type"], "image_url");
        assert_eq!(pieces[0]["image_url"]["url"], image.to_string_lossy().as_ref());
        assert!(!pieces[1]["text"].as_str().unwrap().contains("demo image.png"));
        // macOS/聊天界面常把路径包成中文弯引号；引用符和路径都不能泄漏给文本模型。
        let multimodal = message_content(&format!("> “{}”", image.display()));
        let pieces = multimodal.as_array().unwrap();
        assert_eq!(pieces[0]["type"], "image_url");
        assert_eq!(pieces[0]["image_url"]["url"], image.to_string_lossy().as_ref());
        assert!(!pieces[1]["text"].as_str().unwrap().contains("demo image.png"));
        assert_eq!(pieces[1]["text"], "请描述这张图片");
        // 二进制(全零)与不存在的路径:不内联、原样字符串。
        assert_eq!(message_content(&format!("{} {}", binary.display(), missing.display())), json!(format!("{} {}", binary.display(), missing.display())));
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn minicpm5_稠密gqa按层数线性反解() {
        // MiniCPM5-1B:24 层 GQA,2 KV 头 × head_dim 128,Q8g64 每层 528 B/token。
        let model = DetectedModel::MiniCpm5 { layers: 24, kv_heads: 2, head_dim: 128, context_limit: 131_072 };
        let (seq, per_token) = solve_max_seq_len(&model, 512 * 1024 * 1024).expect("512MiB 必须能反解");
        assert_eq!(per_token, 24 * 528);
        assert_eq!(seq % 1024, 0);
        // 512MiB / 12672 B/token ≈ 42k,远低于 131072 上限,不应被 clamp。
        assert!((40_000..44_000).contains(&seq));
    }

    #[test]
    fn gemma4_滑窗层封顶在窗口而全局层线性增长() {
        // 40 层滑窗(窗口 512)+ 8 层全局;10GiB 预算下全局层线性项足以顶到 131072 上限。
        let hybrid = DetectedModel::Gemma4 { layers: 48, kv_heads: 8, head_dim: 256, sliding_window: 512, sliding_layers: 40, context_limit: 131_072 };
        let (seq, _) = solve_max_seq_len(&hybrid, 10 << 30).expect("hybrid 必须能反解");
        assert_eq!(seq, 131_072);
        // metadata 缺层型时按全层满容量保守:同样预算下解出的上下文显著更小。
        let conservative = DetectedModel::Gemma4 { layers: 48, kv_heads: 8, head_dim: 256, sliding_window: 0, sliding_layers: 0, context_limit: 131_072 };
        let (seq_plain, _) = solve_max_seq_len(&conservative, 10 << 30).expect("保守口径必须能反解");
        assert!(seq_plain < seq && seq_plain % 1024 == 0);
    }

    #[test]
    fn gemma4_mlx按checkpoint配置识别() {
        let directory = std::env::temp_dir().join(format!("zllm-metal-gemma4-mlx-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&json!({
                "quantization": { "mode": "affine", "bits": 4, "group_size": 64 },
                "text_config": { "hidden_size": 2560, "num_hidden_layers": 42 }
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(directory.join("model.safetensors"), []).unwrap();

        let detected = detect(&directory).expect("Gemma4 MLX 应按 config.json 识别");
        assert!(matches!(detected, DetectedModel::Gemma4 { layers: 42, context_limit: 131_072, .. }));
        let _ = std::fs::remove_dir_all(directory);
    }
}
