//! 进程内嵌入式推理入口。
//!
//! 该入口不启动 HTTP、Scheduler 或 iroh。请求仍使用与服务模式相同的结构化
//! OpenAI JSON 语义，由具体模型 runtime 完成 chat template、tokenize、session
//! resume 与 terminal KV cache 管理。
//!
//! 各模型 runtime 与 Node adapter 共享同一条生成路径；新增模型时只在
//! [`Engine::load`] 增加平台组合分支。

use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use serde_json::Value;

use crate::config::{EmbeddedConfig, NodeBackendConfig, NodeModelConfig};

/// session 与 terminal KV cache 的进程内生命周期配置。
#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub cache_directory: std::path::PathBuf,
    /// 对支持 SSD snapshot 的模型与 Node 语义一致：关闭时仅在当前进程内复用，
    /// 开启时在换出/优雅关闭写入并可从 fjall 恢复。
    pub persist_kv_cache: bool,
    pub resident_cache_entries: usize,
}

/// 一次嵌入式生成的终态信息。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationResult {
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// 可供后续同一对话请求命中的 terminal cache 标识。
    pub cache_id: Option<String>,
    /// 模型 runtime 已解析的工具调用，与 Node 返回语义一致。
    pub tool_calls: Vec<crate::runtime::session::ToolCall>,
}

pub enum GenerationEvent<'a> {
    Token { token_id: u32, text: &'a str },
    Text { text: &'a str },
}

/// 库调用方可用于 admission 与诊断的 KV resident 资源快照。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvResourceReport {
    /// 模型、MTP/replay、视觉权重与长期 scratch 的当前常驻量。
    pub engine_resident_bytes: usize,
    pub capacity_bytes: usize,
    pub active_bytes: usize,
    pub resident_bytes: usize,
    pub available_bytes: usize,
    /// 一个固定整块 session 的 KV/recurrent/draft cache 常驻量。
    pub session_resident_bytes: usize,
}

/// 可由另一线程触发的取消句柄。
#[derive(Clone, Default)]
pub struct Cancellation {
    cancelled: Arc<AtomicBool>,
}

impl Cancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// 进程内模型引擎。当前支持 Gemma4 / Qwen3.6-3.8 / Ornith / Mistral / MiniCPM5 的 Metal 组合，
/// 后续模型按相同 runtime 边界接入。
pub struct Engine {
    inner: EngineInner,
}

enum EngineInner {
    #[cfg(target_os = "macos")]
    Gemma4(Box<crate::runtime::gemma4::engine::Gemma4Engine>),
    #[cfg(target_os = "macos")]
    Qwen36(Box<crate::runtime::qwen36::engine::Qwen36Engine>),
    #[cfg(target_os = "macos")]
    MiniCpm5(Box<crate::runtime::minicpm5::engine::MiniCpm5Engine>),
    #[cfg(target_os = "macos")]
    Ornith(Box<crate::runtime::ornith::node::OrnithEngine>),
    #[cfg(target_os = "macos")]
    Mistral(Box<crate::runtime::mistral::node::MistralEngine>),
}

impl Engine {
    /// 从嵌入式 YAML 加载引擎。该配置只包含 session、model 与 backend，
    /// 不接受 HTTP、artifact、iroh 等服务宿主字段。
    pub fn from_config(path: impl AsRef<Path>) -> Result<Self, String> {
        let config = EmbeddedConfig::load(path.as_ref()).map_err(|error| error.to_string())?;
        Self::load(config.model, config.backend, SessionConfig { cache_directory: config.session.cache_directory, persist_kv_cache: config.session.persist_kv_cache, resident_cache_entries: config.session.resident_cache_entries })
    }

    /// 用已经解析并校验过的模型、backend 与 session 配置装载引擎。
    pub fn load(model: NodeModelConfig, backend: NodeBackendConfig, session: SessionConfig) -> Result<Self, String> {
        if session.resident_cache_entries == 0 {
            return Err("session.resident_cache_entries 必须大于 0".to_owned());
        }
        #[cfg(target_os = "macos")]
        let (runtime, compute_steps) = (Arc::new(Mutex::new(crate::runtime::session::RuntimeStatus::default())), Arc::new(crate::runtime::session::AtomicCounterU64::new(0)));
        match (model, backend) {
            (NodeModelConfig::Gemma4(model), NodeBackendConfig::Metal(metal)) => {
                #[cfg(target_os = "macos")]
                {
                    let engine = crate::runtime::gemma4::engine::Gemma4Engine::load(
                        &model.weights_directory,
                        model.max_sequence_length,
                        model.execution,
                        metal.replay,
                        model.lm_head_quantization,
                        session.cache_directory,
                        session.persist_kv_cache,
                        session.resident_cache_entries,
                        runtime,
                        compute_steps,
                    )
                    .map_err(|error| error.to_string())?;
                    Ok(Self { inner: EngineInner::Gemma4(Box::new(engine)) })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (model, metal, session);
                    Err("Gemma4 Metal 嵌入式引擎只支持 macOS".to_owned())
                }
            }
            (NodeModelConfig::Gemma4(_), _) => Err("Gemma4 嵌入式引擎第一阶段只支持 Metal backend".to_owned()),
            (NodeModelConfig::Qwen36(model), NodeBackendConfig::Metal(metal)) => {
                #[cfg(target_os = "macos")]
                {
                    let engine = crate::runtime::qwen36::engine::Qwen36Engine::load(
                        &model.weights_directory,
                        model.max_sequence_length,
                        model.variant,
                        model.execution,
                        metal.replay,
                        model.lm_head_quantization,
                        session.cache_directory,
                        session.persist_kv_cache,
                        session.resident_cache_entries,
                        runtime,
                        compute_steps,
                    )
                    .map_err(|error| error.to_string())?;
                    Ok(Self { inner: EngineInner::Qwen36(Box::new(engine)) })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (model, metal, session);
                    Err("Qwen3.6/Qwen3.8 Metal 嵌入式引擎只支持 macOS".to_owned())
                }
            }
            (NodeModelConfig::Qwen36(_), _) => Err("Qwen3.6/Qwen3.8 嵌入式引擎第一阶段只支持 Metal backend".to_owned()),
            (NodeModelConfig::MiniCpm5(model), NodeBackendConfig::Metal(metal)) => {
                #[cfg(target_os = "macos")]
                {
                    if session.persist_kv_cache || session.resident_cache_entries != 1 {
                        return Err("MiniCPM5 embedded 尚未实现 SSD terminal snapshot，仅支持 persist_kv_cache=false 且 resident_cache_entries=1".to_owned());
                    }
                    let engine = crate::runtime::minicpm5::engine::MiniCpm5Engine::load(
                        &model.weights_directory,
                        model.max_sequence_length,
                        model.execution.kv_cache_format == crate::config::KvCacheFormat::F16,
                        metal.replay,
                        model.lm_head_quantization,
                        runtime,
                        compute_steps,
                    )
                    .map_err(|error| error.to_string())?;
                    Ok(Self { inner: EngineInner::MiniCpm5(Box::new(engine)) })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (model, metal, session);
                    Err("MiniCPM5 Metal 嵌入式引擎只支持 macOS".to_owned())
                }
            }
            (NodeModelConfig::MiniCpm5(_), _) => Err("MiniCPM5 嵌入式引擎第一阶段只支持 Metal backend".to_owned()),
            (NodeModelConfig::Ornith(model), NodeBackendConfig::Metal(metal)) => {
                #[cfg(target_os = "macos")]
                {
                    if session.persist_kv_cache {
                        return Err("Ornith embedded 尚未实现 SSD terminal snapshot，仅支持 persist_kv_cache=false".to_owned());
                    }
                    let mut options = crate::runtime::ornith::options::OrnithOptions::from(model.execution);
                    options.terminal_cache_entries = session.resident_cache_entries;
                    let engine =
                        crate::runtime::ornith::node::OrnithEngine::load(&model.weights_directory, model.max_sequence_length, options, metal.replay, model.lm_head_quantization, runtime, compute_steps).map_err(|error| error.to_string())?;
                    Ok(Self { inner: EngineInner::Ornith(Box::new(engine)) })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (model, metal, session);
                    Err("Ornith Metal 嵌入式引擎只支持 macOS".to_owned())
                }
            }
            (NodeModelConfig::Ornith(_), _) => Err("Ornith 嵌入式引擎当前只支持 Metal backend".to_owned()),
            (NodeModelConfig::Mistral(model), NodeBackendConfig::Metal(metal)) => {
                #[cfg(target_os = "macos")]
                {
                    if session.persist_kv_cache {
                        return Err("Mistral embedded 尚未实现 terminal cache，仅支持 persist_kv_cache=false".to_owned());
                    }
                    let kv_f16 = model.execution.kv_cache_format == crate::config::KvCacheFormat::F16;
                    let engine =
                        crate::runtime::mistral::node::MistralEngine::load(&model.weights_directory, model.max_sequence_length, kv_f16, metal.replay, model.lm_head_quantization, runtime, compute_steps).map_err(|error| error.to_string())?;
                    Ok(Self { inner: EngineInner::Mistral(Box::new(engine)) })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (model, metal, session);
                    Err("Mistral Metal 嵌入式引擎只支持 macOS".to_owned())
                }
            }
            (NodeModelConfig::Mistral(_), _) => Err("Mistral 嵌入式引擎当前只支持 Metal backend".to_owned()),
            (_, _) => Err("该模型尚未接入嵌入式引擎".to_owned()),
        }
    }

    /// 创建一个独立取消句柄；把它传给 `generate` 后可从另一线程终止生成。
    pub fn cancellation(&self) -> Cancellation {
        Cancellation::default()
    }

    pub fn kv_resources(&self) -> KvResourceReport {
        let residency = match &self.inner {
            #[cfg(target_os = "macos")]
            EngineInner::Gemma4(engine) => engine.kv_residency(),
            #[cfg(target_os = "macos")]
            EngineInner::Qwen36(engine) => engine.kv_residency(),
            #[cfg(target_os = "macos")]
            EngineInner::MiniCpm5(engine) => engine.kv_residency(),
            #[cfg(target_os = "macos")]
            EngineInner::Ornith(engine) => engine.kv_residency(),
            #[cfg(target_os = "macos")]
            EngineInner::Mistral(engine) => engine.kv_residency(),
        };
        KvResourceReport {
            engine_resident_bytes: residency.engine_resident_bytes,
            capacity_bytes: residency.capacity_bytes,
            active_bytes: residency.active_bytes,
            resident_bytes: residency.resident_bytes,
            available_bytes: residency.available_bytes(),
            session_resident_bytes: residency.session_resident_bytes,
        }
    }

    /// 生成并流式返回 token。`request` 与 `/v1/chat/completions` 请求体保持同一
    /// 结构语义；回调返回 false 会以 cancelled 结束。
    pub fn generate(&mut self, request: &Value, cancellation: &Cancellation, mut on_token: impl FnMut(u32, &str) -> bool) -> Result<GenerationResult, String> {
        let summary = match &mut self.inner {
            #[cfg(target_os = "macos")]
            EngineInner::Gemma4(engine) => engine.generate(request, &cancellation.cancelled, &mut |token, text| on_token(token.unwrap_or(0), &text))?,
            #[cfg(target_os = "macos")]
            EngineInner::Qwen36(engine) => engine.generate(request, &cancellation.cancelled, &mut |token, text| on_token(token.unwrap_or(0), &text))?,
            #[cfg(target_os = "macos")]
            EngineInner::MiniCpm5(engine) => engine.generate(request, &cancellation.cancelled, &mut |token, text| on_token(token.unwrap_or(0), &text))?,
            #[cfg(target_os = "macos")]
            EngineInner::Ornith(engine) => {
                let scope = crate::runtime::tool::request_tool_scope("ornith", request);
                engine.generate(&scope, request, &cancellation.cancelled, &mut |token, text| on_token(token.unwrap_or(0), &text))?
            }
            #[cfg(target_os = "macos")]
            EngineInner::Mistral(engine) => engine.generate("embedded", request, &cancellation.cancelled, &mut |token, text| on_token(token.unwrap_or(0), &text))?,
        };
        Ok(GenerationResult { finish_reason: summary.finish_reason, prompt_tokens: summary.prompt_tokens, completion_tokens: summary.completion_tokens, cache_id: summary.cache.map(|cache| cache.cache_id), tool_calls: summary.tool_calls })
    }

    /// 无歧义的流式接口：decoder flush 文本不伪装成词表 token 0。
    pub fn generate_events(&mut self, request: &Value, cancellation: &Cancellation, mut on_event: impl FnMut(GenerationEvent<'_>) -> bool) -> Result<GenerationResult, String> {
        let summary = match &mut self.inner {
            #[cfg(target_os = "macos")]
            EngineInner::Gemma4(engine) => engine.generate(request, &cancellation.cancelled, &mut |token, text| match token {
                Some(token_id) => on_event(GenerationEvent::Token { token_id, text: &text }),
                None => on_event(GenerationEvent::Text { text: &text }),
            })?,
            #[cfg(target_os = "macos")]
            EngineInner::Qwen36(engine) => engine.generate(request, &cancellation.cancelled, &mut |token, text| match token {
                Some(token_id) => on_event(GenerationEvent::Token { token_id, text: &text }),
                None => on_event(GenerationEvent::Text { text: &text }),
            })?,
            #[cfg(target_os = "macos")]
            EngineInner::MiniCpm5(engine) => engine.generate(request, &cancellation.cancelled, &mut |token, text| match token {
                Some(token_id) => on_event(GenerationEvent::Token { token_id, text: &text }),
                None => on_event(GenerationEvent::Text { text: &text }),
            })?,
            #[cfg(target_os = "macos")]
            EngineInner::Ornith(engine) => {
                let scope = crate::runtime::tool::request_tool_scope("ornith", request);
                engine.generate(&scope, request, &cancellation.cancelled, &mut |token, text| match token {
                    Some(token_id) => on_event(GenerationEvent::Token { token_id, text: &text }),
                    None => on_event(GenerationEvent::Text { text: &text }),
                })?
            }
            #[cfg(target_os = "macos")]
            EngineInner::Mistral(engine) => engine.generate("embedded", request, &cancellation.cancelled, &mut |token, text| match token {
                Some(token_id) => on_event(GenerationEvent::Token { token_id, text: &text }),
                None => on_event(GenerationEvent::Text { text: &text }),
            })?,
        };
        Ok(GenerationResult { finish_reason: summary.finish_reason, prompt_tokens: summary.prompt_tokens, completion_tokens: summary.completion_tokens, cache_id: summary.cache.map(|cache| cache.cache_id), tool_calls: summary.tool_calls })
    }

    /// 显式执行可失败的优雅关闭。需要确认 SSD cache 已写回的调用方应使用它，
    /// `Drop` 只作为兜底并把错误写入日志。
    pub fn shutdown(&mut self) -> Result<(), String> {
        match &mut self.inner {
            #[cfg(target_os = "macos")]
            EngineInner::Gemma4(engine) => engine.shutdown(),
            #[cfg(target_os = "macos")]
            EngineInner::Qwen36(engine) => engine.shutdown(),
            #[cfg(target_os = "macos")]
            EngineInner::MiniCpm5(engine) => engine.shutdown(),
            #[cfg(target_os = "macos")]
            EngineInner::Ornith(engine) => engine.shutdown(),
            #[cfg(target_os = "macos")]
            EngineInner::Mistral(engine) => engine.shutdown(),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let result = self.shutdown();
        if let Err(error) = result {
            eprintln!("[zllm-embedded] 优雅退出持久化失败: {error}");
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod real_model_tests {
    use super::*;
    use serde_json::json;

    /// 真实 `zllm::Engine` 回归：不启动 server，直接从 embedded YAML
    /// 装载并跑一次结构化请求。没有本地模型环境变量时自动跳过。
    #[test]
    fn embedded_gemma4_real_request() {
        let Some(config) = std::env::var_os("ZLLM_EMBEDDED_GEMMA4_CONFIG").map(std::path::PathBuf::from) else { return };
        let prompt = std::env::var_os("ZLLM_EMBEDDED_PROMPT_FILE").map(std::path::PathBuf::from).map(|path| std::fs::read_to_string(path).expect("读取 embedded prompt")).unwrap_or_else(|| "Reply with one short sentence.".to_owned());
        let max_tokens = std::env::var("ZLLM_EMBEDDED_MAX_TOKENS").ok().and_then(|value| value.parse::<usize>().ok()).filter(|value| *value > 0).unwrap_or(8);
        let mut engine = Engine::from_config(config).expect("加载 embedded Gemma4");
        let request = json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0
        });
        let mut output = String::new();
        let cancellation = engine.cancellation();
        let result = engine
            .generate(&request, &cancellation, |_, text| {
                output.push_str(text);
                true
            })
            .expect("embedded Gemma4 生成");
        println!("[embedded-real] prompt={} completion={} finish={} cache={:?} output={output:?}", result.prompt_tokens, result.completion_tokens, result.finish_reason, result.cache_id);
        assert!(result.prompt_tokens > 0);
        assert!(result.completion_tokens > 0);
    }
}
