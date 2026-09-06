//! Ornith × Metal 节点适配器：组合模型 runtime、设备资源与节点请求协议。
#[cfg(target_os = "macos")]
use super::options::OrnithOptions;
#[cfg(target_os = "macos")]
use super::protocol::{chat_prompt, chat_prompt_suffix};
#[cfg(target_os = "macos")]
use crate::runtime::session::{BatchTokenGuard, FixedSessionResidency, GenerationOutput, GenerationSummary, parse_stops, terminal_cache_id};
#[cfg(target_os = "macos")]
use crate::runtime::tool::{RequestToolCallStream, ToolDialect, emit_request_tool_chunk, finish_request_tool_stream};
#[cfg(not(target_os = "macos"))]
use crate::{config::OrnithNodeModelConfig, server::node::DynError};
#[cfg(target_os = "macos")]
use crate::{
    config::{NodeMetalBackendConfig, OrnithNodeModelConfig},
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::session::{AtomicCounterU64, NodeCapabilities, RuntimeStatus as NodeRuntime},
    server::node::{DynError, NodeEngine},
};
#[cfg(target_os = "macos")]
use serde_json::Value;
#[cfg(target_os = "macos")]
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
#[cfg(target_os = "macos")]
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(target_os = "macos")]
pub async fn run(model: OrnithNodeModelConfig, backend: NodeMetalBackendConfig, config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    let model_path = model.weights_directory;
    let max_seq_len = model.max_sequence_length;
    let lm_head_quantization = model.lm_head_quantization;
    let options = OrnithOptions::from(model.execution);
    let factory = Box::new(move |runtime, compute_steps| OrnithEngine::load(&model_path, max_seq_len, options, backend.replay, lm_head_quantization, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}
#[cfg(not(target_os = "macos"))]
pub async fn run(_model: OrnithNodeModelConfig, _backend: crate::config::NodeMetalBackendConfig, _config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    Err("Ornith Metal Node 需要 macOS".into())
}
#[cfg(target_os = "macos")]
#[path = "metal_session.rs"]
mod metal_session;
#[cfg(target_os = "macos")]
use metal_session::{OrnithMetalSequence, OrnithMetalSession};

#[cfg(target_os = "macos")]
struct OrnithTerminalState {
    sequence: OrnithMetalSequence,
    pending_tokens: Vec<u32>,
    info: CacheInfo,
}
#[cfg(target_os = "macos")]
impl crate::kv_cache::terminal_cache::TerminalSnapshot for OrnithTerminalState {
    type Resources = ();

    fn encode(&self) -> Result<Vec<u8>, String> {
        Err("Ornith 终点快照落盘未实现".to_owned())
    }

    fn decode(_bytes: &[u8], _resources: &Self::Resources) -> Result<Self, String> {
        Err("Ornith 终点快照落盘未实现".to_owned())
    }

    fn terminal_tokens(&self) -> &[u32] {
        self.sequence.tokens()
    }

    fn info(&self) -> &CacheInfo {
        &self.info
    }

    fn set_info(&mut self, info: CacheInfo) {
        self.info = info;
    }
}
#[cfg(target_os = "macos")]
pub struct OrnithEngine {
    session: OrnithMetalSession,
    terminal_states: crate::kv_cache::terminal_cache::TerminalSessions<OrnithTerminalState>,
    residency: FixedSessionResidency,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
}
#[cfg(target_os = "macos")]
impl OrnithEngine {
    pub fn load(
        model_path: &Path,
        max_seq_len: usize,
        options: OrnithOptions,
        replay_enabled: bool,
        lm_head_quantization: crate::weight::LmHeadQuantization,
        runtime: Arc<Mutex<NodeRuntime>>,
        compute_steps: Arc<AtomicCounterU64>,
    ) -> Result<Self, DynError> {
        let mut session = OrnithMetalSession::load_with_replay(model_path, max_seq_len, options, replay_enabled, lm_head_quantization).map_err(|error| -> DynError { error.into() })?;
        // 非 lazy expert 是引擎常驻资源；必须在建立 session admission 快照前装载。
        if let Some((experts, bytes)) = session.preload_experts().map_err(|error| -> DynError { error.into() })? {
            eprintln!("[ornith] expert 常驻完成 experts={experts} resident_gib={:.2}", bytes as f64 / (1024.0 * 1024.0 * 1024.0));
        }
        let session_resident_bytes = session.session_residency_bytes().map_err(|error| -> DynError { error.into() })?;
        let available = crate::backend::metal::available_residency_bytes(session.context()) as usize;
        let engine_resident_bytes = session.context().device.current_allocated_size() as usize;
        let residency = FixedSessionResidency::new(available, engine_resident_bytes, session_resident_bytes)?;
        let mut capabilities = crate::runtime::metal_node::text_capabilities(
            session.context(),
            crate::runtime::node::SessionDescriptor { model_format: "gguf-mixed", model_bytes: session.model_bytes(), max_seq_len, kv_cache_format: if options.kv_f16 { "f16" } else { "q8g64" }, input_modalities: &["text"] },
        );
        residency.configure(&mut capabilities, &runtime);
        let terminal_limit = options.terminal_cache_entries;
        eprintln!("[ornith-kv-admission] available={:.1} MiB session={:.1} MiB", available as f64 / 1048576.0, session_resident_bytes as f64 / 1048576.0);
        Ok(Self { session, terminal_states: crate::kv_cache::terminal_cache::TerminalSessions::new(terminal_limit, None), residency, capabilities, runtime, compute_steps })
    }
}
#[cfg(target_os = "macos")]
impl NodeEngine for OrnithEngine {
    fn model_key(&self) -> &'static str {
        "ornith"
    }
    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        crate::runtime::metal_node::startup_info(self.session.context(), &self.capabilities)
    }
    fn refresh_runtime(&self) {
        self.residency.refresh(&self.runtime, &self.terminal_states);
        if let Ok(mut runtime) = self.runtime.lock() {
            runtime.resident_expert_bytes = self.session.resident_expert_bytes();
        }
    }
    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        self.terminal_states.infos()
    }
    fn terminal_cache_pins(&self) -> Option<std::sync::Arc<std::sync::Mutex<std::collections::HashSet<String>>>> {
        Some(self.terminal_states.pin_handle())
    }
    fn max_concurrency(&self) -> usize {
        1
    }
    fn generate_one(&mut self, request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        self.generate(request_id, request, cancellation, &mut |token, text| on_token(token.unwrap_or(0), text))
    }
}

#[cfg(target_os = "macos")]
impl OrnithEngine {
    pub fn generate(&mut self, request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(Option<u32>, String) -> bool) -> Result<GenerationSummary, String> {
        let prompt = chat_prompt(request)?;
        let tokens = self.session.tokenize(&prompt);
        let requested_tokens = crate::runtime::session::requested_completion_tokens(request);
        if requested_tokens == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        if tokens.is_empty() {
            return Err("Ornith prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.session.max_seq_len() {
            return Err(format!("Ornith prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.session.max_seq_len()));
        }
        let stops = parse_stops(request.get("stop"))?;
        let (resumed, _kv_reservation) = crate::runtime::session::activate_terminal_append(
            &mut self.terminal_states,
            self.residency.budget(),
            self.residency.session_resident_bytes(),
            request,
            |assistant| Ok(self.session.tokenize(&chat_prompt_suffix(request, assistant)?)),
            &(),
        )
        .map_err(|error| format!("Ornith {error}; 请重新开始会话"))?;
        // resume 命中时只对 suffix 计费，避免 dispatch 把整段 prompt 算到当前 batch。
        let batch_tokens = resumed.as_ref().map_or(tokens.len(), |(state, suffix)| state.pending_tokens.len().saturating_add(suffix.len()));
        let _batch_guard = BatchTokenGuard::new(&self.runtime, batch_tokens);
        let mut sequence = if let Some((state, suffix)) = resumed {
            let OrnithTerminalState { mut sequence, pending_tokens, .. } = state;
            let mut append = pending_tokens;
            append.extend(suffix);
            self.session.extend(&mut sequence, &append)?;
            sequence
        } else {
            self.session.prefill(tokens.clone())?
        };
        if sequence.token_count() >= self.session.max_seq_len() {
            return Err(format!("Ornith 会话状态 {} tokens 超过 max_seq_len {}", sequence.token_count(), self.session.max_seq_len()));
        }
        let max_tokens = requested_tokens.min(self.session.max_seq_len() - sequence.token_count());
        if max_tokens == 0 {
            return Err("max_tokens 必须大于 0".to_owned());
        }
        let mut decoder = self.session.begin_decode()?;
        let mut response_text = String::new();
        let mut tool_stream = RequestToolCallStream::new(request, request_id, ToolDialect::ChatmlJson);
        let mut pending_token = None;
        let result: Result<GenerationSummary, String> = (|| {
            let mut output = GenerationOutput::new(&stops);
            for step in 0..max_tokens {
                if cancellation.load(Ordering::Acquire) {
                    output.cancel();
                    break;
                }
                let token = self.session.next_token(&sequence)?;
                if self.session.is_eos(token) {
                    output.stop();
                    break;
                }
                pending_token = Some(token);
                let bytes = self.session.decode_bytes(token)?;
                if !output.push(&bytes, |chunk| emit_request_tool_chunk(&mut tool_stream, Some(token), &chunk, &mut response_text, on_token)) {
                    if output.finish_reason() == "stop" {
                        pending_token = None;
                    }
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                self.session.decode_token(&mut sequence, &mut decoder, token)?;
                // 无锁递增：避免每个 decode step 与 heartbeat / refresh 抢 Mutex。
                self.compute_steps.fetch_add(1);
                pending_token = None;
            }
            output.finish(|chunk| emit_request_tool_chunk(&mut tool_stream, None, &chunk, &mut response_text, on_token));
            if !output.is_cancelled() && !finish_request_tool_stream(&mut tool_stream, None, &mut response_text, on_token) {
                output.cancel();
            }
            if !output.is_cancelled() && !tool_stream.calls.is_empty() {
                output.mark_tool_calls();
            }
            Ok(output.summary(tokens.len()))
        })();
        self.session.finish_decode(decoder);
        let mut summary = result?;
        if summary.finish_reason != "cancelled" && stops.is_empty() {
            let cache_id = terminal_cache_id(request, &response_text, &tool_stream.calls)?;
            let info = CacheInfo {
                cache_id,
                model_key: "ornith".to_owned(),
                cache_format: "ornith-terminal-v1".to_owned(),
                last_layer: self.session.layer_count().saturating_sub(1),
                prompt_tokens: sequence.token_count(),
                bytes: self.residency.session_resident_bytes() as u64,
                modified_unix: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            };
            let state = OrnithTerminalState { sequence, pending_tokens: pending_token.into_iter().collect(), info: info.clone() };
            summary.cache = crate::runtime::session::retain_terminal_session(&mut self.terminal_states, state, info);
        }
        summary.tool_calls = std::mem::take(&mut tool_stream.calls);
        Ok(summary)
    }

    pub(crate) fn kv_residency(&self) -> crate::runtime::session::KvResidency {
        self.residency.report(&self.terminal_states)
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        Ok(())
    }
}
