//! Mistral × Metal 节点适配器：组合平台无关运行时、设备资源与节点请求协议。
//!
//! Mistral-Small-3.2-24B-Instruct-2506 Q6_K GGUF 在 M5 26GB 上的内存预算：
//! 权重 18 GB + KV (Q8G64) ~2 GB + workspace ~0.5 GB ≈ 20.5 GB。
//! 因此 `max_sequence_length` 必须限制（131k 全开会爆），standalone yaml 默认 8k。

#[cfg(target_os = "macos")]
use crate::runtime::session::{FixedSessionResidency, GenerationSummary};
#[cfg(not(target_os = "macos"))]
use crate::{config::MistralNodeModelConfig, server::node::DynError};
#[cfg(target_os = "macos")]
use crate::{
    config::{MistralNodeModelConfig, NodeMetalBackendConfig},
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::session::{AtomicCounterU64, NodeCapabilities, RuntimeStatus as NodeRuntime},
    server::node::{DynError, NodeEngine},
};
#[cfg(target_os = "macos")]
use serde_json::Value;
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(target_os = "macos")]
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

#[cfg(target_os = "macos")]
use crate::{
    attention::gqa::CausalWindow,
    backend::{
        Backend,
        metal::{MetalContext, MetalKvCache, MetalTensor, MetalWeight},
    },
    kv_cache::{KvCacheLayerMap, KvCacheSpec},
    runtime::mistral::{self, MistralConfig, MistralOutputHead, MistralTextLayer, MistralWeights},
    tokenizer::{Detokenizer, Tokenizer},
};

#[cfg(target_os = "macos")]
pub async fn run(model: MistralNodeModelConfig, backend: NodeMetalBackendConfig, config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    let model_path = model.weights_directory;
    let max_seq_len = model.max_sequence_length;
    let kv_f16 = model.execution.kv_cache_format == crate::config::KvCacheFormat::F16;
    let lm_head_quantization = model.lm_head_quantization;
    let factory = Box::new(move |runtime, compute_steps| MistralEngine::load(&model_path, max_seq_len, kv_f16, backend.replay, lm_head_quantization, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

#[cfg(not(target_os = "macos"))]
pub async fn run(_model: MistralNodeModelConfig, _backend: crate::config::NodeMetalBackendConfig, _config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    Err("Mistral Metal Node 需要 macOS".into())
}

#[cfg(target_os = "macos")]
#[path = "metal_session.rs"]
mod metal_session;
#[cfg(target_os = "macos")]
use metal_session::MistralMetalSession;

/// Mistral NodeEngine 主体：Metal session + compute/runtime handles。
#[cfg(target_os = "macos")]
pub struct MistralEngine {
    session: MistralMetalSession,
    residency: FixedSessionResidency,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
}

#[cfg(target_os = "macos")]
impl MistralEngine {
    pub fn load(
        model_path: &Path,
        max_seq_len: usize,
        kv_f16: bool,
        replay_enabled: bool,
        lm_head_quantization: crate::weight::LmHeadQuantization,
        runtime: Arc<Mutex<NodeRuntime>>,
        compute_steps: Arc<AtomicCounterU64>,
    ) -> Result<Self, DynError> {
        let session = MistralMetalSession::load_with_replay(model_path, max_seq_len, kv_f16, replay_enabled, lm_head_quantization).map_err(|error| -> DynError { error.into() })?;
        let session_resident_bytes = session.session_residency_bytes().map_err(|error| -> DynError { error.into() })?;
        let available = crate::backend::metal::available_residency_bytes(session.context()) as usize;
        let engine_resident_bytes = session.context().device.current_allocated_size() as usize;
        let residency = FixedSessionResidency::new(available, engine_resident_bytes, session_resident_bytes)?;
        let mut capabilities = crate::runtime::metal_node::text_capabilities(
            session.context(),
            crate::runtime::node::SessionDescriptor { model_format: "gguf-mixed", model_bytes: session.model_bytes(), max_seq_len, kv_cache_format: if kv_f16 { "f16" } else { "q8g64" }, input_modalities: &["text"] },
        );
        residency.configure(&mut capabilities, &runtime);
        eprintln!("[mistral-kv-admission] available={:.1} MiB session={:.1} MiB", available as f64 / 1048576.0, session_resident_bytes as f64 / 1048576.0);
        Ok(Self { session, residency, capabilities, runtime, compute_steps })
    }
}

#[cfg(target_os = "macos")]
impl NodeEngine for MistralEngine {
    fn model_key(&self) -> &'static str {
        "mistral"
    }
    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        crate::runtime::metal_node::startup_info(self.session.context(), &self.capabilities)
    }
    fn refresh_runtime(&self) {
        self.residency.refresh_without_terminal(&self.runtime);
    }
    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        Vec::new()
    }
    fn max_concurrency(&self) -> usize {
        1
    }
    fn generate_one(&mut self, request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        self.generate(request_id, request, cancellation, &mut |token, text| on_token(token.unwrap_or(0), text))
    }
}

#[cfg(target_os = "macos")]
impl MistralEngine {
    pub fn generate(&mut self, _request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(Option<u32>, String) -> bool) -> Result<GenerationSummary, String> {
        use crate::runtime::session::{BatchTokenGuard, GenerationOutput, parse_stops, requested_completion_tokens};
        let prompt = mistral::mistral_request_prompt(request)?;
        let tokens = self.session.tokenize(&prompt);
        let max_completion = requested_completion_tokens(request);
        if max_completion == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        if tokens.is_empty() {
            return Err("Mistral prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.session.max_seq_len() {
            return Err(format!("Mistral prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.session.max_seq_len()));
        }
        let stops = parse_stops(request.get("stop"))?;
        let _batch_guard = BatchTokenGuard::new(&self.runtime, tokens.len());
        let _kv_reservation = self.residency.reserve().map_err(|error| format!("Mistral {error}; 请减小 max_sequence_length 后重试"))?;
        let mut sequence = self.session.prefill(tokens.clone())?;
        let max_tokens = max_completion.min(self.session.max_seq_len() - sequence.token_count());
        if max_tokens == 0 {
            return Err("max_tokens 必须大于 0".to_owned());
        }
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
            let bytes = self.session.decode_bytes(token)?;
            if !output.push(&bytes, |chunk| on_token(Some(token), chunk)) {
                break;
            }
            if step + 1 == max_tokens {
                break;
            }
            self.session.decode_token(&mut sequence, token)?;
            self.compute_steps.fetch_add(1);
        }
        output.finish(|chunk| on_token(None, chunk));
        Ok(output.summary(tokens.len()))
    }

    pub(crate) fn kv_residency(&self) -> crate::runtime::session::KvResidency {
        self.residency.report_without_terminal()
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        Ok(())
    }
}
