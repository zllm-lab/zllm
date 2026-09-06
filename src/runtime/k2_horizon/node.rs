//! K2-Horizon × Metal 节点适配器：组合平台无关运行时、设备资源与节点请求协议。
//!
//! 36B-A4B IQ3_XS 权重约 14.6 GiB，24 GiB 统一内存机器需限制上下文长度。

#[cfg(target_os = "macos")]
use crate::runtime::session::{FixedSessionResidency, GenerationSummary};
#[cfg(not(target_os = "macos"))]
use crate::{config::K2HorizonNodeModelConfig, server::node::DynError};
#[cfg(target_os = "macos")]
use crate::{
    config::{K2HorizonNodeModelConfig, NodeMetalBackendConfig},
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
pub async fn run(model: K2HorizonNodeModelConfig, backend: NodeMetalBackendConfig, config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    let model_path = model.weights_directory;
    let max_seq_len = model.max_sequence_length;
    let kv_f16 = model.execution.kv_cache_format == crate::config::KvCacheFormat::F16;
    let expert_cache_gib = model.execution.expert_cache_gib;
    let lm_head_quantization = model.lm_head_quantization;
    let factory =
        Box::new(move |runtime, compute_steps| K2Engine::load(&model_path, max_seq_len, kv_f16, backend.replay, expert_cache_gib, lm_head_quantization, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

#[cfg(not(target_os = "macos"))]
pub async fn run(_model: K2HorizonNodeModelConfig, _backend: crate::config::NodeMetalBackendConfig, _config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    Err("K2-Horizon Metal Node 需要 macOS".into())
}

#[cfg(target_os = "macos")]
#[path = "metal_session.rs"]
mod metal_session;
#[cfg(target_os = "macos")]
use metal_session::K2MetalSession;

/// K2-Horizon NodeEngine 主体：Metal session + compute/runtime handles。
#[cfg(target_os = "macos")]
pub struct K2Engine {
    session: K2MetalSession,
    residency: FixedSessionResidency,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
}

#[cfg(target_os = "macos")]
impl K2Engine {
    pub fn load(
        model_path: &Path,
        max_seq_len: usize,
        kv_f16: bool,
        replay_enabled: bool,
        expert_cache_gib: usize,
        lm_head_quantization: crate::weight::LmHeadQuantization,
        runtime: Arc<Mutex<NodeRuntime>>,
        compute_steps: Arc<AtomicCounterU64>,
    ) -> Result<Self, DynError> {
        let mut session = K2MetalSession::load(model_path, max_seq_len, kv_f16, replay_enabled, expert_cache_gib, lm_head_quantization).map_err(|error| -> DynError { error.into() })?;
        let (expert_count, expert_bytes) = session.preload_experts().map_err(|error| -> DynError { error.into() })?;
        let session_resident_bytes = session.session_residency_bytes().map_err(|error| -> DynError { error.into() })?;
        let available = crate::backend::metal::available_residency_bytes(session.context()) as usize;
        let engine_resident_bytes = session.context().device.current_allocated_size() as usize;
        let residency = FixedSessionResidency::new(available, engine_resident_bytes, session_resident_bytes)?;
        let mut capabilities = crate::runtime::metal_node::text_capabilities(
            session.context(),
            crate::runtime::node::SessionDescriptor { model_format: "gguf-mixed", model_bytes: session.model_bytes(), max_seq_len, kv_cache_format: if kv_f16 { "f16" } else { "q8g64" }, input_modalities: &["text"] },
        );
        residency.configure(&mut capabilities, &runtime);
        eprintln!("[k2-horizon-load] experts={expert_count} expert_gib={:.2} available_mib={:.1} session_mib={:.1}", expert_bytes as f64 / 1073741824.0, available as f64 / 1048576.0, session_resident_bytes as f64 / 1048576.0);
        Ok(Self { session, residency, capabilities, runtime, compute_steps })
    }
}

#[cfg(target_os = "macos")]
impl NodeEngine for K2Engine {
    fn model_key(&self) -> &'static str {
        self.session.model_key()
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
impl K2Engine {
    pub fn generate(&mut self, _request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(Option<u32>, String) -> bool) -> Result<GenerationSummary, String> {
        use crate::runtime::session::{BatchTokenGuard, GenerationOutput, parse_stops, requested_completion_tokens};
        let prompt = self.session.render_request_prompt(request)?;
        let tokens = self.session.tokenize(&prompt);
        // 采样优先级:请求显式 temperature>0 > 官方推荐表(runtime::official_sampling,
        // 按模型固定)> 贪心。temperature==0 视为显式贪心。
        let mut sampling_state = match request.get("temperature").and_then(Value::as_f64) {
            Some(temperature) if temperature > 0.0 => {
                let top_p = request.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32;
                Some((temperature as f32, top_p))
            }
            Some(_) => None,
            None => crate::runtime::official_sampling(self.model_key()),
        }
        .map(|(temperature, top_p)| {
            // ZLLM_SAMPLING_SEED=0 固定序列;缺省按请求到达顺序散列,同 seed 可复现。
            let seed = std::env::var("ZLLM_SAMPLING_SEED").ok().and_then(|value| value.parse::<u64>().ok()).unwrap_or_else(|| {
                static REQUEST_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                REQUEST_COUNTER.fetch_add(0x9e37_79b9_7f4a_7c15, std::sync::atomic::Ordering::Relaxed)
            });
            crate::runtime::output::SamplingState::new(crate::runtime::output::SamplingConfig { temperature, top_p, seed }).map_err(|error| format!("K2-Horizon 采样配置: {error}"))
        })
        .transpose()?;
        let max_completion = requested_completion_tokens(request);
        if max_completion == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        if tokens.is_empty() {
            return Err("K2-Horizon prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.session.max_seq_len() {
            return Err(format!("K2-Horizon prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.session.max_seq_len()));
        }
        let stops = parse_stops(request.get("stop"))?;
        let _batch_guard = BatchTokenGuard::new(&self.runtime, tokens.len());
        let _kv_reservation = self.residency.reserve().map_err(|error| format!("K2-Horizon {error}; 请减小 max_sequence_length 后重试"))?;
        let prefill_started = std::time::Instant::now();
        let mut sequence = self.session.prefill(tokens.clone())?;
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();
        let max_tokens = max_completion.min(self.session.max_seq_len() - sequence.token_count());
        if max_tokens == 0 {
            return Err("max_tokens 必须大于 0".to_owned());
        }
        let mut decoder = self.session.begin_decode()?;
        let replay_decode = sampling_state.is_none() && self.session.replay_decode_available(sequence.token_count() + max_tokens);
        if replay_decode {
            self.session.ensure_replay(&sequence, &mut decoder)?;
        }
        let decode_started = std::time::Instant::now();
        let result = (|| {
            let mut output = GenerationOutput::new(&stops);
            let mut replay_token = None;
            for step in 0..max_tokens {
                if cancellation.load(Ordering::Acquire) {
                    output.cancel();
                    break;
                }
                let profile_step = std::env::var_os("ZLLM_K2_PROFILE").is_some();
                if profile_step {
                    self.session.context().reset_gpu_stats();
                }
                let output_started = std::time::Instant::now();
                let token = match replay_token.take() {
                    Some(token) => token,
                    None => match sampling_state.as_mut() {
                        Some(sampling) => {
                            let draw = sampling.next();
                            self.session.next_token_sampled(&sequence, &draw)?
                        }
                        None => self.session.next_token(&sequence)?,
                    },
                };
                if profile_step {
                    let gpu = self.session.context().gpu_stats();
                    eprintln!("[k2-horizon-output] step={step} wall={:.3}s gpu={:.3}s commands={}", output_started.elapsed().as_secs_f64(), gpu.seconds, gpu.command_buffers,);
                    for operator in self.session.context().gpu_profile().into_iter().take(10) {
                        eprintln!("  k2 output {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
                    }
                }
                if self.session.is_eos(token) {
                    output.stop();
                    break;
                }
                let bytes = self.session.decode_bytes(token)?;
                if !output.push(&bytes, |chunk| on_token(Some(token), chunk)) || step + 1 == max_tokens {
                    break;
                }
                if profile_step {
                    self.session.context().reset_gpu_stats();
                }
                let decode_started = std::time::Instant::now();
                if replay_decode {
                    replay_token = Some(self.session.replay_decode_token(&mut sequence, token)?);
                } else {
                    self.session.decode_token(&mut sequence, &mut decoder, token)?;
                }
                if profile_step {
                    let gpu = self.session.context().gpu_stats();
                    eprintln!("[k2-horizon-decode] step={step} wall={:.3}s gpu={:.3}s commands={}", decode_started.elapsed().as_secs_f64(), gpu.seconds, gpu.command_buffers,);
                    for operator in self.session.context().gpu_profile().into_iter().take(20) {
                        eprintln!("  k2 gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
                    }
                }
                self.compute_steps.fetch_add(1);
            }
            output.finish(|chunk| on_token(None, chunk));
            Ok(output.summary(tokens.len()))
        })();
        self.session.finish_decode(decoder);
        if let Ok(summary) = &result {
            let decode_seconds = decode_started.elapsed().as_secs_f64();
            eprintln!(
                "[k2-horizon-generation] prompt={} prefill={prefill_seconds:.3}s completion={} decode={decode_seconds:.3}s throughput={:.3} tok/s",
                summary.prompt_tokens,
                summary.completion_tokens,
                summary.completion_tokens as f64 / decode_seconds.max(f64::EPSILON),
            );
        }
        result
    }

    pub(crate) fn kv_residency(&self) -> crate::runtime::session::KvResidency {
        self.residency.report_without_terminal()
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        Ok(())
    }
}
