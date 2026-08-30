//! MiniCPM5 节点适配器：只负责装载参数、请求转发与状态上报。

#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
use crate::{
    config::MiniCpm5NodeModelConfig,
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::session::{GenerationSummary, NodeCapabilities},
    server::node::{DynError, NodeEngine},
};
#[cfg(not(target_os = "macos"))]
use crate::{config::MiniCpm5NodeModelConfig, server::node::DynError};

#[cfg(target_os = "macos")]
use super::engine::MiniCpm5Engine;

#[cfg(target_os = "macos")]
pub async fn run(model: MiniCpm5NodeModelConfig, config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    let model_path = model.weights_directory;
    let max_seq_len = model.max_sequence_length;
    let kv_f16 = model.execution.kv_cache_format == crate::config::KvCacheFormat::F16;
    let lm_head_quantization = model.lm_head_quantization;
    let factory = Box::new(move |runtime, compute_steps| MiniCpm5Engine::load(&model_path, max_seq_len, kv_f16, lm_head_quantization, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

#[cfg(not(target_os = "macos"))]
pub async fn run(_model: MiniCpm5NodeModelConfig, _config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    Err("MiniCPM5 Metal Node 需要 macOS".into())
}

#[cfg(target_os = "macos")]
impl NodeEngine for MiniCpm5Engine {
    fn model_key(&self) -> &'static str {
        "minicpm5"
    }

    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        MiniCpm5Engine::startup_info(self)
    }

    fn refresh_runtime(&self) {
        MiniCpm5Engine::refresh_runtime(self);
    }

    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        MiniCpm5Engine::terminal_cache_infos(self)
    }

    fn max_concurrency(&self) -> usize {
        1
    }

    fn shutdown(&mut self) -> Result<(), String> {
        MiniCpm5Engine::shutdown(self)
    }

    fn generate_one(&mut self, _request_id: &str, request: &serde_json::Value, cancellation: &std::sync::atomic::AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        MiniCpm5Engine::generate(self, request, cancellation, &mut |token, text| on_token(token.unwrap_or(0), text))
    }
}
