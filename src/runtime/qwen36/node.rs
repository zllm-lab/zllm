//! Qwen3.6 / Qwen3.8 节点适配器。只负责节点装载、批请求转发与状态上报。

#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
use crate::{
    config::Qwen36NodeModelConfig,
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::session::{GenerationSummary, NodeCapabilities},
    server::node::{DynError, NodeEngine},
};
#[cfg(not(target_os = "macos"))]
use crate::{config::Qwen36NodeModelConfig, server::node::DynError};

#[cfg(target_os = "macos")]
use super::engine::Qwen36Engine;

#[cfg(target_os = "macos")]
pub async fn run(model: Qwen36NodeModelConfig, config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    let cache_directory = config.cache_dir.clone();
    let persist_kv_cache = config.persist_kv_cache;
    let resident_cache_entries = config.terminal_cache_global_entries;
    let factory = Box::new(move |runtime, compute_steps| {
        Qwen36Engine::load(&model.weights_directory, model.max_sequence_length, model.variant, model.execution, model.lm_head_quantization, cache_directory, persist_kv_cache, resident_cache_entries, runtime, compute_steps)
            .map(|engine| Box::new(engine) as Box<dyn NodeEngine>)
    });
    crate::server::node::run_node(config, factory).await
}

#[cfg(not(target_os = "macos"))]
pub async fn run(_model: Qwen36NodeModelConfig, _config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    Err("Qwen3.6/Qwen3.8 Metal Node 需要 macOS".into())
}

#[cfg(target_os = "macos")]
impl NodeEngine for Qwen36Engine {
    fn model_key(&self) -> &'static str {
        Qwen36Engine::model_key(self)
    }

    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        Qwen36Engine::startup_info(self)
    }

    fn refresh_runtime(&self) {
        Qwen36Engine::refresh_runtime(self);
    }

    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        Qwen36Engine::terminal_cache_infos(self)
    }

    fn max_concurrency(&self) -> usize {
        1
    }

    fn shutdown(&mut self) -> Result<(), String> {
        Qwen36Engine::shutdown(self)
    }

    fn generate_one(&mut self, _request_id: &str, request: &serde_json::Value, cancellation: &std::sync::atomic::AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        // Qwen3.6/3.8 生成路径不按 request_id 分流,忽略之。
        Qwen36Engine::generate(self, request, cancellation, &mut |token, text| on_token(token.unwrap_or(0), text))
    }
}
