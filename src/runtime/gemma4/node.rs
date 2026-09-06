//! Gemma 4 节点适配器。只负责节点装载、批请求转发与状态上报。

#[cfg(target_os = "macos")]
use std::collections::HashSet;
#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex};

#[cfg(not(target_os = "macos"))]
use crate::{config::Gemma4NodeModelConfig, server::node::DynError};
#[cfg(target_os = "macos")]
use crate::{
    config::{Gemma4NodeModelConfig, NodeMetalBackendConfig},
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::session::{GenerationSummary, NodeCapabilities},
    server::node::{DynError, NodeEngine},
};

#[cfg(target_os = "macos")]
use super::engine::Gemma4Engine;

#[cfg(target_os = "macos")]
pub async fn run(model: Gemma4NodeModelConfig, backend: NodeMetalBackendConfig, config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    let cache_directory = config.cache_dir.clone();
    let persist_kv_cache = config.persist_kv_cache;
    let resident_cache_entries = config.terminal_cache_global_entries;
    let factory = Box::new(move |runtime, compute_steps| {
        Gemma4Engine::load(&model.weights_directory, model.max_sequence_length, model.execution, backend.replay, model.lm_head_quantization, cache_directory, persist_kv_cache, resident_cache_entries, runtime, compute_steps)
            .map(|engine| Box::new(engine) as Box<dyn NodeEngine>)
    });
    crate::server::node::run_node(config, factory).await
}

#[cfg(not(target_os = "macos"))]
pub async fn run(_model: Gemma4NodeModelConfig, _backend: crate::config::NodeMetalBackendConfig, _config: crate::server::node::NodeConfig) -> Result<(), DynError> {
    Err("Gemma 4 Metal Node 需要 macOS".into())
}

#[cfg(target_os = "macos")]
impl NodeEngine for Gemma4Engine {
    fn model_key(&self) -> &'static str {
        "gemma4"
    }

    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        Gemma4Engine::startup_info(self)
    }

    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        Gemma4Engine::terminal_cache_infos(self)
    }

    fn terminal_cache_pins(&self) -> Option<Arc<Mutex<HashSet<String>>>> {
        Some(Gemma4Engine::terminal_cache_pins(self))
    }

    fn refresh_runtime(&self) {
        Gemma4Engine::refresh_runtime(self);
    }

    fn max_concurrency(&self) -> usize {
        1
    }

    fn shutdown(&mut self) -> Result<(), String> {
        Gemma4Engine::shutdown(self)
    }

    fn generate_one(&mut self, _request_id: &str, request: &serde_json::Value, cancellation: &std::sync::atomic::AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        // Gemma4 生成路径不按 request_id 分流,忽略之。
        Gemma4Engine::generate(self, request, cancellation, &mut |token, text| on_token(token.unwrap_or(0), text))
    }
}
