//! Qwen3.6/Qwen3.8 GGUF × CPU/CUDA 连续分层 Node/console engine。

use std::sync::{Arc, Mutex, atomic::AtomicBool};

use serde_json::Value;

use crate::{
    backend::cuda::{CudaContext, CudaContextOptions},
    config::{CudaBackendConfig, Qwen36NodeModelConfig},
    kv_cache::terminal_cache::TerminalInfo,
    runtime::{
        qwen36::cuda_hybrid::HybridCudaOptions,
        session::{AtomicCounterU64, GenerationSummary, NodeCapabilities, RuntimeStatus},
    },
    server::node::{DynError, NodeConfig, NodeEngine},
    weight::container::gguf::GgufReader,
};

pub async fn run(model: Qwen36NodeModelConfig, cuda: CudaBackendConfig, config: NodeConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let factory = Box::new(move |runtime, compute_steps| Qwen36CudaEngine::load(model.clone(), &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

pub struct Qwen36CudaEngine {
    backend: CudaContext,
    model: Qwen36NodeModelConfig,
    options: HybridCudaOptions,
    capabilities: NodeCapabilities,
}

impl Qwen36CudaEngine {
    pub fn load(model: Qwen36NodeModelConfig, cuda: &CudaBackendConfig, _runtime: Arc<Mutex<RuntimeStatus>>, _compute_steps: Arc<AtomicCounterU64>) -> Result<Self, DynError> {
        let backend = CudaContext::new_default_with_options(CudaContextOptions { device: cuda.device as usize, include_dir: cuda.include_directory.clone(), arch: cuda.architecture.clone() })?;
        let execution = &model.execution;
        let options = HybridCudaOptions {
            gpu_layers: execution.cuda_gpu_layers,
            max_sequence_length: model.max_sequence_length,
            prefill_chunk_size: execution.prefill_chunk_size,
            decode_cpu_threads: execution.cuda_decode_cpu_threads,
            cpu_packed_resident: execution.cuda_cpu_packed_resident,
            vram_reserve_bytes: execution.cuda_vram_reserve_gib.saturating_mul(1 << 30),
            precise_gqa_prefill: execution.precise_gqa_prefill,
            kv_f16: execution.kv_cache_format == crate::config::KvCacheFormat::F16,
        };
        let model_bytes = GgufReader::open(&GgufReader::locate(&model.weights_directory)?)?.file_len();
        let capabilities = crate::runtime::node::text_capabilities(
            crate::runtime::node::DeviceDescriptor {
                backend: "cuda-hybrid",
                accelerator: backend.device_name(),
                compute_units: None,
                compute_unit_kind: "sm",
                memory_kind: "dedicated",
                unified_memory: false,
                system_memory_bytes: None,
                accelerator_memory_bytes: None,
                recommended_working_set_bytes: None,
            },
            crate::runtime::node::SessionDescriptor { model_format: "gguf", model_bytes, max_seq_len: model.max_sequence_length, kv_cache_format: if options.kv_f16 { "f16" } else { "q8g64" }, input_modalities: &["text"] },
        );
        Ok(Self { backend, model, options, capabilities })
    }
}

impl NodeEngine for Qwen36CudaEngine {
    fn model_key(&self) -> &'static str {
        self.model.variant.model_key()
    }
    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        (self.capabilities.clone(), Arc::new(|| 0))
    }
    fn terminal_cache_infos(&self) -> Vec<TerminalInfo> {
        Vec::new()
    }
    fn max_concurrency(&self) -> usize {
        1
    }
    fn generate_one(&mut self, request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        let turns = crate::runtime::qwen36::protocol::parse_messages(request)?;
        let thinking_disabled = request.get("thinking").and_then(|value| value.get("type")).and_then(Value::as_str) == Some("disabled") || request.get("enable_thinking").and_then(Value::as_bool) == Some(false);
        let prompt = crate::runtime::qwen36::protocol::render_request_prompt(request, &turns, &[], !thinking_disabled)?;
        let max_tokens = crate::runtime::session::requested_completion_tokens(request);
        if max_tokens == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        let stops = crate::runtime::session::parse_stops(request.get("stop"))?;
        let mut tool_stream = crate::runtime::tool::RequestToolCallStream::new(request, request_id, crate::runtime::tool::ToolDialect::ChatmlJson);
        let mut summary = crate::runtime::qwen36::cuda_hybrid::generate(&self.backend, &self.model.weights_directory, &prompt, max_tokens, self.options, &stops, cancellation, &mut |token, chunk| {
            tool_stream.push(&chunk, |visible| on_token(token, visible))
        })
        .map_err(|error| error.to_string())?;
        if summary.finish_reason != "cancelled" && !tool_stream.finish(|visible| on_token(0, visible)) {
            summary.finish_reason = "cancelled".to_owned();
        }
        if summary.finish_reason != "cancelled" && !tool_stream.calls.is_empty() {
            summary.finish_reason = "tool_calls".to_owned();
        }
        summary.tool_calls = std::mem::take(&mut tool_stream.calls);
        Ok(summary)
    }
}
