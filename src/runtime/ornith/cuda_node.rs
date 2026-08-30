//! Ornith GGUF × CUDA expert streaming Node/console engine。

use crate::{
    backend::cuda::{CudaContext, CudaContextOptions},
    config::{CudaBackendConfig, OrnithNodeModelConfig, OrnithStandaloneExecutionConfig, OrnithStandaloneModelConfig, TextGenerationConfig},
    kv_cache::terminal_cache::TerminalInfo,
    runtime::session::{AtomicCounterU64, GenerationSummary, NodeCapabilities, RuntimeStatus},
    server::node::{DynError, NodeConfig, NodeEngine},
    weight::container::gguf::GgufReader,
};
use serde_json::Value;
use std::sync::{Arc, Mutex, atomic::AtomicBool};

pub async fn run(model: OrnithNodeModelConfig, cuda: CudaBackendConfig, config: NodeConfig) -> Result<(), DynError> {
    let factory = Box::new(move |runtime, compute_steps| OrnithCudaEngine::load(model.clone(), &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

pub struct OrnithCudaEngine {
    backend: CudaContext,
    model: OrnithNodeModelConfig,
    capabilities: NodeCapabilities,
}

impl OrnithCudaEngine {
    pub fn load(model: OrnithNodeModelConfig, cuda: &CudaBackendConfig, _runtime: Arc<Mutex<RuntimeStatus>>, _compute_steps: Arc<AtomicCounterU64>) -> Result<Self, DynError> {
        let backend = CudaContext::new_default_with_options(CudaContextOptions { device: cuda.device as usize, include_dir: cuda.include_directory.clone(), arch: cuda.architecture.clone() })?;
        let model_bytes = GgufReader::open(&GgufReader::locate(&model.weights_directory)?)?.file_len();
        let capabilities = crate::runtime::node::text_capabilities(
            crate::runtime::node::DeviceDescriptor {
                backend: "cuda-expert-streaming",
                accelerator: backend.device_name(),
                compute_units: None,
                compute_unit_kind: "sm",
                memory_kind: "dedicated",
                unified_memory: false,
                system_memory_bytes: None,
                accelerator_memory_bytes: None,
                recommended_working_set_bytes: None,
            },
            crate::runtime::node::SessionDescriptor {
                model_format: "gguf",
                model_bytes,
                max_seq_len: model.max_sequence_length,
                kv_cache_format: if model.execution.kv_cache_format == crate::config::KvCacheFormat::F16 { "f16" } else { "q8g64" },
                input_modalities: &["text"],
            },
        );
        Ok(Self { backend, model, capabilities })
    }
}

impl NodeEngine for OrnithCudaEngine {
    fn model_key(&self) -> &'static str {
        "ornith"
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
        let prompt = crate::runtime::ornith::protocol::chat_prompt(request)?;
        let max_tokens = crate::runtime::session::requested_completion_tokens(request);
        if max_tokens == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        let stops = crate::runtime::session::parse_stops(request.get("stop"))?;
        let execution = &self.model.execution;
        let standalone = OrnithStandaloneModelConfig {
            weights: self.model.weights_directory.clone(),
            lm_head_quantization: self.model.lm_head_quantization,
            generation: TextGenerationConfig { prompt, max_sequence_length: self.model.max_sequence_length, decode_steps: max_tokens },
            execution: OrnithStandaloneExecutionConfig {
                precise_gqa_prefill: execution.precise_gqa_prefill,
                expert_batch_size: execution.expert_batch_size,
                kv_cache_format: execution.kv_cache_format,
                lazy_experts: execution.lazy_experts,
                expert_cache_gib: execution.expert_cache_gib,
                expert_prefetch_count: execution.expert_prefetch_count,
                mtp: false,
            },
        };
        let mut tool_stream = crate::runtime::tool::RequestToolCallStream::new(request, request_id, crate::runtime::tool::ToolDialect::ChatmlJson);
        let mut summary = crate::runtime::ornith::cuda::generate(&self.backend, standalone, &stops, cancellation, &mut |token, chunk| tool_stream.push(&chunk, |visible| on_token(token, visible))).map_err(|error| error.to_string())?;
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
