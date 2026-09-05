//! Laguna GGUF × CUDA expert streaming Node/console engine。

use crate::{
    backend::cuda::{CudaContext, CudaContextOptions},
    config::{CudaBackendConfig, LagunaNodeModelConfig, LagunaStandaloneModelConfig, TextGenerationConfig},
    kv_cache::terminal_cache::TerminalInfo,
    runtime::session::{AtomicCounterU64, GenerationSummary, NodeCapabilities, RuntimeStatus},
    server::node::{DynError, NodeConfig, NodeEngine},
    weight::container::gguf::GgufReader,
};
use serde_json::Value;
use std::sync::{Arc, Mutex, atomic::AtomicBool};

pub async fn run(model: LagunaNodeModelConfig, cuda: CudaBackendConfig, config: NodeConfig) -> Result<(), DynError> {
    let factory = Box::new(move |runtime, compute_steps| LagunaCudaEngine::load(model.clone(), &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

pub struct LagunaCudaEngine {
    backend: CudaContext,
    model: LagunaNodeModelConfig,
    chat_template: Option<crate::runtime::chat_template::ChatTemplate>,
    capabilities: NodeCapabilities,
}

impl LagunaCudaEngine {
    pub fn load(model: LagunaNodeModelConfig, cuda: &CudaBackendConfig, _runtime: Arc<Mutex<RuntimeStatus>>, _compute_steps: Arc<AtomicCounterU64>) -> Result<Self, DynError> {
        let backend = CudaContext::new_default_with_options(CudaContextOptions { device: cuda.device as usize, include_dir: cuda.include_directory.clone(), arch: cuda.architecture.clone() })?;
        let reader = GgufReader::open(&GgufReader::locate(&model.weights_directory)?)?;
        let model_bytes = reader.file_len();
        // chat 模板直接渲染 GGUF 自带的 tokenizer.chat_template(与 gemma4 相同接线)。
        let template = reader.metadata("tokenizer.chat_template").and_then(crate::weight::container::gguf::GgufValue::as_str);
        let chat_template = template
            .map(|source| -> Result<crate::runtime::chat_template::ChatTemplate, String> {
                let mut template = crate::runtime::chat_template::ChatTemplate::new(source)?;
                template.set_special_tokens("<bos>", "<eos>");
                Ok(template)
            })
            .transpose()
            .map_err(DynError::from)?;
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
                // 滑窗层 ring KV(512)+ full 层 dense,f16 存储。
                kv_cache_format: "f16-ring",
                input_modalities: &["text"],
            },
        );
        Ok(Self { backend, model, chat_template, capabilities })
    }
}

impl NodeEngine for LagunaCudaEngine {
    fn model_key(&self) -> &'static str {
        "laguna"
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
        let _ = request_id;
        let prompt = self.chat_template.as_ref().ok_or("Laguna GGUF 缺少 tokenizer.chat_template")?.render(request)?;
        let max_tokens = crate::runtime::session::requested_completion_tokens(request);
        if max_tokens == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        let stops = crate::runtime::session::parse_stops(request.get("stop"))?;
        let standalone = LagunaStandaloneModelConfig {
            weights: self.model.weights_directory.clone(),
            lm_head_quantization: self.model.lm_head_quantization,
            generation: TextGenerationConfig { prompt, max_sequence_length: self.model.max_sequence_length, decode_steps: max_tokens },
            execution: self.model.execution,
        };
        let summary = crate::runtime::laguna::cuda::generate(&self.backend, standalone, &stops, cancellation, on_token).map_err(|error| error.to_string())?;
        Ok(summary)
    }
}
