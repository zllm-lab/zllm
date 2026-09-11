//! Mistral GGUF × CUDA 直接推理 engine。

use crate::{
    backend::{
        Backend,
        cuda::{CudaContext, CudaContextOptions, CudaKvCache, CudaTensor},
    },
    config::{CudaBackendConfig, MistralNodeModelConfig},
    kv_cache::terminal_cache::TerminalInfo,
    runtime::{
        mistral,
        session::{AtomicCounterU64, GenerationSummary, NodeCapabilities, RuntimeStatus},
    },
    server::node::{DynError, NodeConfig, NodeEngine},
    tokenizer::{Detokenizer, Tokenizer},
    weight::model::mistral::{MistralConfig, MistralWeights},
};
use serde_json::Value;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

pub async fn run(model: MistralNodeModelConfig, cuda: CudaBackendConfig, config: NodeConfig) -> Result<(), DynError> {
    let factory = Box::new(move |runtime, compute_steps| MistralCudaEngine::load(model, &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

struct Sequence {
    cache: CudaKvCache,
    hidden: CudaTensor,
    tokens: Vec<u32>,
}

pub struct MistralCudaEngine {
    backend: CudaContext,
    config: MistralConfig,
    weights: Arc<MistralWeights>,
    layers: Vec<mistral::MistralTextLayer<crate::backend::cuda::CudaWeight>>,
    output: mistral::MistralOutputHead<crate::backend::cuda::CudaWeight>,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    rope: crate::attention::rope::RopeTable,
    max_seq_len: usize,
    kv_f16: bool,
    capabilities: NodeCapabilities,
}

impl MistralCudaEngine {
    pub fn load(model: MistralNodeModelConfig, cuda: &CudaBackendConfig, _runtime: Arc<Mutex<RuntimeStatus>>, _compute_steps: Arc<AtomicCounterU64>) -> Result<Self, DynError> {
        let backend = CudaContext::new_default_with_options(CudaContextOptions { device: cuda.device as usize, include_dir: cuda.include_directory.clone(), arch: cuda.architecture.clone() })?;
        let weights = Arc::new(MistralWeights::open(&model.weights_directory)?);
        let config = *weights.config();
        crate::runtime::validate_max_sequence_length("Mistral", model.max_sequence_length, config.max_position_embeddings)?;
        let layers = mistral::prepare_mistral_layers(&backend, weights.as_ref()).map_err(|error| format!("准备 Mistral CUDA layers: {error:?}"))?;
        let output = mistral::prepare_mistral_output_head_quantized(&backend, &config, weights.as_ref(), model.lm_head_quantization).map_err(|error| format!("准备 Mistral CUDA output: {error:?}"))?;
        let tokenizer = weights.tokenizer()?;
        let detokenizer = weights.detokenizer()?;
        let rope = mistral::mistral_rope_table(&config, model.max_sequence_length);
        let kv_f16 = model.execution.kv_cache_format == crate::config::KvCacheFormat::F16;
        let capabilities = crate::runtime::node::text_capabilities(
            crate::runtime::node::DeviceDescriptor {
                backend: "cuda",
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
                model_bytes: weights.reader().file_len(),
                max_seq_len: model.max_sequence_length,
                kv_cache_format: if kv_f16 { "f16" } else { "q8g64" },
                input_modalities: &["text"],
            },
        );
        Ok(Self { backend, config, weights, layers, output, tokenizer, detokenizer, rope, max_seq_len: model.max_sequence_length, kv_f16, capabilities })
    }

    fn prefill(&self, tokens: Vec<u32>) -> Result<Sequence, String> {
        let columns = self.config.num_kv_heads * self.config.head_dim;
        let mut cache = if self.kv_f16 { CudaKvCache::new(self.config.layer_count, self.max_seq_len, columns) } else { CudaKvCache::new_q8g64(self.config.layer_count, self.max_seq_len, columns)? };
        let embedding = self.weights.embedding_rows_f32(&tokens).map_err(|error| error.to_string())?;
        let hidden = self.backend.tensor_from_f32(&embedding, tokens.len(), self.config.hidden_size).map_err(|error| format!("Mistral CUDA embedding: {error:?}"))?;
        let hidden = mistral::mistral_text_hidden(&self.backend, &self.config, &self.layers, Some(&mut cache), hidden, &self.rope, 0).map_err(|error| format!("Mistral CUDA prefill: {error:?}"))?;
        let hidden = self.backend.select_row(&hidden, tokens.len() - 1).map_err(|error| format!("Mistral CUDA select row: {error:?}"))?;
        Ok(Sequence { cache, hidden, tokens })
    }

    fn advance(&self, sequence: &mut Sequence, token: u32) -> Result<(), String> {
        let embedding = self.weights.embedding_rows_f32(&[token]).map_err(|error| error.to_string())?;
        let input = self.backend.tensor_from_f32(&embedding, 1, self.config.hidden_size).map_err(|error| format!("Mistral CUDA decode embedding: {error:?}"))?;
        let position = sequence.tokens.len();
        sequence.hidden = mistral::mistral_decode_round(&self.backend, &self.config, &self.layers, &mut sequence.cache, input, &self.rope, position).map_err(|error| format!("Mistral CUDA decode: {error:?}"))?;
        sequence.tokens.push(token);
        Ok(())
    }
}

impl NodeEngine for MistralCudaEngine {
    fn model_key(&self) -> &'static str {
        "mistral"
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
    fn generate_one(&mut self, _request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        let prompt = mistral::mistral_request_prompt(request)?;
        let tokens = self.tokenizer.tokenize(prompt.as_bytes());
        if tokens.is_empty() || tokens.len() >= self.max_seq_len {
            return Err(format!("Mistral prompt tokens={} 超过 max_seq_len={}", tokens.len(), self.max_seq_len));
        }
        let prompt_tokens = tokens.len();
        let max_tokens = crate::runtime::session::requested_completion_tokens(request).min(self.max_seq_len - prompt_tokens);
        if max_tokens == 0 {
            return Err("max_tokens 必须大于 0".to_owned());
        }
        let stops = crate::runtime::session::parse_stops(request.get("stop"))?;
        let mut sequence = self.prefill(tokens)?;
        let mut output = crate::runtime::session::GenerationOutput::new(&stops);
        for _ in 0..max_tokens {
            if cancellation.load(Ordering::Acquire) {
                output.cancel();
                break;
            }
            let token = mistral::mistral_token_output(&self.backend, &self.config, &self.output, &sequence.hidden).map_err(|error| format!("Mistral CUDA output: {error:?}"))?.token_id;
            if token == mistral::MISTRAL_EOS_TOKEN_ID {
                output.stop();
                break;
            }
            let bytes = crate::runtime::tool::decode_output_token(&self.detokenizer, token).map_err(|error| error.to_string())?;
            if !output.push(&bytes, |chunk| on_token(token, chunk)) {
                break;
            }
            if output.completion_tokens() < max_tokens {
                self.advance(&mut sequence, token)?;
            }
        }
        output.finish(|chunk| on_token(0, chunk));
        Ok(output.summary(prompt_tokens))
    }
}
