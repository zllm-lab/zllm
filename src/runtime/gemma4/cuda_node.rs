//! Gemma 4 × CUDA Node：GGUF 权重与完整 Transformer/KV 全部驻留 NVIDIA GPU。

use half::{bf16, f16};
use serde_json::Value;
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    backend::{
        LinearWeight,
        cuda::{CudaContext, CudaContextOptions, CudaKvCache, CudaTensor, CudaWeight},
    },
    config::{CudaBackendConfig, Gemma4NodeModelConfig},
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::{
        gemma4::{
            self, Gemma4, Gemma4OutputHead, Gemma4PerLayerModel, Gemma4RopeTables, gemma4_decode_round, gemma4_embedding_rows, gemma4_last_token_output, gemma4_per_layer_embedding_rows, gemma4_per_layer_inputs, gemma4_prefill_hidden,
            gemma4_token_output, prepare_gemma4_layers, prepare_gemma4_output_head_quantized, prepare_gemma4_per_layer_model,
        },
        session::{AtomicCounterU64, BatchTokenGuard, GenerationOutput, GenerationSummary, NodeCapabilities, RuntimeStatus as NodeRuntime, parse_stops, requested_completion_tokens},
    },
    server::node::{DynError, NodeBatchRequest, NodeBatchResult, NodeConfig, NodeEngine},
    tokenizer::{Detokenizer, Tokenizer},
    weight::model::gemma4::{Gemma4OutputWeight, Gemma4Weights},
};

pub async fn run(model: Gemma4NodeModelConfig, cuda: CudaBackendConfig, config: NodeConfig) -> Result<(), DynError> {
    let factory = Box::new(move |runtime, compute_steps| Gemma4CudaEngine::load(&model, &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

pub struct Gemma4CudaEngine {
    backend: CudaContext,
    model: Gemma4,
    weights: Gemma4Weights,
    layers: Vec<gemma4::Gemma4Layer<CudaWeight>>,
    output_head: Gemma4OutputHead<CudaWeight>,
    per_layer_model: Option<Gemma4PerLayerModel<CudaWeight>>,
    rope: Gemma4RopeTables,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    chat_template: Option<crate::runtime::chat_template::ChatTemplate>,
    embedding_scale: f32,
    max_seq_len: usize,
    prefill_chunk_size: usize,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
}

impl Gemma4CudaEngine {
    pub fn load(model_config: &Gemma4NodeModelConfig, cuda: &CudaBackendConfig, runtime: Arc<Mutex<NodeRuntime>>, compute_steps: Arc<AtomicCounterU64>) -> Result<Self, DynError> {
        let config = Gemma4Weights::select_config(&model_config.weights_directory)?;
        if config.per_layer_input_size != 0 && config.per_layer_input_size != 256 {
            // E4B (per_layer_input_size=256) 走 backend capability trait 的通用 linear,
            // 不需要专门的 CUDA kernel 也能跑; 其他 PLE 变体暂不接入。
            return Err(format!("Gemma 4 CUDA 当前只接入 12B 变体 (per_layer_input_size=0) 与 E4B (per_layer_input_size=256); 实际 = {}", config.per_layer_input_size).into());
        }
        if config.per_layer_input_size != 0 {
            eprintln!("[gemma4-cuda] E4B per-layer input 使用 backend 通用 linear");
        }
        crate::runtime::validate_max_sequence_length("Gemma4", model_config.max_sequence_length, config.max_position_embeddings)?;
        let model = Gemma4::new(config.clone()).map_err(|error| format!("Gemma4 规格无效: {error:?}"))?;
        let weights = Gemma4Weights::open(&model_config.weights_directory, config.clone())?;
        let (tokenizer, detokenizer, chat_template, model_format) = if let Some(reader) = weights.gguf_reader() {
            let tokenizer = reader.bpe_tokenizer().map_err(|error| format!("Gemma4 GGUF tokenizer: {error}"))?;
            let detokenizer = reader.bpe_detokenizer().map_err(|error| format!("Gemma4 GGUF detokenizer: {error}"))?;
            let template = reader.metadata("tokenizer.chat_template").and_then(crate::weight::container::gguf::GgufValue::as_str);
            (tokenizer, detokenizer, compile_template(template)?, "gguf")
        } else {
            let tokenizer_path = model_config.weights_directory.join("tokenizer.json");
            let tokenizer = Tokenizer::new(&tokenizer_path).map_err(|error| format!("Gemma4 tokenizer: {error}"))?;
            let detokenizer = Detokenizer::load(&tokenizer_path).map_err(|error| format!("Gemma4 detokenizer: {error}"))?;
            let template_path = model_config.weights_directory.join("chat_template.jinja");
            let source = std::fs::read_to_string(&template_path).map_err(|error| format!("读取 {}: {error}", template_path.display()))?;
            (tokenizer, detokenizer, compile_template(Some(&source))?, "mlx-affine")
        };
        let backend = CudaContext::new_default_with_options(CudaContextOptions { device: cuda.device as usize, include_dir: cuda.include_directory.clone(), arch: cuda.architecture.clone() })?;
        let rope = Gemma4RopeTables::new(&config, model_config.max_sequence_length).map_err(|error| format!("Gemma4 RoPE: {error:?}"))?;
        let layers = prepare_gemma4_layers(&backend, &model, &weights).map_err(|error| format!("准备 Gemma4 CUDA 层: {error:?}"))?;
        let per_layer_model = prepare_gemma4_per_layer_model(&backend, &config, &weights).map_err(|error| format!("准备 Gemma4 CUDA per-layer model: {error:?}"))?;
        let output_head = prepare_output_head(&backend, &config, &weights, model_config.lm_head_quantization)?;
        backend.synchronize().map_err(|error| format!("Gemma4 CUDA 权重同步: {error:?}"))?;
        let model_bytes = if model_config.weights_directory.is_file() { model_config.weights_directory.metadata().map(|metadata| metadata.len()).unwrap_or(0) } else { directory_bytes(&model_config.weights_directory) };
        let (_, total) = backend.device().mem_get_info().map_err(|error| format!("读取 CUDA 显存: {error:?}"))?;
        let capabilities = crate::runtime::node::text_capabilities(
            crate::runtime::node::DeviceDescriptor {
                backend: "cuda",
                accelerator: backend.device_name(),
                compute_units: None,
                compute_unit_kind: "cuda_sm",
                memory_kind: "vram",
                unified_memory: false,
                system_memory_bytes: None,
                accelerator_memory_bytes: Some(total as u64),
                recommended_working_set_bytes: None,
            },
            crate::runtime::node::SessionDescriptor { model_format, model_bytes, max_seq_len: model_config.max_sequence_length, kv_cache_format: "f16", input_modalities: &["text"] },
        );
        eprintln!("[gemma4-cuda] device={} layers={} model={:.2}GiB", backend.device_name(), layers.len(), model_bytes as f64 / (1u64 << 30) as f64);
        Ok(Self {
            backend,
            model,
            weights,
            layers,
            output_head,
            per_layer_model,
            rope,
            tokenizer,
            detokenizer,
            chat_template,
            embedding_scale: bf16::from_f32(config.embedding_scale()).to_f32(),
            max_seq_len: model_config.max_sequence_length,
            prefill_chunk_size: model_config.execution.prefill_chunk_size,
            capabilities,
            runtime,
            compute_steps,
        })
    }

    /// token id 到主干/PLE 输入的模型语义由 Gemma4 runtime 统一；这里仅上传 CUDA tensor。
    fn model_inputs(&self, tokens: &[u32]) -> Result<(CudaTensor, Option<Vec<CudaTensor>>), String> {
        let config = self.model.config();
        let embedding = gemma4_embedding_rows(&self.weights, tokens, config.hidden_size, self.embedding_scale)?;
        let input = self.backend.tensor_from_f32(&embedding, tokens.len(), config.hidden_size)?;
        let per_layer_inputs = if config.per_layer_input_size == 0 {
            None
        } else {
            let values = gemma4_per_layer_embedding_rows(config, &self.weights, tokens)?;
            let columns = config.layer_count.checked_mul(config.per_layer_input_size).ok_or("Gemma4 per-layer embedding 列数溢出")?;
            let token_inputs = self.backend.tensor_from_f32(&values, tokens.len(), columns)?;
            gemma4_per_layer_inputs(&self.backend, config, self.per_layer_model.as_ref(), &input, Some(token_inputs)).map_err(|error| format!("Gemma4 CUDA per-layer inputs: {error:?}"))?
        };
        Ok((input, per_layer_inputs))
    }

    pub fn generate(&mut self, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        let prompt = self.chat_template.as_ref().ok_or("Gemma4 GGUF 缺少 tokenizer.chat_template")?.render(request)?;
        let tokens = self.tokenizer.tokenize(prompt.as_bytes());
        if tokens.is_empty() || tokens.len() >= self.max_seq_len {
            return Err(format!("Gemma4 prompt tokens={} 超过 max_seq_len={}", tokens.len(), self.max_seq_len));
        }
        let _batch_guard = BatchTokenGuard::new(&self.runtime, tokens.len());
        let requested = requested_completion_tokens(request);
        let max_tokens = requested.min(self.max_seq_len - tokens.len());
        if max_tokens == 0 {
            return Err("max_tokens 必须大于 0".to_owned());
        }
        let stops = parse_stops(request.get("stop"))?;
        let config = self.model.config();
        let mut cache = CudaKvCache::new(config.layer_count, self.max_seq_len, 0);
        let mut hidden = None;
        for (chunk, chunk_tokens) in tokens.chunks(self.prefill_chunk_size).enumerate() {
            let position = chunk * self.prefill_chunk_size;
            let (input, per_layer_inputs) = self.model_inputs(chunk_tokens)?;
            hidden =
                Some(gemma4_prefill_hidden(&self.backend, &mut cache, &self.model, &self.layers, &self.rope, input, per_layer_inputs.as_deref(), position).map_err(|error| format!("Gemma4 CUDA prefill position={position}: {error:?}"))?);
        }
        self.backend.synchronize().map_err(|error| format!("Gemma4 CUDA prefill 同步: {error:?}"))?;
        let mut hidden = hidden.ok_or("Gemma4 prefill 没有 hidden")?;
        let mut output = GenerationOutput::new(&stops);
        for step in 0..max_tokens {
            if cancellation.load(Ordering::Relaxed) {
                output.cancel();
                break;
            }
            let token = if step == 0 { gemma4_last_token_output(&self.backend, config, &self.output_head, &hidden, hidden.rows - 1) } else { gemma4_token_output(&self.backend, config, &self.output_head, &hidden) }
                .map_err(|error| format!("Gemma4 CUDA output: {error:?}"))?
                .token_id;
            if config.eos_token_ids.contains(&token) {
                output.stop();
                break;
            }
            let bytes = self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("Gemma4 detokenize {token}: {error}"))?;
            if !output.push(&bytes, |chunk| on_token(token, chunk)) {
                break;
            }
            self.compute_steps.fetch_add(1);
            if output.completion_tokens() == max_tokens {
                break;
            }
            let position = tokens.len() + step;
            let (input, per_layer_inputs) = self.model_inputs(&[token])?;
            hidden = gemma4_decode_round(&self.backend, &mut cache, &self.model, &self.layers, &self.rope, input, per_layer_inputs.as_deref(), position).map_err(|error| format!("Gemma4 CUDA decode position={position}: {error:?}"))?;
        }
        output.finish(|chunk| on_token(0, chunk));
        Ok(output.summary(tokens.len()))
    }
}

impl NodeEngine for Gemma4CudaEngine {
    fn model_key(&self) -> &'static str {
        "gemma4"
    }
    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        (self.capabilities.clone(), Arc::new(|| 0))
    }
    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        Vec::new()
    }
    fn max_concurrency(&self) -> usize {
        1
    }
    fn generate_batch(
        &mut self,
        requests: Vec<NodeBatchRequest>,
        _intake: &mut dyn FnMut(usize) -> Vec<NodeBatchRequest>,
        on_token: &mut dyn FnMut(&str, u32, String) -> bool,
        _on_tool_call_delta: &mut dyn FnMut(&str, crate::runtime::session::ToolCallDelta) -> bool,
        _on_result: &mut dyn FnMut(NodeBatchResult),
    ) -> Vec<NodeBatchResult> {
        requests
            .into_iter()
            .map(|request| {
                let request_id = request.request_id;
                let result = self.generate(&request.request, &request.cancellation, &mut |token, text| on_token(&request_id, token, text));
                NodeBatchResult { request_id, result }
            })
            .collect()
    }
}

fn prepare_output_head(backend: &CudaContext, config: &gemma4::Gemma4Config, weights: &Gemma4Weights, quantization: crate::weight::LmHeadQuantization) -> Result<Gemma4OutputHead<CudaWeight>, String> {
    let final_norm = weights.final_norm()?;
    match weights.load_output_weight()? {
        Gemma4OutputWeight::Quantized(weight) => prepare_gemma4_output_head_quantized(backend, config, &final_norm, LinearWeight::Quantized(weight.as_ref()), quantization),
        Gemma4OutputWeight::Dense(weight) if weight.dtype == "BF16" => {
            let values: Vec<f16> = weight.data.chunks_exact(2).map(|bytes| f16::from_f32(bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())).collect();
            prepare_gemma4_output_head_quantized(backend, config, &final_norm, LinearWeight::F16(&values), quantization)
        }
        Gemma4OutputWeight::Dense(weight) if weight.dtype == "F16" => {
            let values: Vec<f16> = weight.data.chunks_exact(2).map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]))).collect();
            prepare_gemma4_output_head_quantized(backend, config, &final_norm, LinearWeight::F16(&values), quantization)
        }
        Gemma4OutputWeight::Dense(weight) if weight.dtype == "F32" => {
            let values: Vec<f32> = weight.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 字节"))).collect();
            prepare_gemma4_output_head_quantized(backend, config, &final_norm, LinearWeight::F32(&values), quantization)
        }
        Gemma4OutputWeight::Dense(weight) => return Err(format!("Gemma4 LM head dtype={} 暂不支持", weight.dtype)),
    }
    .map_err(|error| format!("准备 Gemma4 CUDA output head: {error:?}"))
}

fn directory_bytes(root: &Path) -> u64 {
    std::fs::read_dir(root).map(|entries| entries.filter_map(Result::ok).map(|entry| entry.metadata().map(|metadata| metadata.len()).unwrap_or(0)).sum()).unwrap_or(0)
}

fn compile_template(source: Option<&str>) -> Result<Option<crate::runtime::chat_template::ChatTemplate>, String> {
    source
        .map(|source| {
            let mut template = crate::runtime::chat_template::ChatTemplate::new(source)?;
            template.set_special_tokens("<bos>", "<eos>");
            Ok(template)
        })
        .transpose()
}
