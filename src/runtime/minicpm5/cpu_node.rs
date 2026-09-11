//! MiniCPM5 × CPU Node 组合：GGUF 量化权重常驻，CPU 执行完整 prefill/decode。

use serde_json::Value;
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    attention::rope::RopeTable,
    backend::{
        Backend,
        cpu::{CpuContext, CpuKvCache, CpuWeight},
    },
    config::MiniCpm5NodeModelConfig,
    kernel::cpu::CpuTensor,
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::{
        minicpm5::{self, MiniCpm5Config, MiniCpm5OutputHead, MiniCpm5TextLayer, MiniCpm5Weights},
        session::{AtomicCounterU64, BatchTokenGuard, GenerationOutput, GenerationSummary, NodeCapabilities, RuntimeStatus as NodeRuntime, parse_stops, requested_completion_tokens},
    },
    server::node::{DynError, NodeConfig, NodeEngine},
    tokenizer::{Detokenizer, Tokenizer},
};

pub async fn run(model: MiniCpm5NodeModelConfig, config: NodeConfig) -> Result<(), DynError> {
    let factory =
        Box::new(move |runtime, compute_steps| MiniCpm5CpuEngine::load(&model.weights_directory, model.max_sequence_length, model.lm_head_quantization, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

struct MiniCpm5CpuEngine {
    backend: CpuContext,
    config: MiniCpm5Config,
    weights: MiniCpm5Weights,
    layers: Vec<MiniCpm5TextLayer<CpuWeight>>,
    output_head: MiniCpm5OutputHead<CpuWeight>,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    rope: RopeTable,
    eos_token_ids: Vec<u32>,
    max_seq_len: usize,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
}

impl MiniCpm5CpuEngine {
    fn load(model_path: &Path, max_seq_len: usize, lm_head_quantization: crate::weight::LmHeadQuantization, runtime: Arc<Mutex<NodeRuntime>>, compute_steps: Arc<AtomicCounterU64>) -> Result<Self, DynError> {
        let weights = MiniCpm5Weights::open(model_path).map_err(|error| format!("MiniCPM5 GGUF 打开失败: {error}"))?;
        let config = *weights.config();
        crate::runtime::validate_max_sequence_length("MiniCPM5", max_seq_len, config.max_position_embeddings)?;
        let backend = CpuContext;
        let mut layers = minicpm5::prepare_minicpm5_layers(&backend, &weights).map_err(|error| format!("准备 MiniCPM5 CPU 层: {error:?}"))?;
        let mut output_head = minicpm5::prepare_minicpm5_output_head_quantized(&backend, &config, &weights, lm_head_quantization).map_err(|error| format!("准备 MiniCPM5 CPU output head: {error:?}"))?;
        let mut resident_bytes = 0usize;
        for layer in &mut layers {
            for weight in [&mut layer.query, &mut layer.key, &mut layer.value, &mut layer.output, &mut layer.gate, &mut layer.up, &mut layer.down] {
                resident_bytes = resident_bytes.saturating_add(weight.make_gguf_resident().map_err(|error| format!("MiniCPM5 CPU 权重常驻: {error}"))?);
            }
        }
        resident_bytes = resident_bytes.saturating_add(output_head.lm_head_mut().make_gguf_resident().map_err(|error| format!("MiniCPM5 CPU lm_head 常驻: {error}"))?);
        let tokenizer = weights.tokenizer().map_err(|error| format!("MiniCPM5 tokenizer: {error}"))?;
        let detokenizer = weights.detokenizer().map_err(|error| format!("MiniCPM5 detokenizer: {error}"))?;
        let eos = minicpm5::minicpm5_eos_token_id(&weights);
        let im_end = tokenizer.tokenize_with_special(b"<|im_end|>", true);
        let mut eos_token_ids = vec![eos];
        if im_end.len() == 1 && im_end[0] != eos {
            eos_token_ids.push(im_end[0]);
        }
        let model_bytes = weights.reader().file_len();
        eprintln!("[minicpm5-cpu] layers={} packed_resident={:.1} MiB", layers.len(), resident_bytes as f64 / 1_048_576.0);
        let capabilities = crate::runtime::node::text_capabilities(
            crate::runtime::node::DeviceDescriptor {
                backend: "cpu",
                accelerator: "cpu".to_owned(),
                compute_units: Some(rayon::current_num_threads()),
                compute_unit_kind: "cpu_thread",
                memory_kind: "system",
                unified_memory: true,
                system_memory_bytes: None,
                accelerator_memory_bytes: None,
                recommended_working_set_bytes: None,
            },
            crate::runtime::node::SessionDescriptor { model_format: "gguf-mixed", model_bytes, max_seq_len, kv_cache_format: "f32", input_modalities: &["text"] },
        );
        Ok(Self { backend, config, weights, layers, output_head, tokenizer, detokenizer, rope: RopeTable::precompute(max_seq_len, config.head_dim, config.rope_theta), eos_token_ids, max_seq_len, capabilities, runtime, compute_steps })
    }

    fn generate(&mut self, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        let thinking = request.get("thinking").and_then(|value| value.get("type")).and_then(Value::as_str) != Some("disabled") && request.get("enable_thinking").and_then(Value::as_bool) != Some(false);
        let prompt = chat_prompt(request, thinking)?;
        let tokens = self.tokenizer.tokenize(prompt.as_bytes());
        if tokens.is_empty() || tokens.len() >= self.max_seq_len {
            return Err(format!("MiniCPM5 prompt tokens={} 超过 max_seq_len={}", tokens.len(), self.max_seq_len));
        }
        let _batch_guard = BatchTokenGuard::new(&self.runtime, tokens.len());
        let max_tokens = requested_completion_tokens(request);
        let max_tokens = max_tokens.min(self.max_seq_len - tokens.len());
        if max_tokens == 0 {
            return Err("max_tokens 必须大于 0".to_owned());
        }
        let stops = parse_stops(request.get("stop"))?;
        let mut sampling = match request.get("temperature").and_then(Value::as_f64) {
            Some(temperature) if temperature > 0.0 => {
                Some(crate::runtime::output::SamplingState::new(crate::runtime::output::SamplingConfig { temperature: temperature as f32, top_p: request.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32, seed: 0 })?)
            }
            _ => None,
        };
        let mut cache = CpuKvCache::new(self.config.layer_count);
        let embedding = self.weights.embedding_rows_f32(&tokens).map_err(|error| format!("MiniCPM5 embedding: {error}"))?;
        let hidden = CpuTensor { data: embedding, rows: tokens.len(), cols: self.config.hidden_size };
        let hidden = minicpm5::minicpm5_text_hidden(&self.backend, &self.config, &self.layers, Some(&mut cache), hidden, &self.rope, 0).map_err(|error| format!("MiniCPM5 CPU prefill: {error:?}"))?;
        let mut hidden = self.backend.select_row(&hidden, tokens.len() - 1).map_err(|error| format!("MiniCPM5 CPU last hidden: {error:?}"))?;
        let mut output = GenerationOutput::new(&stops);
        for position in tokens.len()..tokens.len() + max_tokens {
            if cancellation.load(Ordering::Relaxed) {
                output.cancel();
                break;
            }
            let token = match sampling.as_mut() {
                Some(state) => minicpm5::minicpm5_sampled_token_output(&self.backend, &self.config, &self.output_head, &hidden, &state.next()),
                None => minicpm5::minicpm5_token_output(&self.backend, &self.config, &self.output_head, &hidden, &self.eos_token_ids),
            }
            .map_err(|error| format!("MiniCPM5 CPU output: {error:?}"))?
            .token_id;
            if self.eos_token_ids.contains(&token) {
                output.stop();
                break;
            }
            let bytes = crate::runtime::tool::decode_output_token(&self.detokenizer, token).map_err(|error| format!("MiniCPM5 detokenize {token}: {error}"))?;
            if !output.push(&bytes, |chunk| on_token(token, chunk)) {
                break;
            }
            self.compute_steps.fetch_add(1);
            if output.completion_tokens() == max_tokens {
                break;
            }
            let embedding = self.weights.embedding_rows_f32(&[token]).map_err(|error| format!("MiniCPM5 decode embedding: {error}"))?;
            hidden = minicpm5::minicpm5_decode_round(&self.backend, &self.config, &self.layers, &mut cache, CpuTensor { data: embedding, rows: 1, cols: self.config.hidden_size }, &self.rope, position)
                .map_err(|error| format!("MiniCPM5 CPU decode position={position}: {error:?}"))?;
        }
        output.finish(|chunk| on_token(0, chunk));
        Ok(output.summary(tokens.len()))
    }
}

impl NodeEngine for MiniCpm5CpuEngine {
    fn model_key(&self) -> &'static str {
        "minicpm5"
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
    fn generate_one(&mut self, _request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        self.generate(request, cancellation, on_token)
    }
}

use super::protocol::chat_prompt;
