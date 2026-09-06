//! Qwen4-Exp(Qwen3.8-Flash-Next)× CUDA Node 引擎。
//!
//! 权重与专家驻留在 `cuda.rs` 组合之上(原始编码常驻主存 + 显存 LRU),
//! 本文件只做 Node 协议组装:chat template 渲染、流式生成、stop/EOS 语义。
//! KV 缓存沿用 CudaKvCache 的惰性逐层分配——48 层里只有 12 个 QSA 全注意力
//! 层真正持有 KV,显存预算由 expert_cache_gib 让位(默认 4GiB)。

use crate::{
    backend::{
        Backend,
        cuda::{CudaContext, CudaContextOptions, CudaPrefillExperts, CudaTensor, CudaWeight},
    },
    config::{CudaBackendConfig, Qwen4ExpNodeModelConfig},
    kv_cache::terminal_cache::TerminalInfo,
    moe::{
        expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
        prefill::prefill_experts_untraced,
        topk_moe::{MoeFfnRef, SharedExpertRef},
    },
    runtime::{
        expert_pipeline::{ExpertDecodePipeline, ExpertDecodeRequest},
        qwen4exp::{Qwen4ExpGguf, Qwen4ExpHyperConnection, Qwen4ExpLayer},
        session::{AtomicCounterU64, GenerationOutput, GenerationSummary, NodeCapabilities, RuntimeStatus},
    },
    weight::expert_source::ExpertSourceProvider,
};

fn moe_ref(weights: &Qwen4ExpLayer<CudaWeight>) -> SharedExpertRef<'_, CudaWeight> {
    SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }
}
use serde_json::Value;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

pub async fn run(model: Qwen4ExpNodeModelConfig, cuda: CudaBackendConfig, config: crate::server::node::NodeConfig) -> Result<(), crate::server::node::DynError> {
    let factory = Box::new(move |runtime, compute_steps| Qwen4ExpCudaEngine::load(model.clone(), &cuda, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>));
    crate::server::node::run_node(config, factory).await
}

pub struct Qwen4ExpCudaEngine {
    backend: CudaContext,
    model: Qwen4ExpNodeModelConfig,
    cfg: crate::runtime::qwen4exp::Qwen4ExpConfig,
    source: Arc<Qwen4ExpGguf>,
    layers: Vec<Qwen4ExpLayer<CudaWeight>>,
    output_hc: Qwen4ExpHyperConnection<CudaWeight>,
    lm_head: CudaWeight,
    chat_template: Option<crate::runtime::chat_template::ChatTemplate>,
    detokenizer: crate::tokenizer::Detokenizer,
    rope: crate::attention::rope::RopeTable,
    mtp: Option<super::cuda_mtp::Mtp>,
    capabilities: NodeCapabilities,
}

impl Qwen4ExpCudaEngine {
    pub fn load(model: Qwen4ExpNodeModelConfig, cuda: &CudaBackendConfig, _runtime: Arc<Mutex<RuntimeStatus>>, _compute_steps: Arc<AtomicCounterU64>) -> Result<Self, crate::server::node::DynError> {
        let backend = CudaContext::new_default_with_options(CudaContextOptions { device: cuda.device as usize, include_dir: cuda.include_directory.clone(), arch: cuda.architecture.clone() })?;
        let reader = crate::weight::container::gguf::GgufReader::locate(&model.weights_directory)?;
        let mut source = Qwen4ExpGguf::open(&reader)?;
        let cfg = source.config().clone();
        // 首版稠密 QSA 路径的全可见硬边界:超过 top_k 的上下文必须等稀疏索引器
        // CUDA 实现落地后才能放开(见 docs/qwen4exp-flash-next-wiring.md)。
        if model.max_sequence_length > cfg.max_position_embeddings {
            return Err(format!("Qwen4-Exp max_sequence_length 超过 max_position_embeddings({})", cfg.max_position_embeddings).into());
        }
        // 专家以原始 packed 编码常驻主存;拒绝任何会落入 F16 展开的量化格式。
        for tensor in source.reader().tensors().iter().filter(|t| t.name.contains("_exps.weight")) {
            let supported = if tensor.name.contains("ffn_down_exps") { matches!(tensor.tensor_type.0, 7 | 8) } else { matches!(tensor.tensor_type.0, 12 | 13) };
            if !supported {
                return Err(format!("Qwen4-Exp CUDA 原生专家路径不支持 {} type={}", tensor.name, tensor.tensor_type.0).into());
            }
        }
        #[cfg(target_os = "linux")]
        {
            let mem = std::fs::read_to_string("/proc/meminfo")?;
            let available = mem.lines().find_map(|line| line.strip_prefix("MemAvailable:").and_then(|value| value.split_whitespace().next()).and_then(|value| value.parse::<usize>().ok())).ok_or("无法读取 MemAvailable")? * 1024;
            let host_bytes: usize = source.reader().tensors().iter().filter(|t| t.name.contains("_exps.weight")).map(|t| t.bytes).sum();
            if available < host_bytes + 8 * 1024 * 1024 * 1024 {
                return Err(format!("专家驻留需要 {} bytes + 8GiB 余量,可用主存 {} bytes", host_bytes, available).into());
            }
        }
        let started = std::time::Instant::now();
        source.make_experts_resident()?;
        let source = Arc::new(source);
        if model.execution.pin_experts {
            let pinned = std::time::Instant::now();
            let bytes = backend.register_host_memory(Arc::new(source.clone()), |source| {
                let main = source.resident_experts.as_ref().ok_or("主存专家尚未加载".to_owned())?;
                main.iter().flat_map(|expert| [&expert.gate, &expert.up, &expert.down]).map(|matrix| matrix.bytes()).collect()
            })?;
            eprintln!("[qwen4exp-host-pinned] bytes={bytes} wall={:.3}s", pinned.elapsed().as_secs_f64());
        }
        eprintln!("[qwen4exp-resident-ready] wall={:.3}s", started.elapsed().as_secs_f64());
        let template = source.reader().metadata("tokenizer.chat_template").and_then(crate::weight::container::gguf::GgufValue::as_str);
        let chat_template = template
            .map(|source| -> Result<crate::runtime::chat_template::ChatTemplate, String> {
                let mut template = crate::runtime::chat_template::ChatTemplate::new(source)?;
                template.set_special_tokens("<bos>", "<eos>");
                Ok(template)
            })
            .transpose()
            .map_err(crate::server::node::DynError::from)?;
        let layers = crate::runtime::qwen4exp::prepare_qwen4exp_layers(&backend, &source)?;
        let output_hc = Qwen4ExpHyperConnection {
            norm: crate::runtime::prepare_gguf_f32_vector(&backend, source.reader(), "output_hc_norm.weight")?,
            down: crate::runtime::prepare_gguf_matrix(&backend, source.reader(), "output_hc_down.weight")?,
            up: crate::runtime::prepare_gguf_matrix(&backend, source.reader(), "output_hc_up.weight")?,
            inject: None,
        };
        let lm_head = crate::runtime::prepare_gguf_matrix(&backend, source.reader(), "output.weight")?;
        let detokenizer = source.detokenizer()?;
        let rope = crate::attention::rope::RopeTable::precompute(model.max_sequence_length, cfg.rope_dim, cfg.rope_theta);
        backend.rope_window_f16(&rope.cos).map_err(|error| error.to_string())?;
        backend.rope_window_f16(&rope.sin).map_err(|error| error.to_string())?;
        let mtp = model
            .execution
            .mtp_weights
            .as_ref()
            .map(|path| super::cuda_mtp::MtpSource::open(path, &cfg).map(Arc::new).map_err(|error| crate::server::node::DynError::from(error)))
            .transpose()?
            .map(|source| super::cuda_mtp::Mtp::new(&backend, source, model.max_sequence_length, model.execution.mtp_cache_gib as usize * 1024 * 1024 * 1024).map_err(|error| crate::server::node::DynError::from(error.to_string())))
            .transpose()?;
        if mtp.is_some() {
            eprintln!("[qwen4exp-mtp-node] shared draft loaded, steps={} cache={}GiB", model.execution.mtp_steps, model.execution.mtp_cache_gib);
        }
        let capabilities = crate::runtime::node::text_capabilities(
            crate::runtime::node::DeviceDescriptor {
                backend: "cuda-expert-lru",
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
                model_bytes: source.reader().file_len(),
                max_seq_len: model.max_sequence_length,
                // 惰性逐层 KV:仅 12 个 QSA 全注意力层分配。
                kv_cache_format: "f16-full-attn-only",
                input_modalities: &["text"],
            },
        );
        eprintln!("[qwen4exp-cuda-node-ready] layers={} vram_free={} load_wall={:.3}s", layers.len(), backend.device().mem_get_info().map(|(free, _)| free).unwrap_or(0), started.elapsed().as_secs_f64());
        Ok(Self { backend, model, cfg, source, layers, output_hc, lm_head, chat_template, detokenizer, rope, mtp, capabilities })
    }

    /// MTP 草稿/验证解码(语义镜像 bench 入口 run_mtp_decode,已在真机
    /// 验证过与普通 decode 逐 token 一致)。checkpoint 启用、target 采样、
    /// verify_samples 接受判定、retain_prefix 回退与草稿 KV 重写全链保留。
    #[allow(clippy::too_many_arguments)]
    fn generate_mtp(
        &mut self,
        mut state: super::cuda::CudaState,
        mut experts: ExpertDecodePipeline<crate::backend::cuda::CudaMoeState>,
        mut hidden: CudaTensor,
        tokens: &[u32],
        max_tokens: usize,
        output: &mut GenerationOutput,
        cancellation: &AtomicBool,
        on_token: &mut dyn FnMut(u32, String) -> bool,
    ) -> Result<GenerationSummary, String> {
        let mtp = self.mtp.as_mut().expect("generate_mtp 仅在 MTP 配置时调用");
        let steps = self.model.execution.mtp_steps;
        let min_confidence = self.model.execution.mtp_min_confidence;
        let chunk_size = self.model.execution.prefill_chunk_size.max(1);
        let expert_cache_bytes = self.model.execution.expert_cache_gib as usize * 1024 * 1024 * 1024;
        state.enable_checkpoints(&self.backend, steps + 1).map_err(|error| error.to_string())?;
        let spec = self.cfg.moe_spec();
        let target_sample = |hidden: &CudaTensor| -> Result<u32, String> {
            let (mixed, _) = super::cuda::hc_mix(&self.backend, &self.cfg, hidden, &self.output_hc).map_err(|error| error.to_string())?;
            Ok(self.backend.argmax(&self.backend.linear(&mixed, &self.lm_head).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?)
        };
        let mut anchor = target_sample(&hidden)?;
        let mut generated = 0usize;
        {
            let bytes = self.detokenizer.decode_bytes(&[anchor], true).map_err(|error| format!("detokenize {anchor}: {error}"))?;
            generated += 1;
            output.push(&bytes, |chunk| on_token(anchor, chunk));
        }
        let mut position = tokens.len();
        let mut verify_experts = CudaPrefillExperts::gguf(self.source.clone(), expert_cache_bytes);
        while generated < max_tokens && !self.cfg.eos_token_ids.contains(&anchor) {
            if cancellation.load(Ordering::Acquire) {
                output.cancel();
                break;
            }
            let count = steps.min(max_tokens - generated - 1);
            let mut candidates = Vec::with_capacity(count);
            let mut draft_hidden = self.backend.select_row(&hidden, 0).map_err(|error| error.to_string())?;
            let mut token = anchor;
            mtp.truncate(position);
            for step in 0..count {
                draft_hidden = mtp.forward(&self.backend, &self.source, &self.rope, &draft_hidden, token, position + step).map_err(|error| error.to_string())?;
                let (candidate, confidence) = mtp.sample(&self.backend, &draft_hidden, &self.lm_head).map_err(|error| error.to_string())?;
                if confidence < min_confidence {
                    break;
                }
                token = candidate;
                candidates.push(token);
                if self.cfg.eos_token_ids.contains(&token) {
                    break;
                }
            }
            let mut inputs = vec![anchor];
            inputs.extend_from_slice(&candidates);
            let verified = if inputs.len() == 1 {
                state
                    .forward(&self.backend, &self.cfg, &self.source, &self.layers, &self.rope, false, &inputs, position, &mut |layer, input| {
                        let weights = &self.layers[layer];
                        let shared = [SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }];
                        let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
                        let next = (layer + 1 < self.cfg.num_layers).then(|| self.source.source(layer + 1).map(|source| (layer + 1, source))).transpose().map_err(crate::backend::BackendError::ExpertLoad)?;
                        experts.decode(&self.backend, &spec, &reference, ExpertDecodeRequest { layer, source: self.source.source(layer).map_err(crate::backend::BackendError::ExpertLoad)?, position, next }, input)
                    })
                    .map_err(|error| error.to_string())?
            } else {
                verify_experts.swap_decode_state(experts.backend_state_mut());
                let result = state.forward(&self.backend, &self.cfg, &self.source, &self.layers, &self.rope, false, &inputs, position, &mut |layer, input| {
                    let weights = &self.layers[layer];
                    let shared = [SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }];
                    let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
                    prefill_experts_untraced(&self.backend, &spec, &reference, layer, &mut verify_experts, input, None)
                });
                verify_experts.swap_decode_state(experts.backend_state_mut());
                result.map_err(|error| error.to_string())?
            };
            let (mixed, _) = super::cuda::hc_mix(&self.backend, &self.cfg, &verified, &self.output_hc).map_err(|error| error.to_string())?;
            let logits = self.backend.linear(&mixed, &self.lm_head).map_err(|error| error.to_string())?;
            let mut target_tokens = Vec::with_capacity(inputs.len());
            for row in 0..inputs.len() {
                target_tokens.push(self.backend.argmax(&self.backend.select_row(&logits, row).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?);
            }
            let verification = crate::runtime::speculative::verify_samples(&target_tokens, &candidates, &self.cfg.eos_token_ids).map_err(|error| error.to_string())?;
            if verification.retained_rows < inputs.len() {
                state.retain_prefix(&self.backend, position, verification.retained_rows).map_err(|error| error.to_string())?;
            }
            hidden = self.backend.select_row(&verified, verification.retained_rows - 1).map_err(|error| error.to_string())?;
            for &token in &verification.tokens {
                let bytes = self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("detokenize {token}: {error}"))?;
                generated += 1;
                if !output.push(&bytes, |chunk| on_token(token, chunk)) {
                    return Ok(GenerationSummary { finish_reason: output.finish_reason().to_owned(), prompt_tokens: tokens.len(), completion_tokens: generated, cache: None, tool_calls: Vec::new() });
                }
            }
            anchor = verification.pending_token();
            if generated < max_tokens && !verification.eos {
                mtp.truncate(position + 1);
                for row in 1..verification.retained_rows {
                    let previous = self.backend.select_row(&verified, row - 1).map_err(|error| error.to_string())?;
                    mtp.forward(&self.backend, &self.source, &self.rope, &previous, inputs[row], position + row).map_err(|error| error.to_string())?;
                }
            }
            position += verification.retained_rows;
            if verification.eos {
                output.stop();
                break;
            }
            let _ = chunk_size;
        }
        output.finish(|chunk| on_token(0, chunk));
        Ok(GenerationSummary { finish_reason: output.finish_reason().to_owned(), prompt_tokens: tokens.len(), completion_tokens: generated, cache: None, tool_calls: Vec::new() })
    }
}

impl crate::server::node::NodeEngine for Qwen4ExpCudaEngine {
    fn model_key(&self) -> &'static str {
        "qwen4exp"
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
        let prompt = self.chat_template.as_ref().ok_or("Qwen4-Exp GGUF 缺少 tokenizer.chat_template")?.render(request)?;
        let max_tokens = crate::runtime::session::requested_completion_tokens(request);
        if max_tokens == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        let stops = crate::runtime::session::parse_stops(request.get("stop"))?;
        let mut output = GenerationOutput::new(&stops);
        let tokens = self.source.tokenizer().map_err(|error| error.to_string())?.tokenize(prompt.as_bytes());
        if tokens.is_empty() {
            return Err("Qwen4-Exp prompt 不能为空".to_owned());
        }
        let max_seq_len = self.model.max_sequence_length;
        if tokens.len().saturating_add(max_tokens) > max_seq_len {
            return Err(format!("Qwen4-Exp prompt {} + completion {} 超过 max_seq_len {max_seq_len}(稠密 QSA 上限 {})", tokens.len(), max_tokens, self.cfg.indexer.top_k));
        }
        let q8_kv = self.model.execution.kv_cache_format == crate::config::KvCacheFormat::Q8g64;
        let mut state = super::cuda::CudaState::new(&self.backend, &self.cfg, max_seq_len, q8_kv).map_err(|error| error.to_string())?;
        let spec = self.cfg.moe_spec();
        let mut prefill_experts = CudaPrefillExperts::gguf(self.source.clone(), self.model.execution.expert_cache_gib as usize * 1024 * 1024 * 1024);
        prefill_experts.reserve_arena(&self.backend).map_err(|error| error.to_string())?;
        prefill_experts.set_frequency_cache(self.model.execution.frequency_cache);
        prefill_experts.set_transfer_group(self.model.execution.expert_transfer_group);
        // prefill 与 run() 同路径:prefill_experts_untraced 批式推进。
        let chunk_size = self.model.execution.prefill_chunk_size.max(1);
        let mut residual = None;
        for (chunk_index, chunk) in tokens.chunks(chunk_size).enumerate() {
            residual = Some(
                state
                    .forward(&self.backend, &self.cfg, &self.source, &self.layers, &self.rope, false, chunk, chunk_index * chunk_size, &mut |layer, input| {
                        let weights = &self.layers[layer];
                        let shared = [moe_ref(weights)];
                        let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
                        prefill_experts_untraced(&self.backend, &spec, &reference, layer, &mut prefill_experts, input, None)
                    })
                    .map_err(|error| error.to_string())?,
            );
        }
        let mut experts = ExpertDecodePipeline::new(
            prefill_experts.into_decode_state(),
            ExpertPredictorConfig {
                first_layer: 0,
                layer_count: self.cfg.num_layers,
                expert_count: self.cfg.num_experts,
                routed_top_k: self.cfg.num_experts_per_tok,
                prefetch_count: self.model.execution.expert_prefetch_count,
                weights: ExpertPredictorWeights::default(),
            },
        )
        .map_err(|error| error.to_string())?;
        let residual = residual.expect("prefill 产出 residual");
        let mut hidden = self.backend.select_row(&residual, residual.rows - 1).map_err(|error| error.to_string())?;
        if self.mtp.is_some() {
            return self.generate_mtp(state, experts, hidden, &tokens, max_tokens, &mut output, cancellation, on_token);
        }
        let mut generated = 0usize;
        let mut last_token = 0u32;
        for _ in 0..max_tokens {
            if cancellation.load(Ordering::Acquire) {
                output.cancel();
                break;
            }
            let (mixed, _) = super::cuda::hc_mix(&self.backend, &self.cfg, &hidden, &self.output_hc).map_err(|error| error.to_string())?;
            let logits = self.backend.linear(&mixed, &self.lm_head).map_err(|error| error.to_string())?;
            let token = self.backend.argmax(&logits).map_err(|error| error.to_string())?;
            let bytes = self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("detokenize {token}: {error}"))?;
            last_token = token;
            generated += 1;
            if !output.push(&bytes, |chunk| on_token(token, chunk)) {
                break;
            }
            if self.cfg.eos_token_ids.contains(&token) {
                output.stop();
                break;
            }
            if generated == max_tokens {
                break;
            }
            let position = tokens.len() + generated - 1;
            let residual = state
                .forward(&self.backend, &self.cfg, &self.source, &self.layers, &self.rope, false, &[token], position, &mut |layer, input| {
                    let weights = &self.layers[layer];
                    let shared = [moe_ref(weights)];
                    let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
                    let next = (layer + 1 < self.cfg.num_layers).then(|| self.source.source(layer + 1).map(|source| (layer + 1, source))).transpose().map_err(crate::backend::BackendError::ExpertLoad)?;
                    experts.decode(&self.backend, &spec, &reference, ExpertDecodeRequest { layer, source: self.source.source(layer).map_err(crate::backend::BackendError::ExpertLoad)?, position, next }, input)
                })
                .map_err(|error| error.to_string())?;
            hidden = self.backend.select_row(&residual, residual.rows - 1).map_err(|error| error.to_string())?;
        }
        output.finish(|chunk| on_token(last_token, chunk));
        Ok(GenerationSummary { finish_reason: output.finish_reason().to_owned(), prompt_tokens: tokens.len(), completion_tokens: generated, cache: None, tool_calls: Vec::new() })
    }
}
