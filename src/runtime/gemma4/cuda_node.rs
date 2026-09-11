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
        Backend, LinearWeight,
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
    /// 逐层 KV 存储容量(slide 层=窗口,Full 层=max_seq)。
    kv_capacities: Vec<usize>,
    prefill_chunk_size: usize,
    /// 官方 MTP 投机头(可选);启用后 decode 走 draft/verify 轮次。
    mtp: Option<crate::runtime::gemma4::cuda_mtp::Gemma4CudaMtp>,
    mtp_draft_tokens: usize,
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
        // hybrid KV 容量:sliding-window 层只存窗口大小(ring 寻址),Full 层存满 max_seq。
        // prefill chunk 必须不超过最小窗口,否则 ring 在 chunk 内自回绕会覆盖仍需要的键。
        let capacities = (0..config.layer_count)
            .map(|layer| {
                let spec = model.layer_spec(layer).expect("Gemma4 层规格");
                spec.attention.hybrid.window.cache_capacity(model_config.max_sequence_length)
            })
            .collect::<Vec<_>>();
        let min_capacity = capacities.iter().copied().min().unwrap_or(model_config.max_sequence_length);
        let prefill_chunk_size = model_config.execution.prefill_chunk_size.min(min_capacity).max(1);
        if prefill_chunk_size != model_config.execution.prefill_chunk_size {
            eprintln!("[gemma4-cuda] prefill chunk {} 钳制到最小窗口 {min_capacity}", model_config.execution.prefill_chunk_size);
        }
        // MTP 投机头(PLE/E4B 尚不支持:per-layer 输入链未接入 draft)。
        let mtp = match &model_config.execution.mtp_weights {
            Some(path) => {
                if config.per_layer_input_size != 0 {
                    return Err("Gemma4 CUDA MTP 暂不支持 E4B per-layer 输入模型".into());
                }
                let mtp_weights = crate::weight::model::gemma4::Gemma4MtpWeights::open(path).map_err(|error| format!("Gemma4 CUDA MTP 权重: {error}"))?;
                let backbone_types: Vec<bool> = (0..config.layer_count).map(|layer| model.layer_spec(layer).expect("Gemma4 层规格").attention.hybrid.window == crate::attention::gqa::CausalWindow::Full).collect();
                let mtp = crate::runtime::gemma4::cuda_mtp::Gemma4CudaMtp::prepare(&backend, &mtp_weights, &backbone_types, config.num_kv_shared_layers, model_config.max_sequence_length)
                    .map_err(|error| format!("Gemma4 CUDA MTP 准备: {error:?}"))?;
                eprintln!("[gemma4-cuda] MTP draft 就绪 layers={} K={}", mtp_weights.config.layer_count, model_config.execution.mtp_draft_tokens);
                Some(mtp)
            }
            None => None,
        };
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
            kv_capacities: capacities,
            prefill_chunk_size,
            mtp,
            mtp_draft_tokens: model_config.execution.mtp_draft_tokens,
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
        // 低开销诊断:ZLLM_GEMMA4_CUDA_PROFILE=1 时输出 prefill 分 chunk 与 decode round 分解。
        let profile = std::env::var("ZLLM_GEMMA4_CUDA_PROFILE").is_ok();
        let mut cache = CudaKvCache::new_with_capacities(config.layer_count, self.max_seq_len, 0, self.kv_capacities.clone()).map_err(|error| format!("Gemma4 CUDA KV 容量: {error}"))?;
        let mut hidden = None;
        for (chunk, chunk_tokens) in tokens.chunks(self.prefill_chunk_size).enumerate() {
            let position = chunk * self.prefill_chunk_size;
            let chunk_started = std::time::Instant::now();
            let (input, per_layer_inputs) = self.model_inputs(chunk_tokens)?;
            hidden =
                Some(gemma4_prefill_hidden(&self.backend, &mut cache, &self.model, &self.layers, &self.rope, input, per_layer_inputs.as_deref(), position).map_err(|error| format!("Gemma4 CUDA prefill position={position}: {error:?}"))?);
            if profile {
                self.backend.synchronize().map_err(|error| format!("Gemma4 CUDA prefill 同步: {error:?}"))?;
                eprintln!("[gemma4-cuda-profile] prefill chunk={chunk} tokens={} wall={:.0}ms", chunk_tokens.len(), chunk_started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        self.backend.synchronize().map_err(|error| format!("Gemma4 CUDA prefill 同步: {error:?}"))?;
        let mut hidden = hidden.ok_or("Gemma4 prefill 没有 hidden")?;
        if self.mtp.is_some() {
            return self.generate_with_mtp(cache, hidden, &tokens, max_tokens, &stops, cancellation, on_token);
        }
        let mut output = GenerationOutput::new(&stops);
        let mut profile_round_ms = 0.0f64;
        let mut profile_output_ms = 0.0f64;
        let mut profile_steps = 0usize;
        for step in 0..max_tokens {
            if cancellation.load(Ordering::Relaxed) {
                output.cancel();
                break;
            }
            let output_started = std::time::Instant::now();
            let token = if step == 0 { gemma4_last_token_output(&self.backend, config, &self.output_head, &hidden, hidden.rows - 1) } else { gemma4_token_output(&self.backend, config, &self.output_head, &hidden) }
                .map_err(|error| format!("Gemma4 CUDA output: {error:?}"))?
                .token_id;
            if profile {
                profile_output_ms += output_started.elapsed().as_secs_f64() * 1000.0;
            }
            if config.eos_token_ids.contains(&token) {
                output.stop();
                break;
            }
            let bytes = crate::runtime::tool::decode_output_token(&self.detokenizer, token).map_err(|error| format!("Gemma4 detokenize {token}: {error}"))?;
            if !output.push(&bytes, |chunk| on_token(token, chunk)) {
                break;
            }
            self.compute_steps.fetch_add(1);
            if output.completion_tokens() == max_tokens {
                break;
            }
            let position = tokens.len() + step;
            let (input, per_layer_inputs) = self.model_inputs(&[token])?;
            let round_started = std::time::Instant::now();
            hidden = gemma4_decode_round(&self.backend, &mut cache, &self.model, &self.layers, &self.rope, input, per_layer_inputs.as_deref(), position).map_err(|error| format!("Gemma4 CUDA decode position={position}: {error:?}"))?;
            if profile {
                profile_round_ms += round_started.elapsed().as_secs_f64() * 1000.0;
                profile_steps += 1;
                if profile_steps.is_multiple_of(32) {
                    eprintln!(
                        "[gemma4-cuda-profile] steps={profile_steps} decode={profile_round_ms:.0}ms output={profile_output_ms:.0}ms round均={:.2}ms output均={:.2}ms",
                        profile_round_ms / profile_steps as f64,
                        profile_output_ms / profile_steps as f64
                    );
                }
            }
        }
        if profile && profile_steps > 0 {
            eprintln!(
                "[gemma4-cuda-profile] 完成 steps={profile_steps} decode={profile_round_ms:.0}ms output={profile_output_ms:.0}ms round均={:.2}ms output均={:.2}ms",
                profile_round_ms / profile_steps as f64,
                profile_output_ms / profile_steps as f64
            );
        }
        output.finish(|chunk| on_token(0, chunk));
        Ok(output.summary(tokens.len()))
    }

    /// MTP 投机 decode:轮次结构与 Metal engine 一致(pending 延迟一轮发射、
    /// draft 链同 position、K+1 行 verify、verify_samples 接受判定、KV 截断回滚)。
    #[allow(clippy::too_many_arguments)]
    fn generate_with_mtp(
        &self,
        mut cache: CudaKvCache,
        hidden: CudaTensor,
        tokens: &[u32],
        max_tokens: usize,
        stops: &[String],
        cancellation: &AtomicBool,
        on_token: &mut dyn FnMut(u32, String) -> bool,
    ) -> Result<GenerationSummary, String> {
        let mtp = self.mtp.as_ref().expect("MTP 路径要求引擎持有 draft 头");
        let config = self.model.config();
        let draft_tokens = self.mtp_draft_tokens.max(1);
        let mut stats = crate::runtime::speculative::SpeculativeStats::default();
        let mut output = GenerationOutput::new(stops);
        // 发射一个 token:EOS 停止;stop 序列由 GenerationOutput.push 判定。
        macro_rules! emit_token {
            ($token:expr) => {{
                let token: u32 = $token;
                (|| -> Result<bool, String> {
                    if config.eos_token_ids.contains(&token) {
                        output.stop();
                        return Ok(false);
                    }
                    let bytes = crate::runtime::tool::decode_output_token(&self.detokenizer, token).map_err(|error| format!("Gemma4 detokenize {token}: {error}"))?;
                    Ok(output.push(&bytes, |chunk| on_token(token, chunk)))
                })()
            }};
        }
        let first_output = gemma4_last_token_output(&self.backend, config, &self.output_head, &hidden, hidden.rows - 1).map_err(|error| format!("Gemma4 CUDA MTP 首 token: {error:?}"))?;
        let first = first_output.token_id;
        let selected = self.backend.select_row(&first_output.input, first_output.input.rows - 1).map_err(|error| format!("Gemma4 CUDA MTP normed 行: {error:?}"))?;
        let mut first_normed = self.backend.tensor_to_f32(&selected).map_err(|error| format!("Gemma4 CUDA MTP normed 读回: {error}"))?;
        let backbone_hidden_size = mtp.backbone_hidden_size();
        if first_normed.len() != backbone_hidden_size {
            return Err(format!("Gemma4 CUDA MTP 主干 hidden={}，assistant 期望 {backbone_hidden_size}", first_normed.len()));
        }
        let mut history = tokens.to_vec();
        let mut emitted_total = 0usize;
        let mut pending = first;
        let mut verify_normed_rows: Option<Vec<f32>> = None;
        let mut accepted_prev = 0usize;
        while emitted_total < max_tokens {
            if cancellation.load(Ordering::Relaxed) {
                output.cancel();
                break;
            }
            // 轮首发射上轮 pending(verify 已确认为 target 真值)。
            if !emit_token!(pending)? {
                break;
            }
            emitted_total += 1;
            if emitted_total >= max_tokens {
                break;
            }
            let position = history.len();
            let round_draft_tokens = draft_tokens.min(max_tokens - emitted_total).min(self.max_seq_len - position - 1);
            if round_draft_tokens == 0 {
                break;
            }
            // draft 链:所有步用同一 position(llama.cpp gemma4 分支语义)。
            let mut chain_hidden = match verify_normed_rows.as_ref() {
                Some(normed) => normed[accepted_prev * backbone_hidden_size..(accepted_prev + 1) * backbone_hidden_size].to_vec(),
                None => std::mem::take(&mut first_normed),
            };
            let draft_started = std::time::Instant::now();
            let mut drafts = Vec::with_capacity(round_draft_tokens);
            let mut token = pending;
            for _ in 0..round_draft_tokens {
                let embedding = gemma4_embedding_rows(&self.weights, &[token], config.hidden_size, self.embedding_scale)?;
                let mut input = Vec::with_capacity(mtp.concat_columns());
                input.extend_from_slice(&embedding);
                input.extend_from_slice(&chain_hidden);
                let (draft, h_next) = mtp.draft_step(&self.backend, &cache, &input, position).map_err(|error| format!("Gemma4 CUDA MTP draft: {error:?}"))?;
                chain_hidden = h_next;
                drafts.push(draft);
                token = draft;
            }
            let draft_seconds = draft_started.elapsed().as_secs_f64();
            // verify:pending + drafts 一次 K+1 行前向(KV 一次 append,ring 支持截断回滚)。
            let mut rows = Vec::with_capacity(drafts.len() + 1);
            rows.push(pending);
            rows.extend_from_slice(&drafts);
            let embedding = gemma4_embedding_rows(&self.weights, &rows, config.hidden_size, self.embedding_scale)?;
            let input = self.backend.tensor_from_f32(&embedding, rows.len(), config.hidden_size).map_err(|error| format!("Gemma4 CUDA verify embedding 上传: {error}"))?;
            let verify_started = std::time::Instant::now();
            let verify_hidden = gemma4_prefill_hidden(&self.backend, &mut cache, &self.model, &self.layers, &self.rope, input, None, position).map_err(|error| format!("Gemma4 CUDA verify prefill position={position}: {error:?}"))?;
            self.backend.synchronize().map_err(|error| format!("Gemma4 CUDA verify 同步: {error:?}"))?;
            let t_prefill = verify_started.elapsed();
            let plan = crate::runtime::output::OutputPlan { eps: config.rms_eps, norm: crate::runtime::output::OutputNorm::GemmaRms, excluded_tokens: vec![config.end_image_token_id, config.end_audio_token_id] };
            let (normed, logits) = crate::runtime::output::norm_and_lm_head(&self.backend, &self.output_head, &verify_hidden, &plan).map_err(|error| format!("Gemma4 CUDA verify 输出: {error:?}"))?;
            self.backend.synchronize().map_err(|error| format!("Gemma4 CUDA verify 同步: {error:?}"))?;
            let t_head = verify_started.elapsed();
            let targets = crate::backend::SegmentedTensorBackend::argmax_rows_excluding(&self.backend, &logits, &plan.excluded_tokens).map_err(|error| format!("Gemma4 CUDA verify argmax: {error:?}"))?;
            let t_argmax = verify_started.elapsed();
            let normed_f32 = self.backend.tensor_to_f32(&normed).map_err(|error| format!("Gemma4 CUDA verify normed 读回: {error}"))?;
            if std::env::var_os("ZLLM_GEMMA4_CUDA_PROFILE").is_some() && stats.rounds % 32 == 0 {
                eprintln!(
                    "[gemma4-cuda-mtp-split] prefill={:.1}ms head={:.1}ms argmax={:.1}ms readback={:.1}ms",
                    t_prefill.as_secs_f64() * 1e3,
                    (t_head - t_prefill).as_secs_f64() * 1e3,
                    (t_argmax - t_head).as_secs_f64() * 1e3,
                    (verify_started.elapsed() - t_argmax).as_secs_f64() * 1e3
                );
            }
            let verify_seconds = verify_started.elapsed().as_secs_f64();
            let verification = crate::runtime::speculative::verify_samples(&targets, &drafts, &config.eos_token_ids).map_err(|error| format!("Gemma4 CUDA MTP verify: {error:?}"))?;
            verify_normed_rows = Some(normed_f32);
            let retained = verification.retained_rows;
            history.truncate(position);
            history.extend_from_slice(&rows[..retained]);
            cache.truncate(verification.cache_end(position).map_err(|error| format!("Gemma4 CUDA MTP KV 提交边界: {error:?}"))?);
            accepted_prev = verification.accepted_drafts;
            pending = verification.pending_token();
            if std::env::var_os("ZLLM_GEMMA4_MTP_DUMP").is_some() {
                eprintln!("[mtp-dump] round={} position={} anchor={} drafts={drafts:?} targets={targets:?} acc={}", stats.rounds, position, pending, verification.accepted_drafts);
            }
            stats.record(drafts.len(), &verification);
            for token in verification.emitted_tokens() {
                if !emit_token!(*token)? {
                    break;
                }
                emitted_total += 1;
                self.compute_steps.fetch_add(1);
                if emitted_total >= max_tokens {
                    break;
                }
            }
            if std::env::var_os("ZLLM_GEMMA4_CUDA_PROFILE").is_some() {
                eprintln!(
                    "[gemma4-cuda-mtp] round={} acc={} wall={:.3}s draft={:.3}s verify={:.3}s rate={:.1}%",
                    stats.rounds,
                    verification.accepted_drafts,
                    draft_seconds + verify_seconds,
                    draft_seconds,
                    verify_seconds,
                    stats.accepted as f64 * 100.0 / stats.proposed.max(1) as f64
                );
            }
        }
        eprintln!("[gemma4-cuda-mtp] K={} rounds={} accepted={} emitted={} rate={:.1}%", draft_tokens, stats.rounds, stats.accepted, stats.emitted, stats.accepted as f64 * 100.0 / stats.proposed.max(1) as f64);
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
        _on_runtime_changed: &mut dyn FnMut(),
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
