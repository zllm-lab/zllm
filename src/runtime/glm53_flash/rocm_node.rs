//! GLM-5.3-Flash ROCm head node:prompt prefill、跨机 stage 与逐 token decode。

use std::sync::{Arc, Mutex, atomic::Ordering};

use crate::{
    config::{Glm53FlashNodeModelConfig, RocmBackendConfig},
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::{
        glm53_flash::{
            protocol::chat_segments,
            vision::{Glm53FlashMultimodalProcessor, Glm53FlashVisionConfig},
        },
        output::SamplingConfig,
        session::{AtomicCounterU64, GenerationOutput, GenerationSummary, NodeCapabilities, RuntimeStatus as NodeRuntime, parse_stops},
    },
    server::{
        node::{DynError, NodeBatchRequest, NodeBatchResult, NodeConfig, NodeEngine, run_node, with_content_parts},
        stage_transport::{RequestId, StageMessage, StageTransport},
    },
    tokenizer::{Detokenizer, Tokenizer},
    vision::MultimodalProcessor,
};

use super::rocm_engine::{Engine, Options};

pub async fn run(model: Glm53FlashNodeModelConfig, backend: RocmBackendConfig, config: NodeConfig) -> Result<(), DynError> {
    let factory =
        Box::new(move |runtime, compute_steps| Glm53FlashNodeEngine::load(model.clone(), backend.clone(), runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>).map_err(|error| -> DynError { error.into() }));
    run_node(config, factory).await
}

struct Glm53FlashNodeEngine {
    engine: Engine,
    link: StageTransport,
    processor: Glm53FlashMultimodalProcessor,
    /// 输出侧拦截的视觉特殊 token(image/begin/end),不外泄也不回喂 decode。
    vision_special_tokens: [u32; 3],
    /// checkpoint 是否携带视觉塔;决定能力上报与图像请求接受与否。
    accepts_images: bool,
    detokenizer: Detokenizer,
    max_sequence_length: usize,
    prefill_policy: crate::runtime::prefill::AdaptiveChunkPolicy,
    /// MTP speculative 协议开关;head 不装 MTP 权重,只按 Verify/Speculative 驱动。
    mtp: bool,
    mtp_draft_tokens: usize,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
}

impl Glm53FlashNodeEngine {
    fn load(model: Glm53FlashNodeModelConfig, backend: RocmBackendConfig, runtime: Arc<Mutex<NodeRuntime>>, compute_steps: Arc<AtomicCounterU64>) -> Result<Self, String> {
        let tokenizer = Tokenizer::new(&model.tokenizer).map_err(|error| format!("加载 GLM-5.3-Flash tokenizer: {error}"))?;
        let detokenizer = Detokenizer::load(&model.tokenizer).map_err(|error| format!("加载 GLM-5.3-Flash detokenizer: {error}"))?;
        let processor = Glm53FlashMultimodalProcessor::new(tokenizer, &Glm53FlashVisionConfig::standard()).map_err(|error| format!("GLM-5.3-Flash 多模态 processor: {error}"))?;
        let vision_special_tokens = processor.vision_special_token_ids();
        let mtp = model.mtp;
        let mtp_draft_tokens = model.mtp_draft_tokens;
        let engine = Engine::load(&Options {
            weights_directory: model.weights_directory,
            devices: backend.devices,
            layer_ends: model.head.layer_ends,
            layer_start: 0,
            layer_end: model.head.stage_end,
            max_sequence_length: model.max_sequence_length,
            prefill_chunk_size: model.prefill_chunk_size,
            mtp: false,
        })?;
        let accepts_images = engine.vision_available();
        if accepts_images {
            eprintln!("[glm53-head] 视觉塔张量存在,多模态输入已启用");
        }
        let iroh = model.head.downstream.iroh.runtime().map_err(|error| error.to_string())?;
        let mut link = StageTransport::connect(&model.head.downstream.ticket, iroh).map_err(|error| format!("连接 GLM-5.3-Flash tail: {error}"))?;
        match link.recv().map_err(|error| format!("接收 GLM-5.3-Flash tail 显存信息: {error}"))?.message {
            StageMessage::DeviceMemory { devices, .. } => eprintln!("[glm53-head] tail devices={} 已就绪", devices.len()),
            other => return Err(format!("GLM-5.3-Flash tail 首帧期望 DeviceMemory，实际 {other:?}")),
        }
        let prefill_policy = crate::runtime::prefill::AdaptiveChunkPolicy {
            initial_chunk_size: model.prefill_chunk_size,
            append_chunk_size: model.prefill_chunk_size,
            long_context_threshold_tokens: model.long_prefill_threshold_tokens,
            long_context_chunk_size: model.long_prefill_chunk_size.unwrap_or(model.prefill_chunk_size),
        };
        Ok(Self { engine, link, processor, vision_special_tokens, accepts_images, detokenizer, max_sequence_length: model.max_sequence_length, prefill_policy, mtp, mtp_draft_tokens, runtime, compute_steps })
    }

    fn generate_one(&mut self, request: NodeBatchRequest, on_token: &mut dyn FnMut(&str, u32, String) -> bool) -> NodeBatchResult {
        let request_id = request.request_id.clone();
        let result = self.generate_inner(&request, on_token);
        NodeBatchResult { request_id, result }
    }

    fn generate_inner(&mut self, request: &NodeBatchRequest, on_token: &mut dyn FnMut(&str, u32, String) -> bool) -> Result<GenerationSummary, String> {
        let segments = chat_segments(&request.request)?;
        let input = with_content_parts(&segments, |parts| self.processor.process(parts).map_err(|error| format!("GLM-5.3-Flash 图文输入: {error}")))?;
        let prompt_tokens = input.token_ids;
        if !input.images.is_empty() && !self.accepts_images {
            return Err("GLM-5.3-Flash checkpoint 缺少视觉塔,不能处理图像输入".to_owned());
        }
        let max_tokens = crate::runtime::session::requested_completion_tokens(&request.request);
        if prompt_tokens.is_empty() || max_tokens == 0 || prompt_tokens.len() >= self.max_sequence_length {
            return Err(format!("GLM-5.3-Flash prompt/max_tokens 非法: prompt={} max={max_tokens} capacity={}", prompt_tokens.len(), self.max_sequence_length));
        }
        let max_tokens = max_tokens.min(self.max_sequence_length - prompt_tokens.len());
        let stops = parse_stops(request.request.get("stop"))?;
        if request.cancellation.load(Ordering::Acquire) {
            return Err("请求已取消".to_owned());
        }
        self.engine.reset().map_err(backend_error)?;
        let stage_id = RequestId::parse(&request.request_id).unwrap_or_else(|_| RequestId::from_cache_id(&request.request_id));
        self.link.send_open(stage_id, None, 0, false, SamplingConfig::greedy(0), true)?;
        match self.link.recv()?.message {
            StageMessage::Ready { cached_tokens: 0 } => {}
            other => return Err(format!("GLM-5.3-Flash tail Open 回应异常: {other:?}")),
        }

        let generated = (|| {
            let overlay = if input.images.is_empty() {
                None
            } else {
                let overlay = Arc::new(self.engine.vision_overlay(&input.images, &input.image_token_ranges).map_err(backend_error)?);
                Some(overlay)
            };
            let mut submitted = 0usize;
            let mut chunks = Vec::new();
            while submitted < prompt_tokens.len() {
                let chunk_size = self.prefill_policy.chunk_size(0, submitted);
                let end = submitted.saturating_add(chunk_size).min(prompt_tokens.len());
                chunks.push(prompt_tokens[submitted..end].to_vec());
                submitted = end;
            }
            let cols = self.engine.boundary_cols();
            if self.mtp {
                // tail 需要 prompt tokens 做 MTP 移位 embedding，并校验两端递归深度一致。
                self.link.send_mtp_context(stage_id, &prompt_tokens, max_tokens, self.mtp_draft_tokens).map_err(|msg| format!("发送 MtpContext: {msg}"))?;
            }
            let link = &mut self.link;
            let send_chunk = |position: usize, rows: usize, values: Vec<u16>| -> Result<(), crate::backend::BackendError> {
                link.send_prefill(stage_id, position, rows, cols, &values, &[]).map_err(|msg| crate::backend::BackendError::Compute { msg })?;
                Ok(())
            };
            match overlay {
                Some(overlay) => self.engine.prefill_multimodal_pipeline(chunks, overlay, send_chunk).map_err(backend_error)?,
                None => self.engine.prefill_tokens_pipeline(chunks, send_chunk).map_err(backend_error)?,
            }
            self.link.send_prefill_done(stage_id, prompt_tokens.len())?;
            let mut output = GenerationOutput::new(&stops);
            if self.mtp {
                // speculative 主循环:verify [anchor, drafts...] K+1 行，tail 判定连续接受前缀；
                // tail 已先提交本机状态，head 在发送下一轮前按 retained_rows 对齐。
                let (mut pending, _retained, mut drafts, mut pending_eos) = recv_speculative(&mut self.link, stage_id)?;
                let mut last_sent: Option<(usize, usize, usize)> = None;
                loop {
                    let mut stop = false;
                    for &token in &pending {
                        if output.completion_tokens() >= max_tokens {
                            break;
                        }
                        if self.vision_special_tokens.contains(&token) {
                            output.stop();
                            stop = true;
                            break;
                        }
                        let bytes = self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("detokenize {token}: {error}"))?;
                        if !output.push(&bytes, |text| on_token(&request.request_id, token, text)) {
                            stop = true;
                            break;
                        }
                        if request.cancellation.load(Ordering::Acquire) {
                            output.cancel();
                            stop = true;
                            break;
                        }
                    }
                    if stop || output.completion_tokens() >= max_tokens {
                        break;
                    }
                    if pending_eos {
                        output.stop();
                        break;
                    }
                    if let Some((position, rows, retained)) = last_sent {
                        if rows > retained {
                            self.engine.rollback_verify(position + retained).map_err(backend_error)?;
                        } else {
                            self.engine.commit_verify();
                        }
                    }
                    let anchor = *pending.last().ok_or("MTP 轮没有确认 token")?;
                    if drafts.is_empty() {
                        return Err("MTP 轮没有 draft".to_owned());
                    }
                    let verify_inputs = std::iter::once(anchor).chain(drafts.iter().copied()).collect::<Vec<_>>();
                    let position = self.engine.position();
                    let hidden = self.engine.verify_tokens_collected(&verify_inputs).map_err(backend_error)?;
                    let values = self.engine.boundary_bits(&hidden).map_err(backend_error)?;
                    self.link.send_verify(stage_id, position, verify_inputs.len(), self.engine.boundary_cols(), &values, &[])?;
                    let next = recv_speculative(&mut self.link, stage_id)?;
                    let (confirmed, retained, next_drafts, next_eos) = next;
                    last_sent = Some((position, verify_inputs.len(), retained));
                    pending = confirmed;
                    drafts = next_drafts;
                    pending_eos = next_eos;
                }
            } else {
                let mut token = recv_token(&mut self.link, stage_id)?;
                while output.completion_tokens() < max_tokens {
                    if self.vision_special_tokens.contains(&token.0) {
                        // 视觉特殊 token 只属于 prompt;被采样出时立即截断,不外泄也不回喂。
                        output.stop();
                        break;
                    }
                    let bytes = self.detokenizer.decode_bytes(&[token.0], true).map_err(|error| format!("detokenize {}: {error}", token.0))?;
                    if !output.push(&bytes, |text| on_token(&request.request_id, token.0, text)) {
                        break;
                    }
                    if request.cancellation.load(Ordering::Acquire) {
                        output.cancel();
                        break;
                    }
                    if token.1 {
                        output.stop();
                        break;
                    }
                    if output.completion_tokens() == max_tokens {
                        break;
                    }
                    let position = self.engine.position();
                    let hidden = self.engine.forward_tokens(&[token.0]).map_err(backend_error)?;
                    let values = self.engine.boundary_bits(&hidden).map_err(backend_error)?;
                    self.link.send_decode(stage_id, position, self.engine.boundary_cols(), &values, &[])?;
                    token = recv_token(&mut self.link, stage_id)?;
                }
            }
            output.finish(|text| on_token(&request.request_id, 0, text));
            Ok(output.summary(prompt_tokens.len()))
        })();
        let _ = self.link.send_delete(stage_id);
        if generated.is_ok() {
            self.compute_steps.fetch_add(1);
        }
        generated
    }
}

impl NodeEngine for Glm53FlashNodeEngine {
    fn model_key(&self) -> &'static str {
        "glm-5.3-flash"
    }

    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        let input_modalities: &[&str] = if self.accepts_images { &["text", "image"] } else { &["text"] };
        let capabilities = crate::runtime::node::text_capabilities(
            crate::runtime::node::DeviceDescriptor {
                backend: "rocm",
                accelerator: "AMD ROCm dual-node stage".to_owned(),
                compute_units: None,
                compute_unit_kind: "CU",
                memory_kind: "VRAM",
                unified_memory: false,
                system_memory_bytes: None,
                accelerator_memory_bytes: None,
                recommended_working_set_bytes: None,
            },
            crate::runtime::node::SessionDescriptor { model_format: "safetensors-fp8", model_bytes: 0, max_seq_len: self.max_sequence_length, kv_cache_format: "q8g64-mla+kda", input_modalities },
        );
        (capabilities, Arc::new(|| 0))
    }

    fn refresh_runtime(&self) {
        if let Ok(mut runtime) = self.runtime.lock() {
            runtime.kv_cache_entries = 0;
            runtime.kv_cache_tokens = 0;
            runtime.kv_cache_allocated_bytes = 0;
        }
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
        requests.into_iter().map(|request| self.generate_one(request, on_token)).collect()
    }
}

fn recv_token(link: &mut StageTransport, request_id: RequestId) -> Result<(u32, bool), String> {
    let frame = link.recv()?;
    if frame.request_id != request_id {
        return Err(format!("GLM-5.3-Flash tail 回传 request={} 期望={request_id}", frame.request_id));
    }
    match frame.message {
        StageMessage::Token { token, eos } => Ok((token, eos)),
        other => Err(format!("GLM-5.3-Flash tail 期望 Token，实际 {other:?}")),
    }
}

fn recv_speculative(link: &mut StageTransport, request_id: RequestId) -> Result<(Vec<u32>, usize, Vec<u32>, bool), String> {
    let frame = link.recv()?;
    if frame.request_id != request_id {
        return Err(format!("GLM-5.3-Flash tail 回传 request={} 期望={request_id}", frame.request_id));
    }
    match frame.message {
        StageMessage::Speculative { tokens, retained_rows, drafts, eos } => Ok((tokens, retained_rows, drafts, eos)),
        other => Err(format!("GLM-5.3-Flash tail 期望 Speculative，实际 {other:?}")),
    }
}

fn backend_error(error: crate::backend::BackendError) -> String {
    format!("{error:?}")
}
