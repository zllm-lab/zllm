//! Qwen3.6 / Qwen3.8 × Metal 嵌入式推理引擎：统一请求、session 与生成生命周期。
#[cfg(target_os = "macos")]
use super::protocol::{chat_prompt_suffix, parse_messages, render_request_prompt};
#[cfg(target_os = "macos")]
use crate::{
    backend::{Backend, metal::MetalTensor},
    runtime::{
        qwen36,
        session::{BatchTokenGuard, ContentPiece, FixedSessionResidency, GenerationOutput, GenerationSummary, KvResidency, parse_stops},
        tool::{RequestToolCallStream, ToolDialect},
    },
};
#[cfg(target_os = "macos")]
use crate::{
    config::Qwen36Variant,
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::session::{AtomicCounterU64, RuntimeStatus},
    runtime::session::{DynError, NodeCapabilities},
};
#[cfg(target_os = "macos")]
use serde_json::Value;
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(target_os = "macos")]
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[cfg(target_os = "macos")]
use super::metal_session;
#[cfg(target_os = "macos")]
use super::metal_session::Qwen36MetalSession;

#[cfg(target_os = "macos")]
pub struct Qwen36Engine {
    session: Qwen36MetalSession,
    variant: Qwen36Variant,
    mtp: bool,
    /// DSpark 投机解码是否启用(drafter 已装配且未被 MTP 占用)。
    dspark: bool,
    /// MTP 生效的最大上下文 token 数;0 不限制,超过后 decode 降级普通单行。
    mtp_context_limit: usize,
    /// 每轮 MTP 链式 draft 候选数;1 即 nextn=1 双行 verify。
    mtp_draft_tokens: usize,
    snapshot_resources: metal_session::Qwen36SnapshotResources,
    terminal_states: crate::kv_cache::terminal_cache::TerminalSessions<metal_session::Qwen36TerminalState>,
    residency: FixedSessionResidency,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<RuntimeStatus>>,
    compute_steps: Arc<AtomicCounterU64>,
}

#[cfg(target_os = "macos")]
impl Qwen36Engine {
    pub fn load(
        model_path: &Path,
        max_seq_len: usize,
        variant: Qwen36Variant,
        execution: crate::config::Qwen36NodeExecutionConfig,
        replay_enabled: bool,
        lm_head_quantization: crate::weight::LmHeadQuantization,
        cache_directory: PathBuf,
        persist_kv_cache: bool,
        resident_cache_entries: usize,
        runtime: Arc<Mutex<RuntimeStatus>>,
        compute_steps: Arc<AtomicCounterU64>,
    ) -> Result<Self, DynError> {
        let cache_identity = crate::runtime::session::model_cache_identity(model_path, &format!("qwen36-terminal-v3|variant={variant:?}|max_seq_len={max_seq_len}|execution={execution:?}|lm_head={lm_head_quantization:?}"))?;
        let kv_f16 = execution.kv_cache_format == crate::config::KvCacheFormat::F16;
        let session = Qwen36MetalSession::load(
            model_path,
            max_seq_len,
            execution.precise_gqa_prefill,
            kv_f16,
            execution.prefill_chunk_size,
            execution.vision_max_tokens,
            execution.mtp,
            execution.mtp_draft_vocabulary.as_deref(),
            execution.dspark_directory.as_deref(),
            execution.dspark_draft_tokens,
            replay_enabled,
            lm_head_quantization,
        )
        .map_err(|error| -> DynError { error.into() })?;
        if execution.mtp && !session.mtp_available() {
            return Err("qwen36 model.execution.mtp=true 但 GGUF 不含 nextn MTP 块".into());
        }
        let mtp = session.mtp_available();
        let snapshot_resources = metal_session::Qwen36SnapshotResources { context: session.context_handle(), cfg: session.config().clone(), kv_f16: session.kv_f16(), max_seq_len, mtp };
        let swap = persist_kv_cache
            .then(|| {
                let store = crate::kv_cache::fjall::FjallValueStore::open(cache_directory.join("qwen36-terminal"), "terminal")?;
                store.bind_identity(&cache_identity)?;
                Ok::<_, String>(store)
            })
            .transpose()
            .map_err(|error| -> DynError { format!("Qwen3.6 终点缓存 fjall: {error}").into() })?;
        // terminal_cache_entries=0 在启用持久化时是全 SSD 模式；关闭时终点直接释放。
        let terminal_states = crate::kv_cache::terminal_cache::TerminalSessions::new(resident_cache_entries, swap);
        let modalities = if session.vision_available() { &["text", "image"][..] } else { &["text"][..] };
        let session_residency_bytes = session.session_residency_bytes()?;
        let available = crate::backend::metal::available_residency_bytes(session.context()) as usize;
        let engine_resident_bytes = session.context_handle().device.current_allocated_size() as usize;
        let residency = FixedSessionResidency::new(available, engine_resident_bytes, session_residency_bytes)?;
        let mut capabilities = crate::runtime::metal_node::text_capabilities(
            session.context(),
            crate::runtime::node::SessionDescriptor { model_format: session.model_format(), model_bytes: session.model_bytes(), max_seq_len, kv_cache_format: if kv_f16 { "f16" } else { "q8g64" }, input_modalities: modalities },
        );
        residency.configure(&mut capabilities, &runtime);
        eprintln!("[qwen36-kv-admission] available={:.1} MiB session={:.1} MiB", available as f64 / 1048576.0, session_residency_bytes as f64 / 1048576.0);
        let dspark = session.dspark_available();
        Ok(Self { session, variant, mtp, dspark, mtp_context_limit: execution.mtp_context_limit, mtp_draft_tokens: execution.mtp_draft_tokens, snapshot_resources, terminal_states, residency, capabilities, runtime, compute_steps })
    }
}

#[cfg(target_os = "macos")]
impl Qwen36Engine {
    pub fn model_key(&self) -> &'static str {
        self.variant.model_key()
    }
    pub(crate) fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        crate::runtime::metal_node::startup_info(self.session.context(), &self.capabilities)
    }
    pub(crate) fn refresh_runtime(&self) {
        self.residency.refresh(&self.runtime, &self.terminal_states);
    }
    pub(crate) fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        self.terminal_states.infos()
    }
    pub(crate) fn kv_residency(&self) -> KvResidency {
        self.residency.report(&self.terminal_states)
    }
    fn consume_lazy_engine_growth(&mut self, before: u64) {
        let after = self.session.context_handle().device.current_allocated_size();
        let growth = after.saturating_sub(before) as usize;
        self.residency.consume_engine_growth(growth);
        self.refresh_runtime();
    }
    pub fn shutdown(&mut self) -> Result<(), String> {
        self.terminal_states.persist_resident().map(|_| ())
    }
}

#[cfg(target_os = "macos")]
impl Qwen36Engine {
    pub fn generate(&mut self, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(Option<u32>, String) -> bool) -> Result<GenerationSummary, String> {
        let turns = parse_messages(request)?;
        let image_urls = turns
            .iter()
            .flat_map(|(_, pieces)| {
                pieces.iter().filter_map(|piece| match piece {
                    ContentPiece::Image { url } => Some(url.as_str()),
                    ContentPiece::Text(_) => None,
                })
            })
            .collect::<Vec<_>>();
        if !image_urls.is_empty() && !self.session.vision_available() {
            return Err("图像请求需要模型目录内存在 mmproj-*.gguf 视觉权重".to_owned());
        }
        if !image_urls.is_empty() {
            if self.session.vision_initialized() {
                self.session.ensure_vision()?;
            } else {
                let _engine_growth_reservation = self.residency.reserve_engine_growth(&mut self.terminal_states).map_err(|error| format!("Qwen3.6 视觉资源准入失败: {error}"))?;
                let before = self.session.context_handle().device.current_allocated_size();
                let result = self.session.ensure_vision();
                // prepare 中途失败也可能已创建 Metal 常驻 buffer，必须按
                // 设备实际增长扣账，不能只在成功路径更新 budget。
                self.consume_lazy_engine_growth(before);
                result?;
            }
        }
        // 思考开关与 MiniCPM5 同语义:thinking.type=disabled / enable_thinking=false
        // 关闭;关闭时模板注入官方空围栏,慢硬件上省去每轮数百 token 的思考延迟。
        let thinking_disabled = request.get("thinking").and_then(|value| value.get("type")).and_then(Value::as_str) == Some("disabled") || request.get("enable_thinking").and_then(Value::as_bool) == Some(false);
        let requested_tokens = crate::runtime::session::requested_completion_tokens(request);
        if requested_tokens == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        let stops = parse_stops(request.get("stop"))?;
        // terminal cache 保存的是真实多模态 token/KV/M-RoPE 状态；命中时只渲染
        // 最近 assistant 之后的纯文本 suffix，不再解码或重跑历史图像。
        let (resumed, _kv_reservation) = crate::runtime::session::activate_terminal_append(
            &mut self.terminal_states,
            self.residency.budget(),
            self.residency.session_resident_bytes(),
            request,
            |assistant| Ok(self.session.tokenize(&chat_prompt_suffix(request, assistant, !thinking_disabled)?)),
            &self.snapshot_resources,
        )
        .map_err(|error| format!("Qwen3.6 {error}; 请重新开始会话"))?;
        let full_prefill = if resumed.is_none() {
            // 占位符行数取决于 preprocess 后的 grid；cache miss 才物化历史图像。
            let mut visuals = Vec::with_capacity(image_urls.len());
            for (index, url) in image_urls.iter().enumerate() {
                let image = crate::vision::image_from_url(url).map_err(|error| format!("图像 {index}: {error}"))?;
                let visual = self.session.preprocess_image(&image)?;
                eprintln!("[qwen36-node-vision] image {index} {}x{} visual_tokens={}", image.width, image.height, visual.tensor.visual_token_count()?);
                visuals.push(visual);
            }
            let blocks = if visuals.is_empty() { Vec::new() } else { qwen36::qwen36_visual_blocks(self.session.config(), &visuals)?.into_iter().map(|(block, _, _)| block).collect() };
            let prompt = render_request_prompt(request, &turns, &blocks, !thinking_disabled)?;
            let input = if visuals.is_empty() { None } else { Some(self.session.multimodal_input(&prompt, &visuals)?) };
            let tokens = input.as_ref().map_or_else(|| self.session.tokenize(&prompt), |input| input.token_ids.clone());
            if tokens.is_empty() {
                return Err("Qwen3.6 prompt 不能为空".to_owned());
            }
            if tokens.len() >= self.session.max_seq_len() {
                return Err(format!("Qwen3.6 prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.session.max_seq_len()));
            }
            Some((visuals, input, tokens))
        } else {
            None
        };
        // resume 命中时只对 suffix 计费
        let batch_tokens = resumed.as_ref().map_or_else(|| full_prefill.as_ref().map_or(0, |(_, _, tokens)| tokens.len()), |(state, suffix)| state.pending.len().saturating_add(suffix.len()));
        let _batch_guard = BatchTokenGuard::new(&self.runtime, batch_tokens);
        let mut sequence = if let Some((state, suffix)) = resumed {
            let mut sequence = state.sequence;
            let mut append = state.pending;
            append.extend(suffix);
            self.session.extend(&mut sequence, &append)?;
            sequence
        } else {
            let (visuals, input, tokens) = full_prefill.as_ref().expect("cache miss 必须准备完整 prefill");
            match input {
                Some(input) => self.session.prefill_visual(input, visuals)?,
                None => self.session.prefill(tokens.clone())?,
            }
        };
        let prompt_tokens = sequence.token_count();
        let max_tokens = requested_tokens.min(self.session.max_seq_len() - sequence.token_count());
        if max_tokens == 0 {
            return Err("会话已没有可用的生成位置".to_owned());
        }
        let mut output = GenerationOutput::new(&stops);
        let tool_scope = crate::runtime::tool::request_tool_scope(self.variant.model_key(), request);
        let mut tool_stream = RequestToolCallStream::new(request, &tool_scope, ToolDialect::ChatmlJson);
        let mut response_text = String::new();
        let mut token = self.session.token_output(&sequence)?;
        // MTP 循环结束后,length 截断时待前向的最后一个采样 token(terminal cache 续写用)
        let mut final_pending_token = token;
        let mut step = 0usize;
        // MTP 长上下文降级标记:verify 双行扫描超过接受收益后转普通单行循环
        let mut degrade_to_plain = false;
        // 图文请求禁止投机解码：MTP/DSpark drafter 不接收视觉 soft-token
        // 与 M-RoPE 状态；纯文本请求仍保留投机解码。
        let speculative_allowed = image_urls.is_empty();
        // 单 token 的流式 emit(stop 命中/UTF-8 边界)。返回 false 表示已到终态,
        // output 已更新,调用方直接跳出。
        macro_rules! emit_token {
            ($token:expr) => {{
                let token: u32 = $token;
                if self.session.is_eos(token) {
                    output.stop();
                    false
                } else {
                    let bytes = self.session.decode_bytes(token)?;
                    output.push(&bytes, |chunk| {
                        tool_stream.push(&chunk, |visible| {
                            if !on_token(Some(token), visible.clone()) {
                                return false;
                            }
                            response_text.push_str(&visible);
                            true
                        })
                    })
                }
            }};
        }
        let dspark_active = speculative_allowed && self.dspark && self.session.dspark_ready(&sequence);
        let mtp_active = speculative_allowed && self.mtp;
        if dspark_active {
            // DSpark 投机解码:drafter 整块出 k 个候选,主干一次 k+1 行 verify,
            // greedy 逐 token 校验(构造上无损);部分接受时回滚 GDN/KV 游标并
            // 重放接受前缀。resume 会话的 drafter cache 为空,自动落回普通 decode。
            let eos_tokens = self.session.config().eos_token_ids.clone();
            let mut pending = token;
            // 上一轮接受前缀的 captures(R 行 @ aux_position),draft 时消费进 cache
            let mut aux: Option<(Vec<MetalTensor>, usize)> = None;
            let mut accepted_total = 0usize;
            let mut proposed_total = 0usize;
            let mut total_rounds = 0usize;
            loop {
                if cancellation.load(Ordering::Acquire) {
                    output.cancel();
                    break;
                }
                if output.completion_tokens() >= max_tokens {
                    break;
                }
                if !emit_token!(pending) {
                    break;
                }
                if output.completion_tokens() >= max_tokens {
                    break;
                }
                // 长上下文里 verify k+1 行每步扫多份 KV(mlx-dspark 实测 14K 约
                // 1.3x、32K 固定宽亏),越过配置阈值降级单行;mtp_context_limit 复用。
                if self.mtp_context_limit > 0 && sequence.tokens.len() >= self.mtp_context_limit {
                    degrade_to_plain = true;
                    break;
                }
                let position = sequence.tokens.len();
                let debug_timing = std::env::var_os("ZLLM_QWEN36_DSPARK_DEBUG").is_some();
                let phase_started = std::time::Instant::now();
                let mut block = match &aux {
                    // 投影放 draft 内:capture 行进 drafter target cache 后出块
                    Some((captures, aux_position)) => self.session.dspark_draft_with_captures(&mut sequence, pending, captures, *aux_position, position)?,
                    None => self.session.dspark_draft(&mut sequence, pending, None, position)?,
                };
                let round_limit = max_tokens.saturating_sub(output.completion_tokens()).min(self.session.max_seq_len().saturating_sub(position.saturating_add(1)));
                block.drafts.truncate(round_limit);
                block.confidences.truncate(block.drafts.len());
                total_rounds += 1;
                proposed_total += block.drafts.len();
                let drafts = block.drafts.clone();
                let draft_wall = phase_started.elapsed();
                let verify_started = std::time::Instant::now();

                // 快照 → k+1 行 verify(带 capture)→ 逐行采样真值
                let snapshot_started = std::time::Instant::now();
                let delta_snapshot = self.session.snapshot_delta(&sequence)?;
                let kv_lengths = self.session.snapshot_kv_lengths(&sequence);
                let snapshot_wall = snapshot_started.elapsed();
                let mut rows = Vec::with_capacity(drafts.len() + 1);
                rows.push(pending);
                rows.extend_from_slice(&drafts);
                let mut captures = Vec::new();
                let mut gdn_inputs = Vec::new();
                let output = self.session.forward_rows_with_capture(&mut sequence, &rows, &mut captures, &mut gdn_inputs)?;
                let forward_wall = verify_started.elapsed();
                let sample_started = std::time::Instant::now();
                let target_tokens = self.session.rows_token_output(&output)?;
                let sample_wall = sample_started.elapsed();
                self.compute_steps.fetch_add(rows.len() as u64);
                let verify_wall = verify_started.elapsed();

                // 无损校验:tokens 依序 emit;retained 行是 state 应保留的前缀
                let verification = crate::runtime::speculative::verify_samples(&target_tokens, &drafts, &eos_tokens).map_err(|error| format!("DSpark verify: {error:?}"))?;
                if std::env::var_os("ZLLM_QWEN36_DSPARK_DEBUG").is_some() && total_rounds <= 24 {
                    let show = |ids: &[u32]| ids.iter().map(|&id| String::from_utf8_lossy(&self.session.decode_bytes(id).unwrap_or_default()).to_string()).collect::<Vec<_>>().join("|");
                    eprintln!("[dspark-dbg] round={} pos={} anchor=[{}] drafts=[{}] targets=[{}] accepted={}", total_rounds, position, show(&[pending]), show(&drafts), show(&target_tokens), verification.accepted_drafts);
                }
                accepted_total += verification.accepted_drafts;
                let retained = verification.retained_rows;
                // 与 MTP 同款语义:pending 每轮在 loop 顶 emit;verify 产出的
                // tokens 只 emit 前 n-1 个,最后一个成为下一轮的 pending。
                let mut stop = false;
                for token in verification.emitted_tokens() {
                    final_pending_token = *token;
                    if !emit_token!(*token) {
                        stop = true;
                        break;
                    }
                }
                if stop {
                    break;
                }
                pending = verification.pending_token();
                final_pending_token = pending;
                let reject_started = std::time::Instant::now();
                let (captures, committed_hidden) = if retained == rows.len() {
                    let hidden = self.session.context().select_row(&output, rows.len() - 1).map_err(|error| format!("Qwen3.6 DSpark 末行: {error:?}"))?;
                    (captures, hidden)
                } else if std::env::var_os("ZLLM_QWEN36_DSPARK_FULL_REPLAY").is_some() {
                    // 完整重放(基线路径,GDN-only 的对照)
                    sequence.tokens.truncate(position);
                    self.session.restore_delta(&mut sequence, &delta_snapshot, position)?;
                    self.session.restore_kv_lengths(&mut sequence, &kv_lengths)?;
                    let mut replay_captures = Vec::new();
                    let mut replay_gdn = Vec::new();
                    let output1 = self.session.forward_rows_with_capture(&mut sequence, &rows[..retained], &mut replay_captures, &mut replay_gdn)?;
                    self.compute_steps.fetch_add(retained as u64);
                    let hidden = self.session.context().select_row(&output1, retained - 1).map_err(|error| format!("Qwen3.6 DSpark 重放末行: {error:?}"))?;
                    (replay_captures, hidden)
                } else {
                    // GDN-only 重放:verify 的前 R 行处理的正是被接受 token。KV 行
                    // 保留只回退游标;GDN state 从快照恢复后仅重放 R 行 DeltaNet 链
                    // (~30ms),不再重跑整条 R 行前向(~200ms)。
                    sequence.tokens.truncate(position);
                    sequence.tokens.extend_from_slice(&rows[..retained]);
                    self.session.restore_delta(&mut sequence, &delta_snapshot, position)?;
                    self.session.advance_kv_lengths(&mut sequence, position + retained)?;
                    self.session.replay_gdn(&mut sequence, &gdn_inputs, retained, position)?;
                    self.compute_steps.fetch_add(retained as u64);
                    let hidden = self.session.context().select_row(&output, retained - 1).map_err(|error| format!("Qwen3.6 DSpark 接受末行: {error:?}"))?;
                    use crate::backend::SegmentedTensorBackend as _;
                    let sliced = captures.iter().map(|tensor| self.session.context().slice_token_rows(tensor, 0, retained).map_err(|error| format!("DSpark capture 切片: {error:?}"))).collect::<Result<Vec<_>, _>>()?;
                    (sliced, hidden)
                };
                sequence.hidden = committed_hidden;
                let reject_wall = reject_started.elapsed();
                let extend_started = std::time::Instant::now();
                let advanced = self.session.dspark_extend(&mut sequence.dspark_cache, &captures, position).map_err(|error| format!("DSpark cache 推进: {error}"))?;
                let _ = advanced;
                if debug_timing && total_rounds <= 12 {
                    eprintln!(
                        "[dspark-time] round={total_rounds} draft={:.3}s snapshot={:.3}s verify={:.3}s(fwd={:.3}s sample={:.3}s) reject={:.3}s extend={:.3}s retained={retained}/{}",
                        draft_wall.as_secs_f32(),
                        snapshot_wall.as_secs_f32(),
                        verify_wall.as_secs_f32(),
                        forward_wall.as_secs_f32(),
                        sample_wall.as_secs_f32(),
                        reject_wall.as_secs_f32(),
                        extend_started.elapsed().as_secs_f32(),
                        rows.len()
                    );
                }
                aux = Some((captures, position));
            }
            if total_rounds > 0 {
                eprintln!("[qwen36-dspark] rounds={} proposed={} accepted={} rate={:.2}", total_rounds, proposed_total, accepted_total, accepted_total as f64 / proposed_total as f64);
            }
            if degrade_to_plain {
                // pending 已 emit 未前向:单行推进一步并采样,衔接普通 decode 循环
                self.session.decode_token(&mut sequence, pending)?;
                token = self.session.token_output(&sequence)?;
                step = output.completion_tokens();
            }
        } else if mtp_active {
            // MTP 投机解码(nextn 链):pending 是待 emit 且待前向的 token,
            // hidden_prev 是主干对 pending 前一 token 的输出。每轮 K 步链式
            // draft(EAGLE 式自回归,MTP slot 逐 token 推进),主干一次 K+1 行
            // verify 贪心校验;全盘接受直接续跑,部分接受则回滚 GDN state 与
            // KV 游标并 GDN-only 重放接受前缀(~30ms,不再整行重放)。
            // MTP slot 每轮只保留 pending 的真实条目,链上 draft 条目截断,
            // 由后续轮次按真值 hidden 重新写入。
            let draft_tokens = self.mtp_draft_tokens.max(1);
            let eos_tokens = self.session.config().eos_token_ids.clone();
            let mut pending = token;
            let mut hidden_prev = sequence.hidden.clone();
            let mut accepted_total = 0usize;
            let mut full_rounds = 0usize;
            let mut total_rounds = 0usize;
            loop {
                if cancellation.load(Ordering::Acquire) {
                    output.cancel();
                    break;
                }
                if output.completion_tokens() >= max_tokens {
                    break;
                }
                if !emit_token!(pending) {
                    break;
                }
                if output.completion_tokens() >= max_tokens {
                    break;
                }
                // 长上下文里 verify K+1 行每步扫多份 KV,成本超过接受收益
                // (K=1 实测 24GB/M5:2.2K +9%、8.7K -10%、15.6K -43%),越过
                // 配置阈值降级单行。
                if self.mtp_context_limit > 0 && sequence.tokens.len() >= self.mtp_context_limit {
                    degrade_to_plain = true;
                    break;
                }
                let profile_mtp = std::env::var_os("ZLLM_QWEN36_MTP_PROFILE").is_some();
                let draft_started = std::time::Instant::now();
                if profile_mtp {
                    self.session.context().reset_gpu_stats();
                }
                let position = sequence.tokens.len();
                let round_draft_tokens = draft_tokens.min(max_tokens.saturating_sub(output.completion_tokens())).min(self.session.max_seq_len().saturating_sub(position.saturating_add(1)));
                if round_draft_tokens == 0 {
                    degrade_to_plain = true;
                    break;
                }
                let drafts = self.session.mtp_draft_chain(&mut sequence, pending, &hidden_prev, round_draft_tokens)?;
                let draft_wall = draft_started.elapsed();
                total_rounds += 1;
                if profile_mtp {
                    let gpu = self.session.context().gpu_stats();
                    eprintln!("[qwen36-mtp-step] draft K={} wall={:.3}s gpu={:.3}s cmds={}", draft_tokens, draft_wall.as_secs_f64(), gpu.seconds, gpu.command_buffers);
                }
                let mtp_slot_base = sequence.cache.layer_len(self.session.config().num_layers);
                let delta_snapshot = self.session.snapshot_delta(&sequence)?;
                let verify_started = std::time::Instant::now();
                if profile_mtp {
                    self.session.context().reset_gpu_stats();
                }
                let mut rows = Vec::with_capacity(drafts.len() + 1);
                rows.push(pending);
                rows.extend_from_slice(&drafts);
                let mut gdn_inputs = Vec::new();
                let output = self.session.forward_rows_with_gdn_inputs(&mut sequence, &rows, &mut gdn_inputs)?;
                let target_tokens = self.session.rows_token_output(&output)?;
                self.compute_steps.fetch_add(rows.len() as u64);
                let verify_wall = verify_started.elapsed();
                if profile_mtp {
                    let gpu = self.session.context().gpu_stats();
                    eprintln!("[qwen36-mtp-step] verify rows={} wall={:.3}s gpu={:.3}s cmds={}", rows.len(), verify_wall.as_secs_f64(), gpu.seconds, gpu.command_buffers);
                    for operator in self.session.context().gpu_profile().into_iter().take(8) {
                        eprintln!("  mtp verify gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
                    }
                }

                // 贪心逐 token 校验:tokens 是本轮新提交的真值(不含 pending),
                // 末位成为下一轮 pending;retained_rows 是 state 应保留的行数。
                let verification = crate::runtime::speculative::verify_samples(&target_tokens, &drafts, &eos_tokens).map_err(|error| format!("MTP verify: {error:?}"))?;
                accepted_total += verification.accepted_drafts;
                if verification.retained_rows == rows.len() {
                    full_rounds += 1;
                }
                let retained = verification.retained_rows;
                let mut stop = false;
                let committed = &verification.tokens;
                for token in &committed[..committed.len() - 1] {
                    final_pending_token = *token;
                    if !emit_token!(*token) {
                        stop = true;
                        break;
                    }
                }
                if stop {
                    break;
                }
                pending = *committed.last().expect("verify 至少产出一个 token");
                final_pending_token = pending;
                // verify 行因果:行 j 的 hidden 只依赖前 j+1 个 token,任意接受
                // 长度下 retained-1 行都是下一轮 pending 前一 token 的真值 hidden。
                hidden_prev = self.session.context().select_row(&output, retained - 1).map_err(|error| format!("Qwen3.6 MTP verify 行: {error:?}"))?;
                sequence.hidden = hidden_prev.clone();
                if retained < rows.len() {
                    // 部分接受:KV 行保留只回退游标;GDN state 从快照恢复后仅
                    // 重放接受前缀的 DeltaNet 链(与 DSpark 逐位一致的回滚语义)。
                    sequence.tokens.truncate(position);
                    sequence.tokens.extend_from_slice(&rows[..retained]);
                    self.session.restore_delta(&mut sequence, &delta_snapshot, position)?;
                    self.session.advance_kv_lengths(&mut sequence, position + retained)?;
                    self.session.replay_gdn(&mut sequence, &gdn_inputs, retained, position)?;
                    self.compute_steps.fetch_add(retained as u64);
                }
                // MTP slot 截断到 pending 的真实条目:链上 draft 条目是链式 hidden
                // 产物,下一轮首步会用真值 hidden 重新写 committed 末行的条目。
                sequence.cache.set_layer_len(self.session.config().num_layers, mtp_slot_base + 1)?;
            }
            if total_rounds > 0 {
                eprintln!("[qwen36-mtp] K={} rounds={} accepted={} full={} rate={:.2}", draft_tokens, total_rounds, accepted_total, full_rounds, accepted_total as f64 / (total_rounds * draft_tokens) as f64);
            }
            if degrade_to_plain {
                // pending 已 emit 未前向:单行推进一步并采样,衔接普通 decode 循环
                self.session.decode_token(&mut sequence, pending)?;
                token = self.session.token_output(&sequence)?;
                step = output.completion_tokens();
            }
        }
        if degrade_to_plain || (!dspark_active && !mtp_active) {
            while step < max_tokens {
                if cancellation.load(Ordering::Acquire) {
                    output.cancel();
                    break;
                }
                if !emit_token!(token) {
                    break;
                }
                step += 1;
                if step == max_tokens {
                    break;
                }
                let profile_step = std::env::var_os("ZLLM_QWEN36_PROFILE").is_some();
                if profile_step {
                    self.session.context().reset_gpu_stats();
                }
                let step_started = std::time::Instant::now();
                self.session.decode_token(&mut sequence, token)?;
                let sample_started = std::time::Instant::now();
                let forward_wall = step_started.elapsed();
                if profile_step {
                    let gpu = self.session.context().gpu_stats();
                    let forward_gpu = gpu;
                    eprintln!("[qwen36-node-decode] step={step} forward_wall={:.3}s forward_gpu={:.3}s commands={}", forward_wall.as_secs_f64(), forward_gpu.seconds, forward_gpu.command_buffers);
                    if matches!(step, 1 | 16 | 32) {
                        for operator in self.session.context().gpu_profile().into_iter().take(14) {
                            eprintln!("  qwen36 decode gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
                        }
                    }
                }
                // 无锁递增：避免每个 decode step 与 heartbeat / refresh 抢 Mutex。
                self.compute_steps.fetch_add(1);
                token = self.session.token_output(&sequence)?;
                if profile_step && matches!(step, 1 | 16 | 32) {
                    eprintln!("[qwen36-node-decode] step={step} sample_wall={:.3}s (lm_head+argmax)", sample_started.elapsed().as_secs_f64());
                }
            }
        }
        output.finish(|chunk| {
            tool_stream.push(&chunk, |visible| {
                if !on_token(None, visible.clone()) {
                    return false;
                }
                response_text.push_str(&visible);
                true
            })
        });
        if !output.is_cancelled()
            && !tool_stream.finish(|visible| {
                if !on_token(None, visible.clone()) {
                    return false;
                }
                response_text.push_str(&visible);
                true
            })
        {
            output.cancel();
        }
        if !output.is_cancelled() && !tool_stream.calls.is_empty() {
            output.mark_tool_calls();
        }
        let mut summary = output.summary(prompt_tokens);
        // 长度截断时最后采样的 token 尚未前向,作为 pending 保留给下一轮续写;
        // stop/eos 结束时该 token 不属于下一轮 prompt,直接丢弃
        // 多模态状态同样保留；快照中的真实 token、KV 与 rope_delta 允许后续
        // 纯文本 suffix 直接续写，避免历史 image_url 每轮重新视觉编码。
        if summary.finish_reason != "cancelled" && stops.is_empty() {
            let pending = if summary.finish_reason == "length" { vec![final_pending_token] } else { Vec::new() };
            let cache_id = crate::runtime::session::terminal_cache_id(request, &response_text, &tool_stream.calls)?;
            let info = CacheInfo {
                cache_id,
                model_key: self.variant.model_key().to_owned(),
                cache_format: "qwen36-terminal-v2".to_owned(),
                last_layer: self.session.layer_count().saturating_sub(1),
                prompt_tokens: sequence.token_count(),
                bytes: self.residency.session_resident_bytes() as u64,
                modified_unix: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
            };
            let state = metal_session::Qwen36TerminalState { sequence, pending, info: info.clone() };
            summary.cache = crate::runtime::session::retain_terminal_session(&mut self.terminal_states, state, info);
        }
        summary.tool_calls = std::mem::take(&mut tool_stream.calls);
        Ok(summary)
    }
}
