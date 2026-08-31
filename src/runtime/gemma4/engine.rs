//! Gemma 4 × Metal 嵌入式推理引擎：统一请求准备、session 与生成生命周期。
#[cfg(target_os = "macos")]
use super::protocol::{chat_prompt_suffix, chat_segments, segments_text};
#[cfg(target_os = "macos")]
use crate::runtime::session::{BatchTokenGuard, FixedSessionResidency, GenerationOutput, GenerationSummary, KvResidency};
#[cfg(target_os = "macos")]
use crate::runtime::session::{ContentPiece, parse_stops, with_content_parts};
#[cfg(target_os = "macos")]
use crate::{
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
use super::metal_session::Gemma4MetalSession;

#[cfg(target_os = "macos")]
pub struct Gemma4Engine {
    session: Gemma4MetalSession,
    snapshot_resources: metal_session::Gemma4SnapshotResources,
    terminal_states: crate::kv_cache::terminal_cache::TerminalSessions<metal_session::Gemma4TerminalState>,
    residency: FixedSessionResidency,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<RuntimeStatus>>,
    compute_steps: Arc<AtomicCounterU64>,
}

#[cfg(target_os = "macos")]
impl Gemma4Engine {
    pub fn load(
        model_path: &Path,
        max_seq_len: usize,
        execution: crate::config::Gemma4ExecutionConfig,
        replay_enabled: bool,
        lm_head_quantization: crate::weight::LmHeadQuantization,
        cache_directory: PathBuf,
        persist_kv_cache: bool,
        resident_cache_entries: usize,
        runtime: Arc<Mutex<RuntimeStatus>>,
        compute_steps: Arc<AtomicCounterU64>,
    ) -> Result<Self, DynError> {
        let cache_identity = crate::runtime::session::model_cache_identity(model_path, &format!("gemma4-terminal-v3|max_seq_len={max_seq_len}|execution={execution:?}|lm_head={lm_head_quantization:?}"))?;
        let mtp_requested = execution.mtp_weights.is_some();
        let mut session =
            Gemma4MetalSession::load(model_path, max_seq_len, execution.prefill_chunk_size, lm_head_quantization, execution.mtp_weights, execution.mtp_draft_tokens, replay_enabled).map_err(|error| -> DynError { error.into() })?;
        if session.accepts_images() {
            session.prepare_multimodal_resources().map_err(|error| -> DynError { error.into() })?;
        }
        if mtp_requested {
            // MTP 是引擎常驻资源，装载后再拍 session admission 基线，不能把它算进 KV 可用容量。
            session.ensure_mtp().map_err(|error| -> DynError { error.into() })?;
        }
        let snapshot_resources = metal_session::Gemma4SnapshotResources { context: session.context_handle(), hybrid_gqa: session.hybrid_gqa_spec(), max_seq_len };
        let swap = persist_kv_cache
            .then(|| {
                let store = crate::kv_cache::fjall::FjallValueStore::open(cache_directory.join("gemma4-terminal"), "terminal")?;
                store.bind_identity(&cache_identity)?;
                Ok::<_, String>(store)
            })
            .transpose()
            .map_err(|error| -> DynError { format!("Gemma4 终点缓存 fjall: {error}").into() })?;
        let terminal_states = crate::kv_cache::terminal_cache::TerminalSessions::new(resident_cache_entries, swap);
        let modalities = if session.accepts_images() { &["text", "image"][..] } else { &["text"][..] };
        let session_residency_bytes = session.session_residency_bytes()?;
        let available = crate::backend::metal::available_residency_bytes(session.context()) as usize;
        let engine_resident_bytes = session.context().device.current_allocated_size() as usize;
        let residency = FixedSessionResidency::new(available, engine_resident_bytes, session_residency_bytes)?;
        let mut capabilities = crate::runtime::metal_node::text_capabilities(
            session.context(),
            crate::runtime::node::SessionDescriptor { model_format: session.model_format(), model_bytes: session.model_bytes(), max_seq_len, kv_cache_format: "f16", input_modalities: modalities },
        );
        residency.configure(&mut capabilities, &runtime);
        eprintln!("[gemma4-kv-admission] available={:.1} MiB session={:.1} MiB", available as f64 / 1048576.0, session_residency_bytes as f64 / 1048576.0);
        Ok(Self { session, snapshot_resources, terminal_states, residency, capabilities, runtime, compute_steps })
    }
}

#[cfg(target_os = "macos")]
impl Gemma4Engine {
    pub(crate) fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        crate::runtime::metal_node::startup_info(self.session.context(), &self.capabilities)
    }
    pub(crate) fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        self.terminal_states.infos()
    }
    pub(crate) fn refresh_runtime(&self) {
        self.residency.refresh(&self.runtime, &self.terminal_states);
    }
    pub(crate) fn kv_residency(&self) -> KvResidency {
        self.residency.report(&self.terminal_states)
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        self.terminal_states.persist_resident().map(|_| ())
    }
}

#[cfg(target_os = "macos")]
impl Gemma4Engine {
    /// 执行一个完整结构化请求。Node 与嵌入式入口共享此路径，因此模板、tokenize、
    /// terminal cache 命中、pending token 与 stop 语义不会分叉。
    pub fn generate(&mut self, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(Option<u32>, String) -> bool) -> Result<GenerationSummary, String> {
        let request_started = std::time::Instant::now();
        let segments = chat_segments(request)?;
        let has_images = segments.iter().any(|segment| matches!(segment, ContentPiece::Image { .. }));
        // 图文请求:parts 保序展开,图像就地展开为 soft-token 段(GGUF Jinja 模板无法表达
        // soft-token 展开,图文一律走硬编码 thought 模板)。纯文本保持原模板路径。
        let (tokens, multimodal) = if has_images {
            let input = with_content_parts(&segments, |parts| self.session.multimodal_input(parts))?;
            (input.token_ids.clone(), Some(input))
        } else {
            // GGUF 来源用自带的 Jinja chat template（含 tools）；其余来源走硬编码 thought 模板。
            let prompt = match self.session.chat_template() {
                Some(template) => template.render(request)?,
                None => segments_text(&segments),
            };
            (self.session.tokenize(&prompt), None)
        };
        let requested_tokens = crate::runtime::session::requested_completion_tokens(request);
        if requested_tokens == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        if tokens.is_empty() {
            return Err("Gemma4 prompt 不能为空".to_owned());
        }
        if tokens.len() >= self.session.max_seq_len() {
            return Err(format!("Gemma4 prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.session.max_seq_len()));
        }
        let stops = parse_stops(request.get("stop"))?;
        // 图文请求暂不走终点缓存:resume 的 suffix 是纯文本续写,快照不含视觉区间信息
        let (resumed, _kv_reservation) = if multimodal.is_some() {
            let (state, reservation) = self
                .terminal_states
                .activate(None, self.residency.budget(), self.residency.session_resident_bytes(), |_, state| state.info.bytes as usize, &self.snapshot_resources)
                .map_err(|error| format!("Gemma4 {error}; 请重新开始会话"))?;
            debug_assert!(state.is_none());
            (None, reservation)
        } else {
            crate::runtime::session::activate_terminal_append(
                &mut self.terminal_states,
                self.residency.budget(),
                self.residency.session_resident_bytes(),
                request,
                |assistant| Ok(self.session.tokenize(&chat_prompt_suffix(request, assistant)?)),
                &self.snapshot_resources,
            )
            .map_err(|error| format!("Gemma4 {error}; 请重新开始会话"))?
        };
        // resume 命中时只对 suffix 计费
        let batch_tokens = resumed.as_ref().map_or(tokens.len(), |(state, suffix)| state.pending.len().saturating_add(suffix.len()));
        let _batch_guard = BatchTokenGuard::new(&self.runtime, batch_tokens);
        let profile_prefill = std::env::var_os("ZLLM_GEMMA4_PROFILE_PREFILL").is_some();
        if profile_prefill {
            self.session.context_handle().reset_gpu_stats();
        }
        let prefill_started = std::time::Instant::now();
        let mut sequence = if let Some(input) = multimodal.as_ref() {
            self.session.prefill_multimodal(input)?
        } else if let Some((state, suffix)) = resumed {
            let mut sequence = state.sequence;
            let mut append = state.pending;
            append.extend(suffix);
            self.session.extend(&mut sequence, &append)?;
            sequence
        } else {
            self.session.prefill(tokens.clone())?
        };
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();
        if profile_prefill {
            let gpu = self.session.context_handle().gpu_stats();
            let operators = self.session.context_handle().gpu_profile();
            eprintln!(
                "[gemma4-prefill-gpu] wall={prefill_seconds:.3}s gpu={:.3}s commands={} operators={} submit_wait={:.3}s gaps={:.3}s tail={:.3}s",
                gpu.seconds,
                gpu.command_buffers,
                operators.len(),
                gpu.submit_wait_seconds,
                gpu.inter_command_gap_seconds,
                gpu.completion_tail_seconds
            );
            for operator in operators.into_iter().take(20) {
                eprintln!("  gemma4 prefill gpu {:>9.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
            }
        }
        let max_tokens = requested_tokens.min(self.session.max_seq_len() - sequence.token_count());
        if max_tokens == 0 {
            return Err("会话已没有可用的生成位置".to_owned());
        }
        let mut output = GenerationOutput::new(&stops);
        // 图文请求禁止投机解码：MTP drafter 不接收视觉 soft-token，不能复现
        // 主模型的图文状态。纯文本请求仍保留 MTP。
        let use_mtp = multimodal.is_none();
        // 请求级计时不触碰 GPU 同步；逐步算子统计会 reset/read GPU stats，
        // 必须单独显式开启，避免 profiling 本身把 decode 吞吐压低。
        let profile_request = std::env::var_os("ZLLM_GEMMA4_PROFILE").is_some();
        let profile_step = std::env::var_os("ZLLM_GEMMA4_PROFILE_STEPS").is_some();
        let mut first_token_at = None;
        let mut decode_50_at = None;
        // replay 必须在首次真实 prefill 后录制；启动期提前录制会污染随后分配的
        // prefill buffer。惰性常驻增长在本次成功分配后永久扣减后续 session 容量。
        let replay_growth_before = (!self.session.replay_resources_prepared()).then(|| self.session.context_handle().device.current_allocated_size());
        // 重放引擎首请求录制/后续请求绑定 KV cache;必须在 session 不可变借用(emit 闭包)之前完成。
        let replay_decode = self.session.replay_decode_available();
        if profile_request {
            eprintln!("[gemma4-request-route] multimodal={} mtp={} replay={}", multimodal.is_some(), use_mtp && self.session.mtp_available(), replay_decode);
        }
        if replay_decode {
            self.session.ensure_replay(&sequence.cache)?;
        }
        if use_mtp {
            self.session.ensure_mtp()?;
        }
        if use_mtp && self.session.mtp_enabled() {
            self.session.ensure_mtp_replay(&sequence.cache)?;
            self.session.ensure_verify_replay(&sequence.cache, self.session.mtp_draft_tokens() + 1)?;
        }
        if let Some(before) = replay_growth_before {
            let growth = self.session.context_handle().device.current_allocated_size().saturating_sub(before) as usize;
            self.residency.consume_engine_growth(growth);
            self.refresh_runtime();
        }
        let session = &self.session;
        // 消费一个已采样 token:EOS / stop 词判定与流式发射；false 表示生成停止。
        macro_rules! emit_token {
            ($token:expr) => {{
                let token: u32 = $token;
                (|| -> Result<bool, String> {
                    first_token_at.get_or_insert_with(std::time::Instant::now);
                    if session.is_eos(token) {
                        output.stop();
                        return Ok(false);
                    }
                    // 与 llama-bench `tg50` 对齐：首 token 后第 50 次 decode 转移只记
                    // 一个时间点，不引入逐步同步或 GPU profiling 扰动。
                    if output.completion_tokens() == 50 {
                        decode_50_at = Some(std::time::Instant::now());
                    }
                    let bytes = session.decode_bytes(token)?;
                    Ok(output.push(&bytes, |chunk| on_token(Some(token), chunk)))
                })()
            }};
        }
        // 最后一次采样的 token:length 截断时它尚未前向,retain 时作为 pending 留给续写。
        let mut last_token: Option<u32> = None;
        let async_decode = replay_decode;
        // 重放路径自管 CB(每步一个独立 CB),不能进 deferred batch 窗口
        // (窗口内 ctx.command_buffer() 返回共享 batch CB,与重放的独立 commit 冲突)。
        if async_decode && !replay_decode {
            session.begin_async_decode();
        }
        let loop_result = (|| -> Result<(), String> {
            // MTP 投机:官方 4 层 draft 头 + 主干多行 verify。接受判定走模型无关的
            // runtime::speculative::verify_samples(qwen36/glm52 同款),draft 读主干 KV。
            if use_mtp && session.mtp_available() {
                let mtp = session.mtp();
                let context = session.context();
                let draft_tokens = session.mtp_draft_tokens().max(1);
                let eos_tokens = session.eos_token_ids();
                let mut stats = crate::runtime::speculative::SpeculativeStats::default();
                let mut accepted_at_position = vec![0u64; draft_tokens.max(1)];
                let first_output = session.first_token_output(&sequence)?;
                let first = first_output.token_id;
                let first_normed = session.normed_hidden_row_f32(&first_output.input, first_output.input.rows - 1)?;
                let backbone_hidden_size = mtp.backbone_hidden_size();
                if first_normed.len() != backbone_hidden_size {
                    return Err(format!("Gemma4 MTP 主干 hidden={}，assistant 期望 {backbone_hidden_size}", first_normed.len()));
                }
                // first 不在此 emit:它是首个 pending,由轮首的延迟发射覆盖。
                let mut emitted_total = 0usize;
                let mut pending = first;
                // draft 链的 h 起点:上轮 verify 第 accepted 行的 normed hidden
                // (llama.cpp accept() 的 pending_h 语义);verify_normed 为多行张量。
                let mut verify_normed_rows: Option<Vec<f32>> = None;
                let mut accepted_prev = 0usize;
                while emitted_total < max_tokens {
                    if cancellation.load(Ordering::Acquire) {
                        output.cancel();
                        break;
                    }
                    // 轮首 emit 上轮 pending(已被 verify 确认为 target 真值;
                    // qwen36 同款延迟一轮发射)。
                    last_token = Some(pending);
                    if !emit_token!(pending)? {
                        return Ok(());
                    }
                    emitted_total += 1;
                    if emitted_total >= max_tokens {
                        break;
                    }
                    let round_started = std::time::Instant::now();
                    let position = sequence.tokens.len();
                    let round_draft_tokens = draft_tokens.min(max_tokens.saturating_sub(emitted_total)).min(session.max_seq_len().saturating_sub(position.saturating_add(1)));
                    if round_draft_tokens == 0 {
                        break;
                    }
                    let mut chain_hidden = match verify_normed_rows.as_ref() {
                        Some(normed) => {
                            let row = accepted_prev.min(std::env::var_os("ZLLM_GEMMA4_MTP_HCAP1").map_or(usize::MAX, |_| 1));
                            normed[row * backbone_hidden_size..(row + 1) * backbone_hidden_size].to_vec()
                        }
                        None => first_normed.clone(),
                    };
                    let draft_started = std::time::Instant::now();
                    let mut drafts = Vec::with_capacity(round_draft_tokens);
                    let mut token = pending;
                    for _ in 0..round_draft_tokens {
                        let embedding = session.embedding_row(token)?;
                        let mut input = Vec::with_capacity(mtp.concat_columns());
                        input.extend_from_slice(&embedding);
                        input.extend_from_slice(&chain_hidden);
                        // 共享 KV:draft 所有步用同一 position(llama.cpp gemma4 分支)。
                        // 重放路径:命令表 position 无关,步时 17.6ms vs 同步 44ms。
                        let draft = session.mtp_replay().step(context, mtp, &sequence.cache, &input, position).map_err(|error| format!("Gemma4 MTP draft 重放: {error:?}"))?;
                        chain_hidden = session.mtp_replay().h_next_f32(context);
                        drafts.push(draft);
                        token = draft;
                    }
                    let draft_seconds = draft_started.elapsed().as_secs_f64();
                    // verify:pending + drafts 一次 K+1 行前向(KV 一次 append)。
                    let mut rows = Vec::with_capacity(drafts.len() + 1);
                    rows.push(pending);
                    rows.extend_from_slice(&drafts);
                    let (target_tokens, normed) = if round_draft_tokens == draft_tokens {
                        let (tokens, normed) = session.verify_tokens_replay(&mut sequence, &rows)?;
                        if std::env::var_os("ZLLM_GEMMA4_VREPLAY_TRACE").is_some() {
                            eprintln!("[vr-trace] replay targets={tokens:?} rows={rows:?} position={position}");
                        }
                        (tokens, normed)
                    } else {
                        let (tokens, normed_tensor) = session.verify_tokens(&mut sequence, &rows)?;
                        if std::env::var_os("ZLLM_GEMMA4_VREPLAY_TRACE").is_some() {
                            eprintln!("[vr-trace] plain targets={tokens:?} rows={rows:?} position={position}");
                        }
                        let normed = session.context().tensor_to_f32(&normed_tensor);
                        (tokens, normed)
                    };
                    verify_normed_rows = Some(normed);
                    let verification = crate::runtime::speculative::verify_samples(&target_tokens, &drafts, &eos_tokens).map_err(|error| format!("Gemma4 MTP verify: {error:?}"))?;
                    if std::env::var_os("ZLLM_GEMMA4_MTP_DUMP").is_some() {
                        eprintln!("[mtp-dump] round={} position={} anchor={} drafts={drafts:?} targets={target_tokens:?} acc={}", stats.rounds, position, pending, verification.accepted_drafts);
                    }
                    for (position, &draft) in drafts.iter().enumerate() {
                        if target_tokens.get(position) == Some(&draft) {
                            accepted_at_position[position] += 1;
                        }
                    }
                    stats.record(drafts.len(), &verification);
                    if profile_step {
                        eprintln!(
                            "[gemma4-mtp-step] round={} accepted_this={} wall={:.3}s draft={:.3}s verify={:.3}s | rate={:.1}%",
                            stats.rounds,
                            verification.accepted_drafts,
                            round_started.elapsed().as_secs_f64(),
                            draft_seconds,
                            round_started.elapsed().as_secs_f64() - draft_seconds,
                            stats.accepted as f64 * 100.0 / stats.proposed.max(1) as f64
                        );
                    }
                    // KV/tokens 修正:保留 rows[..retained],verify 输出末位是新 pending(未前向)。
                    let retained = verification.retained_rows;
                    sequence.tokens.truncate(position);
                    sequence.tokens.extend_from_slice(&rows[..retained]);
                    if let Some(state) = sequence.cache.hybrid_gqa_state_mut() {
                        state.truncate(verification.cache_end(position).map_err(|error| format!("Gemma4 MTP KV 提交边界: {error:?}"))?).map_err(|error| format!("Gemma4 MTP KV 回滚: {error:?}"))?;
                    }
                    accepted_prev = verification.accepted_drafts;
                    pending = verification.pending_token();
                    for token in verification.emitted_tokens() {
                        last_token = Some(*token);
                        if !emit_token!(*token)? {
                            return Ok(());
                        }
                        emitted_total += 1;
                        self.compute_steps.fetch_add(1);
                        if emitted_total >= max_tokens {
                            break;
                        }
                    }
                }
                // length 截断退出:pending 是最后采样未前向的 token,retain 语义同重放路径。
                last_token = Some(pending);
                eprintln!("[gemma4-mtp] K={} rounds={} accepted={} emitted={} rate={:.1}%", draft_tokens, stats.rounds, stats.accepted, stats.emitted, stats.accepted as f64 * 100.0 / stats.proposed.max(1) as f64);
                if stats.rounds > 0 {
                    for (position, count) in accepted_at_position.iter().enumerate() {
                        eprintln!("[gemma4-mtp-pos] 位置 {} 接受率 {:.1}% ({}/{})", position + 1, *count as f64 * 100.0 / stats.rounds as f64, count, stats.rounds);
                    }
                }
                return Ok(());
            }
            if async_decode {
                // 双缓冲命令重放 + 深度-2 流水:步首 gather 从对侧 readback 抠
                // embedding(queue FIFO 保证读到上一步产出的 token),步 n 在飞时即可
                // 提交步 n+1,GPU 不再等 CPU 的读回/编码;EOS/取消时推测轮白算,
                // drain 后连占位 token 带已前向的停止 token 一并回滚。
                if replay_decode {
                    let replay = session.replay();
                    let first = session.first_token(&sequence)?;
                    last_token = Some(first);
                    if !emit_token!(first)? {
                        return Ok(());
                    }
                    let context = session.context();
                    sequence.tokens.push(first);
                    // max_tokens == 1:首 token 即全部,不进入流水(避免多余推测轮)
                    if max_tokens > 1 {
                        let mut position = sequence.tokens.len() - 1;
                        // 首步 gather 没有上一步产出,预写对侧 readback 取得 prefill 首 token
                        replay.prime(1, first);
                        let mut handle = replay.step_async(context, &mut sequence.cache, 0, position).map_err(|error| format!("Gemma4 重放步: {error:?}"))?;
                        sequence.tokens.push(0);
                        let mut step = 1usize;
                        while step < max_tokens {
                            // 推测提交下一步:取消或末步不再推进(末 token 不前向,retain 语义不变)
                            let next = if step + 1 < max_tokens && !cancellation.load(Ordering::Acquire) {
                                sequence.tokens.push(0);
                                Some(replay.step_async(context, &mut sequence.cache, handle.parity ^ 1, position + 1).map_err(|error| format!("Gemma4 重放步: {error:?}")))
                            } else {
                                None
                            };
                            let step_started = std::time::Instant::now();
                            let token = replay.wait_token(&handle);
                            position += 1;
                            sequence.tokens[position] = token;
                            if profile_step {
                                eprintln!("[gemma4-node-profile] step={step} wall={:.3}s (replay-pipelined)", step_started.elapsed().as_secs_f64());
                            }
                            last_token = Some(token);
                            if !emit_token!(token)? {
                                // 停止 token 已被推测轮前向:drain 后连同占位 token 一起
                                // 从 tokens 与 KV 游标丢弃(cache 字节保留,续写原位覆写)。
                                if let Some(Ok(next_handle)) = next {
                                    replay.wait_token(&next_handle);
                                    session.discard_forwarded_token(&mut sequence)?;
                                }
                                session.discard_forwarded_token(&mut sequence)?;
                                break;
                            }
                            let Some(next_handle) = next else {
                                if cancellation.load(Ordering::Acquire) {
                                    output.cancel();
                                }
                                break;
                            };
                            handle = next_handle?;
                            step += 1;
                            // 无锁递增：避免每个 decode step 与 heartbeat / refresh 抢 Mutex。
                            self.compute_steps.fetch_add(1);
                        }
                    }
                    return Ok(());
                }
                // 设备闭环:argmax 写 per-position 读回区,下一 token 的 embedding 由
                // gather kernel 从常驻 Q4_K lm_head(tied)在 GPU 上抠行;CPU 只在
                // 输出步的 CB 句柄上等 token,不阻塞 GPU 提交。
                let last_row = crate::backend::Backend::select_row(session.context(), &sequence.hidden, sequence.hidden.rows - 1).map_err(|error| format!("Gemma4 输出步 select_row: {error:?}"))?;
                let mut pending = session.submit_output(&last_row, sequence.tokens.len())?;
                let mut step = 0usize;
                // 异步窗口内不能用 reset_gpu_stats(内含 synchronize,会破坏流水线),
                // 改为累计计数差分。
                let mut prev_gpu_seconds = if profile_step { session.context.gpu_stats().seconds } else { 0.0 };
                let mut prev_commands = if profile_step { session.context.gpu_stats().command_buffers } else { 0 };
                while step < max_tokens {
                    if cancellation.load(Ordering::Acquire) {
                        output.cancel();
                        break;
                    }
                    // 推测提交下一轮:EOS 时这轮白算,代价是一轮 GPU 时间;末轮不再推测。
                    let step_started = std::time::Instant::now();
                    let checkpoint_hidden = (step + 1 < max_tokens).then(|| sequence.hidden.clone());
                    let next = if step + 1 < max_tokens { Some(session.submit_step(&mut sequence, &mut pending)?) } else { None };
                    let submit_seconds = step_started.elapsed().as_secs_f64();
                    let token = session.wait_token(&mut sequence, &pending);
                    if profile_step {
                        // wait 后 pending 之前提交的 CB 已全部 complete 并计入 gpu_stats
                        let gpu = session.context.gpu_stats();
                        let (alloc_ns, commit_ns) = session.context.decode_cpu_breakdown();
                        eprintln!(
                            "[gemma4-node-profile] step={} wall={:.3}s submit={:.3}s wait={:.3}s gpu={:.3}s commands={} | cpu: alloc={:.1}ms commit={:.1}ms (async)",
                            step + 1,
                            step_started.elapsed().as_secs_f64(),
                            submit_seconds,
                            step_started.elapsed().as_secs_f64() - submit_seconds,
                            gpu.seconds - prev_gpu_seconds,
                            gpu.command_buffers - prev_commands,
                            alloc_ns as f64 / 1.0e6,
                            commit_ns as f64 / 1.0e6
                        );
                        prev_gpu_seconds = gpu.seconds;
                        prev_commands = gpu.command_buffers;
                        session.context.reset_decode_cpu_breakdown();
                    }
                    last_token = Some(token);
                    if !emit_token!(token)? {
                        // 停止 token 已被推测前向:从 tokens 与 KV cache 游标丢弃,
                        // 与同步路径(EOS/停止 token 不前向)的终点状态一致。
                        if next.is_some() {
                            session.discard_forwarded_token(&mut sequence)?;
                            sequence.hidden = checkpoint_hidden.expect("next 存在时必须保存投机前 hidden");
                        }
                        break;
                    }
                    step += 1;
                    // 无锁递增：避免每个 decode step 与 heartbeat / refresh 抢 Mutex。
                    self.compute_steps.fetch_add(1);
                    let Some(next) = next else { break };
                    pending = next;
                }
                return Ok(());
            }
            let mut token = session.first_token(&sequence)?;
            let mut step = 0usize;
            while step < max_tokens {
                if cancellation.load(Ordering::Acquire) {
                    output.cancel();
                    break;
                }
                if !emit_token!(token)? {
                    break;
                }
                step += 1;
                if step == max_tokens {
                    break;
                }
                if profile_step {
                    session.context.reset_gpu_stats();
                }
                let step_started = std::time::Instant::now();
                session.decode_token(&mut sequence, token)?;
                let decode_seconds = step_started.elapsed().as_secs_f64();
                token = session.step_token(&sequence)?;
                if profile_step {
                    let gpu = session.context.gpu_stats();
                    let operators = session.context.gpu_profile();
                    // wall 为 decode_token + step_token 的整迭代墙钟(decode 内 GPU 已延迟批次化,
                    // step_token 的 norm+lm_head+argmax 是逐算子同步段)。
                    eprintln!(
                        "[gemma4-node-profile] step={step} wall={:.3}s decode={:.3}s gpu={:.3}s commands={} operators={} submit_wait={:.3}s gaps={:.3}s tail={:.3}s",
                        step_started.elapsed().as_secs_f64(),
                        decode_seconds,
                        gpu.seconds,
                        gpu.command_buffers,
                        operators.len(),
                        gpu.submit_wait_seconds,
                        gpu.inter_command_gap_seconds,
                        gpu.completion_tail_seconds
                    );
                    if matches!(step, 1 | 16 | 32) {
                        for operator in operators.into_iter().take(14) {
                            eprintln!("  gemma4 node gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
                        }
                    }
                }
                // 无锁递增：避免每个 decode step 与 heartbeat / refresh 抢 Mutex。
                self.compute_steps.fetch_add(1);
            }
            last_token = Some(token);
            Ok(())
        })();
        if async_decode {
            session.end_async_decode();
        }
        loop_result?;
        output.finish(|chunk| on_token(None, chunk));
        let output_text = output.text().to_owned();
        let mut summary = output.summary(tokens.len());
        // 长度截断时最后采样的 token 尚未前向,作为 pending 保留给下一轮续写;
        // stop/eos 结束时该 token 不属于下一轮 prompt,直接丢弃。
        // 图文请求不 retain(resume 已跳过,保留下来也无人消费)。
        if summary.finish_reason != "cancelled" && stops.is_empty() && multimodal.is_none() {
            let pending = if summary.finish_reason == "length" { last_token.into_iter().collect() } else { Vec::new() };
            let cache_id = crate::runtime::session::terminal_cache_id(request, &output_text, &[])?;
            let info = CacheInfo {
                cache_id,
                model_key: "gemma4".to_owned(),
                cache_format: "gemma4-hybrid-terminal-v2".to_owned(),
                last_layer: self.session.layer_count().saturating_sub(1),
                prompt_tokens: sequence.token_count(),
                bytes: self.residency.session_resident_bytes() as u64,
                modified_unix: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
            };
            let state = metal_session::Gemma4TerminalState { sequence, pending, info: info.clone() };
            summary.cache = crate::runtime::session::retain_terminal_session(&mut self.terminal_states, state, info);
        }
        if profile_request {
            let finished_at = std::time::Instant::now();
            let total_seconds = finished_at.duration_since(request_started).as_secs_f64();
            let ttft_seconds = first_token_at.map_or(total_seconds, |instant| instant.duration_since(request_started).as_secs_f64());
            let decode_seconds = first_token_at.map_or(0.0, |instant| finished_at.duration_since(instant).as_secs_f64());
            let decode_tokens = summary.completion_tokens.saturating_sub(1);
            let decode_50_tps = first_token_at.zip(decode_50_at).map(|(first, end)| 50.0 / end.duration_since(first).as_secs_f64().max(f64::EPSILON));
            eprintln!(
                "[gemma4-request-profile] multimodal={} prompt_tokens={} completion_tokens={} prepare={:.3}s prefill={:.3}s prefill_tps={:.1} ttft={:.3}s decode={:.3}s decode_tps={:.1} tg50={} total={:.3}s",
                multimodal.is_some(),
                summary.prompt_tokens,
                summary.completion_tokens,
                prefill_started.duration_since(request_started).as_secs_f64(),
                prefill_seconds,
                summary.prompt_tokens as f64 / prefill_seconds.max(f64::EPSILON),
                ttft_seconds,
                decode_seconds,
                decode_tokens as f64 / decode_seconds.max(f64::EPSILON),
                decode_50_tps.map(|value| format!("{value:.1}")).unwrap_or_else(|| "n/a".to_owned()),
                total_seconds
            );
        }
        Ok(summary)
    }
}

#[cfg(all(test, target_os = "macos"))]
mod verify_rows_consistency_tests {
    use super::metal_session::Gemma4MetalSession;

    /// 多行 verify 的行一致性:同一 cache 状态下,2 行与 4 行 verify 的
    /// 非首行 argmax 必须一致(行 r 只依赖行 0..=r)。不一致 ⇒ 多行路径
    /// 的行间污染。`ZLLM_GEMMA4_GGUF=... cargo test --release --lib verify_rows_consistency -- --nocapture`
    #[test]
    fn verify_rows_consistency() {
        let Some(root) = std::env::var_os("ZLLM_GEMMA4_GGUF").map(std::path::PathBuf::from) else { return };
        let session = Gemma4MetalSession::load(&root, 2048, 2048, crate::weight::LmHeadQuantization::Native, None, 1, false).expect("session");
        let tokens = session.tokenize("The Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait.");
        let verify_input: Vec<u32> = vec![100, 200, 300, 400, 500, 600, 700, 800];
        for rows in [2usize, 3, 4, 5, 6, 8] {
            let mut seq = session.prefill(tokens.clone()).expect("prefill");
            let (targets, _) = session.verify_tokens_pub(&mut seq, &verify_input[..rows]).expect("verify");
            println!("[rows-cmp] {rows} 行={targets:?}");
        }
        // 参照:逐行单步 verify(每行独立 decode,无多行路径)
        {
            let mut seq = session.prefill(tokens.clone()).expect("prefill");
            let mut stepwise = Vec::new();
            for &token in verify_input.iter() {
                let first = crate::runtime::gemma4::gemma4_last_token_output(session.context(), session.config_pub(), session.output_head_pub(), &seq.hidden, 0).expect("output");
                stepwise.push(first.token_id);
                // 单步 decode 前向当前 token
                session.decode_token_pub(&mut seq, token).expect("decode");
            }
            println!("[rows-cmp] 逐行参照={stepwise:?}");
        }
    }
}
