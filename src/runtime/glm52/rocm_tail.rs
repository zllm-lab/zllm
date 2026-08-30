//! GLM-5.2 ROCm 尾端采样、MTP 与 terminal 驻留生命周期。

use super::*;

pub(super) struct TailSession {
    pub(super) states: Vec<Glm52StageState>,
    pub(super) hidden: Option<RocmTensor>,
    pub(super) completed_hidden: Option<(usize, RocmTensor)>,
    pub(super) prompt_hidden: Option<RocmTensor>,
    pub(super) prompt_position: Option<usize>,
    pub(super) position: usize,
    pub(super) resumed: bool,
    pub(super) started: Instant,
    pub(super) sampling: SamplingState,
    pub(super) completion_tokens: usize,
    pub(super) tail_sampling: bool,
    pub(super) pending_fences: VecDeque<crate::backend::TokenFence>,
    pub(super) verify_position: Option<usize>,
    pub(super) verify_hiddens: Vec<RocmTensor>,
    pub(super) mtp: Option<RocmMtpSession>,
}

pub(super) struct TailHeadWork {
    pub(super) request_id: RequestId,
    pub(super) cohort: u64,
    pub(super) cohort_size: usize,
    pub(super) position: usize,
    pub(super) hidden: RocmTensor,
}

pub(super) struct TailHeadCohort {
    pub(super) size: usize,
    pub(super) missing: usize,
    pub(super) works: Vec<TailHeadWork>,
}

pub(super) fn push_tail_head_work(cohorts: &mut HashMap<u64, TailHeadCohort>, requests: &mut HashMap<RequestId, (u64, usize)>, work: TailHeadWork) -> Result<Option<Vec<TailHeadWork>>, String> {
    if work.cohort == 0 || work.cohort_size == 1 {
        return Ok(Some(vec![work]));
    }
    let cohort = work.cohort;
    let cohort_size = work.cohort_size;
    let group = cohorts.entry(cohort).or_insert_with(|| TailHeadCohort { size: cohort_size, missing: 0, works: Vec::with_capacity(cohort_size) });
    if group.size != cohort_size {
        return Err(format!("tail output cohort={cohort} size 从 {} 变为 {cohort_size}", group.size));
    }
    group.works.push(work);
    let expected = group.size.checked_sub(group.missing).ok_or_else(|| format!("tail output cohort={cohort} 取消数超过大小"))?;
    if group.works.len() > expected {
        return Err(format!("tail output cohort={cohort} 数量={} 超过 expected={expected}", group.works.len()));
    }
    if group.works.len() != expected {
        return Ok(None);
    }
    let works = cohorts.remove(&cohort).expect("tail output cohort 刚检查完成").works;
    for work in &works {
        requests.remove(&work.request_id);
    }
    Ok(Some(works))
}

pub(super) fn cancel_tail_head_work(cohorts: &mut HashMap<u64, TailHeadCohort>, requests: &mut HashMap<RequestId, (u64, usize)>, request_id: RequestId) -> Result<Option<Vec<TailHeadWork>>, String> {
    let Some((cohort, size)) = requests.remove(&request_id) else { return Ok(None) };
    let group = cohorts.entry(cohort).or_insert_with(|| TailHeadCohort { size, missing: 0, works: Vec::with_capacity(size) });
    if group.size != size || group.missing >= group.size {
        return Err(format!("tail output cohort={cohort} 取消状态非法: size={}/{} missing={}", group.size, size, group.missing));
    }
    group.works.retain(|work| work.request_id != request_id);
    group.missing += 1;
    let expected = group.size - group.missing;
    if group.works.len() != expected {
        return Ok(None);
    }
    let works = cohorts.remove(&cohort).expect("tail output cohort 取消后刚检查完成").works;
    for work in &works {
        requests.remove(&work.request_id);
    }
    Ok(Some(works))
}

#[cfg(test)]
mod tail_head_cohort_tests {
    use super::*;

    fn work(request_id: RequestId) -> TailHeadWork {
        TailHeadWork {
            request_id,
            cohort: 7,
            cohort_size: 4,
            position: 11,
            hidden: RocmTensor { data: Vec::new(), rows: 1, cols: 1, dtype: crate::backend::rocm::RocmTensorDType::F32, layout: crate::backend::rocm::RocmTensorLayout::RowMajor, device: None },
        }
    }

    #[test]
    fn 取消一路后剩余cohort仍能收口() {
        let ids = (0..4).map(|index| RequestId::from_cache_id(&format!("tail-{index}"))).collect::<Vec<_>>();
        let mut cohorts = HashMap::new();
        let mut requests = ids.iter().copied().map(|request| (request, (7, 4))).collect::<HashMap<_, _>>();
        assert!(push_tail_head_work(&mut cohorts, &mut requests, work(ids[0])).unwrap().is_none());
        assert!(push_tail_head_work(&mut cohorts, &mut requests, work(ids[1])).unwrap().is_none());
        assert!(cancel_tail_head_work(&mut cohorts, &mut requests, ids[2]).unwrap().is_none());
        let ready = push_tail_head_work(&mut cohorts, &mut requests, work(ids[3])).unwrap().unwrap();
        assert_eq!(ready.iter().map(|work| work.request_id).collect::<Vec<_>>(), [ids[0], ids[1], ids[3]]);
        assert!(cohorts.is_empty());
        assert!(requests.is_empty());
    }
}

pub(super) fn tail_sample_batch(
    link: &mut StageTransport,
    active: &mut HashMap<RequestId, TailSession>,
    output_context: &RocmContext,
    mtp_runtime: &mut Option<RocmMtpRuntime>,
    weights: &Glm52Weights,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    rope: &RopeTable,
    output_head: &Glm52OutputHead<RocmWeight>,
    works: Vec<TailHeadWork>,
) -> Result<(), String> {
    if works.is_empty() {
        return Ok(());
    }
    let mut sampling = Vec::with_capacity(works.len());
    let mut fences = Vec::with_capacity(works.len());
    let mut hiddens = Vec::with_capacity(works.len());
    for work in &works {
        let session = active.get_mut(&work.request_id).ok_or_else(|| format!("tail sampling request={} 已消失", work.request_id))?;
        if !session.tail_sampling {
            return Err(format!("tail sampling request={} 未启用末段采样", work.request_id));
        }
        sampling.push(session.sampling.next());
        fences.push(session.pending_fences.pop_front().ok_or_else(|| format!("tail sampling request={} position={} 缺少围栏", work.request_id, work.position))?);
        hiddens.push(work.hidden.clone());
    }
    let hidden = if hiddens.len() == 1 { hiddens.pop().unwrap() } else { output_context.concat_token_rows(&hiddens.iter().collect::<Vec<_>>()).map_err(|error| format!("拼接 tail output head cohort: {error:?}"))? };
    let tokens = glm52_sampled_token_ids_fenced(output_context, cfg, output_head, &hidden, &sampling, &fences).map_err(|error| format!("tail output head cohort: {error:?}"))?;
    if tokens.len() != works.len() {
        return Err(format!("tail output head 返回 {} tokens，期望 {}", tokens.len(), works.len()));
    }
    let mut taken = Vec::with_capacity(works.len());
    for (work, token) in works.into_iter().zip(tokens) {
        let mut session = active.remove(&work.request_id).ok_or_else(|| format!("tail MTP draft request={} 已消失", work.request_id))?;
        session.completion_tokens += 1;
        taken.push((work, token, session));
    }
    let taken_len = taken.len();
    let initial_tokens = taken.iter().map(|(_, token, _)| *token).collect::<Vec<_>>();
    let mut draft_indices = Vec::new();
    let mut draft_batch = taken
        .iter_mut()
        .enumerate()
        .filter_map(|(index, (_, token, session))| {
            if cfg.eos_token_ids.contains(token) {
                return None;
            }
            let mtp = session.mtp.as_mut().filter(|mtp| mtp.active)?;
            let remaining = mtp.max_decode.saturating_sub(session.completion_tokens);
            let count = mtp.draft_tokens.min(remaining.saturating_sub(1));
            let hidden = mtp.pending_hidden.clone()?;
            draft_indices.push(index);
            Some(RocmMtpDraftBatch {
                session: mtp,
                token: *token,
                hidden,
                count,
                drafts: Vec::new(),
                // draft 只是候选；每个 target verify row 的真实围栏由 A 随 Verify 下发。
                fence: GenerationGuard::new(Glm52ToolFence::new(false), false, []),
            })
        })
        .collect::<Vec<_>>();
    let draft_result = if draft_batch.is_empty() {
        Ok(())
    } else if let Some(runtime) = mtp_runtime.as_mut() {
        mtp_draft_batch(runtime, &mut draft_batch, weights, output_head, cfg, mla, rope).map_err(|error| format!("B7 MTP draft cohort: {error:?}"))
    } else {
        Err("B7 active MTP session 缺少 runtime".to_owned())
    };
    let mut drafts = vec![Vec::new(); taken_len];
    if draft_result.is_ok() {
        for (index, item) in draft_indices.into_iter().zip(draft_batch.iter_mut()) {
            drafts[index] = std::mem::take(&mut item.drafts);
            item.session.verify_inputs = std::iter::once(initial_tokens[index]).chain(drafts[index].iter().copied()).collect();
        }
    }
    drop(draft_batch);
    let mut outbound = Vec::with_capacity(taken.len());
    for ((work, token, session), drafts) in taken.into_iter().zip(drafts) {
        active.insert(work.request_id, session);
        outbound.push((work, token, drafts));
    }
    draft_result?;
    for (work, token, drafts) in outbound {
        let eos = cfg.eos_token_ids.contains(&token);
        if drafts.is_empty() {
            link.send_sampled(work.request_id, work.cohort, work.cohort_size, work.position, token, eos)?;
        } else {
            link.send_speculative(work.request_id, &[token], 0, &drafts, false)?;
        }
    }
    Ok(())
}

/// tail stage 的 resident cache:已完成的 session 状态,等待复用或换出。
pub(super) struct TailResident {
    pub(super) states: Vec<Glm52StageState>,
    pub(super) hidden: RocmTensor,
    pub(super) position: usize,
    pub(super) mtp: Option<RocmMtpSession>,
}

pub(super) struct TailOpenRequest {
    pub(super) request_id: RequestId,
    pub(super) cache_request_id: Option<RequestId>,
    pub(super) cached_tokens: usize,
    pub(super) reserved_rows: usize,
    pub(super) cache_hit: bool,
    pub(super) sampling: SamplingConfig,
    pub(super) tail_sampling: bool,
}

pub(super) struct TailPendingOpen {
    request: TailOpenRequest,
    receiver: std::sync::mpsc::Receiver<Result<Option<Glm52CacheSnapshot>, String>>,
}

/// resident hit 立即完成；SSD hit 只并行读取 host snapshot，GPU 恢复仍留在 stage 线程。
#[allow(clippy::too_many_arguments)]
pub(super) fn begin_tail_stage_open(
    link: &mut StageTransport,
    active: &mut HashMap<RequestId, TailSession>,
    resident: &mut HashMap<RequestId, TailResident>,
    pending: &mut Vec<TailPendingOpen>,
    swap: &Arc<Glm52SwapStore>,
    output_context: &RocmContext,
    templates: &[Glm52StageState],
    cfg: &Glm52Config,
    max_seq_len: usize,
    request: TailOpenRequest,
) -> Result<(), String> {
    if active.contains_key(&request.request_id) || pending.iter().any(|item| item.request.request_id == request.request_id) {
        return Err(format!("后继重复 Open request={}", request.request_id));
    }
    let needs_ssd = request.cache_hit && request.cache_request_id.is_some_and(|cache_id| !resident.contains_key(&cache_id));
    if !needs_ssd {
        return tail_stage_open(link, active, resident, swap, output_context, templates, cfg, max_seq_len, request, None);
    }
    let cache_id = request.cache_request_id.expect("needs_ssd 已检查 cache id").to_string();
    let request_id = request.request_id;
    let swap = Arc::clone(swap);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name(format!("glm52-tail-{}", request_id.to_string().chars().take(12).collect::<String>()))
        .spawn(move || {
            let started = Instant::now();
            let result = swap.get(&cache_id);
            eprintln!("[glm52-swap-prefetch] side=tail request_id={request_id} cache_id={cache_id} hit={} read_ms={:.3}", result.as_ref().ok().is_some_and(Option::is_some), started.elapsed().as_secs_f64() * 1000.0);
            let _ = sender.send(result);
        })
        .map_err(|error| format!("启动 tail SSD prefetch: {error}"))?;
    pending.push(TailPendingOpen { request, receiver });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn poll_tail_stage_opens(
    link: &mut StageTransport,
    active: &mut HashMap<RequestId, TailSession>,
    resident: &mut HashMap<RequestId, TailResident>,
    pending: &mut Vec<TailPendingOpen>,
    swap: &Glm52SwapStore,
    output_context: &RocmContext,
    templates: &[Glm52StageState],
    cfg: &Glm52Config,
    max_seq_len: usize,
) -> Result<bool, String> {
    let mut progressed = false;
    let mut index = 0;
    while index < pending.len() {
        let result = match pending[index].receiver.try_recv() {
            Ok(result) => result,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                index += 1;
                continue;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("tail SSD prefetch 线程提前退出".to_owned()),
        };
        let item = pending.remove(index);
        tail_stage_open(link, active, resident, swap, output_context, templates, cfg, max_seq_len, item.request, Some(result))?;
        progressed = true;
    }
    Ok(progressed)
}

/// 处理 Open:按 cache_request_id 从 resident/SSD 恢复或新建 session。
/// 所有 cache 拒绝路径都 send_ready(0) 告知前机未命中,由前机决定是否重发。
/// continuous 与非 continuous 入口共用;continuous 入口此前漏了 "缺少 cache_request_id"
/// 的拒绝日志,收编后统一补上(纯诊断输出,不改变协议行为)。
#[allow(clippy::too_many_arguments)]
pub(super) fn tail_stage_open(
    link: &mut StageTransport,
    active: &mut HashMap<RequestId, TailSession>,
    resident: &mut HashMap<RequestId, TailResident>,
    swap: &Glm52SwapStore,
    output_context: &RocmContext,
    templates: &[Glm52StageState],
    cfg: &Glm52Config,
    max_seq_len: usize,
    request: TailOpenRequest,
    prefetched: Option<Result<Option<Glm52CacheSnapshot>, String>>,
) -> Result<(), String> {
    let TailOpenRequest { request_id, cache_request_id, cached_tokens, reserved_rows, cache_hit, sampling, tail_sampling } = request;
    if active.contains_key(&request_id) {
        return Err(format!("后继重复 Open active request={request_id}"));
    }
    let mut states = templates.iter().map(|state| state.fresh_session(cfg, max_seq_len)).collect::<Result<Vec<_>, _>>().map_err(|error| format!("创建 tail stage session: {error:?}"))?;
    let mut hidden = None;
    let mut mtp = None;
    if cache_hit {
        let reserved_rows = if reserved_rows == 0 { cached_tokens } else { reserved_rows };
        if reserved_rows < cached_tokens || reserved_rows > max_seq_len {
            return Err(format!("tail cache reserve rows={reserved_rows} 非法: cached={cached_tokens} max={max_seq_len}"));
        }
        let Some(cache_request_id) = cache_request_id else {
            eprintln!("[stage-cache-reject] request_id={request_id} cache_hit=true 但缺少 cache_request_id");
            link.send_ready(request_id, 0)?;
            return Ok(());
        };
        if let Some(saved) = resident.remove(&cache_request_id) {
            if saved.position != cached_tokens {
                let position = saved.position;
                resident.insert(cache_request_id, saved);
                eprintln!("[stage-cache-reject] request_id={request_id} resident tokens={position}，前机命令={cached_tokens}");
                link.send_ready(request_id, 0)?;
                return Ok(());
            }
            states = saved.states;
            hidden = Some(saved.hidden);
            mtp = saved.mtp;
            if let Some(mtp) = mtp.as_mut() {
                mtp.deactivate();
            }
        } else {
            let cache_id = cache_request_id.to_string();
            let snapshot = match prefetched {
                Some(result) => result?,
                None => swap.get(&cache_id)?,
            };
            let Some(snapshot) = snapshot else {
                eprintln!("[stage-cache-reject] request_id={request_id} SSD cache 未命中 cache_id={cache_id}");
                link.send_ready(request_id, 0)?;
                return Ok(());
            };
            if snapshot.token_count != cached_tokens || snapshot.last_hidden.len() != cfg.hidden_size {
                eprintln!("[stage-cache-reject] request_id={request_id} SSD cache 元数据错误: tokens={}/{} hidden={}/{}", snapshot.token_count, cached_tokens, snapshot.last_hidden.len(), cfg.hidden_size,);
                link.send_ready(request_id, 0)?;
                return Ok(());
            }
            upload_glm52_session(&mut states, &snapshot.stages, cfg, max_seq_len, reserved_rows)?;
            hidden = Some(output_context.tensor_from_bf16_bits(snapshot.last_hidden, 1, cfg.hidden_size).map_err(|error| format!("恢复 tail terminal hidden: {error:?}"))?);
            mtp = snapshot.mtp.map(|mtp| RocmMtpSession::restore(mtp, output_context, cfg, max_seq_len, reserved_rows)).transpose()?;
            if let Some(mtp) = &mtp {
                eprintln!("[glm52-mtp-swap-in] cache_id={request_id} position={}", mtp.position);
            }
            eprintln!("[stage-swap-in] cache_id={request_id} tokens={cached_tokens}");
        }
    } else {
        swap.delete(&request_id.to_string())?;
    }
    let sampling = SamplingState::new(sampling)?;
    active.insert(
        request_id,
        TailSession {
            states,
            hidden,
            completed_hidden: None,
            prompt_hidden: None,
            prompt_position: None,
            position: cached_tokens,
            resumed: cache_hit,
            started: Instant::now(),
            sampling,
            completion_tokens: 0,
            tail_sampling,
            pending_fences: VecDeque::new(),
            verify_position: None,
            verify_hiddens: Vec::new(),
            mtp,
        },
    );
    link.send_ready(request_id, cached_tokens)
}

/// 把已结束的 session 截断到 tokens 后以 next_request_id 名义存入 resident 并 ACK。
/// `completed_hidden_fallback`:continuous 路径的输出走 pipeline,tokens 可能落在最近一次
/// 批量输出张量中间,允许从 completed_hidden 切片恢复末行;非 continuous 路径逐条驱动,
/// 只接受 prompt 边界,传 false 保持原有严格语义。
pub(super) fn tail_stage_cache_commit(
    link: &mut StageTransport,
    resident: &mut HashMap<RequestId, TailResident>,
    output_context: &RocmContext,
    mut tail: TailSession,
    request_id: RequestId,
    next_request_id: RequestId,
    tokens: usize,
    completed_hidden_fallback: bool,
) -> Result<(), String> {
    if tokens != tail.position {
        if !completed_hidden_fallback && tail.prompt_position != Some(tokens) {
            return Err(format!("后继 Cache tokens={tokens}，当前 position={} prompt={:?}", tail.position, tail.prompt_position));
        }
        for state in &mut tail.states {
            state.cache.truncate_rows(tokens).map_err(|error| format!("截断 tail cache 到 {tokens}: {error:?}"))?;
            state.dsa.truncate_rows(tokens).map_err(|error| format!("截断 tail DSA 到 {tokens}: {error:?}"))?;
        }
        tail.hidden = if tail.prompt_position == Some(tokens) {
            tail.prompt_hidden.take()
        } else if completed_hidden_fallback
            && let Some((start, hidden)) = tail.completed_hidden.take()
            && tokens > start
            && tokens <= start + hidden.rows
        {
            Some(output_context.slice_token_rows(&hidden, tokens - start - 1, 1).map_err(|error| format!("切 tail terminal hidden: {error:?}"))?)
        } else {
            return Err(format!("后继 Cache tokens={tokens}，当前 position={} prompt={:?}", tail.position, tail.prompt_position));
        };
        tail.position = tokens;
    }
    let hidden = tail.hidden.ok_or_else(|| "后继 Cache 缺少 terminal hidden".to_owned())?;
    let mtp = match tail.mtp.take() {
        Some(mut mtp) if mtp.active => {
            mtp.truncate_to_target(tokens).map_err(|error| format!("截断 tail MTP 到 {tokens}: {error:?}"))?;
            mtp.pending_hidden = Some(hidden.clone());
            mtp.deactivate();
            Some(mtp)
        }
        _ => None,
    };
    resident.insert(next_request_id, TailResident { states: tail.states, hidden, position: tokens, mtp });
    link.send_ready(request_id, tokens)?;
    Ok(())
}

/// 把 resident session 换出到本机 SSD 并 ACK。continuous 入口此前没有换出日志,收编后统一。
pub(super) fn tail_stage_swap_out(link: &mut StageTransport, resident: &mut HashMap<RequestId, TailResident>, swap: &Glm52SwapStore, output_context: &RocmContext, request_id: RequestId) -> Result<(), String> {
    let saved = resident.remove(&request_id).ok_or_else(|| format!("后继 SwapOut cache={request_id} 不在 resident"))?;
    let snapshot = Glm52CacheSnapshot {
        cache_id: request_id.to_string(),
        cache_namespace: None,
        tokens: Vec::new(),
        token_count: saved.position,
        pending_tokens: None,
        last_hidden: output_context.tensor_to_bf16_bits(&saved.hidden).map_err(|error| format!("下载 tail terminal hidden: {error:?}"))?,
        stages: download_glm52_session(&saved.states)?,
        mtp: saved.mtp.as_ref().map(|mtp| mtp.snapshot(output_context)).transpose()?,
        dspark_aux: None,
        dspark_target: None,
    };
    swap.put(&snapshot)?;
    link.send_ready(request_id, saved.position)?;
    eprintln!("[stage-swap-out] cache_id={request_id} tokens={}", saved.position);
    Ok(())
}

/// 把 resident session 镜像到 SSD(不删除 resident、不 ACK)。
/// continuous 入口此前只处理 resident 命中分支,收编后统一三分支,差异仅为诊断日志。
pub(super) fn tail_stage_persist(resident: &mut HashMap<RequestId, TailResident>, swap: &Glm52SwapStore, output_context: &RocmContext, request_id: RequestId) -> Result<(), String> {
    if let Some(saved) = resident.get(&request_id) {
        let snapshot = Glm52CacheSnapshot {
            cache_id: request_id.to_string(),
            cache_namespace: None,
            tokens: Vec::new(),
            token_count: saved.position,
            pending_tokens: None,
            last_hidden: output_context.tensor_to_bf16_bits(&saved.hidden).map_err(|error| format!("下载 tail terminal hidden: {error:?}"))?,
            stages: download_glm52_session(&saved.states)?,
            mtp: saved.mtp.as_ref().map(|mtp| mtp.snapshot(output_context)).transpose()?,
            dspark_aux: None,
            dspark_target: None,
        };
        swap.put(&snapshot)?;
        eprintln!("[stage-persist] cache_id={request_id} tokens={}", saved.position);
    } else if swap.get(&request_id.to_string())?.is_some() {
        eprintln!("[stage-persist] cache_id={request_id} 已在 SSD");
    } else {
        eprintln!("[stage-persist] cache_id={request_id} 不在 resident 或 SSD");
    }
    Ok(())
}

/// 退出时镜像所有 resident session。链头先发 Shutdown 再写本机，两机 SSD I/O 因而可以并行。
pub(super) fn tail_stage_persist_all(resident: &mut HashMap<RequestId, TailResident>, swap: &Glm52SwapStore, output_context: &RocmContext) -> Result<usize, String> {
    let request_ids = resident.keys().copied().collect::<Vec<_>>();
    for request_id in &request_ids {
        tail_stage_persist(resident, swap, output_context, *request_id)?;
    }
    Ok(request_ids.len())
}

/// 从 active/resident/SSD 删除 session,返回删除前 position(continuous 需要随 ACK 上报)。
/// 两张表都删:正常情况 request_id 只在一处,但取消时序下可能残留,与原有调用点中最严格的
/// 语义保持一致。
pub(super) fn tail_stage_delete(active: &mut HashMap<RequestId, TailSession>, resident: &mut HashMap<RequestId, TailResident>, swap: &Glm52SwapStore, request_id: RequestId) -> Result<usize, String> {
    let active_position = active.remove(&request_id).map(|tail| tail.position);
    let resident_position = resident.remove(&request_id).map(|saved| saved.position);
    swap.delete(&request_id.to_string())?;
    Ok(active_position.or(resident_position).unwrap_or(0))
}

/// 处理 MtpContext:为 session 建立/复用 MTP resident state。
/// 仅非 continuous 入口使用;continuous(A0-output)模式在 B 端不跑 MTP,直接拒绝。
pub(super) fn tail_stage_mtp_context(
    session: &mut TailSession,
    mtp_runtime_enabled: bool,
    mtp_draft_tokens: usize,
    cfg: &Glm52Config,
    max_seq_len: usize,
    request_id: RequestId,
    prompt_tokens: Vec<u32>,
    max_decode: usize,
    draft_tokens: usize,
) -> Result<(), String> {
    if !mtp_runtime_enabled {
        return Err("前机请求 MTP，但 tail 未启用 model.execution.mtp".to_owned());
    }
    if session.mtp.is_none() && !session.resumed {
        session.mtp = Some(RocmMtpSession::fresh(cfg, max_seq_len)?);
    }
    if let Some(mtp) = session.mtp.as_mut() {
        mtp.begin_request(session.position, prompt_tokens, max_decode, draft_tokens.min(mtp_draft_tokens))?;
    } else {
        eprintln!("[glm52-mtp-fallback] request_id={request_id} SSD cache 没有 MTP resident state，本轮退回普通 decode");
    }
    Ok(())
}

pub(crate) struct RocmMtpCatchUp<'a> {
    pub(crate) session: &'a mut RocmMtpSession,
    pub(crate) target_position: usize,
    pub(crate) target_inputs: Vec<u32>,
    pub(crate) target_hidden: RocmTensor,
}

// ROCm MLA 单行输出 F32、多行输出 BF16；同一次 segmented 调用不能混用两种行宽。
pub(super) fn mtp_catch_up_groups(metadata: &[Option<(usize, usize)>]) -> [Vec<usize>; 2] {
    let mut groups = [Vec::new(), Vec::new()];
    for (index, item) in metadata.iter().enumerate() {
        if let Some((_, rows)) = item {
            groups[usize::from(*rows > 1)].push(index);
        }
    }
    groups
}

#[cfg(test)]
mod mtp_catch_up_tests {
    use super::mtp_catch_up_groups;

    #[test]
    fn separates_decode_and_prefill_output_widths() {
        let groups = mtp_catch_up_groups(&[Some((10, 1)), Some((20, 3)), None, Some((30, 2)), Some((40, 1))]);
        assert_eq!(groups[0], [0, 4]);
        assert_eq!(groups[1], [1, 3]);
    }
}

/// 同一 ready cohort 的 MTP catch-up 合并为一次 L78 segmented prefill。
#[allow(clippy::too_many_arguments)]
pub(crate) fn mtp_catch_up_batch(runtime: &mut RocmMtpRuntime, items: &mut [RocmMtpCatchUp<'_>], model_weights: &Glm52Weights, cfg: &Glm52Config, mla: &MlaSpec, rope: &RopeTable) -> Result<(), crate::backend::BackendError> {
    let mut metadata = vec![None; items.len()];
    let mut embeddings = std::iter::repeat_with(|| None).take(items.len()).collect::<Vec<Option<RocmTensor>>>();
    let mut shifted = std::iter::repeat_with(|| None).take(items.len()).collect::<Vec<Option<RocmTensor>>>();
    for (index, item) in items.iter_mut().enumerate() {
        let rows = item.target_hidden.rows;
        if rows == 0 || rows != item.target_inputs.len() {
            return Err(crate::backend::BackendError::Compute { msg: format!("MTP batch catch-up shape 非法: tokens={} hidden=[{},{}]", item.target_inputs.len(), rows, item.target_hidden.cols) });
        }
        let (tokens, hidden, position) = if item.target_position == 0 {
            if item.session.position != 0 || item.session.pending_hidden.is_some() {
                return Err(crate::backend::BackendError::Compute { msg: "MTP batch fresh catch-up 状态非空".to_owned() });
            }
            if rows == 1 {
                item.session.pending_hidden = Some(runtime.backend.slice_token_rows(&item.target_hidden, 0, 1)?);
                continue;
            }
            (item.target_inputs[1..].to_vec(), runtime.backend.slice_token_rows(&item.target_hidden, 0, rows - 1)?, 0)
        } else {
            let expected = item.target_position - 1;
            if item.session.position != expected {
                return Err(crate::backend::BackendError::Compute { msg: format!("MTP batch catch-up position={}，期望 {expected}", item.session.position) });
            }
            let previous = item.session.pending_hidden.as_ref().ok_or_else(|| crate::backend::BackendError::Compute { msg: "MTP batch catch-up 缺少跨 batch hidden".to_owned() })?;
            let hidden = if rows == 1 {
                previous.clone()
            } else {
                let prefix = runtime.backend.slice_token_rows(&item.target_hidden, 0, rows - 1)?;
                runtime.backend.concat_token_rows(&[previous, &prefix])?
            };
            (item.target_inputs.clone(), hidden, expected)
        };
        let embedding = runtime.backend.tensor_from_f32(model_weights.embedding_rows(&tokens).map_err(crate::backend::compute_error)?, tokens.len(), cfg.hidden_size).map_err(crate::backend::compute_error)?;
        metadata[index] = Some((position, tokens.len()));
        embeddings[index] = Some(embedding);
        shifted[index] = Some(hidden);
    }
    if metadata.iter().any(Option::is_some) {
        runtime.backend.activate().map_err(crate::backend::compute_error)?;
        for group in mtp_catch_up_groups(&metadata) {
            if group.is_empty() {
                continue;
            }
            let embedding_refs = group.iter().map(|&index| embeddings[index].as_ref().expect("MTP catch-up embedding 已准备")).collect::<Vec<_>>();
            let hidden_refs = group.iter().map(|&index| shifted[index].as_ref().expect("MTP catch-up hidden 已准备")).collect::<Vec<_>>();
            let embeddings = runtime.backend.concat_token_rows(&embedding_refs)?;
            let shifted = runtime.backend.concat_token_rows(&hidden_refs)?;
            let mut selected = group.into_iter().peekable();
            let mut segments = items
                .iter_mut()
                .enumerate()
                .filter_map(|(index, item)| {
                    if selected.peek().copied() != Some(index) {
                        return None;
                    }
                    selected.next();
                    metadata[index].map(|(position, rows)| Glm52PrefillSegment { position, rows, cache: &mut item.session.cache, dsa: &mut item.session.dsa })
                })
                .collect::<Vec<_>>();
            glm52_mtp_cache_segmented(&runtime.backend, cfg, mla, &runtime.weights, &embeddings, &shifted, rope, &mut segments)?;
        }
    }
    for (index, item) in items.iter_mut().enumerate() {
        if let Some((_, rows)) = metadata[index] {
            item.session.position += rows;
        }
        item.session.pending_hidden = Some(runtime.backend.slice_token_rows(&item.target_hidden, item.target_hidden.rows - 1, 1)?);
    }
    Ok(())
}

pub(crate) struct RocmMtpDraftBatch<'a> {
    pub(crate) session: &'a mut RocmMtpSession,
    pub(crate) token: u32,
    pub(crate) hidden: RocmTensor,
    pub(crate) count: usize,
    pub(crate) drafts: Vec<u32>,
    pub(crate) fence: GenerationGuard<Glm52ToolFence>,
}

/// 按 draft depth 编排 cohort：每一深度只跑仍活跃的 session，L78 与 LM head
/// 都各提交一次 batch，不再逐 session 完整走三层深度。
#[allow(clippy::too_many_arguments)]
pub(crate) fn mtp_draft_batch(
    runtime: &mut RocmMtpRuntime,
    items: &mut [RocmMtpDraftBatch<'_>],
    model_weights: &Glm52Weights,
    output_head: &Glm52OutputHead<RocmWeight>,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    rope: &RopeTable,
) -> Result<(), crate::backend::BackendError> {
    let max_depth = items.iter().map(|item| item.count).max().unwrap_or(0);
    let profile = crate::kernel::rocm::hip::options().kernel_profile;
    runtime.backend.activate().map_err(crate::backend::compute_error)?;
    for depth in 0..max_depth {
        let prep_started = profile.then(Instant::now);
        let mut metadata = vec![false; items.len()];
        let mut embeddings = Vec::new();
        let mut hiddens = Vec::new();
        for (index, item) in items.iter().enumerate() {
            if depth >= item.count || item.drafts.last().is_some_and(|token| cfg.eos_token_ids.contains(token)) {
                continue;
            }
            embeddings.push(runtime.backend.tensor_from_f32(model_weights.embedding_rows(&[item.token]).map_err(crate::backend::compute_error)?, 1, cfg.hidden_size).map_err(crate::backend::compute_error)?);
            hiddens.push(item.hidden.clone());
            metadata[index] = true;
        }
        if embeddings.is_empty() {
            break;
        }
        let embedding_refs = embeddings.iter().collect::<Vec<_>>();
        let hidden_refs = hiddens.iter().collect::<Vec<_>>();
        let embeddings = runtime.backend.concat_token_rows(&embedding_refs)?;
        let hiddens = runtime.backend.concat_token_rows(&hidden_refs)?;
        let mut segments = items
            .iter_mut()
            .enumerate()
            .filter(|(index, _)| metadata[*index])
            .map(|(_, item)| Glm52PrefillSegment { position: item.session.position, rows: 1, cache: &mut item.session.cache, dsa: &mut item.session.dsa })
            .collect::<Vec<_>>();
        if profile {
            runtime.backend.synchronize()?;
        }
        let prep_micros = prep_started.map_or(0, |started| started.elapsed().as_micros());
        let layer_started = profile.then(Instant::now);
        let hidden = glm52_mtp_prefill_segmented(&runtime.backend, cfg, mla, &runtime.weights, &mut runtime.experts, &embeddings, &hiddens, rope, &mut segments)?;
        if profile {
            runtime.backend.synchronize()?;
        }
        let layer_micros = layer_started.map_or(0, |started| started.elapsed().as_micros());
        let head_started = profile.then(Instant::now);
        let fences = items.iter().enumerate().filter(|(index, _)| metadata[*index]).map(|(_, item)| item.fence.fence()).collect::<Vec<_>>();
        let tokens = if let Some(draft_head) = &runtime.draft_head {
            let tokens = normalized_draft_token_ids_fenced(&runtime.backend, draft_head, &hidden, &fences)?;
            // profiling 时用同一份 hidden 跑 full head。若 full argmax 已在子集内，
            // reduced head 必须逐 token 一致，否则就是权重行抽取、argmax 或映射错误。
            if profile {
                let full_tokens = glm52_mtp_token_ids_fenced(&runtime.backend, output_head, &hidden, &fences)?;
                for (row, (&draft, &full)) in tokens.iter().zip(&full_tokens).enumerate() {
                    eprintln!("[glm52-fr-spec-oracle] depth={} row={} draft={} full={} full_in_draft={} match={}", depth + 1, row, draft, full, draft_head.contains_token(full), draft == full,);
                }
            }
            tokens
        } else {
            glm52_mtp_token_ids_fenced(&runtime.backend, output_head, &hidden, &fences)?
        };
        if profile {
            runtime.backend.synchronize()?;
        }
        let head_micros = head_started.map_or(0, |started| started.elapsed().as_micros());
        let state_started = profile.then(Instant::now);
        let mut offset = 0usize;
        for (index, item) in items.iter_mut().enumerate() {
            if !metadata[index] {
                continue;
            }
            item.hidden = runtime.backend.slice_token_rows(&hidden, offset, 1)?;
            item.token = tokens[offset];
            item.drafts.push(tokens[offset]);
            item.fence.advance(tokens[offset]);
            item.session.position += 1;
            offset += 1;
        }
        if profile {
            eprintln!(
                "[glm52-mtp-draft-depth] depth={} rows={} prep_ms={:.3} layer_ms={:.3} head_ms={:.3} state_ms={:.3}",
                depth + 1,
                tokens.len(),
                prep_micros as f64 / 1000.0,
                layer_micros as f64 / 1000.0,
                head_micros as f64 / 1000.0,
                state_started.map_or(0, |started| started.elapsed().as_micros()) as f64 / 1000.0,
            );
        }
    }
    Ok(())
}

pub(super) fn prepare_tail_output_runtime(
    output_context: &RocmContext,
    weights: &Glm52Weights,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    mtp_enabled: bool,
    mtp_draft_tokens: usize,
    mtp_draft_vocabulary: Option<&Path>,
    lm_head_quantization: LmHeadQuantization,
) -> Result<(Glm52OutputHead<RocmWeight>, Option<RocmMtpRuntime>), String> {
    let final_norm = weights.final_norm().map_err(|error| format!("加载 tail final norm: {error:?}"))?;
    let lm_head = weights.lm_head_bf16_bytes().map_err(|error| format!("加载 tail LM head: {error:?}"))?;
    let output_head = prepare_glm52_output_head_quantized(output_context, cfg, &final_norm, LinearWeight::Bf16Bytes(&lm_head), lm_head_quantization).map_err(|error| format!("准备 tail ROCm GLM output head: {error:?}"))?;
    let draft_head = if mtp_enabled {
        mtp_draft_vocabulary
            .map(|path| {
                let token_ids = load_draft_vocabulary(path, "glm52", cfg.vocab_size, &cfg.eos_token_ids)?;
                let head = prepare_draft_head(output_context, LinearWeight::Bf16Bytes(&lm_head), cfg.vocab_size, cfg.hidden_size, token_ids).map_err(|error| format!("准备 GLM FR-Spec draft head: {error:?}"))?;
                eprintln!("[glm52-mtp-draft-head] vocabulary={} source={} format=bf16", head.token_ids().len(), path.display());
                Ok::<_, String>(head)
            })
            .transpose()?
    } else {
        None
    };
    let mtp_runtime = if mtp_enabled {
        if weights.source_is_ct() {
            let started = Instant::now();
            output_context.activate()?;
            let source = weights.ct_source()?;
            let layer = source.load_mtp_layer(cfg.layer_count).map_err(|error| format!("加载 tail MTP L{}: {error:?}", cfg.layer_count))?;
            let mtp_weights = prepare_glm52_mtp_ct(output_context, cfg, mla, &layer).map_err(|error| format!("准备 tail ROCm MTP: {error:?}"))?;
            let mut experts = RocmPrefillExperts::ct(source);
            experts.preload_layer(output_context, cfg.layer_count, cfg.expert_count).map_err(|error| format!("常驻 ROCm MTP experts: {error:?}"))?;
            eprintln!("[glm52-mtp-resident] device={} layer={} drafts={} wall={:.3}s", output_context.device_id(), cfg.layer_count, mtp_draft_tokens, started.elapsed().as_secs_f64());
            Some(RocmMtpRuntime { backend: *output_context, weights: mtp_weights, experts, draft_head })
        } else if weights.source_is_gguf() {
            let started = Instant::now();
            output_context.activate()?;
            let layer = weights.load_mtp_layer_gguf().map_err(|error| format!("加载 tail MTP L{}: {error}", cfg.layer_count))?;
            let mtp_weights = prepare_glm52_mtp_gguf(output_context, cfg, mla, &layer).map_err(|error| format!("准备 tail ROCm MTP: {error:?}"))?;
            let mut experts = RocmPrefillExperts::gguf(weights.gguf_source()?);
            experts.preload_layer(output_context, cfg.layer_count, cfg.expert_count).map_err(|error| format!("常驻 ROCm MTP experts: {error:?}"))?;
            eprintln!("[glm52-mtp-resident] device={} layer={} drafts={} wall={:.3}s", output_context.device_id(), cfg.layer_count, mtp_draft_tokens, started.elapsed().as_secs_f64());
            Some(RocmMtpRuntime { backend: *output_context, weights: mtp_weights, experts, draft_head })
        } else {
            return Err("GLM-5.2 distributed MTP 当前只支持 compressed-tensors/GGUF 权重".to_owned());
        }
    } else {
        None
    };
    Ok((output_head, mtp_runtime))
}
