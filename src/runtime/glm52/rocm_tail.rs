//! GLM-5.2 ROCm head 侧输出运行时、MTP 与 terminal 驻留生命周期。

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
}

/// tail stage 的 resident cache:已完成的 session 状态,等待复用或换出。
/// 输出与 MTP 驻留 head 首卡,tail 的 resident 只含 KV/DSA 与 terminal hidden。
pub(super) struct TailResident {
    pub(super) states: Vec<Glm52StageState>,
    pub(super) hidden: RocmTensor,
    pub(super) position: usize,
}

pub(super) struct TailOpenRequest {
    pub(super) request_id: RequestId,
    pub(super) cache_request_id: Option<RequestId>,
    pub(super) cached_tokens: usize,
    pub(super) reserved_rows: usize,
    pub(super) cache_hit: bool,
    pub(super) sampling: SamplingConfig,
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
    let TailOpenRequest { request_id, cache_request_id, cached_tokens, reserved_rows, cache_hit, sampling: sampling_config } = request;
    if active.contains_key(&request_id) {
        return Err(format!("后继重复 Open active request={request_id}"));
    }
    let mut states = templates.iter().map(|state| state.fresh_session(cfg, max_seq_len)).collect::<Result<Vec<_>, _>>().map_err(|error| format!("创建 tail stage session: {error:?}"))?;
    let mut hidden = None;
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
            eprintln!("[stage-swap-in] cache_id={request_id} tokens={cached_tokens}");
        }
    } else {
        swap.delete(&request_id.to_string())?;
    }
    // 采样参数在 head 侧消费;这里只做一次合法性校验,坏参数在 Open 期暴露。
    SamplingState::new(sampling_config)?;
    active.insert(request_id, TailSession { states, hidden, completed_hidden: None, prompt_hidden: None, prompt_position: None, position: cached_tokens, resumed: cache_hit, started: Instant::now() });
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
    resident.insert(next_request_id, TailResident { states: tail.states, hidden, position: tokens });
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
        mtp: None,
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
            mtp: None,
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

/// MtpContext 已随 tail 采样一并移除:MTP resident state 由 head 首卡持有,
/// tail 引擎层收到该帧直接拒绝,这里不再提供处理路径。

/// 首卡常驻 BF16 embedding 表的设备侧行 gather;ids 上传 + 一次 kernel,
/// 输出 F32 [rows, hidden],与 CPU `embedding_rows` 路径语义一致。
pub(in crate::runtime::glm52) fn gather_embedding_rows(context: &RocmContext, table: &std::sync::Arc<ops::hip::DeviceBuffer>, ids: &[u32], hidden: usize) -> Result<RocmTensor, String> {
    if ids.is_empty() {
        return Err("embedding gather ids 为空".to_owned());
    }
    let device_id = context.device_id();
    let ids_bytes = ids.len().checked_mul(std::mem::size_of::<u32>()).ok_or("embedding ids 字节溢出")?;
    let ids_buffer = ops::hip::DeviceBuffer::upload(device_id, unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), ids_bytes) }).map_err(|error| format!("上传 embedding ids: {error}"))?;
    let output = ops::hip::try_gather_bf16_rows_f32(device_id, table, &ids_buffer, ids.len(), hidden).map_err(|error| format!("embedding gather kernel: {error}"))?;
    Ok(RocmTensor { data: Vec::new(), rows: ids.len(), cols: hidden, dtype: crate::backend::rocm::RocmTensorDType::F32, layout: crate::backend::rocm::RocmTensorLayout::RowMajor, device: Some(std::sync::Arc::new(output)) })
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
        let embedding = match runtime.embedding.as_ref() {
            Some(table) => gather_embedding_rows(&runtime.backend, table, &tokens, cfg.hidden_size).map_err(crate::backend::compute_error)?,
            None => runtime.backend.tensor_from_f32(model_weights.embedding_rows(&tokens).map_err(crate::backend::compute_error)?, tokens.len(), cfg.hidden_size).map_err(crate::backend::compute_error)?,
        };
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
            embeddings.push(match runtime.embedding.as_ref() {
                Some(table) => gather_embedding_rows(&runtime.backend, table, &[item.token], cfg.hidden_size).map_err(crate::backend::compute_error)?,
                None => runtime.backend.tensor_from_f32(model_weights.embedding_rows(&[item.token]).map_err(crate::backend::compute_error)?, 1, cfg.hidden_size).map_err(crate::backend::compute_error)?,
            });
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

/// GGUF 源装载期把 token_embd 全表解码为 BF16 并常驻指定卡。
pub(in crate::runtime::glm52) fn load_resident_embedding(context: &RocmContext, weights: &Glm52Weights, cfg: &Glm52Config) -> Result<Option<std::sync::Arc<ops::hip::DeviceBuffer>>, String> {
    let Some(bytes) = weights.embedding_table_bf16()? else { return Ok(None) };
    context.activate().map_err(|error| format!("激活 embedding 常驻卡: {error:?}"))?;
    let started = Instant::now();
    let table = ops::hip::DeviceBuffer::upload(context.device_id(), &bytes).map_err(|error| format!("上传 embedding 表: {error}"))?;
    eprintln!("[glm52-embedding-resident] device={} vocab={} hidden={} bytes={} wall={:.3}s", context.device_id(), cfg.vocab_size, cfg.hidden_size, bytes.len(), started.elapsed().as_secs_f64());
    Ok(Some(std::sync::Arc::new(table)))
}

/// head 首卡(A0)的输出运行时:LM head + 采样常驻,以及可选的 MTP L78 与
/// FR-Spec draft head。tail 采样路径移除后,这是唯一的输出侧装配入口。
pub(in crate::runtime::glm52) fn prepare_head_output_runtime(
    output_context: &RocmContext,
    weights: &Glm52Weights,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    mtp_enabled: bool,
    mtp_draft_tokens: usize,
    mtp_draft_vocabulary: Option<&Path>,
    lm_head_quantization: LmHeadQuantization,
    embedding_table: Option<std::sync::Arc<ops::hip::DeviceBuffer>>,
) -> Result<(Glm52OutputHead<RocmWeight>, Option<RocmMtpRuntime>), String> {
    let final_norm = weights.final_norm().map_err(|error| format!("加载 final norm: {error:?}"))?;
    let lm_head = weights.lm_head_bf16_bytes().map_err(|error| format!("加载 LM head: {error:?}"))?;
    let output_head = prepare_glm52_output_head_quantized(output_context, cfg, &final_norm, LinearWeight::Bf16Bytes(&lm_head), lm_head_quantization).map_err(|error| format!("准备 ROCm GLM output head: {error:?}"))?;
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
            Some(RocmMtpRuntime { backend: *output_context, weights: mtp_weights, experts, draft_head, embedding: embedding_table })
        } else if weights.source_is_gguf() {
            let started = Instant::now();
            output_context.activate()?;
            let layer = weights.load_mtp_layer_gguf().map_err(|error| format!("加载 tail MTP L{}: {error}", cfg.layer_count))?;
            let mtp_weights = prepare_glm52_mtp_gguf(output_context, cfg, mla, &layer).map_err(|error| format!("准备 tail ROCm MTP: {error:?}"))?;
            let mut experts = RocmPrefillExperts::gguf(weights.gguf_source()?);
            experts.preload_layer(output_context, cfg.layer_count, cfg.expert_count).map_err(|error| format!("常驻 ROCm MTP experts: {error:?}"))?;
            eprintln!("[glm52-mtp-resident] device={} layer={} drafts={} wall={:.3}s", output_context.device_id(), cfg.layer_count, mtp_draft_tokens, started.elapsed().as_secs_f64());
            Some(RocmMtpRuntime { backend: *output_context, weights: mtp_weights, experts, draft_head, embedding: embedding_table })
        } else {
            return Err("GLM-5.2 distributed MTP 当前只支持 compressed-tensors/GGUF 权重".to_owned());
        }
    } else {
        None
    };
    Ok((output_head, mtp_runtime))
}
