//! GLM-5.2 ROCm head 侧输出运行时、MTP 与 terminal 驻留生命周期。

use super::*;

/// 只统计原始空闲与已完成、可复用的池内 allocation；不能沿用启动快照。
pub(super) fn refresh_tail_device_memory(devices: &mut [StageDeviceMemory]) -> Result<(), String> {
    for device in devices {
        device.available_bytes = ops::hip::device_admission_available_bytes(device.device)? as u64;
    }
    Ok(())
}

pub(super) struct TailSession {
    pub(super) states: Vec<Glm52StageState>,
    pub(super) placement: usize,
    pub(super) hidden: Option<RocmTensor>,
    pub(super) completed_hidden: Option<(usize, RocmTensor)>,
    pub(super) prompt_hidden: Option<RocmTensor>,
    pub(super) prompt_position: Option<usize>,
    pub(super) position: usize,
    pub(super) resumed: bool,
    pub(super) started: Instant,
}

pub(in crate::runtime::glm52) fn pair_session_reservation(states: &[Glm52StageState], rows: usize, cfg: &Glm52Config) -> Result<Vec<(i32, usize)>, String> {
    let mut result = Vec::with_capacity(states.len() * 2);
    for state in states {
        let peer = state.experts.lock().map_err(|_| "GPU reservation expert 锁中毒")?.operator_peer_context().ok_or("GPU reservation 需要 operator pair")?;
        let layers = state.layer_start..state.layer_start + state.layers.len();
        let mla = state.cache.operator_reservation_bytes(layers.clone(), rows, cfg.kv_lora_rank, cfg.qk_rope_head_dim).map_err(|error| format!("MLA reservation: {error:?}"))?;
        let mut dsa = [0usize; 2];
        // IndexShare 不拥有 history，不能为它创建空 DSA 层；否则首次 decode
        // truncate 会把这个从未写入的空层当成已完成历史。
        for layer in resident_indexer_layers(state.layer_start, &state.layers) {
            let bytes = state.dsa.sequence_shard_reservation_bytes(layer..layer + 1, rows).map_err(|error| format!("DSA reservation: {error:?}"))?;
            for parity in 0..2 {
                dsa[parity] = dsa[parity].saturating_add(bytes[parity]);
            }
        }
        for (parity, context) in [state.backend, peer].into_iter().enumerate() {
            result.push((context.device_id(), mla[parity].saturating_add(dsa[parity])));
        }
    }
    Ok(result)
}

fn resident_indexer_layers<W>(start: usize, layers: &[RuntimeGlm52PrefillLayer<W>]) -> impl Iterator<Item = usize> + '_ {
    layers.iter().enumerate().filter_map(move |(offset, layer)| {
        let has_indexer = match layer {
            RuntimeGlm52PrefillLayer::Dense(weights) => weights.indexer.is_some(),
            RuntimeGlm52PrefillLayer::Moe(weights) => weights.indexer.is_some(),
        };
        has_indexer.then_some(start + offset)
    })
}

pub(in crate::runtime::glm52) fn reserve_pair_session(states: &mut [Glm52StageState], rows: usize, cfg: &Glm52Config) -> Result<(), String> {
    // 每个 stage 独占本层的 MLA/DSA 和设备对；复制主存历史与预留可同时进行。
    // 错误返回前也要 join 全部工作，不能让资源预算早于实际分配生命周期释放。
    std::thread::scope(|scope| {
        let jobs = states
            .iter_mut()
            .map(|state| {
                scope.spawn(move || -> Result<(), String> {
                    state.backend.activate().map_err(|error| format!("预留 L{} 激活 device={}: {error}", state.layer_start, state.backend.device_id()))?;
                    let peer = state.experts.lock().map_err(|_| "GPU reservation expert 锁中毒")?.operator_peer_context().ok_or("GPU reservation 需要 operator pair")?;
                    let layers = state.layer_start..state.layer_start + state.layers.len();
                    let indexers = resident_indexer_layers(state.layer_start, &state.layers).collect::<Vec<_>>();
                    for layer in indexers {
                        state.dsa.reserve_sequence_shard_rows(&state.backend, &peer, layer..layer + 1, rows).map_err(|error| format!("预留 L{layer} DSA 两分块: {error:?}"))?;
                    }
                    state.cache.reserve_operator_rows(&state.backend, &peer, layers, rows, cfg.kv_lora_rank, cfg.qk_rope_head_dim).map_err(|error| format!("预留 L{} MLA 热窗及元数据: {error:?}", state.layer_start))?;
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        let mut error = None;
        for job in jobs {
            if let Err(message) = job.join().map_err(|_| "GLM stage cache 预留线程 panic".to_owned()).and_then(|result| result) {
                error.get_or_insert(message);
            }
        }
        error.map_or(Ok(()), Err)
    })
}

// placement 与 Indexer 分布在服务期间固定；启动时统计，查询余量不等待计算锁。
pub(super) fn tail_indexer_layer_counts(templates: &[Vec<Glm52StageState>], devices: &[StageDeviceMemory]) -> Result<Vec<usize>, String> {
    let mut indexers = HashMap::<i32, usize>::new();
    for template in templates {
        let mut plan = HashMap::<i32, usize>::new();
        for state in template {
            let peer = state.experts.lock().map_err(|_| "GPU indexer 计数 expert 锁中毒")?.operator_peer_context().ok_or("GPU indexer 计数需要 operator pair")?;
            for context in [state.backend, peer] {
                *plan.entry(context.device_id()).or_default() += resident_indexer_layers(state.layer_start, &state.layers).count();
            }
        }
        for (device, count) in plan {
            let total = indexers.entry(device).or_default();
            *total = (*total).max(count);
        }
    }
    Ok(devices.iter().map(|device| indexers.get(&device.device).copied().unwrap_or(0)).collect())
}

fn fresh_tail_reservation(templates: &[Vec<Glm52StageState>], cfg: &Glm52Config, max_seq_len: usize, rows: usize) -> Result<Vec<(i32, usize)>, String> {
    // Open 可以选择不同 placement，逐设备取各方案上界，不能假定首方案。
    let mut worst = HashMap::<i32, usize>::new();
    for template in templates {
        let fresh = template.iter().map(|state| state.fresh_session(cfg, max_seq_len)).collect::<Result<Vec<_>, _>>().map_err(|error| format!("创建 GPU 准入计算用空 cache: {error:?}"))?;
        let mut plan = HashMap::<i32, usize>::new();
        for (device, bytes) in pair_session_reservation(&fresh, rows, cfg)? {
            *plan.entry(device).or_default() += bytes;
        }
        for (device, bytes) in plan {
            let total = worst.entry(device).or_default();
            *total = (*total).max(bytes);
        }
    }
    Ok(worst.into_iter().collect())
}

fn deduct_pending_tail_memory(devices: &mut [StageDeviceMemory], pending: &[TailPendingOpen]) {
    for device in devices {
        let held = pending
            .iter()
            .filter_map(|item| match &item.work {
                TailOpenWork::Read { required, .. } => Some(required),
                TailOpenWork::Prepare { .. } => None,
            })
            .flatten()
            .filter(|(id, _)| *id == device.device)
            .fold(0_u64, |sum, (_, bytes)| sum.saturating_add(*bytes as u64));
        device.available_bytes = device.available_bytes.saturating_sub(held);
    }
}

pub(super) fn report_tail_memory_requirement(
    link: &mut StageTransport,
    devices: &mut [StageDeviceMemory],
    resident: &mut HashMap<RequestId, TailResident>,
    swap: &Glm52SwapStore,
    output_context: &RocmContext,
    templates: &[Vec<Glm52StageState>],
    cfg: &Glm52Config,
    max_seq_len: usize,
    cache_request_id: Option<RequestId>,
    rows: usize,
    indexer_layers: &[usize],
    pending: &[TailPendingOpen],
) -> Result<(), String> {
    let reserve = ops::hip::options().mla_gpu_resident_reserve_bytes.ok_or("实时 GPU 准入需要配置 MLA 保留空间")?;
    if indexer_layers.len() != devices.len() {
        return Err(format!("GPU Indexer 统计数量={}，设备数={}", indexer_layers.len(), devices.len()));
    }
    if rows == 0 {
        refresh_tail_device_memory(devices)?;
        deduct_pending_tail_memory(devices, pending);
        return link.send_memory_requirement(devices, &vec![0; devices.len()], indexer_layers);
    }
    loop {
        let required = if let Some(saved) = cache_request_id.and_then(|id| resident.get(&id)) { pair_session_reservation(&saved.states, rows, cfg)? } else { fresh_tail_reservation(templates, cfg, max_seq_len, rows)? };
        refresh_tail_device_memory(devices)?;
        deduct_pending_tail_memory(devices, pending);
        let bytes = devices.iter().map(|device| required.iter().filter(|(id, _)| *id == device.device).map(|(_, bytes)| *bytes as u64).sum::<u64>()).collect::<Vec<_>>();
        if devices.iter().zip(&bytes).all(|(device, bytes)| device.available_bytes >= bytes.saturating_add(reserve as u64)) {
            return link.send_memory_requirement(devices, &bytes, indexer_layers);
        }
        let victim = resident.iter().filter(|(id, _)| Some(**id) != cache_request_id).min_by_key(|(_, saved)| saved.completed_unix).map(|(id, _)| *id);
        let Some(victim) = victim else { return link.send_memory_requirement(devices, &bytes, indexer_layers) };
        tail_cache_to_host(resident, swap, output_context, victim)?;
    }
}
/// tail stage 的 resident cache:已完成的 session 状态,等待复用或换出。
/// 输出与 MTP 驻留 head 首卡,tail 的 resident 只含 KV/DSA 与 terminal hidden。
pub(super) struct TailResident {
    pub(super) states: Vec<Glm52StageState>,
    pub(super) placement: usize,
    pub(super) hidden: RocmTensor,
    pub(super) position: usize,
    pub(super) completed_unix: u64,
}

#[derive(Clone)]
pub(super) struct TailOpenRequest {
    pub(super) request_id: RequestId,
    pub(super) cache_request_id: Option<RequestId>,
    pub(super) cached_tokens: usize,
    pub(super) reserved_rows: usize,
    pub(super) cache_hit: bool,
    pub(super) sampling: SamplingConfig,
}

pub(super) struct TailPendingOpen {
    pub(super) request: TailOpenRequest,
    work: TailOpenWork,
}

enum TailOpenWork {
    // 尚未实际分配的 GPU 字节；进入 Prepare 后物理 free 已反映占用。
    Read { receiver: std::sync::mpsc::Receiver<Result<Option<Arc<Glm52CacheSnapshot>>, String>>, required: Vec<(i32, usize)> },
    Prepare { session: TailSession, worker: Glm52HotHistoryPrepare },
}

/// resident hit 立即完成；SSD 预读与注册保留在 pending 中，完成后才发送 Ready。
#[allow(clippy::too_many_arguments)]
pub(super) fn begin_tail_stage_open(
    link: &mut StageTransport,
    active: &mut HashMap<RequestId, TailSession>,
    resident: &mut HashMap<RequestId, TailResident>,
    pending: &mut Vec<TailPendingOpen>,
    swap: &Arc<Glm52SwapStore>,
    output_context: &RocmContext,
    templates: &[Vec<Glm52StageState>],
    cfg: &Glm52Config,
    max_seq_len: usize,
    request: TailOpenRequest,
) -> Result<(), String> {
    if active.contains_key(&request.request_id) || pending.iter().any(|item| item.request.request_id == request.request_id) {
        return Err(format!("后继重复 Open request={}", request.request_id));
    }
    let needs_ssd = request.cache_hit && request.cache_request_id.is_some_and(|cache_id| !resident.contains_key(&cache_id));
    if !needs_ssd {
        return tail_stage_open(link, active, resident, pending, swap, output_context, templates, cfg, max_seq_len, request, None);
    }
    let hip = ops::hip::options();
    let required = if hip.mla_gpu_resident_reserve_bytes.is_some()
        && hip.mla_cpu_hot_rows > 64
        && !hip.kv_f16
        && !hip.dsa_cpu_select
        && !hip.prefill_attention_cpu
        && templates.first().and_then(|states| states.first()).is_some_and(|state| state.experts.lock().ok().is_some_and(|experts| experts.operator_peer_context().is_some()))
    {
        let rows = request.reserved_rows.max(request.cached_tokens).min(request.cached_tokens.saturating_add(1).div_ceil(4096).saturating_mul(4096)).min(max_seq_len);
        fresh_tail_reservation(templates, cfg, max_seq_len, rows)?
    } else {
        Vec::new()
    };
    let cache_id = request.cache_request_id.expect("needs_ssd 已检查 cache id").to_string();
    let request_id = request.request_id;
    let swap = Arc::clone(swap);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name(format!("glm52-tail-{}", request_id.to_string().chars().take(12).collect::<String>()))
        .spawn(move || {
            let started = Instant::now();
            let result = swap.load(&cache_id);
            eprintln!("[glm52-swap-prefetch] side=tail request_id={request_id} cache_id={cache_id} hit={} read_ms={:.3}", result.as_ref().ok().is_some_and(Option::is_some), started.elapsed().as_secs_f64() * 1000.0);
            let _ = sender.send(result);
        })
        .map_err(|error| format!("启动 tail SSD prefetch: {error}"))?;
    pending.push(TailPendingOpen { request, work: TailOpenWork::Read { receiver, required } });
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
    templates: &[Vec<Glm52StageState>],
    cfg: &Glm52Config,
    max_seq_len: usize,
) -> Result<bool, String> {
    let mut progressed = false;
    let mut index = 0;
    while index < pending.len() {
        let snapshot = match &pending[index].work {
            TailOpenWork::Read { receiver, .. } => Some(match receiver.try_recv() {
                Ok(result) => result,
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    index += 1;
                    continue;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("tail SSD prefetch 线程提前退出".to_owned()),
            }),
            TailOpenWork::Prepare { worker, .. } => {
                if !worker.is_finished() {
                    index += 1;
                    continue;
                }
                None
            }
        };
        let item = pending.remove(index);
        match item.work {
            TailOpenWork::Read { .. } => tail_stage_open(link, active, resident, pending, swap, output_context, templates, cfg, max_seq_len, item.request, Some(snapshot.expect("Read 必有结果")))?,
            TailOpenWork::Prepare { mut session, worker } => {
                session.states = worker.finish()?;
                session.started = Instant::now();
                let request_id = item.request.request_id;
                eprintln!("[glm52-session-placement] request_id={request_id} plan={} cache_hit={}", session.placement, session.resumed);
                active.insert(request_id, session);
                link.send_ready(request_id, item.request.cached_tokens)?;
            }
        }
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
    pending: &mut Vec<TailPendingOpen>,
    swap: &Glm52SwapStore,
    output_context: &RocmContext,
    templates: &[Vec<Glm52StageState>],
    cfg: &Glm52Config,
    max_seq_len: usize,
    request: TailOpenRequest,
    prefetched: Option<Result<Option<Arc<Glm52CacheSnapshot>>, String>>,
) -> Result<(), String> {
    let TailOpenRequest { request_id, cache_request_id, cached_tokens, reserved_rows, cache_hit, sampling: sampling_config } = request.clone();
    if active.contains_key(&request_id) {
        return Err(format!("后继重复 Open active request={request_id}"));
    }
    let mla_bytes = cfg.kv_lora_rank + cfg.kv_lora_rank / crate::kv_cache::DEFAULT_GROUP_SIZE * 2 + cfg.qk_rope_head_dim * 2;
    let layers = templates.iter().map(|states| states.iter().map(|state| state.layers.len()).sum::<usize>()).max().unwrap_or(0);
    // 后继独立检查主存余量；按两份 mirror 预留，不能只依赖链头机器的 RAM。
    swap.spill_host(reserved_rows.max(cached_tokens).saturating_mul(layers).saturating_mul(2).saturating_mul(mla_bytes) as u64)?;
    let fresh_states = |placement: usize| {
        templates
            .get(placement)
            .ok_or_else(|| format!("tail placement={placement} 不存在"))?
            .iter()
            .map(|state| state.fresh_session(cfg, max_seq_len))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("创建 tail placement={placement} session: {error:?}"))
    };
    let mut placement = choose_tail_placement(active, templates)?;
    let mut states = fresh_states(placement)?;
    let mut hidden = None;
    let mut restored_host = false;
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
            placement = saved.placement;
            states = saved.states;
            hidden = Some(saved.hidden);
        } else {
            let cache_id = cache_request_id.to_string();
            let snapshot = match prefetched {
                Some(result) => result?,
                None => swap.load(&cache_id)?,
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
            placement = templates
                .iter()
                .position(|template| template.len() == snapshot.stages.len() && template.iter().zip(&snapshot.stages).all(|(state, cached)| state.layer_start == cached.layer_start))
                .ok_or_else(|| format!("tail SSD cache 的 placement 与当前 {} 个方案均不匹配", templates.len()))?;
            states = fresh_states(placement)?;
            // 未生成的输出与尚未计算的 append 不占物理 cache；先恢复已有前缀及下一小段。
            let restore_rows = reserved_rows.min(cached_tokens.saturating_add(1).div_ceil(4096).saturating_mul(4096)).min(max_seq_len);
            upload_glm52_session(&mut states, &snapshot.stages, cfg, max_seq_len, restore_rows)?;
            restored_host = true;
            hidden = Some(output_context.tensor_from_bf16_bits(snapshot.last_hidden.clone(), 1, cfg.hidden_size).map_err(|error| format!("恢复 tail terminal hidden: {error:?}"))?);
            eprintln!("[stage-swap-in] cache_id={request_id} tokens={cached_tokens}");
        }
    } else {
        swap.delete(&request_id.to_string())?;
    }
    // 采样参数在 head 侧消费;这里只做一次合法性校验,坏参数在 Open 期暴露。
    SamplingState::new(sampling_config)?;
    let hip = ops::hip::options();
    let live_reservation = hip.mla_gpu_resident_reserve_bytes.is_some()
        && hip.mla_cpu_hot_rows > 64
        && !hip.kv_f16
        && !hip.dsa_cpu_select
        && !hip.prefill_attention_cpu
        && states.first().is_some_and(|state| state.experts.lock().ok().is_some_and(|experts| experts.operator_peer_context().is_some()));
    if live_reservation {
        let next_rows = reserved_rows.max(cached_tokens).min(cached_tokens.saturating_add(1).div_ceil(4096).saturating_mul(4096)).min(max_seq_len);
        reserve_pair_session(&mut states, next_rows, cfg)?;
    }
    let mut session = TailSession { states, placement, hidden, completed_hidden: None, prompt_hidden: None, prompt_position: None, position: cached_tokens, resumed: cache_hit, started: Instant::now() };
    if restored_host && live_reservation {
        // 两份 DSA 已实际分配。注册只拥有这份待激活状态，主循环继续收发已有 decode。
        let rows = reserved_rows.max(cached_tokens).min(cached_tokens.saturating_add(1).div_ceil(4096).saturating_mul(4096)).min(max_seq_len);
        let worker = Glm52HotHistoryPrepare::start(std::mem::take(&mut session.states), rows)?;
        pending.push(TailPendingOpen { request, work: TailOpenWork::Prepare { session, worker } });
        return Ok(());
    }
    active.insert(request_id, session);
    eprintln!("[glm52-session-placement] request_id={request_id} plan={placement} cache_hit={cache_hit}");
    link.send_ready(request_id, cached_tokens)
}

fn choose_tail_placement(active: &HashMap<RequestId, TailSession>, templates: &[Vec<Glm52StageState>]) -> Result<usize, String> {
    let first = templates.first().ok_or("tail 没有 placement template")?;
    let mut active_layers = vec![0_usize; first.len()];
    for session in active.values() {
        let template = templates.get(session.placement).ok_or_else(|| format!("active placement={} 越界", session.placement))?;
        if template.len() != active_layers.len() {
            return Err("tail placement stage 数不一致".to_owned());
        }
        for (load, state) in active_layers.iter_mut().zip(template) {
            *load = load.saturating_add(state.layers.len());
        }
    }
    templates
        .iter()
        .enumerate()
        .map(|(placement, template)| {
            if template.len() != active_layers.len() {
                return Err("tail placement stage 数不一致".to_owned());
            }
            let projected = active_layers.iter().zip(template).map(|(load, state)| load.saturating_add(state.layers.len())).collect::<Vec<_>>();
            let peak = projected.iter().copied().max().unwrap_or(0);
            let imbalance = projected.iter().map(|load| load.saturating_mul(*load)).sum::<usize>();
            Ok((peak, imbalance, placement))
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .min()
        .map(|(_, _, placement)| placement)
        .ok_or_else(|| "tail 没有可用 placement".to_owned())
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
    let completed_unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    resident.insert(next_request_id, TailResident { states: tail.states, placement: tail.placement, hidden, position: tokens, completed_unix });
    link.send_ready(request_id, tokens)?;
    Ok(())
}

/// 把 resident session 的 MLA 镜像与 DSA 两分块交给主存，必要时再溢写 SSD。
pub(super) fn tail_stage_swap_out(link: &mut StageTransport, resident: &mut HashMap<RequestId, TailResident>, swap: &Glm52SwapStore, output_context: &RocmContext, request_id: RequestId) -> Result<(), String> {
    let position = tail_cache_to_host(resident, swap, output_context, request_id)?;
    link.send_ready(request_id, position)
}

fn tail_cache_to_host(resident: &mut HashMap<RequestId, TailResident>, swap: &Glm52SwapStore, output_context: &RocmContext, request_id: RequestId) -> Result<usize, String> {
    // 完成全部下载后才移动 MLA；主存接管后，即使 SSD 写入失败仍可恢复。
    let Some(saved) = resident.get_mut(&request_id) else {
        let info = swap.info(&request_id.to_string())?.ok_or_else(|| format!("后继 SwapOut cache={request_id} 不在 resident 或 SSD"))?;
        return Ok(info.token_count);
    };
    let snapshot = Glm52CacheSnapshot {
        cache_id: request_id.to_string(),
        cache_namespace: None,
        tokens: Vec::new(),
        token_count: saved.position,
        pending_tokens: None,
        last_hidden: output_context.tensor_to_bf16_bits(&saved.hidden).map_err(|error| format!("下载 tail terminal hidden: {error:?}"))?,
        stages: take_glm52_session(&mut saved.states)?,
        mtp: None,
        dspark_aux: None,
        dspark_target: None,
    };
    swap.cache_host(Arc::new(snapshot), saved.completed_unix);
    let position = saved.position;
    resident.remove(&request_id);
    if let Err(error) = swap.spill_host(0) {
        eprintln!("[glm52-host-spill-deferred] side=tail cache_id={request_id}: {error}");
    }
    eprintln!("[stage-swap-out] cache_id={request_id} tokens={position}");
    Ok(position)
}

/// 下游显存独立增长，不能等链头下次 Open 才发现压力。仅换出已完成会话，
/// 保留原 cache id；之后的 Open/SwapOut 继续通过现有 RAM/SSD 路径处理。
pub(super) fn trim_tail_gpu_cache(
    resident: &mut HashMap<RequestId, TailResident>,
    swap: &Glm52SwapStore,
    output_context: &RocmContext,
    devices: &[StageDeviceMemory],
    cfg: &Glm52Config,
    rows: usize,
    keep: Option<RequestId>,
) -> Result<(), String> {
    let Some(reserve) = ops::hip::options().mla_gpu_resident_reserve_bytes else { return Ok(()) };
    let row_bytes = cfg.kv_lora_rank + cfg.kv_lora_rank / crate::kv_cache::DEFAULT_GROUP_SIZE * 2 + cfg.qk_rope_head_dim * 2;
    while !resident.is_empty() {
        let mut pressure = false;
        for device in devices {
            let required = reserve.saturating_add(rows.saturating_mul(row_bytes).saturating_mul(device.model_units)).saturating_add(rows.saturating_mul(cfg.hidden_size).saturating_mul(2));
            let (free, _) = ops::hip::device_memory_info(device.device)?;
            if free < required && ops::hip::device_admission_available_bytes(device.device)? < required {
                pressure = true;
                break;
            }
        }
        if !pressure {
            break;
        }
        let Some(cache_id) = resident.iter().filter(|(id, _)| Some(**id) != keep).min_by_key(|(_, saved)| saved.completed_unix).map(|(id, _)| *id) else { break };
        tail_cache_to_host(resident, swap, output_context, cache_id)?;
    }
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
        swap.put_completed(&snapshot, saved.completed_unix)?;
        eprintln!("[stage-persist] cache_id={request_id} tokens={}", saved.position);
    } else if swap.persist_host(&request_id.to_string())? || swap.info(&request_id.to_string())?.is_some() {
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
    Ok(request_ids.len() + swap.persist_host_all()?)
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
    let bytes = unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), ids_bytes) };
    // 调度线程可能没有 active stream。普通 upload 会独立分配，
    // gather 后的 hipFree 因此等待整卡；临时 ids 改走已有显式池，Drop
    // 通过 gather 所在 stream 的完成事件回收。stage0 原有的 default→
    // background 依赖继续保护输出，MTP 内则直接保持同一 consumer stream。
    let ids_buffer = ops::hip::DeviceBuffer::upload_independent(device_id, bytes).map_err(|error| format!("上传 embedding ids: {error}"))?;
    let output = ops::hip::try_gather_bf16_rows_f32(device_id, table, &ids_buffer, ids.len(), hidden).map_err(|error| format!("embedding gather kernel: {error}"))?;
    Ok(RocmTensor { data: Vec::new(), rows: ids.len(), cols: hidden, dtype: crate::backend::rocm::RocmTensorDType::F32, layout: crate::backend::rocm::RocmTensorLayout::RowMajor, device: Some(std::sync::Arc::new(output)), replica: None })
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

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn embedding_pool_ids_survive_stage_handoff() {
        use super::{RocmContext, gather_embedding_rows, ops};
        use crate::backend::StageExecutionBackend;

        ops::hip::configure(ops::hip::RocmOptions { memory_pool: true, ..Default::default() }).unwrap();
        let context = RocmContext::new(0).unwrap();
        context.activate().unwrap();
        let columns = 256;
        let table_values = (0..37 * columns).map(|index| half::bf16::from_f32(((index * 13 % 71) as f32 - 35.0) / 128.0)).collect::<Vec<_>>();
        let table_bytes = table_values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        let table = std::sync::Arc::new(ops::hip::DeviceBuffer::upload(0, &table_bytes).unwrap());
        let mut pending = Vec::new();
        // default stream 构造后立刻交给 background 消费；交错不同 id 与行数，
        // 让已 Drop 的 ids 槽反复进入复用池，检查回收事件确实保护 gather。
        for round in 0..16 {
            context.activate().unwrap();
            let rows = [1, 2, 4, 8, 16, 33, 128, 1024][round % 8];
            let ids = (0..rows).map(|row| ((row * 7 + round * 11) % 37) as u32).collect::<Vec<_>>();
            let gathered = gather_embedding_rows(&context, &table, &ids, columns).unwrap();
            context.activate_stage_submission(crate::backend::StageSubmissionKind::Background).unwrap();
            let device = gathered.device.as_deref().unwrap();
            let output = ops::hip::try_add_resident_f32(0, device, device, rows * columns, 1.0).unwrap();
            pending.push((ids, output));
        }
        ops::hip::synchronize_compute_stream(0, "embedding pool handoff oracle").unwrap();
        for (ids, output) in pending {
            let actual = output.download_f32(ids.len() * columns).unwrap();
            for (row, id) in ids.into_iter().enumerate() {
                for column in 0..columns {
                    assert_eq!(actual[row * columns + column], table_values[id as usize * columns + column].to_f32() * 2.0);
                }
            }
        }
        context.activate().unwrap();
    }
}

/// 同一 ready cohort 的 MTP catch-up 合并为一次 L78 segmented prefill。
#[allow(clippy::too_many_arguments)]
pub(crate) fn mtp_catch_up_batch(
    runtime: &mut RocmMtpRuntime,
    items: &mut [RocmMtpCatchUp<'_>],
    model_weights: &Glm52Weights,
    output_head: &Glm52OutputHead<RocmWeight>,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    rope: &RopeTable,
) -> Result<(), crate::backend::BackendError> {
    let paired = begin_mtp_pair_submission(runtime, cfg)?;
    let mut metadata = vec![None; items.len()];
    let mut embeddings = std::iter::repeat_with(|| None).take(items.len()).collect::<Vec<Option<RocmTensor>>>();
    let mut shifted = std::iter::repeat_with(|| None).take(items.len()).collect::<Vec<Option<RocmTensor>>>();
    runtime.backend.activate().map_err(crate::backend::compute_error)?;
    for (index, item) in items.iter_mut().enumerate() {
        let rows = item.target_hidden.rows;
        if rows == 0 || rows != item.target_inputs.len() {
            return Err(crate::backend::BackendError::Compute { msg: format!("MTP batch catch-up shape 非法: tokens={} hidden=[{},{}]", item.target_inputs.len(), rows, item.target_hidden.cols) });
        }
        // stage 返回的是最终层残差；NextN 训练输入是 target final norm 后的 hidden，
        // 随后才再经过 MTP 自己的 hnorm。单进程路径也遵循同一顺序。
        item.target_hidden = glm52_normalize_target_hidden(&runtime.backend, cfg, output_head, &item.target_hidden)?;
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
            // 驻留状态是 F32，SSD snapshot 按 BF16 恢复；拼接前只扩展存储
            // 类型，不能对已经 normalized 的 pending hidden 再做一次 norm。
            let previous = runtime.backend.tensor_as_f32(previous.clone())?;
            let hidden = if rows == 1 {
                previous
            } else {
                let prefix = runtime.backend.slice_token_rows(&item.target_hidden, 0, rows - 1)?;
                runtime.backend.concat_token_rows(&[&previous, &prefix])?
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
            glm52_mtp_cache_segmented(&runtime.backend, cfg, mla, &runtime.weights, &runtime.experts, &embeddings, &shifted, rope, &mut segments)?;
        }
    }
    for (index, item) in items.iter_mut().enumerate() {
        if let Some((_, rows)) = metadata[index] {
            item.session.position += rows;
        }
        item.session.pending_hidden = Some(runtime.backend.slice_token_rows(&item.target_hidden, item.target_hidden.rows - 1, 1)?);
    }
    finish_mtp_pair_submission(runtime, paired)?;
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
        let paired = begin_mtp_pair_submission(runtime, cfg)?;
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
        // 短上下文还没有稀疏 selection，跨过 top-k 边界时也必须先计算一次。
        // 仅连续且已准备好的 cohort 复用；其余走正常 indexer，补齐本轮 key。
        let dsa_spec = crate::runtime::glm52::glm52_dsa_spec(cfg, mla);
        let reuse_selection = depth > 0 && segments.iter().all(|segment| crate::backend::DsaPrefillBackend::supports_dsa_prefill_selection_reuse(&runtime.backend, &*segment.dsa, cfg.layer_count, segment.position, segment.rows, &dsa_spec));
        let hidden = glm52_mtp_prefill_segmented(&runtime.backend, cfg, mla, &runtime.weights, &mut runtime.experts, &embeddings, &hiddens, rope, reuse_selection, &mut segments)?;
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
        finish_mtp_pair_submission(runtime, paired)?;
    }
    Ok(())
}

fn begin_mtp_pair_submission(runtime: &mut RocmMtpRuntime, cfg: &Glm52Config) -> Result<bool, crate::backend::BackendError> {
    let paired = runtime.backend.supports_parallel_mla_prefill(cfg.layer_count, &runtime.experts);
    if !paired {
        return Ok(false);
    }
    if let Some(completion) = runtime.pending_pair_completion.take() {
        // 上一轮后面已经排入 output norm/head，并在取 token 时完成 owner
        // 同步；这里通常只是无阻塞退休 peer 的尾部 event。
        runtime.backend.wait_stage_completion(&completion)?;
    }
    runtime.backend.begin_stage_submission()?;
    Ok(true)
}

fn finish_mtp_pair_submission(runtime: &mut RocmMtpRuntime, paired: bool) -> Result<(), crate::backend::BackendError> {
    if paired {
        runtime.pending_pair_completion = Some(runtime.backend.record_stage_completion()?);
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

/// head 首卡(A0)拥有输出运行时:LM head + 采样常驻,以及可选的 MTP L78 与
/// FR-Spec draft head。operator pair 下 L78 同时装配 A1；tail 采样路径移除后,
/// 这是唯一的输出侧装配入口。
pub(in crate::runtime::glm52) fn prepare_head_output_runtime(
    output_context: &RocmContext,
    operator_peer: Option<&RocmContext>,
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
            if operator_peer.is_some() {
                return Err("ROCm MTP operator pair 当前要求 GGUF 权重".to_owned());
            }
            let started = Instant::now();
            output_context.activate()?;
            let source = weights.ct_source()?;
            let layer = source.load_mtp_layer(cfg.layer_count).map_err(|error| format!("加载 tail MTP L{}: {error:?}", cfg.layer_count))?;
            let mtp_weights = prepare_glm52_mtp_ct(output_context, cfg, mla, &layer).map_err(|error| format!("准备 tail ROCm MTP: {error:?}"))?;
            let mut experts = RocmPrefillExperts::ct(source);
            experts.preload_layer(output_context, cfg.layer_count, cfg.expert_count).map_err(|error| format!("常驻 ROCm MTP experts: {error:?}"))?;
            eprintln!("[glm52-mtp-resident] device={} layer={} drafts={} wall={:.3}s", output_context.device_id(), cfg.layer_count, mtp_draft_tokens, started.elapsed().as_secs_f64());
            Some(RocmMtpRuntime { backend: *output_context, weights: mtp_weights, experts, draft_head, embedding: embedding_table, pending_pair_completion: None })
        } else if weights.source_is_gguf() {
            let started = Instant::now();
            output_context.activate()?;
            let layer = weights.load_mtp_layer_gguf().map_err(|error| format!("加载 tail MTP L{}: {error}", cfg.layer_count))?;
            let mtp_weights = prepare_glm52_mtp_gguf(output_context, cfg, mla, &layer).map_err(|error| format!("准备 tail ROCm MTP: {error:?}"))?;
            let mut experts = RocmPrefillExperts::gguf(weights.gguf_source()?);
            if let Some(peer) = operator_peer {
                experts.enable_operator_peer(*peer).map_err(|error| format!("配置 ROCm MTP operator peer: {error:?}"))?;
                super::prepare_operator_mla_layer_gguf(output_context, peer, &mut experts, cfg, mla, cfg.layer_count, layer.layer, (&mtp_weights.layer.q_b_proj, &mtp_weights.layer.kv_b_proj))?;
            }
            experts.preload_layer(output_context, cfg.layer_count, cfg.expert_count).map_err(|error| format!("常驻 ROCm MTP experts: {error:?}"))?;
            eprintln!(
                "[glm52-mtp-resident] device={} peer={} layer={} drafts={} operator_pair={} wall={:.3}s",
                output_context.device_id(),
                operator_peer.map_or(-1, RocmContext::device_id),
                cfg.layer_count,
                mtp_draft_tokens,
                operator_peer.is_some(),
                started.elapsed().as_secs_f64(),
            );
            Some(RocmMtpRuntime { backend: *output_context, weights: mtp_weights, experts, draft_head, embedding: embedding_table, pending_pair_completion: None })
        } else {
            return Err("GLM-5.2 distributed MTP 当前只支持 compressed-tensors/GGUF 权重".to_owned());
        }
    } else {
        None
    };
    Ok((output_head, mtp_runtime))
}

#[cfg(test)]
mod reservation_tests {
    use super::*;
    use crate::runtime::glm52::{Glm52DensePrefillLayer, Glm52IndexerDecodeWeights};
    use std::sync::Mutex;

    #[test]
    #[ignore = "需要四张 ROCm GPU"]
    fn parallel_pair_reservation_preserves_history_and_joins_on_error() {
        use crate::backend::rocm::{DsaLayerSerde, MlaLayerSerde, RocmKvOwnership};
        use crate::runtime::glm52::rocm_swap::Glm52StageCache;
        struct NoExperts;
        impl crate::weight::expert_source::GgufExpertSource for NoExperts {
            fn intermediate(&self) -> usize {
                1
            }
            fn hidden(&self) -> usize {
                1
            }
            fn load_expert_gguf(&self, _: usize, _: usize) -> Result<crate::weight::expert_source::GgufExpertWeights, String> {
                Err("缓存预留测试不能读取权重".to_owned())
            }
        }
        ops::hip::configure(ops::hip::RocmOptions { mla_cpu_hot_rows: 2304, mla_gpu_resident_reserve_bytes: Some(0), ..Default::default() }).unwrap();
        let mut cfg = Glm52Config::standard();
        cfg.layer_count = 2;
        cfg.kv_lora_rank = 64;
        cfg.qk_rope_head_dim = 8;
        cfg.index_head_dim = 16;
        cfg.index_top_k = 2;
        let mut states = (0..2)
            .map(|stage| {
                let backend = RocmContext::new(stage as i32 * 2).unwrap();
                let w = backend.prepare_f32(&[1.0], 1, 1).unwrap();
                let mut experts = RocmPrefillExperts::gguf(Arc::new(NoExperts));
                experts.enable_operator_peer(RocmContext::new(stage as i32 * 2 + 1).unwrap()).unwrap();
                Glm52StageState {
                    backend,
                    layer_start: stage,
                    layers: Arc::new(vec![RuntimeGlm52PrefillLayer::Dense(Glm52DensePrefillLayer {
                        indexer: Some(Glm52IndexerDecodeWeights { wq_b: w.clone(), wk: w.clone(), weights_proj: w.clone(), k_norm_weight: w.clone(), k_norm_bias: w.clone() }),
                        input_norm: w.clone(),
                        q_a_proj: w.clone(),
                        q_a_norm: w.clone(),
                        q_b_proj: w.clone(),
                        kv_a_proj: w.clone(),
                        kv_a_norm: w.clone(),
                        kv_b_proj: w.clone(),
                        o_proj: w.clone(),
                        post_attn_norm: w.clone(),
                        gate_proj: w.clone(),
                        up_proj: w.clone(),
                        down_proj: w,
                    })]),
                    experts: Arc::new(Mutex::new(experts)),
                    cache: RocmKvCache::with_capacity(2, 8192),
                    dsa: RocmDsaState::new(2, 8192, 16, 2).unwrap(),
                    decode_active: false,
                    hidden_projectors: Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        let snapshots = (0..2)
            .map(|stage| {
                let rows = 3001 + stage * 1100;
                let mut kv = vec![None, None];
                kv[stage] = Some(MlaLayerSerde {
                    rows,
                    ownership: RocmKvOwnership::Full,
                    latent_cols: 64,
                    rope_cols: 8,
                    latent_group_size: 64,
                    latent: (0..rows * 64).map(|i| (i * 17 + stage * 11) as u8).collect(),
                    latent_scales: Some(vec![0x38; rows * 2]),
                    rope: vec![stage as u8 + 1; rows * 16],
                });
                let mut dsa = vec![None, None];
                dsa[stage] = Some(DsaLayerSerde { rows, key_group_size: 16, hadamard: false, keys: (0..rows * 16).map(|i| (i * 31 + stage * 7) as u8).collect(), scales: vec![0x38; rows * 2] });
                Glm52StageCache { layer_start: stage, kv, dsa }
            })
            .collect::<Vec<_>>();
        upload_glm52_session(&mut states, &snapshots, &cfg, 8192, 8192).unwrap();
        reserve_pair_session(&mut states, 8192, &cfg).unwrap();
        for (stage, (actual, expected)) in download_glm52_session(&states).unwrap().iter().zip(&snapshots).enumerate() {
            let actual_kv = actual.kv[stage].as_ref().unwrap();
            let expected_kv = expected.kv[stage].as_ref().unwrap();
            assert_eq!(actual_kv.latent, expected_kv.latent);
            assert_eq!(actual_kv.latent_scales, expected_kv.latent_scales);
            assert_eq!(actual_kv.rope, expected_kv.rope);
            assert_eq!(actual.dsa[stage].as_ref().unwrap().keys, expected.dsa[stage].as_ref().unwrap().keys);
            assert_eq!(actual.dsa[stage].as_ref().unwrap().scales, expected.dsa[stage].as_ref().unwrap().scales);
        }
        upload_glm52_session(&mut states, &snapshots, &cfg, 8192, 8192).unwrap();
        // 第一组失败仍必须完成第二组预留，调用方才能安全释放会话及预算。
        states[0].experts = Arc::new(Mutex::new(RocmPrefillExperts::gguf(Arc::new(NoExperts))));
        assert!(reserve_pair_session(&mut states, 8192, &cfg).unwrap_err().contains("需要 operator pair"));
        assert!(pair_session_reservation(&states[1..], 8192, &cfg).unwrap().iter().all(|(_, bytes)| *bytes == 0));
        let restored = states[1].dsa.download_layers().unwrap();
        assert_eq!(restored[1].as_ref().unwrap().keys, snapshots[1].dsa[1].as_ref().unwrap().keys);
    }

    #[test]
    fn reservation_skips_index_share_history() {
        // stage 从 IndexShare 开始；不能按 stage 的首层或层数推测 Indexer。
        let layers = [false, true, false, false, false, true].map(|has_indexer| {
            RuntimeGlm52PrefillLayer::Dense(Glm52DensePrefillLayer {
                indexer: has_indexer.then_some(Glm52IndexerDecodeWeights { wq_b: (), wk: (), weights_proj: (), k_norm_weight: (), k_norm_bias: () }),
                input_norm: (),
                q_a_proj: (),
                q_a_norm: (),
                q_b_proj: (),
                kv_a_proj: (),
                kv_a_norm: (),
                kv_b_proj: (),
                o_proj: (),
                post_attn_norm: (),
                gate_proj: (),
                up_proj: (),
                down_proj: (),
            })
        });
        assert_eq!(resident_indexer_layers(9, &layers).collect::<Vec<_>>(), [10, 14]);
        assert_eq!(resident_indexer_layers(9, &layers[..1]).count(), 0);
        assert_eq!(resident_indexer_layers::<()>(9, &[]).count(), 0);
    }
}

#[cfg(test)]
mod pending_open_tests {
    use super::*;

    #[test]
    fn pending_open_reservation_is_released_after_physical_allocation() {
        let request = |id| TailOpenRequest { request_id: RequestId::from_cache_id(id), cache_request_id: None, cached_tokens: 128, reserved_rows: 4096, cache_hit: true, sampling: SamplingConfig::greedy(0) };
        let (_, rx0) = std::sync::mpsc::channel();
        let (_, rx1) = std::sync::mpsc::channel();
        let session = TailSession { states: Vec::new(), placement: 0, hidden: None, completed_hidden: None, prompt_hidden: None, prompt_position: None, position: 128, resumed: true, started: Instant::now() };
        let mut pending = vec![
            TailPendingOpen { request: request("first"), work: TailOpenWork::Read { receiver: rx0, required: vec![(2, 200), (3, 400)] } },
            TailPendingOpen { request: request("second"), work: TailOpenWork::Read { receiver: rx1, required: vec![(2, 100)] } },
            TailPendingOpen { request: request("allocated"), work: TailOpenWork::Prepare { session, worker: Glm52HotHistoryPrepare::start(Vec::new(), 0).unwrap() } },
        ];
        let mut devices = vec![
            StageDeviceMemory { device: 2, model_units: 10, available_bytes: 1000, total_bytes: 2000 },
            StageDeviceMemory { device: 3, model_units: 10, available_bytes: 300, total_bytes: 2000 },
            StageDeviceMemory { device: 4, model_units: 10, available_bytes: 700, total_bytes: 2000 },
        ];
        deduct_pending_tail_memory(&mut devices, &pending);
        assert_eq!(devices.iter().map(|d| d.available_bytes).collect::<Vec<_>>(), [700, 0, 700]);
        // 第一份已物理分配或被取消，新鲜 free 不能再扣它；Prepare 本身不重复扣除。
        pending.remove(0);
        for (device, free) in devices.iter_mut().zip([1000, 300, 700]) {
            device.available_bytes = free;
        }
        deduct_pending_tail_memory(&mut devices, &pending);
        assert_eq!(devices.iter().map(|d| d.available_bytes).collect::<Vec<_>>(), [900, 300, 700]);
    }
}
