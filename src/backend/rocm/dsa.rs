use std::sync::Arc;

use crate::backend::cpu::CpuDsaState;
use crate::backend::{BackendError, compute_error};
use crate::kernel::rocm as ops;

use super::{ROCM_KV_BLOCK_SIZE, RocmBlockTable, RocmContext, RocmTensor, committed_cache_rows, grow_cache_buffer, upload_cache_buffer};

struct RocmPagedDsaLayer {
    keys: Arc<ops::hip::DeviceBuffer>,
    scales: Arc<ops::hip::DeviceBuffer>,
    /// CPU global Top-K 的 Q8 blocked 全历史副本；按需从 GPU cache 构建，此后每 token 只追加一行。
    cpu_keys: Option<crate::kernel::cpu::dsa::Q8KeyBlocks>,
    /// 仅 profile shadow 使用；不序列化，也不参与生产 selection。
    hadamard_shadow_keys: Option<Arc<ops::hip::DeviceBuffer>>,
    hadamard_shadow_scales: Option<Arc<ops::hip::DeviceBuffer>>,
    hadamard: bool,
    gates: Option<Arc<ops::hip::DeviceBuffer>>,
    pooled_keys: Option<Arc<ops::hip::DeviceBuffer>>,
    pooled_scales: Option<Arc<ops::hip::DeviceBuffer>>,
    pooled_rows: usize,
    interval_lower: Option<Arc<ops::hip::DeviceBuffer>>,
    interval_upper: Option<Arc<ops::hip::DeviceBuffer>>,
    interval_rows: usize,
    rows: usize,
    committed_rows: usize,
}

pub struct RocmDsaState {
    cpu: CpuDsaState,
    layers: Vec<Option<RocmPagedDsaLayer>>,
    capacity: usize,
    head_dim: usize,
    key_group_size: usize,
    hadamard_i8: bool,
    hadamard_shadow_samples: usize,
    hadamard_shadow_counts: Vec<usize>,
    hisa_shadow_samples: usize,
    hisa_shadow_counts: Vec<usize>,
    cpu_select: bool,
    cpu_select_counts: Vec<usize>,
    cpu_workspace: crate::kernel::cpu::dsa::Q8DsaWorkspace,
    cpu_dispatcher: Option<CpuDsaDispatcher>,
    cpu_pending: Option<CpuDsaPending>,
    cpu_transfer: Option<Arc<ops::hip::DeviceBuffer>>,
    cpu_host_download: Option<ops::hip::AsyncHostDownload>,
    top_k: usize,
    block_table: RocmBlockTable,
    pool_block_table: RocmBlockTable,
    kpool_apes: Vec<Option<Arc<ops::hip::DeviceBuffer>>>,
    kpool: usize,
    selection: Option<Arc<ops::hip::DeviceBuffer>>,
    selection_host: Option<Arc<Vec<u32>>>,
    selection_rows: usize,
    selection_start: usize,
    selection_width: usize,
    pub(super) decode_parallelism: usize,
}

pub struct RocmDsaSelection {
    buffer: Arc<ops::hip::DeviceBuffer>,
    host: Option<Arc<Vec<u32>>>,
    rows: usize,
    start: usize,
    width: usize,
}

struct CpuDsaWorkerResult {
    keys: crate::kernel::cpu::dsa::Q8KeyBlocks,
    workspace: crate::kernel::cpu::dsa::Q8DsaWorkspace,
    host_download: ops::hip::AsyncHostDownload,
    transfer_ms: f64,
    candidates: Result<(Vec<u32>, f64, f64, f64), String>,
}

struct CpuDsaTask {
    keys: crate::kernel::cpu::dsa::Q8KeyBlocks,
    workspace: crate::kernel::cpu::dsa::Q8DsaWorkspace,
    host_download: ops::hip::AsyncHostDownload,
    transfer_started: std::time::Instant,
    weight_offset: usize,
    key_offset: usize,
    scale_offset: usize,
    append_row: bool,
    head_count: usize,
    head_dim: usize,
    requested_candidates: usize,
}

struct CpuDsaDispatcher {
    tasks: std::sync::mpsc::SyncSender<CpuDsaTask>,
    results: std::sync::mpsc::Receiver<CpuDsaWorkerResult>,
}

impl CpuDsaDispatcher {
    fn new() -> Result<Self, String> {
        let (task_sender, task_receiver) = std::sync::mpsc::sync_channel::<CpuDsaTask>(1);
        let (result_sender, result_receiver) = std::sync::mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("zllm-dsa-dispatch".to_owned())
            .spawn(move || {
                while let Ok(mut task) = task_receiver.recv() {
                    let transfer = task.host_download.wait().map(|host| host.to_vec());
                    let transfer_ms = task.transfer_started.elapsed().as_secs_f64() * 1e3;
                    let candidates = transfer.and_then(|host| {
                        let select_started = std::time::Instant::now();
                        let query = unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<f32>(), task.head_count * task.head_dim) };
                        let weights = unsafe { std::slice::from_raw_parts(host.as_ptr().add(task.weight_offset).cast::<f32>(), task.head_count) };
                        if task.append_row {
                            let key = unsafe { std::slice::from_raw_parts(host.as_ptr().add(task.key_offset).cast::<i8>(), task.head_dim) };
                            let scales = unsafe { std::slice::from_raw_parts(host.as_ptr().add(task.scale_offset).cast::<u16>(), 1) };
                            task.keys.append_q8_row(key, scales[0])?;
                        }
                        let (candidates, score_ms, topk_ms) = crate::kernel::cpu::dsa::select_q8_blocked_candidates(&task.keys, query, weights, task.head_count, task.head_dim, task.requested_candidates, &mut task.workspace)?;
                        let candidates = candidates.to_vec();
                        Ok((candidates, score_ms, topk_ms, select_started.elapsed().as_secs_f64() * 1e3))
                    });
                    if result_sender.send(CpuDsaWorkerResult { keys: task.keys, workspace: task.workspace, host_download: task.host_download, transfer_ms, candidates }).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| format!("CPU DSA dispatcher 启动失败: {error}"))?;
        Ok(Self { tasks: task_sender, results: result_receiver })
    }
}

struct CpuDsaPending {
    layer: usize,
    context_rows: usize,
    head_count: usize,
    device_id: i32,
    transfer_bytes: usize,
    begin_ms: f64,
    started: std::time::Instant,
    dispatched: std::time::Instant,
    keys: Arc<ops::hip::DeviceBuffer>,
    scales: Arc<ops::hip::DeviceBuffer>,
    table: Arc<ops::hip::DeviceBuffer>,
    query: Arc<ops::hip::DeviceBuffer>,
    weights: Arc<ops::hip::DeviceBuffer>,
    exact: Option<Vec<u32>>,
}

const DSA_SHADOW_GUARDS: [usize; 9] = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
const DSA_HISA_POOL_SIZE: usize = 128;
const DSA_HISA_BLOCK_BUDGETS: [usize; 4] = [32, 48, 64, 96];
const DSA_CPU_SHADOW_REPEATS: usize = 64;
const DSA_CPU_CANDIDATE_GUARD: usize = 2048;

#[derive(Debug)]
struct DsaShadowMetrics {
    required_guard: usize,
    recalls: [usize; DSA_SHADOW_GUARDS.len()],
    prefix_candidate_count: usize,
    prefix_recall: usize,
    prefix16_candidate_count: usize,
    prefix16_recall: usize,
    exact_margin: f32,
    coarse_margin: f32,
}

#[derive(Debug)]
struct DsaIntervalMetrics {
    candidate_blocks: usize,
    candidate_tokens: usize,
    exact_recall: usize,
    violations: usize,
    max_excess: f32,
}

fn score_from_ordered(key: u32) -> f32 {
    let bits = if key & 0x8000_0000 != 0 { key ^ 0x8000_0000 } else { key ^ 0xffff_ffff };
    f32::from_bits(bits)
}

fn analyze_hadamard_shadow(exact_selection: &[u32], exact_scores: &[u32], coarse_scores: &[u32], coarse_candidate_count: usize) -> Result<DsaShadowMetrics, String> {
    if exact_selection.is_empty() || exact_scores.len() != coarse_scores.len() || exact_selection.len() >= exact_scores.len() || coarse_candidate_count <= exact_selection.len() || coarse_candidate_count > exact_scores.len() {
        return Err(format!("DSA shadow shape selection={} exact={} coarse={} candidates={coarse_candidate_count} 非法", exact_selection.len(), exact_scores.len(), coarse_scores.len()));
    }
    let mut selected = vec![false; exact_scores.len()];
    for &token in exact_selection {
        let token = token as usize;
        if token >= selected.len() || std::mem::replace(&mut selected[token], true) {
            return Err(format!("DSA shadow exact selection token={token} 越界或重复"));
        }
    }
    let exact_kth = exact_selection.iter().map(|&token| exact_scores[token as usize]).min().unwrap();
    let exact_next = exact_scores.iter().enumerate().filter(|(token, _)| !selected[*token]).map(|(_, &score)| score).max().unwrap();

    // GPU selection 的同分规则是 token 小者优先；host shadow 使用相同全序。
    let mut coarse_order = (0..coarse_scores.len()).collect::<Vec<_>>();
    coarse_order.sort_unstable_by(|&left, &right| coarse_scores[right].cmp(&coarse_scores[left]).then_with(|| left.cmp(&right)));
    let top_k = exact_selection.len();
    let mut recalls = [0usize; DSA_SHADOW_GUARDS.len()];
    let mut worst_rank = 0usize;
    for (rank, &token) in coarse_order.iter().enumerate() {
        if !selected[token] {
            continue;
        }
        worst_rank = worst_rank.max(rank);
        for (index, guard) in DSA_SHADOW_GUARDS.iter().enumerate() {
            if rank < (top_k + guard).min(coarse_order.len()) {
                recalls[index] += 1;
            }
        }
    }
    let coarse_kth = coarse_scores[coarse_order[top_k - 1]];
    let coarse_next = coarse_scores[coarse_order[top_k]];
    let prefix_threshold = coarse_scores[coarse_order[coarse_candidate_count - 1]] >> 24;
    let prefix_candidate_count = coarse_scores.iter().filter(|&&score| score >> 24 >= prefix_threshold).count();
    let prefix_recall = exact_selection.iter().filter(|&&token| coarse_scores[token as usize] >> 24 >= prefix_threshold).count();
    let prefix16_threshold = coarse_scores[coarse_order[coarse_candidate_count - 1]] >> 16;
    let prefix16_candidate_count = coarse_scores.iter().filter(|&&score| score >> 16 >= prefix16_threshold).count();
    let prefix16_recall = exact_selection.iter().filter(|&&token| coarse_scores[token as usize] >> 16 >= prefix16_threshold).count();
    Ok(DsaShadowMetrics {
        required_guard: (worst_rank + 1).saturating_sub(top_k),
        recalls,
        prefix_candidate_count,
        prefix_recall,
        prefix16_candidate_count,
        prefix16_recall,
        exact_margin: score_from_ordered(exact_kth) - score_from_ordered(exact_next),
        coarse_margin: score_from_ordered(coarse_kth) - score_from_ordered(coarse_next),
    })
}

fn hisa_candidate_blocks(ranked_blocks: &[u32], block_budget: usize, pool_rows: usize) -> Result<Vec<u32>, String> {
    if pool_rows < 2 || block_budget < 2 || block_budget > ranked_blocks.len() || block_budget > pool_rows {
        return Err(format!("DSA HISA block budget={block_budget} ranked={} pools={pool_rows} 非法", ranked_blocks.len()));
    }
    let mut blocks = ranked_blocks[..block_budget].to_vec();
    let last = u32::try_from(pool_rows - 1).map_err(|_| "DSA HISA pool 数超过 u32")?;
    let mut replacement = blocks.len();
    for forced in [0u32, last] {
        if !blocks.contains(&forced) {
            replacement -= 1;
            blocks[replacement] = forced;
        }
    }
    blocks.sort_unstable();
    if blocks.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("DSA HISA forced block 产生重复候选".to_owned());
    }
    Ok(blocks)
}

fn hisa_candidate_tokens(blocks: &[u32], context_rows: usize) -> Result<Vec<u32>, String> {
    let mut tokens = Vec::with_capacity(blocks.len() * DSA_HISA_POOL_SIZE);
    for &block in blocks {
        let first = (block as usize).checked_mul(DSA_HISA_POOL_SIZE).ok_or("DSA HISA candidate block 溢出")?;
        let end = first.checked_add(DSA_HISA_POOL_SIZE).ok_or("DSA HISA candidate end 溢出")?.min(context_rows);
        if first >= end {
            return Err(format!("DSA HISA candidate block={block} context={context_rows} 越界"));
        }
        tokens.extend((first..end).map(|token| u32::try_from(token).expect("DSA context 已受 u32 kernel 约束")));
    }
    Ok(tokens)
}

fn analyze_interval_bounds(exact_selection: &[u32], exact_scores: &[u32], bounds: &[f32], pool_size: usize) -> Result<DsaIntervalMetrics, String> {
    if exact_selection.is_empty() || exact_scores.is_empty() || bounds.is_empty() || pool_size == 0 || bounds.len() != exact_scores.len().div_ceil(pool_size) {
        return Err(format!("DSA interval metrics shape selection={} scores={} bounds={} pool={pool_size} 非法", exact_selection.len(), exact_scores.len(), bounds.len()));
    }
    let threshold = exact_selection.iter().map(|&token| exact_scores.get(token as usize).copied().ok_or_else(|| format!("DSA interval exact token={token} 越界"))).collect::<Result<Vec<_>, _>>()?.into_iter().min().unwrap();
    let threshold = score_from_ordered(threshold);
    let mut candidate_blocks = 0usize;
    let mut candidate_tokens = 0usize;
    let mut violations = 0usize;
    let mut max_excess = 0.0_f32;
    for (block, &bound) in bounds.iter().enumerate() {
        if !bound.is_finite() {
            return Err(format!("DSA interval block={block} bound={bound} 非有限"));
        }
        let first = block * pool_size;
        let end = (first + pool_size).min(exact_scores.len());
        let actual = exact_scores[first..end].iter().map(|&score| score_from_ordered(score)).max_by(f32::total_cmp).unwrap();
        if actual > bound {
            violations += 1;
            max_excess = max_excess.max(actual - bound);
        }
        if bound >= threshold {
            candidate_blocks += 1;
            candidate_tokens += end - first;
        }
    }
    let exact_recall = exact_selection.iter().filter(|&&token| bounds[token as usize / pool_size] >= threshold).count();
    Ok(DsaIntervalMetrics { candidate_blocks, candidate_tokens, exact_recall, violations, max_excess })
}

impl RocmDsaSelection {
    pub(crate) fn move_to_stable_deferred(self) -> Result<Self, BackendError> {
        // DSA selection 由显式 reusable pool 分配时已可跨线程/P2P 持有；
        // 只有 stream-ordered allocation 才需要先复制到稳定存储。
        if !self.buffer.is_async_allocated() {
            return Ok(self);
        }
        let buffer = self.buffer.copy_to_stable_deferred().map_err(compute_error)?;
        Ok(Self { buffer: Arc::new(buffer), host: self.host, rows: self.rows, start: self.start, width: self.width })
    }

    pub(crate) fn move_to_device_ordered(self, device_id: i32) -> Result<Self, BackendError> {
        if self.buffer.device_id() == device_id {
            return Ok(self);
        }
        let buffer = self.buffer.copy_stable_to_device_ordered_async(device_id).map_err(compute_error)?;
        Ok(Self { buffer: Arc::new(buffer), host: self.host, rows: self.rows, start: self.start, width: self.width })
    }

    pub fn to_host(&self) -> Result<Vec<u32>, BackendError> {
        if let Some(host) = &self.host {
            return Ok(host.as_ref().clone());
        }
        let bytes = self.buffer.bytes();
        if bytes == 0 || !bytes.is_multiple_of(std::mem::size_of::<u32>()) {
            return Err(compute_error(format!("ROCm DSA selection bytes={bytes} 无效")));
        }
        let mut values = vec![0_u32; bytes / std::mem::size_of::<u32>()];
        self.buffer.copy_to_host(unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), bytes) }).map_err(compute_error)?;
        Ok(values)
    }

    pub fn to_host_completed(&self) -> Result<Vec<u32>, BackendError> {
        if let Some(host) = &self.host {
            return Ok(host.as_ref().clone());
        }
        let bytes = self.buffer.bytes();
        if bytes == 0 || !bytes.is_multiple_of(std::mem::size_of::<u32>()) {
            return Err(compute_error(format!("ROCm completed DSA selection bytes={bytes} 无效")));
        }
        let mut values = vec![0_u32; bytes / std::mem::size_of::<u32>()];
        self.buffer.copy_completed_to_host(unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), bytes) }).map_err(compute_error)?;
        Ok(values)
    }

    pub fn from_host(context: &RocmContext, rows: usize, start: usize, values: &[u32]) -> Result<Self, BackendError> {
        if rows == 0 || values.is_empty() || !values.len().is_multiple_of(rows) {
            return Err(compute_error(format!("ROCm DSA selection host shape rows={rows} elements={} 无效", values.len())));
        }
        let bytes = std::mem::size_of_val(values);
        let buffer = ops::hip::DeviceBuffer::upload_independent(context.device_id, unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), bytes) }).map_err(compute_error)?;
        Ok(Self { buffer: Arc::new(buffer), host: Some(Arc::new(values.to_vec())), rows, start, width: values.len() / rows })
    }
}

impl RocmDsaState {
    pub fn new(layer_count: usize, capacity: usize, head_dim: usize, top_k: usize) -> Result<Self, String> {
        let key_group_size = [128, 64, 32, 16].into_iter().find(|group| head_dim.is_multiple_of(*group)).ok_or_else(|| format!("ROCm DSA head_dim={head_dim} 不支持 Q8 group"))?;
        let hadamard_i8 = ops::hip::options().dsa_hadamard_i8 && head_dim == 128 && key_group_size == head_dim;
        let hadamard_shadow_samples = if !hadamard_i8 && head_dim == 128 && key_group_size == head_dim { ops::hip::options().dsa_hadamard_shadow_samples } else { 0 };
        let hisa_shadow_samples = if !hadamard_i8 && head_dim == 128 && key_group_size == head_dim { ops::hip::options().dsa_hisa_shadow_samples } else { 0 };
        let cpu_select = ops::hip::options().dsa_cpu_select && !hadamard_i8 && head_dim == 128 && key_group_size == head_dim;
        Ok(Self {
            cpu: CpuDsaState::new(layer_count, capacity, head_dim, top_k)?,
            layers: (0..layer_count).map(|_| None).collect(),
            capacity,
            head_dim,
            key_group_size,
            hadamard_i8,
            hadamard_shadow_samples,
            hadamard_shadow_counts: vec![0; layer_count],
            hisa_shadow_samples,
            hisa_shadow_counts: vec![0; layer_count],
            cpu_select,
            cpu_select_counts: vec![0; layer_count],
            cpu_workspace: crate::kernel::cpu::dsa::Q8DsaWorkspace::default(),
            cpu_dispatcher: None,
            cpu_pending: None,
            cpu_transfer: None,
            cpu_host_download: None,
            top_k,
            block_table: RocmBlockTable::new(),
            pool_block_table: RocmBlockTable::new(),
            kpool_apes: (0..layer_count).map(|_| None).collect(),
            kpool: 0,
            selection: None,
            selection_host: None,
            selection_rows: 0,
            selection_start: 0,
            selection_width: top_k,
            decode_parallelism: 1,
        })
    }

    pub(crate) fn prepare_block_table(&mut self, context: &RocmContext) -> Result<(), BackendError> {
        self.block_table.get("DSA", self.capacity, context.device_id, self.capacity)?;
        if self.kpool > 0 || self.hisa_shadow_samples > 0 {
            let pool_capacity = if self.kpool > 0 { self.capacity / self.kpool } else { self.capacity.div_ceil(DSA_HISA_POOL_SIZE) };
            let tag = if self.kpool > 0 { "DSA kpool" } else { "DSA HISA" };
            self.pool_block_table.get(tag, pool_capacity, context.device_id, pool_capacity)?;
        }
        Ok(())
    }

    /// 只回退逻辑长度；decode 写入的尾部不再可见，prompt 前缀无需搬运。
    pub fn truncate_rows(&mut self, rows: usize) -> Result<(), BackendError> {
        if rows > self.capacity {
            return Err(compute_error(format!("ROCm DSA truncate rows={rows} 超过逻辑上限 {}", self.capacity)));
        }
        for (layer, cached) in self.layers.iter_mut().enumerate().filter_map(|(layer, slot)| slot.as_mut().map(|cached| (layer, cached))) {
            if rows > cached.rows {
                return Err(compute_error(format!("L{layer} ROCm DSA truncate rows={rows} 超过当前长度 {}", cached.rows)));
            }
            cached.rows = rows;
            if cached.cpu_keys.as_ref().is_some_and(|keys| keys.rows() < rows) {
                // CPU mirror 只在 select 时推进；会话保存可能发生在 GPU append 之后。
                // 落后的 mirror 直接失效，下次 select 从 GPU cache 重建即可。
                cached.cpu_keys = None;
            } else if let Some(keys) = &mut cached.cpu_keys {
                keys.truncate(rows).map_err(compute_error)?;
            }
            if self.kpool > 0 {
                cached.pooled_rows = rows / self.kpool;
            }
        }
        self.selection = None;
        self.selection_host = None;
        self.selection_rows = 0;
        self.selection_start = 0;
        self.selection_width = self.top_k;
        Ok(())
    }

    /// 只回退指定层；MTP cache 比主干 target cache 固定落后一行。
    pub fn truncate_layer_rows(&mut self, layer: usize, rows: usize) -> Result<(), BackendError> {
        if rows > self.capacity {
            return Err(compute_error(format!("L{layer} ROCm DSA truncate rows={rows} 超过逻辑上限 {}", self.capacity)));
        }
        let slot = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if let Some(cached) = slot {
            if rows > cached.rows {
                return Err(compute_error(format!("L{layer} ROCm DSA truncate rows={rows} 超过当前长度 {}", cached.rows)));
            }
            cached.rows = rows;
            if cached.cpu_keys.as_ref().is_some_and(|keys| keys.rows() < rows) {
                // 同步 decode 与异步 decode 都允许 CPU mirror 落后 GPU 一行。
                cached.cpu_keys = None;
            } else if let Some(keys) = &mut cached.cpu_keys {
                keys.truncate(rows).map_err(compute_error)?;
            }
            if self.kpool > 0 {
                cached.pooled_rows = rows / self.kpool;
            }
        }
        self.selection = None;
        self.selection_host = None;
        self.selection_rows = 0;
        self.selection_start = 0;
        self.selection_width = self.top_k;
        Ok(())
    }

    /// 清空 selection(kpool 等场景退化为全量注意力)。
    pub fn invalidate_selection(&mut self) {
        self.selection = None;
        self.selection_host = None;
        self.selection_rows = 0;
        self.selection_start = 0;
        self.selection_width = self.top_k;
    }

    /// kpool APE 在加载期同时注入 CPU oracle 与对应 device，forward 不再 H2D。
    pub fn set_kpool_ape(&mut self, context: &RocmContext, layer: usize, ape: Vec<f32>) -> Result<(), String> {
        if layer >= self.kpool_apes.len() || ape.is_empty() || !ape.len().is_multiple_of(self.head_dim) {
            return Err(format!("kpool APE L{layer} elements={} head_dim={} 非法", ape.len(), self.head_dim));
        }
        let kpool = ape.len() / self.head_dim;
        if kpool > 8 || (self.kpool != 0 && self.kpool != kpool) {
            return Err(format!("kpool APE L{layer} pool={kpool} 与 state pool={} 不一致", self.kpool));
        }
        if self.layers.iter().any(Option::is_some) && self.hadamard_i8 {
            return Err("DSA kpool 不能接管已经按 Hadamard 约定写入的 cache".to_owned());
        }
        // kpool 的逐维 gate 不与 Hadamard 交换；该模型路径保持原始 Q8 key 约定。
        self.hadamard_i8 = false;
        let bytes = unsafe { std::slice::from_raw_parts(ape.as_ptr().cast(), std::mem::size_of_val(ape.as_slice())) };
        self.kpool_apes[layer] = Some(Arc::new(ops::hip::DeviceBuffer::upload(context.device_id, bytes)?));
        self.kpool = kpool;
        self.cpu.set_kpool_ape(layer, ape)
    }

    pub(super) fn can_append(&mut self, layer: usize, position: usize) -> bool {
        self.selection = None;
        self.selection_host = None;
        self.selection_rows = 0;
        self.selection_start = 0;
        self.selection_width = self.top_k;
        position < self.capacity && self.layers.get(layer).is_some_and(|cached| cached.as_ref().map_or(position == 0, |cached| cached.rows == position))
    }

    fn prepare_append_storage(&mut self, context: &RocmContext, layer: usize, position: usize, rows: usize) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
        if rows == 0 || position.checked_add(rows).is_none_or(|end| end > self.capacity) {
            return Err(compute_error(format!("L{layer} ROCm paged DSA append position={position} rows={rows} capacity={} 非法", self.capacity)));
        }
        let end = position + rows;
        let committed_rows = self.layers.get(layer).and_then(Option::as_ref).map_or(0, |cached| cached.committed_rows);
        let next_committed = committed_cache_rows(committed_rows, end, self.capacity)?;
        let table = self.block_table.get("DSA", self.capacity, context.device_id, end)?;
        let groups_per_row = self.head_dim / self.key_group_size;
        let slot = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if slot.is_none() {
            if position != 0 {
                return Err(compute_error(format!("L{layer} ROCm DSA 首次 append position={position}，期望 0")));
            }
            let key_bytes = next_committed.checked_mul(self.head_dim).ok_or_else(|| compute_error("ROCm paged DSA Q8 key 大小溢出"))?;
            let scale_bytes = next_committed.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm paged DSA Q8 scale 大小溢出"))?;
            *slot = Some(RocmPagedDsaLayer {
                keys: Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, key_bytes).map_err(compute_error)?),
                scales: Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, scale_bytes).map_err(compute_error)?),
                cpu_keys: None,
                hadamard_shadow_keys: None,
                hadamard_shadow_scales: None,
                hadamard: self.hadamard_i8,
                gates: None,
                pooled_keys: None,
                pooled_scales: None,
                pooled_rows: 0,
                interval_lower: None,
                interval_upper: None,
                interval_rows: 0,
                rows: 0,
                committed_rows: next_committed,
            });
        }
        let cached = slot.as_mut().expect("ROCm DSA append storage 已创建");
        if cached.rows != position {
            return Err(compute_error(format!("L{layer} ROCm DSA append 不连续: cached={} position={position}", cached.rows)));
        }
        if next_committed > cached.committed_rows {
            cached.keys = grow_cache_buffer(context.device_id, &cached.keys, cached.rows * self.head_dim, next_committed * self.head_dim)?;
            cached.scales = grow_cache_buffer(context.device_id, &cached.scales, cached.rows * groups_per_row * 2, next_committed * groups_per_row * 2)?;
            if let (Some(keys), Some(scales)) = (cached.hadamard_shadow_keys.clone(), cached.hadamard_shadow_scales.clone()) {
                cached.hadamard_shadow_keys = Some(grow_cache_buffer(context.device_id, &keys, cached.rows * self.head_dim, next_committed * self.head_dim)?);
                cached.hadamard_shadow_scales = Some(grow_cache_buffer(context.device_id, &scales, cached.rows * 2, next_committed * 2)?);
            }
            cached.committed_rows = next_committed;
        }
        Ok(table)
    }

    pub(super) fn append(&mut self, context: &RocmContext, layer: usize, position: usize, keys: &RocmTensor) -> Result<(), BackendError> {
        if keys.rows == 0 || keys.cols != self.head_dim || position.checked_add(keys.rows).is_none_or(|end| end > self.capacity) {
            return Err(compute_error(format!("L{layer} ROCm paged DSA append shape 非法: position={position} rows={} cols={} head_dim={} capacity={}", keys.rows, keys.cols, self.head_dim, self.capacity)));
        }
        let table = self.prepare_append_storage(context, layer, position, keys.rows)?;
        let key_group_size = self.key_group_size;
        let cached = self.layers.get_mut(layer).and_then(Option::as_mut).ok_or(BackendError::UnsupportedLayer { layer })?;
        let input = keys.device.as_deref().ok_or_else(|| compute_error("ROCm paged DSA keys 缺少 device buffer"))?;
        if cached.hadamard {
            ops::hip::try_paged_cache_append_f32_q8_hadamard(context.device_id, input, &cached.keys, &cached.scales, &table, position, keys.rows, keys.cols, key_group_size, ROCM_KV_BLOCK_SIZE).map_err(compute_error)?;
        } else {
            ops::hip::try_paged_cache_append_f32_q8(context.device_id, input, &cached.keys, &cached.scales, &table, position, keys.rows, keys.cols, key_group_size, ROCM_KV_BLOCK_SIZE).map_err(compute_error)?;
            if let (Some(shadow_keys), Some(shadow_scales)) = (&cached.hadamard_shadow_keys, &cached.hadamard_shadow_scales) {
                ops::hip::try_paged_cache_transform_q8_hadamard(context.device_id, &cached.keys, &cached.scales, shadow_keys, shadow_scales, &table, position, keys.rows, self.head_dim, ROCM_KV_BLOCK_SIZE).map_err(compute_error)?;
            }
        }
        cached.rows += keys.rows;
        self.selection = None;
        self.selection_host = None;
        self.selection_rows = 0;
        self.selection_start = 0;
        self.selection_width = self.top_k;
        Ok(())
    }

    pub(super) fn supports_layernorm_rope(&self, rows: usize, cols: usize) -> bool {
        rows > 0 && cols == self.head_dim && self.kpool == 0 && !self.hadamard_i8 && self.key_group_size == self.head_dim
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn append_layernorm_rope(
        &mut self,
        context: &RocmContext,
        layer: usize,
        position: usize,
        keys: &RocmTensor,
        norm_weight: &ops::hip::DeviceBuffer,
        norm_bias: &ops::hip::DeviceBuffer,
        eps: f32,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<bool, BackendError> {
        // 融合核只覆盖 GLM-5.2 的 raw-Q8 单组 key；Hadamard/kpool 保持各自原约定。
        if !self.supports_layernorm_rope(keys.rows, keys.cols) {
            return Ok(false);
        }
        let input = keys.device.as_deref().ok_or_else(|| compute_error("ROCm fused DSA keys 缺少 device buffer"))?;
        let table = self.prepare_append_storage(context, layer, position, keys.rows)?;
        let cached = self.layers.get_mut(layer).and_then(Option::as_mut).ok_or(BackendError::UnsupportedLayer { layer })?;
        if cached.hadamard || cached.gates.is_some() {
            return Err(compute_error(format!("L{layer} ROCm fused DSA append 与现有 cache 约定不一致")));
        }
        ops::hip::try_paged_dsa_append_layernorm_rope_q8(context.device_id, input, norm_weight, norm_bias, &cached.keys, &cached.scales, &table, position, keys.rows, keys.cols, rotary_dim, layout, cos, sin, ROCM_KV_BLOCK_SIZE, eps)
            .map_err(compute_error)?;
        if let (Some(shadow_keys), Some(shadow_scales)) = (&cached.hadamard_shadow_keys, &cached.hadamard_shadow_scales) {
            ops::hip::try_paged_cache_transform_q8_hadamard(context.device_id, &cached.keys, &cached.scales, shadow_keys, shadow_scales, &table, position, keys.rows, self.head_dim, ROCM_KV_BLOCK_SIZE).map_err(compute_error)?;
        }
        cached.rows += keys.rows;
        self.invalidate_selection();
        Ok(true)
    }

    pub(super) fn append_gated(&mut self, context: &RocmContext, layer: usize, position: usize, keys: &RocmTensor, gate: &RocmTensor, kpool: usize) -> Result<(), BackendError> {
        if kpool == 0 || kpool != self.kpool || gate.rows != keys.rows || gate.cols != self.head_dim {
            return Err(compute_error(format!("L{layer} ROCm DSA kpool append key=[{},{}] gate=[{},{}] pool={kpool}/{} 不匹配", keys.rows, keys.cols, gate.rows, gate.cols, self.kpool)));
        }
        let gate_device = gate.device.as_deref().ok_or_else(|| compute_error("ROCm DSA kpool gate 缺少 device buffer"))?;
        let ape = self.kpool_apes.get(layer).and_then(Option::as_ref).cloned().ok_or_else(|| compute_error(format!("L{layer} ROCm DSA kpool APE 未常驻")))?;
        self.append(context, layer, position, keys)?;
        let end = position + keys.rows;
        let pool_capacity = self.capacity / kpool;
        let table = self.block_table.get("DSA", self.capacity, context.device_id, end)?;
        let pool_table = self.pool_block_table.get("DSA kpool", pool_capacity, context.device_id, pool_capacity)?;
        let cached = self.layers.get_mut(layer).and_then(Option::as_mut).ok_or(BackendError::UnsupportedLayer { layer })?;
        if cached.gates.is_none() {
            let gate_bytes = self.capacity.checked_mul(self.head_dim).and_then(|n| n.checked_mul(4)).ok_or_else(|| compute_error("ROCm DSA kpool gate 容量溢出"))?;
            let pooled_key_bytes = pool_capacity.checked_mul(self.head_dim).ok_or_else(|| compute_error("ROCm DSA pooled key 容量溢出"))?;
            let pooled_scale_bytes = pool_capacity.checked_mul(self.head_dim / self.key_group_size).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm DSA pooled scale 容量溢出"))?;
            cached.gates = Some(Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, gate_bytes).map_err(compute_error)?));
            cached.pooled_keys = Some(Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, pooled_key_bytes).map_err(compute_error)?));
            cached.pooled_scales = Some(Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, pooled_scale_bytes).map_err(compute_error)?));
        }
        let gate_bytes = keys.rows.checked_mul(self.head_dim).and_then(|n| n.checked_mul(4)).ok_or_else(|| compute_error("ROCm DSA kpool gate append 大小溢出"))?;
        cached.gates.as_ref().unwrap().copy_from_device(position * self.head_dim * 4, gate_device, 0, gate_bytes).map_err(compute_error)?;
        let first_pool = position / kpool;
        let completed_pools = end / kpool;
        if completed_pools > first_pool {
            ops::hip::try_dsa_kpool_compress_q8(
                context.device_id,
                &cached.keys,
                &cached.scales,
                &table,
                cached.gates.as_ref().unwrap(),
                &ape,
                cached.pooled_keys.as_ref().unwrap(),
                cached.pooled_scales.as_ref().unwrap(),
                &pool_table,
                first_pool,
                completed_pools - first_pool,
                end,
                self.capacity,
                self.head_dim,
                self.key_group_size,
                kpool,
                ROCM_KV_BLOCK_SIZE,
            )
            .map_err(compute_error)?;
        }
        cached.pooled_rows = completed_pools;
        Ok(())
    }

    pub(super) fn select_kpool(&mut self, context: &RocmContext, layer: usize, query: &RocmTensor, head_weights: &RocmTensor, kpool: usize) -> Result<(), BackendError> {
        let cached = self.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} ROCm DSA cache 尚未初始化")))?;
        let context_rows = cached.rows;
        // top_k+kpool-1 宽度已能覆盖全部可见 token 时保持 dense；
        // 这也避免 2049..2051 行时完整池数尚未超过 512 的空选择。
        if context_rows <= self.top_k + kpool - 1 {
            self.invalidate_selection();
            return Ok(());
        }
        if kpool == 0 || kpool != self.kpool || cached.pooled_rows != context_rows / kpool || query.rows == 0 || query.rows > context_rows || query.cols % self.head_dim != 0 {
            return Err(compute_error(format!("L{layer} ROCm DSA kpool select shape 非法: query={:?} context={context_rows} pooled={} pool={kpool}/{}", (query.rows, query.cols), cached.pooled_rows, self.kpool)));
        }
        let head_count = query.cols / self.head_dim;
        if head_weights.rows != query.rows || head_weights.cols != head_count {
            return Err(compute_error(format!("L{layer} ROCm DSA kpool head weight shape {:?}，期望 ({},{head_count})", (head_weights.rows, head_weights.cols), query.rows)));
        }
        let query_device = query.device.as_deref().ok_or_else(|| compute_error("ROCm DSA kpool query 缺少 device buffer"))?;
        let weight_device = head_weights.device.as_deref().ok_or_else(|| compute_error("ROCm DSA kpool head weights 缺少 device buffer"))?;
        let pool_table = self.pool_block_table.get("DSA kpool", self.capacity / kpool, context.device_id, cached.pooled_rows)?;
        self.selection = Some(Arc::new(
            ops::hip::try_dsa_select_paged_q8_kpool(
                context.device_id,
                cached.pooled_keys.as_ref().ok_or_else(|| compute_error("ROCm DSA pooled keys 未初始化"))?,
                cached.pooled_scales.as_ref().ok_or_else(|| compute_error("ROCm DSA pooled scales 未初始化"))?,
                self.key_group_size,
                &pool_table,
                query_device,
                weight_device,
                query.rows,
                context_rows,
                context_rows - query.rows,
                head_count,
                self.head_dim,
                self.top_k,
                kpool,
                ROCM_KV_BLOCK_SIZE,
            )
            .map_err(compute_error)?,
        ));
        self.selection_host = None;
        self.selection_rows = query.rows;
        self.selection_start = context_rows - query.rows;
        self.selection_width = self.top_k + kpool - 1;
        Ok(())
    }

    fn ensure_hadamard_shadow(&mut self, context: &RocmContext, layer: usize, table: &ops::hip::DeviceBuffer) -> Result<(Arc<ops::hip::DeviceBuffer>, Arc<ops::hip::DeviceBuffer>), BackendError> {
        let cached = self.layers.get(layer).and_then(Option::as_ref).ok_or(BackendError::UnsupportedLayer { layer })?;
        if let (Some(keys), Some(scales)) = (&cached.hadamard_shadow_keys, &cached.hadamard_shadow_scales) {
            return Ok((keys.clone(), scales.clone()));
        }
        if cached.hadamard || cached.gates.is_some() || self.head_dim != 128 || self.key_group_size != self.head_dim {
            return Err(compute_error(format!("L{layer} DSA Hadamard shadow 与当前 cache 约定不兼容")));
        }
        let raw_keys = cached.keys.clone();
        let raw_scales = cached.scales.clone();
        let rows = cached.rows;
        let committed_rows = cached.committed_rows;
        let key_bytes = committed_rows.checked_mul(self.head_dim).ok_or_else(|| compute_error("DSA Hadamard shadow key 大小溢出"))?;
        let scale_bytes = committed_rows.checked_mul(2).ok_or_else(|| compute_error("DSA Hadamard shadow scale 大小溢出"))?;
        let shadow_keys = Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, key_bytes).map_err(compute_error)?);
        let shadow_scales = Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, scale_bytes).map_err(compute_error)?);
        ops::hip::try_paged_cache_transform_q8_hadamard(context.device_id, &raw_keys, &raw_scales, &shadow_keys, &shadow_scales, table, 0, rows, self.head_dim, ROCM_KV_BLOCK_SIZE).map_err(compute_error)?;
        let cached = self.layers.get_mut(layer).and_then(Option::as_mut).ok_or(BackendError::UnsupportedLayer { layer })?;
        cached.hadamard_shadow_keys = Some(shadow_keys.clone());
        cached.hadamard_shadow_scales = Some(shadow_scales.clone());
        Ok((shadow_keys, shadow_scales))
    }

    fn refresh_hisa_pool(&mut self, context: &RocmContext, layer: usize, table: &ops::hip::DeviceBuffer) -> Result<(Arc<ops::hip::DeviceBuffer>, Arc<ops::hip::DeviceBuffer>, Arc<ops::hip::DeviceBuffer>, usize), BackendError> {
        let context_rows = self.layers.get(layer).and_then(Option::as_ref).ok_or(BackendError::UnsupportedLayer { layer })?.rows;
        let pool_rows = context_rows.div_ceil(DSA_HISA_POOL_SIZE);
        let pool_capacity = self.capacity.div_ceil(DSA_HISA_POOL_SIZE);
        let pool_table = self.pool_block_table.get("DSA HISA", pool_capacity, context.device_id, pool_rows)?;
        let groups_per_row = self.head_dim / self.key_group_size;
        let cached = self.layers.get_mut(layer).and_then(Option::as_mut).ok_or(BackendError::UnsupportedLayer { layer })?;
        if cached.pooled_keys.is_none() {
            let key_bytes = pool_capacity.checked_mul(self.head_dim).ok_or_else(|| compute_error("DSA HISA pooled key 容量溢出"))?;
            let scale_bytes = pool_capacity.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("DSA HISA pooled scale 容量溢出"))?;
            cached.pooled_keys = Some(Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, key_bytes).map_err(compute_error)?));
            cached.pooled_scales = Some(Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, scale_bytes).map_err(compute_error)?));
        }
        let first_pool = if cached.pooled_rows == 0 || cached.pooled_rows > pool_rows { 0 } else { cached.pooled_rows.saturating_sub(1).min(pool_rows - 1) };
        let keys = cached.keys.clone();
        let scales = cached.scales.clone();
        let pooled_keys = cached.pooled_keys.as_ref().unwrap().clone();
        let pooled_scales = cached.pooled_scales.as_ref().unwrap().clone();
        ops::hip::try_dsa_mean_pool_q8(
            context.device_id,
            &keys,
            &scales,
            table,
            &pooled_keys,
            &pooled_scales,
            &pool_table,
            first_pool,
            pool_rows - first_pool,
            context_rows,
            self.head_dim,
            self.key_group_size,
            DSA_HISA_POOL_SIZE,
            ROCM_KV_BLOCK_SIZE,
        )
        .map_err(compute_error)?;
        cached.pooled_rows = pool_rows;
        Ok((pooled_keys, pooled_scales, pool_table, pool_rows))
    }

    fn refresh_interval_pool(&mut self, context: &RocmContext, layer: usize, table: &ops::hip::DeviceBuffer) -> Result<(Arc<ops::hip::DeviceBuffer>, Arc<ops::hip::DeviceBuffer>, usize), BackendError> {
        let context_rows = self.layers.get(layer).and_then(Option::as_ref).ok_or(BackendError::UnsupportedLayer { layer })?.rows;
        let pool_rows = context_rows.div_ceil(DSA_HISA_POOL_SIZE);
        let pool_capacity = self.capacity.div_ceil(DSA_HISA_POOL_SIZE);
        let cached = self.layers.get_mut(layer).and_then(Option::as_mut).ok_or(BackendError::UnsupportedLayer { layer })?;
        if cached.interval_lower.is_none() {
            let bytes = pool_capacity.checked_mul(self.head_dim).and_then(|n| n.checked_mul(4)).ok_or_else(|| compute_error("DSA interval summary 容量溢出"))?;
            cached.interval_lower = Some(Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, bytes).map_err(compute_error)?));
            cached.interval_upper = Some(Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, bytes).map_err(compute_error)?));
        }
        let first_pool = if cached.interval_rows == 0 || cached.interval_rows > pool_rows { 0 } else { cached.interval_rows.saturating_sub(1).min(pool_rows - 1) };
        let keys = cached.keys.clone();
        let scales = cached.scales.clone();
        let lower = cached.interval_lower.as_ref().unwrap().clone();
        let upper = cached.interval_upper.as_ref().unwrap().clone();
        ops::hip::try_dsa_interval_pool_q8(context.device_id, &keys, &scales, table, &lower, &upper, first_pool, pool_rows - first_pool, context_rows, self.head_dim, self.key_group_size, DSA_HISA_POOL_SIZE, ROCM_KV_BLOCK_SIZE)
            .map_err(compute_error)?;
        cached.interval_rows = pool_rows;
        Ok((lower, upper, pool_rows))
    }

    fn rebuild_cpu_keys(&mut self, layer: usize, context_rows: usize) -> Result<(), BackendError> {
        let cached = self.layers.get(layer).and_then(Option::as_ref).ok_or(BackendError::UnsupportedLayer { layer })?;
        let groups = self.head_dim / self.key_group_size;
        let key_bytes = context_rows.checked_mul(self.head_dim).ok_or_else(|| compute_error("CPU DSA key 下载大小溢出"))?;
        let scale_elements = context_rows.checked_mul(groups).ok_or_else(|| compute_error("CPU DSA scale 下载大小溢出"))?;
        let mut keys = vec![0_i8; key_bytes];
        let mut scales = vec![0_u16; scale_elements];
        cached.keys.copy_to_host(unsafe { std::slice::from_raw_parts_mut(keys.as_mut_ptr().cast(), keys.len()) }).map_err(compute_error)?;
        cached.scales.copy_to_host(unsafe { std::slice::from_raw_parts_mut(scales.as_mut_ptr().cast(), scales.len() * 2) }).map_err(compute_error)?;
        let blocked = crate::kernel::cpu::dsa::Q8KeyBlocks::from_q8_row_major(&keys, &scales, self.head_dim).map_err(compute_error)?;
        self.layers[layer].as_mut().expect("CPU DSA layer 已校验").cpu_keys = Some(blocked);
        Ok(())
    }

    fn select_cpu_begin(
        &mut self,
        context: &RocmContext,
        layer: usize,
        query_device: Arc<ops::hip::DeviceBuffer>,
        weight_device: Arc<ops::hip::DeviceBuffer>,
        table: Arc<ops::hip::DeviceBuffer>,
        context_rows: usize,
        head_count: usize,
    ) -> Result<(), BackendError> {
        if self.cpu_pending.is_some() {
            return Err(compute_error("CPU DSA begin 前一份 selection 尚未 finish"));
        }
        let total_started = std::time::Instant::now();
        let cpu_rows = self.layers[layer].as_ref().and_then(|cached| cached.cpu_keys.as_ref()).map(crate::kernel::cpu::dsa::Q8KeyBlocks::rows);
        let append_row = match cpu_rows {
            Some(rows) if rows == context_rows => false,
            Some(rows) if rows + 1 == context_rows => true,
            Some(rows) if rows > context_rows => return Err(compute_error(format!("L{layer} CPU DSA rows={rows} 超过 GPU context={context_rows}"))),
            _ => {
                let rebuild_started = std::time::Instant::now();
                self.rebuild_cpu_keys(layer, context_rows)?;
                if ops::hip::options().kernel_profile {
                    eprintln!("[dsa-cpu-select-warm] device={} layer={layer} context={context_rows} rebuild_ms={:.3}", context.device_id, rebuild_started.elapsed().as_secs_f64() * 1e3);
                }
                false
            }
        };

        let query_bytes = head_count.checked_mul(self.head_dim).and_then(|n| n.checked_mul(4)).ok_or_else(|| compute_error("CPU DSA query 下载大小溢出"))?;
        let weight_bytes = head_count.checked_mul(4).ok_or_else(|| compute_error("CPU DSA weight 下载大小溢出"))?;
        let key_bytes = usize::from(append_row) * self.head_dim;
        let scale_bytes = usize::from(append_row) * (self.head_dim / self.key_group_size) * 2;
        let weight_offset = query_bytes;
        let key_offset = weight_offset + weight_bytes;
        let scale_offset = key_offset + key_bytes;
        let transfer_bytes = scale_offset + scale_bytes;
        if self.cpu_transfer.as_ref().is_none_or(|buffer| buffer.device_id() != context.device_id || buffer.bytes() < transfer_bytes) {
            self.cpu_transfer = Some(Arc::new(ops::hip::DeviceBuffer::allocate_reusable(context.device_id, transfer_bytes).map_err(compute_error)?));
        }
        let transfer = self.cpu_transfer.as_ref().expect("CPU DSA transfer 已创建");
        transfer.copy_from_device(0, &query_device, 0, query_bytes).map_err(compute_error)?;
        transfer.copy_from_device(weight_offset, &weight_device, 0, weight_bytes).map_err(compute_error)?;
        if append_row {
            let cached = self.layers[layer].as_ref().expect("CPU DSA layer 已校验");
            transfer.copy_from_device(key_offset, &cached.keys, (context_rows - 1) * self.head_dim, key_bytes).map_err(compute_error)?;
            transfer.copy_from_device(scale_offset, &cached.scales, (context_rows - 1) * (self.head_dim / self.key_group_size) * 2, scale_bytes).map_err(compute_error)?;
        }
        let transfer_started = std::time::Instant::now();
        let mut host_download = match self.cpu_host_download.take() {
            Some(download) => download,
            None => ops::hip::AsyncHostDownload::new(context.device_id, transfer_bytes).map_err(compute_error)?,
        };
        if let Err(error) = host_download.enqueue(transfer, transfer_bytes) {
            self.cpu_host_download = Some(host_download);
            return Err(compute_error(error));
        }

        // profile 下覆盖足够长的真实 decode 前缀，捕获 CPU/GPU score 在后续 query 上的首个 Top-K 分叉。
        let validate = ops::hip::options().kernel_profile && self.cpu_select_counts[layer] < 80;
        let exact = if validate {
            let exact = ops::hip::try_dsa_select_paged_q8(
                context.device_id,
                &self.layers[layer].as_ref().unwrap().keys,
                &self.layers[layer].as_ref().unwrap().scales,
                self.key_group_size,
                false,
                &table,
                &query_device,
                &weight_device,
                1,
                context_rows,
                context_rows - 1,
                head_count,
                self.head_dim,
                self.top_k,
                false,
                ROCM_KV_BLOCK_SIZE,
            )
            .map_err(compute_error)?;
            let mut values = vec![0_u32; self.top_k];
            exact.copy_to_host(unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len() * 4) }).map_err(compute_error)?;
            Some(values)
        } else {
            None
        };

        if self.cpu_dispatcher.is_none() {
            self.cpu_dispatcher = Some(CpuDsaDispatcher::new().map_err(compute_error)?);
        }
        let task = CpuDsaTask {
            keys: self.layers[layer].as_mut().unwrap().cpu_keys.take().expect("CPU DSA keys 已创建"),
            workspace: std::mem::take(&mut self.cpu_workspace),
            host_download,
            transfer_started,
            weight_offset,
            key_offset,
            scale_offset,
            append_row,
            head_count,
            head_dim: self.head_dim,
            requested_candidates: (self.top_k + DSA_CPU_CANDIDATE_GUARD).min(context_rows),
        };
        let dispatched = std::time::Instant::now();
        if let Err(error) = self.cpu_dispatcher.as_ref().unwrap().tasks.send(task) {
            let task = error.0;
            self.layers[layer].as_mut().unwrap().cpu_keys = Some(task.keys);
            self.cpu_workspace = task.workspace;
            self.cpu_host_download = Some(task.host_download);
            return Err(compute_error(format!("L{layer} CPU DSA dispatcher 已退出")));
        }
        let cached = self.layers[layer].as_ref().unwrap();
        self.cpu_pending = Some(CpuDsaPending {
            layer,
            context_rows,
            head_count,
            device_id: context.device_id,
            transfer_bytes,
            begin_ms: total_started.elapsed().as_secs_f64() * 1e3,
            started: total_started,
            dispatched,
            keys: cached.keys.clone(),
            scales: cached.scales.clone(),
            table,
            query: query_device,
            weights: weight_device,
            exact,
        });
        Ok(())
    }

    pub(super) fn select_finish(&mut self, context: &RocmContext) -> Result<(), BackendError> {
        let Some(pending) = self.cpu_pending.take() else { return Ok(()) };
        if context.device_id != pending.device_id {
            return Err(compute_error(format!("CPU DSA finish device={}，begin device={}", context.device_id, pending.device_id)));
        }
        let overlap_ms = pending.dispatched.elapsed().as_secs_f64() * 1e3;
        let wait_started = std::time::Instant::now();
        let worker = self.cpu_dispatcher.as_ref().expect("CPU DSA pending 必有 dispatcher").results.recv().map_err(|_| compute_error(format!("L{} CPU DSA dispatcher 已退出", pending.layer)))?;
        let wait_ms = wait_started.elapsed().as_secs_f64() * 1e3;
        self.layers[pending.layer].as_mut().expect("CPU DSA pending layer 已校验").cpu_keys = Some(worker.keys);
        self.cpu_workspace = worker.workspace;
        self.cpu_host_download = Some(worker.host_download);
        let transfer_ms = worker.transfer_ms;
        let (candidates, score_ms, topk_ms, select_ms) = worker.candidates.map_err(compute_error)?;
        let candidate_count = candidates.len();
        let candidate_bytes = unsafe { std::slice::from_raw_parts(candidates.as_ptr().cast(), std::mem::size_of_val(candidates.as_slice())) };
        let candidate_buffer = Arc::new(ops::hip::DeviceBuffer::upload_ordered(context.device_id, candidate_bytes).map_err(compute_error)?);
        candidate_buffer.enqueue_deferred_upload().map_err(compute_error)?;
        candidate_buffer.retain_for_active_stage();
        let selection = ops::hip::try_dsa_rerank_paged_q8_candidates(
            context.device_id,
            &pending.keys,
            &pending.scales,
            self.key_group_size,
            &pending.table,
            &pending.query,
            &pending.weights,
            &candidate_buffer,
            1,
            pending.context_rows,
            pending.context_rows - 1,
            pending.head_count,
            self.head_dim,
            candidate_count,
            self.top_k,
            ROCM_KV_BLOCK_SIZE,
        )
        .map_err(compute_error)?;
        let need_host = ops::hip::options().mla_cpu_hot_rows != 0;
        let mut selection_host = (pending.exact.is_some() || need_host).then(|| vec![0_u32; self.top_k]);
        if let Some(host) = &mut selection_host {
            selection.copy_to_host(unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast(), host.len() * 4) }).map_err(compute_error)?;
        }
        if let Some(exact) = pending.exact {
            let recall = exact.iter().filter(|token| candidates.binary_search(token).is_ok()).count();
            let reranked = selection_host.as_ref().expect("oracle selection 已下载");
            let rerank_exact = reranked.as_slice() == exact.as_slice();
            let sample = self.cpu_select_counts[pending.layer];
            if sample == 0 || recall != self.top_k || !rerank_exact {
                eprintln!("[dsa-cpu-select-oracle] device={} layer={} sample={sample} context={} candidate_recall={recall}/{} rerank_exact={rerank_exact}", context.device_id, pending.layer, pending.context_rows, self.top_k);
            }
        }
        self.selection = Some(Arc::new(selection));
        self.selection_host = selection_host.map(Arc::new);
        self.selection_rows = 1;
        self.selection_start = pending.context_rows - 1;
        self.selection_width = self.top_k;
        let sample = self.cpu_select_counts[pending.layer];
        self.cpu_select_counts[pending.layer] += 1;
        if ops::hip::options().kernel_profile && (sample < 2 || sample.is_multiple_of(128)) {
            eprintln!(
                "[dsa-cpu-select] device={} layer={} sample={sample} context={} candidates={candidate_count} d2h_bytes={} d2h_ms={:.3} begin_ms={:.3} overlap_ms={overlap_ms:.3} wait_ms={wait_ms:.3} score_ms={score_ms:.3} topk_ms={topk_ms:.3} select_ms={select_ms:.3} total_ms={:.3} selection=cpu_async_guarded_rerank",
                context.device_id,
                pending.layer,
                pending.context_rows,
                pending.transfer_bytes,
                transfer_ms,
                pending.begin_ms,
                pending.started.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(())
    }

    pub(super) fn select_begin(&mut self, context: &RocmContext, layer: usize, query: &RocmTensor, head_weights: &RocmTensor) -> Result<(), BackendError> {
        if self.cpu_pending.is_some() {
            return Err(compute_error("ROCm DSA begin 前一份 CPU selection 尚未 finish"));
        }
        let context_rows = self.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} ROCm DSA cache 尚未初始化")))?.rows;
        if context_rows <= self.top_k {
            self.selection = None;
            self.selection_host = None;
            self.selection_rows = 0;
            self.selection_start = 0;
            self.selection_width = self.top_k;
            return Ok(());
        }
        if query.rows == 0 || query.rows > context_rows || query.cols % self.head_dim != 0 {
            return Err(compute_error(format!("L{layer} ROCm DSA query shape 非法: {:?} context={context_rows} head_dim={}", (query.rows, query.cols), self.head_dim,)));
        }
        let head_count = query.cols / self.head_dim;
        if head_weights.rows != query.rows || head_weights.cols != head_count {
            return Err(compute_error(format!("L{layer} ROCm DSA head weight shape {:?}，期望 ({},{head_count})", (head_weights.rows, head_weights.cols), query.rows,)));
        }
        let query_device = query.device.as_deref().ok_or_else(|| compute_error("ROCm DSA query 缺少 device buffer"))?;
        let weight_device = head_weights.device.as_deref().ok_or_else(|| compute_error("ROCm DSA head weights 缺少 device buffer"))?;
        let keys = self.layers[layer].as_ref().unwrap().keys.clone();
        let scales = self.layers[layer].as_ref().unwrap().scales.clone();
        let hadamard = self.layers[layer].as_ref().unwrap().hadamard;
        let table = self.block_table.get("DSA", self.capacity, context.device_id, context_rows)?;
        if self.cpu_select && context_rows > self.top_k + DSA_CPU_CANDIDATE_GUARD && !hadamard && self.kpool == 0 && self.decode_parallelism == 1 && query.rows == 1 && head_count == 32 && self.head_dim == 128 {
            return self.select_cpu_begin(context, layer, query.device.as_ref().unwrap().clone(), head_weights.device.as_ref().unwrap().clone(), table, context_rows, head_count);
        }
        let exact = ops::hip::try_dsa_select_paged_q8(
            context.device_id,
            &keys,
            &scales,
            self.key_group_size,
            hadamard,
            &table,
            query_device,
            weight_device,
            query.rows,
            context_rows,
            context_rows - query.rows,
            head_count,
            self.head_dim,
            self.top_k,
            self.decode_parallelism >= 3,
            ROCM_KV_BLOCK_SIZE,
        )
        .map_err(compute_error)?;
        let sample_shadow = !hadamard && self.kpool == 0 && query.rows == 1 && self.hadamard_shadow_counts.get(layer).copied().unwrap_or_default() < self.hadamard_shadow_samples;
        if sample_shadow {
            // 必须在 coarse 调用覆盖同一 score workspace 前读取 exact 分数。
            let exact_scores = ops::hip::try_download_last_dsa_score_keys(context.device_id, context_rows).map_err(compute_error)?;
            let mut exact_selection = vec![0_u32; self.top_k];
            exact.copy_to_host(unsafe { std::slice::from_raw_parts_mut(exact_selection.as_mut_ptr().cast(), self.top_k * std::mem::size_of::<u32>()) }).map_err(compute_error)?;
            let cpu_shadow_started = std::time::Instant::now();
            let mut key_codes = vec![0_i8; context_rows * self.head_dim];
            keys.copy_to_host(unsafe { std::slice::from_raw_parts_mut(key_codes.as_mut_ptr().cast(), key_codes.len()) }).map_err(compute_error)?;
            let groups_per_row = self.head_dim / self.key_group_size;
            let mut scale_bits = vec![0_u16; context_rows * groups_per_row];
            scales.copy_to_host(unsafe { std::slice::from_raw_parts_mut(scale_bits.as_mut_ptr().cast(), scale_bits.len() * 2) }).map_err(compute_error)?;
            let mut query_host = vec![0.0_f32; head_count * self.head_dim];
            query_device.copy_to_host(unsafe { std::slice::from_raw_parts_mut(query_host.as_mut_ptr().cast(), query_host.len() * 4) }).map_err(compute_error)?;
            let mut weight_host = vec![0.0_f32; head_count];
            weight_device.copy_to_host(unsafe { std::slice::from_raw_parts_mut(weight_host.as_mut_ptr().cast(), weight_host.len() * 4) }).map_err(compute_error)?;
            let mut cpu_keys = Vec::with_capacity(key_codes.len());
            for row in 0..context_rows {
                for column in 0..self.head_dim {
                    let scale = half::bf16::from_bits(scale_bits[row * groups_per_row + column / self.key_group_size]).to_f32();
                    cpu_keys.push(half::bf16::from_f32(key_codes[row * self.head_dim + column] as f32 * scale));
                }
            }
            let blocked_started = std::time::Instant::now();
            let cpu_keys = crate::kernel::cpu::dsa::Bf16KeyBlocks::from_row_major(&cpu_keys, self.head_dim).map_err(compute_error)?;
            let blocked_ms = blocked_started.elapsed().as_secs_f64() * 1e3;
            let mut selection_ms = Vec::with_capacity(DSA_CPU_SHADOW_REPEATS);
            let mut score_ms = Vec::with_capacity(DSA_CPU_SHADOW_REPEATS);
            let mut topk_ms = Vec::with_capacity(DSA_CPU_SHADOW_REPEATS);
            let mut cpu_result = None;
            let mut first_selection = None;
            for repeat in 0..DSA_CPU_SHADOW_REPEATS {
                let selection_started = std::time::Instant::now();
                let result = crate::kernel::cpu::dsa::select_bf16_blocked_profile(&cpu_keys, &query_host, &weight_host, head_count, self.head_dim, self.top_k).map_err(compute_error)?;
                selection_ms.push(selection_started.elapsed().as_secs_f64() * 1e3);
                score_ms.push(result.2);
                topk_ms.push(result.3);
                if repeat == 0 {
                    first_selection = Some(result.1.clone());
                }
                cpu_result = Some(result);
            }
            let (cpu_scores, cpu_selection, _, _) = cpu_result.expect("CPU DSA shadow 至少执行一次");
            if first_selection.as_deref() != Some(cpu_selection.as_slice()) {
                return Err(compute_error("CPU DSA shadow 重复执行结果不确定"));
            }
            let mut warm_ms = selection_ms[1..].to_vec();
            warm_ms.sort_by(f64::total_cmp);
            let warm_min_ms = warm_ms[0];
            let warm_p50_ms = warm_ms[warm_ms.len() / 2];
            let warm_p95_ms = warm_ms[(warm_ms.len() * 95).div_ceil(100) - 1];
            let mut warm_score_ms = score_ms[1..].to_vec();
            warm_score_ms.sort_by(f64::total_cmp);
            let warm_score_p50_ms = warm_score_ms[warm_score_ms.len() / 2];
            let mut warm_topk_ms = topk_ms[1..].to_vec();
            warm_topk_ms.sort_by(f64::total_cmp);
            let warm_topk_p50_ms = warm_topk_ms[warm_topk_ms.len() / 2];
            let mut exact_members = exact_selection.clone();
            let mut cpu_members = cpu_selection.clone();
            exact_members.sort_unstable();
            cpu_members.sort_unstable();
            let overlap = cpu_members.iter().filter(|token| exact_members.binary_search(token).is_ok()).count();
            let max_score_abs = exact_scores.iter().zip(&cpu_scores).map(|(&gpu, &cpu)| (score_from_ordered(gpu) - score_from_ordered(cpu)).abs()).fold(0.0_f32, f32::max);
            eprintln!(
                "[dsa-cpu-bf16-shadow] device={} layer={layer} context={context_rows} overlap={overlap}/{} exact={} max_score_abs={max_score_abs:.9} repeat={} blocked_ms={blocked_ms:.3} cold_ms={:.3} warm_min_ms={warm_min_ms:.3} warm_p50_ms={warm_p50_ms:.3} warm_p95_ms={warm_p95_ms:.3} warm_score_p50_ms={warm_score_p50_ms:.3} warm_topk_p50_ms={warm_topk_p50_ms:.3} shadow_wall_ms={:.3} selection=gpu_exact",
                context.device_id,
                self.top_k,
                cpu_selection == exact_selection,
                DSA_CPU_SHADOW_REPEATS,
                selection_ms[0],
                cpu_shadow_started.elapsed().as_secs_f64() * 1e3,
            );
            let (shadow_keys, shadow_scales) = self.ensure_hadamard_shadow(context, layer, &table)?;
            let candidate_count = (self.top_k + 1024).min(context_rows - 1);
            let coarse_selection = ops::hip::try_dsa_select_paged_q8(
                context.device_id,
                &shadow_keys,
                &shadow_scales,
                self.head_dim,
                true,
                &table,
                query_device,
                weight_device,
                1,
                context_rows,
                context_rows - 1,
                head_count,
                self.head_dim,
                candidate_count,
                false,
                ROCM_KV_BLOCK_SIZE,
            )
            .map_err(compute_error)?;
            let coarse_scores = ops::hip::try_download_last_dsa_score_keys(context.device_id, context_rows).map_err(compute_error)?;
            let reranked = ops::hip::try_dsa_rerank_paged_q8_candidates(
                context.device_id,
                &keys,
                &scales,
                self.key_group_size,
                &table,
                query_device,
                weight_device,
                &coarse_selection,
                1,
                context_rows,
                context_rows - 1,
                head_count,
                self.head_dim,
                candidate_count,
                self.top_k,
                ROCM_KV_BLOCK_SIZE,
            )
            .map_err(compute_error)?;
            let mut reranked_selection = vec![0_u32; self.top_k];
            reranked.copy_to_host(unsafe { std::slice::from_raw_parts_mut(reranked_selection.as_mut_ptr().cast(), self.top_k * std::mem::size_of::<u32>()) }).map_err(compute_error)?;
            let mut exact_members = exact_selection.clone();
            let mut reranked_members = reranked_selection.clone();
            exact_members.sort_unstable();
            reranked_members.sort_unstable();
            let rerank_overlap = reranked_members.iter().filter(|token| exact_members.binary_search(token).is_ok()).count();
            let rerank_exact = reranked_selection == exact_selection;
            let metrics = analyze_hadamard_shadow(&exact_selection, &exact_scores, &coarse_scores, candidate_count).map_err(compute_error)?;
            let sample = self.hadamard_shadow_counts[layer];
            self.hadamard_shadow_counts[layer] += 1;
            let recalls = DSA_SHADOW_GUARDS.iter().zip(metrics.recalls).map(|(guard, hits)| format!("{guard}:{hits}/{}", self.top_k)).collect::<Vec<_>>().join(",");
            eprintln!(
                "[dsa-hadamard-shadow] device={} layer={layer} sample={sample} context={context_rows} top_k={} candidates={candidate_count} prefix_candidates={} prefix_recall={}/{} prefix16_candidates={} prefix16_recall={}/{} required_guard={} exact_margin={:.9} coarse_margin={:.9} rerank_overlap={rerank_overlap}/{} rerank_exact={rerank_exact} recalls=[{recalls}] selection=exact",
                context.device_id,
                self.top_k,
                metrics.prefix_candidate_count,
                metrics.prefix_recall,
                self.top_k,
                metrics.prefix16_candidate_count,
                metrics.prefix16_recall,
                self.top_k,
                metrics.required_guard,
                metrics.exact_margin,
                metrics.coarse_margin,
                self.top_k,
            );
        }
        let pool_rows = context_rows.div_ceil(DSA_HISA_POOL_SIZE);
        let sample_hisa = !hadamard && self.kpool == 0 && query.rows == 1 && pool_rows > *DSA_HISA_BLOCK_BUDGETS.last().unwrap() && self.hisa_shadow_counts.get(layer).copied().unwrap_or_default() < self.hisa_shadow_samples;
        if sample_hisa {
            let mut exact_selection = vec![0_u32; self.top_k];
            exact.copy_to_host(unsafe { std::slice::from_raw_parts_mut(exact_selection.as_mut_ptr().cast(), self.top_k * std::mem::size_of::<u32>()) }).map_err(compute_error)?;
            let exact_scores = ops::hip::try_download_last_dsa_score_keys(context.device_id, context_rows).map_err(compute_error)?;
            let interval_started = std::time::Instant::now();
            let (interval_lower, interval_upper, interval_rows) = self.refresh_interval_pool(context, layer, &table)?;
            let interval_bounds = ops::hip::DeviceBuffer::allocate(context.device_id, interval_rows.checked_mul(4).ok_or_else(|| compute_error("DSA interval bounds 大小溢出"))?).map_err(compute_error)?;
            ops::hip::try_dsa_interval_score_bounds(context.device_id, &interval_lower, &interval_upper, query_device, weight_device, &interval_bounds, interval_rows, head_count, self.head_dim).map_err(compute_error)?;
            let mut bounds = vec![0.0_f32; interval_rows];
            interval_bounds.copy_to_host(unsafe { std::slice::from_raw_parts_mut(bounds.as_mut_ptr().cast(), interval_rows * 4) }).map_err(compute_error)?;
            let metrics = analyze_interval_bounds(&exact_selection, &exact_scores, &bounds, DSA_HISA_POOL_SIZE).map_err(compute_error)?;
            let mut weights = vec![0.0_f32; head_count];
            weight_device.copy_to_host(unsafe { std::slice::from_raw_parts_mut(weights.as_mut_ptr().cast(), head_count * 4) }).map_err(compute_error)?;
            let negative_weights = weights.iter().filter(|&&weight| weight < 0.0).count();
            eprintln!(
                "[dsa-interval-shadow] device={} layer={layer} sample={} context={context_rows} pools={interval_rows} block_size={DSA_HISA_POOL_SIZE} certified_blocks={}/{} certified_tokens={} pruned_tokens={} exact_recall={}/{} violations={} max_excess={:.9} negative_weights={negative_weights}/{head_count} shadow_wall_ms={:.3} selection=exact",
                context.device_id,
                self.hisa_shadow_counts[layer],
                metrics.candidate_blocks,
                interval_rows,
                metrics.candidate_tokens,
                context_rows - metrics.candidate_tokens,
                metrics.exact_recall,
                self.top_k,
                metrics.violations,
                metrics.max_excess,
                interval_started.elapsed().as_secs_f64() * 1e3,
            );
            let started = std::time::Instant::now();
            let (pooled_keys, pooled_scales, pool_table, pool_rows) = self.refresh_hisa_pool(context, layer, &table)?;
            let ranked_count = *DSA_HISA_BLOCK_BUDGETS.last().unwrap();
            let ranked = ops::hip::try_dsa_select_paged_q8(
                context.device_id,
                &pooled_keys,
                &pooled_scales,
                self.key_group_size,
                false,
                &pool_table,
                query_device,
                weight_device,
                1,
                pool_rows,
                pool_rows - 1,
                head_count,
                self.head_dim,
                ranked_count,
                false,
                ROCM_KV_BLOCK_SIZE,
            )
            .map_err(compute_error)?;
            let mut ranked_blocks = vec![0_u32; ranked_count];
            ranked.copy_to_host(unsafe { std::slice::from_raw_parts_mut(ranked_blocks.as_mut_ptr().cast(), ranked_count * std::mem::size_of::<u32>()) }).map_err(compute_error)?;
            let mut recall_parts = Vec::with_capacity(DSA_HISA_BLOCK_BUDGETS.len());
            for budget in DSA_HISA_BLOCK_BUDGETS {
                let blocks = hisa_candidate_blocks(&ranked_blocks, budget, pool_rows).map_err(compute_error)?;
                let hits = exact_selection.iter().filter(|&&token| blocks.binary_search(&(token / DSA_HISA_POOL_SIZE as u32)).is_ok()).count();
                recall_parts.push(format!("{budget}:{hits}/{}", self.top_k));
            }
            let candidate_blocks = hisa_candidate_blocks(&ranked_blocks, 64, pool_rows).map_err(compute_error)?;
            let candidate_tokens = hisa_candidate_tokens(&candidate_blocks, context_rows).map_err(compute_error)?;
            let candidate_count = candidate_tokens.len();
            let candidate_bytes = unsafe { std::slice::from_raw_parts(candidate_tokens.as_ptr().cast(), std::mem::size_of_val(candidate_tokens.as_slice())) };
            let candidates = ops::hip::DeviceBuffer::upload_independent(context.device_id, candidate_bytes).map_err(compute_error)?;
            let reranked = ops::hip::try_dsa_rerank_paged_q8_candidates(
                context.device_id,
                &keys,
                &scales,
                self.key_group_size,
                &table,
                query_device,
                weight_device,
                &candidates,
                1,
                context_rows,
                context_rows - 1,
                head_count,
                self.head_dim,
                candidate_count,
                self.top_k,
                ROCM_KV_BLOCK_SIZE,
            )
            .map_err(compute_error)?;
            let mut reranked_selection = vec![0_u32; self.top_k];
            reranked.copy_to_host(unsafe { std::slice::from_raw_parts_mut(reranked_selection.as_mut_ptr().cast(), self.top_k * std::mem::size_of::<u32>()) }).map_err(compute_error)?;
            let mut exact_members = exact_selection.clone();
            let mut reranked_members = reranked_selection.clone();
            exact_members.sort_unstable();
            reranked_members.sort_unstable();
            let rerank_overlap = reranked_members.iter().filter(|token| exact_members.binary_search(token).is_ok()).count();
            let rerank_exact = reranked_selection == exact_selection;
            let sample = self.hisa_shadow_counts[layer];
            self.hisa_shadow_counts[layer] += 1;
            eprintln!(
                "[dsa-hisa-shadow] device={} layer={layer} sample={sample} context={context_rows} pools={pool_rows} block_size={DSA_HISA_POOL_SIZE} candidates={candidate_count} rerank_overlap={rerank_overlap}/{} rerank_exact={rerank_exact} recalls=[{}] shadow_wall_ms={:.3} selection=exact",
                context.device_id,
                self.top_k,
                recall_parts.join(","),
                started.elapsed().as_secs_f64() * 1e3,
            );
        }
        let selection_host = if ops::hip::options().mla_cpu_hot_rows != 0 && query.rows == 1 {
            let mut host = vec![0_u32; self.top_k];
            exact.copy_to_host(unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast(), host.len() * 4) }).map_err(compute_error)?;
            Some(Arc::new(host))
        } else {
            None
        };
        self.selection = Some(Arc::new(exact));
        self.selection_host = selection_host;
        self.selection_rows = query.rows;
        self.selection_start = context_rows - query.rows;
        self.selection_width = self.top_k;
        Ok(())
    }

    pub(super) fn select(&mut self, context: &RocmContext, layer: usize, query: &RocmTensor, head_weights: &RocmTensor) -> Result<(), BackendError> {
        self.select_begin(context, layer, query, head_weights)?;
        self.select_finish(context)
    }

    pub(super) fn device_selection(&self, rows: usize, query_start: usize) -> Option<&ops::hip::DeviceBuffer> {
        (self.selection_rows == rows && self.selection_start == query_start).then_some(self.selection.as_deref()).flatten()
    }

    pub(super) fn selection_width(&self) -> usize {
        self.selection_width
    }

    pub(super) fn host_selection(&self, rows: usize, query_start: usize) -> Option<&[u32]> {
        (self.selection_rows == rows && self.selection_start == query_start).then_some(self.selection_host.as_deref()).flatten().map(Vec::as_slice)
    }

    pub(crate) fn export_selection(&self) -> Option<RocmDsaSelection> {
        Some(RocmDsaSelection { buffer: self.selection.as_ref()?.clone(), host: self.selection_host.clone(), rows: self.selection_rows, start: self.selection_start, width: self.selection_width })
    }

    pub(crate) fn import_selection(&mut self, context: &RocmContext, selection: Option<RocmDsaSelection>) -> Result<(), BackendError> {
        let Some(selection) = selection else {
            self.selection = None;
            self.selection_host = None;
            self.selection_rows = 0;
            self.selection_start = 0;
            self.selection_width = self.top_k;
            return Ok(());
        };
        let expected = selection.rows.checked_mul(selection.width).and_then(|n| n.checked_mul(std::mem::size_of::<u32>())).ok_or_else(|| compute_error("ROCm DSA selection P2P 大小溢出"))?;
        if selection.rows == 0 || selection.buffer.bytes() < expected {
            return Err(compute_error(format!("ROCm DSA selection P2P shape 非法: rows={} bytes={} expected={expected}", selection.rows, selection.buffer.bytes())));
        }
        self.selection = Some(if selection.buffer.device_id() == context.device_id { selection.buffer } else { Arc::new(selection.buffer.copy_to_device(context.device_id).map_err(compute_error)?) });
        self.selection_host = selection.host;
        self.selection_rows = selection.rows;
        self.selection_start = selection.start;
        self.selection_width = selection.width;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordered(score: f32) -> u32 {
        let bits = score.to_bits();
        bits ^ if bits & 0x8000_0000 != 0 { 0xffff_ffff } else { 0x8000_0000 }
    }

    #[test]
    fn hadamard_shadow_reports_guard_for_exact_member() {
        let exact = [0, 1];
        let exact_scores = [ordered(4.0), ordered(3.0), ordered(2.0), ordered(1.0)];
        let coarse_scores = [ordered(4.0), ordered(2.0), ordered(3.0), ordered(1.0)];
        let metrics = analyze_hadamard_shadow(&exact, &exact_scores, &coarse_scores, 3).unwrap();
        assert_eq!(metrics.required_guard, 1);
        assert_eq!(metrics.recalls[0], 1);
        assert_eq!(metrics.recalls[1], 2);
        assert_eq!(metrics.prefix16_recall, 2);
        assert_eq!(metrics.exact_margin, 1.0);
    }

    #[test]
    fn hadamard_shadow_uses_stable_token_tie_order() {
        let exact = [0, 2];
        let exact_scores = [ordered(4.0), ordered(2.0), ordered(3.0), ordered(1.0)];
        let coarse_scores = [ordered(1.0); 4];
        let metrics = analyze_hadamard_shadow(&exact, &exact_scores, &coarse_scores, 3).unwrap();
        assert_eq!(metrics.required_guard, 1);
        assert_eq!(metrics.recalls[0], 1);
        assert_eq!(metrics.prefix16_candidate_count, 4);
    }

    #[test]
    fn hisa_candidate_blocks_force_sink_and_local_without_duplicates() {
        let ranked = [7, 6, 5, 4, 3, 2, 1, 8];
        assert_eq!(hisa_candidate_blocks(&ranked, 4, 10).unwrap(), vec![0, 6, 7, 9]);
        let ranked = [0, 9, 7, 6, 5, 4, 3, 2];
        assert_eq!(hisa_candidate_blocks(&ranked, 4, 10).unwrap(), vec![0, 6, 7, 9]);
    }

    #[test]
    fn hisa_candidate_tokens_keep_token_order_and_partial_tail() {
        let tokens = hisa_candidate_tokens(&[0, 2], 2 * DSA_HISA_POOL_SIZE + 7).unwrap();
        assert_eq!(tokens.len(), DSA_HISA_POOL_SIZE + 7);
        assert_eq!(&tokens[..3], &[0, 1, 2]);
        assert_eq!(&tokens[tokens.len() - 3..], &[260, 261, 262]);
    }

    #[test]
    fn interval_metrics_use_exact_threshold_and_reject_inward_bound() {
        let scores = [9.0, 8.0, 7.0, 6.0, 4.0, 3.0, 2.0, 1.0].map(ordered);
        let exact = [0, 1];
        let metrics = analyze_interval_bounds(&exact, &scores, &[10.0, 5.0], 4).unwrap();
        assert_eq!(metrics.candidate_blocks, 1);
        assert_eq!(metrics.candidate_tokens, 4);
        assert_eq!(metrics.exact_recall, 2);
        assert_eq!(metrics.violations, 0);
        let metrics = analyze_interval_bounds(&exact, &scores, &[8.5, 5.0], 4).unwrap();
        assert_eq!(metrics.violations, 1);
        assert!((metrics.max_excess - 0.5).abs() < f32::EPSILON);
    }
}

impl std::ops::Deref for RocmDsaState {
    type Target = CpuDsaState;
    fn deref(&self) -> &Self::Target {
        &self.cpu
    }
}

impl std::ops::DerefMut for RocmDsaState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cpu
    }
}

/// DSA 层的 host 序列化形态：纯字节，无 device 句柄。
#[derive(Clone)]
pub struct DsaLayerSerde {
    pub rows: usize,
    pub key_group_size: usize,
    /// true 表示 key 已按归一化 Hadamard 旋转，query 必须使用相同约定。
    pub hadamard: bool,
    /// Q8 keys：rows × head_dim。
    pub keys: Vec<u8>,
    /// BF16 scales：rows × (head_dim / key_group_size) × 2。
    pub scales: Vec<u8>,
}

impl RocmDsaState {
    /// 把所有已填充层 D2H。空层对应 None。
    pub fn download_layers(&self) -> Result<Vec<Option<DsaLayerSerde>>, BackendError> {
        let head_dim = self.head_dim;
        let key_group_size = self.key_group_size;
        let groups_per_row = head_dim / key_group_size;
        self.layers
            .iter()
            .map(|slot| -> Result<Option<DsaLayerSerde>, BackendError> {
                let Some(cached) = slot.as_ref() else { return Ok(None) };
                let key_bytes = cached.rows.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm DSA Q8 keys 字节溢出"))?;
                let scale_bytes = cached.rows.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm DSA Q8 scales 字节溢出"))?;
                let mut keys = vec![0u8; key_bytes];
                let mut scales = vec![0u8; scale_bytes];
                cached.keys.copy_to_host(&mut keys).map_err(compute_error)?;
                cached.scales.copy_to_host(&mut scales).map_err(compute_error)?;
                Ok(Some(DsaLayerSerde { rows: cached.rows, key_group_size, hadamard: cached.hadamard, keys, scales }))
            })
            .collect()
    }

    /// 从 host blob H2D 重建每层。空 slot 跳过。
    pub fn upload_layers(&mut self, context: &RocmContext, layers: &[Option<DsaLayerSerde>], reserved_rows: usize) -> Result<(), BackendError> {
        let device_id = context.device_id;
        if layers.len() > self.layers.len() || reserved_rows > self.capacity {
            return Err(compute_error(format!("ROCm DSA upload layers={}/{} reserved_rows={}/{} 非法", layers.len(), self.layers.len(), reserved_rows, self.capacity)));
        }
        let groups_per_row = self.head_dim / self.key_group_size;
        for (index, slot) in layers.iter().enumerate() {
            let Some(record) = slot else { continue };
            if record.rows == 0 || record.rows > self.capacity || record.key_group_size != self.key_group_size || record.hadamard != self.hadamard_i8 {
                return Err(compute_error(format!(
                    "ROCm DSA restore L{index} rows={} group={} hadamard={}，当前 capacity={} group={} hadamard={} 不兼容",
                    record.rows, record.key_group_size, record.hadamard, self.capacity, self.key_group_size, self.hadamard_i8
                )));
            }
            let key_bytes = record.rows.checked_mul(self.head_dim).ok_or_else(|| compute_error("ROCm DSA restore Q8 keys 大小溢出"))?;
            let scale_bytes = record.rows.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm DSA restore Q8 scales 大小溢出"))?;
            if record.keys.len() != key_bytes || record.scales.len() != scale_bytes {
                return Err(compute_error(format!("ROCm DSA restore L{index} keys/scales={}/{}，期望 {key_bytes}/{scale_bytes}", record.keys.len(), record.scales.len())));
            }
            let committed_rows = reserved_rows.max(record.rows);
            let key_capacity = committed_rows.checked_mul(self.head_dim).ok_or_else(|| compute_error("ROCm DSA restore Q8 key 容量溢出"))?;
            let scale_capacity = committed_rows.checked_mul(groups_per_row).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm DSA restore scale 容量溢出"))?;
            let keys = upload_cache_buffer(device_id, &record.keys, key_capacity)?;
            let scales = upload_cache_buffer(device_id, &record.scales, scale_capacity)?;
            self.layers[index] = Some(RocmPagedDsaLayer {
                keys,
                scales,
                cpu_keys: None,
                hadamard_shadow_keys: None,
                hadamard_shadow_scales: None,
                hadamard: record.hadamard,
                gates: None,
                pooled_keys: None,
                pooled_scales: None,
                pooled_rows: 0,
                interval_lower: None,
                interval_upper: None,
                interval_rows: 0,
                rows: record.rows,
                committed_rows,
            });
        }
        Ok(())
    }

    /// 所有层 device buffer 实际分配字节。
    pub fn allocated_bytes(&self) -> u64 {
        self.layers
            .iter()
            .filter_map(|slot| slot.as_ref())
            .map(|cached| {
                cached.keys.bytes() as u64
                    + cached.scales.bytes() as u64
                    + cached.hadamard_shadow_keys.as_ref().map_or(0, |buffer| buffer.bytes() as u64)
                    + cached.hadamard_shadow_scales.as_ref().map_or(0, |buffer| buffer.bytes() as u64)
                    + cached.gates.as_ref().map_or(0, |buffer| buffer.bytes() as u64)
                    + cached.pooled_keys.as_ref().map_or(0, |buffer| buffer.bytes() as u64)
                    + cached.pooled_scales.as_ref().map_or(0, |buffer| buffer.bytes() as u64)
                    + cached.interval_lower.as_ref().map_or(0, |buffer| buffer.bytes() as u64)
                    + cached.interval_upper.as_ref().map_or(0, |buffer| buffer.bytes() as u64)
            })
            .sum()
    }
}
