use std::sync::Arc;

use rayon::prelude::*;

use crate::backend::cpu::CpuDsaState;
use crate::backend::{Backend, BackendError, compute_error};
use crate::kernel::rocm as ops;

use super::{ROCM_KV_BLOCK_SIZE, RocmBlockTable, RocmContext, RocmTensor, RocmWeight, committed_cache_rows, grow_cache_buffer, upload_cache_buffer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RocmDsaOwnership {
    Full,
    BlockParity,
}

fn parity_rows(rows: usize, parity: usize) -> usize {
    let blocks = rows / ROCM_KV_BLOCK_SIZE;
    let tail = rows % ROCM_KV_BLOCK_SIZE;
    (blocks / 2) * ROCM_KV_BLOCK_SIZE + usize::from(blocks % 2 > parity) * ROCM_KV_BLOCK_SIZE + usize::from(blocks % 2 == parity) * tail
}

fn parity_row(logical: usize) -> usize {
    let block = logical / ROCM_KV_BLOCK_SIZE;
    (block / 2) * ROCM_KV_BLOCK_SIZE + logical % ROCM_KV_BLOCK_SIZE
}

fn parity_segments(start: usize, rows: usize, parity: usize) -> Vec<(usize, usize, usize)> {
    let end = start + rows;
    let mut logical = start;
    let mut segments = Vec::new();
    while logical < end {
        let block = logical / ROCM_KV_BLOCK_SIZE;
        let block_end = ((block + 1) * ROCM_KV_BLOCK_SIZE).min(end);
        if block % 2 == parity {
            segments.push((logical - start, parity_row(logical), block_end - logical));
        }
        logical = block_end;
    }
    segments
}

fn split_parity_bytes(bytes: &[u8], rows: usize, row_bytes: usize, parity: usize) -> Result<Vec<u8>, BackendError> {
    let expected = rows.checked_mul(row_bytes).ok_or_else(|| compute_error("DSA parity split 大小溢出"))?;
    if bytes.len() != expected {
        return Err(compute_error(format!("DSA parity split bytes={}，期望 {expected}", bytes.len())));
    }
    let mut output = Vec::with_capacity(parity_rows(rows, parity) * row_bytes);
    for (source_row, _, count) in parity_segments(0, rows, parity) {
        output.extend_from_slice(&bytes[source_row * row_bytes..(source_row + count) * row_bytes]);
    }
    Ok(output)
}

fn merge_parity_bytes(owner: &[u8], peer: &[u8], rows: usize, row_bytes: usize) -> Result<Vec<u8>, BackendError> {
    if owner.len() != parity_rows(rows, 0) * row_bytes || peer.len() != parity_rows(rows, 1) * row_bytes {
        return Err(compute_error(format!("DSA parity merge owner/peer={}/{} rows={rows} row_bytes={row_bytes} 非法", owner.len(), peer.len())));
    }
    let mut output = vec![0u8; rows * row_bytes];
    for logical in 0..rows {
        let parity = (logical / ROCM_KV_BLOCK_SIZE) & 1;
        let source_row = parity_row(logical);
        let source = if parity == 0 { owner } else { peer };
        output[logical * row_bytes..(logical + 1) * row_bytes].copy_from_slice(&source[source_row * row_bytes..(source_row + 1) * row_bytes]);
    }
    Ok(output)
}

struct RocmPagedDsaLayer {
    keys: Arc<ops::hip::DeviceBuffer>,
    scales: Arc<ops::hip::DeviceBuffer>,
    /// CPU global Top-K 的 Q8 blocked 全历史副本；按需从 GPU cache 构建，此后每 token 只追加一行。
    cpu_keys: Option<crate::kernel::cpu::dsa::Q8KeyBlocks>,
    /// Prefill 异步建立的全量主存记录；后续 GPU 分块 selection 直接以它为历史 owner。
    cpu_mirror: Option<std::sync::Mutex<RocmDsaCpuMirror>>,
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

struct RocmCooperativeDsaLayer {
    keys: Arc<ops::hip::DeviceBuffer>,
    scales: Arc<ops::hip::DeviceBuffer>,
    rows: usize,
    committed_rows: usize,
}

struct RocmCooperativeDsaPeer {
    device_id: i32,
    layers: Vec<Option<RocmCooperativeDsaLayer>>,
    block_table: RocmBlockTable,
    query: Option<Arc<ops::hip::DeviceBuffer>>,
    head_weights: Option<Arc<ops::hip::DeviceBuffer>>,
    selection: Option<Arc<ops::hip::DeviceBuffer>>,
    selection_rows: usize,
    selection_start: usize,
}

enum RocmCooperativeDsaQuery<'a> {
    Projected(&'a RocmTensor),
    QLora { input: &'a RocmTensor, owner_wq_b: &'a RocmWeight, peer_wq_b: &'a RocmWeight, position: usize, cosine: &'a [f32], sine: &'a [f32], spec: &'a crate::attention::dsa::DsaSpec },
}

struct RocmDsaCpuMirror {
    record: DsaLayerSerde,
    head_dim: usize,
    transfer: Option<Arc<ops::hip::DeviceBuffer>>,
    available_download: Option<ops::hip::AsyncHostDownload>,
    pending_download: Option<(usize, usize, ops::hip::AsyncHostDownload)>,
}

impl RocmDsaCpuMirror {
    fn new(capacity: usize, head_dim: usize, key_group_size: usize, hadamard: bool) -> Result<Self, BackendError> {
        if capacity == 0 || head_dim == 0 || key_group_size == 0 || !head_dim.is_multiple_of(key_group_size) {
            return Err(compute_error(format!("ROCm DSA CPU mirror capacity={capacity} dim={head_dim} group={key_group_size} 非法")));
        }
        Ok(Self {
            record: DsaLayerSerde { rows: 0, key_group_size, hadamard, keys: Vec::with_capacity(capacity.saturating_mul(head_dim)), scales: Vec::with_capacity(capacity.saturating_mul(head_dim / key_group_size).saturating_mul(2)) },
            head_dim,
            transfer: None,
            available_download: None,
            pending_download: None,
        })
    }

    fn from_record(record: DsaLayerSerde, head_dim: usize) -> Self {
        Self { record, head_dim, transfer: None, available_download: None, pending_download: None }
    }

    fn finish_pending(&mut self) -> Result<(), BackendError> {
        let Some((start, rows, mut download)) = self.pending_download.take() else { return Ok(()) };
        let bytes = download.wait().map_err(compute_error)?;
        self.append_downloaded(start, rows, bytes)?;
        self.available_download = Some(download);
        Ok(())
    }

    fn append_downloaded(&mut self, start: usize, rows: usize, bytes: &[u8]) -> Result<(), BackendError> {
        if start != self.record.rows {
            return Err(compute_error(format!("ROCm DSA CPU mirror append start={start}，期望 {}", self.record.rows)));
        }
        let key_bytes = rows.checked_mul(self.head_dim).ok_or_else(|| compute_error("ROCm DSA CPU mirror key 大小溢出"))?;
        let scale_bytes = rows.checked_mul(self.head_dim / self.record.key_group_size).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm DSA CPU mirror scale 大小溢出"))?;
        if bytes.len() != key_bytes + scale_bytes {
            return Err(compute_error(format!("ROCm DSA CPU mirror bytes={}，期望 {}", bytes.len(), key_bytes + scale_bytes)));
        }
        self.record.keys.extend_from_slice(&bytes[..key_bytes]);
        self.record.scales.extend_from_slice(&bytes[key_bytes..]);
        self.record.rows += rows;
        Ok(())
    }

    fn queue(&mut self, device_id: i32, start: usize, rows: usize, head_dim: usize, keys: &ops::hip::DeviceBuffer, scales: &ops::hip::DeviceBuffer) -> Result<(), BackendError> {
        self.finish_pending()?;
        if rows == 0 || start != self.record.rows {
            return Err(compute_error(format!("ROCm DSA CPU mirror queue start={start} rows={rows} mirror={}", self.record.rows)));
        }
        let key_bytes = rows.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm DSA CPU mirror key 大小溢出"))?;
        if head_dim != self.head_dim {
            return Err(compute_error(format!("ROCm DSA CPU mirror dim={head_dim}，期望 {}", self.head_dim)));
        }
        let scale_row_bytes = self.head_dim / self.record.key_group_size * 2;
        let scale_bytes = rows.checked_mul(scale_row_bytes).ok_or_else(|| compute_error("ROCm DSA CPU mirror scale 大小溢出"))?;
        let transfer_bytes = key_bytes + scale_bytes;
        if self.transfer.as_ref().is_none_or(|buffer| buffer.device_id() != device_id || buffer.bytes() < transfer_bytes) {
            self.transfer = Some(Arc::new(ops::hip::DeviceBuffer::allocate_reusable(device_id, transfer_bytes).map_err(compute_error)?));
        }
        let transfer = self.transfer.as_ref().expect("DSA mirror transfer 已创建");
        transfer.copy_from_device(0, keys, start * head_dim, key_bytes).map_err(compute_error)?;
        transfer.copy_from_device(key_bytes, scales, start * scale_row_bytes, scale_bytes).map_err(compute_error)?;
        let mut download = match self.available_download.take() {
            Some(download) => download,
            None => ops::hip::AsyncHostDownload::new(device_id, transfer_bytes).map_err(compute_error)?,
        };
        download.enqueue(transfer, transfer_bytes).map_err(compute_error)?;
        self.pending_download = Some((start, rows, download));
        Ok(())
    }

    fn truncate(&mut self, rows: usize, head_dim: usize) -> Result<(), BackendError> {
        self.finish_pending()?;
        if rows > self.record.rows {
            return Err(compute_error(format!("ROCm DSA CPU mirror truncate={rows} 超过 {}", self.record.rows)));
        }
        self.record.keys.truncate(rows * head_dim);
        self.record.scales.truncate(rows * (head_dim / self.record.key_group_size) * 2);
        self.record.rows = rows;
        Ok(())
    }
}

fn ordered_score(score: f32) -> u32 {
    let bits = score.to_bits();
    bits ^ if bits & 0x8000_0000 != 0 { u32::MAX } else { 0x8000_0000 }
}

/// Prefill 的全历史 selection 直接读取 CPU owner。GPU 只负责把当前 chunk 的
/// index query/head weight 与量化 key 送到主存，不再执行 selection kernel。
fn select_prefill_cpu_q8(record: &DsaLayerSerde, head_dim: usize, query: &[f32], head_weights: &[f32], query_rows: usize, query_start: usize, top_k: usize) -> Result<Vec<u32>, String> {
    if record.hadamard || query_rows == 0 || record.rows != query_start + query_rows || record.key_group_size == 0 || !head_dim.is_multiple_of(record.key_group_size) {
        return Err(format!("CPU prefill DSA shape 非法: rows={} start={query_start} query_rows={query_rows} dim={head_dim} group={} hadamard={}", record.rows, record.key_group_size, record.hadamard));
    }
    let head_count = head_weights.len() / query_rows;
    if head_count == 0 || query.len() != query_rows * head_count * head_dim || head_weights.len() != query_rows * head_count {
        return Err(format!("CPU prefill DSA query/weight={}/{} shape 非法", query.len(), head_weights.len()));
    }
    let groups = head_dim / record.key_group_size;
    if record.keys.len() != record.rows * head_dim || record.scales.len() != record.rows * groups * 2 {
        return Err(format!("CPU prefill DSA cache bytes={}/{} shape 非法", record.keys.len(), record.scales.len()));
    }
    if query_start < top_k {
        return Err(format!("CPU prefill DSA query_start={query_start} 小于 top_k={top_k}"));
    }

    #[cfg(target_arch = "x86_64")]
    if head_count == 32 && head_dim == 128 && record.key_group_size == head_dim && std::arch::is_x86_feature_detected!("avx512vnni") {
        let key_codes = unsafe { std::slice::from_raw_parts(record.keys.as_ptr().cast::<i8>(), record.keys.len()) };
        let scale_bits = record.scales.chunks_exact(2).map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
        let mut keys = crate::kernel::cpu::dsa::Q8KeyBlocks::from_q8_row_major(key_codes, &scale_bits, head_dim)?;
        let mut workspace = crate::kernel::cpu::dsa::Q8DsaWorkspace::default();
        let mut selection = Vec::with_capacity(query_rows * top_k);
        for row in 0..query_rows {
            keys.set_logical_rows(query_start + row + 1)?;
            let row_query = &query[row * head_count * head_dim..(row + 1) * head_count * head_dim];
            let row_weights = &head_weights[row * head_count..(row + 1) * head_count];
            let (selected, _, _) = crate::kernel::cpu::dsa::score_q8_blocked_topk(&keys, row_query, row_weights, head_count, head_dim, top_k, &mut workspace)?;
            selection.extend_from_slice(selected);
        }
        return Ok(selection);
    }

    let rows = (0..query_rows)
        .into_par_iter()
        .map(|row| {
            let valid_rows = query_start + row + 1;
            let row_query = &query[row * head_count * head_dim..(row + 1) * head_count * head_dim];
            let row_weights = &head_weights[row * head_count..(row + 1) * head_count];
            let mut heap = std::collections::BinaryHeap::<std::cmp::Reverse<(u32, u32)>>::with_capacity(top_k + 1);
            for token in 0..valid_rows {
                let key = &record.keys[token * head_dim..(token + 1) * head_dim];
                let scale_bits = &record.scales[token * groups * 2..(token + 1) * groups * 2];
                let mut score = 0.0_f32;
                for head in 0..head_count {
                    let q = &row_query[head * head_dim..(head + 1) * head_dim];
                    let mut dot = 0.0_f32;
                    for column in 0..head_dim {
                        let scale_offset = (column / record.key_group_size) * 2;
                        let scale = half::bf16::from_bits(u16::from_ne_bytes([scale_bits[scale_offset], scale_bits[scale_offset + 1]])).to_f32();
                        dot += q[column] * (key[column] as i8 as f32 * scale);
                    }
                    score += row_weights[head] * dot.max(0.0);
                }
                // 同分时保留 token 较小者，与稳定 Top-K 的历史优先顺序一致。
                let rank = (ordered_score(score), u32::MAX - token as u32);
                heap.push(std::cmp::Reverse(rank));
                if heap.len() > top_k {
                    heap.pop();
                }
            }
            let mut ranked = heap.into_iter().map(|entry| entry.0).collect::<Vec<_>>();
            ranked.sort_unstable_by(|left, right| right.cmp(left));
            ranked.into_iter().map(|(_, inverse_token)| u32::MAX - inverse_token).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(rows.into_iter().flatten().collect())
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
    cpu_host_downloads: std::collections::HashMap<i32, ops::hip::AsyncHostDownload>,
    top_k: usize,
    ownership: RocmDsaOwnership,
    pending_pair_layers: Vec<Option<DsaLayerSerde>>,
    pending_pair_reserved_rows: usize,
    block_table: RocmBlockTable,
    cooperative_staging_table: RocmBlockTable,
    pool_block_table: RocmBlockTable,
    kpool_apes: Vec<Option<Arc<ops::hip::DeviceBuffer>>>,
    kpool: usize,
    selection: Option<Arc<ops::hip::DeviceBuffer>>,
    selection_host: Option<Arc<Vec<u32>>>,
    selection_rows: usize,
    selection_start: usize,
    selection_width: usize,
    gpu_host_download: Option<ops::hip::AsyncHostDownload>,
    gpu_host_inflight: bool,
    gpu_wait_samples: usize,
    gpu_wait_ns_sum: u64,
    gpu_wait_ns_max: u64,
    cooperative_peer: Option<RocmCooperativeDsaPeer>,
    union_probe: Option<super::dsa_union_probe::DsaUnionProbe>,
    pub(super) decode_parallelism: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocmDsaSelectionOwnership {
    /// 这是一次全局 top-k 的完整结果；接收 stage 可再按本地 KV parity 拆表。
    Global,
}

pub struct RocmDsaSelection {
    buffer: Arc<ops::hip::DeviceBuffer>,
    host: Option<Arc<Vec<u32>>>,
    rows: usize,
    start: usize,
    width: usize,
    ownership: RocmDsaSelectionOwnership,
}

struct CpuDsaWorkerResult {
    keys: crate::kernel::cpu::dsa::Q8KeyBlocks,
    workspace: crate::kernel::cpu::dsa::Q8DsaWorkspace,
    host_download: ops::hip::AsyncHostDownload,
    transfer_ms: f64,
    candidates: Result<(Vec<u32>, f64, f64, f64, Option<(Vec<u32>, f64)>), String>,
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
    requested_candidates: Option<usize>,
    exact_top_k: Option<usize>,
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
                        let (candidates, score_ms, topk_ms, exact) = if let Some(candidate_count) = task.requested_candidates {
                            let (candidates, score_ms, topk_ms) = crate::kernel::cpu::dsa::select_q8_blocked_candidates(&task.keys, query, weights, task.head_count, task.head_dim, candidate_count, &mut task.workspace)?;
                            // selection 借用 workspace；先物化候选，再次可变借用同一 workspace 完成 final Top-K。
                            let candidates = candidates.to_vec();
                            let exact = task
                                .exact_top_k
                                .map(|top_k| {
                                    let started = std::time::Instant::now();
                                    let selection = crate::kernel::cpu::dsa::finish_q8_blocked_topk(top_k, &mut task.workspace)?.to_vec();
                                    Ok::<_, String>((selection, started.elapsed().as_secs_f64() * 1e3))
                                })
                                .transpose()?;
                            (candidates, score_ms, topk_ms, exact)
                        } else if let Some(top_k) = task.exact_top_k {
                            // 生产 decode：打分与 final Top-K 融合为单次 team run，省一次分发同步。
                            let (selection, score_ms, topk_ms) = crate::kernel::cpu::dsa::score_q8_blocked_topk(&task.keys, query, weights, task.head_count, task.head_dim, top_k, &mut task.workspace)?;
                            (Vec::new(), score_ms, 0.0, Some((selection.to_vec(), topk_ms)))
                        } else {
                            let score_ms = crate::kernel::cpu::dsa::score_q8_blocked_workspace(&task.keys, query, weights, task.head_count, task.head_dim, &mut task.workspace)?;
                            (Vec::new(), score_ms, 0.0, None)
                        };
                        Ok((candidates, score_ms, topk_ms, select_started.elapsed().as_secs_f64() * 1e3, exact))
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
        Ok(Self { buffer: Arc::new(buffer), host: self.host, rows: self.rows, start: self.start, width: self.width, ownership: self.ownership })
    }

    pub(crate) fn move_to_device_ordered(self, device_id: i32) -> Result<Self, BackendError> {
        if self.buffer.device_id() == device_id {
            return Ok(self);
        }
        let buffer = self.buffer.copy_stable_to_device_ordered_async(device_id).map_err(compute_error)?;
        Ok(Self { buffer: Arc::new(buffer), host: self.host, rows: self.rows, start: self.start, width: self.width, ownership: self.ownership })
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

    /// 双机边界接收侧专用：H2D 延迟到首个消费 stage 的 compute stream
    /// （import_selection 处 enqueue_deferred_upload），避免独立 stream 的
    /// host 同步在 decode 热路径上停 ~0.7ms/token。host 副本保留用于 to_host 快路径。
    pub fn from_host(context: &RocmContext, rows: usize, start: usize, values: &[u32]) -> Result<Self, BackendError> {
        if rows == 0 || values.is_empty() || !values.len().is_multiple_of(rows) {
            return Err(compute_error(format!("ROCm DSA selection host shape rows={rows} elements={} 无效", values.len())));
        }
        let bytes = std::mem::size_of_val(values);
        let buffer = ops::hip::DeviceBuffer::upload_ordered(context.device_id, unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), bytes) }).map_err(compute_error)?;
        Ok(Self { buffer: Arc::new(buffer), host: Some(Arc::new(values.to_vec())), rows, start, width: values.len() / rows, ownership: RocmDsaSelectionOwnership::Global })
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
            cpu_host_downloads: std::collections::HashMap::new(),
            top_k,
            ownership: RocmDsaOwnership::Full,
            pending_pair_layers: (0..layer_count).map(|_| None).collect(),
            pending_pair_reserved_rows: 0,
            block_table: RocmBlockTable::new(),
            cooperative_staging_table: RocmBlockTable::new(),
            pool_block_table: RocmBlockTable::new(),
            kpool_apes: (0..layer_count).map(|_| None).collect(),
            kpool: 0,
            selection: None,
            selection_host: None,
            selection_rows: 0,
            selection_start: 0,
            selection_width: top_k,
            gpu_host_download: None,
            gpu_host_inflight: false,
            gpu_wait_samples: 0,
            gpu_wait_ns_sum: 0,
            gpu_wait_ns_max: 0,
            cooperative_peer: None,
            union_probe: None,
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
            if let Some(mirror) = &cached.cpu_mirror {
                mirror.lock().map_err(|_| compute_error(format!("L{layer} ROCm DSA CPU mirror 锁中毒")))?.truncate(rows, self.head_dim)?;
            }
            if self.kpool > 0 {
                cached.pooled_rows = rows / self.kpool;
            }
        }
        for record in self.pending_pair_layers.iter_mut().filter_map(Option::as_mut) {
            if rows > record.rows {
                return Err(compute_error(format!("ROCm pending DSA truncate rows={rows} 超过当前长度 {}", record.rows)));
            }
            record.keys.truncate(rows * self.head_dim);
            record.scales.truncate(rows * (self.head_dim / self.key_group_size) * 2);
            record.rows = rows;
        }
        if let Some(peer) = &mut self.cooperative_peer {
            for cached in peer.layers.iter_mut().filter_map(Option::as_mut) {
                // 小尾块会回退 owner 单卡，peer 合法地落后；只在目标前缀更短时
                // 回退镜像，之后再次进入 cooperative prefill 会补齐差额。
                if rows < cached.rows {
                    cached.rows = rows;
                }
            }
        }
        self.drain_gpu_host_selection()?;
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
            if let Some(mirror) = &cached.cpu_mirror {
                mirror.lock().map_err(|_| compute_error(format!("L{layer} ROCm DSA CPU mirror 锁中毒")))?.truncate(rows, self.head_dim)?;
            }
            if self.kpool > 0 {
                cached.pooled_rows = rows / self.kpool;
            }
        }
        if let Some(record) = self.pending_pair_layers.get_mut(layer).and_then(Option::as_mut) {
            if rows > record.rows {
                return Err(compute_error(format!("L{layer} pending DSA truncate rows={rows} 超过当前长度 {}", record.rows)));
            }
            record.keys.truncate(rows * self.head_dim);
            record.scales.truncate(rows * (self.head_dim / self.key_group_size) * 2);
            record.rows = rows;
        }
        if let Some(cached) = self.cooperative_peer.as_mut().and_then(|peer| peer.layers.get_mut(layer)).and_then(Option::as_mut) {
            if rows < cached.rows {
                cached.rows = rows;
            }
        }
        self.drain_gpu_host_selection()?;
        self.selection = None;
        self.selection_host = None;
        self.selection_rows = 0;
        self.selection_start = 0;
        self.selection_width = self.top_k;
        Ok(())
    }

    /// 清空 selection(kpool 等场景退化为全量注意力)。
    pub fn invalidate_selection(&mut self) {
        if self.gpu_host_inflight {
            // 在途 GPU selection 下载随 selection 一起作废；event 等待失败则丢弃复用槽。
            let drained = self.gpu_host_download.as_mut().is_some_and(|download| download.wait().is_ok());
            if !drained {
                self.gpu_host_download = None;
            }
            self.gpu_host_inflight = false;
        }
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
        let rows = self.layers.get(layer).and_then(Option::as_ref).map(|cached| cached.rows).or_else(|| self.pending_pair_layers.get(layer).and_then(Option::as_ref).map(|record| record.rows));
        position < self.capacity && rows.map_or(position == 0, |rows| rows == position)
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
                cpu_mirror: (ops::hip::options().mla_cpu_hot_rows != 0 || ops::hip::options().prefill_attention_cpu)
                    .then(|| RocmDsaCpuMirror::new(self.capacity, self.head_dim, self.key_group_size, self.hadamard_i8).map(std::sync::Mutex::new))
                    .transpose()?,
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
        if let Some(mirror) = &cached.cpu_mirror {
            mirror.lock().map_err(|_| compute_error(format!("L{layer} ROCm DSA CPU mirror 锁中毒")))?.queue(context.device_id, position, keys.rows, self.head_dim, &cached.keys, &cached.scales)?;
        }
        self.drain_gpu_host_selection()?;
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
        if let Some(mirror) = &cached.cpu_mirror {
            mirror.lock().map_err(|_| compute_error(format!("L{layer} ROCm DSA CPU mirror 锁中毒")))?.queue(context.device_id, position, keys.rows, self.head_dim, &cached.keys, &cached.scales)?;
        }
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
        let transfer_started = std::time::Instant::now();
        let mut host_download = match self.cpu_host_downloads.remove(&context.device_id) {
            Some(download) => download,
            None => ops::hip::AsyncHostDownload::new(context.device_id, transfer_bytes).map_err(compute_error)?,
        };
        let cached = self.layers[layer].as_ref().expect("CPU DSA layer 已校验");
        let mut segments = vec![(&*query_device, 0, query_bytes), (&*weight_device, 0, weight_bytes)];
        if append_row {
            segments.push((&cached.keys, (context_rows - 1) * self.head_dim, key_bytes));
            segments.push((&cached.scales, (context_rows - 1) * (self.head_dim / self.key_group_size) * 2, scale_bytes));
        }
        if let Err(error) = host_download.enqueue_segments(&segments) {
            self.cpu_host_downloads.insert(context.device_id, host_download);
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
            requested_candidates: validate.then_some((self.top_k + DSA_CPU_CANDIDATE_GUARD).min(context_rows)),
            exact_top_k: Some(self.top_k),
        };
        let dispatched = std::time::Instant::now();
        if let Err(error) = self.cpu_dispatcher.as_ref().unwrap().tasks.send(task) {
            let task = error.0;
            self.layers[layer].as_mut().unwrap().cpu_keys = Some(task.keys);
            self.cpu_workspace = task.workspace;
            self.cpu_host_downloads.insert(context.device_id, task.host_download);
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
        let Some(pending) = self.cpu_pending.take() else {
            return self.finish_gpu_host_selection();
        };
        if context.device_id != pending.device_id {
            return Err(compute_error(format!("CPU DSA finish device={}，begin device={}", context.device_id, pending.device_id)));
        }
        let overlap_ms = pending.dispatched.elapsed().as_secs_f64() * 1e3;
        let wait_started = std::time::Instant::now();
        let worker = self.cpu_dispatcher.as_ref().expect("CPU DSA pending 必有 dispatcher").results.recv().map_err(|_| compute_error(format!("L{} CPU DSA dispatcher 已退出", pending.layer)))?;
        let wait_ms = wait_started.elapsed().as_secs_f64() * 1e3;
        self.layers[pending.layer].as_mut().expect("CPU DSA pending layer 已校验").cpu_keys = Some(worker.keys);
        self.cpu_workspace = worker.workspace;
        self.cpu_host_downloads.insert(pending.device_id, worker.host_download);
        let transfer_ms = worker.transfer_ms;
        let (candidates, score_ms, topk_ms, select_ms, cpu_exact) = worker.candidates.map_err(compute_error)?;
        let candidate_count = candidates.len();
        let (cpu_selection, cpu_rerank_ms) = cpu_exact.ok_or_else(|| compute_error("CPU DSA exact rerank 结果缺失"))?;
        let selection_bytes = unsafe { std::slice::from_raw_parts(cpu_selection.as_ptr().cast(), std::mem::size_of_val(cpu_selection.as_slice())) };
        let selection = Arc::new(ops::hip::DeviceBuffer::upload_ordered(context.device_id, selection_bytes).map_err(compute_error)?);
        selection.enqueue_deferred_upload().map_err(compute_error)?;
        selection.retain_for_active_stage();
        if let Some(exact) = pending.exact {
            let candidate_bytes = unsafe { std::slice::from_raw_parts(candidates.as_ptr().cast(), std::mem::size_of_val(candidates.as_slice())) };
            let candidate_buffer = Arc::new(ops::hip::DeviceBuffer::upload_ordered(context.device_id, candidate_bytes).map_err(compute_error)?);
            candidate_buffer.enqueue_deferred_upload().map_err(compute_error)?;
            candidate_buffer.retain_for_active_stage();
            let gpu_reranked = ops::hip::try_dsa_rerank_paged_q8_candidates(
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
            let mut reranked = vec![0_u32; self.top_k];
            gpu_reranked.copy_to_host(unsafe { std::slice::from_raw_parts_mut(reranked.as_mut_ptr().cast(), reranked.len() * 4) }).map_err(compute_error)?;
            let recall = exact.iter().filter(|token| candidates.binary_search(token).is_ok()).count();
            let rerank_exact = reranked.as_slice() == exact.as_slice();
            let cpu_rerank_exact = cpu_selection.as_slice() == exact.as_slice();
            let sample = self.cpu_select_counts[pending.layer];
            if sample == 0 || recall != self.top_k || !rerank_exact || !cpu_rerank_exact {
                eprintln!(
                    "[dsa-cpu-select-oracle] device={} layer={} sample={sample} context={} candidate_recall={recall}/{} rerank_exact={rerank_exact} cpu_rerank_exact={cpu_rerank_exact} cpu_rerank_ms={cpu_rerank_ms:.3}",
                    context.device_id, pending.layer, pending.context_rows, self.top_k
                );
            }
        }
        self.selection = Some(selection);
        self.selection_host = (ops::hip::options().mla_cpu_hot_rows != 0).then(|| Arc::new(cpu_selection));
        self.selection_rows = 1;
        self.selection_start = pending.context_rows - 1;
        self.selection_width = self.top_k;
        let sample = self.cpu_select_counts[pending.layer];
        self.cpu_select_counts[pending.layer] += 1;
        if ops::hip::options().mla_hot_trace && (sample < 2 || sample.is_multiple_of(128)) {
            eprintln!(
                "[dsa-cpu-select] device={} layer={} sample={sample} context={} candidates={candidate_count} d2h_bytes={} d2h_ms={:.3} begin_ms={:.3} overlap_ms={overlap_ms:.3} wait_ms={wait_ms:.3} score_ms={score_ms:.3} topk_ms={topk_ms:.3} cpu_final_topk_ms={cpu_rerank_ms:.3} select_ms={select_ms:.3} total_ms={:.3} selection=cpu_async_q8_final",
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
        if self.gpu_host_inflight {
            return Err(compute_error("ROCm DSA begin 前一份 GPU selection 尚未 finish"));
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
        if super::dsa_union_probe::enabled() && query.rows == 1 {
            if self.union_probe.is_none() {
                self.union_probe = Some(super::dsa_union_probe::DsaUnionProbe::new(context.device_id)?);
            }
            // score 必须在后续 kernel 覆盖 workspace 前下载（与既有 shadow 同约束）。
            self.union_probe.as_mut().expect("刚创建").maybe_record_scores(layer, context_rows - 1, context_rows)?;
        }
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
        // selection D2H 不在这里整流排空：enqueue 到 compute stream + event 后立即返回，
        // 等待推迟到 select_finish（届时 q_b/KV/append 已提交，GPU 不会在 select 后空转）。
        if ops::hip::options().mla_cpu_hot_rows != 0 {
            if self.gpu_host_inflight {
                return Err(compute_error("ROCm DSA GPU selection 下载尚未 finish"));
            }
            let selection_bytes = query.rows.checked_mul(self.top_k).and_then(|elements| elements.checked_mul(std::mem::size_of::<u32>())).ok_or_else(|| compute_error("ROCm DSA host selection 大小溢出"))?;
            let mut download = match self.gpu_host_download.take() {
                Some(download) => download,
                None => ops::hip::AsyncHostDownload::new(context.device_id, selection_bytes).map_err(compute_error)?,
            };
            // enqueue 失败时内部 pending 已被复位，download 可以安全回到复用池。
            let enqueue = download.enqueue(&exact, selection_bytes);
            self.gpu_host_download = Some(download);
            enqueue.map_err(compute_error)?;
            self.gpu_host_inflight = true;
        } else {
            self.selection_host = None;
        }
        if query.rows == 1
            && let Some(probe) = self.union_probe.as_mut()
        {
            probe.record_selection(layer, context_rows - 1, context_rows, self.top_k, &exact)?;
        }
        self.selection = Some(Arc::new(exact));
        self.selection_rows = query.rows;
        self.selection_start = context_rows - query.rows;
        self.selection_width = self.top_k;
        Ok(())
    }

    /// GPU exact selection 的异步 D2H 在此物化。select_begin 已把复制排在
    /// select kernel 之后；此刻 q_b/KV/append 均已提交，等待 event 不会让 GPU 空转。
    fn finish_gpu_host_selection(&mut self) -> Result<(), BackendError> {
        if !self.gpu_host_inflight {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let download = self.gpu_host_download.as_mut().ok_or_else(|| compute_error("ROCm DSA GPU selection 下载状态缺失"))?;
        let bytes = download.wait().map_err(compute_error)?;
        self.gpu_host_inflight = false;
        let elements = self.selection_rows.checked_mul(self.selection_width).ok_or_else(|| compute_error("ROCm DSA host selection 元素数溢出"))?;
        let mut host = vec![0_u32; elements];
        let expected = elements.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| compute_error("ROCm DSA host selection 字节数溢出"))?;
        if bytes.len() != expected {
            return Err(compute_error(format!("ROCm DSA GPU selection 下载 bytes={}，期望 {expected}", bytes.len())));
        }
        host.copy_from_slice(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u32>(), elements) });
        self.selection_host = Some(Arc::new(host));
        if ops::hip::options().mla_hot_trace {
            let elapsed_ns = started.elapsed().as_nanos() as u64;
            self.gpu_wait_samples += 1;
            self.gpu_wait_ns_sum += elapsed_ns;
            self.gpu_wait_ns_max = self.gpu_wait_ns_max.max(elapsed_ns);
            if self.gpu_wait_samples.is_multiple_of(128) {
                eprintln!("[dsa-hot-trace] selects={} wait_avg_ms={:.3} wait_max_ms={:.3}", self.gpu_wait_samples, self.gpu_wait_ns_sum as f64 / self.gpu_wait_samples as f64 / 1e6, self.gpu_wait_ns_max as f64 / 1e6,);
                self.gpu_wait_ns_sum = 0;
                self.gpu_wait_ns_max = 0;
            }
        }
        Ok(())
    }

    /// selection 被整体作废（truncate/import/invalidate）时排空在途 GPU 下载，
    /// 避免复用池里的 download 停留在 pending 状态。
    fn drain_gpu_host_selection(&mut self) -> Result<(), BackendError> {
        if self.gpu_host_inflight {
            let download = self.gpu_host_download.as_mut().ok_or_else(|| compute_error("ROCm DSA GPU selection 下载状态缺失"))?;
            download.wait().map_err(compute_error)?;
            self.gpu_host_inflight = false;
        }
        Ok(())
    }

    fn ensure_sequence_shard_peer(&mut self, owner: &RocmContext, peer: &RocmContext, layer: usize) -> Result<(), BackendError> {
        if self.kpool != 0 || self.hadamard_i8 {
            return Err(compute_error(format!("L{layer} DSA sequence shard 仅支持 GLM raw-Q8 cache")));
        }
        if let Some(existing) = self.cooperative_peer.as_ref() {
            return if existing.device_id == peer.device_id { Ok(()) } else { Err(compute_error(format!("L{layer} DSA sequence shard peer 已绑定 device={}，不能改为 {}", existing.device_id, peer.device_id))) };
        }
        if self.layers.iter().any(Option::is_some) {
            return Err(compute_error(format!("L{layer} DSA sequence shard 必须在首个 append 前启用")));
        }
        self.ownership = RocmDsaOwnership::BlockParity;
        self.block_table = RocmBlockTable::new();
        self.cooperative_peer = Some(RocmCooperativeDsaPeer {
            device_id: peer.device_id,
            layers: (0..self.layers.len()).map(|_| None).collect(),
            block_table: RocmBlockTable::new(),
            query: None,
            head_weights: None,
            selection: None,
            selection_rows: 0,
            selection_start: 0,
        });
        let reserved_rows = self.pending_pair_reserved_rows;
        let groups = self.head_dim / self.key_group_size;
        for index in 0..self.pending_pair_layers.len() {
            let Some(record) = self.pending_pair_layers[index].take() else { continue };
            let owner_rows = parity_rows(record.rows, 0);
            let peer_rows = parity_rows(record.rows, 1);
            let owner_capacity = parity_rows(reserved_rows.max(record.rows), 0).max(owner_rows);
            let peer_capacity = parity_rows(reserved_rows.max(record.rows), 1).max(peer_rows);
            let owner_keys = split_parity_bytes(&record.keys, record.rows, self.head_dim, 0)?;
            let peer_keys = split_parity_bytes(&record.keys, record.rows, self.head_dim, 1)?;
            let scale_row_bytes = groups * 2;
            let owner_scales = split_parity_bytes(&record.scales, record.rows, scale_row_bytes, 0)?;
            let peer_scales = split_parity_bytes(&record.scales, record.rows, scale_row_bytes, 1)?;
            owner.activate().map_err(compute_error)?;
            self.layers[index] = Some(RocmPagedDsaLayer {
                keys: upload_cache_buffer(owner.device_id, &owner_keys, owner_capacity * self.head_dim)?,
                scales: upload_cache_buffer(owner.device_id, &owner_scales, owner_capacity * scale_row_bytes)?,
                cpu_keys: None,
                cpu_mirror: None,
                hadamard_shadow_keys: None,
                hadamard_shadow_scales: None,
                hadamard: false,
                gates: None,
                pooled_keys: None,
                pooled_scales: None,
                pooled_rows: 0,
                interval_lower: None,
                interval_upper: None,
                interval_rows: 0,
                rows: record.rows,
                committed_rows: owner_capacity,
            });
            peer.activate().map_err(compute_error)?;
            self.cooperative_peer.as_mut().expect("DSA peer 已创建").layers[index] = Some(RocmCooperativeDsaLayer {
                keys: upload_cache_buffer(peer.device_id, &peer_keys, peer_capacity * self.head_dim)?,
                scales: upload_cache_buffer(peer.device_id, &peer_scales, peer_capacity * scale_row_bytes)?,
                rows: record.rows,
                committed_rows: peer_capacity,
            });
        }
        owner.activate().map_err(compute_error)?;
        peer.activate().map_err(compute_error)?;
        Ok(())
    }

    fn ensure_sequence_shard_layer(&mut self, context: &RocmContext, layer: usize, global_end: usize, parity: usize) -> Result<(), BackendError> {
        let required_rows = parity_rows(global_end, parity);
        let physical_capacity = parity_rows(self.capacity, parity);
        let required_storage = required_rows.max(1);
        let groups = self.head_dim / self.key_group_size;
        if parity == 0 {
            let slot = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
            if slot.is_none() {
                let committed_rows = committed_cache_rows(0, required_storage, physical_capacity)?;
                *slot = Some(RocmPagedDsaLayer {
                    keys: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, committed_rows * self.head_dim).map_err(compute_error)?),
                    scales: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, committed_rows * groups * 2).map_err(compute_error)?),
                    cpu_keys: None,
                    cpu_mirror: None,
                    hadamard_shadow_keys: None,
                    hadamard_shadow_scales: None,
                    hadamard: false,
                    gates: None,
                    pooled_keys: None,
                    pooled_scales: None,
                    pooled_rows: 0,
                    interval_lower: None,
                    interval_upper: None,
                    interval_rows: 0,
                    rows: 0,
                    committed_rows,
                });
            }
            let cached = slot.as_mut().expect("DSA owner shard 已创建");
            if global_end < cached.rows {
                return Err(compute_error(format!("L{layer} DSA owner shard end={global_end} 小于 rows={}", cached.rows)));
            }
            let next = committed_cache_rows(cached.committed_rows, required_storage, physical_capacity)?;
            if next > cached.committed_rows {
                let used = parity_rows(cached.rows, parity);
                cached.keys = grow_cache_buffer(context.device_id, &cached.keys, used * self.head_dim, next * self.head_dim)?;
                cached.scales = grow_cache_buffer(context.device_id, &cached.scales, used * groups * 2, next * groups * 2)?;
                cached.committed_rows = next;
            }
            self.block_table.get("DSA owner parity", physical_capacity, context.device_id, required_rows.max(1))?;
        } else {
            let peer = self.cooperative_peer.as_mut().expect("DSA peer shard 已创建");
            let slot = peer.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
            if slot.is_none() {
                let committed_rows = committed_cache_rows(0, required_storage, physical_capacity)?;
                *slot = Some(RocmCooperativeDsaLayer {
                    keys: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, committed_rows * self.head_dim).map_err(compute_error)?),
                    scales: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, committed_rows * groups * 2).map_err(compute_error)?),
                    rows: 0,
                    committed_rows,
                });
            }
            let cached = slot.as_mut().expect("DSA peer shard 已创建");
            if global_end < cached.rows {
                return Err(compute_error(format!("L{layer} DSA peer shard end={global_end} 小于 rows={}", cached.rows)));
            }
            let next = committed_cache_rows(cached.committed_rows, required_storage, physical_capacity)?;
            if next > cached.committed_rows {
                let used = parity_rows(cached.rows, parity);
                cached.keys = grow_cache_buffer(context.device_id, &cached.keys, used * self.head_dim, next * self.head_dim)?;
                cached.scales = grow_cache_buffer(context.device_id, &cached.scales, used * groups * 2, next * groups * 2)?;
                cached.committed_rows = next;
            }
            peer.block_table.get("DSA peer parity", physical_capacity, context.device_id, required_rows.max(1))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn append_cooperative_layernorm_rope(
        &mut self,
        owner: &RocmContext,
        peer: &RocmContext,
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
    ) -> Result<(), BackendError> {
        if !self.supports_layernorm_rope(keys.rows, keys.cols) || position.checked_add(keys.rows).is_none_or(|end| end > self.capacity) || !self.can_append(layer, position) {
            return Err(compute_error(format!("L{layer} cooperative DSA append position={position} key=[{},{}] capacity={} 非法", keys.rows, keys.cols, self.capacity)));
        }
        let end = position + keys.rows;
        if keys.rows == 1 {
            return self.append_cooperative_layernorm_rope_single(owner, peer, layer, position, keys, norm_weight, norm_bias, eps, rotary_dim, layout, cos, sin, end);
        }
        let diagnose = keys.rows == 1;
        let trace_started = std::time::Instant::now();
        let mut trace_last = trace_started;
        let mut trace_step = |step: &str| {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(trace_last);
            trace_last = now;
            if diagnose && elapsed >= std::time::Duration::from_millis(5) {
                eprintln!("[rocm-pair-dsa-step] layer={layer} position={position} step={step} wall_ms={:.3} total_ms={:.3}", elapsed.as_secs_f64() * 1000.0, now.duration_since(trace_started).as_secs_f64() * 1000.0,);
            }
        };
        ops::hip::set_device(owner.device_id).map_err(compute_error)?;
        let owner_stream = ops::hip::active_compute_stream() as usize;
        ops::hip::order_stream_after(owner.device_id, owner_stream, 0).map_err(compute_error)?;
        ops::hip::activate_compute_stream(owner.device_id, 0).map_err(compute_error)?;
        trace_step("entry-handoff");
        self.ensure_sequence_shard_peer(owner, peer, layer)?;
        trace_step("ensure-peer");
        let end = position + keys.rows;
        owner.activate().map_err(compute_error)?;
        self.ensure_sequence_shard_layer(owner, layer, end, 0)?;
        trace_step("ensure-owner-layer");
        peer.activate().map_err(compute_error)?;
        self.ensure_sequence_shard_layer(peer, layer, end, 1)?;
        trace_step("ensure-peer-layer");

        owner.activate().map_err(compute_error)?;
        let groups = self.head_dim / self.key_group_size;
        let staging_keys = Arc::new(ops::hip::DeviceBuffer::allocate_reusable(owner.device_id, keys.rows * self.head_dim).map_err(compute_error)?);
        let staging_scales = Arc::new(ops::hip::DeviceBuffer::allocate_reusable(owner.device_id, keys.rows * groups * 2).map_err(compute_error)?);
        let staging_table = self.cooperative_staging_table.get("DSA cooperative staging", keys.rows, owner.device_id, keys.rows)?;
        trace_step("staging-allocate");
        ops::hip::try_paged_dsa_append_layernorm_rope_q8_at(
            owner.device_id,
            keys.device.as_deref().ok_or_else(|| compute_error("cooperative DSA key 缺少 device buffer"))?,
            norm_weight,
            norm_bias,
            &staging_keys,
            &staging_scales,
            &staging_table,
            0,
            position,
            keys.rows,
            self.head_dim,
            rotary_dim,
            layout,
            cos,
            sin,
            ROCM_KV_BLOCK_SIZE,
            eps,
        )
        .map_err(compute_error)?;
        trace_step("append-kernel");

        let owner_cached = self.layers[layer].as_mut().expect("DSA owner shard 已创建");
        for (source_row, target_row, rows) in parity_segments(position, keys.rows, 0) {
            owner_cached.keys.copy_from_device(target_row * self.head_dim, &staging_keys, source_row * self.head_dim, rows * self.head_dim).map_err(compute_error)?;
            owner_cached.scales.copy_from_device(target_row * groups * 2, &staging_scales, source_row * groups * 2, rows * groups * 2).map_err(compute_error)?;
        }
        owner_cached.rows = end;
        trace_step("owner-cache-copy");

        let stable = |buffer: &Arc<ops::hip::DeviceBuffer>| {
            if buffer.is_async_allocated() { buffer.copy_to_stable_deferred().map(Arc::new).map_err(compute_error) } else { Ok(buffer.clone()) }
        };
        let stable_keys = stable(&staging_keys)?;
        let stable_scales = stable(&staging_scales)?;
        trace_step("staging-stabilize");
        let peer_state = self.cooperative_peer.as_mut().expect("DSA peer shard 已创建");
        let peer_cached = peer_state.layers[layer].as_mut().expect("DSA peer layer 已创建");
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        for (source_row, target_row, rows) in parity_segments(position, keys.rows, 1) {
            for (source, target, row_bytes) in [(&stable_keys, peer_cached.keys.as_ref(), self.head_dim), (&stable_scales, peer_cached.scales.as_ref(), groups * 2)] {
                sources.push(Arc::new(ops::hip::DeviceBuffer::view(source.clone(), source_row * row_bytes, rows * row_bytes).map_err(compute_error)?));
                targets.push((target, target_row * row_bytes));
            }
        }
        trace_step("peer-views");
        if !sources.is_empty() {
            peer.activate().map_err(compute_error)?;
            ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(&sources, &targets, peer.device_id, owner.device_id)
                .map_err(|error| compute_error(format!("L{layer} cooperative DSA append owner->peer: {error}")))?;
        }
        trace_step("owner-to-peer");
        peer_cached.rows = end;
        self.invalidate_selection();
        ops::hip::order_stream_after(owner.device_id, 0, owner_stream).map_err(compute_error)?;
        ops::hip::activate_compute_stream(owner.device_id, owner_stream).map_err(compute_error)?;
        trace_step("exit-handoff");
        Ok(())
    }

    /// decode 单行快速路径：该行只属于一个 parity 半片，LN+RoPE+Q8 融合 kernel
    /// 直写目标 cache 的最终行——零 staging、零 parity 拷贝、零 P2P、不切流。
    /// parity=1 时 kernel 在 owner 流上跨卡直写 peer 显存；对端可见性由后续
    /// select/attention 的 event/P2P 链覆盖（它们的源流排在本 kernel 之后）。
    #[allow(clippy::too_many_arguments)]
    fn append_cooperative_layernorm_rope_single(
        &mut self,
        owner: &RocmContext,
        peer: &RocmContext,
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
        end: usize,
    ) -> Result<(), BackendError> {
        self.ensure_sequence_shard_peer(owner, peer, layer)?;
        self.ensure_sequence_shard_layer(owner, layer, end, 0)?;
        self.ensure_sequence_shard_layer(peer, layer, end, 1)?;
        let parity = (position / ROCM_KV_BLOCK_SIZE) & 1;
        let local_row = parity_rows(position, parity);
        let groups = self.head_dim / self.key_group_size;
        let (target_keys, target_scales, target_table, cache_device) = if parity == 0 {
            let cached = self.layers[layer].as_ref().expect("DSA owner shard 已创建");
            (cached.keys.clone(), cached.scales.clone(), self.block_table.get("DSA owner parity", parity_rows(self.capacity, 0), owner.device_id, parity_rows(end, 0).max(1))?, owner.device_id)
        } else {
            let peer_state = self.cooperative_peer.as_mut().expect("DSA peer shard 已创建");
            let cached = peer_state.layers[layer].as_ref().expect("DSA peer layer 已创建");
            (cached.keys.clone(), cached.scales.clone(), peer_state.block_table.get("DSA peer parity", parity_rows(self.capacity, 1), peer.device_id, parity_rows(end, 1).max(1))?, peer.device_id)
        };
        owner.activate().map_err(compute_error)?;
        ops::hip::try_paged_dsa_append_layernorm_rope_q8_remote(
            owner.device_id,
            cache_device,
            keys.device.as_deref().ok_or_else(|| compute_error("cooperative DSA key 缺少 device buffer"))?,
            norm_weight,
            norm_bias,
            &target_keys,
            &target_scales,
            &target_table,
            local_row,
            position,
            1,
            self.head_dim,
            rotary_dim,
            layout,
            cos,
            sin,
            ROCM_KV_BLOCK_SIZE,
            eps,
        )
        .map_err(compute_error)?;
        self.layers[layer].as_mut().expect("DSA owner shard 已创建").rows = end;
        self.cooperative_peer.as_mut().expect("DSA peer shard 已创建").layers[layer].as_mut().expect("DSA peer layer 已创建").rows = end;
        self.invalidate_selection();
        Ok(())
    }

    fn synchronize_cooperative_peer(&mut self, owner: &RocmContext, peer: &RocmContext, layer: usize) -> Result<(), BackendError> {
        if self.cooperative_peer.as_ref().is_some_and(|state| state.device_id != peer.device_id) {
            return Err(compute_error(format!("L{layer} ROCm cooperative DSA peer 已绑定 device={}，不能改为 {}", self.cooperative_peer.as_ref().expect("peer 已存在").device_id, peer.device_id,)));
        }
        if self.cooperative_peer.is_none() {
            self.cooperative_peer = Some(RocmCooperativeDsaPeer {
                device_id: peer.device_id,
                layers: (0..self.layers.len()).map(|_| None).collect(),
                block_table: RocmBlockTable::new(),
                query: None,
                head_weights: None,
                selection: None,
                selection_rows: 0,
                selection_start: 0,
            });
        }

        let (owner_rows, owner_keys, owner_scales, hadamard) = {
            let cached = self.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} cooperative DSA owner cache 尚未初始化")))?;
            (cached.rows, cached.keys.clone(), cached.scales.clone(), cached.hadamard)
        };
        if hadamard {
            return Err(compute_error(format!("L{layer} cooperative DSA 暂不支持 Hadamard key")));
        }
        let (peer_rows, peer_committed) = self.cooperative_peer.as_ref().and_then(|state| state.layers.get(layer)).and_then(Option::as_ref).map_or((0, 0), |cached| (cached.rows, cached.committed_rows));
        if peer_rows == owner_rows {
            return Ok(());
        }
        if peer_rows > owner_rows {
            return Err(compute_error(format!("L{layer} cooperative DSA peer rows={peer_rows} 超过 owner rows={owner_rows}")));
        }

        let next_committed = committed_cache_rows(peer_committed, owner_rows, self.capacity)?;
        let scale_row_bytes = self.head_dim / self.key_group_size * 2;
        peer.activate().map_err(compute_error)?;
        let (peer_keys, peer_scales) = {
            let peer_state = self.cooperative_peer.as_mut().expect("cooperative DSA peer 已创建");
            let slot = peer_state.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
            if slot.is_none() {
                *slot = Some(RocmCooperativeDsaLayer {
                    keys: Arc::new(ops::hip::DeviceBuffer::allocate(peer.device_id, next_committed * self.head_dim).map_err(compute_error)?),
                    scales: Arc::new(ops::hip::DeviceBuffer::allocate(peer.device_id, next_committed * scale_row_bytes).map_err(compute_error)?),
                    rows: 0,
                    committed_rows: next_committed,
                });
            }
            let cached = slot.as_mut().expect("cooperative DSA layer 已创建");
            if next_committed > cached.committed_rows {
                cached.keys = grow_cache_buffer(peer.device_id, &cached.keys, cached.rows * self.head_dim, next_committed * self.head_dim)?;
                cached.scales = grow_cache_buffer(peer.device_id, &cached.scales, cached.rows * scale_row_bytes, next_committed * scale_row_bytes)?;
                cached.committed_rows = next_committed;
            }
            (cached.keys.clone(), cached.scales.clone())
        };

        let missing_rows = owner_rows - peer_rows;
        // cooperative DSA 与后续 pair attention 共用两卡 legacy default stream。
        // owner 的 stage stream 已由调用方在入口交给 default；这里不能再通过
        // context.activate() 切回 stage stream，否则 P2P 临时量会跨两套回收边界。
        ops::hip::activate_compute_stream(owner.device_id, 0).map_err(compute_error)?;
        let key_view = ops::hip::DeviceBuffer::view(owner_keys, peer_rows * self.head_dim, missing_rows * self.head_dim).map_err(compute_error)?;
        let scale_view = ops::hip::DeviceBuffer::view(owner_scales, peer_rows * scale_row_bytes, missing_rows * scale_row_bytes).map_err(compute_error)?;
        // cache 可能来自 stream-ordered allocation；只稳定本次缺失的紧凑 range，
        // 不复制整层历史。
        let key_view = Arc::new(if key_view.is_async_allocated() { key_view.copy_to_stable_deferred().map_err(compute_error)? } else { key_view });
        let scale_view = Arc::new(if scale_view.is_async_allocated() { scale_view.copy_to_stable_deferred().map_err(compute_error)? } else { scale_view });
        peer.activate().map_err(compute_error)?;
        ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(
            &[key_view, scale_view],
            &[(peer_keys.as_ref(), peer_rows * self.head_dim), (peer_scales.as_ref(), peer_rows * scale_row_bytes)],
            peer.device_id,
            owner.device_id,
        )
        .map_err(|error| compute_error(format!("L{layer} cooperative DSA history owner->peer: {error}")))?;
        let cached = self.cooperative_peer.as_mut().and_then(|state| state.layers.get_mut(layer)).and_then(Option::as_mut).ok_or_else(|| compute_error(format!("L{layer} cooperative DSA peer layer 丢失")))?;
        cached.rows = owner_rows;
        Ok(())
    }

    /// 精确按 query 行拆分 DSA：两卡都持有相同 Q8 历史，owner 处理前半行，
    /// peer 处理后半行。每行仍扫描其完整因果前缀，最终 selection 按原行序拼接。
    pub(super) fn select_prefill_cooperative(&mut self, owner: &RocmContext, peer: &RocmContext, layer: usize, query: &RocmTensor, head_weights: &RocmTensor) -> Result<bool, BackendError> {
        self.select_prefill_cooperative_impl(owner, peer, layer, RocmCooperativeDsaQuery::Projected(query), head_weights)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn select_prefill_cooperative_projected(
        &mut self,
        owner: &RocmContext,
        peer: &RocmContext,
        layer: usize,
        q_lora: &RocmTensor,
        owner_wq_b: &RocmWeight,
        peer_wq_b: &RocmWeight,
        head_weights: &RocmTensor,
        position: usize,
        cosine: &[f32],
        sine: &[f32],
        spec: &crate::attention::dsa::DsaSpec,
    ) -> Result<bool, BackendError> {
        self.select_prefill_cooperative_impl(owner, peer, layer, RocmCooperativeDsaQuery::QLora { input: q_lora, owner_wq_b, peer_wq_b, position, cosine, sine, spec }, head_weights)
    }

    fn select_sequence_sharded_impl(
        &mut self,
        owner: &RocmContext,
        peer: &RocmContext,
        layer: usize,
        query_source: RocmCooperativeDsaQuery<'_>,
        head_weights: &RocmTensor,
        query_rows: usize,
        head_count: usize,
    ) -> Result<bool, BackendError> {
        let context_rows = self.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} DSA owner sequence shard 缺失")))?.rows;
        let query_start = context_rows.checked_sub(query_rows).ok_or_else(|| compute_error(format!("L{layer} DSA context={context_rows} 小于 query={query_rows}")))?;
        if context_rows <= self.top_k || (0..2).any(|parity| parity_rows(query_start + 1, parity) < self.top_k) {
            // 两边都攒够局部 Top-K 前，dense MLA 与 2K sparse 的工作量同阶；
            // 此时不生成 selection，避免给局部候选表引入填充值。
            self.invalidate_selection();
            return Ok(true);
        }
        self.drain_gpu_host_selection()?;
        ops::hip::set_device(owner.device_id).map_err(compute_error)?;
        let owner_stream = ops::hip::active_compute_stream() as usize;
        ops::hip::order_stream_after(owner.device_id, owner_stream, 0).map_err(compute_error)?;
        ops::hip::activate_compute_stream(owner.device_id, 0).map_err(compute_error)?;

        let peer_weights_source = owner.tensor_to_stable_deferred(head_weights.clone())?;
        let (owner_query, peer_input, peer_projection) = match query_source {
            RocmCooperativeDsaQuery::Projected(query) => {
                let query = owner.tensor_to_stable_deferred(query.clone())?;
                (query.clone(), query, None)
            }
            RocmCooperativeDsaQuery::QLora { input, owner_wq_b, peer_wq_b, position, cosine, sine, spec } => {
                let input = owner.tensor_to_stable_deferred(owner.tensor_as_bf16(input.clone())?)?;
                let query = owner.linear(&input, owner_wq_b)?;
                let query = owner.rope_prefix(&query, spec.num_heads, spec.rope_dim, spec.rotary_layout, position, cosine, sine)?;
                (query, input, Some((peer_wq_b, position, cosine, sine, spec)))
            }
        };
        let sources = [
            peer_input.device.as_ref().ok_or_else(|| compute_error("DSA sequence shard peer input 缺少 device buffer"))?.clone(),
            peer_weights_source.device.as_ref().ok_or_else(|| compute_error("DSA sequence shard peer weights 缺少 device buffer"))?.clone(),
        ];
        peer.activate().map_err(compute_error)?;
        let (peer_input_buffer, peer_weights_buffer) = {
            let peer_state = self.cooperative_peer.as_mut().ok_or_else(|| compute_error(format!("L{layer} DSA sequence shard peer 尚未创建")))?;
            if peer_state.query.as_ref().is_none_or(|buffer| buffer.bytes() != sources[0].bytes()) {
                peer_state.query = Some(Arc::new(ops::hip::DeviceBuffer::allocate_reusable(peer.device_id, sources[0].bytes()).map_err(compute_error)?));
            }
            if peer_state.head_weights.as_ref().is_none_or(|buffer| buffer.bytes() != sources[1].bytes()) {
                peer_state.head_weights = Some(Arc::new(ops::hip::DeviceBuffer::allocate_reusable(peer.device_id, sources[1].bytes()).map_err(compute_error)?));
            }
            (peer_state.query.as_ref().unwrap().clone(), peer_state.head_weights.as_ref().unwrap().clone())
        };
        ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(&sources, &[(peer_input_buffer.as_ref(), 0), (peer_weights_buffer.as_ref(), 0)], peer.device_id, owner.device_id)
            .map_err(|error| compute_error(format!("L{layer} DSA sequence shard query owner->peer: {error}")))?;
        let peer_input = super::device_tensor_with_arc(peer_input_buffer, query_rows, peer_input.cols, peer_input.dtype);
        let peer_weights = super::device_tensor_with_arc(peer_weights_buffer, query_rows, head_count, head_weights.dtype);
        let peer_query = if let Some((peer_wq_b, position, cosine, sine, spec)) = peer_projection {
            let query = peer.linear(&peer_input, peer_wq_b)?;
            peer.rope_prefix(&query, spec.num_heads, spec.rope_dim, spec.rotary_layout, position, cosine, sine)?
        } else {
            peer_input
        };

        let (peer_keys, peer_scales, peer_table) = {
            let peer_state = self.cooperative_peer.as_mut().expect("DSA sequence shard peer 已创建");
            let cached = peer_state.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} DSA peer sequence shard cache 缺失")))?;
            let table = peer_state.block_table.get("DSA peer parity", parity_rows(self.capacity, 1), peer.device_id, parity_rows(context_rows, 1).max(1))?;
            (cached.keys.clone(), cached.scales.clone(), table)
        };
        let peer_selection = ops::hip::try_dsa_select_paged_q8_sequence_shard(
            peer.device_id,
            &peer_keys,
            &peer_scales,
            self.key_group_size,
            &peer_table,
            peer_query.device.as_deref().ok_or_else(|| compute_error("DSA peer sequence shard query 缺少 device buffer"))?,
            peer_weights.device.as_deref().ok_or_else(|| compute_error("DSA peer sequence shard weights 缺少 device buffer"))?,
            query_rows,
            context_rows,
            query_start,
            head_count,
            self.head_dim,
            self.top_k,
            ROCM_KV_BLOCK_SIZE,
            1,
        )
        .map_err(compute_error)?;

        ops::hip::activate_compute_stream(owner.device_id, 0).map_err(compute_error)?;
        let cached = self.layers[layer].as_ref().expect("DSA owner sequence shard cache 已检查");
        let owner_keys = cached.keys.clone();
        let owner_scales = cached.scales.clone();
        let owner_table = self.block_table.get("DSA owner parity", parity_rows(self.capacity, 0), owner.device_id, parity_rows(context_rows, 0).max(1))?;
        let owner_selection = ops::hip::try_dsa_select_paged_q8_sequence_shard(
            owner.device_id,
            &owner_keys,
            &owner_scales,
            self.key_group_size,
            &owner_table,
            owner_query.device.as_deref().ok_or_else(|| compute_error("DSA owner sequence shard query 缺少 device buffer"))?,
            head_weights.device.as_deref().ok_or_else(|| compute_error("DSA owner sequence shard weights 缺少 device buffer"))?,
            query_rows,
            context_rows,
            query_start,
            head_count,
            self.head_dim,
            self.top_k,
            ROCM_KV_BLOCK_SIZE,
            0,
        )
        .map_err(compute_error)?;

        let peer_tokens = Arc::new(if peer_selection.selection.is_async_allocated() { peer_selection.selection.copy_to_stable_deferred().map_err(compute_error)? } else { peer_selection.selection });
        let peer_scores = Arc::new(if peer_selection.scores.is_async_allocated() { peer_selection.scores.copy_to_stable_deferred().map_err(compute_error)? } else { peer_selection.scores });
        let copied = ops::hip::DeviceBuffer::copy_stable_group_to_device_ordered_async_retained_by(&[peer_tokens, peer_scores], owner.device_id, owner.device_id)
            .map_err(|error| compute_error(format!("L{layer} DSA sequence shard candidates peer->owner: {error}")))?;
        let [peer_tokens, peer_scores]: [ops::hip::DeviceBuffer; 2] = copied.try_into().map_err(|_| compute_error(format!("L{layer} DSA peer candidate copy 数量异常")))?;
        let peer_selection = ops::hip::DsaSequenceShardSelection { selection: peer_tokens, scores: peer_scores, width: self.top_k };
        let selection = ops::hip::try_dsa_merge_sequence_shard_topk(owner.device_id, &owner_selection, &peer_selection, query_rows, self.top_k).map_err(compute_error)?;
        self.selection = Some(Arc::new(selection));
        self.selection_host = None;
        self.selection_rows = query_rows;
        self.selection_start = query_start;
        self.selection_width = self.top_k;
        if let Some(peer_state) = self.cooperative_peer.as_mut() {
            peer_state.selection = None;
            peer_state.selection_rows = 0;
            peer_state.selection_start = 0;
        }
        ops::hip::order_stream_after(owner.device_id, 0, owner_stream).map_err(compute_error)?;
        ops::hip::activate_compute_stream(owner.device_id, owner_stream).map_err(compute_error)?;
        Ok(true)
    }

    fn select_prefill_cooperative_impl(&mut self, owner: &RocmContext, peer: &RocmContext, layer: usize, query_source: RocmCooperativeDsaQuery<'_>, head_weights: &RocmTensor) -> Result<bool, BackendError> {
        let context_rows = self.layers.get(layer).and_then(Option::as_ref).map_or(0, |cached| cached.rows);
        let (query_rows, query_cols) = match &query_source {
            RocmCooperativeDsaQuery::Projected(query) => (query.rows, query.cols),
            RocmCooperativeDsaQuery::QLora { input, owner_wq_b, peer_wq_b, .. } => {
                if owner_wq_b.cols() != input.cols || peer_wq_b.cols() != input.cols || owner_wq_b.rows() != peer_wq_b.rows() || owner_wq_b.cols() != peer_wq_b.cols() {
                    return Err(compute_error(format!(
                        "L{layer} cooperative DSA wq_b/input shape 非法: input=[{},{}] owner=[{},{}] peer=[{},{}]",
                        input.rows,
                        input.cols,
                        owner_wq_b.rows(),
                        owner_wq_b.cols(),
                        peer_wq_b.rows(),
                        peer_wq_b.cols(),
                    )));
                }
                (input.rows, owner_wq_b.rows())
            }
        };
        if query_rows != head_weights.rows || query_cols == 0 || !query_cols.is_multiple_of(self.head_dim) {
            return Err(compute_error(format!("L{layer} cooperative DSA query=[{query_rows},{query_cols}] weights=[{},{}] head_dim={} 非法", head_weights.rows, head_weights.cols, self.head_dim,)));
        }
        let head_count = query_cols / self.head_dim;
        if head_weights.cols != head_count {
            return Err(compute_error(format!("L{layer} cooperative DSA head weights cols={}，期望 {head_count}", head_weights.cols)));
        }
        if self.ownership == RocmDsaOwnership::BlockParity {
            return self.select_sequence_sharded_impl(owner, peer, layer, query_source, head_weights, query_rows, head_count);
        }
        if ops::hip::options().prefill_attention_cpu
            || self.kpool != 0
            || self.hadamard_i8
            || query_rows < 64
            || !query_rows.is_multiple_of(2)
            || context_rows <= self.top_k
            || context_rows.checked_sub(query_rows).is_none_or(|start| start < self.top_k)
        {
            return Ok(false);
        }
        self.drain_gpu_host_selection()?;
        // scheduler 已经为当前 submission 选好 latency/default 或 background
        // stage stream；这里只切 device，不能把它重置成 context 默认流。
        ops::hip::set_device(owner.device_id).map_err(compute_error)?;
        let owner_stream = ops::hip::active_compute_stream() as usize;
        ops::hip::order_stream_after(owner.device_id, owner_stream, 0).map_err(compute_error)?;
        ops::hip::activate_compute_stream(owner.device_id, 0).map_err(compute_error)?;
        self.synchronize_cooperative_peer(owner, peer, layer)?;

        let half_rows = query_rows / 2;
        let query_start = context_rows - query_rows;
        let first_context_rows = query_start + half_rows;
        ops::hip::activate_compute_stream(owner.device_id, 0).map_err(compute_error)?;
        let owner_weights = crate::backend::SegmentedTensorBackend::slice_token_rows(owner, head_weights, 0, half_rows)?;
        let peer_weights = crate::backend::SegmentedTensorBackend::slice_token_rows(owner, head_weights, half_rows, half_rows)?;
        let (owner_input, peer_input) = match &query_source {
            RocmCooperativeDsaQuery::Projected(query) => (crate::backend::SegmentedTensorBackend::slice_token_rows(owner, query, 0, half_rows)?, crate::backend::SegmentedTensorBackend::slice_token_rows(owner, query, half_rows, half_rows)?),
            RocmCooperativeDsaQuery::QLora { input, .. } => {
                // CT GEMM 本来就把 activation 压成 BF16；提前一次压缩后再切行，
                // owner/peer 共享同一数值路径，同时把 P2P 从 F32 降为 BF16。
                let input = owner.tensor_as_bf16((*input).clone())?;
                (crate::backend::SegmentedTensorBackend::slice_token_rows(owner, &input, 0, half_rows)?, crate::backend::SegmentedTensorBackend::slice_token_rows(owner, &input, half_rows, half_rows)?)
            }
        };
        let peer_input = owner.tensor_to_stable_deferred(peer_input)?;
        let peer_weights = owner.tensor_to_stable_deferred(peer_weights)?;
        let sources = [
            peer_input.device.as_ref().ok_or_else(|| compute_error("cooperative DSA peer input 缺少 device buffer"))?.clone(),
            peer_weights.device.as_ref().ok_or_else(|| compute_error("cooperative DSA peer weights 缺少 device buffer"))?.clone(),
        ];
        peer.activate().map_err(compute_error)?;
        let (peer_input_buffer, peer_weights_buffer) = {
            let peer_state = self.cooperative_peer.as_mut().expect("cooperative DSA peer 已创建");
            if peer_state.query.as_ref().is_none_or(|buffer| buffer.bytes() != sources[0].bytes()) {
                peer_state.query = Some(Arc::new(ops::hip::DeviceBuffer::allocate(peer.device_id, sources[0].bytes()).map_err(compute_error)?));
            }
            if peer_state.head_weights.as_ref().is_none_or(|buffer| buffer.bytes() != sources[1].bytes()) {
                peer_state.head_weights = Some(Arc::new(ops::hip::DeviceBuffer::allocate(peer.device_id, sources[1].bytes()).map_err(compute_error)?));
            }
            (peer_state.query.as_ref().expect("cooperative DSA query scratch 已创建").clone(), peer_state.head_weights.as_ref().expect("cooperative DSA weight scratch 已创建").clone())
        };
        ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(&sources, &[(peer_input_buffer.as_ref(), 0), (peer_weights_buffer.as_ref(), 0)], peer.device_id, owner.device_id)
            .map_err(|error| compute_error(format!("L{layer} cooperative DSA query owner->peer: {error}")))?;
        let peer_input = super::device_tensor_with_arc(peer_input_buffer, half_rows, peer_input.cols, peer_input.dtype);
        let peer_weights = super::device_tensor_with_arc(peer_weights_buffer, half_rows, peer_weights.cols, peer_weights.dtype);
        let (owner_query, peer_query) = match query_source {
            RocmCooperativeDsaQuery::Projected(_) => (owner_input, peer_input),
            RocmCooperativeDsaQuery::QLora { owner_wq_b, peer_wq_b, position, cosine, sine, spec, .. } => {
                peer.activate().map_err(compute_error)?;
                let peer_query = peer.linear(&peer_input, peer_wq_b)?;
                let peer_query = peer.rope_prefix(&peer_query, spec.num_heads, spec.rope_dim, spec.rotary_layout, position + half_rows, cosine, sine)?;
                ops::hip::activate_compute_stream(owner.device_id, 0).map_err(compute_error)?;
                let owner_query = owner.linear(&owner_input, owner_wq_b)?;
                let owner_query = owner.rope_prefix(&owner_query, spec.num_heads, spec.rope_dim, spec.rotary_layout, position, cosine, sine)?;
                (owner_query, peer_query)
            }
        };

        let (peer_keys, peer_scales, peer_table) = {
            let peer_state = self.cooperative_peer.as_mut().expect("cooperative DSA peer 已创建");
            let cached = peer_state.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} cooperative DSA peer cache 缺失")))?;
            let table = peer_state.block_table.get("cooperative DSA", self.capacity, peer.device_id, context_rows)?;
            (cached.keys.clone(), cached.scales.clone(), table)
        };
        let peer_selection = ops::hip::try_dsa_select_paged_q8(
            peer.device_id,
            &peer_keys,
            &peer_scales,
            self.key_group_size,
            false,
            &peer_table,
            peer_query.device.as_deref().ok_or_else(|| compute_error("cooperative DSA peer query 缺少 device buffer"))?,
            peer_weights.device.as_deref().ok_or_else(|| compute_error("cooperative DSA peer weights 缺少 device buffer"))?,
            half_rows,
            context_rows,
            query_start + half_rows,
            head_count,
            self.head_dim,
            self.top_k,
            false,
            ROCM_KV_BLOCK_SIZE,
        )
        .map_err(compute_error)?;
        // DSA selection 在正常 memory-pool 路径本来就是显式 allocation，可直接
        // 作为 P2P source 交给 owner completion 保活；只有 async allocation
        // 才额外稳定化。否则原 selection 会在函数退出时先回到 peer 池。
        let stable_peer_selection = Arc::new(if peer_selection.is_async_allocated() { peer_selection.copy_to_stable_deferred().map_err(compute_error)? } else { peer_selection });
        let selection_bytes = half_rows.checked_mul(self.top_k).and_then(|elements| elements.checked_mul(4)).ok_or_else(|| compute_error("cooperative DSA selection 大小溢出"))?;
        let peer_full_selection = Arc::new(ops::hip::DeviceBuffer::allocate_reusable(peer.device_id, selection_bytes * 2).map_err(compute_error)?);
        peer_full_selection.copy_from_device(selection_bytes, &stable_peer_selection, 0, selection_bytes).map_err(compute_error)?;

        ops::hip::activate_compute_stream(owner.device_id, 0).map_err(compute_error)?;
        let owner_keys = self.layers[layer].as_ref().expect("cooperative DSA owner cache 已检查").keys.clone();
        let owner_scales = self.layers[layer].as_ref().expect("cooperative DSA owner cache 已检查").scales.clone();
        let owner_table = self.block_table.get("DSA", self.capacity, owner.device_id, first_context_rows)?;
        let owner_selection = ops::hip::try_dsa_select_paged_q8(
            owner.device_id,
            &owner_keys,
            &owner_scales,
            self.key_group_size,
            false,
            &owner_table,
            owner_query.device.as_deref().ok_or_else(|| compute_error("cooperative DSA owner query 缺少 device buffer"))?,
            owner_weights.device.as_deref().ok_or_else(|| compute_error("cooperative DSA owner weights 缺少 device buffer"))?,
            half_rows,
            first_context_rows,
            query_start,
            head_count,
            self.head_dim,
            self.top_k,
            false,
            ROCM_KV_BLOCK_SIZE,
        )
        .map_err(compute_error)?;
        let stable_owner_selection = Arc::new(if owner_selection.is_async_allocated() { owner_selection.copy_to_stable_deferred().map_err(compute_error)? } else { owner_selection });
        let selection = Arc::new(ops::hip::DeviceBuffer::allocate_reusable(owner.device_id, selection_bytes * 2).map_err(compute_error)?);
        selection.copy_from_device(0, &stable_owner_selection, 0, selection_bytes).map_err(compute_error)?;
        // Attention 按 query head 分卡，因此两卡都需要所有 query row 的
        // selection。DSA 末尾直接 all-gather 两个行半区，后续 Attention 不再
        // 把 owner 拼好的全量 selection 又传回 peer。
        ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(&[stable_owner_selection], &[(peer_full_selection.as_ref(), 0)], peer.device_id, owner.device_id)
            .map_err(|error| compute_error(format!("L{layer} cooperative DSA selection owner->peer: {error}")))?;
        ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(&[stable_peer_selection], &[(selection.as_ref(), selection_bytes)], owner.device_id, owner.device_id)
            .map_err(|error| compute_error(format!("L{layer} cooperative DSA selection peer->owner: {error}")))?;
        self.selection = Some(selection);
        self.selection_host = None;
        self.selection_rows = query_rows;
        self.selection_start = query_start;
        self.selection_width = self.top_k;
        let peer_state = self.cooperative_peer.as_mut().expect("cooperative DSA peer 已创建");
        peer_state.selection = Some(peer_full_selection);
        peer_state.selection_rows = query_rows;
        peer_state.selection_start = query_start;
        ops::hip::order_stream_after(owner.device_id, 0, owner_stream).map_err(compute_error)?;
        ops::hip::activate_compute_stream(owner.device_id, owner_stream).map_err(compute_error)?;
        Ok(true)
    }

    pub(super) fn select(&mut self, context: &RocmContext, layer: usize, query: &RocmTensor, head_weights: &RocmTensor) -> Result<(), BackendError> {
        self.select_begin(context, layer, query, head_weights)?;
        self.select_finish(context)
    }

    pub(super) fn select_prefill_cpu(&mut self, context: &RocmContext, layer: usize, query: &RocmTensor, head_weights: &RocmTensor) -> Result<(), BackendError> {
        if query.rows <= 1 || query.rows != head_weights.rows || !query.cols.is_multiple_of(self.head_dim) {
            return Err(compute_error(format!("L{layer} CPU prefill DSA query=[{},{}] weights=[{},{}] dim={} 非法", query.rows, query.cols, head_weights.rows, head_weights.cols, self.head_dim)));
        }
        let context_rows = self.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} CPU prefill DSA cache 尚未初始化")))?.rows;
        let query_start = context_rows.checked_sub(query.rows).ok_or_else(|| compute_error(format!("L{layer} CPU prefill DSA context={context_rows} 小于 query={}", query.rows)))?;
        self.drain_gpu_host_selection()?;
        if context_rows <= self.top_k {
            self.selection = None;
            self.selection_host = None;
            self.selection_rows = 0;
            self.selection_start = 0;
            self.selection_width = self.top_k;
            return Ok(());
        }
        let query_host = context.tensor_to_f32(query)?;
        let weights_host = context.tensor_to_f32(head_weights)?;
        let selection = {
            let mirror = self.layers[layer].as_ref().and_then(|cached| cached.cpu_mirror.as_ref()).ok_or_else(|| compute_error(format!("L{layer} CPU prefill DSA mirror 缺失")))?;
            let mut mirror = mirror.lock().map_err(|_| compute_error(format!("L{layer} CPU prefill DSA mirror 锁中毒")))?;
            mirror.finish_pending()?;
            select_prefill_cpu_q8(&mirror.record, self.head_dim, &query_host, &weights_host, query.rows, query_start, self.top_k).map_err(compute_error)?
        };
        let bytes = unsafe { std::slice::from_raw_parts(selection.as_ptr().cast::<u8>(), std::mem::size_of_val(selection.as_slice())) };
        // selection buffer 仅用于 stage/P2P 状态迁移；本层 attention 只读取 host selection。
        let buffer = Arc::new(ops::hip::DeviceBuffer::upload_ordered(context.device_id, bytes).map_err(compute_error)?);
        buffer.enqueue_deferred_upload().map_err(compute_error)?;
        buffer.retain_for_active_stage();
        self.selection = Some(buffer);
        self.selection_host = Some(Arc::new(selection));
        self.selection_rows = query.rows;
        self.selection_start = query_start;
        self.selection_width = self.top_k;
        Ok(())
    }

    pub(super) fn device_selection(&self, rows: usize, query_start: usize) -> Option<&ops::hip::DeviceBuffer> {
        (self.selection_rows == rows && self.selection_start == query_start).then_some(self.selection.as_deref()).flatten()
    }

    pub(super) fn device_selection_arc(&self, rows: usize, query_start: usize) -> Option<Arc<ops::hip::DeviceBuffer>> {
        (self.selection_rows == rows && self.selection_start == query_start).then(|| self.selection.clone()).flatten()
    }

    pub(super) fn cooperative_peer_selection_arc(&self, peer_device: i32, rows: usize, query_start: usize) -> Option<Arc<ops::hip::DeviceBuffer>> {
        self.cooperative_peer.as_ref().filter(|peer| peer.device_id == peer_device && peer.selection_rows == rows && peer.selection_start == query_start).and_then(|peer| peer.selection.clone())
    }

    pub(super) fn selection_width(&self) -> usize {
        self.selection_width
    }

    pub(super) fn host_selection(&self, rows: usize, query_start: usize) -> Option<&[u32]> {
        (self.selection_rows == rows && self.selection_start == query_start).then_some(self.selection_host.as_deref()).flatten().map(Vec::as_slice)
    }

    pub(crate) fn export_selection(&self) -> Option<RocmDsaSelection> {
        Some(RocmDsaSelection {
            buffer: self.selection.as_ref()?.clone(),
            host: self.selection_host.clone(),
            rows: self.selection_rows,
            start: self.selection_start,
            width: self.selection_width,
            ownership: RocmDsaSelectionOwnership::Global,
        })
    }

    pub(crate) fn import_selection(&mut self, context: &RocmContext, selection: Option<RocmDsaSelection>) -> Result<(), BackendError> {
        self.drain_gpu_host_selection()?;
        let Some(selection) = selection else {
            self.selection = None;
            self.selection_host = None;
            self.selection_rows = 0;
            self.selection_start = 0;
            self.selection_width = self.top_k;
            return Ok(());
        };
        if selection.ownership != RocmDsaSelectionOwnership::Global {
            return Err(compute_error(format!("ROCm DSA selection ownership={:?} 不能作为跨 stage 全局候选表", selection.ownership)));
        }
        let expected = selection.rows.checked_mul(selection.width).and_then(|n| n.checked_mul(std::mem::size_of::<u32>())).ok_or_else(|| compute_error("ROCm DSA selection P2P 大小溢出"))?;
        if selection.rows == 0 || selection.buffer.bytes() < expected {
            return Err(compute_error(format!("ROCm DSA selection P2P shape 非法: rows={} bytes={} expected={expected}", selection.rows, selection.buffer.bytes())));
        }
        // 边界接收的 selection 可能携带 deferred H2D（from_host 走 upload_ordered）：
        // 在首个消费 stage 的 stream 上提交上传并保活到 stage 完成；非 deferred buffer 为空操作。
        selection.buffer.enqueue_deferred_upload().map_err(compute_error)?;
        selection.buffer.retain_for_active_stage();
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

    #[test]
    fn dsa_parity_bytes_roundtrip_block_boundaries() {
        for rows in [1, 63, 64, 65, 127, 128, 129, 257] {
            let source = (0..rows * 3).map(|value| (value % 251) as u8).collect::<Vec<_>>();
            let owner = split_parity_bytes(&source, rows, 3, 0).unwrap();
            let peer = split_parity_bytes(&source, rows, 3, 1).unwrap();
            assert_eq!(owner.len() / 3, parity_rows(rows, 0));
            assert_eq!(peer.len() / 3, parity_rows(rows, 1));
            assert_eq!(merge_parity_bytes(&owner, &peer, rows, 3).unwrap(), source);
        }
    }

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

    #[test]
    fn dsa_cpu_mirror_appends_and_truncates_q8_rows() {
        let mut mirror = RocmDsaCpuMirror::new(8, 8, 4, false).unwrap();
        let first = [1_u8, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13];
        mirror.append_downloaded(0, 1, &first).unwrap();
        assert_eq!(mirror.record.rows, 1);
        assert_eq!(mirror.record.keys, first[..8]);
        assert_eq!(mirror.record.scales, first[8..]);

        let second = [21_u8, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33];
        mirror.append_downloaded(1, 1, &second).unwrap();
        assert_eq!(mirror.record.rows, 2);
        assert_eq!(&mirror.record.keys[8..], &second[..8]);
        assert_eq!(&mirror.record.scales[4..], &second[8..]);

        mirror.truncate(1, 8).unwrap();
        assert_eq!(mirror.record.rows, 1);
        assert_eq!(mirror.record.keys, first[..8]);
        assert_eq!(mirror.record.scales, first[8..]);
    }

    #[test]
    fn dsa_cpu_mirror_rejects_gaps_and_bad_download_size() {
        let mut mirror = RocmDsaCpuMirror::new(8, 8, 4, false).unwrap();
        assert!(mirror.append_downloaded(1, 1, &[0; 12]).is_err());
        assert!(mirror.append_downloaded(0, 1, &[0; 11]).is_err());
    }

    #[test]
    fn cpu_prefill_selection_obeys_each_row_causal_boundary() {
        let scale = half::bf16::from_f32(1.0).to_bits().to_ne_bytes();
        let record = DsaLayerSerde { rows: 4, key_group_size: 4, hadamard: false, keys: vec![1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0], scales: scale.repeat(4) };
        let query = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let selection = select_prefill_cpu_q8(&record, 4, &query, &[1.0, 1.0], 2, 2, 2).unwrap();
        assert_eq!(selection, [2, 0, 3, 1]);
    }

    #[test]
    #[ignore = "需要两张 ROCm GPU"]
    fn cooperative_dsa_query_rows_match_single_device_bits() {
        const CHUNK_ROWS: usize = 1024;
        const CHUNK_COUNT: usize = 4;
        const CONTEXT_ROWS: usize = CHUNK_COUNT * CHUNK_ROWS;
        const QUERY_ROWS: usize = CHUNK_ROWS;
        const HEAD_COUNT: usize = 32;
        const HEAD_DIM: usize = 128;
        const LORA_DIM: usize = 128;
        const ROPE_DIM: usize = 64;
        const TOP_K: usize = 2048;
        let Ok(owner) = RocmContext::new(0) else { return };
        let Ok(peer) = RocmContext::new(1) else { return };
        owner.activate().unwrap();
        let key_values = (0..CONTEXT_ROWS * HEAD_DIM).map(|index| ((index.wrapping_mul(29).wrapping_add(index / HEAD_DIM * 17)) % 509) as f32 * (1.0 / 127.0) - 2.0).collect::<Vec<_>>();
        let q_lora_values = (0..QUERY_ROWS * LORA_DIM).map(|index| ((index.wrapping_mul(31).wrapping_add(index / LORA_DIM * 11)) % 257) as f32 * (1.0 / 128.0) - 1.0).collect::<Vec<_>>();
        let wq_packed = (0..HEAD_COUNT * HEAD_DIM * LORA_DIM).map(|index| ((index.wrapping_mul(17).wrapping_add(index / LORA_DIM * 7)) % 127) as u8).collect::<Vec<_>>();
        let wq_scales = (0..HEAD_COUNT * HEAD_DIM).flat_map(|index| half::bf16::from_f32(2.0_f32.powi(-8 + (index % 3) as i32)).to_bits().to_ne_bytes()).collect::<Vec<_>>();
        let wq_matrix = crate::weight::format::quantization::W8A16Matrix::new(wq_packed, wq_scales, crate::weight::format::quantization::ScaleDType::Bf16, LORA_DIM, HEAD_COUNT * HEAD_DIM, LORA_DIM).unwrap();
        let weight_values = (0..QUERY_ROWS * HEAD_COUNT).map(|index| ((index % HEAD_COUNT) + 1) as f32 / HEAD_COUNT as f32).collect::<Vec<_>>();
        let q_lora = owner.tensor_from_f32(q_lora_values, QUERY_ROWS, LORA_DIM).unwrap();
        let wq = crate::backend::LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::W8A16(&wq_matrix));
        let owner_wq_b = crate::backend::BackendResources::prepare_weight(&owner, wq, HEAD_COUNT * HEAD_DIM, LORA_DIM).unwrap();
        let peer_wq_b = crate::backend::BackendResources::prepare_weight(&peer, wq, HEAD_COUNT * HEAD_DIM, LORA_DIM).unwrap();
        let weights = owner.tensor_from_f32(weight_values, QUERY_ROWS, HEAD_COUNT).unwrap();
        let cosine = (0..CONTEXT_ROWS * (ROPE_DIM / 2)).map(|index| ((index * 7 % 101) as f32 * 0.013).cos()).collect::<Vec<_>>();
        let sine = (0..CONTEXT_ROWS * (ROPE_DIM / 2)).map(|index| ((index * 7 % 101) as f32 * 0.013).sin()).collect::<Vec<_>>();
        let spec = crate::attention::dsa::DsaSpec { num_heads: HEAD_COUNT, head_dim: HEAD_DIM, rope_dim: ROPE_DIM, top_k: TOP_K, rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf, kpool: 0, always_select_tail: false };
        let mut reference = RocmDsaState::new(1, CONTEXT_ROWS, HEAD_DIM, TOP_K).unwrap();
        let mut cooperative = RocmDsaState::new(1, CONTEXT_ROWS, HEAD_DIM, TOP_K).unwrap();
        let mut projected = RocmDsaState::new(1, CONTEXT_ROWS, HEAD_DIM, TOP_K).unwrap();
        for chunk in 0..CHUNK_COUNT {
            let position = chunk * CHUNK_ROWS;
            let begin = position * HEAD_DIM;
            let end = begin + CHUNK_ROWS * HEAD_DIM;
            let keys = owner.tensor_from_f32(key_values[begin..end].to_vec(), CHUNK_ROWS, HEAD_DIM).unwrap();
            reference.append(&owner, 0, position, &keys).unwrap();
            cooperative.append(&owner, 0, position, &keys).unwrap();
            projected.append(&owner, 0, position, &keys).unwrap();
            if position < TOP_K {
                continue;
            }
            let compact_q_lora = owner.tensor_as_bf16(q_lora.clone()).unwrap();
            let query = owner.linear(&compact_q_lora, &owner_wq_b).unwrap();
            let query = owner.rope_prefix(&query, HEAD_COUNT, ROPE_DIM, spec.rotary_layout, position, &cosine, &sine).unwrap();
            reference.select(&owner, 0, &query, &weights).unwrap();
            assert!(cooperative.select_prefill_cooperative(&owner, &peer, 0, &query, &weights).unwrap());
            assert!(projected.select_prefill_cooperative_projected(&owner, &peer, 0, &q_lora, &owner_wq_b, &peer_wq_b, &weights, position, &cosine, &sine, &spec).unwrap());
            let expected = reference.export_selection().unwrap().to_host().unwrap();
            let actual = cooperative.export_selection().unwrap().to_host().unwrap();
            assert_eq!(actual, expected, "chunk={chunk}");
            let actual = projected.export_selection().unwrap().to_host().unwrap();
            assert_eq!(actual, expected, "projected chunk={chunk}");
            let peer_selection = cooperative.cooperative_peer_selection_arc(peer.device_id, QUERY_ROWS, position).expect("peer 全量 selection");
            let mut peer_actual = vec![0_u32; expected.len()];
            peer_selection.copy_to_host(unsafe { std::slice::from_raw_parts_mut(peer_actual.as_mut_ptr().cast(), peer_actual.len() * 4) }).unwrap();
            assert_eq!(peer_actual, expected, "peer chunk={chunk}");
            let peer_selection = projected.cooperative_peer_selection_arc(peer.device_id, QUERY_ROWS, position).expect("projected peer 全量 selection");
            peer_selection.copy_to_host(unsafe { std::slice::from_raw_parts_mut(peer_actual.as_mut_ptr().cast(), peer_actual.len() * 4) }).unwrap();
            assert_eq!(peer_actual, expected, "projected peer chunk={chunk}");
        }
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
            .enumerate()
            .map(|(layer, slot)| -> Result<Option<DsaLayerSerde>, BackendError> {
                let Some(cached) = slot.as_ref() else { return Ok(self.pending_pair_layers.get(layer).and_then(Option::as_ref).cloned()) };
                if self.ownership == RocmDsaOwnership::BlockParity {
                    let peer = self.cooperative_peer.as_ref().and_then(|state| state.layers.get(layer)).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} DSA peer parity shard 缺失")))?;
                    if peer.rows != cached.rows {
                        return Err(compute_error(format!("L{layer} DSA owner/peer rows={}/{} 不一致", cached.rows, peer.rows)));
                    }
                    let owner_rows = parity_rows(cached.rows, 0);
                    let peer_rows = parity_rows(cached.rows, 1);
                    let mut owner_keys = vec![0u8; owner_rows * head_dim];
                    let mut peer_keys = vec![0u8; peer_rows * head_dim];
                    cached.keys.copy_to_host(&mut owner_keys).map_err(compute_error)?;
                    peer.keys.copy_to_host(&mut peer_keys).map_err(compute_error)?;
                    let scale_row_bytes = groups_per_row * 2;
                    let mut owner_scales = vec![0u8; owner_rows * scale_row_bytes];
                    let mut peer_scales = vec![0u8; peer_rows * scale_row_bytes];
                    cached.scales.copy_to_host(&mut owner_scales).map_err(compute_error)?;
                    peer.scales.copy_to_host(&mut peer_scales).map_err(compute_error)?;
                    return Ok(Some(DsaLayerSerde {
                        rows: cached.rows,
                        key_group_size,
                        hadamard: false,
                        keys: merge_parity_bytes(&owner_keys, &peer_keys, cached.rows, head_dim)?,
                        scales: merge_parity_bytes(&owner_scales, &peer_scales, cached.rows, scale_row_bytes)?,
                    }));
                }
                if let Some(mirror) = &cached.cpu_mirror {
                    let mut mirror = mirror.lock().map_err(|_| compute_error(format!("L{layer} ROCm DSA CPU mirror 锁中毒")))?;
                    mirror.finish_pending()?;
                    if mirror.record.rows != cached.rows {
                        return Err(compute_error(format!("L{layer} ROCm DSA CPU mirror rows={}，GPU rows={}", mirror.record.rows, cached.rows)));
                    }
                    return Ok(Some(mirror.record.clone()));
                }
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
    pub fn upload_layers(&mut self, context: &RocmContext, layers: &[Option<DsaLayerSerde>], reserved_rows: usize, interleaved_pair: bool) -> Result<(), BackendError> {
        self.cooperative_peer = None;
        self.ownership = RocmDsaOwnership::Full;
        for slot in &mut self.layers {
            *slot = None;
        }
        self.pending_pair_layers.fill(None);
        self.pending_pair_reserved_rows = reserved_rows;
        self.invalidate_selection();
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
            // DSA 与同一 stage 的 KV 共享物理 ownership；恢复语义来自快照，
            // 不能依赖本次进程的性能配置。
            if interleaved_pair {
                self.pending_pair_layers[index] = Some(record.clone());
                continue;
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
                cpu_mirror: (ops::hip::options().mla_cpu_hot_rows != 0 || ops::hip::options().prefill_attention_cpu).then(|| std::sync::Mutex::new(RocmDsaCpuMirror::from_record(record.clone(), self.head_dim))),
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
        let owner = self
            .layers
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
            .sum::<u64>();
        let peer = self.cooperative_peer.as_ref().map_or(0, |state| state.layers.iter().filter_map(Option::as_ref).map(|cached| cached.keys.bytes() as u64 + cached.scales.bytes() as u64).sum());
        owner + peer
    }
}
