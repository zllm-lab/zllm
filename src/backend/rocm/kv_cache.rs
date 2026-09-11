use std::sync::{Arc, Mutex};

use crate::backend::cpu::CpuKvCache;
use crate::backend::{BackendError, compute_error};
use crate::kernel::rocm as ops;

use super::{ROCM_KV_BLOCK_SIZE, RocmBlockTable, RocmContext, RocmTensor, committed_cache_rows, grow_cache_buffer, upload_cache_buffer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocmKvOwnership {
    Full,
    BlockParity(u8),
    InterleavedPair,
}

/// `ZLLM_ROCM_MLA_CPU_MIRROR=1`：无条件维护 MLA 全量 CPU 镜像（Q8 原格式），
/// 不要求 prefill_attention_cpu / hot rows。镜像只增不消费，供 CPU KV 方案取证；
/// 每层每 token 增加约 3 个 D2D pack + 1 个异步 D2H，生产速度轮应保持关闭。
fn mla_cpu_mirror_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLLM_ROCM_MLA_CPU_MIRROR").is_ok_and(|value| value == "1"))
}

impl RocmKvOwnership {
    pub(super) fn parity(self) -> Option<usize> {
        match self {
            Self::BlockParity(parity @ 0..=1) => Some(parity as usize),
            _ => None,
        }
    }
}

/// `[0, rows)` 中归当前 parity 的物理行数。最后一个不完整 block 仍由固定
/// parity 持有，因此 token 增长只追加，不会改变任何已有行的物理地址。
fn parity_rows(rows: usize, parity: usize) -> usize {
    let blocks = rows / ROCM_KV_BLOCK_SIZE;
    let tail = rows % ROCM_KV_BLOCK_SIZE;
    let full = (blocks / 2) * ROCM_KV_BLOCK_SIZE;
    full + usize::from(blocks % 2 > parity) * ROCM_KV_BLOCK_SIZE + usize::from(blocks % 2 == parity) * tail
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

fn mla_q8_row_bytes(latent_cols: usize, rope_cols: usize, group_size: usize) -> Option<(usize, usize, usize)> {
    if group_size == 0 || !latent_cols.is_multiple_of(group_size) {
        return None;
    }
    Some((latent_cols, (latent_cols / group_size).checked_mul(2)?, rope_cols.checked_mul(2)?))
}

fn split_pair_bytes<'a>(bytes: &'a [u8], owner_rows: usize, peer_rows: usize, row_bytes: usize, label: &str) -> Result<(&'a [u8], &'a [u8]), BackendError> {
    let boundary = owner_rows.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("pair restore {label} boundary 溢出")))?;
    let expected = owner_rows.checked_add(peer_rows).and_then(|rows| rows.checked_mul(row_bytes)).ok_or_else(|| compute_error(format!("pair restore {label} 大小溢出")))?;
    if bytes.len() != expected {
        return Err(compute_error(format!("pair restore {label} bytes={}，期望 {expected}", bytes.len())));
    }
    Ok(bytes.split_at(boundary))
}

fn truncate_pair_record(record: &mut MlaLayerSerde, rows: usize) -> Result<(), BackendError> {
    if record.ownership != RocmKvOwnership::InterleavedPair || rows > record.rows {
        return Err(compute_error(format!("pair KV truncate rows={rows} record_rows={} ownership={:?} 非法", record.rows, record.ownership)));
    }
    let old_owner = parity_rows(record.rows, 0);
    let new_owner = parity_rows(rows, 0);
    let new_peer = parity_rows(rows, 1);
    let truncate = |bytes: &mut Vec<u8>, row_bytes: usize| {
        let peer = bytes.split_off(old_owner * row_bytes);
        bytes.truncate(new_owner * row_bytes);
        bytes.extend_from_slice(&peer[..new_peer * row_bytes]);
    };
    let latent_row_bytes = record.latent_cols * if record.latent_group_size == 0 { 2 } else { 1 };
    truncate(&mut record.latent, latent_row_bytes);
    if let Some(scales) = &mut record.latent_scales {
        truncate(scales, record.latent_cols / record.latent_group_size * 2);
    }
    truncate(&mut record.rope, record.rope_cols * 2);
    record.rows = rows;
    Ok(())
}

/// cooperative snapshot 按 parity0、parity1 连续保存。operator pair 两卡都
/// 需要完整逻辑顺序，因此恢复时按 64-token block 重新交织。
fn join_pair_bytes(bytes: &[u8], rows: usize, row_bytes: usize, label: &str) -> Result<Vec<u8>, BackendError> {
    let owner_rows = parity_rows(rows, 0);
    let peer_rows = parity_rows(rows, 1);
    let (owner, peer) = split_pair_bytes(bytes, owner_rows, peer_rows, row_bytes, label)?;
    let output_bytes = rows.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("pair join {label} 大小溢出")))?;
    let mut output = vec![0_u8; output_bytes];
    for logical in 0..rows {
        let parity = (logical / ROCM_KV_BLOCK_SIZE) & 1;
        let source = if parity == 0 { owner } else { peer };
        let source_offset = parity_row(logical) * row_bytes;
        let target_offset = logical * row_bytes;
        output[target_offset..target_offset + row_bytes].copy_from_slice(&source[source_offset..source_offset + row_bytes]);
    }
    Ok(output)
}

fn join_pair_record(mut record: MlaLayerSerde, layer: usize) -> Result<MlaLayerSerde, BackendError> {
    if record.ownership != RocmKvOwnership::InterleavedPair {
        return Err(compute_error(format!("L{layer} pair join ownership={:?} 非法", record.ownership)));
    }
    let latent_row_bytes = record.latent_cols * if record.latent_group_size == 0 { 2 } else { 1 };
    let scale_row_bytes = if record.latent_group_size == 0 { 0 } else { record.latent_cols / record.latent_group_size * 2 };
    let rope_row_bytes = record.rope_cols * 2;
    record.latent = join_pair_bytes(&record.latent, record.rows, latent_row_bytes, &format!("L{layer} latent"))?;
    record.latent_scales = record.latent_scales.as_deref().map(|bytes| join_pair_bytes(bytes, record.rows, scale_row_bytes, &format!("L{layer} scales"))).transpose()?;
    record.rope = join_pair_bytes(&record.rope, record.rows, rope_row_bytes, &format!("L{layer} rope"))?;
    record.ownership = RocmKvOwnership::Full;
    Ok(record)
}

pub(super) struct RocmPagedMlaLayer {
    pub(super) latent: Arc<ops::hip::DeviceBuffer>,
    pub(super) latent_scales: Option<Arc<ops::hip::DeviceBuffer>>,
    pub(super) latent_group_size: usize,
    pub(super) rope: Arc<ops::hip::DeviceBuffer>,
    pub(super) rows: usize,
    pub(super) latent_cols: usize,
    pub(super) rope_cols: usize,
    pub(super) committed_rows: usize,
    cpu_mirror: Option<std::sync::Mutex<RocmMlaCpuMirror>>,
    pub(super) cpu_hot: Option<std::sync::Mutex<RocmMlaCpuHotLayer>>,
}

fn truncate_hot_layer(cached: &mut RocmPagedMlaLayer, rows: usize) -> Result<(), BackendError> {
    cached.cpu_hot.as_ref().expect("调用前已校验 hot").lock().map_err(|_| compute_error("ROCm hot truncate 锁中毒"))?.truncate(rows)?;
    if rows <= cached.committed_rows {
        // 大幅回退后恢复普通 paged 前缀；短上下文可能不再产生 DSA selection，
        // 不能把任意热槽布局当作连续 token 布局读取。
        let mut hot = cached.cpu_hot.take().expect("hot 已校验").into_inner().map_err(|_| compute_error("ROCm hot restore 锁中毒"))?;
        hot.gpu = None;
        if rows != 0 {
            cached.latent.copy_from_host(&hot.mirror.latent).map_err(compute_error)?;
            cached.latent_scales.as_ref().expect("hot 必有 scales").copy_from_host(hot.mirror.latent_scales.as_ref().expect("hot mirror 必有 scales")).map_err(compute_error)?;
            cached.rope.copy_from_host(&hot.mirror.rope).map_err(compute_error)?;
        }
        cached.cpu_mirror = Some(Mutex::new(RocmMlaCpuMirror::from_record(hot.mirror)));
    }
    Ok(())
}

pub struct RocmKvCache {
    cpu: CpuKvCache,
    pub(super) paged_layers: Vec<Option<RocmPagedMlaLayer>>,
    /// GQA 层的常驻设备 K/V（f32，行主序 [rows][kv_heads*head_dim]）。
    /// full-attention prefill/decode 只读设备缓存；需要 CPU fallback 的窗口路径才双写 host。
    pub(super) gqa_layers: Vec<Option<RocmGqaLayer>>,
    capacity: usize,
    /// stage 显式区分小块 prefill 和 MTP verify；非 stage 调用仍按行数兼容。
    stage_decode: Option<bool>,
    pub(super) block_table: RocmBlockTable,
    /// cooperative append 先把当前 chunk 量化到连续 staging；它的 identity
    /// table 生命周期与持续增长的 parity cache table 不同，不能共用。
    cooperative_staging_table: RocmBlockTable,
    /// Full 为普通单卡；pair 模式下 owner=0、peer=1，设备 buffer 只保存本
    /// parity 的 compact block，`RocmPagedMlaLayer::rows` 仍是全局逻辑长度。
    pub(super) ownership: RocmKvOwnership,
    cooperative_peer: Option<RocmCooperativeKvPeer>,
    /// operator pair 保留两张卡各自完整的 KV history；peer worker 只在提交
    /// 期间短暂持锁，GPU 完成关系仍由 stream/event 表达。
    operator_peer: Option<RocmOperatorKvPeer>,
    /// SSD restore 的 pair 记录先留在 host；首次拿到 peer context 时直接分别
    /// 上传 parity0/1，避免先在 owner 重建全量再拆分造成峰值 OOM。
    pending_pair_layers: Vec<Option<MlaLayerSerde>>,
    pending_pair_reserved_rows: usize,
    /// IndexShare 层复用最近一次 full-indexer 的 selection；peer 镜像也只需
    /// 传一次，不能在随后三层重复搬同一份 16MiB 索引。
    cooperative_selection: Option<RocmCooperativeSelection>,
    /// operator pair 不拆 selection，只缓存一份 peer 全量镜像；同一个
    /// IndexShare Arc 在后继层直接复用。
    operator_selection: Option<RocmOperatorSelection>,
}

struct RocmCooperativeKvPeer {
    device_id: i32,
    cache: Box<RocmKvCache>,
}

struct RocmOperatorKvPeer {
    device_id: i32,
    cache: Arc<Mutex<RocmKvCache>>,
}

struct RocmCooperativeSelection {
    peer_device: i32,
    source: Arc<ops::hip::DeviceBuffer>,
    pub(super) owner: Arc<ops::hip::DeviceBuffer>,
    pub(super) owner_counts: Arc<ops::hip::DeviceBuffer>,
    pub(super) peer: Arc<ops::hip::DeviceBuffer>,
    pub(super) peer_counts: Arc<ops::hip::DeviceBuffer>,
    pub(super) width: usize,
}

struct RocmOperatorSelection {
    peer_device: i32,
    source: Arc<ops::hip::DeviceBuffer>,
    peer: Arc<ops::hip::DeviceBuffer>,
}

pub(super) struct RocmGqaLayer {
    pub(super) key: Arc<ops::hip::DeviceBuffer>,
    pub(super) value: Arc<ops::hip::DeviceBuffer>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    committed_rows: usize,
}

impl RocmKvCache {
    pub fn new(layer_count: usize) -> Self {
        Self::with_capacity(layer_count, 4096)
    }

    pub fn with_capacity(layer_count: usize, capacity: usize) -> Self {
        Self {
            cpu: CpuKvCache::new(layer_count),
            paged_layers: (0..layer_count).map(|_| None).collect(),
            gqa_layers: (0..layer_count).map(|_| None).collect(),
            capacity,
            stage_decode: None,
            block_table: RocmBlockTable::new(),
            cooperative_staging_table: RocmBlockTable::new(),
            ownership: RocmKvOwnership::Full,
            cooperative_peer: None,
            operator_peer: None,
            pending_pair_layers: (0..layer_count).map(|_| None).collect(),
            pending_pair_reserved_rows: 0,
            cooperative_selection: None,
            operator_selection: None,
        }
    }

    pub(crate) fn prepare_block_table(&mut self, context: &RocmContext) -> Result<(), BackendError> {
        self.block_table.get("KV", self.capacity, context.device_id, self.capacity).map(|_| ())
    }

    pub(super) fn set_stage_decode(&mut self, decode: bool) -> Result<(), BackendError> {
        if self.stage_decode != Some(decode) {
            if let Some(peer) = &self.operator_peer {
                peer.cache.lock().map_err(|_| compute_error("ROCm operator KV phase 锁中毒"))?.set_stage_decode(decode)?;
            }
            self.stage_decode = Some(decode);
        }
        Ok(())
    }

    /// 大块 append 先补齐 recent 日志，再恢复连续 GPU 前缀。全部上传成功后
    /// 才替换热窗；分配失败时仍保留完整的旧 cache，允许上层回收资源后重试。
    pub(super) fn prepare_mla_append(&mut self, context: &RocmContext, layer: usize, rows: usize) -> Result<(), BackendError> {
        if rows <= MLA_HOT_LOG_ROWS || self.stage_decode == Some(true) || ops::hip::options().prefill_attention_cpu {
            return Ok(());
        }
        let Some(cached) = self.paged_layers.get_mut(layer).and_then(Option::as_mut) else { return Ok(()) };
        let Some(hot_mutex) = &mut cached.cpu_hot else { return Ok(()) };
        let end = cached.rows.checked_add(rows).ok_or_else(|| compute_error(format!("L{layer} MLA prefill restore rows 溢出")))?;
        let committed = committed_cache_rows(0, end, self.capacity)?;
        let started = std::time::Instant::now();
        let hot = hot_mutex.get_mut().map_err(|_| compute_error(format!("L{layer} MLA prefill restore 锁中毒")))?;
        hot.join_prefetch(context.device_id)?;
        hot.flush_log_feed(context.device_id)?;
        hot.finish_pending()?;
        if hot.mirror.rows != cached.rows {
            return Err(compute_error(format!("L{layer} MLA prefill restore mirror={} cache={}", hot.mirror.rows, cached.rows)));
        }
        let scale_row_bytes = cached.latent_cols / cached.latent_group_size * 2;
        let latent = upload_cache_buffer(context.device_id, &hot.mirror.latent, committed * cached.latent_cols)?;
        let scales = upload_cache_buffer(context.device_id, hot.mirror.latent_scales.as_deref().expect("hot 必有 scales"), committed * scale_row_bytes)?;
        let rope = upload_cache_buffer(context.device_id, &hot.mirror.rope, committed * cached.rope_cols * 2)?;
        let mut hot = cached.cpu_hot.take().expect("hot 已校验").into_inner().map_err(|_| compute_error(format!("L{layer} MLA prefill restore 锁中毒")))?;
        // 元数据指向旧热槽，不能在下一轮 decode 沿用。注册 guard 先于 mirror 交接释放。
        hot.gpu = None;
        let mut mirror = RocmMlaCpuMirror::from_record(hot.mirror);
        mirror.transfer = Some(hot.transfer);
        mirror.available_download = hot.available_download;
        cached.latent = latent;
        cached.latent_scales = Some(scales);
        cached.rope = rope;
        cached.committed_rows = committed;
        cached.cpu_mirror = Some(Mutex::new(mirror));
        eprintln!("[mla-gpu-prefill-restore] device={} layer={layer} history={} append={rows} committed={committed} wall_ms={:.3}", context.device_id, cached.rows, started.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    /// 把 f32 设备 K/V 行追加到该层常驻缓冲；调用方负责先确保 tensor 为 f32 resident。
    pub(super) fn append_gqa_rows(&mut self, context: &RocmContext, layer: usize, key: &RocmTensor, value: &RocmTensor) -> Result<(), BackendError> {
        if key.rows == 0 || key.rows != value.rows || key.cols != value.cols {
            return Err(compute_error(format!("ROCm GQA append L{layer} 行列不一致: K=[{},{}] V=[{},{}]", key.rows, key.cols, value.rows, value.cols)));
        }
        let key_input = key.device.as_deref().ok_or_else(|| compute_error("ROCm GQA append key 缺少 device buffer"))?;
        let value_input = value.device.as_deref().ok_or_else(|| compute_error("ROCm GQA append value 缺少 device buffer"))?;
        let slot = self.gqa_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        let rows = slot.as_ref().map_or(0, |cached| cached.rows);
        let cols = key.cols;
        let end = rows.checked_add(key.rows).ok_or_else(|| compute_error("ROCm GQA rows 溢出"))?;
        if end > self.capacity {
            return Err(compute_error(format!("ROCm GQA L{layer} end={end} 超过逻辑上限 {}", self.capacity)));
        }
        let committed = slot.as_ref().map_or(0, |cached| cached.committed_rows);
        let next_committed = committed_cache_rows(committed, end, self.capacity)?;
        if slot.is_none() {
            *slot = Some(RocmGqaLayer {
                key: Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, next_committed.saturating_mul(cols).saturating_mul(4)).map_err(compute_error)?),
                value: Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, next_committed.saturating_mul(cols).saturating_mul(4)).map_err(compute_error)?),
                rows: 0,
                cols,
                committed_rows: next_committed,
            });
        }
        let cached = slot.as_mut().expect("刚创建");
        if next_committed > cached.committed_rows {
            let old = cached.rows.saturating_mul(cached.cols).saturating_mul(4);
            let new = next_committed.saturating_mul(cached.cols).saturating_mul(4);
            cached.key = grow_cache_buffer(context.device_id, &cached.key, old, new)?;
            cached.value = grow_cache_buffer(context.device_id, &cached.value, old, new)?;
            cached.committed_rows = next_committed;
        }
        let elements = key.rows.saturating_mul(cols);
        ops::hip::try_gqa_cache_append_f32(context.device_id, key_input, &cached.key, rows.saturating_mul(cols), elements).map_err(compute_error)?;
        ops::hip::try_gqa_cache_append_f32(context.device_id, value_input, &cached.value, rows.saturating_mul(cols), elements).map_err(compute_error)?;
        cached.rows = end;
        Ok(())
    }

    /// 该层设备 K/V 与统计信息（decode kernel 输入）。
    pub(super) fn gqa_layer_buffers(&self, layer: usize) -> Result<(ops::hip::DeviceBuffer, ops::hip::DeviceBuffer, usize, usize), BackendError> {
        let cached = self.gqa_layers.get(layer).and_then(|slot| slot.as_ref()).ok_or(BackendError::UnsupportedLayer { layer })?;
        let bytes = cached.rows.checked_mul(cached.cols).and_then(|elements| elements.checked_mul(4)).ok_or_else(|| compute_error("ROCm GQA cache view 大小溢出"))?;
        let key = ops::hip::DeviceBuffer::view(cached.key.clone(), 0, bytes).map_err(compute_error)?;
        let value = ops::hip::DeviceBuffer::view(cached.value.clone(), 0, bytes).map_err(compute_error)?;
        Ok((key, value, cached.rows, cached.cols))
    }

    /// 只回退逻辑长度；已分配 buffer 与 prompt 前缀原地保留，后续 append 覆盖尾部。
    pub fn truncate_rows(&mut self, rows: usize) -> Result<(), BackendError> {
        if rows > self.capacity {
            return Err(compute_error(format!("ROCm KV truncate rows={rows} 超过逻辑上限 {}", self.capacity)));
        }
        for (layer, cached) in self.paged_layers.iter_mut().enumerate().filter_map(|(layer, slot)| slot.as_mut().map(|cached| (layer, cached))) {
            if rows > cached.rows {
                return Err(compute_error(format!("L{layer} ROCm KV truncate rows={rows} 超过当前长度 {}", cached.rows)));
            }
            if rows < cached.rows {
                if cached.cpu_hot.is_some() {
                    truncate_hot_layer(cached, rows)?;
                } else if let Some(mirror) = &cached.cpu_mirror {
                    mirror.lock().map_err(|_| compute_error(format!("L{layer} ROCm MLA mirror 锁中毒")))?.truncate(rows)?;
                }
                cached.rows = rows;
            }
        }
        for (layer, cached) in self.gqa_layers.iter_mut().enumerate().filter_map(|(layer, slot)| slot.as_mut().map(|cached| (layer, cached))) {
            if rows > cached.rows {
                return Err(compute_error(format!("L{layer} ROCm GQA truncate rows={rows} 超过当前长度 {}", cached.rows)));
            }
            cached.rows = rows;
        }
        for record in self.pending_pair_layers.iter_mut().filter_map(Option::as_mut) {
            truncate_pair_record(record, rows)?;
        }
        if let Some(peer) = self.cooperative_peer.as_mut() {
            peer.cache.truncate_replica_rows(rows)?;
        }
        if let Some(peer) = self.operator_peer.as_ref() {
            peer.cache.lock().map_err(|_| compute_error("ROCm operator peer KV 锁中毒"))?.truncate_replica_rows(rows)?;
        }
        if let Some(cached) = self.paged_layers.iter().flatten().next() {
            ops::hip::set_device(cached.latent.device_id()).map_err(compute_error)?;
        }
        Ok(())
    }

    /// 只回退指定层；主干与 speculative 辅助层长度不同时不能整库截断。
    pub fn truncate_layer_rows(&mut self, layer: usize, rows: usize) -> Result<(), BackendError> {
        if rows > self.capacity {
            return Err(compute_error(format!("L{layer} ROCm KV truncate rows={rows} 超过逻辑上限 {}", self.capacity)));
        }
        let paged = self.paged_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if let Some(cached) = paged {
            if rows > cached.rows {
                return Err(compute_error(format!("L{layer} ROCm KV truncate rows={rows} 超过当前长度 {}", cached.rows)));
            }
            if rows < cached.rows {
                if cached.cpu_hot.is_some() {
                    truncate_hot_layer(cached, rows)?;
                } else if let Some(mirror) = &cached.cpu_mirror {
                    mirror.lock().map_err(|_| compute_error(format!("L{layer} ROCm MLA mirror 锁中毒")))?.truncate(rows)?;
                }
                cached.rows = rows;
            }
        }
        let gqa = self.gqa_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if let Some(cached) = gqa {
            if rows > cached.rows {
                return Err(compute_error(format!("L{layer} ROCm GQA truncate rows={rows} 超过当前长度 {}", cached.rows)));
            }
            cached.rows = rows;
        }
        if let Some(record) = self.pending_pair_layers.get_mut(layer).and_then(Option::as_mut) {
            truncate_pair_record(record, rows)?;
        }
        if let Some(peer) = self.cooperative_peer.as_mut() {
            peer.cache.truncate_replica_layer_rows(layer, rows)?;
        }
        if let Some(peer) = self.operator_peer.as_ref() {
            peer.cache.lock().map_err(|_| compute_error("ROCm operator peer KV 锁中毒"))?.truncate_replica_layer_rows(layer, rows)?;
        }
        if let Some(cached) = self.paged_layers.get(layer).and_then(Option::as_ref) {
            ops::hip::set_device(cached.latent.device_id()).map_err(compute_error)?;
        }
        Ok(())
    }

    pub(super) fn mla_rows(&self, layer: usize) -> usize {
        self.paged_layers.get(layer).and_then(Option::as_ref).map_or_else(|| self.pending_pair_layers.get(layer).and_then(Option::as_ref).map_or(0, |record| record.rows), |cached| cached.rows)
    }

    pub(super) fn with_cpu_mla_history<R>(&mut self, layer: usize, consume: impl FnOnce(&MlaLayerSerde) -> Result<R, BackendError>) -> Result<R, BackendError> {
        let cached = self.paged_layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| compute_error(format!("L{layer} ROCm MLA cache 尚未初始化")))?;
        let mirror = cached.cpu_mirror.as_mut().ok_or_else(|| compute_error(format!("L{layer} ROCm MLA CPU mirror 未启用")))?;
        let mut mirror = mirror.lock().map_err(|_| compute_error(format!("L{layer} ROCm MLA mirror 锁中毒")))?;
        mirror.flush_feed(cached.latent.device_id(), cached.rows, &cached.latent, cached.latent_scales.as_deref().expect("Q8 MLA 必有 scales"), &cached.rope)?;
        if mirror.mirror.rows != cached.rows {
            return Err(compute_error(format!("L{layer} ROCm MLA CPU mirror rows={}，设备 cache rows={}", mirror.mirror.rows, cached.rows)));
        }
        consume(&mirror.mirror)
    }

    pub(super) fn append_mla(&mut self, context: &RocmContext, layer: usize, latent: &RocmTensor, rope: &RocmTensor) -> Result<(), BackendError> {
        self.append_mla_inner(context, layer, latent, rope, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn append_mla_rope(
        &mut self,
        context: &RocmContext,
        layer: usize,
        latent: &RocmTensor,
        rope: &RocmTensor,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        position: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(), BackendError> {
        self.append_mla_inner(context, layer, latent, rope, Some((position, rotary_dim, layout, cos, sin)))
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_parity_layer(cache: &mut Self, context: &RocmContext, layer: usize, global_end: usize, latent_cols: usize, rope_cols: usize, latent_group_size: usize) -> Result<(), BackendError> {
        let parity = cache.ownership.parity().ok_or_else(|| compute_error(format!("L{layer} cooperative KV ownership={:?} 非法", cache.ownership)))?;
        let required_rows = parity_rows(global_end, parity);
        let physical_limit = parity_rows(cache.capacity, parity);
        let slot = cache.paged_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        let current_committed = slot.as_ref().map_or(0, |cached| cached.committed_rows);
        let committed_rows = committed_cache_rows(current_committed, required_rows.max(1), physical_limit.max(1))?;
        let latent_row_bytes = latent_cols * if latent_group_size == 0 { 2 } else { 1 };
        let scale_row_bytes = if latent_group_size == 0 { 0 } else { latent_cols / latent_group_size * 2 };
        let rope_row_bytes = rope_cols * 2;
        if slot.is_none() {
            *slot = Some(RocmPagedMlaLayer {
                latent: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, committed_rows * latent_row_bytes).map_err(compute_error)?),
                latent_scales: if latent_group_size == 0 { None } else { Some(Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, committed_rows * scale_row_bytes).map_err(compute_error)?)) },
                latent_group_size,
                rope: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, committed_rows * rope_row_bytes).map_err(compute_error)?),
                rows: 0,
                latent_cols,
                rope_cols,
                committed_rows,
                cpu_mirror: None,
                cpu_hot: None,
            });
        }
        let cached = slot.as_mut().expect("刚创建 cooperative KV layer");
        if cached.latent_cols != latent_cols || cached.rope_cols != rope_cols || cached.latent_group_size != latent_group_size {
            return Err(compute_error(format!("L{layer} cooperative KV shape 非法: latent={}/{} rope={}/{} group={}/{}", cached.latent_cols, latent_cols, cached.rope_cols, rope_cols, cached.latent_group_size, latent_group_size,)));
        }
        if committed_rows > cached.committed_rows {
            let used_rows = parity_rows(cached.rows, parity);
            cached.latent = grow_cache_buffer(context.device_id, &cached.latent, used_rows * latent_row_bytes, committed_rows * latent_row_bytes)?;
            if let Some(scales) = &cached.latent_scales {
                cached.latent_scales = Some(grow_cache_buffer(context.device_id, scales, used_rows * scale_row_bytes, committed_rows * scale_row_bytes)?);
            }
            cached.rope = grow_cache_buffer(context.device_id, &cached.rope, used_rows * rope_row_bytes, committed_rows * rope_row_bytes)?;
            cached.committed_rows = committed_rows;
        }
        cache.block_table.get("KV cooperative parity", physical_limit.max(1), context.device_id, required_rows.max(1))?;
        Ok(())
    }

    /// decode 单行快速路径：该行只属于一个 parity 半片，融合 kernel 直写目标
    /// cache 的最终行——零 staging、零 parity 拷贝、零 P2P。parity=1 时 kernel 在
    /// owner 流上跨卡直写 peer 显存；对端可见性由 attention 段随后的
    /// owner->peer 传输的 event 链天然覆盖（该传输在 owner 流上排在本 kernel 之后）。
    #[allow(clippy::too_many_arguments)]
    fn append_cooperative_mla_rope_single(
        &mut self,
        owner: &RocmContext,
        peer: &RocmContext,
        layer: usize,
        position: usize,
        latent: &RocmTensor,
        rope: &RocmTensor,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        latent_group_size: usize,
        cos: &[f32],
        sin: &[f32],
        end: usize,
    ) -> Result<(), BackendError> {
        let parity = (position / ROCM_KV_BLOCK_SIZE) & 1;
        let local_row = parity_rows(position, parity);
        owner.activate().map_err(compute_error)?;
        Self::ensure_parity_layer(self, owner, layer, end, latent.cols, rope.cols, latent_group_size)?;
        {
            let peer_cache = &mut self.cooperative_peer.as_mut().expect("peer 已创建").cache;
            peer.activate().map_err(compute_error)?;
            Self::ensure_parity_layer(peer_cache, peer, layer, end, latent.cols, rope.cols, latent_group_size)?;
        }
        owner.activate().map_err(compute_error)?;
        let (latent_cache, latent_scales, rope_cache, table, cache_device) = if parity == 0 {
            let cached = self.paged_layers[layer].as_ref().expect("owner shard 已创建");
            (
                cached.latent.clone(),
                cached.latent_scales.as_ref().expect("Q8 shard scales").clone(),
                cached.rope.clone(),
                self.block_table.buffer().ok_or_else(|| compute_error("cooperative KV 缺少 owner block table"))?.clone(),
                owner.device_id,
            )
        } else {
            let peer_state = self.cooperative_peer.as_ref().expect("peer 已创建");
            let cached = peer_state.cache.paged_layers[layer].as_ref().expect("peer shard 已创建");
            (
                cached.latent.clone(),
                cached.latent_scales.as_ref().expect("Q8 shard scales").clone(),
                cached.rope.clone(),
                peer_state.cache.block_table.buffer().ok_or_else(|| compute_error("cooperative KV 缺少 peer block table"))?.clone(),
                peer.device_id,
            )
        };
        ops::hip::try_paged_cache_append_mla_rope_remote_f32_q8_bf16(
            owner.device_id,
            cache_device,
            latent.device.as_deref().ok_or_else(|| compute_error("cooperative KV latent 缺少 device buffer"))?,
            &latent_cache,
            &latent_scales,
            rope.device.as_deref().ok_or_else(|| compute_error("cooperative KV rope 缺少 device buffer"))?,
            &rope_cache,
            &table,
            local_row,
            position,
            1,
            latent.cols,
            rope.cols,
            rotary_dim,
            layout,
            latent_group_size,
            ROCM_KV_BLOCK_SIZE,
            cos,
            sin,
        )
        .map_err(compute_error)?;
        self.paged_layers[layer].as_mut().expect("owner shard 已创建").rows = end;
        self.cooperative_peer.as_mut().expect("peer 已创建").cache.paged_layers[layer].as_mut().expect("peer shard 已创建").rows = end;
        Ok(())
    }

    /// owner 只量化一次完整输入，随后按 64-token block parity 把量化字节直接
    /// 落到两卡 compact cache。decode 单行因此只产生一次 656B 记录；当该
    /// block 属于 peer 时通过一个 event-ordered P2P 边界写到固定目标地址。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn append_cooperative_mla_rope(
        &mut self,
        owner: &RocmContext,
        peer: &RocmContext,
        layer: usize,
        latent: &RocmTensor,
        rope: &RocmTensor,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        position: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(), BackendError> {
        if latent.rows == 0 || latent.rows != rope.rows || latent.cols == 0 || rope.cols == 0 || position.checked_add(latent.rows).is_none_or(|end| end > self.capacity) {
            return Err(compute_error(format!("L{layer} cooperative KV append shape/position 非法: position={position} latent=[{},{}] rope=[{},{}] capacity={}", latent.rows, latent.cols, rope.rows, rope.cols, self.capacity)));
        }
        self.ensure_cooperative_peer(owner, peer)?;
        if self.mla_rows(layer) != position || self.cooperative_peer.as_ref().expect("peer 已创建").cache.mla_rows(layer) != position {
            return Err(compute_error(format!("L{layer} cooperative KV append position={position} 与 owner/peer rows={}/{} 不一致", self.mla_rows(layer), self.cooperative_peer.as_ref().expect("peer 已创建").cache.mla_rows(layer))));
        }
        let latent_group_size = if ops::hip::options().kv_f16 { 0 } else { crate::kv_cache::DEFAULT_GROUP_SIZE };
        if latent_group_size == 0 || !latent.cols.is_multiple_of(latent_group_size) {
            return Err(compute_error(format!("L{layer} cooperative KV 当前要求 Q8G{} latent，实际 cols={} group={latent_group_size}", crate::kv_cache::DEFAULT_GROUP_SIZE, latent.cols)));
        }
        let end = position + latent.rows;
        if latent.rows == 1 {
            return self.append_cooperative_mla_rope_single(owner, peer, layer, position, latent, rope, rotary_dim, layout, latent_group_size, cos, sin, end);
        }
        owner.activate().map_err(compute_error)?;
        Self::ensure_parity_layer(self, owner, layer, end, latent.cols, rope.cols, latent_group_size)?;
        {
            let peer_cache = &mut self.cooperative_peer.as_mut().expect("peer 已创建").cache;
            peer.activate().map_err(compute_error)?;
            Self::ensure_parity_layer(peer_cache, peer, layer, end, latent.cols, rope.cols, latent_group_size)?;
        }

        owner.activate().map_err(compute_error)?;
        let latent_row_bytes = latent.cols;
        let scale_row_bytes = latent.cols / latent_group_size * 2;
        let rope_row_bytes = rope.cols * 2;
        // staging 只活一个 stage，不能按长期 cache 走精确尺寸 hipMalloc；否则
        // 每层每 chunk 都会在 Drop 时用 hipFree 同步整张卡。显式复用池的块本身
        // 已可跨卡读取，只有关闭复用池、退化成 stream-ordered allocation 时才
        // 需要额外稳定化。
        let staging_latent = Arc::new(ops::hip::DeviceBuffer::allocate_reusable(owner.device_id, latent.rows * latent_row_bytes).map_err(compute_error)?);
        let staging_scales = Arc::new(ops::hip::DeviceBuffer::allocate_reusable(owner.device_id, latent.rows * scale_row_bytes).map_err(compute_error)?);
        let staging_rope = Arc::new(ops::hip::DeviceBuffer::allocate_reusable(owner.device_id, latent.rows * rope_row_bytes).map_err(compute_error)?);
        let table = self.cooperative_staging_table.get("KV cooperative staging", latent.rows, owner.device_id, latent.rows)?;
        ops::hip::try_paged_cache_append_mla_rope_at_f32_q8_bf16(
            owner.device_id,
            latent.device.as_deref().ok_or_else(|| compute_error("cooperative KV latent 缺少 device buffer"))?,
            &staging_latent,
            &staging_scales,
            rope.device.as_deref().ok_or_else(|| compute_error("cooperative KV rope 缺少 device buffer"))?,
            &staging_rope,
            &table,
            0,
            position,
            latent.rows,
            latent.cols,
            rope.cols,
            rotary_dim,
            layout,
            latent_group_size,
            ROCM_KV_BLOCK_SIZE,
            cos,
            sin,
        )
        .map_err(compute_error)?;

        let owner_layer = self.paged_layers[layer].as_mut().expect("owner shard 已创建");
        for (source_row, target_row, rows) in parity_segments(position, latent.rows, 0) {
            owner_layer.latent.copy_from_device(target_row * latent_row_bytes, &staging_latent, source_row * latent_row_bytes, rows * latent_row_bytes).map_err(compute_error)?;
            owner_layer.latent_scales.as_ref().expect("Q8 shard scales").copy_from_device(target_row * scale_row_bytes, &staging_scales, source_row * scale_row_bytes, rows * scale_row_bytes).map_err(compute_error)?;
            owner_layer.rope.copy_from_device(target_row * rope_row_bytes, &staging_rope, source_row * rope_row_bytes, rows * rope_row_bytes).map_err(compute_error)?;
        }
        owner_layer.rows = end;

        let stable = |buffer: &Arc<ops::hip::DeviceBuffer>| {
            if buffer.is_async_allocated() { buffer.copy_to_stable_deferred().map(Arc::new).map_err(compute_error) } else { Ok(buffer.clone()) }
        };
        let stable_latent = stable(&staging_latent)?;
        let stable_scales = stable(&staging_scales)?;
        let stable_rope = stable(&staging_rope)?;
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let peer_cache = &mut self.cooperative_peer.as_mut().expect("peer 已创建").cache;
        let peer_layer = peer_cache.paged_layers[layer].as_mut().expect("peer shard 已创建");
        for (source_row, target_row, rows) in parity_segments(position, latent.rows, 1) {
            for (source, target, row_bytes) in [
                (&stable_latent, peer_layer.latent.as_ref(), latent_row_bytes),
                (&stable_scales, peer_layer.latent_scales.as_ref().expect("Q8 shard scales").as_ref(), scale_row_bytes),
                (&stable_rope, peer_layer.rope.as_ref(), rope_row_bytes),
            ] {
                sources.push(Arc::new(ops::hip::DeviceBuffer::view(source.clone(), source_row * row_bytes, rows * row_bytes).map_err(compute_error)?));
                targets.push((target, target_row * row_bytes));
            }
        }
        if !sources.is_empty() {
            peer.activate().map_err(compute_error)?;
            ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(&sources, &targets, peer.device_id, owner.device_id).map_err(|error| compute_error(format!("L{layer} cooperative KV owner->peer: {error}")))?;
        }
        peer_layer.rows = end;
        Ok(())
    }

    pub(super) fn cooperative_peer_cache(&self, peer_device: i32) -> Result<&RocmKvCache, BackendError> {
        let peer = self.cooperative_peer.as_ref().ok_or_else(|| compute_error("ROCm cooperative peer KV 尚未创建"))?;
        if peer.device_id != peer_device {
            return Err(compute_error(format!("ROCm cooperative peer KV device={}，期望 {peer_device}", peer.device_id)));
        }
        Ok(&peer.cache)
    }

    pub(super) fn cached_cooperative_selection(
        &self,
        peer_device: i32,
        source: &Arc<ops::hip::DeviceBuffer>,
    ) -> Option<(&Arc<ops::hip::DeviceBuffer>, &Arc<ops::hip::DeviceBuffer>, &Arc<ops::hip::DeviceBuffer>, &Arc<ops::hip::DeviceBuffer>, usize)> {
        self.cooperative_selection
            .as_ref()
            .filter(|selection| selection.peer_device == peer_device && Arc::ptr_eq(&selection.source, source))
            .map(|selection| (&selection.owner, &selection.owner_counts, &selection.peer, &selection.peer_counts, selection.width))
    }

    pub(super) fn cache_cooperative_selection(
        &mut self,
        peer_device: i32,
        source: Arc<ops::hip::DeviceBuffer>,
        owner: Arc<ops::hip::DeviceBuffer>,
        owner_counts: Arc<ops::hip::DeviceBuffer>,
        peer: Arc<ops::hip::DeviceBuffer>,
        peer_counts: Arc<ops::hip::DeviceBuffer>,
        width: usize,
    ) -> Result<(), BackendError> {
        if source.device_id() == peer_device
            || owner.device_id() != source.device_id()
            || owner_counts.device_id() != source.device_id()
            || peer.device_id() != peer_device
            || peer_counts.device_id() != peer_device
            || source.bytes() != owner.bytes()
            || source.bytes() != peer.bytes()
            || owner_counts.bytes() != peer_counts.bytes()
            || width == 0
        {
            return Err(compute_error(format!(
                "ROCm cooperative selection shard 设备/大小异常: source_device={} owner_device={} peer_device={} target={peer_device} source_bytes={} owner_bytes={} peer_bytes={} owner_counts={} peer_counts={} width={width}",
                source.device_id(),
                owner.device_id(),
                peer.device_id(),
                source.bytes(),
                owner.bytes(),
                peer.bytes(),
                owner_counts.bytes(),
                peer_counts.bytes(),
            )));
        }
        self.cooperative_selection = Some(RocmCooperativeSelection { peer_device, source, owner, owner_counts, peer, peer_counts, width });
        Ok(())
    }

    pub(super) fn cached_operator_selection(&self, peer_device: i32, source: &Arc<ops::hip::DeviceBuffer>) -> Option<Arc<ops::hip::DeviceBuffer>> {
        self.operator_selection.as_ref().filter(|selection| selection.peer_device == peer_device && Arc::ptr_eq(&selection.source, source)).map(|selection| selection.peer.clone())
    }

    pub(super) fn cache_operator_selection(&mut self, peer_device: i32, source: Arc<ops::hip::DeviceBuffer>, peer: Arc<ops::hip::DeviceBuffer>) -> Result<(), BackendError> {
        if source.device_id() == peer_device || peer.device_id() != peer_device || source.bytes() != peer.bytes() {
            return Err(compute_error(format!("ROCm operator selection 镜像非法: source_device={} peer_device={} target={peer_device} source_bytes={} peer_bytes={}", source.device_id(), peer.device_id(), source.bytes(), peer.bytes(),)));
        }
        self.operator_selection = Some(RocmOperatorSelection { peer_device, source, peer });
        Ok(())
    }

    fn ensure_cooperative_peer(&mut self, owner: &RocmContext, peer: &RocmContext) -> Result<(), BackendError> {
        if let Some(existing) = self.cooperative_peer.as_ref() {
            return if existing.device_id == peer.device_id { Ok(()) } else { Err(compute_error(format!("ROCm cooperative peer KV 已绑定 device={}，不能改为 {}", existing.device_id, peer.device_id))) };
        }
        // 这里只创建空容器；每个协作层首次到达时再单独补它自己的旧历史。
        // 不能复制当时可见的 dense 层，否则 peer 不再更新它们，session 提交时
        // owner/peer 的逻辑长度会永久分叉。
        if self.ownership == RocmKvOwnership::Full && self.paged_layers.iter().any(Option::is_some) {
            return Err(compute_error("ROCm cooperative KV 必须在首个 append 前启用；已有 full cache 需先走 pair restore"));
        }
        self.ownership = RocmKvOwnership::BlockParity(0);
        let mut cache = Box::new(Self::with_capacity(self.paged_layers.len(), self.capacity));
        cache.ownership = RocmKvOwnership::BlockParity(1);
        cache.prepare_block_table(peer)?;
        self.cooperative_peer = Some(RocmCooperativeKvPeer { device_id: peer.device_id, cache });
        let reserved_rows = self.pending_pair_reserved_rows;
        for layer in 0..self.pending_pair_layers.len() {
            let Some(record) = self.pending_pair_layers[layer].take() else { continue };
            let owner_rows = parity_rows(record.rows, 0);
            let peer_rows = parity_rows(record.rows, 1);
            let owner_capacity = parity_rows(reserved_rows.max(record.rows), 0).max(owner_rows);
            let peer_capacity = parity_rows(reserved_rows.max(record.rows), 1).max(peer_rows);
            let latent_row_bytes = record.latent_cols * if record.latent_group_size == 0 { 2 } else { 1 };
            let scale_row_bytes = if record.latent_group_size == 0 { 0 } else { record.latent_cols / record.latent_group_size * 2 };
            let rope_row_bytes = record.rope_cols * 2;
            let (owner_latent, peer_latent) = split_pair_bytes(&record.latent, owner_rows, peer_rows, latent_row_bytes, &format!("L{layer} latent"))?;
            let (owner_rope, peer_rope) = split_pair_bytes(&record.rope, owner_rows, peer_rows, rope_row_bytes, &format!("L{layer} rope"))?;
            let scales = record.latent_scales.as_deref().map(|bytes| split_pair_bytes(bytes, owner_rows, peer_rows, scale_row_bytes, &format!("L{layer} scales"))).transpose()?;
            let restore = |device_id: i32, bytes: &[u8], capacity: usize| {
                if bytes.is_empty() { ops::hip::DeviceBuffer::allocate_cache(device_id, capacity).map(Arc::new).map_err(compute_error) } else { upload_cache_buffer(device_id, bytes, capacity) }
            };
            owner.activate().map_err(compute_error)?;
            let owner_layer = RocmPagedMlaLayer {
                latent: restore(owner.device_id, owner_latent, owner_capacity * latent_row_bytes)?,
                latent_scales: scales.map(|(bytes, _)| restore(owner.device_id, bytes, owner_capacity * scale_row_bytes)).transpose()?,
                latent_group_size: record.latent_group_size,
                rope: restore(owner.device_id, owner_rope, owner_capacity * rope_row_bytes)?,
                rows: record.rows,
                latent_cols: record.latent_cols,
                rope_cols: record.rope_cols,
                committed_rows: owner_capacity,
                cpu_mirror: None,
                cpu_hot: None,
            };
            peer.activate().map_err(compute_error)?;
            let peer_layer = RocmPagedMlaLayer {
                latent: restore(peer.device_id, peer_latent, peer_capacity * latent_row_bytes)?,
                latent_scales: scales.map(|(_, bytes)| restore(peer.device_id, bytes, peer_capacity * scale_row_bytes)).transpose()?,
                latent_group_size: record.latent_group_size,
                rope: restore(peer.device_id, peer_rope, peer_capacity * rope_row_bytes)?,
                rows: record.rows,
                latent_cols: record.latent_cols,
                rope_cols: record.rope_cols,
                committed_rows: peer_capacity,
                cpu_mirror: None,
                cpu_hot: None,
            };
            self.paged_layers[layer] = Some(owner_layer);
            self.cooperative_peer.as_mut().expect("peer 已创建").cache.paged_layers[layer] = Some(peer_layer);
        }
        Ok(())
    }

    pub(super) fn ensure_operator_peer(&mut self, owner: &RocmContext, peer: &RocmContext) -> Result<Arc<Mutex<RocmKvCache>>, BackendError> {
        if self.cooperative_peer.is_some() {
            return Err(compute_error("ROCm operator KV 不能与 cooperative parity KV 共存"));
        }
        if owner.device_id == peer.device_id {
            return Err(compute_error("ROCm operator KV owner/peer 不能是同一设备"));
        }
        if let Some(existing) = self.operator_peer.as_ref() {
            return if existing.device_id == peer.device_id { Ok(existing.cache.clone()) } else { Err(compute_error(format!("ROCm operator peer KV 已绑定 device={}，不能改为 {}", existing.device_id, peer.device_id))) };
        }
        if self.ownership != RocmKvOwnership::Full {
            return Err(compute_error(format!("ROCm operator KV owner ownership={:?} 非法", self.ownership)));
        }
        let owner_stream = ops::hip::active_compute_stream() as usize;
        let mut peer_cache = Self::with_capacity(self.paged_layers.len(), self.capacity);
        peer_cache.stage_decode = self.stage_decode;
        peer_cache.prepare_block_table(peer)?;

        // 普通 Full snapshot 已经在 owner 恢复时，直接异步 P2P 有效前缀；目标
        // 仍按 committed 容量分配。复制排在 peer default stream，随后 peer
        // attention 同流消费，不引入 host synchronize。
        for (layer, cached) in self.paged_layers.iter().enumerate().filter_map(|(layer, slot)| slot.as_ref().map(|cached| (layer, cached))) {
            if cached.latent.device_id() != owner.device_id || cached.rope.device_id() != owner.device_id || cached.rows > cached.committed_rows || cached.cpu_hot.is_some() {
                return Err(compute_error(format!(
                    "L{layer} operator KV owner cache 非法: device={}/{} rows={}/{} cpu_hot={}",
                    cached.latent.device_id(),
                    cached.rope.device_id(),
                    cached.rows,
                    cached.committed_rows,
                    cached.cpu_hot.is_some(),
                )));
            }
            let latent_row_bytes = cached.latent_cols * if cached.latent_group_size == 0 { 2 } else { 1 };
            let scale_row_bytes = if cached.latent_group_size == 0 { 0 } else { cached.latent_cols / cached.latent_group_size * 2 };
            let rope_row_bytes = cached.rope_cols * 2;
            let copy = |source: &Arc<ops::hip::DeviceBuffer>, used_bytes: usize| -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
                if used_bytes == 0 || used_bytes > source.bytes() {
                    return Err(compute_error(format!("L{layer} operator KV copy bytes={used_bytes}/{} 非法", source.bytes())));
                }
                let destination = Arc::new(ops::hip::DeviceBuffer::allocate_cache(peer.device_id, source.bytes()).map_err(compute_error)?);
                let source = Arc::new(ops::hip::DeviceBuffer::view(source.clone(), 0, used_bytes).map_err(compute_error)?);
                ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(&[source], &[(&destination, 0)], peer.device_id, owner.device_id).map_err(compute_error)?;
                Ok(destination)
            };
            let latent = copy(&cached.latent, cached.rows * latent_row_bytes)?;
            let latent_scales = cached.latent_scales.as_ref().map(|scales| copy(scales, cached.rows * scale_row_bytes)).transpose()?;
            let rope = copy(&cached.rope, cached.rows * rope_row_bytes)?;
            let peer_cpu_mirror = cached
                .cpu_mirror
                .as_ref()
                .map(|mirror| {
                    let mut mirror = mirror.lock().map_err(|_| compute_error(format!("L{layer} operator owner CPU mirror 锁中毒")))?;
                    mirror.flush_feed(owner.device_id, cached.rows, &cached.latent, cached.latent_scales.as_deref().unwrap_or(&cached.latent), &cached.rope)?;
                    if mirror.mirror.rows != cached.rows {
                        return Err(compute_error(format!("L{layer} operator owner CPU mirror rows={}，cache rows={}", mirror.mirror.rows, cached.rows)));
                    }
                    Ok(std::sync::Mutex::new(RocmMlaCpuMirror::from_record(mirror.mirror.clone())))
                })
                .transpose()?;
            peer_cache.paged_layers[layer] = Some(RocmPagedMlaLayer {
                latent,
                latent_scales,
                latent_group_size: cached.latent_group_size,
                rope,
                rows: cached.rows,
                latent_cols: cached.latent_cols,
                rope_cols: cached.rope_cols,
                committed_rows: cached.committed_rows,
                cpu_mirror: peer_cpu_mirror,
                cpu_hot: None,
            });
        }

        // 旧 cooperative snapshot 尚未上传时，先在 host 恢复完整逻辑顺序，
        // 再分别上传两张卡。这样 session 格式不因执行形态切换而失效。
        let mut owner_restored = Vec::new();
        let reserved_rows = self.pending_pair_reserved_rows;
        for layer in 0..self.pending_pair_layers.len() {
            let Some(record) = self.pending_pair_layers[layer].as_ref().cloned() else { continue };
            if self.paged_layers[layer].is_some() || peer_cache.paged_layers[layer].is_some() {
                return Err(compute_error(format!("L{layer} operator KV 同时存在 device 与 pending snapshot")));
            }
            let record = join_pair_record(record, layer)?;
            let committed_rows = reserved_rows.max(record.rows);
            let latent_element_bytes = if record.latent_group_size == 0 { 2 } else { 1 };
            let latent_capacity = committed_rows * record.latent_cols * latent_element_bytes;
            let scale_capacity = if record.latent_group_size == 0 { 0 } else { committed_rows * (record.latent_cols / record.latent_group_size) * 2 };
            let rope_capacity = committed_rows * record.rope_cols * 2;
            let restore = |context: &RocmContext, mirror: bool| -> Result<RocmPagedMlaLayer, BackendError> {
                context.activate().map_err(compute_error)?;
                Ok(RocmPagedMlaLayer {
                    latent: upload_cache_buffer(context.device_id, &record.latent, latent_capacity)?,
                    latent_scales: record.latent_scales.as_deref().map(|bytes| upload_cache_buffer(context.device_id, bytes, scale_capacity)).transpose()?,
                    latent_group_size: record.latent_group_size,
                    rope: upload_cache_buffer(context.device_id, &record.rope, rope_capacity)?,
                    rows: record.rows,
                    latent_cols: record.latent_cols,
                    rope_cols: record.rope_cols,
                    committed_rows,
                    cpu_mirror: mirror.then(|| std::sync::Mutex::new(RocmMlaCpuMirror::from_record(record.clone()))),
                    cpu_hot: None,
                })
            };
            let mirror = (ops::hip::options().mla_cpu_hot_rows != 0 || ops::hip::options().prefill_attention_cpu) && record.latent_group_size != 0;
            owner_restored.push((layer, restore(owner, mirror)?));
            peer_cache.paged_layers[layer] = Some(restore(peer, mirror)?);
        }
        for (layer, restored) in owner_restored {
            self.paged_layers[layer] = Some(restored);
            self.pending_pair_layers[layer] = None;
        }
        ops::hip::activate_compute_stream(owner.device_id, owner_stream).map_err(compute_error)?;
        let cache = Arc::new(Mutex::new(peer_cache));
        self.operator_peer = Some(RocmOperatorKvPeer { device_id: peer.device_id, cache: cache.clone() });
        Ok(cache)
    }

    /// operator pair 的默认 Q8 cache 由 owner 量化一次，再把新增长度的压缩
    /// 记录直接写进 peer 最终 cache；GPU 热窗的小批次也沿用此路径。
    /// BF16 与独立 CPU mirror 诊断保留各卡追加的兼容路径。
    pub(super) fn operator_packed_kv_replication_enabled(&self, layer: usize) -> bool {
        !ops::hip::options().kv_f16
            && !ops::hip::options().prefill_attention_cpu
            && (ops::hip::options().mla_cpu_hot_rows != 0 || !mla_cpu_mirror_enabled())
            && self.ownership == RocmKvOwnership::Full
            && self.operator_peer.is_some()
            && self.paged_layers.get(layer).is_some_and(|slot| slot.as_ref().is_none_or(|cached| cached.latent_group_size == crate::kv_cache::DEFAULT_GROUP_SIZE && cached.latent_scales.is_some()))
    }

    /// 该层 owner cache 是否已切换到热窗；调用方据此选择 CPU 选集的兼容路径。
    pub(super) fn operator_layer_is_hot(&self, layer: usize) -> bool {
        self.paged_layers.get(layer).and_then(Option::as_ref).is_some_and(|cached| cached.cpu_hot.is_some())
    }

    /// peer worker 在自己持锁的 attention 段内喂养本层 mirror(packed 复制路径
    /// 专用;fallback 双端各自 append 已由 peer 的 append_mla 自喂)。全量期走
    /// 批量 feed；热窗期先把新槽位三段合并复制到环形日志，再批量 D2H。
    /// decode 的 channel 顺序保证 worker 运行时 P2P 复制已经入队。
    pub(super) fn feed_peer_mirror_layer(&mut self, context: &RocmContext, layer: usize) -> Result<(), BackendError> {
        if ops::hip::options().mla_cpu_hot_rows == 0 {
            return Ok(());
        }
        let cached = self.paged_layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| compute_error(format!("L{layer} peer mirror feed 层未初始化")))?;
        if cached.latent_group_size == 0 {
            return Ok(());
        }
        let scales = cached.latent_scales.as_deref().expect("Q8 peer cache 必有 scales");
        if let Some(hot_mutex) = cached.cpu_hot.as_ref() {
            let mut hot = hot_mutex.lock().map_err(|_| compute_error(format!("L{layer} operator peer hot 锁中毒")))?;
            if let Some((log_latent, log_scales, log_rope)) = hot.mirror_log.clone() {
                if hot.log_fed_rows == cached.rows {
                    return Ok(());
                }
                if cached.rows < hot.log_fed_rows || cached.rows - hot.log_fed_rows > MLA_HOT_LOG_ROWS {
                    return Err(compute_error(format!("L{layer} operator peer hot log={}，cache={}，增量超过日志环", hot.log_fed_rows, cached.rows)));
                }
                let row_bytes = [cached.latent_cols, cached.latent_cols / cached.latent_group_size * 2, cached.rope_cols * 2];
                while hot.log_fed_rows < cached.rows {
                    let token = hot.log_fed_rows;
                    let slot = hot.token_to_slot[token];
                    if slot == MLA_HOT_INVALID {
                        return Err(compute_error(format!("L{layer} operator peer hot token={token} 缺少日志来源槽位")));
                    }
                    let slot = slot as usize;
                    let ring = token % MLA_HOT_LOG_ROWS;
                    let rows = if hot.uses_gpu() { (MLA_HOT_LOG_ROWS - ring).min(cached.rows - token) } else { 1 };
                    if !hot.uses_gpu() {
                        ops::hip::DeviceBuffer::copy3_from_device(
                            [(&cached.latent, slot * row_bytes[0]), (scales, slot * row_bytes[1]), (&cached.rope, slot * row_bytes[2])],
                            [(&log_latent, ring * row_bytes[0]), (&log_scales, ring * row_bytes[1]), (&log_rope, ring * row_bytes[2])],
                            row_bytes.map(|bytes| rows * bytes),
                        )
                        .map_err(compute_error)?;
                    }
                    hot.log_fed_rows += rows;
                    if hot.log_fed_rows.is_multiple_of(MLA_HOT_LOG_ROWS) {
                        hot.flush_log_feed(context.device_id)?;
                    }
                }
                return Ok(());
            }
            let chain_end = hot.pending_download.as_ref().map_or(hot.mirror.rows, |pending| pending.0 + pending.1);
            if chain_end >= cached.rows {
                return Ok(());
            }
            let slot = *hot.token_to_slot.get(chain_end).ok_or_else(|| compute_error(format!("L{layer} operator peer hot token={chain_end} 越界")))?;
            if slot == MLA_HOT_INVALID {
                return Err(compute_error(format!("L{layer} operator peer hot token={chain_end} 缺少槽位（owner replicate 未先行）")));
            }
            hot.queue_rows(context.device_id, chain_end, 1, slot as usize, &cached.latent, scales, &cached.rope)?;
            return Ok(());
        }
        match cached.cpu_mirror.as_mut() {
            Some(mirror) => {
                mirror.lock().map_err(|_| compute_error(format!("L{layer} operator peer mirror 锁中毒")))?.feed(context.device_id, self.capacity, cached.rows, &cached.latent, scales, &cached.rope)?;
            }
            None => {
                let mut mirror = RocmMlaCpuMirror::new(self.capacity, cached.latent_cols, cached.latent_group_size, cached.rope_cols)?;
                mirror.feed(context.device_id, self.capacity, cached.rows, &cached.latent, scales, &cached.rope)?;
                cached.cpu_mirror = Some(std::sync::Mutex::new(mirror));
            }
        }
        Ok(())
    }

    /// `append_mla` 已经在 owner stream 排入量化后调用。identity block table
    /// 使逻辑行与物理行同址，因此只需把三段压缩字节复制到 peer 的同一 offset。
    /// 返回的来源由 peer stage completion 保活，不能只依赖 owner completion。
    pub(super) fn replicate_operator_mla_append(&mut self, owner: &RocmContext, peer: &RocmContext, layer: usize, position: usize, rows: usize) -> Result<Vec<Arc<ops::hip::DeviceBuffer>>, BackendError> {
        if !self.operator_packed_kv_replication_enabled(layer) || rows == 0 {
            return Err(compute_error(format!("L{layer} operator packed KV replica 当前配置不支持")));
        }
        let end = position.checked_add(rows).ok_or_else(|| compute_error(format!("L{layer} operator packed KV rows 溢出")))?;
        let peer_state = self.operator_peer.as_ref().ok_or_else(|| compute_error("ROCm operator peer KV 尚未创建"))?;
        if peer_state.device_id != peer.device_id || owner.device_id == peer.device_id {
            return Err(compute_error(format!("L{layer} operator packed KV device={}/{} 非法", owner.device_id, peer.device_id)));
        }
        if self.paged_layers.get(layer).and_then(Option::as_ref).is_some_and(|cached| cached.cpu_hot.is_some()) {
            return self.replicate_operator_mla_append_hot(owner, peer, layer, position, rows);
        }
        let peer_cache = peer_state.cache.clone();
        let owner_layer = self.paged_layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} operator owner KV 尚未初始化")))?;
        if owner_layer.rows != end || owner_layer.latent_group_size != crate::kv_cache::DEFAULT_GROUP_SIZE || owner_layer.latent_scales.is_none() || owner_layer.cpu_hot.is_some() {
            return Err(compute_error(format!(
                "L{layer} operator owner packed KV 非法: rows={}/{} group={} scales={} cpu_hot={} cpu_mirror={}",
                owner_layer.rows,
                end,
                owner_layer.latent_group_size,
                owner_layer.latent_scales.is_some(),
                owner_layer.cpu_hot.is_some(),
                owner_layer.cpu_mirror.is_some(),
            )));
        }
        let latent_cols = owner_layer.latent_cols;
        let rope_cols = owner_layer.rope_cols;
        let group_size = owner_layer.latent_group_size;
        let committed_rows = owner_layer.committed_rows;
        let (latent_row_bytes, scale_row_bytes, rope_row_bytes) = mla_q8_row_bytes(latent_cols, rope_cols, group_size).ok_or_else(|| compute_error(format!("L{layer} operator packed KV row layout 非法")))?;
        let owner_latent = owner_layer.latent.clone();
        let owner_scales = owner_layer.latent_scales.as_ref().expect("上面已校验 Q8 scales").clone();
        let owner_rope = owner_layer.rope.clone();

        let owner_stream = ops::hip::active_compute_stream() as usize;
        let peer_buffers = (|| {
            peer.activate().map_err(compute_error)?;
            let mut cache = peer_cache.lock().map_err(|_| compute_error(format!("L{layer} operator peer KV 锁中毒")))?;
            if cache.ownership != RocmKvOwnership::Full || cache.capacity != self.capacity {
                return Err(compute_error(format!("L{layer} operator peer KV ownership/capacity 非法")));
            }
            cache.prepare_mla_append(peer, layer, rows)?;
            let slot = cache.paged_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
            if slot.is_none() {
                let latent_bytes = committed_rows.checked_mul(latent_row_bytes).ok_or_else(|| compute_error(format!("L{layer} operator peer latent 大小溢出")))?;
                let scale_bytes = committed_rows.checked_mul(scale_row_bytes).ok_or_else(|| compute_error(format!("L{layer} operator peer scales 大小溢出")))?;
                let rope_bytes = committed_rows.checked_mul(rope_row_bytes).ok_or_else(|| compute_error(format!("L{layer} operator peer rope 大小溢出")))?;
                *slot = Some(RocmPagedMlaLayer {
                    latent: Arc::new(ops::hip::DeviceBuffer::allocate_cache(peer.device_id, latent_bytes).map_err(compute_error)?),
                    latent_scales: Some(Arc::new(ops::hip::DeviceBuffer::allocate_cache(peer.device_id, scale_bytes).map_err(compute_error)?)),
                    latent_group_size: group_size,
                    rope: Arc::new(ops::hip::DeviceBuffer::allocate_cache(peer.device_id, rope_bytes).map_err(compute_error)?),
                    rows: 0,
                    latent_cols,
                    rope_cols,
                    committed_rows,
                    cpu_mirror: None,
                    cpu_hot: None,
                });
            }
            let cached = slot.as_mut().expect("刚创建 operator peer KV layer");
            if cached.rows != position || cached.latent_cols != latent_cols || cached.rope_cols != rope_cols || cached.latent_group_size != group_size || cached.latent_scales.is_none() || cached.cpu_hot.is_some() {
                return Err(compute_error(format!(
                    "L{layer} operator peer packed KV 非法: rows={}/{} latent={}/{} rope={}/{} group={}/{} scales={} cpu_hot={} cpu_mirror={}",
                    cached.rows,
                    position,
                    cached.latent_cols,
                    latent_cols,
                    cached.rope_cols,
                    rope_cols,
                    cached.latent_group_size,
                    group_size,
                    cached.latent_scales.is_some(),
                    cached.cpu_hot.is_some(),
                    cached.cpu_mirror.is_some(),
                )));
            }
            if committed_rows > cached.committed_rows {
                cached.latent = grow_cache_buffer(peer.device_id, &cached.latent, position * latent_row_bytes, committed_rows * latent_row_bytes)?;
                cached.latent_scales = Some(grow_cache_buffer(peer.device_id, cached.latent_scales.as_ref().expect("Q8 peer cache 必有 scales"), position * scale_row_bytes, committed_rows * scale_row_bytes)?);
                cached.rope = grow_cache_buffer(peer.device_id, &cached.rope, position * rope_row_bytes, committed_rows * rope_row_bytes)?;
                cached.committed_rows = committed_rows;
            }
            if end > cached.committed_rows {
                return Err(compute_error(format!("L{layer} operator peer packed KV end={end} 超过 committed={}", cached.committed_rows)));
            }
            Ok((cached.latent.clone(), cached.latent_scales.as_ref().expect("Q8 peer cache 必有 scales").clone(), cached.rope.clone()))
        })();
        ops::hip::activate_compute_stream(owner.device_id, owner_stream).map_err(compute_error)?;
        let (peer_latent, peer_scales, peer_rope) = peer_buffers?;

        let stable_range = |source: Arc<ops::hip::DeviceBuffer>, row_bytes: usize, name: &str| -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
            let offset = position.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("L{layer} operator {name} offset 溢出")))?;
            let bytes = rows.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("L{layer} operator {name} bytes 溢出")))?;
            let view = Arc::new(ops::hip::DeviceBuffer::view(source, offset, bytes).map_err(compute_error)?);
            if view.is_async_allocated() { view.copy_to_stable_deferred().map(Arc::new).map_err(compute_error) } else { Ok(view) }
        };
        let sources = vec![stable_range(owner_latent, latent_row_bytes, "latent")?, stable_range(owner_scales, scale_row_bytes, "scales")?, stable_range(owner_rope, rope_row_bytes, "rope")?];
        let targets = [(peer_latent.as_ref(), position * latent_row_bytes), (peer_scales.as_ref(), position * scale_row_bytes), (peer_rope.as_ref(), position * rope_row_bytes)];
        ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(&sources, &targets, peer.device_id, owner.device_id).map_err(|error| compute_error(format!("L{layer} operator packed KV owner->peer: {error}")))?;
        {
            // 只推进行数；peer mirror 喂养由 peer worker 在自己的 attention 锁段
            // 内完成(feed_peer_mirror_layer)，避免 owner 线程每层一次 peer 设备
            // 切换 + 额外锁往返。
            let mut cache = peer_cache.lock().map_err(|_| compute_error(format!("L{layer} operator peer KV 锁中毒")))?;
            cache.paged_layers[layer].as_mut().expect("operator peer KV layer 已创建").rows = end;
        }
        ops::hip::activate_compute_stream(owner.device_id, owner_stream).map_err(compute_error)?;
        Ok(sources)
    }

    /// owner 已量化的热槽直接复制到 peer；小批 verify 在 recent 环回绕处分段。
    /// 两卡独立保留 mirror 生命周期，peer worker 在 attention 前喂养自己的日志。
    fn replicate_operator_mla_append_hot(&mut self, owner: &RocmContext, peer: &RocmContext, layer: usize, position: usize, rows: usize) -> Result<Vec<Arc<ops::hip::DeviceBuffer>>, BackendError> {
        if rows == 0 || (rows != 1 && !gpu_hot_selection(rows)) {
            return Err(compute_error(format!("L{layer} operator hot packed KV 不支持 rows={rows}")));
        }
        let end = position + rows;
        let hot_rows_limit = ops::hip::options().mla_cpu_hot_rows;
        let peer_state = self.operator_peer.as_ref().ok_or_else(|| compute_error("ROCm operator peer KV 尚未创建"))?;
        let peer_cache = peer_state.cache.clone();
        let owner_layer = self.paged_layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} operator owner KV 尚未初始化")))?;
        if owner_layer.rows != end || owner_layer.latent_group_size != crate::kv_cache::DEFAULT_GROUP_SIZE || owner_layer.latent_scales.is_none() {
            return Err(compute_error(format!("L{layer} operator owner hot KV rows={}/{} 非法", owner_layer.rows, end)));
        }
        let latent_cols = owner_layer.latent_cols;
        let rope_cols = owner_layer.rope_cols;
        let group_size = owner_layer.latent_group_size;
        let (latent_row_bytes, scale_row_bytes, rope_row_bytes) = mla_q8_row_bytes(latent_cols, rope_cols, group_size).ok_or_else(|| compute_error(format!("L{layer} operator hot KV row layout 非法")))?;
        let owner_slots = {
            let hot = owner_layer.cpu_hot.as_ref().expect("入口已确认 owner hot").lock().map_err(|_| compute_error(format!("L{layer} operator owner hot 锁中毒")))?;
            (position..end)
                .map(|token| {
                    let slot = *hot.token_to_slot.get(token).ok_or_else(|| compute_error(format!("L{layer} operator owner hot token={token} 越界")))?;
                    if slot == MLA_HOT_INVALID {
                        return Err(compute_error(format!("L{layer} operator owner hot token={token} 缺少槽位")));
                    }
                    Ok(slot as usize)
                })
                .collect::<Result<Vec<_>, BackendError>>()?
        };
        let owner_latent = owner_layer.latent.clone();
        let owner_scales = owner_layer.latent_scales.as_ref().expect("上面已校验 Q8 scales").clone();
        let owner_rope = owner_layer.rope.clone();

        let owner_stream = ops::hip::active_compute_stream() as usize;
        // peer 侧第一段：保证热窗已启用（与 owner 同 token 切换），并按 owner 的
        // append 语义分配槽位。enable 需要在 peer 设备上分配窗口 buffer。
        let peer_slots = {
            peer.activate().map_err(compute_error)?;
            let mut cache = peer_cache.lock().map_err(|_| compute_error(format!("L{layer} operator peer KV 锁中毒")))?;
            let peer_rows = cache.paged_layers.get(layer).and_then(Option::as_ref).map_or(0, |cached| cached.rows);
            if cache.paged_layers.get(layer).and_then(Option::as_ref).is_none_or(|cached| cached.cpu_hot.is_none()) {
                if peer_rows != position {
                    return Err(compute_error(format!("L{layer} operator peer KV rows={peer_rows} 切热窗期望 {position}")));
                }
                cache.enable_cpu_hot_layer(peer, layer, hot_rows_limit)?;
            }
            let cached = cache.paged_layers[layer].as_mut().ok_or(BackendError::UnsupportedLayer { layer })?;
            if cached.rows != position || cached.latent_cols != latent_cols || cached.rope_cols != rope_cols || cached.latent_group_size != group_size || cached.latent_scales.is_none() {
                return Err(compute_error(format!("L{layer} operator peer hot KV rows={}/{} shape 非法", cached.rows, position)));
            }
            let buffers = (cached.latent.clone(), cached.latent_scales.as_ref().expect("Q8 peer cache 必有 scales").clone(), cached.rope.clone());
            let mut hot = cached.cpu_hot.as_ref().expect("上面已确保 peer hot").lock().map_err(|_| compute_error(format!("L{layer} operator peer hot 锁中毒")))?;
            hot.prepare_append_gpu_mode(peer.device_id, rows, [&buffers.0, &buffers.1, &buffers.2])?;
            if hot.mirror_log.is_none() {
                hot.finish_pending()?;
            }
            if hot.mirror.rows > cached.rows || (hot.mirror_log.is_some() && hot.log_fed_rows != cached.rows) || (hot.mirror_log.is_none() && hot.mirror.rows != cached.rows) {
                return Err(compute_error(format!("L{layer} operator peer hot mirror rows={} 缺行（cache={}）", hot.mirror.rows, cached.rows)));
            }
            let prefetched = hot.prefetched.as_ref().is_some_and(|(start, _, _)| *start == cached.rows);
            hot.join_prefetch(peer.device_id)?;
            let slot = if prefetched { hot.assign_slot(position)? } else { hot.append_slot(position)? };
            let mut slots = vec![slot];
            for token in position + 1..end {
                slots.push(hot.assign_slot(token)?);
            }
            drop(hot);
            (slots, buffers)
        };
        let (peer_slots, peer_buffers) = peer_slots;
        ops::hip::activate_compute_stream(owner.device_id, owner_stream).map_err(compute_error)?;

        let stable_slot_view = |source: Arc<ops::hip::DeviceBuffer>, row_bytes: usize, slot: usize, count: usize, name: &str| -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
            let view = Arc::new(ops::hip::DeviceBuffer::view(source, slot.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("L{layer} operator hot {name} offset 溢出")))?, count * row_bytes).map_err(compute_error)?);
            if view.is_async_allocated() { view.copy_to_stable_deferred().map(Arc::new).map_err(compute_error) } else { Ok(view) }
        };
        let (peer_latent, peer_scales, peer_rope) = peer_buffers;
        let mut sources = Vec::with_capacity(6);
        let mut row = 0;
        while row < rows {
            let owner_slot = owner_slots[row];
            let peer_slot = peer_slots[row];
            let mut count = 1;
            while row + count < rows && owner_slots[row + count] == owner_slot + count && peer_slots[row + count] == peer_slot + count {
                count += 1;
            }
            let segment = [
                stable_slot_view(owner_latent.clone(), latent_row_bytes, owner_slot, count, "latent")?,
                stable_slot_view(owner_scales.clone(), scale_row_bytes, owner_slot, count, "scales")?,
                stable_slot_view(owner_rope.clone(), rope_row_bytes, owner_slot, count, "rope")?,
            ];
            let targets = [(peer_latent.as_ref(), peer_slot * latent_row_bytes), (peer_scales.as_ref(), peer_slot * scale_row_bytes), (peer_rope.as_ref(), peer_slot * rope_row_bytes)];
            ops::hip::DeviceBuffer::copy_stable_group_into_device_ordered_async_retained_by(&segment, &targets, peer.device_id, owner.device_id)
                .map_err(|error| compute_error(format!("L{layer} operator hot packed KV owner->peer: {error}")))?;
            sources.extend(segment);
            row += count;
            if row < rows {
                // recent 同时充当日志；复制跨环的后半段前，必须先保存旧环，
                // 否则尚未回传的前缀会被本次追加覆盖。
                peer.activate().map_err(compute_error)?;
                let mut cache = peer_cache.lock().map_err(|_| compute_error(format!("L{layer} operator peer KV 锁中毒")))?;
                cache.paged_layers[layer].as_mut().expect("operator peer KV layer 已创建").rows = position + row;
                cache.feed_peer_mirror_layer(peer, layer)?;
                ops::hip::activate_compute_stream(owner.device_id, owner_stream).map_err(compute_error)?;
            }
        }
        {
            let mut cache = peer_cache.lock().map_err(|_| compute_error(format!("L{layer} operator peer KV 锁中毒")))?;
            cache.paged_layers[layer].as_mut().expect("operator peer KV layer 已创建").rows = end;
        }
        ops::hip::activate_compute_stream(owner.device_id, owner_stream).map_err(compute_error)?;
        Ok(sources)
    }

    fn truncate_replica_rows(&mut self, rows: usize) -> Result<(), BackendError> {
        for layer in 0..self.paged_layers.len() {
            self.truncate_replica_layer_rows(layer, rows)?;
        }
        Ok(())
    }

    fn truncate_replica_layer_rows(&mut self, layer: usize, rows: usize) -> Result<(), BackendError> {
        let paged = self.paged_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if let Some(cached) = paged
            && cached.rows > rows
        {
            if cached.cpu_hot.is_some() {
                truncate_hot_layer(cached, rows)?;
            } else if let Some(mirror) = &cached.cpu_mirror {
                mirror.lock().map_err(|_| compute_error(format!("L{layer} ROCm peer mirror 锁中毒")))?.truncate(rows)?;
            }
            cached.rows = rows;
        }
        let gqa = self.gqa_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if let Some(cached) = gqa
            && cached.rows > rows
        {
            cached.rows = rows;
        }
        Ok(())
    }

    fn initialize_cpu_hot_layer(&mut self, context: &RocmContext, layer: usize, hot_rows: usize, latent_cols: usize, rope_cols: usize) -> Result<(), BackendError> {
        if hot_rows <= 2048 || hot_rows > self.capacity || !latent_cols.is_multiple_of(crate::kv_cache::DEFAULT_GROUP_SIZE) {
            return Err(compute_error(format!("L{layer} ROCm MLA CPU prefill hot shape rows={hot_rows} capacity={} latent={latent_cols} rope={rope_cols} 非法", self.capacity)));
        }
        let group_size = crate::kv_cache::DEFAULT_GROUP_SIZE;
        let scale_row_bytes = latent_cols / group_size * 2;
        let rope_row_bytes = rope_cols * 2;
        let mirror = RocmMlaCpuMirror::new(self.capacity, latent_cols, group_size, rope_cols)?.finish()?;
        let hot = RocmMlaCpuHotLayer::new(context.device_id, self.capacity, hot_rows, mirror, None, None, None)?;
        self.paged_layers[layer] = Some(RocmPagedMlaLayer {
            latent: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, hot_rows * latent_cols).map_err(compute_error)?),
            latent_scales: Some(Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, hot_rows * scale_row_bytes).map_err(compute_error)?)),
            latent_group_size: group_size,
            rope: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, hot_rows * rope_row_bytes).map_err(compute_error)?),
            rows: 0,
            latent_cols,
            rope_cols,
            committed_rows: hot_rows,
            cpu_mirror: None,
            cpu_hot: Some(std::sync::Mutex::new(hot)),
        });
        Ok(())
    }

    pub(super) fn prefetch_selected_layers(&mut self, context: &RocmContext, layers: std::ops::Range<usize>, selection: &[u32], position: usize, rows: usize) -> Result<(), BackendError> {
        let hot_rows = ops::hip::options().mla_cpu_hot_rows;
        if hot_rows == 0 || rows != 1 || self.ownership != RocmKvOwnership::Full || (self.stage_decode == Some(false) && !ops::hip::options().prefill_attention_cpu) {
            return Ok(());
        }
        if selection.is_empty() || selection.len().saturating_add(rows) > hot_rows {
            return Ok(());
        }
        ops::hip::set_device(context.device_id).map_err(compute_error)?;
        let main_stream = ops::hip::active_compute_stream() as usize;
        let mut planned = Vec::with_capacity(layers.len());
        for layer in layers {
            let Some(cached) = self.paged_layers.get(layer).and_then(Option::as_ref) else { continue };
            if cached.rows != position || cached.rows <= hot_rows || cached.latent_group_size == 0 {
                continue;
            }
            self.enable_cpu_hot_layer(context, layer, hot_rows)?;
            let cached = self.paged_layers[layer].as_ref().expect("预取层已校验");
            let mut hot = cached.cpu_hot.as_ref().expect("预取 hot 已创建").lock().map_err(|_| compute_error(format!("L{layer} ROCm MLA hot 预取锁中毒")))?;
            if hot.prefetched.as_ref().is_some_and(|(start, _, _)| *start == position) {
                continue;
            }
            hot.join_prefetch(context.device_id)?;
            hot.prefetched = None;
            // 当前 token 的 KV 尚未生成，但可以先保留它最终写入的 slot。这样
            // 预取能一次产出完整 remapped selection，attention 不必再次映射上传。
            if selection.contains(&(position as u32)) {
                hot.reserve_slot(position)?;
            }
            let stream = ops::hip::cache_prefetch_stream(context.device_id, layer).map_err(compute_error)?;
            planned.push((layer, stream));
        }
        let streams = planned.iter().map(|&(_, stream)| stream).collect::<Vec<_>>();
        ops::hip::order_streams_after(context.device_id, main_stream, &streams).map_err(compute_error)?;
        let prepared = (|| {
            for (layer, stream) in planned {
                let cached = self.paged_layers[layer].as_ref().expect("预取层已校验");
                let mut hot = cached.cpu_hot.as_ref().expect("预取 hot 已创建").lock().map_err(|_| compute_error(format!("L{layer} ROCm MLA hot 预取锁中毒")))?;
                ops::hip::activate_compute_stream(context.device_id, stream).map_err(compute_error)?;
                let selection = hot.prepare_selection(context.device_id, layer, selection, &cached.latent, cached.latent_scales.as_deref().expect("Q8 hot 必有 scale"), &cached.rope)?;
                hot.prefetched = Some((position, Some(stream), selection));
            }
            Ok(())
        })();
        let restore = ops::hip::activate_compute_stream(context.device_id, main_stream).map_err(compute_error);
        restore?;
        prepared
    }

    fn enable_cpu_hot_layer(&mut self, context: &RocmContext, layer: usize, hot_rows: usize) -> Result<(), BackendError> {
        let cached = self.paged_layers.get_mut(layer).and_then(Option::as_mut).ok_or_else(|| compute_error(format!("L{layer} ROCm MLA hot 启用前 cache 尚未初始化")))?;
        if cached.cpu_hot.is_some() {
            return Ok(());
        }
        if hot_rows <= 2048 || hot_rows > self.capacity || cached.latent_group_size == 0 {
            return Err(compute_error(format!("L{layer} ROCm MLA CPU hot rows={hot_rows} capacity={} group={} 非法", self.capacity, cached.latent_group_size)));
        }
        let started = std::time::Instant::now();
        let scale_row_bytes = cached.latent_cols / cached.latent_group_size * 2;
        let rope_row_bytes = cached.rope_cols * 2;
        let mut mirror_guard = cached.cpu_mirror.take().ok_or_else(|| compute_error(format!("L{layer} ROCm MLA CPU mirror 未在 prefill 建立")))?.into_inner().map_err(|_| compute_error(format!("L{layer} ROCm MLA mirror 锁中毒")))?;
        mirror_guard.flush_feed(context.device_id, cached.rows, &cached.latent, cached.latent_scales.as_deref().expect("Q8 MLA 必有 scales"), &cached.rope)?;
        let prepared_gpu = mirror_guard.gpu.take();
        // prefill 已有可复用的设备打包区和 pinned 下载区，切换时直接交接，
        // 避免先释放 host 映射，再在首批日志回传时重新注册。
        let transfer = mirror_guard.transfer.take();
        let download = mirror_guard.available_download.take();
        let mirror = mirror_guard.finish()?;
        if mirror.rows != cached.rows {
            return Err(compute_error(format!("L{layer} ROCm MLA CPU mirror rows={}，cache rows={}", mirror.rows, cached.rows)));
        }
        // 暂存旧全量 GPU buffer；首个 selection 走 D2D gather，避开 CPU mirror 冷页 H2D。
        let warm = RocmMlaHotWarm { latent: cached.latent.clone(), scales: cached.latent_scales.clone(), rope: cached.rope.clone(), rows: cached.rows };
        let mut hot = RocmMlaCpuHotLayer::new(context.device_id, self.capacity, hot_rows, mirror, Some(warm), transfer, download)?;
        if let Some(gpu) = prepared_gpu {
            hot.gpu = Some(gpu);
            hot.warm_source = None;
        }
        cached.latent = Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, hot_rows * cached.latent_cols).map_err(compute_error)?);
        cached.latent_scales = Some(Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, hot_rows * scale_row_bytes).map_err(compute_error)?));
        cached.rope = Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, hot_rows * rope_row_bytes).map_err(compute_error)?);
        cached.committed_rows = hot_rows;
        cached.cpu_hot = Some(std::sync::Mutex::new(hot));
        eprintln!(
            "[mla-cpu-hot-enable] device={} layer={layer} history={} hot_rows={hot_rows} host_mib={:.2} gpu_mib={:.2} wall_ms={:.3}",
            context.device_id,
            cached.rows,
            cached.rows as f64 * (cached.latent_cols + scale_row_bytes + rope_row_bytes) as f64 / (1_u64 << 20) as f64,
            hot_rows as f64 * (cached.latent_cols + scale_row_bytes + rope_row_bytes) as f64 / (1_u64 << 20) as f64,
            started.elapsed().as_secs_f64() * 1e3,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn append_mla_hot(&mut self, context: &RocmContext, layer: usize, latent: &RocmTensor, rope: &RocmTensor, rope_rotation: Option<(usize, usize, crate::attention::rope::RotaryLayout, &[f32], &[f32])>) -> Result<(), BackendError> {
        let table = self.block_table.get("KV hot", self.capacity, context.device_id, self.capacity)?;
        let cached = self.paged_layers.get_mut(layer).and_then(Option::as_mut).ok_or(BackendError::UnsupportedLayer { layer })?;
        let hot_mutex = cached.cpu_hot.as_ref().ok_or_else(|| compute_error(format!("L{layer} ROCm MLA hot state 缺失")))?;
        let mut hot = hot_mutex.lock().map_err(|_| compute_error(format!("L{layer} ROCm MLA hot 锁中毒")))?;
        let prefetched = hot.prefetched.as_ref().is_some_and(|(position, _, _)| *position == cached.rows);
        hot.join_prefetch(context.device_id)?;
        hot.prepare_append_gpu_mode(context.device_id, latent.rows, [&cached.latent, cached.latent_scales.as_ref().expect("Q8 hot 必有 scales"), &cached.rope])?;
        // 日志批量让 mirror 合法滞后(≤63 行);只约束不得超前,链连续性由
        // queue_rows 自校验,消费点负责 flush。
        if latent.rows == 0 || latent.rows != rope.rows || hot.mirror.rows > cached.rows {
            return Err(compute_error(format!("L{layer} ROCm MLA hot append latent={} rope={} mirror={} cache={} 非法", latent.rows, rope.rows, hot.mirror.rows, cached.rows)));
        }
        let logical_start = cached.rows;
        let end = logical_start.checked_add(latent.rows).ok_or_else(|| compute_error("ROCm MLA hot append rows 溢出"))?;
        if end > self.capacity {
            return Err(compute_error(format!("L{layer} ROCm MLA hot append end={end} 超过容量 {}", self.capacity)));
        }
        let latent_input = latent.device.as_deref().ok_or_else(|| compute_error("ROCm MLA hot latent 缺少 device buffer"))?;
        let rope_input = rope.device.as_deref().ok_or_else(|| compute_error("ROCm MLA hot rope 缺少 device buffer"))?;
        let scales = cached.latent_scales.as_deref().expect("Q8 MLA hot 必有 scales");
        // decode 单行必须先拿窗口槽位，再把量化结果直接写到该槽位。多行
        // CPU-prefill 的 staging 路径不会建立 token→slot 映射，单行走它会在
        // 下一轮 selection 缺失刚追加的 token。
        if latent.rows == 1 {
            // 预取已为当前 selection 留出新行容量；保留 pin，避免 append
            // 驱逐刚搬到 GPU 的历史行，导致 attention 再次补传。
            let slot = if prefetched { hot.assign_slot(logical_start)? } else { hot.append_slot(logical_start)? };
            // 日志环存在时 kernel 双写窗口槽位 + 日志；mirror 喂养推迟到批量
            // 边界,非边界行不再有任何 D2D/D2H 流操作。
            let (log_latent, log_scales, log_rope, log_row) = match hot.mirror_log.as_ref() {
                Some((log_latent, log_scales, log_rope)) => (Arc::clone(log_latent), Arc::clone(log_scales), Arc::clone(log_rope), logical_start % MLA_HOT_LOG_ROWS),
                None => (cached.latent.clone(), cached.latent_scales.as_ref().expect("Q8 hot 必有 scales").clone(), cached.rope.clone(), 0),
            };
            let use_log = hot.mirror_log.is_some();
            match rope_rotation {
                Some((position, rotary_dim, layout, cos, sin)) => {
                    if position != logical_start {
                        return Err(compute_error(format!("L{layer} ROCm MLA hot RoPE position={position}，期望 {logical_start}")));
                    }
                    if use_log {
                        ops::hip::try_paged_cache_append_mla_rope_at_f32_q8_bf16_with_log(
                            context.device_id,
                            latent_input,
                            &cached.latent,
                            scales,
                            rope_input,
                            &cached.rope,
                            &table,
                            slot,
                            position,
                            1,
                            latent.cols,
                            rope.cols,
                            rotary_dim,
                            layout,
                            cached.latent_group_size,
                            ROCM_KV_BLOCK_SIZE,
                            cos,
                            sin,
                            &log_latent,
                            &log_scales,
                            &log_rope,
                            log_row,
                        )
                        .map_err(compute_error)?;
                    } else {
                        ops::hip::try_paged_cache_append_mla_rope_at_f32_q8_bf16(
                            context.device_id,
                            latent_input,
                            &cached.latent,
                            scales,
                            rope_input,
                            &cached.rope,
                            &table,
                            slot,
                            position,
                            1,
                            latent.cols,
                            rope.cols,
                            rotary_dim,
                            layout,
                            cached.latent_group_size,
                            ROCM_KV_BLOCK_SIZE,
                            cos,
                            sin,
                        )
                        .map_err(compute_error)?;
                    }
                }
                None => {
                    if use_log {
                        ops::hip::try_paged_cache_append_mla_f32_q8_bf16_with_log(
                            context.device_id,
                            latent_input,
                            &cached.latent,
                            scales,
                            rope_input,
                            &cached.rope,
                            &table,
                            slot,
                            1,
                            latent.cols,
                            rope.cols,
                            cached.latent_group_size,
                            ROCM_KV_BLOCK_SIZE,
                            &log_latent,
                            &log_scales,
                            &log_rope,
                            log_row,
                        )
                        .map_err(compute_error)?;
                    } else {
                        ops::hip::try_paged_cache_append_mla_f32_q8_bf16(
                            context.device_id,
                            latent_input,
                            &cached.latent,
                            scales,
                            rope_input,
                            &cached.rope,
                            &table,
                            slot,
                            1,
                            latent.cols,
                            rope.cols,
                            cached.latent_group_size,
                            ROCM_KV_BLOCK_SIZE,
                        )
                        .map_err(compute_error)?
                    }
                }
            }
            hot.log_fed_rows = end;
            if use_log {
                if logical_start % MLA_HOT_LOG_ROWS == MLA_HOT_LOG_ROWS - 1 {
                    // 链可能已被 selection miss 的 flush 部分推进,从链尾补到当前,
                    // 不能按固定 64 行跨度重发。
                    hot.flush_log_feed(context.device_id)?;
                }
            } else {
                hot.queue_rows(context.device_id, logical_start, 1, slot, &cached.latent, scales, &cached.rope)?;
            }
            cached.rows = end;
            return Ok(());
        }
        // 小批 verify/catch-up 也直接追加到热槽和日志。否则刚生成的行会先
        // D2H，再由 selection miss 原样上传，强制排空本层计算流。
        if latent.rows <= MLA_HOT_LOG_ROWS && hot.mirror_log.is_some() {
            hot.reset_pins();
            let mut row = 0;
            while row < latent.rows {
                let start = logical_start + row;
                let log_row = start % MLA_HOT_LOG_ROWS;
                let rows = (MLA_HOT_LOG_ROWS - log_row).min(latent.rows - row);
                let mut slots = Vec::with_capacity(rows);
                for token in start..start + rows {
                    let slot = hot.assign_slot(token)?;
                    hot.pin(slot);
                    slots.push(slot as u32);
                }
                let (slot_table, cache_position, block_size) = if hot.uses_gpu() {
                    // recent 环已在回绕处分段，本段物理连续，直接复用常驻页表。
                    (table.clone(), slots[0] as usize, ROCM_KV_BLOCK_SIZE)
                } else {
                    let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr().cast(), slots.len() * 4) };
                    let slot_table = Arc::new(ops::hip::DeviceBuffer::upload_ordered(context.device_id, slot_bytes).map_err(compute_error)?);
                    slot_table.enqueue_small_deferred_upload().map_err(compute_error)?;
                    slot_table.retain_for_active_stage();
                    (slot_table, 0, 1)
                };
                let input = ops::hip::DeviceBuffer::view(latent.device.as_ref().expect("hot input 已校验").clone(), row * latent.cols * 4, rows * latent.cols * 4).map_err(compute_error)?;
                let rope_input = ops::hip::DeviceBuffer::view(rope.device.as_ref().expect("hot rope 已校验").clone(), row * rope.cols * 4, rows * rope.cols * 4).map_err(compute_error)?;
                let (log_latent, log_scales, log_rope) = hot.mirror_log.as_ref().expect("日志已校验");
                match rope_rotation {
                    Some((position, rotary_dim, layout, cos, sin)) => {
                        if position != logical_start {
                            return Err(compute_error(format!("L{layer} ROCm MLA hot RoPE position={position}，期望 {logical_start}")));
                        }
                        ops::hip::try_paged_cache_append_mla_rope_at_f32_q8_bf16_with_log(
                            context.device_id,
                            &input,
                            &cached.latent,
                            scales,
                            &rope_input,
                            &cached.rope,
                            &slot_table,
                            cache_position,
                            start,
                            rows,
                            latent.cols,
                            rope.cols,
                            rotary_dim,
                            layout,
                            cached.latent_group_size,
                            block_size,
                            cos,
                            sin,
                            log_latent,
                            log_scales,
                            log_rope,
                            log_row,
                        )
                        .map_err(compute_error)?;
                    }
                    None => ops::hip::try_paged_cache_append_mla_f32_q8_bf16_with_log(
                        context.device_id,
                        &input,
                        &cached.latent,
                        scales,
                        &rope_input,
                        &cached.rope,
                        &slot_table,
                        cache_position,
                        rows,
                        latent.cols,
                        rope.cols,
                        cached.latent_group_size,
                        block_size,
                        log_latent,
                        log_scales,
                        log_rope,
                        log_row,
                    )
                    .map_err(compute_error)?,
                }
                hot.log_fed_rows = start + rows;
                if log_row + rows == MLA_HOT_LOG_ROWS {
                    hot.flush_log_feed(context.device_id)?;
                }
                row += rows;
            }
            cached.rows = end;
            return Ok(());
        }
        // 大批多行 append 使用连续 staging；先补齐此前单行日志，保持同一
        // mirror 链连续，随后再把 staging 的多行记录接上。
        hot.flush_log_feed(context.device_id)?;
        hot.finish_pending()?;
        let direct = end <= cached.committed_rows;
        let latent_bytes = latent.rows.checked_mul(cached.latent_cols).ok_or_else(|| compute_error("ROCm MLA hot staging latent 大小溢出"))?;
        let scale_row_bytes = cached.latent_cols / cached.latent_group_size * 2;
        let scale_bytes = latent.rows.checked_mul(scale_row_bytes).ok_or_else(|| compute_error("ROCm MLA hot staging scale 大小溢出"))?;
        let rope_row_bytes = cached.rope_cols * 2;
        let rope_bytes = latent.rows.checked_mul(rope_row_bytes).ok_or_else(|| compute_error("ROCm MLA hot staging rope 大小溢出"))?;
        // 下一次 append 开头会先等待本次 D2H；因此越过 hot 容量后可安全
        // 常驻复用这三个 staging，避免每层、每 chunk 重新 hipMalloc。
        let staging_latent = (!direct).then(|| RocmMlaCpuHotLayer::staging_buffer(&mut hot.staging_latent, context.device_id, latent_bytes)).transpose()?;
        let staging_scales = (!direct).then(|| RocmMlaCpuHotLayer::staging_buffer(&mut hot.staging_scales, context.device_id, scale_bytes)).transpose()?;
        let staging_rope = (!direct).then(|| RocmMlaCpuHotLayer::staging_buffer(&mut hot.staging_rope, context.device_id, rope_bytes)).transpose()?;
        let target_latent = staging_latent.as_deref().unwrap_or(&cached.latent);
        let target_scales = staging_scales.as_deref().unwrap_or(scales);
        let target_rope = staging_rope.as_deref().unwrap_or(&cached.rope);
        let cache_position = if direct { logical_start } else { 0 };
        match rope_rotation {
            Some((position, rotary_dim, layout, cos, sin)) => {
                if position != logical_start {
                    return Err(compute_error(format!("L{layer} ROCm MLA hot RoPE position={position}，期望 {logical_start}")));
                }
                ops::hip::try_paged_cache_append_mla_rope_at_f32_q8_bf16(
                    context.device_id,
                    latent_input,
                    target_latent,
                    target_scales,
                    rope_input,
                    target_rope,
                    &table,
                    cache_position,
                    position,
                    latent.rows,
                    latent.cols,
                    rope.cols,
                    rotary_dim,
                    layout,
                    cached.latent_group_size,
                    ROCM_KV_BLOCK_SIZE,
                    cos,
                    sin,
                )
                .map_err(compute_error)?;
            }
            None => ops::hip::try_paged_cache_append_mla_f32_q8_bf16(
                context.device_id,
                latent_input,
                target_latent,
                target_scales,
                rope_input,
                target_rope,
                &table,
                cache_position,
                latent.rows,
                latent.cols,
                rope.cols,
                cached.latent_group_size,
                ROCM_KV_BLOCK_SIZE,
            )
            .map_err(compute_error)?,
        }
        if direct {
            hot.map_direct_range(logical_start, latent.rows)?;
        }
        hot.queue_rows(context.device_id, logical_start, latent.rows, cache_position, target_latent, target_scales, target_rope)?;
        hot.log_fed_rows = end;
        cached.rows = end;
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn append_mla_inner(&mut self, context: &RocmContext, layer: usize, latent: &RocmTensor, rope: &RocmTensor, rope_rotation: Option<(usize, usize, crate::attention::rope::RotaryLayout, &[f32], &[f32])>) -> Result<(), BackendError> {
        if self.ownership != RocmKvOwnership::Full {
            return Err(compute_error(format!("L{layer} block-sharded MLA 必须走 cooperative append")));
        }
        if latent.rows != rope.rows || latent.rows == 0 {
            return Err(compute_error(format!("L{layer} ROCm paged MLA append 行数非法")));
        }
        self.prepare_mla_append(context, layer, latent.rows)?;
        let hot_rows = ops::hip::options().mla_cpu_hot_rows;
        // GPU prefill 始终写普通 paged cache 并建立 CPU mirror；只有 CPU-prefill
        // 模式才从首个多行 chunk 直接建立 hot 窗口。混用两种语义会在长
        // prefill 跨过窗口容量后把仍被 selection pin 的槽位当作 append 目标。
        if hot_rows != 0 && ops::hip::options().prefill_attention_cpu && self.paged_layers.get(layer).is_some_and(Option::is_none) && latent.rows > 1 {
            self.initialize_cpu_hot_layer(context, layer, hot_rows, latent.cols, rope.cols)?;
        } else if hot_rows != 0
            && (self.stage_decode != Some(false) || ops::hip::options().prefill_attention_cpu)
            && (latent.rows == 1 || gpu_hot_selection(latent.rows))
            && self.paged_layers.get(layer).and_then(Option::as_ref).is_some_and(|cached| cached.rows > hot_rows && cached.cpu_hot.is_none())
        {
            self.enable_cpu_hot_layer(context, layer, hot_rows)?;
        }
        if self.paged_layers.get(layer).and_then(Option::as_ref).is_some_and(|cached| cached.cpu_hot.is_some()) {
            return self.append_mla_hot(context, layer, latent, rope, rope_rotation);
        }
        let slot = self.paged_layers.get(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        let (rows, committed_rows, latent_group_size) = if let Some(cached) = slot {
            if cached.latent_cols != latent.cols || cached.rope_cols != rope.cols {
                return Err(compute_error(format!("L{layer} ROCm paged MLA shape 非法: latent={}/{} rope={}/{}", latent.cols, cached.latent_cols, rope.cols, cached.rope_cols)));
            }
            (cached.rows, cached.committed_rows, cached.latent_group_size)
        } else {
            (0, 0, if ops::hip::options().kv_f16 { 0 } else { crate::kv_cache::DEFAULT_GROUP_SIZE })
        };
        let end = rows.checked_add(latent.rows).ok_or_else(|| compute_error("ROCm paged MLA rows 溢出"))?;
        if end > self.capacity {
            return Err(compute_error(format!("L{layer} ROCm paged MLA end={end} 超过逻辑上限 {}", self.capacity)));
        }
        let next_committed = committed_cache_rows(committed_rows, end, self.capacity)?;
        let table = self.block_table.get("KV", self.capacity, context.device_id, end)?;
        let slot = self.paged_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if slot.is_none() {
            if latent_group_size != 0 && !latent.cols.is_multiple_of(latent_group_size) {
                return Err(compute_error(format!("L{layer} ROCm paged latent cols={} 不能按 Q8G{} 分组", latent.cols, latent_group_size)));
            }
            let latent_element_bytes = if latent_group_size == 0 { 2 } else { 1 };
            let latent_bytes = next_committed.checked_mul(latent.cols).and_then(|n| n.checked_mul(latent_element_bytes)).ok_or_else(|| compute_error("ROCm paged latent 大小溢出"))?;
            let latent_scale_bytes = if latent_group_size == 0 { 0 } else { next_committed.checked_mul(latent.cols / latent_group_size).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm paged latent scale 大小溢出"))? };
            let rope_bytes = next_committed.checked_mul(rope.cols).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm paged rope 大小溢出"))?;
            *slot = Some(RocmPagedMlaLayer {
                latent: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, latent_bytes).map_err(compute_error)?),
                latent_scales: if latent_group_size == 0 { None } else { Some(Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, latent_scale_bytes).map_err(compute_error)?)) },
                latent_group_size,
                rope: Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, rope_bytes).map_err(compute_error)?),
                rows: 0,
                latent_cols: latent.cols,
                rope_cols: rope.cols,
                committed_rows: next_committed,
                cpu_mirror: None,
                cpu_hot: None,
            });
        }
        let cached = slot.as_mut().unwrap();
        if next_committed > cached.committed_rows {
            let latent_element_bytes = if cached.latent_group_size == 0 { 2 } else { 1 };
            cached.latent = grow_cache_buffer(context.device_id, &cached.latent, cached.rows * cached.latent_cols * latent_element_bytes, next_committed * cached.latent_cols * latent_element_bytes)?;
            if let Some(scales) = &cached.latent_scales {
                let groups = cached.latent_cols / cached.latent_group_size;
                cached.latent_scales = Some(grow_cache_buffer(context.device_id, scales, cached.rows * groups * 2, next_committed * groups * 2)?);
            }
            cached.rope = grow_cache_buffer(context.device_id, &cached.rope, cached.rows * cached.rope_cols * 2, next_committed * cached.rope_cols * 2)?;
            cached.committed_rows = next_committed;
        }
        let latent_input = latent.device.as_deref().ok_or_else(|| compute_error("ROCm paged latent 缺少 device buffer"))?;
        let rope_input = rope.device.as_deref().ok_or_else(|| compute_error("ROCm paged rope 缺少 device buffer"))?;
        if let Some(scales) = cached.latent_scales.as_deref() {
            if let Some((position, rotary_dim, layout, cos, sin)) = rope_rotation {
                if position != cached.rows {
                    return Err(compute_error(format!("L{layer} ROCm fused MLA RoPE position={position}，cache rows={}", cached.rows)));
                }
                ops::hip::try_paged_cache_append_mla_rope_f32_q8_bf16(
                    context.device_id,
                    latent_input,
                    &cached.latent,
                    scales,
                    rope_input,
                    &cached.rope,
                    &table,
                    cached.rows,
                    latent.rows,
                    latent.cols,
                    rope.cols,
                    rotary_dim,
                    layout,
                    cached.latent_group_size,
                    ROCM_KV_BLOCK_SIZE,
                    cos,
                    sin,
                )
                .map_err(compute_error)?;
            } else {
                ops::hip::try_paged_cache_append_mla_f32_q8_bf16(
                    context.device_id,
                    latent_input,
                    &cached.latent,
                    scales,
                    rope_input,
                    &cached.rope,
                    &table,
                    cached.rows,
                    latent.rows,
                    latent.cols,
                    rope.cols,
                    cached.latent_group_size,
                    ROCM_KV_BLOCK_SIZE,
                )
                .map_err(compute_error)?;
            }
        } else {
            if rope_rotation.is_some() {
                return Err(compute_error(format!("L{layer} ROCm BF16 MLA cache 不支持 fused RoPE append")));
            }
            ops::hip::try_paged_cache_append_f32_bf16(context.device_id, latent_input, &cached.latent, &table, cached.rows, latent.rows, latent.cols, ROCM_KV_BLOCK_SIZE).map_err(compute_error)?;
            ops::hip::try_paged_cache_append_f32_bf16(context.device_id, rope_input, &cached.rope, &table, cached.rows, rope.rows, rope.cols, ROCM_KV_BLOCK_SIZE).map_err(compute_error)?;
        }
        let appended_rows = latent.rows;
        cached.rows += appended_rows;
        if hot_rows != 0 || ops::hip::options().prefill_attention_cpu || mla_cpu_mirror_enabled() {
            if cached.latent_group_size == 0 {
                return Err(compute_error(format!("L{layer} ROCm MLA CPU mirror 只支持 Q8 cache")));
            }
            if cached.cpu_mirror.is_none() {
                cached.cpu_mirror = Some(std::sync::Mutex::new(RocmMlaCpuMirror::new(self.capacity, cached.latent_cols, cached.latent_group_size, cached.rope_cols)?));
            }
            cached.cpu_mirror.as_ref().expect("刚创建").lock().map_err(|_| compute_error(format!("L{layer} ROCm MLA mirror 锁中毒")))?.feed(
                context.device_id,
                self.capacity,
                cached.rows,
                &cached.latent,
                cached.latent_scales.as_deref().expect("Q8 MLA 必有 scales"),
                &cached.rope,
            )?;
        }
        Ok(())
    }
}

impl std::ops::Deref for RocmKvCache {
    type Target = CpuKvCache;
    fn deref(&self) -> &Self::Target {
        &self.cpu
    }
}

impl std::ops::DerefMut for RocmKvCache {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cpu
    }
}

/// MLA 层的 host 序列化形态：纯字节 + shape 元数据，无 device 句柄。
/// 供整 session KV snapshot 落盘/换入。
#[derive(Clone)]
pub struct MlaLayerSerde {
    pub rows: usize,
    pub ownership: RocmKvOwnership,
    pub latent_cols: usize,
    pub rope_cols: usize,
    /// 0 表示 BF16 模式（latent 无 scales）；非 0 = Q8 分组大小。
    pub latent_group_size: usize,
    /// latent codes：Q8 = rows×latent_cols；BF16 = rows×latent_cols×2。
    pub latent: Vec<u8>,
    /// Q8 scales：rows×(latent_cols/group)×2；BF16 = None。
    pub latent_scales: Option<Vec<u8>>,
    /// rope：rows×rope_cols×2（BF16）。
    pub rope: Vec<u8>,
}

const MLA_HOT_INVALID: u32 = u32::MAX;

struct RocmMlaCpuMirror {
    gpu: Option<super::kv_cache_gpu_hot::GpuHotCache>,
    mirror: MlaLayerSerde,
    transfer: Option<Arc<ops::hip::DeviceBuffer>>,
    available_download: Option<ops::hip::AsyncHostDownload>,
    pending_download: Option<(usize, usize, ops::hip::AsyncHostDownload)>,
    /// cache 已喂给 mirror 链（含 pending）的行数。单行 decode 按固定行距
    /// 批量触发 D2H，把每 token 每层的 PCIe 流操作摊薄；启用热窗/截断/导出
    /// 前用 flush_feed 补齐尾部。
    fed_rows: usize,
}

/// 单行 decode 的 mirror 喂养批量：流上每多一次 PCIe 操作约 30µs，逐行
/// 喂养会把 156 次/token 全部串进关键路径。
const MLA_MIRROR_FEED_BATCH_ROWS: usize = 64;
const MLA_HOT_LOG_ROWS: usize = 64;

pub(super) fn gpu_hot_selection(rows: usize) -> bool {
    let options = ops::hip::options();
    options.mla_cpu_hot_rows != 0 && !options.dsa_cpu_select && !options.prefill_attention_cpu && (1..=MLA_HOT_LOG_ROWS).contains(&rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, BackendResources};

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_hot_storage_rejects_oversized_prefill_pool_blocks() {
        const HOT: usize = 2304;
        const LARGE: usize = 8 * 1024 * 1024;
        ops::hip::configure(ops::hip::RocmOptions { mla_cpu_hot_rows: HOT, memory_pool: true, ..Default::default() }).unwrap();
        let context = RocmContext::new(0).unwrap();
        ops::hip::enable_device_buffer_reuse();
        // 模拟前一请求退回池的大块 prefill activation。
        let buffers = (0..8).map(|_| ops::hip::DeviceBuffer::allocate_reusable(0, LARGE).unwrap()).collect::<Vec<_>>();
        drop(buffers);
        context.synchronize().unwrap();
        let borrowed = ops::hip::DeviceBuffer::allocate(0, 1024).unwrap();
        assert_eq!(borrowed.allocation_bytes(), LARGE, "确认普通池路径会把大块借给小请求");
        drop(borrowed);
        context.synchronize().unwrap();
        let mut cache = RocmKvCache::with_capacity(1, 4096);
        cache.initialize_cpu_hot_layer(&context, 0, HOT, 512, 64).unwrap();
        let layer = cache.paged_layers[0].as_ref().unwrap();
        for buffer in [&layer.latent, layer.latent_scales.as_ref().unwrap(), &layer.rope] {
            assert_eq!(buffer.allocation_bytes(), buffer.bytes(), "长期 KV 不能留住大块 prefill allocation");
        }
        let hot = layer.cpu_hot.as_ref().unwrap().lock().unwrap();
        assert_eq!(hot.transfer.allocation_bytes(), 656);
        assert_eq!(cache.allocated_bytes(), (HOT * 656) as u64);
    }

    #[test]
    #[ignore = "需要两张 ROCm GPU"]
    fn mla_large_append_restores_gpu_prefix_and_keeps_prefill_tail_resident() {
        const LATENT: usize = 512;
        const ROPE: usize = 64;
        ops::hip::configure(ops::hip::RocmOptions { mla_cpu_hot_rows: 2304, ..Default::default() }).unwrap();
        let owner = RocmContext::new(0).unwrap();
        let peer = RocmContext::new(1).unwrap();
        let mut cache = RocmKvCache::with_capacity(1, 8192);
        let mut reference = RocmKvCache::with_capacity(1, 8192);
        reference.set_stage_decode(false).unwrap();
        cache.set_stage_decode(false).unwrap();
        owner.activate().unwrap();
        let replica = cache.ensure_operator_peer(&owner, &peer).unwrap();
        let mut position = 0;
        for (phase, (decode, rows)) in [(false, 2368), (true, 3), (true, 61), (true, 4), (false, 128), (false, 3), (false, 1), (true, 4), (true, 60)].into_iter().enumerate() {
            if phase == 3 {
                position -= 2;
                cache.truncate_rows(position).unwrap();
                reference.truncate_rows(position).unwrap();
            }
            cache.set_stage_decode(decode).unwrap();
            if phase == 4 {
                let cached = cache.paged_layers[0].as_ref().unwrap();
                assert!(cached.cpu_hot.as_ref().unwrap().lock().unwrap().mirror.rows < cached.rows, "恢复必须包含尚未 D2H 的 recent 尾部");
                // 超出容量时不能破坏热窗，随后合法的恢复仍能继续。
                assert!(cache.prepare_mla_append(&owner, 0, 8192).is_err());
                assert!(cache.operator_layer_is_hot(0));
            }
            owner.activate().unwrap();
            let latent = owner.tensor_from_f32((0..rows * LATENT).map(|i| ((i + position * LATENT + phase * 13) % 127) as f32 / 32.0 - 2.0).collect(), rows, LATENT).unwrap();
            let rope = owner.tensor_from_f32((0..rows * ROPE).map(|i| ((i + position * ROPE + phase * 17) % 97) as f32 / 32.0 - 1.5).collect(), rows, ROPE).unwrap();
            reference.append_mla(&owner, 0, &latent, &rope).unwrap();
            cache.append_mla(&owner, 0, &latent, &rope).unwrap();
            let _sources = cache.replicate_operator_mla_append(&owner, &peer, 0, position, rows).unwrap();
            peer.activate().unwrap();
            replica.lock().unwrap().feed_peer_mirror_layer(&peer, 0).unwrap();
            position += rows;
            assert_eq!(cache.operator_layer_is_hot(0), decode, "owner phase={phase}");
            assert_eq!(replica.lock().unwrap().operator_layer_is_hot(0), decode, "peer phase={phase}");
            if !decode {
                // 直接检查连续 GPU 字节，不能用 CPU mirror 导出来掩盖错误的恢复。
                let expected = reference.paged_layers[0].as_ref().unwrap();
                let peer_cache = replica.lock().unwrap();
                for actual in [cache.paged_layers[0].as_ref().unwrap(), peer_cache.paged_layers[0].as_ref().unwrap()] {
                    for (source, target, bytes) in [
                        (&expected.latent, &actual.latent, position * LATENT),
                        (expected.latent_scales.as_ref().unwrap(), actual.latent_scales.as_ref().unwrap(), position * LATENT / 64 * 2),
                        (&expected.rope, &actual.rope, position * ROPE * 2),
                    ] {
                        let mut left = vec![0; bytes];
                        let mut right = vec![0; bytes];
                        source.copy_to_host(&mut left).unwrap();
                        target.copy_to_host(&mut right).unwrap();
                        assert_eq!(left, right, "phase={phase} device={}", target.device_id());
                    }
                }
            }
        }
        let expected = reference.download_layers().unwrap().remove(0).unwrap();
        for actual in [cache.download_layers().unwrap().remove(0).unwrap(), replica.lock().unwrap().download_layers().unwrap().remove(0).unwrap()] {
            assert_eq!(expected.latent, actual.latent);
            assert_eq!(expected.latent_scales, actual.latent_scales);
            assert_eq!(expected.rope, actual.rope);
        }
    }

    #[test]
    #[ignore = "需要两张 ROCm GPU"]
    fn mla_hot_packed_peer_multi_append_and_rollback_preserve_bytes() {
        const LATENT: usize = 512;
        const ROPE: usize = 64;
        ops::hip::configure(ops::hip::RocmOptions { mla_cpu_hot_rows: 2304, ..Default::default() }).unwrap();
        let owner = RocmContext::new(0).unwrap();
        let peer = RocmContext::new(1).unwrap();
        let mut cache = RocmKvCache::with_capacity(1, 4096);
        owner.activate().unwrap();
        let replica = cache.ensure_operator_peer(&owner, &peer).unwrap();
        let mut position = 0;
        let mut prefill_transfer = 0;
        for (phase, rows) in [2368, 1, 3, 59, 4, 64, 2, 4, 60].into_iter().enumerate() {
            if phase == 7 {
                position -= 3;
                cache.truncate_rows(position).unwrap();
            }
            owner.activate().unwrap();
            let latent = (0..rows * LATENT).map(|i| ((i + position * LATENT + phase * 13) % 127) as f32 / 32.0 - 2.0).collect();
            let rope = (0..rows * ROPE).map(|i| ((i + position * ROPE + phase * 17) % 97) as f32 / 32.0 - 1.5).collect();
            cache.append_mla(&owner, 0, &owner.tensor_from_f32(latent, rows, LATENT).unwrap(), &owner.tensor_from_f32(rope, rows, ROPE).unwrap()).unwrap();
            let _sources = cache.replicate_operator_mla_append(&owner, &peer, 0, position, rows).unwrap();
            peer.activate().unwrap();
            replica.lock().unwrap().feed_peer_mirror_layer(&peer, 0).unwrap();
            position += rows;
            if phase == 0 {
                prefill_transfer = cache.paged_layers[0].as_ref().unwrap().cpu_mirror.as_ref().unwrap().lock().unwrap().transfer.as_ref().unwrap().device_pointer();
                continue;
            }
            if phase == 1 {
                let hot = cache.paged_layers[0].as_ref().unwrap().cpu_hot.as_ref().unwrap().lock().unwrap();
                assert_eq!(hot.transfer.device_pointer(), prefill_transfer);
                assert!(hot.available_download.is_some());
            }
            // 每次追加后直接 gather，不能先导出/flush，以暴露跨环时 mirror 滞后。
            let tokens = [0_u32, (position - rows) as u32, (position - 1) as u32];
            let mut collected = Vec::new();
            for (context, layer) in [(&owner, cache.paged_layers[0].as_ref().unwrap()), (&peer, replica.lock().unwrap().paged_layers[0].as_ref().unwrap())] {
                context.activate().unwrap();
                let selection = ops::hip::DeviceBuffer::upload(context.device_id, unsafe { std::slice::from_raw_parts(tokens.as_ptr().cast(), tokens.len() * 4) }).unwrap();
                let mut hot = layer.cpu_hot.as_ref().unwrap().lock().unwrap();
                let (gathered, _, _, count) = hot.prepare_device_selection(context.device_id, &selection, [&layer.latent, layer.latent_scales.as_deref().unwrap(), &layer.rope], 1, tokens.len(), position).unwrap();
                let mut planes = Vec::new();
                for (buffer, row_bytes) in gathered.iter().zip([LATENT, LATENT / 64 * 2, ROPE * 2]) {
                    let mut bytes = vec![0; count * row_bytes];
                    buffer.copy_to_host(&mut bytes).unwrap();
                    planes.push(bytes);
                }
                collected.push(planes);
            }
            assert_eq!(collected[0], collected[1], "phase={phase} rows={rows} position={position}");
        }
        let expected = cache.download_layers().unwrap().remove(0).unwrap();
        let actual = replica.lock().unwrap().download_layers().unwrap().remove(0).unwrap();
        assert_eq!(expected.latent, actual.latent);
        assert_eq!(expected.latent_scales, actual.latent_scales);
        assert_eq!(expected.rope, actual.rope);
    }

    #[test]
    #[ignore = "需要 ROCm GPU"]
    fn mla_hot_log_wrap_rollback_and_multi_append_preserve_bytes() {
        const LATENT: usize = 512;
        const ROPE: usize = 64;
        const HOT: usize = 2304;
        ops::hip::configure(ops::hip::RocmOptions { mla_cpu_hot_rows: HOT, ..Default::default() }).unwrap();
        let context = RocmContext::new(0).unwrap();
        let mut cache = RocmKvCache::with_capacity(1, 4096);
        cache.initialize_cpu_hot_layer(&context, 0, HOT, LATENT, ROPE).unwrap();
        let mut expected_latent = Vec::new();
        let mut expected_rope = Vec::new();
        let cos = (0..4096 * (ROPE / 2)).map(|i| ((i / (ROPE / 2)) % 7 + 1) as f32 / 8.0).collect::<Vec<_>>();
        let sin = vec![0.0; cos.len()];
        // 先越过窗口，再在非 64 行边界混合单行 decode 和多行 verify。
        for (phase, chunks) in [vec![2368], vec![1; 67], vec![2, 64, 7], vec![1; 3], vec![1; 70], vec![128, 1, 64, 7]].into_iter().enumerate() {
            if phase == 4 {
                let keep = expected_latent.len() / LATENT - 5;
                let cached = cache.paged_layers[0].as_mut().unwrap();
                let mut hot = cached.cpu_hot.as_ref().unwrap().lock().unwrap();
                hot.truncate(keep).unwrap();
                assert!(hot.mirror.rows < keep, "GPU MTP 回滚不应把不足一批的 recent 尾部强制 D2H");
                drop(hot);
                cached.rows = keep;
                expected_latent.truncate(keep * LATENT);
                expected_rope.truncate(keep * ROPE);
            }
            for rows in chunks {
                let start = expected_latent.len() / LATENT;
                let latent = (0..rows * LATENT).map(|i| (((start * LATENT + i) * 7 + phase * 11) % 127) as f32 / 32.0 - 2.0).collect::<Vec<_>>();
                let rope = (0..rows * ROPE).map(|i| (((start * ROPE + i) * 13 + phase * 17) % 97) as f32 / 32.0 - 1.5).collect::<Vec<_>>();
                let rotation = (phase == 2).then_some((start, ROPE, crate::attention::rope::RotaryLayout::SplitHalf, cos.as_slice(), sin.as_slice()));
                cache.append_mla_hot(&context, 0, &context.tensor_from_f32(latent.clone(), rows, LATENT).unwrap(), &context.tensor_from_f32(rope.clone(), rows, ROPE).unwrap(), rotation).unwrap();
                expected_latent.extend(latent);
                expected_rope.extend(rope.into_iter().enumerate().map(|(i, value)| if rotation.is_some() { value * cos[(start + i / ROPE) * (ROPE / 2)] } else { value }));
            }
        }
        let rows = expected_latent.len() / LATENT;
        let actual = cache.download_layers().unwrap().remove(0).unwrap();
        let input = context.tensor_from_f32(expected_latent, rows, LATENT).unwrap();
        let rope_input = context.tensor_from_f32(expected_rope, rows, ROPE).unwrap();
        let latent = ops::hip::DeviceBuffer::allocate(0, rows * LATENT).unwrap();
        let scales = ops::hip::DeviceBuffer::allocate(0, rows * (LATENT / 64) * 2).unwrap();
        let rope = ops::hip::DeviceBuffer::allocate(0, rows * ROPE * 2).unwrap();
        let table = cache.block_table.buffer().unwrap();
        ops::hip::try_paged_cache_append_mla_f32_q8_bf16(0, input.device.as_deref().unwrap(), &latent, &scales, rope_input.device.as_deref().unwrap(), &rope, table, 0, rows, LATENT, ROPE, 64, ROCM_KV_BLOCK_SIZE).unwrap();
        let mut bytes = vec![0; actual.latent.len()];
        latent.copy_to_host(&mut bytes).unwrap();
        assert_eq!(actual.latent, bytes);
        let mut bytes = vec![0; actual.latent_scales.as_ref().unwrap().len()];
        scales.copy_to_host(&mut bytes).unwrap();
        assert_eq!(actual.latent_scales.unwrap(), bytes);
        let mut bytes = vec![0; actual.rope.len()];
        rope.copy_to_host(&mut bytes).unwrap();
        assert_eq!(actual.rope, bytes);
        let hot = cache.paged_layers[0].as_ref().unwrap().cpu_hot.as_ref().unwrap().lock().unwrap();
        assert_eq!(hot.log_fed_rows, rows);
        assert_eq!(hot.mirror.rows, rows);
        drop(hot);
        cache.truncate_rows(32).unwrap();
        assert!(cache.paged_layers[0].as_ref().unwrap().cpu_hot.is_none());
        let short = cache.download_layers().unwrap().remove(0).unwrap();
        let mut prefix = vec![0; 32 * LATENT];
        latent.copy_to_host(&mut prefix).unwrap();
        assert_eq!(short.latent, prefix);
    }
}

impl RocmMlaCpuMirror {
    fn new(capacity: usize, latent_cols: usize, latent_group_size: usize, rope_cols: usize) -> Result<Self, BackendError> {
        if latent_group_size == 0 || !latent_cols.is_multiple_of(latent_group_size) {
            return Err(compute_error(format!("ROCm MLA CPU mirror shape latent={latent_cols} group={latent_group_size} 非法")));
        }
        Ok(Self {
            gpu: None,
            fed_rows: 0,
            mirror: MlaLayerSerde {
                rows: 0,
                ownership: RocmKvOwnership::Full,
                latent_cols,
                rope_cols,
                latent_group_size,
                latent: Vec::with_capacity(capacity.saturating_mul(latent_cols)),
                latent_scales: Some(Vec::with_capacity(capacity.saturating_mul(latent_cols / latent_group_size).saturating_mul(2))),
                rope: Vec::with_capacity(capacity.saturating_mul(rope_cols).saturating_mul(2)),
            },
            transfer: None,
            available_download: None,
            pending_download: None,
        })
    }

    fn from_record(mirror: MlaLayerSerde) -> Self {
        let fed_rows = mirror.rows;
        Self { gpu: None, mirror, transfer: None, available_download: None, pending_download: None, fed_rows }
    }

    /// 全量期的批量喂养入口：多行 chunk 立即整段入队，单行 decode 累积到
    /// 批量边界才发一次 D2H。
    fn feed(&mut self, device_id: i32, capacity: usize, cached_rows: usize, latent: &ops::hip::DeviceBuffer, scales: &ops::hip::DeviceBuffer, rope: &ops::hip::DeviceBuffer) -> Result<(), BackendError> {
        // 历史真正超过热窗时才注册，在余下 GPU prefill 中准备好 RAM 读取；
        // 短请求不会为尚未使用的换出路径支付页锁定成本。
        if gpu_hot_selection(1) && self.gpu.is_none() && cached_rows > ops::hip::options().mla_cpu_hot_rows {
            let mut gpu = super::kv_cache_gpu_hot::GpuHotCache::new(device_id, ops::hip::options().mla_cpu_hot_rows - MLA_HOT_LOG_ROWS)?;
            gpu.register_history(device_id, &mut self.mirror, capacity, cached_rows)?;
            self.gpu = Some(gpu);
        }
        let chain_end = self.pending_download.as_ref().map_or(self.mirror.rows, |pending| pending.0 + pending.1);
        if chain_end != self.fed_rows || self.fed_rows > cached_rows {
            return Err(compute_error(format!("ROCm MLA mirror feed 链尾={chain_end} fed={} cache={cached_rows} 不连续", self.fed_rows)));
        }
        if cached_rows - self.fed_rows > 1 {
            self.queue(device_id, self.fed_rows, cached_rows - self.fed_rows, latent, scales, rope)?;
            self.fed_rows = cached_rows;
            return Ok(());
        }
        while cached_rows.saturating_sub(self.fed_rows) >= MLA_MIRROR_FEED_BATCH_ROWS {
            self.queue(device_id, self.fed_rows, MLA_MIRROR_FEED_BATCH_ROWS, latent, scales, rope)?;
            self.fed_rows += MLA_MIRROR_FEED_BATCH_ROWS;
        }
        Ok(())
    }

    /// 补齐 [fed_rows, cached_rows) 的尾部；启用热窗/截断/导出/克隆前调用。
    fn flush_feed(&mut self, device_id: i32, cached_rows: usize, latent: &ops::hip::DeviceBuffer, scales: &ops::hip::DeviceBuffer, rope: &ops::hip::DeviceBuffer) -> Result<(), BackendError> {
        if cached_rows > self.fed_rows {
            self.queue(device_id, self.fed_rows, cached_rows - self.fed_rows, latent, scales, rope)?;
            self.fed_rows = cached_rows;
        }
        self.finish_pending()
    }

    fn finish_pending(&mut self) -> Result<(), BackendError> {
        let Some((start, rows, mut download)) = self.pending_download.take() else { return Ok(()) };
        if start != self.mirror.rows {
            return Err(compute_error(format!("ROCm MLA CPU mirror pending start={start}，期望 {}", self.mirror.rows)));
        }
        let latent_bytes = rows.checked_mul(self.mirror.latent_cols).ok_or_else(|| compute_error("ROCm MLA CPU mirror latent 大小溢出"))?;
        let scale_row_bytes = self.mirror.latent_cols / self.mirror.latent_group_size * 2;
        let scale_bytes = rows.checked_mul(scale_row_bytes).ok_or_else(|| compute_error("ROCm MLA CPU mirror scales 大小溢出"))?;
        let rope_bytes = rows.checked_mul(self.mirror.rope_cols * 2).ok_or_else(|| compute_error("ROCm MLA CPU mirror rope 大小溢出"))?;
        let bytes = download.wait().map_err(compute_error)?;
        if bytes.len() != latent_bytes + scale_bytes + rope_bytes {
            return Err(compute_error(format!("ROCm MLA CPU mirror bytes={}，期望 {}", bytes.len(), latent_bytes + scale_bytes + rope_bytes)));
        }
        self.mirror.latent.extend_from_slice(&bytes[..latent_bytes]);
        self.mirror.latent_scales.as_mut().expect("CPU mirror 只支持 Q8 MLA").extend_from_slice(&bytes[latent_bytes..latent_bytes + scale_bytes]);
        self.mirror.rope.extend_from_slice(&bytes[latent_bytes + scale_bytes..]);
        self.mirror.rows += rows;
        self.available_download = Some(download);
        Ok(())
    }

    fn queue(&mut self, device_id: i32, start: usize, rows: usize, latent: &ops::hip::DeviceBuffer, scales: &ops::hip::DeviceBuffer, rope: &ops::hip::DeviceBuffer) -> Result<(), BackendError> {
        self.finish_pending()?;
        if start != self.mirror.rows || rows == 0 {
            return Err(compute_error(format!("ROCm MLA CPU mirror queue start={start} rows={rows} mirror={}", self.mirror.rows)));
        }
        let latent_row_bytes = self.mirror.latent_cols;
        let scale_row_bytes = self.mirror.latent_cols / self.mirror.latent_group_size * 2;
        let rope_row_bytes = self.mirror.rope_cols * 2;
        let latent_bytes = rows.checked_mul(latent_row_bytes).ok_or_else(|| compute_error("ROCm MLA CPU mirror latent transfer 溢出"))?;
        let scale_bytes = rows.checked_mul(scale_row_bytes).ok_or_else(|| compute_error("ROCm MLA CPU mirror scale transfer 溢出"))?;
        let rope_bytes = rows.checked_mul(rope_row_bytes).ok_or_else(|| compute_error("ROCm MLA CPU mirror rope transfer 溢出"))?;
        let transfer_bytes = latent_bytes + scale_bytes + rope_bytes;
        if self.transfer.as_ref().is_none_or(|buffer| buffer.device_id() != device_id || buffer.bytes() < transfer_bytes) {
            self.transfer = Some(Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, transfer_bytes).map_err(compute_error)?));
        }
        let transfer = self.transfer.as_ref().expect("MLA mirror transfer 已创建");
        transfer.copy_from_device(0, latent, start * latent_row_bytes, latent_bytes).map_err(compute_error)?;
        transfer.copy_from_device(latent_bytes, scales, start * scale_row_bytes, scale_bytes).map_err(compute_error)?;
        transfer.copy_from_device(latent_bytes + scale_bytes, rope, start * rope_row_bytes, rope_bytes).map_err(compute_error)?;
        let mut download = match self.available_download.take() {
            Some(download) => download,
            None => ops::hip::AsyncHostDownload::new(device_id, transfer_bytes).map_err(compute_error)?,
        };
        download.enqueue(transfer, transfer_bytes).map_err(compute_error)?;
        self.pending_download = Some((start, rows, download));
        Ok(())
    }

    fn truncate(&mut self, rows: usize) -> Result<(), BackendError> {
        self.finish_pending()?;
        self.fed_rows = self.fed_rows.min(rows);
        if rows > self.mirror.rows {
            return Err(compute_error(format!("ROCm MLA CPU mirror truncate={rows} 超过 {}", self.mirror.rows)));
        }
        self.mirror.latent.truncate(rows * self.mirror.latent_cols);
        self.mirror.latent_scales.as_mut().expect("CPU mirror 只支持 Q8 MLA").truncate(rows * (self.mirror.latent_cols / self.mirror.latent_group_size) * 2);
        self.mirror.rope.truncate(rows * self.mirror.rope_cols * 2);
        self.mirror.rows = rows;
        Ok(())
    }

    fn finish(mut self) -> Result<MlaLayerSerde, BackendError> {
        self.finish_pending()?;
        Ok(self.mirror)
    }
}

/// hot 启用瞬间旧全量 GPU buffer 的暂存引用。首个 selection 直接 D2D gather
/// 到 hot cache，消费完成后随本结构释放，避免首轮走 CPU mirror 冷页 H2D。
pub(super) struct RocmMlaHotWarm {
    latent: Arc<ops::hip::DeviceBuffer>,
    scales: Option<Arc<ops::hip::DeviceBuffer>>,
    rope: Arc<ops::hip::DeviceBuffer>,
    rows: usize,
}

pub(super) struct RocmMlaCpuHotLayer {
    // 注册 guard 必须先于被借用的 mirror allocation 析构。
    gpu: Option<super::kv_cache_gpu_hot::GpuHotCache>,
    mirror: MlaLayerSerde,
    token_to_slot: Vec<u32>,
    slot_to_token: Vec<u32>,
    referenced: Vec<bool>,
    pinned_epoch: Vec<u32>,
    pin_epoch: u32,
    eviction_cursor: usize,
    transfer: Arc<ops::hip::DeviceBuffer>,
    available_download: Option<ops::hip::AsyncHostDownload>,
    pending_download: Option<(usize, usize, ops::hip::AsyncHostDownload)>,
    staging_latent: Option<Arc<ops::hip::DeviceBuffer>>,
    staging_scales: Option<Arc<ops::hip::DeviceBuffer>>,
    staging_rope: Option<Arc<ops::hip::DeviceBuffer>>,
    warm_source: Option<RocmMlaHotWarm>,
    /// 64 行环形 mirror 日志：append kernel 双写窗口槽位与日志，批量 D2H 从
    /// 日志取连续段，消除热窗期每行一次的 D2D+D2H+等待链。
    mirror_log: Option<(Arc<ops::hip::DeviceBuffer>, Arc<ops::hip::DeviceBuffer>, Arc<ops::hip::DeviceBuffer>)>,
    log_fed_rows: usize,
    /// append 前把预取队列接回计算流；attention 直接消费已经上传的 selection。
    prefetched: Option<(usize, Option<usize>, ops::hip::DeviceBuffer)>,
    lookups: u64,
    hits: u64,
    samples: usize,
    window_map_ns: u64,
    window_pack_ns: u64,
    window_upload_ns: u64,
    window_scatter_ns: u64,
    window_finish_ns: u64,
    window_misses: u64,
    window_h2d_bytes: u64,
    remapped_scratch: Vec<u32>,
    misses_scratch: Vec<(usize, usize)>,
    loaded_scratch: Vec<(usize, u32)>,
}

impl RocmMlaCpuHotLayer {
    fn new(
        device_id: i32,
        history_capacity: usize,
        hot_rows: usize,
        mirror: MlaLayerSerde,
        warm_source: Option<RocmMlaHotWarm>,
        transfer: Option<Arc<ops::hip::DeviceBuffer>>,
        available_download: Option<ops::hip::AsyncHostDownload>,
    ) -> Result<Self, BackendError> {
        let row_bytes = mirror.latent_cols.checked_add(mirror.latent_cols / mirror.latent_group_size * 2).and_then(|bytes| bytes.checked_add(mirror.rope_cols * 2)).ok_or_else(|| compute_error("ROCm MLA CPU hot row 大小溢出"))?;
        let mirror_log = if ops::hip::options().mla_cpu_hot_rows != 0 && !gpu_hot_selection(1) {
            // GPU 模式首次 append 直接借用 recent 环，避免先分配再释放独立日志。
            let scale_row_bytes = mirror.latent_cols / mirror.latent_group_size * 2;
            Some((
                Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, MLA_HOT_LOG_ROWS * mirror.latent_cols).map_err(compute_error)?),
                Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, MLA_HOT_LOG_ROWS * scale_row_bytes).map_err(compute_error)?),
                Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, MLA_HOT_LOG_ROWS * mirror.rope_cols * 2).map_err(compute_error)?),
            ))
        } else {
            None
        };
        let log_fed_rows = mirror.rows;
        Ok(Self {
            gpu: None,
            mirror,
            mirror_log,
            log_fed_rows,
            token_to_slot: vec![MLA_HOT_INVALID; history_capacity],
            slot_to_token: vec![MLA_HOT_INVALID; hot_rows],
            referenced: vec![false; hot_rows],
            pinned_epoch: vec![0; hot_rows],
            pin_epoch: 1,
            eviction_cursor: 0,
            transfer: match transfer {
                Some(transfer) => transfer,
                None => Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, row_bytes).map_err(compute_error)?),
            },
            available_download,
            pending_download: None,
            staging_latent: None,
            staging_scales: None,
            staging_rope: None,
            warm_source,
            prefetched: None,
            lookups: 0,
            hits: 0,
            samples: 0,
            window_map_ns: 0,
            window_pack_ns: 0,
            window_upload_ns: 0,
            window_scatter_ns: 0,
            window_finish_ns: 0,
            window_misses: 0,
            window_h2d_bytes: 0,
            remapped_scratch: Vec::new(),
            misses_scratch: Vec::new(),
            loaded_scratch: Vec::new(),
        })
    }

    fn join_prefetch(&mut self, device_id: i32) -> Result<(), BackendError> {
        if let Some((_, stream, _)) = &mut self.prefetched
            && let Some(stream) = stream.take()
        {
            ops::hip::set_device(device_id).map_err(compute_error)?;
            ops::hip::order_stream_after(device_id, stream, ops::hip::active_compute_stream() as usize).map_err(compute_error)?;
        }
        Ok(())
    }

    fn prepare_append_gpu_mode(&mut self, device_id: i32, rows: usize, cache: [&Arc<ops::hip::DeviceBuffer>; 3]) -> Result<(), BackendError> {
        let enabled = gpu_hot_selection(rows);
        let row_bytes = [self.mirror.latent_cols, self.mirror.latent_cols / self.mirror.latent_group_size * 2, self.mirror.rope_cols * 2];
        if enabled != self.gpu.is_some() {
            self.flush_log_feed(device_id)?;
            self.finish_pending()?;
            if !enabled {
                // CPU 槽位不固定，日志必须恢复独立存储，避免追加覆盖其他热槽。
                let [latent, scales, rope] = row_bytes.map(|bytes| ops::hip::DeviceBuffer::allocate_cache(device_id, MLA_HOT_LOG_ROWS * bytes).map(Arc::new).map_err(compute_error));
                self.mirror_log = Some((latent?, scales?, rope?));
            }
            self.gpu = if enabled { Some(super::kv_cache_gpu_hot::GpuHotCache::new(device_id, self.slot_to_token.len() - MLA_HOT_LOG_ROWS)?) } else { None };
            self.token_to_slot.fill(MLA_HOT_INVALID);
            self.slot_to_token.fill(MLA_HOT_INVALID);
            self.referenced.fill(false);
            self.reset_pins();
            self.warm_source = None;
        }
        if enabled {
            // GPU recent 槽本身就是连续日志环；peer 收到压缩 KV 后无需再复制一次。
            let first = self.slot_to_token.len() - MLA_HOT_LOG_ROWS;
            if self.mirror_log.as_ref().is_none_or(|log| log.0.device_pointer() != cache[0].device_pointer() + first * row_bytes[0]) {
                let [latent, scales, rope] = std::array::from_fn::<_, 3, _>(|i| ops::hip::DeviceBuffer::view(cache[i].clone(), first * row_bytes[i], MLA_HOT_LOG_ROWS * row_bytes[i]).map(Arc::new).map_err(compute_error));
                self.mirror_log = Some((latent?, scales?, rope?));
            }
            // 上个 64 行段已经与后续层计算重叠；在覆盖 recent 环前写入 RAM。
            self.finish_pending()?;
        }
        Ok(())
    }

    pub(super) fn uses_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn prepare_device_selection(
        &mut self,
        device_id: i32,
        selection: &ops::hip::DeviceBuffer,
        cache: [&ops::hip::DeviceBuffer; 3],
        query_rows: usize,
        width: usize,
        context_rows: usize,
    ) -> Result<([Arc<ops::hip::DeviceBuffer>; 3], Arc<ops::hip::DeviceBuffer>, Option<Arc<ops::hip::DeviceBuffer>>, usize), BackendError> {
        self.finish_pending()?;
        self.gpu.as_mut().ok_or_else(|| compute_error("GPU hot state 尚未初始化"))?.prepare(device_id, &mut self.mirror, self.token_to_slot.len(), cache, selection, query_rows, width, context_rows, MLA_HOT_LOG_ROWS)
    }

    pub(super) fn take_prefetched_selection(&mut self, device_id: i32, position: usize) -> Result<Option<ops::hip::DeviceBuffer>, BackendError> {
        if !self.prefetched.as_ref().is_some_and(|(start, _, _)| *start == position) {
            return Ok(None);
        }
        self.join_prefetch(device_id)?;
        Ok(self.prefetched.take().map(|(_, _, selection)| selection))
    }

    fn staging_buffer(slot: &mut Option<Arc<ops::hip::DeviceBuffer>>, device_id: i32, bytes: usize) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
        if slot.as_ref().is_none_or(|buffer| buffer.device_id() != device_id || buffer.bytes() < bytes) {
            *slot = Some(Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, bytes).map_err(compute_error)?));
        }
        Ok(slot.as_ref().expect("MLA hot staging 已创建").clone())
    }

    fn reset_pins(&mut self) {
        self.pin_epoch = self.pin_epoch.wrapping_add(1);
        if self.pin_epoch == 0 {
            self.pinned_epoch.fill(0);
            self.pin_epoch = 1;
        }
    }

    fn is_pinned(&self, slot: usize) -> bool {
        self.pinned_epoch[slot] == self.pin_epoch
    }

    fn pin(&mut self, slot: usize) {
        self.pinned_epoch[slot] = self.pin_epoch;
    }

    /// 把日志环里已双写但未 D2H 的行段补进 mirror 链（按环形回绕分段）。
    /// 消费点（selection miss 取新行/截断/导出/预取）在等待 mirror 前调用。
    fn flush_log_feed(&mut self, device_id: i32) -> Result<(), BackendError> {
        let Some((log_latent, log_scales, log_rope)) = self.mirror_log.clone() else { return Ok(()) };
        let chain_end = self.pending_download.as_ref().map_or(self.mirror.rows, |pending| pending.0 + pending.1);
        let mut start = chain_end;
        while start < self.log_fed_rows {
            let ring_position = start % MLA_HOT_LOG_ROWS;
            let span = (MLA_HOT_LOG_ROWS - ring_position).min(self.log_fed_rows - start);
            // 单飞约束:上一段 pending 必须先排水,环形回绕的分段逐段串行。
            self.finish_pending()?;
            self.queue_rows(device_id, start, span, ring_position, &log_latent, &log_scales, &log_rope)?;
            start += span;
        }
        Ok(())
    }

    fn finish_pending(&mut self) -> Result<(), BackendError> {
        let Some((start, rows, mut download)) = self.pending_download.take() else { return Ok(()) };
        if start != self.mirror.rows || rows == 0 {
            return Err(compute_error(format!("ROCm MLA CPU mirror pending start={start} rows={rows}，期望 {}", self.mirror.rows)));
        }
        let latent_bytes = rows.checked_mul(self.mirror.latent_cols).ok_or_else(|| compute_error("ROCm MLA CPU hot latent 大小溢出"))?;
        let scale_bytes = rows.checked_mul(self.mirror.latent_cols / self.mirror.latent_group_size * 2).ok_or_else(|| compute_error("ROCm MLA CPU hot scale 大小溢出"))?;
        let rope_bytes = rows.checked_mul(self.mirror.rope_cols * 2).ok_or_else(|| compute_error("ROCm MLA CPU hot rope 大小溢出"))?;
        let started = std::time::Instant::now();
        {
            let bytes = download.wait().map_err(compute_error)?;
            if bytes.len() != latent_bytes + scale_bytes + rope_bytes {
                return Err(compute_error(format!("ROCm MLA CPU mirror row bytes={}，期望 {}", bytes.len(), latent_bytes + scale_bytes + rope_bytes)));
            }
            self.mirror.latent.extend_from_slice(&bytes[..latent_bytes]);
            self.mirror.latent_scales.as_mut().expect("CPU hot 只支持 Q8 MLA").extend_from_slice(&bytes[latent_bytes..latent_bytes + scale_bytes]);
            self.mirror.rope.extend_from_slice(&bytes[latent_bytes + scale_bytes..]);
            self.mirror.rows += rows;
        }
        if ops::hip::options().mla_hot_trace {
            self.window_finish_ns += started.elapsed().as_nanos() as u64;
        }
        self.available_download = Some(download);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn queue_rows(&mut self, device_id: i32, start: usize, rows: usize, source_start: usize, latent: &ops::hip::DeviceBuffer, scales: &ops::hip::DeviceBuffer, rope: &ops::hip::DeviceBuffer) -> Result<(), BackendError> {
        if self.pending_download.is_some() || start != self.mirror.rows || rows == 0 {
            return Err(compute_error(format!("ROCm MLA CPU mirror queue start={start} rows={rows} mirror={} pending={}", self.mirror.rows, self.pending_download.is_some())));
        }
        let latent_row_bytes = self.mirror.latent_cols;
        let scale_row_bytes = self.mirror.latent_cols / self.mirror.latent_group_size * 2;
        let rope_row_bytes = self.mirror.rope_cols * 2;
        let latent_bytes = rows.checked_mul(latent_row_bytes).ok_or_else(|| compute_error("ROCm MLA CPU mirror latent transfer 溢出"))?;
        let scale_bytes = rows.checked_mul(scale_row_bytes).ok_or_else(|| compute_error("ROCm MLA CPU mirror scale transfer 溢出"))?;
        let rope_bytes = rows.checked_mul(rope_row_bytes).ok_or_else(|| compute_error("ROCm MLA CPU mirror rope transfer 溢出"))?;
        let transfer_bytes = latent_bytes + scale_bytes + rope_bytes;
        if self.transfer.bytes() < transfer_bytes {
            self.transfer = Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, transfer_bytes).map_err(compute_error)?);
        }
        ops::hip::DeviceBuffer::copy3_from_device(
            [(latent, source_start * latent_row_bytes), (scales, source_start * scale_row_bytes), (rope, source_start * rope_row_bytes)],
            [(&self.transfer, 0), (&self.transfer, latent_bytes), (&self.transfer, latent_bytes + scale_bytes)],
            [latent_bytes, scale_bytes, rope_bytes],
        )
        .map_err(compute_error)?;
        let mut download = match self.available_download.take() {
            Some(download) => download,
            None => ops::hip::AsyncHostDownload::new(device_id, transfer_bytes).map_err(compute_error)?,
        };
        // 热窗日志至多一个 recent 段，直接由 compute kernel 写入 pinned
        // staging，避免每批回传切换 DMA 队列；大段追加仍由入口回退到 DMA。
        download.enqueue_small(&self.transfer, transfer_bytes).map_err(compute_error)?;
        self.pending_download = Some((start, rows, download));
        Ok(())
    }

    fn map_direct_range(&mut self, start: usize, rows: usize) -> Result<(), BackendError> {
        if start.checked_add(rows).is_none_or(|end| end > self.slot_to_token.len() || end > self.token_to_slot.len()) {
            return Err(compute_error(format!("ROCm MLA hot direct range start={start} rows={rows} capacity={} 非法", self.slot_to_token.len())));
        }
        for token in start..start + rows {
            if self.token_to_slot[token] != MLA_HOT_INVALID || self.slot_to_token[token] != MLA_HOT_INVALID {
                return Err(compute_error(format!("ROCm MLA hot direct token={token} 已有映射")));
            }
            self.token_to_slot[token] = token as u32;
            self.slot_to_token[token] = token as u32;
            self.referenced[token] = true;
        }
        self.eviction_cursor = (start + rows) % self.slot_to_token.len();
        Ok(())
    }

    fn assign_slot(&mut self, token: usize) -> Result<usize, BackendError> {
        if token >= self.token_to_slot.len() {
            return Err(compute_error(format!("ROCm MLA hot token={token} 超过容量 {}", self.token_to_slot.len())));
        }
        if self.gpu.is_some() {
            // 末尾 64 槽只保存尚未进入 RAM mirror 的新行，GPU 置换不碰它们。
            let slot = self.slot_to_token.len() - MLA_HOT_LOG_ROWS + token % MLA_HOT_LOG_ROWS;
            let old = self.slot_to_token[slot];
            if old != MLA_HOT_INVALID {
                self.token_to_slot[old as usize] = MLA_HOT_INVALID;
            }
            self.slot_to_token[slot] = token as u32;
            self.token_to_slot[token] = slot as u32;
            return Ok(slot);
        }
        let existing = self.token_to_slot[token];
        if existing != MLA_HOT_INVALID {
            let slot = existing as usize;
            if self.slot_to_token.get(slot).copied() != Some(token as u32) {
                return Err(compute_error(format!("ROCm MLA hot 反向映射 token={token} slot={slot} 不一致")));
            }
            self.referenced[slot] = true;
            return Ok(slot);
        }
        let capacity = self.slot_to_token.len();
        for _ in 0..capacity.saturating_mul(3) {
            let slot = self.eviction_cursor;
            self.eviction_cursor = (self.eviction_cursor + 1) % capacity;
            if self.is_pinned(slot) {
                continue;
            }
            let old = self.slot_to_token[slot];
            if old == MLA_HOT_INVALID {
                self.slot_to_token[slot] = token as u32;
                self.token_to_slot[token] = slot as u32;
                self.referenced[slot] = true;
                return Ok(slot);
            }
            if self.referenced[slot] {
                self.referenced[slot] = false;
                continue;
            }
            self.token_to_slot[old as usize] = MLA_HOT_INVALID;
            self.slot_to_token[slot] = token as u32;
            self.token_to_slot[token] = slot as u32;
            self.referenced[slot] = true;
            return Ok(slot);
        }
        Err(compute_error(format!("ROCm MLA hot rows={capacity} 无可驱逐 slot")))
    }

    /// selection 的 pin 只保护当前 attention tile；进入下一次 append 时 GPU
    /// 已消费完该 tile，可以清 pin 后为新 token 分配窗口槽位。
    fn append_slot(&mut self, token: usize) -> Result<usize, BackendError> {
        self.reset_pins();
        self.assign_slot(token)
    }

    fn reserve_slot(&mut self, token: usize) -> Result<(), BackendError> {
        let slot = self.assign_slot(token)?;
        self.pin(slot);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_selection(
        &mut self,
        device_id: i32,
        layer: usize,
        selection: &[u32],
        latent: &ops::hip::DeviceBuffer,
        scales: &ops::hip::DeviceBuffer,
        rope: &ops::hip::DeviceBuffer,
    ) -> Result<ops::hip::DeviceBuffer, BackendError> {
        if selection.is_empty() {
            return Err(compute_error(format!("L{layer} ROCm MLA hot selection 为空")));
        }
        let trace = ops::hip::options().mla_hot_trace;
        let phase_started = std::time::Instant::now();
        self.reset_pins();
        let mut remapped = std::mem::take(&mut self.remapped_scratch);
        remapped.clear();
        remapped.resize(selection.len(), MLA_HOT_INVALID);
        let mut misses = std::mem::take(&mut self.misses_scratch);
        misses.clear();
        let mut hits = 0usize;
        for (index, &token) in selection.iter().enumerate() {
            let token = token as usize;
            if token >= self.token_to_slot.len() {
                return Err(compute_error(format!("L{layer} ROCm MLA hot selected token={token} 越界")));
            }
            let slot = self.token_to_slot[token];
            if slot == MLA_HOT_INVALID {
                misses.push((index, token));
                continue;
            }
            let slot = slot as usize;
            if self.slot_to_token[slot] != token as u32 {
                return Err(compute_error(format!("L{layer} ROCm MLA hot selection token={token} slot={slot} 映射损坏")));
            }
            self.pin(slot);
            self.referenced[slot] = true;
            remapped[index] = slot as u32;
            hits += 1;
        }
        // 上一 token 已经留在 hot cache 时，预取只读 GPU 槽位，不必在这里
        // 等待它的 CPU mirror 下载。只有它已被驱逐且本轮确实选中时才提前收尾；
        // 常规路径把等待延后到本层 append，让 D2H 与前面的层计算重叠。
        if misses.iter().any(|&(_, token)| token >= self.mirror.rows) {
            self.flush_log_feed(device_id)?;
            self.finish_pending()?;
        }
        let mut loaded = std::mem::take(&mut self.loaded_scratch);
        loaded.clear();
        loaded.reserve(misses.len());
        for (index, token) in misses.drain(..) {
            if token >= self.mirror.rows {
                return Err(compute_error(format!("L{layer} ROCm MLA CPU mirror rows={} 缺少 selected token={token}", self.mirror.rows)));
            }
            // 同一 tile 的各 query row 会大量命中同一批历史 token。只为首次
            // 建立 slot 的 token 打包/H2D，后续重复项只复用映射。
            let needs_load = self.token_to_slot[token] == MLA_HOT_INVALID;
            let slot = self.assign_slot(token)?;
            self.pin(slot);
            remapped[index] = slot as u32;
            if needs_load {
                loaded.push((token, slot as u32));
            }
        }
        if trace {
            self.window_map_ns += phase_started.elapsed().as_nanos() as u64;
        }
        let phase_started = std::time::Instant::now();
        let latent_row_bytes = self.mirror.latent_cols;
        let scale_row_bytes = self.mirror.latent_cols / self.mirror.latent_group_size * 2;
        let rope_row_bytes = self.mirror.rope_cols * 2;
        let selection_bytes = remapped.len() * std::mem::size_of::<u32>();
        let upload = if let Some(warm) = self.warm_source.take() {
            // hot 启用后的首个 selection：全量旧 buffer 还在 GPU，直接 D2D gather，
            // 绕开 ~100MiB 量级 CPU mirror 冷页打包 + H2D。
            if loaded.iter().any(|&(token, _)| token >= warm.rows) {
                return Err(compute_error(format!("L{layer} ROCm MLA warm rows={} 缺少 selected token", warm.rows)));
            }
            let tokens_bytes = loaded.len() * std::mem::size_of::<u32>();
            let slots_bytes = loaded.len() * std::mem::size_of::<u32>();
            let tokens_offset = 0;
            let slots_offset = tokens_offset + tokens_bytes;
            let selection_offset = slots_offset + slots_bytes;
            let packed_bytes = selection_offset + selection_bytes;
            let tokens: Vec<u32> = loaded.iter().map(|&(token, _)| u32::try_from(token).expect("MLA history 已受 u32 kernel 约束")).collect();
            let slots: Vec<u32> = loaded.iter().map(|&(_, slot)| slot).collect();
            let upload = Arc::new(
                ops::hip::DeviceBuffer::upload_ordered_with(device_id, packed_bytes, |packed| {
                    for (target, token) in packed[tokens_offset..slots_offset].chunks_exact_mut(4).zip(&tokens) {
                        target.copy_from_slice(&token.to_ne_bytes());
                    }
                    for (target, slot) in packed[slots_offset..selection_offset].chunks_exact_mut(4).zip(&slots) {
                        target.copy_from_slice(&slot.to_ne_bytes());
                    }
                    for (target, slot) in packed[selection_offset..].chunks_exact_mut(4).zip(&remapped) {
                        target.copy_from_slice(&slot.to_ne_bytes());
                    }
                })
                .map_err(compute_error)?,
            );
            upload.enqueue_small_deferred_upload().map_err(compute_error)?;
            upload.retain_for_active_stage();
            if !loaded.is_empty() {
                let warm_tokens = ops::hip::DeviceBuffer::view(upload.clone(), tokens_offset, tokens_bytes).map_err(compute_error)?;
                let slots = ops::hip::DeviceBuffer::view(upload.clone(), slots_offset, slots_bytes).map_err(compute_error)?;
                ops::hip::try_mla_hot_gather_q8(
                    device_id,
                    &warm.latent,
                    warm.scales.as_deref(),
                    &warm.rope,
                    &warm_tokens,
                    &slots,
                    latent,
                    scales,
                    rope,
                    loaded.len(),
                    self.mirror.latent_cols,
                    self.mirror.latent_cols / self.mirror.latent_group_size,
                    self.mirror.rope_cols,
                    self.slot_to_token.len(),
                )
                .map_err(compute_error)?;
            }
            upload
        } else {
            let latent_bytes = loaded.len() * latent_row_bytes;
            let scale_bytes = loaded.len() * scale_row_bytes;
            let rope_bytes = loaded.len() * rope_row_bytes;
            let slots_bytes = loaded.len() * std::mem::size_of::<u32>();
            let scale_offset = latent_bytes;
            let rope_offset = scale_offset + scale_bytes;
            let slots_offset = rope_offset + rope_bytes;
            let selection_offset = slots_offset + slots_bytes;
            let packed_bytes = selection_offset + selection_bytes;
            let mirror_scales = self.mirror.latent_scales.as_ref().expect("CPU hot 只支持 Q8 MLA");
            let upload = Arc::new(
                ops::hip::DeviceBuffer::upload_ordered_with(device_id, packed_bytes, |packed| {
                    for (target, &(token, _)) in packed[..latent_bytes].chunks_exact_mut(latent_row_bytes).zip(&loaded) {
                        target.copy_from_slice(&self.mirror.latent[token * latent_row_bytes..(token + 1) * latent_row_bytes]);
                    }
                    for (target, &(token, _)) in packed[scale_offset..rope_offset].chunks_exact_mut(scale_row_bytes).zip(&loaded) {
                        target.copy_from_slice(&mirror_scales[token * scale_row_bytes..(token + 1) * scale_row_bytes]);
                    }
                    for (target, &(token, _)) in packed[rope_offset..slots_offset].chunks_exact_mut(rope_row_bytes).zip(&loaded) {
                        target.copy_from_slice(&self.mirror.rope[token * rope_row_bytes..(token + 1) * rope_row_bytes]);
                    }
                    for (target, &(_, slot)) in packed[slots_offset..selection_offset].chunks_exact_mut(4).zip(&loaded) {
                        target.copy_from_slice(&slot.to_ne_bytes());
                    }
                    for (target, slot) in packed[selection_offset..].chunks_exact_mut(4).zip(&remapped) {
                        target.copy_from_slice(&slot.to_ne_bytes());
                    }
                })
                .map_err(compute_error)?,
            );
            if trace {
                self.window_pack_ns += phase_started.elapsed().as_nanos() as u64;
            }
            let phase_started = std::time::Instant::now();
            upload.enqueue_small_deferred_upload().map_err(compute_error)?;
            upload.retain_for_active_stage();
            if !loaded.is_empty() {
                let source_latent = ops::hip::DeviceBuffer::view(upload.clone(), 0, latent_bytes).map_err(compute_error)?;
                let source_scales = ops::hip::DeviceBuffer::view(upload.clone(), scale_offset, scale_bytes).map_err(compute_error)?;
                let source_rope = ops::hip::DeviceBuffer::view(upload.clone(), rope_offset, rope_bytes).map_err(compute_error)?;
                let slots = ops::hip::DeviceBuffer::view(upload.clone(), slots_offset, slots_bytes).map_err(compute_error)?;
                ops::hip::try_mla_hot_scatter_q8(
                    device_id,
                    &source_latent,
                    &source_scales,
                    &source_rope,
                    &slots,
                    latent,
                    scales,
                    rope,
                    loaded.len(),
                    self.mirror.latent_cols,
                    self.mirror.latent_cols / self.mirror.latent_group_size,
                    self.mirror.rope_cols,
                    self.slot_to_token.len(),
                )
                .map_err(compute_error)?;
            }
            if trace {
                self.window_scatter_ns += phase_started.elapsed().as_nanos() as u64;
            }
            upload
        };
        let packed_bytes = upload.bytes();
        self.lookups += selection.len() as u64;
        self.hits += hits as u64;
        self.samples += 1;
        if trace {
            self.window_misses += loaded.len() as u64;
            self.window_h2d_bytes += packed_bytes as u64;
            if self.samples.is_multiple_of(128) {
                eprintln!(
                    "[mla-hot-trace] layer={layer} sample={} history={} hot_rows={} window: map_ms={:.3} pack_ms={:.3} upload_submit_ms={:.3} scatter_ms={:.3} finish_wait_ms={:.3} hits={:.1}% miss_rows={:.0} h2d_mib={:.2} (per-call avg)",
                    self.samples,
                    self.mirror.rows,
                    self.slot_to_token.len(),
                    self.window_map_ns as f64 / 128e6,
                    self.window_pack_ns as f64 / 128e6,
                    self.window_upload_ns as f64 / 128e6,
                    self.window_scatter_ns as f64 / 128e6,
                    self.window_finish_ns as f64 / 128e6,
                    self.hits as f64 * 100.0 / self.lookups as f64,
                    self.window_misses as f64 / 128.0,
                    self.window_h2d_bytes as f64 / 128.0 / (1_u64 << 20) as f64,
                );
                self.window_map_ns = 0;
                self.window_pack_ns = 0;
                self.window_upload_ns = 0;
                self.window_scatter_ns = 0;
                self.window_finish_ns = 0;
                self.window_misses = 0;
                self.window_h2d_bytes = 0;
            }
        }
        if ops::hip::options().kernel_profile && (self.samples <= 2 || self.samples.is_multiple_of(128)) {
            eprintln!(
                "[mla-cpu-hot] device={device_id} layer={layer} sample={} history={} hot_rows={} hits={hits}/{} cumulative={:.3}% loads={} h2d_bytes={}",
                self.samples - 1,
                self.mirror.rows,
                self.slot_to_token.len(),
                selection.len(),
                self.hits as f64 * 100.0 / self.lookups as f64,
                loaded.len(),
                packed_bytes,
            );
        }
        self.remapped_scratch = remapped;
        self.misses_scratch = misses;
        self.loaded_scratch = loaded;
        let selection_offset = upload.bytes() - selection_bytes;
        ops::hip::DeviceBuffer::view(upload, selection_offset, selection_bytes).map_err(compute_error)
    }

    fn truncate(&mut self, rows: usize) -> Result<(), BackendError> {
        let device_id = self.transfer.device_id();
        ops::hip::set_device(device_id).map_err(compute_error)?;
        self.join_prefetch(device_id)?;
        self.prefetched = None;
        if let Some(gpu) = &self.gpu {
            gpu.truncate(device_id, rows, self.log_fed_rows.max(self.mirror.rows))?;
            // GPU recent 槽仍持有已接受的尾部。MTP 拒绝少量 draft 时只回退
            // 日志长度，不把不足一批的尾部强制 D2H，否则每轮都失去批量喂养。
            self.finish_pending()?;
        } else {
            self.log_fed_rows = self.log_fed_rows.min(rows + MLA_HOT_LOG_ROWS);
            self.flush_log_feed(device_id)?;
            self.finish_pending()?;
            if rows > self.mirror.rows {
                return Err(compute_error(format!("ROCm MLA CPU mirror truncate={rows} 超过 {}", self.mirror.rows)));
            }
        }
        let mirror_rows = rows.min(self.mirror.rows);
        let latent_row_bytes = self.mirror.latent_cols;
        let scale_row_bytes = self.mirror.latent_cols / self.mirror.latent_group_size * 2;
        let rope_row_bytes = self.mirror.rope_cols * 2;
        self.mirror.latent.truncate(mirror_rows * latent_row_bytes);
        self.mirror.latent_scales.as_mut().expect("CPU hot 只支持 Q8 MLA").truncate(mirror_rows * scale_row_bytes);
        self.mirror.rope.truncate(mirror_rows * rope_row_bytes);
        self.mirror.rows = mirror_rows;
        self.log_fed_rows = rows;
        let slot_start = if self.gpu.is_some() { self.slot_to_token.len() - MLA_HOT_LOG_ROWS } else { 0 };
        for slot in slot_start..self.slot_to_token.len() {
            let token = self.slot_to_token[slot];
            if token != MLA_HOT_INVALID && token as usize >= rows {
                self.token_to_slot[token as usize] = MLA_HOT_INVALID;
                self.slot_to_token[slot] = MLA_HOT_INVALID;
                self.referenced[slot] = false;
                self.pinned_epoch[slot] = 0;
            }
        }
        Ok(())
    }
}

impl RocmKvCache {
    /// 把所有已填充层 D2H 到 host blob。空层（slot None）对应 `None`。
    pub fn download_layers(&self) -> Result<Vec<Option<MlaLayerSerde>>, BackendError> {
        fn download_device(cached: &RocmPagedMlaLayer, rows: usize, ownership: RocmKvOwnership) -> Result<MlaLayerSerde, BackendError> {
            let element_bytes = if cached.latent_group_size == 0 { 2 } else { 1 };
            let latent_bytes = rows.checked_mul(cached.latent_cols).and_then(|n| n.checked_mul(element_bytes)).ok_or_else(|| compute_error("ROCm MLA latent 字节溢出"))?;
            let mut latent = vec![0u8; latent_bytes];
            cached.latent.copy_to_host(&mut latent).map_err(compute_error)?;
            let latent_scales = if let Some(scales) = &cached.latent_scales {
                let scale_bytes = rows.checked_mul(cached.latent_cols / cached.latent_group_size).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA scales 字节溢出"))?;
                let mut buffer = vec![0u8; scale_bytes];
                scales.copy_to_host(&mut buffer).map_err(compute_error)?;
                Some(buffer)
            } else {
                None
            };
            let rope_bytes = rows.checked_mul(cached.rope_cols).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA rope 字节溢出"))?;
            let mut rope = vec![0u8; rope_bytes];
            cached.rope.copy_to_host(&mut rope).map_err(compute_error)?;
            Ok(MlaLayerSerde { rows: cached.rows, ownership, latent_cols: cached.latent_cols, rope_cols: cached.rope_cols, latent_group_size: cached.latent_group_size, latent, latent_scales, rope })
        }

        self.paged_layers
            .iter()
            .enumerate()
            .map(|(layer, slot)| -> Result<Option<MlaLayerSerde>, BackendError> {
                let Some(cached) = slot.as_ref() else {
                    return Ok(self.pending_pair_layers.get(layer).and_then(Option::as_ref).cloned());
                };
                if let Some(hot) = &cached.cpu_hot {
                    let mut hot = hot.lock().map_err(|_| compute_error("ROCm MLA hot 锁中毒"))?;
                    hot.flush_log_feed(cached.latent.device_id())?;
                    hot.finish_pending()?;
                    if hot.mirror.rows != cached.rows {
                        return Err(compute_error(format!("ROCm MLA CPU mirror rows={}，cache rows={}", hot.mirror.rows, cached.rows)));
                    }
                    return Ok(Some(hot.mirror.clone()));
                }
                if let Some(mirror) = &cached.cpu_mirror {
                    let mut mirror = mirror.lock().map_err(|_| compute_error("ROCm MLA mirror 锁中毒"))?;
                    mirror.flush_feed(cached.latent.device_id(), cached.rows, &cached.latent, cached.latent_scales.as_deref().expect("Q8 MLA 必有 scales"), &cached.rope)?;
                    if mirror.mirror.rows != cached.rows {
                        return Err(compute_error(format!("ROCm MLA CPU mirror rows={}，cache rows={}", mirror.mirror.rows, cached.rows)));
                    }
                    return Ok(Some(mirror.mirror.clone()));
                }
                if self.ownership == RocmKvOwnership::BlockParity(0) {
                    let peer = self.cooperative_peer.as_ref().and_then(|peer| peer.cache.paged_layers.get(layer)).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} pair KV peer shard 缺失")))?;
                    if peer.rows != cached.rows || peer.latent_cols != cached.latent_cols || peer.rope_cols != cached.rope_cols || peer.latent_group_size != cached.latent_group_size {
                        return Err(compute_error(format!("L{layer} pair KV owner/peer shape 不一致")));
                    }
                    let mut owner = download_device(cached, parity_rows(cached.rows, 0), RocmKvOwnership::InterleavedPair)?;
                    let peer = download_device(peer, parity_rows(cached.rows, 1), RocmKvOwnership::InterleavedPair)?;
                    owner.latent.extend_from_slice(&peer.latent);
                    match (&mut owner.latent_scales, peer.latent_scales) {
                        (Some(owner), Some(peer)) => owner.extend_from_slice(&peer),
                        (None, None) => {}
                        _ => return Err(compute_error(format!("L{layer} pair KV scale shape 不一致"))),
                    }
                    owner.rope.extend_from_slice(&peer.rope);
                    return Ok(Some(owner));
                }
                if self.ownership != RocmKvOwnership::Full {
                    return Err(compute_error(format!("L{layer} 不能单独导出 {:?} KV shard", self.ownership)));
                }
                Ok(Some(download_device(cached, cached.rows, RocmKvOwnership::Full)?))
            })
            .collect()
    }

    /// 从 host blob H2D 重建每层（allocate + upload）。空 slot 跳过。
    /// 调用方需先用 `with_capacity` 构造空 cache，再 `upload_layers`。
    pub fn upload_layers(&mut self, context: &RocmContext, layers: &[Option<MlaLayerSerde>], reserved_rows: usize) -> Result<(), BackendError> {
        self.cooperative_peer = None;
        self.operator_peer = None;
        self.operator_selection = None;
        self.ownership = RocmKvOwnership::Full;
        self.pending_pair_layers.fill(None);
        self.pending_pair_reserved_rows = reserved_rows;
        let device_id = context.device_id;
        if layers.len() > self.paged_layers.len() || reserved_rows > self.capacity {
            return Err(compute_error(format!("ROCm MLA upload layers={}/{} reserved_rows={}/{} 非法", layers.len(), self.paged_layers.len(), reserved_rows, self.capacity)));
        }
        for (index, slot) in layers.iter().enumerate() {
            let Some(record) = slot else { continue };
            if record.rows == 0 || record.rows > self.capacity || record.latent_cols == 0 || record.rope_cols == 0 {
                return Err(compute_error(format!("ROCm MLA restore L{index} shape 非法: rows={} capacity={} latent_cols={} rope_cols={}", record.rows, self.capacity, record.latent_cols, record.rope_cols)));
            }
            let latent_element_bytes = if record.latent_group_size == 0 { 2 } else { 1 };
            if record.latent_group_size != 0 && !record.latent_cols.is_multiple_of(record.latent_group_size) {
                return Err(compute_error(format!("ROCm MLA restore L{index} latent_cols={} 不能整除 group={}", record.latent_cols, record.latent_group_size)));
            }
            let latent_used = record.rows.checked_mul(record.latent_cols).and_then(|n| n.checked_mul(latent_element_bytes)).ok_or_else(|| compute_error("ROCm MLA restore latent 大小溢出"))?;
            if record.latent.len() != latent_used {
                return Err(compute_error(format!("ROCm MLA restore L{index} latent={}，期望 {latent_used}", record.latent.len())));
            }
            let scale_used = if record.latent_group_size == 0 { 0 } else { record.rows * (record.latent_cols / record.latent_group_size) * 2 };
            if record.latent_scales.as_ref().map(Vec::len).unwrap_or(0) != scale_used || (record.latent_group_size == 0) != record.latent_scales.is_none() {
                return Err(compute_error(format!("ROCm MLA restore L{index} scales={}，期望 {scale_used}", record.latent_scales.as_ref().map_or(0, Vec::len))));
            }
            let rope_used = record.rows.checked_mul(record.rope_cols).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA restore rope 大小溢出"))?;
            if record.rope.len() != rope_used {
                return Err(compute_error(format!("ROCm MLA restore L{index} rope={}，期望 {rope_used}", record.rope.len())));
            }
            match record.ownership {
                RocmKvOwnership::InterleavedPair => {
                    self.pending_pair_layers[index] = Some(record.clone());
                    continue;
                }
                RocmKvOwnership::Full => {}
                RocmKvOwnership::BlockParity(_) => return Err(compute_error(format!("ROCm MLA restore L{index} 不接受单独 parity shard"))),
            }
            let committed_rows = reserved_rows.max(record.rows);
            let latent_capacity = committed_rows.checked_mul(record.latent_cols).and_then(|n| n.checked_mul(latent_element_bytes)).ok_or_else(|| compute_error("ROCm MLA restore latent 容量溢出"))?;
            let latent = upload_cache_buffer(device_id, &record.latent, latent_capacity)?;
            let latent_scales = match (&record.latent_scales, record.latent_group_size) {
                (None, 0) => None,
                (Some(bytes), group) if group != 0 => {
                    let groups = record.latent_cols / group;
                    let used = record.rows.checked_mul(groups).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA restore scales 大小溢出"))?;
                    if bytes.len() != used {
                        return Err(compute_error(format!("ROCm MLA restore L{index} scales={}，期望 {used}", bytes.len())));
                    }
                    let capacity = committed_rows.checked_mul(groups).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA restore scales 容量溢出"))?;
                    Some(upload_cache_buffer(device_id, bytes, capacity)?)
                }
                _ => return Err(compute_error(format!("ROCm MLA restore L{index} scales 与 group={} 不一致", record.latent_group_size))),
            };
            let rope_capacity = committed_rows.checked_mul(record.rope_cols).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA restore rope 容量溢出"))?;
            let rope = upload_cache_buffer(device_id, &record.rope, rope_capacity)?;
            let cpu_mirror = ((ops::hip::options().mla_cpu_hot_rows != 0 || ops::hip::options().prefill_attention_cpu) && record.latent_group_size != 0).then(|| std::sync::Mutex::new(RocmMlaCpuMirror::from_record(record.clone())));
            self.paged_layers[index] = Some(RocmPagedMlaLayer {
                latent,
                latent_scales,
                latent_group_size: record.latent_group_size,
                rope,
                rows: record.rows,
                latent_cols: record.latent_cols,
                rope_cols: record.rope_cols,
                committed_rows,
                cpu_mirror,
                cpu_hot: None,
            });
        }
        Ok(())
    }

    /// 所有层 device buffer 实际分配字节（供 `CacheInfo.bytes` 上报）。
    pub fn allocated_bytes(&self) -> u64 {
        let local = self
            .paged_layers
            .iter()
            .filter_map(|slot| slot.as_ref())
            .map(|cached| cached.latent.allocation_bytes() as u64 + cached.latent_scales.as_ref().map_or(0, |scales| scales.allocation_bytes() as u64) + cached.rope.allocation_bytes() as u64)
            .sum::<u64>()
            .saturating_add(self.gqa_layers.iter().filter_map(|slot| slot.as_ref()).map(|cached| (cached.key.allocation_bytes() + cached.value.allocation_bytes()) as u64).sum());
        let cooperative = self.cooperative_peer.as_ref().map_or(0, |peer| peer.cache.allocated_bytes());
        let operator = self.operator_peer.as_ref().and_then(|peer| peer.cache.lock().ok()).map_or(0, |cache| cache.allocated_bytes());
        local.saturating_add(cooperative).saturating_add(operator)
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    #[test]
    fn glm_mla_q8_replica_is_656_bytes_per_token_layer() {
        let (latent, scales, rope) = mla_q8_row_bytes(512, 64, 64).unwrap();
        assert_eq!((latent, scales, rope), (512, 16, 128));
        assert_eq!(latent + scales + rope, 656);
        assert_eq!(mla_q8_row_bytes(510, 64, 64), None);
        assert_eq!(mla_q8_row_bytes(512, 64, 0), None);
    }

    #[test]
    fn block_parity_mapping_never_moves_existing_rows() {
        for rows in [1, 63, 64, 65, 127, 128, 129, 191, 192, 257] {
            assert_eq!(parity_rows(rows, 0) + parity_rows(rows, 1), rows);
            for logical in 0..rows {
                let parity = (logical / ROCM_KV_BLOCK_SIZE) & 1;
                assert!(parity_row(logical) < parity_rows(rows, parity));
                for grown in [rows, rows + 1, rows + ROCM_KV_BLOCK_SIZE, rows + 3 * ROCM_KV_BLOCK_SIZE] {
                    assert_eq!(parity_row(logical), (logical / (2 * ROCM_KV_BLOCK_SIZE)) * ROCM_KV_BLOCK_SIZE + logical % ROCM_KV_BLOCK_SIZE);
                    assert!(parity_row(logical) < parity_rows(grown, parity));
                }
            }
        }
    }

    #[test]
    fn pair_record_truncate_keeps_owner_then_peer_compact_layout() {
        let rows = 150;
        let owner_rows = parity_rows(rows, 0);
        let peer_rows = parity_rows(rows, 1);
        let tagged_rows = |owner_tag: u8, peer_tag: u8, row_bytes: usize| {
            let mut bytes = Vec::with_capacity(rows * row_bytes);
            for row in 0..owner_rows {
                bytes.extend(std::iter::repeat_n(owner_tag.wrapping_add(row as u8), row_bytes));
            }
            for row in 0..peer_rows {
                bytes.extend(std::iter::repeat_n(peer_tag.wrapping_add(row as u8), row_bytes));
            }
            bytes
        };
        let mut record = MlaLayerSerde {
            rows,
            ownership: RocmKvOwnership::InterleavedPair,
            latent_cols: 2,
            rope_cols: 1,
            latent_group_size: 2,
            latent: tagged_rows(0, 128, 2),
            latent_scales: Some(tagged_rows(16, 144, 2)),
            rope: tagged_rows(32, 160, 2),
        };
        truncate_pair_record(&mut record, 70).unwrap();
        assert_eq!(record.rows, 70);
        assert_eq!(record.latent.len(), 140);
        assert_eq!(record.latent_scales.as_ref().unwrap().len(), 140);
        assert_eq!(record.rope.len(), 140);
        // 70 行 = owner block0 的 64 行 + peer block1 的前 6 行。
        assert_eq!(&record.latent[126..130], &[63, 63, 128, 128]);
        assert_eq!(&record.latent[138..140], &[133, 133]);
    }

    #[test]
    fn pair_record_join_restores_logical_row_order() {
        let rows = 150;
        let row_bytes = 3;
        let mut compact = Vec::with_capacity(rows * row_bytes);
        for parity in 0..2 {
            for logical in (0..rows).filter(|logical| (logical / ROCM_KV_BLOCK_SIZE) & 1 == parity) {
                compact.extend(std::iter::repeat_n(logical as u8, row_bytes));
            }
        }
        let joined = join_pair_bytes(&compact, rows, row_bytes, "test").unwrap();
        assert_eq!(joined.len(), rows * row_bytes);
        for (logical, bytes) in joined.chunks_exact(row_bytes).enumerate() {
            assert!(bytes.iter().all(|&byte| byte == logical as u8));
        }
    }
}
