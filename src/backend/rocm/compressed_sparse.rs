//! 压缩稀疏注意力的 ROCm resident capability。

use std::{
    collections::HashMap,
    mem,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use crate::attention::compressed_sparse::{
    CompressedBatch, CompressedGatedSegment, CompressedSelection, CompressedSparseAttentionSpec, CompressedSparseKernel, CompressedSparsePrefillSegment, CompressionState, CompressionStream, KvCompressionSpec,
    compress_gated_segmented_fallback,
};
use crate::backend::{SegmentedTensorBackend, compute_error as compute};

use super::*;

mod snapshot;
mod transaction;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RocmRopeTableKey {
    device_id: i32,
    cos: usize,
    sin: usize,
    elements: usize,
}

struct RocmRopeTables {
    key: RocmRopeTableKey,
    cos: Arc<ops::hip::DeviceBuffer>,
    sin: Arc<ops::hip::DeviceBuffer>,
    elements: usize,
}

static ROCM_ROPE_TABLES: OnceLock<Mutex<HashMap<RocmRopeTableKey, Weak<RocmRopeTables>>>> = OnceLock::new();

#[derive(Default)]
struct RocmGatedPoolState {
    state: CompressionState,
    ratio: usize,
    width: usize,
    channels: usize,
    overlap: bool,
    pending_key: Option<Arc<ops::hip::DeviceBuffer>>,
    pending_gate: Option<Arc<ops::hip::DeviceBuffer>>,
    overlap_key: Option<Arc<ops::hip::DeviceBuffer>>,
    overlap_gate: Option<Arc<ops::hip::DeviceBuffer>>,
    rope: Option<Arc<RocmRopeTables>>,
}

struct RocmGatedPoolSnapshot {
    next_position: usize,
    entry_count: usize,
    pending_rows: usize,
    pending_key: Option<Arc<ops::hip::DeviceBuffer>>,
    pending_gate: Option<Arc<ops::hip::DeviceBuffer>>,
    overlap_key: Option<Arc<ops::hip::DeviceBuffer>>,
    overlap_gate: Option<Arc<ops::hip::DeviceBuffer>>,
}

pub struct RocmGatedPoolSerde {
    pub next_position: usize,
    pub entry_count: usize,
    pub pending_rows: usize,
    pub ratio: usize,
    pub width: usize,
    pub channels: usize,
    pub overlap: bool,
    pub pending_key: Option<Vec<u8>>,
    pub pending_gate: Option<Vec<u8>>,
    pub overlap_key: Option<Vec<u8>>,
    pub overlap_gate: Option<Vec<u8>>,
}

pub struct RocmCompressedKvSerde {
    pub window_size: usize,
    pub kv_width: usize,
    pub q8_group_size: usize,
    pub recent_capacity: usize,
    pub recent_start: usize,
    pub recent_len: usize,
    pub recent_first_position: usize,
    pub next_recent_position: Option<usize>,
    pub recent_shared: bool,
    pub recent_key: Vec<u8>,
    pub recent_key_scales: Vec<u8>,
    pub recent_value: Vec<u8>,
    pub recent_value_scales: Vec<u8>,
    pub compressed_positions: Vec<usize>,
    pub compressed_shared: bool,
    pub compressed_key: Vec<u8>,
    pub compressed_key_scales: Vec<u8>,
    pub compressed_value: Vec<u8>,
    pub compressed_value_scales: Vec<u8>,
    pub compressed_index_width: usize,
    pub compressed_index_key: Option<Vec<u8>>,
    pub compressor: RocmGatedPoolSerde,
    pub indexer: RocmGatedPoolSerde,
}

struct RocmCompressionReplay {
    key: RocmTensor,
    gate: RocmTensor,
    position_bias: RocmWeight,
    norm: RocmWeight,
    compression: KvCompressionSpec,
    width: usize,
    rotary_dim: usize,
    eps: f32,
}

struct RocmAttentionReplay {
    positions: Vec<usize>,
    key: RocmTensor,
    value: RocmTensor,
}

struct RocmCompressedTransaction {
    rows: Vec<usize>,
    recent_key: Arc<ops::hip::DeviceBuffer>,
    recent_key_scales: Arc<ops::hip::DeviceBuffer>,
    recent_value: Arc<ops::hip::DeviceBuffer>,
    recent_value_scales: Arc<ops::hip::DeviceBuffer>,
    recent_capacity: usize,
    recent_start: usize,
    recent_len: usize,
    recent_first_position: usize,
    next_recent_position: Option<usize>,
    compressed_len: usize,
    compressor: RocmGatedPoolSnapshot,
    indexer: RocmGatedPoolSnapshot,
    attention_replays: Vec<RocmAttentionReplay>,
    compressor_replays: Vec<RocmCompressionReplay>,
    indexer_replays: Vec<RocmCompressionReplay>,
}

impl RocmGatedPoolState {
    fn fork_session(&self, device_id: i32) -> Result<Self, BackendError> {
        Ok(Self {
            state: self.state,
            ratio: self.ratio,
            width: self.width,
            channels: self.channels,
            overlap: self.overlap,
            pending_key: copy_optional_buffer(device_id, self.pending_key.as_ref(), None)?,
            pending_gate: copy_optional_buffer(device_id, self.pending_gate.as_ref(), None)?,
            overlap_key: copy_optional_buffer(device_id, self.overlap_key.as_ref(), None)?,
            overlap_gate: copy_optional_buffer(device_id, self.overlap_gate.as_ref(), None)?,
            rope: self.rope.clone(),
        })
    }

    fn reset_session(&mut self) {
        self.state.reset();
        // device state 保留；entry_start=0 时 kernel 会忽略旧 overlap，pending_rows=0
        // 时旧 pending 也不可见，因此无需 memset 或重新分配。
    }

    fn snapshot(&self, device_id: i32, reuse: Option<RocmGatedPoolSnapshot>) -> Result<RocmGatedPoolSnapshot, BackendError> {
        let reuse = reuse.unwrap_or(RocmGatedPoolSnapshot { next_position: 0, entry_count: 0, pending_rows: 0, pending_key: None, pending_gate: None, overlap_key: None, overlap_gate: None });
        let (next_position, entry_count, pending_rows) = self.state.parts();
        Ok(RocmGatedPoolSnapshot {
            next_position,
            entry_count,
            pending_rows,
            pending_key: copy_optional_buffer(device_id, self.pending_key.as_ref(), reuse.pending_key)?,
            pending_gate: copy_optional_buffer(device_id, self.pending_gate.as_ref(), reuse.pending_gate)?,
            overlap_key: copy_optional_buffer(device_id, self.overlap_key.as_ref(), reuse.overlap_key)?,
            overlap_gate: copy_optional_buffer(device_id, self.overlap_gate.as_ref(), reuse.overlap_gate)?,
        })
    }

    fn swap_snapshot(&mut self, snapshot: &mut RocmGatedPoolSnapshot) {
        let current = self.state.parts();
        self.state = CompressionState::from_parts(snapshot.next_position, snapshot.entry_count, snapshot.pending_rows);
        (snapshot.next_position, snapshot.entry_count, snapshot.pending_rows) = current;
        std::mem::swap(&mut self.pending_key, &mut snapshot.pending_key);
        std::mem::swap(&mut self.pending_gate, &mut snapshot.pending_gate);
        std::mem::swap(&mut self.overlap_key, &mut snapshot.overlap_key);
        std::mem::swap(&mut self.overlap_gate, &mut snapshot.overlap_gate);
    }

    fn replay_prefix(&mut self, context: &RocmContext, replay: RocmCompressionReplay, rows: usize) -> Result<Vec<usize>, BackendError> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let key = f32_tensor(context, &replay.key)?;
        let gate = f32_tensor(context, &replay.gate)?;
        if rows > key.rows || rows > gate.rows {
            return Err(compute(format!("V4 ROCm compressor replay rows={rows} 超过 key/gate {}/{}", key.rows, gate.rows)));
        }
        let key_device = key.device.as_deref().ok_or_else(|| compute("V4 ROCm compressor replay key 缺少 device buffer"))?;
        let gate_device = gate.device.as_deref().ok_or_else(|| compute("V4 ROCm compressor replay gate 缺少 device buffer"))?;
        let position_bias = constant(&replay.position_bias, replay.compression.ratio * self.channels, "compressor replay position bias")?;
        let norm = constant(&replay.norm, replay.width, "compressor replay norm")?;
        let pending_key = self.pending_key.as_deref().ok_or_else(|| compute("V4 ROCm compressor replay pending key 未初始化"))?;
        let pending_gate = self.pending_gate.as_deref().ok_or_else(|| compute("V4 ROCm compressor replay pending gate 未初始化"))?;
        let overlap_key = self.overlap_key.as_deref().ok_or_else(|| compute("V4 ROCm compressor replay overlap key 未初始化"))?;
        let overlap_gate = self.overlap_gate.as_deref().ok_or_else(|| compute("V4 ROCm compressor replay overlap gate 未初始化"))?;
        let rope = self.rope.as_deref().ok_or_else(|| compute("V4 ROCm compressor replay RoPE 未初始化"))?;
        let plan = self.state.plan_rows(rows, replay.compression.ratio).map_err(compute)?;
        let pending_rows = plan.pending_rows();
        let entry_start = plan.entry_start();
        let (_, remaining) = ops::hip::try_csa_gated_compress_f32(
            context.device_id,
            pending_key,
            pending_gate,
            pending_rows,
            key_device,
            gate_device,
            rows,
            position_bias,
            norm,
            overlap_key,
            overlap_gate,
            replay.compression.ratio,
            replay.width,
            replay.compression.overlap,
            entry_start,
            replay.rotary_dim,
            &rope.cos,
            &rope.sin,
            rope.elements,
            replay.eps,
        )
        .map_err(compute)?;
        self.state.commit(plan, remaining).map_err(compute)?;
        Ok(plan.visible_positions())
    }

    fn ensure_buffers(&mut self, context: &RocmContext, ratio: usize, width: usize, overlap: bool) -> Result<(), BackendError> {
        let channels = if overlap { width.checked_mul(2).ok_or_else(|| compute("V4 ROCm compressor channels 溢出"))? } else { width };
        if self.pending_key.is_none() {
            let state_elements = ratio.checked_mul(channels).ok_or_else(|| compute("V4 ROCm compressor state 溢出"))?;
            self.ratio = ratio;
            self.width = width;
            self.channels = channels;
            self.overlap = overlap;
            self.pending_key = Some(upload_f32(context.device_id, &vec![0.0; state_elements])?);
            self.pending_gate = Some(upload_f32(context.device_id, &vec![0.0; state_elements])?);
            if overlap {
                let overlap_elements = ratio.checked_mul(width).ok_or_else(|| compute("V4 ROCm overlap state 溢出"))?;
                self.overlap_key = Some(upload_f32(context.device_id, &vec![0.0; overlap_elements])?);
                self.overlap_gate = Some(upload_f32(context.device_id, &vec![f32::NEG_INFINITY; overlap_elements])?);
            } else {
                self.overlap_key = Some(upload_f32(context.device_id, &[0.0])?);
                self.overlap_gate = Some(upload_f32(context.device_id, &[0.0])?);
            }
        } else if (self.ratio, self.width, self.channels, self.overlap) != (ratio, width, channels, overlap) {
            return Err(compute(format!("V4 ROCm compressor 规格改变: 原={}/{}/{}/{} 新={ratio}/{width}/{channels}/{overlap}", self.ratio, self.width, self.channels, self.overlap,)));
        }
        Ok(())
    }

    /// RoPE 是模型常量：同一 host 表在一个 device 上由所有 layer/session 共享，
    /// 并只按实际到达的位置提交 prefix，不能为每份 KV storage 上传完整长上下文表。
    fn ensure_rope_tables(&mut self, context: &RocmContext, cos: &[f32], sin: &[f32], required: usize) -> Result<(), BackendError> {
        if cos.len() != sin.len() {
            return Err(compute(format!("V4 ROCm RoPE cos={} sin={} 不一致", cos.len(), sin.len())));
        }
        let required = required.max(1);
        if required > cos.len() {
            return Err(compute(format!("V4 ROCm RoPE 需要 {required} elements，表长 {}", cos.len())));
        }
        let key = RocmRopeTableKey { device_id: context.device_id, cos: cos.as_ptr() as usize, sin: sin.as_ptr() as usize, elements: cos.len() };
        if self.rope.as_ref().is_some_and(|rope| rope.key == key && rope.elements >= required) {
            return Ok(());
        }
        let cache = ROCM_ROPE_TABLES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().map_err(|_| compute("V4 ROCm RoPE cache mutex poisoned"))?;
        let shared = cache.get(&key).and_then(Weak::upgrade);
        if let Some(shared) = shared.as_ref().filter(|shared| shared.elements >= required) {
            self.rope = Some(shared.clone());
            return Ok(());
        }
        let current = shared.as_ref().map_or(0, |shared| shared.elements).max(self.rope.as_ref().filter(|rope| rope.key == key).map_or(0, |rope| rope.elements));
        let elements = required.max(current.saturating_mul(2)).checked_next_power_of_two().unwrap_or(cos.len()).min(cos.len()).max(required);
        let shared = Arc::new(RocmRopeTables { key, cos: upload_f32(context.device_id, &cos[..elements])?, sin: upload_f32(context.device_id, &sin[..elements])?, elements });
        cache.insert(key, Arc::downgrade(&shared));
        self.rope = Some(shared);
        Ok(())
    }
}

pub struct RocmCompressedKvStorage {
    window_size: usize,
    kv_width: usize,
    q8_group_size: usize,
    recent_key: Arc<ops::hip::DeviceBuffer>,
    recent_key_scales: Arc<ops::hip::DeviceBuffer>,
    recent_value: Arc<ops::hip::DeviceBuffer>,
    recent_value_scales: Arc<ops::hip::DeviceBuffer>,
    recent_capacity: usize,
    recent_start: usize,
    recent_len: usize,
    recent_first_position: usize,
    next_recent_position: Option<usize>,
    compressed_key: Arc<ops::hip::DeviceBuffer>,
    compressed_key_scales: Arc<ops::hip::DeviceBuffer>,
    compressed_value: Arc<ops::hip::DeviceBuffer>,
    compressed_value_scales: Arc<ops::hip::DeviceBuffer>,
    compressed_index_key: Option<Arc<ops::hip::DeviceBuffer>>,
    compressed_index_width: usize,
    compressed_positions: Vec<usize>,
    compressed_capacity: usize,
    visible_scalar_value: Option<u32>,
    visible_scalar: Option<Arc<ops::hip::DeviceBuffer>>,
    batch_key: Option<Arc<ops::hip::DeviceBuffer>>,
    batch_key_scales: Option<Arc<ops::hip::DeviceBuffer>>,
    batch_value: Option<Arc<ops::hip::DeviceBuffer>>,
    batch_value_scales: Option<Arc<ops::hip::DeviceBuffer>>,
    batch_capacity: usize,
    block_table: RocmBlockTable,
    compressor: RocmGatedPoolState,
    indexer: RocmGatedPoolState,
    transaction: Option<RocmCompressedTransaction>,
    transaction_pool: Option<RocmCompressedTransaction>,
}

impl RocmCompressedKvStorage {
    /// 从不可变终点 graph 构造可写 session。recent ring 与压缩器状态会被后续
    /// token 原地修改，因此立即私有化；只追加的 compressed prefix 继续共享，
    /// 并把逻辑 capacity 收紧到当前长度，确保首次追加必经 COW。
    pub fn fork_session(&self) -> Result<Self, BackendError> {
        if self.transaction.is_some() {
            return Err(compute("V4 ROCm terminal graph 仍有 speculative transaction"));
        }
        let device_id = self.recent_key.device_id();
        let shared_recent = Arc::ptr_eq(&self.recent_key, &self.recent_value) && Arc::ptr_eq(&self.recent_key_scales, &self.recent_value_scales);
        let recent_key = clone_required_buffer(device_id, &self.recent_key)?;
        let recent_key_scales = clone_required_buffer(device_id, &self.recent_key_scales)?;
        let recent_value = if shared_recent { recent_key.clone() } else { clone_required_buffer(device_id, &self.recent_value)? };
        let recent_value_scales = if shared_recent { recent_key_scales.clone() } else { clone_required_buffer(device_id, &self.recent_value_scales)? };
        Ok(Self {
            window_size: self.window_size,
            kv_width: self.kv_width,
            q8_group_size: self.q8_group_size,
            recent_key,
            recent_key_scales,
            recent_value,
            recent_value_scales,
            recent_capacity: self.recent_capacity,
            recent_start: self.recent_start,
            recent_len: self.recent_len,
            recent_first_position: self.recent_first_position,
            next_recent_position: self.next_recent_position,
            compressed_key: self.compressed_key.clone(),
            compressed_key_scales: self.compressed_key_scales.clone(),
            compressed_value: self.compressed_value.clone(),
            compressed_value_scales: self.compressed_value_scales.clone(),
            compressed_index_key: self.compressed_index_key.clone(),
            compressed_index_width: self.compressed_index_width,
            compressed_positions: self.compressed_positions.clone(),
            compressed_capacity: self.compressed_positions.len(),
            visible_scalar_value: self.visible_scalar_value,
            visible_scalar: self.visible_scalar.clone(),
            batch_key: None,
            batch_key_scales: None,
            batch_value: None,
            batch_value_scales: None,
            batch_capacity: 0,
            block_table: RocmBlockTable::new(),
            compressor: self.compressor.fork_session(device_id)?,
            indexer: self.indexer.fork_session(device_id)?,
            transaction: None,
            transaction_pool: None,
        })
    }

    pub fn reset_session(&mut self) {
        if let Some(transaction) = self.transaction.take() {
            self.transaction_pool = Some(transaction);
        }
        self.recent_start = 0;
        self.recent_len = 0;
        self.recent_first_position = 0;
        self.next_recent_position = None;
        self.compressed_positions.clear();
        // reusable session 不能把上一条长会话的 high-water capacity 继承给新会话。
        // buffer 在首次写入时按新请求实际规模替换，旧 allocation 随后回到可驱逐池。
        self.compressed_capacity = 1;
        self.compressed_index_key = None;
        self.compressed_index_width = 0;
        self.batch_key = None;
        self.batch_key_scales = None;
        self.batch_value = None;
        self.batch_value_scales = None;
        self.batch_capacity = 0;
        self.visible_scalar_value = None;
        self.compressor.reset_session();
        self.indexer.reset_session();
    }

    pub fn allocated_bytes(&self) -> u64 {
        let mut seen = std::collections::HashSet::new();
        let mut bytes = 0u64;
        let mut add = |buffer: Option<&Arc<ops::hip::DeviceBuffer>>| {
            if let Some(buffer) = buffer
                && seen.insert(buffer.device_pointer())
            {
                bytes = bytes.saturating_add(buffer.allocation_bytes() as u64);
            }
        };
        for buffer in [&self.recent_key, &self.recent_key_scales, &self.recent_value, &self.recent_value_scales, &self.compressed_key, &self.compressed_key_scales, &self.compressed_value, &self.compressed_value_scales] {
            add(Some(buffer));
        }
        add(self.compressed_index_key.as_ref());
        add(self.batch_key.as_ref());
        add(self.batch_key_scales.as_ref());
        add(self.batch_value.as_ref());
        add(self.batch_value_scales.as_ref());
        bytes
    }

    pub fn batch_allocated_bytes(&self) -> u64 {
        let mut seen = std::collections::HashSet::new();
        [&self.batch_key, &self.batch_key_scales, &self.batch_value, &self.batch_value_scales].into_iter().filter_map(Option::as_ref).filter(|buffer| seen.insert(buffer.device_pointer())).map(|buffer| buffer.allocation_bytes() as u64).sum()
    }

    /// DSpark target 历史与临时 noise block 使用两份 storage；每轮只复制 128 行 recent ring。
    pub fn copy_recent_from(&mut self, source: &Self) -> Result<(), BackendError> {
        if self.window_size != source.window_size || self.kv_width != source.kv_width || self.q8_group_size != source.q8_group_size || !source.compressed_positions.is_empty() {
            return Err(compute("V4 ROCm DSpark recent cache 规格不一致或 source 含 compressed history"));
        }
        self.ensure_recent_capacity(source.recent_key.device_id(), source.recent_capacity)?;
        let copy = |destination: &Arc<ops::hip::DeviceBuffer>, source: &Arc<ops::hip::DeviceBuffer>| {
            if destination.bytes() < source.bytes() {
                return Err(compute(format!("V4 ROCm DSpark recent cache bytes {} < {}", destination.bytes(), source.bytes())));
            }
            destination.copy_from_device(0, source, 0, source.bytes()).map_err(compute)
        };
        let source_shared = Arc::ptr_eq(&source.recent_key, &source.recent_value) && Arc::ptr_eq(&source.recent_key_scales, &source.recent_value_scales);
        let destination_shared = Arc::ptr_eq(&self.recent_key, &self.recent_value) && Arc::ptr_eq(&self.recent_key_scales, &self.recent_value_scales);
        if source_shared && !destination_shared {
            self.recent_value = self.recent_key.clone();
            self.recent_value_scales = self.recent_key_scales.clone();
        } else if !source_shared && destination_shared {
            let (value, value_scales) = Self::allocate_q8_pair(self.recent_key.device_id(), self.recent_capacity, self.kv_width, self.scale_row_bytes(), "recent value split")?;
            self.recent_value = value;
            self.recent_value_scales = value_scales;
        }
        copy(&self.recent_key, &source.recent_key)?;
        copy(&self.recent_key_scales, &source.recent_key_scales)?;
        if !source_shared {
            copy(&self.recent_value, &source.recent_value)?;
            copy(&self.recent_value_scales, &source.recent_value_scales)?;
        }
        self.recent_start = source.recent_start;
        self.recent_len = source.recent_len;
        self.recent_first_position = source.recent_first_position;
        self.next_recent_position = source.next_recent_position;
        Ok(())
    }

    fn scale_row_bytes(&self) -> usize {
        self.kv_width / self.q8_group_size * mem::size_of::<u16>()
    }

    fn allocate_q8_pair(device_id: i32, rows: usize, row_bytes: usize, scale_row_bytes: usize, name: &str) -> Result<(Arc<ops::hip::DeviceBuffer>, Arc<ops::hip::DeviceBuffer>), BackendError> {
        let code_bytes = rows.checked_mul(row_bytes).ok_or_else(|| compute(format!("V4 ROCm {name} Q8 codes 溢出")))?;
        let scale_bytes = rows.checked_mul(scale_row_bytes).ok_or_else(|| compute(format!("V4 ROCm {name} Q8 scales 溢出")))?;
        Ok((Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, code_bytes).map_err(compute)?), Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, scale_bytes).map_err(compute)?)))
    }

    /// 与通用 ROCm KV cache 一致，只提交当前实际需要的 block；旧 buffer 由 HIP pool 回收。
    fn ensure_recent_capacity(&mut self, device_id: i32, required: usize) -> Result<(), BackendError> {
        let next = committed_cache_rows(self.recent_capacity, required, self.window_size)?;
        if next == self.recent_capacity {
            return Ok(());
        }
        let row_bytes = self.kv_width;
        let scale_row_bytes = self.scale_row_bytes();
        let shared = Arc::ptr_eq(&self.recent_key, &self.recent_value) && Arc::ptr_eq(&self.recent_key_scales, &self.recent_value_scales);
        let key = grow_cache_buffer(device_id, &self.recent_key, self.recent_len * row_bytes, next * row_bytes)?;
        let key_scales = grow_cache_buffer(device_id, &self.recent_key_scales, self.recent_len * scale_row_bytes, next * scale_row_bytes)?;
        let value = if shared { key.clone() } else { grow_cache_buffer(device_id, &self.recent_value, self.recent_len * row_bytes, next * row_bytes)? };
        let value_scales = if shared { key_scales.clone() } else { grow_cache_buffer(device_id, &self.recent_value_scales, self.recent_len * scale_row_bytes, next * scale_row_bytes)? };
        self.recent_key = key;
        self.recent_key_scales = key_scales;
        self.recent_value = value;
        self.recent_value_scales = value_scales;
        self.recent_capacity = next;
        Ok(())
    }

    /// 当前 chunk 只量化一次；同一份 Q8 数据同时供 attention 和 recent ring 使用。
    fn quantize_batch(&mut self, context: &RocmContext, key: &RocmTensor, value: &RocmTensor) -> Result<(), BackendError> {
        if key.rows == 0 || key.rows != value.rows || key.cols != self.kv_width || value.cols != self.kv_width {
            return Err(compute(format!("V4 ROCm batch Q8 shape 非法: key=[{},{}] value=[{},{}] width={}", key.rows, key.cols, value.rows, value.cols, self.kv_width)));
        }
        let shared_input = match (&key.device, &value.device) {
            (Some(key), Some(value)) => Arc::ptr_eq(key, value),
            _ => false,
        };
        let required_capacity = key.rows.next_power_of_two();
        let shrink_decode_workspace = required_capacity <= ROCM_KV_BLOCK_SIZE && required_capacity < self.batch_capacity;
        if shrink_decode_workspace {
            self.batch_key = None;
            self.batch_key_scales = None;
            self.batch_value = None;
            self.batch_value_scales = None;
            self.batch_capacity = 0;
        }
        let shared_cache = match (&self.batch_key, &self.batch_value) {
            (Some(key), Some(value)) => Arc::ptr_eq(key, value),
            _ => false,
        };
        if required_capacity > self.batch_capacity || shared_input != shared_cache || shrink_decode_workspace {
            let capacity = if shrink_decode_workspace { required_capacity } else { required_capacity.max(self.batch_capacity) };
            let (key, key_scales) = Self::allocate_q8_pair(context.device_id, capacity, self.kv_width, self.scale_row_bytes(), "batch key")?;
            self.batch_key = Some(key);
            self.batch_key_scales = Some(key_scales);
            if shared_input {
                self.batch_value = self.batch_key.clone();
                self.batch_value_scales = self.batch_key_scales.clone();
            } else {
                let (value, value_scales) = Self::allocate_q8_pair(context.device_id, capacity, self.kv_width, self.scale_row_bytes(), "batch value")?;
                self.batch_value = Some(value);
                self.batch_value_scales = Some(value_scales);
            }
            self.batch_capacity = capacity;
        }
        let key_input = key.device.as_deref().ok_or_else(|| compute("V4 ROCm batch key 缺少 device buffer"))?;
        let value_input = value.device.as_deref().ok_or_else(|| compute("V4 ROCm batch value 缺少 device buffer"))?;
        let key_cache = self.batch_key.as_deref().ok_or_else(|| compute("V4 ROCm batch Q8 key 未初始化"))?;
        let value_cache = self.batch_value.as_deref().ok_or_else(|| compute("V4 ROCm batch Q8 value 未初始化"))?;
        let key_scales = self.batch_key_scales.as_deref().ok_or_else(|| compute("V4 ROCm batch Q8 key scales 未初始化"))?;
        let value_scales = self.batch_value_scales.as_deref().ok_or_else(|| compute("V4 ROCm batch Q8 value scales 未初始化"))?;
        let table = self.block_table.get("CSA batch", self.batch_capacity, context.device_id, key.rows)?;
        ops::hip::try_paged_cache_append_f32_q8(context.device_id, key_input, key_cache, key_scales, &table, 0, key.rows, self.kv_width, self.q8_group_size, ROCM_KV_BLOCK_SIZE).map_err(compute)?;
        if shared_input { Ok(()) } else { ops::hip::try_paged_cache_append_f32_q8(context.device_id, value_input, value_cache, value_scales, &table, 0, value.rows, self.kv_width, self.q8_group_size, ROCM_KV_BLOCK_SIZE).map_err(compute) }
    }

    fn append_compressed(&mut self, context: &RocmContext, positions: &[usize], key: &RocmTensor, value: &RocmTensor, index_key: Option<&RocmTensor>) -> Result<(), BackendError> {
        if positions.is_empty() {
            return Ok(());
        }
        if key.rows != positions.len()
            || value.rows != positions.len()
            || key.cols != self.kv_width
            || value.cols != self.kv_width
            || positions.windows(2).any(|pair| pair[1] <= pair[0])
            || self.compressed_positions.last().is_some_and(|last| positions[0] <= *last)
        {
            return Err(compute(format!(
                "V4 ROCm compressed append 非法: positions={positions:?} key=[{},{}] value=[{},{}] width={} history={:?}",
                key.rows,
                key.cols,
                value.rows,
                value.cols,
                self.kv_width,
                self.compressed_positions.last(),
            )));
        }
        let shared_input = std::ptr::eq(key, value);
        let key = context.tensor_as_f32(key.clone())?;
        let value = if shared_input { key.clone() } else { context.tensor_as_f32(value.clone())? };
        let index_key = index_key.map(|tensor| context.tensor_as_f32(tensor.clone())).transpose()?;
        if let Some(index) = &index_key {
            if index.rows != positions.len() || index.cols == 0 {
                return Err(compute(format!("V4 ROCm compressed index key=[{},{}] 与 rows={} 不符", index.rows, index.cols, positions.len())));
            }
            if self.compressed_index_width != 0 && self.compressed_index_width != index.cols {
                return Err(compute(format!("V4 ROCm compressed index width 从 {} 变为 {}", self.compressed_index_width, index.cols)));
            }
            if self.compressed_index_key.is_none() && !self.compressed_positions.is_empty() {
                return Err(compute("V4 ROCm compressed 历史缺少 index key"));
            }
        } else if self.compressed_index_key.is_some() {
            return Err(compute("V4 ROCm compressed append 缺少 index key"));
        }

        let old_rows = self.compressed_positions.len();
        let required = old_rows.checked_add(positions.len()).ok_or_else(|| compute("V4 ROCm compressed rows 溢出"))?;
        let compacting_empty = old_rows == 0 && (self.compressed_key.bytes() > self.compressed_capacity * self.kv_width || self.compressed_key_scales.bytes() > self.compressed_capacity * self.scale_row_bytes());
        let growing = required > self.compressed_capacity || compacting_empty;
        let shared_cache = Arc::ptr_eq(&self.compressed_key, &self.compressed_value) && Arc::ptr_eq(&self.compressed_key_scales, &self.compressed_value_scales);
        let share_cache = shared_input && (old_rows == 0 || shared_cache);
        let split_cache = shared_cache && !share_cache;
        let new_capacity = if growing { required.next_power_of_two() } else { self.compressed_capacity };
        let row_bytes = self.kv_width;
        let scale_row_bytes = self.scale_row_bytes();
        let new_key = if growing { Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, new_capacity * row_bytes).map_err(compute)?) } else { self.compressed_key.clone() };
        let new_key_scales = if growing { Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, new_capacity * scale_row_bytes).map_err(compute)?) } else { self.compressed_key_scales.clone() };
        let new_value = if share_cache {
            new_key.clone()
        } else if growing || split_cache {
            Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, new_capacity * row_bytes).map_err(compute)?)
        } else {
            self.compressed_value.clone()
        };
        let new_value_scales = if share_cache {
            new_key_scales.clone()
        } else if growing || split_cache {
            Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, new_capacity * scale_row_bytes).map_err(compute)?)
        } else {
            self.compressed_value_scales.clone()
        };
        let index_width = index_key.as_ref().map_or(self.compressed_index_width, |tensor| tensor.cols);
        let index_row_bytes = index_width.checked_mul(mem::size_of::<f32>()).ok_or_else(|| compute("V4 ROCm index row bytes 溢出"))?;
        let new_index = if index_width == 0 {
            None
        } else if growing || self.compressed_index_key.is_none() {
            Some(Arc::new(ops::hip::DeviceBuffer::allocate_cache(context.device_id, new_capacity * index_row_bytes).map_err(compute)?))
        } else {
            self.compressed_index_key.clone()
        };

        if growing && old_rows != 0 {
            new_key.copy_from_device(0, &self.compressed_key, 0, old_rows * row_bytes).map_err(compute)?;
            new_key_scales.copy_from_device(0, &self.compressed_key_scales, 0, old_rows * scale_row_bytes).map_err(compute)?;
        }
        if !share_cache && (growing || split_cache) && old_rows != 0 {
            new_value.copy_from_device(0, &self.compressed_value, 0, old_rows * row_bytes).map_err(compute)?;
            new_value_scales.copy_from_device(0, &self.compressed_value_scales, 0, old_rows * scale_row_bytes).map_err(compute)?;
        }
        if growing && old_rows != 0 {
            if let (Some(old), Some(new)) = (&self.compressed_index_key, &new_index) {
                new.copy_from_device(0, old, 0, old_rows * index_row_bytes).map_err(compute)?;
            }
        }
        let key_device = key.device.as_deref().ok_or_else(|| compute("V4 ROCm compressed key 缺少 device buffer"))?;
        let value_device = value.device.as_deref().ok_or_else(|| compute("V4 ROCm compressed value 缺少 device buffer"))?;
        let table = self.block_table.get("CSA compressed", new_capacity, context.device_id, required)?;
        ops::hip::try_paged_cache_append_f32_q8(context.device_id, key_device, &new_key, &new_key_scales, &table, old_rows, positions.len(), self.kv_width, self.q8_group_size, ROCM_KV_BLOCK_SIZE).map_err(compute)?;
        if !share_cache {
            ops::hip::try_paged_cache_append_f32_q8(context.device_id, value_device, &new_value, &new_value_scales, &table, old_rows, positions.len(), self.kv_width, self.q8_group_size, ROCM_KV_BLOCK_SIZE).map_err(compute)?;
        }
        if let (Some(index), Some(destination)) = (&index_key, &new_index) {
            let source = index.device.as_deref().ok_or_else(|| compute("V4 ROCm compressed index key 缺少 device buffer"))?;
            destination.copy_from_device(old_rows * index_row_bytes, source, 0, positions.len() * index_row_bytes).map_err(compute)?;
        }
        self.compressed_key = new_key;
        self.compressed_key_scales = new_key_scales;
        self.compressed_value = new_value;
        self.compressed_value_scales = new_value_scales;
        self.compressed_index_key = new_index;
        self.compressed_index_width = index_width;
        self.compressed_capacity = new_capacity;
        self.compressed_positions.extend_from_slice(positions);
        Ok(())
    }

    fn validate_recent_positions(&self, positions: &[usize]) -> Result<(), BackendError> {
        if positions.is_empty() || positions.windows(2).any(|pair| pair[1] != pair[0] + 1) || self.next_recent_position.is_some_and(|next| positions[0] != next) {
            return Err(compute(format!("V4 ROCm recent positions 非连续: positions={positions:?} next={:?}", self.next_recent_position)));
        }
        Ok(())
    }

    fn append_recent(&mut self, positions: &[usize]) -> Result<(), BackendError> {
        self.validate_recent_positions(positions)?;
        let required = self.recent_len.saturating_add(positions.len()).min(self.window_size);
        self.ensure_recent_capacity(self.recent_key.device_id(), required)?;
        let key_device = self.batch_key.as_deref().ok_or_else(|| compute("V4 ROCm recent batch key 缺失"))?;
        let value_device = self.batch_value.as_deref().ok_or_else(|| compute("V4 ROCm recent batch value 缺失"))?;
        let key_scales = self.batch_key_scales.as_deref().ok_or_else(|| compute("V4 ROCm recent batch key scales 缺失"))?;
        let value_scales = self.batch_value_scales.as_deref().ok_or_else(|| compute("V4 ROCm recent batch value scales 缺失"))?;
        let row_bytes = self.kv_width;
        let scale_row_bytes = self.scale_row_bytes();
        let shared_batch = std::ptr::eq(key_device, value_device) && std::ptr::eq(key_scales, value_scales);
        let shared_recent = Arc::ptr_eq(&self.recent_key, &self.recent_value) && Arc::ptr_eq(&self.recent_key_scales, &self.recent_value_scales);
        if shared_batch && self.recent_len == 0 && !shared_recent {
            self.recent_value = self.recent_key.clone();
            self.recent_value_scales = self.recent_key_scales.clone();
        } else if !shared_batch && shared_recent {
            let (value, value_scales) = Self::allocate_q8_pair(key_device.device_id(), self.recent_capacity, row_bytes, scale_row_bytes, "recent value split")?;
            value.copy_from_device(0, &self.recent_value, 0, self.recent_capacity * row_bytes).map_err(compute)?;
            value_scales.copy_from_device(0, &self.recent_value_scales, 0, self.recent_capacity * scale_row_bytes).map_err(compute)?;
            self.recent_value = value;
            self.recent_value_scales = value_scales;
        }
        let shared_recent = Arc::ptr_eq(&self.recent_key, &self.recent_value) && Arc::ptr_eq(&self.recent_key_scales, &self.recent_value_scales);
        // decode 单行在 attention 之后用一个 kernel 同时推进 K/V code 与 scale；
        // 此时旧 recent window 已消费完，覆盖 ring 头不会改变本轮可见历史。
        let fused_decode_copy = positions.len() == 1 && !shared_recent;
        if fused_decode_copy {
            let target_row = if positions.len() >= self.window_size { 0 } else { (self.recent_start + self.recent_len) % self.window_size };
            ops::hip::try_q8_cache_copy_pair(
                key_device.device_id(),
                key_device,
                key_scales,
                value_device,
                value_scales,
                &self.recent_key,
                &self.recent_key_scales,
                &self.recent_value,
                &self.recent_value_scales,
                0,
                target_row,
                1,
                self.kv_width,
                self.kv_width / self.q8_group_size,
                self.recent_capacity,
            )
            .map_err(compute)?;
        }
        if positions.len() >= self.window_size {
            let source_row = positions.len() - self.window_size;
            let copy_bytes = self.window_size * row_bytes;
            if !fused_decode_copy {
                self.recent_key.copy_from_device(0, key_device, source_row * row_bytes, copy_bytes).map_err(compute)?;
                self.recent_key_scales.copy_from_device(0, key_scales, source_row * scale_row_bytes, self.window_size * scale_row_bytes).map_err(compute)?;
                if !shared_recent {
                    self.recent_value.copy_from_device(0, value_device, source_row * row_bytes, copy_bytes).map_err(compute)?;
                    self.recent_value_scales.copy_from_device(0, value_scales, source_row * scale_row_bytes, self.window_size * scale_row_bytes).map_err(compute)?;
                }
            }
            self.recent_start = 0;
            self.recent_len = self.window_size;
            self.recent_first_position = positions[source_row];
        } else {
            let tail = (self.recent_start + self.recent_len) % self.window_size;
            let first_rows = positions.len().min(self.window_size - tail);
            let first_bytes = first_rows * row_bytes;
            let first_scale_bytes = first_rows * scale_row_bytes;
            if !fused_decode_copy {
                self.recent_key.copy_from_device(tail * row_bytes, key_device, 0, first_bytes).map_err(compute)?;
                self.recent_key_scales.copy_from_device(tail * scale_row_bytes, key_scales, 0, first_scale_bytes).map_err(compute)?;
                if !shared_recent {
                    self.recent_value.copy_from_device(tail * row_bytes, value_device, 0, first_bytes).map_err(compute)?;
                    self.recent_value_scales.copy_from_device(tail * scale_row_bytes, value_scales, 0, first_scale_bytes).map_err(compute)?;
                }
                if first_rows < positions.len() {
                    let second_bytes = (positions.len() - first_rows) * row_bytes;
                    self.recent_key.copy_from_device(0, key_device, first_bytes, second_bytes).map_err(compute)?;
                    let second_scale_bytes = (positions.len() - first_rows) * scale_row_bytes;
                    self.recent_key_scales.copy_from_device(0, key_scales, first_scale_bytes, second_scale_bytes).map_err(compute)?;
                    if !shared_recent {
                        self.recent_value.copy_from_device(0, value_device, first_bytes, second_bytes).map_err(compute)?;
                        self.recent_value_scales.copy_from_device(0, value_scales, first_scale_bytes, second_scale_bytes).map_err(compute)?;
                    }
                }
            }
            let overflow = (self.recent_len + positions.len()).saturating_sub(self.window_size);
            if self.recent_len == 0 {
                self.recent_first_position = positions[0];
            } else {
                self.recent_first_position += overflow;
            }
            self.recent_start = (self.recent_start + overflow) % self.window_size;
            self.recent_len = (self.recent_len + positions.len()).min(self.window_size);
        }
        self.next_recent_position = positions.last().map(|position| position + 1);
        Ok(())
    }

    fn visible_counts(&self, positions: &[usize]) -> Result<Vec<u32>, BackendError> {
        positions
            .iter()
            .map(|position| {
                let count = self.compressed_positions.partition_point(|compressed| compressed <= position);
                u32::try_from(count).map_err(|_| compute(format!("V4 ROCm visible compressed rows={count} 超过 u32")))
            })
            .collect()
    }

    /// Decode 时可见压缩行数往往跨多个 token 不变。按 layer 复用这个标量
    /// device buffer，只在值变化时重新 fill；prefill 多行仍保持原有上传路径。
    fn visible_buffer(&mut self, device_id: i32, values: &[u32]) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
        if values.is_empty() {
            return Err(compute("V4 ROCm 不能上传空 u32 buffer"));
        }
        if let [value] = values {
            if self.visible_scalar_value == Some(*value)
                && let Some(buffer) = &self.visible_scalar
            {
                return Ok(buffer.clone());
            }
            let buffer = ops::hip::try_fill_resident_u32(device_id, *value, 1).map(Arc::new).map_err(compute)?;
            self.visible_scalar_value = Some(*value);
            self.visible_scalar = Some(buffer.clone());
            return Ok(buffer);
        }
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), mem::size_of_val(values)) };
        let buffer = ops::hip::DeviceBuffer::allocate_reusable(device_id, bytes.len()).map_err(compute)?;
        buffer.copy_from_host(bytes).map_err(compute)?;
        Ok(Arc::new(buffer))
    }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), mem::size_of_val(values)) }
}

fn upload_f32(device_id: i32, values: &[f32]) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
    if values.is_empty() {
        return Err(compute("V4 ROCm 不能上传空 F32 buffer"));
    }
    ops::hip::DeviceBuffer::upload(device_id, f32_bytes(values)).map(Arc::new).map_err(compute)
}

impl CompressedSparseKernel for RocmContext {
    type CompressedKvStorage = RocmCompressedKvStorage;

    fn allocate_compressed_kv(&self, spec: &CompressedSparseAttentionSpec) -> Result<Self::CompressedKvStorage, BackendError> {
        spec.validate().map_err(compute)?;
        let kv_width = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute("V4 ROCm KV width 溢出"))?;
        let q8_group_size = [crate::kv_cache::DEFAULT_GROUP_SIZE, 32, 16, 8, 4, 2, 1].into_iter().find(|group| spec.head_dim.is_multiple_of(*group)).ok_or_else(|| compute(format!("V4 ROCm head_dim={} 不支持 Q8 group", spec.head_dim)))?;
        let row_bytes = kv_width;
        let scale_row_bytes = kv_width / q8_group_size * mem::size_of::<u16>();
        let recent_capacity = spec.window_size.min(ROCM_KV_BLOCK_SIZE);
        let compressed_capacity = recent_capacity;
        let (recent_key, recent_key_scales) = RocmCompressedKvStorage::allocate_q8_pair(self.device_id, recent_capacity, row_bytes, scale_row_bytes, "recent key")?;
        let recent_value = recent_key.clone();
        let recent_value_scales = recent_key_scales.clone();
        let (compressed_key, compressed_key_scales) = RocmCompressedKvStorage::allocate_q8_pair(self.device_id, compressed_capacity, row_bytes, scale_row_bytes, "compressed key")?;
        let compressed_value = compressed_key.clone();
        let compressed_value_scales = compressed_key_scales.clone();
        Ok(RocmCompressedKvStorage {
            window_size: spec.window_size,
            kv_width,
            q8_group_size,
            recent_key,
            recent_key_scales,
            recent_value,
            recent_value_scales,
            recent_capacity,
            recent_start: 0,
            recent_len: 0,
            recent_first_position: 0,
            next_recent_position: None,
            compressed_key,
            compressed_key_scales,
            compressed_value,
            compressed_value_scales,
            compressed_index_key: None,
            compressed_index_width: 0,
            compressed_positions: Vec::new(),
            compressed_capacity,
            visible_scalar_value: None,
            visible_scalar: None,
            batch_key: None,
            batch_key_scales: None,
            batch_value: None,
            batch_value_scales: None,
            batch_capacity: 0,
            block_table: RocmBlockTable::new(),
            compressor: RocmGatedPoolState::default(),
            indexer: RocmGatedPoolState::default(),
            transaction: None,
            transaction_pool: None,
        })
    }

    fn compressed_rmsnorm_heads(&self, input: &RocmTensor, weight: &RocmWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<RocmTensor, BackendError> {
        if input.cols != head_count.checked_mul(head_dim).ok_or_else(|| compute("V4 ROCm head RMSNorm columns 溢出"))? {
            return Err(compute(format!("V4 ROCm head RMSNorm input=[{},{}] heads={head_count} dim={head_dim}", input.rows, input.cols)));
        }
        let input = f32_tensor(self, input)?;
        let input_device = input.device.as_deref().ok_or_else(|| compute("V4 ROCm head RMSNorm input 缺少 device buffer"))?;
        let weight = constant(weight, head_dim, "head RMSNorm")?;
        let rows = input.rows.checked_mul(head_count).ok_or_else(|| compute("V4 ROCm head RMSNorm rows 溢出"))?;
        let output = ops::hip::try_rmsnorm_resident_weight_to_f32(self.device_id, input_device, weight, rows, head_dim, eps, false).map_err(compute)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn compressed_sparse_store_recent(&self, storage: &mut Self::CompressedKvStorage, positions: &[usize], key: &RocmTensor, value: &RocmTensor) -> Result<(), BackendError> {
        if positions.len() != key.rows || positions.is_empty() || positions.windows(2).any(|pair| pair[1] != pair[0] + 1) {
            return Err(compute(format!("V4 ROCm CSA recent store positions={positions:?} key rows={}", key.rows)));
        }
        let shared_kv = std::ptr::eq(key, value);
        let key = f32_tensor(self, key)?;
        let value = if shared_kv { key.clone() } else { f32_tensor(self, value)? };
        storage.validate_recent_positions(positions)?;
        storage.quantize_batch(self, &key, &value)?;
        storage.record_transaction_attention(positions, &key, &value)?;
        storage.append_recent(positions)
    }

    fn compress_gated(
        &self,
        storage: &mut Self::CompressedKvStorage,
        stream: CompressionStream,
        positions: &[usize],
        key: &RocmTensor,
        gate: &RocmTensor,
        position_bias: &RocmWeight,
        norm: &RocmWeight,
        compression: KvCompressionSpec,
        width: usize,
        rotary_dim: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
    ) -> Result<CompressedBatch<RocmTensor>, BackendError> {
        if positions.len() != key.rows || positions.is_empty() || positions.windows(2).any(|pair| pair[1] != pair[0] + 1) {
            return Err(compute(format!("V4 ROCm compressor positions={positions:?} key rows={}", key.rows)));
        }
        let channels = if compression.overlap { width.checked_mul(2).ok_or_else(|| compute("V4 ROCm compressor channels 溢出"))? } else { width };
        if gate.rows != key.rows || key.cols != channels || gate.cols != channels || rotary_dim == 0 || rotary_dim > width || !rotary_dim.is_multiple_of(2) {
            return Err(compute(format!("V4 ROCm compressor shape 非法: key=[{},{}] gate=[{},{}] width={width} rotary={rotary_dim}", key.rows, key.cols, gate.rows, gate.cols)));
        }
        let key = f32_tensor(self, key)?;
        let gate = f32_tensor(self, gate)?;
        let key_device = key.device.as_deref().ok_or_else(|| compute("V4 ROCm compressor key 缺少 device buffer"))?;
        let gate_device = gate.device.as_deref().ok_or_else(|| compute("V4 ROCm compressor gate 缺少 device buffer"))?;
        if let Some(transaction) = storage.transaction.as_mut() {
            let replay = RocmCompressionReplay { key: key.clone(), gate: gate.clone(), position_bias: position_bias.clone(), norm: norm.clone(), compression, width, rotary_dim, eps };
            let replays = match stream {
                CompressionStream::Attention => &mut transaction.compressor_replays,
                CompressionStream::Indexer => &mut transaction.indexer_replays,
            };
            replays.push(replay);
        }
        let position_bias = constant(position_bias, compression.ratio * channels, "compressor position bias")?;
        let norm = constant(norm, width, "compressor norm")?;
        let state = match stream {
            CompressionStream::Attention => &mut storage.compressor,
            CompressionStream::Indexer => &mut storage.indexer,
        };
        let plan = state.state.plan(positions, compression.ratio).map_err(compute)?;
        let pending_rows = plan.pending_rows();
        let entry_start = plan.entry_start();
        let windows = plan.windows();
        let required_table = (entry_start + windows).checked_mul(compression.ratio).and_then(|rows| rows.checked_mul(rotary_dim / 2)).ok_or_else(|| compute("V4 ROCm compressor RoPE table offset 溢出"))?;
        if windows != 0 && required_table > cos.len() {
            return Err(compute(format!("V4 ROCm compressor RoPE table={}，需要 {required_table}", cos.len())));
        }
        state.ensure_buffers(self, compression.ratio, width, compression.overlap)?;
        state.ensure_rope_tables(self, cos, sin, required_table)?;
        let pending_key = state.pending_key.as_deref().ok_or_else(|| compute("V4 ROCm compressor pending key 未初始化"))?;
        let pending_gate = state.pending_gate.as_deref().ok_or_else(|| compute("V4 ROCm compressor pending gate 未初始化"))?;
        let overlap_key = state.overlap_key.as_deref().ok_or_else(|| compute("V4 ROCm compressor overlap key 未初始化"))?;
        let overlap_gate = state.overlap_gate.as_deref().ok_or_else(|| compute("V4 ROCm compressor overlap gate 未初始化"))?;
        let rope = state.rope.as_deref().ok_or_else(|| compute("V4 ROCm compressor RoPE 未初始化"))?;
        let (output, remaining) = ops::hip::try_csa_gated_compress_f32(
            self.device_id,
            pending_key,
            pending_gate,
            pending_rows,
            key_device,
            gate_device,
            key.rows,
            position_bias,
            norm,
            overlap_key,
            overlap_gate,
            compression.ratio,
            width,
            compression.overlap,
            entry_start,
            rotary_dim,
            &rope.cos,
            &rope.sin,
            rope.elements,
            eps,
        )
        .map_err(compute)?;
        state.state.commit(plan, remaining).map_err(compute)?;
        let visible_positions = plan.visible_positions();
        Ok(CompressedBatch { visible_positions, values: device_tensor_f32(output, windows, width) })
    }

    fn compress_gated_segmented(
        &self,
        segments: &mut [CompressedGatedSegment<'_, Self>],
        stream: CompressionStream,
        kv: &RocmTensor,
        gate: &RocmTensor,
        position_bias: &RocmWeight,
        norm: &RocmWeight,
        compression: KvCompressionSpec,
        width: usize,
        rotary_dim: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
    ) -> Result<Vec<CompressedBatch<RocmTensor>>, BackendError> {
        compress_gated_segmented_fallback(self, segments, stream, kv, gate, position_bias, norm, compression, width, rotary_dim, cos, sin, eps)
    }

    fn compressed_sparse_decode(
        &self,
        storage: &mut Self::CompressedKvStorage,
        position: usize,
        query: &RocmTensor,
        key: &RocmTensor,
        value: &RocmTensor,
        compressed_key: Option<&RocmTensor>,
        compressed_value: Option<&RocmTensor>,
        compressed_index_key: Option<&RocmTensor>,
        index_query: Option<&RocmTensor>,
        index_head_weights: Option<&RocmTensor>,
        sink: Option<&RocmWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<RocmTensor, BackendError> {
        if query.rows != 1 || key.rows != 1 || value.rows != 1 {
            return Err(compute("V4 ROCm CSA decode 的 Q/K/V 必须是单行 tensor"));
        }
        self.compressed_sparse_prefill(
            storage,
            &[position],
            true,
            query,
            key,
            value,
            compressed_key.map(|_| std::slice::from_ref(&position)),
            compressed_key,
            compressed_value,
            compressed_index_key,
            index_query,
            index_head_weights,
            sink,
            spec,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compressed_sparse_prefill(
        &self,
        storage: &mut Self::CompressedKvStorage,
        positions: &[usize],
        causal_batch: bool,
        query: &RocmTensor,
        key: &RocmTensor,
        value: &RocmTensor,
        compressed_positions: Option<&[usize]>,
        compressed_key: Option<&RocmTensor>,
        compressed_value: Option<&RocmTensor>,
        compressed_index_key: Option<&RocmTensor>,
        index_query: Option<&RocmTensor>,
        index_head_weights: Option<&RocmTensor>,
        sink: Option<&RocmWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<RocmTensor, BackendError> {
        if positions.len() != query.rows || positions.is_empty() || positions.windows(2).any(|pair| pair[1] != pair[0] + 1) {
            return Err(compute(format!("V4 ROCm CSA prefill positions={positions:?} query rows={}", query.rows)));
        }
        let shared_kv = std::ptr::eq(key, value);
        let query = f32_tensor(self, query)?;
        let key = f32_tensor(self, key)?;
        let value = if shared_kv { key.clone() } else { f32_tensor(self, value)? };
        match (compressed_positions, compressed_key, compressed_value) {
            (Some(compressed_positions), Some(compressed_key), Some(compressed_value)) => {
                storage.append_compressed(self, compressed_positions, compressed_key, compressed_value, compressed_index_key)?;
            }
            (None, None, None) if compressed_index_key.is_none() => {}
            _ => return Err(compute("V4 ROCm CSA compressed positions/K/V 必须成组出现")),
        }
        storage.validate_recent_positions(positions)?;
        let visible = if causal_batch { storage.visible_counts(positions)? } else { storage.visible_counts(&vec![*positions.last().expect("positions 已非空"); positions.len()])? };
        let visible_buffer = storage.visible_buffer(self.device_id, &visible)?;
        let selection = select_history(self, storage, index_query, index_head_weights, spec, &visible, &visible_buffer)?;
        let sink = sink.map(|weight| constant(weight, spec.num_heads, "attention sink")).transpose()?;
        let query_device = query.device.as_deref().ok_or_else(|| compute("V4 ROCm CSA query 缺少 device buffer"))?;
        storage.quantize_batch(self, &key, &value)?;
        storage.record_transaction_attention(positions, &key, &value)?;
        let batch_key = storage.batch_key.as_deref().ok_or_else(|| compute("V4 ROCm CSA batch Q8 key 缺失"))?;
        let batch_key_scales = storage.batch_key_scales.as_deref().ok_or_else(|| compute("V4 ROCm CSA batch Q8 key scales 缺失"))?;
        let batch_value = storage.batch_value.as_deref().ok_or_else(|| compute("V4 ROCm CSA batch Q8 value 缺失"))?;
        let batch_value_scales = storage.batch_value_scales.as_deref().ok_or_else(|| compute("V4 ROCm CSA batch Q8 value scales 缺失"))?;
        let output = ops::hip::try_csa_attention_q8(
            self.device_id,
            query_device,
            &storage.compressed_key,
            &storage.compressed_key_scales,
            &storage.compressed_value,
            &storage.compressed_value_scales,
            &visible_buffer,
            selection.as_ref().map(|(buffer, top_k)| (buffer.as_ref(), *top_k)),
            &storage.recent_key,
            &storage.recent_key_scales,
            &storage.recent_value,
            &storage.recent_value_scales,
            storage.recent_start,
            storage.recent_len,
            storage.recent_first_position,
            batch_key,
            batch_key_scales,
            batch_value,
            batch_value_scales,
            positions[0],
            causal_batch,
            sink,
            query.rows,
            storage.compressed_capacity,
            spec.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            spec.window_size,
            storage.q8_group_size,
        )
        .map_err(compute)?;
        storage.append_recent(positions)?;
        Ok(device_tensor_f32(output, query.rows, query.cols))
    }

    fn compressed_sparse_prefill_segmented(
        &self,
        query: &RocmTensor,
        key: &RocmTensor,
        value: &RocmTensor,
        index_query: Option<&RocmTensor>,
        index_head_weights: Option<&RocmTensor>,
        segments: &mut [CompressedSparsePrefillSegment<'_, Self>],
        sink: Option<&RocmWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<RocmTensor, BackendError> {
        let shared_kv = std::ptr::eq(key, value);
        let query = f32_tensor(self, query)?;
        let key = f32_tensor(self, key)?;
        let value = if shared_kv { key.clone() } else { f32_tensor(self, value)? };
        let index_query = index_query.map(|tensor| f32_tensor(self, tensor)).transpose()?;
        let index_head_weights = index_head_weights.map(|tensor| f32_tensor(self, tensor)).transpose()?;
        let total_rows = segments.iter().map(|segment| segment.positions.len()).sum::<usize>();
        if segments.is_empty() || total_rows != query.rows || key.rows != total_rows || value.rows != total_rows {
            return Err(compute(format!("V4 ROCm segmented CSA rows={total_rows}，Q/K/V={}/{}/{} segments={}", query.rows, key.rows, value.rows, segments.len())));
        }
        let sink = sink.map(|weight| constant(weight, spec.num_heads, "attention sink")).transpose()?;

        struct Prepared {
            compressed_key: Arc<ops::hip::DeviceBuffer>,
            compressed_key_scales: Arc<ops::hip::DeviceBuffer>,
            compressed_value: Arc<ops::hip::DeviceBuffer>,
            compressed_value_scales: Arc<ops::hip::DeviceBuffer>,
            visible: Arc<ops::hip::DeviceBuffer>,
            selection: Option<(Arc<ops::hip::DeviceBuffer>, usize)>,
            recent_key: Arc<ops::hip::DeviceBuffer>,
            recent_key_scales: Arc<ops::hip::DeviceBuffer>,
            recent_value: Arc<ops::hip::DeviceBuffer>,
            recent_value_scales: Arc<ops::hip::DeviceBuffer>,
            batch_key: Arc<ops::hip::DeviceBuffer>,
            batch_key_scales: Arc<ops::hip::DeviceBuffer>,
            batch_value: Arc<ops::hip::DeviceBuffer>,
            batch_value_scales: Arc<ops::hip::DeviceBuffer>,
            row_start: usize,
            rows: usize,
            recent_start: usize,
            recent_len: usize,
            recent_first_position: usize,
            position_start: usize,
            causal_batch: bool,
            compressed_capacity: usize,
            q8_group_size: usize,
        }

        let mut prepared = Vec::with_capacity(segments.len());
        let mut offset = 0;
        for segment in segments.iter_mut() {
            let rows = segment.positions.len();
            if rows == 0 || segment.positions.windows(2).any(|pair| pair[1] != pair[0] + 1) {
                return Err(compute(format!("V4 ROCm segmented CSA positions 非连续: {:?}", segment.positions)));
            }
            let local_key = self.slice_token_rows(&key, offset, rows)?;
            let local_value = if shared_kv { local_key.clone() } else { self.slice_token_rows(&value, offset, rows)? };
            match (segment.compressed_positions, segment.compressed_key, segment.compressed_value) {
                (Some(positions), Some(compressed_key), Some(compressed_value)) => segment.storage.append_compressed(self, positions, compressed_key, compressed_value, segment.compressed_index_key)?,
                (None, None, None) if segment.compressed_index_key.is_none() => {}
                _ => return Err(compute("V4 ROCm segmented CSA compressed positions/K/V 必须成组出现")),
            }
            segment.storage.validate_recent_positions(segment.positions)?;
            let visible = if segment.causal_batch { segment.storage.visible_counts(segment.positions)? } else { segment.storage.visible_counts(&vec![*segment.positions.last().expect("positions 已非空"); rows])? };
            let visible_buffer = segment.storage.visible_buffer(self.device_id, &visible)?;
            let local_index_query = index_query.as_ref().map(|tensor| self.slice_token_rows(tensor, offset, rows)).transpose()?;
            let local_index_head_weights = index_head_weights.as_ref().map(|tensor| self.slice_token_rows(tensor, offset, rows)).transpose()?;
            let selection = select_history(self, segment.storage, local_index_query.as_ref(), local_index_head_weights.as_ref(), spec, &visible, &visible_buffer)?;
            segment.storage.quantize_batch(self, &local_key, &local_value)?;
            segment.storage.record_transaction_attention(segment.positions, &local_key, &local_value)?;
            prepared.push(Prepared {
                compressed_key: segment.storage.compressed_key.clone(),
                compressed_key_scales: segment.storage.compressed_key_scales.clone(),
                compressed_value: segment.storage.compressed_value.clone(),
                compressed_value_scales: segment.storage.compressed_value_scales.clone(),
                visible: visible_buffer,
                selection,
                recent_key: segment.storage.recent_key.clone(),
                recent_key_scales: segment.storage.recent_key_scales.clone(),
                recent_value: segment.storage.recent_value.clone(),
                recent_value_scales: segment.storage.recent_value_scales.clone(),
                batch_key: segment.storage.batch_key.clone().ok_or_else(|| compute("V4 ROCm segmented CSA batch key 缺失"))?,
                batch_key_scales: segment.storage.batch_key_scales.clone().ok_or_else(|| compute("V4 ROCm segmented CSA batch key scales 缺失"))?,
                batch_value: segment.storage.batch_value.clone().ok_or_else(|| compute("V4 ROCm segmented CSA batch value 缺失"))?,
                batch_value_scales: segment.storage.batch_value_scales.clone().ok_or_else(|| compute("V4 ROCm segmented CSA batch value scales 缺失"))?,
                row_start: offset,
                rows,
                recent_start: segment.storage.recent_start,
                recent_len: segment.storage.recent_len,
                recent_first_position: segment.storage.recent_first_position,
                position_start: segment.positions[0],
                causal_batch: segment.causal_batch,
                compressed_capacity: segment.storage.compressed_capacity,
                q8_group_size: segment.storage.q8_group_size,
            });
            offset += rows;
        }
        let query_device = query.device.as_deref().ok_or_else(|| compute("V4 ROCm segmented CSA query 缺少 device buffer"))?;
        let descriptors = prepared
            .iter()
            .map(|segment| ops::hip::CsaAttentionQ8Segment {
                compressed_key: &segment.compressed_key,
                compressed_key_scales: &segment.compressed_key_scales,
                compressed_value: &segment.compressed_value,
                compressed_value_scales: &segment.compressed_value_scales,
                visible_compressed: &segment.visible,
                selection: segment.selection.as_ref().map(|(buffer, top_k)| (buffer.as_ref(), *top_k)),
                recent_key: &segment.recent_key,
                recent_key_scales: &segment.recent_key_scales,
                recent_value: &segment.recent_value,
                recent_value_scales: &segment.recent_value_scales,
                batch_key: &segment.batch_key,
                batch_key_scales: &segment.batch_key_scales,
                batch_value: &segment.batch_value,
                batch_value_scales: &segment.batch_value_scales,
                row_start: segment.row_start,
                query_rows: segment.rows,
                recent_start: segment.recent_start,
                recent_len: segment.recent_len,
                recent_first_position: segment.recent_first_position,
                position_start: segment.position_start,
                causal_batch: segment.causal_batch,
                compressed_capacity: segment.compressed_capacity,
            })
            .collect::<Vec<_>>();
        let q8_group_size = prepared.first().expect("segments 已非空").q8_group_size;
        if prepared.iter().any(|segment| segment.q8_group_size != q8_group_size) {
            return Err(compute("V4 ROCm segmented CSA Q8 group 不一致"));
        }
        let output = ops::hip::try_csa_attention_q8_segmented(self.device_id, query_device, &descriptors, sink, total_rows, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.window_size, q8_group_size).map_err(compute)?;
        for segment in segments {
            segment.storage.append_recent(segment.positions)?;
        }
        Ok(device_tensor_f32(output, total_rows, query.cols))
    }
}

fn clone_required_buffer(device_id: i32, source: &Arc<ops::hip::DeviceBuffer>) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
    let output = Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, source.bytes()).map_err(compute)?);
    output.copy_from_device(0, source, 0, source.bytes()).map_err(compute)?;
    Ok(output)
}

fn copy_required_buffer(device_id: i32, source: &Arc<ops::hip::DeviceBuffer>, destination: &mut Arc<ops::hip::DeviceBuffer>) -> Result<(), BackendError> {
    if destination.bytes() != source.bytes() {
        *destination = Arc::new(ops::hip::DeviceBuffer::allocate_cache(device_id, source.bytes()).map_err(compute)?);
    }
    destination.copy_from_device(0, source, 0, source.bytes()).map_err(compute)
}

fn copy_optional_buffer(device_id: i32, source: Option<&Arc<ops::hip::DeviceBuffer>>, mut destination: Option<Arc<ops::hip::DeviceBuffer>>) -> Result<Option<Arc<ops::hip::DeviceBuffer>>, BackendError> {
    let Some(source) = source else { return Ok(None) };
    if let Some(buffer) = destination.as_mut() {
        copy_required_buffer(device_id, source, buffer)?;
        Ok(destination)
    } else {
        clone_required_buffer(device_id, source).map(Some)
    }
}

fn select_history(
    context: &RocmContext,
    storage: &RocmCompressedKvStorage,
    index_query: Option<&RocmTensor>,
    index_head_weights: Option<&RocmTensor>,
    spec: &CompressedSparseAttentionSpec,
    visible: &[u32],
    visible_buffer: &ops::hip::DeviceBuffer,
) -> Result<Option<(Arc<ops::hip::DeviceBuffer>, usize)>, BackendError> {
    match spec.compression.map(|compression| compression.selection) {
        Some(CompressedSelection::LearnedIndexer(indexer)) => {
            let query = index_query.ok_or_else(|| compute("V4 ROCm CSA 缺少 index query"))?;
            let head_weights = index_head_weights.ok_or_else(|| compute("V4 ROCm CSA 缺少 index head weights"))?;
            if visible.iter().copied().max().unwrap_or(0) as usize <= indexer.top_k {
                return Ok(None);
            }
            let keys = storage.compressed_index_key.as_deref().ok_or_else(|| compute("V4 ROCm compressed 历史缺少 index key"))?;
            let query = f32_tensor(context, query)?;
            let head_weights = f32_tensor(context, head_weights)?;
            if query.cols != indexer.num_heads * indexer.head_dim || head_weights.cols != indexer.num_heads || query.rows != head_weights.rows {
                return Err(compute(format!("V4 ROCm indexer shape 非法: query=[{},{}] weights=[{},{}] heads={} dim={}", query.rows, query.cols, head_weights.rows, head_weights.cols, indexer.num_heads, indexer.head_dim,)));
            }
            let query_device = query.device.as_deref().ok_or_else(|| compute("V4 ROCm index query 缺少 device buffer"))?;
            let weights_device = head_weights.device.as_deref().ok_or_else(|| compute("V4 ROCm index head weights 缺少 device buffer"))?;
            let selection = ops::hip::try_csa_index_select_f32(context.device_id, query_device, keys, weights_device, visible_buffer, query.rows, storage.compressed_positions.len(), indexer.num_heads, indexer.head_dim, indexer.top_k)
                .map(Arc::new)
                .map_err(compute)?;
            Ok(Some((selection, indexer.top_k)))
        }
        Some(CompressedSelection::All) | None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{
        compressed_sparse::{CompressedKvState, GatedPoolState, normalize_rope_compressed_f32},
        dsa::DsaSpec,
        rope::{RopeSpec, RopeTable, RotaryLayout},
    };

    fn context() -> Option<RocmContext> {
        RocmContext::new(0).ok()
    }

    fn tensor(context: &RocmContext, values: &[f32], rows: usize, cols: usize) -> RocmTensor {
        context.tensor_from_f32(values.to_vec(), rows, cols).unwrap()
    }

    fn weight(context: &RocmContext, values: &[f32]) -> RocmWeight {
        context.prepare_f32(values, 1, values.len()).unwrap()
    }

    fn assert_close(context: &RocmContext, actual: &RocmTensor, expected: &[f32]) {
        let actual = context.tensor_to_f32(actual).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() <= expected.abs() * 1.0e-2 + 1.0e-2, "index={index} actual={actual} expected={expected}");
        }
    }

    fn spec(compression: Option<KvCompressionSpec>) -> CompressedSparseAttentionSpec {
        CompressedSparseAttentionSpec {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            q_lora_rank: 2,
            output_groups: 1,
            output_lora_rank: 2,
            window_size: 2,
            rope: RopeSpec::Default { rotary_dim: 2, theta: 10_000.0 },
            compression,
            attention_sink: true,
        }
    }

    #[test]
    fn decode_visible_scalar_buffer_reuses_unchanged_value() {
        let Some(context) = context() else { return };
        let mut storage = context.allocate_compressed_kv(&spec(None)).unwrap();
        let first = storage.visible_buffer(context.device_id, &[7]).unwrap();
        let unchanged = storage.visible_buffer(context.device_id, &[7]).unwrap();
        assert!(Arc::ptr_eq(&first, &unchanged));
        let changed = storage.visible_buffer(context.device_id, &[8]).unwrap();
        assert!(!Arc::ptr_eq(&first, &changed));
    }

    #[test]
    fn decode_batch_workspace_shrinks_after_prefill() {
        let Some(context) = context() else { return };
        let mut storage = context.allocate_compressed_kv(&spec(None)).unwrap();
        let prefill = tensor(&context, &vec![1.0; 128 * 2], 128, 2);
        storage.quantize_batch(&context, &prefill, &prefill).unwrap();
        let prefill_bytes = storage.batch_allocated_bytes();
        assert_eq!(storage.batch_capacity, 128);

        let decode = tensor(&context, &[1.0, 2.0], 1, 2);
        storage.quantize_batch(&context, &decode, &decode).unwrap();
        assert_eq!(storage.batch_capacity, 1);
        assert!(storage.batch_allocated_bytes() < prefill_bytes);
    }

    #[test]
    fn reset_session_does_not_inherit_compressed_high_water() {
        let Some(context) = context() else { return };
        let mut storage = context.allocate_compressed_kv(&spec(None)).unwrap();
        let positions = (0..128).collect::<Vec<_>>();
        let values = tensor(&context, &vec![1.0; 128 * 2], 128, 2);
        storage.append_compressed(&context, &positions, &values, &values, None).unwrap();
        let long_bytes = storage.allocated_bytes();
        assert_eq!(storage.compressed_capacity, 128);

        storage.reset_session();
        let one = tensor(&context, &[1.0, 2.0], 1, 2);
        storage.append_compressed(&context, &[0], &one, &one, None).unwrap();
        assert_eq!(storage.compressed_capacity, 1);
        assert!(storage.allocated_bytes() < long_bytes);
    }

    #[test]
    fn gated_compressor_matches_cpu_across_overlap_calls() {
        let Some(context) = context() else { return };
        let compression = KvCompressionSpec {
            ratio: 4,
            overlap: true,
            selection: CompressedSelection::LearnedIndexer(DsaSpec { num_heads: 1, head_dim: 2, rope_dim: 2, top_k: 1, rotary_layout: RotaryLayout::Interleaved, kpool: 0, always_select_tail: false }),
        };
        let mut storage = context.allocate_compressed_kv(&spec(Some(compression))).unwrap();
        let position = weight(&context, &[0.0; 16]);
        let norm = weight(&context, &[1.0, 1.0]);
        let rope = RopeTable::precompute(16, 2, 10_000.0);
        let mut oracle = GatedPoolState::default();
        let mut begin = 0;
        for rows in [2, 2, 3, 1] {
            let positions = (begin..begin + rows).collect::<Vec<_>>();
            let key = (0..rows).flat_map(|row| [1.0 + (begin + row) as f32, 2.0, 10.0 + (begin + row) as f32, 20.0]).collect::<Vec<_>>();
            let gate = vec![0.0; key.len()];
            let actual =
                context.compress_gated(&mut storage, CompressionStream::Attention, &positions, &tensor(&context, &key, rows, 4), &tensor(&context, &gate, rows, 4), &position, &norm, compression, 2, 2, &rope.cos, &rope.sin, 1.0e-6).unwrap();
            let (visible, pooled) = oracle.push_f32(&positions, &key, &gate, &[0.0; 16], 4, 2, true).unwrap();
            let expected = normalize_rope_compressed_f32(&pooled, &visible, 4, 2, &[1.0, 1.0], 1.0e-6, 2, &rope.cos, &rope.sin).unwrap();
            assert_eq!(actual.visible_positions, visible);
            assert_close(&context, &actual.values, &expected);
            begin += rows;
        }
    }

    #[test]
    fn resident_prefill_and_wrapped_decode_match_cpu() {
        let Some(context) = context() else { return };
        let spec = spec(None);
        let mut storage = context.allocate_compressed_kv(&spec).unwrap();
        let sink = weight(&context, &[-1.0e9]);
        let positions = [0, 1, 2];
        let query = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let key = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let value = [10.0, 0.0, 20.0, 0.0, 0.0, 30.0];
        let actual =
            context.compressed_sparse_prefill(&mut storage, &positions, true, &tensor(&context, &query, 3, 2), &tensor(&context, &key, 3, 2), &tensor(&context, &value, 3, 2), None, None, None, None, None, None, Some(&sink), &spec).unwrap();
        let expected = CompressedKvState::new(2).unwrap().attend_batch_f32(&query, 3, &positions, true, &key, &value, &[], &[], &[], None, None, None, 1, 1, 2, None, Some(&[-1.0e9])).unwrap();
        assert_close(&context, &actual, &expected);

        let actual =
            context.compressed_sparse_decode(&mut storage, 3, &tensor(&context, &[1.0, 0.0], 1, 2), &tensor(&context, &[1.0, 0.0], 1, 2), &tensor(&context, &[40.0, 0.0], 1, 2), None, None, None, None, None, Some(&sink), &spec).unwrap();
        let mut oracle = CompressedKvState::new(2).unwrap();
        oracle.push_recent_batch(&positions, &key, &value, 2).unwrap();
        oracle.push_recent(3, &[1.0, 0.0], &[40.0, 0.0]).unwrap();
        let expected = oracle.attend_f32(&[1.0, 0.0], 1, 1, 2, None, Some(&[-1.0e9])).unwrap();
        assert_close(&context, &actual, &expected);
    }

    #[test]
    fn resident_store_recent_matches_prefill_cache() {
        let Some(context) = context() else { return };
        let spec = spec(None);
        let mut storage = context.allocate_compressed_kv(&spec).unwrap();
        let positions = [0, 1, 2];
        let key = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let value = [10.0, 0.0, 20.0, 0.0, 0.0, 30.0];
        context.compressed_sparse_store_recent(&mut storage, &positions, &tensor(&context, &key, 3, 2), &tensor(&context, &value, 3, 2)).unwrap();
        let sink = weight(&context, &[-1.0e9]);
        let actual =
            context.compressed_sparse_decode(&mut storage, 3, &tensor(&context, &[1.0, 0.0], 1, 2), &tensor(&context, &[1.0, 0.0], 1, 2), &tensor(&context, &[40.0, 0.0], 1, 2), None, None, None, None, None, Some(&sink), &spec).unwrap();
        let mut oracle = CompressedKvState::new(2).unwrap();
        oracle.push_recent_batch(&positions, &key, &value, 2).unwrap();
        oracle.push_recent(3, &[1.0, 0.0], &[40.0, 0.0]).unwrap();
        let expected = oracle.attend_f32(&[1.0, 0.0], 1, 1, 2, None, Some(&[-1.0e9])).unwrap();
        assert_close(&context, &actual, &expected);
    }

    #[test]
    fn deepseek_shape_causal_batch_first_row_matches_decode_bitwise() {
        let Some(context) = context() else { return };
        const HEADS: usize = 64;
        const DIM: usize = 512;
        const WINDOW: usize = 128;
        const HISTORY: usize = 15;
        const BATCH: usize = 6;
        let spec = CompressedSparseAttentionSpec {
            num_heads: HEADS,
            num_kv_heads: 1,
            head_dim: DIM,
            q_lora_rank: 2048,
            output_groups: 32,
            output_lora_rank: 256,
            window_size: WINDOW,
            rope: RopeSpec::Default { rotary_dim: 64, theta: 10_000.0 },
            compression: None,
            attention_sink: true,
        };
        let history_positions = (0..HISTORY).collect::<Vec<_>>();
        let history_key = (0..HISTORY * DIM).map(|index| ((index % 37) as f32 * 0.017 - 0.3).sin()).collect::<Vec<_>>();
        let mut single_storage = context.allocate_compressed_kv(&spec).unwrap();
        let mut batch_storage = context.allocate_compressed_kv(&spec).unwrap();
        for storage in [&mut single_storage, &mut batch_storage] {
            let history = tensor(&context, &history_key, HISTORY, DIM);
            context.compressed_sparse_store_recent(storage, &history_positions, &history, &history).unwrap();
        }
        let query = (0..BATCH * HEADS * DIM).map(|index| ((index % 43) as f32 * 0.013 - 0.2).cos()).collect::<Vec<_>>();
        let key = (0..BATCH * DIM).map(|index| ((index % 31) as f32 * 0.019 + 0.1).sin()).collect::<Vec<_>>();
        let sink = weight(&context, &(0..HEADS).map(|head| -2.0 - head as f32 * 0.01).collect::<Vec<_>>());
        let single_key = tensor(&context, &key[..DIM], 1, DIM);
        let single =
            context.compressed_sparse_prefill(&mut single_storage, &[HISTORY], true, &tensor(&context, &query[..HEADS * DIM], 1, HEADS * DIM), &single_key, &single_key, None, None, None, None, None, None, Some(&sink), &spec).unwrap();
        let positions = (HISTORY..HISTORY + BATCH).collect::<Vec<_>>();
        let batch_key = tensor(&context, &key, BATCH, DIM);
        let batch = context.compressed_sparse_prefill(&mut batch_storage, &positions, true, &tensor(&context, &query, BATCH, HEADS * DIM), &batch_key, &batch_key, None, None, None, None, None, None, Some(&sink), &spec).unwrap();
        let single = context.tensor_to_f32(&single).unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let batch = context.tensor_to_f32(&batch).unwrap()[..HEADS * DIM].iter().copied().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(single, batch);
    }

    #[test]
    fn deepseek_shape_causal_batch_matches_sequential_decode_bitwise() {
        let Some(context) = context() else { return };
        const HEADS: usize = 64;
        const DIM: usize = 512;
        const WINDOW: usize = 128;
        const HISTORY: usize = 15;
        const BATCH: usize = 6;
        let spec = CompressedSparseAttentionSpec {
            num_heads: HEADS,
            num_kv_heads: 1,
            head_dim: DIM,
            q_lora_rank: 2048,
            output_groups: 32,
            output_lora_rank: 256,
            window_size: WINDOW,
            rope: RopeSpec::Default { rotary_dim: 64, theta: 10_000.0 },
            compression: None,
            attention_sink: true,
        };
        let history_positions = (0..HISTORY).collect::<Vec<_>>();
        let history_key = (0..HISTORY * DIM).map(|index| ((index % 37) as f32 * 0.017 - 0.3).sin()).collect::<Vec<_>>();
        let mut single_storage = context.allocate_compressed_kv(&spec).unwrap();
        let mut batch_storage = context.allocate_compressed_kv(&spec).unwrap();
        for storage in [&mut single_storage, &mut batch_storage] {
            let history = tensor(&context, &history_key, HISTORY, DIM);
            context.compressed_sparse_store_recent(storage, &history_positions, &history, &history).unwrap();
        }
        let query = (0..BATCH * HEADS * DIM).map(|index| ((index % 43) as f32 * 0.013 - 0.2).cos()).collect::<Vec<_>>();
        let key = (0..BATCH * DIM).map(|index| ((index % 31) as f32 * 0.019 + 0.1).sin()).collect::<Vec<_>>();
        let sink = weight(&context, &(0..HEADS).map(|head| -2.0 - head as f32 * 0.01).collect::<Vec<_>>());
        let mut sequential = Vec::with_capacity(BATCH * HEADS * DIM);
        for row in 0..BATCH {
            let query_start = row * HEADS * DIM;
            let key_start = row * DIM;
            let row_key = tensor(&context, &key[key_start..key_start + DIM], 1, DIM);
            let output = context
                .compressed_sparse_prefill(
                    &mut single_storage,
                    &[HISTORY + row],
                    true,
                    &tensor(&context, &query[query_start..query_start + HEADS * DIM], 1, HEADS * DIM),
                    &row_key,
                    &row_key,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(&sink),
                    &spec,
                )
                .unwrap();
            sequential.extend(context.tensor_to_f32(&output).unwrap().into_iter().map(f32::to_bits));
        }
        let positions = (HISTORY..HISTORY + BATCH).collect::<Vec<_>>();
        let batch_key = tensor(&context, &key, BATCH, DIM);
        let batch = context.compressed_sparse_prefill(&mut batch_storage, &positions, true, &tensor(&context, &query, BATCH, HEADS * DIM), &batch_key, &batch_key, None, None, None, None, None, None, Some(&sink), &spec).unwrap();
        let batch = context.tensor_to_f32(&batch).unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
        if let Some(index) = sequential.iter().zip(&batch).position(|(sequential, batch)| sequential != batch) {
            panic!("CSA row={} index={} 不等价: sequential={} batch={}", index / (HEADS * DIM), index % (HEADS * DIM), f32::from_bits(sequential[index]), f32::from_bits(batch[index]));
        }
    }

    #[test]
    fn learned_indexer_and_compressed_attention_match_cpu() {
        let Some(context) = context() else { return };
        let indexer = DsaSpec { num_heads: 1, head_dim: 2, rope_dim: 0, top_k: 1, rotary_layout: RotaryLayout::Interleaved, kpool: 0, always_select_tail: false };
        let compression = KvCompressionSpec { ratio: 4, overlap: true, selection: CompressedSelection::LearnedIndexer(indexer) };
        let spec = spec(Some(compression));
        let mut storage = context.allocate_compressed_kv(&spec).unwrap();
        let positions = [2];
        let query = [1.0, 0.0];
        let key = [1.0, 0.0];
        let value = [5.0, 0.0];
        let compressed_positions = [0, 1];
        let compressed_key = [1.0, 0.0, 1.0, 0.0];
        let compressed_value = [10.0, 0.0, 30.0, 0.0];
        let index_keys = [2.0, 0.0, 0.0, 2.0];
        let index_query = [0.0, 1.0];
        let head_weights = [1.0];
        let sink = weight(&context, &[-1.0e9]);
        let actual = context
            .compressed_sparse_prefill(
                &mut storage,
                &positions,
                true,
                &tensor(&context, &query, 1, 2),
                &tensor(&context, &key, 1, 2),
                &tensor(&context, &value, 1, 2),
                Some(&compressed_positions),
                Some(&tensor(&context, &compressed_key, 2, 2)),
                Some(&tensor(&context, &compressed_value, 2, 2)),
                Some(&tensor(&context, &index_keys, 2, 2)),
                Some(&tensor(&context, &index_query, 1, 2)),
                Some(&tensor(&context, &head_weights, 1, 1)),
                Some(&sink),
                &spec,
            )
            .unwrap();
        let expected = CompressedKvState::new(2)
            .unwrap()
            .attend_batch_f32(&query, 1, &positions, true, &key, &value, &compressed_positions, &compressed_key, &compressed_value, Some(&index_keys), Some(&index_query), Some(&head_weights), 1, 1, 2, Some(indexer), Some(&[-1.0e9]))
            .unwrap();
        assert_close(&context, &actual, &expected);
    }

    /// decode 小批次 split-KV 路径的定向 oracle:16 head/单 KV head、
    /// head_dim=160(>128 避开 WMMA fast path)触发 tiled/split 分派,带
    /// selection、recent ring、batch 行与 sink,覆盖 1 行与 5 行两种批形。
    #[test]
    fn decode_split_kv_attention_matches_cpu() {
        let Some(context) = context() else { return };
        const HEADS: usize = 16;
        const DIM: usize = 160;
        const WINDOW: usize = 64;
        const COMPRESSED_ROWS: usize = 40;
        let indexer = DsaSpec { num_heads: 4, head_dim: 32, rope_dim: 0, top_k: 16, rotary_layout: RotaryLayout::Interleaved, kpool: 0, always_select_tail: false };
        let compression = KvCompressionSpec { ratio: 4, overlap: true, selection: CompressedSelection::LearnedIndexer(indexer) };
        let spec = CompressedSparseAttentionSpec {
            num_heads: HEADS,
            num_kv_heads: 1,
            head_dim: DIM,
            q_lora_rank: 32,
            output_groups: 1,
            output_lora_rank: 32,
            window_size: WINDOW,
            rope: RopeSpec::Default { rotary_dim: 32, theta: 10_000.0 },
            compression: Some(compression),
            attention_sink: true,
        };
        let sink_values = (0..HEADS).map(|head| -1.0e9 - head as f32).collect::<Vec<_>>();
        let sink = weight(&context, &sink_values);
        let wave = |offset: usize, len: usize| (0..len).map(|index| ((offset + index) as f32 * 0.037).sin()).collect::<Vec<_>>();
        let compressed_key = (0..COMPRESSED_ROWS * DIM).map(|index| ((index % 13) as f32 * 0.11 - 0.7).sin()).collect::<Vec<_>>();
        let compressed_value = (0..COMPRESSED_ROWS * DIM).map(|index| ((index % 17) as f32 * 0.09 + 0.4).cos()).collect::<Vec<_>>();
        let index_keys = (0..COMPRESSED_ROWS * 32).map(|index| ((index % 19) as f32 * 0.05 - 0.3).sin()).collect::<Vec<_>>();
        let compressed_positions: Vec<usize> = (0..COMPRESSED_ROWS).map(|row| row * 4 + 3).collect();
        let sink_values = (0..HEADS).map(|head| -1.0e9 - head as f32).collect::<Vec<_>>();
        for rows in [1usize, 5] {
            let mut storage = context.allocate_compressed_kv(&spec).unwrap();
            let positions: Vec<usize> = (COMPRESSED_ROWS * 4..COMPRESSED_ROWS * 4 + rows).collect();
            let query = (0..rows * HEADS * DIM).map(|index| ((index % 23) as f32 * 0.013 - 0.6).sin()).collect::<Vec<_>>();
            let key = wave(7, rows * DIM);
            let value = wave(11, rows * DIM);
            let index_query = (0..rows * 4 * 32).map(|index| ((index % 29) as f32 * 0.017 - 0.2).cos()).collect::<Vec<_>>();
            let head_weights = (0..rows * 4).map(|index| ((index % 7) as f32) * 0.1 + 0.2).collect::<Vec<_>>();
            let actual = context
                .compressed_sparse_prefill(
                    &mut storage,
                    &positions,
                    true,
                    &tensor(&context, &query, rows, HEADS * DIM),
                    &tensor(&context, &key, rows, DIM),
                    &tensor(&context, &value, rows, DIM),
                    Some(&compressed_positions),
                    Some(&tensor(&context, &compressed_key, COMPRESSED_ROWS, DIM)),
                    Some(&tensor(&context, &compressed_value, COMPRESSED_ROWS, DIM)),
                    Some(&tensor(&context, &index_keys, COMPRESSED_ROWS, 32)),
                    Some(&tensor(&context, &index_query, rows, 4 * 32)),
                    Some(&tensor(&context, &head_weights, rows, 4)),
                    Some(&sink),
                    &spec,
                )
                .unwrap();
            let expected = CompressedKvState::new(WINDOW)
                .unwrap()
                .attend_batch_f32(
                    &query,
                    rows,
                    &positions,
                    true,
                    &key,
                    &value,
                    &compressed_positions,
                    &compressed_key,
                    &compressed_value,
                    Some(&index_keys),
                    Some(&index_query),
                    Some(&head_weights),
                    HEADS,
                    1,
                    DIM,
                    Some(indexer),
                    Some(&sink_values),
                )
                .unwrap();
            assert_close(&context, &actual, &expected);
        }
    }

    /// 真实 DeepSeek-V4 decode 形状的 split-KV 回归:64 头/512 维/
    /// top_k=512 selection/5 行 verify 批,复现端到端挂起用。
    #[test]
    fn decode_split_kv_attention_matches_cpu_deepseek_shape() {
        let Some(context) = context() else { return };
        const HEADS: usize = 64;
        const DIM: usize = 512;
        const WINDOW: usize = 128;
        const COMPRESSED_ROWS: usize = 520;
        let indexer = DsaSpec { num_heads: 64, head_dim: 128, rope_dim: 0, top_k: 512, rotary_layout: RotaryLayout::Interleaved, kpool: 0, always_select_tail: false };
        let compression = KvCompressionSpec { ratio: 4, overlap: true, selection: CompressedSelection::LearnedIndexer(indexer) };
        let spec = CompressedSparseAttentionSpec {
            num_heads: HEADS,
            num_kv_heads: 1,
            head_dim: DIM,
            q_lora_rank: 32,
            output_groups: 1,
            output_lora_rank: 32,
            window_size: WINDOW,
            rope: RopeSpec::Default { rotary_dim: 64, theta: 10_000.0 },
            compression: Some(compression),
            attention_sink: true,
        };
        let rows = 5usize;
        let positions: Vec<usize> = (COMPRESSED_ROWS * 4..COMPRESSED_ROWS * 4 + rows).collect();
        let query = (0..rows * HEADS * DIM).map(|index| ((index % 23) as f32 * 0.013 - 0.6).sin()).collect::<Vec<_>>();
        let key = (0..rows * DIM).map(|index| ((index % 13) as f32 * 0.017 - 0.5).sin()).collect::<Vec<_>>();
        let value = (0..rows * DIM).map(|index| ((index % 11) as f32 * 0.019 + 0.3).cos()).collect::<Vec<_>>();
        let compressed_key = (0..COMPRESSED_ROWS * DIM).map(|index| ((index % 13) as f32 * 0.11 - 0.7).sin()).collect::<Vec<_>>();
        let compressed_value = (0..COMPRESSED_ROWS * DIM).map(|index| ((index % 17) as f32 * 0.09 + 0.4).cos()).collect::<Vec<_>>();
        let index_keys = (0..COMPRESSED_ROWS * 128).map(|index| ((index % 19) as f32 * 0.05 - 0.3).sin()).collect::<Vec<_>>();
        let index_query = (0..rows * 64 * 128).map(|index| ((index % 29) as f32 * 0.017 - 0.2).cos()).collect::<Vec<_>>();
        let head_weights = (0..rows * 64).map(|index| ((index % 7) as f32) * 0.1 + 0.2).collect::<Vec<_>>();
        let compressed_positions: Vec<usize> = (0..COMPRESSED_ROWS).map(|row| row * 4 + 3).collect();
        let sink_values = (0..HEADS).map(|head| -1.0e9 - head as f32).collect::<Vec<_>>();
        let mut storage = context.allocate_compressed_kv(&spec).unwrap();
        let actual = context
            .compressed_sparse_prefill(
                &mut storage,
                &positions,
                true,
                &tensor(&context, &query, rows, HEADS * DIM),
                &tensor(&context, &key, rows, DIM),
                &tensor(&context, &value, rows, DIM),
                Some(&compressed_positions),
                Some(&tensor(&context, &compressed_key, COMPRESSED_ROWS, DIM)),
                Some(&tensor(&context, &compressed_value, COMPRESSED_ROWS, DIM)),
                Some(&tensor(&context, &index_keys, COMPRESSED_ROWS, 128)),
                Some(&tensor(&context, &index_query, rows, 64 * 128)),
                Some(&tensor(&context, &head_weights, rows, 64)),
                Some(&weight(&context, &sink_values)),
                &spec,
            )
            .unwrap();
        let expected = CompressedKvState::new(WINDOW)
            .unwrap()
            .attend_batch_f32(
                &query,
                rows,
                &positions,
                true,
                &key,
                &value,
                &compressed_positions,
                &compressed_key,
                &compressed_value,
                Some(&index_keys),
                Some(&index_query),
                Some(&head_weights),
                HEADS,
                1,
                DIM,
                Some(indexer),
                Some(&sink_values),
            )
            .unwrap();
        assert_close(&context, &actual, &expected);
    }
}
