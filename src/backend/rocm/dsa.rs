use std::sync::Arc;

use crate::backend::cpu::CpuDsaState;
use crate::backend::{BackendError, compute_error};
use crate::kernel::rocm as ops;

use super::{ROCM_KV_BLOCK_SIZE, RocmBlockTable, RocmContext, RocmTensor, committed_cache_rows, grow_cache_buffer, upload_cache_buffer};

struct RocmPagedDsaLayer {
    keys: Arc<ops::hip::DeviceBuffer>,
    scales: Arc<ops::hip::DeviceBuffer>,
    /// 仅 profile shadow 使用；不序列化，也不参与生产 selection。
    hadamard_shadow_keys: Option<Arc<ops::hip::DeviceBuffer>>,
    hadamard_shadow_scales: Option<Arc<ops::hip::DeviceBuffer>>,
    hadamard: bool,
    gates: Option<Arc<ops::hip::DeviceBuffer>>,
    pooled_keys: Option<Arc<ops::hip::DeviceBuffer>>,
    pooled_scales: Option<Arc<ops::hip::DeviceBuffer>>,
    pooled_rows: usize,
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
    top_k: usize,
    block_table: RocmBlockTable,
    pool_block_table: RocmBlockTable,
    kpool_apes: Vec<Option<Arc<ops::hip::DeviceBuffer>>>,
    kpool: usize,
    selection: Option<Arc<ops::hip::DeviceBuffer>>,
    selection_rows: usize,
    selection_start: usize,
    selection_width: usize,
}

pub struct RocmDsaSelection {
    buffer: Arc<ops::hip::DeviceBuffer>,
    rows: usize,
    start: usize,
    width: usize,
}

const DSA_SHADOW_GUARDS: [usize; 9] = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

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

impl RocmDsaSelection {
    pub(crate) fn move_to_stable_deferred(self) -> Result<Self, BackendError> {
        // DSA selection 由显式 reusable pool 分配时已可跨线程/P2P 持有；
        // 只有 stream-ordered allocation 才需要先复制到稳定存储。
        if !self.buffer.is_async_allocated() {
            return Ok(self);
        }
        let buffer = self.buffer.copy_to_stable_deferred().map_err(compute_error)?;
        Ok(Self { buffer: Arc::new(buffer), rows: self.rows, start: self.start, width: self.width })
    }

    pub(crate) fn move_to_device_ordered(self, device_id: i32) -> Result<Self, BackendError> {
        if self.buffer.device_id() == device_id {
            return Ok(self);
        }
        let buffer = self.buffer.copy_stable_to_device_ordered_async(device_id).map_err(compute_error)?;
        Ok(Self { buffer: Arc::new(buffer), rows: self.rows, start: self.start, width: self.width })
    }

    pub fn to_host(&self) -> Result<Vec<u32>, BackendError> {
        let bytes = self.buffer.bytes();
        if bytes == 0 || !bytes.is_multiple_of(std::mem::size_of::<u32>()) {
            return Err(compute_error(format!("ROCm DSA selection bytes={bytes} 无效")));
        }
        let mut values = vec![0_u32; bytes / std::mem::size_of::<u32>()];
        self.buffer.copy_to_host(unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), bytes) }).map_err(compute_error)?;
        Ok(values)
    }

    pub fn to_host_completed(&self) -> Result<Vec<u32>, BackendError> {
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
        Ok(Self { buffer: Arc::new(buffer), rows, start, width: values.len() / rows })
    }
}

impl RocmDsaState {
    pub fn new(layer_count: usize, capacity: usize, head_dim: usize, top_k: usize) -> Result<Self, String> {
        let key_group_size = [128, 64, 32, 16].into_iter().find(|group| head_dim.is_multiple_of(*group)).ok_or_else(|| format!("ROCm DSA head_dim={head_dim} 不支持 Q8 group"))?;
        let hadamard_i8 = ops::hip::options().dsa_hadamard_i8 && head_dim == 128 && key_group_size == head_dim;
        let hadamard_shadow_samples = if !hadamard_i8 && head_dim == 128 && key_group_size == head_dim { ops::hip::options().dsa_hadamard_shadow_samples } else { 0 };
        Ok(Self {
            cpu: CpuDsaState::new(layer_count, capacity, head_dim, top_k)?,
            layers: (0..layer_count).map(|_| None).collect(),
            capacity,
            head_dim,
            key_group_size,
            hadamard_i8,
            hadamard_shadow_samples,
            hadamard_shadow_counts: vec![0; layer_count],
            top_k,
            block_table: RocmBlockTable::new(),
            pool_block_table: RocmBlockTable::new(),
            kpool_apes: (0..layer_count).map(|_| None).collect(),
            kpool: 0,
            selection: None,
            selection_rows: 0,
            selection_start: 0,
            selection_width: top_k,
        })
    }

    pub(crate) fn prepare_block_table(&mut self, context: &RocmContext) -> Result<(), BackendError> {
        self.block_table.get("DSA", self.capacity, context.device_id, self.capacity)?;
        if self.kpool > 0 {
            let pool_capacity = self.capacity / self.kpool;
            self.pool_block_table.get("DSA kpool", pool_capacity, context.device_id, pool_capacity)?;
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
            if self.kpool > 0 {
                cached.pooled_rows = rows / self.kpool;
            }
        }
        self.selection = None;
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
            if self.kpool > 0 {
                cached.pooled_rows = rows / self.kpool;
            }
        }
        self.selection = None;
        self.selection_rows = 0;
        self.selection_start = 0;
        self.selection_width = self.top_k;
        Ok(())
    }

    /// 清空 selection(kpool 等场景退化为全量注意力)。
    pub fn invalidate_selection(&mut self) {
        self.selection = None;
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
                hadamard_shadow_keys: None,
                hadamard_shadow_scales: None,
                hadamard: self.hadamard_i8,
                gates: None,
                pooled_keys: None,
                pooled_scales: None,
                pooled_rows: 0,
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

    pub(super) fn select(&mut self, context: &RocmContext, layer: usize, query: &RocmTensor, head_weights: &RocmTensor) -> Result<(), BackendError> {
        let context_rows = self.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} ROCm DSA cache 尚未初始化")))?.rows;
        if context_rows <= self.top_k {
            self.selection = None;
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
            ROCM_KV_BLOCK_SIZE,
        )
        .map_err(compute_error)?;
        let sample_shadow = !hadamard && self.kpool == 0 && query.rows == 1 && self.hadamard_shadow_counts.get(layer).copied().unwrap_or_default() < self.hadamard_shadow_samples;
        if sample_shadow {
            // 必须在 coarse 调用覆盖同一 score workspace 前读取 exact 分数。
            let exact_scores = ops::hip::try_download_last_dsa_score_keys(context.device_id, context_rows).map_err(compute_error)?;
            let mut exact_selection = vec![0_u32; self.top_k];
            exact.copy_to_host(unsafe { std::slice::from_raw_parts_mut(exact_selection.as_mut_ptr().cast(), self.top_k * std::mem::size_of::<u32>()) }).map_err(compute_error)?;
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
        self.selection = Some(Arc::new(exact));
        self.selection_rows = query.rows;
        self.selection_start = context_rows - query.rows;
        self.selection_width = self.top_k;
        Ok(())
    }

    pub(super) fn device_selection(&self, rows: usize, query_start: usize) -> Option<&ops::hip::DeviceBuffer> {
        (self.selection_rows == rows && self.selection_start == query_start).then_some(self.selection.as_deref()).flatten()
    }

    pub(super) fn selection_width(&self) -> usize {
        self.selection_width
    }

    pub(crate) fn export_selection(&self) -> Option<RocmDsaSelection> {
        Some(RocmDsaSelection { buffer: self.selection.as_ref()?.clone(), rows: self.selection_rows, start: self.selection_start, width: self.selection_width })
    }

    pub(crate) fn import_selection(&mut self, context: &RocmContext, selection: Option<RocmDsaSelection>) -> Result<(), BackendError> {
        let Some(selection) = selection else {
            self.selection = None;
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
                hadamard_shadow_keys: None,
                hadamard_shadow_scales: None,
                hadamard: record.hadamard,
                gates: None,
                pooled_keys: None,
                pooled_scales: None,
                pooled_rows: 0,
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
            })
            .sum()
    }
}
