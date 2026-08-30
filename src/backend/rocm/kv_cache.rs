use std::sync::Arc;

use crate::backend::cpu::CpuKvCache;
use crate::backend::{BackendError, compute_error};
use crate::kernel::rocm as ops;

use super::{ROCM_KV_BLOCK_SIZE, RocmBlockTable, RocmContext, RocmTensor, committed_cache_rows, grow_cache_buffer, upload_cache_buffer};

pub(super) struct RocmPagedMlaLayer {
    pub(super) latent: Arc<ops::hip::DeviceBuffer>,
    pub(super) latent_scales: Option<Arc<ops::hip::DeviceBuffer>>,
    pub(super) latent_group_size: usize,
    pub(super) rope: Arc<ops::hip::DeviceBuffer>,
    pub(super) rows: usize,
    pub(super) latent_cols: usize,
    pub(super) rope_cols: usize,
    committed_rows: usize,
}

pub struct RocmKvCache {
    cpu: CpuKvCache,
    pub(super) paged_layers: Vec<Option<RocmPagedMlaLayer>>,
    /// GQA 层的常驻设备 K/V（f32，行主序 [rows][kv_heads*head_dim]）。
    /// full-attention prefill/decode 只读设备缓存；需要 CPU fallback 的窗口路径才双写 host。
    pub(super) gqa_layers: Vec<Option<RocmGqaLayer>>,
    capacity: usize,
    pub(super) block_table: RocmBlockTable,
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
        Self { cpu: CpuKvCache::new(layer_count), paged_layers: (0..layer_count).map(|_| None).collect(), gqa_layers: (0..layer_count).map(|_| None).collect(), capacity, block_table: RocmBlockTable::new() }
    }

    pub(crate) fn prepare_block_table(&mut self, context: &RocmContext) -> Result<(), BackendError> {
        self.block_table.get("KV", self.capacity, context.device_id, self.capacity).map(|_| ())
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
            cached.rows = rows;
        }
        for (layer, cached) in self.gqa_layers.iter_mut().enumerate().filter_map(|(layer, slot)| slot.as_mut().map(|cached| (layer, cached))) {
            if rows > cached.rows {
                return Err(compute_error(format!("L{layer} ROCm GQA truncate rows={rows} 超过当前长度 {}", cached.rows)));
            }
            cached.rows = rows;
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
            cached.rows = rows;
        }
        let gqa = self.gqa_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if let Some(cached) = gqa {
            if rows > cached.rows {
                return Err(compute_error(format!("L{layer} ROCm GQA truncate rows={rows} 超过当前长度 {}", cached.rows)));
            }
            cached.rows = rows;
        }
        Ok(())
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

    #[allow(clippy::type_complexity)]
    fn append_mla_inner(&mut self, context: &RocmContext, layer: usize, latent: &RocmTensor, rope: &RocmTensor, rope_rotation: Option<(usize, usize, crate::attention::rope::RotaryLayout, &[f32], &[f32])>) -> Result<(), BackendError> {
        if latent.rows != rope.rows || latent.rows == 0 {
            return Err(compute_error(format!("L{layer} ROCm paged MLA append 行数非法")));
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
                latent: Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, latent_bytes).map_err(compute_error)?),
                latent_scales: if latent_group_size == 0 { None } else { Some(Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, latent_scale_bytes).map_err(compute_error)?)) },
                latent_group_size,
                rope: Arc::new(ops::hip::DeviceBuffer::allocate(context.device_id, rope_bytes).map_err(compute_error)?),
                rows: 0,
                latent_cols: latent.cols,
                rope_cols: rope.cols,
                committed_rows: next_committed,
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
        cached.rows += latent.rows;
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

impl RocmKvCache {
    /// 把所有已填充层 D2H 到 host blob。空层（slot None）对应 `None`。
    pub fn download_layers(&self) -> Result<Vec<Option<MlaLayerSerde>>, BackendError> {
        self.paged_layers
            .iter()
            .map(|slot| -> Result<Option<MlaLayerSerde>, BackendError> {
                let Some(cached) = slot.as_ref() else { return Ok(None) };
                let element_bytes = if cached.latent_group_size == 0 { 2 } else { 1 };
                let latent_bytes = cached.rows.checked_mul(cached.latent_cols).and_then(|n| n.checked_mul(element_bytes)).ok_or_else(|| compute_error("ROCm MLA latent 字节溢出"))?;
                let mut latent = vec![0u8; latent_bytes];
                cached.latent.copy_to_host(&mut latent).map_err(compute_error)?;
                let latent_scales = if let Some(scales) = &cached.latent_scales {
                    let scale_bytes = cached.rows.checked_mul(cached.latent_cols / cached.latent_group_size).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA scales 字节溢出"))?;
                    let mut buffer = vec![0u8; scale_bytes];
                    scales.copy_to_host(&mut buffer).map_err(compute_error)?;
                    Some(buffer)
                } else {
                    None
                };
                let rope_bytes = cached.rows.checked_mul(cached.rope_cols).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA rope 字节溢出"))?;
                let mut rope = vec![0u8; rope_bytes];
                cached.rope.copy_to_host(&mut rope).map_err(compute_error)?;
                Ok(Some(MlaLayerSerde { rows: cached.rows, latent_cols: cached.latent_cols, rope_cols: cached.rope_cols, latent_group_size: cached.latent_group_size, latent, latent_scales, rope }))
            })
            .collect()
    }

    /// 从 host blob H2D 重建每层（allocate + upload）。空 slot 跳过。
    /// 调用方需先用 `with_capacity` 构造空 cache，再 `upload_layers`。
    pub fn upload_layers(&mut self, context: &RocmContext, layers: &[Option<MlaLayerSerde>], reserved_rows: usize) -> Result<(), BackendError> {
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
            let rope_used = record.rows.checked_mul(record.rope_cols).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA restore rope 大小溢出"))?;
            if record.rope.len() != rope_used {
                return Err(compute_error(format!("ROCm MLA restore L{index} rope={}，期望 {rope_used}", record.rope.len())));
            }
            let rope_capacity = committed_rows.checked_mul(record.rope_cols).and_then(|n| n.checked_mul(2)).ok_or_else(|| compute_error("ROCm MLA restore rope 容量溢出"))?;
            let rope = upload_cache_buffer(device_id, &record.rope, rope_capacity)?;
            self.paged_layers[index] = Some(RocmPagedMlaLayer { latent, latent_scales, latent_group_size: record.latent_group_size, rope, rows: record.rows, latent_cols: record.latent_cols, rope_cols: record.rope_cols, committed_rows });
        }
        Ok(())
    }

    /// 所有层 device buffer 实际分配字节（供 `CacheInfo.bytes` 上报）。
    pub fn allocated_bytes(&self) -> u64 {
        self.paged_layers
            .iter()
            .filter_map(|slot| slot.as_ref())
            .map(|cached| cached.latent.bytes() as u64 + cached.latent_scales.as_ref().map_or(0, |scales| scales.bytes() as u64) + cached.rope.bytes() as u64)
            .sum::<u64>()
            .saturating_add(self.gqa_layers.iter().filter_map(|slot| slot.as_ref()).map(|cached| (cached.key.bytes() + cached.value.bytes()) as u64).sum())
    }
}
