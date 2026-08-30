//! KV cache 领域层。描述 cache 的形态、逻辑布局和有效长度状态,
//! 不描述具体设备上的 buffer 分配、量化 kernel 与同步。
//!
//! 跨架构:用 [`KvCacheSpec`] 描述层形态一致的 MLA/GQA；用 [`HybridGqaCacheLayout`]
//! 描述每层 head 形态和窗口都可变化的 hybrid GQA。具体存储、dtype 与同步由 backend 决定。
//!
//! 设计哲学(参考 colibri `c/glm.c` 的 `KVState` 与父项目 `kv_cache`):
//! - **MLA**:每 token 存 norm 后的 latent `[kv_lora_rank]` + rope `[qk_rope_head_dim]`。
//!   latent INT8 per-group 对称量化(省 ~43% 内存,精度损失 <1%),rope 保留 f16
//!   (只 64 维,量化收益小且参与点积)。attention 时反量化 latent → kv_b_proj 重建。
//! - **GQA**:每 token 存 K `[num_kv_heads × head_dim]` + V `[num_kv_heads × head_dim]`。
//!   本次只建形态,attention 消费留阶段 4。

use std::ops::Range;

pub mod fjall;
pub mod terminal_cache;

pub use fjall::{FjallBlob, FjallCacheStore, FjallChunkKind, FjallValueStore};

use crate::attention::{
    AttentionSpec,
    gqa::GqaSpec,
    gqa::{HybridGqaLayerSpec, HybridGqaSpec},
    mla::MlaSpec,
};

/// INT8 per-group 对称量化的默认 group_size(与父项目 `kv_cache::compression` 一致)。
pub const DEFAULT_GROUP_SIZE: usize = 64;
/// 量化位数。当前固定 8(INT8);4 留给以后(父项目支持 i4)。
pub const QUANT_BITS: u32 = 8;

/// 模型逻辑层到紧凑 cache 槽的映射。只描述哪些层拥有 cache，不绑定模型或设备。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheLayerMap {
    logical_to_slot: Vec<Option<usize>>,
    slot_count: usize,
}

impl KvCacheLayerMap {
    pub fn dense(layer_count: usize) -> Result<Self, String> {
        Self::from_cached_layers(layer_count, 0..layer_count)
    }

    pub fn from_cached_layers(logical_layer_count: usize, layers: impl IntoIterator<Item = usize>) -> Result<Self, String> {
        if logical_layer_count == 0 {
            return Err("KV cache 逻辑层数不能为 0".to_owned());
        }
        let mut logical_to_slot = vec![None; logical_layer_count];
        let mut slot_count = 0;
        for layer in layers {
            if layer >= logical_layer_count {
                return Err(format!("KV cache 逻辑层 {layer} 越界(共 {logical_layer_count} 层)"));
            }
            if logical_to_slot[layer].is_some() {
                return Err(format!("KV cache 逻辑层 {layer} 重复映射"));
            }
            logical_to_slot[layer] = Some(slot_count);
            slot_count += 1;
        }
        if slot_count == 0 {
            return Err("KV cache 至少需要一个物理槽".to_owned());
        }
        Ok(Self { logical_to_slot, slot_count })
    }

    pub fn logical_layer_count(&self) -> usize {
        self.logical_to_slot.len()
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub fn contains(&self, logical_layer: usize) -> bool {
        self.logical_to_slot.get(logical_layer).is_some_and(Option::is_some)
    }

    pub fn cache_slot(&self, logical_layer: usize) -> Result<usize, String> {
        self.logical_to_slot
            .get(logical_layer)
            .copied()
            .flatten()
            .ok_or_else(|| if logical_layer >= self.logical_layer_count() { format!("KV cache 逻辑层 {logical_layer} 越界(共 {} 层)", self.logical_layer_count()) } else { format!("KV cache 逻辑层 {logical_layer} 没有物理槽") })
    }
}

/// 每种注意力对应的 cache 形态。决定每 token 的字节布局与实现层 kernel 分派。
#[derive(Debug, Clone)]
pub enum KvCacheSpec {
    /// MLA(GLM-5.2)。latent INT8 量化 + rope f16。latent 必须 norm 后、rope 必须 RoPE 后再写入。
    Mla { kv_lora_rank: usize, qk_rope_head_dim: usize },
    /// GQA(MiniMax-M3、LLaMA)。K/V 直接存(本次实现留阶段 4)。
    Gqa { num_kv_heads: usize, head_dim: usize },
}

/// Cache 的逻辑编码格式。格式属于持久化与布局语义，不属于某个 backend。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheFormat {
    F16,
    Int8,
}

impl KvCacheFormat {
    pub const fn code(self) -> u32 {
        match self {
            Self::F16 => 0,
            Self::Int8 => 1,
        }
    }

    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::F16),
            1 => Some(Self::Int8),
            _ => None,
        }
    }
}

/// MLA cache 单 token 的字节布局(codes + scales + rope)。
#[derive(Debug, Clone, Copy)]
pub struct MlaRecordLayout {
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub group_size: usize,
}

impl MlaRecordLayout {
    pub fn new(kv_lora_rank: usize, qk_rope_head_dim: usize, group_size: usize) -> Result<Self, String> {
        if kv_lora_rank == 0 || qk_rope_head_dim == 0 || group_size == 0 {
            return Err("MlaRecordLayout 维度不能为 0".to_owned());
        }
        if !kv_lora_rank.is_multiple_of(group_size) {
            return Err(format!("kv_lora_rank {kv_lora_rank} 必须能被 group_size {group_size} 整除"));
        }
        Ok(Self { kv_lora_rank, qk_rope_head_dim, group_size })
    }

    /// 每 token 的 codes 字节(INT8:每元素 1 字节)。
    pub fn codes_bytes_per_token(&self) -> usize {
        self.kv_lora_rank
    }

    /// 每 token 的 scales 字节(per-group 一个 f16)。
    pub fn scale_bytes_per_token(&self) -> usize {
        self.groups_per_row() * 2
    }

    /// 每 token 的 rope 字节(f16)。
    pub fn rope_bytes_per_token(&self) -> usize {
        self.qk_rope_head_dim * 2
    }

    /// 每 token 总字节(codes + scales + rope)。
    pub fn bytes_per_token(&self) -> usize {
        self.codes_bytes_per_token() + self.scale_bytes_per_token() + self.rope_bytes_per_token()
    }

    pub fn groups_per_row(&self) -> usize {
        self.kv_lora_rank / self.group_size
    }
}

/// 与设备无关的整块 KV cache 布局。
#[derive(Debug, Clone)]
pub struct KvCacheLayout {
    spec: KvCacheSpec,
    format: KvCacheFormat,
    layer_map: KvCacheLayerMap,
    capacity: usize,
    group_size: usize,
    mla_layout: Option<MlaRecordLayout>,
    bytes_per_token: usize,
    layer_stride: usize,
    total_bytes: usize,
}

impl KvCacheLayout {
    pub fn new(spec: KvCacheSpec, format: KvCacheFormat, layer_count: usize, capacity: usize, group_size: usize) -> Result<Self, String> {
        let layer_map = KvCacheLayerMap::dense(layer_count)?;
        Self::new_mapped(spec, format, layer_map, capacity, group_size)
    }

    pub fn new_mapped(spec: KvCacheSpec, format: KvCacheFormat, layer_map: KvCacheLayerMap, capacity: usize, group_size: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("KvCacheLayout capacity 不能为 0".to_owned());
        }
        let mla_layout = match &spec {
            KvCacheSpec::Mla { kv_lora_rank, qk_rope_head_dim } => {
                let layout_group_size = if format == KvCacheFormat::F16 { *kv_lora_rank } else { group_size };
                Some(MlaRecordLayout::new(*kv_lora_rank, *qk_rope_head_dim, layout_group_size)?)
            }
            KvCacheSpec::Gqa { .. } => None,
        };
        let bytes_per_token = match (&spec, format) {
            (KvCacheSpec::Mla { kv_lora_rank, qk_rope_head_dim }, KvCacheFormat::F16) => (kv_lora_rank + qk_rope_head_dim) * 2,
            (KvCacheSpec::Gqa { num_kv_heads, head_dim }, KvCacheFormat::F16) => num_kv_heads.checked_mul(*head_dim).and_then(|columns| columns.checked_mul(4)).ok_or_else(|| "GQA F16 bytes_per_token 溢出".to_owned())?,
            _ => spec.bytes_per_token(group_size)?,
        };
        let layer_stride = capacity.checked_mul(bytes_per_token).ok_or_else(|| "KvCacheLayout layer_stride 溢出".to_owned())?;
        let total_bytes = layer_map.slot_count().checked_mul(layer_stride).ok_or_else(|| "KvCacheLayout 总字节数溢出".to_owned())?;
        Ok(Self { spec, format, layer_map, capacity, group_size, mla_layout, bytes_per_token, layer_stride, total_bytes })
    }

    pub fn spec(&self) -> &KvCacheSpec {
        &self.spec
    }

    pub fn format(&self) -> KvCacheFormat {
        self.format
    }

    pub fn layer_count(&self) -> usize {
        self.layer_map.logical_layer_count()
    }

    pub fn cache_slot_count(&self) -> usize {
        self.layer_map.slot_count()
    }

    pub fn layer_map(&self) -> &KvCacheLayerMap {
        &self.layer_map
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn mla_layout(&self) -> Option<&MlaRecordLayout> {
        self.mla_layout.as_ref()
    }

    pub fn bytes_per_token(&self) -> usize {
        self.bytes_per_token
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn layer_base(&self, layer: usize) -> Result<usize, String> {
        let slot = self.layer_map.cache_slot(layer)?;
        Ok(slot * self.layer_stride)
    }

    pub fn layer_latent_offset(&self, layer: usize) -> Result<usize, String> {
        self.layer_base(layer)
    }

    pub fn layer_codes_offset(&self, layer: usize) -> Result<usize, String> {
        if self.format != KvCacheFormat::Int8 {
            return Err("layer_codes_offset 仅 INT8 cache 有意义".to_owned());
        }
        self.layer_latent_offset(layer)
    }

    pub fn layer_scales_offset(&self, layer: usize) -> Result<usize, String> {
        if self.format != KvCacheFormat::Int8 {
            return Err("layer_scales_offset 仅 INT8 cache 有意义".to_owned());
        }
        let base = self.layer_base(layer)?;
        let layout = self.mla_layout.ok_or("layer_scales_offset 仅 MLA 有意义")?;
        Ok(base + self.capacity * layout.codes_bytes_per_token())
    }

    pub fn layer_rope_offset(&self, layer: usize) -> Result<usize, String> {
        let base = self.layer_base(layer)?;
        let layout = self.mla_layout.ok_or("layer_rope_offset 仅 MLA 有意义")?;
        let latent_stride = match self.format {
            KvCacheFormat::F16 => self.capacity * layout.kv_lora_rank * 2,
            KvCacheFormat::Int8 => self.capacity * (layout.codes_bytes_per_token() + layout.scale_bytes_per_token()),
        };
        Ok(base + latent_stride)
    }

    pub fn gqa_columns(&self) -> Result<usize, String> {
        let KvCacheSpec::Gqa { num_kv_heads, head_dim } = &self.spec else {
            return Err("gqa_columns 仅 GQA cache 有意义".to_owned());
        };
        num_kv_heads.checked_mul(*head_dim).ok_or_else(|| "GQA KV columns 溢出".to_owned())
    }

    pub fn gqa_groups_per_token(&self) -> Result<usize, String> {
        let KvCacheSpec::Gqa { num_kv_heads, head_dim } = &self.spec else {
            return Err("gqa_groups_per_token 仅 GQA cache 有意义".to_owned());
        };
        if *head_dim == 0 || !head_dim.is_multiple_of(self.group_size) {
            return Err(format!("GQA head_dim {head_dim} 必须能被 group_size {} 整除", self.group_size));
        }
        num_kv_heads.checked_mul(head_dim / self.group_size).ok_or_else(|| "GQA groups_per_token 溢出".to_owned())
    }

    pub fn layer_gqa_key_offset(&self, layer: usize) -> Result<usize, String> {
        self.gqa_columns()?;
        self.layer_base(layer)
    }

    pub fn layer_gqa_key_scale_offset(&self, layer: usize) -> Result<usize, String> {
        if self.format != KvCacheFormat::Int8 {
            return Err("GQA key scale 仅 INT8 cache 有意义".to_owned());
        }
        let base = self.layer_base(layer)?;
        let columns = self.gqa_columns()?;
        base.checked_add(self.capacity.checked_mul(columns).ok_or("GQA K codes 大小溢出")?).ok_or_else(|| "GQA K scales offset 溢出".to_owned())
    }

    pub fn layer_gqa_value_offset(&self, layer: usize) -> Result<usize, String> {
        let base = self.layer_base(layer)?;
        let columns = self.gqa_columns()?;
        let key_bytes = match self.format {
            KvCacheFormat::F16 => self.capacity.checked_mul(columns).and_then(|elements| elements.checked_mul(2)),
            KvCacheFormat::Int8 => self.capacity.checked_mul(columns).and_then(|codes| self.capacity.checked_mul(self.gqa_groups_per_token().ok()?).and_then(|scales| scales.checked_mul(2)).and_then(|scales| codes.checked_add(scales))),
        }
        .ok_or_else(|| "GQA K segment 字节数溢出".to_owned())?;
        base.checked_add(key_bytes).ok_or_else(|| "GQA V offset 溢出".to_owned())
    }

    pub fn layer_gqa_value_scale_offset(&self, layer: usize) -> Result<usize, String> {
        if self.format != KvCacheFormat::Int8 {
            return Err("GQA value scale 仅 INT8 cache 有意义".to_owned());
        }
        let value = self.layer_gqa_value_offset(layer)?;
        let columns = self.gqa_columns()?;
        value.checked_add(self.capacity.checked_mul(columns).ok_or("GQA V codes 大小溢出")?).ok_or_else(|| "GQA V scales offset 溢出".to_owned())
    }
}

/// 与设备无关的 cache 有效区间状态。
#[derive(Debug, Clone)]
pub struct KvCacheState {
    layout: KvCacheLayout,
    lengths: Vec<usize>,
}

impl KvCacheState {
    pub fn new(layout: KvCacheLayout) -> Self {
        let lengths = vec![0; layout.cache_slot_count()];
        Self { layout, lengths }
    }

    pub fn layout(&self) -> &KvCacheLayout {
        &self.layout
    }

    pub fn layer_len(&self, layer: usize) -> usize {
        self.layout.layer_map().cache_slot(layer).ok().and_then(|slot| self.lengths.get(slot).copied()).unwrap_or(0)
    }

    pub fn append_end(&self, layer: usize, count: usize) -> Result<usize, String> {
        let slot = self.layout.layer_map().cache_slot(layer)?;
        let current = self.lengths[slot];
        let end = current.checked_add(count).ok_or_else(|| "KV cache append 后长度溢出".to_owned())?;
        if end > self.layout.capacity() {
            return Err(format!("append 超出 capacity: 当前 {current} + 新 {count} > capacity {}", self.layout.capacity()));
        }
        Ok(end)
    }

    pub fn set_layer_len(&mut self, layer: usize, len: usize) -> Result<(), String> {
        let slot = self.layout.layer_map().cache_slot(layer)?;
        if len > self.layout.capacity() {
            return Err(format!("KV cache L{layer} 长度 {len} 超过 capacity {}", self.layout.capacity()));
        }
        self.lengths[slot] = len;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.lengths.fill(0);
    }

    pub fn truncate(&mut self, len: usize) {
        for layer_len in &mut self.lengths {
            *layer_len = (*layer_len).min(len);
        }
    }
}

/// 一层 hybrid GQA cache 的逻辑元素布局。offset 单位是元素，不绑定设备 dtype。
#[derive(Debug, Clone)]
pub struct HybridGqaLayerCacheLayout {
    pub attention: HybridGqaLayerSpec,
    pub capacity: usize,
    pub columns: usize,
    pub key_offset: usize,
    pub value_offset: usize,
    pub element_count: usize,
}

/// Hybrid GQA 的逐层 cache 布局。K/V 始终独立存储，即使它们共享投影结果。
#[derive(Debug, Clone)]
pub struct HybridGqaCacheLayout {
    layers: Vec<HybridGqaLayerCacheLayout>,
    max_sequence_len: usize,
    total_elements: usize,
}

impl HybridGqaCacheLayout {
    pub fn new(spec: &HybridGqaSpec, max_sequence_len: usize) -> Result<Self, String> {
        if max_sequence_len == 0 {
            return Err("hybrid GQA cache max_sequence_len 不能为 0".to_owned());
        }
        let mut layers = Vec::with_capacity(spec.layer_count());
        let mut total_elements = 0usize;
        for (layer, attention) in spec.layers().iter().copied().enumerate() {
            attention.validate().map_err(|error| format!("hybrid cache L{layer}: {error}"))?;
            let capacity = attention.window.cache_capacity(max_sequence_len);
            let columns = attention.geometry.kv_columns()?;
            let segment_elements = capacity.checked_mul(columns).ok_or_else(|| format!("hybrid cache L{layer} segment elements 溢出"))?;
            let key_offset = total_elements;
            let value_offset = key_offset.checked_add(segment_elements).ok_or_else(|| format!("hybrid cache L{layer} value offset 溢出"))?;
            let element_count = segment_elements.checked_mul(2).ok_or_else(|| format!("hybrid cache L{layer} elements 溢出"))?;
            total_elements = total_elements.checked_add(element_count).ok_or_else(|| "hybrid cache 总元素数溢出".to_owned())?;
            layers.push(HybridGqaLayerCacheLayout { attention, capacity, columns, key_offset, value_offset, element_count });
        }
        Ok(Self { layers, max_sequence_len, total_elements })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, layer: usize) -> Result<&HybridGqaLayerCacheLayout, String> {
        self.layers.get(layer).ok_or_else(|| format!("hybrid cache layer {layer} 越界(共 {} 层)", self.layers.len()))
    }

    pub fn max_sequence_len(&self) -> usize {
        self.max_sequence_len
    }

    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    fn write_plan(&self, layer: usize, position: usize, count: usize) -> Result<HybridGqaAppendPlan, String> {
        if count == 0 {
            return Err("hybrid cache append count 不能为 0".to_owned());
        }
        let layout = self.layer(layer)?;
        let end = position.checked_add(count).ok_or_else(|| "hybrid cache append position 溢出".to_owned())?;
        if end > self.max_sequence_len {
            return Err(format!("hybrid cache append end={end} 超过 max_sequence_len={}", self.max_sequence_len));
        }
        let write_start = position.max(end.saturating_sub(layout.capacity));
        let token_count = end - write_start;
        let source_start = write_start - position;
        let first_slot = write_start % layout.capacity;
        let first_count = token_count.min(layout.capacity - first_slot);
        Ok(HybridGqaAppendPlan { layer, position, end, source_start, token_count, first_slot, first_count, second_count: token_count - first_count })
    }
}

/// 一次 append 的逻辑写计划。backend 对 K/V 使用同一计划，各自写入对应 segment。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridGqaAppendPlan {
    pub layer: usize,
    pub position: usize,
    pub end: usize,
    /// 输入 batch 中被窗口淘汰、不需要写入 cache 的 token 数。
    pub source_start: usize,
    pub token_count: usize,
    pub first_slot: usize,
    pub first_count: usize,
    /// ring wrap 后从 slot 0 写入的 token 数。
    pub second_count: usize,
}

/// Hybrid cache 的绝对有效区间状态。保留 start，避免 truncate 后假装已淘汰数据仍存在。
#[derive(Debug, Clone)]
pub struct HybridGqaCacheState {
    layout: HybridGqaCacheLayout,
    ranges: Vec<Range<usize>>,
}

impl HybridGqaCacheState {
    pub fn new(layout: HybridGqaCacheLayout) -> Self {
        let ranges = vec![0..0; layout.layer_count()];
        Self { layout, ranges }
    }

    pub fn layout(&self) -> &HybridGqaCacheLayout {
        &self.layout
    }

    pub fn retained_range(&self, layer: usize) -> Result<Range<usize>, String> {
        self.layout.layer(layer)?;
        Ok(self.ranges[layer].clone())
    }

    /// 快照恢复:整体回填各层 append 游标(字节由调用方先行写回 buffer)。
    pub fn restore_ranges(&mut self, ranges: Vec<Range<usize>>) -> Result<(), String> {
        if ranges.len() != self.layout.layer_count() {
            return Err(format!("hybrid cache 范围数 {} 与层数 {} 不符", ranges.len(), self.layout.layer_count()));
        }
        self.ranges = ranges;
        Ok(())
    }

    pub fn plan_append(&self, layer: usize, position: usize, count: usize) -> Result<HybridGqaAppendPlan, String> {
        self.layout.layer(layer)?;
        if position != self.ranges[layer].end {
            return Err(format!("hybrid cache L{layer} append position={position}，期望 {}", self.ranges[layer].end));
        }
        self.layout.write_plan(layer, position, count)
    }

    pub fn commit_append(&mut self, plan: &HybridGqaAppendPlan) -> Result<(), String> {
        let layer = self.layout.layer(plan.layer)?;
        let current = &self.ranges[plan.layer];
        if current.end != plan.position {
            return Err(format!("hybrid cache L{} commit 已过期: state_end={} plan_position={}", plan.layer, current.end, plan.position));
        }
        let start = current.start.max(plan.end.saturating_sub(layer.capacity));
        self.ranges[plan.layer] = start..plan.end;
        Ok(())
    }

    pub fn visible_range(&self, layer: usize, query_position: usize) -> Result<Range<usize>, String> {
        let layout = self.layout.layer(layer)?;
        let retained = &self.ranges[layer];
        if query_position < retained.start || query_position >= retained.end {
            return Err(format!("hybrid cache L{layer} query={query_position} 不在 retained {retained:?}"));
        }
        let causal = layout.attention.window.key_range(query_position, retained.end)?;
        Ok(causal.start.max(retained.start)..causal.end.min(retained.end))
    }

    pub fn slot(&self, layer: usize, position: usize) -> Result<usize, String> {
        let layout = self.layout.layer(layer)?;
        let retained = &self.ranges[layer];
        if !retained.contains(&position) {
            return Err(format!("hybrid cache L{layer} position={position} 不在 retained {retained:?}"));
        }
        Ok(position % layout.capacity)
    }

    pub fn truncate(&mut self, end: usize) -> Result<(), String> {
        for (layer, range) in self.ranges.iter().enumerate() {
            // shared-KV 层自身不 append，范围始终为空；回滚只应收缩真正持有
            // K/V 的层，不能把空层伪造成已有 `end` 个 token。
            if range.start == range.end {
                continue;
            }
            if end > range.end || (end < range.start && end != 0) {
                return Err(format!("hybrid cache L{layer} 无法从 {range:?} truncate 到 {end}"));
            }
        }
        if end == 0 {
            self.clear();
        } else {
            for range in &mut self.ranges {
                if range.start != range.end {
                    range.end = end;
                }
            }
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.ranges.fill(0..0);
    }
}

impl KvCacheSpec {
    /// MLA 单 token 字节数(决定 buffer record stride)。
    pub fn bytes_per_token(&self, group_size: usize) -> Result<usize, String> {
        match self {
            Self::Mla { kv_lora_rank, qk_rope_head_dim } => MlaRecordLayout::new(*kv_lora_rank, *qk_rope_head_dim, group_size).map(|l| l.bytes_per_token()),
            // GQA Q8:K/V 各自保存 i8 codes 与每组一个 f16 scale。
            Self::Gqa { num_kv_heads, head_dim } => {
                if group_size == 0 || !head_dim.is_multiple_of(group_size) {
                    return Err(format!("GQA head_dim {head_dim} 必须能被 group_size {group_size} 整除"));
                }
                let columns = num_kv_heads.checked_mul(*head_dim).ok_or("GQA columns 溢出")?;
                let groups = num_kv_heads.checked_mul(head_dim / group_size).ok_or("GQA groups 溢出")?;
                columns.checked_mul(2).and_then(|codes| groups.checked_mul(4).and_then(|scales| codes.checked_add(scales))).ok_or_else(|| "GQA Q8 bytes_per_token 溢出".to_owned())
            }
        }
    }

    /// 从 [`AttentionSpec`] 构造 cache 形态。同一模型每层形态一致,取任一层即可。
    /// MSA(MiniMax 稀疏)暂映射到其内嵌 GQA 的 cache 形态,indexer 的 `Ic` 留阶段 5。
    pub fn from_attention(attn: &AttentionSpec) -> Result<Self, String> {
        match attn {
            AttentionSpec::Block(_) => Err("Block attention 的可见区间由调用方持有，不进入通用 KV cache".to_owned()),
            AttentionSpec::Mla(MlaSpec { kv_lora_rank, qk_rope_head_dim, .. }) => Ok(Self::Mla { kv_lora_rank: *kv_lora_rank, qk_rope_head_dim: *qk_rope_head_dim }),
            AttentionSpec::GatedMla(spec) => Ok(Self::Mla { kv_lora_rank: spec.mla.kv_lora_rank, qk_rope_head_dim: spec.mla.qk_rope_head_dim }),
            AttentionSpec::Gqa(GqaSpec { num_kv_heads, head_dim, .. }) | AttentionSpec::Msa(crate::attention::msa::MsaSpec { gqa: GqaSpec { num_kv_heads, head_dim, .. }, .. }) => {
                Ok(Self::Gqa { num_kv_heads: *num_kv_heads, head_dim: *head_dim })
            }
            AttentionSpec::GatedDeltaNet(_) => Err("Gated DeltaNet 使用独立 recurrent state，不进入 KV cache".to_owned()),
            AttentionSpec::Kda(_) => Err("KDA 使用独立 recurrent state 与短卷积 state，不进入 KV cache".to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_mla_spec() -> KvCacheSpec {
        KvCacheSpec::Mla { kv_lora_rank: 8, qk_rope_head_dim: 4 }
    }

    fn small_gqa_spec() -> KvCacheSpec {
        KvCacheSpec::Gqa { num_kv_heads: 2, head_dim: 4 }
    }

    #[test]
    fn bytes_per_token_mla_int8() {
        // kv_lora=8, rope=4, group=4 → codes=8B + scales=(8/4)*2=4B + rope=4*2=8B = 20B
        assert_eq!(small_mla_spec().bytes_per_token(4).unwrap(), 20);
    }

    #[test]
    fn bytes_per_token_gqa() {
        // K/V 各 2×4 i8 codes + 2 个 f16 scale，共 24 字节。
        assert_eq!(small_gqa_spec().bytes_per_token(4).unwrap(), 24);
    }

    #[test]
    fn from_attention_mla() {
        let mla = MlaSpec { q_lora_rank: 16, kv_lora_rank: 8, qk_rope_head_dim: 4, q_projection_size: 32, kv_projection_size: 48, num_heads: 2, rope_theta: 1.0, rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf };
        let spec = KvCacheSpec::from_attention(&AttentionSpec::Mla(mla)).expect("MLA → spec");
        assert!(matches!(spec, KvCacheSpec::Mla { kv_lora_rank: 8, qk_rope_head_dim: 4 }));
    }

    #[test]
    fn from_attention_gqa_and_msa() {
        let gqa = GqaSpec { num_heads: 8, num_kv_heads: 2, head_dim: 4, rope_dim: 2, rope_theta: 1.0, use_qk_norm: false, window: crate::attention::gqa::CausalWindow::Full, score_scale: 0.5, output_gate: false };
        let spec = KvCacheSpec::from_attention(&AttentionSpec::Gqa(gqa)).expect("GQA → spec");
        assert!(matches!(spec, KvCacheSpec::Gqa { num_kv_heads: 2, head_dim: 4 }));

        let msa = crate::attention::msa::MsaSpec {
            gqa: GqaSpec { num_heads: 8, num_kv_heads: 2, head_dim: 4, rope_dim: 2, rope_theta: 1.0, use_qk_norm: false, window: crate::attention::gqa::CausalWindow::Full, score_scale: 0.5, output_gate: false },
            index_dim: 4,
            num_index_heads: 2,
            block_size: 128,
            topk_blocks: 16,
            init_block: 0,
            local_block: 1,
        };
        let spec = KvCacheSpec::from_attention(&AttentionSpec::Msa(msa)).expect("MSA → spec");
        assert!(matches!(spec, KvCacheSpec::Gqa { num_kv_heads: 2, head_dim: 4 }));
    }

    #[test]
    fn mla_layout_group_size_must_divide() {
        // kv_lora=8, group=3 不能整除 → 应报错。
        assert!(MlaRecordLayout::new(8, 4, 3).is_err());
        assert!(MlaRecordLayout::new(8, 4, 4).is_ok());
    }

    #[test]
    fn mla_layout_zero_dims_error() {
        assert!(MlaRecordLayout::new(0, 4, 4).is_err());
        assert!(MlaRecordLayout::new(8, 0, 4).is_err());
        assert!(MlaRecordLayout::new(8, 4, 0).is_err());
    }

    #[test]
    fn cache_layout_computes_mla_offsets() {
        let layout = KvCacheLayout::new(small_mla_spec(), KvCacheFormat::Int8, 2, 3, 4).unwrap();
        assert_eq!(layout.bytes_per_token(), 20);
        assert_eq!(layout.total_bytes(), 120);
        assert_eq!(layout.layer_codes_offset(1).unwrap(), 60);
        assert_eq!(layout.layer_scales_offset(1).unwrap(), 84);
        assert_eq!(layout.layer_rope_offset(1).unwrap(), 96);
    }

    #[test]
    fn cache_state_tracks_valid_lengths() {
        let layout = KvCacheLayout::new(small_mla_spec(), KvCacheFormat::F16, 2, 4, 64).unwrap();
        let mut state = KvCacheState::new(layout);
        assert_eq!(state.append_end(1, 3).unwrap(), 3);
        state.set_layer_len(1, 3).unwrap();
        assert!(state.append_end(1, 2).is_err());
        state.truncate(2);
        assert_eq!(state.layer_len(1), 2);
        state.clear();
        assert_eq!(state.layer_len(1), 0);
    }

    #[test]
    fn hybrid_cache_uses_per_layer_capacity_and_ring_plan() {
        use crate::attention::{
            gqa::{CausalWindow, GqaGeometry, GqaKvProjection, HybridGqaLayerSpec, HybridGqaSpec},
            rope::RopeSpec,
        };

        let geometry = GqaGeometry { num_heads: 2, num_kv_heads: 1, head_dim: 2 };
        let layer = |window| HybridGqaLayerSpec { geometry, rope: RopeSpec::Default { rotary_dim: 2, theta: 10_000.0 }, window, score_scale: 1.0, kv_projection: GqaKvProjection::Separate };
        let spec = HybridGqaSpec::new(vec![layer(CausalWindow::Sliding { size: 3 }), layer(CausalWindow::Full)]).unwrap();
        let layout = HybridGqaCacheLayout::new(&spec, 10).unwrap();
        assert_eq!(layout.layer(0).unwrap().capacity, 3);
        assert_eq!(layout.layer(1).unwrap().capacity, 10);

        let mut state = HybridGqaCacheState::new(layout);
        let plan = state.plan_append(0, 0, 5).unwrap();
        assert_eq!((plan.source_start, plan.first_slot, plan.first_count, plan.second_count), (2, 2, 1, 2));
        state.commit_append(&plan).unwrap();
        assert_eq!(state.retained_range(0).unwrap(), 2..5);
        assert_eq!(state.visible_range(0, 4).unwrap(), 2..5);
    }

    #[test]
    fn hybrid_cache_truncate_preserves_empty_shared_kv_layers() {
        use crate::attention::{
            gqa::{CausalWindow, GqaGeometry, GqaKvProjection, HybridGqaLayerSpec, HybridGqaSpec},
            rope::RopeSpec,
        };

        let geometry = GqaGeometry { num_heads: 2, num_kv_heads: 1, head_dim: 2 };
        let layer = HybridGqaLayerSpec { geometry, rope: RopeSpec::Default { rotary_dim: 2, theta: 10_000.0 }, window: CausalWindow::Full, score_scale: 1.0, kv_projection: GqaKvProjection::Separate };
        let layout = HybridGqaCacheLayout::new(&HybridGqaSpec::new(vec![layer, layer]).unwrap(), 10).unwrap();
        let mut state = HybridGqaCacheState::new(layout);
        let plan = state.plan_append(0, 0, 5).unwrap();
        state.commit_append(&plan).unwrap();

        state.truncate(4).unwrap();
        assert_eq!(state.retained_range(0).unwrap(), 0..4);
        assert_eq!(state.retained_range(1).unwrap(), 0..0);
    }
}
