//! 滑窗与压缩 KV 共同组成的稀疏注意力规格。
//!
//! DeepSeek-V4 的 CSA/HCA 都保留最近的滑窗 KV，并把更早 token 通过学习式
//! gated pooling 压成稀疏 KV。压缩率 4 使用 indexer 选择历史块，压缩率 128
//! 直接读取全部高压缩历史；这里仅描述语义，存储与 kernel 留给 backend。

use super::{dsa::DsaSpec, rope::RopeSpec};
use crate::backend::{Backend, BackendError, SegmentedTensorBackend};
#[cfg(test)]
use std::cmp::Ordering;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub enum CompressedSelection {
    LearnedIndexer(DsaSpec),
    All,
}

#[derive(Debug, Clone, Copy)]
pub struct KvCompressionSpec {
    pub ratio: usize,
    pub overlap: bool,
    pub selection: CompressedSelection,
}

#[derive(Debug, Clone, Copy)]
pub struct CompressedSparseAttentionSpec {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub output_groups: usize,
    pub output_lora_rank: usize,
    pub window_size: usize,
    pub rope: RopeSpec,
    pub compression: Option<KvCompressionSpec>,
    pub attention_sink: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStream {
    Attention,
    Indexer,
}

pub struct CompressedBatch<T> {
    /// 压缩项覆盖的最后一个源 token；attention 用它判断该项何时可见。
    pub visible_positions: Vec<usize>,
    pub values: T,
}

/// 一段连续 token 对应一份独立 compressor 状态。
pub struct CompressedGatedSegment<'a, B: CompressedSparseKernel + ?Sized> {
    pub storage: &'a mut B::CompressedKvStorage,
    pub positions: &'a [usize],
}

/// 一段连续 token 对应一份独立 CSA/KV 状态；Q/K/V 与 indexer 输入保持整批连续。
pub struct CompressedSparsePrefillSegment<'a, B: CompressedSparseKernel + ?Sized> {
    pub storage: &'a mut B::CompressedKvStorage,
    pub positions: &'a [usize],
    pub causal_batch: bool,
    pub compressed_positions: Option<&'a [usize]>,
    pub compressed_key: Option<&'a B::Tensor>,
    pub compressed_value: Option<&'a B::Tensor>,
    pub compressed_index_key: Option<&'a B::Tensor>,
}

/// Compressor 跨调用保留的纯逻辑游标。pending/overlap 的物理 buffer 仍由
/// backend 持有，kernel 成功后再提交计划，避免失败时逻辑状态提前推进。
#[cfg(any(target_os = "macos", feature = "with-rocm", test))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompressionState {
    next_position: usize,
    entry_count: usize,
    pending_rows: usize,
}

#[cfg(any(target_os = "macos", feature = "with-rocm", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompressionPlan {
    ratio: usize,
    pending_rows: usize,
    entry_start: usize,
    windows: usize,
    remaining_rows: usize,
    next_position: usize,
    entry_count: usize,
}

#[cfg(any(target_os = "macos", feature = "with-rocm", test))]
impl CompressionState {
    pub(crate) fn plan(&self, positions: &[usize], ratio: usize) -> Result<CompressionPlan, String> {
        if ratio == 0 || positions.is_empty() || positions.iter().enumerate().any(|(index, &position)| self.next_position.checked_add(index) != Some(position)) {
            return Err(format!("compressor positions 非连续: positions={positions:?} next={} ratio={ratio}", self.next_position));
        }
        self.plan_rows(positions.len(), ratio)
    }

    pub(crate) fn plan_rows(&self, rows: usize, ratio: usize) -> Result<CompressionPlan, String> {
        if ratio == 0 || rows == 0 {
            return Err(format!("compressor rows={rows} ratio={ratio} 非法"));
        }
        let total_rows = self.pending_rows.checked_add(rows).ok_or_else(|| "compressor pending rows 溢出".to_owned())?;
        let windows = total_rows / ratio;
        let entry_count = self.entry_count.checked_add(windows).ok_or_else(|| "compressor entry count 溢出".to_owned())?;
        let next_position = self.next_position.checked_add(rows).ok_or_else(|| "compressor position 溢出".to_owned())?;
        if windows != 0 {
            entry_count.checked_mul(ratio).ok_or_else(|| "compressor visible position 溢出".to_owned())?;
        }
        Ok(CompressionPlan { ratio, pending_rows: self.pending_rows, entry_start: self.entry_count, windows, remaining_rows: total_rows % ratio, next_position, entry_count })
    }

    #[cfg(feature = "with-rocm")]
    pub(crate) fn from_parts(next_position: usize, entry_count: usize, pending_rows: usize) -> Self {
        Self { next_position, entry_count, pending_rows }
    }

    #[cfg(feature = "with-rocm")]
    pub(crate) fn parts(self) -> (usize, usize, usize) {
        (self.next_position, self.entry_count, self.pending_rows)
    }

    #[cfg(feature = "with-rocm")]
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn commit(&mut self, plan: CompressionPlan, actual_remaining_rows: usize) -> Result<(), String> {
        if actual_remaining_rows != plan.remaining_rows {
            return Err(format!("compressor kernel remaining rows={actual_remaining_rows}，期望 {}", plan.remaining_rows));
        }
        self.next_position = plan.next_position;
        self.entry_count = plan.entry_count;
        self.pending_rows = actual_remaining_rows;
        Ok(())
    }
}

#[cfg(any(target_os = "macos", feature = "with-rocm", test))]
impl CompressionPlan {
    pub(crate) fn pending_rows(self) -> usize {
        self.pending_rows
    }

    pub(crate) fn entry_start(self) -> usize {
        self.entry_start
    }

    pub(crate) fn windows(self) -> usize {
        self.windows
    }

    pub(crate) fn visible_positions(self) -> Vec<usize> {
        (self.entry_start..self.entry_count).map(|entry| entry * self.ratio + self.ratio - 1).collect()
    }
}

/// Compressor 尚未凑满一个窗口的投影结果。物理存储由 backend 持有，这个
/// reference 状态只定义跨 prefill/decode 调用必须保持的数学语义。
#[derive(Debug, Default)]
pub struct GatedPoolState {
    next_position: usize,
    entry_count: usize,
    kv: Vec<f32>,
    gate: Vec<f32>,
    overlap_kv: Option<Vec<f32>>,
    overlap_gate: Option<Vec<f32>>,
}

impl GatedPoolState {
    #[allow(clippy::too_many_arguments)]
    pub fn push_f32(&mut self, positions: &[usize], kv: &[f32], gate: &[f32], position_bias: &[f32], ratio: usize, width: usize, overlap: bool) -> Result<(Vec<usize>, Vec<f32>), String> {
        let channels = if overlap { 2 * width } else { width };
        if ratio == 0
            || width == 0
            || kv.len() != positions.len() * channels
            || gate.len() != kv.len()
            || position_bias.len() != ratio * channels
            || positions.iter().enumerate().any(|(index, &position)| position != self.next_position + index)
        {
            return Err(format!("compressor 输入非法: positions={positions:?} next={} kv={} gate={} bias={} ratio={ratio} width={width} overlap={overlap}", self.next_position, kv.len(), gate.len(), position_bias.len(),));
        }
        self.next_position += positions.len();
        self.kv.extend_from_slice(kv);
        self.gate.extend_from_slice(gate);

        let buffered_rows = self.kv.len() / channels;
        let usable_rows = buffered_rows / ratio * ratio;
        if usable_rows == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let remaining_kv = self.kv.split_off(usable_rows * channels);
        let remaining_gate = self.gate.split_off(usable_rows * channels);
        let ready_kv = std::mem::replace(&mut self.kv, remaining_kv);
        let ready_gate = std::mem::replace(&mut self.gate, remaining_gate);
        let windows = usable_rows / ratio;
        let mut visible_positions = Vec::with_capacity(windows);
        let mut output = Vec::with_capacity(windows * width);

        for window in 0..windows {
            let begin = window * ratio * channels;
            let window_kv = &ready_kv[begin..begin + ratio * channels];
            let window_gate = &ready_gate[begin..begin + ratio * channels];
            let biased_gate = window_gate.iter().zip(position_bias).map(|(gate, bias)| gate + bias).collect::<Vec<_>>();
            if overlap {
                let mut pool_kv = Vec::with_capacity(2 * ratio * width);
                let mut pool_gate = Vec::with_capacity(2 * ratio * width);
                if let (Some(previous_kv), Some(previous_gate)) = (&self.overlap_kv, &self.overlap_gate) {
                    pool_kv.extend_from_slice(previous_kv);
                    pool_gate.extend_from_slice(previous_gate);
                } else {
                    pool_kv.extend(std::iter::repeat_n(0.0, ratio * width));
                    pool_gate.extend(std::iter::repeat_n(f32::NEG_INFINITY, ratio * width));
                }
                for row in 0..ratio {
                    let row_begin = row * channels;
                    pool_kv.extend_from_slice(&window_kv[row_begin + width..row_begin + channels]);
                    pool_gate.extend_from_slice(&biased_gate[row_begin + width..row_begin + channels]);
                }
                output.extend(gated_pool_f32(&pool_kv, &pool_gate, width)?);

                let mut next_kv = Vec::with_capacity(ratio * width);
                let mut next_gate = Vec::with_capacity(ratio * width);
                for row in 0..ratio {
                    let row_begin = row * channels;
                    next_kv.extend_from_slice(&window_kv[row_begin..row_begin + width]);
                    next_gate.extend_from_slice(&biased_gate[row_begin..row_begin + width]);
                }
                self.overlap_kv = Some(next_kv);
                self.overlap_gate = Some(next_gate);
            } else {
                output.extend(gated_pool_f32(window_kv, &biased_gate, width)?);
            }
            let rope_position = self.entry_count * ratio;
            visible_positions.push(rope_position + ratio - 1);
            self.entry_count += 1;
        }
        Ok((visible_positions, output))
    }
}

impl CompressedSparseAttentionSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.num_heads == 0
            || self.num_kv_heads == 0
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
            || self.head_dim == 0
            || self.q_lora_rank == 0
            || self.output_groups == 0
            || !self.num_heads.is_multiple_of(self.output_groups)
            || self.output_lora_rank == 0
            || self.window_size == 0
        {
            return Err(format!(
                "压缩稀疏注意力维度非法: heads={}/{} head_dim={} q_rank={} o_groups={} o_rank={} window={}",
                self.num_heads, self.num_kv_heads, self.head_dim, self.q_lora_rank, self.output_groups, self.output_lora_rank, self.window_size,
            ));
        }
        self.rope.validate()?;
        if self.rope.rotary_dim() > self.head_dim {
            return Err(format!("压缩稀疏注意力 RoPE {} 超过 head_dim {}", self.rope.rotary_dim(), self.head_dim));
        }
        if let Some(compression) = self.compression {
            if compression.ratio < 2 {
                return Err(format!("KV compression ratio={} 非法", compression.ratio));
            }
            if compression.overlap && compression.ratio != 4 {
                return Err(format!("KV overlap 仅适用于 ratio=4，实际为 {}", compression.ratio));
            }
            if let CompressedSelection::LearnedIndexer(indexer) = compression.selection
                && (compression.ratio != 4 || indexer.num_heads == 0 || indexer.head_dim == 0 || indexer.top_k == 0 || indexer.rope_dim > indexer.head_dim)
            {
                return Err(format!("压缩 KV indexer 非法: ratio={} heads={} head_dim={} rope_dim={} top_k={}", compression.ratio, indexer.num_heads, indexer.head_dim, indexer.rope_dim, indexer.top_k,));
            }
        }
        Ok(())
    }

    /// Decode cache 行数：固定滑窗加压缩历史，不把临时 compressor state 算入 KV。
    pub fn cache_rows(&self, context: usize) -> Result<usize, String> {
        let compressed = self.compression.map_or(0, |spec| context / spec.ratio);
        self.window_size.checked_add(compressed).ok_or_else(|| "压缩稀疏 KV cache 行数溢出".to_owned())
    }
}

#[derive(Debug, Clone)]
struct KvRow {
    position: usize,
    key: Vec<f32>,
    value: Vec<f32>,
    index_key: Option<Vec<f32>>,
}

/// CPU oracle 使用的逻辑 KV；具体设备存储仍由 backend 拥有。
#[derive(Debug)]
pub struct CompressedKvState {
    window_size: usize,
    recent: VecDeque<KvRow>,
    compressed: Vec<KvRow>,
}

impl CompressedKvState {
    pub fn new(window_size: usize) -> Result<Self, String> {
        if window_size == 0 {
            return Err("压缩稀疏 KV 的 window_size 不能为 0".to_owned());
        }
        Ok(Self { window_size, recent: VecDeque::with_capacity(window_size), compressed: Vec::new() })
    }

    pub fn push_recent(&mut self, position: usize, key: &[f32], value: &[f32]) -> Result<(), String> {
        validate_kv(key, value)?;
        if self.recent.len() == self.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(KvRow { position, key: key.to_vec(), value: value.to_vec(), index_key: None });
        Ok(())
    }

    pub fn push_compressed(&mut self, position: usize, key: &[f32], value: &[f32]) -> Result<(), String> {
        validate_kv(key, value)?;
        self.compressed.push(KvRow { position, key: key.to_vec(), value: value.to_vec(), index_key: None });
        Ok(())
    }

    pub fn push_compressed_indexed(&mut self, position: usize, key: &[f32], value: &[f32], index_key: &[f32]) -> Result<(), String> {
        validate_kv(key, value)?;
        if index_key.is_empty() {
            return Err("压缩 KV index key 不能为空".to_owned());
        }
        self.compressed.push(KvRow { position, key: key.to_vec(), value: value.to_vec(), index_key: Some(index_key.to_vec()) });
        Ok(())
    }

    pub fn recent_len(&self) -> usize {
        self.recent.len()
    }

    pub fn compressed_len(&self) -> usize {
        self.compressed.len()
    }

    /// 历史压缩行只读访问（prefill 合并历史 + 本批压缩时使用）。
    pub fn compressed_position(&self, index: usize) -> usize {
        self.compressed[index].position
    }
    pub fn compressed_key(&self, index: usize) -> &[f32] {
        &self.compressed[index].key
    }
    pub fn compressed_value(&self, index: usize) -> &[f32] {
        &self.compressed[index].value
    }
    pub fn compressed_index_key(&self, index: usize) -> Option<&[f32]> {
        self.compressed[index].index_key.as_deref()
    }

    /// Prefill 后批量更新滑窗：保留最后 `window_size` 个 token。
    /// `positions`/`keys`/`values` 是本批全部 token 的 KV（行优先）。
    pub fn push_recent_batch(&mut self, positions: &[usize], keys: &[f32], values: &[f32], kv_width: usize) -> Result<(), String> {
        if positions.is_empty() {
            return if keys.is_empty() && values.is_empty() { Ok(()) } else { Err("空 recent batch 不能携带 KV 数据".to_owned()) };
        }
        if keys.len() != positions.len() * kv_width || values.len() != keys.len() {
            return Err(format!("批量 recent KV 维度非法: positions={} keys={} values={} kv_width={kv_width}", positions.len(), keys.len(), values.len(),));
        }
        let start = positions.len().saturating_sub(self.window_size);
        for row in start..positions.len() {
            let key = &keys[row * kv_width..(row + 1) * kv_width];
            let value = &values[row * kv_width..(row + 1) * kv_width];
            if self.recent.len() == self.window_size {
                self.recent.pop_front();
            }
            self.recent.push_back(KvRow { position: positions[row], key: key.to_vec(), value: value.to_vec(), index_key: None });
        }
        Ok(())
    }

    /// Prefill 后批量追加压缩历史。
    pub fn push_compressed_batch(&mut self, positions: &[usize], keys: &[f32], values: &[f32], kv_width: usize) -> Result<(), String> {
        if positions.is_empty() {
            return if keys.is_empty() && values.is_empty() { Ok(()) } else { Err("空 compressed batch 不能携带 KV 数据".to_owned()) };
        }
        if keys.len() != positions.len() * kv_width || values.len() != keys.len() {
            return Err(format!("批量 compressed KV 维度非法: positions={} keys={} values={} kv_width={kv_width}", positions.len(), keys.len(), values.len(),));
        }
        for row in 0..positions.len() {
            let key = &keys[row * kv_width..(row + 1) * kv_width];
            let value = &values[row * kv_width..(row + 1) * kv_width];
            self.compressed.push(KvRow { position: positions[row], key: key.to_vec(), value: value.to_vec(), index_key: None });
        }
        Ok(())
    }

    /// Prefill 后批量追加带 indexer 的压缩历史。
    pub fn push_compressed_indexed_batch(&mut self, positions: &[usize], keys: &[f32], values: &[f32], index_keys: &[f32], kv_width: usize) -> Result<(), String> {
        if positions.is_empty() {
            return if keys.is_empty() && values.is_empty() && index_keys.is_empty() { Ok(()) } else { Err("空 indexed compressed batch 不能携带 KV 数据".to_owned()) };
        }
        let index_width = index_keys.len() / positions.len();
        if keys.len() != positions.len() * kv_width || values.len() != keys.len() || index_keys.len() != positions.len() * index_width || index_width == 0 {
            return Err(format!("批量 indexed compressed KV 维度非法: positions={} keys={} index_keys={} kv_width={kv_width} index_width={index_width}", positions.len(), keys.len(), index_keys.len(),));
        }
        for row in 0..positions.len() {
            let key = &keys[row * kv_width..(row + 1) * kv_width];
            let value = &values[row * kv_width..(row + 1) * kv_width];
            let index_key = &index_keys[row * index_width..(row + 1) * index_width];
            self.compressed.push(KvRow { position: positions[row], key: key.to_vec(), value: value.to_vec(), index_key: Some(index_key.to_vec()) });
        }
        Ok(())
    }

    pub fn selected_indices(&self, query: &[f32], head_weights: &[f32], spec: &DsaSpec) -> Result<Vec<usize>, String> {
        if self.compressed.is_empty() {
            return Ok(Vec::new());
        }
        let width = self.compressed[0].index_key.as_ref().ok_or_else(|| "压缩 KV 缺少 index key".to_owned())?.len();
        if width != spec.head_dim || query.len() != spec.num_heads * spec.head_dim || head_weights.len() != spec.num_heads || self.compressed.iter().any(|row| row.index_key.as_ref().is_none_or(|key| key.len() != width)) {
            return Err(format!("压缩 KV indexer 维度非法: query={} weights={} width={width} heads={}x{} rows={}", query.len(), head_weights.len(), spec.num_heads, spec.head_dim, self.compressed.len(),));
        }
        let mut keys = Vec::with_capacity(self.compressed.len() * width);
        for row in &self.compressed {
            keys.extend_from_slice(row.index_key.as_ref().expect("上方已验证 index key"));
        }
        topk_indexer_f32(query, &keys, head_weights, spec, spec.top_k)
    }

    /// 输入 Q/K 已完成当前层要求的 RoPE；sink 只参与 softmax 分母。
    pub fn attend_f32(&self, query: &[f32], num_heads: usize, num_kv_heads: usize, head_dim: usize, selected_compressed: Option<&[usize]>, sink: Option<&[f32]>) -> Result<Vec<f32>, String> {
        if num_heads == 0 || num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) || head_dim == 0 || query.len() != num_heads * head_dim || sink.is_some_and(|values| values.len() != num_heads) {
            return Err(format!("压缩稀疏 attention 维度非法: query={} heads={num_heads}/{num_kv_heads} head_dim={head_dim} sink={}", query.len(), sink.map_or(0, <[f32]>::len),));
        }
        let kv_width = num_kv_heads * head_dim;
        if self.recent.iter().chain(&self.compressed).any(|row| row.key.len() != kv_width || row.value.len() != kv_width) {
            return Err(format!("压缩稀疏 KV 宽度与预期 {kv_width} 不一致"));
        }
        let compressed_rows: Vec<&KvRow> = match selected_compressed {
            Some(selected) => selected.iter().map(|&index| self.compressed.get(index).ok_or_else(|| format!("压缩 KV 索引越界: index={index} len={}", self.compressed.len()))).collect::<Result<_, _>>()?,
            None => self.compressed.iter().collect(),
        };
        let mut output = vec![0.0; query.len()];
        let scale = 1.0 / (head_dim as f32).sqrt();
        for head in 0..num_heads {
            let kv_head = head * num_kv_heads / num_heads;
            let q = &query[head * head_dim..(head + 1) * head_dim];
            let mut scores = Vec::with_capacity(compressed_rows.len() + self.recent.len());
            let mut rows = Vec::with_capacity(scores.capacity());
            for row in compressed_rows.iter().copied().chain(self.recent.iter()) {
                let key = &row.key[kv_head * head_dim..(kv_head + 1) * head_dim];
                scores.push(q.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scale);
                rows.push(row);
            }
            let sink_score = sink.map(|values| values[head]);
            let max = scores.iter().copied().chain(sink_score).fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = sink_score.map_or(0.0, |value| (value - max).exp());
            let weights: Vec<f32> = scores
                .iter()
                .map(|score| {
                    let weight = (*score - max).exp();
                    denominator += weight;
                    weight
                })
                .collect();
            if !denominator.is_finite() || denominator == 0.0 {
                continue; // 非有限分母:输出保持 0 而非传播 NaN(batch 路径同判)
            }
            let out = &mut output[head * head_dim..(head + 1) * head_dim];
            for (weight, row) in weights.into_iter().zip(rows) {
                let value = &row.value[kv_head * head_dim..(kv_head + 1) * head_dim];
                for (out, value) in out.iter_mut().zip(value) {
                    *out += weight / denominator * value;
                }
            }
        }
        Ok(output)
    }

    /// 对一批 query 做压缩稀疏 attention。
    ///
    /// `recent_keys`/`recent_values`：本批 token 的 KV（[rows, kv_width]，行优先）。
    /// `compressed_*`：本批 + 历史 compressed KV（行优先）。每行 query 只看到
    /// position ≤ 自身的 recent 行，以及 position ≤ 自身的 compressed 行。
    /// indexer 选择（c4a 层）按每行 query 独立 top-k。
    #[allow(clippy::too_many_arguments)]
    pub fn attend_batch_f32(
        &self,
        queries: &[f32],
        rows: usize,
        recent_positions: &[usize],
        causal_batch: bool,
        recent_keys: &[f32],
        recent_values: &[f32],
        compressed_positions: &[usize],
        compressed_keys: &[f32],
        compressed_values: &[f32],
        compressed_index_keys: Option<&[f32]>,
        index_queries: Option<&[f32]>,
        index_head_weights: Option<&[f32]>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        indexer: Option<DsaSpec>,
        sink: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        if num_heads == 0
            || num_kv_heads == 0
            || !num_heads.is_multiple_of(num_kv_heads)
            || head_dim == 0
            || queries.len() != rows * num_heads * head_dim
            || recent_positions.len() != rows
            || recent_keys.len() != rows * num_kv_heads * head_dim
            || recent_values.len() != recent_keys.len()
            || compressed_keys.len() != compressed_values.len()
            || !compressed_keys.len().is_multiple_of(num_kv_heads * head_dim)
            || compressed_positions.len() * num_kv_heads * head_dim != compressed_keys.len()
            || sink.is_some_and(|values| values.len() != num_heads)
        {
            return Err(format!("因果 CSA batch 维度非法: rows={rows} queries={} recent={} compressed={} heads={num_heads}/{num_kv_heads} head_dim={head_dim}", queries.len(), recent_keys.len(), compressed_keys.len(),));
        }
        let kv_width = num_kv_heads * head_dim;
        let compressed_rows = compressed_positions.len();
        let index_queries = index_queries.unwrap_or(&[]);
        let compressed_index_keys = compressed_index_keys.unwrap_or(&[]);
        let index_head_weights = index_head_weights.unwrap_or(&[]);
        let indexer = match indexer {
            Some(indexer) => {
                if indexer.top_k == 0
                    || index_queries.len() != rows * indexer.num_heads * indexer.head_dim
                    || index_head_weights.len() != rows * indexer.num_heads
                    || (compressed_rows > 0 && compressed_index_keys.len() != compressed_rows * indexer.head_dim)
                {
                    return Err(format!("因果 CSA indexer 维度非法: top_k={} query={} weights={} keys={}", indexer.top_k, index_queries.len(), index_head_weights.len(), compressed_index_keys.len(),));
                }
                Some(indexer)
            }
            None => None,
        };
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0; queries.len()];
        for row in 0..rows {
            let query_position = recent_positions[row];
            let visible_position = if causal_batch { query_position } else { *recent_positions.last().expect("rows 已非零") };
            let visible_compressed: Vec<usize> = compressed_positions.iter().enumerate().filter_map(|(index, &position)| (position <= visible_position).then_some(index)).collect();
            // c4a 层按 indexer top-k 进一步筛选
            let selected: Vec<usize> = match indexer {
                Some(indexer) => {
                    let query_width = indexer.num_heads * indexer.head_dim;
                    let iq = &index_queries[row * query_width..(row + 1) * query_width];
                    let weights = &index_head_weights[row * indexer.num_heads..(row + 1) * indexer.num_heads];
                    let mut keys = Vec::with_capacity(visible_compressed.len() * indexer.head_dim);
                    for &index in &visible_compressed {
                        keys.extend_from_slice(&compressed_index_keys[index * indexer.head_dim..(index + 1) * indexer.head_dim]);
                    }
                    topk_indexer_f32(iq, &keys, weights, &indexer, indexer.top_k)?.into_iter().map(|local| visible_compressed[local]).collect()
                }
                None => visible_compressed,
            };
            for head in 0..num_heads {
                let kv_head = head * num_kv_heads / num_heads;
                let q = &queries[row * num_heads * head_dim + head * head_dim..row * num_heads * head_dim + (head + 1) * head_dim];
                let mut scores: Vec<f32> = Vec::with_capacity(selected.len() + self.recent.len() + rows);
                let mut value_refs: Vec<&[f32]> = Vec::with_capacity(scores.capacity());
                // compressed（selected 子集）
                for &index in &selected {
                    let key = &compressed_keys[index * kv_width + kv_head * head_dim..index * kv_width + (kv_head + 1) * head_dim];
                    scores.push(q.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scale);
                    let value = &compressed_values[index * kv_width + kv_head * head_dim..index * kv_width + (kv_head + 1) * head_dim];
                    value_refs.push(value);
                }
                // 上一批保留的滑窗 recent。
                for recent in &self.recent {
                    if recent.position > query_position || query_position - recent.position >= self.window_size {
                        continue;
                    }
                    let key = &recent.key[kv_head * head_dim..(kv_head + 1) * head_dim];
                    scores.push(q.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scale);
                    let value = &recent.value[kv_head * head_dim..(kv_head + 1) * head_dim];
                    value_refs.push(value);
                }
                // 普通 prefill 只见当前行以前；block 模式完整读取本批 KV。
                for recent_row in 0..rows {
                    let key_position = recent_positions[recent_row];
                    if (causal_batch && key_position > query_position) || (key_position <= query_position && query_position - key_position >= self.window_size) {
                        continue;
                    }
                    let key = &recent_keys[recent_row * kv_width + kv_head * head_dim..recent_row * kv_width + (kv_head + 1) * head_dim];
                    scores.push(q.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scale);
                    let value = &recent_values[recent_row * kv_width + kv_head * head_dim..recent_row * kv_width + (kv_head + 1) * head_dim];
                    value_refs.push(value);
                }
                let sink_score = sink.map(|values| values[head]);
                let max = scores.iter().copied().chain(sink_score).fold(f32::NEG_INFINITY, f32::max);
                let mut denominator = sink_score.map_or(0.0, |value| (value - max).exp());
                let weights: Vec<f32> = scores
                    .iter()
                    .map(|score| {
                        let weight = (*score - max).exp();
                        denominator += weight;
                        weight
                    })
                    .collect();
                if !denominator.is_finite() || denominator == 0.0 {
                    continue; // 非有限分母:输出保持 0 而非传播 NaN(batch 路径同判)
                }
                let out = &mut output[row * num_heads * head_dim + head * head_dim..row * num_heads * head_dim + (head + 1) * head_dim];
                for (weight, value) in weights.into_iter().zip(value_refs) {
                    for (out, value) in out.iter_mut().zip(value) {
                        *out += weight / denominator * value;
                    }
                }
            }
        }
        Ok(output)
    }
}

/// 压缩稀疏注意力的窄后端能力。状态的物理位置和选择实现由 backend 拥有。
pub trait CompressedSparseKernel: Backend {
    type CompressedKvStorage;

    fn allocate_compressed_kv(&self, spec: &CompressedSparseAttentionSpec) -> Result<Self::CompressedKvStorage, BackendError>;

    /// 每个 attention head 独立做普通 RMSNorm。V4 的 q_b 输出在 RoPE 前使用。
    fn compressed_rmsnorm_heads(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError>;

    /// 只把已经完成 norm/RoPE 的 recent K/V 写入滑窗 cache，不执行 query 与 attention。
    /// draft 模型初始化自己的 KV 时复用该能力，避免为整段 prompt 重跑完整 attention。
    fn compressed_sparse_store_recent(&self, storage: &mut Self::CompressedKvStorage, positions: &[usize], key: &Self::Tensor, value: &Self::Tensor) -> Result<(), BackendError>;

    /// 接收 runtime 已完成的 KV/gate 投影，保留未满窗口与 overlap 状态，并返回
    /// 本次刚闭合的压缩项。压缩项已经完成 weighted RMSNorm 与 suffix RoPE。
    #[allow(clippy::too_many_arguments)]
    fn compress_gated(
        &self,
        storage: &mut Self::CompressedKvStorage,
        stream: CompressionStream,
        positions: &[usize],
        kv: &Self::Tensor,
        gate: &Self::Tensor,
        position_bias: &Self::Weight,
        norm: &Self::Weight,
        compression: KvCompressionSpec,
        width: usize,
        rotary_dim: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
    ) -> Result<CompressedBatch<Self::Tensor>, BackendError>;

    /// 拼接后的投影一次进入 backend；每段只携带独立状态与绝对位置。
    #[allow(clippy::too_many_arguments)]
    fn compress_gated_segmented(
        &self,
        segments: &mut [CompressedGatedSegment<'_, Self>],
        stream: CompressionStream,
        kv: &Self::Tensor,
        gate: &Self::Tensor,
        position_bias: &Self::Weight,
        norm: &Self::Weight,
        compression: KvCompressionSpec,
        width: usize,
        rotary_dim: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
    ) -> Result<Vec<CompressedBatch<Self::Tensor>>, BackendError>
    where
        Self: Sized;

    #[allow(clippy::too_many_arguments)]
    fn compressed_sparse_decode(
        &self,
        storage: &mut Self::CompressedKvStorage,
        position: usize,
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        compressed_key: Option<&Self::Tensor>,
        compressed_value: Option<&Self::Tensor>,
        compressed_index_key: Option<&Self::Tensor>,
        index_query: Option<&Self::Tensor>,
        index_head_weights: Option<&Self::Tensor>,
        sink: Option<&Self::Weight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<Self::Tensor, BackendError>;

    /// Prefill 一批 token 的压缩稀疏注意力。
    ///
    /// runtime 已完成 query/kv/compressor/indexer 投影：
    /// - `query`/`key`/`value`：[rows] 个 token 的 attention Q/K/V（已 RoPE）
    /// - `compressed_*`：runtime 按 ratio 分组压缩后的历史 KV（[compressed_rows]），
    ///   以及 indexer 用的 index_key/query（仅 c4a 层）
    /// - `positions`：每个 query 的绝对位置；`causal_batch=false` 时本批 KV 全可见
    ///
    /// backend 负责把 recent KV 写入 cache（按滑窗语义滚动）、把 compressed KV 写入
    /// 压缩历史，并按 `causal_batch` 控制本批 KV 的可见范围。
    #[allow(clippy::too_many_arguments)]
    fn compressed_sparse_prefill(
        &self,
        storage: &mut Self::CompressedKvStorage,
        positions: &[usize],
        causal_batch: bool,
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        compressed_positions: Option<&[usize]>,
        compressed_key: Option<&Self::Tensor>,
        compressed_value: Option<&Self::Tensor>,
        compressed_index_key: Option<&Self::Tensor>,
        index_query: Option<&Self::Tensor>,
        index_head_weights: Option<&Self::Tensor>,
        sink: Option<&Self::Weight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<Self::Tensor, BackendError>;

    /// 多 session 的 cache 边界由 segment 描述，backend 决定一次 kernel 还是逐段 oracle。
    #[allow(clippy::too_many_arguments)]
    fn compressed_sparse_prefill_segmented(
        &self,
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        index_query: Option<&Self::Tensor>,
        index_head_weights: Option<&Self::Tensor>,
        segments: &mut [CompressedSparsePrefillSegment<'_, Self>],
        sink: Option<&Self::Weight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<Self::Tensor, BackendError>
    where
        Self: Sized;
}

#[allow(clippy::too_many_arguments)]
pub fn compress_gated_segmented_fallback<B>(
    backend: &B,
    segments: &mut [CompressedGatedSegment<'_, B>],
    stream: CompressionStream,
    kv: &B::Tensor,
    gate: &B::Tensor,
    position_bias: &B::Weight,
    norm: &B::Weight,
    compression: KvCompressionSpec,
    width: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
    eps: f32,
) -> Result<Vec<CompressedBatch<B::Tensor>>, BackendError>
where
    B: CompressedSparseKernel + SegmentedTensorBackend,
{
    let rows = segments.iter().try_fold(0usize, |rows, segment| rows.checked_add(segment.positions.len()).ok_or_else(|| BackendError::Compute { msg: "segmented compressor rows 溢出".to_owned() }))?;
    if segments.is_empty() || segments.iter().any(|segment| segment.positions.is_empty()) || backend.token_rows(kv) != rows || backend.token_rows(gate) != rows {
        return Err(BackendError::Compute { msg: format!("segmented compressor shape 非法: segments={} rows={rows} kv={} gate={}", segments.len(), backend.token_rows(kv), backend.token_rows(gate)) });
    }
    let mut offset = 0;
    let mut outputs = Vec::with_capacity(segments.len());
    for segment in segments {
        let segment_rows = segment.positions.len();
        let kv = backend.slice_token_rows(kv, offset, segment_rows)?;
        let gate = backend.slice_token_rows(gate, offset, segment_rows)?;
        outputs.push(backend.compress_gated(segment.storage, stream, segment.positions, &kv, &gate, position_bias, norm, compression, width, rotary_dim, cos, sin, eps)?);
        offset += segment_rows;
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
pub fn compressed_sparse_prefill_segmented_fallback<B>(
    backend: &B,
    query: &B::Tensor,
    key: &B::Tensor,
    value: &B::Tensor,
    index_query: Option<&B::Tensor>,
    index_head_weights: Option<&B::Tensor>,
    segments: &mut [CompressedSparsePrefillSegment<'_, B>],
    sink: Option<&B::Weight>,
    spec: &CompressedSparseAttentionSpec,
) -> Result<B::Tensor, BackendError>
where
    B: CompressedSparseKernel + SegmentedTensorBackend,
{
    let rows = segments.iter().try_fold(0usize, |rows, segment| rows.checked_add(segment.positions.len()).ok_or_else(|| BackendError::Compute { msg: "segmented CSA rows 溢出".to_owned() }))?;
    if segments.is_empty() || segments.iter().any(|segment| segment.positions.is_empty()) || [backend.token_rows(query), backend.token_rows(key), backend.token_rows(value)].into_iter().any(|actual| actual != rows) {
        return Err(BackendError::Compute { msg: format!("segmented CSA shape 非法: segments={} rows={rows} query={} key={} value={}", segments.len(), backend.token_rows(query), backend.token_rows(key), backend.token_rows(value)) });
    }
    let mut offset = 0;
    let mut outputs = Vec::with_capacity(segments.len());
    for segment in segments {
        let segment_rows = segment.positions.len();
        let query = backend.slice_token_rows(query, offset, segment_rows)?;
        let key = backend.slice_token_rows(key, offset, segment_rows)?;
        let value = backend.slice_token_rows(value, offset, segment_rows)?;
        let index_query = index_query.map(|tensor| backend.slice_token_rows(tensor, offset, segment_rows)).transpose()?;
        let index_head_weights = index_head_weights.map(|tensor| backend.slice_token_rows(tensor, offset, segment_rows)).transpose()?;
        outputs.push(backend.compressed_sparse_prefill(
            segment.storage,
            segment.positions,
            segment.causal_batch,
            &query,
            &key,
            &value,
            segment.compressed_positions,
            segment.compressed_key,
            segment.compressed_value,
            segment.compressed_index_key,
            index_query.as_ref(),
            index_head_weights.as_ref(),
            sink,
            spec,
        )?);
        offset += segment_rows;
    }
    let outputs = outputs.iter().collect::<Vec<_>>();
    backend.concat_token_rows(&outputs)
}

fn validate_kv(key: &[f32], value: &[f32]) -> Result<(), String> {
    if key.is_empty() || key.len() != value.len() {
        return Err(format!("压缩 KV 维度非法: key={} value={}", key.len(), value.len()));
    }
    Ok(())
}

/// 对一个 compression group 做学习式 gated pooling；overlap 的窗口推进由调用方决定。
fn gated_pool_f32(rows: &[f32], gate_logits: &[f32], width: usize) -> Result<Vec<f32>, String> {
    if width == 0 || rows.is_empty() || rows.len() != gate_logits.len() || !rows.len().is_multiple_of(width) || gate_logits.iter().any(|value| value.is_nan()) {
        return Err(format!("压缩 KV gated pooling 维度非法: rows={} gates={} width={width}", rows.len(), gate_logits.len(),));
    }
    let slots = rows.len() / width;
    let mut output = vec![0.0; width];
    for column in 0..width {
        let max = (0..slots).map(|slot| gate_logits[slot * width + column]).fold(f32::NEG_INFINITY, f32::max);
        let denominator = (0..slots).map(|slot| (gate_logits[slot * width + column] - max).exp()).sum::<f32>();
        if denominator == 0.0 || !denominator.is_finite() {
            return Err(format!("压缩 KV gated pooling column={column} softmax 分母非法: {denominator}"));
        }
        for slot in 0..slots {
            output[column] += rows[slot * width + column] * (gate_logits[slot * width + column] - max).exp() / denominator;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn normalize_rope_compressed_f32(values: &[f32], visible_positions: &[usize], ratio: usize, width: usize, norm: &[f32], eps: f32, rotary_dim: usize, cos: &[f32], sin: &[f32]) -> Result<Vec<f32>, String> {
    if values.len() != visible_positions.len() * width || norm.len() != width || ratio == 0 {
        return Err(format!("压缩 KV norm/RoPE 维度非法: values={} positions={} ratio={ratio} width={width} norm={}", values.len(), visible_positions.len(), norm.len()));
    }
    let mut output = Vec::with_capacity(values.len());
    for (row, &visible_position) in values.chunks_exact(width).zip(visible_positions) {
        let variance = row.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let scale = (variance + eps).sqrt().recip();
        let normalized = row.iter().zip(norm).map(|(value, weight)| value * scale * weight).collect::<Vec<_>>();
        // 压缩项的 rope 起点 = 窗口第一个源 token；visible_position + 1 < ratio
        // 说明调用方传入的可见位置与压缩率矛盾，直接减法会下溢。
        let rope_position = (visible_position + 1).checked_sub(ratio).ok_or_else(|| format!("压缩 KV rope 位置下溢: visible_position={visible_position} ratio={ratio}"))?;
        output.extend(crate::attention::rope::apply_f32(&normalized, 1, width, 1, rotary_dim, rope_position, cos, sin, crate::attention::rope::RotaryLayout::Interleaved, crate::attention::rope::RotaryPlacement::Suffix)?);
    }
    Ok(output)
}

#[cfg(test)]
fn topk_dot_f32(query: &[f32], keys: &[f32], top_k: usize) -> Result<Vec<usize>, String> {
    if query.is_empty() || top_k == 0 || !keys.len().is_multiple_of(query.len()) {
        return Err(format!("压缩 KV indexer 维度非法: query={} keys={} top_k={top_k}", query.len(), keys.len(),));
    }
    let mut scores: Vec<(usize, f32)> = keys.chunks_exact(query.len()).enumerate().map(|(index, key)| (index, query.iter().zip(key).map(|(query, key)| query * key).sum())).collect();
    scores.sort_unstable_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal).then_with(|| left.0.cmp(&right.0)));
    scores.truncate(top_k.min(scores.len()));
    Ok(scores.into_iter().map(|(index, _)| index).collect())
}

fn topk_indexer_f32(query: &[f32], keys: &[f32], head_weights: &[f32], spec: &DsaSpec, top_k: usize) -> Result<Vec<usize>, String> {
    let query_width = spec.num_heads.checked_mul(spec.head_dim).ok_or("indexer query 宽度溢出")?;
    if query.len() != query_width || head_weights.len() != spec.num_heads || spec.head_dim == 0 || top_k == 0 || !keys.len().is_multiple_of(spec.head_dim) {
        return Err(format!("V4 indexer 维度非法: query={} keys={} weights={} heads={} dim={} top_k={top_k}", query.len(), keys.len(), head_weights.len(), spec.num_heads, spec.head_dim));
    }
    let dot_scale = (spec.head_dim as f32).sqrt().recip();
    let weight_scale = (spec.num_heads as f32).sqrt().recip();
    let mut scores = keys
        .chunks_exact(spec.head_dim)
        .enumerate()
        .map(|(index, key)| {
            let score: f32 = (0..spec.num_heads)
                .map(|head| {
                    let q = &query[head * spec.head_dim..(head + 1) * spec.head_dim];
                    let dot = q.iter().zip(key).map(|(query, key)| query * key).sum::<f32>();
                    head_weights[head] * weight_scale * dot.max(0.0) * dot_scale
                })
                .sum();
            (index, score)
        })
        .collect::<Vec<_>>();
    scores.sort_unstable_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    scores.truncate(top_k.min(scores.len()));
    Ok(scores.into_iter().map(|(index, _)| index).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_state跨调用提交窗口() {
        let mut state = CompressionState::default();
        let first = state.plan(&[0, 1], 4).unwrap();
        assert_eq!((first.pending_rows(), first.entry_start(), first.windows()), (0, 0, 0));
        state.commit(first, 2).unwrap();

        let second = state.plan(&[2, 3, 4, 5, 6, 7], 4).unwrap();
        assert_eq!((second.pending_rows(), second.entry_start(), second.windows()), (2, 0, 2));
        assert_eq!(second.visible_positions(), [3, 7]);
        state.commit(second, 0).unwrap();
        assert!(state.plan(&[9], 4).is_err());
    }

    #[test]
    fn csa_cache_keeps_window_and_compressed_history() {
        let spec = CompressedSparseAttentionSpec {
            num_heads: 64,
            num_kv_heads: 1,
            head_dim: 512,
            q_lora_rank: 1024,
            output_groups: 8,
            output_lora_rank: 1024,
            window_size: 128,
            rope: RopeSpec::Default { rotary_dim: 64, theta: 10_000.0 },
            compression: Some(KvCompressionSpec {
                ratio: 4,
                overlap: true,
                selection: CompressedSelection::LearnedIndexer(DsaSpec { num_heads: 64, head_dim: 128, rope_dim: 64, top_k: 512, rotary_layout: crate::attention::rope::RotaryLayout::Interleaved, kpool: 0, always_select_tail: false }),
            }),
            attention_sink: true,
        };
        spec.validate().unwrap();
        assert_eq!(spec.cache_rows(1_024).unwrap(), 384);
    }

    #[test]
    fn reference_state_keeps_recent_window_and_selected_history() {
        let mut state = CompressedKvState::new(2).unwrap();
        state.push_compressed(0, &[1.0, 0.0], &[3.0, 0.0]).unwrap();
        state.push_compressed(1, &[0.0, 1.0], &[0.0, 7.0]).unwrap();
        state.push_recent(2, &[1.0, 0.0], &[5.0, 0.0]).unwrap();
        state.push_recent(3, &[1.0, 0.0], &[9.0, 0.0]).unwrap();
        state.push_recent(4, &[1.0, 0.0], &[11.0, 0.0]).unwrap();
        assert_eq!(state.recent_len(), 2);
        assert_eq!(state.compressed_len(), 2);
        let output = state.attend_f32(&[1.0, 0.0], 1, 1, 2, Some(&[0]), None).unwrap();
        assert!(output[0] > 3.0 && output[0] < 11.0);
        assert_eq!(output[1], 0.0);
    }

    #[test]
    fn indexer_topk_is_stable_for_equal_scores() {
        assert_eq!(topk_dot_f32(&[1.0, 0.0], &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0], 2).unwrap(), [0, 1]);
    }

    #[test]
    fn v4_indexer使用逐头relu与学习权重() {
        let spec = DsaSpec { num_heads: 2, head_dim: 2, rope_dim: 0, top_k: 1, rotary_layout: crate::attention::rope::RotaryLayout::Interleaved, kpool: 0, always_select_tail: false };
        let query = [1.0, 0.0, 0.0, 2.0];
        let keys = [2.0, 0.0, 0.0, 2.0];
        assert_eq!(topk_indexer_f32(&query, &keys, &[2.0, 0.1], &spec, 1).unwrap(), [0]);
        assert_eq!(topk_indexer_f32(&query, &keys, &[0.1, 2.0], &spec, 1).unwrap(), [1]);
    }

    #[test]
    fn causal_batch_attention_only_sees_past_positions() {
        let queries = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let positions = vec![0, 1, 2];
        let recent_keys = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let recent_values = vec![10.0, 0.0, 20.0, 0.0, 0.0, 30.0];
        let output = CompressedKvState::new(4).unwrap().attend_batch_f32(&queries, 3, &positions, true, &recent_keys, &recent_values, &[], &[], &[], None, None, None, 1, 1, 2, None, None).unwrap();
        assert!((output[0] - 10.0).abs() < 1e-4 && output[1].abs() < 1e-4);
        assert!((output[2] - 15.0).abs() < 1e-4 && output[3].abs() < 1e-4);
        assert!((output[4] - 12.03).abs() < 0.5 && (output[5] - 5.94).abs() < 0.5);
    }

    #[test]
    fn block_attention_sees_all_current_positions() {
        let queries = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let positions = vec![0, 1, 2];
        let keys = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let values = vec![10.0, 0.0, 20.0, 0.0, 30.0, 0.0];
        let output = CompressedKvState::new(4).unwrap().attend_batch_f32(&queries, 3, &positions, false, &keys, &values, &[], &[], &[], None, None, None, 1, 1, 2, None, None).unwrap();
        for row in 0..3 {
            assert!((output[row * 2] - 20.0).abs() < 1e-4 && output[row * 2 + 1].abs() < 1e-4);
        }
    }

    #[test]
    fn 后续prefill继续读取上一批recent滑窗() {
        let mut state = CompressedKvState::new(4).unwrap();
        state.push_recent(0, &[1.0, 0.0], &[10.0, 0.0]).unwrap();
        let output = state.attend_batch_f32(&[1.0, 0.0], 1, &[1], true, &[1.0, 0.0], &[20.0, 0.0], &[], &[], &[], None, None, None, 1, 1, 2, None, None).unwrap();
        assert!((output[0] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn causal_batch_attention_uses_compressed_history() {
        let output = CompressedKvState::new(4)
            .unwrap()
            .attend_batch_f32(
                &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                3,
                &[0, 1, 2],
                true,
                &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                &[1, 2],
                &[1.0, 0.0, 1.0, 0.0],
                &[100.0, 0.0, 200.0, 0.0],
                None,
                None,
                None,
                1,
                1,
                2,
                None,
                None,
            )
            .unwrap();
        assert!((output[0] - 1.0).abs() < 1e-4);
        assert!((output[2] - 34.0).abs() < 1.0 && output[3].abs() < 1e-4);
        assert!((output[4] - 60.6).abs() < 1.0 && output[5].abs() < 1e-4);
    }

    #[test]
    fn batch_update_keeps_last_window_and_all_compressed() {
        let mut state = CompressedKvState::new(2).unwrap();
        state.push_recent_batch(&[0, 1, 2, 3], &[1.0; 8], &[2.0; 8], 2).unwrap();
        assert_eq!(state.recent_len(), 2);
        state.push_compressed_batch(&[0, 4], &[3.0; 4], &[4.0; 4], 2).unwrap();
        assert_eq!(state.compressed_len(), 2);
    }

    #[test]
    fn learned_indexer_accepts_empty_compressed_history() {
        let indexer = DsaSpec { num_heads: 1, head_dim: 3, rope_dim: 0, top_k: 1, rotary_layout: crate::attention::rope::RotaryLayout::Interleaved, kpool: 0, always_select_tail: false };
        let output = CompressedKvState::new(4).unwrap().attend_batch_f32(&[1.0, 0.0], 1, &[0], true, &[1.0, 0.0], &[3.0, 4.0], &[], &[], &[], Some(&[]), Some(&[1.0, 2.0, 3.0]), Some(&[1.0]), 1, 1, 2, Some(indexer), None).unwrap();
        assert_eq!(output, [3.0, 4.0]);
    }

    #[test]
    fn csa_overlap跨调用保留上一窗口ca() {
        let mut state = GatedPoolState::default();
        let bias = vec![0.0; 8];
        let (positions, first) = state.push_f32(&[0, 1, 2, 3], &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], &[0.0; 8], &bias, 4, 1, true).unwrap();
        assert_eq!(positions, [3]);
        assert!((first[0] - 25.0).abs() < 1e-5);
        let (positions, second) = state.push_f32(&[4, 5, 6, 7], &[5.0, 50.0, 6.0, 60.0, 7.0, 70.0, 8.0, 80.0], &[0.0; 8], &bias, 4, 1, true).unwrap();
        assert_eq!(positions, [7]);
        assert!((second[0] - 33.75).abs() < 1e-5);
    }
}
