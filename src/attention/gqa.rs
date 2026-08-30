//! 分组查询注意力(GQA)规格。MiniMax-M3、LLaMA 系列使用。

use std::ops::Range;

use rayon::prelude::*;

use crate::attention::rope::RopeSpec;

/// GQA 的纯几何规格，不包含位置编码、窗口或模型投影策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaGeometry {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GqaGeometry {
    pub fn validate(&self) -> Result<(), String> {
        if self.num_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 || !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(format!("GQA geometry 非法: query_heads={} kv_heads={} head_dim={}", self.num_heads, self.num_kv_heads, self.head_dim,));
        }
        Ok(())
    }

    pub fn group_size(&self) -> usize {
        debug_assert!(self.num_heads.is_multiple_of(self.num_kv_heads));
        self.num_heads / self.num_kv_heads
    }

    pub fn query_columns(&self) -> Result<usize, String> {
        self.num_heads.checked_mul(self.head_dim).ok_or_else(|| "GQA query columns 溢出".to_owned())
    }

    pub fn kv_columns(&self) -> Result<usize, String> {
        self.num_kv_heads.checked_mul(self.head_dim).ok_or_else(|| "GQA KV columns 溢出".to_owned())
    }
}

/// 单个 query 的 causal 可见范围。区间使用绝对 token 位置，右边界不包含。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalWindow {
    Full,
    Sliding { size: usize },
}

impl CausalWindow {
    pub fn validate(&self) -> Result<(), String> {
        if matches!(self, Self::Sliding { size: 0 }) {
            return Err("sliding attention window 不能为 0".to_owned());
        }
        Ok(())
    }

    pub fn key_range(&self, query_position: usize, key_end: usize) -> Result<Range<usize>, String> {
        let causal_end = query_position.checked_add(1).ok_or_else(|| "attention query position 溢出".to_owned())?.min(key_end);
        let start = match self {
            Self::Full => 0,
            Self::Sliding { size } => query_position.saturating_add(1).saturating_sub(*size),
        }
        .min(causal_end);
        Ok(start..causal_end)
    }

    /// 多模态视觉块允许当前 query 读到同块末尾；左边界仍由 query 的滑窗位置决定。
    pub fn key_range_with_visible_end(&self, query_position: usize, key_end: usize, visible_end: usize) -> Result<Range<usize>, String> {
        let causal = self.key_range(query_position, key_end)?;
        Ok(causal.start..visible_end.max(causal.end).min(key_end))
    }

    pub fn cache_capacity(&self, max_sequence_len: usize) -> usize {
        match self {
            Self::Full => max_sequence_len,
            Self::Sliding { size } => (*size).min(max_sequence_len),
        }
    }
}

/// K/V 投影关系。`KeyAsValue` 只共享线性投影结果；K/V 后续 norm、RoPE 与 cache 仍独立。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GqaKvProjection {
    Separate,
    KeyAsValue,
}

/// 一层 hybrid GQA 的完整平台无关 attention 规格。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridGqaLayerSpec {
    pub geometry: GqaGeometry,
    pub rope: RopeSpec,
    pub window: CausalWindow,
    /// QK 点积使用的显式缩放。Gemma 4 为 1.0，不使用常见的 1/sqrt(head_dim)。
    pub score_scale: f32,
    pub kv_projection: GqaKvProjection,
}

impl HybridGqaLayerSpec {
    pub fn validate(&self) -> Result<(), String> {
        self.geometry.validate()?;
        self.rope.validate()?;
        self.window.validate()?;
        if self.rope.rotary_dim() > self.geometry.head_dim {
            return Err(format!("RoPE dim {} 超过 GQA head_dim {}", self.rope.rotary_dim(), self.geometry.head_dim));
        }
        if !self.score_scale.is_finite() || self.score_scale <= 0.0 {
            return Err(format!("GQA score_scale {} 非法", self.score_scale));
        }
        Ok(())
    }
}

/// 同一模型中逐层变化的 GQA 规格。backend 按 layer 读取能力，不感知模型名称。
#[derive(Debug, Clone)]
pub struct HybridGqaSpec {
    layers: Vec<HybridGqaLayerSpec>,
}

impl HybridGqaSpec {
    pub fn new(layers: Vec<HybridGqaLayerSpec>) -> Result<Self, String> {
        if layers.is_empty() {
            return Err("HybridGqaSpec 至少需要一层".to_owned());
        }
        for (layer, spec) in layers.iter().enumerate() {
            spec.validate().map_err(|error| format!("hybrid GQA L{layer}: {error}"))?;
        }
        Ok(Self { layers })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, layer: usize) -> Result<&HybridGqaLayerSpec, String> {
        self.layers.get(layer).ok_or_else(|| format!("hybrid GQA layer {layer} 越界(共 {} 层)", self.layers.len()))
    }

    pub fn layers(&self) -> &[HybridGqaLayerSpec] {
        &self.layers
    }
}

/// GQA 规格。`num_kv_heads` 个 KV head 被 `num_heads` 个 query head 共享,
/// 组大小 `G = num_heads / num_kv_heads`。
#[derive(Debug, Clone, Copy)]
pub struct GqaSpec {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    /// 是否对 Q/K 每 head 独立做 RMSNorm(MiniMax-M3 用)。
    pub use_qk_norm: bool,
    pub window: CausalWindow,
    pub score_scale: f32,
    /// attention 输出是否先乘 `sigmoid(gate)` 再进入输出投影。
    pub output_gate: bool,
}

impl GqaSpec {
    pub fn geometry(&self) -> GqaGeometry {
        GqaGeometry { num_heads: self.num_heads, num_kv_heads: self.num_kv_heads, head_dim: self.head_dim }
    }

    /// 每 KV head 服务多少个 query head。
    pub fn group_size(&self) -> usize {
        debug_assert!(self.num_heads.is_multiple_of(self.num_kv_heads), "GQA head 数必须能被 KV head 数整除");
        self.num_heads / self.num_kv_heads
    }
}

/// 带绝对 source 区间和显式 score scale 的单 head reference。
#[allow(clippy::too_many_arguments)]
pub fn causal_head_range_f32<D>(query: &[f32], key: &[f32], value: &[f32], key_value_stride: usize, source_range: Range<usize>, score_scale: f32, output: &mut [f32], dot: D) -> Result<(), String>
where
    D: Fn(&[f32], &[f32]) -> f32,
{
    let head_dim = query.len();
    if head_dim == 0 || output.len() != head_dim || source_range.is_empty() || key_value_stride < head_dim || !score_scale.is_finite() || score_scale <= 0.0 {
        return Err(format!("GQA head 参数非法: query={} output={} stride={key_value_stride} range={source_range:?} scale={score_scale}", query.len(), output.len(),));
    }
    let required = (source_range.end - 1).checked_mul(key_value_stride).and_then(|offset| offset.checked_add(head_dim)).ok_or("GQA KV 长度溢出")?;
    if key.len() < required || value.len() < required {
        return Err(format!("GQA KV 长度不足: key={} value={}，至少需要 {required}", key.len(), value.len()));
    }

    let mut scores = Vec::with_capacity(source_range.len());
    let mut maximum = f32::NEG_INFINITY;
    for source in source_range.clone() {
        let start = source * key_value_stride;
        let score = dot(query, &key[start..start + head_dim]) * score_scale;
        maximum = maximum.max(score);
        scores.push(score);
    }
    let denominator = scores
        .iter_mut()
        .map(|score| {
            *score = (*score - maximum).exp();
            *score
        })
        .sum::<f32>();
    output.fill(0.0);
    if !denominator.is_finite() || denominator == 0.0 {
        // 非有限分母：输出保持 0 而非传播 NaN（与压缩稀疏 attention 同判）。
        return Ok(());
    }
    for (source, score) in source_range.zip(scores) {
        let probability = score / denominator;
        let start = source * key_value_stride;
        for column in 0..head_dim {
            output[column] += probability * value[start + column];
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(crate) fn prefill_attention_at_f32(query: &[f32], key: &[f32], value: &[f32], query_tokens: usize, kv_start: usize, kv_end: usize, position: usize, spec: &GqaSpec, output: &mut [f32]) -> Result<(), String> {
    prefill_attention_at_visible_f32(query, key, value, query_tokens, kv_start, kv_end, position, spec, None, output)
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(crate) fn prefill_attention_at_visible_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    query_tokens: usize,
    kv_start: usize,
    kv_end: usize,
    position: usize,
    spec: &GqaSpec,
    visible_ends: Option<&[u32]>,
    output: &mut [f32],
) -> Result<(), String> {
    let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or("GQA query columns 溢出")?;
    if query_tokens.checked_mul(query_cols) != Some(query.len()) {
        return Err(format!("GQA query_tokens={query_tokens} 与 query.len={}/query_cols={query_cols} 不一致", query.len()));
    }
    prefill_attention_at_visible_with_dot_f32(query, key, value, kv_start, kv_end, position, spec, visible_ends, output, |left, right| left.iter().zip(right).map(|(left, right)| left * right).sum())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_attention_at_visible_with_dot_f32<D>(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    kv_start: usize,
    kv_end: usize,
    position: usize,
    spec: &GqaSpec,
    visible_ends: Option<&[u32]>,
    output: &mut [f32],
    dot: D,
) -> Result<(), String>
where
    D: Fn(&[f32], &[f32]) -> f32 + Sync,
{
    let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or("GQA query columns 溢出")?;
    let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or("GQA KV columns 溢出")?;
    if query_cols == 0 || !query.len().is_multiple_of(query_cols) {
        return Err(format!("GQA query.len={} 不能按 query_cols={query_cols} 分行", query.len()));
    }
    let query_tokens = query.len() / query_cols;
    let output_len = query_tokens.checked_mul(query_cols).ok_or("GQA output 长度溢出")?;
    if output.len() < output_len {
        return Err(format!("GQA output={}，至少需要 {output_len}", output.len()));
    }
    let group_size = spec.group_size();
    output[..output_len].par_chunks_mut(spec.head_dim).enumerate().try_for_each(|(item, out)| {
        let token = item / spec.num_heads;
        let head = item % spec.num_heads;
        let kv_head = head / group_size;
        let query_start = token * query_cols + head * spec.head_dim;
        let kv_head_start = kv_head * spec.head_dim;
        let absolute_range = match visible_ends {
            Some(ends) => spec.window.key_range_with_visible_end(position + token, kv_end, ends[token] as usize),
            None => spec.window.key_range(position + token, kv_end),
        }?;
        let absolute_start = absolute_range.start.max(kv_start);
        causal_head_range_f32(&query[query_start..query_start + spec.head_dim], &key[kv_head_start..], &value[kv_head_start..], kv_cols, absolute_start - kv_start..absolute_range.end - kv_start, spec.score_scale, out, &dot)
    })
}

/// 平台无关的单序列 causal GQA prefill reference。`tokens` 从 `query.len() / query_cols` 推。
#[cfg(test)]
pub fn reference_prefill(query: &[f32], key: &[f32], value: &[f32], spec: &GqaSpec) -> Result<Vec<f32>, String> {
    if spec.num_heads == 0 || spec.num_kv_heads == 0 || spec.head_dim == 0 || !spec.num_heads.is_multiple_of(spec.num_kv_heads) {
        return Err("GQA head 参数非法".to_owned());
    }
    let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or("GQA query cols 溢出")?;
    let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or("GQA KV cols 溢出")?;
    if query_cols == 0 || !query.len().is_multiple_of(query_cols) {
        return Err(format!("GQA query.len={} 不能按 query_cols={query_cols} 分行", query.len()));
    }
    let tokens = query.len() / query_cols;
    let kv_len = tokens.checked_mul(kv_cols).ok_or("GQA KV 长度溢出")?;
    if key.len() != kv_len || value.len() != kv_len {
        return Err(format!("GQA shape 不符: query={} key={} value={}，期望 KV {kv_len}", query.len(), key.len(), value.len()));
    }

    let group_size = spec.group_size();
    let mut output = vec![0.0; query.len()];
    for token in 0..tokens {
        for head in 0..spec.num_heads {
            let kv_head = head / group_size;
            let query_start = token * query_cols + head * spec.head_dim;
            let kv_start = kv_head * spec.head_dim;
            causal_head_range_f32(
                &query[query_start..query_start + spec.head_dim],
                &key[kv_start..],
                &value[kv_start..],
                kv_cols,
                spec.window.key_range(token, tokens)?,
                spec.score_scale,
                &mut output[query_start..query_start + spec.head_dim],
                |left, right| left.iter().zip(right).map(|(left, right)| left * right).sum(),
            )?;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_prefill_is_causal() {
        let spec = GqaSpec { num_heads: 1, num_kv_heads: 1, head_dim: 1, rope_dim: 1, rope_theta: 10_000.0, use_qk_norm: false, window: CausalWindow::Full, score_scale: 1.0, output_gate: false };
        let output = reference_prefill(&[0.0, 0.0], &[0.0, 0.0], &[2.0, 6.0], &spec).unwrap();
        assert_eq!(output, vec![2.0, 4.0]);
    }

    #[test]
    fn sliding_window_uses_absolute_causal_range() {
        assert_eq!(CausalWindow::Sliding { size: 3 }.key_range(4, 5).unwrap(), 2..5);
        assert_eq!(CausalWindow::Sliding { size: 3 }.key_range(1, 2).unwrap(), 0..2);
        assert_eq!(CausalWindow::Full.key_range(4, 5).unwrap(), 0..5);
        assert_eq!(CausalWindow::Sliding { size: 4 }.key_range_with_visible_end(2, 6, 5).unwrap(), 0..5);
    }
}
