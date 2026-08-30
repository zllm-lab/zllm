//! GLM-5.2 的 DSpark drafter 组合；block attention 与基础算子保持模型无关。

use crate::{
    attention::{block::BlockAttentionSpec, gqa::GqaGeometry, rope::RotaryLayout},
    backend::{BackendError, BlockAttentionBackend, GqaPrefillBackend, SegmentedTensorBackend},
    moe::Activation,
};

fn visible_ranges(context_rows: usize, query_rows: usize, sliding_window: Option<usize>, sliding_window_non_causal: bool) -> Vec<std::ops::Range<usize>> {
    (0..query_rows)
        .map(|row| {
            let start = sliding_window.map_or(0, |window| context_rows.saturating_sub(window));
            let end = if sliding_window.is_some() && !sliding_window_non_causal { context_rows + row + 1 } else { context_rows + query_rows };
            start..end
        })
        .collect()
}

#[derive(Clone)]
pub struct DsparkLayer<W> {
    pub input_norm: W,
    pub q_proj: W,
    pub q_norm: W,
    pub k_proj: W,
    pub k_norm: W,
    pub v_proj: W,
    pub o_proj: W,
    pub post_attention_norm: W,
    pub gate_proj: W,
    pub up_proj: W,
    pub down_proj: W,
    pub sliding_window: Option<usize>,
    pub sliding_window_non_causal: bool,
}

pub struct DsparkBackbone<W> {
    pub layers: Vec<DsparkLayer<W>>,
    pub output_norm: W,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
}

/// 只负责把 target hidden 投影为 drafter K/V。CPU proposal 模式下 GPU
/// 不执行 drafter backbone，因此只需常驻这三组权重，不能继续加载 Q/O/FFN/head。
pub struct DsparkTargetLayer<W> {
    pub k_proj: W,
    pub k_norm: W,
    pub v_proj: W,
    pub sliding_window: Option<usize>,
}

pub struct DsparkTargetProjector<W> {
    pub layers: Vec<DsparkTargetLayer<W>>,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
}

struct DsparkLayerTargetCache<T> {
    start_position: usize,
    key: T,
    value: T,
}

pub struct DsparkTargetCache<T> {
    layers: Vec<Option<DsparkLayerTargetCache<T>>>,
}

/// 设备无关的 target cache 快照；具体 tensor 编码由持久化边界决定。
#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
pub(crate) struct DsparkTargetCacheSnapshot<T> {
    pub layers: Vec<DsparkTargetLayerSnapshot<T>>,
}

#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
pub(crate) struct DsparkTargetLayerSnapshot<T> {
    pub start_position: usize,
    pub key: T,
    pub value: T,
}

pub struct DsparkTargetSegment<'a, T> {
    pub cache: &'a mut DsparkTargetCache<T>,
    pub hidden: &'a T,
    pub position: usize,
    pub block_position: usize,
}

impl<T> DsparkTargetCache<T> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// 已预热的 drafter 层数;等于 backbone 层数才可 draft。
    pub fn warmed_layers(&self) -> usize {
        self.layers.len()
    }

    /// 会话结束或无法保留事务前缀时释放全部 target KV；权重不属于此生命周期。
    pub fn reset_session(&mut self) {
        self.layers.clear();
    }

    /// 统计 cache 持有的真实 tensor allocation，而不是当前逻辑行数。
    pub fn allocated_bytes<B>(&self, backend: &B) -> u64
    where
        B: crate::backend::BackendResources<Tensor = T>,
    {
        self.layers.iter().flatten().fold(0_u64, |bytes, layer| bytes.saturating_add(backend.tensor_allocated_bytes(&layer.key)).saturating_add(backend.tensor_allocated_bytes(&layer.value)))
    }

    /// 只有每层都已预热时才能形成持久化快照，避免恢复出半初始化 cache。
    #[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
    pub(crate) fn try_snapshot<U, E>(&self, mut map: impl FnMut(&T) -> Result<U, E>) -> Result<Option<DsparkTargetCacheSnapshot<U>>, E> {
        if self.layers.is_empty() || self.layers.iter().any(Option::is_none) {
            return Ok(None);
        }
        let layers = self.layers.iter().flatten().map(|layer| Ok(DsparkTargetLayerSnapshot { start_position: layer.start_position, key: map(&layer.key)?, value: map(&layer.value)? })).collect::<Result<Vec<_>, E>>()?;
        Ok(Some(DsparkTargetCacheSnapshot { layers }))
    }

    #[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
    pub(crate) fn from_snapshot(snapshot: DsparkTargetCacheSnapshot<T>) -> Self {
        Self { layers: snapshot.layers.into_iter().map(|layer| Some(DsparkLayerTargetCache { start_position: layer.start_position, key: layer.key, value: layer.value })).collect() }
    }

    /// 所有层必须覆盖同一 target 区间；不完整或 shape 不一致时返回 None。
    pub(crate) fn covered_range<B>(&self, backend: &B) -> Option<std::ops::Range<usize>>
    where
        B: crate::backend::BackendResources<Tensor = T>,
    {
        let first = self.layers.first()?.as_ref()?;
        let rows = backend.token_rows(&first.key);
        if rows == 0 || backend.token_rows(&first.value) != rows {
            return None;
        }
        let end = first.start_position.checked_add(rows)?;
        self.layers
            .iter()
            .all(|layer| layer.as_ref().is_some_and(|layer| layer.start_position == first.start_position && backend.token_rows(&layer.key) == rows && backend.token_rows(&layer.value) == rows))
            .then_some(first.start_position..end)
    }

    /// 回退到已缓存区间内的 terminal 边界；失败时保持原 cache 不变。
    #[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
    pub(crate) fn truncate_end<B>(&mut self, backend: &B, end_position: usize) -> Result<bool, BackendError>
    where
        B: SegmentedTensorBackend<Tensor = T>,
    {
        let Some(range) = self.covered_range(backend) else { return Ok(false) };
        if end_position <= range.start || end_position > range.end {
            return Ok(false);
        }
        let rows = end_position - range.start;
        if rows == range.end - range.start {
            return Ok(true);
        }
        let mut truncated = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter().flatten() {
            truncated.push(Some(DsparkLayerTargetCache { start_position: layer.start_position, key: backend.slice_token_rows(&layer.key, 0, rows)?, value: backend.slice_token_rows(&layer.value, 0, rows)? }));
        }
        self.layers = truncated;
        Ok(true)
    }

    /// 滑窗右移时丢弃 cache 头部，保留仍可见的 K/V，随后只需追加新 suffix。
    #[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
    pub(crate) fn truncate_start<B>(&mut self, backend: &B, start_position: usize) -> Result<bool, BackendError>
    where
        B: SegmentedTensorBackend<Tensor = T>,
    {
        let Some(range) = self.covered_range(backend) else { return Ok(false) };
        if start_position < range.start || start_position >= range.end {
            return Ok(false);
        }
        if start_position == range.start {
            return Ok(true);
        }
        let offset = start_position - range.start;
        let rows = range.end - start_position;
        let mut truncated = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter().flatten() {
            truncated.push(Some(DsparkLayerTargetCache { start_position, key: backend.slice_token_rows(&layer.key, offset, rows)?, value: backend.slice_token_rows(&layer.value, offset, rows)? }));
        }
        self.layers = truncated;
        Ok(true)
    }
}

impl<T> Default for DsparkTargetCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W> DsparkBackbone<W> {
    pub fn validate(&self) -> Result<(), BackendError> {
        GqaGeometry { num_heads: self.head_count, num_kv_heads: self.kv_head_count, head_dim: self.head_dim }.validate().map_err(|msg| BackendError::Compute { msg })?;
        if self.layers.is_empty() || !self.rms_eps.is_finite() || self.rms_eps <= 0.0 {
            return Err(BackendError::Compute { msg: format!("DSpark backbone 非法: layers={} eps={}", self.layers.len(), self.rms_eps) });
        }
        Ok(())
    }

    /// 只有全部 drafter layer 都是滑窗 attention 时才能安全裁掉更早的
    /// target hidden；不同 layer 的窗口不同时保留最大窗口。
    pub fn target_history_window(&self) -> Option<usize> {
        self.layers.iter().try_fold(0_usize, |window, layer| layer.sliding_window.map(|candidate| window.max(candidate)))
    }
}

impl<W> DsparkTargetProjector<W> {
    pub fn target_history_window(&self) -> Option<usize> {
        self.layers.iter().try_fold(0_usize, |window, layer| layer.sliding_window.map(|candidate| window.max(candidate)))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn backbone_forward<B>(
    backend: &B,
    backbone: &DsparkBackbone<B::Weight>,
    cache: &mut DsparkTargetCache<B::Tensor>,
    mut noise: B::Tensor,
    target_hidden: &B::Tensor,
    target_position: usize,
    block_position: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
) -> Result<B::Tensor, BackendError>
where
    B: BlockAttentionBackend + GqaPrefillBackend + SegmentedTensorBackend,
{
    backbone.validate()?;
    if cache.layers.is_empty() {
        cache.layers = std::iter::repeat_with(|| None).take(backbone.layers.len()).collect();
    } else if cache.layers.len() != backbone.layers.len() {
        return Err(BackendError::Compute { msg: format!("DSpark target cache layers={}，期望 {}", cache.layers.len(), backbone.layers.len()) });
    }
    let geometry = GqaGeometry { num_heads: backbone.head_count, num_kv_heads: backbone.kv_head_count, head_dim: backbone.head_dim };
    // target_hidden 可为空，或 cache 已由 warm_target_cache 精确预热到当前位；
    // 后者只执行 query block，避免每个 draft round 重投影完整 target history。
    let target_rows = backend.token_rows(target_hidden);
    let target_end = target_position.checked_add(target_rows).ok_or_else(|| BackendError::Compute { msg: "DSpark target cache position 溢出".to_owned() })?;
    let warm_cache = target_rows > 0 && cache.covered_range(backend) != Some(target_position..target_end);
    for (layer_index, layer) in backbone.layers.iter().enumerate() {
        let residual = noise;
        let (query, noise_key, noise_value) = backend.rmsnorm_triple_linear(&residual, &layer.input_norm, backbone.rms_eps, &layer.q_proj, &layer.k_proj, &layer.v_proj)?;
        let query = backend.rmsnorm_heads(&query, &layer.q_norm, backbone.head_count, backbone.head_dim, backbone.rms_eps)?;
        let query = backend.rope(&query, backbone.head_count, backbone.head_dim, RotaryLayout::SplitHalf, block_position, rope_cos, rope_sin)?;

        if warm_cache {
            update_target_cache(backend, layer, backbone, &mut cache.layers[layer_index], target_hidden, target_position, rope_cos, rope_sin)?;
        }
        let target_cache = cache.layers[layer_index].as_ref().expect("DSpark target cache 已更新");
        let noise_key = backend.rmsnorm_heads(&noise_key, &layer.k_norm, backbone.kv_head_count, backbone.head_dim, backbone.rms_eps)?;
        let noise_key = backend.rope(&noise_key, backbone.kv_head_count, backbone.head_dim, RotaryLayout::SplitHalf, block_position, rope_cos, rope_sin)?;
        let query_rows = backend.token_rows(&query);
        // 可见区间以 cache 实际行数为准:全 full-attention drafter 的 cache 覆盖
        // 全历史,远多于本轮 target_hidden 行(rolling 语义下二者相等)。
        let context_rows = backend.token_rows(&target_cache.key);
        let visible = visible_ranges(context_rows, query_rows, layer.sliding_window, layer.sliding_window_non_causal);
        let spec = BlockAttentionSpec { geometry, score_scale: 1.0 / (backbone.head_dim as f32).sqrt(), visible };
        let attention = backend.block_attention_prefix_suffix(&query, &target_cache.key, &target_cache.value, &noise_key, &noise_value, &spec)?;
        let attention = backend.linear(&attention, &layer.o_proj)?;
        let attention_output = backend.add(&residual, &attention)?;
        let activated = backend.rmsnorm_gated_linear(&attention_output, &layer.post_attention_norm, backbone.rms_eps, &layer.gate_proj, &layer.up_proj, &Activation::Silu)?;
        let down = backend.linear(&activated, &layer.down_proj)?;
        noise = backend.add(&attention_output, &down)?;
    }
    backend.rmsnorm(&noise, &backbone.output_norm, backbone.rms_eps)
}

#[allow(clippy::too_many_arguments)]
pub fn backbone_forward_batch<B>(
    backend: &B,
    backbone: &DsparkBackbone<B::Weight>,
    segments: &mut [DsparkTargetSegment<'_, B::Tensor>],
    block_rows: usize,
    mut noise: B::Tensor,
    rope_cos: &[f32],
    rope_sin: &[f32],
) -> Result<B::Tensor, BackendError>
where
    B: BlockAttentionBackend + GqaPrefillBackend + SegmentedTensorBackend,
{
    backbone.validate()?;
    if segments.len() < 2 || block_rows == 0 || backend.token_rows(&noise) != segments.len().saturating_mul(block_rows) {
        return Err(BackendError::Compute { msg: format!("DSpark batch shape 非法: sessions={} block_rows={block_rows} noise_rows={}", segments.len(), backend.token_rows(&noise)) });
    }
    for segment in segments.iter_mut() {
        if segment.cache.layers.is_empty() {
            segment.cache.layers = std::iter::repeat_with(|| None).take(backbone.layers.len()).collect();
        } else if segment.cache.layers.len() != backbone.layers.len() {
            return Err(BackendError::Compute { msg: format!("DSpark target cache layers={}，期望 {}", segment.cache.layers.len(), backbone.layers.len()) });
        }
    }
    let geometry = GqaGeometry { num_heads: backbone.head_count, num_kv_heads: backbone.kv_head_count, head_dim: backbone.head_dim };
    let rope_segments = segments.iter().map(|segment| crate::backend::TokenSegment { position: segment.block_position, rows: block_rows }).collect::<Vec<_>>();
    for (layer_index, layer) in backbone.layers.iter().enumerate() {
        let residual = noise;
        let (query, noise_key, noise_value) = backend.rmsnorm_triple_linear(&residual, &layer.input_norm, backbone.rms_eps, &layer.q_proj, &layer.k_proj, &layer.v_proj)?;
        let query = backend.rmsnorm_heads(&query, &layer.q_norm, backbone.head_count, backbone.head_dim, backbone.rms_eps)?;
        let query = backend.rope_segmented(&query, backbone.head_count, backbone.head_dim, RotaryLayout::SplitHalf, false, &rope_segments, rope_cos, rope_sin)?;

        for segment in segments.iter_mut() {
            let target_rows = backend.token_rows(segment.hidden);
            let target_end = segment.position.checked_add(target_rows).ok_or_else(|| BackendError::Compute { msg: "DSpark batch target cache position 溢出".to_owned() })?;
            if target_rows > 0 && segment.cache.covered_range(backend) != Some(segment.position..target_end) {
                update_target_cache(backend, layer, backbone, &mut segment.cache.layers[layer_index], segment.hidden, segment.position, rope_cos, rope_sin)?;
            }
        }
        let noise_key = backend.rmsnorm_heads(&noise_key, &layer.k_norm, backbone.kv_head_count, backbone.head_dim, backbone.rms_eps)?;
        let noise_key = backend.rope_segmented(&noise_key, backbone.kv_head_count, backbone.head_dim, RotaryLayout::SplitHalf, false, &rope_segments, rope_cos, rope_sin)?;
        let noise_keys = (0..segments.len()).map(|index| backend.slice_token_rows(&noise_key, index * block_rows, block_rows)).collect::<Result<Vec<_>, _>>()?;
        let noise_values = (0..segments.len()).map(|index| backend.slice_token_rows(&noise_value, index * block_rows, block_rows)).collect::<Result<Vec<_>, _>>()?;
        let mut key_refs = Vec::with_capacity(segments.len() * 2);
        let mut value_refs = Vec::with_capacity(segments.len() * 2);
        let mut visible = Vec::with_capacity(segments.len() * block_rows);
        let mut key_offset = 0usize;
        for (index, segment) in segments.iter().enumerate() {
            let target = segment.cache.layers[layer_index].as_ref().expect("DSpark target cache 已更新");
            let context_rows = backend.token_rows(&target.key);
            let hidden_rows = backend.token_rows(segment.hidden);
            let hidden_end = segment.position.checked_add(hidden_rows).ok_or_else(|| BackendError::Compute { msg: "DSpark batch target position 溢出".to_owned() })?;
            let cache_end = target.start_position.checked_add(context_rows).ok_or_else(|| BackendError::Compute { msg: "DSpark batch cache position 溢出".to_owned() })?;
            if backend.token_rows(&target.value) != context_rows || hidden_end != segment.block_position || cache_end != segment.block_position {
                return Err(BackendError::Compute {
                    msg: format!(
                        "DSpark batch target cache range=[{},{cache_end}) rows={}/{}，suffix=[{},{hidden_end})，block={}",
                        target.start_position,
                        context_rows,
                        backend.token_rows(&target.value),
                        segment.position,
                        segment.block_position,
                    ),
                });
            }
            key_refs.extend([&target.key, &noise_keys[index]]);
            value_refs.extend([&target.value, &noise_values[index]]);
            for range in visible_ranges(context_rows, block_rows, layer.sliding_window, layer.sliding_window_non_causal) {
                visible.push(key_offset + range.start..key_offset + range.end);
            }
            key_offset += context_rows + block_rows;
        }
        let spec = BlockAttentionSpec { geometry, score_scale: 1.0 / (backbone.head_dim as f32).sqrt(), visible };
        let attention = backend.block_attention_segments(&query, &key_refs, &value_refs, &spec)?;
        let attention = backend.linear(&attention, &layer.o_proj)?;
        let attention_output = backend.add(&residual, &attention)?;
        let activated = backend.rmsnorm_gated_linear(&attention_output, &layer.post_attention_norm, backbone.rms_eps, &layer.gate_proj, &layer.up_proj, &Activation::Silu)?;
        let down = backend.linear(&activated, &layer.down_proj)?;
        noise = backend.add(&attention_output, &down)?;
    }
    backend.rmsnorm(&noise, &backbone.output_norm, backbone.rms_eps)
}

/// 只用 target hidden 扩展 drafter 的 target KV cache(prefill 预热/verify 接受
/// 前缀推进),不执行 drafter backbone。
#[allow(clippy::too_many_arguments)]
pub fn warm_target_cache<B>(backend: &B, backbone: &DsparkBackbone<B::Weight>, cache: &mut DsparkTargetCache<B::Tensor>, target_hidden: &B::Tensor, target_position: usize, rope_cos: &[f32], rope_sin: &[f32]) -> Result<(), BackendError>
where
    B: GqaPrefillBackend + SegmentedTensorBackend,
{
    if cache.layers.is_empty() {
        cache.layers = std::iter::repeat_with(|| None).take(backbone.layers.len()).collect();
    } else if cache.layers.len() != backbone.layers.len() {
        return Err(BackendError::Compute { msg: format!("DSpark target cache layers={}，期望 {}", cache.layers.len(), backbone.layers.len()) });
    }
    for (layer_index, layer) in backbone.layers.iter().enumerate() {
        update_target_cache(backend, layer, backbone, &mut cache.layers[layer_index], target_hidden, target_position, rope_cos, rope_sin)?;
    }
    Ok(())
}

/// CPU proposal 模式的 GPU cache-only 路径：只读 K/KNorm/V resident，避免
/// 为 cache 维护加载完整 drafter backbone 与输出头。
#[allow(clippy::too_many_arguments)]
pub fn warm_target_cache_projected<B>(
    backend: &B,
    projector: &DsparkTargetProjector<B::Weight>,
    cache: &mut DsparkTargetCache<B::Tensor>,
    target_hidden: &B::Tensor,
    target_position: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
) -> Result<(), BackendError>
where
    B: GqaPrefillBackend + SegmentedTensorBackend,
{
    if cache.layers.is_empty() {
        cache.layers = std::iter::repeat_with(|| None).take(projector.layers.len()).collect();
    } else if cache.layers.len() != projector.layers.len() {
        return Err(BackendError::Compute { msg: format!("DSpark target cache layers={}，期望 {}", cache.layers.len(), projector.layers.len()) });
    }
    let history_window = projector.target_history_window();
    for (layer_index, layer) in projector.layers.iter().enumerate() {
        update_target_cache_weights(
            backend,
            &layer.k_proj,
            &layer.k_norm,
            &layer.v_proj,
            layer.sliding_window,
            history_window,
            projector.kv_head_count,
            projector.head_dim,
            projector.rms_eps,
            &mut cache.layers[layer_index],
            target_hidden,
            target_position,
            rope_cos,
            rope_sin,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_target_cache<B>(
    backend: &B,
    layer: &DsparkLayer<B::Weight>,
    backbone: &DsparkBackbone<B::Weight>,
    slot: &mut Option<DsparkLayerTargetCache<B::Tensor>>,
    target_hidden: &B::Tensor,
    target_position: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
) -> Result<(), BackendError>
where
    B: GqaPrefillBackend + SegmentedTensorBackend,
{
    update_target_cache_weights(
        backend,
        &layer.k_proj,
        &layer.k_norm,
        &layer.v_proj,
        layer.sliding_window,
        backbone.target_history_window(),
        backbone.kv_head_count,
        backbone.head_dim,
        backbone.rms_eps,
        slot,
        target_hidden,
        target_position,
        rope_cos,
        rope_sin,
    )
}

#[allow(clippy::too_many_arguments)]
fn update_target_cache_weights<B>(
    backend: &B,
    k_proj: &B::Weight,
    k_norm: &B::Weight,
    v_proj: &B::Weight,
    sliding_window: Option<usize>,
    target_history_window: Option<usize>,
    kv_head_count: usize,
    head_dim: usize,
    rms_eps: f32,
    slot: &mut Option<DsparkLayerTargetCache<B::Tensor>>,
    target_hidden: &B::Tensor,
    target_position: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
) -> Result<(), BackendError>
where
    B: GqaPrefillBackend + SegmentedTensorBackend,
{
    let target_rows = backend.token_rows(target_hidden);
    let target_end = target_position.checked_add(target_rows).ok_or_else(|| BackendError::Compute { msg: "DSpark target cache position 溢出".to_owned() })?;
    let cached = slot.take();
    // 全 full-attention drafter(target_history_window()=None,如 dflash)必须保留完整
    // 前缀历史:cache = 前缀[0..target_position) ++ 本轮新行。滑窗 drafter 维持
    // 滚动语义(cache 仅覆盖本轮区间,窗口由 visible 区间表达)。
    if target_history_window.is_none() {
        let prefix_rows = cached.as_ref().map_or(0, |cached| {
            let rows = backend.token_rows(&cached.key);
            if backend.token_rows(&cached.value) != rows || target_position < cached.start_position { 0 } else { rows.min(target_position - cached.start_position) }
        });
        let prefix = match cached {
            Some(cached) if prefix_rows > 0 => Some((backend.slice_token_rows(&cached.key, 0, prefix_rows)?, backend.slice_token_rows(&cached.value, 0, prefix_rows)?, cached.start_position)),
            _ => None,
        };
        let key = backend.linear(target_hidden, k_proj)?;
        let key = backend.rmsnorm_heads(&key, k_norm, kv_head_count, head_dim, rms_eps)?;
        let key = backend.rope(&key, kv_head_count, head_dim, RotaryLayout::SplitHalf, target_position, rope_cos, rope_sin)?;
        let value = backend.linear(target_hidden, v_proj)?;
        let start_position = prefix.as_ref().map_or(target_position, |item| item.2);
        let capacity_rows = target_end.saturating_sub(start_position);
        let (key, value) = match prefix {
            Some((prefix_key, prefix_value, _)) => (backend.concat_token_rows_reserved(&[&prefix_key, &key], capacity_rows)?, backend.concat_token_rows_reserved(&[&prefix_value, &value], capacity_rows)?),
            None => (backend.concat_token_rows_reserved(&[&key], capacity_rows)?, backend.concat_token_rows_reserved(&[&value], capacity_rows)?),
        };
        *slot = Some(DsparkLayerTargetCache { start_position, key, value });
        return Ok(());
    }
    // CPU 异步 drafter 只传 cache 尚未覆盖的新 suffix。它与 GPU 侧传完整
    // rolling window 语义等价，但避免每轮下载并扫描整个 aux history。
    let append_cached_rows = if let Some(cached) = cached.as_ref() {
        let cached_rows = backend.token_rows(&cached.key);
        let cached_end = cached.start_position.checked_add(cached_rows).ok_or_else(|| BackendError::Compute { msg: "DSpark sliding cache position 溢出".to_owned() })?;
        (target_rows > 0 && backend.token_rows(&cached.value) == cached_rows && target_position == cached_end).then_some(cached_rows)
    } else {
        None
    };
    if let Some(cached_rows) = append_cached_rows {
        let key = backend.linear(target_hidden, k_proj)?;
        let key = backend.rmsnorm_heads(&key, k_norm, kv_head_count, head_dim, rms_eps)?;
        let key = backend.rope(&key, kv_head_count, head_dim, RotaryLayout::SplitHalf, target_position, rope_cos, rope_sin)?;
        let value = backend.linear(target_hidden, v_proj)?;
        let capacity_rows = target_history_window.unwrap().max(cached_rows + target_rows);
        let cached = cached.expect("sliding cache 已检查");
        let key = backend.append_token_rows_reserved(cached.key, &key, capacity_rows)?;
        let value = backend.append_token_rows_reserved(cached.value, &value, capacity_rows)?;
        *slot = Some(DsparkLayerTargetCache { start_position: cached.start_position, key, value });
        return Ok(());
    }
    let retained = cached
        .as_ref()
        .filter(|cached| {
            let rows = backend.token_rows(&cached.key);
            backend.token_rows(&cached.value) == rows && target_position >= cached.start_position && target_position < cached.start_position.saturating_add(rows)
        })
        .map(|cached| {
            let cached_end = cached.start_position + backend.token_rows(&cached.key);
            let rows = target_end.min(cached_end) - target_position;
            let offset = target_position - cached.start_position;
            Ok((backend.slice_token_rows(&cached.key, offset, rows)?, backend.slice_token_rows(&cached.value, offset, rows)?, target_position + rows))
        })
        .transpose()?;
    let projected_start = retained.as_ref().map_or(target_position, |item| item.2);
    let projected = if projected_start < target_end {
        let hidden = backend.slice_token_rows(target_hidden, projected_start - target_position, target_end - projected_start)?;
        let key = backend.linear(&hidden, k_proj)?;
        let key = backend.rmsnorm_heads(&key, k_norm, kv_head_count, head_dim, rms_eps)?;
        let key = backend.rope(&key, kv_head_count, head_dim, RotaryLayout::SplitHalf, projected_start, rope_cos, rope_sin)?;
        Some((key, backend.linear(&hidden, v_proj)?))
    } else {
        None
    };
    let capacity_rows = sliding_window.unwrap_or(target_rows).max(target_rows);
    let (key, value) = match (retained, projected) {
        (Some((retained_key, retained_value, _)), Some((projected_key, projected_value))) => {
            (backend.concat_token_rows_reserved(&[&retained_key, &projected_key], capacity_rows)?, backend.concat_token_rows_reserved(&[&retained_value, &projected_value], capacity_rows)?)
        }
        (Some((key, value, _)), None) => (key, value),
        (None, Some((key, value))) => (backend.concat_token_rows_reserved(&[&key], capacity_rows)?, backend.concat_token_rows_reserved(&[&value], capacity_rows)?),
        (None, None) => return Err(BackendError::Compute { msg: "DSpark target cache 既无保留行也无新增行".to_owned() }),
    };
    *slot = Some(DsparkLayerTargetCache { start_position: target_position, key, value });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{BackendResources, SegmentedTensorBackend, cpu::CpuContext},
        kernel::cpu::CpuTensor,
    };

    #[test]
    fn cpu_reserved_append_reuses_owned_allocation() {
        let backend = CpuContext;
        let mut data = Vec::with_capacity(8);
        data.extend_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let prefix = CpuTensor { data, rows: 2, cols: 2 };
        let pointer = prefix.data.as_ptr();
        let suffix = CpuTensor { data: vec![5.0, 6.0], rows: 1, cols: 2 };

        let appended = backend.append_token_rows_reserved(prefix, &suffix, 4).unwrap();

        assert_eq!(appended.data.as_ptr(), pointer);
        assert_eq!((appended.rows, appended.cols), (3, 2));
        assert_eq!(appended.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn target_cache_reports_owned_allocations_and_resets() {
        let backend = CpuContext;
        let cache = DsparkLayerTargetCache { start_position: 0, key: CpuTensor { data: vec![0.0; 8], rows: 2, cols: 4 }, value: CpuTensor { data: vec![0.0; 8], rows: 2, cols: 4 } };
        let mut cache = DsparkTargetCache { layers: vec![Some(cache)] };
        assert_eq!(cache.allocated_bytes(&backend), 16 * std::mem::size_of::<f32>() as u64);
        cache.reset_session();
        assert_eq!(cache.warmed_layers(), 0);
        assert_eq!(cache.allocated_bytes(&backend), 0);
    }

    #[test]
    fn target_cache_snapshot_roundtrip_and_truncate() {
        let backend = CpuContext;
        let layer = DsparkLayerTargetCache { start_position: 7, key: CpuTensor { data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rows: 3, cols: 2 }, value: CpuTensor { data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], rows: 3, cols: 2 } };
        let cache = DsparkTargetCache { layers: vec![Some(layer)] };
        let snapshot = cache.try_snapshot(|tensor| Ok::<_, ()>((tensor.data.clone(), tensor.rows, tensor.cols))).unwrap().unwrap();
        let snapshot = DsparkTargetCacheSnapshot {
            layers: snapshot
                .layers
                .into_iter()
                .map(|layer| DsparkTargetLayerSnapshot {
                    start_position: layer.start_position,
                    key: CpuTensor { data: layer.key.0, rows: layer.key.1, cols: layer.key.2 },
                    value: CpuTensor { data: layer.value.0, rows: layer.value.1, cols: layer.value.2 },
                })
                .collect(),
        };
        let mut restored = DsparkTargetCache::from_snapshot(snapshot);
        assert_eq!(restored.covered_range(&backend), Some(7..10));
        assert!(restored.truncate_start(&backend, 8).unwrap());
        assert_eq!(restored.covered_range(&backend), Some(8..10));
        assert!(restored.truncate_end(&backend, 9).unwrap());
        assert_eq!(restored.covered_range(&backend), Some(8..9));
        assert_eq!(restored.allocated_bytes(&backend), 4 * std::mem::size_of::<f32>() as u64);
    }

    #[test]
    fn tiny_backbone_accepts_asymmetric_target_and_block() {
        let backend = CpuContext;
        let weight = |values: &[f32], rows, cols| backend.prepare_f32(values, rows, cols).unwrap();
        let identity = [1.0, 0.0, 0.0, 1.0];
        let norm = [1.0, 1.0];
        let layer = DsparkLayer {
            input_norm: weight(&norm, 1, 2),
            q_proj: weight(&identity, 2, 2),
            q_norm: weight(&norm, 1, 2),
            k_proj: weight(&identity, 2, 2),
            k_norm: weight(&norm, 1, 2),
            v_proj: weight(&identity, 2, 2),
            o_proj: weight(&identity, 2, 2),
            post_attention_norm: weight(&norm, 1, 2),
            gate_proj: weight(&identity, 2, 2),
            up_proj: weight(&identity, 2, 2),
            down_proj: weight(&identity, 2, 2),
            sliding_window: None,
            sliding_window_non_causal: false,
        };
        let backbone = DsparkBackbone { layers: vec![layer], output_norm: weight(&norm, 1, 2), head_count: 1, kv_head_count: 1, head_dim: 2, rms_eps: 1.0e-5 };
        let noise = CpuTensor { data: vec![1.0, 0.0, 0.0, 1.0], rows: 2, cols: 2 };
        let target = CpuTensor { data: vec![0.5, 0.5], rows: 1, cols: 2 };
        let output = backbone_forward(&backend, &backbone, &mut DsparkTargetCache::new(), noise, &target, 0, 1, &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0]).unwrap();
        assert_eq!((output.rows, output.cols), (2, 2));
        assert!(output.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prewarmed_full_attention_cache_skips_target_reprojection() {
        let backend = CpuContext;
        let weight = |values: &[f32], rows, cols| backend.prepare_f32(values, rows, cols).unwrap();
        let identity = [1.0, 0.0, 0.0, 1.0];
        let norm = [1.0, 1.0];
        let layer = DsparkLayer {
            input_norm: weight(&norm, 1, 2),
            q_proj: weight(&identity, 2, 2),
            q_norm: weight(&norm, 1, 2),
            k_proj: weight(&identity, 2, 2),
            k_norm: weight(&norm, 1, 2),
            v_proj: weight(&identity, 2, 2),
            o_proj: weight(&identity, 2, 2),
            post_attention_norm: weight(&norm, 1, 2),
            gate_proj: weight(&identity, 2, 2),
            up_proj: weight(&identity, 2, 2),
            down_proj: weight(&identity, 2, 2),
            sliding_window: None,
            sliding_window_non_causal: false,
        };
        let backbone = DsparkBackbone { layers: vec![layer], output_norm: weight(&norm, 1, 2), head_count: 1, kv_head_count: 1, head_dim: 2, rms_eps: 1.0e-5 };
        let target = CpuTensor { data: vec![0.5, 0.5], rows: 1, cols: 2 };
        let mut cache = DsparkTargetCache::new();
        warm_target_cache(&backend, &backbone, &mut cache, &target, 0, &[1.0; 4], &[0.0; 4]).unwrap();

        // cols=3 若进入 K/V projection 会 shape 报错；精确命中 cache 时它只提供
        // 目标区间行数，draft 必须直接复用已经投影的 K/V。
        let already_cached = CpuTensor { data: vec![9.0; 3], rows: 1, cols: 3 };
        let noise = CpuTensor { data: vec![1.0, 0.0, 0.0, 1.0], rows: 2, cols: 2 };
        let output = backbone_forward(&backend, &backbone, &mut cache, noise, &already_cached, 0, 1, &[1.0; 4], &[0.0; 4]).unwrap();
        assert_eq!((output.rows, output.cols), (2, 2));
        assert_eq!(cache.covered_range(&backend), Some(0..1));
    }

    #[test]
    fn cache_only_target_projector_matches_full_backbone() {
        let backend = CpuContext;
        let weight = |values: &[f32], rows, cols| backend.prepare_f32(values, rows, cols).unwrap();
        let identity = [1.0, 0.0, 0.0, 1.0];
        let norm = [1.0, 1.0];
        let k_proj = weight(&identity, 2, 2);
        let k_norm = weight(&norm, 1, 2);
        let v_proj = weight(&identity, 2, 2);
        let layer = DsparkLayer {
            input_norm: weight(&norm, 1, 2),
            q_proj: weight(&identity, 2, 2),
            q_norm: weight(&norm, 1, 2),
            k_proj: k_proj.clone(),
            k_norm: k_norm.clone(),
            v_proj: v_proj.clone(),
            o_proj: weight(&identity, 2, 2),
            post_attention_norm: weight(&norm, 1, 2),
            gate_proj: weight(&identity, 2, 2),
            up_proj: weight(&identity, 2, 2),
            down_proj: weight(&identity, 2, 2),
            sliding_window: Some(2),
            sliding_window_non_causal: false,
        };
        let backbone = DsparkBackbone { layers: vec![layer], output_norm: weight(&norm, 1, 2), head_count: 1, kv_head_count: 1, head_dim: 2, rms_eps: 1.0e-5 };
        let projector = DsparkTargetProjector { layers: vec![DsparkTargetLayer { k_proj, k_norm, v_proj, sliding_window: Some(2) }], kv_head_count: 1, head_dim: 2, rms_eps: 1.0e-5 };
        let target = CpuTensor { data: vec![0.5, 0.5, 0.25, 0.75], rows: 2, cols: 2 };
        let cosine = [1.0; 8];
        let sine = [0.0; 8];
        let mut full = DsparkTargetCache::new();
        let mut cache_only = DsparkTargetCache::new();
        warm_target_cache(&backend, &backbone, &mut full, &target, 3, &cosine, &sine).unwrap();
        warm_target_cache_projected(&backend, &projector, &mut cache_only, &target, 3, &cosine, &sine).unwrap();
        let snapshot = |cache: &DsparkTargetCache<CpuTensor>| cache.try_snapshot(|tensor| Ok::<_, ()>(tensor.data.clone())).unwrap().unwrap();
        let full = snapshot(&full);
        let cache_only = snapshot(&cache_only);
        assert_eq!(full.layers.len(), cache_only.layers.len());
        for (full, cache_only) in full.layers.iter().zip(cache_only.layers.iter()) {
            assert_eq!(full.start_position, cache_only.start_position);
            assert_eq!(full.key, cache_only.key);
            assert_eq!(full.value, cache_only.value);
        }
    }

    #[test]
    fn rolling_target_cache_matches_reprojection() {
        let backend = CpuContext;
        let weight = |values: &[f32], rows, cols| backend.prepare_f32(values, rows, cols).unwrap();
        let identity = [1.0, 0.0, 0.0, 1.0];
        let norm = [1.0, 1.0];
        let layer = DsparkLayer {
            input_norm: weight(&norm, 1, 2),
            q_proj: weight(&identity, 2, 2),
            q_norm: weight(&norm, 1, 2),
            k_proj: weight(&identity, 2, 2),
            k_norm: weight(&norm, 1, 2),
            v_proj: weight(&identity, 2, 2),
            o_proj: weight(&identity, 2, 2),
            post_attention_norm: weight(&norm, 1, 2),
            gate_proj: weight(&identity, 2, 2),
            up_proj: weight(&identity, 2, 2),
            down_proj: weight(&identity, 2, 2),
            sliding_window: Some(2),
            sliding_window_non_causal: false,
        };
        let backbone = DsparkBackbone { layers: vec![layer], output_norm: weight(&norm, 1, 2), head_count: 1, kv_head_count: 1, head_dim: 2, rms_eps: 1.0e-5 };
        let noise = CpuTensor { data: vec![1.0, 0.0, 0.0, 1.0], rows: 2, cols: 2 };
        let first = CpuTensor { data: vec![0.5, 0.5, 0.25, 0.75], rows: 2, cols: 2 };
        let shifted = CpuTensor { data: vec![0.25, 0.75, 0.75, 0.25], rows: 2, cols: 2 };
        let cosine = [1.0; 8];
        let sine = [0.0; 8];
        let mut rolling = DsparkTargetCache::new();
        backbone_forward(&backend, &backbone, &mut rolling, noise.clone(), &first, 0, 2, &cosine, &sine).unwrap();
        let suffix = backend.slice_token_rows(&shifted, 1, 1).unwrap();
        let cached = backbone_forward(&backend, &backbone, &mut rolling, noise.clone(), &suffix, 2, 3, &cosine, &sine).unwrap();
        let fresh = backbone_forward(&backend, &backbone, &mut DsparkTargetCache::new(), noise, &shifted, 1, 3, &cosine, &sine).unwrap();
        assert_eq!(cached.data, fresh.data);
        assert_eq!(rolling.covered_range(&backend), Some(0..3));
    }

    #[test]
    fn ready_draft_batch_with_cached_prefix_matches_independent_sessions() {
        let backend = CpuContext;
        let weight = |values: &[f32], rows, cols| backend.prepare_f32(values, rows, cols).unwrap();
        let identity = [1.0, 0.0, 0.0, 1.0];
        let norm = [1.0, 1.0];
        let layer = DsparkLayer {
            input_norm: weight(&norm, 1, 2),
            q_proj: weight(&identity, 2, 2),
            q_norm: weight(&norm, 1, 2),
            k_proj: weight(&identity, 2, 2),
            k_norm: weight(&norm, 1, 2),
            v_proj: weight(&identity, 2, 2),
            o_proj: weight(&identity, 2, 2),
            post_attention_norm: weight(&norm, 1, 2),
            gate_proj: weight(&identity, 2, 2),
            up_proj: weight(&identity, 2, 2),
            down_proj: weight(&identity, 2, 2),
            sliding_window: Some(3),
            sliding_window_non_causal: false,
        };
        let backbone = DsparkBackbone { layers: vec![layer], output_norm: weight(&norm, 1, 2), head_count: 1, kv_head_count: 1, head_dim: 2, rms_eps: 1.0e-5 };
        let noise_a = CpuTensor { data: vec![1.0, 0.0, 0.0, 1.0], rows: 2, cols: 2 };
        let noise_b = CpuTensor { data: vec![0.5, 0.5, 0.75, 0.25], rows: 2, cols: 2 };
        let target_a = CpuTensor { data: vec![0.5, 0.5, 0.25, 0.75], rows: 2, cols: 2 };
        let target_b = CpuTensor { data: vec![0.25, 0.75, 0.75, 0.25, 1.0, 0.5], rows: 3, cols: 2 };
        let cosine = [1.0; 32];
        let sine = [0.0; 32];
        let expected_a = backbone_forward(&backend, &backbone, &mut DsparkTargetCache::new(), noise_a.clone(), &target_a, 0, 2, &cosine, &sine).unwrap();
        let expected_b = backbone_forward(&backend, &backbone, &mut DsparkTargetCache::new(), noise_b.clone(), &target_b, 4, 7, &cosine, &sine).unwrap();
        let expected = backend.concat_token_rows(&[&expected_a, &expected_b]).unwrap();
        let noise = backend.concat_token_rows(&[&noise_a, &noise_b]).unwrap();
        let target_a_prefix = backend.slice_token_rows(&target_a, 0, 1).unwrap();
        let target_a_suffix = backend.slice_token_rows(&target_a, 1, 1).unwrap();
        let target_b_prefix = backend.slice_token_rows(&target_b, 0, 2).unwrap();
        let target_b_suffix = backend.slice_token_rows(&target_b, 2, 1).unwrap();
        let mut cache_a = DsparkTargetCache::new();
        let mut cache_b = DsparkTargetCache::new();
        warm_target_cache(&backend, &backbone, &mut cache_a, &target_a_prefix, 0, &cosine, &sine).unwrap();
        warm_target_cache(&backend, &backbone, &mut cache_b, &target_b_prefix, 4, &cosine, &sine).unwrap();
        let mut segments = [DsparkTargetSegment { cache: &mut cache_a, hidden: &target_a_suffix, position: 1, block_position: 2 }, DsparkTargetSegment { cache: &mut cache_b, hidden: &target_b_suffix, position: 6, block_position: 7 }];
        let actual = backbone_forward_batch(&backend, &backbone, &mut segments, 2, noise, &cosine, &sine).unwrap();
        let max_abs = actual.data.iter().zip(&expected.data).map(|(left, right)| (left - right).abs()).fold(0.0_f32, f32::max);
        assert!(max_abs <= 1.0e-6, "batch 与独立执行 max_abs={max_abs}");
    }

    #[test]
    fn sliding_attention_keeps_anchor_window_and_causal_noise_prefix() {
        assert_eq!(visible_ranges(10, 3, Some(4), false), [6..11, 6..12, 6..13]);
        assert_eq!(visible_ranges(10, 3, Some(4), true), [6..13, 6..13, 6..13]);
        assert_eq!(visible_ranges(10, 3, None, false), [0..13, 0..13, 0..13]);
    }

    #[test]
    fn target_history_window_requires_all_layers_sliding() {
        let layer = |window| DsparkLayer {
            input_norm: (),
            q_proj: (),
            q_norm: (),
            k_proj: (),
            k_norm: (),
            v_proj: (),
            o_proj: (),
            post_attention_norm: (),
            gate_proj: (),
            up_proj: (),
            down_proj: (),
            sliding_window: window,
            sliding_window_non_causal: false,
        };
        let mut backbone = DsparkBackbone { layers: vec![layer(Some(1024)), layer(Some(2048))], output_norm: (), head_count: 1, kv_head_count: 1, head_dim: 1, rms_eps: 1.0e-5 };
        assert_eq!(backbone.target_history_window(), Some(2048));
        backbone.layers.push(layer(None));
        assert_eq!(backbone.target_history_window(), None);
    }
}
