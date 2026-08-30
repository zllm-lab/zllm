//! 动态稀疏注意力规格。

/// GLM DSA 索引器的模型常量。
///
/// `kpool > 0` 启用 glm5_next 的池化压缩(官方 `Glm5NextTextIndexer`):
/// 每 kpool 个连续 token 压成 1 个索引条目,top_k 语义从"token 数"变为
/// "token 预算"(实际选择 top_k/kpool 个池后展开)。
#[derive(Debug, Clone, Copy)]
pub struct DsaSpec {
    pub num_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub top_k: usize,
    pub rotary_layout: super::rope::RotaryLayout,
    /// 池宽度;0 表示逐 token 索引(glm52/deepseek 语义)。
    pub kpool: usize,
    /// 是否把不完整尾池的原始 token 直接追加进选择结果。
    pub always_select_tail: bool,
}

/// DSA 索引器权重视图。prefill/decode 共享同一组数据依赖。
#[derive(Clone, Copy)]
pub struct DsaWeightsRef<'a, W> {
    pub wq_b: &'a W,
    pub wk: &'a W,
    pub weights_proj: &'a W,
    pub k_norm_weight: &'a W,
    pub k_norm_bias: &'a W,
}

/// CPU/reference DSA 评分与 top-k 选择。
///
/// decode 的单行 query 可以查看全部 key；prefill 的第 N 行只能查看 `0..=N`。
/// `query_rows == 1` 是 decode 的语义开关，由调用方显式声明；
/// `key_rows` 从 `keys.len() / head_dim` 推出。
#[allow(clippy::too_many_arguments)]
pub fn select_rows(keys: &[f32], query: &[f32], query_rows: usize, head_weights: &[f32], spec: &DsaSpec, query_row: usize, count: usize) -> Result<Vec<usize>, String> {
    let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or("DSA query 列数溢出")?;
    let expected_query = query_rows.checked_mul(query_cols).ok_or("DSA query 元素数溢出")?;
    let expected_weights = query_rows.checked_mul(spec.num_heads).ok_or("DSA head weight 元素数溢出")?;
    if !keys.len().is_multiple_of(spec.head_dim) {
        return Err(format!("DSA selection keys.len={} 不能按 head_dim={} 分行", keys.len(), spec.head_dim));
    }
    let key_rows = keys.len() / spec.head_dim;
    let query_index = if query_rows == 1 { 0 } else { query_row };
    if query.len() != expected_query || head_weights.len() != expected_weights || query_index >= query_rows {
        return Err(format!("DSA selection shape keys={} query=[{query_rows},{query_cols}]/{} weights=[{query_rows},{}]/{}", keys.len(), query.len(), spec.num_heads, head_weights.len(),));
    }
    let candidates = if query_rows == 1 { key_rows } else { query_row + 1 }.min(key_rows);
    if count > candidates {
        return Err(format!("DSA selection count={count} 超过候选数 {candidates}"));
    }
    let mut scored = (0..candidates)
        .map(|token| {
            let mut score = 0.0;
            for head in 0..spec.num_heads {
                let query_begin = query_index * query_cols + head * spec.head_dim;
                let key_begin = token * spec.head_dim;
                let dot = query[query_begin..query_begin + spec.head_dim].iter().zip(&keys[key_begin..key_begin + spec.head_dim]).map(|(left, right)| left * right).sum::<f32>();
                score += head_weights[query_index * spec.num_heads + head] * dot.max(0.0);
            }
            (token, score)
        })
        .collect::<Vec<_>>();
    scored.sort_unstable_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    Ok(scored.into_iter().take(count).map(|(token, _)| token).collect())
}

#[cfg(test)]
mod tests {
    use super::{DsaSpec, select_rows};
    use half::bf16;

    const SPEC: DsaSpec = DsaSpec { num_heads: 1, head_dim: 2, rope_dim: 0, top_k: 1, rotary_layout: super::super::rope::RotaryLayout::SplitHalf, kpool: 0, always_select_tail: false };

    #[test]
    fn decode单行query选择全部历史中的最高分() {
        let selected = select_rows(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[1.0, 0.0], 1, &[1.0], &SPEC, 2, 1).unwrap();
        assert_eq!(selected, vec![2]);
    }

    #[test]
    fn prefill不会选择未来token() {
        let selected = select_rows(&[1.0, 0.0, 2.0, 0.0, 100.0, 0.0], &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 3, &[1.0, 1.0, 1.0], &SPEC, 1, 1).unwrap();
        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn dsa在逐head_relu之后加权() {
        let spec = DsaSpec { num_heads: 2, head_dim: 2, rope_dim: 0, top_k: 1, rotary_layout: super::super::rope::RotaryLayout::SplitHalf, kpool: 0, always_select_tail: false };
        // 先合并 query 会错误选择 token 1；逐 head ReLU 后 token 0 的得分为 10，token 1 为 2。
        let selected = select_rows(&[1.0, 0.0, 0.0, 1.0], &[10.0, 1.0, -9.0, 1.0], 1, &[1.0, 1.0], &spec, 1, 1).unwrap();
        assert_eq!(selected, vec![0]);
    }

    fn hadamard(values: &[f32]) -> Vec<f32> {
        assert!(values.len().is_power_of_two());
        let mut output = values.to_vec();
        let mut stride = 1;
        while stride < output.len() {
            for base in (0..output.len()).step_by(stride * 2) {
                for index in 0..stride {
                    let left = output[base + index];
                    let right = output[base + stride + index];
                    output[base + index] = left + right;
                    output[base + stride + index] = left - right;
                }
            }
            stride *= 2;
        }
        let normalization = 1.0 / (output.len() as f32).sqrt();
        output.iter_mut().for_each(|value| *value *= normalization);
        output
    }

    fn q8(values: &[f32], bf16_scale: bool) -> Vec<f32> {
        let maximum = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let mut scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
        if bf16_scale {
            scale = bf16::from_f32(scale).to_f32();
        }
        values.iter().map(|value| (value / scale).round().clamp(-127.0, 127.0) * scale).collect()
    }

    fn topk(keys: &[Vec<f32>], query: &[f32], quantized: bool, rotate: bool, count: usize) -> Vec<usize> {
        let query = if rotate { hadamard(query) } else { query.to_vec() };
        let query = if quantized { q8(&query, false) } else { query };
        let mut scores = keys
            .iter()
            .enumerate()
            .map(|(token, key)| {
                let key = if rotate { hadamard(key) } else { key.clone() };
                let key = if quantized { q8(&key, true) } else { key };
                (token, query.iter().zip(key).map(|(left, right)| left * right).sum::<f32>())
            })
            .collect::<Vec<_>>();
        scores.sort_unstable_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        scores.into_iter().take(count).map(|(token, _)| token).collect()
    }

    #[test]
    fn hadamard两边同时旋转保持点积() {
        let left = (0..128).map(|index| ((index * 17 + 3) as f32 * 0.071).sin()).collect::<Vec<_>>();
        let right = (0..128).map(|index| ((index * 29 + 5) as f32 * 0.043).cos()).collect::<Vec<_>>();
        let expected = left.iter().zip(&right).map(|(left, right)| left * right).sum::<f32>();
        let rotated_left = hadamard(&left);
        let rotated_right = hadamard(&right);
        let actual = rotated_left.iter().zip(rotated_right).map(|(left, right)| left * right).sum::<f32>();
        assert!((actual - expected).abs() < 2e-5, "Hadamard 点积误差: expected={expected} actual={actual}");
    }

    #[test]
    fn hadamard改善离群激活的q8_topk_overlap() {
        let mut state = 15_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 * (1.0 / 16_777_216.0) - 0.5) * 0.4
        };
        let mut query = (0..128).map(|_| next()).collect::<Vec<_>>();
        query[0] = 50.0;
        let keys = (0..256)
            .map(|_| {
                let mut key = (0..128).map(|_| next()).collect::<Vec<_>>();
                key[0] = 50.0 + next() * 0.005;
                key
            })
            .collect::<Vec<_>>();
        let exact = topk(&keys, &query, false, false, 16);
        let plain = topk(&keys, &query, true, false, 16);
        let rotated = topk(&keys, &query, true, true, 16);
        let overlap = |selected: &[usize]| selected.iter().filter(|token| exact.contains(token)).count();
        assert!(overlap(&rotated) >= overlap(&plain) + 2, "离群样本 Top-16 overlap 未改善: plain={} hadamard={}", overlap(&plain), overlap(&rotated));
    }

    #[test]
    fn kpool零gate零ape时池key为均匀平均() {
        // 4 token、kpool=2、head_dim=2,每池是成员的平均。
        let keys = [1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 0.0];
        let gate = [0.0; 8];
        let ape = [0.0; 4];
        let valid = [true, true, true, true];
        let pools = super::pool_states(&keys, &gate, &valid, &ape, 2, 2).unwrap();
        assert_eq!(pools.pool_count(), 2);
        assert_eq!(pools.keys, vec![0.5, 0.5, 3.0, 1.0]);
        assert_eq!(&pools.indices[..4], &[0, 1, 2, 3]);
        assert!(pools.valid.iter().all(|&flag| flag));
    }

    #[test]
    fn kpool池内gate偏置把权重集中到高分slot() {
        // kpool=2、head_dim=1:gate 极端偏置时池 key 逼近高分 token 的 key。
        let keys = [1.0, 100.0];
        let gate = [0.0, 50.0];
        let ape = [0.0, 0.0];
        let pools = super::pool_states(&keys, &gate, &[true, true], &ape, 1, 2).unwrap();
        assert!((pools.keys[0] - 100.0).abs() < 1e-3, "gate 偏置后池 key 应逼近 100,实际 {}", pools.keys[0]);
    }

    #[test]
    fn kpool从首个有效token起分池且不完整池不入topk() {
        // 前置 padding 1 个;A B C D E(5 个有效,索引 1..=5)→ 池 [A,B] [C,D] + 不完整池 [E]。
        let keys = [0.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0];
        let gate = [0.0; 7];
        let ape = [0.0; 2];
        let valid = [false, true, true, true, true, true, false];
        let pools = super::pool_states(&keys, &gate, &valid, &ape, 1, 2).unwrap();
        assert_eq!(pools.pool_count(), 3);
        assert_eq!(&pools.indices[..6], &[1, 2, 3, 4, 5, usize::MAX]);
        assert_eq!(pools.valid, vec![true, true, false]);
    }

    #[test]
    fn kpool_decode选池展开并追加尾池() {
        // 4 token 两池 + 1 个尾 token(E);预算足够两个全选,断言全集 + 尾 token 4。
        let keys = [0.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 5.0, 0.0, 0.0];
        let gate = [0.0; 10];
        let ape = [0.0; 4];
        let valid = [true; 5];
        let pools = super::pool_states(&keys, &gate, &valid, &ape, 2, 2).unwrap();
        let spec = DsaSpec { num_heads: 1, head_dim: 2, rope_dim: 0, top_k: 4, rotary_layout: super::super::rope::RotaryLayout::SplitHalf, kpool: 2, always_select_tail: true };
        let selected = super::select_rows_kpool(&pools, &[0.0, 1.0], 1, &[1.0], &spec, &valid, 0, 5).unwrap();
        assert_eq!(selected, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn kpool_prefill只选池末token可见的池() {
        // query_row=2(prefill 第 3 行):池 1(末 token=3)不可见,只能选池 0。
        let keys = [1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let gate = [0.0; 8];
        let ape = [0.0; 4];
        let valid = [true; 4];
        let pools = super::pool_states(&keys, &gate, &valid, &ape, 2, 2).unwrap();
        let spec = DsaSpec { num_heads: 1, head_dim: 2, rope_dim: 0, top_k: 4, rotary_layout: super::super::rope::RotaryLayout::SplitHalf, kpool: 2, always_select_tail: true };
        let selected = super::select_rows_kpool(&pools, &[2.0, 2.0, 0.0, 0.0, 2.0, 2.0], 3, &[1.0, 1.0, 1.0], &spec, &valid, 2, 4).unwrap();
        // 池 1 得分高但末 token 3 > query_row 2 不可选;尾池 = 可见 3 个中的 1 个(token 2)。
        assert_eq!(selected, vec![0, 1, 2]);
    }
}
// ============================================================================
// kpool 池化压缩(glm5_next)。reference 实现对齐官方 Glm5NextTextIndexer:
// get_pooled_states / forward / append_visible_tail。
// ============================================================================

/// 池化后的索引状态:每池一条加权平均 key 与池内原始 token 索引。
#[derive(Debug, Default)]
pub struct KpoolStates {
    /// [pools, head_dim] 池内 softmax 加权平均 key。
    pub keys: Vec<f32>,
    /// [pools * kpool] 池内原始 token 索引;无效位为 usize::MAX。
    pub indices: Vec<usize>,
    /// [pools] 池是否完整(池内全部 token 有效且在界内)。不完整池只参与尾池直选。
    pub valid: Vec<bool>,
    /// 池宽(kpool),indices/keys 分块步长。
    pub kpool: usize,
}

impl KpoolStates {
    pub fn pool_count(&self) -> usize {
        self.valid.len()
    }
}

/// 从首个有效 token 起按 kpool 分组;池内 logits = gate + APE,softmax 后
/// 对 key 加权平均。gate/APE 为全零时退化为均匀平均。
/// 官方 `get_pooled_states` 的 reference。
#[allow(clippy::too_many_arguments)]
pub fn pool_states(keys: &[f32], gate_scores: &[f32], valid: &[bool], ape: &[f32], head_dim: usize, kpool: usize) -> Result<KpoolStates, String> {
    let tokens = valid.len();
    if kpool == 0 || head_dim == 0 {
        return Err(format!("kpool 池化参数非法: kpool={kpool} head_dim={head_dim}"));
    }
    if keys.len() != tokens * head_dim || gate_scores.len() != tokens * head_dim || ape.len() != kpool * head_dim {
        return Err(format!("kpool 输入形状非法: tokens={tokens} head_dim={head_dim} keys={} gate={} ape={}", keys.len(), gate_scores.len(), ape.len()));
    }
    let first_key = valid.iter().position(|&flag| flag).unwrap_or(tokens);
    let pools = (tokens - first_key).div_ceil(kpool);
    if kpool > 64 {
        return Err(format!("kpool 池宽 {kpool} 超过 reference 支持的 64"));
    }
    let mut states = KpoolStates { keys: Vec::with_capacity(pools * head_dim), indices: Vec::with_capacity(pools * kpool), valid: Vec::with_capacity(pools), kpool };
    for pool in 0..pools {
        let start = first_key + pool * kpool;
        let slot_valid = |slot: usize| {
            let token = start + slot;
            token < tokens && valid[token]
        };
        let mut pool_valid = true;
        let mut indices = [usize::MAX; 64];
        for slot in 0..kpool {
            pool_valid &= slot_valid(slot);
            indices[slot] = if slot_valid(slot) { start + slot } else { usize::MAX };
        }
        // 池内 softmax 是 per-dim 独立的:logits[slot][dim] = gate[token][dim] + ape[slot][dim]。
        for dim in 0..head_dim {
            let mut logits = [f32::NEG_INFINITY; 64];
            for slot in 0..kpool {
                if slot_valid(slot) {
                    let token = start + slot;
                    logits[slot] = gate_scores[token * head_dim + dim] + ape[slot * head_dim + dim];
                }
            }
            let maximum = logits[..kpool].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut pooled = 0.0;
            if maximum != f32::NEG_INFINITY {
                let denominator: f32 = logits[..kpool].iter().map(|logit| (logit - maximum).exp()).sum();
                for slot in 0..kpool {
                    if slot_valid(slot) {
                        let token = start + slot;
                        pooled += ((logits[slot] - maximum).exp() / denominator) * keys[token * head_dim + dim];
                    }
                }
            }
            states.keys.push(pooled);
        }
        states.indices.extend_from_slice(&indices[..kpool]);
        states.valid.push(pool_valid);
    }
    Ok(states)
}

/// kpool top-k:评分选池(ReLU + head 加权,与 select_rows 同数学)后展开回
/// 原始 token,并按 always_select_tail 追加不完整尾池。
/// `query_rows == 1` 是 decode 语义(可见全部历史);prefill 只可见 `0..=query_row`。
/// 池的因果边界:池内最后一个 token 对 query 可见才可选。
#[allow(clippy::too_many_arguments)]
pub fn select_rows_kpool(pools: &KpoolStates, query: &[f32], query_rows: usize, head_weights: &[f32], spec: &DsaSpec, valid_keys: &[bool], query_row: usize, count: usize) -> Result<Vec<usize>, String> {
    let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or("kpool query 列数溢出")?;
    if query.len() != query_rows * query_cols || head_weights.len() != query_rows * spec.num_heads {
        return Err(format!("kpool selection shape query=[{query_rows},{query_cols}]/{} weights/{}", query.len(), head_weights.len()));
    }
    if pools.kpool == 0 || pools.kpool != spec.kpool {
        return Err(format!("kpool selection 池宽不一致: states={} spec={}", pools.kpool, spec.kpool));
    }
    if spec.top_k % spec.kpool != 0 {
        return Err(format!("kpool top_k {} 不是池宽 {} 的整数倍", spec.top_k, spec.kpool));
    }
    let token_count = valid_keys.len();
    let query_index = if query_rows == 1 { 0 } else { query_row };
    if query_index >= query_rows {
        return Err(format!("kpool query_row {query_index} 越界于 {query_rows}"));
    }
    // decode 可见全部;prefill 第 N 行只见 0..=N。
    let visible_bound = if query_rows == 1 { token_count.saturating_sub(1) } else { query_row.min(token_count.saturating_sub(1)) };
    let pool_count = pools.pool_count();
    let budget = (spec.top_k / spec.kpool).min(count.div_ceil(spec.kpool));
    let mut scored = Vec::with_capacity(pool_count);
    for pool in 0..pool_count {
        // 池末 token 的因果可见性:池内最后一个有效 token 索引。
        let pool_indices = &pools.indices[pool * pools.kpool..(pool + 1) * pools.kpool];
        let Some(&pool_end) = pool_indices.iter().rev().find(|&&index| index != usize::MAX) else { continue };
        if !pools.valid[pool] || pool_end > visible_bound {
            continue;
        }
        let mut score = 0.0;
        for head in 0..spec.num_heads {
            let query_begin = query_index * query_cols + head * spec.head_dim;
            let key_begin = pool * spec.head_dim;
            let dot = query[query_begin..query_begin + spec.head_dim].iter().zip(&pools.keys[key_begin..key_begin + spec.head_dim]).map(|(left, right)| left * right).sum::<f32>();
            score += head_weights[query_index * spec.num_heads + head] * dot.max(0.0);
        }
        scored.push((pool, score));
    }
    scored.sort_unstable_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let mut selected = Vec::with_capacity(spec.top_k + pools.kpool.saturating_sub(1));
    for (pool, _) in scored.into_iter().take(budget) {
        for &index in &pools.indices[pool * pools.kpool..(pool + 1) * pools.kpool] {
            if index != usize::MAX {
                selected.push(index);
            }
        }
    }
    if spec.always_select_tail {
        selected.extend(tail_indices(valid_keys, visible_bound, pools.kpool));
    }
    selected.sort_unstable();
    selected.dedup();
    Ok(selected)
}

/// 不完整尾池直选:从首个有效 token 起,可见数量对池宽取余的尾部原始 token。
/// 官方 `append_visible_tail` 的 reference(单 batch)。
fn tail_indices(valid_keys: &[bool], visible_bound: usize, kpool: usize) -> Vec<usize> {
    let Some(first_key) = valid_keys.iter().position(|&flag| flag) else { return Vec::new() };
    // 有效且因果可见的数量:前 visible_bound+1 个里 valid 的部分。
    let visible_count = valid_keys[..=visible_bound.min(valid_keys.len() - 1)].iter().filter(|&&flag| flag).count();
    let tail_count = visible_count % kpool;
    if tail_count == 0 {
        return Vec::new();
    }
    let tail_start = first_key + visible_count - tail_count;
    (0..tail_count).map(|offset| tail_start + offset).collect()
}
