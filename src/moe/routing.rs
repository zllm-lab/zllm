//! 平台无关的 MoE 路由结果整理算法。

use crate::backend::{BackendError, MoePrefillBackend};

pub type ExpertAssignments = Vec<Vec<(u32, f32)>>;

/// 一个 token 的 top-k 路由结果。
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    pub experts: Vec<u32>,
    pub weights: Vec<f32>,
}

fn select_top_k<T, F>(values: &mut Vec<T>, top_k: usize, compare: F)
where
    F: FnMut(&T, &T) -> std::cmp::Ordering + Copy,
{
    if top_k < values.len() {
        values.select_nth_unstable_by(top_k, compare);
        values.truncate(top_k);
    }
    values.sort_unstable_by(compare);
}

/// 对 router logits 取 softmax 与 top-k；可选择保留全专家概率质量或重新归一化选中项。
pub fn route_softmax_logits(logits: &[f32], top_k: usize, scaling_factor: f32, normalize_selected: bool) -> Result<Routing, String> {
    if logits.is_empty() || top_k == 0 || top_k > logits.len() || !scaling_factor.is_finite() {
        return Err(format!("softmax router 参数非法: logits={} top_k={top_k} scaling={scaling_factor}", logits.len()));
    }
    let mut ranked = logits.iter().copied().enumerate().map(|(expert, logit)| (logit, expert as u32)).collect::<Vec<_>>();
    select_top_k(&mut ranked, top_k, |left, right| right.0.total_cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let selected = &ranked;
    let max_logit = selected[0].0;
    let denominator = if normalize_selected { selected.iter().map(|(logit, _)| (*logit - max_logit).exp()).sum::<f32>() } else { logits.iter().map(|logit| (*logit - max_logit).exp()).sum::<f32>() };
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(format!("softmax router 质量和非法: {denominator}"));
    }
    Ok(Routing { experts: selected.iter().map(|(_, expert)| *expert).collect(), weights: selected.iter().map(|(logit, _)| (*logit - max_logit).exp() / denominator * scaling_factor).collect() })
}

/// bias 修正路由的公共骨架：`raw_score` 给出每专家原始权重，correction bias
/// 只参与 top-k 选择，选中项按 raw 归一化。`what` 是错误信息里的路由名。
fn route_bias_corrected_logits(logits: &[f32], bias: &[f32], top_k: usize, scaling_factor: f32, what: &str, raw_score: impl Fn(f32) -> f32) -> Result<Routing, String> {
    if logits.is_empty() || logits.len() != bias.len() || top_k == 0 || top_k > logits.len() || !scaling_factor.is_finite() {
        return Err(format!("{what} 参数非法: logits={} bias={} top_k={top_k}", logits.len(), bias.len()));
    }
    let mut corrected = Vec::with_capacity(logits.len());
    for (expert, (&logit, &bias)) in logits.iter().zip(bias).enumerate() {
        let raw = raw_score(logit);
        let expert = u32::try_from(expert).map_err(|_| format!("expert {expert} 超出 u32"))?;
        corrected.push((raw + bias, raw, expert));
    }
    select_top_k(&mut corrected, top_k, |left, right| right.0.partial_cmp(&left.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| left.2.cmp(&right.2)));
    let sum_raw: f32 = corrected.iter().map(|(_, raw, _)| *raw).sum();
    if !sum_raw.is_finite() || sum_raw <= 0.0 {
        return Err(format!("{what} top-k raw 权重和非法: {sum_raw}"));
    }
    let mut experts = Vec::with_capacity(top_k);
    let mut weights = Vec::with_capacity(top_k);
    for (_, raw, expert) in corrected.into_iter().take(top_k) {
        experts.push(expert);
        weights.push(raw / sum_raw * scaling_factor);
    }
    Ok(Routing { experts, weights })
}

/// DeepSeek-V4 原始路由分数：`sqrt(softplus(logit))`，softplus 用数值稳定形式。
fn sqrt_softplus_score(logit: f32) -> f32 {
    let softplus = if logit > 0.0 { logit + (-logit).exp().ln_1p() } else { logit.exp().ln_1p() };
    softplus.sqrt()
}

/// 对 router logits 应用 sigmoid、correction bias、top-k 与权重归一化。
pub fn route_sigmoid_bias_logits(logits: &[f32], bias: &[f32], top_k: usize, scaling_factor: f32) -> Result<Routing, String> {
    route_bias_corrected_logits(logits, bias, top_k, scaling_factor, "router", |logit| 1.0 / (1.0 + (-logit).exp()))
}

/// DeepSeek-V4 路由：`sqrt(softplus(logit))` 作为原始权重，bias 只参与 top-k。
pub fn route_sqrt_softplus_bias_logits(logits: &[f32], bias: &[f32], top_k: usize, scaling_factor: f32) -> Result<Routing, String> {
    route_bias_corrected_logits(logits, bias, top_k, scaling_factor, "sqrt-softplus router", sqrt_softplus_score)
}

/// Token-Hash 只固定专家选择，权重仍来自对应专家的 sqrt(softplus(logit))。
#[allow(clippy::too_many_arguments)]
pub fn route_sqrt_softplus_selected(input: &[f32], rows: usize, columns: usize, weight: &[f32], expert_count: usize, selected_experts: &[u32], top_k: usize, scaling_factor: f32) -> Result<MoePrefillRouting, String> {
    if input.len() != rows.checked_mul(columns).ok_or("固定路由 input 大小溢出")?
        || weight.len() != expert_count.checked_mul(columns).ok_or("固定路由 weight 大小溢出")?
        || selected_experts.len() != rows.checked_mul(top_k).ok_or("固定路由 ID 大小溢出")?
        || top_k == 0
        || !scaling_factor.is_finite()
    {
        return Err("固定专家路由 shape 非法".to_owned());
    }
    let mut weights = Vec::with_capacity(selected_experts.len());
    for row in 0..rows {
        let x = &input[row * columns..(row + 1) * columns];
        let ids = &selected_experts[row * top_k..(row + 1) * top_k];
        let mut raw = Vec::with_capacity(top_k);
        for &expert in ids {
            let expert = expert as usize;
            if expert >= expert_count {
                return Err(format!("固定路由 expert {expert} 越界于 {expert_count}"));
            }
            let w = &weight[expert * columns..(expert + 1) * columns];
            let logit = x.iter().zip(w).map(|(x, w)| x * w).sum::<f32>();
            raw.push(sqrt_softplus_score(logit));
        }
        let sum = raw.iter().sum::<f32>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(format!("固定路由 row {row} 权重和非法: {sum}"));
        }
        weights.extend(raw.into_iter().map(|value| value / sum * scaling_factor));
    }
    Ok(MoePrefillRouting { expert_ids: selected_experts.to_vec(), weights, rows, top_k })
}

#[derive(Debug, Clone)]
pub struct MoePrefillRouting {
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
    pub rows: usize,
    pub top_k: usize,
}

impl MoePrefillRouting {
    pub fn grouped(&self, expert_count: usize) -> Result<ExpertAssignments, String> {
        group_routing(&self.expert_ids, &self.weights, self.top_k, expert_count)
    }
}

pub struct ExpertPrefillBatch<T> {
    pub expert: usize,
    pub input: T,
}

fn group_routing(expert_ids: &[u32], weights: &[f32], top_k: usize, expert_count: usize) -> Result<ExpertAssignments, String> {
    if top_k == 0 || !expert_ids.len().is_multiple_of(top_k) || expert_ids.len() != weights.len() {
        return Err(format!("router shape 不一致: ids={} weights={} top_k={top_k}", expert_ids.len(), weights.len()));
    }
    let rows = expert_ids.len() / top_k;
    let route_count = expert_ids.len();
    let routes_per_expert = if expert_count == 0 { 0 } else { route_count.div_ceil(expert_count) };
    let mut assignments: ExpertAssignments = (0..expert_count).map(|_| Vec::with_capacity(routes_per_expert)).collect();
    for token in 0..rows {
        let token = u32::try_from(token).map_err(|_| format!("router token {token} 超出 u32"))?;
        for slot in 0..top_k {
            let route = token as usize * top_k + slot;
            let expert = expert_ids[route] as usize;
            if expert >= expert_count {
                return Err(format!("router 返回越界专家 {expert}"));
            }
            assignments[expert].push((token, weights[route]));
        }
    }
    Ok(assignments)
}

pub fn active_experts(assignments: &ExpertAssignments) -> Result<Vec<u16>, String> {
    assignments.iter().enumerate().filter(|(_, rows)| !rows.is_empty()).map(|(expert, _)| u16::try_from(expert).map_err(|_| format!("expert {expert} 超出 u16"))).collect()
}

/// 从 flat router expert_ids 去重提取被激活的专家下标（按升序）。
/// 用于 decode/prefill 输出 `active_experts` 供下游专家预取。
pub fn active_experts_from_ids(expert_ids: &[u32], expert_count: usize) -> Result<Vec<usize>, String> {
    let mut active = vec![false; expert_count];
    for &expert in expert_ids {
        let expert = expert as usize;
        if expert >= expert_count {
            return Err(format!("router 返回越界专家 {expert}"));
        }
        active[expert] = true;
    }
    Ok(active.into_iter().enumerate().filter_map(|(expert, active)| active.then_some(expert)).collect())
}

/// 执行已经按专家分组的 routed expert 数据流。
///
/// 权重加载和 expert kernel 由回调提供；gather、加权 scatter 与累加顺序在所有
/// backend 上保持一致。单 token decode 直接复用输入，避免无意义的设备内拷贝。
pub fn execute_routed_experts<B, F>(backend: &B, input: &B::Tensor, assignments: &[Vec<(u32, f32)>], output_rows: usize, output_cols: usize, mut execute: F) -> Result<B::Tensor, BackendError>
where
    B: MoePrefillBackend,
    F: FnMut(usize, &B::Tensor) -> Result<B::Tensor, BackendError>,
{
    let mut output = backend.moe_zeros(output_rows, output_cols)?;
    if output_rows > 1 {
        let active = assignments.iter().enumerate().filter_map(|(expert, assigned)| (!assigned.is_empty()).then_some((expert, assigned))).collect::<Vec<_>>();
        let rows = active.iter().map(|(_, assigned)| assigned.iter().map(|&(row, _)| row).collect::<Vec<_>>()).collect::<Vec<_>>();
        let weights = active.iter().map(|(_, assigned)| assigned.iter().map(|&(_, weight)| weight).collect::<Vec<_>>()).collect::<Vec<_>>();
        let gathered = backend.moe_gather_rows_batch(input, &rows)?;
        if gathered.len() != active.len() {
            return Err(BackendError::Compute { msg: format!("routed expert batch gather 返回 {} 项，期望 {} 项", gathered.len(), active.len()) });
        }
        let mut computed = Vec::with_capacity(active.len());
        for ((expert, _), expert_input) in active.into_iter().zip(&gathered) {
            computed.push(execute(expert, expert_input)?);
        }
        backend.moe_scatter_add_rows_batch(&mut output, &computed, &rows, &weights)?;
        return backend.moe_finish(output);
    }
    for (expert, assigned) in assignments.iter().enumerate() {
        if assigned.is_empty() {
            continue;
        }
        let rows: Vec<u32> = assigned.iter().map(|&(row, _)| row).collect();
        let weights: Vec<f32> = assigned.iter().map(|&(_, weight)| weight).collect();
        let gathered = if output_rows == 1 && rows.as_slice() == [0] { None } else { Some(backend.moe_gather_rows(input, &rows)?) };
        let expert_input = gathered.as_ref().unwrap_or(input);
        let expert_output = execute(expert, expert_input)?;
        backend.moe_scatter_add_rows(&mut output, &expert_output, &rows, &weights)?;
    }
    backend.moe_finish(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_groups_tokens_and_lists_active_experts() {
        let routing = MoePrefillRouting { expert_ids: vec![2, 0, 2, 1], weights: vec![0.7, 0.3, 0.6, 0.4], rows: 2, top_k: 2 };
        let grouped = routing.grouped(3).unwrap();
        assert_eq!(grouped[0], vec![(0, 0.3)]);
        assert_eq!(grouped[1], vec![(1, 0.4)]);
        assert_eq!(grouped[2], vec![(0, 0.7), (1, 0.6)]);
        assert_eq!(active_experts(&grouped).unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn routing_rejects_mismatched_shape() {
        assert!(group_routing(&[0], &[], 1, 1).is_err());
    }

    #[test]
    fn sigmoid_bias_uses_bias_for_selection_and_raw_for_weights() {
        let routing = route_sigmoid_bias_logits(&[0.0, 1.0, -1.0], &[1.0, 0.0, 0.0], 2, 2.0).unwrap();
        assert_eq!(routing.experts, vec![0, 1]);
        let expected_first = 0.5 / (0.5 + 0.731_058_6) * 2.0;
        assert!((routing.weights[0] - expected_first).abs() < 1e-6);
    }

    #[test]
    fn softmax_topk_renormalizes_selected_experts() {
        let routing = route_softmax_logits(&[0.0, 2.0, 1.0], 2, 1.0, true).unwrap();
        assert_eq!(routing.experts, vec![1, 2]);
        assert!((routing.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(routing.weights[0] > routing.weights[1]);
    }

    #[test]
    fn softmax_topk_can_preserve_global_probability_mass() {
        let routing = route_softmax_logits(&[0.0, 2.0, 1.0], 2, 1.0, false).unwrap();
        assert_eq!(routing.experts, vec![1, 2]);
        let expected = (2.0f32.exp() + 1.0f32.exp()) / (2.0f32.exp() + 1.0f32.exp() + 1.0);
        assert!((routing.weights.iter().sum::<f32>() - expected).abs() < 1e-6);
    }

    #[test]
    fn sqrt_softplus_bias_separates_selection_from_weight() {
        let routing = route_sqrt_softplus_bias_logits(&[0.0, 2.0, 1.0], &[2.0, 0.0, 0.0], 2, 1.5).unwrap();
        assert_eq!(routing.experts, vec![0, 1]);
        assert!((routing.weights.iter().sum::<f32>() - 1.5).abs() < 1e-6);
        assert!(routing.weights[1] > routing.weights[0]);
    }
}
