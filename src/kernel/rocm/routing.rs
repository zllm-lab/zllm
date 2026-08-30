use crate::moe::routing::{Routing, route_sigmoid_bias_logits, route_softmax_logits, route_sqrt_softplus_bias_logits};

pub fn route_softmax(input: &[f32], weight: &[f32], num_experts: usize, top_k: usize, scaling_factor: f32, normalize_selected: bool) -> Result<Routing, String> {
    let logits = router_logits(input, weight, num_experts)?;
    route_softmax_logits(&logits, top_k, scaling_factor, normalize_selected).map_err(|error| format!("ROCm softmax route 失败: {error}"))
}

pub fn route_sigmoid_bias(input: &[f32], weight: &[f32], bias: &[f32], num_experts: usize, top_k: usize, scaling_factor: f32) -> Result<Routing, String> {
    let logits = router_logits(input, weight, num_experts)?;
    route_sigmoid_bias_logits(&logits, bias, top_k, scaling_factor).map_err(|error| format!("ROCm sigmoid-bias route 失败: {error}"))
}

pub fn route_sqrt_softplus_bias(input: &[f32], weight: &[f32], bias: &[f32], num_experts: usize, top_k: usize, scaling_factor: f32) -> Result<Routing, String> {
    let logits = router_logits(input, weight, num_experts)?;
    route_sqrt_softplus_bias_logits(&logits, bias, top_k, scaling_factor).map_err(|error| format!("ROCm sqrt-softplus-bias route 失败: {error}"))
}

fn router_logits(input: &[f32], weight: &[f32], num_experts: usize) -> Result<Vec<f32>, String> {
    let hidden = input.len();
    let expected = num_experts.checked_mul(hidden).ok_or_else(|| "ROCm router 特征维度溢出".to_owned())?;
    if weight.len() != expected {
        return Err("ROCm router weight 形状应与 [num_experts, hidden] 匹配".to_owned());
    }
    let mut logits = Vec::with_capacity(num_experts);
    for expert in 0..num_experts {
        let base = expert * hidden;
        let mut value = 0.0;
        if !input.is_empty() && !weight.is_empty() {
            for (&left, &right) in input.iter().zip(&weight[base..base + hidden]) {
                let left = if left.is_finite() { left } else { 0.0 };
                let right = if right.is_finite() { right } else { 0.0 };
                value += left * right;
            }
        }
        if !value.is_finite() {
            value = 0.0;
        }
        logits.push(value);
    }
    Ok(logits)
}
