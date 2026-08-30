//! KDA CPU recurrent reference 算子。

use crate::{
    attention::kda::KdaSpec,
    kernel::cpu::{CpuTensor, sigmoid, softplus},
};

fn short_conv_silu(input: &[f32], state: &mut [f32], weight: &[f32], channels: usize, kernel_size: usize, output: &mut [f32]) {
    let history = kernel_size - 1;
    for channel in 0..channels {
        let state_begin = channel * history;
        let weight_begin = channel * kernel_size;
        let mut sum = input[channel] * weight[weight_begin + history];
        for index in 0..history {
            sum += state[state_begin + index] * weight[weight_begin + index];
        }
        if history > 0 {
            state.copy_within(state_begin + 1..state_begin + history, state_begin);
            state[state_begin + history - 1] = input[channel];
        }
        output[channel] = sum * sigmoid(sum);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn recurrent(
    query: &CpuTensor,
    key: &CpuTensor,
    value: &CpuTensor,
    decay: &CpuTensor,
    beta: &CpuTensor,
    output_gate: &CpuTensor,
    query_conv_state: &mut [f32],
    key_conv_state: &mut [f32],
    value_conv_state: &mut [f32],
    recurrent_state: &mut [f32],
    query_conv_weight: &[f32],
    key_conv_weight: &[f32],
    value_conv_weight: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    output_norm_weight: &[f32],
    spec: &KdaSpec,
) -> Result<CpuTensor, String> {
    let rows = query.rows;
    let projection_size = spec.projection_size();
    let conv_elements = projection_size * (spec.short_conv_kernel_size - 1);
    let projected = [key, value, decay, output_gate].iter().all(|tensor| tensor.rows == rows && tensor.cols == projection_size);
    if query.cols != projection_size
        || !projected
        || beta.rows != rows
        || beta.cols != spec.num_heads
        || query_conv_state.len() != conv_elements
        || key_conv_state.len() != conv_elements
        || value_conv_state.len() != conv_elements
        || recurrent_state.len() != spec.recurrent_state_elements()
        || query_conv_weight.len() != projection_size * spec.short_conv_kernel_size
        || key_conv_weight.len() != projection_size * spec.short_conv_kernel_size
        || value_conv_weight.len() != projection_size * spec.short_conv_kernel_size
        || a_log.len() != spec.num_heads
        || dt_bias.len() != projection_size
        || output_norm_weight.len() != spec.head_dim
    {
        return Err("CPU KDA tensor/weight/state shape 与 spec 不一致".to_owned());
    }

    let mut output = vec![0.0; rows * projection_size];
    let mut q = vec![0.0; projection_size];
    let mut k = vec![0.0; projection_size];
    let mut v = vec![0.0; projection_size];
    let mut core = vec![0.0; projection_size];
    let query_scale = 1.0 / (spec.head_dim as f32).sqrt();
    let recurrent_head_stride = spec.head_dim * spec.head_dim;

    for token in 0..rows {
        let begin = token * projection_size;
        let end = begin + projection_size;
        short_conv_silu(&query.data[begin..end], query_conv_state, query_conv_weight, projection_size, spec.short_conv_kernel_size, &mut q);
        short_conv_silu(&key.data[begin..end], key_conv_state, key_conv_weight, projection_size, spec.short_conv_kernel_size, &mut k);
        short_conv_silu(&value.data[begin..end], value_conv_state, value_conv_weight, projection_size, spec.short_conv_kernel_size, &mut v);

        for head in 0..spec.num_heads {
            let head_begin = head * spec.head_dim;
            let head_end = head_begin + spec.head_dim;
            if spec.use_qk_l2norm {
                let q_inv = 1.0 / (q[head_begin..head_end].iter().map(|value| value * value).sum::<f32>() + 1.0e-6).sqrt();
                let k_inv = 1.0 / (k[head_begin..head_end].iter().map(|value| value * value).sum::<f32>() + 1.0e-6).sqrt();
                for column in head_begin..head_end {
                    q[column] *= q_inv;
                    k[column] *= k_inv;
                }
            }
            let state_begin = head * recurrent_head_stride;
            let state = &mut recurrent_state[state_begin..state_begin + recurrent_head_stride];
            for key_column in 0..spec.head_dim {
                let raw_gate = decay.data[begin + head_begin + key_column] + dt_bias[head_begin + key_column];
                let log_decay = match spec.gate_lower_bound {
                    Some(lower_bound) => lower_bound * sigmoid(a_log[head].exp() * raw_gate),
                    None => -a_log[head].exp() * softplus(raw_gate),
                };
                let factor = log_decay.exp();
                for value_column in 0..spec.head_dim {
                    state[key_column * spec.head_dim + value_column] *= factor;
                }
            }
            let beta = sigmoid(beta.data[token * spec.num_heads + head]);
            for value_column in 0..spec.head_dim {
                let mut predicted = 0.0;
                for key_column in 0..spec.head_dim {
                    predicted += state[key_column * spec.head_dim + value_column] * k[head_begin + key_column];
                }
                let delta = (v[head_begin + value_column] - predicted) * beta;
                let mut mixed = 0.0;
                for key_column in 0..spec.head_dim {
                    let item = &mut state[key_column * spec.head_dim + value_column];
                    *item += k[head_begin + key_column] * delta;
                    mixed += q[head_begin + key_column] * query_scale * *item;
                }
                core[head_begin + value_column] = mixed;
            }

            let variance = core[head_begin..head_end].iter().map(|value| value * value).sum::<f32>() / spec.head_dim as f32;
            let inv_rms = 1.0 / (variance + spec.output_norm_eps).sqrt();
            for column in 0..spec.head_dim {
                let index = head_begin + column;
                output[begin + index] = core[index] * inv_rms * output_norm_weight[column] * sigmoid(output_gate.data[begin + index]);
            }
        }
    }
    Ok(CpuTensor { data: output, rows, cols: projection_size })
}

/// WY 表示的 chunk 并行 KDA reference。
///
/// 与 [`recurrent`] 数学等价:chunk 内把逐 token 的 delta rule 递归改写为
/// (I+L)^{-1} 三角线性解(U 即 WY 表示的 δ 序列),chunk 间仍按序推进 state。
/// 作为 GPU chunked kernel 的 oracle,也验证算法推导本身。
///
/// 推导(每 head,S ∈ R^{key×value},行 j 为 key 通道):
///   S_s = Λ_s S_{s-1} + k̂_s δ_sᵀ,  δ_s = β_s (ṽ_s − S_{s-1}ᵀ Λ_s k̂_s)
///   ⇒ S_s = D_s S_0 + Σ_{r≤s} diag(w(r→s)) k̂_r δ_rᵀ,  w(r→s) = e^{cum_s − cum_r}
///   δ = (I+L)^{-1} diag(β)(V − Q0),  L[s][r] = β_s Σ_j k̂_r[j] k̂_s[j] w(r→s)[j]
///   读出 mixed_s = scale·S_sᵀ q̂_s;输出 RMSNorm+gate 与串行版一致。
/// 所有 w 因子 ≤ 1(chunk 内 cum 单调不增),指数不会向上溢出。
#[allow(clippy::too_many_arguments)]
pub fn recurrent_chunked(
    query: &CpuTensor,
    key: &CpuTensor,
    value: &CpuTensor,
    decay: &CpuTensor,
    beta: &CpuTensor,
    output_gate: &CpuTensor,
    query_conv_state: &mut [f32],
    key_conv_state: &mut [f32],
    value_conv_state: &mut [f32],
    recurrent_state: &mut [f32],
    query_conv_weight: &[f32],
    key_conv_weight: &[f32],
    value_conv_weight: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    output_norm_weight: &[f32],
    spec: &KdaSpec,
    chunk_size: usize,
) -> Result<CpuTensor, String> {
    let rows = query.rows;
    let projection_size = spec.projection_size();
    let heads = spec.num_heads;
    let dim = spec.head_dim;
    let conv_elements = projection_size * (spec.short_conv_kernel_size - 1);
    let projected = [key, value, decay, output_gate].iter().all(|tensor| tensor.rows == rows && tensor.cols == projection_size);
    if query.cols != projection_size
        || !projected
        || beta.rows != rows
        || beta.cols != heads
        || query_conv_state.len() != conv_elements
        || key_conv_state.len() != conv_elements
        || value_conv_state.len() != conv_elements
        || recurrent_state.len() != spec.recurrent_state_elements()
        || query_conv_weight.len() != projection_size * spec.short_conv_kernel_size
        || key_conv_weight.len() != projection_size * spec.short_conv_kernel_size
        || value_conv_weight.len() != projection_size * spec.short_conv_kernel_size
        || a_log.len() != heads
        || dt_bias.len() != projection_size
        || output_norm_weight.len() != dim
        || chunk_size == 0
    {
        return Err("CPU KDA chunked tensor/weight/state shape 与 spec 不一致".to_owned());
    }

    let mut output = vec![0.0f32; rows * projection_size];
    let query_scale = 1.0f32 / (dim as f32).sqrt();
    let state_head_stride = dim * dim;
    // conv + SiLU 保持 token 顺序(shift register 语义),结果缓存整 chunk 供 head 循环复用。
    let mut conv_q = vec![0.0f32; chunk_size * projection_size];
    let mut conv_k = vec![0.0f32; chunk_size * projection_size];
    let mut conv_v = vec![0.0f32; chunk_size * projection_size];
    // 每 head 的 chunk 工作区:累计 log decay、归一化 q̂/k̂、β、三角矩阵、WY 解。
    let mut cum = vec![0.0f32; chunk_size * dim];
    let mut khat = vec![0.0f32; chunk_size * dim];
    let mut qhat = vec![0.0f32; chunk_size * dim];
    let mut betas = vec![0.0f32; chunk_size];
    let mut tbeta = vec![0.0f32; chunk_size * chunk_size];
    let mut aq = vec![0.0f32; chunk_size * chunk_size];
    let mut u = vec![0.0f32; chunk_size * dim];
    let mut rhs = vec![0.0f32; chunk_size * dim];
    let mut mixed = vec![0.0f32; dim];

    for chunk_start in (0..rows).step_by(chunk_size) {
        let chunk = chunk_size.min(rows - chunk_start);
        for token in 0..chunk {
            let begin = (chunk_start + token) * projection_size;
            let end = begin + projection_size;
            short_conv_silu(&query.data[begin..end], query_conv_state, query_conv_weight, projection_size, spec.short_conv_kernel_size, &mut conv_q[token * projection_size..(token + 1) * projection_size]);
            short_conv_silu(&key.data[begin..end], key_conv_state, key_conv_weight, projection_size, spec.short_conv_kernel_size, &mut conv_k[token * projection_size..(token + 1) * projection_size]);
            short_conv_silu(&value.data[begin..end], value_conv_state, value_conv_weight, projection_size, spec.short_conv_kernel_size, &mut conv_v[token * projection_size..(token + 1) * projection_size]);
        }
        for head in 0..heads {
            let head_begin = head * dim;
            let a = a_log[head].exp();
            for token in 0..chunk {
                betas[token] = sigmoid(beta.data[(chunk_start + token) * heads + head]);
                let q_slice = &conv_q[token * projection_size + head_begin..token * projection_size + head_begin + dim];
                let k_slice = &conv_k[token * projection_size + head_begin..token * projection_size + head_begin + dim];
                if spec.use_qk_l2norm {
                    let q_inv = 1.0 / (q_slice.iter().map(|value| value * value).sum::<f32>() + 1.0e-6).sqrt();
                    let k_inv = 1.0 / (k_slice.iter().map(|value| value * value).sum::<f32>() + 1.0e-6).sqrt();
                    for column in 0..dim {
                        qhat[token * dim + column] = q_slice[column] * q_inv;
                        khat[token * dim + column] = k_slice[column] * k_inv;
                    }
                } else {
                    qhat[token * dim..token * dim + dim].copy_from_slice(q_slice);
                    khat[token * dim..token * dim + dim].copy_from_slice(k_slice);
                }
                // chunk 内累计 log decay(含当前 token);token 0 直接落 cum,之后累加。
                for column in 0..dim {
                    let raw_gate = decay.data[(chunk_start + token) * projection_size + head_begin + column] + dt_bias[head_begin + column];
                    let log_decay = match spec.gate_lower_bound {
                        Some(lower_bound) => lower_bound * sigmoid(a * raw_gate),
                        None => -a * softplus(raw_gate),
                    };
                    cum[token * dim + column] = if token == 0 { log_decay } else { cum[(token - 1) * dim + column] + log_decay };
                }
            }
            let state = &mut recurrent_state[head * state_head_stride..head * state_head_stride + state_head_stride];
            // rhs[s][c] = β_s (ṽ_s[c] − Σ_j state[j][c]·k̂_s[j]·e^{cum_s[j]})。
            for token in 0..chunk {
                for column in 0..dim {
                    let mut q0 = 0.0;
                    for key_column in 0..dim {
                        q0 += state[key_column * dim + column] * khat[token * dim + key_column] * cum[token * dim + key_column].exp();
                    }
                    rhs[token * dim + column] = betas[token] * (conv_v[token * projection_size + head_begin + column] - q0);
                }
            }
            // L[s][r] = β_s Σ_j k̂_r[j] k̂_s[j] e^{cum_s[j] − cum_r[j]}(r<s)。
            for token in 1..chunk {
                for prior in 0..token {
                    let mut sum = 0.0;
                    for column in 0..dim {
                        sum += khat[prior * dim + column] * khat[token * dim + column] * (cum[token * dim + column] - cum[prior * dim + column]).exp();
                    }
                    tbeta[token * chunk_size + prior] = betas[token] * sum;
                }
            }
            // 前向替代解 U = (I+L)^{-1} rhs。
            for token in 0..chunk {
                for column in 0..dim {
                    let mut acc = rhs[token * dim + column];
                    for prior in 0..token {
                        acc -= tbeta[token * chunk_size + prior] * u[prior * dim + column];
                    }
                    u[token * dim + column] = acc;
                }
            }
            // 读出矩阵 aq[s][r] = Σ_j q̂_s[j] k̂_r[j] e^{cum_s[j] − cum_r[j]}(含对角)。
            for token in 0..chunk {
                for prior in 0..=token {
                    let mut sum = 0.0;
                    for column in 0..dim {
                        let weight = if prior == token { 1.0 } else { (cum[token * dim + column] - cum[prior * dim + column]).exp() };
                        sum += qhat[token * dim + column] * khat[prior * dim + column] * weight;
                    }
                    aq[token * chunk_size + prior] = sum;
                }
            }
            // mixed_s = scale·(state 读出 + Σ_{r≤s} aq·U),随后 RMSNorm+gate 落 output。
            for token in 0..chunk {
                for column in 0..dim {
                    let mut inter = 0.0;
                    for key_column in 0..dim {
                        inter += state[key_column * dim + column] * qhat[token * dim + key_column] * cum[token * dim + key_column].exp();
                    }
                    let mut intra = 0.0;
                    for prior in 0..=token {
                        intra += aq[token * chunk_size + prior] * u[prior * dim + column];
                    }
                    mixed[column] = query_scale * (inter + intra);
                }
                let variance = mixed.iter().map(|value| value * value).sum::<f32>() / dim as f32;
                let inv_rms = 1.0 / (variance + spec.output_norm_eps).sqrt();
                for column in 0..dim {
                    let index = (chunk_start + token) * projection_size + head_begin + column;
                    output[index] = mixed[column] * inv_rms * output_norm_weight[column] * sigmoid(output_gate.data[index]);
                }
            }
            // state 推进:S ← e^{cum_end} ⊙ S + Σ_s (k̂_s ⊙ e^{cum_end − cum_s}) u_sᵀ。
            let cum_end = &cum[(chunk - 1) * dim..chunk * dim];
            for key_column in 0..dim {
                let end = cum_end[key_column].exp();
                for column in 0..dim {
                    let mut sum = state[key_column * dim + column] * end;
                    for token in 0..chunk {
                        let after = (cum_end[key_column] - cum[token * dim + key_column]).exp();
                        sum += khat[token * dim + key_column] * after * u[token * dim + column];
                    }
                    state[key_column * dim + column] = sum;
                }
            }
        }
    }
    Ok(CpuTensor { data: output, rows, cols: projection_size })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: &[f32], rows: usize) -> CpuTensor {
        CpuTensor { data: data.to_vec(), rows, cols: data.len() / rows }
    }

    /// 确定性伪随机:小 LCG,数值幅度接近真实激活。
    fn pseudo_random(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let bits = (state >> 33) as i32;
                (bits as f32 / u32::MAX as f32) * 2.0 * scale - scale
            })
            .collect()
    }

    /// chunked WY 与串行递归在随机数据上必须逐元素一致(含非零初始 conv/recurrent state,
    /// 验证跨 chunk 的 state 推进)。
    fn assert_chunked_matches_sequential(spec: &KdaSpec, rows: usize) {
        let projection = spec.projection_size();
        let conv_elements = projection * (spec.short_conv_kernel_size - 1);
        let query = tensor(&pseudo_random(11, rows * projection, 0.6), rows);
        let key = tensor(&pseudo_random(23, rows * projection, 0.5), rows);
        let value = tensor(&pseudo_random(37, rows * projection, 0.4), rows);
        let decay = tensor(&pseudo_random(53, rows * projection, 0.35), rows);
        let beta = tensor(&pseudo_random(67, rows * spec.num_heads, 1.5), rows);
        let gate = tensor(&pseudo_random(83, rows * projection, 0.3), rows);
        let conv_weight = {
            let mut weight = pseudo_random(97, projection * spec.short_conv_kernel_size, 0.25);
            // 末位权重给大值,让卷积输出不至于全被压制
            for channel in 0..projection {
                weight[channel * spec.short_conv_kernel_size + spec.short_conv_kernel_size - 1] += 1.0;
            }
            weight
        };
        let a_log = vec![0.1; spec.num_heads];
        let dt_bias = pseudo_random(101, projection, 0.1);
        let norm_weight = vec![1.0; spec.head_dim];

        // 同一份非零初始 state,分别交给两条路径。
        let initial_conv = [pseudo_random(110, conv_elements, 0.4), pseudo_random(111, conv_elements, 0.4), pseudo_random(112, conv_elements, 0.4)];
        let initial_recurrent = pseudo_random(131, spec.recurrent_state_elements(), 0.2);

        let reference = {
            let mut conv = initial_conv.clone();
            let mut state = initial_recurrent.clone();
            let [q, k, v] = &mut conv;
            recurrent(&query, &key, &value, &decay, &beta, &gate, q, k, v, &mut state, &conv_weight, &conv_weight, &conv_weight, &a_log, &dt_bias, &norm_weight, spec).unwrap()
        };
        for chunk_size in [1usize, 2, 3, 5, 16, 64, rows] {
            let mut conv = initial_conv.clone();
            let mut state = initial_recurrent.clone();
            let [q, k, v] = &mut conv;
            let chunked = recurrent_chunked(&query, &key, &value, &decay, &beta, &gate, q, k, v, &mut state, &conv_weight, &conv_weight, &conv_weight, &a_log, &dt_bias, &norm_weight, spec, chunk_size).unwrap();
            let mut max_diff = 0.0f32;
            for (index, (left, right)) in reference.data.iter().zip(chunked.data.iter()).enumerate() {
                let diff = (left - right).abs();
                assert!(diff <= 2.0e-4 + 2.0e-4 * left.abs(), "spec({spec:?}) chunk={chunk_size} 位置 {index}: 串行={left:.7} chunked={right:.7} diff={diff:.3e}");
                max_diff = max_diff.max(diff);
            }
            eprintln!("KDA chunked(chunk={chunk_size}) max_diff={max_diff:.3e} rows={rows}");
        }
    }

    #[test]
    fn chunked_wy等价于串行_glm53形态() {
        let spec = KdaSpec { num_heads: 2, head_dim: 128, short_conv_kernel_size: 4, use_full_rank_gate: false, gate_lower_bound: Some(-5.0), use_qk_l2norm: true, output_norm_eps: 1.0e-5 };
        assert_chunked_matches_sequential(&spec, 300);
    }

    #[test]
    fn chunked_wy等价于串行_softplus衰减分支() {
        let spec = KdaSpec { num_heads: 3, head_dim: 16, short_conv_kernel_size: 4, use_full_rank_gate: true, gate_lower_bound: None, use_qk_l2norm: true, output_norm_eps: 1.0e-5 };
        assert_chunked_matches_sequential(&spec, 70);
    }

    #[test]
    fn chunked_wy等价于串行_无l2归一化与conv1() {
        let spec = KdaSpec { num_heads: 2, head_dim: 32, short_conv_kernel_size: 1, use_full_rank_gate: false, gate_lower_bound: Some(-15.0), use_qk_l2norm: false, output_norm_eps: 1.0e-5 };
        assert_chunked_matches_sequential(&spec, 33);
    }

    #[test]
    fn chunked_wy终态state与串行一致() {
        let spec = KdaSpec { num_heads: 2, head_dim: 8, short_conv_kernel_size: 4, use_full_rank_gate: false, gate_lower_bound: Some(-5.0), use_qk_l2norm: true, output_norm_eps: 1.0e-5 };
        let projection = spec.projection_size();
        let rows = 37;
        let query = tensor(&pseudo_random(7, rows * projection, 0.6), rows);
        let key = tensor(&pseudo_random(17, rows * projection, 0.5), rows);
        let value = tensor(&pseudo_random(27, rows * projection, 0.4), rows);
        let decay = tensor(&pseudo_random(47, rows * projection, 0.35), rows);
        let beta = tensor(&pseudo_random(57, rows * spec.num_heads, 1.5), rows);
        let gate = tensor(&pseudo_random(77, rows * projection, 0.3), rows);
        let conv_weight = {
            let mut weight = pseudo_random(87, projection * spec.short_conv_kernel_size, 0.25);
            for channel in 0..projection {
                weight[channel * spec.short_conv_kernel_size + spec.short_conv_kernel_size - 1] += 1.0;
            }
            weight
        };
        let a_log = vec![0.1; spec.num_heads];
        let dt_bias = pseudo_random(91, projection, 0.1);
        let norm_weight = vec![1.0; spec.head_dim];
        let initial_conv = [pseudo_random(113, projection * 3, 0.4), pseudo_random(114, projection * 3, 0.4), pseudo_random(115, projection * 3, 0.4)];
        let initial_state = pseudo_random(127, spec.recurrent_state_elements(), 0.2);
        for chunk_size in [1usize, 4, 37] {
            let mut seq_conv = initial_conv.clone();
            let mut seq_state = initial_state.clone();
            let [q, k, v] = &mut seq_conv;
            recurrent(&query, &key, &value, &decay, &beta, &gate, q, k, v, &mut seq_state, &conv_weight, &conv_weight, &conv_weight, &a_log, &dt_bias, &norm_weight, &spec).unwrap();
            let mut chunk_conv = initial_conv.clone();
            let mut chunk_state = initial_state.clone();
            let [q, k, v] = &mut chunk_conv;
            recurrent_chunked(&query, &key, &value, &decay, &beta, &gate, q, k, v, &mut chunk_state, &conv_weight, &conv_weight, &conv_weight, &a_log, &dt_bias, &norm_weight, &spec, chunk_size).unwrap();
            for index in 0..spec.recurrent_state_elements() {
                let diff = (seq_state[index] - chunk_state[index]).abs();
                assert!(diff <= 2.0e-4 + 2.0e-4 * seq_state[index].abs(), "终态 state chunk={chunk_size} 位置 {index}: 串行={:.7} chunked={:.7}", seq_state[index], chunk_state[index]);
            }
            for index in 0..initial_conv.len() {
                assert!((seq_conv[0][index] - chunk_conv[0][index]).abs() < 1.0e-7, "conv state 不一致");
            }
        }
    }

    #[test]
    fn 分段执行与连续执行保持相同state语义() {
        let spec = KdaSpec { num_heads: 1, head_dim: 2, short_conv_kernel_size: 2, use_full_rank_gate: true, gate_lower_bound: Some(-5.0), use_qk_l2norm: true, output_norm_eps: 1.0e-5 };
        let inputs = tensor(&[1.0, 0.5, 0.2, 1.0], 2);
        let decay = tensor(&[0.1, -0.2, 0.3, 0.4], 2);
        let beta = tensor(&[0.0, 0.5], 2);
        let gate = tensor(&[0.2, -0.1, 0.4, 0.3], 2);
        let conv_weight = [0.0, 1.0, 0.0, 1.0];
        let mut batch_conv = [vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]];
        let mut batch_state = vec![0.0; 4];
        let [batch_q_conv, batch_k_conv, batch_v_conv] = &mut batch_conv;
        let batch = recurrent(&inputs, &inputs, &inputs, &decay, &beta, &gate, batch_q_conv, batch_k_conv, batch_v_conv, &mut batch_state, &conv_weight, &conv_weight, &conv_weight, &[0.0], &[0.0, 0.0], &[1.0, 1.0], &spec).unwrap();

        let mut split_conv = [vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]];
        let mut split_state = vec![0.0; 4];
        let mut split_output = Vec::new();
        for row in 0..2 {
            let range = row * 2..row * 2 + 2;
            let [split_q_conv, split_k_conv, split_v_conv] = &mut split_conv;
            let output = recurrent(
                &tensor(&inputs.data[range.clone()], 1),
                &tensor(&inputs.data[range.clone()], 1),
                &tensor(&inputs.data[range.clone()], 1),
                &tensor(&decay.data[range.clone()], 1),
                &tensor(&beta.data[row..row + 1], 1),
                &tensor(&gate.data[range], 1),
                split_q_conv,
                split_k_conv,
                split_v_conv,
                &mut split_state,
                &conv_weight,
                &conv_weight,
                &conv_weight,
                &[0.0],
                &[0.0, 0.0],
                &[1.0, 1.0],
                &spec,
            )
            .unwrap();
            split_output.extend(output.data);
        }
        for (left, right) in batch.data.iter().zip(split_output) {
            assert!((left - right).abs() < 1.0e-6);
        }
    }
}
