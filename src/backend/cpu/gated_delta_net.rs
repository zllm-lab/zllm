//! Gated DeltaNet CPU fused kernel capability。

use rayon::prelude::*;

use crate::{
    attention::gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetInputs, GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetStorage, GatedDeltaNetWeightsRef, GdnOutputGate},
    backend::cpu::{CpuContext, CpuWeight},
    backend::{BackendError, compute_error as compute},
    kernel::cpu::{CpuTensor, sigmoid, softplus},
};

pub struct CpuGatedDeltaNetStorage {
    pub(crate) conv: Vec<f32>,
    pub(crate) recurrent: Vec<f32>,
}

/// 以 `[key, value]` 连续行更新 recurrent，避免按 value 列跨步读取。
fn recurrent_head_step(recurrent: &mut [f32], output: &mut [f32], value: &mut [f32], key: &[f32], query: &[f32], beta: f32) {
    let value_dim = value.len();
    output.fill(0.0);
    for (key_column, state_row) in recurrent.chunks_exact(value_dim).enumerate() {
        let key = key[key_column];
        for value_column in 0..value_dim {
            output[value_column] += state_row[value_column] * key;
        }
    }
    for value_column in 0..value_dim {
        value[value_column] = (value[value_column] - output[value_column]) * beta;
        output[value_column] = 0.0;
    }
    for (key_column, state_row) in recurrent.chunks_exact_mut(value_dim).enumerate() {
        let key = key[key_column];
        let query = query[key_column];
        for value_column in 0..value_dim {
            state_row[value_column] += key * value[value_column];
            output[value_column] += state_row[value_column] * query;
        }
    }
}

impl GatedDeltaNetStorage for CpuGatedDeltaNetStorage {
    fn allocated_bytes(&self) -> usize {
        (self.conv.len() + self.recurrent.len()) * std::mem::size_of::<f32>()
    }
}

impl GatedDeltaNetKernel for CpuContext {
    type GatedDeltaNetStorage = CpuGatedDeltaNetStorage;

    fn allocate_gated_delta_net_storage(&self, spec: &GatedDeltaNetSpec) -> Result<Self::GatedDeltaNetStorage, BackendError> {
        Ok(CpuGatedDeltaNetStorage { conv: vec![0.0; spec.conv_state_elements()], recurrent: vec![0.0; spec.recurrent_elements()] })
    }

    fn gated_delta_net_fused(&self, storage: &mut Self::GatedDeltaNetStorage, inputs: GatedDeltaNetInputs<'_, CpuTensor>, weights: GatedDeltaNetWeightsRef<'_, CpuWeight>, spec: &GatedDeltaNetSpec) -> Result<CpuTensor, BackendError> {
        self.gated_delta_net_fused_layout(storage, inputs, weights, GatedDeltaNetHeadLayout::Tiled, spec)
    }

    fn gated_delta_net_fused_layout(
        &self,
        storage: &mut Self::GatedDeltaNetStorage,
        inputs: GatedDeltaNetInputs<'_, CpuTensor>,
        weights: GatedDeltaNetWeightsRef<'_, CpuWeight>,
        head_layout: GatedDeltaNetHeadLayout,
        spec: &GatedDeltaNetSpec,
    ) -> Result<CpuTensor, BackendError> {
        let GatedDeltaNetInputs { qkv, z, alpha, beta } = inputs;
        let GatedDeltaNetWeightsRef { conv: conv_weight, a_log, dt_bias, norm: norm_weight } = weights;
        if (conv_weight.rows(), conv_weight.cols()) != (spec.conv_dim(), spec.conv_kernel) || a_log.data().len() != spec.value_heads || dt_bias.data().len() != spec.value_heads || norm_weight.data().len() != spec.value_head_dim {
            return Err(compute("CPU Gated DeltaNet weight shape 与 spec 不一致"));
        }

        let rows = qkv.rows;
        let key_dim = spec.key_dim();
        let value_dim = spec.value_dim();
        let conv_dim = spec.conv_dim();
        // CPU 是跨后端一致性的 oracle：输入张量 shape 不符要返回错误，不能让切片越界 panic。
        if qkv.cols != conv_dim || z.cols != value_dim || alpha.cols != spec.value_heads || beta.cols != spec.value_heads {
            return Err(compute(format!(
                "CPU Gated DeltaNet 输入列数与 spec 不符: qkv.cols={} 期望 {conv_dim}, z.cols={} 期望 {value_dim}, alpha.cols={} 期望 {}, beta.cols={} 期望 {}",
                qkv.cols, z.cols, alpha.cols, spec.value_heads, beta.cols, spec.value_heads
            )));
        }
        if z.rows != rows || alpha.rows != rows || beta.rows != rows {
            return Err(compute(format!("CPU Gated DeltaNet 输入行数与 qkv 不符: qkv.rows={rows}, z.rows={}, alpha.rows={}, beta.rows={}", z.rows, alpha.rows, beta.rows)));
        }
        let mut output = vec![0.0; rows * value_dim];
        let mut mixed = vec![0.0; conv_dim];
        let mut query = vec![0.0; key_dim];
        let mut key = vec![0.0; key_dim];
        let mut core = vec![0.0; value_dim];
        let recurrent_head_stride = spec.key_head_dim * spec.value_head_dim;
        let query_scale = 1.0 / (spec.key_head_dim as f32).sqrt();
        let value_head_dim = spec.value_head_dim;
        let key_head_dim = spec.key_head_dim;

        // 把 value_heads 和 conv_channels 的内层循环改成 rayon,并在 token 维度保持串行
        // 因为它们共享 storage.recurrent / storage.conv 状态。
        for token in 0..rows {
            let qkv_row = &qkv.data[token * conv_dim..(token + 1) * conv_dim];

            // Conv: 跨 channel 并行,每个 channel 有独立的 conv_kernel 状态区。
            // 但 copy_within 对同一 Vec 的别名调用不是 Sync,所以按 channel 分块串行
            // 写入。改成 thread-local 缓冲再聚合会让小 conv_kernel 收益 < 开销。
            // 10240 channels × conv_kernel=4: 单纯顺序也比 fused recurrent step 小很多,
            // 留给后续 inner-SIMD 优化。
            for channel in 0..conv_dim {
                let begin = channel * spec.conv_kernel;
                let end = begin + spec.conv_kernel;
                storage.conv.copy_within(begin + 1..end, begin);
                storage.conv[end - 1] = qkv_row[channel];
                let sum = storage.conv[begin..end].iter().zip(&conv_weight.data()[begin..end]).map(|(value, weight)| value * weight).sum::<f32>();
                mixed[channel] = sum * sigmoid(sum);
            }

            query.copy_from_slice(&mixed[..key_dim]);
            key.copy_from_slice(&mixed[key_dim..key_dim * 2]);
            for head in 0..spec.key_heads {
                let begin = head * key_head_dim;
                let end = begin + key_head_dim;
                let q_inv = 1.0 / (query[begin..end].iter().map(|value| value * value).sum::<f32>() + 1e-6).sqrt();
                let k_inv = 1.0 / (key[begin..end].iter().map(|value| value * value).sum::<f32>() + 1e-6).sqrt();
                for index in begin..end {
                    query[index] *= q_inv * query_scale;
                    key[index] *= k_inv;
                }
            }

            let value = &mut mixed[key_dim * 2..];
            let alpha_row = &alpha.data[token * spec.value_heads..(token + 1) * spec.value_heads];
            let beta_row = &beta.data[token * spec.value_heads..(token + 1) * spec.value_heads];

            // 跨 value_head 并行 — 每个 head 拥有独立的 recurrent[head_begin..] 区间
            // 与 core 切片,无数据竞争。Q/K/V/alpha_row/beta_row/a_log/dt_bias 都是只读共享。
            // rayon 的 par_iter 是 Fn,先把 recurrent 与 core 拆成 per-head disjoint 切片,
            // 闭包捕获独立的 &mut 切片。
            let head_chunks = core.chunks_exact_mut(value_head_dim).zip(storage.recurrent.chunks_exact_mut(recurrent_head_stride)).zip(value.chunks_exact_mut(value_head_dim)).collect::<Vec<_>>();
            let query_ref = &query;
            let key_ref = &key;
            let alpha_ref = alpha_row;
            let beta_ref = beta_row;
            let a_log_data = a_log.data();
            let dt_bias_data = dt_bias.data();

            head_chunks.into_par_iter().enumerate().for_each(|(value_head, ((out, recurrent_chunk), value))| {
                let key_head = head_layout.key_head_for_value(spec, value_head);
                let q = &query_ref[key_head * key_head_dim..(key_head + 1) * key_head_dim];
                let k = &key_ref[key_head * key_head_dim..(key_head + 1) * key_head_dim];
                let decay = (-a_log_data[value_head].exp() * softplus(alpha_ref[value_head] + dt_bias_data[value_head])).exp();
                for item in recurrent_chunk.iter_mut() {
                    *item *= decay;
                }
                let beta = sigmoid(beta_ref[value_head]);
                recurrent_head_step(recurrent_chunk, out, value, k, q, beta);
            });

            let z_row = &z.data[token * value_dim..(token + 1) * value_dim];
            let output_row = &mut output[token * value_dim..(token + 1) * value_dim];
            // 输出 norm + gate: 拆成 per-head disjoint 切片后再跨 head 并行。
            let core_chunks_read: Vec<&[f32]> = core.chunks_exact(value_head_dim).collect();
            let output_chunks: Vec<&mut [f32]> = output_row.chunks_exact_mut(value_head_dim).collect();
            let z_ref = z_row;
            let norm_ref = norm_weight.data();

            core_chunks_read.into_par_iter().zip(output_chunks.into_par_iter()).enumerate().for_each(|(head, (core_chunk, out_chunk))| {
                let variance = core_chunk.iter().map(|value| value * value).sum::<f32>() / value_head_dim as f32;
                let inv_rms = 1.0 / (variance + spec.rms_eps).sqrt();
                match spec.output_gate {
                    GdnOutputGate::Silu => {
                        for column in 0..value_head_dim {
                            let z = z_ref[head * value_head_dim + column];
                            out_chunk[column] = core_chunk[column] * inv_rms * norm_ref[column] * z * sigmoid(z);
                        }
                    }
                    GdnOutputGate::Sigmoid => {
                        for column in 0..value_head_dim {
                            out_chunk[column] = core_chunk[column] * inv_rms * norm_ref[column] * sigmoid(z_ref[head * value_head_dim + column]);
                        }
                    }
                }
            });
        }
        Ok(CpuTensor { data: output, rows, cols: value_dim })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendResources;

    #[test]
    fn recurrent连续行更新匹配逐列reference() {
        let key = [0.25, -0.5, 0.75, -1.0];
        let query = [-0.125, 0.375, -0.625, 0.875];
        let mut state = (0..12).map(|index| index as f32 * 0.03125 - 0.2).collect::<Vec<_>>();
        let mut expected_state = state.clone();
        let mut value = [0.3, -0.4, 0.5];
        let mut expected = [0.0; 3];
        let beta = 0.6;
        for value_column in 0..value.len() {
            let mut memory = 0.0;
            for key_column in 0..key.len() {
                memory += expected_state[key_column * value.len() + value_column] * key[key_column];
            }
            let delta = (value[value_column] - memory) * beta;
            for key_column in 0..key.len() {
                let item = &mut expected_state[key_column * value.len() + value_column];
                *item += key[key_column] * delta;
                expected[value_column] += *item * query[key_column];
            }
        }
        let mut actual = [0.0; 3];
        recurrent_head_step(&mut state, &mut actual, &mut value, &key, &query, beta);
        assert_eq!(state, expected_state);
        assert_eq!(actual, expected);
    }

    #[test]
    fn fused_kernel_obeys_head_layout() {
        let backend = CpuContext;
        let spec = GatedDeltaNetSpec { key_heads: 2, value_heads: 4, key_head_dim: 2, value_head_dim: 1, conv_kernel: 1, rms_eps: 1e-6, output_gate: GdnOutputGate::Silu };
        // value head 1 在 grouped/tiled 下分别读取 key head 0/1；两组 Q/K
        // 分别平行和正交，确保数值结果能锁定映射，而不只测试 helper。
        let qkv = CpuTensor { data: vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], rows: 1, cols: spec.conv_dim() };
        let z = CpuTensor { data: vec![1.0; spec.value_dim()], rows: 1, cols: spec.value_dim() };
        let alpha = CpuTensor { data: vec![0.0; spec.value_heads], rows: 1, cols: spec.value_heads };
        let beta = alpha.clone();
        let conv = backend.prepare_f32(&vec![1.0; spec.conv_state_elements()], spec.conv_dim(), spec.conv_kernel).unwrap();
        let a_log = backend.prepare_f32(&vec![0.0; spec.value_heads], 1, spec.value_heads).unwrap();
        let dt_bias = a_log.clone();
        let norm = backend.prepare_f32(&[1.0], 1, spec.value_head_dim).unwrap();
        let inputs = GatedDeltaNetInputs { qkv: &qkv, z: &z, alpha: &alpha, beta: &beta };
        let weights = GatedDeltaNetWeightsRef { conv: &conv, a_log: &a_log, dt_bias: &dt_bias, norm: &norm };

        let mut grouped_state = backend.allocate_gated_delta_net_storage(&spec).unwrap();
        let grouped = backend.gated_delta_net_fused_layout(&mut grouped_state, inputs.clone(), weights.clone(), GatedDeltaNetHeadLayout::Grouped, &spec).unwrap();
        let mut tiled_state = backend.allocate_gated_delta_net_storage(&spec).unwrap();
        let tiled = backend.gated_delta_net_fused_layout(&mut tiled_state, inputs, weights, GatedDeltaNetHeadLayout::Tiled, &spec).unwrap();

        assert!(grouped.data[1].abs() > 0.5, "grouped value head 1 应读取平行的 key head 0: {}", grouped.data[1]);
        assert!(tiled.data[1].abs() < 1e-5, "tiled value head 1 应读取正交的 key head 1: {}", tiled.data[1]);
    }
}
