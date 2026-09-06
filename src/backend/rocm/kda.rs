//! KDA HIP kernel 实现。
//!
//! conv_state(三段 [q|k|v])与 recurrent_state 作为持久 device buffer 跨 token
//! 保持。fused kernel 每 head 一个 block 按 token 顺序执行递归,线程并行 head_dim。

use super::*;
use crate::attention::kda::{KdaInputs, KdaWeightsRef};

/// 设备 resident KDA 状态:三条短卷积 shift register(单 buffer 三段)+ recurrent delta 矩阵。
pub struct RocmKdaStorage {
    pub conv: ops::hip::DeviceBuffer,
    pub recurrent: ops::hip::DeviceBuffer,
}

impl crate::attention::kda::KdaStorage for RocmKdaStorage {
    fn allocated_bytes(&self) -> usize {
        self.conv.bytes() + self.recurrent.bytes()
    }
}

impl KdaKernel for RocmContext {
    type KdaStorage = RocmKdaStorage;

    fn allocate_kda_storage(&self, spec: &crate::attention::kda::KdaSpec) -> Result<Self::KdaStorage, BackendError> {
        // recurrent state 首次创建与 session reset 都处于层执行期，必须直接在
        // device 上清零，不能为全零常量建立数十 MiB 的 host shadow 再 H2D。
        let conv = ops::hip::try_zeros_resident_f32(self.device_id, spec.conv_state_elements()).map_err(compute_error)?;
        let recurrent = ops::hip::try_zeros_resident_f32(self.device_id, spec.recurrent_state_elements()).map_err(compute_error)?;
        Ok(RocmKdaStorage { conv, recurrent })
    }

    fn kda_fused(&self, storage: &mut Self::KdaStorage, inputs: KdaInputs<'_, Self::Tensor>, weights: KdaWeightsRef<'_, Self::Weight>, spec: &crate::attention::kda::KdaSpec) -> Result<Self::Tensor, BackendError> {
        let KdaInputs { query, key, value, decay, beta, output_gate } = inputs;
        let KdaWeightsRef { query_conv, key_conv, value_conv, a_log, dt_bias, output_norm } = weights;
        let projection_size = spec.projection_size();
        let conv_shape = (projection_size, spec.short_conv_kernel_size);
        if (query_conv.rows(), query_conv.cols()) != conv_shape
            || (key_conv.rows(), key_conv.cols()) != conv_shape
            || (value_conv.rows(), value_conv.cols()) != conv_shape
            || a_log.data().len() != spec.num_heads
            || dt_bias.data().len() != projection_size
            || output_norm.data().len() != spec.head_dim
        {
            return Err(compute_error("Rocm KDA weight shape 与 spec 不一致"));
        }

        let device_id = self.device_id;
        let rows = query.rows;

        // 输入可能来自上一张卡或为 Bf16;统一迁到本 device 并转成 F32。
        let query = self.tensor_as_f32(query.clone())?;
        let key = self.tensor_as_f32(key.clone())?;
        let value = self.tensor_as_f32(value.clone())?;
        let decay = self.tensor_as_f32(decay.clone())?;
        let beta = self.tensor_as_f32(beta.clone())?;
        let output_gate = self.tensor_as_f32(output_gate.clone())?;
        let query_device = query.device.as_deref().ok_or_else(|| compute_error("ROCm KDA query 缺少 device buffer"))?;
        let key_device = key.device.as_deref().ok_or_else(|| compute_error("ROCm KDA key 缺少 device buffer"))?;
        let value_device = value.device.as_deref().ok_or_else(|| compute_error("ROCm KDA value 缺少 device buffer"))?;
        let decay_device = decay.device.as_deref().ok_or_else(|| compute_error("ROCm KDA decay 缺少 device buffer"))?;
        let beta_device = beta.device.as_deref().ok_or_else(|| compute_error("ROCm KDA beta 缺少 device buffer"))?;
        let output_gate_device = output_gate.device.as_deref().ok_or_else(|| compute_error("ROCm KDA output_gate 缺少 device buffer"))?;

        let query_conv_device = ensure_weight_device(device_id, query_conv)?;
        let key_conv_device = ensure_weight_device(device_id, key_conv)?;
        let value_conv_device = ensure_weight_device(device_id, value_conv)?;
        let a_log_device = ensure_weight_device(device_id, a_log)?;
        let dt_bias_device = ensure_weight_device(device_id, dt_bias)?;
        let output_norm_device = ensure_weight_device(device_id, output_norm)?;

        // 多行(≥2)走 chunked WY kernel:chunk 内三角解并行替代逐 token 递归;
        // 单 token decode 与 kda_sequential 对照组走 fused 串行 kernel。
        let chunked = rows >= 2 && !ops::hip::options().kda_sequential && ops::hip::kda_chunked_prefers(spec);
        let output = if chunked {
            ops::hip::try_kda_chunked_resident_device_f32(
                device_id,
                &storage.conv,
                &storage.recurrent,
                KdaInputs { query: query_device, key: key_device, value: value_device, decay: decay_device, beta: beta_device, output_gate: output_gate_device },
                KdaWeightsRef {
                    query_conv: query_conv_device.as_ref(),
                    key_conv: key_conv_device.as_ref(),
                    value_conv: value_conv_device.as_ref(),
                    a_log: a_log_device.as_ref(),
                    dt_bias: dt_bias_device.as_ref(),
                    output_norm: output_norm_device.as_ref(),
                },
                rows,
                spec,
            )
            .map_err(compute_error)?
        } else {
            ops::hip::try_kda_fused_resident_device_f32(
                device_id,
                &storage.conv,
                &storage.recurrent,
                KdaInputs { query: query_device, key: key_device, value: value_device, decay: decay_device, beta: beta_device, output_gate: output_gate_device },
                KdaWeightsRef {
                    query_conv: query_conv_device.as_ref(),
                    key_conv: key_conv_device.as_ref(),
                    value_conv: value_conv_device.as_ref(),
                    a_log: a_log_device.as_ref(),
                    dt_bias: dt_bias_device.as_ref(),
                    output_norm: output_norm_device.as_ref(),
                },
                rows,
                spec,
            )
            .map_err(compute_error)?
        };

        Ok(device_tensor_f32(output, rows, projection_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attention::kda::{KdaInputs, KdaKernel, KdaSpec, KdaWeightsRef},
        backend::cpu::CpuContext,
        backend::{BackendResources, LinearWeight},
        kernel::cpu::CpuTensor,
    };

    /// CPU vs HIP 一致性:kimi 量级 spec(2 heads / head_dim 128 / conv 4),
    /// 确定性正弦数据,atol=rtol=1e-2;ROCm 不可用时跳过。
    /// rows=130 同时覆盖 chunked 路径(多 chunk + 尾部不完整 chunk,分派条件 rows≥2)。
    #[test]
    fn rocm_kda_matches_cpu() {
        let spec = KdaSpec { num_heads: 2, head_dim: 128, short_conv_kernel_size: 4, use_full_rank_gate: true, gate_lower_bound: Some(-5.0), use_qk_l2norm: true, output_norm_eps: 1.0e-5 };
        let rows = 130;
        let projection = spec.projection_size();
        let values = |frequency: f32, scale: f32, len: usize| (0..len).map(|index| ((index as f32 + 1.0) * frequency).sin() * scale).collect::<Vec<_>>();
        let query = values(0.013, 0.2, rows * projection);
        let key = values(0.017, 0.15, rows * projection);
        let value = values(0.019, 0.1, rows * projection);
        let decay = values(0.007, 0.3, rows * projection);
        let beta = values(0.11, 0.4, rows * spec.num_heads);
        let gate = values(0.023, 0.2, rows * projection);
        let conv_weight = (0..projection * spec.short_conv_kernel_size).map(|index| if index % spec.short_conv_kernel_size == spec.short_conv_kernel_size - 1 { 1.0 } else { 0.01 }).collect::<Vec<_>>();
        let a_log = vec![0.0; spec.num_heads];
        let dt_bias = values(0.005, 0.1, projection);
        let norm_weight = vec![1.0; spec.head_dim];

        // CPU 参考实现
        let cpu = CpuContext;
        let mut cpu_storage = cpu.allocate_kda_storage(&spec).expect("CPU KDA storage 分配失败");
        let cpu_tensor = |data: &[f32], cols: usize| CpuTensor { data: data.to_vec(), rows, cols };
        let cpu_conv_weight = cpu.prepare_weight(LinearWeight::F32(&conv_weight), projection, spec.short_conv_kernel_size).expect("CPU conv_weight prepare 失败");
        let cpu_a_log = cpu.prepare_f32(&a_log, 1, spec.num_heads).expect("CPU a_log prepare 失败");
        let cpu_dt_bias = cpu.prepare_f32(&dt_bias, 1, projection).expect("CPU dt_bias prepare 失败");
        let cpu_norm_weight = cpu.prepare_f32(&norm_weight, 1, spec.head_dim).expect("CPU norm_weight prepare 失败");
        let cpu_output = cpu
            .kda_fused(
                &mut cpu_storage,
                KdaInputs {
                    query: &cpu_tensor(&query, projection),
                    key: &cpu_tensor(&key, projection),
                    value: &cpu_tensor(&value, projection),
                    decay: &cpu_tensor(&decay, projection),
                    beta: &cpu_tensor(&beta, spec.num_heads),
                    output_gate: &cpu_tensor(&gate, projection),
                },
                KdaWeightsRef { query_conv: &cpu_conv_weight, key_conv: &cpu_conv_weight, value_conv: &cpu_conv_weight, a_log: &cpu_a_log, dt_bias: &cpu_dt_bias, output_norm: &cpu_norm_weight },
                &spec,
            )
            .expect("CPU KDA 执行失败");

        // HIP 实现
        let rocm = match RocmContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => {
                eprintln!("跳过 ROCm KDA 测试:ROCm runtime 不可用");
                return;
            }
        };
        let mut rocm_storage = rocm.allocate_kda_storage(&spec).expect("ROCm KDA storage 分配失败");
        let rocm_tensor = |data: &[f32], cols: usize| RocmTensor { data: data.to_vec(), rows, cols, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: None, replica: None };
        let rocm_conv_weight = rocm.prepare_weight(LinearWeight::F32(&conv_weight), projection, spec.short_conv_kernel_size).expect("ROCm conv_weight prepare 失败");
        let rocm_a_log = rocm.prepare_f32(&a_log, 1, spec.num_heads).expect("ROCm a_log prepare 失败");
        let rocm_dt_bias = rocm.prepare_f32(&dt_bias, 1, projection).expect("ROCm dt_bias prepare 失败");
        let rocm_norm_weight = rocm.prepare_f32(&norm_weight, 1, spec.head_dim).expect("ROCm norm_weight prepare 失败");
        let rocm_output = rocm
            .kda_fused(
                &mut rocm_storage,
                KdaInputs {
                    query: &rocm_tensor(&query, projection),
                    key: &rocm_tensor(&key, projection),
                    value: &rocm_tensor(&value, projection),
                    decay: &rocm_tensor(&decay, projection),
                    beta: &rocm_tensor(&beta, spec.num_heads),
                    output_gate: &rocm_tensor(&gate, projection),
                },
                KdaWeightsRef { query_conv: &rocm_conv_weight, key_conv: &rocm_conv_weight, value_conv: &rocm_conv_weight, a_log: &rocm_a_log, dt_bias: &rocm_dt_bias, output_norm: &rocm_norm_weight },
                &spec,
            )
            .expect("ROCm KDA 执行失败");
        let rocm_output_data = rocm.tensor_to_f32(&rocm_output).expect("下载 ROCm KDA 输出失败");

        assert_eq!(rocm_output_data.len(), cpu_output.data.len(), "ROCm KDA 输出元素数不一致");

        // 比较:BF16 精度阈值 atol=rtol=1e-2
        let tolerance = 1e-2;
        // 两条 HIP 路径都必须与 CPU 一致:上面 kda_fused 已按分派走 chunked,
        // 这里镜像其内部上传路径,用独立 state 直接调 fused 串行 kernel 复核。
        let fused_storage = rocm.allocate_kda_storage(&spec).expect("ROCm KDA fused storage 分配失败");
        let upload = |data: &[f32], cols: usize| rocm.tensor_as_f32(rocm_tensor(data, cols)).expect("ROCm KDA 测试输入上传失败");
        let fused_query = upload(&query, projection);
        let fused_key = upload(&key, projection);
        let fused_value = upload(&value, projection);
        let fused_decay = upload(&decay, projection);
        let fused_beta = upload(&beta, spec.num_heads);
        let fused_gate = upload(&gate, projection);
        let fused_output = ops::hip::try_kda_fused_resident_device_f32(
            rocm.device_id,
            &fused_storage.conv,
            &fused_storage.recurrent,
            KdaInputs {
                query: fused_query.device.as_deref().expect("ROCm KDA fused query 缺少 device buffer"),
                key: fused_key.device.as_deref().expect("ROCm KDA fused key 缺少 device buffer"),
                value: fused_value.device.as_deref().expect("ROCm KDA fused value 缺少 device buffer"),
                decay: fused_decay.device.as_deref().expect("ROCm KDA fused decay 缺少 device buffer"),
                beta: fused_beta.device.as_deref().expect("ROCm KDA fused beta 缺少 device buffer"),
                output_gate: fused_gate.device.as_deref().expect("ROCm KDA fused gate 缺少 device buffer"),
            },
            KdaWeightsRef {
                query_conv: ensure_weight_device(rocm.device_id, &rocm_conv_weight).expect("ROCm KDA fused conv weight 上传失败").as_ref(),
                key_conv: ensure_weight_device(rocm.device_id, &rocm_conv_weight).expect("ROCm KDA fused conv weight 上传失败").as_ref(),
                value_conv: ensure_weight_device(rocm.device_id, &rocm_conv_weight).expect("ROCm KDA fused conv weight 上传失败").as_ref(),
                a_log: ensure_weight_device(rocm.device_id, &rocm_a_log).expect("ROCm KDA fused a_log 上传失败").as_ref(),
                dt_bias: ensure_weight_device(rocm.device_id, &rocm_dt_bias).expect("ROCm KDA fused dt_bias 上传失败").as_ref(),
                output_norm: ensure_weight_device(rocm.device_id, &rocm_norm_weight).expect("ROCm KDA fused norm weight 上传失败").as_ref(),
            },
            rows,
            &spec,
        )
        .expect("ROCm KDA fused kernel 执行失败");
        let fused_output_tensor = device_tensor_f32(fused_output, rows, projection);
        let fused_output_data = rocm.tensor_to_f32(&fused_output_tensor).expect("下载 ROCm KDA fused 输出失败");
        for (i, (cpu_val, fused_val)) in cpu_output.data.iter().zip(fused_output_data.iter()).enumerate() {
            let diff = (cpu_val - fused_val).abs();
            assert!(diff <= tolerance + tolerance * cpu_val.abs(), "KDA fused 位置 {i}: CPU={cpu_val:.6} HIP={fused_val:.6} diff={diff:.6}");
        }

        let mut max_diff = 0.0f32;
        for (i, (cpu_val, rocm_val)) in cpu_output.data.iter().zip(rocm_output_data.iter()).enumerate() {
            let diff = (cpu_val - rocm_val).abs();
            let limit = tolerance + tolerance * cpu_val.abs();
            if diff > limit {
                panic!("KDA 输出位置 {i}: CPU={cpu_val:.6} HIP={rocm_val:.6} diff={diff:.6} 超过阈值 {limit:.6}");
            }
            max_diff = max_diff.max(diff);
        }
        eprintln!("KDA CPU vs HIP 最大差异: {max_diff:.6} (阈值 {tolerance})");
    }
}
