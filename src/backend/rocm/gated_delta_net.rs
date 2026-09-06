//! GatedDeltaNet HIP kernel 实现。
//!
//! conv_state 与 recurrent_state 作为持久 device buffer 跨 token 保持。
//! fused kernel 单 block 按 token 顺序执行递归,在 value_head × value_head_dim 维度并行。

use super::*;
use crate::attention::gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetInputs, GatedDeltaNetWeightsRef};

/// 设备 resident GatedDeltaNet 状态:conv shift register + recurrent delta matrix。
pub struct RocmGatedDeltaNetStorage {
    pub conv: ops::hip::DeviceBuffer,
    pub recurrent: ops::hip::DeviceBuffer,
}

impl crate::attention::gated_delta_net::GatedDeltaNetStorage for RocmGatedDeltaNetStorage {
    fn allocated_bytes(&self) -> usize {
        self.conv.bytes() + self.recurrent.bytes()
    }
}

impl GatedDeltaNetKernel for RocmContext {
    type GatedDeltaNetStorage = RocmGatedDeltaNetStorage;

    fn allocate_gated_delta_net_storage(&self, spec: &crate::attention::gated_delta_net::GatedDeltaNetSpec) -> Result<Self::GatedDeltaNetStorage, BackendError> {
        let device_id = self.device_id;
        let conv_zeros = vec![0.0f32; spec.conv_state_elements()];
        let recurrent_zeros = vec![0.0f32; spec.recurrent_elements()];
        let conv_bytes = unsafe { std::slice::from_raw_parts(conv_zeros.as_ptr().cast(), conv_zeros.len() * 4) };
        let recurrent_bytes = unsafe { std::slice::from_raw_parts(recurrent_zeros.as_ptr().cast(), recurrent_zeros.len() * 4) };
        let conv = ops::hip::DeviceBuffer::upload(device_id, conv_bytes).map_err(compute_error)?;
        let recurrent = ops::hip::DeviceBuffer::upload(device_id, recurrent_bytes).map_err(compute_error)?;
        Ok(RocmGatedDeltaNetStorage { conv, recurrent })
    }

    fn gated_delta_net_fused(
        &self,
        storage: &mut Self::GatedDeltaNetStorage,
        inputs: GatedDeltaNetInputs<'_, Self::Tensor>,
        weights: GatedDeltaNetWeightsRef<'_, Self::Weight>,
        spec: &crate::attention::gated_delta_net::GatedDeltaNetSpec,
    ) -> Result<Self::Tensor, BackendError> {
        self.gated_delta_net_fused_layout(storage, inputs, weights, GatedDeltaNetHeadLayout::Tiled, spec)
    }

    fn gated_delta_net_fused_layout(
        &self,
        storage: &mut Self::GatedDeltaNetStorage,
        inputs: GatedDeltaNetInputs<'_, Self::Tensor>,
        weights: GatedDeltaNetWeightsRef<'_, Self::Weight>,
        head_layout: GatedDeltaNetHeadLayout,
        spec: &crate::attention::gated_delta_net::GatedDeltaNetSpec,
    ) -> Result<Self::Tensor, BackendError> {
        let GatedDeltaNetInputs { qkv, z, alpha, beta } = inputs;
        let GatedDeltaNetWeightsRef { conv: conv_weight, a_log, dt_bias, norm: norm_weight } = weights;
        if (conv_weight.rows(), conv_weight.cols()) != (spec.conv_dim(), spec.conv_kernel) || a_log.data().len() != spec.value_heads || dt_bias.data().len() != spec.value_heads || norm_weight.data().len() != spec.value_head_dim {
            return Err(compute_error("Rocm Gated DeltaNet weight shape 与 spec 不一致"));
        }

        let device_id = self.device_id;
        let rows = qkv.rows;
        let value_dim = spec.value_dim();

        // 输入可能来自上一张卡或 host；统一通过 context 校正 device 所有权。
        let qkv = self.tensor_on_device(qkv.clone())?;
        let z = self.tensor_on_device(z.clone())?;
        let alpha = self.tensor_on_device(alpha.clone())?;
        let beta = self.tensor_on_device(beta.clone())?;
        let qkv_device = qkv.device.as_deref().ok_or_else(|| compute_error("ROCm Gated DeltaNet qkv 缺少 device buffer"))?;
        let z_device = z.device.as_deref().ok_or_else(|| compute_error("ROCm Gated DeltaNet z 缺少 device buffer"))?;
        let alpha_device = alpha.device.as_deref().ok_or_else(|| compute_error("ROCm Gated DeltaNet alpha 缺少 device buffer"))?;
        let beta_device = beta.device.as_deref().ok_or_else(|| compute_error("ROCm Gated DeltaNet beta 缺少 device buffer"))?;

        let conv_weight_device = ensure_weight_device(device_id, conv_weight)?;
        let a_log_device = ensure_weight_device(device_id, a_log)?;
        let dt_bias_device = ensure_weight_device(device_id, dt_bias)?;
        let norm_weight_device = ensure_weight_device(device_id, norm_weight)?;

        let output = ops::hip::try_gated_delta_net_fused_resident_device_f32(
            device_id,
            &storage.conv,
            &storage.recurrent,
            GatedDeltaNetInputs { qkv: qkv_device, z: z_device, alpha: alpha_device, beta: beta_device },
            GatedDeltaNetWeightsRef { conv: conv_weight_device.as_ref(), a_log: a_log_device.as_ref(), dt_bias: dt_bias_device.as_ref(), norm: norm_weight_device.as_ref() },
            rows,
            head_layout,
            spec,
        )
        .map_err(compute_error)?;

        Ok(device_tensor_f32(output, rows, value_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attention::gated_delta_net::{GatedDeltaNetInputs, GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetWeightsRef, GdnOutputGate},
        backend::cpu::CpuContext,
        backend::{BackendResources, LinearWeight},
        kernel::cpu::CpuTensor,
    };

    /// CPU vs HIP 一致性:小规模 GatedDeltaNet spec,确定性 dummy 数据,atol=rtol=1e-2。
    #[test]
    fn rocm_gated_delta_net_matches_cpu() {
        // 使用小 spec 加速测试:2 key_heads, 4 value_heads, 16 dim。
        let spec = GatedDeltaNetSpec { key_heads: 2, value_heads: 4, key_head_dim: 16, value_head_dim: 16, conv_kernel: 4, rms_eps: 1e-6, output_gate: GdnOutputGate::Silu };
        let rows = 3;
        let value_dim = spec.value_dim();
        let conv_dim = spec.conv_dim();

        // 确定性 dummy 数据:用简单 pattern 避免全零退化。
        let qkv_data: Vec<f32> = (0..rows * conv_dim).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
        let z_data: Vec<f32> = (0..rows * value_dim).map(|i| ((i as f32) * 0.02).cos() * 0.3 + 0.1).collect();
        let alpha_data: Vec<f32> = (0..rows * spec.value_heads).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let beta_data: Vec<f32> = (0..rows * spec.value_heads).map(|i| (i as f32) * 0.01).collect();
        let conv_weight_data: Vec<f32> = (0..conv_dim * spec.conv_kernel).map(|i| ((i as f32) * 0.1).sin() * 0.1).collect();
        let a_log_data: Vec<f32> = (0..spec.value_heads).map(|i| (i as f32) * 0.1).collect();
        let dt_bias_data: Vec<f32> = (0..spec.value_heads).map(|i| (i as f32) * 0.05).collect();
        let norm_weight_data: Vec<f32> = (0..spec.value_head_dim).map(|i| 1.0 + (i as f32) * 0.01).collect();

        // CPU 参考实现
        let cpu = CpuContext;
        let mut cpu_storage = cpu.allocate_gated_delta_net_storage(&spec).expect("CPU GDN storage 分配失败");
        let cpu_qkv = CpuTensor { data: qkv_data.clone(), rows, cols: conv_dim };
        let cpu_z = CpuTensor { data: z_data.clone(), rows, cols: value_dim };
        let cpu_alpha = CpuTensor { data: alpha_data.clone(), rows, cols: spec.value_heads };
        let cpu_beta = CpuTensor { data: beta_data.clone(), rows, cols: spec.value_heads };
        let cpu_conv_weight = cpu.prepare_weight(LinearWeight::F32(&conv_weight_data), conv_dim, spec.conv_kernel).expect("CPU conv_weight prepare 失败");
        let cpu_a_log = cpu.prepare_f32(&a_log_data, 1, spec.value_heads).expect("CPU a_log prepare 失败");
        let cpu_dt_bias = cpu.prepare_f32(&dt_bias_data, 1, spec.value_heads).expect("CPU dt_bias prepare 失败");
        let cpu_norm_weight = cpu.prepare_f32(&norm_weight_data, 1, spec.value_head_dim).expect("CPU norm_weight prepare 失败");
        let cpu_output = cpu
            .gated_delta_net_fused(
                &mut cpu_storage,
                GatedDeltaNetInputs { qkv: &cpu_qkv, z: &cpu_z, alpha: &cpu_alpha, beta: &cpu_beta },
                GatedDeltaNetWeightsRef { conv: &cpu_conv_weight, a_log: &cpu_a_log, dt_bias: &cpu_dt_bias, norm: &cpu_norm_weight },
                &spec,
            )
            .expect("CPU GDN 执行失败");

        // HIP 实现
        let rocm = match RocmContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => {
                eprintln!("跳过 ROCm GDN 测试:ROCm runtime 不可用");
                return;
            }
        };
        let mut rocm_storage = rocm.allocate_gated_delta_net_storage(&spec).expect("ROCm GDN storage 分配失败");
        let rocm_qkv = RocmTensor { data: qkv_data.clone(), rows, cols: conv_dim, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: None, replica: None };
        let rocm_z = RocmTensor { data: z_data.clone(), rows, cols: value_dim, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: None, replica: None };
        let rocm_alpha = RocmTensor { data: alpha_data.clone(), rows, cols: spec.value_heads, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: None, replica: None };
        let rocm_beta = RocmTensor { data: beta_data.clone(), rows, cols: spec.value_heads, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: None, replica: None };
        let rocm_conv_weight = rocm.prepare_weight(LinearWeight::F32(&conv_weight_data), conv_dim, spec.conv_kernel).expect("ROCm conv_weight prepare 失败");
        let rocm_a_log = rocm.prepare_f32(&a_log_data, 1, spec.value_heads).expect("ROCm a_log prepare 失败");
        let rocm_dt_bias = rocm.prepare_f32(&dt_bias_data, 1, spec.value_heads).expect("ROCm dt_bias prepare 失败");
        let rocm_norm_weight = rocm.prepare_f32(&norm_weight_data, 1, spec.value_head_dim).expect("ROCm norm_weight prepare 失败");
        let rocm_output = rocm
            .gated_delta_net_fused(
                &mut rocm_storage,
                GatedDeltaNetInputs { qkv: &rocm_qkv, z: &rocm_z, alpha: &rocm_alpha, beta: &rocm_beta },
                GatedDeltaNetWeightsRef { conv: &rocm_conv_weight, a_log: &rocm_a_log, dt_bias: &rocm_dt_bias, norm: &rocm_norm_weight },
                &spec,
            )
            .expect("ROCm GDN 执行失败");
        let rocm_output_data = rocm.tensor_to_f32(&rocm_output).expect("下载 ROCm GDN 输出失败");

        assert_eq!(rocm_output_data.len(), cpu_output.data.len(), "ROCm GDN 输出元素数不一致");

        // 比较:BF16 精度阈值 atol=rtol=1e-2
        let tolerance = 1e-2;
        let mut max_diff = 0.0f32;
        for (i, (cpu_val, rocm_val)) in cpu_output.data.iter().zip(rocm_output_data.iter()).enumerate() {
            let diff = (cpu_val - rocm_val).abs();
            let limit = tolerance + tolerance * cpu_val.abs();
            if diff > limit {
                panic!("GDN 输出位置 {i}: CPU={cpu_val:.6} HIP={rocm_val:.6} diff={diff:.6} 超过阈值 {limit:.6}");
            }
            max_diff = max_diff.max(diff);
        }
        eprintln!("GDN CPU vs HIP 最大差异: {max_diff:.6} (阈值 {tolerance})");
    }
}
