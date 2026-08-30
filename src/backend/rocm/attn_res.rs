//! AttnRes HIP kernel capability。

use super::*;

impl AttnResBackend for RocmContext {
    fn attn_res_mix(&self, current: &Self::Tensor, block_residuals: &[Self::Tensor], norm_weight: &Self::Weight, projection_weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        if current.rows == 0 || current.cols == 0 || (norm_weight.rows(), norm_weight.cols()) != (1, current.cols) || (projection_weight.rows(), projection_weight.cols()) != (1, current.cols) {
            return Err(compute_error("Rocm AttnRes current/weight shape 非法"));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(compute_error(format!("Rocm AttnRes eps 非法: {eps}")));
        }
        if block_residuals.iter().any(|tensor| tensor.rows != current.rows || tensor.cols != current.cols) {
            return Err(compute_error("Rocm AttnRes block residual shape 不一致"));
        }
        let candidate_count = block_residuals.len() + 1;
        if candidate_count > 32 {
            return Err(compute_error(format!("Rocm AttnRes candidate 数量 {candidate_count} 超过 kernel 上限 32")));
        }

        let device_id = self.device_id;
        let rows = current.rows;
        let cols = current.cols;

        // 输入可能来自上一张卡或为 Bf16;统一迁到本 device 并转成 F32。
        let current = self.tensor_as_f32(current.clone())?;
        let residuals = block_residuals.iter().map(|tensor| self.tensor_as_f32(tensor.clone())).collect::<Result<Vec<_>, _>>()?;
        let mut candidates = Vec::with_capacity(candidate_count);
        for (index, tensor) in residuals.iter().chain(std::iter::once(&current)).enumerate() {
            let device = tensor.device.as_deref().ok_or_else(|| compute_error(format!("ROCm AttnRes candidate {index} 缺少 device buffer")))?;
            candidates.push(device);
        }

        let norm_weight_device = ensure_weight_device(device_id, norm_weight)?;
        let projection_weight_device = ensure_weight_device(device_id, projection_weight)?;

        let output = ops::hip::try_attn_res_mix_resident_device_f32(device_id, &candidates, norm_weight_device.as_ref(), projection_weight_device.as_ref(), rows, cols, eps).map_err(compute_error)?;

        Ok(device_tensor_f32(output, rows, cols))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attention::attn_res::AttnResBackend,
        backend::{BackendResources, LinearWeight},
        kernel::cpu::{CpuTensor, attn_res as cpu_attn_res},
    };

    /// CPU vs HIP 一致性:F32 对 F32,容差 1e-4;ROCm 不可用时跳过。
    #[test]
    fn rocm_attn_res_matches_cpu() {
        let current_values = [3.0, 5.0, 2.0, 4.0];
        let residual_values = [1.0, 3.0, 4.0, 0.0];
        let norm_values = [1.0, 0.5];
        let projection_values = [0.25, -0.75];
        let eps = 1.0e-5;
        let expected =
            cpu_attn_res::mix(&CpuTensor { data: current_values.to_vec(), rows: 2, cols: 2 }, &[CpuTensor { data: residual_values.to_vec(), rows: 2, cols: 2 }], &norm_values, &projection_values, eps).expect("CPU AttnRes 执行失败");

        let rocm = match RocmContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => {
                eprintln!("跳过 ROCm AttnRes 测试:ROCm runtime 不可用");
                return;
            }
        };
        let rocm_tensor = |data: &[f32]| RocmTensor { data: data.to_vec(), rows: 2, cols: 2, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: None };
        let current = rocm_tensor(&current_values);
        let residual = rocm_tensor(&residual_values);
        let norm_weight = rocm.prepare_weight(LinearWeight::F32(&norm_values), 1, 2).expect("ROCm norm_weight prepare 失败");
        let projection_weight = rocm.prepare_weight(LinearWeight::F32(&projection_values), 1, 2).expect("ROCm projection_weight prepare 失败");
        let actual = rocm.attn_res_mix(&current, &[residual], &norm_weight, &projection_weight, eps).expect("ROCm AttnRes 执行失败");
        let actual_data = rocm.tensor_to_f32(&actual).expect("下载 ROCm AttnRes 输出失败");

        let tolerance = 1e-4;
        let mut max_diff = 0.0f32;
        for (i, (expected, actual)) in expected.data.iter().zip(actual_data.iter()).enumerate() {
            let diff = (expected - actual).abs();
            if diff > tolerance {
                panic!("AttnRes 输出位置 {i}: CPU={expected:.6} HIP={actual:.6} diff={diff:.6} 超过阈值 {tolerance}");
            }
            max_diff = max_diff.max(diff);
        }
        eprintln!("AttnRes CPU vs HIP 最大差异: {max_diff:.6} (阈值 {tolerance})");
    }
}
