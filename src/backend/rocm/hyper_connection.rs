//! mHC ROCm capability;张量变换和 Sinkhorn 全程留在设备端 F32 buffer。

use crate::{
    attention::hyper_connection::{HyperConnectionKernel, HyperConnectionPrepared, HyperConnectionSpec, HyperConnectionSplit},
    backend::{BackendError, StageTensorBackend, compute_error as compute},
};

use super::{RocmContext, RocmTensor, RocmWeight, constant, device_tensor_f32, f32_tensor};
use crate::kernel::rocm::hip as mhc;

impl RocmContext {
    /// DSpark capture 使用各 mHC copy 的等权均值；系数保持 device resident，
    /// 复用通用 reduce kernel，不为模型增加专用算子。
    pub fn hyper_connection_mean(&self, hidden: &RocmTensor, copies: usize) -> Result<RocmTensor, BackendError> {
        if hidden.rows == 0 || hidden.cols == 0 || copies == 0 || !hidden.cols.is_multiple_of(copies) {
            return Err(compute(format!("ROCm mHC mean shape 非法: hidden=[{},{}] copies={copies}", hidden.rows, hidden.cols)));
        }
        let width = hidden.cols / copies;
        let hidden_device = f32_tensor(self, hidden)?.device;
        let hidden_buffer = hidden_device.as_deref().ok_or_else(|| compute("ROCm mHC mean 输入缺少 device buffer"))?;
        let output = mhc::try_mhc_mean_f32(self.device_id, hidden_buffer, hidden.rows, width, copies).map_err(compute)?;
        Ok(device_tensor_f32(output, hidden.rows, width))
    }
}

impl HyperConnectionKernel for RocmContext {
    /// copies 无权重平均(glm5_next 收尾):全 1 系数复用 reduce kernel。
    fn hyper_connection_collapse(&self, hidden: &RocmTensor, copies: usize) -> Result<RocmTensor, BackendError> {
        if copies == 0 || hidden.cols % copies != 0 {
            return Err(compute(format!("ROCm mHC collapse shape 非法: hidden=[{},{}] copies={copies}", hidden.rows, hidden.cols)));
        }
        // reduce 的系数是 per-row 的 [rows, copies];collapse 全 1 即每行同系数。
        let coefficients = self.stage_tensor_from_f32(vec![1.0_f32; hidden.rows * copies], hidden.rows, copies)?;
        let summed = self.hyper_connection_reduce(hidden, &coefficients, copies)?;
        // 折叠语义是平均;对后续 RMSNorm 等价,直接返回和避免一次逐元素乘。
        Ok(summed)
    }

    fn hyper_connection_expand(&self, hidden: &RocmTensor, copies: usize) -> Result<RocmTensor, BackendError> {
        if hidden.rows == 0 || hidden.cols == 0 || copies == 0 {
            return Err(compute(format!("ROCm mHC expand shape 非法: hidden=[{},{}] copies={copies}", hidden.rows, hidden.cols)));
        }
        let input = f32_tensor(self, hidden)?;
        let device = input.device.as_deref().ok_or_else(|| compute("ROCm mHC expand 输入缺少 device buffer"))?;
        let output = mhc::try_mhc_expand_f32(self.device_id, device, input.rows, input.cols, copies).map_err(compute)?;
        Ok(device_tensor_f32(output, input.rows, input.cols * copies))
    }

    fn hyper_connection_reduce(&self, hidden: &RocmTensor, coefficients: &RocmTensor, copies: usize) -> Result<RocmTensor, BackendError> {
        if hidden.rows == 0 || hidden.rows != coefficients.rows || copies == 0 || !hidden.cols.is_multiple_of(copies) || coefficients.cols != copies {
            return Err(compute(format!("ROCm mHC reduce shape 不一致: hidden=[{},{}] coefficients=[{},{}] copies={copies}", hidden.rows, hidden.cols, coefficients.rows, coefficients.cols)));
        }
        let width = hidden.cols / copies;
        let hidden_device = f32_tensor(self, hidden)?.device;
        let coefficient_device = f32_tensor(self, coefficients)?.device;
        let hidden_buffer = hidden_device.as_deref().ok_or_else(|| compute("ROCm mHC reduce 输入缺少 device buffer"))?;
        let coefficient_buffer = coefficient_device.as_deref().ok_or_else(|| compute("ROCm mHC reduce 系数缺少 device buffer"))?;
        let output = mhc::try_mhc_reduce_f32(self.device_id, hidden_buffer, coefficient_buffer, hidden.rows, width, copies).map_err(compute)?;
        Ok(device_tensor_f32(output, hidden.rows, width))
    }

    fn hyper_connection_mix(&self, hidden: &RocmTensor, matrix: &RocmTensor, spec: &HyperConnectionSpec) -> Result<RocmTensor, BackendError> {
        spec.validate().map_err(compute)?;
        let copies = spec.copies;
        if hidden.rows == 0 || hidden.rows != matrix.rows || !hidden.cols.is_multiple_of(copies) || matrix.cols != copies * copies {
            return Err(compute(format!("ROCm mHC mix shape 不一致: hidden=[{},{}] matrix=[{},{}] copies={copies}", hidden.rows, hidden.cols, matrix.rows, matrix.cols)));
        }
        let width = hidden.cols / copies;
        let hidden_device = f32_tensor(self, hidden)?.device;
        let matrix_device = f32_tensor(self, matrix)?.device;
        let hidden_buffer = hidden_device.as_deref().ok_or_else(|| compute("ROCm mHC mix 输入缺少 device buffer"))?;
        let matrix_buffer = matrix_device.as_deref().ok_or_else(|| compute("ROCm mHC mix 矩阵缺少 device buffer"))?;
        let output = mhc::try_mhc_mix_f32(self.device_id, hidden_buffer, matrix_buffer, hidden.rows, width, copies).map_err(compute)?;
        Ok(device_tensor_f32(output, hidden.rows, hidden.cols))
    }

    fn hyper_connection_expand_scaled(&self, hidden: &RocmTensor, coefficients: &RocmTensor, copies: usize) -> Result<RocmTensor, BackendError> {
        if hidden.rows == 0 || hidden.cols == 0 || hidden.rows != coefficients.rows || copies == 0 || coefficients.cols != copies {
            return Err(compute(format!("ROCm mHC expand_scaled shape 不一致: hidden=[{},{}] coefficients=[{},{}] copies={copies}", hidden.rows, hidden.cols, coefficients.rows, coefficients.cols)));
        }
        let hidden_device = f32_tensor(self, hidden)?.device;
        let coefficient_device = f32_tensor(self, coefficients)?.device;
        let hidden_buffer = hidden_device.as_deref().ok_or_else(|| compute("ROCm mHC expand_scaled 输入缺少 device buffer"))?;
        let coefficient_buffer = coefficient_device.as_deref().ok_or_else(|| compute("ROCm mHC expand_scaled 系数缺少 device buffer"))?;
        let output = mhc::try_mhc_expand_scaled_f32(self.device_id, hidden_buffer, coefficient_buffer, hidden.rows, hidden.cols, copies).map_err(compute)?;
        Ok(device_tensor_f32(output, hidden.rows, hidden.cols * copies))
    }

    fn hyper_connection_expand_scaled_add(&self, hidden: &RocmTensor, coefficients: &RocmTensor, residual: &RocmTensor, copies: usize) -> Result<RocmTensor, BackendError> {
        if hidden.rows == 0
            || hidden.cols == 0
            || hidden.rows != coefficients.rows
            || hidden.rows != residual.rows
            || copies == 0
            || coefficients.cols != copies
            || residual.cols != hidden.cols.checked_mul(copies).ok_or_else(|| compute("ROCm mHC fused expand_scaled_add 列数溢出"))?
        {
            return Err(compute(format!(
                "ROCm mHC fused expand_scaled_add shape 不一致: hidden=[{},{}] coefficients=[{},{}] residual=[{},{}] copies={copies}",
                hidden.rows, hidden.cols, coefficients.rows, coefficients.cols, residual.rows, residual.cols
            )));
        }
        let hidden_device = f32_tensor(self, hidden)?.device;
        let coefficient_device = f32_tensor(self, coefficients)?.device;
        let residual_device = f32_tensor(self, residual)?.device;
        let hidden_buffer = hidden_device.as_deref().ok_or_else(|| compute("ROCm mHC fused expand_scaled_add 输入缺少 device buffer"))?;
        let coefficient_buffer = coefficient_device.as_deref().ok_or_else(|| compute("ROCm mHC fused expand_scaled_add 系数缺少 device buffer"))?;
        let residual_buffer = residual_device.as_deref().ok_or_else(|| compute("ROCm mHC fused expand_scaled_add residual 缺少 device buffer"))?;
        let output = mhc::try_mhc_expand_scaled_add_f32(self.device_id, hidden_buffer, coefficient_buffer, residual_buffer, hidden.rows, hidden.cols, copies).map_err(compute)?;
        Ok(device_tensor_f32(output, hidden.rows, residual.cols))
    }

    fn hyper_connection_split(&self, mixes: &RocmTensor, base: &RocmWeight, scale: &RocmWeight, spec: &HyperConnectionSpec) -> Result<HyperConnectionSplit<RocmTensor>, BackendError> {
        spec.validate().map_err(compute)?;
        let copies = spec.copies;
        let mix_columns = copies.checked_mul(copies.checked_add(2).ok_or_else(|| compute("ROCm mHC copies 溢出"))?).ok_or_else(|| compute("ROCm mHC mix columns 溢出"))?;
        if mixes.rows == 0 || mixes.cols != mix_columns {
            return Err(compute(format!("ROCm mHC split shape 非法: mixes=[{},{}] 期望列 {mix_columns}", mixes.rows, mixes.cols)));
        }
        let base = constant(base, mix_columns, "mHC base")?;
        let scale = constant(scale, 3, "mHC scale")?;
        let input = f32_tensor(self, mixes)?;
        let device = input.device.as_deref().ok_or_else(|| compute("ROCm mHC split 输入缺少 device buffer"))?;
        let (pre, post, combination) = mhc::try_mhc_split_f32(self.device_id, device, base, scale, mixes.rows, copies, spec.sinkhorn_iterations, spec.eps).map_err(compute)?;
        Ok(HyperConnectionSplit { pre: device_tensor_f32(pre, mixes.rows, copies), post: device_tensor_f32(post, mixes.rows, copies), combination: device_tensor_f32(combination, mixes.rows, copies * copies) })
    }

    fn hyper_connection_prepare_sublayer(&self, hidden: &RocmTensor, mixes: &RocmTensor, base: &RocmWeight, scale: &RocmWeight, spec: &HyperConnectionSpec) -> Result<HyperConnectionPrepared<RocmTensor>, BackendError> {
        spec.validate().map_err(compute)?;
        let copies = spec.copies;
        if hidden.rows == 0 || hidden.rows != mixes.rows || !hidden.cols.is_multiple_of(copies) || mixes.cols != copies * (copies + 2) || copies > 16 {
            let split = self.hyper_connection_split(mixes, base, scale, spec)?;
            let residual = self.hyper_connection_mix(hidden, &split.combination, spec)?;
            let reduced = self.hyper_connection_reduce(hidden, &split.pre, copies)?;
            return Ok(HyperConnectionPrepared { residual, reduced, post: split.post });
        }
        let width = hidden.cols / copies;
        let hidden = f32_tensor(self, hidden)?;
        let mixes = f32_tensor(self, mixes)?;
        let hidden_device = hidden.device.as_deref().ok_or_else(|| compute("ROCm mHC prepare hidden 缺少 device buffer"))?;
        let mixes_device = mixes.device.as_deref().ok_or_else(|| compute("ROCm mHC prepare mixes 缺少 device buffer"))?;
        let base = constant(base, mixes.cols, "mHC base")?;
        let scale = constant(scale, 3, "mHC scale")?;
        let (residual, reduced, post) = mhc::try_mhc_prepare_sublayer_f32(self.device_id, hidden_device, mixes_device, base, scale, hidden.rows, width, copies, spec.sinkhorn_iterations, spec.eps).map_err(compute)?;
        Ok(HyperConnectionPrepared { residual: device_tensor_f32(residual, hidden.rows, hidden.cols), reduced: device_tensor_f32(reduced, hidden.rows, width), post: device_tensor_f32(post, hidden.rows, copies) })
    }

    fn hyper_connection_head_reduce(&self, hidden: &RocmTensor, mixes: &RocmTensor, base: &RocmWeight, scale: &RocmWeight, copies: usize, eps: f32) -> Result<RocmTensor, BackendError> {
        if hidden.rows == 0 || hidden.rows != mixes.rows || copies == 0 || !hidden.cols.is_multiple_of(copies) || mixes.cols != copies || !eps.is_finite() || eps <= 0.0 {
            return Err(compute(format!("ROCm output mHC shape 非法: hidden=[{},{}] mixes=[{},{}] copies={copies} eps={eps}", hidden.rows, hidden.cols, mixes.rows, mixes.cols)));
        }
        let width = hidden.cols / copies;
        let base = constant(base, copies, "output mHC base")?;
        let scale = constant(scale, 1, "output mHC scale")?;
        let hidden_device = f32_tensor(self, hidden)?.device;
        let mixes_device = f32_tensor(self, mixes)?.device;
        let hidden_buffer = hidden_device.as_deref().ok_or_else(|| compute("ROCm output mHC 输入缺少 device buffer"))?;
        let mixes_buffer = mixes_device.as_deref().ok_or_else(|| compute("ROCm output mHC mixes 缺少 device buffer"))?;
        let output = mhc::try_mhc_head_reduce_f32(self.device_id, hidden_buffer, mixes_buffer, base, scale, hidden.rows, width, copies, eps).map_err(compute)?;
        Ok(device_tensor_f32(output, hidden.rows, width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::hyper_connection::{expand_scaled_f32, head_reduce_f32, mix_f32, reduce_f32, split_f32};

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 2.0e-4, "actual={actual}, expected={expected}");
        }
    }

    fn upload(device_id: i32, values: &[f32]) -> crate::kernel::rocm::hip::DeviceBuffer {
        crate::kernel::rocm::hip::DeviceBuffer::upload_f32(device_id, values).unwrap()
    }

    fn download(buffer: &crate::kernel::rocm::hip::DeviceBuffer, elements: usize) -> Vec<f32> {
        buffer.download_f32(elements).expect("mHC 测试回读")
    }

    /// 与 Metal 版同款哑数据校验:GPU 结果必须逐元素贴合 CPU reference。
    ///
    /// 上传一律走 `upload_f32`:字面量数组直接 `from_raw_parts` 传 FFI 时,
    /// LLVM 可复用其未读取的栈槽,导致 GPU 收到被覆盖的栈内容(dev 构建
    /// 实测必现),这里同时是对该封装的回归验证。
    #[test]
    fn rocm_mhc_matches_cpu_reference() {
        use crate::backend::BackendResources;
        let Ok(context) = RocmContext::new(0) else { return };
        let device_id = context.device_id;
        let hidden_values = [1.0, -2.0, 3.0, 4.0, 0.5, 1.5, -0.5, 2.5];
        let hidden = upload(device_id, &hidden_values);
        let expanded = mhc::try_mhc_expand_f32(device_id, &hidden, 2, 4, 2).unwrap();
        assert_close(&download(&expanded, 16), &[1.0, -2.0, 3.0, 4.0, 1.0, -2.0, 3.0, 4.0, 0.5, 1.5, -0.5, 2.5, 0.5, 1.5, -0.5, 2.5]);

        let coefficient_values = [0.25, 0.75, 1.25, -0.25];
        let coefficients = upload(device_id, &coefficient_values);
        let reduced = mhc::try_mhc_reduce_f32(device_id, &expanded, &coefficients, 2, 4, 2).unwrap();
        let expected = [reduce_f32(&[1.0, -2.0, 3.0, 4.0, 1.0, -2.0, 3.0, 4.0], &[0.25, 0.75], 2).unwrap(), reduce_f32(&[0.5, 1.5, -0.5, 2.5, 0.5, 1.5, -0.5, 2.5], &[1.25, -0.25], 2).unwrap()].concat();
        assert_close(&download(&reduced, 8), &expected);

        let scaled = mhc::try_mhc_expand_scaled_f32(device_id, &hidden, &coefficients, 2, 4, 2).unwrap();
        let expected = [expand_scaled_f32(&hidden_values[..4], &[0.25, 0.75], 2).unwrap(), expand_scaled_f32(&hidden_values[4..], &[1.25, -0.25], 2).unwrap()].concat();
        assert_close(&download(&scaled, 16), &expected);
        let fused = mhc::try_mhc_expand_scaled_add_f32(device_id, &hidden, &coefficients, &expanded, 2, 4, 2).unwrap();
        let expected_fused = download(&scaled, 16).into_iter().zip(download(&expanded, 16)).map(|(scaled, residual)| scaled + residual).collect::<Vec<_>>();
        assert_eq!(download(&fused, 16), expected_fused);

        let matrix_values = [0.8, 0.2, 0.3, 0.7, 0.6, 0.4, 0.1, 0.9];
        let matrix = upload(device_id, &matrix_values);
        let mixed = mhc::try_mhc_mix_f32(device_id, &expanded, &matrix, 2, 4, 2).unwrap();
        let expanded_host = download(&expanded, 16);
        let expected = [mix_f32(&expanded_host[..8], &matrix_values[..4], 2).unwrap(), mix_f32(&expanded_host[8..], &matrix_values[4..], 2).unwrap()].concat();
        assert_close(&download(&mixed, 16), &expected);

        let spec = HyperConnectionSpec { copies: 2, sinkhorn_iterations: 20, eps: 1.0e-6 };
        let mixes_values = [0.2, -0.3, 0.5, -0.7, 0.1, 0.4, -0.2, 0.6, -0.4, 0.1, 0.7, 0.2, -0.5, 0.3, 0.8, -0.1];
        let base_values = [0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.4, -0.3];
        let scale_values = [0.75f32, 1.25, 0.5];
        let mixes = upload(device_id, &mixes_values);
        let base = upload(device_id, &base_values);
        let scale = upload(device_id, &scale_values);
        let (pre, post, combination) = mhc::try_mhc_split_f32(device_id, &mixes, &base, &scale, 2, 2, 20, 1.0e-6).unwrap();
        let expected = split_f32(&mixes_values, &base_values, &scale_values, &spec).unwrap();
        context.synchronize().unwrap();
        assert_close(&download(&pre, 4), &expected.pre);
        assert_close(&download(&post, 4), &expected.post);
        assert_close(&download(&combination, 8), &expected.combination);

        let (residual, reduced, post) = mhc::try_mhc_prepare_sublayer_f32(device_id, &expanded, &mixes, &base, &scale, 2, 4, 2, 20, 1.0e-6).unwrap();
        let expected_residual = mix_f32(&expanded_host[..8], &expected.combination[..4], 2).unwrap();
        let expected_residual = [expected_residual, mix_f32(&expanded_host[8..], &expected.combination[4..], 2).unwrap()].concat();
        let expected_reduced = [reduce_f32(&expanded_host[..8], &expected.pre[..2], 2).unwrap(), reduce_f32(&expanded_host[8..], &expected.pre[2..], 2).unwrap()].concat();
        assert_close(&download(&residual, 16), &expected_residual);
        assert_close(&download(&reduced, 8), &expected_reduced);
        assert_close(&download(&post, 4), &expected.post);

        // V4 实际配置是 copies=4；覆盖并行 Sinkhorn 的完整行列协作路径。
        let spec4 = HyperConnectionSpec { copies: 4, sinkhorn_iterations: 20, eps: 1.0e-6 };
        let hidden4_values = (0..32).map(|index| index as f32 * 0.07 - 0.9).collect::<Vec<_>>();
        let mixes4_values = (0..24).map(|index| index as f32 * 0.031 - 0.4).collect::<Vec<_>>();
        let base4_values = (0..24).map(|index| index as f32 * -0.017 + 0.2).collect::<Vec<_>>();
        let hidden4 = upload(device_id, &hidden4_values);
        let mixes4 = upload(device_id, &mixes4_values);
        let base4 = upload(device_id, &base4_values);
        let expected4 = split_f32(&mixes4_values, &base4_values, &scale_values, &spec4).unwrap();
        let (residual4, reduced4, post4) = mhc::try_mhc_prepare_sublayer_f32(device_id, &hidden4, &mixes4, &base4, &scale, 1, 8, 4, 20, 1.0e-6).unwrap();
        assert_close(&download(&residual4, 32), &mix_f32(&hidden4_values, &expected4.combination, 4).unwrap());
        assert_close(&download(&reduced4, 8), &reduce_f32(&hidden4_values, &expected4.pre, 4).unwrap());
        assert_close(&download(&post4, 4), &expected4.post);

        let head_hidden_values = [1.0, 2.0, 5.0, 8.0, -1.0, 3.0, 2.0, -4.0];
        let head_mixes_values = [0.2, -0.4, 0.5, 0.1];
        let head_base_values = [0.3f32, -0.2];
        let head_scale_values = [0.75f32];
        let head_hidden = upload(device_id, &head_hidden_values);
        let head_mixes = upload(device_id, &head_mixes_values);
        let head_base = upload(device_id, &head_base_values);
        let head_scale = upload(device_id, &head_scale_values);
        let actual = mhc::try_mhc_head_reduce_f32(device_id, &head_hidden, &head_mixes, &head_base, &head_scale, 2, 2, 2, 1.0e-6).unwrap();
        let expected = head_reduce_f32(&head_hidden_values, &head_mixes_values, &head_base_values, head_scale_values[0], 2, 1.0e-6).unwrap();
        assert_close(&download(&actual, 4), &expected);
    }
}
