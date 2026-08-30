//! ROCm 视觉算子能力；图像与视频 patch 全程保持 F32 device resident。

use crate::{
    backend::{BackendError, DiffusionBackend, VaeBackend, VisionBackend},
    kernel::rocm::hip,
};

use super::{RocmContext, compute_error, device_tensor_f32};

impl VisionBackend for RocmContext {
    fn vision_tensor_from_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        self.tensor_from_f32(values.to_vec(), rows, cols).map_err(compute_error)
    }

    fn vision_tensor_zeros(&self, rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        let elements = rows.checked_mul(cols).ok_or_else(|| compute_error("ROCm vision zeros 大小溢出"))?;
        let output = hip::try_zeros_resident_f32(self.device_id, elements).map_err(compute_error)?;
        Ok(device_tensor_f32(output, rows, cols))
    }

    fn vision_tensor_from_f32_bf16(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        // ROCm 视觉路径当前统一使用 resident F32，避免 BF16 patch 在普通 GEMM 前反复转换。
        self.tensor_from_f32(values.to_vec(), rows, cols).map_err(compute_error)
    }

    fn add_bias(&self, input: &Self::Tensor, bias: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        DiffusionBackend::add_row_bias(self, input, bias)
    }

    fn gelu(&self, input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        VaeBackend::vae_gelu(self, input)
    }

    fn vision_attention(&self, query: &Self::Tensor, key: &Self::Tensor, value: &Self::Tensor, cos: &Self::Tensor, sin: &Self::Tensor, head_count: usize) -> Result<Self::Tensor, BackendError> {
        if query.rows != key.rows || query.rows != value.rows || query.cols != key.cols || query.cols != value.cols || cos.rows != query.rows || sin.rows != query.rows || cos.cols != sin.cols {
            return Err(compute_error("ROCm vision attention Q/K/V/RoPE shape 不兼容"));
        }
        if head_count == 0 || !query.cols.is_multiple_of(head_count) || cos.cols == 0 || !cos.cols.is_multiple_of(2) || cos.cols > query.cols / head_count {
            return Err(compute_error(format!("ROCm vision attention query=[{},{}] cos=[{},{}] heads={head_count} 非法", query.rows, query.cols, cos.rows, cos.cols)));
        }
        let query = self.tensor_as_f32(query.clone())?;
        let key = self.tensor_as_f32(key.clone())?;
        let value = self.tensor_as_f32(value.clone())?;
        let cos = self.tensor_as_f32(cos.clone())?;
        let sin = self.tensor_as_f32(sin.clone())?;
        let query_device = query.device.as_deref().ok_or_else(|| compute_error("ROCm vision query 缺少 device buffer"))?;
        let key_device = key.device.as_deref().ok_or_else(|| compute_error("ROCm vision key 缺少 device buffer"))?;
        let value_device = value.device.as_deref().ok_or_else(|| compute_error("ROCm vision value 缺少 device buffer"))?;
        let cos_device = cos.device.as_deref().ok_or_else(|| compute_error("ROCm vision cosine 缺少 device buffer"))?;
        let sin_device = sin.device.as_deref().ok_or_else(|| compute_error("ROCm vision sine 缺少 device buffer"))?;
        let rows = query.rows;
        let head_dim = query.cols / head_count;
        let output = if head_dim <= 128 {
            let (query, key) = hip::try_vision_rope_resident_f32(self.device_id, query_device, key_device, cos_device, sin_device, rows, query.cols, head_count, cos.cols).map_err(compute_error)?;
            let padded_head_dim = head_dim.div_ceil(16) * 16;
            hip::try_full_attention_padded_heads_resident_f32(self.device_id, &query, &key, value_device, rows, head_count, head_dim, padded_head_dim, (head_dim as f32).sqrt().recip())
        } else {
            hip::try_vision_attention_resident_f32(self.device_id, query_device, key_device, value_device, cos_device, sin_device, query.rows, query.cols, head_count, cos.cols)
        }
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, query.rows, query.cols))
    }

    fn merge_spatial(&self, input: &Self::Tensor, merge_size: usize) -> Result<Self::Tensor, BackendError> {
        let merge = merge_size.checked_mul(merge_size).ok_or_else(|| compute_error("ROCm vision merge_size 溢出"))?;
        if merge == 0 || !input.rows.is_multiple_of(merge) {
            return Err(compute_error(format!("ROCm vision patch rows={} 不能按 merge_size={merge_size} 合并", input.rows)));
        }
        let mut output = self.tensor_as_f32(input.clone())?;
        output.rows /= merge;
        output.cols = output.cols.checked_mul(merge).ok_or_else(|| compute_error("ROCm vision merged columns 溢出"))?;
        Ok(output)
    }

    fn scatter_rows(&self, destination: &mut Self::Tensor, start_row: usize, source: &Self::Tensor) -> Result<(), BackendError> {
        if destination.cols != source.cols || start_row.checked_add(source.rows).is_none_or(|end| end > destination.rows) {
            return Err(compute_error(format!("ROCm vision scatter destination=[{},{}] start={start_row} source=[{},{}] 非法", destination.rows, destination.cols, source.rows, source.cols)));
        }
        let resident_destination = self.tensor_as_f32(destination.clone())?;
        let resident_source = self.tensor_as_f32(source.clone())?;
        hip::try_scatter_rows_resident_f32(
            self.device_id,
            resident_destination.device.as_deref().ok_or_else(|| compute_error("ROCm vision scatter destination 缺少 device buffer"))?,
            resident_destination.rows,
            resident_source.device.as_deref().ok_or_else(|| compute_error("ROCm vision scatter source 缺少 device buffer"))?,
            resident_source.rows,
            resident_source.cols,
            start_row,
        )
        .map_err(compute_error)?;
        *destination = resident_destination;
        Ok(())
    }
}
