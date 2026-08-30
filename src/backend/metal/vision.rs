//! Metal 视觉算子能力实现。

use crate::{
    backend::{BackendError, VisionBackend},
    kernel::metal as ops,
};

use super::{
    context::{MetalContext, MetalTensor},
    expect_resident_f16,
    resident::MetalWeight,
};

impl VisionBackend for MetalContext {
    fn vision_tensor_from_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<MetalTensor, BackendError> {
        self.tensor_from_f32(values, rows, cols).map_err(|msg| BackendError::Compute { msg })
    }

    fn vision_tensor_from_f32_bf16(&self, values: &[f32], rows: usize, cols: usize) -> Result<MetalTensor, BackendError> {
        self.tensor_from_f32_bf16(values, rows, cols).map_err(|msg| BackendError::Compute { msg })
    }

    fn add_bias(&self, input: &MetalTensor, bias: &MetalWeight) -> Result<MetalTensor, BackendError> {
        let bias = expect_resident_f16(bias, "视觉 linear bias 需要 resident F16 权重")?;
        ops::tensor::add_bias_tensor(self, input, bias).map_err(|msg| BackendError::Compute { msg })
    }

    fn gelu(&self, input: &MetalTensor) -> Result<MetalTensor, BackendError> {
        ops::tensor::gelu_tensor(self, input).map_err(|msg| BackendError::Compute { msg })
    }

    fn vision_attention(&self, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, cos: &MetalTensor, sin: &MetalTensor, head_count: usize) -> Result<MetalTensor, BackendError> {
        let (query, key) = ops::vision::vision_rope_tensor(self, query, key, cos, sin, head_count).map_err(|msg| BackendError::Compute { msg })?;
        ops::vision::vision_attention_tensor(self, &query, &key, value, head_count).map_err(|msg| BackendError::Compute { msg })
    }

    fn vision_attention_2d(&self, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, cos: &MetalTensor, sin: &MetalTensor, head_count: usize, score_scale: f32) -> Result<MetalTensor, BackendError> {
        let (query, key) = ops::vision::vision_rope_2d_tensor(self, query, key, cos, sin, head_count).map_err(|msg| BackendError::Compute { msg })?;
        ops::vision::vision_attention_tensor_scaled(self, &query, &key, value, head_count, score_scale).map_err(|msg| BackendError::Compute { msg })
    }

    fn vision_clamp(&self, input: &MetalTensor, minimum: f32, maximum: f32) -> Result<MetalTensor, BackendError> {
        ops::vision::vision_clamp_tensor(self, input, minimum, maximum).map_err(|msg| BackendError::Compute { msg })
    }

    fn vision_quick_gelu_gated(&self, gate: &MetalTensor, up: &MetalTensor) -> Result<MetalTensor, BackendError> {
        ops::vision::vision_quick_gelu_gated_tensor(self, gate, up).map_err(|msg| BackendError::Compute { msg })
    }

    fn vision_average_pool(&self, input: &MetalTensor, grid_height: usize, grid_width: usize, kernel_size: usize, output_scale: f32) -> Result<MetalTensor, BackendError> {
        ops::vision::vision_average_pool_tensor(self, input, grid_height, grid_width, kernel_size, output_scale).map_err(|msg| BackendError::Compute { msg })
    }

    fn merge_spatial(&self, input: &MetalTensor, merge_size: usize) -> Result<MetalTensor, BackendError> {
        let merge = merge_size.checked_mul(merge_size).ok_or_else(|| BackendError::Compute { msg: "视觉 merge_size 溢出".to_owned() })?;
        if merge == 0 || !input.rows.is_multiple_of(merge) {
            return Err(BackendError::Compute { msg: format!("视觉 patch rows={} 不能按 merge_size={merge_size} 合并", input.rows) });
        }
        let cols = input.cols.checked_mul(merge).ok_or_else(|| BackendError::Compute { msg: "视觉 merged columns 溢出".to_owned() })?;
        Ok(MetalTensor::new(input.buffer.clone(), input.rows / merge, cols))
    }

    fn scatter_rows(&self, destination: &mut MetalTensor, start_row: usize, source: &MetalTensor) -> Result<(), BackendError> {
        ops::vision::scatter_rows_tensor(self, destination, start_row, source).map_err(|msg| BackendError::Compute { msg })
    }
}
