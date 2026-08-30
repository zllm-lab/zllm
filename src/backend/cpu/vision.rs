//! CPU 视觉算子能力实现:tensor 包装、bias、gelu、视觉 RoPE+注意力、空间合并与 scatter。

use crate::{
    backend::{BackendError, VisionBackend, compute_error as compute},
    kernel::cpu::{
        CpuTensor,
        silu::gelu,
        vision::{vision_attention as vision_attention_kernel, vision_attention_scaled, vision_rope, vision_rope_2d},
    },
};

use super::context::{CpuContext, CpuWeight};

impl VisionBackend for CpuContext {
    fn vision_tensor_from_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<CpuTensor, BackendError> {
        let expected = rows.checked_mul(cols).ok_or_else(|| compute("CPU 视觉 tensor 大小溢出"))?;
        if values.len() != expected {
            return Err(compute(format!("CPU 视觉 tensor 元素 {}，期望 {expected}", values.len())));
        }
        Ok(CpuTensor { data: values.to_vec(), rows, cols })
    }

    fn add_bias(&self, input: &CpuTensor, bias: &CpuWeight) -> Result<CpuTensor, BackendError> {
        if bias.data.len() != input.cols {
            return Err(compute(format!("CPU 视觉 add_bias bias={} input cols={}", bias.data.len(), input.cols)));
        }
        let mut data = input.data.clone();
        for row in data.chunks_exact_mut(input.cols) {
            for (slot, value) in row.iter_mut().enumerate() {
                *value += bias.data[slot];
            }
        }
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn gelu(&self, input: &CpuTensor) -> Result<CpuTensor, BackendError> {
        let mut data = vec![0.0; input.data.len()];
        gelu(&input.data, &mut data);
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn vision_attention(&self, query: &CpuTensor, key: &CpuTensor, value: &CpuTensor, cos: &CpuTensor, sin: &CpuTensor, head_count: usize) -> Result<CpuTensor, BackendError> {
        if head_count == 0 || !query.cols.is_multiple_of(head_count) {
            return Err(compute(format!("CPU 视觉 attention head_count={head_count} 与 cols={} 不匹配", query.cols)));
        }
        if query.rows != key.rows || query.rows != value.rows || query.cols != key.cols || query.cols != value.cols {
            return Err(compute(format!("CPU 视觉 attention shape 异常: Q=[{},{}] K=[{},{}] V=[{},{}]", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols)));
        }
        let head_dim = query.cols / head_count;
        let rotary_dim = cos.cols;
        if cos.rows != query.rows || sin.rows != query.rows || cos.cols != sin.cols || rotary_dim == 0 || rotary_dim > head_dim {
            return Err(compute(format!("CPU 视觉 RoPE shape 异常: cos=[{},{}] sin=[{},{}] head_dim={head_dim}", cos.rows, cos.cols, sin.rows, sin.cols)));
        }
        // 视觉 RoPE 在 attention 内部完成,与 Metal `vision_rope_f16`+`vision_attention_f16` 一致。
        let mut rotated_query = query.data.clone();
        let mut rotated_key = key.data.clone();
        vision_rope(&query.data, &key.data, &cos.data, &sin.data, head_count, head_dim, rotary_dim, &mut rotated_query, &mut rotated_key);
        let mut output = vec![0.0; query.rows * query.cols];
        vision_attention_kernel(&rotated_query, &rotated_key, &value.data, query.cols, head_count, &mut output);
        Ok(CpuTensor { data: output, rows: query.rows, cols: query.cols })
    }

    fn vision_attention_2d(&self, query: &CpuTensor, key: &CpuTensor, value: &CpuTensor, cos: &CpuTensor, sin: &CpuTensor, head_count: usize, score_scale: f32) -> Result<CpuTensor, BackendError> {
        if head_count == 0 || !query.cols.is_multiple_of(head_count) || query.rows != key.rows || query.rows != value.rows || query.cols != key.cols || query.cols != value.cols {
            return Err(compute("CPU 二维视觉 attention shape 异常"));
        }
        let head_dim = query.cols / head_count;
        if !head_dim.is_multiple_of(4) || cos.rows != query.rows || sin.rows != query.rows || cos.cols != head_dim || sin.cols != head_dim {
            return Err(compute("CPU 二维视觉 RoPE shape 异常"));
        }
        let mut rotated_query = vec![0.0; query.data.len()];
        let mut rotated_key = vec![0.0; key.data.len()];
        vision_rope_2d(&query.data, &key.data, &cos.data, &sin.data, head_count, head_dim, &mut rotated_query, &mut rotated_key);
        let mut output = vec![0.0; query.data.len()];
        vision_attention_scaled(&rotated_query, &rotated_key, &value.data, query.cols, head_count, score_scale, &mut output);
        Ok(CpuTensor { data: output, rows: query.rows, cols: query.cols })
    }

    fn vision_clamp(&self, input: &CpuTensor, minimum: f32, maximum: f32) -> Result<CpuTensor, BackendError> {
        if minimum > maximum {
            return Err(compute(format!("CPU 视觉 clamp [{minimum},{maximum}] 非法")));
        }
        Ok(CpuTensor { data: input.data.iter().map(|value| value.clamp(minimum, maximum)).collect(), rows: input.rows, cols: input.cols })
    }

    fn vision_quick_gelu_gated(&self, gate: &CpuTensor, up: &CpuTensor) -> Result<CpuTensor, BackendError> {
        if gate.rows != up.rows || gate.cols != up.cols {
            return Err(compute("CPU QuickGEGLU shape 不一致"));
        }
        let data = gate.data.iter().zip(&up.data).map(|(gate, up)| gate / (1.0 + (-1.702 * gate).exp()) * up).collect();
        Ok(CpuTensor { data, rows: gate.rows, cols: gate.cols })
    }

    fn vision_average_pool(&self, input: &CpuTensor, grid_height: usize, grid_width: usize, kernel_size: usize, output_scale: f32) -> Result<CpuTensor, BackendError> {
        if kernel_size == 0 || input.rows != grid_height * grid_width || !grid_height.is_multiple_of(kernel_size) || !grid_width.is_multiple_of(kernel_size) {
            return Err(compute("CPU 视觉平均池化 shape 非法"));
        }
        let out_height = grid_height / kernel_size;
        let out_width = grid_width / kernel_size;
        let mut data = vec![0.0; out_height * out_width * input.cols];
        let factor = output_scale / (kernel_size * kernel_size) as f32;
        for out_y in 0..out_height {
            for out_x in 0..out_width {
                let output = (out_y * out_width + out_x) * input.cols;
                for y in 0..kernel_size {
                    for x in 0..kernel_size {
                        let source = ((out_y * kernel_size + y) * grid_width + out_x * kernel_size + x) * input.cols;
                        for column in 0..input.cols {
                            data[output + column] += input.data[source + column] * factor;
                        }
                    }
                }
            }
        }
        Ok(CpuTensor { data, rows: out_height * out_width, cols: input.cols })
    }

    fn merge_spatial(&self, input: &CpuTensor, merge_size: usize) -> Result<CpuTensor, BackendError> {
        let merge = merge_size.checked_mul(merge_size).ok_or_else(|| compute("CPU 视觉 merge_size 溢出"))?;
        if merge == 0 || !input.rows.is_multiple_of(merge) {
            return Err(compute(format!("CPU 视觉 merge rows={} 不能按 merge_size={merge_size} 合并", input.rows)));
        }
        let cols = input.cols.checked_mul(merge).ok_or_else(|| compute("CPU 视觉 merged columns 溢出"))?;
        // 纯 reshape:patch 行序已使每 merge 行连续对应一行 merge 段,底层内存顺序就绪。
        Ok(CpuTensor { data: input.data.clone(), rows: input.rows / merge, cols })
    }

    fn scatter_rows(&self, destination: &mut CpuTensor, start_row: usize, source: &CpuTensor) -> Result<(), BackendError> {
        if destination.cols != source.cols || start_row.checked_add(source.rows).is_none_or(|end| end > destination.rows) {
            return Err(compute(format!("CPU 视觉 scatter destination=[{},{}] start={start_row} source=[{},{}]", destination.rows, destination.cols, source.rows, source.cols)));
        }
        let base = start_row * destination.cols;
        destination.data[base..base + source.data.len()].copy_from_slice(&source.data);
        Ok(())
    }
}
