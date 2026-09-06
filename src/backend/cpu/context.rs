//! CPU tensor、权重与基础算子能力。

use std::borrow::Cow;

use crate::{
    backend::{Backend, BackendError, BackendResources, LinearWeight, checked_elements, compute_error as compute},
    kernel::cpu::{
        CpuTensor, ggml_quant,
        matmul::matmul,
        rmsnorm::{grouped_rmsnorm, rmsnorm_with_weight_offset},
        silu::{gelu_tanh_mul, silu_mul, situ_mul, swiglu_oai_mul},
        w4a16::{W8A8VnniMatrix, matmul_w4a16_matrix, matmul_w8a16_matrix, matvec_w4a16_matrix, matvec_w8a16_matrix},
    },
    moe::Activation,
    weight::{
        container::gguf::GgufMatrix,
        format::quantization::{QuantizedMatrixRef, W4A16Matrix, W8A16Matrix},
    },
};

use super::attention::CpuKvCache;

#[cfg(target_os = "linux")]
fn advise_huge_pages(values: &mut [f32]) {
    const HUGE_PAGE: usize = 2 * 1024 * 1024;
    const MADV_HUGEPAGE: i32 = 14;
    let begin = values.as_mut_ptr() as usize;
    let end = begin.saturating_add(std::mem::size_of_val(values));
    let aligned_begin = begin.saturating_add(HUGE_PAGE - 1) & !(HUGE_PAGE - 1);
    let aligned_end = end & !(HUGE_PAGE - 1);
    if aligned_begin >= aligned_end {
        return;
    }
    unsafe extern "C" {
        fn madvise(address: *mut std::ffi::c_void, length: usize, advice: i32) -> i32;
    }
    unsafe {
        let _ = madvise(aligned_begin as *mut std::ffi::c_void, aligned_end - aligned_begin, MADV_HUGEPAGE);
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_huge_pages(_values: &mut [f32]) {}

/// CPU resident 权重。量化权重只在准备阶段解码一次，token 循环直接读 f32。
#[derive(Debug, Clone)]
pub struct CpuWeight {
    pub(crate) data: Vec<f32>,
    pub(crate) gguf: Option<GgufMatrix>,
    gguf_resident: bool,
    /// compressed-tensors W4A16 保持打包常驻，matvec 时在线反量化(对标 GGUF)。
    pub(crate) w4a16: Option<W4A16Matrix>,
    /// DSpark 等 resident W8 权重保持压缩态，避免 CPU 路径展开成 4 倍 F32。
    pub(crate) w8a16: Option<W8A16Matrix>,
    /// Zen 4 AVX512-VNNI 路径的 16-output interleave；只供 W8×A8 DSpark 计算。
    pub(crate) w8a8_vnni: Option<W8A8VnniMatrix>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl CpuWeight {
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// 将 packed GGUF 权重固定在主存；资源规划确认容量充足时由 runtime 显式启用。
    #[cfg_attr(not(feature = "with-cuda"), allow(dead_code))]
    pub(crate) fn make_gguf_resident(&mut self) -> Result<usize, String> {
        let Some(matrix) = &self.gguf else {
            return Ok(0);
        };
        let bytes = matrix.bytes()?.len();
        self.gguf_resident = true;
        Ok(bytes)
    }
}

/// 无设备状态的 CPU 执行上下文。
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuContext;

impl BackendResources for CpuContext {
    type Tensor = CpuTensor;
    type Weight = CpuWeight;
    type Cache = CpuKvCache;
    type LayerScope<'a>
        = ()
    where
        Self: 'a;

    fn layer_scope(&self) -> Self::LayerScope<'_> {}

    fn token_rows(&self, tensor: &CpuTensor) -> usize {
        tensor.rows
    }

    fn token_cols(&self, tensor: &CpuTensor) -> usize {
        tensor.cols
    }

    fn tensor_allocated_bytes(&self, tensor: &CpuTensor) -> u64 {
        tensor.data.capacity().saturating_mul(std::mem::size_of::<f32>()) as u64
    }

    fn begin_batch(&self) {}

    fn finish_batch(&self) {}

    fn prepare_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<CpuWeight, BackendError> {
        let expected = checked_elements(rows, cols, "CPU weight")?;
        if let LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(matrix)) = weight {
            if matrix.rows != rows || matrix.columns != cols {
                return Err(compute(format!("GGUF weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.columns)));
            }
            return Ok(CpuWeight { data: Vec::new(), gguf: Some(matrix.clone()), gguf_resident: false, w4a16: None, w8a16: None, w8a8_vnni: None, rows, cols });
        }
        if let LinearWeight::Quantized(QuantizedMatrixRef::W4A16(matrix)) = weight {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute(format!("W4A16 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            // 保持打包常驻，避免展开成 ~6× 体积的 f32；matvec 在线反量化。
            return Ok(CpuWeight { data: Vec::new(), gguf: None, gguf_resident: false, w4a16: Some(matrix.clone()), w8a16: None, w8a8_vnni: None, rows, cols });
        }
        if let LinearWeight::Quantized(QuantizedMatrixRef::W8A16(matrix)) = weight {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute(format!("W8A16 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            let w8a8_vnni = W8A8VnniMatrix::try_repack(matrix.packed(), matrix.scales(), matrix.scale_dtype(), matrix.group_size(), rows, cols).map_err(compute)?;
            return Ok(CpuWeight { data: Vec::new(), gguf: None, gguf_resident: false, w4a16: None, w8a16: Some(matrix.clone()), w8a8_vnni, rows, cols });
        }
        let mut data = match weight {
            LinearWeight::F32(values) => values.to_vec(),
            LinearWeight::F16(values) => values.iter().map(|value| value.to_f32()).collect(),
            LinearWeight::Bf16Bytes(bytes) => {
                let expected_bytes = expected.checked_mul(2).ok_or_else(|| compute("CPU BF16 weight 大小溢出"))?;
                if bytes.len() != expected_bytes {
                    return Err(compute(format!("CPU BF16 weight 字节数 {}，期望 {expected_bytes}", bytes.len())));
                }
                bytes.chunks_exact(2).map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32()).collect()
            }
            LinearWeight::Quantized(matrix) => {
                if matrix.rows() != rows || matrix.cols() != cols {
                    return Err(compute(format!("{} weight shape [{},{}]，期望 [{rows},{cols}]", matrix.name(), matrix.rows(), matrix.cols())));
                }
                matrix.decode().map_err(compute)?
            }
        };
        if data.len() != expected {
            return Err(compute(format!("CPU weight 元素数 {}，期望 {expected}", data.len())));
        }
        advise_huge_pages(&mut data);
        Ok(CpuWeight { data, gguf: None, gguf_resident: false, w4a16: None, w8a16: None, w8a8_vnni: None, rows, cols })
    }

    fn prepare_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<CpuWeight, BackendError> {
        let expected = checked_elements(rows, cols, "CPU F32 weight")?;
        if values.len() != expected {
            return Err(compute(format!("CPU F32 weight 元素数 {}，期望 {expected}", values.len())));
        }
        Ok(CpuWeight { data: values.to_vec(), gguf: None, gguf_resident: false, w4a16: None, w8a16: None, w8a8_vnni: None, rows, cols })
    }
}

impl Backend for CpuContext {
    fn linear(&self, input: &CpuTensor, weight: &CpuWeight) -> Result<CpuTensor, BackendError> {
        if input.cols != weight.cols {
            return Err(compute(format!("CPU linear input cols={}，weight=[{},{}]", input.cols, weight.rows, weight.cols)));
        }
        if input.data.len() != checked_elements(input.rows, input.cols, "CPU linear input")? {
            return Err(compute("CPU linear input 数据长度与 shape 不符"));
        }
        let mut output = CpuTensor { data: vec![0.0; checked_elements(input.rows, weight.rows, "CPU linear output")?], rows: input.rows, cols: weight.rows };
        if let Some(matrix) = &weight.gguf {
            let packed = if weight.gguf_resident { Cow::Borrowed(matrix.bytes().map_err(compute)?) } else { Cow::Owned(matrix.read_bytes().map_err(compute)?) };
            if input.rows > 1 {
                ggml_quant::matmul(matrix.tensor_type.0, &packed, matrix.rows, matrix.columns, input.rows, &input.data, &mut output.data).map_err(compute)?;
            } else {
                ggml_quant::matvec(matrix.tensor_type.0, &packed, matrix.rows, matrix.columns, &input.data, &mut output.data).map_err(compute)?;
            }
        } else if let Some(matrix) = &weight.w4a16 {
            if input.rows > 1 {
                // prefill:每权重行只反量化一次,复用于全部 token,避免逐 token 重读打包权重。
                matmul_w4a16_matrix(matrix.packed(), matrix.scales(), matrix.scale_dtype(), matrix.group_size(), weight.rows, weight.cols, input.rows, &input.data, &mut output.data).map_err(compute)?;
            } else {
                for (input, output) in input.data.chunks_exact(input.cols).zip(output.data.chunks_exact_mut(weight.rows)) {
                    matvec_w4a16_matrix(matrix.packed(), matrix.scales(), matrix.scale_dtype(), matrix.group_size(), weight.rows, weight.cols, input, output).map_err(compute)?;
                }
            }
        } else if let Some(matrix) = &weight.w8a8_vnni {
            matrix.matmul(input.rows, &input.data, &mut output.data).map_err(compute)?;
        } else if let Some(matrix) = &weight.w8a16 {
            if input.rows > 1 {
                matmul_w8a16_matrix(matrix.packed(), matrix.scales(), matrix.scale_dtype(), matrix.group_size(), weight.rows, weight.cols, input.rows, &input.data, &mut output.data).map_err(compute)?;
            } else {
                matvec_w8a16_matrix(matrix.packed(), matrix.scales(), matrix.scale_dtype(), matrix.group_size(), weight.rows, weight.cols, &input.data, &mut output.data).map_err(compute)?;
            }
        } else {
            matmul(&input.data, &weight.data, input.rows, input.cols, weight.rows, &mut output.data);
        }
        Ok(output)
    }

    fn dual_linear(&self, input: &CpuTensor, first: &CpuWeight, second: &CpuWeight) -> Result<(CpuTensor, CpuTensor), BackendError> {
        if let Some(result) = shared_quant_linear(input, &[first, second]) {
            let mut outputs = result?;
            let second = outputs.pop().expect("共享量化输出与权重数量一致");
            let first = outputs.pop().expect("共享量化输出与权重数量一致");
            return Ok((first, second));
        }
        Ok((self.linear(input, first)?, self.linear(input, second)?))
    }

    fn triple_linear(&self, input: &CpuTensor, first: &CpuWeight, second: &CpuWeight, third: &CpuWeight) -> Result<(CpuTensor, CpuTensor, CpuTensor), BackendError> {
        if let Some(result) = shared_quant_linear(input, &[first, second, third]) {
            let mut outputs = result?;
            let third = outputs.pop().expect("共享量化输出与权重数量一致");
            let second = outputs.pop().expect("共享量化输出与权重数量一致");
            let first = outputs.pop().expect("共享量化输出与权重数量一致");
            return Ok((first, second, third));
        }
        Ok((self.linear(input, first)?, self.linear(input, second)?, self.linear(input, third)?))
    }

    fn select_row(&self, input: &CpuTensor, row: usize) -> Result<CpuTensor, BackendError> {
        if row >= input.rows {
            return Err(compute(format!("CPU select_row {row} 越界，rows={}", input.rows)));
        }
        Ok(CpuTensor { data: input.row(row).to_vec(), rows: 1, cols: input.cols })
    }

    fn select_rows(&self, input: &CpuTensor, rows: &[u32]) -> Result<CpuTensor, BackendError> {
        if rows.is_empty() || rows.iter().any(|&row| row as usize >= input.rows) {
            return Err(compute(format!("CPU select_rows 越界，input_rows={} rows={rows:?}", input.rows)));
        }
        let mut data = Vec::with_capacity(rows.len() * input.cols);
        for &row in rows {
            data.extend_from_slice(input.row(row as usize));
        }
        Ok(CpuTensor { data, rows: rows.len(), cols: input.cols })
    }

    fn argmax(&self, input: &CpuTensor) -> Result<u32, BackendError> {
        let (&first, rest) = input.data.split_first().ok_or_else(|| compute("CPU argmax 输入为空"))?;
        let mut best_index = 0usize;
        let mut best_value = first;
        for (offset, &value) in rest.iter().enumerate() {
            if value > best_value {
                best_index = offset + 1;
                best_value = value;
            }
        }
        u32::try_from(best_index).map_err(|_| compute("CPU argmax index 超出 u32"))
    }

    fn argmax_excluding(&self, input: &CpuTensor, excluded: &[u32]) -> Result<u32, BackendError> {
        let mut best = None;
        for (index, &value) in input.data.iter().enumerate() {
            if excluded.contains(&(index as u32)) {
                continue;
            }
            if best.is_none_or(|(_, best_value)| value > best_value) {
                best = Some((index, value));
            }
        }
        let (index, _) = best.ok_or_else(|| compute("CPU argmax 没有可选 token"))?;
        u32::try_from(index).map_err(|_| compute("CPU argmax index 超出 u32"))
    }

    fn sample_top_p(&self, input: &CpuTensor, temperature: f32, top_p: f32, random: f32) -> Result<u32, BackendError> {
        crate::kernel::cpu::sample_top_p(&input.data, temperature, top_p, random).map_err(compute)
    }

    fn sample_top_p_excluding(&self, input: &CpuTensor, temperature: f32, top_p: f32, random: f32, excluded: &[u32]) -> Result<u32, BackendError> {
        crate::kernel::cpu::sample_top_p_excluding(&input.data, temperature, top_p, random, excluded).map_err(compute)
    }

    fn rmsnorm(&self, input: &CpuTensor, weight: &CpuWeight, eps: f32) -> Result<CpuTensor, BackendError> {
        apply_rmsnorm(input, weight, eps, 0.0, "RMSNorm")
    }

    fn grouped_rmsnorm(&self, input: &CpuTensor, weight: &CpuWeight, eps: f32, groups: usize) -> Result<CpuTensor, BackendError> {
        if weight.data.len() != input.cols {
            return Err(compute(format!("CPU GroupedRMSNorm weight={}，input cols={}", weight.data.len(), input.cols)));
        }
        if groups == 0 || !input.cols.is_multiple_of(groups) {
            return Err(compute(format!("CPU GroupedRMSNorm groups={groups} 无法整除 cols={}", input.cols)));
        }
        let mut output = CpuTensor { data: vec![0.0; input.data.len()], rows: input.rows, cols: input.cols };
        for row in 0..input.rows {
            grouped_rmsnorm(input.row(row), &weight.data, eps, groups, output.row_mut(row));
        }
        Ok(output)
    }

    fn gemma_rmsnorm(&self, input: &CpuTensor, weight: &CpuWeight, eps: f32) -> Result<CpuTensor, BackendError> {
        apply_rmsnorm(input, weight, eps, 1.0, "GemmaRMSNorm")
    }

    fn layernorm_bias(&self, input: &CpuTensor, weight: &CpuWeight, bias: &CpuWeight, eps: f32) -> Result<CpuTensor, BackendError> {
        let data = crate::kernel::cpu::vae::layer_norm(&input.data, &weight.data, &bias.data, input.cols, eps).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn split_columns(&self, input: &CpuTensor, left_columns: usize) -> Result<(CpuTensor, CpuTensor), BackendError> {
        if left_columns > input.cols {
            return Err(compute(format!("CPU split left={left_columns} 超过 cols={}", input.cols)));
        }
        let right_columns = input.cols - left_columns;
        let mut left = CpuTensor { data: Vec::with_capacity(input.rows * left_columns), rows: input.rows, cols: left_columns };
        let mut right = CpuTensor { data: Vec::with_capacity(input.rows * right_columns), rows: input.rows, cols: right_columns };
        for row in 0..input.rows {
            left.data.extend_from_slice(&input.row(row)[..left_columns]);
            right.data.extend_from_slice(&input.row(row)[left_columns..]);
        }
        Ok((left, right))
    }

    fn split_interleaved_columns(&self, input: &CpuTensor, block_columns: usize) -> Result<(CpuTensor, CpuTensor), BackendError> {
        let pair_columns = block_columns.checked_mul(2).ok_or_else(|| compute("CPU interleaved split block 溢出"))?;
        if block_columns == 0 || !input.cols.is_multiple_of(pair_columns) {
            return Err(compute(format!("CPU interleaved split cols={} block={block_columns} 非法", input.cols)));
        }
        let output_columns = input.cols / 2;
        let mut left = CpuTensor { data: Vec::with_capacity(input.rows * output_columns), rows: input.rows, cols: output_columns };
        let mut right = CpuTensor { data: Vec::with_capacity(input.rows * output_columns), rows: input.rows, cols: output_columns };
        for row in 0..input.rows {
            for pair in input.row(row).chunks_exact(pair_columns) {
                left.data.extend_from_slice(&pair[..block_columns]);
                right.data.extend_from_slice(&pair[block_columns..]);
            }
        }
        Ok((left, right))
    }

    fn concat_columns(&self, left: &CpuTensor, right: &CpuTensor) -> Result<CpuTensor, BackendError> {
        if left.rows != right.rows {
            return Err(compute(format!("CPU concat rows {} 与 {} 不一致", left.rows, right.rows)));
        }
        let cols = left.cols.checked_add(right.cols).ok_or_else(|| compute("CPU concat columns 溢出"))?;
        let mut data = Vec::with_capacity(left.rows * cols);
        for row in 0..left.rows {
            data.extend_from_slice(left.row(row));
            data.extend_from_slice(right.row(row));
        }
        Ok(CpuTensor { data, rows: left.rows, cols })
    }

    fn rope(&self, input: &CpuTensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<CpuTensor, BackendError> {
        apply_rope(input, head_count, rotary_dim, layout, position, cos, sin, false)
    }

    fn rope_prefix(&self, input: &CpuTensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<CpuTensor, BackendError> {
        apply_rope(input, head_count, rotary_dim, layout, position, cos, sin, true)
    }

    fn add(&self, left: &CpuTensor, right: &CpuTensor) -> Result<CpuTensor, BackendError> {
        if left.rows != right.rows || left.cols != right.cols || left.data.len() != right.data.len() {
            return Err(compute(format!("CPU add shape [{},{}] 与 [{},{}] 不一致", left.rows, left.cols, right.rows, right.cols)));
        }
        // DSpark residual add 在热路径上,逐元素迭代改为 SIMD;尾部按标量收尾。
        let mut data = vec![0.0_f32; left.data.len()];
        let full = left.data.len() / 8 * 8;
        let (body, tail) = data.split_at_mut(full);
        for ((left, right), output) in left.data.chunks_exact(8).zip(right.data.chunks_exact(8)).zip(body.chunks_exact_mut(8)) {
            let left = wide::f32x8::from(<[f32; 8]>::try_from(left).unwrap());
            let right = wide::f32x8::from(<[f32; 8]>::try_from(right).unwrap());
            output.copy_from_slice(&(left + right).to_array());
        }
        for index in 0..tail.len() {
            tail[index] = left.data[full + index] + right.data[full + index];
        }
        Ok(CpuTensor { data, rows: left.rows, cols: left.cols })
    }

    fn add_scaled(&self, left: &CpuTensor, right: &CpuTensor, scale: f32) -> Result<CpuTensor, BackendError> {
        if left.rows != right.rows || left.cols != right.cols || left.data.len() != right.data.len() {
            return Err(compute(format!("CPU add_scaled shape [{},{}] 与 [{},{}] 不一致", left.rows, left.cols, right.rows, right.cols)));
        }
        if !scale.is_finite() {
            return Err(compute(format!("CPU add_scaled scale={scale} 非法")));
        }
        Ok(CpuTensor { data: left.data.iter().zip(&right.data).map(|(left, right)| (left + right) * scale).collect(), rows: left.rows, cols: left.cols })
    }

    fn cast_f16(&self, input: &CpuTensor) -> Result<CpuTensor, BackendError> {
        // CPU 张量本就 f32,恒等返回。
        Ok(input.clone())
    }

    fn sigmoid_gate(&self, input: &CpuTensor, gate: &CpuTensor) -> Result<CpuTensor, BackendError> {
        if input.rows != gate.rows || (gate.cols != 1 && gate.cols != input.cols) {
            return Err(compute(format!("CPU sigmoid gate shape input=[{},{}], gate=[{},{}]", input.rows, input.cols, gate.rows, gate.cols)));
        }
        let mut output = input.clone();
        for row in 0..input.rows {
            for column in 0..input.cols {
                let gate_column = if gate.cols == 1 { 0 } else { column };
                let value = gate.data[row * gate.cols + gate_column];
                output.data[row * input.cols + column] *= 1.0 / (1.0 + (-value).exp());
            }
        }
        Ok(output)
    }

    fn softplus_gate(&self, input: &CpuTensor, gate: &CpuTensor) -> Result<CpuTensor, BackendError> {
        if input.rows != gate.rows || gate.cols == 0 || input.cols % gate.cols != 0 {
            return Err(compute(format!("CPU softplus gate shape input=[{},{}], gate=[{},{}] 不满足逐头广播", input.rows, input.cols, gate.rows, gate.cols)));
        }
        let span = input.cols / gate.cols;
        let softplus = |value: f32| if value > 20.0 { value } else { value.exp().ln_1p() };
        let mut output = input.clone();
        for row in 0..input.rows {
            for column in 0..input.cols {
                let value = gate.data[row * gate.cols + column / span];
                output.data[row * input.cols + column] *= softplus(value);
            }
        }
        Ok(output)
    }

    fn gated_linear(&self, input: &CpuTensor, gate: &CpuWeight, up: &CpuWeight, activation: &Activation) -> Result<CpuTensor, BackendError> {
        if matches!(activation, Activation::Silu) && gate.rows == up.rows && gate.cols == up.cols && input.cols == gate.cols {
            if let (Some(gate_matrix), Some(up_matrix)) = (&gate.gguf, &up.gguf)
                && gate_matrix.tensor_type.0 == 12
                && up_matrix.tensor_type.0 == 12
            {
                let gate_packed = if gate.gguf_resident { Cow::Borrowed(gate_matrix.bytes().map_err(compute)?) } else { Cow::Owned(gate_matrix.read_bytes().map_err(compute)?) };
                let up_packed = if up.gguf_resident { Cow::Borrowed(up_matrix.bytes().map_err(compute)?) } else { Cow::Owned(up_matrix.read_bytes().map_err(compute)?) };
                let mut output = CpuTensor { data: vec![0.0; checked_elements(input.rows, gate.rows, "CPU gated linear output")?], rows: input.rows, cols: gate.rows };
                if input.rows == 1 {
                    ggml_quant::gated_silu_matvec(12, &gate_packed, &up_packed, gate.rows, gate.cols, &input.data, &mut output.data).map_err(compute)?;
                } else {
                    ggml_quant::gated_silu_matmul(12, &gate_packed, &up_packed, gate.rows, gate.cols, input.rows, &input.data, &mut output.data).map_err(compute)?;
                }
                return Ok(output);
            }
        }
        let (gate, up) = self.dual_linear(input, gate, up)?;
        self.gated_activation(&gate, &up, activation)
    }

    fn gated_activation(&self, gate: &CpuTensor, up: &CpuTensor, activation: &Activation) -> Result<CpuTensor, BackendError> {
        if gate.rows != up.rows || gate.cols != up.cols || gate.data.len() != up.data.len() {
            return Err(compute(format!("CPU gated activation shape [{},{}] 与 [{},{}] 不一致", gate.rows, gate.cols, up.rows, up.cols)));
        }
        let mut data = vec![0.0; gate.data.len()];
        match activation {
            Activation::Silu => silu_mul(&gate.data, &up.data, &mut data),
            Activation::SiluClamped { limit } => crate::kernel::cpu::silu::silu_clamped_mul(&gate.data, &up.data, *limit, &mut data),
            Activation::Situ { beta, linear_beta } => situ_mul(&gate.data, &up.data, *beta, *linear_beta, &mut data),
            Activation::SwigluOai { alpha, limit } => swiglu_oai_mul(&gate.data, &up.data, *alpha, *limit, &mut data),
            Activation::GeluTanh => gelu_tanh_mul(&gate.data, &up.data, &mut data),
        }
        Ok(CpuTensor { data, rows: gate.rows, cols: gate.cols })
    }
}

impl crate::backend::SegmentedTensorBackend for CpuContext {
    fn concat_token_rows(&self, tensors: &[&CpuTensor]) -> Result<CpuTensor, BackendError> {
        let Some(first) = tensors.first() else {
            return Err(compute("CPU concat rows 输入为空"));
        };
        if tensors.iter().any(|tensor| tensor.cols != first.cols) {
            return Err(compute("CPU concat rows 列数不一致"));
        }
        let rows = tensors.iter().try_fold(0usize, |rows, tensor| rows.checked_add(tensor.rows).ok_or_else(|| compute("CPU concat rows 溢出")))?;
        let mut data = Vec::with_capacity(rows.checked_mul(first.cols).ok_or_else(|| compute("CPU concat elements 溢出"))?);
        for tensor in tensors {
            data.extend_from_slice(&tensor.data);
        }
        Ok(CpuTensor { data, rows, cols: first.cols })
    }

    fn concat_token_rows_reserved(&self, tensors: &[&CpuTensor], capacity_rows: usize) -> Result<CpuTensor, BackendError> {
        let Some(first) = tensors.first() else {
            return Err(compute("CPU concat reserved 输入为空"));
        };
        let rows = tensors.iter().try_fold(0usize, |rows, tensor| rows.checked_add(tensor.rows).ok_or_else(|| compute("CPU concat reserved rows 溢出")))?;
        if capacity_rows < rows || tensors.iter().any(|tensor| tensor.cols != first.cols) {
            return Err(compute(format!("CPU concat reserved capacity={capacity_rows} rows={rows} columns 不一致")));
        }
        let mut data = Vec::<f32>::with_capacity(capacity_rows.checked_mul(first.cols).ok_or_else(|| compute("CPU concat reserved elements 溢出"))?);
        // resident K/V cache 等大张量:先建议 THP 再首写,attention 按 head
        // 跨步扫描时 TLB 覆盖 2MB/页,避免逐行换页。
        crate::kernel::cpu::advise_huge_pages_region(data.as_mut_ptr().cast(), data.capacity() * std::mem::size_of::<f32>());
        for tensor in tensors {
            data.extend_from_slice(&tensor.data);
        }
        Ok(CpuTensor { data, rows, cols: first.cols })
    }

    fn append_token_rows_reserved(&self, mut prefix: CpuTensor, suffix: &CpuTensor, capacity_rows: usize) -> Result<CpuTensor, BackendError> {
        let rows = prefix.rows.checked_add(suffix.rows).ok_or_else(|| compute("CPU append reserved rows 溢出"))?;
        if capacity_rows < rows || prefix.cols != suffix.cols {
            return Err(compute(format!("CPU append reserved capacity={capacity_rows} rows={rows} columns={}/{}", prefix.cols, suffix.cols)));
        }
        let capacity = capacity_rows.checked_mul(prefix.cols).ok_or_else(|| compute("CPU append reserved elements 溢出"))?;
        if prefix.data.capacity() < capacity {
            prefix.data.reserve(capacity - prefix.data.len());
            crate::kernel::cpu::advise_huge_pages_region(prefix.data.as_mut_ptr().cast(), prefix.data.capacity() * std::mem::size_of::<f32>());
        }
        prefix.data.extend_from_slice(&suffix.data);
        prefix.rows = rows;
        Ok(prefix)
    }

    fn slice_token_rows(&self, tensor: &CpuTensor, row_start: usize, rows: usize) -> Result<CpuTensor, BackendError> {
        let row_end = row_start.checked_add(rows).ok_or_else(|| compute("CPU slice rows 溢出"))?;
        if row_end > tensor.rows {
            return Err(compute(format!("CPU slice rows=[{row_start},{row_end}) 超出 {}", tensor.rows)));
        }
        Ok(CpuTensor { data: tensor.data[row_start * tensor.cols..row_end * tensor.cols].to_vec(), rows, cols: tensor.cols })
    }
}

/// 多路 W8A8 权重共享同一输入时只做一次 A8 量化(Q/K/V、gate/up 共享
/// encoder);任一路布局不匹配返回 None,调用方回退逐路 linear。
#[cfg(target_arch = "x86_64")]
fn shared_quant_linear(input: &CpuTensor, weights: &[&CpuWeight]) -> Option<Result<Vec<CpuTensor>, BackendError>> {
    use crate::kernel::cpu::w4a16::quantize_w8a8_input;

    if weights.iter().any(|weight| weight.w8a8_vnni.is_none()) {
        return None;
    }
    let matrices = weights.iter().map(|weight| weight.w8a8_vnni.as_ref().expect("已检查 w8a8_vnni")).collect::<Vec<_>>();
    let (cols, group_size) = matrices.first()?.quant_layout();
    if cols != input.cols || matrices.iter().any(|matrix| matrix.quant_layout() != (cols, group_size)) {
        return None;
    }
    let rows = weights.iter().map(|weight| weight.rows).collect::<Vec<_>>();
    Some((|| {
        let quantized = quantize_w8a8_input(&input.data, input.rows, input.cols, group_size).map_err(compute)?;
        let mut buffers = Vec::with_capacity(matrices.len());
        for &rows in &rows {
            buffers.push(vec![0.0_f32; checked_elements(input.rows, rows, "CPU shared quant linear output")?]);
        }
        let mut slices = buffers.iter_mut().map(|buffer| buffer.as_mut_slice()).collect::<Vec<_>>();
        // 多路投影合并成一次 fixed team 分发,消除逐矩阵唤醒/barrier。
        W8A8VnniMatrix::matmul_quantized_many(&matrices, input.rows, &quantized, &mut slices).map_err(compute)?;
        Ok(buffers.into_iter().zip(&rows).map(|(data, &rows)| CpuTensor { data, rows: input.rows, cols: rows }).collect())
    })())
}

#[cfg(not(target_arch = "x86_64"))]
fn shared_quant_linear(_input: &CpuTensor, _weights: &[&CpuWeight]) -> Option<Result<Vec<CpuTensor>, BackendError>> {
    None
}

fn apply_rmsnorm(input: &CpuTensor, weight: &CpuWeight, eps: f32, weight_offset: f32, name: &str) -> Result<CpuTensor, BackendError> {
    if weight.data.len() != input.cols {
        return Err(compute(format!("CPU {name} weight={}，input cols={}", weight.data.len(), input.cols)));
    }
    let mut output = CpuTensor { data: vec![0.0; input.data.len()], rows: input.rows, cols: input.cols };
    for row in 0..input.rows {
        rmsnorm_with_weight_offset(input.row(row), &weight.data, eps, weight_offset, output.row_mut(row), name);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn apply_rope(input: &CpuTensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32], prefix: bool) -> Result<CpuTensor, BackendError> {
    let placement = if prefix { crate::attention::rope::RotaryPlacement::Prefix } else { crate::attention::rope::RotaryPlacement::Suffix };
    // SplitHalf 且半维是 8 的倍数时走 f32x8 快路径;运算顺序与标量
    // reference 逐位一致(mul/sub/add 不换序),其余布局回退 reference。
    let head_dim = if head_count > 0 { input.cols / head_count } else { 0 };
    if matches!(layout, crate::attention::rope::RotaryLayout::SplitHalf) && rotary_dim % 16 == 0 && rotary_dim <= head_dim && cos.len() == sin.len() && (position + input.rows) * (rotary_dim / 2) <= cos.len() {
        let half = rotary_dim / 2;
        let mut output = input.data.clone();
        for row in 0..input.rows {
            let cos_row = &cos[(position + row) * half..(position + row + 1) * half];
            let sin_row = &sin[(position + row) * half..(position + row + 1) * half];
            for head in 0..head_count {
                let head_start = row * input.cols + head * head_dim;
                let start = if prefix { head_start } else { head_start + head_dim - rotary_dim };
                let real = &input.data[start..start + half];
                let imaginary = &input.data[start + half..start + rotary_dim];
                let rotary = &mut output[start..start + rotary_dim];
                let (real_out, imaginary_out) = rotary.split_at_mut(half);
                for ((((real, imaginary), cos), sin), (real_out, imaginary_out)) in
                    real.chunks_exact(8).zip(imaginary.chunks_exact(8)).zip(cos_row.chunks_exact(8)).zip(sin_row.chunks_exact(8)).zip(real_out.chunks_exact_mut(8).zip(imaginary_out.chunks_exact_mut(8)))
                {
                    let real = wide::f32x8::from(<[f32; 8]>::try_from(real).unwrap());
                    let imaginary = wide::f32x8::from(<[f32; 8]>::try_from(imaginary).unwrap());
                    let cos = wide::f32x8::from(<[f32; 8]>::try_from(cos).unwrap());
                    let sin = wide::f32x8::from(<[f32; 8]>::try_from(sin).unwrap());
                    real_out.copy_from_slice(&(real * cos - imaginary * sin).to_array());
                    imaginary_out.copy_from_slice(&(imaginary * cos + real * sin).to_array());
                }
            }
        }
        return Ok(CpuTensor { data: output, rows: input.rows, cols: input.cols });
    }
    let data = crate::attention::rope::apply_f32(&input.data, input.rows, input.cols, head_count, rotary_dim, position, cos, sin, layout, placement).map_err(compute)?;
    Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 共享量化多路投影与逐路linear一致() {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::{kernel::cpu::w4a16::W8A8VnniMatrix, weight::format::quantization::ScaleDType};

            let build = |rows: usize, seed: u8| {
                let cols = 256;
                let group_size = 128;
                let packed = (0..rows * cols).map(|index| ((index * 31 + seed as usize * 7) % 255 + 1) as u8).collect::<Vec<_>>();
                let scales = (0..rows * (cols / group_size)).flat_map(|index| (0.001_f32 * (index + 1) as f32).to_le_bytes()).collect::<Vec<_>>();
                let matrix = W8A8VnniMatrix::try_repack(&packed, &scales, ScaleDType::F32, group_size, rows, cols).unwrap().unwrap();
                CpuWeight { data: Vec::new(), gguf: None, gguf_resident: false, w4a16: None, w8a16: None, w8a8_vnni: Some(matrix), rows, cols }
            };
            let backend = CpuContext;
            let input = CpuTensor { data: (0..5 * 256).map(|index| ((index as f32 * 0.017).sin() * 2.0).clamp(-2.7, 2.7)).collect(), rows: 5, cols: 256 };

            let first = build(33, 1);
            let second = build(19, 2);
            let third = build(48, 3);
            let (dual_a, dual_b) = backend.dual_linear(&input, &first, &second).unwrap();
            assert_eq!(dual_a.data, backend.linear(&input, &first).unwrap().data);
            assert_eq!(dual_b.data, backend.linear(&input, &second).unwrap().data);
            let (triple_a, triple_b, triple_c) = backend.triple_linear(&input, &first, &second, &third).unwrap();
            assert_eq!(triple_a.data, backend.linear(&input, &first).unwrap().data);
            assert_eq!(triple_b.data, backend.linear(&input, &second).unwrap().data);
            assert_eq!(triple_c.data, backend.linear(&input, &third).unwrap().data);
        }
    }

    #[test]
    fn 多行门控投影与双投影一致() {
        let backend = CpuContext;
        let input = CpuTensor { data: vec![1.0, -2.0, 0.5, 3.0, 2.0, -1.0], rows: 3, cols: 2 };
        let gate = backend.prepare_f32(&[1.0, 2.0, -1.0, 0.5, 0.25, -0.75], 3, 2).unwrap();
        let up = backend.prepare_f32(&[0.5, -1.0, 2.0, 1.0, -0.5, 0.25], 3, 2).unwrap();

        let actual = backend.gated_linear(&input, &gate, &up, &Activation::Silu).unwrap();
        let (gate_output, up_output) = backend.dual_linear(&input, &gate, &up).unwrap();
        let expected = backend.gated_activation(&gate_output, &up_output, &Activation::Silu).unwrap();

        assert_eq!(actual.rows, 3);
        assert_eq!(actual.cols, 3);
        assert_eq!(actual.data, expected.data);
    }
}
