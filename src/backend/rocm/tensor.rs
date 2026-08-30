//! ROCm 通用张量算子能力。

use super::*;

impl RocmContext {
    /// H3 block cache 等 ROCm 模型组合使用的 resident 差值；不扩展所有 backend
    /// 的公共 capability，因为 cache 判据本身仍是平台组合策略。
    pub(crate) fn subtract_resident(&self, left: &RocmTensor, right: &RocmTensor) -> Result<RocmTensor, BackendError> {
        if left.rows != right.rows || left.cols != right.cols || left.dtype != RocmTensorDType::F32 || right.dtype != RocmTensorDType::F32 {
            return Err(compute_error(format!("ROCm subtract shape/dtype 不兼容: left=[{},{}] {:?} right=[{},{}] {:?}", left.rows, left.cols, left.dtype, right.rows, right.cols, right.dtype)));
        }
        let (rows, cols) = (left.rows, left.cols);
        let elements = checked_elements(rows, cols, "ROCm subtract")?;
        let left_device = left.device.as_deref().ok_or_else(|| compute_error("ROCm subtract left 缺少 device buffer"))?;
        let right_device = right.device.as_deref().ok_or_else(|| compute_error("ROCm subtract right 缺少 device buffer"))?;
        let output = ops::hip::try_subtract_resident_f32(self.device_id, left_device, right_device, elements).map_err(compute_error)?;
        Ok(device_tensor_f32(output, rows, cols))
    }

    pub(crate) fn relative_l1_delta_partial(&self, output: &RocmTensor, input: &RocmTensor, previous: &RocmTensor) -> Result<(f64, f64), BackendError> {
        if output.rows != input.rows
            || output.rows != previous.rows
            || output.cols != input.cols
            || output.cols != previous.cols
            || output.dtype != RocmTensorDType::F32
            || input.dtype != RocmTensorDType::F32
            || previous.dtype != RocmTensorDType::F32
        {
            return Err(compute_error(format!(
                "ROCm relative L1 delta shape/dtype 不兼容: output=[{},{}] {:?} input=[{},{}] {:?} previous=[{},{}] {:?}",
                output.rows, output.cols, output.dtype, input.rows, input.cols, input.dtype, previous.rows, previous.cols, previous.dtype
            )));
        }
        let elements = checked_elements(output.rows, output.cols, "ROCm relative L1 delta")?;
        let output = output.device.as_deref().ok_or_else(|| compute_error("ROCm relative L1 output 缺少 device buffer"))?;
        let input = input.device.as_deref().ok_or_else(|| compute_error("ROCm relative L1 input 缺少 device buffer"))?;
        let previous = previous.device.as_deref().ok_or_else(|| compute_error("ROCm relative L1 previous 缺少 device buffer"))?;
        ops::hip::try_relative_l1_delta_resident_f32(self.device_id, output, input, previous, elements).map_err(compute_error)
    }

    /// DSpark 的三份 target capture 直接对应同一 `[M,3K]` Block-FP8
    /// projection。小批量直接消费 BF16，避免三次展开和两次 concat；其他形态
    /// 保持通用 linear 路径。
    pub(crate) fn block_fp8_three_segment_linear(&self, inputs: [&RocmTensor; 3], weight: &RocmWeight) -> Result<RocmTensor, BackendError> {
        let rows = inputs[0].rows;
        let columns = inputs[0].cols;
        if rows == 0 || inputs.iter().any(|input| input.rows != rows || input.cols != columns) || weight.cols != columns.checked_mul(3).ok_or_else(|| compute_error("ROCm three-segment linear columns 溢出"))? {
            return Err(compute_error(format!(
                "ROCm three-segment linear input=[{},{}]/[{},{}]/[{},{}] weight=[{},{}] 不匹配",
                inputs[0].rows, inputs[0].cols, inputs[1].rows, inputs[1].cols, inputs[2].rows, inputs[2].cols, weight.rows, weight.cols,
            )));
        }
        if rows <= 8
            && let Some(RocmQuantizedWeight::BlockFp8 { codes, scales, block_rows, block_cols, .. }) = weight.quantized()
            && let [Some(first), Some(second), Some(third)] = inputs.map(|input| input.device.as_deref())
            && inputs.iter().all(|input| input.dtype == RocmTensorDType::Bf16 && input.layout == RocmTensorLayout::RowMajor && validate_tensor_buffer(input, rows * columns).is_ok())
            && [first, second, third].iter().all(|buffer| buffer.device_id() == self.device_id)
        {
            let output = ops::hip::try_block_fp8_three_segment_bf16_resident_f32(self.device_id, first, second, third, codes, scales, rows, columns, weight.rows, *block_rows, *block_cols).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, rows, weight.rows));
        }
        let first = self.tensor_as_f32(inputs[0].clone())?;
        let second = self.tensor_as_f32(inputs[1].clone())?;
        let third = self.tensor_as_f32(inputs[2].clone())?;
        let joined = self.concat_columns(&first, &second)?;
        let joined = self.concat_columns(&joined, &third)?;
        self.linear(&joined, weight)
    }
}

#[allow(clippy::too_many_arguments)]
fn rope_with_prefix(
    context: &RocmContext,
    input: &RocmTensor,
    head_count: usize,
    rotary_dim: usize,
    layout: crate::attention::rope::RotaryLayout,
    position: usize,
    cos: &[f32],
    sin: &[f32],
    prefix: bool,
) -> Result<RocmTensor, BackendError> {
    if let Some(input_device) = input.device.as_deref() {
        let output = ops::hip::try_rope_resident_f32(context.device_id, input_device, input.rows, input.cols, head_count, rotary_dim, layout, position, cos, sin, prefix).map_err(compute_error)?;
        return Ok(device_tensor_f32(output, input.rows, input.cols));
    }
    let data = ops::hip::try_rope_f32(input.data.as_slice(), input.rows, input.cols, head_count, rotary_dim, layout, position, cos, sin, prefix).map_err(compute_error)?;
    context.upload_cpu_reference(if prefix { "host prefix RoPE" } else { "host RoPE" }, data, input.rows, input.cols)
}

impl Backend for RocmContext {
    fn linear(&self, input: &Self::Tensor, weight: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        if input.cols != weight.cols {
            return Err(compute_error(format!("ROCm linear input cols={}，weight=[{},{}]", input.cols, weight.rows, weight.cols)));
        }

        if let Some(quantized) = weight.quantized() {
            if let RocmQuantizedWeight::ConvRotInt8 { packed, scales, group_size } = quantized {
                let device = ops::hip::try_convrot_int8_matmul_f32(self.device_id, &input.data, input.device.as_deref(), packed, scales, *group_size, input.rows, input.cols, weight.rows)
                    .map_err(|error| compute_error(format!("ROCm INT8 ConvRot linear launch 失败: input=[{},{}] output_rows={} group_size={}: {error}", input.rows, input.cols, weight.rows, group_size)))?;
                return Ok(device_tensor_f32(device, input.rows, weight.rows));
            }
            if let RocmQuantizedWeight::Mxfp4 { packed, scales } = quantized {
                let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm MXFP4 linear input 缺少 device buffer"))?;
                let output = ops::hip::try_mxfp4_matmul_resident_f32(self.device_id, input_device, packed, scales, input.rows, input.cols, weight.rows)
                    .map_err(|error| compute_error(format!("ROCm MXFP4 linear 失败: input=[{},{}] output_rows={}: {error}", input.rows, input.cols, weight.rows)))?;
                return Ok(device_tensor_f32(output, input.rows, weight.rows));
            }
            if let RocmQuantizedWeight::BlockFp8 { codes, scales, block_rows, block_cols, bf16_cache } = quantized {
                let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm BlockFp8 linear input 缺少 device buffer"))?;
                if input.rows > 8 {
                    // prefill 大 GEMM:gfx11 无 FP8 计算单元,权重解码成 BF16 走
                    // WMMA GEMM;解码结果按权重缓存,只做一次(仿 router_bf16)。
                    let decoded = match bf16_cache.get() {
                        Some(buffer) => buffer.clone(),
                        None => {
                            let buffer = Arc::new(
                                ops::hip::try_block_fp8_decode_bf16(self.device_id, codes, scales, weight.rows, input.cols, *block_rows, *block_cols)
                                    .map_err(|error| compute_error(format!("ROCm BlockFp8 decode 失败: output_rows={} block=[{},{}]: {error}", weight.rows, block_rows, block_cols)))?,
                            );
                            let _ = bf16_cache.set(buffer.clone());
                            buffer
                        }
                    };
                    let output = ops::hip::try_dense_matmul_bf16_f32(self.device_id, input_device, &decoded, input.rows, input.cols, weight.rows)
                        .map_err(|error| compute_error(format!("ROCm BlockFp8 BF16 linear 失败: input=[{},{}] output_rows={}: {error}", input.rows, input.cols, weight.rows)))?;
                    return Ok(device_tensor_f32(output, input.rows, weight.rows));
                }
                let output = ops::hip::try_block_fp8_matmul_resident_f32(self.device_id, input_device, codes, scales, input.rows, input.cols, weight.rows, *block_rows, *block_cols)
                    .map_err(|error| compute_error(format!("ROCm BlockFp8 linear launch 失败: input=[{},{}] output_rows={} block=[{},{}]: {error}", input.rows, input.cols, weight.rows, block_rows, block_cols)))?;
                return Ok(device_tensor_f32(output, input.rows, weight.rows));
            }
            // GgufPacked 只保存原始 GGUF block，dense linear 复用 QK matmul
            // kernel 并在寄存器中解码 scale/min。
            let gguf_codes = match quantized {
                RocmQuantizedWeight::GgufPacked { codes, tensor_type, .. } => Some((codes, *tensor_type)),
                _ => None,
            };
            if let Some((packed, tensor_type)) = gguf_codes {
                // 设备输入(F32/BF16)直用，host 输入才上传；kernel 内联反量化，无同步。
                let output = ops::hip::try_qk_matmul_resident_f32(self.device_id, tensor_type, &input.data, input.device.as_deref(), packed, input.rows, input.cols, weight.rows)
                    .map_err(|error| compute_error(format!("ROCm GGUF QK linear launch 失败: input=[{},{}] output_rows={} type={tensor_type}: {error}", input.rows, input.cols, weight.rows)))?;
                return Ok(device_tensor_f32(output, input.rows, weight.rows));
            }
            let (bits, packed, scales, scale_dtype, group_size) = match quantized {
                RocmQuantizedWeight::W4A16 { packed, scales, scale_dtype, group_size } => (4, packed, scales, scale_dtype, group_size),
                RocmQuantizedWeight::W8A16 { packed, scales, scale_dtype, group_size } => (8, packed, scales, scale_dtype, group_size),
                RocmQuantizedWeight::BlockFp8 { .. } | RocmQuantizedWeight::Mxfp4 { .. } | RocmQuantizedWeight::ConvRotInt8 { .. } | RocmQuantizedWeight::GgufPacked { .. } => unreachable!(),
            };
            let scale_dtype = match scale_dtype {
                ScaleDType::Bf16 => 0,
                ScaleDType::F16 => 1,
                ScaleDType::F32 => 2,
            };
            if input.rows != 0 && ops::hip::options().debug_finite {
                if let Some(input_device) = input.device.as_deref() {
                    let probe = if input.dtype == RocmTensorDType::Bf16 {
                        ops::hip::try_validate_finite_resident_range_bf16(self.device_id, input_device, (input.rows - 1) * input.cols, input.cols)
                    } else {
                        ops::hip::try_validate_finite_resident_range_f32(self.device_id, input_device, (input.rows - 1) * input.cols, input.cols)
                    };
                    probe.map_err(|error| compute_error(format!("ROCm CT linear 输入尾行包含非有限值: bits={bits} input=[{},{}] output_rows={} group_size={}: {error}", input.rows, input.cols, weight.rows, group_size)))?;
                }
            }
            let device = ops::hip::try_ct_quantized_matmul_bf16(self.device_id, bits, &input.data, input.device.as_deref(), packed, scales, scale_dtype, *group_size, input.rows, input.cols, weight.rows)
                .map_err(|error| compute_error(format!("ROCm CT linear launch 失败: bits={bits} input=[{},{}] output_rows={} group_size={}: {error}", input.rows, input.cols, weight.rows, group_size)))?;
            if input.rows != 0 && ops::hip::options().debug_finite {
                ops::hip::try_validate_finite_resident_range_f32(self.device_id, &device, (input.rows - 1) * weight.rows, weight.rows)
                    .map_err(|error| compute_error(format!("ROCm CT linear 尾行包含非有限值: bits={bits} input=[{},{}] output_rows={} group_size={}: {error}", input.rows, input.cols, weight.rows, group_size)))?;
            }
            return Ok(device_tensor_f32(device, input.rows, weight.rows));
        }
        if weight.resident_bf16() {
            let input = self.tensor_on_device(input.clone())?;
            let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm BF16 linear input 缺少 device buffer"))?;
            if input.rows != 0 && ops::hip::options().debug_finite {
                let probe = if input.dtype == RocmTensorDType::Bf16 {
                    ops::hip::try_validate_finite_resident_range_bf16(self.device_id, input_device, (input.rows - 1) * input.cols, input.cols)
                } else {
                    ops::hip::try_validate_finite_resident_range_f32(self.device_id, input_device, (input.rows - 1) * input.cols, input.cols)
                };
                probe.map_err(|error| compute_error(format!("ROCm BF16 linear 输入包含非有限值: input=[{},{}] output_rows={}: {error}", input.rows, input.cols, weight.rows)))?;
            }
            let weight_device = weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm BF16 linear weight 缺少 device buffer"))?;
            // Decode 小批次保持与单路 GEMV 完全相同的 F32/FMA 归约树；长 prefill 才切换 WMMA。
            let output = if input.rows <= 64 {
                ops::hip::try_bf16_gemv_resident_f32(self.device_id, input_device, weight_device, input.rows, input.cols, weight.rows, false)
            } else {
                ops::hip::try_dense_matmul_bf16_f32(self.device_id, input_device, weight_device, input.rows, input.cols, weight.rows)
            }
            .map_err(compute_error)?;
            if input.rows != 0 && ops::hip::options().debug_finite {
                ops::hip::try_validate_finite_resident_f32(self.device_id, &output, weight.rows)
                    .map_err(|error| compute_error(format!("ROCm BF16 linear 输出包含非有限值: input=[{},{}] output_rows={}: {error}", input.rows, input.cols, weight.rows)))?;
            }
            return Ok(device_tensor_f32(output, input.rows, weight.rows));
        }
        if let (Some(input_device), Some(weight_device)) = (input.device.as_deref(), weight.resident().map(Arc::as_ref)) {
            let resident_bf16 = weight.resident_bf16();
            let output = if resident_bf16 && input.rows <= 64 {
                ops::hip::try_bf16_gemv_resident_f32(self.device_id, input_device, weight_device, input.rows, input.cols, weight.rows, false)
            } else if resident_bf16 {
                ops::hip::try_dense_matmul_bf16_f32(self.device_id, input_device, weight_device, input.rows, input.cols, weight.rows)
            } else if input.rows <= 64 {
                ops::hip::try_f32_gemv_resident_f32(self.device_id, input_device, weight_device, input.rows, input.cols, weight.rows)
            } else {
                ops::hip::try_sgemm_resident_f32(self.device_id, input_device, weight_device, input.rows, input.cols, weight.rows)
            }
            .map_err(compute_error)?;
            return Ok(device_tensor_f32(output, input.rows, weight.rows));
        }
        self.require_cpu_reference_fallback("host linear")?;
        let mut output = vec![0.0; checked_elements(input.rows, weight.rows, "ROCm host linear")?];
        let input_data = tensor_data(input)?;
        if let Some(weight) = weight.gguf() {
            let weight_bytes = weight.bytes().map_err(compute_error)?;
            if matches!(weight.tensor_type.0, 12 | 13) && input.rows > 1 {
                ops::hip::try_qk_matmul_f32(self.device_id, weight.tensor_type.0, &input_data, weight_bytes, input.rows, weight.columns, weight.rows, &mut output).map_err(compute_error)?;
                return self.upload_cpu_reference("host GGUF linear", output, input.rows, weight.rows);
            }
            for (input, output) in input_data.chunks_exact(input.cols).zip(output.chunks_exact_mut(weight.rows)) {
                ggml_quant::matvec(weight.tensor_type.0, weight_bytes, weight.rows, weight.columns, input, output).map_err(compute_error)?;
            }
            return self.upload_cpu_reference("host GGUF linear", output, input.rows, weight.rows);
        }

        ops::hip::try_sgemm_f32(self.device_id, &input_data, weight.data(), input.rows, input.cols, weight.rows, &mut output).map_err(compute_error)?;

        self.upload_cpu_reference("host dense linear", output, input.rows, weight.rows)
    }

    fn grouped_linear_columns(&self, input: &Self::Tensor, weights: &[Self::Weight]) -> Result<Self::Tensor, BackendError> {
        let fallback = || -> Result<Self::Tensor, BackendError> {
            if weights.is_empty() || !input.cols.is_multiple_of(weights.len()) {
                return Err(compute_error(format!("ROCm grouped linear input=[{},{}] groups={} 不兼容", input.rows, input.cols, weights.len())));
            }
            if weights.len() == 1 {
                return self.linear(input, &weights[0]);
            }
            let group_columns = input.cols / weights.len();
            let (head, mut tail) = self.split_columns(input, group_columns)?;
            let mut projected = self.linear(&head, &weights[0])?;
            for weight in &weights[1..weights.len() - 1] {
                let (head, rest) = self.split_columns(&tail, group_columns)?;
                tail = rest;
                let output = self.linear(&head, weight)?;
                projected = self.concat_columns(&projected, &output)?;
            }
            let output = self.linear(&tail, weights.last().expect("ROCm grouped linear 非空已检查"))?;
            self.concat_columns(&projected, &output)
        };
        if input.rows == 0 || input.rows > 8 || weights.is_empty() || !input.cols.is_multiple_of(weights.len()) {
            return fallback();
        }
        let group_columns = input.cols / weights.len();
        let rows_per_group = weights[0].rows;
        let mut block_shape = None;
        let mut grouped = Vec::with_capacity(weights.len());
        for weight in weights {
            let Some(RocmQuantizedWeight::BlockFp8 { codes, scales, block_rows, block_cols, .. }) = weight.quantized() else {
                return fallback();
            };
            if weight.rows != rows_per_group || weight.cols != group_columns || block_shape.is_some_and(|shape| shape != (*block_rows, *block_cols)) {
                return fallback();
            }
            block_shape = Some((*block_rows, *block_cols));
            grouped.push((codes.as_ref(), scales.as_ref()));
        }
        let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm grouped BlockFP8 input 缺少 device buffer"))?;
        let (block_rows, block_cols) = block_shape.expect("ROCm grouped BlockFP8 非空已检查");
        let output = ops::hip::try_block_fp8_grouped_columns_gemv_resident_f32(self.device_id, input_device, &grouped, input.rows, rows_per_group, group_columns, block_rows, block_cols).map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, weights.len() * rows_per_group))
    }

    fn linear_add(&self, input: &Self::Tensor, weight: &Self::Weight, residual: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        if input.rows == residual.rows
            && input.cols == weight.cols
            && weight.rows == residual.cols
            && (input.rows == 1 || input.rows >= 16)
            && let (Some(input_device), Some(residual_device)) = (input.device.as_deref(), residual.device.as_deref())
            // 单行也可能是 prefill 尾块，其 residual 保持 BF16。只有 F32
            // residual 才是 decode，不能单凭 rows=1 改变 prefill epilogue。
            && (input.rows >= 16 || residual_device.bytes() == input.rows.checked_mul(residual.cols).and_then(|elements| elements.checked_mul(4)).unwrap_or(0))
            && let Some(RocmQuantizedWeight::W8A16 { packed, scales, scale_dtype: ScaleDType::Bf16, group_size: 128 }) = weight.quantized()
        {
            let output = ops::hip::try_ct_quantized_matmul_bf16_add(self.device_id, 8, &input.data, Some(input_device), packed, scales, 0, 128, input.rows, input.cols, weight.rows, residual_device)
                .map_err(|error| compute_error(format!("ROCm W8 residual linear launch 失败: input=[{},{}] output_rows={}: {error}", input.rows, input.cols, weight.rows)))?;
            return Ok(device_tensor_f32(output, input.rows, weight.rows));
        }
        let output = self.linear(input, weight)?;
        self.add(residual, &output)
    }

    fn rmsnorm_linear(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, linear_weight: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        if linear_weight.quantized().is_some() {
            if let Some(input_device) = input.device.as_deref() {
                let weight_device = norm_weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm RMSNorm weight 缺少 resident buffer"))?;
                let output = ops::hip::try_rmsnorm_resident_weight_to_bf16(self.device_id, input_device, weight_device, input.rows, input.cols, eps, false).map_err(compute_error)?;
                let normalized = device_tensor_bf16(output, input.rows, input.cols);
                return self.linear(&normalized, linear_weight);
            }
        }
        let normalized = self.rmsnorm(input, norm_weight, eps)?;
        self.linear(&normalized, linear_weight)
    }

    fn rmsnorm_dual_linear(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, first: &Self::Weight, second: &Self::Weight) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        let dual_ct = match (first.quantized(), second.quantized()) {
            (Some(RocmQuantizedWeight::W4A16 { group_size: first_group_size, .. }), Some(RocmQuantizedWeight::W4A16 { group_size: second_group_size, .. })) => *first_group_size == 128 && *second_group_size == 128,
            (Some(RocmQuantizedWeight::W8A16 { .. }), Some(RocmQuantizedWeight::W8A16 { .. })) => true,
            _ => false,
        };
        let options = ops::hip::options();
        if input.rows == 1 && dual_ct && options.dual_w8 && !options.debug_finite && !options.debug_decode_layer_finite {
            if let Some(input_device) = input.device.as_deref() {
                let weight_device = norm_weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm dual RMSNorm weight 缺少 resident buffer"))?;
                let output = ops::hip::try_rmsnorm_resident_weight_to_bf16(self.device_id, input_device, weight_device, input.rows, input.cols, eps, false).map_err(compute_error)?;
                return self.dual_linear(&device_tensor_bf16(output, input.rows, input.cols), first, second);
            }
        }
        let normalized = self.rmsnorm(input, norm_weight, eps)?;
        self.dual_linear(&normalized, first, second)
    }

    fn rmsnorm(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        if let Some(input_device) = input.device.as_deref() {
            let debug_decode = input.rows == 1 && ops::hip::options().debug_decode_layer_finite;
            let debug_finite = ops::hip::options().debug_finite || debug_decode;
            let debug_sequence = debug_decode.then(|| {
                std::thread_local! {
                    static DEBUG_RMSNORM_SEQUENCE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
                }
                DEBUG_RMSNORM_SEQUENCE.with(|sequence| {
                    let next = sequence.get() + 1;
                    sequence.set(next);
                    next
                })
            });
            if input.rows != 0 && debug_finite {
                let probe = if input.dtype == RocmTensorDType::Bf16 {
                    ops::hip::try_validate_finite_resident_range_bf16(self.device_id, input_device, (input.rows - 1) * input.cols, input.cols)
                } else {
                    ops::hip::try_validate_finite_resident_range_f32(self.device_id, input_device, (input.rows - 1) * input.cols, input.cols)
                };
                probe.map_err(|error| compute_error(format!("ROCm RMSNorm 输入包含非有限值: sequence={debug_sequence:?} rows={} cols={}: {error}", input.rows, input.cols)))?;
                if let Some(sequence) = debug_sequence {
                    eprintln!("[decode-layer-finite] device={} op=rmsnorm sequence={sequence} stage=input", self.device_id);
                }
            }
            let weight_device = weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm RMSNorm weight 缺少 resident buffer"))?;
            let output = ops::hip::try_rmsnorm_resident_weight_to_f32(self.device_id, input_device, weight_device, input.rows, input.cols, eps, false).map_err(compute_error)?;
            if input.rows != 0 && debug_finite {
                ops::hip::try_validate_finite_resident_range_f32(self.device_id, &output, (input.rows - 1) * input.cols, input.cols)
                    .map_err(|error| compute_error(format!("ROCm RMSNorm 输出包含非有限值: sequence={debug_sequence:?} rows={} cols={}: {error}", input.rows, input.cols)))?;
                if let Some(sequence) = debug_sequence {
                    eprintln!("[decode-layer-finite] device={} op=rmsnorm sequence={sequence} stage=output", self.device_id);
                }
            }
            return Ok(device_tensor_f32(output, input.rows, input.cols));
        }
        let mut output = vec![0.0; input.data.len()];
        ops::hip::try_rmsnorm_f32(&input.data, weight.data(), eps, &mut output).map_err(compute_error)?;
        self.upload_cpu_reference("host RMSNorm", output, input.rows, input.cols)
    }

    fn rmsnorm_quantized(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        if input.rows > 1
            && let Some(input_device) = input.device.as_deref()
        {
            let weight_device = weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm quantized RMSNorm weight 缺少 resident buffer"))?;
            let output = ops::hip::try_rmsnorm_resident_weight_to_bf16(self.device_id, input_device, weight_device, input.rows, input.cols, eps, false).map_err(compute_error)?;
            return Ok(device_tensor_bf16(output, input.rows, input.cols));
        }
        self.rmsnorm(input, weight, eps)
    }

    fn rmsnorm_quantized_pair(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Option<(Self::Tensor, Self::Tensor)>, BackendError> {
        let Some(input_device) = input.device.as_deref() else {
            return Ok(None);
        };
        let weight_device = weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm dual RMSNorm weight 缺少 resident buffer"))?;
        let (precise, quantized) = ops::hip::try_rmsnorm_resident_weight_to_f32_bf16(self.device_id, input_device, weight_device, input.rows, input.cols, eps, false).map_err(compute_error)?;
        Ok(Some((device_tensor_f32(precise, input.rows, input.cols), device_tensor_bf16(quantized, input.rows, input.cols))))
    }

    fn gemma_rmsnorm(&self, input: &Self::Tensor, weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        if let Some(input_device) = input.device.as_deref() {
            let weight_device = weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm GemmaRMSNorm weight 缺少 resident buffer"))?;
            let output = ops::hip::try_rmsnorm_resident_weight_to_f32(self.device_id, input_device, weight_device, input.rows, input.cols, eps, true).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, input.rows, input.cols));
        }
        let mut output = vec![0.0; input.data.len()];
        ops::hip::try_gemma_rmsnorm_f32(&input.data, weight.data(), eps, &mut output).map_err(compute_error)?;
        self.upload_cpu_reference("host GemmaRMSNorm", output, input.rows, input.cols)
    }

    fn layernorm_bias(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        if let Some(input_device) = input.device.as_deref() {
            let weight_device = weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm LayerNorm weight 缺少 resident buffer"))?;
            let bias_device = bias.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm LayerNorm bias 缺少 resident buffer"))?;
            let output = ops::hip::try_layernorm_bias_resident_f32(self.device_id, input_device, weight_device, bias_device, input.rows, input.cols, eps).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, input.rows, input.cols));
        }
        let mut output = vec![0.0; input.data.len()];
        ops::hip::try_layernorm_bias_f32(&input.data, weight.data(), bias.data(), input.rows, input.cols, eps, &mut output).map_err(compute_error)?;
        self.upload_cpu_reference("host LayerNorm", output, input.rows, input.cols)
    }

    fn split_gated_activation(&self, input: Self::Tensor, left_columns: usize, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        let expected_columns = left_columns.checked_mul(2).ok_or_else(|| compute_error("ROCm packed gated activation 列数溢出"))?;
        if input.cols != expected_columns {
            return Err(compute_error(format!("ROCm packed gated activation cols={}，期望 {expected_columns}", input.cols)));
        }
        if let Some(input_device) = input.device {
            let input_device = Arc::try_unwrap(input_device).map_err(|_| compute_error("ROCm packed gated activation 输入仍被共享，无法转移所有权"))?;
            // 下投影只消费 BF16；在激活核写回时直接收缩，避免完整 F32 中间张量再读回转换。
            let output = ops::hip::try_split_gated_activation_owned_bf16(self.device_id, input_device, None, input.rows, left_columns, activation).map_err(compute_error)?;
            return Ok(device_tensor_bf16(output, input.rows, left_columns));
        }
        let (gate, up) = ops::hip::try_split_columns_f32(&input.data, input.rows, input.cols, left_columns).map_err(compute_error)?;
        let mut output = vec![0.0; gate.len()];
        ops::hip::try_gated_activation_f32(&gate, &up, input.rows, left_columns, activation, &mut output).map_err(compute_error)?;
        self.upload_cpu_reference("host packed gated activation", output, input.rows, left_columns)
    }

    fn split_columns(&self, input: &Self::Tensor, left_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        if let Some(input_device) = input.device.as_deref() {
            let right_columns = input.cols.checked_sub(left_columns).ok_or_else(|| compute_error(format!("split 左侧列 {left_columns} 超过总列数 {}", input.cols)))?;
            if input.rows == 1 && input.dtype == RocmTensorDType::F32 && left_columns != 0 && right_columns != 0 {
                let input_bytes = input.cols.checked_mul(RocmTensorDType::F32.element_bytes()).ok_or_else(|| compute_error("ROCm 单行 split 大小溢出"))?;
                if input_device.bytes() == input_bytes {
                    // 单行左右列在内存中连续，直接共享来源 allocation；decode 每层无需
                    // 为 kv_a 的 latent/k_rope 拆分再提交一次纯复制 kernel。
                    let owner = input.device.as_ref().expect("上方已检查 device").clone();
                    let left_bytes = left_columns * std::mem::size_of::<f32>();
                    let right_bytes = right_columns * std::mem::size_of::<f32>();
                    let left = ops::hip::DeviceBuffer::view(owner.clone(), 0, left_bytes).map_err(compute_error)?;
                    let right = ops::hip::DeviceBuffer::view(owner, left_bytes, right_bytes).map_err(compute_error)?;
                    return Ok((device_tensor_f32(left, 1, left_columns), device_tensor_f32(right, 1, right_columns)));
                }
            }
            let (left, right) = ops::hip::try_split_columns_resident_f32(self.device_id, input_device, input.rows, input.cols, left_columns).map_err(compute_error)?;
            return Ok((device_tensor_f32(left, input.rows, left_columns), device_tensor_f32(right, input.rows, right_columns)));
        }
        let (left, right) = ops::hip::try_split_columns_f32(&input.data, input.rows, input.cols, left_columns).map_err(compute_error)?;
        let right_columns = input.cols.checked_sub(left_columns).ok_or_else(|| compute_error(format!("split 左侧列 {left_columns} 超过总列数 {}", input.cols)))?;
        self.require_cpu_reference_fallback("host split columns")?;
        Ok((self.tensor_from_f32(left, input.rows, left_columns).map_err(compute_error)?, self.tensor_from_f32(right, input.rows, right_columns).map_err(compute_error)?))
    }

    fn split_interleaved_columns(&self, input: &Self::Tensor, block_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        if let Some(device) = input.device.as_deref() {
            let (left, right) = ops::hip::try_split_interleaved_columns_resident_f32(self.device_id, device, input.rows, input.cols, block_columns).map_err(compute_error)?;
            return Ok((device_tensor_f32(left, input.rows, input.cols / 2), device_tensor_f32(right, input.rows, input.cols / 2)));
        }
        let (left, right) = ops::hip::try_split_interleaved_columns_f32(&input.data, input.rows, input.cols, block_columns).map_err(compute_error)?;
        self.require_cpu_reference_fallback("host split interleaved columns")?;
        Ok((self.tensor_from_f32(left, input.rows, input.cols / 2).map_err(compute_error)?, self.tensor_from_f32(right, input.rows, input.cols / 2).map_err(compute_error)?))
    }

    fn concat_columns(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        if left.rows != right.rows || left.dtype != right.dtype {
            return Err(compute_error(format!("ROCm concat shape/dtype [{},{},{:?}] 与 [{},{},{:?}] 不一致", left.rows, left.cols, left.dtype, right.rows, right.cols, right.dtype)));
        }
        let columns = left.cols.checked_add(right.cols).ok_or_else(|| compute_error("ROCm concat 列数溢出".to_owned()))?;
        if let (Some(left_device), Some(right_device)) = (left.device.as_deref(), right.device.as_deref()) {
            let output = ops::hip::try_concat_columns_resident_f32(self.device_id, left_device, right_device, left.rows, left.cols, right.cols).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, left.rows, columns));
        }
        let left_data = tensor_data(left)?;
        let right_data = tensor_data(right)?;
        let output = ops::hip::try_concat_columns_f32(&left_data, &right_data, left.rows, left.cols, right.cols).map_err(compute_error)?;
        self.upload_cpu_reference("host concat columns", output, left.rows, columns)
    }

    fn rope(&self, input: &Self::Tensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<Self::Tensor, BackendError> {
        rope_with_prefix(self, input, head_count, rotary_dim, layout, position, cos, sin, false)
    }

    fn rope_prefix(&self, input: &Self::Tensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<Self::Tensor, BackendError> {
        rope_with_prefix(self, input, head_count, rotary_dim, layout, position, cos, sin, true)
    }

    fn rope_pair_prefix(
        &self,
        query: Self::Tensor,
        key: Self::Tensor,
        head_count: usize,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        position: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        if query.rows != key.rows || query.cols != key.cols {
            return Err(compute_error("ROCm paired RoPE Q/K shape 不一致"));
        }
        let rows = query.rows;
        let cols = query.cols;
        let query = self.tensor_as_f32(query)?.device.ok_or_else(|| compute_error("ROCm paired RoPE query 缺少 device buffer"))?;
        let key = self.tensor_as_f32(key)?.device.ok_or_else(|| compute_error("ROCm paired RoPE key 缺少 device buffer"))?;
        let query = std::sync::Arc::try_unwrap(query).map_err(|_| compute_error("ROCm paired RoPE query device buffer 仍被共享"))?;
        let key = std::sync::Arc::try_unwrap(key).map_err(|_| compute_error("ROCm paired RoPE key device buffer 仍被共享"))?;
        let (query, key) = ops::hip::try_rope_pair_resident_f32(self.device_id, query, key, rows, cols, head_count, rotary_dim, layout, position, cos, sin, true).map_err(compute_error)?;
        Ok((device_tensor_f32(query, rows, cols), device_tensor_f32(key, rows, cols)))
    }

    fn add(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        if left.rows != right.rows || left.cols != right.cols {
            return Err(compute_error(format!("ROCm add shape [{},{}] 与 [{},{}] 不一致", left.rows, left.cols, right.rows, right.cols)));
        }
        if let (Some(left_device), Some(right_device)) = (left.device.as_deref(), right.device.as_deref()) {
            let elements = checked_elements(left.rows, left.cols, "ROCm add")?;
            let output = match (left.dtype, right.dtype) {
                (RocmTensorDType::Bf16, RocmTensorDType::F32) => ops::hip::try_add_resident_bf16_f32(self.device_id, left_device, right_device, elements).map_err(compute_error)?,
                (RocmTensorDType::F32, RocmTensorDType::F32) => ops::hip::try_add_resident_f32(self.device_id, left_device, right_device, elements, 1.0).map_err(compute_error)?,
                (left_dtype, right_dtype) => return Err(compute_error(format!("ROCm add dtype={left_dtype:?}/{right_dtype:?}，期望 BF16+F32 或 F32+F32"))),
            };
            let debug_decode = left.rows == 1 && ops::hip::options().debug_decode_layer_finite;
            let debug_finite = ops::hip::options().debug_finite || debug_decode;
            if left.rows != 0 && debug_finite {
                std::thread_local! {
                    static DEBUG_ADD_SEQUENCE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
                }
                let debug_add_sequence = DEBUG_ADD_SEQUENCE.with(|sequence| {
                    let next = sequence.get() + 1;
                    sequence.set(next);
                    next
                });
                let phase = if debug_add_sequence % 2 == 1 { "attention" } else { "ffn" };
                // Decode microbatch 必须检查全部行；只扫尾行会漏掉非尾 session 的显存污染。
                let scan_rows = left.rows.min(64);
                let scan_offset = (left.rows - scan_rows) * left.cols;
                let scan_elements = scan_rows * left.cols;
                let left_probe = if left.dtype == RocmTensorDType::Bf16 {
                    ops::hip::try_validate_finite_resident_range_bf16(self.device_id, left_device, scan_offset, scan_elements)
                } else {
                    ops::hip::try_validate_finite_resident_range_f32(self.device_id, left_device, scan_offset, scan_elements)
                };
                left_probe.map_err(|error| compute_error(format!("ROCm add left 包含非有限值: sequence={debug_add_sequence} phase={phase} rows={} cols={}: {error}", left.rows, left.cols)))?;
                ops::hip::try_validate_finite_resident_range_f32(self.device_id, right_device, scan_offset, scan_elements)
                    .map_err(|error| compute_error(format!("ROCm add right 包含非有限值: sequence={debug_add_sequence} phase={phase} rows={} cols={}: {error}", right.rows, right.cols)))?;
                ops::hip::try_validate_finite_resident_range_f32(self.device_id, &output, scan_offset, scan_elements)
                    .map_err(|error| compute_error(format!("ROCm add output 包含非有限值: sequence={debug_add_sequence} phase={phase} rows={} cols={}: {error}", left.rows, left.cols)))?;
                if debug_decode {
                    eprintln!("[decode-layer-finite] device={} op=add sequence={debug_add_sequence} phase={phase} stage=output", self.device_id);
                }
            }
            return Ok(device_tensor_f32(output, left.rows, left.cols));
        }
        let left_data = tensor_data(left)?;
        let right_data = tensor_data(right)?;
        let mut output = vec![0.0; left_data.len()];
        ops::hip::try_add_f32(&left_data, &right_data, &mut output).map_err(compute_error)?;
        self.upload_cpu_reference("host add", output, left.rows, left.cols)
    }

    fn add_scaled(&self, left: &Self::Tensor, right: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError> {
        if left.rows != right.rows || left.cols != right.cols {
            return Err(compute_error(format!("ROCm add_scaled shape [{},{}] 与 [{},{}] 不一致", left.rows, left.cols, right.rows, right.cols)));
        }
        if let (Some(left_device), Some(right_device)) = (left.device.as_deref(), right.device.as_deref()) {
            let output = ops::hip::try_add_resident_f32(self.device_id, left_device, right_device, checked_elements(left.rows, left.cols, "ROCm add_scaled")?, scale).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, left.rows, left.cols));
        }
        let left_data = tensor_data(left)?;
        let right_data = tensor_data(right)?;
        let mut output = vec![0.0; left_data.len()];
        ops::hip::try_add_scaled_f32(&left_data, &right_data, scale, &mut output).map_err(compute_error)?;
        self.upload_cpu_reference("host add_scaled", output, left.rows, left.cols)
    }

    fn sigmoid_gate(&self, input: &Self::Tensor, gate: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        if let (Some(input_device), Some(gate_device)) = (input.device.as_deref(), gate.device.as_deref()) {
            let output = ops::hip::try_sigmoid_gate_resident_f32(self.device_id, input_device, gate_device, input.rows, input.cols).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, input.rows, input.cols));
        }
        let input_data = tensor_data(input)?;
        let gate_data = tensor_data(gate)?;
        let mut output = vec![0.0; input_data.len()];
        ops::hip::try_sigmoid_gate_f32(&input_data, &gate_data, input.rows, input.cols, &mut output).map_err(compute_error)?;
        self.upload_cpu_reference("host sigmoid gate", output, input.rows, input.cols)
    }

    fn argmax(&self, input: &Self::Tensor) -> Result<u32, BackendError> {
        let input = self.tensor_as_f32(input.clone())?;
        let elements = checked_elements(input.rows, input.cols, "ROCm argmax")?;
        let device = input.device.as_deref().ok_or_else(|| compute_error("ROCm argmax 缺少 device buffer"))?;
        ops::hip::try_argmax_excluding_resident_f32(self.device_id, device, elements, &[]).map_err(compute_error)
    }

    fn sample_top_p(&self, input: &Self::Tensor, temperature: f32, top_p: f32, random: f32) -> Result<u32, BackendError> {
        self.sample_top_p_excluding(input, temperature, top_p, random, &[])
    }

    fn sample_top_p_excluding(&self, input: &Self::Tensor, temperature: f32, top_p: f32, random: f32, excluded: &[u32]) -> Result<u32, BackendError> {
        let input = self.tensor_as_f32(input.clone())?;
        let device = input.device.as_deref().ok_or_else(|| compute_error("ROCm top-p 缺少 device buffer"))?;
        let sampling = [crate::backend::TokenSampling { temperature, top_p, random }];
        let mut output = ops::hip::try_sample_top_p_rows_excluding_resident_f32(self.device_id, device, 1, input.cols, &sampling, excluded).map_err(compute_error)?;
        Ok(output.remove(0))
    }

    fn gated_activation(&self, gate: &Self::Tensor, up: &Self::Tensor, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        if let (Some(gate_device), Some(up_device)) = (gate.device.as_deref(), up.device.as_deref()) {
            let elements = checked_elements(gate.rows, gate.cols, "ROCm gated activation")?;
            let output = ops::hip::try_gated_activation_resident_f32(self.device_id, gate_device, up_device, elements, activation).map_err(compute_error)?;
            if ops::hip::options().kernel_sync {
                ops::hip::synchronize_device(self.device_id, "ROCm gated activation synchronize").map_err(compute_error)?;
            }
            if gate.rows != 0 && ops::hip::options().debug_finite {
                ops::hip::try_validate_finite_resident_range_f32(self.device_id, &output, (gate.rows - 1) * gate.cols, gate.cols)
                    .map_err(|error| compute_error(format!("ROCm gated activation 输出包含非有限值: elements={elements}: {error}")))?;
            }
            return Ok(device_tensor_f32(output, gate.rows, gate.cols));
        }
        let gate_data = tensor_data(gate)?;
        let up_data = tensor_data(up)?;
        let mut output = vec![0.0; gate_data.len()];
        ops::hip::try_gated_activation_f32(&gate_data, &up_data, gate.rows, gate.cols, activation, &mut output).map_err(compute_error)?;
        self.upload_cpu_reference("host gated activation", output, gate.rows, gate.cols)
    }

    fn select_row(&self, input: &Self::Tensor, row: usize) -> Result<Self::Tensor, BackendError> {
        if row >= input.rows {
            return Err(compute_error(format!("ROCm select_row 行 {row} 越界，rows={}", input.rows)));
        }
        if let Some(device) = input.device.as_deref() {
            let row = u32::try_from(row).map_err(|_| compute_error("ROCm select_row index 超过 u32"))?;
            let output = ops::hip::try_select_rows_resident_f32(self.device_id, device, input.rows, input.cols, &[row]).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, 1, input.cols));
        }
        let data = ops::hip::try_select_row_f32(&input.data, input.rows, input.cols, row).map_err(compute_error)?;
        self.upload_cpu_reference("host select row", data, 1, input.cols)
    }

    fn select_rows(&self, input: &Self::Tensor, rows: &[u32]) -> Result<Self::Tensor, BackendError> {
        if rows.is_empty() {
            return Err(compute_error("ROCm select_rows 行列表为空".to_owned()));
        }
        if let Some(device) = input.device.as_deref() {
            let output = ops::hip::try_select_rows_resident_f32(self.device_id, device, input.rows, input.cols, rows).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, rows.len(), input.cols));
        }
        let output = ops::hip::try_select_rows_f32(&input.data, input.rows, input.cols, rows).map_err(compute_error)?;
        self.upload_cpu_reference("host select rows", output, rows.len(), input.cols)
    }

    fn argmax_excluding(&self, input: &Self::Tensor, excluded: &[u32]) -> Result<u32, BackendError> {
        let input = self.tensor_as_f32(input.clone())?;
        let elements = checked_elements(input.rows, input.cols, "ROCm argmax_excluding")?;
        let device = input.device.as_deref().ok_or_else(|| compute_error("ROCm argmax_excluding 缺少 device buffer"))?;
        ops::hip::try_argmax_excluding_resident_f32(self.device_id, device, elements, excluded).map_err(compute_error)
    }

    fn linear_sigmoid_gate(&self, input: &Self::Tensor, weight: &Self::Weight, value: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        let gate = self.linear(input, weight)?;
        self.sigmoid_gate(value, &gate)
    }

    fn dual_linear(&self, input: &Self::Tensor, first: &Self::Weight, second: &Self::Weight) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        if input.rows > 0
            && input.rows <= 8
            && let (
                Some(RocmQuantizedWeight::BlockFp8 { codes: first_codes, scales: first_scales, block_rows: first_block_rows, block_cols: first_block_cols, .. }),
                Some(RocmQuantizedWeight::BlockFp8 { codes: second_codes, scales: second_scales, block_rows: second_block_rows, block_cols: second_block_cols, .. }),
            ) = (first.quantized(), second.quantized())
        {
            if input.cols != first.cols || input.cols != second.cols || first_block_rows != second_block_rows || first_block_cols != second_block_cols {
                return Err(compute_error(format!(
                    "ROCm dual BlockFp8 input cols={}，weights=[{},{}]/[{},{}] block=[{},{}]/[{},{}]",
                    input.cols, first.rows, first.cols, second.rows, second.cols, first_block_rows, first_block_cols, second_block_rows, second_block_cols
                )));
            }
            let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm dual BlockFp8 input 缺少 device buffer"))?;
            let (first_output, second_output) =
                ops::hip::try_block_fp8_dual_gemv_resident_f32(self.device_id, input_device, input.rows, first_codes, first_scales, first.rows, second_codes, second_scales, second.rows, input.cols, *first_block_rows, *first_block_cols)
                    .map_err(compute_error)?;
            return Ok((device_tensor_f32(first_output, input.rows, first.rows), device_tensor_f32(second_output, input.rows, second.rows)));
        }
        if input.rows == 1 && first.quantized().is_none() && second.quantized().is_none() && first.resident_bf16() && second.resident_bf16() {
            if input.cols != first.cols || input.cols != second.cols {
                return Err(compute_error(format!("ROCm dual BF16 input cols={}，weights=[{},{}]/[{},{}]", input.cols, first.rows, first.cols, second.rows, second.cols)));
            }
            let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm dual BF16 input 缺少 device buffer"))?;
            let first_weight = first.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm dual BF16 first weight 缺少 resident buffer"))?;
            let second_weight = second.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm dual BF16 second weight 缺少 resident buffer"))?;
            let (first_output, second_output) = ops::hip::try_bf16_dual_gemv_resident_f32(self.device_id, input_device, first_weight, first.rows, second_weight, second.rows, input.cols).map_err(compute_error)?;
            return Ok((device_tensor_f32(first_output, 1, first.rows), device_tensor_f32(second_output, 1, second.rows)));
        }
        if input.rows == 1 && first.quantized().is_none() && second.quantized().is_none() && !first.resident_bf16() && !second.resident_bf16() {
            if input.cols != first.cols || input.cols != second.cols {
                return Err(compute_error(format!("ROCm dual F32 input cols={}，weights=[{},{}]/[{},{}]", input.cols, first.rows, first.cols, second.rows, second.cols)));
            }
            if let (Some(input_device), Some(first_weight), Some(second_weight)) = (input.device.as_deref(), first.resident().map(Arc::as_ref), second.resident().map(Arc::as_ref))
                && first_weight.bytes() == first.rows.checked_mul(first.cols).and_then(|n| n.checked_mul(4)).unwrap_or(0)
                && second_weight.bytes() == second.rows.checked_mul(second.cols).and_then(|n| n.checked_mul(4)).unwrap_or(0)
            {
                let (first_output, second_output) = ops::hip::try_f32_dual_gemv_resident_f32(self.device_id, input_device, first_weight, first.rows, second_weight, second.rows, input.cols).map_err(compute_error)?;
                return Ok((device_tensor_f32(first_output, 1, first.rows), device_tensor_f32(second_output, 1, second.rows)));
            }
        }
        if input.rows > 0 && input.rows <= 8 && ops::hip::options().dual_w8 {
            let dual = match (first.quantized(), second.quantized()) {
                (
                    Some(RocmQuantizedWeight::W4A16 { packed: first_packed, scales: first_scales, scale_dtype: first_scale_dtype, group_size: first_group_size }),
                    Some(RocmQuantizedWeight::W4A16 { packed: second_packed, scales: second_scales, scale_dtype: second_scale_dtype, group_size: second_group_size }),
                ) if *first_group_size == 128 && *second_group_size == 128 => Some((4, first_packed, first_scales, first_scale_dtype, first_group_size, second_packed, second_scales, second_scale_dtype, second_group_size)),
                (
                    Some(RocmQuantizedWeight::W8A16 { packed: first_packed, scales: first_scales, scale_dtype: first_scale_dtype, group_size: first_group_size }),
                    Some(RocmQuantizedWeight::W8A16 { packed: second_packed, scales: second_scales, scale_dtype: second_scale_dtype, group_size: second_group_size }),
                ) => Some((8, first_packed, first_scales, first_scale_dtype, first_group_size, second_packed, second_scales, second_scale_dtype, second_group_size)),
                _ => None,
            };
            if let Some((bits, first_packed, first_scales, first_scale_dtype, first_group_size, second_packed, second_scales, second_scale_dtype, second_group_size)) = dual {
                if input.cols != first.cols || input.cols != second.cols {
                    return Err(compute_error(format!("ROCm dual CT input cols={}，weights=[{},{}]/[{},{}]", input.cols, first.rows, first.cols, second.rows, second.cols,)));
                }
                let scale_dtype = |dtype: &ScaleDType| match dtype {
                    ScaleDType::Bf16 => 0,
                    ScaleDType::F16 => 1,
                    ScaleDType::F32 => 2,
                };
                let (first_device, second_device) = ops::hip::try_ct_dual_gemv_bf16(
                    self.device_id,
                    bits,
                    &input.data,
                    input.device.as_deref(),
                    input.cols,
                    input.rows,
                    first_packed,
                    first_scales,
                    scale_dtype(first_scale_dtype),
                    *first_group_size,
                    first.rows,
                    second_packed,
                    second_scales,
                    scale_dtype(second_scale_dtype),
                    *second_group_size,
                    second.rows,
                )
                .map_err(compute_error)?;
                return Ok((device_tensor_f32(first_device, input.rows, first.rows), device_tensor_f32(second_device, input.rows, second.rows)));
            }
        }
        Ok((self.linear(input, first)?, self.linear(input, second)?))
    }

    fn gated_linear(&self, input: &Self::Tensor, gate: &Self::Weight, up: &Self::Weight, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        if input.rows > 0
            && input.rows <= 8
            && gate.rows == up.rows
            && input.cols == gate.cols
            && input.cols == up.cols
            && let (
                Some(RocmQuantizedWeight::BlockFp8 { codes: gate_codes, scales: gate_scales, block_rows: gate_block_rows, block_cols: gate_block_cols, .. }),
                Some(RocmQuantizedWeight::BlockFp8 { codes: up_codes, scales: up_scales, block_rows: up_block_rows, block_cols: up_block_cols, .. }),
            ) = (gate.quantized(), up.quantized())
            && gate_block_rows == up_block_rows
            && gate_block_cols == up_block_cols
        {
            let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm gated BlockFp8 input 缺少 device buffer"))?;
            let output = ops::hip::try_block_fp8_gated_gemv_resident_f32(self.device_id, input_device, input.rows, gate_codes, gate_scales, up_codes, up_scales, gate.rows, input.cols, *gate_block_rows, *gate_block_cols, activation)
                .map_err(compute_error)?;
            return Ok(device_tensor_f32(output, input.rows, gate.rows));
        }
        let (gate, up) = self.dual_linear(input, gate, up)?;
        self.gated_activation(&gate, &up, activation)
    }

    fn gated_mlp(&self, input: &Self::Tensor, gate: &Self::Weight, up: &Self::Weight, down: &Self::Weight, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        let decode_w8 =
            input.rows == 1 && matches!(gate.quantized(), Some(RocmQuantizedWeight::W8A16 { .. })) && matches!(up.quantized(), Some(RocmQuantizedWeight::W8A16 { .. })) && matches!(down.quantized(), Some(RocmQuantizedWeight::W8A16 { .. }));
        if decode_w8 {
            let (gate, up) = self.dual_linear(input, gate, up)?;
            let gate_device = gate.device.as_deref().ok_or_else(|| compute_error("ROCm gated MLP gate 缺少 device buffer"))?;
            let up_device = up.device.as_deref().ok_or_else(|| compute_error("ROCm gated MLP up 缺少 device buffer"))?;
            let elements = checked_elements(gate.rows, gate.cols, "ROCm gated MLP activation")?;
            let activated = ops::hip::try_gated_activation_resident_bf16(self.device_id, gate_device, up_device, elements, activation).map_err(compute_error)?;
            if gate.rows != 0 && ops::hip::options().debug_finite {
                ops::hip::try_validate_finite_resident_range_bf16(self.device_id, &activated, 0, elements).map_err(|error| compute_error(format!("ROCm gated MLP BF16 激活包含非有限值: elements={elements}: {error}")))?;
            }
            return self.linear(&device_tensor_bf16(activated, gate.rows, gate.cols), down);
        }
        let activated = self.gated_linear(input, gate, up, activation)?;
        self.linear(&activated, down)
    }

    fn linear_gated_activation(&self, input: &Self::Tensor, gate: &Self::Weight, up: &Self::Tensor, activation: &Activation) -> Result<Self::Tensor, BackendError> {
        let gate = self.linear(input, gate)?;
        self.gated_activation(&gate, up, activation)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::rocm::RocmContext;

    #[test]
    #[ignore = "需要 ROCm device 0"]
    fn resident_subtract_and_relative_l1_only_download_partials() {
        let context = RocmContext::new(0).unwrap();
        let current = context.tensor_from_f32(vec![1.0, -2.0, 3.0, -4.0], 2, 2).unwrap();
        let previous = context.tensor_from_f32(vec![0.5, -1.0, 1.0, -6.0], 2, 2).unwrap();
        let residual = context.subtract_resident(&current, &previous).unwrap();
        assert_eq!(context.tensor_to_f32(&residual).unwrap(), vec![0.5, -1.0, 2.0, 2.0]);
        let zero = context.tensor_from_f32(vec![0.0; 4], 2, 2).unwrap();
        let (numerator, denominator) = context.relative_l1_delta_partial(&current, &zero, &previous).unwrap();
        assert!((numerator - 5.5).abs() < 1.0e-6, "numerator={numerator}");
        assert!((denominator - 8.5).abs() < 1.0e-6, "denominator={denominator}");
    }
}
