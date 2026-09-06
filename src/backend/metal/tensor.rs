//! Metal 基础 tensor、权重驻留和通用后端能力。

use std::mem::ManuallyDrop;

use objc2::rc::Retained;
use objc2_foundation::NSAutoreleasePool;

use crate::{
    backend::{Backend, BackendError, BackendResources, LinearWeight, SegmentedTensorBackend},
    kernel::metal as ops,
    moe::Activation,
};

use super::{
    context::{self, MetalContext, MetalTensor, MetalTensorDType},
    expect_resident_f16,
    kv_cache::MetalKvCache,
    resident::MetalWeight,
};

impl MetalContext {
    /// norm 类 kernel 只接受 resident F16 权重；F32 权重（`prepare_f32` 上传）
    /// 在这里经缓存的 cast kernel 转成 F16 tensor，避免每个 decode step 重新上传。
    pub(crate) fn resident_f16_norm_weight(&self, weight: &MetalWeight, name: &str) -> Result<MetalTensor, BackendError> {
        match weight {
            MetalWeight::F16(tensor) => Ok(tensor.clone()),
            MetalWeight::F32 { buffer, len } => {
                if let Some(cached) = self.weight_f16_view(buffer, *len) {
                    return Ok(cached);
                }
                let view = MetalTensor::new_f32(buffer.clone(), 1, *len);
                let converted = ops::to_f16_tensor(self, &view).map_err(|msg| BackendError::Compute { msg: format!("{name} 权重 F32->F16 转换失败: {msg}") })?;
                self.retain_weight_f16_view(buffer, &converted);
                Ok(converted)
            }
            _ => Err(BackendError::Compute { msg: format!("{name} 需要 resident F16/F32 权重") }),
        }
    }

    /// segmented RMSNorm kernel 接受 F16/Bf16 operand；F16 原生直通，
    /// 其余 dtype 统一转到 Bf16（linear 输出可能是 F16 之外的类型）。
    fn bf16_norm_operands(&self, left: &MetalTensor, right: &MetalTensor, name: &str) -> Result<(MetalTensor, MetalTensor), BackendError> {
        if left.dtype == right.dtype && matches!(left.dtype, MetalTensorDType::F16 | MetalTensorDType::Bf16) {
            return Ok((left.clone(), right.clone()));
        }
        let cast = |tensor: &MetalTensor| -> Result<MetalTensor, BackendError> { ops::to_bf16_tensor(self, tensor).map_err(|msg| BackendError::Compute { msg: format!("{name} operand 转换 Bf16 失败: {msg}") }) };
        Ok((cast(left)?, cast(right)?))
    }
}

pub struct MetalLayerScope<'a> {
    ctx: &'a MetalContext,
    pool: ManuallyDrop<Retained<NSAutoreleasePool>>,
}

impl<'a> MetalLayerScope<'a> {
    fn new(ctx: &'a MetalContext) -> Self {
        let pool = unsafe { NSAutoreleasePool::new() };
        Self { ctx, pool: ManuallyDrop::new(pool) }
    }
}

impl Drop for MetalLayerScope<'_> {
    fn drop(&mut self) {
        // 诊断:deferred 模式也在 drain 前同步,排除 autorelease pool 提前释放嫌疑
        if self.ctx.layer_scope_requires_sync() || std::env::var_os("ZLLM_DEBUG_SYNC_POOL").is_some() {
            self.ctx.synchronize();
        }
        // 诊断:跳过 drain,区分"pool 提前释放"与"同步消竞态"
        if std::env::var_os("ZLLM_DEBUG_NO_DRAIN").is_none() {
            unsafe { self.pool.drain() };
        }
    }
}

impl BackendResources for MetalContext {
    type Tensor = MetalTensor;
    type Weight = MetalWeight;
    type Cache = MetalKvCache;
    type LayerScope<'a>
        = MetalLayerScope<'a>
    where
        Self: 'a;

    fn layer_scope(&self) -> Self::LayerScope<'_> {
        MetalLayerScope::new(self)
    }

    fn token_rows(&self, tensor: &MetalTensor) -> usize {
        tensor.rows
    }

    fn token_cols(&self, tensor: &MetalTensor) -> usize {
        tensor.cols
    }

    fn tensor_allocated_bytes(&self, tensor: &MetalTensor) -> u64 {
        tensor.buffer.length()
    }

    fn begin_batch(&self) {
        self.clear_f16_casts();
        // GPU-only 层间依赖由同一 queue 保序；CPU 路由读回和最终输出才需要同步。
        self.set_deferred_layer_scope_sync(true);
        // 诊断模式(reset_gpu_stats 已开启 detailed)按 1 op/command 提交,
        // 否则 prefill 折叠统计会把层内多 kernel 记成一条 decode_batch。
        let operations = if self.detailed_gpu_profiles_enabled() { 1 } else { context::PREFILL_BATCH_MAX_OPERATIONS };
        self.set_deferred_batch_max_operations(operations);
        self.set_deferred_waits(true);
    }

    fn begin_decode_batch(&self) {
        self.clear_f16_casts();
        self.set_deferred_layer_scope_sync(true);
        let operations = if self.detailed_gpu_profiles_enabled() { 1 } else { self.decode_batch_max_operations() };
        self.set_deferred_batch_max_operations(operations);
        self.set_deferred_waits(true);
    }

    fn submit_batch(&self) {
        MetalContext::submit_batch(self);
    }

    fn finish_batch(&self) {
        // begin_* 修改的是 context 级状态，必须在公开执行边界恢复；否则后续无关算子
        // 会继续复用旧 command buffer，错误路径还可能永久跳过 layer-scope 同步。
        self.set_deferred_waits(false);
        self.set_deferred_layer_scope_sync(false);
        self.clear_f16_casts();
    }

    fn finish_batch_deferred(&self) {
        // 保持 defer_waits:等待点由 decode 流水线的 argmax CB 句柄决定;
        // 只恢复层作用域同步与 cast 暂存,避免一轮结束触发全队列 drain。
        self.set_deferred_layer_scope_sync(false);
        self.clear_f16_casts();
    }

    fn finish_stream_chunk(&self) {
        // 流式权重即将释放，只需确保本块设备工作完成；外层层/模型 batch 仍继续。
        self.synchronize();
        self.clear_f16_casts();
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        // 泛型 runtime 只能看到 trait 默认实现(空操作);必须显式转发到
        // inherent 版本,h3 staging 等处的 step 边界同步才真正生效。
        MetalContext::synchronize(self);
        Ok(())
    }

    fn prepare_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<MetalWeight, BackendError> {
        MetalWeight::upload(self, weight, rows, cols).map_err(|msg| BackendError::Compute { msg })
    }

    fn prepare_weight_rows(&self, weight: LinearWeight<'_>, source_rows: usize, cols: usize, selected_rows: &[u32]) -> Result<MetalWeight, BackendError> {
        if let LinearWeight::Quantized(matrix) = weight {
            if let crate::weight::format::quantization::QuantizedMatrixRef::Gguf(matrix) = matrix {
                if matrix.rows != source_rows || matrix.columns != cols {
                    return Err(BackendError::Compute { msg: format!("Metal selected GGUF shape [{},{}]，期望 [{source_rows},{cols}]", matrix.rows, matrix.columns) });
                }
                return MetalWeight::upload_gguf_rows(self, matrix, selected_rows).map_err(|msg| BackendError::Compute { msg });
            }
            return Err(BackendError::Compute { msg: format!("Metal selected rows 缺少 {} 原生算子", matrix.name()) });
        }
        crate::backend::prepare_weight_rows_default(self, weight, source_rows, cols, selected_rows)
    }

    fn prepare_weight_pair(&self, first: LinearWeight<'_>, second: LinearWeight<'_>, rows: usize, cols: usize) -> Result<(MetalWeight, MetalWeight), BackendError> {
        MetalWeight::upload_pair(self, first, second, rows, cols).map_err(|msg| BackendError::Compute { msg })
    }

    fn prepare_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<MetalWeight, BackendError> {
        if values.len() != rows.checked_mul(cols).ok_or_else(|| BackendError::Compute { msg: "Metal F32 weight 大小溢出".to_owned() })? {
            return Err(BackendError::Compute { msg: format!("Metal F32 weight 元素数 {}，期望 {}", values.len(), rows * cols) });
        }
        MetalWeight::upload_f32(self, values).map_err(|msg| BackendError::Compute { msg })
    }
}

impl Backend for MetalContext {
    fn linear(&self, input: &MetalTensor, weight: &MetalWeight) -> Result<MetalTensor, BackendError> {
        if input.rows == 1
            && input.dtype == MetalTensorDType::Bf16
            && let MetalWeight::W4A16 { packed, scales, scale_dtype, group_size, rows, cols } = weight
        {
            return ops::low_bit::w4a16_matmul_tensor_resident(self, input, packed, scales, *scale_dtype, *group_size, *rows, *cols).map_err(|msg| BackendError::Compute { msg });
        }
        if let MetalWeight::F32 { buffer, len } = weight {
            let input = ops::to_f32_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
            return ops::dense::matmul_tensor_resident_f32(self, &input, buffer, *len).map_err(|msg| BackendError::Compute { msg });
        }
        let requested_dtype = input.dtype;
        let input_f16 = if requested_dtype == MetalTensorDType::Bf16 && matches!(weight, MetalWeight::MlxAffine { .. }) { None } else { Some(ops::to_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?) };
        let input = input_f16.as_ref().unwrap_or(input);
        let output = match weight {
            MetalWeight::F16(weight) if requested_dtype == MetalTensorDType::Bf16 => ops::dense::matmul_tensor_resident_weight_f32(self, input, weight),
            MetalWeight::F16(weight) => ops::dense::matmul_tensor_resident_weight(self, input, weight),
            MetalWeight::Fp8 { codes, scale_inv, rows, cols } => ops::fp8::fp8_matmul_tensor_resident(self, input, codes, scale_inv, *rows, *cols),
            MetalWeight::Fp8PerTensor { codes, scale, rows, cols } => ops::fp8::fp8_matmul_tensor_per_tensor_resident(self, input, codes, scale, *rows, *cols),
            MetalWeight::Mxfp8 { codes, scale_inv, rows, cols } => ops::fp8::mxfp8_matmul_tensor_resident(self, input, codes, scale_inv, *rows, *cols),
            MetalWeight::Mxfp4 { packed, scales, rows, cols } => ops::low_bit::mxfp4_matmul_tensor_resident(self, input, packed, scales, *rows, *cols),
            MetalWeight::Nvfp4 { codes, codes_offset, scales, scales_offset, global_scale, global_scale_offset, rows, cols } => {
                ops::low_bit::nvfp4_matmul_tensor_resident(self, input, codes, *codes_offset, scales, *scales_offset, global_scale, *global_scale_offset, *rows, *cols)
            }
            MetalWeight::Gguf { blob, tensor_type, row_bytes, rows, cols } => ops::gguf::gguf_matmul_tensor_resident(self, input, blob, *tensor_type, *row_bytes, *rows, *cols),
            MetalWeight::W4A16 { packed, scales, scale_dtype, group_size, rows, cols } => ops::low_bit::w4a16_matmul_tensor_resident(self, input, packed, scales, *scale_dtype, *group_size, *rows, *cols),
            MetalWeight::W8A16 { packed, scales, scale_dtype, group_size, rows, cols } => ops::low_bit::w8a16_matmul_tensor_resident(self, input, packed, scales, *scale_dtype, *group_size, *rows, *cols),
            MetalWeight::MlxAffine { packed, scales, biases, scale_dtype, bits, group_size, rows, cols, pair_id, pair_role } => {
                if *pair_id == 0 {
                    ops::mlx::mlx_affine_matmul_tensor_resident(self, input, packed, scales, biases, *scale_dtype, *bits, *group_size, *rows, *cols)
                } else {
                    ops::mlx::mlx_affine_interleaved_matmul_tensor_resident(self, input, packed, scales, biases, *pair_role, *scale_dtype, *bits, *group_size, *rows, *cols)
                }
            }
            MetalWeight::F32 { .. } => return Err(BackendError::Compute { msg: "F32 常量不能作为线性权重".to_owned() }),
        }
        .map_err(|msg| BackendError::Compute { msg })?;
        if requested_dtype == MetalTensorDType::Bf16 { ops::to_bf16_tensor(self, &output).map_err(|msg| BackendError::Compute { msg }) } else { Ok(output) }
    }

    fn linear_f32(&self, input: &MetalTensor, weight: &MetalWeight) -> Result<MetalTensor, BackendError> {
        let input = ops::to_f32_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
        let output = match weight {
            MetalWeight::F16(weight) => ops::dense::matmul_tensor_resident_weight_mixed_f32(self, &input, weight),
            MetalWeight::Gguf { blob, tensor_type, row_bytes, rows, cols } => ops::gguf::gguf_matmul_tensor_resident(self, &input, blob, *tensor_type, *row_bytes, *rows, *cols),
            MetalWeight::F32 { buffer, len } => ops::dense::matmul_tensor_resident_f32(self, &input, buffer, *len),
            _ => return self.linear(&input, weight).and_then(|output| ops::to_f32_tensor(self, &output).map_err(|msg| BackendError::Compute { msg })),
        }
        .map_err(|msg| BackendError::Compute { msg })?;
        Ok(output)
    }

    fn linear_add(&self, input: &MetalTensor, weight: &MetalWeight, residual: &MetalTensor) -> Result<MetalTensor, BackendError> {
        if input.rows == 1
            && input.dtype == MetalTensorDType::F16
            && residual.dtype == MetalTensorDType::F16
            && let MetalWeight::Gguf { blob, tensor_type, row_bytes, rows, cols } = weight
            && matches!(tensor_type, 12 | 14)
            && input.cols == *cols
            && residual.len() == *rows
        {
            // GGUF q4k/q6k decode:残差并入 gemv epilogue,与 gemv + add_f16 两步逐位一致
            return ops::gguf::gguf_gemv_add_tensor(self, input, blob, *tensor_type, *row_bytes, *rows, *cols, residual).map_err(|msg| BackendError::Compute { msg });
        }
        if input.rows == 1
            && input.dtype == MetalTensorDType::F16
            && residual.dtype == MetalTensorDType::F16
            && let MetalWeight::Mxfp8 { codes, scale_inv, rows, cols } = weight
        {
            return ops::fp8::mxfp8_matmul_add_tensor_resident(self, input, codes, scale_inv, *rows, *cols, residual).map_err(|msg| BackendError::Compute { msg });
        }
        let output = self.linear(input, weight)?;
        self.add(residual, &output)
    }

    /// Dense FFN 后接残差:走 `gated_linear` + `linear_add`,使 MXFP8 down 投影
    /// 走 fused epilogue(`linear_add` 的 Mxfp8 分支)。GGUF 量化路径在
    /// `linear_add` 默认实现下走 `linear + add`,与原两步行为等价,无回退。
    fn gated_mlp_add_residual(&self, input: &MetalTensor, gate: &MetalWeight, up: &MetalWeight, down: &MetalWeight, activation: &crate::moe::Activation, residual: &MetalTensor) -> Result<MetalTensor, BackendError> {
        let activated = self.gated_linear(input, gate, up, activation)?;
        self.linear_add(&activated, down, residual)
    }

    fn linear_sigmoid_gate(&self, input: &MetalTensor, weight: &MetalWeight, value: &MetalTensor) -> Result<MetalTensor, BackendError> {
        if input.rows == 1
            && input.dtype == MetalTensorDType::F16
            && value.rows == 1
            && value.dtype == MetalTensorDType::F16
            && let MetalWeight::F16(weight) = weight
            && weight.rows == 1
            && weight.cols == input.cols
        {
            return ops::dense::linear_sigmoid_gate_tensor(self, input, weight, value).map_err(|msg| BackendError::Compute { msg });
        }
        let gate = self.linear(input, weight)?;
        self.sigmoid_gate(value, &gate)
    }

    fn select_row(&self, input: &MetalTensor, row: usize) -> Result<MetalTensor, BackendError> {
        let row = u32::try_from(row).map_err(|_| BackendError::Compute { msg: "Metal select_row 行号超出 u32".to_owned() })?;
        ops::moe::gather_rows_tensor(self, input, &[row]).map_err(|msg| BackendError::Compute { msg })
    }

    fn select_rows(&self, input: &MetalTensor, rows: &[u32]) -> Result<MetalTensor, BackendError> {
        ops::moe::gather_rows_tensor(self, input, rows).map_err(|msg| BackendError::Compute { msg })
    }

    fn argmax(&self, input: &MetalTensor) -> Result<u32, BackendError> {
        ops::moe::argmax_tensor(self, input, &[]).map_err(|msg| BackendError::Compute { msg })
    }

    fn argmax_excluding(&self, input: &MetalTensor, excluded: &[u32]) -> Result<u32, BackendError> {
        ops::moe::argmax_tensor(self, input, excluded).map_err(|msg| BackendError::Compute { msg })
    }

    fn sample_top_p(&self, input: &MetalTensor, temperature: f32, top_p: f32, random: f32) -> Result<u32, BackendError> {
        let input = ops::to_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
        ops::moe::sample_top_p_tensor(self, &input, temperature, top_p, random).map_err(|msg| BackendError::Compute { msg })
    }

    fn dual_linear(&self, input: &MetalTensor, first: &MetalWeight, second: &MetalWeight) -> Result<(MetalTensor, MetalTensor), BackendError> {
        let w4a16 = input.rows == 1 && matches!((first, second), (MetalWeight::W4A16 { .. }, MetalWeight::W4A16 { .. }));
        if matches!(first, MetalWeight::F32 { .. }) || matches!(second, MetalWeight::F32 { .. }) {
            return Ok((self.linear(input, first)?, self.linear(input, second)?));
        }
        if input.dtype == MetalTensorDType::Bf16 && !w4a16 {
            return Ok((self.linear(input, first)?, self.linear(input, second)?));
        }
        let input_f16 = if input.dtype == MetalTensorDType::Bf16 { None } else { Some(ops::to_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?) };
        let input = input_f16.as_ref().unwrap_or(input);
        match (first, second) {
            (
                MetalWeight::W4A16 { packed: first_packed, scales: first_scales, scale_dtype: first_scale_dtype, group_size: first_group_size, rows: first_rows, cols: first_cols },
                MetalWeight::W4A16 { packed: second_packed, scales: second_scales, scale_dtype: second_scale_dtype, group_size: second_group_size, rows: second_rows, cols: second_cols },
            ) if input.rows == 1 => ops::low_bit::w4a16_dual_gemv_tensor_resident(
                self,
                input,
                first_packed,
                first_scales,
                *first_scale_dtype,
                *first_group_size,
                *first_rows,
                *first_cols,
                second_packed,
                second_scales,
                *second_scale_dtype,
                *second_group_size,
                *second_rows,
                *second_cols,
            )
            .map_err(|msg| BackendError::Compute { msg }),
            (MetalWeight::Fp8 { codes: first_codes, scale_inv: first_scales, rows: first_rows, cols: first_cols }, MetalWeight::Fp8 { codes: second_codes, scale_inv: second_scales, rows: second_rows, cols: second_cols }) => {
                ops::fp8::fp8_dual_matmul_tensor_resident(self, input, first_codes, first_scales, *first_rows, *first_cols, second_codes, second_scales, *second_rows, *second_cols).map_err(|msg| BackendError::Compute { msg })
            }
            (MetalWeight::Mxfp8 { codes: first_codes, scale_inv: first_scales, rows: first_rows, cols: first_cols }, MetalWeight::Mxfp8 { codes: second_codes, scale_inv: second_scales, rows: second_rows, cols: second_cols }) => {
                ops::fp8::mxfp8_dual_matmul_tensor_resident(self, input, first_codes, first_scales, *first_rows, *first_cols, second_codes, second_scales, *second_rows, *second_cols).map_err(|msg| BackendError::Compute { msg })
            }
            (
                MetalWeight::Nvfp4 {
                    codes: first_codes,
                    codes_offset: first_codes_offset,
                    scales: first_scales,
                    scales_offset: first_scales_offset,
                    global_scale: first_global_scale,
                    global_scale_offset: first_global_scale_offset,
                    rows: first_rows,
                    cols: first_cols,
                },
                MetalWeight::Nvfp4 {
                    codes: second_codes,
                    codes_offset: second_codes_offset,
                    scales: second_scales,
                    scales_offset: second_scales_offset,
                    global_scale: second_global_scale,
                    global_scale_offset: second_global_scale_offset,
                    rows: second_rows,
                    cols: second_cols,
                },
            ) => ops::low_bit::nvfp4_dual_matmul_tensor_resident(
                self,
                input,
                first_codes,
                *first_codes_offset,
                first_scales,
                *first_scales_offset,
                first_global_scale,
                *first_global_scale_offset,
                *first_rows,
                *first_cols,
                second_codes,
                *second_codes_offset,
                second_scales,
                *second_scales_offset,
                second_global_scale,
                *second_global_scale_offset,
                *second_rows,
                *second_cols,
            )
            .map_err(|msg| BackendError::Compute { msg }),
            (
                MetalWeight::Gguf { blob: first_blob, tensor_type: 12, row_bytes: first_row_bytes, rows: first_rows, cols: first_cols },
                MetalWeight::Gguf { blob: second_blob, tensor_type: 12, row_bytes: second_row_bytes, rows: second_rows, cols: second_cols },
            ) if input.rows == 1 && first_cols == second_cols && input.cols == *first_cols => {
                // decode 单行 Q/K(q4k×q4k)合并 dispatch;行数可不同(GQA)
                ops::gguf::gguf_dual_gemv_q4k_tensor(self, input, first_blob, *first_row_bytes, *first_rows, second_blob, *second_row_bytes, *second_rows, *first_cols).map_err(|msg| BackendError::Compute { msg })
            }
            (
                MetalWeight::Gguf { blob: first_blob, tensor_type: first_type @ (18 | 21), row_bytes: first_row_bytes, rows: first_rows, cols: first_cols },
                MetalWeight::Gguf { blob: second_blob, tensor_type: second_type @ (18 | 21), row_bytes: second_row_bytes, rows: second_rows, cols: second_cols },
            ) if input.rows == 1 && first_cols == second_cols && input.cols == *first_cols => {
                ops::gguf::gguf_dual_gemv_iq3_tensor(self, input, first_blob, *first_type, *first_row_bytes, *first_rows, second_blob, *second_type, *second_row_bytes, *second_rows, *first_cols).map_err(|msg| BackendError::Compute { msg })
            }
            _ => Ok((self.linear(input, first)?, self.linear(input, second)?)),
        }
    }

    fn triple_linear(&self, input: &MetalTensor, first: &MetalWeight, second: &MetalWeight, third: &MetalWeight) -> Result<(MetalTensor, MetalTensor, MetalTensor), BackendError> {
        if let (
            MetalWeight::MlxAffine {
                packed: first_packed,
                scales: first_scales,
                biases: first_biases,
                scale_dtype: first_scale_dtype,
                bits: first_bits,
                group_size: first_group_size,
                rows: first_rows,
                cols: first_cols,
                pair_id: first_pair_id,
                ..
            },
            MetalWeight::MlxAffine {
                packed: second_packed,
                scales: second_scales,
                biases: second_biases,
                scale_dtype: second_scale_dtype,
                bits: second_bits,
                group_size: second_group_size,
                rows: second_rows,
                cols: second_cols,
                pair_id: second_pair_id,
                ..
            },
            MetalWeight::MlxAffine {
                packed: third_packed,
                scales: third_scales,
                biases: third_biases,
                scale_dtype: third_scale_dtype,
                bits: third_bits,
                group_size: third_group_size,
                rows: third_rows,
                cols: third_cols,
                pair_id: third_pair_id,
                ..
            },
        ) = (first, second, third)
            && input.rows == 1
            && input.dtype == MetalTensorDType::F16
            && *first_bits == 4
            && *second_bits == 4
            && *third_bits == 4
            && *first_pair_id == 0
            && *second_pair_id == 0
            && *third_pair_id == 0
            && first_scale_dtype == second_scale_dtype
            && first_scale_dtype == third_scale_dtype
            && first_group_size == second_group_size
            && first_group_size == third_group_size
            && first_cols == second_cols
            && first_cols == third_cols
            && *first_cols == input.cols
        {
            return ops::mlx::mlx_affine_triple_gemv_f16_u4(
                self,
                input,
                first_packed,
                first_scales,
                first_biases,
                *first_rows,
                second_packed,
                second_scales,
                second_biases,
                *second_rows,
                third_packed,
                third_scales,
                third_biases,
                *third_rows,
                *first_scale_dtype,
                *first_group_size,
                *first_cols,
            )
            .map_err(|msg| BackendError::Compute { msg });
        }
        Ok((self.linear(input, first)?, self.linear(input, second)?, self.linear(input, third)?))
    }

    fn gated_linear(&self, input: &MetalTensor, gate: &MetalWeight, up: &MetalWeight, activation: &crate::moe::Activation) -> Result<MetalTensor, BackendError> {
        // MLX affine 对的形状指纹:命中 fused gated gemv 的判定基于它,不掺输入 dtype。
        let affine_fingerprint = |weight: &MetalWeight| match weight {
            MetalWeight::MlxAffine { scale_dtype, bits, group_size, rows, cols, pair_id, pair_role, .. } => Some((*scale_dtype, *bits, *group_size, *rows, *cols, *pair_id, *pair_role)),
            _ => None,
        };
        // F16 输入只在确定命中 bf16 fused gated gemv(paired 或 u8)时才 cast;
        // u4(E4B)没有 fused 版,保持 F16 走 dual gemv + gated activation 原生直通,
        // 否则每层 mlp 输入/输出各付一次 cast 且整条链被拖进 bf16。
        let fusable = input.rows == 1
            && affine_fingerprint(gate).zip(affine_fingerprint(up)).is_some_and(|(gate, up)| {
                let geometry = |value: &(u32, usize, usize, usize, usize, u64, u32)| (value.0, value.1, value.2, value.3, value.4);
                (gate.5 != 0 && gate.5 == up.5 && gate.6 == 0 && up.6 == 1 || gate.0 == 0 && up.0 == 0 && gate.1 == 8 && up.1 == 8 && gate.2 == 64 && up.2 == 64) && geometry(&gate) == geometry(&up)
            });
        let cast_input;
        let input = if fusable && input.dtype == MetalTensorDType::F16 {
            cast_input = ops::to_bf16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
            &cast_input
        } else {
            input
        };
        if input.rows == 1
            && input.dtype == MetalTensorDType::Bf16
            && let (
                MetalWeight::MlxAffine {
                    packed: gate_packed,
                    scales: gate_scales,
                    biases: gate_biases,
                    scale_dtype: gate_scale_dtype,
                    bits: gate_bits,
                    group_size: gate_group_size,
                    rows: gate_rows,
                    cols: gate_cols,
                    pair_id: gate_pair_id,
                    pair_role: gate_pair_role,
                },
                MetalWeight::MlxAffine {
                    packed: up_packed,
                    scales: up_scales,
                    biases: up_biases,
                    scale_dtype: up_scale_dtype,
                    bits: up_bits,
                    group_size: up_group_size,
                    rows: up_rows,
                    cols: up_cols,
                    pair_id: up_pair_id,
                    pair_role: up_pair_role,
                },
            ) = (gate, up)
        {
            if *gate_pair_id != 0
                && gate_pair_id == up_pair_id
                && *gate_pair_role == 0
                && *up_pair_role == 1
                && gate_scale_dtype == up_scale_dtype
                && gate_bits == up_bits
                && gate_group_size == up_group_size
                && gate_rows == up_rows
                && gate_cols == up_cols
            {
                return ops::mlx::mlx_affine_gated_interleaved_gemv_bf16_u8(self, input, gate_packed, gate_scales, gate_biases, *gate_scale_dtype, *gate_bits, *gate_group_size, *gate_rows, *gate_cols, activation)
                    .map_err(|msg| BackendError::Compute { msg });
            }
            if *gate_scale_dtype == 0 && *up_scale_dtype == 0 && *gate_bits == 8 && *up_bits == 8 && *gate_group_size == 64 && gate_group_size == up_group_size && gate_rows == up_rows && gate_cols == up_cols {
                return ops::mlx::mlx_affine_gated_gemv_bf16_u8(self, input, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases, *gate_group_size, *gate_rows, *gate_cols, activation)
                    .map_err(|msg| BackendError::Compute { msg });
            }
        }
        let compressed_gated = matches!((gate, up), (MetalWeight::W4A16 { .. }, MetalWeight::W4A16 { .. }) | (MetalWeight::W8A16 { .. }, MetalWeight::W8A16 { .. }));
        if input.dtype == MetalTensorDType::Bf16 && !compressed_gated {
            let (gate, up) = self.dual_linear(input, gate, up)?;
            return self.gated_activation(&gate, &up, activation);
        }
        match (gate, up) {
            (
                MetalWeight::MlxAffine {
                    packed: gate_packed,
                    scales: gate_scales,
                    biases: gate_biases,
                    scale_dtype: gate_scale_dtype,
                    bits: gate_bits,
                    group_size: gate_group_size,
                    rows: gate_rows,
                    cols: gate_cols,
                    pair_id: gate_pair_id,
                    ..
                },
                MetalWeight::MlxAffine { packed: up_packed, scales: up_scales, biases: up_biases, scale_dtype: up_scale_dtype, bits: up_bits, group_size: up_group_size, rows: up_rows, cols: up_cols, pair_id: up_pair_id, .. },
            ) if input.rows == 1
                && input.dtype == MetalTensorDType::F16
                && *gate_bits == 4
                && *up_bits == 4
                && *gate_pair_id == 0
                && *up_pair_id == 0
                && gate_scale_dtype == up_scale_dtype
                && gate_group_size == up_group_size
                && gate_rows == up_rows
                && gate_cols == up_cols
                && *gate_cols == input.cols =>
            {
                ops::mlx::mlx_affine_gated_gemv_f16_u4(self, input, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases, *gate_scale_dtype, *gate_group_size, *gate_rows, *gate_cols, activation)
                    .map_err(|msg| BackendError::Compute { msg })
            }
            (
                MetalWeight::W4A16 { packed: gate_packed, scales: gate_scales, scale_dtype: gate_scale_dtype, group_size: gate_group_size, rows: gate_rows, cols: gate_cols },
                MetalWeight::W4A16 { packed: up_packed, scales: up_scales, scale_dtype: up_scale_dtype, group_size: up_group_size, rows: up_rows, cols: up_cols },
            ) => {
                let input_f16 = if input.dtype == MetalTensorDType::Bf16 { None } else { Some(ops::to_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?) };
                let input = input_f16.as_ref().unwrap_or(input);
                ops::low_bit::w4a16_gated_gemv_tensor_resident(
                    self,
                    input,
                    gate_packed,
                    gate_scales,
                    *gate_scale_dtype,
                    *gate_group_size,
                    *gate_rows,
                    *gate_cols,
                    up_packed,
                    up_scales,
                    *up_scale_dtype,
                    *up_group_size,
                    *up_rows,
                    *up_cols,
                    activation,
                )
                .map_err(|msg| BackendError::Compute { msg })
            }
            (
                MetalWeight::W8A16 { packed: gate_packed, scales: gate_scales, scale_dtype: gate_scale_dtype, group_size: gate_group_size, rows: gate_rows, cols: gate_cols },
                MetalWeight::W8A16 { packed: up_packed, scales: up_scales, scale_dtype: up_scale_dtype, group_size: up_group_size, rows: up_rows, cols: up_cols },
            ) => ops::low_bit::w8a16_gated_matmul_tensor_resident(
                self,
                input,
                gate_packed,
                gate_scales,
                *gate_scale_dtype,
                *gate_group_size,
                *gate_rows,
                *gate_cols,
                up_packed,
                up_scales,
                *up_scale_dtype,
                *up_group_size,
                *up_rows,
                *up_cols,
                activation,
            )
            .map_err(|msg| BackendError::Compute { msg }),
            (
                MetalWeight::Gguf { blob: gate_blob, tensor_type: gate_type, row_bytes: gate_row_bytes, rows: gate_rows, cols: gate_cols },
                MetalWeight::Gguf { blob: up_blob, tensor_type: up_type, row_bytes: up_row_bytes, rows: up_rows, cols: up_cols },
            ) => {
                let input = ops::to_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
                ops::gguf::gguf_gated_matmul_tensor_resident(self, &input, gate_blob, *gate_type, *gate_row_bytes, *gate_rows, *gate_cols, up_blob, *up_type, *up_row_bytes, *up_rows, *up_cols, activation)
                    .map_err(|msg| BackendError::Compute { msg })
            }
            (MetalWeight::Mxfp8 { codes: gate_codes, scale_inv: gate_scales, rows: gate_rows, cols: gate_cols }, MetalWeight::Mxfp8 { codes: up_codes, scale_inv: up_scales, rows: up_rows, cols: up_cols }) if input.rows == 1 => {
                let input = ops::to_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
                ops::fp8::mxfp8_gated_gemv_tensor_resident(self, &input, gate_codes, gate_scales, *gate_rows, *gate_cols, up_codes, up_scales, *up_rows, *up_cols, activation).map_err(|msg| BackendError::Compute { msg })
            }
            (MetalWeight::Mxfp4 { packed: gate_packed, scales: gate_scales, rows: gate_rows, cols: gate_cols }, MetalWeight::Mxfp4 { packed: up_packed, scales: up_scales, rows: up_rows, cols: up_cols })
                if gate_rows == up_rows && gate_cols == up_cols =>
            {
                let input = ops::to_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
                ops::low_bit::mxfp4_gated_matmul_tensor_resident(self, &input, gate_packed, gate_scales, up_packed, up_scales, *gate_rows, *gate_cols, activation).map_err(|msg| BackendError::Compute { msg })
            }
            _ => {
                let (gate, up) = self.dual_linear(input, gate, up)?;
                self.gated_activation(&gate, &up, activation)
            }
        }
    }

    fn linear_gated_activation(&self, input: &MetalTensor, gate: &MetalWeight, up: &MetalTensor, activation: &crate::moe::Activation) -> Result<MetalTensor, BackendError> {
        if input.rows == 1
            && input.dtype == MetalTensorDType::Bf16
            && up.rows == 1
            && up.dtype == MetalTensorDType::Bf16
            && let MetalWeight::MlxAffine { packed, scales, biases, scale_dtype, bits, group_size, rows, cols, .. } = gate
            && *scale_dtype == 0
            && *bits == 4
            && *group_size == 64
            && up.cols == *rows
        {
            return ops::mlx::mlx_affine_linear_gated_bf16_u4(self, input, packed, scales, biases, up, *group_size, *rows, *cols, activation).map_err(|msg| BackendError::Compute { msg });
        }
        let gate = self.linear(input, gate)?;
        self.gated_activation(&gate, up, activation)
    }

    fn rmsnorm(&self, input: &MetalTensor, weight: &MetalWeight, eps: f32) -> Result<MetalTensor, BackendError> {
        // F32 权重(CT 源 norm 走 prepare_f32)复用 rmsnorm_f32 的 f32 内核,
        // 避免"只收 F16"把 CT 权重路径整个挡死。
        match weight {
            MetalWeight::F16(weight) => {
                let output = ops::tensor::rmsnorm_tensor_resident_weight(self, input, weight, eps, 0.0).map_err(|msg| BackendError::Compute { msg })?;
                Ok(output)
            }
            MetalWeight::F32 { buffer, len } => {
                // F16 输入直读 F32 权重:省掉 cast 链两侧的 f16→f32 输入 cast 与
                // f32→f16 输出 cast(每 norm 2 个 dispatch),数值与 cast 链逐位一致
                // (kernel 注释与一致性测试保证)。
                if input.dtype == MetalTensorDType::F16 {
                    return ops::tensor::rmsnorm_f16_in_f32_weight_tensor_resident(self, input, buffer, *len, eps, 0.0).map_err(|msg| BackendError::Compute { msg });
                }
                let output = self.rmsnorm_f32(input, weight, eps)?;
                // 统一返回 F16:下游(cache append/blit/split)全部假定 F16 布局,
                // F32 输出会让后续按 F16 字节数读取 F32 buffer,产生 NaN。
                ops::to_f16_tensor(self, &output).map_err(|msg| BackendError::Compute { msg })
            }
            _ => Err(BackendError::Compute { msg: "RMSNorm 需要 resident F16/F32 权重".to_owned() }),
        }
    }

    fn rmsnorm_f32(&self, input: &MetalTensor, weight: &MetalWeight, eps: f32) -> Result<MetalTensor, BackendError> {
        let input = ops::to_f32_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
        match weight {
            MetalWeight::F16(weight) => ops::tensor::rmsnorm_tensor_resident_weight(self, &input, weight, eps, 0.0),
            MetalWeight::F32 { buffer, len } => ops::tensor::rmsnorm_tensor_resident_f32_weight(self, &input, buffer, *len, eps, 0.0),
            _ => Err("F32 RMSNorm 需要 resident F16/F32 权重".to_owned()),
        }
        .map_err(|msg| BackendError::Compute { msg })
    }

    fn grouped_rmsnorm(&self, input: &MetalTensor, weight: &MetalWeight, eps: f32, groups: usize) -> Result<MetalTensor, BackendError> {
        // 层 norm 权重统一走 prepare_f32(见 prepare_mistral_layer),只接 F32 权重;
        // F16 权重形态没有分组内核,显式报错避免静默走错精度。
        let MetalWeight::F32 { buffer, len } = weight else {
            return Err(BackendError::Compute { msg: "Metal grouped RMSNorm 需要 resident F32 权重(prepare_f32 源)".to_owned() });
        };
        match input.dtype {
            MetalTensorDType::F16 => ops::tensor::grouped_rmsnorm_f16_in_f32_weight_tensor_resident(self, input, buffer, *len, eps, groups),
            MetalTensorDType::F32 => ops::tensor::grouped_rmsnorm_f32_in_f32_weight_to_f16_tensor_resident(self, input, buffer, *len, eps, groups),
            dtype => return Err(BackendError::Compute { msg: format!("Metal grouped RMSNorm 输入必须是 F16/F32，实际 {dtype:?}") }),
        }
        .map_err(|msg| BackendError::Compute { msg })
    }

    fn grouped_rmsnorm_f32(&self, input: &MetalTensor, weight: &MetalWeight, eps: f32, groups: usize) -> Result<MetalTensor, BackendError> {
        let MetalWeight::F32 { buffer, len } = weight else {
            return Err(BackendError::Compute { msg: "Metal F32 grouped RMSNorm 需要 resident F32 权重(prepare_f32 源)".to_owned() });
        };
        let input = ops::to_f32_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
        ops::tensor::grouped_rmsnorm_f32_tensor_resident(self, &input, buffer, *len, eps, groups).map_err(|msg| BackendError::Compute { msg })
    }

    fn rmsnorm_add_scaled(&self, left: &MetalTensor, right: &MetalTensor, weight: &MetalWeight, eps: f32, scale: f32) -> Result<MetalTensor, BackendError> {
        let weight = self.resident_f16_norm_weight(weight, "RMSNorm add")?;
        let (left, right) = self.bf16_norm_operands(left, right, "RMSNorm add")?;
        ops::tensor::segmented_rmsnorm_add_scaled_tensor(self, &left, &right, &weight, 1, left.cols, eps, scale).map_err(|msg| BackendError::Compute { msg })
    }

    fn segmented_rmsnorm_add_scaled(&self, left: &MetalTensor, right: &MetalTensor, weight: &MetalWeight, segments: usize, segment_columns: usize, eps: f32, scale: f32) -> Result<Vec<MetalTensor>, BackendError> {
        let weight = self.resident_f16_norm_weight(weight, "segmented RMSNorm")?;
        let (left, right) = self.bf16_norm_operands(left, right, "segmented RMSNorm")?;
        let mut tensor = ops::tensor::segmented_rmsnorm_add_scaled_tensor(self, &left, &right, &weight, segments, segment_columns, eps, scale).map_err(|msg| BackendError::Compute { msg })?;
        let mut output = Vec::with_capacity(segments);
        // 剥皮切分走池化:decode 每 token 42 个同宽切片,新分配+清零的 churn 占 8ms+
        for slot in 1..segments {
            let (head, tail) = ops::shape::split_columns_pooled_tensor(self, &tensor, segment_columns, slot - 1).map_err(|msg| BackendError::Compute { msg })?;
            output.push(head);
            tensor = tail;
        }
        output.push(tensor);
        Ok(output)
    }

    fn gemma_rmsnorm(&self, input: &MetalTensor, weight: &MetalWeight, eps: f32) -> Result<MetalTensor, BackendError> {
        let weight = expect_resident_f16(weight, "GemmaRMSNorm 需要 resident F16 权重")?;
        ops::tensor::rmsnorm_tensor_resident_weight(self, input, weight, eps, 1.0).map_err(|msg| BackendError::Compute { msg })
    }

    fn gemma_rmsnorm_f32(&self, input: &MetalTensor, weight: &MetalWeight, eps: f32) -> Result<MetalTensor, BackendError> {
        match weight {
            // F16 权重与 F16 输入原生直通(kernel 内 F32 累加),层内数据流保持 F16
            MetalWeight::F16(weight) => ops::tensor::rmsnorm_tensor_resident_weight(self, input, weight, eps, 1.0),
            MetalWeight::F32 { buffer, len } => {
                let input = ops::to_f32_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
                ops::tensor::rmsnorm_tensor_resident_f32_weight(self, &input, buffer, *len, eps, 1.0)
            }
            _ => Err("F32 GemmaRMSNorm 需要 resident F16/F32 权重".to_owned()),
        }
        .map_err(|msg| BackendError::Compute { msg })
    }

    fn layernorm_bias(&self, input: &MetalTensor, weight: &MetalWeight, bias: &MetalWeight, eps: f32) -> Result<MetalTensor, BackendError> {
        // F32 权重经 resident_f16_norm_weight 的缓存 cast 转换,避免每层每 token
        // 重新做 host F32→F16 转换并新建 MTLBuffer。
        let weight = self.resident_f16_norm_weight(weight, "LayerNorm")?;
        let bias = self.resident_f16_norm_weight(bias, "LayerNorm bias")?;
        ops::mla::layernorm_bias_tensor(self, input, &weight, &bias, eps).map_err(|msg| BackendError::Compute { msg })
    }

    fn split_columns(&self, input: &MetalTensor, left_columns: usize) -> Result<(MetalTensor, MetalTensor), BackendError> {
        ops::shape::split_columns_tensor(self, input, left_columns).map_err(|msg| BackendError::Compute { msg })
    }

    fn split_gated_activation(&self, input: MetalTensor, left_columns: usize, activation: &Activation) -> Result<MetalTensor, BackendError> {
        // 单 pass 消费 packed gate/up,替代默认 split_columns(两中间 buffer + 发散 kernel)+gated_activation。
        ops::dense::gated_activation_packed_tensor(self, &input, left_columns, activation).map_err(|msg| BackendError::Compute { msg })
    }

    fn split_interleaved_columns(&self, input: &MetalTensor, block_columns: usize) -> Result<(MetalTensor, MetalTensor), BackendError> {
        ops::shape::split_interleaved_columns_tensor(self, input, block_columns).map_err(|msg| BackendError::Compute { msg })
    }

    fn concat_columns(&self, left: &MetalTensor, right: &MetalTensor) -> Result<MetalTensor, BackendError> {
        ops::shape::concat_columns_tensor(self, left, right).map_err(|msg| BackendError::Compute { msg })
    }

    fn rope(&self, input: &MetalTensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<MetalTensor, BackendError> {
        ops::shape::apply_rope_partial_tensor(self, input, head_count, rotary_dim, layout, position, cos, sin).map_err(|msg| BackendError::Compute { msg })
    }

    fn rope_prefix(&self, input: &MetalTensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<MetalTensor, BackendError> {
        let output = ops::shape::apply_rope_prefix_tensor(self, input, head_count, rotary_dim, layout, position, cos, sin).map_err(|msg| BackendError::Compute { msg })?;
        Ok(output)
    }

    fn rope_prefix_qk(
        &self,
        query: &MetalTensor,
        key: &MetalTensor,
        num_heads: usize,
        num_kv_heads: usize,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        position: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(MetalTensor, MetalTensor), BackendError> {
        // decode 单行 + interleaved + 全维 rotary:单 dispatch 融合,省一次小算子延迟;
        // cos/sin 全表常驻 GPU,position 走 set_bytes,CPU 每 token 不再碰 rope 数据。
        if layout == crate::attention::rope::RotaryLayout::Interleaved
            && query.rows == 1
            && key.rows == 1
            && query.dtype == MetalTensorDType::F16
            && key.dtype == MetalTensorDType::F16
            && num_heads > 0
            && query.cols / num_heads == rotary_dim
        {
            let (cos_table, sin_table) = self.decode_rope_table_buffers(cos, sin, rotary_dim / 2).map_err(|msg| BackendError::Compute { msg })?;
            return ops::shape::apply_rope_qk_interleaved_prefix_f16_tensor(self, query, key, num_heads, num_kv_heads, position, &cos_table, &sin_table, rotary_dim / 2).map_err(|msg| BackendError::Compute { msg });
        }
        let query_output = self.rope_prefix(query, num_heads, rotary_dim, layout, position, cos, sin)?;
        let key_output = self.rope_prefix(key, num_kv_heads, rotary_dim, layout, position, cos, sin)?;
        Ok((query_output, key_output))
    }

    fn add(&self, left: &MetalTensor, right: &MetalTensor) -> Result<MetalTensor, BackendError> {
        let left_bf16;
        let right_bf16;
        let (left, right) = if left.dtype == MetalTensorDType::Bf16 || right.dtype == MetalTensorDType::Bf16 {
            left_bf16 = ops::to_bf16_tensor(self, left).map_err(|msg| BackendError::Compute { msg })?;
            right_bf16 = ops::to_bf16_tensor(self, right).map_err(|msg| BackendError::Compute { msg })?;
            (&left_bf16, &right_bf16)
        } else {
            (left, right)
        };
        let output = ops::tensor::add_tensor(self, left, right).map_err(|msg| BackendError::Compute { msg })?;
        Ok(output)
    }

    fn add_scaled(&self, left: &MetalTensor, right: &MetalTensor, scale: f32) -> Result<MetalTensor, BackendError> {
        let left_bf16;
        let right_bf16;
        let (left, right) = if left.dtype == MetalTensorDType::Bf16 || right.dtype == MetalTensorDType::Bf16 {
            left_bf16 = ops::to_bf16_tensor(self, left).map_err(|msg| BackendError::Compute { msg })?;
            right_bf16 = ops::to_bf16_tensor(self, right).map_err(|msg| BackendError::Compute { msg })?;
            (&left_bf16, &right_bf16)
        } else {
            (left, right)
        };
        let output = ops::tensor::add_scaled_tensor(self, left, right, scale).map_err(|msg| BackendError::Compute { msg })?;
        Ok(output)
    }

    fn sigmoid_gate(&self, input: &MetalTensor, gate: &MetalTensor) -> Result<MetalTensor, BackendError> {
        ops::tensor::sigmoid_gate_tensor(self, input, gate).map_err(|msg| BackendError::Compute { msg })
    }

    fn gated_activation(&self, gate: &MetalTensor, up: &MetalTensor, activation: &crate::moe::Activation) -> Result<MetalTensor, BackendError> {
        let gate_bf16;
        let up_bf16;
        let (gate, up) = if gate.dtype == MetalTensorDType::Bf16 || up.dtype == MetalTensorDType::Bf16 {
            gate_bf16 = ops::to_bf16_tensor(self, gate).map_err(|msg| BackendError::Compute { msg })?;
            up_bf16 = ops::to_bf16_tensor(self, up).map_err(|msg| BackendError::Compute { msg })?;
            (&gate_bf16, &up_bf16)
        } else {
            (gate, up)
        };
        let output = ops::tensor::gated_activation_tensor(self, gate, up, activation).map_err(|msg| BackendError::Compute { msg })?;
        Ok(output)
    }
}

impl SegmentedTensorBackend for MetalContext {
    fn concat_token_rows(&self, tensors: &[&MetalTensor]) -> Result<MetalTensor, BackendError> {
        if tensors.is_empty() {
            return Err(BackendError::Compute { msg: "concat_token_rows 输入为空".to_owned() });
        }
        let columns = tensors[0].cols;
        if tensors.iter().any(|tensor| tensor.cols != columns || tensor.dtype != tensors[0].dtype) {
            return Err(BackendError::Compute {
                msg: format!("concat_token_rows 列宽/dtype 不一致: cols={:?} dtype={:?}", tensors.iter().map(|tensor| tensor.cols).collect::<Vec<_>>(), tensors.iter().map(|tensor| tensor.dtype).collect::<Vec<_>>()),
            });
        }
        let rows = tensors.iter().map(|tensor| tensor.rows).sum::<usize>();
        let element_bytes = match tensors[0].dtype {
            MetalTensorDType::F16 | MetalTensorDType::Bf16 => std::mem::size_of::<u16>(),
            MetalTensorDType::F32 => std::mem::size_of::<f32>(),
        };
        let output = match tensors[0].dtype {
            MetalTensorDType::F16 => self.tensor_kernel_output(rows, columns),
            MetalTensorDType::Bf16 => self.tensor_kernel_output_bf16(rows, columns),
            MetalTensorDType::F32 => self.tensor_kernel_output_f32(rows, columns),
        };
        let command = self.command_buffer();
        let encoder = command.new_blit_command_encoder();
        let mut offset = 0u64;
        for tensor in tensors {
            let size = tensor.rows.checked_mul(columns).and_then(|elements| elements.checked_mul(element_bytes)).ok_or_else(|| BackendError::Compute { msg: "concat_token_rows 字节数溢出".to_owned() })? as u64;
            if size > 0 {
                encoder.copy_from_buffer(&tensor.buffer, 0, &output.buffer, offset, size);
            }
            offset += size;
        }
        encoder.end_encoding();
        self.commit_and_wait(command.as_ref());
        Ok(output)
    }

    fn slice_token_rows(&self, tensor: &MetalTensor, row_start: usize, rows: usize) -> Result<MetalTensor, BackendError> {
        if row_start.checked_add(rows).map_or(true, |end| end > tensor.rows) {
            return Err(BackendError::Compute { msg: format!("slice_token_rows 区间 {row_start}..{} 超出 rows={}", row_start + rows, tensor.rows) });
        }
        if rows == tensor.rows {
            // blit 全量拷贝保持与 concat 相同的独立 buffer 语义
            let whole = [&*tensor];
            return self.concat_token_rows(&whole);
        }
        let range: Vec<u32> = (row_start..row_start + rows).map(|row| u32::try_from(row).expect("slice_token_rows 行号超出 u32")).collect();
        self.select_rows(tensor, &range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::format::mxfp4::Mxfp4Matrix;

    #[test]
    fn mxfp4_weight_stays_packed_and_executes() {
        let context = MetalContext::new_default().unwrap();
        let (rows, cols) = (3usize, 32usize);
        let matrix = Mxfp4Matrix::new(rows, cols, vec![0x22; rows * cols / 2], vec![127; rows]).unwrap();
        let weight = context.prepare_weight(LinearWeight::mxfp4(&matrix), rows, cols).unwrap();
        assert!(matches!(weight, MetalWeight::Mxfp4 { rows: 3, cols: 32, .. }));
        let input = context.tensor_from_f32(&vec![0.5; cols], 1, cols).unwrap();
        let output = context.linear(&input, &weight).unwrap();
        assert_eq!(context.tensor_to_f32(&output), vec![16.0; rows]);
    }

    #[test]
    fn concat_and_slice_token_rows_round_trip() {
        let context = MetalContext::new_default().unwrap();
        let left = context.tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let right = context.tensor_from_f32(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 3, 2).unwrap();
        let merged = SegmentedTensorBackend::concat_token_rows(&context, &[&left, &right]).unwrap();
        assert_eq!((merged.rows, merged.cols), (5, 2));
        assert_eq!(context.tensor_to_f32(&merged), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let slice = SegmentedTensorBackend::slice_token_rows(&context, &left, 1, 1).unwrap();
        assert_eq!((slice.rows, slice.cols), (1, 2));
        assert_eq!(context.tensor_to_f32(&slice), vec![3.0, 4.0]);
        assert!(SegmentedTensorBackend::slice_token_rows(&context, &left, 1, 2).is_err());
    }
    #[test]
    fn f32_weight_layernorm_bias_matches_reference() {
        let context = MetalContext::new_default().unwrap();
        let rows = 2;
        let columns = 4;
        let input = vec![1.0, 2.0, 3.0, 4.0, -2.0, -1.0, 1.0, 2.0];
        let weight = vec![1.0, 0.5, 1.5, -0.5];
        let bias = vec![0.25, -0.5, 0.75, 1.0];
        let input_tensor = context.tensor_from_f32(&input, rows, columns).unwrap();
        let weight_tensor = context.prepare_f32(&weight, 1, columns).unwrap();
        let bias_tensor = context.prepare_f32(&bias, 1, columns).unwrap();
        let actual = context.tensor_to_f32(&context.layernorm_bias(&input_tensor, &weight_tensor, &bias_tensor, 1.0e-6).unwrap());

        let mut expected = Vec::with_capacity(input.len());
        for row in input.chunks_exact(columns) {
            let mean = row.iter().sum::<f32>() / columns as f32;
            let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / columns as f32;
            let inverse_std = (variance + 1.0e-6).sqrt().recip();
            expected.extend(row.iter().enumerate().map(|(column, value)| (value - mean) * inverse_std * weight[column] + bias[column]));
        }
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() <= 0.004, "index={index} actual={actual} expected={expected}");
        }
    }
}
