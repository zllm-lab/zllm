use super::*;
use crate::moe::topk_moe::{RoutedMoeInputs, RoutedMoeWeightsRef, SharedExpertRef};

mod reference;
use reference::{f32_expert as reference_f32_expert_batch, nvfp4_expert as reference_nvfp4_expert_batch, route_cpu};

// 中等 route 批次优先保留逐 route F32 累加；更大的并发批次才用 WMMA grouped。
const MXFP4_DIRECT_ROUTE_LIMIT: usize = 4 * 1024;
static GGUF_MOE_PREFLIGHT_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

impl MoePrefillBackend for RocmContext {
    type MoeAccumulator = RocmTensor;

    fn moe_route(&self, input: &Self::Tensor, router_weight: &Self::Weight, router_bias: &Self::Weight, spec: &crate::moe::topk_moe::TopkMoeSpec) -> Result<MoePrefillRouting, BackendError> {
        if router_weight.rows != spec.num_experts || router_weight.cols != input.cols || router_bias.data().len() != spec.num_experts {
            return Err(compute_error(format!("Rocm MoE router shape 异常: input=[{},{}], weight=[{},{}], bias={}, experts={}", input.rows, input.cols, router_weight.rows, router_weight.cols, router_bias.data().len(), spec.num_experts)));
        }
        if let Some(input_device) = input.device.as_deref() {
            let scoring = match spec.scoring_func {
                crate::moe::topk_moe::ScoringFunc::Softmax => 0,
                crate::moe::topk_moe::ScoringFunc::SigmoidBias => 1,
                crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => 2,
            };
            let weight_device = router_weight.router_resident(self.device_id, ops::hip::options().precise_router)?;
            let bias_device = resident_weight(router_bias, "router bias")?;
            let (expert_ids, weights) =
                ops::hip::try_moe_route_resident_f32(self.device_id, input_device, weight_device, bias_device, input.rows, input.cols, spec.num_experts, spec.top_k, scoring, spec.routed_scaling_factor).map_err(compute_error)?;
            return Ok(MoePrefillRouting { expert_ids, weights, rows: input.rows, top_k: spec.top_k });
        }
        self.require_cpu_reference_fallback("host MoE router")?;
        let input_data = tensor_data(input)?;
        let routes: Vec<_> = input_data.par_chunks_exact(input.cols).map(|row| route_cpu(row, router_weight.data(), router_bias.data(), spec)).collect::<Result<Vec<_>, String>>().map_err(compute_error)?;
        let mut expert_ids = Vec::with_capacity(input.rows * spec.top_k);
        let mut weights = Vec::with_capacity(input.rows * spec.top_k);
        for routing in routes {
            expert_ids.extend(routing.experts);
            weights.extend(routing.weights);
        }
        Ok(MoePrefillRouting { expert_ids, weights, rows: input.rows, top_k: spec.top_k })
    }

    fn moe_route_selected(&self, input: &Self::Tensor, router_weight: &Self::Weight, selected_experts: &[u32], spec: &crate::moe::topk_moe::TopkMoeSpec) -> Result<MoePrefillRouting, BackendError> {
        if spec.scoring_func != crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias {
            return Err(compute_error("Rocm 固定专家路由只支持 sqrt-softplus"));
        }
        if selected_experts.len() != input.rows.checked_mul(spec.top_k).ok_or_else(|| compute_error("ROCm selected router id 数溢出"))? {
            return Err(compute_error(format!("ROCm selected router ids={}，期望 {}x{}", selected_experts.len(), input.rows, spec.top_k)));
        }
        // GPU:预选集合上算 sqrt(softplus) 分数并归一;权重回 host 复用既有 MoePrefillRouting 消费端。
        let input_tensor = f32_tensor(self, input)?;
        let input_device = input_tensor.device.as_deref().ok_or_else(|| compute_error("ROCm selected router input 缺少 device buffer"))?;
        let weight_device = resident_weight(router_weight, "selected router")?;
        let weight_bytes = spec.num_experts.checked_mul(input.cols).and_then(|n| n.checked_mul(4)).ok_or_else(|| compute_error("ROCm selected router weight 字节溢出"))?;
        if weight_device.bytes() != weight_bytes {
            return Err(compute_error("ROCm selected router weight 需要 resident F32"));
        }
        let selected_bytes = unsafe { std::slice::from_raw_parts(selected_experts.as_ptr().cast::<u8>(), selected_experts.len() * 4) };
        let weights = ops::hip::with_tensor_workspace(self.device_id, &[selected_bytes.len()], |workspace| {
            let selected_buffer = workspace.buffer(0);
            selected_buffer.copy_from_host(selected_bytes)?;
            ops::hip::try_moe_router_selected_resident_f32(self.device_id, input_device, weight_device, selected_buffer, input.rows, input.cols, spec.num_experts, spec.top_k, spec.routed_scaling_factor)?.download_f32(selected_experts.len())
        })
        .map_err(|error| compute_error(format!("ROCm selected router kernel 失败: {error}")))?;
        Ok(MoePrefillRouting { expert_ids: selected_experts.to_vec(), weights, rows: input.rows, top_k: spec.top_k })
    }

    fn moe_zeros(&self, rows: usize, cols: usize) -> Result<Self::MoeAccumulator, BackendError> {
        let output = ops::hip::try_moe_zeros_resident_f32(self.device_id, checked_elements(rows, cols, "Rocm MoE output")?).map_err(compute_error)?;
        Ok(device_tensor_f32(output, rows, cols))
    }

    fn moe_gather_rows(&self, input: &Self::Tensor, rows: &[u32]) -> Result<Self::Tensor, BackendError> {
        if let Some(input_device) = input.device.as_deref() {
            let output = ops::hip::try_moe_gather_resident(self.device_id, input_device, input.rows, input.cols, rows).map_err(compute_error)?;
            return Ok(device_tensor_with_dtype(output, rows.len(), input.cols, input.dtype));
        }
        self.require_cpu_reference_fallback("host MoE gather")?;
        let input_data = tensor_data(input)?;
        let mut data = Vec::with_capacity(checked_elements(rows.len(), input.cols, "Rocm MoE gather")?);
        for &row in rows {
            let row = row as usize;
            if row >= input.rows {
                return Err(compute_error(format!("Rocm MoE gather row {row} 越界，rows={}", input.rows)));
            }
            data.extend_from_slice(&input_data[row * input.cols..(row + 1) * input.cols]);
        }
        self.tensor_from_f32(data, rows.len(), input.cols).map_err(compute_error)
    }

    fn moe_gather_rows_batch(&self, input: &Self::Tensor, batches: &[Vec<u32>]) -> Result<Vec<Self::Tensor>, BackendError> {
        batches.iter().map(|rows| self.moe_gather_rows(input, rows)).collect()
    }

    fn moe_scatter_add_rows(&self, output: &mut Self::MoeAccumulator, input: &Self::Tensor, rows: &[u32], weights: &[f32]) -> Result<(), BackendError> {
        if input.rows != rows.len() || rows.len() != weights.len() || input.cols != output.cols {
            return Err(compute_error(format!("Rocm MoE scatter shape 异常: output=[{},{}], input=[{},{}], rows={}, weights={}", output.rows, output.cols, input.rows, input.cols, rows.len(), weights.len())));
        }
        if let (Some(output_device), Some(input_device)) = (output.device.as_deref(), input.device.as_deref()) {
            return ops::hip::try_moe_scatter_add_resident_f32(self.device_id, output_device, output.rows, input_device, input.rows, input.cols, rows, weights).map_err(compute_error);
        }
        self.require_cpu_reference_fallback("host MoE scatter")?;
        let input_data = tensor_data(input)?;
        if output.data.len() != output.rows * output.cols {
            output.data = tensor_data(output)?;
            output.device = None;
        }
        for (source_row, (&target_row, &weight)) in rows.iter().zip(weights).enumerate() {
            let target_row = target_row as usize;
            if target_row >= output.rows {
                return Err(compute_error(format!("Rocm MoE scatter row {target_row} 越界，rows={}", output.rows)));
            }
            let source = &input_data[source_row * input.cols..(source_row + 1) * input.cols];
            let target = &mut output.data[target_row * output.cols..(target_row + 1) * output.cols];
            for (target, source) in target.iter_mut().zip(source) {
                *target += source * weight;
            }
        }
        Ok(())
    }

    fn moe_scatter_add_rows_batch(&self, output: &mut Self::MoeAccumulator, inputs: &[Self::Tensor], rows: &[Vec<u32>], weights: &[Vec<f32>]) -> Result<(), BackendError> {
        if inputs.len() != rows.len() || rows.len() != weights.len() {
            return Err(compute_error(format!("Rocm MoE batch scatter 数量异常: inputs={}, rows={}, weights={}", inputs.len(), rows.len(), weights.len())));
        }
        for ((input, rows), weights) in inputs.iter().zip(rows).zip(weights) {
            self.moe_scatter_add_rows(output, input, rows, weights)?;
        }
        Ok(())
    }

    fn moe_finish(&self, output: Self::MoeAccumulator) -> Result<Self::Tensor, BackendError> {
        if output.device.is_some() { Ok(output) } else { self.upload_cpu_reference("host MoE accumulation", output.data, output.rows, output.cols) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_moe_gather_preserves_rows() {
        let Ok(context) = RocmContext::new(0) else { return };
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bits = values.into_iter().map(|value| half::bf16::from_f32(value).to_bits()).collect();
        let input = context.tensor_from_bf16_bits(bits, 3, 2).expect("上传 BF16 MoE gather 输入");
        let output = context.moe_gather_rows(&input, &[2, 0]).expect("执行 BF16 MoE gather");

        assert_eq!(context.tensor_to_f32(&output).unwrap(), [5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn gguf_packed_column_halves_preserve_each_row() {
        let tensor_type = crate::weight::container::gguf::GgmlType(12);
        let row_bytes = tensor_type.storage_bytes(512).unwrap();
        let half_row_bytes = tensor_type.storage_bytes(256).unwrap();
        let mut bytes = vec![0_u8; row_bytes * 3];
        for (row, data) in bytes.chunks_exact_mut(row_bytes).enumerate() {
            data[..half_row_bytes].fill((row * 2 + 1) as u8);
            data[half_row_bytes..].fill((row * 2 + 2) as u8);
        }

        let [low, high] = gguf_packed_column_halves_bytes(&bytes, 3, 512, tensor_type).unwrap();
        for row in 0..3 {
            assert!(low[row * half_row_bytes..(row + 1) * half_row_bytes].iter().all(|&value| value == (row * 2 + 1) as u8));
            assert!(high[row * half_row_bytes..(row + 1) * half_row_bytes].iter().all(|&value| value == (row * 2 + 2) as u8));
        }
    }

    #[test]
    fn gguf_packed_column_halves_reject_unaligned_shape() {
        let error = gguf_packed_column_halves_bytes(&[], 1, 256, crate::weight::container::gguf::GgmlType(12)).unwrap_err();
        assert!(error.contains("256 列 block"));
    }
}

impl RocmContext {
    #[allow(clippy::too_many_arguments)]
    fn prefill_resident_routed_experts_impl(
        &self,
        spec: &crate::moe::topk_moe::TopkMoeSpec,
        weights: crate::moe::topk_moe::RoutedMoeWeightsRef<'_, RocmWeight>,
        layer: usize,
        experts: &mut RocmPrefillExperts,
        inputs: crate::moe::topk_moe::RoutedMoeInputs<'_, RocmTensor>,
        shared: Option<&RocmTensor>,
        residual: Option<&RocmTensor>,
    ) -> Result<Option<RocmTensor>, BackendError> {
        let router_weight = weights.router;
        let router_bias = weights.bias;
        let route_input = inputs.route;
        let expert_input = inputs.expert;
        if ops::hip::options().trace_moe_segment_rows.is_some() {
            eprintln!("[moe-segment-call] layer={layer} route_rows={} expert_rows={}", route_input.rows, expert_input.rows,);
        }
        if route_input.rows == 0 || route_input.rows != expert_input.rows {
            return Ok(None);
        }
        let (Some(route_device), Some(expert_device)) = (route_input.device.as_deref(), expert_input.device.as_deref()) else {
            eprintln!("[rocm-moe-resident] L{layer} 未命中: route/expert 输入缺 device buffer");
            return Ok(None);
        };
        // 小批量复用 grouped 层级权重与设备路由，只按真实 route 启动两段
        // fused kernel；阈值限制固定工作区，长 prefill 仍走 CSR/WMMA 路径。
        let route_count = route_input.rows.saturating_mul(spec.top_k);
        if matches!(experts.archive, RocmExpertArchive::Fp8(_) | RocmExpertArchive::Fp8Source(_)) {
            if !matches!(spec.activation, Activation::Silu) {
                eprintln!("[rocm-moe-resident] L{layer} 未命中: activation 非 Silu");
                return Ok(None);
            }
            let Some(metas) = experts.fp8_grouped_metas(self.device_id, layer, spec.num_experts)? else {
                eprintln!("[rocm-moe-resident] L{layer} 未命中: FP8 metas 缺失");
                return Ok(None);
            };
            let expert_input = f32_tensor(self, expert_input)?;
            let expert_device = expert_input.device.as_deref().ok_or_else(|| compute_error("FP8 grouped expert input 缺少 device buffer"))?;
            let shared = shared.map(|tensor| f32_tensor(self, tensor)).transpose()?;
            let residual = residual.map(|tensor| f32_tensor(self, tensor)).transpose()?;
            let shared_device = shared.as_ref().and_then(|tensor| tensor.device.as_deref());
            let residual_device = residual.as_ref().and_then(|tensor| tensor.device.as_deref());
            let grouped_epilogue = match (shared_device, residual_device) {
                (Some(shared), Some(residual)) => Some((shared, residual)),
                (None, None) => None,
                _ => None,
            };
            let grouped_compatible = route_input.rows > 1 && ops::hip::options().grouped_wmma && ops::hip::options().grouped_down_route_buffer && shared_device.is_some() == residual_device.is_some();
            let grouped = grouped_compatible.then(|| experts.fp8_grouped(self.device_id, layer, spec.num_experts)).transpose()?;
            let weight_device = router_weight.router_resident(self.device_id, ops::hip::options().precise_router)?;
            let bias_device = resident_weight(router_bias, "router bias")?;
            let scoring = match spec.scoring_func {
                crate::moe::topk_moe::ScoringFunc::Softmax => 0,
                crate::moe::topk_moe::ScoringFunc::SigmoidBias => 1,
                crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => 2,
            };
            let output = ops::hip::with_moe_route_resident_device_f32(
                self.device_id,
                route_device,
                weight_device,
                bias_device,
                route_input.rows,
                route_input.cols,
                spec.num_experts,
                spec.top_k,
                scoring,
                spec.routed_scaling_factor,
                |route_ids, route_weights, route_len| {
                    debug_assert_eq!(route_len, route_count);
                    if let Some(grouped) = grouped.as_deref() {
                        ops::hip::try_ct_grouped_experts_bf16(
                            self.device_id,
                            expert_device,
                            route_input.rows,
                            expert_input.cols,
                            spec.intermediate_size,
                            &[],
                            &[],
                            &[],
                            grouped,
                            Some((route_ids, route_weights, route_len)),
                            grouped_epilogue,
                            None,
                        )
                    } else {
                        ops::hip::try_fp8_grouped_experts_f32(
                            self.device_id,
                            expert_device,
                            route_input.rows,
                            expert_input.cols,
                            spec.intermediate_size,
                            spec.top_k,
                            spec.num_experts,
                            route_ids,
                            route_weights,
                            route_len,
                            &metas,
                            shared_device,
                            residual_device,
                        )
                    }
                },
            )
            .map_err(|error| compute_error(format!("FP8 resident routed experts L{layer}: {error}")))?;
            return Ok(Some(device_tensor_f32(output, expert_input.rows, expert_input.cols)));
        }
        if !ops::hip::options().grouped_wmma || residual.is_some() && (!ops::hip::options().grouped_down_route_buffer || route_input.rows <= 1) {
            return Ok(None);
        }
        if residual.is_none() && matches!(experts.archive, RocmExpertArchive::Mxfp4(_)) {
            let Activation::SiluClamped { limit } = spec.activation else {
                return Ok(None);
            };
            let grouped = experts.mxfp4_grouped(self, layer, spec)?;
            let shared_device = shared.map(|tensor| tensor.device.as_deref().ok_or_else(|| compute_error("MXFP4 decode shared 缺少 device buffer"))).transpose()?;
            let run = |route_ids: &ops::hip::DeviceBuffer, route_weights: &ops::hip::DeviceBuffer, route_len| {
                debug_assert_eq!(route_len, route_count);
                if route_count <= MXFP4_DIRECT_ROUTE_LIMIT {
                    ops::hip::try_mxfp4_decode_experts_f32(
                        self.device_id,
                        expert_device,
                        route_input.rows,
                        spec.top_k,
                        route_ids,
                        route_weights,
                        &grouped.gate_up_packed,
                        &grouped.gate_up_scales,
                        &grouped.down_packed,
                        &grouped.down_scales,
                        expert_input.cols,
                        spec.intermediate_size,
                        spec.num_experts,
                        limit,
                        shared_device,
                    )
                } else {
                    // backlog 已经形成批次时按 expert 压成 device CSR，让跨 session
                    // activation 共享一次权重读取；空闲单路仍保持 direct 低延迟路径。
                    ops::hip::try_mxfp4_grouped_decode_experts_f32(
                        self.device_id,
                        expert_device,
                        route_input.rows,
                        spec.top_k,
                        route_ids,
                        route_weights,
                        &grouped.gate_up_packed,
                        &grouped.gate_up_scales,
                        &grouped.down_packed,
                        &grouped.down_scales,
                        expert_input.cols,
                        spec.intermediate_size,
                        spec.num_experts,
                        limit,
                        shared_device,
                    )
                }
            };
            let output = match weights.selected_experts {
                Some(selected) => {
                    let weight_device = resident_weight(router_weight, "selected router")?;
                    ops::hip::with_moe_route_selected_resident_device_f32(self.device_id, route_device, weight_device, selected, route_input.rows, route_input.cols, spec.num_experts, spec.top_k, spec.routed_scaling_factor, run)
                }
                None => {
                    let weight_device = router_weight.router_resident(self.device_id, ops::hip::options().precise_router)?;
                    let bias_device = resident_weight(router_bias, "router bias")?;
                    let scoring = match spec.scoring_func {
                        crate::moe::topk_moe::ScoringFunc::Softmax => 0,
                        crate::moe::topk_moe::ScoringFunc::SigmoidBias => 1,
                        crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => 2,
                    };
                    ops::hip::with_moe_route_resident_device_f32(self.device_id, route_device, weight_device, bias_device, route_input.rows, route_input.cols, spec.num_experts, spec.top_k, scoring, spec.routed_scaling_factor, run)
                }
            }
            .map_err(|error| compute_error(format!("MXFP4 resident routed experts L{layer}: {error}")))?;
            return Ok(Some(device_tensor_f32(output, expert_input.rows, expert_input.cols)));
        }
        if !matches!(&spec.activation, Activation::Silu) {
            return Ok(None);
        }
        // WMMA kernel 吃设备侧 BF16/F32(F32 由 wrapper 内 cast 转换)；
        // host-only 或 MTP draft 的非常规形态退逐专家路径。
        if !matches!(expert_input.dtype, RocmTensorDType::Bf16 | RocmTensorDType::F32) {
            return Ok(None);
        }
        let epilogue_device = match (shared, residual) {
            (Some(shared), Some(residual)) => {
                let (Some(shared), Some(residual)) = (shared.device.as_deref(), residual.device.as_deref()) else {
                    return Ok(None);
                };
                Some((shared, residual))
            }
            (None, None) => None,
            _ => return Ok(None),
        };
        // GGUF 分支：小批次(route≤64)走 fused 两段 kernel(对标 CT decode fused，
        // 常驻 workspace 零分配)；大批次(≤512)走 WMMA grouped；更大退逐专家。
        if matches!(experts.archive, RocmExpertArchive::Gguf(_)) {
            // WMMA kernel 吃设备侧 BF16/F32(F32 由 wrapper 内 cast 转换)；
            // host-only 或 MTP draft 的非常规形态退逐专家路径。
            let device_input_ok = matches!(expert_input.dtype, RocmTensorDType::Bf16 | RocmTensorDType::F32);
            if epilogue_device.is_some() || !device_input_ok || !matches!(&spec.activation, Activation::Silu) {
                return Ok(None);
            }
            let weight_device = router_weight.router_resident(self.device_id, ops::hip::options().precise_router)?;
            let bias_device = resident_weight(router_bias, "router bias")?;
            let scoring = match spec.scoring_func {
                crate::moe::topk_moe::ScoringFunc::Softmax => 0,
                crate::moe::topk_moe::ScoringFunc::SigmoidBias => 1,
                crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => 2,
            };
            let metas = experts.gguf_routed_grouped(self.device_id, layer)?;
            let route_count = route_input.rows * spec.top_k;
            let output = if route_count <= 64 {
                let d_output = ops::hip::DeviceBuffer::allocate_reusable(self.device_id, route_input.rows.saturating_mul(expert_input.cols) * 4).map_err(compute_error)?;
                ops::hip::with_moe_route_resident_device_f32(
                    self.device_id,
                    route_device,
                    weight_device,
                    bias_device,
                    route_input.rows,
                    route_input.cols,
                    spec.num_experts,
                    spec.top_k,
                    scoring,
                    spec.routed_scaling_factor,
                    |route_ids, route_weights, route_len| {
                        ops::hip::try_gguf_fused_decode_experts(self.device_id, expert_device, route_input.rows, expert_input.cols, spec.intermediate_size, spec.top_k, route_ids, route_weights, route_len, metas, &d_output)
                    },
                )
                .map_err(compute_error)?;
                let output = d_output;
                output
            } else {
                ops::hip::with_moe_route_resident_device_f32(
                    self.device_id,
                    route_device,
                    weight_device,
                    bias_device,
                    route_input.rows,
                    route_input.cols,
                    spec.num_experts,
                    spec.top_k,
                    scoring,
                    spec.routed_scaling_factor,
                    |route_ids, route_weights, route_len| {
                        ops::hip::try_gguf_grouped_wmma_experts(self.device_id, expert_device, route_input.rows, expert_input.cols, spec.intermediate_size, spec.top_k, route_ids, route_weights, route_len, metas, spec.num_experts)
                    },
                )
                .map_err(compute_error)?
            };
            return Ok(Some(device_tensor_f32(output, expert_input.rows, expert_input.cols)));
        }
        let Some(resident) = experts.resident_ct_layer(self.device_id, layer, spec.num_experts) else {
            return Ok(None);
        };
        let grouped_w4 = resident.iter().map(|expert| Some(ops::hip::CtGroupedExpertRef { gate: grouped_w4(&expert.gate)?, up: grouped_w4(&expert.up)?, down: grouped_w4(&expert.down)? })).collect::<Option<Vec<_>>>();
        let grouped_w8 = (route_input.rows == 1 && epilogue_device.is_none())
            .then(|| resident.iter().map(|expert| Some(ops::hip::CtGroupedExpertRef { gate: grouped_w8(&expert.gate)?, up: grouped_w8(&expert.up)?, down: grouped_w8(&expert.down)? })).collect::<Option<Vec<_>>>())
            .flatten();
        if grouped_w4.is_none() && grouped_w8.is_none() {
            return Ok(None);
        }
        let scoring = match spec.scoring_func {
            crate::moe::topk_moe::ScoringFunc::Softmax => 0,
            crate::moe::topk_moe::ScoringFunc::SigmoidBias => 1,
            crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => 2,
        };
        let trace_segment_rows = ops::hip::options().trace_moe_segment_rows.and_then(|(target_layer, segment_rows)| (matches!(target_layer, usize::MAX) || target_layer == layer).then_some(segment_rows).filter(|&rows| rows > 0));
        let trace_segment_hashes = |label: &str, tensor: &RocmTensor| -> Result<(), BackendError> {
            let Some(segment_rows) = trace_segment_rows else { return Ok(()) };
            // 全层诊断只针对 decode 小批次，避免把整段 prefill 下载并刷入日志。
            if tensor.rows > 64 || tensor.rows % segment_rows != 0 {
                return Ok(());
            }
            let values = tensor_data(tensor)?;
            let segment_elements = segment_rows * tensor.cols;
            for (session, segment) in values.chunks_exact(segment_elements).enumerate() {
                let hash = segment.iter().fold(0xcbf29ce484222325_u64, |hash, value| (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3));
                eprintln!("[moe-segment-hash] layer={layer} part={label} session={session} rows={segment_rows} hash={hash:016x}",);
            }
            Ok(())
        };
        trace_segment_hashes("input", route_input)?;
        let weight_device = router_weight.router_resident(self.device_id, ops::hip::options().precise_router)?;
        let bias_device = resident_weight(router_bias, "router bias")?;
        let output = ops::hip::with_moe_route_resident_device_f32(
            self.device_id,
            route_device,
            weight_device,
            bias_device,
            route_input.rows,
            route_input.cols,
            spec.num_experts,
            spec.top_k,
            scoring,
            spec.routed_scaling_factor,
            |route_ids, route_weights, route_len| match &grouped_w4 {
                Some(grouped) => ops::hip::try_ct_grouped_experts_bf16(
                    self.device_id,
                    expert_device,
                    expert_input.rows,
                    expert_input.cols,
                    spec.intermediate_size,
                    &[],
                    &[],
                    &[],
                    grouped,
                    Some((route_ids, route_weights, route_len)),
                    epilogue_device,
                    None,
                ),
                None => ops::hip::try_ct_w8_decode_experts_bf16(self.device_id, expert_device, expert_input.cols, spec.intermediate_size, route_ids, route_weights, route_len, grouped_w8.as_ref().expect("W4/W8 grouped 已检查")),
            },
        )
        .map_err(compute_error)?;
        if ops::hip::options().kernel_sync {
            ops::hip::synchronize_device(self.device_id, &format!("L{layer} ROCm grouped expert synchronize")).map_err(compute_error)?;
        }
        if ops::hip::options().debug_finite {
            ops::hip::try_validate_finite_resident_range_f32(self.device_id, &output, 0, expert_input.cols).map_err(|error| compute_error(format!("L{layer} ROCm resident routed expert 包含非有限值: {error}")))?;
        }
        let output = device_tensor_f32(output, expert_input.rows, expert_input.cols);
        trace_segment_hashes(if residual.is_some() { "final" } else { "routed" }, &output)?;
        Ok(Some(output))
    }

    fn cooperative_moe_add(
        &self,
        spec: &TopkMoeSpec,
        weights: RoutedMoeWeightsRef<'_, RocmWeight>,
        shared_experts: &[SharedExpertRef<'_, RocmWeight>],
        layer: usize,
        experts: &mut RocmPrefillExperts,
        inputs: RoutedMoeInputs<'_, RocmTensor>,
        residual: &RocmTensor,
    ) -> Result<RocmTensor, BackendError> {
        experts.cooperative_peer.ok_or_else(|| compute_error("ROCm cooperative expert peer 缺失"))?;
        if inputs.route.rows == 0 || inputs.route.rows != inputs.expert.rows || residual.rows != inputs.expert.rows {
            return Err(compute_error(format!("ROCm cooperative experts 行数不一致，L{layer} rows={}/{}/{}", inputs.route.rows, inputs.expert.rows, residual.rows,)));
        }
        if !ops::hip::options().decode_moe_fused
            || !matches!(&experts.archive, RocmExpertArchive::Ct(_) | RocmExpertArchive::Gguf(_))
            || !matches!(spec.activation, Activation::Silu)
            || spec.num_shared_experts != 1
            || shared_experts.len() != 1
            || spec.shared_intermediate_size != spec.intermediate_size
            || inputs.route.cols != inputs.expert.cols
            || residual.cols != inputs.expert.cols
            || !spec.intermediate_size.is_multiple_of(2)
            || !inputs.expert.cols.is_multiple_of(2)
            || !matches!(inputs.expert.dtype, RocmTensorDType::Bf16 | RocmTensorDType::F32)
        {
            return Err(compute_error(format!("ROCm cooperative experts L{layer} 不满足 fused TP 前提")));
        }
        let shared = &shared_experts[0];
        if shared.output_gate.is_some()
            || shared.gate.rows != spec.intermediate_size
            || shared.up.rows != spec.intermediate_size
            || shared.down.rows != inputs.expert.cols
            || shared.gate.cols != inputs.expert.cols
            || shared.up.cols != inputs.expert.cols
            || shared.down.cols != spec.intermediate_size
        {
            return Err(compute_error(format!("ROCm cooperative shared expert L{layer} shape 不兼容")));
        }
        let (Some(route_device), Some(_), Some(_)) = (inputs.route.device.as_deref(), inputs.expert.device.as_deref(), residual.device.as_deref()) else {
            return Err(compute_error("ROCm cooperative experts 缺少 device input"));
        };
        // 双卡 consumer 会在同一闭包中切换 current device；单卡 deferred
        // route workspace 无法在闭包返回后可靠地给 owner stream 记录 event。
        // owned route buffer 保活到两卡提交完成，析构仍排在 owner join 之后。
        let route = match weights.selected_experts {
            Some(selected) => {
                let weight_device = resident_weight(weights.router, "selected router")?;
                ops::hip::try_moe_route_selected_resident_device_f32(self.device_id, route_device, weight_device, selected, inputs.route.rows, inputs.route.cols, spec.num_experts, spec.top_k, spec.routed_scaling_factor)
            }
            None => {
                let weight_device = weights.router.router_resident(self.device_id, ops::hip::options().precise_router)?;
                let bias_device = resident_weight(weights.bias, "router bias")?;
                let scoring = match spec.scoring_func {
                    crate::moe::topk_moe::ScoringFunc::Softmax => 0,
                    crate::moe::topk_moe::ScoringFunc::SigmoidBias => 1,
                    crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => 2,
                };
                ops::hip::try_moe_route_resident_device_f32(self.device_id, route_device, weight_device, bias_device, inputs.route.rows, inputs.route.cols, spec.num_experts, spec.top_k, scoring, spec.routed_scaling_factor)
            }
        }
        .map_err(|error| compute_error(format!("ROCm cooperative device route L{layer}: {error}")))?;
        let expected_routes = inputs.route.rows.checked_mul(spec.top_k).ok_or_else(|| compute_error("ROCm cooperative route 数溢出"))?;
        if route.len != expected_routes {
            return Err(compute_error(format!("ROCm cooperative route 数异常: {}/{}", route.len, expected_routes)));
        }
        self.cooperative_moe_device_add(spec, layer, experts, inputs.expert, residual, Arc::new(route.expert_ids), Arc::new(route.weights), route.len).map_err(|error| compute_error(format!("ROCm cooperative L{layer}: {error:?}")))
    }

    #[allow(clippy::too_many_arguments)]
    fn cooperative_moe_device_add(
        &self,
        spec: &TopkMoeSpec,
        layer: usize,
        experts: &RocmPrefillExperts,
        input: &RocmTensor,
        residual: &RocmTensor,
        route_ids: Arc<ops::hip::DeviceBuffer>,
        route_weights: Arc<ops::hip::DeviceBuffer>,
        route_count: usize,
    ) -> Result<RocmTensor, BackendError> {
        let peer = experts.cooperative_peer.ok_or_else(|| compute_error("ROCm cooperative expert peer 缺失"))?;
        let is_gguf = matches!(experts.archive, RocmExpertArchive::Gguf(_));
        let local = (!is_gguf).then(|| experts.cooperative_resident_partition(self, layer, spec.num_experts, peer.local_partition)).transpose()?;
        let remote = (!is_gguf).then(|| experts.cooperative_resident_partition(&peer.context, layer, spec.num_experts, 1 - peer.local_partition)).transpose()?;
        let local_shared = (!is_gguf).then(|| experts.cooperative_resident_shared(self, layer, spec.num_experts, peer.local_partition)).transpose()?;
        let remote_shared = (!is_gguf).then(|| experts.cooperative_resident_shared(&peer.context, layer, spec.num_experts, 1 - peer.local_partition)).transpose()?;
        let local_grouped = local.as_deref().map(grouped_w4_experts).transpose()?;
        let remote_grouped = remote.as_deref().map(grouped_w4_experts).transpose()?;
        let local_shared_grouped = local_shared.as_ref().map(|resident| grouped_w4_experts(std::slice::from_ref(resident))).transpose()?;
        let remote_shared_grouped = remote_shared.as_ref().map(|resident| grouped_w4_experts(std::slice::from_ref(resident))).transpose()?;
        // GGUF pointer table 与常驻权重同生命周期；preload 已上传并保存，
        // decode 每层只借用，不再重建 Arc/meta/identity 临时容器。
        let local_gguf = is_gguf.then(|| experts.cooperative_gguf_grouped(self.device_id, layer)).transpose()?;
        let remote_gguf = is_gguf.then(|| experts.cooperative_gguf_grouped(peer.context.device_id, layer)).transpose()?;

        // scheduler 已为本次 submission 选择 latency/default 或 background
        // stream。cooperative owner 与 peer 都在各自默认流执行；进入与退出时
        // 分别用同卡 event 桥接 stage stream，不能仅修改 TLS stream 映射。
        ops::hip::set_device(self.device_id).map_err(compute_error)?;
        let owner_stream = ops::hip::active_compute_stream() as usize;
        ops::hip::order_stream_after(self.device_id, owner_stream, 0).map_err(compute_error)?;
        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        // 三个小输入均先稳定再排入 peer stream；源 buffer 统一由 owner stage
        // completion 保活，整个路由与传输路径不再访问 host。
        let stable_route_ids = route_ids;
        let stable_route_weights = route_weights;
        let stable_input = self.tensor_to_stable_deferred(input.clone())?;
        let source = stable_input.device.as_ref().ok_or_else(|| compute_error("ROCm cooperative stable input 缺少 device buffer"))?;
        // 每个双卡组只有 owner 拥有 pipeline stage；peer 的 legacy stream 专供
        // 本组 MoE。显式 source/destination event 串起跨卡依赖，同时复用 HIP
        // 默认流成熟的 buffer 生命周期，避免临时 dedicated stream 脱离 stage 回收。
        let profile_pair = ops::hip::device_profile_enabled();
        peer.context.activate().map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_begin(peer.context.device_id, "glm_pair_moe_peer").map_err(compute_error)?;
        }
        let [peer_route_ids, peer_route_weights, peer_input]: [ops::hip::DeviceBuffer; 3] =
            ops::hip::DeviceBuffer::copy_stable_group_to_device_ordered_async_retained_by(&[stable_route_ids.clone(), stable_route_weights.clone(), source.clone()], peer.context.device_id, self.device_id)
                .map_err(|error| compute_error(format!("L{layer} route/input owner->peer: {error}")))?
                .try_into()
                .map_err(|_| compute_error(format!("L{layer} cooperative P2P 输入数量异常")))?;
        let peer_input = device_tensor_with_dtype(peer_input, input.rows, input.cols, input.dtype);

        let peer_input_device = peer_input.device.as_deref().ok_or_else(|| compute_error("ROCm cooperative peer input 缺少 device buffer"))?;
        let half_intermediate = spec.intermediate_size / 2;
        // 标准 TP：gate/up 行分片后，down 按 K 列分片，各卡直接把本地
        // activation 投影为完整 hidden partial。peer 先完整排队，使其 down
        // 与 owner 的 gate/down/shared 真正并行；不再 all-gather activation。
        let peer_shared_stream = ops::hip::cooperative_shared_stream(peer.context.device_id).map_err(compute_error)?;
        ops::hip::order_stream_after(peer.context.device_id, 0, peer_shared_stream).map_err(compute_error)?;
        ops::hip::activate_compute_stream(peer.context.device_id, peer_shared_stream).map_err(compute_error)?;
        let peer_shared_output = if let Some(grouped) = &remote_shared_grouped {
            ops::hip::try_ct_cooperative_shared_bf16(peer.context.device_id, peer_input_device, input.rows, input.cols, half_intermediate, &grouped[0])
        } else {
            ops::hip::try_gguf_cooperative_shared_bf16(peer.context.device_id, peer_input_device, input.rows, input.cols, half_intermediate, remote_gguf.expect("GGUF peer metas").shared.as_ref().expect("cooperative GGUF shared metas"))
        }
        .map_err(compute_error)?;
        ops::hip::activate_compute_stream(peer.context.device_id, 0).map_err(compute_error)?;
        let peer_output = if let Some(grouped) = &remote_grouped {
            ops::hip::try_ct_cooperative_routed_bf16(peer.context.device_id, peer_input_device, input.rows, input.cols, half_intermediate, &peer_route_ids, &peer_route_weights, route_count, spec.top_k, grouped, input.cols)
        } else {
            ops::hip::try_gguf_cooperative_routed_bf16(
                peer.context.device_id,
                peer_input_device,
                input.rows,
                input.cols,
                half_intermediate,
                &peer_route_ids,
                &peer_route_weights,
                route_count,
                spec.top_k,
                &remote_gguf.expect("GGUF peer metas").routed,
                spec.num_experts,
            )
        }
        .map_err(compute_error)?;
        ops::hip::order_stream_after(peer.context.device_id, peer_shared_stream, 0).map_err(compute_error)?;
        // peer 直接把 routed/shared 融合压成稳定 BF16；不生成 F32 combined，
        // 也不再为跨卡生命周期额外复制一次 12MiB BF16。
        let stable_peer_combined = Arc::new(ops::hip::try_ct_cooperative_combine_partial_bf16(peer.context.device_id, &peer_output, &peer_shared_output, input.rows * input.cols).map_err(compute_error)?);
        if profile_pair {
            ops::hip::device_profile_scope_end(peer.context.device_id).map_err(compute_error)?;
        }

        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_begin(self.device_id, "glm_pair_moe_owner").map_err(compute_error)?;
        }
        let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm cooperative local input 缺少 device buffer"))?;
        let local_shared_stream = ops::hip::cooperative_shared_stream(self.device_id).map_err(compute_error)?;
        ops::hip::order_stream_after(self.device_id, 0, local_shared_stream).map_err(compute_error)?;
        ops::hip::activate_compute_stream(self.device_id, local_shared_stream).map_err(compute_error)?;
        let local_shared_output = if let Some(grouped) = &local_shared_grouped {
            ops::hip::try_ct_cooperative_shared_bf16(self.device_id, input_device, input.rows, input.cols, half_intermediate, &grouped[0])
        } else {
            ops::hip::try_gguf_cooperative_shared_bf16(self.device_id, input_device, input.rows, input.cols, half_intermediate, local_gguf.expect("GGUF local metas").shared.as_ref().expect("cooperative GGUF shared metas"))
        }
        .map_err(compute_error)?;
        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        let local_output = if let Some(grouped) = &local_grouped {
            ops::hip::try_ct_cooperative_routed_bf16(self.device_id, input_device, input.rows, input.cols, half_intermediate, &stable_route_ids, &stable_route_weights, route_count, spec.top_k, grouped, input.cols)
        } else {
            ops::hip::try_gguf_cooperative_routed_bf16(
                self.device_id,
                input_device,
                input.rows,
                input.cols,
                half_intermediate,
                &stable_route_ids,
                &stable_route_weights,
                route_count,
                spec.top_k,
                &local_gguf.expect("GGUF local metas").routed,
                spec.num_experts,
            )
        }
        .map_err(compute_error)?;
        ops::hip::order_stream_after(self.device_id, local_shared_stream, 0).map_err(compute_error)?;
        let local_combined = ops::hip::try_add_resident_f32(self.device_id, &local_output, &local_shared_output, input.rows * input.cols, 1.0).map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
            ops::hip::device_profile_scope_begin(self.device_id, "glm_pair_moe_join").map_err(compute_error)?;
        }
        let peer_combined_on_owner = stable_peer_combined.copy_stable_to_device_ordered_async_retained_by(self.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} output peer->owner: {error}")))?;

        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        let residual = f32_tensor(self, residual)?;
        let residual_device = residual.device.as_deref().ok_or_else(|| compute_error("ROCm cooperative residual 缺少 F32 device buffer"))?;
        let output = ops::hip::try_ct_cooperative_partial_join_f32(self.device_id, &local_combined, &peer_combined_on_owner, residual_device, input.rows, input.cols).map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
        }
        ops::hip::order_stream_after(self.device_id, 0, owner_stream).map_err(compute_error)?;
        ops::hip::activate_compute_stream(self.device_id, owner_stream).map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }
}

impl ExpertPrefillBackend for RocmContext {
    type PrefillExperts = RocmPrefillExperts;

    fn supports_cooperative_mla_prefill(&self, layer: usize, experts: &Self::PrefillExperts) -> bool {
        !ops::hip::options().prefill_attention_cpu && experts.cooperative_mla.contains_key(&layer)
    }

    fn cooperative_dsa_append_keys_layernorm_rope(
        &self,
        layer: usize,
        experts: &Self::PrefillExperts,
        state: &mut <Self as DecodeBackend>::DsaState,
        position: usize,
        keys: &Self::Tensor,
        norm_weight: &Self::Weight,
        norm_bias: &Self::Weight,
        eps: f32,
        cosine: &[f32],
        sine: &[f32],
        spec: &crate::attention::dsa::DsaSpec,
    ) -> Result<bool, BackendError> {
        if spec.kpool != 0 || !self.supports_cooperative_mla_prefill(layer, experts) || !super::attention::supports_dsa_fused_prologue(state, keys, norm_weight, norm_bias) {
            return Ok(false);
        }
        let Some(norm_weight) = norm_weight.resident().map(Arc::as_ref) else { return Ok(false) };
        let Some(norm_bias) = norm_bias.resident().map(Arc::as_ref) else { return Ok(false) };
        let (peer, _) = experts.cooperative_mla_layer(layer)?;
        state.append_cooperative_layernorm_rope(self, &peer, layer, position, keys, norm_weight, norm_bias, eps, spec.rope_dim, spec.rotary_layout, cosine, sine)?;
        Ok(true)
    }

    fn cooperative_dsa_select_prefill(
        &self,
        layer: usize,
        experts: &Self::PrefillExperts,
        state: &mut <Self as DecodeBackend>::DsaState,
        query: &Self::Tensor,
        head_weights: &Self::Tensor,
        spec: &crate::attention::dsa::DsaSpec,
    ) -> Result<bool, BackendError> {
        if spec.kpool != 0 || !self.supports_cooperative_mla_prefill(layer, experts) {
            return Ok(false);
        }
        let (peer, _) = experts.cooperative_mla_layer(layer)?;
        state.select_prefill_cooperative(self, &peer, layer, query, head_weights)
    }

    fn cooperative_dsa_project_select_prefill(
        &self,
        layer: usize,
        experts: &Self::PrefillExperts,
        state: &mut <Self as DecodeBackend>::DsaState,
        q_lora: &Self::Tensor,
        owner_wq_b: &Self::Weight,
        head_weights: &Self::Tensor,
        position: usize,
        cosine: &[f32],
        sine: &[f32],
        spec: &crate::attention::dsa::DsaSpec,
    ) -> Result<bool, BackendError> {
        if spec.kpool != 0 || !self.supports_cooperative_mla_prefill(layer, experts) {
            return Ok(false);
        }
        let Some(peer_wq_b) = experts.cooperative_dsa_wq_b.get(&layer) else { return Ok(false) };
        let (peer, _) = experts.cooperative_mla_layer(layer)?;
        state.select_prefill_cooperative_projected(self, &peer, layer, q_lora, owner_wq_b, peer_wq_b, head_weights, position, cosine, sine, spec)
    }

    fn cooperative_mla_prefill_add(
        &self,
        layer: usize,
        experts: &Self::PrefillExperts,
        normalized_q_lora: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        residual: &Self::Tensor,
        cache: Option<&mut Self::Cache>,
        dsa_state: Option<&<Self as DecodeBackend>::DsaState>,
        position: usize,
        cosine: &[f32],
        sine: &[f32],
        mla: &crate::attention::mla::MlaSpec,
        dsa: &crate::attention::dsa::DsaSpec,
    ) -> Result<Self::Tensor, BackendError> {
        self.cooperative_mla_prefill_add_impl(layer, experts, normalized_q_lora, latent, k_rope, residual, cache, dsa_state, position, cosine, sine, mla, dsa)
    }

    fn prefill_resident_moe_add(
        &self,
        spec: &crate::moe::topk_moe::TopkMoeSpec,
        weights: crate::moe::topk_moe::RoutedMoeWeightsRef<'_, Self::Weight>,
        shared_experts: &[crate::moe::topk_moe::SharedExpertRef<'_, Self::Weight>],
        layer: usize,
        experts: &mut Self::PrefillExperts,
        inputs: crate::moe::topk_moe::RoutedMoeInputs<'_, Self::Tensor>,
        residual: &Self::Tensor,
    ) -> Result<Option<Self::Tensor>, BackendError> {
        let route_input = inputs.route;
        let expert_input = inputs.expert;
        if experts.cooperative_peer.is_some() {
            return self.cooperative_moe_add(spec, weights, shared_experts, layer, experts, inputs, residual).map(Some);
        }
        if route_input.rows <= 8 && GGUF_MOE_PREFLIGHT_LOGGED.set(()).is_ok() {
            let shared = shared_experts.first();
            eprintln!(
                "[gguf-moe-fused-preflight] device={} layer={layer} enabled={} archive={} activation={} shared_count={}/{} shared_width={}/{} route={:?} expert={:?} residual={:?} rows={}/{}/{} cols={}/{}/{} selected={} shared_gate_quantized={} shared_up_quantized={} shared_down_quantized={}",
                self.device_id,
                ops::hip::options().decode_moe_fused,
                matches!(experts.archive, RocmExpertArchive::Gguf(_)),
                matches!(spec.activation, Activation::Silu),
                shared_experts.len(),
                spec.num_shared_experts,
                spec.shared_intermediate_size,
                spec.intermediate_size,
                route_input.dtype,
                expert_input.dtype,
                residual.dtype,
                route_input.rows,
                expert_input.rows,
                residual.rows,
                route_input.cols,
                expert_input.cols,
                residual.cols,
                weights.selected_experts.is_some(),
                shared.is_some_and(|shared| shared.gate.quantized().is_some()),
                shared.is_some_and(|shared| shared.up.quantized().is_some()),
                shared.is_some_and(|shared| shared.down.quantized().is_some()),
            );
        }
        if !ops::hip::options().decode_moe_fused
            || !matches!(experts.archive, RocmExpertArchive::Ct(_) | RocmExpertArchive::Gguf(_))
            || !matches!(spec.activation, Activation::Silu)
            || spec.num_shared_experts != 1
            || shared_experts.len() != 1
            || spec.shared_intermediate_size != spec.intermediate_size
            || route_input.rows == 0
            || route_input.rows > 8
            || route_input.rows != expert_input.rows
            || residual.rows != expert_input.rows
            || residual.cols != expert_input.cols
            || !matches!(expert_input.dtype, RocmTensorDType::Bf16 | RocmTensorDType::F32)
        {
            return Ok(None);
        }
        let shared = &shared_experts[0];
        if shared.output_gate.is_some()
            || shared.gate.rows != spec.intermediate_size
            || shared.up.rows != spec.intermediate_size
            || shared.down.rows != expert_input.cols
            || shared.gate.cols != expert_input.cols
            || shared.up.cols != expert_input.cols
            || shared.down.cols != spec.intermediate_size
        {
            return Ok(None);
        }
        let (Some(route_device), Some(expert_device), Some(residual_device)) = (route_input.device.as_deref(), expert_input.device.as_deref(), residual.device.as_deref()) else {
            return Ok(None);
        };
        if matches!(experts.archive, RocmExpertArchive::Gguf(_)) {
            if weights.selected_experts.is_some() || residual.dtype != RocmTensorDType::F32 {
                return Ok(None);
            }
            let Some(shared_grouped) = experts.gguf_shared_grouped(self.device_id, layer, shared)? else {
                return Ok(None);
            };
            let routed_grouped = experts.gguf_routed_grouped(self.device_id, layer)?;
            let route_count = route_input.rows.checked_mul(spec.top_k).ok_or_else(|| compute_error("ROCm integrated GGUF MoE route 数溢出"))?;
            self.profile_device_operator("moe_shared")?;
            let shared_output = ops::hip::try_gguf_cooperative_shared_bf16(self.device_id, expert_device, expert_input.rows, expert_input.cols, spec.intermediate_size, &shared_grouped).map_err(compute_error)?;
            self.profile_device_operator("moe_routed")?;
            let routed_output = ops::hip::DeviceBuffer::allocate_reusable(self.device_id, expert_input.rows.saturating_mul(expert_input.cols) * 4).map_err(compute_error)?;
            let weight_device = weights.router.router_resident(self.device_id, ops::hip::options().precise_router)?;
            let bias_device = resident_weight(weights.bias, "router bias")?;
            let scoring = match spec.scoring_func {
                crate::moe::topk_moe::ScoringFunc::Softmax => 0,
                crate::moe::topk_moe::ScoringFunc::SigmoidBias => 1,
                crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => 2,
            };
            ops::hip::with_moe_route_resident_device_f32(
                self.device_id,
                route_device,
                weight_device,
                bias_device,
                route_input.rows,
                route_input.cols,
                spec.num_experts,
                spec.top_k,
                scoring,
                spec.routed_scaling_factor,
                |route_ids, route_weights, route_len| {
                    if route_len != route_count {
                        return Err(format!("ROCm integrated GGUF MoE route 数异常: {route_len}/{route_count}"));
                    }
                    ops::hip::try_gguf_fused_decode_experts(self.device_id, expert_device, expert_input.rows, expert_input.cols, spec.intermediate_size, spec.top_k, route_ids, route_weights, route_len, routed_grouped, &routed_output)
                },
            )
            .map_err(|error| compute_error(format!("ROCm integrated GGUF MoE L{layer}: {error}")))?;
            self.profile_device_operator("moe_epilogue")?;
            let combined = ops::hip::try_add_resident_f32(self.device_id, &routed_output, &shared_output, expert_input.rows * expert_input.cols, 1.0).map_err(compute_error)?;
            let output = ops::hip::try_add_resident_f32(self.device_id, residual_device, &combined, expert_input.rows * expert_input.cols, 1.0).map_err(compute_error)?;
            return Ok(Some(device_tensor_f32(output, expert_input.rows, expert_input.cols)));
        }
        let Some(resident) = experts.resident_ct_layer(self.device_id, layer, spec.num_experts) else {
            return Ok(None);
        };
        let Some(grouped) = resident.iter().map(|expert| Some(ops::hip::CtGroupedExpertRef { gate: grouped_w4(&expert.gate)?, up: grouped_w4(&expert.up)?, down: grouped_w4(&expert.down)? })).collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let Some(shared_grouped) = (|| Some(ops::hip::CtGroupedExpertRef { gate: grouped_w4(shared.gate)?, up: grouped_w4(shared.up)?, down: grouped_w4(shared.down)? }))() else {
            return Ok(None);
        };
        let route_count = route_input.rows.checked_mul(spec.top_k).ok_or_else(|| compute_error("ROCm integrated MoE route 数溢出"))?;
        if route_count > grouped.len() {
            return Ok(None);
        }
        let graph_eligible = ops::hip::options().decode_graph
            && route_input.rows == 1
            && weights.selected_experts.is_none()
            && !ops::hip::options().kernel_sync
            && !ops::hip::options().kernel_profile
            && !ops::hip::options().trace_moe_route
            && !ops::hip::options().debug_finite
            && route_device.bytes() == expert_input.cols * RocmTensorDType::F32.element_bytes()
            && residual_device.bytes() == expert_input.cols * RocmTensorDType::F32.element_bytes();
        if graph_eligible {
            let key = (self.device_id, layer);
            if !experts.ct_moe_graphs.contains_key(&key) {
                let weight_device = weights.router.router_resident(self.device_id, ops::hip::options().precise_router)?;
                let bias_device = resident_weight(weights.bias, "router bias")?;
                let scoring = match spec.scoring_func {
                    crate::moe::topk_moe::ScoringFunc::Softmax => 0,
                    crate::moe::topk_moe::ScoringFunc::SigmoidBias => 1,
                    crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => 2,
                };
                let built = RocmCtMoeGraph::build(
                    self.device_id,
                    route_device.bytes(),
                    expert_device.bytes(),
                    residual_device.bytes(),
                    expert_input.cols,
                    spec.intermediate_size,
                    spec.num_experts,
                    spec.top_k,
                    scoring,
                    spec.routed_scaling_factor,
                    weight_device,
                    bias_device,
                    &grouped,
                    &shared_grouped,
                );
                match built {
                    Ok(graph) => {
                        eprintln!("[rocm-decode-graph] device={} layer={layer} nodes={} status=ready", self.device_id, graph.graph.node_count());
                        experts.ct_moe_graphs.insert(key, RocmCtMoeGraphState::Ready(graph));
                    }
                    Err(error) => {
                        eprintln!("[rocm-decode-graph] device={} layer={layer} status=disabled error={error}", self.device_id);
                        experts.ct_moe_graphs.insert(key, RocmCtMoeGraphState::Disabled);
                    }
                }
            }
            if let Some(RocmCtMoeGraphState::Ready(graph)) = experts.ct_moe_graphs.get(&key) {
                match graph.launch(route_device, expert_device, residual_device) {
                    Ok(output) => return Ok(Some(device_tensor_with_arc(output, 1, expert_input.cols, RocmTensorDType::F32))),
                    Err(error) => {
                        eprintln!("[rocm-decode-graph] device={} layer={layer} status=replay-disabled error={error}", self.device_id);
                    }
                }
            }
            if matches!(experts.ct_moe_graphs.get(&key), Some(RocmCtMoeGraphState::Ready(_))) {
                experts.ct_moe_graphs.insert(key, RocmCtMoeGraphState::Disabled);
            }
        }
        let run = |route_ids: &ops::hip::DeviceBuffer, route_weights: &ops::hip::DeviceBuffer, route_len| {
            if route_len != route_count {
                return Err(format!("ROCm integrated MoE route 数异常: {route_len}/{route_count}"));
            }
            ops::hip::try_ct_grouped_experts_bf16(
                self.device_id,
                expert_device,
                expert_input.rows,
                expert_input.cols,
                spec.intermediate_size,
                &[],
                &[],
                &[],
                &grouped,
                Some((route_ids, route_weights, route_len)),
                None,
                Some((&shared_grouped, residual_device)),
            )
        };
        let output = match weights.selected_experts {
            Some(selected) => {
                let weight_device = resident_weight(weights.router, "selected router")?;
                ops::hip::with_moe_route_selected_resident_device_f32(self.device_id, route_device, weight_device, selected, route_input.rows, route_input.cols, spec.num_experts, spec.top_k, spec.routed_scaling_factor, run)
            }
            None => {
                let weight_device = weights.router.router_resident(self.device_id, ops::hip::options().precise_router)?;
                let bias_device = resident_weight(weights.bias, "router bias")?;
                let scoring = match spec.scoring_func {
                    crate::moe::topk_moe::ScoringFunc::Softmax => 0,
                    crate::moe::topk_moe::ScoringFunc::SigmoidBias => 1,
                    crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => 2,
                };
                ops::hip::with_moe_route_resident_device_f32(self.device_id, route_device, weight_device, bias_device, route_input.rows, route_input.cols, spec.num_experts, spec.top_k, scoring, spec.routed_scaling_factor, run)
            }
        }
        .map_err(|error| compute_error(format!("ROCm integrated shared expert L{layer}: {error}")))?;
        if ops::hip::options().kernel_sync {
            ops::hip::synchronize_device(self.device_id, &format!("L{layer} ROCm integrated shared expert synchronize")).map_err(compute_error)?;
        }
        if ops::hip::options().debug_finite {
            let elements = expert_input.rows.checked_mul(expert_input.cols).ok_or_else(|| compute_error("ROCm integrated MoE output 元素数溢出"))?;
            ops::hip::try_validate_finite_resident_range_f32(self.device_id, &output, 0, elements).map_err(|error| compute_error(format!("L{layer} ROCm integrated shared expert 包含非有限值: {error}")))?;
        }
        Ok(Some(device_tensor_f32(output, expert_input.rows, expert_input.cols)))
    }

    fn prefill_expert_batch(&self, spec: &crate::moe::topk_moe::TopkMoeSpec, layer: usize, experts: &mut Self::PrefillExperts, batch: Vec<crate::backend::ExpertPrefillBatch<Self::Tensor>>) -> Result<Vec<Self::Tensor>, BackendError> {
        let expert_ids = batch.iter().map(|item| item.expert).collect::<Vec<_>>();
        let loaded = experts.load_batch(self, layer, &expert_ids)?;
        let outputs = batch
            .into_iter()
            .zip(loaded)
            .map(|(item, expert)| match expert {
                RocmPrefillExpert::Nvfp4(weights) => reference_nvfp4_expert_batch(self, &item.input, &weights, &spec.activation),
                RocmPrefillExpert::Resident(weights) => resident_expert_forward(self, &item.input, &weights, &spec.activation),
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        Ok(outputs)
    }

    fn prefill_resident_routed_experts(
        &self,
        spec: &crate::moe::topk_moe::TopkMoeSpec,
        weights: crate::moe::topk_moe::RoutedMoeWeightsRef<'_, Self::Weight>,
        layer: usize,
        experts: &mut Self::PrefillExperts,
        inputs: crate::moe::topk_moe::RoutedMoeInputs<'_, Self::Tensor>,
    ) -> Result<Option<Self::Tensor>, BackendError> {
        self.prefill_resident_routed_experts_impl(spec, weights, layer, experts, inputs, None, None)
    }

    fn prefill_resident_routed_experts_add_shared(
        &self,
        spec: &crate::moe::topk_moe::TopkMoeSpec,
        weights: crate::moe::topk_moe::RoutedMoeWeightsRef<'_, Self::Weight>,
        layer: usize,
        experts: &mut Self::PrefillExperts,
        inputs: crate::moe::topk_moe::RoutedMoeInputs<'_, Self::Tensor>,
        shared: &Self::Tensor,
    ) -> Result<Option<Self::Tensor>, BackendError> {
        // route-major reduce 的 fused epilogue 需要 shared/residual 成对输入；
        // official FP8 prefill 先走 grouped routed，再复用调用方已有的 GPU add。
        if inputs.route.rows > 1 && matches!(experts.archive, RocmExpertArchive::Fp8(_) | RocmExpertArchive::Fp8Source(_)) && ops::hip::options().grouped_wmma && ops::hip::options().grouped_down_route_buffer {
            return Ok(None);
        }
        self.prefill_resident_routed_experts_impl(spec, weights, layer, experts, inputs, Some(shared), None)
    }

    fn prefill_resident_routed_experts_add(
        &self,
        spec: &crate::moe::topk_moe::TopkMoeSpec,
        weights: crate::moe::topk_moe::RoutedMoeWeightsRef<'_, Self::Weight>,
        layer: usize,
        experts: &mut Self::PrefillExperts,
        inputs: crate::moe::topk_moe::RoutedMoeInputs<'_, Self::Tensor>,
        shared: Option<&Self::Tensor>,
        residual: &Self::Tensor,
    ) -> Result<Option<Self::Tensor>, BackendError> {
        let Some(shared) = shared else {
            return Ok(None);
        };
        self.prefill_resident_routed_experts_impl(spec, weights, layer, experts, inputs, Some(shared), Some(residual))
    }

    fn prefill_routed_experts(&self, spec: &crate::moe::topk_moe::TopkMoeSpec, layer: usize, experts: &mut Self::PrefillExperts, input: &Self::Tensor, assignments: &ExpertAssignments) -> Result<Option<Self::Tensor>, BackendError> {
        if ops::hip::options().trace_moe_layer == Some(layer) {
            let input_hash = tensor_data(input)?.into_iter().fold(0xcbf29ce484222325_u64, |hash, value| (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3));
            let mut route_hash = 0xcbf29ce484222325_u64;
            for (expert, routes) in assignments.iter().enumerate() {
                route_hash = (route_hash ^ expert as u64).wrapping_mul(0x100000001b3);
                for &(token, weight) in routes {
                    route_hash = (route_hash ^ u64::from(token)).wrapping_mul(0x100000001b3);
                    route_hash = (route_hash ^ u64::from(weight.to_bits())).wrapping_mul(0x100000001b3);
                }
            }
            eprintln!("[moe-prefill-hash] device={} layer={layer} rows={} input={input_hash:016x} route={route_hash:016x}", self.device_id, input.rows,);
        }
        // MXFP4 grouped:整层两次 launch,CSR 由 assignments(expert-major)直接构建。
        if matches!(experts.archive, RocmExpertArchive::Mxfp4(_)) {
            if let Activation::SiluClamped { limit } = spec.activation {
                if let Some(input_device) = input.device.as_deref() {
                    let grouped = experts.mxfp4_grouped(self, layer, spec)?;
                    let mut route_tokens = Vec::new();
                    let mut route_weights = Vec::new();
                    let mut route_offsets = vec![0u32; spec.num_experts + 1];
                    for (expert, rows) in assignments.iter().enumerate() {
                        for &(token, weight) in rows {
                            route_tokens.push(token);
                            route_weights.push(weight);
                        }
                        route_offsets[expert + 1] = route_tokens.len() as u32;
                    }
                    if route_tokens.is_empty() {
                        return Ok(None);
                    }
                    let output = ops::hip::try_mxfp4_grouped_experts_f32(
                        self.device_id,
                        input_device,
                        input.rows,
                        &route_tokens,
                        &route_weights,
                        &route_offsets,
                        &grouped.gate_up_packed,
                        &grouped.gate_up_scales,
                        &grouped.down_packed,
                        &grouped.down_scales,
                        input.cols,
                        spec.intermediate_size,
                        spec.num_experts,
                        limit,
                    )
                    .map_err(|error| compute_error(format!("MXFP4 grouped prefill L{layer}: {error}")))?;
                    return Ok(Some(device_tensor_f32(output, input.rows, input.cols)));
                }
            }
            return Ok(None);
        }
        if !ops::hip::options().grouped_wmma {
            return Ok(None);
        }
        if !matches!(experts.archive, RocmExpertArchive::Ct(_)) {
            return Ok(None);
        }
        if !matches!(&spec.activation, Activation::Silu) {
            return Ok(None);
        }
        let Some(input_device) = input.device.as_deref() else {
            return Ok(None);
        };
        let mut active = assignments.iter().enumerate().filter_map(|(expert, rows)| (!rows.is_empty()).then_some(expert)).collect::<Vec<_>>();
        active.sort_unstable_by(|&left, &right| assignments[right].len().cmp(&assignments[left].len()).then_with(|| left.cmp(&right)));
        if active.is_empty() {
            return Ok(None);
        }
        let loaded = experts.load_batch(self, layer, &active)?;
        let resident = loaded
            .into_iter()
            .map(|expert| match expert {
                RocmPrefillExpert::Resident(expert) => Ok(expert),
                _ => Err(compute_error("ROCm grouped expert 只接受 resident W4A16 权重")),
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let grouped = resident.iter().map(|expert| Some(ops::hip::CtGroupedExpertRef { gate: grouped_w4(&expert.gate)?, up: grouped_w4(&expert.up)?, down: grouped_w4(&expert.down)? })).collect::<Option<Vec<_>>>();
        let Some(grouped) = grouped else {
            return Ok(None);
        };
        let mut route_tokens = Vec::new();
        let mut route_weights = Vec::new();
        let mut route_offsets = Vec::with_capacity(active.len() + 1);
        route_offsets.push(0);
        for &expert in &active {
            for &(token, weight) in &assignments[expert] {
                route_tokens.push(token);
                route_weights.push(weight);
            }
            route_offsets.push(u32::try_from(route_tokens.len()).map_err(|_| compute_error("ROCm grouped route 数超过 u32"))?);
        }
        if ops::hip::options().log_expert_pointers_device == Some(self.device_id) {
            let routes = active.iter().map(|&expert| (expert, assignments[expert].len())).collect::<Vec<_>>();
            eprintln!("[grouped-routes] device={} layer={layer} rows={} token_max={:?} routes={} active={routes:?}", self.device_id, input.rows, route_tokens.iter().max(), route_tokens.len());
        }
        let output = ops::hip::try_ct_grouped_experts_bf16(self.device_id, input_device, input.rows, input.cols, spec.intermediate_size, &route_tokens, &route_weights, &route_offsets, &grouped, None, None, None).map_err(compute_error)?;
        if ops::hip::options().kernel_sync {
            ops::hip::synchronize_device(self.device_id, &format!("L{layer} ROCm grouped expert synchronize")).map_err(compute_error)?;
        }
        if input.rows != 0 && ops::hip::options().debug_finite {
            ops::hip::try_validate_finite_resident_range_f32(self.device_id, &output, (input.rows - 1) * input.cols, input.cols).map_err(|error| compute_error(format!("L{layer} ROCm grouped expert 尾行包含非有限值: {error}")))?;
        }
        Ok(Some(device_tensor_f32(output, input.rows, input.cols)))
    }
}

impl ExpertDecodeBackend for RocmContext {
    type MoeState = crate::moe::UncachedMoeState;
    type DecodeRouting = ();

    fn decode_route(&self, input: &Self::Tensor, router_weight: &Self::Weight, router_bias: &Self::Weight, spec: &crate::moe::topk_moe::TopkMoeSpec) -> Result<(MoePrefillRouting, Self::DecodeRouting), BackendError> {
        Ok((self.moe_route(input, router_weight, router_bias, spec)?, ()))
    }

    fn prefetch_experts(&self, _spec: &crate::moe::topk_moe::TopkMoeSpec, _state: &mut Self::MoeState, _request: crate::backend::ExpertPrefetchRequest<'_>) -> Result<usize, BackendError> {
        Ok(0)
    }

    fn decode_routed_experts<'a, F>(
        &self,
        spec: &crate::moe::topk_moe::TopkMoeSpec,
        layer: usize,
        source: crate::weight::expert_source::ExpertSource<'_>,
        state: &mut Self::MoeState,
        input: &Self::Tensor,
        assignments: &crate::moe::routing::ExpertAssignments,
        _routing: &Self::DecodeRouting,
        on_ready: F,
    ) -> Result<Self::Tensor, BackendError>
    where
        F: FnOnce(&mut Self::MoeState) -> Result<Option<crate::backend::ExpertPrefetchRequest<'a>>, BackendError>,
    {
        let active = assignments.iter().filter(|rows| !rows.is_empty()).count();
        state.record_routed_experts(active);
        if let Some(request) = on_ready(state)? {
            self.prefetch_experts(spec, state, request)?;
        }
        match source {
            crate::weight::expert_source::ExpertSource::Fp8(source) => decode_fp8_routed(self, spec, layer, source, input, assignments),
            crate::weight::expert_source::ExpertSource::Nvfp4(source) => decode_nvfp4_routed(self, spec, layer, source, input, assignments),
            crate::weight::expert_source::ExpertSource::Gguf(source) => decode_gguf_routed(self, spec, layer, source, input, assignments),
            crate::weight::expert_source::ExpertSource::Mxfp8(_) => Err(BackendError::ExpertLoad("Rocm decode 尚未实现 MXFP8 expert kernel".to_owned())),
            crate::weight::expert_source::ExpertSource::Mxfp4(source) => decode_mxfp4_routed(self, spec, layer, source, input, assignments),
            crate::weight::expert_source::ExpertSource::W4A16(source) => decode_w4a16_routed(self, spec, layer, source, input, assignments),
        }
    }
}

fn decode_fp8_routed(
    ctx: &RocmContext,
    spec: &TopkMoeSpec,
    layer: usize,
    source: &dyn crate::weight::expert_source::Fp8ExpertSource,
    input: &RocmTensor,
    assignments: &crate::moe::routing::ExpertAssignments,
) -> Result<RocmTensor, BackendError> {
    crate::moe::routing::execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let weights = source.load_expert_fp8(layer, expert).map_err(BackendError::ExpertLoad)?;
        let resident = prepare_fp8_expert(ctx, weights)?;
        resident_expert_forward(ctx, expert_input, &resident, &spec.activation)
    })
}

fn decode_nvfp4_routed(
    _ctx: &RocmContext,
    spec: &TopkMoeSpec,
    layer: usize,
    source: &dyn crate::weight::expert_source::Nvfp4ExpertSource,
    input: &RocmTensor,
    assignments: &crate::moe::routing::ExpertAssignments,
) -> Result<RocmTensor, BackendError> {
    if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
        return Err(BackendError::ExpertLoad(format!("NVFP4 expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
    }
    crate::moe::routing::execute_routed_experts(_ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let weights = source.load_expert_nvfp4(layer, expert).map_err(BackendError::ExpertLoad)?;
        reference_nvfp4_expert_batch(_ctx, expert_input, &weights, &spec.activation)
    })
}

fn decode_mxfp4_routed(
    ctx: &RocmContext,
    spec: &TopkMoeSpec,
    layer: usize,
    source: &dyn crate::weight::expert_source::Mxfp4ExpertSource,
    input: &RocmTensor,
    assignments: &crate::moe::routing::ExpertAssignments,
) -> Result<RocmTensor, BackendError> {
    if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
        return Err(BackendError::ExpertLoad(format!("MXFP4 expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
    }
    crate::moe::routing::execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let weights = source.load_expert_mxfp4(layer, expert).map_err(BackendError::ExpertLoad)?;
        let resident = prepare_mxfp4_expert(ctx, &weights)?;
        resident_expert_forward(ctx, expert_input, &resident, &spec.activation)
    })
}

fn prepare_mxfp4_expert(backend: &RocmContext, weights: &Mxfp4ExpertWeights) -> Result<RocmResidentExpert, BackendError> {
    let intermediate = weights.gate.rows();
    let hidden = weights.gate.cols();
    let prepare = |matrix: &crate::weight::format::mxfp4::Mxfp4Matrix, rows: usize, cols: usize| backend.prepare_weight(LinearWeight::mxfp4(matrix), rows, cols);
    Ok(RocmResidentExpert { gate: prepare(&weights.gate, intermediate, hidden)?, up: prepare(&weights.up, intermediate, hidden)?, down: prepare(&weights.down, hidden, intermediate)? })
}

fn decode_w4a16_routed(
    ctx: &RocmContext,
    spec: &TopkMoeSpec,
    layer: usize,
    source: &dyn crate::weight::expert_source::W4A16ExpertSource,
    input: &RocmTensor,
    assignments: &crate::moe::routing::ExpertAssignments,
) -> Result<RocmTensor, BackendError> {
    crate::moe::routing::execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let weights = source.load_expert_w4a16(layer, expert).map_err(BackendError::ExpertLoad)?;
        let gate = weights.gate.decode().map_err(BackendError::ExpertLoad)?;
        let up = weights.up.decode().map_err(BackendError::ExpertLoad)?;
        let down = weights.down.decode().map_err(BackendError::ExpertLoad)?;
        reference_f32_expert_batch(ctx, expert_input, &gate, &up, &down, spec.intermediate_size, &spec.activation)
    })
}

fn decode_gguf_routed(
    ctx: &RocmContext,
    spec: &TopkMoeSpec,
    layer: usize,
    source: &dyn crate::weight::expert_source::GgufExpertSource,
    input: &RocmTensor,
    assignments: &crate::moe::routing::ExpertAssignments,
) -> Result<RocmTensor, BackendError> {
    if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
        return Err(BackendError::ExpertLoad(format!("GGUF expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
    }
    crate::moe::routing::execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        // decode 侧暂无跨 token 缓存：按次 prepare 上传，执行与 prefill 同一 GPU 路径。
        let weights = source.load_expert_gguf(layer, expert).map_err(BackendError::ExpertLoad)?;
        let resident = prepare_gguf_expert(ctx, &weights)?;
        resident_expert_forward(ctx, expert_input, &resident, &spec.activation)
    })
}

/// 常驻 expert 的 GPU 前向：gate/up 双线性 + 门控激活 + down 投影。
fn resident_expert_forward(ctx: &RocmContext, input: &RocmTensor, weights: &RocmResidentExpert, activation: &Activation) -> Result<RocmTensor, BackendError> {
    let (gate, up) = ctx.dual_linear(input, &weights.gate, &weights.up)?;
    let activated = ctx.gated_activation(&gate, &up, activation)?;
    ctx.linear(&activated, &weights.down)
}
enum RocmExpertArchive {
    Fp8(OfficialExpertArchive),
    Fp8Source(std::sync::Arc<dyn crate::weight::expert_source::Fp8ExpertSource + Send + Sync>),
    Mxfp4(Arc<dyn Mxfp4ExpertSource>),
    Nvfp4(NvidiaNvfp4Experts),
    Gguf(Arc<dyn GgufExpertSource>),
    Ct(CompressedTensorsSource),
}

#[derive(Clone)]
struct RocmResidentExpert {
    gate: RocmWeight,
    up: RocmWeight,
    down: RocmWeight,
}

/// 官方 block-FP8 专家保持 codes/scale 原始形态常驻；gfx11 在 routed kernel
/// 中按真实 top-k 即时解码，避免把整层专家膨胀成 BF16 或回 host 路由。
struct RocmFp8Weight {
    codes: Arc<ops::hip::DeviceBuffer>,
    scales: Arc<ops::hip::DeviceBuffer>,
}

struct RocmFp8ResidentExpert {
    gate: RocmFp8Weight,
    up: RocmFp8Weight,
    down: RocmFp8Weight,
}

fn grouped_w4(weight: &RocmWeight) -> Option<ops::hip::CtGroupedWeightRef<'_>> {
    match weight.quantized() {
        Some(RocmQuantizedWeight::W4A16 { packed, scales, scale_dtype, group_size }) if group_size.is_multiple_of(16) => Some(ops::hip::CtGroupedWeightRef {
            packed,
            scales,
            scale_dtype: match scale_dtype {
                ScaleDType::Bf16 => 0,
                ScaleDType::F16 => 1,
                ScaleDType::F32 => 2,
            },
            group_size: *group_size,
            format: 0,
        }),
        _ => None,
    }
}

fn grouped_w4_experts(resident: &[Arc<RocmResidentExpert>]) -> Result<Vec<ops::hip::CtGroupedExpertRef<'_>>, BackendError> {
    resident
        .iter()
        .map(|expert| Some(ops::hip::CtGroupedExpertRef { gate: grouped_w4(&expert.gate)?, up: grouped_w4(&expert.up)?, down: grouped_w4(&expert.down)? }))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| compute_error("ROCm cooperative expert 需要 W4A16 grouped 权重"))
}

fn gguf_grouped_metas_from_residents(layer: usize, resident: &[Arc<RocmResidentExpert>]) -> Result<Vec<ops::hip::GgufGroupedExpertMeta>, BackendError> {
    resident.iter().map(|expert| gguf_grouped_meta_from_weights(layer, &expert.gate, &expert.up, &expert.down)).collect()
}

fn gguf_grouped_meta_from_weights(layer: usize, gate: &RocmWeight, up: &RocmWeight, down: &RocmWeight) -> Result<ops::hip::GgufGroupedExpertMeta, BackendError> {
    let meta = |weight: &RocmWeight| -> Result<(u32, u64), BackendError> {
        weight.expert_gguf().map(|(codes, tensor_type)| (tensor_type, codes.device_pointer() as u64)).ok_or_else(|| BackendError::ExpertLoad(format!("GGUF grouped L{layer} 权重不是 GGUF 打包形态")))
    };
    let (gate_type, gate) = meta(gate)?;
    let (up_type, up) = meta(up)?;
    let (down_type, down) = meta(down)?;
    Ok(ops::hip::GgufGroupedExpertMeta { gate, up, down, gate_type, up_type, down_type })
}

fn grouped_w8(weight: &RocmWeight) -> Option<ops::hip::CtGroupedWeightRef<'_>> {
    match weight.quantized() {
        Some(RocmQuantizedWeight::W8A16 { packed, scales, scale_dtype, group_size }) if group_size.is_multiple_of(4) => Some(ops::hip::CtGroupedWeightRef {
            packed,
            scales,
            scale_dtype: match scale_dtype {
                ScaleDType::Bf16 => 0,
                ScaleDType::F16 => 1,
                ScaleDType::F32 => 2,
            },
            group_size: *group_size,
            format: 0,
        }),
        _ => None,
    }
}

/// 一层单行 integrated MoE 的固定地址 graph owner。输入每轮先 D2D 写入固定
/// buffer；graph 本体只含 router、top-k、gate/up、down 四个 kernel node。
struct RocmCtMoeGraph {
    graph: ops::hip::StaticHipGraph,
    route_input: Arc<ops::hip::DeviceBuffer>,
    expert_input: Arc<ops::hip::DeviceBuffer>,
    residual: Arc<ops::hip::DeviceBuffer>,
    _route: ops::hip::MoeRouteGraphBuffers,
    decode: ops::hip::CtIntegratedDecodeGraphBuffers,
    device_id: i32,
    stream: usize,
}

impl RocmCtMoeGraph {
    #[allow(clippy::too_many_arguments)]
    fn build(
        device_id: i32,
        route_input_bytes: usize,
        expert_input_bytes: usize,
        residual_bytes: usize,
        hidden: usize,
        intermediate: usize,
        expert_count: usize,
        top_k: usize,
        scoring: u32,
        scaling: f32,
        router_weight: &ops::hip::DeviceBuffer,
        router_bias: &ops::hip::DeviceBuffer,
        experts: &[ops::hip::CtGroupedExpertRef<'_>],
        shared: &ops::hip::CtGroupedExpertRef<'_>,
    ) -> Result<Self, String> {
        let route_input = Arc::new(ops::hip::DeviceBuffer::allocate(device_id, route_input_bytes)?);
        let expert_input = Arc::new(ops::hip::DeviceBuffer::allocate(device_id, expert_input_bytes)?);
        let residual = Arc::new(ops::hip::DeviceBuffer::allocate(device_id, residual_bytes)?);
        let route = ops::hip::MoeRouteGraphBuffers::new(device_id, 1, expert_count, top_k)?;
        let decode = ops::hip::CtIntegratedDecodeGraphBuffers::new(device_id, 1, hidden, intermediate, top_k, experts, shared)?;

        // buffer 分配和 HIPRTC 预热都会触碰 current device；完成全部准备后再绑定
        // stream，保证模板 owner 与各 kernel 实际提交到同一条 stage stream。
        let stream = ops::hip::active_compute_stream();
        let recorder = ops::hip::StaticGraphRecorder::begin(device_id, stream)?;
        route.launch(device_id, &route_input, router_weight, router_bias, 1, hidden, expert_count, top_k, scoring, scaling)?;
        decode.launch(device_id, &expert_input, route.expert_ids(), route.weights(), &residual)?;
        let graph = recorder.finish()?.ok_or("ROCm integrated MoE graph 录制段没有 kernel 节点")?;
        if graph.node_count() < 4 {
            return Err(format!("ROCm integrated MoE graph 节点过少: {}", graph.node_count()));
        }
        Ok(Self { graph, route_input, expert_input, residual, _route: route, decode, device_id, stream: stream as usize })
    }

    fn launch(&self, route_input: &ops::hip::DeviceBuffer, expert_input: &ops::hip::DeviceBuffer, residual: &ops::hip::DeviceBuffer) -> Result<Arc<ops::hip::DeviceBuffer>, String> {
        if route_input.bytes() != self.route_input.bytes() || expert_input.bytes() != self.expert_input.bytes() || residual.bytes() != self.residual.bytes() {
            return Err(format!(
                "ROCm integrated MoE graph 输入 shape 变化: route={}/{} expert={}/{} residual={}/{}",
                route_input.bytes(),
                self.route_input.bytes(),
                expert_input.bytes(),
                self.expert_input.bytes(),
                residual.bytes(),
                self.residual.bytes(),
            ));
        }
        self.route_input.copy_from_device(0, route_input, 0, route_input.bytes())?;
        self.expert_input.copy_from_device(0, expert_input, 0, expert_input.bytes())?;
        self.residual.copy_from_device(0, residual, 0, residual.bytes())?;
        self.graph.launch(self.device_id, self.stream as *mut std::ffi::c_void)?;
        Ok(self.decode.output())
    }
}

enum RocmCtMoeGraphState {
    Ready(RocmCtMoeGraph),
    Disabled,
}

fn grouped_fp8(weight: &RocmFp8Weight) -> ops::hip::CtGroupedWeightRef<'_> {
    ops::hip::CtGroupedWeightRef { packed: &weight.codes, scales: &weight.scales, scale_dtype: 2, group_size: 128, format: 1 }
}

fn prepare_ct_matrix(backend: &RocmContext, matrix: crate::weight::format::compressed_tensors_hybrid::CtMatrix) -> Result<RocmWeight, BackendError> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    match &matrix {
        crate::weight::format::compressed_tensors_hybrid::CtMatrix::W4(matrix) => backend.prepare_weight(LinearWeight::w4a16(matrix), rows, cols),
        crate::weight::format::compressed_tensors_hybrid::CtMatrix::W8(matrix) => backend.prepare_weight(LinearWeight::w8a16(matrix), rows, cols),
    }
}

fn prepare_ct_expert(backend: &RocmContext, weights: crate::weight::expert_source::CtExpertWeights) -> Result<Arc<RocmResidentExpert>, BackendError> {
    Ok(Arc::new(RocmResidentExpert { gate: prepare_ct_matrix(backend, weights.gate)?, up: prepare_ct_matrix(backend, weights.up)?, down: prepare_ct_matrix(backend, weights.down)? }))
}

fn ct_matrix_row_shard(matrix: &crate::weight::format::compressed_tensors_hybrid::CtMatrix, partition: usize) -> Result<crate::weight::format::compressed_tensors_hybrid::CtMatrix, BackendError> {
    use crate::weight::format::{compressed_tensors_hybrid::CtMatrix, quantization::W4A16Matrix, quantization::W8A16Matrix};

    if partition >= 2 || !matrix.rows().is_multiple_of(2) {
        return Err(compute_error(format!("CT expert row shard partition={partition} rows={} 无效", matrix.rows())));
    }
    let rows = matrix.rows() / 2;
    let row_start = partition * rows;
    let slice_rows = |bytes: &[u8], row_bytes: usize, label: &str| -> Result<Vec<u8>, BackendError> {
        let start = row_start.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("CT {label} row offset 溢出")))?;
        let end = start.checked_add(rows.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("CT {label} row bytes 溢出")))?).ok_or_else(|| compute_error(format!("CT {label} row end 溢出")))?;
        bytes.get(start..end).map(<[u8]>::to_vec).ok_or_else(|| compute_error(format!("CT {label} row shard {start}..{end}/{} 越界", bytes.len())))
    };
    match matrix {
        CtMatrix::W4(matrix) => {
            let packed = slice_rows(matrix.packed(), matrix.cols.div_ceil(8) * 4, "W4 packed")?;
            let scales = slice_rows(matrix.scales(), matrix.cols / matrix.group_size() * matrix.scale_dtype().bytes(), "W4 scales")?;
            W4A16Matrix::new(packed, scales, matrix.scale_dtype(), matrix.group_size(), rows, matrix.cols).map(CtMatrix::W4).map_err(compute_error)
        }
        CtMatrix::W8(matrix) => {
            let packed = slice_rows(matrix.packed(), matrix.cols.div_ceil(4) * 4, "W8 packed")?;
            let scales = slice_rows(matrix.scales(), matrix.cols / matrix.group_size() * matrix.scale_dtype().bytes(), "W8 scales")?;
            W8A16Matrix::new(packed, scales, matrix.scale_dtype(), matrix.group_size(), rows, matrix.cols).map(CtMatrix::W8).map_err(compute_error)
        }
    }
}

fn ct_matrix_column_shard(matrix: &crate::weight::format::compressed_tensors_hybrid::CtMatrix, partition: usize) -> Result<crate::weight::format::compressed_tensors_hybrid::CtMatrix, BackendError> {
    use crate::weight::format::{compressed_tensors_hybrid::CtMatrix, quantization::W4A16Matrix, quantization::W8A16Matrix};

    if partition >= 2 || !matrix.cols().is_multiple_of(2) {
        return Err(compute_error(format!("CT expert column shard partition={partition} cols={} 无效", matrix.cols())));
    }
    let rows = matrix.rows();
    let cols = matrix.cols() / 2;
    let slice_columns = |bytes: &[u8], source_row_bytes: usize, shard_row_bytes: usize, label: &str| -> Result<Vec<u8>, BackendError> {
        let mut shard = Vec::with_capacity(rows.checked_mul(shard_row_bytes).ok_or_else(|| compute_error(format!("CT {label} column shard 容量溢出")))?);
        let column_offset = partition * shard_row_bytes;
        for row in 0..rows {
            let start = row.checked_mul(source_row_bytes).and_then(|offset| offset.checked_add(column_offset)).ok_or_else(|| compute_error(format!("CT {label} column offset 溢出")))?;
            let end = start.checked_add(shard_row_bytes).ok_or_else(|| compute_error(format!("CT {label} column end 溢出")))?;
            shard.extend_from_slice(bytes.get(start..end).ok_or_else(|| compute_error(format!("CT {label} column shard {start}..{end}/{} 越界", bytes.len())))?);
        }
        Ok(shard)
    };
    match matrix {
        CtMatrix::W4(matrix) => {
            if !cols.is_multiple_of(matrix.group_size()) {
                return Err(compute_error(format!("CT expert W4 column shard cols={cols} group={} 无效", matrix.group_size())));
            }
            let packed = slice_columns(matrix.packed(), matrix.cols.div_ceil(8) * 4, cols.div_ceil(8) * 4, "W4 packed")?;
            let scales = slice_columns(matrix.scales(), matrix.cols / matrix.group_size() * matrix.scale_dtype().bytes(), cols / matrix.group_size() * matrix.scale_dtype().bytes(), "W4 scales")?;
            W4A16Matrix::new(packed, scales, matrix.scale_dtype(), matrix.group_size(), rows, cols).map(CtMatrix::W4).map_err(compute_error)
        }
        CtMatrix::W8(matrix) => {
            if !cols.is_multiple_of(matrix.group_size()) {
                return Err(compute_error(format!("CT expert W8 column shard cols={cols} group={} 无效", matrix.group_size())));
            }
            let packed = slice_columns(matrix.packed(), matrix.cols.div_ceil(4) * 4, cols.div_ceil(4) * 4, "W8 packed")?;
            let scales = slice_columns(matrix.scales(), matrix.cols / matrix.group_size() * matrix.scale_dtype().bytes(), cols / matrix.group_size() * matrix.scale_dtype().bytes(), "W8 scales")?;
            W8A16Matrix::new(packed, scales, matrix.scale_dtype(), matrix.group_size(), rows, cols).map(CtMatrix::W8).map_err(compute_error)
        }
    }
}

fn prepare_ct_expert_tp_shard(backend: &RocmContext, weights: &crate::weight::expert_source::CtExpertWeights, partition: usize) -> Result<Arc<RocmResidentExpert>, BackendError> {
    let gate = prepare_ct_matrix(backend, ct_matrix_row_shard(&weights.gate, partition)?)?;
    let up = prepare_ct_matrix(backend, ct_matrix_row_shard(&weights.up, partition)?)?;
    let down = prepare_ct_matrix(backend, ct_matrix_column_shard(&weights.down, partition)?)?;
    Ok(Arc::new(RocmResidentExpert { gate, up, down }))
}

fn load_ct_shared_expert(source: &CompressedTensorsSource, layer: usize) -> Result<crate::weight::expert_source::CtExpertWeights, String> {
    let prefix = format!("model.layers.{layer}.mlp.shared_experts");
    let load = |projection: &str| source.load_matrix(&format!("{prefix}.{projection}.weight"));
    std::thread::scope(|scope| {
        let gate = scope.spawn(|| load("gate_proj"));
        let up = scope.spawn(|| load("up_proj"));
        let down = load("down_proj")?;
        Ok(crate::weight::expert_source::CtExpertWeights { gate: gate.join().map_err(|_| format!("CT L{layer} shared gate 读取线程 panic"))??, up: up.join().map_err(|_| format!("CT L{layer} shared up 读取线程 panic"))??, down })
    })
}

fn prepare_fp8_expert(backend: &RocmContext, weights: crate::weight::format::official_fp8::Fp8ExpertWeights) -> Result<Arc<RocmResidentExpert>, BackendError> {
    let prepare = |matrix: &crate::weight::Fp8Matrix| backend.prepare_weight(LinearWeight::fp8(matrix), matrix.rows, matrix.cols);
    let ((gate, up), down) = rayon::join(|| rayon::join(|| prepare(&weights.gate), || prepare(&weights.up)), || prepare(&weights.down));
    Ok(Arc::new(RocmResidentExpert { gate: gate?, up: up?, down: down? }))
}

fn prepare_fp8_packed_expert(backend: &RocmContext, weights: crate::weight::format::official_fp8::Fp8ExpertWeights) -> Result<Arc<RocmFp8ResidentExpert>, BackendError> {
    let prepare = |matrix: crate::weight::Fp8Matrix| -> Result<RocmFp8Weight, BackendError> {
        let codes = Arc::new(ops::hip::DeviceBuffer::upload(backend.device_id, &matrix.codes).map_err(compute_error)?);
        let scales = Arc::new(ops::hip::DeviceBuffer::upload(backend.device_id, &matrix.scale_inv).map_err(compute_error)?);
        Ok(RocmFp8Weight { codes, scales })
    };
    Ok(Arc::new(RocmFp8ResidentExpert { gate: prepare(weights.gate)?, up: prepare(weights.up)?, down: prepare(weights.down)? }))
}

/// GGUF expert → GPU 常驻形态：Q4_K/Q5_K 打包常驻，其余(Q6_K)由 prepare_weight 反量化为 BF16。
fn prepare_gguf_expert(backend: &RocmContext, weights: &crate::weight::expert_source::GgufExpertWeights) -> Result<Arc<RocmResidentExpert>, BackendError> {
    let prepare = |matrix: &crate::weight::container::gguf::GgufMatrix| backend.prepare_gguf_packed(matrix);
    Ok(Arc::new(RocmResidentExpert { gate: prepare(&weights.gate)?, up: prepare(&weights.up)?, down: prepare(&weights.down)? }))
}

/// GGUF K-quant 的每行由独立 256 列 block 组成；half-intermediate 与 block
/// 对齐时可直接抽取每行前/后半段，不需要解量化或重新编码。
fn gguf_packed_column_halves_bytes(bytes: &[u8], rows: usize, columns: usize, tensor_type: crate::weight::container::gguf::GgmlType) -> Result<[Vec<u8>; 2], String> {
    if !columns.is_multiple_of(512) {
        return Err(format!("GGUF TP column shape [{rows},{columns}] 不能二等分到 256 列 block"));
    }
    let row_bytes = tensor_type.storage_bytes(columns)?;
    let half_row_bytes = tensor_type.storage_bytes(columns / 2)?;
    let expected = rows.checked_mul(row_bytes).ok_or_else(|| "GGUF TP packed bytes 数溢出".to_string())?;
    if row_bytes != half_row_bytes * 2 || bytes.len() != expected {
        return Err(format!("GGUF TP packed bytes={} row={row_bytes} half={half_row_bytes} shape=[{rows},{columns}] 不一致", bytes.len()));
    }
    let capacity = rows.checked_mul(half_row_bytes).ok_or_else(|| "GGUF TP half packed bytes 数溢出".to_string())?;
    let mut low = Vec::with_capacity(capacity);
    let mut high = Vec::with_capacity(capacity);
    for row in bytes.chunks_exact(row_bytes) {
        low.extend_from_slice(&row[..half_row_bytes]);
        high.extend_from_slice(&row[half_row_bytes..]);
    }
    Ok([low, high])
}

fn gguf_packed_column_halves(matrix: &crate::weight::container::gguf::GgufMatrix) -> Result<[Vec<u8>; 2], BackendError> {
    let bytes = matrix.read_bytes().map_err(compute_error)?;
    gguf_packed_column_halves_bytes(&bytes, matrix.rows, matrix.columns, matrix.tensor_type).map_err(compute_error)
}

/// 一次顺序读取三块 expert 权重，再把两个 TP shard 并行上传到 owner/peer。
/// gate/up 的行半段本来连续；down 只做 block-aligned 行内拷贝。
fn prepare_gguf_expert_tp_shards(owner: &RocmContext, peer: &RocmContext, weights: &crate::weight::expert_source::GgufExpertWeights, owner_partition: usize) -> Result<(Arc<RocmResidentExpert>, Arc<RocmResidentExpert>), BackendError> {
    if owner_partition >= 2
        || !weights.gate.rows.is_multiple_of(2)
        || weights.gate.rows != weights.up.rows
        || weights.gate.columns != weights.up.columns
        || weights.down.rows != weights.gate.columns
        || weights.down.columns != weights.gate.rows
    {
        return Err(compute_error(format!(
            "GGUF expert TP shape gate=[{},{}] up=[{},{}] down=[{},{}] partition={owner_partition} 非法",
            weights.gate.rows, weights.gate.columns, weights.up.rows, weights.up.columns, weights.down.rows, weights.down.columns
        )));
    }
    let gate = weights.gate.read_bytes().map_err(compute_error)?;
    let up = weights.up.read_bytes().map_err(compute_error)?;
    if !gate.len().is_multiple_of(2) || !up.len().is_multiple_of(2) {
        return Err(compute_error("GGUF expert gate/up packed bytes 不能二等分"));
    }
    let gate = [&gate[..gate.len() / 2], &gate[gate.len() / 2..]];
    let up = [&up[..up.len() / 2], &up[up.len() / 2..]];
    let down = gguf_packed_column_halves(&weights.down)?;
    let half = weights.gate.rows / 2;
    let prepare = |backend: &RocmContext, partition: usize| -> Result<Arc<RocmResidentExpert>, BackendError> {
        backend.activate().map_err(compute_error)?;
        let gate = backend.prepare_gguf_packed_bytes(gate[partition], weights.gate.tensor_type.0, half, weights.gate.columns)?;
        let up = backend.prepare_gguf_packed_bytes(up[partition], weights.up.tensor_type.0, half, weights.up.columns)?;
        let down = backend.prepare_gguf_packed_bytes(&down[partition], weights.down.tensor_type.0, weights.down.rows, half)?;
        Ok(Arc::new(RocmResidentExpert { gate, up, down }))
    };
    let peer_partition = 1 - owner_partition;
    let (local, remote) = rayon::join(|| prepare(owner, owner_partition), || prepare(peer, peer_partition));
    Ok((local?, remote?))
}

enum RocmPrefillExpert {
    Nvfp4(Nvfp4ExpertWeights),
    Resident(Arc<RocmResidentExpert>),
}

/// ROCm prefill expert 数据源与按 device/layer/expert 缓存的压缩 resident 权重。
pub struct RocmPrefillExperts {
    archive: RocmExpertArchive,
    resident: HashMap<(i32, usize, usize), Arc<RocmResidentExpert>>,
    fp8_resident: HashMap<(i32, usize, usize), Arc<RocmFp8ResidentExpert>>,
    fp8_metas: HashMap<(i32, usize), Arc<ops::hip::DeviceBuffer>>,
    /// GGUF routed/shared pointer 表在预载时上传，decode 只借用。
    gguf_grouped: HashMap<(i32, usize), RocmGgufGroupedLayer>,
    /// MXFP4 层级大 buffer(gate/up 拼段 + down),grouped kernel 直接消费。
    mxfp4_grouped: HashMap<(i32, usize), Arc<Mxfp4GroupedLayer>>,
    /// 与 resident 权重同 generation 的单行固定地址 graph；Disabled 避免失败后逐轮重试。
    ct_moe_graphs: HashMap<(i32, usize), RocmCtMoeGraphState>,
    /// 单路实验只在同机相邻卡间拆 routed experts；Attention/KV/DSA 仍由当前卡持有。
    cooperative_peer: Option<RocmCooperativeExpertPeer>,
    /// sequence-parallel attention 的 q_b/kv_b/o_proj 在两卡完整驻留。两边
    /// 各自把本地 KV shard 算到 full-hidden partial，最后只归约 hidden。
    cooperative_mla: HashMap<usize, RocmCooperativeMlaWeights>,
    /// Indexer 只在稀疏层出现；peer 常驻完整 wq_b 后，两卡可按 query 行各算一半。
    cooperative_dsa_wq_b: HashMap<usize, RocmWeight>,
}

#[derive(Clone, Copy)]
struct RocmCooperativeExpertPeer {
    context: RocmContext,
    local_partition: usize,
}

struct RocmGgufGroupedLayer {
    routed: ops::hip::GgufGroupedMetas,
    /// 单卡 shared expert 由模型权重持有，不属于 expert archive；只有 cooperative
    /// preload 会在这里同时保存其 TP shard 指针表。
    shared: Option<ops::hip::GgufGroupedMetas>,
}

#[derive(Clone)]
pub(super) struct RocmCooperativeMlaWeights {
    pub(super) owner_q_b: RocmWeight,
    pub(super) peer_q_b: RocmWeight,
    /// decode 单行按 attention head 均分 q_b 输出行；prefill 仍使用完整权重按 token 行分片。
    pub(super) owner_decode_q_b: Option<RocmWeight>,
    pub(super) peer_decode_q_b: Option<RocmWeight>,
    pub(super) owner_kv_b: RocmWeight,
    pub(super) peer_kv_b: RocmWeight,
    pub(super) owner_o: RocmWeight,
    pub(super) peer_o: RocmWeight,
}

impl RocmCooperativeMlaWeights {
    pub(super) fn decode_q_b_shards(&self) -> Option<(&RocmWeight, &RocmWeight)> {
        Some((self.owner_decode_q_b.as_ref()?, self.peer_decode_q_b.as_ref()?))
    }

    pub(super) fn o_head_sharded(&self) -> bool {
        self.owner_o.cols == self.peer_o.cols && self.owner_o.cols.checked_add(self.peer_o.cols) == Some(self.owner_q_b.rows)
    }
}

/// 一层全部专家的 4bit 驻留大 buffer:grouped kernel 的权重布局。
pub struct Mxfp4GroupedLayer {
    pub gate_up_packed: Arc<ops::hip::DeviceBuffer>,
    pub gate_up_scales: Arc<ops::hip::DeviceBuffer>,
    pub down_packed: Arc<ops::hip::DeviceBuffer>,
    pub down_scales: Arc<ops::hip::DeviceBuffer>,
}

impl RocmPrefillExperts {
    pub fn fp8(root: &Path, intermediate: usize, hidden: usize, expert_count: usize) -> Result<Self, String> {
        Ok(Self {
            archive: RocmExpertArchive::Fp8(OfficialExpertArchive::open(root, intermediate, hidden, expert_count)?),
            resident: HashMap::new(),
            fp8_resident: HashMap::new(),
            fp8_metas: HashMap::new(),
            gguf_grouped: HashMap::new(),
            mxfp4_grouped: HashMap::new(),
            ct_moe_graphs: HashMap::new(),
            cooperative_peer: None,
            cooperative_mla: HashMap::new(),
            cooperative_dsa_wq_b: HashMap::new(),
        })
    }

    /// 任意 Fp8ExpertSource(非 DeepSeek 目录命名,如 GLM-5.3-Flash)的 FP8 专家通道。
    pub fn fp8_source(source: std::sync::Arc<dyn crate::weight::expert_source::Fp8ExpertSource + Send + Sync>) -> Self {
        Self {
            archive: RocmExpertArchive::Fp8Source(source),
            resident: HashMap::new(),
            fp8_resident: HashMap::new(),
            fp8_metas: HashMap::new(),
            gguf_grouped: HashMap::new(),
            mxfp4_grouped: HashMap::new(),
            ct_moe_graphs: HashMap::new(),
            cooperative_peer: None,
            cooperative_mla: HashMap::new(),
            cooperative_dsa_wq_b: HashMap::new(),
        }
    }

    pub fn nvfp4(source: NvidiaNvfp4Experts) -> Self {
        Self {
            archive: RocmExpertArchive::Nvfp4(source),
            resident: HashMap::new(),
            fp8_resident: HashMap::new(),
            fp8_metas: HashMap::new(),
            gguf_grouped: HashMap::new(),
            mxfp4_grouped: HashMap::new(),
            ct_moe_graphs: HashMap::new(),
            cooperative_peer: None,
            cooperative_mla: HashMap::new(),
            cooperative_dsa_wq_b: HashMap::new(),
        }
    }

    pub fn mxfp4(source: Arc<dyn Mxfp4ExpertSource>) -> Self {
        Self {
            archive: RocmExpertArchive::Mxfp4(source),
            resident: HashMap::new(),
            fp8_resident: HashMap::new(),
            fp8_metas: HashMap::new(),
            gguf_grouped: HashMap::new(),
            mxfp4_grouped: HashMap::new(),
            ct_moe_graphs: HashMap::new(),
            cooperative_peer: None,
            cooperative_mla: HashMap::new(),
            cooperative_dsa_wq_b: HashMap::new(),
        }
    }

    pub fn gguf(source: Arc<dyn GgufExpertSource>) -> Self {
        Self {
            archive: RocmExpertArchive::Gguf(source),
            resident: HashMap::new(),
            fp8_resident: HashMap::new(),
            fp8_metas: HashMap::new(),
            gguf_grouped: HashMap::new(),
            mxfp4_grouped: HashMap::new(),
            ct_moe_graphs: HashMap::new(),
            cooperative_peer: None,
            cooperative_mla: HashMap::new(),
            cooperative_dsa_wq_b: HashMap::new(),
        }
    }

    pub fn ct(source: CompressedTensorsSource) -> Self {
        Self {
            archive: RocmExpertArchive::Ct(source),
            resident: HashMap::new(),
            fp8_resident: HashMap::new(),
            fp8_metas: HashMap::new(),
            gguf_grouped: HashMap::new(),
            mxfp4_grouped: HashMap::new(),
            ct_moe_graphs: HashMap::new(),
            cooperative_peer: None,
            cooperative_mla: HashMap::new(),
            cooperative_dsa_wq_b: HashMap::new(),
        }
    }

    pub fn enable_cooperative_peer(&mut self, context: RocmContext, local_partition: usize) -> Result<(), BackendError> {
        if !matches!(&self.archive, RocmExpertArchive::Ct(_) | RocmExpertArchive::Gguf(_)) {
            return Err(compute_error("ROCm cooperative experts 当前只支持 compressed-tensors/GGUF"));
        }
        if local_partition >= 2 {
            return Err(compute_error(format!("ROCm cooperative expert partition={local_partition} 非法")));
        }
        self.cooperative_peer = Some(RocmCooperativeExpertPeer { context, local_partition });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_cooperative_mla_layer(&mut self, layer: usize, owner_q_b: RocmWeight, peer_q_b: RocmWeight, owner_kv_b: RocmWeight, peer_kv_b: RocmWeight, owner_o: RocmWeight, peer_o: RocmWeight) -> Result<(), BackendError> {
        if self.cooperative_peer.is_none() {
            return Err(compute_error(format!("L{layer} cooperative MLA 缺少 peer")));
        }
        let full_o = owner_o.cols == owner_q_b.rows && peer_o.cols == owner_q_b.rows;
        let sharded_o = owner_o.cols == peer_o.cols && owner_o.cols.checked_add(peer_o.cols) == Some(owner_q_b.rows);
        if owner_q_b.rows != peer_q_b.rows || owner_q_b.cols != peer_q_b.cols || owner_kv_b.rows != peer_kv_b.rows || owner_kv_b.cols != peer_kv_b.cols || owner_o.rows != peer_o.rows || !(full_o || sharded_o) {
            return Err(compute_error(format!(
                "L{layer} cooperative MLA shard shape 非法: q={:?}/{:?} kv={:?}/{:?} o={:?}/{:?}",
                (owner_q_b.rows, owner_q_b.cols),
                (peer_q_b.rows, peer_q_b.cols),
                (owner_kv_b.rows, owner_kv_b.cols),
                (peer_kv_b.rows, peer_kv_b.cols),
                (owner_o.rows, owner_o.cols),
                (peer_o.rows, peer_o.cols),
            )));
        }
        let (owner_decode_q_b, peer_decode_q_b) = if owner_q_b.rows.is_multiple_of(2) {
            let half = owner_q_b.rows / 2;
            match (owner_q_b.w8_row_view(0..half)?, peer_q_b.w8_row_view(half..owner_q_b.rows)?) {
                (Some(owner), Some(peer)) => (Some(owner), Some(peer)),
                _ => (None, None),
            }
        } else {
            (None, None)
        };
        self.cooperative_mla.insert(layer, RocmCooperativeMlaWeights { owner_q_b, peer_q_b, owner_decode_q_b, peer_decode_q_b, owner_kv_b, peer_kv_b, owner_o, peer_o });
        Ok(())
    }

    pub(super) fn cooperative_mla_layer(&self, layer: usize) -> Result<(RocmContext, &RocmCooperativeMlaWeights), BackendError> {
        let peer = self.cooperative_peer.ok_or_else(|| compute_error(format!("L{layer} cooperative MLA peer 缺失")))?;
        let weights = self.cooperative_mla.get(&layer).ok_or_else(|| compute_error(format!("L{layer} cooperative MLA 权重缺失")))?;
        Ok((peer.context, weights))
    }

    pub fn set_cooperative_dsa_wq_b(&mut self, layer: usize, weight: RocmWeight) -> Result<(), BackendError> {
        if self.cooperative_peer.is_none() {
            return Err(compute_error(format!("L{layer} cooperative DSA wq_b 缺少 peer")));
        }
        self.cooperative_dsa_wq_b.insert(layer, weight);
        Ok(())
    }

    fn load(&mut self, backend: &RocmContext, layer: usize, expert: usize) -> Result<RocmPrefillExpert, BackendError> {
        match &mut self.archive {
            RocmExpertArchive::Fp8(source) => {
                // 与 Gguf/Ct/Mxfp4 相同的常驻缓存:否则每次 fallback 都会把
                // 288×3×8MiB 的 FP8 权重整层重新读盘上传(单请求 141GB H2D)。
                let key = (backend.device_id, layer, expert);
                if let Some(resident) = self.resident.get(&key) {
                    return Ok(RocmPrefillExpert::Resident(resident.clone()));
                }
                let weights = source.load_expert_fp8(layer, expert).map_err(BackendError::ExpertLoad)?;
                let resident = prepare_fp8_expert(backend, weights)?;
                self.resident.insert(key, resident.clone());
                Ok(RocmPrefillExpert::Resident(resident))
            }
            RocmExpertArchive::Fp8Source(source) => {
                let key = (backend.device_id, layer, expert);
                if let Some(resident) = self.resident.get(&key) {
                    return Ok(RocmPrefillExpert::Resident(resident.clone()));
                }
                let weights = source.load_expert_fp8(layer, expert).map_err(BackendError::ExpertLoad)?;
                let resident = prepare_fp8_expert(backend, weights)?;
                self.resident.insert(key, resident.clone());
                Ok(RocmPrefillExpert::Resident(resident))
            }
            RocmExpertArchive::Mxfp4(source) => {
                // MXFP4 以 4bit 原始形态驻留(12.6MB/专家),linear 时 in-kernel 反量化。
                let key = (backend.device_id, layer, expert);
                if let Some(resident) = self.resident.get(&key) {
                    return Ok(RocmPrefillExpert::Resident(resident.clone()));
                }
                let weights = source.load_expert_mxfp4(layer, expert).map_err(BackendError::ExpertLoad)?;
                let resident = Arc::new(prepare_mxfp4_expert(backend, &weights)?);
                self.resident.insert(key, resident.clone());
                Ok(RocmPrefillExpert::Resident(resident))
            }
            RocmExpertArchive::Nvfp4(source) => source.load_expert(layer, expert).map(RocmPrefillExpert::Nvfp4).map_err(BackendError::ExpertLoad),
            RocmExpertArchive::Gguf(source) => {
                // 与 CT 相同的常驻缓存：prepare 一次后驻留，prefill/decode 复用。
                let key = (backend.device_id, layer, expert);
                if let Some(resident) = self.resident.get(&key) {
                    return Ok(RocmPrefillExpert::Resident(resident.clone()));
                }
                let weights = source.load_expert_gguf(layer, expert).map_err(BackendError::ExpertLoad)?;
                let resident = prepare_gguf_expert(backend, &weights)?;
                self.resident.insert(key, resident.clone());
                Ok(RocmPrefillExpert::Resident(resident))
            }
            RocmExpertArchive::Ct(source) => {
                let key = (backend.device_id, layer, expert);
                if let Some(resident) = self.resident.get(&key) {
                    return Ok(RocmPrefillExpert::Resident(resident.clone()));
                }
                let weights = source.load_expert_ct(layer, expert).map_err(BackendError::ExpertLoad)?;
                let resident = prepare_ct_expert(backend, weights)?;
                self.resident.insert(key, resident.clone());
                Ok(RocmPrefillExpert::Resident(resident))
            }
        }
    }

    fn load_batch(&mut self, backend: &RocmContext, layer: usize, expert_ids: &[usize]) -> Result<Vec<RocmPrefillExpert>, BackendError> {
        if let RocmExpertArchive::Fp8(source) = &self.archive {
            let prepared = expert_ids
                .par_iter()
                .map(|&expert| {
                    if let Some(resident) = self.resident.get(&(backend.device_id, layer, expert)) {
                        return Ok((expert, resident.clone(), false));
                    }
                    let weights = source.load_expert_fp8(layer, expert).map_err(BackendError::ExpertLoad)?;
                    prepare_fp8_expert(backend, weights).map(|resident| (expert, resident, true))
                })
                .collect::<Result<Vec<_>, BackendError>>()?;
            if expert_ids.len() <= 8 {
                // decode 每层最多保留当前 top-8；prefill 的大活跃集只临时存在，
                // 避免官方 FP8 expert 吞掉长期 KV 预算。
                self.resident.retain(|&(device, resident_layer, expert), _| device != backend.device_id || resident_layer != layer || expert_ids.contains(&expert));
                for (expert, resident, fresh) in &prepared {
                    if *fresh {
                        self.resident.insert((backend.device_id, layer, *expert), resident.clone());
                    }
                }
            }
            return Ok(prepared.into_iter().map(|(_, resident, _)| RocmPrefillExpert::Resident(resident)).collect());
        }
        if let RocmExpertArchive::Ct(source) = &self.archive {
            let missing = expert_ids.iter().copied().filter(|&expert| !self.resident.contains_key(&(backend.device_id, layer, expert))).collect::<Vec<_>>();
            let load_width = ops::hip::options().expert_load_width;
            for chunk in missing.chunks(load_width) {
                let prepared = chunk
                    .par_iter()
                    .map(|&expert| {
                        let weights = source.load_expert_ct(layer, expert).map_err(BackendError::ExpertLoad)?;
                        prepare_ct_expert(backend, weights).map(|resident| (expert, resident))
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                for (expert, resident) in prepared {
                    self.resident.insert((backend.device_id, layer, expert), resident);
                }
            }
            return expert_ids
                .iter()
                .map(|&expert| self.resident.get(&(backend.device_id, layer, expert)).cloned().map(RocmPrefillExpert::Resident).ok_or_else(|| BackendError::ExpertLoad(format!("ROCm resident L{layer} E{expert} 缺失"))))
                .collect();
        }
        expert_ids.iter().map(|&expert| self.load(backend, layer, expert)).collect()
    }

    fn cooperative_resident_partition(&self, backend: &RocmContext, layer: usize, expert_count: usize, partition: usize) -> Result<Vec<Arc<RocmResidentExpert>>, BackendError> {
        if partition >= 2 {
            return Err(compute_error(format!("ROCm cooperative partition={partition} experts={expert_count} 非法")));
        }
        (0..expert_count)
            .map(|expert| self.resident.get(&(backend.device_id, layer, expert)).cloned().ok_or_else(|| compute_error(format!("ROCm cooperative resident device={} L{layer} E{expert} row-partition={partition} 缺失", backend.device_id))))
            .collect()
    }

    fn cooperative_resident_shared(&self, backend: &RocmContext, layer: usize, expert_count: usize, partition: usize) -> Result<Arc<RocmResidentExpert>, BackendError> {
        if partition >= 2 {
            return Err(compute_error(format!("ROCm cooperative shared partition={partition} 非法")));
        }
        self.resident.get(&(backend.device_id, layer, expert_count)).cloned().ok_or_else(|| compute_error(format!("ROCm cooperative shared resident device={} L{layer} row-partition={partition} 缺失", backend.device_id)))
    }

    /// MXFP4 层级大 buffer:全部专家 gate/up/down 拼段一次上传,grouped 消费。
    pub fn mxfp4_grouped(&mut self, backend: &RocmContext, layer: usize, spec: &crate::moe::topk_moe::TopkMoeSpec) -> Result<Arc<Mxfp4GroupedLayer>, BackendError> {
        let key = (backend.device_id, layer);
        if let Some(grouped) = self.mxfp4_grouped.get(&key) {
            return Ok(grouped.clone());
        }
        let RocmExpertArchive::Mxfp4(source) = &self.archive else {
            return Err(compute_error("mxfp4_grouped 仅支持 MXFP4 archive"));
        };
        let hidden = source.hidden();
        let intermediate = spec.intermediate_size;
        let expert_count = spec.num_experts;
        if source.hidden() != hidden || source.intermediate() != intermediate {
            return Err(compute_error(format!("MXFP4 grouped shape hidden={}/{} intermediate={}/{}", source.hidden(), hidden, source.intermediate(), intermediate)));
        }
        backend.activate().map_err(|error| compute_error(format!("激活设备: {error}")))?;
        let mut gate_up_packed = Vec::with_capacity(2 * expert_count * intermediate * (hidden / 2));
        let mut gate_up_scales = Vec::with_capacity(2 * expert_count * intermediate * (hidden / 32));
        let mut down_packed = Vec::with_capacity(expert_count * hidden * (intermediate / 2));
        let mut down_scales = Vec::with_capacity(expert_count * hidden * (intermediate / 32));
        // gate 段先整体铺满,up 段追加其后;单次 load 同时取三矩阵。
        for expert in 0..expert_count {
            let weights = source.load_expert_mxfp4(layer, expert).map_err(BackendError::ExpertLoad)?;
            let (packed, scales) = ops::hip::preshuffle_mxfp4_16x16(weights.gate.packed(), weights.gate.scales(), intermediate, hidden);
            gate_up_packed.extend_from_slice(&packed);
            gate_up_scales.extend_from_slice(&scales);
            let (packed, scales) = ops::hip::preshuffle_mxfp4_16x16(weights.down.packed(), weights.down.scales(), hidden, intermediate);
            down_packed.extend_from_slice(&packed);
            down_scales.extend_from_slice(&scales);
        }
        for expert in 0..expert_count {
            let weights = source.load_expert_mxfp4(layer, expert).map_err(BackendError::ExpertLoad)?;
            let (packed, scales) = ops::hip::preshuffle_mxfp4_16x16(weights.up.packed(), weights.up.scales(), intermediate, hidden);
            gate_up_packed.extend_from_slice(&packed);
            gate_up_scales.extend_from_slice(&scales);
        }
        let grouped = Arc::new(Mxfp4GroupedLayer {
            gate_up_packed: Arc::new(ops::hip::DeviceBuffer::upload(backend.device_id, &gate_up_packed).map_err(|error| compute_error(format!("MXFP4 grouped gate_up 上传: {error}")))?),
            gate_up_scales: Arc::new(ops::hip::DeviceBuffer::upload(backend.device_id, &gate_up_scales).map_err(|error| compute_error(format!("MXFP4 grouped scales 上传: {error}")))?),
            down_packed: Arc::new(ops::hip::DeviceBuffer::upload(backend.device_id, &down_packed).map_err(|error| compute_error(format!("MXFP4 grouped down 上传: {error}")))?),
            down_scales: Arc::new(ops::hip::DeviceBuffer::upload(backend.device_id, &down_scales).map_err(|error| compute_error(format!("MXFP4 grouped down scales 上传: {error}")))?),
        });
        self.mxfp4_grouped.insert(key, grouped.clone());
        Ok(grouped)
    }

    pub fn preload_layer(&mut self, backend: &RocmContext, layer: usize, expert_count: usize) -> Result<(), BackendError> {
        if let Some(peer) = self.cooperative_peer {
            let remote_partition = 1 - peer.local_partition;
            match &self.archive {
                RocmExpertArchive::Ct(source) => {
                    let source = source.clone();
                    let missing = (0..expert_count).filter(|&expert| !self.resident.contains_key(&(backend.device_id, layer, expert)) || !self.resident.contains_key(&(peer.context.device_id, layer, expert))).collect::<Vec<_>>();
                    let load_width = ops::hip::options().expert_load_width.max(1);
                    for chunk in missing.chunks(load_width) {
                        let prepared = chunk
                            .par_iter()
                            .map(|&expert| {
                                let weights = source.load_expert_ct(layer, expert).map_err(BackendError::ExpertLoad)?;
                                let (local, remote) = rayon::join(|| prepare_ct_expert_tp_shard(backend, &weights, peer.local_partition), || prepare_ct_expert_tp_shard(&peer.context, &weights, remote_partition));
                                Ok((expert, local?, remote?))
                            })
                            .collect::<Result<Vec<_>, BackendError>>()?;
                        for (expert, local, remote) in prepared {
                            self.resident.insert((backend.device_id, layer, expert), local);
                            self.resident.insert((peer.context.device_id, layer, expert), remote);
                        }
                    }
                    if !self.resident.contains_key(&(backend.device_id, layer, expert_count)) || !self.resident.contains_key(&(peer.context.device_id, layer, expert_count)) {
                        let weights = load_ct_shared_expert(&source, layer).map_err(BackendError::ExpertLoad)?;
                        let (local, remote) = rayon::join(|| prepare_ct_expert_tp_shard(backend, &weights, peer.local_partition), || prepare_ct_expert_tp_shard(&peer.context, &weights, remote_partition));
                        self.resident.insert((backend.device_id, layer, expert_count), local?);
                        self.resident.insert((peer.context.device_id, layer, expert_count), remote?);
                    }
                    let local = self.cooperative_resident_partition(backend, layer, expert_count, peer.local_partition)?;
                    let remote = self.cooperative_resident_partition(&peer.context, layer, expert_count, remote_partition)?;
                    ops::hip::preload_ct_grouped_expert_metas(backend.device_id, &grouped_w4_experts(&local)?).map_err(compute_error)?;
                    ops::hip::preload_ct_grouped_expert_metas(peer.context.device_id, &grouped_w4_experts(&remote)?).map_err(compute_error)?;
                    let local_shared = self.cooperative_resident_shared(backend, layer, expert_count, peer.local_partition)?;
                    let remote_shared = self.cooperative_resident_shared(&peer.context, layer, expert_count, remote_partition)?;
                    ops::hip::preload_ct_grouped_expert_metas(backend.device_id, &grouped_w4_experts(std::slice::from_ref(&local_shared))?).map_err(compute_error)?;
                    ops::hip::preload_ct_grouped_expert_metas(peer.context.device_id, &grouped_w4_experts(std::slice::from_ref(&remote_shared))?).map_err(compute_error)?;
                }
                RocmExpertArchive::Gguf(source) => {
                    let source = source.clone();
                    let missing = (0..expert_count).filter(|&expert| !self.resident.contains_key(&(backend.device_id, layer, expert)) || !self.resident.contains_key(&(peer.context.device_id, layer, expert))).collect::<Vec<_>>();
                    let load_width = ops::hip::options().expert_load_width.max(1);
                    for chunk in missing.chunks(load_width) {
                        let prepared = chunk
                            .par_iter()
                            .map(|&expert| {
                                let weights = source.load_expert_gguf(layer, expert).map_err(BackendError::ExpertLoad)?;
                                let (local, remote) = prepare_gguf_expert_tp_shards(backend, &peer.context, &weights, peer.local_partition)?;
                                Ok((expert, local, remote))
                            })
                            .collect::<Result<Vec<_>, BackendError>>()?;
                        for (expert, local, remote) in prepared {
                            self.resident.insert((backend.device_id, layer, expert), local);
                            self.resident.insert((peer.context.device_id, layer, expert), remote);
                        }
                    }
                    if !self.resident.contains_key(&(backend.device_id, layer, expert_count)) || !self.resident.contains_key(&(peer.context.device_id, layer, expert_count)) {
                        let weights = source.load_shared_expert_gguf(layer).map_err(BackendError::ExpertLoad)?;
                        let (local, remote) = prepare_gguf_expert_tp_shards(backend, &peer.context, &weights, peer.local_partition)?;
                        self.resident.insert((backend.device_id, layer, expert_count), local);
                        self.resident.insert((peer.context.device_id, layer, expert_count), remote);
                    }
                    let local = self.cooperative_resident_partition(backend, layer, expert_count, peer.local_partition)?;
                    let remote = self.cooperative_resident_partition(&peer.context, layer, expert_count, remote_partition)?;
                    let local_shared = self.cooperative_resident_shared(backend, layer, expert_count, peer.local_partition)?;
                    let remote_shared = self.cooperative_resident_shared(&peer.context, layer, expert_count, remote_partition)?;
                    let local_routed = gguf_grouped_metas_from_residents(layer, &local)?;
                    let remote_routed = gguf_grouped_metas_from_residents(layer, &remote)?;
                    let local_shared = gguf_grouped_metas_from_residents(layer, std::slice::from_ref(&local_shared))?;
                    let remote_shared = gguf_grouped_metas_from_residents(layer, std::slice::from_ref(&remote_shared))?;
                    let local_grouped = RocmGgufGroupedLayer {
                        routed: ops::hip::resident_gguf_grouped_metas(backend.device_id, &local_routed).map_err(compute_error)?,
                        shared: Some(ops::hip::resident_gguf_grouped_metas(backend.device_id, &local_shared).map_err(compute_error)?),
                    };
                    let remote_grouped = RocmGgufGroupedLayer {
                        routed: ops::hip::resident_gguf_grouped_metas(peer.context.device_id, &remote_routed).map_err(compute_error)?,
                        shared: Some(ops::hip::resident_gguf_grouped_metas(peer.context.device_id, &remote_shared).map_err(compute_error)?),
                    };
                    self.gguf_grouped.insert((backend.device_id, layer), local_grouped);
                    self.gguf_grouped.insert((peer.context.device_id, layer), remote_grouped);
                }
                _ => return Err(compute_error("ROCm cooperative row shard 只支持 compressed-tensors/GGUF")),
            }
            return Ok(());
        }
        if matches!(self.archive, RocmExpertArchive::Fp8(_) | RocmExpertArchive::Fp8Source(_)) {
            let missing = (0..expert_count).filter(|&expert| !self.fp8_resident.contains_key(&(backend.device_id, layer, expert))).collect::<Vec<_>>();
            let load_width = ops::hip::options().expert_load_width.max(1);
            for chunk in missing.chunks(load_width) {
                let prepared = match &self.archive {
                    RocmExpertArchive::Fp8(source) => chunk
                        .par_iter()
                        .map(|&expert| {
                            let weights = source.load_expert_fp8(layer, expert).map_err(BackendError::ExpertLoad)?;
                            prepare_fp8_packed_expert(backend, weights).map(|resident| (expert, resident))
                        })
                        .collect::<Result<Vec<_>, BackendError>>()?,
                    RocmExpertArchive::Fp8Source(source) => chunk
                        .par_iter()
                        .map(|&expert| {
                            let weights = source.load_expert_fp8(layer, expert).map_err(BackendError::ExpertLoad)?;
                            prepare_fp8_packed_expert(backend, weights).map(|resident| (expert, resident))
                        })
                        .collect::<Result<Vec<_>, BackendError>>()?,
                    _ => unreachable!("FP8 archive 已由 matches 校验"),
                };
                for (expert, resident) in prepared {
                    self.fp8_resident.insert((backend.device_id, layer, expert), resident);
                }
            }
            // grouped kernel 消费的 device pointer 表也是模型常量，随专家一起
            // 预载；forward 只读常驻表，不能在首次路由时再从 host 上传。
            self.fp8_grouped_metas(backend.device_id, layer, expert_count)?;
            let grouped = self.fp8_grouped(backend.device_id, layer, expert_count)?;
            ops::hip::preload_ct_grouped_expert_metas(backend.device_id, &grouped).map_err(compute_error)?;
            return Ok(());
        }
        let expert_ids = (0..expert_count).collect::<Vec<_>>();
        self.load_batch(backend, layer, &expert_ids)?;
        if matches!(self.archive, RocmExpertArchive::Gguf(_)) {
            let metas = self.gguf_grouped_metas(backend.device_id, layer, expert_count)?;
            let routed = ops::hip::resident_gguf_grouped_metas(backend.device_id, &metas).map_err(compute_error)?;
            self.gguf_grouped.insert((backend.device_id, layer), RocmGgufGroupedLayer { routed, shared: None });
        }
        Ok(())
    }

    fn cooperative_gguf_grouped(&self, device: i32, layer: usize) -> Result<&RocmGgufGroupedLayer, BackendError> {
        self.gguf_grouped.get(&(device, layer)).ok_or_else(|| compute_error(format!("ROCm cooperative GGUF grouped device={device} L{layer} 未预载")))
    }

    fn gguf_routed_grouped(&self, device: i32, layer: usize) -> Result<&ops::hip::GgufGroupedMetas, BackendError> {
        self.gguf_grouped.get(&(device, layer)).map(|grouped| &grouped.routed).ok_or_else(|| compute_error(format!("ROCm GGUF grouped device={device} L{layer} 未预载")))
    }

    fn gguf_shared_grouped(&mut self, device: i32, layer: usize, shared: &SharedExpertRef<'_, RocmWeight>) -> Result<Option<ops::hip::GgufGroupedMetas>, BackendError> {
        let grouped = self.gguf_grouped.get_mut(&(device, layer)).ok_or_else(|| compute_error(format!("ROCm GGUF grouped device={device} L{layer} 未预载")))?;
        if let Some(shared) = &grouped.shared {
            return Ok(Some(shared.clone()));
        }
        if ![shared.gate, shared.up, shared.down].iter().all(|weight| weight.expert_gguf().is_some()) {
            return Ok(None);
        }
        let meta = gguf_grouped_meta_from_weights(layer, shared.gate, shared.up, shared.down)?;
        let resident = ops::hip::resident_gguf_grouped_metas(device, &[meta]).map_err(compute_error)?;
        grouped.shared = Some(resident.clone());
        Ok(Some(resident))
    }

    fn fp8_grouped_metas(&mut self, device: i32, layer: usize, expert_count: usize) -> Result<Option<Arc<ops::hip::DeviceBuffer>>, BackendError> {
        if !matches!(self.archive, RocmExpertArchive::Fp8(_) | RocmExpertArchive::Fp8Source(_)) {
            return Ok(None);
        }
        let key = (device, layer);
        if let Some(metas) = self.fp8_metas.get(&key) {
            return Ok(Some(metas.clone()));
        }
        let metas = (0..expert_count)
            .map(|expert| {
                let resident = self.fp8_resident.get(&(device, layer, expert)).ok_or_else(|| BackendError::ExpertLoad(format!("FP8 grouped L{layer} E{expert} 未常驻")))?;
                let weight = |weight: &RocmFp8Weight| ops::hip::Fp8GroupedWeightMeta { codes: weight.codes.device_pointer() as u64, scales: weight.scales.device_pointer() as u64 };
                Ok(ops::hip::Fp8GroupedExpertMeta { gate: weight(&resident.gate), up: weight(&resident.up), down: weight(&resident.down) })
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let bytes = unsafe { std::slice::from_raw_parts(metas.as_ptr().cast::<u8>(), std::mem::size_of_val(metas.as_slice())) };
        let metas = Arc::new(ops::hip::DeviceBuffer::upload(device, bytes).map_err(compute_error)?);
        self.fp8_metas.insert(key, metas.clone());
        Ok(Some(metas))
    }

    /// official FP8 只暴露常驻 codes/scale；grouped WMMA 在装填 16x16 tile 时解码。
    fn fp8_grouped(&self, device: i32, layer: usize, expert_count: usize) -> Result<Vec<ops::hip::CtGroupedExpertRef<'_>>, BackendError> {
        (0..expert_count)
            .map(|expert| {
                let resident = self.fp8_resident.get(&(device, layer, expert)).ok_or_else(|| BackendError::ExpertLoad(format!("FP8 grouped L{layer} E{expert} 未常驻")))?;
                Ok(ops::hip::CtGroupedExpertRef { gate: grouped_fp8(&resident.gate), up: grouped_fp8(&resident.up), down: grouped_fp8(&resident.down) })
            })
            .collect()
    }

    /// 从常驻缓存构造 grouped kernel 的 meta 列表(设备指针 + 量化类型)。
    fn gguf_grouped_metas(&self, device: i32, layer: usize, expert_count: usize) -> Result<Vec<ops::hip::GgufGroupedExpertMeta>, BackendError> {
        let resident = (0..expert_count).map(|expert| self.resident.get(&(device, layer, expert)).cloned().ok_or_else(|| BackendError::ExpertLoad(format!("GGUF grouped L{layer} E{expert} 未常驻")))).collect::<Result<Vec<_>, _>>()?;
        gguf_grouped_metas_from_residents(layer, &resident)
    }

    fn resident_ct_layer(&self, device: i32, layer: usize, expert_count: usize) -> Option<Vec<Arc<RocmResidentExpert>>> {
        if !matches!(self.archive, RocmExpertArchive::Ct(_)) {
            return None;
        }
        (0..expert_count).map(|expert| self.resident.get(&(device, layer, expert)).cloned()).collect()
    }
}
