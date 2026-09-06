//! ROCm expert 的显式 CPU/reference 回退。

use super::*;

pub(super) fn route_cpu(input: &[f32], weight: &[f32], bias: &[f32], spec: &TopkMoeSpec) -> Result<crate::moe::routing::Routing, String> {
    match spec.scoring_func {
        crate::moe::topk_moe::ScoringFunc::Softmax => ops::routing::route_softmax(input, weight, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected),
        crate::moe::topk_moe::ScoringFunc::SigmoidBias => ops::routing::route_sigmoid_bias(input, weight, bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor),
        crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias => ops::routing::route_sqrt_softplus_bias(input, weight, bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor),
    }
}

fn matmul(ctx: &RocmContext, input: &RocmTensor, weight: &[f32], output_columns: usize) -> Result<RocmTensor, BackendError> {
    let weight_elements = input.cols.checked_mul(output_columns).ok_or_else(|| compute_error("ROCm reference expert weight 维度溢出"))?;
    if weight.len() != weight_elements {
        return Err(compute_error(format!("ROCm reference expert weight={}，期望 [{output_columns},{}]", weight.len(), input.cols)));
    }
    let output_elements = input.rows.checked_mul(output_columns).ok_or_else(|| compute_error("ROCm reference expert output 维度溢出"))?;
    let input_data = tensor_data(input)?;
    let mut output = vec![0.0; output_elements];
    if output_elements != 0 {
        ops::hip::try_sgemm_f32(ctx.device_id, &input_data, weight, input.rows, input.cols, output_columns, &mut output).map_err(compute_error)?;
    }
    Ok(RocmTensor { data: output, rows: input.rows, cols: output_columns, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: None, replica: None })
}

fn activate(gate: &RocmTensor, up: &RocmTensor, activation: &Activation) -> Result<RocmTensor, BackendError> {
    let mut output = vec![0.0; gate.data.len()];
    crate::kernel::rocm::hip::try_gated_activation_f32(&gate.data, &up.data, gate.rows, gate.cols, activation, &mut output).map_err(compute_error)?;
    Ok(RocmTensor { data: output, rows: gate.rows, cols: gate.cols, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: None, replica: None })
}

pub(super) fn f32_expert(ctx: &RocmContext, input: &RocmTensor, gate_weight: &[f32], up_weight: &[f32], down_weight: &[f32], intermediate: usize, activation: &Activation) -> Result<RocmTensor, BackendError> {
    ctx.require_cpu_reference_fallback("non-resident F32 expert")?;
    let gate = matmul(ctx, input, gate_weight, intermediate)?;
    let up = matmul(ctx, input, up_weight, intermediate)?;
    let activated = activate(&gate, &up, activation)?;
    let output = matmul(ctx, &activated, down_weight, input.cols)?;
    ctx.tensor_from_f32(output.data, output.rows, output.cols).map_err(compute_error)
}

pub(super) fn nvfp4_expert(ctx: &RocmContext, input: &RocmTensor, weights: &crate::weight::format::nvfp4::Nvfp4ExpertWeights, activation: &Activation) -> Result<RocmTensor, BackendError> {
    ctx.require_cpu_reference_fallback("non-resident NVFP4 expert")?;
    let mut scratch = Vec::new();
    let gate = nvfp4_matmul(ctx, input, &weights.gate, &mut scratch)?;
    let up = nvfp4_matmul(ctx, input, &weights.up, &mut scratch)?;
    let activated = activate(&gate, &up, activation)?;
    let output = nvfp4_matmul(ctx, &activated, &weights.down, &mut scratch)?;
    ctx.tensor_from_f32(output.data, output.rows, output.cols).map_err(compute_error)
}

fn nvfp4_matmul(ctx: &RocmContext, input: &RocmTensor, weight: &crate::weight::format::nvfp4::Nvfp4Matrix, scratch: &mut Vec<f32>) -> Result<RocmTensor, BackendError> {
    let expected = weight.rows.checked_mul(weight.cols).ok_or_else(|| compute_error("ROCm NVFP4 reference weight 元素数量溢出"))?;
    scratch.resize(expected, 0.0);
    crate::kernel::cpu::nvfp4::decode_nvfp4_matrix(weight.codes(), weight.scales(), weight.global_scale, weight.rows, weight.cols, scratch).map_err(compute_error)?;
    matmul(ctx, input, scratch, weight.rows)
}
