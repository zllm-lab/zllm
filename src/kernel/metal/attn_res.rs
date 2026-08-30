//! AttnRes Metal 算子。

/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: attn_res_pack_f16, attn_res_mix_f16
pub const SHADERS: &str = r#"
kernel void attn_res_pack_f16(
    device const half *input [[buffer(0)]],
    device half *packed [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant ulong &offset [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) packed[offset + index] = input[index];
}
kernel void attn_res_mix_f16(
    device const half *candidates [[buffer(0)]],
    device const half *norm_weight [[buffer(1)]],
    device const half *projection_weight [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &candidate_count [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint thread_count [[threads_per_threadgroup]])
{
    if (row >= rows || candidate_count == 0 || candidate_count > 32) return;
    threadgroup float2 reductions[256];
    threadgroup float probabilities[32];

    for (uint candidate = 0; candidate < candidate_count; ++candidate) {
        const ulong base = (ulong(candidate) * rows + row) * columns;
        float sum_squares = 0.0f;
        float projection = 0.0f;
        for (uint column = lane; column < columns; column += thread_count) {
            const float value = float(candidates[base + column]);
            sum_squares += value * value;
            projection += value * float(norm_weight[column]) * float(projection_weight[column]);
        }
        reductions[lane] = float2(sum_squares, projection);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint active = thread_count; active > 1;) {
            const uint half_active = (active + 1) >> 1;
            if (lane < half_active && lane + half_active < active) {
                reductions[lane] += reductions[lane + half_active];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            active = half_active;
        }
        if (lane == 0) {
            const float inv_rms = rsqrt(reductions[0].x / float(columns) + eps);
            probabilities[candidate] = reductions[0].y * inv_rms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        float maximum = -INFINITY;
        for (uint candidate = 0; candidate < candidate_count; ++candidate) {
            maximum = max(maximum, probabilities[candidate]);
        }
        float denominator = 0.0f;
        for (uint candidate = 0; candidate < candidate_count; ++candidate) {
            probabilities[candidate] = exp(probabilities[candidate] - maximum);
            denominator += probabilities[candidate];
        }
        const float inverse = 1.0f / denominator;
        for (uint candidate = 0; candidate < candidate_count; ++candidate) {
            probabilities[candidate] *= inverse;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint column = lane; column < columns; column += thread_count) {
        float value = 0.0f;
        for (uint candidate = 0; candidate < candidate_count; ++candidate) {
            const ulong index = (ulong(candidate) * rows + row) * columns + column;
            value += probabilities[candidate] * float(candidates[index]);
        }
        output[ulong(row) * columns + column] = half(value);
    }
}
"#;

use super::{launch_1d, launch_rows_with_pipeline, set_bytes, validate_u32};
use crate::backend::metal::{MetalContext, MetalTensor, MetalTensorDType};

const MAX_CANDIDATES: usize = 32;

pub fn mix_tensor(ctx: &MetalContext, current: &MetalTensor, block_residuals: &[MetalTensor], norm_weight: &MetalTensor, projection_weight: &MetalTensor, eps: f32) -> Result<MetalTensor, String> {
    if current.rows == 0 || current.cols == 0 || current.dtype != MetalTensorDType::F16 {
        return Err("Metal AttnRes current 需要非空 F16 tensor".to_owned());
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(format!("Metal AttnRes eps 非法: {eps}"));
    }
    if norm_weight.dtype != MetalTensorDType::F16 || projection_weight.dtype != MetalTensorDType::F16 || (norm_weight.rows, norm_weight.cols) != (1, current.cols) || (projection_weight.rows, projection_weight.cols) != (1, current.cols) {
        return Err("Metal AttnRes weight 需要 [1, hidden] F16 tensor".to_owned());
    }
    if block_residuals.iter().any(|tensor| tensor.rows != current.rows || tensor.cols != current.cols || tensor.dtype != MetalTensorDType::F16) {
        return Err("Metal AttnRes block residual shape/dtype 不一致".to_owned());
    }

    let candidate_count = block_residuals.len() + 1;
    if candidate_count > MAX_CANDIDATES {
        return Err(format!("Metal AttnRes candidate 数量 {candidate_count} 超过 kernel 上限 {MAX_CANDIDATES}"));
    }
    let candidate_elements = current.len();
    let packed_rows = current.rows.checked_mul(candidate_count).ok_or("Metal AttnRes packed rows 溢出")?;
    let packed = ctx.tensor_uninit(packed_rows, current.cols);
    let count = validate_u32("AttnRes candidate elements", candidate_elements)?;
    let candidates = block_residuals.iter().chain(std::iter::once(current));
    for (slot, candidate) in candidates.enumerate() {
        let offset = slot.checked_mul(candidate_elements).ok_or("Metal AttnRes candidate offset 溢出")? as u64;
        let shape = format!("slot={slot},candidate=[{},{}]", current.rows, current.cols);
        launch_1d(ctx, "attn_res_pack_f16", &shape, candidate_elements, candidate.buffer.length(), candidate.buffer.length(), |encoder| {
            encoder.set_buffer(0, Some(&candidate.buffer), 0);
            encoder.set_buffer(1, Some(&packed.buffer), 0);
            set_bytes(encoder, 2, &count);
            set_bytes(encoder, 3, &offset);
        })?;
    }

    let output = ctx.tensor_kernel_output(current.rows, current.cols);
    let rows = validate_u32("AttnRes rows", current.rows)?;
    let columns = validate_u32("AttnRes columns", current.cols)?;
    let candidate_count = validate_u32("AttnRes candidates", candidate_count)?;
    launch_rows_with_pipeline(ctx, "attn_res_mix_f16", current.rows, current.cols, packed.buffer.length() + norm_weight.buffer.length() + projection_weight.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&packed.buffer), 0);
        encoder.set_buffer(1, Some(&norm_weight.buffer), 0);
        encoder.set_buffer(2, Some(&projection_weight.buffer), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        set_bytes(encoder, 4, &rows);
        set_bytes(encoder, 5, &columns);
        set_bytes(encoder, 6, &candidate_count);
        set_bytes(encoder, 7, &eps);
    })?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::cpu::{CpuTensor, attn_res as cpu_attn_res};

    #[test]
    fn metal_attn_res_matches_cpu() {
        if crate::kernel::metal::metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let current_values = [3.0, 5.0, 2.0, 4.0];
        let residual_values = [1.0, 3.0, 4.0, 0.0];
        let norm_values = [1.0, 0.5];
        let projection_values = [0.25, -0.75];
        let current = ctx.tensor_from_f32(&current_values, 2, 2).unwrap();
        let residual = ctx.tensor_from_f32(&residual_values, 2, 2).unwrap();
        let norm = ctx.tensor_from_f32(&norm_values, 1, 2).unwrap();
        let projection = ctx.tensor_from_f32(&projection_values, 1, 2).unwrap();
        let actual = mix_tensor(&ctx, &current, &[residual], &norm, &projection, 1.0e-5).unwrap();
        let expected = cpu_attn_res::mix(&CpuTensor { data: current_values.to_vec(), rows: 2, cols: 2 }, &[CpuTensor { data: residual_values.to_vec(), rows: 2, cols: 2 }], &norm_values, &projection_values, 1.0e-5).unwrap();
        for (actual, expected) in ctx.tensor_to_f32(&actual).iter().zip(expected.data) {
            assert!((actual - expected).abs() < 2.0e-3, "actual={actual}, expected={expected}");
        }
    }
}
