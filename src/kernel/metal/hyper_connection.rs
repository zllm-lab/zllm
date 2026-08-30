//! mHC 的 Metal 张量变换；所有中间结果留在设备端 F32 buffer。

/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: mhc_expand_f32, mhc_reduce_f32, mhc_expand_scaled_f32, mhc_mix_f32, mhc_split_f32, mhc_head_reduce_f32
// private helpers: mhc_load
pub const SHADERS: &str = r#"
inline float mhc_load(device const uchar *values, uint dtype, ulong index) {
    if (dtype == 0) return float(reinterpret_cast<device const half *>(values)[index]);
    if (dtype == 1) {
        const uint bits = uint(reinterpret_cast<device const ushort *>(values)[index]) << 16;
        return as_type<float>(bits);
    }
    return reinterpret_cast<device const float *>(values)[index];
}
kernel void mhc_expand_f32(
    device const uchar *hidden [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &columns [[buffer(2)]],
    constant uint &copies [[buffer(3)]],
    constant uint &hidden_dtype [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint output_columns = columns * copies;
    const uint row = gid / output_columns;
    const uint column = gid - row * output_columns;
    output[gid] = mhc_load(hidden, hidden_dtype, ulong(row) * columns + column % columns);
}
kernel void mhc_reduce_f32(
    device const uchar *hidden [[buffer(0)]],
    device const uchar *coefficients [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &copies [[buffer(4)]],
    constant uint &hidden_dtype [[buffer(5)]],
    constant uint &coefficient_dtype [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint row = gid / width;
    const uint column = gid - row * width;
    const ulong hidden_row = ulong(row) * copies * width;
    const ulong coefficient_row = ulong(row) * copies;
    float sum = 0.0f;
    for (uint copy = 0; copy < copies; ++copy) {
        sum += mhc_load(coefficients, coefficient_dtype, coefficient_row + copy)
            * mhc_load(hidden, hidden_dtype, hidden_row + ulong(copy) * width + column);
    }
    output[gid] = sum;
}
kernel void mhc_expand_scaled_f32(
    device const uchar *hidden [[buffer(0)]],
    device const uchar *coefficients [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &copies [[buffer(4)]],
    constant uint &hidden_dtype [[buffer(5)]],
    constant uint &coefficient_dtype [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint output_columns = width * copies;
    const uint row = gid / output_columns;
    const uint column = gid - row * output_columns;
    const uint copy = column / width;
    const uint source_column = column - copy * width;
    output[gid] = mhc_load(coefficients, coefficient_dtype, ulong(row) * copies + copy)
        * mhc_load(hidden, hidden_dtype, ulong(row) * width + source_column);
}
kernel void mhc_mix_f32(
    device const uchar *hidden [[buffer(0)]],
    device const uchar *matrix [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &copies [[buffer(4)]],
    constant uint &hidden_dtype [[buffer(5)]],
    constant uint &matrix_dtype [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint output_columns = width * copies;
    const uint row = gid / output_columns;
    const uint column = gid - row * output_columns;
    const uint output_copy = column / width;
    const uint hidden_column = column - output_copy * width;
    const ulong hidden_row = ulong(row) * output_columns;
    const ulong matrix_row = ulong(row) * copies * copies + ulong(output_copy) * copies;
    float sum = 0.0f;
    for (uint input_copy = 0; input_copy < copies; ++input_copy) {
        sum += mhc_load(matrix, matrix_dtype, matrix_row + input_copy)
            * mhc_load(hidden, hidden_dtype, hidden_row + ulong(input_copy) * width + hidden_column);
    }
    output[gid] = sum;
}
kernel void mhc_split_f32(
    device const uchar *mixes [[buffer(0)]],
    device const float *base [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device float *pre [[buffer(3)]],
    device float *post [[buffer(4)]],
    device float *combination [[buffer(5)]],
    constant uint &copies [[buffer(6)]],
    constant uint &iterations [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    constant uint &mixes_dtype [[buffer(9)]],
    constant uint &rows [[buffer(10)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const uint mix_columns = (2 + copies) * copies;
    const ulong mix_row = ulong(row) * mix_columns;
    const ulong coefficient_row = ulong(row) * copies;
    const ulong matrix_row = ulong(row) * copies * copies;

    for (uint copy = 0; copy < copies; ++copy) {
        const float pre_logit = mhc_load(mixes, mixes_dtype, mix_row + copy) * scale[0] + base[copy];
        pre[coefficient_row + copy] = 1.0f / (1.0f + exp(-pre_logit)) + eps;
        const uint post_offset = copies + copy;
        const float post_logit = mhc_load(mixes, mixes_dtype, mix_row + post_offset) * scale[1] + base[post_offset];
        post[coefficient_row + copy] = 2.0f / (1.0f + exp(-post_logit));
    }

    for (uint output_copy = 0; output_copy < copies; ++output_copy) {
        float maximum = -INFINITY;
        for (uint input_copy = 0; input_copy < copies; ++input_copy) {
            const uint index = output_copy * copies + input_copy;
            const uint offset = 2 * copies + index;
            const float logit = mhc_load(mixes, mixes_dtype, mix_row + offset) * scale[2] + base[offset];
            maximum = max(maximum, logit);
        }
        float sum = 0.0f;
        for (uint input_copy = 0; input_copy < copies; ++input_copy) {
            const uint index = output_copy * copies + input_copy;
            const uint offset = 2 * copies + index;
            const float logit = mhc_load(mixes, mixes_dtype, mix_row + offset) * scale[2] + base[offset];
            const float value = exp(logit - maximum);
            combination[matrix_row + index] = value;
            sum += value;
        }
        for (uint input_copy = 0; input_copy < copies; ++input_copy) {
            const uint index = output_copy * copies + input_copy;
            combination[matrix_row + index] = combination[matrix_row + index] / sum + eps;
        }
    }
    for (uint input_copy = 0; input_copy < copies; ++input_copy) {
        float sum = 0.0f;
        for (uint output_copy = 0; output_copy < copies; ++output_copy) {
            sum += combination[matrix_row + output_copy * copies + input_copy];
        }
        for (uint output_copy = 0; output_copy < copies; ++output_copy) {
            const ulong index = matrix_row + output_copy * copies + input_copy;
            combination[index] /= sum + eps;
        }
    }
    for (uint iteration = 1; iteration < iterations; ++iteration) {
        for (uint output_copy = 0; output_copy < copies; ++output_copy) {
            float sum = 0.0f;
            for (uint input_copy = 0; input_copy < copies; ++input_copy) {
                sum += combination[matrix_row + output_copy * copies + input_copy];
            }
            for (uint input_copy = 0; input_copy < copies; ++input_copy) {
                const ulong index = matrix_row + output_copy * copies + input_copy;
                combination[index] /= sum + eps;
            }
        }
        for (uint input_copy = 0; input_copy < copies; ++input_copy) {
            float sum = 0.0f;
            for (uint output_copy = 0; output_copy < copies; ++output_copy) {
                sum += combination[matrix_row + output_copy * copies + input_copy];
            }
            for (uint output_copy = 0; output_copy < copies; ++output_copy) {
                const ulong index = matrix_row + output_copy * copies + input_copy;
                combination[index] /= sum + eps;
            }
        }
    }
}
kernel void mhc_head_reduce_f32(
    device const uchar *hidden [[buffer(0)]],
    device const uchar *mixes [[buffer(1)]],
    device const float *base [[buffer(2)]],
    device const float *scale [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &width [[buffer(5)]],
    constant uint &copies [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant uint &hidden_dtype [[buffer(8)]],
    constant uint &mixes_dtype [[buffer(9)]],
    constant uint &count [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint row = gid / width;
    const uint column = gid - row * width;
    const ulong hidden_row = ulong(row) * copies * width;
    const ulong mix_row = ulong(row) * copies;
    float sum = 0.0f;
    for (uint copy = 0; copy < copies; ++copy) {
        const float logit = mhc_load(mixes, mixes_dtype, mix_row + copy) * scale[0] + base[copy];
        const float coefficient = 1.0f / (1.0f + exp(-logit)) + eps;
        sum += coefficient * mhc_load(hidden, hidden_dtype, hidden_row + ulong(copy) * width + column);
    }
    output[gid] = sum;
}
"#;

use super::{MetalContext, MetalTensor, MetalTensorDType, launch_1d, set_bytes, validate_u32};
use crate::backend::metal::api::Buffer;

fn dtype_code(dtype: MetalTensorDType) -> u32 {
    match dtype {
        MetalTensorDType::F16 => 0,
        MetalTensorDType::Bf16 => 1,
        MetalTensorDType::F32 => 2,
    }
}

pub fn expand_tensor(ctx: &MetalContext, hidden: &MetalTensor, copies: usize) -> Result<MetalTensor, String> {
    if hidden.rows == 0 || hidden.cols == 0 || copies == 0 {
        return Err(format!("Metal mHC expand shape 非法: hidden={}x{} copies={copies}", hidden.rows, hidden.cols));
    }
    let output_columns = hidden.cols.checked_mul(copies).ok_or("Metal mHC expand columns 溢出")?;
    let output = ctx.tensor_kernel_output_f32(hidden.rows, output_columns);
    let columns = validate_u32("mHC expand columns", hidden.cols)?;
    let copies = validate_u32("mHC expand copies", copies)?;
    let count = validate_u32("mHC expand count", output.len())?;
    let input_dtype = dtype_code(hidden.dtype);
    let shape = format!("hidden=[{},{}],copies={copies}", hidden.rows, hidden.cols);
    launch_1d(ctx, "mhc_expand_f32", &shape, output.len(), hidden.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&hidden.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &columns);
        set_bytes(encoder, 3, &copies);
        set_bytes(encoder, 4, &input_dtype);
        set_bytes(encoder, 5, &count);
    })?;
    Ok(output)
}

pub fn reduce_tensor(ctx: &MetalContext, hidden: &MetalTensor, coefficients: &MetalTensor, copies: usize) -> Result<MetalTensor, String> {
    if hidden.rows == 0 || hidden.rows != coefficients.rows || copies == 0 || !hidden.cols.is_multiple_of(copies) || coefficients.cols != copies {
        return Err(format!("Metal mHC reduce shape 不一致: hidden={}x{} coefficients={}x{} copies={copies}", hidden.rows, hidden.cols, coefficients.rows, coefficients.cols));
    }
    let width = hidden.cols / copies;
    let output = ctx.tensor_kernel_output_f32(hidden.rows, width);
    let width = validate_u32("mHC reduce width", width)?;
    let copies = validate_u32("mHC reduce copies", copies)?;
    let hidden_dtype = dtype_code(hidden.dtype);
    let coefficient_dtype = dtype_code(coefficients.dtype);
    let count = validate_u32("mHC reduce count", output.len())?;
    let shape = format!("hidden=[{},{}],copies={copies}", hidden.rows, hidden.cols);
    launch_1d(ctx, "mhc_reduce_f32", &shape, output.len(), hidden.buffer.length() + coefficients.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&hidden.buffer), 0);
        encoder.set_buffer(1, Some(&coefficients.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &width);
        set_bytes(encoder, 4, &copies);
        set_bytes(encoder, 5, &hidden_dtype);
        set_bytes(encoder, 6, &coefficient_dtype);
        set_bytes(encoder, 7, &count);
    })?;
    Ok(output)
}

pub fn expand_scaled_tensor(ctx: &MetalContext, hidden: &MetalTensor, coefficients: &MetalTensor, copies: usize) -> Result<MetalTensor, String> {
    if hidden.rows == 0 || hidden.cols == 0 || hidden.rows != coefficients.rows || copies == 0 || coefficients.cols != copies {
        return Err(format!("Metal mHC expand_scaled shape 不一致: hidden={}x{} coefficients={}x{} copies={copies}", hidden.rows, hidden.cols, coefficients.rows, coefficients.cols));
    }
    let output_columns = hidden.cols.checked_mul(copies).ok_or("Metal mHC expand_scaled columns 溢出")?;
    let output = ctx.tensor_kernel_output_f32(hidden.rows, output_columns);
    let width = validate_u32("mHC expand_scaled width", hidden.cols)?;
    let copies = validate_u32("mHC expand_scaled copies", copies)?;
    let hidden_dtype = dtype_code(hidden.dtype);
    let coefficient_dtype = dtype_code(coefficients.dtype);
    let count = validate_u32("mHC expand_scaled count", output.len())?;
    let shape = format!("hidden=[{},{}],copies={copies}", hidden.rows, hidden.cols);
    launch_1d(ctx, "mhc_expand_scaled_f32", &shape, output.len(), hidden.buffer.length() + coefficients.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&hidden.buffer), 0);
        encoder.set_buffer(1, Some(&coefficients.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &width);
        set_bytes(encoder, 4, &copies);
        set_bytes(encoder, 5, &hidden_dtype);
        set_bytes(encoder, 6, &coefficient_dtype);
        set_bytes(encoder, 7, &count);
    })?;
    Ok(output)
}

pub fn mix_tensor(ctx: &MetalContext, hidden: &MetalTensor, matrix: &MetalTensor, copies: usize) -> Result<MetalTensor, String> {
    if hidden.rows == 0 || hidden.rows != matrix.rows || copies == 0 || !hidden.cols.is_multiple_of(copies) || matrix.cols != copies * copies {
        return Err(format!("Metal mHC mix shape 不一致: hidden={}x{} matrix={}x{} copies={copies}", hidden.rows, hidden.cols, matrix.rows, matrix.cols));
    }
    let output = ctx.tensor_kernel_output_f32(hidden.rows, hidden.cols);
    let width = validate_u32("mHC mix width", hidden.cols / copies)?;
    let copies = validate_u32("mHC mix copies", copies)?;
    let hidden_dtype = dtype_code(hidden.dtype);
    let matrix_dtype = dtype_code(matrix.dtype);
    let count = validate_u32("mHC mix count", output.len())?;
    let shape = format!("hidden=[{},{}],copies={copies}", hidden.rows, hidden.cols);
    launch_1d(ctx, "mhc_mix_f32", &shape, output.len(), hidden.buffer.length() + matrix.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&hidden.buffer), 0);
        encoder.set_buffer(1, Some(&matrix.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &width);
        set_bytes(encoder, 4, &copies);
        set_bytes(encoder, 5, &hidden_dtype);
        set_bytes(encoder, 6, &matrix_dtype);
        set_bytes(encoder, 7, &count);
    })?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn split_tensor(ctx: &MetalContext, mixes: &MetalTensor, base: &Buffer, scale: &Buffer, copies: usize, iterations: usize, eps: f32) -> Result<(MetalTensor, MetalTensor, MetalTensor), String> {
    let mix_columns = copies.checked_mul(copies.checked_add(2).ok_or("Metal mHC split copies 溢出")?).ok_or("Metal mHC split columns 溢出")?;
    if mixes.rows == 0 || mixes.cols != mix_columns || copies == 0 || iterations == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(format!("Metal mHC split shape 非法: mixes={}x{} copies={copies} iterations={iterations} eps={eps}", mixes.rows, mixes.cols));
    }
    let pre = ctx.tensor_kernel_output_f32(mixes.rows, copies);
    let post = ctx.tensor_kernel_output_f32(mixes.rows, copies);
    let combination = ctx.tensor_kernel_output_f32(mixes.rows, copies * copies);
    let copies = validate_u32("mHC split copies", copies)?;
    let iterations = validate_u32("mHC split iterations", iterations)?;
    let mixes_dtype = dtype_code(mixes.dtype);
    let rows = validate_u32("mHC split rows", mixes.rows)?;
    let shape = format!("mixes=[{},{}],copies={copies},iterations={iterations}", mixes.rows, mixes.cols);
    launch_1d(ctx, "mhc_split_f32", &shape, mixes.rows, mixes.buffer.length() + base.length() + scale.length(), pre.buffer.length() + post.buffer.length() + combination.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&mixes.buffer), 0);
        encoder.set_buffer(1, Some(base), 0);
        encoder.set_buffer(2, Some(scale), 0);
        encoder.set_buffer(3, Some(&pre.buffer), 0);
        encoder.set_buffer(4, Some(&post.buffer), 0);
        encoder.set_buffer(5, Some(&combination.buffer), 0);
        set_bytes(encoder, 6, &copies);
        set_bytes(encoder, 7, &iterations);
        set_bytes(encoder, 8, &eps);
        set_bytes(encoder, 9, &mixes_dtype);
        set_bytes(encoder, 10, &rows);
    })?;
    Ok((pre, post, combination))
}

#[allow(clippy::too_many_arguments)]
pub fn head_reduce_tensor(ctx: &MetalContext, hidden: &MetalTensor, mixes: &MetalTensor, base: &Buffer, scale: &Buffer, copies: usize, eps: f32) -> Result<MetalTensor, String> {
    if hidden.rows == 0 || hidden.rows != mixes.rows || copies == 0 || !hidden.cols.is_multiple_of(copies) || mixes.cols != copies || !eps.is_finite() || eps <= 0.0 {
        return Err(format!("Metal output mHC shape 非法: hidden={}x{} mixes={}x{} copies={copies} eps={eps}", hidden.rows, hidden.cols, mixes.rows, mixes.cols));
    }
    let width = hidden.cols / copies;
    let output = ctx.tensor_kernel_output_f32(hidden.rows, width);
    let width = validate_u32("output mHC width", width)?;
    let copies = validate_u32("output mHC copies", copies)?;
    let hidden_dtype = dtype_code(hidden.dtype);
    let mixes_dtype = dtype_code(mixes.dtype);
    let count = validate_u32("output mHC count", output.len())?;
    let shape = format!("hidden=[{},{}],copies={copies}", hidden.rows, hidden.cols);
    launch_1d(ctx, "mhc_head_reduce_f32", &shape, output.len(), hidden.buffer.length() + mixes.buffer.length() + base.length() + scale.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&hidden.buffer), 0);
        encoder.set_buffer(1, Some(&mixes.buffer), 0);
        encoder.set_buffer(2, Some(base), 0);
        encoder.set_buffer(3, Some(scale), 0);
        encoder.set_buffer(4, Some(&output.buffer), 0);
        set_bytes(encoder, 5, &width);
        set_bytes(encoder, 6, &copies);
        set_bytes(encoder, 7, &eps);
        set_bytes(encoder, 8, &hidden_dtype);
        set_bytes(encoder, 9, &mixes_dtype);
        set_bytes(encoder, 10, &count);
    })?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::hyper_connection::{HyperConnectionSpec, expand_scaled_f32, head_reduce_f32, mix_f32, reduce_f32, split_f32};

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 2.0e-4, "actual={actual}, expected={expected}");
        }
    }

    #[test]
    fn metal_mhc_matches_cpu_reference_without_host_intermediates() {
        if crate::kernel::metal::metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let hidden_values = [1.0, -2.0, 3.0, 4.0, 0.5, 1.5, -0.5, 2.5];
        let hidden = ctx.tensor_from_f32(&hidden_values, 2, 4).unwrap();
        let expanded = expand_tensor(&ctx, &hidden, 2).unwrap();
        assert_close(&ctx.tensor_to_f32(&expanded), &[1.0, -2.0, 3.0, 4.0, 1.0, -2.0, 3.0, 4.0, 0.5, 1.5, -0.5, 2.5, 0.5, 1.5, -0.5, 2.5]);

        let coefficients_values = [0.25, 0.75, 1.25, -0.25];
        let coefficients = ctx.tensor_from_f32_preserve(&coefficients_values, 2, 2).unwrap();
        let reduced = reduce_tensor(&ctx, &expanded, &coefficients, 2).unwrap();
        let expected = [reduce_f32(&[1.0, -2.0, 3.0, 4.0, 1.0, -2.0, 3.0, 4.0], &[0.25, 0.75], 2).unwrap(), reduce_f32(&[0.5, 1.5, -0.5, 2.5, 0.5, 1.5, -0.5, 2.5], &[1.25, -0.25], 2).unwrap()].concat();
        assert_close(&ctx.tensor_to_f32(&reduced), &expected);

        let scaled = expand_scaled_tensor(&ctx, &hidden, &coefficients, 2).unwrap();
        let expected = [expand_scaled_f32(&hidden_values[..4], &coefficients_values[..2], 2).unwrap(), expand_scaled_f32(&hidden_values[4..], &coefficients_values[2..], 2).unwrap()].concat();
        assert_close(&ctx.tensor_to_f32(&scaled), &expected);

        let matrix_values = [0.8, 0.2, 0.3, 0.7, 0.6, 0.4, 0.1, 0.9];
        let matrix = ctx.tensor_from_f32_preserve(&matrix_values, 2, 4).unwrap();
        let mixed = mix_tensor(&ctx, &expanded, &matrix, 2).unwrap();
        let expected = [mix_f32(&ctx.tensor_to_f32(&expanded)[..8], &matrix_values[..4], 2).unwrap(), mix_f32(&ctx.tensor_to_f32(&expanded)[8..], &matrix_values[4..], 2).unwrap()].concat();
        assert_close(&ctx.tensor_to_f32(&mixed), &expected);
    }

    #[test]
    fn metal_mhc_split_and_head_match_cpu_reference() {
        if crate::kernel::metal::metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let spec = HyperConnectionSpec { copies: 2, sinkhorn_iterations: 20, eps: 1.0e-6 };
        let mixes_values = [0.2, -0.3, 0.5, -0.7, 0.1, 0.4, -0.2, 0.6, -0.4, 0.1, 0.7, 0.2, -0.5, 0.3, 0.8, -0.1];
        let base_values = [0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.4, -0.3];
        let scale_values = [0.75, 1.25, 0.5];
        let mixes = ctx.tensor_from_f32(&mixes_values, 2, 8).unwrap();
        let base = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(base_values.as_ptr().cast::<u8>(), std::mem::size_of_val(&base_values)) });
        let scale = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(scale_values.as_ptr().cast::<u8>(), std::mem::size_of_val(&scale_values)) });
        let (pre, post, combination) = split_tensor(&ctx, &mixes, &base, &scale, 2, spec.sinkhorn_iterations, spec.eps).unwrap();
        let expected = split_f32(&mixes_values, &base_values, &scale_values, &spec).unwrap();
        assert_close(&ctx.tensor_to_f32(&pre), &expected.pre);
        assert_close(&ctx.tensor_to_f32(&post), &expected.post);
        assert_close(&ctx.tensor_to_f32(&combination), &expected.combination);

        let hidden_values = [1.0, 2.0, 5.0, 8.0, -1.0, 3.0, 2.0, -4.0];
        let head_mixes_values = [0.2, -0.4, 0.5, 0.1];
        let head_base_values = [0.3, -0.2];
        let head_scale_values = [0.75];
        let hidden = ctx.tensor_from_f32(&hidden_values, 2, 4).unwrap();
        let head_mixes = ctx.tensor_from_f32(&head_mixes_values, 2, 2).unwrap();
        let head_base = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(head_base_values.as_ptr().cast::<u8>(), std::mem::size_of_val(&head_base_values)) });
        let head_scale = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(head_scale_values.as_ptr().cast::<u8>(), std::mem::size_of_val(&head_scale_values)) });
        let actual = head_reduce_tensor(&ctx, &hidden, &head_mixes, &head_base, &head_scale, 2, 1.0e-6).unwrap();
        let expected = head_reduce_f32(&hidden_values, &head_mixes_values, &head_base_values, head_scale_values[0], 2, 1.0e-6).unwrap();
        assert_close(&ctx.tensor_to_f32(&actual), &expected);
    }
}
