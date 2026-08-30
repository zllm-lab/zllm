/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: mxfp8_matmul_f16, mxfp8_gated_gemv_f16
// private helpers: decode_e8m0
pub const SHADERS: &str = r#"
inline float decode_e8m0(uchar exponent) {
    const uint bits = exponent == 0 ? 0x00400000u : uint(exponent) << 23;
    return as_type<float>(bits);
}
kernel void mxfp8_matmul_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const uchar *scale_inv [[buffer(2)]],
    device half *output [[buffer(3)]],
    device const half *residual [[buffer(4)]],
    constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]],
    constant uint &output_columns [[buffer(7)]],
    constant uint &scale_columns [[buffer(8)]],
    constant uint &add_residual [[buffer(9)]],
    uint2 lane [[thread_position_in_threadgroup]],
    uint flat_lane [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    if (input_rows == 1) {
        const uint output_row = group.x * 128 + flat_lane;
        if (output_row >= output_columns) return;
        const ulong weight_base = ulong(output_row) * input_columns;
        const ulong scale_base = ulong(output_row) * scale_columns;
        float sum = 0.0f;
        for (uint block = 0; block < input_columns; block += 32) {
            const uchar exponent = scale_inv[scale_base + block / 32];
            const float scale = decode_e8m0(exponent);
            const uint end = min(block + 32, input_columns);
            for (uint column = block; column < end; ++column) {
                const half weight = half(decode_f8_e4m3(weights[weight_base + column]) * scale);
                sum += float(input[column]) * float(weight);
            }
        }
        const float projected = float(half(sum));
        output[output_row] = half(projected + (add_residual != 0 ? float(residual[output_row]) : 0.0f));
        return;
    }

    threadgroup half input_tile[8 * 16];
    threadgroup half weight_tile[16 * 16];
    const uint input_row = group.y * 8 + lane.y;
    const uint output_column = group.x * 16 + lane.x;
    float sum = 0.0f;

    for (uint input_base = 0; input_base < input_columns; input_base += 16) {
        const uint tile_input_row = flat_lane / 16;
        const uint tile_input_column = flat_lane % 16;
        const uint source_row = group.y * 8 + tile_input_row;
        const uint source_column = input_base + tile_input_column;
        input_tile[flat_lane] = source_row < input_rows && source_column < input_columns
            ? input[(ulong)source_row * input_columns + source_column]
            : half(0.0h);

        for (uint index = flat_lane; index < 256; index += 128) {
            const uint local_output = index / 16;
            const uint local_input = index % 16;
            const uint weight_row = group.x * 16 + local_output;
            const uint weight_column = input_base + local_input;
            if (weight_row < output_columns && weight_column < input_columns) {
                const uchar code = weights[(ulong)weight_row * input_columns + weight_column];
                const uchar exponent = scale_inv[(ulong)weight_row * scale_columns + weight_column / 32];
                const float scale = decode_e8m0(exponent);
                weight_tile[index] = half(decode_f8_e4m3(code) * scale);
            } else {
                weight_tile[index] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (input_row < input_rows && output_column < output_columns) {
            const uint input_tile_base = lane.y * 16;
            const uint weight_tile_base = lane.x * 16;
            for (uint index = 0; index < 16; ++index) {
                sum += float(input_tile[input_tile_base + index]) * float(weight_tile[weight_tile_base + index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (input_row < input_rows && output_column < output_columns) {
        const ulong index = (ulong)input_row * output_columns + output_column;
        const float projected = float(half(sum));
        output[index] = half(projected + (add_residual != 0 ? float(residual[index]) : 0.0f));
    }
}
kernel void mxfp8_gated_gemv_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *gate_scales [[buffer(2)]],
    device const uchar *up_weights [[buffer(3)]],
    device const uchar *up_scales [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]],
    constant uint &output_columns [[buffer(7)]],
    constant uint &scale_columns [[buffer(8)]],
    constant uint &activation_kind [[buffer(9)]],
    constant float &alpha [[buffer(10)]],
    constant float &limit [[buffer(11)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    const uint output_row = group * 128 + lane;
    if (output_row >= output_columns) return;
    const ulong weight_base = ulong(output_row) * input_columns;
    const ulong scale_base = ulong(output_row) * scale_columns;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint block = 0; block < input_columns; block += 32) {
        const float gate_scale = decode_e8m0(gate_scales[scale_base + block / 32]);
        const float up_scale = decode_e8m0(up_scales[scale_base + block / 32]);
        const uint end = min(block + 32, input_columns);
        for (uint column = block; column < end; ++column) {
            const float value = float(input[column]);
            const half gate_weight = half(decode_f8_e4m3(gate_weights[weight_base + column]) * gate_scale);
            const half up_weight = half(decode_f8_e4m3(up_weights[weight_base + column]) * up_scale);
            gate_sum += value * float(gate_weight);
            up_sum += value * float(up_weight);
        }
    }
    const float gate_value = float(half(gate_sum));
    const float up_value = float(half(up_sum));
    output[output_row] = finite_f16(gated_activation_value(gate_value, up_value, activation_kind, alpha, limit));
}
"#;

use crate::backend::metal::api as metal;
use crate::moe::Activation;

use super::dense::{GatedActivation, encode_fp8_matmul, launch_official_fp8_dual_matmul, launch_official_fp8_matmul, launch_per_tensor_fp8_matmul, validate_tensor};
use super::{FP8_PREFILL_MPS_ROWS, Fp8Matrix, MTLSize, MetalContext, MetalTensor, THREADS, set_bytes, validate_u32};

pub(super) fn validate_buffer_region(name: &str, offset: usize, len: usize, buffer_len: usize) -> Result<(), String> {
    let end = offset.checked_add(len).ok_or_else(|| format!("{name} offset 溢出"))?;
    if end <= buffer_len {
        return Ok(());
    }
    Err(format!("{name} 区域越界: offset={offset}, len={len}, buffer={buffer_len}"))
}

fn fp8_prefill_mps_tensor(ctx: &MetalContext, input: &MetalTensor, weight: &Fp8Matrix, codes: &metal::Buffer, scale_inv: &metal::Buffer) -> Result<MetalTensor, String> {
    let total = weight.rows.checked_mul(weight.cols).ok_or("FP8 权重大小溢出")?;
    let rows = validate_u32("FP8 weight rows", weight.rows)?;
    let columns = validate_u32("FP8 weight columns", weight.cols)?;
    let scale_columns = validate_u32("FP8 scale columns", weight.cols.div_ceil(128))?;
    let weight_f16 = ctx.tensor_uninit(weight.rows, weight.cols);
    let output = ctx.tensor_uninit(input.rows, weight.rows);
    let pipeline = ctx.pipeline("dequantize_fp8_matrix_f16")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(codes), 0);
    encoder.set_buffer(1, Some(scale_inv), 0);
    encoder.set_buffer(2, Some(&weight_f16.buffer), 0);
    set_bytes(&encoder, 3, &rows);
    set_bytes(&encoder, 4, &columns);
    set_bytes(&encoder, 5, &scale_columns);
    let threads = total.min(THREADS);
    encoder.dispatch_thread_groups(MTLSize::new(total.div_ceil(threads) as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{},{}],weight=[{},{}]", input.rows, input.cols, weight.rows, weight.cols);
    ctx.commit_and_wait_profiled(&command, "fp8_dequantize_f16", &shape, codes.length() + scale_inv.length(), weight_f16.buffer.length());
    super::dense::launch_matmul_f16(ctx, &input.buffer, &weight_f16.buffer, &output.buffer, input.rows, input.cols, weight.rows)?;
    Ok(output)
}

/// FP8 权重直接读取 device tensor，不再上传 activation。
pub fn fp8_matmul_tensor(ctx: &MetalContext, input: &MetalTensor, weight: &Fp8Matrix) -> Result<MetalTensor, String> {
    validate_tensor("FP8 input", input, input.rows, weight.cols)?;
    let codes = ctx.resident_byte_weight_buffer(&weight.codes);
    let scale_inv = ctx.resident_byte_weight_buffer(&weight.scale_inv);
    let rows = validate_u32("rows", input.rows)?;
    let in_cols = validate_u32("FP8 in_cols", weight.cols)?;
    let out_cols = validate_u32("FP8 out_cols", weight.rows)?;
    let scale_cols = validate_u32("FP8 scale_cols", weight.cols.div_ceil(128))?;
    if rows == 1 {
        let output = ctx.tensor_uninit(input.rows, weight.rows);
        launch_official_fp8_matmul(ctx, &input.buffer, &codes, &scale_inv, &output.buffer, rows, in_cols, out_cols, scale_cols)?;
        return Ok(output);
    }
    if input.rows >= FP8_PREFILL_MPS_ROWS {
        return fp8_prefill_mps_tensor(ctx, input, weight, &codes, &scale_inv);
    }

    let output = ctx.tensor_uninit(input.rows, weight.rows);
    let pipeline = ctx.pipeline("matmul_tiled_fp8_f16")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encode_fp8_matmul(&encoder, &input.buffer, &codes, &scale_inv, &output.buffer, rows, in_cols, out_cols, scale_cols);
    encoder.end_encoding();
    let shape = format!("input=[{rows},{in_cols}],weight=[{out_cols},{in_cols}]");
    ctx.commit_and_wait_profiled(&command, "matmul_tiled_fp8_f16", &shape, input.buffer.length() + codes.length() + scale_inv.length(), output.buffer.length());
    Ok(output)
}

/// Decode resident FP8 权重直乘；codes/scales 已在 token 循环前上传。
pub fn fp8_matmul_tensor_resident(ctx: &MetalContext, input: &MetalTensor, codes: &metal::Buffer, scale_inv: &metal::Buffer, weight_rows: usize, weight_cols: usize) -> Result<MetalTensor, String> {
    validate_tensor("resident FP8 input", input, input.rows, weight_cols)?;
    let rows = validate_u32("rows", input.rows)?;
    let in_cols = validate_u32("resident FP8 in_cols", weight_cols)?;
    let out_cols = validate_u32("resident FP8 out_cols", weight_rows)?;
    let scale_cols = validate_u32("resident FP8 scale_cols", weight_cols.div_ceil(128))?;
    let output = ctx.tensor_uninit(input.rows, weight_rows);
    launch_official_fp8_matmul(ctx, &input.buffer, codes, scale_inv, &output.buffer, rows, in_cols, out_cols, scale_cols)?;
    Ok(output)
}

/// Decode resident per-tensor FP8 权重直乘:scale 是单个 F32,
/// 不是 block-grid。codes/scales 已在 token 循环前上传。
pub fn fp8_matmul_tensor_per_tensor_resident(ctx: &MetalContext, input: &MetalTensor, codes: &metal::Buffer, scale: &metal::Buffer, weight_rows: usize, weight_cols: usize) -> Result<MetalTensor, String> {
    validate_tensor("resident per-tensor FP8 input", input, input.rows, weight_cols)?;
    let rows = validate_u32("rows", input.rows)?;
    let in_cols = validate_u32("resident per-tensor FP8 in_cols", weight_cols)?;
    let out_cols = validate_u32("resident per-tensor FP8 out_cols", weight_rows)?;
    let output = ctx.tensor_uninit(input.rows, weight_rows);
    launch_per_tensor_fp8_matmul(ctx, &input.buffer, codes, scale, &output.buffer, rows, in_cols, out_cols)?;
    Ok(output)
}

/// MXFP8 resident 权重直乘；E4M3 code 与 E8M0 scale 均保持压缩形态，
/// kernel 按连续 32 个输入元素共享一个 scale。
#[allow(clippy::too_many_arguments)]
fn encode_mxfp8_matmul(
    encoder: &metal::ComputeCommandEncoderRef,
    input: &metal::Buffer,
    codes: &metal::Buffer,
    scale_inv: &metal::Buffer,
    output: &metal::Buffer,
    residual: Option<&metal::Buffer>,
    rows: u32,
    in_cols: u32,
    out_cols: u32,
    scale_cols: u32,
) {
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(codes), 0);
    encoder.set_buffer(2, Some(scale_inv), 0);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_buffer(4, Some(residual.unwrap_or(input)), 0);
    set_bytes(encoder, 5, &rows);
    set_bytes(encoder, 6, &in_cols);
    set_bytes(encoder, 7, &out_cols);
    set_bytes(encoder, 8, &scale_cols);
    let add_residual = u32::from(residual.is_some());
    set_bytes(encoder, 9, &add_residual);
    let (thread_groups, threads_per_group) =
        if rows == 1 { (MTLSize::new(out_cols.div_ceil(128) as u64, 1, 1), MTLSize::new(128, 1, 1)) } else { (MTLSize::new(out_cols.div_ceil(16) as u64, (rows as usize).div_ceil(8) as u64, 1), MTLSize::new(16, 8, 1)) };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);
}

pub fn mxfp8_matmul_tensor_resident(ctx: &MetalContext, input: &MetalTensor, codes: &metal::Buffer, scale_inv: &metal::Buffer, weight_rows: usize, weight_cols: usize) -> Result<MetalTensor, String> {
    validate_tensor("resident MXFP8 input", input, input.rows, weight_cols)?;
    let expected_codes = weight_rows.checked_mul(weight_cols).ok_or("MXFP8 codes 大小溢出")?;
    let scale_cols_usize = weight_cols.div_ceil(32);
    let expected_scales = weight_rows.checked_mul(scale_cols_usize).ok_or("MXFP8 scales 大小溢出")?;
    if codes.length() != expected_codes as u64 || scale_inv.length() != expected_scales as u64 {
        return Err(format!("resident MXFP8 buffer 大小不符: codes={}/{expected_codes}, scales={}/{expected_scales}", codes.length(), scale_inv.length()));
    }

    let rows = validate_u32("MXFP8 rows", input.rows)?;
    let in_cols = validate_u32("MXFP8 in_cols", weight_cols)?;
    let out_cols = validate_u32("MXFP8 out_cols", weight_rows)?;
    let scale_cols = validate_u32("MXFP8 scale_cols", scale_cols_usize)?;
    let output = ctx.tensor_uninit(input.rows, weight_rows);
    let pipeline = ctx.pipeline("mxfp8_matmul_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("MXFP8 matmul 需要至少 128 threads/threadgroup".to_owned());
    }

    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encode_mxfp8_matmul(&encoder, &input.buffer, codes, scale_inv, &output.buffer, None, rows, in_cols, out_cols, scale_cols);
    encoder.end_encoding();

    let shape = format!("input=[{rows},{in_cols}],weight=[{out_cols},{in_cols}]");
    ctx.commit_and_wait_profiled(&command, "mxfp8_matmul_f16", &shape, input.buffer.length() + codes.length() + scale_inv.length(), output.buffer.length());
    Ok(output)
}

pub fn mxfp8_matmul_add_tensor_resident(ctx: &MetalContext, input: &MetalTensor, codes: &metal::Buffer, scale_inv: &metal::Buffer, weight_rows: usize, weight_cols: usize, residual: &MetalTensor) -> Result<MetalTensor, String> {
    validate_tensor("resident MXFP8 add input", input, input.rows, weight_cols)?;
    validate_tensor("resident MXFP8 residual", residual, input.rows, weight_rows)?;
    let expected_codes = weight_rows.checked_mul(weight_cols).ok_or("MXFP8 add codes 大小溢出")?;
    let scale_cols_usize = weight_cols.div_ceil(32);
    let expected_scales = weight_rows.checked_mul(scale_cols_usize).ok_or("MXFP8 add scales 大小溢出")?;
    if codes.length() != expected_codes as u64 || scale_inv.length() != expected_scales as u64 {
        return Err("resident MXFP8 add buffer 大小不符".to_owned());
    }
    let rows = validate_u32("MXFP8 add rows", input.rows)?;
    let in_cols = validate_u32("MXFP8 add in_cols", weight_cols)?;
    let out_cols = validate_u32("MXFP8 add out_cols", weight_rows)?;
    let scale_cols = validate_u32("MXFP8 add scale cols", scale_cols_usize)?;
    let output = ctx.tensor_uninit(input.rows, weight_rows);
    let pipeline = ctx.pipeline("mxfp8_matmul_f16")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encode_mxfp8_matmul(&encoder, &input.buffer, codes, scale_inv, &output.buffer, Some(&residual.buffer), rows, in_cols, out_cols, scale_cols);
    encoder.end_encoding();
    let shape = format!("input=[{rows},{in_cols}],weight=[{out_cols},{in_cols}]");
    ctx.commit_and_wait_profiled(&command, "mxfp8_matmul_add_f16", &shape, input.buffer.length() + codes.length() + scale_inv.length() + residual.buffer.length(), output.buffer.length());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn mxfp8_dual_matmul_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    first_codes: &metal::Buffer,
    first_scales: &metal::Buffer,
    first_rows: usize,
    first_cols: usize,
    second_codes: &metal::Buffer,
    second_scales: &metal::Buffer,
    second_rows: usize,
    second_cols: usize,
) -> Result<(MetalTensor, MetalTensor), String> {
    if first_cols != second_cols || input.cols != first_cols {
        return Err(format!("resident MXFP8 dual 输入宽度不一致: input={}, first={first_cols}, second={second_cols}", input.cols));
    }
    let first_scale_cols_usize = first_cols.div_ceil(32);
    let second_scale_cols_usize = second_cols.div_ceil(32);
    let first_codes_len = first_rows.checked_mul(first_cols).ok_or("first MXFP8 codes 大小溢出")?;
    let first_scales_len = first_rows.checked_mul(first_scale_cols_usize).ok_or("first MXFP8 scales 大小溢出")?;
    let second_codes_len = second_rows.checked_mul(second_cols).ok_or("second MXFP8 codes 大小溢出")?;
    let second_scales_len = second_rows.checked_mul(second_scale_cols_usize).ok_or("second MXFP8 scales 大小溢出")?;
    if first_codes.length() != first_codes_len as u64 || first_scales.length() != first_scales_len as u64 || second_codes.length() != second_codes_len as u64 || second_scales.length() != second_scales_len as u64 {
        return Err("resident MXFP8 dual buffer 大小不符".to_owned());
    }

    let rows = validate_u32("MXFP8 dual rows", input.rows)?;
    let in_cols = validate_u32("MXFP8 dual in_cols", input.cols)?;
    let first_out_cols = validate_u32("first MXFP8 dual out_cols", first_rows)?;
    let second_out_cols = validate_u32("second MXFP8 dual out_cols", second_rows)?;
    let first_scale_cols = validate_u32("first MXFP8 dual scale_cols", first_scale_cols_usize)?;
    let second_scale_cols = validate_u32("second MXFP8 dual scale_cols", second_scale_cols_usize)?;
    let first_output = ctx.tensor_uninit(input.rows, first_rows);
    let second_output = ctx.tensor_uninit(input.rows, second_rows);
    let pipeline = ctx.pipeline("mxfp8_matmul_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("MXFP8 dual matmul 需要至少 128 threads/threadgroup".to_owned());
    }

    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encode_mxfp8_matmul(&encoder, &input.buffer, first_codes, first_scales, &first_output.buffer, None, rows, in_cols, first_out_cols, first_scale_cols);
    encode_mxfp8_matmul(&encoder, &input.buffer, second_codes, second_scales, &second_output.buffer, None, rows, in_cols, second_out_cols, second_scale_cols);
    encoder.end_encoding();
    let shape = format!("input=[{rows},{in_cols}],outputs=[{first_out_cols},{second_out_cols}]");
    ctx.commit_and_wait_profiled(
        &command,
        "mxfp8_matmul_f16.dual",
        &shape,
        input.buffer.length() + first_codes.length() + first_scales.length() + second_codes.length() + second_scales.length(),
        first_output.buffer.length() + second_output.buffer.length(),
    );
    Ok((first_output, second_output))
}

#[allow(clippy::too_many_arguments)]
pub fn mxfp8_gated_gemv_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    gate_codes: &metal::Buffer,
    gate_scales: &metal::Buffer,
    gate_rows: usize,
    gate_cols: usize,
    up_codes: &metal::Buffer,
    up_scales: &metal::Buffer,
    up_rows: usize,
    up_cols: usize,
    activation: &Activation,
) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != gate_cols || gate_cols != up_cols || gate_rows != up_rows {
        return Err(format!("resident MXFP8 gated shape 不一致: input=[{},{}], gate=[{gate_rows},{gate_cols}], up=[{up_rows},{up_cols}]", input.rows, input.cols,));
    }
    let scale_cols_usize = gate_cols.div_ceil(32);
    let codes_len = gate_rows.checked_mul(gate_cols).ok_or("MXFP8 gated codes 大小溢出")?;
    let scales_len = gate_rows.checked_mul(scale_cols_usize).ok_or("MXFP8 gated scales 大小溢出")?;
    if gate_codes.length() != codes_len as u64 || up_codes.length() != codes_len as u64 || gate_scales.length() != scales_len as u64 || up_scales.length() != scales_len as u64 {
        return Err("resident MXFP8 gated buffer 大小不符".to_owned());
    }
    let input_columns = validate_u32("MXFP8 gated input columns", gate_cols)?;
    let output_columns = validate_u32("MXFP8 gated output columns", gate_rows)?;
    let scale_columns = validate_u32("MXFP8 gated scale columns", scale_cols_usize)?;
    let params = GatedActivation::from_spec(activation)?;
    let output = ctx.tensor_uninit(1, gate_rows);
    let pipeline = ctx.pipeline("mxfp8_gated_gemv_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("MXFP8 gated GEMV 需要至少 128 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(gate_codes), 0);
    encoder.set_buffer(2, Some(gate_scales), 0);
    encoder.set_buffer(3, Some(up_codes), 0);
    encoder.set_buffer(4, Some(up_scales), 0);
    encoder.set_buffer(5, Some(&output.buffer), 0);
    set_bytes(&encoder, 6, &input_columns);
    set_bytes(&encoder, 7, &output_columns);
    set_bytes(&encoder, 8, &scale_columns);
    set_bytes(&encoder, 9, &params.kind);
    set_bytes(&encoder, 10, &params.alpha);
    set_bytes(&encoder, 11, &params.limit);
    encoder.dispatch_thread_groups(MTLSize::new(output_columns.div_ceil(128) as u64, 1, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{input_columns}],output={output_columns}");
    ctx.commit_and_wait_profiled(&command, "mxfp8_gated_gemv_f16", &shape, input.buffer.length() + gate_codes.length() + gate_scales.length() + up_codes.length() + up_scales.length(), output.buffer.length());
    Ok(output)
}

/// Decode resident FP8 双投影，共用一次 activation 读取。
#[allow(clippy::too_many_arguments)]
pub fn fp8_dual_matmul_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    first_codes: &metal::Buffer,
    first_scales: &metal::Buffer,
    first_rows: usize,
    first_cols: usize,
    second_codes: &metal::Buffer,
    second_scales: &metal::Buffer,
    second_rows: usize,
    second_cols: usize,
) -> Result<(MetalTensor, MetalTensor), String> {
    if first_cols != second_cols || input.cols != first_cols {
        return Err(format!("resident FP8 dual 输入宽度不一致: input={}, first={first_cols}, second={second_cols}", input.cols));
    }
    let rows = validate_u32("rows", input.rows)?;
    let in_cols = validate_u32("resident dual FP8 in_cols", input.cols)?;
    let first_out = validate_u32("resident first FP8 out_cols", first_rows)?;
    let second_out = validate_u32("resident second FP8 out_cols", second_rows)?;
    let first_scale_cols = validate_u32("resident first FP8 scale_cols", first_cols.div_ceil(128))?;
    let second_scale_cols = validate_u32("resident second FP8 scale_cols", second_cols.div_ceil(128))?;
    let first_output = ctx.tensor_uninit(input.rows, first_rows);
    let second_output = ctx.tensor_uninit(input.rows, second_rows);
    launch_official_fp8_dual_matmul(ctx, &input.buffer, first_codes, first_scales, &first_output.buffer, first_out, first_scale_cols, second_codes, second_scales, &second_output.buffer, second_out, second_scale_cols, rows, in_cols)?;
    Ok((first_output, second_output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp8_gated_gemv_matches_dual_activation() {
        let ctx = MetalContext::new_default().unwrap();
        let columns = 64;
        let rows = 7;
        let input_values: Vec<f32> = (0..columns).map(|index| (index as f32 - 31.0) / 64.0).collect();
        let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let gate_codes: Vec<u8> = (0..rows * columns).map(|index| [0x38, 0xb8, 0x30, 0x34][index % 4]).collect();
        let up_codes: Vec<u8> = (0..rows * columns).map(|index| [0x34, 0x30, 0xb8, 0x38][index % 4]).collect();
        let scales = vec![127u8; rows * columns.div_ceil(32)];
        let gate_codes = ctx.shared_buffer(&gate_codes);
        let up_codes = ctx.shared_buffer(&up_codes);
        let gate_scales = ctx.shared_buffer(&scales);
        let up_scales = ctx.shared_buffer(&scales);

        let (gate, up) = mxfp8_dual_matmul_tensor_resident(&ctx, &input, &gate_codes, &gate_scales, rows, columns, &up_codes, &up_scales, rows, columns).unwrap();
        let expected = crate::kernel::metal::tensor::gated_activation_tensor(&ctx, &gate, &up, &Activation::Silu).unwrap();
        let actual = mxfp8_gated_gemv_tensor_resident(&ctx, &input, &gate_codes, &gate_scales, rows, columns, &up_codes, &up_scales, rows, columns, &Activation::Silu).unwrap();
        assert_eq!(ctx.tensor_to_f32(&actual), ctx.tensor_to_f32(&expected));
    }

    #[test]
    fn mxfp8_matmul_add_matches_separate_residual_add() {
        let ctx = MetalContext::new_default().unwrap();
        let input_columns = 64;
        let output_columns = 31;
        let input_values: Vec<f32> = (0..input_columns).map(|index| (index as f32 * 0.17).sin() * 0.5).collect();
        let residual_values: Vec<f32> = (0..output_columns).map(|index| (index as f32 * 0.11).cos() * 0.25).collect();
        let input = ctx.tensor_from_f32(&input_values, 1, input_columns).unwrap();
        let residual = ctx.tensor_from_f32(&residual_values, 1, output_columns).unwrap();
        let codes: Vec<u8> = (0..output_columns * input_columns).map(|index| [0x38, 0xb8, 0x30, 0x34][index % 4]).collect();
        let scales = vec![127u8; output_columns * input_columns.div_ceil(32)];
        let codes = ctx.shared_buffer(&codes);
        let scales = ctx.shared_buffer(&scales);

        let projected = mxfp8_matmul_tensor_resident(&ctx, &input, &codes, &scales, output_columns, input_columns).unwrap();
        let expected = crate::kernel::metal::tensor::add_tensor(&ctx, &projected, &residual).unwrap();
        let actual = mxfp8_matmul_add_tensor_resident(&ctx, &input, &codes, &scales, output_columns, input_columns, &residual).unwrap();

        assert_eq!(ctx.tensor_to_f32(&actual), ctx.tensor_to_f32(&expected));
    }
}
