//! MXFP4 / MXFP8 微缩放格式:packed 权重直接在 GPU 解码累加,激活可在线量化 MXFP8。

/// 本家族的 Metal shader。家族私有 helper 先于 kernel 定义;
/// 跨家族共享 helper 见 `super::super::preamble`,由 `kernels_source()` 统一拼接。
pub const MXFP_SHADERS: &str = r#"
inline float mxfp4_value(uchar code)
{
    const uchar magnitude = code & 7;
    float value = 0.0f;
    switch (magnitude) {
        case 1: value = 0.5f; break;
        case 2: value = 1.0f; break;
        case 3: value = 1.5f; break;
        case 4: value = 2.0f; break;
        case 5: value = 3.0f; break;
        case 6: value = 4.0f; break;
        case 7: value = 6.0f; break;
        default: break;
    }
    return (code & 8) == 0 ? value : -value;
}
inline uchar encode_f8_e4m3_rte(float value)
{
    const uchar sign = signbit(value) ? 128 : 0;
    float magnitude = abs(value);
    if (isnan(magnitude)) return sign | 127;
    if (magnitude >= 448.0f) return sign | 126;
    if (magnitude < 0.015625f) {
        const uint mantissa = uint(rint(magnitude * 512.0f));
        if (mantissa >= 8) return sign | 8;
        return sign | uchar(mantissa);
    }
    int exponent = int(floor(log2(magnitude)));
    uint mantissa = uint(rint((magnitude * exp2(float(-exponent)) - 1.0f) * 8.0f));
    if (mantissa == 8) {
        exponent += 1;
        mantissa = 0;
    }
    if (exponent > 8 || (exponent == 8 && mantissa > 6)) return sign | 126;
    return sign | uchar(uint(exponent + 7) * 8 + mantissa);
}
kernel void quantize_mxfp8_activation_f16(
    device const half *input [[buffer(0)]],
    device uchar *codes [[buffer(1)]],
    device uchar *scales [[buffer(2)]],
    constant uint &input_rows [[buffer(3)]],
    constant uint &input_columns [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint quant_group = group.x;
    const uint input_row = group.y;
    const uint groups = input_columns / 32;
    if (input_row >= input_rows || quant_group >= groups) return;
    const ulong index = ulong(input_row) * input_columns + ulong(quant_group) * 32 + lane;
    threadgroup float magnitudes[32];
    threadgroup int shared_exponent;
    magnitudes[lane] = abs(float(input[index]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) magnitudes[lane] = max(magnitudes[lane], magnitudes[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const float maximum = magnitudes[0];
        shared_exponent = maximum == 0.0f ? -127 : clamp(int(floor(log2(maximum))) - 8, -127, 127);
        scales[ulong(input_row) * groups + quant_group] = uchar(shared_exponent + 127);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    codes[index] = encode_f8_e4m3_rte(float(input[index]) * exp2(float(-shared_exponent)));
}
kernel void mxfp4_mxfp8_matmul_f16(
    device const uchar *input_codes [[buffer(0)]],
    device const uchar *input_scales [[buffer(1)]],
    device const uchar *packed [[buffer(2)]],
    device const uchar *weight_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]],
    constant uint &output_columns [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint output_column = group.x;
    const uint input_row = group.y;
    if (output_column >= output_columns || input_row >= input_rows) return;
    threadgroup float sums[64];
    const ulong packed_columns = ulong(input_columns) / 2;
    const ulong groups = ulong(input_columns) / 32;
    const ulong input_base = ulong(input_row) * input_columns;
    float sum = 0.0f;
    for (uint quant_group = lane; quant_group < groups; quant_group += 64) {
        const ulong packed_base = ulong(output_column) * packed_columns + ulong(quant_group) * 16;
        const ulong input_group = input_base + ulong(quant_group) * 32;
        float block_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < 32; ++index) {
            const uchar pair = packed[packed_base + index / 2];
            const uchar weight_code = (index & 1) == 0 ? pair & 15 : pair >> 4;
            block_sum += decode_f8_e4m3(input_codes[input_group + index]) * mxfp4_value(weight_code);
        }
        const int input_exponent = int(input_scales[ulong(input_row) * groups + quant_group]) - 127;
        const int weight_exponent = int(weight_scales[ulong(output_column) * groups + quant_group]) - 127;
        sum += block_sum * exp2(float(input_exponent + weight_exponent));
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[ulong(input_row) * output_columns + output_column] = finite_f16(sums[0]);
}
kernel void mxfp4_mxfp8_gated_matmul_f16(
    device const uchar *input_codes [[buffer(0)]],
    device const uchar *input_scales [[buffer(1)]],
    device const uchar *gate_packed [[buffer(2)]],
    device const uchar *gate_scales [[buffer(3)]],
    device const uchar *up_packed [[buffer(4)]],
    device const uchar *up_scales [[buffer(5)]],
    device half *output [[buffer(6)]],
    constant uint &input_rows [[buffer(7)]],
    constant uint &input_columns [[buffer(8)]],
    constant uint &output_columns [[buffer(9)]],
    constant uint &activation_kind [[buffer(10)]],
    constant float &activation_alpha [[buffer(11)]],
    constant float &activation_limit [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint output_column = group.x;
    const uint input_row = group.y;
    if (output_column >= output_columns || input_row >= input_rows) return;
    threadgroup float gate_sums[64];
    threadgroup float up_sums[64];
    const ulong packed_columns = ulong(input_columns) / 2;
    const ulong groups = ulong(input_columns) / 32;
    const ulong input_base = ulong(input_row) * input_columns;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint quant_group = lane; quant_group < groups; quant_group += 64) {
        const ulong packed_base = ulong(output_column) * packed_columns + ulong(quant_group) * 16;
        const ulong input_group = input_base + ulong(quant_group) * 32;
        float gate_block = 0.0f;
        float up_block = 0.0f;
#pragma unroll
        for (uint index = 0; index < 32; ++index) {
            const uchar gate_pair = gate_packed[packed_base + index / 2];
            const uchar up_pair = up_packed[packed_base + index / 2];
            const uchar gate_code = (index & 1) == 0 ? gate_pair & 15 : gate_pair >> 4;
            const uchar up_code = (index & 1) == 0 ? up_pair & 15 : up_pair >> 4;
            const float value = decode_f8_e4m3(input_codes[input_group + index]);
            gate_block += value * mxfp4_value(gate_code);
            up_block += value * mxfp4_value(up_code);
        }
        const int input_exponent = int(input_scales[ulong(input_row) * groups + quant_group]) - 127;
        const int gate_exponent = int(gate_scales[ulong(output_column) * groups + quant_group]) - 127;
        const int up_exponent = int(up_scales[ulong(output_column) * groups + quant_group]) - 127;
        gate_sum += gate_block * exp2(float(input_exponent + gate_exponent));
        up_sum += up_block * exp2(float(input_exponent + up_exponent));
    }
    gate_sums[lane] = gate_sum;
    up_sums[lane] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            gate_sums[lane] += gate_sums[lane + stride];
            up_sums[lane] += up_sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const half gate = finite_f16(gate_sums[0]);
        const half up = finite_f16(up_sums[0]);
        output[ulong(input_row) * output_columns + output_column] = finite_f16(gated_activation_value(
            float(gate), float(up), activation_kind, activation_alpha, activation_limit));
    }
}
kernel void mxfp4_matmul_f16(
    device const half *input [[buffer(0)]],
    device const uchar *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &input_rows [[buffer(4)]],
    constant uint &input_columns [[buffer(5)]],
    constant uint &output_columns [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint output_column = group.x;
    const uint input_row = group.y;
    if (output_column >= output_columns || input_row >= input_rows) return;
    threadgroup float sums[64];
    const ulong packed_columns = ulong(input_columns) / 2;
    const ulong groups = ulong(input_columns) / 32;
    const ulong input_base = ulong(input_row) * input_columns;
    float sum = 0.0f;
    for (uint quant_group = lane; quant_group < groups; quant_group += 64) {
        const ulong packed_base = ulong(output_column) * packed_columns + ulong(quant_group) * 16;
        const ulong input_group = input_base + ulong(quant_group) * 32;
        const int exponent = int(scales[ulong(output_column) * groups + quant_group]) - 127;
        float block_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < 32; ++index) {
            const uchar pair = packed[packed_base + index / 2];
            const uchar code = (index & 1) == 0 ? pair & 15 : pair >> 4;
            block_sum += float(input[input_group + index]) * mxfp4_value(code);
        }
        sum += block_sum * exp2(float(exponent));
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[ulong(input_row) * output_columns + output_column] = finite_f16(sums[0]);
}
kernel void mxfp4_gated_matmul_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_packed [[buffer(1)]],
    device const uchar *gate_scales [[buffer(2)]],
    device const uchar *up_packed [[buffer(3)]],
    device const uchar *up_scales [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &input_rows [[buffer(6)]],
    constant uint &input_columns [[buffer(7)]],
    constant uint &output_columns [[buffer(8)]],
    constant uint &activation_kind [[buffer(9)]],
    constant float &activation_alpha [[buffer(10)]],
    constant float &activation_limit [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint output_column = group.x;
    const uint input_row = group.y;
    if (output_column >= output_columns || input_row >= input_rows) return;
    threadgroup float gate_sums[64];
    threadgroup float up_sums[64];
    const ulong packed_columns = ulong(input_columns) / 2;
    const uint groups = input_columns / 32;
    const ulong input_base = ulong(input_row) * input_columns;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint quant_group = lane; quant_group < groups; quant_group += 64) {
        const ulong packed_base = ulong(output_column) * packed_columns + ulong(quant_group) * 16;
        const ulong input_group = input_base + ulong(quant_group) * 32;
        float gate_block = 0.0f;
        float up_block = 0.0f;
#pragma unroll
        for (uint index = 0; index < 32; ++index) {
            const uchar gate_pair = gate_packed[packed_base + index / 2];
            const uchar up_pair = up_packed[packed_base + index / 2];
            const uchar gate_code = (index & 1) == 0 ? gate_pair & 15 : gate_pair >> 4;
            const uchar up_code = (index & 1) == 0 ? up_pair & 15 : up_pair >> 4;
            const float value = float(input[input_group + index]);
            gate_block += value * mxfp4_value(gate_code);
            up_block += value * mxfp4_value(up_code);
        }
        const int gate_exponent = int(gate_scales[ulong(output_column) * groups + quant_group]) - 127;
        const int up_exponent = int(up_scales[ulong(output_column) * groups + quant_group]) - 127;
        gate_sum += gate_block * exp2(float(gate_exponent));
        up_sum += up_block * exp2(float(up_exponent));
    }
    gate_sums[lane] = gate_sum;
    up_sums[lane] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            gate_sums[lane] += gate_sums[lane + stride];
            up_sums[lane] += up_sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const half gate = finite_f16(gate_sums[0]);
        const half up = finite_f16(up_sums[0]);
        output[ulong(input_row) * output_columns + output_column] = finite_f16(gated_activation_value(
            float(gate), float(up), activation_kind, activation_alpha, activation_limit));
    }
}
"#;

use crate::backend::metal::api as metal;

use super::super::dense::GatedActivation;
use super::super::{Activation, MTLSize, MetalContext, MetalTensor, set_bytes, validate_u32};

/// MXFP4 保持 packed；decode/prefill 都直接在 GPU 内解码并累加。
pub fn mxfp4_matmul_tensor(ctx: &MetalContext, input: &MetalTensor, weight: &crate::weight::format::mxfp4::Mxfp4Matrix) -> Result<MetalTensor, String> {
    let packed = ctx.shared_buffer(weight.packed());
    let scales = ctx.shared_buffer(weight.scales());
    mxfp4_matmul_tensor_resident(ctx, input, &packed, &scales, weight.rows(), weight.cols())
}

pub fn mxfp4_matmul_tensor_resident(ctx: &MetalContext, input: &MetalTensor, packed: &metal::Buffer, scales: &metal::Buffer, rows: usize, cols: usize) -> Result<MetalTensor, String> {
    if input.dtype != super::MetalTensorDType::F16 {
        return Err(format!("MXFP4 matmul 当前需要 F16 activation，实际为 {:?}", input.dtype));
    }
    if input.cols != cols {
        return Err(format!("MXFP4 input=[{},{}] weight=[{rows},{cols}] 不兼容", input.rows, input.cols));
    }
    if !cols.is_multiple_of(crate::weight::format::mxfp4::MXFP4_GROUP_SIZE) {
        return Err(format!("MXFP4 weight columns={cols} 不是 group size {} 的倍数", crate::weight::format::mxfp4::MXFP4_GROUP_SIZE));
    }
    let expected_packed = rows.checked_mul(cols / 2).ok_or("MXFP4 packed 大小溢出")?;
    let expected_scales = rows.checked_mul(cols / crate::weight::format::mxfp4::MXFP4_GROUP_SIZE).ok_or("MXFP4 scale 大小溢出")?;
    if packed.length() as usize != expected_packed || scales.length() as usize != expected_scales {
        return Err(format!("MXFP4 storage packed={}/{expected_packed}, scales={}/{expected_scales}", packed.length(), scales.length()));
    }

    let input_rows = validate_u32("MXFP4 input rows", input.rows)?;
    let input_columns = validate_u32("MXFP4 input columns", input.cols)?;
    let output_columns = validate_u32("MXFP4 output columns", rows)?;
    let output = ctx.tensor_zeros(input.rows, rows);
    let pipeline = ctx.pipeline("mxfp4_matmul_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("MXFP4 matmul 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(packed), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &input_rows);
    set_bytes(&encoder, 5, &input_columns);
    set_bytes(&encoder, 6, &output_columns);
    encoder.dispatch_thread_groups(MTLSize::new(output_columns as u64, input_rows as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_wait(&command);
    Ok(output)
}

/// gate/up 共用输入读取，并在写回前完成激活，避免两个 GEMV 和独立激活的同步边界。
pub fn mxfp4_gated_matmul_tensor(ctx: &MetalContext, input: &MetalTensor, gate: &crate::weight::format::mxfp4::Mxfp4Matrix, up: &crate::weight::format::mxfp4::Mxfp4Matrix, activation: &Activation) -> Result<MetalTensor, String> {
    let gate_packed = ctx.shared_buffer(gate.packed());
    let gate_scales = ctx.shared_buffer(gate.scales());
    let up_packed = ctx.shared_buffer(up.packed());
    let up_scales = ctx.shared_buffer(up.scales());
    mxfp4_gated_matmul_tensor_resident(ctx, input, &gate_packed, &gate_scales, &up_packed, &up_scales, gate.rows(), gate.cols(), activation)
}

pub fn mxfp4_gated_matmul_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    gate_packed: &metal::Buffer,
    gate_scales: &metal::Buffer,
    up_packed: &metal::Buffer,
    up_scales: &metal::Buffer,
    rows: usize,
    cols: usize,
    activation: &Activation,
) -> Result<MetalTensor, String> {
    if input.dtype != super::MetalTensorDType::F16 {
        return Err(format!("MXFP4 gated matmul 当前需要 F16 activation，实际为 {:?}", input.dtype));
    }
    if input.cols != cols {
        return Err(format!("MXFP4 gated shape 不兼容: input=[{},{}], weight=[{rows},{cols}]", input.rows, input.cols));
    }
    let expected_packed = rows.checked_mul(cols / 2).ok_or("MXFP4 gated packed 大小溢出")?;
    let expected_scales = rows.checked_mul(cols / crate::weight::format::mxfp4::MXFP4_GROUP_SIZE).ok_or("MXFP4 gated scale 大小溢出")?;
    for (name, packed, scales) in [("gate", gate_packed, gate_scales), ("up", up_packed, up_scales)] {
        if packed.length() as usize != expected_packed || scales.length() as usize != expected_scales {
            return Err(format!("MXFP4 {name} storage packed={}/{expected_packed}, scales={}/{expected_scales}", packed.length(), scales.length()));
        }
    }

    let activation = GatedActivation::from_spec(activation)?;
    let input_rows = validate_u32("MXFP4 gated input rows", input.rows)?;
    let input_columns = validate_u32("MXFP4 gated input columns", input.cols)?;
    let output_columns = validate_u32("MXFP4 gated output columns", rows)?;
    let output = ctx.tensor_zeros(input.rows, rows);
    let pipeline = ctx.pipeline("mxfp4_gated_matmul_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("MXFP4 gated matmul 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(gate_packed), 0);
    encoder.set_buffer(2, Some(gate_scales), 0);
    encoder.set_buffer(3, Some(up_packed), 0);
    encoder.set_buffer(4, Some(up_scales), 0);
    encoder.set_buffer(5, Some(&output.buffer), 0);
    set_bytes(&encoder, 6, &input_rows);
    set_bytes(&encoder, 7, &input_columns);
    set_bytes(&encoder, 8, &output_columns);
    set_bytes(&encoder, 9, &activation.kind);
    set_bytes(&encoder, 10, &activation.alpha);
    set_bytes(&encoder, 11, &activation.limit);
    encoder.dispatch_thread_groups(MTLSize::new(output_columns as u64, input_rows as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{},{}],output={rows}", input.rows, input.cols);
    ctx.commit_and_wait_profiled(&command, "mxfp4_gated_matmul_f16", &shape, input.buffer.length() + gate_packed.length() + gate_scales.length() + up_packed.length() + up_scales.length(), output.buffer.length());
    Ok(output)
}

/// MXFP8 activation 由 E4M3 codes 和逐 32 元素 E8M0 scales 共同组成。
/// 它是线性层边界上的短生命周期执行数据，不混入通用 F16 tensor。
pub struct MetalMxfp8Activation {
    codes: metal::Buffer,
    scales: metal::Buffer,
    rows: usize,
    cols: usize,
}

impl MetalMxfp8Activation {
    fn allocate(ctx: &MetalContext, input: &MetalTensor) -> Result<Self, String> {
        if input.dtype != super::MetalTensorDType::F16 {
            return Err(format!("MXFP8 activation 仅接受 F16 输入，实际 {:?}", input.dtype));
        }
        if input.rows == 0 || input.cols == 0 || !input.cols.is_multiple_of(crate::weight::format::mxfp4::MXFP4_GROUP_SIZE) {
            return Err(format!("MXFP8 activation shape 必须非空且 cols 是 {} 的倍数，实际 [{},{}]", crate::weight::format::mxfp4::MXFP4_GROUP_SIZE, input.rows, input.cols,));
        }
        let code_bytes = input.rows.checked_mul(input.cols).ok_or_else(|| "MXFP8 activation codes 大小溢出".to_owned())?;
        let scale_bytes = input.rows.checked_mul(input.cols / crate::weight::format::mxfp4::MXFP4_GROUP_SIZE).ok_or_else(|| "MXFP8 activation scales 大小溢出".to_owned())?;
        let codes = ctx.tensor_zeros(1, code_bytes.div_ceil(2)).buffer;
        let scales = ctx.tensor_zeros(1, scale_bytes.div_ceil(2)).buffer;
        Ok(Self { codes, scales, rows: input.rows, cols: input.cols })
    }
}

fn encode_mxfp8_activation(ctx: &MetalContext, input: &MetalTensor, activation: &MetalMxfp8Activation, command: &metal::CommandBuffer) -> Result<(), String> {
    let input_rows = validate_u32("MXFP8 activation rows", activation.rows)?;
    let input_columns = validate_u32("MXFP8 activation columns", activation.cols)?;
    let groups = activation.cols / crate::weight::format::mxfp4::MXFP4_GROUP_SIZE;
    let pipeline = ctx.pipeline("quantize_mxfp8_activation_f16")?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&activation.codes), 0);
    encoder.set_buffer(2, Some(&activation.scales), 0);
    set_bytes(&encoder, 3, &input_rows);
    set_bytes(&encoder, 4, &input_columns);
    encoder.dispatch_thread_groups(MTLSize::new(groups as u64, activation.rows as u64, 1), MTLSize::new(crate::weight::format::mxfp4::MXFP4_GROUP_SIZE as u64, 1, 1));
    encoder.end_encoding();
    Ok(())
}

pub fn mxfp4_mxfp8_matmul_tensor(ctx: &MetalContext, input: &MetalTensor, weight: &crate::weight::format::mxfp4::Mxfp4Matrix) -> Result<MetalTensor, String> {
    if input.cols != weight.cols() {
        return Err(format!("MXFP4 x MXFP8 matmul shape 不兼容: input=[{},{}], weight=[{},{}]", input.rows, input.cols, weight.rows(), weight.cols(),));
    }
    let activation = MetalMxfp8Activation::allocate(ctx, input)?;
    let packed = ctx.shared_buffer(weight.packed());
    let weight_scales = ctx.shared_buffer(weight.scales());
    let output = ctx.tensor_zeros(input.rows, weight.rows());
    let input_rows = validate_u32("MXFP4 x MXFP8 input rows", input.rows)?;
    let input_columns = validate_u32("MXFP4 x MXFP8 input columns", input.cols)?;
    let output_columns = validate_u32("MXFP4 x MXFP8 output columns", weight.rows())?;
    let pipeline = ctx.pipeline("mxfp4_mxfp8_matmul_f16")?;
    let command = ctx.command_buffer();
    encode_mxfp8_activation(ctx, input, &activation, &command)?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&activation.codes), 0);
    encoder.set_buffer(1, Some(&activation.scales), 0);
    encoder.set_buffer(2, Some(&packed), 0);
    encoder.set_buffer(3, Some(&weight_scales), 0);
    encoder.set_buffer(4, Some(&output.buffer), 0);
    set_bytes(&encoder, 5, &input_rows);
    set_bytes(&encoder, 6, &input_columns);
    set_bytes(&encoder, 7, &output_columns);
    encoder.dispatch_thread_groups(MTLSize::new(weight.rows() as u64, input.rows as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{},{}],weight=[{},{}]", input.rows, input.cols, weight.rows(), weight.cols());
    ctx.commit_and_wait_profiled(&command, "mxfp4_mxfp8_matmul_f16", &shape, input.buffer.length() + packed.length() + weight_scales.length(), activation.codes.length() + activation.scales.length() + output.buffer.length());
    Ok(output)
}

pub fn mxfp4_mxfp8_gated_matmul_tensor(ctx: &MetalContext, input: &MetalTensor, gate: &crate::weight::format::mxfp4::Mxfp4Matrix, up: &crate::weight::format::mxfp4::Mxfp4Matrix, activation_spec: &Activation) -> Result<MetalTensor, String> {
    if input.cols != gate.cols() || gate.rows() != up.rows() || gate.cols() != up.cols() {
        return Err(format!("MXFP4 x MXFP8 gated matmul shape 不兼容: input=[{},{}], gate=[{},{}], up=[{},{}]", input.rows, input.cols, gate.rows(), gate.cols(), up.rows(), up.cols(),));
    }
    let activation = MetalMxfp8Activation::allocate(ctx, input)?;
    let gate_packed = ctx.shared_buffer(gate.packed());
    let gate_scales = ctx.shared_buffer(gate.scales());
    let up_packed = ctx.shared_buffer(up.packed());
    let up_scales = ctx.shared_buffer(up.scales());
    let output = ctx.tensor_zeros(input.rows, gate.rows());
    let params = GatedActivation::from_spec(activation_spec)?;
    let input_rows = validate_u32("MXFP4 x MXFP8 gated input rows", input.rows)?;
    let input_columns = validate_u32("MXFP4 x MXFP8 gated input columns", input.cols)?;
    let output_columns = validate_u32("MXFP4 x MXFP8 gated output columns", gate.rows())?;
    let pipeline = ctx.pipeline("mxfp4_mxfp8_gated_matmul_f16")?;
    let command = ctx.command_buffer();
    encode_mxfp8_activation(ctx, input, &activation, &command)?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&activation.codes), 0);
    encoder.set_buffer(1, Some(&activation.scales), 0);
    encoder.set_buffer(2, Some(&gate_packed), 0);
    encoder.set_buffer(3, Some(&gate_scales), 0);
    encoder.set_buffer(4, Some(&up_packed), 0);
    encoder.set_buffer(5, Some(&up_scales), 0);
    encoder.set_buffer(6, Some(&output.buffer), 0);
    set_bytes(&encoder, 7, &input_rows);
    set_bytes(&encoder, 8, &input_columns);
    set_bytes(&encoder, 9, &output_columns);
    set_bytes(&encoder, 10, &params.kind);
    set_bytes(&encoder, 11, &params.alpha);
    set_bytes(&encoder, 12, &params.limit);
    encoder.dispatch_thread_groups(MTLSize::new(gate.rows() as u64, input.rows as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{},{}],output={}", input.rows, input.cols, gate.rows());
    ctx.commit_and_wait_profiled(
        &command,
        "mxfp4_mxfp8_gated_matmul_f16",
        &shape,
        input.buffer.length() + gate_packed.length() + gate_scales.length() + up_packed.length() + up_scales.length(),
        activation.codes.length() + activation.scales.length() + output.buffer.length(),
    );
    Ok(output)
}

#[cfg(test)]
mod mxfp4_gated_tests {
    use super::*;
    use crate::kernel::metal::tensor::gated_activation_tensor;
    use crate::weight::format::mxfp4::{MXFP4_GROUP_SIZE, Mxfp4Matrix};

    #[test]
    fn fused_gated_matmul_matches_unfused() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let rows = 3;
        let columns = MXFP4_GROUP_SIZE;
        let gate = Mxfp4Matrix::new(rows, columns, vec![0x22; rows * columns / 2], vec![127; rows]).unwrap();
        let up = Mxfp4Matrix::new(rows, columns, vec![0x44; rows * columns / 2], vec![127; rows]).unwrap();
        let input = ctx.tensor_from_f32(&vec![0.01; columns], 1, columns).unwrap();
        let gate_output = mxfp4_matmul_tensor(&ctx, &input, &gate).unwrap();
        let up_output = mxfp4_matmul_tensor(&ctx, &input, &up).unwrap();
        let expected = gated_activation_tensor(&ctx, &gate_output, &up_output, &Activation::GeluTanh).unwrap();
        let actual = mxfp4_gated_matmul_tensor(&ctx, &input, &gate, &up, &Activation::GeluTanh).unwrap();
        for (actual, expected) in ctx.tensor_to_f32(&actual).into_iter().zip(ctx.tensor_to_f32(&expected)) {
            assert!((actual - expected).abs() <= 0.01, "actual={actual}, expected={expected}");
        }
    }
}

#[cfg(test)]
mod mxfp8_activation_tests {
    use crate::{backend::metal::MetalContext, moe::Activation, weight::format::mxfp4::Mxfp4Matrix};

    use super::{mxfp4_gated_matmul_tensor, mxfp4_matmul_tensor, mxfp4_mxfp8_gated_matmul_tensor, mxfp4_mxfp8_matmul_tensor};
    use crate::kernel::metal::kernels_source;

    #[test]
    fn mxfp4乘mxfp8与f16_activation一致() {
        let ctx = MetalContext::new(kernels_source()).unwrap();
        let data = (0..64).map(|index| ((index % 13) as f32 - 6.0) * 0.125).collect::<Vec<_>>();
        let input = ctx.tensor_from_f32(&data, 2, 32).unwrap();
        let gate = matrix(5, 32, 0x32);
        let up = matrix(5, 32, 0x21);

        let expected = mxfp4_matmul_tensor(&ctx, &input, &gate).unwrap();
        let actual = mxfp4_mxfp8_matmul_tensor(&ctx, &input, &gate).unwrap();
        assert_close(&ctx.tensor_to_f32(&actual), &ctx.tensor_to_f32(&expected));

        let expected = mxfp4_gated_matmul_tensor(&ctx, &input, &gate, &up, &Activation::Silu).unwrap();
        let actual = mxfp4_mxfp8_gated_matmul_tensor(&ctx, &input, &gate, &up, &Activation::Silu).unwrap();
        assert_close(&ctx.tensor_to_f32(&actual), &ctx.tensor_to_f32(&expected));
    }

    fn matrix(rows: usize, cols: usize, packed_code: u8) -> Mxfp4Matrix {
        Mxfp4Matrix::new(rows, cols, vec![packed_code; rows * cols / 2], vec![127; rows * cols / 32]).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            let tolerance = 0.02 + 0.02 * expected.abs();
            assert!((actual - expected).abs() <= tolerance, "actual={actual}, expected={expected}, tolerance={tolerance}",);
        }
    }
}
