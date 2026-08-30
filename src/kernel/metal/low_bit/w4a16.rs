//! compressed-tensors W4A16/W8A16 group-wise symmetric INT4/INT8,decode GEMV 与 prefill matmul。

/// 本家族的 Metal shader。跨家族共享 helper(w4a16_scale 等)见
/// `super::super::preamble`,由 `kernels_source()` 统一拼接。
pub const W4A16_SHADERS: &str = r#"
kernel void w4a16_gemv_f16(
    device const ushort *input [[buffer(0)]],
    device const uint *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device ushort *output [[buffer(3)]],
    constant uint &input_columns [[buffer(4)]],
    constant uint &output_columns [[buffer(5)]],
    constant uint &group_size [[buffer(6)]],
    constant uint &scale_dtype [[buffer(7)]],
    constant uint &input_bf16 [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const ulong groups = (ulong(input_columns) + group_size - 1) / group_size;
    const bool paired_rows = groups <= 256;
    const uint first_row = row * (paired_rows ? 2 : 1);
    if (first_row >= output_columns) return;
    const uint second_row = first_row + 1;
    const bool second_active = paired_rows && second_row < output_columns;
    threadgroup float first_sums[2];
    threadgroup float second_sums[2];
    threadgroup float scale_tile[512];
    const ulong packed_columns = (ulong(input_columns) + 7) / 8;
    const bool tiled_scales = groups <= 512;
    if (tiled_scales) {
        for (uint group = lane; group < uint(groups); group += 64) {
            scale_tile[group] = w4a16_scale(scales, ulong(first_row) * groups + group, scale_dtype);
            if (second_active) {
                scale_tile[256 + group] = w4a16_scale(scales, ulong(second_row) * groups + group, scale_dtype);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    for (uint column = lane; column < input_columns; column += 64) {
        const uint shift = (column & 7) * 4;
        const uint first_word = packed[ulong(first_row) * packed_columns + column / 8];
        const int first_code = int((first_word >> shift) & 15) - 8;
        const float first_scale = tiled_scales
            ? scale_tile[column / group_size]
            : w4a16_scale(scales, ulong(first_row) * groups + column / group_size, scale_dtype);
        const ushort input_bits = input[column];
        const float input_value = input_bf16 != 0 ? zllm_bf16_to_f32(input_bits) : float(as_type<half>(input_bits));
        first_sum += input_value * float(first_code) * first_scale;
        if (second_active) {
            const uint second_word = packed[ulong(second_row) * packed_columns + column / 8];
            const int second_code = int((second_word >> shift) & 15) - 8;
            const float second_scale = tiled_scales
                ? scale_tile[256 + column / group_size]
                : w4a16_scale(scales, ulong(second_row) * groups + column / group_size, scale_dtype);
            second_sum += input_value * float(second_code) * second_scale;
        }
    }
    const float first_simd_sum = simd_sum(first_sum);
    const float second_simd_sum = simd_sum(second_sum);
    if (simd_lane == 0) {
        first_sums[simd_group] = first_simd_sum;
        second_sums[simd_group] = second_simd_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        const float first_result = first_sums[0] + first_sums[1];
        output[first_row] = input_bf16 != 0
            ? zllm_f32_to_bf16(first_result)
            : as_type<ushort>(finite_f16(first_result));
        if (second_active) {
            const float second_result = second_sums[0] + second_sums[1];
            output[second_row] = input_bf16 != 0
                ? zllm_f32_to_bf16(second_result)
                : as_type<ushort>(finite_f16(second_result));
        }
    }
}
kernel void w8a16_gemv_f16(
    device const ushort *input [[buffer(0)]],
    device const uchar *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &input_columns [[buffer(4)]],
    constant uint &output_columns [[buffer(5)]],
    constant uint &group_size [[buffer(6)]],
    constant uint &scale_dtype [[buffer(7)]],
    constant uint &input_bf16 [[buffer(8)]],
    constant uint &input_rows [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    // 布局同 w4a16_gemv_f16：64 threads/tg、最多 2 个输出行共享 scale tile 预载，
    // simd_sum 归约替代旧 w8a16_matmul_f16 的 6 次 barrier 树归约；
    // group.y 维度覆盖 MTP verify 的多输入行。
    const ulong groups = ulong(input_columns) / group_size;
    const bool paired_rows = groups <= 256;
    const uint first_row = group.x * (paired_rows ? 2 : 1);
    if (first_row >= output_columns) return;
    const uint second_row = first_row + 1;
    const bool second_active = paired_rows && second_row < output_columns;
    const uint input_row = group.y;
    if (input_row >= input_rows) return;
    threadgroup float first_sums[2];
    threadgroup float second_sums[2];
    threadgroup float scale_tile[512];
    if (paired_rows) {
        for (uint g = lane; g < uint(groups); g += 64) {
            scale_tile[g] = w4a16_scale(scales, ulong(first_row) * groups + g, scale_dtype);
            if (second_active) {
                scale_tile[256 + g] = w4a16_scale(scales, ulong(second_row) * groups + g, scale_dtype);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device const ushort *input_ptr = input + ulong(input_row) * input_columns;
    const ulong packed_columns = (ulong(input_columns) + 3) / 4 * 4;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    for (uint column = lane; column < input_columns; column += 64) {
        const ushort bits = input_ptr[column];
        const float value = input_bf16 != 0
            ? as_type<float>(uint(bits) << 16)
            : float(as_type<half>(bits));
        const uint group_index = column / group_size;
        const int first_code = int(packed[ulong(first_row) * packed_columns + column]) - 128;
        const float first_scale = paired_rows
            ? scale_tile[group_index]
            : w4a16_scale(scales, ulong(first_row) * groups + group_index, scale_dtype);
        first_sum += value * float(first_code) * first_scale;
        if (second_active) {
            const int second_code = int(packed[ulong(second_row) * packed_columns + column]) - 128;
            second_sum += value * float(second_code) * scale_tile[256 + group_index];
        }
    }
    const float first_simd_sum = simd_sum(first_sum);
    const float second_simd_sum = simd_sum(second_sum);
    if (simd_lane == 0) {
        first_sums[simd_group] = first_simd_sum;
        second_sums[simd_group] = second_simd_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        const ulong output_base = ulong(input_row) * output_columns;
        output[output_base + first_row] = finite_f16(first_sums[0] + first_sums[1]);
        if (second_active) {
            output[output_base + second_row] = finite_f16(second_sums[0] + second_sums[1]);
        }
    }
}
kernel void w8a16_matmul_f16(
    device const ushort *input [[buffer(0)]],
    device const uchar *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &input_columns [[buffer(4)]],
    constant uint &output_columns [[buffer(5)]],
    constant uint &group_size [[buffer(6)]],
    constant uint &scale_dtype [[buffer(7)]],
    constant uint &input_bf16 [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint output_column = group.x;
    const uint input_row = group.y;
    if (output_column >= output_columns) return;
    threadgroup float sums[64];
    const ulong packed_columns = ((ulong(input_columns) + 3) / 4) * 4;
    const ulong groups = ulong(input_columns) / group_size;
    const ulong input_base = ulong(input_row) * input_columns;
    const ulong packed_base = ulong(output_column) * packed_columns;
    float sum = 0.0f;
    for (uint column = lane; column < input_columns; column += 64) {
        const ushort bits = input[input_base + column];
        const float value = input_bf16 != 0
            ? as_type<float>(uint(bits) << 16)
            : float(as_type<half>(bits));
        const int code = int(packed[packed_base + column]) - 128;
        const ulong parameter = ulong(output_column) * groups + column / group_size;
        sum += value * float(code) * w4a16_scale(scales, parameter, scale_dtype);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[ulong(input_row) * output_columns + output_column] = finite_f16(sums[0]);
}
kernel void w8a16_gated_matmul_f16(
    device const ushort *input [[buffer(0)]],
    device const uchar *gate_packed [[buffer(1)]],
    device const uchar *gate_scales [[buffer(2)]],
    device const uchar *up_packed [[buffer(3)]],
    device const uchar *up_scales [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]],
    constant uint &output_columns [[buffer(7)]],
    constant uint &gate_group_size [[buffer(8)]],
    constant uint &up_group_size [[buffer(9)]],
    constant uint &gate_scale_dtype [[buffer(10)]],
    constant uint &up_scale_dtype [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &activation_alpha [[buffer(13)]],
    constant float &activation_limit [[buffer(14)]],
    constant uint &input_bf16 [[buffer(15)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint output_column = group.x;
    const uint input_row = group.y;
    if (output_column >= output_columns) return;
    threadgroup float gate_sums[64];
    threadgroup float up_sums[64];
    const ulong packed_columns = ((ulong(input_columns) + 3) / 4) * 4;
    const ulong gate_groups = ulong(input_columns) / gate_group_size;
    const ulong up_groups = ulong(input_columns) / up_group_size;
    const ulong input_base = ulong(input_row) * input_columns;
    const ulong packed_base = ulong(output_column) * packed_columns;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint column = lane; column < input_columns; column += 64) {
        const ushort bits = input[input_base + column];
        const float value = input_bf16 != 0 ? as_type<float>(uint(bits) << 16) : float(as_type<half>(bits));
        const int gate_code = int(gate_packed[packed_base + column]) - 128;
        const int up_code = int(up_packed[packed_base + column]) - 128;
        gate_sum += value * float(gate_code) * w4a16_scale(
            gate_scales, ulong(output_column) * gate_groups + column / gate_group_size, gate_scale_dtype);
        up_sum += value * float(up_code) * w4a16_scale(
            up_scales, ulong(output_column) * up_groups + column / up_group_size, up_scale_dtype);
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
        const float gate = float(finite_f16(gate_sums[0]));
        const float up = float(finite_f16(up_sums[0]));
        output[ulong(input_row) * output_columns + output_column] = finite_f16(gated_activation_value(
            gate, up, activation_kind, activation_alpha, activation_limit));
    }
}
kernel void w4a16_matmul_f16(
    device const ushort *input [[buffer(0)]],
    device const uchar *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &input_columns [[buffer(4)]],
    constant uint &output_columns [[buffer(5)]],
    constant uint &group_size [[buffer(6)]],
    constant uint &scale_dtype [[buffer(7)]],
    constant uint &input_bf16 [[buffer(8)]],
    constant uint &input_rows [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint output_column = group.x;
    constexpr uint row_tile = 8;
    const uint input_row = group.y * row_tile;
    if (output_column >= output_columns) return;
    threadgroup float sums[row_tile][64];
    const ulong packed_columns = (ulong(input_columns) + 7) / 8 * 4;
    const ulong groups = ulong(input_columns) / group_size;
    const ulong packed_base = ulong(output_column) * packed_columns;
    float sum[row_tile];
    for (uint row = 0; row < row_tile; ++row) sum[row] = 0.0f;
    for (uint column = lane; column < input_columns; column += 64) {
        const uchar pair = packed[packed_base + (column / 8) * 4 + (column % 8) / 2];
        const int code = int((column & 1) == 0 ? pair & 15 : pair >> 4) - 8;
        const ulong parameter = ulong(output_column) * groups + column / group_size;
        const float scaled_code = float(code) * w4a16_scale(scales, parameter, scale_dtype);
        for (uint row = 0; row < row_tile && input_row + row < input_rows; ++row) {
            const ushort bits = input[ulong(input_row + row) * input_columns + column];
            const float value = input_bf16 != 0 ? as_type<float>(uint(bits) << 16) : float(as_type<half>(bits));
            sum[row] += value * scaled_code;
        }
    }
    for (uint row = 0; row < row_tile; ++row) sums[row][lane] = sum[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint row = 0; row < row_tile; ++row) sums[row][lane] += sums[row][lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint row = 0; row < row_tile && input_row + row < input_rows; ++row) {
            output[ulong(input_row + row) * output_columns + output_column] = finite_f16(sums[row][0]);
        }
    }
}
kernel void w4a16_gated_matmul_f16(
    device const ushort *input [[buffer(0)]],
    device const uchar *gate_packed [[buffer(1)]],
    device const uchar *gate_scales [[buffer(2)]],
    device const uchar *up_packed [[buffer(3)]],
    device const uchar *up_scales [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]],
    constant uint &output_columns [[buffer(7)]],
    constant uint &gate_group_size [[buffer(8)]],
    constant uint &up_group_size [[buffer(9)]],
    constant uint &gate_scale_dtype [[buffer(10)]],
    constant uint &up_scale_dtype [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &activation_alpha [[buffer(13)]],
    constant float &activation_limit [[buffer(14)]],
    constant uint &input_bf16 [[buffer(15)]],
    constant uint &input_rows [[buffer(16)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint output_column = group.x;
    constexpr uint row_tile = 8;
    const uint input_row = group.y * row_tile;
    if (output_column >= output_columns) return;
    threadgroup float gate_sums[row_tile][64];
    threadgroup float up_sums[row_tile][64];
    const ulong packed_columns = (ulong(input_columns) + 7) / 8 * 4;
    const ulong gate_groups = ulong(input_columns) / gate_group_size;
    const ulong up_groups = ulong(input_columns) / up_group_size;
    const ulong packed_base = ulong(output_column) * packed_columns;
    float gate_sum[row_tile];
    float up_sum[row_tile];
    for (uint row = 0; row < row_tile; ++row) {
        gate_sum[row] = 0.0f;
        up_sum[row] = 0.0f;
    }
    for (uint column = lane; column < input_columns; column += 64) {
        const ulong packed_index = packed_base + (column / 8) * 4 + (column % 8) / 2;
        const uchar gate_pair = gate_packed[packed_index];
        const uchar up_pair = up_packed[packed_index];
        const int gate_code = int((column & 1) == 0 ? gate_pair & 15 : gate_pair >> 4) - 8;
        const int up_code = int((column & 1) == 0 ? up_pair & 15 : up_pair >> 4) - 8;
        const float scaled_gate = float(gate_code) * w4a16_scale(gate_scales, ulong(output_column) * gate_groups + column / gate_group_size, gate_scale_dtype);
        const float scaled_up = float(up_code) * w4a16_scale(up_scales, ulong(output_column) * up_groups + column / up_group_size, up_scale_dtype);
        for (uint row = 0; row < row_tile && input_row + row < input_rows; ++row) {
            const ushort bits = input[ulong(input_row + row) * input_columns + column];
            const float value = input_bf16 != 0 ? as_type<float>(uint(bits) << 16) : float(as_type<half>(bits));
            gate_sum[row] += value * scaled_gate;
            up_sum[row] += value * scaled_up;
        }
    }
    for (uint row = 0; row < row_tile; ++row) {
        gate_sums[row][lane] = gate_sum[row];
        up_sums[row][lane] = up_sum[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint row = 0; row < row_tile; ++row) {
                gate_sums[row][lane] += gate_sums[row][lane + stride];
                up_sums[row][lane] += up_sums[row][lane + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint row = 0; row < row_tile && input_row + row < input_rows; ++row) {
            output[ulong(input_row + row) * output_columns + output_column] = finite_f16(gated_activation_value(
                gate_sums[row][0], up_sums[row][0], activation_kind, activation_alpha, activation_limit));
        }
    }
}
kernel void w4a16_dual_gemv_f16(
    device const ushort *input [[buffer(0)]],
    device const uint *first_packed [[buffer(1)]],
    device const uchar *first_scales [[buffer(2)]],
    device const uint *second_packed [[buffer(3)]],
    device const uchar *second_scales [[buffer(4)]],
    device ushort *first_output [[buffer(5)]],
    device ushort *second_output [[buffer(6)]],
    constant uint &input_columns [[buffer(7)]],
    constant uint &first_output_columns [[buffer(8)]],
    constant uint &second_output_columns [[buffer(9)]],
    constant uint &first_group_size [[buffer(10)]],
    constant uint &second_group_size [[buffer(11)]],
    constant uint &first_scale_dtype [[buffer(12)]],
    constant uint &second_scale_dtype [[buffer(13)]],
    constant uint &input_bf16 [[buffer(14)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const ulong first_groups = (ulong(input_columns) + first_group_size - 1) / first_group_size;
    const ulong second_groups = (ulong(input_columns) + second_group_size - 1) / second_group_size;
    const bool first_active = row < first_output_columns;
    const bool second_active = row < second_output_columns;
    if (!first_active && !second_active) return;
    threadgroup float first_sums[2];
    threadgroup float second_sums[2];
    threadgroup float first_scale_tile[512];
    threadgroup float second_scale_tile[512];
    const ulong packed_columns = (ulong(input_columns) + 7) / 8;
    const bool first_tiled_scales = first_groups <= 512;
    const bool second_tiled_scales = second_groups <= 512;
    if (first_active && first_tiled_scales) {
        for (uint group = lane; group < uint(first_groups); group += 64) {
            first_scale_tile[group] = w4a16_scale(first_scales, ulong(row) * first_groups + group, first_scale_dtype);
        }
    }
    if (second_active && second_tiled_scales) {
        for (uint group = lane; group < uint(second_groups); group += 64) {
            second_scale_tile[group] = w4a16_scale(second_scales, ulong(row) * second_groups + group, second_scale_dtype);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    for (uint column = lane; column < input_columns; column += 64) {
        const ushort input_bits = input[column];
        const float value = input_bf16 != 0 ? zllm_bf16_to_f32(input_bits) : float(as_type<half>(input_bits));
        if (first_active) {
            const uint word = first_packed[ulong(row) * packed_columns + column / 8];
            const int code = int((word >> ((column & 7) * 4)) & 15) - 8;
            const float scale = first_tiled_scales
                ? first_scale_tile[column / first_group_size]
                : w4a16_scale(first_scales, ulong(row) * first_groups + column / first_group_size, first_scale_dtype);
            first_sum += value * float(code) * scale;
        }
        if (second_active) {
            const uint word = second_packed[ulong(row) * packed_columns + column / 8];
            const int code = int((word >> ((column & 7) * 4)) & 15) - 8;
            const float scale = second_tiled_scales
                ? second_scale_tile[column / second_group_size]
                : w4a16_scale(second_scales, ulong(row) * second_groups + column / second_group_size, second_scale_dtype);
            second_sum += value * float(code) * scale;
        }
    }
    const float first_simd_sum = simd_sum(first_sum);
    const float second_simd_sum = simd_sum(second_sum);
    if (simd_lane == 0) {
        first_sums[simd_group] = first_simd_sum;
        second_sums[simd_group] = second_simd_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        const float first_result = first_sums[0] + first_sums[1];
        const float second_result = second_sums[0] + second_sums[1];
        if (first_active) first_output[row] = input_bf16 != 0
            ? zllm_f32_to_bf16(first_result)
            : as_type<ushort>(finite_f16(first_result));
        if (second_active) second_output[row] = input_bf16 != 0
            ? zllm_f32_to_bf16(second_result)
            : as_type<ushort>(finite_f16(second_result));
    }
}
kernel void w4a16_gated_gemv_f16(
    device const ushort *input [[buffer(0)]],
    device const uint *gate_packed [[buffer(1)]],
    device const uchar *gate_scales [[buffer(2)]],
    device const uint *up_packed [[buffer(3)]],
    device const uchar *up_scales [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]],
    constant uint &output_columns [[buffer(7)]],
    constant uint &gate_group_size [[buffer(8)]],
    constant uint &up_group_size [[buffer(9)]],
    constant uint &gate_scale_dtype [[buffer(10)]],
    constant uint &up_scale_dtype [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &activation_alpha [[buffer(13)]],
    constant float &activation_limit [[buffer(14)]],
    constant uint &input_bf16 [[buffer(15)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const ulong gate_groups = (ulong(input_columns) + gate_group_size - 1) / gate_group_size;
    const ulong up_groups = (ulong(input_columns) + up_group_size - 1) / up_group_size;
    const bool paired_rows = gate_groups <= 256 && up_groups <= 256;
    const uint first_row = row * (paired_rows ? 2 : 1);
    if (first_row >= output_columns) return;
    const uint second_row = first_row + 1;
    const bool second_active = paired_rows && second_row < output_columns;
    threadgroup float first_gate_sums[2];
    threadgroup float first_up_sums[2];
    threadgroup float second_gate_sums[2];
    threadgroup float second_up_sums[2];
    threadgroup float gate_scale_tile[512];
    threadgroup float up_scale_tile[512];
    const ulong packed_columns = (ulong(input_columns) + 7) / 8;
    const bool gate_tiled_scales = gate_groups <= 512;
    const bool up_tiled_scales = up_groups <= 512;
    if (gate_tiled_scales) {
        for (uint group = lane; group < uint(gate_groups); group += 64) {
            gate_scale_tile[group] = w4a16_scale(gate_scales, ulong(first_row) * gate_groups + group, gate_scale_dtype);
            if (second_active) {
                gate_scale_tile[256 + group] = w4a16_scale(gate_scales, ulong(second_row) * gate_groups + group, gate_scale_dtype);
            }
        }
    }
    if (up_tiled_scales) {
        for (uint group = lane; group < uint(up_groups); group += 64) {
            up_scale_tile[group] = w4a16_scale(up_scales, ulong(first_row) * up_groups + group, up_scale_dtype);
            if (second_active) {
                up_scale_tile[256 + group] = w4a16_scale(up_scales, ulong(second_row) * up_groups + group, up_scale_dtype);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float first_gate_sum = 0.0f;
    float first_up_sum = 0.0f;
    float second_gate_sum = 0.0f;
    float second_up_sum = 0.0f;
    for (uint column = lane; column < input_columns; column += 64) {
        const ushort input_bits = input[column];
        const float value = input_bf16 != 0 ? zllm_bf16_to_f32(input_bits) : float(as_type<half>(input_bits));
        const uint shift = (column & 7) * 4;
        const uint first_gate_word = gate_packed[ulong(first_row) * packed_columns + column / 8];
        const uint first_up_word = up_packed[ulong(first_row) * packed_columns + column / 8];
        const int first_gate_code = int((first_gate_word >> shift) & 15) - 8;
        const int first_up_code = int((first_up_word >> shift) & 15) - 8;
        const float first_gate_scale = gate_tiled_scales
            ? gate_scale_tile[column / gate_group_size]
            : w4a16_scale(gate_scales, ulong(first_row) * gate_groups + column / gate_group_size, gate_scale_dtype);
        const float first_up_scale = up_tiled_scales
            ? up_scale_tile[column / up_group_size]
            : w4a16_scale(up_scales, ulong(first_row) * up_groups + column / up_group_size, up_scale_dtype);
        first_gate_sum += value * float(first_gate_code) * first_gate_scale;
        first_up_sum += value * float(first_up_code) * first_up_scale;
        if (second_active) {
            const uint second_gate_word = gate_packed[ulong(second_row) * packed_columns + column / 8];
            const uint second_up_word = up_packed[ulong(second_row) * packed_columns + column / 8];
            const int second_gate_code = int((second_gate_word >> shift) & 15) - 8;
            const int second_up_code = int((second_up_word >> shift) & 15) - 8;
            const float second_gate_scale = gate_tiled_scales
                ? gate_scale_tile[256 + column / gate_group_size]
                : w4a16_scale(gate_scales, ulong(second_row) * gate_groups + column / gate_group_size, gate_scale_dtype);
            const float second_up_scale = up_tiled_scales
                ? up_scale_tile[256 + column / up_group_size]
                : w4a16_scale(up_scales, ulong(second_row) * up_groups + column / up_group_size, up_scale_dtype);
            second_gate_sum += value * float(second_gate_code) * second_gate_scale;
            second_up_sum += value * float(second_up_code) * second_up_scale;
        }
    }
    const float first_gate_simd_sum = simd_sum(first_gate_sum);
    const float first_up_simd_sum = simd_sum(first_up_sum);
    const float second_gate_simd_sum = simd_sum(second_gate_sum);
    const float second_up_simd_sum = simd_sum(second_up_sum);
    if (simd_lane == 0) {
        first_gate_sums[simd_group] = first_gate_simd_sum;
        first_up_sums[simd_group] = first_up_simd_sum;
        second_gate_sums[simd_group] = second_gate_simd_sum;
        second_up_sums[simd_group] = second_up_simd_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        const half first_gate_f16 = finite_f16(first_gate_sums[0] + first_gate_sums[1]);
        const half first_up_f16 = finite_f16(first_up_sums[0] + first_up_sums[1]);
        const float first_value = gated_activation_value(
            float(first_gate_f16), float(first_up_f16), activation_kind, activation_alpha, activation_limit);
        output[first_row] = input_bf16 != 0
            ? zllm_f32_to_bf16(first_value)
            : as_type<ushort>(finite_f16(first_value));
        if (second_active) {
            const half second_gate_f16 = finite_f16(second_gate_sums[0] + second_gate_sums[1]);
            const half second_up_f16 = finite_f16(second_up_sums[0] + second_up_sums[1]);
            const float second_value = gated_activation_value(
                float(second_gate_f16), float(second_up_f16), activation_kind, activation_alpha, activation_limit);
            output[second_row] = input_bf16 != 0
                ? zllm_f32_to_bf16(second_value)
                : as_type<ushort>(finite_f16(second_value));
        }
    }
}
"#;

use crate::backend::metal::api as metal;

use super::super::dense::GatedActivation;
use super::super::mlx::validate_w4a16_storage;
use super::super::{Activation, MTLSize, MetalContext, MetalTensor, set_bytes, validate_u32};

#[cfg(test)]
use super::super::as_bytes;

/// W4A16 decode 使用专用 GEMV；prefill 直接读取压缩权重，不展开完整 F16 矩阵。
#[allow(clippy::too_many_arguments)]
pub fn w4a16_matmul_tensor_resident(ctx: &MetalContext, input: &MetalTensor, packed: &metal::Buffer, scales: &metal::Buffer, scale_dtype: u32, group_size: usize, weight_rows: usize, weight_cols: usize) -> Result<MetalTensor, String> {
    if input.cols != weight_cols {
        return Err(format!("W4A16 input=[{},{}] weight=[{weight_rows},{weight_cols}] group_size={group_size} 不兼容", input.rows, input.cols,));
    }
    validate_w4a16_storage("resident W4A16", packed, scales, scale_dtype, group_size, weight_rows, weight_cols)?;
    if input.rows != 1 {
        let input_columns = validate_u32("W4A16 input columns", weight_cols)?;
        let output_columns = validate_u32("W4A16 output columns", weight_rows)?;
        let group_size = validate_u32("W4A16 group size", group_size)?;
        let input_bf16 = u32::from(input.dtype == super::MetalTensorDType::Bf16);
        let output = ctx.tensor_zeros(input.rows, weight_rows);
        let pipeline = ctx.pipeline("w4a16_matmul_f16")?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        set_bytes(&encoder, 4, &input_columns);
        set_bytes(&encoder, 5, &output_columns);
        set_bytes(&encoder, 6, &group_size);
        set_bytes(&encoder, 7, &scale_dtype);
        set_bytes(&encoder, 8, &input_bf16);
        let input_rows = validate_u32("W4A16 input rows", input.rows)?;
        set_bytes(&encoder, 9, &input_rows);
        encoder.dispatch_thread_groups(MTLSize::new(weight_rows as u64, input.rows.div_ceil(8) as u64, 1), MTLSize::new(64, 1, 1));
        encoder.end_encoding();
        let shape = format!("input=[{},{}],weight=[{weight_rows},{weight_cols}],group={group_size}", input.rows, input.cols);
        ctx.commit_and_wait_profiled(&command, "w4a16_matmul_f16", &shape, input.buffer.length() + packed.length() + scales.length(), output.buffer.length());
        return Ok(output);
    }

    let input_columns = validate_u32("W4A16 input columns", weight_cols)?;
    let output_columns = validate_u32("W4A16 output columns", weight_rows)?;
    let rows_per_group = if weight_cols.div_ceil(group_size) <= 256 { 2 } else { 1 };
    let group_size = validate_u32("W4A16 group size", group_size)?;
    let input_bf16 = u32::from(input.dtype == super::MetalTensorDType::Bf16);
    let output = if input_bf16 != 0 { ctx.tensor_zeros_bf16(1, weight_rows) } else { ctx.tensor_zeros(1, weight_rows) };
    let pipeline = ctx.pipeline("w4a16_gemv_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("W4A16 GEMV 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(packed), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &input_columns);
    set_bytes(&encoder, 5, &output_columns);
    set_bytes(&encoder, 6, &group_size);
    set_bytes(&encoder, 7, &scale_dtype);
    set_bytes(&encoder, 8, &input_bf16);
    encoder.dispatch_thread_groups(MTLSize::new(weight_rows.div_ceil(rows_per_group) as u64, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{weight_cols}],weight=[{weight_rows},{weight_cols}],group={group_size}");
    ctx.commit_and_wait_profiled(&command, "w4a16_gemv_f16", &shape, input.buffer.length() + packed.length() + scales.length(), output.buffer.length());
    Ok(output)
}

/// W8A16 group-wise symmetric matmul；权重保持压缩态，prefill/decode 共用同一 kernel。
#[allow(clippy::too_many_arguments)]
pub fn w8a16_matmul_tensor_resident(ctx: &MetalContext, input: &MetalTensor, packed: &metal::Buffer, scales: &metal::Buffer, scale_dtype: u32, group_size: usize, weight_rows: usize, weight_cols: usize) -> Result<MetalTensor, String> {
    if input.cols != weight_cols || weight_rows == 0 || weight_cols == 0 || group_size == 0 || !weight_cols.is_multiple_of(group_size) {
        return Err(format!("W8A16 input=[{},{}] weight=[{weight_rows},{weight_cols}] group_size={group_size} 不兼容", input.rows, input.cols,));
    }
    let packed_bytes = weight_rows.checked_mul(weight_cols.div_ceil(4)).and_then(|words| words.checked_mul(4)).ok_or_else(|| "W8A16 packed 大小溢出".to_owned())?;
    let scale_bytes = match scale_dtype {
        0 | 1 => 2,
        2 => 4,
        _ => return Err(format!("W8A16 scale dtype id {scale_dtype} 无效")),
    };
    let expected_scales = weight_rows.checked_mul(weight_cols / group_size).and_then(|count| count.checked_mul(scale_bytes)).ok_or_else(|| "W8A16 scale 大小溢出".to_owned())?;
    if packed.length() != packed_bytes as u64 || scales.length() != expected_scales as u64 {
        return Err(format!("W8A16 buffer packed={}/{} scales={}/{}", packed.length(), packed_bytes, scales.length(), expected_scales,));
    }

    let input_columns = validate_u32("W8A16 input columns", weight_cols)?;
    let output_columns = validate_u32("W8A16 output columns", weight_rows)?;
    let group_size = validate_u32("W8A16 group size", group_size)?;
    let input_bf16 = u32::from(input.dtype == super::MetalTensorDType::Bf16);
    let output = ctx.tensor_zeros(input.rows, weight_rows);
    // decode/MTP verify（rows ≤ 2）走 simdgroup gemv；prefill 大 rows 保留逐列 matmul。
    if input.rows <= 2 {
        let pipeline = ctx.pipeline("w8a16_gemv_f16")?;
        if pipeline.max_total_threads_per_threadgroup() < 64 {
            return Err("W8A16 gemv 需要至少 64 threads/threadgroup".to_owned());
        }
        let rows_per_group = if weight_cols as u32 / group_size <= 256 { 2 } else { 1 };
        let input_rows = validate_u32("W8A16 gemv input rows", input.rows)?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        set_bytes(&encoder, 4, &input_columns);
        set_bytes(&encoder, 5, &output_columns);
        set_bytes(&encoder, 6, &group_size);
        set_bytes(&encoder, 7, &scale_dtype);
        set_bytes(&encoder, 8, &input_bf16);
        set_bytes(&encoder, 9, &input_rows);
        encoder.dispatch_thread_groups(MTLSize::new(weight_rows.div_ceil(rows_per_group) as u64, input.rows as u64, 1), MTLSize::new(64, 1, 1));
        encoder.end_encoding();
        let shape = format!("input=[{},{}],weight=[{weight_rows},{weight_cols}],group={group_size}", input.rows, input.cols);
        ctx.commit_and_wait_profiled(&command, "w8a16_gemv_f16", &shape, input.buffer.length() + packed.length() + scales.length(), output.buffer.length());
        return Ok(output);
    }
    let pipeline = ctx.pipeline("w8a16_matmul_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("W8A16 matmul 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(packed), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &input_columns);
    set_bytes(&encoder, 5, &output_columns);
    set_bytes(&encoder, 6, &group_size);
    set_bytes(&encoder, 7, &scale_dtype);
    set_bytes(&encoder, 8, &input_bf16);
    encoder.dispatch_thread_groups(MTLSize::new(weight_rows as u64, input.rows as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{},{}],weight=[{weight_rows},{weight_cols}],group={group_size}", input.rows, input.cols);
    ctx.commit_and_wait_profiled(&command, "w8a16_matmul_f16", &shape, input.buffer.length() + packed.length() + scales.length(), output.buffer.length());
    Ok(output)
}

/// W8A16 gate/up 共用输入读取并在写回前完成激活，避免两个完整投影中间张量。
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gated_matmul_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    gate_packed: &metal::Buffer,
    gate_scales: &metal::Buffer,
    gate_scale_dtype: u32,
    gate_group_size: usize,
    gate_rows: usize,
    gate_cols: usize,
    up_packed: &metal::Buffer,
    up_scales: &metal::Buffer,
    up_scale_dtype: u32,
    up_group_size: usize,
    up_rows: usize,
    up_cols: usize,
    activation: &Activation,
) -> Result<MetalTensor, String> {
    if input.cols != gate_cols || gate_cols != up_cols || gate_rows != up_rows {
        return Err(format!("W8A16 gated shape 不兼容: input=[{},{}], gate=[{gate_rows},{gate_cols}]/group={gate_group_size}, up=[{up_rows},{up_cols}]/group={up_group_size}", input.rows, input.cols,));
    }
    let validate = |name: &str, packed: &metal::Buffer, scales: &metal::Buffer, scale_dtype: u32, group_size: usize, rows: usize, cols: usize| {
        if group_size == 0 || !cols.is_multiple_of(group_size) {
            return Err(format!("{name} group_size={group_size} 与 cols={cols} 不兼容"));
        }
        let packed_bytes = rows.checked_mul(cols.div_ceil(4)).and_then(|words| words.checked_mul(4)).ok_or_else(|| format!("{name} packed 大小溢出"))?;
        let scale_bytes = match scale_dtype {
            0 | 1 => 2,
            2 => 4,
            _ => return Err(format!("{name} scale dtype id {scale_dtype} 无效")),
        };
        let scale_bytes = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(scale_bytes)).ok_or_else(|| format!("{name} scale 大小溢出"))?;
        if packed.length() != packed_bytes as u64 || scales.length() != scale_bytes as u64 {
            return Err(format!("{name} buffer packed={}/{} scales={}/{}", packed.length(), packed_bytes, scales.length(), scale_bytes,));
        }
        Ok(())
    };
    validate("W8A16 gate", gate_packed, gate_scales, gate_scale_dtype, gate_group_size, gate_rows, gate_cols)?;
    validate("W8A16 up", up_packed, up_scales, up_scale_dtype, up_group_size, up_rows, up_cols)?;

    let params = GatedActivation::from_spec(activation)?;
    let input_columns = validate_u32("W8A16 gated input columns", gate_cols)?;
    let output_columns = validate_u32("W8A16 gated output columns", gate_rows)?;
    let gate_group_size = validate_u32("W8A16 gated gate group size", gate_group_size)?;
    let up_group_size = validate_u32("W8A16 gated up group size", up_group_size)?;
    let input_bf16 = u32::from(input.dtype == super::MetalTensorDType::Bf16);
    let output = ctx.tensor_zeros(input.rows, gate_rows);
    let pipeline = ctx.pipeline("w8a16_gated_matmul_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("W8A16 gated matmul 需要至少 64 threads/threadgroup".to_owned());
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
    set_bytes(&encoder, 6, &input_columns);
    set_bytes(&encoder, 7, &output_columns);
    set_bytes(&encoder, 8, &gate_group_size);
    set_bytes(&encoder, 9, &up_group_size);
    set_bytes(&encoder, 10, &gate_scale_dtype);
    set_bytes(&encoder, 11, &up_scale_dtype);
    set_bytes(&encoder, 12, &params.kind);
    set_bytes(&encoder, 13, &params.alpha);
    set_bytes(&encoder, 14, &params.limit);
    set_bytes(&encoder, 15, &input_bf16);
    encoder.dispatch_thread_groups(MTLSize::new(gate_rows as u64, input.rows as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{},{}],output={gate_rows}", input.rows, input.cols);
    ctx.commit_and_wait_profiled(&command, "w8a16_gated_matmul_f16", &shape, input.buffer.length() + gate_packed.length() + gate_scales.length() + up_packed.length() + up_scales.length(), output.buffer.length());
    Ok(output)
}

#[cfg(test)]
mod w8a16_tests {
    use super::*;

    #[test]
    fn metal_matches_cpu_oracle() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let rows = 3usize;
        let cols = 32usize;
        let group_size = 16usize;
        let packed = (0..rows * cols).map(|index| ((index % 31) as i16 - 15 + 128) as u8).collect::<Vec<_>>();
        let scale_values = vec![0.03125f32; rows * cols / group_size];
        let scales = as_bytes(&scale_values).to_vec();
        let input_values = (0..cols).map(|index| index as f32 * 0.01 - 0.15).collect::<Vec<_>>();
        let mut expected = vec![0.0f32; rows];
        crate::kernel::cpu::w4a16::matvec_w8a16_matrix(&packed, &scales, crate::weight::format::quantization::ScaleDType::F32, group_size, rows, cols, &input_values, &mut expected).unwrap();
        let input = ctx.tensor_from_f32(&input_values, 1, cols).unwrap();
        let packed = ctx.shared_buffer(&packed);
        let scales = ctx.shared_buffer(&scales);
        let actual = w8a16_matmul_tensor_resident(&ctx, &input, &packed, &scales, 2, group_size, rows, cols).unwrap();
        for (actual, expected) in ctx.tensor_to_f32(&actual).into_iter().zip(expected) {
            let tolerance = 0.01 + expected.abs() * 0.01;
            assert!((actual - expected).abs() <= tolerance, "actual={actual}, expected={expected}");
        }
    }

    #[test]
    fn gemv_two_rows_matches_cpu_oracle() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let input_rows = 2usize;
        let rows = 5usize;
        let cols = 64usize;
        let group_size = 32usize;
        // 覆盖 paired 双行与奇数行尾（第 5 行独占一个 threadgroup）
        let packed = (0..rows * cols).map(|index| ((index % 251) as i16 - 125 + 128) as u8).collect::<Vec<_>>();
        let scale_values: Vec<f32> = (0..rows * cols / group_size).map(|index| 0.02 + (index % 3) as f32 * 0.01).collect();
        let scales = as_bytes(&scale_values).to_vec();
        let input_values = (0..input_rows * cols).map(|index| ((index % 17) as f32 - 8.0) * 0.05).collect::<Vec<_>>();
        let mut expected = Vec::with_capacity(input_rows * rows);
        for input in input_values.chunks_exact(cols) {
            let mut row_output = vec![0.0f32; rows];
            crate::kernel::cpu::w4a16::matvec_w8a16_matrix(&packed, &scales, crate::weight::format::quantization::ScaleDType::F32, group_size, rows, cols, input, &mut row_output).unwrap();
            expected.extend(row_output);
        }
        let input = ctx.tensor_from_f32(&input_values, input_rows, cols).unwrap();
        let packed = ctx.shared_buffer(&packed);
        let scales = ctx.shared_buffer(&scales);
        let actual = w8a16_matmul_tensor_resident(&ctx, &input, &packed, &scales, 2, group_size, rows, cols).unwrap();
        for (actual, expected) in ctx.tensor_to_f32(&actual).into_iter().zip(expected) {
            let tolerance = 0.01 + expected.abs() * 0.01;
            assert!((actual - expected).abs() <= tolerance, "actual={actual}, expected={expected}");
        }
    }

    #[test]
    fn gated_prefill_matches_cpu_oracle() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let input_rows = 2usize;
        let rows = 3usize;
        let cols = 32usize;
        let group_size = 16usize;
        let gate = (0..rows * cols).map(|index| ((index % 29) as i16 - 14 + 128) as u8).collect::<Vec<_>>();
        let up = (0..rows * cols).map(|index| (14 - (index % 29) as i16 + 128) as u8).collect::<Vec<_>>();
        let scale_values = vec![0.03125f32; rows * cols / group_size];
        let scales = as_bytes(&scale_values).to_vec();
        let input_values = (0..input_rows * cols).map(|index| index as f32 * 0.01 - 0.25).collect::<Vec<_>>();
        let mut expected = Vec::with_capacity(input_rows * rows);
        for input in input_values.chunks_exact(cols) {
            let mut gate_output = vec![0.0f32; rows];
            let mut up_output = vec![0.0f32; rows];
            crate::kernel::cpu::w4a16::matvec_w8a16_matrix(&gate, &scales, crate::weight::format::quantization::ScaleDType::F32, group_size, rows, cols, input, &mut gate_output).unwrap();
            crate::kernel::cpu::w4a16::matvec_w8a16_matrix(&up, &scales, crate::weight::format::quantization::ScaleDType::F32, group_size, rows, cols, input, &mut up_output).unwrap();
            expected.extend(gate_output.into_iter().zip(up_output).map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up));
        }
        let input = ctx.tensor_from_f32(&input_values, input_rows, cols).unwrap();
        let gate = ctx.shared_buffer(&gate);
        let up = ctx.shared_buffer(&up);
        let scales = ctx.shared_buffer(&scales);
        let actual = w8a16_gated_matmul_tensor_resident(&ctx, &input, &gate, &scales, 2, group_size, rows, cols, &up, &scales, 2, group_size, rows, cols, &Activation::Silu).unwrap();
        for (actual, expected) in ctx.tensor_to_f32(&actual).into_iter().zip(expected) {
            let tolerance = 0.01 + expected.abs() * 0.01;
            assert!((actual - expected).abs() <= tolerance, "actual={actual}, expected={expected}");
        }
    }
}

/// Decode 同时计算 W4A16 gate/up 并完成激活，避免重复读取 input 和两个中间 tensor。
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gated_gemv_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    gate_packed: &metal::Buffer,
    gate_scales: &metal::Buffer,
    gate_scale_dtype: u32,
    gate_group_size: usize,
    gate_rows: usize,
    gate_cols: usize,
    up_packed: &metal::Buffer,
    up_scales: &metal::Buffer,
    up_scale_dtype: u32,
    up_group_size: usize,
    up_rows: usize,
    up_cols: usize,
    activation: &Activation,
) -> Result<MetalTensor, String> {
    if input.cols != gate_cols || gate_cols != up_cols || gate_rows != up_rows {
        return Err(format!("W4A16 gated GEMV shape 不兼容: input=[{},{}], gate=[{gate_rows},{gate_cols}]/group={gate_group_size}, up=[{up_rows},{up_cols}]/group={up_group_size}", input.rows, input.cols,));
    }
    validate_w4a16_storage("resident W4A16 gate", gate_packed, gate_scales, gate_scale_dtype, gate_group_size, gate_rows, gate_cols)?;

    if input.rows != 1 {
        let params = GatedActivation::from_spec(activation)?;
        let input_columns = validate_u32("W4A16 gated input columns", gate_cols)?;
        let output_columns = validate_u32("W4A16 gated output columns", gate_rows)?;
        let gate_group_size = validate_u32("W4A16 gated gate group size", gate_group_size)?;
        let up_group_size = validate_u32("W4A16 gated up group size", up_group_size)?;
        let input_bf16 = u32::from(input.dtype == super::MetalTensorDType::Bf16);
        let output = ctx.tensor_zeros(input.rows, gate_rows);
        let pipeline = ctx.pipeline("w4a16_gated_matmul_f16")?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(gate_packed), 0);
        encoder.set_buffer(2, Some(gate_scales), 0);
        encoder.set_buffer(3, Some(up_packed), 0);
        encoder.set_buffer(4, Some(up_scales), 0);
        encoder.set_buffer(5, Some(&output.buffer), 0);
        set_bytes(&encoder, 6, &input_columns);
        set_bytes(&encoder, 7, &output_columns);
        set_bytes(&encoder, 8, &gate_group_size);
        set_bytes(&encoder, 9, &up_group_size);
        set_bytes(&encoder, 10, &gate_scale_dtype);
        set_bytes(&encoder, 11, &up_scale_dtype);
        set_bytes(&encoder, 12, &params.kind);
        set_bytes(&encoder, 13, &params.alpha);
        set_bytes(&encoder, 14, &params.limit);
        set_bytes(&encoder, 15, &input_bf16);
        let input_rows = validate_u32("W4A16 gated input rows", input.rows)?;
        set_bytes(&encoder, 16, &input_rows);
        encoder.dispatch_thread_groups(MTLSize::new(gate_rows as u64, input.rows.div_ceil(8) as u64, 1), MTLSize::new(64, 1, 1));
        encoder.end_encoding();
        let shape = format!("input=[{},{}],output={gate_rows},groups={gate_group_size}/{up_group_size}", input.rows, input.cols);
        ctx.commit_and_wait_profiled(&command, "w4a16_gated_matmul_f16", &shape, input.buffer.length() + gate_packed.length() + gate_scales.length() + up_packed.length() + up_scales.length(), output.buffer.length());
        return Ok(output);
    }
    validate_w4a16_storage("resident W4A16 up", up_packed, up_scales, up_scale_dtype, up_group_size, up_rows, up_cols)?;

    let params = GatedActivation::from_spec(activation)?;
    let input_columns = validate_u32("W4A16 gated input columns", gate_cols)?;
    let output_columns = validate_u32("W4A16 gated output columns", gate_rows)?;
    let rows_per_group = if gate_cols.div_ceil(gate_group_size) <= 256 && up_cols.div_ceil(up_group_size) <= 256 { 2 } else { 1 };
    let gate_group_size = validate_u32("W4A16 gated gate group size", gate_group_size)?;
    let up_group_size = validate_u32("W4A16 gated up group size", up_group_size)?;
    let input_bf16 = u32::from(input.dtype == super::MetalTensorDType::Bf16);
    let output = if input_bf16 != 0 { ctx.tensor_zeros_bf16(1, gate_rows) } else { ctx.tensor_zeros(1, gate_rows) };
    let pipeline = ctx.pipeline("w4a16_gated_gemv_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("W4A16 gated GEMV 需要至少 64 threads/threadgroup".to_owned());
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
    set_bytes(&encoder, 6, &input_columns);
    set_bytes(&encoder, 7, &output_columns);
    set_bytes(&encoder, 8, &gate_group_size);
    set_bytes(&encoder, 9, &up_group_size);
    set_bytes(&encoder, 10, &gate_scale_dtype);
    set_bytes(&encoder, 11, &up_scale_dtype);
    set_bytes(&encoder, 12, &params.kind);
    set_bytes(&encoder, 13, &params.alpha);
    set_bytes(&encoder, 14, &params.limit);
    set_bytes(&encoder, 15, &input_bf16);
    encoder.dispatch_thread_groups(MTLSize::new(gate_rows.div_ceil(rows_per_group) as u64, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{gate_cols}],output={gate_rows},groups={gate_group_size}/{up_group_size}");
    ctx.commit_and_wait_profiled(&command, "w4a16_gated_gemv_f16", &shape, input.buffer.length() + gate_packed.length() + gate_scales.length() + up_packed.length() + up_scales.length(), output.buffer.length());
    Ok(output)
}

/// Decode 同一 input 的两个 W4A16 投影；允许输出宽度不同，供 GQA query/key 等独立分支复用。
#[allow(clippy::too_many_arguments)]
pub fn w4a16_dual_gemv_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    first_packed: &metal::Buffer,
    first_scales: &metal::Buffer,
    first_scale_dtype: u32,
    first_group_size: usize,
    first_rows: usize,
    first_cols: usize,
    second_packed: &metal::Buffer,
    second_scales: &metal::Buffer,
    second_scale_dtype: u32,
    second_group_size: usize,
    second_rows: usize,
    second_cols: usize,
) -> Result<(MetalTensor, MetalTensor), String> {
    if input.rows != 1 || input.cols != first_cols || first_cols != second_cols {
        return Err(format!("W4A16 dual GEMV shape 不兼容: input=[{},{}], first=[{first_rows},{first_cols}], second=[{second_rows},{second_cols}]", input.rows, input.cols,));
    }
    validate_w4a16_storage("resident W4A16 first", first_packed, first_scales, first_scale_dtype, first_group_size, first_rows, first_cols)?;
    validate_w4a16_storage("resident W4A16 second", second_packed, second_scales, second_scale_dtype, second_group_size, second_rows, second_cols)?;
    let input_columns = validate_u32("W4A16 dual input columns", first_cols)?;
    let first_output_columns = validate_u32("W4A16 dual first output columns", first_rows)?;
    let second_output_columns = validate_u32("W4A16 dual second output columns", second_rows)?;
    let first_group_size = validate_u32("W4A16 dual first group size", first_group_size)?;
    let second_group_size = validate_u32("W4A16 dual second group size", second_group_size)?;
    let input_bf16 = u32::from(input.dtype == super::MetalTensorDType::Bf16);
    let first_output = if input_bf16 != 0 { ctx.tensor_zeros_bf16(1, first_rows) } else { ctx.tensor_zeros(1, first_rows) };
    let second_output = if input_bf16 != 0 { ctx.tensor_zeros_bf16(1, second_rows) } else { ctx.tensor_zeros(1, second_rows) };
    let pipeline = ctx.pipeline("w4a16_dual_gemv_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("W4A16 dual GEMV 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(first_packed), 0);
    encoder.set_buffer(2, Some(first_scales), 0);
    encoder.set_buffer(3, Some(second_packed), 0);
    encoder.set_buffer(4, Some(second_scales), 0);
    encoder.set_buffer(5, Some(&first_output.buffer), 0);
    encoder.set_buffer(6, Some(&second_output.buffer), 0);
    set_bytes(&encoder, 7, &input_columns);
    set_bytes(&encoder, 8, &first_output_columns);
    set_bytes(&encoder, 9, &second_output_columns);
    set_bytes(&encoder, 10, &first_group_size);
    set_bytes(&encoder, 11, &second_group_size);
    set_bytes(&encoder, 12, &first_scale_dtype);
    set_bytes(&encoder, 13, &second_scale_dtype);
    set_bytes(&encoder, 14, &input_bf16);
    encoder.dispatch_thread_groups(MTLSize::new(first_rows.max(second_rows) as u64, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{first_cols}],outputs={first_rows}/{second_rows},groups={first_group_size}/{second_group_size}");
    ctx.commit_and_wait_profiled(
        &command,
        "w4a16_dual_gemv_f16",
        &shape,
        input.buffer.length() + first_packed.length() + first_scales.length() + second_packed.length() + second_scales.length(),
        first_output.buffer.length() + second_output.buffer.length(),
    );
    Ok((first_output, second_output))
}

#[cfg(test)]
mod w4a16_gated_tests {
    use super::*;
    use crate::kernel::metal::tensor::gated_activation_tensor;
    use half::bf16;

    #[test]
    fn gated_gemv_matches_unfused() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let rows = 3usize;
        let columns = 32usize;
        let group_size = 16usize;
        let packed_word = |code: u32| (0..8).fold(0, |word, index| word | (code << (index * 4)));
        let gate_codes = vec![packed_word(9); rows * columns.div_ceil(8)];
        let up_codes = vec![packed_word(10); rows * columns.div_ceil(8)];
        let scales = vec![1.0f32; rows * columns / group_size];
        let gate_packed = ctx.shared_buffer(as_bytes(&gate_codes));
        let up_packed = ctx.shared_buffer(as_bytes(&up_codes));
        let gate_scales = ctx.shared_buffer(as_bytes(&scales));
        let up_scales = ctx.shared_buffer(as_bytes(&scales));
        let input = ctx.tensor_from_f32(&vec![0.01; columns], 1, columns).unwrap();
        let gate = w4a16_matmul_tensor_resident(&ctx, &input, &gate_packed, &gate_scales, 2, group_size, rows, columns).unwrap();
        let up = w4a16_matmul_tensor_resident(&ctx, &input, &up_packed, &up_scales, 2, group_size, rows, columns).unwrap();
        let expected = gated_activation_tensor(&ctx, &gate, &up, &Activation::GeluTanh).unwrap();
        let actual = w4a16_gated_gemv_tensor_resident(&ctx, &input, &gate_packed, &gate_scales, 2, group_size, rows, columns, &up_packed, &up_scales, 2, group_size, rows, columns, &Activation::GeluTanh).unwrap();
        for (actual, expected) in ctx.tensor_to_f32(&actual).into_iter().zip(ctx.tensor_to_f32(&expected)) {
            assert!((actual - expected).abs() <= 0.01, "actual={actual}, expected={expected}");
        }
    }

    #[test]
    fn gated_prefill_matches_unfused() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let input_rows = 11usize;
        let rows = 5usize;
        let columns = 32usize;
        let group_size = 16usize;
        let packed_word = |code: u32| (0..8).fold(0, |word, index| word | (code << (index * 4)));
        let gate_codes = vec![packed_word(9); rows * columns.div_ceil(8)];
        let up_codes = vec![packed_word(10); rows * columns.div_ceil(8)];
        let scales = vec![1.0f32; rows * columns / group_size];
        let gate_packed = ctx.shared_buffer(as_bytes(&gate_codes));
        let up_packed = ctx.shared_buffer(as_bytes(&up_codes));
        let gate_scales = ctx.shared_buffer(as_bytes(&scales));
        let up_scales = ctx.shared_buffer(as_bytes(&scales));
        let input = ctx.tensor_from_f32(&vec![0.01; input_rows * columns], input_rows, columns).unwrap();
        let gate = w4a16_matmul_tensor_resident(&ctx, &input, &gate_packed, &gate_scales, 2, group_size, rows, columns).unwrap();
        let up = w4a16_matmul_tensor_resident(&ctx, &input, &up_packed, &up_scales, 2, group_size, rows, columns).unwrap();
        let expected = gated_activation_tensor(&ctx, &gate, &up, &Activation::GeluTanh).unwrap();
        let actual = w4a16_gated_gemv_tensor_resident(&ctx, &input, &gate_packed, &gate_scales, 2, group_size, rows, columns, &up_packed, &up_scales, 2, group_size, rows, columns, &Activation::GeluTanh).unwrap();
        for (actual, expected) in ctx.tensor_to_f32(&actual).into_iter().zip(ctx.tensor_to_f32(&expected)) {
            assert!((actual - expected).abs() <= 0.01, "actual={actual}, expected={expected}");
        }
    }

    #[test]
    fn dual_gemv_with_different_output_rows_matches_unfused() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let columns = 32usize;
        let group_size = 16usize;
        let first_rows = 3usize;
        let second_rows = 2usize;
        let packed_word = |code: u32| (0..8).fold(0, |word, index| word | (code << (index * 4)));
        let first_codes = vec![packed_word(9); first_rows * columns.div_ceil(8)];
        let second_codes = vec![packed_word(10); second_rows * columns.div_ceil(8)];
        let first_scale_values = vec![1.0f32; first_rows * columns / group_size];
        let second_scale_values = vec![1.0f32; second_rows * columns / group_size];
        let first_packed = ctx.shared_buffer(as_bytes(&first_codes));
        let second_packed = ctx.shared_buffer(as_bytes(&second_codes));
        let first_scales = ctx.shared_buffer(as_bytes(&first_scale_values));
        let second_scales = ctx.shared_buffer(as_bytes(&second_scale_values));
        let input = ctx.tensor_from_f32(&vec![0.01; columns], 1, columns).unwrap();
        let expected_first = w4a16_matmul_tensor_resident(&ctx, &input, &first_packed, &first_scales, 2, group_size, first_rows, columns).unwrap();
        let expected_second = w4a16_matmul_tensor_resident(&ctx, &input, &second_packed, &second_scales, 2, group_size, second_rows, columns).unwrap();
        let (actual_first, actual_second) = w4a16_dual_gemv_tensor_resident(&ctx, &input, &first_packed, &first_scales, 2, group_size, first_rows, columns, &second_packed, &second_scales, 2, group_size, second_rows, columns).unwrap();
        assert_eq!(ctx.tensor_to_f32(&actual_first), ctx.tensor_to_f32(&expected_first));
        assert_eq!(ctx.tensor_to_f32(&actual_second), ctx.tensor_to_f32(&expected_second));
    }

    /// 把 W4A16 GEMV Metal 输出与 CPU `matvec_w4a16_matrix` (即 `decode_w4a16_matrix` 的 GEMV 包装)
    /// 对照,确保 packed/scales 解码语义一致。`w4a16_*_tests` 里其余的对比是 Metal-vs-Metal,
    /// 不能捕获 packed/scales 的字节序或 scale dtype 解释 bug。
    #[test]
    fn gemv_matches_cpu_oracle() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let rows = 5usize;
        // cols 选 5120(= Qwen3-VL hidden) 来覆盖 Qwen 真实尺寸 + group_size=32 倍数。
        let columns = 5120usize;
        let group_size = 32usize;
        // 构造 packed bytes:每 (column % 8) 写入不同 nibble 值,避免常值打包掩盖字节序 bug。
        let packed = {
            let mut bytes = vec![0u8; rows * columns.div_ceil(8) * 4];
            for row in 0..rows {
                for column in 0..columns {
                    let nibble = ((row * 17 + column * 31) % 16) as u8;
                    let word_offset = (row * columns.div_ceil(8) + column / 8) * 4;
                    let mut word = u32::from_le_bytes(bytes[word_offset..word_offset + 4].try_into().unwrap());
                    word |= u32::from(nibble) << ((column % 8) * 4);
                    bytes[word_offset..word_offset + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
            bytes
        };
        // scales 选 F32(避免 BF16/F16 编码差异),按 group 变化幅度。
        let scale_values: Vec<f32> = (0..(rows * columns / group_size)).map(|i| 0.005 + 0.002 * ((i % 9) as f32)).collect();
        let scales = as_bytes(&scale_values).to_vec();
        let input_values: Vec<f32> = (0..columns).map(|i| i as f32 * 0.0007 - 1.7).collect();
        // CPU oracle
        let mut expected = vec![0.0f32; rows];
        crate::kernel::cpu::w4a16::matvec_w4a16_matrix(&packed, &scales, crate::weight::format::quantization::ScaleDType::F32, group_size, rows, columns, &input_values, &mut expected).unwrap();
        // Metal path:input rows == 1 走 w4a16_gemv_f16 分支。
        let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let packed_buf = ctx.shared_buffer(&packed);
        let scales_buf = ctx.shared_buffer(&scales);
        let actual = w4a16_matmul_tensor_resident(&ctx, &input, &packed_buf, &scales_buf, 2, group_size, rows, columns).unwrap();
        let actual_values = ctx.tensor_to_f32(&actual);
        for (row, (actual, expected)) in actual_values.iter().zip(expected.iter()).enumerate() {
            let tolerance = 0.5 + expected.abs() * 0.05;
            assert!(
                (actual - expected).abs() <= tolerance,
                "row={row} actual={actual} expected={expected} max_abs_input={} scale_max={}",
                input_values.iter().fold(0.0f32, |m, v| m.max(v.abs())),
                scale_values.iter().fold(0.0f32, |m, v| m.max(v.abs()))
            );
        }
    }

    /// 覆盖 BF16 scale dtype(Qwen3-VL 真实使用):Metal GEMV vs CPU oracle。
    #[test]
    fn gemv_bf16_scales_matches_cpu_oracle() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let rows = 3usize;
        let columns = 5120usize;
        let group_size = 32usize;
        let packed = {
            let mut bytes = vec![0u8; rows * columns.div_ceil(8) * 4];
            for row in 0..rows {
                for column in 0..columns {
                    let nibble = ((row * 13 + column * 7 + 5) % 16) as u8;
                    let word_offset = (row * columns.div_ceil(8) + column / 8) * 4;
                    let mut word = u32::from_le_bytes(bytes[word_offset..word_offset + 4].try_into().unwrap());
                    word |= u32::from(nibble) << ((column % 8) * 4);
                    bytes[word_offset..word_offset + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
            bytes
        };
        // BF16 scales
        let scale_values: Vec<f32> = (0..(rows * columns / group_size)).map(|i| 0.01 + 0.003 * ((i % 11) as f32)).collect();
        let scales: Vec<u8> = scale_values
            .iter()
            .flat_map(|value| {
                let bf = bf16::from_f32(*value);
                bf.to_le_bytes().to_vec()
            })
            .collect();
        let input_values: Vec<f32> = (0..columns).map(|i| 0.4 - 0.0007 * (i as f32)).collect();
        let mut expected = vec![0.0f32; rows];
        crate::kernel::cpu::w4a16::matvec_w4a16_matrix(&packed, &scales, crate::weight::format::quantization::ScaleDType::Bf16, group_size, rows, columns, &input_values, &mut expected).unwrap();
        let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let packed_buf = ctx.shared_buffer(&packed);
        let scales_buf = ctx.shared_buffer(&scales);
        let actual = w4a16_matmul_tensor_resident(&ctx, &input, &packed_buf, &scales_buf, 0, group_size, rows, columns).unwrap();
        let actual_values = ctx.tensor_to_f32(&actual);
        for (row, (actual, expected)) in actual_values.iter().zip(expected.iter()).enumerate() {
            let tolerance = 0.5 + expected.abs() * 0.1;
            assert!((actual - expected).abs() <= tolerance, "row={row} actual={actual} expected={expected}");
        }
    }

    /// 覆盖 BF16 input + BF16 scale(Qwen3-VL hidden 真实 dtype):Metal matmul vs CPU oracle。
    /// 这一路径对应 input 是上一次 layer 输出 + 走 BF16 量化路径。
    #[test]
    fn matmul_bf16_input_matches_cpu_oracle() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let input_rows = 4usize;
        let rows = 5usize;
        let columns = 5120usize;
        let group_size = 32usize;
        let packed = {
            let mut bytes = vec![0u8; rows * columns.div_ceil(8) * 4];
            for row in 0..rows {
                for column in 0..columns {
                    let nibble = ((row * 17 + column * 31) % 16) as u8;
                    let word_offset = (row * columns.div_ceil(8) + column / 8) * 4;
                    let mut word = u32::from_le_bytes(bytes[word_offset..word_offset + 4].try_into().unwrap());
                    word |= u32::from(nibble) << ((column % 8) * 4);
                    bytes[word_offset..word_offset + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
            bytes
        };
        let scale_values: Vec<f32> = (0..(rows * columns / group_size)).map(|i| 0.008 + 0.002 * ((i % 7) as f32)).collect();
        let scales: Vec<u8> = scale_values
            .iter()
            .flat_map(|value| {
                let bf = bf16::from_f32(*value);
                bf.to_le_bytes().to_vec()
            })
            .collect();
        let input_values: Vec<f32> = (0..input_rows * columns).map(|i| 0.5 - 0.0009 * (i as f32)).collect();
        // CPU oracle:对每个 input_row 单独 GEMV
        let mut expected = vec![0.0f32; input_rows * rows];
        for n in 0..input_rows {
            let mut row_out = vec![0.0f32; rows];
            crate::kernel::cpu::w4a16::matvec_w4a16_matrix(&packed, &scales, crate::weight::format::quantization::ScaleDType::Bf16, group_size, rows, columns, &input_values[n * columns..(n + 1) * columns], &mut row_out).unwrap();
            expected[n * rows..(n + 1) * rows].copy_from_slice(&row_out);
        }
        // Metal path:input rows > 1 走 w4a16_matmul_f16 kernel;为了让 input 用 BF16,直接构造 BF16 tensor。
        let input = ctx.tensor_from_f32_bf16(&input_values, input_rows, columns).unwrap();
        let packed_buf = ctx.shared_buffer(&packed);
        let scales_buf = ctx.shared_buffer(&scales);
        let actual = w4a16_matmul_tensor_resident(&ctx, &input, &packed_buf, &scales_buf, 0, group_size, rows, columns).unwrap();
        let actual_values = ctx.tensor_to_f32(&actual);
        for (idx, (actual, expected)) in actual_values.iter().zip(expected.iter()).enumerate() {
            let tolerance = 0.5 + expected.abs() * 0.1;
            assert!((actual - expected).abs() <= tolerance, "idx={idx} actual={actual} expected={expected}");
        }
    }
}
