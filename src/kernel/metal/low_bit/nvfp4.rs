//! ModelOpt NVFP4(E2M1 + block scale)保持压缩态的 GEMV/GEMM 与 top-k 专家 decode。

/// 本家族的 Metal shader。家族私有 helper 先于 kernel 定义;
/// 跨家族共享 helper 见 `super::super::preamble`,由 `kernels_source()` 统一拼接。
pub const NVFP4_SHADERS: &str = r#"
constant float NVFP4_E2M1_VALUES[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
inline float decode_f4_e2m1(uchar bits) {
    return NVFP4_E2M1_VALUES[bits & 0x0f];
}
kernel void nvfp4_dequant_matrix_f16(
    device const uchar *codes [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device const float *global_scale [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &count [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint row = gid / columns;
    if (row >= rows) return;
    const uint column = gid - row * columns;
    const uchar packed = codes[gid >> 1];
    const uchar code = (gid & 1) == 0 ? packed & 0x0f : packed >> 4;
    const uchar scale_code = scales[ulong(row) * (columns / 16) + column / 16];
    output[gid] = finite_f16(decode_f4_e2m1(code) * decode_f8_e4m3(scale_code) * global_scale[0]);
}
kernel void nvfp4_gemv_f16(
    device const half *input [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const float *global_scale [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &input_columns [[buffer(5)]],
    constant uint &output_columns [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    const uint row = group.x;
    const uint input_row = group.y;
    const uint width = group_width.x;
    if (row >= output_columns) return;
    threadgroup float simd_sums[8];
    const ulong weight_base = ulong(row) * input_columns;
    const ulong scale_base = ulong(row) * (input_columns / 16);
    const uint block_count = input_columns / 16;
    const float tensor_scale = global_scale[0];
    device const half2 *input_pairs = (device const half2 *)(input + ulong(input_row) * input_columns);
    float sum = 0.0f;
    for (uint block = lane; block < block_count; block += width) {
        const uint column_base = block * 16;
        const float scale = decode_f8_e4m3(scales[scale_base + block]) * tensor_scale;
        const ulong code_base = (weight_base + column_base) >> 1;
        const uint input_pair_base = column_base >> 1;
        for (uint pair = 0; pair < 8; ++pair) {
            const uchar packed = codes[code_base + pair];
            const half2 values = input_pairs[input_pair_base + pair];
            sum += (float(values.x) * decode_f4_e2m1(packed & 0x0f)
                + float(values.y) * decode_f4_e2m1(packed >> 4)) * scale;
        }
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) simd_sums[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const uint simd_count = width >> 5;
        float total = simd_lane < simd_count ? simd_sums[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) output[ulong(input_row) * output_columns + row] = finite_f16(total);
    }
}
kernel void nvfp4_dual_gemv_f16(
    device const half *input [[buffer(0)]],
    device const uchar *first_codes [[buffer(1)]],
    device const uchar *first_scales [[buffer(2)]],
    device const float *first_global_scale [[buffer(3)]],
    device const uchar *second_codes [[buffer(4)]],
    device const uchar *second_scales [[buffer(5)]],
    device const float *second_global_scale [[buffer(6)]],
    device half *first_output [[buffer(7)]],
    device half *second_output [[buffer(8)]],
    constant uint &input_columns [[buffer(9)]],
    constant uint &output_columns [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    const uint row = group.x;
    const uint input_row = group.y;
    const uint width = group_width.x;
    if (row >= output_columns) return;
    threadgroup float first_simd_sums[8];
    threadgroup float second_simd_sums[8];
    const ulong weight_base = ulong(row) * input_columns;
    const ulong scale_base = ulong(row) * (input_columns / 16);
    const uint block_count = input_columns / 16;
    const float first_tensor_scale = first_global_scale[0];
    const float second_tensor_scale = second_global_scale[0];
    device const half2 *input_pairs = (device const half2 *)(input + ulong(input_row) * input_columns);
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    for (uint block = lane; block < block_count; block += width) {
        const uint column_base = block * 16;
        const ulong code_base = (weight_base + column_base) >> 1;
        const uint input_pair_base = column_base >> 1;
        const float first_scale = decode_f8_e4m3(first_scales[scale_base + block]) * first_tensor_scale;
        const float second_scale = decode_f8_e4m3(second_scales[scale_base + block]) * second_tensor_scale;
        for (uint pair = 0; pair < 8; ++pair) {
            const half2 values = input_pairs[input_pair_base + pair];
            const uchar first_packed = first_codes[code_base + pair];
            const uchar second_packed = second_codes[code_base + pair];
            first_sum += (float(values.x) * decode_f4_e2m1(first_packed & 0x0f)
                + float(values.y) * decode_f4_e2m1(first_packed >> 4)) * first_scale;
            second_sum += (float(values.x) * decode_f4_e2m1(second_packed & 0x0f)
                + float(values.y) * decode_f4_e2m1(second_packed >> 4)) * second_scale;
        }
    }
    first_sum = simd_sum(first_sum);
    second_sum = simd_sum(second_sum);
    if (simd_lane == 0) {
        first_simd_sums[simd_group] = first_sum;
        second_simd_sums[simd_group] = second_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const uint simd_count = width >> 5;
        float first_total = simd_lane < simd_count ? first_simd_sums[simd_lane] : 0.0f;
        float second_total = simd_lane < simd_count ? second_simd_sums[simd_lane] : 0.0f;
        first_total = simd_sum(first_total);
        second_total = simd_sum(second_total);
        if (simd_lane == 0) {
            const ulong output_index = ulong(input_row) * output_columns + row;
            first_output[output_index] = finite_f16(first_total);
            second_output[output_index] = finite_f16(second_total);
        }
    }
}
kernel void nvfp4_grouped_expert_gate_up_f16(
    device const half *input [[buffer(0)]],
    device const uchar *first_chunk [[buffer(1)]],
    device const uchar *second_chunk [[buffer(2)]],
    device half *activated [[buffer(3)]],
    constant uint &hidden [[buffer(4)]],
    constant uint &intermediate [[buffer(5)]],
    constant uint &active_experts [[buffer(6)]],
    constant uint &experts_per_chunk [[buffer(7)]],
    constant ulong &matrix_stride [[buffer(8)]],
    constant ulong &code_stride [[buffer(9)]],
    constant ulong &scale_stride [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &activation_alpha [[buffer(12)]],
    constant float &activation_limit [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    const uint output_column = group.x;
    const uint expert = group.y;
    const uint width = group_width.x;
    if (output_column >= intermediate || expert >= active_experts) return;
    threadgroup float gate_simd_sums[8];
    threadgroup float up_simd_sums[8];
    const bool second = expert >= experts_per_chunk;
    const uint local_expert = second ? expert - experts_per_chunk : expert;
    device const uchar *chunk = second ? second_chunk : first_chunk;
    const ulong expert_base = ulong(local_expert) * matrix_stride * 3ul;
    const ulong gate_base = expert_base;
    const ulong up_base = expert_base + matrix_stride;
    device const uchar *gate_codes = chunk + gate_base;
    device const uchar *gate_scales = gate_codes + code_stride;
    device const float *gate_global = (device const float *)(gate_scales + scale_stride);
    device const uchar *up_codes = chunk + up_base;
    device const uchar *up_scales = up_codes + code_stride;
    device const float *up_global = (device const float *)(up_scales + scale_stride);
    const ulong weight_base = ulong(output_column) * hidden;
    const ulong scale_base = ulong(output_column) * (hidden / 16);
    const uint block_count = hidden / 16;
    device const half2 *input_pairs = (device const half2 *)input;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint block = lane; block < block_count; block += width) {
        const uint column_base = block * 16;
        const ulong code_base = (weight_base + column_base) >> 1;
        const uint input_pair_base = column_base >> 1;
        const float gate_scale = decode_f8_e4m3(gate_scales[scale_base + block]) * gate_global[0];
        const float up_scale = decode_f8_e4m3(up_scales[scale_base + block]) * up_global[0];
        for (uint pair = 0; pair < 8; ++pair) {
            const half2 values = input_pairs[input_pair_base + pair];
            const uchar gate_packed = gate_codes[code_base + pair];
            const uchar up_packed = up_codes[code_base + pair];
            gate_sum += (float(values.x) * decode_f4_e2m1(gate_packed & 0x0f) + float(values.y) * decode_f4_e2m1(gate_packed >> 4)) * gate_scale;
            up_sum += (float(values.x) * decode_f4_e2m1(up_packed & 0x0f) + float(values.y) * decode_f4_e2m1(up_packed >> 4)) * up_scale;
        }
    }
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (simd_lane == 0) {
        gate_simd_sums[simd_group] = gate_sum;
        up_simd_sums[simd_group] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const uint simd_count = width >> 5;
        float gate_total = simd_lane < simd_count ? gate_simd_sums[simd_lane] : 0.0f;
        float up_total = simd_lane < simd_count ? up_simd_sums[simd_lane] : 0.0f;
        gate_total = simd_sum(gate_total);
        up_total = simd_sum(up_total);
        if (simd_lane == 0) {
            const half gate_value = finite_f16(gate_total);
            const half up_value = finite_f16(up_total);
            activated[ulong(expert) * intermediate + output_column] = finite_f16(gated_activation_value(
                float(gate_value), float(up_value), activation_kind, activation_alpha, activation_limit));
        }
    }
}
kernel void nvfp4_grouped_expert_down_f16(
    device const half *activated [[buffer(0)]],
    device const uchar *first_chunk [[buffer(1)]],
    device const uchar *second_chunk [[buffer(2)]],
    device const float *route_weights [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    constant uint &intermediate [[buffer(6)]],
    constant uint &active_experts [[buffer(7)]],
    constant uint &experts_per_chunk [[buffer(8)]],
    constant ulong &matrix_stride [[buffer(9)]],
    constant ulong &code_stride [[buffer(10)]],
    constant ulong &scale_stride [[buffer(11)]],
    uint output_column [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    if (output_column >= hidden) return;
    threadgroup float simd_sums[8];
    const uint simd_count = width >> 5;
    // MiniMax top-k=4，四个 SIMD group 各算一个专家，避免专家循环中的反复 barrier。
    if (active_experts <= simd_count) {
        float sum = 0.0f;
        if (simd_group < active_experts) {
            const uint expert = simd_group;
            const bool second = expert >= experts_per_chunk;
            const uint local_expert = second ? expert - experts_per_chunk : expert;
            device const uchar *chunk = second ? second_chunk : first_chunk;
            const ulong down_base = ulong(local_expert) * matrix_stride * 3ul + matrix_stride * 2ul;
            device const uchar *codes = chunk + down_base;
            device const uchar *scales = codes + code_stride;
            device const float *global_scale = (device const float *)(scales + scale_stride);
            const ulong weight_base = ulong(output_column) * intermediate;
            const ulong scale_base = ulong(output_column) * (intermediate / 16);
            const uint block_count = intermediate / 16;
            const float tensor_scale = global_scale[0];
            device const half2 *input_pairs = (device const half2 *)(activated + ulong(expert) * intermediate);
            for (uint block = simd_lane; block < block_count; block += 32) {
                const uint column_base = block * 16;
                const float scale = decode_f8_e4m3(scales[scale_base + block]) * tensor_scale;
                const ulong code_base = (weight_base + column_base) >> 1;
                const uint input_pair_base = column_base >> 1;
                for (uint pair = 0; pair < 8; ++pair) {
                    const uchar packed = codes[code_base + pair];
                    const half2 values = input_pairs[input_pair_base + pair];
                    sum += (float(values.x) * decode_f4_e2m1(packed & 0x0f) + float(values.y) * decode_f4_e2m1(packed >> 4)) * scale;
                }
            }
            sum = simd_sum(sum);
        }
        if (simd_lane == 0) {
            simd_sums[simd_group] = simd_group < active_experts
                ? route_weights[simd_group] * float(finite_f16(sum))
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            float total = simd_lane < active_experts ? simd_sums[simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) output[output_column] = finite_f16(total);
        }
        return;
    }
    threadgroup float routed_sum;
    if (lane == 0) routed_sum = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint expert = 0; expert < active_experts; ++expert) {
        const bool second = expert >= experts_per_chunk;
        const uint local_expert = second ? expert - experts_per_chunk : expert;
        device const uchar *chunk = second ? second_chunk : first_chunk;
        const ulong down_base = ulong(local_expert) * matrix_stride * 3ul + matrix_stride * 2ul;
        device const uchar *codes = chunk + down_base;
        device const uchar *scales = codes + code_stride;
        device const float *global_scale = (device const float *)(scales + scale_stride);
        const ulong weight_base = ulong(output_column) * intermediate;
        const ulong scale_base = ulong(output_column) * (intermediate / 16);
        const uint block_count = intermediate / 16;
        device const half2 *input_pairs = (device const half2 *)(activated + ulong(expert) * intermediate);
        float sum = 0.0f;
        for (uint block = lane; block < block_count; block += width) {
            const uint column_base = block * 16;
            const float scale = decode_f8_e4m3(scales[scale_base + block]) * global_scale[0];
            const ulong code_base = (weight_base + column_base) >> 1;
            const uint input_pair_base = column_base >> 1;
            for (uint pair = 0; pair < 8; ++pair) {
                const uchar packed = codes[code_base + pair];
                const half2 values = input_pairs[input_pair_base + pair];
                sum += (float(values.x) * decode_f4_e2m1(packed & 0x0f) + float(values.y) * decode_f4_e2m1(packed >> 4)) * scale;
            }
        }
        sum = simd_sum(sum);
        if (simd_lane == 0) simd_sums[simd_group] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            float total = simd_lane < simd_count ? simd_sums[simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) routed_sum += route_weights[expert] * float(finite_f16(total));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[output_column] = finite_f16(routed_sum);
}
kernel void nvfp4_grouped_matrix_gate_up_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_codes [[buffer(1)]],
    device const uchar *gate_scales [[buffer(2)]],
    device const float *gate_globals [[buffer(3)]],
    device const uchar *up_codes [[buffer(4)]],
    device const uchar *up_scales [[buffer(5)]],
    device const float *up_globals [[buffer(6)]],
    device half *activated [[buffer(7)]],
    constant uint &hidden [[buffer(8)]],
    constant uint &intermediate [[buffer(9)]],
    constant uint &active_experts [[buffer(10)]],
    constant ulong &code_stride [[buffer(11)]],
    constant ulong &scale_stride [[buffer(12)]],
    constant uint &activation_kind [[buffer(13)]],
    constant float &activation_alpha [[buffer(14)]],
    constant float &activation_limit [[buffer(15)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    const uint output_column = group.x;
    const uint expert = group.y;
    const uint width = group_width.x;
    if (output_column >= intermediate || expert >= active_experts) return;
    threadgroup float gate_simd_sums[8];
    threadgroup float up_simd_sums[8];
    const ulong expert_code_base = ulong(expert) * code_stride;
    const ulong expert_scale_base = ulong(expert) * scale_stride;
    const ulong weight_base = ulong(output_column) * hidden;
    const ulong scale_base = ulong(output_column) * (hidden / 16);
    const uint block_count = hidden / 16;
    device const half2 *input_pairs = (device const half2 *)input;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint block = lane; block < block_count; block += width) {
        const uint column_base = block * 16;
        const ulong code_base = expert_code_base + ((weight_base + column_base) >> 1);
        const ulong matrix_scale_base = expert_scale_base + scale_base + block;
        const uint input_pair_base = column_base >> 1;
        const float gate_scale = decode_f8_e4m3(gate_scales[matrix_scale_base]) * gate_globals[expert];
        const float up_scale = decode_f8_e4m3(up_scales[matrix_scale_base]) * up_globals[expert];
        for (uint pair = 0; pair < 8; ++pair) {
            const half2 values = input_pairs[input_pair_base + pair];
            const uchar gate_packed = gate_codes[code_base + pair];
            const uchar up_packed = up_codes[code_base + pair];
            gate_sum += (float(values.x) * decode_f4_e2m1(gate_packed & 0x0f) + float(values.y) * decode_f4_e2m1(gate_packed >> 4)) * gate_scale;
            up_sum += (float(values.x) * decode_f4_e2m1(up_packed & 0x0f) + float(values.y) * decode_f4_e2m1(up_packed >> 4)) * up_scale;
        }
    }
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (simd_lane == 0) {
        gate_simd_sums[simd_group] = gate_sum;
        up_simd_sums[simd_group] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const uint simd_count = width >> 5;
        float gate_total = simd_lane < simd_count ? gate_simd_sums[simd_lane] : 0.0f;
        float up_total = simd_lane < simd_count ? up_simd_sums[simd_lane] : 0.0f;
        gate_total = simd_sum(gate_total);
        up_total = simd_sum(up_total);
        if (simd_lane == 0) {
            const half gate_value = finite_f16(gate_total);
            const half up_value = finite_f16(up_total);
            activated[ulong(expert) * intermediate + output_column] = finite_f16(gated_activation_value(
                float(gate_value), float(up_value), activation_kind, activation_alpha, activation_limit));
        }
    }
}
kernel void nvfp4_grouped_matrix_down_f16(
    device const half *activated [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const float *globals [[buffer(3)]],
    device const float *route_weights [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &hidden [[buffer(6)]],
    constant uint &intermediate [[buffer(7)]],
    constant uint &active_experts [[buffer(8)]],
    constant ulong &code_stride [[buffer(9)]],
    constant ulong &scale_stride [[buffer(10)]],
    uint output_column [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    if (output_column >= hidden) return;
    threadgroup float simd_sums[8];
    const uint simd_count = width >> 5;
    // 常见 top-k 不超过 SIMD group 数时并行专家，保留每个专家先舍入 F16 的语义。
    if (active_experts <= simd_count) {
        const ulong weight_base = ulong(output_column) * intermediate;
        const ulong scale_base = ulong(output_column) * (intermediate / 16);
        const uint block_count = intermediate / 16;
        float sum = 0.0f;
        if (simd_group < active_experts) {
            const uint expert = simd_group;
            const ulong expert_code_base = ulong(expert) * code_stride;
            const ulong expert_scale_base = ulong(expert) * scale_stride;
            const float tensor_scale = globals[expert];
            device const half2 *input_pairs = (device const half2 *)(activated + ulong(expert) * intermediate);
            for (uint block = simd_lane; block < block_count; block += 32) {
                const uint column_base = block * 16;
                const float scale = decode_f8_e4m3(scales[expert_scale_base + scale_base + block]) * tensor_scale;
                const ulong code_base = expert_code_base + ((weight_base + column_base) >> 1);
                const uint input_pair_base = column_base >> 1;
                for (uint pair = 0; pair < 8; ++pair) {
                    const uchar packed = codes[code_base + pair];
                    const half2 values = input_pairs[input_pair_base + pair];
                    sum += (float(values.x) * decode_f4_e2m1(packed & 0x0f) + float(values.y) * decode_f4_e2m1(packed >> 4)) * scale;
                }
            }
            sum = simd_sum(sum);
        }
        if (simd_lane == 0) {
            simd_sums[simd_group] = simd_group < active_experts
                ? route_weights[simd_group] * float(finite_f16(sum))
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            float total = simd_lane < active_experts ? simd_sums[simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) output[output_column] = finite_f16(total);
        }
        return;
    }
    threadgroup float routed_sum;
    if (lane == 0) routed_sum = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint expert = 0; expert < active_experts; ++expert) {
        const ulong expert_code_base = ulong(expert) * code_stride;
        const ulong expert_scale_base = ulong(expert) * scale_stride;
        const ulong weight_base = ulong(output_column) * intermediate;
        const ulong scale_base = ulong(output_column) * (intermediate / 16);
        const uint block_count = intermediate / 16;
        device const half2 *input_pairs = (device const half2 *)(activated + ulong(expert) * intermediate);
        float sum = 0.0f;
        for (uint block = lane; block < block_count; block += width) {
            const uint column_base = block * 16;
            const float scale = decode_f8_e4m3(scales[expert_scale_base + scale_base + block]) * globals[expert];
            const ulong code_base = expert_code_base + ((weight_base + column_base) >> 1);
            const uint input_pair_base = column_base >> 1;
            for (uint pair = 0; pair < 8; ++pair) {
                const uchar packed = codes[code_base + pair];
                const half2 values = input_pairs[input_pair_base + pair];
                sum += (float(values.x) * decode_f4_e2m1(packed & 0x0f) + float(values.y) * decode_f4_e2m1(packed >> 4)) * scale;
            }
        }
        sum = simd_sum(sum);
        if (simd_lane == 0) simd_sums[simd_group] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            float total = simd_lane < simd_count ? simd_sums[simd_lane] : 0.0f;
            total = simd_sum(total);
            if (simd_lane == 0) routed_sum += route_weights[expert] * float(finite_f16(total));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[output_column] = finite_f16(routed_sum);
}
"#;

use crate::backend::metal::api as metal;

use super::super::dense::GatedActivation;
use super::super::fp8::validate_buffer_region;
use super::super::{Activation, MTLSize, MetalContext, MetalTensor, THREADS, as_bytes, f16, mem, nvfp4_direct_rows, set_bytes, validate_u32};

/// ModelOpt NVFP4 W4A16：权重保持 E2M1 压缩态，激活沿用 zLLM F16 tensor。
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_matmul_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    codes: &metal::Buffer,
    codes_offset: usize,
    scales: &metal::Buffer,
    scales_offset: usize,
    global_scale: &metal::Buffer,
    global_scale_offset: usize,
    weight_rows: usize,
    weight_cols: usize,
) -> Result<MetalTensor, String> {
    if input.cols != weight_cols || weight_cols == 0 || !weight_cols.is_multiple_of(16) {
        return Err(format!("NVFP4 input=[{},{}] 与 weight=[{weight_rows},{weight_cols}] 不兼容", input.rows, input.cols));
    }
    let elements = weight_rows.checked_mul(weight_cols).ok_or("NVFP4 weight 大小溢出")?;
    let code_bytes = elements / 2;
    let scale_bytes = weight_rows * (weight_cols / 16);
    let global_scale_bytes = mem::size_of::<f32>();
    validate_buffer_region("NVFP4 codes", codes_offset, code_bytes, codes.length() as usize)?;
    validate_buffer_region("NVFP4 scales", scales_offset, scale_bytes, scales.length() as usize)?;
    validate_buffer_region("NVFP4 global scale", global_scale_offset, global_scale_bytes, global_scale.length() as usize)?;
    let rows = validate_u32("NVFP4 weight rows", weight_rows)?;
    let columns = validate_u32("NVFP4 weight columns", weight_cols)?;
    let output = ctx.tensor_zeros(input.rows, weight_rows);

    if input.rows == 1 {
        const NVFP4_THREADS: usize = 128;
        let pipeline = ctx.pipeline("nvfp4_gemv_f16")?;
        if pipeline.max_total_threads_per_threadgroup() < NVFP4_THREADS as u64 {
            return Err(format!("NVFP4 GEMV 需要至少 {NVFP4_THREADS} threads/threadgroup"));
        }
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(codes), codes_offset as u64);
        encoder.set_buffer(2, Some(scales), scales_offset as u64);
        encoder.set_buffer(3, Some(global_scale), global_scale_offset as u64);
        encoder.set_buffer(4, Some(&output.buffer), 0);
        set_bytes(&encoder, 5, &columns);
        set_bytes(&encoder, 6, &rows);
        encoder.dispatch_thread_groups(MTLSize::new(weight_rows as u64, input.rows as u64, 1), MTLSize::new(NVFP4_THREADS as u64, 1, 1));
        encoder.end_encoding();
        let shape = format!("input=[{},{weight_cols}],weight=[{weight_rows},{weight_cols}]", input.rows);
        ctx.commit_and_wait_profiled(&command, "nvfp4_direct_f16", &shape, input.buffer.length() + code_bytes as u64 + scale_bytes as u64 + global_scale_bytes as u64, output.buffer.length());
        return Ok(output);
    }

    let count = validate_u32("NVFP4 weight elements", elements)?;
    let materialized = ctx.shared_buffer_zeros(elements * mem::size_of::<f16>());
    let pipeline = ctx.pipeline("nvfp4_dequant_matrix_f16")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(codes), codes_offset as u64);
    encoder.set_buffer(1, Some(scales), scales_offset as u64);
    encoder.set_buffer(2, Some(global_scale), global_scale_offset as u64);
    encoder.set_buffer(3, Some(&materialized), 0);
    set_bytes(&encoder, 4, &rows);
    set_bytes(&encoder, 5, &columns);
    set_bytes(&encoder, 6, &count);
    let threads = elements.min(THREADS);
    encoder.dispatch_thread_groups(MTLSize::new(elements.div_ceil(threads) as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{},{}],weight=[{weight_rows},{weight_cols}]", input.rows, input.cols);
    ctx.commit_and_wait_profiled(&command, "nvfp4_dequantize_f16", &shape, code_bytes as u64 + scale_bytes as u64 + global_scale_bytes as u64, materialized.length());
    super::super::dense::launch_matmul_f16(ctx, &input.buffer, &materialized, &output.buffer, input.rows, input.cols, weight_rows)?;
    Ok(output)
}

/// 两个同形 NVFP4 decode GEMV 共用一次 input 读取和一次 Metal dispatch。
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_dual_matmul_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    first_codes: &metal::Buffer,
    first_codes_offset: usize,
    first_scales: &metal::Buffer,
    first_scales_offset: usize,
    first_global_scale: &metal::Buffer,
    first_global_scale_offset: usize,
    first_rows: usize,
    first_cols: usize,
    second_codes: &metal::Buffer,
    second_codes_offset: usize,
    second_scales: &metal::Buffer,
    second_scales_offset: usize,
    second_global_scale: &metal::Buffer,
    second_global_scale_offset: usize,
    second_rows: usize,
    second_cols: usize,
) -> Result<(MetalTensor, MetalTensor), String> {
    if input.rows > nvfp4_direct_rows() || first_rows != second_rows || first_cols != second_cols {
        return Ok((
            nvfp4_matmul_tensor_resident(ctx, input, first_codes, first_codes_offset, first_scales, first_scales_offset, first_global_scale, first_global_scale_offset, first_rows, first_cols)?,
            nvfp4_matmul_tensor_resident(ctx, input, second_codes, second_codes_offset, second_scales, second_scales_offset, second_global_scale, second_global_scale_offset, second_rows, second_cols)?,
        ));
    }
    if input.cols != first_cols || first_cols == 0 || !first_cols.is_multiple_of(16) {
        return Err(format!("NVFP4 dual input=[{},{}] 与 weight=[{first_rows},{first_cols}] 不兼容", input.rows, input.cols));
    }
    let elements = first_rows.checked_mul(first_cols).ok_or("NVFP4 dual weight 大小溢出")?;
    let code_bytes = elements / 2;
    let scale_bytes = first_rows * (first_cols / 16);
    let global_scale_bytes = mem::size_of::<f32>();
    validate_buffer_region("NVFP4 first codes", first_codes_offset, code_bytes, first_codes.length() as usize)?;
    validate_buffer_region("NVFP4 first scales", first_scales_offset, scale_bytes, first_scales.length() as usize)?;
    validate_buffer_region("NVFP4 first global scale", first_global_scale_offset, global_scale_bytes, first_global_scale.length() as usize)?;
    validate_buffer_region("NVFP4 second codes", second_codes_offset, code_bytes, second_codes.length() as usize)?;
    validate_buffer_region("NVFP4 second scales", second_scales_offset, scale_bytes, second_scales.length() as usize)?;
    validate_buffer_region("NVFP4 second global scale", second_global_scale_offset, global_scale_bytes, second_global_scale.length() as usize)?;

    const NVFP4_THREADS: usize = 128;
    let rows = validate_u32("NVFP4 dual rows", first_rows)?;
    let columns = validate_u32("NVFP4 dual columns", first_cols)?;
    let first_output = ctx.tensor_zeros(input.rows, first_rows);
    let second_output = ctx.tensor_zeros(input.rows, second_rows);
    let pipeline = ctx.pipeline("nvfp4_dual_gemv_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < NVFP4_THREADS as u64 {
        return Err(format!("NVFP4 dual GEMV 需要至少 {NVFP4_THREADS} threads/threadgroup"));
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(first_codes), first_codes_offset as u64);
    encoder.set_buffer(2, Some(first_scales), first_scales_offset as u64);
    encoder.set_buffer(3, Some(first_global_scale), first_global_scale_offset as u64);
    encoder.set_buffer(4, Some(second_codes), second_codes_offset as u64);
    encoder.set_buffer(5, Some(second_scales), second_scales_offset as u64);
    encoder.set_buffer(6, Some(second_global_scale), second_global_scale_offset as u64);
    encoder.set_buffer(7, Some(&first_output.buffer), 0);
    encoder.set_buffer(8, Some(&second_output.buffer), 0);
    set_bytes(&encoder, 9, &columns);
    set_bytes(&encoder, 10, &rows);
    encoder.dispatch_thread_groups(MTLSize::new(first_rows as u64, input.rows as u64, 1), MTLSize::new(NVFP4_THREADS as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{},{first_cols}],weights=2x[{first_rows},{first_cols}]", input.rows);
    ctx.commit_and_wait_profiled(&command, "nvfp4_dual_gemv_f16", &shape, input.buffer.length() + 2 * (code_bytes + scale_bytes + global_scale_bytes) as u64, first_output.buffer.length() + second_output.buffer.length());
    Ok((first_output, second_output))
}

/// NVFP4 top-k 专家 decode 局部融合；不改变层级同步和 archive 生命周期。
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_experts_decode(
    ctx: &MetalContext,
    input: &MetalTensor,
    first_chunk: &metal::Buffer,
    second_chunk: &metal::Buffer,
    active_experts: usize,
    experts_per_chunk: usize,
    code_stride: usize,
    scale_stride: usize,
    hidden: usize,
    intermediate: usize,
    route_weights: &[f32],
    activation: &Activation,
) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != hidden || active_experts == 0 || active_experts > 8 || route_weights.len() != active_experts {
        return Err(format!("NVFP4 grouped decode shape 非法: input=[{},{}] active={active_experts} weights={}", input.rows, input.cols, route_weights.len()));
    }
    if experts_per_chunk == 0 || !hidden.is_multiple_of(16) || !intermediate.is_multiple_of(16) {
        return Err("NVFP4 grouped decode stride/维度非法".to_owned());
    }
    const NVFP4_GATE_UP_THREADS: usize = 128;
    const NVFP4_DOWN_THREADS: usize = 128;
    let activation = GatedActivation::from_spec(activation)?;
    let hidden_u32 = validate_u32("NVFP4 grouped hidden", hidden)?;
    let intermediate_u32 = validate_u32("NVFP4 grouped intermediate", intermediate)?;
    let active_u32 = validate_u32("NVFP4 grouped active", active_experts)?;
    let experts_per_chunk_u32 = validate_u32("NVFP4 grouped experts/chunk", experts_per_chunk)?;
    let code_stride_u64 = code_stride as u64;
    let scale_stride_u64 = scale_stride as u64;
    let matrix_stride_u64 = code_stride_u64 + scale_stride_u64 + mem::size_of::<f32>() as u64;
    let route_weights = ctx.shared_buffer(as_bytes(route_weights));
    let activated = ctx.shared_buffer_zeros(active_experts * intermediate * mem::size_of::<f16>());
    let output = ctx.tensor_zeros(1, hidden);
    let gate_up = ctx.pipeline("nvfp4_grouped_expert_gate_up_f16")?;
    let down = ctx.pipeline("nvfp4_grouped_expert_down_f16")?;
    if gate_up.max_total_threads_per_threadgroup() < NVFP4_GATE_UP_THREADS as u64 || down.max_total_threads_per_threadgroup() < NVFP4_DOWN_THREADS as u64 {
        return Err("NVFP4 grouped decode threadgroup 上限不足".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&gate_up);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(first_chunk), 0);
    encoder.set_buffer(2, Some(second_chunk), 0);
    encoder.set_buffer(3, Some(&activated), 0);
    set_bytes(&encoder, 4, &hidden_u32);
    set_bytes(&encoder, 5, &intermediate_u32);
    set_bytes(&encoder, 6, &active_u32);
    set_bytes(&encoder, 7, &experts_per_chunk_u32);
    set_bytes(&encoder, 8, &matrix_stride_u64);
    set_bytes(&encoder, 9, &code_stride_u64);
    set_bytes(&encoder, 10, &scale_stride_u64);
    set_bytes(&encoder, 11, &activation.kind);
    set_bytes(&encoder, 12, &activation.alpha);
    set_bytes(&encoder, 13, &activation.limit);
    encoder.dispatch_thread_groups(MTLSize::new(intermediate as u64, active_experts as u64, 1), MTLSize::new(NVFP4_GATE_UP_THREADS as u64, 1, 1));
    encoder.memory_barrier_with_resources(&[activated.as_ref()]);
    encoder.set_compute_pipeline_state(&down);
    encoder.set_buffer(0, Some(&activated), 0);
    encoder.set_buffer(1, Some(first_chunk), 0);
    encoder.set_buffer(2, Some(second_chunk), 0);
    encoder.set_buffer(3, Some(&route_weights), 0);
    encoder.set_buffer(4, Some(&output.buffer), 0);
    set_bytes(&encoder, 5, &hidden_u32);
    set_bytes(&encoder, 6, &intermediate_u32);
    set_bytes(&encoder, 7, &active_u32);
    set_bytes(&encoder, 8, &experts_per_chunk_u32);
    set_bytes(&encoder, 9, &matrix_stride_u64);
    set_bytes(&encoder, 10, &code_stride_u64);
    set_bytes(&encoder, 11, &scale_stride_u64);
    encoder.dispatch_thread_groups(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(NVFP4_DOWN_THREADS as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{hidden}],intermediate={intermediate},active={active_experts}");
    ctx.commit_and_wait_profiled(&command, "nvfp4_grouped_experts_decode_f16", &shape, input.buffer.length() + first_chunk.length() + second_chunk.length() + route_weights.length(), activated.length() + output.buffer.length());
    Ok(output)
}

/// matrix-major selected arena 的 grouped decode；与 archive 版本保持同一数值顺序。
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_matrix_experts_decode(
    ctx: &MetalContext,
    input: &MetalTensor,
    codes: &[metal::Buffer; 3],
    scales: &[metal::Buffer; 3],
    global_scales: &[metal::Buffer; 3],
    active_experts: usize,
    code_stride: usize,
    scale_stride: usize,
    hidden: usize,
    intermediate: usize,
    route_weights: &[f32],
    activation: &Activation,
) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != hidden || active_experts == 0 || active_experts > 8 || route_weights.len() != active_experts {
        return Err(format!("NVFP4 grouped matrix decode shape 非法: input=[{},{}] active={active_experts} weights={}", input.rows, input.cols, route_weights.len()));
    }
    const NVFP4_GATE_UP_THREADS: usize = 128;
    const NVFP4_DOWN_THREADS: usize = 128;
    let activation = GatedActivation::from_spec(activation)?;
    let hidden_u32 = validate_u32("NVFP4 grouped matrix hidden", hidden)?;
    let intermediate_u32 = validate_u32("NVFP4 grouped matrix intermediate", intermediate)?;
    let active_u32 = validate_u32("NVFP4 grouped matrix active", active_experts)?;
    let code_stride_u64 = code_stride as u64;
    let scale_stride_u64 = scale_stride as u64;
    let route_weights = ctx.shared_buffer(as_bytes(route_weights));
    let activated = ctx.shared_buffer_zeros(active_experts * intermediate * mem::size_of::<f16>());
    let output = ctx.tensor_zeros(1, hidden);
    let gate_up = ctx.pipeline("nvfp4_grouped_matrix_gate_up_f16")?;
    let down = ctx.pipeline("nvfp4_grouped_matrix_down_f16")?;
    if gate_up.max_total_threads_per_threadgroup() < NVFP4_GATE_UP_THREADS as u64 || down.max_total_threads_per_threadgroup() < NVFP4_DOWN_THREADS as u64 {
        return Err("NVFP4 grouped matrix decode threadgroup 上限不足".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&gate_up);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&codes[0]), 0);
    encoder.set_buffer(2, Some(&scales[0]), 0);
    encoder.set_buffer(3, Some(&global_scales[0]), 0);
    encoder.set_buffer(4, Some(&codes[1]), 0);
    encoder.set_buffer(5, Some(&scales[1]), 0);
    encoder.set_buffer(6, Some(&global_scales[1]), 0);
    encoder.set_buffer(7, Some(&activated), 0);
    set_bytes(&encoder, 8, &hidden_u32);
    set_bytes(&encoder, 9, &intermediate_u32);
    set_bytes(&encoder, 10, &active_u32);
    set_bytes(&encoder, 11, &code_stride_u64);
    set_bytes(&encoder, 12, &scale_stride_u64);
    set_bytes(&encoder, 13, &activation.kind);
    set_bytes(&encoder, 14, &activation.alpha);
    set_bytes(&encoder, 15, &activation.limit);
    encoder.dispatch_thread_groups(MTLSize::new(intermediate as u64, active_experts as u64, 1), MTLSize::new(NVFP4_GATE_UP_THREADS as u64, 1, 1));
    encoder.memory_barrier_with_resources(&[activated.as_ref()]);
    encoder.set_compute_pipeline_state(&down);
    encoder.set_buffer(0, Some(&activated), 0);
    encoder.set_buffer(1, Some(&codes[2]), 0);
    encoder.set_buffer(2, Some(&scales[2]), 0);
    encoder.set_buffer(3, Some(&global_scales[2]), 0);
    encoder.set_buffer(4, Some(&route_weights), 0);
    encoder.set_buffer(5, Some(&output.buffer), 0);
    set_bytes(&encoder, 6, &hidden_u32);
    set_bytes(&encoder, 7, &intermediate_u32);
    set_bytes(&encoder, 8, &active_u32);
    set_bytes(&encoder, 9, &code_stride_u64);
    set_bytes(&encoder, 10, &scale_stride_u64);
    encoder.dispatch_thread_groups(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(NVFP4_DOWN_THREADS as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{hidden}],intermediate={intermediate},active={active_experts},layout=matrix-major");
    ctx.commit_and_wait_profiled(
        &command,
        "nvfp4_grouped_matrix_experts_decode_f16",
        &shape,
        input.buffer.length() + codes.iter().map(|buffer| buffer.length()).sum::<u64>() + scales.iter().map(|buffer| buffer.length()).sum::<u64>() + global_scales.iter().map(|buffer| buffer.length()).sum::<u64>() + route_weights.length(),
        activated.length() + output.buffer.length(),
    );
    Ok(output)
}
