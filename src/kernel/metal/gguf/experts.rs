//! GGUF MoE:indexed experts gate/up/down 融合 decode 与 expert accumulate。

/// 本家族的 Metal shader(*_indexed 与 reduce kernel)。
pub const SHADERS: &str = r#"
kernel void gguf_gated_gemv_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device const uint *expert_ids [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const ulong *iq2s_grid [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant uint &output_rows [[buffer(7)]],
    constant uint &gate_row_bytes [[buffer(8)]],
    constant uint &up_row_bytes [[buffer(9)]],
    constant uint &gate_expert_bytes [[buffer(10)]],
    constant uint &up_expert_bytes [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &alpha [[buffer(13)]],
    constant float &limit [[buffer(14)]],
    constant uint &gate_type [[buffer(15)]],
    constant uint &up_type [[buffer(16)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint row = group.x;
    if (row >= output_rows) return;
    const uint expert = expert_ids[group.y];
    device const uchar *gate_row = gate_weights + ulong(expert) * gate_expert_bytes + ulong(row) * gate_row_bytes;
    device const uchar *up_row = up_weights + ulong(expert) * up_expert_bytes + ulong(row) * up_row_bytes;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint column = lane; column < columns; column += 64) {
        const float value = float(input[column]);
        gate_sum += gguf_weight_f32(gate_row, iq2s_grid, gate_type, column) * value;
        up_sum += gguf_weight_f32(up_row, iq2s_grid, up_type, column) * value;
    }
    threadgroup float gate_partial[64];
    threadgroup float up_partial[64];
    gate_partial[lane] = gate_sum;
    up_partial[lane] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            gate_partial[lane] += gate_partial[lane + stride];
            up_partial[lane] += up_partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[ulong(group.y) * output_rows + row] = finite_f16(gated_activation_value(
            float(finite_f16(gate_partial[0])), float(finite_f16(up_partial[0])), activation_kind, alpha, limit));
    }
}
kernel void gguf_gemv_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const uint *expert_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    device const ulong *iq2s_grid [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &expert_bytes [[buffer(8)]],
    constant uint &tensor_type [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint row = group.x;
    if (row >= output_rows) return;
    const uint slot = group.y;
    device const half *input_row = input + ulong(slot) * columns;
    device const uchar *weight_row = weights + ulong(expert_ids[slot]) * expert_bytes + ulong(row) * row_bytes;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 64) {
        sum += gguf_weight_f32(weight_row, iq2s_grid, tensor_type, column) * float(input_row[column]);
    }
    threadgroup float partial[64];
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[ulong(slot) * output_rows + row] = finite_f16(partial[0]);
}
kernel void gguf_gated_gemv_q3k_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device const uint *expert_ids [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const ulong *iq2s_grid [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant uint &output_rows [[buffer(7)]],
    constant uint &gate_row_bytes [[buffer(8)]],
    constant uint &up_row_bytes [[buffer(9)]],
    constant uint &gate_expert_bytes [[buffer(10)]],
    constant uint &up_expert_bytes [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &alpha [[buffer(13)]],
    constant float &limit [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint first_row = group.x * 4 + simd_group * 2;
    if (first_row >= output_rows) return;
    const uint slot = group.y;
    const uint expert = expert_ids[slot];
    device const uchar *gate_weight = gate_weights + ulong(expert) * gate_expert_bytes;
    device const uchar *up_weight = up_weights + ulong(expert) * up_expert_bytes;
    const float2 gate_totals = q3k_gemv2_f16(input, gate_weight, columns, gate_row_bytes, first_row, output_rows, simd_lane);
    const float2 up_totals = q3k_gemv2_f16(input, up_weight, columns, up_row_bytes, first_row, output_rows, simd_lane);
    if (simd_lane == 0) {
        const ulong base = ulong(slot) * output_rows + first_row;
        output[base] = finite_f16(gated_activation_value(
            float(finite_f16(gate_totals.x)), float(finite_f16(up_totals.x)), activation_kind, alpha, limit));
        if (first_row + 1 < output_rows) {
            output[base + 1] = finite_f16(gated_activation_value(
                float(finite_f16(gate_totals.y)), float(finite_f16(up_totals.y)), activation_kind, alpha, limit));
        }
    }
}
kernel void gguf_gated_gemv_q4k_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device const uint *expert_ids [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const ulong *iq2s_grid [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant uint &output_rows [[buffer(7)]],
    constant uint &gate_row_bytes [[buffer(8)]],
    constant uint &up_row_bytes [[buffer(9)]],
    constant uint &gate_expert_bytes [[buffer(10)]],
    constant uint &up_expert_bytes [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &alpha [[buffer(13)]],
    constant float &limit [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group.x * 8 + simd_group;
    if (row >= output_rows) return;
    const uint slot = group.y;
    const uint expert = expert_ids[slot];
    device const uchar *gate_row = gate_weights + ulong(expert) * gate_expert_bytes + ulong(row) * gate_row_bytes;
    device const uchar *up_row = up_weights + ulong(expert) * up_expert_bytes + ulong(row) * up_row_bytes;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint group_count = columns >> 5;
    for (uint quant_group = simd_lane; quant_group < group_count; quant_group += 32) {
        device const half *input_group = input + quant_group * 32;
        gate_sum += q4k_dot32_f16(input_group, gate_row + ulong(quant_group >> 3) * 144, quant_group & 7);
        up_sum += q4k_dot32_f16(input_group, up_row + ulong(quant_group >> 3) * 144, quant_group & 7);
    }
    const float gate_total = simd_sum(gate_sum);
    const float up_total = simd_sum(up_sum);
    if (simd_lane == 0) {
        output[ulong(slot) * output_rows + row] = finite_f16(gated_activation_value(
            float(finite_f16(gate_total)), float(finite_f16(up_total)), activation_kind, alpha, limit));
    }
}
kernel void gguf_gated_gemv_iq2s_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device const uint *expert_ids [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const ulong *iq2s_grid [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant uint &output_rows [[buffer(7)]],
    constant uint &gate_row_bytes [[buffer(8)]],
    constant uint &up_row_bytes [[buffer(9)]],
    constant uint &gate_expert_bytes [[buffer(10)]],
    constant uint &up_expert_bytes [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &alpha [[buffer(13)]],
    constant float &limit [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group.x * 8 + simd_group;
    if (row >= output_rows) return;
    const uint slot = group.y;
    const uint expert = expert_ids[slot];
    device const uchar *gate_row = gate_weights + ulong(expert) * gate_expert_bytes + ulong(row) * gate_row_bytes;
    device const uchar *up_row = up_weights + ulong(expert) * up_expert_bytes + ulong(row) * up_row_bytes;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint subblock_count = columns >> 5;
    for (uint subblock = simd_lane; subblock < subblock_count; subblock += 32) {
        const uint block_index = subblock >> 3;
        const uint quant_group = subblock & 7;
        device const half *input_block = input + subblock * 32;
        gate_sum += iq2s_dot32_f16(input_block, gate_row + ulong(block_index) * 82, iq2s_grid, quant_group);
        up_sum += iq2s_dot32_f16(input_block, up_row + ulong(block_index) * 82, iq2s_grid, quant_group);
    }
    const float gate_total = simd_sum(gate_sum);
    const float up_total = simd_sum(up_sum);
    if (simd_lane == 0) {
        output[ulong(slot) * output_rows + row] = finite_f16(gated_activation_value(
            float(finite_f16(gate_total)), float(finite_f16(up_total)), activation_kind, alpha, limit));
    }
}
kernel void gguf_gated_gemv_iq3xxs_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device const uint *expert_ids [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const ulong *iq2s_grid [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant uint &output_rows [[buffer(7)]],
    constant uint &gate_row_bytes [[buffer(8)]],
    constant uint &up_row_bytes [[buffer(9)]],
    constant uint &gate_expert_bytes [[buffer(10)]],
    constant uint &up_expert_bytes [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &alpha [[buffer(13)]],
    constant float &limit [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    threadgroup uint shared_grid[256];
    threadgroup uchar shared_signs[128];
    #pragma unroll
    for (uint index = 0; index < 4; ++index) shared_grid[thread_index * 4 + index] = iq3xxs_grid[thread_index * 4 + index];
    #pragma unroll
    for (uint index = 0; index < 2; ++index) shared_signs[thread_index * 2 + index] = ksigns_iq2xs[thread_index * 2 + index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint first_row = group.x * 8 + simd_group * 4;
    if (first_row >= output_rows) return;
    const uint slot = group.y;
    const uint expert = expert_ids[slot];
    device const uchar *gate_expert = gate_weights + ulong(expert) * gate_expert_bytes;
    device const uchar *up_expert = up_weights + ulong(expert) * up_expert_bytes;
    float4 gate_sums = 0.0f;
    float4 up_sums = 0.0f;
    const uint ib32_count = columns >> 5;
    for (uint ib32 = simd_lane; ib32 < ib32_count; ib32 += 32) {
        const uint block_index = ib32 >> 3;
        device const half *input_block = input + ib32 * 32;
        gate_sums += iq3xxs_dot32_4r_f16(
            input_block, gate_expert, gate_row_bytes, first_row, output_rows,
            block_index, ib32 & 7, shared_grid, shared_signs);
        up_sums += iq3xxs_dot32_4r_f16(
            input_block, up_expert, up_row_bytes, first_row, output_rows,
            block_index, ib32 & 7, shared_grid, shared_signs);
    }
    #pragma unroll
    for (uint offset = 0; offset < 4; ++offset) {
        const float gate_total = simd_sum(gate_sums[offset]);
        const float up_total = simd_sum(up_sums[offset]);
        if (simd_lane == 0 && first_row + offset < output_rows) {
            output[ulong(slot) * output_rows + first_row + offset] = finite_f16(gated_activation_value(
                gate_total, up_total, activation_kind, alpha, limit));
        }
    }
}
kernel void gguf_gated_gemv_iq3s_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],
    device const uchar *up_weights [[buffer(2)]],
    device const uint *expert_ids [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const ulong *iq2s_grid [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant uint &output_rows [[buffer(7)]],
    constant uint &gate_row_bytes [[buffer(8)]],
    constant uint &up_row_bytes [[buffer(9)]],
    constant uint &gate_expert_bytes [[buffer(10)]],
    constant uint &up_expert_bytes [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &alpha [[buffer(13)]],
    constant float &limit [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint first_row = group.x * 8 + simd_group * 4;
    if (first_row >= output_rows) return;
    const uint slot = group.y;
    const uint expert = expert_ids[slot];
    device const uchar *gate_expert = gate_weights + ulong(expert) * gate_expert_bytes;
    device const uchar *up_expert = up_weights + ulong(expert) * up_expert_bytes;
    const float4 gate_sums = iq3s_row_sum4_f16(
        input, gate_expert, gate_row_bytes, first_row, output_rows,
        columns >> 8, simd_lane >> 2, simd_lane & 3);
    const float4 up_sums = iq3s_row_sum4_f16(
        input, up_expert, up_row_bytes, first_row, output_rows,
        columns >> 8, simd_lane >> 2, simd_lane & 3);
    #pragma unroll
    for (uint offset = 0; offset < 4; ++offset) {
        const float gate_total = simd_sum(gate_sums[offset]);
        const float up_total = simd_sum(up_sums[offset]);
        if (simd_lane == 0 && first_row + offset < output_rows) {
            output[ulong(slot) * output_rows + first_row + offset] = finite_f16(gated_activation_value(
                float(finite_f16(gate_total)), float(finite_f16(up_total)), activation_kind, alpha, limit));
        }
    }
}
kernel void gguf_gemv_q3k_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const uint *expert_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    device const ulong *iq2s_grid [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &expert_bytes [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint first_row = group.x * 4 + simd_group * 2;
    if (first_row >= output_rows) return;
    const uint slot = group.y;
    device const uchar *weight = weights + ulong(expert_ids[slot]) * expert_bytes;
    device const half *input_row = input + ulong(slot) * columns;
    const float2 totals = q3k_gemv2_f16(input_row, weight, columns, row_bytes, first_row, output_rows, simd_lane);
    if (simd_lane == 0) {
        const ulong base = ulong(slot) * output_rows + first_row;
        output[base] = finite_f16(totals.x);
        if (first_row + 1 < output_rows) output[base + 1] = finite_f16(totals.y);
    }
}
kernel void gguf_gemv_q4k_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const uint *expert_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    device const ulong *iq2s_grid [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &expert_bytes [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group.x * 8 + simd_group;
    if (row >= output_rows) return;
    const uint slot = group.y;
    device const uchar *weight_row = weights + ulong(expert_ids[slot]) * expert_bytes + ulong(row) * row_bytes;
    device const half *input_row = input + ulong(slot) * columns;
    float sum = 0.0f;
    const uint group_count = columns >> 5;
    for (uint quant_group = simd_lane; quant_group < group_count; quant_group += 32) {
        sum += q4k_dot32_f16(input_row + quant_group * 32, weight_row + ulong(quant_group >> 3) * 144, quant_group & 7);
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) output[ulong(slot) * output_rows + row] = finite_f16(total);
}
kernel void gguf_gemv_iq2s_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const uint *expert_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    device const ulong *iq2s_grid [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &expert_bytes [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group.x * 8 + simd_group;
    if (row >= output_rows) return;
    const uint slot = group.y;
    device const uchar *weight_row = weights + ulong(expert_ids[slot]) * expert_bytes + ulong(row) * row_bytes;
    device const half *input_row = input + ulong(slot) * columns;
    float sum = 0.0f;
    const uint subblock_count = columns >> 5;
    for (uint subblock = simd_lane; subblock < subblock_count; subblock += 32) {
        sum += iq2s_dot32_f16(
            input_row + subblock * 32,
            weight_row + ulong(subblock >> 3) * 82,
            iq2s_grid,
            subblock & 7);
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) output[ulong(slot) * output_rows + row] = finite_f16(total);
}
kernel void gguf_gemv_iq3xxs_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const uint *expert_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    device const ulong *iq2s_grid [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &expert_bytes [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    threadgroup uint shared_grid[256];
    threadgroup uchar shared_signs[128];
    #pragma unroll
    for (uint index = 0; index < 4; ++index) shared_grid[thread_index * 4 + index] = iq3xxs_grid[thread_index * 4 + index];
    #pragma unroll
    for (uint index = 0; index < 2; ++index) shared_signs[thread_index * 2 + index] = ksigns_iq2xs[thread_index * 2 + index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint first_row = group.x * 8 + simd_group * 4;
    if (first_row >= output_rows) return;
    const uint slot = group.y;
    device const half *input_row = input + ulong(slot) * columns;
    device const uchar *expert = weights + ulong(expert_ids[slot]) * expert_bytes;
    float4 sums = 0.0f;
    const uint ib32_count = columns >> 5;
    for (uint ib32 = simd_lane; ib32 < ib32_count; ib32 += 32) {
        const uint block_index = ib32 >> 3;
        sums += iq3xxs_dot32_4r_f16(
            input_row + ib32 * 32, expert, row_bytes, first_row, output_rows,
            block_index, ib32 & 7, shared_grid, shared_signs);
    }
    #pragma unroll
    for (uint offset = 0; offset < 4; ++offset) {
        const float total = simd_sum(sums[offset]);
        if (simd_lane == 0 && first_row + offset < output_rows) {
            output[ulong(slot) * output_rows + first_row + offset] = finite_f16(total);
        }
    }
}
kernel void gguf_gemv_iq3s_indexed_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const uint *expert_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    device const ulong *iq2s_grid [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &expert_bytes [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint first_row = group.x * 8 + simd_group * 4;
    if (first_row >= output_rows) return;
    const uint slot = group.y;
    device const uchar *expert = weights + ulong(expert_ids[slot]) * expert_bytes;
    device const half *input_row = input + ulong(slot) * columns;
    const float4 sums = iq3s_row_sum4_f16(
        input_row, expert, row_bytes, first_row, output_rows,
        columns >> 8, simd_lane >> 2, simd_lane & 3);
    #pragma unroll
    for (uint offset = 0; offset < 4; ++offset) {
        const float total = simd_sum(sums[offset]);
        if (simd_lane == 0 && first_row + offset < output_rows) {
            output[ulong(slot) * output_rows + first_row + offset] = finite_f16(total);
        }
    }
}
kernel void gguf_indexed_experts_reduce_f16(
    device const half *experts [[buffer(0)]],
    device const float *route_weights [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &top_k [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= columns) return;
    float sum = 0.0f;
    for (uint slot = 0; slot < top_k; ++slot) {
        sum += float(experts[ulong(slot) * columns + column]) * route_weights[slot];
    }
    output[column] = finite_f16(sum);
}

kernel void gguf_routed_value_iq3s_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const uint *expert_ids [[buffer(2)]],
    device const float *route_weights [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &expert_bytes [[buffer(8)]],
    constant uint &top_k [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint first_row = group.x * 8 + simd_group * 4;
    if (first_row >= output_rows) return;
    const uint token = group.y;
    device const half *input_row = input + ulong(token) * columns;
    float4 weighted = 0.0f;
    for (uint slot = 0; slot < top_k; ++slot) {
        const ulong route = ulong(token) * top_k + slot;
        device const uchar *expert = weights + ulong(expert_ids[route]) * expert_bytes;
        const float4 partials = iq3s_row_sum4_f16(
            input_row, expert, row_bytes, first_row, output_rows,
            columns >> 8, simd_lane >> 2, simd_lane & 3);
        #pragma unroll
        for (uint offset = 0; offset < 4; ++offset) {
            const float value = simd_sum(partials[offset]);
            if (simd_lane == 0) {
                weighted[offset] += (value / (1.0f + exp(-value))) * route_weights[route];
            }
        }
    }
    if (simd_lane == 0) {
        #pragma unroll
        for (uint offset = 0; offset < 4; ++offset) {
            if (first_row + offset < output_rows) {
                output[ulong(token) * output_rows + first_row + offset] = finite_f16(weighted[offset]);
            }
        }
    }
}
"#;

use crate::backend::metal::MetalWeight;
use crate::backend::metal::api as metal;

use super::super::dense::GatedActivation;
use super::super::moe::MetalF32Accumulator;
use super::super::{Activation, MTLSize, MetalContext, MetalTensor, MetalTensorDType, f16, mem, set_bytes, to_f16_tensor, validate_size, validate_u32};
use super::{gguf_expected_row_bytes, gguf_iq2s_grid_buffer};

#[cfg(test)]
use super::super::as_bytes;

#[allow(clippy::too_many_arguments)]
pub fn gguf_indexed_experts_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    gate: &MetalWeight,
    up: &MetalWeight,
    down: &MetalWeight,
    expert_ids: &metal::Buffer,
    route_weights: &metal::Buffer,
    top_k: usize,
    activation: &Activation,
) -> Result<MetalTensor, String> {
    let (
        MetalWeight::Gguf { blob: gate_blob, tensor_type: gate_type, row_bytes: gate_row_bytes, rows: gate_rows, cols: gate_cols },
        MetalWeight::Gguf { blob: up_blob, tensor_type: up_type, row_bytes: up_row_bytes, rows: up_rows, cols: up_cols },
        MetalWeight::Gguf { blob: down_blob, tensor_type: down_type, row_bytes: down_row_bytes, rows: down_rows, cols: down_cols },
    ) = (gate, up, down)
    else {
        return Err("GGUF indexed experts 需要三个 GGUF projection".to_owned());
    };
    if input.rows != 1
        || input.cols != *gate_cols
        || gate_cols != up_cols
        || gate_rows != up_rows
        || down_cols != gate_rows
        || down_rows != gate_cols
        || gate_type != up_type
        || !crate::weight::codec::ggml::supports_decode(*gate_type)
        || !crate::weight::codec::ggml::supports_decode(*down_type)
        || top_k == 0
    {
        return Err(format!(
            "GGUF indexed experts 布局不兼容: input=[{},{}], gate=[{gate_rows},{gate_cols}] type={gate_type}, up=[{up_rows},{up_cols}] type={up_type}, down=[{down_rows},{down_cols}] type={down_type}, top_k={top_k}",
            input.rows, input.cols,
        ));
    }
    let gate_expert_bytes = gate_row_bytes.checked_mul(*gate_rows).ok_or("GGUF indexed gate stride 溢出")?;
    let up_expert_bytes = up_row_bytes.checked_mul(*up_rows).ok_or("GGUF indexed up stride 溢出")?;
    let down_expert_bytes = down_row_bytes.checked_mul(*down_rows).ok_or("GGUF indexed down stride 溢出")?;
    let expert_count = gate_blob.length() as usize / gate_expert_bytes;
    if expert_count == 0 || gate_blob.length() as usize != expert_count * gate_expert_bytes || up_blob.length() as usize != expert_count * up_expert_bytes || down_blob.length() as usize != expert_count * down_expert_bytes {
        return Err("GGUF indexed experts 连续 buffer 大小不一致".to_owned());
    }
    let route_bytes = top_k * mem::size_of::<u32>();
    if (expert_ids.length() as usize) < route_bytes || (route_weights.length() as usize) < route_bytes {
        return Err(format!("GGUF indexed route buffer 太小: ids={}, weights={}, 需要={route_bytes}", expert_ids.length(), route_weights.length(),));
    }

    let input = to_f16_tensor(ctx, input)?;
    let activated = ctx.tensor_kernel_output(top_k, *gate_rows);
    let expert_output = ctx.tensor_kernel_output(top_k, *down_rows);
    let output = ctx.tensor_kernel_output(1, *down_rows);
    let grid = gguf_iq2s_grid_buffer(ctx);
    let params = GatedActivation::from_spec(activation)?;
    let gate_pipeline_name = match *gate_type {
        11 => "gguf_gated_gemv_q3k_indexed_f16",
        12 => "gguf_gated_gemv_q4k_indexed_f16",
        18 => "gguf_gated_gemv_iq3xxs_indexed_f16",
        21 => "gguf_gated_gemv_iq3s_indexed_f16",
        22 => "gguf_gated_gemv_iq2s_indexed_f16",
        _ => "gguf_gated_gemv_indexed_f16",
    };
    let down_pipeline_name = match *down_type {
        11 => "gguf_gemv_q3k_indexed_f16",
        12 => "gguf_gemv_q4k_indexed_f16",
        18 => "gguf_gemv_iq3xxs_indexed_f16",
        21 => "gguf_gemv_iq3s_indexed_f16",
        22 => "gguf_gemv_iq2s_indexed_f16",
        _ => "gguf_gemv_indexed_f16",
    };
    let gate_pipeline = ctx.pipeline(gate_pipeline_name)?;
    let down_pipeline = ctx.pipeline(down_pipeline_name)?;
    let reduce_pipeline = ctx.pipeline("gguf_indexed_experts_reduce_f16")?;
    let columns = validate_u32("GGUF indexed columns", *gate_cols)?;
    let intermediate = validate_u32("GGUF indexed intermediate", *gate_rows)?;
    let hidden = validate_u32("GGUF indexed hidden", *down_rows)?;
    let gate_row_bytes_u32 = validate_u32("GGUF indexed gate row bytes", *gate_row_bytes)?;
    let up_row_bytes_u32 = validate_u32("GGUF indexed up row bytes", *up_row_bytes)?;
    let down_row_bytes_u32 = validate_u32("GGUF indexed down row bytes", *down_row_bytes)?;
    let gate_expert_bytes_u32 = validate_u32("GGUF indexed gate expert bytes", gate_expert_bytes)?;
    let up_expert_bytes_u32 = validate_u32("GGUF indexed up expert bytes", up_expert_bytes)?;
    let down_expert_bytes_u32 = validate_u32("GGUF indexed down expert bytes", down_expert_bytes)?;
    let top_k_u32 = validate_u32("GGUF indexed top_k", top_k)?;

    let generic_gate = !matches!(*gate_type, 11 | 12 | 18 | 21 | 22);
    let gate_q3 = *gate_type == 11;
    let gate_iq3xxs = *gate_type == 18;
    let gate_iq3s = *gate_type == 21;
    let generic_down = !matches!(*down_type, 11 | 12 | 18 | 21 | 22);
    let down_q3 = *down_type == 11;
    let down_iq3xxs = *down_type == 18;
    let down_iq3s = *down_type == 21;
    let encode_gate = |encoder: &metal::ComputeCommandEncoderRef, barrier: bool| {
        encoder.set_compute_pipeline_state(&gate_pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(gate_blob), 0);
        encoder.set_buffer(2, Some(up_blob), 0);
        encoder.set_buffer(3, Some(expert_ids), 0);
        encoder.set_buffer(4, Some(&activated.buffer), 0);
        encoder.set_buffer(5, Some(&grid), 0);
        set_bytes(encoder, 6, &columns);
        set_bytes(encoder, 7, &intermediate);
        set_bytes(encoder, 8, &gate_row_bytes_u32);
        set_bytes(encoder, 9, &up_row_bytes_u32);
        set_bytes(encoder, 10, &gate_expert_bytes_u32);
        set_bytes(encoder, 11, &up_expert_bytes_u32);
        set_bytes(encoder, 12, &params.kind);
        set_bytes(encoder, 13, &params.alpha);
        set_bytes(encoder, 14, &params.limit);
        if generic_gate {
            set_bytes(encoder, 15, gate_type);
            set_bytes(encoder, 16, up_type);
        }
        encoder.dispatch_thread_groups(
            MTLSize::new(
                if generic_gate {
                    *gate_rows
                } else if gate_q3 {
                    gate_rows.div_ceil(4)
                } else {
                    gate_rows.div_ceil(8)
                } as u64,
                top_k as u64,
                1,
            ),
            MTLSize::new(
                if gate_iq3xxs || gate_iq3s {
                    64
                } else if generic_gate || gate_q3 {
                    64
                } else {
                    256
                },
                1,
                1,
            ),
        );
        if barrier {
            encoder.memory_barrier_with_resources(&[activated.buffer.as_ref()]);
        }
    };
    let encode_down = |encoder: &metal::ComputeCommandEncoderRef, barrier: bool| {
        encoder.set_compute_pipeline_state(&down_pipeline);
        encoder.set_buffer(0, Some(&activated.buffer), 0);
        encoder.set_buffer(1, Some(down_blob), 0);
        encoder.set_buffer(2, Some(expert_ids), 0);
        encoder.set_buffer(3, Some(&expert_output.buffer), 0);
        encoder.set_buffer(4, Some(&grid), 0);
        set_bytes(encoder, 5, &intermediate);
        set_bytes(encoder, 6, &hidden);
        set_bytes(encoder, 7, &down_row_bytes_u32);
        set_bytes(encoder, 8, &down_expert_bytes_u32);
        if generic_down {
            set_bytes(encoder, 9, down_type);
        }
        encoder.dispatch_thread_groups(
            MTLSize::new(
                if generic_down {
                    *down_rows
                } else if down_q3 {
                    down_rows.div_ceil(4)
                } else {
                    down_rows.div_ceil(8)
                } as u64,
                top_k as u64,
                1,
            ),
            MTLSize::new(
                if down_iq3xxs || down_iq3s {
                    64
                } else if generic_down || down_q3 {
                    64
                } else {
                    256
                },
                1,
                1,
            ),
        );
        if barrier {
            encoder.memory_barrier_with_resources(&[expert_output.buffer.as_ref()]);
        }
    };
    let encode_reduce = |encoder: &metal::ComputeCommandEncoderRef| {
        encoder.set_compute_pipeline_state(&reduce_pipeline);
        encoder.set_buffer(0, Some(&expert_output.buffer), 0);
        encoder.set_buffer(1, Some(route_weights), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &top_k_u32);
        set_bytes(encoder, 4, &hidden);
        encoder.dispatch_threads(MTLSize::new(*down_rows as u64, 1, 1), MTLSize::new(128, 1, 1));
    };
    let gate_read_bytes = input.buffer.length() + ((gate_expert_bytes + up_expert_bytes) * top_k) as u64 + expert_ids.length();
    let down_read_bytes = activated.buffer.length() + (down_expert_bytes * top_k) as u64 + expert_ids.length();
    let reduce_read_bytes = expert_output.buffer.length() + route_weights.length();
    let shape = format!("experts={top_k},hidden={down_rows},intermediate={gate_rows},types={gate_type}/{down_type}");
    if ctx.detailed_gpu_profiles_enabled() {
        for (operator, read_bytes, write_bytes, encode) in [
            (gate_pipeline_name, gate_read_bytes, activated.buffer.length(), 0_u8),
            (down_pipeline_name, down_read_bytes, expert_output.buffer.length(), 1_u8),
            ("gguf_indexed_experts_reduce_f16", reduce_read_bytes, output.buffer.length(), 2_u8),
        ] {
            let command = ctx.command_buffer();
            let encoder = command.new_compute_command_encoder();
            match encode {
                0 => encode_gate(&encoder, false),
                1 => encode_down(&encoder, false),
                _ => encode_reduce(&encoder),
            }
            encoder.end_encoding();
            ctx.commit_and_wait_profiled(&command, operator, &shape, read_bytes, write_bytes);
            ctx.submit_batch();
        }
    } else {
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encode_gate(&encoder, true);
        encode_down(&encoder, true);
        encode_reduce(&encoder);
        encoder.end_encoding();
        ctx.commit_and_wait_profiled(&command, "gguf_indexed_experts_f16", &shape, gate_read_bytes + down_read_bytes + reduce_read_bytes, activated.buffer.length() + expert_output.buffer.length() + output.buffer.length());
    }
    Ok(output)
}

/// K2-Horizon MoVA：对每个 token 的 top-k IQ3_S value expert 做
/// `sum(weight * silu(expert(input)))`。路由和专家投影留在同一设备队列，
/// 不物化 `[tokens, top_k, value_dim]` 中间张量。
#[allow(clippy::too_many_arguments)]
pub fn gguf_routed_value_iq3s_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    packed: &MetalWeight,
    expert_ids: &metal::Buffer,
    route_weights: &metal::Buffer,
    expert_count: usize,
    top_k: usize,
    output_rows: usize,
) -> Result<MetalTensor, String> {
    let MetalWeight::Gguf { blob, tensor_type, row_bytes, rows, cols } = packed else {
        return Err("MoVA packed value experts 必须是 GGUF 权重".to_owned());
    };
    if *tensor_type != 21 || expert_count == 0 || top_k == 0 || top_k > expert_count || output_rows == 0 || *rows != expert_count.checked_mul(output_rows).ok_or("MoVA expert 行数溢出")? || input.cols != *cols || !cols.is_multiple_of(256)
    {
        return Err(format!("MoVA IQ3_S 布局不兼容: input=[{},{}] packed=[{rows},{cols}] type={tensor_type} experts={expert_count} top_k={top_k} output_rows={output_rows}", input.rows, input.cols,));
    }
    let route_count = input.rows.checked_mul(top_k).ok_or("MoVA route 数溢出")?;
    let route_bytes = route_count.checked_mul(mem::size_of::<u32>()).ok_or("MoVA route bytes 溢出")?;
    if (expert_ids.length() as usize) < route_bytes || (route_weights.length() as usize) < route_bytes {
        return Err(format!("MoVA route buffer 太小: ids={} weights={} 需要={route_bytes}", expert_ids.length(), route_weights.length()));
    }
    let expert_bytes = row_bytes.checked_mul(output_rows).ok_or("MoVA expert stride 溢出")?;
    if blob.length() as usize != expert_bytes.checked_mul(expert_count).ok_or("MoVA packed bytes 溢出")? {
        return Err(format!("MoVA packed buffer={} 与 expert stride={expert_bytes} count={expert_count} 不一致", blob.length()));
    }
    let input = to_f16_tensor(ctx, input)?;
    let output = ctx.tensor_kernel_output(input.rows, output_rows);
    let pipeline_name = "gguf_routed_value_iq3s_f16";
    let pipeline = ctx.pipeline(pipeline_name)?;
    let columns = validate_u32("MoVA columns", *cols)?;
    let output_rows_u32 = validate_u32("MoVA output rows", output_rows)?;
    let row_bytes_u32 = validate_u32("MoVA row bytes", *row_bytes)?;
    let expert_bytes_u32 = validate_u32("MoVA expert bytes", expert_bytes)?;
    let top_k_u32 = validate_u32("MoVA top-k", top_k)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(blob), 0);
    encoder.set_buffer(2, Some(expert_ids), 0);
    encoder.set_buffer(3, Some(route_weights), 0);
    encoder.set_buffer(4, Some(&output.buffer), 0);
    set_bytes(&encoder, 5, &columns);
    set_bytes(&encoder, 6, &output_rows_u32);
    set_bytes(&encoder, 7, &row_bytes_u32);
    set_bytes(&encoder, 8, &expert_bytes_u32);
    set_bytes(&encoder, 9, &top_k_u32);
    encoder.dispatch_thread_groups(MTLSize::new(output_rows.div_ceil(8) as u64, input.rows as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("tokens={},experts={expert_count},top_k={top_k},output={output_rows},input={cols}", input.rows);
    let selected_weight_bytes = expert_bytes.saturating_mul(top_k).saturating_mul(input.rows);
    ctx.commit_and_wait_profiled(&command, pipeline_name, &shape, input.buffer.length() + selected_weight_bytes as u64 + expert_ids.length() + route_weights.length(), output.buffer.length());
    Ok(output)
}

pub fn prewarm_indexed_experts(ctx: &MetalContext, gate: &MetalWeight, down: &MetalWeight) -> Result<(), String> {
    let MetalWeight::Gguf { tensor_type: gate_type, .. } = gate else {
        return Err("indexed expert gate 不是 GGUF".to_owned());
    };
    let MetalWeight::Gguf { tensor_type: down_type, .. } = down else {
        return Err("indexed expert down 不是 GGUF".to_owned());
    };
    if !crate::weight::codec::ggml::supports_decode(*gate_type) || !crate::weight::codec::ggml::supports_decode(*down_type) {
        return Err(format!("indexed expert type={gate_type}/{down_type} 不受支持"));
    }
    let gate_pipeline = match *gate_type {
        11 => "gguf_gated_gemv_q3k_indexed_f16",
        12 => "gguf_gated_gemv_q4k_indexed_f16",
        18 => "gguf_gated_gemv_iq3xxs_indexed_f16",
        21 => "gguf_gated_gemv_iq3s_indexed_f16",
        22 => "gguf_gated_gemv_iq2s_indexed_f16",
        _ => "gguf_gated_gemv_indexed_f16",
    };
    let down_pipeline = match *down_type {
        11 => "gguf_gemv_q3k_indexed_f16",
        12 => "gguf_gemv_q4k_indexed_f16",
        18 => "gguf_gemv_iq3xxs_indexed_f16",
        21 => "gguf_gemv_iq3s_indexed_f16",
        22 => "gguf_gemv_iq2s_indexed_f16",
        _ => "gguf_gemv_indexed_f16",
    };
    for pipeline in [gate_pipeline, down_pipeline, "gguf_indexed_experts_reduce_f16"] {
        drop(ctx.pipeline(pipeline)?);
    }
    Ok(())
}

pub fn prefetch_indexed_expert_buffers(ctx: &MetalContext, gate: &MetalWeight, up: &MetalWeight, down: &MetalWeight) -> Result<(), String> {
    const PAGE_BYTES: usize = 16 * 1024;
    let blobs = [gate, up, down].map(|weight| match weight {
        MetalWeight::Gguf { blob, .. } => Ok(blob),
        _ => Err("indexed expert prefetch 收到非 GGUF 权重".to_owned()),
    });
    let blobs = blobs.into_iter().collect::<Result<Vec<_>, _>>()?;
    let max_pages = blobs.iter().map(|blob| (blob.length() as usize).div_ceil(PAGE_BYTES)).max().unwrap_or(0);
    if max_pages == 0 {
        return Ok(());
    }
    let scratch = ctx.shared_buffer_uninit(max_pages);
    let pipeline = ctx.pipeline("prefetch_shared_pages")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for blob in blobs {
        let length = blob.length();
        let pages = (length as usize).div_ceil(PAGE_BYTES);
        encoder.set_buffer(0, Some(blob), 0);
        encoder.set_buffer(1, Some(&scratch), 0);
        set_bytes(&encoder, 2, &length);
        encoder.dispatch_threads(MTLSize::new(pages as u64, 1, 1), MTLSize::new(128, 1, 1));
    }
    encoder.end_encoding();
    ctx.commit_and_force_wait_profiled(&command, "prefetch_shared_pages", "resident GGUF expert buffers", gguf_blob_bytes(gate) + gguf_blob_bytes(up) + gguf_blob_bytes(down), scratch.length());
    Ok(())
}

fn gguf_blob_bytes(weight: &MetalWeight) -> u64 {
    match weight {
        MetalWeight::Gguf { blob, .. } => blob.length(),
        _ => 0,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn encode_gguf_expert_accumulate_f32(
    ctx: &MetalContext,
    encoder: &metal::ComputeCommandEncoderRef,
    input: &MetalTensor,
    activated: &MetalTensor,
    activated_row: usize,
    grid: &metal::Buffer,
    gate_blob: &metal::Buffer,
    gate_type: u32,
    gate_row_bytes: usize,
    gate_rows: usize,
    gate_cols: usize,
    up_blob: &metal::Buffer,
    up_type: u32,
    up_row_bytes: usize,
    up_rows: usize,
    up_cols: usize,
    down_blob: &metal::Buffer,
    down_type: u32,
    down_row_bytes: usize,
    down_rows: usize,
    down_cols: usize,
    activation: &Activation,
    output: &MetalF32Accumulator,
    route_weight: f32,
) -> Result<(), String> {
    if input.dtype != MetalTensorDType::F16
        || input.rows != 1
        || input.cols != gate_cols
        || gate_cols != up_cols
        || gate_rows != up_rows
        || down_cols != gate_rows
        || activated_row >= activated.rows
        || activated.cols != gate_rows
        || output.rows != 1
        || output.cols != down_rows
    {
        return Err(format!(
            "GGUF expert accumulate shape 不兼容: input=[{},{}], activated=[{},{}] row={activated_row}, gate=[{gate_rows},{gate_cols}], up=[{up_rows},{up_cols}], down=[{down_rows},{down_cols}], output=[{},{}]",
            input.rows, input.cols, activated.rows, activated.cols, output.rows, output.cols,
        ));
    }
    for (name, blob, tensor_type, row_bytes, rows, cols) in
        [("gate", gate_blob, gate_type, gate_row_bytes, gate_rows, gate_cols), ("up", up_blob, up_type, up_row_bytes, up_rows, up_cols), ("down", down_blob, down_type, down_row_bytes, down_rows, down_cols)]
    {
        let expected_row_bytes = gguf_expected_row_bytes(tensor_type, cols)?;
        if row_bytes != expected_row_bytes {
            return Err(format!("GGUF expert {name} row bytes={row_bytes}，期望 {expected_row_bytes}"));
        }
        validate_size(&format!("GGUF expert {name} buffer"), row_bytes.checked_mul(rows).ok_or_else(|| format!("GGUF expert {name} 大小溢出"))?, blob.length() as usize)?;
    }

    let params = GatedActivation::from_spec(activation)?;
    let gate_up_pipeline = if gate_type == 11 && up_type == 11 {
        "gguf_gated_gemv_q3k_f16"
    } else if gate_type == 12 && up_type == 12 {
        "gguf_gated_gemv_q4k_f16"
    } else if gate_type == 22 && up_type == 22 {
        "gguf_gated_gemv_iq2s_f16"
    } else if gate_type == 18 && up_type == 18 {
        "gguf_gated_gemv_iq3xxs_f16"
    } else {
        "gguf_gated_gemv_f16"
    };
    let down_pipeline = match down_type {
        11 => "gguf_gemv_q3k_accumulate_f32",
        22 => "gguf_gemv_iq2s_accumulate_f32",
        _ => "gguf_gemv_accumulate_f32",
    };
    let gate_up_pipeline = ctx.pipeline(gate_up_pipeline)?;
    let down_pipeline = ctx.pipeline(down_pipeline)?;
    let gate_cols_u32 = validate_u32("GGUF expert gate columns", gate_cols)?;
    let gate_rows_u32 = validate_u32("GGUF expert gate rows", gate_rows)?;
    let gate_row_bytes_u32 = validate_u32("GGUF expert gate row bytes", gate_row_bytes)?;
    let up_row_bytes_u32 = validate_u32("GGUF expert up row bytes", up_row_bytes)?;
    let down_cols_u32 = validate_u32("GGUF expert down columns", down_cols)?;
    let down_rows_u32 = validate_u32("GGUF expert down rows", down_rows)?;
    let down_row_bytes_u32 = validate_u32("GGUF expert down row bytes", down_row_bytes)?;
    let activated_offset = activated_row.checked_mul(gate_rows).and_then(|elements| elements.checked_mul(mem::size_of::<f16>())).ok_or("GGUF expert activation offset 溢出")? as u64;

    encoder.set_compute_pipeline_state(&gate_up_pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(gate_blob), 0);
    encoder.set_buffer(2, Some(up_blob), 0);
    encoder.set_buffer(3, Some(grid), 0);
    encoder.set_buffer(4, Some(&activated.buffer), activated_offset);
    set_bytes(encoder, 5, &gate_cols_u32);
    set_bytes(encoder, 6, &gate_rows_u32);
    set_bytes(encoder, 7, &gate_type);
    set_bytes(encoder, 8, &up_type);
    set_bytes(encoder, 9, &gate_row_bytes_u32);
    set_bytes(encoder, 10, &up_row_bytes_u32);
    set_bytes(encoder, 11, &params.kind);
    set_bytes(encoder, 12, &params.alpha);
    set_bytes(encoder, 13, &params.limit);
    let packed_gate_up = matches!(gate_type, 11 | 12 | 18 | 22) && gate_type == up_type;
    encoder.dispatch_thread_groups(
        MTLSize::new(
            if gate_type == 11 && up_type == 11 {
                gate_rows.div_ceil(4)
            } else if packed_gate_up {
                gate_rows.div_ceil(8)
            } else {
                gate_rows
            } as u64,
            1,
            1,
        ),
        MTLSize::new(
            if gate_type == 11 && up_type == 11 {
                64
            } else if packed_gate_up {
                256
            } else {
                64
            },
            1,
            1,
        ),
    );
    encoder.memory_barrier_with_resources(&[activated.buffer.as_ref()]);

    encoder.set_compute_pipeline_state(&down_pipeline);
    encoder.set_buffer(0, Some(&activated.buffer), activated_offset);
    encoder.set_buffer(1, Some(down_blob), 0);
    encoder.set_buffer(2, Some(grid), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(encoder, 4, &down_cols_u32);
    set_bytes(encoder, 5, &down_rows_u32);
    set_bytes(encoder, 6, &down_type);
    set_bytes(encoder, 7, &down_row_bytes_u32);
    set_bytes(encoder, 8, &route_weight);
    let (groups, threads) = match down_type {
        11 => (down_rows.div_ceil(4), 64),
        22 => (down_rows.div_ceil(8), 256),
        _ => (down_rows, 64),
    };
    encoder.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(threads, 1, 1));
    encoder.memory_barrier_with_resources(&[output.buffer.as_ref()]);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gguf_expert_accumulate_f32(
    ctx: &MetalContext,
    input: &MetalTensor,
    activated: &MetalTensor,
    activated_row: usize,
    grid: &metal::Buffer,
    fence: &metal::FenceRef,
    wait_for_fence: bool,
    gate_blob: &metal::Buffer,
    gate_type: u32,
    gate_row_bytes: usize,
    gate_rows: usize,
    gate_cols: usize,
    up_blob: &metal::Buffer,
    up_type: u32,
    up_row_bytes: usize,
    up_rows: usize,
    up_cols: usize,
    down_blob: &metal::Buffer,
    down_type: u32,
    down_row_bytes: usize,
    down_rows: usize,
    down_cols: usize,
    activation: &Activation,
    output: &MetalF32Accumulator,
    route_weight: f32,
) -> Result<(), String> {
    let input = to_f16_tensor(ctx, input)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    if wait_for_fence {
        encoder.wait_for_fence(fence);
    }
    encode_gguf_expert_accumulate_f32(
        ctx,
        &encoder,
        &input,
        activated,
        activated_row,
        grid,
        gate_blob,
        gate_type,
        gate_row_bytes,
        gate_rows,
        gate_cols,
        up_blob,
        up_type,
        up_row_bytes,
        up_rows,
        up_cols,
        down_blob,
        down_type,
        down_row_bytes,
        down_rows,
        down_cols,
        activation,
        output,
        route_weight,
    )?;
    encoder.update_fence(fence);
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(
        &command,
        "gguf_expert_accumulate_f32",
        "resident expert",
        input.buffer.length() + gate_blob.length() + up_blob.length() + down_blob.length() + grid.length(),
        activated.buffer.length() + output.buffer.length(),
    );
    Ok(())
}

#[cfg(test)]
mod gguf_accumulate_tests {
    use super::*;
    use crate::kernel::metal::gguf::{gguf_gated_gemv_tensor_resident, gguf_iq2s_gemv_accumulate_f32, gguf_matmul_tensor_resident, gguf_q3k_gemv_accumulate_f32};
    use crate::kernel::metal::moe::{finish_moe_accumulator, moe_accumulator_zeros, scatter_add_rows_weighted_f32};

    #[test]
    fn iq3xxs_indexed_expert_matches_unfused_selected_expert() {
        let ctx = MetalContext::new_default().unwrap();
        let (hidden, intermediate, expert_count) = (256, 256, 2);
        let (iq3xxs_row_bytes, iq3s_row_bytes) = (98, 110);
        let input_values: Vec<f32> = (0..hidden).map(|index| (index as f32 * 0.03125).sin()).collect();
        let input = ctx.tensor_from_f32(&input_values, 1, hidden).unwrap();
        let make_experts = |row_bytes: usize, rows: usize| {
            (0..expert_count)
                .map(|expert| {
                    let mut block = vec![0u8; row_bytes];
                    block[..2].copy_from_slice(&f16::from_f32(0.001 * (expert + 1) as f32).to_bits().to_le_bytes());
                    (0..rows).flat_map(|_| block.iter().copied()).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let gate_experts = make_experts(iq3xxs_row_bytes, intermediate);
        let up_experts = make_experts(iq3xxs_row_bytes, intermediate);
        let down_experts = make_experts(iq3s_row_bytes, hidden);
        let pack = |experts: &[Vec<u8>]| experts.iter().flat_map(|expert| expert.iter().copied()).collect::<Vec<_>>();
        let gate = MetalWeight::Gguf { blob: ctx.resident_byte_weight_buffer(&pack(&gate_experts)), tensor_type: 18, row_bytes: iq3xxs_row_bytes, rows: intermediate, cols: hidden };
        let up = MetalWeight::Gguf { blob: ctx.resident_byte_weight_buffer(&pack(&up_experts)), tensor_type: 18, row_bytes: iq3xxs_row_bytes, rows: intermediate, cols: hidden };
        let down = MetalWeight::Gguf { blob: ctx.resident_byte_weight_buffer(&pack(&down_experts)), tensor_type: 21, row_bytes: iq3s_row_bytes, rows: hidden, cols: intermediate };
        let route_weight = 0.375f32;
        let ids = ctx.shared_buffer(as_bytes(&[1u32]));
        let weights = ctx.shared_buffer(as_bytes(&[route_weight]));
        let actual = gguf_indexed_experts_tensor_resident(&ctx, &input, &gate, &up, &down, &ids, &weights, 1, &Activation::Silu).unwrap();

        let gate_blob = ctx.resident_byte_weight_buffer(&gate_experts[1]);
        let up_blob = ctx.resident_byte_weight_buffer(&up_experts[1]);
        let down_blob = ctx.resident_byte_weight_buffer(&down_experts[1]);
        let activated = gguf_gated_gemv_tensor_resident(&ctx, &input, &gate_blob, 18, iq3xxs_row_bytes, intermediate, hidden, &up_blob, 18, iq3xxs_row_bytes, intermediate, hidden, &Activation::Silu).unwrap();
        let projected = gguf_matmul_tensor_resident(&ctx, &activated, &down_blob, 21, iq3s_row_bytes, hidden, intermediate).unwrap();
        let accumulator = moe_accumulator_zeros(&ctx, 1, hidden).unwrap();
        scatter_add_rows_weighted_f32(&ctx, &accumulator, &projected, &[0], &[route_weight]).unwrap();
        let expected = finish_moe_accumulator(&ctx, accumulator, None).unwrap();
        assert_eq!(ctx.read_f16_to_f32(&actual.buffer, hidden), ctx.read_f16_to_f32(&expected.buffer, hidden));
    }

    #[test]
    fn q3k_accumulate_matches_gemv_then_scatter() {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, columns, row_bytes) = (4, 256, 110);
        let input_values: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.03125).sin()).collect();
        let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let mut block = vec![0u8; row_bytes];
        block[108..110].copy_from_slice(&f16::from_f32(0.0078125).to_bits().to_le_bytes());
        let weights: Vec<u8> = (0..rows).flat_map(|_| block.iter().copied()).collect();
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let route_weight = 0.375f32;

        let gemv = gguf_matmul_tensor_resident(&ctx, &input, &blob, 11, row_bytes, rows, columns).unwrap();
        let expected_accumulator = moe_accumulator_zeros(&ctx, 1, rows).unwrap();
        scatter_add_rows_weighted_f32(&ctx, &expected_accumulator, &gemv, &[0], &[route_weight]).unwrap();
        let expected = finish_moe_accumulator(&ctx, expected_accumulator, None).unwrap();

        let actual_accumulator = moe_accumulator_zeros(&ctx, 1, rows).unwrap();
        gguf_q3k_gemv_accumulate_f32(&ctx, &input, &blob, row_bytes, rows, columns, &actual_accumulator, route_weight).unwrap();
        let actual = finish_moe_accumulator(&ctx, actual_accumulator, None).unwrap();
        assert_eq!(ctx.read_f16_to_f32(&actual.buffer, rows), ctx.read_f16_to_f32(&expected.buffer, rows));
    }

    #[test]
    fn iq2s_accumulate_matches_gemv_then_scatter() {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, columns, row_bytes) = (4, 256, 82);
        let input_values: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.03125).cos()).collect();
        let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let mut block = vec![0u8; row_bytes];
        block[..2].copy_from_slice(&f16::from_f32(0.0078125).to_bits().to_le_bytes());
        let weights: Vec<u8> = (0..rows).flat_map(|_| block.iter().copied()).collect();
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let route_weight = 0.375f32;

        let gemv = gguf_matmul_tensor_resident(&ctx, &input, &blob, 22, row_bytes, rows, columns).unwrap();
        let expected_accumulator = moe_accumulator_zeros(&ctx, 1, rows).unwrap();
        scatter_add_rows_weighted_f32(&ctx, &expected_accumulator, &gemv, &[0], &[route_weight]).unwrap();
        let expected = finish_moe_accumulator(&ctx, expected_accumulator, None).unwrap();

        let actual_accumulator = moe_accumulator_zeros(&ctx, 1, rows).unwrap();
        gguf_iq2s_gemv_accumulate_f32(&ctx, &input, &blob, row_bytes, rows, columns, &actual_accumulator, route_weight).unwrap();
        let actual = finish_moe_accumulator(&ctx, actual_accumulator, None).unwrap();
        assert_eq!(ctx.read_f16_to_f32(&actual.buffer, rows), ctx.read_f16_to_f32(&expected.buffer, rows));
    }

    #[test]
    fn mxfp4_expert_accumulate_matches_unfused_path() {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, columns, row_bytes) = (32, 32, 17);
        let input_values: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.03125).sin()).collect();
        let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let mut block = vec![0u8; row_bytes];
        block[0] = 127;
        for (index, value) in block[1..].iter_mut().enumerate() {
            *value = ((index + 1) as u8 & 7) | ((((index + 3) as u8 & 7) | 8) << 4);
        }
        let weights: Vec<u8> = (0..rows).flat_map(|row| block.iter().copied().map(move |value| value.wrapping_add((row % 3) as u8))).collect();
        let gate = ctx.resident_byte_weight_buffer(&weights);
        let up = ctx.resident_byte_weight_buffer(&weights);
        let down = ctx.resident_byte_weight_buffer(&weights);
        let route_weight = 0.375f32;

        let activated = gguf_gated_gemv_tensor_resident(&ctx, &input, &gate, 39, row_bytes, rows, columns, &up, 39, row_bytes, rows, columns, &Activation::Silu).unwrap();
        let projected = gguf_matmul_tensor_resident(&ctx, &activated, &down, 39, row_bytes, rows, columns).unwrap();
        let expected_accumulator = moe_accumulator_zeros(&ctx, 1, rows).unwrap();
        scatter_add_rows_weighted_f32(&ctx, &expected_accumulator, &projected, &[0], &[route_weight]).unwrap();
        let expected = finish_moe_accumulator(&ctx, expected_accumulator, None).unwrap();

        let actual_accumulator = moe_accumulator_zeros(&ctx, 1, rows).unwrap();
        let scratch = ctx.tensor_zeros(1, rows);
        let grid = gguf_iq2s_grid_buffer(&ctx);
        let fence = ctx.device.new_fence();
        gguf_expert_accumulate_f32(
            &ctx,
            &input,
            &scratch,
            0,
            &grid,
            &fence,
            false,
            &gate,
            39,
            row_bytes,
            rows,
            columns,
            &up,
            39,
            row_bytes,
            rows,
            columns,
            &down,
            39,
            row_bytes,
            rows,
            columns,
            &Activation::Silu,
            &actual_accumulator,
            route_weight,
        )
        .unwrap();
        let actual = finish_moe_accumulator(&ctx, actual_accumulator, Some(&fence)).unwrap();
        assert_eq!(ctx.read_f16_to_f32(&actual.buffer, rows), ctx.read_f16_to_f32(&expected.buffer, rows));
    }
}

#[cfg(test)]
mod router_parallel_tests {
    use super::*;
    use crate::kernel::metal::moe::moe_router_softmax_tensor_resident_f32;

    #[test]
    fn parallel_f32_router_matches_single_group() {
        let ctx = MetalContext::new_default().unwrap();
        let (columns, experts, top_k) = (64, 256, 8);
        let input_values: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.03125).sin()).collect();
        let weight_values: Vec<f32> = (0..experts * columns).map(|index| ((index % columns) as f32 * 0.017 + (index / columns) as f32 * 0.003).cos() * 0.125).collect();
        let input = ctx.tensor_from_f32_preserve(&input_values, 1, columns).unwrap();
        let weight = ctx.shared_buffer(as_bytes(&weight_values));
        let parallel = moe_router_softmax_tensor_resident_f32(&ctx, &input, &weight, weight_values.len(), experts, top_k, 1.0, true).unwrap();

        let output_ids = ctx.shared_buffer_zeros(top_k * mem::size_of::<u32>());
        let output_weights = ctx.shared_buffer_zeros(top_k * mem::size_of::<f32>());
        crate::kernel::metal::moe::encode_softmax_router(&ctx, "moe_router_softmax_topk_f32_input_weight", &input, &weight, experts, top_k, 1.0, true, experts, &output_ids, &output_weights, true).unwrap();
        let expected_ids = unsafe { std::slice::from_raw_parts(output_ids.contents().cast::<u32>(), top_k) };
        let expected_weights = unsafe { std::slice::from_raw_parts(output_weights.contents().cast::<f32>(), top_k) };
        assert_eq!(parallel.expert_ids, expected_ids);
        for (actual, expected) in parallel.weights.iter().zip(expected_weights) {
            assert!((actual - expected).abs() <= 1.0e-5, "actual={actual}, expected={expected}");
        }
    }
}
