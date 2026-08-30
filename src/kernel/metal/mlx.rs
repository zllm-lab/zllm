/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: mlx_affine_gemv_f16, mlx_affine_qmm_bf16, mlx_affine_dequant_f16, mlx_affine_dequant_f16_u4_group64, mlx_affine_gemv_bf16_u4, mlx_affine_gemv_f16_u4, mlx_affine_linear_gated_bf16_u4, mlx_affine_gemv_bf16_u8, mlx_affine_deinterleave_pair, mlx_affine_dequant_interleaved_f16_u8, mlx_affine_qmm_bf16_u8_interleaved, mlx_affine_gemv_bf16_u8_interleaved, mlx_affine_gated_interleaved_gemv_bf16_u8, mlx_affine_gated_gemv_bf16_u8, mlx_affine_gemv_bf16
pub const SHADERS: &str = r#"
kernel void mlx_affine_gemv_f16(
    device const half *input [[buffer(0)]],
    device const uint *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]],
    constant uint &output_columns [[buffer(7)]],
    constant uint &bits [[buffer(8)]],
    constant uint &group_size [[buffer(9)]],
    constant uint &scale_dtype [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint row = group.x;
    const uint input_row = group.y;
    if (row >= output_columns || input_row >= input_rows) return;
    threadgroup float sums[64];
    const uint values_per_word = 32 / bits;
    const uint mask = (1u << bits) - 1u;
    const ulong packed_columns = (ulong(input_columns) + values_per_word - 1) / values_per_word;
    const ulong groups = ulong(input_columns) / group_size;
    const ulong input_base = ulong(input_row) * input_columns;
    float sum = 0.0f;
    for (uint column = lane; column < input_columns; column += 64) {
        const uint word = packed[ulong(row) * packed_columns + column / values_per_word];
        const uint code = (word >> ((column % values_per_word) * bits)) & mask;
        const ulong parameter = ulong(row) * groups + column / group_size;
        const float scale = w4a16_scale(scales, parameter, scale_dtype);
        const float bias = w4a16_scale(biases, parameter, scale_dtype);
        sum += float(input[input_base + column]) * (scale * float(code) + bias);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[ulong(input_row) * output_columns + row] = finite_f16(sums[0]);
}
kernel void mlx_affine_qmm_bf16(
    device const ushort *input [[buffer(0)]], device const uint *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]], device const uchar *biases [[buffer(3)]],
    device ushort *output [[buffer(4)]], constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]], constant uint &output_columns [[buffer(7)]],
    constant uint &bits [[buffer(8)]], constant uint &group_size [[buffer(9)]],
    constant uint &scale_dtype [[buffer(10)]], uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup half input_tile[512];
    threadgroup half weight_tile[512];
    simdgroup_float8x8 accumulator_0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 accumulator_1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 accumulator_2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 accumulator_3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint row_base = group.y * 32;
    const uint output_base = group.x * 32;
    const uint values_per_word = 32 / bits;
    const uint mask = (1u << bits) - 1u;
    const ulong packed_columns = (ulong(input_columns) + values_per_word - 1) / values_per_word;
    const ulong parameter_columns = ulong(input_columns) / group_size;

    for (uint input_base = 0; input_base < input_columns; input_base += 16) {
        const uint begin = thread_index * 4;
        for (uint item = 0; item < 4; ++item) {
            const uint flat = begin + item;
            const uint local_row = flat >> 4;
            const uint local_column = flat & 15;
            const uint row = row_base + local_row;
            const uint input_column = input_base + local_column;
            input_tile[flat] = row < input_rows && input_column < input_columns
                ? half(zllm_bf16_to_f32(input[ulong(row) * input_columns + input_column]))
                : half(0.0h);

            const uint weight_input = input_base + (flat >> 5);
            const uint output_column = output_base + (flat & 31);
            if (output_column < output_columns && weight_input < input_columns) {
                const uint word = packed[ulong(output_column) * packed_columns + weight_input / values_per_word];
                const uint code = (word >> ((weight_input % values_per_word) * bits)) & mask;
                const ulong parameter = ulong(output_column) * parameter_columns + weight_input / group_size;
                const float scale = w4a16_scale(scales, parameter, scale_dtype);
                const float bias = w4a16_scale(biases, parameter, scale_dtype);
                weight_tile[flat] = half(zllm_bf16_to_f32(zllm_f32_to_bf16(scale * float(code) + bias)));
            } else {
                weight_tile[flat] = half(0.0h);
            }
        };
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 16; k += 8) {
            simdgroup_half8x8 input_matrix_0;
            simdgroup_half8x8 input_matrix_1;
            simdgroup_half8x8 input_matrix_2;
            simdgroup_half8x8 input_matrix_3;
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(input_matrix_0, input_tile + k, 16);
            simdgroup_load(input_matrix_1, input_tile + 128 + k, 16);
            simdgroup_load(input_matrix_2, input_tile + 256 + k, 16);
            simdgroup_load(input_matrix_3, input_tile + 384 + k, 16);
            simdgroup_load(weight_matrix, weight_tile + k * 32 + simd_index * 8, 32);
            simdgroup_multiply_accumulate(accumulator_0, input_matrix_0, weight_matrix, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, input_matrix_1, weight_matrix, accumulator_1);
            simdgroup_multiply_accumulate(accumulator_2, input_matrix_2, weight_matrix, accumulator_2);
            simdgroup_multiply_accumulate(accumulator_3, input_matrix_3, weight_matrix, accumulator_3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float result_tile[1024];
    simdgroup_store(accumulator_0, result_tile + simd_index * 8, 32);
    simdgroup_store(accumulator_1, result_tile + 256 + simd_index * 8, 32);
    simdgroup_store(accumulator_2, result_tile + 512 + simd_index * 8, 32);
    simdgroup_store(accumulator_3, result_tile + 768 + simd_index * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint begin = thread_index * 8;
    for (uint item = 0; item < 8; ++item) {
        const uint flat = begin + item;
        const uint row = row_base + (flat >> 5);
        const uint output_column = output_base + (flat & 31);
        if (row < input_rows && output_column < output_columns) {
            output[ulong(row) * output_columns + output_column] = zllm_f32_to_bf16(result_tile[flat]);
        }
    }
}
kernel void mlx_affine_dequant_f16(
    device const uint *packed [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *biases [[buffer(2)]], device half *output [[buffer(3)]],
    constant uint &packed_word_count [[buffer(4)]], constant uint &input_columns [[buffer(5)]],
    constant uint &bits [[buffer(6)]], constant uint &group_size [[buffer(7)]],
    constant uint &scale_dtype [[buffer(8)]], uint id [[thread_position_in_grid]])
{
    if (id >= packed_word_count) return;
    const uint values_per_word = 32 / bits;
    const ulong packed_columns = (ulong(input_columns) + values_per_word - 1) / values_per_word;
    const uint row = uint(ulong(id) / packed_columns);
    const uint packed_column = uint(ulong(id) - ulong(row) * packed_columns);
    const uint first_column = packed_column * values_per_word;
    const uint word = packed[id];
    const ulong parameter_columns = ulong(input_columns) / group_size;
    const ulong parameter = ulong(row) * parameter_columns + first_column / group_size;
    const float scale = w4a16_scale(scales, parameter, scale_dtype);
    const float bias = w4a16_scale(biases, parameter, scale_dtype);
    for (uint item = 0; item < values_per_word; ++item) {
        const uint column = first_column + item;
        if (column < input_columns) {
            const uint code = (word >> (item * bits)) & ((1u << bits) - 1u);
            output[ulong(row) * input_columns + column] = half(zllm_bf16_to_f32(zllm_f32_to_bf16(scale * float(code) + bias)));
        }
    }
}
kernel void mlx_affine_dequant_f16_u4_group64(
    device const uint *packed [[buffer(0)]],
    device const ushort *scales [[buffer(1)]],
    device const ushort *biases [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &packed_columns [[buffer(4)]],
    constant uint &input_columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &scale_dtype [[buffer(7)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= packed_columns || id.y >= output_rows) return;
    const ulong word_index = ulong(id.y) * packed_columns + id.x;
    const uint word = packed[word_index];
    const uint first_column = id.x * 8;
    const ulong parameter = ulong(id.y) * (input_columns / 64) + (first_column >> 6);
    const float scale = w4a16_scale((device const uchar *)scales, parameter, scale_dtype);
    const float bias = w4a16_scale((device const uchar *)biases, parameter, scale_dtype);
    const ulong output_base = ulong(id.y) * input_columns + first_column;
#pragma unroll
    for (uint item = 0; item < 8; ++item) {
        const uint code = (word >> (item * 4)) & 15u;
        output[output_base + item] = half(zllm_bf16_to_f32(
            zllm_f32_to_bf16(scale * float(code) + bias)));
    }
}
kernel void mlx_affine_gemv_bf16_u4(
    device const ushort *input [[buffer(0)]], device const uchar *codes [[buffer(1)]],
    device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
    device ushort *output [[buffer(4)]], constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]], constant uint &output_columns [[buffer(7)]],
    constant uint &bits [[buffer(8)]], constant uint &group_size [[buffer(9)]],
    constant uint &scale_dtype [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    (void)bits;
    (void)scale_dtype;
    // 行放 x 快速维:同列 tile 的各行组相邻调度,同时读同一份权重,靠 L1/L2 广播
    const uint input_row = group.x;
    if (input_row >= input_rows) return;
    const ulong packed_columns = (ulong(input_columns) + 7) / 8;
    const ulong weight_bytes = packed_columns * 4;
    const ulong groups = ulong(input_columns) / group_size;
    const ulong input_base = ulong(input_row) * input_columns;
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    const uint output_base = group.y * 8 + simd_group * 4;
    const uint values_per_thread = input_columns % 512 == 0 ? 16 : 8;
    const uint block_size = values_per_thread * 32;
    const uint lanes_per_group = max(group_size / values_per_thread, 1u);
    const uint scale_lane = simd_lane - simd_lane % lanes_per_group;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += block_size) {
        const uint begin = block + simd_lane * values_per_thread;
        const uint count = begin < input_columns ? min(values_per_thread, input_columns - begin) : 0;
        float input_values[16];
        float input_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < values_per_thread; index += 4) {
            if (index + 3 < count) {
                const ushort4 packed_input = *((device const ushort4 *)(input + input_base + begin + index));
                const float4 values = as_type<float4>(uint4(packed_input) << 16);
                input_values[index] = values.x;
                input_values[index + 1] = values.y * 0.0625f;
                input_values[index + 2] = values.z * 0.00390625f;
                input_values[index + 3] = values.w * 0.000244140625f;
                input_sum += values.x + values.y + values.z + values.w;
            } else {
                for (uint tail = index; tail < values_per_thread; ++tail) {
                    const float value = tail < count ? zllm_bf16_to_f32(input[input_base + begin + tail]) : 0.0f;
                    input_values[tail] = value / float(1u << ((tail & 3) * 4));
                    input_sum += value;
                }
            }
        }
        for (uint output_index = 0; output_index < 4; ++output_index) {
            const uint output_column = output_base + output_index;
            if (output_column >= output_columns || count == 0) continue;
            const device ushort *weight = (const device ushort *)(codes + ulong(output_column) * weight_bytes);
            float code_sum = 0.0f;
            if (values_per_thread == 16 && count == 16) {
                const ushort4 packed_values = *((device const ushort4 *)(weight + (begin >> 2)));
                // 4 个独立累加器:把 16 深的串行 FMA 链拆成 4 条 4 深,提高 ILP
                float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (uint index = 0; index < 4; ++index) {
                    const ushort packed = packed_values[index];
                    partial[0] += input_values[index * 4] * float(packed & 0x000f);
                    partial[1] += input_values[index * 4 + 1] * float(packed & 0x00f0);
                    partial[2] += input_values[index * 4 + 2] * float(packed & 0x0f00);
                    partial[3] += input_values[index * 4 + 3] * float(packed & 0xf000);
                }
                code_sum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
            } else {
                for (uint index = 0; index < values_per_thread / 4; ++index) {
                    if (index * 4 >= count) break;
                    const ushort packed = weight[(begin >> 2) + index];
                    code_sum +=
                        input_values[index * 4] * float(packed & 0x000f) +
                        input_values[index * 4 + 1] * float(packed & 0x00f0) +
                        input_values[index * 4 + 2] * float(packed & 0x0f00) +
                        input_values[index * 4 + 3] * float(packed & 0xf000);
                }
            }
            const ulong parameter = ulong(output_column) * groups + begin / group_size;
            float scale = simd_lane == scale_lane ? zllm_bf16_to_f32(scales[parameter]) : 0.0f;
            float bias = simd_lane == scale_lane ? zllm_bf16_to_f32(biases[parameter]) : 0.0f;
            scale = simd_shuffle(scale, scale_lane);
            bias = simd_shuffle(bias, scale_lane);
            result[output_index] += scale * code_sum + input_sum * bias;
        }
    }
    for (uint output_index = 0; output_index < 4; ++output_index) {
        result[output_index] = simd_sum(result[output_index]);
        const uint output_column = output_base + output_index;
        if (simd_lane == 0 && output_column < output_columns) {
            output[ulong(input_row) * output_columns + output_column] = zllm_f32_to_bf16(result[output_index]);
        }
    }
}
kernel void mlx_affine_gemv_f16_u4(
    device const half *input [[buffer(0)]], device const uchar *codes [[buffer(1)]],
    device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
    device half *output [[buffer(4)]], constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]], constant uint &output_columns [[buffer(7)]],
    constant uint &bits [[buffer(8)]], constant uint &group_size [[buffer(9)]],
    constant uint &scale_dtype [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    // 行放 x 快速维:同列 tile 的各行组相邻调度,同时读同一份权重,靠 L1/L2 广播
    const uint input_row = group.x;
    if (input_row >= input_rows) return;
    const ulong packed_columns = (ulong(input_columns) + 7) / 8;
    const ulong weight_bytes = packed_columns * 4;
    const ulong groups = ulong(input_columns) / group_size;
    const ulong input_base = ulong(input_row) * input_columns;
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    // 每 SIMD 组 8 列、每 threadgroup 16 行:单 lane 同时挂 8 路权重读,翻倍访存并行度;
    // 提到 32 行会因寄存器压力反而变慢,16 行是实测最优
    const uint output_base = group.y * 16 + simd_group * 8;
    // 16 值快路径只需列数对齐 16(12B 的 3840/15360 宽都满足),尾块由 count 归零自然跳过
    const uint values_per_thread = input_columns % 16 == 0 ? 16 : 8;
    const uint block_size = values_per_thread * 32;
    float result[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += block_size) {
        const uint begin = block + simd_lane * values_per_thread;
        const uint count = begin < input_columns ? min(values_per_thread, input_columns - begin) : 0;
        float input_values[16];
        float input_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < values_per_thread; index += 4) {
            if (index + 3 < count) {
                const half4 packed_input = *((device const half4 *)(input + input_base + begin + index));
                input_values[index] = float(packed_input.x);
                input_values[index + 1] = float(packed_input.y) * 0.0625f;
                input_values[index + 2] = float(packed_input.z) * 0.00390625f;
                input_values[index + 3] = float(packed_input.w) * 0.000244140625f;
                input_sum += float(packed_input.x) + float(packed_input.y) + float(packed_input.z) + float(packed_input.w);
            } else {
                for (uint tail = index; tail < values_per_thread; ++tail) {
                    const float value = tail < count ? float(input[input_base + begin + tail]) : 0.0f;
                    input_values[tail] = value / float(1u << ((tail & 3) * 4));
                    input_sum += value;
                }
            }
        }
#pragma unroll
        for (uint output_index = 0; output_index < 8; ++output_index) {
            const uint output_column = output_base + output_index;
            if (output_column >= output_columns || count == 0) continue;
            const device ushort *weight = (const device ushort *)(codes + ulong(output_column) * weight_bytes);
            float code_sum = 0.0f;
            if (values_per_thread == 16 && count == 16) {
                const ushort4 packed_values = *((device const ushort4 *)(weight + (begin >> 2)));
#pragma unroll
                for (uint index = 0; index < 4; ++index) {
                    const ushort packed = packed_values[index];
                    code_sum +=
                        input_values[index * 4] * float(packed & 0x000f) +
                        input_values[index * 4 + 1] * float(packed & 0x00f0) +
                        input_values[index * 4 + 2] * float(packed & 0x0f00) +
                        input_values[index * 4 + 3] * float(packed & 0xf000);
                }
            } else {
                for (uint index = 0; index < values_per_thread / 4; ++index) {
                    if (index * 4 >= count) break;
                    const ushort packed = weight[(begin >> 2) + index];
                    code_sum +=
                        input_values[index * 4] * float(packed & 0x000f) +
                        input_values[index * 4 + 1] * float(packed & 0x00f0) +
                        input_values[index * 4 + 2] * float(packed & 0x0f00) +
                        input_values[index * 4 + 3] * float(packed & 0xf000);
                }
            }
            const ulong parameter = ulong(output_column) * groups + begin / group_size;
            // 每 lane 直读参数:同组 lane 命中同一 L1 行,免去每组两次 simd_shuffle 的 SIMD 收敛开销;
            // count==0 的 lane 已被上面的 continue 拦下,不会越界读行尾
            const float scale = w4a16_scale((device const uchar *)scales, parameter, scale_dtype);
            const float bias = w4a16_scale((device const uchar *)biases, parameter, scale_dtype);
            result[output_index] += scale * code_sum + input_sum * bias;
        }
    }
    for (uint output_index = 0; output_index < 8; ++output_index) {
        result[output_index] = simd_sum(result[output_index]);
        const uint output_column = output_base + output_index;
        if (simd_lane == 0 && output_column < output_columns) {
            output[ulong(input_row) * output_columns + output_column] = half(result[output_index]);
        }
    }
}
kernel void mlx_affine_linear_gated_bf16_u4(
    device const ushort *input [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device const ushort *up [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]],
    constant uint &output_columns [[buffer(7)]],
    constant uint &group_size [[buffer(8)]],
    constant uint &activation_kind [[buffer(9)]],
    constant float &activation_alpha [[buffer(10)]],
    constant float &activation_limit [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const ulong packed_columns = (ulong(input_columns) + 7) / 8;
    const ulong weight_bytes = packed_columns * 4;
    const ulong groups = ulong(input_columns) / group_size;
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    const uint output_base = group * 8 + simd_group * 4;
    const uint values_per_thread = input_columns % 512 == 0 ? 16 : 8;
    const uint block_size = values_per_thread * 32;
    const uint lanes_per_group = max(group_size / values_per_thread, 1u);
    const uint scale_lane = simd_lane - simd_lane % lanes_per_group;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += block_size) {
        const uint begin = block + simd_lane * values_per_thread;
        const uint count = begin < input_columns ? min(values_per_thread, input_columns - begin) : 0;
        float input_values[16];
        float input_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < values_per_thread; index += 4) {
            if (index + 3 < count) {
                const ushort4 packed_input = *((device const ushort4 *)(input + begin + index));
                const float4 values = as_type<float4>(uint4(packed_input) << 16);
                input_values[index] = values.x;
                input_values[index + 1] = values.y * 0.0625f;
                input_values[index + 2] = values.z * 0.00390625f;
                input_values[index + 3] = values.w * 0.000244140625f;
                input_sum += values.x + values.y + values.z + values.w;
            } else {
                for (uint tail = index; tail < values_per_thread; ++tail) {
                    const float value = tail < count ? zllm_bf16_to_f32(input[begin + tail]) : 0.0f;
                    input_values[tail] = value / float(1u << ((tail & 3) * 4));
                    input_sum += value;
                }
            }
        }
#pragma unroll
        for (uint output_index = 0; output_index < 4; ++output_index) {
            const uint output_column = output_base + output_index;
            if (output_column >= output_columns || count == 0) continue;
            const device ushort *weight = (const device ushort *)(codes + ulong(output_column) * weight_bytes);
            float code_sum = 0.0f;
            if (values_per_thread == 16 && count == 16) {
                const ushort4 packed_values = *((device const ushort4 *)(weight + (begin >> 2)));
#pragma unroll
                for (uint index = 0; index < 4; ++index) {
                    const ushort packed = packed_values[index];
                    code_sum +=
                        input_values[index * 4] * float(packed & 0x000f) +
                        input_values[index * 4 + 1] * float(packed & 0x00f0) +
                        input_values[index * 4 + 2] * float(packed & 0x0f00) +
                        input_values[index * 4 + 3] * float(packed & 0xf000);
                }
            } else {
                for (uint index = 0; index < values_per_thread / 4; ++index) {
                    if (index * 4 >= count) break;
                    const ushort packed = weight[(begin >> 2) + index];
                    code_sum +=
                        input_values[index * 4] * float(packed & 0x000f) +
                        input_values[index * 4 + 1] * float(packed & 0x00f0) +
                        input_values[index * 4 + 2] * float(packed & 0x0f00) +
                        input_values[index * 4 + 3] * float(packed & 0xf000);
                }
            }
            const ulong parameter = ulong(output_column) * groups + begin / group_size;
            float weight_scale = simd_lane == scale_lane ? zllm_bf16_to_f32(scales[parameter]) : 0.0f;
            float weight_bias = simd_lane == scale_lane ? zllm_bf16_to_f32(biases[parameter]) : 0.0f;
            weight_scale = simd_shuffle(weight_scale, scale_lane);
            weight_bias = simd_shuffle(weight_bias, scale_lane);
            result[output_index] += weight_scale * code_sum + input_sum * weight_bias;
        }
    }
#pragma unroll
    for (uint output_index = 0; output_index < 4; ++output_index) {
        const float gate_value = simd_sum(result[output_index]);
        const uint output_column = output_base + output_index;
        if (simd_lane == 0 && output_column < output_columns) {
            const float rounded_gate = zllm_bf16_to_f32(zllm_f32_to_bf16(gate_value));
            output[output_column] = zllm_f32_to_bf16(gated_activation_value(
                rounded_gate,
                zllm_bf16_to_f32(up[output_column]),
                activation_kind,
                activation_alpha,
                activation_limit));
        }
    }
}
kernel void mlx_affine_gemv_bf16_u8(
    device const ushort *input [[buffer(0)]], device const uchar *codes [[buffer(1)]],
    device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
    device ushort *output [[buffer(4)]], constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]], constant uint &output_columns [[buffer(7)]],
    constant uint &bits [[buffer(8)]], constant uint &group_size [[buffer(9)]],
    constant uint &scale_dtype [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    (void)bits;
    (void)scale_dtype;
    const uint input_row = group.y;
    if (input_row >= input_rows) return;
    const ulong groups = ulong(input_columns) / group_size;
    const ulong input_base = ulong(input_row) * input_columns;
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    const uint output_base = group.x * 32 + simd_group * 8;
    float result[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += 512) {
        const uint begin = block + simd_lane * 16;
        float input_values[16];
        float input_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < 16; index += 4) {
            if (begin + index + 3 < input_columns) {
                const ushort4 packed_input = *((device const ushort4 *)(input + input_base + begin + index));
                const float4 values = as_type<float4>(uint4(packed_input) << 16);
                input_values[index] = values.x;
                input_values[index + 1] = values.y;
                input_values[index + 2] = values.z;
                input_values[index + 3] = values.w;
                input_sum += values.x + values.y + values.z + values.w;
            } else {
                for (uint tail = index; tail < 16; ++tail) {
                    const uint column = begin + tail;
                    const float value = column < input_columns ? zllm_bf16_to_f32(input[input_base + column]) : 0.0f;
                    input_values[tail] = value;
                    input_sum += value;
                }
            }
        }
#pragma unroll
        for (uint output_index = 0; output_index < 8; ++output_index) {
            const uint output_column = output_base + output_index;
            if (output_column >= output_columns || begin >= input_columns) continue;
            const device uchar *weight = codes + ulong(output_column) * input_columns + begin;
            float code_sum = 0.0f;
#pragma unroll
            for (uint index = 0; index < 16; index += 4) {
                if (begin + index + 3 < input_columns) {
                    const uchar4 packed_codes = *((device const uchar4 *)(weight + index));
                    const float4 values = float4(
                        input_values[index], input_values[index + 1],
                        input_values[index + 2], input_values[index + 3]);
                    code_sum += dot(values, float4(packed_codes));
                } else {
                    for (uint tail = index; tail < 16 && begin + tail < input_columns; ++tail) {
                        code_sum += input_values[tail] * float(weight[tail]);
                    }
                }
            }
            const ulong parameter = ulong(output_column) * groups + begin / group_size;
            float scale = (simd_lane & 3) == 0 ? zllm_bf16_to_f32(scales[parameter]) : 0.0f;
            float bias = (simd_lane & 3) == 0 ? zllm_bf16_to_f32(biases[parameter]) : 0.0f;
            scale = simd_shuffle(scale, simd_lane & ~3u);
            bias = simd_shuffle(bias, simd_lane & ~3u);
            result[output_index] += scale * code_sum + input_sum * bias;
        }
    }
#pragma unroll
    for (uint output_index = 0; output_index < 8; ++output_index) {
        result[output_index] = simd_sum(result[output_index]);
        const uint output_column = output_base + output_index;
        if (simd_lane == 0 && output_column < output_columns) {
            output[ulong(input_row) * output_columns + output_column] = zllm_f32_to_bf16(result[output_index]);
        }
    }
}
kernel void mlx_affine_deinterleave_pair(
    device const uchar *source [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &block_bytes [[buffer(2)]],
    constant uint &output_bytes [[buffer(3)]],
    constant uint &role [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= output_bytes) return;
    const uint block = id / block_bytes;
    const uint within = id - block * block_bytes;
    output[id] = source[(ulong(block) * 2ul + ulong(role)) * ulong(block_bytes) + within];
}
kernel void mlx_affine_dequant_interleaved_f16_u8(
    device const uchar *codes [[buffer(0)]],
    device const ushort *scales [[buffer(1)]],
    device const ushort *biases [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &packed_word_count [[buffer(4)]],
    constant uint &input_columns [[buffer(5)]],
    constant uint &role [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= packed_word_count) return;
    const ulong standard_byte = ulong(id) * 4ul;
    const ulong source_byte = ((standard_byte >> 4) * 2ul + ulong(role)) * 16ul + (standard_byte & 15ul);
    const uint word = *((device const uint *)(codes + source_byte));
    const ulong first_value = ulong(id) * 4ul;
    const ulong row = first_value / input_columns;
    const ulong first_column = first_value - row * input_columns;
    const ulong parameter_columns = ulong(input_columns) / 64ul;
    const ulong parameter = (row * parameter_columns + first_column / 64ul) * 2ul + role;
    const float scale = zllm_bf16_to_f32(scales[parameter]);
    const float bias = zllm_bf16_to_f32(biases[parameter]);
#pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const ulong column = first_column + item;
        if (column < input_columns) {
            const uint code = (word >> (item * 8)) & 255u;
            output[row * input_columns + column] = half(zllm_bf16_to_f32(
                zllm_f32_to_bf16(scale * float(code) + bias)));
        }
    }
}
kernel void mlx_affine_qmm_bf16_u8_interleaved(
    device const ushort *input [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device ushort *output [[buffer(4)]],
    constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]],
    constant uint &output_columns [[buffer(7)]],
    constant uint &role [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup half input_tile[512];
    threadgroup half weight_tile[512];
    simdgroup_float8x8 accumulator_0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 accumulator_1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 accumulator_2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 accumulator_3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint row_base = group.y * 32;
    const uint output_base = group.x * 32;
    const ulong packed_columns = (ulong(input_columns) + 3ul) / 4ul;
    const ulong parameter_columns = ulong(input_columns) / 64ul;

    for (uint input_base = 0; input_base < input_columns; input_base += 16) {
        const uint begin = thread_index * 4;
        for (uint item = 0; item < 4; ++item) {
            const uint flat = begin + item;
            const uint local_row = flat >> 4;
            const uint local_column = flat & 15;
            const uint row = row_base + local_row;
            const uint input_column = input_base + local_column;
            input_tile[flat] = row < input_rows && input_column < input_columns
                ? half(zllm_bf16_to_f32(input[ulong(row) * input_columns + input_column]))
                : half(0.0h);

            const uint weight_input = input_base + (flat >> 5);
            const uint output_column = output_base + (flat & 31);
            if (output_column < output_columns && weight_input < input_columns) {
                const ulong standard_word = ulong(output_column) * packed_columns + weight_input / 4;
                const ulong standard_byte = standard_word * 4ul;
                const ulong source_byte = ((standard_byte >> 4) * 2ul + ulong(role)) * 16ul + (standard_byte & 15ul);
                const uint word = *((device const uint *)(codes + source_byte));
                const uint code = (word >> ((weight_input & 3u) * 8)) & 255u;
                const ulong parameter = (ulong(output_column) * parameter_columns + weight_input / 64) * 2ul + role;
                const float scale = zllm_bf16_to_f32(scales[parameter]);
                const float bias = zllm_bf16_to_f32(biases[parameter]);
                weight_tile[flat] = half(zllm_bf16_to_f32(zllm_f32_to_bf16(scale * float(code) + bias)));
            } else {
                weight_tile[flat] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 16; k += 8) {
            simdgroup_half8x8 input_matrix_0;
            simdgroup_half8x8 input_matrix_1;
            simdgroup_half8x8 input_matrix_2;
            simdgroup_half8x8 input_matrix_3;
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(input_matrix_0, input_tile + k, 16);
            simdgroup_load(input_matrix_1, input_tile + 128 + k, 16);
            simdgroup_load(input_matrix_2, input_tile + 256 + k, 16);
            simdgroup_load(input_matrix_3, input_tile + 384 + k, 16);
            simdgroup_load(weight_matrix, weight_tile + k * 32 + simd_index * 8, 32);
            simdgroup_multiply_accumulate(accumulator_0, input_matrix_0, weight_matrix, accumulator_0);
            simdgroup_multiply_accumulate(accumulator_1, input_matrix_1, weight_matrix, accumulator_1);
            simdgroup_multiply_accumulate(accumulator_2, input_matrix_2, weight_matrix, accumulator_2);
            simdgroup_multiply_accumulate(accumulator_3, input_matrix_3, weight_matrix, accumulator_3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float result_tile[1024];
    simdgroup_store(accumulator_0, result_tile + simd_index * 8, 32);
    simdgroup_store(accumulator_1, result_tile + 256 + simd_index * 8, 32);
    simdgroup_store(accumulator_2, result_tile + 512 + simd_index * 8, 32);
    simdgroup_store(accumulator_3, result_tile + 768 + simd_index * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint begin = thread_index * 8;
    for (uint item = 0; item < 8; ++item) {
        const uint flat = begin + item;
        const uint row = row_base + (flat >> 5);
        const uint output_column = output_base + (flat & 31);
        if (row < input_rows && output_column < output_columns) {
            output[ulong(row) * output_columns + output_column] = zllm_f32_to_bf16(result_tile[flat]);
        }
    }
}
kernel void mlx_affine_gemv_bf16_u8_interleaved(
    device const ushort *input [[buffer(0)]], device const uchar *codes [[buffer(1)]],
    device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
    device ushort *output [[buffer(4)]], constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]], constant uint &output_columns [[buffer(7)]],
    constant uint &bits [[buffer(8)]], constant uint &group_size [[buffer(9)]],
    constant uint &scale_dtype [[buffer(10)]], constant uint &role [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]])
{
    (void)bits;
    (void)scale_dtype;
    const uint input_row = group.y;
    if (input_row >= input_rows) return;
    const ulong groups = ulong(input_columns) / group_size;
    const ulong input_base = ulong(input_row) * input_columns;
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    const uint output_base = group.x * 32 + simd_group * 8;
    float result[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += 512) {
        const uint begin = block + simd_lane * 16;
        float input_values[16];
        float input_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < 16; index += 4) {
            if (begin + index + 3 < input_columns) {
                const ushort4 packed_input = *((device const ushort4 *)(input + input_base + begin + index));
                const float4 values = as_type<float4>(uint4(packed_input) << 16);
                input_values[index] = values.x;
                input_values[index + 1] = values.y;
                input_values[index + 2] = values.z;
                input_values[index + 3] = values.w;
                input_sum += values.x + values.y + values.z + values.w;
            } else {
                for (uint tail = index; tail < 16; ++tail) {
                    const uint column = begin + tail;
                    const float value = column < input_columns ? zllm_bf16_to_f32(input[input_base + column]) : 0.0f;
                    input_values[tail] = value;
                    input_sum += value;
                }
            }
        }
#pragma unroll
        for (uint output_index = 0; output_index < 8; ++output_index) {
            const uint output_column = output_base + output_index;
            if (output_column >= output_columns || begin >= input_columns) continue;
            const ulong linear = ulong(output_column) * input_columns + begin;
            const device uchar *weight = codes + (linear >> 4) * 32ul + ulong(role) * 16ul;
            float code_sum = 0.0f;
#pragma unroll
            for (uint index = 0; index < 16; index += 4) {
                if (begin + index + 3 < input_columns) {
                    const uchar4 packed_codes = *((device const uchar4 *)(weight + index));
                    const float4 values = float4(
                        input_values[index], input_values[index + 1],
                        input_values[index + 2], input_values[index + 3]);
                    code_sum += dot(values, float4(packed_codes));
                } else {
                    for (uint tail = index; tail < 16 && begin + tail < input_columns; ++tail) {
                        code_sum += input_values[tail] * float(weight[tail]);
                    }
                }
            }
            const ulong parameter = (ulong(output_column) * groups + begin / group_size) * 2ul + role;
            float scale = (simd_lane & 3) == 0 ? zllm_bf16_to_f32(scales[parameter]) : 0.0f;
            float bias = (simd_lane & 3) == 0 ? zllm_bf16_to_f32(biases[parameter]) : 0.0f;
            scale = simd_shuffle(scale, simd_lane & ~3u);
            bias = simd_shuffle(bias, simd_lane & ~3u);
            result[output_index] += scale * code_sum + input_sum * bias;
        }
    }
#pragma unroll
    for (uint output_index = 0; output_index < 8; ++output_index) {
        result[output_index] = simd_sum(result[output_index]);
        const uint output_column = output_base + output_index;
        if (simd_lane == 0 && output_column < output_columns) {
            output[ulong(input_row) * output_columns + output_column] = zllm_f32_to_bf16(result[output_index]);
        }
    }
}
kernel void mlx_affine_gated_interleaved_gemv_bf16_u8(
    device const ushort *input [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device ushort *output [[buffer(4)]],
    constant uint &input_columns [[buffer(5)]],
    constant uint &output_columns [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    constant uint &activation_kind [[buffer(8)]],
    constant float &activation_alpha [[buffer(9)]],
    constant float &activation_limit [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const ulong groups = ulong(input_columns) / group_size;
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    const uint output_base = group * 16 + simd_group * 4;
    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += 512) {
        const uint begin = block + simd_lane * 16;
        float input_values[16];
        float input_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < 16; index += 4) {
            if (begin + index + 3 < input_columns) {
                const ushort4 packed_input = *((device const ushort4 *)(input + begin + index));
                const float4 values = as_type<float4>(uint4(packed_input) << 16);
                input_values[index] = values.x;
                input_values[index + 1] = values.y;
                input_values[index + 2] = values.z;
                input_values[index + 3] = values.w;
                input_sum += values.x + values.y + values.z + values.w;
            } else {
                for (uint tail = index; tail < 16; ++tail) {
                    const uint column = begin + tail;
                    const float value = column < input_columns ? zllm_bf16_to_f32(input[column]) : 0.0f;
                    input_values[tail] = value;
                    input_sum += value;
                }
            }
        }
#pragma unroll
        for (uint output_index = 0; output_index < 4; ++output_index) {
            const uint output_column = output_base + output_index;
            if (output_column >= output_columns || begin >= input_columns) continue;
            const ulong linear = ulong(output_column) * input_columns + begin;
            const device uchar *gate_weight = codes + (linear >> 4) * 32ul;
            const device uchar *up_weight = gate_weight + 16;
            float gate_code_sum = 0.0f;
            float up_code_sum = 0.0f;
#pragma unroll
            for (uint index = 0; index < 16; index += 4) {
                if (begin + index + 3 < input_columns) {
                    const float4 values = float4(
                        input_values[index], input_values[index + 1],
                        input_values[index + 2], input_values[index + 3]);
                    gate_code_sum += dot(values, float4(*((device const uchar4 *)(gate_weight + index))));
                    up_code_sum += dot(values, float4(*((device const uchar4 *)(up_weight + index))));
                } else {
                    for (uint tail = index; tail < 16 && begin + tail < input_columns; ++tail) {
                        gate_code_sum += input_values[tail] * float(gate_weight[tail]);
                        up_code_sum += input_values[tail] * float(up_weight[tail]);
                    }
                }
            }
            const ulong parameter = (ulong(output_column) * groups + begin / group_size) * 2ul;
            const uint scale_lane = simd_lane & ~3u;
            float gate_scale = simd_lane == scale_lane ? zllm_bf16_to_f32(scales[parameter]) : 0.0f;
            float gate_bias = simd_lane == scale_lane ? zllm_bf16_to_f32(biases[parameter]) : 0.0f;
            float up_scale = simd_lane == scale_lane ? zllm_bf16_to_f32(scales[parameter + 1]) : 0.0f;
            float up_bias = simd_lane == scale_lane ? zllm_bf16_to_f32(biases[parameter + 1]) : 0.0f;
            gate_scale = simd_shuffle(gate_scale, scale_lane);
            gate_bias = simd_shuffle(gate_bias, scale_lane);
            up_scale = simd_shuffle(up_scale, scale_lane);
            up_bias = simd_shuffle(up_bias, scale_lane);
            gate_result[output_index] += gate_scale * gate_code_sum + input_sum * gate_bias;
            up_result[output_index] += up_scale * up_code_sum + input_sum * up_bias;
        }
    }
#pragma unroll
    for (uint output_index = 0; output_index < 4; ++output_index) {
        const float gate_value = simd_sum(gate_result[output_index]);
        const float up_value = simd_sum(up_result[output_index]);
        const uint output_column = output_base + output_index;
        if (simd_lane == 0 && output_column < output_columns) {
            const float rounded_gate = zllm_bf16_to_f32(zllm_f32_to_bf16(gate_value));
            const float rounded_up = zllm_bf16_to_f32(zllm_f32_to_bf16(up_value));
            output[output_column] = zllm_f32_to_bf16(gated_activation_value(
                rounded_gate, rounded_up, activation_kind, activation_alpha, activation_limit));
        }
    }
}
kernel void mlx_affine_gated_gemv_bf16_u8(
    device const ushort *input [[buffer(0)]],
    device const uchar *gate_codes [[buffer(1)]],
    device const ushort *gate_scales [[buffer(2)]],
    device const ushort *gate_biases [[buffer(3)]],
    device const uchar *up_codes [[buffer(4)]],
    device const ushort *up_scales [[buffer(5)]],
    device const ushort *up_biases [[buffer(6)]],
    device ushort *output [[buffer(7)]],
    constant uint &input_columns [[buffer(8)]],
    constant uint &output_columns [[buffer(9)]],
    constant uint &group_size [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &activation_alpha [[buffer(12)]],
    constant float &activation_limit [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const ulong groups = ulong(input_columns) / group_size;
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    const uint output_base = group * 16 + simd_group * 4;
    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += 512) {
        const uint begin = block + simd_lane * 16;
        float input_values[16];
        float input_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < 16; index += 4) {
            if (begin + index + 3 < input_columns) {
                const ushort4 packed_input = *((device const ushort4 *)(input + begin + index));
                const float4 values = as_type<float4>(uint4(packed_input) << 16);
                input_values[index] = values.x;
                input_values[index + 1] = values.y;
                input_values[index + 2] = values.z;
                input_values[index + 3] = values.w;
                input_sum += values.x + values.y + values.z + values.w;
            } else {
                for (uint tail = index; tail < 16; ++tail) {
                    const uint column = begin + tail;
                    const float value = column < input_columns ? zllm_bf16_to_f32(input[column]) : 0.0f;
                    input_values[tail] = value;
                    input_sum += value;
                }
            }
        }
#pragma unroll
        for (uint output_index = 0; output_index < 4; ++output_index) {
            const uint output_column = output_base + output_index;
            if (output_column >= output_columns || begin >= input_columns) continue;
            const device uchar *gate_weight = gate_codes + ulong(output_column) * input_columns + begin;
            const device uchar *up_weight = up_codes + ulong(output_column) * input_columns + begin;
            float gate_code_sum = 0.0f;
            float up_code_sum = 0.0f;
#pragma unroll
            for (uint index = 0; index < 16; index += 4) {
                if (begin + index + 3 < input_columns) {
                    const float4 values = float4(
                        input_values[index], input_values[index + 1],
                        input_values[index + 2], input_values[index + 3]);
                    gate_code_sum += dot(values, float4(*((device const uchar4 *)(gate_weight + index))));
                    up_code_sum += dot(values, float4(*((device const uchar4 *)(up_weight + index))));
                } else {
                    for (uint tail = index; tail < 16 && begin + tail < input_columns; ++tail) {
                        gate_code_sum += input_values[tail] * float(gate_weight[tail]);
                        up_code_sum += input_values[tail] * float(up_weight[tail]);
                    }
                }
            }
            const ulong parameter = ulong(output_column) * groups + begin / group_size;
            const uint scale_lane = simd_lane & ~3u;
            float gate_scale = simd_lane == scale_lane ? zllm_bf16_to_f32(gate_scales[parameter]) : 0.0f;
            float gate_bias = simd_lane == scale_lane ? zllm_bf16_to_f32(gate_biases[parameter]) : 0.0f;
            float up_scale = simd_lane == scale_lane ? zllm_bf16_to_f32(up_scales[parameter]) : 0.0f;
            float up_bias = simd_lane == scale_lane ? zllm_bf16_to_f32(up_biases[parameter]) : 0.0f;
            gate_scale = simd_shuffle(gate_scale, scale_lane);
            gate_bias = simd_shuffle(gate_bias, scale_lane);
            up_scale = simd_shuffle(up_scale, scale_lane);
            up_bias = simd_shuffle(up_bias, scale_lane);
            gate_result[output_index] += gate_scale * gate_code_sum + input_sum * gate_bias;
            up_result[output_index] += up_scale * up_code_sum + input_sum * up_bias;
        }
    }
#pragma unroll
    for (uint output_index = 0; output_index < 4; ++output_index) {
        const float gate_value = simd_sum(gate_result[output_index]);
        const float up_value = simd_sum(up_result[output_index]);
        const uint output_column = output_base + output_index;
        if (simd_lane == 0 && output_column < output_columns) {
            const float rounded_gate = zllm_bf16_to_f32(zllm_f32_to_bf16(gate_value));
            const float rounded_up = zllm_bf16_to_f32(zllm_f32_to_bf16(up_value));
            output[output_column] = zllm_f32_to_bf16(gated_activation_value(
                rounded_gate, rounded_up, activation_kind, activation_alpha, activation_limit));
        }
    }
}
kernel void mlx_affine_triple_gemv_f16_u4(
    device const half *input [[buffer(0)]],
    device const uchar *first_codes [[buffer(1)]],
    device const ushort *first_scales [[buffer(2)]],
    device const ushort *first_biases [[buffer(3)]],
    device const uchar *second_codes [[buffer(4)]],
    device const ushort *second_scales [[buffer(5)]],
    device const ushort *second_biases [[buffer(6)]],
    device const uchar *third_codes [[buffer(7)]],
    device const ushort *third_scales [[buffer(8)]],
    device const ushort *third_biases [[buffer(9)]],
    device half *first_output [[buffer(10)]],
    device half *second_output [[buffer(11)]],
    device half *third_output [[buffer(12)]],
    constant uint &input_columns [[buffer(13)]],
    constant uint &first_columns [[buffer(14)]],
    constant uint &second_columns [[buffer(15)]],
    constant uint &third_columns [[buffer(16)]],
    constant uint &group_size [[buffer(17)]],
    constant uint &scale_dtype [[buffer(18)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    // 每 SIMD 组 4 行、每 threadgroup 8 行;行落在哪一段就读写哪一路权重/输出
    const uint output_base = group * 8 + simd_group * 4;
    const uint values_per_thread = input_columns % 16 == 0 ? 16 : 8;
    const uint block_size = values_per_thread * 32;
    const ulong packed_columns = (ulong(input_columns) + 7) / 8;
    const ulong weight_bytes = packed_columns * 4;
    const ulong groups = ulong(input_columns) / group_size;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += block_size) {
        const uint begin = block + simd_lane * values_per_thread;
        const uint count = begin < input_columns ? min(values_per_thread, input_columns - begin) : 0;
        float input_values[16];
        float input_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < values_per_thread; index += 4) {
            if (index + 3 < count) {
                const half4 packed_input = *((device const half4 *)(input + begin + index));
                input_values[index] = float(packed_input.x);
                input_values[index + 1] = float(packed_input.y) * 0.0625f;
                input_values[index + 2] = float(packed_input.z) * 0.00390625f;
                input_values[index + 3] = float(packed_input.w) * 0.000244140625f;
                input_sum += float(packed_input.x) + float(packed_input.y) + float(packed_input.z) + float(packed_input.w);
            } else {
                for (uint tail = index; tail < values_per_thread; ++tail) {
                    const float value = tail < count ? float(input[begin + tail]) : 0.0f;
                    input_values[tail] = value / float(1u << ((tail & 3) * 4));
                    input_sum += value;
                }
            }
        }
#pragma unroll
        for (uint output_index = 0; output_index < 4; ++output_index) {
            const uint row = output_base + output_index;
            if (row >= first_columns + second_columns + third_columns || count == 0) continue;
            const bool is_first = row < first_columns;
            const bool is_second = !is_first && row < first_columns + second_columns;
            const uint local_row = is_first ? row : (is_second ? row - first_columns : row - first_columns - second_columns);
            const device uchar *codes = is_first ? first_codes : (is_second ? second_codes : third_codes);
            const device ushort *scales = is_first ? first_scales : (is_second ? second_scales : third_scales);
            const device ushort *biases = is_first ? first_biases : (is_second ? second_biases : third_biases);
            const device ushort *weight_row = (const device ushort *)(codes + ulong(local_row) * weight_bytes);
            float code_sum = 0.0f;
            if (values_per_thread == 16 && count == 16) {
                const ushort4 packed_values = *((device const ushort4 *)(weight_row + (begin >> 2)));
#pragma unroll
                for (uint index = 0; index < 4; ++index) {
                    const ushort packed_word = packed_values[index];
                    code_sum +=
                        input_values[index * 4] * float(packed_word & 0x000f) +
                        input_values[index * 4 + 1] * float(packed_word & 0x00f0) +
                        input_values[index * 4 + 2] * float(packed_word & 0x0f00) +
                        input_values[index * 4 + 3] * float(packed_word & 0xf000);
                }
            } else {
                for (uint index = 0; index < values_per_thread / 4; ++index) {
                    if (index * 4 >= count) break;
                    const ushort packed_word = weight_row[(begin >> 2) + index];
                    code_sum +=
                        input_values[index * 4] * float(packed_word & 0x000f) +
                        input_values[index * 4 + 1] * float(packed_word & 0x00f0) +
                        input_values[index * 4 + 2] * float(packed_word & 0x0f00) +
                        input_values[index * 4 + 3] * float(packed_word & 0xf000);
                }
            }
            const ulong parameter = ulong(local_row) * groups + begin / group_size;
            result[output_index] += w4a16_scale((device const uchar *)scales, parameter, scale_dtype) * code_sum + input_sum * w4a16_scale((device const uchar *)biases, parameter, scale_dtype);
        }
    }
    for (uint output_index = 0; output_index < 4; ++output_index) {
        const float value = simd_sum(result[output_index]);
        const uint row = output_base + output_index;
        if (simd_lane == 0 && row < first_columns) {
            first_output[row] = half(value);
        } else if (simd_lane == 0 && row < first_columns + second_columns) {
            second_output[row - first_columns] = half(value);
        } else if (simd_lane == 0 && row < first_columns + second_columns + third_columns) {
            third_output[row - first_columns - second_columns] = half(value);
        }
    }
}
// MLX qmv_fast 结构移植(u4):4 行/SIMD x 8 行/TG,lane 固定 16 列,
// 指针循环前一次预偏移、主循环零守卫零重寻址;尾块单独一段,只付一次分支成本。
// 行内 32 lane 各覆盖不同列段,simd_sum 归约;scale 由 4 lane 一组直读(L1 广播)。
kernel void mlx_affine_qmv_f16_ml(
    device const half *input [[buffer(0)]], device const uchar *codes [[buffer(1)]],
    device const uchar *scales [[buffer(2)]], device const uchar *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &input_columns [[buffer(5)]], constant uint &output_columns [[buffer(6)]],
    constant uint &scale_dtype [[buffer(7)]], constant uint &scale_bytes [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    const uint row_base = group.y * 8 + simd_group * 4;
    const uint groups_per_row = input_columns / 64u;
    const uint row_bytes = input_columns / 2u;
    const device const half *x = input + ulong(simd_lane) * 16u;
    const device const uchar *w = codes + ulong(row_base) * row_bytes + ulong(simd_lane) * 8u;
    const device const uchar *sc = scales + (ulong(row_base) * groups_per_row + simd_lane / 4u) * scale_bytes;
    const device const uchar *bi = biases + (ulong(row_base) * groups_per_row + simd_lane / 4u) * scale_bytes;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint full_blocks = input_columns / 512u;
    const uint remain = input_columns - full_blocks * 512u;

    // 对齐主循环:每块 512 列;lane 的 16 列 = 4 个 ushort 词,预缩放按词内位置
    for (uint block = 0; block < full_blocks; ++block) {
        const half4 x0 = *((device const half4 *)x);
        const half4 x1 = *((device const half4 *)(x + 4));
        const half4 x2 = *((device const half4 *)(x + 8));
        const half4 x3 = *((device const half4 *)(x + 12));
        const float raw[16] = {float(x0.x), float(x0.y), float(x0.z), float(x0.w), float(x1.x), float(x1.y), float(x1.z), float(x1.w),
                               float(x2.x), float(x2.y), float(x2.z), float(x2.w), float(x3.x), float(x3.y), float(x3.z), float(x3.w)};
        const float input_sum = raw[0] + raw[1] + raw[2] + raw[3] + raw[4] + raw[5] + raw[6] + raw[7] + raw[8] + raw[9] + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
        // 预缩放按 ushort 内位置 p:xs[4u+p] = raw × 16^-p
        const float xs[16] = {
            raw[0], raw[1] * 0.0625f, raw[2] * 0.00390625f, raw[3] * 0.000244140625f,
            raw[4], raw[5] * 0.0625f, raw[6] * 0.00390625f, raw[7] * 0.000244140625f,
            raw[8], raw[9] * 0.0625f, raw[10] * 0.00390625f, raw[11] * 0.000244140625f,
            raw[12], raw[13] * 0.0625f, raw[14] * 0.00390625f, raw[15] * 0.000244140625f,
        };
        // 每「行」有独立 scales/biases:先读齐本块 4 行参数再进热循环
        float scales_row[4];
        float biases_row[4];
#pragma unroll
        for (uint row = 0; row < 4; ++row) {
            scales_row[row] = w4a16_scale(sc + ulong(row) * groups_per_row * scale_bytes, 0, scale_dtype);
            biases_row[row] = w4a16_scale(bi + ulong(row) * groups_per_row * scale_bytes, 0, scale_dtype);
        }
        float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (uint row = 0; row < 4; ++row) {
            if (row_base + row >= output_columns) break;
            const ushort2 wa = *((device const ushort2 *)(w + ulong(row) * row_bytes));
            const ushort2 wb = *((device const ushort2 *)(w + ulong(row) * row_bytes + 4));
            partial[row] =
                xs[0] * float(wa.x & 0x000f) + xs[1] * float(wa.x & 0x00f0) +
                xs[2] * float(wa.x & 0x0f00) + xs[3] * float(wa.x & 0xf000) +
                xs[4] * float(wa.y & 0x000f) + xs[5] * float(wa.y & 0x00f0) +
                xs[6] * float(wa.y & 0x0f00) + xs[7] * float(wa.y & 0xf000) +
                xs[8] * float(wb.x & 0x000f) + xs[9] * float(wb.x & 0x00f0) +
                xs[10] * float(wb.x & 0x0f00) + xs[11] * float(wb.x & 0xf000) +
                xs[12] * float(wb.y & 0x000f) + xs[13] * float(wb.y & 0x00f0) +
                xs[14] * float(wb.y & 0x0f00) + xs[15] * float(wb.y & 0xf000);
            result[row] += scales_row[row] * partial[row] + input_sum * biases_row[row];
        }
        x += 512;
        w += 256;
        sc += 8u * scale_bytes;
        bi += 8u * scale_bytes;
    }
    // 尾块:剩余列 < 512,逐词守卫;只执行一次
    if (remain != 0) {
        const uint lane_columns = remain > simd_lane * 16u ? min(16u, remain - simd_lane * 16u) : 0u;
        float raw[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint index = 0; index < lane_columns; ++index) {
            raw[index] = float(x[index]);
        }
        const float input_sum = raw[0] + raw[1] + raw[2] + raw[3] + raw[4] + raw[5] + raw[6] + raw[7] + raw[8] + raw[9] + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
        const float xs[16] = {
            raw[0], raw[1] * 0.0625f, raw[2] * 0.00390625f, raw[3] * 0.000244140625f,
            raw[4], raw[5] * 0.0625f, raw[6] * 0.00390625f, raw[7] * 0.000244140625f,
            raw[8], raw[9] * 0.0625f, raw[10] * 0.00390625f, raw[11] * 0.000244140625f,
            raw[12], raw[13] * 0.0625f, raw[14] * 0.00390625f, raw[15] * 0.000244140625f,
        };
        float scales_row[4];
        float biases_row[4];
#pragma unroll
        for (uint row = 0; row < 4; ++row) {
            scales_row[row] = w4a16_scale(sc + ulong(row) * groups_per_row * scale_bytes, 0, scale_dtype);
            biases_row[row] = w4a16_scale(bi + ulong(row) * groups_per_row * scale_bytes, 0, scale_dtype);
        }
#pragma unroll
        for (uint row = 0; row < 4; ++row) {
            if (row_base + row >= output_columns || lane_columns == 0) continue;
            const device const ushort *words = (const device ushort *)(w + ulong(row) * row_bytes);
            float code_sum = 0.0f;
            for (uint word = 0; word < 4; ++word) {
                if (word * 4 >= lane_columns) break;
                const ushort packed_word = words[word];
                code_sum +=
                    xs[word * 4] * float(packed_word & 0x000f) +
                    xs[word * 4 + 1] * float(packed_word & 0x00f0) +
                    xs[word * 4 + 2] * float(packed_word & 0x0f00) +
                    xs[word * 4 + 3] * float(packed_word & 0xf000);
            }
            result[row] += scales_row[row] * code_sum + input_sum * biases_row[row];
        }
    }
#pragma unroll
    for (uint row = 0; row < 4; ++row) {
        result[row] = simd_sum(result[row]);
        if (simd_lane == 0 && row_base + row < output_columns) {
            output[row_base + row] = half(result[row]);
        }
    }
}
kernel void mlx_affine_gated_gemv_f16_u4(
    device const half *input [[buffer(0)]],
    device const uchar *gate_codes [[buffer(1)]],
    device const ushort *gate_scales [[buffer(2)]],
    device const ushort *gate_biases [[buffer(3)]],
    device const uchar *up_codes [[buffer(4)]],
    device const ushort *up_scales [[buffer(5)]],
    device const ushort *up_biases [[buffer(6)]],
    device half *output [[buffer(7)]],
    constant uint &input_columns [[buffer(8)]],
    constant uint &output_columns [[buffer(9)]],
    constant uint &group_size [[buffer(10)]],
    constant uint &scale_dtype [[buffer(11)]],
    constant uint &activation_kind [[buffer(12)]],
    constant float &activation_alpha [[buffer(13)]],
    constant float &activation_limit [[buffer(14)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    // 每 SIMD 组 4 列、每 threadgroup 8 行;gate/up 两路权重共享同一份输入加载
    const uint output_base = group * 8 + simd_group * 4;
    // 16 值快路径只需列数对齐 16,尾块由 count 归零自然跳过
    const uint values_per_thread = input_columns % 16 == 0 ? 16 : 8;
    const uint block_size = values_per_thread * 32;
    const ulong packed_columns = (ulong(input_columns) + 7) / 8;
    const ulong weight_bytes = packed_columns * 4;
    const ulong groups = ulong(input_columns) / group_size;
    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += block_size) {
        const uint begin = block + simd_lane * values_per_thread;
        const uint count = begin < input_columns ? min(values_per_thread, input_columns - begin) : 0;
        float input_values[16];
        float input_sum = 0.0f;
#pragma unroll
        for (uint index = 0; index < values_per_thread; index += 4) {
            if (index + 3 < count) {
                const half4 packed_input = *((device const half4 *)(input + begin + index));
                input_values[index] = float(packed_input.x);
                input_values[index + 1] = float(packed_input.y) * 0.0625f;
                input_values[index + 2] = float(packed_input.z) * 0.00390625f;
                input_values[index + 3] = float(packed_input.w) * 0.000244140625f;
                input_sum += float(packed_input.x) + float(packed_input.y) + float(packed_input.z) + float(packed_input.w);
            } else {
                for (uint tail = index; tail < values_per_thread; ++tail) {
                    const float value = tail < count ? float(input[begin + tail]) : 0.0f;
                    input_values[tail] = value / float(1u << ((tail & 3) * 4));
                    input_sum += value;
                }
            }
        }
#pragma unroll
        for (uint output_index = 0; output_index < 4; ++output_index) {
            const uint output_column = output_base + output_index;
            if (output_column >= output_columns || count == 0) continue;
            const device ushort *gate_row = (const device ushort *)(gate_codes + ulong(output_column) * weight_bytes);
            const device ushort *up_row = (const device ushort *)(up_codes + ulong(output_column) * weight_bytes);
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            if (values_per_thread == 16 && count == 16) {
                const ushort4 gate_words = *((device const ushort4 *)(gate_row + (begin >> 2)));
                const ushort4 up_words = *((device const ushort4 *)(up_row + (begin >> 2)));
#pragma unroll
                for (uint index = 0; index < 4; ++index) {
                    const ushort gate_word = gate_words[index];
                    gate_sum +=
                        input_values[index * 4] * float(gate_word & 0x000f) +
                        input_values[index * 4 + 1] * float(gate_word & 0x00f0) +
                        input_values[index * 4 + 2] * float(gate_word & 0x0f00) +
                        input_values[index * 4 + 3] * float(gate_word & 0xf000);
                    const ushort up_word = up_words[index];
                    up_sum +=
                        input_values[index * 4] * float(up_word & 0x000f) +
                        input_values[index * 4 + 1] * float(up_word & 0x00f0) +
                        input_values[index * 4 + 2] * float(up_word & 0x0f00) +
                        input_values[index * 4 + 3] * float(up_word & 0xf000);
                }
            } else {
                for (uint index = 0; index < values_per_thread / 4; ++index) {
                    if (index * 4 >= count) break;
                    const ushort gate_word = gate_row[(begin >> 2) + index];
                    gate_sum +=
                        input_values[index * 4] * float(gate_word & 0x000f) +
                        input_values[index * 4 + 1] * float(gate_word & 0x00f0) +
                        input_values[index * 4 + 2] * float(gate_word & 0x0f00) +
                        input_values[index * 4 + 3] * float(gate_word & 0xf000);
                    const ushort up_word = up_row[(begin >> 2) + index];
                    up_sum +=
                        input_values[index * 4] * float(up_word & 0x000f) +
                        input_values[index * 4 + 1] * float(up_word & 0x00f0) +
                        input_values[index * 4 + 2] * float(up_word & 0x0f00) +
                        input_values[index * 4 + 3] * float(up_word & 0xf000);
                }
            }
            const ulong parameter = ulong(output_column) * groups + begin / group_size;
            gate_result[output_index] += w4a16_scale((device const uchar *)gate_scales, parameter, scale_dtype) * gate_sum + input_sum * w4a16_scale((device const uchar *)gate_biases, parameter, scale_dtype);
            up_result[output_index] += w4a16_scale((device const uchar *)up_scales, parameter, scale_dtype) * up_sum + input_sum * w4a16_scale((device const uchar *)up_biases, parameter, scale_dtype);
        }
    }
    // 与顺序链一致:gate/up 各过 F16 边界再进激活,激活输出 finite_f16
    for (uint output_index = 0; output_index < 4; ++output_index) {
        const float gate_value = float(half(simd_sum(gate_result[output_index])));
        const float up_value = float(half(simd_sum(up_result[output_index])));
        const uint output_column = output_base + output_index;
        if (simd_lane == 0 && output_column < output_columns) {
            output[output_column] = finite_f16(gated_activation_value(gate_value, up_value, activation_kind, activation_alpha, activation_limit));
        }
    }
}
kernel void mlx_affine_gemv_bf16(
    device const ushort *input [[buffer(0)]], device const uint *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]], device const uchar *biases [[buffer(3)]],
    device ushort *output [[buffer(4)]], constant uint &input_rows [[buffer(5)]],
    constant uint &input_columns [[buffer(6)]], constant uint &output_columns [[buffer(7)]],
    constant uint &bits [[buffer(8)]], constant uint &group_size [[buffer(9)]],
    constant uint &scale_dtype [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint input_row = group.y;
    if (input_row >= input_rows) return;
    const uint values_per_word = 32 / bits;
    const uint mask = (1u << bits) - 1u;
    const ulong packed_columns = (ulong(input_columns) + values_per_word - 1) / values_per_word;
    const ulong groups = ulong(input_columns) / group_size;
    const ulong input_base = ulong(input_row) * input_columns;
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    const uint output_base = group.x * 8 + simd_group * 4;
    const uint values_per_thread = input_columns % 512 == 0
        ? (bits == 4 ? 16 : 8)
        : (bits == 4 ? 8 : 4);
    const uint block_size = values_per_thread * 32;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < input_columns; block += block_size) {
        const uint begin = block + simd_lane * values_per_thread;
        const uint count = begin < input_columns ? min(values_per_thread, input_columns - begin) : 0;
        float input_values[16];
        float input_sum = 0.0f;
        for (uint index = 0; index < values_per_thread; ++index) {
            const float value = index < count ? zllm_bf16_to_f32(input[input_base + begin + index]) : 0.0f;
            input_values[index] = value;
            input_sum += value;
        }
        for (uint output_index = 0; output_index < 4; ++output_index) {
            const uint output_column = output_base + output_index;
            if (output_column >= output_columns || count == 0) continue;
            float code_sum = 0.0f;
            for (uint index = 0; index < values_per_thread; ++index) {
                if (index >= count) break;
                const uint column = begin + index;
                const uint word = packed[ulong(output_column) * packed_columns + column / values_per_word];
                const uint code = (word >> ((column % values_per_word) * bits)) & mask;
                code_sum += input_values[index] * float(code);
            }
            const ulong parameter = ulong(output_column) * groups + begin / group_size;
            const float scale = w4a16_scale(scales, parameter, scale_dtype);
            const float bias = w4a16_scale(biases, parameter, scale_dtype);
            result[output_index] += scale * code_sum + input_sum * bias;
        }
    }
    for (uint output_index = 0; output_index < 4; ++output_index) {
        result[output_index] = simd_sum(result[output_index]);
        const uint output_column = output_base + output_index;
        if (simd_lane == 0 && output_column < output_columns) {
            output[ulong(input_row) * output_columns + output_column] = zllm_f32_to_bf16(result[output_index]);
        }
    }
}
"#;

use crate::backend::metal::api as metal;

use super::dense::GatedActivation;
use super::{MTLSize, MetalContext, MetalTensor, MetalTensorDType, launch_1d, set_bytes, to_bf16_tensor, to_f16_tensor, validate_size, validate_u32};

pub(super) fn validate_w4a16_storage(name: &str, packed: &metal::Buffer, scales: &metal::Buffer, scale_dtype: u32, group_size: usize, rows: usize, cols: usize) -> Result<(), String> {
    if group_size == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("{name} shape=[{rows},{cols}] group_size={group_size} 不兼容"));
    }
    let scale_bytes = match scale_dtype {
        0 | 1 => 2,
        2 => 4,
        _ => return Err(format!("{name} scale dtype code {scale_dtype} 不受支持")),
    };
    let expected_packed = rows.checked_mul(cols.div_ceil(8)).and_then(|words| words.checked_mul(4)).ok_or_else(|| format!("{name} packed 大小溢出"))?;
    let expected_scales = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(scale_bytes)).ok_or_else(|| format!("{name} scale 大小溢出"))?;
    validate_size(&format!("{name} packed"), expected_packed, packed.length() as usize)?;
    validate_size(&format!("{name} scales"), expected_scales, scales.length() as usize)
}

/// MLX qmv_fast 结构移植版 u4 gemv(group=64,单行):小形状 ALU/寻址受限路径的
/// 结构性对照,热循环零守卫零重寻址。
#[allow(clippy::too_many_arguments)]
pub fn mlx_affine_qmv_f16_ml_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    packed: &metal::Buffer,
    scales: &metal::Buffer,
    biases: &metal::Buffer,
    scale_dtype: u32,
    group_size: usize,
    weight_rows: usize,
    weight_cols: usize,
) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != weight_cols || input.dtype != MetalTensorDType::F16 || group_size != 64 || !weight_cols.is_multiple_of(8) || weight_rows == 0 {
        return Err(format!("MLX qmv F16 U4 input=[{},{},{:?}] weight=[{weight_rows},{weight_cols}] group={group_size} 不兼容", input.rows, input.cols, input.dtype));
    }
    let scale_bytes: u32 = match scale_dtype {
        0 | 1 => 2,
        2 => 4,
        _ => return Err(format!("MLX qmv F16 U4 scale dtype code {scale_dtype} 不受支持")),
    };
    let expected_codes = weight_rows.checked_mul(weight_cols.div_ceil(8)).and_then(|words| words.checked_mul(4)).ok_or("MLX qmv F16 U4 code 大小溢出")?;
    let expected_parameters = weight_rows * (weight_cols / group_size) * scale_bytes as usize;
    validate_size("MLX qmv F16 U4 codes", expected_codes, packed.length() as usize)?;
    validate_size("MLX qmv F16 U4 scales", expected_parameters, scales.length() as usize)?;
    validate_size("MLX qmv F16 U4 biases", expected_parameters, biases.length() as usize)?;
    let input_columns = validate_u32("MLX qmv F16 U4 input columns", weight_cols)?;
    let output_columns = validate_u32("MLX qmv F16 U4 output columns", weight_rows)?;
    let output = ctx.tensor_zeros(input.rows, weight_rows);
    let pipeline = ctx.pipeline("mlx_affine_qmv_f16_ml")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("MLX qmv F16 U4 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(packed), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(biases), 0);
    encoder.set_buffer(4, Some(&output.buffer), 0);
    set_bytes(&encoder, 5, &input_columns);
    set_bytes(&encoder, 6, &output_columns);
    set_bytes(&encoder, 7, &scale_dtype);
    set_bytes(&encoder, 8, &scale_bytes);
    encoder.dispatch_thread_groups(MTLSize::new(1, weight_rows.div_ceil(8) as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{weight_cols}],weight=[{weight_rows},{weight_cols}],bits=4,ml-structure");
    ctx.commit_and_wait_profiled(&command, "mlx_affine_qmv_f16_ml", &shape, input.buffer.length() + packed.length() + scales.length() + biases.length(), output.buffer.length());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn mlx_affine_matmul_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    packed: &metal::Buffer,
    scales: &metal::Buffer,
    biases: &metal::Buffer,
    scale_dtype: u32,
    bits: usize,
    group_size: usize,
    weight_rows: usize,
    weight_cols: usize,
) -> Result<MetalTensor, String> {
    if input.cols != weight_cols || !matches!(bits, 4 | 8) || group_size == 0 || !weight_cols.is_multiple_of(group_size) {
        return Err(format!("MLX affine input=[{},{}] weight=[{weight_rows},{weight_cols}] bits={bits} group={group_size} 不兼容", input.rows, input.cols));
    }
    let parameter_bytes = match scale_dtype {
        0 | 1 => 2,
        2 => 4,
        _ => return Err(format!("MLX affine scale dtype code {scale_dtype} 不受支持")),
    };
    let expected_packed = weight_rows.checked_mul(weight_cols.div_ceil(32 / bits)).and_then(|n| n.checked_mul(4)).ok_or("MLX affine packed 大小溢出")?;
    let expected_parameters = weight_rows.checked_mul(weight_cols / group_size).and_then(|n| n.checked_mul(parameter_bytes)).ok_or("MLX affine 参数大小溢出")?;
    validate_size("MLX affine packed", expected_packed, packed.length() as usize)?;
    validate_size("MLX affine scales", expected_parameters, scales.length() as usize)?;
    validate_size("MLX affine biases", expected_parameters, biases.length() as usize)?;
    let input_rows = validate_u32("MLX affine input rows", input.rows)?;
    let input_columns = validate_u32("MLX affine input columns", weight_cols)?;
    let output_columns = validate_u32("MLX affine output columns", weight_rows)?;
    let bits = validate_u32("MLX affine bits", bits)?;
    let group_size = validate_u32("MLX affine group size", group_size)?;
    // prefill 的多行 F16 若继续走 GEMV，会按 token 行数重复扫描整份权重；
    // 只转换 activation 后复用原生量化 QMM，权重保持压缩常驻，输出恢复 F16。
    if input.dtype == MetalTensorDType::F16 && input.rows > 1 {
        let input = to_bf16_tensor(ctx, input)?;
        let output = mlx_affine_matmul_tensor_resident(ctx, &input, packed, scales, biases, scale_dtype, bits as usize, group_size as usize, weight_rows, weight_cols)?;
        return to_f16_tensor(ctx, &output);
    }
    // MLX qmv_fast 结构移植版:指针预偏移、热循环零守卫零重寻址,
    // 小形状(o_proj 等 [·,3840])实测比现役 gemv 快 ~40%
    if input.dtype == MetalTensorDType::F16 && bits == 4 && group_size == 64 && input.rows == 1 && weight_cols.is_multiple_of(8) {
        return mlx_affine_qmv_f16_ml_resident(ctx, input, packed, scales, biases, scale_dtype, group_size as usize, weight_rows, weight_cols);
    }
    // 2-8 行的投机校验形状走 gemv 的多行网格:行组同时读同一份权重,
    // 权重小于 L2 时靠硬件广播,DRAM 流量≈单行,绕开 MPS 与行数无关的反量化固定成本
    // 多行也必须直接消费量化权重；禁止为了 MPS 展开整块 F16 权重。
    // F16 走原生多行 GEMV，BF16 走量化 QMM。
    let (output, kernel) = match input.dtype {
        MetalTensorDType::F16 if bits == 4 && matches!(scale_dtype, 0 | 1) => (ctx.tensor_zeros(input.rows, weight_rows), "mlx_affine_gemv_f16_u4"),
        MetalTensorDType::F16 => (ctx.tensor_zeros(input.rows, weight_rows), "mlx_affine_gemv_f16"),
        MetalTensorDType::Bf16 if input.rows > 1 => (ctx.tensor_zeros_bf16(input.rows, weight_rows), "mlx_affine_qmm_bf16"),
        MetalTensorDType::Bf16 if bits == 8 && scale_dtype == 0 => (ctx.tensor_zeros_bf16(input.rows, weight_rows), "mlx_affine_gemv_bf16_u8"),
        MetalTensorDType::Bf16 if bits == 4 && scale_dtype == 0 => (ctx.tensor_zeros_bf16(input.rows, weight_rows), "mlx_affine_gemv_bf16_u4"),
        MetalTensorDType::Bf16 => (ctx.tensor_zeros_bf16(input.rows, weight_rows), "mlx_affine_gemv_bf16"),
        MetalTensorDType::F32 => return Err("MLX affine linear 需要 F16/BF16 输入".to_owned()),
    };
    let pipeline = ctx.pipeline(kernel)?;
    let required_threads = if kernel == "mlx_affine_qmm_bf16" { 128 } else { 64 };
    if pipeline.max_total_threads_per_threadgroup() < required_threads {
        return Err(format!("MLX affine {kernel} 需要至少 {required_threads} threads/threadgroup"));
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(packed), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(biases), 0);
    encoder.set_buffer(4, Some(&output.buffer), 0);
    set_bytes(&encoder, 5, &input_rows);
    set_bytes(&encoder, 6, &input_columns);
    set_bytes(&encoder, 7, &output_columns);
    set_bytes(&encoder, 8, &bits);
    set_bytes(&encoder, 9, &group_size);
    set_bytes(&encoder, 10, &scale_dtype);
    if kernel == "mlx_affine_qmm_bf16" {
        encoder.dispatch_thread_groups(MTLSize::new(weight_rows.div_ceil(32) as u64, input.rows.div_ceil(32) as u64, 1), MTLSize::new(128, 1, 1));
    } else if kernel == "mlx_affine_gemv_bf16_u8" {
        encoder.dispatch_thread_groups(MTLSize::new(weight_rows.div_ceil(32) as u64, input.rows as u64, 1), MTLSize::new(128, 1, 1));
    } else if kernel == "mlx_affine_gemv_f16_u4" {
        // f16_u4 每 threadgroup 16 行,其余 8 行
        encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, weight_rows.div_ceil(16) as u64, 1), MTLSize::new(64, 1, 1));
    } else if matches!(kernel, "mlx_affine_gemv_bf16" | "mlx_affine_gemv_bf16_u4") {
        // 行在 x:小批量多行时同列 tile 的行组相邻调度,L1/L2 广播同一份权重
        encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, weight_rows.div_ceil(8) as u64, 1), MTLSize::new(64, 1, 1));
    } else {
        encoder.dispatch_thread_groups(MTLSize::new(weight_rows as u64, input.rows as u64, 1), MTLSize::new(64, 1, 1));
    }
    encoder.end_encoding();
    let shape = format!("input=[{},{weight_cols}],weight=[{weight_rows},{weight_cols}],bits={bits},group={group_size}", input.rows);
    ctx.commit_and_wait_profiled(&command, kernel, &shape, input.buffer.length() + packed.length() + scales.length() + biases.length(), output.buffer.length());
    Ok(output)
}

pub fn mlx_affine_interleaved_matmul_tensor_resident(
    ctx: &MetalContext,
    input: &MetalTensor,
    packed: &metal::Buffer,
    scales: &metal::Buffer,
    biases: &metal::Buffer,
    role: u32,
    scale_dtype: u32,
    bits: usize,
    group_size: usize,
    weight_rows: usize,
    weight_cols: usize,
) -> Result<MetalTensor, String> {
    if role > 1 || bits != 8 || scale_dtype != 0 || group_size != 64 {
        return Err(format!("MLX affine interleaved role={role} bits={bits} scale_dtype={scale_dtype} group={group_size} 不受支持",));
    }
    let packed_bytes = weight_rows.checked_mul(weight_cols).ok_or("MLX affine interleaved code 大小溢出")?;
    let parameter_bytes = weight_rows.checked_mul(weight_cols / group_size).and_then(|count| count.checked_mul(2)).ok_or("MLX affine interleaved 参数大小溢出")?;
    validate_size("MLX affine interleaved codes", packed_bytes * 2, packed.length() as usize)?;
    validate_size("MLX affine interleaved scales", parameter_bytes * 2, scales.length() as usize)?;
    validate_size("MLX affine interleaved biases", parameter_bytes * 2, biases.length() as usize)?;

    // decode gemv 的 fused kernel 消费 Bf16；F16 输入只 cast 单行输入（cols 个
    // 元素），避免每次调用都为整个 8-bit 权重三元组走 deinterleave 兜底路径。
    let cast_input;
    let input = if input.rows == 1 && input.dtype == MetalTensorDType::F16 {
        cast_input = to_bf16_tensor(ctx, input)?;
        &cast_input
    } else {
        input
    };
    if input.rows == 1 && input.dtype == MetalTensorDType::Bf16 {
        return mlx_affine_interleaved_gemv_bf16_u8(ctx, input, packed, scales, biases, role, group_size, weight_rows, weight_cols);
    }
    if input.dtype == MetalTensorDType::Bf16 && input.rows > 1 {
        return mlx_affine_interleaved_qmm_bf16_u8(ctx, input, packed, scales, biases, role, weight_rows, weight_cols);
    }

    // 非默认调试路径按需生成标准临时布局，不保留第二份常驻权重。
    let packed = deinterleave_mlx_affine_pair(ctx, packed, role, 16, packed_bytes, "codes")?;
    let scales = deinterleave_mlx_affine_pair(ctx, scales, role, 2, parameter_bytes, "scales")?;
    let biases = deinterleave_mlx_affine_pair(ctx, biases, role, 2, parameter_bytes, "biases")?;
    mlx_affine_matmul_tensor_resident(ctx, input, &packed, &scales, &biases, scale_dtype, bits, group_size, weight_rows, weight_cols)
}

#[allow(clippy::too_many_arguments)]
fn mlx_affine_interleaved_qmm_bf16_u8(ctx: &MetalContext, input: &MetalTensor, packed: &metal::Buffer, scales: &metal::Buffer, biases: &metal::Buffer, role: u32, weight_rows: usize, weight_cols: usize) -> Result<MetalTensor, String> {
    let input_rows = validate_u32("MLX affine interleaved QMM input rows", input.rows)?;
    let input_columns = validate_u32("MLX affine interleaved QMM input columns", weight_cols)?;
    let output_columns = validate_u32("MLX affine interleaved QMM output columns", weight_rows)?;
    let output = ctx.tensor_zeros_bf16(input.rows, weight_rows);
    let pipeline = ctx.pipeline("mlx_affine_qmm_bf16_u8_interleaved")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("MLX affine interleaved QMM U8 需要至少 128 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(packed), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(biases), 0);
    encoder.set_buffer(4, Some(&output.buffer), 0);
    set_bytes(&encoder, 5, &input_rows);
    set_bytes(&encoder, 6, &input_columns);
    set_bytes(&encoder, 7, &output_columns);
    set_bytes(&encoder, 8, &role);
    encoder.dispatch_thread_groups(MTLSize::new(weight_rows.div_ceil(32) as u64, input.rows.div_ceil(32) as u64, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{},{}],weight=[{weight_rows},{weight_cols}],bits=8,group=64,role={role}", input.rows, input.cols,);
    ctx.commit_and_wait_profiled(&command, "mlx_affine_qmm_bf16_u8_interleaved", &shape, input.buffer.length() + packed.length() + scales.length() + biases.length(), output.buffer.length());
    Ok(output)
}

fn deinterleave_mlx_affine_pair(ctx: &MetalContext, source: &metal::Buffer, role: u32, block_bytes: usize, output_bytes: usize, name: &str) -> Result<metal::Buffer, String> {
    let output = ctx.shared_buffer_zeros(output_bytes);
    let block_bytes = validate_u32("MLX affine interleaved block bytes", block_bytes)?;
    let output_bytes_u32 = validate_u32("MLX affine interleaved output bytes", output_bytes)?;
    let shape = format!("{name}={output_bytes},role={role},block={block_bytes}");
    launch_1d(ctx, "mlx_affine_deinterleave_pair", &shape, output_bytes, source.length(), output.length(), |encoder| {
        encoder.set_buffer(0, Some(source), 0);
        encoder.set_buffer(1, Some(&output), 0);
        set_bytes(encoder, 2, &block_bytes);
        set_bytes(encoder, 3, &output_bytes_u32);
        set_bytes(encoder, 4, &role);
    })?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn mlx_affine_interleaved_gemv_bf16_u8(ctx: &MetalContext, input: &MetalTensor, packed: &metal::Buffer, scales: &metal::Buffer, biases: &metal::Buffer, role: u32, group_size: usize, rows: usize, cols: usize) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != cols || input.dtype != MetalTensorDType::Bf16 {
        return Err(format!("MLX affine interleaved U8 input=[{},{},{:?}] weight=[{rows},{cols}] 不兼容", input.rows, input.cols, input.dtype,));
    }
    let input_rows = 1_u32;
    let input_columns = validate_u32("MLX affine interleaved U8 input columns", cols)?;
    let output_columns = validate_u32("MLX affine interleaved U8 output columns", rows)?;
    let bits = 8_u32;
    let group_size = validate_u32("MLX affine interleaved U8 group size", group_size)?;
    let scale_dtype = 0_u32;
    let output = ctx.tensor_zeros_bf16(1, rows);
    let pipeline = ctx.pipeline("mlx_affine_gemv_bf16_u8_interleaved")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("MLX affine interleaved U8 需要至少 128 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(packed), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(biases), 0);
    encoder.set_buffer(4, Some(&output.buffer), 0);
    set_bytes(&encoder, 5, &input_rows);
    set_bytes(&encoder, 6, &input_columns);
    set_bytes(&encoder, 7, &output_columns);
    set_bytes(&encoder, 8, &bits);
    set_bytes(&encoder, 9, &group_size);
    set_bytes(&encoder, 10, &scale_dtype);
    set_bytes(&encoder, 11, &role);
    encoder.dispatch_thread_groups(MTLSize::new(rows.div_ceil(32) as u64, 1, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{cols}],weight=[{rows},{cols}],bits=8,group=64,role={role}");
    ctx.commit_and_wait_profiled(&command, "mlx_affine_gemv_bf16_u8_interleaved", &shape, input.buffer.length() + packed.length() + scales.length() + biases.length(), output.buffer.length());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn mlx_affine_gated_interleaved_gemv_bf16_u8(
    ctx: &MetalContext,
    input: &MetalTensor,
    packed: &metal::Buffer,
    scales: &metal::Buffer,
    biases: &metal::Buffer,
    scale_dtype: u32,
    bits: usize,
    group_size: usize,
    rows: usize,
    cols: usize,
    activation: &crate::moe::Activation,
) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != cols || input.dtype != MetalTensorDType::Bf16 || scale_dtype != 0 || bits != 8 || group_size != 64 || rows == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("MLX affine gated interleaved U8 input=[{},{},{:?}] weight=[{rows},{cols}] group={group_size} 不兼容", input.rows, input.cols, input.dtype,));
    }
    let expected_codes = rows.checked_mul(cols).and_then(|count| count.checked_mul(2)).ok_or("MLX affine gated interleaved U8 code 大小溢出")?;
    let expected_parameters = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(4)).ok_or("MLX affine gated interleaved U8 参数大小溢出")?;
    validate_size("MLX affine gated interleaved U8 codes", expected_codes, packed.length() as usize)?;
    validate_size("MLX affine gated interleaved U8 scales", expected_parameters, scales.length() as usize)?;
    validate_size("MLX affine gated interleaved U8 biases", expected_parameters, biases.length() as usize)?;

    let input_columns = validate_u32("MLX affine gated interleaved U8 input columns", cols)?;
    let output_columns = validate_u32("MLX affine gated interleaved U8 output columns", rows)?;
    let group_size = validate_u32("MLX affine gated interleaved U8 group size", group_size)?;
    let activation = GatedActivation::from_spec(activation)?;
    let output = ctx.tensor_zeros_bf16(1, rows);
    let pipeline = ctx.pipeline("mlx_affine_gated_interleaved_gemv_bf16_u8")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("MLX affine gated interleaved U8 需要至少 128 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(packed), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(biases), 0);
    encoder.set_buffer(4, Some(&output.buffer), 0);
    set_bytes(&encoder, 5, &input_columns);
    set_bytes(&encoder, 6, &output_columns);
    set_bytes(&encoder, 7, &group_size);
    set_bytes(&encoder, 8, &activation.kind);
    set_bytes(&encoder, 9, &activation.alpha);
    set_bytes(&encoder, 10, &activation.limit);
    encoder.dispatch_thread_groups(MTLSize::new(rows.div_ceil(16) as u64, 1, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{cols}],weights=interleaved[{rows},{cols}],bits=8,group=64");
    ctx.commit_and_wait_profiled(&command, "mlx_affine_gated_interleaved_gemv_bf16_u8", &shape, input.buffer.length() + packed.length() + scales.length() + biases.length(), output.buffer.length());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn mlx_affine_gated_gemv_bf16_u8(
    ctx: &MetalContext,
    input: &MetalTensor,
    gate_packed: &metal::Buffer,
    gate_scales: &metal::Buffer,
    gate_biases: &metal::Buffer,
    up_packed: &metal::Buffer,
    up_scales: &metal::Buffer,
    up_biases: &metal::Buffer,
    group_size: usize,
    rows: usize,
    cols: usize,
    activation: &crate::moe::Activation,
) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != cols || input.dtype != MetalTensorDType::Bf16 || group_size != 64 || rows == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("MLX affine gated U8 input=[{},{},{:?}] weight=[{rows},{cols}] group={group_size} 不兼容", input.rows, input.cols, input.dtype,));
    }
    let expected_codes = rows.checked_mul(cols).ok_or("MLX affine gated U8 code 大小溢出")?;
    let expected_parameters = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(2)).ok_or("MLX affine gated U8 参数大小溢出")?;
    validate_size("MLX affine gated U8 gate codes", expected_codes, gate_packed.length() as usize)?;
    validate_size("MLX affine gated U8 gate scales", expected_parameters, gate_scales.length() as usize)?;
    validate_size("MLX affine gated U8 gate biases", expected_parameters, gate_biases.length() as usize)?;
    validate_size("MLX affine gated U8 up codes", expected_codes, up_packed.length() as usize)?;
    validate_size("MLX affine gated U8 up scales", expected_parameters, up_scales.length() as usize)?;
    validate_size("MLX affine gated U8 up biases", expected_parameters, up_biases.length() as usize)?;

    let input_columns = validate_u32("MLX affine gated U8 input columns", cols)?;
    let output_columns = validate_u32("MLX affine gated U8 output columns", rows)?;
    let group_size = validate_u32("MLX affine gated U8 group size", group_size)?;
    let activation = GatedActivation::from_spec(activation)?;
    let output = ctx.tensor_zeros_bf16(1, rows);
    let pipeline = ctx.pipeline("mlx_affine_gated_gemv_bf16_u8")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("MLX affine gated U8 需要至少 128 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(gate_packed), 0);
    encoder.set_buffer(2, Some(gate_scales), 0);
    encoder.set_buffer(3, Some(gate_biases), 0);
    encoder.set_buffer(4, Some(up_packed), 0);
    encoder.set_buffer(5, Some(up_scales), 0);
    encoder.set_buffer(6, Some(up_biases), 0);
    encoder.set_buffer(7, Some(&output.buffer), 0);
    set_bytes(&encoder, 8, &input_columns);
    set_bytes(&encoder, 9, &output_columns);
    set_bytes(&encoder, 10, &group_size);
    set_bytes(&encoder, 11, &activation.kind);
    set_bytes(&encoder, 12, &activation.alpha);
    set_bytes(&encoder, 13, &activation.limit);
    encoder.dispatch_thread_groups(MTLSize::new(rows.div_ceil(16) as u64, 1, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{cols}],weights=2x[{rows},{cols}],bits=8,group=64");
    ctx.commit_and_wait_profiled(
        &command,
        "mlx_affine_gated_gemv_bf16_u8",
        &shape,
        input.buffer.length() + gate_packed.length() + gate_scales.length() + gate_biases.length() + up_packed.length() + up_scales.length() + up_biases.length(),
        output.buffer.length(),
    );
    Ok(output)
}

/// F16 输入 + 双 u4 affine 权重的 gated gemv:gate/up 共享输入加载,单 kernel
/// 产出激活结果,替代「两次 gemv + gated_activation」三次 dispatch。
#[allow(clippy::too_many_arguments)]
pub fn mlx_affine_gated_gemv_f16_u4(
    ctx: &MetalContext,
    input: &MetalTensor,
    gate_packed: &metal::Buffer,
    gate_scales: &metal::Buffer,
    gate_biases: &metal::Buffer,
    up_packed: &metal::Buffer,
    up_scales: &metal::Buffer,
    up_biases: &metal::Buffer,
    scale_dtype: u32,
    group_size: usize,
    rows: usize,
    cols: usize,
    activation: &crate::moe::Activation,
) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != cols || input.dtype != MetalTensorDType::F16 || rows == 0 || group_size == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("MLX affine gated F16 U4 input=[{},{},{:?}] weight=[{rows},{cols}] group={group_size} 不兼容", input.rows, input.cols, input.dtype));
    }
    let parameter_bytes = match scale_dtype {
        0 | 1 => 2,
        2 => 4,
        _ => return Err(format!("MLX affine gated F16 U4 scale dtype code {scale_dtype} 不受支持")),
    };
    let expected_codes = rows.checked_mul(cols.div_ceil(8)).and_then(|words| words.checked_mul(4)).ok_or("MLX affine gated F16 U4 code 大小溢出")?;
    let expected_parameters = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(parameter_bytes)).ok_or("MLX affine gated F16 U4 参数大小溢出")?;
    validate_size("MLX affine gated F16 U4 gate codes", expected_codes, gate_packed.length() as usize)?;
    validate_size("MLX affine gated F16 U4 gate scales", expected_parameters, gate_scales.length() as usize)?;
    validate_size("MLX affine gated F16 U4 gate biases", expected_parameters, gate_biases.length() as usize)?;
    validate_size("MLX affine gated F16 U4 up codes", expected_codes, up_packed.length() as usize)?;
    validate_size("MLX affine gated F16 U4 up scales", expected_parameters, up_scales.length() as usize)?;
    validate_size("MLX affine gated F16 U4 up biases", expected_parameters, up_biases.length() as usize)?;

    let input_columns = validate_u32("MLX affine gated F16 U4 input columns", cols)?;
    let output_columns = validate_u32("MLX affine gated F16 U4 output columns", rows)?;
    let group_size = validate_u32("MLX affine gated F16 U4 group size", group_size)?;
    let activation = GatedActivation::from_spec(activation)?;
    let output = ctx.tensor_zeros(1, rows);
    let pipeline = ctx.pipeline("mlx_affine_gated_gemv_f16_u4")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("MLX affine gated F16 U4 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(gate_packed), 0);
    encoder.set_buffer(2, Some(gate_scales), 0);
    encoder.set_buffer(3, Some(gate_biases), 0);
    encoder.set_buffer(4, Some(up_packed), 0);
    encoder.set_buffer(5, Some(up_scales), 0);
    encoder.set_buffer(6, Some(up_biases), 0);
    encoder.set_buffer(7, Some(&output.buffer), 0);
    set_bytes(&encoder, 8, &input_columns);
    set_bytes(&encoder, 9, &output_columns);
    set_bytes(&encoder, 10, &group_size);
    set_bytes(&encoder, 11, &scale_dtype);
    set_bytes(&encoder, 12, &activation.kind);
    set_bytes(&encoder, 13, &activation.alpha);
    set_bytes(&encoder, 14, &activation.limit);
    encoder.dispatch_thread_groups(MTLSize::new(rows.div_ceil(8) as u64, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{cols}],weights=2x[{rows},{cols}],bits=4,group={group_size}");
    ctx.commit_and_wait_profiled(
        &command,
        "mlx_affine_gated_gemv_f16_u4",
        &shape,
        input.buffer.length() + gate_packed.length() + gate_scales.length() + gate_biases.length() + up_packed.length() + up_scales.length() + up_biases.length(),
        output.buffer.length(),
    );
    Ok(output)
}
/// F16 输入 + 三路 u4 affine 权重的合并 gemv(Q/K/V 共享输入):单次 dispatch
/// 写出三段输出,替代三次独立 gemv;每行算术与单矩阵 gemv 逐位一致。
#[allow(clippy::too_many_arguments)]
pub fn mlx_affine_triple_gemv_f16_u4(
    ctx: &MetalContext,
    input: &MetalTensor,
    first_packed: &metal::Buffer,
    first_scales: &metal::Buffer,
    first_biases: &metal::Buffer,
    first_rows: usize,
    second_packed: &metal::Buffer,
    second_scales: &metal::Buffer,
    second_biases: &metal::Buffer,
    second_rows: usize,
    third_packed: &metal::Buffer,
    third_scales: &metal::Buffer,
    third_biases: &metal::Buffer,
    third_rows: usize,
    scale_dtype: u32,
    group_size: usize,
    cols: usize,
) -> Result<(MetalTensor, MetalTensor, MetalTensor), String> {
    if input.rows != 1 || input.cols != cols || input.dtype != MetalTensorDType::F16 || first_rows == 0 || second_rows == 0 || third_rows == 0 || group_size == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("MLX affine triple F16 U4 input=[{},{},{:?}] weight=[{first_rows}+{second_rows}+{third_rows},{cols}] group={group_size} 不兼容", input.rows, input.cols, input.dtype));
    }
    let parameter_bytes = match scale_dtype {
        0 | 1 => 2,
        2 => 4,
        _ => return Err(format!("MLX affine triple F16 U4 scale dtype code {scale_dtype} 不受支持")),
    };
    let validate_stream = |name: &str, packed: &metal::Buffer, scales: &metal::Buffer, biases: &metal::Buffer, rows: usize| -> Result<(), String> {
        let expected_codes = rows.checked_mul(cols.div_ceil(8)).and_then(|words| words.checked_mul(4)).ok_or_else(|| format!("MLX affine triple F16 U4 {name} code 大小溢出"))?;
        let expected_parameters = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(parameter_bytes)).ok_or_else(|| format!("MLX affine triple F16 U4 {name} 参数大小溢出"))?;
        validate_size(&format!("MLX affine triple F16 U4 {name} codes"), expected_codes, packed.length() as usize)?;
        validate_size(&format!("MLX affine triple F16 U4 {name} scales"), expected_parameters, scales.length() as usize)?;
        validate_size(&format!("MLX affine triple F16 U4 {name} biases"), expected_parameters, biases.length() as usize)
    };
    validate_stream("first", first_packed, first_scales, first_biases, first_rows)?;
    validate_stream("second", second_packed, second_scales, second_biases, second_rows)?;
    validate_stream("third", third_packed, third_scales, third_biases, third_rows)?;

    let input_columns = validate_u32("MLX affine triple F16 U4 input columns", cols)?;
    let first_columns = validate_u32("MLX affine triple F16 U4 first rows", first_rows)?;
    let second_columns = validate_u32("MLX affine triple F16 U4 second rows", second_rows)?;
    let third_columns = validate_u32("MLX affine triple F16 U4 third rows", third_rows)?;
    let group_size = validate_u32("MLX affine triple F16 U4 group size", group_size)?;
    let first = ctx.tensor_zeros(1, first_rows);
    let second = ctx.tensor_zeros(1, second_rows);
    let third = ctx.tensor_zeros(1, third_rows);
    let pipeline = ctx.pipeline("mlx_affine_triple_gemv_f16_u4")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("MLX affine triple F16 U4 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(first_packed), 0);
    encoder.set_buffer(2, Some(first_scales), 0);
    encoder.set_buffer(3, Some(first_biases), 0);
    encoder.set_buffer(4, Some(second_packed), 0);
    encoder.set_buffer(5, Some(second_scales), 0);
    encoder.set_buffer(6, Some(second_biases), 0);
    encoder.set_buffer(7, Some(third_packed), 0);
    encoder.set_buffer(8, Some(third_scales), 0);
    encoder.set_buffer(9, Some(third_biases), 0);
    encoder.set_buffer(10, Some(&first.buffer), 0);
    encoder.set_buffer(11, Some(&second.buffer), 0);
    encoder.set_buffer(12, Some(&third.buffer), 0);
    set_bytes(&encoder, 13, &input_columns);
    set_bytes(&encoder, 14, &first_columns);
    set_bytes(&encoder, 15, &second_columns);
    set_bytes(&encoder, 16, &third_columns);
    set_bytes(&encoder, 17, &group_size);
    set_bytes(&encoder, 18, &scale_dtype);
    let total_rows = first_rows + second_rows + third_rows;
    encoder.dispatch_thread_groups(MTLSize::new(total_rows.div_ceil(8) as u64, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{cols}],weights=[{first_rows}+{second_rows}+{third_rows},{cols}],bits=4,group={group_size}");
    ctx.commit_and_wait_profiled(
        &command,
        "mlx_affine_triple_gemv_f16_u4",
        &shape,
        input.buffer.length() + first_packed.length() + second_packed.length() + third_packed.length(),
        first.buffer.length() + second.buffer.length() + third.buffer.length(),
    );
    Ok((first, second, third))
}

#[allow(clippy::too_many_arguments)]
pub fn mlx_affine_linear_gated_bf16_u4(
    ctx: &MetalContext,
    input: &MetalTensor,
    packed: &metal::Buffer,
    scales: &metal::Buffer,
    biases: &metal::Buffer,
    up: &MetalTensor,
    group_size: usize,
    rows: usize,
    cols: usize,
    activation: &crate::moe::Activation,
) -> Result<MetalTensor, String> {
    if input.rows != 1 || input.cols != cols || input.dtype != MetalTensorDType::Bf16 || up.rows != 1 || up.cols != rows || up.dtype != MetalTensorDType::Bf16 || group_size != 64 || rows == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("MLX affine linear-gated U4 input=[{},{},{:?}] up=[{},{},{:?}] weight=[{rows},{cols}] group={group_size} 不兼容", input.rows, input.cols, input.dtype, up.rows, up.cols, up.dtype,));
    }
    let expected_codes = rows.checked_mul(cols.div_ceil(8)).and_then(|words| words.checked_mul(4)).ok_or("MLX affine linear-gated U4 code 大小溢出")?;
    let expected_parameters = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(2)).ok_or("MLX affine linear-gated U4 参数大小溢出")?;
    validate_size("MLX affine linear-gated U4 codes", expected_codes, packed.length() as usize)?;
    validate_size("MLX affine linear-gated U4 scales", expected_parameters, scales.length() as usize)?;
    validate_size("MLX affine linear-gated U4 biases", expected_parameters, biases.length() as usize)?;

    let input_columns = validate_u32("MLX affine linear-gated U4 input columns", cols)?;
    let output_columns = validate_u32("MLX affine linear-gated U4 output columns", rows)?;
    let group_size = validate_u32("MLX affine linear-gated U4 group size", group_size)?;
    let activation = GatedActivation::from_spec(activation)?;
    let output = ctx.tensor_zeros_bf16(1, rows);
    let pipeline = ctx.pipeline("mlx_affine_linear_gated_bf16_u4")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("MLX affine linear-gated U4 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(packed), 0);
    encoder.set_buffer(2, Some(scales), 0);
    encoder.set_buffer(3, Some(biases), 0);
    encoder.set_buffer(4, Some(&up.buffer), 0);
    encoder.set_buffer(5, Some(&output.buffer), 0);
    set_bytes(&encoder, 6, &input_columns);
    set_bytes(&encoder, 7, &output_columns);
    set_bytes(&encoder, 8, &group_size);
    set_bytes(&encoder, 9, &activation.kind);
    set_bytes(&encoder, 10, &activation.alpha);
    set_bytes(&encoder, 11, &activation.limit);
    encoder.dispatch_thread_groups(MTLSize::new(rows.div_ceil(8) as u64, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[1,{cols}],weight=[{rows},{cols}],bits=4,group=64");
    ctx.commit_and_wait_profiled(&command, "mlx_affine_linear_gated_bf16_u4", &shape, input.buffer.length() + packed.length() + scales.length() + biases.length() + up.buffer.length(), output.buffer.length());
    Ok(output)
}

#[cfg(test)]
mod mlx_affine_tests {
    use super::super::as_bytes;
    use super::*;

    /// 混合精度:融合 rmsnorm+gated gemv 的 Bf16 输入必须直读(不经过整行 cast),
    /// 与 F16 输入路径在可精确表示的值域上逐位一致。
    #[test]
    fn rmsnorm_gated_gemv_accepts_bf16_input() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let columns = 8usize;
        let weight_rows = 4usize;
        let group_size = 4usize;
        // 全部取 f16/bf16 都可精确表示的小整数,两条 dtype 路径数学上应逐位一致
        let input_values: Vec<f32> = vec![0.5, -1.0, 1.5, -2.0, 2.5, -0.25, 3.0, 0.75];
        let norm_values: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0, 0.25, 1.25, 0.75, 1.75];
        let packed = vec![0x2101_2310u32; weight_rows * columns / 8 * 2 / 4];
        let scales = vec![half::f16::from_f32(0.5); weight_rows * columns / group_size];
        let biases = vec![half::f16::from_f32(-0.5); weight_rows * columns / group_size];
        let norm_weight = ctx.shared_buffer(as_bytes(&norm_values.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>()));
        let gate_packed = ctx.shared_buffer(as_bytes(&packed));
        let gate_scales = ctx.shared_buffer(as_bytes(&scales));
        let gate_biases = ctx.shared_buffer(as_bytes(&biases));
        let up_packed = ctx.shared_buffer(as_bytes(&packed));
        let up_scales = ctx.shared_buffer(as_bytes(&scales));
        let up_biases = ctx.shared_buffer(as_bytes(&biases));

        let input_f16 = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let input_bf16 = to_bf16_tensor(&ctx, &input_f16).unwrap();
        let eps = 1.0e-6f32;
        let activation = crate::moe::Activation::Silu;
        let from_f16 = mlx_affine_rmsnorm_gated_gemv_f16_u4(&ctx, &input_f16, &norm_weight, eps, &gate_packed, &gate_scales, &gate_biases, &up_packed, &up_scales, &up_biases, 1, group_size, weight_rows, columns, &activation).unwrap();
        let from_bf16 = mlx_affine_rmsnorm_gated_gemv_f16_u4(&ctx, &input_bf16, &norm_weight, eps, &gate_packed, &gate_scales, &gate_biases, &up_packed, &up_scales, &up_biases, 1, group_size, weight_rows, columns, &activation).unwrap();
        assert_eq!(ctx.tensor_to_f32(&from_f16), ctx.tensor_to_f32(&from_bf16), "bf16 直读与 f16 路径输出必须逐位一致");
    }

    #[test]
    fn affine_4bit_prefill_matches_cpu() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let columns = 8usize;
        let rows = 2usize;
        let packed = [0x1111_1111_u32, 0x2222_2222_u32];
        let scales = [1.0_f32; 4];
        let biases = [0.0_f32, 0.0, -1.0, -1.0];
        let input_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let input = ctx.tensor_from_f32(&input_values, 2, columns).unwrap();
        let output = mlx_affine_matmul_tensor_resident(&ctx, &input, &ctx.shared_buffer(as_bytes(&packed)), &ctx.shared_buffer(as_bytes(&scales)), &ctx.shared_buffer(as_bytes(&biases)), 2, 4, 4, rows, columns).unwrap();
        let actual = ctx.tensor_to_f32(&output);
        assert_eq!(actual, vec![36.0, 36.0, -36.0, -36.0]);
    }

    #[test]
    fn affine_4bit_f16_gemv_matches_cpu() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        // 512 覆盖 values_per_thread=16 快路径,128 覆盖 8 值路径,3840 覆盖 12B 实宽的尾块;
        // scales 覆盖 BF16 与 F16 两种 dtype
        for columns in [512_usize, 128, 3840] {
            for scale_f16 in [false, true] {
                let scale_code = u32::from(scale_f16);
                let weight_rows = 6usize;
                let mut packed = vec![0u32; weight_rows * columns.div_ceil(8)];
                let code_at = |row: usize, column: usize| ((row * 31 + column * 17) % 16) as u32;
                for row in 0..weight_rows {
                    for column in 0..columns {
                        packed[row * (columns / 8) + column / 8] |= code_at(row, column) << ((column % 8) * 4);
                    }
                }
                let group_count = columns / 64;
                let scale_values = (0..weight_rows * group_count).map(|index| if index % 2 == 0 { 0.5_f32 } else { 0.25 }).collect::<Vec<_>>();
                let bias_values = (0..weight_rows * group_count).map(|index| if index % 3 == 0 { -0.25_f32 } else { 0.125 }).collect::<Vec<_>>();
                // 0.5/0.25/-0.25/0.125 在 BF16 与 F16 中都精确表示,参考值直接用 f32
                let encode = |values: &[f32]| -> Vec<u8> {
                    if scale_f16 {
                        let encoded: Vec<half::f16> = values.iter().map(|value| half::f16::from_f32(*value)).collect();
                        as_bytes(&encoded).to_vec()
                    } else {
                        let encoded: Vec<half::bf16> = values.iter().map(|value| half::bf16::from_f32(*value)).collect();
                        as_bytes(&encoded).to_vec()
                    }
                };
                let scales = encode(&scale_values);
                let biases = encode(&bias_values);
                // 输入在 GPU 上以 F16 存储,参考值同样过一遍 F16 取整
                let input = (0..columns).map(|index| ((index as f32) * 0.037).sin() * 0.5 + 0.1).collect::<Vec<f32>>();
                let rounded = input.iter().map(|value| half::f16::from_f32(*value).to_f32()).collect::<Vec<f32>>();
                let reference = (0..weight_rows)
                    .map(|row| {
                        (0..columns)
                            .map(|column| {
                                let group = row * group_count + column / 64;
                                rounded[column] * (scale_values[group] * code_at(row, column) as f32 + bias_values[group])
                            })
                            .sum::<f32>()
                    })
                    .collect::<Vec<f32>>();
                let close = |actual: f32, expected: f32| (actual - expected).abs() < 0.02 * expected.abs().max(1.0);
                let single = ctx.tensor_from_f32(&input, 1, columns).unwrap();
                let output = mlx_affine_matmul_tensor_resident(&ctx, &single, &ctx.shared_buffer(as_bytes(&packed)), &ctx.shared_buffer(&scales), &ctx.shared_buffer(&biases), scale_code, 4, 64, weight_rows, columns).unwrap();
                let actual = ctx.tensor_to_f32(&output);
                for (actual, expected) in actual.iter().zip(&reference) {
                    assert!(close(*actual, *expected), "columns={columns} scale_f16={scale_f16} 单行 gemv actual={actual} expected={expected}");
                }
                // 多行 F16 先转 BF16，再走量化 QMM；结果必须与单行 GEMV 一致。
                let doubled = ctx.tensor_from_f32(&[input.clone(), input.clone()].concat(), 2, columns).unwrap();
                let batched = mlx_affine_matmul_tensor_resident(&ctx, &doubled, &ctx.shared_buffer(as_bytes(&packed)), &ctx.shared_buffer(&scales), &ctx.shared_buffer(&biases), scale_code, 4, 64, weight_rows, columns).unwrap();
                let batched = ctx.tensor_to_f32(&batched);
                for row in 0..2 {
                    for column in 0..weight_rows {
                        let value = batched[row * weight_rows + column];
                        assert!(close(value, actual[column]), "columns={columns} scale_f16={scale_f16} 多行 gemv 行{row} actual={value} 单行={}", actual[column]);
                    }
                }
                // 更宽批次仍走同一量化 QMM；参考值复刻输入和权重的 BF16 边界。
                let rounded_weight = |row: usize, column: usize| -> f32 {
                    let group = row * group_count + column / 64;
                    let value = scale_values[group] * code_at(row, column) as f32 + bias_values[group];
                    half::f16::from_f32(half::bf16::from_f32(value).to_f32()).to_f32()
                };
                let rounded_input = rounded.iter().map(|value| half::bf16::from_f32(*value).to_f32()).collect::<Vec<_>>();
                let reference_qmm = (0..weight_rows).map(|row| (0..columns).map(|column| rounded_input[column] * rounded_weight(row, column)).sum::<f32>()).collect::<Vec<f32>>();
                let wide = ctx.tensor_from_f32(&vec![input.clone(); 9].concat(), 9, columns).unwrap();
                let qmm = mlx_affine_matmul_tensor_resident(&ctx, &wide, &ctx.shared_buffer(as_bytes(&packed)), &ctx.shared_buffer(&scales), &ctx.shared_buffer(&biases), scale_code, 4, 64, weight_rows, columns).unwrap();
                let qmm = ctx.tensor_to_f32(&qmm);
                for row in 0..9 {
                    for column in 0..weight_rows {
                        let value = qmm[row * weight_rows + column];
                        assert!(close(value, reference_qmm[column]), "columns={columns} scale_f16={scale_f16} QMM 行{row} actual={value} expected={}", reference_qmm[column]);
                    }
                }
            }
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod gated_f16_u4_tests {
    use super::*;

    /// F16-u4 gated gemv 必须与「单矩阵 gemv x2 + gated_activation」链一致
    #[test]
    fn affine_gated_f16_u4_matches_sequential_chain() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        // 512/3840 覆盖 16 值快路径,128 覆盖 8 值路径
        for columns in [512_usize, 128, 3840] {
            let rows = 6usize;
            let build = |seed: usize| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
                let mut packed = vec![0u32; rows * columns.div_ceil(8)];
                let mut scales = Vec::new();
                let mut biases = Vec::new();
                for row in 0..rows {
                    for group in 0..columns / 64 {
                        let scale = 1.0e-3 + ((row * 7 + group * 3 + seed) as f32 * 0.021).sin().abs() * 3.0e-3;
                        let bias = ((row * 11 + group * 5 + seed) as f32 * 0.017).sin() * 1.0e-3;
                        scales.extend_from_slice(&half::bf16::from_f32(scale).to_le_bytes());
                        biases.extend_from_slice(&half::bf16::from_f32(bias).to_le_bytes());
                        for element in 0..64 {
                            let column = group * 64 + element;
                            let code = ((row * 31 + column * 17 + seed) % 16) as u32;
                            packed[row * (columns / 8) + column / 8] |= code << ((column % 8) * 4);
                        }
                    }
                }
                let packed_bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 4) }.to_vec();
                (packed_bytes, scales, biases)
            };
            let (gate_packed, gate_scales, gate_biases) = build(1);
            let (up_packed, up_scales, up_biases) = build(2);
            let input_values = (0..columns).map(|index| ((index as f32) * 0.037).sin() * 0.5 + 0.1).collect::<Vec<f32>>();
            let single = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();

            let gate = mlx_affine_matmul_tensor_resident(&ctx, &single, &ctx.shared_buffer(&gate_packed), &ctx.shared_buffer(&gate_scales), &ctx.shared_buffer(&gate_biases), 0, 4, 64, rows, columns).unwrap();
            let up = mlx_affine_matmul_tensor_resident(&ctx, &single, &ctx.shared_buffer(&up_packed), &ctx.shared_buffer(&up_scales), &ctx.shared_buffer(&up_biases), 0, 4, 64, rows, columns).unwrap();
            use crate::backend::Backend;
            let chained = ctx.gated_activation(&gate, &up, &crate::moe::Activation::GeluTanh).map_err(|error| error.to_string()).unwrap();
            let fused = mlx_affine_gated_gemv_f16_u4(
                &ctx,
                &single,
                &ctx.shared_buffer(&gate_packed),
                &ctx.shared_buffer(&gate_scales),
                &ctx.shared_buffer(&gate_biases),
                &ctx.shared_buffer(&up_packed),
                &ctx.shared_buffer(&up_scales),
                &ctx.shared_buffer(&up_biases),
                0,
                64,
                rows,
                columns,
                &crate::moe::Activation::GeluTanh,
            )
            .unwrap();

            let expected = ctx.tensor_to_f32(&chained);
            let actual = ctx.tensor_to_f32(&fused);
            for (index, (want, got)) in expected.iter().zip(&actual).enumerate() {
                assert!((want - got).abs() < 2.0e-2 * want.abs().max(1.0), "columns={columns} index={index}: 链式={want} 融合={got}");
            }
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod triple_f16_u4_tests {
    use super::*;

    /// 三路合并 gemv 的每行算术必须与三次单矩阵 gemv 逐位一致
    #[test]
    fn affine_triple_f16_u4_matches_single_gemvs() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        for columns in [512_usize, 128, 3840] {
            let build = |rows: usize, seed: usize| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
                let mut packed = vec![0u32; rows * columns.div_ceil(8)];
                let mut scales = Vec::new();
                let mut biases = Vec::new();
                for row in 0..rows {
                    for group in 0..columns / 64 {
                        scales.extend_from_slice(&half::bf16::from_f32(1.0e-3 + ((row * 7 + group * 3 + seed) as f32 * 0.021).sin().abs() * 3.0e-3).to_le_bytes());
                        biases.extend_from_slice(&half::bf16::from_f32(((row * 11 + group * 5 + seed) as f32 * 0.017).sin() * 1.0e-3).to_le_bytes());
                        for element in 0..64 {
                            let column = group * 64 + element;
                            packed[row * (columns / 8) + column / 8] |= (((row * 31 + column * 17 + seed) % 16) as u32) << ((column % 8) * 4);
                        }
                    }
                }
                let packed_bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 4) }.to_vec();
                (packed_bytes, scales, biases)
            };
            let (q_rows, k_rows, v_rows) = (10usize, 6usize, 6usize);
            let (q_packed, q_scales, q_biases) = build(q_rows, 1);
            let (k_packed, k_scales, k_biases) = build(k_rows, 2);
            let (v_packed, v_scales, v_biases) = build(v_rows, 3);
            let input_values = (0..columns).map(|index| ((index as f32) * 0.037).sin() * 0.5 + 0.1).collect::<Vec<f32>>();
            let single = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();

            let singles = [(q_rows, q_packed.clone(), q_scales.clone(), q_biases.clone()), (k_rows, k_packed.clone(), k_scales.clone(), k_biases.clone()), (v_rows, v_packed.clone(), v_scales.clone(), v_biases.clone())]
                .map(|(rows, packed, scales, biases)| mlx_affine_matmul_tensor_resident(&ctx, &single, &ctx.shared_buffer(&packed), &ctx.shared_buffer(&scales), &ctx.shared_buffer(&biases), 0, 4, 64, rows, columns).unwrap());
            let fused = mlx_affine_triple_gemv_f16_u4(
                &ctx,
                &single,
                &ctx.shared_buffer(&q_packed),
                &ctx.shared_buffer(&q_scales),
                &ctx.shared_buffer(&q_biases),
                q_rows,
                &ctx.shared_buffer(&k_packed),
                &ctx.shared_buffer(&k_scales),
                &ctx.shared_buffer(&k_biases),
                k_rows,
                &ctx.shared_buffer(&v_packed),
                &ctx.shared_buffer(&v_scales),
                &ctx.shared_buffer(&v_biases),
                v_rows,
                0,
                64,
                columns,
            )
            .unwrap();

            for (index, (tensor, expected)) in [(&fused.0, &singles[0]), (&fused.1, &singles[1]), (&fused.2, &singles[2])].iter().enumerate() {
                let want = ctx.tensor_to_f32(expected);
                let got = ctx.tensor_to_f32(tensor);
                assert_eq!(want, got, "columns={columns} 流{index} 应与单矩阵 gemv 逐位一致");
            }
        }
    }
}
#[cfg(all(test, target_os = "macos"))]
mod qmv_ml_tests {
    use super::*;

    /// MLX 结构移植版必须与现役 f16_u4 gemv 数值一致(容差:仅累加顺序差异)
    #[test]
    fn qmv_f16_ml_matches_f16_u4() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        // 3840 覆盖尾块路径(7x512+256),512 覆盖纯对齐路径
        for columns in [512_usize, 128, 3840] {
            let rows = 6usize;
            let mut rng: u32 = 0x1234_5678;
            let mut next_u32 = || {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                rng
            };
            let packed_values: Vec<u32> = (0..rows * columns.div_ceil(8)).map(|_| next_u32()).collect();
            let packed_bytes = unsafe { std::slice::from_raw_parts(packed_values.as_ptr().cast::<u8>(), packed_values.len() * 4) };
            let scales_values: Vec<u16> = (0..rows * columns / 64).map(|_| half::bf16::from_f32(0.5f32).to_le_bytes().iter().fold(0u16, |acc, byte| (acc << 8) | *byte as u16).swap_bytes()).collect();
            let scales_bytes = unsafe { std::slice::from_raw_parts(scales_values.as_ptr().cast::<u8>(), scales_values.len() * 2) };
            let input_values: Vec<f32> = (0..columns).map(|_| (next_u32() >> 8) as f32 / 8388608.0).collect();
            let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
            let packed = ctx.shared_buffer(packed_bytes);
            let scales = ctx.shared_buffer(scales_bytes);
            let biases_values: Vec<u16> =
                (0..rows * columns / 64).map(|index| half::bf16::from_f32(if index % 3 == 0 { -0.25f32 } else { 0.125f32 }).to_le_bytes().iter().fold(0u16, |acc, byte| (acc << 8) | *byte as u16).swap_bytes()).collect();
            let biases = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(biases_values.as_ptr().cast::<u8>(), biases_values.len() * 2) });

            let ported = mlx_affine_qmv_f16_ml_resident(&ctx, &input, &packed, &scales, &biases, 0, 64, rows, columns).unwrap();
            // CPU 参照:定位偏差来自哪个 kernel
            let mut reference = vec![0.0f32; rows];
            for row in 0..rows {
                let mut sum = 0.0f32;
                for column in 0..columns {
                    let word = packed_values[row * (columns / 8) + column / 8];
                    let code = ((word >> ((column % 8) * 4)) & 15) as f32;
                    let scale = 0.5f32;
                    let bias = if (row * (columns / 64) + column / 64) % 3 == 0 { -0.25f32 } else { 0.125f32 };
                    sum += input_values[column] * (scale * code + bias);
                }
                reference[row] = sum;
            }
            let actual = ctx.tensor_to_f32(&ported);
            for index in 0..rows {
                println!("[dbg] columns={columns} row={index}: cpu={:.2} ml={:.2}", reference[index], actual[index]);
                if index > 5 {
                    break;
                }
            }
            for (index, (want, got)) in reference.iter().zip(&actual).enumerate() {
                assert!((want - got).abs() < 2.0e-2 * want.abs().max(1.0), "columns={columns} row={index}: cpu={want} ml={got}");
            }
        }
    }
}

/// rmsnorm → gated gemv 共享一个 encoder(norm dispatch + barrier + gemv dispatch),
/// 省去一个 encoder 边界(~12µs)。
#[allow(clippy::too_many_arguments)]
pub fn mlx_affine_rmsnorm_gated_gemv_f16_u4(
    ctx: &MetalContext,
    hidden: &MetalTensor,
    norm_weight: &metal::Buffer,
    norm_eps: f32,
    gate_packed: &metal::Buffer,
    gate_scales: &metal::Buffer,
    gate_biases: &metal::Buffer,
    up_packed: &metal::Buffer,
    up_scales: &metal::Buffer,
    up_biases: &metal::Buffer,
    scale_dtype: u32,
    group_size: usize,
    weight_rows: usize,
    weight_cols: usize,
    activation: &crate::moe::Activation,
) -> Result<MetalTensor, String> {
    if hidden.rows != 1 || hidden.cols != weight_cols || !matches!(hidden.dtype, MetalTensorDType::F16 | MetalTensorDType::Bf16) || group_size == 0 || !weight_cols.is_multiple_of(group_size) {
        return Err(format!("rmsnorm+gated input=[{},{}] weight=[{weight_rows},{weight_cols}] 不兼容", hidden.rows, hidden.cols));
    }
    let normed = ctx.tensor_zeros(1, weight_cols);
    let output = ctx.tensor_zeros(1, weight_rows);
    // Bf16 输入(混合精度层上游 8-bit gemv 输出)直读,不再先 cast 到 F16
    let norm_pipeline = ctx.pipeline(match hidden.dtype {
        MetalTensorDType::F16 => "rms_norm_f16_simd",
        _ => "rms_norm_bf16_in_f16_simd",
    })?;
    let gemv_pipeline = ctx.pipeline("mlx_affine_gated_gemv_f16_u4")?;
    let norm_columns = validate_u32("rmsnorm+gated norm columns", weight_cols)?;
    let norm_offset: f32 = 0.0; // plain RMSNorm(no +1 offset)
    let gemv_columns = validate_u32("rmsnorm+gated gemv columns", weight_cols)?;
    let gemv_rows = validate_u32("rmsnorm+gated gemv rows", weight_rows)?;
    let gemv_group = validate_u32("rmsnorm+gated group", group_size)?;
    let act = GatedActivation::from_spec(activation)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();

    // dispatch 1: rmsnorm
    encoder.set_compute_pipeline_state(&norm_pipeline);
    encoder.set_buffer(0, Some(&hidden.buffer), 0);
    encoder.set_buffer(1, Some(norm_weight), 0);
    encoder.set_buffer(2, Some(&normed.buffer), 0);
    set_bytes(&encoder, 3, &norm_columns);
    set_bytes(&encoder, 4, &norm_eps);
    set_bytes(&encoder, 5, &norm_offset);
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));

    // 数据依赖:norm 的写入对 gemv 可见
    encoder.memory_barrier();

    // dispatch 2: gated gemv(读 normed)
    encoder.set_compute_pipeline_state(&gemv_pipeline);
    encoder.set_buffer(0, Some(&normed.buffer), 0);
    encoder.set_buffer(1, Some(gate_packed), 0);
    encoder.set_buffer(2, Some(gate_scales), 0);
    encoder.set_buffer(3, Some(gate_biases), 0);
    encoder.set_buffer(4, Some(up_packed), 0);
    encoder.set_buffer(5, Some(up_scales), 0);
    encoder.set_buffer(6, Some(up_biases), 0);
    encoder.set_buffer(7, Some(&output.buffer), 0);
    set_bytes(&encoder, 8, &gemv_columns);
    set_bytes(&encoder, 9, &gemv_rows);
    set_bytes(&encoder, 10, &gemv_group);
    set_bytes(&encoder, 11, &scale_dtype);
    set_bytes(&encoder, 12, &act.kind);
    set_bytes(&encoder, 13, &act.alpha);
    set_bytes(&encoder, 14, &act.limit);
    encoder.dispatch_thread_groups(MTLSize::new(weight_rows.div_ceil(8) as u64, 1, 1), MTLSize::new(64, 1, 1));

    encoder.end_encoding();
    let shape = format!("input=[1,{weight_cols}],weights=2x[{weight_rows},{weight_cols}]");
    ctx.commit_and_wait_profiled(&command, "rmsnorm+gated_gemv", &shape, hidden.buffer.length() + gate_packed.length() + up_packed.length(), output.buffer.length());
    Ok(output)
}

/// rmsnorm → triple gemv 共享一个 encoder(norm dispatch + barrier + triple dispatch)。
#[allow(clippy::too_many_arguments)]
pub fn mlx_affine_rmsnorm_triple_gemv_f16_u4(
    ctx: &MetalContext,
    hidden: &MetalTensor,
    norm_weight: &metal::Buffer,
    norm_eps: f32,
    first_packed: &metal::Buffer,
    first_scales: &metal::Buffer,
    first_biases: &metal::Buffer,
    first_rows: usize,
    second_packed: &metal::Buffer,
    second_scales: &metal::Buffer,
    second_biases: &metal::Buffer,
    second_rows: usize,
    third_packed: &metal::Buffer,
    third_scales: &metal::Buffer,
    third_biases: &metal::Buffer,
    third_rows: usize,
    scale_dtype: u32,
    group_size: usize,
    cols: usize,
) -> Result<(MetalTensor, MetalTensor, MetalTensor), String> {
    if hidden.rows != 1 || hidden.cols != cols || !matches!(hidden.dtype, MetalTensorDType::F16 | MetalTensorDType::Bf16) || group_size == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("rmsnorm+triple input=[{},{}] 不兼容", hidden.rows, hidden.cols));
    }
    let normed = ctx.tensor_zeros(1, cols);
    let first = ctx.tensor_zeros(1, first_rows);
    let second = ctx.tensor_zeros(1, second_rows);
    let third = ctx.tensor_zeros(1, third_rows);
    // Bf16 输入直读(同 gated 版)
    let norm_pipeline = ctx.pipeline(match hidden.dtype {
        MetalTensorDType::F16 => "rms_norm_f16_simd",
        _ => "rms_norm_bf16_in_f16_simd",
    })?;
    let triple_pipeline = ctx.pipeline("mlx_affine_triple_gemv_f16_u4")?;
    let norm_columns = validate_u32("rmsnorm+triple norm cols", cols)?;
    let norm_offset: f32 = 0.0;
    let triple_cols = validate_u32("rmsnorm+triple cols", cols)?;
    let f_rows = validate_u32("rmsnorm+triple first rows", first_rows)?;
    let s_rows = validate_u32("rmsnorm+triple second rows", second_rows)?;
    let t_rows = validate_u32("rmsnorm+triple third rows", third_rows)?;
    let triple_group = validate_u32("rmsnorm+triple group", group_size)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();

    // dispatch 1: rmsnorm
    encoder.set_compute_pipeline_state(&norm_pipeline);
    encoder.set_buffer(0, Some(&hidden.buffer), 0);
    encoder.set_buffer(1, Some(norm_weight), 0);
    encoder.set_buffer(2, Some(&normed.buffer), 0);
    set_bytes(&encoder, 3, &norm_columns);
    set_bytes(&encoder, 4, &norm_eps);
    set_bytes(&encoder, 5, &norm_offset);
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));

    encoder.memory_barrier();

    // dispatch 2: triple gemv
    encoder.set_compute_pipeline_state(&triple_pipeline);
    encoder.set_buffer(0, Some(&normed.buffer), 0);
    encoder.set_buffer(1, Some(first_packed), 0);
    encoder.set_buffer(2, Some(first_scales), 0);
    encoder.set_buffer(3, Some(first_biases), 0);
    encoder.set_buffer(4, Some(second_packed), 0);
    encoder.set_buffer(5, Some(second_scales), 0);
    encoder.set_buffer(6, Some(second_biases), 0);
    encoder.set_buffer(7, Some(third_packed), 0);
    encoder.set_buffer(8, Some(third_scales), 0);
    encoder.set_buffer(9, Some(third_biases), 0);
    encoder.set_buffer(10, Some(&first.buffer), 0);
    encoder.set_buffer(11, Some(&second.buffer), 0);
    encoder.set_buffer(12, Some(&third.buffer), 0);
    set_bytes(&encoder, 13, &triple_cols);
    set_bytes(&encoder, 14, &f_rows);
    set_bytes(&encoder, 15, &s_rows);
    set_bytes(&encoder, 16, &t_rows);
    set_bytes(&encoder, 17, &triple_group);
    set_bytes(&encoder, 18, &scale_dtype);
    let total_rows = first_rows + second_rows + third_rows;
    encoder.dispatch_thread_groups(MTLSize::new(total_rows.div_ceil(8) as u64, 1, 1), MTLSize::new(64, 1, 1));

    encoder.end_encoding();
    let shape = format!("input=[1,{cols}],weights=[{first_rows}+{second_rows}+{third_rows},{cols}]");
    ctx.commit_and_wait_profiled(
        &command,
        "rmsnorm+triple_gemv",
        &shape,
        hidden.buffer.length() + first_packed.length() + second_packed.length() + third_packed.length(),
        first.buffer.length() + second.buffer.length() + third.buffer.length(),
    );
    Ok((first, second, third))
}
