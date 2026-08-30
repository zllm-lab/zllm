    unsigned long long packed;
    unsigned long long scales;
    unsigned int group_size;
    unsigned int scale_dtype;
    unsigned int format;
};

struct ZllmCtExpertMeta {
    ZllmCtWeightMeta gate;
    ZllmCtWeightMeta up;
    ZllmCtWeightMeta down;
};

// RDNA3 没有 FP8 dot；official FP8 仅在 weight tile 装填时逐位转 BF16，
// codes/F32 scale 始终保持压缩常驻。位拼接与 block_fp8 GEMV 的已验证实现一致。
__device__ __forceinline__ float zllm_ct_e4m3_to_f32(unsigned char bits)
{
    const unsigned int exponent = (bits >> 3u) & 15u;
    const unsigned int mantissa = bits & 7u;
    const unsigned int sign = ((unsigned int)bits & 128u) << 24u;
    const float normal = __uint_as_float(sign | ((exponent + 120u) << 23u) | (mantissa << 20u));
    const float subnormal = __uint_as_float(sign | __float_as_uint((float)mantissa * 0.001953125f));
    const float finite = exponent == 0u ? subnormal : normal;
    return exponent == 15u && mantissa == 7u ? __uint_as_float(sign | 0x7fc00000u) : finite;
}

// Router 输出保持 device-resident；按 expert 紧凑排列 B*top_k 条 route。
// offsets 只为实际可能活跃的 slot 分配，N_e=0 的尾 slot 指向 route_count。
extern "C" __global__ void ct_group_decode_routes(
    const unsigned int* route_ids,
    const float* route_weights,
    const ZllmCtExpertMeta* expert_metas,
    unsigned int* grouped_tokens,
    float* grouped_weights,
    unsigned int* route_to_grouped,
    unsigned int* grouped_offsets,
    ZllmCtExpertMeta* grouped_metas,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int expert_count,
    unsigned int slot_count)
{
    extern __shared__ unsigned int route_state[];
    unsigned int* counts = route_state;
    unsigned int* expert_slots = counts + expert_count;
    unsigned int* cursors = expert_slots + expert_count;
    const unsigned int lane = threadIdx.x;
    for (unsigned int expert = lane; expert < expert_count; expert += blockDim.x) {
        counts[expert] = 0u;
        expert_slots[expert] = 0xffffffffu;
        cursors[expert] = 0u;
    }
    __syncthreads();
    for (unsigned int route = lane; route < route_count; route += blockDim.x) {
        const unsigned int expert = route_ids[route];
        if (expert < expert_count) atomicAdd(&counts[expert], 1u);
    }
    __syncthreads();
    if (lane == 0u) {
        unsigned int active = 0u;
        unsigned int offset = 0u;
        for (unsigned int expert = 0u; expert < expert_count; ++expert) {
            const unsigned int count = counts[expert];
            if (count == 0u) continue;
            expert_slots[expert] = active;
            grouped_offsets[active] = offset;
            grouped_metas[active] = expert_metas[expert];
            cursors[active] = offset;
            offset += count;
            ++active;
        }
        for (unsigned int slot = active; slot < slot_count; ++slot) {
            grouped_offsets[slot] = offset;
            grouped_metas[slot] = expert_metas[0];
            cursors[slot] = offset;
        }
        grouped_offsets[slot_count] = offset;
    }
    __syncthreads();

    // 每个线程负责一个 expert，按原 route 顺序写入；避免单线程 scatter，
    // 同时保持同 expert 内的确定性累加顺序。
    for (unsigned int expert = lane; expert < expert_count; expert += blockDim.x) {
        const unsigned int slot = expert_slots[expert];
        if (slot == 0xffffffffu) continue;
        unsigned int grouped = grouped_offsets[slot];
        for (unsigned int route = 0u; route < route_count; ++route) {
            if (route_ids[route] != expert) continue;
            grouped_tokens[grouped] = route / top_k;
            grouped_weights[grouped] = route_weights[route];
            route_to_grouped[route] = grouped;
            ++grouped;
        }
    }
}

__device__ __forceinline__ void zllm_grouped_dot(
    const unsigned short* input,
    const unsigned int* token_ids,
    const ZllmCtWeightMeta& weight_meta,
    unsigned int route_begin,
    unsigned int input_base,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_row,
    float* sums)
{
    const unsigned int lane = threadIdx.x;
    if (weight_meta.format == 1u) {
        const unsigned char* codes = (const unsigned char*)weight_meta.packed;
        const unsigned char* scales = (const unsigned char*)weight_meta.scales;
        const unsigned int scale_columns = (input_columns + weight_meta.group_size - 1u) / weight_meta.group_size;
        for (unsigned int column = lane; column < input_columns; column += blockDim.x) {
            const float scale = zllm_scale(
                scales,
                (unsigned long long)(output_row / weight_meta.group_size) * scale_columns + column / weight_meta.group_size,
                weight_meta.scale_dtype);
            const float value = zllm_ct_e4m3_to_f32(codes[(unsigned long long)output_row * input_columns + column]) * scale;
            #pragma unroll
            for (unsigned int row = 0; row < 16; ++row) {
                if (input_base + row < input_rows) {
                    const unsigned int route = route_begin + input_base + row;
                    const unsigned int input_row = token_ids ? token_ids[route] : route;
                    sums[row] += zllm_bf16_to_f32(input[(unsigned long long)input_row * input_columns + column]) * value;
                }
            }
        }
        return;
    }
    const unsigned int values_per_word = 8;
    const unsigned long long packed_columns =
        ((unsigned long long)input_columns + values_per_word - 1) / values_per_word;
    const unsigned long long groups = input_columns / weight_meta.group_size;
    const unsigned int* packed = (const unsigned int*)weight_meta.packed;
    const unsigned char* scales = (const unsigned char*)weight_meta.scales;
    for (unsigned long long packed_column = lane; packed_column < packed_columns; packed_column += 256) {
        const unsigned int column_base = packed_column * values_per_word;
        const unsigned int word = packed[(unsigned long long)output_row * packed_columns + packed_column];
        const bool one_group = column_base / weight_meta.group_size ==
            (column_base + values_per_word - 1) / weight_meta.group_size;
        const unsigned long long row_groups = (unsigned long long)output_row * groups;
        const float word_scale = one_group
            ? zllm_scale(scales, row_groups + column_base / weight_meta.group_size, weight_meta.scale_dtype)
            : 0.0f;
        #pragma unroll
        for (unsigned int element = 0; element < 8; ++element) {
            const unsigned int column = column_base + element;
            if (column < input_columns) {
                const unsigned int code = (word >> (element * 4)) & 15u;
                const float scale = one_group ? word_scale : zllm_scale(
                    scales, row_groups + column / weight_meta.group_size, weight_meta.scale_dtype);
                const float value = (float)((int)code - 8) * scale;
                #pragma unroll
                for (unsigned int row = 0; row < 16; ++row) {
                    if (input_base + row < input_rows) {
                        const unsigned int route = route_begin + input_base + row;
                        const unsigned int input_row = token_ids ? token_ids[route] : route;
                        sums[row] += zllm_bf16_to_f32(
                            input[(unsigned long long)input_row * input_columns + column]) * value;
                    }
                }
            }
        }
    }
}

// 两个独立 W8 projection 合并为一次 dispatch；grid.z 选择权重，
// 保持每个输出行一个完整 wave，不降低权重流的并行度。
extern "C" __global__ void ct_quantized_dual_gemv_bf16_w8_scalar(
    const unsigned short* input,
    const unsigned int* first_packed,
    const unsigned char* first_scales,
    float* first_output,
    unsigned int first_output_rows,
    unsigned int first_scale_dtype,
    const unsigned int* second_packed,
    const unsigned char* second_scales,
    float* second_output,
    unsigned int second_output_rows,
    unsigned int second_scale_dtype,
    unsigned int input_columns
) {
    const bool second = blockIdx.z != 0u;
    const unsigned int* packed = second ? second_packed : first_packed;
    const unsigned char* scales = second ? second_scales : first_scales;
    float* output = second ? second_output : first_output;
    const unsigned int output_rows =
        second ? second_output_rows : first_output_rows;
    const unsigned int scale_dtype =
        second ? second_scale_dtype : first_scale_dtype;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wave = threadIdx.x >> 5u;
    const unsigned int outputs_per_block = blockDim.x >> 5u;
    const unsigned int output_row =
        blockIdx.x * outputs_per_block + wave;
    if (output_row >= output_rows) return;

    const unsigned int group_count = input_columns / 128u;
    const unsigned long long packed_base =
        (unsigned long long)output_row * (input_columns / 4u);
    float even_sum = 0.0f;
    float odd_sum = 0.0f;
    unsigned int group = 0u;
    for (; group + 1u < group_count; group += 2u) {
        even_sum += zllm_ct_w8_group128_wave32_dot(
            input, packed, 0u, packed_base, group, lane) * zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group,
                scale_dtype);
        odd_sum += zllm_ct_w8_group128_wave32_dot(
            input, packed, 0u, packed_base, group + 1u, lane) * zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group + 1u,
                scale_dtype);
    }
    if (group < group_count) {
        even_sum += zllm_ct_w8_group128_wave32_dot(
            input, packed, 0u, packed_base, group, lane) * zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group,
                scale_dtype);
    }
    float sum = even_sum + odd_sum;
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        sum += __shfl_down(sum, offset, 32);
    }
    if (lane == 0u) output[output_row] = sum;
}

// 两个独立 W4G128 projection 共用一次 dispatch 和 activation 转换。
// grid.z 只选择权重，单个 16-lane subgroup 的累加顺序与单路快路径一致。
extern "C" __global__ void ct_quantized_dual_gemv_bf16_w4_scalar(
    const unsigned short* input,
    const unsigned int* first_packed,
    const unsigned char* first_scales,
    float* first_output,
    unsigned int first_output_rows,
    unsigned int first_scale_dtype,
    const unsigned int* second_packed,
    const unsigned char* second_scales,
    float* second_output,
    unsigned int second_output_rows,
    unsigned int second_scale_dtype,
    unsigned int input_columns
) {
    const bool second = blockIdx.z != 0u;
    const unsigned int* packed = second ? second_packed : first_packed;
    const unsigned char* scales = second ? second_scales : first_scales;
    float* output = second ? second_output : first_output;
    const unsigned int output_rows =
        second ? second_output_rows : first_output_rows;
    const unsigned int scale_dtype =
        second ? second_scale_dtype : first_scale_dtype;
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * (blockDim.x >> 4u) + subgroup;
    if (output_row >= output_rows) return;

    const unsigned int group_count = input_columns / 128u;
    const unsigned long long packed_base =
        (unsigned long long)output_row * (input_columns / 8u);
    float even_sum = 0.0f;
    float odd_sum = 0.0f;
    unsigned int group = 0u;
    for (; group + 1u < group_count; group += 2u) {
        even_sum += zllm_ct_w4_group128_dot(
            input, packed, 0u, packed_base, group, lane) * zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group,
                scale_dtype);
        odd_sum += zllm_ct_w4_group128_dot(
            input, packed, 0u, packed_base, group + 1u, lane) * zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group + 1u,
                scale_dtype);
    }
    if (group < group_count) {
        even_sum += zllm_ct_w4_group128_dot(
            input, packed, 0u, packed_base, group, lane) * zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group,
                scale_dtype);
    }
    float sum = even_sum + odd_sum;
    for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
        sum += __shfl_down(sum, offset, 16);
    }
    if (lane == 0u) output[output_row] = sum;
}

// 2-8 行的两个独立 W4 projection 合并为一次 dispatch。grid.z 选择权重，
// 每个 projection 内仍让所有行共享一次 code/scale 装载；既不把 B3 退回
// 逐行读权重，也给 q_a/kv_a 两个无依赖分支同时提供可发射 block。
extern "C" __global__ void ct_quantized_dual_gemv_bf16_w4_rows8(
    const unsigned short* input,
    const unsigned int* first_packed,
    const unsigned char* first_scales,
    float* first_output,
    unsigned int first_output_rows,
    unsigned int first_scale_dtype,
    const unsigned int* second_packed,
    const unsigned char* second_scales,
    float* second_output,
    unsigned int second_output_rows,
    unsigned int second_scale_dtype,
    unsigned int input_columns,
    unsigned int input_rows)
{
    if (input_rows < 2u || input_rows > 8u) return;
    const bool second = blockIdx.z != 0u;
    const unsigned int* packed = second ? second_packed : first_packed;
    const unsigned char* scales = second ? second_scales : first_scales;
    float* output = second ? second_output : first_output;
    const unsigned int output_rows = second ? second_output_rows : first_output_rows;
    const unsigned int scale_dtype = second ? second_scale_dtype : first_scale_dtype;
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * (blockDim.x >> 4u) + subgroup;
    if (output_row >= output_rows) return;

    const unsigned int groups = input_columns / 128u;
    const unsigned long long packed_base =
        (unsigned long long)output_row * (input_columns / 8u);
    const unsigned long long scale_base =
        (unsigned long long)output_row * groups;
    float even[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float odd[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int group = 0u; group < groups; ++group) {
        const unsigned int word = packed[packed_base + group * 16u + lane];
        const unsigned int column = group * 128u + lane * 8u;
        const zllm_ct_bf16x2 w01 = {
            zllm_ct_i8_to_bf16(120u + (word & 0x0fu)),
            zllm_ct_i8_to_bf16(120u + ((word >> 4u) & 0x0fu))
        };
        const zllm_ct_bf16x2 w23 = {
            zllm_ct_i8_to_bf16(120u + ((word >> 8u) & 0x0fu)),
            zllm_ct_i8_to_bf16(120u + ((word >> 12u) & 0x0fu))
        };
        const zllm_ct_bf16x2 w45 = {
            zllm_ct_i8_to_bf16(120u + ((word >> 16u) & 0x0fu)),
            zllm_ct_i8_to_bf16(120u + ((word >> 20u) & 0x0fu))
        };
        const zllm_ct_bf16x2 w67 = {
            zllm_ct_i8_to_bf16(120u + ((word >> 24u) & 0x0fu)),
            zllm_ct_i8_to_bf16(120u + ((word >> 28u) & 0x0fu))
        };
        const float scale = zllm_scale(scales, scale_base + group, scale_dtype);
        for (unsigned int row = 0u; row < input_rows; ++row) {
            const unsigned long long base = (unsigned long long)row * input_columns + column;
            const zllm_ct_bf16x2 x01 = {input[base], input[base + 1u]};
            const zllm_ct_bf16x2 x23 = {input[base + 2u], input[base + 3u]};
            const zllm_ct_bf16x2 x45 = {input[base + 4u], input[base + 5u]};
            const zllm_ct_bf16x2 x67 = {input[base + 6u], input[base + 7u]};
            float dot = __builtin_amdgcn_fdot2_f32_bf16(x01, w01, 0.0f, false);
            dot = __builtin_amdgcn_fdot2_f32_bf16(x23, w23, dot, false);
            dot = __builtin_amdgcn_fdot2_f32_bf16(x45, w45, dot, false);
            dot = __builtin_amdgcn_fdot2_f32_bf16(x67, w67, dot, false) * scale;
            if ((group & 1u) == 0u) even[row] += dot;
            else odd[row] += dot;
        }
    }
    for (unsigned int row = 0u; row < input_rows; ++row) {
        float sum = even[row] + odd[row];
        for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
            sum += __shfl_down(sum, offset, 16);
        }
        if (lane == 0u) {
            output[(unsigned long long)row * output_rows + output_row] = sum;
        }
    }
}

// W4 decode 在寄存器内组成 BF16 operand，避免发散 constant-table 读取。
__device__ __forceinline__ unsigned short zllm_ct_i4_to_bf16(unsigned int code)
{
    union { float value; unsigned int bits; } converted;
    converted.value = (float)((int)(code & 15u) - 8);
    return (unsigned short)(converted.bits >> 16u);
}

// Decode 单 token 直接消费 router 的 device route，不建立 grouped workspace。
struct ZllmDecodeGateUpSums {
    float gate;
    float up;
};

// 单个 packed word 的 gate/up 解码与乘加；批量预取路径与逐组通用路径共用
// 同一份指令序列，保证两条路径的 fdot2/FMA 顺序逐条一致。
__device__ __forceinline__ void zllm_decode_gate_up_word(
    const unsigned short* input,
    unsigned int gate_word,
    unsigned int up_word,
    const unsigned short* i4_bf16,
    float gate_scale,
    float up_scale,
    float& gate_sum,
    float& up_sum)
{
    const zllm_ct_bf16x2 x01 = {
        input[0], input[1] };
    const zllm_ct_bf16x2 x23 = {
        input[2], input[3] };
    const zllm_ct_bf16x2 x45 = {
        input[4], input[5] };
    const zllm_ct_bf16x2 x67 = {
        input[6], input[7] };
    const zllm_ct_bf16x2 gate_w01 = {
        i4_bf16[gate_word & 15u],
        i4_bf16[(gate_word >> 4u) & 15u] };
    const zllm_ct_bf16x2 gate_w23 = {
        i4_bf16[(gate_word >> 8u) & 15u],
        i4_bf16[(gate_word >> 12u) & 15u] };
    const zllm_ct_bf16x2 gate_w45 = {
        i4_bf16[(gate_word >> 16u) & 15u],
        i4_bf16[(gate_word >> 20u) & 15u] };
    const zllm_ct_bf16x2 gate_w67 = {
        i4_bf16[(gate_word >> 24u) & 15u],
        i4_bf16[(gate_word >> 28u) & 15u] };
    const zllm_ct_bf16x2 up_w01 = {
        i4_bf16[up_word & 15u],
        i4_bf16[(up_word >> 4u) & 15u] };
    const zllm_ct_bf16x2 up_w23 = {
        i4_bf16[(up_word >> 8u) & 15u],
        i4_bf16[(up_word >> 12u) & 15u] };
    const zllm_ct_bf16x2 up_w45 = {
        i4_bf16[(up_word >> 16u) & 15u],
        i4_bf16[(up_word >> 20u) & 15u] };
    const zllm_ct_bf16x2 up_w67 = {
        i4_bf16[(up_word >> 24u) & 15u],
        i4_bf16[(up_word >> 28u) & 15u] };
    float gate_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, gate_w01, 0.0f, false);
    float up_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, up_w01, 0.0f, false);
    gate_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, gate_w23, gate_word_sum, false);
    up_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, up_w23, up_word_sum, false);
    gate_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, gate_w45, gate_word_sum, false);
    up_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, up_w45, up_word_sum, false);
    gate_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x67, gate_w67, gate_word_sum, false);
    up_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x67, up_w67, up_word_sum, false);
    gate_sum += gate_word_sum * gate_scale;
    up_sum += up_word_sum * up_scale;
}

template <bool Bf16Scales>
__device__ __forceinline__ ZllmDecodeGateUpSums zllm_decode_gate_up_dot(
    const unsigned short* input,
    const unsigned int* gate_packed,
    const unsigned int* up_packed,
    const unsigned char* gate_scales,
    const unsigned char* up_scales,
    unsigned int gate_scale_dtype,
    unsigned int up_scale_dtype,
    const unsigned short* i4_bf16,
    unsigned int input_columns,
    unsigned int output_row,
    unsigned int group_size,
    unsigned int lane)
{
    const unsigned int packed_columns = (input_columns + 7u) / 8u;
    const unsigned int groups = input_columns / group_size;
    const unsigned int words_per_group = group_size / 8u;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    unsigned int group = 0u;
    if (words_per_group == 16u) {
        // 每个 lane 每组恰好一个 packed word。按 4 组一批先发射本批全部权重与
        // scale load，再按原 group 顺序执行相同的解码/乘加序列：增加在途字节
        // 改善 HBM 延迟隐藏，累加顺序不变，输出逐位一致。
        for (; group + 4u <= groups; group += 4u) {
            unsigned int gate_words[4];
            unsigned int up_words[4];
            float gate_scale_values[4];
            float up_scale_values[4];
            const unsigned short* word_inputs[4];
            for (unsigned int j = 0u; j < 4u; ++j) {
                const unsigned long long scale_index =
                    (unsigned long long)output_row * groups + group + j;
                gate_scale_values[j] = Bf16Scales
                    ? zllm_bf16_to_f32(((const unsigned short*)gate_scales)[scale_index])
                    : zllm_scale(gate_scales, scale_index, gate_scale_dtype);
                up_scale_values[j] = Bf16Scales
                    ? zllm_bf16_to_f32(((const unsigned short*)up_scales)[scale_index])
                    : zllm_scale(up_scales, scale_index, up_scale_dtype);
                const unsigned int packed_index = (group + j) * 16u + lane;
                const unsigned long long weight_index =
                    (unsigned long long)output_row * packed_columns + packed_index;
                gate_words[j] = gate_packed[weight_index];
                up_words[j] = up_packed[weight_index];
                word_inputs[j] = input + (unsigned long long)packed_index * 8u;
            }
            for (unsigned int j = 0u; j < 4u; ++j) {
                zllm_decode_gate_up_word(
                    word_inputs[j], gate_words[j], up_words[j], i4_bf16,
                    gate_scale_values[j], up_scale_values[j],
                    gate_sum, up_sum);
            }
        }
    }
    for (; group < groups; ++group) {
        const unsigned long long scale_index =
            (unsigned long long)output_row * groups + group;
        const float gate_scale = Bf16Scales
            ? zllm_bf16_to_f32(((const unsigned short*)gate_scales)[scale_index])
            : zllm_scale(gate_scales, scale_index, gate_scale_dtype);
        const float up_scale = Bf16Scales
            ? zllm_bf16_to_f32(((const unsigned short*)up_scales)[scale_index])
            : zllm_scale(up_scales, scale_index, up_scale_dtype);
        const unsigned int word_base = group * words_per_group;
        for (unsigned int word_index = lane;
             word_index < words_per_group;
             word_index += 16u) {
            const unsigned int packed_index = word_base + word_index;
            const unsigned long long weight_index =
                (unsigned long long)output_row * packed_columns + packed_index;
            const unsigned int gate_word = gate_packed[weight_index];
            const unsigned int up_word = up_packed[weight_index];
            const unsigned int column = packed_index * 8u;
            zllm_decode_gate_up_word(
                input + column, gate_word, up_word, i4_bf16,
                gate_scale, up_scale, gate_sum, up_sum);
        }
    }
    return { gate_sum, up_sum };
}


extern "C" __global__ void ct_decode_gate_up_bf16(
    const unsigned short* input,
    const unsigned int* route_ids,
    const ZllmCtExpertMeta* experts,
    unsigned short* activated,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int expert_count,
    unsigned int shared_expert)
{
    const unsigned int route = blockIdx.z;
    const bool shared_route = route >= route_count;
    if (shared_route) {
        if (shared_expert == 0xffffffffu) return;
        input += (unsigned long long)(route - route_count) * input_columns;
    } else {
        if (top_k == 0u) return;
        input += (unsigned long long)(route / top_k) * input_columns;
    }
    const unsigned int expert = shared_route ? shared_expert : route_ids[route];
    if (!shared_route && expert >= expert_count) return;
    const ZllmCtWeightMeta& gate_weight = experts[expert].gate;
    const ZllmCtWeightMeta& up_weight = experts[expert].up;
    const unsigned int group_size = gate_weight.group_size;
    if (group_size == 0u
        || up_weight.group_size != group_size
        || group_size % 8u != 0u
        || input_columns % group_size != 0u) return;

    __shared__ unsigned short i4_bf16[16];
    if (threadIdx.x < 16u) {
        i4_bf16[threadIdx.x] = zllm_ct_i4_to_bf16(threadIdx.x);
    }
    __syncthreads();
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * 16u + subgroup;
    if (output_row >= output_rows) return;

    const unsigned int* gate_packed =
        (const unsigned int*)gate_weight.packed;
    const unsigned int* up_packed =
        (const unsigned int*)up_weight.packed;
    const unsigned char* gate_scales =
        (const unsigned char*)gate_weight.scales;
    const unsigned char* up_scales =
        (const unsigned char*)up_weight.scales;
    ZllmDecodeGateUpSums sums;
    if (gate_weight.scale_dtype == 0u && up_weight.scale_dtype == 0u) {
        sums = zllm_decode_gate_up_dot<true>(input, gate_packed, up_packed, gate_scales, up_scales, gate_weight.scale_dtype, up_weight.scale_dtype, i4_bf16, input_columns, output_row, group_size, lane);
    } else {
        sums = zllm_decode_gate_up_dot<false>(input, gate_packed, up_packed, gate_scales, up_scales, gate_weight.scale_dtype, up_weight.scale_dtype, i4_bf16, input_columns, output_row, group_size, lane);
    }
    float gate_sum = sums.gate;
    float up_sum = sums.up;
    for (unsigned int offset = 8u; offset; offset >>= 1u) {
        gate_sum += __shfl_down(gate_sum, offset, 16);
        up_sum += __shfl_down(up_sum, offset, 16);
    }
    if (lane == 0u) {
        const float silu = gate_sum / (1.0f + expf(-gate_sum));
        activated[(unsigned long long)route * output_rows + output_row] =
            zllm_f32_to_bf16(silu * up_sum);
    }
}

extern "C" __global__ void ct_grouped_linear_bf16(
    const unsigned short* input,
    const unsigned int* token_ids,
    const ZllmCtExpertMeta* experts,
    const unsigned int* route_offsets,
    float* output,
    const float* gate,
    unsigned short* activated,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int expert_base,
    unsigned int expert_count,
    unsigned int projection,
    unsigned int dynamic_rows)
{
    __shared__ unsigned short zllm_fused_i4_bf16[16];
    const bool zllm_singleton_pass = gridDim.x != output_rows;
    // 单 token 时 gate/up 独立读同一 hidden；在一个 kernel 中并行累加，
    // 保留两份独立权重语义，只消除重复 input 读取与第二次完整 projection。
    if (zllm_singleton_pass
        && blockIdx.y == 0u
        && blockIdx.z < expert_count) {
        const unsigned int zllm_fused_expert =
            expert_base + blockIdx.z;
        const unsigned int zllm_fused_route =
            route_offsets[zllm_fused_expert];
        const unsigned int zllm_fused_route_end =
            route_offsets[zllm_fused_expert + 1u];
        const ZllmCtWeightMeta& zllm_fused_gate_weight =
            experts[zllm_fused_expert].gate;
        const ZllmCtWeightMeta& zllm_fused_up_weight =
            experts[zllm_fused_expert].up;
        const unsigned int zllm_fused_group_size =
            zllm_fused_gate_weight.group_size;
        if (zllm_fused_route_end == zllm_fused_route + 1u
            && zllm_fused_group_size != 0u
            && zllm_fused_up_weight.group_size == zllm_fused_group_size
            && zllm_fused_gate_weight.scale_dtype == 0u
            && zllm_fused_up_weight.scale_dtype == 0u
            && input_columns % zllm_fused_group_size == 0u
            && zllm_fused_group_size % 8u == 0u) {
            // projection=0 已经同时生成 activated，projection=1 只保留
            // 原有 host 调度边界，避免改变 backend/runtime 接口。
            if (projection != 0u) return;
            if (threadIdx.x < 16u) {
                zllm_fused_i4_bf16[threadIdx.x] =
                    zllm_ct_i4_to_bf16(threadIdx.x);
            }
            __syncthreads();
            const unsigned int zllm_fused_lane = threadIdx.x & 15u;
            const unsigned int zllm_fused_subgroup = threadIdx.x >> 4u;
            const unsigned int zllm_fused_output_row =
                blockIdx.x * 16u + zllm_fused_subgroup;
            if (zllm_fused_output_row >= output_rows) return;
            const unsigned int zllm_fused_token =
                token_ids
                    ? token_ids[zllm_fused_route]
                    : zllm_fused_route;
            const unsigned int* zllm_fused_gate_packed =
                (const unsigned int*)zllm_fused_gate_weight.packed;
            const unsigned int* zllm_fused_up_packed =
                (const unsigned int*)zllm_fused_up_weight.packed;
            const unsigned char* zllm_fused_gate_scales =
                (const unsigned char*)zllm_fused_gate_weight.scales;
            const unsigned char* zllm_fused_up_scales =
                (const unsigned char*)zllm_fused_up_weight.scales;
            const unsigned int zllm_fused_packed_columns =
                (input_columns + 7u) / 8u;
            const unsigned int zllm_fused_groups =
                input_columns / zllm_fused_group_size;
            const unsigned int zllm_fused_words_per_group =
                zllm_fused_group_size / 8u;
            float zllm_fused_gate_sum = 0.0f;
            float zllm_fused_up_sum = 0.0f;
            for (unsigned int zllm_fused_group = 0u;
                 zllm_fused_group < zllm_fused_groups;
                 ++zllm_fused_group) {
                const unsigned long long zllm_fused_scale_index =
                    (unsigned long long)zllm_fused_output_row
                        * zllm_fused_groups
                        + zllm_fused_group;
                const float zllm_fused_gate_scale = zllm_bf16_to_f32(
                    ((const unsigned short*)zllm_fused_gate_scales)[
                        zllm_fused_scale_index]);
                const float zllm_fused_up_scale = zllm_bf16_to_f32(
                    ((const unsigned short*)zllm_fused_up_scales)[
                        zllm_fused_scale_index]);
                const unsigned int zllm_fused_word_base =
                    zllm_fused_group * zllm_fused_words_per_group;
                for (unsigned int zllm_fused_word_index =
                         zllm_fused_lane;
                     zllm_fused_word_index
                         < zllm_fused_words_per_group;
                     zllm_fused_word_index += 16u) {
                    const unsigned int zllm_fused_packed_index =
                        zllm_fused_word_base
                            + zllm_fused_word_index;
                    const unsigned long long zllm_fused_weight_index =
                        (unsigned long long)zllm_fused_output_row
                            * zllm_fused_packed_columns
                            + zllm_fused_packed_index;
                    const unsigned int zllm_fused_gate_word =
                        zllm_fused_gate_packed[
                            zllm_fused_weight_index];
                    const unsigned int zllm_fused_up_word =
                        zllm_fused_up_packed[
                            zllm_fused_weight_index];
                    const unsigned int zllm_fused_column =
                        zllm_fused_packed_index * 8u;
                    const unsigned long long zllm_fused_input_base =
                        (unsigned long long)zllm_fused_token
                            * input_columns
                            + zllm_fused_column;
                    const zllm_ct_bf16x2 zllm_fused_x01 = {
                        input[zllm_fused_input_base],
                        input[zllm_fused_input_base + 1u] };
                    const zllm_ct_bf16x2 zllm_fused_x23 = {
                        input[zllm_fused_input_base + 2u],
                        input[zllm_fused_input_base + 3u] };
                    const zllm_ct_bf16x2 zllm_fused_x45 = {
                        input[zllm_fused_input_base + 4u],
                        input[zllm_fused_input_base + 5u] };
                    const zllm_ct_bf16x2 zllm_fused_x67 = {
                        input[zllm_fused_input_base + 6u],
                        input[zllm_fused_input_base + 7u] };
                    const zllm_ct_bf16x2 zllm_fused_gate_w01 = {
                        zllm_fused_i4_bf16[zllm_fused_gate_word & 15u],
                        zllm_fused_i4_bf16[(zllm_fused_gate_word >> 4u) & 15u] };
                    const zllm_ct_bf16x2 zllm_fused_gate_w23 = {
                        zllm_fused_i4_bf16[(zllm_fused_gate_word >> 8u) & 15u],
                        zllm_fused_i4_bf16[(zllm_fused_gate_word >> 12u) & 15u] };
                    const zllm_ct_bf16x2 zllm_fused_gate_w45 = {
                        zllm_fused_i4_bf16[(zllm_fused_gate_word >> 16u) & 15u],
                        zllm_fused_i4_bf16[(zllm_fused_gate_word >> 20u) & 15u] };
                    const zllm_ct_bf16x2 zllm_fused_gate_w67 = {
                        zllm_fused_i4_bf16[(zllm_fused_gate_word >> 24u) & 15u],
                        zllm_fused_i4_bf16[(zllm_fused_gate_word >> 28u) & 15u] };
                    const zllm_ct_bf16x2 zllm_fused_up_w01 = {
                        zllm_fused_i4_bf16[zllm_fused_up_word & 15u],
                        zllm_fused_i4_bf16[(zllm_fused_up_word >> 4u) & 15u] };
                    const zllm_ct_bf16x2 zllm_fused_up_w23 = {
                        zllm_fused_i4_bf16[(zllm_fused_up_word >> 8u) & 15u],
                        zllm_fused_i4_bf16[(zllm_fused_up_word >> 12u) & 15u] };
                    const zllm_ct_bf16x2 zllm_fused_up_w45 = {
                        zllm_fused_i4_bf16[(zllm_fused_up_word >> 16u) & 15u],
                        zllm_fused_i4_bf16[(zllm_fused_up_word >> 20u) & 15u] };
                    const zllm_ct_bf16x2 zllm_fused_up_w67 = {
                        zllm_fused_i4_bf16[(zllm_fused_up_word >> 24u) & 15u],
                        zllm_fused_i4_bf16[(zllm_fused_up_word >> 28u) & 15u] };
                    float zllm_fused_gate_word_sum =
                        __builtin_amdgcn_fdot2_f32_bf16(
                            zllm_fused_x01,
                            zllm_fused_gate_w01,
                            0.0f,
                            false);
                    float zllm_fused_up_word_sum =
                        __builtin_amdgcn_fdot2_f32_bf16(
                            zllm_fused_x01,
                            zllm_fused_up_w01,
                            0.0f,
                            false);
                    zllm_fused_gate_word_sum =
                        __builtin_amdgcn_fdot2_f32_bf16(
                            zllm_fused_x23,
                            zllm_fused_gate_w23,
                            zllm_fused_gate_word_sum,
                            false);
                    zllm_fused_up_word_sum =
                        __builtin_amdgcn_fdot2_f32_bf16(
                            zllm_fused_x23,
                            zllm_fused_up_w23,
                            zllm_fused_up_word_sum,
                            false);
                    zllm_fused_gate_word_sum =
                        __builtin_amdgcn_fdot2_f32_bf16(
                            zllm_fused_x45,
                            zllm_fused_gate_w45,
                            zllm_fused_gate_word_sum,
                            false);
                    zllm_fused_up_word_sum =
                        __builtin_amdgcn_fdot2_f32_bf16(
                            zllm_fused_x45,
                            zllm_fused_up_w45,
                            zllm_fused_up_word_sum,
                            false);
                    zllm_fused_gate_word_sum =
                        __builtin_amdgcn_fdot2_f32_bf16(
                            zllm_fused_x67,
                            zllm_fused_gate_w67,
                            zllm_fused_gate_word_sum,
                            false);
                    zllm_fused_up_word_sum =
                        __builtin_amdgcn_fdot2_f32_bf16(
                            zllm_fused_x67,
                            zllm_fused_up_w67,
                            zllm_fused_up_word_sum,
                            false);
                    zllm_fused_gate_sum +=
                        zllm_fused_gate_word_sum
                            * zllm_fused_gate_scale;
                    zllm_fused_up_sum +=
                        zllm_fused_up_word_sum
                            * zllm_fused_up_scale;
                }
            }
            for (unsigned int zllm_fused_offset = 8u;
                 zllm_fused_offset;
                 zllm_fused_offset >>= 1u) {
                zllm_fused_gate_sum += __shfl_down(
                    zllm_fused_gate_sum,
                    zllm_fused_offset,
                    16);
                zllm_fused_up_sum += __shfl_down(
                    zllm_fused_up_sum,
                    zllm_fused_offset,
                    16);
            }
            if (zllm_fused_lane == 0u) {
                const unsigned long long zllm_fused_output_index =
                    (unsigned long long)zllm_fused_route
                        * output_rows
                        + zllm_fused_output_row;
                const float zllm_fused_value =
                    (zllm_fused_gate_sum
                        / (1.0f + expf(-zllm_fused_gate_sum)))
                        * zllm_fused_up_sum;
                activated[zllm_fused_output_index] =
                    zllm_f32_to_bf16(zllm_fused_value);
            }
            return;
        }
    }

    // official FP8 的动态 route 也走同一个紧凑 pass。WMMA 处理不少于
    // 4 行的 expert；这里每个 16-lane subgroup 一次补齐 1..3 行，避免
    // 为 288 个 slot 发射 output_rows x experts 的宽网格空 block。
    if (zllm_singleton_pass && blockIdx.y == 0u && blockIdx.z < expert_count) {
        const unsigned int expert = expert_base + blockIdx.z;
        const unsigned int route_begin = route_offsets[expert];
        const unsigned int route_end = route_offsets[expert + 1u];
        const unsigned int route_rows = route_end - route_begin;
        const ZllmCtWeightMeta& gate_weight = experts[expert].gate;
        const ZllmCtWeightMeta& up_weight = experts[expert].up;
        if (route_rows > 0u && route_rows < 4u
            && gate_weight.format == 1u && up_weight.format == 1u) {
            if (projection != 0u) return;
            const unsigned int lane = threadIdx.x & 15u;
            const unsigned int subgroup = threadIdx.x >> 4u;
            const unsigned int output_row = blockIdx.x * 16u + subgroup;
            if (output_row >= output_rows) return;
            const unsigned char* gate_codes =
                (const unsigned char*)gate_weight.packed;
            const unsigned char* gate_scales =
                (const unsigned char*)gate_weight.scales;
            const unsigned char* up_codes =
                (const unsigned char*)up_weight.packed;
            const unsigned char* up_scales =
                (const unsigned char*)up_weight.scales;
            const unsigned int gate_scale_columns =
                (input_columns + gate_weight.group_size - 1u)
                    / gate_weight.group_size;
            const unsigned int up_scale_columns =
                (input_columns + up_weight.group_size - 1u)
                    / up_weight.group_size;
            for (unsigned int route = route_begin; route < route_end; ++route) {
                const unsigned int token = token_ids ? token_ids[route] : route;
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                for (unsigned int column = lane; column < input_columns; column += 16u) {
                    const float input_value = zllm_bf16_to_f32(
                        input[(unsigned long long)token * input_columns + column]);
                    const float gate_scale = zllm_scale(
                        gate_scales,
                        (unsigned long long)(output_row / gate_weight.group_size)
                                * gate_scale_columns
                            + column / gate_weight.group_size,
                        gate_weight.scale_dtype);
                    const float up_scale = zllm_scale(
                        up_scales,
                        (unsigned long long)(output_row / up_weight.group_size)
                                * up_scale_columns
                            + column / up_weight.group_size,
                        up_weight.scale_dtype);
                    gate_sum += input_value * zllm_ct_e4m3_to_f32(
                        gate_codes[(unsigned long long)output_row * input_columns + column]) * gate_scale;
                    up_sum += input_value * zllm_ct_e4m3_to_f32(
                        up_codes[(unsigned long long)output_row * input_columns + column]) * up_scale;
                }
                for (unsigned int offset = 8u; offset; offset >>= 1u) {
                    gate_sum += __shfl_down(gate_sum, offset, 16);
                    up_sum += __shfl_down(up_sum, offset, 16);
                }
                if (lane == 0u) {
                    activated[(unsigned long long)route * output_rows + output_row]
                        = zllm_f32_to_bf16(
                            gate_sum / (1.0f + expf(-gate_sum)) * up_sum);
                }
            }
            return;
        }
    }

    const unsigned int zllm_decode_expert = expert_base + blockIdx.z;
    if (blockIdx.z >= expert_count) return;
    const unsigned int zllm_decode_route_begin = route_offsets[zllm_decode_expert];
    const unsigned int zllm_decode_input_rows =
        route_offsets[zllm_decode_expert + 1] - zllm_decode_route_begin;
    // 多 session 合并后先用紧凑网格处理 singleton route；宽网格只补多 route
    // expert，避免每个 singleton expert 产生约 15/16 的空 block。
    if (zllm_singleton_pass != (zllm_decode_input_rows == 1u)) return;
    if (zllm_decode_input_rows == 1u) {
        const unsigned int zllm_lane = threadIdx.x & 15u;
        const unsigned int zllm_subgroup = threadIdx.x >> 4u;
        const unsigned int zllm_output_row = blockIdx.x * 16u + zllm_subgroup;
        if (zllm_output_row >= output_rows) return;
        const ZllmCtWeightMeta& zllm_weight = projection == 0
            ? experts[zllm_decode_expert].gate : experts[zllm_decode_expert].up;
        const unsigned int* zllm_packed = (const unsigned int*)zllm_weight.packed;
        const unsigned char* zllm_scales = (const unsigned char*)zllm_weight.scales;
        const unsigned int zllm_packed_columns = (input_columns + 7u) / 8u;
        const unsigned int zllm_groups = input_columns / zllm_weight.group_size;
        const unsigned int zllm_words_per_group = zllm_weight.group_size / 8u;
        const unsigned int zllm_input_row = token_ids
            ? token_ids[zllm_decode_route_begin] : zllm_decode_route_begin;
        float zllm_sum = 0.0f;
        for (unsigned int zllm_group = 0; zllm_group < zllm_groups; ++zllm_group) {
            const float zllm_scale_value = zllm_scale(
                zllm_scales,
                (unsigned long long)zllm_output_row * zllm_groups + zllm_group,
                zllm_weight.scale_dtype);
            const unsigned int zllm_word_base = zllm_group * zllm_words_per_group;
            for (unsigned int zllm_word_index = zllm_lane;
                 zllm_word_index < zllm_words_per_group;
                 zllm_word_index += 16u) {
                const unsigned int zllm_word = zllm_packed[
                    (unsigned long long)zllm_output_row * zllm_packed_columns
                    + zllm_word_base + zllm_word_index];
                const unsigned int zllm_column =
                    (zllm_word_base + zllm_word_index) * 8u;
                const unsigned long long zllm_input_base =
                    (unsigned long long)zllm_input_row * input_columns + zllm_column;
                const zllm_ct_bf16x2 zllm_x01 = {
                    input[zllm_input_base], input[zllm_input_base + 1u] };
                const zllm_ct_bf16x2 zllm_x23 = {
                    input[zllm_input_base + 2u], input[zllm_input_base + 3u] };
                const zllm_ct_bf16x2 zllm_x45 = {
                    input[zllm_input_base + 4u], input[zllm_input_base + 5u] };
                const zllm_ct_bf16x2 zllm_x67 = {
                    input[zllm_input_base + 6u], input[zllm_input_base + 7u] };
                const zllm_ct_bf16x2 zllm_w01 = {
                    zllm_ct_i4_to_bf16(zllm_word & 15u),
                    zllm_ct_i4_to_bf16((zllm_word >> 4u) & 15u) };
                const zllm_ct_bf16x2 zllm_w23 = {
                    zllm_ct_i4_to_bf16((zllm_word >> 8u) & 15u),
                    zllm_ct_i4_to_bf16((zllm_word >> 12u) & 15u) };
                const zllm_ct_bf16x2 zllm_w45 = {
                    zllm_ct_i4_to_bf16((zllm_word >> 16u) & 15u),
                    zllm_ct_i4_to_bf16((zllm_word >> 20u) & 15u) };
                const zllm_ct_bf16x2 zllm_w67 = {
                    zllm_ct_i4_to_bf16((zllm_word >> 24u) & 15u),
                    zllm_ct_i4_to_bf16((zllm_word >> 28u) & 15u) };
                float zllm_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
                    zllm_x01, zllm_w01, 0.0f, false);
                zllm_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
                    zllm_x23, zllm_w23, zllm_word_sum, false);
                zllm_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
                    zllm_x45, zllm_w45, zllm_word_sum, false);
                zllm_word_sum = __builtin_amdgcn_fdot2_f32_bf16(
                    zllm_x67, zllm_w67, zllm_word_sum, false);
                zllm_sum += zllm_word_sum * zllm_scale_value;
            }
        }
        for (unsigned int zllm_offset = 8u; zllm_offset; zllm_offset >>= 1u)
            zllm_sum += __shfl_down(zllm_sum, zllm_offset, 16);
        if (zllm_lane == 0u) {
            const unsigned long long zllm_index =
                (unsigned long long)zllm_decode_route_begin * output_rows + zllm_output_row;
            if (projection == 0) {
                output[zllm_index] = zllm_sum;
            } else {
                const float zllm_gate = gate[zllm_index];
                activated[zllm_index] = zllm_f32_to_bf16(
                    zllm_gate / (1.0f + expf(-zllm_gate)) * zllm_sum);
            }
        }
        return;
    }

    const unsigned int expert = expert_base + blockIdx.z;
    const unsigned int output_row = blockIdx.x;
    const unsigned int input_base = blockIdx.y * 16;
    if (blockIdx.z >= expert_count || output_row >= output_rows) return;
    const unsigned int route_begin = route_offsets[expert];
    const unsigned int input_rows = route_offsets[expert + 1] - route_begin;
    if ((!dynamic_rows && input_rows >= 4u) || input_base >= input_rows) return;
    const ZllmCtWeightMeta& weight = projection == 0 ? experts[expert].gate : experts[expert].up;
    float sums[16] = {0.0f};
    zllm_grouped_dot(
        input, token_ids, weight, route_begin, input_base, input_rows,
        input_columns, output_row, sums);
    #pragma unroll
    for (unsigned int row = 0; row < 16; ++row) {
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sums[row] += __shfl_down(sums[row], offset);
        }
    }
    __shared__ float partial[16][8];
    const unsigned int wave_lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    if (wave_lane == 0) {
        #pragma unroll
        for (unsigned int row = 0; row < 16; ++row) partial[row][wave] = sums[row];
    }
    __syncthreads();
    if (wave == 0) {
        float totals[16];
        #pragma unroll
        for (unsigned int row = 0; row < 16; ++row) totals[row] = wave_lane < 8 ? partial[row][wave_lane] : 0.0f;
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            #pragma unroll
            for (unsigned int row = 0; row < 16; ++row) totals[row] += __shfl_down(totals[row], offset);
        }
        if (wave_lane == 0) {
            #pragma unroll
            for (unsigned int row = 0; row < 16; ++row) {
                if (input_base + row < input_rows) {
                    const unsigned long long index =
                        (unsigned long long)(route_begin + input_base + row) * output_rows + output_row;
                    if (projection == 0) {
                        output[index] = totals[row];
                    } else {
                        const float gate_value = gate[index];
                        activated[index] = zllm_f32_to_bf16(
                            gate_value / (1.0f + expf(-gate_value)) * totals[row]);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void ct_grouped_gate_up_wmma_bf16(
    const unsigned short* input,
    const unsigned int* token_ids,
    const ZllmCtExpertMeta* experts,
    const unsigned int* route_offsets,
    unsigned short* activated,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int expert_base,
    unsigned int expert_count)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    const unsigned int expert = expert_base + blockIdx.z;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int projection = wave >> 2;
    const unsigned int tile_wave = wave & 3u;
    const unsigned int first_input_base = blockIdx.y * 128;
    const unsigned int output_base = (blockIdx.x * 4 + tile_wave) * 16;
    if (blockIdx.z >= expert_count || wave >= 8u) return;
    const unsigned int route_begin = route_offsets[expert];
    const unsigned int input_rows = route_offsets[expert + 1] - route_begin;
    if (input_rows < 4u || first_input_base >= input_rows) return;
    const ZllmCtWeightMeta& weight_meta
        = projection == 0u ? experts[expert].gate : experts[expert].up;
    const unsigned int* packed = (const unsigned int*)weight_meta.packed;
    const unsigned char* fp8_codes = (const unsigned char*)weight_meta.packed;
    const unsigned char* scales = (const unsigned char*)weight_meta.scales;
    const unsigned long long packed_columns = ((unsigned long long)input_columns + 7) / 8;
    const unsigned long long groups = input_columns / weight_meta.group_size;
    const unsigned long long fp8_scale_columns = (input_columns + weight_meta.group_size - 1u) / weight_meta.group_size;
    __shared__ unsigned short input_tiles[16][16 * 16];
    __shared__ ZllmWmmaScratch scratch;
    // device route 无法在 host 侧得知最大 expert 行数。dynamic launch 只发
    // 一个 y block，由同一 expert/output tile 继续覆盖后续 128-row tile。
    for (unsigned int input_base = first_input_base;
         input_base < input_rows;
         input_base += gridDim.y * 128u) {
    CFragment accumulator_fragments[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (input_base + tile * 16 < input_rows)
            fill_fragment(accumulator_fragments[tile], 0.0f);
    }
    // K32 单 stage：input_tiles 翻倍到 16 个（两组 8×16 列），每次迭代一次装载
    // 32 列、一次 barrier，替代原先每 16 列两次 barrier。stage_prefetch 预取下
    // 一组 32 列进寄存器；装载映射 64 线程一个 16x16 tile，256 线程四轮覆盖 16 个。
    const unsigned int stage_row = (threadIdx.x >> 2) & 15u;
    const unsigned int stage_chunk = threadIdx.x & 3u;
    const unsigned int stage_group = threadIdx.x >> 6;
    uint2 stage_data[4];
    unsigned int stage_valid[4] = {0u, 0u, 0u, 0u};
    auto stage_prefetch = [&](unsigned int prefetch_base) {
        #pragma unroll
        for (unsigned int half = 0; half < 4; ++half) {
            const unsigned int tile = stage_group + half * 4;
            const unsigned int column_offset = (half >> 1) * 16u;
            const unsigned int route_row = input_base + (tile & 7u) * 16 + stage_row;
            stage_valid[half] = (route_row < input_rows && prefetch_base + column_offset < input_columns) ? 1u : 0u;
            const unsigned int token = stage_valid[half]
                ? token_ids[route_begin + route_row] : 0u;
            stage_data[half] = stage_valid[half]
                ? *reinterpret_cast<const uint2*>(
                    input + (unsigned long long)token * input_columns
                    + prefetch_base + column_offset + stage_chunk * 4)
                : make_uint2(0u, 0u);
        }
    };
    stage_prefetch(0);
    for (unsigned int column_base = 0; column_base < input_columns; column_base += 32) {
        // 装载 32 列进 16 个 tile：half 0/1 写低 16 列（tile 0-7），half 2/3 写高 16 列（tile 8-15）。
        #pragma unroll
        for (unsigned int half = 0; half < 4; ++half) {
            const unsigned int tile = stage_group + half * 4;
            if (stage_valid[half]) {
                *reinterpret_cast<uint2*>(
                    &input_tiles[tile][stage_row * 16 + stage_chunk * 4])
                    = stage_data[half];
            }
        }
        if (column_base + 32 < input_columns) stage_prefetch(column_base + 32);
        __syncthreads();
        #pragma unroll
        for (unsigned int segment = 0; segment < 2; ++segment) {
            const unsigned int seg_column_base = column_base + segment * 16;
            if (seg_column_base >= input_columns) break;
        BFragment weight_fragment;
        if (weight_meta.format == 1u) {
            static_assert(BFragment::num_elements == 8, "GFX11 FP8 B fragment lane layout 已改变");
            const unsigned int output_row = output_base + (lane & 15u);
            const unsigned int lane_column = seg_column_base + (lane >> 4u) * 8u;
            unsigned long long codes = 0u;
            float scale = 0.0f;
            if (output_row < output_rows) {
                codes = *reinterpret_cast<const unsigned long long*>(
                    fp8_codes + (unsigned long long)output_row * input_columns + lane_column);
                scale = zllm_scale(
                    scales,
                    (unsigned long long)(output_row / weight_meta.group_size) * fp8_scale_columns
                        + lane_column / weight_meta.group_size,
                    weight_meta.scale_dtype);
            }
            #pragma unroll
            for (unsigned int element = 0; element < 8; ++element) {
                weight_fragment[element].data = zllm_f32_to_bf16(
                    zllm_ct_e4m3_to_f32((unsigned char)(codes >> (element * 8u))) * scale);
            }
        } else {
            static_assert(BFragment::num_elements == 8, "GFX11 W4 B fragment lane layout 已改变");
            const unsigned int output_row = output_base + (lane & 15u);
            const unsigned int lane_column = seg_column_base + (lane >> 4u) * 8u;
            unsigned int word = 0u;
            float scale = 0.0f;
            if (output_row < output_rows) {
                const unsigned long long row_words
                    = (unsigned long long)output_row * packed_columns;
                word = packed[row_words + lane_column / 8u];
                scale = zllm_scale(
                    scales,
                    (unsigned long long)output_row * groups
                        + seg_column_base / weight_meta.group_size,
                    weight_meta.scale_dtype);
            }
            #pragma unroll
            for (unsigned int element = 0; element < 8; ++element) {
                const unsigned int code = (word >> (element * 4u)) & 15u;
                weight_fragment[element].data = zllm_f32_to_bf16(
                    (float)((int)code - 8) * scale);
            }
        }
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            if (input_base + tile * 16 < input_rows) {
                AFragment input_fragment;
                load_matrix_sync(input_fragment, (const bfloat16_t*)input_tiles[segment * 8 + tile], 16);
                mma_sync(
                    accumulator_fragments[tile],
                    input_fragment,
                    weight_fragment,
                    accumulator_fragments[tile]);
            }
        }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (input_base + tile * 16 < input_rows)
            store_matrix_sync(
                scratch.output[wave], accumulator_fragments[tile], 16, mem_row_major);
        __syncthreads();
        if (wave < 4u && input_base + tile * 16 < input_rows) {
            for (unsigned int index = lane; index < 16 * 16; index += warpSize) {
                const unsigned int local_row = input_base + tile * 16 + index / 16;
                const unsigned int output_row = output_base + index % 16;
                if (local_row < input_rows && output_row < output_rows) {
                    const unsigned long long output_index
                        = (unsigned long long)(route_begin + local_row) * output_rows + output_row;
                    const float gate = scratch.output[wave][index];
                    const float up = scratch.output[wave + 4u][index];
                    activated[output_index]
                        = zllm_f32_to_bf16(gate / (1.0f + expf(-gate)) * up);
                }
            }
        }
        __syncthreads();
    }
    }
}

extern "C" __global__ void ct_grouped_zero_f32(float* output, unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = 0.0f;
}

// 单个 packed word 的 down 解码与乘加；批量预取路径与逐组通用路径共用
// 同一份指令序列，保证两条路径的 fdot2/FMA 顺序逐条一致。
__device__ __forceinline__ float zllm_grouped_down_decode_word(
    const unsigned short* input,
    unsigned int word,
    float scale_value,
    float sum)
{
    const zllm_ct_bf16x2 x01 = {
        input[0], input[1] };
    const zllm_ct_bf16x2 x23 = {
        input[2], input[3] };
    const zllm_ct_bf16x2 x45 = {
        input[4], input[5] };
    const zllm_ct_bf16x2 x67 = {
        input[6], input[7] };
    const zllm_ct_bf16x2 w01 = {
        zllm_ct_i4_to_bf16(word & 15u),
        zllm_ct_i4_to_bf16((word >> 4u) & 15u) };
    const zllm_ct_bf16x2 w23 = {
        zllm_ct_i4_to_bf16((word >> 8u) & 15u),
        zllm_ct_i4_to_bf16((word >> 12u) & 15u) };
    const zllm_ct_bf16x2 w45 = {
        zllm_ct_i4_to_bf16((word >> 16u) & 15u),
        zllm_ct_i4_to_bf16((word >> 20u) & 15u) };
    const zllm_ct_bf16x2 w67 = {
        zllm_ct_i4_to_bf16((word >> 24u) & 15u),
        zllm_ct_i4_to_bf16((word >> 28u) & 15u) };
    float word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, w01, 0.0f, false);
    word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, w23, word_sum, false);
    word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, w45, word_sum, false);
    word_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x67, w67, word_sum, false);
    return sum + word_sum * scale_value;
}

template <bool Bf16Scale>
__device__ __forceinline__ float zllm_grouped_down_decode_dot(
    const unsigned short* input,
    unsigned int route,
    float route_weight,
    const ZllmCtWeightMeta& weight,
    unsigned int output_row,
    unsigned int input_columns,
    unsigned int lane,
    float sum)
{
    if (weight.format == 1u) {
        const unsigned char* codes = (const unsigned char*)weight.packed;
        const unsigned char* scales = (const unsigned char*)weight.scales;
        const unsigned int scale_columns = (input_columns + weight.group_size - 1u) / weight.group_size;
        for (unsigned int column = lane; column < input_columns; column += 16u) {
            const float scale = zllm_scale(
                scales,
                (unsigned long long)(output_row / weight.group_size) * scale_columns + column / weight.group_size,
                weight.scale_dtype);
            const float value = zllm_ct_e4m3_to_f32(codes[(unsigned long long)output_row * input_columns + column]) * scale * route_weight;
            sum += zllm_bf16_to_f32(input[(unsigned long long)route * input_columns + column]) * value;
        }
        return sum;
    }
    const unsigned int* packed = (const unsigned int*)weight.packed;
    const unsigned char* scales = (const unsigned char*)weight.scales;
    const unsigned int packed_columns = (input_columns + 7u) / 8u;
    const unsigned int groups = input_columns / weight.group_size;
    const unsigned int words_per_group = weight.group_size / 8u;
    unsigned int group = 0u;
    if (words_per_group == 16u) {
        // 每个 lane 每组恰好一个 packed word。按 4 组一批先发射本批全部权重与
        // scale load，再按原 group 顺序执行相同的解码/乘加序列：增加在途字节
        // 改善 HBM 延迟隐藏，累加顺序不变，输出逐位一致。
        for (; group + 4u <= groups; group += 4u) {
            unsigned int words[4];
            float scale_values[4];
            const unsigned short* word_inputs[4];
            for (unsigned int j = 0u; j < 4u; ++j) {
                const unsigned long long scale_index =
                    (unsigned long long)output_row * groups + group + j;
                scale_values[j] = (
                    Bf16Scale
                        ? zllm_bf16_to_f32(((const unsigned short*)scales)[scale_index])
                        : zllm_scale(scales, scale_index, weight.scale_dtype)) * route_weight;
                const unsigned int packed_index = (group + j) * 16u + lane;
                words[j] = packed[
                    (unsigned long long)output_row * packed_columns + packed_index];
                word_inputs[j] = input +
                    (unsigned long long)route * input_columns + (unsigned long long)packed_index * 8u;
            }
            for (unsigned int j = 0u; j < 4u; ++j) {
                sum = zllm_grouped_down_decode_word(
                    word_inputs[j], words[j], scale_values[j], sum);
            }
        }
    }
    for (; group < groups; ++group) {
        const unsigned long long scale_index =
            (unsigned long long)output_row * groups + group;
        const float scale_value = (
            Bf16Scale
                ? zllm_bf16_to_f32(((const unsigned short*)scales)[scale_index])
                : zllm_scale(scales, scale_index, weight.scale_dtype)) * route_weight;
        const unsigned int word_base = group * words_per_group;
        for (unsigned int word_index = lane;
             word_index < words_per_group;
             word_index += 16u) {
            const unsigned int packed_index = word_base + word_index;
            const unsigned int word = packed[
                (unsigned long long)output_row * packed_columns + packed_index];
            const unsigned int column = packed_index * 8u;
            sum = zllm_grouped_down_decode_word(
                input + (unsigned long long)route * input_columns + column,
                word, scale_value, sum);
        }
    }
    return sum;
}

// Decode down 直接按 route ID 取 expert；只在 block 开头排序 slot，
// 保持与 grouped 路径相同的 expert-id F32 累加顺序。
extern "C" __global__ void ct_decode_down_bf16(
    const unsigned short* input,
    const unsigned int* route_ids,
    const float* route_weights,
    const ZllmCtExpertMeta* experts,
    float* output,
    const float* shared_output,
    const float* residual,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int expert_count,
    unsigned int shared_expert,
    unsigned int fused_epilogue)
{
    const unsigned int token = blockIdx.y;
    const unsigned int route_base = token * top_k;
    if (top_k == 0u || top_k > 16u || route_base + top_k > route_count) return;
    __shared__ unsigned int sorted_slots[16];
    if (threadIdx.x == 0u) {
        for (unsigned int slot = 0u; slot < top_k; ++slot) {
            unsigned int position = slot;
            while (position > 0u
                && route_ids[route_base + sorted_slots[position - 1u]] > route_ids[route_base + slot]) {
                sorted_slots[position] = sorted_slots[position - 1u];
                --position;
            }
            sorted_slots[position] = slot;
        }
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * 16u + subgroup;
    if (output_row >= output_rows) return;
    float sum = 0.0f;
    for (unsigned int rank = 0u; rank < top_k; ++rank) {
        const unsigned int route = route_base + sorted_slots[rank];
        const unsigned int expert = route_ids[route];
        if (expert >= expert_count) continue;
        const ZllmCtWeightMeta& weight = experts[expert].down;
        if (weight.scale_dtype == 0u) {
            sum = zllm_grouped_down_decode_dot<true>(
                input, route, route_weights[route], weight,
                output_row, input_columns, lane, sum);
        } else {
            sum = zllm_grouped_down_decode_dot<false>(
                input, route, route_weights[route], weight,
                output_row, input_columns, lane, sum);
        }
    }
    float shared_sum = 0.0f;
    if (shared_expert != 0xffffffffu) {
        const ZllmCtWeightMeta& weight = experts[shared_expert].down;
        const unsigned int route = route_count + token;
        if (weight.scale_dtype == 0u) {
            shared_sum = zllm_grouped_down_decode_dot<true>(
                input, route, 1.0f, weight,
                output_row, input_columns, lane, 0.0f);
        } else {
            shared_sum = zllm_grouped_down_decode_dot<false>(
                input, route, 1.0f, weight,
                output_row, input_columns, lane, 0.0f);
        }
    }
    for (unsigned int offset = 8u; offset; offset >>= 1u) {
        sum += __shfl_down(sum, offset, 16);
        shared_sum += __shfl_down(shared_sum, offset, 16);
    }
    if (lane == 0u) {
        const unsigned long long output_index =
            (unsigned long long)token * output_rows + output_row;
        output[output_index] = fused_epilogue == 2u
            ? residual[output_index] + (shared_sum + sum)
            : fused_epilogue == 1u
                ? residual[output_index] + (shared_output[output_index] + sum)
                : sum;
    }
}

// MTP L78 的 compressed-tensors expert 是逐行 scale 的 W8。单路直接消费
// device route，用两个 kernel 完成全部 top-k expert，避免逐 expert 反复 launch。
__device__ __forceinline__ ZllmDecodeGateUpSums zllm_decode_gate_up_w8_dot(
    const unsigned short* input,
    const ZllmCtWeightMeta& gate_weight,
    const ZllmCtWeightMeta& up_weight,
    unsigned int input_columns,
    unsigned int output_row,
    unsigned int lane)
{
    const unsigned int groups = input_columns / gate_weight.group_size;
    const unsigned int words_per_group = gate_weight.group_size / 4u;
    const unsigned int packed_columns = input_columns / 4u;
    const unsigned int* gate_packed = (const unsigned int*)gate_weight.packed;
    const unsigned int* up_packed = (const unsigned int*)up_weight.packed;
    const unsigned char* gate_scales = (const unsigned char*)gate_weight.scales;
    const unsigned char* up_scales = (const unsigned char*)up_weight.scales;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (unsigned int group = 0u; group < groups; ++group) {
        const unsigned long long scale_index =
            (unsigned long long)output_row * groups + group;
        const float gate_scale = zllm_scale(
            gate_scales, scale_index, gate_weight.scale_dtype);
        const float up_scale = zllm_scale(
            up_scales, scale_index, up_weight.scale_dtype);
        const unsigned int word_base = group * words_per_group;
        for (unsigned int word_index = lane;
             word_index < words_per_group;
             word_index += 16u) {
            const unsigned int packed_index = word_base + word_index;
            const unsigned long long weight_index =
                (unsigned long long)output_row * packed_columns + packed_index;
            const unsigned int gate_codes = gate_packed[weight_index];
            const unsigned int up_codes = up_packed[weight_index];
            const unsigned int column = packed_index * 4u;
            const zllm_ct_bf16x2 x01 = { input[column], input[column + 1u] };
            const zllm_ct_bf16x2 x23 = { input[column + 2u], input[column + 3u] };
            const zllm_ct_bf16x2 gate_w01 = {
                zllm_ct_i8_to_bf16(gate_codes),
                zllm_ct_i8_to_bf16(gate_codes >> 8u) };
            const zllm_ct_bf16x2 gate_w23 = {
                zllm_ct_i8_to_bf16(gate_codes >> 16u),
                zllm_ct_i8_to_bf16(gate_codes >> 24u) };
            const zllm_ct_bf16x2 up_w01 = {
                zllm_ct_i8_to_bf16(up_codes),
                zllm_ct_i8_to_bf16(up_codes >> 8u) };
            const zllm_ct_bf16x2 up_w23 = {
                zllm_ct_i8_to_bf16(up_codes >> 16u),
                zllm_ct_i8_to_bf16(up_codes >> 24u) };
            float gate_word = __builtin_amdgcn_fdot2_f32_bf16(
                x01, gate_w01, 0.0f, false);
            gate_word = __builtin_amdgcn_fdot2_f32_bf16(
                x23, gate_w23, gate_word, false);
            float up_word = __builtin_amdgcn_fdot2_f32_bf16(
                x01, up_w01, 0.0f, false);
            up_word = __builtin_amdgcn_fdot2_f32_bf16(
                x23, up_w23, up_word, false);
            gate_sum += gate_word * gate_scale;
            up_sum += up_word * up_scale;
        }
    }
    return { gate_sum, up_sum };
}

extern "C" __global__ void ct_decode_gate_up_w8_bf16(
    const unsigned short* input,
    const unsigned int* route_ids,
    const ZllmCtExpertMeta* experts,
    unsigned short* activated,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int expert_count,
    unsigned int shared_expert)
{
    const unsigned int route = blockIdx.z;
    const bool shared_route = route >= route_count;
    if (shared_route && shared_expert == 0xffffffffu) return;
    if (!shared_route && top_k == 0u) return;
    const unsigned int expert = shared_route ? shared_expert : route_ids[route];
    if (!shared_route && expert >= expert_count) return;
    const ZllmCtWeightMeta& gate_weight = experts[expert].gate;
    const ZllmCtWeightMeta& up_weight = experts[expert].up;
    if (gate_weight.group_size == 0u
        || gate_weight.group_size != up_weight.group_size
        || input_columns % gate_weight.group_size != 0u) return;
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * 16u + subgroup;
    if (output_row >= output_rows) return;
    input += (unsigned long long)(shared_route ? route - route_count : route / top_k) * input_columns;
    ZllmDecodeGateUpSums sums = zllm_decode_gate_up_w8_dot(
        input, gate_weight, up_weight, input_columns, output_row, lane);
    for (unsigned int offset = 8u; offset; offset >>= 1u) {
        sums.gate += __shfl_down(sums.gate, offset, 16);
        sums.up += __shfl_down(sums.up, offset, 16);
    }
    if (lane == 0u) {
        const float silu = sums.gate / (1.0f + expf(-sums.gate));
        activated[(unsigned long long)route * output_rows + output_row] =
            zllm_f32_to_bf16(silu * sums.up);
    }
}

__device__ __forceinline__ float zllm_decode_down_w8_dot(
    const unsigned short* input,
    const ZllmCtWeightMeta& weight,
    unsigned int input_columns,
    unsigned int output_row,
    unsigned int lane)
{
    const unsigned int groups = input_columns / weight.group_size;
    const unsigned int words_per_group = weight.group_size / 4u;
    const unsigned int packed_columns = input_columns / 4u;
    const unsigned int* packed = (const unsigned int*)weight.packed;
    const unsigned char* scales = (const unsigned char*)weight.scales;
    float sum = 0.0f;
    for (unsigned int group = 0u; group < groups; ++group) {
        const float scale = zllm_scale(
            scales,
            (unsigned long long)output_row * groups + group,
            weight.scale_dtype);
        const unsigned int word_base = group * words_per_group;
        for (unsigned int word_index = lane;
             word_index < words_per_group;
             word_index += 16u) {
            const unsigned int packed_index = word_base + word_index;
            const unsigned int codes = packed[
                (unsigned long long)output_row * packed_columns + packed_index];
            const unsigned int column = packed_index * 4u;
            const zllm_ct_bf16x2 x01 = { input[column], input[column + 1u] };
            const zllm_ct_bf16x2 x23 = { input[column + 2u], input[column + 3u] };
            const zllm_ct_bf16x2 w01 = {
                zllm_ct_i8_to_bf16(codes),
                zllm_ct_i8_to_bf16(codes >> 8u) };
            const zllm_ct_bf16x2 w23 = {
                zllm_ct_i8_to_bf16(codes >> 16u),
                zllm_ct_i8_to_bf16(codes >> 24u) };
            float word_sum = __builtin_amdgcn_fdot2_f32_bf16(
                x01, w01, 0.0f, false);
            word_sum = __builtin_amdgcn_fdot2_f32_bf16(
                x23, w23, word_sum, false);
            sum += word_sum * scale;
        }
    }
    return sum;
}

extern "C" __global__ void ct_decode_down_w8_bf16(
    const unsigned short* input,
    const unsigned int* route_ids,
    const float* route_weights,
    const ZllmCtExpertMeta* experts,
    float* output,
    const float* shared_output,
    const float* residual,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int expert_count,
    unsigned int shared_expert,
    unsigned int fused_epilogue)
{
    const unsigned int token = blockIdx.y;
    const unsigned int route_base = token * top_k;
    if (top_k == 0u || top_k > 16u || route_base + top_k > route_count) return;
    __shared__ unsigned int sorted_slots[16];
    if (threadIdx.x == 0u) {
        for (unsigned int slot = 0u; slot < top_k; ++slot) {
            unsigned int position = slot;
            while (position > 0u
                && route_ids[route_base + sorted_slots[position - 1u]]
                    > route_ids[route_base + slot]) {
                sorted_slots[position] = sorted_slots[position - 1u];
                --position;
            }
            sorted_slots[position] = slot;
        }
    }
    __syncthreads();
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * 16u + subgroup;
    if (output_row >= output_rows) return;
    float sum = 0.0f;
    for (unsigned int rank = 0u; rank < top_k; ++rank) {
        const unsigned int route = route_base + sorted_slots[rank];
        const unsigned int expert = route_ids[route];
        if (expert >= expert_count) continue;
        const ZllmCtWeightMeta& weight = experts[expert].down;
        float route_sum = zllm_decode_down_w8_dot(
            input + (unsigned long long)route * input_columns,
            weight, input_columns, output_row, lane);
        for (unsigned int offset = 8u; offset; offset >>= 1u)
            route_sum += __shfl_down(route_sum, offset, 16);
        if (lane == 0u) sum += route_sum * route_weights[route];
    }
    float shared_sum = 0.0f;
    if (shared_expert != 0xffffffffu) {
        shared_sum = zllm_decode_down_w8_dot(
            input + (unsigned long long)(route_count + token) * input_columns,
            experts[shared_expert].down, input_columns, output_row, lane);
        for (unsigned int offset = 8u; offset; offset >>= 1u)
            shared_sum += __shfl_down(shared_sum, offset, 16);
    }
    if (lane == 0u) {
        const unsigned long long output_index =
            (unsigned long long)token * output_rows + output_row;
        output[output_index] = fused_epilogue == 2u
            ? residual[output_index] + (shared_sum + sum)
            : fused_epilogue == 1u
                ? residual[output_index] + (shared_output[output_index] + sum)
                : sum;
    }
}

extern "C" __global__ void ct_grouped_down_scatter_bf16(
    const unsigned short* input,
    const unsigned int* token_ids,
    const float* route_weights,
    const ZllmCtExpertMeta* experts,
    const unsigned int* route_offsets,
    float* output,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int expert_base,
    unsigned int expert_count)
{
    if (gridDim.x != output_rows) {
        if (blockIdx.z != 0u || blockIdx.y != 0u) return;
        const unsigned int zllm_lane = threadIdx.x & 15u;
        const unsigned int zllm_subgroup = threadIdx.x >> 4u;
        const unsigned int zllm_output_row = blockIdx.x * 16u + zllm_subgroup;
        if (zllm_output_row >= output_rows) return;
        float zllm_sum = 0.0f;
        unsigned int zllm_output_token = 0u;
        for (unsigned int zllm_local_expert = 0;
             zllm_local_expert < expert_count;
             ++zllm_local_expert) {
            const unsigned int zllm_expert = expert_base + zllm_local_expert;
            const unsigned int zllm_route = route_offsets[zllm_expert];
            if (route_offsets[zllm_expert + 1] == zllm_route) continue;
            const unsigned int zllm_token = token_ids ? token_ids[zllm_route] : zllm_route;
            if (zllm_local_expert == 0u) zllm_output_token = zllm_token;
            const float zllm_route_weight = route_weights[zllm_route];
            const ZllmCtWeightMeta& zllm_weight = experts[zllm_expert].down;
            if (zllm_weight.scale_dtype == 0u)
                zllm_sum = zllm_grouped_down_decode_dot<true>(
                    input, zllm_route, zllm_route_weight, zllm_weight,
                    zllm_output_row, input_columns, zllm_lane, zllm_sum);
            else
                zllm_sum = zllm_grouped_down_decode_dot<false>(
                    input, zllm_route, zllm_route_weight, zllm_weight,
                    zllm_output_row, input_columns, zllm_lane, zllm_sum);
        }
        for (unsigned int zllm_offset = 8u; zllm_offset; zllm_offset >>= 1u)
            zllm_sum += __shfl_down(zllm_sum, zllm_offset, 16);
        if (zllm_lane == 0u)
            output[(unsigned long long)zllm_output_token * output_rows + zllm_output_row] = zllm_sum;
        return;
    }

    const unsigned int expert = expert_base + blockIdx.z;
    const unsigned int output_row = blockIdx.x;
    const unsigned int input_base = blockIdx.y * 16;
    if (blockIdx.z >= expert_count || output_row >= output_rows) return;
    const unsigned int route_begin = route_offsets[expert];
    const unsigned int input_rows = route_offsets[expert + 1] - route_begin;
    if (input_rows >= 4u || input_base >= input_rows) return;
    float sums[16] = {0.0f};
    zllm_grouped_dot(
        input, nullptr, experts[expert].down, route_begin, input_base, input_rows,
        input_columns, output_row, sums);
    #pragma unroll
    for (unsigned int row = 0; row < 16; ++row) {
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sums[row] += __shfl_down(sums[row], offset);
        }
    }
    __shared__ float partial[16][8];
    const unsigned int wave_lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    if (wave_lane == 0) {
        #pragma unroll
        for (unsigned int row = 0; row < 16; ++row) partial[row][wave] = sums[row];
    }
    __syncthreads();
    if (wave == 0) {
        float totals[16];
        #pragma unroll
        for (unsigned int row = 0; row < 16; ++row) totals[row] = wave_lane < 8 ? partial[row][wave_lane] : 0.0f;
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            #pragma unroll
            for (unsigned int row = 0; row < 16; ++row) totals[row] += __shfl_down(totals[row], offset);
        }
        if (wave_lane == 0) {
            #pragma unroll
            for (unsigned int row = 0; row < 16; ++row) {
                if (input_base + row < input_rows) {
                    const unsigned int route = route_begin + input_base + row;
                    atomicAdd(
                        &output[(unsigned long long)token_ids[route] * output_rows + output_row],
                        totals[row] * route_weights[route]);
                }
            }
        }
    }
}

extern "C" __global__ void ct_grouped_down_small_bf16(
    const unsigned short* input,
    const unsigned int* token_ids,
    const float* route_weights,
    const ZllmCtExpertMeta* experts,
    const unsigned int* route_offsets,
    float* output,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int expert_base,
    unsigned int expert_count,
    unsigned int output_row_base,
    unsigned int output_slice_rows,
    unsigned int route_major_output)
{
    const unsigned int local_expert = blockIdx.z;
    if (local_expert >= expert_count || blockIdx.y != 0u) return;
    const unsigned int expert = expert_base + local_expert;
    const unsigned int route_begin = route_offsets[expert];
    const unsigned int route_end = route_offsets[expert + 1u];
    const unsigned int input_rows = route_end - route_begin;
    if (input_rows == 0u || input_rows >= 4u) return;

    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_delta = blockIdx.x * 16u + subgroup;
    const unsigned int output_row = output_row_base + output_delta;
    if (output_delta >= output_slice_rows || output_row >= output_rows) return;
    const ZllmCtWeightMeta& weight = experts[expert].down;
    for (unsigned int route = route_begin; route < route_end; ++route) {
        float sum = 0.0f;
        if (weight.scale_dtype == 0u)
            sum = zllm_grouped_down_decode_dot<true>(
                input, route, route_weights[route], weight,
                output_row, input_columns, lane, sum);
        else
            sum = zllm_grouped_down_decode_dot<false>(
                input, route, route_weights[route], weight,
                output_row, input_columns, lane, sum);
        for (unsigned int offset = 8u; offset; offset >>= 1u)
            sum += __shfl_down(sum, offset, 16);
        if (lane == 0u) {
            const unsigned int token = token_ids ? token_ids[route] : route;
            if (route_major_output)
                output[(unsigned long long)route * output_slice_rows + output_delta] = sum;
            else
                atomicAdd(&output[(unsigned long long)token * output_rows + output_row], sum);
        }
    }
}

// Dynamic grouped route 不展开 down 输出；每个 token 按原 top-k slot 顺序完成 F32 归约。
extern "C" __global__ void ct_grouped_down_reduce_bf16(
    const unsigned short* input,
    const unsigned int* route_ids,
    const float* route_weights,
    const unsigned int* route_to_grouped,
    const ZllmCtExpertMeta* experts,
    float* output,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int expert_count)
{
    const unsigned int token = blockIdx.y;
    const unsigned int route_base = token * top_k;
    if (top_k == 0u || top_k > 16u || route_base + top_k > route_count) return;
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * 16u + subgroup;
    if (output_row >= output_rows) return;

    float total = 0.0f;
    for (unsigned int slot = 0u; slot < top_k; ++slot) {
        const unsigned int route = route_base + slot;
        const unsigned int expert = route_ids[route];
        if (expert >= expert_count) continue;
        const unsigned int grouped = route_to_grouped[route];
        const ZllmCtWeightMeta& weight = experts[expert].down;
        float route_sum = 0.0f;
        if (weight.scale_dtype == 0u)
            route_sum = zllm_grouped_down_decode_dot<true>(
                input, grouped, route_weights[route], weight,
                output_row, input_columns, lane, route_sum);
        else
            route_sum = zllm_grouped_down_decode_dot<false>(
                input, grouped, route_weights[route], weight,
                output_row, input_columns, lane, route_sum);
        for (unsigned int offset = 8u; offset; offset >>= 1u)
            route_sum += __shfl_down(route_sum, offset, 16);
        if (lane == 0u) total += route_sum;
    }
    if (lane == 0u)
        output[(unsigned long long)token * output_rows + output_row] = total;
}

// WMMA 先按 grouped route 写完整 F32 contribution，再由本 kernel 按固定 slot
// 顺序归约。矩阵阶段不再向同一 token 做 atomicAdd，因此调度顺序不影响结果。
extern "C" __global__ void ct_grouped_down_reduce_routes_f32(
    const float* route_output,
    const unsigned int* route_to_grouped,
    float* output,
    const float* shared_output,
    const float* residual,
    unsigned int output_rows,
    unsigned int output_row_base,
    unsigned int output_slice_rows,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int fused_epilogue)
{
    if (top_k == 0u || top_k > 16u) return;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int token_count = route_count / top_k;
    const unsigned int element_count = token_count * output_slice_rows;
    if (index >= element_count) return;
    const unsigned int token = index / output_slice_rows;
    const unsigned int output_delta = index - token * output_slice_rows;
    const unsigned int output_row = output_row_base + output_delta;
    if (output_row >= output_rows) return;
    const unsigned int route_base = token * top_k;
    float total = 0.0f;
    for (unsigned int slot = 0u; slot < top_k; ++slot) {
        const unsigned int grouped = route_to_grouped[route_base + slot];
        total += route_output[(unsigned long long)grouped * output_slice_rows + output_delta];
    }
    const unsigned long long output_index
        = (unsigned long long)token * output_rows + output_row;
    output[output_index] = fused_epilogue
        ? residual[output_index] + (shared_output[output_index] + total)
        : total;
}

extern "C" __global__ void ct_grouped_down_scatter_wmma_bf16(
    const unsigned short* input,
    const unsigned int* token_ids,
    const float* route_weights,
    const ZllmCtExpertMeta* experts,
    const unsigned int* route_offsets,
    float* output,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int expert_base,
    unsigned int expert_count,
    unsigned int output_row_base,
    unsigned int output_slice_rows,
    unsigned int route_major_output)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    const unsigned int expert = expert_base + blockIdx.z;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int first_input_base = blockIdx.y * 128;
    const unsigned int output_base = output_row_base + (blockIdx.x * 8 + wave) * 16;
    if (blockIdx.z >= expert_count) return;
    const unsigned int route_begin = route_offsets[expert];
    const unsigned int input_rows = route_offsets[expert + 1] - route_begin;
    if (input_rows < 4u || first_input_base >= input_rows) return;
    const ZllmCtWeightMeta& weight_meta = experts[expert].down;
    const unsigned int* packed = (const unsigned int*)weight_meta.packed;
    const unsigned char* fp8_codes = (const unsigned char*)weight_meta.packed;
    const unsigned char* scales = (const unsigned char*)weight_meta.scales;
    const unsigned long long packed_columns = ((unsigned long long)input_columns + 7) / 8;
    const unsigned long long groups = input_columns / weight_meta.group_size;
    const unsigned long long fp8_scale_columns = (input_columns + weight_meta.group_size - 1u) / weight_meta.group_size;
    __shared__ unsigned short input_tiles[8][16 * 16];
    __shared__ float output_scratch[8][16 * 16];
    for (unsigned int input_base = first_input_base;
         input_base < input_rows;
         input_base += gridDim.y * 128u) {
    CFragment accumulator_fragments[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (input_base + tile * 16 < input_rows) {
            fill_fragment(accumulator_fragments[tile], 0.0f);
        }
    }
    for (unsigned int column_base = 0; column_base < input_columns; column_base += 16) {
        const unsigned int flat = threadIdx.x;
        const unsigned int local_row = flat / 16;
        const unsigned int column = column_base + flat % 16;
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            const unsigned int route_row = input_base + tile * 16 + local_row;
            if (route_row < input_rows) {
                input_tiles[tile][flat]
                    = input[(unsigned long long)(route_begin + route_row) * input_columns + column];
            }
        }
        __syncwarp();
        BFragment weight_fragment;
        if (weight_meta.format == 1u) {
            static_assert(BFragment::num_elements == 8, "GFX11 FP8 B fragment lane layout 已改变");
            const unsigned int output_row = output_base + (lane & 15u);
            const unsigned int lane_column = column_base + (lane >> 4u) * 8u;
            unsigned long long codes = 0u;
            float scale = 0.0f;
            if (output_row < output_rows) {
                codes = *reinterpret_cast<const unsigned long long*>(
                    fp8_codes + (unsigned long long)output_row * input_columns + lane_column);
                scale = zllm_scale(
                    scales,
                    (unsigned long long)(output_row / weight_meta.group_size) * fp8_scale_columns
                        + lane_column / weight_meta.group_size,
                    weight_meta.scale_dtype);
            }
            #pragma unroll
            for (unsigned int element = 0; element < 8; ++element) {
                weight_fragment[element].data = zllm_f32_to_bf16(
                    zllm_ct_e4m3_to_f32((unsigned char)(codes >> (element * 8u))) * scale);
            }
        } else {
            // W4G128 直读：同 gate/up 的 B-fragment lane 布局，寄存器 (code-8)*scale 解包。
            static_assert(BFragment::num_elements == 8, "GFX11 W4 B fragment lane layout 已改变");
            const unsigned int output_row = output_base + (lane & 15u);
            const unsigned int lane_column = column_base + (lane >> 4u) * 8u;
            unsigned int word = 0u;
            float scale = 0.0f;
            if (output_row < output_rows) {
                const unsigned long long row_words = (unsigned long long)output_row * packed_columns;
                word = packed[row_words + lane_column / 8u];
                scale = zllm_scale(
                    scales,
                    (unsigned long long)output_row * groups + column_base / weight_meta.group_size,
                    weight_meta.scale_dtype);
            }
            #pragma unroll
            for (unsigned int element = 0; element < 8; ++element) {
                const unsigned int code = (word >> (element * 4u)) & 15u;
                weight_fragment[element].data = zllm_f32_to_bf16(
                    (float)((int)code - 8) * scale);
            }
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            if (input_base + tile * 16 < input_rows) {
                AFragment input_fragment;
                load_matrix_sync(input_fragment, (const bfloat16_t*)input_tiles[tile], 16);
                mma_sync(
                    accumulator_fragments[tile],
                    input_fragment,
                    weight_fragment,
                    accumulator_fragments[tile]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (input_base + tile * 16 >= input_rows) continue;
        store_matrix_sync(output_scratch[wave], accumulator_fragments[tile], 16, mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += warpSize) {
            const unsigned int local_row = input_base + tile * 16 + index / 16;
            const unsigned int output_row = output_base + index % 16;
            const unsigned int output_delta = output_row - output_row_base;
            if (local_row < input_rows && output_delta < output_slice_rows && output_row < output_rows) {
                const unsigned int route = route_begin + local_row;
                const float contribution = output_scratch[wave][index] * route_weights[route];
                if (route_major_output)
                    output[(unsigned long long)route * output_slice_rows + output_delta] = contribution;
                else
                    atomicAdd(&output[(unsigned long long)token_ids[route] * output_rows + output_row], contribution);
            }
