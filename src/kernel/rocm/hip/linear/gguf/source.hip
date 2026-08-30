    unsigned long long gate;
    unsigned long long up;
    unsigned long long down;
    unsigned int gate_type;
    unsigned int up_type;
    unsigned int down_type;
};

union GgufWmmaScratch {
    unsigned short lut[8][16 * 32];
    float output[8][16 * 16];
};

__device__ __forceinline__ float gguf_kq_f16(const unsigned char* bytes)
{
    const unsigned int bits = unsigned(bytes[0]) | (unsigned(bytes[1]) << 8);
    const unsigned int sign = (bits & 0x8000) << 16;
    unsigned int exponent = (bits >> 10) & 0x1f;
    unsigned int mantissa = bits & 0x03ff;
    unsigned int value;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            exponent = 113;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            value = sign | (exponent << 23) | ((mantissa & 0x03ff) << 13);
        }
    } else if (exponent == 31) {
        value = sign | 0x7f800000 | (mantissa << 13);
    } else {
        value = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    union { unsigned int bits; float number; } converted;
    converted.bits = value;
    return converted.number;
}

__device__ __forceinline__ unsigned int gguf_kq_block_bytes(const unsigned int tensor_type)
{
    return tensor_type == 12u ? 144u : (tensor_type == 13u ? 176u : 210u);
}

__device__ __forceinline__ const unsigned char* gguf_kq_row(
    const unsigned long long weight, const unsigned int tensor_type,
    const unsigned int row, const unsigned int columns)
{
    return (const unsigned char*)weight
        + (unsigned long long)row * (columns >> 8) * gguf_kq_block_bytes(tensor_type);
}

// 行基址已知时的块内解码(fused kernel 逐行循环复用基址)。
__device__ __forceinline__ float gguf_kq_value_at(
    const unsigned char* row_base, const unsigned int tensor_type,
    const unsigned int column, const unsigned int block_bytes)
{
    const unsigned char* block = row_base + (unsigned long long)(column >> 8) * block_bytes;
    const unsigned int local = column & 255;
    if (tensor_type == 14u) {
        const unsigned int half = local >> 7;
        const unsigned int slot = (local & 127) >> 5;
        const unsigned int index = local & 31;
        const unsigned char low = block[half * 64 + index + (slot & 1u) * 32];
        const unsigned int nibble = (slot < 2u) ? (low & 0x0fu) : (low >> 4);
        const unsigned int high = (block[128 + half * 32 + index] >> (slot * 2)) & 3u;
        const int quant = (int)(nibble | (high << 4)) - 32;
        const signed char group_scale = (signed char)block[192 + half * 8 + (index >> 4) + slot * 2];
        return gguf_kq_f16(block + 208) * (float)group_scale * (float)quant;
    }
    const unsigned int group = local >> 5;
    const unsigned char* scales = block + 4;
    unsigned int scale;
    unsigned int minimum;
    if (group < 4u) {
        scale = scales[group] & 0x3f;
        minimum = scales[group + 4] & 0x3f;
    } else {
        scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
    const unsigned char packed = block[(tensor_type == 12u ? 16u : 48u) + (group >> 1) * 32 + (local & 31)];
    unsigned int quant = (group & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
    if (tensor_type == 13u && (block[16 + (local & 31)] & (1u << group)) != 0u) quant += 16u;
    return gguf_kq_f16(block) * (float)(scale * quant) - gguf_kq_f16(block + 2) * (float)minimum;
}

// BF16 输入时以两个元素一组使用 AMD fdot2；量化值先转 BF16，保持与
// 预载阶段展开参数相同的舍入语义。非 BF16 输入继续走原有标量路径。
__device__ __forceinline__ float gguf_kq_pair_fdot2(
    const unsigned short* input, unsigned long long input_base,
    const unsigned char* row, unsigned int type, unsigned int column,
    unsigned int block_bytes
) {
    const zllm_ct_bf16x2 x = { input[input_base + column], input[input_base + column + 1u] };
    const zllm_ct_bf16x2 w = {
        zllm_f32_to_bf16(gguf_kq_value_at(row, type, column, block_bytes)),
        zllm_f32_to_bf16(gguf_kq_value_at(row, type, column + 1u, block_bytes))
    };
    return __builtin_amdgcn_fdot2_f32_bf16(x, w, 0.0f, false);
}

// 单元素完整反量化(标量路径)。与 codec::ggml 的 decode_q4_k/q5_k/q6_k 逐位一致。
__device__ __forceinline__ float gguf_kq_value(
    const unsigned long long weight, const unsigned int tensor_type,
    const unsigned int row, const unsigned int columns, const unsigned int column)
{
    return gguf_kq_value_at(
        gguf_kq_row(weight, tensor_type, row, columns),
        tensor_type, column, gguf_kq_block_bytes(tensor_type));
}

// tile 内单列 code 提取(WMMA LUT 路径)。
__device__ __forceinline__ unsigned int gguf_kq_code(
    const unsigned long long weight, const unsigned int tensor_type,
    const unsigned int row, const unsigned int columns, const unsigned int column)
{
    const unsigned char* block = gguf_kq_row(weight, tensor_type, row, columns)
        + (unsigned long long)(column >> 8) * gguf_kq_block_bytes(tensor_type);
    const unsigned int local = column & 255;
    if (tensor_type == 14u) {
        const unsigned int half = local >> 7;
        const unsigned int slot = (local & 127) >> 5;
        const unsigned int index = local & 31;
        const unsigned char low = block[half * 64 + index + (slot & 1u) * 32];
        const unsigned int nibble = (slot < 2u) ? (low & 0x0fu) : (low >> 4);
        const unsigned int high = (block[128 + half * 32 + index] >> (slot * 2)) & 3u;
        return nibble | (high << 4);
    }
    const unsigned int group = local >> 5;
    const unsigned char packed = block[(tensor_type == 12u ? 16u : 48u) + (group >> 1) * 32 + (local & 31)];
    const unsigned int nibble = (group & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
    if (tensor_type == 13u && (block[16 + (local & 31)] & (1u << group)) != 0u) return nibble + 16u;
    return nibble;
}

// 为 (行, tile) 构建 LUT：tile 的 16 列共享同一 32 元素组，反量化值域只有 code→bf16。
// 仅服务 gate/up(Q4_K/Q5_K，码本 ≤32)；Q6_K 只出现在 down，走标量 gguf_kq_value。
__device__ __forceinline__ void gguf_kq_fill_lut(
    unsigned short* lut_row,
    const unsigned long long weight, const unsigned int tensor_type,
    const unsigned int row, const unsigned int columns, const unsigned int column_base)
{
    const unsigned char* block = gguf_kq_row(weight, tensor_type, row, columns)
        + (unsigned long long)(column_base >> 8) * gguf_kq_block_bytes(tensor_type);
    const unsigned int local = column_base & 255;
    const unsigned int group = local >> 5;
    const unsigned char* scales = block + 4;
    unsigned int scale;
    unsigned int minimum;
    if (group < 4u) {
        scale = scales[group] & 0x3f;
        minimum = scales[group + 4] & 0x3f;
    } else {
        scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
    const float d = gguf_kq_f16(block);
    const float dmin = gguf_kq_f16(block + 2);
    if (tensor_type == 12u) {
        #pragma unroll
        for (unsigned int code = 0; code < 16; ++code)
            lut_row[code] = zllm_f32_to_bf16(d * (float)(scale * code) - dmin * (float)minimum);
    } else {
        #pragma unroll
        for (unsigned int code = 0; code < 32; ++code) {
            const unsigned int quant = (code & 15u) + 16u * (code >> 4);
            lut_row[code] = zllm_f32_to_bf16(d * (float)(scale * quant) - dmin * (float)minimum);
        }
    }
}

// ===== GGUF fused decode experts(小批次：route_count×top_k ≤ 64) =====
// 对应 CT 的 decode fused 路径：token-major，无路由分组 kernel，activated 常驻 workspace。

// 段一：每 (route, 8 intermediate 维) block 计算 gate/up 点积并写激活(bf16)。
extern "C" __global__ void gguf_fused_gate_up_f32(
    const void* input,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    unsigned short* activated,
    const unsigned int assignments,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate,
    const unsigned int input_is_bf16)
{
    const unsigned int dim_base = blockIdx.x * 8;
    const unsigned int assignment = blockIdx.y;
    if (assignment >= assignments) return;
    const unsigned int token = assignment / top_k;
    const GgufKqMeta meta = metas[route_slots[assignment]];
    const unsigned int lane = threadIdx.x;
    float sums_g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sums_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned char* gate_rows[8];
    const unsigned char* up_rows[8];
    #pragma unroll
    for (unsigned int d = 0; d < 8; ++d) {
        gate_rows[d] = gguf_kq_row(meta.gate, meta.gate_type, dim_base + d, hidden);
        up_rows[d] = gguf_kq_row(meta.up, meta.up_type, dim_base + d, hidden);
    }
    const unsigned int gate_bb = gguf_kq_block_bytes(meta.gate_type);
    const unsigned int up_bb = gguf_kq_block_bytes(meta.up_type);
    for (unsigned int column = lane; column < hidden; column += 256) {
        if (input_is_bf16 && (column & 1u) != 0u) continue;
        if (input_is_bf16 && (column & 1u) == 0u && column + 1u < hidden) {
            #pragma unroll
            for (unsigned int d = 0; d < 8; ++d) {
                sums_g[d] += gguf_kq_pair_fdot2((const unsigned short*)input, (unsigned long long)token * hidden, gate_rows[d], meta.gate_type, column, gate_bb);
                sums_u[d] += gguf_kq_pair_fdot2((const unsigned short*)input, (unsigned long long)token * hidden, up_rows[d], meta.up_type, column, up_bb);
            }
            continue;
        }
        const float x = input_is_bf16
            ? __uint_as_float((unsigned int)((const unsigned short*)input)[(unsigned long long)token * hidden + column] << 16)
            : ((const float*)input)[(unsigned long long)token * hidden + column];
        #pragma unroll
        for (unsigned int d = 0; d < 8; ++d) {
            sums_g[d] += x * gguf_kq_value_at(gate_rows[d], meta.gate_type, column, gate_bb);
            sums_u[d] += x * gguf_kq_value_at(up_rows[d], meta.up_type, column, up_bb);
        }
    }
    __shared__ float partial_g[8][256];
    __shared__ float partial_u[8][256];
    #pragma unroll
    for (unsigned int d = 0; d < 8; ++d) {
        partial_g[d][lane] = sums_g[d];
        partial_u[d][lane] = sums_u[d];
    }
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            #pragma unroll
            for (unsigned int d = 0; d < 8; ++d) {
                partial_g[d][lane] += partial_g[d][lane + stride];
                partial_u[d][lane] += partial_u[d][lane + stride];
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int d = 0; d < 8; ++d) {
            const float gate = partial_g[d][0] / (1.0f + expf(-partial_g[d][0]));
            activated[(unsigned long long)assignment * intermediate + dim_base + d]
                = zllm_f32_to_bf16(gate * partial_u[d][0]);
        }
    }
}

// 段二：每 (token, 8 hidden 维) block 跨 top_k 固定顺序寄存器累加，无原子。
extern "C" __global__ void gguf_fused_down_f32(
    const unsigned short* activated,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    const float* route_weights,
    float* output,
    const unsigned int tokens,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate)
{
    const unsigned int out_base = blockIdx.x * 8;
    const unsigned int token = blockIdx.y;
    if (token >= tokens) return;
    const unsigned int lane = threadIdx.x;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int k = 0; k < top_k; ++k) {
        const unsigned int assignment = token * top_k + k;
        const float route_weight = route_weights[assignment];
        const GgufKqMeta meta = metas[route_slots[assignment]];
        const unsigned int down_bb = gguf_kq_block_bytes(meta.down_type);
        const unsigned char* down_rows[8];
        #pragma unroll
        for (unsigned int d = 0; d < 8; ++d)
            down_rows[d] = gguf_kq_row(meta.down, meta.down_type, out_base + d, intermediate);
        for (unsigned int j = lane; j < intermediate; j += 256) {
            if ((j & 1u) != 0u) continue;
            if (j + 1u < intermediate) {
                const zllm_ct_bf16x2 x = {
                    activated[(unsigned long long)assignment * intermediate + j],
                    activated[(unsigned long long)assignment * intermediate + j + 1u]
                };
                for (unsigned int d = 0; d < 8; ++d) {
                    const unsigned char* row = down_rows[d];
                    const zllm_ct_bf16x2 w = {
                        zllm_f32_to_bf16(gguf_kq_value_at(row, meta.down_type, j, down_bb)),
                        zllm_f32_to_bf16(gguf_kq_value_at(row, meta.down_type, j + 1u, down_bb))
                    };
                    sums[d] += __builtin_amdgcn_fdot2_f32_bf16(x, w, 0.0f, false) * route_weight;
                }
                continue;
            }
            const float a = zllm_bf16_to_f32(activated[(unsigned long long)assignment * intermediate + j]) * route_weight;
            #pragma unroll
            for (unsigned int d = 0; d < 8; ++d)
                sums[d] += a * gguf_kq_value_at(down_rows[d], meta.down_type, j, down_bb);
        }
    }
    __shared__ float partial[8][256];
    #pragma unroll
    for (unsigned int d = 0; d < 8; ++d) partial[d][lane] = sums[d];
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            #pragma unroll
            for (unsigned int d = 0; d < 8; ++d) partial[d][lane] += partial[d][lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int d = 0; d < 8; ++d)
            output[(unsigned long long)token * hidden + out_base + d] = partial[d][0];
    }
}
extern "C" __global__ void gguf_group_decode_routes(
    const unsigned int* route_ids,
    const float* route_weights,
    const GgufKqMeta* expert_metas,
    unsigned int* grouped_tokens,
    float* grouped_weights,
    unsigned int* route_to_grouped,
    unsigned int* grouped_offsets,
    GgufKqMeta* grouped_metas,
    unsigned int* grouped_experts,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int expert_count,
    unsigned int slot_count)
{
    __shared__ unsigned int counts[256];
    __shared__ unsigned int expert_slots[256];
    __shared__ unsigned int cursors[256];
    const unsigned int lane = threadIdx.x;
    if (lane < 256u) {
        counts[lane] = 0u;
        expert_slots[lane] = 0xffffffffu;
        cursors[lane] = 0u;
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
    for (unsigned int expert = lane; expert < expert_count; expert += blockDim.x) {
        const unsigned int slot = expert_slots[expert];
        if (slot == 0xffffffffu) continue;
        unsigned int grouped = grouped_offsets[slot];
        for (unsigned int route = 0u; route < route_count; ++route) {
            if (route_ids[route] != expert) continue;
            grouped_tokens[grouped] = route / top_k;
            grouped_weights[grouped] = route_weights[route];
            route_to_grouped[route] = grouped;
            grouped_experts[grouped] = slot;
            ++grouped;
        }
    }
}

// gate/up 融合 WMMA：结构对齐 ct_grouped_gate_up_wmma_bf16，仅权重解码换成 GGUF LUT。
extern "C" __global__ void gguf_gate_up_wmma_bf16(
    const unsigned short* input,
    const unsigned int* token_ids,
    const GgufKqMeta* experts,
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
    const unsigned int input_base = blockIdx.y * 128;
    const unsigned int output_base = (blockIdx.x * 4 + tile_wave) * 16;
    if (blockIdx.z >= expert_count || wave >= 8u) return;
    const unsigned int route_begin = route_offsets[expert];
    const unsigned int input_rows = route_offsets[expert + 1] - route_begin;
    // 与 CT 不同：不设 input_rows>=4 门槛——不足 16 行由 staging 零填充，
    // 单 token(MTP draft)与稀疏 expert 也必须产出 activated，否则 down 读到未初始化内存。
    if (input_base >= input_rows) return;
    const GgufKqMeta& meta = experts[expert];
    const unsigned long long weight = projection == 0u ? meta.gate : meta.up;
    const unsigned int tensor_type = projection == 0u ? meta.gate_type : meta.up_type;
    __shared__ unsigned short input_tiles[8][16 * 16];
    __shared__ unsigned short weight_tiles[8][16 * 16];
    __shared__ GgufWmmaScratch scratch;
    CFragment accumulator_fragments[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (input_base + tile * 16 < input_rows)
            fill_fragment(accumulator_fragments[tile], 0.0f);
    }
    const unsigned int stage_row = (threadIdx.x >> 2) & 15u;
    const unsigned int stage_chunk = threadIdx.x & 3u;
    const unsigned int stage_group = threadIdx.x >> 6;
    uint2 stage_data[2];
    unsigned int stage_valid[2] = {0u, 0u};
    auto stage_prefetch = [&](unsigned int prefetch_base) {
        #pragma unroll
        for (unsigned int half = 0; half < 2; ++half) {
            const unsigned int tile = stage_group + half * 4;
            const unsigned int route_row = input_base + tile * 16 + stage_row;
            stage_valid[half] = route_row < input_rows ? 1u : 0u;
            const unsigned int token = stage_valid[half]
                ? token_ids[route_begin + route_row] : 0u;
            stage_data[half] = stage_valid[half]
                ? *reinterpret_cast<const uint2*>(
                    input + (unsigned long long)token * input_columns
                    + prefetch_base + stage_chunk * 4)
                : make_uint2(0u, 0u);
        }
    };
    stage_prefetch(0);
    for (unsigned int column_base = 0; column_base < input_columns; column_base += 16) {
        if (threadIdx.x < 16 * 16) {
            #pragma unroll
            for (unsigned int half = 0; half < 2; ++half) {
                const unsigned int tile = stage_group + half * 4;
                if (stage_valid[half]) {
                    *reinterpret_cast<uint2*>(
                        &input_tiles[tile][stage_row * 16 + stage_chunk * 4])
                        = stage_data[half];
                }
            }
        }
        if (column_base + 16 < input_columns) stage_prefetch(column_base + 16);
        if (lane < 16) {
            const unsigned int output_row = output_base + lane;
            if (output_row < output_rows) {
                // 行基址只算一次；16 列共享同一 K-quant 块序列，块内偏移增量推进。
                const unsigned char* row_base = gguf_kq_row(weight, tensor_type, output_row, input_columns);
                const unsigned int block_stride = gguf_kq_block_bytes(tensor_type);
                // LUT 重建降频：K-quant 的 scale 每 32 列才变化，与 CT(组边界重建)对齐。
                if ((column_base & 31u) == 0u) {
                    gguf_kq_fill_lut(scratch.lut[wave] + lane * 32, weight, tensor_type, output_row, input_columns, column_base);
                }
                const unsigned short* lut = scratch.lut[wave] + lane * 32;
                if (tensor_type == 14u) {
                    #pragma unroll
                    for (unsigned int column = 0; column < 16; ++column) {
                        const unsigned char* block = row_base + (unsigned long long)((column_base + column) >> 8) * block_stride;
                        const unsigned int local = (column_base + column) & 255;
                        const unsigned int half = local >> 7;
                        const unsigned int slot = (local & 127) >> 5;
                        const unsigned int index = local & 31;
                        const unsigned char low = block[half * 64 + index + (slot & 1u) * 32];
                        const unsigned int nibble = (slot < 2u) ? (low & 0x0fu) : (low >> 4);
                        const unsigned int high = (block[128 + half * 32 + index] >> (slot * 2)) & 3u;
                        weight_tiles[wave][lane * 16 + column] = lut[nibble | (high << 4)];
                    }
                } else if (tensor_type == 12u) {
                    #pragma unroll
                    for (unsigned int column = 0; column < 16; ++column) {
                        const unsigned char* block = row_base + (unsigned long long)((column_base + column) >> 8) * block_stride;
                        const unsigned int local = (column_base + column) & 255;
                        const unsigned int group = local >> 5;
                        const unsigned char packed = block[16 + (group >> 1) * 32 + (local & 31)];
                        const unsigned int nibble = (group & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
                        weight_tiles[wave][lane * 16 + column] = lut[nibble];
                    }
                } else {
                    #pragma unroll
                    for (unsigned int column = 0; column < 16; ++column) {
                        const unsigned char* block = row_base + (unsigned long long)((column_base + column) >> 8) * block_stride;
                        const unsigned int local = (column_base + column) & 255;
                        const unsigned int group = local >> 5;
                        const unsigned char packed = block[48 + (group >> 1) * 32 + (local & 31)];
                        const unsigned int nibble = (group & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
                        const unsigned int high = (block[16 + (local & 31)] >> group) & 1u;
                        weight_tiles[wave][lane * 16 + column] = lut[nibble | (high << 4)];
                    }
                }
            } else {
                #pragma unroll
                for (unsigned int column = 0; column < 16; ++column)
                    weight_tiles[wave][lane * 16 + column] = 0;
            }
        }
        __syncthreads();
        BFragment weight_fragment;
        load_matrix_sync(weight_fragment, (const bfloat16_t*)weight_tiles[wave], 16);
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

// 确定性 down：每个 (token, 8 hidden 维) 块跨 top_k 按固定顺序在寄存器累加，无原子。
extern "C" __global__ void gguf_grouped_down_f32(
    const unsigned short* activated,
    const GgufKqMeta* grouped_metas,
    const unsigned int* grouped_experts,
    const float* grouped_weights,
    const unsigned int* route_to_grouped,
    float* output,
    unsigned int tokens,
    unsigned int top_k,
    unsigned int hidden,
    unsigned int intermediate)
{
    const unsigned int out_base = blockIdx.x * 8;
    const unsigned int token = blockIdx.y;
    if (token >= tokens) return;
    const unsigned int lane = threadIdx.x;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int k = 0; k < top_k; ++k) {
        const unsigned int grouped = route_to_grouped[token * top_k + k];
        const float route_weight = grouped_weights[grouped];
        const GgufKqMeta meta = grouped_metas[grouped_experts[grouped]];
        for (unsigned int j = lane; j < intermediate; j += 256) {
            const float a = zllm_bf16_to_f32(activated[(unsigned long long)grouped * intermediate + j]) * route_weight;
            #pragma unroll
            for (unsigned int d = 0; d < 8; ++d) {
                sums[d] += a * gguf_kq_value(meta.down, meta.down_type, out_base + d, intermediate, j);
            }
        }
    }
    __shared__ float partial[8][256];
    #pragma unroll
    for (unsigned int d = 0; d < 8; ++d) partial[d][lane] = sums[d];
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            #pragma unroll
            for (unsigned int d = 0; d < 8; ++d) partial[d][lane] += partial[d][lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int d = 0; d < 8; ++d)
            output[(unsigned long long)token * hidden + out_base + d] = partial[d][0];
    }
}
