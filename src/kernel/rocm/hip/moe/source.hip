#include <hip/hip_bfloat16.h>
#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void moe_router_logits_decode_f32(
    const float* input,
    const unsigned short* weight,
    float* logits,
    unsigned int columns,
    unsigned int experts)
{
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    // 固定四个 expert/block，使 wave32 与 wave64 使用同一 host grid。
    if (wave >= 4U) return;
    const unsigned int expert = blockIdx.x * 4U + wave;
    if (expert >= experts) return;
    const unsigned short* expert_weight = weight + (unsigned long long)expert * columns;
    float sum = 0.0F;
    for (unsigned int column = lane * 2U; column + 1U < columns; column += warpSize * 2U) {
        float input0 = input[column];
        float input1 = input[column + 1U];
        if (!isfinite(input0)) input0 = 0.0F;
        if (!isfinite(input1)) input1 = 0.0F;
        const float input_bf16_0 = zllm_bf16_to_f32(zllm_f32_to_bf16(input0));
        const float input_bf16_1 = zllm_bf16_to_f32(zllm_f32_to_bf16(input1));
        const float weight_bf16_0 = zllm_bf16_to_f32(expert_weight[column]);
        const float weight_bf16_1 = zllm_bf16_to_f32(expert_weight[column + 1U]);
        sum = fmaf(input_bf16_0, weight_bf16_0, sum);
        sum = fmaf(input_bf16_1, weight_bf16_1, sum);
    }
    if ((columns & 1U) != 0U && lane == 0U) {
        float input_value = input[columns - 1U];
        if (!isfinite(input_value)) input_value = 0.0F;
        input_value = zllm_bf16_to_f32(zllm_f32_to_bf16(input_value));
        const float weight_value = zllm_bf16_to_f32(expert_weight[columns - 1U]);
        sum = fmaf(input_value, weight_value, sum);
    }
    for (unsigned int offset = warpSize >> 1; offset > 0U; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0U) logits[expert] = isfinite(sum) ? sum : 0.0F;
}

extern "C" __global__ void moe_router_logits_precise_f32(
    const float* input,
    const float* weight,
    float* logits,
    unsigned int rows,
    unsigned int columns,
    unsigned int experts)
{
    const unsigned int row = blockIdx.y;
    const unsigned int expert = blockIdx.x;
    if (row >= rows || expert >= experts) return;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int wave_count = blockDim.x / warpSize;
    const float* row_input = input + (unsigned long long)row * columns;
    const float* expert_weight = weight + (unsigned long long)expert * columns;
    float sum = 0.0F;
    for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
        float input_value = row_input[column];
        float weight_value = expert_weight[column];
        if (!isfinite(input_value)) input_value = 0.0F;
        if (!isfinite(weight_value)) weight_value = 0.0F;
        sum = fmaf(input_value, weight_value, sum);
    }
    for (unsigned int offset = warpSize >> 1; offset > 0U; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    __shared__ float wave_sums[8];
    if (lane == 0U) wave_sums[wave] = sum;
    __syncthreads();
    if (wave == 0U) {
        sum = lane < wave_count ? wave_sums[lane] : 0.0F;
        for (unsigned int offset = warpSize >> 1; offset > 0U; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
        if (lane == 0U) {
            logits[(unsigned long long)row * experts + expert] =
                isfinite(sum) ? sum : 0.0F;
        }
    }
}

// 四行共享一次 router 权重读取；每行仍按原 256-thread 步长累加，再按
// 相同 wave/CTA 顺序归约，避免路由边界附近因精度变化选中不同 expert。
extern "C" __global__ void moe_router_logits_precise_rows4_f32(
    const float* input,
    const float* weight,
    float* logits,
    unsigned int rows,
    unsigned int columns,
    unsigned int experts)
{
    const unsigned int row_base = blockIdx.y * 4U;
    const unsigned int expert = blockIdx.x;
    if (row_base >= rows || expert >= experts) return;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int wave_count = blockDim.x / warpSize;
    const float* expert_weight = weight + (unsigned long long)expert * columns;
    float sums[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
        float w = expert_weight[column];
        if (!isfinite(w)) w = 0.0F;
        #pragma unroll
        for (unsigned int r = 0; r < 4U; ++r) {
            float x = row_base + r < rows
                ? input[(unsigned long long)(row_base + r) * columns + column] : 0.0F;
            if (!isfinite(x)) x = 0.0F;
            sums[r] = fmaf(x, w, sums[r]);
        }
    }
    __shared__ float wave_sums[4][8];
    #pragma unroll
    for (unsigned int r = 0; r < 4U; ++r) {
        for (unsigned int offset = warpSize >> 1; offset > 0U; offset >>= 1)
            sums[r] += __shfl_down(sums[r], offset);
        if (lane == 0U) wave_sums[r][wave] = sums[r];
    }
    __syncthreads();
    if (wave == 0U) {
        #pragma unroll
        for (unsigned int r = 0; r < 4U; ++r) {
            float sum = lane < wave_count ? wave_sums[r][lane] : 0.0F;
            for (unsigned int offset = warpSize >> 1; offset > 0U; offset >>= 1)
                sum += __shfl_down(sum, offset);
            if (lane == 0U && row_base + r < rows)
                logits[(unsigned long long)(row_base + r) * experts + expert]
                    = isfinite(sum) ? sum : 0.0F;
        }
    }
}

extern "C" __global__ void moe_router_logits_f32(
    const float* input,
    const unsigned short* weight,
    float* logits,
    unsigned int rows,
    unsigned int columns,
    unsigned int experts)
{
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row_base = blockIdx.y * 16U;
    const unsigned int expert_base = blockIdx.x * 128U;
    __shared__ unsigned short input_tile[16 * 16];
    __shared__ unsigned short weight_tiles[8][16 * 16];
    __shared__ float output_tiles[8][16 * 16];
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major> left;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::col_major> right;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> sum;
    rocwmma::fill_fragment(sum, 0.0F);
    for (unsigned int column_base = 0; column_base < columns; column_base += 16U) {
        const unsigned int input_index = threadIdx.x;
        const unsigned int input_row = input_index >> 4;
        const unsigned int input_column = input_index & 15U;
        float input_value = 0.0F;
        if (row_base + input_row < rows && column_base + input_column < columns) {
            input_value = input[(unsigned long long)(row_base + input_row) * columns + column_base + input_column];
            if (!isfinite(input_value)) input_value = 0.0F;
        }
        input_tile[input_index] = zllm_f32_to_bf16(input_value);
        for (unsigned int index = threadIdx.x; index < 8U * 16U * 16U; index += blockDim.x) {
            const unsigned int tile = index >> 8;
            const unsigned int element = index & 255U;
            const unsigned int expert_delta = element >> 4;
            const unsigned int weight_column = element & 15U;
            const unsigned int expert = expert_base + tile * 16U + expert_delta;
            if (expert < experts && column_base + weight_column < columns) {
                weight_tiles[tile][element] = weight[(unsigned long long)expert * columns + column_base + weight_column];
            } else {
                weight_tiles[tile][element] = 0U;
            }
        }
        __syncthreads();
        rocwmma::load_matrix_sync(left, (const rocwmma::bfloat16_t*)input_tile, 16);
        rocwmma::load_matrix_sync(right, (const rocwmma::bfloat16_t*)weight_tiles[wave], 16);
        rocwmma::mma_sync(sum, left, right, sum);
        __syncthreads();
    }
    rocwmma::store_matrix_sync(output_tiles[wave], sum, 16, rocwmma::mem_row_major);
    for (unsigned int index = lane; index < 16U * 16U; index += 32U) {
        const unsigned int output_row = index >> 4;
        const unsigned int expert_delta = index & 15U;
        const unsigned int row = row_base + output_row;
        const unsigned int expert = expert_base + wave * 16U + expert_delta;
        if (row < rows && expert < experts) {
            const float value = output_tiles[wave][index];
            logits[(unsigned long long)row * experts + expert] = isfinite(value) ? value : 0.0F;
        }
    }
}

extern "C" __global__ void moe_router_topk_256_f32(
    const float* logits,
    const float* bias,
    unsigned int* expert_ids,
    float* route_weights,
    unsigned int rows,
    unsigned int experts,
    unsigned int top_k,
    unsigned int scoring,
    float scaling)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || blockDim.x != 256 || top_k == 0 || top_k > 16) return;
    // 每个 lane 串行扫描 lane, lane+256, ... 覆盖任意 experts 数(GLM5.3 为
    // 288);已选专家在重扫时跳过,平局取小 expert id,与逐专家插入排序语义一致。
    __shared__ float raw_values[256];
    __shared__ float wave_scores[8];
    __shared__ unsigned int wave_ids[8];
    __shared__ float selected_raw[16];
    __shared__ unsigned int selected_id[16];
    __shared__ float denominator;
    for (unsigned int slot = 0; slot < top_k; ++slot) {
        float score = -3.402823466e+38F;
        float raw = 0.0F;
        unsigned int expert_id = lane;
        for (unsigned int expert = lane; expert < experts; expert += 256U) {
            bool taken = false;
            for (unsigned int seen = 0; seen < slot; ++seen) taken = taken || selected_id[seen] == expert;
            if (taken) continue;
            const float logit = logits[(unsigned long long)row * experts + expert];
            float current_raw;
            float current_score;
            if (scoring == 0) {
                current_raw = logit;
                current_score = logit;
            } else if (scoring == 1) {
                current_raw = 1.0F / (1.0F + expf(-logit));
                current_score = current_raw + bias[expert];
            } else {
                const float softplus = logit > 0.0F ? logit + log1pf(expf(-logit)) : log1pf(expf(logit));
                current_raw = sqrtf(softplus);
                current_score = current_raw + bias[expert];
            }
            if (current_score > score || (current_score == score && expert < expert_id)) {
                score = current_score;
                raw = current_raw;
                expert_id = expert;
            }
        }
        raw_values[lane] = raw;
        const unsigned int wave_lane = lane % warpSize;
        const unsigned int wave = lane / warpSize;
        const unsigned int wave_count = blockDim.x / warpSize;
        float best_score = score;
        unsigned int best_id = expert_id;
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            const float other_score = __shfl_down(best_score, offset);
            const unsigned int other_id = __shfl_down(best_id, offset);
            if (other_score > best_score || (other_score == best_score && other_id < best_id)) {
                best_score = other_score;
                best_id = other_id;
            }
        }
        if (wave_lane == 0) {
            wave_scores[wave] = best_score;
            wave_ids[wave] = best_id;
        }
        __syncthreads();
        if (lane == 0) {
            float block_score = wave_scores[0];
            unsigned int block_id = wave_ids[0];
            for (unsigned int item = 1; item < wave_count; ++item) {
                const float item_score = wave_scores[item];
                const unsigned int item_id = wave_ids[item];
                if (item_score > block_score || (item_score == block_score && item_id < block_id)) {
                    block_score = item_score;
                    block_id = item_id;
                }
            }
            selected_id[slot] = block_id;
            selected_raw[slot] = raw_values[block_id % 256U];
        }
        __syncthreads();
    }
    if (lane == 0) {
        float sum = 0.0F;
        if (scoring == 0) {
            for (unsigned int slot = 0; slot < top_k; ++slot) sum += expf(selected_raw[slot] - selected_raw[0]);
        } else {
            for (unsigned int slot = 0; slot < top_k; ++slot) sum += selected_raw[slot];
        }
        denominator = sum;
    }
    __syncthreads();
    if (lane < top_k) {
        const unsigned long long route = (unsigned long long)row * top_k + lane;
        expert_ids[route] = selected_id[lane];
        const float mass = scoring == 0 ? expf(selected_raw[lane] - selected_raw[0]) : selected_raw[lane];
        route_weights[route] = mass / denominator * scaling;
    }
}

extern "C" __global__ void moe_zero_f32(float* output, unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = 0.0F;
}

extern "C" __global__ void moe_gather_f32(
    const float* input,
    const unsigned int* row_ids,
    float* output,
    unsigned int gathered_rows,
    unsigned int columns)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = gathered_rows * columns;
    if (index >= elements) return;
    const unsigned int source_row = row_ids[index / columns];
    output[index] = input[(unsigned long long)source_row * columns + index % columns];
}

extern "C" __global__ void moe_gather_bf16(
    const unsigned short* input,
    const unsigned int* row_ids,
    unsigned short* output,
    unsigned int gathered_rows,
    unsigned int columns)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = gathered_rows * columns;
    if (index >= elements) return;
    const unsigned int source_row = row_ids[index / columns];
    output[index] = input[(unsigned long long)source_row * columns + index % columns];
}

extern "C" __global__ void moe_scatter_add_f32(
    float* output,
    const float* input,
    const unsigned int* row_ids,
    const float* weights,
    unsigned int source_rows,
    unsigned int columns)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = source_rows * columns;
    if (index >= elements) return;
    const unsigned int source_row = index / columns;
    const unsigned int target_row = row_ids[source_row];
    output[(unsigned long long)target_row * columns + index % columns] += input[index] * weights[source_row];
}

// MXFP4(E2M1 packed + E8M0 32-group scale) in-kernel 反量化 matmul。
// 每 WG 处理 weight 一行 × 全部 input 行;scale 组内 32 元素,packed 每
// 字节两列(低 nibble 偶数列),位序与 weight::format::mxfp4 CPU 解码一致。
__device__ __forceinline__ float zllm_mxfp4_e2m1(unsigned int nibble)
{
    // 8 个非负 E2M1 值的 BF16 高 16 位恰好能装进两个 64-bit immediate；
    // 避免编译器把 float LUT 降成每个 nibble 一次 global constant load。
    const unsigned int code = nibble & 7u;
    const unsigned long long packed = (code & 4u) != 0u ? 0x40c0408040404000ULL : 0x3fc03f803f000000ULL;
    const unsigned int bits = (unsigned int)(packed >> ((code & 3u) * 16u));
    const float magnitude = __uint_as_float((bits & 0xffffu) << 16u);
    return (nibble & 8u) != 0u ? -magnitude : magnitude;
}

extern "C" __global__ void mxfp4_matmul_f32(
    const float* __restrict__ input,          // [N, K]
    const unsigned char* __restrict__ packed, // [M, K/2]
    const unsigned char* __restrict__ scales, // [M, K/32] e8m0
    float* __restrict__ output,               // [N, M]
    const unsigned int input_rows,            // N
    const unsigned int columns,               // K
    const unsigned int weight_rows)           // M
{
    const unsigned int row = blockIdx.x;
    if (row >= weight_rows) return;
    const unsigned int lane = threadIdx.x;
    // RDNA3 默认 wave32;shfl width 超过 warpSize 会回绕(lane0 加到自身),上限 32。
    const unsigned int wavefront_size = (blockDim.x < 32u) ? blockDim.x : 32u;
    const unsigned long long packed_row = (unsigned long long)row * (columns / 2u);
    const unsigned long long scale_row = (unsigned long long)row * (columns / 32u);
    for (unsigned int r = 0u; r < input_rows; ++r) {
        const float* input_row = input + (unsigned long long)r * columns;
        float partial = 0.0F;

        for (unsigned int column = lane * 16u; column < columns; column += wavefront_size * 16u) {
            #pragma unroll
            for (unsigned int offset = 0; offset < 16u; ++offset) {
                const unsigned int col = column + offset;
                if (col < columns) {
                    const unsigned char byte = packed[packed_row + col / 2u];
                    const unsigned int nibble = (col & 1u) != 0u ? (unsigned int)(byte >> 4) : (unsigned int)(byte & 0x0fu);
                    const unsigned int bits = ((unsigned int)scales[scale_row + col / 32u]) << 23;
                    const float scale = __uint_as_float(bits);
                    partial += zllm_mxfp4_e2m1(nibble) * scale * input_row[col];
                }
            }
        }
        for (unsigned int offset = wavefront_size >> 1; offset > 0u; offset >>= 1u) {
            partial += __shfl_down(partial, offset, wavefront_size);
        }
        if (lane == 0u) {
            output[(unsigned long long)r * weight_rows + row] = partial;
        }
    }
}

// 固定专家集合路由(V4 哈希层):每 WG 一个 token,顺序算 top_k 个预选
// expert 的 sqrt(softplus(dot)),行内归一化后乘 scaling。wave32 安全归约。
extern "C" __global__ void moe_router_selected_f32(
    const float* __restrict__ input,        // [N, K]
    const float* __restrict__ weight,       // [E, K]
    const unsigned int* __restrict__ selected, // [N, top_k]
    float* __restrict__ output,             // [N, top_k]
    const unsigned int rows,                // N
    const unsigned int columns,             // K
    const unsigned int expert_count,        // E
    const unsigned int top_k,
    const float scaling)
{
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int waves = min(blockDim.x >> 5, top_k);
    const float* input_row = input + (unsigned long long)row * columns;
    // 每 wave 独立算一个预选 expert 的 dot,取代单 wave 串行 top_k 次;
    // 单 expert 的 K 遍历与 wave32 归约树保持逐位一致。
    const unsigned int slot = wave;
    float value = 0.0F;
    if (slot < top_k) {
        const unsigned int expert = selected[(unsigned long long)row * top_k + slot];
        float partial = 0.0F;
        if (expert < expert_count) {
            const float* weight_row = weight + (unsigned long long)expert * columns;
            for (unsigned int column = lane; column < columns; column += 32u) {
                partial += input_row[column] * weight_row[column];
            }
        }
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            partial += __shfl_down(partial, offset, 32u);
        }
        if (lane == 0u && expert < expert_count) {
            const float softplus = partial > 0.0F ? partial + log1pf(expf(-partial)) : log1pf(expf(partial));
            value = sqrtf(softplus);
        }
    }
    __shared__ float slot_values[16];
    if (lane == 0u && slot < top_k) slot_values[slot] = value;
    __syncthreads();
    if (wave == 0u && lane == 0u) {
        float sum = 0.0F;
        for (unsigned int item = 0u; item < waves; ++item) sum += slot_values[item];
        for (unsigned int item = 0u; item < waves; ++item) {
            output[(unsigned long long)row * top_k + item] = slot_values[item] / sum * scaling;
        }
    }
}

// ===== MXFP4 grouped routed experts(整层一次装填,4bit 驻留 in-kernel 反量化)=====
__device__ __forceinline__ const unsigned char* zllm_mxfp4_native_packed(
    const unsigned char* packed,
    unsigned long long row,
    unsigned int column,
    unsigned int columns)
{
    const unsigned long long tile = (row >> 4) * (columns >> 4) + (column >> 4);
    const unsigned int lane = (unsigned int)(row & 15u) + ((column & 15u) >> 3) * 16u;
    return packed + tile * 128u + lane * 4u + ((column & 7u) >> 1);
}

__device__ __forceinline__ float zllm_mxfp4_native_scale(
    const unsigned char* scales,
    unsigned long long row,
    unsigned int column,
    unsigned int columns)
{
    const unsigned long long index = ((row >> 4) * (columns >> 5) + (column >> 5)) * 16u + (row & 15u);
    return __uint_as_float((unsigned int)scales[index] << 23);
}



// 小批量复用 grouped 权重布局，但直接消费 token-major top-k route。
// grid.y 隔离 token，每个 block 只服务该 token 的真实 route，避免仍沿 expert 维启动 256 组。
extern "C" __global__ void mxfp4_decode_gate_up_f32(
    const float* __restrict__ input,
    const unsigned int* __restrict__ route_ids,
    float* __restrict__ activated,
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const unsigned int hidden,
    const unsigned int intermediate,
    const unsigned int expert_count,
    const unsigned int top_k,
    const float limit)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int token = blockIdx.y;
    const unsigned int route = wave;
    if (route >= top_k) return;
    const unsigned int route_index = token * top_k + route;
    const unsigned int expert = route_ids[route_index];
    if (expert >= expert_count) return;
    const unsigned int column = blockIdx.x;
    const bool valid = column < intermediate;
    const unsigned int gate_row = expert * intermediate + column;
    const unsigned int up_row = (expert_count + expert) * intermediate + column;
    float gate_partial = 0.0F;
    float up_partial = 0.0F;
    const float* input_row = input + (unsigned long long)token * hidden;
    // 主循环每 lane 一次 8-byte packed(16 列,nibble 低位=偶数列,与 CPU 解码
    // 位序一致),wave 每迭代覆盖 512 连续列,访存从 32B/迭代提到 256B;
    // 不足 16 列的尾段回落逐对路径,保持 hidden 为 32 倍数的通用性。
    unsigned int k = lane * 16u;
    for (; valid && k + 16u <= hidden; k += 512u) {
        const unsigned long long gate_packed8 =
            (unsigned long long)*reinterpret_cast<const unsigned int*>(zllm_mxfp4_native_packed(packed, gate_row, k, hidden)) |
            ((unsigned long long)*reinterpret_cast<const unsigned int*>(zllm_mxfp4_native_packed(packed, gate_row, k + 8u, hidden)) << 32);
        const unsigned long long up_packed8 =
            (unsigned long long)*reinterpret_cast<const unsigned int*>(zllm_mxfp4_native_packed(packed, up_row, k, hidden)) |
            ((unsigned long long)*reinterpret_cast<const unsigned int*>(zllm_mxfp4_native_packed(packed, up_row, k + 8u, hidden)) << 32);
        const float gate_scale = zllm_mxfp4_native_scale(scales, gate_row, k, hidden);
        const float up_scale = zllm_mxfp4_native_scale(scales, up_row, k, hidden);
        float xs[16];
        *reinterpret_cast<float4*>(xs) = *reinterpret_cast<const float4*>(input_row + k);
        *reinterpret_cast<float4*>(xs + 4) = *reinterpret_cast<const float4*>(input_row + k + 4u);
        *reinterpret_cast<float4*>(xs + 8) = *reinterpret_cast<const float4*>(input_row + k + 8u);
        *reinterpret_cast<float4*>(xs + 12) = *reinterpret_cast<const float4*>(input_row + k + 12u);
        #pragma unroll
        for (unsigned int column = 0u; column < 16u; ++column) {
            gate_partial += zllm_mxfp4_e2m1((unsigned int)((gate_packed8 >> (column * 4u)) & 0xfu)) * gate_scale * xs[column];
            up_partial += zllm_mxfp4_e2m1((unsigned int)((up_packed8 >> (column * 4u)) & 0xfu)) * up_scale * xs[column];
        }
    }
    for (; valid && k < hidden; k += 64u) {
        const unsigned char gate_byte = *zllm_mxfp4_native_packed(packed, gate_row, k, hidden);
        const unsigned char up_byte = *zllm_mxfp4_native_packed(packed, up_row, k, hidden);
        const float gate_scale = zllm_mxfp4_native_scale(scales, gate_row, k, hidden);
        const float up_scale = zllm_mxfp4_native_scale(scales, up_row, k, hidden);
        const float2 input_pair = *reinterpret_cast<const float2*>(input_row + k);
        gate_partial += zllm_mxfp4_e2m1(gate_byte & 0x0fu) * gate_scale * input_pair.x;
        gate_partial += zllm_mxfp4_e2m1((unsigned int)(gate_byte >> 4)) * gate_scale * input_pair.y;
        up_partial += zllm_mxfp4_e2m1(up_byte & 0x0fu) * up_scale * input_pair.x;
        up_partial += zllm_mxfp4_e2m1((unsigned int)(up_byte >> 4)) * up_scale * input_pair.y;
    }
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        gate_partial += __shfl_down(gate_partial, offset, 32u);
        up_partial += __shfl_down(up_partial, offset, 32u);
    }
    if (lane == 0u && valid) {
        const float gate = gate_partial > limit ? limit : gate_partial;
        const float up = up_partial > limit ? limit : (up_partial < -limit ? -limit : up_partial);
        activated[((unsigned long long)token * top_k + route) * intermediate + column] = gate * (1.0F / (1.0F + expf(-gate))) * up;
    }
}

// 每个 wave 固定一个 token 的 hidden 输出，在同一 block 内顺序归并该 token 的 top-k route；
// 不需要清零 output，也不需要 atomic scatter。
extern "C" __global__ void mxfp4_decode_down_reduce_f32(
    const float* __restrict__ activated,
    const unsigned int* __restrict__ route_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ output,
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ shared,
    const unsigned int hidden,
    const unsigned int intermediate,
    const unsigned int expert_count,
    const unsigned int top_k,
    const unsigned int has_shared)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int token = blockIdx.y;
    const unsigned int column = blockIdx.x * 16u + wave;
    if (column >= hidden) return;
    float reduced = 0.0F;
    for (unsigned int route = 0u; route < top_k; ++route) {
        const unsigned int route_index = token * top_k + route;
        const unsigned int expert = route_ids[route_index];
        if (expert >= expert_count) continue;
        const unsigned int weight_row = expert * hidden + column;
        const float* input = activated + (unsigned long long)route_index * intermediate;
        float partial = 0.0F;
        // 与 gate/up 相同的 16 列/lane 加宽;down 权重行按 intermediate 切分。
        unsigned int k = lane * 16u;
        for (; k + 16u <= intermediate; k += 512u) {
            const unsigned long long packed8 =
                (unsigned long long)*reinterpret_cast<const unsigned int*>(zllm_mxfp4_native_packed(packed, weight_row, k, intermediate)) |
                ((unsigned long long)*reinterpret_cast<const unsigned int*>(zllm_mxfp4_native_packed(packed, weight_row, k + 8u, intermediate)) << 32);
            const float scale = zllm_mxfp4_native_scale(scales, weight_row, k, intermediate);
            float xs[16];
            *reinterpret_cast<float4*>(xs) = *reinterpret_cast<const float4*>(input + k);
            *reinterpret_cast<float4*>(xs + 4) = *reinterpret_cast<const float4*>(input + k + 4u);
            *reinterpret_cast<float4*>(xs + 8) = *reinterpret_cast<const float4*>(input + k + 8u);
            *reinterpret_cast<float4*>(xs + 12) = *reinterpret_cast<const float4*>(input + k + 12u);
            #pragma unroll
            for (unsigned int column = 0u; column < 16u; ++column) {
                partial += zllm_mxfp4_e2m1((unsigned int)((packed8 >> (column * 4u)) & 0xfu)) * scale * xs[column];
            }
        }
        for (; k < intermediate; k += 64u) {
            const unsigned char byte = *zllm_mxfp4_native_packed(packed, weight_row, k, intermediate);
            const float scale = zllm_mxfp4_native_scale(scales, weight_row, k, intermediate);
            const float2 input_pair = *reinterpret_cast<const float2*>(input + k);
            partial += zllm_mxfp4_e2m1(byte & 0x0fu) * scale * input_pair.x;
            partial += zllm_mxfp4_e2m1((unsigned int)(byte >> 4)) * scale * input_pair.y;
        }
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) partial += __shfl_down(partial, offset, 32u);
        if (lane == 0u) reduced = fmaf(partial, route_weights[route_index], reduced);
    }
    if (lane == 0u) {
        const unsigned long long output_index = (unsigned long long)token * hidden + column;
        output[output_index] = has_shared != 0u ? shared[output_index] + reduced : reduced;
    }
}

// GLM grouped WMMA 同款 tile：每个 block 用 8 个 wave 同时计算 gate/up
// 的 4x16 输出列与最多 128 条 route。MXFP4 已按 B fragment lane 顺序
// 预排，直接在寄存器解码为 BF16，矩阵乘由 gfx11 WMMA 完成。
extern "C" __global__ void moe_group_routes_small_f32(
    const unsigned int* __restrict__ route_ids,
    const float* __restrict__ route_weights,
    unsigned int* __restrict__ token_ids,
    float* __restrict__ grouped_weights,
    unsigned int* __restrict__ route_offsets,
    unsigned int* __restrict__ active_experts,
    unsigned int* __restrict__ active_count,
    unsigned int* __restrict__ counts,
    unsigned int* __restrict__ cursors,
    unsigned int* __restrict__ route_to_grouped,
    const unsigned int rows,
    const unsigned int top_k,
    const unsigned int expert_count)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    for (unsigned int expert = 0; expert < expert_count; ++expert) {
        counts[expert] = 0u;
        cursors[expert] = 0u;
    }
    const unsigned int route_count = rows * top_k;
    for (unsigned int route = 0; route < route_count; ++route) {
        const unsigned int expert = route_ids[route];
        if (expert < expert_count) ++counts[expert];
    }
    unsigned int groups = 0u;
    unsigned int total = 0u;
    for (unsigned int expert = 0; expert < expert_count; ++expert) {
        if (counts[expert] == 0u) continue;
        active_experts[groups] = expert;
        route_offsets[groups] = total;
        cursors[expert] = total;
        total += counts[expert];
        ++groups;
    }
    route_offsets[groups] = total;
    *active_count = groups;
    for (unsigned int route = 0; route < route_count; ++route) {
        const unsigned int expert = route_ids[route];
        if (expert >= expert_count) continue;
        const unsigned int grouped = cursors[expert]++;
        token_ids[grouped] = route / top_k;
        grouped_weights[grouped] = route_weights[route];
        route_to_grouped[route] = grouped;
    }
}

extern "C" __global__ void mxfp4_grouped_gate_up_wmma_f32(
    const float* __restrict__ input,
    const unsigned int* __restrict__ token_ids,
    const unsigned int* __restrict__ route_offsets,
    const unsigned int* __restrict__ active_experts,
    const unsigned int* __restrict__ active_count,
    float* __restrict__ activated,
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const unsigned int hidden,
    const unsigned int intermediate,
    const unsigned int expert_count,
    const float limit)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    const unsigned int group = blockIdx.z;
    const unsigned int groups = active_count != nullptr ? *active_count : expert_count;
    if (group >= groups) return;
    const unsigned int expert = active_experts != nullptr ? active_experts[group] : group;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int projection = wave >> 2;
    const unsigned int tile_wave = wave & 3u;
    const unsigned int input_base = blockIdx.y * 128u;
    const unsigned int output_base = (blockIdx.x * 4u + tile_wave) * 16u;
    if (expert >= expert_count || wave >= 8u) return;
    const unsigned int route_begin = route_offsets[group];
    const unsigned int input_rows = route_offsets[group + 1] - route_begin;
    if (input_base >= input_rows) return;
    const unsigned int route_tiles = min(8u, (input_rows - input_base + 15u) / 16u);
    __shared__ unsigned short input_tiles[8][16 * 16];
    __shared__ unsigned short input_residual_tiles[8][16 * 16];
    __shared__ float output_tiles[8][16 * 16];
    CFragment accumulators[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (tile < route_tiles) fill_fragment(accumulators[tile], 0.0f);
    }
    const unsigned int weight_expert = (projection != 0u ? expert_count : 0u) + expert;
    for (unsigned int column_base = 0; column_base < hidden; column_base += 16u) {
        const unsigned int local_row = threadIdx.x >> 4;
        const unsigned int local_column = threadIdx.x & 15u;
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            if (tile >= route_tiles) continue;
            const unsigned int route_delta = input_base + tile * 16u + local_row;
            float value = 0.0f;
            if (route_delta < input_rows && column_base + local_column < hidden) {
                const unsigned int token = token_ids[route_begin + route_delta];
                value = input[(unsigned long long)token * hidden + column_base + local_column];
                if (!isfinite(value)) value = 0.0f;
            }
            const unsigned short high = zllm_f32_to_bf16(value);
            input_tiles[tile][threadIdx.x] = high;
            input_residual_tiles[tile][threadIdx.x] = zllm_f32_to_bf16(value - zllm_bf16_to_f32(high));
        }
        __syncthreads();
        BFragment weight_fragment;
        unsigned short* weight_access = reinterpret_cast<unsigned short*>(&weight_fragment.mAccess);
        const unsigned int output_row = output_base + (lane & 15u);
        if (output_row < intermediate) {
            const unsigned long long weight_row = (unsigned long long)weight_expert * intermediate + output_row;
            const unsigned int codes = *reinterpret_cast<const unsigned int*>(
                zllm_mxfp4_native_packed(packed, weight_row, column_base + (lane >> 4) * 8u, hidden));
            const float scale = zllm_mxfp4_native_scale(scales, weight_row, column_base, hidden);
            #pragma unroll
            for (unsigned int element = 0; element < 8; ++element) {
                weight_access[element] = zllm_f32_to_bf16(zllm_mxfp4_e2m1((codes >> (element * 4u)) & 15u) * scale);
            }
        } else {
            #pragma unroll
            for (unsigned int element = 0; element < 8; ++element) weight_access[element] = 0u;
        }
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            if (tile >= route_tiles) continue;
            AFragment input_fragment;
            load_matrix_sync(input_fragment, (const bfloat16_t*)input_tiles[tile], 16);
            mma_sync(accumulators[tile], input_fragment, weight_fragment, accumulators[tile]);
            load_matrix_sync(input_fragment, (const bfloat16_t*)input_residual_tiles[tile], 16);
            mma_sync(accumulators[tile], input_fragment, weight_fragment, accumulators[tile]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (tile >= route_tiles) continue;
        store_matrix_sync(output_tiles[wave], accumulators[tile], 16, mem_row_major);
        __syncthreads();
        if (wave < 4u) {
            for (unsigned int index = lane; index < 16u * 16u; index += 32u) {
                const unsigned int route_delta = input_base + tile * 16u + index / 16u;
                const unsigned int output_row = output_base + index % 16u;
                if (route_delta < input_rows && output_row < intermediate) {
                    const unsigned long long output_index =
                        (unsigned long long)(route_begin + route_delta) * intermediate + output_row;
                    const float gate_raw = output_tiles[wave][index];
                    const float up_raw = output_tiles[wave + 4u][index];
                    const float gate = gate_raw > limit ? limit : gate_raw;
                    const float up = up_raw > limit ? limit : (up_raw < -limit ? -limit : up_raw);
                    activated[output_index] = gate * (1.0f / (1.0f + expf(-gate))) * up;
                }
            }
        }
        __syncthreads();
    }
}

// down 每个 block 用 8 个 wave 计算 128 个 hidden 输出列和最多 128 条
// route。host CSR 路径保留 atomic scatter；设备路由路径先写 route-major
// 中间结果，再由固定顺序 kernel 按 token 的原始 top-k 槽位归并。
extern "C" __global__ void mxfp4_grouped_down_scatter_wmma_f32(
    const float* __restrict__ activated,
    const unsigned int* __restrict__ token_ids,
    const unsigned int* __restrict__ route_offsets,
    const unsigned int* __restrict__ active_experts,
    const unsigned int* __restrict__ active_count,
    const float* __restrict__ route_weights,
    float* __restrict__ output,
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const unsigned int hidden,
    const unsigned int intermediate,
    const unsigned int expert_count,
    const unsigned int route_major_output)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    const unsigned int group = blockIdx.z;
    const unsigned int groups = active_count != nullptr ? *active_count : expert_count;
    if (group >= groups) return;
    const unsigned int expert = active_experts != nullptr ? active_experts[group] : group;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int input_base = blockIdx.y * 128u;
    const unsigned int output_base = (blockIdx.x * 8u + wave) * 16u;
    if (expert >= expert_count || wave >= 8u) return;
    const unsigned int route_begin = route_offsets[group];
    const unsigned int input_rows = route_offsets[group + 1] - route_begin;
    if (input_base >= input_rows) return;
    const unsigned int route_tiles = min(8u, (input_rows - input_base + 15u) / 16u);
    __shared__ unsigned short input_tiles[8][16 * 16];
    __shared__ unsigned short input_residual_tiles[8][16 * 16];
    __shared__ float output_tiles[8][16 * 16];
    CFragment accumulators[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (tile < route_tiles) fill_fragment(accumulators[tile], 0.0f);
    }
    for (unsigned int column_base = 0; column_base < intermediate; column_base += 16u) {
        const unsigned int local_row = threadIdx.x >> 4;
        const unsigned int local_column = threadIdx.x & 15u;
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            if (tile >= route_tiles) continue;
            const unsigned int route_delta = input_base + tile * 16u + local_row;
            float value = 0.0f;
            if (route_delta < input_rows && column_base + local_column < intermediate) {
                value = activated[(unsigned long long)(route_begin + route_delta) * intermediate + column_base + local_column];
                if (!isfinite(value)) value = 0.0f;
            }
            const unsigned short high = zllm_f32_to_bf16(value);
            input_tiles[tile][threadIdx.x] = high;
            input_residual_tiles[tile][threadIdx.x] = zllm_f32_to_bf16(value - zllm_bf16_to_f32(high));
        }
        __syncthreads();
        BFragment weight_fragment;
        unsigned short* weight_access = reinterpret_cast<unsigned short*>(&weight_fragment.mAccess);
        const unsigned int output_row = output_base + (lane & 15u);
        if (output_row < hidden) {
            const unsigned long long weight_row = (unsigned long long)expert * hidden + output_row;
            const unsigned int codes = *reinterpret_cast<const unsigned int*>(
                zllm_mxfp4_native_packed(packed, weight_row, column_base + (lane >> 4) * 8u, intermediate));
            const float scale = zllm_mxfp4_native_scale(scales, weight_row, column_base, intermediate);
            #pragma unroll
            for (unsigned int element = 0; element < 8; ++element) {
                weight_access[element] = zllm_f32_to_bf16(zllm_mxfp4_e2m1((codes >> (element * 4u)) & 15u) * scale);
            }
        } else {
            #pragma unroll
            for (unsigned int element = 0; element < 8; ++element) weight_access[element] = 0u;
        }
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            if (tile >= route_tiles) continue;
            AFragment input_fragment;
            load_matrix_sync(input_fragment, (const bfloat16_t*)input_tiles[tile], 16);
            mma_sync(accumulators[tile], input_fragment, weight_fragment, accumulators[tile]);
            load_matrix_sync(input_fragment, (const bfloat16_t*)input_residual_tiles[tile], 16);
            mma_sync(accumulators[tile], input_fragment, weight_fragment, accumulators[tile]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (tile >= route_tiles) continue;
        store_matrix_sync(output_tiles[wave], accumulators[tile], 16, mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16u * 16u; index += 32u) {
            const unsigned int route_delta = input_base + tile * 16u + index / 16u;
            const unsigned int output_row = output_base + index % 16u;
            if (route_delta < input_rows && output_row < hidden) {
                const unsigned int route = route_begin + route_delta;
                if (route_major_output != 0u) {
                    output[(unsigned long long)route * hidden + output_row] = output_tiles[wave][index];
                } else {
                    atomicAdd(&output[(unsigned long long)token_ids[route] * hidden + output_row], output_tiles[wave][index] * route_weights[route]);
                }
            }
        }
        __syncwarp();
    }
}

extern "C" __global__ void mxfp4_grouped_down_reduce_f32(
    const float* __restrict__ route_output,
    const unsigned int* __restrict__ route_to_grouped,
    const float* __restrict__ route_weights,
    const float* __restrict__ shared,
    float* __restrict__ output,
    const unsigned int rows,
    const unsigned int top_k,
    const unsigned int hidden)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = rows * hidden;
    if (index >= elements) return;
    const unsigned int token = index / hidden;
    const unsigned int column = index - token * hidden;
    float reduced = 0.0f;
    for (unsigned int slot = 0; slot < top_k; ++slot) {
        const unsigned int route = token * top_k + slot;
        const unsigned int grouped = route_to_grouped[route];
        reduced = fmaf(route_output[(unsigned long long)grouped * hidden + column], route_weights[route], reduced);
    }
    output[index] = shared != nullptr ? shared[index] + reduced : reduced;
}

struct ZllmFp8WeightMeta {
    const unsigned char* codes;
    const float* scales;
};

struct ZllmFp8ExpertMeta {
    ZllmFp8WeightMeta gate;
    ZllmFp8WeightMeta up;
    ZllmFp8WeightMeta down;
};

typedef unsigned short __attribute__((ext_vector_type(2))) ZllmFp8Bf16x2;

__device__ __forceinline__ float zllm_fp8_e4m3(unsigned char bits)
{
    const unsigned int exponent = (bits >> 3U) & 15U;
    const unsigned int mantissa = bits & 7U;
    const unsigned int sign = ((unsigned int)bits & 128U) << 24U;
    const float normal = __uint_as_float(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
    const float subnormal = __uint_as_float(sign | __float_as_uint((float)mantissa * 0.001953125F));
    const float finite = exponent == 0U ? subnormal : normal;
    return exponent == 15U && mantissa == 7U ? __uint_as_float(sign | 0x7fc00000U) : finite;
}

__device__ __forceinline__ float zllm_fp8_weight(
    const ZllmFp8WeightMeta& weight,
    unsigned int row,
    unsigned int column,
    unsigned int rows,
    unsigned int columns)
{
    const unsigned int scale_columns = (columns + 127U) / 128U;
    const unsigned long long code = (unsigned long long)row * columns + column;
    const unsigned long long scale = (unsigned long long)(row / 128U) * scale_columns + column / 128U;
    return zllm_fp8_e4m3(weight.codes[code]) * weight.scales[scale];
}

// E4M3 值在 BF16 中精确(3 位尾数 ⊂ 7 位尾数,subnormal = m*2^-9 同样可精确表示),
// 因此解码是纯位拼接,不需要经过 F32 乘法与 RNE 舍入;块 scale 的乘法推迟到
// 段部分和之后,由调用方完成。normal 段指数偏移 = 127-7 = 120,subnormal 用
// 前导零归一化;0x7F/0xFF 保持 NaN 语义。
__device__ __forceinline__ unsigned short zllm_fp8_e4m3_bf16(unsigned char bits)
{
    const unsigned int exponent = (bits >> 3U) & 15U;
    const unsigned int mantissa = bits & 7U;
    const unsigned int sign = ((unsigned int)bits & 128U) << 8U;
    if (exponent == 15U && mantissa == 7U) {
        return (unsigned short)(sign | 0x7FC0U);
    }
    if (exponent != 0U) {
        return (unsigned short)(sign | ((exponent + 120U) << 7U) | (mantissa << 4U));
    }
    if (mantissa == 0U) {
        return (unsigned short)sign;
    }
    const unsigned int lead = 31U - __builtin_clz(mantissa);
    return (unsigned short)(sign | ((118U + lead) << 7U) | ((mantissa ^ (1U << lead)) << (7U - lead)));
}

// decode 向量装填的寄存器视图:常量下标访问由编译器标量化进寄存器,
// __align__ 保证全局指针重解释后生成 v4/v2 宽 load。
struct __align__(16) ZllmFp8Vec16 {
    unsigned char bytes[16];
};

struct __align__(8) ZllmFp8Vec8 {
    unsigned char bytes[8];
};

struct __align__(16) ZllmFp8Shorts8 {
    unsigned short shorts[8];
};

// 每个 wave 计算一条 route 的一个 intermediate 元素；route ids 全程留在 GPU。
extern "C" __global__ void fp8_grouped_gate_up_f32(
    const float* __restrict__ input,
    const unsigned int* __restrict__ route_ids,
    const ZllmFp8ExpertMeta* __restrict__ experts,
    unsigned short* __restrict__ activated,
    unsigned int hidden,
    unsigned int intermediate,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int expert_count)
{
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int wave_count = blockDim.x / warpSize;
    const unsigned int neuron = blockIdx.x * wave_count + wave;
    const unsigned int route = blockIdx.y;
    if (route >= route_count || neuron >= intermediate) return;
    const unsigned int expert = route_ids[route];
    if (expert >= expert_count) {
        if (lane == 0U) activated[(unsigned long long)route * intermediate + neuron] = 0U;
        return;
    }
    const unsigned int token = route / top_k;
    const float* row_input = input + (unsigned long long)token * hidden;
    const ZllmFp8ExpertMeta meta = experts[expert];
    // 快/慢路径共用同一数值语义:权重解码为精确 BF16(E4M3 ⊂ BF16),块
    // scale 在段 dot 之后以 F32 相乘。偶数下标的元素对不跨 128 列 scale 块。
    const unsigned int scale_columns = (hidden + 127U) / 128U;
    const unsigned char* gate_codes = meta.gate.codes + (unsigned long long)neuron * hidden;
    const unsigned char* up_codes = meta.up.codes + (unsigned long long)neuron * hidden;
    const float* gate_scales = meta.gate.scales + (unsigned long long)(neuron / 128U) * scale_columns;
    const float* up_scales = meta.up.scales + (unsigned long long)(neuron / 128U) * scale_columns;
    float gate = 0.0F;
    float up = 0.0F;
    if ((hidden & 15U) == 0U) {
        // decode 主路径:codes 按 16B 向量装填,单条宽 load 覆盖 warp 的
        // 1024 列,替代逐对 2 字节请求;scale 每段只读一次,输入对的 BF16
        // 转换在 gate/up 间复用。
        for (unsigned int column = lane * 16U; column < hidden; column += warpSize * 16U) {
            const ZllmFp8Vec16 gate_vector = *reinterpret_cast<const ZllmFp8Vec16*>(gate_codes + column);
            const ZllmFp8Vec16 up_vector = *reinterpret_cast<const ZllmFp8Vec16*>(up_codes + column);
            const float4 in0 = *reinterpret_cast<const float4*>(row_input + column);
            const float4 in1 = *reinterpret_cast<const float4*>(row_input + column + 4U);
            const float4 in2 = *reinterpret_cast<const float4*>(row_input + column + 8U);
            const float4 in3 = *reinterpret_cast<const float4*>(row_input + column + 12U);
            const float inputs[16] = {in0.x, in0.y, in0.z, in0.w, in1.x, in1.y, in1.z, in1.w, in2.x, in2.y, in2.z, in2.w, in3.x, in3.y, in3.z, in3.w};
            ZllmFp8Bf16x2 values[8];
            #pragma unroll
            for (unsigned int pair = 0; pair < 8U; ++pair) {
                values[pair] = ZllmFp8Bf16x2{
                    zllm_f32_to_bf16(inputs[pair * 2U]),
                    zllm_f32_to_bf16(inputs[pair * 2U + 1U]),
                };
            }
            float gate_part = 0.0F;
            float up_part = 0.0F;
            #pragma unroll
            for (unsigned int pair = 0; pair < 8U; ++pair) {
                gate_part = __builtin_amdgcn_fdot2_f32_bf16(
                    values[pair],
                    ZllmFp8Bf16x2{
                        zllm_fp8_e4m3_bf16(gate_vector.bytes[pair * 2U]),
                        zllm_fp8_e4m3_bf16(gate_vector.bytes[pair * 2U + 1U]),
                    },
                    gate_part,
                    false);
                up_part = __builtin_amdgcn_fdot2_f32_bf16(
                    values[pair],
                    ZllmFp8Bf16x2{
                        zllm_fp8_e4m3_bf16(up_vector.bytes[pair * 2U]),
                        zllm_fp8_e4m3_bf16(up_vector.bytes[pair * 2U + 1U]),
                    },
                    up_part,
                    false);
            }
            gate = fmaf(gate_part, gate_scales[column >> 7U], gate);
            up = fmaf(up_part, up_scales[column >> 7U], up);
        }
    } else {
        unsigned int column = lane * 2U;
        for (; column + 1U < hidden; column += warpSize * 2U) {
            const ZllmFp8Bf16x2 value = {
                zllm_f32_to_bf16(row_input[column]),
                zllm_f32_to_bf16(row_input[column + 1U]),
            };
            const float gate_pair = __builtin_amdgcn_fdot2_f32_bf16(
                value,
                ZllmFp8Bf16x2{
                    zllm_fp8_e4m3_bf16(gate_codes[column]),
                    zllm_fp8_e4m3_bf16(gate_codes[column + 1U]),
                },
                0.0F,
                false);
            const float up_pair = __builtin_amdgcn_fdot2_f32_bf16(
                value,
                ZllmFp8Bf16x2{
                    zllm_fp8_e4m3_bf16(up_codes[column]),
                    zllm_fp8_e4m3_bf16(up_codes[column + 1U]),
                },
                0.0F,
                false);
            gate = fmaf(gate_pair, gate_scales[column >> 7U], gate);
            up = fmaf(up_pair, up_scales[column >> 7U], up);
        }
        if (column < hidden) {
            const float value = zllm_bf16_to_f32(zllm_f32_to_bf16(row_input[column]));
            gate = fmaf(value, zllm_fp8_weight(meta.gate, neuron, column, intermediate, hidden), gate);
            up = fmaf(value, zllm_fp8_weight(meta.up, neuron, column, intermediate, hidden), up);
        }
    }
    for (unsigned int offset = warpSize >> 1U; offset > 0U; offset >>= 1U) {
        gate += __shfl_down(gate, offset);
        up += __shfl_down(up, offset);
    }
    if (lane == 0U) {
        const float silu = gate / (1.0F + expf(-gate));
        activated[(unsigned long long)route * intermediate + neuron] = zllm_f32_to_bf16(silu * up);
    }
}

// 每个 wave 计算一个 token 的一个 hidden 输出；top-k 路由在寄存器中归并，
// shared/residual epilogue 同 kernel 完成，不产生 host scatter。
extern "C" __global__ void fp8_grouped_down_f32(
    const unsigned short* __restrict__ activated,
    const unsigned int* __restrict__ route_ids,
    const float* __restrict__ route_weights,
    const ZllmFp8ExpertMeta* __restrict__ experts,
    const float* __restrict__ shared,
    const float* __restrict__ residual,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int hidden,
    unsigned int intermediate,
    unsigned int top_k,
    unsigned int expert_count)
{
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int wave_count = blockDim.x / warpSize;
    const unsigned int column = blockIdx.x * wave_count + wave;
    const unsigned int token = blockIdx.y;
    if (token >= rows || column >= hidden) return;
    float sum = 0.0F;
    for (unsigned int slot = 0U; slot < top_k; ++slot) {
        const unsigned int route = token * top_k + slot;
        const unsigned int expert = route_ids[route];
        if (expert >= expert_count) continue;
        const ZllmFp8WeightMeta down = experts[expert].down;
        float route_sum = 0.0F;
        // 与 gate/up 相同语义:精确 BF16 解码 + 段后 scale;8 对齐走 8 元素
        // 向量段(codes 8B + activated 16B),否则退回逐对路径。
        const unsigned int scale_columns = (intermediate + 127U) / 128U;
        const unsigned char* down_codes = down.codes + (unsigned long long)column * intermediate;
        const float* down_scales = down.scales + (unsigned long long)(column / 128U) * scale_columns;
        const unsigned short* row_activated = activated + (unsigned long long)route * intermediate;
        if ((intermediate & 7U) == 0U) {
            for (unsigned int inner = lane * 8U; inner < intermediate; inner += warpSize * 8U) {
                const ZllmFp8Vec8 codes = *reinterpret_cast<const ZllmFp8Vec8*>(down_codes + inner);
                const ZllmFp8Shorts8 acts = *reinterpret_cast<const ZllmFp8Shorts8*>(row_activated + inner);
                float part = 0.0F;
                #pragma unroll
                for (unsigned int pair = 0; pair < 4U; ++pair) {
                    part = __builtin_amdgcn_fdot2_f32_bf16(
                        ZllmFp8Bf16x2{acts.shorts[pair * 2U], acts.shorts[pair * 2U + 1U]},
                        ZllmFp8Bf16x2{
                            zllm_fp8_e4m3_bf16(codes.bytes[pair * 2U]),
                            zllm_fp8_e4m3_bf16(codes.bytes[pair * 2U + 1U]),
                        },
                        part,
                        false);
                }
                route_sum = fmaf(part, down_scales[inner >> 7U], route_sum);
            }
        } else {
            unsigned int inner = lane * 2U;
            for (; inner + 1U < intermediate; inner += warpSize * 2U) {
                const ZllmFp8Bf16x2 value = {row_activated[inner], row_activated[inner + 1U]};
                const float pair_dot = __builtin_amdgcn_fdot2_f32_bf16(
                    value,
                    ZllmFp8Bf16x2{
                        zllm_fp8_e4m3_bf16(down_codes[inner]),
                        zllm_fp8_e4m3_bf16(down_codes[inner + 1U]),
                    },
                    0.0F,
                    false);
                route_sum = fmaf(pair_dot, down_scales[inner >> 7U], route_sum);
            }
            if (inner < intermediate) {
                const float value = zllm_bf16_to_f32(row_activated[inner]);
                route_sum = fmaf(value, zllm_fp8_weight(down, column, inner, hidden, intermediate), route_sum);
            }
        }
        sum = fmaf(route_sum, route_weights[route], sum);
    }
    for (unsigned int offset = warpSize >> 1U; offset > 0U; offset >>= 1U) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0U) {
        const unsigned long long index = (unsigned long long)token * hidden + column;
        if (shared != nullptr) sum += shared[index];
        if (residual != nullptr) sum += residual[index];
        output[index] = sum;
    }
}
