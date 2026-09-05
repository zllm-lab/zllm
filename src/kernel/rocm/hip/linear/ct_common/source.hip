#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void f32_to_bf16(
    const float* input,
    unsigned short* output,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = zllm_f32_to_bf16(input[index]);
}

extern "C" __global__ void bf16_to_f32(
    const unsigned short* input,
    float* output,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = zllm_bf16_to_f32(input[index]);
}

// 常驻 BF16 embedding 表按 token id 行 gather 并展开为 F32;
// decode 每 token 一个 24KB 读,消除 host 反量化 + 同步 H2D。
extern "C" __global__ void gather_bf16_rows_f32(
    const unsigned short* __restrict__ table,
    const unsigned int* __restrict__ ids,
    float* __restrict__ output,
    const unsigned int cols)
{
    const unsigned int row = blockIdx.x;
    const unsigned long long base = (unsigned long long)ids[row] * cols;
    const unsigned long long out_base = (unsigned long long)row * cols;
    for (unsigned int col = threadIdx.x; col < cols; col += blockDim.x) {
        output[out_base + col] = __uint_as_float(((unsigned int)table[base + col]) << 16);
    }
}

union ZllmDenseScratch {
    unsigned short input[8][16 * 64];
    float output[8][16 * 16];
};

// 一个 workgroup 的八个 wave 各算 16 个输出通道，同时覆盖 128 个 token row。
template <bool Aligned, bool DsaQuery>
__device__ __forceinline__ void zllm_dense_matmul_bf16_f32_wmma(
    const unsigned short* input,
    const unsigned short* weight,
    void* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int input_row_block_offset,
    const float* cosine,
    const float* sine,
    unsigned int head_count,
    unsigned int rotary_dim,
    unsigned int position,
    unsigned int split_half,
    ZllmDenseScratch& scratch,
    unsigned short* zero_weight)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int input_base = (blockIdx.y + input_row_block_offset) * 128;
    const unsigned int output_base = (blockIdx.x * 8 + wave) * 16;
    if constexpr (!Aligned) {
        zero_weight[threadIdx.x] = 0;
        __syncthreads();
    }
    CFragment accumulators[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) fill_fragment(accumulators[tile], 0.0f);

    for (unsigned int column_base = 0; column_base < input_columns; column_base += 64) {
        const unsigned int quad = threadIdx.x & 63u;
        const unsigned int row = quad / 4;
        const unsigned int column_quad = (quad & 3u) * 4;
        const unsigned int column_tile = threadIdx.x >> 6;
        const unsigned int column_offset = column_tile * 16;
        const unsigned int column = column_base + column_offset + column_quad;
        const bool valid_column = column + 3 < input_columns;
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            const unsigned int input_row = input_base + tile * 16 + row;
            const unsigned long long source = (unsigned long long)input_row * input_columns + column;
            ((unsigned long long*)scratch.input[tile])[column_tile * 64 + quad] = valid_column && input_row < input_rows
                ? *(const unsigned long long*)(input + source)
                : 0ull;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned int column_tile = 0; column_tile < 4; ++column_tile) {
            const unsigned int column_offset = column_tile * 16;
            const bool valid_weight = output_base < output_rows && column_base + column_offset < input_columns;
            BFragment weight_fragment;
            load_matrix_sync(
                weight_fragment,
                valid_weight
                    ? (const bfloat16_t*)(weight + (unsigned long long)output_base * input_columns + column_base + column_offset)
                    : (const bfloat16_t*)zero_weight,
                valid_weight ? input_columns : 16);
            #pragma unroll
            for (unsigned int tile = 0; tile < 8; ++tile) {
                AFragment input_fragment;
                load_matrix_sync(input_fragment, (const bfloat16_t*)(scratch.input[tile] + column_tile * 16 * 16), 16);
                mma_sync(accumulators[tile], input_fragment, weight_fragment, accumulators[tile]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if constexpr (Aligned) {
            const unsigned int input_row = input_base + tile * 16;
            store_matrix_sync(
                (float*)output + (unsigned long long)input_row * output_rows + output_base,
                accumulators[tile],
                output_rows,
                mem_row_major);
        } else {
            store_matrix_sync(scratch.output[wave], accumulators[tile], 16, mem_row_major);
            if constexpr (DsaQuery) {
                // 八个 wave 恰好覆盖连续 128 个 query 通道；在 LDS 汇合成对
                // accumulator 后完成 RoPE，最终只写消费者需要的 BF16。
                __syncthreads();
                for (unsigned int flat = threadIdx.x; flat < 16u * 128u; flat += blockDim.x) {
                    const unsigned int local_row = flat / 128u;
                    const unsigned int block_column = flat % 128u;
                    const unsigned int input_row = input_base + tile * 16u + local_row;
                    const unsigned int output_row = blockIdx.x * 128u + block_column;
                    if (input_row >= input_rows || output_row >= output_rows) continue;
                    const unsigned int source_wave = block_column / 16u;
                    const unsigned int source_column = block_column % 16u;
                    float value = scratch.output[source_wave][local_row * 16u + source_column];
                    const unsigned int head_dim = output_rows / head_count;
                    const unsigned int head_local = output_row % head_dim;
                    if (head_local < rotary_dim) {
                        const unsigned int half = rotary_dim / 2u;
                        const unsigned int pair = split_half ? head_local % half : head_local / 2u;
                        const unsigned int real_local = split_half ? pair : pair * 2u;
                        const unsigned int imag_local = split_half ? half + pair : real_local + 1u;
                        const unsigned int head_start = output_row - head_local;
                        const unsigned int block_start = blockIdx.x * 128u;
                        const unsigned int real_block = head_start + real_local - block_start;
                        const unsigned int imag_block = head_start + imag_local - block_start;
                        const float real = scratch.output[real_block / 16u][local_row * 16u + real_block % 16u];
                        const float imaginary = scratch.output[imag_block / 16u][local_row * 16u + imag_block % 16u];
                        const float c = cosine[(unsigned long long)(position + input_row) * half + pair];
                        const float s = sine[(unsigned long long)(position + input_row) * half + pair];
                        value = head_local == real_local ? real * c - imaginary * s : imaginary * c + real * s;
                    }
                    ((unsigned short*)output)[(unsigned long long)input_row * output_rows + output_row] = zllm_f32_to_bf16(value);
                }
                __syncthreads();
            } else {
                __syncwarp();
                for (unsigned int index = lane; index < 16 * 16; index += warpSize) {
                    const unsigned int input_row = input_base + tile * 16 + index / 16;
                    const unsigned int output_row = output_base + index % 16;
                    if (input_row < input_rows && output_row < output_rows) {
                        ((float*)output)[(unsigned long long)input_row * output_rows + output_row] = scratch.output[wave][index];
                    }
                }
                __syncwarp();
            }
        }
    }
}

extern "C" __global__ void dense_matmul_bf16_f32_wmma(
    const unsigned short* input,
    const unsigned short* weight,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int input_row_block_offset)
{
    __shared__ ZllmDenseScratch scratch;
    __shared__ unsigned short zero_weight[16 * 16];
    zllm_dense_matmul_bf16_f32_wmma<false, false>(input, weight, output, input_rows, input_columns, output_rows, input_row_block_offset, nullptr, nullptr, 0u, 0u, 0u, 0u, scratch, zero_weight);
}

extern "C" __global__ void dense_matmul_bf16_f32_wmma_aligned(
    const unsigned short* input,
    const unsigned short* weight,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int input_row_block_offset)
{
    __shared__ ZllmDenseScratch scratch;
    zllm_dense_matmul_bf16_f32_wmma<true, false>(input, weight, output, input_rows, input_columns, output_rows, input_row_block_offset, nullptr, nullptr, 0u, 0u, 0u, 0u, scratch, nullptr);
}

extern "C" __global__ void dense_matmul_bf16_dsa_query_wmma(
    const unsigned short* input,
    const unsigned short* weight,
    unsigned short* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int input_row_block_offset,
    const float* cosine,
    const float* sine,
    unsigned int head_count,
    unsigned int rotary_dim,
    unsigned int position,
    unsigned int split_half)
{
    __shared__ ZllmDenseScratch scratch;
    __shared__ unsigned short zero_weight[16 * 16];
    zllm_dense_matmul_bf16_f32_wmma<false, true>(input, weight, output, input_rows, input_columns, output_rows, input_row_block_offset, cosine, sine, head_count, rotary_dim, position, split_half, scratch, zero_weight);
}

// compressed-tensors W4A16/W8A16：压缩权重原地解码，每个 block 同时计算十六个 token row。
typedef unsigned short __attribute__((ext_vector_type(2))) zllm_ct_bf16x2;

// int8 到 BF16 的完整映射常驻常量缓存，避免 decode 热路径逐元素转换。
__device__ __constant__ unsigned short zllm_ct_i8_bf16[256] = {
    0xc300, 0xc2fe, 0xc2fc, 0xc2fa, 0xc2f8, 0xc2f6, 0xc2f4, 0xc2f2, 0xc2f0, 0xc2ee, 0xc2ec, 0xc2ea, 0xc2e8, 0xc2e6, 0xc2e4, 0xc2e2,
    0xc2e0, 0xc2de, 0xc2dc, 0xc2da, 0xc2d8, 0xc2d6, 0xc2d4, 0xc2d2, 0xc2d0, 0xc2ce, 0xc2cc, 0xc2ca, 0xc2c8, 0xc2c6, 0xc2c4, 0xc2c2,
    0xc2c0, 0xc2be, 0xc2bc, 0xc2ba, 0xc2b8, 0xc2b6, 0xc2b4, 0xc2b2, 0xc2b0, 0xc2ae, 0xc2ac, 0xc2aa, 0xc2a8, 0xc2a6, 0xc2a4, 0xc2a2,
    0xc2a0, 0xc29e, 0xc29c, 0xc29a, 0xc298, 0xc296, 0xc294, 0xc292, 0xc290, 0xc28e, 0xc28c, 0xc28a, 0xc288, 0xc286, 0xc284, 0xc282,
    0xc280, 0xc27c, 0xc278, 0xc274, 0xc270, 0xc26c, 0xc268, 0xc264, 0xc260, 0xc25c, 0xc258, 0xc254, 0xc250, 0xc24c, 0xc248, 0xc244,
    0xc240, 0xc23c, 0xc238, 0xc234, 0xc230, 0xc22c, 0xc228, 0xc224, 0xc220, 0xc21c, 0xc218, 0xc214, 0xc210, 0xc20c, 0xc208, 0xc204,
    0xc200, 0xc1f8, 0xc1f0, 0xc1e8, 0xc1e0, 0xc1d8, 0xc1d0, 0xc1c8, 0xc1c0, 0xc1b8, 0xc1b0, 0xc1a8, 0xc1a0, 0xc198, 0xc190, 0xc188,
    0xc180, 0xc170, 0xc160, 0xc150, 0xc140, 0xc130, 0xc120, 0xc110, 0xc100, 0xc0e0, 0xc0c0, 0xc0a0, 0xc080, 0xc040, 0xc000, 0xbf80,
    0x0000, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110, 0x4120, 0x4130, 0x4140, 0x4150, 0x4160, 0x4170,
    0x4180, 0x4188, 0x4190, 0x4198, 0x41a0, 0x41a8, 0x41b0, 0x41b8, 0x41c0, 0x41c8, 0x41d0, 0x41d8, 0x41e0, 0x41e8, 0x41f0, 0x41f8,
    0x4200, 0x4204, 0x4208, 0x420c, 0x4210, 0x4214, 0x4218, 0x421c, 0x4220, 0x4224, 0x4228, 0x422c, 0x4230, 0x4234, 0x4238, 0x423c,
    0x4240, 0x4244, 0x4248, 0x424c, 0x4250, 0x4254, 0x4258, 0x425c, 0x4260, 0x4264, 0x4268, 0x426c, 0x4270, 0x4274, 0x4278, 0x427c,
    0x4280, 0x4282, 0x4284, 0x4286, 0x4288, 0x428a, 0x428c, 0x428e, 0x4290, 0x4292, 0x4294, 0x4296, 0x4298, 0x429a, 0x429c, 0x429e,
    0x42a0, 0x42a2, 0x42a4, 0x42a6, 0x42a8, 0x42aa, 0x42ac, 0x42ae, 0x42b0, 0x42b2, 0x42b4, 0x42b6, 0x42b8, 0x42ba, 0x42bc, 0x42be,
    0x42c0, 0x42c2, 0x42c4, 0x42c6, 0x42c8, 0x42ca, 0x42cc, 0x42ce, 0x42d0, 0x42d2, 0x42d4, 0x42d6, 0x42d8, 0x42da, 0x42dc, 0x42de,
    0x42e0, 0x42e2, 0x42e4, 0x42e6, 0x42e8, 0x42ea, 0x42ec, 0x42ee, 0x42f0, 0x42f2, 0x42f4, 0x42f6, 0x42f8, 0x42fa, 0x42fc, 0x42fe,
};

__device__ __forceinline__ unsigned short zllm_ct_i8_to_bf16(unsigned int code) {
    const int signed_value = (int)(code & 0xffu) - 128;
    union {
        float value;
        unsigned int bits;
    } converted;
    converted.value = (float)signed_value;
    return (unsigned short)(converted.bits >> 16u);
}

// 8 个连续 BF16 activation 的单次 16B 装载。调用点列偏移恒为 8 的倍数，
// 底层 DeviceBuffer 按 256B 对齐分配，只减少装载指令数，不改变数值与归约顺序。
__device__ __forceinline__ void zllm_ct_load_bf16x8(
    const unsigned short* input,
    zllm_ct_bf16x2& x01,
    zllm_ct_bf16x2& x23,
    zllm_ct_bf16x2& x45,
    zllm_ct_bf16x2& x67
) {
    const uint4 packed = *(const uint4*)__builtin_assume_aligned(input, 16);
    x01 = { (unsigned short)(packed.x & 0xffffu), (unsigned short)(packed.x >> 16u) };
    x23 = { (unsigned short)(packed.y & 0xffffu), (unsigned short)(packed.y >> 16u) };
    x45 = { (unsigned short)(packed.z & 0xffffu), (unsigned short)(packed.z >> 16u) };
    x67 = { (unsigned short)(packed.w & 0xffffu), (unsigned short)(packed.w >> 16u) };
}

// W4G128 单行 GEMV：每个 16-lane subgroup 恰好覆盖一个 128-column group。
// packed word 只读一次，八个 offset-binary INT4 直接组成四对 BF16 operand，
// 避免通用标量路径逐元素执行 BF16->F32、INT4->F32 和八次乘法。
__device__ __forceinline__ float zllm_ct_w4_group128_dot(
    const unsigned short* input,
    const unsigned int* packed,
    unsigned long long input_base,
    unsigned long long packed_base,
    unsigned int group,
    unsigned int lane
) {
    const unsigned int word = packed[packed_base + group * 16u + lane];
    const unsigned int column = group * 128u + lane * 8u;
    zllm_ct_bf16x2 x01;
    zllm_ct_bf16x2 x23;
    zllm_ct_bf16x2 x45;
    zllm_ct_bf16x2 x67;
    zllm_ct_load_bf16x8(input + input_base + column, x01, x23, x45, x67);
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
    float sum = __builtin_amdgcn_fdot2_f32_bf16(x01, w01, 0.0f, false);
    sum = __builtin_amdgcn_fdot2_f32_bf16(x23, w23, sum, false);
    sum = __builtin_amdgcn_fdot2_f32_bf16(x45, w45, sum, false);
    return __builtin_amdgcn_fdot2_f32_bf16(x67, w67, sum, false);
}

__device__ __forceinline__ float zllm_ct_w8_group128_dot(
    const unsigned short* input,
    const unsigned int* packed,
    unsigned long long input_base,
    unsigned long long packed_base,
    unsigned int group,
    unsigned int lane
) {
    const unsigned int word = group * 32u + lane * 2u;
    const unsigned int column = group * 128u + lane * 8u;
    const unsigned int codes0 = packed[packed_base + word];
    const unsigned int codes1 = packed[packed_base + word + 1u];
    const zllm_ct_bf16x2 x01 = {
        input[input_base + column],
        input[input_base + column + 1u]
    };
    const zllm_ct_bf16x2 x23 = {
        input[input_base + column + 2u],
        input[input_base + column + 3u]
    };
    const zllm_ct_bf16x2 x45 = {
        input[input_base + column + 4u],
        input[input_base + column + 5u]
    };
    const zllm_ct_bf16x2 x67 = {
        input[input_base + column + 6u],
        input[input_base + column + 7u]
    };
    const zllm_ct_bf16x2 w01 = {
        zllm_ct_i8_to_bf16(codes0),
        zllm_ct_i8_to_bf16(codes0 >> 8u)
    };
    const zllm_ct_bf16x2 w23 = {
        zllm_ct_i8_to_bf16(codes0 >> 16u),
        zllm_ct_i8_to_bf16(codes0 >> 24u)
    };
    const zllm_ct_bf16x2 w45 = {
        zllm_ct_i8_to_bf16(codes1),
        zllm_ct_i8_to_bf16(codes1 >> 8u)
    };
    const zllm_ct_bf16x2 w67 = {
        zllm_ct_i8_to_bf16(codes1 >> 16u),
        zllm_ct_i8_to_bf16(codes1 >> 24u)
    };
    float sum = __builtin_amdgcn_fdot2_f32_bf16(x01, w01, 0.0f, false);
    sum = __builtin_amdgcn_fdot2_f32_bf16(x23, w23, sum, false);
    sum = __builtin_amdgcn_fdot2_f32_bf16(x45, w45, sum, false);
    return __builtin_amdgcn_fdot2_f32_bf16(x67, w67, sum, false);
}

__device__ __forceinline__ float zllm_ct_w8_group128_wave32_dot(
    const unsigned short* input,
    const unsigned int* packed,
    unsigned long long input_base,
    unsigned long long packed_base,
    unsigned int group,
    unsigned int lane
) {
    const unsigned int word = group * 32u + lane;
    const unsigned int column = group * 128u + lane * 4u;
    const unsigned int codes = packed[packed_base + word];
    const zllm_ct_bf16x2 x01 = {
        input[input_base + column],
        input[input_base + column + 1u]
    };
    const zllm_ct_bf16x2 x23 = {
        input[input_base + column + 2u],
        input[input_base + column + 3u]
    };
    const zllm_ct_bf16x2 w01 = {
        zllm_ct_i8_to_bf16(codes),
        zllm_ct_i8_to_bf16(codes >> 8u)
    };
    const zllm_ct_bf16x2 w23 = {
        zllm_ct_i8_to_bf16(codes >> 16u),
        zllm_ct_i8_to_bf16(codes >> 24u)
    };
    float sum = __builtin_amdgcn_fdot2_f32_bf16(x01, w01, 0.0f, false);
    return __builtin_amdgcn_fdot2_f32_bf16(x23, w23, sum, false);
}

struct ZllmCtW8Rows2Dot {
    float first;
    float second;
};

// MTP target verify 固定两行；量化 code 和 scale 只读一次，同时计算两行。
// 每行内部 fdot2、奇偶 group 累加与 wave 归约顺序保持单行 W8 路径不变。
__device__ __forceinline__ ZllmCtW8Rows2Dot zllm_ct_w8_group128_wave32_rows2_dot(
    const unsigned short* input,
    const unsigned int* packed,
    unsigned long long packed_base,
    unsigned int input_columns,
    unsigned int group,
    unsigned int lane
) {
    const unsigned int word = group * 32u + lane;
    const unsigned int column = group * 128u + lane * 4u;
    const unsigned int codes = packed[packed_base + word];
    const zllm_ct_bf16x2 w01 = {
        zllm_ct_i8_to_bf16(codes),
        zllm_ct_i8_to_bf16(codes >> 8u)
    };
    const zllm_ct_bf16x2 w23 = {
        zllm_ct_i8_to_bf16(codes >> 16u),
        zllm_ct_i8_to_bf16(codes >> 24u)
    };
    const zllm_ct_bf16x2 first01 = { input[column], input[column + 1u] };
    const zllm_ct_bf16x2 first23 = { input[column + 2u], input[column + 3u] };
    const unsigned long long second = (unsigned long long)input_columns + column;
    const zllm_ct_bf16x2 second01 = { input[second], input[second + 1u] };
    const zllm_ct_bf16x2 second23 = { input[second + 2u], input[second + 3u] };
    float first = __builtin_amdgcn_fdot2_f32_bf16(first01, w01, 0.0f, false);
    first = __builtin_amdgcn_fdot2_f32_bf16(first23, w23, first, false);
    float second_sum = __builtin_amdgcn_fdot2_f32_bf16(second01, w01, 0.0f, false);
    second_sum = __builtin_amdgcn_fdot2_f32_bf16(second23, w23, second_sum, false);
    return { first, second_sum };
}

extern "C" __global__ void ct_quantized_matmul_bf16_w8_rows2(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits
) {
    if (input_rows != 2u || group_size != 128u || bits != 8u) return;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wave = threadIdx.x >> 5u;
    const unsigned int output_row = blockIdx.x * (blockDim.x >> 5u) + wave;
    if (output_row >= output_rows) return;
    const unsigned int group_count = input_columns / 128u;
    const unsigned long long packed_base =
        (unsigned long long)output_row * (input_columns / 4u);
    float first_even = 0.0f;
    float first_odd = 0.0f;
    float second_even = 0.0f;
    float second_odd = 0.0f;
    unsigned int group = 0u;
    for (; group + 1u < group_count; group += 2u) {
        const unsigned long long scale_base =
            (unsigned long long)output_row * group_count;
        const float even_scale = zllm_scale(scales, scale_base + group, scale_dtype);
        const float odd_scale = zllm_scale(scales, scale_base + group + 1u, scale_dtype);
        const ZllmCtW8Rows2Dot even = zllm_ct_w8_group128_wave32_rows2_dot(
            input, packed, packed_base, input_columns, group, lane);
        const ZllmCtW8Rows2Dot odd = zllm_ct_w8_group128_wave32_rows2_dot(
            input, packed, packed_base, input_columns, group + 1u, lane);
        first_even += even.first * even_scale;
        first_odd += odd.first * odd_scale;
        second_even += even.second * even_scale;
        second_odd += odd.second * odd_scale;
    }
    if (group < group_count) {
        const float scale = zllm_scale(
            scales, (unsigned long long)output_row * group_count + group, scale_dtype);
        const ZllmCtW8Rows2Dot dot = zllm_ct_w8_group128_wave32_rows2_dot(
            input, packed, packed_base, input_columns, group, lane);
        first_even += dot.first * scale;
        second_even += dot.second * scale;
    }
    float first = first_even + first_odd;
    float second = second_even + second_odd;
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        first += __shfl_down(first, offset, 32);
        second += __shfl_down(second, offset, 32);
    }
    if (lane == 0u) {
        output[output_row] = first;
        output[(unsigned long long)output_rows + output_row] = second;
    }
}

// 多行 verify 共享一次量化 code 与 scale 装载。W8 GEMV 的逐行 grid.y 会让
// 超过 L2 的权重矩阵(如 154880 输出的 lm head ≈952MB)被每个 input row 重读
// 一遍;ROWS 行共享后权重只读一次,每行累加顺序与单行路径逐位一致。
template <int ROWS>
__device__ __forceinline__ void zllm_w8_rows_shared_wave32(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int scale_dtype
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wave = threadIdx.x >> 5u;
    const unsigned int output_row = blockIdx.x * (blockDim.x >> 5u) + wave;
    if (output_row >= output_rows) return;
    const unsigned int group_count = input_columns / 128u;
    const unsigned long long packed_base =
        (unsigned long long)output_row * (input_columns / 4u);
    const unsigned long long scale_base =
        (unsigned long long)output_row * group_count;
    float even[ROWS];
    float odd[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        even[r] = 0.0f;
        odd[r] = 0.0f;
    }
    unsigned int group = 0u;
    for (; group + 1u < group_count; group += 2u) {
        const float even_scale = zllm_scale(scales, scale_base + group, scale_dtype);
        const float odd_scale = zllm_scale(scales, scale_base + group + 1u, scale_dtype);
        #pragma unroll
        for (int parity = 0; parity < 2; ++parity) {
            const unsigned int active = parity == 0 ? group : group + 1u;
            const float scale = parity == 0 ? even_scale : odd_scale;
            const unsigned int word = active * 32u + lane;
            const unsigned int column = active * 128u + lane * 4u;
            const unsigned int codes = packed[packed_base + word];
            const zllm_ct_bf16x2 w01 = {
                zllm_ct_i8_bf16[codes & 0xffu],
                zllm_ct_i8_bf16[(codes >> 8) & 0xffu]
            };
            const zllm_ct_bf16x2 w23 = {
                zllm_ct_i8_bf16[(codes >> 16) & 0xffu],
                zllm_ct_i8_bf16[(codes >> 24) & 0xffu]
            };
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const unsigned long long base =
                    (unsigned long long)r * input_columns + column;
                const zllm_ct_bf16x2 x01 = { input[base], input[base + 1u] };
                const zllm_ct_bf16x2 x23 = { input[base + 2u], input[base + 3u] };
                float sum = __builtin_amdgcn_fdot2_f32_bf16(x01, w01, 0.0f, false);
                sum = __builtin_amdgcn_fdot2_f32_bf16(x23, w23, sum, false);
                if (parity == 0) { even[r] += sum * scale; }
                else { odd[r] += sum * scale; }
            }
        }
    }
    if (group < group_count) {
        const float scale = zllm_scale(scales, scale_base + group, scale_dtype);
        const unsigned int word = group * 32u + lane;
        const unsigned int column = group * 128u + lane * 4u;
        const unsigned int codes = packed[packed_base + word];
        const zllm_ct_bf16x2 w01 = {
            zllm_ct_i8_to_bf16(codes),
            zllm_ct_i8_to_bf16(codes >> 8u)
        };
        const zllm_ct_bf16x2 w23 = {
            zllm_ct_i8_to_bf16(codes >> 16u),
            zllm_ct_i8_to_bf16(codes >> 24u)
        };
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const unsigned long long base =
                (unsigned long long)r * input_columns + column;
            const zllm_ct_bf16x2 x01 = { input[base], input[base + 1u] };
            const zllm_ct_bf16x2 x23 = { input[base + 2u], input[base + 3u] };
            float sum = __builtin_amdgcn_fdot2_f32_bf16(x01, w01, 0.0f, false);
            sum = __builtin_amdgcn_fdot2_f32_bf16(x23, w23, sum, false);
            even[r] += sum * scale;
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float sum = even[r] + odd[r];
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            sum += __shfl_down(sum, offset, 32);
        }
        if (lane == 0u) {
            output[(unsigned long long)r * output_rows + output_row] = sum;
        }
    }
}

extern "C" __global__ void ct_quantized_matmul_bf16_w8_rows3(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits
) {
    if (input_rows != 3u || group_size != 128u || bits != 8u) return;
    zllm_w8_rows_shared_wave32<3>(
        input, packed, scales, output, input_columns, output_rows, scale_dtype);
}

extern "C" __global__ void ct_quantized_matmul_bf16_w8_rows4(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits
) {
    if (input_rows != 4u || group_size != 128u || bits != 8u) return;
    zllm_w8_rows_shared_wave32<4>(
        input, packed, scales, output, input_columns, output_rows, scale_dtype);
}

extern "C" __global__ void ct_quantized_matmul_bf16_w8_rows5(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits
) {
    if (input_rows != 5u || group_size != 128u || bits != 8u) return;
    zllm_w8_rows_shared_wave32<5>(
        input, packed, scales, output, input_columns, output_rows, scale_dtype);
}

extern "C" __global__ void ct_quantized_matmul_bf16_w8_rows6(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits
) {
    if (input_rows != 6u || group_size != 128u || bits != 8u) return;
    zllm_w8_rows_shared_wave32<6>(
        input, packed, scales, output, input_columns, output_rows, scale_dtype);
}

extern "C" __global__ void ct_quantized_matmul_bf16_w8_rows7(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits
) {
    if (input_rows != 7u || group_size != 128u || bits != 8u) return;
    zllm_w8_rows_shared_wave32<7>(
        input, packed, scales, output, input_columns, output_rows, scale_dtype);
}

extern "C" __global__ void ct_quantized_matmul_bf16_w8_rows8(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits
) {
    if (input_rows != 8u || group_size != 128u || bits != 8u) return;
    zllm_w8_rows_shared_wave32<8>(
        input, packed, scales, output, input_columns, output_rows, scale_dtype);
}

extern "C" __global__ void ct_quantized_matmul_bf16_w8_scalar(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits,
    const float* residual
) {
    const unsigned int input_row = blockIdx.y;
    if (input_row >= input_rows || bits != 8u) return;
    const unsigned int group_count = input_columns / group_size;
    const unsigned long long input_base = (unsigned long long)input_row * input_columns;

    if (group_size == 128u && input_columns == 2048u) {
        const unsigned int lane = threadIdx.x & 31u;
        const unsigned int wave = threadIdx.x >> 5u;
        const unsigned int outputs_per_block = blockDim.x >> 5u;
        const unsigned int output_row = blockIdx.x * outputs_per_block + wave;
        if (output_row >= output_rows) return;
        const unsigned long long packed_base =
            (unsigned long long)output_row * (input_columns / 4u);
        float even_sum = 0.0f;
        float odd_sum = 0.0f;
        unsigned int group = 0u;
        for (; group + 1u < group_count; group += 2u) {
            const float even_scale = zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group,
                scale_dtype
            );
            const float odd_scale = zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group + 1u,
                scale_dtype
            );
            const float even_dot = zllm_ct_w8_group128_wave32_dot(
                input, packed, input_base, packed_base, group, lane
            );
            const float odd_dot = zllm_ct_w8_group128_wave32_dot(
                input, packed, input_base, packed_base, group + 1u, lane
            );
            even_sum += even_dot * even_scale;
            odd_sum += odd_dot * odd_scale;
        }
        if (group < group_count) {
            even_sum += zllm_ct_w8_group128_wave32_dot(
                input, packed, input_base, packed_base, group, lane
            ) * zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group,
                scale_dtype
            );
        }
        float sum = even_sum + odd_sum;
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            sum += __shfl_down(sum, offset, 32);
        }
        if (lane == 0u) {
            const unsigned long long output_index =
                (unsigned long long)input_row * output_rows + output_row;
            output[output_index] = residual ? residual[output_index] + sum : sum;
        }
        return;
    }
    if (group_size == 128u && group_count >= 32u) {
        const unsigned int lane32 = threadIdx.x & 31u;
        const unsigned int wave32 = threadIdx.x >> 5u;
        const unsigned int outputs_per_block32 = blockDim.x >> 5u;
        const unsigned int output_row32
            = blockIdx.x * outputs_per_block32 + wave32;
        if (output_row32 >= output_rows) return;
        const unsigned long long packed_base32
            = (unsigned long long)output_row32 * (input_columns / 4u);
        float even_sum32 = 0.0f;
        float odd_sum32 = 0.0f;
        unsigned int group32 = 0u;
        for (; group32 + 1u < group_count; group32 += 2u) {
            even_sum32 += zllm_ct_w8_group128_wave32_dot(
                input, packed, input_base, packed_base32, group32, lane32
            ) * zllm_scale(
                scales,
                (unsigned long long)output_row32 * group_count + group32,
                scale_dtype
            );
            odd_sum32 += zllm_ct_w8_group128_wave32_dot(
                input, packed, input_base, packed_base32, group32 + 1u, lane32
            ) * zllm_scale(
                scales,
                (unsigned long long)output_row32 * group_count + group32 + 1u,
                scale_dtype
            );
        }
        if (group32 < group_count) {
            even_sum32 += zllm_ct_w8_group128_wave32_dot(
                input, packed, input_base, packed_base32, group32, lane32
            ) * zllm_scale(
                scales,
                (unsigned long long)output_row32 * group_count + group32,
                scale_dtype
            );
        }
        float sum32 = even_sum32 + odd_sum32;
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            sum32 += __shfl_down(sum32, offset, 32);
        }
        if (lane32 == 0u) {
            const unsigned long long output_index =
                (unsigned long long)input_row * output_rows + output_row32;
            output[output_index] = residual ? residual[output_index] + sum32 : sum32;
        }
        return;
    }
    // GGUF Q8_0 group32 的 wave32 变体：每 wave 一行、每 lane 一个 group
    // （32 列 = 8 个 packed word，lane 内连续 32B 读取），每 lane 迭代数降 32 倍、
    // 同 blockDim 下 block 数较 8-lane subgroup 翻四倍，小输出矩阵不再欠占用。
    // 归约为组内顺序累加 + 32-lane shfl 树，与 group128 wave32 路径同一数值形态。
    if (group_size == 32u) {
        const unsigned int lane = threadIdx.x & 31u;
        const unsigned int wave = threadIdx.x >> 5u;
        const unsigned int outputs_per_block = blockDim.x >> 5u;
        const unsigned int output_row = blockIdx.x * outputs_per_block + wave;
        if (output_row >= output_rows) return;
        const unsigned long long packed_base =
            (unsigned long long)output_row * (input_columns / 4u);
        const unsigned long long scale_base =
            (unsigned long long)output_row * group_count;
        float sum = 0.0f;
        for (unsigned int group = lane; group < group_count; group += 32u) {
            const unsigned int word_base = group * 8u;
            float group_dot = 0.0f;
            #pragma unroll
            for (unsigned int word = 0u; word < 8u; ++word) {
                const unsigned int codes = packed[packed_base + word_base + word];
                const unsigned int column = group * 32u + word * 4u;
                const zllm_ct_bf16x2 x01 = {
                    input[input_base + column], input[input_base + column + 1u]
                };
                const zllm_ct_bf16x2 x23 = {
                    input[input_base + column + 2u], input[input_base + column + 3u]
                };
                const zllm_ct_bf16x2 w01 = {
                    zllm_ct_i8_to_bf16(codes),
                    zllm_ct_i8_to_bf16(codes >> 8u)
                };
                const zllm_ct_bf16x2 w23 = {
                    zllm_ct_i8_to_bf16(codes >> 16u),
                    zllm_ct_i8_to_bf16(codes >> 24u)
                };
                group_dot = __builtin_amdgcn_fdot2_f32_bf16(
                    x01, w01, group_dot, false
                );
                group_dot = __builtin_amdgcn_fdot2_f32_bf16(
                    x23, w23, group_dot, false
                );
            }
            sum += group_dot * zllm_scale(
                scales, scale_base + group, scale_dtype
            );
        }
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            sum += __shfl_down(sum, offset, 32);
        }
        if (lane == 0u) {
            const unsigned long long output_index =
                (unsigned long long)input_row * output_rows + output_row;
            output[output_index] = residual
                ? residual[output_index] + sum
                : sum;
        }
        return;
    }
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int outputs_per_block = blockDim.x >> 4u;
    const unsigned int output_row = blockIdx.x * outputs_per_block + subgroup;
    if (output_row >= output_rows) return;
    const unsigned int words_per_group = group_size / 4u;
    const unsigned long long packed_base =
        (unsigned long long)output_row * (input_columns / 4u);
    float sum = 0.0f;
    if (group_size == 128u) {
        float even_sum = 0.0f;
        float odd_sum = 0.0f;
        unsigned int group = 0u;
        for (; group + 1u < group_count; group += 2u) {
            const float even_scale = zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group,
                scale_dtype
            );
            const float odd_scale = zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group + 1u,
                scale_dtype
            );
            const float even_dot = zllm_ct_w8_group128_dot(
                input, packed, input_base, packed_base, group, lane
            );
            const float odd_dot = zllm_ct_w8_group128_dot(
                input, packed, input_base, packed_base, group + 1u, lane
            );
            even_sum += even_dot * even_scale;
            odd_sum += odd_dot * odd_scale;
        }
        if (group < group_count) {
            even_sum += zllm_ct_w8_group128_dot(
                input, packed, input_base, packed_base, group, lane
            ) * zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group,
                scale_dtype
            );
        }
        sum = even_sum + odd_sum;
    } else {
        for (unsigned int group = 0; group < group_count; ++group) {
            const float scale = zllm_scale(
                scales,
                (unsigned long long)output_row * group_count + group,
                scale_dtype
            );
            const unsigned int column_base = group * group_size;
            const unsigned int word_base = group * words_per_group;
            for (unsigned int word = lane; word < words_per_group; word += 16u) {
                const unsigned int codes = packed[packed_base + word_base + word];
                const unsigned int column = column_base + word * 4u;
                const zllm_ct_bf16x2 x01 = { input[input_base + column], input[input_base + column + 1u] };
                const zllm_ct_bf16x2 x23 = { input[input_base + column + 2u], input[input_base + column + 3u] };
                const zllm_ct_bf16x2 w01 = { zllm_ct_i8_bf16[codes & 0xffu], zllm_ct_i8_bf16[(codes >> 8) & 0xffu] };
                const zllm_ct_bf16x2 w23 = { zllm_ct_i8_bf16[(codes >> 16) & 0xffu], zllm_ct_i8_bf16[(codes >> 24) & 0xffu] };
                float word_sum = __builtin_amdgcn_fdot2_f32_bf16(x01, w01, 0.0f, false);
                word_sum = __builtin_amdgcn_fdot2_f32_bf16(x23, w23, word_sum, false);
                sum += word_sum * scale;
            }
        }
    }
    for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
        sum += __shfl_down(sum, offset, 16);
    }
    if (lane == 0u) {
        const unsigned long long output_index =
            (unsigned long long)input_row * output_rows + output_row;
        output[output_index] = residual ? residual[output_index] + sum : sum;
    }
}

extern "C" __global__ void ct_quantized_matmul_bf16_scalar(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits)
{
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int subgroups_per_block = blockDim.x >> 4u;
    const unsigned int output_row = blockIdx.x * subgroups_per_block + subgroup;
    const unsigned int input_row = blockIdx.y;
    if (output_row >= output_rows || input_row >= input_rows) return;

    const unsigned int values_per_word = 32u / bits;
    const unsigned long long packed_columns =
        ((unsigned long long)input_columns + values_per_word - 1) / values_per_word;
    const unsigned long long groups = input_columns / group_size;
    const unsigned int words_per_group = group_size / values_per_word;

    if (bits == 4u && group_size == 128u) {
        const unsigned long long input_base =
            (unsigned long long)input_row * input_columns;
        const unsigned long long packed_base =
            (unsigned long long)output_row * (input_columns / 8u);
        float even_sum = 0.0f;
        float odd_sum = 0.0f;
        unsigned int group = 0u;
        for (; group + 1u < groups; group += 2u) {
            even_sum += zllm_ct_w4_group128_dot(
                input, packed, input_base, packed_base, group, lane
            ) * zllm_scale(
                scales,
                (unsigned long long)output_row * groups + group,
                scale_dtype
            );
            odd_sum += zllm_ct_w4_group128_dot(
                input, packed, input_base, packed_base, group + 1u, lane
            ) * zllm_scale(
                scales,
                (unsigned long long)output_row * groups + group + 1u,
                scale_dtype
            );
        }
        if (group < groups) {
            even_sum += zllm_ct_w4_group128_dot(
                input, packed, input_base, packed_base, group, lane
            ) * zllm_scale(
                scales,
                (unsigned long long)output_row * groups + group,
                scale_dtype
            );
        }
        float sum = even_sum + odd_sum;
        for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
            sum += __shfl_down(sum, offset, 16);
        }
        if (lane == 0u) {
            output[(unsigned long long)input_row * output_rows + output_row] = sum;
        }
        return;
    }

    float sum = 0.0f;

    for (unsigned int group = 0; group < groups; ++group) {
        const unsigned long long parameter =
            (unsigned long long)output_row * groups + group;
        const float scale = zllm_scale(scales, parameter, scale_dtype);
        const unsigned int word_base = group * words_per_group;
        for (unsigned int word_index = lane;
             word_index < words_per_group;
             word_index += 16u) {
            const unsigned int word = packed[
                (unsigned long long)output_row * packed_columns + word_base + word_index];
            for (unsigned int value = 0; value < values_per_word; ++value) {
                const unsigned int column =
                    (word_base + word_index) * values_per_word + value;
                if (column >= input_columns) continue;
                const unsigned int code =
                    (word >> (value * bits)) & ((1u << bits) - 1u);
                const int signed_code = bits == 4 ? (int)code - 8 : (int)code - 128;
                sum += zllm_bf16_to_f32(
                    input[(unsigned long long)input_row * input_columns + column])
                    * ((float)signed_code * scale);
            }
        }
    }

    for (unsigned int offset = 8u; offset > 0; offset >>= 1u)
        sum += __shfl_down(sum, offset, 16);
    if (lane == 0u)
        output[(unsigned long long)input_row * output_rows + output_row] = sum;
}

// B2 是高并发 decode 的主要形态；固定两行可避免通用 B8 kernel
// 为未使用的六行保留累加器，但仍与单行保持相同累加顺序。
extern "C" __global__ void ct_quantized_matmul_bf16_w4_rows2(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits)
{
    if (input_rows != 2u || group_size != 128u || bits != 4u) return;
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int subgroups_per_block = blockDim.x >> 4u;
    const unsigned int output_row = blockIdx.x * subgroups_per_block + subgroup;
    if (output_row >= output_rows) return;

    const unsigned int groups = input_columns / 128u;
    const unsigned long long packed_base =
        (unsigned long long)output_row * (input_columns / 8u);
    const unsigned long long scale_base =
        (unsigned long long)output_row * groups;
    float even0 = 0.0f;
    float odd0 = 0.0f;
    float even1 = 0.0f;
    float odd1 = 0.0f;
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
        const unsigned long long base0 = column;
        const unsigned long long base1 = (unsigned long long)input_columns + column;
        const zllm_ct_bf16x2 x001 = {input[base0], input[base0 + 1u]};
        const zllm_ct_bf16x2 x023 = {input[base0 + 2u], input[base0 + 3u]};
        const zllm_ct_bf16x2 x045 = {input[base0 + 4u], input[base0 + 5u]};
        const zllm_ct_bf16x2 x067 = {input[base0 + 6u], input[base0 + 7u]};
        const zllm_ct_bf16x2 x101 = {input[base1], input[base1 + 1u]};
        const zllm_ct_bf16x2 x123 = {input[base1 + 2u], input[base1 + 3u]};
        const zllm_ct_bf16x2 x145 = {input[base1 + 4u], input[base1 + 5u]};
        const zllm_ct_bf16x2 x167 = {input[base1 + 6u], input[base1 + 7u]};
        float dot0 = __builtin_amdgcn_fdot2_f32_bf16(x001, w01, 0.0f, false);
        dot0 = __builtin_amdgcn_fdot2_f32_bf16(x023, w23, dot0, false);
        dot0 = __builtin_amdgcn_fdot2_f32_bf16(x045, w45, dot0, false);
        dot0 = __builtin_amdgcn_fdot2_f32_bf16(x067, w67, dot0, false) * scale;
        float dot1 = __builtin_amdgcn_fdot2_f32_bf16(x101, w01, 0.0f, false);
        dot1 = __builtin_amdgcn_fdot2_f32_bf16(x123, w23, dot1, false);
        dot1 = __builtin_amdgcn_fdot2_f32_bf16(x145, w45, dot1, false);
        dot1 = __builtin_amdgcn_fdot2_f32_bf16(x167, w67, dot1, false) * scale;
        if ((group & 1u) == 0u) {
            even0 += dot0;
            even1 += dot1;
        } else {
            odd0 += dot0;
            odd1 += dot1;
        }
    }
    float sum0 = even0 + odd0;
    float sum1 = even1 + odd1;
    for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
        sum0 += __shfl_down(sum0, offset, 16);
        sum1 += __shfl_down(sum1, offset, 16);
    }
    if (lane == 0u) {
        output[output_row] = sum0;
        output[(unsigned long long)output_rows + output_row] = sum1;
    }
}

// 2-8 路 decode lane 共享一次 W4 code/scale 装载。每行保持与单行
// GEMV 相同的 fdot2、奇偶 group 累加和 16-lane 归约顺序；这里只共享
// 模型权重，不合并 session 的 KV、DSA 或逻辑位置。
extern "C" __global__ void ct_quantized_matmul_bf16_w4_rows8(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits)
{
    if (input_rows < 2u || input_rows > 8u || group_size != 128u || bits != 4u) return;
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int subgroups_per_block = blockDim.x >> 4u;
    const unsigned int output_row = blockIdx.x * subgroups_per_block + subgroup;
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

union ZllmWmmaScratch {
    unsigned short lut[8][16 * 16];
    float output[8][16 * 16];
};

// 一个 workgroup 的八个 wave 各算一个 16x16 输出 tile；W4/W8 只在 LDS tile 内解码。
extern "C" __global__ void ct_quantized_matmul_bf16_wmma(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int input_base = blockIdx.y * 128;
    const unsigned int output_base =
        (blockIdx.x * (blockDim.x / warpSize) + wave) * 16;
    const unsigned int values_per_word = 32u / bits;
    const unsigned long long packed_columns =
        ((unsigned long long)input_columns + values_per_word - 1) / values_per_word;
    const unsigned long long groups = input_columns / group_size;

    __shared__ unsigned short input_tiles[8][16 * 16];
    __shared__ unsigned short weight_tiles[16][16 * 16];
    __shared__ union {
        unsigned short lut[16][16 * 16];
        float output[16][16 * 16];
    } scratch;
    CFragment accumulator_fragments[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) fill_fragment(accumulator_fragments[tile], 0.0f);

    for (unsigned int column_base = 0; column_base < input_columns; column_base += 16) {
        const unsigned int flat = threadIdx.x;
        if (flat < 16 * 16) {
            const unsigned int input_row = flat / 16;
            const unsigned int input_column = column_base + flat % 16;
            #pragma unroll
            for (unsigned int tile = 0; tile < 8; ++tile) {
                input_tiles[tile][flat]
                    = input_base + tile * 16 + input_row < input_rows
                        && input_column < input_columns
                    ? input[(unsigned long long)(input_base + tile * 16 + input_row)
                        * input_columns + input_column]
                    : 0;
            }
        }

        const bool use_lut = bits == 4 && group_size % 16 == 0;
        if (use_lut && column_base % group_size == 0 && lane < 16) {
            const unsigned int output_row = output_base + lane;
            #pragma unroll
            for (unsigned int code = 0; code < 16; ++code) {
                scratch.lut[wave][lane * 16 + code] = output_row < output_rows
                    ? zllm_f32_to_bf16((float)((int)code - 8) * zllm_scale(
                        scales,
                        (unsigned long long)output_row * groups + column_base / group_size,
                        scale_dtype))
                    : 0;
            }
        }
        if (use_lut) __syncwarp();
        if (lane < 16) {
            const unsigned int output_row = output_base + lane;
            if (output_row < output_rows) {
                const unsigned long long row_words =
                    (unsigned long long)output_row * packed_columns;
                const unsigned long long row_groups = (unsigned long long)output_row * groups;
                const bool w8_group128 = bits == 8u && group_size == 128u;
                const bool one_group = column_base / group_size == (column_base + 15) / group_size;
                const float tile_scale = !use_lut && one_group
                    ? zllm_scale(scales, row_groups + column_base / group_size, scale_dtype) : 0.0f;
                if (w8_group128) {
                    #pragma unroll
                    for (unsigned int word_column = 0; word_column < 16; word_column += 4) {
                        const unsigned int word = packed[row_words + (column_base + word_column) / 4u];
                        #pragma unroll
                        for (unsigned int element = 0; element < 4; ++element) {
                            const unsigned int code =
                                (word >> (element * 8u)) & 255u;
                            weight_tiles[wave][lane * 16 + word_column + element] =
                                zllm_f32_to_bf16((float)((int)code - 128) * tile_scale);
                        }
                    }
                } else {
                    #pragma unroll
                    for (unsigned int word_column = 0; word_column < 16; word_column += values_per_word) {
                        const unsigned int column = column_base + word_column;
                        const unsigned int word = packed[row_words + column / values_per_word];
                        #pragma unroll
                        for (unsigned int element = 0; element < 8; ++element) {
                            if (element < values_per_word) {
                                const unsigned int weight_column = column + element;
                                const unsigned int code =
                                    (word >> (element * bits)) & ((1u << bits) - 1u);
                                const int signed_code = bits == 4 ? (int)code - 8 : (int)code - 128;
                                if (use_lut) {
                                    weight_tiles[wave][lane * 16 + word_column + element] =
                                        scratch.lut[wave][lane * 16 + code];
                                } else {
                                    const float scale = one_group ? tile_scale : zllm_scale(
                                        scales, row_groups + weight_column / group_size, scale_dtype);
                                    weight_tiles[wave][lane * 16 + word_column + element] =
                                        zllm_f32_to_bf16((float)signed_code * scale);
                                }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (unsigned int column = 0; column < 16; ++column) {
                    weight_tiles[wave][lane * 16 + column] = 0;
                }
            }
        }
        __syncthreads();

        BFragment weight_fragment;
        load_matrix_sync(weight_fragment, (const bfloat16_t*)weight_tiles[wave], 16);
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            AFragment input_fragment;
            load_matrix_sync(input_fragment, (const bfloat16_t*)input_tiles[tile], 16);
            mma_sync(accumulator_fragments[tile], input_fragment, weight_fragment, accumulator_fragments[tile]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        store_matrix_sync(scratch.output[wave], accumulator_fragments[tile], 16, mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += warpSize) {
            const unsigned int input_row = input_base + tile * 16 + index / 16;
            const unsigned int output_row = output_base + index % 16;
            if (input_row < input_rows && output_row < output_rows) {
                output[(unsigned long long)input_row * output_rows + output_row] = scratch.output[wave][index];
            }
        }
        __syncwarp();
    }
}

// GFX11 的 16x16 B fragment 每 lane 持有同一输出行连续八个 K 元素。
// W8 的 group 边界按 16 列对齐时直接在寄存器中解码，避免 weight tile
// 写入 LDS 后再读回；G32/G128 只在 scale 索引上不同。
extern "C" __global__ void ct_quantized_matmul_bf16_w8_g128_wmma(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits,
    const unsigned short* residual)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    static_assert(BFragment::num_elements == 8, "GFX11 W8 B fragment lane layout 已改变");
    (void)bits;

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int input_base = blockIdx.y * 128;
    const unsigned int output_base =
        (blockIdx.x * (blockDim.x / warpSize) + wave) * 16;
    const unsigned long long packed_columns = input_columns / 4u;
    const unsigned long long groups = input_columns / group_size;

    __shared__ unsigned short input_tiles[8][16 * 16];
    __shared__ float output_scratch[16][16 * 16];
    CFragment accumulator_fragments[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        fill_fragment(accumulator_fragments[tile], 0.0f);
    }

    for (unsigned int column_base = 0; column_base < input_columns; column_base += 16u) {
        const unsigned int flat = threadIdx.x;
        if (flat < 16u * 16u) {
            const unsigned int input_row = flat / 16u;
            const unsigned int input_column = column_base + flat % 16u;
            #pragma unroll
            for (unsigned int tile = 0; tile < 8; ++tile) {
                input_tiles[tile][flat]
                    = input_base + tile * 16u + input_row < input_rows
                    ? input[(unsigned long long)(input_base + tile * 16u + input_row)
                        * input_columns + input_column]
                    : 0;
            }
        }

        const unsigned int output_row = output_base + (lane & 15u);
        const unsigned int lane_column = column_base + (lane >> 4u) * 8u;
        unsigned int word0 = 0x80808080u;
        unsigned int word1 = 0x80808080u;
        float tile_scale = 0.0f;
        if (output_row < output_rows) {
            const unsigned long long row_words =
                (unsigned long long)output_row * packed_columns;
            word0 = packed[row_words + lane_column / 4u];
            word1 = packed[row_words + lane_column / 4u + 1u];
            tile_scale = zllm_scale(
                scales,
                (unsigned long long)output_row * groups + column_base / group_size,
                scale_dtype);
        }

        BFragment weight_fragment;
        #pragma unroll
        for (unsigned int element = 0; element < 8; ++element) {
            const unsigned int word = element < 4u ? word0 : word1;
            const unsigned int code = (word >> ((element & 3u) * 8u)) & 255u;
            weight_fragment[element].data =
                zllm_f32_to_bf16((float)((int)code - 128) * tile_scale);
        }

        __syncthreads();
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            AFragment input_fragment;
            load_matrix_sync(input_fragment, (const bfloat16_t*)input_tiles[tile], 16);
            mma_sync(
                accumulator_fragments[tile],
                input_fragment,
                weight_fragment,
                accumulator_fragments[tile]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        store_matrix_sync(
            output_scratch[wave], accumulator_fragments[tile], 16, mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16u * 16u; index += warpSize) {
            const unsigned int input_row = input_base + tile * 16u + index / 16u;
            const unsigned int output_row = output_base + index % 16u;
            if (input_row < input_rows && output_row < output_rows) {
                const unsigned long long output_index
                    = (unsigned long long)input_row * output_rows + output_row;
                output[output_index] = residual
                    ? zllm_bf16_to_f32(residual[output_index]) + output_scratch[wave][index]
                    : output_scratch[wave][index];
            }
        }
        __syncwarp();
    }
}

// GFX11 的 W4 group128 直接填充 B fragment；每 lane 一次读取八个连续权重。
// 避免通用路径为每个 16x16 K tile 建 LUT、写 LDS，再由 rocWMMA 读回。
template <unsigned int SEGMENTS>
__device__ __forceinline__ void ct_quantized_matmul_bf16_w4_g128_wmma_impl(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits,
    const unsigned short* residual,
    unsigned short* input_tiles,
    float* output_scratch)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    static_assert(BFragment::num_elements == 8, "GFX11 W4 B fragment lane layout 已改变");
    (void)group_size;
    (void)scale_dtype;
    (void)bits;
    (void)residual;

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int input_base = blockIdx.y * 128;
    const unsigned int output_base =
        (blockIdx.x * (blockDim.x / warpSize) + wave) * 16;
    const unsigned long long packed_columns = input_columns / 8u;
    const unsigned long long groups = input_columns / 128u;

    CFragment accumulator_fragments[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        fill_fragment(accumulator_fragments[tile], 0.0f);
    }

    for (unsigned int column_base = 0;
         column_base < input_columns;
         column_base += SEGMENTS * 16u) {
        const unsigned int local = threadIdx.x;
        if (local < 16u * 16u) {
            const unsigned int input_row = local / 16u;
            #pragma unroll
            for (unsigned int segment = 0; segment < SEGMENTS; ++segment) {
                const unsigned int input_column =
                    column_base + segment * 16u + local % 16u;
                #pragma unroll
                for (unsigned int tile = 0; tile < 8; ++tile) {
                    input_tiles[(segment * 8u + tile) * 16u * 16u + local]
                        = input_base + tile * 16u + input_row < input_rows
                        ? input[(unsigned long long)(input_base + tile * 16u + input_row)
                            * input_columns + input_column]
                        : 0;
                }
            }
        }

        const unsigned int output_row = output_base + (lane & 15u);
        const unsigned int lane_column = column_base + (lane >> 4u) * 8u;
        unsigned int words[SEGMENTS];
        #pragma unroll
        for (unsigned int segment = 0; segment < SEGMENTS; ++segment) {
            words[segment] = 0x88888888u;
        }
        float tile_scale = 0.0f;
        if (output_row < output_rows) {
            const unsigned long long row_words =
                (unsigned long long)output_row * packed_columns;
            #pragma unroll
            for (unsigned int segment = 0; segment < SEGMENTS; ++segment) {
                words[segment] = packed[
                    row_words + lane_column / 8u + segment * 2u];
            }
            tile_scale = zllm_bf16_to_f32(((const unsigned short*)scales)[
                (unsigned long long)output_row * groups + column_base / 128u]);
        }

        __syncthreads();
        #pragma unroll
        for (unsigned int segment = 0; segment < SEGMENTS; ++segment) {
            BFragment weight_fragment;
            #pragma unroll
            for (unsigned int element = 0; element < 8; ++element) {
                const unsigned int code = (words[segment] >> (element * 4u)) & 15u;
                weight_fragment[element].data =
                    zllm_f32_to_bf16((float)((int)code - 8) * tile_scale);
            }
            #pragma unroll
            for (unsigned int tile = 0; tile < 8; ++tile) {
                AFragment input_fragment;
                load_matrix_sync(
                    input_fragment,
                    (const bfloat16_t*)(input_tiles
                        + (segment * 8u + tile) * 16u * 16u),
                    16);
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
        store_matrix_sync(
            output_scratch + wave * 16u * 16u,
            accumulator_fragments[tile],
            16,
            mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16u * 16u; index += warpSize) {
            const unsigned int input_row = input_base + tile * 16u + index / 16u;
            const unsigned int output_row = output_base + index % 16u;
            if (input_row < input_rows && output_row < output_rows) {
                output[(unsigned long long)input_row * output_rows + output_row]
                    = output_scratch[wave * 16u * 16u + index];
            }
        }
        __syncwarp();
    }
}

// 8-wave 小输出保持 32 列，维持四个 workgroup/CU 的 LDS 档位。
extern "C" __global__ void ct_quantized_matmul_bf16_w4_g128_wmma(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits,
    const unsigned short* residual)
{
    __shared__ unsigned short input_tiles[16][16 * 16];
    __shared__ float output_scratch[16][16 * 16];
    ct_quantized_matmul_bf16_w4_g128_wmma_impl<2>(
        input, packed, scales, output, input_rows, input_columns, output_rows,
        group_size, scale_dtype, bits, residual, &input_tiles[0][0],
        &output_scratch[0][0]);
}

// 16-wave paired q_b/o_proj 已由 wave 数限制为每 CU 两个 workgroup；把同一
// G128 scale 的四段 K 合成一次 LDS barrier，不再降低实际 occupancy。
extern "C" __global__ void ct_quantized_matmul_bf16_w4_g128_wmma_k64(
    const unsigned short* input,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int input_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits,
    const unsigned short* residual)
{
    __shared__ unsigned short input_tiles[32][16 * 16];
    __shared__ float output_scratch[16][16 * 16];
    ct_quantized_matmul_bf16_w4_g128_wmma_impl<4>(
        input, packed, scales, output, input_rows, input_columns, output_rows,
        group_size, scale_dtype, bits, residual, &input_tiles[0][0],
        &output_scratch[0][0]);
}

struct ZllmCtWeightMeta {
