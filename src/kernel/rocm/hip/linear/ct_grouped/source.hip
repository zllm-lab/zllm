        }
        __syncwarp();
    }
    }
}

__device__ __forceinline__ void zllm_convrot_hadamard_256(float* values)
{
    const unsigned int lane = threadIdx.x;
    for (unsigned int stride = 1u; stride < 256u; stride *= 4u) {
        if (lane < 64u) {
            const unsigned int base = (lane / stride) * stride * 4u + lane % stride;
            const float a = values[base];
            const float b = values[base + stride];
            const float c = values[base + stride * 2u];
            const float d = values[base + stride * 3u];
            values[base] = a + b + c - d;
            values[base + stride] = a + b - c + d;
            values[base + stride * 2u] = a - b + c + d;
            values[base + stride * 3u] = -a + b + c + d;
        }
        __syncthreads();
    }
}

extern "C" __global__ void convrot_quantize_i8(
    const void* input,
    signed char* output,
    float* scales,
    unsigned int rows,
    unsigned int columns,
    unsigned int input_bf16)
{
    __shared__ float values[256];
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows) {
        for (unsigned int column = lane; column < columns; column += blockDim.x)
            output[(unsigned long long)row * columns + column] = 0;
        if (lane == 0u) scales[row] = 0.0f;
        return;
    }

    float maximum = 0.0f;
    for (unsigned int base = 0; base < columns; base += 256u) {
        const unsigned long long index = (unsigned long long)row * columns + base + lane;
        values[lane] = input_bf16
            ? zllm_bf16_to_f32(((const unsigned short*)input)[index])
            : ((const float*)input)[index];
        __syncthreads();
        zllm_convrot_hadamard_256(values);
        maximum = fmaxf(maximum, fabsf(values[lane] * 0.0625f));
        __syncthreads();
    }

    values[lane] = maximum;
    __syncthreads();
    for (unsigned int offset = 128u; offset; offset >>= 1u) {
        if (lane < offset) values[lane] = fmaxf(values[lane], values[lane + offset]);
        __syncthreads();
    }
    if (lane == 0u) values[0] = values[0] > 0.0f ? values[0] / 127.0f : 1.0f;
    __syncthreads();
    const float scale = values[0];
    if (lane == 0u) scales[row] = scale;
    __syncthreads();

    for (unsigned int base = 0; base < columns; base += 256u) {
        const unsigned long long index = (unsigned long long)row * columns + base + lane;
        values[lane] = input_bf16
            ? zllm_bf16_to_f32(((const unsigned short*)input)[index])
            : ((const float*)input)[index];
        __syncthreads();
        zllm_convrot_hadamard_256(values);
        int code = __float2int_rn(values[lane] * 0.0625f / scale);
        code = code < -127 ? -127 : (code > 127 ? 127 : code);
        output[index] = (signed char)code;
        __syncthreads();
    }
}

// 旋转结果以 F32 暂存在 LDS，避免为了获得 row scale 重读输入并重复一次 Hadamard。
// 这里刻意不降到 BF16：旋转、absmax 和量化比例都保持 F32 精度。
extern "C" __global__ void convrot_quantize_i8_cached(
    const void* input,
    signed char* output,
    float* scales,
    unsigned int rows,
    unsigned int columns,
    unsigned int input_bf16)
{
    __shared__ float values[256];
    extern __shared__ float rotated[];
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows) {
        for (unsigned int column = lane; column < columns; column += blockDim.x)
            output[(unsigned long long)row * columns + column] = 0;
        if (lane == 0u) scales[row] = 0.0f;
        return;
    }

    float maximum = 0.0f;
    for (unsigned int base = 0; base < columns; base += 256u) {
        const unsigned long long index = (unsigned long long)row * columns + base + lane;
        values[lane] = input_bf16
            ? zllm_bf16_to_f32(((const unsigned short*)input)[index])
            : ((const float*)input)[index];
        __syncthreads();
        zllm_convrot_hadamard_256(values);
        const float value = values[lane] * 0.0625f;
        rotated[base + lane] = value;
        maximum = fmaxf(maximum, fabsf(value));
        __syncthreads();
    }

    values[lane] = maximum;
    __syncthreads();
    for (unsigned int offset = 128u; offset; offset >>= 1u) {
        if (lane < offset) values[lane] = fmaxf(values[lane], values[lane + offset]);
        __syncthreads();
    }
    if (lane == 0u) values[0] = values[0] > 0.0f ? values[0] / 127.0f : 1.0f;
    __syncthreads();
    const float scale = values[0];
    if (lane == 0u) scales[row] = scale;

    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        int code = __float2int_rn(rotated[column] / scale);
        code = code < -127 ? -127 : (code > 127 ? 127 : code);
        output[(unsigned long long)row * columns + column] = (signed char)code;
    }
}

extern "C" __global__ void convrot_int8_matmul_wmma(
    const signed char* input,
    const signed char* weight,
    const float* input_scales,
    const float* weight_scales,
    float* output,
    unsigned int logical_rows,
    unsigned int padded_rows,
    unsigned int input_columns,
    unsigned int output_rows)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, signed char, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, signed char, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, int32_t>;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int output_tiles = output_rows / 16u;
    const unsigned int tile = blockIdx.x * waves + wave;
    const unsigned int tile_count = (padded_rows / 16u) * output_tiles;
    if (tile >= tile_count) return;
    const unsigned int tile_row = tile / output_tiles;
    const unsigned int tile_column = tile % output_tiles;
    const unsigned int row_base = tile_row * 16u;
    const unsigned int output_base = tile_column * 16u;

    CFragment accumulator;
    fill_fragment(accumulator, 0);
    for (unsigned int column = 0; column < input_columns; column += 16u) {
        AFragment input_fragment;
        BFragment weight_fragment;
        load_matrix_sync(input_fragment, input + (unsigned long long)row_base * input_columns + column, input_columns);
        load_matrix_sync(weight_fragment, weight + (unsigned long long)output_base * input_columns + column, input_columns);
        mma_sync(accumulator, input_fragment, weight_fragment, accumulator);
    }
    __shared__ int32_t scratch[8][16 * 16];
    store_matrix_sync(scratch[wave], accumulator, 16, mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < 16u * 16u; index += warpSize) {
        const unsigned int row = row_base + index / 16u;
        const unsigned int output_row = output_base + index % 16u;
        if (row < logical_rows) {
            output[(unsigned long long)row * output_rows + output_row] =
                (float)scratch[wave][index] * input_scales[row] * weight_scales[output_row];
        }
    }

}
extern "C" __global__ void convrot_int8_matmul_wmma_row16(
    const signed char* input,
    const signed char* weight,
    const float* input_scales,
    const float* weight_scales,
    float* output,
    unsigned int logical_rows,
    unsigned int padded_rows,
    unsigned int input_columns,
    unsigned int output_rows)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, signed char, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, signed char, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, int32_t>;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int output_tiles = output_rows / 16u;
    const unsigned int tile = blockIdx.x * waves + wave;
    const unsigned int tile_count = (padded_rows / 256u) * output_tiles;
    if (tile >= tile_count) return;
    const unsigned int tile_row = tile / output_tiles;
    const unsigned int tile_column = tile % output_tiles;
    const unsigned int row_base = tile_row * 256u;
    const unsigned int output_base = tile_column * 16u;

    CFragment accumulators[16];
    #pragma unroll
    for (unsigned int part = 0; part < 16; ++part) {
        fill_fragment(accumulators[part], 0);
    }
    for (unsigned int column = 0; column < input_columns; column += 16u) {
        BFragment weight_fragment;
        load_matrix_sync(weight_fragment, weight + (unsigned long long)output_base * input_columns + column, input_columns);
        #pragma unroll
        for (unsigned int part = 0; part < 16; ++part) {
            AFragment input_fragment;
            load_matrix_sync(input_fragment, input + (unsigned long long)(row_base + part * 16u) * input_columns + column, input_columns);
            mma_sync(accumulators[part], input_fragment, weight_fragment, accumulators[part]);
        }
    }
    __shared__ int32_t scratch[16][16 * 16];
    #pragma unroll
    for (unsigned int part = 0; part < 16; ++part) {
        store_matrix_sync(scratch[wave], accumulators[part], 16, mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16u * 16u; index += warpSize) {
            const unsigned int row = row_base + part * 16u + index / 16u;
            const unsigned int output_row = output_base + index % 16u;
            if (row < logical_rows) {
                output[(unsigned long long)row * output_rows + output_row] =
                    (float)scratch[wave][index] * input_scales[row] * weight_scales[output_row];
            }
        }
        __syncwarp();
    }
}

extern "C" __global__ void convrot_int8_matmul_wmma_split4(
    const signed char* input,
    const signed char* weight,
    const float* input_scales,
    const float* weight_scales,
    float* output,
    unsigned int logical_rows,
    unsigned int padded_rows,
    unsigned int input_columns,
    unsigned int output_rows)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, signed char, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, signed char, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, int32_t>;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int output_tiles = output_rows / 16u;
    const unsigned int tile = blockIdx.x * waves + wave;
    const unsigned int tile_count = (padded_rows / 16u) * output_tiles;
    if (tile >= tile_count) return;
    const unsigned int tile_row = tile / output_tiles;
    const unsigned int tile_column = tile % output_tiles;
    const unsigned int row_base = tile_row * 16u;
    const unsigned int output_base = tile_column * 16u;

    CFragment accumulators[4];
    #pragma unroll
    for (unsigned int part = 0; part < 4; ++part) fill_fragment(accumulators[part], 0);
    for (unsigned int column = 0; column < input_columns; column += 64u) {
        #pragma unroll
        for (unsigned int part = 0; part < 4; ++part) {
            const unsigned int offset = column + part * 16u;
            AFragment input_fragment;
            BFragment weight_fragment;
            load_matrix_sync(input_fragment, input + (unsigned long long)row_base * input_columns + offset, input_columns);
            load_matrix_sync(weight_fragment, weight + (unsigned long long)output_base * input_columns + offset, input_columns);
            mma_sync(accumulators[part], input_fragment, weight_fragment, accumulators[part]);
        }
    }
    #pragma unroll
    for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
        accumulators[0][element] += accumulators[1][element] + accumulators[2][element] + accumulators[3][element];
    }
    __shared__ int32_t scratch[8][16 * 16];
    store_matrix_sync(scratch[wave], accumulators[0], 16, mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < 16u * 16u; index += warpSize) {
        const unsigned int row = row_base + index / 16u;
        const unsigned int output_row = output_base + index % 16u;
        if (row < logical_rows) {
            output[(unsigned long long)row * output_rows + output_row] =
                (float)scratch[wave][index] * input_scales[row] * weight_scales[output_row];
        }
    }
}

constexpr unsigned int ZLLM_CONVROT_LDS_PAD = 8u;
