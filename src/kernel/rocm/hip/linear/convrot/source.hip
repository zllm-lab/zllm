
// 一个 block 协同搬运 A/B tile；下一块 K 数据先进入寄存器，再覆盖当前 LDS tile。
template <unsigned int Rows, unsigned int Bytes, unsigned int Threads>
struct ZllmConvRotTileStager {
    static constexpr unsigned int chunks_per_row = Bytes / 16u;
    static constexpr unsigned int chunks = Rows * chunks_per_row;
    static constexpr unsigned int chunks_per_thread = chunks / Threads;
    static constexpr unsigned int stride = Bytes + ZLLM_CONVROT_LDS_PAD;
    static_assert(Bytes % 16u == 0u, "ConvRot tile K 必须按 16 bytes 对齐");
    static_assert(chunks % Threads == 0u, "ConvRot tile 必须由 block 线程完整覆盖");

    uint4 registers[chunks_per_thread];

    __device__ __forceinline__ void load(
        const unsigned char* source,
        unsigned int row_base,
        unsigned int rows,
        unsigned int column_base,
        unsigned int columns)
    {
        #pragma unroll
        for (unsigned int part = 0; part < chunks_per_thread; ++part) {
            const unsigned int chunk = threadIdx.x * chunks_per_thread + part;
            const unsigned int row = row_base + chunk / chunks_per_row;
            const unsigned int column = column_base + (chunk % chunks_per_row) * 16u;
            registers[part] = row < rows && column < columns
                ? *((const uint4*)(source + (unsigned long long)row * columns + column))
                : make_uint4(0u, 0u, 0u, 0u);
        }
    }

    __device__ __forceinline__ void store(unsigned char* destination) const
    {
        #pragma unroll
        for (unsigned int part = 0; part < chunks_per_thread; ++part) {
            const unsigned int chunk = threadIdx.x * chunks_per_thread + part;
            unsigned char* target =
                destination + (chunk / chunks_per_row) * stride + (chunk % chunks_per_row) * 16u;
            *((uint2*)target) = make_uint2(registers[part].x, registers[part].y);
            *((uint2*)(target + 8u)) = make_uint2(registers[part].z, registers[part].w);
        }
    }
};

typedef int ZllmConvRotFragment __attribute__((ext_vector_type(4)));
typedef int ZllmConvRotAccumulator __attribute__((ext_vector_type(8)));

__device__ __forceinline__ ZllmConvRotFragment zllm_convrot_load_fragment(
    const unsigned char* values,
    unsigned int row,
    unsigned int column,
    unsigned int stride)
{
    // LDS 行间 pad 使地址只有 8-byte 对齐，分四个 dword 读取，避免编译成错误的 b128 load。
    const int* source = (const int*)(values + row * stride + column);
    ZllmConvRotFragment fragment;
    fragment[0] = source[0];
    fragment[1] = source[1];
    fragment[2] = source[2];
    fragment[3] = source[3];
    return fragment;
}

__device__ __forceinline__ ZllmConvRotAccumulator zllm_convrot_mma_i8(
    ZllmConvRotFragment input,
    ZllmConvRotFragment weight,
    ZllmConvRotAccumulator accumulator)
{
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, input, true, weight, accumulator, false);
}

template <
    unsigned int BlockRows,
    unsigned int BlockColumns,
    unsigned int BlockK,
    unsigned int WavesM,
    unsigned int WavesN,
    unsigned int TilesM,
    unsigned int TilesN>
__device__ __forceinline__ void zllm_convrot_int8_matmul_wmma_tiled(
    const signed char* input,
    const signed char* weight,
    const float* input_scales,
    const float* weight_scales,
    float* output,
    unsigned int logical_rows,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned char* input_shared,
    unsigned char* weight_shared)
{
    constexpr unsigned int Threads = WavesM * WavesN * 32u;
    constexpr unsigned int InputStride = BlockK + ZLLM_CONVROT_LDS_PAD;
    static_assert(BlockRows == WavesM * TilesM * 16u);
    static_assert(BlockColumns == WavesN * TilesN * 16u);
    static_assert(Threads == 256u);

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int wave_m = wave / WavesN;
    const unsigned int wave_n = wave % WavesN;

    // 相邻 block 优先沿 M 前进，使并发 block 共享同一批权重列的 L2 cache。
    const unsigned int blocks_n = gridDim.x;
    const unsigned int blocks_m = gridDim.y;
    const unsigned int linear_block = blockIdx.y * blocks_n + blockIdx.x;
    const unsigned int blocks_per_group = 4u * blocks_n;
    const unsigned int group = linear_block / blocks_per_group;
    const unsigned int index_in_group = linear_block - group * blocks_per_group;
    const unsigned int remaining_rows = blocks_m - group * 4u;
    const unsigned int group_rows = remaining_rows < 4u ? remaining_rows : 4u;
    const unsigned int block_m = group * 4u + index_in_group % group_rows;
    const unsigned int block_n = index_in_group / group_rows;
    const unsigned int row_base = block_m * BlockRows;
    const unsigned int output_base = block_n * BlockColumns;

    ZllmConvRotAccumulator accumulators[TilesM][TilesN];
    #pragma unroll
    for (unsigned int tile_m = 0; tile_m < TilesM; ++tile_m)
        #pragma unroll
        for (unsigned int tile_n = 0; tile_n < TilesN; ++tile_n)
            accumulators[tile_m][tile_n] =
                ZllmConvRotAccumulator{0, 0, 0, 0, 0, 0, 0, 0};

    ZllmConvRotTileStager<BlockRows, BlockK, Threads> input_stager;
    ZllmConvRotTileStager<BlockColumns, BlockK, Threads> weight_stager;
    input_stager.load((const unsigned char*)input, row_base, logical_rows, 0u, input_columns);
    weight_stager.load((const unsigned char*)weight, output_base, output_rows, 0u, input_columns);
    input_stager.store(input_shared);
    weight_stager.store(weight_shared);
    __syncthreads();

    for (unsigned int column_base = 0; column_base < input_columns; column_base += BlockK) {
        const unsigned int next_column = column_base + BlockK;
        const bool has_next = next_column < input_columns;
        if (has_next) {
            input_stager.load(
                (const unsigned char*)input,
                row_base,
                logical_rows,
                next_column,
                input_columns);
            weight_stager.load(
                (const unsigned char*)weight,
                output_base,
                output_rows,
                next_column,
                input_columns);
        }

        ZllmConvRotFragment input_fragments[2][TilesM];
        ZllmConvRotFragment weight_fragments[2][TilesN];
        #pragma unroll
        for (unsigned int tile_m = 0; tile_m < TilesM; ++tile_m) {
            const unsigned int local_row = (wave_m * TilesM + tile_m) * 16u;
            input_fragments[0][tile_m] = zllm_convrot_load_fragment(
                input_shared, local_row + lane % 16u, 0u, InputStride);
        }
        #pragma unroll
        for (unsigned int tile_n = 0; tile_n < TilesN; ++tile_n) {
            const unsigned int local_output = (wave_n * TilesN + tile_n) * 16u;
            weight_fragments[0][tile_n] = zllm_convrot_load_fragment(
                weight_shared, local_output + lane % 16u, 0u, InputStride);
        }

        #pragma unroll
        for (unsigned int step = 0; step < BlockK / 16u; ++step) {
            const unsigned int current = step & 1u;
            const unsigned int next = current ^ 1u;
            const unsigned int next_k = (step + 1u) * 16u;
            if (step + 1u < BlockK / 16u) {
                #pragma unroll
                for (unsigned int tile_m = 0; tile_m < TilesM; ++tile_m) {
                    const unsigned int local_row = (wave_m * TilesM + tile_m) * 16u;
                    input_fragments[next][tile_m] = zllm_convrot_load_fragment(
                        input_shared, local_row + lane % 16u, next_k, InputStride);
                }
                #pragma unroll
                for (unsigned int tile_n = 0; tile_n < TilesN; ++tile_n) {
                    const unsigned int local_output = (wave_n * TilesN + tile_n) * 16u;
                    weight_fragments[next][tile_n] = zllm_convrot_load_fragment(
                        weight_shared, local_output + lane % 16u, next_k, InputStride);
                }
            }
            #pragma unroll
            for (unsigned int tile_m = 0; tile_m < TilesM; ++tile_m)
                #pragma unroll
                for (unsigned int tile_n = 0; tile_n < TilesN; ++tile_n)
                    accumulators[tile_m][tile_n] = zllm_convrot_mma_i8(
                        input_fragments[current][tile_m],
                        weight_fragments[current][tile_n],
                        accumulators[tile_m][tile_n]);
        }

        if (has_next) {
            __syncthreads();
            input_stager.store(input_shared);
            weight_stager.store(weight_shared);
            __syncthreads();
        }
    }

    // gfx1100 每 lane 持有同一列的八个 INT32；直接做 F32 epilogue，不再落 LDS 中转。
    #pragma unroll
    for (unsigned int tile_m = 0; tile_m < TilesM; ++tile_m) {
        #pragma unroll
        for (unsigned int tile_n = 0; tile_n < TilesN; ++tile_n) {
            const unsigned int output_row =
                output_base + (wave_n * TilesN + tile_n) * 16u + lane % 16u;
            const bool valid_output = output_row < output_rows;
            const float output_scale = valid_output ? weight_scales[output_row] : 0.0f;
            #pragma unroll
            for (unsigned int element = 0; element < 8u; ++element) {
                const unsigned int row =
                    row_base + (wave_m * TilesM + tile_m) * 16u
                    + element * 2u + lane / 16u;
                if (row < logical_rows && valid_output) {
                    output[(unsigned long long)row * output_rows + output_row] =
                        (float)accumulators[tile_m][tile_n][element]
                        * input_scales[row]
                        * output_scale;
                }
            }
        }
    }
}

extern "C" __global__ void convrot_int8_matmul_wmma_tiled128(
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
    (void)padded_rows;
    __shared__ __align__(16) unsigned char input_shared[128u * (64u + ZLLM_CONVROT_LDS_PAD)];
    __shared__ __align__(16) unsigned char weight_shared[128u * (64u + ZLLM_CONVROT_LDS_PAD)];
    zllm_convrot_int8_matmul_wmma_tiled<128, 128, 64, 4, 2, 2, 4>(
        input, weight, input_scales, weight_scales, output,
        logical_rows, input_columns, output_rows,
        input_shared, weight_shared);
}

extern "C" __global__ void convrot_int8_matmul_wmma_tiled256(
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
    (void)padded_rows;
    __shared__ __align__(16) unsigned char input_shared[256u * (64u + ZLLM_CONVROT_LDS_PAD)];
    __shared__ __align__(16) unsigned char weight_shared[128u * (64u + ZLLM_CONVROT_LDS_PAD)];
    zllm_convrot_int8_matmul_wmma_tiled<256, 128, 64, 4, 2, 4, 4>(
        input, weight, input_scales, weight_scales, output,
        logical_rows, input_columns, output_rows,
        input_shared, weight_shared);
}

// ===== GGUF K-quant grouped decode experts(WMMA gate/up + 确定性 down) =====
// 布局与 ct_grouped_* 一致：按 expert 分组的紧凑 route；activated 为 grouped 序 bf16。

struct GgufKqMeta {
