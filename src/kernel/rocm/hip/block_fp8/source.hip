// E4M3 → F32(单字节解码,RDNA3 上没有 FP8 dot,必须先升精度)。
__device__ __forceinline__ float zllm_block_fp8_e4m3_to_f32(unsigned char bits)
{
    const unsigned int exponent = (bits >> 3) & 0x0fu;
    const unsigned int mantissa = bits & 0x07u;
    const unsigned int sign = ((unsigned int)bits & 0x80u) << 24u;
    // normal 的指数和 3-bit mantissa 可直接嵌入 IEEE F32，结果逐位精确。
    const float normal = __uint_as_float(sign | ((exponent + 120u) << 23u) | (mantissa << 20u));
    // subnormal = mantissa × 2^-9；只拼接 sign，保留 ±0。
    const float subnormal = __uint_as_float(sign | __float_as_uint((float)mantissa * 0.001953125F));
    const float finite = exponent == 0u ? subnormal : normal;
    // 用选择而不是改变 exec mask，避免随机 FP8 code 让 wave 在每个元素分叉。
    return exponent == 0x0fu && mantissa == 0x07u ? __uint_as_float(sign | 0x7fc00000u) : finite;
}

// E8M0 byte → F32 scale(2^(byte - 127))。
__device__ __forceinline__ float zllm_block_fp8_scale(unsigned char byte)
{
    // 与 `BlockFp8Matrix::decode()` host 路径一致:scale = 2^(byte - 127)。
    // 用 bit-trick:scale_byte + 127 放到 exponent 段,sign/mantissa 置 0。
    unsigned int bits = ((unsigned int)byte << 23);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float zllm_block_fp8_dot_row_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const unsigned int input_columns,
    const unsigned int row,
    const unsigned int block_row_shift,
    const unsigned int block_col_shift,
    const unsigned int lane,
    const unsigned int wavefront_size)
{
    const unsigned int block_row_index = row >> block_row_shift;
    const unsigned int block_cols = 1u << block_col_shift;
    const unsigned int scale_cols = (input_columns + block_cols - 1u) >> block_col_shift;
    float partial = 0.0F;
    for (unsigned int column = lane * 16u; column < input_columns; column += wavefront_size * 16u) {
        const unsigned int block_col_index = column >> block_col_shift;
        const float scale = zllm_block_fp8_scale(scales[block_row_index * scale_cols + block_col_index]);
        #pragma unroll
        for (unsigned int offset = 0; offset < 16u; ++offset) {
            const unsigned int col = column + offset;
            const unsigned char code = codes[(unsigned long long)row * input_columns + col];
            partial += zllm_block_fp8_e4m3_to_f32(code) * scale * input[col];
        }
    }
    for (unsigned int offset = wavefront_size >> 1; offset > 0u; offset >>= 1u) partial += __shfl_down(partial, offset, wavefront_size);
    return partial;
}

// GEMV:output[M] = input[K] @ weight[M,K]^T。
// 每个 WG 处理 weight 一行，每个 lane 累计 1 个 K 段。
extern "C" __global__ void block_fp8_gemv_f32(
    const float* __restrict__ input,        // [K] f32
    const unsigned char* __restrict__ codes, // [M, K] e4m3 行优先
    const unsigned char* __restrict__ scales,// [M/BR, K/BC] e8m0
    float* __restrict__ output,             // [M] f32
    unsigned int input_rows,                // 与 gemm 共用同一份 launch 参数;gemv 恒为 1
    unsigned int input_columns,             // K
    unsigned int weight_rows,               // M
    unsigned int block_row_shift,           // log2(BR)
    unsigned int block_col_shift)           // log2(BC)
{
    (void)input_rows;
    const unsigned int row = blockIdx.x;
    if (row >= weight_rows) return;
    const unsigned int lane = threadIdx.x;
    const unsigned int wavefront_size = (blockDim.x < 64u) ? blockDim.x : 64u;
    const float partial = zllm_block_fp8_dot_row_f32(input, codes, scales, input_columns, row, block_row_shift, block_col_shift, lane, wavefront_size);
    if (lane == 0u) output[row] = partial;
}

// Small-N GEMM:output[N,M] = input[N,K] @ weight[M,K]^T。
// 一个 wave 负责一个输出列，weight code/scale 只读取和解码一次；每个 token
// 保持与单行 GEMV 相同的 K 遍历和 wave32 归约顺序。
extern "C" __global__ void block_fp8_small_n_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    float* __restrict__ output,
    const unsigned int input_rows,
    const unsigned int input_columns,
    const unsigned int weight_rows,
    const unsigned int block_row_shift,
    const unsigned int block_col_shift)
{
    const unsigned int row = blockIdx.x;
    if (row >= weight_rows || input_rows == 0u || input_rows > 8u) return;
    const unsigned int lane = threadIdx.x;
    const unsigned int block_row_index = row >> block_row_shift;
    const unsigned int block_cols = 1u << block_col_shift;
    const unsigned int scale_cols = (input_columns + block_cols - 1u) >> block_col_shift;
    const unsigned char* code_row = codes + (unsigned long long)row * input_columns;
    float partial[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned int column = lane * 16u; column < input_columns; column += 32u * 16u) {
        const unsigned int block_col_index = column >> block_col_shift;
        const float scale = zllm_block_fp8_scale(scales[block_row_index * scale_cols + block_col_index]);
        const uint4 packed = *reinterpret_cast<const uint4*>(code_row + column);
        const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
        #pragma unroll
        for (unsigned int offset = 0; offset < 16u; ++offset) {
            const unsigned int col = column + offset;
            const unsigned char code = (unsigned char)(words[offset >> 2u] >> ((offset & 3u) * 8u));
            const float weight = zllm_block_fp8_e4m3_to_f32(code) * scale;
            #pragma unroll
            for (unsigned int token = 0; token < 8u; ++token) {
                if (token < input_rows) partial[token] += weight * input[(unsigned long long)token * input_columns + col];
            }
        }
    }
    #pragma unroll
    for (unsigned int token = 0; token < 8u; ++token) {
        if (token >= input_rows) break;
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) partial[token] += __shfl_down(partial[token], offset, 32u);
        if (lane == 0u) output[(unsigned long long)token * weight_rows + row] = partial[token];
    }
}

// 三份等宽 BF16 hidden 对应同一块按列拼接的权重。直接按原 K 顺序读取三份
// 输入并在一个 wave 内归约，避免先展开 F32 再做两次 concat。输出与
// concat([first, second, third]) 后的 BlockFP8 small-N GEMV 保持同一列顺序。
extern "C" __global__ void block_fp8_three_segment_bf16_f32(
    const unsigned short* __restrict__ first,
    const unsigned short* __restrict__ second,
    const unsigned short* __restrict__ third,
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    float* __restrict__ output,
    const unsigned int input_rows,
    const unsigned int segment_columns,
    const unsigned int weight_rows,
    const unsigned int block_row_shift,
    const unsigned int block_col_shift)
{
    const unsigned int row = blockIdx.x;
    if (row >= weight_rows || input_rows == 0u || input_rows > 8u) return;
    const unsigned int lane = threadIdx.x;
    const unsigned int input_columns = segment_columns * 3u;
    const unsigned int block_row_index = row >> block_row_shift;
    const unsigned int block_cols = 1u << block_col_shift;
    const unsigned int scale_cols = (input_columns + block_cols - 1u) >> block_col_shift;
    const unsigned char* code_row = codes + (unsigned long long)row * input_columns;
    for (unsigned int token = 0u; token < input_rows; ++token) {
        float partial = 0.0F;
        for (unsigned int column = lane * 16u; column < input_columns; column += 32u * 16u) {
            const unsigned int block_col_index = column >> block_col_shift;
            const float scale = zllm_block_fp8_scale(scales[block_row_index * scale_cols + block_col_index]);
            const uint4 packed = *reinterpret_cast<const uint4*>(code_row + column);
            const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
            #pragma unroll
            for (unsigned int offset = 0; offset < 16u; ++offset) {
                const unsigned int global_column = column + offset;
                const unsigned int segment = global_column / segment_columns;
                const unsigned int local_column = global_column - segment * segment_columns;
                const unsigned short* input = segment == 0u ? first : (segment == 1u ? second : third);
                const unsigned short bits = input[(unsigned long long)token * segment_columns + local_column];
                const float value = __uint_as_float((unsigned int)bits << 16u);
                const unsigned char code = (unsigned char)(words[offset >> 2u] >> ((offset & 3u) * 8u));
                partial += zllm_block_fp8_e4m3_to_f32(code) * scale * value;
            }
        }
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) partial += __shfl_down(partial, offset, 32u);
        if (lane == 0u) output[(unsigned long long)token * weight_rows + row] = partial;
    }
}

// 两个独立矩阵共享同一批输入时合并提交；x 维选择矩阵/行，y 维选择 token。
// 输出保持两个连续 token-major 矩阵，调用方可以零拷贝切成两个 view。
extern "C" __global__ void block_fp8_dual_gemv_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ first_codes,
    const unsigned char* __restrict__ first_scales,
    const unsigned char* __restrict__ second_codes,
    const unsigned char* __restrict__ second_scales,
    float* __restrict__ output,
    const unsigned int input_rows,
    const unsigned int input_columns,
    const unsigned int first_rows,
    const unsigned int second_rows,
    const unsigned int block_row_shift,
    const unsigned int block_col_shift)
{
    const unsigned int combined_row = blockIdx.x;
    const unsigned int token = blockIdx.y;
    if (combined_row >= first_rows + second_rows || token >= input_rows) return;
    const bool second = combined_row >= first_rows;
    const unsigned int row = second ? combined_row - first_rows : combined_row;
    const unsigned int matrix_rows = second ? second_rows : first_rows;
    const unsigned char* codes = second ? second_codes : first_codes;
    const unsigned char* scales = second ? second_scales : first_scales;
    const float* token_input = input + (unsigned long long)token * input_columns;
    const unsigned int lane = threadIdx.x;
    const unsigned int wavefront_size = (blockDim.x < 64u) ? blockDim.x : 64u;
    const float partial = zllm_block_fp8_dot_row_f32(token_input, codes, scales, input_columns, row, block_row_shift, block_col_shift, lane, wavefront_size);
    if (lane == 0u) {
        float* matrix_output = output + (second ? (unsigned long long)input_rows * first_rows : 0u);
        matrix_output[(unsigned long long)token * matrix_rows + row] = partial;
    }
}

// Gated linear 的 gate/up 行一一对应。同一 WG 连续计算同一 token 的两路 dot，
// 复用参数化 FP8 dot 与原归约树，lane 0 直接写激活后的结果。
extern "C" __global__ void block_fp8_gated_gemv_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate_codes,
    const unsigned char* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_codes,
    const unsigned char* __restrict__ up_scales,
    float* __restrict__ output,
    const unsigned int input_rows,
    const unsigned int input_columns,
    const unsigned int rows,
    const unsigned int block_row_shift,
    const unsigned int block_col_shift,
    const unsigned int activation,
    const float alpha,
    const float beta,
    const float limit,
    const float linear_beta,
    const unsigned int has_linear_beta)
{
    const unsigned int row = blockIdx.x;
    const unsigned int token = blockIdx.y;
    if (row >= rows || token >= input_rows) return;
    const float* token_input = input + (unsigned long long)token * input_columns;
    const unsigned int lane = threadIdx.x;
    const unsigned int wavefront_size = (blockDim.x < 64u) ? blockDim.x : 64u;
    float gate = zllm_block_fp8_dot_row_f32(token_input, gate_codes, gate_scales, input_columns, row, block_row_shift, block_col_shift, lane, wavefront_size);
    float up = zllm_block_fp8_dot_row_f32(token_input, up_codes, up_scales, input_columns, row, block_row_shift, block_col_shift, lane, wavefront_size);
    if (lane == 0u) {
        float value;
        if (activation == 0u) {
            value = gate / (1.0f + expf(-gate)) * up;
        } else if (activation == 1u) {
            gate = fminf(gate, limit);
            up = fmaxf(-limit, fminf(up, limit));
            value = gate / (1.0f + expf(-gate)) * up;
        } else if (activation == 2u) {
            if (has_linear_beta) up = linear_beta * tanhf(up / linear_beta);
            value = beta * tanhf(gate / beta) / (1.0f + expf(-gate)) * up;
        } else if (activation == 3u) {
            gate = fminf(gate, limit);
            up = fmaxf(-limit, fminf(up, limit));
            value = gate / (1.0f + expf(-alpha * gate)) * (up + 1.0f);
        } else {
            const float inner = 0.7978846f * (gate + 0.044715f * gate * gate * gate);
            value = 0.5f * gate * (1.0f + tanhf(inner)) * up;
        }
        output[(unsigned long long)token * rows + row] = value;
    }
}

// 输入按列等宽分组、权重按输出行连续分组。一个 WG 读取一次权重，
// 同时累计最多 8 个 token，输出保持 token-major `[N, groups * M]`。
extern "C" __global__ void block_fp8_grouped_columns_gemv_f32(
    const float* __restrict__ input,
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    float* __restrict__ output,
    const unsigned int input_rows,
    const unsigned int group_columns,
    const unsigned int rows_per_group,
    const unsigned int group_count,
    const unsigned int block_row_shift,
    const unsigned int block_col_shift)
{
    const unsigned int combined_row = blockIdx.x;
    const unsigned int total_rows = rows_per_group * group_count;
    if (combined_row >= total_rows || input_rows == 0u || input_rows > 8u) return;
    const unsigned int group = combined_row / rows_per_group;
    const unsigned int lane = threadIdx.x;
    const unsigned int wavefront_size = (blockDim.x < 32u) ? blockDim.x : 32u;
    // 逐 token 复用单行 dot helper:其 16 列/lane 完全展开形态已在 q_b/output_b
    // 实测 600+GB/s;旧版的动态 token 内循环让编译器收缩成窄循环(VGPR 24),
    // MLP 不足只剩 175GB/s。乘法与累加顺序与旧版逐位一致。
    for (unsigned int token = 0u; token < input_rows; ++token) {
        const float* input_row = input + (unsigned long long)token * group_count * group_columns + (unsigned long long)group * group_columns;
        const float partial = zllm_block_fp8_dot_row_f32(input_row, codes, scales, group_columns, combined_row, block_row_shift, block_col_shift, lane, wavefront_size);
        if (lane == 0u) output[(unsigned long long)token * total_rows + combined_row] = partial;
    }
}

// FP8 block 权重 → BF16:gfx11 无 FP8 计算单元,prefill 大 GEMM 先解码成
// BF16 走 WMMA GEMM;decode 结果由调用方按权重缓存,摊销到整个 prefill。
// f32→bf16 用标准 RNE 位技巧。
extern "C" __global__ void block_fp8_decode_bf16(
    const unsigned char* __restrict__ codes,   // [M, K] e4m3 行优先
    const unsigned char* __restrict__ scales,  // [M/BR, K/BC] e8m0
    unsigned short* __restrict__ output,       // [M, K] bf16 位模式
    const unsigned long long elements,         // M * K
    const unsigned int columns,                // K
    const unsigned int block_rows,             // BR
    const unsigned int block_cols)             // BC
{
    const unsigned long long index = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int row = (unsigned int)(index / columns);
    const unsigned int column = (unsigned int)(index - (unsigned long long)row * columns);
    const unsigned int scale_cols = (columns + block_cols - 1u) / block_cols;
    const unsigned char scale_byte = scales[(row / block_rows) * scale_cols + (column / block_cols)];
    const float value = zllm_block_fp8_e4m3_to_f32(codes[index]) * zllm_block_fp8_scale(scale_byte);
    const unsigned int bits = __float_as_uint(value);
    const unsigned int rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    output[index] = (unsigned short)(rounded >> 16);
}
