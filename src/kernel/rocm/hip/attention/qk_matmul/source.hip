__device__ __forceinline__ float gguf_f16(const unsigned char *bytes)
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

__device__ __forceinline__ float qk_input(const void *__restrict__ input, unsigned long long offset, unsigned int is_bf16)
{
    if (is_bf16) {
        const unsigned short bits = ((const unsigned short *)input)[offset];
        return __uint_as_float((unsigned int)bits << 16);
    }
    return ((const float *)input)[offset];
}

extern "C" __global__ void qk_matmul_f32(
    const void *__restrict__ input,
    const unsigned char *__restrict__ weight,
    float *__restrict__ output,
    const unsigned int input_rows,
    const unsigned int columns,
    const unsigned int output_rows,
    const unsigned int tensor_type,
    const unsigned int input_is_bf16)
{
    const unsigned int output_row = blockIdx.x;
    const unsigned int input_base = blockIdx.y * 8;
    const unsigned int lane = threadIdx.x;
    if (output_row >= output_rows) return;

    const unsigned int blocks_per_row = columns >> 8;
    const unsigned int block_bytes = zllm_gguf_block_bytes(tensor_type);
    const unsigned long long row_bytes = (unsigned long long)blocks_per_row * block_bytes;
    const unsigned char *weight_row = weight + (unsigned long long)output_row * row_bytes;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned int column = lane; column < columns; column += 256) {
        if (tensor_type == 11u || tensor_type == 21u || tensor_type == 23u) {
            const float value = zllm_gguf_extra_value(weight_row, tensor_type, column);
            #pragma unroll
            for (unsigned int row = 0; row < 8; ++row) {
                if (input_base + row < input_rows)
                    sums[row] += qk_input(input, (unsigned long long)(input_base + row) * columns + column, input_is_bf16) * value;
            }
            continue;
        }
        const unsigned int local = column & 255;
        const unsigned int group = local >> 5;
        const unsigned int index = local & 31;
        const unsigned char *block = weight_row + (unsigned long long)(column >> 8) * block_bytes;
        const unsigned char *scales = block + 4;
        unsigned int scale;
        unsigned int minimum;
        if (group < 4) {
            scale = scales[group] & 0x3f;
            minimum = scales[group + 4] & 0x3f;
        } else {
            scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
            minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
        }
        const unsigned int low_offset = tensor_type == 12 ? 16 : 48;
        const unsigned char packed = block[low_offset + (group >> 1) * 32 + index];
        unsigned int quant = (group & 1) == 0 ? packed & 0x0f : packed >> 4;
        if (tensor_type == 13 && (block[16 + index] & (1u << group)) != 0) quant += 16;
        const float d = gguf_f16(block);
        const float dmin = gguf_f16(block + 2);
        const float value = d * float(scale * quant) - dmin * float(minimum);
        #pragma unroll
        for (unsigned int row = 0; row < 8; ++row) {
            if (input_base + row < input_rows) {
                sums[row] += qk_input(input, (unsigned long long)(input_base + row) * columns + column, input_is_bf16) * value;
            }
        }
    }

    __shared__ float partial[8][256];
    #pragma unroll
    for (unsigned int row = 0; row < 8; ++row) partial[row][lane] = sums[row];
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            #pragma unroll
            for (unsigned int row = 0; row < 8; ++row) partial[row][lane] += partial[row][lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int row = 0; row < 8; ++row) {
            if (input_base + row < input_rows) {
                output[(unsigned long long)(input_base + row) * output_rows + output_row] = partial[row][0];
            }
        }
    }
}

// Q6_K(210B/256 元素块)：低 4bit 在 ql[128]、高 2bit 在 qh[64]、每组 6bit 有符号 scale 在 sc[16]、
// 块尾 f16 d。元素 local 的解码与 codec::ggml::decode_q6_k 逐位一致。
extern "C" __global__ void q6_matmul_f32(
    const void *__restrict__ input,
    const unsigned char *__restrict__ weight,
    float *__restrict__ output,
    const unsigned int input_rows,
    const unsigned int columns,
    const unsigned int output_rows,
    const unsigned int tensor_type,
    const unsigned int input_is_bf16)
{
    // tensor_type 参数仅为与 qk_matmul_f32 的参数布局对齐(统一 8 参)，此处不使用。
    (void)tensor_type;
    const unsigned int output_row = blockIdx.x;
    const unsigned int input_base = blockIdx.y * 8;
    const unsigned int lane = threadIdx.x;
    if (output_row >= output_rows) return;

    const unsigned int blocks_per_row = columns >> 8;
    const unsigned char *weight_row = weight + (unsigned long long)output_row * blocks_per_row * 210;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned int column = lane; column < columns; column += 256) {
        const unsigned int local = column & 255;
        const unsigned int half = local >> 7;
        const unsigned int remaining = local & 127;
        const unsigned int index = remaining & 31;
        const unsigned int slot = remaining >> 5;
        const unsigned char *block = weight_row + (unsigned long long)(column >> 8) * 210;
        const unsigned char low = block[half * 64 + index + (slot & 1) * 32];
        const unsigned int nibble = (slot < 2) ? (low & 0x0fu) : (low >> 4);
        const unsigned int high = (block[128 + half * 32 + index] >> (slot * 2)) & 3u;
        const int quant = (int)(nibble | (high << 4)) - 32;
        const signed char group_scale = (signed char)block[192 + half * 8 + (index >> 4) + slot * 2];
        const float value = gguf_f16(block + 208) * (float)group_scale * (float)quant;
        #pragma unroll
        for (unsigned int row = 0; row < 8; ++row) {
            if (input_base + row < input_rows) {
                sums[row] += qk_input(input, (unsigned long long)(input_base + row) * columns + column, input_is_bf16) * value;
            }
        }
    }

    __shared__ float q6_partial[8][256];
    #pragma unroll
    for (unsigned int row = 0; row < 8; ++row) q6_partial[row][lane] = sums[row];
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            #pragma unroll
            for (unsigned int row = 0; row < 8; ++row) q6_partial[row][lane] += q6_partial[row][lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int row = 0; row < 8; ++row) {
            if (input_base + row < input_rows) {
                output[(unsigned long long)(input_base + row) * output_rows + output_row] = q6_partial[row][0];
            }
        }
    }
}

// ===== GGUF grouped decode experts(与 CT grouped 对应的 GGUF 形态) =====
// meta 与 host 侧 GgufGroupedExpertMeta(repr(C)) 布局一致。
struct GgufExpertMeta {
    unsigned long long gate;
    unsigned long long up;
    unsigned long long down;
    unsigned int gate_type;
    unsigned int up_type;
    unsigned int down_type;
};

__device__ __forceinline__ unsigned int qk_block_bytes(const unsigned int tensor_type)
{
    return zllm_gguf_block_bytes(tensor_type);
}

// 行基址：weight 为 [rows, cols] 的 K-quant 打包，每行 (cols>>8) 个块。
__device__ __forceinline__ const unsigned char *gguf_row_base(const unsigned long long weight, const unsigned int tensor_type, const unsigned int row, const unsigned int cols)
{
    return (const unsigned char *)weight + (unsigned long long)row * (cols >> 8) * qk_block_bytes(tensor_type);
}

// 与 codec::ggml 的 decode_q4_k/decode_q5_k/decode_q6_k 逐位一致。
__device__ __forceinline__ float gguf_kq_dequant(const unsigned char *row_base, const unsigned int tensor_type, const unsigned int column)
{
    const unsigned char *block = row_base + (unsigned long long)(column >> 8) * qk_block_bytes(tensor_type);
    const unsigned int local = column & 255;
    // Q8_0：[f16 scale][32×i8] 交替 block，每 32 元素 34B。
    if (tensor_type == 8u) {
        const unsigned int block_idx = local >> 5;
        const unsigned int offset = local & 31u;
        const unsigned char *q8_block = block + (unsigned long long)block_idx * 34u;
        return gguf_f16(q8_block) * (float)(signed char)q8_block[2 + offset];
    }
    if (tensor_type == 11u || tensor_type == 21u || tensor_type == 23u)
        return zllm_gguf_extra_value(row_base, tensor_type, column);
    if (tensor_type == 14) {
        const unsigned int half = local >> 7;
        const unsigned int remaining = local & 127;
        const unsigned int index = remaining & 31;
        const unsigned int slot = remaining >> 5;
        const unsigned char low = block[half * 64 + index + (slot & 1) * 32];
        const unsigned int nibble = (slot < 2) ? (low & 0x0fu) : (low >> 4);
        const unsigned int high = (block[128 + half * 32 + index] >> (slot * 2)) & 3u;
        const int quant = (int)(nibble | (high << 4)) - 32;
        const signed char group_scale = (signed char)block[192 + half * 8 + (index >> 4) + slot * 2];
        return gguf_f16(block + 208) * (float)group_scale * (float)quant;
    }
    const unsigned int group = local >> 5;
    const unsigned int index = local & 31;
    const unsigned char *scales = block + 4;
    unsigned int scale;
    unsigned int minimum;
    if (group < 4) {
        scale = scales[group] & 0x3f;
        minimum = scales[group + 4] & 0x3f;
    } else {
        scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
    const unsigned int low_offset = tensor_type == 12 ? 16 : 48;
    const unsigned char packed = block[low_offset + (group >> 1) * 32 + index];
    unsigned int quant = (group & 1) == 0 ? packed & 0x0f : packed >> 4;
    if (tensor_type == 13 && (block[16 + index] & (1u << group)) != 0) quant += 16;
    return gguf_f16(block) * float(scale * quant) - gguf_f16(block + 2) * float(minimum);
}

// ===== decode 少行 fdot2 快路径：与 CT W8 家族对齐的数值形态 =====
typedef __bf16 __attribute__((ext_vector_type(2))) qk_bf16x2;

__device__ __forceinline__ unsigned int qk_bf16_pair(const float low, const float high)
{
    return ((__float_as_uint(high) >> 16) << 16) | (__float_as_uint(low) >> 16);
}

// 与 qk_matmul_f32 相同的 GGUF 块解码(gguf_kq_dequant)，但反量化值经 bf16 舍入后
// 用 fdot2 累加，每个 workgroup 处理一个输出行、最多 8 个输入行（decode 主路径）。
// 数值与 CT W8 GEMV 同族（bf16 舍入 + fdot2），跨后端容差沿用 1e-2。
extern "C" __global__ void qk_matmul_fdot2(
    const void *__restrict__ input,
    const unsigned char *__restrict__ weight,
    float *__restrict__ output,
    const unsigned int input_rows,
    const unsigned int columns,
    const unsigned int output_rows,
    const unsigned int tensor_type,
    const unsigned int input_is_bf16)
{
    const unsigned int output_row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (output_row >= output_rows || input_rows == 0u || input_rows > 8u) return;

    const unsigned int blocks_per_row = columns >> 8;
    const unsigned int block_bytes = zllm_gguf_block_bytes(tensor_type);
    const unsigned char *weight_row = weight + (unsigned long long)output_row * blocks_per_row * block_bytes;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // 列成对分配：lane 处理 (2*lane, 2*lane+1)，步进 512；columns 为 256 倍数，无尾列。
    for (unsigned int pair = lane; pair * 2u + 1u < columns; pair += 256u) {
        const unsigned int c0 = pair * 2u;
        const unsigned int weight_pair = qk_bf16_pair(
            gguf_kq_dequant(weight_row, tensor_type, c0),
            gguf_kq_dequant(weight_row, tensor_type, c0 + 1u));
        const qk_bf16x2 weight_value = *reinterpret_cast<const qk_bf16x2 *>(&weight_pair);
        #pragma unroll
        for (unsigned int row = 0; row < 8; ++row) {
            if (row < input_rows) {
                const unsigned int input_pair = qk_bf16_pair(
                    qk_input(input, (unsigned long long)row * columns + c0, input_is_bf16),
                    qk_input(input, (unsigned long long)row * columns + c0 + 1u, input_is_bf16));
                const qk_bf16x2 input_value = *reinterpret_cast<const qk_bf16x2 *>(&input_pair);
                sums[row] = __builtin_amdgcn_fdot2_f32_bf16(input_value, weight_value, sums[row], false);
            }
        }
    }

    __shared__ float fdot2_partial[8][256];
    #pragma unroll
    for (unsigned int row = 0; row < 8; ++row) fdot2_partial[row][lane] = sums[row];
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            #pragma unroll
            for (unsigned int row = 0; row < 8; ++row) fdot2_partial[row][lane] += fdot2_partial[row][lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int row = 0; row < 8; ++row) {
            if (row < input_rows) {
                output[(unsigned long long)row * output_rows + output_row] = fdot2_partial[row][0];
            }
        }
    }
}


// ===== Q8_0 预重排 decode GEMV：scales/quants 平面分离 =====
// 布局：每行 [cols/32 个 f16 scale（小端）][cols 个 i8 quant]，row_stride =
// cols/16 + cols（cols 为 256 倍数 → quants 平面 8B 对齐）。数值形态与
// qk_matmul_fdot2 相同：f16 scale × i8 → f32 → bf16 舍入 → fdot2 累加，
// 仅 lane 内列分配从跨步成对改为 8 连续列（归约顺序差异沿用 bf16 容差族）。
// 每 lane 一次 8B quant 加载 + 每 32 列一次 scale 广播，替代逐字节标量加载。
__device__ __forceinline__ qk_bf16x2 q8_pair(const float scale, const unsigned int word, const unsigned int byte)
{
    const int low = (int)(word << (24u - byte * 8u)) >> 24;
    const int high = (int)(word << (16u - byte * 8u)) >> 24;
    const unsigned int pair = qk_bf16_pair(scale * (float)low, scale * (float)high);
    return *reinterpret_cast<const qk_bf16x2 *>(&pair);
}

__device__ __forceinline__ qk_bf16x2 q8_input_pair(const void *__restrict__ input, const unsigned long long offset, const unsigned int is_bf16)
{
    const unsigned int pair = qk_bf16_pair(qk_input(input, offset, is_bf16), qk_input(input, offset + 1u, is_bf16));
    return *reinterpret_cast<const qk_bf16x2 *>(&pair);
}

extern "C" __global__ void q8_gemv_repacked(
    const void *__restrict__ input,
    const unsigned char *__restrict__ weight,
    float *__restrict__ output,
    const unsigned int input_rows,
    const unsigned int columns,
    const unsigned int output_rows,
    const unsigned int input_is_bf16)
{
    const unsigned int output_row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (output_row >= output_rows || input_rows == 0u || input_rows > 8u) return;

    const unsigned int scale_bytes = columns >> 4;
    const unsigned char *row_base = weight + (unsigned long long)output_row * ((unsigned long long)scale_bytes + columns);
    const unsigned char *row_scales = row_base;
    const signed char *row_quants = (const signed char *)(row_base + scale_bytes);
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned int col0 = lane * 8u; col0 < columns; col0 += 2048u) {
        const int2 packed = *(const int2 *)(row_quants + col0);
        const float scale = gguf_f16(row_scales + ((col0 >> 5) << 1));
        const qk_bf16x2 wa = q8_pair(scale, packed.x, 0u);
        const qk_bf16x2 wb = q8_pair(scale, packed.x, 2u);
        const qk_bf16x2 wc = q8_pair(scale, packed.y, 0u);
        const qk_bf16x2 wd = q8_pair(scale, packed.y, 2u);
        #pragma unroll
        for (unsigned int row = 0; row < 8; ++row) {
            if (row < input_rows) {
                const unsigned long long base = (unsigned long long)row * columns + col0;
                float acc = sums[row];
                acc = __builtin_amdgcn_fdot2_f32_bf16(q8_input_pair(input, base + 0u, input_is_bf16), wa, acc, false);
                acc = __builtin_amdgcn_fdot2_f32_bf16(q8_input_pair(input, base + 2u, input_is_bf16), wb, acc, false);
                acc = __builtin_amdgcn_fdot2_f32_bf16(q8_input_pair(input, base + 4u, input_is_bf16), wc, acc, false);
                acc = __builtin_amdgcn_fdot2_f32_bf16(q8_input_pair(input, base + 6u, input_is_bf16), wd, acc, false);
                sums[row] = acc;
            }
        }
    }

    __shared__ float q8_partial[8][256];
    #pragma unroll
    for (unsigned int row = 0; row < 8; ++row) q8_partial[row][lane] = sums[row];
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            #pragma unroll
            for (unsigned int row = 0; row < 8; ++row) q8_partial[row][lane] += q8_partial[row][lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int row = 0; row < 8; ++row) {
            if (row < input_rows) {
                output[(unsigned long long)row * output_rows + output_row] = q8_partial[row][0];
            }
        }
    }
}
