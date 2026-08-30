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
    const unsigned int block_bytes = tensor_type == 12 ? 144 : 176;
    const unsigned long long row_bytes = (unsigned long long)blocks_per_row * block_bytes;
    const unsigned char *weight_row = weight + (unsigned long long)output_row * row_bytes;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned int column = lane; column < columns; column += 256) {
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
    return tensor_type == 12 ? 144u : (tensor_type == 13 ? 176u : 210u);
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
