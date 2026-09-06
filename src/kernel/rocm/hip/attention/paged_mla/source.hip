#include <rocwmma/rocwmma.hpp>
__device__ __forceinline__ float paged_weight(
    const unsigned int* packed,
    const unsigned char* scales,
    unsigned int row,
    unsigned int column,
    unsigned int columns,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits)
{
    if (bits == 32) {
        return ((const float*)packed)[(unsigned long long)row * columns + column];
    }
    if (bits == 16) {
        return zllm_bf16_to_f32(
            ((const unsigned short*)packed)[(unsigned long long)row * columns + column]);
    }
    const unsigned int values_per_word = 32u / bits;
    const unsigned long long packed_columns = ((unsigned long long)columns + values_per_word - 1) / values_per_word;
    const unsigned int code = (packed[(unsigned long long)row * packed_columns + column / values_per_word]
        >> ((column % values_per_word) * bits)) & ((1u << bits) - 1u);
    const int signed_code = bits == 4 ? (int)code - 8 : (int)code - 128;
    const unsigned long long groups = columns / group_size;
    return (float)signed_code * zllm_scale(scales, (unsigned long long)row * groups + column / group_size, scale_dtype);
}

__device__ __forceinline__ float paged_weight_w8g128(
    const unsigned int* packed,
    const unsigned char* scales,
    unsigned int row,
    unsigned int column,
    unsigned int columns,
    unsigned int scale_dtype)
{
    const unsigned long long packed_columns
        = ((unsigned long long)columns + 3u) / 4u;
    const unsigned int word
        = packed[(unsigned long long)row * packed_columns + column / 4u];
    const int code = (int)((word >> ((column & 3u) * 8u)) & 255u) - 128;
    const unsigned long long groups = columns / 128u;
    return (float)code * zllm_scale(
        scales, (unsigned long long)row * groups + (column >> 7u), scale_dtype);
}

__device__ __forceinline__ unsigned long long paged_row(
    const unsigned int* block_table,
    unsigned int token,
    unsigned int block_size)
{
    return (unsigned long long)block_table[token / block_size] * block_size + token % block_size;
}

__device__ __forceinline__ bool paged_shard_owns(
    unsigned int token,
    unsigned int block_size,
    unsigned int shard_parity)
{
    return shard_parity > 1u || ((token / block_size) & 1u) == shard_parity;
}

__device__ __forceinline__ unsigned long long paged_shard_row(
    const unsigned int* block_table,
    unsigned int token,
    unsigned int block_size,
    unsigned int shard_parity)
{
    if (shard_parity > 1u) return paged_row(block_table, token, block_size);
    const unsigned int block = token / block_size;
    // dense shard 路径会在 score mask 前预取 physical row；非本卡 block
    // 必须先定向安全行 0，否则尾部 block 会越过 compact buffer。
    if ((block & 1u) != shard_parity) return 0;
    const unsigned int local = (block >> 1u) * block_size + token % block_size;
    return paged_row(block_table, local, block_size);
}

__device__ __forceinline__ unsigned int paged_shard_global_token(
    unsigned int local,
    unsigned int block_size,
    unsigned int shard_parity)
{
    if (shard_parity > 1u) return local;
    const unsigned int block = local / block_size;
    return (block * 2u + shard_parity) * block_size + local % block_size;
}

// DSA 先产生一张全局 top-k；这里只按固定 block parity 拆表，
// 不在两卡分别做 top-k，因此候选精度与单卡路径一致。
extern "C" __global__ void mla_split_selection_parity(
    const unsigned int* selection,
    unsigned int* owner,
    unsigned int* peer,
    unsigned int* owner_counts,
    unsigned int* peer_counts,
    unsigned int rows,
    unsigned int width,
    unsigned int block_size)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || block_size == 0) return;
    // 每个线程负责连续列区间，lane 0 对各区间做独占前缀和；这样输出
    // 严格保持 global selection 的 token 顺序，不让跨 wave atomic 顺序
    // 改变 softmax 归约次序。width=2048 时每线程只顺序处理 8 项。
    __shared__ unsigned int owner_offsets[256];
    __shared__ unsigned int peer_offsets[256];
    const unsigned int begin = (unsigned int)(((unsigned long long)width * lane) / blockDim.x);
    const unsigned int end = (unsigned int)(((unsigned long long)width * (lane + 1u)) / blockDim.x);
    unsigned int owner_count = 0u;
    unsigned int peer_count = 0u;
    for (unsigned int column = begin; column < end; ++column) {
        const unsigned int token = selection[(unsigned long long)row * width + column];
        if (((token / block_size) & 1u) == 0u) ++owner_count;
        else ++peer_count;
    }
    owner_offsets[lane] = owner_count;
    peer_offsets[lane] = peer_count;
    __syncthreads();
    if (lane == 0) {
        unsigned int owner_total = 0u;
        unsigned int peer_total = 0u;
        for (unsigned int thread = 0; thread < blockDim.x; ++thread) {
            const unsigned int thread_owner = owner_offsets[thread];
            const unsigned int thread_peer = peer_offsets[thread];
            owner_offsets[thread] = owner_total;
            peer_offsets[thread] = peer_total;
            owner_total += thread_owner;
            peer_total += thread_peer;
        }
        owner_counts[row] = owner_total;
        peer_counts[row] = peer_total;
    }
    __syncthreads();
    unsigned int owner_target = owner_offsets[lane];
    unsigned int peer_target = peer_offsets[lane];
    for (unsigned int column = begin; column < end; ++column) {
        const unsigned int token = selection[(unsigned long long)row * width + column];
        const unsigned int parity = (token / block_size) & 1u;
        if (parity == 0) owner[(unsigned long long)row * width + owner_target++] = token;
        else peer[(unsigned long long)row * width + peer_target++] = token;
    }
}

extern "C" __global__ void paged_cache_append_f32_bf16(
    const float* input,
    unsigned short* cache,
    const unsigned int* block_table,
    unsigned int position,
    unsigned int rows,
    unsigned int columns,
    unsigned int block_size)
{
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long elements = (unsigned long long)rows * columns;
    if (index >= elements) return;
    const unsigned int row = index / columns;
    const unsigned int column = index % columns;
    const unsigned long long physical = paged_row(block_table, position + row, block_size);
    cache[physical * columns + column] = zllm_f32_to_bf16(input[index]);
}

extern "C" __global__ void paged_cache_append_f32_q8(
    const float* input,
    signed char* cache,
    unsigned short* scales,
    const unsigned int* block_table,
    unsigned int position,
    unsigned int rows,
    unsigned int columns,
    unsigned int group_size,
    unsigned int block_size)
{
    const unsigned int groups_per_row = columns / group_size;
    const unsigned int group_index = blockIdx.x;
    const unsigned int row = group_index / groups_per_row;
    const unsigned int group = group_index - row * groups_per_row;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= group_size) return;
    const unsigned int column = group * group_size + lane;
    const float value = input[(unsigned long long)row * columns + column];
    __shared__ float maxima[256];
    __shared__ float quant_scale;
    maxima[lane] = fabsf(value);
    __syncthreads();
    for (unsigned int offset = group_size >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) maxima[lane] = fmaxf(maxima[lane], maxima[lane + offset]);
        __syncthreads();
    }
    const unsigned long long physical = paged_row(block_table, position + row, block_size);
    if (lane == 0) {
        const float scale = maxima[0] == 0.0f ? 1.0f : maxima[0] * (1.0f / 127.0f);
        const unsigned short encoded = zllm_f32_to_bf16(scale);
        scales[physical * groups_per_row + group] = encoded;
        quant_scale = zllm_bf16_to_f32(encoded);
    }
    __syncthreads();
    const float quantized = nearbyintf(value / quant_scale);
    cache[physical * columns + column]
        = (signed char)fmaxf(-127.0f, fminf(127.0f, quantized));
}

// decode indexer 单行 prologue：保持原 256-thread LayerNorm 归约、RoPE
// 算术和 Q8 tree max 的顺序，直接写 raw-Q8 cache，避免两个中间 tensor。
extern "C" __global__ void paged_dsa_append_layernorm_rope_q8(
    const float* input,
    const float* weight,
    const float* bias,
    const float* cosine,
    const float* sine,
    signed char* cache,
    unsigned short* scales,
    const unsigned int* block_table,
    unsigned int cache_position,
    unsigned int rope_position,
    unsigned int rows,
    unsigned int columns,
    unsigned int rotary_dim,
    unsigned int split_half,
    unsigned int block_size,
    float eps)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows) return;
    const unsigned long long base = (unsigned long long)row * columns;
    float sum = 0.0f;
    float square = 0.0f;
    if (lane < columns) {
        const float value = input[base + lane];
        sum = value;
        square = value * value;
    }
    __shared__ float partial_sum[256];
    __shared__ float partial_square[256];
    __shared__ float normalized[256];
    __shared__ float values[256];
    __shared__ float maxima[256];
    __shared__ float quant_scale;
    partial_sum[lane] = sum;
    partial_square[lane] = square;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial_sum[lane] += partial_sum[lane + stride];
            partial_square[lane] += partial_square[lane + stride];
        }
        __syncthreads();
    }
    const float mean = partial_sum[0] / (float)columns;
    const float variance = fmaxf(partial_square[0] / (float)columns - mean * mean, 0.0f);
    const float inverse = rsqrtf(variance + eps);
    if (lane < columns) {
        normalized[lane] = (input[base + lane] - mean) * inverse * weight[lane] + bias[lane];
    }
    __syncthreads();

    if (lane < columns) {
        float value = normalized[lane];
        if (lane < rotary_dim) {
            const unsigned int half = rotary_dim >> 1u;
            const unsigned int pair = split_half ? lane % half : lane >> 1u;
            const unsigned int real = split_half ? pair : pair * 2u;
            const unsigned int imaginary = split_half ? half + pair : real + 1u;
            const float real_value = normalized[real];
            const float imaginary_value = normalized[imaginary];
            const float c = cosine[(unsigned long long)(rope_position + row) * half + pair];
            const float s = sine[(unsigned long long)(rope_position + row) * half + pair];
            value = lane == real ? real_value * c - imaginary_value * s
                                 : imaginary_value * c + real_value * s;
        }
        values[lane] = value;
        maxima[lane] = fabsf(value);
    } else {
        maxima[lane] = 0.0f;
    }
    __syncthreads();
    for (unsigned int offset = columns >> 1u; offset > 0; offset >>= 1u) {
        if (lane < offset) maxima[lane] = fmaxf(maxima[lane], maxima[lane + offset]);
        __syncthreads();
    }
    const unsigned long long physical = paged_row(block_table, cache_position + row, block_size);
    if (lane == 0) {
        const float scale = maxima[0] == 0.0f ? 1.0f : maxima[0] * (1.0f / 127.0f);
        const unsigned short encoded = zllm_f32_to_bf16(scale);
        scales[physical] = encoded;
        quant_scale = zllm_bf16_to_f32(encoded);
    }
    __syncthreads();
    if (lane < columns) {
        const float quantized = nearbyintf(values[lane] / quant_scale);
        cache[physical * columns + lane]
            = (signed char)fmaxf(-127.0f, fminf(127.0f, quantized));
    }
}

__device__ __forceinline__ void zllm_dsa_hadamard(float* values, unsigned int size)
{
    const unsigned int lane = threadIdx.x;
    for (unsigned int stride = 1u; stride < size; stride <<= 1u) {
        if (lane < (size >> 1u)) {
            const unsigned int base = (lane / stride) * stride * 2u + lane % stride;
            const float a = values[base];
            const float b = values[base + stride];
            values[base] = a + b;
            values[base + stride] = a - b;
        }
        __syncthreads();
    }
}

// DSA key 的量化约定：LayerNorm+RoPE 后对完整 head_dim 做归一化 Hadamard，
// 再用单行 scale 写 Q8。query 侧必须使用同一个变换。
extern "C" __global__ void paged_cache_append_f32_q8_hadamard(
    const float* input,
    signed char* cache,
    unsigned short* scales,
    const unsigned int* block_table,
    unsigned int position,
    unsigned int rows,
    unsigned int columns,
    unsigned int group_size,
    unsigned int block_size)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= columns) return;
    __shared__ float values[256];
    __shared__ float maxima[256];
    __shared__ float quant_scale;
    values[lane] = input[(unsigned long long)row * columns + lane];
    __syncthreads();
    zllm_dsa_hadamard(values, columns);
    const float value = values[lane] * rsqrtf((float)columns);
    maxima[lane] = fabsf(value);
    __syncthreads();
    for (unsigned int offset = columns >> 1u; offset; offset >>= 1u) {
        if (lane < offset) maxima[lane] = fmaxf(maxima[lane], maxima[lane + offset]);
        __syncthreads();
    }
    const unsigned long long physical = paged_row(block_table, position + row, block_size);
    if (lane == 0u) {
        const float scale = maxima[0] == 0.0f ? 1.0f : maxima[0] * (1.0f / 127.0f);
        const unsigned short encoded = zllm_f32_to_bf16(scale);
        scales[physical] = encoded;
        quant_scale = zllm_bf16_to_f32(encoded);
    }
    __syncthreads();
    const float quantized = nearbyintf(value / quant_scale);
    cache[physical * columns + lane]
        = (signed char)fmaxf(-127.0f, fminf(127.0f, quantized));
    (void)group_size;
}

// shadow 只从生产 raw-Q8 cache 构造低精度粗排 cache。先按生产 WMMA 的
// BF16 operand 约定还原 K，再做 Hadamard，避免把原始 F32 与 cache 误差混在一起。
extern "C" __global__ void paged_cache_transform_q8_hadamard(
    const signed char* source,
    const unsigned short* source_scales,
    signed char* target,
    unsigned short* target_scales,
    const unsigned int* block_table,
    unsigned int position,
    unsigned int rows,
    unsigned int columns,
    unsigned int block_size)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= columns) return;
    const unsigned long long physical = paged_row(block_table, position + row, block_size);
    __shared__ float values[256];
    __shared__ float maxima[256];
    __shared__ float quant_scale;
    const float raw_scale = zllm_bf16_to_f32(source_scales[physical]);
    values[lane] = zllm_bf16_to_f32(zllm_f32_to_bf16(
        (float)source[physical * columns + lane] * raw_scale));
    __syncthreads();
    zllm_dsa_hadamard(values, columns);
    const float value = values[lane] * rsqrtf((float)columns);
    maxima[lane] = fabsf(value);
    __syncthreads();
    for (unsigned int offset = columns >> 1u; offset; offset >>= 1u) {
        if (lane < offset) maxima[lane] = fmaxf(maxima[lane], maxima[lane + offset]);
        __syncthreads();
    }
    if (lane == 0u) {
        const float scale = maxima[0] == 0.0f ? 1.0f : maxima[0] * (1.0f / 127.0f);
        const unsigned short encoded = zllm_f32_to_bf16(scale);
        target_scales[physical] = encoded;
        quant_scale = zllm_bf16_to_f32(encoded);
    }
    __syncthreads();
    const float code = nearbyintf(value / quant_scale);
    target[physical * columns + lane]
        = (signed char)fmaxf(-127.0f, fminf(127.0f, code));
}

// HISA profile shadow：直接从生产 raw-Q8 key 计算 block mean，再量化成
// Q8G summary。summary 只服务粗筛，原始 cache 与生产 selection 均不改变。
extern "C" __global__ void dsa_mean_pool_q8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    signed char* pooled_keys,
    unsigned short* pooled_scales,
    const unsigned int* pool_block_table,
    unsigned int first_pool,
    unsigned int pool_rows,
    unsigned int context_rows,
    unsigned int columns,
    unsigned int group_size,
    unsigned int pool_size,
    unsigned int block_size)
{
    const unsigned int groups_per_row = columns / group_size;
    const unsigned int local_pool = blockIdx.x / groups_per_row;
    const unsigned int group = blockIdx.x - local_pool * groups_per_row;
    const unsigned int lane = threadIdx.x;
    if (local_pool >= pool_rows || lane >= group_size || pool_size == 0u) return;
    const unsigned int pool = first_pool + local_pool;
    const unsigned int first_token = pool * pool_size;
    if (first_token >= context_rows) return;
    const unsigned int valid = min(pool_size, context_rows - first_token);
    const unsigned int column = group * group_size + lane;
    float total = 0.0f;
    for (unsigned int slot = 0u; slot < valid; ++slot) {
        const unsigned int token = first_token + slot;
        const unsigned long long physical = paged_row(block_table, token, block_size);
        const float scale = zllm_bf16_to_f32(
            key_scales[physical * groups_per_row + group]);
        total += (float)keys[physical * columns + column] * scale;
    }
    const float pooled = total / (float)valid;
    __shared__ float maxima[256];
    __shared__ float quant_scale;
    maxima[lane] = fabsf(pooled);
    __syncthreads();
    for (unsigned int offset = group_size >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) maxima[lane] = fmaxf(maxima[lane], maxima[lane + offset]);
        __syncthreads();
    }
    const unsigned long long pooled_physical = paged_row(pool_block_table, pool, block_size);
    if (lane == 0u) {
        const float scale = maxima[0] == 0.0f ? 1.0f : maxima[0] * (1.0f / 127.0f);
        const unsigned short encoded = zllm_f32_to_bf16(scale);
        pooled_scales[pooled_physical * groups_per_row + group] = encoded;
        quant_scale = zllm_bf16_to_f32(encoded);
    }
    __syncthreads();
    const float quantized = nearbyintf(pooled / quant_scale);
    pooled_keys[pooled_physical * columns + column]
        = (signed char)fmaxf(-127.0f, fminf(127.0f, quantized));
}

// exact-pruning profile shadow：保存每个 token block、每个 key 维度在生产
// BF16 WMMA 输入域中的 min/max。F32 summary 避免端点量化向内收缩。
extern "C" __global__ void dsa_interval_pool_q8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    float* lower,
    float* upper,
    unsigned int first_pool,
    unsigned int pool_rows,
    unsigned int context_rows,
    unsigned int columns,
    unsigned int group_size,
    unsigned int pool_size,
    unsigned int block_size)
{
    const unsigned int groups_per_row = columns / group_size;
    const unsigned int local_pool = blockIdx.x / groups_per_row;
    const unsigned int group = blockIdx.x - local_pool * groups_per_row;
    const unsigned int lane = threadIdx.x;
    if (local_pool >= pool_rows || lane >= group_size || pool_size == 0u) return;
    const unsigned int pool = first_pool + local_pool;
    const unsigned int first_token = pool * pool_size;
    if (first_token >= context_rows) return;
    const unsigned int valid = min(pool_size, context_rows - first_token);
    const unsigned int column = group * group_size + lane;
    float minimum = 3.402823466e+38F;
    float maximum = -3.402823466e+38F;
    for (unsigned int slot = 0u; slot < valid; ++slot) {
        const unsigned int token = first_token + slot;
        const unsigned long long physical = paged_row(block_table, token, block_size);
        const float scale = zllm_bf16_to_f32(
            key_scales[physical * groups_per_row + group]);
        const float value = zllm_bf16_to_f32(zllm_f32_to_bf16(
            (float)keys[physical * columns + column] * scale));
        minimum = fminf(minimum, value);
        maximum = fmaxf(maximum, value);
    }
    const unsigned long long output = (unsigned long long)pool * columns + column;
    lower[output] = minimum;
    upper[output] = maximum;
}

// 根据 axis-aligned block interval 计算 DSA score 上界。逐 head 的 ReLU
// 与 signed head weight 都按区间方向处理；误差项覆盖 BF16 WMMA/F32 归约。
extern "C" __global__ void dsa_interval_score_bounds(
    const float* lower,
    const float* upper,
    const float* query,
    const float* head_weights,
    float* bounds,
    unsigned int pool_rows,
    unsigned int head_count,
    unsigned int head_dim)
{
    const unsigned int pool = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (pool >= pool_rows || lane >= head_dim) return;
    __shared__ float lows[256];
    __shared__ float highs[256];
    __shared__ float magnitudes[256];
    float score_upper = 0.0f;
    float score_magnitude = 0.0f;
    for (unsigned int head = 0u; head < head_count; ++head) {
        const float q = zllm_bf16_to_f32(zllm_f32_to_bf16(
            query[(unsigned long long)head * head_dim + lane]));
        const unsigned long long index = (unsigned long long)pool * head_dim + lane;
        const float left = q * lower[index];
        const float right = q * upper[index];
        lows[lane] = fminf(left, right);
        highs[lane] = fmaxf(left, right);
        magnitudes[lane] = fmaxf(fabsf(left), fabsf(right));
        __syncthreads();
        for (unsigned int offset = head_dim >> 1u; offset > 0u; offset >>= 1u) {
            if (lane < offset) {
                lows[lane] += lows[lane + offset];
                highs[lane] += highs[lane + offset];
                magnitudes[lane] += magnitudes[lane + offset];
            }
            __syncthreads();
        }
        if (lane == 0u) {
            // gamma_128 < 7.7e-6；留出 2.0e-5 覆盖 WMMA 的分段归约。
            const float dot_error = magnitudes[0] * 2.0e-5f + 1.0e-6f;
            const float relu_lower = fmaxf(lows[0] - dot_error, 0.0f);
            const float relu_upper = fmaxf(highs[0] + dot_error, 0.0f);
            const float weight = head_weights[head];
            const float term = weight >= 0.0f
                ? weight * relu_upper : weight * relu_lower;
            score_upper += term;
            score_magnitude += fabsf(weight)
                * fmaxf(fabsf(relu_lower), fabsf(relu_upper));
        }
        __syncthreads();
    }
    if (lane == 0u) {
        bounds[pool] = score_upper + score_magnitude * 1.0e-5f + 1.0e-5f;
    }
}

// glm5_next kpool:原始 key 保持 Q8G 常驻，每个完整池在当前 kernel 内
// 解码 4 行、用 gate+APE 逐维 softmax 池化，再直接量化成池 key。
// 不生成 BF16/F32 全量权重或 key shadow。
extern "C" __global__ void dsa_kpool_compress_q8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* gates,
    const float* ape,
    signed char* pooled_keys,
    unsigned short* pooled_scales,
    const unsigned int* pool_block_table,
    unsigned int first_pool,
    unsigned int pool_rows,
    unsigned int columns,
    unsigned int group_size,
    unsigned int kpool,
    unsigned int block_size)
{
    const unsigned int groups_per_row = columns / group_size;
    const unsigned int local_pool = blockIdx.x / groups_per_row;
    const unsigned int group = blockIdx.x - local_pool * groups_per_row;
    const unsigned int lane = threadIdx.x;
    if (local_pool >= pool_rows || lane >= group_size || kpool == 0u || kpool > 8u) return;
    const unsigned int pool = first_pool + local_pool;
    const unsigned int column = group * group_size + lane;
    float logits[8];
    float maximum = -3.402823466e+38F;
    #pragma unroll
    for (unsigned int slot = 0; slot < 8u; ++slot) {
        float logit = -3.402823466e+38F;
        if (slot < kpool) {
            const unsigned int token = pool * kpool + slot;
            logit = gates[(unsigned long long)token * columns + column]
                + ape[(unsigned long long)slot * columns + column];
            maximum = fmaxf(maximum, logit);
        }
        logits[slot] = logit;
    }
    float denominator = 0.0f;
    float pooled = 0.0f;
    #pragma unroll
    for (unsigned int slot = 0; slot < 8u; ++slot) {
        if (slot < kpool) denominator += expf(logits[slot] - maximum);
    }
    #pragma unroll
    for (unsigned int slot = 0; slot < 8u; ++slot) {
        if (slot < kpool) {
            const unsigned int token = pool * kpool + slot;
            const unsigned long long physical = paged_row(block_table, token, block_size);
            const float scale = zllm_bf16_to_f32(
                key_scales[physical * groups_per_row + group]);
            const float key = (float)keys[physical * columns + column] * scale;
            pooled += expf(logits[slot] - maximum) * key / denominator;
        }
    }
    __shared__ float maxima[256];
    __shared__ float quant_scale;
    maxima[lane] = fabsf(pooled);
    __syncthreads();
    for (unsigned int offset = group_size >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) maxima[lane] = fmaxf(maxima[lane], maxima[lane + offset]);
        __syncthreads();
    }
    const unsigned long long pooled_physical = paged_row(pool_block_table, pool, block_size);
    if (lane == 0u) {
        const float scale = maxima[0] == 0.0f ? 1.0f : maxima[0] * (1.0f / 127.0f);
        const unsigned short encoded = zllm_f32_to_bf16(scale);
        pooled_scales[pooled_physical * groups_per_row + group] = encoded;
        quant_scale = zllm_bf16_to_f32(encoded);
    }
    __syncthreads();
    const float quantized = nearbyintf(pooled / quant_scale);
    pooled_keys[pooled_physical * columns + column]
        = (signed char)fmaxf(-127.0f, fminf(127.0f, quantized));
}

// CSA decode 在 attention 完成后把本轮 K/V 的 Q8 code 与 scale 一起推进
// recent ring。四份数据共享行映射，合成一个 launch 可避免四次短 D2D copy。
extern "C" __global__ void q8_cache_copy_pair(
    const signed char* source_key,
    const unsigned short* source_key_scales,
    const signed char* source_value,
    const unsigned short* source_value_scales,
    signed char* target_key,
    unsigned short* target_key_scales,
    signed char* target_value,
    unsigned short* target_value_scales,
    unsigned int source_row,
    unsigned int target_row,
    unsigned int rows,
    unsigned int columns,
    unsigned int scale_columns,
    unsigned int target_capacity)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int code_elements = rows * columns;
    if (index < code_elements) {
        const unsigned int row = index / columns;
        const unsigned int column = index - row * columns;
        const unsigned long long source = (unsigned long long)(source_row + row) * columns + column;
        const unsigned long long target = (unsigned long long)((target_row + row) % target_capacity) * columns + column;
        target_key[target] = source_key[source];
        target_value[target] = source_value[source];
    }
    const unsigned int scale_elements = rows * scale_columns;
    if (index < scale_elements) {
        const unsigned int row = index / scale_columns;
        const unsigned int column = index - row * scale_columns;
        const unsigned long long source = (unsigned long long)(source_row + row) * scale_columns + column;
        const unsigned long long target = (unsigned long long)((target_row + row) % target_capacity) * scale_columns + column;
        target_key_scales[target] = source_key_scales[source];
        target_value_scales[target] = source_value_scales[source];
    }
}

__device__ __forceinline__ void paged_cache_append_mla_body(
    const float* latent_input,
    signed char* latent_cache,
    unsigned short* latent_scales,
    const float* rope_input,
    const float* cosine,
    const float* sine,
    unsigned short* rope_cache,
    const unsigned int* block_table,
    unsigned int position,
    unsigned int rope_position,
    unsigned int rows,
    unsigned int latent_columns,
    unsigned int rope_columns,
    unsigned int rotary_dim,
    unsigned int split_half,
    unsigned int group_size,
    unsigned int block_size)
{
    const unsigned int groups_per_row = latent_columns / group_size;
    const unsigned int latent_groups = rows * groups_per_row;
    const unsigned int lane = threadIdx.x;
    if (blockIdx.x >= latent_groups) {
        const unsigned long long rope_index
            = (unsigned long long)(blockIdx.x - latent_groups) * blockDim.x + lane;
        const unsigned long long rope_elements = (unsigned long long)rows * rope_columns;
        if (rope_index >= rope_elements) return;
        const unsigned int row = rope_index / rope_columns;
        const unsigned int column = rope_index - (unsigned long long)row * rope_columns;
        const unsigned long long physical = paged_row(block_table, position + row, block_size);
        float value = rope_input[rope_index];
        if (rotary_dim != 0 && column >= rope_columns - rotary_dim) {
            const unsigned int start = rope_columns - rotary_dim;
            const unsigned int rotary = column - start;
            const unsigned int half = rotary_dim / 2;
            const unsigned int pair = split_half ? rotary % half : rotary / 2;
            const unsigned int real_column = split_half ? start + pair : start + pair * 2;
            const unsigned int imag_column = split_half ? start + half + pair : real_column + 1;
            const float real = rope_input[(unsigned long long)row * rope_columns + real_column];
            const float imaginary = rope_input[(unsigned long long)row * rope_columns + imag_column];
            const float c = cosine[(unsigned long long)(rope_position + row) * half + pair];
            const float s = sine[(unsigned long long)(rope_position + row) * half + pair];
            value = column == real_column ? real * c - imaginary * s : imaginary * c + real * s;
        }
        rope_cache[physical * rope_columns + column] = zllm_f32_to_bf16(value);
        return;
    }

    const unsigned int group_index = blockIdx.x;
    const unsigned int row = group_index / groups_per_row;
    const unsigned int group = group_index - row * groups_per_row;
    const unsigned int column = group * group_size + lane;
    const float value = latent_input[(unsigned long long)row * latent_columns + column];
    __shared__ float maxima[256];
    __shared__ float quant_scale;
    maxima[lane] = fabsf(value);
    __syncthreads();
    for (unsigned int offset = group_size >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) maxima[lane] = fmaxf(maxima[lane], maxima[lane + offset]);
        __syncthreads();
    }
    const unsigned long long physical = paged_row(block_table, position + row, block_size);
    if (lane == 0) {
        const float scale = maxima[0] == 0.0f ? 1.0f : maxima[0] * (1.0f / 127.0f);
        const unsigned short encoded = zllm_f32_to_bf16(scale);
        latent_scales[physical * groups_per_row + group] = encoded;
        quant_scale = zllm_bf16_to_f32(encoded);
    }
    __syncthreads();
    const float quantized = nearbyintf(value / quant_scale);
    latent_cache[physical * latent_columns + column]
        = (signed char)fmaxf(-127.0f, fminf(127.0f, quantized));
}

// 既有 kernel：position 来自 host 参数。
extern "C" __global__ void paged_cache_append_mla_f32_q8_bf16(
    const float* latent_input,
    signed char* latent_cache,
    unsigned short* latent_scales,
    const float* rope_input,
    const float* cosine,
    const float* sine,
    unsigned short* rope_cache,
    const unsigned int* block_table,
    unsigned int position,
    unsigned int rope_position,
    unsigned int rows,
    unsigned int latent_columns,
    unsigned int rope_columns,
    unsigned int rotary_dim,
    unsigned int split_half,
    unsigned int group_size,
    unsigned int block_size)
{
    paged_cache_append_mla_body(latent_input, latent_cache, latent_scales, rope_input, cosine, sine, rope_cache, block_table, position, rope_position, rows, latent_columns, rope_columns, rotary_dim, split_half, group_size, block_size);
}

// 间接变体：position/rope_position 从 device buffer 读（graph replay 固定参数）。
extern "C" __global__ void paged_cache_append_mla_f32_q8_bf16_indirect(
    const float* latent_input,
    signed char* latent_cache,
    unsigned short* latent_scales,
    const float* rope_input,
    const float* cosine,
    const float* sine,
    unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* positions,
    unsigned int rows,
    unsigned int latent_columns,
    unsigned int rope_columns,
    unsigned int rotary_dim,
    unsigned int split_half,
    unsigned int group_size,
    unsigned int block_size)
{
    paged_cache_append_mla_body(latent_input, latent_cache, latent_scales, rope_input, cosine, sine, rope_cache, block_table, positions[0], positions[1], rows, latent_columns, rope_columns, rotary_dim, split_half, group_size, block_size);
}

// CPU 全历史只上传本轮缺失行；slot 是 GPU hot cache 中的精确落点。
extern "C" __global__ void mla_hot_scatter_q8(
    const signed char* source_latent,
    const unsigned short* source_scales,
    const unsigned short* source_rope,
    const unsigned int* slots,
    signed char* target_latent,
    unsigned short* target_scales,
    unsigned short* target_rope,
    unsigned int rows,
    unsigned int latent_columns,
    unsigned int scale_columns,
    unsigned int rope_columns,
    unsigned int target_rows)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows) return;
    const unsigned int slot = slots[row];
    if (slot >= target_rows) return;
    for (unsigned int column = lane; column < latent_columns; column += blockDim.x) {
        target_latent[(unsigned long long)slot * latent_columns + column]
            = source_latent[(unsigned long long)row * latent_columns + column];
    }
    for (unsigned int column = lane; column < scale_columns; column += blockDim.x) {
        target_scales[(unsigned long long)slot * scale_columns + column]
            = source_scales[(unsigned long long)row * scale_columns + column];
    }
    for (unsigned int column = lane; column < rope_columns; column += blockDim.x) {
        target_rope[(unsigned long long)slot * rope_columns + column]
            = source_rope[(unsigned long long)row * rope_columns + column];
    }
}

extern "C" __global__ void mla_hot_gather_q8(
    const signed char* source_latent,
    const unsigned short* source_scales,
    const unsigned short* source_rope,
    const unsigned int* tokens,
    const unsigned int* slots,
    signed char* target_latent,
    unsigned short* target_scales,
    unsigned short* target_rope,
    unsigned int rows,
    unsigned int latent_columns,
    unsigned int scale_columns,
    unsigned int rope_columns,
    unsigned int target_rows)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows) return;
    const unsigned int token = tokens[row];
    const unsigned int slot = slots[row];
    if (slot >= target_rows) return;
    for (unsigned int column = lane; column < latent_columns; column += blockDim.x) {
        target_latent[(unsigned long long)slot * latent_columns + column]
            = source_latent[(unsigned long long)token * latent_columns + column];
    }
    for (unsigned int column = lane; column < scale_columns; column += blockDim.x) {
        target_scales[(unsigned long long)slot * scale_columns + column]
            = source_scales[(unsigned long long)token * scale_columns + column];
    }
    for (unsigned int column = lane; column < rope_columns; column += blockDim.x) {
        target_rope[(unsigned long long)slot * rope_columns + column]
            = source_rope[(unsigned long long)token * rope_columns + column];
    }
}

__device__ __forceinline__ unsigned int dsa_ordered_score(float score)
{
    const unsigned int bits = __float_as_uint(score);
    return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

extern "C" __global__ void dsa_quantize_query_hadamard_i8(
    const float* query,
    signed char* quantized,
    float* scales,
    unsigned int rows,
    unsigned int columns)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= columns) return;
    __shared__ float values[256];
    __shared__ float maxima[256];
    __shared__ float quant_scale;
    // raw exact WMMA 会先把 query 转成 BF16；粗排以同一 operand 为起点，
    // 只量 Hadamard+Q8 自身是否破坏 Top-K 边界。
    values[lane] = zllm_bf16_to_f32(zllm_f32_to_bf16(
        query[(unsigned long long)row * columns + lane]));
    __syncthreads();
    zllm_dsa_hadamard(values, columns);
    const float value = values[lane] * rsqrtf((float)columns);
    maxima[lane] = fabsf(value);
    __syncthreads();
    for (unsigned int offset = columns >> 1u; offset; offset >>= 1u) {
        if (lane < offset) maxima[lane] = fmaxf(maxima[lane], maxima[lane + offset]);
        __syncthreads();
    }
    if (lane == 0u) {
        quant_scale = maxima[0] == 0.0f ? 1.0f : maxima[0] * (1.0f / 127.0f);
        scales[row] = quant_scale;
    }
    __syncthreads();
    const float code = nearbyintf(value / quant_scale);
    quantized[(unsigned long long)row * columns + lane]
        = (signed char)fmaxf(-127.0f, fminf(127.0f, code));
}

extern "C" __global__ void dsa_score_tiles_i8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const signed char* query,
    const float* query_scales,
    const float* head_weights,
    unsigned int* score_keys,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int block_size,
    unsigned int tile_rows)
{
    const unsigned int tile = blockIdx.x;
    const unsigned int query_row = blockIdx.y;
    if (query_row >= query_rows) return;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const bool collect_histogram = coarse_histograms != nullptr;
    extern __shared__ unsigned int coarse_histogram[];
    if (collect_histogram) {
        coarse_histogram[threadIdx.x] = 0u;
        __syncthreads();
    }
    for (unsigned int local = wave; local < tile_rows; local += waves) {
        const unsigned int token = tile * tile_rows + local;
        if (token < context_rows && token <= query_start + query_row) {
            const unsigned long long physical = paged_row(block_table, token, block_size);
            const float key_scale = zllm_bf16_to_f32(key_scales[physical]);
            float score = 0.0f;
            for (unsigned int head = 0; head < head_count; ++head) {
                int dot = 0;
                for (unsigned int column = lane; column < head_dim; column += warpSize) {
                    dot += (int)query[((unsigned long long)query_row * head_count + head) * head_dim + column]
                        * (int)keys[physical * head_dim + column];
                }
                for (unsigned int offset = warpSize >> 1u; offset; offset >>= 1u) {
                    dot += __shfl_down(dot, offset);
                }
                if (lane == 0u) {
                    const float value = (float)dot
                        * query_scales[(unsigned long long)query_row * head_count + head]
                        * key_scale;
                    score += head_weights[(unsigned long long)query_row * head_count + head]
                        * fmaxf(value, 0.0f);
                }
            }
            if (lane == 0u) {
                const unsigned long long stride = (unsigned long long)gridDim.x * tile_rows;
                const unsigned int key = dsa_ordered_score(score);
                score_keys[(unsigned long long)query_row * stride + token] = key;
                if (collect_histogram) atomicAdd(&coarse_histogram[key >> 24], 1u);
            }
        }
    }
    if (collect_histogram) {
        __syncthreads();
        const unsigned int count = coarse_histogram[threadIdx.x];
        if (count != 0u) {
            atomicAdd(&coarse_histograms[(unsigned long long)query_row * 256u + threadIdx.x], count);
        }
    }
}

#if defined(__gfx950__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
#define ZLLM_PAGED_BF16_DOT2 1
using zllm_paged_bf16x2 = __bf16 __attribute__((ext_vector_type(2)));
#endif

extern "C" __global__ void dsa_score_tiles_q8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* query,
    const float* head_weights,
    unsigned int* score_keys,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int key_group_size,
    unsigned int block_size,
    unsigned int tile_rows)
{
    const unsigned int groups_per_row = head_dim / key_group_size;
    const unsigned int tile = blockIdx.x;
    const unsigned int query_row = blockIdx.y;
    if (query_row >= query_rows) return;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const bool collect_histogram = coarse_histograms != nullptr;
    extern __shared__ unsigned int coarse_histogram[];
    if (collect_histogram) {
        coarse_histogram[threadIdx.x] = 0u;
        __syncthreads();
    }
    for (unsigned int local = wave; local < tile_rows; local += waves) {
        const unsigned int token = tile * tile_rows + local;
        if (token < context_rows && token <= query_start + query_row) {
            const unsigned long long physical = paged_row(block_table, token, block_size);
            float score = 0.0f;
            for (unsigned int head = 0; head < head_count; ++head) {
                float dot = 0.0f;
                for (unsigned int column = lane; column < head_dim; column += warpSize) {
                    dot += query[
                        ((unsigned long long)query_row * head_count + head) * head_dim
                            + column]
                        * ((float)keys[physical * head_dim + column]
                            * zllm_bf16_to_f32(key_scales[physical * groups_per_row + column / key_group_size]));
                }
                for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
                    dot += __shfl_down(dot, offset);
                }
                if (lane == 0) {
                    score += head_weights[
                        (unsigned long long)query_row * head_count + head]
                        * fmaxf(dot, 0.0f);
                }
            }
            if (lane == 0) {
                const unsigned long long stride = (unsigned long long)gridDim.x * tile_rows;
                const unsigned int key = dsa_ordered_score(score);
                score_keys[(unsigned long long)query_row * stride + token] = key;
                if (collect_histogram) atomicAdd(&coarse_histogram[key >> 24], 1u);
            }
        }
    }
    if (collect_histogram) {
        __syncthreads();
        const unsigned int count = coarse_histogram[threadIdx.x];
        if (count != 0u) {
            atomicAdd(&coarse_histograms[(unsigned long long)query_row * 256u + threadIdx.x], count);
        }
    }
}

// Prefill 索引器以 16×16 BF16 WMMA 计算逐 head QK，ReLU 后再用 F32 head weight 归约。
// 每个 block 复用同一 query，并把 128 个分页 K 分给各 wave，避免标量路径重复读取 K。
// gfx11+ 原生 WMMA：每个 block 同时处理四行 query，共享 128 行 K。
template <bool PrefixOnly, bool Rows2 = false>
__device__ __forceinline__ void dsa_score_tiles_native_wmma_q8_impl(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* query,
    const float* head_weights,
    void* score_output,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int visibility_divisor,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int key_group_size,
    unsigned int block_size,
    unsigned int tile_rows,
    unsigned int score_stride,
    unsigned int shard_parity)
{
#if defined(__gfx950__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    using DsaWmmaShort16 = short __attribute__((ext_vector_type(16)));
    using DsaWmmaFloat8 = float __attribute__((ext_vector_type(8)));

    const unsigned int tile = blockIdx.x;
    const unsigned int queries_per_block = Rows2 ? 2u : 4u;
    const unsigned int query_base = blockIdx.y * queries_per_block;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int query_offset = wave >> 3;
    const unsigned int key_wave = wave & 7u;
    const unsigned int query_row = query_base + query_offset;
    const unsigned int query_row1 = query_row + 2u;
    const bool collect_histogram = coarse_histograms != nullptr;

    __shared__ unsigned short query_tiles[4][16 * 16];
    __shared__ unsigned short key_tiles[8][16 * 16];
    __shared__ unsigned int physicals[256];
    __shared__ float key_row_scales[256];
    extern __shared__ unsigned int coarse_histogram[];
    if (collect_histogram) {
        for (unsigned int flat = threadIdx.x;
             flat < queries_per_block * 256u; flat += blockDim.x) {
            coarse_histogram[flat] = 0u;
        }
        __syncthreads();
    }

    for (unsigned int local = threadIdx.x; local < tile_rows; local += blockDim.x) {
        const unsigned int compact_token = tile * tile_rows + local;
        const unsigned int token = paged_shard_global_token(compact_token, block_size, shard_parity);
        physicals[local] = token < context_rows
            ? (unsigned int)paged_row(block_table, compact_token, block_size)
            : 0u;
    }
    __syncthreads();
    // GLM-5.2 的 Q8G128 每行只有一个 scale。先转成 F32 放入 LDS，避免后续
    // 每个 column、head batch 都重复读取同一 BF16 scale并做类型转换。
    if (key_group_size == head_dim) {
        for (unsigned int local = threadIdx.x; local < tile_rows; local += blockDim.x) {
            const unsigned int compact_token = tile * tile_rows + local;
            const unsigned int token = paged_shard_global_token(compact_token, block_size, shard_parity);
            key_row_scales[local] = token < context_rows
                ? zllm_bf16_to_f32(key_scales[physicals[local]])
                : 1.0f;
        }
        __syncthreads();
    }
    // K 与 head 无关。先按 column 载入一次，再供两个 16-head 组复用；
    // 避免 GLM-5.2 的 64 heads 对同一长上下文 K tile 只读两遍。
    for (unsigned int candidate_base = 0; candidate_base < tile_rows; candidate_base += 128u) {
        float score0 = 0.0f;
        float score1 = 0.0f;
        for (unsigned int head_batch = 0; head_batch < head_count; head_batch += 32u) {
            DsaWmmaFloat8 sums0[2] = {}, sums1[2] = {};
            for (unsigned int column_base = 0; column_base < head_dim; column_base += 16u) {
                // Q8 key 在进入 WMMA 前按 group scale 直接解码到 LDS，不生成全局 BF16 展开。
                const unsigned int groups_per_row = head_dim / key_group_size;
                for (unsigned int flat = threadIdx.x; flat < 8u * 16u * 16u; flat += blockDim.x) {
                    const unsigned int target_wave = flat / (16u * 16u);
                    const unsigned int element = flat % (16u * 16u);
                    const unsigned int row = element / 16u;
                    const unsigned int column = column_base + element % 16u;
                    const unsigned int local = candidate_base + target_wave * 16u + row;
                    const unsigned int compact_token = tile * tile_rows + local;
                    const unsigned int token = paged_shard_global_token(compact_token, block_size, shard_parity);
                    float value = 0.0f;
                    if (local < tile_rows && token < context_rows) {
                        const unsigned long long physical = physicals[local];
                        const float scale = key_group_size == head_dim
                            ? key_row_scales[local]
                            : zllm_bf16_to_f32(key_scales[
                                physical * groups_per_row + column / key_group_size]);
                        value = (float)keys[physical * head_dim + column] * scale;
                    }
                    key_tiles[target_wave][element] = zllm_f32_to_bf16(value);
                }
                for (unsigned int head_group = 0; head_group < 2u; ++head_group) {
                    const unsigned int head_base = head_batch + head_group * 16u;
                    for (unsigned int flat = threadIdx.x;
                         flat < queries_per_block * 16u * 16u;
                         flat += blockDim.x) {
                        const unsigned int target_query = flat / (16u * 16u);
                        const unsigned int element = flat % (16u * 16u);
                        const unsigned int head = head_base + element / 16u;
                        const unsigned int column = column_base + element % 16u;
                        const unsigned int source_row = query_base + target_query;
                        query_tiles[target_query][element] = source_row < query_rows && head < head_count
                            ? zllm_f32_to_bf16(query[
                                ((unsigned long long)source_row * head_count + head) * head_dim
                                    + column])
                            : 0u;
                    }
                    __syncthreads();
                    const DsaWmmaShort16 query_fragment0
                        = *reinterpret_cast<const DsaWmmaShort16*>(
                            query_tiles[query_offset] + (lane & 15u) * 16u);
                    const DsaWmmaShort16 key_fragment
                        = *reinterpret_cast<const DsaWmmaShort16*>(
                            key_tiles[key_wave] + (lane & 15u) * 16u);
                    sums0[head_group] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                        query_fragment0, key_fragment, sums0[head_group]);
                    if constexpr (!Rows2) {
                        const DsaWmmaShort16 query_fragment1
                            = *reinterpret_cast<const DsaWmmaShort16*>(
                                query_tiles[query_offset + 2u]
                                    + (lane & 15u) * 16u);
                        sums1[head_group] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                            query_fragment1, key_fragment, sums1[head_group]);
                    }
                    __syncthreads();
                }
            }

            const unsigned int parity = lane >> 4;
            #pragma unroll
            for (unsigned int head_group = 0; head_group < 2u; ++head_group) {
                const unsigned int head_base = head_batch + head_group * 16u;
                float partial0 = 0.0f;
                float partial1 = 0.0f;
                #pragma unroll
                for (unsigned int index = 0; index < 8u; ++index) {
                    const unsigned int head = head_base + index * 2u + parity;
                    if (query_row < query_rows && head < head_count) {
                        partial0 += head_weights[
                            (unsigned long long)query_row * head_count + head]
                            * fmaxf(sums0[head_group][index], 0.0f);
                    }
                    if constexpr (!Rows2) {
                        if (query_row1 < query_rows && head < head_count) {
                            partial1 += head_weights[
                                (unsigned long long)query_row1 * head_count + head]
                                * fmaxf(sums1[head_group][index], 0.0f);
                        }
                    }
                }
                partial0 += __shfl_down(partial0, 16u, 32u);
                if constexpr (!Rows2) {
                    partial1 += __shfl_down(partial1, 16u, 32u);
                }
                if (lane < 16u) {
                    score0 += partial0;
                    if constexpr (!Rows2) {
                        score1 += partial1;
                    }
                }
            }
        }

        if (lane < 16u) {
            const unsigned int candidate_slot
                = candidate_base + key_wave * 16u + (lane & 15u);
            const unsigned int compact_token = tile * tile_rows + candidate_slot;
            const unsigned int token = paged_shard_global_token(compact_token, block_size, shard_parity);
            if (candidate_slot < tile_rows && token < context_rows) {
                const unsigned long long stride = score_stride;
                if (query_row < query_rows && token < (query_start + query_row + 1u) / visibility_divisor) {
                    const unsigned int key = dsa_ordered_score(score0);
                    if constexpr (PrefixOnly) {
                        ((unsigned char*)score_output)[
                            (unsigned long long)query_row * stride + token]
                            = (unsigned char)(key >> 24);
                    } else {
                        ((unsigned int*)score_output)[
                            (unsigned long long)query_row * stride + token] = key;
                    }
                    if (collect_histogram) {
                        atomicAdd(&coarse_histogram[query_offset * 256u + (key >> 24)], 1u);
                    }
                }
                if constexpr (!Rows2) {
                    if (query_row1 < query_rows && token < (query_start + query_row1 + 1u) / visibility_divisor) {
                        const unsigned int key = dsa_ordered_score(score1);
                        if constexpr (PrefixOnly) {
                            ((unsigned char*)score_output)[
                                (unsigned long long)query_row1 * stride + token]
                                = (unsigned char)(key >> 24);
                        } else {
                            ((unsigned int*)score_output)[
                                (unsigned long long)query_row1 * stride + token] = key;
                        }
                        if (collect_histogram) {
                            atomicAdd(&coarse_histogram[(query_offset + 2u) * 256u + (key >> 24)], 1u);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (collect_histogram) {
        for (unsigned int flat = threadIdx.x;
             flat < queries_per_block * 256u; flat += blockDim.x) {
            const unsigned int local_query = flat / 256u;
            const unsigned int query = query_base + local_query;
            const unsigned int count = coarse_histogram[flat];
            if (query < query_rows && count != 0u) {
                atomicAdd(&coarse_histograms[(unsigned long long)query * 256u + (flat & 255u)], count);
            }
        }
    }
#else
    (void)keys;
    (void)key_scales;
    (void)block_table;
    (void)query;
    (void)head_weights;
    (void)score_output;
    (void)coarse_histograms;
    (void)query_rows;
    (void)context_rows;
    (void)query_start;
    (void)visibility_divisor;
    (void)head_count;
    (void)head_dim;
    (void)key_group_size;
    (void)block_size;
    (void)tile_rows;
    (void)score_stride;
    (void)shard_parity;
#endif
}

// Verify 两行沿用通用核的 sums0 累加顺序，但不再构造永远越界的 row+2
// accumulator。16 个 wave 仍共同复用同一 256-token K tile。
extern "C" __global__ void dsa_score_tiles_native_wmma_q8_rows2(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* query,
    const float* head_weights,
    unsigned int* score_keys,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int key_group_size,
    unsigned int block_size,
    unsigned int tile_rows,
    unsigned int score_stride,
    unsigned int shard_parity)
{
    dsa_score_tiles_native_wmma_q8_impl<false, true>(
        keys, key_scales, block_table, query, head_weights, score_keys,
        coarse_histograms, query_rows, context_rows, query_start, 1u,
        head_count, head_dim, key_group_size, block_size, tile_rows,
        score_stride, shard_parity);
}

extern "C" __global__ void dsa_score_tiles_native_wmma_q8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* query,
    const float* head_weights,
    unsigned int* score_keys,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int key_group_size,
    unsigned int block_size,
    unsigned int tile_rows,
    unsigned int score_stride,
    unsigned int shard_parity)
{
    dsa_score_tiles_native_wmma_q8_impl<false>(
        keys, key_scales, block_table, query, head_weights, score_keys,
        coarse_histograms, query_rows, context_rows, query_start, 1u,
        head_count, head_dim, key_group_size, block_size, tile_rows,
        score_stride, shard_parity);
}

// Decode 单行不走四行通用映射：8 个 wave 各负责 16 个 token，既保持
// 原 kernel 第一套 accumulator 的 WMMA/F32 累加顺序，也删除其余三套
// 无效 query 与 accumulator。tile_rows 由 host 固定为 128。
extern "C" __global__ void dsa_score_tiles_native_wmma_q8_decode(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* query,
    const float* head_weights,
    unsigned int* score_keys,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int key_group_size,
    unsigned int block_size,
    unsigned int tile_rows,
    unsigned int score_stride,
    unsigned int shard_parity)
{
#if defined(__gfx950__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    using DsaWmmaShort16 = short __attribute__((ext_vector_type(16)));
    using DsaWmmaFloat8 = float __attribute__((ext_vector_type(8)));

    constexpr unsigned int max_tile_rows = 128u;
    const unsigned int tile = blockIdx.x;
    const unsigned int query_row = blockIdx.y;
    if (query_row >= query_rows || tile_rows > max_tile_rows) return;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const bool collect_histogram = coarse_histograms != nullptr;

    __shared__ unsigned short query_tile[16 * 16];
    __shared__ unsigned short key_tiles[8][16 * 16];
    __shared__ unsigned int physicals[max_tile_rows];
    __shared__ float key_row_scales[max_tile_rows];
    extern __shared__ unsigned int coarse_histogram[];
    if (collect_histogram) {
        coarse_histogram[threadIdx.x] = 0u;
        __syncthreads();
    }

    for (unsigned int local = threadIdx.x; local < tile_rows;
         local += blockDim.x) {
        const unsigned int compact_token = tile * tile_rows + local;
        const unsigned int token = paged_shard_global_token(compact_token, block_size, shard_parity);
        physicals[local] = token < context_rows
            ? (unsigned int)paged_row(block_table, compact_token, block_size)
            : 0u;
    }
    __syncthreads();
    if (key_group_size == head_dim) {
        for (unsigned int local = threadIdx.x; local < tile_rows;
             local += blockDim.x) {
            const unsigned int compact_token = tile * tile_rows + local;
            const unsigned int token = paged_shard_global_token(compact_token, block_size, shard_parity);
            key_row_scales[local] = token < context_rows
                ? zllm_bf16_to_f32(key_scales[physicals[local]])
                : 1.0f;
        }
        __syncthreads();
    }

    float score = 0.0f;
    for (unsigned int head_batch = 0; head_batch < head_count;
         head_batch += 32u) {
        DsaWmmaFloat8 sums[2] = {};
        for (unsigned int column_base = 0; column_base < head_dim;
             column_base += 16u) {
            const unsigned int groups_per_row = head_dim / key_group_size;
            for (unsigned int flat = threadIdx.x;
                 flat < 8u * 16u * 16u; flat += blockDim.x) {
                const unsigned int target_wave = flat / (16u * 16u);
                const unsigned int element = flat % (16u * 16u);
                const unsigned int row = element / 16u;
                const unsigned int column = column_base + element % 16u;
                const unsigned int local = target_wave * 16u + row;
                const unsigned int compact_token = tile * tile_rows + local;
                const unsigned int token = paged_shard_global_token(compact_token, block_size, shard_parity);
                float value = 0.0f;
                if (local < tile_rows && token < context_rows) {
                    const unsigned long long physical = physicals[local];
                    const float scale = key_group_size == head_dim
                        ? key_row_scales[local]
                        : zllm_bf16_to_f32(key_scales[
                            physical * groups_per_row
                                + column / key_group_size]);
                    value = (float)keys[physical * head_dim + column] * scale;
                }
                key_tiles[target_wave][element] = zllm_f32_to_bf16(value);
            }
            for (unsigned int head_group = 0; head_group < 2u;
                 ++head_group) {
                const unsigned int head_base = head_batch + head_group * 16u;
                for (unsigned int flat = threadIdx.x; flat < 16u * 16u;
                     flat += blockDim.x) {
                    const unsigned int head = head_base + flat / 16u;
                    const unsigned int column = column_base + flat % 16u;
                    query_tile[flat] = head < head_count
                        ? zllm_f32_to_bf16(query[
                            ((unsigned long long)query_row * head_count + head)
                                * head_dim + column])
                        : 0u;
                }
                __syncthreads();
                const DsaWmmaShort16 query_fragment
                    = *reinterpret_cast<const DsaWmmaShort16*>(
                        query_tile + (lane & 15u) * 16u);
                const DsaWmmaShort16 key_fragment
                    = *reinterpret_cast<const DsaWmmaShort16*>(
                        key_tiles[wave] + (lane & 15u) * 16u);
                sums[head_group]
                    = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                        query_fragment, key_fragment, sums[head_group]);
                __syncthreads();
            }
        }

        const unsigned int parity = lane >> 4;
#pragma unroll
        for (unsigned int head_group = 0; head_group < 2u; ++head_group) {
            const unsigned int head_base = head_batch + head_group * 16u;
            float partial = 0.0f;
#pragma unroll
            for (unsigned int index = 0; index < 8u; ++index) {
                const unsigned int head = head_base + index * 2u + parity;
                if (head < head_count) {
                    partial += head_weights[
                        (unsigned long long)query_row * head_count + head]
                        * fmaxf(sums[head_group][index], 0.0f);
                }
            }
            partial += __shfl_down(partial, 16u, 32u);
            if (lane < 16u) score += partial;
        }
    }

    if (lane < 16u) {
        const unsigned int local = wave * 16u + lane;
        const unsigned int compact_token = tile * tile_rows + local;
        const unsigned int token = paged_shard_global_token(compact_token, block_size, shard_parity);
        if (local < tile_rows && token < context_rows
            && token <= query_start + query_row) {
            const unsigned long long stride = score_stride;
            const unsigned int key = dsa_ordered_score(score);
            const unsigned int score_token
                = shard_parity < 2u ? compact_token : token;
            score_keys[(unsigned long long)query_row * stride + score_token] = key;
            if (collect_histogram) {
                atomicAdd(&coarse_histogram[key >> 24], 1u);
            }
        }
    }
    if (collect_histogram) {
        __syncthreads();
        const unsigned int count = coarse_histogram[threadIdx.x];
        if (count != 0u) {
            atomicAdd(&coarse_histograms[
                (unsigned long long)query_row * 256u + threadIdx.x], count);
        }
    }
#else
    (void)keys;
    (void)key_scales;
    (void)block_table;
    (void)query;
    (void)head_weights;
    (void)score_keys;
    (void)coarse_histograms;
    (void)query_rows;
    (void)context_rows;
    (void)query_start;
    (void)head_count;
    (void)head_dim;
    (void)key_group_size;
    (void)block_size;
    (void)tile_rows;
    (void)score_stride;
    (void)shard_parity;
#endif
}

// gfx11 的 signed i8 WMMA。K 保持分页 Q8，Q 只在每次 select 开头量化一次；
// i32 累加后分别乘 Q/K scale，避免把完整 K tile 反量化成 BF16 LDS。
extern "C" __global__ void dsa_score_tiles_native_wmma_i8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const signed char* query,
    const float* query_scales,
    const float* head_weights,
    unsigned int* score_keys,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int block_size,
    unsigned int tile_rows)
{
#if defined(__gfx950__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    using DsaWmmaInt4 = int __attribute__((ext_vector_type(4)));
    using DsaWmmaInt8 = int __attribute__((ext_vector_type(8)));

    const unsigned int tile = blockIdx.x;
    const unsigned int query_base = blockIdx.y * 4u;
    const unsigned int wave = threadIdx.x >> 5u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int query_offset = wave >> 3u;
    const unsigned int key_wave = wave & 7u;
    const unsigned int query_row = query_base + query_offset;
    const unsigned int query_row1 = query_row + 2u;
    const bool collect_histogram = coarse_histograms != nullptr;

    __shared__ signed char query_tiles[4][16 * 16];
    __shared__ signed char key_tiles[8][16 * 16];
    __shared__ unsigned int physicals[256];
    extern __shared__ unsigned int coarse_histogram[];
    if (collect_histogram) {
        for (unsigned int flat = threadIdx.x; flat < 4u * 256u; flat += blockDim.x) {
            coarse_histogram[flat] = 0u;
        }
        __syncthreads();
    }
    for (unsigned int local = threadIdx.x; local < tile_rows; local += blockDim.x) {
        const unsigned int token = tile * tile_rows + local;
        physicals[local] = token < context_rows
            ? (unsigned int)paged_row(block_table, token, block_size)
            : 0u;
    }
    __syncthreads();

    for (unsigned int candidate_base = 0; candidate_base < tile_rows; candidate_base += 128u) {
        float score0 = 0.0f;
        float score1 = 0.0f;
        const unsigned int candidate_slot = candidate_base + key_wave * 16u + (lane & 15u);
        const unsigned int token = tile * tile_rows + candidate_slot;
        const float key_scale = candidate_slot < tile_rows && token < context_rows
            ? zllm_bf16_to_f32(key_scales[physicals[candidate_slot]])
            : 0.0f;
        for (unsigned int head_batch = 0; head_batch < head_count; head_batch += 32u) {
            DsaWmmaInt8 sums0[2] = {}, sums1[2] = {};
            for (unsigned int column_base = 0; column_base < head_dim; column_base += 16u) {
                for (unsigned int flat = threadIdx.x; flat < 8u * 16u * 16u; flat += blockDim.x) {
                    const unsigned int target_wave = flat / (16u * 16u);
                    const unsigned int element = flat % (16u * 16u);
                    const unsigned int row = element / 16u;
                    const unsigned int column = column_base + element % 16u;
                    const unsigned int local = candidate_base + target_wave * 16u + row;
                    const unsigned int key_token = tile * tile_rows + local;
                    key_tiles[target_wave][element] = local < tile_rows && key_token < context_rows
                        ? keys[(unsigned long long)physicals[local] * head_dim + column]
                        : 0;
                }
                for (unsigned int head_group = 0; head_group < 2u; ++head_group) {
                    const unsigned int head_base = head_batch + head_group * 16u;
                    for (unsigned int flat = threadIdx.x; flat < 4u * 16u * 16u; flat += blockDim.x) {
                        const unsigned int target_query = flat / (16u * 16u);
                        const unsigned int element = flat % (16u * 16u);
                        const unsigned int head = head_base + element / 16u;
                        const unsigned int column = column_base + element % 16u;
                        const unsigned int source_row = query_base + target_query;
                        query_tiles[target_query][element] = source_row < query_rows && head < head_count
                            ? query[((unsigned long long)source_row * head_count + head) * head_dim + column]
                            : 0;
                    }
                    __syncthreads();
                    const DsaWmmaInt4 query_fragment0 = *reinterpret_cast<const DsaWmmaInt4*>(
                        query_tiles[query_offset] + (lane & 15u) * 16u);
                    const DsaWmmaInt4 query_fragment1 = *reinterpret_cast<const DsaWmmaInt4*>(
                        query_tiles[query_offset + 2u] + (lane & 15u) * 16u);
                    const DsaWmmaInt4 key_fragment = *reinterpret_cast<const DsaWmmaInt4*>(
                        key_tiles[key_wave] + (lane & 15u) * 16u);
                    sums0[head_group] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, query_fragment0, true, key_fragment, sums0[head_group], false);
                    sums1[head_group] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, query_fragment1, true, key_fragment, sums1[head_group], false);
                    __syncthreads();
                }
            }

            const unsigned int parity = lane >> 4u;
            #pragma unroll
            for (unsigned int head_group = 0; head_group < 2u; ++head_group) {
                const unsigned int head_base = head_batch + head_group * 16u;
                float partial0 = 0.0f;
                float partial1 = 0.0f;
                #pragma unroll
                for (unsigned int index = 0; index < 8u; ++index) {
                    const unsigned int head = head_base + index * 2u + parity;
                    if (query_row < query_rows && head < head_count) {
                        const float dot = (float)sums0[head_group][index]
                            * query_scales[(unsigned long long)query_row * head_count + head]
                            * key_scale;
                        partial0 += head_weights[(unsigned long long)query_row * head_count + head]
                            * fmaxf(dot, 0.0f);
                    }
                    if (query_row1 < query_rows && head < head_count) {
                        const float dot = (float)sums1[head_group][index]
                            * query_scales[(unsigned long long)query_row1 * head_count + head]
                            * key_scale;
                        partial1 += head_weights[(unsigned long long)query_row1 * head_count + head]
                            * fmaxf(dot, 0.0f);
                    }
                }
                partial0 += __shfl_down(partial0, 16u, 32u);
                partial1 += __shfl_down(partial1, 16u, 32u);
                if (lane < 16u) {
                    score0 += partial0;
                    score1 += partial1;
                }
            }
        }

        if (lane < 16u && candidate_slot < tile_rows && token < context_rows) {
            const unsigned long long stride = (unsigned long long)gridDim.x * tile_rows;
            if (query_row < query_rows && token <= query_start + query_row) {
                const unsigned int key = dsa_ordered_score(score0);
                score_keys[(unsigned long long)query_row * stride + token] = key;
                if (collect_histogram) atomicAdd(&coarse_histogram[query_offset * 256u + (key >> 24u)], 1u);
            }
            if (query_row1 < query_rows && token <= query_start + query_row1) {
                const unsigned int key = dsa_ordered_score(score1);
                score_keys[(unsigned long long)query_row1 * stride + token] = key;
                if (collect_histogram) atomicAdd(&coarse_histogram[(query_offset + 2u) * 256u + (key >> 24u)], 1u);
            }
        }
        __syncthreads();
    }
    if (collect_histogram) {
        for (unsigned int flat = threadIdx.x; flat < 4u * 256u; flat += blockDim.x) {
            const unsigned int local_query = flat / 256u;
            const unsigned int row = query_base + local_query;
            const unsigned int count = coarse_histogram[flat];
            if (row < query_rows && count != 0u) {
                atomicAdd(&coarse_histograms[(unsigned long long)row * 256u + (flat & 255u)], count);
            }
        }
    }
#else
    (void)keys;
    (void)key_scales;
    (void)block_table;
    (void)query;
    (void)query_scales;
    (void)head_weights;
    (void)score_keys;
    (void)coarse_histograms;
    (void)query_rows;
    (void)context_rows;
    (void)query_start;
    (void)head_count;
    (void)head_dim;
    (void)block_size;
    (void)tile_rows;
#endif
}

extern "C" __global__ void dsa_score_tiles_native_wmma_q8_kpool(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* query,
    const float* head_weights,
    unsigned int* score_keys,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int key_group_size,
    unsigned int block_size,
    unsigned int tile_rows,
    unsigned int kpool)
{
    dsa_score_tiles_native_wmma_q8_impl<false>(
        keys, key_scales, block_table, query, head_weights, score_keys,
        coarse_histograms, query_rows, context_rows, query_start, kpool,
        head_count, head_dim, key_group_size, block_size, tile_rows,
        gridDim.x * tile_rows, 2u);
}

extern "C" __global__ void dsa_score_prefix_tiles_native_wmma_q8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* query,
    const float* head_weights,
    unsigned char* score_prefixes,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int key_group_size,
    unsigned int block_size,
    unsigned int tile_rows,
    unsigned int score_stride,
    unsigned int shard_parity)
{
    dsa_score_tiles_native_wmma_q8_impl<true>(
        keys, key_scales, block_table, query, head_weights, score_prefixes,
        coarse_histograms, query_rows, context_rows,
        query_start, 1u, head_count, head_dim, key_group_size, block_size,
        tile_rows, score_stride, shard_parity);
}

// 正常路径只对最高字节阈值桶重算完整 score；极端聚集分布由固定数量
// persistent blocks 扫描溢出行，避免 CPU 回读后再决定是否启动大 grid。
extern "C" __global__ void dsa_score_selected_native_wmma_q8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* query,
    const float* head_weights,
    const unsigned int* candidate_tokens,
    const unsigned int* candidate_counts,
    unsigned int candidate_capacity,
    const unsigned int* overflow_rows,
    const unsigned int* overflow_count,
    unsigned int* overflow_cursor,
    unsigned int* score_keys,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int key_group_size,
    unsigned int block_size,
    unsigned int score_stride,
    unsigned int candidate_local)
{
#if defined(__gfx950__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    using DsaWmmaShort16 = short __attribute__((ext_vector_type(16)));
    using DsaWmmaFloat8 = float __attribute__((ext_vector_type(8)));

    const bool compact = candidate_tokens != nullptr;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int key_wave = wave;
    const unsigned int tile_rows = 128u;
    const unsigned int tile_count = (context_rows + tile_rows - 1u) / tile_rows;
    const unsigned int groups_per_row = head_dim / key_group_size;

    __shared__ unsigned short query_tile[16 * 16];
    __shared__ unsigned short key_tiles[8][16 * 16];
    __shared__ unsigned int tokens[128];
    __shared__ unsigned int physicals[128];
    __shared__ float key_row_scales[128];
    __shared__ unsigned int work_row;
    __shared__ unsigned int work_base;
    __shared__ unsigned int work_count;
    __shared__ unsigned int work_done;

    for (;;) {
        if (threadIdx.x == 0u) {
            if (compact) {
                work_row = blockIdx.y;
                work_base = blockIdx.x * tile_rows;
                work_count = work_row < query_rows
                    ? (candidate_counts == nullptr
                        ? candidate_capacity : candidate_counts[work_row])
                    : 0u;
                work_done = work_row >= query_rows
                    || work_count > candidate_capacity || work_base >= work_count;
            } else {
                const unsigned int task = atomicAdd(overflow_cursor, 1u);
                const unsigned int rows = *overflow_count;
                work_done = task >= rows * tile_count;
                if (!work_done) {
                    work_row = overflow_rows[task / tile_count];
                    work_base = (task % tile_count) * tile_rows;
                    work_count = query_start + work_row + 1u;
                }
            }
        }
        __syncthreads();
        if (work_done) return;

        for (unsigned int local = threadIdx.x; local < tile_rows;
             local += blockDim.x) {
            const unsigned int slot = work_base + local;
            const unsigned int token = compact
                ? candidate_tokens[(unsigned long long)work_row * candidate_capacity + slot]
                : slot;
            const bool active = slot < work_count && token < context_rows;
            tokens[local] = active ? token : 0u;
            physicals[local] = active
                ? (unsigned int)paged_row(block_table, token, block_size)
                : 0u;
        }
        __syncthreads();
        if (key_group_size == head_dim) {
            for (unsigned int local = threadIdx.x; local < tile_rows;
                 local += blockDim.x) {
                const unsigned int slot = work_base + local;
                key_row_scales[local] = slot < work_count
                    && tokens[local] < context_rows
                    ? zllm_bf16_to_f32(key_scales[physicals[local]])
                    : 1.0f;
            }
            __syncthreads();
        }

        float score = 0.0f;
        for (unsigned int head_batch = 0; head_batch < head_count;
             head_batch += 32u) {
            DsaWmmaFloat8 sums[2] = {};
            for (unsigned int column_base = 0; column_base < head_dim;
                 column_base += 16u) {
                for (unsigned int flat = threadIdx.x;
                     flat < 8u * 16u * 16u; flat += blockDim.x) {
                    const unsigned int target_wave = flat / (16u * 16u);
                    const unsigned int element = flat % (16u * 16u);
                    const unsigned int row = element / 16u;
                    const unsigned int column = column_base + element % 16u;
                    const unsigned int local = target_wave * 16u + row;
                    const unsigned int slot = work_base + local;
                    float value = 0.0f;
                    if (slot < work_count && tokens[local] < context_rows) {
                        const unsigned long long physical = physicals[local];
                        const float scale = key_group_size == head_dim
                            ? key_row_scales[local]
                            : zllm_bf16_to_f32(key_scales[
                                physical * groups_per_row
                                    + column / key_group_size]);
                        value = (float)keys[physical * head_dim + column] * scale;
                    }
                    key_tiles[target_wave][element] = zllm_f32_to_bf16(value);
                }
                for (unsigned int head_group = 0; head_group < 2u;
                     ++head_group) {
                    const unsigned int head_base = head_batch + head_group * 16u;
                    for (unsigned int flat = threadIdx.x; flat < 16u * 16u;
                         flat += blockDim.x) {
                        const unsigned int head = head_base + flat / 16u;
                        const unsigned int column = column_base + flat % 16u;
                        query_tile[flat] = work_row < query_rows && head < head_count
                            ? zllm_f32_to_bf16(query[
                                ((unsigned long long)work_row * head_count + head)
                                    * head_dim + column])
                            : 0u;
                    }
                    __syncthreads();
                    const DsaWmmaShort16 query_fragment
                        = *reinterpret_cast<const DsaWmmaShort16*>(
                            query_tile + (lane & 15u) * 16u);
                    const DsaWmmaShort16 key_fragment
                        = *reinterpret_cast<const DsaWmmaShort16*>(
                            key_tiles[key_wave] + (lane & 15u) * 16u);
                    sums[head_group]
                        = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                            query_fragment, key_fragment, sums[head_group]);
                    __syncthreads();
                }
            }

            const unsigned int parity = lane >> 4;
#pragma unroll
            for (unsigned int head_group = 0; head_group < 2u;
                 ++head_group) {
                const unsigned int head_base = head_batch + head_group * 16u;
                float partial = 0.0f;
#pragma unroll
                for (unsigned int index = 0; index < 8u; ++index) {
                    const unsigned int head = head_base + index * 2u + parity;
                    if (head < head_count) {
                        partial += head_weights[
                            (unsigned long long)work_row * head_count + head]
                            * fmaxf(sums[head_group][index], 0.0f);
                    }
                }
                partial += __shfl_down(partial, 16u, 32u);
                if (lane < 16u) score += partial;
            }
        }

        if (lane < 16u) {
            const unsigned int local = key_wave * 16u + lane;
            const unsigned int slot = work_base + local;
            if (slot < work_count) {
                const unsigned int token = tokens[local];
                const unsigned int output = candidate_local ? slot : token;
                score_keys[(unsigned long long)work_row * score_stride + output]
                    = dsa_ordered_score(score);
            }
        }
        __syncthreads();
        if (compact) return;
    }
#else
    (void)keys;
    (void)key_scales;
    (void)block_table;
    (void)query;
    (void)head_weights;
    (void)candidate_tokens;
    (void)candidate_counts;
    (void)candidate_capacity;
    (void)overflow_rows;
    (void)overflow_count;
    (void)overflow_cursor;
    (void)score_keys;
    (void)query_rows;
    (void)context_rows;
    (void)query_start;
    (void)head_count;
    (void)head_dim;
    (void)key_group_size;
    (void)block_size;
    (void)score_stride;
    (void)candidate_local;
#endif
}

extern "C" __global__ void dsa_map_candidate_selection(
    const unsigned int* candidate_tokens,
    const unsigned int* candidate_slots,
    unsigned int* selection,
    unsigned int query_rows,
    unsigned int candidate_capacity,
    unsigned int top_k)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = query_rows * top_k;
    if (index >= elements) return;
    const unsigned int row = index / top_k;
    const unsigned int slot = candidate_slots[index];
    selection[index] = slot < candidate_capacity
        ? candidate_tokens[(unsigned long long)row * candidate_capacity + slot]
        : 0xffffffffu;
}

extern "C" __global__ void dsa_score_tiles_wmma_q8(
    const signed char* keys,
    const unsigned short* key_scales,
    const unsigned int* block_table,
    const float* query,
    const float* head_weights,
    unsigned int* score_keys,
    unsigned int* coarse_histograms,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int key_group_size,
    unsigned int block_size,
    unsigned int tile_rows)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;

    const unsigned int tile = blockIdx.x;
    const unsigned int query_row = blockIdx.y;
    if (query_row >= query_rows) return;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const bool collect_histogram = coarse_histograms != nullptr;

    __shared__ unsigned short query_tile[16 * 16];
    __shared__ unsigned short key_tiles[8][16 * 16];
    __shared__ float output_tiles[8][16 * 16];
    __shared__ unsigned int physicals[128];
    extern __shared__ unsigned int coarse_histogram[];
    if (collect_histogram) coarse_histogram[threadIdx.x] = 0u;
    for (unsigned int local = threadIdx.x; local < tile_rows; local += blockDim.x) {
        const unsigned int token = tile * tile_rows + local;
        physicals[local] = token < context_rows
            ? (unsigned int)paged_row(block_table, token, block_size)
            : 0u;
    }
    __syncthreads();

    for (unsigned int group = 0; group < tile_rows; group += waves * 16u) {
        const unsigned int local_base = group + wave * 16u;
        float score = 0.0f;
        for (unsigned int head_base = 0; head_base < head_count; head_base += 16u) {
            CFragment sum;
            fill_fragment(sum, 0.0f);
            for (unsigned int column_base = 0; column_base < head_dim; column_base += 16u) {
                for (unsigned int flat = threadIdx.x; flat < 16u * 16u; flat += blockDim.x) {
                    const unsigned int head = head_base + flat / 16u;
                    const unsigned int column = column_base + flat % 16u;
                    query_tile[flat] = zllm_f32_to_bf16(query[
                        ((unsigned long long)query_row * head_count + head) * head_dim
                            + column]);
                }
                for (unsigned int flat = threadIdx.x; flat < waves * 16u * 16u; flat += blockDim.x) {
                    const unsigned int target_wave = flat / (16u * 16u);
                    const unsigned int element = flat % (16u * 16u);
                    const unsigned int local = group + target_wave * 16u + element / 16u;
                    const unsigned int column = column_base + element % 16u;
                    const unsigned int token = tile * tile_rows + local;
                    if (local < tile_rows && token < context_rows) {
                        const unsigned long long physical = physicals[local];
                        const unsigned int groups_per_row = head_dim / key_group_size;
                        const float scale = zllm_bf16_to_f32(
                            key_scales[physical * groups_per_row + column / key_group_size]);
                        key_tiles[target_wave][element] = zllm_f32_to_bf16(
                            (float)keys[physical * head_dim + column] * scale);
                    } else {
                        key_tiles[target_wave][element] = 0u;
                    }
                }
                __syncthreads();
                AFragment q_fragment;
                BFragment k_fragment;
                load_matrix_sync(q_fragment, (const bfloat16_t*)query_tile, 16);
                load_matrix_sync(k_fragment, (const bfloat16_t*)key_tiles[wave], 16);
                mma_sync(sum, q_fragment, k_fragment, sum);
                __syncthreads();
            }
            store_matrix_sync(output_tiles[wave], sum, 16, mem_row_major);
            __syncwarp();
            if (lane < 16u) {
                #pragma unroll
                for (unsigned int head = 0; head < 16u; ++head) {
                    score += head_weights[
                        (unsigned long long)query_row * head_count + head_base + head]
                        * fmaxf(output_tiles[wave][head * 16u + lane], 0.0f);
                }
            }
            __syncwarp();
        }
        if (lane < 16u) {
            const unsigned int local = local_base + lane;
            const unsigned int token = tile * tile_rows + local;
            if (local < tile_rows && token < context_rows
                && token <= query_start + query_row) {
                const unsigned long long stride =
                    (unsigned long long)gridDim.x * tile_rows;
                const unsigned int key = dsa_ordered_score(score);
                score_keys[(unsigned long long)query_row * stride + token] = key;
                if (collect_histogram) atomicAdd(&coarse_histogram[key >> 24], 1u);
            }
        }
    }
    if (collect_histogram) {
        __syncthreads();
        const unsigned int count = coarse_histogram[threadIdx.x];
        if (count != 0u) {
            atomicAdd(&coarse_histograms[(unsigned long long)query_row * 256u + threadIdx.x], count);
        }
    }
}

extern "C" __global__ void dsa_clear_u32(unsigned int* values, unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) values[index] = 0u;
}

extern "C" __global__ void dsa_gather_selection_scores(
    const unsigned int* scores,
    unsigned int* selection,
    unsigned int* selected_scores,
    unsigned int rows,
    unsigned int score_stride,
    unsigned int width,
    unsigned int block_size,
    unsigned int shard_parity,
    unsigned int compact_tokens)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = rows * width;
    if (index >= elements) return;
    const unsigned int row = index / width;
    const unsigned int token = selection[index];
    selected_scores[index] = scores[(unsigned long long)row * score_stride + token];
    if (compact_tokens != 0u) {
        selection[index] = paged_shard_global_token(token, block_size, shard_parity);
    }
}

// 每个 sequence shard 的局部 Top-K 已足以覆盖全局 Top-K。先按
// (score desc, token asc) 精确选出成员，再按 token 升序恢复原 compact
// selection 的稳定消费顺序，避免改变后续 attention 的归约次序。
extern "C" __global__ void dsa_merge_sequence_shard_topk(
    const unsigned int* owner_tokens,
    const unsigned int* owner_scores,
    unsigned int owner_width,
    const unsigned int* peer_tokens,
    const unsigned int* peer_scores,
    unsigned int peer_width,
    unsigned int* output,
    unsigned int rows,
    unsigned int output_width)
{
    constexpr unsigned int capacity = 4096u;
    const unsigned int row = blockIdx.x;
    const unsigned int candidates = owner_width + peer_width;
    if (row >= rows || candidates == 0u || candidates > capacity
        || output_width == 0u || output_width > candidates) return;
    unsigned int padded = 1u;
    while (padded < candidates) padded <<= 1u;
    __shared__ unsigned long long ranks[capacity];
    for (unsigned int index = threadIdx.x; index < padded; index += blockDim.x) {
        unsigned int token = 0xffffffffu;
        unsigned int score = 0u;
        if (index < owner_width) {
            token = owner_tokens[(unsigned long long)row * owner_width + index];
            score = owner_scores[(unsigned long long)row * owner_width + index];
        } else if (index < candidates) {
            const unsigned int peer = index - owner_width;
            token = peer_tokens[(unsigned long long)row * peer_width + peer];
            score = peer_scores[(unsigned long long)row * peer_width + peer];
        }
        ranks[index] = ((unsigned long long)score << 32u)
            | (unsigned long long)(0xffffffffu - token);
    }
    __syncthreads();
    for (unsigned int size = 2u; size <= padded; size <<= 1u) {
        for (unsigned int stride = size >> 1u; stride > 0u; stride >>= 1u) {
            for (unsigned int left = threadIdx.x; left < padded; left += blockDim.x) {
                const unsigned int right = left ^ stride;
                if (right > left) {
                    const bool ascending = (left & size) == 0u;
                    const unsigned long long a = ranks[left];
                    const unsigned long long b = ranks[right];
                    if ((a > b) == ascending) {
                        ranks[left] = b;
                        ranks[right] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    // ranks 目前升序；尾部是全局 Top-K。完整 radix 的稳定输出约定是：
    // 先按 token 顺序写 score 高于阈值的成员，再写阈值同分成员。这里保持
    // 同一顺序，避免 sequence shard 改变后续 attention 的归约次序。
    const unsigned int threshold_score
        = (unsigned int)(ranks[padded - output_width] >> 32u);
    for (unsigned int index = threadIdx.x; index < padded; index += blockDim.x) {
        if (index < output_width) {
            const unsigned long long selected = ranks[padded - 1u - index];
            const unsigned int score = (unsigned int)(selected >> 32u);
            const unsigned int token = 0xffffffffu - (unsigned int)selected;
            ranks[index] = ((unsigned long long)(score == threshold_score) << 32u)
                | token;
        } else {
            ranks[index] = 0xffffffffffffffffull;
        }
    }
    __syncthreads();
    for (unsigned int size = 2u; size <= padded; size <<= 1u) {
        for (unsigned int stride = size >> 1u; stride > 0u; stride >>= 1u) {
            for (unsigned int left = threadIdx.x; left < padded; left += blockDim.x) {
                const unsigned int right = left ^ stride;
                if (right > left) {
                    const bool ascending = (left & size) == 0u;
                    const unsigned long long a = ranks[left];
                    const unsigned long long b = ranks[right];
                    if ((a > b) == ascending) {
                        ranks[left] = b;
                        ranks[right] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (unsigned int index = threadIdx.x; index < output_width; index += blockDim.x) {
        output[(unsigned long long)row * output_width + index]
            = (unsigned int)ranks[index];
    }
}

// coarse histogram 给出最高字节阈值；这里只写阈值桶 token。正常分布的
// candidate scratch 与 context 无关，极端聚集行登记给 persistent fallback。
extern "C" __global__ void dsa_compact_score_prefixes(
    const unsigned char* score_prefixes,
    const unsigned int* coarse_histograms,
    unsigned int* candidate_tokens,
    unsigned int* candidate_counts,
    unsigned int* threshold_bytes,
    unsigned int* overflow_rows,
    unsigned int* overflow_count,
    unsigned int query_rows,
    unsigned int score_stride,
    unsigned int query_start,
    unsigned int top_k,
    unsigned int candidate_capacity)
{
    const unsigned int row = blockIdx.x;
    if (row >= query_rows) return;
    const unsigned int visible = query_start + row + 1u;
    const unsigned int target_count = top_k < visible ? top_k : visible;
    const unsigned long long row_base = (unsigned long long)row * score_stride;
    __shared__ unsigned int threshold;
    __shared__ unsigned int prefix_count;
    __shared__ unsigned int candidate_span;

    if (threadIdx.x == 0u) {
        unsigned int rank = target_count - 1u;
        threshold = 0u;
        prefix_count = 0u;
        for (int bucket = 255; bucket >= 0; --bucket) {
            const unsigned int count
                = coarse_histograms[(unsigned long long)row * 256u + bucket];
            if (rank < count) {
                threshold = (unsigned int)bucket;
                prefix_count = count;
                break;
            }
            rank -= count;
        }
        candidate_counts[row] = prefix_count;
        threshold_bytes[row] = threshold;
        candidate_span = 0u;
        if (prefix_count > candidate_capacity) {
            const unsigned int slot = atomicAdd(overflow_count, 1u);
            overflow_rows[slot] = row;
        }
    }
    __syncthreads();
    if (prefix_count > candidate_capacity) return;

    for (unsigned int token = threadIdx.x; token < visible;
         token += blockDim.x) {
        if ((unsigned int)score_prefixes[row_base + token] == threshold) {
            const unsigned int slot = atomicAdd(&candidate_span, 1u);
            if (slot < candidate_capacity) {
                candidate_tokens[(unsigned long long)row * candidate_capacity + slot]
                    = token;
            }
        }
    }
}

// 正常行仅扫描 1-byte prefix，并对阈值桶读取稀疏完整 score；溢出行由
// persistent scorer 写满 score 后走相同的精确 radix 与稳定 token 顺序。
extern "C" __global__ void dsa_prefix_radix_select_topk(
    const unsigned char* score_prefixes,
    const unsigned int* score_keys,
    const unsigned int* coarse_histograms,
    const unsigned int* candidate_counts,
    const unsigned int* threshold_bytes,
    unsigned int* selection,
    unsigned int query_rows,
    unsigned int score_stride,
    unsigned int query_start,
    unsigned int top_k,
    unsigned int candidate_capacity)
{
    const unsigned int row = blockIdx.x;
    if (row >= query_rows) return;
    const unsigned int visible = query_start + row + 1u;
    const unsigned int target_count = top_k < visible ? top_k : visible;
    const unsigned long long row_base = (unsigned long long)row * score_stride;
    const bool overflow = candidate_counts[row] > candidate_capacity;
    const unsigned int threshold_byte = threshold_bytes[row];

    __shared__ unsigned int histogram[256];
    __shared__ unsigned int packed_wave_offsets[2][8];
    __shared__ unsigned int prefix;
    __shared__ unsigned int prefix_mask;
    __shared__ unsigned int rank;
    __shared__ unsigned int threshold;
    __shared__ unsigned int packed_output_count;
    __shared__ unsigned int packed_output_bases[2];

    if (threadIdx.x == 0u) {
        prefix = overflow ? 0u : threshold_byte << 24;
        prefix_mask = overflow ? 0u : 255u << 24;
        rank = target_count - 1u;
        if (!overflow) {
            for (int bucket = 255; bucket > (int)threshold_byte; --bucket) {
                rank -= coarse_histograms[
                    (unsigned long long)row * 256u + (unsigned int)bucket];
            }
        }
    }
    __syncthreads();

    for (int shift = overflow ? 24 : 16; shift >= 0; shift -= 8) {
        histogram[threadIdx.x] = 0u;
        __syncthreads();
        for (unsigned int token = threadIdx.x; token < visible;
             token += blockDim.x) {
            if (!overflow
                && (unsigned int)score_prefixes[row_base + token]
                    != threshold_byte) {
                continue;
            }
            const unsigned int key = score_keys[row_base + token];
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&histogram[(key >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            for (int bucket = 255; bucket >= 0; --bucket) {
                const unsigned int count = histogram[bucket];
                if (rank < count) {
                    prefix |= (unsigned int)bucket << shift;
                    prefix_mask |= 255u << shift;
                    break;
                }
                rank -= count;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        threshold = prefix;
        packed_output_count = 0u;
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int equal_needed = rank + 1u;
    const unsigned int greater_total = target_count - equal_needed;
    constexpr unsigned int items_per_thread = 4u;
    const unsigned int tile_items = blockDim.x * items_per_thread;
    for (unsigned int base = 0u, tile = 0u; base < visible;
         base += tile_items, ++tile) {
        const unsigned int token = base + threadIdx.x * items_per_thread;
        const unsigned int remaining = token < visible ? visible - token : 0u;
        const unsigned int active = remaining < items_per_thread
            ? remaining : items_per_thread;
        unsigned int greater_mask = 0u;
        unsigned int equal_mask = 0u;
#pragma unroll
        for (unsigned int item = 0u; item < items_per_thread; ++item) {
            if (item >= active) continue;
            const unsigned int current = token + item;
            const unsigned int current_prefix
                = overflow ? threshold_byte
                    : (unsigned int)score_prefixes[row_base + current];
            if (!overflow && current_prefix > threshold_byte) {
                greater_mask |= 1u << item;
            } else if (overflow || current_prefix == threshold_byte) {
                const unsigned int key = score_keys[row_base + current];
                if (key > threshold) greater_mask |= 1u << item;
                else if (key == threshold) equal_mask |= 1u << item;
            }
        }

        const unsigned int packed_count
            = __popc(greater_mask) | (__popc(equal_mask) << 16);
        unsigned int packed_scan = packed_count;
        for (unsigned int offset = 1u; offset < warpSize; offset <<= 1u) {
            const unsigned int previous = __shfl_up(packed_scan, offset);
            if (lane >= offset) packed_scan += previous;
        }
        const unsigned int packed_lane_offset = packed_scan - packed_count;
        const unsigned int buffer = tile & 1u;
        if (lane + 1u == warpSize) {
            packed_wave_offsets[buffer][wave] = packed_scan;
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            unsigned int packed_offset = 0u;
            packed_output_bases[buffer] = packed_output_count;
            for (unsigned int current_wave = 0u; current_wave < waves;
                 ++current_wave) {
                const unsigned int packed_wave_count
                    = packed_wave_offsets[buffer][current_wave];
                packed_wave_offsets[buffer][current_wave] = packed_offset;
                packed_offset += packed_wave_count;
            }
            const unsigned int greater_count
                = (packed_output_count & 0xffffu)
                    + (packed_offset & 0xffffu);
            const unsigned int equal_count
                = (packed_output_count >> 16) + (packed_offset >> 16);
            packed_output_count = greater_count
                | ((equal_count < equal_needed ? equal_count : equal_needed)
                    << 16);
        }
        __syncthreads();

        const unsigned int packed_offset = packed_output_bases[buffer]
            + packed_wave_offsets[buffer][wave] + packed_lane_offset;
        unsigned int greater_position = packed_offset & 0xffffu;
        unsigned int equal_position = packed_offset >> 16;
#pragma unroll
        for (unsigned int item = 0u; item < items_per_thread; ++item) {
            const unsigned int bit = 1u << item;
            if (greater_mask & bit) {
                if (greater_position < greater_total) {
                    selection[(unsigned long long)row * top_k + greater_position]
                        = token + item;
                }
                ++greater_position;
            }
            if (equal_mask & bit) {
                if (equal_position < equal_needed) {
                    selection[(unsigned long long)row * top_k + greater_total
                        + equal_position] = token + item;
                }
                ++equal_position;
            }
        }
    }
}

extern "C" __global__ void dsa_radix_select_topk(
    const unsigned int* score_keys,
    unsigned int* selection,
    unsigned int query_rows,
    unsigned int score_stride,
    unsigned int query_start,
    unsigned int top_k)
{
    const unsigned int row = blockIdx.x;
    if (row >= query_rows) return;
    const unsigned int visible = query_start + row + 1;
    const unsigned int target_count = top_k < visible ? top_k : visible;
    const unsigned long long row_base = (unsigned long long)row * score_stride;
    __shared__ unsigned int histogram[256];
    __shared__ unsigned int offsets[256];
    __shared__ unsigned int prefix;
    __shared__ unsigned int prefix_mask;
    __shared__ unsigned int rank;
    __shared__ unsigned int threshold;
    __shared__ unsigned int greater_count;
    __shared__ unsigned int first_equal;

    if (threadIdx.x == 0) {
        prefix = 0;
        prefix_mask = 0;
        rank = target_count - 1;
    }
    __syncthreads();

    // 直接扫描紧凑 score；避免为每 128 个 score 展开 256 桶直方图。
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[threadIdx.x] = 0;
        __syncthreads();
        for (unsigned int token = threadIdx.x; token < visible; token += blockDim.x) {
            const unsigned int key = score_keys[row_base + token];
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&histogram[(key >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int bucket = 255; bucket >= 0; --bucket) {
                const unsigned int count = histogram[bucket];
                if (rank < count) {
                    prefix |= (unsigned int)bucket << shift;
                    prefix_mask |= 255u << shift;
                    break;
                }
                rank -= count;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) threshold = prefix;
    __syncthreads();

    const unsigned int begin = ((unsigned long long)visible * threadIdx.x) / blockDim.x;
    const unsigned int end = ((unsigned long long)visible * (threadIdx.x + 1)) / blockDim.x;
    if (threadIdx.x == 0) first_equal = 0xffffffffu;
    __syncthreads();
    unsigned int count = 0;
    unsigned int local_first_equal = 0xffffffffu;
    for (unsigned int token = begin; token < end; ++token) {
        const unsigned int key = score_keys[row_base + token];
        count += key > threshold;
        if (key == threshold && local_first_equal == 0xffffffffu) local_first_equal = token;
    }
    if (local_first_equal != 0xffffffffu) atomicMin(&first_equal, local_first_equal);
    offsets[threadIdx.x] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int offset = 0;
        for (unsigned int thread = 0; thread < blockDim.x; ++thread) {
            const unsigned int thread_count = offsets[thread];
            offsets[thread] = offset;
            offset += thread_count;
        }
        greater_count = offset;
    }
    __syncthreads();

    unsigned int output = offsets[threadIdx.x];
    for (unsigned int token = begin; token < end; ++token) {
        if (score_keys[row_base + token] > threshold) {
            selection[(unsigned long long)row * top_k + output++] = token;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int output = greater_count;
        if (output + 1 == target_count) {
            selection[(unsigned long long)row * top_k + output] = first_equal;
        } else {
            for (unsigned int token = 0; token < visible && output < target_count; ++token) {
                if (score_keys[row_base + token] == threshold) {
                    selection[(unsigned long long)row * top_k + output++] = token;
                }
            }
        }
    }
}

// score kernel 已经生成最高字节直方图。这里仅在阈值前缀仍然过大时扫描
// 完整 score；一旦前缀缩到 LDS 容量内，剩余 radix 都只访问候选索引。
// 最后一遍按 token 顺序压紧输出，保持旧 radix kernel 的 selection 顺序。
extern "C" __global__ void dsa_compact_radix_select_topk(
    const unsigned int* score_keys,
    const unsigned int* coarse_histograms,
    unsigned int* selection,
    unsigned int query_rows,
    unsigned int score_stride,
    unsigned int query_start,
    unsigned int top_k,
    unsigned int visibility_divisor)
{
    constexpr unsigned int candidate_capacity = 4096u;
    const unsigned int row = blockIdx.x;
    if (row >= query_rows) return;
    const unsigned int visible = (query_start + row + 1u) / visibility_divisor;
    const unsigned int target_count = top_k < visible ? top_k : visible;
    const unsigned long long row_base = (unsigned long long)row * score_stride;

    __shared__ unsigned int histogram[256];
    __shared__ unsigned int candidates[candidate_capacity];
    __shared__ unsigned int packed_wave_offsets[2][8];
    __shared__ unsigned int prefix;
    __shared__ unsigned int prefix_mask;
    __shared__ unsigned int rank;
    __shared__ unsigned int prefix_count;
    __shared__ unsigned int candidate_span;
    __shared__ unsigned int compacted;
    __shared__ unsigned int threshold;
    __shared__ unsigned int packed_output_count;
    __shared__ unsigned int packed_output_bases[2];

    histogram[threadIdx.x]
        = coarse_histograms[(unsigned long long)row * 256u + threadIdx.x];
    if (threadIdx.x == 0u) {
        prefix = 0u;
        prefix_mask = 0u;
        rank = target_count - 1u;
        candidate_span = 0u;
        compacted = 0u;
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        for (int bucket = 255; bucket >= 0; --bucket) {
            const unsigned int count = histogram[bucket];
            if (rank < count) {
                prefix = (unsigned int)bucket << 24;
                prefix_mask = 255u << 24;
                prefix_count = count;
                break;
            }
            rank -= count;
        }
    }
    __syncthreads();

    for (int shift = 16; shift >= 0; shift -= 8) {
        histogram[threadIdx.x] = 0u;
        const unsigned int gather = !compacted
            && prefix_count <= candidate_capacity && shift != 0;
        if (threadIdx.x == 0u && gather) candidate_span = 0u;
        __syncthreads();

        if (compacted) {
            const unsigned int span = candidate_span;
            for (unsigned int i = threadIdx.x; i < span; i += blockDim.x) {
                const unsigned int token = candidates[i];
                const unsigned int key = score_keys[row_base + token];
                if ((key & prefix_mask) == prefix) {
                    atomicAdd(&histogram[(key >> shift) & 255u], 1u);
                }
            }
        } else {
            for (unsigned int token = threadIdx.x; token < visible; token += blockDim.x) {
                const unsigned int key = score_keys[row_base + token];
                if ((key & prefix_mask) == prefix) {
                    if (gather) {
                        const unsigned int position = atomicAdd(&candidate_span, 1u);
                        if (position < candidate_capacity) candidates[position] = token;
                    }
                    atomicAdd(&histogram[(key >> shift) & 255u], 1u);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0u) {
            for (int bucket = 255; bucket >= 0; --bucket) {
                const unsigned int count = histogram[bucket];
                if (rank < count) {
                    prefix |= (unsigned int)bucket << shift;
                    prefix_mask |= 255u << shift;
                    prefix_count = count;
                    break;
                }
                rank -= count;
            }
            if (gather) compacted = 1u;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        threshold = prefix;
        packed_output_count = 0u;
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int equal_needed = rank + 1u;
    const unsigned int greater_total = target_count - equal_needed;
    constexpr unsigned int items_per_thread = 4u;
    const unsigned int tile_items = blockDim.x * items_per_thread;
    for (unsigned int base = 0u, tile = 0u; base < visible;
         base += tile_items, ++tile) {
        const unsigned int token = base + threadIdx.x * items_per_thread;
        const unsigned int remaining = token < visible ? visible - token : 0u;
        const unsigned int active = remaining < items_per_thread
            ? remaining : items_per_thread;
        uint4 keys = make_uint4(0u, 0u, 0u, 0u);
        // 生产 score stride 天然对齐；通用 oracle 的奇数 stride 退回标量读。
        if (active == items_per_thread && ((row_base + token) & 3ull) == 0ull) {
            keys = *((const uint4*)(score_keys + row_base + token));
        } else {
            if (active > 0u) keys.x = score_keys[row_base + token];
            if (active > 1u) keys.y = score_keys[row_base + token + 1u];
            if (active > 2u) keys.z = score_keys[row_base + token + 2u];
            if (active > 3u) keys.w = score_keys[row_base + token + 3u];
        }
        const unsigned int active_mask = (1u << active) - 1u;
        const unsigned int greater_mask = (
              (keys.x > threshold ? 1u : 0u)
            | (keys.y > threshold ? 2u : 0u)
            | (keys.z > threshold ? 4u : 0u)
            | (keys.w > threshold ? 8u : 0u)) & active_mask;
        const unsigned int equal_mask = (
              (keys.x == threshold ? 1u : 0u)
            | (keys.y == threshold ? 2u : 0u)
            | (keys.z == threshold ? 4u : 0u)
            | (keys.w == threshold ? 8u : 0u)) & active_mask;

        // 低 16 位为 greater、高 16 位为 equal；单次 wave scan 同时生成两类偏移。
        const unsigned int packed_count
            = __popc(greater_mask) | (__popc(equal_mask) << 16);
        unsigned int packed_scan = packed_count;
        for (unsigned int offset = 1u; offset < warpSize; offset <<= 1u) {
            const unsigned int previous = __shfl_up(packed_scan, offset);
            if (lane >= offset) packed_scan += previous;
        }
        const unsigned int packed_lane_offset = packed_scan - packed_count;
        const unsigned int buffer = tile & 1u;
        if (lane + 1u == warpSize) {
            packed_wave_offsets[buffer][wave] = packed_scan;
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            unsigned int packed_offset = 0u;
            packed_output_bases[buffer] = packed_output_count;
            for (unsigned int current_wave = 0u; current_wave < waves; ++current_wave) {
                const unsigned int packed_wave_count
                    = packed_wave_offsets[buffer][current_wave];
                packed_wave_offsets[buffer][current_wave] = packed_offset;
                packed_offset += packed_wave_count;
            }
            const unsigned int greater_count
                = (packed_output_count & 0xffffu) + (packed_offset & 0xffffu);
            const unsigned int equal_count
                = (packed_output_count >> 16) + (packed_offset >> 16);
            packed_output_count = greater_count
                | ((equal_count < equal_needed ? equal_count : equal_needed) << 16);
        }
        __syncthreads();

        const unsigned int packed_offset = packed_output_bases[buffer]
            + packed_wave_offsets[buffer][wave] + packed_lane_offset;
        unsigned int greater_position = packed_offset & 0xffffu;
        unsigned int equal_position = packed_offset >> 16;
#pragma unroll
        for (unsigned int item = 0u; item < items_per_thread; ++item) {
            const unsigned int bit = 1u << item;
            if (greater_mask & bit) {
                if (greater_position < greater_total) {
                    selection[(unsigned long long)row * top_k + greater_position]
                        = token + item;
                }
                ++greater_position;
            }
            if (equal_mask & bit) {
                if (equal_position < equal_needed) {
                    selection[(unsigned long long)row * top_k + greater_total
                        + equal_position] = token + item;
                }
                ++equal_position;
            }
        }
    }
}

// 单行长上下文的精确选择先保留现有 radix 阈值算法，再把最终稳定压紧拆给
// 多个 CTA。阈值只依赖计数，shared candidate 的收集顺序不会影响结果。
// threshold 只负责 byte1（coarse histogram）与 byte2（tile 并行 histogram），
// byte3/byte4 由 dsa_compact_radix_stage 逐级接力（kernel 边界=全局 barrier）。
extern "C" __global__ void dsa_compact_radix_threshold(
    const unsigned int* score_keys,
    const unsigned int* coarse_histograms,
    unsigned int* tile_histograms,
    unsigned int* completed_tiles,
    unsigned int* radix_state,
    unsigned int query_rows,
    unsigned int score_stride,
    unsigned int query_start,
    unsigned int top_k,
    unsigned int tile_count,
    unsigned int tile_rows)
{
    const unsigned int tile = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (row >= query_rows || tile >= tile_count) return;
    const unsigned int visible = query_start + row + 1u;
    const unsigned int target_count = top_k < visible ? top_k : visible;
    const unsigned long long row_base = (unsigned long long)row * score_stride;

    __shared__ unsigned int histogram[256];
    __shared__ unsigned int prefix;
    __shared__ unsigned int prefix_mask;
    __shared__ unsigned int rank;
    __shared__ unsigned int last_tile;

    histogram[threadIdx.x]
        = coarse_histograms[(unsigned long long)row * 256u + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0u) {
        prefix = 0u;
        prefix_mask = 0u;
        rank = target_count - 1u;
        for (int bucket = 255; bucket >= 0; --bucket) {
            const unsigned int count = histogram[bucket];
            if (rank < count) {
                prefix = (unsigned int)bucket << 24;
                prefix_mask = 255u << 24;
                break;
            }
            rank -= count;
        }
    }
    __syncthreads();

    // 128K 以上由单 CTA 扫完整 history 生成第二字节直方图，已经成为
    // exact select 的长度热点。每个 history tile 先独立计数，最后完成的
    // CTA 再归约；计数与原 radix 完全相同，不改变阈值或稳定输出顺序。
    histogram[threadIdx.x] = 0u;
    __syncthreads();
    const unsigned int tile_begin = tile * tile_rows;
    const unsigned int tile_end = tile_begin + tile_rows < visible
        ? tile_begin + tile_rows : visible;
    for (unsigned int token = tile_begin + threadIdx.x; token < tile_end;
         token += blockDim.x) {
        const unsigned int key = score_keys[row_base + token];
        if ((key & prefix_mask) == prefix) {
            atomicAdd(&histogram[(key >> 16) & 255u], 1u);
        }
    }
    __syncthreads();
    tile_histograms[((unsigned long long)row * tile_count + tile) * 256u
        + threadIdx.x] = histogram[threadIdx.x];
    // completed_tiles 是跨 CTA 的发布点。先等 256 个 histogram 写入都
    // 完成，再由 thread 0 做一次 device fence；每线程各自 fence 既不能
    // 保证 thread 0 晚于其他 wave 发布，还会重复执行 256 次全局栅栏。
    __syncthreads();
    if (threadIdx.x == 0u) {
        __threadfence();
        last_tile = atomicAdd(&completed_tiles[row], 1u) + 1u == tile_count;
    }
    __syncthreads();
    if (!last_tile) return;

    unsigned int bucket_count = 0u;
    for (unsigned int current = 0u; current < tile_count; ++current) {
        bucket_count += tile_histograms[
            ((unsigned long long)row * tile_count + current) * 256u
                + threadIdx.x];
    }
    histogram[threadIdx.x] = bucket_count;
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (int bucket = 255; bucket >= 0; --bucket) {
            const unsigned int count = histogram[bucket];
            if (rank < count) {
                prefix |= (unsigned int)bucket << 16;
                prefix_mask |= 255u << 16;
                break;
            }
            rank -= count;
        }
    }
    __syncthreads();

    // byte1/byte2 已定；byte3/byte4 拆给 dsa_compact_radix_stage kernel
    // （kernel 边界即全局 barrier，每级全 tile 并行重扫，替代本 CTA 内
    // O(visible) 的 shift 循环串行尾巴）。last CTA 写出状态并复位
    // completed_tiles，下一级无需额外 clear。
    if (threadIdx.x == 0u) {
        radix_state[(unsigned long long)row * 3u + 0u] = prefix;
        radix_state[(unsigned long long)row * 3u + 1u] = prefix_mask;
        radix_state[(unsigned long long)row * 3u + 2u] = rank;
        completed_tiles[row] = 0u;
    }
}

// radix select 的 byte3/byte4 级：threshold kernel 写出 [prefix, mask, rank]
// 状态后，每级一次本 kernel。全部 tile CTA 并行扫各自 history 段做本字节
// histogram，最后完成的 CTA 归约并决策（只做 256-bin 扫描，无全量过滤），
// 再写出状态、复位 completed_tiles。末级（shift==0）同时落最终
// thresholds/equal_needed。每级决策序列与原单 CTA shift 循环完全相同。
extern "C" __global__ void dsa_compact_radix_stage(
    const unsigned int* score_keys,
    unsigned int* tile_histograms,
    unsigned int* radix_state,
    unsigned int* completed_tiles,
    unsigned int* thresholds,
    unsigned int* equal_needed_values,
    unsigned int query_rows,
    unsigned int score_stride,
    unsigned int query_start,
    unsigned int tile_count,
    unsigned int tile_rows,
    unsigned int shift)
{
    const unsigned int tile = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (row >= query_rows || tile >= tile_count) return;
    const unsigned int visible = query_start + row + 1u;
    const unsigned long long row_base = (unsigned long long)row * score_stride;
    const unsigned int prefix = radix_state[(unsigned long long)row * 3u + 0u];
    const unsigned int prefix_mask = radix_state[(unsigned long long)row * 3u + 1u];
    unsigned int rank = radix_state[(unsigned long long)row * 3u + 2u];

    __shared__ unsigned int histogram[256];
    __shared__ unsigned int last_tile;
    histogram[threadIdx.x] = 0u;
    __syncthreads();
    const unsigned int tile_begin = tile * tile_rows;
    const unsigned int tile_end = tile_begin + tile_rows < visible
        ? tile_begin + tile_rows : visible;
    for (unsigned int token = tile_begin + threadIdx.x; token < tile_end;
         token += blockDim.x) {
        const unsigned int key = score_keys[row_base + token];
        if ((key & prefix_mask) == prefix) {
            atomicAdd(&histogram[(key >> shift) & 255u], 1u);
        }
    }
    __syncthreads();
    tile_histograms[((unsigned long long)row * tile_count + tile) * 256u
        + threadIdx.x] = histogram[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0u) {
        __threadfence();
        last_tile = atomicAdd(&completed_tiles[row], 1u) + 1u == tile_count;
    }
    __syncthreads();
    if (!last_tile) return;

    unsigned int bucket_count = 0u;
    for (unsigned int current = 0u; current < tile_count; ++current) {
        bucket_count += tile_histograms[
            ((unsigned long long)row * tile_count + current) * 256u
                + threadIdx.x];
    }
    histogram[threadIdx.x] = bucket_count;
    __syncthreads();
    if (threadIdx.x == 0u) {
        unsigned int next_prefix = prefix;
        unsigned int next_mask = prefix_mask;
        for (int bucket = 255; bucket >= 0; --bucket) {
            const unsigned int count = histogram[bucket];
            if (rank < count) {
                next_prefix |= (unsigned int)bucket << shift;
                next_mask |= 255u << shift;
                break;
            }
            rank -= count;
        }
        radix_state[(unsigned long long)row * 3u + 0u] = next_prefix;
        radix_state[(unsigned long long)row * 3u + 1u] = next_mask;
        radix_state[(unsigned long long)row * 3u + 2u] = rank;
        completed_tiles[row] = 0u;
        if (shift == 0u) {
            thresholds[row] = next_prefix;
            equal_needed_values[row] = rank + 1u;
        }
    }
}

extern "C" __global__ void dsa_count_selected_tiles(
    const unsigned int* score_keys,
    const unsigned int* thresholds,
    unsigned int* tile_counts,
    unsigned int query_rows,
    unsigned int score_stride,
    unsigned int query_start,
    unsigned int tile_count,
    unsigned int tile_rows)
{
    const unsigned int tile = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (row >= query_rows || tile >= tile_count) return;
    const unsigned int visible = query_start + row + 1u;
    const unsigned long long row_base = (unsigned long long)row * score_stride;
    const unsigned int threshold = thresholds[row];
    const unsigned int items_per_thread
        = (tile_rows + blockDim.x - 1u) / blockDim.x;
    const unsigned int token = tile * tile_rows + threadIdx.x * items_per_thread;
    unsigned int greater = 0u;
    unsigned int equal = 0u;
    for (unsigned int item = 0u; item < items_per_thread; ++item) {
        const unsigned int current = token + item;
        if (current < visible && current < (tile + 1u) * tile_rows) {
            const unsigned int key = score_keys[row_base + current];
            greater += key > threshold;
            equal += key == threshold;
        }
    }
    for (unsigned int offset = warpSize >> 1; offset > 0u; offset >>= 1u) {
        greater += __shfl_down(greater, offset);
        equal += __shfl_down(equal, offset);
    }
    __shared__ unsigned int wave_greater[8];
    __shared__ unsigned int wave_equal[8];
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    if (lane == 0u) {
        wave_greater[wave] = greater;
        wave_equal[wave] = equal;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        unsigned int block_greater = 0u;
        unsigned int block_equal = 0u;
        for (unsigned int current = 0u; current < blockDim.x / warpSize;
             ++current) {
            block_greater += wave_greater[current];
            block_equal += wave_equal[current];
        }
        const unsigned long long output
            = ((unsigned long long)row * tile_count + tile) * 2u;
        tile_counts[output] = block_greater;
        tile_counts[output + 1u] = block_equal;
    }
}

extern "C" __global__ void dsa_scan_selected_tiles(
    const unsigned int* tile_counts,
    unsigned int* tile_offsets,
    unsigned int query_rows,
    unsigned int tile_count)
{
    const unsigned int row = blockIdx.x;
    if (row >= query_rows) return;
    const unsigned int items_per_thread
        = (tile_count + blockDim.x - 1u) / blockDim.x;
    const unsigned int begin = threadIdx.x * items_per_thread;
    const unsigned int end = begin + items_per_thread < tile_count
        ? begin + items_per_thread : tile_count;
    unsigned int local_greater = 0u;
    unsigned int local_equal = 0u;
    for (unsigned int tile = begin; tile < end; ++tile) {
        const unsigned long long index
            = ((unsigned long long)row * tile_count + tile) * 2u;
        local_greater += tile_counts[index];
        local_equal += tile_counts[index + 1u];
    }

    // 先在 wave 内求 inclusive prefix，再只扫描最多 8 个 wave 总和。
    // 每个线程拥有连续 tile 段，因此 exclusive base 仍严格按 token 顺序。
    unsigned int greater_scan = local_greater;
    unsigned int equal_scan = local_equal;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    for (unsigned int offset = 1u; offset < warpSize; offset <<= 1u) {
        const unsigned int previous_greater = __shfl_up(greater_scan, offset);
        const unsigned int previous_equal = __shfl_up(equal_scan, offset);
        if (lane >= offset) {
            greater_scan += previous_greater;
            equal_scan += previous_equal;
        }
    }
    __shared__ unsigned int wave_greater[8];
    __shared__ unsigned int wave_equal[8];
    if (lane + 1u == warpSize) {
        wave_greater[wave] = greater_scan;
        wave_equal[wave] = equal_scan;
    }
    __syncthreads();
    const unsigned int wave_count = blockDim.x / warpSize;
    if (wave == 0u) {
        unsigned int greater = lane < wave_count ? wave_greater[lane] : 0u;
        unsigned int equal = lane < wave_count ? wave_equal[lane] : 0u;
        const unsigned int own_greater = greater;
        const unsigned int own_equal = equal;
        for (unsigned int offset = 1u; offset < warpSize; offset <<= 1u) {
            const unsigned int previous_greater = __shfl_up(greater, offset);
            const unsigned int previous_equal = __shfl_up(equal, offset);
            if (lane >= offset) {
                greater += previous_greater;
                equal += previous_equal;
            }
        }
        if (lane < wave_count) {
            wave_greater[lane] = greater - own_greater;
            wave_equal[lane] = equal - own_equal;
        }
    }
    __syncthreads();

    unsigned int greater = wave_greater[wave] + greater_scan - local_greater;
    unsigned int equal = wave_equal[wave] + equal_scan - local_equal;
    for (unsigned int tile = begin; tile < end; ++tile) {
        const unsigned long long index
            = ((unsigned long long)row * tile_count + tile) * 2u;
        tile_offsets[index] = greater;
        tile_offsets[index + 1u] = equal;
        greater += tile_counts[index];
        equal += tile_counts[index + 1u];
    }
}

extern "C" __global__ void dsa_scatter_selected_tiles(
    const unsigned int* score_keys,
    const unsigned int* thresholds,
    const unsigned int* equal_needed_values,
    const unsigned int* tile_offsets,
    unsigned int* selection,
    unsigned int query_rows,
    unsigned int score_stride,
    unsigned int query_start,
    unsigned int top_k,
    unsigned int tile_count,
    unsigned int tile_rows)
{
    const unsigned int tile = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (row >= query_rows || tile >= tile_count) return;
    const unsigned int visible = query_start + row + 1u;
    const unsigned int target_count = top_k < visible ? top_k : visible;
    const unsigned int threshold = thresholds[row];
    const unsigned int equal_needed = equal_needed_values[row];
    const unsigned int greater_total = target_count - equal_needed;
    const unsigned long long row_base = (unsigned long long)row * score_stride;
    const unsigned long long tile_index
        = ((unsigned long long)row * tile_count + tile) * 2u;
    const unsigned int tile_greater_base = tile_offsets[tile_index];
    const unsigned int tile_equal_base = tile_offsets[tile_index + 1u];
    const unsigned int items_per_thread
        = (tile_rows + blockDim.x - 1u) / blockDim.x;
    const unsigned int token = tile * tile_rows + threadIdx.x * items_per_thread;
    unsigned int greater_mask = 0u;
    unsigned int equal_mask = 0u;
    for (unsigned int item = 0u; item < items_per_thread; ++item) {
        const unsigned int current = token + item;
        if (item < 32u && current < visible && current < (tile + 1u) * tile_rows) {
            const unsigned int key = score_keys[row_base + current];
            if (key > threshold) greater_mask |= 1u << item;
            else if (key == threshold) equal_mask |= 1u << item;
        }
    }

    const unsigned int packed_count
        = __popc(greater_mask) | (__popc(equal_mask) << 16);
    unsigned int packed_scan = packed_count;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    for (unsigned int offset = 1u; offset < warpSize; offset <<= 1u) {
        const unsigned int previous = __shfl_up(packed_scan, offset);
        if (lane >= offset) packed_scan += previous;
    }
    const unsigned int packed_lane_offset = packed_scan - packed_count;
    __shared__ unsigned int wave_offsets[8];
    if (lane + 1u == warpSize) wave_offsets[wave] = packed_scan;
    __syncthreads();
    if (threadIdx.x == 0u) {
        unsigned int offset = 0u;
        for (unsigned int current = 0u; current < blockDim.x / warpSize;
             ++current) {
            const unsigned int count = wave_offsets[current];
            wave_offsets[current] = offset;
            offset += count;
        }
    }
    __syncthreads();

    const unsigned int packed_offset
        = wave_offsets[wave] + packed_lane_offset;
    unsigned int greater = tile_greater_base + (packed_offset & 0xffffu);
    unsigned int equal = tile_equal_base + (packed_offset >> 16);
    for (unsigned int item = 0u; item < items_per_thread; ++item) {
        const unsigned int bit = item < 32u ? 1u << item : 0u;
        if (greater_mask & bit) {
            if (greater < greater_total) {
                selection[(unsigned long long)row * top_k + greater] = token + item;
            }
            ++greater;
        }
        if (equal_mask & bit) {
            if (equal < equal_needed) {
                selection[(unsigned long long)row * top_k + greater_total + equal]
                    = token + item;
            }
            ++equal;
        }
    }
}

// 选中的池索引直接在 GPU 上展开回原 token 索引，并追加不完整
// 尾池。输出宽度对齐 CPU oracle 的 top_k+kpool-1，剩余位置填 0。
extern "C" __global__ void dsa_expand_kpool_selection(
    const unsigned int* pool_selection,
    unsigned int* token_selection,
    unsigned int query_rows,
    unsigned int query_start,
    unsigned int pool_top_k,
    unsigned int kpool,
    unsigned int output_width)
{
    const unsigned long long index
        = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long elements = (unsigned long long)query_rows * output_width;
    if (index >= elements) return;
    const unsigned int row = index / output_width;
    const unsigned int slot = index - (unsigned long long)row * output_width;
    const unsigned int visible = query_start + row + 1u;
    const unsigned int complete_pools = visible / kpool;
    const unsigned int selected_pools
        = pool_top_k < complete_pools ? pool_top_k : complete_pools;
    const unsigned int selected_tokens = selected_pools * kpool;
    const unsigned int tail = visible - complete_pools * kpool;
    unsigned int token = 0u;
    if (slot < selected_tokens) {
        const unsigned int pool = pool_selection[
            (unsigned long long)row * pool_top_k + slot / kpool];
        token = pool * kpool + slot % kpool;
    } else if (slot < selected_tokens + tail) {
        token = visible - tail + slot - selected_tokens;
    }
    token_selection[index] = token;
}

extern "C" __global__ void mla_absorb_query_ct(
    const float* query,
    const unsigned int* packed,
    const unsigned char* scales,
    unsigned short* absorbed,
    unsigned int query_rows,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int kv_head_dim,
    unsigned int latent_dim,
    unsigned int rope_dim,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits)
{
    if (query_rows == 2u && bits == 8u && group_size == 32u) {
        const unsigned int nope_dim = q_head_dim - rope_dim;
        if ((latent_dim & 63u) == 0u && (nope_dim & 15u) == 0u
            && nope_dim <= 1024u) {
            const unsigned int latent_quads = latent_dim / 64u;
            const unsigned int head = blockIdx.x / latent_quads;
            const unsigned int quad = blockIdx.x % latent_quads;
            if (head >= head_count) return;
            const unsigned int group_lane = threadIdx.x >> 4u;
            const unsigned int lane = threadIdx.x & 15u;
            const unsigned int col0 = quad * 64u + group_lane * 4u;
            const unsigned int groups_per_row = latent_dim >> 5u;
            const unsigned int row_words = latent_dim >> 2u;
            __shared__ float query_tile[2][1024];
            for (unsigned int index = threadIdx.x;
                 index < 2u * nope_dim;
                 index += blockDim.x) {
                const unsigned int query_row = index / nope_dim;
                const unsigned int column = index - query_row * nope_dim;
                query_tile[query_row][column] = query[
                    ((unsigned long long)query_row * head_count + head)
                        * q_head_dim + column];
            }
            __syncthreads();
            float acc[2][4] = {};
            unsigned int row = lane;
            for (; row + 48u < nope_dim; row += 64u) {
                const unsigned long long weight_row0 =
                    (unsigned long long)head * kv_head_dim + row;
                const unsigned int words[4] = {
                    packed[weight_row0 * row_words + (col0 >> 2u)],
                    packed[(weight_row0 + 16u) * row_words + (col0 >> 2u)],
                    packed[(weight_row0 + 32u) * row_words + (col0 >> 2u)],
                    packed[(weight_row0 + 48u) * row_words + (col0 >> 2u)]
                };
                const float row_scales[4] = {
                    zllm_scale(scales, weight_row0 * groups_per_row + (col0 >> 5u), scale_dtype),
                    zllm_scale(scales, (weight_row0 + 16u) * groups_per_row + (col0 >> 5u), scale_dtype),
                    zllm_scale(scales, (weight_row0 + 32u) * groups_per_row + (col0 >> 5u), scale_dtype),
                    zllm_scale(scales, (weight_row0 + 48u) * groups_per_row + (col0 >> 5u), scale_dtype)
                };
                #pragma unroll
                for (unsigned int source = 0u; source < 4u; ++source) {
                    const float weights[4] = {
                        (float)((int)(words[source] & 255u) - 128) * row_scales[source],
                        (float)((int)((words[source] >> 8u) & 255u) - 128) * row_scales[source],
                        (float)((int)((words[source] >> 16u) & 255u) - 128) * row_scales[source],
                        (float)((int)(words[source] >> 24u) - 128) * row_scales[source]
                    };
                    #pragma unroll
                    for (unsigned int query_row = 0u; query_row < 2u; ++query_row) {
                        const float value = query_tile[query_row][row + source * 16u];
                        #pragma unroll
                        for (unsigned int output = 0u; output < 4u; ++output)
                            acc[query_row][output] += value * weights[output];
                    }
                }
            }
            for (; row < nope_dim; row += 16u) {
                const unsigned long long weight_row =
                    (unsigned long long)head * kv_head_dim + row;
                const unsigned int word =
                    packed[weight_row * row_words + (col0 >> 2u)];
                const float scale = zllm_scale(
                    scales, weight_row * groups_per_row + (col0 >> 5u), scale_dtype);
                const float weights[4] = {
                    (float)((int)(word & 255u) - 128) * scale,
                    (float)((int)((word >> 8u) & 255u) - 128) * scale,
                    (float)((int)((word >> 16u) & 255u) - 128) * scale,
                    (float)((int)(word >> 24u) - 128) * scale
                };
                #pragma unroll
                for (unsigned int query_row = 0u; query_row < 2u; ++query_row) {
                    const float value = query_tile[query_row][row];
                    #pragma unroll
                    for (unsigned int output = 0u; output < 4u; ++output)
                        acc[query_row][output] += value * weights[output];
                }
            }
            #pragma unroll
            for (unsigned int query_row = 0u; query_row < 2u; ++query_row) {
                #pragma unroll
                for (unsigned int output = 0u; output < 4u; ++output) {
                    for (unsigned int offset = 8u; offset > 0u; offset >>= 1u)
                        acc[query_row][output] += __shfl_down(
                            acc[query_row][output], offset, 16);
                }
            }
            if (lane == 0u) {
                #pragma unroll
                for (unsigned int query_row = 0u; query_row < 2u; ++query_row) {
                    const unsigned long long out_base =
                        ((unsigned long long)query_row * head_count + head)
                            * latent_dim + col0;
                    #pragma unroll
                    for (unsigned int output = 0u; output < 4u; ++output)
                        absorbed[out_base + output]
                            = zllm_f32_to_bf16(acc[query_row][output]);
                }
            }
            return;
        }
    }
    // Decode 只有一行，16-lane/output 避免执行 15 行无效 tile FMA。
    if (query_rows == 1u) {
        const unsigned int nope_dim = q_head_dim - rope_dim;
        // W8G32 向量化臂：每 16-lane 组产 4 连续 latent 输出列，同 block 16 组
        // 在同一迭代读同一权重行的 16 个连续 word（两条完整 cache line），
        // 字节量 4×↓ 且 L2 完全合并。逐输出归约顺序与 F32 路径相同（逐位一致）。
        if (bits == 8u && group_size == 32u && (latent_dim & 63u) == 0u
            && (nope_dim & 15u) == 0u && nope_dim <= 1024u) {
            // grid.x 必须由调用方按 heads × latent/64 发射。
            const unsigned int latent_quads = latent_dim / 64u;
            const unsigned int head = blockIdx.x / latent_quads;
            const unsigned int quad = blockIdx.x % latent_quads;
            if (head >= head_count) return;
            const unsigned int group_lane = threadIdx.x >> 4u;
            const unsigned int lane = threadIdx.x & 15u;
            const unsigned int col0 = quad * 64u + group_lane * 4u;
            const unsigned int groups_per_row = latent_dim >> 5u;
            const unsigned int row_words = latent_dim >> 2u;
            __shared__ float query_tile[1024];
            for (unsigned int index = threadIdx.x; index < nope_dim; index += blockDim.x) {
                query_tile[index] = query[(unsigned long long)head * q_head_dim + index];
            }
            __syncthreads();
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            // 行主循环 4 路展开：原形态每迭代 1 次权重装载进 1 条依赖 FMA 链，
            // nope=192 时每 lane 仅 12 次装载，延迟逐次暴露（bench 14.3µs 只有
            // 440GB/s）。4 行装载批量发射后再按行序 FMA——累加顺序不变，逐位一致。
            unsigned int row = lane;
            for (; row + 48u < nope_dim; row += 64u) {
                const unsigned long long weight_row0 =
                    (unsigned long long)head * kv_head_dim + row;
                const unsigned int word0 = packed[weight_row0 * row_words + (col0 >> 2u)];
                const unsigned int word1 = packed[(weight_row0 + 16u) * row_words + (col0 >> 2u)];
                const unsigned int word2 = packed[(weight_row0 + 32u) * row_words + (col0 >> 2u)];
                const unsigned int word3 = packed[(weight_row0 + 48u) * row_words + (col0 >> 2u)];
                const float scale0 = zllm_scale(
                    scales, weight_row0 * groups_per_row + (col0 >> 5u), scale_dtype);
                const float scale1 = zllm_scale(
                    scales, (weight_row0 + 16u) * groups_per_row + (col0 >> 5u), scale_dtype);
                const float scale2 = zllm_scale(
                    scales, (weight_row0 + 32u) * groups_per_row + (col0 >> 5u), scale_dtype);
                const float scale3 = zllm_scale(
                    scales, (weight_row0 + 48u) * groups_per_row + (col0 >> 5u), scale_dtype);
                const float q0 = query_tile[row];
                const float q1 = query_tile[row + 16u];
                const float q2 = query_tile[row + 32u];
                const float q3 = query_tile[row + 48u];
                acc0 += q0 * ((float)((int)(word0 & 255u) - 128) * scale0);
                acc1 += q0 * ((float)((int)((word0 >> 8u) & 255u) - 128) * scale0);
                acc2 += q0 * ((float)((int)((word0 >> 16u) & 255u) - 128) * scale0);
                acc3 += q0 * ((float)((int)(word0 >> 24u) - 128) * scale0);
                acc0 += q1 * ((float)((int)(word1 & 255u) - 128) * scale1);
                acc1 += q1 * ((float)((int)((word1 >> 8u) & 255u) - 128) * scale1);
                acc2 += q1 * ((float)((int)((word1 >> 16u) & 255u) - 128) * scale1);
                acc3 += q1 * ((float)((int)(word1 >> 24u) - 128) * scale1);
                acc0 += q2 * ((float)((int)(word2 & 255u) - 128) * scale2);
                acc1 += q2 * ((float)((int)((word2 >> 8u) & 255u) - 128) * scale2);
                acc2 += q2 * ((float)((int)((word2 >> 16u) & 255u) - 128) * scale2);
                acc3 += q2 * ((float)((int)(word2 >> 24u) - 128) * scale2);
                acc0 += q3 * ((float)((int)(word3 & 255u) - 128) * scale3);
                acc1 += q3 * ((float)((int)((word3 >> 8u) & 255u) - 128) * scale3);
                acc2 += q3 * ((float)((int)((word3 >> 16u) & 255u) - 128) * scale3);
                acc3 += q3 * ((float)((int)(word3 >> 24u) - 128) * scale3);
            }
            for (; row < nope_dim; row += 16u) {
                const unsigned long long weight_row =
                    (unsigned long long)head * kv_head_dim + row;
                const unsigned int word = packed[weight_row * row_words + (col0 >> 2u)];
                const float scale = zllm_scale(
                    scales, weight_row * groups_per_row + (col0 >> 5u), scale_dtype);
                const float q = query_tile[row];
                acc0 += q * ((float)((int)(word & 255u) - 128) * scale);
                acc1 += q * ((float)((int)((word >> 8u) & 255u) - 128) * scale);
                acc2 += q * ((float)((int)((word >> 16u) & 255u) - 128) * scale);
                acc3 += q * ((float)((int)(word >> 24u) - 128) * scale);
            }
            for (unsigned int offset = 8u; offset > 0u; offset >>= 1u) {
                acc0 += __shfl_down(acc0, offset, 16);
                acc1 += __shfl_down(acc1, offset, 16);
                acc2 += __shfl_down(acc2, offset, 16);
                acc3 += __shfl_down(acc3, offset, 16);
            }
            if (lane == 0u) {
                const unsigned long long out_base =
                    (unsigned long long)head * latent_dim + col0;
                absorbed[out_base] = zllm_f32_to_bf16(acc0);
                absorbed[out_base + 1u] = zllm_f32_to_bf16(acc1);
                absorbed[out_base + 2u] = zllm_f32_to_bf16(acc2);
                absorbed[out_base + 3u] = zllm_f32_to_bf16(acc3);
            }
            return;
        }
        const unsigned int latent_tiles = (latent_dim + 15u) / 16u;
        const unsigned int head = blockIdx.x / latent_tiles;
        const unsigned int latent_column = (blockIdx.x % latent_tiles) * 16u + (threadIdx.x >> 4u);
        const unsigned int lane = threadIdx.x & 15u;
        float sum = 0.0f;
        if (latent_column < latent_dim) {
            for (unsigned int column = lane; column < nope_dim; column += 16u) {
                sum += query[(unsigned long long)head * q_head_dim + column]
                    * paged_weight(
                        packed, scales, head * kv_head_dim + column,
                        latent_column, latent_dim, group_size, scale_dtype, bits);
            }
            for (unsigned int offset = 8u; offset > 0u; offset >>= 1u)
                sum += __shfl_down(sum, offset, 16);
            if (lane == 0u)
                absorbed[(unsigned long long)head * latent_dim + latent_column]
                    = zllm_f32_to_bf16(sum);
        }
        return;
    }

    const unsigned int nope_dim = q_head_dim - rope_dim;
    const unsigned int latent_tiles = (latent_dim + 15) / 16;
    const unsigned int head = blockIdx.x / latent_tiles;
    const unsigned int latent_base = (blockIdx.x % latent_tiles) * 16;
    const unsigned int query_local = threadIdx.x / 16;
    const unsigned int latent_local = threadIdx.x % 16;
    const unsigned int row = blockIdx.y * 16 + query_local;
    const unsigned int latent_column = latent_base + latent_local;
    __shared__ float query_tile[256];
    __shared__ float weight_tile[256];
    float sum = 0.0f;
    for (unsigned int column_base = 0; column_base < nope_dim; column_base += 16) {
        const unsigned int query_column = column_base + latent_local;
        query_tile[threadIdx.x] = row < query_rows && query_column < nope_dim
            ? query[((unsigned long long)row * head_count + head) * q_head_dim + query_column]
            : 0.0f;
        const unsigned int weight_column = column_base + query_local;
        weight_tile[threadIdx.x] = latent_column < latent_dim && weight_column < nope_dim
            ? paged_weight(
                packed, scales, head * kv_head_dim + weight_column,
                latent_column, latent_dim, group_size, scale_dtype, bits)
            : 0.0f;
        __syncthreads();
        #pragma unroll
        for (unsigned int column = 0; column < 16; ++column) {
            sum += query_tile[query_local * 16 + column] * weight_tile[column * 16 + latent_local];
        }
        __syncthreads();
    }
    if (row < query_rows && latent_column < latent_dim) {
        absorbed[((unsigned long long)row * head_count + head) * latent_dim + latent_column]
            = zllm_f32_to_bf16(sum);
    }
}


#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) \
    || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) \
    || defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx11_generic__)
#define ZLLM_PAGED_WMMA_BF16 1
typedef short zllm_wmma_short16 __attribute__((ext_vector_type(16)));
typedef float zllm_wmma_float8 __attribute__((ext_vector_type(8)));
typedef __bf16 zllm_wmma_bf16x2 __attribute__((ext_vector_type(2)));
union zllm_wmma_bf16_pair {
    unsigned int packed;
    zllm_wmma_bf16x2 values;
};
__device__ __forceinline__ zllm_wmma_short16 paged_wmma_expand_bf16(
    zllm_wmma_short16 values)
{
    #pragma unroll
    for (unsigned int index = 0; index < 8; index += 2) {
        int packed = (int)((unsigned int)(unsigned short)values[index]
            | ((unsigned int)(unsigned short)values[index + 1] << 16));
        packed = __builtin_amdgcn_ds_swizzle(packed, 0x401f);
        values[8 + index] = (short)((unsigned int)packed & 0xffffu);
        values[9 + index] = (short)((unsigned int)packed >> 16);
    }
    return values;
}
extern "C" __global__ void zllm_paged_dense_wmma_marker() {}
#endif

__device__ __forceinline__ unsigned int paged_latent_bf16_pair(
    const unsigned short* cache,
    const unsigned short* scales,
    unsigned long long physical,
    unsigned int column,
    unsigned int latent_dim,
    unsigned int group_size)
{
    if (group_size == 0) {
        const unsigned short* value = cache + physical * latent_dim + column;
        return column + 1 < latent_dim
            ? *reinterpret_cast<const unsigned int*>(value)
            : value[0];
    }
    const signed char* codes = reinterpret_cast<const signed char*>(cache);
    const unsigned int groups = latent_dim / group_size;
    const unsigned long long base = physical * latent_dim + column;
    const unsigned long long scale_base = physical * groups;
    const unsigned int group0 = column / group_size;
    const float scale0 = zllm_bf16_to_f32(scales[scale_base + group0]);
    const float value0 = (float)codes[base] * scale0;
    const unsigned int group1 = (column + 1) / group_size;
    const float scale1 = group1 == group0
        ? scale0
        : zllm_bf16_to_f32(scales[scale_base + group1]);
    const float value1 = column + 1 < latent_dim
        ? (float)codes[base + 1] * scale1
        : 0.0f;
    return (unsigned int)zllm_f32_to_bf16(value0)
        | ((unsigned int)zllm_f32_to_bf16(value1) << 16);
}

__device__ __forceinline__ unsigned int paged_latent_bf16_pair_q8g64(
    const unsigned short* cache,
    const unsigned short* scales,
    unsigned long long physical,
    unsigned int column,
    unsigned int latent_dim)
{
    const signed char* codes = reinterpret_cast<const signed char*>(cache);
    const unsigned long long base = physical * latent_dim + column;
    const unsigned long long scale_base = physical * (latent_dim >> 6);
    const float scale = zllm_bf16_to_f32(scales[scale_base + (column >> 6)]);
    const float value0 = (float)codes[base] * scale;
    const float value1 = (float)codes[base + 1] * scale;
    return (unsigned int)zllm_f32_to_bf16(value0)
        | ((unsigned int)zllm_f32_to_bf16(value1) << 16);
}

extern "C" __global__ void mla_paged_dense_latent_f32(

    const float* query,
    const unsigned short* absorbed,
    const unsigned short* latent_cache,
    const unsigned short* latent_scales,
    const unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* selection,
    const unsigned int* selection_counts,
    unsigned short* weighted_latent,
    float* split_partial,
    float* split_stats,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int latent_dim,
    unsigned int latent_group_size,
    unsigned int rope_dim,
    unsigned int top_k,
    unsigned int block_size,
    unsigned int selected,
    unsigned int split_count,
    unsigned int split_size,
    unsigned int shard_parity)
{
    const unsigned int head_base = blockIdx.x * 8;
#ifdef ZLLM_PAGED_WMMA_BF16
    const bool dense_grid = !selected && query_rows > 1
        && (latent_dim & 15) == 0 && (rope_dim & 15) == 0
        && (latent_group_size == 0 || (latent_group_size & 15) == 0)
        && split_count > 0
        && gridDim.y == ((query_rows + 1) / 2) * split_count;
    const unsigned int row = dense_grid
        ? (blockIdx.y / split_count) * 2
        : blockIdx.y;
#else
    const unsigned int row = blockIdx.y;
#endif
    const unsigned int lane = threadIdx.x;
    if (head_base >= head_count || row >= query_rows) return;
    const unsigned int visible = query_start + row + 1;
    // selection 只在候选数超过 top-k 时有定义；同一 prefill batch 的前缀行
    // 仍需直接遍历完整 causal 范围，不能读取尚未填满的 selection 槽。
    const bool use_selection = selected && top_k < visible;
    const unsigned int count = use_selection
        ? (selection_counts == nullptr ? top_k : selection_counts[row])
        : visible;
    __shared__ float scores[16 * 128];
    float* weights = scores;
    __shared__ unsigned int physicals[128];
    __shared__ float alpha[16];
    __shared__ float denominator[16];
    __shared__ float maximum[16];
    extern __shared__ unsigned char query_cache[];
    unsigned short* absorbed_query = reinterpret_cast<unsigned short*>(query_cache);
    float* rope_query = reinterpret_cast<float*>(absorbed_query + 8 * latent_dim);
    const unsigned int nope_dim = q_head_dim - rope_dim;

    // Dense prefill 使用原生 WMMA 计算两行、八头，跨 query 复用同一份分页 K/V。
#ifdef ZLLM_PAGED_WMMA_BF16
    if (dense_grid) {
        const unsigned int split = blockIdx.y % split_count;
        const unsigned int pair_rows = row + 1 < query_rows ? 2 : 1;
        const unsigned int visible_count[2] = {count, count + (pair_rows - 1)};
        const unsigned int split_start = split * split_size;
        const unsigned int split_limit = split_start + split_size
            < visible_count[pair_rows - 1]
            ? split_start + split_size
            : visible_count[pair_rows - 1];
        const unsigned int dense_count = split_limit > split_start
            ? split_limit - split_start
            : 0;
        if (lane < 16) {
            denominator[lane] = 0.0f;
            maximum[lane] = -3.402823466e+38F;
        }
        // 第一行 latent query 留在 LDS；第二行直接读只读全局缓存，避免扩大动态 LDS 接口。
        for (unsigned int index = lane; index < 8 * latent_dim; index += blockDim.x) {
            const unsigned int head = index / latent_dim;
            const unsigned int column = index - head * latent_dim;
            absorbed_query[index] = head_base + head < head_count
                ? absorbed[((unsigned long long)row * head_count + head_base + head)
                    * latent_dim + column]
                : 0;
        }
        // 原 F32 rope scratch 的字节数正好容纳两行 BF16 rope query。
        unsigned short* dense_rope_query = absorbed_query + 8 * latent_dim;
        for (unsigned int index = lane; index < 16 * rope_dim; index += blockDim.x) {
            const unsigned int combo = index / rope_dim;
            const unsigned int query_offset = combo >> 3;
            const unsigned int head = combo & 7;
            const unsigned int column = index - combo * rope_dim;
            dense_rope_query[index] = query_offset < pair_rows && head_base + head < head_count
                ? zllm_f32_to_bf16(query[((unsigned long long)(row + query_offset)
                    * head_count + head_base + head) * q_head_dim + nope_dim + column])
                : 0;
        }
        __syncthreads();

        float dense_acc0[16] = {};
        float dense_acc1[16] = {};
        const unsigned int dense_wave = lane >> 5;
        const unsigned int dense_wave_lane = lane & 31;
        const unsigned int dense_column = dense_wave_lane & 15;
        const unsigned int dense_query_combo = dense_wave_lane & 15;
        const unsigned int dense_dimension0 = lane * 2;
        const unsigned int dense_dimension1 = dense_dimension0 + 1;
        const float dense_scale = rsqrtf((float)q_head_dim);
        // score 生命周期结束后按 combo 原地压缩为 BF16 softmax 权重。
        unsigned short* dense_weights = reinterpret_cast<unsigned short*>(scores);

        for (unsigned int candidate_base = 0; candidate_base < dense_count; candidate_base += 128) {
            const unsigned int tile_count = dense_count - candidate_base < 128
                ? dense_count - candidate_base : 128;
            if (lane < tile_count) {
                physicals[lane] = (unsigned int)paged_shard_row(
                    block_table, split_start + candidate_base + lane, block_size, shard_parity);
            }
            __syncthreads();

            const unsigned int candidate_slot = dense_wave * 16 + dense_column;
            const unsigned int candidate = candidate_base + candidate_slot;
            const unsigned int token = split_start + candidate;
            const unsigned long long physical
                = candidate_slot < tile_count ? physicals[candidate_slot] : 0;
            zllm_wmma_float8 score_fragment
                = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            const unsigned int query_offset = dense_query_combo >> 3;
            const unsigned int query_head = dense_query_combo & 7;
            for (unsigned int column = 0; column < latent_dim; column += 16) {
                zllm_wmma_short16 query_fragment = {};
                zllm_wmma_short16 cache_fragment = {};
                if (query_offset < pair_rows && head_base + query_head < head_count) {
                    const unsigned short* query_ptr = query_offset == 0
                        ? absorbed_query + query_head * latent_dim + column
                        : absorbed + ((unsigned long long)(row + 1) * head_count
                            + head_base + query_head) * latent_dim + column;
                    query_fragment
                        = *reinterpret_cast<const zllm_wmma_short16*>(query_ptr);
                }
                if (candidate < dense_count) {
                    if (latent_group_size == 0) {
                        cache_fragment = *reinterpret_cast<const zllm_wmma_short16*>(
                            latent_cache + physical * latent_dim + column);
                    } else {
                        const signed char* codes
                            = reinterpret_cast<const signed char*>(latent_cache);
                        const unsigned int groups = latent_dim / latent_group_size;
                        const float scale = zllm_bf16_to_f32(latent_scales[
                            physical * groups + column / latent_group_size]);
                        const unsigned long long code_base
                            = physical * latent_dim + column;
                        #pragma unroll
                        for (unsigned int index = 0; index < 16; ++index) {
                            cache_fragment[index] = (short)zllm_f32_to_bf16(
                                (float)codes[code_base + index] * scale);
                        }
                    }
                }
                score_fragment = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    query_fragment, cache_fragment, score_fragment);
            }
            for (unsigned int column = 0; column < rope_dim; column += 16) {
                zllm_wmma_short16 query_fragment = {};
                zllm_wmma_short16 cache_fragment = {};
                if (query_offset < pair_rows && head_base + query_head < head_count) {
                    query_fragment = *reinterpret_cast<const zllm_wmma_short16*>(
                        dense_rope_query + dense_query_combo * rope_dim + column);
                }
                if (candidate < dense_count) {
                    cache_fragment = *reinterpret_cast<const zllm_wmma_short16*>(
                        rope_cache + physical * rope_dim + column);
                }
                score_fragment = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    query_fragment, cache_fragment, score_fragment);
            }
            if (candidate < dense_count) {
                const unsigned int parity = dense_wave_lane >> 4;
                #pragma unroll
                for (unsigned int index = 0; index < 8; ++index) {
                    const unsigned int combo = index * 2 + parity;
                    const unsigned int combo_row = combo >> 3;
                    const unsigned int head = combo & 7;
                    scores[combo * 128 + candidate_slot]
                        = combo_row < pair_rows && token < visible_count[combo_row]
                            && paged_shard_owns(token, block_size, shard_parity)
                            && head_base + head < head_count
                        ? score_fragment[index] * dense_scale
                        : -3.402823466e+38F;
                }
            }
            __syncthreads();

            // 八个 wave 各负责两个 query-head combo 的在线 softmax。
            #pragma unroll
            for (unsigned int combo_part = 0; combo_part < 2; ++combo_part) {
                const unsigned int combo = dense_wave + combo_part * 8;
                float tile_max = -3.402823466e+38F;
                for (unsigned int slot = dense_wave_lane; slot < tile_count; slot += 32) {
                    tile_max = fmaxf(tile_max, scores[combo * 128 + slot]);
                }
                for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                    tile_max = fmaxf(tile_max, __shfl_down(tile_max, offset, 32));
                }
                if (dense_wave_lane == 0) {
                    const float next_max = fmaxf(maximum[combo], tile_max);
                    alpha[combo] = expf(maximum[combo] - next_max);
                    maximum[combo] = next_max;
                }
            }
            __syncthreads();

            #pragma unroll
            for (unsigned int combo_part = 0; combo_part < 2; ++combo_part) {
                const unsigned int combo = dense_wave + combo_part * 8;
                float local_weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float weight_sum = 0.0f;
                #pragma unroll
                for (unsigned int part = 0; part < 4; ++part) {
                    const unsigned int slot = dense_wave_lane + part * 32;
                    if (slot < tile_count) {
                        const float score = scores[combo * 128 + slot];
                        const float weight = score > -3.0e+38F
                            ? expf(score - maximum[combo])
                            : 0.0f;
                        local_weights[part] = weight;
                        weight_sum += weight;
                    }
                }
                __syncwarp();
                #pragma unroll
                for (unsigned int part = 0; part < 4; ++part) {
                    const unsigned int slot = dense_wave_lane + part * 32;
                    if (slot < tile_count) {
                        dense_weights[combo * 256 + slot]
                            = zllm_f32_to_bf16(local_weights[part]);
                    }
                }
                for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                    weight_sum += __shfl_down(weight_sum, offset, 32);
                }
                if (dense_wave_lane == 0) {
                    denominator[combo]
                        = denominator[combo] * alpha[combo] + weight_sum;
                }
            }
            __syncthreads();

            if (dense_dimension0 < latent_dim) {
                #pragma unroll
                for (unsigned int combo = 0; combo < 16; ++combo) {
                    dense_acc0[combo] *= alpha[combo];
                    dense_acc1[combo] *= alpha[combo];
                }
                if (tile_count == 128u) {
                    for (unsigned int slot = 0; slot < 128u; slot += 2u) {
                        const unsigned int cached0 = paged_latent_bf16_pair(
                            latent_cache, latent_scales, physicals[slot],
                            dense_dimension0, latent_dim, latent_group_size);
                        const unsigned int cached1 = paged_latent_bf16_pair(
                            latent_cache, latent_scales, physicals[slot + 1u],
                            dense_dimension0, latent_dim, latent_group_size);
                        zllm_wmma_bf16_pair value0;
                        zllm_wmma_bf16_pair value1;
                        value0.packed = (cached0 & 0xffffu) | (cached1 << 16);
                        value1.packed = (cached0 >> 16) | (cached1 & 0xffff0000u);
                        #pragma unroll
                        for (unsigned int combo = 0; combo < 16; ++combo) {
                            const unsigned int weight = *reinterpret_cast<const unsigned int*>(
                                dense_weights + combo * 256 + slot);
                            zllm_wmma_bf16_pair weight_pair;
                            weight_pair.packed = weight;
                            dense_acc0[combo] = __builtin_amdgcn_fdot2_f32_bf16(
                                weight_pair.values, value0.values, dense_acc0[combo], false);
                            dense_acc1[combo] = __builtin_amdgcn_fdot2_f32_bf16(
                                weight_pair.values, value1.values, dense_acc1[combo], false);
                        }
                    }
                } else {
                    for (unsigned int slot = 0; slot < tile_count; slot += 2u) {
                        const unsigned int cached0 = paged_latent_bf16_pair(
                            latent_cache, latent_scales, physicals[slot],
                            dense_dimension0, latent_dim, latent_group_size);
                        const unsigned int cached1 = slot + 1u < tile_count
                            ? paged_latent_bf16_pair(
                                latent_cache, latent_scales, physicals[slot + 1u],
                                dense_dimension0, latent_dim, latent_group_size)
                            : 0u;
                        zllm_wmma_bf16_pair value0;
                        zllm_wmma_bf16_pair value1;
                        value0.packed = (cached0 & 0xffffu) | (cached1 << 16);
                        value1.packed = (cached0 >> 16) | (cached1 & 0xffff0000u);
                        #pragma unroll
                        for (unsigned int combo = 0; combo < 16; ++combo) {
                            const unsigned short* weight_ptr
                                = dense_weights + combo * 256 + slot;
                            zllm_wmma_bf16_pair weight;
                            weight.packed = slot + 1u < tile_count
                                ? *reinterpret_cast<const unsigned int*>(weight_ptr)
                                : weight_ptr[0];
                            dense_acc0[combo] = __builtin_amdgcn_fdot2_f32_bf16(
                                weight.values, value0.values, dense_acc0[combo], false);
                            dense_acc1[combo] = __builtin_amdgcn_fdot2_f32_bf16(
                                weight.values, value1.values, dense_acc1[combo], false);
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (split_partial != nullptr) {
            if (lane < 16) {
                const unsigned int query_offset = lane >> 3;
                const unsigned int head = lane & 7;
                if (query_offset < pair_rows && head_base + head < head_count) {
                    const unsigned long long split_head
                        = (((unsigned long long)(row + query_offset) * split_count + split)
                            * head_count + head_base + head);
                    split_stats[split_head * 2] = maximum[lane];
                    split_stats[split_head * 2 + 1] = denominator[lane];
                }
            }
            #pragma unroll
            for (unsigned int combo = 0; combo < 16; ++combo) {
                const unsigned int query_offset = combo >> 3;
                const unsigned int head = combo & 7;
                if (query_offset < pair_rows && head_base + head < head_count) {
                    const unsigned long long split_head
                        = (((unsigned long long)(row + query_offset) * split_count + split)
                            * head_count + head_base + head);
                    if (dense_dimension0 < latent_dim) {
                        split_partial[split_head * latent_dim + dense_dimension0]
                            = dense_acc0[combo];
                    }
                    if (dense_dimension1 < latent_dim) {
                        split_partial[split_head * latent_dim + dense_dimension1]
                            = dense_acc1[combo];
                    }
                }
            }
        } else {
            if (lane < 16) {
                denominator[lane] = 1.0f / denominator[lane];
            }
            __syncthreads();
            #pragma unroll
            for (unsigned int combo = 0; combo < 16; ++combo) {
                const unsigned int query_offset = combo >> 3;
                const unsigned int head = combo & 7;
                if (query_offset < pair_rows && head_base + head < head_count) {
                    if (dense_dimension0 < latent_dim) {
                        weighted_latent[((unsigned long long)(row + query_offset) * head_count
                            + head_base + head) * latent_dim + dense_dimension0]
                            = zllm_f32_to_bf16(dense_acc0[combo] * denominator[combo]);
                    }
                    if (dense_dimension1 < latent_dim) {
                        weighted_latent[((unsigned long long)(row + query_offset) * head_count
                            + head_base + head) * latent_dim + dense_dimension1]
                            = zllm_f32_to_bf16(dense_acc1[combo] * denominator[combo]);
                    }
                }
            }
        }
        return;
    }
#endif



    if (lane < 8) { denominator[lane] = 0.0f; maximum[lane] = -3.402823466e+38F; }
    for (unsigned int index = lane; index < 8 * latent_dim; index += blockDim.x) {
        const unsigned int head = index / latent_dim;
        const unsigned int column = index - head * latent_dim;
        absorbed_query[index] = head_base + head < head_count
            ? absorbed[((unsigned long long)row * head_count + head_base + head) * latent_dim + column]
            : 0;
    }
    for (unsigned int index = lane; index < 8 * rope_dim; index += blockDim.x) {
        const unsigned int head = index / rope_dim;
        const unsigned int column = index - head * rope_dim;
        rope_query[index] = head_base + head < head_count
            ? query[((unsigned long long)row * head_count + head_base + head) * q_head_dim + nope_dim + column]
            : 0.0f;
    }
    __syncthreads();
    float acc0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int group = lane >> 3;
    const unsigned int sublane = lane & 7;
    const unsigned int dimension0 = lane * 2;
    const unsigned int dimension1 = dimension0 + 1;
    // Q8 score 快路径：列步进 16 且 group_size 为 16 的倍数时，pair 恒落在同一 group 内
    // （column 与 group_size 均为偶数），scale 可在循环内递增装载、每 group 一次，
    // 消除逐 pair 的整数除法与重复 scale 读取；数值与 paged_latent_bf16_pair 完全一致。
    const bool q8_score_fast = latent_group_size >= 16 && (latent_group_size & 15) == 0;
    // g64 时可用 8 线程 × 8 列 = 一个 group 的超快路径（8B 对齐装载）。
    const bool q8_score_g64 = latent_group_size == 64 && (latent_dim & 63) == 0;
    const unsigned int q8_group_stride = latent_group_size >> 4;
    const unsigned int q8_group_count = q8_score_fast ? latent_dim / latent_group_size : 0;
    for (unsigned int candidate_base = 0; candidate_base < count; candidate_base += 128) {
        const unsigned int tile_count = count - candidate_base < 128 ? count - candidate_base : 128;
        if (lane < tile_count) {
            const unsigned int candidate = candidate_base + lane;
            const unsigned int token = use_selection
                ? selection[(unsigned long long)row * top_k + candidate]
                : candidate;
            physicals[lane] = (unsigned int)paged_shard_row(block_table, token, block_size, shard_parity);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int candidate_half = 0; candidate_half < 4; ++candidate_half) {
            const unsigned int slot = group + candidate_half * 32;
            const unsigned int candidate = candidate_base + slot;
            const unsigned long long physical = slot < tile_count ? physicals[slot] : 0;
            const signed char* q8_codes = reinterpret_cast<const signed char*>(latent_cache);
            const unsigned long long q8_row_base = physical * latent_dim;
            const unsigned long long q8_scale_base = physical * q8_group_count;
            unsigned int q8_index = 0;
            unsigned int q8_left = 0;
            float q8_scale = 0.0f;
            float score[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (q8_score_g64 && candidate < count) {
                // g64 超快路径：8 线程 × 8 列恰好覆盖一个 group。每迭代 1 次 8B code 装载
                // + 1 次 scale 装载，query 侧每 head 1 次 16B LDS 装载支撑 4 个 fdot2，
                // 装载指令数约为 16 列步进版本的 1/4。列在线程间的重新分配只改变归约
                // 结合顺序，不改变数学结果。
                const unsigned int group_total = latent_dim >> 6;
                for (unsigned int g = 0; g < group_total; ++g) {
                    const float scale = zllm_bf16_to_f32(latent_scales[q8_scale_base + g]);
                    const unsigned int column = (g << 6) + sublane * 8;
                    const uint2 raw = *reinterpret_cast<const uint2*>(q8_codes + q8_row_base + column);
                    const signed char* code = reinterpret_cast<const signed char*>(&raw);
                    unsigned int packed_pairs[4];
                    #pragma unroll
                    for (unsigned int pair = 0; pair < 4; ++pair) {
                        const float value0 = (float)code[pair * 2] * scale;
                        const float value1 = (float)code[pair * 2 + 1] * scale;
                        packed_pairs[pair] = (unsigned int)zllm_f32_to_bf16(value0)
                            | ((unsigned int)zllm_f32_to_bf16(value1) << 16);
                    }
                    #pragma unroll
                    for (unsigned int head = 0; head < 8; ++head) {
                        if (head_base + head < head_count) {
                            const uint4 query_quad = *reinterpret_cast<const uint4*>(
                                absorbed_query + head * latent_dim + column);
                            const unsigned int query_packed[4]
                                = {query_quad.x, query_quad.y, query_quad.z, query_quad.w};
                            #pragma unroll
                            for (unsigned int pair = 0; pair < 4; ++pair) {
#ifdef ZLLM_PAGED_BF16_DOT2
                                score[head] = __builtin_amdgcn_fdot2_f32_bf16(
                                    *reinterpret_cast<const zllm_paged_bf16x2*>(&query_packed[pair]),
                                    *reinterpret_cast<const zllm_paged_bf16x2*>(&packed_pairs[pair]),
                                    score[head], false);
#else
                                const float query0 = zllm_bf16_to_f32((unsigned short)query_packed[pair]);
                                const float query1 = zllm_bf16_to_f32((unsigned short)(query_packed[pair] >> 16));
                                const float cached0 = zllm_bf16_to_f32((unsigned short)packed_pairs[pair]);
                                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed_pairs[pair] >> 16));
                                score[head] += query0 * cached0 + query1 * cached1;
#endif
                            }
                        }
                    }
                }
            } else {
            for (unsigned int column = sublane * 2; candidate < count && column < latent_dim; column += 16) {
                unsigned int packed;
                if (q8_score_fast) {
                    if (q8_left == 0) {
                        q8_scale = zllm_bf16_to_f32(latent_scales[q8_scale_base + q8_index++]);
                        q8_left = q8_group_stride;
                    }
                    --q8_left;
                    // column+1 与同 group 内下一列恒不越界（column 偶数、latent_dim 为 group_size 倍数）。
                    const unsigned long long base = q8_row_base + column;
                    const float value0 = (float)q8_codes[base] * q8_scale;
                    const float value1 = (float)q8_codes[base + 1] * q8_scale;
                    packed = (unsigned int)zllm_f32_to_bf16(value0)
                        | ((unsigned int)zllm_f32_to_bf16(value1) << 16);
                } else {
                    packed = paged_latent_bf16_pair(
                        latent_cache, latent_scales, physical, column, latent_dim, latent_group_size);
                }
#ifdef ZLLM_PAGED_BF16_DOT2
                const zllm_paged_bf16x2 cached_pair
                    = *reinterpret_cast<const zllm_paged_bf16x2*>(&packed);
#else
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
#endif
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) {
                    if (head_base + head < head_count) {
                        const unsigned short* query_ptr = absorbed_query + head * latent_dim + column;
                        const unsigned int query_packed = column + 1 < latent_dim
                            ? *reinterpret_cast<const unsigned int*>(query_ptr)
                            : query_ptr[0];
#ifdef ZLLM_PAGED_BF16_DOT2
                        const zllm_paged_bf16x2 query_pair
                            = *reinterpret_cast<const zllm_paged_bf16x2*>(&query_packed);
                        score[head] = __builtin_amdgcn_fdot2_f32_bf16(
                            query_pair, cached_pair, score[head], false);
#else
                        const float query0 = zllm_bf16_to_f32((unsigned short)query_packed);
                        const float query1 = zllm_bf16_to_f32((unsigned short)(query_packed >> 16));
                        score[head] += query0 * cached0 + query1 * cached1;
#endif
                    }
                }
            }
            }
            for (unsigned int column = sublane * 2; candidate < count && column < rope_dim; column += 16) {
                const unsigned short* cached_ptr = rope_cache + physical * rope_dim + column;
                const unsigned int packed = column + 1 < rope_dim
                    ? *reinterpret_cast<const unsigned int*>(cached_ptr)
                    : cached_ptr[0];
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) {
                    if (head_base + head < head_count) {
                        const float* query_ptr = rope_query + head * rope_dim + column;
                        const float2 query_pair = column + 1 < rope_dim
                            ? *reinterpret_cast<const float2*>(query_ptr)
                            : make_float2(query_ptr[0], 0.0f);
                        score[head] += query_pair.x * cached0 + query_pair.y * cached1;
                    }
                }
            }
            for (unsigned int offset = 4; offset > 0; offset >>= 1) {
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) score[head] += __shfl_down(score[head], offset, 8);
            }
            if (sublane == 0) {
                const unsigned int token = candidate < count
                    ? (use_selection ? selection[(unsigned long long)row * top_k + candidate] : candidate)
                    : 0u;
                const bool owned = candidate < count && paged_shard_owns(token, block_size, shard_parity);
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) {
                    scores[head * 128 + slot] = owned && head_base + head < head_count
                        ? score[head] * rsqrtf((float)q_head_dim) : -3.402823466e+38F;
                }
            }
        }
        __syncthreads();

        const unsigned int wave = lane >> 5;
        const unsigned int wave_lane = lane & 31;
        float reduced = -3.402823466e+38F;
        for (unsigned int candidate = wave_lane; candidate < tile_count; candidate += 32) {
            reduced = fmaxf(reduced, scores[wave * 128 + candidate]);
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            reduced = fmaxf(reduced, __shfl_down(reduced, offset, 32));
        }
        if (wave_lane == 0) {
            const float next_max = fmaxf(maximum[wave], reduced);
            alpha[wave] = expf(maximum[wave] - next_max);
            maximum[wave] = next_max;
        }
        __syncthreads();
        for (unsigned int candidate = wave_lane; candidate < tile_count; candidate += 32) {
            const float score = scores[wave * 128 + candidate];
            weights[wave * 128 + candidate] = score > -3.0e+38F
                ? expf(score - maximum[wave])
                : 0.0f;
        }
        __syncthreads();
        float reduced0 = wave_lane < tile_count ? weights[wave * 128 + wave_lane] : 0.0f;
        if (wave_lane + 64 < tile_count) reduced0 += weights[wave * 128 + wave_lane + 64];
        float reduced1 = wave_lane + 32 < tile_count ? weights[wave * 128 + wave_lane + 32] : 0.0f;
        if (wave_lane + 96 < tile_count) reduced1 += weights[wave * 128 + wave_lane + 96];
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            reduced0 += __shfl_down(reduced0, offset, 32);
            reduced1 += __shfl_down(reduced1, offset, 32);
        }
        if (wave_lane == 0) denominator[wave] = denominator[wave] * alpha[wave] + reduced0 + reduced1;
        __syncthreads();

        if (dimension0 < latent_dim) {
            float sum0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float sum1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (unsigned int candidate = 0; candidate < tile_count; ++candidate) {
                const unsigned int packed = paged_latent_bf16_pair(
                    latent_cache, latent_scales, physicals[candidate], dimension0,
                    latent_dim, latent_group_size);
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) {
                    const float weight = weights[head * 128 + candidate];
                    sum0[head] += weight * cached0;
                    sum1[head] += weight * cached1;
                }
            }
            #pragma unroll
            for (unsigned int head = 0; head < 8; ++head) {
                acc0[head] = acc0[head] * alpha[head] + sum0[head];
                acc1[head] = acc1[head] * alpha[head] + sum1[head];
            }
        }
        __syncthreads();
    }
    if (split_partial != nullptr && lane < 8 && head_base + lane < head_count) {
        const unsigned long long split_head = ((unsigned long long)row * head_count + head_base + lane);
        split_stats[split_head * 2] = maximum[lane];
        split_stats[split_head * 2 + 1] = denominator[lane];
    }
    #pragma unroll
    for (unsigned int head = 0; head < 8; ++head) {
        if (head_base + head < head_count) {
            if (dimension0 < latent_dim) {
                const unsigned long long output = ((unsigned long long)row * head_count + head_base + head) * latent_dim + dimension0;
                if (split_partial != nullptr) split_partial[output] = acc0[head];
                else weighted_latent[output] = zllm_f32_to_bf16(acc0[head] / denominator[head]);
            }
            if (dimension1 < latent_dim) {
                const unsigned long long output = ((unsigned long long)row * head_count + head_base + head) * latent_dim + dimension1;
                if (split_partial != nullptr) split_partial[output] = acc1[head];
                else weighted_latent[output] = zllm_f32_to_bf16(acc1[head] / denominator[head]);
            }
        }
    }
}

// Dense 与 sparse prefill 分开编译，避免 DSA 继承 WMMA 分支的 LDS/VGPR 上限。
extern "C" __global__ void mla_paged_sparse_latent_f32(

    const float* query,
    const unsigned short* absorbed,
    const unsigned short* latent_cache,
    const unsigned short* latent_scales,
    const unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* selection,
    unsigned short* weighted_latent,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int latent_dim,
    unsigned int latent_group_size,
    unsigned int rope_dim,
    unsigned int top_k,
    unsigned int block_size,
    unsigned int selected)
{
    const unsigned int head_base = blockIdx.x * 8;
    const unsigned int row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (head_base >= head_count || row >= query_rows) return;
    const unsigned int visible = query_start + row + 1;
    // selection 只在候选数超过 top-k 时有定义；同一 prefill batch 的前缀行
    // 仍需直接遍历完整 causal 范围，不能读取尚未填满的 selection 槽。
    const bool use_selection = selected && top_k < visible;
    const unsigned int count = use_selection ? top_k : visible;
    __shared__ float scores[8 * 128];
#ifdef ZLLM_PAGED_BF16_DOT2
    // score 仍会被各 wave 读取，不能原地压成 BF16；否则后一个 head 的权重会覆盖
    // 前一个 head 尚未消费的 F32 score，长上下文首次进入 sparse 路径即发生数据竞争。
    __shared__ unsigned short weights[8 * 128];
#else
    __shared__ float weights[8 * 128];
#endif
    __shared__ unsigned int physicals[128];
    __shared__ float alpha[8];
    __shared__ float denominator[8];
    __shared__ float maximum[8];
    extern __shared__ unsigned char query_cache[];
    unsigned short* absorbed_query = reinterpret_cast<unsigned short*>(query_cache);
    float* rope_query = reinterpret_cast<float*>(absorbed_query + 8 * latent_dim);
    const unsigned int nope_dim = q_head_dim - rope_dim;

    if (lane < 8) { denominator[lane] = 0.0f; maximum[lane] = -3.402823466e+38F; }
    for (unsigned int index = lane; index < 8 * latent_dim; index += blockDim.x) {
        const unsigned int head = index / latent_dim;
        const unsigned int column = index - head * latent_dim;
        absorbed_query[index] = head_base + head < head_count
            ? absorbed[((unsigned long long)row * head_count + head_base + head) * latent_dim + column]
            : 0;
    }
    for (unsigned int index = lane; index < 8 * rope_dim; index += blockDim.x) {
        const unsigned int head = index / rope_dim;
        const unsigned int column = index - head * rope_dim;
        rope_query[index] = head_base + head < head_count
            ? query[((unsigned long long)row * head_count + head_base + head) * q_head_dim + nope_dim + column]
            : 0.0f;
    }
    __syncthreads();
    float acc0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int group = lane >> 3;
    const unsigned int sublane = lane & 7;
    const unsigned int dimension0 = lane * 2;
    const unsigned int dimension1 = dimension0 + 1;
    // Q8 score 快路径：列步进 16 且 group_size 为 16 的倍数时，pair 恒落在同一 group 内
    // （column 与 group_size 均为偶数），scale 可在循环内递增装载、每 group 一次，
    // 消除逐 pair 的整数除法与重复 scale 读取；数值与 paged_latent_bf16_pair 完全一致。
    const bool q8_score_fast = latent_group_size >= 16 && (latent_group_size & 15) == 0;
    // g64 时可用 8 线程 × 8 列 = 一个 group 的超快路径（8B 对齐装载）。
    const bool q8_score_g64 = latent_group_size == 64 && (latent_dim & 63) == 0;
    const unsigned int q8_group_stride = latent_group_size >> 4;
    const unsigned int q8_group_count = q8_score_fast ? latent_dim / latent_group_size : 0;
    for (unsigned int candidate_base = 0; candidate_base < count; candidate_base += 128) {
        const unsigned int tile_count = count - candidate_base < 128 ? count - candidate_base : 128;
        if (lane < tile_count) {
            const unsigned int candidate = candidate_base + lane;
            const unsigned int token = use_selection
                ? selection[(unsigned long long)row * top_k + candidate]
                : candidate;
            physicals[lane] = (unsigned int)paged_row(block_table, token, block_size);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int candidate_half = 0; candidate_half < 4; ++candidate_half) {
            const unsigned int slot = group + candidate_half * 32;
            const unsigned int candidate = candidate_base + slot;
            const unsigned long long physical = slot < tile_count ? physicals[slot] : 0;
            const signed char* q8_codes = reinterpret_cast<const signed char*>(latent_cache);
            const unsigned long long q8_row_base = physical * latent_dim;
            const unsigned long long q8_scale_base = physical * q8_group_count;
            unsigned int q8_index = 0;
            unsigned int q8_left = 0;
            float q8_scale = 0.0f;
            float score[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (q8_score_g64 && candidate < count) {
                // g64 超快路径：8 线程 × 8 列恰好覆盖一个 group。每迭代 1 次 8B code 装载
                // + 1 次 scale 装载，query 侧每 head 1 次 16B LDS 装载支撑 4 个 fdot2，
                // 装载指令数约为 16 列步进版本的 1/4。列在线程间的重新分配只改变归约
                // 结合顺序，不改变数学结果。
                const unsigned int group_total = latent_dim >> 6;
                for (unsigned int g = 0; g < group_total; ++g) {
                    const float scale = zllm_bf16_to_f32(latent_scales[q8_scale_base + g]);
                    const unsigned int column = (g << 6) + sublane * 8;
                    const uint2 raw = *reinterpret_cast<const uint2*>(q8_codes + q8_row_base + column);
                    const signed char* code = reinterpret_cast<const signed char*>(&raw);
                    unsigned int packed_pairs[4];
                    #pragma unroll
                    for (unsigned int pair = 0; pair < 4; ++pair) {
                        const float value0 = (float)code[pair * 2] * scale;
                        const float value1 = (float)code[pair * 2 + 1] * scale;
                        packed_pairs[pair] = (unsigned int)zllm_f32_to_bf16(value0)
                            | ((unsigned int)zllm_f32_to_bf16(value1) << 16);
                    }
                    #pragma unroll
                    for (unsigned int head = 0; head < 8; ++head) {
                        if (head_base + head < head_count) {
                            const uint4 query_quad = *reinterpret_cast<const uint4*>(
                                absorbed_query + head * latent_dim + column);
                            const unsigned int query_packed[4]
                                = {query_quad.x, query_quad.y, query_quad.z, query_quad.w};
                            #pragma unroll
                            for (unsigned int pair = 0; pair < 4; ++pair) {
#ifdef ZLLM_PAGED_BF16_DOT2
                                score[head] = __builtin_amdgcn_fdot2_f32_bf16(
                                    *reinterpret_cast<const zllm_paged_bf16x2*>(&query_packed[pair]),
                                    *reinterpret_cast<const zllm_paged_bf16x2*>(&packed_pairs[pair]),
                                    score[head], false);
#else
                                const float query0 = zllm_bf16_to_f32((unsigned short)query_packed[pair]);
                                const float query1 = zllm_bf16_to_f32((unsigned short)(query_packed[pair] >> 16));
                                const float cached0 = zllm_bf16_to_f32((unsigned short)packed_pairs[pair]);
                                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed_pairs[pair] >> 16));
                                score[head] += query0 * cached0 + query1 * cached1;
#endif
                            }
                        }
                    }
                }
            } else {
            for (unsigned int column = sublane * 2; candidate < count && column < latent_dim; column += 16) {
                unsigned int packed;
                if (q8_score_fast) {
                    if (q8_left == 0) {
                        q8_scale = zllm_bf16_to_f32(latent_scales[q8_scale_base + q8_index++]);
                        q8_left = q8_group_stride;
                    }
                    --q8_left;
                    // column+1 与同 group 内下一列恒不越界（column 偶数、latent_dim 为 group_size 倍数）。
                    const unsigned long long base = q8_row_base + column;
                    const float value0 = (float)q8_codes[base] * q8_scale;
                    const float value1 = (float)q8_codes[base + 1] * q8_scale;
                    packed = (unsigned int)zllm_f32_to_bf16(value0)
                        | ((unsigned int)zllm_f32_to_bf16(value1) << 16);
                } else {
                    packed = paged_latent_bf16_pair(
                        latent_cache, latent_scales, physical, column, latent_dim, latent_group_size);
                }
#ifdef ZLLM_PAGED_BF16_DOT2
                const zllm_paged_bf16x2 cached_pair
                    = *reinterpret_cast<const zllm_paged_bf16x2*>(&packed);
#else
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
#endif
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) {
                    if (head_base + head < head_count) {
                        const unsigned short* query_ptr = absorbed_query + head * latent_dim + column;
                        const unsigned int query_packed = column + 1 < latent_dim
                            ? *reinterpret_cast<const unsigned int*>(query_ptr)
                            : query_ptr[0];
#ifdef ZLLM_PAGED_BF16_DOT2
                        const zllm_paged_bf16x2 query_pair
                            = *reinterpret_cast<const zllm_paged_bf16x2*>(&query_packed);
                        score[head] = __builtin_amdgcn_fdot2_f32_bf16(
                            query_pair, cached_pair, score[head], false);
#else
                        const float query0 = zllm_bf16_to_f32((unsigned short)query_packed);
                        const float query1 = zllm_bf16_to_f32((unsigned short)(query_packed >> 16));
                        score[head] += query0 * cached0 + query1 * cached1;
#endif
                    }
                }
            }
            }
            for (unsigned int column = sublane * 2; candidate < count && column < rope_dim; column += 16) {
                const unsigned short* cached_ptr = rope_cache + physical * rope_dim + column;
                const unsigned int packed = column + 1 < rope_dim
                    ? *reinterpret_cast<const unsigned int*>(cached_ptr)
                    : cached_ptr[0];
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) {
                    if (head_base + head < head_count) {
                        const float* query_ptr = rope_query + head * rope_dim + column;
                        const float2 query_pair = column + 1 < rope_dim
                            ? *reinterpret_cast<const float2*>(query_ptr)
                            : make_float2(query_ptr[0], 0.0f);
                        score[head] += query_pair.x * cached0 + query_pair.y * cached1;
                    }
                }
            }
            for (unsigned int offset = 4; offset > 0; offset >>= 1) {
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) score[head] += __shfl_down(score[head], offset, 8);
            }
            if (sublane == 0) {
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) {
                    scores[head * 128 + slot] = candidate < count && head_base + head < head_count
                        ? score[head] * rsqrtf((float)q_head_dim) : -3.402823466e+38F;
                }
            }
        }
        __syncthreads();

        const unsigned int wave = lane >> 5;
        const unsigned int wave_lane = lane & 31;
        float reduced = -3.402823466e+38F;
        for (unsigned int candidate = wave_lane; candidate < tile_count; candidate += 32) {
            reduced = fmaxf(reduced, scores[wave * 128 + candidate]);
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            reduced = fmaxf(reduced, __shfl_down(reduced, offset, 32));
        }
        if (wave_lane == 0) {
            const float next_max = fmaxf(maximum[wave], reduced);
            alpha[wave] = expf(maximum[wave] - next_max);
            maximum[wave] = next_max;
        }
        __syncthreads();
        for (unsigned int candidate = wave_lane; candidate < tile_count; candidate += 32) {
            const float score = scores[wave * 128 + candidate];
#ifdef ZLLM_PAGED_BF16_DOT2
            weights[wave * 128 + candidate] = zllm_f32_to_bf16(
                score > -3.0e+38F ? expf(score - maximum[wave]) : 0.0f);
#else
            weights[wave * 128 + candidate] = score > -3.0e+38F
                ? expf(score - maximum[wave])
                : 0.0f;
#endif
        }
        __syncthreads();
#ifdef ZLLM_PAGED_BF16_DOT2
        float reduced0 = wave_lane < tile_count
            ? zllm_bf16_to_f32(weights[wave * 128 + wave_lane]) : 0.0f;
        if (wave_lane + 64 < tile_count)
            reduced0 += zllm_bf16_to_f32(weights[wave * 128 + wave_lane + 64]);
        float reduced1 = wave_lane + 32 < tile_count
            ? zllm_bf16_to_f32(weights[wave * 128 + wave_lane + 32]) : 0.0f;
        if (wave_lane + 96 < tile_count)
            reduced1 += zllm_bf16_to_f32(weights[wave * 128 + wave_lane + 96]);
#else
        float reduced0 = wave_lane < tile_count ? weights[wave * 128 + wave_lane] : 0.0f;
        if (wave_lane + 64 < tile_count) reduced0 += weights[wave * 128 + wave_lane + 64];
        float reduced1 = wave_lane + 32 < tile_count ? weights[wave * 128 + wave_lane + 32] : 0.0f;
        if (wave_lane + 96 < tile_count) reduced1 += weights[wave * 128 + wave_lane + 96];
#endif
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            reduced0 += __shfl_down(reduced0, offset, 32);
            reduced1 += __shfl_down(reduced1, offset, 32);
        }
        if (wave_lane == 0) denominator[wave] = denominator[wave] * alpha[wave] + reduced0 + reduced1;
        __syncthreads();

        const bool q8_pv_g64 = latent_group_size == 64 && (latent_dim & 63) == 0;
        if (dimension0 < latent_dim) {
            float sum0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float sum1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#ifdef ZLLM_PAGED_BF16_DOT2
            for (unsigned int candidate = 0; candidate < tile_count; candidate += 2u) {
                const unsigned int cached0 = q8_pv_g64
                    ? paged_latent_bf16_pair_q8g64(
                        latent_cache, latent_scales, physicals[candidate],
                        dimension0, latent_dim)
                    : paged_latent_bf16_pair(
                        latent_cache, latent_scales, physicals[candidate],
                        dimension0, latent_dim, latent_group_size);
                const unsigned int cached1 = candidate + 1u < tile_count
                    ? (q8_pv_g64
                        ? paged_latent_bf16_pair_q8g64(
                            latent_cache, latent_scales, physicals[candidate + 1u],
                            dimension0, latent_dim)
                        : paged_latent_bf16_pair(
                            latent_cache, latent_scales, physicals[candidate + 1u],
                            dimension0, latent_dim, latent_group_size))
                    : 0u;
                zllm_wmma_bf16_pair value0;
                zllm_wmma_bf16_pair value1;
                value0.packed = (cached0 & 0xffffu) | (cached1 << 16);
                value1.packed = (cached0 >> 16) | (cached1 & 0xffff0000u);
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) {
                    const unsigned short* weight_ptr
                        = weights + head * 128 + candidate;
                    zllm_wmma_bf16_pair weight_pair;
                    weight_pair.packed = candidate + 1u < tile_count
                        ? *reinterpret_cast<const unsigned int*>(weight_ptr)
                        : weight_ptr[0];
                    sum0[head] = __builtin_amdgcn_fdot2_f32_bf16(
                        weight_pair.values, value0.values, sum0[head], false);
                    sum1[head] = __builtin_amdgcn_fdot2_f32_bf16(
                        weight_pair.values, value1.values, sum1[head], false);
                }
            }
#else
            for (unsigned int candidate = 0; candidate < tile_count; ++candidate) {
                const unsigned int packed = q8_pv_g64
                    ? paged_latent_bf16_pair_q8g64(
                        latent_cache, latent_scales, physicals[candidate],
                        dimension0, latent_dim)
                    : paged_latent_bf16_pair(
                        latent_cache, latent_scales, physicals[candidate],
                        dimension0, latent_dim, latent_group_size);
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                #pragma unroll
                for (unsigned int head = 0; head < 8; ++head) {
                    const float weight = weights[head * 128 + candidate];
                    sum0[head] += weight * cached0;
                    sum1[head] += weight * cached1;
                }
            }
#endif
            #pragma unroll
            for (unsigned int head = 0; head < 8; ++head) {
                acc0[head] = acc0[head] * alpha[head] + sum0[head];
                acc1[head] = acc1[head] * alpha[head] + sum1[head];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int head = 0; head < 8; ++head) {
        if (head_base + head < head_count) {
            if (dimension0 < latent_dim) {
                weighted_latent[((unsigned long long)row * head_count + head_base + head) * latent_dim + dimension0]
                    = zllm_f32_to_bf16(acc0[head] / denominator[head]);
            }
            if (dimension1 < latent_dim) {
                weighted_latent[((unsigned long long)row * head_count + head_base + head) * latent_dim + dimension1]
                    = zllm_f32_to_bf16(acc1[head] / denominator[head]);
            }
        }
    }
}

#ifdef ZLLM_PAGED_BF16_DOT2
#undef ZLLM_PAGED_BF16_DOT2
#endif

// Decode score 直接使用 BF16 operand，保持 F32 累加，避免逐元素展开。
#define ZLLM_PAGED_BF16_DOT2 1
extern "C" __global__ void mla_paged_decode_partial_f32(
    const float* query,
    const unsigned short* absorbed,
    const unsigned short* latent_cache,
    const unsigned short* latent_scales,
    const unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* selection,
    const unsigned int* selection_counts,
    float* partial,
    float* stats,
    unsigned short* weighted_latent,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int latent_dim,
    unsigned int latent_group_size,
    unsigned int rope_dim,
    unsigned int top_k,
    unsigned int block_size,
    unsigned int selected,
    unsigned int tile_size,
    unsigned int stage_chunk,
    unsigned int shard_parity)
{
    const unsigned int head_base = blockIdx.x * 4;
    const unsigned int tile = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int dense_visible = query_start + 1 < context_rows ? query_start + 1 : context_rows;
    const bool use_selection = selected && top_k < dense_visible;
    const unsigned int visible = use_selection
        ? (selection_counts == nullptr ? top_k : selection_counts[0])
        : dense_visible;
    const unsigned int tile_start = tile * tile_size;
    if (head_base >= head_count) return;
    if (tile_start >= visible) {
        const unsigned long long tile_head_base
            = (unsigned long long)tile * head_count + head_base;
        #pragma unroll
        for (unsigned int head = 0; head < 4; ++head) {
            if (head_base + head < head_count) {
                if (lane == head) {
                    stats[(tile_head_base + head) * 2] = -3.402823466e+38F;
                    stats[(tile_head_base + head) * 2 + 1] = 0.0f;
                }
                for (unsigned int column = lane; column < latent_dim; column += blockDim.x) {
                    partial[(tile_head_base + head) * latent_dim + column] = 0.0f;
                }
            }
        }
        return;
    }
    const unsigned int count = visible - tile_start < tile_size ? visible - tile_start : tile_size;
    __shared__ float scores[4 * 128];
    __shared__ float weights[4 * 128];
    __shared__ unsigned int physicals[128];
    __shared__ float reduction[4 * 2];
    __shared__ float alpha[4];
    __shared__ float denominator[4];
    __shared__ float maximum[4];
#ifdef ZLLM_PAGED_BF16_DOT2
    extern __shared__ unsigned char decode_partial_query_cache[];
    unsigned short* absorbed_query
        = reinterpret_cast<unsigned short*>(decode_partial_query_cache);
    float* rope_query
        = reinterpret_cast<float*>(absorbed_query + 4 * latent_dim);
#else
    extern __shared__ float decode_partial_query_cache[];
    float* absorbed_query = decode_partial_query_cache;
    float* rope_query = decode_partial_query_cache + 4 * latent_dim;
#endif
    const unsigned int nope_dim = q_head_dim - rope_dim;
    if (lane < 4) {
        denominator[lane] = 0.0f;
        maximum[lane] = -3.402823466e+38F;
    }
    for (unsigned int index = lane; index < 4 * latent_dim; index += blockDim.x) {
        const unsigned int head = index / latent_dim;
        const unsigned int column = index - head * latent_dim;
#ifdef ZLLM_PAGED_BF16_DOT2
        absorbed_query[index] = head_base + head < head_count
            ? absorbed[(unsigned long long)(head_base + head) * latent_dim + column]
            : 0;
#else
        absorbed_query[index] = head_base + head < head_count
            ? zllm_bf16_to_f32(absorbed[(unsigned long long)(head_base + head) * latent_dim + column])
            : 0.0f;
#endif
    }
    for (unsigned int index = lane; index < 4 * rope_dim; index += blockDim.x) {
        const unsigned int head = index / rope_dim;
        const unsigned int column = index - head * rope_dim;
        rope_query[index] = head_base + head < head_count
            ? query[(unsigned long long)(head_base + head) * q_head_dim
                + nope_dim + column]
            : 0.0f;
    }
    __syncthreads();

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int group = lane >> 3;
    const unsigned int sublane = lane & 7;
    const unsigned int dimension0 = lane * 2;
    const unsigned int dimension1 = dimension0 + 1;
    if (stage_chunk != 0) {
        // Step2 软件流水：把 WMMA 路径已有的 eager tile materialization 补给 fdot2
        // 标量路径。latent 以 Q8 code+scale（group==0 时为原始 bf16）tile 进 LDS，
        // QK 与 PV 共用同一份 tile，全局 cache 每 token 只读一次；下一个 chunk 的
        // physicals 在当前 chunk 装载阶段提前 gather，页表/selection 延迟与计算重叠。
        const unsigned int chunk = stage_chunk;
        const unsigned int groups = latent_group_size != 0 ? latent_dim / latent_group_size : 0;
#ifdef ZLLM_PAGED_BF16_DOT2
        const unsigned int query_cache_bytes = 4 * latent_dim * 2 + 4 * rope_dim * 4;
#else
        const unsigned int query_cache_bytes = 4 * latent_dim * 4 + 4 * rope_dim * 4;
#endif
        unsigned char* code_tile = decode_partial_query_cache + query_cache_bytes;
        unsigned short* scale_tile = reinterpret_cast<unsigned short*>(code_tile + chunk * latent_dim);
        unsigned short* rope_tile = latent_group_size != 0
            ? scale_tile + chunk * groups
            : reinterpret_cast<unsigned short*>(code_tile) + chunk * latent_dim;
        if (lane < chunk) {
            const unsigned int candidate = tile_start + lane;
            const unsigned int token = selected ? selection[candidate] : candidate;
            physicals[lane] = candidate < visible
                ? (unsigned int)paged_shard_row(block_table, token, block_size, shard_parity)
                : 0;
        }
        __syncthreads();
        for (unsigned int base = 0; base < count; base += chunk) {
            const unsigned int sub = count - base < chunk ? count - base : chunk;
            for (unsigned int index = lane; index < sub * latent_dim; index += blockDim.x) {
                const unsigned long long physical = physicals[index / latent_dim];
                const unsigned int column = index % latent_dim;
                if (latent_group_size != 0) {
                    code_tile[index] = reinterpret_cast<const unsigned char*>(
                        latent_cache)[physical * latent_dim + column];
                } else {
                    reinterpret_cast<unsigned short*>(code_tile)[index]
                        = latent_cache[physical * latent_dim + column];
                }
            }
            for (unsigned int index = lane; index < sub * groups; index += blockDim.x) {
                scale_tile[index]
                    = latent_scales[physicals[index / groups] * groups + index % groups];
            }
            for (unsigned int index = lane; index < sub * rope_dim; index += blockDim.x) {
                rope_tile[index]
                    = rope_cache[physicals[index / rope_dim] * rope_dim + index % rope_dim];
            }
            // 预取下一个 chunk 的 physicals；gather 延迟与当前 chunk 的计算重叠。
            if (lane < chunk) {
                const unsigned int next = base + chunk + lane;
                const unsigned int candidate = tile_start + next;
                const unsigned int token = selected ? selection[candidate] : candidate;
                physicals[chunk + lane] = next < count
                    ? (unsigned int)paged_shard_row(block_table, token, block_size, shard_parity)
                    : 0;
            }
            __syncthreads();
            // 每个 32-slot 组由 8 线程组覆盖；chunk=64 走两组，chunk=32 只走一组。
            for (unsigned int candidate_half = 0; candidate_half * 32 < chunk; ++candidate_half) {
                const unsigned int slot = group + candidate_half * 32;
                const unsigned int candidate = base + slot;
                float score[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (unsigned int column = sublane * 2; candidate < count && column < latent_dim; column += 16) {
                    unsigned int packed;
                    if (latent_group_size != 0) {
                        const unsigned int local = slot * latent_dim + column;
                        const float scale = zllm_bf16_to_f32(
                            scale_tile[slot * groups + column / latent_group_size]);
                        const float value0 = (float)(signed char)code_tile[local] * scale;
                        const float value1 = column + 1 < latent_dim
                            ? (float)(signed char)code_tile[local + 1] * scale
                            : 0.0f;
                        packed = (unsigned int)zllm_f32_to_bf16(value0)
                            | ((unsigned int)zllm_f32_to_bf16(value1) << 16);
                    } else {
                        const unsigned short* value
                            = reinterpret_cast<unsigned short*>(code_tile) + slot * latent_dim + column;
                        packed = column + 1 < latent_dim
                            ? *reinterpret_cast<const unsigned int*>(value)
                            : value[0];
                    }
#ifdef ZLLM_PAGED_BF16_DOT2
                    const zllm_paged_bf16x2 cached_pair
                        = *reinterpret_cast<const zllm_paged_bf16x2*>(&packed);
#else
                    const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                    const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
#endif
                    #pragma unroll
                    for (unsigned int head = 0; head < 4; ++head) {
                        if (head_base + head < head_count) {
#ifdef ZLLM_PAGED_BF16_DOT2
                            const unsigned short* query_ptr
                                = absorbed_query + head * latent_dim + column;
                            const unsigned int query_packed = column + 1 < latent_dim
                                ? *reinterpret_cast<const unsigned int*>(query_ptr)
                                : query_ptr[0];
                            const zllm_paged_bf16x2 query_pair
                                = *reinterpret_cast<const zllm_paged_bf16x2*>(&query_packed);
                            score[head] = __builtin_amdgcn_fdot2_f32_bf16(
                                query_pair, cached_pair, score[head], false);
#else
                            const float* query_ptr = absorbed_query + head * latent_dim + column;
                            const float2 query_pair = column + 1 < latent_dim
                                ? *reinterpret_cast<const float2*>(query_ptr)
                                : make_float2(query_ptr[0], 0.0f);
                            score[head] += query_pair.x * cached0 + query_pair.y * cached1;
#endif
                        }
                    }
                }
                for (unsigned int column = sublane * 2; candidate < count && column < rope_dim; column += 16) {
                    const unsigned short* cached_ptr = rope_tile + slot * rope_dim + column;
                    const unsigned int packed = column + 1 < rope_dim
                        ? *reinterpret_cast<const unsigned int*>(cached_ptr)
                        : cached_ptr[0];
                    const float cached0
                        = zllm_bf16_to_f32((unsigned short)packed);
                    const float cached1
                        = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                    #pragma unroll
                    for (unsigned int head = 0; head < 4; ++head) {
                        if (head_base + head < head_count) {
                            const float* query_ptr = rope_query + head * rope_dim + column;
                            const float2 query_pair = column + 1 < rope_dim
                                ? *reinterpret_cast<const float2*>(query_ptr)
                                : make_float2(query_ptr[0], 0.0f);
                            score[head] += query_pair.x * cached0 + query_pair.y * cached1;
                        }
                    }
                }
                for (unsigned int offset = 4; offset > 0; offset >>= 1) {
                    #pragma unroll
                    for (unsigned int head = 0; head < 4; ++head) {
                        score[head] += __shfl_down(score[head], offset, 8);
                    }
                }
                if (sublane == 0) {
                    const unsigned int logical = tile_start + candidate;
                    const unsigned int token = candidate < count ? (selected ? selection[logical] : logical) : 0u;
                    const bool owned = candidate < count && paged_shard_owns(token, block_size, shard_parity);
                    #pragma unroll
                    for (unsigned int head = 0; head < 4; ++head) {
                        scores[head * chunk + slot] = owned && head_base + head < head_count
                            ? score[head] * rsqrtf((float)q_head_dim)
                            : -3.402823466e+38F;
                    }
                }
            }
            __syncthreads();

            const unsigned int wave = lane >> 6;
            const unsigned int wave_lane = lane & 63;
            const unsigned int half = wave_lane >> 5;
            const unsigned int half_lane = wave_lane & 31;
            float reduced = wave_lane < sub
                ? scores[wave * chunk + wave_lane]
                : -3.402823466e+38F;
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                reduced = fmaxf(reduced, __shfl_down(reduced, offset));
            }
            if (half_lane == 0) reduction[wave * 2 + half] = reduced;
            __syncthreads();
            if (wave_lane == 0) {
                const float tile_max = fmaxf(reduction[wave * 2], reduction[wave * 2 + 1]);
                const float next_max = fmaxf(maximum[wave], tile_max);
                alpha[wave] = expf(maximum[wave] - next_max);
                maximum[wave] = next_max;
            }
            __syncthreads();

            if (wave_lane < sub) {
                const float score = scores[wave * chunk + wave_lane];
                weights[wave * chunk + wave_lane] = score > -3.0e+38F
                    ? expf(score - maximum[wave])
                    : 0.0f;
            }
            __syncwarp();
            reduced = wave_lane < sub ? weights[wave * chunk + wave_lane] : 0.0f;
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                reduced += __shfl_down(reduced, offset);
            }
            if (half_lane == 0) reduction[wave * 2 + half] = reduced;
            __syncthreads();
            if (wave_lane == 0) {
                denominator[wave] = denominator[wave] * alpha[wave]
                    + reduction[wave * 2] + reduction[wave * 2 + 1];
            }
            __syncthreads();

            if (dimension0 < latent_dim) {
                float sum0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float sum1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (unsigned int candidate = 0; candidate < sub; ++candidate) {
                    unsigned int packed;
                    if (latent_group_size != 0) {
                        const unsigned int local = candidate * latent_dim + dimension0;
                        const float scale = zllm_bf16_to_f32(
                            scale_tile[candidate * groups + dimension0 / latent_group_size]);
                        const float value0 = (float)(signed char)code_tile[local] * scale;
                        const float value1 = dimension0 + 1 < latent_dim
                            ? (float)(signed char)code_tile[local + 1] * scale
                            : 0.0f;
                        packed = (unsigned int)zllm_f32_to_bf16(value0)
                            | ((unsigned int)zllm_f32_to_bf16(value1) << 16);
                    } else {
                        const unsigned short* value
                            = reinterpret_cast<unsigned short*>(code_tile) + candidate * latent_dim + dimension0;
                        packed = dimension0 + 1 < latent_dim
                            ? *reinterpret_cast<const unsigned int*>(value)
                            : value[0];
                    }
                    const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                    const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                    #pragma unroll
                    for (unsigned int head = 0; head < 4; ++head) {
                        const float weight = weights[head * chunk + candidate];
                        sum0[head] += weight * cached0;
                        sum1[head] += weight * cached1;
                    }
                }
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    acc0[head] = acc0[head] * alpha[head] + sum0[head];
                    acc1[head] = acc1[head] * alpha[head] + sum1[head];
                }
            }
            __syncthreads();
            if (lane < chunk) {
                physicals[lane] = physicals[chunk + lane];
            }
            __syncthreads();
        }
    } else
    for (unsigned int candidate_base = 0; candidate_base < count; candidate_base += 128) {
        const unsigned int tile_count = count - candidate_base < 128 ? count - candidate_base : 128;
        if (lane < tile_count) {
            const unsigned int candidate = tile_start + candidate_base + lane;
            const unsigned int token = selected ? selection[candidate] : candidate;
            physicals[lane] = (unsigned int)paged_shard_row(block_table, token, block_size, shard_parity);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int candidate_half = 0; candidate_half < 4; ++candidate_half) {
            const unsigned int slot = group + candidate_half * 32;
            const unsigned int candidate = candidate_base + slot;
            const unsigned long long physical = slot < tile_count ? physicals[slot] : 0;
            float score[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (unsigned int column = sublane * 2; candidate < count && column < latent_dim; column += 16) {
                const unsigned int packed = paged_latent_bf16_pair(
                    latent_cache, latent_scales, physical, column, latent_dim, latent_group_size);
#ifdef ZLLM_PAGED_BF16_DOT2
                const zllm_paged_bf16x2 cached_pair
                    = *reinterpret_cast<const zllm_paged_bf16x2*>(&packed);
#else
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
#endif
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    if (head_base + head < head_count) {
#ifdef ZLLM_PAGED_BF16_DOT2
                        const unsigned short* query_ptr
                            = absorbed_query + head * latent_dim + column;
                        const unsigned int query_packed = column + 1 < latent_dim
                            ? *reinterpret_cast<const unsigned int*>(query_ptr)
                            : query_ptr[0];
                        const zllm_paged_bf16x2 query_pair
                            = *reinterpret_cast<const zllm_paged_bf16x2*>(&query_packed);
                        score[head] = __builtin_amdgcn_fdot2_f32_bf16(
                            query_pair, cached_pair, score[head], false);
#else
                        const float* query_ptr = absorbed_query + head * latent_dim + column;
                        const float2 query_pair = column + 1 < latent_dim
                            ? *reinterpret_cast<const float2*>(query_ptr)
                            : make_float2(query_ptr[0], 0.0f);
                        score[head] += query_pair.x * cached0 + query_pair.y * cached1;
#endif
                    }
                }
            }
            for (unsigned int column = sublane * 2; candidate < count && column < rope_dim; column += 16) {
                const unsigned short* cached_ptr = rope_cache + physical * rope_dim + column;
                const unsigned int packed = column + 1 < rope_dim
                    ? *reinterpret_cast<const unsigned int*>(cached_ptr)
                    : cached_ptr[0];
                const float cached0
                    = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1
                    = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    if (head_base + head < head_count) {
                        const float* query_ptr = rope_query + head * rope_dim + column;
                        const float2 query_pair = column + 1 < rope_dim
                            ? *reinterpret_cast<const float2*>(query_ptr)
                            : make_float2(query_ptr[0], 0.0f);
                        score[head] += query_pair.x * cached0 + query_pair.y * cached1;
                    }
                }
            }
            for (unsigned int offset = 4; offset > 0; offset >>= 1) {
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    score[head] += __shfl_down(score[head], offset, 8);
                }
            }
            if (sublane == 0) {
                const unsigned int logical = tile_start + candidate_base + slot;
                const unsigned int token = candidate < count ? (selected ? selection[logical] : logical) : 0u;
                const bool owned = candidate < count && paged_shard_owns(token, block_size, shard_parity);
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    scores[head * 128 + slot] = owned && head_base + head < head_count
                        ? score[head] * rsqrtf((float)q_head_dim)
                        : -3.402823466e+38F;
                }
            }
        }
        __syncthreads();

        const unsigned int wave = lane >> 6;
        const unsigned int wave_lane = lane & 63;
        const unsigned int half = wave_lane >> 5;
        const unsigned int half_lane = wave_lane & 31;
        float reduced = wave_lane < tile_count
            ? scores[wave * 128 + wave_lane]
            : -3.402823466e+38F;
        if (wave_lane + 64 < tile_count) {
            reduced = fmaxf(reduced, scores[wave * 128 + wave_lane + 64]);
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            reduced = fmaxf(reduced, __shfl_down(reduced, offset));
        }
        if (half_lane == 0) reduction[wave * 2 + half] = reduced;
        __syncthreads();
        if (wave_lane == 0) {
            const float tile_max = fmaxf(reduction[wave * 2], reduction[wave * 2 + 1]);
            const float next_max = fmaxf(maximum[wave], tile_max);
            alpha[wave] = expf(maximum[wave] - next_max);
            maximum[wave] = next_max;
        }
        __syncthreads();

        if (wave_lane < tile_count) {
            const float score = scores[wave * 128 + wave_lane];
            weights[wave * 128 + wave_lane] = score > -3.0e+38F
                ? expf(score - maximum[wave])
                : 0.0f;
        }
        if (wave_lane + 64 < tile_count) {
            const float score = scores[wave * 128 + wave_lane + 64];
            weights[wave * 128 + wave_lane + 64] = score > -3.0e+38F
                ? expf(score - maximum[wave])
                : 0.0f;
        }
        __syncwarp();
        reduced = wave_lane < tile_count ? weights[wave * 128 + wave_lane] : 0.0f;
        if (wave_lane + 64 < tile_count) {
            reduced += weights[wave * 128 + wave_lane + 64];
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            reduced += __shfl_down(reduced, offset);
        }
        if (half_lane == 0) reduction[wave * 2 + half] = reduced;
        __syncthreads();
        if (wave_lane == 0) {
            denominator[wave] = denominator[wave] * alpha[wave]
                + reduction[wave * 2] + reduction[wave * 2 + 1];
        }
        __syncthreads();

        if (dimension0 < latent_dim) {
            float sum0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sum1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (unsigned int candidate = 0; candidate < tile_count; ++candidate) {
                const unsigned int packed = paged_latent_bf16_pair(
                    latent_cache, latent_scales, physicals[candidate], dimension0,
                    latent_dim, latent_group_size);
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    const float weight = weights[head * 128 + candidate];
                    sum0[head] += weight * cached0;
                    sum1[head] += weight * cached1;
                }
            }
            #pragma unroll
            for (unsigned int head = 0; head < 4; ++head) {
                acc0[head] = acc0[head] * alpha[head] + sum0[head];
                acc1[head] = acc1[head] * alpha[head] + sum1[head];
            }
        }
        __syncthreads();
    }

    const unsigned long long tile_head_base = (unsigned long long)tile * head_count + head_base;
    #pragma unroll
    for (unsigned int head = 0; head < 4; ++head) {
        if (head_base + head < head_count) {
            if (lane == head) {
                stats[(tile_head_base + head) * 2] = maximum[head];
                stats[(tile_head_base + head) * 2 + 1] = denominator[head];
            }
            if (dimension0 < latent_dim) {
                partial[(tile_head_base + head) * latent_dim + dimension0] = acc0[head];
            }
            if (dimension1 < latent_dim) {
                partial[(tile_head_base + head) * latent_dim + dimension1] = acc1[head];
            }
        }
    }
}

#ifdef ZLLM_PAGED_WMMA_BF16
// Q8 selected attention: 每个 32-token tile 的 Q8 codes/scales 只读取一次全局 cache，
// score 留在寄存器中，随后从 LDS 复用同一份原始 latent 完成 P×V。
// decode 通过 partial/stats 输出 split；prefill 每个 query 只发一个 block，直接归一化输出，
// 避免 query×split×head×latent 的大 scratch。
extern "C" __global__ __launch_bounds__(256, 1) void mla_paged_decode_partial_q8_wmma_f32(
    const float* query,
    const unsigned short* absorbed,
    const unsigned short* latent_cache,
    const unsigned short* latent_scales,
    const unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* selection,
    const unsigned int* selection_counts,
    float* partial,
    float* stats,
    unsigned short* weighted_latent,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int latent_dim,
    unsigned int latent_group_size,
    unsigned int rope_dim,
    unsigned int top_k,
    unsigned int block_size,
    unsigned int selected,
    unsigned int tile_size,
    unsigned int split_tile_count,
    unsigned int shard_parity)
{
    const unsigned int lane = threadIdx.x;
    const unsigned int wave = lane >> 5;
    const unsigned int wave_lane = lane & 31;
    const unsigned int row_column = wave_lane & 15;
    const unsigned int half = wave_lane >> 4;
    // 一个 workgroup 固定处理一个 16-head WMMA tile；8 个 wave
    // 覆盖 8 组 value dimension，每个 wave 保留四个输出 accumulator。
    const unsigned int heads_per_block = 16;
    const unsigned int head_start = blockIdx.x * heads_per_block;
    if (blockDim.x != 256 || head_start >= head_count || latent_group_size == 0) return;
    const unsigned int block_head_count = head_count - head_start < heads_per_block
        ? head_count - head_start : heads_per_block;
    const unsigned int query_waves = 1;
    const unsigned int dimension_groups = 8;
    const unsigned int token_tile = 32;
    const unsigned int output_stride = 128;
    const unsigned int output_passes = latent_dim / output_stride;
    if (block_head_count != heads_per_block || output_passes != 4
        || latent_dim % output_stride != 0 || latent_dim % 16 != 0
        || rope_dim % 16 != 0) return;

    const bool direct_output = weighted_latent != nullptr;
    if (!direct_output && split_tile_count == 0u) return;
    const unsigned int query_row = direct_output ? blockIdx.y : blockIdx.y / split_tile_count;
    const unsigned int dense_visible = query_start + query_row + 1 < context_rows
        ? query_start + query_row + 1 : context_rows;
    const bool use_selection = selected && top_k < dense_visible;
    const unsigned int visible = use_selection
        ? (selection_counts == nullptr ? top_k : selection_counts[query_row])
        : dense_visible;
    const unsigned int tile = direct_output ? 0u : blockIdx.y % split_tile_count;
    const unsigned int tile_start = tile * tile_size;
    if (tile_start >= visible) {
        const unsigned long long output_head_base
            = (unsigned long long)(direct_output ? query_row : query_row * split_tile_count + tile) * head_count
                + head_start;
        if (stats != nullptr) {
            for (unsigned int head = lane; head < block_head_count; head += blockDim.x) {
                stats[(output_head_base + head) * 2] = -3.402823466e+38F;
                stats[(output_head_base + head) * 2 + 1] = 0.0f;
            }
        }
        for (unsigned int index = lane;
             index < block_head_count * latent_dim;
             index += blockDim.x) {
            if (direct_output) {
                weighted_latent[output_head_base * latent_dim + index] = 0u;
            } else {
                partial[output_head_base * latent_dim + index] = 0.0f;
            }
        }
        return;
    }
    const unsigned int count = visible - tile_start < tile_size
        ? visible - tile_start : tile_size;
    const unsigned int groups = latent_dim / latent_group_size;
    const unsigned int nope_dim = q_head_dim - rope_dim;
    const unsigned int head_group = wave % query_waves;
    const unsigned int dimension_group = wave / query_waves;
    const unsigned int head_base = head_group * 16;
    const float attention_scale = rsqrtf((float)q_head_dim);

    extern __shared__ unsigned char decode_wmma_cache[];
    unsigned short* key_tile = reinterpret_cast<unsigned short*>(decode_wmma_cache);
    unsigned short* probability_tile = key_tile + token_tile * 16;
    unsigned short* value_tile = probability_tile + heads_per_block * 16;
    float* alpha = reinterpret_cast<float*>(value_tile + 16 * latent_dim);
    unsigned int* physicals = reinterpret_cast<unsigned int*>(alpha + heads_per_block);
    unsigned int* latent_code_tile = reinterpret_cast<unsigned int*>(physicals + token_tile);
    unsigned short* latent_scale_tile = reinterpret_cast<unsigned short*>(
        latent_code_tile + token_tile * latent_dim / 4);

    zllm_wmma_float8 output_accumulator[4] = {};
    float running_max[8];
    float running_denominator[8];
    #pragma unroll
    for (unsigned int index = 0; index < 8; ++index) {
        running_max[index] = -3.402823466e+38F;
        running_denominator[index] = 0.0f;
    }

    for (unsigned int tile_base = 0; tile_base < count; tile_base += token_tile) {
        if (lane < token_tile) {
            const unsigned int candidate = tile_base + lane;
            if (candidate < count) {
                const unsigned int logical = tile_start + candidate;
                const unsigned int token = use_selection
                    ? selection[(unsigned long long)query_row * top_k + logical]
                    : logical;
                physicals[lane] = (unsigned int)(shard_parity < 2
                    ? paged_shard_row(block_table, token, block_size, shard_parity)
                    : paged_row(block_table, token, block_size));
            } else {
                physicals[lane] = 0;
            }
        }
        __syncthreads();

        // Eager tile materialization：一次并行装载整个 32-token tile 的 code/scale 进
        // LDS（QK/PV 共用的 tile），QK 内循环不再触碰全局散读，每 tile 的 DRAM 延迟
        // 只暴露一次，替代原先 32 个 16 列块逐块串行暴露。
        {
            const unsigned int words_per_token = latent_dim / 4;
            for (unsigned int word = lane; word < token_tile * words_per_token; word += blockDim.x) {
                const unsigned int token_column = word / words_per_token;
                const unsigned int candidate = tile_base + token_column;
                latent_code_tile[word] = candidate < count
                    ? reinterpret_cast<const unsigned int*>(latent_cache)[
                          (unsigned long long)physicals[token_column] * words_per_token
                          + (word - token_column * words_per_token)]
                    : 0u;
            }
            for (unsigned int index = lane; index < token_tile * groups; index += blockDim.x) {
                const unsigned int token_column = index / groups;
                const unsigned int group = index - token_column * groups;
                const unsigned int candidate = tile_base + token_column;
                latent_scale_tile[index] = candidate < count
                    ? latent_scales[(unsigned long long)physicals[token_column] * groups + group]
                    : 0x3f80u;
            }
            __syncthreads();
        }

        zllm_wmma_float8 score_accumulator[2] = {};
        for (unsigned int column_base = 0; column_base < latent_dim; column_base += 16) {
            // 一次读取一个 Q8 word，打包写回四个 BF16；避免四个线程
            // 重复读取同一份 code/scale，也减少 LDS store 指令。
            const unsigned int key_words = token_tile * 16 / 4;
            for (unsigned int word = lane; word < key_words; word += blockDim.x) {
                const unsigned int token_column = word / 4;
                const unsigned int column = (word & 3) * 4;
                const unsigned int candidate = tile_base + token_column;
                unsigned long long packed_bf16 = 0;
                if (candidate < count) {
                    const unsigned int group = column_base / latent_group_size;
                    const unsigned short encoded_scale =
                        latent_scale_tile[token_column * groups + group];
                    const float scale = zllm_bf16_to_f32(encoded_scale);
                    const unsigned int code_word = latent_code_tile[
                        (token_column * latent_dim + column_base + column) / 4];
                    #pragma unroll
                    for (unsigned int element = 0; element < 4; ++element) {
                        const signed char code = (signed char)(
                            (code_word >> (element * 8)) & 0xffu);
                        packed_bf16 |= (unsigned long long)zllm_f32_to_bf16(
                            (float)code * scale) << (element * 16);
                    }
                }
                reinterpret_cast<unsigned long long*>(key_tile)[word] = packed_bf16;
            }
            __syncthreads();
            if (wave < query_waves) {
                zllm_wmma_short16 query_fragment;
                #pragma unroll
                for (unsigned int index = 0; index < 8; ++index) {
                    const unsigned int local_head = wave * 16 + row_column;
                    const unsigned int head = head_start + local_head;
                    query_fragment[index] = local_head < block_head_count
                        ? (short)absorbed[((unsigned long long)query_row * head_count + head) * latent_dim
                            + column_base + half * 8 + index]
                        : 0;
                }
                query_fragment = paged_wmma_expand_bf16(query_fragment);
                #pragma unroll
                for (unsigned int token_block = 0; token_block < token_tile / 16; ++token_block) {
                    zllm_wmma_short16 key_fragment;
                    #pragma unroll
                    for (unsigned int index = 0; index < 8; ++index) {
                        key_fragment[index] = (short)key_tile[
                            (token_block * 16 + row_column) * 16 + half * 8 + index];
                    }
                    key_fragment = paged_wmma_expand_bf16(key_fragment);
                    score_accumulator[token_block]
                        = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                            query_fragment, key_fragment, score_accumulator[token_block]);
                }
            }
            __syncthreads();
        }

        for (unsigned int column_base = 0; column_base < rope_dim; column_base += 16) {
            const unsigned int key_words = token_tile * 16 / 4;
            for (unsigned int word = lane; word < key_words; word += blockDim.x) {
                const unsigned int token_column = word / 4;
                const unsigned int column = (word & 3) * 4;
                const unsigned int candidate = tile_base + token_column;
                reinterpret_cast<unsigned long long*>(key_tile)[word] = candidate < count
                    ? *reinterpret_cast<const unsigned long long*>(
                        rope_cache + (unsigned long long)physicals[token_column] * rope_dim
                        + column_base + column)
                    : 0ull;
            }
            __syncthreads();
            if (wave < query_waves) {
                zllm_wmma_short16 query_fragment;
                #pragma unroll
                for (unsigned int index = 0; index < 8; ++index) {
                    const unsigned int local_head = wave * 16 + row_column;
                    const unsigned int head = head_start + local_head;
                    query_fragment[index] = local_head < block_head_count
                        ? (short)zllm_f32_to_bf16(query[((unsigned long long)query_row * head_count + head) * q_head_dim
                            + nope_dim + column_base + half * 8 + index])
                        : 0;
                }
                query_fragment = paged_wmma_expand_bf16(query_fragment);
                #pragma unroll
                for (unsigned int token_block = 0; token_block < token_tile / 16; ++token_block) {
                    zllm_wmma_short16 key_fragment;
                    #pragma unroll
                    for (unsigned int index = 0; index < 8; ++index) {
                        key_fragment[index] = (short)key_tile[
                            (token_block * 16 + row_column) * 16 + half * 8 + index];
                    }
                    key_fragment = paged_wmma_expand_bf16(key_fragment);
                    score_accumulator[token_block]
                        = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                            query_fragment, key_fragment, score_accumulator[token_block]);
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (unsigned int token_block = 0; token_block < token_tile / 16; ++token_block) {
            const unsigned int token_base = tile_base + token_block * 16;
            if (token_base >= count) break;
            if (wave < query_waves) {
                #pragma unroll
                for (unsigned int index = 0; index < 8; ++index) {
                    const unsigned int head = wave * 16 + half + 2 * index;
                    const unsigned int candidate = token_base + row_column;
                    const unsigned int logical = tile_start + candidate;
                    const unsigned int token = candidate < count
                        ? (use_selection
                            ? selection[(unsigned long long)query_row * top_k + logical]
                            : logical)
                        : 0u;
                    const bool owned = candidate < count
                        && (shard_parity >= 2
                            || paged_shard_owns(token, block_size, shard_parity));
                    const float score = owned
                        ? score_accumulator[token_block][index] * attention_scale
                        : -3.402823466e+38F;
                    float tile_max = score;
                    #pragma unroll
                    for (unsigned int offset = 8; offset > 0; offset >>= 1) {
                        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset, 16));
                    }
                    const float next_max = fmaxf(running_max[index], tile_max);
                    const float old_scale = running_max[index] == -3.402823466e+38F
                        ? 0.0f : __builtin_amdgcn_exp2f((running_max[index] - next_max) * 1.4426950408889634f);
                    const float probability = owned
                        ? __builtin_amdgcn_exp2f((score - next_max) * 1.4426950408889634f) : 0.0f;
                    float tile_sum = probability;
                    #pragma unroll
                    for (unsigned int offset = 8; offset > 0; offset >>= 1) {
                        tile_sum += __shfl_xor(tile_sum, offset, 16);
                    }
                    running_denominator[index]
                        = old_scale * running_denominator[index] + tile_sum;
                    running_max[index] = next_max;
                    probability_tile[head * 16 + row_column]
                        = zllm_f32_to_bf16(probability);
                    if (row_column == 0) alpha[head] = old_scale;
                }
            }
            __syncthreads();

            const unsigned int value_words = 16 * latent_dim / 4;
            for (unsigned int word = lane; word < value_words; word += blockDim.x) {
                const unsigned int index = word * 4;
                const unsigned int token_column = index / latent_dim;
                const unsigned int dimension = index - token_column * latent_dim;
                const unsigned int candidate = token_base + token_column;
                if (candidate < count) {
                    const float scale = zllm_bf16_to_f32(
                        latent_scale_tile[(token_block * 16 + token_column) * groups
                            + dimension / latent_group_size]);
                    const unsigned int code_word = latent_code_tile[
                        ((token_block * 16 + token_column) * latent_dim + dimension) / 4];
                    unsigned long long packed_bf16 = 0;
                    #pragma unroll
                    for (unsigned int element = 0; element < 4; ++element) {
                        const signed char code = (signed char)(
                            (code_word >> (element * 8)) & 0xffu);
                        packed_bf16 |= (unsigned long long)zllm_f32_to_bf16(
                            (float)code * scale) << (element * 16);
                    }
                    reinterpret_cast<unsigned long long*>(value_tile)[word] = packed_bf16;
                } else {
                    reinterpret_cast<unsigned long long*>(value_tile)[word] = 0ull;
                }
            }
            __syncthreads();

            zllm_wmma_short16 probability_fragment;
            #pragma unroll
            for (unsigned int index = 0; index < 8; ++index) {
                const unsigned int head = head_base + half + 2 * index;
                const float old_scale = head < block_head_count ? alpha[head] : 0.0f;
                #pragma unroll
                for (unsigned int pass = 0; pass < 4; ++pass) {
                    output_accumulator[pass][index] *= old_scale;
                }
                probability_fragment[index] = head_base + row_column < block_head_count
                    ? (short)probability_tile[(head_base + row_column) * 16 + half * 8 + index]
                    : 0;
            }
            probability_fragment = paged_wmma_expand_bf16(probability_fragment);
            #pragma unroll
            for (unsigned int pass = 0; pass < 4; ++pass) {
                const unsigned int dimension_base = pass * output_stride + dimension_group * 16;
                zllm_wmma_short16 value_fragment;
                #pragma unroll
                for (unsigned int index = 0; index < 8; ++index) {
                    value_fragment[index] = (short)value_tile[
                        (half * 8 + index) * latent_dim + dimension_base + row_column];
                }
                value_fragment = paged_wmma_expand_bf16(value_fragment);
                output_accumulator[pass]
                    = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                        probability_fragment, value_fragment, output_accumulator[pass]);
            }
            __syncthreads();
        }
    }

    if (direct_output) {
        if (wave < query_waves && row_column == 0) {
            #pragma unroll
            for (unsigned int index = 0; index < 8; ++index) {
                const unsigned int head = wave * 16 + half + 2 * index;
                if (head < block_head_count) alpha[head] = running_denominator[index];
            }
        }
        __syncthreads();
    }
    const unsigned long long tile_head_base
        = (unsigned long long)(direct_output ? query_row : query_row * split_tile_count + tile) * head_count;
    if (stats != nullptr && wave < query_waves && row_column == 0) {
        #pragma unroll
        for (unsigned int index = 0; index < 8; ++index) {
            const unsigned int local_head = wave * 16 + half + 2 * index;
            const unsigned int head = head_start + local_head;
            if (local_head < block_head_count) {
                stats[(tile_head_base + head) * 2] = running_max[index];
                stats[(tile_head_base + head) * 2 + 1] = running_denominator[index];
            }
        }
    }
    #pragma unroll
    for (unsigned int pass = 0; pass < 4; ++pass) {
        const unsigned int dimension_base = pass * output_stride + dimension_group * 16;
        #pragma unroll
        for (unsigned int index = 0; index < 8; ++index) {
            const unsigned int local_head = head_base + half + 2 * index;
            const unsigned int head = head_start + local_head;
            if (local_head < block_head_count) {
                if (direct_output) {
                    const float denominator = alpha[local_head];
                    weighted_latent[((unsigned long long)query_row * head_count + head)
                            * latent_dim + dimension_base + row_column]
                        = zllm_f32_to_bf16(denominator == 0.0f
                            ? 0.0f
                            : output_accumulator[pass][index] / denominator);
                } else {
                    partial[(tile_head_base + head) * latent_dim
                            + dimension_base + row_column] = output_accumulator[pass][index];
                }
            }
        }
    }
}

// 列并行 QK 变体：与 mla_paged_decode_partial_q8_wmma_f32 同形同参，但 QK 段
// 把 36 个 16 列切片分给 8 个 wave 并行累加（latent 32 片 = 4 个超级迭代，
// rope 4 片 = 1 个超级迭代），再经 LDS 按固定顺序跨 wave 求和。
// 原 kernel 只有 wave0 做 QK WMMA（36 次串行迭代、72 次 __syncthreads）；
// 这里 5 次超级迭代、10 次 sync。数值上改变了 score 的求和顺序（容差族，
// 非逐位一致），softmax/PV/epilogue 与原 kernel 完全一致。
// score_reduce 与 value_tile 分时复用同一段 LDS（归约结束后才写 value_tile）。
// 模板参数固定维度时（FIXED_* 非零）所有内层循环上界编译期已知，
// gather/dequant 循环可展开、全局加载批量发射；FIXED_*=0 退化为运行时维度。
template <unsigned int FIXED_LATENT, unsigned int FIXED_ROPE, unsigned int FIXED_QHEAD, unsigned int FIXED_GROUP>
__device__ __forceinline__ void mla_decode_partial_q8_wmma_colpar_impl(
    const float* query,
    const unsigned short* absorbed,
    const unsigned short* latent_cache,
    const unsigned short* latent_scales,
    const unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* selection,
    const unsigned int* selection_counts,
    float* partial,
    float* stats,
    unsigned short* weighted_latent,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int q_head_dim_,
    unsigned int latent_dim_,
    unsigned int latent_group_size_,
    unsigned int rope_dim_,
    unsigned int top_k,
    unsigned int block_size,
    unsigned int selected,
    unsigned int tile_size,
    unsigned int split_tile_count,
    unsigned int shard_parity,
    unsigned int phase_mask)
{
    const unsigned int q_head_dim = FIXED_QHEAD ? FIXED_QHEAD : q_head_dim_;
    const unsigned int latent_dim = FIXED_LATENT ? FIXED_LATENT : latent_dim_;
    const unsigned int latent_group_size = FIXED_GROUP ? FIXED_GROUP : latent_group_size_;
    const unsigned int rope_dim = FIXED_ROPE ? FIXED_ROPE : rope_dim_;
    const unsigned int lane = threadIdx.x;
    const unsigned int wave = lane >> 5;
    const unsigned int wave_lane = lane & 31;
    const unsigned int row_column = wave_lane & 15;
    const unsigned int half = wave_lane >> 4;
    const unsigned int heads_per_block = 16;
    const unsigned int head_start = blockIdx.x * heads_per_block;
    if (blockDim.x != 256 || head_start >= head_count || latent_group_size == 0) return;
    const unsigned int block_head_count = head_count - head_start < heads_per_block
        ? head_count - head_start : heads_per_block;
    const unsigned int token_tile = 32;
    const unsigned int output_stride = 128;
    const unsigned int output_passes = latent_dim / output_stride;
    const unsigned int latent_slices = latent_dim / 16;
    const unsigned int rope_slices = rope_dim / 16;
    // 列并行形状约束：latent 切片能被 8 整除、rope 切片不超过 8。
    if (block_head_count != heads_per_block || output_passes != 4
        || latent_dim % output_stride != 0 || latent_slices % 8 != 0
        || rope_dim == 0 || rope_dim % 16 != 0 || rope_slices > 8) return;

    const bool direct_output = weighted_latent != nullptr;
    if (!direct_output && split_tile_count == 0u) return;
    const unsigned int query_row = direct_output ? blockIdx.y : blockIdx.y / split_tile_count;
    const unsigned int dense_visible = query_start + query_row + 1 < context_rows
        ? query_start + query_row + 1 : context_rows;
    const bool use_selection = selected && top_k < dense_visible;
    const unsigned int visible = use_selection
        ? (selection_counts == nullptr ? top_k : selection_counts[query_row])
        : dense_visible;
    const unsigned int tile = direct_output ? 0u : blockIdx.y % split_tile_count;
    const unsigned int tile_start = tile * tile_size;
    if (tile_start >= visible) {
        const unsigned long long output_head_base
            = (unsigned long long)(direct_output ? query_row : query_row * split_tile_count + tile) * head_count
                + head_start;
        if (stats != nullptr) {
            for (unsigned int head = lane; head < block_head_count; head += blockDim.x) {
                stats[(output_head_base + head) * 2] = -3.402823466e+38F;
                stats[(output_head_base + head) * 2 + 1] = 0.0f;
            }
        }
        for (unsigned int index = lane;
             index < block_head_count * latent_dim;
             index += blockDim.x) {
            if (direct_output) {
                weighted_latent[output_head_base * latent_dim + index] = 0u;
            } else {
                partial[output_head_base * latent_dim + index] = 0.0f;
            }
        }
        return;
    }
    const unsigned int count = visible - tile_start < tile_size
        ? visible - tile_start : tile_size;
    const unsigned int groups = latent_dim / latent_group_size;
    const unsigned int nope_dim = q_head_dim - rope_dim;
    const float attention_scale = rsqrtf((float)q_head_dim);

    extern __shared__ unsigned char decode_wmma_cache[];
    unsigned short* key_tile = reinterpret_cast<unsigned short*>(decode_wmma_cache);
    unsigned short* probability_tile = key_tile + 8 * token_tile * 16;
    unsigned short* value_tile = probability_tile + heads_per_block * 16;
    float* score_reduce = reinterpret_cast<float*>(value_tile);
    float* alpha = reinterpret_cast<float*>(value_tile + 16 * latent_dim);
    unsigned int* physicals = reinterpret_cast<unsigned int*>(alpha + heads_per_block);
    unsigned int* latent_code_tile = physicals + token_tile;
    unsigned short* latent_scale_tile = reinterpret_cast<unsigned short*>(
        latent_code_tile + token_tile * latent_dim / 4);

    zllm_wmma_float8 output_accumulator[4] = {};
    float running_max[8];
    float running_denominator[8];
    #pragma unroll
    for (unsigned int index = 0; index < 8; ++index) {
        running_max[index] = -3.402823466e+38F;
        running_denominator[index] = 0.0f;
    }

    const unsigned int latent_supers = latent_slices / 8;
    const unsigned int total_supers = latent_supers + 1;

    for (unsigned int tile_base = 0; tile_base < count; tile_base += token_tile) {
        // phase_mask bit0：gather（physicals + eager materialization）。
        if (phase_mask & 1u) {
        if (lane < token_tile) {
            const unsigned int candidate = tile_base + lane;
            if (candidate < count) {
                const unsigned int logical = tile_start + candidate;
                const unsigned int token = use_selection
                    ? selection[(unsigned long long)query_row * top_k + logical]
                    : logical;
                physicals[lane] = (unsigned int)(shard_parity < 2
                    ? paged_shard_row(block_table, token, block_size, shard_parity)
                    : paged_row(block_table, token, block_size));
            } else {
                physicals[lane] = 0;
            }
        }
        __syncthreads();

        // 与原 kernel 相同的 eager tile materialization。
        {
            const unsigned int words_per_token = latent_dim / 4;
            #pragma unroll 4
            for (unsigned int word = lane; word < token_tile * words_per_token; word += blockDim.x) {
                const unsigned int token_column = word / words_per_token;
                const unsigned int candidate = tile_base + token_column;
                latent_code_tile[word] = candidate < count
                    ? reinterpret_cast<const unsigned int*>(latent_cache)[
                          (unsigned long long)physicals[token_column] * words_per_token
                          + (word - token_column * words_per_token)]
                    : 0u;
            }
            for (unsigned int index = lane; index < token_tile * groups; index += blockDim.x) {
                const unsigned int token_column = index / groups;
                const unsigned int group = index - token_column * groups;
                const unsigned int candidate = tile_base + token_column;
                latent_scale_tile[index] = candidate < count
                    ? latent_scales[(unsigned long long)physicals[token_column] * groups + group]
                    : 0x3f80u;
            }
            __syncthreads();
        }
        }

        zllm_wmma_float8 score_accumulator[2] = {};
        // 列段并行 QK：每个超级迭代装载 8 个 16 列切片进 key_tile，
        // wave w 处理切片 super*8+w 并累加自己的部分和。phase_mask bit1。
        if (phase_mask & 2u) {
        #pragma unroll
        for (unsigned int super = 0; super < total_supers; ++super) {
            const bool rope_super = super == latent_supers;
            const unsigned int slices_this = rope_super ? rope_slices : 8u;
            const unsigned int slice = super * 8u + wave;
            const unsigned int words_per_slice = token_tile * 16 / 4;
            #pragma unroll 2
            for (unsigned int word = lane; word < slices_this * words_per_slice; word += blockDim.x) {
                const unsigned int s = word / words_per_slice;
                const unsigned int local = word - s * words_per_slice;
                const unsigned int token_column = local / 4;
                const unsigned int column = (local & 3u) * 4u;
                const unsigned int candidate = tile_base + token_column;
                unsigned long long packed_bf16 = 0;
                if (candidate < count) {
                    if (!rope_super) {
                        const unsigned int column_base = (super * 8u + s) * 16u;
                        const unsigned int group = column_base / latent_group_size;
                        const unsigned short encoded_scale =
                            latent_scale_tile[token_column * groups + group];
                        const float scale = zllm_bf16_to_f32(encoded_scale);
                        const unsigned int code_word = latent_code_tile[
                            (token_column * latent_dim + column_base + column) / 4];
                        #pragma unroll
                        for (unsigned int element = 0; element < 4; ++element) {
                            const signed char code = (signed char)(
                                (code_word >> (element * 8)) & 0xffu);
                            packed_bf16 |= (unsigned long long)zllm_f32_to_bf16(
                                (float)code * scale) << (element * 16);
                        }
                    } else {
                        const unsigned int column_base = s * 16u;
                        packed_bf16 = *reinterpret_cast<const unsigned long long*>(
                            rope_cache + (unsigned long long)physicals[token_column] * rope_dim
                            + column_base + column);
                    }
                }
                reinterpret_cast<unsigned long long*>(key_tile)[
                    s * words_per_slice + local] = packed_bf16;
            }
            __syncthreads();
            if (wave < slices_this) {
                zllm_wmma_short16 query_fragment;
                #pragma unroll
                for (unsigned int index = 0; index < 8; ++index) {
                    const unsigned int local_head = row_column;
                    const unsigned int head = head_start + local_head;
                    if (rope_super) {
                        const unsigned int column_base = wave * 16u;
                        query_fragment[index] = local_head < block_head_count
                            ? (short)zllm_f32_to_bf16(query[((unsigned long long)query_row * head_count + head) * q_head_dim
                                + nope_dim + column_base + half * 8 + index])
                            : 0;
                    } else {
                        const unsigned int column_base = slice * 16u;
                        query_fragment[index] = local_head < block_head_count
                            ? (short)absorbed[((unsigned long long)query_row * head_count + head) * latent_dim
                                + column_base + half * 8 + index]
                            : 0;
                    }
                }
                query_fragment = paged_wmma_expand_bf16(query_fragment);
                #pragma unroll
                for (unsigned int token_block = 0; token_block < token_tile / 16; ++token_block) {
                    zllm_wmma_short16 key_fragment;
                    #pragma unroll
                    for (unsigned int index = 0; index < 8; ++index) {
                        key_fragment[index] = (short)key_tile[
                            wave * (token_tile * 16)
                            + (token_block * 16 + row_column) * 16 + half * 8 + index];
                    }
                    key_fragment = paged_wmma_expand_bf16(key_fragment);
                    score_accumulator[token_block]
                        = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                            query_fragment, key_fragment, score_accumulator[token_block]);
                }
            }
            __syncthreads();
        }

        // 跨 wave 归约：8 份列段部分和按 wave 顺序求和（确定性）。
        #pragma unroll
        for (unsigned int token_block = 0; token_block < token_tile / 16; ++token_block) {
            #pragma unroll
            for (unsigned int index = 0; index < 8; ++index) {
                score_reduce[((wave * 2 + token_block) * 8 + index) * 32 + wave_lane]
                    = score_accumulator[token_block][index];
            }
        }
        __syncthreads();
        if (wave == 0) {
            #pragma unroll
            for (unsigned int token_block = 0; token_block < token_tile / 16; ++token_block) {
                #pragma unroll
                for (unsigned int index = 0; index < 8; ++index) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (unsigned int w = 0; w < 8; ++w) {
                        sum += score_reduce[((w * 2 + token_block) * 8 + index) * 32 + wave_lane];
                    }
                    score_accumulator[token_block][index] = sum;
                }
            }
        }
        __syncthreads();
        }

        // 以下 softmax/PV/epilogue 与原 kernel 逐行一致。phase_mask bit2=softmax，bit3=PV。
        #pragma unroll
        for (unsigned int token_block = 0; token_block < token_tile / 16; ++token_block) {
            const unsigned int token_base = tile_base + token_block * 16;
            if (token_base >= count) break;
            if ((phase_mask & 4u) && wave < 1) {
                #pragma unroll
                for (unsigned int index = 0; index < 8; ++index) {
                    const unsigned int head = wave * 16 + half + 2 * index;
                    const unsigned int candidate = token_base + row_column;
                    // 非分片路径（shard_parity>=2）owned 与 token 无关：
                    // selection 散读只在分片掩码需要时发生，避免 1-wave softmax
                    // 串行暴露 16 次全局延迟。
                    const unsigned int logical = tile_start + candidate;
                    const unsigned int token = (shard_parity < 2 && candidate < count)
                        ? (use_selection
                            ? selection[(unsigned long long)query_row * top_k + logical]
                            : logical)
                        : 0u;
                    const bool owned = candidate < count
                        && (shard_parity >= 2
                            || paged_shard_owns(token, block_size, shard_parity));
                    const float score = owned
                        ? score_accumulator[token_block][index] * attention_scale
                        : -3.402823466e+38F;
                    float tile_max = score;
                    #pragma unroll
                    for (unsigned int offset = 8; offset > 0; offset >>= 1) {
                        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset, 16));
                    }
                    const float next_max = fmaxf(running_max[index], tile_max);
                    const float old_scale = running_max[index] == -3.402823466e+38F
                        ? 0.0f : __builtin_amdgcn_exp2f((running_max[index] - next_max) * 1.4426950408889634f);
                    const float probability = owned
                        ? __builtin_amdgcn_exp2f((score - next_max) * 1.4426950408889634f) : 0.0f;
                    float tile_sum = probability;
                    #pragma unroll
                    for (unsigned int offset = 8; offset > 0; offset >>= 1) {
                        tile_sum += __shfl_xor(tile_sum, offset, 16);
                    }
                    running_denominator[index]
                        = old_scale * running_denominator[index] + tile_sum;
                    running_max[index] = next_max;
                    probability_tile[head * 16 + row_column]
                        = zllm_f32_to_bf16(probability);
                    if (row_column == 0) alpha[head] = old_scale;
                }
            }
            __syncthreads();

            if (phase_mask & 8u) {
            const unsigned int value_words = 16 * latent_dim / 4;
            #pragma unroll 4
            for (unsigned int word = lane; word < value_words; word += blockDim.x) {
                const unsigned int index = word * 4;
                const unsigned int token_column = index / latent_dim;
                const unsigned int dimension = index - token_column * latent_dim;
                const unsigned int candidate = token_base + token_column;
                if (candidate < count) {
                    const float scale = zllm_bf16_to_f32(
                        latent_scale_tile[(token_block * 16 + token_column) * groups
                            + dimension / latent_group_size]);
                    const unsigned int code_word = latent_code_tile[
                        ((token_block * 16 + token_column) * latent_dim + dimension) / 4];
                    unsigned long long packed_bf16 = 0;
                    #pragma unroll
                    for (unsigned int element = 0; element < 4; ++element) {
                        const signed char code = (signed char)(
                            (code_word >> (element * 8)) & 0xffu);
                        packed_bf16 |= (unsigned long long)zllm_f32_to_bf16(
                            (float)code * scale) << (element * 16);
                    }
                    reinterpret_cast<unsigned long long*>(value_tile)[word] = packed_bf16;
                } else {
                    reinterpret_cast<unsigned long long*>(value_tile)[word] = 0ull;
                }
            }
            __syncthreads();

            const unsigned int dimension_group = wave;
            zllm_wmma_short16 probability_fragment;
            #pragma unroll
            for (unsigned int index = 0; index < 8; ++index) {
                const unsigned int head = half + 2 * index;
                const float old_scale = head < block_head_count ? alpha[head] : 0.0f;
                #pragma unroll
                for (unsigned int pass = 0; pass < 4; ++pass) {
                    output_accumulator[pass][index] *= old_scale;
                }
                probability_fragment[index] = row_column < block_head_count
                    ? (short)probability_tile[row_column * 16 + half * 8 + index]
                    : 0;
            }
            probability_fragment = paged_wmma_expand_bf16(probability_fragment);
            #pragma unroll
            for (unsigned int pass = 0; pass < 4; ++pass) {
                const unsigned int dimension_base = pass * output_stride + dimension_group * 16;
                zllm_wmma_short16 value_fragment;
                #pragma unroll
                for (unsigned int index = 0; index < 8; ++index) {
                    value_fragment[index] = (short)value_tile[
                        (half * 8 + index) * latent_dim + dimension_base + row_column];
                }
                value_fragment = paged_wmma_expand_bf16(value_fragment);
                output_accumulator[pass]
                    = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                        probability_fragment, value_fragment, output_accumulator[pass]);
            }
            }
            __syncthreads();
        }
    }

    if (direct_output) {
        if (wave < 1 && row_column == 0) {
            #pragma unroll
            for (unsigned int index = 0; index < 8; ++index) {
                const unsigned int head = wave * 16 + half + 2 * index;
                if (head < block_head_count) alpha[head] = running_denominator[index];
            }
        }
        __syncthreads();
    }
    const unsigned long long tile_head_base
        = (unsigned long long)(direct_output ? query_row : query_row * split_tile_count + tile) * head_count;
    if (stats != nullptr && wave < 1 && row_column == 0) {
        #pragma unroll
        for (unsigned int index = 0; index < 8; ++index) {
            const unsigned int local_head = wave * 16 + half + 2 * index;
            const unsigned int head = head_start + local_head;
            if (local_head < block_head_count) {
                stats[(tile_head_base + head) * 2] = running_max[index];
                stats[(tile_head_base + head) * 2 + 1] = running_denominator[index];
            }
        }
    }
    #pragma unroll
    for (unsigned int pass = 0; pass < 4; ++pass) {
        const unsigned int dimension_group = wave;
        const unsigned int dimension_base = pass * output_stride + dimension_group * 16;
        #pragma unroll
        for (unsigned int index = 0; index < 8; ++index) {
            const unsigned int local_head = half + 2 * index;
            const unsigned int head = head_start + local_head;
            if (local_head < block_head_count) {
                if (direct_output) {
                    const float denominator = alpha[local_head];
                    weighted_latent[((unsigned long long)query_row * head_count + head)
                            * latent_dim + dimension_base + row_column]
                        = zllm_f32_to_bf16(denominator == 0.0f
                            ? 0.0f
                            : output_accumulator[pass][index] / denominator);
                } else {
                    partial[(tile_head_base + head) * latent_dim
                        + dimension_base + row_column] = output_accumulator[pass][index];
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1) void mla_paged_decode_partial_q8_wmma_f32_colpar(
    const float* query,
    const unsigned short* absorbed,
    const unsigned short* latent_cache,
    const unsigned short* latent_scales,
    const unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* selection,
    const unsigned int* selection_counts,
    float* partial,
    float* stats,
    unsigned short* weighted_latent,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int latent_dim,
    unsigned int latent_group_size,
    unsigned int rope_dim,
    unsigned int top_k,
    unsigned int block_size,
    unsigned int selected,
    unsigned int tile_size,
    unsigned int split_tile_count,
    unsigned int shard_parity)
{
    mla_decode_partial_q8_wmma_colpar_impl<0, 0, 0, 0>(
        query, absorbed, latent_cache, latent_scales, rope_cache, block_table,
        selection, selection_counts, partial, stats, weighted_latent,
        context_rows, query_start, head_count, q_head_dim, latent_dim,
        latent_group_size, rope_dim, top_k, block_size, selected, tile_size,
        split_tile_count, shard_parity, 15u);
}

// GLM-5.3 生产形状特化（latent 512 / rope 64 / Q8G64）：
// 全部内层循环编译期展开，gather 全局加载批量发射。
// q_head_dim 保持运行时（只出现在 rope query 寻址，不进循环上界）——
// 生产实际 q_head=256（nope 192 + rope 64），不要按 576 门控。
extern "C" __global__ __launch_bounds__(256, 1) void mla_paged_decode_partial_q8_wmma_f32_colpar512(
    const float* query,
    const unsigned short* absorbed,
    const unsigned short* latent_cache,
    const unsigned short* latent_scales,
    const unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* selection,
    const unsigned int* selection_counts,
    float* partial,
    float* stats,
    unsigned short* weighted_latent,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int latent_dim,
    unsigned int latent_group_size,
    unsigned int rope_dim,
    unsigned int top_k,
    unsigned int block_size,
    unsigned int selected,
    unsigned int tile_size,
    unsigned int split_tile_count,
    unsigned int shard_parity)
{
    if (latent_dim != 512 || latent_group_size != 64 || rope_dim != 64) return;
    mla_decode_partial_q8_wmma_colpar_impl<512, 64, 0, 64>(
        query, absorbed, latent_cache, latent_scales, rope_cache, block_table,
        selection, selection_counts, partial, stats, weighted_latent,
        context_rows, query_start, head_count, q_head_dim, latent_dim,
        latent_group_size, rope_dim, top_k, block_size, selected, tile_size,
        split_tile_count, shard_parity, 15u);
}

// 消融专用：phase_mask 逐位开关 gather(1)/QK(2)/softmax(4)/PV(8)，仅用于计时分解，
// 数值无意义。生产路径不得使用。
extern "C" __global__ __launch_bounds__(256, 1) void mla_paged_decode_partial_q8_wmma_f32_colpar512_abl(
    const float* query,
    const unsigned short* absorbed,
    const unsigned short* latent_cache,
    const unsigned short* latent_scales,
    const unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* selection,
    const unsigned int* selection_counts,
    float* partial,
    float* stats,
    unsigned short* weighted_latent,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int latent_dim,
    unsigned int latent_group_size,
    unsigned int rope_dim,
    unsigned int top_k,
    unsigned int block_size,
    unsigned int selected,
    unsigned int tile_size,
    unsigned int split_tile_count,
    unsigned int shard_parity,
    unsigned int phase_mask)
{
    if (q_head_dim == 0 || latent_dim != 512 || latent_group_size != 64 || rope_dim != 64) return;
    mla_decode_partial_q8_wmma_colpar_impl<512, 64, 0, 64>(
        query, absorbed, latent_cache, latent_scales, rope_cache, block_table,
        selection, selection_counts, partial, stats, weighted_latent,
        context_rows, query_start, head_count, q_head_dim, latent_dim,
        latent_group_size, rope_dim, top_k, block_size, selected, tile_size,
        split_tile_count, shard_parity, phase_mask);
}
#endif

#undef ZLLM_PAGED_BF16_DOT2
// merge 按 latent 维切片的粒度：128 维一片，64 头 × 4 片 = 256 block。
#define MERGE_SLICE_DIM 128u

extern "C" __global__ void mla_paged_split_merge_f32(
    const float* partial,
    const float* stats,
    unsigned short* weighted_latent,
    unsigned int head_count,
    unsigned int latent_dim,
    unsigned int tile_count,
    float* merged_stats)
{
    const unsigned int slice_count = (latent_dim + MERGE_SLICE_DIM - 1u) / MERGE_SLICE_DIM;
    const unsigned int head = blockIdx.x % head_count;
    const unsigned int slice = blockIdx.x / head_count;
    const unsigned int row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    // tile 上限由 shared scale 数组容量决定；decode split-k 在 cooperative 半头
    // 拆分下每卡 grid.x=1，需要 ~target_blocks 个 tile 才能打满 CU。
    // latent 维按 MERGE_SLICE_DIM 切片进 grid.x：64 头 × 4 片 = 256 block，
    // 替代原来每头一个 block 的 64-block 低占用形态。
    if (head >= head_count || slice >= slice_count || tile_count == 0 || tile_count > 512) return;
    const unsigned int dimension_begin = slice * MERGE_SLICE_DIM;
    const unsigned int dimension_end = dimension_begin + MERGE_SLICE_DIM < latent_dim
        ? dimension_begin + MERGE_SLICE_DIM : latent_dim;
    __shared__ float old_scales[512];
    __shared__ float tile_scales[512];
    __shared__ float denominator;
    if (lane == 0) {
        float running_max = -3.402823466e+38F;
        float running_denominator = 0.0f;
        for (unsigned int tile = 0; tile < tile_count; ++tile) {
            const unsigned long long stat
                = (((unsigned long long)row * tile_count + tile) * head_count + head) * 2;
            const float tile_max = stats[stat];
            const float tile_denominator = stats[stat + 1];
            const float next_max = running_denominator == 0.0f ? tile_max : fmaxf(running_max, tile_max);
            const float old_scale = running_denominator == 0.0f ? 0.0f : expf(running_max - next_max);
            const float tile_scale = tile_denominator == 0.0f ? 0.0f : expf(tile_max - next_max);
            old_scales[tile] = old_scale;
            tile_scales[tile] = tile_scale;
            running_denominator = running_denominator * old_scale
                + tile_denominator * tile_scale;
            if (tile_denominator != 0.0f) running_max = next_max;
        }
        denominator = running_denominator;
        if (merged_stats != nullptr && slice == 0u) {
            const unsigned long long merged = ((unsigned long long)row * head_count + head) * 2;
            merged_stats[merged] = running_max;
            merged_stats[merged + 1] = running_denominator;
        }
    }
    __syncthreads();
    for (unsigned int dimension = dimension_begin + lane; dimension < dimension_end; dimension += blockDim.x) {
        float value = 0.0f;
        for (unsigned int tile = 0; tile < tile_count; ++tile) {
            value = value * old_scales[tile]
                + partial[(((unsigned long long)row * tile_count + tile) * head_count + head)
                    * latent_dim + dimension]
                    * tile_scales[tile];
        }
        weighted_latent[((unsigned long long)row * head_count + head) * latent_dim + dimension]
            = zllm_f32_to_bf16(denominator == 0.0f ? 0.0f : value / denominator);
    }
}

// 软件流水变体：partial 读取 4 深预取，打破 "全局加载 → 依赖 FMA" 的逐次延迟暴露。
// FMA 归约链仍按 tile 顺序逐个进行，数值与原 merge 逐位一致。
extern "C" __global__ void mla_paged_split_merge_f32_pl(
    const float* partial,
    const float* stats,
    unsigned short* weighted_latent,
    unsigned int head_count,
    unsigned int latent_dim,
    unsigned int tile_count,
    float* merged_stats)
{
    const unsigned int slice_count = (latent_dim + MERGE_SLICE_DIM - 1u) / MERGE_SLICE_DIM;
    const unsigned int head = blockIdx.x % head_count;
    const unsigned int slice = blockIdx.x / head_count;
    const unsigned int row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (head >= head_count || slice >= slice_count || tile_count == 0 || tile_count > 512) return;
    const unsigned int dimension_begin = slice * MERGE_SLICE_DIM;
    const unsigned int dimension_end = dimension_begin + MERGE_SLICE_DIM < latent_dim
        ? dimension_begin + MERGE_SLICE_DIM : latent_dim;
    __shared__ float old_scales[512];
    __shared__ float tile_scales[512];
    __shared__ float denominator;
    if (lane == 0) {
        // stats 同样是逐 tile 串行全局读（stride head_count*2），静态 4 路展开批量发射。
        const float* stat_source = stats
            + ((unsigned long long)row * tile_count * head_count + head) * 2;
        const unsigned long long stat_stride = (unsigned long long)head_count * 2;
        float running_max = -3.402823466e+38F;
        float running_denominator = 0.0f;
        unsigned int tile = 0;
        for (; tile + 4 <= tile_count; tile += 4) {
            const float m0 = stat_source[(unsigned long long)tile * stat_stride];
            const float d0 = stat_source[(unsigned long long)tile * stat_stride + 1];
            const float m1 = stat_source[(unsigned long long)(tile + 1) * stat_stride];
            const float d1 = stat_source[(unsigned long long)(tile + 1) * stat_stride + 1];
            const float m2 = stat_source[(unsigned long long)(tile + 2) * stat_stride];
            const float d2 = stat_source[(unsigned long long)(tile + 2) * stat_stride + 1];
            const float m3 = stat_source[(unsigned long long)(tile + 3) * stat_stride];
            const float d3 = stat_source[(unsigned long long)(tile + 3) * stat_stride + 1];
            #pragma unroll
            for (unsigned int k = 0; k < 4; ++k) {
                const float tile_max = k == 0 ? m0 : k == 1 ? m1 : k == 2 ? m2 : m3;
                const float tile_denominator = k == 0 ? d0 : k == 1 ? d1 : k == 2 ? d2 : d3;
                const float next_max = running_denominator == 0.0f ? tile_max : fmaxf(running_max, tile_max);
                const float old_scale = running_denominator == 0.0f ? 0.0f : expf(running_max - next_max);
                const float tile_scale = tile_denominator == 0.0f ? 0.0f : expf(tile_max - next_max);
                old_scales[tile + k] = old_scale;
                tile_scales[tile + k] = tile_scale;
                running_denominator = running_denominator * old_scale
                    + tile_denominator * tile_scale;
                if (tile_denominator != 0.0f) running_max = next_max;
            }
        }
        for (; tile < tile_count; ++tile) {
            const float tile_max = stat_source[(unsigned long long)tile * stat_stride];
            const float tile_denominator = stat_source[(unsigned long long)tile * stat_stride + 1];
            const float next_max = running_denominator == 0.0f ? tile_max : fmaxf(running_max, tile_max);
            const float old_scale = running_denominator == 0.0f ? 0.0f : expf(running_max - next_max);
            const float tile_scale = tile_denominator == 0.0f ? 0.0f : expf(tile_max - next_max);
            old_scales[tile] = old_scale;
            tile_scales[tile] = tile_scale;
            running_denominator = running_denominator * old_scale
                + tile_denominator * tile_scale;
            if (tile_denominator != 0.0f) running_max = next_max;
        }
        denominator = running_denominator;
        if (merged_stats != nullptr && slice == 0u) {
            const unsigned long long merged = ((unsigned long long)row * head_count + head) * 2;
            merged_stats[merged] = running_max;
            merged_stats[merged + 1] = running_denominator;
        }
    }
    __syncthreads();
    for (unsigned int dimension = dimension_begin + lane; dimension < dimension_end; dimension += blockDim.x) {
        const float* source = partial
            + ((unsigned long long)row * tile_count * head_count + head) * latent_dim + dimension;
        const unsigned long long stride = (unsigned long long)head_count * latent_dim;
        // 静态 4 路展开：4 个独立全局加载批量发射后按序进入 FMA 链（顺序不变，逐位一致）。
        float value = 0.0f;
        unsigned int tile = 0;
        for (; tile + 4 <= tile_count; tile += 4) {
            const float p0 = source[(unsigned long long)tile * stride];
            const float p1 = source[(unsigned long long)(tile + 1) * stride];
            const float p2 = source[(unsigned long long)(tile + 2) * stride];
            const float p3 = source[(unsigned long long)(tile + 3) * stride];
            value = value * old_scales[tile] + p0 * tile_scales[tile];
            value = value * old_scales[tile + 1] + p1 * tile_scales[tile + 1];
            value = value * old_scales[tile + 2] + p2 * tile_scales[tile + 2];
            value = value * old_scales[tile + 3] + p3 * tile_scales[tile + 3];
        }
        for (; tile < tile_count; ++tile) {
            value = value * old_scales[tile]
                + source[(unsigned long long)tile * stride] * tile_scales[tile];
        }
        weighted_latent[((unsigned long long)row * head_count + head) * latent_dim + dimension]
            = zllm_f32_to_bf16(denominator == 0.0f ? 0.0f : value / denominator);
    }
}

extern "C" __global__ void mla_paged_shard_scale_bf16(
    unsigned short* weighted_latent,
    const float* local_stats,
    const float* remote_stats,
    unsigned int rows,
    unsigned int head_count,
    unsigned int latent_dim)
{
    const unsigned long long index
        = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long elements
        = (unsigned long long)rows * head_count * latent_dim;
    if (index >= elements) return;
    const unsigned long long head = index / latent_dim;
    const float local_denominator = local_stats[head * 2 + 1];
    const float remote_denominator = remote_stats[head * 2 + 1];
    const float maximum = fmaxf(local_stats[head * 2], remote_stats[head * 2]);
    const float local_mass = local_denominator == 0.0f
        ? 0.0f : local_denominator * expf(local_stats[head * 2] - maximum);
    const float remote_mass = remote_denominator == 0.0f
        ? 0.0f : remote_denominator * expf(remote_stats[head * 2] - maximum);
    const float total = local_mass + remote_mass;
    const float scale = total == 0.0f ? 0.0f : local_mass / total;
    weighted_latent[index] = zllm_f32_to_bf16(
        zllm_bf16_to_f32(weighted_latent[index]) * scale);
}

// sequence-parallel 的两边各自产生“本地归一化”的 weighted latent。
// 在 head 维做 reduce-scatter：只合并本卡负责的连续 head，避免两卡随后
// 重复执行完整 value/o projection。remote 可以是 peer BAR 地址。
extern "C" __global__ void mla_paged_shard_merge_heads_bf16(
    const unsigned short* local_weighted,
    const unsigned short* remote_weighted,
    const float* local_stats,
    const float* remote_stats,
    unsigned short* output,
    unsigned int rows,
    unsigned int total_heads,
    unsigned int head_start,
    unsigned int head_count,
    unsigned int latent_dim)
{
    const unsigned long long local_head_index = blockIdx.x;
    const unsigned long long heads = (unsigned long long)rows * head_count;
    if (local_head_index >= heads) return;
    const unsigned int local_head = local_head_index % head_count;
    const unsigned int row = local_head_index / head_count;
    const unsigned int head = head_start + local_head;
    const unsigned long long global_head
        = (unsigned long long)row * total_heads + head;
    __shared__ float masses[3];
    if (threadIdx.x == 0) {
        const float local_denominator = local_stats[global_head * 2 + 1];
        const float remote_denominator = remote_stats[global_head * 2 + 1];
        const float maximum = fmaxf(
            local_stats[global_head * 2], remote_stats[global_head * 2]);
        masses[0] = local_denominator == 0.0f
            ? 0.0f : local_denominator * expf(local_stats[global_head * 2] - maximum);
        masses[1] = remote_denominator == 0.0f
            ? 0.0f : remote_denominator * expf(remote_stats[global_head * 2] - maximum);
        masses[2] = masses[0] + masses[1];
    }
    __syncthreads();
    for (unsigned int dimension = threadIdx.x;
         dimension < latent_dim;
         dimension += blockDim.x) {
        const unsigned long long source
            = global_head * latent_dim + dimension;
        const float value = masses[2] == 0.0f ? 0.0f : (
            zllm_bf16_to_f32(local_weighted[source]) * masses[0]
            + zllm_bf16_to_f32(remote_weighted[source]) * masses[1]) / masses[2];
        output[local_head_index * latent_dim + dimension]
            = zllm_f32_to_bf16(value);
    }
}

extern "C" __global__ void mla_paged_decode_latent_f32(

    const float* query,
    const unsigned short* absorbed,
    const unsigned short* latent_cache,
    const unsigned short* latent_scales,
    const unsigned short* rope_cache,
    const unsigned int* block_table,
    const unsigned int* selection,
    unsigned short* weighted_latent,
    unsigned int query_rows,
    unsigned int context_rows,
    unsigned int query_start,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int latent_dim,
    unsigned int latent_group_size,
    unsigned int rope_dim,
    unsigned int top_k,
    unsigned int block_size,
    unsigned int selected)
{
    const unsigned int head_base = blockIdx.x * 4;
    const unsigned int row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (head_base >= head_count || row >= query_rows) return;
    const unsigned int visible = query_start + row + 1;
    const unsigned int count = selected ? (top_k < visible ? top_k : visible) : visible;
    __shared__ float scores[4 * 128];
    __shared__ float weights[4 * 128];
    __shared__ unsigned int physicals[128];
    __shared__ float reduction[4 * 2];
    __shared__ float alpha[4];
    __shared__ float denominator[4];
    __shared__ float maximum[4];
#ifdef ZLLM_PAGED_BF16_DOT2
    extern __shared__ unsigned char decode_query_cache[];
    unsigned short* absorbed_query = reinterpret_cast<unsigned short*>(decode_query_cache);
    float* rope_query = reinterpret_cast<float*>(absorbed_query + 4 * latent_dim);
#else
    extern __shared__ float decode_query_cache[];
    float* absorbed_query = decode_query_cache;
    float* rope_query = decode_query_cache + 4 * latent_dim;
#endif
    const unsigned int nope_dim = q_head_dim - rope_dim;
    if (lane < 4) { denominator[lane] = 0.0f; maximum[lane] = -3.402823466e+38F; }
    for (unsigned int index = lane; index < 4 * latent_dim; index += blockDim.x) {
        const unsigned int head = index / latent_dim;
        const unsigned int column = index - head * latent_dim;
#ifdef ZLLM_PAGED_BF16_DOT2
        absorbed_query[index] = head_base + head < head_count
            ? absorbed[((unsigned long long)row * head_count + head_base + head) * latent_dim + column]
            : 0;
#else
        absorbed_query[index] = head_base + head < head_count
            ? zllm_bf16_to_f32(absorbed[((unsigned long long)row * head_count + head_base + head) * latent_dim + column])
            : 0.0f;
#endif
    }
    for (unsigned int index = lane; index < 4 * rope_dim; index += blockDim.x) {
        const unsigned int head = index / rope_dim;
        const unsigned int column = index - head * rope_dim;
        rope_query[index] = head_base + head < head_count
            ? query[((unsigned long long)row * head_count + head_base + head) * q_head_dim + nope_dim + column]
            : 0.0f;
    }
    __syncthreads();
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int group = lane >> 3;
    const unsigned int sublane = lane & 7;
    const unsigned int dimension0 = lane * 2;
    const unsigned int dimension1 = dimension0 + 1;
    for (unsigned int candidate_base = 0; candidate_base < count; candidate_base += 128) {
        const unsigned int tile_count = count - candidate_base < 128 ? count - candidate_base : 128;
        if (lane < tile_count) {
            const unsigned int candidate = candidate_base + lane;
            const unsigned int token = selected
                ? selection[(unsigned long long)row * top_k + candidate]
                : candidate;
            physicals[lane] = (unsigned int)paged_row(block_table, token, block_size);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int candidate_half = 0; candidate_half < 4; ++candidate_half) {
            const unsigned int slot = group + candidate_half * 32;
            const unsigned int candidate = candidate_base + slot;
            const unsigned long long physical = slot < tile_count ? physicals[slot] : 0;
            float score[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (unsigned int column = sublane * 2; candidate < count && column < latent_dim; column += 16) {
                const unsigned int packed = paged_latent_bf16_pair(
                    latent_cache, latent_scales, physical, column, latent_dim, latent_group_size);
#ifdef ZLLM_PAGED_BF16_DOT2
                const zllm_paged_bf16x2 cached_pair
                    = *reinterpret_cast<const zllm_paged_bf16x2*>(&packed);
#else
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
#endif
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    if (head_base + head < head_count) {
#ifdef ZLLM_PAGED_BF16_DOT2
                        const unsigned short* query_ptr = absorbed_query + head * latent_dim + column;
                        const unsigned int query_packed = column + 1 < latent_dim
                            ? *reinterpret_cast<const unsigned int*>(query_ptr)
                            : query_ptr[0];
                        const zllm_paged_bf16x2 query_pair
                            = *reinterpret_cast<const zllm_paged_bf16x2*>(&query_packed);
                        score[head] = __builtin_amdgcn_fdot2_f32_bf16(
                            query_pair, cached_pair, score[head], false);
#else
                        const float* query_ptr = absorbed_query + head * latent_dim + column;
                        const float2 query_pair = column + 1 < latent_dim
                            ? *reinterpret_cast<const float2*>(query_ptr)
                            : make_float2(query_ptr[0], 0.0f);
                        score[head] += query_pair.x * cached0 + query_pair.y * cached1;
#endif
                    }
                }
            }
            for (unsigned int column = sublane * 2; candidate < count && column < rope_dim; column += 16) {
                const unsigned short* cached_ptr = rope_cache + physical * rope_dim + column;
                const unsigned int packed = column + 1 < rope_dim
                    ? *reinterpret_cast<const unsigned int*>(cached_ptr)
                    : cached_ptr[0];
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    if (head_base + head < head_count) {
                        const float* query_ptr = rope_query + head * rope_dim + column;
                        const float2 query_pair = column + 1 < rope_dim
                            ? *reinterpret_cast<const float2*>(query_ptr)
                            : make_float2(query_ptr[0], 0.0f);
                        score[head] += query_pair.x * cached0 + query_pair.y * cached1;
                    }
                }
            }
            for (unsigned int offset = 4; offset > 0; offset >>= 1) {
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) score[head] += __shfl_down(score[head], offset, 8);
            }
            if (sublane == 0) {
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    scores[head * 128 + slot] = candidate < count && head_base + head < head_count
                        ? score[head] * rsqrtf((float)q_head_dim) : -3.402823466e+38F;
                }
            }
        }
        __syncthreads();

        const unsigned int wave = lane >> 6;
        const unsigned int wave_lane = lane & 63;
        const unsigned int half = wave_lane >> 5;
        const unsigned int half_lane = wave_lane & 31;
        float reduced = wave_lane < tile_count ? scores[wave * 128 + wave_lane] : -3.402823466e+38F;
        if (wave_lane + 64 < tile_count) {
            reduced = fmaxf(reduced, scores[wave * 128 + wave_lane + 64]);
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) reduced = fmaxf(reduced, __shfl_down(reduced, offset));
        if (half_lane == 0) reduction[wave * 2 + half] = reduced;
        __syncthreads();
        if (wave_lane == 0) {
            const float tile_max = fmaxf(reduction[wave * 2], reduction[wave * 2 + 1]);
            const float next_max = fmaxf(maximum[wave], tile_max);
            alpha[wave] = expf(maximum[wave] - next_max);
            maximum[wave] = next_max;
        }
        __syncthreads();

        if (wave_lane < tile_count) {
            const float score = scores[wave * 128 + wave_lane];
            weights[wave * 128 + wave_lane] = score > -3.0e+38F
                ? expf(score - maximum[wave])
                : 0.0f;
        }
        if (wave_lane + 64 < tile_count) {
            const float score = scores[wave * 128 + wave_lane + 64];
            weights[wave * 128 + wave_lane + 64] = score > -3.0e+38F
                ? expf(score - maximum[wave])
                : 0.0f;
        }
        __syncwarp();
        reduced = wave_lane < tile_count ? weights[wave * 128 + wave_lane] : 0.0f;
        if (wave_lane + 64 < tile_count) reduced += weights[wave * 128 + wave_lane + 64];
        for (unsigned int offset = 16; offset > 0; offset >>= 1) reduced += __shfl_down(reduced, offset);
        if (half_lane == 0) reduction[wave * 2 + half] = reduced;
        __syncthreads();
        if (wave_lane == 0) denominator[wave] = denominator[wave] * alpha[wave]
            + reduction[wave * 2] + reduction[wave * 2 + 1];
        __syncthreads();

        if (dimension0 < latent_dim) {
            float sum0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sum1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (unsigned int candidate = 0; candidate < tile_count; ++candidate) {
                const unsigned int packed = paged_latent_bf16_pair(
                    latent_cache, latent_scales, physicals[candidate], dimension0,
                    latent_dim, latent_group_size);
                const float cached0 = zllm_bf16_to_f32((unsigned short)packed);
                const float cached1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
                #pragma unroll
                for (unsigned int head = 0; head < 4; ++head) {
                    const float weight = weights[head * 128 + candidate];
                    sum0[head] += weight * cached0;
                    sum1[head] += weight * cached1;
                }
            }
            #pragma unroll
            for (unsigned int head = 0; head < 4; ++head) {
                acc0[head] = acc0[head] * alpha[head] + sum0[head];
                acc1[head] = acc1[head] * alpha[head] + sum1[head];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int head = 0; head < 4; ++head) {
        if (head_base + head < head_count) {
            if (dimension0 < latent_dim) {
                weighted_latent[((unsigned long long)row * head_count + head_base + head) * latent_dim + dimension0]
                    = zllm_f32_to_bf16(acc0[head] / denominator[head]);
            }
            if (dimension1 < latent_dim) {
                weighted_latent[((unsigned long long)row * head_count + head_base + head) * latent_dim + dimension1]
                    = zllm_f32_to_bf16(acc1[head] / denominator[head]);
            }
        }
    }
}

#ifdef ZLLM_PAGED_BF16_DOT2
#undef ZLLM_PAGED_BF16_DOT2
#endif

// Prefill 以 BF16 tile 直接吸收 query，压缩权重只在 LDS 中展开当前 tile，F32 仅用于累加。
extern "C" __global__ void mla_absorb_query_ct_wmma(
    const float* query,
    const unsigned int* packed,
    const unsigned char* scales,
    unsigned short* absorbed,
    unsigned int query_rows,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int kv_head_dim,
    unsigned int latent_dim,
    unsigned int rope_dim,
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
    const unsigned int nope_dim = q_head_dim - rope_dim;
    const bool w8g128 = bits == 8u && group_size == 128u;
    const unsigned int tile_groups = (latent_dim + 127u) / 128u;
    const unsigned int head = blockIdx.x / tile_groups;
    const unsigned int latent_base =
        ((blockIdx.x % tile_groups) * 8u + wave) * 16u;
    const unsigned int query_base = blockIdx.y * 128u;

    __shared__ unsigned short input_tiles[8][16 * 16];
    __shared__ union {
        unsigned short weight_tiles[8][16 * 16];
        float output_tiles[8][16 * 16];
    } scratch;

    CFragment accumulators[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile)
        fill_fragment(accumulators[tile], 0.0f);

    for (unsigned int column_base = 0; column_base < nope_dim; column_base += 16u) {
        for (unsigned int flat = threadIdx.x; flat < 16u * 16u; flat += blockDim.x) {
            const unsigned int local_row = flat / 16u;
            const unsigned int column = column_base + flat % 16u;
            #pragma unroll
            for (unsigned int tile = 0; tile < 8; ++tile) {
                const unsigned int row = query_base + tile * 16u + local_row;
                input_tiles[tile][flat] = row < query_rows && column < nope_dim
                    ? zllm_f32_to_bf16(query[
                        ((unsigned long long)row * head_count + head) * q_head_dim + column])
                    : 0u;
            }
        }
        if (lane < 16u) {
            const unsigned int latent_column = latent_base + lane;
            #pragma unroll
            for (unsigned int local_column = 0; local_column < 16u; ++local_column) {
                const unsigned int column = column_base + local_column;
                scratch.weight_tiles[wave][lane * 16u + local_column] =
                    latent_column < latent_dim && column < nope_dim
                    ? zllm_f32_to_bf16(w8g128
                        ? paged_weight_w8g128(
                            packed, scales, head * kv_head_dim + column,
                            latent_column, latent_dim, scale_dtype)
                        : paged_weight(
                            packed, scales, head * kv_head_dim + column,
                            latent_column, latent_dim, group_size, scale_dtype, bits))
                    : 0u;
            }
        }
        __syncthreads();

        BFragment weight_fragment;
        load_matrix_sync(
            weight_fragment,
            (const bfloat16_t*)scratch.weight_tiles[wave],
            16);
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            AFragment input_fragment;
            load_matrix_sync(
                input_fragment,
                (const bfloat16_t*)input_tiles[tile],
                16);
            mma_sync(
                accumulators[tile],
                input_fragment,
                weight_fragment,
                accumulators[tile]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        store_matrix_sync(
            scratch.output_tiles[wave],
            accumulators[tile],
            16,
            mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16u * 16u; index += warpSize) {
            const unsigned int row = query_base + tile * 16u + index / 16u;
            const unsigned int latent_column = latent_base + index % 16u;
            if (row < query_rows && latent_column < latent_dim) {
                absorbed[
                    ((unsigned long long)row * head_count + head) * latent_dim
                        + latent_column]
                    = zllm_f32_to_bf16(scratch.output_tiles[wave][index]);
            }
        }
        __syncwarp();
    }
}

// Prefill 的 value projection 直接读取 BF16 latent，压缩权重不落全局展开缓冲。
extern "C" __global__ void mla_project_value_ct_wmma(
    const unsigned short* weighted_latent,
    const unsigned int* packed,
    const unsigned char* scales,
    unsigned short* output,
    unsigned int query_rows,
    unsigned int head_count,
    unsigned int weight_head_start,
    unsigned int q_head_dim,
    unsigned int kv_head_dim,
    unsigned int latent_dim,
    unsigned int rope_dim,
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
    const unsigned int nope_dim = q_head_dim - rope_dim;
    const unsigned int value_dim = kv_head_dim - nope_dim;
    const unsigned int tile_groups = (value_dim + 127u) / 128u;
    const unsigned int head = blockIdx.x / tile_groups;
    const unsigned int value_base =
        ((blockIdx.x % tile_groups) * 8u + wave) * 16u;
    const unsigned int query_base = blockIdx.y * 128u;

    __shared__ unsigned short input_tiles[8][16 * 16];
    __shared__ union {
        unsigned short weight_tiles[8][16 * 16];
        float output_tiles[8][16 * 16];
    } scratch;

    CFragment accumulators[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile)
        fill_fragment(accumulators[tile], 0.0f);

    for (unsigned int column_base = 0; column_base < latent_dim; column_base += 16u) {
        for (unsigned int flat = threadIdx.x; flat < 16u * 16u; flat += blockDim.x) {
            const unsigned int local_row = flat / 16u;
            const unsigned int column = column_base + flat % 16u;
            #pragma unroll
            for (unsigned int tile = 0; tile < 8; ++tile) {
                const unsigned int row = query_base + tile * 16u + local_row;
                input_tiles[tile][flat] = row < query_rows && column < latent_dim
                    ? weighted_latent[
                        ((unsigned long long)row * head_count + head) * latent_dim + column]
                    : 0u;
            }
        }
        if (lane < 16u) {
            const unsigned int value = value_base + lane;
            const unsigned int row = (weight_head_start + head) * kv_head_dim + nope_dim + value;
            const unsigned long long groups = latent_dim / group_size;
            const bool one_group = column_base / group_size
                == (column_base + 15u) / group_size;
            if (value < value_dim && one_group && bits == 4u) {
                const unsigned long long packed_columns
                    = ((unsigned long long)latent_dim + 7u) / 8u;
                const unsigned long long word_base
                    = (unsigned long long)row * packed_columns + column_base / 8u;
                const unsigned int word0 = column_base < latent_dim
                    ? packed[word_base] : 0u;
                const unsigned int word1 = column_base + 8u < latent_dim
                    ? packed[word_base + 1u] : 0u;
                const float tile_scale = zllm_scale(
                    scales,
                    (unsigned long long)row * groups + column_base / group_size,
                    scale_dtype);
                #pragma unroll
                for (unsigned int local_column = 0;
                     local_column < 16u;
                     ++local_column) {
                    const unsigned int column = column_base + local_column;
                    const unsigned int word
                        = local_column < 8u ? word0 : word1;
                    const unsigned int code
                        = (word >> ((local_column & 7u) * 4u)) & 15u;
                    scratch.weight_tiles[wave][lane * 16u + local_column]
                        = column < latent_dim
                        ? zllm_f32_to_bf16(
                            (float)((int)code - 8) * tile_scale)
                        : 0u;
                }
            } else if (value < value_dim && one_group && bits == 8u) {
                const unsigned long long packed_columns
                    = ((unsigned long long)latent_dim + 3u) / 4u;
                const unsigned long long word_base
                    = (unsigned long long)row * packed_columns + column_base / 4u;
                const unsigned int words[4] = {
                    column_base < latent_dim ? packed[word_base] : 0u,
                    column_base + 4u < latent_dim ? packed[word_base + 1u] : 0u,
                    column_base + 8u < latent_dim ? packed[word_base + 2u] : 0u,
                    column_base + 12u < latent_dim ? packed[word_base + 3u] : 0u,
                };
                const float tile_scale = zllm_scale(
                    scales,
                    (unsigned long long)row * groups + column_base / group_size,
                    scale_dtype);
                #pragma unroll
                for (unsigned int local_column = 0;
                     local_column < 16u;
                     ++local_column) {
                    const unsigned int column = column_base + local_column;
                    const unsigned int code
                        = (words[local_column >> 2u]
                            >> ((local_column & 3u) * 8u)) & 255u;
                    scratch.weight_tiles[wave][lane * 16u + local_column]
                        = column < latent_dim
                        ? zllm_f32_to_bf16(
                            (float)((int)code - 128) * tile_scale)
                        : 0u;
                }
            } else {
                #pragma unroll
                for (unsigned int local_column = 0;
                     local_column < 16u;
                     ++local_column) {
                    const unsigned int column = column_base + local_column;
                    scratch.weight_tiles[wave][lane * 16u + local_column]
                        = value < value_dim && column < latent_dim
                        ? zllm_f32_to_bf16(paged_weight(
                            packed, scales, row, column, latent_dim,
                            group_size, scale_dtype, bits))
                        : 0u;
                }
            }
        }
        __syncthreads();

        BFragment weight_fragment;
        load_matrix_sync(
            weight_fragment,
            (const bfloat16_t*)scratch.weight_tiles[wave],
            16);
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            AFragment input_fragment;
            load_matrix_sync(
                input_fragment,
                (const bfloat16_t*)input_tiles[tile],
                16);
            mma_sync(
                accumulators[tile],
                input_fragment,
                weight_fragment,
                accumulators[tile]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        store_matrix_sync(
            scratch.output_tiles[wave],
            accumulators[tile],
            16,
            mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16u * 16u; index += warpSize) {
            const unsigned int row = query_base + tile * 16u + index / 16u;
            const unsigned int value = value_base + index % 16u;
            if (row < query_rows && value < value_dim) {
                output[
                    ((unsigned long long)row * head_count + head) * q_head_dim + value]
                    = zllm_f32_to_bf16(scratch.output_tiles[wave][index]);
            }
        }
        __syncwarp();
    }

    const unsigned int tail_dim = q_head_dim - value_dim;
    if (blockIdx.x % tile_groups == 0u) {
        for (unsigned int flat = threadIdx.x;
             flat < 128u * tail_dim;
             flat += blockDim.x) {
            const unsigned int row = query_base + flat / tail_dim;
            if (row < query_rows) {
                output[
                    ((unsigned long long)row * head_count + head) * q_head_dim
                        + value_dim + flat % tail_dim] = 0.0f;
            }
        }
    }
}

extern "C" __global__ void mla_project_value_ct(
    const unsigned short* weighted_latent,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int query_rows,
    unsigned int head_count,
    unsigned int weight_head_start,
    unsigned int q_head_dim,
    unsigned int kv_head_dim,
    unsigned int latent_dim,
    unsigned int rope_dim,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits)
{
    if (query_rows == 2u && bits == 8u && group_size == 32u
        && (latent_dim & 127u) == 0u) {
        const unsigned int nope_dim = q_head_dim - rope_dim;
        const unsigned int value_dim = kv_head_dim - nope_dim;
        const unsigned int value_tiles = (value_dim + 15u) / 16u;
        const unsigned int head = blockIdx.x / value_tiles;
        const unsigned int value =
            (blockIdx.x % value_tiles) * 16u + (threadIdx.x >> 4u);
        const unsigned int lane = threadIdx.x & 15u;
        if (value < value_dim) {
            float sums[2] = {};
            const unsigned int weight_row =
                (weight_head_start + head) * kv_head_dim + nope_dim + value;
            const unsigned int groups = latent_dim >> 5u;
            const unsigned char* weight_bytes =
                reinterpret_cast<const unsigned char*>(packed)
                    + (unsigned long long)weight_row * latent_dim;
            for (unsigned int col0 = lane * 8u;
                 col0 < latent_dim;
                 col0 += 128u) {
                const int2 pack8 =
                    *reinterpret_cast<const int2*>(weight_bytes + col0);
                const float scale = zllm_scale(
                    scales,
                    (unsigned long long)weight_row * groups + (col0 >> 5u),
                    scale_dtype);
                const uint4 input0 = *reinterpret_cast<const uint4*>(
                    weighted_latent
                        + (unsigned long long)head * latent_dim + col0);
                const uint4 input1 = *reinterpret_cast<const uint4*>(
                    weighted_latent
                        + ((unsigned long long)head_count + head) * latent_dim
                        + col0);
                const unsigned int* input_words[2] = {
                    reinterpret_cast<const unsigned int*>(&input0),
                    reinterpret_cast<const unsigned int*>(&input1)
                };
                const unsigned int pack_words[2] = {
                    (unsigned int)pack8.x, (unsigned int)pack8.y
                };
                #pragma unroll
                for (unsigned int half = 0u; half < 2u; ++half) {
                    #pragma unroll
                    for (unsigned int element = 0u; element < 4u; ++element) {
                        const int code = (int)((pack_words[half]
                            >> (element * 8u)) & 255u) - 128;
                        const float weight = (float)code * scale;
                        #pragma unroll
                        for (unsigned int query_row = 0u;
                             query_row < 2u;
                             ++query_row) {
                            const unsigned int input_pair = input_words[query_row]
                                [half * 2u + element / 2u];
                            const unsigned short input_bits = (unsigned short)(
                                input_pair >> ((element & 1u) * 16u));
                            sums[query_row] +=
                                zllm_bf16_to_f32(input_bits) * weight;
                        }
                    }
                }
            }
            #pragma unroll
            for (unsigned int query_row = 0u; query_row < 2u; ++query_row) {
                for (unsigned int offset = 8u; offset > 0u; offset >>= 1u)
                    sums[query_row] += __shfl_down(
                        sums[query_row], offset, 16);
            }
            if (lane == 0u) {
                unsigned short* output_bf16 =
                    reinterpret_cast<unsigned short*>(output);
                #pragma unroll
                for (unsigned int query_row = 0u;
                     query_row < 2u;
                     ++query_row) {
                    output_bf16[
                        ((unsigned long long)query_row * head_count + head)
                            * q_head_dim + value]
                        = zllm_f32_to_bf16(sums[query_row]);
                }
            }
        }
        const unsigned int tail_dim = q_head_dim - value_dim;
        if (blockIdx.x % value_tiles == 0u) {
            unsigned short* output_bf16 =
                reinterpret_cast<unsigned short*>(output);
            for (unsigned int index = threadIdx.x;
                 index < 2u * tail_dim;
                 index += blockDim.x) {
                const unsigned int query_row = index / tail_dim;
                const unsigned int tail_column = index - query_row * tail_dim;
                output_bf16[
                    ((unsigned long long)query_row * head_count + head)
                        * q_head_dim + value_dim + tail_column] = 0u;
            }
        }
        return;
    }
    // Decode 复用压缩权重，但不再为不存在的另外 15 行做矩阵乘。
    if (query_rows == 1u) {
        const unsigned int nope_dim = q_head_dim - rope_dim;
        const unsigned int value_dim = kv_head_dim - nope_dim;
        const unsigned int value_tiles = (value_dim + 15u) / 16u;
        const unsigned int head = blockIdx.x / value_tiles;
        const unsigned int value = (blockIdx.x % value_tiles) * 16u + (threadIdx.x >> 4u);
        const unsigned int lane = threadIdx.x & 15u;
        if (value < value_dim) {
            float sum = 0.0f;
            const unsigned int weight_row = (weight_head_start + head) * kv_head_dim + nope_dim + value;
            if (bits == 32u && (latent_dim & 127u) == 0u) {
                // F32 dense decode 向量化：每 lane 8 连续列两次 float4 加载，
                // 替代逐元素 4B 标量加载。数值不变，仅 lane 内归约从跨步 16
                // 改为 8 连续列。
                const float *weight_row_f32 = reinterpret_cast<const float *>(packed)
                    + (unsigned long long)weight_row * latent_dim;
                const unsigned short *input_row = weighted_latent + (unsigned long long)head * latent_dim;
                for (unsigned int col0 = lane * 8u; col0 < latent_dim; col0 += 128u) {
                    const float4 wa = *reinterpret_cast<const float4 *>(weight_row_f32 + col0);
                    const float4 wb = *reinterpret_cast<const float4 *>(weight_row_f32 + col0 + 4u);
                    const uint4 in8 = *reinterpret_cast<const uint4 *>(input_row + col0);
                    const unsigned int *in_words = reinterpret_cast<const unsigned int *>(&in8);
                    #pragma unroll
                    for (unsigned int element = 0; element < 4; ++element) {
                        const unsigned int in_pair = in_words[element / 2];
                        const unsigned short in_bits = (unsigned short)(in_pair >> ((element & 1u) * 16u));
                        const float input_value = zllm_bf16_to_f32(in_bits);
                        const float weight_value = element == 0u ? wa.x : element == 1u ? wa.y : element == 2u ? wa.z : wa.w;
                        sum += input_value * weight_value;
                    }
                    #pragma unroll
                    for (unsigned int element = 0; element < 4; ++element) {
                        const unsigned int in_pair = in_words[element / 2 + 2];
                        const unsigned short in_bits = (unsigned short)(in_pair >> ((element & 1u) * 16u));
                        const float input_value = zllm_bf16_to_f32(in_bits);
                        const float weight_value = element == 0u ? wb.x : element == 1u ? wb.y : element == 2u ? wb.z : wb.w;
                        sum += input_value * weight_value;
                    }
                }
            } else if (bits == 8u && group_size == 32u && (latent_dim & 127u) == 0u) {
                // W8（i8+128 偏置码）decode 向量化：每 lane 8 连续列，一次 8B
                // packed 加载 + 每 32 列一次 scale，替代逐元素 4B 字加载与逐元素
                // scale 读取。每元素数值不变，lane 归约从跨步改为 8 连续列。
                const unsigned int groups = latent_dim >> 5u;
                const unsigned char *weight_bytes = reinterpret_cast<const unsigned char *>(packed)
                    + (unsigned long long)weight_row * latent_dim;
                const unsigned short *input_row = weighted_latent + (unsigned long long)head * latent_dim;
                for (unsigned int col0 = lane * 8u; col0 < latent_dim; col0 += 128u) {
                    const int2 pack8 = *reinterpret_cast<const int2 *>(weight_bytes + col0);
                    const float scale = zllm_scale(scales, (unsigned long long)weight_row * groups + (col0 >> 5u), scale_dtype);
                    const uint4 in8 = *reinterpret_cast<const uint4 *>(input_row + col0);
                    const unsigned int *in_words = reinterpret_cast<const unsigned int *>(&in8);
                    const unsigned int pack_words[2] = {(unsigned int)pack8.x, (unsigned int)pack8.y};
                    #pragma unroll
                    for (unsigned int half = 0; half < 2; ++half) {
                        #pragma unroll
                        for (unsigned int element = 0; element < 4; ++element) {
                            const int code = (int)((pack_words[half] >> (element * 8u)) & 255u) - 128;
                            const unsigned int in_pair = in_words[half * 2 + element / 2];
                            const unsigned short in_bits = (unsigned short)(in_pair >> ((element & 1u) * 16u));
                            sum += zllm_bf16_to_f32(in_bits) * ((float)code * scale);
                        }
                    }
                }
            } else if (bits == 4u && group_size == 128u && (latent_dim & 127u) == 0u) {
                const unsigned int packed_columns = latent_dim >> 3u;
                const unsigned int groups = latent_dim >> 7u;
                #pragma unroll
                for (unsigned int group = 0; group < groups; ++group) {
                    const unsigned int group_base = group << 7u;
                    const float scale = zllm_scale(
                        scales, (unsigned long long)weight_row * groups + group, scale_dtype);
                    for (unsigned int local = lane; local < 128u; local += 16u) {
                        const unsigned int column = group_base + local;
                        const unsigned int word = packed[
                            (unsigned long long)weight_row * packed_columns + (column >> 3u)];
                        const int code = (int)((word >> ((column & 7u) * 4u)) & 15u) - 8;
                        sum += zllm_bf16_to_f32(
                            weighted_latent[(unsigned long long)head * latent_dim + column])
                            * ((float)code * scale);
                    }
                }
            } else {
                for (unsigned int column = lane; column < latent_dim; column += 16u) {
                    sum += zllm_bf16_to_f32(
                        weighted_latent[(unsigned long long)head * latent_dim + column])
                        * paged_weight(
                            packed, scales, weight_row, column, latent_dim,
                            group_size, scale_dtype, bits);
                }
            }
            for (unsigned int offset = 8u; offset > 0u; offset >>= 1u)
                sum += __shfl_down(sum, offset, 16);
            if (lane == 0u)
                output[(unsigned long long)head * q_head_dim + value] = sum;
        }
        const unsigned int tail_dim = q_head_dim - value_dim;
        if (blockIdx.x % value_tiles == 0u) {
            for (unsigned int index = threadIdx.x; index < tail_dim; index += blockDim.x)
                output[(unsigned long long)head * q_head_dim + value_dim + index] = 0.0f;
        }
        return;
    }

    const unsigned int nope_dim = q_head_dim - rope_dim;
    const unsigned int value_dim = kv_head_dim - nope_dim;
    const unsigned int value_tiles = (value_dim + 15) / 16;
    const unsigned int head = blockIdx.x / value_tiles;
    const unsigned int value_base = (blockIdx.x % value_tiles) * 16;
    const unsigned int query_local = threadIdx.x / 16;
    const unsigned int value_local = threadIdx.x % 16;
    const unsigned int query_row = blockIdx.y * 16 + query_local;
    const unsigned int value = value_base + value_local;
    __shared__ float input_tile[256];
    __shared__ float weight_tile[256];
    float sum = 0.0f;
    for (unsigned int column_base = 0; column_base < latent_dim; column_base += 16) {
        const unsigned int input_column = column_base + value_local;
        input_tile[threadIdx.x] = query_row < query_rows && input_column < latent_dim
            ? zllm_bf16_to_f32(weighted_latent[
                ((unsigned long long)query_row * head_count + head) * latent_dim + input_column])
            : 0.0f;
        const unsigned int weight_column = column_base + query_local;
        weight_tile[threadIdx.x] = value < value_dim && weight_column < latent_dim
            ? paged_weight(
                packed, scales, (weight_head_start + head) * kv_head_dim + nope_dim + value,
                weight_column, latent_dim, group_size, scale_dtype, bits)
            : 0.0f;
        __syncthreads();
        #pragma unroll
        for (unsigned int column = 0; column < 16; ++column) {
            sum += input_tile[query_local * 16 + column]
                * weight_tile[column * 16 + value_local];
        }
        __syncthreads();
    }
    if (query_row < query_rows && value < value_dim) {
        output[((unsigned long long)query_row * head_count + head) * q_head_dim + value] = sum;
    }
    const unsigned int tail_dim = q_head_dim - value_dim;
    if (blockIdx.x % value_tiles == 0 && tail_dim != 0) {
        for (unsigned int index = threadIdx.x; index < 16 * tail_dim; index += blockDim.x) {
            const unsigned int tail_row = blockIdx.y * 16 + index / tail_dim;
            const unsigned int tail_column = value_dim + index % tail_dim;
            if (tail_row < query_rows) {
                output[((unsigned long long)tail_row * head_count + head) * q_head_dim + tail_column] = 0.0f;
            }
        }
    }
}

// PV decode 的 perm 直读臂：与 mla_project_value_ct 的 W8G32 分支同形
// （decode 单行、16-lane/value、每 lane 8 连续列、grid=heads×value_tiles），
// 差别：本 head 的 weighted latent 行经 LDS 预转 f16 并按 32 列 group 预累加
// Σx_g；权重码由 zllm_w8_perm_f16 直接构造 f16(1024+u) 配 zllm_f16x2_dot2
// 累加，+1024 偏置由 dot 初值 -1152·Σx_g 一次修正。替代逐元素
// extract/cvt/mul/mul/add 标量链（当前 23µs 仅 365GB/s，每 lane 仅 4 次迭代的
// 延迟暴露），数值 bf16 容差族。形状门控在 host 侧（bits==8、G32、
// latent%128==0）；kernel 只需 256 线程 decode 形态。
extern "C" __global__ void mla_project_value_ct_perm(
    const unsigned short* weighted_latent,
    const unsigned int* packed,
    const unsigned char* scales,
    float* output,
    unsigned int query_rows,
    unsigned int head_count,
    unsigned int weight_head_start,
    unsigned int q_head_dim,
    unsigned int kv_head_dim,
    unsigned int latent_dim,
    unsigned int rope_dim,
    unsigned int group_size,
    unsigned int scale_dtype,
    unsigned int bits)
{
    if (query_rows != 1u || bits != 8u || group_size != 32u
        || (latent_dim & 127u) != 0u || blockDim.x != 256u) {
        return;
    }
    const unsigned int nope_dim = q_head_dim - rope_dim;
    const unsigned int value_dim = kv_head_dim - nope_dim;
    const unsigned int value_tiles = (value_dim + 15u) / 16u;
    const unsigned int head = blockIdx.x / value_tiles;
    if (head >= head_count) return;
    const unsigned int value = (blockIdx.x % value_tiles) * 16u + (threadIdx.x >> 4u);
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int groups = latent_dim >> 5u;

    extern __shared__ unsigned char pv_perm_cache[];
    zllm_f16x2* input_f16 = reinterpret_cast<zllm_f16x2*>(pv_perm_cache);
    float* group_bias = reinterpret_cast<float*>(pv_perm_cache + latent_dim * 2u);
    for (unsigned int column = threadIdx.x * 2u; column < latent_dim;
         column += blockDim.x * 2u) {
        const unsigned int raw = *reinterpret_cast<const unsigned int*>(
            weighted_latent + (unsigned long long)head * latent_dim + column);
        union ZllmPvF16x2Word {
            zllm_f16x2 pair;
            unsigned int word;
        } converted;
        converted.word = (unsigned int)zllm_f32_to_f16(
            zllm_bf16_to_f32((unsigned short)(raw & 0xffffu)))
            | ((unsigned int)zllm_f32_to_f16(
                zllm_bf16_to_f32((unsigned short)(raw >> 16u))) << 16u);
        input_f16[column >> 1u] = converted.pair;
    }
    __syncthreads();
    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        const zllm_f16x2* source = input_f16 + group * 16u;
        float sum = 0.0f;
        #pragma unroll 8
        for (unsigned int pair = 0u; pair < 16u; ++pair) {
            sum += (float)source[pair].x + (float)source[pair].y;
        }
        group_bias[group] = 1152.0f * sum;
    }
    __syncthreads();

    if (value < value_dim) {
        const unsigned int weight_row =
            (weight_head_start + head) * kv_head_dim + nope_dim + value;
        const unsigned char* weight_bytes = reinterpret_cast<const unsigned char*>(packed)
            + (unsigned long long)weight_row * latent_dim;
        float sum = 0.0f;
        for (unsigned int col0 = lane * 8u; col0 < latent_dim; col0 += 128u) {
            // 8 列 = uint2 码字（8B 对齐：col0 为 8 的倍数、行基址 256B 对齐）。
            const uint2 codes = *reinterpret_cast<const uint2*>(weight_bytes + col0);
            const float scale = zllm_scale(
                scales, (unsigned long long)weight_row * groups + (col0 >> 5u), scale_dtype);
            const zllm_f16x2* source = input_f16 + (col0 >> 1u);
            float group_dot = -group_bias[col0 >> 5u];
            group_dot = zllm_f16x2_dot2(
                source[0], zllm_w8_perm_f16(codes.x, 0x0504u), group_dot);
            group_dot = zllm_f16x2_dot2(
                source[1], zllm_w8_perm_f16(codes.x, 0x0706u), group_dot);
            group_dot = zllm_f16x2_dot2(
                source[2], zllm_w8_perm_f16(codes.y, 0x0504u), group_dot);
            group_dot = zllm_f16x2_dot2(
                source[3], zllm_w8_perm_f16(codes.y, 0x0706u), group_dot);
            sum += group_dot * scale;
        }
        for (unsigned int offset = 8u; offset > 0u; offset >>= 1u)
            sum += __shfl_down(sum, offset, 16);
        if (lane == 0u)
            output[(unsigned long long)head * q_head_dim + value] = sum;
    }
    const unsigned int tail_dim = q_head_dim - value_dim;
    if (blockIdx.x % value_tiles == 0u) {
        for (unsigned int index = threadIdx.x; index < tail_dim; index += blockDim.x)
            output[(unsigned long long)head * q_head_dim + value_dim + index] = 0.0f;
    }
}

// selected decode 的 latent 预重排：把 top-k 选中行的 Q8 code/scale/rope 按
// selection 顺序收集到连续 workspace，scan 的 4 个 head-group 全部由随机散读
// 变为一次散读 + 连续读。物理行号经 block_table 映射（identity 时即 token）。
extern "C" __global__ void mla_gather_selected_q8(
    const signed char* latent,
    const unsigned short* latent_scales,
    const unsigned short* rope,
    const unsigned int* block_table,
    const unsigned int* selection,
    signed char* gathered_latent,
    unsigned short* gathered_scales,
    unsigned short* gathered_rope,
    unsigned int count,
    unsigned int latent_dim,
    unsigned int latent_group_size,
    unsigned int rope_dim,
    unsigned int block_size)
{
    const unsigned int index = blockIdx.x;
    if (index >= count || latent_group_size == 0) return;
    const unsigned int token = selection[index];
    const unsigned long long physical = paged_row(block_table, token, block_size);
    const unsigned int groups = latent_dim / latent_group_size;
    const unsigned int lane = threadIdx.x;
    // 行内 576B code + 16B scale + 128B rope，整行连续装载。
    for (unsigned int column = lane; column < latent_dim; column += blockDim.x) {
        gathered_latent[(unsigned long long)index * latent_dim + column]
            = latent[physical * latent_dim + column];
    }
    for (unsigned int group = lane; group < groups; group += blockDim.x) {
        gathered_scales[index * groups + group]
            = latent_scales[physical * groups + group];
    }
    for (unsigned int column = lane; column < rope_dim; column += blockDim.x) {
        gathered_rope[(unsigned long long)index * rope_dim + column]
            = rope[physical * rope_dim + column];
    }
}
