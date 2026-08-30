/// 本模块的 CUDA shader(本文件用到的 kernel + 文件私有 `__device__` helper)。
///
/// 共用 helper 见 `mod.rs` 前导。`mod.rs` 的 `kernels_source()`
/// 把 mod.rs 的 `PREAMBLE_SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: sigmoid_gate_f16, gemma_rmsnorm_heads_f16, append_rows_f16, gqa_attention_f16, gqa_attention_f16_flash, gqa_attention_f16_tiled, q8g64_quantize_rows, gqa_attention_q8g64, gqa_attention_q8g64_flash, gated_delta_conv_f16, gated_delta_recurrent_f16, gated_delta_norm_gate_f16
pub const SHADERS: &str = r#"
extern "C" __global__ void sigmoid_gate_f16(const __half *input, const __half *gate, __half *output, unsigned int columns, unsigned int gate_columns, unsigned int count) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int row = id / columns;
    unsigned int column = id - row * columns;
    float g = __half2float(gate[row * gate_columns + (gate_columns == 1 ? 0 : column)]);
    output[id] = __float2half(__half2float(input[id]) / (1.0f + expf(-g)));
}
extern "C" __global__ void gemma_rmsnorm_heads_f16(const __half *input, const __half *weight, __half *output, unsigned int head_count, unsigned int head_dim, float eps) {
    extern __shared__ float sums[];
    unsigned int lane = threadIdx.x;
    unsigned int group = blockIdx.x;
    unsigned long long begin = (unsigned long long)group * head_dim;
    float sum = 0.0f;
    for (unsigned int column = lane; column < head_dim; column += blockDim.x) {
        float value = __half2float(input[begin + column]);
        sum += value * value;
    }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    float scale = rsqrtf(sums[0] / head_dim + eps);
    for (unsigned int column = lane; column < head_dim; column += blockDim.x) {
        float value = __half2float(input[begin + column]);
        output[begin + column] = __float2half(value * scale * (1.0f + __half2float(weight[column])));
    }
}
// GemmaRMSNorm + residual fusion:output = gemma_rmsnorm(input + residual)。
// 等价先 add(input, residual) 再 gemma_rmsnorm,但省一份 sum tensor 的读写 + 一次 kernel launch。
// 类 LLaMA 每层 1 次:hidden + mixed → post_attention_norm → ffn_input(原本 2 次 kernel)。
extern "C" __global__ void gemma_rmsnorm_residual_heads_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ residual,
    const __half * __restrict__ weight,
    __half * __restrict__ output,
    unsigned int head_count,
    unsigned int head_dim,
    float eps)
{
    extern __shared__ float sums[];
    unsigned int lane = threadIdx.x;
    unsigned int group = blockIdx.x;
    unsigned long long begin = (unsigned long long)group * head_dim;
    // 1. 读 input + residual,累加 sum²(写成 fused load + add:每次循环只一次读主存 → shared 复用)。
    //    元素数 = rows * cols = head_count * head_dim,假设 hidden 已在 device 上,residual 来自上一次计算。
    float sum = 0.0f;
    for (unsigned int column = lane; column < head_dim; column += blockDim.x) {
        float v = __half2float(input[begin + column]) + __half2float(residual[begin + column]);
        sum += v * v;
    }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    float scale = rsqrtf(sums[0] / head_dim + eps);
    for (unsigned int column = lane; column < head_dim; column += blockDim.x) {
        float v = __half2float(input[begin + column]) + __half2float(residual[begin + column]);
        output[begin + column] = __float2half(v * scale * (1.0f + __half2float(weight[column])));
    }
}
extern "C" __global__ void append_rows_f16(const __half *input, __half *output, unsigned int position, unsigned int columns, unsigned int count) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) output[(unsigned long long)position * columns + id] = input[id];
}
extern "C" __global__ void gqa_attention_f16(const __half *query, const __half *key, const __half *value, __half *output, unsigned int query_rows, unsigned int kv_rows, unsigned int position, unsigned int heads, unsigned int kv_heads, unsigned int head_dim, float attention_scale) {
    extern __shared__ float shared[];
    unsigned int block = blockIdx.x;
    unsigned int row = block / heads;
    unsigned int head = block - row * heads;
    unsigned int lane = threadIdx.x;
    unsigned int warp = lane >> 5;
    unsigned int warp_lane = lane & 31;
    unsigned int warp_count = blockDim.x >> 5;
    unsigned int kv_head = head / (heads / kv_heads);
    unsigned long long q_base = ((unsigned long long)row * heads + head) * head_dim;
    float *partial_maximum = shared;
    float *partial_denominator = shared + warp_count;
    float *partial_values = shared + warp_count * 2;
    for (unsigned int index = lane; index < warp_count * head_dim; index += blockDim.x) {
        partial_values[index] = 0.0f;
    }
    __syncthreads();

    // 每个 warp 独立扫描一段 KV，最后合并在线 softmax，避免按 token 做 block 同步。
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    unsigned int last = position + row;
    if (last >= kv_rows) last = kv_rows - 1;
    for (unsigned int token = warp; token <= last; token += warp_count) {
        unsigned long long kv_base = ((unsigned long long)token * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32) {
            partial += __half2float(query[q_base + dimension]) * __half2float(key[kv_base + dimension]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffff, partial, offset);
        float old_scale = 0.0f;
        float new_scale = 0.0f;
        if (warp_lane == 0) {
            float score = partial * attention_scale;
            float next_maximum = fmaxf(maximum, score);
            old_scale = expf(maximum - next_maximum);
            new_scale = expf(score - next_maximum);
            denominator = denominator * old_scale + new_scale;
            maximum = next_maximum;
        }
        old_scale = __shfl_sync(0xffffffff, old_scale, 0);
        new_scale = __shfl_sync(0xffffffff, new_scale, 0);
        float *accumulator = partial_values + warp * head_dim;
        for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32) {
            accumulator[dimension] = accumulator[dimension] * old_scale
                + new_scale * __half2float(value[kv_base + dimension]);
        }
    }
    if (warp_lane == 0) {
        partial_maximum[warp] = maximum;
        partial_denominator[warp] = denominator;
    }
    __syncthreads();

    for (unsigned int dimension = lane; dimension < head_dim; dimension += blockDim.x) {
        float global_maximum = -3.402823466e+38F;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) {
                global_maximum = fmaxf(global_maximum, partial_maximum[item]);
            }
        }
        float global_denominator = 0.0f;
        float accumulator = 0.0f;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) {
                float scale = expf(partial_maximum[item] - global_maximum);
                global_denominator += partial_denominator[item] * scale;
                accumulator += partial_values[item * head_dim + dimension] * scale;
            }
        }
        output[q_base + dimension] = __float2half(accumulator / global_denominator);
    }
}
// 共享内存 tiled 版 GQA attention:把 KV 分块加载到 smem,消除对 HBM 的每 token 重读,
// 适合 seq_len >> warp_count 的长序列(L2 thrash 场景)。block 维度=(row, head),每 block 256
// 线程;Q 用每个 lane 的寄存器数组 q_reg[dims_per_lane];KV 按 block_kv 步长推进,先合作
// 加载 K_tile/V_tile 到 smem,再每个 warp 串行走完当前 tile 的 token、做在线 softmax 更新。
// 最后跨 warp 合并(同 gqa_attention_f16)。layout:
//   smem_raw[0 .. block_kv*head_dim]           = K_tile (f16)
//   smem_raw[..] + 2*block_kv*head_dim          = V_tile (f16)
//   smem_raw[..] + 2*block_kv*head_dim*2(f16B)  = partial_maximum[warp_count]
//   ... + 1*warp_count                          = partial_denominator[warp_count]
//   ... + 1*warp_count                          = partial_values[warp_count * head_dim]

// Flash 式 GQA prefill attention:block = 8 行 × 1 head(8 warp,每 warp 一行)。K/V 的
// 32-token tile 由全 block 合作加载进 smem(stride = head_dim+2 个 half,word stride 为
// 奇数 → lane=token 逐维读取无 bank 冲突),8 行共享同一 tile 把 KV 全局重复读缩小 8 倍
// (旧版 5 head × 全部行重复读 KV,DRAM 368GB/s 打满)。warp 内 lane=token 做完整 QK dot
// (零跨 lane 归约;tile 的 max/denom 各 5 次 shuffle 摊销到 32 token),softmax 权重经
// smem 广播后切 lane=dim 累加 V。跨 tile 在线 rescale;每 warp 独占一行,无需跨 warp 合并。
#define FLASH_ROWS 8u
#define FLASH_TILE 32u
extern "C" __global__ void gqa_attention_f16_flash(
    const __half * __restrict__ query,
    const __half * __restrict__ key,
    const __half * __restrict__ value,
    __half * __restrict__ output,
    unsigned int query_rows,
    unsigned int kv_rows,
    unsigned int position,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    float attention_scale)
{
    extern __shared__ unsigned char smem_raw[];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * FLASH_ROWS + warp;
    const unsigned int head = blockIdx.y;
    const unsigned int kv_head = head / (heads / kv_heads);
    const bool row_valid = row < query_rows;
    unsigned int last = position + row;
    if (last >= kv_rows) last = kv_rows - 1;

    // smem:K_tile + V_tile(各 FLASH_TILE × stride 个 half)+ 每 warp 32 个 softmax 权重。
    const unsigned int stride = head_dim + 2u;
    __half *K_tile = reinterpret_cast<__half *>(smem_raw);
    __half *V_tile = K_tile + FLASH_TILE * stride;
    float *weights = reinterpret_cast<float *>(V_tile + FLASH_TILE * stride) + warp * FLASH_TILE;

    const __half *q_row = query + ((unsigned long long)row * heads + head) * head_dim;
    const unsigned long long kv_base = (unsigned long long)kv_head * head_dim;
    const unsigned int dims_per_lane = head_dim >> 5;

    // block 内最长因果长度(8 行的 last 至多差 7,tile 浪费可忽略)。
    unsigned int block_last = position + blockIdx.x * FLASH_ROWS + FLASH_ROWS - 1u;
    if (block_last >= kv_rows) block_last = kv_rows - 1;

    float accumulator[8];
    #pragma unroll
    for (unsigned int d = 0; d < 8; ++d) accumulator[d] = 0.0f;
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;

    for (unsigned int tile = 0; tile <= block_last; tile += FLASH_TILE) {
        // 合作加载 K/V tile:连续 idx → 行内连续 half,全局合并访问;越界 token 填 0。
        for (unsigned int index = threadIdx.x; index < FLASH_TILE * head_dim; index += blockDim.x) {
            unsigned int t = index / head_dim;
            unsigned int d = index - t * head_dim;
            unsigned int token = tile + t;
            __half k = __float2half(0.0f);
            __half v = __float2half(0.0f);
            if (token <= block_last) {
                unsigned long long kv = (unsigned long long)token * kv_heads * head_dim + kv_base + d;
                k = key[kv];
                v = value[kv];
            }
            K_tile[t * stride + d] = k;
            V_tile[t * stride + d] = v;
        }
        __syncthreads();

        // QK dot:lane=token,q 全 lane 同址广播(K/L1),k 从 smem(奇 word stride 无冲突)。
        unsigned int token = tile + lane;
        bool active = row_valid && token <= last;
        float score = -3.402823466e+38F;
        if (active) {
            const __half *k = K_tile + lane * stride;
            float partial = 0.0f;
            for (unsigned int d = 0; d < head_dim; d += 2u) {
                float2 qq = __half22float2(*reinterpret_cast<const __half2 *>(q_row + d));
                float2 kk = __half22float2(*reinterpret_cast<const __half2 *>(k + d));
                partial += qq.x * kk.x + qq.y * kk.y;
            }
            score = partial * attention_scale;
        }
        // tile 内 softmax:max/denom 各 5 次 shuffle,摊销到 32 个 token。
        float tile_max = score;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffffu, tile_max, offset));
        tile_max = __shfl_sync(0xffffffffu, tile_max, 0);
        float next_maximum = fmaxf(maximum, tile_max);
        float weight = active ? expf(score - next_maximum) : 0.0f;
        float tile_denom = weight;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) tile_denom += __shfl_down_sync(0xffffffffu, tile_denom, offset);
        tile_denom = __shfl_sync(0xffffffffu, tile_denom, 0);
        float old_scale = expf(maximum - next_maximum);
        maximum = next_maximum;
        denominator = denominator * old_scale + tile_denom;
        weights[lane] = weight;
        __syncwarp();

        // V 累加:lane=dim(每 lane dims_per_lane 个维度),权重经 smem 广播。
        #pragma unroll
        for (unsigned int d = 0; d < 8; ++d) {
            if (d < dims_per_lane) {
                unsigned int dimension = lane + d * 32u;
                float sum = 0.0f;
                for (unsigned int t = 0; t < FLASH_TILE; ++t) {
                    sum += weights[t] * __half2float(V_tile[t * stride + dimension]);
                }
                accumulator[d] = accumulator[d] * old_scale + sum;
            }
        }
        __syncthreads();
    }

    if (row_valid) {
        __half *out = output + ((unsigned long long)row * heads + head) * head_dim;
        float inverse = 1.0f / denominator;
        #pragma unroll
        for (unsigned int d = 0; d < 8; ++d) {
            if (d < dims_per_lane) out[lane + d * 32u] = __float2half(accumulator[d] * inverse);
        }
    }
}


// ===================== Q8G64(symmetric INT8 + per-64 组 scale)KV cache =====================
// 布局镜像 Metal 的 Ornith Q8G64:codes 为 signed int8、行布局与 f16 相同
// `[token, kv_heads*head_dim]`;scales 行优先 `[token, columns/64]`,每 64 列一个
// f32(scale = amax/127,解码 w = code × scale)。KV 显存 ~49% 于 f16。

// append 时量化:f16 K/V 行写入为 codes + scales。每线程负责一个 (row, 64-group)。
extern "C" __global__ void q8g64_quantize_rows(
    const __half * __restrict__ input,
    signed char * __restrict__ codes,
    float * __restrict__ scales,
    const unsigned int position,
    const unsigned int columns,
    const unsigned int count)
{
    const unsigned int groups = columns >> 6;
    const unsigned int total = count * groups;
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += gridDim.x * blockDim.x) {
        unsigned int row = index / groups;
        unsigned int group = index - row * groups;
        // 写入目标按 position 偏移:decode 追加必须落到绝对行,否则覆盖 prefill KV。
        unsigned int target = position + row;
        unsigned int begin = row * columns + group * 64u;
        unsigned int target_begin = target * columns + group * 64u;
        float maximum = 0.0f;
        for (unsigned int j = 0; j < 64u; ++j) {
            maximum = fmaxf(maximum, fabsf(__half2float(input[begin + j])));
        }
        float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scales[target * groups + group] = scale;
        float inverse = 1.0f / scale;
        for (unsigned int j = 0; j < 64u; ++j) {
            float quantized = __half2float(input[begin + j]) * inverse;
            codes[target_begin + j] = (signed char)fminf(127.0f, fmaxf(-127.0f, rintf(quantized)));
        }
    }
}

// decode 用 Q8G64 KV 的 GQA attention:与 gqa_attention_f16 同结构,K/V 读取时
// codes × scale 内联反量化(scale 按 (token*columns + 列)/64 定位)。
extern "C" __global__ void gqa_attention_q8g64(
    const __half * __restrict__ query,
    const signed char * __restrict__ key_codes,
    const float * __restrict__ key_scales,
    const signed char * __restrict__ value_codes,
    const float * __restrict__ value_scales,
    __half * __restrict__ output,
    unsigned int query_rows,
    unsigned int kv_rows,
    unsigned int position,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    float attention_scale)
{
    extern __shared__ float shared[];
    unsigned int block = blockIdx.x;
    unsigned int row = block / heads;
    unsigned int head = block - row * heads;
    unsigned int lane = threadIdx.x;
    unsigned int warp = lane >> 5;
    unsigned int warp_lane = lane & 31u;
    unsigned int warp_count = blockDim.x >> 5;
    unsigned int kv_head = head / (heads / kv_heads);
    unsigned int groups_per_row = (kv_heads * head_dim) >> 6;
    unsigned long long q_base = ((unsigned long long)row * heads + head) * head_dim;
    unsigned long long kv_head_column = (unsigned long long)kv_head * head_dim;
    float *partial_maximum = shared;
    float *partial_denominator = shared + warp_count;
    float *partial_values = shared + warp_count * 2;
    for (unsigned int index = lane; index < warp_count * head_dim; index += blockDim.x) {
        partial_values[index] = 0.0f;
    }
    __syncthreads();

    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    unsigned int last = position + row;
    if (last >= kv_rows) last = kv_rows - 1;
    for (unsigned int token = warp; token <= last; token += warp_count) {
        unsigned long long kv_base = ((unsigned long long)token * kv_heads + kv_head) * head_dim;
        unsigned long long scale_base = (unsigned long long)token * groups_per_row + kv_head_column / 64u;
        float partial = 0.0f;
        for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32u) {
            float k = (float)key_codes[kv_base + dimension] * key_scales[scale_base + dimension / 64u];
            partial += __half2float(query[q_base + dimension]) * k;
        }
        for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, offset);
        float old_scale = 0.0f;
        float new_scale = 0.0f;
        if (warp_lane == 0u) {
            float score = partial * attention_scale;
            float next_maximum = fmaxf(maximum, score);
            old_scale = expf(maximum - next_maximum);
            new_scale = expf(score - next_maximum);
            denominator = denominator * old_scale + new_scale;
            maximum = next_maximum;
        }
        old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
        new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
        float *accumulator = partial_values + warp * head_dim;
        for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32u) {
            float v = (float)value_codes[kv_base + dimension] * value_scales[scale_base + dimension / 64u];
            accumulator[dimension] = accumulator[dimension] * old_scale + new_scale * v;
        }
    }
    if (warp_lane == 0u) {
        partial_maximum[warp] = maximum;
        partial_denominator[warp] = denominator;
    }
    __syncthreads();
    for (unsigned int dimension = lane; dimension < head_dim; dimension += blockDim.x) {
        float global_maximum = -3.402823466e+38F;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) global_maximum = fmaxf(global_maximum, partial_maximum[item]);
        }
        float global_denominator = 0.0f;
        float accumulator = 0.0f;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) {
                float rescale = expf(partial_maximum[item] - global_maximum);
                global_denominator += partial_denominator[item] * rescale;
                accumulator += partial_values[item * head_dim + dimension] * rescale;
            }
        }
        output[q_base + dimension] = __float2half(accumulator / global_denominator);
    }
}

// prefill flash 的 Q8G64 变体:tile 协作加载时 codes × scale 反量化进现有 smem f16
// 布局,主体循环与 gqa_attention_f16_flash 完全一致。
extern "C" __global__ void gqa_attention_q8g64_flash(
    const __half * __restrict__ query,
    const signed char * __restrict__ key_codes,
    const float * __restrict__ key_scales,
    const signed char * __restrict__ value_codes,
    const float * __restrict__ value_scales,
    __half * __restrict__ output,
    unsigned int query_rows,
    unsigned int kv_rows,
    unsigned int position,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    float attention_scale)
{
    extern __shared__ unsigned char smem_raw[];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8u + warp;
    const unsigned int head = blockIdx.y;
    const unsigned int kv_head = head / (heads / kv_heads);
    const bool row_valid = row < query_rows;
    unsigned int last = position + row;
    if (last >= kv_rows) last = kv_rows - 1;

    const unsigned int stride = head_dim + 2u;
    __half *K_tile = reinterpret_cast<__half *>(smem_raw);
    __half *V_tile = K_tile + 32u * stride;
    float *weights = reinterpret_cast<float *>(V_tile + 32u * stride) + warp * 32u;

    const __half *q_row = query + ((unsigned long long)row * heads + head) * head_dim;
    const unsigned long long kv_base = (unsigned long long)kv_head * head_dim;
    const unsigned int columns = kv_heads * head_dim;
    const unsigned int groups_per_row = columns >> 6;
    const unsigned int dims_per_lane = head_dim >> 5;

    unsigned int block_last = position + blockIdx.x * 8u + 7u;
    if (block_last >= kv_rows) block_last = kv_rows - 1;

    float accumulator[8];
    #pragma unroll
    for (unsigned int d = 0; d < 8; ++d) accumulator[d] = 0.0f;
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;

    for (unsigned int tile = 0; tile <= block_last; tile += 32u) {
        for (unsigned int index = threadIdx.x; index < 32u * head_dim; index += blockDim.x) {
            unsigned int t = index / head_dim;
            unsigned int d = index - t * head_dim;
            unsigned int token = tile + t;
            __half k = __float2half(0.0f);
            __half v = __float2half(0.0f);
            if (token <= block_last) {
                unsigned long long kv = (unsigned long long)token * columns + kv_base + d;
                float scale = key_scales[(unsigned long long)token * groups_per_row + (kv_base + d) / 64u];
                k = __float2half((float)key_codes[kv] * scale);
                scale = value_scales[(unsigned long long)token * groups_per_row + (kv_base + d) / 64u];
                v = __float2half((float)value_codes[kv] * scale);
            }
            K_tile[t * stride + d] = k;
            V_tile[t * stride + d] = v;
        }
        __syncthreads();

        unsigned int token = tile + lane;
        bool active = row_valid && token <= last;
        float score = -3.402823466e+38F;
        if (active) {
            const __half *k = K_tile + lane * stride;
            float partial = 0.0f;
            for (unsigned int d = 0; d < head_dim; d += 2u) {
                float2 qq = __half22float2(*reinterpret_cast<const __half2 *>(q_row + d));
                float2 kk = __half22float2(*reinterpret_cast<const __half2 *>(k + d));
                partial += qq.x * kk.x + qq.y * kk.y;
            }
            score = partial * attention_scale;
        }
        float tile_max = score;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffffu, tile_max, offset));
        tile_max = __shfl_sync(0xffffffffu, tile_max, 0);
        float next_maximum = fmaxf(maximum, tile_max);
        float weight = active ? expf(score - next_maximum) : 0.0f;
        float tile_denom = weight;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) tile_denom += __shfl_down_sync(0xffffffffu, tile_denom, offset);
        tile_denom = __shfl_sync(0xffffffffu, tile_denom, 0);
        float old_scale = expf(maximum - next_maximum);
        maximum = next_maximum;
        denominator = denominator * old_scale + tile_denom;
        weights[lane] = weight;
        __syncwarp();

        #pragma unroll
        for (unsigned int d = 0; d < 8; ++d) {
            if (d < dims_per_lane) {
                unsigned int dimension = lane + d * 32u;
                float sum = 0.0f;
                for (unsigned int t = 0; t < 32u; ++t) {
                    sum += weights[t] * __half2float(V_tile[t * stride + dimension]);
                }
                accumulator[d] = accumulator[d] * old_scale + sum;
            }
        }
        __syncthreads();
    }

    if (row_valid) {
        __half *out = output + ((unsigned long long)row * heads + head) * head_dim;
        float inverse = 1.0f / denominator;
        #pragma unroll
        for (unsigned int d = 0; d < 8; ++d) {
            if (d < dims_per_lane) out[lane + d * 32u] = __float2half(accumulator[d] * inverse);
        }
    }
}

extern "C" __global__ void gqa_attention_f16_tiled(const __half *query, const __half *key, const __half *value, __half *output, unsigned int query_rows, unsigned int kv_rows, unsigned int position, unsigned int heads, unsigned int kv_heads, unsigned int head_dim, unsigned int block_kv, float attention_scale) {
    extern __shared__ unsigned char smem_raw[];
    unsigned int block = blockIdx.x;
    unsigned int row = block / heads;
    unsigned int head = block - row * heads;
    unsigned int lane = threadIdx.x;
    unsigned int warp = lane >> 5;
    unsigned int warp_lane = lane & 31;
    unsigned int warp_count = blockDim.x >> 5;
    unsigned int kv_head = head / (heads / kv_heads);
    unsigned long long q_base = ((unsigned long long)row * heads + head) * head_dim;

    __half *K_tile = reinterpret_cast<__half *>(smem_raw);
    __half *V_tile = K_tile + block_kv * head_dim;
    float *partial_maximum = reinterpret_cast<float *>(V_tile + block_kv * head_dim);
    float *partial_denominator = partial_maximum + warp_count;
    float *partial_values = partial_denominator + warp_count;

    // 把 Q 加载到寄存器:每个 lane 负责 head_dim/32 个维度,dims_per_lane = head_dim/32。
    constexpr unsigned int MAX_DIMS = 16;
    unsigned int dims_per_lane = head_dim / 32;
    float q_reg[MAX_DIMS];
    #pragma unroll
    for (unsigned int i = 0; i < MAX_DIMS; ++i) q_reg[i] = 0.0f;
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        if (i < dims_per_lane) {
            q_reg[i] = __half2float(query[q_base + warp_lane * dims_per_lane + i]);
        }
    }

    // 每个 warp 独立维护在线 softmax 状态 (max, denom),部分和直接写 partial_values,
    // 跨 warp 合并用 partial_values + partial_maximum + partial_denominator。
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;

    unsigned int last = position + row;
    if (last >= kv_rows) last = kv_rows - 1;

    // partial_values 初始化
    for (unsigned int index = lane; index < warp_count * head_dim; index += blockDim.x) {
        partial_values[index] = 0.0f;
    }
    __syncthreads();

    // tile 循环:每个 tile 加载 block_kv 个 token到 smem,然后每个 warp 走完本 tile。
    unsigned int token_base = 0;
    // 把 partial_maximum/denominator 也用于暂存每个 warp 当前的 state,避免重读 smem。
    while (token_base <= last) {
        unsigned int tile_end = token_base + block_kv;
        if (tile_end > last + 1) tile_end = last + 1;
        unsigned int tile_size = tile_end - token_base;

        // 合作加载 K_tile (block_kv * head_dim 个 half)
        unsigned int kv_total = tile_size * head_dim;
        for (unsigned int idx = lane; idx < kv_total; idx += blockDim.x) {
            unsigned int t = idx / head_dim;
            unsigned int d = idx - t * head_dim;
            unsigned int token = token_base + t;
            unsigned long long kv_base = ((unsigned long long)token * kv_heads + kv_head) * head_dim;
            K_tile[t * head_dim + d] = key[kv_base + d];
        }
        // 合作加载 V_tile
        for (unsigned int idx = lane; idx < kv_total; idx += blockDim.x) {
            unsigned int t = idx / head_dim;
            unsigned int d = idx - t * head_dim;
            unsigned int token = token_base + t;
            unsigned long long kv_base = ((unsigned long long)token * kv_heads + kv_head) * head_dim;
            V_tile[t * head_dim + d] = value[kv_base + d];
        }
        __syncthreads();

        // 每个 warp 走完当前 tile
        for (unsigned int t = warp; t < tile_size; t += warp_count) {
            unsigned int token = token_base + t;
            __half *K_row = K_tile + t * head_dim;
            __half *V_row = V_tile + t * head_dim;
            // 1. 计算 Q·K (sum over dims_per_lane per lane)
            float partial = 0.0f;
            #pragma unroll
            for (unsigned int i = 0; i < 16; ++i) {
                if (i < dims_per_lane) {
                    unsigned int dim = warp_lane * dims_per_lane + i;
                    partial += q_reg[i] * __half2float(K_row[dim]);
                }
            }
            // 32-lane 求和
            for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffff, partial, offset);
            float old_scale = 0.0f;
            float new_scale = 0.0f;
            if (warp_lane == 0) {
                float score = partial * attention_scale;
                float next_maximum = fmaxf(maximum, score);
                old_scale = expf(maximum - next_maximum);
                new_scale = expf(score - next_maximum);
                denominator = denominator * old_scale + new_scale;
                maximum = next_maximum;
            }
            old_scale = __shfl_sync(0xffffffff, old_scale, 0);
            new_scale = __shfl_sync(0xffffffff, new_scale, 0);

            // 2. 更新 partial_values[warp] (rescale + add new_scale * V)
            float *accumulator = partial_values + warp * head_dim;
            #pragma unroll
            for (unsigned int i = 0; i < 16; ++i) {
                if (i < dims_per_lane) {
                    unsigned int dim = warp_lane * dims_per_lane + i;
                    accumulator[dim] = accumulator[dim] * old_scale
                        + new_scale * __half2float(V_row[dim]);
                }
            }
        }
        __syncthreads();
        token_base = tile_end;
    }

    // 把每个 warp 的 final max/denom 写入 smem,准备合并
    if (warp_lane == 0) {
        partial_maximum[warp] = maximum;
        partial_denominator[warp] = denominator;
    }
    __syncthreads();

    // 跨 warp 合并:只让 warp 0 处理 head_dim/32 维,其余 warp 写出的值相同、纯属冗余写。
    if (warp == 0) {
    #pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        if (i < dims_per_lane) {
            unsigned int dim = warp_lane * dims_per_lane + i;
            float global_maximum = -3.402823466e+38F;
            for (unsigned int item = 0; item < warp_count; ++item) {
                if (partial_denominator[item] > 0.0f) {
                    global_maximum = fmaxf(global_maximum, partial_maximum[item]);
                }
            }
            float global_denominator = 0.0f;
            float acc_sum = 0.0f;
            for (unsigned int item = 0; item < warp_count; ++item) {
                if (partial_denominator[item] > 0.0f) {
                    float scale = expf(partial_maximum[item] - global_maximum);
                    global_denominator += partial_denominator[item] * scale;
                    acc_sum += partial_values[item * head_dim + dim] * scale;
                }
            }
            output[q_base + dim] = __float2half(acc_sum / global_denominator);
        }
    }
    }
}
// 全序列 self-attention 的 batched 版本:Q/K/V 连续存放 `batch` 个等长序列(各 `rows` 行),
// 每个 query 行只 attend 自己 batch 内的 KV(严格隔离,跨 batch 边界为零权重)。与
// `gqa_attention_f16` 同源(多 warp 在线 softmax + 跨 warp 合并),区别仅在 KV 扫描范围:
// 由 `row / rows` 算出 batch 索引,把 token 循环钳到 `[batch_idx*rows, batch_idx*rows+rows)`。
// 无 GQA(kv_heads==heads)、无 causal(position 概念移除),Q/K 的 RoPE 由调用方预先施加。
extern "C" __global__ void full_attention_batched_f16(const __half * __restrict__ query, const __half * __restrict__ key, const __half * __restrict__ value, __half * __restrict__ output, unsigned int batch, unsigned int rows, unsigned int heads, unsigned int head_dim, float attention_scale) {
    extern __shared__ float shared[];
    unsigned int block = blockIdx.x;
    unsigned int row = block / heads;
    unsigned int head = block - row * heads;
    unsigned int batch_idx = row / rows;          // 该 query 行所属 batch
    unsigned int kv_start = batch_idx * rows;     // 本 batch KV 扫描起点
    unsigned int kv_last = kv_start + rows - 1;   // 本 batch KV 扫描终点(含)
    unsigned int lane = threadIdx.x;
    unsigned int warp = lane >> 5;
    unsigned int warp_lane = lane & 31;
    unsigned int warp_count = blockDim.x >> 5;
    unsigned long long q_base = ((unsigned long long)row * heads + head) * head_dim;
    float *partial_maximum = shared;
    float *partial_denominator = shared + warp_count;
    float *partial_values = shared + warp_count * 2;
    for (unsigned int index = lane; index < warp_count * head_dim; index += blockDim.x) {
        partial_values[index] = 0.0f;
    }
    __syncthreads();

    // 每个 warp 独立扫描一段本 batch 内的 KV,最后合并在线 softmax。
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    for (unsigned int token = kv_start + warp; token <= kv_last; token += warp_count) {
        unsigned long long kv_base = ((unsigned long long)token * heads + head) * head_dim;
        float partial = 0.0f;
        for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32) {
            partial += __half2float(query[q_base + dimension]) * __half2float(key[kv_base + dimension]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffff, partial, offset);
        float old_scale = 0.0f;
        float new_scale = 0.0f;
        if (warp_lane == 0) {
            float score = partial * attention_scale;
            float next_maximum = fmaxf(maximum, score);
            old_scale = expf(maximum - next_maximum);
            new_scale = expf(score - next_maximum);
            denominator = denominator * old_scale + new_scale;
            maximum = next_maximum;
        }
        old_scale = __shfl_sync(0xffffffff, old_scale, 0);
        new_scale = __shfl_sync(0xffffffff, new_scale, 0);
        float *accumulator = partial_values + warp * head_dim;
        for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32) {
            accumulator[dimension] = accumulator[dimension] * old_scale
                + new_scale * __half2float(value[kv_base + dimension]);
        }
    }
    if (warp_lane == 0) {
        partial_maximum[warp] = maximum;
        partial_denominator[warp] = denominator;
    }
    __syncthreads();

    for (unsigned int dimension = lane; dimension < head_dim; dimension += blockDim.x) {
        float global_maximum = -3.402823466e+38F;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) {
                global_maximum = fmaxf(global_maximum, partial_maximum[item]);
            }
        }
        float global_denominator = 0.0f;
        float accumulator = 0.0f;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) {
                float scale = expf(partial_maximum[item] - global_maximum);
                global_denominator += partial_denominator[item] * scale;
                accumulator += partial_values[item * head_dim + dimension] * scale;
            }
        }
        output[q_base + dimension] = __float2half(accumulator / global_denominator);
    }
}
extern "C" __global__ void gated_delta_conv_f16(const __half *qkv, const float *weight, float *state, __half *output, unsigned int rows, unsigned int channels, unsigned int kernel_size) {
    unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    unsigned long long base = (unsigned long long)channel * kernel_size;
    for (unsigned int row = 0; row < rows; ++row) {
        for (unsigned int item = 1; item < kernel_size; ++item) state[base + item - 1] = state[base + item];
        state[base + kernel_size - 1] = __half2float(qkv[(unsigned long long)row * channels + channel]);
        float sum = 0.0f;
        for (unsigned int item = 0; item < kernel_size; ++item) sum += state[base + item] * weight[base + item];
        output[(unsigned long long)row * channels + channel] = __float2half(sum / (1.0f + expf(-sum)));
    }
}
extern "C" __global__ void gated_delta_conv_weight_f16(const __half *qkv, const __half *weight, float *state, __half *output, unsigned int rows, unsigned int channels, unsigned int kernel_size) {
    unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    unsigned long long base = (unsigned long long)channel * kernel_size;
    for (unsigned int row = 0; row < rows; ++row) {
        for (unsigned int item = 1; item < kernel_size; ++item) state[base + item - 1] = state[base + item];
        state[base + kernel_size - 1] = __half2float(qkv[(unsigned long long)row * channels + channel]);
        float sum = 0.0f;
        for (unsigned int item = 0; item < kernel_size; ++item) sum += state[base + item] * __half2float(weight[base + item]);
        output[(unsigned long long)row * channels + channel] = __float2half(sum / (1.0f + expf(-sum)));
    }
}
extern "C" __global__ void gated_delta_recurrent_f16(const __half *mixed, const __half *alpha, const __half *beta, const float *a_log, const float *dt_bias, float *state, __half *output, unsigned int rows, unsigned int key_heads, unsigned int value_heads, unsigned int key_head_dim, unsigned int value_head_dim, unsigned int grouped_heads) {
    unsigned int value_head = blockIdx.x;
    unsigned int value_column = threadIdx.x;
    if (value_head >= value_heads || value_column >= value_head_dim) return;
    unsigned int key_head = grouped_heads ? value_head / (value_heads / key_heads) : value_head % key_heads;
    unsigned int key_dim = key_heads * key_head_dim;
    unsigned int value_dim = value_heads * value_head_dim;
    unsigned int conv_dim = key_dim * 2 + value_dim;
    unsigned long long state_head = (unsigned long long)value_head * key_head_dim * value_head_dim;
    for (unsigned int row = 0; row < rows; ++row) {
        unsigned long long mixed_row = (unsigned long long)row * conv_dim;
        unsigned long long query_base = mixed_row + (unsigned long long)key_head * key_head_dim;
        unsigned long long key_base = mixed_row + key_dim + (unsigned long long)key_head * key_head_dim;
        float query_sum = 0.0f, key_sum = 0.0f;
        for (unsigned int k = 0; k < key_head_dim; ++k) {
            float q = __half2float(mixed[query_base + k]);
            float key_value = __half2float(mixed[key_base + k]);
            query_sum += q * q; key_sum += key_value * key_value;
        }
        float query_scale = rsqrtf(fmaxf(query_sum, 1.0e-12f)) * rsqrtf((float)key_head_dim);
        float key_scale = rsqrtf(fmaxf(key_sum, 1.0e-12f));
        float step = __half2float(alpha[(unsigned long long)row * value_heads + value_head]) + dt_bias[value_head];
        float softplus = step > 20.0f ? step : (step < -20.0f ? expf(step) : log1pf(expf(step)));
        float decay = expf(-expf(a_log[value_head]) * softplus);
        float beta_value = 1.0f / (1.0f + expf(-__half2float(beta[(unsigned long long)row * value_heads + value_head])));
        unsigned long long value_index = mixed_row + key_dim * 2 + (unsigned long long)value_head * value_head_dim + value_column;
        float memory = 0.0f;
        for (unsigned int k = 0; k < key_head_dim; ++k) {
            unsigned long long state_index = state_head + (unsigned long long)k * value_head_dim + value_column;
            float decayed = state[state_index] * decay;
            state[state_index] = decayed;
            memory += decayed * __half2float(mixed[key_base + k]) * key_scale;
        }
        float delta = (__half2float(mixed[value_index]) - memory) * beta_value;
        float result = 0.0f;
        for (unsigned int k = 0; k < key_head_dim; ++k) {
            unsigned long long state_index = state_head + (unsigned long long)k * value_head_dim + value_column;
            float key_value = __half2float(mixed[key_base + k]) * key_scale;
            float updated = state[state_index] + key_value * delta;
            state[state_index] = updated;
            result += updated * __half2float(mixed[query_base + k]) * query_scale;
        }
        output[(unsigned long long)row * value_dim + (unsigned long long)value_head * value_head_dim + value_column] = __float2half(result);
    }
}
extern "C" __global__ void gated_delta_norm_gate_f16(const __half *input, const __half *gate, const float *weight, __half *output, unsigned int value_heads, unsigned int value_head_dim, float eps) {
    extern __shared__ float sums[];
    unsigned int group = blockIdx.x, lane = threadIdx.x;
    unsigned long long begin = (unsigned long long)group * value_head_dim;
    float value = lane < value_head_dim ? __half2float(input[begin + lane]) : 0.0f;
    sums[lane] = value * value;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) { if (lane < stride) sums[lane] += sums[lane + stride]; __syncthreads(); }
    if (lane < value_head_dim) {
        float g = __half2float(gate[begin + lane]);
        output[begin + lane] = __float2half(value * rsqrtf(sums[0] / value_head_dim + eps) * weight[lane] * g / (1.0f + expf(-g)));
    }
}
"#;

use super::tensor::grid_1d;
use super::{CudaContext, CudaSliceF16, CudaTensor, LaunchConfig, PushKernelArg, THREADS};
use cudarc::driver::safe::CudaSlice;

pub fn sigmoid_gate_f16(ctx: &CudaContext, input: &CudaTensor, gate: &CudaTensor) -> Result<CudaTensor, String> {
    if input.rows != gate.rows || (gate.cols != 1 && gate.cols != input.cols) {
        return Err(format!("CUDA sigmoid gate shape input=[{},{}], gate=[{},{}]", input.rows, input.cols, gate.rows, gate.cols));
    }
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("sigmoid_gate_f16")?;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(&gate.slice)
            .arg(&output.slice)
            .arg(&(input.cols as u32))
            .arg(&(gate.cols as u32))
            .arg(&(input.len() as u32))
            .launch(grid_1d(input.len()))
            .map_err(|error| format!("launch sigmoid_gate_f16: {error:?}"))?;
    }
    Ok(output)
}

pub fn gemma_rmsnorm_heads_f16(ctx: &CudaContext, input: &CudaTensor, weight: &CudaSliceF16, head_count: usize, head_dim: usize, eps: f32) -> Result<CudaTensor, String> {
    if input.cols != head_count * head_dim || weight.len() != head_dim {
        return Err(format!("CUDA Gemma head norm shape input=[{},{}], heads={head_count}, dim={head_dim}, weight={}", input.rows, input.cols, weight.len()));
    }
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("gemma_rmsnorm_heads_f16")?;
    let cfg = LaunchConfig { grid_dim: ((input.rows * head_count) as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: THREADS * std::mem::size_of::<f32>() as u32 };
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(weight).arg(&output.slice).arg(&(head_count as u32)).arg(&(head_dim as u32)).arg(&eps).launch(cfg).map_err(|error| format!("launch gemma_rmsnorm_heads_f16: {error:?}"))?;
    }
    Ok(output)
}

/// 融合 add + GemmaRMSNorm:output = gemma_rmsnorm(input + residual, weight, eps)。
/// 等价先 add(input, residual) 再 gemma_rmsnorm,但省去 sum 中间 tensor 的读写 + 一次 kernel launch。
/// 类 LLaMA 单层 run_layer 内 2 次 add + 2 次 RMSNorm,融合后每层省 1 个 full pass over hidden(约 7KB/token)。
pub fn gemma_rmsnorm_residual_heads_f16(ctx: &CudaContext, input: &CudaTensor, residual: &CudaTensor, weight: &CudaSliceF16, head_count: usize, head_dim: usize, eps: f32) -> Result<CudaTensor, String> {
    if input.rows != residual.rows || input.cols != residual.cols {
        return Err(format!("CUDA Gemma residual norm shape input=[{},{}], residual=[{},{}]", input.rows, input.cols, residual.rows, residual.cols));
    }
    if input.cols != head_count * head_dim || weight.len() != head_dim {
        return Err(format!("CUDA Gemma residual norm heads/dim input=[{},{}], heads={head_count}, dim={head_dim}, weight={}", input.rows, input.cols, weight.len()));
    }
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("gemma_rmsnorm_residual_heads_f16")?;
    let cfg = LaunchConfig { grid_dim: ((input.rows * head_count) as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: THREADS * std::mem::size_of::<f32>() as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(&residual.slice)
            .arg(weight)
            .arg(&output.slice)
            .arg(&(head_count as u32))
            .arg(&(head_dim as u32))
            .arg(&eps)
            .launch(cfg)
            .map_err(|error| format!("launch gemma_rmsnorm_residual_heads_f16: {error:?}"))?;
    }
    Ok(output)
}

pub fn append_rows_f16(ctx: &CudaContext, input: &CudaTensor, destination: &CudaSliceF16, position: usize) -> Result<(), String> {
    // kernel 按 position*columns + id 写入,先校验整个写入窗口不越出 destination。
    let end = position.checked_mul(input.cols).and_then(|base| base.checked_add(input.len())).ok_or("CUDA append_rows 偏移溢出")?;
    if end > destination.len() {
        return Err(format!("CUDA append_rows 写入越界: position={position} columns={} count={} 需要 {end} > destination={}", input.cols, input.len(), destination.len()));
    }
    let func = ctx.function("append_rows_f16")?;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(destination)
            .arg(&(position as u32))
            .arg(&(input.cols as u32))
            .arg(&(input.len() as u32))
            .launch(grid_1d(input.len()))
            .map_err(|error| format!("launch append_rows_f16: {error:?}"))?;
    }
    Ok(())
}

pub fn gqa_attention_f16(ctx: &CudaContext, query: &CudaTensor, key: &CudaSliceF16, value: &CudaSliceF16, kv_rows: usize, position: usize, spec: &crate::attention::gqa::GqaSpec) -> Result<CudaTensor, String> {
    let query_cols = spec.num_heads * spec.head_dim;
    let kv_cols = spec.num_kv_heads * spec.head_dim;
    if query.cols != query_cols || key.len() < kv_rows * kv_cols || value.len() < kv_rows * kv_cols {
        return Err(format!("CUDA GQA shape query=[{},{}], kv_rows={kv_rows}, kv_cols={kv_cols}", query.rows, query.cols));
    }
    let output = ctx.tensor_uninit(query.rows, query.cols)?;
    let func = ctx.function("gqa_attention_f16")?;
    // decode 的 head 数有限，长 KV 用更多 warp 填满 SM；prefill 保持较小 block。
    let threads = if query.rows == 1 && kv_rows >= 4096 {
        1024
    } else if query.rows == 1 && kv_rows >= 1024 {
        512
    } else {
        256
    };
    let warp_count = threads / 32;
    let shared_floats = warp_count * (spec.head_dim + 2);
    let cfg = LaunchConfig { grid_dim: ((query.rows * spec.num_heads) as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&query.slice)
            .arg(key)
            .arg(value)
            .arg(&output.slice)
            .arg(&(query.rows as u32))
            .arg(&(kv_rows as u32))
            .arg(&(position as u32))
            .arg(&(spec.num_heads as u32))
            .arg(&(spec.num_kv_heads as u32))
            .arg(&(spec.head_dim as u32))
            .arg(&spec.score_scale)
            .launch(cfg)
            .map_err(|error| format!("launch gqa_attention_f16: {error:?}"))?;
    }
    Ok(output)
}

/// 共享内存 tiled 版 GQA attention。把 KV 分块加载到 smem,减少 HBM 重读,适合长 KV。
/// `block_kv` 是每个 tile 包含的 KV token 数,默认 32(head_dim=64 时 smem ≈ 8KB)。
/// Flash 式 prefill(多行)GQA attention:block=8 行×1 head,K/V tile 合作加载进 smem
/// 由 8 行共享(KV 全局重复读缩小 8 倍),warp 内 lane=token dot + lane=dim 累加。
///
/// 消除旧版「5 head × 全部行重复读 KV 打满 DRAM」与每 token 7 次 shuffle 的结构问题;
/// 要求 `head_dim ∈ [32, 256]` 且按 32 对齐、`kv_heads` 整除 `heads`。decode(单行)继续
/// 走 `gqa_attention_f16`。
pub fn gqa_attention_f16_flash(ctx: &CudaContext, query: &CudaTensor, key: &CudaSliceF16, value: &CudaSliceF16, kv_rows: usize, position: usize, spec: &crate::attention::gqa::GqaSpec) -> Result<CudaTensor, String> {
    const THREADS: usize = 256;
    let query_cols = spec.num_heads * spec.head_dim;
    let kv_cols = spec.num_kv_heads * spec.head_dim;
    if query.rows <= 1 {
        return Err(format!("CUDA GQA flash 路径面向 prefill 多行,实际 rows={}", query.rows));
    }
    if query.cols != query_cols || key.len() < kv_rows * kv_cols || value.len() < kv_rows * kv_cols {
        return Err(format!("CUDA GQA flash shape query=[{},{}], kv_rows={kv_rows}, kv_cols={kv_cols}", query.rows, query.cols));
    }
    if spec.head_dim % 32 != 0 || spec.head_dim > 256 {
        return Err(format!("CUDA GQA flash 要求 head_dim ∈ {{32..256}} 且 head_dim % 32 == 0(实际 {})", spec.head_dim));
    }
    if spec.num_heads % spec.num_kv_heads != 0 {
        return Err(format!("CUDA GQA flash 要求 heads({}) 整除于 kv_heads({})", spec.num_heads, spec.num_kv_heads));
    }
    let output = ctx.tensor_uninit(query.rows, query_cols)?;
    let func = ctx.function("gqa_attention_f16_flash")?;
    let stride = spec.head_dim + 2;
    // smem:K/V tile(各 32×stride 个 f16)+ 8 warp × 32 个 softmax 权重(f32)。
    let shared_bytes = 2 * 32 * stride * std::mem::size_of::<half::f16>() + 8 * 32 * std::mem::size_of::<f32>();
    let row_groups = query.rows.div_ceil(8);
    let cfg = LaunchConfig { grid_dim: (row_groups as u32, spec.num_heads as u32, 1), block_dim: (THREADS as u32, 1, 1), shared_mem_bytes: shared_bytes as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&query.slice)
            .arg(key)
            .arg(value)
            .arg(&output.slice)
            .arg(&(query.rows as u32))
            .arg(&(kv_rows as u32))
            .arg(&(position as u32))
            .arg(&(spec.num_heads as u32))
            .arg(&(spec.num_kv_heads as u32))
            .arg(&(spec.head_dim as u32))
            .arg(&spec.score_scale)
            .launch(cfg)
            .map_err(|error| format!("launch gqa_attention_f16_flash: {error:?}"))?;
    }
    Ok(output)
}

/// Q8G64 append:f16 K/V 行 `[count, columns]` 量化写入 codes + per-64 组 scale。
///
/// codes 布局与 f16 行一致(signed int8);scales 行优先 `[count, columns/64]`,
/// `scale = amax/127`、`w = code × scale`。要求 columns 按 64 对齐。
pub fn q8g64_quantize_rows(ctx: &CudaContext, input: &CudaTensor, codes: &mut CudaSlice<i8>, scales: &mut CudaSlice<f32>, position: usize) -> Result<(), String> {
    let columns = input.cols;
    if columns == 0 || columns % 64 != 0 {
        return Err(format!("Q8G64 columns={columns} 必须按 64 对齐"));
    }
    let elements = position.checked_add(input.rows).and_then(|rows| rows.checked_mul(columns)).ok_or("Q8G64 写入范围溢出")?;
    if codes.len() < elements || scales.len() < elements / 64 {
        return Err(format!("Q8G64 目标容量不足 codes={}/{} scales={}/{}", codes.len(), elements, scales.len(), elements / 64));
    }
    let func = ctx.function("q8g64_quantize_rows")?;
    let total = input.rows * (columns / 64);
    let blocks = total.min(THREADS as usize * 1024).div_ceil(THREADS as usize).max(1) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(codes)
            .arg(scales)
            .arg(&(position as u32))
            .arg(&(columns as u32))
            .arg(&(input.rows as u32))
            .launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|error| format!("launch q8g64_quantize_rows: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_q8g64(
    ctx: &CudaContext,
    query: &CudaTensor,
    key_codes: &CudaSlice<i8>,
    key_scales: &CudaSlice<f32>,
    value_codes: &CudaSlice<i8>,
    value_scales: &CudaSlice<f32>,
    kv_rows: usize,
    position: usize,
    spec: &crate::attention::gqa::GqaSpec,
) -> Result<CudaTensor, String> {
    let query_cols = spec.num_heads * spec.head_dim;
    let kv_cols = spec.num_kv_heads * spec.head_dim;
    if query.cols != query_cols || kv_cols % 64 != 0 || key_codes.len() < kv_rows * kv_cols || key_scales.len() < kv_rows * kv_cols / 64 || value_codes.len() < kv_rows * kv_cols || value_scales.len() < kv_rows * kv_cols / 64 {
        return Err(format!("CUDA GQA Q8G64 shape query=[{},{}], kv_rows={kv_rows}, kv_cols={kv_cols}", query.rows, query.cols));
    }
    let output = ctx.tensor_uninit(query.rows, query_cols)?;
    let func = ctx.function("gqa_attention_q8g64")?;
    let threads = if query.rows == 1 && kv_rows >= 4096 {
        1024
    } else if query.rows == 1 && kv_rows >= 1024 {
        512
    } else {
        256
    };
    let warp_count = threads / 32;
    let shared_floats = warp_count * (spec.head_dim + 2);
    let cfg = LaunchConfig { grid_dim: ((query.rows * spec.num_heads) as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&query.slice)
            .arg(key_codes)
            .arg(key_scales)
            .arg(value_codes)
            .arg(value_scales)
            .arg(&output.slice)
            .arg(&(query.rows as u32))
            .arg(&(kv_rows as u32))
            .arg(&(position as u32))
            .arg(&(spec.num_heads as u32))
            .arg(&(spec.num_kv_heads as u32))
            .arg(&(spec.head_dim as u32))
            .arg(&spec.score_scale)
            .launch(cfg)
            .map_err(|error| format!("launch gqa_attention_q8g64: {error:?}"))?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_q8g64_flash(
    ctx: &CudaContext,
    query: &CudaTensor,
    key_codes: &CudaSlice<i8>,
    key_scales: &CudaSlice<f32>,
    value_codes: &CudaSlice<i8>,
    value_scales: &CudaSlice<f32>,
    kv_rows: usize,
    position: usize,
    spec: &crate::attention::gqa::GqaSpec,
) -> Result<CudaTensor, String> {
    const THREADS: usize = 256;
    let query_cols = spec.num_heads * spec.head_dim;
    let kv_cols = spec.num_kv_heads * spec.head_dim;
    if query.rows <= 1 {
        return Err(format!("CUDA GQA Q8G64 flash 路径面向 prefill 多行,实际 rows={}", query.rows));
    }
    if query.cols != query_cols || kv_cols % 64 != 0 || key_codes.len() < kv_rows * kv_cols || key_scales.len() < kv_rows * kv_cols / 64 || value_codes.len() < kv_rows * kv_cols || value_scales.len() < kv_rows * kv_cols / 64 {
        return Err(format!("CUDA GQA Q8G64 flash shape query=[{},{}], kv_rows={kv_rows}, kv_cols={kv_cols}", query.rows, query.cols));
    }
    if spec.head_dim % 32 != 0 || spec.head_dim > 256 || spec.num_heads % spec.num_kv_heads != 0 {
        return Err(format!("CUDA GQA Q8G64 flash 要求 head_dim ∈ {{32..256}} 且 %32==0、heads 整除于 kv_heads(实际 head_dim={}, heads={}, kv_heads={})", spec.head_dim, spec.num_heads, spec.num_kv_heads));
    }
    let output = ctx.tensor_uninit(query.rows, query_cols)?;
    let func = ctx.function("gqa_attention_q8g64_flash")?;
    let stride = spec.head_dim + 2;
    let shared_bytes = 2 * 32 * stride * std::mem::size_of::<half::f16>() + 8 * 32 * std::mem::size_of::<f32>();
    let row_groups = query.rows.div_ceil(8);
    let cfg = LaunchConfig { grid_dim: (row_groups as u32, spec.num_heads as u32, 1), block_dim: (THREADS as u32, 1, 1), shared_mem_bytes: shared_bytes as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&query.slice)
            .arg(key_codes)
            .arg(key_scales)
            .arg(value_codes)
            .arg(value_scales)
            .arg(&output.slice)
            .arg(&(query.rows as u32))
            .arg(&(kv_rows as u32))
            .arg(&(position as u32))
            .arg(&(spec.num_heads as u32))
            .arg(&(spec.num_kv_heads as u32))
            .arg(&(spec.head_dim as u32))
            .arg(&spec.score_scale)
            .launch(cfg)
            .map_err(|error| format!("launch gqa_attention_q8g64_flash: {error:?}"))?;
    }
    Ok(output)
}

pub fn gqa_attention_f16_tiled(ctx: &CudaContext, query: &CudaTensor, key: &CudaSliceF16, value: &CudaSliceF16, kv_rows: usize, position: usize, spec: &crate::attention::gqa::GqaSpec, block_kv: usize) -> Result<CudaTensor, String> {
    let query_cols = spec.num_heads * spec.head_dim;
    let kv_cols = spec.num_kv_heads * spec.head_dim;
    if query.cols != query_cols || key.len() < kv_rows * kv_cols || value.len() < kv_rows * kv_cols {
        return Err(format!("CUDA GQA tiled shape query=[{},{}], kv_rows={kv_rows}, kv_cols={kv_cols}", query.rows, query.cols));
    }
    if spec.head_dim % 32 != 0 || spec.head_dim / 32 > 16 {
        return Err(format!("CUDA GQA tiled 要求 head_dim ∈ {{32..512}} 且 head_dim % 32 == 0(实际 {head_dim})", head_dim = spec.head_dim));
    }
    let output = ctx.tensor_uninit(query.rows, query.cols)?;
    let func = ctx.function("gqa_attention_f16_tiled")?;
    // tiled 路径主要给 prefill 长 KV 用;统一 256 线程 = 8 warps,适合 block_kv=32。
    let threads = 256;
    let warp_count = threads / 32;
    // smem: K_tile + V_tile (f16) + partial_max + partial_denom + partial_values (f32)
    let kv_tile_bytes = 2 * block_kv * spec.head_dim * std::mem::size_of::<half::f16>();
    let partial_bytes = warp_count * 2 * std::mem::size_of::<f32>() + warp_count * spec.head_dim * std::mem::size_of::<f32>();
    let shared_bytes = kv_tile_bytes + partial_bytes;
    let cfg = LaunchConfig { grid_dim: ((query.rows * spec.num_heads) as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: shared_bytes as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&query.slice)
            .arg(key)
            .arg(value)
            .arg(&output.slice)
            .arg(&(query.rows as u32))
            .arg(&(kv_rows as u32))
            .arg(&(position as u32))
            .arg(&(spec.num_heads as u32))
            .arg(&(spec.num_kv_heads as u32))
            .arg(&(spec.head_dim as u32))
            .arg(&(block_kv as u32))
            .arg(&spec.score_scale)
            .launch(cfg)
            .map_err(|error| format!("launch gqa_attention_f16_tiled: {error:?}"))?;
    }
    Ok(output)
}

/// batched 全序列 self-attention(Q/K/V 连续存放 `batch` 个等长序列,各 `rows` 行;每 query 行
/// 仅 attend 自己 batch 内的 KV)。与 `gqa_attention_f16` 同源(在线 softmax),无 GQA/无 causal。
#[allow(clippy::too_many_arguments)]
pub fn full_attention_batched_f16(ctx: &CudaContext, query: &CudaTensor, key: &CudaSliceF16, value: &CudaSliceF16, batch: usize, rows: usize, heads: usize, head_dim: usize, score_scale: f32) -> Result<CudaTensor, String> {
    let total_rows = batch.checked_mul(rows).ok_or_else(|| "CUDA batched full attention rows 溢出".to_string())?;
    let cols = heads.checked_mul(head_dim).ok_or_else(|| "CUDA batched full attention columns 溢出".to_string())?;
    if batch == 0 || query.rows != total_rows || query.cols != cols || key.len() < total_rows * cols || value.len() < total_rows * cols {
        return Err(format!("CUDA batched full attention shape query=[{},{}] batch={batch} rows={rows} heads={heads} head_dim={head_dim}", query.rows, query.cols));
    }
    let output = ctx.tensor_uninit(query.rows, query.cols)?;
    let func = ctx.function("full_attention_batched_f16")?;
    // block 规模与 gqa 一致:rows 较小时 256 线程足够;rows≥1024 用更多 warp 填满 SM。
    let threads = if rows >= 1024 { 512 } else { 256 };
    let warp_count = threads / 32;
    let shared_floats = warp_count * (head_dim + 2);
    let cfg = LaunchConfig { grid_dim: ((total_rows * heads) as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&query.slice)
            .arg(key)
            .arg(value)
            .arg(&output.slice)
            .arg(&(batch as u32))
            .arg(&(rows as u32))
            .arg(&(heads as u32))
            .arg(&(head_dim as u32))
            .arg(&score_scale)
            .launch(cfg)
            .map_err(|error| format!("launch full_attention_batched_f16: {error:?}"))?;
    }
    Ok(output)
}

pub fn gated_delta_net_f16(
    ctx: &CudaContext,
    inputs: crate::attention::gated_delta_net::GatedDeltaNetInputs<'_, CudaTensor>,
    weights: crate::attention::gated_delta_net::GatedDeltaNetWeightsRef<'_, crate::backend::cuda::CudaWeight>,
    conv_state: &cudarc::driver::safe::CudaSlice<f32>,
    recurrent_state: &cudarc::driver::safe::CudaSlice<f32>,
    head_layout: crate::attention::gated_delta_net::GatedDeltaNetHeadLayout,
    spec: &crate::attention::gated_delta_net::GatedDeltaNetSpec,
) -> Result<CudaTensor, String> {
    let crate::attention::gated_delta_net::GatedDeltaNetInputs { qkv, z, alpha, beta } = inputs;
    let crate::attention::gated_delta_net::GatedDeltaNetWeightsRef { conv: conv_weight, a_log, dt_bias, norm: norm_weight } = weights;
    if qkv.cols != spec.conv_dim() || z.cols != spec.value_dim() || alpha.cols != spec.value_heads || beta.cols != spec.value_heads {
        return Err("CUDA Gated DeltaNet input shape 与 spec 不一致".to_owned());
    }
    let a_log_f32 = a_log.data_f32.as_ref().ok_or("CUDA Gated DeltaNet a_log 缺少 F32 device storage")?;
    let dt_bias_f32 = dt_bias.data_f32.as_ref().ok_or("CUDA Gated DeltaNet dt_bias 缺少 F32 device storage")?;
    let norm_weight_f32 = norm_weight.data_f32.as_ref().ok_or("CUDA Gated DeltaNet norm 缺少 F32 device storage")?;
    if conv_weight.data.len() != spec.conv_state_elements() || a_log_f32.len() != spec.value_heads || dt_bias_f32.len() != spec.value_heads || norm_weight_f32.len() != spec.value_head_dim {
        return Err("CUDA Gated DeltaNet F32 control weight shape 与 spec 不一致".to_owned());
    }
    let mixed = ctx.tensor_uninit(qkv.rows, spec.conv_dim())?;
    let core = ctx.tensor_uninit(qkv.rows, spec.value_dim())?;
    let output = ctx.tensor_uninit(qkv.rows, spec.value_dim())?;
    if let Some(conv_weight_f32) = &conv_weight.data_f32 {
        let conv = ctx.function("gated_delta_conv_f16")?;
        unsafe {
            ctx.stream()
                .launch_builder(&conv)
                .arg(&qkv.slice)
                .arg(conv_weight_f32)
                .arg(conv_state)
                .arg(&mixed.slice)
                .arg(&(qkv.rows as u32))
                .arg(&(spec.conv_dim() as u32))
                .arg(&(spec.conv_kernel as u32))
                .launch(grid_1d(spec.conv_dim()))
                .map_err(|error| format!("launch gated_delta_conv_f16: {error:?}"))?;
        }
    } else {
        // GGUF 路径与 Metal 一样让 conv 常驻 F16；状态和累加仍保持 F32。
        let conv = ctx.function("gated_delta_conv_weight_f16")?;
        unsafe {
            ctx.stream()
                .launch_builder(&conv)
                .arg(&qkv.slice)
                .arg(&conv_weight.data)
                .arg(conv_state)
                .arg(&mixed.slice)
                .arg(&(qkv.rows as u32))
                .arg(&(spec.conv_dim() as u32))
                .arg(&(spec.conv_kernel as u32))
                .launch(grid_1d(spec.conv_dim()))
                .map_err(|error| format!("launch gated_delta_conv_weight_f16: {error:?}"))?;
        }
    }
    let recurrent = ctx.function("gated_delta_recurrent_f16")?;
    let recurrent_cfg = LaunchConfig { grid_dim: (spec.value_heads as u32, 1, 1), block_dim: (spec.value_head_dim as u32, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        ctx.stream()
            .launch_builder(&recurrent)
            .arg(&mixed.slice)
            .arg(&alpha.slice)
            .arg(&beta.slice)
            .arg(a_log_f32)
            .arg(dt_bias_f32)
            .arg(recurrent_state)
            .arg(&core.slice)
            .arg(&(qkv.rows as u32))
            .arg(&(spec.key_heads as u32))
            .arg(&(spec.value_heads as u32))
            .arg(&(spec.key_head_dim as u32))
            .arg(&(spec.value_head_dim as u32))
            .arg(&u32::from(matches!(head_layout, crate::attention::gated_delta_net::GatedDeltaNetHeadLayout::Grouped)))
            .launch(recurrent_cfg)
            .map_err(|error| format!("launch gated_delta_recurrent_f16: {error:?}"))?;
    }
    let norm = ctx.function("gated_delta_norm_gate_f16")?;
    let norm_cfg = LaunchConfig { grid_dim: ((qkv.rows * spec.value_heads) as u32, 1, 1), block_dim: (spec.value_head_dim.next_power_of_two() as u32, 1, 1), shared_mem_bytes: (spec.value_head_dim.next_power_of_two() * 4) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&norm)
            .arg(&core.slice)
            .arg(&z.slice)
            .arg(norm_weight_f32)
            .arg(&output.slice)
            .arg(&(spec.value_heads as u32))
            .arg(&(spec.value_head_dim as u32))
            .arg(&spec.rms_eps)
            .launch(norm_cfg)
            .map_err(|error| format!("launch gated_delta_norm_gate_f16: {error:?}"))?;
    }
    Ok(output)
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use super::*;

    fn ctx() -> CudaContext {
        CudaContext::new_default().expect("CUDA 初始化")
    }

    fn htod(ctx: &CudaContext, values: &[f32]) -> CudaSliceF16 {
        ctx.stream().clone_htod::<half::f16, _>(&values.iter().map(|v| half::f16::from_f32(*v)).collect::<Vec<_>>()).expect("htod 上传")
    }

    fn check_close(name: &str, actual: &[f32], expect: &[f32]) {
        let atol = 2e-2f32;
        let rtol = 2e-2f32;
        assert_eq!(actual.len(), expect.len(), "{name}: 长度 {} != {}", actual.len(), expect.len());
        for (i, (a, e)) in actual.iter().zip(expect).enumerate() {
            let err = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(err <= tol, "{name}[{i}] 偏差 {err:.4} 超阈值 {tol:.4}(actual={a:.4} expect={e:.4})");
        }
    }

    /// 纯 host 参考实现:每 query 行只 attend 自己 batch 内的 KV(softmax(QK^T/sqrt(d))V),
    /// 验证 kernel 的 batch 间严格隔离。
    fn ref_batched_attn(q: &[f32], k: &[f32], v: &[f32], batch: usize, rows: usize, heads: usize, head_dim: usize) -> Vec<f32> {
        let cols = heads * head_dim;
        let total = batch * rows;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; total * cols];
        for row in 0..total {
            let b = row / rows;
            let hoff = |head: usize| head * head_dim;
            for head in 0..heads {
                let ho = hoff(head);
                let mut scores = vec![0.0f32; rows];
                let mut maxv = f32::NEG_INFINITY;
                for (j, _) in (0..rows).enumerate() {
                    let kr = b * rows + j;
                    let mut s = 0.0f32;
                    for d in 0..head_dim {
                        s += q[row * cols + ho + d] * k[kr * cols + ho + d];
                    }
                    s *= scale;
                    scores[j] = s;
                    if s > maxv {
                        maxv = s;
                    }
                }
                let mut den = 0.0f32;
                for s in &mut scores {
                    *s = (*s - maxv).exp();
                    den += *s;
                }
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (j, s) in scores.iter().enumerate() {
                        let kr = b * rows + j;
                        acc += s * v[kr * cols + ho + d];
                    }
                    out[row * cols + ho + d] = acc / den;
                }
            }
        }
        out
    }

    #[test]
    fn full_attention_batched_matches_oracle() {
        let ctx = ctx();
        let (batch, rows, heads, head_dim) = (2usize, 3, 2, 4);
        // 区分性数据:batch 1 的数值整体更大,若 kernel 误把 batch 0 的 query attend 到 batch 1,
        // 输出会明显偏离 host 参考实现。
        let n = batch * rows * heads * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.07 + 0.3).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();
        let expect = ref_batched_attn(&q, &k, &v, batch, rows, heads, head_dim);
        let gpu_q = ctx.tensor_from_f32(&q, batch * rows, heads * head_dim).unwrap();
        let gpu_k = htod(&ctx, &k);
        let gpu_v = htod(&ctx, &v);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let out = full_attention_batched_f16(&ctx, &gpu_q, &gpu_k, &gpu_v, batch, rows, heads, head_dim, scale).unwrap();
        check_close("full_attention_batched", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    /// batch=1 必须与现有 full_attention(走 gqa)路径等价。
    #[test]
    fn full_attention_batched_batch1_matches_oracle() {
        let ctx = ctx();
        let (batch, rows, heads, head_dim) = (1usize, 5, 2, 4);
        let n = batch * rows * heads * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.09 - 0.5).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.06 + 0.2).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.04 + 0.1).collect();
        let expect = ref_batched_attn(&q, &k, &v, batch, rows, heads, head_dim);
        let gpu_q = ctx.tensor_from_f32(&q, rows, heads * head_dim).unwrap();
        let gpu_k = htod(&ctx, &k);
        let gpu_v = htod(&ctx, &v);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let out = full_attention_batched_f16(&ctx, &gpu_q, &gpu_k, &gpu_v, batch, rows, heads, head_dim, scale).unwrap();
        check_close("full_attention_batched_b1", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    /// 纯 host 参考实现:GQA causal attention (kv_heads 可以整除 heads)。
    /// 每 query row r 只 attend KV[0, position+r],GQA 共享 K/V by `heads/kv_heads`。
    /// `query_rows` 从 `q.len() / q_cols` 推,`kv_rows` 从 `k.len() / kv_cols` 推。
    fn ref_gqa_attn(q: &[f32], k: &[f32], v: &[f32], position: usize, heads: usize, kv_heads: usize, head_dim: usize) -> Vec<f32> {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_cols = heads * head_dim;
        let kv_cols = kv_heads * head_dim;
        if q_cols == 0 || kv_cols == 0 || !q.len().is_multiple_of(q_cols) || !k.len().is_multiple_of(kv_cols) {
            return Vec::new();
        }
        let query_rows = q.len() / q_cols;
        let kv_rows = k.len() / kv_cols;
        let mut out = vec![0.0f32; query_rows * q_cols];
        let group_size = heads / kv_heads;
        for row in 0..query_rows {
            let last = (position + row).min(kv_rows - 1);
            for head in 0..heads {
                let ho = head * head_dim;
                let kv_head = head / group_size;
                let kho = kv_head * head_dim;
                let mut scores = vec![0.0f32; last + 1];
                let mut maxv = f32::NEG_INFINITY;
                for token in 0..=last {
                    let mut s = 0.0f32;
                    for d in 0..head_dim {
                        s += q[row * q_cols + ho + d] * k[token * kv_cols + kho + d];
                    }
                    s *= scale;
                    scores[token] = s;
                    if s > maxv {
                        maxv = s;
                    }
                }
                let mut den = 0.0f32;
                for s in &mut scores {
                    *s = (*s - maxv).exp();
                    den += *s;
                }
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for token in 0..=last {
                        acc += scores[token] * v[token * kv_cols + kho + d];
                    }
                    out[row * q_cols + ho + d] = acc / den;
                }
            }
        }
        out
    }

    fn make_gqa_spec(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> crate::attention::gqa::GqaSpec {
        use crate::attention::gqa::{CausalWindow, GqaSpec};
        GqaSpec { num_heads, num_kv_heads, head_dim, rope_dim: 0, rope_theta: 10000.0, use_qk_norm: false, window: CausalWindow::Full, score_scale: 1.0f32 / (head_dim as f32).sqrt(), output_gate: false }
    }

    #[test]
    fn gqa_attention_flash_matches_oracle_gqa_tail() {
        let ctx = ctx();
        // 行数 70(尾部行不是 8 的倍数)、GQA 2:1、head_dim 128(Qwen3 形状)、kv 跨 tile 边界。
        let (q_rows, kv_rows, pos, heads, kv_heads, head_dim) = (70usize, 96, 0, 4, 2, 128);
        let q: Vec<f32> = (0..q_rows * heads * head_dim).map(|i| ((i * 31) % 97) as f32 * 0.012 - 0.5).collect();
        let k: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 17) % 89) as f32 * 0.010 + 0.3).collect();
        let v: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 11) % 73) as f32 * 0.014 - 0.25).collect();
        let expect = ref_gqa_attn(&q, &k, &v, pos, heads, kv_heads, head_dim);
        let gpu_q = ctx.tensor_from_f32(&q, q_rows, heads * head_dim).unwrap();
        let gpu_k = htod(&ctx, &k);
        let gpu_v = htod(&ctx, &v);
        let spec = make_gqa_spec(heads, kv_heads, head_dim);
        let out = gqa_attention_f16_flash(&ctx, &gpu_q, &gpu_k, &gpu_v, kv_rows, pos, &spec).unwrap();
        check_close("gqa_flash_gqa_tail", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn gqa_attention_flash_matches_untiled_offset() {
        let ctx = ctx();
        // 与现有 non-tiled kernel 数值一致性:非零 position 偏移 + GQA 5:1(40/8 的形状比例)。
        let (q_rows, kv_rows, pos, heads, kv_heads, head_dim) = (130usize, 160, 32, 5, 1, 64);
        let q: Vec<f32> = (0..q_rows * heads * head_dim).map(|i| ((i * 37) % 101) as f32 * 0.011 - 0.4).collect();
        let k: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 13) % 83) as f32 * 0.013 + 0.2).collect();
        let v: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 7) % 71) as f32 * 0.015 - 0.3).collect();
        let gpu_q = ctx.tensor_from_f32(&q, q_rows, heads * head_dim).unwrap();
        let gpu_k = htod(&ctx, &k);
        let gpu_v = htod(&ctx, &v);
        let spec = make_gqa_spec(heads, kv_heads, head_dim);
        let expect_kernel = gqa_attention_f16(&ctx, &gpu_q, &gpu_k, &gpu_v, kv_rows, pos, &spec).unwrap();
        let out = gqa_attention_f16_flash(&ctx, &gpu_q, &gpu_k, &gpu_v, kv_rows, pos, &spec).unwrap();
        check_close("gqa_flash_vs_plain", &ctx.tensor_to_f32(&out).unwrap(), &ctx.tensor_to_f32(&expect_kernel).unwrap());
    }

    #[test]
    fn q8g64_quantize_roundtrip_error_bounded() {
        let ctx = ctx();
        // 量化往返:每元素误差 ≤ scale/2(scale = amax/127 → 相对误差 ~0.4%)。
        let (rows, cols) = (64usize, 1024);
        let input: Vec<f32> = (0..rows * cols).map(|i| ((i * 37) % 199) as f32 * 0.021 - 2.1).collect();
        let gpu_input = ctx.tensor_from_f32(&input, rows, cols).unwrap();
        let mut codes = ctx.buffer_uninit::<i8>(rows * cols).unwrap();
        let mut scales = ctx.buffer_uninit::<f32>(rows * cols / 64).unwrap();
        q8g64_quantize_rows(&ctx, &gpu_input, &mut codes, &mut scales, 0).unwrap();
        let codes_host = ctx.stream().clone_dtoh::<i8, _>(&codes).unwrap();
        let scales_host = ctx.stream().clone_dtoh::<f32, _>(&scales).unwrap();
        // 参考值用 f16(输入实际以 f16 上传),量化误差界按组内 scale/2。
        let reference: Vec<f32> = input.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
        for (index, value) in reference.iter().enumerate() {
            let group = index / 64;
            let reconstructed = codes_host[index] as f32 * scales_host[group];
            let bound = scales_host[group] / 2.0 + 1e-6;
            assert!((reconstructed - value).abs() <= bound, "[{index}] {reconstructed} vs {value}, bound={bound}");
        }
    }

    #[test]
    fn q8g64_quantize_appends_at_position_offset() {
        let ctx = ctx();
        // 两次追加(position=0 与 position=rows):第二次必须落到绝对行,不覆盖第一次。
        let (rows, cols) = (8usize, 1024);
        let first: Vec<f32> = (0..rows * cols).map(|i| ((i * 37) % 199) as f32 * 0.02 - 1.0).collect();
        let second: Vec<f32> = (0..rows * cols).map(|i| ((i * 53) % 211) as f32 * 0.03 + 0.5).collect();
        let gpu_first = ctx.tensor_from_f32(&first, rows, cols).unwrap();
        let gpu_second = ctx.tensor_from_f32(&second, rows, cols).unwrap();
        let mut codes = ctx.buffer_uninit::<i8>(2 * rows * cols).unwrap();
        let mut scales = ctx.buffer_uninit::<f32>(2 * rows * cols / 64).unwrap();
        q8g64_quantize_rows(&ctx, &gpu_first, &mut codes, &mut scales, 0).unwrap();
        q8g64_quantize_rows(&ctx, &gpu_second, &mut codes, &mut scales, rows).unwrap();
        let codes_host = ctx.stream().clone_dtoh::<i8, _>(&codes).unwrap();
        let scales_host = ctx.stream().clone_dtoh::<f32, _>(&scales).unwrap();
        for (index, value) in second.iter().enumerate() {
            let absolute = rows * cols + index;
            let group = absolute / 64;
            let reconstructed = codes_host[absolute] as f32 * scales_host[group];
            assert!((reconstructed - half::f16::from_f32(*value).to_f32()).abs() <= scales_host[group] / 2.0 + 1e-6, "[{index}] {reconstructed}");
        }
    }

    #[test]
    fn gqa_attention_q8g64_matches_f16_within_quant_error() {
        let ctx = ctx();
        // q8 KV 的 attention 与 f16 KV 的差异只来自 K/V 量化(~0.4% 相对),阈值 2e-2。
        let (q_rows, kv_rows, pos, heads, kv_heads, head_dim) = (70usize, 96, 0, 4, 2, 128);
        let q: Vec<f32> = (0..q_rows * heads * head_dim).map(|i| ((i * 31) % 97) as f32 * 0.012 - 0.5).collect();
        let k: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 17) % 89) as f32 * 0.010 + 0.3).collect();
        let v: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 11) % 73) as f32 * 0.014 - 0.25).collect();
        let spec = make_gqa_spec(heads, kv_heads, head_dim);
        let gpu_q = ctx.tensor_from_f32(&q, q_rows, heads * head_dim).unwrap();
        let gpu_k = ctx.tensor_from_f32(&k, kv_rows, kv_heads * head_dim).unwrap();
        let gpu_v = ctx.tensor_from_f32(&v, kv_rows, kv_heads * head_dim).unwrap();
        let expect = gqa_attention_f16(&ctx, &gpu_q, &gpu_k.slice, &gpu_v.slice, kv_rows, pos, &spec).unwrap();

        let mut key_codes = ctx.buffer_uninit::<i8>(kv_rows * kv_heads * head_dim).unwrap();
        let mut key_scales = ctx.buffer_uninit::<f32>(kv_rows * kv_heads * head_dim / 64).unwrap();
        let mut value_codes = ctx.buffer_uninit::<i8>(kv_rows * kv_heads * head_dim).unwrap();
        let mut value_scales = ctx.buffer_uninit::<f32>(kv_rows * kv_heads * head_dim / 64).unwrap();
        q8g64_quantize_rows(&ctx, &gpu_k, &mut key_codes, &mut key_scales, 0).unwrap();
        q8g64_quantize_rows(&ctx, &gpu_v, &mut value_codes, &mut value_scales, 0).unwrap();

        let out = gqa_attention_q8g64(&ctx, &gpu_q, &key_codes, &key_scales, &value_codes, &value_scales, kv_rows, pos, &spec).unwrap();
        check_close("gqa_q8_vs_f16", &ctx.tensor_to_f32(&out).unwrap(), &ctx.tensor_to_f32(&expect).unwrap());
        let flash = gqa_attention_q8g64_flash(&ctx, &gpu_q, &key_codes, &key_scales, &value_codes, &value_scales, kv_rows, pos, &spec).unwrap();
        check_close("gqa_q8_flash_vs_f16", &ctx.tensor_to_f32(&flash).unwrap(), &ctx.tensor_to_f32(&expect).unwrap());
    }

    #[test]
    fn gqa_attention_tiled_matches_oracle_short() {
        let ctx = ctx();
        // 短序列,kv_rows=16,query_rows=4;tile 边界和顺序的常规覆盖。
        let (q_rows, kv_rows, pos, heads, kv_heads, head_dim) = (4usize, 16, 0, 2, 2, 64);
        let q: Vec<f32> = (0..q_rows * heads * head_dim).map(|i| (i as f32) * 0.013 - 0.7).collect();
        let k: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| (i as f32) * 0.009 + 0.4).collect();
        let v: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| (i as f32) * 0.011 - 0.2).collect();
        let expect = ref_gqa_attn(&q, &k, &v, pos, heads, kv_heads, head_dim);
        let gpu_q = ctx.tensor_from_f32(&q, q_rows, heads * head_dim).unwrap();
        let gpu_k = htod(&ctx, &k);
        let gpu_v = htod(&ctx, &v);
        let spec = make_gqa_spec(heads, kv_heads, head_dim);
        let out = gqa_attention_f16_tiled(&ctx, &gpu_q, &gpu_k, &gpu_v, kv_rows, pos, &spec, 8).unwrap();
        check_close("gqa_tiled_short", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn gqa_attention_tiled_matches_oracle_long_gqa() {
        let ctx = ctx();
        // 长序列模拟生产场景:kv_rows=300、query_rows=20、heads=4、kv_heads=2 → GQA 2:1。
        let (q_rows, kv_rows, pos, heads, kv_heads, head_dim) = (20usize, 300, 0, 4, 2, 64);
        let q: Vec<f32> = (0..q_rows * heads * head_dim).map(|i| ((i * 31) % 97) as f32 * 0.012 - 0.5).collect();
        let k: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 17) % 89) as f32 * 0.010 + 0.3).collect();
        let v: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 11) % 73) as f32 * 0.014 - 0.25).collect();
        let expect = ref_gqa_attn(&q, &k, &v, pos, heads, kv_heads, head_dim);
        let gpu_q = ctx.tensor_from_f32(&q, q_rows, heads * head_dim).unwrap();
        let gpu_k = htod(&ctx, &k);
        let gpu_v = htod(&ctx, &v);
        let spec = make_gqa_spec(heads, kv_heads, head_dim);
        let out = gqa_attention_f16_tiled(&ctx, &gpu_q, &gpu_k, &gpu_v, kv_rows, pos, &spec, 32).unwrap();
        check_close("gqa_tiled_long", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    /// 必须与现有 gqa_attention_f16(non-tiled) 在统计意义上一致:相同的数值结果。
    #[test]
    fn gqa_attention_tiled_matches_untiled() {
        let ctx = ctx();
        let (q_rows, kv_rows, pos, heads, kv_heads, head_dim) = (8usize, 64, 16, 4, 2, 64);
        let q: Vec<f32> = (0..q_rows * heads * head_dim).map(|i| ((i * 23) % 71) as f32 * 0.015 - 0.6).collect();
        let k: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 19) % 83) as f32 * 0.011 + 0.2).collect();
        let v: Vec<f32> = (0..kv_rows * kv_heads * head_dim).map(|i| ((i * 13) % 67) as f32 * 0.013 - 0.15).collect();
        let gpu_q = ctx.tensor_from_f32(&q, q_rows, heads * head_dim).unwrap();
        let gpu_k = htod(&ctx, &k);
        let gpu_v = htod(&ctx, &v);
        let spec = make_gqa_spec(heads, kv_heads, head_dim);
        let out_tiled = gqa_attention_f16_tiled(&ctx, &gpu_q, &gpu_k, &gpu_v, kv_rows, pos, &spec, 32).unwrap();
        let out_untiled = gqa_attention_f16(&ctx, &gpu_q, &gpu_k, &gpu_v, kv_rows, pos, &spec).unwrap();
        check_close("gqa_tiled_vs_untiled", &ctx.tensor_to_f32(&out_tiled).unwrap(), &ctx.tensor_to_f32(&out_untiled).unwrap());
    }

    #[test]
    fn gemma_rmsnorm_residual_matches_reference() {
        let ctx = ctx();
        let head_count = 4;
        let head_dim = 8;
        let rows = 6;
        let eps = 1e-6f32;
        let cols = head_count * head_dim;
        let hidden: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.013 - 0.4).collect();
        let residual: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.011 + 0.1).collect();
        let weight: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.05 - 0.02).collect();

        let hidden_gpu = ctx.tensor_from_f32(&hidden, rows, cols).unwrap();
        let residual_gpu = ctx.tensor_from_f32(&residual, rows, cols).unwrap();
        let weight_gpu = ctx.stream().clone_htod::<half::f16, _>(&weight.iter().map(|v| half::f16::from_f32(*v)).collect::<Vec<_>>()).unwrap();

        let fused = gemma_rmsnorm_residual_heads_f16(&ctx, &hidden_gpu, &residual_gpu, &weight_gpu, head_count, head_dim, eps).unwrap();
        let fused_out = ctx.tensor_to_f32(&fused).unwrap();

        // CPU 参考:先 add 再 GemmaRMSNorm(output = (input+residual) * rsqrt(mean+eps) * (1+w))。
        // Gemma head norm:每个 head 独立计算 mean,weight 按 head 内列索引。
        let mut expect = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for head in 0..head_count {
                let begin = head * head_dim;
                let mut sum_sq = 0.0f32;
                for c in 0..head_dim {
                    let idx = row * cols + begin + c;
                    let v = hidden[idx] + residual[idx];
                    sum_sq += v * v;
                }
                let scale = (sum_sq / head_dim as f32 + eps).sqrt().recip();
                for c in 0..head_dim {
                    let idx = row * cols + begin + c;
                    let v = hidden[idx] + residual[idx];
                    expect[idx] = v * scale * (1.0 + weight[c]);
                }
            }
        }
        check_close("gemma_rmsnorm_residual", &fused_out, &expect);
    }
}
