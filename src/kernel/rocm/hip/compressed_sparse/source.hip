#include <rocwmma/rocwmma.hpp>

__device__ __forceinline__ float csa_input(
    const float* pending,
    const float* input,
    unsigned int pending_rows,
    unsigned int channels,
    unsigned int row,
    unsigned int column)
{
    return row < pending_rows
        ? pending[(unsigned long long)row * channels + column]
        : input[(unsigned long long)(row - pending_rows) * channels + column];
}

__device__ __forceinline__ unsigned int csa_ordered_score(float value)
{
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}

// 与 ROCm paged KV/DSA 共用 Q8G 编码：i8 codes + 每组一个 BF16 scale。
__device__ __forceinline__ float csa_q8_element(
    const signed char* values,
    const unsigned short* scales,
    unsigned long long index,
    unsigned int group_size)
{
    const float scale = __uint_as_float((unsigned int)scales[index / group_size] << 16);
    return (float)values[index] * scale;
}

typedef unsigned short __attribute__((ext_vector_type(2))) csa_bf16x2;

extern "C" __global__ void csa_gated_compress_f32(
    const float* pending_key,
    const float* pending_gate,
    const float* key,
    const float* gate,
    const float* position_bias,
    const float* norm,
    const float* overlap_key,
    const float* overlap_gate,
    const float* cos_table,
    const float* sin_table,
    float* output,
    unsigned int pending_rows,
    unsigned int windows,
    unsigned int ratio,
    unsigned int width,
    unsigned int channels,
    unsigned int overlap,
    unsigned int entry_start,
    unsigned int rotary_dim,
    float eps)
{
    const unsigned int window = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (window >= windows) return;
    __shared__ float reduced[1024];
    __shared__ float normalized[1024];
    float pooled = 0.0f;
    if (lane < width) {
        const unsigned int slots = overlap != 0 ? ratio * 2 : ratio;
        float maximum = -3.402823466e+38F;
        for (unsigned int slot = 0; slot < slots; ++slot) {
            float logit;
            if (overlap != 0 && slot < ratio) {
                if (window == 0) {
                    logit = entry_start == 0 ? -3.402823466e+38F : overlap_gate[(unsigned long long)slot * width + lane];
                } else {
                    const unsigned int row = (window - 1) * ratio + slot;
                    logit = csa_input(pending_gate, gate, pending_rows, channels, row, lane)
                        + position_bias[(unsigned long long)slot * channels + lane];
                }
            } else {
                const unsigned int current_slot = overlap != 0 ? slot - ratio : slot;
                const unsigned int row = window * ratio + current_slot;
                const unsigned int column = overlap != 0 ? width + lane : lane;
                logit = csa_input(pending_gate, gate, pending_rows, channels, row, column)
                    + position_bias[(unsigned long long)current_slot * channels + column];
            }
            maximum = fmaxf(maximum, logit);
        }
        float denominator = 0.0f;
        for (unsigned int slot = 0; slot < slots; ++slot) {
            float value;
            float logit;
            if (overlap != 0 && slot < ratio) {
                if (window == 0) {
                    value = entry_start == 0 ? 0.0F : overlap_key[(unsigned long long)slot * width + lane];
                    logit = entry_start == 0 ? -3.402823466e+38F : overlap_gate[(unsigned long long)slot * width + lane];
                } else {
                    const unsigned int row = (window - 1) * ratio + slot;
                    value = csa_input(pending_key, key, pending_rows, channels, row, lane);
                    logit = csa_input(pending_gate, gate, pending_rows, channels, row, lane)
                        + position_bias[(unsigned long long)slot * channels + lane];
                }
            } else {
                const unsigned int current_slot = overlap != 0 ? slot - ratio : slot;
                const unsigned int row = window * ratio + current_slot;
                const unsigned int column = overlap != 0 ? width + lane : lane;
                value = csa_input(pending_key, key, pending_rows, channels, row, column);
                logit = csa_input(pending_gate, gate, pending_rows, channels, row, column)
                    + position_bias[(unsigned long long)current_slot * channels + column];
            }
            const float weight = expf(logit - maximum);
            denominator += weight;
            pooled += value * weight;
        }
        pooled /= denominator;
        reduced[lane] = pooled * pooled;
    } else {
        reduced[lane] = 0.0f;
    }
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) reduced[lane] += reduced[lane + stride];
        __syncthreads();
    }
    if (lane < width) normalized[lane] = pooled * rsqrtf(reduced[0] / (float)width + eps) * norm[lane];
    __syncthreads();
    if (lane < width) {
        float result = normalized[lane];
        const unsigned int rotary_start = width - rotary_dim;
        if (lane >= rotary_start) {
            const unsigned int relative = lane - rotary_start;
            const unsigned int pair = relative >> 1;
            const unsigned int partner = rotary_start + (relative ^ 1u);
            const unsigned long long table = (unsigned long long)(entry_start + window) * ratio * (rotary_dim >> 1) + pair;
            const float cosine = cos_table[table];
            const float sine = sin_table[table];
            result = (relative & 1u) == 0
                ? normalized[lane] * cosine - normalized[partner] * sine
                : normalized[lane] * cosine + normalized[partner] * sine;
        }
        output[(unsigned long long)window * width + lane] = result;
    }
}

extern "C" __global__ void csa_store_overlap_f32(
    const float* pending_key,
    const float* pending_gate,
    const float* key,
    const float* gate,
    const float* position_bias,
    float* overlap_key,
    float* overlap_gate,
    unsigned int pending_rows,
    unsigned int windows,
    unsigned int ratio,
    unsigned int width,
    unsigned int channels,
    unsigned int count)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned int slot = index / width;
    const unsigned int column = index - slot * width;
    const unsigned int row = (windows - 1) * ratio + slot;
    overlap_key[index] = csa_input(pending_key, key, pending_rows, channels, row, column);
    overlap_gate[index] = csa_input(pending_gate, gate, pending_rows, channels, row, column)
        + position_bias[(unsigned long long)slot * channels + column];
}

extern "C" __global__ void csa_store_pending_f32(
    const float* old_pending_key,
    const float* old_pending_gate,
    const float* key,
    const float* gate,
    float* pending_key,
    float* pending_gate,
    unsigned int old_pending_rows,
    unsigned int input_rows,
    unsigned int ratio,
    unsigned int channels)
{
    const unsigned int remaining = (old_pending_rows + input_rows) % ratio;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= remaining * channels) return;
    const unsigned int source_row = old_pending_rows + input_rows - remaining + index / channels;
    const unsigned int column = index % channels;
    pending_key[index] = csa_input(old_pending_key, key, old_pending_rows, channels, source_row, column);
    pending_gate[index] = csa_input(old_pending_gate, gate, old_pending_rows, channels, source_row, column);
}

extern "C" __global__ void csa_index_scores_f32(
    const float* query,
    const float* keys,
    const float* head_weights,
    const unsigned int* visible_counts,
    unsigned int* score_keys,
    unsigned int query_rows,
    unsigned int compressed_rows,
    unsigned int head_count,
    unsigned int head_dim)
{
    const unsigned int key_row = blockIdx.x;
    const unsigned int query_row = blockIdx.y;
    const unsigned int thread = threadIdx.x;
    if (query_row >= query_rows || key_row >= visible_counts[query_row]) return;
    const unsigned int wave = thread / warpSize;
    const unsigned int lane = thread % warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    __shared__ float partial[16];
    float subtotal = 0.0f;
    for (unsigned int head = wave; head < head_count; head += waves) {
        float dot = 0.0f;
        const unsigned long long query_base = ((unsigned long long)query_row * head_count + head) * head_dim;
        const unsigned long long key_base = (unsigned long long)key_row * head_dim;
        for (unsigned int column = lane; column < head_dim; column += warpSize) {
            dot += query[query_base + column] * keys[key_base + column];
        }
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            dot += __shfl_down(dot, offset, warpSize);
        }
        if (lane == 0) subtotal += head_weights[(unsigned long long)query_row * head_count + head]
            * fmaxf(0.0f, dot * rsqrtf((float)head_dim));
    }
    if (lane == 0) partial[wave] = subtotal;
    __syncthreads();
    if (thread == 0) {
        float total = 0.0f;
        for (unsigned int index = 0; index < waves; ++index) total += partial[index];
        score_keys[(unsigned long long)query_row * compressed_rows + key_row]
            = csa_ordered_score(total * rsqrtf((float)head_count));
    }
}

// 学习式 indexer 的矩阵路径：四个 query 并行共用 32 行压缩 K tile。query
// 预先转为 resident BF16 后直接读取，LDS 只保存 K 与分头 score。
#define CSA_INDEX_QUERY_TILE 4
#define CSA_INDEX_KEY_TILE 32
#define CSA_INDEX_HEAD_TILE 16
#define CSA_INDEX_MAX_HEAD_TILES 4
#define CSA_INDEX_MAX_DIM_TILES 8
#define CSA_INDEX_MAX_HEADS (CSA_INDEX_HEAD_TILE * CSA_INDEX_MAX_HEAD_TILES)
#define CSA_INDEX_MAX_DIM (16 * CSA_INDEX_MAX_DIM_TILES)

extern "C" __global__ void csa_index_scores_wmma_f32(
    const unsigned short* query,
    const float* keys,
    const float* head_weights,
    const unsigned int* visible_counts,
    unsigned int* score_keys,
    unsigned int query_rows,
    unsigned int compressed_rows,
    unsigned int head_count,
    unsigned int head_dim)
{
    const unsigned int key_start = blockIdx.x * CSA_INDEX_KEY_TILE;
    const unsigned int query_start = blockIdx.y * CSA_INDEX_QUERY_TILE;
    const unsigned int head_tiles = head_count / CSA_INDEX_HEAD_TILE;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int query_local = wave / head_tiles;
    const unsigned int head_tile = wave - query_local * head_tiles;
    const bool wave_active = query_local < CSA_INDEX_QUERY_TILE && query_start + query_local < query_rows;
    using Bf16 = rocwmma::bfloat16_t;
    __shared__ Bf16 key_tile[CSA_INDEX_KEY_TILE * CSA_INDEX_MAX_DIM];
    __shared__ float score_tile[CSA_INDEX_QUERY_TILE * CSA_INDEX_MAX_HEADS * CSA_INDEX_KEY_TILE];

    const unsigned int key_values = CSA_INDEX_KEY_TILE * head_dim;
    for (unsigned int index = threadIdx.x; index < key_values; index += blockDim.x) {
        const unsigned int local_key = index / head_dim;
        const unsigned int column = index - local_key * head_dim;
        const unsigned int key_row = key_start + local_key;
        key_tile[index] = key_row < compressed_rows ? static_cast<Bf16>(keys[(unsigned long long)key_row * head_dim + column]) : static_cast<Bf16>(0.0f);
    }
    __syncthreads();

    if (wave_active) {
        using QueryFragment = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, Bf16, rocwmma::row_major>;
        using KeyFragment = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, Bf16, rocwmma::col_major>;
        using ScoreFragment = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
        QueryFragment query_fragment;
        KeyFragment key_fragment;
        ScoreFragment score_fragments[CSA_INDEX_KEY_TILE / 16];
        #pragma unroll
        for (unsigned int key_tile_index = 0; key_tile_index < CSA_INDEX_KEY_TILE / 16; ++key_tile_index) {
            rocwmma::fill_fragment(score_fragments[key_tile_index], 0.0f);
        }
        const unsigned int head_start = head_tile * CSA_INDEX_HEAD_TILE;
        const Bf16* query_base = reinterpret_cast<const Bf16*>(query) + ((unsigned long long)(query_start + query_local) * head_count + head_start) * head_dim;
        #pragma unroll
        for (unsigned int dim_start = 0; dim_start < CSA_INDEX_MAX_DIM; dim_start += 16) {
            if (dim_start < head_dim) {
                rocwmma::load_matrix_sync(query_fragment, query_base + dim_start, head_dim);
                #pragma unroll
                for (unsigned int key_local = 0; key_local < CSA_INDEX_KEY_TILE; key_local += 16) {
                    rocwmma::load_matrix_sync(key_fragment, key_tile + key_local * head_dim + dim_start, head_dim);
                    rocwmma::mma_sync(score_fragments[key_local / 16], query_fragment, key_fragment, score_fragments[key_local / 16]);
                }
            }
        }
        #pragma unroll
        for (unsigned int key_local = 0; key_local < CSA_INDEX_KEY_TILE; key_local += 16) {
            float* score_base = score_tile + (query_local * head_count + head_start) * CSA_INDEX_KEY_TILE + key_local;
            rocwmma::store_matrix_sync(score_base, score_fragments[key_local / 16], CSA_INDEX_KEY_TILE, rocwmma::mem_row_major);
        }
    }
    __syncthreads();

    for (unsigned int result = threadIdx.x; result < CSA_INDEX_QUERY_TILE * CSA_INDEX_KEY_TILE; result += blockDim.x) {
        const unsigned int query_local_result = result / CSA_INDEX_KEY_TILE;
        const unsigned int key_local = result - query_local_result * CSA_INDEX_KEY_TILE;
        const unsigned int query_row = query_start + query_local_result;
        const unsigned int key_row = key_start + key_local;
        if (query_row >= query_rows || key_row >= compressed_rows || key_row >= visible_counts[query_row]) continue;
        float total = 0.0f;
        const float* tile = score_tile + query_local_result * head_count * CSA_INDEX_KEY_TILE + key_local;
        for (unsigned int head = 0; head < head_count; ++head) {
            total += head_weights[(unsigned long long)query_row * head_count + head]
                * fmaxf(0.0f, tile[head * CSA_INDEX_KEY_TILE] * rsqrtf((float)head_dim));
        }
        score_keys[(unsigned long long)query_row * compressed_rows + key_row]
            = csa_ordered_score(total * rsqrtf((float)head_count));
    }
}

// 一个 wave 独立处理一个 query/head。head_dim 最大 1024，因此每个 lane
// 最多持有 32 个通道；固定上界让编译器把 query 与 online-softmax 累加器留在寄存器。
#define CSA_WAVE_CHANNELS 32
#define CSA_TILED_CHANNELS 16
#define CSA_TILED_HEAD_DIM (CSA_TILED_CHANNELS * 32)

__device__ __forceinline__ void csa_attention_wave_step(
    const float* query_values,
    float* value_sums,
    const float* key,
    const float* value,
    unsigned int lane,
    unsigned int head_dim,
    float score_scale,
    float* maximum,
    float* denominator)
{
    float dot = 0.0f;
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_WAVE_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        if (column < head_dim) dot += query_values[slot] * key[column];
    }
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        dot += __shfl_down(dot, offset, warpSize);
    }
    dot = __shfl(dot, 0, warpSize);
    const float score = dot * score_scale;
    const float next_maximum = fmaxf(*maximum, score);
    const float previous_scale = expf(*maximum - next_maximum);
    const float value_scale = expf(score - next_maximum);
    *denominator = *denominator * previous_scale + value_scale;
    *maximum = next_maximum;
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_WAVE_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        if (column < head_dim) value_sums[slot] = value_sums[slot] * previous_scale + value[column] * value_scale;
    }
}

__device__ __forceinline__ void csa_attention_wave_step_q8(
    const float* query_values,
    float* value_sums,
    const signed char* key,
    const unsigned short* key_scales,
    const signed char* value,
    const unsigned short* value_scales,
    unsigned long long base,
    unsigned int lane,
    unsigned int head_dim,
    unsigned int group_size,
    float score_scale,
    float* maximum,
    float* denominator)
{
    float dot = 0.0f;
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_WAVE_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        if (column < head_dim) dot += query_values[slot] * csa_q8_element(key, key_scales, base + column, group_size);
    }
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) dot += __shfl_down(dot, offset, warpSize);
    dot = __shfl(dot, 0, warpSize);
    const float score = dot * score_scale;
    const float next_maximum = fmaxf(*maximum, score);
    const float previous_scale = expf(*maximum - next_maximum);
    const float value_scale = expf(score - next_maximum);
    *denominator = *denominator * previous_scale + value_scale;
    *maximum = next_maximum;
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_WAVE_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        if (column < head_dim) value_sums[slot] = value_sums[slot] * previous_scale + csa_q8_element(value, value_scales, base + column, group_size) * value_scale;
    }
}

// gfx1100 wave32 tiled 路径的固定 lane 映射：每 lane 持有相隔 32 列的
// 16 个通道，成对送入 BF16 dot2，QK 保持 F32 累加；PV 仍使用 F32，
// 避免同时改变两段数值。head_dim 小于 512 时越界半对补零。
__device__ __forceinline__ void csa_attention_wave_step_tiled(
    const unsigned short* query_values,
    float* value_sums,
    const unsigned short* key,
    const float* value,
    unsigned int lane,
    unsigned int head_dim,
    float score_scale,
    float* maximum,
    float* denominator)
{
    float dot = 0.0f;
    #pragma unroll
    for (unsigned int pair = 0; pair < CSA_TILED_CHANNELS / 2; ++pair) {
        const unsigned int first_slot = pair * 2u;
        const unsigned int first_column = lane + first_slot * warpSize;
        const unsigned int second_column = first_column + warpSize;
        const csa_bf16x2 query_pair = {
            first_column < head_dim ? query_values[first_slot] : (unsigned short)0,
            second_column < head_dim ? query_values[first_slot + 1u] : (unsigned short)0
        };
        const csa_bf16x2 key_pair = {
            first_column < head_dim ? key[first_column] : (unsigned short)0,
            second_column < head_dim ? key[second_column] : (unsigned short)0
        };
        dot = __builtin_amdgcn_fdot2_f32_bf16(query_pair, key_pair, dot, false);
    }
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) dot += __shfl_down(dot, offset, warpSize);
    dot = __shfl(dot, 0, warpSize);
    const float score = dot * score_scale;
    const float next_maximum = fmaxf(*maximum, score);
    const float previous_scale = expf(*maximum - next_maximum);
    const float value_scale = expf(score - next_maximum);
    *denominator = *denominator * previous_scale + value_scale;
    *maximum = next_maximum;
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_TILED_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        if (column < head_dim) value_sums[slot] = value_sums[slot] * previous_scale + value[column] * value_scale;
    }
}

extern "C" __global__ void csa_attention_wave_q8(
    const float* query,
    const signed char* compressed_key,
    const unsigned short* compressed_key_scales,
    const signed char* compressed_value,
    const unsigned short* compressed_value_scales,
    const unsigned int* visible_compressed,
    const unsigned int* selection,
    const signed char* recent_key,
    const unsigned short* recent_key_scales,
    const signed char* recent_value,
    const unsigned short* recent_value_scales,
    const signed char* batch_key,
    const unsigned short* batch_key_scales,
    const signed char* batch_value,
    const unsigned short* batch_value_scales,
    const float* sink,
    float* output,
    unsigned int query_rows,
    unsigned int selection_top_k,
    unsigned int recent_start,
    unsigned int recent_len,
    unsigned int recent_first_position,
    unsigned int position_start,
    unsigned int causal_batch,
    unsigned int batch_rows,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int head_dim,
    unsigned int window_size,
    unsigned int q8_group_size,
    unsigned int has_sink)
{
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int group = blockIdx.x * waves + wave;
    if (group >= query_rows * head_count) return;
    const unsigned int head = group % head_count;
    const unsigned int query_row = group / head_count;
    const unsigned int kv_head = head * kv_head_count / head_count;
    const unsigned int query_position = position_start + query_row;
    const unsigned long long query_base = ((unsigned long long)query_row * head_count + head) * head_dim;
    float query_values[CSA_WAVE_CHANNELS];
    float value_sums[CSA_WAVE_CHANNELS];
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_WAVE_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        query_values[slot] = column < head_dim ? query[query_base + column] : 0.0f;
        value_sums[slot] = 0.0f;
    }
    float maximum = has_sink != 0 ? sink[head] : -3.402823466e+38F;
    float denominator = has_sink != 0 ? 1.0f : 0.0f;
    const float score_scale = rsqrtf((float)head_dim);

    const unsigned int visible = visible_compressed[query_row];
    const unsigned int compressed_count = selection_top_k == 0 ? visible : min(selection_top_k, visible);
    for (unsigned int slot = 0; slot < compressed_count; ++slot) {
        const unsigned int row = selection_top_k == 0 ? slot : selection[(unsigned long long)query_row * selection_top_k + slot];
        const unsigned long long kv_base = ((unsigned long long)row * kv_head_count + kv_head) * head_dim;
        csa_attention_wave_step_q8(query_values, value_sums, compressed_key, compressed_key_scales, compressed_value, compressed_value_scales, kv_base, lane, head_dim, q8_group_size, score_scale, &maximum, &denominator);
    }

    for (unsigned int logical = 0; logical < recent_len; ++logical) {
        const unsigned int key_position = recent_first_position + logical;
        if (key_position > query_position || query_position - key_position >= window_size) continue;
        const unsigned int physical = (recent_start + logical) % window_size;
        const unsigned long long kv_base = ((unsigned long long)physical * kv_head_count + kv_head) * head_dim;
        csa_attention_wave_step_q8(query_values, value_sums, recent_key, recent_key_scales, recent_value, recent_value_scales, kv_base, lane, head_dim, q8_group_size, score_scale, &maximum, &denominator);
    }

    const unsigned int batch_begin = query_row + 1 > window_size ? query_row + 1 - window_size : 0;
    const unsigned int batch_end = causal_batch != 0 ? min(query_row + 1, batch_rows) : batch_rows;
    for (unsigned int row = batch_begin; row < batch_end; ++row) {
        const unsigned long long kv_base = ((unsigned long long)row * kv_head_count + kv_head) * head_dim;
        csa_attention_wave_step_q8(query_values, value_sums, batch_key, batch_key_scales, batch_value, batch_value_scales, kv_base, lane, head_dim, q8_group_size, score_scale, &maximum, &denominator);
    }
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_WAVE_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        if (column < head_dim) output[query_base + column] = denominator == 0.0f ? 0.0f : value_sums[slot] / denominator;
    }
}

// GQA group 至少包含 8 个 heads 时，同一 block 的 wave 共享 K/V tile。
// host 在 group 可整除 16 时使用 16 waves，否则保持 8 waves，避免跨 KV head。
#define CSA_TILE_ROWS 4

// 四行作为一个 online-softmax block：QK 仍逐行计算，PV 只缩放一次旧累加器。
__device__ __forceinline__ void csa_attention_wave_tile_tiled(
    const unsigned short* query_values,
    float* value_sums,
    const unsigned short* key_tile,
    const float* value_tile,
    unsigned int tile_rows,
    unsigned int lane,
    unsigned int head_dim,
    float score_scale,
    float* maximum,
    float* denominator)
{
    float scores[CSA_TILE_ROWS];
    #pragma unroll
    for (unsigned int tile_row = 0; tile_row < CSA_TILE_ROWS; ++tile_row) {
        float dot = 0.0f;
        if (tile_row < tile_rows) {
            #pragma unroll
            for (unsigned int pair = 0; pair < CSA_TILED_CHANNELS / 2; ++pair) {
                const unsigned int first_slot = pair * 2u;
                const unsigned int first_column = lane + first_slot * warpSize;
                const unsigned int second_column = first_column + warpSize;
                const csa_bf16x2 query_pair = { query_values[first_slot], query_values[first_slot + 1u] };
                const csa_bf16x2 key_pair = {
                    key_tile[tile_row * head_dim + first_column],
                    key_tile[tile_row * head_dim + second_column]
                };
                dot = __builtin_amdgcn_fdot2_f32_bf16(query_pair, key_pair, dot, false);
            }
            for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) dot += __shfl_down(dot, offset, warpSize);
            dot = __shfl(dot, 0, warpSize);
            dot *= score_scale;
        } else {
            dot = -3.402823466e+38F;
        }
        scores[tile_row] = dot;
    }

    float next_maximum = *maximum;
    #pragma unroll
    for (unsigned int tile_row = 0; tile_row < CSA_TILE_ROWS; ++tile_row) next_maximum = fmaxf(next_maximum, scores[tile_row]);
    const float previous_scale = expf(*maximum - next_maximum);
    float weights[CSA_TILE_ROWS];
    float tile_denominator = 0.0f;
    #pragma unroll
    for (unsigned int tile_row = 0; tile_row < CSA_TILE_ROWS; ++tile_row) {
        const float weight = tile_row < tile_rows ? expf(scores[tile_row] - next_maximum) : 0.0f;
        weights[tile_row] = weight;
        tile_denominator += weight;
    }

    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_TILED_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        float tile_value = 0.0f;
        #pragma unroll
        for (unsigned int tile_row = 0; tile_row < CSA_TILE_ROWS; ++tile_row) {
            if (tile_row < tile_rows) tile_value = fmaf(value_tile[tile_row * head_dim + column], weights[tile_row], tile_value);
        }
        value_sums[slot] = fmaf(value_sums[slot], previous_scale, tile_value);
    }
    *maximum = next_maximum;
    *denominator = fmaf(*denominator, previous_scale, tile_denominator);
}

#define CSA_ATTN_HEAD_TILE 16
#define CSA_ATTN_KEY_TILE 16
#define CSA_ATTN_MAX_DIM_TILES 32
#define CSA_ATTN_OUTPUT_TILES_PER_WAVE 4

// 从 compressed / recent / batch 三段逻辑序列读取一个 K/V 元素。
__device__ __forceinline__ void csa_attention_kv_element(
    const signed char* compressed_key,
    const unsigned short* compressed_key_scales,
    const signed char* compressed_value,
    const unsigned short* compressed_value_scales,
    const unsigned int* selection,
    const signed char* recent_key,
    const unsigned short* recent_key_scales,
    const signed char* recent_value,
    const unsigned short* recent_value_scales,
    const signed char* batch_key,
    const unsigned short* batch_key_scales,
    const signed char* batch_value,
    const unsigned short* batch_value_scales,
    unsigned int selection_top_k,
    unsigned int query_row,
    unsigned int compressed_count,
    unsigned int recent_begin,
    unsigned int recent_count,
    unsigned int batch_begin,
    unsigned int recent_start,
    unsigned int window_size,
    unsigned int kv_head_count,
    unsigned int kv_head,
    unsigned int head_dim,
    unsigned int q8_group_size,
    unsigned int item,
    unsigned int column,
    unsigned int load_value,
    float* key_result,
    float* value_result)
{
    const signed char* key_source;
    const unsigned short* key_scale_source;
    const signed char* value_source;
    const unsigned short* value_scale_source;
    unsigned int row;
    if (item < compressed_count) {
        row = selection_top_k == 0 ? item : selection[(unsigned long long)query_row * selection_top_k + item];
        key_source = compressed_key;
        key_scale_source = compressed_key_scales;
        value_source = compressed_value;
        value_scale_source = compressed_value_scales;
    } else if (item < compressed_count + recent_count) {
        const unsigned int logical = recent_begin + item - compressed_count;
        row = (recent_start + logical) % window_size;
        key_source = recent_key;
        key_scale_source = recent_key_scales;
        value_source = recent_value;
        value_scale_source = recent_value_scales;
    } else {
        row = batch_begin + item - compressed_count - recent_count;
        key_source = batch_key;
        key_scale_source = batch_key_scales;
        value_source = batch_value;
        value_scale_source = batch_value_scales;
    }
    const unsigned long long base = ((unsigned long long)row * kv_head_count + kv_head) * head_dim;
    const float key_value = csa_q8_element(key_source, key_scale_source, base + column, q8_group_size);
    *key_result = key_value;
    if (load_value != 0) {
        *value_result = key_source == value_source && key_scale_source == value_scale_source ? key_value : csa_q8_element(value_source, value_scale_source, base + column, q8_group_size);
    }
}

// SGLang sparse prefill 的 QK/PV tile 结构：一个 block 处理一个 query 的
// 16 个 GQA heads，selected K/V 分 16-row tile，两遍 softmax 避免保存全量 scores。
extern "C" __global__ void csa_attention_wmma_q8(
    const float* query,
    const signed char* compressed_key,
    const unsigned short* compressed_key_scales,
    const signed char* compressed_value,
    const unsigned short* compressed_value_scales,
    const unsigned int* visible_compressed,
    const unsigned int* selection,
    const signed char* recent_key,
    const unsigned short* recent_key_scales,
    const signed char* recent_value,
    const unsigned short* recent_value_scales,
    const signed char* batch_key,
    const unsigned short* batch_key_scales,
    const signed char* batch_value,
    const unsigned short* batch_value_scales,
    const float* sink,
    float* output,
    unsigned int query_rows,
    unsigned int selection_top_k,
    unsigned int recent_start,
    unsigned int recent_len,
    unsigned int recent_first_position,
    unsigned int position_start,
    unsigned int causal_batch,
    unsigned int batch_rows,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int head_dim,
    unsigned int window_size,
    unsigned int q8_group_size,
    unsigned int has_sink)
{
    const unsigned int head_groups = head_count / CSA_ATTN_HEAD_TILE;
    const unsigned int query_row = blockIdx.x / head_groups;
    const unsigned int head_start = (blockIdx.x % head_groups) * CSA_ATTN_HEAD_TILE;
    if (query_row >= query_rows) return;
    const unsigned int kv_head = head_start * kv_head_count / head_count;
    const unsigned int query_position = position_start + query_row;
    const unsigned int visible = visible_compressed[query_row];
    const unsigned int compressed_count = selection_top_k == 0 ? visible : min(selection_top_k, visible);
    unsigned int recent_begin = 0;
    unsigned int recent_end = 0;
    if (query_position >= recent_first_position) {
        recent_end = min(recent_len, query_position - recent_first_position + 1);
        const unsigned int first_visible_position = query_position + 1 > window_size ? query_position + 1 - window_size : 0;
        if (first_visible_position > recent_first_position) recent_begin = min(recent_end, first_visible_position - recent_first_position);
    }
    const unsigned int recent_count = recent_end - recent_begin;
    const unsigned int batch_begin = query_row + 1 > window_size ? query_row + 1 - window_size : 0;
    const unsigned int batch_end = causal_batch != 0 ? min(query_row + 1, batch_rows) : batch_rows;
    const unsigned int batch_count = batch_end - batch_begin;
    const unsigned int total_keys = compressed_count + recent_count + batch_count;
    using Bf16 = rocwmma::bfloat16_t;
    using QueryFragment = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, Bf16, rocwmma::row_major>;
    using KeyFragment = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, Bf16, rocwmma::col_major>;
    using ProbabilityFragment = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, Bf16, rocwmma::row_major>;
    using ValueFragment = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, Bf16, rocwmma::row_major>;
    using AccumulatorFragment = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    __shared__ Bf16 query_tile[CSA_ATTN_HEAD_TILE * CSA_ATTN_MAX_DIM_TILES * 16];
    __shared__ Bf16 key_tile[CSA_ATTN_KEY_TILE * CSA_ATTN_MAX_DIM_TILES * 16];
    __shared__ Bf16 value_tile[CSA_ATTN_KEY_TILE * CSA_ATTN_MAX_DIM_TILES * 16];
    __shared__ Bf16 probability_tile[CSA_ATTN_HEAD_TILE * CSA_ATTN_KEY_TILE];
    __shared__ float score_tile[CSA_ATTN_HEAD_TILE * CSA_ATTN_KEY_TILE];
    __shared__ float running_maximum[CSA_ATTN_HEAD_TILE];
    __shared__ float running_denominator[CSA_ATTN_HEAD_TILE];

    const unsigned long long query_base = ((unsigned long long)query_row * head_count + head_start) * head_dim;
    for (unsigned int index = threadIdx.x; index < CSA_ATTN_HEAD_TILE * head_dim; index += blockDim.x) {
        query_tile[index] = static_cast<Bf16>(query[query_base + index]);
    }
    if (threadIdx.x < CSA_ATTN_HEAD_TILE) {
        running_maximum[threadIdx.x] = has_sink != 0 ? sink[head_start + threadIdx.x] : -3.402823466e+38F;
        running_denominator[threadIdx.x] = has_sink != 0 ? 1.0f : 0.0f;
    }
    __syncthreads();

    // 第一遍只累计每个 head 的最终 softmax maximum 与 denominator。
    for (unsigned int key_start = 0; key_start < total_keys; key_start += CSA_ATTN_KEY_TILE) {
        const unsigned int tile_rows = min((unsigned int)CSA_ATTN_KEY_TILE, total_keys - key_start);
        for (unsigned int index = threadIdx.x; index < CSA_ATTN_KEY_TILE * head_dim; index += blockDim.x) {
            const unsigned int key_local = index / head_dim;
            const unsigned int column = index - key_local * head_dim;
            float key_value = 0.0f;
            float ignored_value = 0.0f;
            if (key_local < tile_rows) csa_attention_kv_element(compressed_key, compressed_key_scales, compressed_value, compressed_value_scales, selection, recent_key, recent_key_scales, recent_value, recent_value_scales, batch_key, batch_key_scales, batch_value, batch_value_scales, selection_top_k, query_row, compressed_count, recent_begin, recent_count, batch_begin, recent_start, window_size, kv_head_count, kv_head, head_dim, q8_group_size, key_start + key_local, column, 0, &key_value, &ignored_value);
            key_tile[index] = static_cast<Bf16>(key_value);
        }
        __syncthreads();
        if (threadIdx.x / warpSize == 0) {
            QueryFragment query_fragment;
            KeyFragment key_fragment;
            AccumulatorFragment scores;
            rocwmma::fill_fragment(scores, 0.0f);
            #pragma unroll
            for (unsigned int dim_start = 0; dim_start < CSA_ATTN_MAX_DIM_TILES * 16; dim_start += 16) {
                if (dim_start < head_dim) {
                    rocwmma::load_matrix_sync(query_fragment, query_tile + dim_start, head_dim);
                    rocwmma::load_matrix_sync(key_fragment, key_tile + dim_start, head_dim);
                    rocwmma::mma_sync(scores, query_fragment, key_fragment, scores);
                }
            }
            rocwmma::store_matrix_sync(score_tile, scores, CSA_ATTN_KEY_TILE, rocwmma::mem_row_major);
        }
        __syncthreads();
        if (threadIdx.x < CSA_ATTN_HEAD_TILE) {
            const unsigned int head = threadIdx.x;
            float tile_maximum = -3.402823466e+38F;
            for (unsigned int key_local = 0; key_local < tile_rows; ++key_local) tile_maximum = fmaxf(tile_maximum, score_tile[head * CSA_ATTN_KEY_TILE + key_local] * rsqrtf((float)head_dim));
            const float next_maximum = fmaxf(running_maximum[head], tile_maximum);
            float denominator = running_denominator[head] * expf(running_maximum[head] - next_maximum);
            for (unsigned int key_local = 0; key_local < tile_rows; ++key_local) denominator += expf(score_tile[head * CSA_ATTN_KEY_TILE + key_local] * rsqrtf((float)head_dim) - next_maximum);
            running_maximum[head] = next_maximum;
            running_denominator[head] = denominator;
        }
        __syncthreads();
    }

    const unsigned int wave = threadIdx.x / warpSize;
    AccumulatorFragment output_fragments[CSA_ATTN_OUTPUT_TILES_PER_WAVE];
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_ATTN_OUTPUT_TILES_PER_WAVE; ++slot) rocwmma::fill_fragment(output_fragments[slot], 0.0f);

    // 第二遍用固定 softmax 状态形成 P tile，并通过 PV WMMA 累计最终输出。
    for (unsigned int key_start = 0; key_start < total_keys; key_start += CSA_ATTN_KEY_TILE) {
        const unsigned int tile_rows = min((unsigned int)CSA_ATTN_KEY_TILE, total_keys - key_start);
        for (unsigned int index = threadIdx.x; index < CSA_ATTN_KEY_TILE * head_dim; index += blockDim.x) {
            const unsigned int key_local = index / head_dim;
            const unsigned int column = index - key_local * head_dim;
            float key_value = 0.0f;
            float value = 0.0f;
            if (key_local < tile_rows) csa_attention_kv_element(compressed_key, compressed_key_scales, compressed_value, compressed_value_scales, selection, recent_key, recent_key_scales, recent_value, recent_value_scales, batch_key, batch_key_scales, batch_value, batch_value_scales, selection_top_k, query_row, compressed_count, recent_begin, recent_count, batch_begin, recent_start, window_size, kv_head_count, kv_head, head_dim, q8_group_size, key_start + key_local, column, 1, &key_value, &value);
            key_tile[index] = static_cast<Bf16>(key_value);
            value_tile[index] = static_cast<Bf16>(value);
        }
        __syncthreads();
        if (wave == 0) {
            QueryFragment query_fragment;
            KeyFragment key_fragment;
            AccumulatorFragment scores;
            rocwmma::fill_fragment(scores, 0.0f);
            #pragma unroll
            for (unsigned int dim_start = 0; dim_start < CSA_ATTN_MAX_DIM_TILES * 16; dim_start += 16) {
                if (dim_start < head_dim) {
                    rocwmma::load_matrix_sync(query_fragment, query_tile + dim_start, head_dim);
                    rocwmma::load_matrix_sync(key_fragment, key_tile + dim_start, head_dim);
                    rocwmma::mma_sync(scores, query_fragment, key_fragment, scores);
                }
            }
            rocwmma::store_matrix_sync(score_tile, scores, CSA_ATTN_KEY_TILE, rocwmma::mem_row_major);
        }
        __syncthreads();
        for (unsigned int index = threadIdx.x; index < CSA_ATTN_HEAD_TILE * CSA_ATTN_KEY_TILE; index += blockDim.x) {
            const unsigned int key_local = index % CSA_ATTN_KEY_TILE;
            const unsigned int head = index / CSA_ATTN_KEY_TILE;
            const float probability = key_local < tile_rows ? expf(score_tile[index] * rsqrtf((float)head_dim) - running_maximum[head]) / running_denominator[head] : 0.0f;
            probability_tile[index] = static_cast<Bf16>(probability);
        }
        __syncthreads();
        ProbabilityFragment probabilities;
        rocwmma::load_matrix_sync(probabilities, probability_tile, CSA_ATTN_KEY_TILE);
        #pragma unroll
        for (unsigned int slot = 0; slot < CSA_ATTN_OUTPUT_TILES_PER_WAVE; ++slot) {
            const unsigned int dim_tile = wave + slot * (blockDim.x / warpSize);
            if (dim_tile * 16 < head_dim) {
                ValueFragment values;
                rocwmma::load_matrix_sync(values, value_tile + dim_tile * 16, head_dim);
                rocwmma::mma_sync(output_fragments[slot], probabilities, values, output_fragments[slot]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_ATTN_OUTPUT_TILES_PER_WAVE; ++slot) {
        const unsigned int dim_tile = wave + slot * (blockDim.x / warpSize);
        if (dim_tile * 16 < head_dim) rocwmma::store_matrix_sync(output + query_base + dim_tile * 16, output_fragments[slot], head_dim, rocwmma::mem_row_major);
    }
}

#define CSA_SEGMENT_CAPACITY 16

struct CsaAttentionQ8SegmentDescriptor {
    unsigned long long compressed_key;
    unsigned long long compressed_key_scales;
    unsigned long long compressed_value;
    unsigned long long compressed_value_scales;
    unsigned long long visible_compressed;
    unsigned long long selection;
    unsigned long long recent_key;
    unsigned long long recent_key_scales;
    unsigned long long recent_value;
    unsigned long long recent_value_scales;
    unsigned long long batch_key;
    unsigned long long batch_key_scales;
    unsigned long long batch_value;
    unsigned long long batch_value_scales;
    unsigned int row_start;
    unsigned int query_rows;
    unsigned int selection_top_k;
    unsigned int recent_start;
    unsigned int recent_len;
    unsigned int recent_first_position;
    unsigned int position_start;
    unsigned int causal_batch;
};

struct CsaAttentionQ8SegmentTable {
    CsaAttentionQ8SegmentDescriptor segments[CSA_SEGMENT_CAPACITY];
};

extern "C" __global__ void csa_store_segment_table(
    CsaAttentionQ8SegmentTable table,
    CsaAttentionQ8SegmentDescriptor* output)
{
    const unsigned int index = threadIdx.x;
    if (index < CSA_SEGMENT_CAPACITY) output[index] = table.segments[index];
}

__device__ __forceinline__ const CsaAttentionQ8SegmentDescriptor* csa_attention_segment(
    const CsaAttentionQ8SegmentDescriptor* table,
    unsigned int segment_count,
    unsigned int global_row,
    unsigned int* local_row)
{
    #pragma unroll
    for (unsigned int index = 0; index < CSA_SEGMENT_CAPACITY; ++index) {
        if (index >= segment_count) break;
        const CsaAttentionQ8SegmentDescriptor* segment = table + index;
        if (global_row >= segment->row_start && global_row - segment->row_start < segment->query_rows) {
            *local_row = global_row - segment->row_start;
            return segment;
        }
    }
    *local_row = 0;
    return table;
}

extern "C" __global__ void csa_attention_tiled_q8(
    const float* query,
    const signed char* compressed_key,
    const unsigned short* compressed_key_scales,
    const signed char* compressed_value,
    const unsigned short* compressed_value_scales,
    const unsigned int* visible_compressed,
    const unsigned int* selection,
    const signed char* recent_key,
    const unsigned short* recent_key_scales,
    const signed char* recent_value,
    const unsigned short* recent_value_scales,
    const signed char* batch_key,
    const unsigned short* batch_key_scales,
    const signed char* batch_value,
    const unsigned short* batch_value_scales,
    const float* sink,
    float* output,
    unsigned int query_rows,
    unsigned int selection_top_k,
    unsigned int recent_start,
    unsigned int recent_len,
    unsigned int recent_first_position,
    unsigned int position_start,
    unsigned int causal_batch,
    unsigned int batch_rows,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int runtime_head_dim,
    unsigned int window_size,
    unsigned int q8_group_size,
    unsigned int has_sink)
{
    if (runtime_head_dim != CSA_TILED_HEAD_DIM) return;
    constexpr unsigned int head_dim = CSA_TILED_HEAD_DIM;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int head_groups = (head_count + waves - 1) / waves;
    const unsigned int query_row = blockIdx.x / head_groups;
    const unsigned int head_group = blockIdx.x % head_groups;
    const unsigned int head = head_group * waves + wave;
    const bool active = query_row < query_rows && head < head_count;
    const unsigned int first_head = head_group * waves;
    const unsigned int kv_head = first_head * kv_head_count / head_count;
    const unsigned int query_position = position_start + query_row;
    const unsigned long long query_base = ((unsigned long long)query_row * head_count + head) * head_dim;
    unsigned short query_values[CSA_TILED_CHANNELS];
    float value_sums[CSA_TILED_CHANNELS];
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_TILED_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        query_values[slot] = active && column < head_dim ? zllm_f32_to_bf16(query[query_base + column]) : 0u;
        value_sums[slot] = 0.0f;
    }
    float maximum = active && has_sink != 0 ? sink[head] : -3.402823466e+38F;
    float denominator = active && has_sink != 0 ? 1.0f : 0.0f;
    const float score_scale = rsqrtf((float)head_dim);
    __shared__ unsigned short key_tile[CSA_TILE_ROWS * CSA_TILED_CHANNELS * 32];
    __shared__ float value_tile[CSA_TILE_ROWS * CSA_TILED_CHANNELS * 32];
    const bool compressed_shared = compressed_key == compressed_value && compressed_key_scales == compressed_value_scales;
    const bool recent_shared = recent_key == recent_value && recent_key_scales == recent_value_scales;
    const bool batch_shared = batch_key == batch_value && batch_key_scales == batch_value_scales;

    const unsigned int visible = visible_compressed[query_row];
    const unsigned int compressed_count = selection_top_k == 0 ? visible : min(selection_top_k, visible);
    for (unsigned int base = 0; base < compressed_count; base += CSA_TILE_ROWS) {
        const unsigned int tile_rows = min((unsigned int)CSA_TILE_ROWS, compressed_count - base);
        for (unsigned int index = threadIdx.x; index < tile_rows * head_dim; index += blockDim.x) {
            const unsigned int tile_row = index / head_dim;
            const unsigned int column = index - tile_row * head_dim;
            const unsigned int slot = base + tile_row;
            const unsigned int row = selection_top_k == 0 ? slot : selection[(unsigned long long)query_row * selection_top_k + slot];
            const unsigned long long kv_base = ((unsigned long long)row * kv_head_count + kv_head) * head_dim;
            const float key_value = csa_q8_element(compressed_key, compressed_key_scales, kv_base + column, q8_group_size);
            key_tile[index] = zllm_f32_to_bf16(key_value);
            value_tile[index] = compressed_shared ? key_value : csa_q8_element(compressed_value, compressed_value_scales, kv_base + column, q8_group_size);
        }
        __syncthreads();
        if (active) csa_attention_wave_tile_tiled(query_values, value_sums, key_tile, value_tile, tile_rows, lane, head_dim, score_scale, &maximum, &denominator);
        __syncthreads();
    }

    unsigned int recent_begin = 0;
    unsigned int recent_end = 0;
    if (query_position >= recent_first_position) {
        recent_end = min(recent_len, query_position - recent_first_position + 1);
        const unsigned int first_visible_position = query_position + 1 > window_size ? query_position + 1 - window_size : 0;
        if (first_visible_position > recent_first_position) recent_begin = min(recent_end, first_visible_position - recent_first_position);
    }
    for (unsigned int base = recent_begin; base < recent_end; base += CSA_TILE_ROWS) {
        const unsigned int tile_rows = min((unsigned int)CSA_TILE_ROWS, recent_end - base);
        for (unsigned int index = threadIdx.x; index < tile_rows * head_dim; index += blockDim.x) {
            const unsigned int tile_row = index / head_dim;
            const unsigned int column = index - tile_row * head_dim;
            const unsigned int logical = base + tile_row;
            const unsigned int physical = (recent_start + logical) % window_size;
            const unsigned long long kv_base = ((unsigned long long)physical * kv_head_count + kv_head) * head_dim;
            const float key_value = csa_q8_element(recent_key, recent_key_scales, kv_base + column, q8_group_size);
            key_tile[index] = zllm_f32_to_bf16(key_value);
            value_tile[index] = recent_shared ? key_value : csa_q8_element(recent_value, recent_value_scales, kv_base + column, q8_group_size);
        }
        __syncthreads();
        if (active) csa_attention_wave_tile_tiled(query_values, value_sums, key_tile, value_tile, tile_rows, lane, head_dim, score_scale, &maximum, &denominator);
        __syncthreads();
    }

    const unsigned int batch_begin = query_row + 1 > window_size ? query_row + 1 - window_size : 0;
    const unsigned int batch_end = causal_batch != 0 ? min(query_row + 1, batch_rows) : batch_rows;
    for (unsigned int base = batch_begin; base < batch_end; base += CSA_TILE_ROWS) {
        const unsigned int tile_rows = min((unsigned int)CSA_TILE_ROWS, batch_end - base);
        for (unsigned int index = threadIdx.x; index < tile_rows * head_dim; index += blockDim.x) {
            const unsigned int tile_row = index / head_dim;
            const unsigned int column = index - tile_row * head_dim;
            const unsigned int row = base + tile_row;
            const unsigned long long kv_base = ((unsigned long long)row * kv_head_count + kv_head) * head_dim;
            const float key_value = csa_q8_element(batch_key, batch_key_scales, kv_base + column, q8_group_size);
            key_tile[index] = zllm_f32_to_bf16(key_value);
            value_tile[index] = batch_shared ? key_value : csa_q8_element(batch_value, batch_value_scales, kv_base + column, q8_group_size);
        }
        __syncthreads();
        if (active) csa_attention_wave_tile_tiled(query_values, value_sums, key_tile, value_tile, tile_rows, lane, head_dim, score_scale, &maximum, &denominator);
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (unsigned int slot = 0; slot < CSA_TILED_CHANNELS; ++slot) {
            const unsigned int column = lane + slot * warpSize;
            if (column < head_dim) output[query_base + column] = denominator == 0.0f ? 0.0f : value_sums[slot] / denominator;
        }
    }
}

// 多 session 共享连续 Q/output；descriptor 只保存独立 cache 指针与 ring 元数据。
extern "C" __global__ void csa_attention_segmented_tiled_q8(
    const float* query,
    const CsaAttentionQ8SegmentDescriptor* table,
    const float* sink,
    float* output,
    unsigned int segment_count,
    unsigned int total_query_rows,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int runtime_head_dim,
    unsigned int window_size,
    unsigned int q8_group_size,
    unsigned int has_sink)
{
    if (runtime_head_dim != CSA_TILED_HEAD_DIM) return;
    constexpr unsigned int head_dim = CSA_TILED_HEAD_DIM;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int head_groups = (head_count + waves - 1) / waves;
    const unsigned int global_row = blockIdx.x / head_groups;
    const unsigned int head_group = blockIdx.x % head_groups;
    unsigned int query_row = 0;
    const CsaAttentionQ8SegmentDescriptor* descriptor = csa_attention_segment(table, segment_count, global_row, &query_row);
    const unsigned int head = head_group * waves + wave;
    const bool active = global_row < total_query_rows && head < head_count;
    const unsigned int first_head = head_group * waves;
    const unsigned int kv_head = first_head * kv_head_count / head_count;
    const unsigned int query_position = descriptor->position_start + query_row;
    const unsigned long long query_base = ((unsigned long long)global_row * head_count + head) * head_dim;
    const signed char* compressed_key = reinterpret_cast<const signed char*>(descriptor->compressed_key);
    const unsigned short* compressed_key_scales = reinterpret_cast<const unsigned short*>(descriptor->compressed_key_scales);
    const signed char* compressed_value = reinterpret_cast<const signed char*>(descriptor->compressed_value);
    const unsigned short* compressed_value_scales = reinterpret_cast<const unsigned short*>(descriptor->compressed_value_scales);
    const unsigned int* visible_compressed = reinterpret_cast<const unsigned int*>(descriptor->visible_compressed);
    const unsigned int* selection = reinterpret_cast<const unsigned int*>(descriptor->selection);
    const signed char* recent_key = reinterpret_cast<const signed char*>(descriptor->recent_key);
    const unsigned short* recent_key_scales = reinterpret_cast<const unsigned short*>(descriptor->recent_key_scales);
    const signed char* recent_value = reinterpret_cast<const signed char*>(descriptor->recent_value);
    const unsigned short* recent_value_scales = reinterpret_cast<const unsigned short*>(descriptor->recent_value_scales);
    const signed char* batch_key = reinterpret_cast<const signed char*>(descriptor->batch_key);
    const unsigned short* batch_key_scales = reinterpret_cast<const unsigned short*>(descriptor->batch_key_scales);
    const signed char* batch_value = reinterpret_cast<const signed char*>(descriptor->batch_value);
    const unsigned short* batch_value_scales = reinterpret_cast<const unsigned short*>(descriptor->batch_value_scales);
    unsigned short query_values[CSA_TILED_CHANNELS];
    float value_sums[CSA_TILED_CHANNELS];
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_TILED_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        query_values[slot] = active && column < head_dim ? zllm_f32_to_bf16(query[query_base + column]) : 0u;
        value_sums[slot] = 0.0f;
    }
    float maximum = active && has_sink != 0 ? sink[head] : -3.402823466e+38F;
    float denominator = active && has_sink != 0 ? 1.0f : 0.0f;
    const float score_scale = rsqrtf((float)head_dim);
    __shared__ unsigned short key_tile[CSA_TILE_ROWS * CSA_TILED_CHANNELS * 32];
    __shared__ float value_tile[CSA_TILE_ROWS * CSA_TILED_CHANNELS * 32];
    const bool compressed_shared = compressed_key == compressed_value && compressed_key_scales == compressed_value_scales;
    const bool recent_shared = recent_key == recent_value && recent_key_scales == recent_value_scales;
    const bool batch_shared = batch_key == batch_value && batch_key_scales == batch_value_scales;

    const unsigned int visible = visible_compressed[query_row];
    const unsigned int compressed_count = descriptor->selection_top_k == 0 ? visible : min(descriptor->selection_top_k, visible);
    for (unsigned int base = 0; base < compressed_count; base += CSA_TILE_ROWS) {
        const unsigned int tile_rows = min((unsigned int)CSA_TILE_ROWS, compressed_count - base);
        for (unsigned int index = threadIdx.x; index < tile_rows * head_dim; index += blockDim.x) {
            const unsigned int tile_row = index / head_dim;
            const unsigned int column = index - tile_row * head_dim;
            const unsigned int slot = base + tile_row;
            const unsigned int row = descriptor->selection_top_k == 0 ? slot : selection[(unsigned long long)query_row * descriptor->selection_top_k + slot];
            const unsigned long long kv_base = ((unsigned long long)row * kv_head_count + kv_head) * head_dim;
            const float key_value = csa_q8_element(compressed_key, compressed_key_scales, kv_base + column, q8_group_size);
            key_tile[index] = zllm_f32_to_bf16(key_value);
            value_tile[index] = compressed_shared ? key_value : csa_q8_element(compressed_value, compressed_value_scales, kv_base + column, q8_group_size);
        }
        __syncthreads();
        if (active) csa_attention_wave_tile_tiled(query_values, value_sums, key_tile, value_tile, tile_rows, lane, head_dim, score_scale, &maximum, &denominator);
        __syncthreads();
    }

    unsigned int recent_begin = 0;
    unsigned int recent_end = 0;
    if (query_position >= descriptor->recent_first_position) {
        recent_end = min(descriptor->recent_len, query_position - descriptor->recent_first_position + 1);
        const unsigned int first_visible_position = query_position + 1 > window_size ? query_position + 1 - window_size : 0;
        if (first_visible_position > descriptor->recent_first_position) recent_begin = min(recent_end, first_visible_position - descriptor->recent_first_position);
    }
    for (unsigned int base = recent_begin; base < recent_end; base += CSA_TILE_ROWS) {
        const unsigned int tile_rows = min((unsigned int)CSA_TILE_ROWS, recent_end - base);
        for (unsigned int index = threadIdx.x; index < tile_rows * head_dim; index += blockDim.x) {
            const unsigned int tile_row = index / head_dim;
            const unsigned int column = index - tile_row * head_dim;
            const unsigned int physical = (descriptor->recent_start + base + tile_row) % window_size;
            const unsigned long long kv_base = ((unsigned long long)physical * kv_head_count + kv_head) * head_dim;
            const float key_value = csa_q8_element(recent_key, recent_key_scales, kv_base + column, q8_group_size);
            key_tile[index] = zllm_f32_to_bf16(key_value);
            value_tile[index] = recent_shared ? key_value : csa_q8_element(recent_value, recent_value_scales, kv_base + column, q8_group_size);
        }
        __syncthreads();
        if (active) csa_attention_wave_tile_tiled(query_values, value_sums, key_tile, value_tile, tile_rows, lane, head_dim, score_scale, &maximum, &denominator);
        __syncthreads();
    }

    const unsigned int batch_begin = query_row + 1 > window_size ? query_row + 1 - window_size : 0;
    const unsigned int batch_end = descriptor->causal_batch != 0 ? min(query_row + 1, descriptor->query_rows) : descriptor->query_rows;
    for (unsigned int base = batch_begin; base < batch_end; base += CSA_TILE_ROWS) {
        const unsigned int tile_rows = min((unsigned int)CSA_TILE_ROWS, batch_end - base);
        for (unsigned int index = threadIdx.x; index < tile_rows * head_dim; index += blockDim.x) {
            const unsigned int tile_row = index / head_dim;
            const unsigned int column = index - tile_row * head_dim;
            const unsigned int row = base + tile_row;
            const unsigned long long kv_base = ((unsigned long long)row * kv_head_count + kv_head) * head_dim;
            const float key_value = csa_q8_element(batch_key, batch_key_scales, kv_base + column, q8_group_size);
            key_tile[index] = zllm_f32_to_bf16(key_value);
            value_tile[index] = batch_shared ? key_value : csa_q8_element(batch_value, batch_value_scales, kv_base + column, q8_group_size);
        }
        __syncthreads();
        if (active) csa_attention_wave_tile_tiled(query_values, value_sums, key_tile, value_tile, tile_rows, lane, head_dim, score_scale, &maximum, &denominator);
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (unsigned int slot = 0; slot < CSA_TILED_CHANNELS; ++slot) {
            const unsigned int column = lane + slot * warpSize;
            if (column < head_dim) output[query_base + column] = denominator == 0.0f ? 0.0f : value_sums[slot] / denominator;
        }
    }
}

// Decode 小批次(query_rows<=8)的 split-KV 第一阶段:把拼接后的逻辑 key 序列
// 按 segment 切给多个 block,每 block 只扫自己的 [begin,end) 段并输出该段的
// (maximum, denominator, acc[head_dim]) partial。wave=head 的映射、LDS tile
// 装载与 online-softmax 步进与 tiled kernel 完全一致,仅循环范围与 epilogue
// 不同;sink 只由 segment 0 注入,保证全局恰好一次。tiled 路径在 decode 时
// grid 仅 rows*4 个 block,长上下文下 96 CU 大量闲置,本路径用
// rows*head_groups*segments 个 block 恢复并行度。
extern "C" __global__ void csa_attention_decode_partial_q8(
    const float* query,
    const signed char* compressed_key,
    const unsigned short* compressed_key_scales,
    const signed char* compressed_value,
    const unsigned short* compressed_value_scales,
    const unsigned int* visible_compressed,
    const unsigned int* selection,
    const signed char* recent_key,
    const unsigned short* recent_key_scales,
    const signed char* recent_value,
    const unsigned short* recent_value_scales,
    const signed char* batch_key,
    const unsigned short* batch_key_scales,
    const signed char* batch_value,
    const unsigned short* batch_value_scales,
    const float* sink,
    float* partials,
    unsigned int query_rows,
    unsigned int selection_top_k,
    unsigned int recent_start,
    unsigned int recent_len,
    unsigned int recent_first_position,
    unsigned int position_start,
    unsigned int causal_batch,
    unsigned int batch_rows,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int head_dim,
    unsigned int window_size,
    unsigned int q8_group_size,
    unsigned int has_sink,
    unsigned int segments)
{
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int head_groups = (head_count + waves - 1) / waves;
    const unsigned int block_per_row = head_groups * segments;
    const unsigned int query_row = blockIdx.x / block_per_row;
    const unsigned int remainder = blockIdx.x - query_row * block_per_row;
    const unsigned int head_group = remainder / segments;
    const unsigned int segment = remainder - head_group * segments;
    const unsigned int head = head_group * waves + wave;
    const bool active = query_row < query_rows && head < head_count;
    const unsigned int first_head = head_group * waves;
    const unsigned int kv_head = first_head * kv_head_count / head_count;
    const unsigned int query_position = position_start + query_row;
    const unsigned long long query_base = ((unsigned long long)query_row * head_count + head) * head_dim;
    float query_values[CSA_WAVE_CHANNELS];
    float value_sums[CSA_WAVE_CHANNELS];
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_WAVE_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        query_values[slot] = active && column < head_dim ? query[query_base + column] : 0.0f;
        value_sums[slot] = 0.0f;
    }
    float maximum = active && has_sink != 0 && segment == 0 ? sink[head] : -3.402823466e+38F;
    float denominator = active && has_sink != 0 && segment == 0 ? 1.0f : 0.0f;
    const float score_scale = rsqrtf((float)head_dim);
    __shared__ float key_tile[CSA_TILE_ROWS * 1024];
    __shared__ float value_tile[CSA_TILE_ROWS * 1024];

    const unsigned int visible = visible_compressed[query_row];
    const unsigned int compressed_count = selection_top_k == 0 ? visible : min(selection_top_k, visible);
    unsigned int recent_begin = 0;
    unsigned int recent_end = 0;
    if (query_position >= recent_first_position) {
        recent_end = min(recent_len, query_position - recent_first_position + 1);
        const unsigned int first_visible_position = query_position + 1 > window_size ? query_position + 1 - window_size : 0;
        if (first_visible_position > recent_first_position) recent_begin = min(recent_end, first_visible_position - recent_first_position);
    }
    const unsigned int batch_begin = query_row + 1 > window_size ? query_row + 1 - window_size : 0;
    const unsigned int batch_end = causal_batch != 0 ? min(query_row + 1, batch_rows) : batch_rows;
    const unsigned int total_keys = compressed_count + (recent_end - recent_begin) + (batch_end - batch_begin);
    const unsigned int per_segment = (total_keys + segments - 1u) / segments;
    const unsigned int key_begin = min(total_keys, segment * per_segment);
    const unsigned int key_end = min(total_keys, key_begin + per_segment);

    for (unsigned int base = key_begin; base < key_end; base += CSA_TILE_ROWS) {
        const unsigned int tile_rows = min((unsigned int)CSA_TILE_ROWS, key_end - base);
        for (unsigned int index = threadIdx.x; index < tile_rows * head_dim; index += blockDim.x) {
            const unsigned int tile_row = index / head_dim;
            const unsigned int column = index - tile_row * head_dim;
            float key_value = 0.0f;
            float value = 0.0f;
            csa_attention_kv_element(compressed_key, compressed_key_scales, compressed_value, compressed_value_scales, selection, recent_key, recent_key_scales, recent_value, recent_value_scales, batch_key, batch_key_scales, batch_value, batch_value_scales, selection_top_k, query_row, compressed_count, recent_begin, recent_end - recent_begin, batch_begin, recent_start, window_size, kv_head_count, kv_head, head_dim, q8_group_size, base + tile_row, column, 1, &key_value, &value);
            key_tile[index] = key_value;
            value_tile[index] = value;
        }
        __syncthreads();
        if (active) {
            #pragma unroll
            for (unsigned int tile_row = 0; tile_row < CSA_TILE_ROWS; ++tile_row) {
                if (tile_row < tile_rows) csa_attention_wave_step(query_values, value_sums, key_tile + tile_row * head_dim, value_tile + tile_row * head_dim, lane, head_dim, score_scale, &maximum, &denominator);
            }
        }
        __syncthreads();
    }
    if (active) {
        float* slot = partials + (((unsigned long long)query_row * head_count + head) * segments + segment) * (head_dim + 2u);
        slot[0] = maximum;
        slot[1] = denominator;
        #pragma unroll
        for (unsigned int slot_index = 0; slot_index < CSA_WAVE_CHANNELS; ++slot_index) {
            const unsigned int column = lane + slot_index * warpSize;
            if (column < head_dim) slot[2 + column] = value_sums[slot_index];
        }
    }
}

// segmented split-KV：grid 同时展开 session row、head group 与 key segment。
extern "C" __global__ void csa_attention_segmented_decode_partial_q8(
    const float* query,
    const CsaAttentionQ8SegmentDescriptor* table,
    const float* sink,
    float* partials,
    unsigned int segment_count,
    unsigned int total_query_rows,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int head_dim,
    unsigned int window_size,
    unsigned int q8_group_size,
    unsigned int has_sink,
    unsigned int split_segments)
{
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int waves = blockDim.x / warpSize;
    const unsigned int head_groups = (head_count + waves - 1) / waves;
    const unsigned int blocks_per_row = head_groups * split_segments;
    const unsigned int global_row = blockIdx.x / blocks_per_row;
    const unsigned int remainder = blockIdx.x - global_row * blocks_per_row;
    const unsigned int head_group = remainder / split_segments;
    const unsigned int split = remainder - head_group * split_segments;
    unsigned int query_row = 0;
    const CsaAttentionQ8SegmentDescriptor* descriptor = csa_attention_segment(table, segment_count, global_row, &query_row);
    const unsigned int head = head_group * waves + wave;
    const bool active = global_row < total_query_rows && head < head_count;
    const unsigned int first_head = head_group * waves;
    const unsigned int kv_head = first_head * kv_head_count / head_count;
    const unsigned int query_position = descriptor->position_start + query_row;
    const unsigned long long query_base = ((unsigned long long)global_row * head_count + head) * head_dim;
    const signed char* compressed_key = reinterpret_cast<const signed char*>(descriptor->compressed_key);
    const unsigned short* compressed_key_scales = reinterpret_cast<const unsigned short*>(descriptor->compressed_key_scales);
    const signed char* compressed_value = reinterpret_cast<const signed char*>(descriptor->compressed_value);
    const unsigned short* compressed_value_scales = reinterpret_cast<const unsigned short*>(descriptor->compressed_value_scales);
    const unsigned int* visible_compressed = reinterpret_cast<const unsigned int*>(descriptor->visible_compressed);
    const unsigned int* selection = reinterpret_cast<const unsigned int*>(descriptor->selection);
    const signed char* recent_key = reinterpret_cast<const signed char*>(descriptor->recent_key);
    const unsigned short* recent_key_scales = reinterpret_cast<const unsigned short*>(descriptor->recent_key_scales);
    const signed char* recent_value = reinterpret_cast<const signed char*>(descriptor->recent_value);
    const unsigned short* recent_value_scales = reinterpret_cast<const unsigned short*>(descriptor->recent_value_scales);
    const signed char* batch_key = reinterpret_cast<const signed char*>(descriptor->batch_key);
    const unsigned short* batch_key_scales = reinterpret_cast<const unsigned short*>(descriptor->batch_key_scales);
    const signed char* batch_value = reinterpret_cast<const signed char*>(descriptor->batch_value);
    const unsigned short* batch_value_scales = reinterpret_cast<const unsigned short*>(descriptor->batch_value_scales);
    float query_values[CSA_WAVE_CHANNELS];
    float value_sums[CSA_WAVE_CHANNELS];
    #pragma unroll
    for (unsigned int slot = 0; slot < CSA_WAVE_CHANNELS; ++slot) {
        const unsigned int column = lane + slot * warpSize;
        query_values[slot] = active && column < head_dim ? query[query_base + column] : 0.0f;
        value_sums[slot] = 0.0f;
    }
    float maximum = active && has_sink != 0 && split == 0 ? sink[head] : -3.402823466e+38F;
    float denominator = active && has_sink != 0 && split == 0 ? 1.0f : 0.0f;
    const float score_scale = rsqrtf((float)head_dim);
    __shared__ float key_tile[CSA_TILE_ROWS * 1024];
    __shared__ float value_tile[CSA_TILE_ROWS * 1024];
    const unsigned int visible = visible_compressed[query_row];
    const unsigned int compressed_count = descriptor->selection_top_k == 0 ? visible : min(descriptor->selection_top_k, visible);
    unsigned int recent_begin = 0;
    unsigned int recent_end = 0;
    if (query_position >= descriptor->recent_first_position) {
        recent_end = min(descriptor->recent_len, query_position - descriptor->recent_first_position + 1);
        const unsigned int first_visible_position = query_position + 1 > window_size ? query_position + 1 - window_size : 0;
        if (first_visible_position > descriptor->recent_first_position) recent_begin = min(recent_end, first_visible_position - descriptor->recent_first_position);
    }
    const unsigned int batch_begin = query_row + 1 > window_size ? query_row + 1 - window_size : 0;
    const unsigned int batch_end = descriptor->causal_batch != 0 ? min(query_row + 1, descriptor->query_rows) : descriptor->query_rows;
    const unsigned int total_keys = compressed_count + (recent_end - recent_begin) + (batch_end - batch_begin);
    const unsigned int keys_per_split = (total_keys + split_segments - 1u) / split_segments;
    const unsigned int key_begin = min(total_keys, split * keys_per_split);
    const unsigned int key_end = min(total_keys, key_begin + keys_per_split);
    for (unsigned int base = key_begin; base < key_end; base += CSA_TILE_ROWS) {
        const unsigned int tile_rows = min((unsigned int)CSA_TILE_ROWS, key_end - base);
        for (unsigned int index = threadIdx.x; index < tile_rows * head_dim; index += blockDim.x) {
            const unsigned int tile_row = index / head_dim;
            const unsigned int column = index - tile_row * head_dim;
            float key_value = 0.0f;
            float value = 0.0f;
            csa_attention_kv_element(compressed_key, compressed_key_scales, compressed_value, compressed_value_scales, selection, recent_key, recent_key_scales, recent_value, recent_value_scales, batch_key, batch_key_scales, batch_value, batch_value_scales, descriptor->selection_top_k, query_row, compressed_count, recent_begin, recent_end - recent_begin, batch_begin, descriptor->recent_start, window_size, kv_head_count, kv_head, head_dim, q8_group_size, base + tile_row, column, 1, &key_value, &value);
            key_tile[index] = key_value;
            value_tile[index] = value;
        }
        __syncthreads();
        if (active) {
            #pragma unroll
            for (unsigned int tile_row = 0; tile_row < CSA_TILE_ROWS; ++tile_row) {
                if (tile_row < tile_rows) csa_attention_wave_step(query_values, value_sums, key_tile + tile_row * head_dim, value_tile + tile_row * head_dim, lane, head_dim, score_scale, &maximum, &denominator);
            }
        }
        __syncthreads();
    }
    if (active) {
        float* output_slot = partials + (((unsigned long long)global_row * head_count + head) * split_segments + split) * (head_dim + 2u);
        output_slot[0] = maximum;
        output_slot[1] = denominator;
        #pragma unroll
        for (unsigned int slot = 0; slot < CSA_WAVE_CHANNELS; ++slot) {
            const unsigned int column = lane + slot * warpSize;
            if (column < head_dim) output_slot[2 + column] = value_sums[slot];
        }
    }
}

// split-KV 第二阶段:每 block 归并一个 (query_row, head) 的全部 segment
// partial。全局 maximum/denominator 与 numerator 重标定与 online-softmax 的
// 跨段合并等价;d==0(全部为空)时输出 0,与单遍 kernel 一致。
extern "C" __global__ void csa_attention_decode_merge_f32(
    const float* partials,
    float* output,
    unsigned int query_rows,
    unsigned int head_count,
    unsigned int segments,
    unsigned int head_dim)
{
    const unsigned int item = blockIdx.x;
    if (item >= query_rows * head_count) return;
    const unsigned long long slot_base = ((unsigned long long)item) * segments * (head_dim + 2u);
    float maximum = -3.402823466e+38F;
    for (unsigned int segment = 0; segment < segments; ++segment) {
        maximum = fmaxf(maximum, partials[slot_base + (unsigned long long)segment * (head_dim + 2u)]);
    }
    float denominator = 0.0f;
    for (unsigned int segment = 0; segment < segments; ++segment) {
        const float* slot = partials + slot_base + (unsigned long long)segment * (head_dim + 2u);
        denominator += slot[1] * expf(slot[0] - maximum);
    }
    const unsigned long long output_base = ((unsigned long long)item) * head_dim;
    for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
        float numerator = 0.0f;
        for (unsigned int segment = 0; segment < segments; ++segment) {
            const float* slot = partials + slot_base + (unsigned long long)segment * (head_dim + 2u);
            numerator += slot[2 + column] * expf(slot[0] - maximum);
        }
        output[output_base + column] = denominator == 0.0f ? 0.0f : numerator / denominator;
    }
}
