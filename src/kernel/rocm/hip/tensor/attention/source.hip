__device__ __forceinline__ unsigned short zllm_attention_f32_to_bf16(float value)
{
    const unsigned int bits = __float_as_uint(value);
    const unsigned int rounding = 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)((bits + rounding) >> 16);
}

__device__ __forceinline__ float zllm_attention_pair_dot_rn(float a0, float b0, float a1, float b1)
{
#pragma clang fp contract(off)
    return a0 * b0 + a1 * b1;
}

extern "C" __global__ void prepare_attention_qkv_bf16(
    const float* qkv,
    const float* query_weight,
    const float* key_weight,
    const float* cosine,
    const float* sine,
    unsigned short* packed_query,
    unsigned short* packed_key,
    unsigned short* packed_value,
    unsigned int rows,
    unsigned int packed_rows,
    unsigned int total_head_count,
    unsigned int head_start,
    unsigned int head_count,
    unsigned int head_dim,
    unsigned int rotary_dim,
    float eps)
{
    const unsigned int row_head = blockIdx.x;
    const unsigned int row = row_head / head_count;
    const unsigned int local_head = row_head - row * head_count;
    const unsigned int input_head = head_start + local_head;
    const unsigned int lane = threadIdx.x;
    const unsigned int total_columns = total_head_count * head_dim;
    const unsigned int columns = head_count * head_dim;
    const unsigned long long output_base = (unsigned long long)row * columns + local_head * head_dim;
    if (row >= packed_rows || local_head >= head_count || input_head >= total_head_count) return;
    if (row >= rows) {
        if (lane < head_dim) {
            packed_query[output_base + lane] = 0;
            packed_key[output_base + lane] = 0;
            packed_value[output_base + lane] = 0;
        }
        return;
    }

    __shared__ float query_values[128];
    __shared__ float key_values[128];
    __shared__ float query_wave_sums[4];
    __shared__ float key_wave_sums[4];
    float query_value = 0.0f;
    float key_value = 0.0f;
    float value_value = 0.0f;
    if (lane < head_dim) {
        const unsigned long long input_base = (unsigned long long)row * (total_columns * 3u) + input_head * head_dim + lane;
        query_value = qkv[input_base];
        key_value = qkv[input_base + total_columns];
        value_value = qkv[input_base + total_columns * 2u];
    }
    float query_sum = lane < head_dim ? query_value * query_value : 0.0f;
    float key_sum = lane < head_dim ? key_value * key_value : 0.0f;
#pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        query_sum += __shfl_down(query_sum, offset, 32);
        key_sum += __shfl_down(key_sum, offset, 32);
    }
    if ((lane & 31u) == 0) {
        query_wave_sums[lane >> 5] = query_sum;
        key_wave_sums[lane >> 5] = key_sum;
    }
    __syncthreads();
    if (lane < 32) {
        query_sum = lane < 4 ? query_wave_sums[lane] : 0.0f;
        key_sum = lane < 4 ? key_wave_sums[lane] : 0.0f;
#pragma unroll
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            query_sum += __shfl_down(query_sum, offset, 32);
            key_sum += __shfl_down(key_sum, offset, 32);
        }
        if (lane == 0) {
            query_wave_sums[0] = rsqrtf(query_sum / (float)head_dim + eps);
            key_wave_sums[0] = rsqrtf(key_sum / (float)head_dim + eps);
        }
    }
    __syncthreads();
    if (lane < head_dim) {
        query_values[lane] = query_value * query_wave_sums[0] * query_weight[lane];
        key_values[lane] = key_value * key_wave_sums[0] * key_weight[lane];
    }
    __syncthreads();
    if (lane < head_dim) {
        const unsigned int half = rotary_dim / 2u;
        if (lane < half) {
            const float c = cosine[(unsigned long long)row * half + lane];
            const float s = sine[(unsigned long long)row * half + lane];
            const float query_real = query_values[lane];
            const float query_imaginary = query_values[half + lane];
            const float key_real = key_values[lane];
            const float key_imaginary = key_values[half + lane];
            packed_query[output_base + lane] =
                zllm_attention_f32_to_bf16(query_real * c - query_imaginary * s);
            packed_query[output_base + half + lane] =
                zllm_attention_f32_to_bf16(query_imaginary * c + query_real * s);
            packed_key[output_base + lane] =
                zllm_attention_f32_to_bf16(key_real * c - key_imaginary * s);
            packed_key[output_base + half + lane] =
                zllm_attention_f32_to_bf16(key_imaginary * c + key_real * s);
        } else if (lane >= rotary_dim) {
            packed_query[output_base + lane] =
                zllm_attention_f32_to_bf16(query_values[lane]);
            packed_key[output_base + lane] =
                zllm_attention_f32_to_bf16(key_values[lane]);
        }
        packed_value[output_base + lane] = zllm_attention_f32_to_bf16(value_value);
    }
}

// 每个 CTA 独占一个 16x128 head tile：先完整载入 fragment，再把 native
// mAccess 字节写回同一 tile，避免 attention 的每个 query block 重复转换。
extern "C" __global__ void pack_attention_kv_native_bf16(
    unsigned short* key,
    unsigned short* value,
    unsigned int packed_rows,
    unsigned int head_count)
{
    using namespace rocwmma;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using BRowFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major>;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int tile_head = blockIdx.x;
    const unsigned int head = tile_head % head_count;
    const unsigned int tile = tile_head / head_count;
    const unsigned int tile_row = tile * 16;
    const unsigned int columns = head_count * 128;
    if (tile_row >= packed_rows || wave >= 8) return;

    BFragment key_fragment;
    BRowFragment value_fragment;
    load_matrix_sync(key_fragment, (const bfloat16_t*)(key + (unsigned long long)tile_row * columns + head * 128 + wave * 16), columns);
    load_matrix_sync(value_fragment, (const bfloat16_t*)(value + (unsigned long long)tile_row * columns + head * 128 + wave * 16), columns);
    __syncthreads();

    using BAccess = decltype(key_fragment.mAccess);
    using BRowAccess = decltype(value_fragment.mAccess);
    static_assert(sizeof(BAccess) == 16 && sizeof(BRowAccess) == 16, "rocWMMA native K/V access 必须为 16 字节");
    const unsigned int native_index = (wave * 32 + lane) * 8;
    const unsigned int native_row = native_index / 128;
    const unsigned int native_column = native_index - native_row * 128;
    *(BAccess*)(key + (unsigned long long)(tile_row + native_row) * columns + head * 128 + native_column) = key_fragment.mAccess;
    *(BRowAccess*)(value + (unsigned long long)(tile_row + native_row) * columns + head * 128 + native_column) = value_fragment.mAccess;
}

extern "C" __global__ void pack_qkv_bf16(
    const float* query,
    const float* key,
    const float* value,
    unsigned short* packed_query,
    unsigned short* packed_key,
    unsigned short* packed_value,
    unsigned int elements,
    unsigned int packed_elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= packed_elements) return;
    packed_query[index] = index < elements ? zllm_attention_f32_to_bf16(query[index]) : 0;
    packed_key[index] = index < elements ? zllm_attention_f32_to_bf16(key[index]) : 0;
    packed_value[index] = index < elements ? zllm_attention_f32_to_bf16(value[index]) : 0;
}

extern "C" __global__ void pack_attention_heads_bf16(
    const float* query, const float* key, const float* value,
    unsigned short* packed_query, unsigned short* packed_key, unsigned short* packed_value,
    unsigned int rows, unsigned int heads, unsigned int head_dim, unsigned int padded_head_dim)
{
    const unsigned long long elements = (unsigned long long)rows * heads * padded_head_dim;
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int dimension = index % padded_head_dim;
    const unsigned long long item = index / padded_head_dim;
    if (dimension < head_dim) {
        const unsigned long long source = item * head_dim + dimension;
        packed_query[index] = zllm_attention_f32_to_bf16(query[source]);
        packed_key[index] = zllm_attention_f32_to_bf16(key[source]);
        packed_value[index] = zllm_attention_f32_to_bf16(value[source]);
    } else {
        packed_query[index] = 0;
        packed_key[index] = 0;
        packed_value[index] = 0;
    }
}

extern "C" __global__ void unpack_attention_heads_f32(
    const float* input, float* output,
    unsigned int rows, unsigned int heads, unsigned int head_dim, unsigned int padded_head_dim)
{
    const unsigned long long elements = (unsigned long long)rows * heads * head_dim;
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int dimension = index % head_dim;
    const unsigned long long item = index / head_dim;
    output[index] = input[item * padded_head_dim + dimension];
}

extern "C" __global__ void pack_gqa_bf16(
    const float* query,
    const float* key,
    const float* value,
    unsigned short* packed_query,
    unsigned short* packed_key,
    unsigned short* packed_value,
    unsigned int query_elements,
    unsigned int packed_query_elements,
    unsigned int kv_elements,
    unsigned int packed_elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= packed_elements) return;
    if (index < packed_query_elements) {
        packed_query[index] = index < query_elements ? zllm_attention_f32_to_bf16(query[index]) : 0;
    }
    if (index < kv_elements) {
        packed_key[index] = zllm_attention_f32_to_bf16(key[index]);
        packed_value[index] = zllm_attention_f32_to_bf16(value[index]);
    }
}

// 首个 key tile 为每个 query 建立指数原点，避免全负 logits 在 BF16 概率中下溢。
// 后续 tile 先用 ballot 走固定原点快路，仅在 logit 接近原点时归约 max 并重标定历史累加器。
// head_dim=128 将下一块 K/V 预取到线程寄存器，当前 QK/PV 完成后再提交备用 LDS。
template <unsigned int HeadDim, bool Causal, unsigned int QueryRows = 128, unsigned int KeyRows = 16>
__device__ __forceinline__ void zllm_full_attention_f32(
    const unsigned short* query,
    const unsigned short* key,
    const unsigned short* value,
    float* output,
    unsigned int rows,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int head_dim,
    float score_scale,
    unsigned int attention_block,
    unsigned short* shared_key,
    unsigned short* shared_value,
    unsigned short (&shared_probability)[QueryRows / 16][16 * KeyRows])
{
    head_dim = HeadDim == 0 ? head_dim : HeadDim;
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using BRowFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    typedef unsigned short U16x8 __attribute__((ext_vector_type(8)));

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int blocks_per_head = (rows + QueryRows - 1) / QueryRows;
    const unsigned int head = attention_block / blocks_per_head;
    const unsigned int query_block = attention_block - head * blocks_per_head;
    const unsigned int query_base_row = query_block * QueryRows;
    const unsigned int query_columns = head_count * head_dim;
    const unsigned int kv_columns = kv_head_count * head_dim;
    const unsigned int kv_head = head / (head_count / kv_head_count);
    // gfx11 WMMA SOA：lane 固定一列，每个 fragment 元素对应半 wave 中的一行。
    const unsigned int key_slot = lane & 15u;
    const unsigned int query_half = lane >> 4;

    AFragment query_fragments[8];
    #pragma unroll
    for (unsigned int feature = 0; feature < 8; ++feature) {
        if (feature * 16 < head_dim) {
            load_matrix_sync(
                query_fragments[feature],
                (const bfloat16_t*)(query + (unsigned long long)(query_base_row + wave * 16) * query_columns + head * head_dim + feature * 16),
                query_columns);
        }
    }

    CFragment accumulators[8];
    #pragma unroll
    for (unsigned int feature = 0; feature < 8; ++feature) fill_fragment(accumulators[feature], 0.0f);
    CFragment maximum;
    CFragment denominator;
    #pragma unroll
    for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
        maximum[element] = -3.402823466e+38F;
        denominator[element] = 0.0f;
    }

    if constexpr (HeadDim == 128) {
        #pragma unroll
        for (unsigned int load = 0; load < KeyRows / 16; ++load) {
            const unsigned int index = (load * blockDim.x + threadIdx.x) * 8;
            const unsigned int tile_key = index / 128;
            const unsigned int column = index - tile_key * 128;
            U16x8 key_vector = {};
            U16x8 value_vector = {};
            if (tile_key < rows) {
                key_vector = *(const U16x8*)(key + (unsigned long long)tile_key * kv_columns + kv_head * 128 + column);
                value_vector = *(const U16x8*)(value + (unsigned long long)tile_key * kv_columns + kv_head * 128 + column);
            }
            *(U16x8*)(shared_key + index) = key_vector;
            *(U16x8*)(shared_value + index) = value_vector;
        }
        __syncthreads();
    }

    for (unsigned int key_base_row = 0; key_base_row < rows; key_base_row += KeyRows) {
        unsigned short* tile_key_values = shared_key;
        unsigned short* tile_value_values = shared_value;
        U16x8 next_key_vectors[KeyRows / 16] = {};
        U16x8 next_value_vectors[KeyRows / 16] = {};
        unsigned int native_buffer_offset = 0;
        unsigned int next_buffer_offset = 0;
        const unsigned int next_key_base_row = key_base_row + KeyRows;
        if constexpr (HeadDim == 128) {
            const unsigned int tile = key_base_row / KeyRows;
            const unsigned int buffer = (3u - tile % 3u) % 3u;
            tile_key_values += buffer * KeyRows * 128;
            tile_value_values += buffer * KeyRows * 128;
            native_buffer_offset = ((buffer + 1u) % 3u) * KeyRows * 128;
            if (next_key_base_row < rows) {
                next_buffer_offset = ((buffer + 2u) % 3u) * KeyRows * 128;
                #pragma unroll
                for (unsigned int load = 0; load < KeyRows / 16; ++load) {
                    const unsigned int index = (load * blockDim.x + threadIdx.x) * 8;
                    const unsigned int tile_key = index / 128;
                    const unsigned int column = index - tile_key * 128;
                    const unsigned int key_row = next_key_base_row + tile_key;
                    if (key_row < rows) {
                        next_key_vectors[load] = *(const U16x8*)(key + (unsigned long long)key_row * kv_columns + kv_head * 128 + column);
                        next_value_vectors[load] = *(const U16x8*)(value + (unsigned long long)key_row * kv_columns + kv_head * 128 + column);
                    }
                }
            }
        } else {
            const unsigned int tile_elements = KeyRows * head_dim;
            for (unsigned int index = threadIdx.x; index < tile_elements; index += blockDim.x) {
                const unsigned int tile_key = index / head_dim;
                const unsigned int column = index - tile_key * head_dim;
                const unsigned int key_row = key_base_row + tile_key;
                unsigned short key_value = 0;
                unsigned short value_value = 0;
                if (key_row < rows) {
                    key_value = key[(unsigned long long)key_row * kv_columns + kv_head * head_dim + column];
                    value_value = value[(unsigned long long)key_row * kv_columns + kv_head * head_dim + column];
                }
                shared_key[tile_key * 128 + column] = key_value;
                shared_value[tile_key * 128 + column] = value_value;
            }
            __syncthreads();
        }

        if constexpr (HeadDim == 128) {
            #pragma unroll
            for (unsigned int part = 0; part < KeyRows / 16; ++part) {
                BRowFragment native_value_fragment;
                load_matrix_sync(native_value_fragment, (const bfloat16_t*)(tile_value_values + part * 16 * 128 + wave * 16), 128);
                using BRowAccess = decltype(native_value_fragment.mAccess);
                BRowAccess* native_values = (BRowAccess*)(shared_value + native_buffer_offset);
                native_values[(part * 8 + wave) * 32 + lane] = native_value_fragment.mAccess;

                BFragment native_key_fragment;
                load_matrix_sync(native_key_fragment, (const bfloat16_t*)(tile_key_values + part * 16 * 128 + wave * 16), 128);
                using BAccess = decltype(native_key_fragment.mAccess);
                BAccess* native_keys = (BAccess*)(shared_key + native_buffer_offset);
                native_keys[(part * 8 + wave) * 32 + lane] = native_key_fragment.mAccess;
            }
            __syncthreads();
            if (next_key_base_row < rows) {
                #pragma unroll
                for (unsigned int load = 0; load < KeyRows / 16; ++load) {
                    const unsigned int index = (load * blockDim.x + threadIdx.x) * 8;
                    *(U16x8*)(shared_key + next_buffer_offset + index) = next_key_vectors[load];
                    *(U16x8*)(shared_value + next_buffer_offset + index) = next_value_vectors[load];
                }
            }
        }

        constexpr unsigned int ScoreParts = HeadDim == 128 ? KeyRows / 16 : 4u;
        CFragment scores[ScoreParts];
        #pragma unroll
        for (unsigned int part = 0; part < ScoreParts; ++part) fill_fragment(scores[part], 0.0f);
        #pragma unroll
        for (unsigned int feature = 0; feature < 8; ++feature) {
            if (feature * 16 >= head_dim) continue;
            if constexpr (HeadDim == 128) {
                BFragment key_fragment;
                using BAccess = decltype(key_fragment.mAccess);
                const BAccess* native_keys = (const BAccess*)(shared_key + native_buffer_offset);
                #pragma unroll
                for (unsigned int part = 0; part < KeyRows / 16; ++part) {
                    key_fragment.mAccess = native_keys[(part * 8 + feature) * 32 + lane];
                    mma_sync(scores[part], query_fragments[feature], key_fragment, scores[part]);
                }
            } else {
                BFragment key_fragment;
                load_matrix_sync(key_fragment, (const bfloat16_t*)(tile_key_values + feature * 16), 128);
                const unsigned int part = feature & 3u;
                mma_sync(scores[part], query_fragments[feature], key_fragment, scores[part]);
            }
        }
        if constexpr (HeadDim != 128) {
            #pragma unroll
            for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
                scores[0][element] += scores[1][element] + scores[2][element] + scores[3][element];
            }
        }

        const unsigned int key_count = min(KeyRows, rows - key_base_row);
        AFragment probability_fragments[KeyRows / 16];
        CFragment rescale;
        #pragma unroll
        for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
            const unsigned int local_query = element * 2 + query_half;
            const bool valid_query = head < head_count && query_base_row + wave * 16 + local_query < rows;
            const unsigned int query_row = query_base_row + wave * 16 + local_query;
            float scaled_scores[KeyRows / 16];
            bool near_origin = false;
            #pragma unroll
            for (unsigned int part = 0; part < KeyRows / 16; ++part) {
                const unsigned int local_key = part * 16 + key_slot;
                const bool visible = !Causal || key_base_row + local_key <= query_row;
                scaled_scores[part] = valid_query && local_key < key_count && visible
                    ? scores[part][element] * score_scale
                    : -3.402823466e+38F;
                near_origin |= valid_query && local_key < key_count && visible
                    && scaled_scores[part] > maximum[element] - 16.0f;
            }
            bool recenter = key_base_row == 0;
            if (key_base_row != 0) {
                const unsigned long long recenter_mask = __ballot(near_origin);
                recenter = ((recenter_mask >> (query_half * 16u)) & 0xffffu) != 0u;
            }
            float previous_scale = 1.0f;
            if (recenter) {
                float tile_maximum = scaled_scores[0];
                #pragma unroll
                for (unsigned int part = 1; part < KeyRows / 16; ++part) {
                    tile_maximum = fmaxf(tile_maximum, scaled_scores[part]);
                }
                #pragma unroll
                for (unsigned int offset = 1; offset < 16; offset <<= 1) {
                    tile_maximum = fmaxf(tile_maximum, __shfl_xor(tile_maximum, offset));
                }
                const float new_maximum = fmaxf(maximum[element], tile_maximum + 80.0f);
                if (key_base_row != 0) {
                    previous_scale = __builtin_amdgcn_exp2f(
                        (maximum[element] - new_maximum) * 1.4426950408889634f);
                    denominator[element] *= previous_scale;
                }
                maximum[element] = new_maximum;
            }
            rescale[element] = previous_scale;
            float tile_sum = 0.0f;
            #pragma unroll
            for (unsigned int part = 0; part < KeyRows / 16; ++part) {
                const unsigned int local_key = part * 16 + key_slot;
                float probability = 0.0f;
                if (valid_query && local_key < key_count && scaled_scores[part] > -3.402823466e+38F) {
                    probability = __builtin_amdgcn_exp2f(
                        (scaled_scores[part] - maximum[element]) * 1.4426950408889634f);
                }
                tile_sum += probability;
                shared_probability[wave][local_query * KeyRows + local_key] =
                    zllm_attention_f32_to_bf16(probability);
            }
            #pragma unroll
            for (unsigned int offset = 1; offset < 16; offset <<= 1) {
                tile_sum += __shfl_xor(tile_sum, offset);
            }
            denominator[element] += tile_sum;
        }
        __syncwarp();
        #pragma unroll
        for (unsigned int part = 0; part < KeyRows / 16; ++part) {
            load_matrix_sync(probability_fragments[part], (const bfloat16_t*)(shared_probability[wave] + part * 16), KeyRows);
        }
        #pragma unroll
        for (unsigned int feature = 0; feature < 8; ++feature) {
            if (feature * 16 < head_dim) {
                #pragma unroll
                for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
                    if (rescale[element] != 1.0f) accumulators[feature][element] *= rescale[element];
                }
            }
        }

        #pragma unroll
        for (unsigned int feature = 0; feature < 8; ++feature) {
            if (feature * 16 < head_dim) {
                BRowFragment value_fragment;
                if constexpr (HeadDim == 128) {
                            using BRowAccess = decltype(value_fragment.mAccess);
                    const BRowAccess* native_values = (const BRowAccess*)(shared_value + native_buffer_offset);
                    #pragma unroll
                    for (unsigned int part = 0; part < KeyRows / 16; ++part) {
                        value_fragment.mAccess = native_values[(part * 8 + feature) * 32 + lane];
                        mma_sync(accumulators[feature], probability_fragments[part], value_fragment, accumulators[feature]);
                    }
                } else {
                    load_matrix_sync(value_fragment, (const bfloat16_t*)(tile_value_values + feature * 16), 128);
                    mma_sync(accumulators[feature], probability_fragments[0], value_fragment, accumulators[feature]);
                }
            }
        }

        if constexpr (HeadDim == 128) {
            if (next_key_base_row < rows) {
                __syncthreads();
            }
        } else if (next_key_base_row < rows) {
            __syncthreads();
        }
    }

    #pragma unroll
    for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
        #pragma unroll
        for (unsigned int feature = 0; feature < 8; ++feature) {
            if (feature * 16 < head_dim) accumulators[feature][element] /= denominator[element];
        }
    }
    #pragma unroll
    for (unsigned int feature = 0; feature < 8; ++feature) {
        if (feature * 16 < head_dim) {
            #pragma unroll
            for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
                const unsigned int local_query = element * 2 + query_half;
                const unsigned int query_row = query_base_row + wave * 16 + local_query;
                const unsigned int column = feature * 16 + key_slot;
                if (head < head_count && query_row < rows && column < head_dim) {
                    output[(unsigned long long)query_row * query_columns + head * head_dim + column] =
                        accumulators[feature][element];
                }
            }
        }
    }
}

extern "C" __global__ void full_attention_f32(
    const unsigned short* query,
    const unsigned short* key,
    const unsigned short* value,
    float* output,
    unsigned int rows,
    unsigned int head_count,
    unsigned int head_dim,
    float score_scale)
{
    __shared__ unsigned short shared_key[16 * 128];
    __shared__ unsigned short shared_value[16 * 128];
    __shared__ unsigned short shared_probability[8][16 * 16];
    zllm_full_attention_f32<0, false>(query, key, value, output, rows, head_count, head_count, head_dim, score_scale, blockIdx.x,
        shared_key, shared_value, shared_probability);
}

// 非对称 Q/KV 与逐 query 可见区间。一个 block 负责一个 query head，
// 维度线程共同归约 QK，并各自累加对应的 V 列。
extern "C" __global__ void block_attention_f32(
    const float* query,
    const float* key,
    const float* value,
    const unsigned int* visible_ranges,
    float* output,
    unsigned int query_rows,
    unsigned int kv_rows,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int head_dim,
    float score_scale)
{
    const unsigned int query_row = blockIdx.x / head_count;
    const unsigned int head = blockIdx.x - query_row * head_count;
    if (query_row >= query_rows) return;
    const unsigned int kv_head = head / (head_count / kv_head_count);
    const unsigned int query_columns = head_count * head_dim;
    const unsigned int kv_columns = kv_head_count * head_dim;
    const unsigned int start = visible_ranges[query_row * 2];
    const unsigned int end = visible_ranges[query_row * 2 + 1];
    __shared__ float reduction[256];
    __shared__ float maximum;
    __shared__ float denominator;

    if (threadIdx.x == 0) maximum = -3.402823466e+38F;
    __syncthreads();
    for (unsigned int row = start; row < end && row < kv_rows; ++row) {
        float partial = 0.0f;
        for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
            partial += query[(unsigned long long)query_row * query_columns + head * head_dim + column]
                * key[(unsigned long long)row * kv_columns + kv_head * head_dim + column];
        }
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) maximum = fmaxf(maximum, reduction[0] * score_scale);
        __syncthreads();
    }

    if (threadIdx.x == 0) denominator = 0.0f;
    float accumulator = 0.0f;
    __syncthreads();
    for (unsigned int row = start; row < end && row < kv_rows; ++row) {
        float partial = 0.0f;
        for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
            partial += query[(unsigned long long)query_row * query_columns + head * head_dim + column]
                * key[(unsigned long long)row * kv_columns + kv_head * head_dim + column];
        }
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        const float probability = expf(reduction[0] * score_scale - maximum);
        if (threadIdx.x == 0) denominator += probability;
        for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
            accumulator += probability * value[(unsigned long long)row * kv_columns + kv_head * head_dim + column];
        }
        __syncthreads();
    }
    for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
        output[(unsigned long long)query_row * query_columns + head * head_dim + column] = accumulator / denominator;
    }
}

// DSpark 的长前缀已经常驻设备；短 suffix 只含本轮 noise block。按逻辑行
// 选择两个 buffer，避免每个 draft round 为每层重新复制完整 K/V。
extern "C" __global__ void block_attention_prefix_suffix_f32(
    const float* query,
    const float* prefix_key,
    const float* prefix_value,
    const float* suffix_key,
    const float* suffix_value,
    const unsigned int* visible_ranges,
    float* output,
    unsigned int query_rows,
    unsigned int prefix_rows,
    unsigned int suffix_rows,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int head_dim,
    float score_scale)
{
    const unsigned int query_row = blockIdx.x / head_count;
    const unsigned int head = blockIdx.x - query_row * head_count;
    if (query_row >= query_rows) return;
    const unsigned int kv_head = head / (head_count / kv_head_count);
    const unsigned int query_columns = head_count * head_dim;
    const unsigned int kv_columns = kv_head_count * head_dim;
    const unsigned int start = visible_ranges[query_row * 2];
    const unsigned int end = visible_ranges[query_row * 2 + 1];
    const unsigned int kv_rows = prefix_rows + suffix_rows;

    // head_dim=64 时一个 wave 的 lane c/c+32 正好复现原 64-thread 归约的
    // 第一层加法，后续 shuffle 树也与 shared reduction 顺序一致。每个 lane
    // 独立保留两个 value accumulator，删除每个历史 token 的 block barrier。
    if (head_dim == 64u && warpSize == 32u) {
        if (threadIdx.x >= warpSize) return;
        const unsigned int lane = threadIdx.x;
        const unsigned long long query_base
            = (unsigned long long)query_row * query_columns + head * 64u;
        const unsigned int value_column = kv_head * 64u + lane;
        float maximum = -3.402823466e+38F;
        for (unsigned int row = start; row < end && row < kv_rows; ++row) {
            const bool prefix = row < prefix_rows;
            const unsigned int physical_row = prefix ? row : row - prefix_rows;
            const float* key = prefix ? prefix_key : suffix_key;
            const unsigned long long key_base
                = (unsigned long long)physical_row * kv_columns + value_column;
            float dot = zllm_attention_pair_dot_rn(
                query[query_base + lane], key[key_base],
                query[query_base + lane + warpSize], key[key_base + warpSize]);
            for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
                dot += __shfl_down(dot, offset, warpSize);
            }
            dot = __shfl(dot, 0, warpSize);
            maximum = fmaxf(maximum, dot * score_scale);
        }

        float denominator = 0.0f;
        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;
        for (unsigned int row = start; row < end && row < kv_rows; ++row) {
            const bool prefix = row < prefix_rows;
            const unsigned int physical_row = prefix ? row : row - prefix_rows;
            const float* key = prefix ? prefix_key : suffix_key;
            const float* value = prefix ? prefix_value : suffix_value;
            const unsigned long long key_base
                = (unsigned long long)physical_row * kv_columns + value_column;
            float dot = zllm_attention_pair_dot_rn(
                query[query_base + lane], key[key_base],
                query[query_base + lane + warpSize], key[key_base + warpSize]);
            for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
                dot += __shfl_down(dot, offset, warpSize);
            }
            dot = __shfl(dot, 0, warpSize);
            const float probability = expf(dot * score_scale - maximum);
            denominator += probability;
            accumulator0 += probability * value[key_base];
            accumulator1 += probability * value[key_base + warpSize];
        }
        output[query_base + lane] = accumulator0 / denominator;
        output[query_base + lane + warpSize] = accumulator1 / denominator;
        return;
    }

    __shared__ float reduction[256];
    __shared__ float maximum;
    __shared__ float denominator;

    if (threadIdx.x == 0) maximum = -3.402823466e+38F;
    __syncthreads();
    for (unsigned int row = start; row < end && row < kv_rows; ++row) {
        const bool prefix = row < prefix_rows;
        const unsigned int physical_row = prefix ? row : row - prefix_rows;
        const float* key = prefix ? prefix_key : suffix_key;
        float partial = 0.0f;
        for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
            partial += query[(unsigned long long)query_row * query_columns + head * head_dim + column]
                * key[(unsigned long long)physical_row * kv_columns + kv_head * head_dim + column];
        }
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) maximum = fmaxf(maximum, reduction[0] * score_scale);
        __syncthreads();
    }

    if (threadIdx.x == 0) denominator = 0.0f;
    float accumulator = 0.0f;
    __syncthreads();
    for (unsigned int row = start; row < end && row < kv_rows; ++row) {
        const bool prefix = row < prefix_rows;
        const unsigned int physical_row = prefix ? row : row - prefix_rows;
        const float* key = prefix ? prefix_key : suffix_key;
        const float* value = prefix ? prefix_value : suffix_value;
        float partial = 0.0f;
        for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
            partial += query[(unsigned long long)query_row * query_columns + head * head_dim + column]
                * key[(unsigned long long)physical_row * kv_columns + kv_head * head_dim + column];
        }
        reduction[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        const float probability = expf(reduction[0] * score_scale - maximum);
        if (threadIdx.x == 0) denominator += probability;
        for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
            accumulator += probability * value[(unsigned long long)physical_row * kv_columns + kv_head * head_dim + column];
        }
        __syncthreads();
    }
    for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
        output[(unsigned long long)query_row * query_columns + head * head_dim + column] = accumulator / denominator;
    }
}

// N32 非因果 attention 直接把 K/V 转成 rocWMMA 原生布局。
// 两个 16-key 子块顺序消费同一个 score fragment，避免 N64 路径同时保留
// 多个 accumulator 导致 VGPR/occupancy 崩塌；常规组只需要加载/复用两次同步。
template <bool NativeKv>
__device__ __forceinline__ void zllm_full_attention_f32_128_n32(
    const unsigned short* query,
    const unsigned short* key,
    const unsigned short* value,
    float* output,
    unsigned int rows,
    unsigned int head_count,
    float score_scale,
    unsigned int attention_block,
    unsigned short* shared_key,
    unsigned short* shared_value,
    unsigned short* shared_tail_key,
    unsigned short* shared_tail_value,
    unsigned short (&shared_probability)[8][16 * 16])
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using BRowFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    typedef unsigned short U16x8 __attribute__((ext_vector_type(8)));

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int blocks_per_head = (rows + 127) / 128;
    const unsigned int head = attention_block / blocks_per_head;
    const unsigned int query_block = attention_block - head * blocks_per_head;
    const unsigned int query_base_row = query_block * 128;
    const unsigned int columns = head_count * 128;
    const unsigned int key_slot = lane & 15u;
    const unsigned int query_half = lane >> 4;
    const float score_multiplier = NativeKv
        ? score_scale * 1.4426950408889634f
        : score_scale;
    const float recenter_threshold = NativeKv ? 23.083120654223414f : 16.0f;
    const float recenter_margin = NativeKv ? 115.4156032711173f : 80.0f;

    AFragment query_fragments[8];
    #pragma unroll
    for (unsigned int feature = 0; feature < 8; ++feature) {
        load_matrix_sync(
            query_fragments[feature],
            (const bfloat16_t*)(query + (unsigned long long)(query_base_row + wave * 16) * columns + head * 128 + feature * 16),
            columns);
    }

    CFragment accumulators[8];
    #pragma unroll
    for (unsigned int feature = 0; feature < 8; ++feature) fill_fragment(accumulators[feature], 0.0f);
    CFragment maximum;
    CFragment denominator;
    #pragma unroll
    for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
        maximum[element] = -3.402823466e+38F;
        denominator[element] = 0.0f;
    }

    for (unsigned int group_base_row = 0; group_base_row < rows; group_base_row += 32) {
        if constexpr (NativeKv) {
            const unsigned int index = threadIdx.x * 8;
            const unsigned int native_row = index / 128;
            const unsigned int native_column = index - native_row * 128;
            #pragma unroll
            for (unsigned int part = 0; part < 2; ++part) {
                const unsigned int key_row = group_base_row + part * 16 + native_row;
                const U16x8 key_vector = *(const U16x8*)(key + (unsigned long long)key_row * columns + head * 128 + native_column);
                const U16x8 value_vector = *(const U16x8*)(value + (unsigned long long)key_row * columns + head * 128 + native_column);
                *(U16x8*)(shared_key + part * 16 * 128 + index) = key_vector;
                *(U16x8*)(shared_value + part * 16 * 128 + index) = value_vector;
            }
        } else {
            #pragma unroll
            for (unsigned int part = 0; part < 2; ++part) {
                const unsigned int key_base_row = group_base_row + part * 16;
                BFragment key_fragment;
                BRowFragment value_fragment;
                if (key_base_row + 16 <= rows) {
                    load_matrix_sync(key_fragment, (const bfloat16_t*)(key + (unsigned long long)key_base_row * columns + head * 128 + wave * 16), columns);
                    load_matrix_sync(value_fragment, (const bfloat16_t*)(value + (unsigned long long)key_base_row * columns + head * 128 + wave * 16), columns);
                } else {
                    const unsigned int index = threadIdx.x * 8;
                    const unsigned int tile_key = index / 128;
                    const unsigned int column = index - tile_key * 128;
                    const unsigned int key_row = key_base_row + tile_key;
                    U16x8 key_vector = {};
                    U16x8 value_vector = {};
                    if (key_row < rows) {
                        key_vector = *(const U16x8*)(key + (unsigned long long)key_row * columns + head * 128 + column);
                        value_vector = *(const U16x8*)(value + (unsigned long long)key_row * columns + head * 128 + column);
                    }
                    *(U16x8*)(shared_tail_key + index) = key_vector;
                    *(U16x8*)(shared_tail_value + index) = value_vector;
                    __syncthreads();
                    load_matrix_sync(key_fragment, (const bfloat16_t*)(shared_tail_key + wave * 16), 128);
                    load_matrix_sync(value_fragment, (const bfloat16_t*)(shared_tail_value + wave * 16), 128);
                    __syncthreads();
                }
                using BAccess = decltype(key_fragment.mAccess);
                using BRowAccess = decltype(value_fragment.mAccess);
                BAccess* native_keys = (BAccess*)shared_key;
                BRowAccess* native_values = (BRowAccess*)shared_value;
                native_keys[(part * 8 + wave) * 32 + lane] = key_fragment.mAccess;
                native_values[(part * 8 + wave) * 32 + lane] = value_fragment.mAccess;
            }
        }
        __syncthreads();

        #pragma unroll 1
        for (unsigned int part = 0; part < 2; ++part) {
            const unsigned int key_base_row = group_base_row + part * 16;
            const unsigned int key_count = key_base_row < rows ? min(16u, rows - key_base_row) : 0u;
            CFragment scores;
            fill_fragment(scores, 0.0f);
            using BAccess = decltype(((BFragment*)nullptr)->mAccess);
            const BAccess* native_keys = (const BAccess*)shared_key;
            #pragma unroll
            for (unsigned int feature = 0; feature < 8; ++feature) {
                BFragment key_fragment;
                key_fragment.mAccess = native_keys[(part * 8 + feature) * 32 + lane];
                mma_sync(scores, query_fragments[feature], key_fragment, scores);
            }

            AFragment probability_fragment;
            #pragma unroll
            for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
                const unsigned int local_query = element * 2 + query_half;
                const bool valid_query = head < head_count && query_base_row + wave * 16 + local_query < rows;
                const float scaled_score = valid_query && key_slot < key_count
                    ? scores[element] * score_multiplier
                    : -3.402823466e+38F;
                const bool near_origin = valid_query && key_slot < key_count
                    && scaled_score > maximum[element] - recenter_threshold;
                bool recenter = key_base_row == 0;
                if (key_base_row != 0) {
                    const unsigned long long recenter_mask = __ballot(near_origin);
                    recenter = ((recenter_mask >> (query_half * 16u)) & 0xffffu) != 0u;
                }
                float previous_scale = 1.0f;
                if (recenter) {
                    float tile_maximum = scaled_score;
                    #pragma unroll
                    for (unsigned int offset = 1; offset < 16; offset <<= 1) {
                        tile_maximum = fmaxf(tile_maximum, __shfl_xor(tile_maximum, offset));
                    }
                    const float new_maximum = fmaxf(maximum[element], tile_maximum + recenter_margin);
                    if (key_base_row != 0) {
                        previous_scale = __builtin_amdgcn_exp2f(
                            NativeKv
                                ? maximum[element] - new_maximum
                                : (maximum[element] - new_maximum) * 1.4426950408889634f);
                        denominator[element] *= previous_scale;
                    }
                    maximum[element] = new_maximum;
                }
                if (previous_scale != 1.0f) {
                    #pragma unroll
                    for (unsigned int feature = 0; feature < 8; ++feature) {
                        accumulators[feature][element] *= previous_scale;
                    }
                }
                float probability = 0.0f;
                if (valid_query && key_slot < key_count) {
                    probability = __builtin_amdgcn_exp2f(
                        NativeKv
                            ? scaled_score - maximum[element]
                            : (scaled_score - maximum[element]) * 1.4426950408889634f);
                }
                float tile_sum = probability;
                shared_probability[wave][local_query * 16 + key_slot] =
                    zllm_attention_f32_to_bf16(probability);
                #pragma unroll
                for (unsigned int offset = 1; offset < 16; offset <<= 1) {
                    tile_sum += __shfl_xor(tile_sum, offset);
                }
                denominator[element] += tile_sum;
            }
            __syncwarp();
            load_matrix_sync(probability_fragment, (const bfloat16_t*)shared_probability[wave], 16);

            using BRowAccess = decltype(((BRowFragment*)nullptr)->mAccess);
            const BRowAccess* native_values = (const BRowAccess*)shared_value;
            #pragma unroll
            for (unsigned int feature = 0; feature < 8; ++feature) {
                BRowFragment value_fragment;
                value_fragment.mAccess = native_values[(part * 8 + feature) * 32 + lane];
                mma_sync(accumulators[feature], probability_fragment, value_fragment, accumulators[feature]);
            }
        }
        if (group_base_row + 32 < rows) __syncthreads();
    }

    #pragma unroll
    for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
        #pragma unroll
        for (unsigned int feature = 0; feature < 8; ++feature) {
            accumulators[feature][element] /= denominator[element];
        }
    }
    #pragma unroll
    for (unsigned int feature = 0; feature < 8; ++feature) {
        #pragma unroll
        for (unsigned int element = 0; element < CFragment::num_elements; ++element) {
            const unsigned int local_query = element * 2 + query_half;
            const unsigned int query_row = query_base_row + wave * 16 + local_query;
            if (head < head_count && query_row < rows) {
                output[(unsigned long long)query_row * columns + head * 128 + feature * 16 + key_slot] =
                    accumulators[feature][element];
            }
        }
    }
}


extern "C" __global__ void full_attention_f32_128(
    const unsigned short* query,
    const unsigned short* key,
    const unsigned short* value,
    float* output,
    unsigned int rows,
    unsigned int head_count,
    unsigned int head_dim,
    float score_scale)
{
    __shared__ unsigned short shared_key[32 * 128];
    __shared__ unsigned short shared_value[32 * 128];
    __shared__ unsigned short shared_tail_key[16 * 128];
    __shared__ unsigned short shared_tail_value[16 * 128];
    __shared__ unsigned short shared_probability[8][16 * 16];
    zllm_full_attention_f32_128_n32<false>(query, key, value, output, rows, head_count, score_scale, blockIdx.x,
        shared_key, shared_value, shared_tail_key, shared_tail_value, shared_probability);
}

extern "C" __global__ void full_attention_f32_128_native_kv(
    const unsigned short* query,
    const unsigned short* key,
    const unsigned short* value,
    float* output,
    unsigned int rows,
    unsigned int head_count,
    unsigned int head_dim,
    float score_scale)
{
    __shared__ unsigned short shared_key[32 * 128];
    __shared__ unsigned short shared_value[32 * 128];
    __shared__ unsigned short shared_probability[8][16 * 16];
    zllm_full_attention_f32_128_n32<true>(query, key, value, output, rows, head_count, score_scale, blockIdx.x,
        shared_key, shared_value, shared_key, shared_value, shared_probability);
}

extern "C" __global__ void full_attention_gqa_causal_f32_128(
    const unsigned short* query, const unsigned short* key, const unsigned short* value,
    float* output, unsigned int rows, unsigned int head_count, unsigned int kv_head_count,
    unsigned int head_dim, float score_scale)
{
    __shared__ unsigned short shared_key[3][16 * 128];
    __shared__ unsigned short shared_value[3][16 * 128];
    __shared__ unsigned short shared_probability[8][16 * 16];
    zllm_full_attention_f32<128, true>(query, key, value, output, rows, head_count,
        kv_head_count, head_dim, score_scale, blockIdx.x, shared_key[0], shared_value[0], shared_probability);
}

extern "C" __global__ void full_attention_batched_f32(
    const unsigned short* query, const unsigned short* key, const unsigned short* value,
    float* output, unsigned int rows, unsigned int head_count, unsigned int head_dim, float score_scale)
{
    const unsigned int blocks_per_sample = ((rows + 127) / 128) * head_count;
    const unsigned int sample = blockIdx.x / blocks_per_sample;
    const unsigned int attention_block = blockIdx.x - sample * blocks_per_sample;
    const unsigned long long offset = (unsigned long long)sample * rows * head_count * head_dim;
    __shared__ unsigned short shared_key[16 * 128];
    __shared__ unsigned short shared_value[16 * 128];
    __shared__ unsigned short shared_probability[8][16 * 16];
    zllm_full_attention_f32<0, false>(query + offset, key + offset, value + offset, output + offset,
        rows, head_count, head_count, head_dim, score_scale, attention_block, shared_key, shared_value, shared_probability);
}

extern "C" __global__ void full_attention_batched_f32_128(
    const unsigned short* query, const unsigned short* key, const unsigned short* value,
    float* output, unsigned int rows, unsigned int head_count, unsigned int head_dim, float score_scale)
{
    const unsigned int blocks_per_sample = ((rows + 127) / 128) * head_count;
    const unsigned int sample = blockIdx.x / blocks_per_sample;
    const unsigned int attention_block = blockIdx.x - sample * blocks_per_sample;
    const unsigned long long offset = (unsigned long long)sample * rows * head_count * head_dim;
    __shared__ unsigned short shared_key[3][16 * 128];
    __shared__ unsigned short shared_value[3][16 * 128];
    __shared__ unsigned short shared_probability[8][16 * 16];
    zllm_full_attention_f32<128, false>(query + offset, key + offset, value + offset, output + offset,
        rows, head_count, head_count, head_dim, score_scale, attention_block, shared_key[0], shared_value[0], shared_probability);
}

extern "C" __global__ void full_attention_batched_f32_64(
    const unsigned short* query, const unsigned short* key, const unsigned short* value,
    float* output, unsigned int rows, unsigned int head_count, unsigned int head_dim, float score_scale)
{
    const unsigned int blocks_per_sample = ((rows + 191) / 192) * head_count;
    const unsigned int sample = blockIdx.x / blocks_per_sample;
    const unsigned int attention_block = blockIdx.x - sample * blocks_per_sample;
    const unsigned long long offset = (unsigned long long)sample * rows * head_count * head_dim;
    __shared__ unsigned short shared_key[16 * 128];
    __shared__ unsigned short shared_value[16 * 128];
    __shared__ unsigned short shared_probability[12][16 * 16];
    zllm_full_attention_f32<64, false, 192>(query + offset, key + offset, value + offset, output + offset,
        rows, head_count, head_count, head_dim, score_scale, attention_block, shared_key, shared_value, shared_probability);
}

extern "C" __global__ void concat_weight_rows_batched_f32(
    const float* input, const float* weight, float* output,
    unsigned int batch, unsigned int input_rows, unsigned int weight_rows, unsigned int columns)
{
    const unsigned int output_rows = input_rows + weight_rows;
    const unsigned long long elements = (unsigned long long)batch * output_rows * columns;
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int column = index % columns;
    const unsigned long long row = index / columns;
    const unsigned int local_row = row % output_rows;
    const unsigned int sample = row / output_rows;
    output[index] = local_row < input_rows
        ? input[((unsigned long long)sample * input_rows + local_row) * columns + column]
        : weight[(unsigned long long)(local_row - input_rows) * columns + column];
}

extern "C" __global__ void take_rows_batched_f32(
    const float* input, float* output,
    unsigned int batch, unsigned int input_rows, unsigned int output_rows, unsigned int columns)
{
    const unsigned long long elements = (unsigned long long)batch * output_rows * columns;
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int column = index % columns;
    const unsigned long long row = index / columns;
    const unsigned int local_row = row % output_rows;
    const unsigned int sample = row / output_rows;
    output[index] = input[((unsigned long long)sample * input_rows + local_row) * columns + column];
}

// GQA cache append：把 f32 K/V 行追加到常驻设备缓冲（纯 D2D 拷贝）。
extern "C" __global__ void gqa_cache_append_f32(const float* __restrict__ src, float* __restrict__ dst, unsigned long long offset, unsigned long long elements) {
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    dst[offset + index] = src[index];
}

// GQA decode 阶段 1：flat grid = q_heads * tiles；每 workgroup 处理一个 (head, tile)。
// 256 线程：先协作计算 tile 内每个 key 的点积，再做 tile 内在线 softmax，
// 最后每 dim 线程累加 softmax 加权 V，输出未归一化的 partial 与 (max, sum) 统计。
extern "C" __global__ void gqa_decode_partial_f32(
    const float* __restrict__ query, const float* __restrict__ key, const float* __restrict__ value,
    float* __restrict__ partial, float* __restrict__ stats,
    unsigned int rows, unsigned int kv_cols, unsigned int head_dim, unsigned int group, float score_scale, unsigned int tiles)
{
    __shared__ float q_shared[256];
    __shared__ float scores[128];
    __shared__ float m_shared, l_shared;
    const unsigned int head = blockIdx.x / tiles;
    const unsigned int tile_index = blockIdx.x % tiles;
    const unsigned int kv_head = head / group;
    const unsigned int tile = 128;
    const unsigned int begin = tile_index * tile;
    const unsigned int end = min(begin + tile, rows);
    const float* q_head = query + (unsigned long long)head * head_dim;
    for (unsigned int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = q_head[i];
    }
    __syncthreads();
    // 阶段1: 每 warp 负责一个 key；lane 以步长 32 取 dim 分量，连续 lane 读连续地址
    // （coalesced，对齐 llama.cpp fattn-tile 的访存形态），warp shuffle 归约点积。
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warps_per_block = blockDim.x >> 5;
    for (unsigned int k = begin + warp; k < end; k += warps_per_block) {
        const float* k_row = key + (unsigned long long)k * kv_cols + (unsigned long long)kv_head * head_dim;
        float dot = 0.0f;
        for (unsigned int d = lane; d < head_dim; d += 32) {
            dot += q_shared[d] * k_row[d];
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor(dot, offset);
        }
        if (lane == 0) {
            scores[k - begin] = dot * score_scale;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = __int_as_float(0xff800000u);
        for (unsigned int k = 0; k < end - begin; k++) {
            if (scores[k] > m) m = scores[k];
        }
        float l = 0.0f;
        for (unsigned int k = 0; k < end - begin; k++) {
            l += __expf(scores[k] - m);
        }
        m_shared = m;
        l_shared = l;
    }
    __syncthreads();
    if (threadIdx.x < head_dim) {
        const unsigned int d = threadIdx.x;
        float acc = 0.0f;
        for (unsigned int k = begin; k < end; k++) {
            acc += __expf(scores[k - begin] - m_shared) * value[(unsigned long long)k * kv_cols + (unsigned long long)kv_head * head_dim + d];
        }
        partial[((unsigned long long)head * tiles + tile_index) * head_dim + d] = acc;
    }
    if (threadIdx.x == 0) {
        const unsigned long long index = (unsigned long long)head * tiles + tile_index;
        stats[index * 2] = m_shared;
        stats[index * 2 + 1] = l_shared;
    }
}

// GQA decode 阶段 2：每 workgroup 合并一个 head 的所有 tile partial。
extern "C" __global__ void gqa_decode_merge_f32(
    const float* __restrict__ partial, const float* __restrict__ stats, float* __restrict__ output,
    unsigned int tiles, unsigned int head_dim)
{
    const unsigned int head = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (d >= head_dim) return;
    float m = __int_as_float(0xff800000u);
    for (unsigned int t = 0; t < tiles; t++) {
        const float tile_max = stats[((unsigned long long)head * tiles + t) * 2];
        if (tile_max > m) m = tile_max;
    }
    float l = 0.0f;
    float acc = 0.0f;
    for (unsigned int t = 0; t < tiles; t++) {
        const float tile_max = stats[((unsigned long long)head * tiles + t) * 2];
        const float tile_sum = stats[((unsigned long long)head * tiles + t) * 2 + 1];
        if (tile_sum <= 0.0f) continue;
        const float scale = __expf(tile_max - m);
        l += tile_sum * scale;
        acc += partial[((unsigned long long)head * tiles + t) * head_dim + d] * scale;
    }
    output[(unsigned long long)head * head_dim + d] = l > 0.0f ? acc / l : 0.0f;
}
