//! GQA/MLA attention kernel:prefill 分派、decode、MPS 链路与 KV 量化。

/// 本模块的 Metal shader(decode/prefill/softmax/kv-quantize 全族)。
/// kernel 之间互不调用,共享 helper 均在全局 preamble。
pub const SHADERS: &str = r#"
kernel void gqa_prefill_attention_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &query_rows [[buffer(4)]],
    constant uint &kv_rows [[buffer(5)]],
    constant uint &query_position [[buffer(6)]],
    constant uint &head_count [[buffer(7)]],
    constant uint &kv_head_count [[buffer(8)]],
    constant uint &head_dim [[buffer(9)]],
    constant float &score_scale [[buffer(10)]],
    constant uint &sliding_window [[buffer(11)]],
    constant uint &kv_start [[buffer(12)]],
    constant uint &kv_capacity [[buffer(13)]],
    device const uint *visible_ends [[buffer(14)]],
    constant uint &use_visible_ends [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (group >= query_rows * head_count) {
        return;
    }

    threadgroup float reduction[1024];
    threadgroup float control[4];
    const uint query_row = group / head_count;
    const uint query_head = group % head_count;
    const uint kv_head = query_head / (head_count / kv_head_count);
    const ulong query_base = ((ulong)query_row * head_count + query_head) * head_dim;
    float accumulated = 0.0f;

    const uint causal_rows = min(kv_rows, query_position + query_row + 1);
    const uint visible_rows = use_visible_ends == 0 ? causal_rows : min(kv_rows, visible_ends[query_row]);
    const uint window_start = sliding_window == 0 || causal_rows <= sliding_window ? 0 : causal_rows - sliding_window;
    const uint first_visible = max(kv_start, window_start);
    for (uint token = first_visible; token < visible_rows; ++token) {
        const uint slot = kv_capacity == 0 ? token : token % kv_capacity;
        const ulong key_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        reduction[lane] = lane < head_dim ? float(query[query_base + lane]) * float(key[key_base + lane]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                reduction[lane] += reduction[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            const float score = reduction[0] * score_scale;
            if (token == first_visible) {
                control[0] = 0.0f;
                control[1] = 1.0f;
                control[2] = 1.0f;
                control[3] = score;
            } else if (score > control[3]) {
                const float rescale = exp(control[3] - score);
                control[0] = rescale;
                control[1] = 1.0f;
                control[2] = control[2] * rescale + 1.0f;
                control[3] = score;
            } else {
                const float weight = exp(score - control[3]);
                control[0] = 1.0f;
                control[1] = weight;
                control[2] += weight;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < head_dim) {
            const ulong value_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
            accumulated = accumulated * control[0] + control[1] * float(value[value_base + lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane < head_dim) {
        output[query_base + lane] = half(accumulated / control[2]);
    }
}
kernel void gqa_kv_quantize_q8(
    device const ushort *key [[buffer(0)]],
    device const ushort *value [[buffer(1)]],
    device char *key_codes [[buffer(2)]],
    device half *key_scales [[buffer(3)]],
    device char *value_codes [[buffer(4)]],
    device half *value_scales [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &kv_head_count [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &group_size [[buffer(9)]],
    constant uint &groups_per_head [[buffer(10)]],
    constant uint &bf16 [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    const uint groups_per_tensor = rows * kv_head_count * groups_per_head;
    if (gid >= groups_per_tensor * 2 || group_size == 0) return;
    const bool is_value = gid >= groups_per_tensor;
    const uint group = gid - (is_value ? groups_per_tensor : 0);
    const uint row_head = group / groups_per_head;
    const uint group_in_head = group - row_head * groups_per_head;
    const ulong base = ulong(row_head) * head_dim + group_in_head * group_size;
    device const ushort *input = is_value ? value : key;
    device char *codes = is_value ? value_codes : key_codes;
    device half *scales = is_value ? value_scales : key_scales;
    float maximum = 0.0f;
    for (uint index = 0; index < group_size; ++index) {
        const ushort bits = input[base + index];
        const float element = bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits));
        maximum = max(maximum, abs(element));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    scales[group] = half(scale);
    const float inverse_scale = 1.0f / scale;
    for (uint index = 0; index < group_size; ++index) {
        const ushort bits = input[base + index];
        const float element = bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits));
        codes[base + index] = char(clamp(rint(element * inverse_scale), -127.0f, 127.0f));
    }
}
kernel void gqa_kv_dequantize_q8(
    device const char *key_codes [[buffer(0)]],
    device const half *key_scales [[buffer(1)]],
    device const char *value_codes [[buffer(2)]],
    device const half *value_scales [[buffer(3)]],
    device ushort *key [[buffer(4)]],
    device ushort *value [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &kv_head_count [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &group_size [[buffer(9)]],
    constant uint &groups_per_head [[buffer(10)]],
    constant uint &bf16 [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    const uint elements = rows * kv_head_count * head_dim;
    if (gid >= elements || group_size == 0) return;
    const uint row_head = gid / head_dim;
    const uint dimension = gid - row_head * head_dim;
    const uint scale = row_head * groups_per_head + dimension / group_size;
    const float key_value = float(key_codes[gid]) * float(key_scales[scale]);
    const float value_value = float(value_codes[gid]) * float(value_scales[scale]);
    key[gid] = bf16 != 0 ? zllm_f32_to_bf16(key_value) : as_type<ushort>(finite_f16(key_value));
    value[gid] = bf16 != 0 ? zllm_f32_to_bf16(value_value) : as_type<ushort>(finite_f16(value_value));
}
kernel void gqa_decode_scores_gqa_bf16(
    device const ushort *query [[buffer(0)]],
    device const ushort *key [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant uint &source_rows [[buffer(3)]],
    constant uint &head_count [[buffer(4)]],
    constant uint &kv_head_count [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant float &score_scale [[buffer(7)]],
    constant uint &first_visible [[buffer(8)]],
    constant uint &kv_capacity [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint token = group.x;
    const uint kv_head = group.y;
    if (token >= source_rows || kv_head >= kv_head_count || head_dim > 512) return;

    threadgroup float key_tile[512];
    const uint source = first_visible + token;
    const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
    const ulong key_base = (ulong(slot) * kv_head_count + kv_head) * head_dim;
    for (uint dim = tid; dim < head_dim; dim += 128) {
        key_tile[dim] = zllm_bf16_to_f32(key[key_base + dim]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint heads_per_kv = head_count / kv_head_count;
    if (simdgroup >= heads_per_kv) return;
    const uint query_head = kv_head * heads_per_kv + simdgroup;
    const ulong query_base = ulong(query_head) * head_dim;
    float partial = 0.0f;
    for (uint dim = lane; dim < head_dim; dim += 32) {
        partial += zllm_bf16_to_f32(query[query_base + dim]) * key_tile[dim];
    }
    const float score = simd_sum(partial) * score_scale;
    if (lane == 0) scores[ulong(query_head) * source_rows + token] = score;
}
kernel void gqa_decode_weighted_value_tiled_bf16(
    device const float *scores [[buffer(0)]],
    device const ushort *value [[buffer(1)]],
    device ushort *output [[buffer(2)]],
    constant uint &source_rows [[buffer(3)]],
    constant uint &head_count [[buffer(4)]],
    constant uint &kv_head_count [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &first_visible [[buffer(7)]],
    constant uint &kv_capacity [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint dimension_tiles = (head_dim + 31) / 32;
    const uint query_head = group / dimension_tiles;
    const uint dimension = (group % dimension_tiles) * 32 + lane;
    if (query_head >= head_count || dimension >= head_dim) return;

    const uint heads_per_kv = head_count / kv_head_count;
    const uint kv_head = query_head / heads_per_kv;
    const ulong score_base = ulong(query_head) * source_rows;
    float accumulated = 0.0f;
    for (uint token = 0; token < source_rows; ++token) {
        const uint source = first_visible + token;
        const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
        const ulong value_index = (ulong(slot) * kv_head_count + kv_head) * head_dim + dimension;
        accumulated += scores[score_base + token] * zllm_bf16_to_f32(value[value_index]);
    }
    output[ulong(query_head) * head_dim + dimension] = zllm_f32_to_bf16(accumulated);
}
kernel void gqa_decode_split_kv_bf16_vectorized(
    device const ushort *query [[buffer(0)]],
    device const ushort *key [[buffer(1)]],
    device const ushort *value [[buffer(2)]],
    device float *statistics [[buffer(3)]],
    device float *partial_values [[buffer(4)]],
    constant uint &source_rows [[buffer(5)]],
    constant uint &head_count [[buffer(6)]],
    constant uint &kv_head_count [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &block_tokens [[buffer(9)]],
    constant uint &block_count [[buffer(10)]],
    constant float &score_scale [[buffer(11)]],
    constant uint &bf16 [[buffer(12)]],
    constant uint &first_visible [[buffer(13)]],
    constant uint &kv_capacity [[buffer(14)]],
    device const half *key_scales [[buffer(15)]],
    device const half *value_scales [[buffer(16)]],
    constant uint &group_size [[buffer(17)]],
    constant uint &groups_per_head [[buffer(18)]],
    constant uint &q8 [[buffer(19)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint heads_per_group = 4;
    if (bf16 == 0 || kv_head_count == 0 || head_count % kv_head_count != 0
        || head_dim > 512 || block_tokens == 0 || block_tokens > 256) return;
    const uint heads_per_kv = head_count / kv_head_count;
    const uint head_groups_per_kv = (heads_per_kv + heads_per_group - 1) / heads_per_group;
    const uint kv_head = group.y / head_groups_per_kv;
    const uint head_group = group.y % head_groups_per_kv;
    const uint first_head_offset = head_group * heads_per_group;
    if (group.x >= block_count || kv_head >= kv_head_count || first_head_offset >= heads_per_kv) return;
    const uint active_heads = min(heads_per_group, heads_per_kv - first_head_offset);
    const uint first_query_head = kv_head * heads_per_kv + first_head_offset;
    const uint row_begin = group.x * block_tokens;
    const uint rows = min(block_tokens, source_rows - row_begin);

    threadgroup float query_tile[2048];
    threadgroup float weights[1024];
    threadgroup half value_scale_tile[1024];
    const uint query_elements = active_heads * head_dim;
    for (uint index = tid; index < query_elements; index += 256) {
        query_tile[index] = zllm_bf16_to_f32(query[ulong(first_query_head) * head_dim + index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < rows) {
        const uint source = first_visible + row_begin + tid;
        const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
        const ulong key_base = (ulong(slot) * kv_head_count + kv_head) * head_dim;
        float scores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (q8 != 0) {
            const device char *codes = reinterpret_cast<device const char *>(key);
            const ulong scale_base = (ulong(slot) * kv_head_count + kv_head) * groups_per_head;
            for (uint group_index = 0; group_index < groups_per_head; ++group_index) {
                const float scale = float(key_scales[scale_base + group_index]);
                uint dimension = group_index * group_size;
                const uint group_end = min(dimension + group_size, head_dim);
                for (; dimension + 3 < group_end; dimension += 4) {
                    const char4 packed = *((device const char4 *)(codes + key_base + dimension));
                    const float4 key_values = float4(packed) * scale;
                    for (uint head = 0; head < active_heads; ++head) {
                        const threadgroup float4 *query_values = (threadgroup float4 *)(query_tile + head * head_dim + dimension);
                        scores[head] += dot(*query_values, key_values);
                    }
                }
                for (; dimension < group_end; ++dimension) {
                    const float key_value = float(codes[key_base + dimension]) * scale;
                    for (uint head = 0; head < active_heads; ++head) {
                        scores[head] += float(query_tile[head * head_dim + dimension]) * key_value;
                    }
                }
            }
        } else {
            uint dimension = 0;
            for (; dimension + 1 < head_dim; dimension += 2) {
                const ushort2 packed = *((device const ushort2 *)(key + key_base + dimension));
                const float2 key_pair = as_type<float2>(uint2(packed) << 16);
                for (uint head = 0; head < active_heads; ++head) {
                    const threadgroup float2 *query_pair = (threadgroup float2 *)(query_tile + head * head_dim + dimension);
                    const float2 q = *query_pair;
                    scores[head] += q.x * key_pair.x + q.y * key_pair.y;
                }
            }
            if (dimension < head_dim) {
                const float key_value = zllm_bf16_to_f32(key[key_base + dimension]);
                for (uint head = 0; head < active_heads; ++head) {
                    scores[head] += float(query_tile[head * head_dim + dimension]) * key_value;
                }
            }
        }
        for (uint head = 0; head < active_heads; ++head) {
            weights[head * block_tokens + tid] = scores[head] * score_scale;
        }
    }
    for (uint head = 0; head < active_heads; ++head) {
        if (tid >= rows && tid < block_tokens) weights[head * block_tokens + tid] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < active_heads; ++head) {
        // softmax 由一个 SIMD group 完成，避免多个 SIMD group 复用同一归约区时
        // 偶发读取到尚未稳定的部分结果；每 lane 最多处理 8 个 token。
        if (tid < 32) {
            float local_maximum = -INFINITY;
            for (uint token = tid; token < rows; token += 32) {
                local_maximum = max(local_maximum, weights[head * block_tokens + token]);
            }
            const float maximum = simd_max(local_maximum);
            float local_denominator = 0.0f;
            for (uint token = tid; token < rows; token += 32) {
                const float weight = exp(weights[head * block_tokens + token] - maximum);
                weights[head * block_tokens + token] = weight;
                local_denominator += weight;
            }
            const float denominator = simd_sum(local_denominator);
            if (tid == 0) {
                const uint query_head = first_query_head + head;
                const ulong statistic = (ulong(query_head) * block_count + group.x) * 2;
                statistics[statistic] = maximum;
                statistics[statistic + 1] = denominator;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q8 != 0) {
        const uint scale_count = rows * groups_per_head;
        for (uint index = tid; index < scale_count; index += 256) {
            const uint row = index / groups_per_head;
            const uint group_index = index - row * groups_per_head;
            const uint source = first_visible + row_begin + row;
            const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
            value_scale_tile[index] = value_scales[(ulong(slot) * kv_head_count + kv_head) * groups_per_head + group_index];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (q8 != 0) {
        const device char *codes = reinterpret_cast<device const char *>(value);
        for (uint dimension = tid * 2; dimension < head_dim; dimension += 512) {
            float2 accumulated[4] = {float2(0.0f), float2(0.0f), float2(0.0f), float2(0.0f)};
            for (uint row = 0; row < rows; ++row) {
                const uint source = first_visible + row_begin + row;
                const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
                const ulong value_index = (ulong(slot) * kv_head_count + kv_head) * head_dim + dimension;
                const float first_scale = float(value_scale_tile[row * groups_per_head + dimension / group_size]);
                float2 elements;
                if (dimension + 1 < head_dim) {
                    const char2 packed = *((device const char2 *)(codes + value_index));
                    const float second_scale = float(value_scale_tile[row * groups_per_head + (dimension + 1) / group_size]);
                    elements = float2(float(packed.x) * first_scale, float(packed.y) * second_scale);
                } else {
                    elements = float2(float(codes[value_index]) * first_scale, 0.0f);
                }
                for (uint head = 0; head < active_heads; ++head) {
                    accumulated[head] += weights[head * block_tokens + row] * elements;
                }
            }
            for (uint head = 0; head < active_heads; ++head) {
                const uint query_head = first_query_head + head;
                const ulong partial = (ulong(query_head) * block_count + group.x) * head_dim + dimension;
                partial_values[partial] = accumulated[head].x;
                if (dimension + 1 < head_dim) partial_values[partial + 1] = accumulated[head].y;
            }
        }
    } else {
        for (uint dimension = tid; dimension < head_dim; dimension += 256) {
            float accumulated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint row = 0; row < rows; ++row) {
                const uint source = first_visible + row_begin + row;
                const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
                const ulong value_index = (ulong(slot) * kv_head_count + kv_head) * head_dim + dimension;
                const float element = zllm_bf16_to_f32(value[value_index]);
                for (uint head = 0; head < active_heads; ++head) {
                    accumulated[head] += weights[head * block_tokens + row] * element;
                }
            }
            for (uint head = 0; head < active_heads; ++head) {
                const uint query_head = first_query_head + head;
                const ulong partial = (ulong(query_head) * block_count + group.x) * head_dim + dimension;
                partial_values[partial] = accumulated[head];
            }
        }
    }
}
kernel void gqa_decode_flash_q8_grouped(
    device const ushort *query [[buffer(0)]],
    device const ushort *key [[buffer(1)]],
    device const ushort *value [[buffer(2)]],
    device float *statistics [[buffer(3)]],
    device float *partial_values [[buffer(4)]],
    constant uint &source_rows [[buffer(5)]],
    constant uint &head_count [[buffer(6)]],
    constant uint &kv_head_count [[buffer(7)]],
    constant uint &head_dim_argument [[buffer(8)]],
    constant uint &block_tokens [[buffer(9)]],
    constant uint &block_count [[buffer(10)]],
    constant float &score_scale [[buffer(11)]],
    constant uint &bf16 [[buffer(12)]],
    constant uint &first_visible [[buffer(13)]],
    constant uint &kv_capacity [[buffer(14)]],
    device const half *key_scales [[buffer(15)]],
    device const half *value_scales [[buffer(16)]],
    constant uint &group_size_argument [[buffer(17)]],
    constant uint &groups_per_head_argument [[buffer(18)]],
    constant uint &q8 [[buffer(19)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]])
{
    constexpr uint heads_per_group = 8;
    constexpr uint tile_tokens = 256;
    constexpr uint threads = 256;
    const uint head_dim = zllm_fc_u32_0;
    const uint group_size = zllm_fc_u32_1;
    const uint groups_per_head = zllm_fc_u32_2;
    const uint heads_per_kv = zllm_fc_u32_3;
    (void)block_tokens;
    if (q8 == 0 || kv_head_count == 0 || head_count % kv_head_count != 0
        || group.x >= block_count || head_dim > threads || group_size == 0
        || groups_per_head > 8 || head_dim_argument != head_dim
        || group_size_argument != group_size || groups_per_head_argument != groups_per_head
        || head_count / kv_head_count != heads_per_kv) return;
    const uint head_groups_per_kv = (heads_per_kv + heads_per_group - 1) / heads_per_group;
    const uint kv_head = group.y / head_groups_per_kv;
    const uint head_group = group.y - kv_head * head_groups_per_kv;
    const uint first_head_offset = head_group * heads_per_group;
    if (kv_head >= kv_head_count || first_head_offset >= heads_per_kv) return;
    const uint active_heads = min(heads_per_group, heads_per_kv - first_head_offset);
    const uint first_query_head = kv_head * heads_per_kv + first_head_offset;
    const uint partition = group.x;
    const uint partition_begin = uint((ulong(source_rows) * partition) / block_count);
    const uint partition_end = uint((ulong(source_rows) * (partition + 1)) / block_count);

    threadgroup half query_tile[heads_per_group * threads];
    threadgroup float weights[heads_per_group * tile_tokens];
    threadgroup float reduction[threads];
    threadgroup half key_scale_tile[tile_tokens * 8];
    threadgroup half value_scale_tile[tile_tokens * 8];
    threadgroup float state[heads_per_group * 4];
    const uint query_elements = active_heads * head_dim;
    for (uint index = thread_index; index < query_elements; index += threads) {
        const ushort bits = query[ulong(first_query_head) * head_dim + index];
        query_tile[index] = half(bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits)));
    }
    if (thread_index < active_heads) {
        state[thread_index * 4] = -INFINITY;
        state[thread_index * 4 + 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 accumulated[heads_per_group] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint tile_begin = partition_begin; tile_begin < partition_end; tile_begin += tile_tokens) {
        const uint rows = min(tile_tokens, partition_end - tile_begin);
        const uint scale_count = rows * groups_per_head;
        for (uint index = thread_index; index < scale_count; index += threads) {
            const uint token = index / groups_per_head;
            const uint quant_group = index - token * groups_per_head;
            const uint source = first_visible + tile_begin + token;
            const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
            const ulong scale = (ulong(slot) * kv_head_count + kv_head) * groups_per_head + quant_group;
            key_scale_tile[index] = key_scales[scale];
            value_scale_tile[index] = value_scales[scale];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint token = simd_group; token < rows; token += simd_groups) {
            const uint source = first_visible + tile_begin + token;
            const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
            const ulong key_base = (ulong(slot) * kv_head_count + kv_head) * head_dim;
            const device char *codes = reinterpret_cast<device const char *>(key);
            float dot_products[heads_per_group] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint dimension = simd_lane * 8; dimension < head_dim; dimension += 256) {
                const uint scale = token * groups_per_head + dimension / group_size;
                const char4 first_key = *((device const char4 *)(codes + key_base + dimension));
                const char4 second_key = *((device const char4 *)(codes + key_base + dimension + 4));
                const float quant_scale = float(key_scale_tile[scale]);
                for (uint head = 0; head < active_heads; ++head) {
                    const threadgroup half4 *query =
                        (threadgroup half4 *)(query_tile + head * head_dim + dimension);
                    dot_products[head] += (dot(float4(query[0]), float4(first_key))
                        + dot(float4(query[1]), float4(second_key))) * quant_scale;
                }
            }
            for (uint head = 0; head < active_heads; ++head) {
                const float score = simd_sum(dot_products[head]) * score_scale;
                if (simd_lane == 0) weights[head * tile_tokens + token] = score;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint head = 0; head < active_heads; ++head) {
            const uint weight_base = head * tile_tokens;
            const float local_score = thread_index < rows ? weights[weight_base + thread_index] : -INFINITY;
            const float simd_maximum = simd_max(local_score);
            if (simd_lane == 0) reduction[simd_group] = simd_maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                const float group_maximum = simd_lane < simd_groups ? reduction[simd_lane] : -INFINITY;
                const float tile_maximum = simd_max(group_maximum);
                if (simd_lane == 0) reduction[0] = tile_maximum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float tile_maximum = reduction[0];

            float local_denominator = 0.0f;
            if (thread_index < rows) {
                const float weight = exp(weights[weight_base + thread_index] - tile_maximum);
                weights[weight_base + thread_index] = weight;
                local_denominator = weight;
            }
            local_denominator = simd_sum(local_denominator);
            if (simd_lane == 0) reduction[simd_group] = local_denominator;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                const float group_denominator = simd_lane < simd_groups ? reduction[simd_lane] : 0.0f;
                const float tile_denominator = simd_sum(group_denominator);
                if (simd_lane == 0) {
                    const uint state_base = head * 4;
                    const float previous_maximum = state[state_base];
                    const float merged_maximum = max(previous_maximum, tile_maximum);
                    const float previous_scale = state[state_base + 1] == 0.0f ? 0.0f : exp(previous_maximum - merged_maximum);
                    const float tile_scale = exp(tile_maximum - merged_maximum);
                    state[state_base] = merged_maximum;
                    state[state_base + 1] = state[state_base + 1] * previous_scale + tile_denominator * tile_scale;
                    state[state_base + 2] = previous_scale;
                    state[state_base + 3] = tile_scale;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const uint value_dimension = thread_index * 4;
        if (value_dimension < head_dim) {
            const device char *codes = reinterpret_cast<device const char *>(value);
            const uint quant_group = value_dimension / group_size;
            float4 tile_accumulated[heads_per_group] = {
                float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)
            };
            for (uint token = 0; token < rows; ++token) {
                const uint source = first_visible + tile_begin + token;
                const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
                const ulong value_base = (ulong(slot) * kv_head_count + kv_head) * head_dim;
                const float4 value_element = float4(*((device const char4 *)(codes + value_base + value_dimension)))
                    * float(value_scale_tile[token * groups_per_head + quant_group]);
                for (uint head = 0; head < active_heads; ++head) {
                    tile_accumulated[head] += weights[head * tile_tokens + token] * value_element;
                }
            }
            for (uint head = 0; head < active_heads; ++head) {
                const uint state_base = head * 4;
                accumulated[head] = accumulated[head] * state[state_base + 2]
                    + tile_accumulated[head] * state[state_base + 3];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint head = 0; head < active_heads; ++head) {
        const uint query_head = first_query_head + head;
        const ulong statistic = (ulong(query_head) * block_count + partition) * 2;
        const ulong partial = (ulong(query_head) * block_count + partition) * head_dim;
        if (thread_index == 0) {
            statistics[statistic] = state[head * 4];
            statistics[statistic + 1] = state[head * 4 + 1];
        }
        const uint dimension = thread_index * 4;
        if (dimension < head_dim) {
            *((device float4 *)(partial_values + partial + dimension)) = accumulated[head];
        }
    }
}
kernel void gqa_decode_split_kv(
    device const ushort *query [[buffer(0)]],
    device const ushort *key [[buffer(1)]],
    device const ushort *value [[buffer(2)]],
    device float *statistics [[buffer(3)]],
    device float *partial_values [[buffer(4)]],
    constant uint &source_rows [[buffer(5)]],
    constant uint &head_count [[buffer(6)]],
    constant uint &kv_head_count [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &block_tokens [[buffer(9)]],
    constant uint &block_count [[buffer(10)]],
    constant float &score_scale [[buffer(11)]],
    constant uint &bf16 [[buffer(12)]],
    constant uint &first_visible [[buffer(13)]],
    constant uint &kv_capacity [[buffer(14)]],
    device const half *key_scales [[buffer(15)]],
    device const half *value_scales [[buffer(16)]],
    constant uint &group_size [[buffer(17)]],
    constant uint &groups_per_head [[buffer(18)]],
    constant uint &q8 [[buffer(19)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]])
{
    constexpr uint heads_per_group = 4;
    constexpr uint max_head_dim = 512;
    constexpr uint max_block_tokens = 256;
    if (kv_head_count == 0 || head_count % kv_head_count != 0) return;
    const uint heads_per_kv = head_count / kv_head_count;
    if (heads_per_kv == 0) return;
    const uint block = group.x;
    const uint head_groups_per_kv = (heads_per_kv + heads_per_group - 1) / heads_per_group;
    const uint kv_head = group.y / head_groups_per_kv;
    const uint head_group = group.y - kv_head * head_groups_per_kv;
    const uint first_head_offset = head_group * heads_per_group;
    const uint active_heads = min(heads_per_group, heads_per_kv - first_head_offset);
    if (block >= block_count || kv_head >= kv_head_count || head_dim > max_head_dim
        || block_tokens == 0 || block_tokens > max_block_tokens) return;

    const uint row_begin = block * block_tokens;
    const uint rows = min(block_tokens, source_rows - row_begin);
    // Metal function constant 不能用于数组长度；容量对应上面的通用 kernel 上限。
    threadgroup half query_tile[2048];
    threadgroup float weights[1024];
    threadgroup float reduction[256];
    threadgroup half key_scale_tile[2048];
    threadgroup half value_scale_tile[2048];

    const uint query_elements = active_heads * head_dim;
    const uint first_query_head = kv_head * heads_per_kv + first_head_offset;
    for (uint index = thread_index; index < query_elements; index += max_block_tokens) {
        const ushort bits = query[ulong(first_query_head) * head_dim + index];
        query_tile[index] = half(bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits)));
    }
    // Ornith Q8G64 每个 block 只含少量 scale，协作加载后供 K/V 两段复用。
    const bool tiled_q8_scales = q8 != 0 && groups_per_head <= 8;
    if (tiled_q8_scales) {
        const uint scale_count = rows * groups_per_head;
        for (uint index = thread_index; index < scale_count; index += max_block_tokens) {
            const uint token = index / groups_per_head;
            const uint quant_group = index - token * groups_per_head;
            const uint source = first_visible + row_begin + token;
            const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
            const ulong scale = (ulong(slot) * kv_head_count + kv_head) * groups_per_head + quant_group;
            key_scale_tile[index] = key_scales[scale];
            value_scale_tile[index] = value_scales[scale];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 每个 SIMD group 处理一个 token，并在寄存器中同时完成全部 query heads。
    for (uint token = simd_group; token < rows; token += simd_groups) {
        float dot_products[heads_per_group];
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            dot_products[query_head] = 0.0f;
        }
        const uint source = first_visible + row_begin + token;
        const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
        const ulong key_base = ((ulong(slot) * kv_head_count + kv_head) * head_dim);
        for (uint dimension = simd_lane; dimension < head_dim; dimension += 32) {
            float key_value;
            if (q8 != 0) {
                const device char *codes = reinterpret_cast<device const char *>(key);
                const ulong scale = (ulong(slot) * kv_head_count + kv_head) * groups_per_head + dimension / group_size;
                const half quant_scale = tiled_q8_scales
                    ? key_scale_tile[token * groups_per_head + dimension / group_size]
                    : key_scales[scale];
                key_value = float(codes[key_base + dimension]) * float(quant_scale);
            } else {
                const ushort bits = key[key_base + dimension];
                key_value = bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits));
            }
            for (uint query_head = 0; query_head < active_heads; ++query_head) {
                dot_products[query_head] += float(query_tile[query_head * head_dim + dimension]) * key_value;
            }
        }
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            const float score = simd_sum(dot_products[query_head]) * score_scale;
            if (simd_lane == 0) weights[query_head * max_block_tokens + token] = score;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        const uint weight_base = query_head * max_block_tokens;
        const float local_score = thread_index < rows ? weights[weight_base + thread_index] : -INFINITY;
        const float simd_maximum = simd_max(local_score);
        if (simd_lane == 0) reduction[simd_group] = simd_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            const float group_maximum = simd_lane < simd_groups ? reduction[simd_lane] : -INFINITY;
            const float block_maximum = simd_max(group_maximum);
            if (simd_lane == 0) reduction[0] = block_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float block_maximum = reduction[0];
        const float weight = thread_index < rows
            ? exp(weights[weight_base + thread_index] - block_maximum)
            : 0.0f;
        if (thread_index < rows) weights[weight_base + thread_index] = weight;
        const float simd_denominator = simd_sum(weight);
        // 所有 SIMD group 读完广播的 maximum 后，group 0 才能复用 reduction[0]。
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) reduction[simd_group] = simd_denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            const float group_denominator = simd_lane < simd_groups ? reduction[simd_lane] : 0.0f;
            const float block_denominator = simd_sum(group_denominator);
            if (simd_lane == 0) reduction[0] = block_denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index == 0) {
            const uint global_head = first_query_head + query_head;
            const ulong statistic = (ulong(global_head) * block_count + block) * 2;
            statistics[statistic] = block_maximum;
            statistics[statistic + 1] = reduction[0];
        }
    }

    for (uint dimension = thread_index; dimension < head_dim; dimension += max_block_tokens) {
        float accumulated[heads_per_group];
        for (uint query_head = 0; query_head < active_heads; ++query_head) accumulated[query_head] = 0.0f;
        for (uint token = 0; token < rows; ++token) {
            const uint source = first_visible + row_begin + token;
            const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
            const ulong value_index = ((ulong(slot) * kv_head_count + kv_head) * head_dim) + dimension;
            float value_element;
            if (q8 != 0) {
                const device char *codes = reinterpret_cast<device const char *>(value);
                const ulong scale = (ulong(slot) * kv_head_count + kv_head) * groups_per_head + dimension / group_size;
                const half quant_scale = tiled_q8_scales
                    ? value_scale_tile[token * groups_per_head + dimension / group_size]
                    : value_scales[scale];
                value_element = float(codes[value_index]) * float(quant_scale);
            } else {
                const ushort bits = value[value_index];
                value_element = bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits));
            }
            for (uint query_head = 0; query_head < active_heads; ++query_head) {
                accumulated[query_head] += weights[query_head * max_block_tokens + token] * value_element;
            }
        }
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            const uint global_head = first_query_head + query_head;
            const ulong partial = (ulong(global_head) * block_count + block) * head_dim + dimension;
            partial_values[partial] = accumulated[query_head];
        }
    }
}
kernel void gqa_decode_split_kv_merge(
    device const float *statistics [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device ushort *output [[buffer(2)]],
    constant uint &block_count [[buffer(3)]],
    constant uint &head_count [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant uint &bf16 [[buffer(6)]],
    uint query_head [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    if (query_head >= head_count) return;
    threadgroup float reduction[2];
    if (thread_index < 32) {
        float local_maximum = -INFINITY;
        for (uint block = thread_index; block < block_count; block += 32) {
            local_maximum = max(local_maximum, statistics[(ulong(query_head) * block_count + block) * 2]);
        }
        const float maximum = simd_max(local_maximum);
        if (thread_index == 0) reduction[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = reduction[0];

    if (thread_index < 32) {
        float local_denominator = 0.0f;
        for (uint block = thread_index; block < block_count; block += 32) {
            const ulong statistic = (ulong(query_head) * block_count + block) * 2;
            local_denominator += statistics[statistic + 1] * exp(statistics[statistic] - maximum);
        }
        const float denominator = simd_sum(local_denominator);
        if (thread_index == 0) reduction[1] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denominator = reduction[1];

    for (uint dimension = thread_index; dimension < head_dim; dimension += width) {
        float accumulated = 0.0f;
        for (uint block = 0; block < block_count; ++block) {
            const ulong statistic = (ulong(query_head) * block_count + block) * 2;
            const ulong partial = (ulong(query_head) * block_count + block) * head_dim + dimension;
            accumulated += partial_values[partial] * exp(statistics[statistic] - maximum);
        }
        const float result = accumulated / denominator;
        output[ulong(query_head) * head_dim + dimension] = bf16 != 0
            ? zllm_f32_to_bf16(result)
            : as_type<ushort>(finite_f16(result));
    }
}
// 小 KV decode 直通:source_rows <= 256 时 split+merge 的临时 buffer 分配与第二个
// encoder 是纯开销(实测 24 层 ~55µs/层/token)。单 kernel 内完成 online softmax +
// 加权 V,数学上与 block_count=1 的 split+merge 逐位一致,无临时 buffer。
kernel void gqa_decode_direct_q8(
    device const ushort *query [[buffer(0)]],
    device const char *key_codes [[buffer(1)]],
    device const char *value_codes [[buffer(2)]],
    device ushort *output [[buffer(3)]],
    device const half *key_scales [[buffer(4)]],
    device const half *value_scales [[buffer(5)]],
    constant uint &source_rows [[buffer(6)]],
    constant uint &head_count [[buffer(7)]],
    constant uint &kv_head_count [[buffer(8)]],
    constant uint &head_dim [[buffer(9)]],
    constant float &score_scale [[buffer(10)]],
    constant uint &bf16 [[buffer(11)]],
    constant uint &first_visible [[buffer(12)]],
    constant uint &kv_capacity [[buffer(13)]],
    constant uint &group_size [[buffer(14)]],
    constant uint &groups_per_head [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]])
{
    constexpr uint heads_per_group = 4;
    constexpr uint max_rows = 256;
    constexpr uint max_head_dim = 128;
    if (kv_head_count == 0 || head_count % kv_head_count != 0) return;
    const uint heads_per_kv = head_count / kv_head_count;
    if (heads_per_kv == 0) return;
    const uint head_groups_per_kv = (heads_per_kv + heads_per_group - 1) / heads_per_group;
    const uint kv_head = group / head_groups_per_kv;
    const uint head_group = group - kv_head * head_groups_per_kv;
    const uint first_head_offset = head_group * heads_per_group;
    const uint active_heads = min(heads_per_group, heads_per_kv - first_head_offset);
    if (kv_head >= kv_head_count || source_rows > max_rows || head_dim > max_head_dim || head_dim % 32 != 0) return;

    threadgroup half query_tile[heads_per_group * max_head_dim];
    threadgroup float weights[heads_per_group * max_rows];
    threadgroup float reduction[256];
    threadgroup float denominators[heads_per_group];
    // 第三阶段按 SIMD group 切 token 的组间 partial:[simd_group][head][dimension]。
    // 8 组 × 4 头 × 128 维 = 16KB;维度串行循环在只有 4 个 threadgroup 时
    // 是纯访存延迟链(~50µs),切开后缩短 8 倍。
    threadgroup float partials[8 * heads_per_group * max_head_dim];

    const uint query_elements = active_heads * head_dim;
    const uint first_query_head = kv_head * heads_per_kv + first_head_offset;
    for (uint index = thread_index; index < query_elements; index += 256) {
        const ushort bits = query[ulong(first_query_head) * head_dim + index];
        query_tile[index] = half(bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 每个 SIMD group 负责按 stride 分布的 token,寄存器内同时算全部 query heads。
    for (uint token = simd_group; token < source_rows; token += simd_groups) {
        float dot_products[heads_per_group];
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            dot_products[query_head] = 0.0f;
        }
        const uint source = first_visible + token;
        const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
        const ulong key_base = ((ulong(slot) * kv_head_count + kv_head) * head_dim);
        // lane 覆盖 4 个连续维:uchar4 一次装载(head_dim 为 32 倍数,group_size 为 4 倍数,
        // 4 维共享同一 quant scale),单 token 一次访存波取代 4 次字节装载。
        if (simd_lane * 4 < head_dim) {
            const uint dimension = simd_lane * 4;
            const half quant_scale = key_scales[(ulong(slot) * kv_head_count + kv_head) * groups_per_head + dimension / group_size];
            const char4 codes = reinterpret_cast<device const char4 *>(key_codes)[(key_base >> 2) + simd_lane];
            const float key_values[4] = { float(codes.x) * float(quant_scale), float(codes.y) * float(quant_scale), float(codes.z) * float(quant_scale), float(codes.w) * float(quant_scale) };
            for (uint query_head = 0; query_head < active_heads; ++query_head) {
                const uint base = query_head * head_dim + dimension;
                dot_products[query_head] += float(query_tile[base]) * key_values[0]
                    + float(query_tile[base + 1]) * key_values[1]
                    + float(query_tile[base + 2]) * key_values[2]
                    + float(query_tile[base + 3]) * key_values[3];
            }
        }
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            const float score = simd_sum(dot_products[query_head]) * score_scale;
            if (simd_lane == 0) weights[query_head * max_rows + token] = score;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // softmax:4 个头同做一次归约(thread t 持 token t 的全部头分数),
    // 4 次 barrier;逐头循环归约要 ~20 次 barrier,是纯延迟大头。
    float local_scores[heads_per_group];
    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        local_scores[query_head] = thread_index < source_rows ? weights[query_head * max_rows + thread_index] : -INFINITY;
    }
    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        const float simd_maximum = simd_max(local_scores[query_head]);
        if (simd_lane == 0) reduction[query_head * 8 + simd_group] = simd_maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < active_heads) {
        float block_maximum = -INFINITY;
        for (uint part = 0; part < simd_groups; ++part) block_maximum = max(block_maximum, reduction[thread_index * 8 + part]);
        reduction[128 + thread_index] = block_maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_weights[heads_per_group];
    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        local_weights[query_head] = thread_index < source_rows ? exp(local_scores[query_head] - reduction[128 + query_head]) : 0.0f;
        if (thread_index < source_rows) weights[query_head * max_rows + thread_index] = local_weights[query_head];
    }
    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        const float simd_denominator = simd_sum(local_weights[query_head]);
        if (simd_lane == 0) reduction[query_head * 8 + simd_group] = simd_denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < active_heads) {
        float block_denominator = 0.0f;
        for (uint part = 0; part < simd_groups; ++part) block_denominator += reduction[thread_index * 8 + part];
        denominators[thread_index] = block_denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 第三阶段:token 按 SIMD group 切分,lane 覆盖 4 个连续维(uchar4 装载),
    // 组间经 threadgroup partial 归约,消除只有 4 个 threadgroup 时的串行访存延迟链。
    if (simd_lane * 4 < head_dim) {
        const uint dimension = simd_lane * 4;
        float accumulated[heads_per_group][4];
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            for (uint i = 0; i < 4; ++i) accumulated[query_head][i] = 0.0f;
        }
        for (uint token = simd_group; token < source_rows; token += simd_groups) {
            const uint source = first_visible + token;
            const uint slot = kv_capacity == 0 ? source : source % kv_capacity;
            const ulong value_base = ((ulong(slot) * kv_head_count + kv_head) * head_dim);
            const half quant_scale = value_scales[(ulong(slot) * kv_head_count + kv_head) * groups_per_head + dimension / group_size];
            const char4 codes = reinterpret_cast<device const char4 *>(value_codes)[(value_base >> 2) + simd_lane];
            const float value_elements[4] = { float(codes.x) * float(quant_scale), float(codes.y) * float(quant_scale), float(codes.z) * float(quant_scale), float(codes.w) * float(quant_scale) };
            for (uint query_head = 0; query_head < active_heads; ++query_head) {
                const float weight = weights[query_head * max_rows + token];
                for (uint i = 0; i < 4; ++i) accumulated[query_head][i] += weight * value_elements[i];
            }
        }
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            for (uint i = 0; i < 4; ++i) {
                partials[(simd_group * heads_per_group + query_head) * head_dim + dimension + i] = accumulated[query_head][i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = thread_index; dimension < head_dim; dimension += 256) {
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            float sum = 0.0f;
            for (uint part = 0; part < simd_groups; ++part) {
                sum += partials[(part * heads_per_group + query_head) * head_dim + dimension];
            }
            const uint global_head = first_query_head + query_head;
            const float result = sum / denominators[query_head];
            output[ulong(global_head) * head_dim + dimension] = bf16 != 0
                ? zllm_f32_to_bf16(result)
                : as_type<ushort>(finite_f16(result));
        }
    }
}
// 小 KV decode 直通 + 当前行量化融合:省去独立的 kv_quantize dispatch(24 层 ~0.24ms/token)
// 与其 buffer 往返。每个 threadgroup 用与 gqa_kv_quantize_q8 逐位相同的公式在
// threadgroup 内存里量化当前行(max/127 的 f32 scale 定 codes,f16 舍入的 scale 参与
// 注意力),group 0 回写 cache;注意力读取当前行走 threadgroup 副本,数学上与
// “先 quantize 再读 cache” 完全一致。要求 kv_capacity==0(连续 append)。
kernel void gqa_decode_direct_q8_append(
    device const ushort *query [[buffer(0)]],
    device char *key_codes [[buffer(1)]],
    device char *value_codes [[buffer(2)]],
    device ushort *output [[buffer(3)]],
    device half *key_scales [[buffer(4)]],
    device half *value_scales [[buffer(5)]],
    device const ushort *new_key [[buffer(6)]],
    device const ushort *new_value [[buffer(7)]],
    constant uint &source_rows [[buffer(8)]],
    constant uint &head_count [[buffer(9)]],
    constant uint &kv_head_count [[buffer(10)]],
    constant uint &head_dim [[buffer(11)]],
    constant float &score_scale [[buffer(12)]],
    constant uint &bf16 [[buffer(13)]],
    constant uint &first_visible [[buffer(14)]],
    constant uint &group_size [[buffer(15)]],
    constant uint &groups_per_head [[buffer(16)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]])
{
    constexpr uint heads_per_group = 2;   // heads_per_kv=8 时 8 个 threadgroup,延迟链减半
    constexpr uint max_rows = 512;
    constexpr uint max_head_dim = 128;
    constexpr uint max_kv_columns = 512;
    if (kv_head_count == 0 || head_count % kv_head_count != 0) return;
    const uint heads_per_kv = head_count / kv_head_count;
    if (heads_per_kv == 0) return;
    const uint head_groups_per_kv = (heads_per_kv + heads_per_group - 1) / heads_per_group;
    const uint kv_head = group / head_groups_per_kv;
    const uint head_group = group - kv_head * head_groups_per_kv;
    const uint first_head_offset = head_group * heads_per_group;
    const uint active_heads = min(heads_per_group, heads_per_kv - first_head_offset);
    const uint kv_columns = kv_head_count * head_dim;
    if (kv_head >= kv_head_count || source_rows > max_rows || source_rows == 0
        || head_dim > max_head_dim || head_dim % 32 != 0 || kv_columns > max_kv_columns) return;

    threadgroup half query_tile[heads_per_group * max_head_dim];
    threadgroup float weights[heads_per_group * max_rows];
    threadgroup float reduction[256];
    threadgroup float denominators[heads_per_group];
    threadgroup float partials[8 * heads_per_group * max_head_dim];
    // 当前行(最后一行)的量化副本:全 threadgroup 共享,注意力与 group 0 回写都用它。
    threadgroup char new_key_codes[max_kv_columns];
    threadgroup char new_value_codes[max_kv_columns];
    threadgroup half new_key_scales[8];
    threadgroup half new_value_scales[8];

    // 量化当前行:每个 64 元素 group 一个 SIMD group(K/V 合计 group 数 = 2*kv*groups_per_head)。
    {
        const uint total_groups = kv_head_count * groups_per_head;
        for (uint g = simd_group; g < total_groups * 2; g += simd_groups) {
            const bool is_value = g >= total_groups;
            const uint group_index = g - (is_value ? total_groups : 0);
            device const ushort *input = is_value ? new_value : new_key;
            const uint base = group_index * group_size;
            float local_max = 0.0f;
            for (uint index = simd_lane; index < group_size; index += 32) {
                const ushort bits = input[base + index];
                local_max = max(local_max, abs(bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits))));
            }
            const float scale = simd_max(local_max) / 127.0f;
            const float safe_scale = scale > 0.0f ? scale : 1.0f;
            const half scale_f16 = half(safe_scale);
            if (simd_lane == 0) {
                (is_value ? new_value_scales : new_key_scales)[group_index] = scale_f16;
            }
            const float inverse_scale = 1.0f / safe_scale;
            for (uint index = simd_lane; index < group_size; index += 32) {
                const ushort bits = input[base + index];
                const float element = bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits));
                (is_value ? new_value_codes : new_key_codes)[base + index] = char(clamp(rint(element * inverse_scale), -127.0f, 127.0f));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // group 0 回写 cache(当前行 = source_rows - 1;scales 的 threadgroup 数组布局
    // 与设备布局相同:group 索引 = kv_head * groups_per_head + group_in_head)。
    if (group == 0) {
        const uint row = source_rows - 1;
        for (uint index = thread_index; index < kv_columns; index += 256) {
            key_codes[row * kv_columns + index] = new_key_codes[index];
            value_codes[row * kv_columns + index] = new_value_codes[index];
        }
        const uint total_scales = kv_head_count * groups_per_head;
        for (uint index = thread_index; index < total_scales; index += 256) {
            key_scales[row * total_scales + index] = new_key_scales[index];
            value_scales[row * total_scales + index] = new_value_scales[index];
        }
    }

    const uint query_elements = active_heads * head_dim;
    const uint first_query_head = kv_head * heads_per_kv + first_head_offset;
    for (uint index = thread_index; index < query_elements; index += 256) {
        const ushort bits = query[ulong(first_query_head) * head_dim + index];
        query_tile[index] = half(bf16 != 0 ? zllm_bf16_to_f32(bits) : float(as_type<half>(bits)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 每个 SIMD group 负责按 stride 分布的 token,寄存器内同时算全部 query heads。
    const uint last_token = source_rows - 1;
    for (uint token = simd_group; token < source_rows; token += simd_groups) {
        float dot_products[heads_per_group];
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            dot_products[query_head] = 0.0f;
        }
        const uint source = first_visible + token;
        const ulong key_base = ((ulong(source) * kv_head_count + kv_head) * head_dim);
        // lane 覆盖 4 个连续维:char4 一次装载;当前行走 threadgroup 量化副本。
        if (simd_lane * 4 < head_dim) {
            const uint dimension = simd_lane * 4;
            if (token == last_token) {
                const float quant_scale = float(new_key_scales[kv_head * groups_per_head + dimension / group_size]);
                const uint new_base = kv_head * head_dim + dimension;
                const float key_values[4] = {
                    float(new_key_codes[new_base]) * quant_scale,
                    float(new_key_codes[new_base + 1]) * quant_scale,
                    float(new_key_codes[new_base + 2]) * quant_scale,
                    float(new_key_codes[new_base + 3]) * quant_scale,
                };
                for (uint query_head = 0; query_head < active_heads; ++query_head) {
                    const uint base = query_head * head_dim + dimension;
                    dot_products[query_head] += float(query_tile[base]) * key_values[0]
                        + float(query_tile[base + 1]) * key_values[1]
                        + float(query_tile[base + 2]) * key_values[2]
                        + float(query_tile[base + 3]) * key_values[3];
                }
            } else {
                const half quant_scale = key_scales[(ulong(source) * kv_head_count + kv_head) * groups_per_head + dimension / group_size];
                const char4 codes = reinterpret_cast<device const char4 *>(key_codes)[(key_base >> 2) + simd_lane];
                const float key_values[4] = { float(codes.x) * float(quant_scale), float(codes.y) * float(quant_scale), float(codes.z) * float(quant_scale), float(codes.w) * float(quant_scale) };
                for (uint query_head = 0; query_head < active_heads; ++query_head) {
                    const uint base = query_head * head_dim + dimension;
                    dot_products[query_head] += float(query_tile[base]) * key_values[0]
                        + float(query_tile[base + 1]) * key_values[1]
                        + float(query_tile[base + 2]) * key_values[2]
                        + float(query_tile[base + 3]) * key_values[3];
                }
            }
        }
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            const float score = simd_sum(dot_products[query_head]) * score_scale;
            if (simd_lane == 0) weights[query_head * max_rows + token] = score;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // softmax:max_rows 512 > 256 线程,每线程负责 2 个 token 槽;
    // 归约布局 [head][slot][simdgroup],数学上与逐 token 单槽版本一致(全序 max/sum)。
    float local_scores[heads_per_group][2];
    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        for (uint s = 0; s < 2; ++s) {
            const uint token = thread_index + s * 256;
            local_scores[query_head][s] = token < source_rows ? weights[query_head * max_rows + token] : -INFINITY;
        }
    }
    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        for (uint s = 0; s < 2; ++s) {
            const float simd_maximum = simd_max(local_scores[query_head][s]);
            if (simd_lane == 0) reduction[(query_head * 2 + s) * 8 + simd_group] = simd_maximum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < active_heads) {
        float block_maximum = -INFINITY;
        for (uint part = 0; part < simd_groups * 2; ++part) block_maximum = max(block_maximum, reduction[thread_index * 16 + part]);
        reduction[128 + thread_index] = block_maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_weights[heads_per_group][2];
    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        const float block_maximum = reduction[128 + query_head];
        for (uint s = 0; s < 2; ++s) {
            const uint token = thread_index + s * 256;
            local_weights[query_head][s] = token < source_rows ? exp(local_scores[query_head][s] - block_maximum) : 0.0f;
            if (token < source_rows) weights[query_head * max_rows + token] = local_weights[query_head][s];
        }
    }
    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        for (uint s = 0; s < 2; ++s) {
            const float simd_denominator = simd_sum(local_weights[query_head][s]);
            if (simd_lane == 0) reduction[(query_head * 2 + s) * 8 + simd_group] = simd_denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < active_heads) {
        float block_denominator = 0.0f;
        for (uint part = 0; part < simd_groups * 2; ++part) block_denominator += reduction[thread_index * 16 + part];
        denominators[thread_index] = block_denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 第三阶段:token 按 SIMD group 切分,lane 覆盖 4 个连续维,组间 partial 归约。
    if (simd_lane * 4 < head_dim) {
        const uint dimension = simd_lane * 4;
        float accumulated[heads_per_group][4];
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            for (uint i = 0; i < 4; ++i) accumulated[query_head][i] = 0.0f;
        }
        for (uint token = simd_group; token < source_rows; token += simd_groups) {
            float value_elements[4];
            if (token == last_token) {
                const float quant_scale = float(new_value_scales[kv_head * groups_per_head + dimension / group_size]);
                const uint new_base = kv_head * head_dim + dimension;
                for (uint i = 0; i < 4; ++i) value_elements[i] = float(new_value_codes[new_base + i]) * quant_scale;
            } else {
                const uint source = first_visible + token;
                const ulong value_base = ((ulong(source) * kv_head_count + kv_head) * head_dim);
                const half quant_scale = value_scales[(ulong(source) * kv_head_count + kv_head) * groups_per_head + dimension / group_size];
                const char4 codes = reinterpret_cast<device const char4 *>(value_codes)[(value_base >> 2) + simd_lane];
                value_elements[0] = float(codes.x) * float(quant_scale);
                value_elements[1] = float(codes.y) * float(quant_scale);
                value_elements[2] = float(codes.z) * float(quant_scale);
                value_elements[3] = float(codes.w) * float(quant_scale);
            }
            for (uint query_head = 0; query_head < active_heads; ++query_head) {
                const float weight = weights[query_head * max_rows + token];
                for (uint i = 0; i < 4; ++i) accumulated[query_head][i] += weight * value_elements[i];
            }
        }
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            for (uint i = 0; i < 4; ++i) {
                partials[(simd_group * heads_per_group + query_head) * head_dim + dimension + i] = accumulated[query_head][i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = thread_index; dimension < head_dim; dimension += 256) {
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            float sum = 0.0f;
            for (uint part = 0; part < simd_groups; ++part) {
                sum += partials[(part * heads_per_group + query_head) * head_dim + dimension];
            }
            const uint global_head = first_query_head + query_head;
            const float result = sum / denominators[query_head];
            output[ulong(global_head) * head_dim + dimension] = bf16 != 0
                ? zllm_f32_to_bf16(result)
                : as_type<ushort>(finite_f16(result));
        }
    }
}
kernel void gqa_decode_softmax_f32(
    device float *scores [[buffer(0)]],
    constant uint &kv_rows [[buffer(1)]],
    constant uint &head_count [[buffer(2)]],
    uint query_head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    if (query_head >= head_count) return;
    threadgroup float reduction[256];
    const ulong begin = ulong(query_head) * kv_rows;
    float maximum = -INFINITY;
    for (uint token = lane; token < kv_rows; token += width) {
        maximum = max(maximum, scores[begin + token]);
    }
    reduction[lane] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] = max(reduction[lane], reduction[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float row_maximum = reduction[0];
    float denominator = 0.0f;
    for (uint token = lane; token < kv_rows; token += width) {
        const float probability = exp(scores[begin + token] - row_maximum);
        scores[begin + token] = probability;
        denominator += probability;
    }
    reduction[lane] = denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inverse = 1.0f / reduction[0];
    for (uint token = lane; token < kv_rows; token += width) {
        scores[begin + token] *= inverse;
    }
}
kernel void gqa_prefill_attention_tiled_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &query_rows [[buffer(4)]],
    constant uint &kv_rows [[buffer(5)]],
    constant uint &query_position [[buffer(6)]],
    constant uint &head_count [[buffer(7)]],
    constant uint &kv_head_count [[buffer(8)]],
    constant uint &head_dim [[buffer(9)]],
    constant float &score_scale [[buffer(10)]],
    constant uint &sliding_window [[buffer(11)]],
    constant uint &kv_start [[buffer(12)]],
    constant uint &kv_capacity [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]])
{
    // 每个 simd group 承担 4 个连续 query 行(同 head),token 循环共享 K/V 读取
    const uint rows_per_group = 4;
    const uint query_tile = group / head_count;
    const uint query_head = group % head_count;
    const uint row_base = query_tile * (simd_groups * rows_per_group) + simd_group * rows_per_group;
    if (row_base >= query_rows) {
        return;
    }

    const uint kv_head = query_head / (head_count / kv_head_count);
    float query_value[rows_per_group][8];
    float accumulated[rows_per_group][8];
    float maximum[rows_per_group];
    float denominator[rows_per_group];

    for (uint row_index = 0; row_index < rows_per_group; ++row_index) {
        const uint query_row = row_base + row_index;
        const ulong query_base = ((ulong)query_row * head_count + query_head) * head_dim;
        for (uint part = 0; part < 8; ++part) {
            const uint dimension = simd_lane + part * 32;
            query_value[row_index][part] = query_row < query_rows && dimension < head_dim ? float(query[query_base + dimension]) : 0.0f;
            accumulated[row_index][part] = 0.0f;
        }
        maximum[row_index] = -INFINITY;
        denominator[row_index] = 0.0f;
    }

    // token 范围取组内行可见区间的并集:起点按最早可见行计算,终点按最晚可见行
    const uint min_visible = min(kv_rows, query_position + row_base + 1);
    const uint max_visible = min(kv_rows, query_position + min(row_base + rows_per_group, query_rows));
    const uint min_window_start = sliding_window == 0 || min_visible <= sliding_window ? 0 : min_visible - sliding_window;
    const uint first_visible = max(kv_start, min_window_start);

    for (uint token = first_visible; token < max_visible; ++token) {
        const uint slot = kv_capacity == 0 ? token : token % kv_capacity;
        const ulong key_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        float key_value[8];
        float value_value[8];
        for (uint part = 0; part < 8; ++part) {
            const uint dimension = simd_lane + part * 32;
            if (dimension < head_dim) {
                key_value[part] = float(key[key_base + dimension]);
                value_value[part] = float(value[((ulong)slot * kv_head_count + kv_head) * head_dim + dimension]);
            } else {
                key_value[part] = 0.0f;
                value_value[part] = 0.0f;
            }
        }

        for (uint row_index = 0; row_index < rows_per_group; ++row_index) {
            const uint query_row = row_base + row_index;
            if (query_row >= query_rows) break;
            const uint visible = min(kv_rows, query_position + query_row + 1);
            const uint window_start = sliding_window == 0 || visible <= sliding_window ? 0 : visible - sliding_window;
            if (token < max(kv_start, window_start) || token >= visible) continue;
            float partial = 0.0f;
            for (uint part = 0; part < 8; ++part) {
                partial += query_value[row_index][part] * key_value[part];
            }

            const float score = simd_sum(partial) * score_scale;
            float rescale = 1.0f;
            float weight = 1.0f;
            if (score > maximum[row_index]) {
                rescale = exp(maximum[row_index] - score);
                maximum[row_index] = score;
            } else {
                weight = exp(score - maximum[row_index]);
            }
            denominator[row_index] = denominator[row_index] * rescale + weight;

            for (uint part = 0; part < 8; ++part) {
                accumulated[row_index][part] = accumulated[row_index][part] * rescale + weight * value_value[part];
            }
        }
    }

    for (uint row_index = 0; row_index < rows_per_group; ++row_index) {
        const uint query_row = row_base + row_index;
        if (query_row >= query_rows) break;
        const ulong query_base = ((ulong)query_row * head_count + query_head) * head_dim;
        for (uint part = 0; part < 8; ++part) {
            const uint dimension = simd_lane + part * 32;
            if (dimension < head_dim) {
                output[query_base + dimension] = half(accumulated[row_index][part] / denominator[row_index]);
            }
        }
    }
}
kernel void causal_softmax_windowed_rows_f32(
    device const float *scores [[buffer(0)]],
    device float *probabilities [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    constant uint &query_begin [[buffer(5)]],
    constant uint &window [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    const uint visible = min(columns, query_begin + row + 1);
    const uint first = window == 0 || visible <= window ? 0 : visible - window;
    const ulong begin = ulong(row) * stride;
    float maximum = -INFINITY;
    for (uint column = thread_index; column < visible; column += width) {
        if (column >= first) maximum = max(maximum, scores[begin + column]);
    }
    const float row_maximum = simd_max(maximum);
    float denominator = 0.0f;
    for (uint column = thread_index; column < visible; column += width) {
        if (column >= first) denominator += exp(scores[begin + column] - row_maximum);
    }
    const float inverse = 1.0f / simd_sum(denominator);
    for (uint column = thread_index; column < columns; column += width) {
        const float probability = column >= first && column < visible
            ? exp(scores[begin + column] - row_maximum) * inverse
            : 0.0f;
        probabilities[begin + column] = probability;
    }
}
kernel void causal_softmax_rows_f32(
    device const float *scores [[buffer(0)]],
    device float *probabilities [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    constant uint &query_begin [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    const uint visible = min(columns, query_begin + row + 1);
    const ulong begin = ulong(row) * stride;
    float maximum = -INFINITY;
    for (uint column = thread_index; column < visible; column += width) {
        maximum = max(maximum, scores[begin + column]);
    }
    const float row_maximum = simd_max(maximum);
    float denominator = 0.0f;
    for (uint column = thread_index; column < visible; column += width) {
        denominator += exp(scores[begin + column] - row_maximum);
    }
    const float inverse = 1.0f / simd_sum(denominator);
    for (uint column = thread_index; column < columns; column += width) {
        probabilities[begin + column] = column < visible
            ? exp(scores[begin + column] - row_maximum) * inverse
            : 0.0f;
    }
}
kernel void causal_softmax_rows_f32_f16(
    device const float *scores [[buffer(0)]],
    device half *probabilities [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    constant uint &query_begin [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    const uint visible = min(columns, query_begin + row + 1);
    const ulong begin = ulong(row) * stride;
    float maximum = -INFINITY;
    for (uint column = thread_index; column < visible; column += width) {
        maximum = max(maximum, scores[begin + column]);
    }
    const float row_maximum = simd_max(maximum);
    float denominator = 0.0f;
    for (uint column = thread_index; column < visible; column += width) {
        denominator += exp(scores[begin + column] - row_maximum);
    }
    const float inverse = 1.0f / simd_sum(denominator);
    for (uint column = thread_index; column < columns; column += width) {
        const float probability = column < visible
            ? exp(scores[begin + column] - row_maximum) * inverse
            : 0.0f;
        probabilities[begin + column] = half(probability);
    }
}
kernel void gqa_prefill_attention_bf16(
    device const ushort *query [[buffer(0)]], device const ushort *key [[buffer(1)]],
    device const ushort *value [[buffer(2)]], device ushort *output [[buffer(3)]],
    constant uint &query_rows [[buffer(4)]], constant uint &kv_rows [[buffer(5)]],
    constant uint &query_position [[buffer(6)]], constant uint &head_count [[buffer(7)]],
    constant uint &kv_head_count [[buffer(8)]], constant uint &head_dim [[buffer(9)]],
    constant float &score_scale [[buffer(10)]], constant uint &sliding_window [[buffer(11)]],
    constant uint &kv_start [[buffer(12)]], constant uint &kv_capacity [[buffer(13)]],
    device const uint *visible_ends [[buffer(14)]], constant uint &use_visible_ends [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (group >= query_rows * head_count) return;
    threadgroup float reduction[1024];
    threadgroup float control[4];
    const uint query_row = group / head_count;
    const uint query_head = group % head_count;
    const uint kv_head = query_head / (head_count / kv_head_count);
    const ulong query_base = ((ulong)query_row * head_count + query_head) * head_dim;
    float accumulated = 0.0f;
    const uint causal_rows = min(kv_rows, query_position + query_row + 1);
    const uint visible_rows = use_visible_ends == 0 ? causal_rows : min(kv_rows, visible_ends[query_row]);
    const uint window_start = sliding_window == 0 || causal_rows <= sliding_window ? 0 : causal_rows - sliding_window;
    const uint first_visible = max(kv_start, window_start);
    for (uint token = first_visible; token < visible_rows; ++token) {
        const uint slot = kv_capacity == 0 ? token : token % kv_capacity;
        const ulong key_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        reduction[lane] = lane < head_dim ? zllm_bf16_to_f32(query[query_base + lane]) * zllm_bf16_to_f32(key[key_base + lane]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float score = reduction[0] * score_scale;
            if (token == first_visible) {
                control[0] = 0.0f; control[1] = 1.0f; control[2] = 1.0f; control[3] = score;
            } else if (score > control[3]) {
                const float rescale = exp(control[3] - score);
                control[0] = rescale; control[1] = 1.0f; control[2] = control[2] * rescale + 1.0f; control[3] = score;
            } else {
                const float weight = exp(score - control[3]);
                control[0] = 1.0f; control[1] = weight; control[2] += weight;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < head_dim) {
            const ulong value_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
            accumulated = accumulated * control[0] + control[1] * zllm_bf16_to_f32(value[value_base + lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < head_dim) output[query_base + lane] = zllm_f32_to_bf16(accumulated / control[2]);
}
kernel void gqa_prefill_attention_tiled_bf16(
    device const ushort *query [[buffer(0)]], device const ushort *key [[buffer(1)]],
    device const ushort *value [[buffer(2)]], device ushort *output [[buffer(3)]],
    constant uint &query_rows [[buffer(4)]], constant uint &kv_rows [[buffer(5)]],
    constant uint &query_position [[buffer(6)]], constant uint &head_count [[buffer(7)]],
    constant uint &kv_head_count [[buffer(8)]], constant uint &head_dim [[buffer(9)]],
    constant float &score_scale [[buffer(10)]], constant uint &sliding_window [[buffer(11)]],
    constant uint &kv_start [[buffer(12)]], constant uint &kv_capacity [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint simd_groups [[simdgroups_per_threadgroup]])
{
    const uint query_tile = group / head_count;
    const uint query_head = group % head_count;
    const uint query_row = query_tile * simd_groups + simd_group;
    if (query_row >= query_rows) return;
    const uint kv_head = query_head / (head_count / kv_head_count);
    const ulong query_base = ((ulong)query_row * head_count + query_head) * head_dim;
    const uint visible_rows = min(kv_rows, query_position + query_row + 1);
    const uint window_start = sliding_window == 0 || visible_rows <= sliding_window ? 0 : visible_rows - sliding_window;
    const uint first_visible = max(kv_start, window_start);
    const uint dimension_parts = (head_dim + 31) / 32;
    float query_value[16];
    float accumulated[16];
    for (uint part = 0; part < dimension_parts; ++part) {
        const uint dimension = simd_lane + part * 32;
        query_value[part] = dimension < head_dim ? zllm_bf16_to_f32(query[query_base + dimension]) : 0.0f;
        accumulated[part] = 0.0f;
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint token = first_visible; token < visible_rows; ++token) {
        const uint slot = kv_capacity == 0 ? token : token % kv_capacity;
        const ulong key_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        float partial = 0.0f;
        for (uint part = 0; part < dimension_parts; ++part) {
            const uint dimension = simd_lane + part * 32;
            if (dimension < head_dim) partial += query_value[part] * zllm_bf16_to_f32(key[key_base + dimension]);
        }
        const float score = simd_sum(partial) * score_scale;
        float rescale = 1.0f;
        float weight = 1.0f;
        if (score > maximum) { rescale = exp(maximum - score); maximum = score; }
        else { weight = exp(score - maximum); }
        denominator = denominator * rescale + weight;
        const ulong value_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        for (uint part = 0; part < dimension_parts; ++part) {
            const uint dimension = simd_lane + part * 32;
            if (dimension < head_dim) accumulated[part] = accumulated[part] * rescale + weight * zllm_bf16_to_f32(value[value_base + dimension]);
        }
    }
    for (uint part = 0; part < dimension_parts; ++part) {
        const uint dimension = simd_lane + part * 32;
        if (dimension < head_dim) output[query_base + dimension] = zllm_f32_to_bf16(accumulated[part] / denominator);
    }
}
kernel void segmented_rmsnorm_add_scaled_bf16(
    device const ushort *left [[buffer(0)]],
    device const ushort *right [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device ushort *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &segments [[buffer(5)]],
    constant uint &segment_columns [[buffer(6)]],
    constant uint &total_columns [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant uint &simdgroups [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    const uint row = group / segments;
    const uint segment = group - row * segments;
    if (row >= rows) return;
    const ulong base = ulong(row) * total_columns + ulong(segment) * segment_columns;
    float squared = 0.0f;
    for (uint column = thread_index; column < segment_columns; column += threads) {
        const float value = zllm_bf16_to_f32(right[base + column]);
        squared += value * value;
    }
    squared = simd_sum(squared);
    threadgroup float partial[32];
    threadgroup float inverse;
    if (simd_lane == 0) partial[simd_index] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float total = 0.0f;
        for (uint index = 0; index < simdgroups; ++index) total += partial[index];
        inverse = rsqrt(total / float(segment_columns) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = thread_index; column < segment_columns; column += threads) {
        const ulong index = base + column;
        const float normalized = zllm_bf16_to_f32(zllm_f32_to_bf16(
            zllm_bf16_to_f32(right[index]) * inverse * float(weight[column])
        ));
        output[index] = zllm_f32_to_bf16((zllm_bf16_to_f32(left[index]) + normalized) * scale);
    }
}
kernel void segmented_rmsnorm_add_scaled_f16(
    device const half *left [[buffer(0)]],
    device const half *right [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &segments [[buffer(5)]],
    constant uint &segment_columns [[buffer(6)]],
    constant uint &total_columns [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant uint &simdgroups [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    const uint row = group / segments;
    const uint segment = group - row * segments;
    if (row >= rows) return;
    const ulong base = ulong(row) * total_columns + ulong(segment) * segment_columns;
    float squared = 0.0f;
    for (uint column = thread_index; column < segment_columns; column += threads) {
        const float value = float(right[base + column]);
        squared += value * value;
    }
    squared = simd_sum(squared);
    threadgroup float partial[32];
    threadgroup float inverse;
    if (simd_lane == 0) partial[simd_index] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float total = 0.0f;
        for (uint index = 0; index < simdgroups; ++index) total += partial[index];
        inverse = rsqrt(total / float(segment_columns) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = thread_index; column < segment_columns; column += threads) {
        const ulong index = base + column;
        const float normalized = float(half(float(right[index]) * inverse * float(weight[column])));
        output[index] = half((float(left[index]) + normalized) * scale);
    }
}
// ICB 重放的 decode 专用:position 由 uniform buffer 提供(槽布局 [position, kv_rows, kv_start]),
// slot = position % capacity 环绕寻址,替代随 position 变化的 blit copy。
kernel void gqa_kv_append_f16_position(
    device const half *key [[buffer(0)]],
    device const half *value [[buffer(1)]],
    device half *cache_key [[buffer(2)]],
    device half *cache_value [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &capacity [[buffer(5)]],
    constant uint *decode_state [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= columns) return;
    const uint slot = decode_state[0] % capacity;
    const ulong target = (ulong)slot * columns + idx;
    cache_key[target] = key[idx];
    cache_value[target] = value[idx];
}
// 多行(verify 重放)append:K/V 是 [rows, columns],行 r 写 (position_base+r)%capacity 槽。
kernel void gqa_kv_append_rows_f16_position(
    device const half *key [[buffer(0)]],
    device const half *value [[buffer(1)]],
    device half *cache_key [[buffer(2)]],
    device half *cache_value [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &capacity [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint *decode_state [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= rows * columns) return;
    const uint row = idx / columns;
    const uint column = idx - row * columns;
    const uint slot = (decode_state[0] + row) % capacity;
    const ulong target = (ulong)slot * columns + column;
    cache_key[target] = key[idx];
    cache_value[target] = value[idx];
}

// 多行 attention:grid.x = head_count*query_rows,group 解出 (head,row);
// 行 r 的 causal_rows = min(kv_rows, position_base+r+1)。几何与单行版一致。
// verify 重放的 head-major 多行 attention:每 TG 一个 head,一次 K/V 读取服务
// 全部 query 行(行数据在寄存器,KV 带宽随行数摊薄)。position 基址经 state 槽。
// verify 重放的 head-major 无 barrier 版:每 simdgroup 独占一个 head(TG=32),
// head_dim 拆 32 值段由 lane 循环常驻,dot 与 softmax 全在 simdgroup 内——
// 零 threadgroup_barrier(前版每 token 2 次 barrier × KV 长 × 48 层)。
// K/V 每 token 每 lane 各读 segments 次,4 行共享(带宽摊薄)。
kernel void gqa_decode_attention_nobar_f16_position(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint *decode_state [[buffer(4)]],
    constant uint &head_count [[buffer(5)]],
    constant uint &kv_head_count [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant float &score_scale [[buffer(8)]],
    constant uint &sliding_window [[buffer(9)]],
    constant uint &kv_capacity [[buffer(10)]],
    constant uint &query_rows [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (group >= head_count) return;
    const uint kv_rows = decode_state[1];
    const uint kv_start = decode_state[2];
    const uint position_base = decode_state[0];
    const uint kv_head = group / (head_count / kv_head_count);
    const uint segments = head_dim / 32u;
    // lane 常驻全部行的 q 分量(段 s 的第 lane 分量);q 是静态的,可常驻
    float q[4][16];
    #pragma unroll
    for (uint s = 0; s < 16; ++s) {
        if (s < segments) {
            #pragma unroll
            for (uint row = 0; row < 4; ++row) {
                q[row][s] = (row < query_rows) ? float(query[(ulong)row * head_count * head_dim + (ulong)group * head_dim + s * 32u + simd_lane]) : 0.0f;
            }
        }
    }

    float maximum[4]; float denominator[4];
    float acc[4][16];
    #pragma unroll
    for (uint row = 0; row < 4; ++row) {
        maximum[row] = -INFINITY; denominator[row] = 0.0f;
        #pragma unroll
        for (uint s = 0; s < 16; ++s) { acc[row][s] = 0.0f; }
    }

    const uint last_causal = min(kv_rows, position_base + query_rows);
    const uint window_start = sliding_window == 0 || last_causal <= sliding_window ? 0 : last_causal - sliding_window;
    const uint first_visible = max(kv_start, window_start);
    for (uint token = first_visible; token < last_causal; ++token) {
        const uint slot = kv_capacity == 0 ? token : token % kv_capacity;
        const ulong kv_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        // K 段读 + 行 dot(每行一次 simd_sum;simd_sum 要求全 lane 一致执行,
        // 行守卫由 row<query_rows 与 token<=position_base+row 决定——
        // 同一 simdgroup 内这些是统一常量,守卫对全 lane 一致 ✓)
        #pragma unroll
        for (uint row = 0; row < 4; ++row) {
            const bool row_active = row < query_rows && token <= position_base + row;
            float partial = 0.0f;
            #pragma unroll
            for (uint s = 0; s < 16; ++s) {
                if (s < segments) {
                    partial += row_active ? q[row][s] * float(key[kv_base + s * 32u + simd_lane]) : 0.0f;
                }
            }
            const float score = simd_sum(partial) * score_scale;
            if (row_active) {
                const float next_maximum = fmax(maximum[row], score);
                const float rescale = exp(maximum[row] - next_maximum);
                const float weight = exp(score - next_maximum);
                denominator[row] = denominator[row] * rescale + weight;
                maximum[row] = next_maximum;
                #pragma unroll
                for (uint s = 0; s < 16; ++s) {
                    if (s < segments) {
                        const float v_component = float(value[kv_base + s * 32u + simd_lane]);
                        acc[row][s] = acc[row][s] * rescale + weight * v_component;
                    }
                }
            } else {
                // 保持 simd 均衡:非激活行的 score 仍计算但不落地
                maximum[row] = maximum[row];
            }
        }
    }
    #pragma unroll
    for (uint row = 0; row < 4; ++row) {
        if (row < query_rows) {
            #pragma unroll
            for (uint s = 0; s < 16; ++s) {
                if (s < segments) {
                    output[(ulong)row * head_count * head_dim + (ulong)group * head_dim + s * 32u + simd_lane] = half(acc[row][s] / denominator[row]);
                }
            }
        }
    }
}

kernel void gqa_decode_attention_headmajor_f16_position(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint *decode_state [[buffer(4)]],
    constant uint &head_count [[buffer(5)]],
    constant uint &kv_head_count [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant float &score_scale [[buffer(8)]],
    constant uint &sliding_window [[buffer(9)]],
    constant uint &kv_capacity [[buffer(10)]],
    constant uint &query_rows [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (group >= head_count) return;
    const uint kv_rows = decode_state[1];
    const uint kv_start = decode_state[2];
    const uint position_base = decode_state[0];
    const uint kv_head = group / (head_count / kv_head_count);
    const uint simd_lanes = 32u;
    const uint dim_simd_groups = head_dim / simd_lanes;
    const uint element = simd_group * simd_lanes + simd_lane;
    const bool active = element < head_dim;
    // 每 lane 常驻全部 query 行的对应分量(行数 ≤ 4,寄存器可容)
    float q[4];
    #pragma unroll
    for (uint row = 0; row < 4; ++row) {
        q[row] = (active && row < query_rows) ? float(query[(ulong)row * head_count * head_dim + (ulong)group * head_dim + element]) : 0.0f;
    }

    threadgroup float cross_scores[16][4];
    float maximum[4]; float denominator[4]; float accumulated[4];
    #pragma unroll
    for (uint row = 0; row < 4; ++row) {
        maximum[row] = -INFINITY; denominator[row] = 0.0f; accumulated[row] = 0.0f;
    }

    // 可见窗口取全部行的并集;逐行 causal 由 position_base+row 决定
    const uint last_causal = min(kv_rows, position_base + query_rows);
    const uint window_start = sliding_window == 0 || last_causal <= sliding_window ? 0 : last_causal - sliding_window;
    const uint first_visible = max(kv_start, window_start);
    for (uint token = first_visible; token < last_causal; ++token) {
        const uint slot = kv_capacity == 0 ? token : token % kv_capacity;
        const ulong key_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        const float k = active ? float(key[key_base + element]) : 0.0f;
        const float v = active ? float(value[((ulong)slot * kv_head_count + kv_head) * head_dim + element]) : 0.0f;
        #pragma unroll
        for (uint row = 0; row < 4; ++row) {
            if (row < query_rows && token <= position_base + row) {
                const float partial = q[row] * k;
                const float reduced = simd_sum(partial);
                if (simd_lane == 0) {
                    cross_scores[simd_group][row] = reduced;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint row = 0; row < 4; ++row) {
            if (row < query_rows && token <= position_base + row) {
                float score = 0.0f;
                for (uint index = 0; index < dim_simd_groups; ++index) {
                    score += cross_scores[index][row];
                }
                score *= score_scale;
                const float next_maximum = fmax(maximum[row], score);
                const float rescale = exp(maximum[row] - next_maximum);
                const float weight = exp(score - next_maximum);
                accumulated[row] = accumulated[row] * rescale + weight * v;
                denominator[row] = denominator[row] * rescale + weight;
                maximum[row] = next_maximum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        #pragma unroll
        for (uint row = 0; row < 4; ++row) {
            if (row < query_rows) {
                output[(ulong)row * head_count * head_dim + (ulong)group * head_dim + element] = half(accumulated[row] / denominator[row]);
            }
        }
    }
}

kernel void gqa_decode_attention_rows_f16_position(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint *decode_state [[buffer(4)]],
    constant uint &head_count [[buffer(5)]],
    constant uint &kv_head_count [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant float &score_scale [[buffer(8)]],
    constant uint &sliding_window [[buffer(9)]],
    constant uint &kv_capacity [[buffer(10)]],
    constant uint &query_rows [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (group >= head_count * query_rows) return;
    const uint row = group / head_count;
    const uint head = group - row * head_count;
    const uint query_position = decode_state[0] + row;
    const uint kv_rows = decode_state[1];
    const uint kv_start = decode_state[2];
    const uint kv_head = head / (head_count / kv_head_count);
    const ulong query_base = (ulong)row * head_count * head_dim + (ulong)head * head_dim;
    const uint simd_lanes = 32u;
    const uint dim_simd_groups = head_dim / simd_lanes;
    const uint element = simd_group * simd_lanes + simd_lane;
    const bool active = element < head_dim;
    const float q = active ? float(query[query_base + element]) : 0.0f;

    threadgroup float cross_scores[16];
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulated = 0.0f;

    const uint causal_rows = min(kv_rows, query_position + 1);
    const uint window_start = sliding_window == 0 || causal_rows <= sliding_window ? 0 : causal_rows - sliding_window;
    const uint first_visible = max(kv_start, window_start);
    for (uint token = first_visible; token < causal_rows; ++token) {
        const uint slot = kv_capacity == 0 ? token : token % kv_capacity;
        const ulong key_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        const float partial = q * float(key[key_base + element]);
        const float reduced = simd_sum(partial);
        if (simd_lane == 0) {
            cross_scores[simd_group] = reduced;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = 0.0f;
        for (uint index = 0; index < dim_simd_groups; ++index) {
            score += cross_scores[index];
        }
        score *= score_scale;
        const float next_maximum = fmax(maximum, score);
        const float rescale = exp(maximum - next_maximum);
        const float weight = exp(score - next_maximum);
        accumulated = accumulated * rescale + weight * float(value[((ulong)slot * kv_head_count + kv_head) * head_dim + element]);
        denominator = denominator * rescale + weight;
        maximum = next_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        output[query_base + element] = half(accumulated / denominator);
    }
}

// ICB 重放的 decode 单 token attention:query_rows=1,kv_rows/query_position/kv_start
// 全部来自 uniform buffer 槽,常量参数(heads/dim/scale/window/capacity)录制时固化。
kernel void gqa_decode_attention_f16_position(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint *decode_state [[buffer(4)]],
    constant uint &head_count [[buffer(5)]],
    constant uint &kv_head_count [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant float &score_scale [[buffer(8)]],
    constant uint &sliding_window [[buffer(9)]],
    constant uint &kv_capacity [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (group >= head_count) return;
    const uint query_position = decode_state[0];
    const uint kv_rows = decode_state[1];
    const uint kv_start = decode_state[2];
    const uint kv_head = group / (head_count / kv_head_count);
    const ulong query_base = (ulong)group * head_dim;
    // threads == next_pow2(head_dim) 且 head_dim 是 32 的倍数(gemma4 256/512):
    // simdgroup 恰好铺满 head_dim,本 lane 常驻一个 head_dim 分量;q·k 用 simd_sum
    // 硬件归约 + 单次 threadgroup 合并,替代逐 token 十余次 barrier 的树形归约。
    const uint simd_lanes = 32u;
    const uint dim_simd_groups = head_dim / simd_lanes;
    const uint element = simd_group * simd_lanes + simd_lane;
    const bool active = element < head_dim;
    const float q = active ? float(query[query_base + element]) : 0.0f;

    threadgroup float cross_scores[16];
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulated = 0.0f;

    const uint causal_rows = min(kv_rows, query_position + 1);
    const uint window_start = sliding_window == 0 || causal_rows <= sliding_window ? 0 : causal_rows - sliding_window;
    const uint first_visible = max(kv_start, window_start);
    for (uint token = first_visible; token < causal_rows; ++token) {
        const uint slot = kv_capacity == 0 ? token : token % kv_capacity;
        const ulong key_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        const float partial = q * float(key[key_base + element]);
        // simd_sum 必须由 simdgroup 全体 lane 一致执行(单 lane 分支内调用结果未定义)
        const float reduced = simd_sum(partial);
        if (simd_lane == 0) {
            cross_scores[simd_group] = reduced;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = 0.0f;
        for (uint index = 0; index < dim_simd_groups; ++index) {
            score += cross_scores[index];
        }
        score *= score_scale;
        // online softmax:各 lane 输入一致 ⇒ 结果一致,无需单 lane 串行更新再广播。
        const float next_maximum = fmax(maximum, score);
        const float rescale = exp(maximum - next_maximum);
        const float weight = exp(score - next_maximum);
        accumulated = accumulated * rescale + weight * float(value[((ulong)slot * kv_head_count + kv_head) * head_dim + element]);
        denominator = denominator * rescale + weight;
        maximum = next_maximum;
        // cross_scores 复用前的写读隔离(每 token 单次 barrier)。
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        output[query_base + element] = half(accumulated / denominator);
    }
}

// split-KV decode attention(position 化):每 simdgroup 承担 token 的一个切片
// (first_visible + sg,步进 sg_count),片内是 nobar 结构(lane 常驻 q/acc 分量,
// 片内逐 token 零 barrier),片末经 threadgroup 合并各片 online softmax(3 次
// barrier 总量)。barrier 版(gqa_decode_attention_f16_position)逐 token 2 次全
// threadgroup 同步,短中上下文 decode 实测 0.46ms/层;本版消掉串行同步。
// 单 warp/head 的 nobar 版实测更慢(2.6ms/层,head_dim=512 寄存器溢出 + 串行链),
// 故按 simdgroup 切片而非按 head 独占。要求 head_dim 为 2 的幂且 ≤512
// (threads == head_dim ⇒ sg_count == segments,merge_acc 寄存器/内存可容)。
kernel void gqa_decode_attention_split_f16_position(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint *decode_state [[buffer(4)]],
    constant uint &head_count [[buffer(5)]],
    constant uint &kv_head_count [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant float &score_scale [[buffer(8)]],
    constant uint &sliding_window [[buffer(9)]],
    constant uint &kv_capacity [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    if (group >= head_count) return;
    const uint kv_rows = decode_state[1];
    const uint kv_start = decode_state[2];
    const uint position = decode_state[0];
    const uint kv_head = group / (head_count / kv_head_count);
    const uint segments = head_dim / 32u;
    const uint sg_count = head_dim / 32u;
    // 每 lane 常驻 q 的 segments 个分量(片内复用,无逐 token 重读)
    float q[16];
    #pragma unroll
    for (uint s = 0; s < 16; ++s) {
        q[s] = (s < segments) ? float(query[(ulong)group * head_dim + s * 32u + simd_lane]) : 0.0f;
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[16];
    #pragma unroll
    for (uint s = 0; s < 16; ++s) { acc[s] = 0.0f; }

    const uint causal_rows = min(kv_rows, position + 1);
    const uint window_start = sliding_window == 0 || causal_rows <= sliding_window ? 0 : causal_rows - sliding_window;
    const uint first_visible = max(kv_start, window_start);
    for (uint token = first_visible + simd_group; token < causal_rows; token += sg_count) {
        const uint slot = kv_capacity == 0 ? token : token % kv_capacity;
        const ulong kv_base = ((ulong)slot * kv_head_count + kv_head) * head_dim;
        float partial = 0.0f;
        #pragma unroll
        for (uint s = 0; s < 16; ++s) {
            if (s < segments) {
                partial += q[s] * float(key[kv_base + s * 32u + simd_lane]);
            }
        }
        const float score = simd_sum(partial) * score_scale;
        const float next_maximum = fmax(maximum, score);
        const float rescale = exp(maximum - next_maximum);
        const float weight = exp(score - next_maximum);
        denominator = denominator * rescale + weight;
        maximum = next_maximum;
        #pragma unroll
        for (uint s = 0; s < 16; ++s) {
            if (s < segments) {
                acc[s] = acc[s] * rescale + weight * float(value[kv_base + s * 32u + simd_lane]);
            }
        }
    }

    // 合并各片 online softmax:空片(maximum=-INF,denominator=0)rescale 自然为 0。
    // 本机工具链不支持 threadgroup float atomics;共享单缓冲的树形写读又有片间
    // 覆写竞态(曾导致输出乱码)。改为每片独立 staging 行 f32、按 256 分量分批
    // (head_dim=512 两批,内存 16KB),末线程按分量求 sg_count 片和。
    threadgroup float merge_m[16];
    threadgroup float merge_d[16];
    threadgroup float merge_acc[16][256];
    if (simd_lane == 0) {
        merge_m[simd_group] = maximum;
        merge_d[simd_group] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_maximum = -INFINITY;
    for (uint sg = 0; sg < sg_count; ++sg) {
        global_maximum = fmax(global_maximum, merge_m[sg]);
    }
    float global_denominator = 0.0f;
    for (uint sg = 0; sg < sg_count; ++sg) {
        global_denominator += merge_d[sg] * exp(merge_m[sg] - global_maximum);
    }
    const float merge_rescale = exp(maximum - global_maximum);
    const uint merge_passes = (head_dim + 255u) / 256u;
    for (uint pass = 0; pass < merge_passes; ++pass) {
        const uint component_base = pass * 256u;
        #pragma unroll
        for (uint s = 0; s < 16; ++s) {
            const uint component = s * 32u + simd_lane;
            if (s < segments && component >= component_base && component < component_base + 256u) {
                merge_acc[simd_group][component - component_base] = acc[s] * merge_rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = thread_index; e < 256u; e += sg_count * 32u) {
            const uint component = component_base + e;
            if (component < head_dim) {
                float total = 0.0f;
                for (uint sg = 0; sg < sg_count; ++sg) {
                    total += merge_acc[sg][e];
                }
                output[(ulong)group * head_dim + component] = half(total / global_denominator);
            }
        }
        // 下一 pass 覆写 staging 前的写读隔离
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// ============================================================================
// DSpark drafter 的非对称块注意力:每个 query 行独立可见区间(split-block
// online softmax + merge,与 gqa_decode_split_kv 同构)。服务小 drafter:
// query 行数 = block_size,heads_per_kv 典型 4,KV 是完整张量而非 paged cache。
// ============================================================================
kernel void block_attention_split(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device float *statistics [[buffer(3)]],
    device float *partial_values [[buffer(4)]],
    device const uint *visible [[buffer(5)]],
    constant uint &source_rows [[buffer(6)]],
    constant uint &head_count [[buffer(7)]],
    constant uint &kv_head_count [[buffer(8)]],
    constant uint &head_dim [[buffer(9)]],
    constant uint &block_tokens [[buffer(10)]],
    constant uint &block_count [[buffer(11)]],
    constant float &score_scale [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]])
{
    constexpr uint heads_per_group = 4;
    constexpr uint max_head_dim = 512;
    constexpr uint max_block_tokens = 256;
    constexpr uint threads = 256;
    if (kv_head_count == 0 || head_count % kv_head_count != 0) return;
    const uint heads_per_kv = head_count / kv_head_count;
    const uint query_row = group.z;
    const uint head_groups_per_kv = (heads_per_kv + heads_per_group - 1) / heads_per_group;
    const uint kv_head = group.y / head_groups_per_kv;
    const uint head_group = group.y - kv_head * head_groups_per_kv;
    const uint first_head_offset = head_group * heads_per_group;
    const uint active_heads = min(heads_per_group, heads_per_kv - first_head_offset);
    const uint block = group.x;
    if (block >= block_count || kv_head >= kv_head_count || head_dim > max_head_dim
        || block_tokens == 0 || block_tokens > max_block_tokens) return;

    const uint visible_start = visible[query_row * 2];
    const uint visible_end = visible[query_row * 2 + 1];
    const uint row_begin = block * block_tokens;
    const uint rows = min(block_tokens, source_rows - row_begin);

    threadgroup half query_tile[heads_per_group * max_head_dim];
    threadgroup float weights[heads_per_group * max_block_tokens];
    threadgroup float reduction[256];

    const uint query_elements = active_heads * head_dim;
    const uint first_query_head = kv_head * heads_per_kv + first_head_offset;
    for (uint index = thread_index; index < query_elements; index += threads) {
        query_tile[index] = query[ulong(query_row) * head_count * head_dim + ulong(first_query_head) * head_dim + index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint token = simd_group; token < rows; token += simd_groups) {
        const uint source = row_begin + token;
        const bool in_range = source >= visible_start && source < visible_end;
        float dot_products[heads_per_group];
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            dot_products[query_head] = 0.0f;
        }
        const ulong key_base = (ulong(source) * kv_head_count + kv_head) * head_dim;
        for (uint dimension = simd_lane; dimension < head_dim; dimension += 32) {
            const float key_value = float(key[key_base + dimension]);
            for (uint query_head = 0; query_head < active_heads; ++query_head) {
                dot_products[query_head] += float(query_tile[query_head * head_dim + dimension]) * key_value;
            }
        }
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            const float total = simd_sum(dot_products[query_head]) * score_scale;
            const float score = in_range ? total : -INFINITY;
            if (simd_lane == 0) weights[query_head * max_block_tokens + token] = score;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint query_head = 0; query_head < active_heads; ++query_head) {
        const uint weight_base = query_head * max_block_tokens;
        const float local_score = thread_index < rows ? weights[weight_base + thread_index] : -INFINITY;
        const float simd_maximum = simd_max(local_score);
        if (simd_lane == 0) reduction[simd_group] = simd_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            const float group_maximum = simd_lane < simd_groups ? reduction[simd_lane] : -INFINITY;
            const float block_maximum = simd_max(group_maximum);
            if (simd_lane == 0) reduction[0] = block_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float block_maximum = reduction[0];
        const float weight = local_score > -INFINITY && thread_index < rows ? exp(local_score - block_maximum) : 0.0f;
        if (thread_index < rows) weights[weight_base + thread_index] = weight;
        const float simd_denominator = simd_sum(weight);
        // 所有 SIMD group 读完广播的 maximum 后，group 0 才能复用 reduction[0]。
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) reduction[simd_group] = simd_denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            const float group_denominator = simd_lane < simd_groups ? reduction[simd_lane] : 0.0f;
            const float block_denominator = simd_sum(group_denominator);
            if (simd_lane == 0) reduction[0] = block_denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index == 0) {
            const uint head_row = query_row * head_count + first_query_head + query_head;
            const ulong statistic = (ulong(head_row) * block_count + block) * 2;
            statistics[statistic] = block_maximum;
            statistics[statistic + 1] = reduction[0];
        }
    }

    for (uint dimension = thread_index; dimension < head_dim; dimension += threads) {
        float accumulated[heads_per_group];
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            accumulated[query_head] = 0.0f;
        }
        for (uint token = 0; token < rows; ++token) {
            const ulong value_index = (ulong(row_begin + token) * kv_head_count + kv_head) * head_dim + dimension;
            const float value_element = float(value[value_index]);
            for (uint query_head = 0; query_head < active_heads; ++query_head) {
                accumulated[query_head] += weights[query_head * max_block_tokens + token] * value_element;
            }
        }
        for (uint query_head = 0; query_head < active_heads; ++query_head) {
            const uint head_row = query_row * head_count + first_query_head + query_head;
            const ulong partial = (ulong(head_row) * block_count + block) * head_dim + dimension;
            partial_values[partial] = accumulated[query_head];
        }
    }
}
kernel void block_attention_merge(
    device const float *statistics [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &block_count [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    uint head_row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float reduction[2];
    if (thread_index < 32) {
        float local_maximum = -INFINITY;
        for (uint block = thread_index; block < block_count; block += 32) {
            local_maximum = max(local_maximum, statistics[(ulong(head_row) * block_count + block) * 2]);
        }
        const float maximum = simd_max(local_maximum);
        if (thread_index == 0) reduction[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = reduction[0];
    if (thread_index < 32) {
        float local_denominator = 0.0f;
        for (uint block = thread_index; block < block_count; block += 32) {
            const ulong statistic = (ulong(head_row) * block_count + block) * 2;
            local_denominator += statistics[statistic + 1] * exp(statistics[statistic] - maximum);
        }
        const float denominator = simd_sum(local_denominator);
        if (thread_index == 0) reduction[1] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denominator = max(reduction[1], 1.0e-12f);
    for (uint dimension = thread_index; dimension < head_dim; dimension += width) {
        float accumulated = 0.0f;
        for (uint block = 0; block < block_count; ++block) {
            const ulong statistic = (ulong(head_row) * block_count + block) * 2;
            const ulong partial = (ulong(head_row) * block_count + block) * head_dim + dimension;
            accumulated += partial_values[partial] * exp(statistics[statistic] - maximum);
        }
        output[ulong(head_row) * head_dim + dimension] = finite_f16(accumulated / denominator);
    }
}
"#;

use crate::backend::metal::api as metal;

use super::{CausalWindow, GqaSpec, MTLSize, MetalContext, MetalGqaCacheView, MetalKvCacheFormat, MetalTensor, MetalTensorDType, THREADS, f16, launch_1d, mem, set_bytes, to_f16_tensor, validate_u32};

mod block;
mod decode;
mod mps;

pub use block::block_attention_tensor;
pub use decode::gqa_decode_attention_split_kv_buffers;
use decode::{gqa_decode_attention_parallel_bf16_buffers, gqa_decode_attention_split_kv_buffers_q8};
use mps::*;

#[allow(clippy::too_many_arguments)]
pub fn mla_attention_tensor(ctx: &MetalContext, q: &MetalTensor, expanded_kv: &MetalTensor, k_rope: &MetalTensor, head_count: usize, rope_dim: usize) -> Result<MetalTensor, String> {
    if q.rows != expanded_kv.rows || q.rows != k_rope.rows || k_rope.cols != rope_dim || !q.cols.is_multiple_of(head_count) || !expanded_kv.cols.is_multiple_of(head_count) {
        return Err("MLA MetalTensor shape 非法".to_owned());
    }
    let q_head_dim = q.cols / head_count;
    let kv_head_dim = expanded_kv.cols / head_count;
    if q_head_dim > THREADS || rope_dim > q_head_dim || kv_head_dim < q_head_dim - rope_dim || kv_head_dim - (q_head_dim - rope_dim) != q_head_dim {
        return Err(format!("MLA head shape 不支持: q={q_head_dim}, kv={kv_head_dim}, rope={rope_dim}"));
    }
    let pipeline = ctx.pipeline("mla_prefill_attention_simd_f16")?;
    let output = ctx.tensor_kernel_output(q.rows, q.cols);
    let rows = validate_u32("rows", q.rows)?;
    let q_columns = validate_u32("q_columns", q.cols)?;
    let kv_columns = validate_u32("kv_columns", expanded_kv.cols)?;
    let heads = validate_u32("head_count", head_count)?;
    let q_head = validate_u32("q_head_dim", q_head_dim)?;
    let kv_head = validate_u32("kv_head_dim", kv_head_dim)?;
    let rope = validate_u32("rope_dim", rope_dim)?;
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&q.buffer), 0);
    encoder.set_buffer(1, Some(&expanded_kv.buffer), 0);
    encoder.set_buffer(2, Some(&k_rope.buffer), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &rows);
    set_bytes(&encoder, 5, &q_columns);
    set_bytes(&encoder, 6, &kv_columns);
    set_bytes(&encoder, 7, &heads);
    set_bytes(&encoder, 8, &q_head);
    set_bytes(&encoder, 9, &kv_head);
    set_bytes(&encoder, 10, &rope);
    set_bytes(&encoder, 11, &scale);
    let query_groups = q.rows.div_ceil(64) as u64;
    encoder.dispatch_thread_groups(MTLSize::new(heads as u64, query_groups, 1), MTLSize::new(512, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={rows},q={q_columns},kv={kv_columns},heads={heads},rope={rope}");
    ctx.commit_and_wait_profiled(&command, "mla_prefill_attention_simd_f16", &shape, q.buffer.length() + expanded_kv.buffer.length() + k_rope.buffer.length(), output.buffer.length());
    Ok(output)
}

/// 单序列 causal GQA prefill。每个 `(query, head)` 使用一个 threadgroup，
/// 以在线 softmax 累加 value，不物化 `[rows, rows]` score 矩阵。
pub fn gqa_prefill_attention_tensor(ctx: &MetalContext, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, spec: &GqaSpec) -> Result<MetalTensor, String> {
    let head_count = spec.num_heads;
    let kv_head_count = spec.num_kv_heads;
    let head_dim = spec.head_dim;
    if query.rows == 0
        || query.cols != head_count * head_dim
        || key.rows != query.rows
        || value.rows != query.rows
        || key.cols != kv_head_count * head_dim
        || value.cols != key.cols
        || key.dtype != value.dtype
        || !matches!(key.dtype, MetalTensorDType::F16 | MetalTensorDType::Bf16)
        || (query.dtype != key.dtype && query.dtype != MetalTensorDType::F32)
        || kv_head_count == 0
        || !head_count.is_multiple_of(kv_head_count)
    {
        return Err(format!("GQA MetalTensor shape 非法: q=[{},{}], k=[{},{}], v=[{},{}], heads={head_count}, kv_heads={kv_head_count}, head_dim={head_dim}", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols));
    }

    gqa_prefill_attention_buffers(ctx, query, &key.buffer, 0, &value.buffer, 0, key.rows, 0, 0, 0, spec, None)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_prefill_attention_cached_tensor(ctx: &MetalContext, query: &MetalTensor, view: &MetalGqaCacheView, position: usize, spec: &GqaSpec) -> Result<MetalTensor, String> {
    if view.format == MetalKvCacheFormat::Int8 {
        return gqa_prefill_attention_q8_buffers(ctx, query, view, position, spec);
    }
    gqa_prefill_attention_buffers(ctx, query, &view.buffer, view.key_offset, &view.buffer, view.value_offset, view.rows, view.start, view.capacity, position, spec, None)
}

/// ICB 重放的 decode K/V append:单行 K/V 写入 ring 槽,slot=position%capacity 由 GPU
/// 从 decode_state 寻址,替代随 position 变化 offset 的 blit copy。state 槽 [position, kv_rows, kv_start]。
#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_kv_append_position_tensor(ctx: &MetalContext, view: &MetalGqaCacheView, key: &MetalTensor, value: &MetalTensor, decode_state: &metal::Buffer, state_offset: u64) -> Result<(), String> {
    if key.dtype != MetalTensorDType::F16 || value.dtype != MetalTensorDType::F16 || key.rows != 1 || value.rows != 1 || key.cols != value.cols {
        return Err(format!("decode append position 需要单行 F16 K/V，实际 K={:?}[{},{}] V={:?}[{},{}]", key.dtype, key.rows, key.cols, value.dtype, value.rows, value.cols));
    }
    if view.format != MetalKvCacheFormat::F16 || view.capacity == 0 {
        return Err("decode append position 只支持带 ring 容量的 F16 hybrid cache".to_owned());
    }
    let columns = validate_u32("decode append columns", key.cols)?;
    let capacity = validate_u32("decode append capacity", view.capacity)?;
    let shape = format!("columns={columns},capacity={capacity}");
    launch_1d(ctx, "gqa_kv_append_f16_position", &shape, key.cols, key.buffer.length() + value.buffer.length(), (key.cols * 4) as u64, |encoder| {
        encoder.set_buffer(0, Some(&key.buffer), 0);
        encoder.set_buffer(1, Some(&value.buffer), 0);
        encoder.set_buffer(2, Some(&view.buffer), view.key_offset);
        encoder.set_buffer(3, Some(&view.buffer), view.value_offset);
        set_bytes(encoder, 4, &columns);
        set_bytes(encoder, 5, &capacity);
        encoder.set_buffer(6, Some(decode_state), state_offset);
    })
}

/// 多行(verify 重放)append:K/V 是 [rows, columns],行 r 写 (state[0]+r)%capacity。
pub(crate) fn gqa_kv_append_rows_position_tensor(ctx: &MetalContext, view: &MetalGqaCacheView, key: &MetalTensor, value: &MetalTensor, decode_state: &metal::Buffer, state_offset: u64) -> Result<(), String> {
    if key.dtype != MetalTensorDType::F16 || value.dtype != MetalTensorDType::F16 || key.rows < 1 || key.rows != value.rows || key.cols != value.cols {
        return Err(format!("verify append position 需要同行数 F16 K/V，实际 K={:?}[{},{}] V={:?}[{},{}]", key.dtype, key.rows, key.cols, value.dtype, value.rows, value.cols));
    }
    if view.format != MetalKvCacheFormat::F16 || view.capacity == 0 {
        return Err("verify append position 只支持带 ring 容量的 F16 hybrid cache".to_owned());
    }
    let columns = validate_u32("verify append columns", key.cols)?;
    let capacity = validate_u32("verify append capacity", view.capacity)?;
    let rows = validate_u32("verify append rows", key.rows)?;
    let shape = format!("rows={rows},columns={columns},capacity={capacity}");
    launch_1d(ctx, "gqa_kv_append_rows_f16_position", &shape, key.len(), key.buffer.length() + value.buffer.length(), (key.len() * 4) as u64, |encoder| {
        encoder.set_buffer(0, Some(&key.buffer), 0);
        encoder.set_buffer(1, Some(&value.buffer), 0);
        encoder.set_buffer(2, Some(&view.buffer), view.key_offset);
        encoder.set_buffer(3, Some(&view.buffer), view.value_offset);
        set_bytes(encoder, 4, &columns);
        set_bytes(encoder, 5, &capacity);
        set_bytes(encoder, 6, &rows);
        encoder.set_buffer(7, Some(decode_state), state_offset);
    })
}

/// verify 重放的无 barrier 多行 attention(每 simdgroup 一个 head,TG=32;
/// segments≤16)。
#[cfg(test)]
pub(crate) fn gqa_decode_attention_nobar_position_tensor(ctx: &MetalContext, query: &MetalTensor, view: &MetalGqaCacheView, spec: &GqaSpec, decode_state: &metal::Buffer, state_offset: u64) -> Result<MetalTensor, String> {
    if query.dtype != MetalTensorDType::F16 || query.rows < 1 || query.rows > 4 {
        return Err(format!("nobar attention 需要 1-4 行 F16 query，实际 {:?}[{}]", query.dtype, query.rows));
    }
    if view.format != MetalKvCacheFormat::F16 || view.capacity == 0 {
        return Err("nobar attention 只支持 F16 hybrid cache".to_owned());
    }
    if !spec.head_dim.is_multiple_of(32) || spec.head_dim > 512 {
        return Err(format!("nobar attention head_dim={} 必须是 32 的倍数且 ≤512", spec.head_dim));
    }
    let pipeline = ctx.pipeline("gqa_decode_attention_nobar_f16_position")?;
    if 32 > pipeline.max_total_threads_per_threadgroup() {
        return Err("nobar attention 需要 32 threads/TG".to_owned());
    }
    let heads = validate_u32("nobar heads", spec.num_heads)?;
    let kv_heads = validate_u32("nobar kv heads", spec.num_kv_heads)?;
    let dimension = validate_u32("nobar head_dim", spec.head_dim)?;
    let query_rows = validate_u32("nobar rows", query.rows)?;
    if kv_heads == 0 || heads % kv_heads != 0 {
        return Err(format!("nobar heads={heads} kv_heads={kv_heads} 不匹配"));
    }
    let sliding_window = match spec.window {
        CausalWindow::Full => 0,
        CausalWindow::Sliding { size } => validate_u32("nobar window", size)?,
    };
    let capacity = validate_u32("nobar capacity", view.capacity)?;
    let score_scale = spec.score_scale;
    let output = ctx.tensor_kernel_output(query.rows, query.cols);
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&view.buffer), view.key_offset);
    encoder.set_buffer(2, Some(&view.buffer), view.value_offset);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    encoder.set_buffer(4, Some(decode_state), state_offset);
    set_bytes(&encoder, 5, &heads);
    set_bytes(&encoder, 6, &kv_heads);
    set_bytes(&encoder, 7, &dimension);
    set_bytes(&encoder, 8, &score_scale);
    set_bytes(&encoder, 9, &sliding_window);
    set_bytes(&encoder, 10, &capacity);
    set_bytes(&encoder, 11, &query_rows);
    encoder.dispatch_thread_groups(MTLSize::new(heads as u64, 1, 1), MTLSize::new(32, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={query_rows},heads={heads},dim={dimension}");
    ctx.commit_and_wait_profiled(&command, "gqa_decode_attention_nobar_f16_position", &shape, query.buffer.length() + view.buffer.length(), output.buffer.length());
    Ok(output)
}

/// verify 重放的 head-major 多行 attention(每 TG 一个 head,KV 读一次服务
/// 全部行;行数 ≤ 4)。
pub(crate) fn gqa_decode_attention_headmajor_position_tensor(ctx: &MetalContext, query: &MetalTensor, view: &MetalGqaCacheView, spec: &GqaSpec, decode_state: &metal::Buffer, state_offset: u64) -> Result<MetalTensor, String> {
    if query.dtype != MetalTensorDType::F16 || query.rows < 1 || query.rows > 4 {
        return Err(format!("head-major attention 需要 1-4 行 F16 query，实际 {:?}[{}]", query.dtype, query.rows));
    }
    if view.format != MetalKvCacheFormat::F16 || view.capacity == 0 {
        return Err("head-major attention 只支持 F16 hybrid cache".to_owned());
    }
    let threads = spec.head_dim.next_power_of_two();
    let pipeline = ctx.pipeline("gqa_decode_attention_headmajor_f16_position")?;
    if threads as u64 > pipeline.max_total_threads_per_threadgroup() {
        return Err(format!("head-major attention head_dim={} 需要 {threads} threads，超过 pipeline 上限", spec.head_dim));
    }
    let heads = validate_u32("head-major heads", spec.num_heads)?;
    let kv_heads = validate_u32("head-major kv heads", spec.num_kv_heads)?;
    let dimension = validate_u32("head-major head_dim", spec.head_dim)?;
    let query_rows = validate_u32("head-major rows", query.rows)?;
    if kv_heads == 0 || heads % kv_heads != 0 {
        return Err(format!("head-major heads={heads} kv_heads={kv_heads} 不匹配"));
    }
    let sliding_window = match spec.window {
        CausalWindow::Full => 0,
        CausalWindow::Sliding { size } => validate_u32("head-major window", size)?,
    };
    let capacity = validate_u32("head-major capacity", view.capacity)?;
    let score_scale = spec.score_scale;
    let output = ctx.tensor_kernel_output(query.rows, query.cols);
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&view.buffer), view.key_offset);
    encoder.set_buffer(2, Some(&view.buffer), view.value_offset);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    encoder.set_buffer(4, Some(decode_state), state_offset);
    set_bytes(&encoder, 5, &heads);
    set_bytes(&encoder, 6, &kv_heads);
    set_bytes(&encoder, 7, &dimension);
    set_bytes(&encoder, 8, &score_scale);
    set_bytes(&encoder, 9, &sliding_window);
    set_bytes(&encoder, 10, &capacity);
    set_bytes(&encoder, 11, &query_rows);
    encoder.dispatch_thread_groups(MTLSize::new(heads as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={query_rows},heads={heads},dim={dimension}");
    ctx.commit_and_wait_profiled(&command, "gqa_decode_attention_headmajor_f16_position", &shape, query.buffer.length() + view.buffer.length(), output.buffer.length());
    Ok(output)
}

/// 多行(verify 重放)attention:query 是 [rows, heads*dim],行 r 的 position=state[0]+r。
#[cfg(test)]
#[allow(dead_code)] // verify position 变体保留作 kernel 对拍入口。
pub(crate) fn gqa_decode_attention_rows_position_tensor(ctx: &MetalContext, query: &MetalTensor, view: &MetalGqaCacheView, spec: &GqaSpec, decode_state: &metal::Buffer, state_offset: u64) -> Result<MetalTensor, String> {
    if query.dtype != MetalTensorDType::F16 || query.rows < 1 {
        return Err(format!("verify attention position 需要非空 F16 query，实际 {:?}", query.dtype));
    }
    if view.format != MetalKvCacheFormat::F16 || view.capacity == 0 {
        return Err("verify attention position 只支持 F16 hybrid cache".to_owned());
    }
    let threads = spec.head_dim.next_power_of_two();
    let pipeline = ctx.pipeline("gqa_decode_attention_rows_f16_position")?;
    if threads as u64 > pipeline.max_total_threads_per_threadgroup() {
        return Err(format!("verify attention head_dim={} 需要 {threads} threads，超过 pipeline 上限", spec.head_dim));
    }
    let heads = validate_u32("verify attention heads", spec.num_heads)?;
    let kv_heads = validate_u32("verify attention kv heads", spec.num_kv_heads)?;
    let dimension = validate_u32("verify attention head_dim", spec.head_dim)?;
    let query_rows = validate_u32("verify attention query rows", query.rows)?;
    if kv_heads == 0 || heads % kv_heads != 0 {
        return Err(format!("verify attention heads={heads} kv_heads={kv_heads} 不匹配"));
    }
    let sliding_window = match spec.window {
        CausalWindow::Full => 0,
        CausalWindow::Sliding { size } => validate_u32("verify attention window", size)?,
    };
    let capacity = validate_u32("verify attention capacity", view.capacity)?;
    let score_scale = spec.score_scale;
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err(format!("verify attention score_scale={score_scale} 非法"));
    }
    let output = ctx.tensor_kernel_output(query.rows, query.cols);
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&view.buffer), view.key_offset);
    encoder.set_buffer(2, Some(&view.buffer), view.value_offset);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    encoder.set_buffer(4, Some(decode_state), state_offset);
    set_bytes(&encoder, 5, &heads);
    set_bytes(&encoder, 6, &kv_heads);
    set_bytes(&encoder, 7, &dimension);
    set_bytes(&encoder, 8, &score_scale);
    set_bytes(&encoder, 9, &sliding_window);
    set_bytes(&encoder, 10, &capacity);
    set_bytes(&encoder, 11, &query_rows);
    encoder.dispatch_thread_groups(MTLSize::new(heads as u64 * query_rows as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={query_rows},heads={heads},kv_heads={kv_heads},dim={dimension},window={sliding_window}");
    ctx.commit_and_wait_profiled(&command, "gqa_decode_attention_rows_f16_position", &shape, query.buffer.length() + view.buffer.length(), output.buffer.length());
    Ok(output)
}

/// ICB 重放的 decode 单 token attention:kv_rows/position/kv_start 由 decode_state 槽提供,/// ICB 重放的 decode 单 token attention:kv_rows/position/kv_start 由 decode_state 槽提供,
/// 几何与窗口常量照旧走 set_bytes(录制时固化)。view 是被读层的 gqa_layer_view。
pub(crate) fn gqa_decode_attention_position_tensor(ctx: &MetalContext, query: &MetalTensor, view: &MetalGqaCacheView, spec: &GqaSpec, decode_state: &metal::Buffer, state_offset: u64) -> Result<MetalTensor, String> {
    if query.dtype != MetalTensorDType::F16 || query.rows != 1 {
        return Err(format!("decode attention position 需要单行 F16 query，实际 {:?}", query.dtype));
    }
    if view.format != MetalKvCacheFormat::F16 {
        return Err("decode attention position 只支持 F16 hybrid cache".to_owned());
    }
    let threads = spec.head_dim.next_power_of_two();
    if threads > 1024 {
        return Err(format!("decode attention position head_dim={} 需要 {threads} threads，超出 reduction 容量", spec.head_dim));
    }
    // split-KV 版:simdgroup 按 token 切片、片内零 barrier( barrier 版逐 token 2 次
    // 全组同步,短中上下文 decode 实测 0.46ms/层)。要求 threads == head_dim
    // (sg_count == segments),非 2 的幂 head_dim 回落 barrier 版。
    let split_kv = spec.head_dim.is_power_of_two() && spec.head_dim <= 512;
    let pipeline = ctx.pipeline(if split_kv { "gqa_decode_attention_split_f16_position" } else { "gqa_decode_attention_f16_position" })?;
    if threads as u64 > pipeline.max_total_threads_per_threadgroup() {
        return Err(format!("decode attention head_dim={} 需要 {threads} threads，超过 pipeline 上限", spec.head_dim));
    }
    let heads = validate_u32("decode attention heads", spec.num_heads)?;
    let kv_heads = validate_u32("decode attention kv heads", spec.num_kv_heads)?;
    let dimension = validate_u32("decode attention head_dim", spec.head_dim)?;
    if kv_heads == 0 || heads % kv_heads != 0 {
        return Err(format!("decode attention position heads={heads} kv_heads={kv_heads} 不匹配"));
    }
    let sliding_window = match spec.window {
        CausalWindow::Full => 0,
        CausalWindow::Sliding { size } => validate_u32("decode attention window", size)?,
    };
    let capacity = validate_u32("decode attention capacity", view.capacity)?;
    if view.capacity == 0 {
        return Err("decode attention position 只支持带 ring 容量的 hybrid cache".to_owned());
    }
    let score_scale = spec.score_scale;
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err(format!("decode attention position score_scale={score_scale} 非法"));
    }
    let output = ctx.tensor_kernel_output(1, query.cols);
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&view.buffer), view.key_offset);
    encoder.set_buffer(2, Some(&view.buffer), view.value_offset);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    encoder.set_buffer(4, Some(decode_state), state_offset);
    set_bytes(&encoder, 5, &heads);
    set_bytes(&encoder, 6, &kv_heads);
    set_bytes(&encoder, 7, &dimension);
    set_bytes(&encoder, 8, &score_scale);
    set_bytes(&encoder, 9, &sliding_window);
    set_bytes(&encoder, 10, &capacity);
    encoder.dispatch_thread_groups(MTLSize::new(heads as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("heads={heads},kv_heads={kv_heads},dim={dimension},window={sliding_window}");
    ctx.commit_and_wait_profiled(&command, "gqa_decode_attention_f16_position", &shape, query.buffer.length() + view.buffer.length(), output.buffer.length());
    Ok(output)
}

/// Int8 cache 单行 decode 的融合入口:小 KV 直通 attention + 当前行 Q8 量化回写,
/// 一次 dispatch 取代 kv_quantize + attention 两次。调用前 cache 长度必须先推进
/// (reserve_layer_gqa_row),当前行内容完全由 kernel 写入。
#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_decode_attention_append_direct_q8_tensor(ctx: &MetalContext, query: &MetalTensor, new_key: &MetalTensor, new_value: &MetalTensor, view: &MetalGqaCacheView, position: usize, spec: &GqaSpec) -> Result<MetalTensor, String> {
    let key_scale_offset = view.key_scale_offset.ok_or("GQA direct-append 缺少 K scales")?;
    let value_scale_offset = view.value_scale_offset.ok_or("GQA direct-append 缺少 V scales")?;
    let first_visible = match spec.window {
        CausalWindow::Full => view.start,
        CausalWindow::Sliding { size } => view.start.max((position + 1).saturating_sub(size)),
    };
    decode::gqa_decode_attention_direct_q8_append_buffers(
        ctx,
        query,
        new_key,
        new_value,
        &view.buffer,
        view.key_offset,
        key_scale_offset,
        &view.buffer,
        view.value_offset,
        value_scale_offset,
        position + 1,
        first_visible,
        spec.num_heads,
        spec.num_kv_heads,
        spec.head_dim,
        spec.score_scale,
        view.group_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_prefill_attention_cached_visible_tensor(ctx: &MetalContext, query: &MetalTensor, view: &MetalGqaCacheView, position: usize, spec: &GqaSpec, visible_ends: &[u32]) -> Result<MetalTensor, String> {
    if view.format == MetalKvCacheFormat::Int8 {
        return Err("视觉块双向 prefill 暂不支持 INT8 GQA cache".to_owned());
    }
    gqa_prefill_attention_buffers(ctx, query, &view.buffer, view.key_offset, &view.buffer, view.value_offset, view.rows, view.start, view.capacity, position, spec, Some(visible_ends))
}

#[allow(clippy::too_many_arguments)]
fn gqa_prefill_attention_q8_buffers(ctx: &MetalContext, query: &MetalTensor, view: &MetalGqaCacheView, position: usize, spec: &GqaSpec) -> Result<MetalTensor, String> {
    let key_scale_offset = view.key_scale_offset.ok_or("GQA Q8 缺少 K scales")?;
    let value_scale_offset = view.value_scale_offset.ok_or("GQA Q8 缺少 V scales")?;
    if query.rows == 1 {
        let first_visible = match spec.window {
            CausalWindow::Full => view.start,
            CausalWindow::Sliding { size } => view.start.max(view.rows.saturating_sub(size)),
        };
        // 小 KV 直通:去 split/merge 的临时 buffer 与第二 encoder(24 层实测省 ~1ms/token)
        if view.rows - first_visible <= 256 && spec.head_dim <= 128 && spec.head_dim.is_multiple_of(32) && view.group_size.is_multiple_of(4) {
            return decode::gqa_decode_attention_direct_q8_buffers(
                ctx,
                query,
                &view.buffer,
                view.key_offset,
                key_scale_offset,
                &view.buffer,
                view.value_offset,
                value_scale_offset,
                view.rows,
                first_visible,
                view.capacity,
                spec.num_heads,
                spec.num_kv_heads,
                spec.head_dim,
                spec.score_scale,
                view.group_size,
            );
        }
        return gqa_decode_attention_split_kv_buffers_q8(
            ctx,
            query,
            &view.buffer,
            view.key_offset,
            key_scale_offset,
            &view.buffer,
            view.value_offset,
            value_scale_offset,
            view.rows,
            first_visible,
            view.capacity,
            spec.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            spec.score_scale,
            view.group_size,
        );
    }
    let storage_rows = if view.capacity == 0 { view.rows } else { view.capacity };
    let elements = storage_rows.checked_mul(spec.num_kv_heads).and_then(|count| count.checked_mul(spec.head_dim)).ok_or("GQA Q8 dequant elements 溢出")?;
    let key = ctx.shared_buffer_zeros(elements * mem::size_of::<f16>());
    let value = ctx.shared_buffer_zeros(elements * mem::size_of::<f16>());
    let pipeline = ctx.pipeline("gqa_kv_dequantize_q8")?;
    let rows = validate_u32("GQA Q8 dequant rows", storage_rows)?;
    let kv_heads = validate_u32("GQA Q8 dequant kv heads", spec.num_kv_heads)?;
    let head_dim = validate_u32("GQA Q8 dequant head dim", spec.head_dim)?;
    let group_size = validate_u32("GQA Q8 dequant group size", view.group_size)?;
    let groups_per_head = validate_u32("GQA Q8 dequant groups", spec.head_dim / view.group_size)?;
    let bf16 = u32::from(query.dtype == MetalTensorDType::Bf16);
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&view.buffer), view.key_offset);
    encoder.set_buffer(1, Some(&view.buffer), key_scale_offset);
    encoder.set_buffer(2, Some(&view.buffer), view.value_offset);
    encoder.set_buffer(3, Some(&view.buffer), value_scale_offset);
    encoder.set_buffer(4, Some(&key), 0);
    encoder.set_buffer(5, Some(&value), 0);
    set_bytes(&encoder, 6, &rows);
    set_bytes(&encoder, 7, &kv_heads);
    set_bytes(&encoder, 8, &head_dim);
    set_bytes(&encoder, 9, &group_size);
    set_bytes(&encoder, 10, &groups_per_head);
    set_bytes(&encoder, 11, &bf16);
    let width = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
    encoder.dispatch_threads(MTLSize::new(elements as u64, 1, 1), MTLSize::new(width, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command, "gqa_kv_dequantize_q8", &format!("rows={storage_rows},heads={},dim={}", spec.num_kv_heads, spec.head_dim), view.buffer.length(), key.length() + value.length());
    gqa_prefill_attention_buffers(ctx, query, &key, 0, &value, 0, view.rows, view.start, view.capacity, position, spec, None)
}

#[allow(clippy::too_many_arguments)]
fn gqa_prefill_attention_buffers(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &metal::Buffer,
    key_offset: u64,
    value: &metal::Buffer,
    value_offset: u64,
    kv_rows: usize,
    kv_start: usize,
    kv_capacity: usize,
    position: usize,
    spec: &GqaSpec,
    visible_ends: Option<&[u32]>,
) -> Result<MetalTensor, String> {
    let head_count = spec.num_heads;
    let kv_head_count = spec.num_kv_heads;
    let head_dim = spec.head_dim;
    if position.checked_add(query.rows) != Some(kv_rows) {
        return Err(format!("GQA cache 区间不连续: position={position}, query_rows={}, kv_rows={kv_rows}", query.rows));
    }
    if let Some(ends) = visible_ends {
        if ends.len() != query.rows {
            return Err(format!("GQA visible_ends={}，期望 {}", ends.len(), query.rows));
        }
        for (row, &end) in ends.iter().enumerate() {
            let causal_end = position.checked_add(row + 1).ok_or("GQA visible end 溢出")?.min(kv_rows);
            if end as usize > kv_rows || (end as usize) < causal_end {
                return Err(format!("GQA visible_ends[{row}]={end}，合法范围 {causal_end}..={kv_rows}"));
            }
        }
    }
    let storage_rows = if kv_capacity == 0 { kv_rows } else { kv_capacity };
    let kv_bytes = storage_rows.checked_mul(kv_head_count).and_then(|count| count.checked_mul(head_dim)).and_then(|count| count.checked_mul(2)).ok_or("GQA cache bytes 溢出")? as u64;
    if key_offset.checked_add(kv_bytes).is_none_or(|end| end > key.length()) || value_offset.checked_add(kv_bytes).is_none_or(|end| end > value.length()) {
        return Err("GQA cache buffer 长度不足".to_owned());
    }
    let standard_scale = 1.0 / (head_dim as f32).sqrt();
    let bf16 = query.dtype == MetalTensorDType::Bf16;
    if !matches!(query.dtype, MetalTensorDType::F16 | MetalTensorDType::Bf16 | MetalTensorDType::F32) {
        return Err(format!("GQA attention dtype={:?} 不受支持", query.dtype));
    }
    let f32_mps = query.dtype == MetalTensorDType::F32 && visible_ends.is_none() && query.rows >= 8192 && kv_start == 0 && matches!(spec.window, CausalWindow::Full) && (kv_capacity == 0 || kv_rows <= kv_capacity);
    if query.dtype == MetalTensorDType::F32 && !f32_mps {
        let query = to_f16_tensor(ctx, query)?;
        return gqa_prefill_attention_buffers(ctx, &query, key, key_offset, value, value_offset, kv_rows, kv_start, kv_capacity, position, spec, visible_ends);
    }
    let optimized_full = !bf16 && kv_start == 0 && kv_capacity == 0 && matches!(spec.window, CausalWindow::Full) && spec.score_scale == standard_scale;
    let contiguous_full = kv_start == 0 && matches!(spec.window, CausalWindow::Full) && (kv_capacity == 0 || kv_rows <= kv_capacity);
    let decode_first_visible = match spec.window {
        CausalWindow::Full => kv_start,
        CausalWindow::Sliding { size } => kv_start.max(kv_rows.saturating_sub(size)),
    };
    let decode_source_rows = kv_rows.saturating_sub(decode_first_visible);
    // 单行 decode 用 split-kv kernel(SIMD 归约、无逐 key 树归约):F16 与 BF16 都走,
    // 长 full attention(如 Gemma4 2k+ 上下文)落到逐 key 归约的 prefill kernel 会慢一倍
    if visible_ends.is_none() && query.rows == 1 && decode_source_rows >= 256 && (bf16 || query.dtype == MetalTensorDType::F16) {
        if bf16 && kv_head_count != 0 && head_count / kv_head_count == 4 && head_count.is_multiple_of(kv_head_count) && head_dim <= 512 && decode_source_rows <= 1024 {
            return gqa_decode_attention_parallel_bf16_buffers(ctx, query, key, key_offset, value, value_offset, kv_rows, decode_first_visible, kv_capacity, head_count, kv_head_count, head_dim, spec.score_scale);
        }
        return gqa_decode_attention_split_kv_buffers(ctx, query, key, key_offset, value, value_offset, kv_rows, decode_first_visible, kv_capacity, head_count, kv_head_count, head_dim, spec.score_scale, bf16, None);
    }
    if visible_ends.is_none() && bf16 && query.rows > 1 && contiguous_full {
        return gqa_prefill_attention_mps_bf16_buffers(ctx, query, key, key_offset, value, value_offset, kv_rows, position, head_count, kv_head_count, head_dim, spec.score_scale);
    }
    // F16 query + F16 cache 的连续 full attention(如 Gemma4 head_dim=512 层)没有
    // 对应容量的 tiled kernel,直接走 MPS 链路,scale 经 MPS alpha 传递不受标准值限制
    if visible_ends.is_none() && query.dtype == MetalTensorDType::F16 && query.rows > 1 && contiguous_full {
        return gqa_prefill_attention_mps_buffers(ctx, query, key, key_offset, value, value_offset, kv_rows, position, head_count, kv_head_count, head_dim, spec.score_scale);
    }
    // 滑窗层(如 Gemma4 head_dim=256 层)同样走窗口版 MPS,K/V 列范围平移到窗口内
    if visible_ends.is_none()
        && query.dtype == MetalTensorDType::F16
        && query.rows > 1
        && (kv_capacity == 0 || kv_rows <= kv_capacity)
        && let CausalWindow::Sliding { size } = spec.window
    {
        return gqa_prefill_attention_mps_windowed_buffers(ctx, query, key, key_offset, value, value_offset, kv_rows, position, head_count, kv_head_count, head_dim, spec.score_scale, size);
    }
    if visible_ends.is_none() && query.rows >= 8192 && optimized_full {
        return gqa_prefill_attention_mps_buffers(ctx, query, key, key_offset, value, value_offset, kv_rows, position, head_count, kv_head_count, head_dim, spec.score_scale);
    }
    // tiled kernel 的寄存器数组按 dtype 定容:F16 query_value[8] 只覆盖 head_dim ≤ 256,BF16 [16] 覆盖 ≤ 512。
    let tiled = visible_ends.is_none() && query.rows >= 8 && head_dim <= if bf16 { 512 } else { 256 };
    let kernel = match (bf16, tiled) {
        (false, false) => "gqa_prefill_attention_f16",
        (false, true) => "gqa_prefill_attention_tiled_f16",
        (true, false) => "gqa_prefill_attention_bf16",
        (true, true) => "gqa_prefill_attention_tiled_bf16",
    };
    let pipeline = ctx.pipeline(kernel)?;
    let threads = if tiled { 512 } else { head_dim.next_power_of_two() };
    if threads > 1024 || threads as u64 > pipeline.max_total_threads_per_threadgroup() {
        return Err(format!("GQA head_dim={head_dim} 需要 {threads} threads，超过 Metal pipeline 上限"));
    }
    let query_rows = u32::try_from(query.rows).map_err(|_| "GQA query rows 超过 u32".to_owned())?;
    let kv_rows = u32::try_from(kv_rows).map_err(|_| "GQA KV rows 超过 u32".to_owned())?;
    let query_position = u32::try_from(position).map_err(|_| "GQA position 超过 u32".to_owned())?;
    let heads = u32::try_from(head_count).map_err(|_| "GQA head_count 超过 u32".to_owned())?;
    let kv_heads = u32::try_from(kv_head_count).map_err(|_| "GQA kv_head_count 超过 u32".to_owned())?;
    let dimension = u32::try_from(head_dim).map_err(|_| "GQA head_dim 超过 u32".to_owned())?;
    let score_scale = spec.score_scale;
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err(format!("GQA score_scale={score_scale} 非法"));
    }
    let sliding_window = match spec.window {
        CausalWindow::Full => 0,
        CausalWindow::Sliding { size } => u32::try_from(size).map_err(|_| "GQA sliding window 超过 u32".to_owned())?,
    };
    let kv_start = u32::try_from(kv_start).map_err(|_| "GQA cache start 超过 u32".to_owned())?;
    let kv_capacity = u32::try_from(kv_capacity).map_err(|_| "GQA cache capacity 超过 u32".to_owned())?;
    let output = if bf16 { ctx.tensor_kernel_output_bf16(query.rows, query.cols) } else { ctx.tensor_kernel_output(query.rows, query.cols) };
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(key), key_offset);
    encoder.set_buffer(2, Some(value), value_offset);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, (&query_rows as *const u32).cast());
    encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, (&kv_rows as *const u32).cast());
    encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, (&query_position as *const u32).cast());
    encoder.set_bytes(7, std::mem::size_of::<u32>() as u64, (&heads as *const u32).cast());
    encoder.set_bytes(8, std::mem::size_of::<u32>() as u64, (&kv_heads as *const u32).cast());
    encoder.set_bytes(9, std::mem::size_of::<u32>() as u64, (&dimension as *const u32).cast());
    set_bytes(&encoder, 10, &score_scale);
    set_bytes(&encoder, 11, &sliding_window);
    set_bytes(&encoder, 12, &kv_start);
    set_bytes(&encoder, 13, &kv_capacity);
    if !tiled {
        let use_visible_ends = u32::from(visible_ends.is_some());
        let zero = 0u32;
        let visible_buffer = visible_ends.map(|ends| {
            let bytes = unsafe { std::slice::from_raw_parts(ends.as_ptr().cast::<u8>(), std::mem::size_of_val(ends)) };
            ctx.shared_buffer(bytes)
        });
        match visible_buffer.as_ref() {
            Some(buffer) => encoder.set_buffer(14, Some(buffer), 0),
            None => set_bytes(&encoder, 14, &zero),
        }
        set_bytes(&encoder, 15, &use_visible_ends);
    }
    // f16 tiled kernel 每个 simd group 承担 4 个连续 query 行,group 数相应收缩
    let rows_per_group: usize = if kernel == "gqa_prefill_attention_tiled_f16" { 4 } else { 1 };
    let query_groups = if tiled { query.rows.div_ceil((threads / 32) * rows_per_group) } else { query.rows };
    encoder.dispatch_thread_groups(metal::MTLSize { width: query_groups as u64 * head_count as u64, height: 1, depth: 1 }, metal::MTLSize { width: threads as u64, height: 1, depth: 1 });
    encoder.end_encoding();
    let shape = format!("q=[{},{}],kv_rows={kv_rows},position={position},heads={head_count},kv_heads={kv_head_count},head_dim={head_dim}", query.rows, query.cols);
    ctx.commit_and_wait_profiled(&command, kernel, &shape, query.buffer.length() + kv_bytes * 2, output.buffer.length());
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::cpu::gqa::gqa_prefill_attention_at_visible;
    use half::bf16;

    /// MiniCPM5 形状(16 heads / 2 kv / 128 dim)按行数扫描:17 行曾是实际观察到
    /// 的分叉点(2026-08-26 诊断)。
    #[test]
    fn gqa_prefill_matches_cpu_reference_row_scan() {
        let ctx = MetalContext::new_default().unwrap();
        let (heads, kv_heads, head_dim) = (16usize, 2usize, 128usize);
        let spec = crate::attention::gqa::GqaSpec {
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
            rope_dim: head_dim,
            rope_theta: 10000.0,
            use_qk_norm: false,
            window: crate::attention::gqa::CausalWindow::Full,
            score_scale: 1.0 / (head_dim as f32).sqrt(),
            output_gate: false,
        };
        for rows in [13usize, 15, 16, 17, 18, 24, 32] {
            let mut rng: u32 = 0xA77E;
            let mut next = || {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                (rng >> 8) as f32 / 8388608.0 - 1.0
            };
            let q: Vec<f32> = (0..rows * heads * head_dim).map(|_| next()).collect();
            let k: Vec<f32> = (0..rows * kv_heads * head_dim).map(|_| next()).collect();
            let v: Vec<f32> = (0..rows * kv_heads * head_dim).map(|_| next()).collect();
            let query = ctx.tensor_from_f32(&q, rows, heads * head_dim).unwrap();
            let key = ctx.tensor_from_f32(&k, rows, kv_heads * head_dim).unwrap();
            let value = ctx.tensor_from_f32(&v, rows, kv_heads * head_dim).unwrap();
            let actual = gqa_prefill_attention_tensor(&ctx, &query, &key, &value, &spec).unwrap();
            let actual = ctx.tensor_to_f32(&actual);
            let expected = crate::attention::gqa::reference_prefill(&q, &k, &v, &spec).unwrap();
            let mut worst = 0.0f32;
            for (a, e) in actual.iter().zip(&expected) {
                worst = worst.max((a - e).abs() / (1.0 + e.abs()));
            }
            eprintln!("rows={rows} worst_rel_err={worst:.4}");
            assert!(worst < 0.02, "rows={rows} worst_rel_err={worst}");
        }
    }

    /// Metal 块注意力对 CPU reference(attention_f32)逐元素一致;覆盖 GQA 分组、
    /// 跨 block 宽区间、窄区间与"整块不可见"(640..700 行只激活最后一个 block)。
    #[test]
    fn block_attention_matches_cpu_reference() {
        let ctx = MetalContext::new_default().unwrap();
        let (heads, kv_heads, head_dim) = (8usize, 2usize, 64usize);
        let (query_rows, kv_rows) = (6usize, 700usize);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut rng: u32 = 0xB10C;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0 - 1.0) * 2.0
        };
        let query_values = (0..query_rows * heads * head_dim).map(|_| next()).collect::<Vec<_>>();
        let key_values = (0..kv_rows * kv_heads * head_dim).map(|_| next()).collect::<Vec<_>>();
        let value_values = (0..kv_rows * kv_heads * head_dim).map(|_| next()).collect::<Vec<_>>();
        let visible = vec![0..kv_rows, 100..650, 0..1, 640..kv_rows, 255..260, 0..kv_rows];
        let spec = crate::attention::block::BlockAttentionSpec { geometry: crate::attention::gqa::GqaGeometry { num_heads: heads, num_kv_heads: kv_heads, head_dim }, score_scale: scale, visible };
        let query = ctx.tensor_from_f32(&query_values, query_rows, heads * head_dim).unwrap();
        let key = ctx.tensor_from_f32(&key_values, kv_rows, kv_heads * head_dim).unwrap();
        let value = ctx.tensor_from_f32(&value_values, kv_rows, kv_heads * head_dim).unwrap();
        let output = block_attention_tensor(&ctx, &query, &key, &value, &spec).unwrap();
        let actual = ctx.tensor_to_f32(&output);
        let expected = crate::attention::block::attention_f32(&query_values, &key_values, &value_values, query_rows, kv_rows, &spec).unwrap();
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() <= 2.0e-2 * (1.0 + expected.abs()), "index={index} actual={actual} expected={expected}");
        }
    }

    #[test]
    fn f16_mps_qk_scores_match_cpu() {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, heads, head_dim) = (300usize, 2usize, 64usize);
        let stride = rows.next_multiple_of(8);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut rng: u32 = 12345;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0 - 1.0) * 2.0
        };
        let query = ctx.tensor_from_f32(&(0..rows * heads * head_dim).map(|_| next()).collect::<Vec<_>>(), rows, heads * head_dim).unwrap();
        let key = ctx.tensor_from_f32(&(0..rows * head_dim).map(|_| next()).collect::<Vec<_>>(), rows, head_dim).unwrap();
        let value = ctx.tensor_from_f32(&(0..rows * head_dim).map(|_| next()).collect::<Vec<_>>(), rows, head_dim).unwrap();
        let query_f16 = ctx.tensor_to_f32(&query);
        let key_f16 = ctx.tensor_to_f32(&key);
        let value_f16 = ctx.tensor_to_f32(&value);
        let scores = ctx.shared_buffer_zeros(rows * stride * mem::size_of::<f32>());
        let probabilities = ctx.shared_buffer_zeros(rows * stride * mem::size_of::<f16>());
        let output = ctx.shared_buffer_zeros(rows * head_dim * mem::size_of::<f32>());
        let qk_command = ctx.command_buffer();
        super::super::mps::encode_f16_matmul_f32(
            &qk_command,
            &ctx.device,
            &query.buffer,
            0,
            heads * head_dim * mem::size_of::<f16>(),
            &key.buffer,
            0,
            head_dim * mem::size_of::<f16>(),
            &scores,
            0,
            stride * mem::size_of::<f32>(),
            rows,
            head_dim,
            rows,
            true,
            scale as f64,
        )
        .unwrap();
        ctx.commit_and_wait(&qk_command);
        let pipeline = ctx.pipeline("causal_softmax_rows_f32_f16").unwrap();
        let softmax_command = ctx.command_buffer();
        let encoder = softmax_command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&scores), 0);
        encoder.set_buffer(1, Some(&probabilities), 0);
        set_bytes(&encoder, 2, &(rows as u32));
        set_bytes(&encoder, 3, &(rows as u32));
        set_bytes(&encoder, 4, &(stride as u32));
        set_bytes(&encoder, 5, &0u32);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(32, 1, 1));
        encoder.end_encoding();
        ctx.commit_and_wait(&softmax_command);
        let actual_probabilities = ctx.read_f16_to_f32(&probabilities, rows * stride);
        let pv_command = ctx.command_buffer();
        super::super::mps::encode_f16_matmul_f32(
            &pv_command,
            &ctx.device,
            &probabilities,
            0,
            stride * mem::size_of::<f16>(),
            &value.buffer,
            0,
            head_dim * mem::size_of::<f16>(),
            &output,
            0,
            head_dim * mem::size_of::<f32>(),
            rows,
            rows,
            head_dim,
            false,
            1.0,
        )
        .unwrap();
        ctx.commit_and_wait(&pv_command);
        let actual = unsafe { std::slice::from_raw_parts(scores.contents().cast::<f32>(), rows * stride) };
        let actual_output = unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), rows * head_dim) };
        let mut error = 0.0f32;
        let mut reference = 0.0f32;
        for row in 0..rows {
            for column in 0..rows {
                let expected = (0..head_dim).map(|d| query_f16[row * heads * head_dim + d] * key_f16[column * head_dim + d]).sum::<f32>() * scale;
                error += (actual[row * stride + column] - expected).powi(2);
                reference += expected.powi(2);
            }
        }
        assert!((error / reference).sqrt() < 1.0e-2, "MPS QK score rel_l2={}", (error / reference).sqrt());
        let mut probability_error = 0.0f32;
        let mut probability_reference = 0.0f32;
        for row in 0..rows {
            let visible = row + 1;
            let maximum = actual[row * stride..row * stride + visible].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = actual[row * stride..row * stride + visible].iter().map(|score| (score - maximum).exp()).sum::<f32>();
            for column in 0..visible {
                let expected = (actual[row * stride + column] - maximum).exp() / denominator;
                probability_error += (actual_probabilities[row * stride + column] - expected).powi(2);
                probability_reference += expected.powi(2);
            }
        }
        assert!((probability_error / probability_reference).sqrt() < 1.0e-2, "MPS softmax rel_l2={}", (probability_error / probability_reference).sqrt());
        let mut output_error = 0.0f32;
        let mut output_reference = 0.0f32;
        for row in 0..rows {
            for dimension in 0..head_dim {
                let expected = (0..rows).map(|column| actual_probabilities[row * stride + column] * value_f16[column * head_dim + dimension]).sum::<f32>();
                output_error += (actual_output[row * head_dim + dimension] - expected).powi(2);
                output_reference += expected.powi(2);
            }
        }
        assert!((output_error / output_reference).sqrt() < 1.0e-2, "MPS PV rel_l2={}", (output_error / output_reference).sqrt());
    }

    #[test]
    fn f16_prefill_attention_matches_cpu() {
        // MPS F16 输入链路必须在较大 logits 下仍与 CPU 参考一致。
        let ctx = MetalContext::new_default().unwrap();
        let (rows, heads, kv_heads, head_dim) = (300usize, 2, 1, 64);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut rng: u32 = 12345;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0 - 1.0) * 2.0
        };
        let query: Vec<f32> = (0..rows * heads * head_dim).map(|_| next()).collect();
        let key: Vec<f32> = (0..rows * kv_heads * head_dim).map(|_| next()).collect();
        let value: Vec<f32> = (0..rows * kv_heads * head_dim).map(|_| next()).collect();
        let query = ctx.tensor_from_f32(&query, rows, heads * head_dim).unwrap();
        let key = ctx.tensor_from_f32(&key, rows, kv_heads * head_dim).unwrap();
        let value = ctx.tensor_from_f32(&value, rows, kv_heads * head_dim).unwrap();
        let spec = GqaSpec { num_heads: heads, num_kv_heads: kv_heads, head_dim, rope_dim: head_dim, rope_theta: 10000.0, use_qk_norm: false, window: CausalWindow::Full, score_scale: scale, output_gate: false };
        let output = gqa_prefill_attention_tensor(&ctx, &query, &key, &value, &spec).unwrap();
        let actual = ctx.read_f16_to_f32(&output.buffer, rows * heads * head_dim);
        let query = ctx.read_f16_to_f32(&query.buffer, rows * heads * head_dim);
        let key = ctx.read_f16_to_f32(&key.buffer, rows * kv_heads * head_dim);
        let value = ctx.read_f16_to_f32(&value.buffer, rows * kv_heads * head_dim);
        for row in 0..rows {
            for head in 0..heads {
                let mut scores = vec![0.0f32; row + 1];
                for (index, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += query[row * heads * head_dim + head * head_dim + d] * key[index * kv_heads * head_dim + d];
                    }
                    *score = dot * scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|s| (s - maximum).exp()).sum();
                for d in 0..head_dim {
                    let expected: f32 = scores.iter().enumerate().map(|(index, s)| (s - maximum).exp() / denominator * value[index * kv_heads * head_dim + d]).sum();
                    let got = actual[row * heads * head_dim + head * head_dim + d];
                    assert!((got - expected).abs() < 1.0e-2 * (1.0 + expected.abs()), "row={row} head={head} d={d}: {got} vs {expected}");
                }
            }
        }
    }

    #[test]
    fn bf16_sliding_visible_matches_cpu() {
        if metal::Device::system_default().is_none() {
            return;
        }

        let rows = 6;
        let head_count = 4;
        let kv_head_count = 2;
        let head_dim = 8;
        let query_columns = head_count * head_dim;
        let kv_columns = kv_head_count * head_dim;
        let rounded = |value: f32| bf16::from_f32(value).to_f32();
        let query = (0..rows * query_columns).map(|index| rounded(((index as f32 + 1.0) * 0.071).sin())).collect::<Vec<_>>();
        let key = (0..rows * kv_columns).map(|index| rounded(((index as f32 + 3.0) * 0.053).cos())).collect::<Vec<_>>();
        let value = (0..rows * kv_columns).map(|index| rounded(((index as f32 + 5.0) * 0.037).sin() * 0.75)).collect::<Vec<_>>();
        let visible_ends = [1, 5, 5, 5, 5, 6];
        let spec = GqaSpec {
            num_heads: head_count,
            num_kv_heads: kv_head_count,
            head_dim,
            rope_dim: head_dim,
            rope_theta: 10_000.0,
            use_qk_norm: false,
            score_scale: 1.0 / (head_dim as f32).sqrt(),
            window: CausalWindow::Sliding { size: 4 },
            output_gate: false,
        };

        let mut expected = vec![0.0; rows * query_columns];
        gqa_prefill_attention_at_visible(&query, &key, &value, 0, rows, 0, &spec, Some(&visible_ends), &mut expected);

        let ctx = MetalContext::new_default().unwrap();
        let query_tensor = ctx.tensor_from_f32_bf16(&query, rows, query_columns).unwrap();
        let key_tensor = ctx.tensor_from_f32_bf16(&key, rows, kv_columns).unwrap();
        let value_tensor = ctx.tensor_from_f32_bf16(&value, rows, kv_columns).unwrap();
        let actual = gqa_prefill_attention_buffers(&ctx, &query_tensor, &key_tensor.buffer, 0, &value_tensor.buffer, 0, rows, 0, 0, 0, &spec, Some(&visible_ends)).unwrap();
        let actual = ctx.tensor_to_f32(&actual);
        let error = actual.iter().zip(&expected).map(|(actual, expected)| (actual - expected).powi(2)).sum::<f32>().sqrt();
        let reference = expected.iter().map(|value| value.powi(2)).sum::<f32>().sqrt();
        assert!(error / reference < 0.01, "GQA BF16 visible rel_l2={}", error / reference);
    }

    #[test]
    fn f16_windowed_mps_sliding_prefill_matches_cpu() {
        if metal::Device::system_default().is_none() {
            return;
        }

        // 覆盖窗口版 MPS 链路:多行 F16 + Sliding 窗口,与 CPU 参考含 GQA 一致
        let rows = 300;
        let head_count = 4;
        let kv_head_count = 2;
        let head_dim = 64;
        let window = 40;
        let query_columns = head_count * head_dim;
        let kv_columns = kv_head_count * head_dim;
        let rounded = |value: f32| half::f16::from_f32(value).to_f32();
        let query = (0..rows * query_columns).map(|index| rounded(((index as f32 + 1.0) * 0.071).sin() * 2.0)).collect::<Vec<_>>();
        let key = (0..rows * kv_columns).map(|index| rounded(((index as f32 + 3.0) * 0.053).cos() * 2.0)).collect::<Vec<_>>();
        let value = (0..rows * kv_columns).map(|index| rounded(((index as f32 + 5.0) * 0.037).sin() * 0.75)).collect::<Vec<_>>();
        let spec = GqaSpec {
            num_heads: head_count,
            num_kv_heads: kv_head_count,
            head_dim,
            rope_dim: head_dim,
            rope_theta: 10_000.0,
            use_qk_norm: false,
            score_scale: 1.0 / (head_dim as f32).sqrt(),
            window: CausalWindow::Sliding { size: window },
            output_gate: false,
        };

        let mut expected = vec![0.0; rows * query_columns];
        gqa_prefill_attention_at_visible(&query, &key, &value, 0, rows, 0, &spec, None, &mut expected);

        let ctx = MetalContext::new_default().unwrap();
        let query_tensor = ctx.tensor_from_f32(&query, rows, query_columns).unwrap();
        let key_tensor = ctx.tensor_from_f32(&key, rows, kv_columns).unwrap();
        let value_tensor = ctx.tensor_from_f32(&value, rows, kv_columns).unwrap();
        let actual = gqa_prefill_attention_buffers(&ctx, &query_tensor, &key_tensor.buffer, 0, &value_tensor.buffer, 0, rows, 0, 0, 0, &spec, None).unwrap();
        let actual = ctx.tensor_to_f32(&actual);
        let error = actual.iter().zip(&expected).map(|(actual, expected)| (actual - expected).powi(2)).sum::<f32>().sqrt();
        let reference = expected.iter().map(|value| value.powi(2)).sum::<f32>().sqrt();
        assert!(error / reference < 0.01, "GQA F16 windowed MPS rel_l2={}", error / reference);
    }

    #[test]
    fn bf16_split_kv_decode_matches_cpu() {
        if metal::Device::system_default().is_none() {
            return;
        }

        for (kv_head_count, head_dim) in [(8, 256), (1, 512)] {
            let kv_rows = 294;
            let kv_capacity = 1024;
            let head_count = 16;
            let query_columns = head_count * head_dim;
            let kv_columns = kv_head_count * head_dim;
            let rounded = |value: f32| bf16::from_f32(value).to_f32();
            let query = (0..query_columns).map(|index| rounded(((index as f32 + 1.0) * 0.071).sin())).collect::<Vec<_>>();
            let key = (0..kv_capacity * kv_columns).map(|index| rounded(((index as f32 + 3.0) * 0.053).cos())).collect::<Vec<_>>();
            let value = (0..kv_capacity * kv_columns).map(|index| rounded(((index as f32 + 5.0) * 0.037).sin() * 0.75)).collect::<Vec<_>>();
            let spec = GqaSpec {
                num_heads: head_count,
                num_kv_heads: kv_head_count,
                head_dim,
                rope_dim: head_dim,
                rope_theta: 10_000.0,
                use_qk_norm: false,
                score_scale: 1.0,
                window: if kv_head_count == 1 { CausalWindow::Full } else { CausalWindow::Sliding { size: 1024 } },
                output_gate: false,
            };

            let mut expected = vec![0.0; query_columns];
            gqa_prefill_attention_at_visible(&query, &key, &value, 0, kv_rows, kv_rows - 1, &spec, None, &mut expected);

            let ctx = MetalContext::new_default().unwrap();
            let query_tensor = ctx.tensor_from_f32_bf16(&query, 1, query_columns).unwrap();
            let prefix_elements = 128;
            let mut cache = vec![0.0; prefix_elements];
            cache.extend_from_slice(&key);
            cache.extend_from_slice(&value);
            let cache_tensor = ctx.tensor_from_f32_bf16(&cache, 1, cache.len()).unwrap();
            let key_offset = (prefix_elements * std::mem::size_of::<bf16>()) as u64;
            let value_offset = key_offset + (kv_capacity * kv_columns * std::mem::size_of::<bf16>()) as u64;
            let actual = gqa_prefill_attention_buffers(&ctx, &query_tensor, &cache_tensor.buffer, key_offset, &cache_tensor.buffer, value_offset, kv_rows, 0, kv_capacity, kv_rows - 1, &spec, None).unwrap();
            let actual = ctx.tensor_to_f32(&actual);
            let error = actual.iter().zip(&expected).map(|(actual, expected)| (actual - expected).powi(2)).sum::<f32>().sqrt();
            let reference = expected.iter().map(|value| value.powi(2)).sum::<f32>().sqrt();
            assert!(error / reference < 0.01, "GQA BF16 split-KV decode kv_heads={kv_head_count}, head_dim={head_dim}, rel_l2={}", error / reference);
        }
    }

    /// 小 KV 直通 kernel 与 split+merge 路径在同一 Q8 cache 上逐元素一致。
    #[test]
    fn direct_q8_matches_split_kv() {
        let ctx = MetalContext::new_default().unwrap();
        let (heads, kv_heads, head_dim, group_size, rows) = (16usize, 2usize, 128usize, 64usize, 64usize);
        let groups_per_head = head_dim / group_size;
        let score_scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut rng: u32 = 0xD1E6;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0 - 0.5) * 2.0
        };
        let q: Vec<f32> = (0..heads * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..rows * kv_heads * head_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..rows * kv_heads * head_dim).map(|_| next()).collect();
        let codes_len = rows * kv_heads * head_dim;
        // CPU 对称 Q8 量化:布局 = [codes | scales(f16)],与 MetalGqaCacheView 的
        // codes/scales 分段一致(scales offset = codes_len)。
        let quant = |data: &[f32]| -> Vec<u8> {
            let mut out = vec![0u8; codes_len + rows * kv_heads * groups_per_head * 2];
            let (codes, scales) = out.split_at_mut(codes_len);
            for token in 0..rows {
                for kv in 0..kv_heads {
                    for g in 0..groups_per_head {
                        let base = (token * kv_heads + kv) * head_dim + g * group_size;
                        // 生产路径的量化输入是 F16(roped 输出),先舍入到 f16 再量化才逐位一致
                        let slice: Vec<f32> = data[base..base + group_size].iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
                        let s = (slice.iter().fold(0.0f32, |a, &b| a.max(b.abs())) / 127.0).max(1e-6);
                        for (i, &x) in slice.iter().enumerate() {
                            codes[base + i] = (x / s).round_ties_even().clamp(-127.0, 127.0) as i8 as u8;
                        }
                        let bits = half::f16::from_f32(s).to_bits();
                        let so = ((token * kv_heads + kv) * groups_per_head + g) * 2;
                        scales[so] = (bits & 0xff) as u8;
                        scales[so + 1] = (bits >> 8) as u8;
                    }
                }
            }
            out
        };
        let upload = |bytes: &[u8]| -> metal::Buffer {
            let buffer = ctx.shared_buffer_uninit(bytes.len());
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().cast::<u8>(), bytes.len()) };
            buffer
        };
        let key_buf = upload(&quant(&k));
        let value_buf = upload(&quant(&v));
        let query = ctx.tensor_from_f32(&q, 1, heads * head_dim).unwrap();
        let scale_offset = codes_len as u64;
        let split = gqa_decode_attention_split_kv_buffers_q8(&ctx, &query, &key_buf, 0, scale_offset, &value_buf, 0, scale_offset, rows, 0, 0, heads, kv_heads, head_dim, score_scale, group_size).unwrap();
        let direct = decode::gqa_decode_attention_direct_q8_buffers(&ctx, &query, &key_buf, 0, scale_offset, &value_buf, 0, scale_offset, rows, 0, 0, heads, kv_heads, head_dim, score_scale, group_size).unwrap();
        let expected = ctx.read_f16_to_f32(&split.buffer, heads * head_dim);
        let actual = ctx.read_f16_to_f32(&direct.buffer, heads * head_dim);
        let mut mismatches = 0usize;
        for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            if (a - e).abs() > 1.0e-2 * (1.0 + e.abs()) {
                if mismatches < 8 {
                    eprintln!("mismatch index={index} head={} dim={} direct={a} split={e}", index / head_dim, index % head_dim);
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "direct vs split 不一致 {mismatches}/{}", expected.len());
    }

    /// append 直通 kernel(含当前行量化融合)与 split+merge 在 kv>256 区间逐元素一致;
    /// 同时校验 kernel 内量化副本与 CPU 量化逐位一致(最后一行走 threadgroup 副本)。
    #[test]
    fn direct_q8_append_matches_split_kv() {
        let ctx = MetalContext::new_default().unwrap();
        let (heads, kv_heads, head_dim, group_size, rows) = (16usize, 2usize, 128usize, 64usize, 300usize);
        let groups_per_head = head_dim / group_size;
        let score_scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut rng: u32 = 0xA99E;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0 - 0.5) * 2.0
        };
        let q: Vec<f32> = (0..heads * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..rows * kv_heads * head_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..rows * kv_heads * head_dim).map(|_| next()).collect();
        let codes_len = rows * kv_heads * head_dim;
        let quant = |data: &[f32], upto: usize| -> Vec<u8> {
            let mut out = vec![0u8; codes_len + rows * kv_heads * groups_per_head * 2];
            let (codes, scales) = out.split_at_mut(codes_len);
            for token in 0..upto {
                for kv in 0..kv_heads {
                    for g in 0..groups_per_head {
                        let base = (token * kv_heads + kv) * head_dim + g * group_size;
                        // 生产路径的量化输入是 F16(roped 输出),先舍入到 f16 再量化才逐位一致
                        let slice: Vec<f32> = data[base..base + group_size].iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
                        let s = (slice.iter().fold(0.0f32, |a, &b| a.max(b.abs())) / 127.0).max(1e-6);
                        for (i, &x) in slice.iter().enumerate() {
                            codes[base + i] = (x / s).round_ties_even().clamp(-127.0, 127.0) as i8 as u8;
                        }
                        let bits = half::f16::from_f32(s).to_bits();
                        let so = ((token * kv_heads + kv) * groups_per_head + g) * 2;
                        scales[so] = (bits & 0xff) as u8;
                        scales[so + 1] = (bits >> 8) as u8;
                    }
                }
            }
            out
        };
        let upload = |bytes: &[u8]| -> metal::Buffer {
            let buffer = ctx.shared_buffer_uninit(bytes.len());
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().cast::<u8>(), bytes.len()) };
            buffer
        };
        let query = ctx.tensor_from_f32(&q, 1, heads * head_dim).unwrap();
        let scale_offset = codes_len as u64;
        // 参考:全部 300 行 CPU 量化后走 split+merge
        let split = gqa_decode_attention_split_kv_buffers_q8(&ctx, &query, &upload(&quant(&k, rows)), 0, scale_offset, &upload(&quant(&v, rows)), 0, scale_offset, rows, 0, 0, heads, kv_heads, head_dim, score_scale, group_size).unwrap();
        // append:前 299 行 CPU 量化进 cache,最后一行以原始 F16 交给 kernel 量化
        let key_buf = upload(&quant(&k, rows - 1));
        let value_buf = upload(&quant(&v, rows - 1));
        let new_key = ctx.tensor_from_f32(&k[(rows - 1) * kv_heads * head_dim..], 1, kv_heads * head_dim).unwrap();
        let new_value = ctx.tensor_from_f32(&v[(rows - 1) * kv_heads * head_dim..], 1, kv_heads * head_dim).unwrap();
        let direct = decode::gqa_decode_attention_direct_q8_append_buffers(&ctx, &query, &new_key, &new_value, &key_buf, 0, scale_offset, &value_buf, 0, scale_offset, rows, 0, heads, kv_heads, head_dim, score_scale, group_size).unwrap();
        let expected = ctx.read_f16_to_f32(&split.buffer, heads * head_dim);
        let actual = ctx.read_f16_to_f32(&direct.buffer, heads * head_dim);
        let mut worst = 0.0f32;
        for (&a, &e) in actual.iter().zip(&expected) {
            worst = worst.max((a - e).abs() / (1.0 + e.abs()));
        }
        eprintln!("append vs split worst_rel={worst:.5}");
        assert!(worst < 0.02, "append 与 split 不一致 worst_rel={worst}");
        // kernel 回写的最后一行量化结果与 CPU 量化逐位一致(rint vs round 的 .5 边界除外)
        let written = unsafe { std::slice::from_raw_parts(key_buf.contents().cast::<u8>(), codes_len) };
        let expected_codes = quant(&k, rows);
        let base = (rows - 1) * kv_heads * head_dim;
        let mut code_mismatch = 0usize;
        for i in 0..kv_heads * head_dim {
            if written[base + i] != expected_codes[base + i] {
                code_mismatch += 1;
            }
        }
        assert!(code_mismatch == 0, "append 量化回写与 CPU 不一致 {code_mismatch} 个 code");
    }
}
