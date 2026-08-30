//! MLA(Multi-head Latent Attention)的 prefill/decode kernel 与 weight-absorption decode。

pub mod dsa;
pub use dsa::*;

const MLA_SHADERS: &str = r#"
kernel void mla_prefill_attention_f16(
    device const half *q [[buffer(0)]],
    device const half *expanded_kv [[buffer(1)]],
    device const half *k_rope [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &q_columns [[buffer(5)]],
    constant uint &kv_columns [[buffer(6)]],
    constant uint &head_count [[buffer(7)]],
    constant uint &q_head_dim [[buffer(8)]],
    constant uint &kv_head_dim [[buffer(9)]],
    constant uint &rope_dim [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint head = group.x;
    const uint query = group.y;
    if (query >= rows || head >= head_count) return;
    const uint nope_dim = q_head_dim - rope_dim;
    const uint value_dim = kv_head_dim - nope_dim;
    const ulong q_base = (ulong)query * q_columns + head * q_head_dim;
    threadgroup float score_sum[256];
    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    float value_sum = 0.0f;

    for (uint key = 0; key <= query; ++key) {
        const ulong kv_base = (ulong)key * kv_columns + head * kv_head_dim;
        float partial = lane < nope_dim
            ? float(q[q_base + lane]) * float(expanded_kv[kv_base + lane])
            : 0.0f;
        if (lane < rope_dim) {
            partial += float(q[q_base + nope_dim + lane]) * float(k_rope[(ulong)key * rope_dim + lane]);
        }
        score_sum[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride >= 32; stride >>= 1) {
            if (lane < stride) score_sum[lane] += score_sum[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane < 32) {
            float reduced = score_sum[lane];
            reduced += simd_shuffle_down(reduced, 16);
            reduced += simd_shuffle_down(reduced, 8);
            reduced += simd_shuffle_down(reduced, 4);
            reduced += simd_shuffle_down(reduced, 2);
            reduced += simd_shuffle_down(reduced, 1);
            if (lane == 0) score_sum[0] = reduced;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float score = score_sum[0] * scale;
        const float next_maximum = max(maximum, score);
        const float previous_weight = exp(maximum - next_maximum);
        const float current_weight = exp(score - next_maximum);
        denominator = denominator * previous_weight + current_weight;
        const float value = lane < value_dim ? float(expanded_kv[kv_base + nope_dim + lane]) : 0.0f;
        value_sum = value_sum * previous_weight + current_weight * value;
        maximum = next_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < value_dim) {
        output[(ulong)query * q_columns + head * value_dim + lane] = half(value_sum / denominator);
    }
}
kernel void mla_prefill_attention_simd_f16(
    device const half *q [[buffer(0)]],
    device const half *expanded_kv [[buffer(1)]],
    device const half *k_rope [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &q_columns [[buffer(5)]],
    constant uint &kv_columns [[buffer(6)]],
    constant uint &head_count [[buffer(7)]],
    constant uint &q_head_dim [[buffer(8)]],
    constant uint &kv_head_dim [[buffer(9)]],
    constant uint &rope_dim [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint head = group.x;
    constexpr uint query_slot_count = 4;
    const uint query_begin = group.y * 64;
    uint queries[query_slot_count];
    bool active[query_slot_count];
    ulong q_base[query_slot_count];
    for (uint query_slot = 0; query_slot < query_slot_count; ++query_slot) {
        queries[query_slot] =
            query_begin + simd_index + query_slot * 16;
        active[query_slot] = head < head_count && queries[query_slot] < rows;
        q_base[query_slot] =
            (ulong)queries[query_slot] * q_columns + head * q_head_dim;
    }
    const uint nope_dim = q_head_dim - rope_dim;
    const uint value_dim = kv_head_dim - nope_dim;
    const uint maximum_key = min(query_begin + 63, rows - 1);

    constexpr uint key_tile_size = 16;
    threadgroup half key_tile[key_tile_size * 256];
    threadgroup half value_tile[key_tile_size * 256];
    half4 query_values_lo[query_slot_count];
    half4 query_values_hi[query_slot_count];
    float4 value_sums_lo[query_slot_count];
    float4 value_sums_hi[query_slot_count];
    float maximum[query_slot_count];
    float denominator[query_slot_count];
    for (uint query_slot = 0; query_slot < query_slot_count; ++query_slot) {
        query_values_lo[query_slot] = half4(0.0h);
        query_values_hi[query_slot] = half4(0.0h);
        for (uint slot = 0; slot < 4; ++slot) {
            const uint lo_dim = lane + slot * 32;
            const uint hi_dim = lo_dim + 128;
            query_values_lo[query_slot][slot] =
                active[query_slot] && lo_dim < q_head_dim
                    ? q[q_base[query_slot] + lo_dim]
                    : half(0.0h);
            query_values_hi[query_slot][slot] =
                active[query_slot] && hi_dim < q_head_dim
                    ? q[q_base[query_slot] + hi_dim]
                    : half(0.0h);
        }
        value_sums_lo[query_slot] = float4(0.0f);
        value_sums_hi[query_slot] = float4(0.0f);
        maximum[query_slot] = -3.402823466e+38f;
        denominator[query_slot] = 0.0f;
    }
    for (uint key_begin = 0; key_begin <= maximum_key; key_begin += key_tile_size) {
        const uint loader = thread_index & 63;
        const uint dim_begin = loader * 4;
        for (uint load_round = 0; load_round < key_tile_size / 8; ++load_round) {
            const uint key_offset = (thread_index >> 6) + load_round * 8;
            const uint key = key_begin + key_offset;
            half4 values = half4(0.0h);
            if (key <= maximum_key) {
                const ulong kv_base =
                    (ulong)key * kv_columns + head * kv_head_dim;
                if (dim_begin + 3 < q_head_dim
                    && dim_begin + 3 < nope_dim) {
                    device const half4 *source =
                        reinterpret_cast<device const half4 *>(
                            expanded_kv + kv_base + dim_begin);
                    values = *source;
                } else if (dim_begin < q_head_dim
                    && dim_begin >= nope_dim
                    && dim_begin + 3 < q_head_dim) {
                    device const half4 *source =
                        reinterpret_cast<device const half4 *>(
                            k_rope
                                + (ulong)key * rope_dim
                                + dim_begin - nope_dim);
                    values = *source;
                } else {
                    for (uint item = 0; item < 4; ++item) {
                        const uint dim = dim_begin + item;
                        if (dim < q_head_dim) {
                            values[item] = dim < nope_dim
                                ? expanded_kv[kv_base + dim]
                                : k_rope[
                                    (ulong)key * rope_dim
                                    + dim - nope_dim
                                ];
                        }
                    }
                }
            }
            for (uint item = 0; item < 4; ++item) {
                const uint dim = dim_begin + item;
                const uint tile_index =
                    key_offset * 256 + (dim & 31) * 8 + (dim >> 5);
                key_tile[tile_index] = values[item];
            }

            half4 value_values = half4(0.0h);
            if (key <= maximum_key) {
                const ulong kv_base =
                    (ulong)key * kv_columns + head * kv_head_dim;
                if (dim_begin + 3 < value_dim) {
                    device const half4 *source =
                        reinterpret_cast<device const half4 *>(
                            expanded_kv
                                + kv_base + nope_dim + dim_begin);
                    value_values = *source;
                } else {
                    for (uint item = 0; item < 4; ++item) {
                        const uint dim = dim_begin + item;
                        if (dim < value_dim) {
                            value_values[item] =
                                expanded_kv[
                                    kv_base + nope_dim + dim
                                ];
                        }
                    }
                }
            }
            for (uint item = 0; item < 4; ++item) {
                const uint dim = dim_begin + item;
                const uint tile_index =
                    key_offset * 256 + (dim & 31) * 8 + (dim >> 5);
                value_tile[tile_index] = value_values[item];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key_offset = 0; key_offset < key_tile_size; ++key_offset) {
            const uint key = key_begin + key_offset;
            threadgroup const half4 *key_vectors =
                reinterpret_cast<threadgroup const half4 *>(
                    key_tile + key_offset * 256 + lane * 8);
            threadgroup const half4 *value_vectors =
                reinterpret_cast<threadgroup const half4 *>(
                    value_tile + key_offset * 256 + lane * 8);
            const half4 key_values_lo = key_vectors[0];
            const half4 key_values_hi = key_vectors[1];
            const float4 current_values_lo =
                float4(value_vectors[0]);
            const float4 current_values_hi =
                float4(value_vectors[1]);
            for (uint query_slot = 0; query_slot < query_slot_count; ++query_slot) {
                if (!active[query_slot]
                    || key > queries[query_slot]
                    || key > maximum_key) {
                    continue;
                }
                const float partial =
                    float(dot(
                        query_values_lo[query_slot],
                        key_values_lo))
                    + float(dot(
                        query_values_hi[query_slot],
                        key_values_hi));
                const float score = simd_sum(partial) * scale;
                const float next_maximum =
                    max(maximum[query_slot], score);
                const bool score_wins =
                    score > maximum[query_slot];
                const float decay =
                    fast::exp(-fabs(score - maximum[query_slot]));
                if (score_wins) {
                    denominator[query_slot] =
                        fma(denominator[query_slot], decay, 1.0f);
                    value_sums_lo[query_slot] = fma(
                        value_sums_lo[query_slot],
                        float4(decay),
                        current_values_lo);
                    value_sums_hi[query_slot] = fma(
                        value_sums_hi[query_slot],
                        float4(decay),
                        current_values_hi);
                } else {
                    denominator[query_slot] += decay;
                    value_sums_lo[query_slot] = fma(
                        current_values_lo,
                        float4(decay),
                        value_sums_lo[query_slot]);
                    value_sums_hi[query_slot] = fma(
                        current_values_hi,
                        float4(decay),
                        value_sums_hi[query_slot]);
                }
                maximum[query_slot] = next_maximum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint query_slot = 0; query_slot < query_slot_count; ++query_slot) {
        if (!active[query_slot]) {
            continue;
        }
        const ulong output_base =
            (ulong)queries[query_slot] * q_columns + head * value_dim;
        for (uint slot = 0; slot < 4; ++slot) {
            const uint lo_dim = lane + slot * 32;
            const uint hi_dim = lo_dim + 128;
            if (lo_dim < value_dim) {
                output[output_base + lo_dim] =
                    half(value_sums_lo[query_slot][slot]
                        / denominator[query_slot]);
            }
            if (hi_dim < value_dim) {
                output[output_base + hi_dim] =
                    half(value_sums_hi[query_slot][slot]
                        / denominator[query_slot]);
            }
        }
    }
}
kernel void mla_kv_quantize_f16(
    device const half *latent_input [[buffer(0)]],
    device uchar *codes             [[buffer(1)]],
    device half   *scales           [[buffer(2)]],
    constant uint &rows             [[buffer(3)]],
    constant uint &kv_lora_rank     [[buffer(4)]],
    constant uint &group_size       [[buffer(5)]],
    constant uint &quant_bits       [[buffer(6)]],
    device const half *rope_input   [[buffer(7)]],
    device half   *rope_output      [[buffer(8)]],
    constant uint &qk_rope_head_dim [[buffer(9)]],
    uint group_index [[thread_position_in_grid]])
{
    if (quant_bits < 2 || quant_bits > 8 || group_size == 0) return;
    uint groups_per_row = kv_lora_rank / group_size;
    if (groups_per_row == 0 || kv_lora_rank % group_size != 0) return;
    uint total_groups = rows * groups_per_row;

    // rope 拷贝:每个线程顺带拷若干元素(总元素数可能 > total_groups,跨组循环)。
    uint rope_elements = rows * qk_rope_head_dim;
    for (uint index = group_index; index < rope_elements; index += total_groups) {
        rope_output[index] = rope_input[index];
    }

    if (group_index >= total_groups) return;

    uint row = group_index / groups_per_row;
    uint group = group_index - row * groups_per_row;
    uint base = (ulong)row * kv_lora_rank + (ulong)group * group_size;

    float maximum = 0.0f;
    for (uint index = 0; index < group_size; ++index) {
        float value = fabs(float(latent_input[base + index]));
        if (value > maximum) maximum = value;
    }

    uint qmax = (1u << (quant_bits - 1)) - 1u;
    float scale = maximum > 0.0f ? (maximum / float(qmax)) : 1.0f;
    scales[group_index] = half(scale);

    float inverse_scale = scale == 0.0f ? 0.0f : (1.0f / scale);
    // zllm 当前固定 INT8(quant_bits=8);i4 路径留以后,这里只实现 8-bit。
    for (uint index = 0; index < group_size; ++index) {
        int quantized_value = (int)rint(float(latent_input[base + index]) * inverse_scale);
        quantized_value = clamp(quantized_value, -int(qmax), int(qmax));
        codes[base + index] = uchar(quantized_value & 0xff);
    }
}
kernel void mla_kv_dequantize_f16(
    device const uchar *codes  [[buffer(0)]],
    device const half  *scales [[buffer(1)]],
    device half *output        [[buffer(2)]],
    constant uint &rows        [[buffer(3)]],
    constant uint &kv_lora_rank [[buffer(4)]],
    constant uint &group_size  [[buffer(5)]],
    constant uint &quant_bits  [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    uint total = rows * kv_lora_rank;
    if (index >= total || group_size == 0 || kv_lora_rank % group_size != 0) return;

    uint row = index / kv_lora_rank;
    uint column = index - row * kv_lora_rank;
    uint groups_per_row = kv_lora_rank / group_size;
    uint scale_index = row * groups_per_row + column / group_size;

    int quantized_value = int(char(codes[index]));  // INT8 路径
    output[index] = half(float(quantized_value) * float(scales[scale_index]));
}
kernel void mla_decode_q_latent_fp8_f16(
    device const half *q                 [[buffer(0)]],   // [head_count, q_head_dim](nope 在前,rope 在后)
    device const uchar *kv_b_weights     [[buffer(1)]],   // FP8 codes 或 f16 字节流(看 weight_is_fp8)
    device const float *kv_b_scale_inv   [[buffer(2)]],   // F32 scale_inv(fp8 时用;f16 时忽略)
    device half *q_latent                [[buffer(3)]],   // 输出 [head_count, kv_lora]
    constant uint &head_count            [[buffer(4)]],
    constant uint &qk_nope_dim           [[buffer(5)]],
    constant uint &rope_dim              [[buffer(6)]],   // q_head_dim = qk_nope + rope
    constant uint &kv_head_dim           [[buffer(7)]],   // = qk_nope + value_dim(kv_b_proj 每 head 行数)
    constant uint &kv_lora_dim           [[buffer(8)]],
    constant uint &scale_columns         [[buffer(9)]],   // = ceil(kv_lora_dim / 128)
    constant uint &weight_is_fp8         [[buffer(10)]],  // 1=fp8+scale_inv;0=f16 buffer(dense 用)
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (head >= head_count) return;
    const uint q_head_dim = qk_nope_dim + rope_dim;
    for (uint slot = 0; slot < 2; ++slot) {
        const uint latent = lane + slot * 256;
        if (latent >= kv_lora_dim) continue;
        float sum = 0.0f;
        for (uint nope = 0; nope < qk_nope_dim; ++nope) {
            const uint weight_row = head * kv_head_dim + nope;
            const ulong w_idx = (ulong)weight_row * kv_lora_dim + latent;
            float w;
            if (weight_is_fp8 != 0) {
                const uint scale_index = (weight_row / 128) * scale_columns + latent / 128;
                w = decode_f8_e4m3(kv_b_weights[w_idx]) * kv_b_scale_inv[scale_index];
            } else {
                // f16:2 字节/元素,小端读 u16 → half。
                const uint16_t bits = uint16_t(kv_b_weights[w_idx * 2]) | (uint16_t(kv_b_weights[w_idx * 2 + 1]) << 8);
                w = float(as_type<half>(bits));
            }
            sum += float(q[head * q_head_dim + nope]) * w;
        }
        q_latent[head * kv_lora_dim + latent] = half(sum);
    }
}
kernel void mla_decode_attention_i8_latent(
    device const half *q_latent          [[buffer(0)]],   // [head_count, kv_lora_dim](吸收后的 query)
    device const char *kv_codes          [[buffer(1)]],   // cache i8 codes [position, kv_lora_dim]
    device const half *kv_scales         [[buffer(2)]],   // cache f16 scales [position, kv_lora_dim/group_size]
    device const half *q_rope            [[buffer(3)]],   // 完整 q [head_count, q_head_dim]
    device const half *k_rope            [[buffer(4)]],   // [position, rope_dim]
    device half *latent_output           [[buffer(5)]],   // 输出 [head_count, kv_lora_dim]
    constant uint &rows                  [[buffer(6)]],   // position(历史 token 数)
    constant uint &kv_lora_dim           [[buffer(7)]],
    constant uint &group_size            [[buffer(8)]],
    constant uint &rope_dim              [[buffer(9)]],
    constant float &scale                [[buffer(10)]],  // attention scale = 1/sqrt(q_head_dim)
    constant uint &q_head_dim            [[buffer(11)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (rows == 0) return;
    const uint groups_per_row = kv_lora_dim / group_size;
    threadgroup float score_sum[256];
    const ulong q_latent_base = (ulong)head * kv_lora_dim;

    // 每 lane 负责 kv_lora_dim 的两列(column_0, column_1)用于 value 累加。
    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    float value_sum_0 = 0.0f;
    float value_sum_1 = 0.0f;
    const uint column_0 = lane;
    const uint column_1 = lane + 256;
    const bool has_col_0 = column_0 < kv_lora_dim;
    const bool has_col_1 = column_1 < kv_lora_dim;

    for (uint key = 0; key < rows; ++key) {
        float partial = 0.0f;
        for (uint c = lane; c < kv_lora_dim; c += 256) {
            const uint g = c / group_size;
            const float latent = float(kv_codes[(ulong)key * kv_lora_dim + c])
                * float(kv_scales[(ulong)key * groups_per_row + g]);
            partial += float(q_latent[q_latent_base + c]) * latent;
        }
        // rope 部分:只 lane < rope_dim 累加(每 lane 一列 rope)。
        if (lane < rope_dim) {
            partial += float(q_rope[head * q_head_dim + q_head_dim - rope_dim + lane]) * float(k_rope[(ulong)key * rope_dim + lane]);
        }
        score_sum[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 256-thread reduce。
        for (uint stride = 128; stride >= 32; stride >>= 1) {
            if (lane < stride) score_sum[lane] += score_sum[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane < 32) {
            float r = score_sum[lane];
            r += simd_shuffle_down(r, 16);
            r += simd_shuffle_down(r, 8);
            r += simd_shuffle_down(r, 4);
            r += simd_shuffle_down(r, 2);
            r += simd_shuffle_down(r, 1);
            if (lane == 0) score_sum[0] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float score = score_sum[0] * scale;
        const float next_maximum = max(maximum, score);
        const float previous_weight = exp(maximum - next_maximum);
        const float current_weight = exp(score - next_maximum);
        denominator = denominator * previous_weight + current_weight;

        // value 累加:直接用 cache 的 i8 latent 当 value(因为 context = W_V · Σ a_t L_t)。
        if (has_col_0) {
            const float lv0 = float(kv_codes[(ulong)key * kv_lora_dim + column_0])
                * float(kv_scales[(ulong)key * groups_per_row + column_0 / group_size]);
            value_sum_0 = value_sum_0 * previous_weight + current_weight * lv0;
        }
        if (has_col_1) {
            const float lv1 = float(kv_codes[(ulong)key * kv_lora_dim + column_1])
                * float(kv_scales[(ulong)key * groups_per_row + column_1 / group_size]);
            value_sum_1 = value_sum_1 * previous_weight + current_weight * lv1;
        }
        maximum = next_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (has_col_0) {
        latent_output[q_latent_base + column_0] = half(value_sum_0 / denominator);
    }
    if (has_col_1) {
        latent_output[q_latent_base + column_1] = half(value_sum_1 / denominator);
    }
}
kernel void mla_decode_attention_f16_latent(
    device const half *q_latent          [[buffer(0)]],
    device const half *kv_latent         [[buffer(1)]],
    device const half *q_rope            [[buffer(2)]],
    device const half *k_rope            [[buffer(3)]],
    device half *latent_output           [[buffer(4)]],
    constant uint &rows                  [[buffer(5)]],
    constant uint &kv_lora_dim           [[buffer(6)]],
    constant uint &rope_dim              [[buffer(7)]],
    constant float &scale                [[buffer(8)]],
    constant uint &q_head_dim            [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (rows == 0) return;
    threadgroup float score_sum[256];
    const ulong q_latent_base = (ulong)head * kv_lora_dim;
    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    float value_sum_0 = 0.0f;
    float value_sum_1 = 0.0f;
    const uint column_0 = lane;
    const uint column_1 = lane + 256;
    const bool has_col_0 = column_0 < kv_lora_dim;
    const bool has_col_1 = column_1 < kv_lora_dim;

    for (uint key = 0; key < rows; ++key) {
        float partial = 0.0f;
        for (uint column = lane; column < kv_lora_dim; column += 256) {
            partial += float(q_latent[q_latent_base + column])
                * float(kv_latent[(ulong)key * kv_lora_dim + column]);
        }
        if (lane < rope_dim) {
            partial += float(q_rope[head * q_head_dim + q_head_dim - rope_dim + lane])
                * float(k_rope[(ulong)key * rope_dim + lane]);
        }
        score_sum[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride >= 32; stride >>= 1) {
            if (lane < stride) score_sum[lane] += score_sum[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane < 32) {
            float reduced = score_sum[lane];
            reduced += simd_shuffle_down(reduced, 16);
            reduced += simd_shuffle_down(reduced, 8);
            reduced += simd_shuffle_down(reduced, 4);
            reduced += simd_shuffle_down(reduced, 2);
            reduced += simd_shuffle_down(reduced, 1);
            if (lane == 0) score_sum[0] = reduced;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float score = score_sum[0] * scale;
        const float next_maximum = max(maximum, score);
        const float previous_weight = exp(maximum - next_maximum);
        const float current_weight = exp(score - next_maximum);
        denominator = denominator * previous_weight + current_weight;
        if (has_col_0) {
            const float value = float(kv_latent[(ulong)key * kv_lora_dim + column_0]);
            value_sum_0 = value_sum_0 * previous_weight + current_weight * value;
        }
        if (has_col_1) {
            const float value = float(kv_latent[(ulong)key * kv_lora_dim + column_1]);
            value_sum_1 = value_sum_1 * previous_weight + current_weight * value;
        }
        maximum = next_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (has_col_0) latent_output[q_latent_base + column_0] = half(value_sum_0 / denominator);
    if (has_col_1) latent_output[q_latent_base + column_1] = half(value_sum_1 / denominator);
}
kernel void mla_decode_v_fp8_f16(
    device const half *latent_output     [[buffer(0)]],   // [head_count, kv_lora_dim](attention 输出)
    device const uchar *kv_b_weights     [[buffer(1)]],   // FP8 codes 或 f16 字节流(看 weight_is_fp8)
    device const float *kv_b_scale_inv   [[buffer(2)]],   // F32 scale_inv(fp8 时用;f16 时忽略)
    device half *output                  [[buffer(3)]],   // 输出 [head_count, value_dim]
    constant uint &head_count            [[buffer(4)]],
    constant uint &qk_nope_dim           [[buffer(5)]],
    constant uint &kv_head_dim           [[buffer(6)]],
    constant uint &kv_lora_dim           [[buffer(7)]],
    constant uint &value_dim             [[buffer(8)]],
    constant uint &scale_columns         [[buffer(9)]],
    constant uint &weight_is_fp8         [[buffer(10)]],  // 1=fp8+scale_inv;0=f16 buffer(dense 用)
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (head >= head_count || lane >= value_dim) return;
    const uint weight_row = head * kv_head_dim + qk_nope_dim + lane;
    const ulong latent_base = (ulong)head * kv_lora_dim;
    const ulong weight_base = (ulong)weight_row * kv_lora_dim;
    const uint scale_base = (weight_row / 128) * scale_columns;
    float sum = 0.0f;
    for (uint latent = 0; latent < kv_lora_dim; ++latent) {
        float w;
        if (weight_is_fp8 != 0) {
            const float s = kv_b_scale_inv[scale_base + latent / 128];
            w = decode_f8_e4m3(kv_b_weights[weight_base + latent]) * s;
        } else {
            // f16:2 字节/元素,小端读 u16 → half。
            const uint16_t bits = uint16_t(kv_b_weights[(weight_base + latent) * 2])
                | (uint16_t(kv_b_weights[(weight_base + latent) * 2 + 1]) << 8);
            w = float(as_type<half>(bits));
        }
        sum += float(latent_output[latent_base + latent]) * w;
    }
    output[head * value_dim + lane] = half(sum);
}
kernel void mla_prefill_attention_selected_f16(
    device const half *q                [[buffer(0)]],
    device const half *compressed_kv    [[buffer(1)]],
    device const half *k_rope           [[buffer(2)]],
    device const uint *indices          [[buffer(3)]],
    device const half *kv_b             [[buffer(4)]],
    device half *output                 [[buffer(5)]],
    constant uint &rows                 [[buffer(6)]],
    constant uint &top_k                [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint head_count = zllm_fc_u32_0;
    const uint qk_nope = zllm_fc_u32_1;
    const uint rope_dim = zllm_fc_u32_2;
    const uint value_dim = zllm_fc_u32_3;
    const uint q_head_dim = zllm_fc_u32_4;
    const uint kv_head_dim = zllm_fc_u32_5;
    const uint kv_lora = zllm_fc_u32_6;
    const float attention_scale = rsqrt(float(q_head_dim));
    const uint head = group.x * 8 + simd_group;
    const uint query_row = group.y;
    if (head >= head_count || query_row >= rows) return;
    const ulong q_base = ulong(query_row) * head_count * q_head_dim + ulong(head) * q_head_dim;
    float q_latent[16];
    float context[16];
    for (uint part = 0; part < 16; ++part) {
        const uint latent = simd_lane + part * 32;
        float sum = 0.0f;
        if (latent < kv_lora) {
            for (uint nope = 0; nope < qk_nope; ++nope) {
                const uint weight_row = head * kv_head_dim + nope;
                sum += float(q[q_base + nope]) * float(kv_b[ulong(weight_row) * kv_lora + latent]);
            }
        }
        q_latent[part] = sum;
        context[part] = 0.0f;
    }
    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    threadgroup half latent_tile[8][512];
    threadgroup half rope_tile[8][128];
    const uint count = min(top_k, query_row + 1);
    const ulong index_base = ulong(query_row) * top_k;
    for (uint tile_begin = 0; tile_begin < count; tile_begin += 8) {
        const uint tile_count = min(8u, count - tile_begin);
        for (uint element = thread_index; element < tile_count * kv_lora; element += 256) {
            const uint tile = element / kv_lora;
            const uint latent = element - tile * kv_lora;
            const uint key_row = indices[index_base + tile_begin + tile];
            latent_tile[tile][latent] = compressed_kv[ulong(key_row) * kv_lora + latent];
        }
        for (uint element = thread_index; element < tile_count * rope_dim; element += 256) {
            const uint tile = element / rope_dim;
            const uint rope = element - tile * rope_dim;
            const uint key_row = indices[index_base + tile_begin + tile];
            rope_tile[tile][rope] = k_rope[ulong(key_row) * rope_dim + rope];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint tile = 0; tile < tile_count; ++tile) {
            float partial = 0.0f;
            for (uint part = 0; part < 16; ++part) {
                const uint latent = simd_lane + part * 32;
                if (latent < kv_lora) partial += q_latent[part] * float(latent_tile[tile][latent]);
            }
            for (uint rope = simd_lane; rope < rope_dim; rope += 32) {
                partial += float(q[q_base + qk_nope + rope]) * float(rope_tile[tile][rope]);
            }
            const float score = simd_sum(partial) * attention_scale;
            float previous_weight = 0.0f;
            float current_weight = 0.0f;
            if (simd_lane == 0) {
                const float next_maximum = max(maximum, score);
                previous_weight = exp(maximum - next_maximum);
                current_weight = exp(score - next_maximum);
                denominator = denominator * previous_weight + current_weight;
                maximum = next_maximum;
            }
            previous_weight = simd_broadcast_first(previous_weight);
            current_weight = simd_broadcast_first(current_weight);
            for (uint part = 0; part < 16; ++part) {
                const uint latent = simd_lane + part * 32;
                if (latent < kv_lora) context[part] = context[part] * previous_weight + float(latent_tile[tile][latent]) * current_weight;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_denominator = simd_lane == 0 ? 1.0f / denominator : 0.0f;
    inverse_denominator = simd_broadcast_first(inverse_denominator);
    for (uint part = 0; part < 16; ++part) context[part] *= inverse_denominator;
    for (uint value = 0; value < value_dim; ++value) {
        const uint weight_row = head * kv_head_dim + qk_nope + value;
        const ulong weight_base = ulong(weight_row) * kv_lora;
        float partial = 0.0f;
        for (uint part = 0; part < 16; ++part) {
            const uint latent = simd_lane + part * 32;
            if (latent < kv_lora) partial += context[part] * float(kv_b[weight_base + latent]);
        }
        const float result = simd_sum(partial);
        if (simd_lane == 0) {
            output[ulong(query_row) * head_count * value_dim + ulong(head) * value_dim + value] = half(result);
        }
    }
}
kernel void mla_prefill_attention_selected_fp8_f16(
    device const half *q                [[buffer(0)]],
    device const half *compressed_kv    [[buffer(1)]],
    device const half *k_rope           [[buffer(2)]],
    device const uint *indices          [[buffer(3)]],
    device const uchar *kv_b            [[buffer(4)]],
    device const float *kv_b_scale_inv  [[buffer(5)]],
    device half *output                 [[buffer(6)]],
    constant uint &rows                 [[buffer(7)]],
    constant uint &top_k                [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint head_count = zllm_fc_u32_0;
    const uint qk_nope = zllm_fc_u32_1;
    const uint rope_dim = zllm_fc_u32_2;
    const uint value_dim = zllm_fc_u32_3;
    const uint q_head_dim = zllm_fc_u32_4;
    const uint kv_head_dim = zllm_fc_u32_5;
    const uint kv_lora = zllm_fc_u32_6;
    const uint scale_columns = zllm_fc_u32_7;
    const float attention_scale = rsqrt(float(q_head_dim));
    const uint head = group.x * 8 + simd_group;
    const uint query_row = group.y;
    if (head >= head_count || query_row >= rows) return;
    const ulong q_base = ulong(query_row) * head_count * q_head_dim + ulong(head) * q_head_dim;
    float q_latent[16];
    float context[16];
    for (uint part = 0; part < 16; ++part) {
        const uint latent = simd_lane + part * 32;
        float sum = 0.0f;
        if (latent < kv_lora) {
            for (uint nope = 0; nope < qk_nope; ++nope) {
                const uint weight_row = head * kv_head_dim + nope;
                const uchar code = kv_b[ulong(weight_row) * kv_lora + latent];
                const uint scale_index = (weight_row / 128) * scale_columns + latent / 128;
                sum += float(q[q_base + nope]) * decode_f8_e4m3(code) * kv_b_scale_inv[scale_index];
            }
        }
        q_latent[part] = sum;
        context[part] = 0.0f;
    }
    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    threadgroup half latent_tile[8][512];
    threadgroup half rope_tile[8][128];
    const uint count = min(top_k, query_row + 1);
    const ulong index_base = ulong(query_row) * top_k;
    for (uint tile_begin = 0; tile_begin < count; tile_begin += 8) {
        const uint tile_count = min(8u, count - tile_begin);
        for (uint element = thread_index; element < tile_count * kv_lora; element += 256) {
            const uint tile = element / kv_lora;
            const uint latent = element - tile * kv_lora;
            const uint key_row = indices[index_base + tile_begin + tile];
            latent_tile[tile][latent] = compressed_kv[ulong(key_row) * kv_lora + latent];
        }
        for (uint element = thread_index; element < tile_count * rope_dim; element += 256) {
            const uint tile = element / rope_dim;
            const uint rope = element - tile * rope_dim;
            const uint key_row = indices[index_base + tile_begin + tile];
            rope_tile[tile][rope] = k_rope[ulong(key_row) * rope_dim + rope];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint tile = 0; tile < tile_count; ++tile) {
            float partial = 0.0f;
            for (uint part = 0; part < 16; ++part) {
                const uint latent = simd_lane + part * 32;
                if (latent < kv_lora) partial += q_latent[part] * float(latent_tile[tile][latent]);
            }
            for (uint rope = simd_lane; rope < rope_dim; rope += 32) {
                partial += float(q[q_base + qk_nope + rope]) * float(rope_tile[tile][rope]);
            }
            const float score = simd_sum(partial) * attention_scale;
            float previous_weight = 0.0f;
            float current_weight = 0.0f;
            if (simd_lane == 0) {
                const float next_maximum = max(maximum, score);
                previous_weight = exp(maximum - next_maximum);
                current_weight = exp(score - next_maximum);
                denominator = denominator * previous_weight + current_weight;
                maximum = next_maximum;
            }
            previous_weight = simd_broadcast_first(previous_weight);
            current_weight = simd_broadcast_first(current_weight);
            for (uint part = 0; part < 16; ++part) {
                const uint latent = simd_lane + part * 32;
                if (latent < kv_lora) context[part] = context[part] * previous_weight + float(latent_tile[tile][latent]) * current_weight;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_denominator = simd_lane == 0 ? 1.0f / denominator : 0.0f;
    inverse_denominator = simd_broadcast_first(inverse_denominator);
    for (uint part = 0; part < 16; ++part) context[part] *= inverse_denominator;
    for (uint value = 0; value < value_dim; ++value) {
        const uint weight_row = head * kv_head_dim + qk_nope + value;
        const ulong weight_base = ulong(weight_row) * kv_lora;
        const uint scale_base = (weight_row / 128) * scale_columns;
        float partial = 0.0f;
        for (uint part = 0; part < 16; ++part) {
            const uint latent = simd_lane + part * 32;
            if (latent < kv_lora) {
                partial += context[part] * decode_f8_e4m3(kv_b[weight_base + latent]) * kv_b_scale_inv[scale_base + latent / 128];
            }
        }
        const float result = simd_sum(partial);
        if (simd_lane == 0) {
            output[ulong(query_row) * head_count * value_dim + ulong(head) * value_dim + value] = half(result);
        }
    }
}
kernel void mla_decode_attention_f16_latent_selected(
    device const half *q_latent          [[buffer(0)]],
    device const half *kv_latent         [[buffer(1)]],
    device const half *q_rope            [[buffer(2)]],
    device const half *k_rope            [[buffer(3)]],
    device const uint *selection         [[buffer(4)]],
    device float *partial_values         [[buffer(5)]],
    device float *partial_stats          [[buffer(6)]],
    constant uint &selected_rows         [[buffer(7)]],
    constant uint &kv_lora_dim           [[buffer(8)]],
    constant uint &rope_dim              [[buffer(9)]],
    constant float &scale                [[buffer(10)]],
    constant uint &q_head_dim            [[buffer(11)]],
    constant uint &head_count            [[buffer(12)]],
    constant uint &chunk_size            [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (selected_rows == 0) return;
    const uint head_group = group.x;
    const uint chunk = group.y;
    const uint selected_begin = chunk * chunk_size;
    const uint selected_end = min(selected_rows, selected_begin + chunk_size);
    if (selected_begin >= selected_end) return;
    const uint head = head_group * 8 + simd_group;
    const bool active = head < head_count;
    const ulong q_latent_base = (ulong)head * kv_lora_dim;
    const uint latent_parts = (kv_lora_dim + 31) / 32;
    float value_sum[16];
    for (uint part = 0; part < 16; ++part) value_sum[part] = 0.0f;

    threadgroup half latent_tile[512];
    threadgroup half rope_tile[128];
    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;

    for (uint selected = selected_begin; selected < selected_end; ++selected) {
        const uint key = selection[selected];
        for (uint column = thread_index; column < kv_lora_dim; column += 256) {
            latent_tile[column] = kv_latent[(ulong)key * kv_lora_dim + column];
        }
        for (uint column = thread_index; column < rope_dim; column += 256) {
            rope_tile[column] = k_rope[(ulong)key * rope_dim + column];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float partial = 0.0f;
        if (active) {
            for (uint part = 0; part < latent_parts; ++part) {
                const uint column = simd_lane + part * 32;
                if (column < kv_lora_dim) {
                    partial += float(q_latent[q_latent_base + column])
                        * float(latent_tile[column]);
                }
            }
            for (uint column = simd_lane; column < rope_dim; column += 32) {
                partial += float(q_rope[head * q_head_dim + q_head_dim - rope_dim + column])
                    * float(rope_tile[column]);
            }
        }

        const float score = simd_sum(partial) * scale;
        const float next_maximum = max(maximum, score);
        const float previous_weight = exp(maximum - next_maximum);
        const float current_weight = exp(score - next_maximum);
        denominator = denominator * previous_weight + current_weight;
        for (uint part = 0; part < latent_parts; ++part) {
            const uint column = simd_lane + part * 32;
            if (column < kv_lora_dim) {
                value_sum[part] = value_sum[part] * previous_weight
                    + current_weight * float(latent_tile[column]);
            }
        }
        maximum = next_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        const ulong partial_base =
            (ulong(chunk) * head_count + head) * kv_lora_dim;
        for (uint part = 0; part < latent_parts; ++part) {
            const uint column = simd_lane + part * 32;
            if (column < kv_lora_dim) {
                partial_values[partial_base + column] = value_sum[part];
            }
        }
        if (simd_lane == 0) {
            const ulong stats_base =
                (ulong(chunk) * head_count + head) * 2;
            partial_stats[stats_base] = maximum;
            partial_stats[stats_base + 1] = denominator;
        }
    }
}
kernel void mla_decode_attention_i8_latent_selected(
    device const half *q_latent          [[buffer(0)]],
    device const char *kv_codes          [[buffer(1)]],
    device const half *kv_scales         [[buffer(2)]],
    device const half *q_rope            [[buffer(3)]],
    device const half *k_rope            [[buffer(4)]],
    device const uint *selection         [[buffer(5)]],
    device float *partial_values         [[buffer(6)]],
    device float *partial_stats          [[buffer(7)]],
    constant uint &selected_rows         [[buffer(8)]],
    constant uint &kv_lora_dim           [[buffer(9)]],
    constant uint &group_size            [[buffer(10)]],
    constant uint &rope_dim              [[buffer(11)]],
    constant float &scale                [[buffer(12)]],
    constant uint &q_head_dim            [[buffer(13)]],
    constant uint &head_count            [[buffer(14)]],
    constant uint &chunk_size            [[buffer(15)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (selected_rows == 0 || group_size == 0) return;
    const uint groups_per_row = kv_lora_dim / group_size;
    if (groups_per_row == 0 || groups_per_row > 128 || kv_lora_dim % group_size != 0) return;
    const uint selected_begin = group.y * chunk_size;
    const uint selected_end = min(selected_rows, selected_begin + chunk_size);
    if (selected_begin >= selected_end) return;

    const uint head = group.x * 8 + simd_group;
    const bool active = head < head_count;
    const ulong q_latent_base = ulong(head) * kv_lora_dim;
    const uint latent_parts = (kv_lora_dim + 31) / 32;
    threadgroup half latent_tile[512];
    threadgroup half rope_tile[128];

    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    float value_sum[16];
    for (uint part = 0; part < 16; ++part) value_sum[part] = 0.0f;

    for (uint selected = selected_begin; selected < selected_end; ++selected) {
        const uint key = selection[selected];
        for (uint column = thread_index; column < kv_lora_dim; column += 256) {
            const uint quant_group = column / group_size;
            latent_tile[column] = half(
                float(kv_codes[ulong(key) * kv_lora_dim + column])
                * float(kv_scales[ulong(key) * groups_per_row + quant_group]));
        }
        for (uint column = thread_index; column < rope_dim; column += 256) {
            rope_tile[column] = k_rope[ulong(key) * rope_dim + column];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float partial = 0.0f;
        if (active) {
            for (uint part = 0; part < latent_parts; ++part) {
                const uint column = simd_lane + part * 32;
                if (column < kv_lora_dim) {
                    partial += float(q_latent[q_latent_base + column]) * float(latent_tile[column]);
                }
            }
            for (uint column = simd_lane; column < rope_dim; column += 32) {
                partial += float(q_rope[head * q_head_dim + q_head_dim - rope_dim + column])
                    * float(rope_tile[column]);
            }
        }

        const float score = simd_sum(partial) * scale;
        const float next_maximum = max(maximum, score);
        const float previous_weight = exp(maximum - next_maximum);
        const float current_weight = exp(score - next_maximum);
        denominator = denominator * previous_weight + current_weight;
        for (uint part = 0; part < latent_parts; ++part) {
            const uint column = simd_lane + part * 32;
            if (column < kv_lora_dim) {
                value_sum[part] = value_sum[part] * previous_weight
                    + current_weight * float(latent_tile[column]);
            }
        }
        maximum = next_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        const ulong partial_base = (ulong(group.y) * head_count + head) * kv_lora_dim;
        for (uint part = 0; part < latent_parts; ++part) {
            const uint column = simd_lane + part * 32;
            if (column < kv_lora_dim) partial_values[partial_base + column] = value_sum[part];
        }
        if (simd_lane == 0) {
            const ulong stats_base = (ulong(group.y) * head_count + head) * 2;
            partial_stats[stats_base] = maximum;
            partial_stats[stats_base + 1] = denominator;
        }
    }
}
kernel void mla_decode_attention_f16_latent_selected_merge(
    device const float *partial_values   [[buffer(0)]],
    device const float *partial_stats    [[buffer(1)]],
    device half *latent_output           [[buffer(2)]],
    constant uint &chunk_count           [[buffer(3)]],
    constant uint &head_count            [[buffer(4)]],
    constant uint &kv_lora_dim           [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (head >= head_count || chunk_count == 0) return;
    const uint column_0 = lane;
    const uint column_1 = lane + 256;
    const bool has_column_0 = column_0 < kv_lora_dim;
    const bool has_column_1 = column_1 < kv_lora_dim;
    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    float value_0 = 0.0f;
    float value_1 = 0.0f;

    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
        const ulong chunk_head = ulong(chunk) * head_count + head;
        const float chunk_maximum = partial_stats[chunk_head * 2];
        const float chunk_denominator = partial_stats[chunk_head * 2 + 1];
        const float next_maximum = max(maximum, chunk_maximum);
        const float previous_weight = exp(maximum - next_maximum);
        const float chunk_weight = exp(chunk_maximum - next_maximum);
        denominator =
            denominator * previous_weight + chunk_denominator * chunk_weight;
        const ulong partial_base = chunk_head * kv_lora_dim;
        if (has_column_0) {
            value_0 = value_0 * previous_weight
                + partial_values[partial_base + column_0] * chunk_weight;
        }
        if (has_column_1) {
            value_1 = value_1 * previous_weight
                + partial_values[partial_base + column_1] * chunk_weight;
        }
        maximum = next_maximum;
    }

    const ulong output_base = ulong(head) * kv_lora_dim;
    if (has_column_0) {
        latent_output[output_base + column_0] = half(value_0 / denominator);
    }
    if (has_column_1) {
        latent_output[output_base + column_1] = half(value_1 / denominator);
    }
}
"#;

use crate::backend::metal::api as metal;

use super::attention::mla_attention_tensor;
use super::dense::matmul_tensor_resident_weight;
use super::fp8::{fp8_matmul_tensor, fp8_matmul_tensor_resident};
use super::{Fp8Matrix, MTLSize, MetalContext, MetalKvCache, MetalKvCacheFormat, MetalTensor, MlaSpec, THREADS, f16, mem, set_bytes, validate_u32};

use std::sync::OnceLock;

/// 聚合 MLA 与 DSA 子模块 shader,供 `super::kernels_source()` 拼接。
/// kernel 之间互不调用,家族私有 helper 在各文件内先于 kernel 定义。
pub fn shaders() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| [MLA_SHADERS, dsa::SHADERS].concat()).as_str()
}

/// device-resident 的 cache attention:从 cache 反量化 latent + rope(全程 GPU)→
/// fp8_matmul 重建 expanded_kv → `mla_attention_tensor`。q 是本批 `[n_new, q_columns]`。
///
/// 当前范围(prefill 接入):第一次 prefill 时 cache 为空,append 后 layer_len == n_new,
/// q 与 expanded_kv 行数相等,现有 causal kernel 语义正确。decode(query=1 看全部历史)
/// 需要不同 kernel,留范围 D。
#[allow(clippy::too_many_arguments)]
pub fn mla_attention_with_cache_tensor(ctx: &MetalContext, q: &MetalTensor, cache: &MetalKvCache, kv_b: KvBWeight<'_>, head_count: usize, rope_dim: usize, layer: usize) -> Result<MetalTensor, String> {
    let layer_len = cache.layer_len(layer);
    if layer_len == 0 {
        return Err("mla_attention_with_cache_tensor: layer_len=0,需先 append 本批 latent".to_owned());
    }
    if layer_len < q.rows {
        return Err(format!("layer_len {layer_len} < q.rows {}:cache 未 append 完整本批", q.rows));
    }
    // 1) GPU 反量化:latent [layer_len, kv_lora_rank] + rope [layer_len, rope_dim]。
    let (latent, k_rope) = cache.dequant_layer_mla_tensor(ctx, layer)?;
    // 2) latent × kv_b_proj → expanded_kv [layer_len, kv_projection_size]。FP8(MoE)或 f16(dense)。
    let expanded_kv = match kv_b {
        KvBWeight::Fp8(m) => fp8_matmul_tensor(ctx, &latent, m)?,
        KvBWeight::F16(w) => {
            let kv_proj = w.len() / latent.cols;
            let weight_t = ctx.tensor_from_f32(w, kv_proj, latent.cols).map_err(|e| format!("dense kv_b 上传: {e}"))?;
            matmul_tensor_resident_weight(ctx, &latent, &weight_t)?
        }
        KvBWeight::Fp8Resident { codes, scale_inv, rows, cols } => fp8_matmul_tensor_resident(ctx, &latent, codes, scale_inv, rows, cols)?,
        KvBWeight::F16Resident(weight) => matmul_tensor_resident_weight(ctx, &latent, weight)?,
        KvBWeight::W8A16Resident { packed, scales, scale_dtype, group_size, rows, cols } => super::low_bit::w8a16_matmul_tensor_resident(ctx, &latent, packed, scales, scale_dtype, group_size, rows, cols)?,
    };
    // 3) causal attention。prefill 内 layer_len == q.rows,kernel 的 `for key in 0..=query`
    //    上界 = q.rows,覆盖全部 kv。
    mla_attention_tensor(ctx, q, &expanded_kv, &k_rope, head_count, rope_dim)
}

pub fn layernorm_bias_tensor(ctx: &MetalContext, input: &MetalTensor, weight: &MetalTensor, bias: &MetalTensor, eps: f32) -> Result<MetalTensor, String> {
    if input.rows == 0 || input.cols == 0 {
        return Err(format!("LayerNorm+bias 要求非空，实际 [{},{}]", input.rows, input.cols));
    }
    if weight.len() != input.cols || bias.len() != input.cols {
        return Err(format!("LayerNorm+bias 权重长度不符: input={} weight={} bias={}", input.cols, weight.len(), bias.len()));
    }
    let columns = validate_u32("LayerNorm columns", input.cols)?;
    let (output, kernel) = match input.dtype {
        crate::backend::metal::MetalTensorDType::F16 => (ctx.tensor_zeros(input.rows, input.cols), if input.cols <= THREADS { "layernorm_bias_f16" } else { "layernorm_bias_wide_f16" }),
        crate::backend::metal::MetalTensorDType::Bf16 => (ctx.tensor_zeros_bf16(input.rows, input.cols), if input.cols <= THREADS { "layernorm_bias_bf16" } else { "layernorm_bias_wide_bf16" }),
        crate::backend::metal::MetalTensorDType::F32 => return Err("LayerNorm+bias 暂不支持 F32 activation".to_owned()),
    };
    let pipeline = ctx.pipeline(kernel)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&weight.buffer), 0);
    encoder.set_buffer(2, Some(&bias.buffer), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &columns);
    set_bytes(&encoder, 5, &eps);
    encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("columns={}", input.cols);
    ctx.commit_and_wait_profiled(&command, kernel, &shape, input.buffer.length() + weight.buffer.length() + bias.buffer.length(), output.buffer.length());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn mla_prefill_attention_selected_f16(ctx: &MetalContext, query: &MetalTensor, latent: &MetalTensor, k_rope: &MetalTensor, kv_b: &MetalTensor, selection: &metal::Buffer, top_k: usize, spec: &MlaSpec) -> Result<MetalTensor, String> {
    let q_head = spec.q_head_dim();
    let kv_head = spec.kv_head_dim();
    let value_dim = spec.value_dim();
    if query.rows == 0
        || query.cols != spec.num_heads * q_head
        || latent.rows != query.rows
        || latent.cols != spec.kv_lora_rank
        || k_rope.rows != query.rows
        || k_rope.cols != spec.qk_rope_head_dim
        || kv_b.rows != spec.num_heads * kv_head
        || kv_b.cols != spec.kv_lora_rank
        || spec.kv_lora_rank > 512
        || spec.qk_rope_head_dim > 128
    {
        return Err(format!("selected F16 MLA 形状不符: q=[{},{}] latent=[{},{}] rope=[{},{}] weight=[{},{}]", query.rows, query.cols, latent.rows, latent.cols, k_rope.rows, k_rope.cols, kv_b.rows, kv_b.cols));
    }
    let rows = validate_u32("selected MLA rows", query.rows)?;
    let top_k_u32 = validate_u32("selected MLA top_k", top_k)?;
    let constants = [
        validate_u32("selected MLA heads", spec.num_heads)?,
        validate_u32("selected MLA nope", spec.qk_nope_dim())?,
        validate_u32("selected MLA rope", spec.qk_rope_head_dim)?,
        validate_u32("selected MLA value", value_dim)?,
        validate_u32("selected MLA q_head", q_head)?,
        validate_u32("selected MLA kv_head", kv_head)?,
        validate_u32("selected MLA latent", spec.kv_lora_rank)?,
    ];
    let output = ctx.shared_buffer_zeros(query.rows * spec.num_heads * value_dim * mem::size_of::<f16>());
    let output = MetalTensor::new(output, query.rows, spec.num_heads * value_dim);
    let pipeline = ctx.pipeline_u32_constants("mla_prefill_attention_selected_f16", &constants)?;
    if pipeline.max_total_threads_per_threadgroup() < 256 {
        return Err("selected F16 MLA 需要至少 256 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&latent.buffer), 0);
    encoder.set_buffer(2, Some(&k_rope.buffer), 0);
    encoder.set_buffer(3, Some(selection), 0);
    encoder.set_buffer(4, Some(&kv_b.buffer), 0);
    encoder.set_buffer(5, Some(&output.buffer), 0);
    set_bytes(&encoder, 6, &rows);
    set_bytes(&encoder, 7, &top_k_u32);
    encoder.dispatch_thread_groups(MTLSize::new(spec.num_heads.div_ceil(8) as u64, query.rows as u64, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={},heads={},top_k={top_k},f16", query.rows, spec.num_heads);
    ctx.commit_and_wait_profiled(&command, "mla_prefill_attention_selected", &shape, query.buffer.length() + latent.buffer.length() + k_rope.buffer.length() + selection.length() + kv_b.buffer.length(), output.buffer.length());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn mla_prefill_attention_selected_fp8(
    ctx: &MetalContext,
    query: &MetalTensor,
    latent: &MetalTensor,
    k_rope: &MetalTensor,
    kv_b: &metal::Buffer,
    kv_b_scale_inv: &metal::Buffer,
    weight_rows: usize,
    weight_cols: usize,
    selection: &metal::Buffer,
    top_k: usize,
    spec: &MlaSpec,
) -> Result<MetalTensor, String> {
    let q_head = spec.q_head_dim();
    let kv_head = spec.kv_head_dim();
    let value_dim = spec.value_dim();
    if query.rows == 0
        || query.cols != spec.num_heads * q_head
        || latent.rows != query.rows
        || latent.cols != spec.kv_lora_rank
        || k_rope.rows != query.rows
        || k_rope.cols != spec.qk_rope_head_dim
        || weight_rows != spec.num_heads * kv_head
        || weight_cols != spec.kv_lora_rank
        || spec.kv_lora_rank > 512
        || spec.qk_rope_head_dim > 128
    {
        return Err(format!("selected FP8 MLA 形状不符: q=[{},{}] latent=[{},{}] rope=[{},{}] weight=[{weight_rows},{weight_cols}]", query.rows, query.cols, latent.rows, latent.cols, k_rope.rows, k_rope.cols));
    }
    let rows = validate_u32("selected MLA rows", query.rows)?;
    let top_k_u32 = validate_u32("selected MLA top_k", top_k)?;
    let constants = [
        validate_u32("selected MLA heads", spec.num_heads)?,
        validate_u32("selected MLA nope", spec.qk_nope_dim())?,
        validate_u32("selected MLA rope", spec.qk_rope_head_dim)?,
        validate_u32("selected MLA value", value_dim)?,
        validate_u32("selected MLA q_head", q_head)?,
        validate_u32("selected MLA kv_head", kv_head)?,
        validate_u32("selected MLA latent", spec.kv_lora_rank)?,
        validate_u32("selected MLA FP8 scale cols", weight_cols.div_ceil(128))?,
    ];
    let output = ctx.shared_buffer_zeros(query.rows * spec.num_heads * value_dim * mem::size_of::<f16>());
    let output = MetalTensor::new(output, query.rows, spec.num_heads * value_dim);
    let pipeline = ctx.pipeline_u32_constants("mla_prefill_attention_selected_fp8_f16", &constants)?;
    if pipeline.max_total_threads_per_threadgroup() < 256 {
        return Err("selected FP8 MLA 需要至少 256 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&latent.buffer), 0);
    encoder.set_buffer(2, Some(&k_rope.buffer), 0);
    encoder.set_buffer(3, Some(selection), 0);
    encoder.set_buffer(4, Some(kv_b), 0);
    encoder.set_buffer(5, Some(kv_b_scale_inv), 0);
    encoder.set_buffer(6, Some(&output.buffer), 0);
    set_bytes(&encoder, 7, &rows);
    set_bytes(&encoder, 8, &top_k_u32);
    encoder.dispatch_thread_groups(MTLSize::new(spec.num_heads.div_ceil(8) as u64, query.rows as u64, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={},heads={},top_k={top_k},fp8", query.rows, spec.num_heads);
    ctx.commit_and_wait_profiled(
        &command,
        "mla_prefill_attention_selected",
        &shape,
        query.buffer.length() + latent.buffer.length() + k_rope.buffer.length() + selection.length() + kv_b.length() + kv_b_scale_inv.length(),
        output.buffer.length(),
    );
    Ok(output)
}

pub enum KvBWeight<'a> {
    /// FP8 压缩权重(MoE 层)。
    Fp8(&'a Fp8Matrix),
    /// f32 权重(dense 层),内部转 f16 上传。
    F16(&'a [f32]),
    /// 已驻留的 FP8 codes/scales。
    Fp8Resident { codes: &'a metal::Buffer, scale_inv: &'a metal::Buffer, rows: usize, cols: usize },
    /// 已驻留的 F16 矩阵。
    F16Resident(&'a MetalTensor),
    /// 已驻留的 W8A16 packed/scales(CT Int4-Int8Mix 的 kv_b_proj)。
    W8A16Resident { packed: &'a metal::Buffer, scales: &'a metal::Buffer, scale_dtype: u32, group_size: usize, rows: usize, cols: usize },
}

/// MLA decode attention(weight absorption)，按 cache format 选择 F16 或 INT8 latent kernel。
///
/// 串 3 个 kernel:q_latent(吸收 W_K)→ i8 attention(在线量化 + 点积 + softmax)→ v 投影(W_V)。
/// q 是本 token 的 query `[1, head_count × q_head_dim]`(已 RoPE)。cache 存 [0, position) 历史 latent。
/// 返回 attention context `[1, head_count × value_dim]`。
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_attention(
    ctx: &MetalContext,
    q: &MetalTensor,
    cache: &MetalKvCache,
    kv_b: KvBWeight<'_>,
    layer: usize,
    position: usize,
    head_count: usize,
    qk_nope_dim: usize,
    rope_dim: usize,
    value_dim: usize,
) -> Result<MetalTensor, String> {
    mla_decode_attention_impl(ctx, q, cache, kv_b, layer, position, head_count, qk_nope_dim, rope_dim, value_dim, None)
}

#[allow(clippy::too_many_arguments)]
pub fn mla_decode_attention_selected(
    ctx: &MetalContext,
    q: &MetalTensor,
    cache: &MetalKvCache,
    kv_b: KvBWeight<'_>,
    layer: usize,
    position: usize,
    head_count: usize,
    qk_nope_dim: usize,
    rope_dim: usize,
    value_dim: usize,
    selection: &metal::Buffer,
    selected_rows: usize,
) -> Result<MetalTensor, String> {
    mla_decode_attention_impl(ctx, q, cache, kv_b, layer, position, head_count, qk_nope_dim, rope_dim, value_dim, Some((selection, selected_rows)))
}

#[allow(clippy::too_many_arguments)]
fn mla_decode_attention_impl(
    ctx: &MetalContext,
    q: &MetalTensor,
    cache: &MetalKvCache,
    kv_b: KvBWeight<'_>,
    layer: usize,
    position: usize,
    head_count: usize,
    qk_nope_dim: usize,
    rope_dim: usize,
    value_dim: usize,
    selected: Option<(&metal::Buffer, usize)>,
) -> Result<MetalTensor, String> {
    let layout = cache.mla_layout().ok_or("mla_decode_attention 需 MLA cache")?;
    let kv_lora = layout.kv_lora_rank;
    let group_size = layout.group_size;
    let kv_head_dim = qk_nope_dim + value_dim;
    // q_latent/attention kernel 每 lane 只覆盖 2×256 列 latent,value 投影每 head 256 threads;
    // selected tile 版 rope_tile[128] 进一步把 rope 限到 128(非 selected 每 lane 一列，上限 256)。
    let rope_cap = if selected.is_some() { 128 } else { 256 };
    if kv_lora > 512 || rope_dim > rope_cap {
        return Err(format!("MLA decode kernel 容量不足：kv_lora={kv_lora}/512, rope_dim={rope_dim}/{rope_cap}"));
    }
    if value_dim > 256 {
        return Err(format!("MLA decode value_dim={value_dim} 超过 mla_decode_v_fp8_f16 上限 256(每 head 一个 256-thread threadgroup)"));
    }
    let (weight_is_fp8, kv_b_codes, kv_b_scale_inv): (u32, metal::Buffer, metal::Buffer) = match &kv_b {
        KvBWeight::Fp8(m) => {
            if m.cols != kv_lora || m.rows != head_count * kv_head_dim {
                return Err(format!("kv_b_proj FP8 形状 [out={},in={}] 与期望 [out={},in={kv_lora}] 不符(heads={head_count}, kv_head_dim={kv_head_dim})", m.rows, m.cols, head_count * kv_head_dim));
            }
            (1, ctx.shared_buffer(&m.codes), ctx.shared_buffer(&m.scale_inv))
        }
        KvBWeight::F16(w) => {
            let expected = head_count * kv_head_dim * kv_lora;
            if w.len() != expected {
                return Err(format!("kv_b f16 权重长度 {} 与期望 {expected} 不符", w.len()));
            }
            // f32 → f16 buffer(shared_buffer_from_f32 内部转 f16)。
            (0, ctx.shared_buffer_from_f32(w), ctx.shared_buffer(&[0u8]))
        }
        KvBWeight::Fp8Resident { codes, scale_inv, rows, cols } => {
            if *cols != kv_lora || *rows != head_count * kv_head_dim {
                return Err(format!("resident kv_b_proj FP8 形状 [out={rows},in={cols}] 与期望 [out={},in={kv_lora}] 不符", head_count * kv_head_dim));
            }
            (1, (*codes).clone(), (*scale_inv).clone())
        }
        KvBWeight::F16Resident(weight) => {
            let expected = head_count * kv_head_dim * kv_lora;
            if weight.rows * weight.cols != expected || weight.cols != kv_lora {
                return Err(format!("resident kv_b F16 形状 [{},{}] 与期望 [{},{}] 不符", weight.rows, weight.cols, head_count * kv_head_dim, kv_lora));
            }
            (0, weight.buffer.clone(), ctx.shared_buffer(&[0u8]))
        }
        KvBWeight::W8A16Resident { .. } => {
            return Err("MLA decode kernel 暂不支持 W8A16 kv_b(CT 源 decode 走 prefill 路径前需实现)".to_owned());
        }
    };
    if q.cols != head_count * (qk_nope_dim + rope_dim) || q.rows != 1 {
        return Err(format!("decode q 形状 [{},{}] 与期望 [1,{}]", q.rows, q.cols, head_count * (qk_nope_dim + rope_dim)));
    }
    let scale_columns = kv_lora.div_ceil(128);
    let position_u32 = validate_u32("decode position", position)?;
    let head_count_u32 = validate_u32("head_count", head_count)?;
    let qk_nope_u32 = validate_u32("qk_nope_dim", qk_nope_dim)?;
    let rope_u32 = validate_u32("rope_dim", rope_dim)?;
    let kv_head_u32 = validate_u32("kv_head_dim", kv_head_dim)?;
    let kv_lora_u32 = validate_u32("kv_lora_dim", kv_lora)?;
    let group_u32 = validate_u32("group_size", group_size)?;
    let scale_cols_u32 = validate_u32("scale_columns", scale_columns)?;
    let value_u32 = validate_u32("value_dim", value_dim)?;
    let q_head_dim = qk_nope_dim + rope_dim;
    let q_head_u32 = validate_u32("q_head_dim", q_head_dim)?;
    let attn_scale = 1.0f32 / (q_head_dim as f32).sqrt();

    let q_latent = ctx.tensor_zeros(head_count, kv_lora);
    let latent_output = ctx.tensor_zeros(head_count, kv_lora);
    let output = ctx.tensor_zeros(1, head_count * value_dim);
    let selected_rows_count = selected.map_or(0, |(_, rows)| rows);
    let selected_chunks = if selected_rows_count == 0 { 0 } else { selected_rows_count.div_ceil(256).min(8) };
    let selected_chunk_size = if selected_chunks == 0 { 0 } else { selected_rows_count.div_ceil(selected_chunks) };
    let selected_partial_values = (selected_chunks > 0).then(|| ctx.shared_buffer_zeros(selected_chunks * head_count * kv_lora * mem::size_of::<f32>()));
    let selected_partial_stats = (selected_chunks > 0).then(|| ctx.shared_buffer_zeros(selected_chunks * head_count * 2 * mem::size_of::<f32>()));

    let rope_offset = cache.layer_rope_offset(layer)?;
    let cache_buf = cache.buffer();

    let command = ctx.command_buffer();

    // Step 1: q_latent(吸收 W_K)。每 head 一个 threadgroup,256 threads。
    let q_latent_pipeline = ctx.pipeline("mla_decode_q_latent_fp8_f16")?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&q_latent_pipeline);
    encoder.set_buffer(0, Some(&q.buffer), 0);
    encoder.set_buffer(1, Some(&kv_b_codes), 0);
    encoder.set_buffer(2, Some(&kv_b_scale_inv), 0);
    encoder.set_buffer(3, Some(&q_latent.buffer), 0);
    set_bytes(&encoder, 4, &head_count_u32);
    set_bytes(&encoder, 5, &qk_nope_u32);
    set_bytes(&encoder, 6, &rope_u32);
    set_bytes(&encoder, 7, &kv_head_u32);
    set_bytes(&encoder, 8, &kv_lora_u32);
    set_bytes(&encoder, 9, &scale_cols_u32);
    set_bytes(&encoder, 10, &weight_is_fp8);
    encoder.dispatch_thread_groups(MTLSize::new(head_count as u64, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();

    // Step 2:latent-space attention。F16 基线直接点积；INT8 保留作压缩对照。
    match cache.format() {
        MetalKvCacheFormat::F16 => {
            let attn_pipeline = ctx.pipeline(if selected.is_some() { "mla_decode_attention_f16_latent_selected" } else { "mla_decode_attention_f16_latent" })?;
            let attn_threads = 256;
            if attn_pipeline.max_total_threads_per_threadgroup() < attn_threads {
                return Err(format!("mla_decode_attention_f16 kernel 需要 {attn_threads} threads/threadgroup"));
            }
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&attn_pipeline);
            encoder.set_buffer(0, Some(&q_latent.buffer), 0);
            encoder.set_buffer(1, Some(cache_buf), cache.layer_latent_offset(layer)?);
            encoder.set_buffer(2, Some(&q.buffer), 0);
            encoder.set_buffer(3, Some(cache_buf), rope_offset);
            if let Some((selection, selected_rows)) = selected {
                let selected_u32 = validate_u32("selected rows", selected_rows)?;
                let chunks_u32 = validate_u32("selected chunks", selected_chunks)?;
                let chunk_size_u32 = validate_u32("selected chunk size", selected_chunk_size)?;
                let partial_values = selected_partial_values.as_ref().ok_or("selected MLA 缺少 partial values")?;
                let partial_stats = selected_partial_stats.as_ref().ok_or("selected MLA 缺少 partial stats")?;
                encoder.set_buffer(4, Some(selection), 0);
                encoder.set_buffer(5, Some(partial_values), 0);
                encoder.set_buffer(6, Some(partial_stats), 0);
                set_bytes(&encoder, 7, &selected_u32);
                set_bytes(&encoder, 8, &kv_lora_u32);
                set_bytes(&encoder, 9, &rope_u32);
                set_bytes(&encoder, 10, &attn_scale);
                set_bytes(&encoder, 11, &q_head_u32);
                set_bytes(&encoder, 12, &head_count_u32);
                set_bytes(&encoder, 13, &chunk_size_u32);
                encoder.dispatch_thread_groups(MTLSize::new(head_count.div_ceil(8) as u64, chunks_u32 as u64, 1), MTLSize::new(attn_threads, 1, 1));
                encoder.memory_barrier_with_resources(&[partial_values, partial_stats]);

                let merge_pipeline = ctx.pipeline("mla_decode_attention_f16_latent_selected_merge")?;
                if merge_pipeline.max_total_threads_per_threadgroup() < 256 {
                    return Err("selected MLA merge 需要 256 threads/threadgroup".to_owned());
                }
                encoder.set_compute_pipeline_state(&merge_pipeline);
                encoder.set_buffer(0, Some(partial_values), 0);
                encoder.set_buffer(1, Some(partial_stats), 0);
                encoder.set_buffer(2, Some(&latent_output.buffer), 0);
                set_bytes(&encoder, 3, &chunks_u32);
                set_bytes(&encoder, 4, &head_count_u32);
                set_bytes(&encoder, 5, &kv_lora_u32);
                encoder.dispatch_thread_groups(MTLSize::new(head_count as u64, 1, 1), MTLSize::new(256, 1, 1));
            } else {
                encoder.set_buffer(4, Some(&latent_output.buffer), 0);
                set_bytes(&encoder, 5, &position_u32);
                set_bytes(&encoder, 6, &kv_lora_u32);
                set_bytes(&encoder, 7, &rope_u32);
                set_bytes(&encoder, 8, &attn_scale);
                set_bytes(&encoder, 9, &q_head_u32);
                encoder.dispatch_thread_groups(MTLSize::new(head_count as u64, 1, 1), MTLSize::new(attn_threads, 1, 1));
            }
            encoder.end_encoding();
        }
        MetalKvCacheFormat::Int8 => {
            let attn_pipeline = ctx.pipeline(if selected.is_some() { "mla_decode_attention_i8_latent_selected" } else { "mla_decode_attention_i8_latent" })?;
            if attn_pipeline.max_total_threads_per_threadgroup() < 256 {
                return Err("mla_decode_attention_i8 kernel 需要 256 threads/threadgroup".to_owned());
            }
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&attn_pipeline);
            encoder.set_buffer(0, Some(&q_latent.buffer), 0);
            encoder.set_buffer(1, Some(cache_buf), cache.layer_codes_offset(layer)?);
            encoder.set_buffer(2, Some(cache_buf), cache.layer_scales_offset(layer)?);
            encoder.set_buffer(3, Some(&q.buffer), 0);
            encoder.set_buffer(4, Some(cache_buf), rope_offset);
            if let Some((selection, selected_rows)) = selected {
                let selected_u32 = validate_u32("selected rows", selected_rows)?;
                let chunks_u32 = validate_u32("selected chunks", selected_chunks)?;
                let chunk_size_u32 = validate_u32("selected chunk size", selected_chunk_size)?;
                let partial_values = selected_partial_values.as_ref().ok_or("selected INT8 MLA 缺少 partial values")?;
                let partial_stats = selected_partial_stats.as_ref().ok_or("selected INT8 MLA 缺少 partial stats")?;
                encoder.set_buffer(5, Some(selection), 0);
                encoder.set_buffer(6, Some(partial_values), 0);
                encoder.set_buffer(7, Some(partial_stats), 0);
                set_bytes(&encoder, 8, &selected_u32);
                set_bytes(&encoder, 9, &kv_lora_u32);
                set_bytes(&encoder, 10, &group_u32);
                set_bytes(&encoder, 11, &rope_u32);
                set_bytes(&encoder, 12, &attn_scale);
                set_bytes(&encoder, 13, &q_head_u32);
                set_bytes(&encoder, 14, &head_count_u32);
                set_bytes(&encoder, 15, &chunk_size_u32);
                encoder.dispatch_thread_groups(MTLSize::new(head_count.div_ceil(8) as u64, chunks_u32 as u64, 1), MTLSize::new(256, 1, 1));
                encoder.memory_barrier_with_resources(&[partial_values, partial_stats]);

                let merge_pipeline = ctx.pipeline("mla_decode_attention_f16_latent_selected_merge")?;
                encoder.set_compute_pipeline_state(&merge_pipeline);
                encoder.set_buffer(0, Some(partial_values), 0);
                encoder.set_buffer(1, Some(partial_stats), 0);
                encoder.set_buffer(2, Some(&latent_output.buffer), 0);
                set_bytes(&encoder, 3, &chunks_u32);
                set_bytes(&encoder, 4, &head_count_u32);
                set_bytes(&encoder, 5, &kv_lora_u32);
                encoder.dispatch_thread_groups(MTLSize::new(head_count as u64, 1, 1), MTLSize::new(256, 1, 1));
            } else {
                encoder.set_buffer(5, Some(&latent_output.buffer), 0);
                set_bytes(&encoder, 6, &position_u32);
                set_bytes(&encoder, 7, &kv_lora_u32);
                set_bytes(&encoder, 8, &group_u32);
                set_bytes(&encoder, 9, &rope_u32);
                set_bytes(&encoder, 10, &attn_scale);
                set_bytes(&encoder, 11, &q_head_u32);
                encoder.dispatch_thread_groups(MTLSize::new(head_count as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            encoder.end_encoding();
        }
    }

    // Step 3: v 投影(W_V)。每 head 一个 threadgroup,value_dim threads。
    let v_pipeline = ctx.pipeline("mla_decode_v_fp8_f16")?;
    let v_threads = value_dim.clamp(1, 256) as u64;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&v_pipeline);
    encoder.set_buffer(0, Some(&latent_output.buffer), 0);
    encoder.set_buffer(1, Some(&kv_b_codes), 0);
    encoder.set_buffer(2, Some(&kv_b_scale_inv), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &head_count_u32);
    set_bytes(&encoder, 5, &qk_nope_u32);
    set_bytes(&encoder, 6, &kv_head_u32);
    set_bytes(&encoder, 7, &kv_lora_u32);
    set_bytes(&encoder, 8, &value_u32);
    set_bytes(&encoder, 9, &scale_cols_u32);
    set_bytes(&encoder, 10, &weight_is_fp8);
    encoder.dispatch_thread_groups(MTLSize::new(head_count as u64, 1, 1), MTLSize::new(v_threads, 1, 1));
    encoder.end_encoding();

    let attended_rows = selected.map_or(position, |(_, rows)| rows);
    let operator = if selected.is_some() { "mla_decode_attention_selected" } else { "mla_decode_attention" };
    let shape = format!("decode_attn heads={head_count},position={position},attended={attended_rows},kv_lora={kv_lora},cache={:?}", cache.format());
    ctx.commit_and_wait_profiled(
        &command,
        operator,
        &shape,
        q.buffer.length() + (attended_rows * cache.bytes_per_token()) as u64 + kv_b_codes.length() + kv_b_scale_inv.length(),
        q_latent.buffer.length() + latent_output.buffer.length() + output.buffer.length(),
    );
    Ok(output)
}
