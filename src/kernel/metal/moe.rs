/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: sample_top_p_f16, argmax_f16_bf16, argmax_f32, moe_router_sigmoid_topk_f16_bias_f16, moe_router_bias_topk_f32_weight, moe_router_sqrt_softplus_selected_f32_weight, moe_router_softmax_topk_f16, moe_router_softmax_topk_f32_weight, moe_router_logits_parallel_f32_input_weight, moe_router_softmax_topk_f32_input_weight, moe_router_softmax_topk_logits_f32, gather_rows_f16, gather_rows_f32, gather_rows_f32_to_f16, scatter_add_rows_weighted_f32, moe_sort_topk_by_id
// private helpers: threadgroup_sum_64
pub const SHADERS: &str = r#"
inline float threadgroup_sum_64(
    float value,
    threadgroup float *partial,
    uint simd_group,
    uint simd_lane)
{
    const float local_sum = simd_sum(value);
    if (simd_lane == 0) partial[simd_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0] + partial[1];
}
inline bool moe_ordered_before(float left, uint left_id, float right, uint right_id)
{
    return left > right || (left == right && left_id < right_id);
}
inline void moe_bitonic_sort(
    threadgroup float *scores,
    threadgroup float *values,
    threadgroup uint *expert_ids,
    bool move_values,
    uint lane,
    uint width)
{
    for (uint size = 2; size <= width; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            const uint other = lane ^ stride;
            if (other > lane) {
                const float left_score = scores[lane];
                const float right_score = scores[other];
                const uint left_id = expert_ids[lane];
                const uint right_id = expert_ids[other];
                const bool descending = (lane & size) == 0;
                const bool swap = descending
                    ? moe_ordered_before(right_score, right_id, left_score, left_id)
                    : moe_ordered_before(left_score, left_id, right_score, right_id);
                if (swap) {
                    scores[lane] = right_score;
                    scores[other] = left_score;
                    if (move_values) {
                        const float left_value = values[lane];
                        values[lane] = values[other];
                        values[other] = left_value;
                    }
                    expert_ids[lane] = right_id;
                    expert_ids[other] = left_id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
kernel void sample_top_p_f16(
    device const half *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &length [[buffer(2)]],
    constant float &temperature [[buffer(3)]],
    constant float &top_p [[buffer(4)]],
    constant float &random [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float minimums[256];
    threadgroup float maximums[256];
    threadgroup float sums[256];
    threadgroup float state[4];
    float local_minimum = 3.402823466e+38f;
    float local_maximum = -3.402823466e+38f;
    for (uint index = lane; index < length; index += 256) {
        const float value = float(input[index]) / temperature;
        local_minimum = min(local_minimum, value);
        local_maximum = max(local_maximum, value);
    }
    minimums[lane] = local_minimum;
    maximums[lane] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            minimums[lane] = min(minimums[lane], minimums[lane + stride]);
            maximums[lane] = max(maximums[lane], maximums[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_logit = maximums[0];
    float sum = 0.0f;
    for (uint index = lane; index < length; index += 256) {
        sum += exp(float(input[index]) / temperature - max_logit);
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        state[0] = minimums[0];
        state[1] = maximums[0];
        state[2] = top_p * sums[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 找到最高的 logit 阈值，使阈值以上概率质量仍覆盖 top-p。
    for (uint iteration = 0; iteration < 24; ++iteration) {
        const float threshold = (state[0] + state[1]) * 0.5f;
        float mass = 0.0f;
        for (uint index = lane; index < length; index += 256) {
            const float value = float(input[index]) / temperature;
            if (value >= threshold) mass += exp(value - max_logit);
        }
        sums[lane] = mass;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) sums[lane] += sums[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            if (sums[0] >= state[2]) state[0] = threshold;
            else state[1] = threshold;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 二分值落在两个 F16 logit 之间，向上吸附到真实候选值。
    float threshold_candidate = 3.402823466e+38f;
    for (uint index = lane; index < length; index += 256) {
        const float value = float(input[index]) / temperature;
        if (value >= state[0]) threshold_candidate = min(threshold_candidate, value);
    }
    minimums[lane] = threshold_candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) minimums[lane] = min(minimums[lane], minimums[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) state[0] = minimums[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 连续 token-id 分片使最终选择只需扫描一个约 length/256 的小区间。
    const uint begin = uint((ulong(length) * lane) / 256ul);
    const uint end = uint((ulong(length) * (lane + 1)) / 256ul);
    float allowed_mass = 0.0f;
    for (uint index = begin; index < end; ++index) {
        const float value = float(input[index]) / temperature;
        if (value >= state[0]) allowed_mass += exp(value - max_logit);
    }
    sums[lane] = allowed_mass;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float total = 0.0f;
        for (uint index = 0; index < 256; ++index) total += sums[index];
        float target = random * total;
        uint selected_lane = 0;
        for (; selected_lane < 255 && target >= sums[selected_lane]; ++selected_lane) target -= sums[selected_lane];
        state[1] = float(selected_lane);
        state[2] = target;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == uint(state[1])) {
        float cumulative = 0.0f;
        uint selected = begin;
        for (uint index = begin; index < end; ++index) {
            const float value = float(input[index]) / temperature;
            if (value < state[0]) continue;
            cumulative += exp(value - max_logit);
            selected = index;
            if (cumulative >= state[2]) break;
        }
        output[0] = selected;
    }
}
// 两阶段 argmax(大词表):单 TG 串行扫 130k 元素是纯访存延迟链(~134µs/token);
// 32 TG 分段各算局部最优,第二段 1 TG 归约。moe_ordered_before 是全序
// (值降序、并列取小 index),分组归约结果与单段逐位一致。
kernel void argmax_partial_f16_bf16(
    device const ushort *input [[buffer(0)]],
    device float *partial_values [[buffer(1)]],
    device uint *partial_indices [[buffer(2)]],
    constant uint &length [[buffer(3)]],
    constant uint *excluded [[buffer(4)]],
    constant uint &excluded_count [[buffer(5)]],
    constant uint &input_bf16 [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint groups [[threadgroups_per_grid]])
{
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best_value = -3.402823466e+38f;
    uint best_index = 0;
    const uint span = (length + groups - 1) / groups;
    const uint begin = group * span;
    const uint end = min(begin + span, length);
    for (uint index = begin + lane; index < end; index += 256) {
        bool is_excluded = false;
        for (uint excluded_index = 0; excluded_index < excluded_count; ++excluded_index) {
            is_excluded = is_excluded || index == excluded[excluded_index];
        }
        if (is_excluded) continue;
        const float value = input_bf16 != 0
            ? as_type<float>(uint(input[index]) << 16)
            : float(as_type<half>(input[index]));
        if (moe_ordered_before(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[lane] = best_value;
    indices[lane] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other_value = values[lane + stride];
            const uint other_index = indices[lane + stride];
            if (moe_ordered_before(other_value, other_index, values[lane], indices[lane])) {
                values[lane] = other_value;
                indices[lane] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        partial_values[group] = values[0];
        partial_indices[group] = indices[0];
    }
}
kernel void argmax_reduce_f32(
    device const float *partial_values [[buffer(0)]],
    device const uint *partial_indices [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &partial_count [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best_value = -3.402823466e+38f;
    uint best_index = 0;
    for (uint index = lane; index < partial_count; index += 256) {
        if (moe_ordered_before(partial_values[index], partial_indices[index], best_value, best_index)) {
            best_value = partial_values[index];
            best_index = partial_indices[index];
        }
    }
    values[lane] = best_value;
    indices[lane] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other_value = values[lane + stride];
            const uint other_index = indices[lane + stride];
            if (moe_ordered_before(other_value, other_index, values[lane], indices[lane])) {
                values[lane] = other_value;
                indices[lane] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[0] = indices[0];
}
kernel void argmax_f16_bf16(
    device const ushort *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &length [[buffer(2)]],
    constant uint *excluded [[buffer(3)]],
    constant uint &excluded_count [[buffer(4)]],
    constant uint &input_bf16 [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best_value = -3.402823466e+38f;
    uint best_index = 0;
    for (uint index = lane; index < length; index += 256) {
        bool is_excluded = false;
        for (uint excluded_index = 0; excluded_index < excluded_count; ++excluded_index) {
            is_excluded = is_excluded || index == excluded[excluded_index];
        }
        if (is_excluded) continue;
        const float value = input_bf16 != 0
            ? as_type<float>(uint(input[index]) << 16)
            : float(as_type<half>(input[index]));
        if (moe_ordered_before(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[lane] = best_value;
    indices[lane] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other_value = values[lane + stride];
            const uint other_index = indices[lane + stride];
            if (moe_ordered_before(other_value, other_index, values[lane], indices[lane])) {
                values[lane] = other_value;
                indices[lane] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[0] = indices[0];
}
kernel void argmax_f32(
    device const float *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &length [[buffer(2)]],
    constant uint *excluded [[buffer(3)]],
    constant uint &excluded_count [[buffer(4)]],
    constant uint &unused [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best_value = -3.402823466e+38f;
    uint best_index = 0;
    for (uint index = lane; index < length; index += 256) {
        bool is_excluded = false;
        for (uint excluded_index = 0; excluded_index < excluded_count; ++excluded_index) {
            is_excluded = is_excluded || index == excluded[excluded_index];
        }
        if (is_excluded) continue;
        const float value = input[index];
        if (moe_ordered_before(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[lane] = best_value;
    indices[lane] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other_value = values[lane + stride];
            const uint other_index = indices[lane + stride];
            if (moe_ordered_before(other_value, other_index, values[lane], indices[lane])) {
                values[lane] = other_value;
                indices[lane] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[0] = indices[0];
}
kernel void moe_router_sigmoid_topk_f16_bias_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device uint *output_ids [[buffer(3)]],
    device float *output_weights [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &expert_count [[buffer(6)]],
    constant uint &top_k [[buffer(7)]],
    constant float &scaling_factor [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float corrected_scores[256];
    threadgroup float raw_scores[256];
    threadgroup uint expert_ids[256];
    threadgroup float top_sum;

    float raw = 0.0f;
    float corrected = -INFINITY;
    if (lane < expert_count) {
        ulong input_base = ulong(row) * columns;
        ulong weight_base = ulong(lane) * columns;
        float logit = 0.0f;
        for (uint column = 0; column < columns; ++column) {
            logit += float(input[input_base + column]) * float(weight[weight_base + column]);
        }
        raw = 1.0f / (1.0f + exp(-logit));
        corrected = raw + float(bias[lane]);
    }
    corrected_scores[lane] = corrected;
    raw_scores[lane] = raw;
    expert_ids[lane] = lane;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_bitonic_sort(corrected_scores, raw_scores, expert_ids, true, lane, width);

    if (lane == 0) {
        float sum = 0.0f;
        for (uint index = 0; index < top_k; ++index) sum += raw_scores[index];
        top_sum = max(sum, 1.0e-20f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < top_k) {
        ulong output = ulong(row) * top_k + lane;
        output_ids[output] = expert_ids[lane];
        output_weights[output] = raw_scores[lane] / top_sum * scaling_factor;
    }
}
kernel void moe_router_bias_topk_f32_weight(
    device const half *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device uint *output_ids [[buffer(3)]],
    device float *output_weights [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &expert_count [[buffer(6)]],
    constant uint &top_k [[buffer(7)]],
    constant float &scaling_factor [[buffer(8)]],
    constant uint &scoring [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float corrected_scores[256];
    threadgroup float raw_scores[256];
    threadgroup uint expert_ids[256];
    threadgroup float top_sum;
    float raw = -INFINITY;
    float corrected = -INFINITY;
    if (lane < expert_count) {
        float logit = 0.0f;
        const ulong input_base = ulong(row) * columns;
        const ulong weight_base = ulong(lane) * columns;
        for (uint column = 0; column < columns; ++column) {
            logit += float(input[input_base + column]) * weight[weight_base + column];
        }
        const float softplus = max(logit, 0.0f) + log(1.0f + exp(-fabs(logit)));
        raw = scoring == 2 ? sqrt(max(softplus, 0.0f)) : 1.0f / (1.0f + exp(-logit));
        corrected = raw + bias[lane];
    }
    corrected_scores[lane] = corrected;
    raw_scores[lane] = raw;
    expert_ids[lane] = lane;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_bitonic_sort(corrected_scores, raw_scores, expert_ids, true, lane, width);

    if (lane == 0) {
        float sum = 0.0f;
        for (uint index = 0; index < top_k; ++index) {
            sum += raw_scores[index];
        }
        top_sum = max(sum, 1.0e-20f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < top_k) {
        const ulong output = ulong(row) * top_k + lane;
        output_ids[output] = expert_ids[lane];
        output_weights[output] = raw_scores[lane] / top_sum * scaling_factor;
    }
}
kernel void moe_router_sqrt_softplus_selected_f32_weight(
    device const half *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const uint *selected_experts [[buffer(2)]],
    device uint *output_ids [[buffer(3)]],
    device float *output_weights [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &expert_count [[buffer(6)]],
    constant uint &top_k [[buffer(7)]],
    constant float &scaling_factor [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float raw_scores[256];
    threadgroup float top_sum;
    float raw = 0.0f;
    const ulong output = ulong(row) * top_k + lane;
    if (lane < top_k) {
        const uint expert = selected_experts[output];
        float logit = 0.0f;
        const ulong input_base = ulong(row) * columns;
        const ulong weight_base = ulong(expert) * columns;
        for (uint column = 0; column < columns; ++column) {
            logit += float(input[input_base + column]) * weight[weight_base + column];
        }
        const float softplus = max(logit, 0.0f) + log(1.0f + exp(-fabs(logit)));
        raw = sqrt(max(softplus, 0.0f));
    }
    raw_scores[lane] = raw;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float sum = 0.0f;
        for (uint index = 0; index < top_k; ++index) sum += raw_scores[index];
        top_sum = max(sum, 1.0e-20f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < top_k) {
        output_ids[output] = selected_experts[output];
        output_weights[output] = raw_scores[lane] / top_sum * scaling_factor;
    }
}
kernel void moe_router_softmax_topk_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device uint *output_ids [[buffer(2)]],
    device float *output_weights [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &expert_count [[buffer(5)]],
    constant uint &top_k [[buffer(6)]],
    constant float &scaling_factor [[buffer(7)]],
    constant uint &normalize_selected [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float logits[256];
    threadgroup uint expert_ids[256];
    threadgroup float denominator;
    float logit = -INFINITY;
    if (lane < expert_count) {
        logit = 0.0f;
        const ulong input_base = ulong(row) * columns;
        const ulong weight_base = ulong(lane) * columns;
        for (uint column = 0; column < columns; ++column) {
            logit += float(input[input_base + column]) * float(weight[weight_base + column]);
        }
    }
    logits[lane] = logit;
    expert_ids[lane] = lane;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_bitonic_sort(logits, logits, expert_ids, false, lane, width);

    if (lane == 0) {
        float sum = 0.0f;
        const uint count = normalize_selected != 0 ? top_k : expert_count;
        for (uint index = 0; index < count; ++index) sum += exp(logits[index] - logits[0]);
        denominator = max(sum, 1.0e-20f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < top_k) {
        const ulong output = ulong(row) * top_k + lane;
        output_ids[output] = expert_ids[lane];
        output_weights[output] = exp(logits[lane] - logits[0]) / denominator * scaling_factor;
    }
}
kernel void moe_router_softmax_topk_f32_weight(
    device const half *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device uint *output_ids [[buffer(2)]],
    device float *output_weights [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &expert_count [[buffer(5)]],
    constant uint &top_k [[buffer(6)]],
    constant float &scaling_factor [[buffer(7)]],
    constant uint &normalize_selected [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float logits[256];
    threadgroup uint expert_ids[256];
    threadgroup float denominator;
    float logit = -INFINITY;
    if (lane < expert_count) {
        logit = 0.0f;
        const ulong input_base = ulong(row) * columns;
        const ulong weight_base = ulong(lane) * columns;
        for (uint column = 0; column < columns; ++column) {
            logit += float(input[input_base + column]) * weight[weight_base + column];
        }
    }
    logits[lane] = logit;
    expert_ids[lane] = lane;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_bitonic_sort(logits, logits, expert_ids, false, lane, width);

    if (lane == 0) {
        float sum = 0.0f;
        const uint count = normalize_selected != 0 ? top_k : expert_count;
        for (uint index = 0; index < count; ++index) sum += exp(logits[index] - logits[0]);
        denominator = max(sum, 1.0e-20f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < top_k) {
        const ulong output = ulong(row) * top_k + lane;
        output_ids[output] = expert_ids[lane];
        output_weights[output] = exp(logits[lane] - logits[0]) / denominator * scaling_factor;
    }
}
kernel void moe_router_logits_parallel_f32_input_weight(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *logits [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &expert_count [[buffer(4)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (expert >= expert_count) return;
    const ulong weight_base = ulong(expert) * columns;
    float logit = 0.0f;
    for (uint column = lane; column < columns; column += 64) {
        logit += input[column] * weight[weight_base + column];
    }
    threadgroup float partial[2];
    const float total = threadgroup_sum_64(logit, partial, simd_group, simd_lane);
    if (lane == 0) logits[expert] = total;
}
kernel void moe_router_softmax_topk_f32_input_weight(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device uint *output_ids [[buffer(2)]],
    device float *output_weights [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &expert_count [[buffer(5)]],
    constant uint &top_k [[buffer(6)]],
    constant float &scaling_factor [[buffer(7)]],
    constant uint &normalize_selected [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float logits[256];
    threadgroup uint expert_ids[256];
    threadgroup float denominator;
    float logit = -INFINITY;
    if (lane < expert_count) {
        logit = 0.0f;
        const ulong input_base = ulong(row) * columns;
        const ulong weight_base = ulong(lane) * columns;
        for (uint column = 0; column < columns; ++column) logit += input[input_base + column] * weight[weight_base + column];
    }
    logits[lane] = logit;
    expert_ids[lane] = lane;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_bitonic_sort(logits, logits, expert_ids, false, lane, width);
    if (lane == 0) {
        float sum = 0.0f;
        const uint count = normalize_selected != 0 ? top_k : expert_count;
        for (uint index = 0; index < count; ++index) sum += exp(logits[index] - logits[0]);
        denominator = max(sum, 1.0e-20f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < top_k) {
        const ulong output = ulong(row) * top_k + lane;
        output_ids[output] = expert_ids[lane];
        output_weights[output] = exp(logits[lane] - logits[0]) / denominator * scaling_factor;
    }
}
kernel void moe_router_softmax_topk_logits_f32(
    device const float *input_logits [[buffer(0)]],
    device uint *output_ids [[buffer(1)]],
    device float *output_weights [[buffer(2)]],
    constant uint &expert_count [[buffer(3)]],
    constant uint &top_k [[buffer(4)]],
    constant float &scaling_factor [[buffer(5)]],
    constant uint &normalize_selected [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float logits[256];
    threadgroup uint expert_ids[256];
    threadgroup float denominator;
    logits[lane] = lane < expert_count ? input_logits[ulong(row) * expert_count + lane] : -INFINITY;
    expert_ids[lane] = lane;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    moe_bitonic_sort(logits, logits, expert_ids, false, lane, width);
    if (lane == 0) {
        float sum = 0.0f;
        const uint count = normalize_selected != 0 ? top_k : expert_count;
        for (uint index = 0; index < count; ++index) sum += exp(logits[index] - logits[0]);
        denominator = max(sum, 1.0e-20f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < top_k) {
        const ulong output = ulong(row) * top_k + lane;
        output_ids[output] = expert_ids[lane];
        output_weights[output] = exp(logits[lane] - logits[0]) / denominator * scaling_factor;
    }
}
kernel void gather_rows_f16(
    device const half *input [[buffer(0)]],
    device const uint *rows [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= row_count * columns) return;
    const uint row = index / columns;
    const uint column = index % columns;
    output[index] = input[ulong(rows[row]) * columns + column];
}
kernel void gather_rows_f32(
    device const float *input [[buffer(0)]], device const uint *rows [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]], constant uint &row_count [[buffer(4)]], uint index [[thread_position_in_grid]])
{
    if (index >= row_count * columns) return;
    const uint row = index / columns;
    const uint column = index % columns;
    output[index] = input[ulong(rows[row]) * columns + column];
}
kernel void gather_rows_f32_to_f16(
    device const float *input [[buffer(0)]], device const uint *rows [[buffer(1)]], device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]], constant uint &row_count [[buffer(4)]], uint index [[thread_position_in_grid]])
{
    if (index >= row_count * columns) return;
    const uint row = index / columns;
    const uint column = index % columns;
    output[index] = finite_f16(input[ulong(rows[row]) * columns + column]);
}
kernel void scatter_add_rows_weighted_f32(
    device const half *input [[buffer(0)]],
    device const uint *rows [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &row_count [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= row_count * columns) return;
    const uint row = index / columns;
    const uint column = index % columns;
    const ulong output_index = ulong(rows[row]) * columns + column;
    output[output_index] += float(input[index]) * weights[row];
}
kernel void moe_sort_topk_by_id(
    device uint *expert_ids [[buffer(0)]],
    device float *route_weights [[buffer(1)]],
    constant uint &top_k [[buffer(2)]])
{
    for (uint index = 1; index < top_k; ++index) {
        const uint expert = expert_ids[index];
        const float weight = route_weights[index];
        uint position = index;
        while (position > 0 && expert_ids[position - 1] > expert) {
            expert_ids[position] = expert_ids[position - 1];
            route_weights[position] = route_weights[position - 1];
            --position;
        }
        expert_ids[position] = expert;
        route_weights[position] = weight;
    }
}
"#;

use crate::backend::metal::api as metal;

use super::{MTLSize, MetalContext, MetalTensor, MetalTensorDType, launch_1d, mem, set_bytes, to_f16_tensor, validate_u32};

pub struct MetalRouting {
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
    pub rows: usize,
    pub top_k: usize,
    pub expert_ids_buffer: metal::Buffer,
    pub weights_buffer: metal::Buffer,
}

/// Router 的 weight 与 correction bias 均为常驻 F16，路由分数仍使用 F32。
pub fn moe_router_tensor_resident_f16_bias(ctx: &MetalContext, input: &MetalTensor, weight: &MetalTensor, bias: &MetalTensor, expert_count: usize, top_k: usize, scaling_factor: f32) -> Result<MetalRouting, String> {
    if weight.rows != expert_count || weight.cols != input.cols {
        return Err(format!("resident router weight 形状 [{},{}] 与 [{expert_count},{}] 不符", weight.rows, weight.cols, input.cols));
    }
    if bias.rows.checked_mul(bias.cols) != Some(expert_count) {
        return Err(format!("resident F16 router bias 形状 [{},{}] 与 experts={expert_count} 不符", bias.rows, bias.cols));
    }
    if expert_count == 0 || expert_count > 256 || top_k == 0 || top_k > expert_count {
        return Err(format!("Metal router 参数非法: experts={expert_count}, top_k={top_k}"));
    }
    let sort_size = expert_count.next_power_of_two();
    let pipeline_name = "moe_router_sigmoid_topk_f16_bias_f16";
    let pipeline = ctx.pipeline(pipeline_name)?;
    if sort_size > pipeline.max_total_threads_per_threadgroup() as usize {
        return Err(format!("Metal F16 router 需要 {sort_size} threads，设备最多支持 {}", pipeline.max_total_threads_per_threadgroup()));
    }
    let output_len = input.rows.checked_mul(top_k).ok_or_else(|| "Metal router 输出大小溢出".to_owned())?;
    let (output_ids, output_weights) = ctx.routing_readback_buffers(output_len);
    let columns = validate_u32("columns", input.cols)?;
    let experts = validate_u32("expert_count", expert_count)?;
    let top_k_u32 = validate_u32("top_k", top_k)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&weight.buffer), 0);
    encoder.set_buffer(2, Some(&bias.buffer), 0);
    encoder.set_buffer(3, Some(&output_ids), 0);
    encoder.set_buffer(4, Some(&output_weights), 0);
    set_bytes(&encoder, 5, &columns);
    set_bytes(&encoder, 6, &experts);
    set_bytes(&encoder, 7, &top_k_u32);
    set_bytes(&encoder, 8, &scaling_factor);
    encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new(sort_size as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={},columns={columns},experts={experts},top_k={top_k_u32}", input.rows);
    ctx.commit_and_force_wait_profiled(&command, pipeline_name, &shape, input.buffer.length() + weight.buffer.length() + bias.buffer.length(), output_ids.length() + output_weights.length());
    let expert_ids = unsafe { std::slice::from_raw_parts(output_ids.contents().cast::<u32>(), output_len) }.to_vec();
    let weights = unsafe { std::slice::from_raw_parts(output_weights.contents().cast::<f32>(), output_len) }.to_vec();
    Ok(MetalRouting { expert_ids, weights, rows: input.rows, top_k, expert_ids_buffer: output_ids, weights_buffer: output_weights })
}

/// F32 router 权重保持原始精度；scoring=1 为 sigmoid，2 为 sqrt(softplus)。
#[allow(clippy::too_many_arguments)]
pub fn moe_router_tensor_resident_f32(
    ctx: &MetalContext,
    input: &MetalTensor,
    weight: &metal::Buffer,
    weight_len: usize,
    bias: &metal::Buffer,
    bias_len: usize,
    expert_count: usize,
    top_k: usize,
    scaling_factor: f32,
    scoring: u32,
) -> Result<MetalRouting, String> {
    if weight_len != expert_count.checked_mul(input.cols).ok_or("F32 router weight 大小溢出")? || bias_len != expert_count {
        return Err(format!("F32 router shape 异常: weight={weight_len}, bias={bias_len}, experts={expert_count}, hidden={}", input.cols));
    }
    if expert_count == 0 || expert_count > 256 || top_k == 0 || top_k > expert_count {
        return Err(format!("Metal F32 router 参数非法: experts={expert_count}, top_k={top_k}"));
    }
    if !matches!(scoring, 1 | 2) {
        return Err(format!("Metal F32 router scoring={scoring} 不受支持"));
    }
    let input = to_f16_tensor(ctx, input)?;
    let sort_size = expert_count.next_power_of_two();
    let pipeline_name = "moe_router_bias_topk_f32_weight";
    let pipeline = ctx.pipeline(pipeline_name)?;
    if sort_size > pipeline.max_total_threads_per_threadgroup() as usize {
        return Err(format!("Metal F32 router 需要 {sort_size} threads，设备最多支持 {}", pipeline.max_total_threads_per_threadgroup()));
    }
    let output_len = input.rows.checked_mul(top_k).ok_or("F32 router 输出大小溢出")?;
    let (output_ids, output_weights) = ctx.routing_readback_buffers(output_len);
    let columns = validate_u32("columns", input.cols)?;
    let experts = validate_u32("expert_count", expert_count)?;
    let top_k_u32 = validate_u32("top_k", top_k)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(bias), 0);
    encoder.set_buffer(3, Some(&output_ids), 0);
    encoder.set_buffer(4, Some(&output_weights), 0);
    set_bytes(&encoder, 5, &columns);
    set_bytes(&encoder, 6, &experts);
    set_bytes(&encoder, 7, &top_k_u32);
    set_bytes(&encoder, 8, &scaling_factor);
    set_bytes(&encoder, 9, &scoring);
    encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new(sort_size as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={},columns={columns},experts={experts},top_k={top_k_u32}", input.rows);
    ctx.commit_and_force_wait_profiled(&command, pipeline_name, &shape, input.buffer.length() + weight.length() + bias.length(), output_ids.length() + output_weights.length());
    let expert_ids = unsafe { std::slice::from_raw_parts(output_ids.contents().cast::<u32>(), output_len) }.to_vec();
    let weights = unsafe { std::slice::from_raw_parts(output_weights.contents().cast::<f32>(), output_len) }.to_vec();
    Ok(MetalRouting { expert_ids, weights, rows: input.rows, top_k, expert_ids_buffer: output_ids, weights_buffer: output_weights })
}

/// Token-hash 层只计算表中指定的专家；权重仍按原始 sqrt(softplus) 分数归一化。
#[allow(clippy::too_many_arguments)]
pub fn moe_router_sqrt_softplus_selected_f32(
    ctx: &MetalContext,
    input: &MetalTensor,
    weight: &metal::Buffer,
    weight_len: usize,
    selected_experts: &[u32],
    expert_count: usize,
    top_k: usize,
    scaling_factor: f32,
) -> Result<MetalRouting, String> {
    if weight_len != expert_count.checked_mul(input.cols).ok_or("F32 selected router weight 大小溢出")? {
        return Err(format!("F32 selected router weight={weight_len}，期望 {}", expert_count * input.cols));
    }
    let output_len = input.rows.checked_mul(top_k).ok_or("F32 selected router 输出大小溢出")?;
    if top_k == 0 || top_k > 256 || selected_experts.len() != output_len || selected_experts.iter().any(|&expert| expert as usize >= expert_count) || !scaling_factor.is_finite() || scaling_factor <= 0.0 {
        return Err(format!("Metal selected router 参数非法: rows={} experts={expert_count} top_k={top_k} selected={}", input.rows, selected_experts.len()));
    }
    let input = to_f16_tensor(ctx, input)?;
    let selected_bytes = unsafe { std::slice::from_raw_parts(selected_experts.as_ptr().cast::<u8>(), std::mem::size_of_val(selected_experts)) };
    let selected = ctx.shared_buffer(selected_bytes);
    let (output_ids, output_weights) = ctx.routing_readback_buffers(output_len);
    let columns = validate_u32("columns", input.cols)?;
    let experts = validate_u32("expert_count", expert_count)?;
    let top_k_u32 = validate_u32("top_k", top_k)?;
    let width = top_k.next_power_of_two();
    let pipeline_name = "moe_router_sqrt_softplus_selected_f32_weight";
    let pipeline = ctx.pipeline(pipeline_name)?;
    if width > pipeline.max_total_threads_per_threadgroup() as usize {
        return Err(format!("Metal selected router 需要 {width} threads，设备最多支持 {}", pipeline.max_total_threads_per_threadgroup()));
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(&selected), 0);
    encoder.set_buffer(3, Some(&output_ids), 0);
    encoder.set_buffer(4, Some(&output_weights), 0);
    set_bytes(&encoder, 5, &columns);
    set_bytes(&encoder, 6, &experts);
    set_bytes(&encoder, 7, &top_k_u32);
    set_bytes(&encoder, 8, &scaling_factor);
    encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new(width as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={},columns={columns},experts={experts},top_k={top_k_u32}", input.rows);
    ctx.commit_and_force_wait_profiled(&command, pipeline_name, &shape, input.buffer.length() + weight.length() + selected.length(), output_ids.length() + output_weights.length());
    let expert_ids = unsafe { std::slice::from_raw_parts(output_ids.contents().cast::<u32>(), output_len) }.to_vec();
    let weights = unsafe { std::slice::from_raw_parts(output_weights.contents().cast::<f32>(), output_len) }.to_vec();
    Ok(MetalRouting { expert_ids, weights, rows: input.rows, top_k, expert_ids_buffer: output_ids, weights_buffer: output_weights })
}

pub fn moe_router_softmax_tensor_resident(ctx: &MetalContext, input: &MetalTensor, weight: &MetalTensor, expert_count: usize, top_k: usize, scaling_factor: f32, normalize_selected: bool) -> Result<MetalRouting, String> {
    if weight.rows != expert_count || weight.cols != input.cols {
        return Err(format!("softmax router weight 形状 [{},{}] 与 [{expert_count},{}] 不符", weight.rows, weight.cols, input.cols));
    }
    let (sort_size, output_len) = validate_softmax_router(input, expert_count, top_k, scaling_factor)?;
    let pipeline_name = "moe_router_softmax_topk_f16";
    let pipeline = ctx.pipeline(pipeline_name)?;
    if sort_size > pipeline.max_total_threads_per_threadgroup() as usize {
        return Err(format!("Metal softmax router 需要 {sort_size} threads，设备最多支持 {}", pipeline.max_total_threads_per_threadgroup()));
    }
    let (output_ids, output_weights) = ctx.routing_readback_buffers(output_len);
    encode_softmax_router(ctx, pipeline_name, input, &weight.buffer, expert_count, top_k, scaling_factor, normalize_selected, sort_size, &output_ids, &output_weights, true)?;
    let expert_ids = unsafe { std::slice::from_raw_parts(output_ids.contents().cast::<u32>(), output_len) }.to_vec();
    let weights = unsafe { std::slice::from_raw_parts(output_weights.contents().cast::<f32>(), output_len) }.to_vec();
    Ok(MetalRouting { expert_ids, weights, rows: input.rows, top_k, expert_ids_buffer: output_ids, weights_buffer: output_weights })
}

#[allow(clippy::too_many_arguments)]
pub fn moe_router_softmax_tensor_resident_f32(
    ctx: &MetalContext,
    input: &MetalTensor,
    weight: &metal::Buffer,
    weight_len: usize,
    expert_count: usize,
    top_k: usize,
    scaling_factor: f32,
    normalize_selected: bool,
) -> Result<MetalRouting, String> {
    if weight_len != expert_count.checked_mul(input.cols).ok_or("F32 softmax router weight 大小溢出")? {
        return Err(format!("F32 softmax router weight={weight_len}，期望 {}", expert_count * input.cols));
    }
    let (sort_size, output_len) = validate_softmax_router(input, expert_count, top_k, scaling_factor)?;
    if input.rows >= 16 {
        let converted;
        let input = if input.dtype == MetalTensorDType::F32 {
            input
        } else {
            converted = super::to_f32_tensor(ctx, input)?;
            &converted
        };
        let logits = ctx.tensor_kernel_output_f32(input.rows, expert_count);
        let (output_ids, output_weights) = ctx.routing_readback_buffers(output_len);
        let command = ctx.command_buffer();
        super::mps::encode_f32_matmul_transposed(&command, &ctx.device, &input.buffer, input.rows, input.cols, weight, expert_count, &logits.buffer)?;
        let pipeline_name = "moe_router_softmax_topk_logits_f32";
        let pipeline = ctx.pipeline(pipeline_name)?;
        let experts = validate_u32("expert_count", expert_count)?;
        let top_k_u32 = validate_u32("top_k", top_k)?;
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&logits.buffer), 0);
        encoder.set_buffer(1, Some(&output_ids), 0);
        encoder.set_buffer(2, Some(&output_weights), 0);
        set_bytes(&encoder, 3, &experts);
        set_bytes(&encoder, 4, &top_k_u32);
        set_bytes(&encoder, 5, &scaling_factor);
        set_bytes(&encoder, 6, &u32::from(normalize_selected));
        encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new(sort_size as u64, 1, 1));
        encoder.end_encoding();
        let shape = format!("rows={},columns={},experts={experts},top_k={top_k_u32}", input.rows, input.cols);
        ctx.commit_and_force_wait_profiled(&command, "moe_router_softmax_topk_mps_f32", &shape, input.buffer.length() + weight.length(), logits.buffer.length() + output_ids.length() + output_weights.length());
        let expert_ids = unsafe { std::slice::from_raw_parts(output_ids.contents().cast::<u32>(), output_len) }.to_vec();
        let weights = unsafe { std::slice::from_raw_parts(output_weights.contents().cast::<f32>(), output_len) }.to_vec();
        return Ok(MetalRouting { expert_ids, weights, rows: input.rows, top_k, expert_ids_buffer: output_ids, weights_buffer: output_weights });
    }
    if input.dtype == MetalTensorDType::F32 && input.rows == 1 {
        const LOGIT_THREADS: usize = 64;
        let logits_pipeline = ctx.pipeline("moe_router_logits_parallel_f32_input_weight")?;
        let topk_pipeline = ctx.pipeline("moe_router_softmax_topk_logits_f32")?;
        if LOGIT_THREADS as u64 > logits_pipeline.max_total_threads_per_threadgroup() || sort_size as u64 > topk_pipeline.max_total_threads_per_threadgroup() {
            return Err("Metal F32 decode router 超过 pipeline threadgroup 上限".to_owned());
        }
        let logits = ctx.tensor_kernel_output_f32(1, expert_count);
        let (output_ids, output_weights) = ctx.routing_readback_buffers(output_len);
        let columns = validate_u32("columns", input.cols)?;
        let experts = validate_u32("expert_count", expert_count)?;
        let top_k_u32 = validate_u32("top_k", top_k)?;
        let command = ctx.command_buffer();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&logits_pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(&logits.buffer), 0);
        set_bytes(&encoder, 3, &columns);
        set_bytes(&encoder, 4, &experts);
        encoder.dispatch_thread_groups(MTLSize::new(expert_count as u64, 1, 1), MTLSize::new(LOGIT_THREADS as u64, 1, 1));
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&topk_pipeline);
        encoder.set_buffer(0, Some(&logits.buffer), 0);
        encoder.set_buffer(1, Some(&output_ids), 0);
        encoder.set_buffer(2, Some(&output_weights), 0);
        set_bytes(&encoder, 3, &experts);
        set_bytes(&encoder, 4, &top_k_u32);
        set_bytes(&encoder, 5, &scaling_factor);
        set_bytes(&encoder, 6, &u32::from(normalize_selected));
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(sort_size as u64, 1, 1));
        encoder.end_encoding();

        let shape = format!("rows=1,columns={columns},experts={experts},top_k={top_k_u32}");
        ctx.commit_and_force_wait_profiled(&command, "moe_router_softmax_topk_parallel_f32", &shape, input.buffer.length() + weight.length(), logits.buffer.length() + output_ids.length() + output_weights.length());
        let expert_ids = unsafe { std::slice::from_raw_parts(output_ids.contents().cast::<u32>(), output_len) }.to_vec();
        let weights = unsafe { std::slice::from_raw_parts(output_weights.contents().cast::<f32>(), output_len) }.to_vec();
        return Ok(MetalRouting { expert_ids, weights, rows: 1, top_k, expert_ids_buffer: output_ids, weights_buffer: output_weights });
    }
    let pipeline_name = if input.dtype == MetalTensorDType::F32 { "moe_router_softmax_topk_f32_input_weight" } else { "moe_router_softmax_topk_f32_weight" };
    let pipeline = ctx.pipeline(pipeline_name)?;
    if sort_size > pipeline.max_total_threads_per_threadgroup() as usize {
        return Err(format!("Metal F32 softmax router 需要 {sort_size} threads，设备最多支持 {}", pipeline.max_total_threads_per_threadgroup()));
    }
    let (output_ids, output_weights) = ctx.routing_readback_buffers(output_len);
    encode_softmax_router(ctx, pipeline_name, input, weight, expert_count, top_k, scaling_factor, normalize_selected, sort_size, &output_ids, &output_weights, true)?;
    let expert_ids = unsafe { std::slice::from_raw_parts(output_ids.contents().cast::<u32>(), output_len) }.to_vec();
    let weights = unsafe { std::slice::from_raw_parts(output_weights.contents().cast::<f32>(), output_len) }.to_vec();
    Ok(MetalRouting { expert_ids, weights, rows: input.rows, top_k, expert_ids_buffer: output_ids, weights_buffer: output_weights })
}

/// Decode route 不做 CPU readback；同一 queue 上的 indexed expert 直接消费排序后的结果。
#[allow(clippy::too_many_arguments)]
pub fn moe_router_softmax_decode_resident_f32(
    ctx: &MetalContext,
    input: &MetalTensor,
    weight: &metal::Buffer,
    weight_len: usize,
    expert_count: usize,
    top_k: usize,
    scaling_factor: f32,
    normalize_selected: bool,
) -> Result<(metal::Buffer, metal::Buffer), String> {
    if input.dtype != MetalTensorDType::F32 || input.rows != 1 {
        return Err(format!("Metal resident decode router 需要单行 F32 input，实际 [{},{}] {:?}", input.rows, input.cols, input.dtype));
    }
    if weight_len != expert_count.checked_mul(input.cols).ok_or("resident decode router weight 大小溢出")? {
        return Err(format!("resident decode router weight={weight_len}，期望 {}", expert_count * input.cols));
    }
    let (sort_size, output_len) = validate_softmax_router(input, expert_count, top_k, scaling_factor)?;
    const LOGIT_THREADS: usize = 64;
    let logits_pipeline = ctx.pipeline("moe_router_logits_parallel_f32_input_weight")?;
    let topk_pipeline = ctx.pipeline("moe_router_softmax_topk_logits_f32")?;
    let sort_pipeline = ctx.pipeline("moe_sort_topk_by_id")?;
    if LOGIT_THREADS as u64 > logits_pipeline.max_total_threads_per_threadgroup() || sort_size as u64 > topk_pipeline.max_total_threads_per_threadgroup() {
        return Err("Metal resident decode router 超过 pipeline threadgroup 上限".to_owned());
    }
    let logits = ctx.tensor_kernel_output_f32(1, expert_count);
    let (output_ids, output_weights) = ctx.routing_readback_buffers(output_len);
    let columns = validate_u32("columns", input.cols)?;
    let experts = validate_u32("expert_count", expert_count)?;
    let top_k_u32 = validate_u32("top_k", top_k)?;
    let command = ctx.command_buffer();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&logits_pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(&logits.buffer), 0);
    set_bytes(&encoder, 3, &columns);
    set_bytes(&encoder, 4, &experts);
    encoder.dispatch_thread_groups(MTLSize::new(expert_count as u64, 1, 1), MTLSize::new(LOGIT_THREADS as u64, 1, 1));
    encoder.end_encoding();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&topk_pipeline);
    encoder.set_buffer(0, Some(&logits.buffer), 0);
    encoder.set_buffer(1, Some(&output_ids), 0);
    encoder.set_buffer(2, Some(&output_weights), 0);
    set_bytes(&encoder, 3, &experts);
    set_bytes(&encoder, 4, &top_k_u32);
    set_bytes(&encoder, 5, &scaling_factor);
    set_bytes(&encoder, 6, &u32::from(normalize_selected));
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(sort_size as u64, 1, 1));
    encoder.end_encoding();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&sort_pipeline);
    encoder.set_buffer(0, Some(&output_ids), 0);
    encoder.set_buffer(1, Some(&output_weights), 0);
    set_bytes(&encoder, 2, &top_k_u32);
    encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(
        &command,
        "moe_router_softmax_topk_resident_f32",
        &format!("rows=1,columns={columns},experts={experts},top_k={top_k_u32}"),
        input.buffer.length() + weight.length(),
        logits.buffer.length() + output_ids.length() + output_weights.length(),
    );
    Ok((output_ids, output_weights))
}

pub fn prewarm_resident_decode_router(ctx: &MetalContext) -> Result<(), String> {
    for pipeline in ["moe_router_logits_parallel_f32_input_weight", "moe_router_softmax_topk_logits_f32", "moe_sort_topk_by_id"] {
        drop(ctx.pipeline(pipeline)?);
    }
    Ok(())
}

fn validate_softmax_router(input: &MetalTensor, expert_count: usize, top_k: usize, scaling_factor: f32) -> Result<(usize, usize), String> {
    if input.rows == 0 || expert_count == 0 || expert_count > 256 || top_k == 0 || top_k > expert_count {
        return Err(format!("Metal softmax router 参数非法: rows={},experts={expert_count},top_k={top_k}", input.rows));
    }
    if !scaling_factor.is_finite() || scaling_factor <= 0.0 {
        return Err(format!("Metal softmax router scaling_factor 非法: {scaling_factor}"));
    }
    Ok((expert_count.next_power_of_two(), input.rows.checked_mul(top_k).ok_or("Metal softmax router 输出大小溢出")?))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_softmax_router(
    ctx: &MetalContext,
    pipeline_name: &str,
    input: &MetalTensor,
    weight: &metal::Buffer,
    expert_count: usize,
    top_k: usize,
    scaling_factor: f32,
    normalize_selected: bool,
    sort_size: usize,
    output_ids: &metal::Buffer,
    output_weights: &metal::Buffer,
    wait: bool,
) -> Result<(), String> {
    let pipeline = ctx.pipeline(pipeline_name)?;
    let columns = validate_u32("columns", input.cols)?;
    let experts = validate_u32("expert_count", expert_count)?;
    let top_k_u32 = validate_u32("top_k", top_k)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(output_ids), 0);
    encoder.set_buffer(3, Some(output_weights), 0);
    set_bytes(&encoder, 4, &columns);
    set_bytes(&encoder, 5, &experts);
    set_bytes(&encoder, 6, &top_k_u32);
    set_bytes(&encoder, 7, &scaling_factor);
    set_bytes(&encoder, 8, &u32::from(normalize_selected));
    encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new(sort_size as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={},columns={columns},experts={experts},top_k={top_k_u32}", input.rows);
    if wait {
        ctx.commit_and_force_wait_profiled(&command, pipeline_name, &shape, input.buffer.length() + weight.length(), output_ids.length() + output_weights.length());
    } else {
        ctx.commit_and_wait_profiled(&command, pipeline_name, &shape, input.buffer.length() + weight.length(), output_ids.length() + output_weights.length());
    }
    Ok(())
}

pub fn argmax_tensor(ctx: &MetalContext, input: &MetalTensor, excluded: &[u32]) -> Result<u32, String> {
    let length = input.len();
    if length == 0 {
        return Err("Metal argmax 输入为空".to_owned());
    }
    let output = ctx.token_readback_buffer();
    argmax_tensor_encode(ctx, input, excluded, &output, 0, true)?;
    Ok(unsafe { *(output.contents().cast::<u32>()) })
}

/// argmax 写入调用方给的设备 buffer;`wait=false` 时只编码提交不等待,
/// 供 decode 流水线把读回推迟一轮(GPU 继续跑下一 token 的 decode round)。
pub fn argmax_tensor_into(ctx: &MetalContext, input: &MetalTensor, excluded: &[u32], output: &metal::Buffer) -> Result<(), String> {
    argmax_tensor_encode(ctx, input, excluded, output, 0, false)
}

/// 带输出偏移的 argmax:写入 output[output_offset..],供异步流水线按 token 位置
/// 写入不复用的读回区(覆写竞态见 minicpm5 token_readback 注释)。
pub fn argmax_tensor_into_offset(ctx: &MetalContext, input: &MetalTensor, excluded: &[u32], output: &metal::Buffer, output_offset: u64) -> Result<(), String> {
    argmax_tensor_encode(ctx, input, excluded, output, output_offset, false)
}

fn argmax_tensor_encode(ctx: &MetalContext, input: &MetalTensor, excluded: &[u32], output: &metal::Buffer, output_offset: u64, wait: bool) -> Result<(), String> {
    let length = input.len();
    if length == 0 {
        return Err("Metal argmax 输入为空".to_owned());
    }
    if excluded.len() >= length || excluded.iter().any(|&token| token as usize >= length) {
        return Err(format!("Metal argmax 禁止 token 非法: length={length}, excluded={excluded:?}"));
    }
    let length_u32 = validate_u32("Metal argmax length", length)?;
    let excluded_count = validate_u32("Metal argmax excluded count", excluded.len())?;
    let no_excluded = [u32::MAX];
    let excluded_bytes = if excluded.is_empty() { &no_excluded[..] } else { excluded };
    let input_bf16 = u32::from(input.dtype == MetalTensorDType::Bf16);
    // 大词表走两阶段:单 TG 串行扫是纯访存延迟链(130k 词表 ~134µs);
    // 32 TG 分段 + 归约,全序(moe_ordered_before)保证与单段逐位一致。
    const TWO_STAGE_MIN: usize = 32768;
    const PARTIALS: usize = 32;
    if input.dtype != MetalTensorDType::F32 && length >= TWO_STAGE_MIN {
        let partial_values = ctx.shared_buffer_uninit(PARTIALS * mem::size_of::<f32>());
        let partial_indices = ctx.shared_buffer_uninit(PARTIALS * mem::size_of::<u32>());
        // 诊断:partials 保活,排除 scratch 页面复用
        let partials_keep = if std::env::var_os("ZLLM_DEBUG_KEEP_PARTIALS").is_some() { Some((partial_values.clone(), partial_indices.clone())) } else { None };
        let partial_pipeline = ctx.pipeline("argmax_partial_f16_bf16")?;
        let reduce_pipeline = ctx.pipeline("argmax_reduce_f32")?;
        if partial_pipeline.max_total_threads_per_threadgroup() < 256 || reduce_pipeline.max_total_threads_per_threadgroup() < 256 {
            return Err("Metal 两阶段 argmax 需要至少 256 threads/threadgroup".to_owned());
        }
        let partial_count = validate_u32("Metal argmax partials", PARTIALS)?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&partial_pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&partial_values), 0);
        encoder.set_buffer(2, Some(&partial_indices), 0);
        set_bytes(&encoder, 3, &length_u32);
        encoder.set_bytes(4, std::mem::size_of_val(excluded_bytes) as u64, excluded_bytes.as_ptr().cast());
        set_bytes(&encoder, 5, &excluded_count);
        set_bytes(&encoder, 6, &input_bf16);
        encoder.dispatch_thread_groups(MTLSize::new(PARTIALS as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        // 两阶段之间拆 CB:stage2 与 stage1 同 CB 时存在实测乱序(异步流水线偶发
        // argmax 读到未写完的 partials,token 分叉;详见 minicpm5 竞态定位记录)。
        // 拆 CB 只在 deferred 批模式有意义;非 deferred 时 command 由末尾统一提交。
        let command = if ctx.deferred_waits_enabled() {
            ctx.submit_batch();
            ctx.command_buffer()
        } else {
            command
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&reduce_pipeline);
        encoder.set_buffer(0, Some(&partial_values), 0);
        encoder.set_buffer(1, Some(&partial_indices), 0);
        encoder.set_buffer(2, Some(output), output_offset);
        set_bytes(&encoder, 3, &partial_count);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        if wait {
            ctx.commit_and_force_wait_profiled(&command, "argmax_two_stage_f16_bf16", &format!("length={length},excluded={excluded_count}"), input.buffer.length(), output.length());
        } else {
            ctx.commit_and_wait_profiled(&command, "argmax_two_stage_f16_bf16", &format!("length={length},excluded={excluded_count}"), input.buffer.length(), output.length());
        }
        if let Some(keep) = partials_keep {
            ctx.retain_debug_buffers(vec![keep.0, keep.1]);
        }
        return Ok(());
    }
    let pipeline_name = if input.dtype == MetalTensorDType::F32 { "argmax_f32" } else { "argmax_f16_bf16" };
    let pipeline = ctx.pipeline(pipeline_name)?;
    if pipeline.max_total_threads_per_threadgroup() < 256 {
        return Err("Metal argmax 需要至少 256 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(output), output_offset);
    set_bytes(&encoder, 2, &length_u32);
    encoder.set_bytes(3, std::mem::size_of_val(excluded_bytes) as u64, excluded_bytes.as_ptr().cast());
    set_bytes(&encoder, 4, &excluded_count);
    set_bytes(&encoder, 5, &input_bf16);
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    if wait {
        ctx.commit_and_force_wait_profiled(&command, pipeline_name, &format!("length={length},excluded={excluded_count}"), input.buffer.length(), output.length());
    } else {
        ctx.commit_and_wait_profiled(&command, pipeline_name, &format!("length={length},excluded={excluded_count}"), input.buffer.length(), output.length());
    }
    Ok(())
}

pub fn sample_top_p_tensor(ctx: &MetalContext, input: &MetalTensor, temperature: f32, top_p: f32, random: f32) -> Result<u32, String> {
    let length = input.len();
    if length == 0 || !temperature.is_finite() || temperature <= 0.0 || !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) || !random.is_finite() || !(0.0..1.0).contains(&random) {
        return Err("Metal top-p sampling 参数非法".to_owned());
    }
    let length_u32 = validate_u32("Metal top-p sampling length", length)?;
    let output = ctx.token_readback_buffer();
    let pipeline = ctx.pipeline("sample_top_p_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 256 {
        return Err("Metal top-p sampling 需要至少 256 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&output), 0);
    set_bytes(&encoder, 2, &length_u32);
    set_bytes(&encoder, 3, &temperature);
    set_bytes(&encoder, 4, &top_p);
    set_bytes(&encoder, 5, &random);
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_force_wait_profiled(&command, "sample_top_p_f16", &format!("length={length}"), input.buffer.length(), output.length());
    Ok(unsafe { *(output.contents().cast::<u32>()) })
}

pub fn gather_rows_tensor(ctx: &MetalContext, input: &MetalTensor, rows: &[u32]) -> Result<MetalTensor, String> {
    if rows.is_empty() || rows.iter().any(|&row| row as usize >= input.rows) {
        return Err(format!("Metal gather rows 非法: input_rows={}, rows={rows:?}", input.rows));
    }
    let output = match input.dtype {
        MetalTensorDType::F16 => ctx.tensor_zeros(rows.len(), input.cols),
        MetalTensorDType::Bf16 => ctx.tensor_zeros_bf16(rows.len(), input.cols),
        MetalTensorDType::F32 => ctx.tensor_zeros_f32(rows.len(), input.cols),
    };
    let row_bytes = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) };
    let row_buffer = ctx.shared_buffer(row_bytes);
    let columns = validate_u32("gather columns", input.cols)?;
    let count = validate_u32("gather rows", rows.len())?;
    let pipeline_name = if input.dtype == MetalTensorDType::F32 { "gather_rows_f32" } else { "gather_rows_f16" };
    let pipeline = ctx.pipeline(pipeline_name)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&row_buffer), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &columns);
    set_bytes(&encoder, 4, &count);
    let total = rows.len().checked_mul(input.cols).ok_or("Metal gather 大小溢出")?;
    encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command, pipeline_name, &format!("rows={},cols={}", rows.len(), input.cols), input.buffer.length() + row_buffer.length(), output.buffer.length());
    Ok(output)
}

pub fn gather_rows_tensor_f16(ctx: &MetalContext, input: &MetalTensor, rows: &[u32]) -> Result<MetalTensor, String> {
    if input.dtype == MetalTensorDType::F16 {
        return gather_rows_tensor(ctx, input, rows);
    }
    if input.dtype == MetalTensorDType::Bf16 {
        return gather_rows_tensor(ctx, &to_f16_tensor(ctx, input)?, rows);
    }
    if rows.is_empty() || rows.iter().any(|&row| row as usize >= input.rows) {
        return Err(format!("Metal gather F32->F16 rows 非法: input_rows={}, rows={rows:?}", input.rows));
    }
    let output = ctx.tensor_kernel_output(rows.len(), input.cols);
    let row_bytes = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) };
    let row_buffer = ctx.shared_buffer(row_bytes);
    let columns = validate_u32("gather F32->F16 columns", input.cols)?;
    let count = validate_u32("gather F32->F16 rows", rows.len())?;
    let total = rows.len().checked_mul(input.cols).ok_or("Metal gather F32->F16 大小溢出")?;
    let shape = format!("rows={},cols={}", rows.len(), input.cols);
    launch_1d(ctx, "gather_rows_f32_to_f16", &shape, total, input.buffer.length() + row_buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&row_buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &count);
    })?;
    Ok(output)
}

pub fn gather_rows_batch_tensor_f16(ctx: &MetalContext, input: &MetalTensor, rows: &[Vec<u32>]) -> Result<Vec<MetalTensor>, String> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    if rows.iter().any(|items| items.is_empty() || items.iter().any(|&row| row as usize >= input.rows)) {
        return Err(format!("Metal batch gather rows 非法: input_rows={}, groups={}", input.rows, rows.len()));
    }
    let input = to_f16_tensor(ctx, input)?;
    let flat_rows = rows.iter().flatten().copied().collect::<Vec<_>>();
    let row_bytes = unsafe { std::slice::from_raw_parts(flat_rows.as_ptr().cast::<u8>(), std::mem::size_of_val(flat_rows.as_slice())) };
    let row_buffer = ctx.shared_buffer(row_bytes);
    let columns = validate_u32("batch gather columns", input.cols)?;
    let pipeline = ctx.pipeline("gather_rows_f16")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    let mut offset = 0usize;
    let mut outputs = Vec::with_capacity(rows.len());
    for items in rows {
        let output = ctx.tensor_kernel_output(items.len(), input.cols);
        encoder.set_buffer(1, Some(&row_buffer), (offset * mem::size_of::<u32>()) as u64);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(&encoder, 3, &columns);
        set_bytes(&encoder, 4, &validate_u32("batch gather rows", items.len())?);
        let total = items.len().checked_mul(input.cols).ok_or("Metal batch gather 大小溢出")?;
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(128, 1, 1));
        offset += items.len();
        outputs.push(output);
    }
    encoder.end_encoding();
    let output_bytes = outputs.iter().map(|output| output.buffer.length()).sum();
    ctx.commit_and_wait_profiled(&command, "gather_rows_batch_f16", &format!("groups={},rows={},cols={}", rows.len(), flat_rows.len(), input.cols), input.buffer.length() + row_buffer.length(), output_bytes);
    Ok(outputs)
}

pub struct MetalF32Accumulator {
    pub(super) buffer: metal::Buffer,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

pub fn moe_accumulator_zeros(ctx: &MetalContext, rows: usize, cols: usize) -> Result<MetalF32Accumulator, String> {
    let bytes = rows.checked_mul(cols).and_then(|count| count.checked_mul(mem::size_of::<f32>())).ok_or("Metal MoE accumulator 大小溢出")?;
    Ok(MetalF32Accumulator { buffer: ctx.shared_buffer_zeros(bytes), rows, cols })
}

pub fn scatter_add_rows_weighted_f32(ctx: &MetalContext, output: &MetalF32Accumulator, input: &MetalTensor, rows: &[u32], weights: &[f32]) -> Result<(), String> {
    if rows.is_empty() || rows.len() != weights.len() || input.rows != rows.len() || input.cols != output.cols || rows.iter().any(|&row| row as usize >= output.rows) {
        return Err(format!("Metal scatter shape 非法: output=[{},{}], input=[{},{}], rows={}, weights={}", output.rows, output.cols, input.rows, input.cols, rows.len(), weights.len()));
    }
    let row_bytes = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) };
    let weight_bytes = unsafe { std::slice::from_raw_parts(weights.as_ptr().cast::<u8>(), std::mem::size_of_val(weights)) };
    let row_buffer = ctx.shared_buffer(row_bytes);
    let weight_buffer = ctx.shared_buffer(weight_bytes);
    let columns = validate_u32("scatter columns", input.cols)?;
    let count = validate_u32("scatter rows", rows.len())?;
    let pipeline = ctx.pipeline("scatter_add_rows_weighted_f32")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&row_buffer), 0);
    encoder.set_buffer(2, Some(&weight_buffer), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &columns);
    set_bytes(&encoder, 5, &count);
    let total = rows.len().checked_mul(input.cols).ok_or("Metal scatter 大小溢出")?;
    encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(
        &command,
        "scatter_add_rows_weighted_f32",
        &format!("rows={},cols={}", rows.len(), input.cols),
        input.buffer.length() + row_buffer.length() + weight_buffer.length() + output.buffer.length(),
        output.buffer.length(),
    );
    Ok(())
}

pub fn scatter_add_rows_batch_weighted_f32(ctx: &MetalContext, output: &MetalF32Accumulator, inputs: &[MetalTensor], rows: &[Vec<u32>], weights: &[Vec<f32>]) -> Result<(), String> {
    if inputs.len() != rows.len() || rows.len() != weights.len() {
        return Err(format!("Metal batch scatter 数量异常: inputs={}, rows={}, weights={}", inputs.len(), rows.len(), weights.len()));
    }
    if inputs.is_empty() {
        return Ok(());
    }
    for ((input, items), route_weights) in inputs.iter().zip(rows).zip(weights) {
        if items.is_empty() || items.len() != route_weights.len() || input.rows != items.len() || input.cols != output.cols || items.iter().any(|&row| row as usize >= output.rows) {
            return Err(format!("Metal batch scatter shape 非法: output=[{},{}], input=[{},{}], rows={}, weights={}", output.rows, output.cols, input.rows, input.cols, items.len(), route_weights.len()));
        }
    }
    let flat_rows = rows.iter().flatten().copied().collect::<Vec<_>>();
    let flat_weights = weights.iter().flatten().copied().collect::<Vec<_>>();
    let row_bytes = unsafe { std::slice::from_raw_parts(flat_rows.as_ptr().cast::<u8>(), std::mem::size_of_val(flat_rows.as_slice())) };
    let weight_bytes = unsafe { std::slice::from_raw_parts(flat_weights.as_ptr().cast::<u8>(), std::mem::size_of_val(flat_weights.as_slice())) };
    let row_buffer = ctx.shared_buffer(row_bytes);
    let weight_buffer = ctx.shared_buffer(weight_bytes);
    let columns = validate_u32("batch scatter columns", output.cols)?;
    let pipeline = ctx.pipeline("scatter_add_rows_weighted_f32")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    let mut offset = 0usize;
    for (input, items) in inputs.iter().zip(rows) {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&row_buffer), (offset * mem::size_of::<u32>()) as u64);
        encoder.set_buffer(2, Some(&weight_buffer), (offset * mem::size_of::<f32>()) as u64);
        set_bytes(&encoder, 4, &columns);
        set_bytes(&encoder, 5, &validate_u32("batch scatter rows", items.len())?);
        let total = items.len().checked_mul(output.cols).ok_or("Metal batch scatter 大小溢出")?;
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(128, 1, 1));
        offset += items.len();
    }
    encoder.end_encoding();
    let input_bytes = inputs.iter().map(|input| input.buffer.length()).sum::<u64>();
    ctx.commit_and_wait_profiled(
        &command,
        "scatter_add_rows_batch_weighted_f32",
        &format!("groups={},rows={},cols={}", rows.len(), flat_rows.len(), output.cols),
        input_bytes + row_buffer.length() + weight_buffer.length() + output.buffer.length(),
        output.buffer.length(),
    );
    Ok(())
}

pub fn finish_moe_accumulator(ctx: &MetalContext, accumulator: MetalF32Accumulator, wait_for: Option<&metal::FenceRef>) -> Result<MetalTensor, String> {
    let count = accumulator.rows.checked_mul(accumulator.cols).ok_or("Metal MoE accumulator 元素数溢出")?;
    let count_u32 = validate_u32("Metal MoE accumulator count", count)?;
    let output = ctx.tensor_kernel_output(accumulator.rows, accumulator.cols);
    launch_1d(ctx, "cast_f32_to_f16", &format!("[{},{}]", accumulator.rows, accumulator.cols), count, accumulator.buffer.length(), output.buffer.length(), |encoder| {
        if let Some(fence) = wait_for {
            encoder.wait_for_fence(fence);
        }
        encoder.set_buffer(0, Some(&accumulator.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count_u32);
    })?;
    Ok(output)
}

#[cfg(test)]
mod deepseek_router_tests {
    use super::*;
    use crate::moe::routing::{route_sqrt_softplus_bias_logits, route_sqrt_softplus_selected};

    fn f32_buffer(ctx: &MetalContext, values: &[f32]) -> metal::Buffer {
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
        ctx.shared_buffer(bytes)
    }

    #[test]
    fn deepseek_sqrt_softplus_matches_cpu_reference() {
        let ctx = MetalContext::new_default().unwrap();
        let input_values = [1.0, 2.0, -1.0, 0.5];
        let weight_values = [0.5, 1.0, -1.0, 0.25, 0.75, -0.5, 0.1, 0.2];
        let bias_values = [0.0, 0.2, -0.1, 0.05];
        let input = ctx.tensor_from_f32(&input_values, 2, 2).unwrap();
        let weight = f32_buffer(&ctx, &weight_values);
        let bias = f32_buffer(&ctx, &bias_values);
        let actual = moe_router_tensor_resident_f32(&ctx, &input, &weight, weight_values.len(), &bias, bias_values.len(), 4, 2, 1.5, 2).unwrap();
        for row in 0..2 {
            let x = &input_values[row * 2..(row + 1) * 2];
            let logits = weight_values.chunks_exact(2).map(|weight| x.iter().zip(weight).map(|(x, weight)| x * weight).sum()).collect::<Vec<f32>>();
            let expected = route_sqrt_softplus_bias_logits(&logits, &bias_values, 2, 1.5).unwrap();
            assert_eq!(&actual.expert_ids[row * 2..(row + 1) * 2], expected.experts);
            for (actual, expected) in actual.weights[row * 2..(row + 1) * 2].iter().zip(expected.weights) {
                assert!((*actual - expected).abs() < 2.0e-3, "actual={actual} expected={expected}");
            }
        }

        let selected = [3, 0, 2, 1];
        let actual = moe_router_sqrt_softplus_selected_f32(&ctx, &input, &weight, weight_values.len(), &selected, 4, 2, 1.5).unwrap();
        let expected = route_sqrt_softplus_selected(&input_values, 2, 2, &weight_values, 4, &selected, 2, 1.5).unwrap();
        assert_eq!(actual.expert_ids, expected.expert_ids);
        for (actual, expected) in actual.weights.iter().zip(expected.weights) {
            assert!((*actual - expected).abs() < 2.0e-3, "actual={actual} expected={expected}");
        }
    }

    #[test]
    fn shared_ordering_keeps_score_descending_and_id_ascending() {
        let ctx = MetalContext::new_default().unwrap();
        let input = ctx.tensor_from_f32(&[1.0, 0.0], 1, 2).unwrap();
        let weight = ctx.tensor_from_f32(&[1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0], 4, 2).unwrap();
        let routing = moe_router_softmax_tensor_resident(&ctx, &input, &weight, 4, 3, 1.0, true).unwrap();
        assert_eq!(routing.expert_ids, [2, 0, 1]);

        let logits = ctx.tensor_from_f32_preserve(&[1.0, 2.0, 2.0, 0.0], 1, 4).unwrap();
        assert_eq!(argmax_tensor(&ctx, &logits, &[]).unwrap(), 1);
    }

    /// 两阶段 argmax(>=32768 触发):与单段路径同一全序,结果必须逐位一致;
    /// 覆盖排除列表、并列取小 index、负值。
    #[test]
    fn two_stage_argmax_matches_cpu() {
        let ctx = MetalContext::new_default().unwrap();
        let length = 130560usize;
        let mut rng: u32 = 0xA9E5;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0 - 1.0) * 4.0
        };
        let mut values: Vec<f32> = (0..length).map(|_| next()).collect();
        // 构造并列最大值与排除项
        values[123] = 9.0;
        values[65432] = 9.0;
        values[7] = 10.0;
        let excluded = [7u32, 99999];
        let expected = values.iter().enumerate().filter(|(index, _)| !excluded.contains(&(*index as u32))).fold(0u32, |best, (index, &value)| {
            let best_value = values[best as usize];
            if value > best_value || (value == best_value && (index as u32) < best) { index as u32 } else { best }
        });
        assert_eq!(expected, 123);
        let input = ctx.tensor_from_f32(&values, 1, length).unwrap();
        assert_eq!(argmax_tensor(&ctx, &input, &excluded).unwrap(), 123);
        // 无排除时全局最大值(在第二个 partial 段尾部附近)
        assert_eq!(argmax_tensor(&ctx, &input, &[]).unwrap(), 7);
    }
}
