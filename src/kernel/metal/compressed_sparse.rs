//! DeepSeek-V4 压缩 KV、学习式 indexer 与滑窗注意力算子。

/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: csa_gated_compress_f16, csa_store_overlap_f16, csa_index_scores_f16, csa_index_topk, csa_attention_f16
// private helpers: csa_constant, csa_ordered_score, csa_compressor_input
pub const SHADERS: &str = r#"
inline float csa_constant(device const uchar *values, ulong index, uint f32_values)
{
    return f32_values != 0
        ? reinterpret_cast<device const float *>(values)[index]
        : float(reinterpret_cast<device const half *>(values)[index]);
}
inline uint csa_ordered_score(float value)
{
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}
inline float csa_compressor_input(
    device const half *pending,
    device const half *input,
    uint pending_rows,
    uint channels,
    uint row,
    uint column)
{
    return row < pending_rows
        ? float(pending[ulong(row) * channels + column])
        : float(input[ulong(row - pending_rows) * channels + column]);
}
kernel void csa_gated_compress_f16(
    device const half *pending_key       [[buffer(0)]],
    device const half *pending_gate      [[buffer(1)]],
    device const half *key               [[buffer(2)]],
    device const half *gate              [[buffer(3)]],
    device const uchar *position_bias    [[buffer(4)]],
    device const uchar *norm             [[buffer(5)]],
    device const half *overlap_key       [[buffer(6)]],
    device const half *overlap_gate      [[buffer(7)]],
    device const half *cos_table         [[buffer(8)]],
    device const half *sin_table         [[buffer(9)]],
    device half *output                  [[buffer(10)]],
    constant uint &pending_rows          [[buffer(11)]],
    constant uint &input_rows            [[buffer(12)]],
    constant uint &windows               [[buffer(13)]],
    constant uint &ratio                 [[buffer(14)]],
    constant uint &width                 [[buffer(15)]],
    constant uint &channels              [[buffer(16)]],
    constant uint &overlap               [[buffer(17)]],
    constant uint &entry_start           [[buffer(18)]],
    constant uint &rotary_dim            [[buffer(19)]],
    constant uint &position_bias_f32     [[buffer(20)]],
    constant uint &norm_f32              [[buffer(21)]],
    constant float &eps                  [[buffer(22)]],
    uint window [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint thread_count [[threads_per_threadgroup]])
{
    if (window >= windows) return;
    threadgroup float reduced[1024];
    threadgroup float normalized[1024];
    float pooled = 0.0f;
    if (lane < width) {
        const uint slots = overlap != 0 ? ratio * 2 : ratio;
        float maximum = -INFINITY;
        for (uint slot = 0; slot < slots; ++slot) {
            float logit;
            if (overlap != 0 && slot < ratio) {
                if (window == 0) {
                    logit = float(overlap_gate[ulong(slot) * width + lane]);
                } else {
                    const uint row = (window - 1) * ratio + slot;
                    logit = csa_compressor_input(pending_gate, gate, pending_rows, channels, row, lane)
                        + csa_constant(position_bias, ulong(slot) * channels + lane, position_bias_f32);
                }
            } else {
                const uint current_slot = overlap != 0 ? slot - ratio : slot;
                const uint row = window * ratio + current_slot;
                const uint column = overlap != 0 ? width + lane : lane;
                logit = csa_compressor_input(pending_gate, gate, pending_rows, channels, row, column)
                    + csa_constant(position_bias, ulong(current_slot) * channels + column, position_bias_f32);
            }
            maximum = max(maximum, logit);
        }
        float denominator = 0.0f;
        for (uint slot = 0; slot < slots; ++slot) {
            float value;
            float logit;
            if (overlap != 0 && slot < ratio) {
                if (window == 0) {
                    value = float(overlap_key[ulong(slot) * width + lane]);
                    logit = float(overlap_gate[ulong(slot) * width + lane]);
                } else {
                    const uint row = (window - 1) * ratio + slot;
                    value = csa_compressor_input(pending_key, key, pending_rows, channels, row, lane);
                    logit = csa_compressor_input(pending_gate, gate, pending_rows, channels, row, lane)
                        + csa_constant(position_bias, ulong(slot) * channels + lane, position_bias_f32);
                }
            } else {
                const uint current_slot = overlap != 0 ? slot - ratio : slot;
                const uint row = window * ratio + current_slot;
                const uint column = overlap != 0 ? width + lane : lane;
                value = csa_compressor_input(pending_key, key, pending_rows, channels, row, column);
                logit = csa_compressor_input(pending_gate, gate, pending_rows, channels, row, column)
                    + csa_constant(position_bias, ulong(current_slot) * channels + column, position_bias_f32);
            }
            const float weight = exp(logit - maximum);
            denominator += weight;
            pooled += value * weight;
        }
        pooled /= denominator;
        reduced[lane] = pooled * pooled;
    } else {
        reduced[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = thread_count >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) reduced[lane] += reduced[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < width) {
        const float scale = rsqrt(reduced[0] / float(width) + eps);
        normalized[lane] = pooled * scale * csa_constant(norm, lane, norm_f32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < width) {
        float result = normalized[lane];
        const uint rotary_start = width - rotary_dim;
        if (lane >= rotary_start) {
            const uint relative = lane - rotary_start;
            const uint pair = relative >> 1;
            const uint partner = rotary_start + (relative ^ 1u);
            const ulong table = ulong((entry_start + window) * ratio) * (rotary_dim >> 1) + pair;
            const float cosine = float(cos_table[table]);
            const float sine = float(sin_table[table]);
            result = (relative & 1u) == 0
                ? normalized[lane] * cosine - normalized[partner] * sine
                : normalized[lane] * cosine + normalized[partner] * sine;
        }
        output[ulong(window) * width + lane] = half(result);
    }
    (void)input_rows;
}
kernel void csa_store_overlap_f16(
    device const half *pending_key       [[buffer(0)]],
    device const half *pending_gate      [[buffer(1)]],
    device const half *key               [[buffer(2)]],
    device const half *gate              [[buffer(3)]],
    device const uchar *position_bias    [[buffer(4)]],
    device half *overlap_key             [[buffer(5)]],
    device half *overlap_gate            [[buffer(6)]],
    constant uint &pending_rows          [[buffer(7)]],
    constant uint &input_rows            [[buffer(8)]],
    constant uint &windows               [[buffer(9)]],
    constant uint &ratio                 [[buffer(10)]],
    constant uint &width                 [[buffer(11)]],
    constant uint &channels              [[buffer(12)]],
    constant uint &position_bias_f32     [[buffer(13)]],
    constant uint &count                 [[buffer(14)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const uint slot = index / width;
    const uint column = index - slot * width;
    const uint row = (windows - 1) * ratio + slot;
    overlap_key[index] = half(csa_compressor_input(pending_key, key, pending_rows, channels, row, column));
    overlap_gate[index] = half(csa_compressor_input(pending_gate, gate, pending_rows, channels, row, column)
        + csa_constant(position_bias, ulong(slot) * channels + column, position_bias_f32));
    (void)input_rows;
}
kernel void csa_index_scores_f16(
    device const half *query             [[buffer(0)]],
    device const half *keys              [[buffer(1)]],
    device const half *head_weights      [[buffer(2)]],
    device const uint *visible_counts    [[buffer(3)]],
    device uint *ordered_scores          [[buffer(4)]],
    constant uint &query_rows            [[buffer(5)]],
    constant uint &compressed_rows       [[buffer(6)]],
    constant uint &head_count            [[buffer(7)]],
    constant uint &head_dim              [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]])
{
    const uint key_row = group.x;
    const uint query_row = group.y;
    if (query_row >= query_rows || key_row >= visible_counts[query_row]) return;
    threadgroup float partial[8];
    float subtotal = 0.0f;
    for (uint head = simd_index; head < head_count; head += 8) {
        float dot = 0.0f;
        const ulong query_base = ulong(query_row) * head_count * head_dim + ulong(head) * head_dim;
        const ulong key_base = ulong(key_row) * head_dim;
        for (uint column = simd_lane; column < head_dim; column += 32) {
            dot += float(query[query_base + column]) * float(keys[key_base + column]);
        }
        dot = simd_sum(dot);
        if (simd_lane == 0) subtotal += float(head_weights[ulong(query_row) * head_count + head]) * max(0.0f, dot * rsqrt(float(head_dim)));
    }
    if (simd_lane == 0) partial[simd_index] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0) {
        float total = simd_lane < 8 ? partial[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) ordered_scores[ulong(query_row) * compressed_rows + key_row] = csa_ordered_score(total * rsqrt(float(head_count)));
    }
}
kernel void csa_index_topk(
    device const uint *ordered_scores    [[buffer(0)]],
    device const uint *visible_counts    [[buffer(1)]],
    device uint *selection               [[buffer(2)]],
    constant uint &compressed_rows       [[buffer(3)]],
    constant uint &top_k                 [[buffer(4)]],
    uint query_row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint visible = visible_counts[query_row];
    if (visible == 0) return;
    threadgroup atomic_uint histogram[256];
    threadgroup uint prefix;
    threadgroup uint mask;
    threadgroup uint remaining;
    if (lane == 0) {
        prefix = 0;
        mask = 0;
        remaining = min(top_k, visible);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong score_base = ulong(query_row) * compressed_rows;
    for (uint pass = 0; pass < 4; ++pass) {
        atomic_store_explicit(&histogram[lane], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint shift = 24u - pass * 8u;
        for (uint row = lane; row < visible; row += 256) {
            const uint score = ordered_scores[score_base + row];
            if ((score & mask) == prefix) atomic_fetch_add_explicit(&histogram[(score >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            for (int bucket = 255; bucket >= 0; --bucket) {
                const uint count = atomic_load_explicit(&histogram[uint(bucket)], memory_order_relaxed);
                if (remaining > count) {
                    remaining -= count;
                } else {
                    prefix |= uint(bucket) << shift;
                    mask |= 255u << shift;
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const uint keep = min(top_k, visible);
        uint written = 0;
        const ulong output_base = ulong(query_row) * top_k;
        for (uint row = 0; row < visible && written < keep; ++row) {
            if (ordered_scores[score_base + row] >= prefix) selection[output_base + written++] = row;
        }
    }
}
kernel void csa_attention_f16(
    device const half *query             [[buffer(0)]],
    device const half *compressed_key    [[buffer(1)]],
    device const half *compressed_value  [[buffer(2)]],
    device const uint *visible_compressed [[buffer(3)]],
    device const uint *selection         [[buffer(4)]],
    device const half *recent_key        [[buffer(5)]],
    device const half *recent_value      [[buffer(6)]],
    device const half *batch_key         [[buffer(7)]],
    device const half *batch_value       [[buffer(8)]],
    device const uchar *sink             [[buffer(9)]],
    device half *output                  [[buffer(10)]],
    constant uint &query_rows            [[buffer(11)]],
    constant uint &selection_top_k       [[buffer(12)]],
    constant uint &recent_start          [[buffer(13)]],
    constant uint &recent_len            [[buffer(14)]],
    constant uint &recent_first_position [[buffer(15)]],
    constant uint &position_start        [[buffer(16)]],
    constant uint &batch_rows            [[buffer(17)]],
    constant uint &head_count            [[buffer(18)]],
    constant uint &kv_head_count         [[buffer(19)]],
    constant uint &head_dim              [[buffer(20)]],
    constant uint &window_size           [[buffer(21)]],
    constant uint &sink_dtype            [[buffer(22)]],
    constant uint &causal_batch          [[buffer(23)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]])
{
    const uint head = group.x;
    const uint query_row = group.y;
    if (head >= head_count || query_row >= query_rows) return;
    const uint kv_head = head * kv_head_count / head_count;
    const uint query_position = position_start + query_row;
    const ulong query_base = ulong(query_row) * head_count * head_dim + ulong(head) * head_dim;
    const float query_value = lane < head_dim ? float(query[query_base + lane]) : 0.0f;
    float value_sum = 0.0f;
    float maximum = sink_dtype == 2 ? -INFINITY : csa_constant(sink, head, sink_dtype);
    float denominator = sink_dtype == 2 ? 0.0f : 1.0f;
    threadgroup float score_parts[1024];
    threadgroup float softmax_state[3];
    const float score_scale = rsqrt(float(head_dim));

    const uint visible = visible_compressed[query_row];
    const uint compressed_count = selection_top_k == 0 ? visible : min(selection_top_k, visible);
    for (uint slot = 0; slot < compressed_count; ++slot) {
        const uint row = selection_top_k == 0 ? slot : selection[ulong(query_row) * selection_top_k + slot];
        const ulong kv_base = ulong(row) * kv_head_count * head_dim + ulong(kv_head) * head_dim;
        score_parts[lane] = lane < head_dim ? query_value * float(compressed_key[kv_base + lane]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threadgroup_size.x >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) score_parts[lane] += score_parts[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float score = score_parts[0] * score_scale;
            const float next_maximum = max(maximum, score);
            softmax_state[0] = exp(maximum - next_maximum);
            softmax_state[1] = exp(score - next_maximum);
            denominator = denominator * softmax_state[0] + softmax_state[1];
            maximum = next_maximum;
            softmax_state[2] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < head_dim) value_sum = value_sum * softmax_state[0] + float(compressed_value[kv_base + lane]) * softmax_state[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint logical = 0; logical < recent_len; ++logical) {
        const uint key_position = recent_first_position + logical;
        if (key_position > query_position || query_position - key_position >= window_size) continue;
        const uint physical = (recent_start + logical) % window_size;
        const ulong kv_base = ulong(physical) * kv_head_count * head_dim + ulong(kv_head) * head_dim;
        score_parts[lane] = lane < head_dim ? query_value * float(recent_key[kv_base + lane]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threadgroup_size.x >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) score_parts[lane] += score_parts[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float score = score_parts[0] * score_scale;
            const float next_maximum = max(maximum, score);
            softmax_state[0] = exp(maximum - next_maximum);
            softmax_state[1] = exp(score - next_maximum);
            denominator = denominator * softmax_state[0] + softmax_state[1];
            maximum = next_maximum;
            softmax_state[2] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < head_dim) value_sum = value_sum * softmax_state[0] + float(recent_value[kv_base + lane]) * softmax_state[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint batch_begin = query_row + 1 > window_size ? query_row + 1 - window_size : 0;
    const uint batch_end = causal_batch != 0 ? min(query_row + 1, batch_rows) : batch_rows;
    for (uint row = batch_begin; row < batch_end; ++row) {
        const ulong kv_base = ulong(row) * kv_head_count * head_dim + ulong(kv_head) * head_dim;
        score_parts[lane] = lane < head_dim ? query_value * float(batch_key[kv_base + lane]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threadgroup_size.x >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) score_parts[lane] += score_parts[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float score = score_parts[0] * score_scale;
            const float next_maximum = max(maximum, score);
            softmax_state[0] = exp(maximum - next_maximum);
            softmax_state[1] = exp(score - next_maximum);
            denominator = denominator * softmax_state[0] + softmax_state[1];
            maximum = next_maximum;
            softmax_state[2] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < head_dim) value_sum = value_sum * softmax_state[0] + float(batch_value[kv_base + lane]) * softmax_state[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < head_dim) output[query_base + lane] = half(softmax_state[2] == 0.0f ? 0.0f : value_sum / softmax_state[2]);
}
"#;

use std::mem;

use crate::backend::metal::{MetalContext, MetalTensor};

use super::{MTLSize, THREADS, metal, set_bytes, validate_u32};

const INDEX_SCRATCH_BYTES: usize = 64 * 1024 * 1024;

#[allow(clippy::too_many_arguments)]
pub fn gated_compress(
    ctx: &MetalContext,
    pending_key: &metal::Buffer,
    pending_gate: &metal::Buffer,
    pending_rows: usize,
    key: &MetalTensor,
    gate: &MetalTensor,
    position_bias: &metal::Buffer,
    position_bias_f32: bool,
    norm: &metal::Buffer,
    norm_f32: bool,
    overlap_key: &metal::Buffer,
    overlap_gate: &metal::Buffer,
    ratio: usize,
    width: usize,
    overlap: bool,
    entry_start: usize,
    rotary_dim: usize,
    cos: &metal::Buffer,
    sin: &metal::Buffer,
    eps: f32,
) -> Result<(MetalTensor, usize), String> {
    let channels = if overlap { width.checked_mul(2).ok_or("V4 compressor channels 溢出")? } else { width };
    if ratio == 0 || width == 0 || pending_rows >= ratio || key.cols != channels || gate.rows != key.rows || gate.cols != channels {
        return Err(format!("V4 Metal compressor 维度非法: pending={pending_rows} key=[{},{}] gate=[{},{}] ratio={ratio} width={width} overlap={overlap}", key.rows, key.cols, gate.rows, gate.cols));
    }
    let total_rows = pending_rows.checked_add(key.rows).ok_or("V4 compressor rows 溢出")?;
    // kernel/blit 向 pending 最多写 ratio 行(windows==0 时累计 total_rows < ratio;否则 remaining < ratio),容量不足会越界写。
    let pending_capacity = ratio.checked_mul(channels).and_then(|n| n.checked_mul(mem::size_of::<half::f16>())).ok_or("V4 compressor pending 容量溢出")? as u64;
    if pending_key.length() < pending_capacity || pending_gate.length() < pending_capacity {
        return Err(format!("V4 compressor pending buffer 容量不足: key={}B gate={}B，需 ≥ {pending_capacity}B(ratio={ratio} 行 × channels={channels})", pending_key.length(), pending_gate.length()));
    }
    let windows = total_rows / ratio;
    let remaining = total_rows % ratio;
    let output = if windows == 0 { MetalTensor::new(ctx.shared_buffer_zeros(mem::size_of::<half::f16>()), 0, width) } else { ctx.tensor_kernel_output(windows, width) };
    let command = ctx.command_buffer();

    if windows != 0 {
        let ratio_u32 = validate_u32("V4 compressor ratio", ratio)?;
        let width_u32 = validate_u32("V4 compressor width", width)?;
        let channels_u32 = validate_u32("V4 compressor channels", channels)?;
        let pending_rows_u32 = validate_u32("V4 compressor pending rows", pending_rows)?;
        let input_rows_u32 = validate_u32("V4 compressor input rows", key.rows)?;
        let windows_u32 = validate_u32("V4 compressor windows", windows)?;
        let entry_start_u32 = validate_u32("V4 compressor entry", entry_start)?;
        let rotary_dim_u32 = validate_u32("V4 compressor rotary dim", rotary_dim)?;
        let position_bias_dtype = u32::from(position_bias_f32);
        let norm_dtype = u32::from(norm_f32);
        let overlap_u32 = u32::from(overlap);
        let pipeline = ctx.pipeline("csa_gated_compress_f16")?;
        let threads = width.next_power_of_two();
        if threads > pipeline.max_total_threads_per_threadgroup() as usize || threads > 1024 {
            return Err(format!("V4 compressor width={width} 需要 {threads} threads，Metal 上限={}", pipeline.max_total_threads_per_threadgroup()));
        }
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(pending_key), 0);
        encoder.set_buffer(1, Some(pending_gate), 0);
        encoder.set_buffer(2, Some(&key.buffer), 0);
        encoder.set_buffer(3, Some(&gate.buffer), 0);
        encoder.set_buffer(4, Some(position_bias), 0);
        encoder.set_buffer(5, Some(norm), 0);
        encoder.set_buffer(6, Some(overlap_key), 0);
        encoder.set_buffer(7, Some(overlap_gate), 0);
        encoder.set_buffer(8, Some(cos), 0);
        encoder.set_buffer(9, Some(sin), 0);
        encoder.set_buffer(10, Some(&output.buffer), 0);
        set_bytes(&encoder, 11, &pending_rows_u32);
        set_bytes(&encoder, 12, &input_rows_u32);
        set_bytes(&encoder, 13, &windows_u32);
        set_bytes(&encoder, 14, &ratio_u32);
        set_bytes(&encoder, 15, &width_u32);
        set_bytes(&encoder, 16, &channels_u32);
        set_bytes(&encoder, 17, &overlap_u32);
        set_bytes(&encoder, 18, &entry_start_u32);
        set_bytes(&encoder, 19, &rotary_dim_u32);
        set_bytes(&encoder, 20, &position_bias_dtype);
        set_bytes(&encoder, 21, &norm_dtype);
        set_bytes(&encoder, 22, &eps);
        encoder.dispatch_thread_groups(MTLSize::new(windows as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
        encoder.end_encoding();

        if overlap {
            let count = ratio.checked_mul(width).ok_or("V4 compressor overlap state 溢出")?;
            let count_u32 = validate_u32("V4 compressor overlap state", count)?;
            let pipeline = ctx.pipeline("csa_store_overlap_f16")?;
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(pending_key), 0);
            encoder.set_buffer(1, Some(pending_gate), 0);
            encoder.set_buffer(2, Some(&key.buffer), 0);
            encoder.set_buffer(3, Some(&gate.buffer), 0);
            encoder.set_buffer(4, Some(position_bias), 0);
            encoder.set_buffer(5, Some(overlap_key), 0);
            encoder.set_buffer(6, Some(overlap_gate), 0);
            set_bytes(&encoder, 7, &pending_rows_u32);
            set_bytes(&encoder, 8, &input_rows_u32);
            set_bytes(&encoder, 9, &windows_u32);
            set_bytes(&encoder, 10, &ratio_u32);
            set_bytes(&encoder, 11, &width_u32);
            set_bytes(&encoder, 12, &channels_u32);
            set_bytes(&encoder, 13, &position_bias_dtype);
            set_bytes(&encoder, 14, &count_u32);
            encoder.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(count.min(THREADS) as u64, 1, 1));
            encoder.end_encoding();
        }
    }

    if key.rows != 0 {
        let copy_rows = if windows == 0 { key.rows } else { remaining };
        if copy_rows != 0 {
            let source_row = key.rows - copy_rows;
            let row_bytes = channels.checked_mul(mem::size_of::<half::f16>()).ok_or("V4 compressor row bytes 溢出")? as u64;
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&key.buffer, source_row as u64 * row_bytes, pending_key, if windows == 0 { pending_rows as u64 * row_bytes } else { 0 }, copy_rows as u64 * row_bytes);
            blit.copy_from_buffer(&gate.buffer, source_row as u64 * row_bytes, pending_gate, if windows == 0 { pending_rows as u64 * row_bytes } else { 0 }, copy_rows as u64 * row_bytes);
            blit.end_encoding();
        }
    }

    let shape = format!("rows={}+{} ratio={ratio} width={width} overlap={overlap}", pending_rows, key.rows);
    let read_bytes = key.buffer.length() + gate.buffer.length() + position_bias.length() + norm.length();
    ctx.commit_and_wait_profiled(&command, "csa_gated_compress_f16", &shape, read_bytes, output.buffer.length());
    Ok((output, remaining))
}

#[allow(clippy::too_many_arguments)]
pub fn select_indexed_history(
    ctx: &MetalContext,
    query: &MetalTensor,
    keys: &metal::Buffer,
    head_weights: &MetalTensor,
    visible_counts: &metal::Buffer,
    compressed_rows: usize,
    head_count: usize,
    head_dim: usize,
    top_k: usize,
) -> Result<metal::Buffer, String> {
    if query.rows == 0 || compressed_rows == 0 || query.cols != head_count * head_dim || head_weights.rows != query.rows || head_weights.cols != head_count || top_k == 0 {
        return Err(format!("V4 Metal indexer 维度非法: query=[{},{}] weights=[{},{}] compressed={compressed_rows} heads={head_count} dim={head_dim} top_k={top_k}", query.rows, query.cols, head_weights.rows, head_weights.cols));
    }
    let compressed_u32 = validate_u32("V4 indexer compressed rows", compressed_rows)?;
    let heads_u32 = validate_u32("V4 indexer heads", head_count)?;
    let dim_u32 = validate_u32("V4 indexer head dim", head_dim)?;
    let topk_u32 = validate_u32("V4 indexer top_k", top_k)?;
    let selection_bytes = query.rows.checked_mul(top_k).and_then(|count| count.checked_mul(mem::size_of::<u32>())).ok_or("V4 indexer selection 大小溢出")?;
    let selection = ctx.shared_buffer_zeros(selection_bytes);
    let score_row_bytes = compressed_rows.checked_mul(mem::size_of::<u32>()).ok_or("V4 indexer score row 溢出")?;
    let chunk_rows = (INDEX_SCRATCH_BYTES / score_row_bytes.max(1)).max(1).min(query.rows);
    let scores = ctx.shared_buffer_uninit(chunk_rows.checked_mul(score_row_bytes).ok_or("V4 indexer score scratch 溢出")?);
    let score_pipeline = ctx.pipeline("csa_index_scores_f16")?;
    let topk_pipeline = ctx.pipeline("csa_index_topk")?;
    if score_pipeline.max_total_threads_per_threadgroup() < THREADS as u64 || topk_pipeline.max_total_threads_per_threadgroup() < THREADS as u64 {
        return Err("V4 indexer 需要 256 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    for query_start in (0..query.rows).step_by(chunk_rows) {
        let rows = (query.rows - query_start).min(chunk_rows);
        let rows_u32 = validate_u32("V4 indexer query rows", rows)?;
        let query_offset = query_start.checked_mul(query.cols).and_then(|n| n.checked_mul(mem::size_of::<half::f16>())).ok_or("V4 indexer query offset 溢出")? as u64;
        let weight_offset = query_start.checked_mul(head_count).and_then(|n| n.checked_mul(mem::size_of::<half::f16>())).ok_or("V4 indexer weight offset 溢出")? as u64;
        let count_offset = query_start.checked_mul(mem::size_of::<u32>()).ok_or("V4 indexer count offset 溢出")? as u64;
        let selection_offset = query_start.checked_mul(top_k).and_then(|n| n.checked_mul(mem::size_of::<u32>())).ok_or("V4 indexer selection offset 溢出")? as u64;

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&score_pipeline);
        encoder.set_buffer(0, Some(&query.buffer), query_offset);
        encoder.set_buffer(1, Some(keys), 0);
        encoder.set_buffer(2, Some(&head_weights.buffer), weight_offset);
        encoder.set_buffer(3, Some(visible_counts), count_offset);
        encoder.set_buffer(4, Some(&scores), 0);
        set_bytes(&encoder, 5, &rows_u32);
        set_bytes(&encoder, 6, &compressed_u32);
        set_bytes(&encoder, 7, &heads_u32);
        set_bytes(&encoder, 8, &dim_u32);
        encoder.dispatch_thread_groups(MTLSize::new(compressed_rows as u64, rows as u64, 1), MTLSize::new(THREADS as u64, 1, 1));
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&topk_pipeline);
        encoder.set_buffer(0, Some(&scores), 0);
        encoder.set_buffer(1, Some(visible_counts), count_offset);
        encoder.set_buffer(2, Some(&selection), selection_offset);
        set_bytes(&encoder, 3, &compressed_u32);
        set_bytes(&encoder, 4, &topk_u32);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
        encoder.end_encoding();
    }
    let shape = format!("query={} compressed={compressed_rows} heads={head_count} dim={head_dim} top_k={top_k}", query.rows);
    ctx.commit_and_wait_profiled(&command, "csa_index_select", &shape, query.buffer.length() + keys.length() + head_weights.buffer.length(), selection.length());
    Ok(selection)
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &MetalContext,
    query: &MetalTensor,
    compressed_key: &metal::Buffer,
    compressed_value: &metal::Buffer,
    visible_compressed: &metal::Buffer,
    selection: Option<(&metal::Buffer, usize)>,
    recent_key: &metal::Buffer,
    recent_value: &metal::Buffer,
    recent_start: usize,
    recent_len: usize,
    recent_first_position: usize,
    key: &MetalTensor,
    value: &MetalTensor,
    position_start: usize,
    causal_batch: bool,
    sink: &metal::Buffer,
    sink_dtype: u32,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
) -> Result<MetalTensor, String> {
    if query.rows == 0 || query.cols != num_heads * head_dim || key.rows != query.rows || value.rows != query.rows || key.cols != num_kv_heads * head_dim || value.cols != key.cols {
        return Err(format!("V4 Metal CSA 维度非法: query=[{},{}] key=[{},{}] value=[{},{}] heads={num_heads}/{num_kv_heads} dim={head_dim}", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols));
    }
    let output = ctx.tensor_kernel_output(query.rows, query.cols);
    let dummy_selection = ctx.shared_buffer_zeros(mem::size_of::<u32>());
    let (selection_buffer, selection_top_k) = selection.unwrap_or((&dummy_selection, 0));
    let query_rows = validate_u32("V4 CSA query rows", query.rows)?;
    let recent_start = validate_u32("V4 CSA recent start", recent_start)?;
    let recent_len = validate_u32("V4 CSA recent len", recent_len)?;
    let recent_first_position = validate_u32("V4 CSA recent position", recent_first_position)?;
    let position_start = validate_u32("V4 CSA position", position_start)?;
    let batch_rows = validate_u32("V4 CSA batch rows", key.rows)?;
    let heads = validate_u32("V4 CSA heads", num_heads)?;
    let kv_heads = validate_u32("V4 CSA KV heads", num_kv_heads)?;
    let dim = validate_u32("V4 CSA head dim", head_dim)?;
    let window = validate_u32("V4 CSA window", window_size)?;
    let selection_top_k = validate_u32("V4 CSA selection top_k", selection_top_k)?;
    let causal_batch = u32::from(causal_batch);
    let pipeline = ctx.pipeline("csa_attention_f16")?;
    let threads = head_dim.next_power_of_two();
    if threads > pipeline.max_total_threads_per_threadgroup() as usize || threads > 1024 {
        return Err(format!("V4 CSA head_dim={head_dim} 需要 {threads} threads，Metal 上限={}", pipeline.max_total_threads_per_threadgroup()));
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(compressed_key), 0);
    encoder.set_buffer(2, Some(compressed_value), 0);
    encoder.set_buffer(3, Some(visible_compressed), 0);
    encoder.set_buffer(4, Some(selection_buffer), 0);
    encoder.set_buffer(5, Some(recent_key), 0);
    encoder.set_buffer(6, Some(recent_value), 0);
    encoder.set_buffer(7, Some(&key.buffer), 0);
    encoder.set_buffer(8, Some(&value.buffer), 0);
    encoder.set_buffer(9, Some(sink), 0);
    encoder.set_buffer(10, Some(&output.buffer), 0);
    set_bytes(&encoder, 11, &query_rows);
    set_bytes(&encoder, 12, &selection_top_k);
    set_bytes(&encoder, 13, &recent_start);
    set_bytes(&encoder, 14, &recent_len);
    set_bytes(&encoder, 15, &recent_first_position);
    set_bytes(&encoder, 16, &position_start);
    set_bytes(&encoder, 17, &batch_rows);
    set_bytes(&encoder, 18, &heads);
    set_bytes(&encoder, 19, &kv_heads);
    set_bytes(&encoder, 20, &dim);
    set_bytes(&encoder, 21, &window);
    set_bytes(&encoder, 22, &sink_dtype);
    set_bytes(&encoder, 23, &causal_batch);
    encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, query.rows as u64, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={} heads={num_heads}/{num_kv_heads} dim={head_dim} recent={recent_len} selected={selection_top_k}", query.rows);
    let read_bytes = query.buffer.length() + compressed_key.length() + compressed_value.length() + recent_key.length() + recent_value.length() + key.buffer.length() + value.buffer.length();
    ctx.commit_and_wait_profiled(&command, "csa_attention_f16", &shape, read_bytes, output.buffer.length());
    Ok(output)
}
