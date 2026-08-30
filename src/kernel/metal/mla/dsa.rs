//! DSA(DeepSeek Sparse Attention)的 key 存储与 top-k 选择 kernel。

/// 本家族的 Metal shader(radix top-k 专用 helper 先于 kernel 定义)。
pub const SHADERS: &str = r#"
inline uint dsa_prefill_ordered_score(float value)
{
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}
inline uint dsa_ordered_score(float value)
{
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}
kernel void dsa_store_key_f16(
    device const half *key               [[buffer(0)]],
    device half *key_cache               [[buffer(1)]],
    constant uint &position              [[buffer(2)]],
    constant uint &rows                  [[buffer(3)]],
    constant uint &head_dim              [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < rows * head_dim) {
        key_cache[(ulong)position * head_dim + index] = key[index];
    }
}
kernel void dsa_prefill_scores_f16(
    device const half *query             [[buffer(0)]],
    device const half *key               [[buffer(1)]],
    device const half *head_weights      [[buffer(2)]],
    device float *scores                 [[buffer(3)]],
    constant uint &rows                  [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    const uint head_count = zllm_fc_u32_0;
    const uint head_dim = zllm_fc_u32_1;
    const ulong pairs = ulong(rows) * rows;
    if (ulong(index) >= pairs) return;
    const uint query_row = index / rows;
    const uint key_row = index - query_row * rows;
    if (key_row > query_row) return;
    float score = 0.0f;
    const ulong query_base = ulong(query_row) * head_count * head_dim;
    const ulong key_base = ulong(key_row) * head_dim;
    const ulong weight_base = ulong(query_row) * head_count;
    const float head_scale = rsqrt(float(head_dim));
    for (uint head = 0; head < head_count; ++head) {
        float product = 0.0f;
        const ulong head_base = query_base + ulong(head) * head_dim;
        for (uint dimension = 0; dimension < head_dim; ++dimension) {
            product += float(query[head_base + dimension]) * float(key[key_base + dimension]);
        }
        score += float(head_weights[weight_base + head]) * max(product * head_scale, 0.0f);
    }
    scores[ulong(query_row) * rows + key_row] = score * rsqrt(float(head_count));
}
kernel void dsa_prefill_radix_init(
    device uint *prefix                 [[buffer(0)]],
    device uint *rank                   [[buffer(1)]],
    constant uint &rows                 [[buffer(2)]],
    constant uint &top_k                [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row < rows) {
        prefix[row] = 0;
        rank[row] = min(top_k, row + 1);
    }
}
kernel void dsa_prefill_radix_pass(
    device const float *scores          [[buffer(0)]],
    device uint *prefix                 [[buffer(1)]],
    device uint *rank                   [[buffer(2)]],
    constant uint &rows                 [[buffer(3)]],
    constant uint &shift                [[buffer(4)]],
    constant uint &mask                 [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= rows) return;
    threadgroup atomic_uint histogram[256];
    atomic_store_explicit(&histogram[lane], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint wanted = prefix[row];
    const ulong base = ulong(row) * rows;
    for (uint key_row = lane; key_row <= row; key_row += 256) {
        const uint ordered = dsa_prefill_ordered_score(scores[base + key_row]);
        if ((ordered & mask) == wanted) {
            atomic_fetch_add_explicit(&histogram[(ordered >> shift) & 255u], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        uint remaining = rank[row];
        for (int bucket = 255; bucket >= 0; --bucket) {
            const uint count = atomic_load_explicit(&histogram[uint(bucket)], memory_order_relaxed);
            if (remaining > count) {
                remaining -= count;
            } else {
                prefix[row] = wanted | (uint(bucket) << shift);
                rank[row] = remaining;
                break;
            }
        }
    }
}
kernel void dsa_prefill_gather_topk(
    device const float *scores          [[buffer(0)]],
    device const uint *thresholds       [[buffer(1)]],
    device const uint *threshold_ranks  [[buffer(2)]],
    device uint *indices                [[buffer(3)]],
    constant uint &rows                 [[buffer(4)]],
    constant uint &top_k                [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const uint keep = min(top_k, row + 1);
    const uint threshold = thresholds[row];
    const uint equal_limit = threshold_ranks[row];
    uint equal_seen = 0;
    uint written = 0;
    const ulong score_base = ulong(row) * rows;
    const ulong output_base = ulong(row) * top_k;
    for (uint key_row = 0; key_row <= row && written < keep; ++key_row) {
        const uint ordered = dsa_prefill_ordered_score(scores[score_base + key_row]);
        bool selected = ordered > threshold;
        if (ordered == threshold) {
            selected = equal_seen < equal_limit;
            ++equal_seen;
        }
        if (selected) indices[output_base + written++] = key_row;
    }
}
kernel void dsa_decode_scores_f16(
    device const half *query             [[buffer(0)]],
    device const half *key_cache         [[buffer(1)]],
    device const half *head_weights      [[buffer(2)]],
    device uint *ordered_scores          [[buffer(3)]],
    constant uint &rows                  [[buffer(4)]],
    constant uint &head_count            [[buffer(5)]],
    constant uint &head_dim              [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]])
{
    if (token >= rows) return;
    threadgroup float partial[8];
    const float head_scale = rsqrt(float(head_dim));
    float subtotal = 0.0f;
    for (uint head = simd_index; head < head_count; head += 8) {
        float dot = 0.0f;
        for (uint column = simd_lane; column < head_dim; column += 32) {
            dot += float(query[(ulong)head * head_dim + column])
                * float(key_cache[(ulong)token * head_dim + column]);
        }
        dot = simd_sum(dot);
        if (simd_lane == 0) {
            subtotal += float(head_weights[head]) * max(0.0f, dot * head_scale);
        }
    }
    if (simd_lane == 0) partial[simd_index] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0) {
        float total = simd_lane < 8 ? partial[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) {
            ordered_scores[token] = dsa_ordered_score(total * rsqrt(float(head_count)));
        }
    }
}
kernel void dsa_decode_radix_topk(
    device const uint *ordered_scores    [[buffer(0)]],
    device uint *selection               [[buffer(1)]],
    device uint *radix_state             [[buffer(2)]],
    constant uint &rows                  [[buffer(3)]],
    constant uint &top_k                 [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint histogram[256];
    threadgroup uint prefix;
    threadgroup uint mask;
    threadgroup uint remaining;
    if (lane == 0) {
        prefix = 0;
        mask = 0;
        remaining = top_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pass = 0; pass < 4; ++pass) {
        atomic_store_explicit(&histogram[lane], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint shift = 24u - pass * 8u;
        for (uint token = lane; token < rows; token += 256) {
            const uint key = ordered_scores[token];
            if ((key & mask) == prefix) {
                atomic_fetch_add_explicit(&histogram[(key >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            for (int bin = 255; bin >= 0; --bin) {
                const uint count = atomic_load_explicit(&histogram[uint(bin)], memory_order_relaxed);
                if (remaining > count) {
                    remaining -= count;
                } else {
                    prefix |= uint(bin) << shift;
                    mask |= 255u << shift;
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        uint count = 0;
        for (uint token = 0; token < rows && count < top_k; ++token) {
            if (ordered_scores[token] >= prefix) {
                selection[count++] = token;
            }
        }
        radix_state[0] = prefix;
        radix_state[1] = count;
    }
}
"#;

use crate::backend::metal::api as metal;

use super::super::{MTLSize, MetalContext, MetalTensor, THREADS, f16, mem, set_bytes, validate_u32};

pub fn dsa_store_keys(ctx: &MetalContext, key: &MetalTensor, key_cache: &metal::Buffer, position: usize, capacity: usize) -> Result<(), String> {
    if key.rows == 0 || key.cols == 0 || position.checked_add(key.rows).is_none_or(|end| end > capacity) {
        return Err(format!("DSA key append 越界: key=[{},{}] position={position} capacity={capacity}", key.rows, key.cols));
    }
    let position_u32 = validate_u32("DSA key position", position)?;
    let rows_u32 = validate_u32("DSA key rows", key.rows)?;
    let head_dim_u32 = validate_u32("DSA head_dim", key.cols)?;
    let pipeline = ctx.pipeline("dsa_store_key_f16")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&key.buffer), 0);
    encoder.set_buffer(1, Some(key_cache), 0);
    set_bytes(&encoder, 2, &position_u32);
    set_bytes(&encoder, 3, &rows_u32);
    set_bytes(&encoder, 4, &head_dim_u32);
    encoder.dispatch_threads(MTLSize::new(key.len() as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("position={position},rows={},head_dim={}", key.rows, key.cols);
    ctx.commit_and_wait_profiled(&command, "dsa_store_key_f16", &shape, key.buffer.length(), (key.len() * mem::size_of::<f16>()) as u64);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dsa_select_prefill(ctx: &MetalContext, query: &MetalTensor, key_cache: &metal::Buffer, head_weights: &MetalTensor, selection: &metal::Buffer, rows: usize, head_count: usize, head_dim: usize, top_k: usize) -> Result<(), String> {
    if query.rows != rows || query.cols != head_count * head_dim || head_weights.rows != rows || head_weights.cols != head_count {
        return Err(format!("DSA prefill select 形状不符: query=[{},{}] weights=[{},{}] rows={rows} heads={head_count} dim={head_dim}", query.rows, query.cols, head_weights.rows, head_weights.cols));
    }
    if top_k == 0 || top_k > rows {
        return Err(format!("DSA prefill top_k={top_k} 与 rows={rows} 不符"));
    }
    let rows_u32 = validate_u32("DSA prefill rows", rows)?;
    let heads_u32 = validate_u32("DSA prefill heads", head_count)?;
    let dim_u32 = validate_u32("DSA prefill head_dim", head_dim)?;
    let topk_u32 = validate_u32("DSA prefill top_k", top_k)?;
    let score_values = rows.checked_mul(rows).ok_or("DSA prefill score shape 溢出")?;
    let scores = ctx.shared_buffer_zeros(score_values.checked_mul(mem::size_of::<f32>()).ok_or("DSA prefill score bytes 溢出")?);
    let state_bytes = rows.checked_mul(mem::size_of::<u32>()).ok_or("DSA prefill radix state 溢出")?;
    let prefix = ctx.shared_buffer_zeros(state_bytes);
    let rank = ctx.shared_buffer_zeros(state_bytes);
    let score_pipeline = ctx.pipeline_u32_constants("dsa_prefill_scores_f16", &[heads_u32, dim_u32])?;
    let init_pipeline = ctx.pipeline("dsa_prefill_radix_init")?;
    let pass_pipeline = ctx.pipeline("dsa_prefill_radix_pass")?;
    let gather_pipeline = ctx.pipeline("dsa_prefill_gather_topk")?;
    if pass_pipeline.max_total_threads_per_threadgroup() < 256 {
        return Err("DSA prefill radix 需要至少 256 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&score_pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(key_cache), 0);
    encoder.set_buffer(2, Some(&head_weights.buffer), 0);
    encoder.set_buffer(3, Some(&scores), 0);
    set_bytes(&encoder, 4, &rows_u32);
    let score_width = score_pipeline.max_total_threads_per_threadgroup().min(256);
    encoder.dispatch_threads(MTLSize::new(score_values as u64, 1, 1), MTLSize::new(score_width, 1, 1));
    encoder.end_encoding();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&init_pipeline);
    encoder.set_buffer(0, Some(&prefix), 0);
    encoder.set_buffer(1, Some(&rank), 0);
    set_bytes(&encoder, 2, &rows_u32);
    set_bytes(&encoder, 3, &topk_u32);
    let init_width = init_pipeline.max_total_threads_per_threadgroup().min(256);
    encoder.dispatch_threads(MTLSize::new(rows as u64, 1, 1), MTLSize::new(init_width, 1, 1));
    encoder.end_encoding();

    for (shift, mask) in [(24_u32, 0_u32), (16, 0xff00_0000), (8, 0xffff_0000), (0, 0xffff_ff00)] {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pass_pipeline);
        encoder.set_buffer(0, Some(&scores), 0);
        encoder.set_buffer(1, Some(&prefix), 0);
        encoder.set_buffer(2, Some(&rank), 0);
        set_bytes(&encoder, 3, &rows_u32);
        set_bytes(&encoder, 4, &shift);
        set_bytes(&encoder, 5, &mask);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&gather_pipeline);
    encoder.set_buffer(0, Some(&scores), 0);
    encoder.set_buffer(1, Some(&prefix), 0);
    encoder.set_buffer(2, Some(&rank), 0);
    encoder.set_buffer(3, Some(selection), 0);
    set_bytes(&encoder, 4, &rows_u32);
    set_bytes(&encoder, 5, &topk_u32);
    let gather_width = gather_pipeline.max_total_threads_per_threadgroup().min(256);
    encoder.dispatch_threads(MTLSize::new(rows as u64, 1, 1), MTLSize::new(gather_width, 1, 1));
    encoder.end_encoding();

    let shape = format!("rows={rows},heads={head_count},dim={head_dim},top_k={top_k}");
    let read_bytes = query.buffer.length() + key_cache.length() + head_weights.buffer.length();
    ctx.commit_and_wait_profiled(&command, "dsa_select_prefill", &shape, read_bytes, selection.length());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dsa_select_topk(
    ctx: &MetalContext,
    query: &MetalTensor,
    key_cache: &metal::Buffer,
    head_weights: &MetalTensor,
    ordered_scores: &metal::Buffer,
    selection: &metal::Buffer,
    radix_state: &metal::Buffer,
    rows: usize,
    head_count: usize,
    head_dim: usize,
    top_k: usize,
) -> Result<(), String> {
    if query.rows != 1 || query.cols != head_count * head_dim || head_weights.rows != 1 || head_weights.cols != head_count {
        return Err(format!("DSA select 形状不符: query=[{},{}] weights=[{},{}] heads={head_count} dim={head_dim}", query.rows, query.cols, head_weights.rows, head_weights.cols));
    }
    if top_k == 0 || top_k > rows {
        return Err(format!("DSA top_k={top_k} 与 rows={rows} 不符"));
    }
    let rows_u32 = validate_u32("DSA rows", rows)?;
    let heads_u32 = validate_u32("DSA heads", head_count)?;
    let dim_u32 = validate_u32("DSA head_dim", head_dim)?;
    let topk_u32 = validate_u32("DSA top_k", top_k)?;
    let score_pipeline = ctx.pipeline("dsa_decode_scores_f16")?;
    let radix_pipeline = ctx.pipeline("dsa_decode_radix_topk")?;
    let command = ctx.command_buffer();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&score_pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(key_cache), 0);
    encoder.set_buffer(2, Some(&head_weights.buffer), 0);
    encoder.set_buffer(3, Some(ordered_scores), 0);
    set_bytes(&encoder, 4, &rows_u32);
    set_bytes(&encoder, 5, &heads_u32);
    set_bytes(&encoder, 6, &dim_u32);
    encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&radix_pipeline);
    encoder.set_buffer(0, Some(ordered_scores), 0);
    encoder.set_buffer(1, Some(selection), 0);
    encoder.set_buffer(2, Some(radix_state), 0);
    set_bytes(&encoder, 3, &rows_u32);
    set_bytes(&encoder, 4, &topk_u32);
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();

    let shape = format!("rows={rows},heads={head_count},dim={head_dim},top_k={top_k}");
    let read_bytes = query.buffer.length() + (rows * head_dim * mem::size_of::<f16>()) as u64 + head_weights.buffer.length();
    ctx.commit_and_wait_profiled(&command, "dsa_select_topk", &shape, read_bytes, (rows * mem::size_of::<u32>() + top_k * mem::size_of::<u32>()) as u64);
    Ok(())
}
