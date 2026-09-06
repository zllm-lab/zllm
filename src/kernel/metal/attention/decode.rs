//! GQA decode attention 的 kernel 封装:BF16 并行、split-kv 与 Q8 cache 变体。

use crate::backend::metal::api as metal;

use super::super::{MTLSize, MetalContext, MetalTensor, MetalTensorDType, mem, set_bytes, validate_u32};

#[allow(clippy::too_many_arguments)]
pub(super) fn gqa_decode_attention_parallel_bf16_buffers(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &metal::Buffer,
    key_offset: u64,
    value: &metal::Buffer,
    value_offset: u64,
    kv_rows: usize,
    first_visible: usize,
    kv_capacity: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
) -> Result<MetalTensor, String> {
    const SCORE_THREADS: usize = 128;
    const SOFTMAX_THREADS: usize = 256;
    const VALUE_THREADS: usize = 32;
    if first_visible >= kv_rows {
        return Err(format!("GQA BF16 并行 decode 可见范围 {first_visible}..{kv_rows} 为空"));
    }
    if kv_head_count == 0 || !head_count.is_multiple_of(kv_head_count) || head_count / kv_head_count != 4 || head_dim > 512 {
        return Err(format!("GQA BF16 并行 decode 要求每个 KV head 对应 4 个 query head 且 head_dim <= 512，实际 heads={head_count}, kv_heads={kv_head_count}, head_dim={head_dim}"));
    }

    let source_rows = kv_rows - first_visible;
    let score_bytes = head_count.checked_mul(source_rows).and_then(|count| count.checked_mul(mem::size_of::<f32>())).ok_or("GQA BF16 并行 decode score bytes 溢出")?;
    let scores = ctx.shared_buffer_zeros(score_bytes);
    let output = ctx.tensor_zeros_bf16(1, query.cols);
    let score_pipeline = ctx.pipeline("gqa_decode_scores_gqa_bf16")?;
    let softmax_pipeline = ctx.pipeline("gqa_decode_softmax_f32")?;
    let value_pipeline = ctx.pipeline("gqa_decode_weighted_value_tiled_bf16")?;
    if SCORE_THREADS as u64 > score_pipeline.max_total_threads_per_threadgroup() || SOFTMAX_THREADS as u64 > softmax_pipeline.max_total_threads_per_threadgroup() || VALUE_THREADS as u64 > value_pipeline.max_total_threads_per_threadgroup() {
        return Err("GQA BF16 并行 decode threads 超过 Metal pipeline 上限".to_owned());
    }

    let source_rows_u32 = validate_u32("GQA BF16 并行 decode rows", source_rows)?;
    let first_visible_u32 = validate_u32("GQA BF16 并行 decode first visible", first_visible)?;
    let kv_capacity_u32 = validate_u32("GQA BF16 并行 decode capacity", kv_capacity)?;
    let heads_u32 = validate_u32("GQA BF16 并行 decode heads", head_count)?;
    let kv_heads_u32 = validate_u32("GQA BF16 并行 decode KV heads", kv_head_count)?;
    let dimension_u32 = validate_u32("GQA BF16 并行 decode head dim", head_dim)?;
    let dimension_tiles = head_dim.div_ceil(VALUE_THREADS);
    let command = ctx.command_buffer();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&score_pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(key), key_offset);
    encoder.set_buffer(2, Some(&scores), 0);
    set_bytes(&encoder, 3, &source_rows_u32);
    set_bytes(&encoder, 4, &heads_u32);
    set_bytes(&encoder, 5, &kv_heads_u32);
    set_bytes(&encoder, 6, &dimension_u32);
    set_bytes(&encoder, 7, &score_scale);
    set_bytes(&encoder, 8, &first_visible_u32);
    set_bytes(&encoder, 9, &kv_capacity_u32);
    encoder.dispatch_thread_groups(MTLSize::new(source_rows as u64, kv_head_count as u64, 1), MTLSize::new(SCORE_THREADS as u64, 1, 1));
    encoder.end_encoding();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&softmax_pipeline);
    encoder.set_buffer(0, Some(&scores), 0);
    set_bytes(&encoder, 1, &source_rows_u32);
    set_bytes(&encoder, 2, &heads_u32);
    encoder.dispatch_thread_groups(MTLSize::new(head_count as u64, 1, 1), MTLSize::new(SOFTMAX_THREADS as u64, 1, 1));
    encoder.end_encoding();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&value_pipeline);
    encoder.set_buffer(0, Some(&scores), 0);
    encoder.set_buffer(1, Some(value), value_offset);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &source_rows_u32);
    set_bytes(&encoder, 4, &heads_u32);
    set_bytes(&encoder, 5, &kv_heads_u32);
    set_bytes(&encoder, 6, &dimension_u32);
    set_bytes(&encoder, 7, &first_visible_u32);
    set_bytes(&encoder, 8, &kv_capacity_u32);
    encoder.dispatch_thread_groups(MTLSize::new((head_count * dimension_tiles) as u64, 1, 1), MTLSize::new(VALUE_THREADS as u64, 1, 1));
    encoder.end_encoding();

    let shape = format!("q=[1,{}],kv_rows={kv_rows},visible={first_visible}..{kv_rows},heads={head_count},kv_heads={kv_head_count},head_dim={head_dim}", query.cols);
    ctx.commit_and_wait_profiled(&command, "gqa_decode_attention_parallel_bf16", &shape, query.buffer.length() + key.length() + value.length(), output.buffer.length() + scores.length());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub struct GqaQ8Scales<'a> {
    key: &'a metal::Buffer,
    key_offset: u64,
    value: &'a metal::Buffer,
    value_offset: u64,
    group_size: usize,
}

/// 小 KV 直通 + 当前行量化融合:kernel 内完成新 K/V 行的 Q8 量化与回写,
/// 省掉独立的 gqa_kv_quantize_q8 dispatch。数学上与“先量化再读 cache”逐位一致。
/// `kv_rows` 含当前行;cache 中当前行位置的内容由本 kernel 写入(调用方不得先量化)。
#[allow(clippy::too_many_arguments)]
pub(super) fn gqa_decode_attention_direct_q8_append_buffers(
    ctx: &MetalContext,
    query: &MetalTensor,
    new_key: &MetalTensor,
    new_value: &MetalTensor,
    key: &metal::Buffer,
    key_offset: u64,
    key_scale_offset: u64,
    value: &metal::Buffer,
    value_offset: u64,
    value_scale_offset: u64,
    kv_rows: usize,
    first_visible: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
    group_size: usize,
) -> Result<MetalTensor, String> {
    const THREADS: usize = 256;
    if kv_rows == 0 || first_visible >= kv_rows {
        return Err(format!("GQA direct-append 可见范围 {first_visible}..{kv_rows} 非法"));
    }
    let source_rows = kv_rows - first_visible;
    if source_rows > 512 {
        return Err(format!("GQA direct-append source_rows={source_rows} 超过 512"));
    }
    if kv_head_count == 0
        || !head_count.is_multiple_of(kv_head_count)
        || head_dim > 128
        || !head_dim.is_multiple_of(32)
        || !head_dim.is_multiple_of(group_size)
        || !group_size.is_multiple_of(4)
        || kv_head_count * head_dim > 1024
        || kv_head_count * (head_dim / group_size) > 16
    {
        return Err(format!("GQA direct-append 维度不符: heads={head_count}, kv_heads={kv_head_count}, head_dim={head_dim}, group={group_size}"));
    }
    if new_key.cols != kv_head_count * head_dim || new_value.cols != kv_head_count * head_dim || new_key.rows != 1 || new_value.rows != 1 {
        return Err(format!("GQA direct-append 新 K/V shape 不符: K=[{},{}] V=[{},{}]", new_key.rows, new_key.cols, new_value.rows, new_value.cols));
    }
    let bf16 = query.dtype == MetalTensorDType::Bf16;
    let output = if bf16 { ctx.tensor_kernel_output_bf16(1, query.cols) } else { ctx.tensor_kernel_output(1, query.cols) };
    let pipeline = ctx.pipeline("gqa_decode_direct_q8_append")?;
    if THREADS as u64 > pipeline.max_total_threads_per_threadgroup() {
        return Err("GQA direct-append 需要 256 threads，超过 Metal pipeline 上限".to_owned());
    }
    let first_visible_u32 = validate_u32("GQA direct-append first visible", first_visible)?;
    let kv_rows_u32 = validate_u32("GQA direct-append KV rows", kv_rows)?;
    let heads_u32 = validate_u32("GQA direct-append heads", head_count)?;
    let kv_heads_u32 = validate_u32("GQA direct-append KV heads", kv_head_count)?;
    let dimension_u32 = validate_u32("GQA direct-append head dim", head_dim)?;
    let bf16_flag = u32::from(bf16);
    let group_size_u32 = validate_u32("GQA direct-append group size", group_size)?;
    let groups_per_head_u32 = validate_u32("GQA direct-append groups per head", head_dim / group_size)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(key), key_offset);
    encoder.set_buffer(2, Some(value), value_offset);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    encoder.set_buffer(4, Some(key), key_scale_offset);
    encoder.set_buffer(5, Some(value), value_scale_offset);
    encoder.set_buffer(6, Some(&new_key.buffer), 0);
    encoder.set_buffer(7, Some(&new_value.buffer), 0);
    let decode_state = [kv_rows_u32 - 1, kv_rows_u32, first_visible_u32];
    set_bytes(&encoder, 8, &decode_state);
    set_bytes(&encoder, 9, &heads_u32);
    set_bytes(&encoder, 10, &kv_heads_u32);
    set_bytes(&encoder, 11, &dimension_u32);
    set_bytes(&encoder, 12, &score_scale);
    set_bytes(&encoder, 13, &bf16_flag);
    set_bytes(&encoder, 15, &group_size_u32);
    set_bytes(&encoder, 16, &groups_per_head_u32);
    let head_groups = kv_head_count * (head_count / kv_head_count).div_ceil(2);
    encoder.dispatch_thread_groups(MTLSize::new(head_groups as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("q=[1,{}],kv_rows={kv_rows},heads={head_count},kv_heads={kv_head_count},head_dim={head_dim}", query.cols);
    ctx.commit_and_wait_profiled(&command, "gqa_decode_attention_direct_q8_append", &shape, query.buffer.length() + key.length() + value.length(), output.buffer.length());
    Ok(output)
}

/// 小 KV(<=256 行)decode 直通:单 kernel 完成 softmax+加权 V,无临时 buffer、无 merge。
/// 数学上与 block_count=1 的 split+merge 逐位一致。
#[allow(clippy::too_many_arguments)]
pub(super) fn gqa_decode_attention_direct_q8_buffers(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &metal::Buffer,
    key_offset: u64,
    key_scale_offset: u64,
    value: &metal::Buffer,
    value_offset: u64,
    value_scale_offset: u64,
    kv_rows: usize,
    first_visible: usize,
    kv_capacity: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
    group_size: usize,
) -> Result<MetalTensor, String> {
    const THREADS: usize = 256;
    if first_visible >= kv_rows {
        return Err(format!("GQA direct-Q8 可见范围 {first_visible}..{kv_rows} 为空"));
    }
    let source_rows = kv_rows - first_visible;
    if source_rows > 256 {
        return Err(format!("GQA direct-Q8 source_rows={source_rows} 超过 256"));
    }
    if kv_head_count == 0 || !head_count.is_multiple_of(kv_head_count) || head_dim > 128 || !head_dim.is_multiple_of(32) || !head_dim.is_multiple_of(group_size) || !group_size.is_multiple_of(4) {
        return Err(format!("GQA direct-Q8 维度不符: heads={head_count}, kv_heads={kv_head_count}, head_dim={head_dim}, group={group_size}"));
    }
    let bf16 = query.dtype == MetalTensorDType::Bf16;
    let output = if bf16 { ctx.tensor_kernel_output_bf16(1, query.cols) } else { ctx.tensor_kernel_output(1, query.cols) };
    let pipeline = ctx.pipeline("gqa_decode_direct_q8")?;
    if THREADS as u64 > pipeline.max_total_threads_per_threadgroup() {
        return Err("GQA direct-Q8 需要 256 threads，超过 Metal pipeline 上限".to_owned());
    }
    let source_rows_u32 = validate_u32("GQA direct-Q8 rows", source_rows)?;
    let first_visible_u32 = validate_u32("GQA direct-Q8 first visible", first_visible)?;
    let kv_capacity_u32 = validate_u32("GQA direct-Q8 capacity", kv_capacity)?;
    let heads_u32 = validate_u32("GQA direct-Q8 heads", head_count)?;
    let kv_heads_u32 = validate_u32("GQA direct-Q8 KV heads", kv_head_count)?;
    let dimension_u32 = validate_u32("GQA direct-Q8 head dim", head_dim)?;
    let bf16_flag = u32::from(bf16);
    let group_size_u32 = validate_u32("GQA direct-Q8 group size", group_size)?;
    let groups_per_head_u32 = validate_u32("GQA direct-Q8 groups per head", head_dim / group_size)?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(key), key_offset);
    encoder.set_buffer(2, Some(value), value_offset);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    encoder.set_buffer(4, Some(key), key_scale_offset);
    encoder.set_buffer(5, Some(value), value_scale_offset);
    set_bytes(&encoder, 6, &source_rows_u32);
    set_bytes(&encoder, 7, &heads_u32);
    set_bytes(&encoder, 8, &kv_heads_u32);
    set_bytes(&encoder, 9, &dimension_u32);
    set_bytes(&encoder, 10, &score_scale);
    set_bytes(&encoder, 11, &bf16_flag);
    set_bytes(&encoder, 12, &first_visible_u32);
    set_bytes(&encoder, 13, &kv_capacity_u32);
    set_bytes(&encoder, 14, &group_size_u32);
    set_bytes(&encoder, 15, &groups_per_head_u32);
    let head_groups = kv_head_count * (head_count / kv_head_count).div_ceil(4);
    encoder.dispatch_thread_groups(MTLSize::new(head_groups as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("q=[1,{}],kv_rows={kv_rows},visible={first_visible}..{kv_rows},heads={head_count},kv_heads={kv_head_count},head_dim={head_dim}", query.cols);
    ctx.commit_and_wait_profiled(&command, "gqa_decode_attention_direct_q8", &shape, query.buffer.length() + key.length() + value.length(), output.buffer.length());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gqa_decode_attention_split_kv_buffers_q8(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &metal::Buffer,
    key_offset: u64,
    key_scale_offset: u64,
    value: &metal::Buffer,
    value_offset: u64,
    value_scale_offset: u64,
    kv_rows: usize,
    first_visible: usize,
    kv_capacity: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
    group_size: usize,
) -> Result<MetalTensor, String> {
    gqa_decode_attention_split_kv_buffers(
        ctx,
        query,
        key,
        key_offset,
        value,
        value_offset,
        kv_rows,
        first_visible,
        kv_capacity,
        head_count,
        kv_head_count,
        head_dim,
        score_scale,
        query.dtype == MetalTensorDType::Bf16,
        Some(GqaQ8Scales { key, key_offset: key_scale_offset, value, value_offset: value_scale_offset, group_size }),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_decode_attention_split_kv_buffers(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &metal::Buffer,
    key_offset: u64,
    value: &metal::Buffer,
    value_offset: u64,
    kv_rows: usize,
    first_visible: usize,
    kv_capacity: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
    bf16: bool,
    q8: Option<GqaQ8Scales<'_>>,
) -> Result<MetalTensor, String> {
    const THREADS: usize = 256;
    if first_visible >= kv_rows {
        return Err(format!("GQA Split-KV 可见范围 {first_visible}..{kv_rows} 为空"));
    }
    let source_rows = kv_rows - first_visible;
    let group_size = validate_u32("GQA Split-KV Q8 group size", q8.as_ref().map_or(1, |storage| storage.group_size))?;
    if q8.is_some() && !head_dim.is_multiple_of(group_size as usize) {
        return Err(format!("GQA Split-KV head_dim={head_dim} 不能被 Q8 group={group_size} 整除"));
    }
    let groups_per_head = validate_u32("GQA Split-KV Q8 groups per head", head_dim / group_size as usize)?;
    let use_flash_q8 = q8.is_some() && source_rows >= 4096 && head_dim <= THREADS && head_dim.is_multiple_of(8) && groups_per_head <= 8;
    let block_tokens = if use_flash_q8 || source_rows >= 32 * 1024 { 256 } else { 128 };
    let block_count = if use_flash_q8 { source_rows.min(64) } else { source_rows.div_ceil(block_tokens) };
    let statistics_bytes = head_count.checked_mul(block_count).and_then(|count| count.checked_mul(2 * mem::size_of::<f32>())).ok_or("GQA Split-KV statistics bytes 溢出")?;
    let partial_bytes = head_count.checked_mul(block_count).and_then(|count| count.checked_mul(head_dim)).and_then(|count| count.checked_mul(mem::size_of::<f32>())).ok_or("GQA Split-KV partial bytes 溢出")?;
    let statistics = ctx.shared_buffer_uninit(statistics_bytes);
    let partial_values = ctx.shared_buffer_uninit(partial_bytes);
    let output = if bf16 { ctx.tensor_kernel_output_bf16(1, query.cols) } else { ctx.tensor_kernel_output(1, query.cols) };
    let split_pipeline = if use_flash_q8 {
        ctx.pipeline_u32_constants("gqa_decode_flash_q8_grouped", &[validate_u32("GQA Flash-Q8 head dim", head_dim)?, group_size, groups_per_head, validate_u32("GQA Flash-Q8 heads per KV", head_count / kv_head_count)?])?
    } else {
        ctx.pipeline(if bf16 { "gqa_decode_split_kv_bf16_vectorized" } else { "gqa_decode_split_kv" })?
    };
    let merge_pipeline = ctx.pipeline("gqa_decode_split_kv_merge")?;
    if THREADS as u64 > split_pipeline.max_total_threads_per_threadgroup() || THREADS as u64 > merge_pipeline.max_total_threads_per_threadgroup() {
        return Err("GQA Split-KV 需要 256 threads，超过 Metal pipeline 上限".to_owned());
    }

    let source_rows_u32 = validate_u32("GQA Split-KV rows", source_rows)?;
    let kv_rows_u32 = validate_u32("GQA Split-KV KV rows", kv_rows)?;
    let first_visible_u32 = validate_u32("GQA Split-KV first visible", first_visible)?;
    let kv_capacity_u32 = validate_u32("GQA Split-KV capacity", kv_capacity)?;
    let heads_u32 = validate_u32("GQA Split-KV heads", head_count)?;
    let kv_heads_u32 = validate_u32("GQA Split-KV KV heads", kv_head_count)?;
    let dimension_u32 = validate_u32("GQA Split-KV head dim", head_dim)?;
    let block_tokens_u32 = block_tokens as u32;
    let block_count_u32 = validate_u32("GQA Split-KV block count", block_count)?;
    let bf16_flag = u32::from(bf16);
    let q8_flag = u32::from(q8.is_some());
    let command = ctx.command_buffer();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&split_pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(key), key_offset);
    encoder.set_buffer(2, Some(value), value_offset);
    encoder.set_buffer(3, Some(&statistics), 0);
    encoder.set_buffer(4, Some(&partial_values), 0);
    let decode_state = [kv_rows_u32.saturating_sub(1), kv_rows_u32, first_visible_u32];
    // BF16 vectorized kernel 保留原有 source_rows/first_visible 参数；只有
    // gqa_decode_split_kv 从三槽 state 读取动态范围，供 replay 复用。
    if use_flash_q8 || bf16 {
        set_bytes(&encoder, 5, &source_rows_u32);
    } else {
        set_bytes(&encoder, 5, &decode_state);
    }
    set_bytes(&encoder, 6, &heads_u32);
    set_bytes(&encoder, 7, &kv_heads_u32);
    set_bytes(&encoder, 8, &dimension_u32);
    set_bytes(&encoder, 9, &block_tokens_u32);
    set_bytes(&encoder, 10, &block_count_u32);
    set_bytes(&encoder, 11, &score_scale);
    set_bytes(&encoder, 12, &bf16_flag);
    if use_flash_q8 || bf16 {
        set_bytes(&encoder, 13, &first_visible_u32);
    }
    set_bytes(&encoder, 14, &kv_capacity_u32);
    if let Some(storage) = &q8 {
        encoder.set_buffer(15, Some(storage.key), storage.key_offset);
        encoder.set_buffer(16, Some(storage.value), storage.value_offset);
    } else {
        encoder.set_buffer(15, Some(key), key_offset);
        encoder.set_buffer(16, Some(value), value_offset);
    }
    set_bytes(&encoder, 17, &group_size);
    set_bytes(&encoder, 18, &groups_per_head);
    set_bytes(&encoder, 19, &q8_flag);
    let split_height = kv_head_count * (head_count / kv_head_count).div_ceil(4);
    encoder.dispatch_thread_groups(MTLSize::new(block_count as u64, split_height as u64, 1), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&merge_pipeline);
    encoder.set_buffer(0, Some(&statistics), 0);
    encoder.set_buffer(1, Some(&partial_values), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &block_count_u32);
    set_bytes(&encoder, 4, &heads_u32);
    set_bytes(&encoder, 5, &dimension_u32);
    set_bytes(&encoder, 6, &bf16_flag);
    encoder.dispatch_thread_groups(MTLSize::new(head_count as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();

    let shape = format!("q=[1,{}],kv_rows={kv_rows},visible={first_visible}..{kv_rows},heads={head_count},kv_heads={kv_head_count},head_dim={head_dim},block={block_tokens},partitions={block_count}", query.cols);
    ctx.commit_and_wait_profiled(
        &command,
        if use_flash_q8 {
            "gqa_decode_attention_flash_q8"
        } else if q8.is_some() {
            "gqa_decode_attention_split_kv_q8"
        } else if bf16 {
            "gqa_decode_attention_split_kv_bf16"
        } else {
            "gqa_decode_attention_split_kv_f16"
        },
        &shape,
        query.buffer.length() + key.length() + value.length(),
        output.buffer.length() + statistics.length() + partial_values.length(),
    );
    Ok(output)
}
