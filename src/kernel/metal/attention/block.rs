//! DSpark drafter 非对称块注意力的 kernel 封装:split-block online softmax + merge。

use super::super::{MTLSize, MetalContext, MetalTensor, THREADS, mem, set_bytes, to_f16_tensor, validate_u32};
use crate::attention::block::BlockAttentionSpec;

/// query [rows, heads×dim] / key,value [kv_rows, kv_heads×dim],每行可见区间由
/// spec.visible 给出。输出 F16 [rows, heads×dim]。
pub fn block_attention_tensor(ctx: &MetalContext, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, spec: &BlockAttentionSpec) -> Result<MetalTensor, String> {
    spec.validate(query.rows, key.rows)?;
    let head_count = spec.geometry.num_heads;
    let kv_head_count = spec.geometry.num_kv_heads;
    let head_dim = spec.geometry.head_dim;
    let query_cols = head_count.checked_mul(head_dim).ok_or("block attention query 维度溢出")?;
    let kv_cols = kv_head_count.checked_mul(head_dim).ok_or("block attention KV 维度溢出")?;
    if query.cols != query_cols || key.rows != value.rows || key.cols != kv_cols || value.cols != kv_cols {
        return Err(format!("block attention shape 异常: Q=[{},{}] K=[{},{}] V=[{},{}]", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols));
    }
    let query = to_f16_tensor(ctx, query)?;
    let key = to_f16_tensor(ctx, key)?;
    let value = to_f16_tensor(ctx, value)?;

    const BLOCK_TOKENS: usize = 256;
    let block_count = key.rows.div_ceil(BLOCK_TOKENS);
    let source_rows = validate_u32("block attention kv rows", key.rows)?;
    let query_rows = validate_u32("block attention query rows", query.rows)?;
    let head_count = validate_u32("block attention heads", head_count)?;
    let kv_head_count = validate_u32("block attention kv heads", kv_head_count)?;
    let head_dim = validate_u32("block attention head dim", head_dim)?;
    let block_tokens: u32 = BLOCK_TOKENS as u32;
    let block_count = validate_u32("block attention block count", block_count)?;
    let head_rows = query.rows * spec.geometry.num_heads;
    let statistics = ctx.shared_buffer_uninit(head_rows * block_count as usize * 2 * mem::size_of::<f32>());
    let partial_values = ctx.shared_buffer_uninit(head_rows * block_count as usize * spec.geometry.head_dim * mem::size_of::<f32>());
    let visible = spec.visible.iter().flat_map(|range| [u32::try_from(range.start), u32::try_from(range.end)]).collect::<Result<Vec<_>, _>>().map_err(|_| "block attention 可见区间超过 u32".to_owned())?;
    let mut visible_bytes = Vec::with_capacity(visible.len() * 4);
    for value in &visible {
        visible_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let visible_buffer = ctx.shared_buffer(&visible_bytes);
    let output = ctx.tensor_kernel_output(query.rows, query.cols);

    let command = ctx.command_buffer();
    let split_pipeline = ctx.pipeline("block_attention_split")?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&split_pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&key.buffer), 0);
    encoder.set_buffer(2, Some(&value.buffer), 0);
    encoder.set_buffer(3, Some(&statistics), 0);
    encoder.set_buffer(4, Some(&partial_values), 0);
    encoder.set_buffer(5, Some(&visible_buffer), 0);
    set_bytes(&encoder, 6, &source_rows);
    set_bytes(&encoder, 7, &head_count);
    set_bytes(&encoder, 8, &kv_head_count);
    set_bytes(&encoder, 9, &head_dim);
    set_bytes(&encoder, 10, &block_tokens);
    set_bytes(&encoder, 11, &block_count);
    set_bytes(&encoder, 12, &spec.score_scale);
    let heads_per_kv = spec.geometry.num_heads / spec.geometry.num_kv_heads;
    let head_groups = heads_per_kv.div_ceil(4);
    encoder.dispatch_thread_groups(MTLSize::new(block_count as u64, (spec.geometry.num_kv_heads * head_groups) as u64, query_rows as u64), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();

    let merge_pipeline = ctx.pipeline("block_attention_merge")?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&merge_pipeline);
    encoder.set_buffer(0, Some(&statistics), 0);
    encoder.set_buffer(1, Some(&partial_values), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &block_count);
    set_bytes(&encoder, 4, &head_dim);
    encoder.dispatch_thread_groups(MTLSize::new(head_rows as u64, 1, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();

    let shape = format!("q=[{},{}],kv={},heads={}/{},dim={},blocks={}", query.rows, query.cols, key.rows, spec.geometry.num_heads, spec.geometry.num_kv_heads, spec.geometry.head_dim, block_count);
    ctx.commit_and_wait_profiled(&command, "block_attention_f16", &shape, query.buffer.length() + key.buffer.length() + value.buffer.length(), output.buffer.length() + statistics.length() + partial_values.length());
    Ok(output)
}
