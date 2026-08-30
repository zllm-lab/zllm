//! GQA prefill attention 的 MPSMatrix 链路:F32/F16/BF16 与滑窗变体。

use crate::backend::metal::api as metal;

use super::super::{MTLSize, MetalContext, MetalTensor, MetalTensorDType, f16, launch_1d, mem, set_bytes, to_bf16_tensor, to_f16_tensor, validate_u32};
use crate::kernel::metal::mps as mps_kernels;
use mps_kernels::{encode_f16_matmul_f32, encode_f32_f16_matmul_f32};

#[allow(clippy::too_many_arguments)]
pub(super) fn gqa_prefill_attention_mps_bf16_buffers(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &metal::Buffer,
    key_offset: u64,
    value: &metal::Buffer,
    value_offset: u64,
    kv_rows: usize,
    position: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
) -> Result<MetalTensor, String> {
    let query = to_f16_tensor(ctx, query)?;
    let kv_elements = kv_rows.checked_mul(kv_head_count).and_then(|count| count.checked_mul(head_dim)).ok_or("GQA MPS BF16 KV elements 溢出")?;
    let cast = |input: &metal::Buffer, input_offset: u64, label: &str| -> Result<metal::Buffer, String> {
        let output = ctx.shared_buffer_zeros(kv_elements * mem::size_of::<f16>());
        let count = validate_u32(label, kv_elements)?;
        let shape = format!("elements={kv_elements}");
        launch_1d(ctx, "cast_bf16_f16", &shape, kv_elements, output.length(), output.length(), |encoder| {
            encoder.set_buffer(0, Some(input), input_offset);
            encoder.set_buffer(1, Some(&output), 0);
            set_bytes(encoder, 2, &count);
        })?;
        Ok(output)
    };
    let key = cast(key, key_offset, "GQA MPS BF16 key elements")?;
    let value = cast(value, value_offset, "GQA MPS BF16 value elements")?;
    let output = gqa_prefill_attention_mps_buffers(ctx, &query, &key, 0, &value, 0, kv_rows, position, head_count, kv_head_count, head_dim, score_scale)?;
    to_bf16_tensor(ctx, &output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gqa_prefill_attention_mps_buffers(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &metal::Buffer,
    key_offset: u64,
    value: &metal::Buffer,
    value_offset: u64,
    kv_rows: usize,
    position: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
) -> Result<MetalTensor, String> {
    const SOFTMAX_THREADS: usize = 32;
    const LINEAR_THREADS: usize = 256;
    let score_stride = kv_rows.next_multiple_of(8);
    // M5 实测 1024 行吞吐最佳；超长序列只向下收缩，把 F32 score scratch 限制在约 64 MiB。
    let scratch_rows = (64 * 1024 * 1024 / (score_stride * mem::size_of::<f32>())).max(1);
    let default_query_block = [1024, 512, 256, 128].into_iter().find(|&block| block <= scratch_rows).unwrap_or(128);
    let query_block = default_query_block;
    let score_rows = query.rows.min(query_block);
    // QK 分数必须保留 F32；在较大且相近的 logits 上提前落 F16 会改变 softmax 排序。
    // F16 query 只把归一后的概率落到 F16，使 P×V 仍走快速的 F16×F16、F32 累加路径。
    let f16_probabilities = query.dtype == MetalTensorDType::F16;
    let scores = ctx.shared_buffer_zeros(score_rows * score_stride * mem::size_of::<f32>());
    let probabilities = f16_probabilities.then(|| ctx.shared_buffer_zeros(score_rows * score_stride * mem::size_of::<f16>()));
    let pv_accumulator = ctx.shared_buffer_zeros(score_rows * head_dim * mem::size_of::<f32>());
    let output = ctx.tensor_kernel_output(query.rows, query.cols);
    let softmax = ctx.pipeline(if f16_probabilities { "causal_softmax_rows_f32_f16" } else { "causal_softmax_rows_f32" })?;
    let cast_pv = ctx.pipeline("cast_f32_to_f16_strided")?;
    let score_fence = ctx.device.new_fence();
    let probability_fence = ctx.device.new_fence();
    let pv_fence = ctx.device.new_fence();
    let scratch_fence = ctx.device.new_fence();
    let mut scratch_ready = false;
    let command = ctx.command_buffer();
    let query_f32 = query.dtype == MetalTensorDType::F32;
    let query_element_bytes = if query_f32 { mem::size_of::<f32>() } else { mem::size_of::<f16>() };
    let query_row_bytes = query.cols * query_element_bytes;
    let output_row_bytes = query.cols * mem::size_of::<f16>();
    let kv_row_bytes = kv_head_count * head_dim * mem::size_of::<f16>();
    let scale = score_scale as f64;
    let score_stride_u32 = validate_u32("GQA MPS score stride", score_stride)?;

    for query_head in 0..head_count {
        let kv_head = query_head / (head_count / kv_head_count);
        for block_begin in (0..query.rows).step_by(query_block) {
            let block_rows = (query.rows - block_begin).min(query_block);
            let query_begin = position + block_begin;
            let visible = kv_rows.min(query_begin + block_rows);
            let query_offset = block_begin * query_row_bytes + query_head * head_dim * query_element_bytes;
            let kv_column_offset = kv_head * head_dim * mem::size_of::<f16>();
            if scratch_ready {
                let barrier = command.new_compute_command_encoder();
                barrier.wait_for_fence(&scratch_fence);
                barrier.end_encoding();
            }
            if query_f32 {
                encode_f32_f16_matmul_f32(
                    &command,
                    &ctx.device,
                    &query.buffer,
                    query_offset,
                    query_row_bytes,
                    key,
                    key_offset as usize + kv_column_offset,
                    kv_row_bytes,
                    &scores,
                    0,
                    score_stride * mem::size_of::<f32>(),
                    block_rows,
                    head_dim,
                    visible,
                    true,
                    scale,
                )?;
            } else {
                encode_f16_matmul_f32(
                    &command,
                    &ctx.device,
                    &query.buffer,
                    query_offset,
                    query_row_bytes,
                    key,
                    key_offset as usize + kv_column_offset,
                    kv_row_bytes,
                    &scores,
                    0,
                    score_stride * mem::size_of::<f32>(),
                    block_rows,
                    head_dim,
                    visible,
                    true,
                    scale,
                )?;
            }
            let signal = command.new_compute_command_encoder();
            signal.update_fence(&score_fence);
            signal.end_encoding();

            let block_rows_u32 = validate_u32("GQA MPS block rows", block_rows)?;
            let visible_u32 = validate_u32("GQA MPS visible rows", visible)?;
            let query_begin_u32 = validate_u32("GQA MPS query begin", query_begin)?;
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&softmax);
            encoder.wait_for_fence(&score_fence);
            encoder.set_buffer(0, Some(&scores), 0);
            encoder.set_buffer(1, Some(probabilities.as_ref().unwrap_or(&scores)), 0);
            set_bytes(&encoder, 2, &block_rows_u32);
            set_bytes(&encoder, 3, &visible_u32);
            set_bytes(&encoder, 4, &score_stride_u32);
            set_bytes(&encoder, 5, &query_begin_u32);
            encoder.dispatch_thread_groups(MTLSize::new(block_rows as u64, 1, 1), MTLSize::new(SOFTMAX_THREADS as u64, 1, 1));
            encoder.update_fence(&probability_fence);
            encoder.end_encoding();
            let barrier = command.new_compute_command_encoder();
            barrier.wait_for_fence(&probability_fence);
            barrier.end_encoding();

            if let Some(probabilities) = &probabilities {
                encode_f16_matmul_f32(
                    &command,
                    &ctx.device,
                    probabilities,
                    0,
                    score_stride * mem::size_of::<f16>(),
                    value,
                    value_offset as usize + kv_column_offset,
                    kv_row_bytes,
                    &pv_accumulator,
                    0,
                    head_dim * mem::size_of::<f32>(),
                    block_rows,
                    visible,
                    head_dim,
                    false,
                    1.0,
                )?;
            } else {
                encode_f32_f16_matmul_f32(
                    &command,
                    &ctx.device,
                    &scores,
                    0,
                    score_stride * mem::size_of::<f32>(),
                    value,
                    value_offset as usize + kv_column_offset,
                    kv_row_bytes,
                    &pv_accumulator,
                    0,
                    head_dim * mem::size_of::<f32>(),
                    block_rows,
                    visible,
                    head_dim,
                    false,
                    1.0,
                )?;
            }
            let signal = command.new_compute_command_encoder();
            signal.update_fence(&pv_fence);
            signal.end_encoding();
            let output_offset = block_begin * output_row_bytes + query_head * head_dim * mem::size_of::<f16>();
            let count = validate_u32("GQA P×V cast count", block_rows * head_dim)?;
            let rows = validate_u32("GQA P×V cast rows", block_rows)?;
            let columns = validate_u32("GQA P×V cast columns", head_dim)?;
            let output_stride = validate_u32("GQA P×V output stride", query.cols)?;
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&cast_pv);
            encoder.wait_for_fence(&pv_fence);
            encoder.set_buffer(0, Some(&pv_accumulator), 0);
            encoder.set_buffer(1, Some(&output.buffer), output_offset as u64);
            set_bytes(&encoder, 2, &rows);
            set_bytes(&encoder, 3, &columns);
            set_bytes(&encoder, 4, &output_stride);
            set_bytes(&encoder, 5, &count);
            encoder.dispatch_thread_groups(MTLSize::new((block_rows * head_dim).div_ceil(LINEAR_THREADS) as u64, 1, 1), MTLSize::new(LINEAR_THREADS as u64, 1, 1));
            encoder.update_fence(&scratch_fence);
            encoder.end_encoding();
            scratch_ready = true;
        }
    }

    let shape = format!("q=[{},{}],kv_rows={kv_rows},position={position},heads={head_count},kv_heads={kv_head_count},head_dim={head_dim},block={query_block}", query.rows, query.cols);
    let probability_bytes = probabilities.as_ref().map_or(0, |buffer| buffer.length());
    ctx.commit_and_wait_profiled(&command, "gqa_prefill_attention_mps_f32", &shape, query.buffer.length() + key.length() + value.length(), output.buffer.length() + scores.length() + probability_bytes + pv_accumulator.length());
    Ok(output)
}

/// 滑窗版 MPS prefill attention:每个 query block 的 K/V 列范围平移到
/// [max(kv_start, 块首行窗口起点), 块尾行可见数),浪费率 ~(window+block)/window;
/// softmax 用逐行下界掩码保证窗口语义,P·V 对窗口外零概率天然安全。
#[allow(clippy::too_many_arguments)]
pub(super) fn gqa_prefill_attention_mps_windowed_buffers(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &metal::Buffer,
    key_offset: u64,
    value: &metal::Buffer,
    value_offset: u64,
    kv_rows: usize,
    position: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
    window: usize,
) -> Result<MetalTensor, String> {
    const SOFTMAX_THREADS: usize = 32;
    const LINEAR_THREADS: usize = 256;
    // block 越小 K/V 列浪费越低,但 encoder 数量按 heads×blocks 增长、CPU 编码成本上升;
    // 512 行让浪费率停在 2×,同时把 encoder 数砍半
    const WINDOW_QUERY_BLOCK: usize = 512;
    if query.dtype != MetalTensorDType::F16 {
        return Err(format!("GQA windowed MPS 只支持 F16 query，实际 {:?}", query.dtype));
    }
    let window = window.max(1);
    let score_rows = query.rows.min(WINDOW_QUERY_BLOCK);
    let pv_bytes_per_row = head_dim * mem::size_of::<f32>();
    let pv_buffer_len = score_rows * pv_bytes_per_row;
    let mut score_buffer_len = 0usize;
    let mut probability_buffer_len = 0usize;
    let output = ctx.tensor_kernel_output(query.rows, query.cols);
    let softmax = ctx.pipeline("causal_softmax_windowed_rows_f32")?;
    let cast_pv = ctx.pipeline("cast_f32_to_f16_strided")?;
    let score_fence = ctx.device.new_fence();
    let probability_fence = ctx.device.new_fence();
    let pv_fence = ctx.device.new_fence();
    let scratch_fence = ctx.device.new_fence();
    let mut scratch_ready = false;
    let command = ctx.command_buffer();
    let query_row_bytes = query.cols * mem::size_of::<f16>();
    let output_row_bytes = query.cols * mem::size_of::<f16>();
    let kv_row_bytes = kv_head_count * head_dim * mem::size_of::<f16>();

    for query_head in 0..head_count {
        let kv_head = query_head / (head_count / kv_head_count);
        let pv_accumulator = ctx.cached_zero_buffer_slot("gqa_mps_windowed_pv", query_head, pv_buffer_len);
        for block_begin in (0..query.rows).step_by(WINDOW_QUERY_BLOCK) {
            let block_rows = (query.rows - block_begin).min(WINDOW_QUERY_BLOCK);
            let query_begin = position + block_begin;
            // 块内首行的窗口起点;块内后续行窗口更晚,交给 softmax 的逐行下界掩码
            let common_start = query_begin.saturating_sub(window - 1);
            let visible = kv_rows.min(query_begin + block_rows);
            let columns = visible.saturating_sub(common_start).max(1);
            let score_stride = columns.next_multiple_of(8);
            let score_bytes = score_rows * score_stride * mem::size_of::<f32>();
            let probability_bytes = score_rows * score_stride * mem::size_of::<f32>();
            if score_bytes > score_buffer_len {
                score_buffer_len = score_bytes;
            }
            if probability_bytes > probability_buffer_len {
                probability_buffer_len = probability_bytes;
            }
            // MPS 窗口矩阵 view 会延迟读取 score/probability；它们按 head 分槽，
            // 每个 block 结束时用 fence 保护下一次复用。
            let scores = ctx.cached_zero_buffer_slot("gqa_mps_windowed_scores", query_head, score_bytes);
            let probabilities = ctx.cached_zero_buffer_slot("gqa_mps_windowed_probabilities", query_head, probability_bytes);
            let query_offset = block_begin * query_row_bytes + query_head * head_dim * mem::size_of::<f16>();
            let kv_column_offset = kv_head * head_dim * mem::size_of::<f16>();
            let kv_start_offset = common_start * kv_row_bytes;
            if scratch_ready {
                let barrier = command.new_compute_command_encoder();
                barrier.wait_for_fence(&scratch_fence);
                barrier.end_encoding();
            }
            encode_f16_matmul_f32(
                &command,
                &ctx.device,
                &query.buffer,
                query_offset,
                query_row_bytes,
                key,
                key_offset as usize + kv_start_offset + kv_column_offset,
                kv_row_bytes,
                &scores,
                0,
                score_stride * mem::size_of::<f32>(),
                block_rows,
                head_dim,
                columns,
                true,
                score_scale as f64,
            )?;
            let signal = command.new_compute_command_encoder();
            signal.update_fence(&score_fence);
            signal.end_encoding();

            let block_rows_u32 = validate_u32("GQA windowed MPS block rows", block_rows)?;
            let columns_u32 = validate_u32("GQA windowed MPS columns", columns)?;
            let stride_u32 = validate_u32("GQA windowed MPS stride", score_stride)?;
            let query_begin_u32 = validate_u32("GQA windowed MPS query begin", query_begin.saturating_sub(common_start))?;
            let window_u32 = validate_u32("GQA windowed MPS window", window)?;
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&softmax);
            encoder.wait_for_fence(&score_fence);
            encoder.set_buffer(0, Some(&scores), 0);
            encoder.set_buffer(1, Some(&probabilities), 0);
            set_bytes(&encoder, 2, &block_rows_u32);
            set_bytes(&encoder, 3, &columns_u32);
            set_bytes(&encoder, 4, &stride_u32);
            set_bytes(&encoder, 5, &query_begin_u32);
            set_bytes(&encoder, 6, &window_u32);
            encoder.dispatch_thread_groups(MTLSize::new(block_rows as u64, 1, 1), MTLSize::new(SOFTMAX_THREADS as u64, 1, 1));
            encoder.update_fence(&probability_fence);
            encoder.end_encoding();
            let barrier = command.new_compute_command_encoder();
            barrier.wait_for_fence(&probability_fence);
            barrier.end_encoding();

            encode_f32_f16_matmul_f32(
                &command,
                &ctx.device,
                &probabilities,
                0,
                score_stride * mem::size_of::<f32>(),
                value,
                value_offset as usize + kv_start_offset + kv_column_offset,
                kv_row_bytes,
                &pv_accumulator,
                0,
                pv_bytes_per_row,
                block_rows,
                columns,
                head_dim,
                false,
                1.0,
            )?;
            let signal = command.new_compute_command_encoder();
            signal.update_fence(&pv_fence);
            signal.end_encoding();
            let output_offset = block_begin * output_row_bytes + query_head * head_dim * mem::size_of::<f16>();
            let count = validate_u32("GQA windowed P×V cast count", block_rows * head_dim)?;
            let rows = validate_u32("GQA windowed P×V cast rows", block_rows)?;
            let columns = validate_u32("GQA windowed P×V cast columns", head_dim)?;
            let output_stride = validate_u32("GQA windowed P×V output stride", query.cols)?;
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&cast_pv);
            encoder.wait_for_fence(&pv_fence);
            encoder.set_buffer(0, Some(&pv_accumulator), 0);
            encoder.set_buffer(1, Some(&output.buffer), output_offset as u64);
            set_bytes(&encoder, 2, &rows);
            set_bytes(&encoder, 3, &columns);
            set_bytes(&encoder, 4, &output_stride);
            set_bytes(&encoder, 5, &count);
            encoder.dispatch_thread_groups(MTLSize::new((block_rows * head_dim).div_ceil(LINEAR_THREADS) as u64, 1, 1), MTLSize::new(LINEAR_THREADS as u64, 1, 1));
            encoder.update_fence(&scratch_fence);
            encoder.end_encoding();
            scratch_ready = true;
        }
    }

    let shape = format!("q=[{},{}],kv_rows={kv_rows},position={position},heads={head_count},kv_heads={kv_head_count},head_dim={head_dim},window={window},block={WINDOW_QUERY_BLOCK}", query.rows, query.cols);
    ctx.commit_and_wait_profiled(
        &command,
        "gqa_prefill_attention_mps_windowed",
        &shape,
        query.buffer.length() + key.length() + value.length(),
        output.buffer.length() + (score_buffer_len + probability_buffer_len + pv_buffer_len) as u64 * head_count as u64,
    );
    Ok(output)
}

#[cfg(test)]
mod position_tests {
    use super::*;
    use crate::attention::gqa::{CausalWindow, GqaGeometry, GqaKvProjection, GqaSpec, HybridGqaLayerSpec, HybridGqaSpec};
    use crate::attention::rope::RopeSpec;
    use crate::backend::metal::MetalKvCache;
    use crate::kernel::metal::attention as super_attention;

    /// ICB 重放原语与 legacy 路径一致性:ring 容量 6(滑窗)、prefill 10 行(发生环绕)后
    /// decode position=10 —— position 版 append+attention 与 blit append+普通 kernel
    /// 的 cache 字节和输出必须一致。
    #[test]
    fn decode_position_append_attention_match_legacy() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let (heads, kv_heads, head_dim) = (4usize, 1usize, 64usize);
        let window = CausalWindow::Sliding { size: 6 };
        let layer = HybridGqaLayerSpec {
            geometry: GqaGeometry { num_heads: heads, num_kv_heads: kv_heads, head_dim },
            rope: RopeSpec::Default { rotary_dim: 32, theta: 10_000.0 },
            window,
            score_scale: 1.0 / (head_dim as f32).sqrt(),
            kv_projection: GqaKvProjection::Separate,
        };
        let spec = HybridGqaSpec::new(vec![layer]).unwrap();
        let spec_gqa = GqaSpec { num_heads: heads, num_kv_heads: kv_heads, head_dim, rope_dim: 32, rope_theta: 10_000.0, use_qk_norm: false, window, score_scale: 1.0 / (head_dim as f32).sqrt(), output_gate: false };
        let kv_columns = kv_heads * head_dim;
        let ctx = MetalContext::new_default().unwrap();
        let mut rng: u32 = 98765;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0 - 1.0) * 0.5
        };
        let prefill_rows = 10usize;
        let position = 10usize;
        let keys: Vec<f32> = (0..(prefill_rows + 1) * kv_columns).map(|_| next()).collect();
        let values: Vec<f32> = (0..(prefill_rows + 1) * kv_columns).map(|_| next()).collect();
        let queries: Vec<f32> = (0..(prefill_rows + 1) * heads * head_dim).map(|_| next()).collect();

        let run_round = |use_position_kernel: bool| -> (Vec<u8>, Vec<f32>) {
            let mut cache = MetalKvCache::new_hybrid_gqa(&ctx, spec.clone(), 32).unwrap();
            let kv_len = prefill_rows * kv_columns;
            // prefill 只需 append:把 ring 填满并制造一次环绕(容量 6 < 10 行)
            let prefill_key = ctx.tensor_from_f32(&keys[..kv_len], prefill_rows, kv_columns).unwrap();
            let prefill_value = ctx.tensor_from_f32(&values[..kv_len], prefill_rows, kv_columns).unwrap();
            cache.append_layer_gqa_tensor(&ctx, 0, 0, &prefill_key, &prefill_value).unwrap();

            let decode_key = ctx.tensor_from_f32(&keys[kv_len..], 1, kv_columns).unwrap();
            let decode_value = ctx.tensor_from_f32(&values[kv_len..], 1, kv_columns).unwrap();
            let decode_query = ctx.tensor_from_f32(&queries[prefill_rows * heads * head_dim..], 1, heads * head_dim).unwrap();
            let output = if use_position_kernel {
                // 槽 [position, kv_rows, kv_start] 由 CPU 填写,数值与 hybrid 状态推进一致
                let retained = cache.hybrid_gqa_state().unwrap().retained_range(0).unwrap();
                assert_eq!((retained.start, retained.end), (prefill_rows.saturating_sub(6), prefill_rows), "prefill 后 retained 应为环绕窗口");
                let state = [position as u32, (position + 1) as u32, retained.start as u32];
                let state_buffer = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(state.as_ptr().cast::<u8>(), 12) });
                let view = cache.gqa_layer_view(0).unwrap();
                super_attention::gqa_kv_append_position_tensor(&ctx, &view, &decode_key, &decode_value, &state_buffer, 0).unwrap();
                super_attention::gqa_decode_attention_position_tensor(&ctx, &decode_query, &view, &spec_gqa, &state_buffer, 0).unwrap()
            } else {
                cache.append_layer_gqa_tensor(&ctx, 0, position, &decode_key, &decode_value).unwrap();
                let view = cache.gqa_layer_view(0).unwrap();
                super_attention::gqa_prefill_attention_cached_tensor(&ctx, &decode_query, &view, position, &spec_gqa).unwrap()
            };
            let cache_bytes = unsafe { std::slice::from_raw_parts(cache.buffer().contents().cast::<u8>(), cache.buffer().length() as usize) }.to_vec();
            (cache_bytes, ctx.read_f16_to_f32(&output.buffer, heads * head_dim))
        };

        let (legacy_cache, legacy_output) = run_round(false);
        let (position_cache, position_output) = run_round(true);
        assert_eq!(legacy_cache, position_cache, "position 版 append 写入的 cache 字节与 blit 路径不一致");
        for (index, (left, right)) in legacy_output.iter().zip(&position_output).enumerate() {
            // MPS GEMM 与 split-KV online softmax 的归约顺序不同，F16 输出允许 2 ulp 级舍入差异。
            assert!((left - right).abs() <= 2.5e-4, "attention 输出 d={index}: legacy={left} position={right}");
        }
        // nobar(单 warp/head)与 barrier 版 position attention 数值一致
        {
            let mut cache = MetalKvCache::new_hybrid_gqa(&ctx, spec.clone(), 32).unwrap();
            let kv_len = prefill_rows * kv_columns;
            let prefill_key = ctx.tensor_from_f32(&keys[..kv_len], prefill_rows, kv_columns).unwrap();
            let prefill_value = ctx.tensor_from_f32(&values[..kv_len], prefill_rows, kv_columns).unwrap();
            cache.append_layer_gqa_tensor(&ctx, 0, 0, &prefill_key, &prefill_value).unwrap();
            let decode_key = ctx.tensor_from_f32(&keys[kv_len..], 1, kv_columns).unwrap();
            let decode_value = ctx.tensor_from_f32(&values[kv_len..], 1, kv_columns).unwrap();
            let decode_query = ctx.tensor_from_f32(&queries[prefill_rows * heads * head_dim..], 1, heads * head_dim).unwrap();
            let retained = cache.hybrid_gqa_state().unwrap().retained_range(0).unwrap();
            let state = [position as u32, (position + 1) as u32, retained.start as u32];
            let state_buffer = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(state.as_ptr().cast::<u8>(), 12) });
            let view = cache.gqa_layer_view(0).unwrap();
            super_attention::gqa_kv_append_position_tensor(&ctx, &view, &decode_key, &decode_value, &state_buffer, 0).unwrap();
            let nobar = super_attention::gqa_decode_attention_nobar_position_tensor(&ctx, &decode_query, &view, &spec_gqa, &state_buffer, 0).unwrap();
            let nobar_output = ctx.read_f16_to_f32(&nobar.buffer, heads * head_dim);
            for (index, (left, right)) in legacy_output.iter().zip(&nobar_output).enumerate() {
                assert!((left - right).abs() < 1.0e-3, "nobar 输出 d={index}: legacy={left} nobar={right}");
            }
        }
    }
}
