//! DiT 扩散 Transformer Metal 算子。
//!
//! 与 `vae.rs`(VAE 解码器算子)平行,本模块覆盖 DiffusionBackend trait
//! 需要但 VAE 不用的算子:flow step / row bias / 行拼接 / 分段调制 /
//! 调制切块 / 无 mask 全序列 attention。数值 oracle 在 `src/kernel/cpu/vae.rs`。

/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: flow_step_f16, add_row_bias_f16, concat_rows_f16, concat_rows_f32, adaln_modulate_segmented_f16, gated_residual_segmented_f16, modulation_chunks_f16, full_attention_f16
pub const SHADERS: &str = r#"
kernel void flow_step_f16(
    device const half *sample [[buffer(0)]],
    device const half *velocity [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    output[gid] = half(float(sample[gid]) + scale * float(velocity[gid]));
}
kernel void add_row_bias_f16(
    device const half *input [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    output[gid] = half(float(input[gid]) + float(bias[gid % columns]));
}
kernel void concat_rows_f16(
    device const half *left [[buffer(0)]],
    device const half *right [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &left_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    output[gid] = gid < left_count ? left[gid] : right[gid - left_count];
}
kernel void concat_rows_f32(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &left_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    output[gid] = gid < left_count ? left[gid] : right[gid - left_count];
}
kernel void adaln_modulate_segmented_f16(
    device const half *input [[buffer(0)]],
    device const half *shift [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    device const uint *row_map [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &count [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint modulation = row_map[gid / columns] * columns + gid % columns;
    const float x = float(input[gid]);
    output[gid] = half(x * (1.0f + float(scale[modulation])) + float(shift[modulation]));
}
kernel void gated_residual_segmented_f16(
    device const half *residual [[buffer(0)]],
    device const half *update [[buffer(1)]],
    device const half *gate [[buffer(2)]],
    device const uint *row_map [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &count [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint modulation = row_map[gid / columns] * columns + gid % columns;
    output[gid] = half(float(residual[gid]) + float(update[gid]) * float(gate[modulation]));
}
kernel void modulation_chunks_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &chunk_id [[buffer(2)]],
    constant uint &chunks [[buffer(3)]],
    constant uint &modalities [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &input_cols [[buffer(7)]],
    constant uint &count [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint output_row = gid / hidden;
    const uint h = gid - output_row * hidden;
    const uint time_row = output_row / modalities;
    const uint modality = output_row - time_row * modalities;
    const uint source = time_row * input_cols + (modality * chunks + chunk_id) * hidden + h;
    output[gid] = input[source];
}
kernel void full_attention_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &head_count [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant float &score_scale [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (group >= rows * head_count) return;

    threadgroup float reduction[1024];
    threadgroup float control[4]; // [0]=rescale [1]=weight [2]=denominator [3]=max
    const uint query_row = group / head_count;
    const uint head = group % head_count;
    const ulong query_base = ((ulong)query_row * head_count + head) * head_dim;
    float accumulated = 0.0f;

    for (uint token = 0; token < rows; ++token) {
        const ulong key_base = ((ulong)token * head_count + head) * head_dim;
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
            if (token == 0) {
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
            const ulong value_base = ((ulong)token * head_count + head) * head_dim;
            accumulated = accumulated * control[0] + control[1] * float(value[value_base + lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane < head_dim) {
        output[query_base + lane] = half(accumulated / control[2]);
    }
}
// 全可见 self-attention tiled 版:每 SIMD-group(32 threads)处理一个 query_row,
// 寄存器 online softmax(simd_sum 零 barrier 归约),消除原版每 token 10 个 threadgroup_barrier。
// port 自 gqa_prefill_attention_tiled_f16,剥离 causal/sliding/KV-cache/GQA。
kernel void full_attention_tiled_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &head_count [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant float &score_scale [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]])
{
    const uint query_tile = group / head_count;
    const uint query_head = group % head_count;
    const uint query_row = query_tile * simd_groups + simd_group;
    if (query_row >= rows) return;

    const uint parts = (head_dim + 31u) / 32u;
    const ulong query_base = ((ulong)query_row * head_count + query_head) * head_dim;
    float query_value[8];   // cap 256(head_dim ≤ 256)
    float accumulated[8];
    for (uint part = 0; part < parts; ++part) {
        const uint dimension = simd_lane + part * 32u;
        query_value[part] = dimension < head_dim ? float(query[query_base + dimension]) : 0.0f;
        accumulated[part] = 0.0f;
    }

    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint token = 0u; token < rows; ++token) {
        const ulong key_base = ((ulong)token * head_count + query_head) * head_dim;
        float partial = 0.0f;
        for (uint part = 0; part < parts; ++part) {
            const uint dimension = simd_lane + part * 32u;
            if (dimension < head_dim) {
                partial += query_value[part] * float(key[key_base + dimension]);
            }
        }

        const float score = simd_sum(partial) * score_scale;
        float rescale = 1.0f;
        float weight = 1.0f;
        if (score > maximum) {
            rescale = exp(maximum - score);
            maximum = score;
        } else {
            weight = exp(score - maximum);
        }
        denominator = denominator * rescale + weight;

        const ulong value_base = ((ulong)token * head_count + query_head) * head_dim;
        for (uint part = 0; part < parts; ++part) {
            const uint dimension = simd_lane + part * 32u;
            if (dimension < head_dim) {
                accumulated[part] = accumulated[part] * rescale + weight * float(value[value_base + dimension]);
            }
        }
    }

    for (uint part = 0; part < parts; ++part) {
        const uint dimension = simd_lane + part * 32u;
        if (dimension < head_dim) {
            output[query_base + dimension] = half(accumulated[part] / denominator);
        }
    }
}
"#;

use super::{MTLSize, MetalContext, MetalTensor, MetalTensorDType, launch_1d, set_bytes, to_f32_tensor, validate_u32};

/// 把 u32 行映射表转成字节 slice 供 GPU buffer 上传。
fn row_map_bytes(row_map: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(row_map.as_ptr() as *const u8, std::mem::size_of_val(row_map)) }
}

/// Flow Matching Euler 步进:`sample + scale * velocity`,elementwise。
pub fn flow_step_tensor(ctx: &MetalContext, sample: &MetalTensor, velocity: &MetalTensor, scale: f32) -> Result<MetalTensor, String> {
    if sample.rows != velocity.rows || sample.cols != velocity.cols {
        return Err(format!("flow_step shape 不兼容: sample=[{},{}] velocity=[{},{}]", sample.rows, sample.cols, velocity.rows, velocity.cols));
    }
    let count = validate_u32("flow_step count", sample.len())?;
    let output = ctx.tensor_kernel_output(sample.rows, sample.cols);
    let shape = format!("elements={}", sample.len());
    launch_1d(ctx, "flow_step_f16", &shape, sample.len(), sample.buffer.length() + velocity.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&sample.buffer), 0);
        encoder.set_buffer(1, Some(&velocity.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &scale);
        set_bytes(encoder, 4, &count);
    })?;
    Ok(output)
}

/// 行偏置:每行加同一个 bias 向量(长度 = cols),elementwise。
pub fn add_row_bias_tensor(ctx: &MetalContext, input: &MetalTensor, bias: &MetalTensor) -> Result<MetalTensor, String> {
    if bias.rows != 1 || bias.cols != input.cols {
        return Err(format!("add_row_bias shape 不兼容: input=[{},{}] bias=[{},{}]", input.rows, input.cols, bias.rows, bias.cols));
    }
    let count = validate_u32("add_row_bias count", input.len())?;
    let columns = validate_u32("add_row_bias columns", input.cols)?;
    let output = ctx.tensor_kernel_output(input.rows, input.cols);
    let shape = format!("input=[{},{}]", input.rows, input.cols);
    launch_1d(ctx, "add_row_bias_f16", &shape, input.len(), input.buffer.length() + bias.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&bias.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &count);
    })?;
    Ok(output)
}

/// 行拼接:两个同列张量沿行方向拼成 `[left.rows + right.rows, cols]`。
pub fn concat_rows_tensor(ctx: &MetalContext, left: &MetalTensor, right: &MetalTensor) -> Result<MetalTensor, String> {
    if left.cols != right.cols {
        return Err(format!("concat_rows cols 不一致: left={} right={}", left.cols, right.cols));
    }
    if left.dtype != right.dtype {
        // dtype 不一致时统一到 F32 再拼接,保持与 CPU reference 一致。
        return concat_rows_tensor(ctx, &to_f32_tensor(ctx, left)?, &to_f32_tensor(ctx, right)?);
    }
    let rows = left.rows.checked_add(right.rows).ok_or("concat_rows rows 溢出")?;
    let (output, pipeline) = match left.dtype {
        MetalTensorDType::F16 => (ctx.tensor_kernel_output(rows, left.cols), "concat_rows_f16"),
        // BF16 与 F16 同为 2 字节，纯拷贝可复用 f16 kernel(bitwise)。
        MetalTensorDType::Bf16 => (ctx.tensor_kernel_output_bf16(rows, left.cols), "concat_rows_f16"),
        MetalTensorDType::F32 => (ctx.tensor_kernel_output_f32(rows, left.cols), "concat_rows_f32"),
    };
    let left_count = validate_u32("concat_rows left count", left.len())?;
    let shape = format!("left=[{},{}]+right=[{},{}]", left.rows, left.cols, right.rows, right.cols);
    launch_1d(ctx, pipeline, &shape, left.len() + right.len(), left.buffer.length() + right.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&left.buffer), 0);
        encoder.set_buffer(1, Some(&right.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &left_count);
    })?;
    Ok(output)
}

/// 分段 AdaLN 调制:`out = input * (1 + scale) + shift`,shift/scale 按 row_map 广播。
///
/// `row_map[input_row]` 给出该行应使用的 modulation 行索引。
/// shift/scale 形状 `[modulation_rows, cols]`。对应 CPU `adaln_modulate_segmented`。
pub fn adaln_modulate_segmented_tensor(ctx: &MetalContext, input: &MetalTensor, shift: &MetalTensor, scale: &MetalTensor, row_map: &[u32]) -> Result<MetalTensor, String> {
    if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows || row_map.len() != input.rows {
        return Err(format!("adaln_modulate_segmented shape 不兼容: input=[{},{}] shift=[{},{}] scale=[{},{}] row_map={}", input.rows, input.cols, shift.rows, shift.cols, scale.rows, scale.cols, row_map.len()));
    }
    let count = validate_u32("adaln_modulate_segmented count", input.len())?;
    let columns = validate_u32("adaln_modulate_segmented columns", input.cols)?;
    let row_map_buf = ctx.shared_buffer(row_map_bytes(row_map));
    let output = ctx.tensor_kernel_output(input.rows, input.cols);
    let shape = format!("input=[{},{}],modulation_rows={}", input.rows, input.cols, shift.rows);
    launch_1d(ctx, "adaln_modulate_segmented_f16", &shape, input.len(), input.buffer.length() + shift.buffer.length() + scale.buffer.length() + row_map_buf.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&shift.buffer), 0);
        encoder.set_buffer(2, Some(&scale.buffer), 0);
        encoder.set_buffer(3, Some(&row_map_buf), 0);
        encoder.set_buffer(4, Some(&output.buffer), 0);
        set_bytes(encoder, 5, &columns);
        set_bytes(encoder, 6, &count);
    })?;
    Ok(output)
}

/// 分段门控残差:`out = residual + update * gate`,gate 按 row_map 广播。
pub fn gated_residual_segmented_tensor(ctx: &MetalContext, residual: &MetalTensor, update: &MetalTensor, gate: &MetalTensor, row_map: &[u32]) -> Result<MetalTensor, String> {
    if update.rows != residual.rows || update.cols != residual.cols || gate.cols != residual.cols || row_map.len() != residual.rows {
        return Err(format!("gated_residual_segmented shape 不兼容: residual=[{},{}] update=[{},{}] gate=[{},{}] row_map={}", residual.rows, residual.cols, update.rows, update.cols, gate.rows, gate.cols, row_map.len()));
    }
    let count = validate_u32("gated_residual_segmented count", residual.len())?;
    let columns = validate_u32("gated_residual_segmented columns", residual.cols)?;
    let row_map_buf = ctx.shared_buffer(row_map_bytes(row_map));
    let output = ctx.tensor_kernel_output(residual.rows, residual.cols);
    let shape = format!("residual=[{},{}],gate_rows={}", residual.rows, residual.cols, gate.rows);
    launch_1d(ctx, "gated_residual_segmented_f16", &shape, residual.len(), residual.buffer.length() + update.buffer.length() + gate.buffer.length() + row_map_buf.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&residual.buffer), 0);
        encoder.set_buffer(1, Some(&update.buffer), 0);
        encoder.set_buffer(2, Some(&gate.buffer), 0);
        encoder.set_buffer(3, Some(&row_map_buf), 0);
        encoder.set_buffer(4, Some(&output.buffer), 0);
        set_bytes(encoder, 5, &columns);
        set_bytes(encoder, 6, &count);
    })?;
    Ok(output)
}

/// 调制切块:`[rows, modalities*chunks*hidden]` → `chunks` 个 `[rows*modalities, hidden]`。
///
/// 源端布局 `in[t][(m*chunks + c)*hidden + h]`,目标 `out[c][(t*modalities + m)*hidden + h]`。
/// 对每个 chunk 单独 launch(每输出一个独立 buffer),语义与 CPU `modulation_chunks` 一致。
pub fn modulation_chunks_tensor(ctx: &MetalContext, input: &MetalTensor, modalities: usize, chunks: usize, hidden: usize) -> Result<Vec<MetalTensor>, String> {
    let input_cols = modalities.checked_mul(chunks).and_then(|n| n.checked_mul(hidden)).ok_or("modulation_chunks 列数溢出")?;
    if input.cols != input_cols {
        return Err(format!("modulation_chunks input cols={} 期望 {input_cols}", input.cols));
    }
    let rows_u32 = validate_u32("modulation_chunks rows", input.rows)?;
    let modalities_u32 = validate_u32("modulation_chunks modalities", modalities)?;
    let hidden_u32 = validate_u32("modulation_chunks hidden", hidden)?;
    let input_cols_u32 = validate_u32("modulation_chunks input_cols", input_cols)?;
    let output_rows = input.rows.checked_mul(modalities).ok_or("modulation_chunks output rows 溢出")?;
    let output_elements = output_rows.checked_mul(hidden).ok_or("modulation_chunks output elements 溢出")?;
    let output_count = validate_u32("modulation_chunks output_elements", output_elements)?;
    let mut outputs = Vec::with_capacity(chunks);
    for chunk in 0..chunks {
        let output = ctx.tensor_kernel_output(output_rows, hidden);
        let chunk_u32 = validate_u32("modulation_chunks chunk", chunk)?;
        let chunks_u32 = validate_u32("modulation_chunks chunks", chunks)?;
        let shape = format!("chunk={chunk}/input=[{},{}],M={modalities},H={hidden}", input.rows, input.cols);
        launch_1d(ctx, "modulation_chunks_f16", &shape, output_elements, input.buffer.length(), output.buffer.length(), |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            set_bytes(encoder, 2, &chunk_u32);
            set_bytes(encoder, 3, &chunks_u32);
            set_bytes(encoder, 4, &modalities_u32);
            set_bytes(encoder, 5, &hidden_u32);
            set_bytes(encoder, 6, &rows_u32);
            set_bytes(encoder, 7, &input_cols_u32);
            set_bytes(encoder, 8, &output_count);
        })?;
        outputs.push(output);
    }
    Ok(outputs)
}

/// 无 mask 全序列 self-attention:Q/K/V 形状 `[rows, head_count*head_dim]`。
///
/// 每 threadgroup 处理一个 (query_row, head):先遍历所有 key 算 score + running max,
/// 再遍历一次算 exp sum(denominator),最后遍历一次加权 V。
/// score_scale = 1/sqrt(head_dim),由调用方传入。
pub fn full_attention_tensor(ctx: &MetalContext, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, head_count: usize, head_dim: usize, score_scale: f32) -> Result<MetalTensor, String> {
    let columns = head_count.checked_mul(head_dim).ok_or("full_attention columns 溢出")?;
    if query.rows != key.rows || query.rows != value.rows || query.cols != columns || key.cols != columns || value.cols != columns {
        return Err(format!("full_attention Q/K/V shape 不兼容: q=[{},{}] k=[{},{}] v=[{},{}] heads={head_count} head_dim={head_dim}", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols));
    }
    if !head_dim.is_power_of_two() {
        return Err(format!("full_attention head_dim={head_dim} 必须是 2 的幂(threadgroup 归约要求)"));
    }
    // tiled kernel 寄存器 query_value[8] 上限 256;非 tiled threadgroup reduction[1024] 上限 1024。
    let head_dim_cap = if query.rows >= 8 { 256 } else { 1024 };
    if head_dim > head_dim_cap {
        return Err(format!("full_attention head_dim={head_dim} 超过 kernel 上限 {head_dim_cap}(rows={})", query.rows));
    }
    // 中等以上 seq 走 tiled kernel(寄存器 online softmax,无 threadgroup_barrier storm);
    // 极小 seq 保留原 1-threadgroup-per-(row,head) kernel 兼容。
    if query.rows >= 8 {
        return full_attention_tiled_tensor(ctx, query, key, value, head_count, head_dim, score_scale);
    }
    let rows_u32 = validate_u32("full_attention rows", query.rows)?;
    let head_count_u32 = validate_u32("full_attention head_count", head_count)?;
    let head_dim_u32 = validate_u32("full_attention head_dim", head_dim)?;
    let output = ctx.tensor_kernel_output(query.rows, columns);
    let shape = format!("rows={},heads={head_count},head_dim={head_dim}", query.rows);
    let pipeline = ctx.pipeline("full_attention_f16")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&key.buffer), 0);
    encoder.set_buffer(2, Some(&value.buffer), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &rows_u32);
    set_bytes(&encoder, 5, &head_count_u32);
    set_bytes(&encoder, 6, &head_dim_u32);
    set_bytes(&encoder, 7, &score_scale);
    // 每 threadgroup 处理一个 (query_row, head),threads = head_dim 做维内点积。
    let groups = MTLSize::new((query.rows * head_count) as u64, 1, 1);
    let threads = MTLSize::new(head_dim as u64, 1, 1);
    encoder.dispatch_thread_groups(groups, threads);
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command, "full_attention_f16", &shape, query.buffer.length() + key.buffer.length() + value.buffer.length(), output.buffer.length());
    Ok(output)
}

/// `full_attention_tiled_f16` 的 dispatch:threads=512(16 SIMD-groups/threadgroup),
/// 每 SIMD-group 处理一个 query_row,grid = head_count × ceil(rows/16)。
fn full_attention_tiled_tensor(ctx: &MetalContext, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, head_count: usize, head_dim: usize, score_scale: f32) -> Result<MetalTensor, String> {
    let columns = head_count.checked_mul(head_dim).ok_or("full_attention_tiled columns 溢出")?;
    if query.rows != key.rows || query.rows != value.rows || query.cols != columns || key.cols != columns || value.cols != columns {
        return Err(format!("full_attention_tiled Q/K/V shape 不兼容: q=[{},{}] k=[{},{}] v=[{},{}] heads={head_count} head_dim={head_dim}", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols));
    }
    let rows_u32 = validate_u32("full_attention_tiled rows", query.rows)?;
    let head_count_u32 = validate_u32("full_attention_tiled head_count", head_count)?;
    let head_dim_u32 = validate_u32("full_attention_tiled head_dim", head_dim)?;
    let output = ctx.tensor_kernel_output(query.rows, columns);
    let shape = format!("rows={},heads={head_count},head_dim={head_dim}", query.rows);
    let pipeline = ctx.pipeline("full_attention_tiled_f16")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&key.buffer), 0);
    encoder.set_buffer(2, Some(&value.buffer), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &rows_u32);
    set_bytes(&encoder, 5, &head_count_u32);
    set_bytes(&encoder, 6, &head_dim_u32);
    set_bytes(&encoder, 7, &score_scale);
    // 每 SIMD-group(32 threads)处理一个 query_row;512 threads = 16 SIMD-groups/threadgroup。
    let query_groups = query.rows.div_ceil(16) as u64;
    let groups = MTLSize::new(head_count as u64, query_groups, 1);
    let threads = MTLSize::new(512, 1, 1);
    encoder.dispatch_thread_groups(groups, threads);
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command, "full_attention_tiled_f16", &shape, query.buffer.length() + key.buffer.length() + value.buffer.length(), output.buffer.length());
    Ok(output)
}
