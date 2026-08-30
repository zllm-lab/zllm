/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: add_bias_f16, gelu_f16, vision_rope_f16, vision_rope_2d_f16, vision_attention_f16, vision_clamp_f16, vision_quick_gelu_gated_f16, vision_average_pool_f16, scatter_rows_f16, add_bias_bf16_f16
pub const SHADERS: &str = r#"
kernel void add_bias_f16(
    device const half *input [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) output[index] = input[index] + bias[index % columns];
}
kernel void gelu_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const float value = float(input[index]);
    if (value >= 10.0f) {
        output[index] = input[index];
        return;
    }
    if (value <= -10.0f) {
        output[index] = half(0.0h);
        return;
    }
    float cubic = value * value * value;
    output[index] = half(0.5f * value * (1.0f + tanh(0.7978845608028654f * (value + 0.044715f * cubic))));
}
kernel void vision_rope_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *cosine [[buffer(2)]],
    device const half *sine [[buffer(3)]],
    device half *query_output [[buffer(4)]],
    device half *key_output [[buffer(5)]],
    constant uint &head_count [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant uint &rotary_dim [[buffer(8)]],
    constant uint &count [[buffer(9)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const uint dimension = index % head_dim;
    if (dimension >= rotary_dim) {
        query_output[index] = query[index];
        key_output[index] = key[index];
        return;
    }
    const uint row = index / (head_count * head_dim);
    const uint half_dim = rotary_dim / 2;
    const uint paired = dimension < half_dim ? dimension + half_dim : dimension - half_dim;
    const ulong head_base = ulong(index / head_dim) * head_dim;
    const float rotated_query = dimension < half_dim ? -float(query[head_base + paired]) : float(query[head_base + paired]);
    const float rotated_key = dimension < half_dim ? -float(key[head_base + paired]) : float(key[head_base + paired]);
    const ulong rope_index = ulong(row) * rotary_dim + dimension;
    const float cos_value = float(cosine[rope_index]);
    const float sin_value = float(sine[rope_index]);
    query_output[index] = half(float(query[index]) * cos_value + rotated_query * sin_value);
    key_output[index] = half(float(key[index]) * cos_value + rotated_key * sin_value);
}
kernel void vision_rope_2d_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *cosine [[buffer(2)]],
    device const half *sine [[buffer(3)]],
    device half *query_output [[buffer(4)]],
    device half *key_output [[buffer(5)]],
    constant uint &head_count [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant uint &count [[buffer(8)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const uint dimension = index % head_dim;
    const uint axis_dim = head_dim / 2;
    const uint within_axis = dimension % axis_dim;
    const uint axis_half = axis_dim / 2;
    const uint paired_within = within_axis < axis_half ? within_axis + axis_half : within_axis - axis_half;
    const uint paired = (dimension / axis_dim) * axis_dim + paired_within;
    const ulong head_base = ulong(index / head_dim) * head_dim;
    const uint row = index / (head_count * head_dim);
    const ulong rope_index = ulong(row) * head_dim + dimension;
    const float sign = within_axis < axis_half ? -1.0f : 1.0f;
    query_output[index] = half(float(query[index]) * float(cosine[rope_index]) + sign * float(query[head_base + paired]) * float(sine[rope_index]));
    key_output[index] = half(float(key[index]) * float(cosine[rope_index]) + sign * float(key[head_base + paired]) * float(sine[rope_index]));
}
kernel void vision_attention_f16(
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
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]])
{
    constexpr uint query_tile = 4;
    const uint query_block = group / head_count;
    const uint head = group % head_count;
    const uint query_start = query_block * query_tile;
    if (query_start >= rows) return;
    const uint query_count = min(query_tile, rows - query_start);
    threadgroup float reduction[query_tile][32];
    threadgroup float control[query_tile][4];
    float query_values[query_tile];
    float accumulated[query_tile];
    for (uint query_index = 0; query_index < query_count; ++query_index) {
        const ulong query_base = (ulong(query_start + query_index) * head_count + head) * head_dim;
        query_values[query_index] = lane < head_dim ? float(query[query_base + lane]) : 0.0f;
        accumulated[query_index] = 0.0f;
    }

    for (uint token = 0; token < rows; ++token) {
        const ulong key_base = (ulong(token) * head_count + head) * head_dim;
        const float key_value = lane < head_dim ? float(key[key_base + lane]) : 0.0f;
        for (uint query_index = 0; query_index < query_count; ++query_index) {
            const float partial = simd_sum(query_values[query_index] * key_value);
            if (simd_lane == 0) reduction[query_index][simd_group] = partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < query_count) {
            const uint query_index = lane;
            float dot = 0.0f;
            for (uint index = 0; index < simd_groups; ++index) dot += reduction[query_index][index];
            const float score = dot * score_scale;
            if (token == 0) {
                control[query_index][0] = 0.0f;
                control[query_index][1] = 1.0f;
                control[query_index][2] = 1.0f;
                control[query_index][3] = score;
            } else if (score > control[query_index][3]) {
                const float rescale = exp(control[query_index][3] - score);
                control[query_index][0] = rescale;
                control[query_index][1] = 1.0f;
                control[query_index][2] = control[query_index][2] * rescale + 1.0f;
                control[query_index][3] = score;
            } else {
                const float weight = exp(score - control[query_index][3]);
                control[query_index][0] = 1.0f;
                control[query_index][1] = weight;
                control[query_index][2] += weight;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < head_dim) {
            const float value_element = float(value[key_base + lane]);
            for (uint query_index = 0; query_index < query_count; ++query_index) {
                accumulated[query_index] = accumulated[query_index] * control[query_index][0] + control[query_index][1] * value_element;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < head_dim) {
        for (uint query_index = 0; query_index < query_count; ++query_index) {
            const ulong query_base = (ulong(query_start + query_index) * head_count + head) * head_dim;
            output[query_base + lane] = finite_f16(accumulated[query_index] / control[query_index][2]);
        }
    }
}
kernel void vision_clamp_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant float &minimum [[buffer(2)]],
    constant float &maximum [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) output[index] = half(clamp(float(input[index]), minimum, maximum));
}
kernel void vision_quick_gelu_gated_f16(
    device const half *gate [[buffer(0)]],
    device const half *up [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const float value = float(gate[index]);
    output[index] = finite_f16((value / (1.0f + exp(-1.702f * value))) * float(up[index]));
}
kernel void vision_average_pool_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &grid_width [[buffer(2)]],
    constant uint &output_width [[buffer(3)]],
    constant uint &kernel_size [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant float &factor [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const uint column = index % columns;
    const uint output_row = index / columns;
    const uint output_y = output_row / output_width;
    const uint output_x = output_row % output_width;
    float sum = 0.0f;
    for (uint y = 0; y < kernel_size; ++y) {
        for (uint x = 0; x < kernel_size; ++x) {
            const ulong input_row = ulong(output_y * kernel_size + y) * grid_width + output_x * kernel_size + x;
            sum += float(input[input_row * columns + column]);
        }
    }
    output[index] = finite_f16(sum * factor);
}
kernel void scatter_rows_f16(
    device const half *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant uint &start_row [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) destination[ulong(start_row) * columns + index] = source[index];
}
kernel void add_bias_bf16_f16(
    device const ushort *input [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device ushort *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        output[index] = zllm_f32_to_bf16(zllm_bf16_to_f32(input[index]) + float(bias[index % columns]));
    }
}
"#;

use super::dense::validate_tensor;
use super::{MTLSize, MetalContext, MetalTensor, launch_1d, set_bytes, validate_u32};

pub fn vision_rope_tensor(ctx: &MetalContext, query: &MetalTensor, key: &MetalTensor, cos: &MetalTensor, sin: &MetalTensor, head_count: usize) -> Result<(MetalTensor, MetalTensor), String> {
    validate_tensor("vision rope key", key, query.rows, query.cols)?;
    if head_count == 0 || !query.cols.is_multiple_of(head_count) {
        return Err(format!("视觉 RoPE shape 非法: query=[{},{}], heads={head_count}", query.rows, query.cols));
    }
    let head_dim = query.cols / head_count;
    if !head_dim.is_multiple_of(2) {
        return Err(format!("视觉 RoPE head_dim={head_dim} 不是偶数"));
    }
    if cos.rows != query.rows || sin.rows != query.rows || cos.cols != sin.cols || cos.cols == 0 || cos.cols > head_dim || !cos.cols.is_multiple_of(2) {
        return Err(format!("视觉 RoPE 频率 shape 非法: q=[{},{}], cos=[{},{}], sin=[{},{}], head_dim={head_dim}", query.rows, query.cols, cos.rows, cos.cols, sin.rows, sin.cols,));
    }
    let count = validate_u32("vision rope count", query.len())?;
    let heads = validate_u32("vision rope heads", head_count)?;
    let dimension = validate_u32("vision rope head_dim", head_dim)?;
    let rotary_dimension = validate_u32("vision rope rotary_dim", cos.cols)?;
    let query_output = ctx.tensor_zeros(query.rows, query.cols);
    let key_output = ctx.tensor_zeros(key.rows, key.cols);
    let shape = format!("rows={},heads={head_count},dim={head_dim}", query.rows);
    launch_1d(ctx, "vision_rope_f16", &shape, query.len(), query.buffer.length() + key.buffer.length() + cos.buffer.length() + sin.buffer.length(), query_output.buffer.length() + key_output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&query.buffer), 0);
        encoder.set_buffer(1, Some(&key.buffer), 0);
        encoder.set_buffer(2, Some(&cos.buffer), 0);
        encoder.set_buffer(3, Some(&sin.buffer), 0);
        encoder.set_buffer(4, Some(&query_output.buffer), 0);
        encoder.set_buffer(5, Some(&key_output.buffer), 0);
        set_bytes(encoder, 6, &heads);
        set_bytes(encoder, 7, &dimension);
        set_bytes(encoder, 8, &rotary_dimension);
        set_bytes(encoder, 9, &count);
    })?;
    Ok((query_output, key_output))
}

pub fn vision_rope_2d_tensor(ctx: &MetalContext, query: &MetalTensor, key: &MetalTensor, cos: &MetalTensor, sin: &MetalTensor, head_count: usize) -> Result<(MetalTensor, MetalTensor), String> {
    validate_tensor("vision 2d rope key", key, query.rows, query.cols)?;
    if head_count == 0 || !query.cols.is_multiple_of(head_count) {
        return Err(format!("二维视觉 RoPE shape 非法: query=[{},{}], heads={head_count}", query.rows, query.cols));
    }
    let head_dim = query.cols / head_count;
    if !head_dim.is_multiple_of(4) || cos.rows != query.rows || sin.rows != query.rows || cos.cols != head_dim || sin.cols != head_dim {
        return Err(format!("二维视觉 RoPE 频率 shape 非法: q=[{},{}], cos=[{},{}], sin=[{},{}]", query.rows, query.cols, cos.rows, cos.cols, sin.rows, sin.cols));
    }
    let count = validate_u32("vision 2d rope count", query.len())?;
    let heads = validate_u32("vision 2d rope heads", head_count)?;
    let dimension = validate_u32("vision 2d rope head_dim", head_dim)?;
    let query_output = ctx.tensor_zeros(query.rows, query.cols);
    let key_output = ctx.tensor_zeros(key.rows, key.cols);
    let shape = format!("rows={},heads={head_count},dim={head_dim}", query.rows);
    launch_1d(ctx, "vision_rope_2d_f16", &shape, query.len(), query.buffer.length() + key.buffer.length() + cos.buffer.length() + sin.buffer.length(), query_output.buffer.length() + key_output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&query.buffer), 0);
        encoder.set_buffer(1, Some(&key.buffer), 0);
        encoder.set_buffer(2, Some(&cos.buffer), 0);
        encoder.set_buffer(3, Some(&sin.buffer), 0);
        encoder.set_buffer(4, Some(&query_output.buffer), 0);
        encoder.set_buffer(5, Some(&key_output.buffer), 0);
        set_bytes(encoder, 6, &heads);
        set_bytes(encoder, 7, &dimension);
        set_bytes(encoder, 8, &count);
    })?;
    Ok((query_output, key_output))
}

pub fn vision_attention_tensor(ctx: &MetalContext, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, head_count: usize) -> Result<MetalTensor, String> {
    let head_dim = query.cols.checked_div(head_count).unwrap_or(0);
    vision_attention_tensor_scaled(ctx, query, key, value, head_count, 1.0 / (head_dim as f32).sqrt())
}

pub fn vision_attention_tensor_scaled(ctx: &MetalContext, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, head_count: usize, score_scale: f32) -> Result<MetalTensor, String> {
    validate_tensor("vision attention key", key, query.rows, query.cols)?;
    validate_tensor("vision attention value", value, query.rows, query.cols)?;
    if query.rows == 0 || head_count == 0 || !query.cols.is_multiple_of(head_count) {
        return Err(format!("视觉 attention shape 非法: q=[{},{}], heads={head_count}", query.rows, query.cols));
    }
    let head_dim = query.cols / head_count;
    let pipeline = ctx.pipeline("vision_attention_f16")?;
    let simd_width = pipeline.thread_execution_width() as usize;
    if simd_width == 0 {
        return Err("视觉 attention Metal SIMD width 为 0".to_owned());
    }
    let threads = head_dim.div_ceil(simd_width) * simd_width;
    if threads > 1024 || threads as u64 > pipeline.max_total_threads_per_threadgroup() {
        return Err(format!("视觉 attention head_dim={head_dim} 需要 {threads} threads，超过 Metal 上限"));
    }
    let rows = validate_u32("vision attention rows", query.rows)?;
    let heads = validate_u32("vision attention heads", head_count)?;
    let dimension = validate_u32("vision attention head_dim", head_dim)?;
    let output = ctx.tensor_zeros(query.rows, query.cols);
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&key.buffer), 0);
    encoder.set_buffer(2, Some(&value.buffer), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &rows);
    set_bytes(&encoder, 5, &heads);
    set_bytes(&encoder, 6, &dimension);
    set_bytes(&encoder, 7, &score_scale);
    const QUERY_TILE: usize = 4;
    let query_blocks = query.rows.div_ceil(QUERY_TILE);
    encoder.dispatch_thread_groups(MTLSize::new((query_blocks * head_count) as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={},heads={head_count},dim={head_dim}", query.rows);
    ctx.commit_and_wait_profiled(&command, "vision_attention_f16", &shape, query.buffer.length() + key.buffer.length() + value.buffer.length(), output.buffer.length());
    Ok(output)
}

pub fn vision_clamp_tensor(ctx: &MetalContext, input: &MetalTensor, minimum: f32, maximum: f32) -> Result<MetalTensor, String> {
    if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
        return Err(format!("视觉 clamp [{minimum},{maximum}] 非法"));
    }
    let count = validate_u32("vision clamp count", input.len())?;
    let output = ctx.tensor_zeros(input.rows, input.cols);
    launch_1d(ctx, "vision_clamp_f16", &format!("rows={},cols={}", input.rows, input.cols), input.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &minimum);
        set_bytes(encoder, 3, &maximum);
        set_bytes(encoder, 4, &count);
    })?;
    Ok(output)
}

pub fn vision_quick_gelu_gated_tensor(ctx: &MetalContext, gate: &MetalTensor, up: &MetalTensor) -> Result<MetalTensor, String> {
    validate_tensor("vision quick gelu up", up, gate.rows, gate.cols)?;
    let count = validate_u32("vision quick gelu count", gate.len())?;
    let output = ctx.tensor_zeros(gate.rows, gate.cols);
    launch_1d(ctx, "vision_quick_gelu_gated_f16", &format!("rows={},cols={}", gate.rows, gate.cols), gate.len(), gate.buffer.length() + up.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&gate.buffer), 0);
        encoder.set_buffer(1, Some(&up.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &count);
    })?;
    Ok(output)
}

pub fn vision_average_pool_tensor(ctx: &MetalContext, input: &MetalTensor, grid_height: usize, grid_width: usize, kernel_size: usize, output_scale: f32) -> Result<MetalTensor, String> {
    if kernel_size == 0 || input.rows != grid_height.checked_mul(grid_width).ok_or("视觉池化 grid 溢出")? || !grid_height.is_multiple_of(kernel_size) || !grid_width.is_multiple_of(kernel_size) || !output_scale.is_finite() {
        return Err(format!("视觉平均池化 input=[{},{}] grid={grid_height}x{grid_width} kernel={kernel_size} scale={output_scale} 非法", input.rows, input.cols));
    }
    let output_height = grid_height / kernel_size;
    let output_width_value = grid_width / kernel_size;
    let output = ctx.tensor_zeros(output_height * output_width_value, input.cols);
    let count = validate_u32("vision pool count", output.len())?;
    let width = validate_u32("vision pool grid width", grid_width)?;
    let output_width = validate_u32("vision pool output width", output_width_value)?;
    let kernel = validate_u32("vision pool kernel", kernel_size)?;
    let columns = validate_u32("vision pool columns", input.cols)?;
    let factor = output_scale / (kernel_size * kernel_size) as f32;
    launch_1d(ctx, "vision_average_pool_f16", &format!("grid={grid_height}x{grid_width},kernel={kernel_size},cols={}", input.cols), output.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &width);
        set_bytes(encoder, 3, &output_width);
        set_bytes(encoder, 4, &kernel);
        set_bytes(encoder, 5, &columns);
        set_bytes(encoder, 6, &factor);
        set_bytes(encoder, 7, &count);
    })?;
    Ok(output)
}

pub fn scatter_rows_tensor(ctx: &MetalContext, destination: &MetalTensor, start_row: usize, source: &MetalTensor) -> Result<(), String> {
    if destination.cols != source.cols || start_row.checked_add(source.rows).is_none_or(|end| end > destination.rows) {
        return Err(format!("scatter rows shape 非法: destination=[{},{}], start={start_row}, source=[{},{}]", destination.rows, destination.cols, source.rows, source.cols,));
    }
    let count = validate_u32("scatter rows count", source.len())?;
    let columns = validate_u32("scatter rows columns", source.cols)?;
    let start = validate_u32("scatter rows start", start_row)?;
    let shape = format!("destination=[{},{}],start={start_row},rows={}", destination.rows, destination.cols, source.rows);
    launch_1d(ctx, "scatter_rows_f16", &shape, source.len(), source.buffer.length() + destination.buffer.length(), destination.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&source.buffer), 0);
        encoder.set_buffer(1, Some(&destination.buffer), 0);
        set_bytes(encoder, 2, &start);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &count);
    })
}
