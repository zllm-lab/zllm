/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: apply_rope_partial_f16, apply_rope_interleaved_partial_f16, apply_rope_partial_f32, apply_rope_interleaved_partial_f32, apply_rope_partial_bf16, apply_rope_interleaved_partial_bf16, apply_rope_prefix_f16, apply_rope_interleaved_prefix_f16, apply_rope_prefix_f32, apply_rope_interleaved_prefix_f32, apply_rope_prefix_bf16, apply_rope_interleaved_prefix_bf16
pub const SHADERS: &str = r#"
inline void apply_rope_f16_impl(
    device const half *input,
    device half *output,
    uint rows,
    uint columns,
    uint head_count,
    uint rotary_dim,
    uint position_offset,
    device const half *cos_data,
    device const half *sin_data,
    uint idx,
    bool interleaved,
    bool partial)
{
    const uint count = rows * columns;
    if (idx >= count) return;
    const uint row = idx / columns;
    const uint column = idx - row * columns;
    const uint head_dim = columns / head_count;
    const uint head = column / head_dim;
    const uint head_column = column - head * head_dim;
    const uint rotary_start = partial ? head_dim - rotary_dim : 0;
    if (head_column < rotary_start || head_column >= rotary_start + rotary_dim) {
        output[idx] = input[idx];
        return;
    }

    const uint component = head_column - rotary_start;
    if (interleaved) {
        const uint pair = component >> 1;
        const uint base = row * columns + head * head_dim + rotary_start + pair * 2;
        const uint angle = (position_offset + row) * (rotary_dim >> 1) + pair;
        const float real = float(input[base]);
        const float imaginary = float(input[base + 1]);
        const float c = float(cos_data[angle]);
        const float s = float(sin_data[angle]);
        output[idx] = (component & 1) == 0 ? half(real * c - imaginary * s) : half(real * s + imaginary * c);
        return;
    }

    const uint half_dim = rotary_dim >> 1;
    const uint pair = component < half_dim ? component : component - half_dim;
    const uint base = row * columns + head * head_dim + rotary_start;
    const uint angle = (position_offset + row) * half_dim + pair;
    const float even = float(input[base + pair]);
    const float odd = float(input[base + half_dim + pair]);
    const float c = float(cos_data[angle]);
    const float s = float(sin_data[angle]);
    output[idx] = component < half_dim ? half(even * c - odd * s) : half(even * s + odd * c);
}

inline void apply_rope_f32_impl(
    device const float *input,
    device float *output,
    uint rows,
    uint columns,
    uint head_count,
    uint rotary_dim,
    uint position_offset,
    device const float *cos_table,
    device const float *sin_table,
    uint idx,
    bool interleaved,
    bool partial)
{
    if (idx >= rows * columns) return;
    const uint row = idx / columns;
    const uint head_dim = columns / head_count;
    const uint head_column = idx % head_dim;
    const uint rotary_start = partial ? head_dim - rotary_dim : 0;
    if (head_column < rotary_start || head_column >= rotary_start + rotary_dim) {
        output[idx] = input[idx];
        return;
    }

    const uint component = head_column - rotary_start;
    const ulong head_base = ulong(idx / head_dim) * head_dim + rotary_start;
    if (interleaved) {
        const uint pair = component >> 1;
        const float real = input[head_base + pair * 2];
        const float imaginary = input[head_base + pair * 2 + 1];
        const float c = cos_table[ulong(position_offset + row) * (rotary_dim >> 1) + pair];
        const float s = sin_table[ulong(position_offset + row) * (rotary_dim >> 1) + pair];
        output[idx] = (component & 1) == 0 ? real * c - imaginary * s : real * s + imaginary * c;
        return;
    }

    const uint half_dim = rotary_dim >> 1;
    const uint pair = component < half_dim ? component : component - half_dim;
    const float even = input[head_base + pair];
    const float odd = input[head_base + half_dim + pair];
    const float c = cos_table[ulong(position_offset + row) * half_dim + pair];
    const float s = sin_table[ulong(position_offset + row) * half_dim + pair];
    output[idx] = component < half_dim ? even * c - odd * s : even * s + odd * c;
}

inline void apply_rope_bf16_impl(
    device const ushort *input,
    device ushort *output,
    uint rows,
    uint columns,
    uint head_count,
    uint rotary_dim,
    uint position_offset,
    device const half *cos_data,
    device const half *sin_data,
    uint idx,
    bool interleaved,
    bool partial)
{
    const uint count = rows * columns;
    if (idx >= count) return;
    const uint row = idx / columns;
    const uint column = idx - row * columns;
    const uint head_dim = columns / head_count;
    const uint head = column / head_dim;
    const uint head_column = column - head * head_dim;
    const uint rotary_start = partial ? head_dim - rotary_dim : 0;
    if (head_column < rotary_start || head_column >= rotary_start + rotary_dim) {
        output[idx] = input[idx];
        return;
    }

    const uint component = head_column - rotary_start;
    if (interleaved) {
        const uint pair = component >> 1;
        const uint base = row * columns + head * head_dim + rotary_start + pair * 2;
        const uint angle = (position_offset + row) * (rotary_dim >> 1) + pair;
        const float real = zllm_bf16_to_f32(input[base]);
        const float imaginary = zllm_bf16_to_f32(input[base + 1]);
        const float c = float(cos_data[angle]);
        const float s = float(sin_data[angle]);
        output[idx] = zllm_f32_to_bf16((component & 1) == 0 ? real * c - imaginary * s : real * s + imaginary * c);
        return;
    }

    const uint half_dim = rotary_dim >> 1;
    const uint pair = component < half_dim ? component : component - half_dim;
    const uint base = row * columns + head * head_dim + rotary_start;
    const uint angle = (position_offset + row) * half_dim + pair;
    const float even = zllm_bf16_to_f32(input[base + pair]);
    const float odd = zllm_bf16_to_f32(input[base + half_dim + pair]);
    const float c = float(cos_data[angle]);
    const float s = float(sin_data[angle]);
    output[idx] = zllm_f32_to_bf16(component < half_dim ? even * c - odd * s : even * s + odd * c);
}

kernel void apply_rope_partial_f16(
    device const half *input [[buffer(0)]], device half *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, idx, false, true);
}

kernel void apply_rope_interleaved_partial_f16(
    device const half *input [[buffer(0)]], device half *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, idx, true, true);
}

kernel void apply_rope_prefix_f16(
    device const half *input [[buffer(0)]], device half *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, idx, false, false);
}

// 多行(verify 重放)变体:position 基址经 decode_state[0],行 r 的 position = base+r。
kernel void apply_rope_prefix_position_f16(
    device const half *input [[buffer(0)]], device half *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint *decode_state [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f16_impl(input, output, rows, columns, head_count, rotary_dim, decode_state[0], cos_data, sin_data, idx, false, false);
}

// Interleaved 布局(MiniCPM5/llama 系)的 decode 重放变体:position 由 decode_state[0] 提供。
kernel void apply_rope_interleaved_prefix_position_f16(
    device const half *input [[buffer(0)]], device half *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint *decode_state [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f16_impl(input, output, rows, columns, head_count, rotary_dim, decode_state[0], cos_data, sin_data, idx, true, false);
}

kernel void apply_rope_interleaved_prefix_f16(
    device const half *input [[buffer(0)]], device half *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, idx, true, false);
}

kernel void apply_rope_partial_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const float *cos_table [[buffer(7)]],
    device const float *sin_table [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f32_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_table, sin_table, idx, false, true);
}

kernel void apply_rope_interleaved_partial_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const float *cos_table [[buffer(7)]],
    device const float *sin_table [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f32_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_table, sin_table, idx, true, true);
}

kernel void apply_rope_prefix_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const float *cos_table [[buffer(7)]],
    device const float *sin_table [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f32_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_table, sin_table, idx, false, false);
}

kernel void apply_rope_interleaved_prefix_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const float *cos_table [[buffer(7)]],
    device const float *sin_table [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_f32_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_table, sin_table, idx, true, false);
}

kernel void apply_rope_partial_bf16(
    device const ushort *input [[buffer(0)]], device ushort *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_bf16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, idx, false, true);
}

kernel void apply_rope_interleaved_partial_bf16(
    device const ushort *input [[buffer(0)]], device ushort *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_bf16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, idx, true, true);
}

kernel void apply_rope_prefix_bf16(
    device const ushort *input [[buffer(0)]], device ushort *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_bf16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, idx, false, false);
}

kernel void apply_rope_interleaved_prefix_bf16(
    device const ushort *input [[buffer(0)]], device ushort *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &head_count [[buffer(4)]], constant uint &rotary_dim [[buffer(5)]],
    constant uint &position_offset [[buffer(6)]], device const half *cos_data [[buffer(7)]],
    device const half *sin_data [[buffer(8)]], uint idx [[thread_position_in_grid]])
{
    apply_rope_bf16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, idx, true, false);
}

// 设备端行 gather:行号来自 GPU buffer(argmax 结果),decode 流水线的 embedding 查表
// 不需要 CPU 同步读回 token id。
kernel void gather_row_f16(
    device const uint *token_id [[buffer(0)]],
    device const half *matrix [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= columns) return;
    output[idx] = matrix[ulong(token_id[0]) * columns + idx];
}

// decode 单行 Q/K RoPE 合并:Q(query_columns)与 K(key_columns)同一 dispatch,
// interleaved + 全维 rotary(rotary_dim == head_dim),cos/sin 是该 position 的行。
// 逐元素数学与 apply_rope_interleaved_prefix_f16 完全一致。
kernel void apply_rope_qk_interleaved_prefix_f16(
    device const half *query [[buffer(0)]],
    device half *query_output [[buffer(1)]],
    device const half *key [[buffer(2)]],
    device half *key_output [[buffer(3)]],
    constant uint &query_columns [[buffer(4)]],
    constant uint &key_columns [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    device const half *cos_table [[buffer(7)]],
    device const half *sin_table [[buffer(8)]],
    constant uint &position [[buffer(9)]],
    uint idx [[thread_position_in_grid]])
{
    device const half *input = query;
    device half *output = query_output;
    uint column = idx;
    if (idx >= query_columns) {
        if (idx >= query_columns + key_columns) return;
        input = key;
        output = key_output;
        column = idx - query_columns;
    }
    const uint head_column = column - (column / head_dim) * head_dim;
    const uint pair = head_column >> 1;
    const uint angle = position * (head_dim >> 1) + pair;
    const float c = float(cos_table[angle]);
    const float s = float(sin_table[angle]);
    const uint base = column - (head_column & 1);
    const float real = float(input[base]);
    const float imaginary = float(input[base + 1]);
    output[column] = (head_column & 1) == 0 ? half(real * c - imaginary * s) : half(real * s + imaginary * c);
}
"#;

use super::{MetalContext, MetalTensor, MetalTensorDType, as_bytes, f32_to_f16, launch_1d, metal, set_bytes, to_f32_tensor, validate_u32};
use crate::attention::rope::RotaryLayout;

enum RopeApplyKind {
    Partial,
    Prefix,
}

fn select_rope_output(ctx: &MetalContext, rows: usize, cols: usize, dtype: MetalTensorDType) -> MetalTensor {
    match dtype {
        MetalTensorDType::F16 => ctx.tensor_zeros(rows, cols),
        MetalTensorDType::Bf16 => ctx.tensor_zeros_bf16(rows, cols),
        MetalTensorDType::F32 => ctx.tensor_zeros_f32(rows, cols),
    }
}

fn select_rope_pipeline(dtype: MetalTensorDType, layout: RotaryLayout, kind: RopeApplyKind) -> &'static str {
    // 12 种组合全部 &'static str,避免每次 RoPE 调用新建 String。
    match (layout, kind, dtype) {
        (RotaryLayout::SplitHalf, RopeApplyKind::Partial, MetalTensorDType::F16) => "apply_rope_partial_f16",
        (RotaryLayout::SplitHalf, RopeApplyKind::Partial, MetalTensorDType::Bf16) => "apply_rope_partial_bf16",
        (RotaryLayout::SplitHalf, RopeApplyKind::Partial, MetalTensorDType::F32) => "apply_rope_partial_f32",
        (RotaryLayout::SplitHalf, RopeApplyKind::Prefix, MetalTensorDType::F16) => "apply_rope_prefix_f16",
        (RotaryLayout::SplitHalf, RopeApplyKind::Prefix, MetalTensorDType::Bf16) => "apply_rope_prefix_bf16",
        (RotaryLayout::SplitHalf, RopeApplyKind::Prefix, MetalTensorDType::F32) => "apply_rope_prefix_f32",
        (RotaryLayout::Interleaved, RopeApplyKind::Partial, MetalTensorDType::F16) => "apply_rope_interleaved_partial_f16",
        (RotaryLayout::Interleaved, RopeApplyKind::Partial, MetalTensorDType::Bf16) => "apply_rope_interleaved_partial_bf16",
        (RotaryLayout::Interleaved, RopeApplyKind::Partial, MetalTensorDType::F32) => "apply_rope_interleaved_partial_f32",
        (RotaryLayout::Interleaved, RopeApplyKind::Prefix, MetalTensorDType::F16) => "apply_rope_interleaved_prefix_f16",
        (RotaryLayout::Interleaved, RopeApplyKind::Prefix, MetalTensorDType::Bf16) => "apply_rope_interleaved_prefix_bf16",
        (RotaryLayout::Interleaved, RopeApplyKind::Prefix, MetalTensorDType::F32) => "apply_rope_interleaved_prefix_f32",
    }
}

/// 返回 (cos, sin, table_offset):kernel 按 `(table_offset + row) * half_dim` 索引。
/// F32/多行走"CPU 切段上传、offset=0";F16/Bf16 单行走"全表常驻、offset=position"
/// (见 [`MetalContext::decode_rope_table_buffers`]),消除逐 token 的 CPU 覆写。
fn prepare_rope_tables(ctx: &MetalContext, x: &MetalTensor, position_offset: usize, half_dim: usize, cos: &[f32], sin: &[f32]) -> Result<(metal::Buffer, metal::Buffer, u32), String> {
    let needed = (position_offset + x.rows) * half_dim;
    if cos.len() < needed || sin.len() < needed {
        return Err("cos/sin 长度不足".to_owned());
    }
    if x.dtype == MetalTensorDType::F32 {
        Ok((ctx.shared_buffer(as_bytes(&cos[position_offset * half_dim..needed])), ctx.shared_buffer(as_bytes(&sin[position_offset * half_dim..needed])), 0))
    } else if x.rows == 1 {
        let (cos_table, sin_table) = ctx.decode_rope_table_buffers(cos, sin, half_dim)?;
        let offset = u32::try_from(position_offset).map_err(|_| format!("RoPE 全表偏移 {position_offset} 超出 u32"))?;
        Ok((cos_table, sin_table, offset))
    } else {
        Ok((f32_to_f16(ctx, &cos[position_offset * half_dim..needed]), f32_to_f16(ctx, &sin[position_offset * half_dim..needed]), 0))
    }
}

fn apply_rope_tensor(ctx: &MetalContext, x: &MetalTensor, head_count: usize, rotary_dim: usize, layout: RotaryLayout, position_offset: usize, cos: &[f32], sin: &[f32], kind: RopeApplyKind) -> Result<MetalTensor, String> {
    if head_count == 0 || !x.cols.is_multiple_of(head_count) || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > x.cols / head_count {
        return Err(match kind {
            RopeApplyKind::Partial => "MetalTensor RoPE shape 非法".to_owned(),
            RopeApplyKind::Prefix => "MetalTensor prefix RoPE shape 非法".to_owned(),
        });
    }
    let half_dim = rotary_dim / 2;
    let (cos, sin, table_offset) = prepare_rope_tables(ctx, x, position_offset, half_dim, cos, sin)?;
    let output = select_rope_output(ctx, x.rows, x.cols, x.dtype);
    let pipeline = select_rope_pipeline(x.dtype, layout, kind);
    let rows = validate_u32("rows", x.rows)?;
    let columns = validate_u32("columns", x.cols)?;
    let head_count = validate_u32("head_count", head_count)?;
    let rotary_dim = validate_u32("rotary_dim", rotary_dim)?;
    let shape = format!("input=[{rows},{columns}],heads={head_count},rotary_dim={rotary_dim}");
    launch_1d(ctx, &pipeline, &shape, x.len(), x.buffer.length() + cos.length() + sin.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(&encoder, 2, &rows);
        set_bytes(&encoder, 3, &columns);
        set_bytes(&encoder, 4, &head_count);
        set_bytes(&encoder, 5, &rotary_dim);
        set_bytes(&encoder, 6, &table_offset);
        encoder.set_buffer(7, Some(&cos), 0);
        encoder.set_buffer(8, Some(&sin), 0);
    })?;
    Ok(output)
}

pub fn split_columns_tensor(ctx: &MetalContext, input: &MetalTensor, left_columns: usize) -> Result<(MetalTensor, MetalTensor), String> {
    if left_columns == 0 || left_columns >= input.cols {
        return Err(format!("split columns 非法: left={left_columns}, total={}", input.cols));
    }
    let right_columns = input.cols - left_columns;
    let (left, right) = if input.dtype == MetalTensorDType::Bf16 {
        (ctx.tensor_zeros_bf16(input.rows, left_columns), ctx.tensor_zeros_bf16(input.rows, right_columns))
    } else {
        (ctx.tensor_zeros(input.rows, left_columns), ctx.tensor_zeros(input.rows, right_columns))
    };
    let columns = validate_u32("columns", input.cols)?;
    let left_u32 = validate_u32("left_columns", left_columns)?;
    let count = validate_u32("count", input.len())?;
    let shape = format!("input=[{},{}],left={left_columns}", input.rows, input.cols);
    launch_1d(ctx, "split_columns_f16", &shape, input.len(), input.buffer.length(), left.buffer.length() + right.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&left.buffer), 0);
        encoder.set_buffer(2, Some(&right.buffer), 0);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &left_u32);
        set_bytes(encoder, 5, &count);
    })?;
    Ok((left, right))
}

/// 池化版列切分:left 按槽位池化(同尺寸多切片并存),right 按尺寸池化(剥皮尾部长度天然互异)。
/// kernel 完整写出两侧,复用安全由同 queue 编码顺序保证。
pub fn split_columns_pooled_tensor(ctx: &MetalContext, input: &MetalTensor, left_columns: usize, slot: usize) -> Result<(MetalTensor, MetalTensor), String> {
    if left_columns == 0 || left_columns >= input.cols {
        return Err(format!("split columns 非法: left={left_columns}, total={}", input.cols));
    }
    let right_columns = input.cols - left_columns;
    let (left, right) = if input.dtype == MetalTensorDType::Bf16 {
        (ctx.tensor_pooled_bf16("split_pooled_bf16", input.rows, left_columns), ctx.tensor_pooled_bf16("split_pooled_bf16", input.rows, right_columns))
    } else {
        (ctx.tensor_pooled_slot("split_pooled_left_f16", slot, input.rows, left_columns), ctx.tensor_pooled("split_pooled_tail_f16", input.rows, right_columns))
    };
    let columns = validate_u32("columns", input.cols)?;
    let left_u32 = validate_u32("left_columns", left_columns)?;
    let count = validate_u32("count", input.len())?;
    let shape = format!("input=[{},{}],left={left_columns}", input.rows, input.cols);
    launch_1d(ctx, "split_columns_f16", &shape, input.len(), input.buffer.length(), left.buffer.length() + right.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&left.buffer), 0);
        encoder.set_buffer(2, Some(&right.buffer), 0);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &left_u32);
        set_bytes(encoder, 5, &count);
    })?;
    Ok((left, right))
}

pub fn split_interleaved_columns_tensor(ctx: &MetalContext, input: &MetalTensor, block_columns: usize) -> Result<(MetalTensor, MetalTensor), String> {
    let pair_columns = block_columns.checked_mul(2).ok_or("interleaved split block 溢出")?;
    if block_columns == 0 || !input.cols.is_multiple_of(pair_columns) {
        return Err(format!("interleaved split cols={} block={block_columns} 非法", input.cols));
    }
    let output_columns = input.cols / 2;
    let (left, right, kernel) = if input.dtype == MetalTensorDType::F32 {
        (ctx.tensor_kernel_output_f32(input.rows, output_columns), ctx.tensor_kernel_output_f32(input.rows, output_columns), "split_interleaved_columns_f32")
    } else {
        (ctx.tensor_kernel_output(input.rows, output_columns), ctx.tensor_kernel_output(input.rows, output_columns), "split_interleaved_columns_f16")
    };
    let input_columns = validate_u32("interleaved split input columns", input.cols)?;
    let block_columns = validate_u32("interleaved split block columns", block_columns)?;
    let count = validate_u32("interleaved split count", left.len())?;
    let shape = format!("input=[{},{}],block={block_columns}", input.rows, input.cols);
    launch_1d(ctx, kernel, &shape, left.len(), input.buffer.length(), left.buffer.length() + right.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&left.buffer), 0);
        encoder.set_buffer(2, Some(&right.buffer), 0);
        set_bytes(encoder, 3, &input_columns);
        set_bytes(encoder, 4, &block_columns);
        set_bytes(encoder, 5, &count);
    })?;
    Ok((left, right))
}

pub fn concat_columns_tensor(ctx: &MetalContext, left: &MetalTensor, right: &MetalTensor) -> Result<MetalTensor, String> {
    if left.rows != right.rows {
        return Err(format!("concat columns rows {} 与 {} 不一致", left.rows, right.rows));
    }
    let columns = left.cols.checked_add(right.cols).ok_or("concat columns 溢出")?;
    if left.dtype != right.dtype {
        return concat_columns_tensor(ctx, &to_f32_tensor(ctx, left)?, &to_f32_tensor(ctx, right)?);
    }
    let output = match left.dtype {
        MetalTensorDType::F16 => ctx.tensor_zeros(left.rows, columns),
        MetalTensorDType::Bf16 => ctx.tensor_zeros_bf16(left.rows, columns),
        MetalTensorDType::F32 => ctx.tensor_zeros_f32(left.rows, columns),
    };
    let left_columns = validate_u32("concat left columns", left.cols)?;
    let right_columns = validate_u32("concat right columns", right.cols)?;
    let count = validate_u32("concat count", output.len())?;
    let shape = format!("left=[{},{}],right=[{},{}]", left.rows, left.cols, right.rows, right.cols);
    let pipeline = if left.dtype == MetalTensorDType::F32 { "concat_columns_f32" } else { "concat_columns_f16" };
    launch_1d(ctx, pipeline, &shape, output.len(), left.buffer.length() + right.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&left.buffer), 0);
        encoder.set_buffer(1, Some(&right.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &left_columns);
        set_bytes(encoder, 4, &right_columns);
        set_bytes(encoder, 5, &count);
    })?;
    Ok(output)
}

#[cfg(test)]
mod split_interleaved_columns_tests {
    use super::*;

    #[test]
    fn metal_splits_each_interleaved_block() {
        if crate::kernel::metal::metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let input = ctx.tensor_from_f32(&[1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0, 5.0, 6.0, 15.0, 16.0, 7.0, 8.0, 17.0, 18.0], 2, 8).unwrap();
        let (left, right) = split_interleaved_columns_tensor(&ctx, &input, 2).unwrap();
        assert_eq!(ctx.tensor_to_f32(&left), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(ctx.tensor_to_f32(&right), vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);

        let input = ctx.tensor_from_f32_preserve(&[1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0], 1, 8).unwrap();
        let (left, right) = split_interleaved_columns_tensor(&ctx, &input, 2).unwrap();
        assert_eq!(ctx.tensor_to_f32(&left), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ctx.tensor_to_f32(&right), vec![11.0, 12.0, 13.0, 14.0]);
    }
}

#[cfg(test)]
mod concat_columns_tests {
    use super::*;

    #[test]
    fn metal_concat_columns_keeps_row_order() {
        if crate::kernel::metal::metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let left = ctx.tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let right = ctx.tensor_from_f32(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3).unwrap();
        let output = concat_columns_tensor(&ctx, &left, &right).unwrap();
        assert_eq!(ctx.read_f16_to_f32(&output.buffer, output.len()), vec![1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);

        let left = ctx.tensor_from_f32_preserve(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let right = ctx.tensor_from_f32_preserve(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3).unwrap();
        let output = concat_columns_tensor(&ctx, &left, &right).unwrap();
        assert_eq!(ctx.tensor_to_f32(&output), vec![1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);

        let right = ctx.tensor_from_f32(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3).unwrap();
        let output = concat_columns_tensor(&ctx, &left, &right).unwrap();
        assert_eq!(output.dtype, MetalTensorDType::F32);
        assert_eq!(ctx.tensor_to_f32(&output), vec![1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn f32_prefix_rope_reads_f32_tables() {
        let ctx = MetalContext::new_default().unwrap();
        let input = ctx.tensor_from_f32_preserve(&[1.0, 2.0, 3.0, 4.0], 1, 4).unwrap();
        let output = apply_rope_prefix_tensor(&ctx, &input, 1, 4, crate::attention::rope::RotaryLayout::SplitHalf, 0, &[0.0, 1.0], &[1.0, 0.0]).unwrap();
        assert_eq!(output.dtype, MetalTensorDType::F32);
        assert_eq!(ctx.tensor_to_f32(&output), vec![-3.0, 2.0, 1.0, 4.0]);
    }

    #[test]
    fn f32_partial_rope_reads_f32_tables() {
        let ctx = MetalContext::new_default().unwrap();
        let input = ctx.tensor_from_f32_preserve(&[1.0, 2.0, 3.0, 4.0], 1, 4).unwrap();
        let output = apply_rope_partial_tensor(&ctx, &input, 1, 2, crate::attention::rope::RotaryLayout::SplitHalf, 0, &[0.0], &[1.0]).unwrap();
        assert_eq!(output.dtype, MetalTensorDType::F32);
        assert_eq!(ctx.tensor_to_f32(&output), vec![1.0, 2.0, -4.0, 3.0]);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn apply_rope_partial_tensor(ctx: &MetalContext, x: &MetalTensor, head_count: usize, rotary_dim: usize, layout: RotaryLayout, position_offset: usize, cos: &[f32], sin: &[f32]) -> Result<MetalTensor, String> {
    apply_rope_tensor(ctx, x, head_count, rotary_dim, layout, position_offset, cos, sin, RopeApplyKind::Partial)
}

#[allow(clippy::too_many_arguments)]
pub fn apply_rope_prefix_tensor(ctx: &MetalContext, x: &MetalTensor, head_count: usize, rotary_dim: usize, layout: RotaryLayout, position_offset: usize, cos: &[f32], sin: &[f32]) -> Result<MetalTensor, String> {
    apply_rope_tensor(ctx, x, head_count, rotary_dim, layout, position_offset, cos, sin, RopeApplyKind::Prefix)
}

/// 设备端行 gather:按 `token_id`(设备 u32 buffer)从 F16 常驻矩阵抠一行。
pub fn gather_row_f16_tensor(ctx: &MetalContext, token_id: &metal::Buffer, matrix: &MetalTensor) -> Result<MetalTensor, String> {
    gather_row_f16_tensor_offset(ctx, token_id, 0, matrix)
}

/// `id_offset` 是 token id 在 buffer 中的字节偏移(异步流水线 per-position 读回区)。
pub fn gather_row_f16_tensor_offset(ctx: &MetalContext, token_id: &metal::Buffer, id_offset: u64, matrix: &MetalTensor) -> Result<MetalTensor, String> {
    if matrix.dtype != MetalTensorDType::F16 || matrix.rows == 0 {
        return Err(format!("gather F16 矩阵 dtype/shape 不符: {:?} [{},{}]", matrix.dtype, matrix.rows, matrix.cols));
    }
    let output = ctx.tensor_uninit(1, matrix.cols);
    let columns = validate_u32("gather F16 columns", matrix.cols)?;
    launch_1d(ctx, "gather_row_f16", &format!("columns={}", matrix.cols), matrix.cols, matrix.cols as u64 * 2, output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(token_id), id_offset);
        encoder.set_buffer(1, Some(&matrix.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &columns);
    })?;
    Ok(output)
}

/// decode 单行 Q/K RoPE 合并 dispatch:interleaved + 全维 rotary(rotary_dim==head_dim),
/// 逐元素数学与两次 apply_rope_interleaved_prefix_f16 完全一致。
#[allow(clippy::too_many_arguments)]
/// Decode 单行 Q/K 融合 RoPE(Interleaved,全维 rotary)。cos/sin 是**常驻 GPU 的
/// 全表 F16**(见 [`MetalContext::decode_rope_table_buffers`]),kernel 按 position
/// 自行索引;每 token CPU 不再切片/转换/上传 rope 数据,position 经 set_bytes
/// 值拷贝进 command buffer,无 CPU 写 GPU buffer 的竞态窗口。
pub fn apply_rope_qk_interleaved_prefix_f16_tensor(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &MetalTensor,
    num_heads: usize,
    num_kv_heads: usize,
    position: usize,
    cos_table: &metal::Buffer,
    sin_table: &metal::Buffer,
    half_dim: usize,
) -> Result<(MetalTensor, MetalTensor), String> {
    if query.dtype != MetalTensorDType::F16 || key.dtype != MetalTensorDType::F16 || query.rows != 1 || key.rows != 1 {
        return Err(format!("QK RoPE 合并只支持单行 F16，实际 Q={:?}[{},{}] K={:?}[{},{}]", query.dtype, query.rows, query.cols, key.dtype, key.rows, key.cols));
    }
    if num_heads == 0 || num_kv_heads == 0 || !query.cols.is_multiple_of(num_heads) || !key.cols.is_multiple_of(num_kv_heads) {
        return Err(format!("QK RoPE 合并 heads 非法: q={}x{num_heads} k={}x{num_kv_heads}", query.cols, key.cols));
    }
    let head_dim = query.cols / num_heads;
    if head_dim == 0 || !head_dim.is_multiple_of(2) || key.cols / num_kv_heads != head_dim || head_dim / 2 != half_dim {
        return Err(format!("QK RoPE 合并 head_dim 不一致: q={} k={} half={half_dim}", head_dim, key.cols / num_kv_heads));
    }
    let position_u32 = u32::try_from(position).map_err(|_| format!("QK RoPE position {position} 超出 u32"))?;
    let needed = position.checked_add(1).and_then(|rows| rows.checked_mul(half_dim)).ok_or("QK RoPE position 偏移溢出")?;
    if cos_table.length() as usize / 2 < needed || sin_table.length() as usize / 2 < needed {
        return Err(format!("QK RoPE 全表长度 cos={} sin={}，position {position} 需要 {needed} 个 f16", cos_table.length() / 2, sin_table.length() / 2));
    }
    let query_output = ctx.tensor_uninit(1, query.cols);
    let key_output = ctx.tensor_uninit(1, key.cols);
    let query_columns = validate_u32("QK RoPE query columns", query.cols)?;
    let key_columns = validate_u32("QK RoPE key columns", key.cols)?;
    let head_dim_u32 = validate_u32("QK RoPE head dim", head_dim)?;
    let total = query.cols + key.cols;
    let row_bytes = (half_dim * 2) as u64;
    let shape = format!("q=[1,{}],k=[1,{}],head_dim={head_dim}", query.cols, key.cols);
    launch_1d(ctx, "apply_rope_qk_interleaved_prefix_f16", &shape, total, query.buffer.length() + key.buffer.length() + row_bytes * 2, query_output.buffer.length() + key_output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&query.buffer), 0);
        encoder.set_buffer(1, Some(&query_output.buffer), 0);
        encoder.set_buffer(2, Some(&key.buffer), 0);
        encoder.set_buffer(3, Some(&key_output.buffer), 0);
        set_bytes(encoder, 4, &query_columns);
        set_bytes(encoder, 5, &key_columns);
        set_bytes(encoder, 6, &head_dim_u32);
        encoder.set_buffer(7, Some(cos_table), 0);
        encoder.set_buffer(8, Some(sin_table), 0);
        set_bytes(encoder, 9, &position_u32);
    })?;
    Ok((query_output, key_output))
}

/// ICB 重放的 decode RoPE:F16 全表常驻 GPU,position 从 decode_state 槽读取,
/// 复用 apply_rope_prefix_f16 kernel(其 constant uint &position_offset 绑定到槽首)。
/// 每 token CPU 不再切片/转换 cos/sin 行,rope 数据一次上传后不离开 GPU。
#[allow(clippy::too_many_arguments)]
/// 多行(verify 重放)RoPE:position 基址经 state 槽,行 r 的 position = base+r。
pub fn apply_rope_rows_position_tensor(
    ctx: &MetalContext,
    x: &MetalTensor,
    head_count: usize,
    rotary_dim: usize,
    layout: crate::attention::rope::RotaryLayout,
    cos_f16: &metal::Buffer,
    sin_f16: &metal::Buffer,
    max_positions: usize,
    decode_state: &metal::Buffer,
    state_offset: u64,
) -> Result<MetalTensor, String> {
    if x.dtype != MetalTensorDType::F16 || x.rows < 1 || layout != crate::attention::rope::RotaryLayout::SplitHalf {
        return Err(format!("verify RoPE position 只支持非空 F16 SplitHalf，实际 {:?}[{},{}]", x.dtype, x.rows, x.cols));
    }
    if head_count == 0 || !x.cols.is_multiple_of(head_count) || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > x.cols / head_count {
        return Err("verify RoPE shape 非法".to_owned());
    }
    let half_dim = rotary_dim / 2;
    let needed = max_positions * half_dim;
    if (cos_f16.length() as usize) < needed * 2 || (sin_f16.length() as usize) < needed * 2 {
        return Err(format!("verify RoPE 全表长度 cos={} sin={}，需要 {needed} 个 f16", cos_f16.length() / 2, sin_f16.length() / 2));
    }
    let output = ctx.tensor_zeros(x.rows, x.cols);
    let rows = validate_u32("verify rope rows", x.rows)?;
    let columns = validate_u32("verify rope columns", x.cols)?;
    let head_count = validate_u32("verify rope head_count", head_count)?;
    let rotary_dim = validate_u32("verify rope rotary_dim", rotary_dim)?;
    let shape = format!("rows={rows},columns={columns},heads={head_count},rotary_dim={rotary_dim}");
    launch_1d(ctx, "apply_rope_prefix_position_f16", &shape, x.len(), x.buffer.length() + cos_f16.length() + sin_f16.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &rows);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &head_count);
        set_bytes(encoder, 5, &rotary_dim);
        encoder.set_buffer(6, Some(decode_state), state_offset);
        encoder.set_buffer(7, Some(cos_f16), 0);
        encoder.set_buffer(8, Some(sin_f16), 0);
    })?;
    Ok(output)
}

pub fn apply_rope_position_tensor(
    ctx: &MetalContext,
    x: &MetalTensor,
    head_count: usize,
    rotary_dim: usize,
    layout: crate::attention::rope::RotaryLayout,
    cos_f16: &metal::Buffer,
    sin_f16: &metal::Buffer,
    max_positions: usize,
    decode_state: &metal::Buffer,
    state_offset: u64,
) -> Result<MetalTensor, String> {
    if x.dtype != MetalTensorDType::F16 || x.rows != 1 || layout != crate::attention::rope::RotaryLayout::SplitHalf {
        return Err(format!("decode RoPE position 只支持单行 F16 SplitHalf，实际 {:?}[{},{}]", x.dtype, x.rows, x.cols));
    }
    rope_position_launch(ctx, x, head_count, rotary_dim, cos_f16, sin_f16, max_positions, decode_state, state_offset, "apply_rope_prefix_f16")
}

/// Interleaved 布局(MiniCPM5)的 decode 重放 RoPE:position 从 `decode_state[state_offset..]` 读取,
/// 命令表与 position 无关。
pub fn apply_rope_interleaved_position_tensor(
    ctx: &MetalContext,
    x: &MetalTensor,
    head_count: usize,
    rotary_dim: usize,
    cos_f16: &metal::Buffer,
    sin_f16: &metal::Buffer,
    max_positions: usize,
    decode_state: &metal::Buffer,
    state_offset: u64,
) -> Result<MetalTensor, String> {
    if x.dtype != MetalTensorDType::F16 || x.rows != 1 {
        return Err(format!("Interleaved decode RoPE position 只支持单行 F16，实际 {:?}[{},{}]", x.dtype, x.rows, x.cols));
    }
    rope_position_launch(ctx, x, head_count, rotary_dim, cos_f16, sin_f16, max_positions, decode_state, state_offset, "apply_rope_interleaved_prefix_position_f16")
}

fn rope_position_launch(
    ctx: &MetalContext,
    x: &MetalTensor,
    head_count: usize,
    rotary_dim: usize,
    cos_f16: &metal::Buffer,
    sin_f16: &metal::Buffer,
    max_positions: usize,
    decode_state: &metal::Buffer,
    state_offset: u64,
    pipeline_name: &str,
) -> Result<MetalTensor, String> {
    if head_count == 0 || !x.cols.is_multiple_of(head_count) || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > x.cols / head_count {
        return Err("decode RoPE position shape 非法".to_owned());
    }
    let half_dim = rotary_dim / 2;
    let needed = max_positions * half_dim;
    if (cos_f16.length() as usize) < needed * 2 || (sin_f16.length() as usize) < needed * 2 {
        return Err(format!("decode RoPE position 全表长度 cos={} sin={}，需要 {needed} 个 f16", cos_f16.length() / 2, sin_f16.length() / 2));
    }
    let output = ctx.tensor_zeros(x.rows, x.cols);
    let rows = 1u32;
    let columns = validate_u32("decode rope columns", x.cols)?;
    let head_count = validate_u32("decode rope head_count", head_count)?;
    let rotary_dim = validate_u32("decode rope rotary_dim", rotary_dim)?;
    let shape = format!("columns={columns},heads={head_count},rotary_dim={rotary_dim}");
    launch_1d(ctx, pipeline_name, &shape, x.len(), x.buffer.length() + cos_f16.length() + sin_f16.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &rows);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &head_count);
        set_bytes(encoder, 5, &rotary_dim);
        encoder.set_buffer(6, Some(decode_state), state_offset);
        encoder.set_buffer(7, Some(cos_f16), 0);
        encoder.set_buffer(8, Some(sin_f16), 0);
    })?;
    Ok(output)
}

#[cfg(test)]
mod position_tests {
    use super::*;
    use half::f16;

    /// decode RoPE position 版(F16 全表常驻 + uniform position)与 legacy 切片行对拍。
    #[test]
    fn rope_position_matches_prefix_row() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let (heads, head_dim, rotary_dim) = (4usize, 64usize, 32usize);
        let half_dim = rotary_dim / 2;
        let max_positions = 16usize;
        let position = 7usize;
        let mut rng: u32 = 4242;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0 - 1.0) * 0.5
        };
        let cos: Vec<f32> = (0..max_positions * half_dim).map(|index| ((index as f32 * 0.13) % std::f32::consts::TAU).cos()).collect();
        let sin: Vec<f32> = (0..max_positions * half_dim).map(|index| ((index as f32 * 0.17) % std::f32::consts::TAU).sin()).collect();
        let input: Vec<f32> = (0..heads * head_dim).map(|_| next()).collect();
        let ctx = MetalContext::new_default().unwrap();
        let x = ctx.tensor_from_f32(&input, 1, heads * head_dim).unwrap();
        let legacy = apply_rope_prefix_tensor(&ctx, &x, heads, rotary_dim, crate::attention::rope::RotaryLayout::SplitHalf, position, &cos, &sin).unwrap();

        let f16_bytes = |values: &[f16]| unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2) };
        let cos_f16: Vec<f16> = cos.iter().map(|&value| f16::from_f32(value)).collect();
        let sin_f16: Vec<f16> = sin.iter().map(|&value| f16::from_f32(value)).collect();
        let cos_buffer = ctx.shared_buffer(f16_bytes(&cos_f16));
        let sin_buffer = ctx.shared_buffer(f16_bytes(&sin_f16));
        let state = [position as u32];
        let state_buffer = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(state.as_ptr().cast::<u8>(), 4) });
        let replay = apply_rope_position_tensor(&ctx, &x, heads, rotary_dim, crate::attention::rope::RotaryLayout::SplitHalf, &cos_buffer, &sin_buffer, max_positions, &state_buffer, 0).unwrap();

        let legacy = ctx.read_f16_to_f32(&legacy.buffer, heads * head_dim);
        let replay = ctx.read_f16_to_f32(&replay.buffer, heads * head_dim);
        for (index, (left, right)) in legacy.iter().zip(&replay).enumerate() {
            assert!((left - right).abs() < 1.0e-3, "rope d={index}: legacy={left} position={right}");
        }
    }

    /// Interleaved decode RoPE position 版(MiniCPM5 重放用)与 legacy 切片行对拍。
    #[test]
    fn rope_interleaved_position_matches_prefix_row() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let (heads, head_dim, rotary_dim) = (4usize, 64usize, 64usize);
        let half_dim = rotary_dim / 2;
        let max_positions = 16usize;
        let position = 5usize;
        let mut rng: u32 = 1717;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng >> 8) as f32 / 8388608.0 - 1.0) * 0.5
        };
        let cos: Vec<f32> = (0..max_positions * half_dim).map(|index| ((index as f32 * 0.11) % std::f32::consts::TAU).cos()).collect();
        let sin: Vec<f32> = (0..max_positions * half_dim).map(|index| ((index as f32 * 0.23) % std::f32::consts::TAU).sin()).collect();
        let input: Vec<f32> = (0..heads * head_dim).map(|_| next()).collect();
        let ctx = MetalContext::new_default().unwrap();
        let x = ctx.tensor_from_f32(&input, 1, heads * head_dim).unwrap();
        let legacy = apply_rope_prefix_tensor(&ctx, &x, heads, rotary_dim, crate::attention::rope::RotaryLayout::Interleaved, position, &cos, &sin).unwrap();

        let f16_bytes = |values: &[f16]| unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2) };
        let cos_f16: Vec<f16> = cos.iter().map(|&value| f16::from_f32(value)).collect();
        let sin_f16: Vec<f16> = sin.iter().map(|&value| f16::from_f32(value)).collect();
        let cos_buffer = ctx.shared_buffer(f16_bytes(&cos_f16));
        let sin_buffer = ctx.shared_buffer(f16_bytes(&sin_f16));
        let state = [position as u32];
        let state_buffer = ctx.shared_buffer(unsafe { std::slice::from_raw_parts(state.as_ptr().cast::<u8>(), 4) });
        let replay = apply_rope_interleaved_position_tensor(&ctx, &x, heads, rotary_dim, &cos_buffer, &sin_buffer, max_positions, &state_buffer, 0).unwrap();

        let legacy = ctx.read_f16_to_f32(&legacy.buffer, heads * head_dim);
        let replay = ctx.read_f16_to_f32(&replay.buffer, heads * head_dim);
        for (index, (left, right)) in legacy.iter().zip(&replay).enumerate() {
            assert!((left - right).abs() < 1.0e-3, "interleaved rope d={index}: legacy={left} position={right}");
        }
    }
}
