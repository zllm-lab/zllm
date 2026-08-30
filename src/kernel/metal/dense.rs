/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: split_columns_f16, split_interleaved_columns_f16, split_interleaved_columns_f32, concat_columns_f16, concat_columns_f32, cast_f32_to_f16, cast_f32_to_f16_strided, cast_f16_to_f32, cast_bf16_to_f32, gated_activation_packed_f16, rms_norm_f16, rms_norm_f32, rms_norm_bf16_in_f16_simd, cast_f32_f16, rms_norm_f32_f16, cast_f32_f16_x4, matmul_f16, silu_mul_f16, gated_activation_f16, sigmoid_gate_f16, sigmoid_gate_f16_f32, linear_sigmoid_gate_f16, add_f16, add_scaled_f16, add_scaled_f32_f16, add_scaled_f16_f32, add_scaled_f32, add_f32_f16, add_f16_f32, add_f32, matmul_tiled_f16, gemv_f16, dequantize_fp8_matrix_f16, matmul_tiled_fp8_f16, matmul_per_tensor_fp8_f16, official_fp8_matmul_f16, official_fp8_dual_matmul_f16, layernorm_bias_f16, layernorm_bias_bf16, layernorm_bias_wide_f16, layernorm_bias_wide_bf16, cast_bf16_f16, cast_f16_bf16, cast_f32_bf16, rms_norm_bf16, gated_activation_bf16, add_bf16, add_scaled_bf16
pub const SHADERS: &str = r#"
kernel void split_columns_f16(
    device const half *input [[buffer(0)]],
    device half *left [[buffer(1)]],
    device half *right [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &left_columns [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint row = gid / columns;
    uint column = gid - row * columns;
    if (column < left_columns) {
        left[ulong(row) * left_columns + column] = input[gid];
    } else {
        uint right_columns = columns - left_columns;
        right[ulong(row) * right_columns + column - left_columns] = input[gid];
    }
}
kernel void split_interleaved_columns_f16(
    device const half *input [[buffer(0)]],
    device half *left [[buffer(1)]],
    device half *right [[buffer(2)]],
    constant uint &input_columns [[buffer(3)]],
    constant uint &block_columns [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint output_columns = input_columns / 2;
    const uint row = gid / output_columns;
    const uint column = gid - row * output_columns;
    const uint block = column / block_columns;
    const uint offset = column - block * block_columns;
    const ulong source = ulong(row) * input_columns + ulong(block) * block_columns * 2 + offset;
    left[gid] = input[source];
    right[gid] = input[source + block_columns];
}
kernel void split_interleaved_columns_f32(
    device const float *input [[buffer(0)]], device float *left [[buffer(1)]], device float *right [[buffer(2)]],
    constant uint &input_columns [[buffer(3)]], constant uint &block_columns [[buffer(4)]],
    constant uint &count [[buffer(5)]], uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    const uint output_columns = input_columns / 2;
    const uint row = id / output_columns;
    const uint column = id - row * output_columns;
    const uint block = column / block_columns;
    const uint offset = column - block * block_columns;
    const ulong source = ulong(row) * input_columns + ulong(block) * block_columns * 2 + offset;
    left[id] = input[source];
    right[id] = input[source + block_columns];
}
kernel void concat_columns_f16(
    device const half *left [[buffer(0)]],
    device const half *right [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &left_columns [[buffer(3)]],
    constant uint &right_columns [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint columns = left_columns + right_columns;
    const uint row = gid / columns;
    const uint column = gid - row * columns;
    output[gid] = column < left_columns
        ? left[ulong(row) * left_columns + column]
        : right[ulong(row) * right_columns + column - left_columns];
}
kernel void concat_columns_f32(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &left_columns [[buffer(3)]],
    constant uint &right_columns [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint columns = left_columns + right_columns;
    const uint row = gid / columns;
    const uint column = gid - row * columns;
    output[gid] = column < left_columns
        ? left[ulong(row) * left_columns + column]
        : right[ulong(row) * right_columns + column - left_columns];
}
kernel void cast_f32_to_f16(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) output[gid] = half(clamp(input[gid], -65504.0f, 65504.0f));
}
kernel void cast_f32_to_f16_strided(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &output_stride [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const uint row = gid / columns;
    const uint column = gid - row * columns;
    if (row < rows) output[ulong(row) * output_stride + column] = half(clamp(input[gid], -65504.0f, 65504.0f));
}
kernel void cast_f16_to_f32(
    device const half *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) output[gid] = float(input[gid]);
}
kernel void cast_bf16_to_f32(
    device const ushort *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) output[gid] = as_type<float>(uint(input[gid]) << 16);
}
kernel void gated_activation_packed_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &activation_kind [[buffer(4)]],
    constant float &activation_alpha [[buffer(5)]],
    constant float &activation_limit [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    const uint row = id / columns;
    const uint column = id - row * columns;
    const ulong base = ulong(row) * columns * 2ul;
    output[id] = finite_f16(gated_activation_value(
        float(input[base + column]),
        float(input[base + columns + column]),
        activation_kind,
        activation_alpha,
        activation_limit));
}
kernel void rms_norm_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float sums[256];
    ulong offset = (ulong)row * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += width) {
        float value = float(input[offset + column]);
        sum += value * value;
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += width) {
        output[offset + column] = half(float(input[offset + column]) * scale * (float(weight[column]) + weight_offset));
    }
}
kernel void rms_norm_f32(
    device const float *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float sums[256];
    const ulong offset = ulong(row) * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += width) {
        const float value = input[offset + column];
        sum += value * value;
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += width) {
        output[offset + column] = input[offset + column] * scale * (float(weight[column]) + weight_offset);
    }
}
kernel void cast_f32_f16(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = finite_f16(input[id]);
}
kernel void rms_norm_f32_f16(
    device const float *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    device half *output_f16 [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float sums[256];
    const ulong offset = ulong(row) * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += width) {
        const float value = input[offset + column];
        sum += value * value;
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += width) {
        const ulong index = offset + column;
        const float value = input[index] * scale * (float(weight[column]) + weight_offset);
        output[index] = value;
        output_f16[index] = finite_f16(value);
    }
}
kernel void rms_norm_f32_weight_f32_f16(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    device half *output_f16 [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float sums[256];
    const ulong offset = ulong(row) * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += width) {
        const float value = input[offset + column];
        sum += value * value;
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += width) {
        const ulong index = offset + column;
        const float value = input[index] * scale * (weight[column] + weight_offset);
        output[index] = value;
        output_f16[index] = finite_f16(value);
    }
}
// 单 SIMD group(32 lanes) + simd_sum + float4 连续段版本:消除 256 线程组
// 的手动归约树 barrier,长序列 prefill 的行数巨大时把带宽打满。
// 要求 columns 是 4 的倍数(行起点保持 float4 对齐)。
kernel void rms_norm_f32_weight_f32_f16_simd(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    device half *output_f16 [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const ulong offset = ulong(row) * columns;
    device const float4 *in4 = (device const float4 *)(input + offset);
    device const float4 *weight4 = (device const float4 *)weight;
    const uint vectors = columns >> 2;
    float sum = 0.0f;
    for (uint v = lane; v < vectors; v += 32) {
        const float4 value = in4[v];
        sum += dot(value, value);
    }
    for (uint column = (vectors << 2) + lane; column < columns; column += 32) {
        const float value = input[offset + column];
        sum += value * value;
    }
    const float scale = rsqrt(simd_sum(sum) / float(columns) + epsilon);
    device float4 *out4 = (device float4 *)(output + offset);
    device half4 *out_f16 = (device half4 *)(output_f16 + offset);
    for (uint v = lane; v < vectors; v += 32) {
        const float4 value = in4[v] * scale * (weight4[v] + weight_offset);
        out4[v] = value;
        out_f16[v] = half4(clamp(value, float4(-65504.0f), float4(65504.0f)));
    }
    for (uint column = (vectors << 2) + lane; column < columns; column += 32) {
        const ulong index = offset + column;
        const float value = input[index] * scale * (weight[column] + weight_offset);
        output[index] = value;
        output_f16[index] = finite_f16(value);
    }
}
// F16 输入 + F32 权重直读,输出仅 F16:逐头 GemmaRMSNorm 的 decode 主路径,
// 消掉 F32 权重路径两侧的 f16↔f32 cast 与多余的 F32 输出写出。
// 列 4 对齐时行起点保持 half4/float4 对齐。
kernel void rms_norm_f16_in_f32_weight_f16_simd(
    device const half *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const ulong offset = ulong(row) * columns;
    device const half4 *in4 = (device const half4 *)(input + offset);
    device const float4 *weight4 = (device const float4 *)weight;
    device half4 *out4 = (device half4 *)(output + offset);
    const uint vectors = columns >> 2;
    float sum = 0.0f;
    for (uint v = lane; v < vectors; v += 32) {
        const float4 value = float4(in4[v]);
        sum += dot(value, value);
    }
    for (uint column = (vectors << 2) + lane; column < columns; column += 32) {
        const float value = float(input[offset + column]);
        sum += value * value;
    }
    const float scale = rsqrt(simd_sum(sum) / float(columns) + epsilon);
    for (uint v = lane; v < vectors; v += 32) {
        const float4 value = float4(in4[v]) * scale * (weight4[v] + weight_offset);
        out4[v] = half4(clamp(value, float4(-65504.0f), float4(65504.0f)));
    }
    for (uint column = (vectors << 2) + lane; column < columns; column += 32) {
        const float value = float(input[offset + column]) * scale * (weight[column] + weight_offset);
        output[offset + column] = finite_f16(value);
    }
}
// F16 输入 + F16 权重的单 SIMD group 版:消除 256 线程树状归约的逐层 barrier,
// 单行 [1,·] 算子从 ~27µs 降到 simd 版的延迟地板。列 4 对齐。
kernel void rms_norm_f16_simd(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const ulong offset = ulong(row) * columns;
    device const half4 *in4 = (device const half4 *)(input + offset);
    device const half4 *weight4 = (device const half4 *)weight;
    device half4 *out4 = (device half4 *)(output + offset);
    const uint vectors = columns >> 2;
    float sum = 0.0f;
    for (uint v = lane; v < vectors; v += 32) {
        const float4 value = float4(in4[v]);
        sum += dot(value, value);
    }
    for (uint column = (vectors << 2) + lane; column < columns; column += 32) {
        const float value = float(input[offset + column]);
        sum += value * value;
    }
    const float scale = rsqrt(simd_sum(sum) / float(columns) + epsilon);
    for (uint v = lane; v < vectors; v += 32) {
        const float4 value = float4(in4[v]) * scale * (float4(weight4[v]) + weight_offset);
        out4[v] = half4(clamp(value, float4(-65504.0f), float4(65504.0f)));
    }
    for (uint column = (vectors << 2) + lane; column < columns; column += 32) {
        const float value = float(input[offset + column]) * scale * (float(weight[column]) + weight_offset);
        output[offset + column] = finite_f16(value);
    }
}
// bf16 输入直读、F16 输出的单 simdgroup 版:混合精度层(8-bit gemv 输出 Bf16)
// 进融合 rmsnorm+gemv 不再需要先整行 cast 到 F16(省一次读写 + 一个 dispatch)。
kernel void rms_norm_bf16_in_f16_simd(
    device const ushort *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const ulong offset = ulong(row) * columns;
    device const ushort4 *in4 = (device const ushort4 *)(input + offset);
    device const half4 *weight4 = (device const half4 *)weight;
    device half4 *out4 = (device half4 *)(output + offset);
    const uint vectors = columns >> 2;
    float sum = 0.0f;
    for (uint v = lane; v < vectors; v += 32) {
        const ushort4 raw = in4[v];
        const float4 value = float4(zllm_bf16_to_f32(raw.x), zllm_bf16_to_f32(raw.y), zllm_bf16_to_f32(raw.z), zllm_bf16_to_f32(raw.w));
        sum += dot(value, value);
    }
    for (uint column = (vectors << 2) + lane; column < columns; column += 32) {
        const float value = zllm_bf16_to_f32(input[offset + column]);
        sum += value * value;
    }
    const float scale = rsqrt(simd_sum(sum) / float(columns) + epsilon);
    for (uint v = lane; v < vectors; v += 32) {
        const ushort4 raw = in4[v];
        const float4 value = float4(zllm_bf16_to_f32(raw.x), zllm_bf16_to_f32(raw.y), zllm_bf16_to_f32(raw.z), zllm_bf16_to_f32(raw.w)) * scale * (float4(weight4[v]) + weight_offset);
        out4[v] = half4(clamp(value, float4(-65504.0f), float4(65504.0f)));
    }
    for (uint column = (vectors << 2) + lane; column < columns; column += 32) {
        const float value = zllm_bf16_to_f32(input[offset + column]) * scale * (float(weight[column]) + weight_offset);
        output[offset + column] = finite_f16(value);
    }
}
kernel void rms_norm_f16_in_f32_weight_f16(
    device const half *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant float &weight_offset [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float sums[256];
    const ulong offset = ulong(row) * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += width) {
        const float value = float(input[offset + column]);
        sum += value * value;
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += width) {
        output[offset + column] = finite_f16(float(input[offset + column]) * scale * (weight[column] + weight_offset));
    }
}
kernel void cast_f32_f16_x4(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    const uint begin = id * 4;
    if (begin >= count) return;
    if (begin + 4 <= count) {
        const float4 values = *reinterpret_cast<device const float4 *>(input + begin);
        *reinterpret_cast<device half4 *>(output + begin) = half4(clamp(values, float4(-65504.0f), float4(65504.0f)));
        return;
    }
    for (uint item = begin; item < count; ++item) output[item] = finite_f16(input[item]);
}
kernel void matmul_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &in_cols [[buffer(4)]],
    constant uint &out_cols [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = rows * out_cols;
    if (idx >= total) return;

    uint row = idx / out_cols;
    uint col = idx - row * out_cols;
    uint x_base = row * in_cols;
    uint w_base = col * in_cols;

    float acc = 0.0f;
    for (uint i = 0; i < in_cols; ++i) {
        acc += float(input[x_base + i]) * float(weight[w_base + i]);
    }
    output[idx] = half(acc);
}
kernel void silu_mul_f16(
    device const half *gate [[buffer(0)]],
    device const half *up [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    float g = float(gate[id]);
    float u = float(up[id]);
    float silu = g / (1.0f + exp(-g));
    output[id] = half(silu * u);
}
kernel void gated_activation_f16(
    device const half *gate [[buffer(0)]],
    device const half *up [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &activation_kind [[buffer(4)]],
    constant float &alpha [[buffer(5)]],
    constant float &limit [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    float gate_value = float(gate[id]);
    output[id] = finite_f16(gated_activation_value(
        gate_value, float(up[id]), activation_kind, alpha, limit));
}
kernel void sigmoid_gate_f16(
    device const half *input [[buffer(0)]],
    device const half *gate [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &gate_columns [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    const uint row = id / columns;
    const uint column = id - row * columns;
    const uint gate_column = gate_columns == 1 ? 0 : column;
    const float gate_value = float(gate[ulong(row) * gate_columns + gate_column]);
    output[id] = finite_f16(float(input[id]) / (1.0f + exp(-gate_value)));
}
kernel void sigmoid_gate_f16_f32(
    device const half *input [[buffer(0)]], device const float *gate [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]], constant uint &gate_columns [[buffer(4)]],
    constant uint &count [[buffer(5)]], uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    const uint column = id % columns;
    const uint gate_index = gate_columns == 1 ? id / columns : id;
    output[id] = float(input[id]) / (1.0f + exp(-gate[gate_index]));
}
kernel void linear_sigmoid_gate_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &in_cols [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    constant uint &grid_threads [[buffer(6)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup half gate;
    if (simd_group == 0) {
        float sum = 0.0f;
        for (uint column = simd_lane; column < in_cols; column += 32) {
            sum += float(input[column]) * float(weight[column]);
        }
        sum = simd_sum(sum);
        if (simd_lane == 0) gate = finite_f16(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate_value = float(gate);
    for (uint id = group_index * 256 + thread_index; id < count; id += grid_threads) {
        output[id] = finite_f16(float(value[id]) / (1.0f + exp(-gate_value)));
    }
}
kernel void add_f16(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    output[id] = half(float(a[id]) + float(b[id]));
}
// 标量乘(gemma4 MTP 的 layer_output_scale):F16 单行小张量,逐元素即可。
kernel void mul_scalar_f16(
    device const half *a [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    output[id] = half(float(a[id]) * scale);
}
// 向量化版本:每线程处理 8 个 half(16 字节),把 144M 元素的残差加线程数降到 18M,
// 缓解 GPU 前端发射瓶颈。count 必须是 8 的倍数(调用方保证)。
kernel void add_f16_vec8(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    const uint base = id * 8;
    if (base >= count) return;
    device const half4 *a4 = (device const half4 *)(a + base);
    device const half4 *b4 = (device const half4 *)(b + base);
    device half4 *o4 = (device half4 *)(output + base);
    const half4 va0 = a4[0]; const half4 va1 = a4[1];
    const half4 vb0 = b4[0]; const half4 vb1 = b4[1];
    o4[0] = half4(float4(va0) + float4(vb0));
    o4[1] = half4(float4(va1) + float4(vb1));
}
kernel void add_scaled_f16(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    output[id] = half((float(a[id]) + float(b[id])) * scale);
}
kernel void add_scaled_f32_f16(
    device const float *a [[buffer(0)]], device const half *b [[buffer(1)]], device float *output [[buffer(2)]],
    constant float &scale [[buffer(3)]], constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = (a[id] + float(b[id])) * scale;
}
kernel void add_scaled_f16_f32(
    device const half *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *output [[buffer(2)]],
    constant float &scale [[buffer(3)]], constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = (float(a[id]) + b[id]) * scale;
}
kernel void add_scaled_f32(
    device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *output [[buffer(2)]],
    constant float &scale [[buffer(3)]], constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = (a[id] + b[id]) * scale;
}
kernel void add_f32_f16(
    device const float *a [[buffer(0)]], device const half *b [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = a[id] + float(b[id]);
}
kernel void add_f16_f32(
    device const half *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = float(a[id]) + b[id];
}
kernel void add_f32(
    device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = a[id] + b[id];
}
// 64×64 tile 的 simdgroup GEMM:output[m,n] = input[m,k] × weight[n,k]^T。
// 128 threads(4 SIMD groups)每组 32×32 输出(4×4 个 8×8 f32 累加器)。
// threadgroup 只驻 8KB 装载缓冲(每 SM 可驻多组提高并发);结果直写
// device f32,由调用方统一 cast 成 F16。m/n/k 均要求 64 对齐。
kernel void gemm_simd64_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &in_cols [[buffer(4)]],
    constant uint &out_cols [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]])
{
    threadgroup half stage_a[64 * 32];
    threadgroup half stage_b[32 * 64];
    const uint row_base = group.y * 64;
    const uint col_base = group.x * 64;
    const uint gi = simd_index & 1;
    const uint gj = simd_index >> 1;

    simdgroup_float8x8 acc[4][4];
    for (uint mi = 0; mi < 4; ++mi) {
        for (uint nj = 0; nj < 4; ++nj) {
            acc[mi][nj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k_base = 0; k_base < in_cols; k_base += 32) {
        for (uint index = thread_index; index < (64 * 32) / 4; index += 128) {
            const uint local_row = (index * 4) >> 5;
            const uint local_column = (index * 4) & 31;
            half4 value = 0.0h;
            if (row_base + local_row < rows) {
                device const half4 *source = (device const half4 *)(input + ulong(row_base + local_row) * in_cols + k_base + local_column);
                value = *source;
            }
            stage_a[local_row * 32 + local_column] = value.x;
            stage_a[local_row * 32 + local_column + 1] = value.y;
            stage_a[local_row * 32 + local_column + 2] = value.z;
            stage_a[local_row * 32 + local_column + 3] = value.w;
        }
        for (uint index = thread_index; index < (32 * 64) / 4; index += 128) {
            const uint linear_half = index * 4;
            const uint n_local = linear_half >> 5;
            const uint k_local = linear_half & 31;
            device const half4 *source = (device const half4 *)(weight + ulong(col_base + n_local) * in_cols + k_base + k_local);
            const half4 value = *source;
            stage_b[(k_local + 0) * 64 + n_local] = value.x;
            stage_b[(k_local + 1) * 64 + n_local] = value.y;
            stage_b[(k_local + 2) * 64 + n_local] = value.z;
            stage_b[(k_local + 3) * 64 + n_local] = value.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint mi = 0; mi < 4; ++mi) {
                simdgroup_load(a[mi], stage_a + (gi * 32 + mi * 8) * 32 + k, 32);
            }
            for (uint nj = 0; nj < 4; ++nj) {
                simdgroup_load(b[nj], stage_b + k * 64 + gj * 32 + nj * 8, 64);
            }
            for (uint mi = 0; mi < 4; ++mi) {
                for (uint nj = 0; nj < 4; ++nj) {
                    simdgroup_multiply_accumulate(acc[mi][nj], a[mi], b[nj], acc[mi][nj]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint mi = 0; mi < 4; ++mi) {
        for (uint nj = 0; nj < 4; ++nj) {
            simdgroup_store(acc[mi][nj], output + ulong(row_base + gi * 32 + mi * 8) * out_cols + col_base + gj * 32 + nj * 8, out_cols);
        }
    }
}
kernel void matmul_tiled_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &in_cols [[buffer(4)]],
    constant uint &out_cols [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup half input_tile[256];
    threadgroup half weight_tile[256];
    simdgroup_float8x8 accumulator_lo = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 accumulator_hi = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint row_base = group.y * 16;
    const uint output_base = group.x * 16;

    for (uint input_base = 0; input_base < in_cols; input_base += 16) {
        const uint begin = thread_index * 4;
        for (uint item = 0; item < 4; ++item) {
            const uint flat = begin + item;
            const uint local_row = flat >> 4;
            const uint local_column = flat & 15;
            const uint row = row_base + local_row;
            const uint input_column = input_base + local_column;
            const uint output_column = output_base + local_column;
            const uint weight_input = input_base + local_row;
            input_tile[flat] = row < rows && input_column < in_cols
                ? input[(ulong)row * in_cols + input_column]
                : half(0.0h);
            weight_tile[flat] = output_column < out_cols && weight_input < in_cols
                ? weight[(ulong)output_column * in_cols + weight_input]
                : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 16; k += 8) {
            simdgroup_half8x8 input_matrix_lo;
            simdgroup_half8x8 input_matrix_hi;
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(input_matrix_lo, input_tile + k, 16);
            simdgroup_load(input_matrix_hi, input_tile + 128 + k, 16);
            simdgroup_load(weight_matrix, weight_tile + k * 16 + simd_index * 8, 16);
            simdgroup_multiply_accumulate(accumulator_lo, input_matrix_lo, weight_matrix, accumulator_lo);
            simdgroup_multiply_accumulate(accumulator_hi, input_matrix_hi, weight_matrix, accumulator_hi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float result_tile[256];
    simdgroup_store(accumulator_lo, result_tile + simd_index * 8, 16);
    simdgroup_store(accumulator_hi, result_tile + 128 + simd_index * 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint begin = thread_index * 4;
    for (uint item = 0; item < 4; ++item) {
        const uint flat = begin + item;
        const uint row = row_base + (flat >> 4);
        const uint output_column = output_base + (flat & 15);
        if (row < rows && output_column < out_cols) {
            output[(ulong)row * out_cols + output_column] = half(result_tile[flat]);
        }
    }
}
kernel void gemv_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &in_cols [[buffer(3)]],
    constant uint &out_cols [[buffer(4)]],
    constant uint &input_rows [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint output_column [[threadgroup_position_in_grid]])
{
    // 权重元素只读一次,对批内 ≤8 个 input 行独立累加(ir 全展开保持常量索引,
    // sums 驻寄存器)。rows≤8 的小批(MTP/DSpark verify 的 alpha/beta 等小投影)
    // 曾走 matmul_tiled:16x16 tile 只有 3 个 threadgroup,4 行实测 4.5x 单行。
    if (output_column >= out_cols) return;
    const ulong weight_base = ulong(output_column) * in_cols;
    float sums[8];
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        sums[ir] = 0.0f;
    }
    for (uint column = lane; column < in_cols; column += 32) {
        const float value = float(weight[weight_base + column]);
        #pragma unroll
        for (uint ir = 0; ir < 8; ++ir) {
            if (ir < input_rows) {
                sums[ir] += float(input[ulong(ir) * in_cols + column]) * value;
            }
        }
    }
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        if (ir < input_rows) {
            const float sum = simd_sum(sums[ir]);
            if (lane == 0) {
                output[ulong(ir) * out_cols + output_column] = finite_f16(sum);
            }
        }
    }
}
kernel void dequantize_fp8_matrix_f16(
    device const uchar *codes [[buffer(0)]],
    device const float *scale_inv [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &scale_columns [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    const uint total = rows * columns;
    if (index >= total) return;
    const uint row = index / columns;
    const uint column = index - row * columns;
    const uint scale_index = (row / 128) * scale_columns + column / 128;
    output[index] = half(decode_f8_e4m3(codes[index]) * scale_inv[scale_index]);
}
kernel void matmul_tiled_fp8_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const float *scale_inv [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &in_cols [[buffer(5)]],
    constant uint &out_cols [[buffer(6)]],
    constant uint &scale_cols [[buffer(7)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup half input_tile[256];
    threadgroup half weight_tile[256];
    simdgroup_float8x8 accumulator_lo = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 accumulator_hi = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint row_base = group.y * 16;
    const uint output_base = group.x * 16;

    for (uint input_base = 0; input_base < in_cols; input_base += 16) {
        const uint begin = thread_index * 4;
        for (uint item = 0; item < 4; ++item) {
            const uint flat = begin + item;
            const uint local_row = flat >> 4;
            const uint local_column = flat & 15;
            const uint row = row_base + local_row;
            const uint input_column = input_base + local_column;
            const uint output_column = output_base + local_column;
            const uint weight_input = input_base + local_row;
            input_tile[flat] = row < rows && input_column < in_cols
                ? input[(ulong)row * in_cols + input_column]
                : half(0.0h);
            if (output_column < out_cols && weight_input < in_cols) {
                const uchar code = weight[(ulong)output_column * in_cols + weight_input];
                const uint scale_index = (output_column / 128) * scale_cols + weight_input / 128;
                weight_tile[flat] = half(decode_f8_e4m3(code) * scale_inv[scale_index]);
            } else {
                weight_tile[flat] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 16; k += 8) {
            simdgroup_half8x8 input_matrix_lo;
            simdgroup_half8x8 input_matrix_hi;
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(input_matrix_lo, input_tile + k, 16);
            simdgroup_load(input_matrix_hi, input_tile + 128 + k, 16);
            simdgroup_load(weight_matrix, weight_tile + k * 16 + simd_index * 8, 16);
            simdgroup_multiply_accumulate(accumulator_lo, input_matrix_lo, weight_matrix, accumulator_lo);
            simdgroup_multiply_accumulate(accumulator_hi, input_matrix_hi, weight_matrix, accumulator_hi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float result_tile[256];
    simdgroup_store(accumulator_lo, result_tile + simd_index * 8, 16);
    simdgroup_store(accumulator_hi, result_tile + 128 + simd_index * 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint begin = thread_index * 4;
    for (uint item = 0; item < 4; ++item) {
        const uint flat = begin + item;
        const uint row = row_base + (flat >> 4);
        const uint output_column = output_base + (flat & 15);
        if (row < rows && output_column < out_cols) {
            output[(ulong)row * out_cols + output_column] = half(result_tile[flat]);
        }
    }
}
kernel void matmul_per_tensor_fp8_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const float *scale_ptr [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &in_cols [[buffer(5)]],
    constant uint &out_cols [[buffer(6)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    const float scale = scale_ptr[0];
    threadgroup half input_tile[256];
    threadgroup half weight_tile[256];
    simdgroup_float8x8 accumulator_lo = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 accumulator_hi = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint row_base = group.y * 16;
    const uint output_base = group.x * 16;

    for (uint input_base = 0; input_base < in_cols; input_base += 16) {
        const uint begin = thread_index * 4;
        for (uint item = 0; item < 4; ++item) {
            const uint flat = begin + item;
            const uint local_row = flat >> 4;
            const uint local_column = flat & 15;
            const uint row = row_base + local_row;
            const uint input_column = input_base + local_column;
            const uint output_column = output_base + local_column;
            const uint weight_input = input_base + local_row;
            input_tile[flat] = row < rows && input_column < in_cols
                ? input[(ulong)row * in_cols + input_column]
                : half(0.0h);
            if (output_column < out_cols && weight_input < in_cols) {
                const uchar code = weight[(ulong)output_column * in_cols + weight_input];
                weight_tile[flat] = half(decode_f8_e4m3(code) * scale);
            } else {
                weight_tile[flat] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 16; k += 8) {
            simdgroup_half8x8 input_matrix_lo;
            simdgroup_half8x8 input_matrix_hi;
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(input_matrix_lo, input_tile + k, 16);
            simdgroup_load(input_matrix_hi, input_tile + 128 + k, 16);
            simdgroup_load(weight_matrix, weight_tile + k * 16 + simd_index * 8, 16);
            simdgroup_multiply_accumulate(accumulator_lo, input_matrix_lo, weight_matrix, accumulator_lo);
            simdgroup_multiply_accumulate(accumulator_hi, input_matrix_hi, weight_matrix, accumulator_hi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float result_tile[256];
    simdgroup_store(accumulator_lo, result_tile + simd_index * 8, 16);
    simdgroup_store(accumulator_hi, result_tile + 128 + simd_index * 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint begin = thread_index * 4;
    for (uint item = 0; item < 4; ++item) {
        const uint flat = begin + item;
        const uint row = row_base + (flat >> 4);
        const uint output_column = output_base + (flat & 15);
        if (row < rows && output_column < out_cols) {
            output[(ulong)row * out_cols + output_column] = half(result_tile[flat]);
        }
    }
}
kernel void official_fp8_matmul_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *scale_inv [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &input_rows [[buffer(4)]],
    constant uint &input_columns [[buffer(5)]],
    constant uint &output_columns [[buffer(6)]],
    constant uint &scale_columns [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]],
    uint flat_lane [[thread_index_in_threadgroup]])
{
    if (input_rows == 1) {
        const uint output_row = group.x * 128 + flat_lane;
        if (output_row >= output_columns) return;
        const ulong weight_base = (ulong)output_row * input_columns;
        const uint scale_base = (output_row / 128) * scale_columns;
        float sum = 0.0f;
        for (uint block = 0; block < input_columns; block += 128) {
            const float scale = scale_inv[scale_base + block / 128];
            const uint end = min(block + 128, input_columns);
            for (uint column = block; column < end; ++column) {
                const uchar code = weights[weight_base + column];
                const half weight = half(decode_f8_e4m3(code) * scale);
                sum += float(input[column]) * float(weight);
            }
        }
        output[output_row] = half(sum);
        return;
    }

    threadgroup half input_tile[8 * 16];
    threadgroup half weight_tile[16 * 16];

    const uint input_row = group.y * 8 + lane.y;
    const uint output_column = group.x * 16 + lane.x;
    const uint block_columns = input_columns / 16;
    float sum = 0.0f;

    for (uint block_column = 0; block_column < block_columns; ++block_column) {
        const uint tile_input_row = flat_lane / 16;
        const uint tile_input_column = flat_lane % 16;
        const uint source_row = group.y * 8 + tile_input_row;
        const uint source_column = block_column * 16 + tile_input_column;
        input_tile[flat_lane] = source_row < input_rows
            ? input[(ulong)source_row * input_columns + source_column]
            : half(0.0h);

        for (uint index = flat_lane; index < 256; index += 128) {
            const uint local_output = index / 16;
            const uint local_input = index % 16;
            const uint weight_row = group.x * 16 + local_output;
            const uint weight_column = block_column * 16 + local_input;
            uchar code = weights[(ulong)weight_row * input_columns + weight_column];
            uint scale_index = (weight_row / 128) * scale_columns + weight_column / 128;
            weight_tile[index] = half(decode_f8_e4m3(code) * scale_inv[scale_index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (input_row < input_rows) {
            uint weight_base = lane.x * 16;
            uint input_base = lane.y * 16;
            for (uint index = 0; index < 16; ++index) {
                sum += float(input_tile[input_base + index])
                    * float(weight_tile[weight_base + index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (input_row < input_rows) {
        output[(ulong)input_row * output_columns + output_column] = half(sum);
    }
}
kernel void official_fp8_dual_matmul_f16(
    device const half *input [[buffer(0)]],
    device const uchar *first_weights [[buffer(1)]],
    device const float *first_scale_inv [[buffer(2)]],
    device half *first_output [[buffer(3)]],
    device const uchar *second_weights [[buffer(4)]],
    device const float *second_scale_inv [[buffer(5)]],
    device half *second_output [[buffer(6)]],
    constant uint &input_rows [[buffer(7)]],
    constant uint &input_columns [[buffer(8)]],
    constant uint &first_output_columns [[buffer(9)]],
    constant uint &second_output_columns [[buffer(10)]],
    constant uint &first_scale_columns [[buffer(11)]],
    constant uint &second_scale_columns [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]],
    uint flat_lane [[thread_index_in_threadgroup]])
{
    if (input_rows == 1) {
        const uint combined_output = group.x * 128 + flat_lane;
        const uint combined_columns = first_output_columns + second_output_columns;
        if (combined_output >= combined_columns) return;

        const bool use_first = combined_output < first_output_columns;
        const uint output_row = use_first
            ? combined_output
            : combined_output - first_output_columns;
        const uint scale_columns = use_first
            ? first_scale_columns
            : second_scale_columns;
        device const uchar *weights = use_first ? first_weights : second_weights;
        device const float *scale_inv = use_first
            ? first_scale_inv
            : second_scale_inv;
        device half *output = use_first ? first_output : second_output;

        const ulong weight_base = (ulong)output_row * input_columns;
        const uint scale_base = (output_row / 128) * scale_columns;
        float sum = 0.0f;
        for (uint block = 0; block < input_columns; block += 128) {
            const float scale = scale_inv[scale_base + block / 128];
            const uint end = min(block + 128, input_columns);
            for (uint column = block; column < end; ++column) {
                const uchar code = weights[weight_base + column];
                const half weight = half(decode_f8_e4m3(code) * scale);
                sum += float(input[column]) * float(weight);
            }
        }
        output[output_row] = half(sum);
        return;
    }

    const uint first_blocks = first_output_columns / 16;
    const bool first = group.x < first_blocks;
    const uint output_block = first ? group.x : group.x - first_blocks;
    const uint output_columns = first ? first_output_columns : second_output_columns;
    const uint scale_columns = first ? first_scale_columns : second_scale_columns;
    device const uchar *weights = first ? first_weights : second_weights;
    device const float *scale_inv = first ? first_scale_inv : second_scale_inv;
    device half *output = first ? first_output : second_output;

    threadgroup half input_tile[8 * 16];
    threadgroup half weight_tile[16 * 16];
    const uint input_row = group.y * 8 + lane.y;
    const uint output_column = output_block * 16 + lane.x;
    const uint block_columns = input_columns / 16;
    float sum = 0.0f;

    for (uint block_column = 0; block_column < block_columns; ++block_column) {
        const uint tile_input_row = flat_lane / 16;
        const uint tile_input_column = flat_lane % 16;
        const uint source_row = group.y * 8 + tile_input_row;
        const uint source_column = block_column * 16 + tile_input_column;
        input_tile[flat_lane] = source_row < input_rows
            ? input[(ulong)source_row * input_columns + source_column]
            : half(0.0h);

        for (uint index = flat_lane; index < 256; index += 128) {
            const uint local_output = index / 16;
            const uint local_input = index % 16;
            const uint weight_row = output_block * 16 + local_output;
            const uint weight_column = block_column * 16 + local_input;
            const uchar code = weights[(ulong)weight_row * input_columns + weight_column];
            const uint scale_index = (weight_row / 128) * scale_columns
                + weight_column / 128;
            weight_tile[index] = half(decode_f8_e4m3(code) * scale_inv[scale_index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (input_row < input_rows) {
            const uint weight_base = lane.x * 16;
            const uint input_base = lane.y * 16;
            for (uint index = 0; index < 16; ++index) {
                sum += float(input_tile[input_base + index])
                    * float(weight_tile[weight_base + index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (input_row < input_rows && output_column < output_columns) {
        output[(ulong)input_row * output_columns + output_column] = half(sum);
    }
}
kernel void layernorm_bias_f16(
    device const half *input             [[buffer(0)]],
    device const half *weight            [[buffer(1)]],
    device const half *bias              [[buffer(2)]],
    device half *output                  [[buffer(3)]],
    constant uint &columns               [[buffer(4)]],
    constant float &eps                  [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float sums[256];
    threadgroup float squares[256];
    const ulong base = (ulong)row * columns;
    const float value = lane < columns ? float(input[base + lane]) : 0.0f;
    sums[lane] = value;
    squares[lane] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            squares[lane] += squares[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < columns) {
        const float mean = sums[0] / float(columns);
        const float variance = max(0.0f, squares[0] / float(columns) - mean * mean);
        output[base + lane] = half((value - mean) * rsqrt(variance + eps) * float(weight[lane]) + float(bias[lane]));
    }
}
kernel void layernorm_bias_bf16(
    device const ushort *input           [[buffer(0)]],
    device const half *weight            [[buffer(1)]],
    device const half *bias              [[buffer(2)]],
    device ushort *output                [[buffer(3)]],
    constant uint &columns               [[buffer(4)]],
    constant float &eps                  [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float sums[256];
    threadgroup float squares[256];
    const ulong base = (ulong)row * columns;
    const float value = lane < columns ? zllm_bf16_to_f32(input[base + lane]) : 0.0f;
    sums[lane] = value;
    squares[lane] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            squares[lane] += squares[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < columns) {
        const float mean = sums[0] / float(columns);
        const float variance = max(0.0f, squares[0] / float(columns) - mean * mean);
        output[base + lane] = zllm_f32_to_bf16((value - mean) * rsqrt(variance + eps) * float(weight[lane]) + float(bias[lane]));
    }
}
kernel void layernorm_bias_wide_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float sums[256];
    threadgroup float squares[256];
    const ulong base = ulong(row) * columns;
    float sum = 0.0f;
    float square = 0.0f;
    for (uint column = lane; column < columns; column += 256) {
        const float value = float(input[base + column]);
        sum += value;
        square += value * value;
    }
    sums[lane] = sum;
    squares[lane] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            squares[lane] += squares[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = sums[0] / float(columns);
    const float variance = max(0.0f, squares[0] / float(columns) - mean * mean);
    const float inverse_std = rsqrt(variance + eps);
    for (uint column = lane; column < columns; column += 256) {
        const float value = float(input[base + column]);
        output[base + column] = finite_f16((value - mean) * inverse_std * float(weight[column]) + float(bias[column]));
    }
}
kernel void layernorm_bias_wide_bf16(
    device const ushort *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device ushort *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float sums[256];
    threadgroup float squares[256];
    const ulong base = ulong(row) * columns;
    float sum = 0.0f;
    float square = 0.0f;
    for (uint column = lane; column < columns; column += 256) {
        const float value = zllm_bf16_to_f32(input[base + column]);
        sum += value;
        square += value * value;
    }
    sums[lane] = sum;
    squares[lane] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            squares[lane] += squares[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = sums[0] / float(columns);
    const float variance = max(0.0f, squares[0] / float(columns) - mean * mean);
    const float inverse_std = rsqrt(variance + eps);
    for (uint column = lane; column < columns; column += 256) {
        const float value = zllm_bf16_to_f32(input[base + column]);
        output[base + column] = zllm_f32_to_bf16((value - mean) * inverse_std * float(weight[column]) + float(bias[column]));
    }
}
kernel void cast_bf16_f16(
    device const ushort *input [[buffer(0)]], device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = finite_f16(zllm_bf16_to_f32(input[id]));
}
kernel void cast_f16_bf16(
    device const half *input [[buffer(0)]], device ushort *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = zllm_f32_to_bf16(float(input[id]));
}
kernel void cast_f32_bf16(
    device const float *input [[buffer(0)]], device ushort *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = zllm_f32_to_bf16(input[id]);
}
kernel void rms_norm_bf16(
    device const ushort *input [[buffer(0)]], device const half *weight [[buffer(1)]],
    device ushort *output [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant float &epsilon [[buffer(4)]], constant float &weight_offset [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float sums[256];
    const ulong offset = ulong(row) * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += width) {
        const float value = zllm_bf16_to_f32(input[offset + column]);
        sum += value * value;
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(sums[0] / float(columns) + epsilon);
    for (uint column = lane; column < columns; column += width) {
        const float value = zllm_bf16_to_f32(input[offset + column]);
        const float normalized = zllm_bf16_to_f32(zllm_f32_to_bf16(value * scale));
        output[offset + column] = zllm_f32_to_bf16(normalized * (float(weight[column]) + weight_offset));
    }
}
kernel void gated_activation_bf16(
    device const ushort *gate [[buffer(0)]], device const ushort *up [[buffer(1)]],
    device ushort *output [[buffer(2)]], constant uint &count [[buffer(3)]],
    constant uint &activation_kind [[buffer(4)]], constant float &alpha [[buffer(5)]],
    constant float &limit [[buffer(6)]], uint id [[thread_position_in_grid]])
{
    if (id >= count) return;
    output[id] = zllm_f32_to_bf16(gated_activation_value(
        zllm_bf16_to_f32(gate[id]), zllm_bf16_to_f32(up[id]), activation_kind, alpha, limit));
}
kernel void add_bf16(
    device const ushort *a [[buffer(0)]], device const ushort *b [[buffer(1)]],
    device ushort *output [[buffer(2)]], constant uint &count [[buffer(3)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = zllm_f32_to_bf16(zllm_bf16_to_f32(a[id]) + zllm_bf16_to_f32(b[id]));
}
kernel void add_scaled_bf16(
    device const ushort *a [[buffer(0)]], device const ushort *b [[buffer(1)]],
    device ushort *output [[buffer(2)]], constant float &scale [[buffer(3)]],
    constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]])
{
    if (id < count) output[id] = zllm_f32_to_bf16((zllm_bf16_to_f32(a[id]) + zllm_bf16_to_f32(b[id])) * scale);
}
"#;

use crate::backend::metal::api as metal;

use super::tensor::gated_activation_tensor;
use super::{Activation, Fp8Matrix, MTLSize, MetalContext, MetalTensor, f16, f32_to_f16, launch_1d, mem, set_bytes, validate_size, validate_u32};

pub(super) fn launch_matmul_f16(ctx: &MetalContext, input: &metal::Buffer, weight: &metal::Buffer, output: &metal::Buffer, rows: usize, in_cols: usize, out_cols: usize) -> Result<(), String> {
    // 全维 64 对齐的大矩阵走 simdgroup GEMM(M5 实测 35.4 TFLOPS,MPS 的 2.8 倍);
    // F32 累加按 2048 行分块,与 MPS 路径共用 accumulator/cast 内存预算。
    if rows >= 256 && in_cols.is_multiple_of(64) && out_cols.is_multiple_of(64) {
        // 24GB 统一内存上 2048 行块(285MB/尺寸)会把权重挤出内存引发缺页,
        // 256 行块 accumulator 约 35MB/尺寸,池化后总驻留 ~250MB。
        const GEMM_BLOCK_ROWS: usize = 256;
        const CAST_THREADS: usize = 256;
        // 分块行数取 64 的倍数(尾部 padding 行只写 accumulator,不参与 cast)。
        let block_rows_aligned = rows.min(GEMM_BLOCK_ROWS).next_multiple_of(64);
        // 每次矩阵乘调用分配一次、块间复用、调用结束释放:M5 统一内存下
        // 跨投影持有多种尺寸的池(2GB 级)会把 12.5GB 权重挤出高效带,所有
        // kernel 整体退化 10×;稳态 285MB 与 MPS 路径的内存足迹一致。
        let accumulator = ctx.shared_buffer_zeros(block_rows_aligned * out_cols * mem::size_of::<f32>());
        let gemm = ctx.pipeline("gemm_simd64_f16")?;
        let cast = ctx.pipeline("cast_f32_to_f16")?;
        let in_cols_u32 = validate_u32("GEMM in_cols", in_cols)?;
        let out_cols_u32 = validate_u32("GEMM out_cols", out_cols)?;
        let command = ctx.command_buffer();
        for row_begin in (0..rows).step_by(GEMM_BLOCK_ROWS) {
            let block_rows = (rows - row_begin).min(GEMM_BLOCK_ROWS);
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&gemm);
            encoder.set_buffer(0, Some(input), (row_begin * in_cols * mem::size_of::<f16>()) as u64);
            encoder.set_buffer(1, Some(weight), 0);
            encoder.set_buffer(2, Some(&accumulator), 0);
            let block_rows_u32 = validate_u32("GEMM block rows", block_rows)?;
            set_bytes(&encoder, 3, &block_rows_u32);
            set_bytes(&encoder, 4, &in_cols_u32);
            set_bytes(&encoder, 5, &out_cols_u32);
            encoder.dispatch_thread_groups(MTLSize::new((out_cols / 64) as u64, block_rows.div_ceil(64) as u64, 1), MTLSize::new(128, 1, 1));
            encoder.end_encoding();
            let count = validate_u32("GEMM cast count", block_rows * out_cols)?;
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&cast);
            encoder.set_buffer(0, Some(&accumulator), 0);
            encoder.set_buffer(1, Some(output), (row_begin * out_cols * mem::size_of::<f16>()) as u64);
            set_bytes(&encoder, 2, &count);
            encoder.dispatch_thread_groups(MTLSize::new((count as usize).div_ceil(CAST_THREADS) as u64, 1, 1), MTLSize::new(CAST_THREADS as u64, 1, 1));
            encoder.end_encoding();
        }
        let shape = format!("input=[{rows},{in_cols}],weight=[{out_cols},{in_cols}]");
        ctx.commit_and_wait_profiled(&command, "gemm_simd64_f16", &shape, input.length() + weight.length(), output.length() + accumulator.length());
        return Ok(());
    }
    if rows >= 16 {
        const ACCUMULATOR_ROWS: usize = 2048;
        const CAST_THREADS: usize = 256;
        let tile_rows = rows.min(ACCUMULATOR_ROWS);
        let accumulator_elements = tile_rows.checked_mul(out_cols).ok_or("F32 matmul accumulator 大小溢出")?;
        let accumulator = ctx.shared_buffer_zeros(accumulator_elements * mem::size_of::<f32>());
        let cast = ctx.pipeline("cast_f32_to_f16")?;
        let command = ctx.command_buffer();
        for row_begin in (0..rows).step_by(ACCUMULATOR_ROWS) {
            let block_rows = (rows - row_begin).min(ACCUMULATOR_ROWS);
            let block_elements = block_rows.checked_mul(out_cols).ok_or("F32 matmul block 大小溢出")?;
            super::mps::encode_f16_matmul_f32(
                &command,
                &ctx.device,
                input,
                row_begin * in_cols * mem::size_of::<f16>(),
                in_cols * mem::size_of::<f16>(),
                weight,
                0,
                in_cols * mem::size_of::<f16>(),
                &accumulator,
                0,
                out_cols * mem::size_of::<f32>(),
                block_rows,
                in_cols,
                out_cols,
                true,
                1.0,
            )?;
            let count = validate_u32("F32 matmul cast count", block_elements)?;
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&cast);
            encoder.set_buffer(0, Some(&accumulator), 0);
            encoder.set_buffer(1, Some(output), (row_begin * out_cols * mem::size_of::<f16>()) as u64);
            set_bytes(&encoder, 2, &count);
            encoder.dispatch_thread_groups(MTLSize::new(block_elements.div_ceil(CAST_THREADS) as u64, 1, 1), MTLSize::new(CAST_THREADS as u64, 1, 1));
            encoder.end_encoding();
        }
        let shape = format!("input=[{rows},{in_cols}],weight=[{out_cols},{in_cols}],tile_rows={tile_rows}");
        ctx.commit_and_wait_profiled(&command, "mps_matrix_multiplication_f32_accum_f16_store", &shape, input.length() + weight.length(), output.length() + accumulator.length());
        return Ok(());
    }
    let rows = validate_u32("rows", rows)?;
    let in_cols = validate_u32("in_cols", in_cols)?;
    let out_cols = validate_u32("out_cols", out_cols)?;
    let shape = format!("input=[{rows},{in_cols}],weight=[{out_cols},{in_cols}]");
    let pipeline = ctx.pipeline("matmul_tiled_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("tiled matmul 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(output), 0);
    set_bytes(&encoder, 3, &rows);
    set_bytes(&encoder, 4, &in_cols);
    set_bytes(&encoder, 5, &out_cols);
    encoder.dispatch_thread_groups(MTLSize::new((out_cols as usize).div_ceil(16) as u64, (rows as usize).div_ceil(16) as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    ctx.commit_and_wait_profiled(&command, "matmul_tiled_f16", &shape, input.length() + weight.length(), output.length());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_fp8_matmul(encoder: &metal::ComputeCommandEncoderRef, input: &metal::Buffer, weight: &metal::Buffer, scale_inv: &metal::Buffer, output: &metal::Buffer, rows: u32, in_cols: u32, out_cols: u32, scale_cols: u32) {
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(scale_inv), 0);
    encoder.set_buffer(3, Some(output), 0);
    set_bytes(encoder, 4, &rows);
    set_bytes(encoder, 5, &in_cols);
    set_bytes(encoder, 6, &out_cols);
    set_bytes(encoder, 7, &scale_cols);
    encoder.dispatch_thread_groups(MTLSize::new((out_cols as usize).div_ceil(16) as u64, (rows as usize).div_ceil(16) as u64, 1), MTLSize::new(64, 1, 1));
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_official_fp8_matmul(ctx: &MetalContext, input: &metal::Buffer, weight: &metal::Buffer, scale_inv: &metal::Buffer, output: &metal::Buffer, rows: u32, in_cols: u32, out_cols: u32, scale_cols: u32) -> Result<(), String> {
    let pipeline = ctx.pipeline("official_fp8_matmul_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("官方 FP8 matmul 需要至少 128 threads/threadgroup".to_owned());
    }

    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(scale_inv), 0);
    encoder.set_buffer(3, Some(output), 0);
    set_bytes(&encoder, 4, &rows);
    set_bytes(&encoder, 5, &in_cols);
    set_bytes(&encoder, 6, &out_cols);
    set_bytes(&encoder, 7, &scale_cols);

    let (thread_groups, threads_per_group) =
        if rows == 1 { (MTLSize::new(out_cols.div_ceil(128) as u64, 1, 1), MTLSize::new(128, 1, 1)) } else { (MTLSize::new(out_cols.div_ceil(16) as u64, (rows as usize).div_ceil(8) as u64, 1), MTLSize::new(16, 8, 1)) };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);
    encoder.end_encoding();

    let shape = format!("input=[{rows},{in_cols}],weight=[{out_cols},{in_cols}]");
    ctx.commit_and_wait_profiled(&command, "official_fp8_matmul_f16", &shape, input.length() + weight.length() + scale_inv.length(), output.length());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_official_fp8_dual_matmul(
    ctx: &MetalContext,
    input: &metal::Buffer,
    first: &metal::Buffer,
    first_scale_inv: &metal::Buffer,
    first_output: &metal::Buffer,
    first_cols: u32,
    first_scale_cols: u32,
    second: &metal::Buffer,
    second_scale_inv: &metal::Buffer,
    second_output: &metal::Buffer,
    second_cols: u32,
    second_scale_cols: u32,
    rows: u32,
    in_cols: u32,
) -> Result<(), String> {
    let pipeline = ctx.pipeline("official_fp8_dual_matmul_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 128 {
        return Err("官方 FP8 dual matmul 需要至少 128 threads/threadgroup".to_owned());
    }

    let combined = first_cols.checked_add(second_cols).ok_or_else(|| "官方 FP8 dual 输出列数越界".to_owned())?;
    let (thread_groups, threads_per_group) =
        if rows == 1 { (MTLSize::new(combined.div_ceil(128) as u64, 1, 1), MTLSize::new(128, 1, 1)) } else { (MTLSize::new(combined.div_ceil(16) as u64, (rows as usize).div_ceil(8) as u64, 1), MTLSize::new(16, 8, 1)) };
    let shape = format!("input=[{rows},{in_cols}],outputs=[{first_cols},{second_cols}]");
    super::launch_nd(
        ctx,
        "official_fp8_dual_matmul_f16",
        &shape,
        thread_groups,
        threads_per_group,
        input.length() + first.length() + first_scale_inv.length() + second.length() + second_scale_inv.length(),
        first_output.length() + second_output.length(),
        |encoder| {
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(first), 0);
            encoder.set_buffer(2, Some(first_scale_inv), 0);
            encoder.set_buffer(3, Some(first_output), 0);
            encoder.set_buffer(4, Some(second), 0);
            encoder.set_buffer(5, Some(second_scale_inv), 0);
            encoder.set_buffer(6, Some(second_output), 0);
            set_bytes(encoder, 7, &rows);
            set_bytes(encoder, 8, &in_cols);
            set_bytes(encoder, 9, &first_cols);
            set_bytes(encoder, 10, &second_cols);
            set_bytes(encoder, 11, &first_scale_cols);
            set_bytes(encoder, 12, &second_scale_cols);
        },
    )
}

/// per-tensor FP8 权重(`matmul_per_tensor_fp8_f16`):scale 是单个 F32,
/// 16×16 simdgroup tile,64 threads/threadgroup(2 个 simdgroup)。
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_per_tensor_fp8_matmul(ctx: &MetalContext, input: &metal::Buffer, weight: &metal::Buffer, scale: &metal::Buffer, output: &metal::Buffer, rows: u32, in_cols: u32, out_cols: u32) -> Result<(), String> {
    let pipeline = ctx.pipeline("matmul_per_tensor_fp8_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("per-tensor FP8 matmul 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(scale), 0);
    encoder.set_buffer(3, Some(output), 0);
    set_bytes(&encoder, 4, &rows);
    set_bytes(&encoder, 5, &in_cols);
    set_bytes(&encoder, 6, &out_cols);
    encoder.dispatch_thread_groups(MTLSize::new((out_cols as usize).div_ceil(16) as u64, (rows as usize).div_ceil(16) as u64, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let shape = format!("input=[{rows},{in_cols}],weight=[{out_cols},{in_cols}]");
    ctx.commit_and_wait_profiled(&command, "matmul_per_tensor_fp8_f16", &shape, input.length() + weight.length() + scale.length(), output.length());
    Ok(())
}

/// `y = x @ w^T`，FP8 权重在 GPU tile 内直接解码，不生成完整 f16 权重。
pub fn fp8_matmul(ctx: &MetalContext, input: &[f32], weight: &Fp8Matrix, rows: usize) -> Result<Vec<f32>, String> {
    validate_size("FP8 input", rows * weight.cols, input.len())?;
    if rows == 0 {
        return Ok(Vec::new());
    }

    let input = f32_to_f16(ctx, input);
    let codes = ctx.shared_buffer(&weight.codes);
    let scale_inv = ctx.shared_buffer(&weight.scale_inv);
    let output = ctx.shared_buffer_zeros(rows * weight.rows * mem::size_of::<f16>());
    let rows_u32 = validate_u32("rows", rows)?;
    let in_cols = validate_u32("FP8 in_cols", weight.cols)?;
    let out_cols = validate_u32("FP8 out_cols", weight.rows)?;
    let scale_cols = validate_u32("FP8 scale_cols", weight.cols.div_ceil(128))?;
    if rows_u32 == 1 {
        launch_official_fp8_matmul(ctx, &input, &codes, &scale_inv, &output, rows_u32, in_cols, out_cols, scale_cols)?;
        return Ok(ctx.read_f16_to_f32(&output, rows * weight.rows));
    }

    let pipeline = ctx.pipeline("matmul_tiled_fp8_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 64 {
        return Err("tiled FP8 matmul 需要至少 64 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encode_fp8_matmul(&encoder, &input, &codes, &scale_inv, &output, rows_u32, in_cols, out_cols, scale_cols);
    encoder.end_encoding();
    let shape = format!("input=[{rows_u32},{in_cols}],weight=[{out_cols},{in_cols}]");
    ctx.commit_and_wait_profiled(&command, "matmul_tiled_fp8_f16", &shape, input.length() + codes.length() + scale_inv.length(), output.length());
    Ok(ctx.read_f16_to_f32(&output, rows * weight.rows))
}

#[derive(Clone, Copy)]
pub(super) struct GatedActivation {
    pub(super) kind: u32,
    pub(super) alpha: f32,
    pub(super) limit: f32,
}

impl GatedActivation {
    pub(super) fn from_spec(activation: &Activation) -> Result<Self, String> {
        match activation {
            Activation::Silu => Ok(Self { kind: 0, alpha: 1.0, limit: f32::MAX }),
            Activation::SiluClamped { limit } if limit.is_finite() && *limit > 0.0 => Ok(Self { kind: 4, alpha: 1.0, limit: *limit }),
            Activation::SiluClamped { limit } => Err(format!("限幅 SwiGLU 参数非法: limit={limit}")),
            Activation::Situ { beta, linear_beta } if beta.is_finite() && *beta > 0.0 && linear_beta.is_none_or(|value| value.is_finite() && value > 0.0) => Ok(Self { kind: 3, alpha: *beta, limit: linear_beta.unwrap_or(0.0) }),
            Activation::Situ { beta, linear_beta } => Err(format!("SiTU 参数非法: beta={beta}, linear_beta={linear_beta:?}")),
            Activation::SwigluOai { alpha, limit } if alpha.is_finite() && *alpha > 0.0 && limit.is_finite() && *limit > 0.0 => Ok(Self { kind: 1, alpha: *alpha, limit: *limit }),
            Activation::SwigluOai { alpha, limit } => Err(format!("SwiGLU-OAI 参数非法: alpha={alpha}, limit={limit}")),
            Activation::GeluTanh => Ok(Self { kind: 2, alpha: 1.0, limit: f32::MAX }),
        }
    }
}

/// 根据 activation 规格执行 gated activation。
pub fn gated_activation(ctx: &MetalContext, gate: &[f32], up: &[f32], activation: &Activation) -> Result<Vec<f32>, String> {
    validate_size("gate", up.len(), gate.len())?;
    if gate.is_empty() {
        return Ok(Vec::new());
    }
    let gate = ctx.tensor_from_f32(gate, gate.len(), 1)?;
    let up = ctx.tensor_from_f32(up, up.len(), 1)?;
    let output = gated_activation_tensor(ctx, &gate, &up, activation)?;
    Ok(ctx.tensor_to_f32(&output))
}

/// 消费 packed `[rows, 2*columns]` gate/up（前 columns 列 gate、后 columns 列 up），
/// 单 pass 输出 `activation(gate, up)` `[rows, columns]`。替代 split_columns + gated_activation
/// 两步,省两个中间 buffer 与发散 split kernel。kernel: gated_activation_packed_f16。
pub fn gated_activation_packed_tensor(ctx: &MetalContext, input: &MetalTensor, columns: usize, activation: &Activation) -> Result<MetalTensor, String> {
    let expected = columns.checked_mul(2).ok_or("gated_activation_packed columns 溢出")?;
    if input.cols != expected {
        return Err(format!("gated_activation_packed input cols={} 与 2*columns={expected} 不符", input.cols));
    }
    let output = ctx.tensor_kernel_output(input.rows, columns);
    let count = validate_u32("gated_activation_packed count", output.len())?;
    let columns_u32 = validate_u32("gated_activation_packed columns", columns)?;
    let params = GatedActivation::from_spec(activation)?;
    let shape = format!("rows={},columns={columns}", input.rows);
    launch_1d(ctx, "gated_activation_packed_f16", &shape, output.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count);
        set_bytes(encoder, 3, &columns_u32);
        set_bytes(encoder, 4, &params.kind);
        set_bytes(encoder, 5, &params.alpha);
        set_bytes(encoder, 6, &params.limit);
    })?;
    Ok(output)
}

pub(super) fn validate_tensor(name: &str, tensor: &MetalTensor, rows: usize, cols: usize) -> Result<(), String> {
    if tensor.rows == rows && tensor.cols == cols { Ok(()) } else { Err(format!("{name} shape 不匹配: 实际=[{},{}], 期望=[{rows},{cols}]", tensor.rows, tensor.cols)) }
}

fn launch_gemv_f16(ctx: &MetalContext, input: &metal::Buffer, weight: &metal::Buffer, output: &metal::Buffer, rows: usize, in_cols: usize, out_cols: usize) -> Result<(), String> {
    let in_cols_u32 = validate_u32("F16 GEMV input columns", in_cols)?;
    let out_cols_u32 = validate_u32("F16 GEMV output columns", out_cols)?;
    let input_rows = validate_u32("F16 GEMV input rows", rows.max(1))?;
    let pipeline = ctx.pipeline("gemv_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 32 {
        return Err("F16 GEMV 需要至少 32 threads/threadgroup".to_owned());
    }
    let shape = format!("input=[{rows},{in_cols}],weight=[{out_cols},{in_cols}]");
    super::launch_2d(ctx, "gemv_f16", &shape, out_cols, 1, 32, input.length() + weight.length(), output.length(), |encoder| {
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        set_bytes(encoder, 3, &in_cols_u32);
        set_bytes(encoder, 4, &out_cols_u32);
        set_bytes(encoder, 5, &input_rows);
    })
}

/// 单 token 标量 gate 投影后直接门控 value，保留 GEMV 的 F16 舍入边界。
pub fn linear_sigmoid_gate_tensor(ctx: &MetalContext, input: &MetalTensor, weight: &MetalTensor, value: &MetalTensor) -> Result<MetalTensor, String> {
    if input.rows != 1 || weight.rows != 1 || input.cols != weight.cols || value.rows != 1 {
        return Err(format!("F16 linear sigmoid gate shape 不兼容: input=[{},{}], weight=[{},{}], value=[{},{}]", input.rows, input.cols, weight.rows, weight.cols, value.rows, value.cols,));
    }
    let in_cols = validate_u32("F16 linear sigmoid gate input columns", input.cols)?;
    let count = validate_u32("F16 linear sigmoid gate value count", value.len())?;
    let output = ctx.tensor_kernel_output(value.rows, value.cols);
    let pipeline = ctx.pipeline("linear_sigmoid_gate_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < 256 {
        return Err("F16 linear sigmoid gate 需要至少 256 threads/threadgroup".to_owned());
    }
    let groups = value.len().div_ceil(256).max(1);
    let grid_threads = validate_u32("F16 linear sigmoid gate grid threads", groups * 256)?;
    let shape = format!("input=[1,{}],value=[1,{}]", input.cols, value.cols);
    super::launch_2d(ctx, "linear_sigmoid_gate_f16", &shape, groups, 1, 256, input.buffer.length() + weight.buffer.length() + value.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&weight.buffer), 0);
        encoder.set_buffer(2, Some(&value.buffer), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        set_bytes(encoder, 4, &in_cols);
        set_bytes(encoder, 5, &count);
        set_bytes(encoder, 6, &grid_threads);
    })?;
    Ok(output)
}

/// `y = x @ w^T`,x=[rows,in_cols],w=[out_cols,in_cols](已在 GPU 的 f16 MetalTensor,不重新上传)。
/// 给 lm_head 这种大权重常驻 GPU 的场景用(decode 循环复用)。
pub fn matmul_tensor_resident_weight(ctx: &MetalContext, input: &MetalTensor, weight: &MetalTensor) -> Result<MetalTensor, String> {
    let out_cols = weight.rows;
    validate_size("resident weight cols", out_cols * input.cols, weight.rows * weight.cols)?;
    if input.cols != weight.cols {
        return Err(format!("resident weight in_cols {} 与 input cols {} 不符", weight.cols, input.cols));
    }
    let output = ctx.tensor_kernel_output(input.rows, out_cols);
    if input.rows <= 8 {
        // 小批走多行 gemv(权重读一次跨行共享);9..15 行维持 tiled GEMM
        launch_gemv_f16(ctx, &input.buffer, &weight.buffer, &output.buffer, input.rows, input.cols, out_cols)?;
    } else {
        launch_matmul_f16(ctx, &input.buffer, &weight.buffer, &output.buffer, input.rows, input.cols, out_cols)?;
    }
    Ok(output)
}

/// BF16 activation 使用 F16 resident weight，MPS 以 F32 保存结果，避免 F16 输出动态范围截断。
pub fn matmul_tensor_resident_weight_f32(ctx: &MetalContext, input: &MetalTensor, weight: &MetalTensor) -> Result<MetalTensor, String> {
    let out_cols = weight.rows;
    validate_size("resident weight cols", out_cols * input.cols, weight.rows * weight.cols)?;
    if input.cols != weight.cols {
        return Err(format!("resident weight in_cols {} 与 input cols {} 不符", weight.cols, input.cols));
    }
    let output = ctx.tensor_kernel_output_f32(input.rows, out_cols);
    let command = ctx.command_buffer();
    super::mps::encode_f16_matmul_transposed_f32(&command, &ctx.device, &input.buffer, input.rows, input.cols, &weight.buffer, out_cols, &output.buffer)?;
    let shape = format!("input=[{},{}],weight=[{out_cols},{}]", input.rows, input.cols, weight.cols);
    ctx.commit_and_wait_profiled(&command, "mps_matrix_multiplication_f16_f32", &shape, input.buffer.length() + weight.buffer.length(), output.buffer.length());
    Ok(output)
}

/// F32 activation 乘 F16 resident weight，MPS 以 F32 保存结果。
pub fn matmul_tensor_resident_weight_mixed_f32(ctx: &MetalContext, input: &MetalTensor, weight: &MetalTensor) -> Result<MetalTensor, String> {
    let out_cols = weight.rows;
    if input.dtype != super::MetalTensorDType::F32 || weight.dtype != super::MetalTensorDType::F16 || input.cols != weight.cols {
        return Err(format!("mixed F32/F16 matmul shape/dtype 不兼容: input=[{},{}] {:?}, weight=[{},{}] {:?}", input.rows, input.cols, input.dtype, weight.rows, weight.cols, weight.dtype));
    }
    let output = ctx.tensor_kernel_output_f32(input.rows, out_cols);
    let command = ctx.command_buffer();
    super::mps::encode_f32_f16_matmul_transposed_f32(&command, &ctx.device, &input.buffer, input.rows, input.cols, &weight.buffer, out_cols, &output.buffer)?;
    let shape = format!("input=[{},{}],weight=[{out_cols},{}]", input.rows, input.cols, weight.cols);
    ctx.commit_and_wait_profiled(&command, "mps_matrix_multiplication_f32_f16_f32", &shape, input.buffer.length() + weight.buffer.length(), output.buffer.length());
    Ok(output)
}

#[cfg(test)]
mod mixed_matmul_tests {
    use super::*;

    #[test]
    fn f32_input_f16_weight_keeps_f32_result() {
        let ctx = MetalContext::new_default().unwrap();
        let input = ctx.tensor_from_f32_preserve(&[1.0, 2.0], 1, 2).unwrap();
        let weight = ctx.tensor_from_f32(&[3.0, 4.0, 5.0, 6.0], 2, 2).unwrap();
        let output = matmul_tensor_resident_weight_mixed_f32(&ctx, &input, &weight).unwrap();
        assert_eq!(output.dtype, super::super::MetalTensorDType::F32);
        assert_eq!(ctx.tensor_to_f32(&output), vec![11.0, 17.0]);
    }

    #[test]
    fn f16_inputs_use_f32_accumulation() {
        const COLUMNS: usize = 8192;
        let ctx = MetalContext::new_default().unwrap();
        let input = ctx.tensor_from_f32(&vec![1.0; COLUMNS], 1, COLUMNS).unwrap();
        let weight = ctx.tensor_from_f32(&vec![1.0; COLUMNS], 1, COLUMNS).unwrap();
        let output = ctx.tensor_kernel_output_f32(1, 1);
        let command = ctx.command_buffer();
        super::super::mps::encode_f16_matmul_transposed_f32(&command, &ctx.device, &input.buffer, 1, COLUMNS, &weight.buffer, 1, &output.buffer).unwrap();
        ctx.commit_and_wait(&command);
        assert_eq!(ctx.tensor_to_f32(&output), vec![COLUMNS as f32]);
    }
}

/// `y = x @ w^T`，输入、权重和输出均保持 F32；用于路由、门控等小型控制投影。
pub fn matmul_tensor_resident_f32(ctx: &MetalContext, input: &MetalTensor, weight: &metal::Buffer, weight_len: usize) -> Result<MetalTensor, String> {
    if input.cols == 0 || !weight_len.is_multiple_of(input.cols) {
        return Err(format!("F32 resident weight 元素数 {weight_len} 不能按 input cols {} 分行", input.cols));
    }
    let out_cols = weight_len / input.cols;
    let output = ctx.tensor_kernel_output_f32(input.rows, out_cols);
    let command = ctx.command_buffer();
    super::mps::encode_f32_matmul_transposed(&command, &ctx.device, &input.buffer, input.rows, input.cols, weight, out_cols, &output.buffer)?;
    let shape = format!("input=[{},{}],weight=[{out_cols},{}]", input.rows, input.cols, input.cols);
    ctx.commit_and_wait_profiled(&command, "mps_matrix_multiplication_f32", &shape, input.buffer.length() + weight.length(), output.buffer.length());
    Ok(output)
}

#[cfg(test)]
mod gemm_simd64_tests {
    use super::*;

    #[test]
    fn gemm_simd64_matches_cpu() {
        let ctx = MetalContext::new_default().unwrap();
        let (m, k, n) = (256usize, 512, 640);
        let input_values: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let weight_values: Vec<f32> = (0..n * k).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();
        let input_bytes: Vec<u8> = input_values.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect();
        let weight_bytes: Vec<u8> = weight_values.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect();
        let input = ctx.shared_buffer(&input_bytes);
        let weight = ctx.shared_buffer(&weight_bytes);
        let output = ctx.shared_buffer_zeros(m * n * 4);
        let pipeline = ctx.pipeline("gemm_simd64_f16").unwrap();
        let (in_cols, out_cols) = (k as u32, n as u32);
        let encode = |big: &metal::Buffer, bweight: &metal::Buffer, bout: &metal::Buffer, big_m: u32, big_n: u32, command: &metal::CommandBuffer| {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(big), 0);
            encoder.set_buffer(1, Some(bweight), 0);
            encoder.set_buffer(2, Some(bout), 0);
            set_bytes(&encoder, 3, &big_m);
            set_bytes(&encoder, 4, &in_cols);
            set_bytes(&encoder, 5, &out_cols);
            encoder.dispatch_thread_groups(MTLSize::new((big_n / 64) as u64, (big_m / 64) as u64, 1), MTLSize::new(128, 1, 1));
            encoder.end_encoding();
        };
        {
            let command = ctx.command_buffer();
            encode(&input, &weight, &output, m as u32, n as u32, &command);
            ctx.commit_and_wait(&command);
        }
        let output_values = {
            let ptr = output.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, m * n) }.to_vec()
        };
        for row in 0..m {
            for column in 0..n {
                let expected: f32 = (0..k).map(|index| input_values[row * k + index] * weight_values[column * k + index]).sum();
                let got = output_values[row * n + column];
                assert!((got - expected).abs() < 5.0e-2 * (1.0 + expected.abs()), "row={row} col={column}: {got} vs {expected}");
            }
        }
        // 恒等权重脉冲对照:验证 simdgroup MMA 的基本行为
        let (m2, k2, n2) = (64usize, 256usize, 64usize);
        let mut input2 = vec![0.0f32; m2 * k2];
        for row in 0..m2 {
            input2[row * k2] = 1.0;
        }
        let input2_bytes: Vec<u8> = input2.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect();
        let weight2_bytes: Vec<u8> = vec![1.0f32; n2 * k2].iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect();
        let input2_buf = ctx.shared_buffer(&input2_bytes);
        let weight2_buf = ctx.shared_buffer(&weight2_bytes);
        let output2 = ctx.shared_buffer_zeros(m2 * n2 * 4);
        let (in_cols2, out_cols2) = (k2 as u32, n2 as u32);
        let encode2 = |command: &metal::CommandBuffer| {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&input2_buf), 0);
            encoder.set_buffer(1, Some(&weight2_buf), 0);
            encoder.set_buffer(2, Some(&output2), 0);
            set_bytes(&encoder, 3, &(m2 as u32));
            set_bytes(&encoder, 4, &in_cols2);
            set_bytes(&encoder, 5, &out_cols2);
            encoder.dispatch_thread_groups(MTLSize::new((n2 / 64) as u64, (m2 / 64) as u64, 1), MTLSize::new(128, 1, 1));
            encoder.end_encoding();
        };
        {
            let command = ctx.command_buffer();
            encode2(&command);
            ctx.commit_and_wait(&command);
        }
        let output2_values = unsafe { std::slice::from_raw_parts(output2.contents() as *const f32, m2 * n2) }.to_vec();
        // 脉冲:input[row,0]=1 → output[row,col] = weight[col][0] = 1.0。
        for (index, value) in output2_values.into_iter().enumerate() {
            assert_eq!(value, 1.0, "脉冲输出 index={index}");
        }
    }
}
