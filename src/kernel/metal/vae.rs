//! VAE Metal 算子包装：GroupNorm + SiLU + AdaLN modulation + timestep embedding + Conv3D/PixelShuffle + 音频 VAE。
//!
//! 数值正确性由 src/kernel/cpu/vae.rs 的 reference 验证。

/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: vae_group_norm_f16, vae_silu_f16, vae_adaln_modulate_f16, vae_timestep_embedding_f16, conv3d_f16, pixel_shuffle_f16
pub const SHADERS: &str = r#"
kernel void vae_group_norm_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    constant uint &channels [[buffer(4)]],
    constant uint &spatial [[buffer(5)]],
    constant uint &num_groups [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    // gid 索引空间 = channels × spatial
    if (gid >= channels * spatial) return;
    uint channel = gid / spatial;
    uint group = channel / (channels / num_groups);
    uint group_start = group * (channels / num_groups) * spatial;
    uint group_len = (channels / num_groups) * spatial;

    // 两遍:先算 mean/var(用 threadgroup 共享),简化版直接全局遍历
    // 注意:这个 kernel 是 elementwise 的,每线程独立遍历组数据(非最优,验证用)
    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (uint i = 0; i < group_len; ++i) {
        float v = float(input[group_start + i]);
        sum += v;
        sq_sum += v * v;
    }
    float mean = sum / float(group_len);
    float var = sq_sum / float(group_len) - mean * mean;
    float rstd = rsqrt(var + eps);

    float val = float(input[gid]);
    output[gid] = half((val - mean) * rstd * float(scale[channel]) + float(bias[channel]));
}
kernel void vae_silu_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = float(input[gid]);
    output[gid] = half(x / (1.0f + exp(-x)));
}
kernel void vae_adaln_modulate_f16(
    device const half *input [[buffer(0)]],
    device const half *shift [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &modulation_rows [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint modulation_index = modulation_rows == 1 ? gid % columns : gid;
    float x = float(input[gid]);
    output[gid] = half(x * (1.0f + float(scale[modulation_index])) + float(shift[modulation_index]));
}
kernel void vae_timestep_embedding_f16(
    device const half *timesteps [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &batch [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // 每线程算一对 (cos, sin),dim 必须是偶数
    uint half_dim = dim / 2;
    if (gid >= batch * half_dim) return;
    uint row = gid / half_dim;
    uint index = gid % half_dim;
    float freq = exp(-log(10000.0f) * float(index) / float(half_dim));
    float arg = float(timesteps[row]) * freq;
    output[row * dim + index] = half(cos(arg));
    output[row * dim + half_dim + index] = half(sin(arg));
}
kernel void conv3d_f16(
    device const half *input [[buffer(0)]],     // [in_channels, in_depth, in_height, in_width]
    device half *output [[buffer(1)]],           // [out_channels, out_depth, out_height, out_width]
    device const half *weight [[buffer(2)]],     // [out_channels, in_channels, kd, kh, kw]
    device const half *bias [[buffer(3)]],       // [out_channels] 或空
    constant uint *dims [[buffer(4)]],           // [..., has_bias, causal, pad_d, pad_h, pad_w]
    uint gid [[thread_position_in_grid]])
{
    uint in_ch = dims[0];
    uint out_ch = dims[1];
    uint in_d = dims[2];
    uint in_h = dims[3];
    uint in_w = dims[4];
    uint kd = dims[5];
    uint kh = dims[6];
    uint kw = dims[7];
    uint sd = dims[8];
    uint sh = dims[9];
    uint sw = dims[10];
    uint has_bias = dims[11];
    uint causal = dims[12];

    // 计算 output shape
    uint pad_d = dims[13];
    uint pad_h = dims[14];
    uint pad_w = dims[15];
    uint out_d = (in_d + (causal ? pad_d : 2 * pad_d) - kd) / sd + 1;
    uint out_h = (in_h + 2 * pad_h - kh) / sh + 1;
    uint out_w = (in_w + 2 * pad_w - kw) / sw + 1;

    uint total = out_ch * out_d * out_h * out_w;
    if (gid >= total) return;

    // 解码 gid → (oc, od, oh, ow)
    uint ow = gid % out_w;
    uint rem = gid / out_w;
    uint oh = rem % out_h;
    rem /= out_h;
    uint od = rem % out_d;
    uint oc = rem / out_d;

    float acc = 0.0f;

    for (uint ic = 0; ic < in_ch; ++ic) {
        for (uint fd = 0; fd < kd; ++fd) {
            int id = int(od * sd) + int(fd) - int(pad_d);
            if (causal && id < 0) continue;  // causal: 前端裁剪
            if (!causal && (id < 0 || id >= int(in_d))) continue;
            if (id >= int(in_d)) continue;

            for (uint fh = 0; fh < kh; ++fh) {
                int ih = int(oh * sh) + int(fh) - int(pad_h);
                if (ih < 0 || ih >= int(in_h)) continue;

                for (uint fw = 0; fw < kw; ++fw) {
                    int iw = int(ow * sw) + int(fw) - int(pad_w);
                    if (iw < 0 || iw >= int(in_w)) continue;

                    uint w_idx = ((oc * in_ch + ic) * kd + fd) * kh * kw + fh * kw + fw;
                    uint i_idx = (ic * in_d + uint(id)) * in_h * in_w + uint(ih) * in_w + uint(iw);
                    acc += float(input[i_idx]) * float(weight[w_idx]);
                }
            }
        }
    }

    if (has_bias) {
        acc += float(bias[oc]);
    }
    output[gid] = half(acc);
}
kernel void pixel_shuffle_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint *dims [[buffer(2)]],  // [channels, height, width, upscale]
    uint gid [[thread_position_in_grid]])
{
    uint channels = dims[0];
    uint height = dims[1];
    uint width = dims[2];
    uint r = dims[3];

    uint out_h = height * r;
    uint out_w = width * r;
    uint total = channels * out_h * out_w;
    if (gid >= total) return;

    uint ow = gid % out_w;
    uint rem = gid / out_w;
    uint oh = rem % out_h;
    uint c = rem / out_h;

    // 输入索引: [c * r^2 + (oh%r) * r + (ow%r), oh/r, ow/r]
    uint ih = oh / r;
    uint iw = ow / r;
    uint sub_r = oh % r;
    uint sub_c = ow % r;
    uint in_channel = c * r * r + sub_r * r + sub_c;
    uint in_idx = (in_channel * height + ih) * width + iw;

    output[gid] = input[in_idx];
}

// 音频 VAE weight_norm: output[i] = (weight_g[i] / max(||weight_v[i,:]||, 1e-12)) * weight_v[i, j]
// has_g=0 时直接复制 weight_v(用于不带 weight-norm 的情形)。
// 1 threadgroup/row, threads 协同 reduce 列平方和。
kernel void weight_norm_f16(
    device const half *weight_g [[buffer(0)]],
    device const half *weight_v [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &has_g [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float partial[256];
    float local = 0.0;
    for (uint c = lid; c < columns; c += lsize) {
        float v = float(weight_v[row * columns + c]);
        local += v * v;
    }
    partial[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        float norm = sqrt(partial[0]);
        float factor = (has_g != 0) ? float(weight_g[row]) / max(norm, 1.0e-12) : 1.0;
        partial[0] = factor;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float factor = partial[0];
    for (uint c = lid; c < columns; c += lsize) {
        output[row * columns + c] = half(float(weight_v[row * columns + c]) * factor);
    }
}

// 音频 VAE conv1d:
//   input 形状 [batch, input_channels, input_length] 平铺为 batch*input_channels 行 × input_length 列。
//   weight_norm 后的 weight_v 是 [output_channels, input_channels*kernel]。
//   has_b=0 时 bias 为零。
//   1 threadgroup/(item, out_channel),threads 协同 output_step。
//   dims: [batch, input_channels, output_channels, input_length, k_size, dilation, padding,
//          output_length, threads_per_out, has_b]
kernel void conv1d_f16_coop(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    uint batch = dims[0];
    uint input_channels = dims[1];
    uint output_channels = dims[2];
    uint input_length = dims[3];
    uint k_size = dims[4];
    uint dilation = dims[5];
    uint padding = dims[6];
    uint output_length = dims[7];
    uint threads_per_out = dims[8];
    uint has_b = dims[9];
    uint total = batch * output_channels;
    if (gid >= total) return;
    uint item = gid / output_channels;
    uint out_channel = gid % output_channels;
    for (uint out_step = lid; out_step < output_length; out_step += threads_per_out) {
        float sum = (has_b != 0) ? float(bias[out_channel]) : 0.0;
        for (uint in_channel = 0; in_channel < input_channels; in_channel++) {
            uint row_base = (item * input_channels + in_channel) * input_length;
            for (uint tap = 0; tap < k_size; tap++) {
                uint source = out_step + tap * dilation;
                if (source >= padding && source - padding < input_length) {
                    uint idx = row_base + source - padding;
                    float w = float(weight[(out_channel * input_channels + in_channel) * k_size + tap]);
                    sum += w * float(input[idx]);
                }
            }
        }
        output[(item * output_channels + out_channel) * output_length + out_step] = half(sum);
    }
}

// 音频 VAE conv_transpose1d:
//   weight 形状 [input_channels, output_channels * kernel]。
//   output_length = (input_length - 1) * stride + kernel - 2 * padding。
//   1 threadgroup/(item, out_channel),threads 协同 output_step。
//   weight[in_channel, out_channel, kernel] 按 row-major。
//   conv_transpose1d 必带 bias(按 CPU 调用)。
//   dims: [batch, input_channels, output_channels, input_length, k_size, stride, padding,
//          output_length, threads_per_out]
kernel void conv_transpose1d_f16_coop(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{
    uint batch = dims[0];
    uint input_channels = dims[1];
    uint output_channels = dims[2];
    uint input_length = dims[3];
    uint k_size = dims[4];
    uint stride = dims[5];
    uint padding = dims[6];
    uint output_length = dims[7];
    uint threads_per_out = dims[8];
    uint total = batch * output_channels;
    if (gid >= total) return;
    uint item = gid / output_channels;
    uint out_channel = gid % output_channels;
    for (uint out_step = lid; out_step < output_length; out_step += threads_per_out) {
        float sum = float(bias[out_channel]);
        for (uint in_channel = 0; in_channel < input_channels; in_channel++) {
            uint row_base = (item * input_channels + in_channel) * input_length;
            for (uint tap = 0; tap < k_size; tap++) {
                uint shifted = out_step + padding;
                if (shifted >= tap && (shifted - tap) % stride == 0) {
                    uint source = (shifted - tap) / stride;
                    if (source < input_length) {
                        uint weight_idx = (in_channel * output_channels + out_channel) * k_size + tap;
                        float w = float(weight[weight_idx]);
                        sum += w * float(input[row_base + source]);
                    }
                }
            }
        }
        output[(item * output_channels + out_channel) * output_length + out_step] = half(sum);
    }
}

// 音频 VAE snake_beta 上采样 2x:
//   输入 [batch, channels, length], 输出 [batch, channels, length*2]。
//   对每个 (batch, channel, step_long):
//     raw = step_long + crop
//     value = Σ over tap: up_filter[tap] * input[source] * 2  if raw >= tap && (raw-tap)%2 == 0 && padded<length+2*pad
//       where padded = (raw-tap)/2, source = clamp(padded - pad, 0, length-1)
//   然后 output[step_long] = value + sin(value*a)^2 / b
//   其中 a = exp(alpha[channel]), b = max(exp(beta[channel]), 1e-9)。
//   1 thread = 1 (batch, channel, step_long)。
kernel void snake_beta_upsample_f16(
    device const half *input [[buffer(0)]],
    device const half *alpha [[buffer(1)]],
    device const half *beta [[buffer(2)]],
    device const half *up_filter [[buffer(3)]],
    device const half *down_filter [[buffer(4)]],  // unused for upsample
    device half *output [[buffer(5)]],
    constant uint &batch [[buffer(6)]],
    constant uint &channels [[buffer(7)]],
    constant uint &input_length [[buffer(8)]],
    constant uint &filter [[buffer(9)]],
    constant uint &output_length [[buffer(10)]],
    constant uint &pad [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batch * channels * output_length;
    if (gid >= total) return;
    uint item = gid / (channels * output_length);
    uint rem = gid % (channels * output_length);
    uint channel = rem / output_length;
    uint step_long = rem % output_length;
    uint crop = pad * 2 + (filter - 2) / 2;
    uint raw = step_long + crop;
    float value = 0.0;
    for (uint tap = 0; tap < filter; tap++) {
        if (raw >= tap && (raw - tap) % 2 == 0) {
            uint padded = (raw - tap) / 2;
            if (padded < input_length + 2 * pad) {
                uint source = (padded > pad) ? padded - pad : 0;
                source = min(source, input_length - 1);
                value += float(input[(item * channels + channel) * input_length + source]) * float(up_filter[tap]) * 2.0;
            }
        }
    }
    float a = exp(float(alpha[channel]));
    float b = max(exp(float(beta[channel])), 1.0e-9);
    float sn = sin(value * a);
    output[gid] = half(value + sn * sn / b);
}

// 音频 VAE snake_beta 下采样 0.5x:
//   输入 [batch, channels, long_length], 输出 [batch, channels, length]。
//   对每个 (batch, channel, step):
//     value = Σ over tap: down_filter[tap] * input[clamp((step*2+tap) - left, 0, long_length-1)]
//     output[step] = value
kernel void snake_beta_downsample_f16(
    device const half *input [[buffer(0)]],
    device const half *alpha [[buffer(1)]],
    device const half *beta [[buffer(2)]],
    device const half *up_filter [[buffer(3)]],  // unused for downsample
    device const half *down_filter [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &batch [[buffer(6)]],
    constant uint &channels [[buffer(7)]],
    constant uint &long_length [[buffer(8)]],
    constant uint &filter [[buffer(9)]],
    constant uint &length [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batch * channels * length;
    if (gid >= total) return;
    uint item = gid / (channels * length);
    uint rem = gid % (channels * length);
    uint channel = rem / length;
    uint step = rem % length;
    uint row_base = (item * channels + channel) * long_length;
    uint left = (filter - 1) / 2;
    float value = 0.0;
    for (uint tap = 0; tap < filter; tap++) {
        int idx_signed = (int)(step * 2 + tap) - (int)left;
        uint clamped;
        if (idx_signed <= 0) clamped = 0u;
        else if ((uint)idx_signed >= long_length) clamped = long_length - 1;
        else clamped = (uint)idx_signed;
        value += float(input[row_base + clamped]) * float(down_filter[tap]);
    }
    output[gid] = half(value);
}

// 音频 VAE audio_unpack_affine:
//   input 形状 [batch*time, channels](flat: input[(item*time + step)*channels + channel])
//   output 形状 [batch*channels, time](flat: output[(item*channels + channel)*time + step])
//   per-channel affine: scale[channel] / bias[channel]。
//   1 thread = 1 output element。
kernel void audio_unpack_affine_f16(
    device const half *input [[buffer(0)]],
    device const half *scale [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint *dims [[buffer(4)]],  // [batch, time, channels]
    uint gid [[thread_position_in_grid]])
{
    uint batch = dims[0];
    uint time = dims[1];
    uint channels = dims[2];
    uint total = batch * channels * time;
    if (gid >= total) return;
    uint item = gid / (channels * time);
    uint rem = gid % (channels * time);
    uint channel = rem / time;
    uint step = rem % time;
    float v = float(input[(item * time + step) * channels + channel]) * float(scale[channel]) + float(bias[channel]);
    output[gid] = half(v);
}

// 音频 VAE scale_tensor:output[i] = input[i] * scale。
kernel void scale_tensor_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    output[gid] = half(float(input[gid]) * scale);
}

// 音频 VAE tanh:output[i] = tanh(input[i])。
kernel void vae_tanh_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    output[gid] = half(tanh(float(input[gid])));
}
"#;

use super::{MTLSize, MetalContext, MetalTensor, launch_1d, launch_rows_with_pipeline, set_bytes, validate_u32};
use crate::vae::{Conv3dSpec, PixelShuffleSpec};

/// VAE GroupNorm：按通道分组归一化。
///
/// `input`: `[channels, spatial]` f16。
/// `scale`/`bias`: `[channels]` f16。
#[allow(clippy::too_many_arguments)]
pub fn vae_group_norm_tensor(ctx: &MetalContext, input: &MetalTensor, scale: &MetalTensor, bias: &MetalTensor, channels: usize, spatial: usize, num_groups: usize, eps: f32) -> Result<MetalTensor, String> {
    validate_u32("VAE group_norm count", channels * spatial)?;
    let ch = validate_u32("VAE group_norm channels", channels)?;
    let sp = validate_u32("VAE group_norm spatial", spatial)?;
    let ng = validate_u32("VAE group_norm num_groups", num_groups)?;
    if !channels.is_multiple_of(num_groups) {
        return Err(format!("VAE group_norm channels {channels} 不能被 num_groups {num_groups} 整除"));
    }
    let output = ctx.tensor_zeros(input.rows, input.cols);
    let shape = format!("channels={channels},spatial={spatial},groups={num_groups}");
    launch_1d(ctx, "vae_group_norm_f16", &shape, channels * spatial, input.buffer.length() + scale.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        encoder.set_buffer(2, Some(&scale.buffer), 0);
        encoder.set_buffer(3, Some(&bias.buffer), 0);
        set_bytes(encoder, 4, &ch);
        set_bytes(encoder, 5, &sp);
        set_bytes(encoder, 6, &ng);
        set_bytes(encoder, 7, &eps);
    })?;
    Ok(output)
}

/// VAE SiLU：`x * sigmoid(x)`，elementwise。
pub fn vae_silu_tensor(ctx: &MetalContext, input: &MetalTensor) -> Result<MetalTensor, String> {
    let count = validate_u32("VAE silu count", input.len())?;
    let output = ctx.tensor_zeros(input.rows, input.cols);
    let shape = format!("elements={}", input.len());
    launch_1d(ctx, "vae_silu_f16", &shape, input.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count);
    })?;
    Ok(output)
}

/// VAE AdaLN modulation：`out = x * (1 + scale) + shift`。
pub fn vae_adaln_modulate_tensor(ctx: &MetalContext, input: &MetalTensor, shift: &MetalTensor, scale: &MetalTensor) -> Result<MetalTensor, String> {
    if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows || (shift.rows != 1 && shift.rows != input.rows) {
        return Err(format!("VAE AdaLN shape 不兼容: input=[{},{}] shift=[{},{}] scale=[{},{}]", input.rows, input.cols, shift.rows, shift.cols, scale.rows, scale.cols,));
    }
    let count = validate_u32("VAE adaln count", input.len())?;
    let columns = validate_u32("VAE adaln columns", input.cols)?;
    let modulation_rows = validate_u32("VAE adaln modulation rows", shift.rows)?;
    let output = ctx.tensor_zeros(input.rows, input.cols);
    let shape = format!("elements={}", input.len());
    launch_1d(ctx, "vae_adaln_modulate_f16", &shape, input.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&shift.buffer), 0);
        encoder.set_buffer(2, Some(&scale.buffer), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        set_bytes(encoder, 4, &count);
        set_bytes(encoder, 5, &columns);
        set_bytes(encoder, 6, &modulation_rows);
    })?;
    Ok(output)
}

/// Timestep sinusoidal embedding。
///
/// 给标量 `timestep` 生成 `[1, dim]` 的 f16 tensor（前半 cos、后半 sin)。
pub fn vae_timestep_embedding_tensor(ctx: &MetalContext, timesteps: &[f32], dim: usize) -> Result<MetalTensor, String> {
    if timesteps.is_empty() || dim == 0 || !dim.is_multiple_of(2) {
        return Err(format!("timestep embedding batch={} dim={dim}，要求 batch 非零且 dim 为非零偶数", timesteps.len()));
    }
    let timestep_tensor = ctx.tensor_from_f32(timesteps, timesteps.len(), 1)?;
    let batch = validate_u32("timestep batch", timesteps.len())?;
    let d = validate_u32("timestep dim", dim)?;
    let output = ctx.tensor_zeros(timesteps.len(), dim);
    let shape = format!("batch={},dim={dim}", timesteps.len());
    launch_1d(ctx, "vae_timestep_embedding_f16", &shape, timesteps.len() * dim / 2, timestep_tensor.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&timestep_tensor.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &batch);
        set_bytes(encoder, 3, &d);
    })?;
    Ok(output)
}

/// Conv3D：时空 3D 卷积（VAE 核心算子）。
///
/// 输入/输出/权重都是 `[channels, depth, height, width]` 的 f16 MetalTensor（展平为 1D）。
/// 权重布局：`[out_channels, in_channels, kd, kh, kw]`。
pub fn conv3d_tensor(ctx: &MetalContext, input: &MetalTensor, weight: &MetalTensor, bias: Option<&MetalTensor>, spec: &Conv3dSpec) -> Result<MetalTensor, String> {
    let [in_depth, in_height, in_width] = spec.input_shape;
    let [kd, kh, kw] = spec.kernel;
    let [sd, sh, sw] = spec.stride;
    let [pad_d, pad_h, pad_w] = spec.padding;
    let [out_depth, out_height, out_width] = spec.output_shape()?;
    let total = spec.output_channels * out_depth * out_height * out_width;
    let expected_input = spec.input_channels * spec.input_spatial()?;
    let expected_weight = spec.output_channels * spec.input_channels * kd * kh * kw;
    if input.len() != expected_input || weight.len() != expected_weight || bias.is_some_and(|bias| bias.len() != spec.output_channels) {
        return Err(format!("Metal Conv3D tensor 长度不兼容: input={} weight={} bias={:?} spec={spec:?}", input.len(), weight.len(), bias.map(MetalTensor::len)));
    }

    let dims: [u32; 16] = [
        validate_u32("Conv3D input channels", spec.input_channels)?,
        validate_u32("Conv3D output channels", spec.output_channels)?,
        validate_u32("Conv3D input depth", in_depth)?,
        validate_u32("Conv3D input height", in_height)?,
        validate_u32("Conv3D input width", in_width)?,
        validate_u32("Conv3D kernel depth", kd)?,
        validate_u32("Conv3D kernel height", kh)?,
        validate_u32("Conv3D kernel width", kw)?,
        validate_u32("Conv3D stride depth", sd)?,
        validate_u32("Conv3D stride height", sh)?,
        validate_u32("Conv3D stride width", sw)?,
        if bias.is_some() { 1 } else { 0 },
        if spec.causal { 1 } else { 0 },
        validate_u32("Conv3D padding depth", pad_d)?,
        validate_u32("Conv3D padding height", pad_h)?,
        validate_u32("Conv3D padding width", pad_w)?,
    ];

    // 输出用 1D 布局 [out_channels * out_depth * out_height * out_width]
    let output = ctx.shared_buffer_zeros(total * 2); // f16 = 2 bytes
    // MetalTensor 需要 rows/cols，用 total × 1
    let output_tensor = MetalTensor::new(output, spec.output_channels, out_depth * out_height * out_width);

    let command = ctx.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    let pipeline = ctx.pipeline("conv3d_f16")?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&output_tensor.buffer), 0);
    encoder.set_buffer(2, Some(&weight.buffer), 0);
    if let Some(b) = bias {
        encoder.set_buffer(3, Some(&b.buffer), 0);
    } else {
        encoder.set_buffer(3, Some(&weight.buffer), 0); // 占位，has_bias=0 时不读
    }
    // dims 共 16 项，kernel 会读到 dims[13..15] 的 padding，必须完整发送。
    encoder.set_bytes(4, std::mem::size_of_val(&dims) as u64, dims.as_ptr() as *const _);

    let threads = total.min(256);
    let groups = total.div_ceil(threads);
    encoder.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();

    let shape = format!("in_ch={},out_ch={},dhw={in_depth}x{in_height}x{in_width},k={kd}x{kh}x{kw}", spec.input_channels, spec.output_channels);
    ctx.commit_and_wait_profiled(&command, "conv3d_f16", &shape, input.buffer.length() + weight.buffer.length(), output_tensor.buffer.length());
    Ok(output_tensor)
}

/// Pixel Shuffle：空间上采样。
///
/// 输入 `[channels × r², height, width]` → 输出 `[channels, height × r, width × r]`。
pub fn pixel_shuffle_tensor(ctx: &MetalContext, input: &MetalTensor, spec: &PixelShuffleSpec) -> Result<MetalTensor, String> {
    spec.validate()?;
    let input_channels = spec.channels * spec.upscale * spec.upscale;
    if input.rows != input_channels || input.cols != spec.height * spec.width {
        return Err(format!("Metal pixel shuffle input=[{},{}]，期望 [{input_channels},{}]", input.rows, input.cols, spec.height * spec.width));
    }
    let out_h = spec.height * spec.upscale;
    let out_w = spec.width * spec.upscale;
    let total = spec.channels * out_h * out_w;

    let dims: [u32; 4] = [validate_u32("pixel shuffle channels", spec.channels)?, validate_u32("pixel shuffle height", spec.height)?, validate_u32("pixel shuffle width", spec.width)?, validate_u32("pixel shuffle upscale", spec.upscale)?];
    let output = ctx.shared_buffer_zeros(total * 2); // f16 = 2 bytes
    let output_tensor = MetalTensor::new(output, spec.channels, out_h * out_w);

    let command = ctx.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    let pipeline = ctx.pipeline("pixel_shuffle_f16")?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&output_tensor.buffer), 0);
    encoder.set_bytes(2, std::mem::size_of::<[u32; 4]>() as u64, dims.as_ptr() as *const _);

    let threads = total.min(256);
    let groups = total.div_ceil(threads);
    encoder.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();

    let shape = format!("ch={},hw={}x{}->{out_h}x{out_w},r={}", spec.channels, spec.height, spec.width, spec.upscale);
    ctx.commit_and_wait_profiled(&command, "pixel_shuffle_f16", &shape, input.buffer.length(), output_tensor.buffer.length());
    Ok(output_tensor)
}

// ==== 音频 VAE（H3 AudioVAE BigVGAN-style:weight_norm + conv1d + snake_beta 上/下采样） ====

/// 音频 VAE weight_norm:`output[i, j] = (weight_g[i] / max(||weight_v[i,:]||, 1e-12)) * weight_v[i, j]`。
/// `weight_g=None` 时直接复制 weight_v。
pub fn weight_norm_f16_tensor(ctx: &MetalContext, weight_g: Option<&MetalTensor>, weight_v: &MetalTensor, rows: usize, columns: usize) -> Result<MetalTensor, String> {
    if weight_v.len() != rows * columns {
        return Err(format!("VAE weight_norm weight_v={} ≠ rows*columns={}", weight_v.len(), rows * columns));
    }
    if let Some(g) = weight_g
        && g.len() != rows
    {
        return Err(format!("VAE weight_norm weight_g={} ≠ rows={}", g.len(), rows));
    }
    let output = ctx.tensor_zeros(rows, columns);
    let has_g: u32 = if weight_g.is_some() { 1 } else { 0 };
    let dummy_g: MetalTensor = ctx.tensor_zeros(1, 1);
    let g_buffer = weight_g.map(|g| &g.buffer).unwrap_or(&dummy_g.buffer);
    let rows_u32 = validate_u32("weight_norm rows", rows)?;
    let columns_u32 = validate_u32("weight_norm columns", columns)?;
    let threads = columns.clamp(1, 256).next_power_of_two();
    launch_rows_with_pipeline(ctx, "weight_norm_f16", rows, threads, weight_v.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(g_buffer), 0);
        encoder.set_buffer(1, Some(&weight_v.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &rows_u32);
        set_bytes(encoder, 4, &columns_u32);
        set_bytes(encoder, 5, &has_g);
    })?;
    Ok(output)
}

/// 音频 VAE Conv1D:`output[item, out_channel, out_step] = bias[out_channel]
///   + Σ_{in_channel, tap} weight[oc, ic, tap] * input[item, ic, out_step*dilation + tap*dilation - padding]`。
#[allow(clippy::too_many_arguments)]
pub fn conv1d_f16_tensor(
    ctx: &MetalContext,
    input: &MetalTensor,
    weight: &MetalTensor,
    bias: Option<&MetalTensor>,
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
) -> Result<MetalTensor, String> {
    let input_length = input.cols;
    if input.rows != batch * input_channels {
        return Err(format!("VAE conv1d input.rows={} ≠ batch*input_channels={}", input.rows, batch * input_channels));
    }
    if weight.len() != output_channels * input_channels * kernel {
        return Err(format!("VAE conv1d weight={} ≠ out*in*kernel={}", weight.len(), output_channels * input_channels * kernel));
    }
    if let Some(b) = bias
        && b.len() != output_channels
    {
        return Err(format!("VAE conv1d bias={} ≠ out_channels={}", b.len(), output_channels));
    }
    if kernel == 0 || dilation == 0 {
        return Err(format!("VAE conv1d kernel={kernel} dilation={dilation} 必须非零"));
    }
    let effective = dilation * (kernel - 1) + 1;
    // 输出长度 ≤0 时 usize 减法下溢，前置拒绝。
    if input_length + 2 * padding < effective {
        return Err(format!("VAE conv1d 输出长度非正: input_length={input_length} padding={padding} kernel={kernel} dilation={dilation}"));
    }
    let output_length = input_length + 2 * padding - effective + 1;
    let output = ctx.tensor_zeros(batch * output_channels, output_length);
    let has_b: u32 = if bias.is_some() { 1 } else { 0 };
    let dummy_b = ctx.tensor_zeros(1, 1);
    let b_buffer = bias.map(|b| &b.buffer).unwrap_or(&dummy_b.buffer);
    // kernel 按 dims[8]=threads_per_out 步进，必须与 launch_rows_with_pipeline 的 pow2 dispatch 宽度一致。
    let actual_threads = output_length.clamp(1, 256).next_power_of_two();
    let dims: [u32; 10] = [
        validate_u32("conv1d batch", batch)?,
        validate_u32("conv1d input_channels", input_channels)?,
        validate_u32("conv1d output_channels", output_channels)?,
        validate_u32("conv1d input_length", input_length)?,
        validate_u32("conv1d kernel", kernel)?,
        validate_u32("conv1d dilation", dilation)?,
        validate_u32("conv1d padding", padding)?,
        validate_u32("conv1d output_length", output_length)?,
        validate_u32("conv1d actual_threads", actual_threads)?,
        has_b,
    ];
    launch_rows_with_pipeline(ctx, "conv1d_f16_coop", batch * output_channels, output_length, input.buffer.length() + weight.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&weight.buffer), 0);
        encoder.set_buffer(2, Some(b_buffer), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        encoder.set_bytes(4, std::mem::size_of::<[u32; 10]>() as u64, dims.as_ptr() as *const _);
    })?;
    Ok(output)
}

/// 音频 VAE ConvTranspose1D:`output[item, out_channel, out_step] = bias[out_channel]
///   + Σ_{in_channel, tap} weight[ic, oc, tap] * input[item, ic, (out_step+padding-tap)/stride]`。
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_f16_tensor(
    ctx: &MetalContext,
    input: &MetalTensor,
    weight: &MetalTensor,
    bias: &MetalTensor,
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<MetalTensor, String> {
    let input_length = input.cols;
    if input.rows != batch * input_channels {
        return Err(format!("VAE conv_transpose1d input.rows={} ≠ batch*input_channels={}", input.rows, batch * input_channels));
    }
    if weight.len() != input_channels * output_channels * kernel {
        return Err(format!("VAE conv_transpose1d weight={} ≠ in*out*kernel={}", weight.len(), input_channels * output_channels * kernel));
    }
    if bias.len() != output_channels {
        return Err(format!("VAE conv_transpose1d bias={} ≠ out_channels={}", bias.len(), output_channels));
    }
    // input_length==0 或 kernel < 2*padding 时 usize 减法下溢，前置拒绝。
    if input_length == 0 || stride == 0 || (input_length - 1) * stride + kernel < 2 * padding + 1 {
        return Err(format!("VAE conv_transpose1d 输出长度非正: input_length={input_length} stride={stride} kernel={kernel} padding={padding}"));
    }
    let output_length = (input_length - 1) * stride + kernel - 2 * padding;
    let output = ctx.tensor_zeros(batch * output_channels, output_length);
    // kernel 按 dims[8]=threads_per_out 步进，必须与 launch_rows_with_pipeline 的 pow2 dispatch 宽度一致。
    let actual_threads = output_length.clamp(1, 256).next_power_of_two();
    let dims: [u32; 9] = [
        validate_u32("conv_transpose1d batch", batch)?,
        validate_u32("conv_transpose1d input_channels", input_channels)?,
        validate_u32("conv_transpose1d output_channels", output_channels)?,
        validate_u32("conv_transpose1d input_length", input_length)?,
        validate_u32("conv_transpose1d kernel", kernel)?,
        validate_u32("conv_transpose1d stride", stride)?,
        validate_u32("conv_transpose1d padding", padding)?,
        validate_u32("conv_transpose1d output_length", output_length)?,
        validate_u32("conv_transpose1d actual_threads", actual_threads)?,
    ];
    launch_rows_with_pipeline(ctx, "conv_transpose1d_f16_coop", batch * output_channels, output_length, input.buffer.length() + weight.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&weight.buffer), 0);
        encoder.set_buffer(2, Some(&bias.buffer), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        encoder.set_bytes(4, std::mem::size_of::<[u32; 9]>() as u64, dims.as_ptr() as *const _);
    })?;
    Ok(output)
}

/// 音频 VAE SnakeBeta 上采样 2x:对每个 (item, channel, step_long),
/// 计算 value = Σ up_filter[tap] * input[source] * 2,然后 value + sin(value*a)^2 / b。
/// `pad = filter/2 - 1`(参考 CPU kernel/cpu/vae.rs)。
#[allow(clippy::too_many_arguments)]
pub fn snake_beta_upsample_f16_tensor(ctx: &MetalContext, input: &MetalTensor, alpha: &MetalTensor, beta: &MetalTensor, up_filter: &MetalTensor, down_filter: &MetalTensor, channels: usize) -> Result<MetalTensor, String> {
    let input_length = input.cols;
    if channels == 0 || !input.rows.is_multiple_of(channels) {
        return Err(format!("VAE snake_beta upsample input.rows={} 不能按 channels={} 拆分", input.rows, channels));
    }
    let batch = input.rows / channels;
    let filter = up_filter.len();
    // filter < 2 时 pad = filter/2 - 1 与 (filter - 2)/2 的 usize 减法下溢。
    if filter < 2 || down_filter.len() != filter {
        return Err(format!("VAE snake_beta upsample filter={} up/down={}/{}，要求 filter ≥ 2 且 up/down 等长", filter, up_filter.len(), down_filter.len()));
    }
    if alpha.len() != channels || beta.len() != channels {
        return Err(format!("VAE snake_beta upsample alpha/beta={}/{} ≠ channels={}", alpha.len(), beta.len(), channels));
    }
    let output_length = input_length * 2;
    let output = ctx.tensor_zeros(batch * channels, output_length);
    let pad = filter / 2 - 1;
    let crop = pad * 2 + (filter - 2) / 2;
    let b_u32 = validate_u32("snake upsample batch", batch)?;
    let c_u32 = validate_u32("snake upsample channels", channels)?;
    let il_u32 = validate_u32("snake upsample input_length", input_length)?;
    let f_u32 = validate_u32("snake upsample filter", filter)?;
    let ol_u32 = validate_u32("snake upsample output_length", output_length)?;
    let p_u32 = validate_u32("snake upsample pad", pad)?;
    let cr_u32 = validate_u32("snake upsample crop", crop)?;
    let _ = cr_u32; // not used by current kernel;保留语义
    launch_1d(
        ctx,
        "snake_beta_upsample_f16",
        &format!("b={batch}c={channels}il={input_length}->ol={output_length}"),
        batch * channels * output_length,
        input.buffer.length() + alpha.buffer.length() + beta.buffer.length() + up_filter.buffer.length(),
        output.buffer.length(),
        |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&alpha.buffer), 0);
            encoder.set_buffer(2, Some(&beta.buffer), 0);
            encoder.set_buffer(3, Some(&up_filter.buffer), 0);
            encoder.set_buffer(4, Some(&down_filter.buffer), 0);
            encoder.set_buffer(5, Some(&output.buffer), 0);
            set_bytes(encoder, 6, &b_u32);
            set_bytes(encoder, 7, &c_u32);
            set_bytes(encoder, 8, &il_u32);
            set_bytes(encoder, 9, &f_u32);
            set_bytes(encoder, 10, &ol_u32);
            set_bytes(encoder, 11, &p_u32);
        },
    )?;
    Ok(output)
}

/// 音频 VAE SnakeBeta 下采样 0.5x:对每个 (item, channel, step),
/// 计算 value = Σ down_filter[tap] * input[clamp((step*2+tap)-left, 0, long_length-1)],然后 value + sin(value*a)^2 / b。
#[allow(clippy::too_many_arguments)]
pub fn snake_beta_downsample_f16_tensor(ctx: &MetalContext, input: &MetalTensor, alpha: &MetalTensor, beta: &MetalTensor, up_filter: &MetalTensor, down_filter: &MetalTensor, channels: usize) -> Result<MetalTensor, String> {
    let long_length = input.cols;
    if channels == 0 || !input.rows.is_multiple_of(channels) {
        return Err(format!("VAE snake_beta downsample input.rows={} 不能按 channels={} 拆分", input.rows, channels));
    }
    let batch = input.rows / channels;
    let filter = down_filter.len();
    if filter == 0 || up_filter.len() != filter {
        return Err(format!("VAE snake_beta downsample filter={} up/down={}/{}", filter, up_filter.len(), down_filter.len()));
    }
    if alpha.len() != channels || beta.len() != channels {
        return Err(format!("VAE snake_beta downsample alpha/beta={}/{} ≠ channels={}", alpha.len(), beta.len(), channels));
    }
    let length = long_length / 2;
    let output = ctx.tensor_zeros(batch * channels, length);
    let b_u32 = validate_u32("snake downsample batch", batch)?;
    let c_u32 = validate_u32("snake downsample channels", channels)?;
    let ll_u32 = validate_u32("snake downsample long_length", long_length)?;
    let f_u32 = validate_u32("snake downsample filter", filter)?;
    let l_u32 = validate_u32("snake downsample length", length)?;
    launch_1d(
        ctx,
        "snake_beta_downsample_f16",
        &format!("b={batch}c={channels}ll={long_length}->l={length}"),
        batch * channels * length,
        input.buffer.length() + alpha.buffer.length() + beta.buffer.length() + down_filter.buffer.length(),
        output.buffer.length(),
        |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&alpha.buffer), 0);
            encoder.set_buffer(2, Some(&beta.buffer), 0);
            encoder.set_buffer(3, Some(&up_filter.buffer), 0);
            encoder.set_buffer(4, Some(&down_filter.buffer), 0);
            encoder.set_buffer(5, Some(&output.buffer), 0);
            set_bytes(encoder, 6, &b_u32);
            set_bytes(encoder, 7, &c_u32);
            set_bytes(encoder, 8, &ll_u32);
            set_bytes(encoder, 9, &f_u32);
            set_bytes(encoder, 10, &l_u32);
        },
    )?;
    Ok(output)
}

/// 音频 VAE audio_unpack_affine:
///   output[item, channel, step] = input[item, step, channel] * scale[channel] + bias[channel]
///   layout 输入 [batch*time, channels],输出 [batch*channels, time]。
pub fn audio_unpack_affine_f16_tensor(ctx: &MetalContext, input: &MetalTensor, scale: &MetalTensor, bias: &MetalTensor, batch: usize, time: usize, channels: usize) -> Result<MetalTensor, String> {
    if input.rows != batch * time || input.cols != channels {
        return Err(format!("audio_unpack_affine input=[{},{}] 期望 [{},{}]", input.rows, input.cols, batch * time, channels));
    }
    if scale.len() != channels || bias.len() != channels {
        return Err(format!("audio_unpack_affine scale/bias={}/{} ≠ channels={}", scale.len(), bias.len(), channels));
    }
    let output = ctx.tensor_zeros(batch * channels, time);
    let dims: [u32; 3] = [validate_u32("audio_unpack batch", batch)?, validate_u32("audio_unpack time", time)?, validate_u32("audio_unpack channels", channels)?];
    launch_1d(ctx, "audio_unpack_affine_f16", &format!("b={batch}t={time}c={channels}"), batch * channels * time, input.buffer.length() + scale.buffer.length() + bias.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&scale.buffer), 0);
        encoder.set_buffer(2, Some(&bias.buffer), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        encoder.set_bytes(4, std::mem::size_of::<[u32; 3]>() as u64, dims.as_ptr() as *const _);
    })?;
    Ok(output)
}

/// 音频 VAE scale_tensor:output[i] = input[i] * scale。
pub fn scale_tensor_f16_tensor(ctx: &MetalContext, input: &MetalTensor, scale: f32) -> Result<MetalTensor, String> {
    let count = validate_u32("scale_tensor count", input.len())?;
    let output = ctx.tensor_zeros(input.rows, input.cols);
    launch_1d(ctx, "scale_tensor_f16", &format!("scale={scale} count={count}"), input.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count);
        set_bytes(encoder, 3, &scale);
    })?;
    Ok(output)
}

/// 音频 VAE tanh:output[i] = tanh(input[i])。
pub fn vae_tanh_f16_tensor(ctx: &MetalContext, input: &MetalTensor) -> Result<MetalTensor, String> {
    let count = validate_u32("vae tanh count", input.len())?;
    let output = ctx.tensor_zeros(input.rows, input.cols);
    launch_1d(ctx, "vae_tanh_f16", &format!("count={count}"), input.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count);
    })?;
    Ok(output)
}
