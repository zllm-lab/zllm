/// 本模块的 CUDA shader(视频/音频 VaeBackend 算子)。
///
/// 共用 helper 见 `mod.rs` 前导。`mod.rs` 的 `kernels_source()`
/// 把 mod.rs 的 `PREAMBLE_SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
///
/// 数值对齐 `kernel/cpu/vae.rs`(oracle);布局参考 `kernel/rocm/hip/tensor.rs`。
// kernels: rmsnorm_heads_unit_f16, scaled_residual_columns_f16, unpatch_affine_f16,
//   concat_weight_rows_batched_f16, take_rows_batched_f16
pub const SHADERS: &str = r#"
// 每对 (row, head) 一组 block:无仿射 RMSNorm,out = x * rsqrt(mean(x²)+eps)。
// 对称 ROCm `rmsnorm_heads_unit_f32`(hip/tensor.rs:138);oracle cpu/vae.rs:848。
extern "C" __global__ void rmsnorm_heads_unit_f16(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int heads,
    const unsigned int head_dim,
    const float eps)
{
    extern __shared__ float sums[];
    unsigned int block_id = blockIdx.x;
    unsigned int row = block_id / heads;
    unsigned int head = block_id - row * heads;
    unsigned int lane = threadIdx.x;
    unsigned long long base = ((unsigned long long)row * heads + head) * head_dim;
    float sum = 0.0f;
    for (unsigned int c = lane; c < head_dim; c += blockDim.x) {
        float v = __half2float(input[base + c]);
        sum += v * v;
    }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    float inv = rsqrtf(sums[0] / (float)head_dim + eps);
    for (unsigned int c = lane; c < head_dim; c += blockDim.x) {
        output[base + c] = __float2half(__half2float(input[base + c]) * inv);
    }
}

// output[i] = input[i] + update[i] * scale[i % columns]。scale 是 [columns] 向量。
extern "C" __global__ void scaled_residual_columns_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ update,
    const __half * __restrict__ scale,
    __half * __restrict__ output,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    float s = __half2float(scale[id % columns]);
    output[id] = __float2half(__half2float(input[id]) + __half2float(update[id]) * s);
}

// DiT patch 行 → VAE voxel 行重排 + 每通道仿射 value*scale[ch]+bias[ch]。
// 输出驱动(遍历输出元素)。索引与 ROCm hip/tensor.rs:180 / cpu/vae.rs:870 逐位一致。
extern "C" __global__ void unpatch_affine_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ scale,
    const __half * __restrict__ bias,
    __half * __restrict__ output,
    const unsigned int channels,
    const unsigned int width,
    const unsigned int height,
    const unsigned int time,
    const unsigned int patch_w,
    const unsigned int patch_h,
    const unsigned int patch_t,
    const unsigned int count)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int channel = index % channels;
    unsigned int voxel = index / channels;
    unsigned int x = voxel % width;
    unsigned int y = (voxel / width) % height;
    unsigned int t = voxel / (height * width);
    unsigned int grid_h = height / patch_h;
    unsigned int grid_w = width / patch_w;
    unsigned int patch_row = ((t / patch_t) * grid_h + y / patch_h) * grid_w + x / patch_w;
    unsigned int patch_columns = channels * patch_t * patch_h * patch_w;
    unsigned int patch_column = (((channel * patch_t + t % patch_t) * patch_h + y % patch_h) * patch_w) + x % patch_w;
    float v = __half2float(input[(unsigned long long)patch_row * patch_columns + patch_column]);
    float s = __half2float(scale[channel]);
    float b = __half2float(bias[channel]);
    output[index] = __float2half(v * s + b);
}

// 每个 sample 拼接:[activation(input_rows 行) | weight(weight_rows 行)]。
// weight 在所有 sample 间广播(同一组 register/suffix token)。输出按 sample 交错。
extern "C" __global__ void concat_weight_rows_batched_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int input_rows,
    const unsigned int weight_rows,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int column = index % columns;
    unsigned int row = index / columns;
    unsigned int output_rows = input_rows + weight_rows;
    unsigned int local_row = row % output_rows;
    unsigned int sample = row / output_rows;
    if (local_row < input_rows) {
        output[index] = input[((unsigned long long)sample * input_rows + local_row) * columns + column];
    } else {
        output[index] = weight[((unsigned long long)(local_row - input_rows)) * columns + column];
    }
}

// 每个 sample 取前 output_rows 行(丢弃 register/suffix token)。输出按 sample 交错。
extern "C" __global__ void take_rows_batched_f16(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int input_rows,
    const unsigned int output_rows,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int column = index % columns;
    unsigned int row = index / columns;
    unsigned int local_row = row % output_rows;
    unsigned int sample = row / output_rows;
    output[index] = input[((unsigned long long)sample * input_rows + local_row) * columns + column];
}

// —— 音频 VAE 解码算子(M5)。oracle `kernel/cpu/vae.rs`;布局 `[batch*channels, time]`。 ——

// [batch*time, channels] 转置为 [batch*channels, time],并每通道仿射 v*scale[ch]+bias[ch]。
// 对称 ROCm `try_audio_unpack_affine`;oracle cpu/vae.rs:895。
extern "C" __global__ void audio_unpack_affine_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ scale,
    const __half * __restrict__ bias,
    __half * __restrict__ output,
    const unsigned int batch,
    const unsigned int channels,
    const unsigned int time,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int step = id % time;
    unsigned int rest = id / time;
    unsigned int channel = rest % channels;
    unsigned int item = rest / channels;
    float v = __half2float(input[((unsigned long long)item * time + step) * channels + channel]);
    output[id] = __float2half(v * __half2float(scale[channel]) + __half2float(bias[channel]));
}

// 每行 weight-norm:w[r,:] = v[r,:] * g[r] / max(sqrt(sum v[r,:]^2), 1e-12)。一 block 一行。
// 供 conv_transpose1d 预归一化 weight(其 norm 按 in_ch,与输出 out_ch 不对齐)。oracle cpu/vae.rs:910。
extern "C" __global__ void audio_weight_norm_f16(
    const __half * __restrict__ weight_g,
    const __half * __restrict__ weight_v,
    __half * __restrict__ output,
    const unsigned int columns)
{
    extern __shared__ float sums[];
    unsigned int row = blockIdx.x;
    unsigned int lane = threadIdx.x;
    unsigned long long base = (unsigned long long)row * columns;
    float sum = 0.0f;
    for (unsigned int c = lane; c < columns; c += blockDim.x) {
        float v = __half2float(weight_v[base + c]);
        sum += v * v;
    }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    float denom = sqrtf(sums[0]);
    if (denom < 1.0e-12f) denom = 1.0e-12f;
    float factor = __half2float(weight_g[row]) / denom;
    for (unsigned int c = lane; c < columns; c += blockDim.x) {
        output[base + c] = __float2half(__half2float(weight_v[base + c]) * factor);
    }
}

// 1D 卷积(stride=1)。输入 [batch*in_ch, in_len],输出 [batch*out_ch, out_len]。
// weight-norm 内联(每 out_channel 重算,g/bias 必填)。oracle cpu/vae.rs:927。
extern "C" __global__ void audio_conv1d_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ weight_g,
    const __half * __restrict__ weight_v,
    const __half * __restrict__ bias,
    __half * __restrict__ output,
    const unsigned int batch,
    const unsigned int in_ch,
    const unsigned int out_ch,
    const unsigned int in_len,
    const unsigned int out_len,
    const unsigned int kernel,
    const unsigned int dilation,
    const unsigned int padding,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int out_step = id % out_len;
    unsigned int rest = id / out_len;
    unsigned int out_channel = rest % out_ch;
    unsigned int item = rest / out_ch;
    unsigned long long span = (unsigned long long)in_ch * kernel;
    unsigned long long wrow = (unsigned long long)out_channel * span;
    float norm2 = 0.0f;
    for (unsigned long long k = 0; k < span; k++) {
        float v = __half2float(weight_v[wrow + k]);
        norm2 += v * v;
    }
    float denom = sqrtf(norm2);
    if (denom < 1.0e-12f) denom = 1.0e-12f;
    float factor = __half2float(weight_g[out_channel]) / denom;
    float sum = __half2float(bias[out_channel]);
    for (unsigned int in_channel = 0; in_channel < in_ch; in_channel++) {
        unsigned long long in_base = ((unsigned long long)item * in_ch + in_channel) * in_len;
        unsigned long long wbase = wrow + (unsigned long long)in_channel * kernel;
        for (unsigned int tap = 0; tap < kernel; tap++) {
            int src = (int)out_step + (int)tap * (int)dilation - (int)padding;
            if (src >= 0 && src < (int)in_len) {
                sum += __half2float(input[in_base + (unsigned int)src]) * __half2float(weight_v[wbase + tap]) * factor;
            }
        }
    }
    output[id] = __float2half(sum);
}

// 1D 卷积(stride=1),无 weight-norm(factor=1,weight_v 直接用)。供 audio VAE 里
// normalized=false 的普通 conv(h3_vae.rs conv loader:weight_g=None,weight_v 装 {prefix}.weight)。
// bias 由调用方保证(无 bias 时零填)。oracle cpu/vae.rs:927 的 weight_g=None 分支。
extern "C" __global__ void audio_conv1d_plain_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ weight_v,
    const __half * __restrict__ bias,
    __half * __restrict__ output,
    const unsigned int batch,
    const unsigned int in_ch,
    const unsigned int out_ch,
    const unsigned int in_len,
    const unsigned int out_len,
    const unsigned int kernel,
    const unsigned int dilation,
    const unsigned int padding,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int out_step = id % out_len;
    unsigned int rest = id / out_len;
    unsigned int out_channel = rest % out_ch;
    unsigned int item = rest / out_ch;
    unsigned long long span = (unsigned long long)in_ch * kernel;
    unsigned long long wrow = (unsigned long long)out_channel * span;
    float sum = __half2float(bias[out_channel]);
    for (unsigned int in_channel = 0; in_channel < in_ch; in_channel++) {
        unsigned long long in_base = ((unsigned long long)item * in_ch + in_channel) * in_len;
        unsigned long long wbase = wrow + (unsigned long long)in_channel * kernel;
        for (unsigned int tap = 0; tap < kernel; tap++) {
            int src = (int)out_step + (int)tap * (int)dilation - (int)padding;
            if (src >= 0 && src < (int)in_len) {
                sum += __half2float(input[in_base + (unsigned int)src]) * __half2float(weight_v[wbase + tap]);
            }
        }
    }
    output[id] = __float2half(sum);
}

// 1D 转置卷积(读取已 weight-norm 的 weight)。输入 [batch*in_ch, in_len],输出 [batch*out_ch, out_len]。
// weight [in_ch, out_ch*kernel]。oracle cpu/vae.rs:958。
extern "C" __global__ void audio_conv_transpose1d_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ weight,
    const __half * __restrict__ bias,
    __half * __restrict__ output,
    const unsigned int batch,
    const unsigned int in_ch,
    const unsigned int out_ch,
    const unsigned int in_len,
    const unsigned int out_len,
    const unsigned int kernel,
    const unsigned int stride,
    const unsigned int padding,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int out_step = id % out_len;
    unsigned int rest = id / out_len;
    unsigned int out_channel = rest % out_ch;
    unsigned int item = rest / out_ch;
    float sum = __half2float(bias[out_channel]);
    int shifted = (int)out_step + (int)padding;
    for (unsigned int in_channel = 0; in_channel < in_ch; in_channel++) {
        unsigned long long in_base = ((unsigned long long)item * in_ch + in_channel) * in_len;
        unsigned long long wbase = ((unsigned long long)in_channel * out_ch + out_channel) * kernel;
        for (unsigned int tap = 0; tap < kernel; tap++) {
            int diff = shifted - (int)tap;
            if (diff >= 0 && diff % (int)stride == 0) {
                int src = diff / (int)stride;
                if (src < (int)in_len) {
                    sum += __half2float(input[in_base + (unsigned int)src]) * __half2float(weight[wbase + tap]);
                }
            }
        }
    }
    output[id] = __float2half(sum);
}

// SnakeBeta 上采样+激活:[batch*channels, length] -> activated [batch*channels, length*2] (f32)。
// value = Σ input[up_src]*up_filter[tap]*2;out = value + sin(value*a)^2 / b。oracle cpu/vae.rs:995。
extern "C" __global__ void audio_snake_beta_up_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ alpha,
    const __half * __restrict__ beta,
    const __half * __restrict__ up_filter,
    float * __restrict__ activated,
    const unsigned int channels,
    const unsigned int length,
    const unsigned int filter,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int step = id % (length * 2);
    unsigned int rest = id / (length * 2);
    unsigned int channel = rest % channels;
    unsigned int item = rest / channels;
    int pad = (int)filter / 2 - 1;
    int crop = pad * 2 + ((int)filter - 2) / 2;
    int limit = (int)length + 2 * pad;
    float a = expf(__half2float(alpha[channel]));
    float b = expf(__half2float(beta[channel]));
    if (b < 1.0e-9f) b = 1.0e-9f;
    int raw = (int)step + crop;
    unsigned long long in_row = ((unsigned long long)item * channels + channel) * length;
    float value = 0.0f;
    for (unsigned int tap = 0; tap < filter; tap++) {
        int diff = raw - (int)tap;
        if (diff >= 0 && (diff & 1) == 0) {
            int padded = diff / 2;
            if (padded < limit) {
                int source = padded - pad;
                if (source < 0) source = 0;
                if (source > (int)length - 1) source = (int)length - 1;
                value += __half2float(input[in_row + (unsigned int)source]) * __half2float(up_filter[tap]) * 2.0f;
            }
        }
    }
    float s = sinf(value * a);
    activated[id] = value + s * s / b;
}

// SnakeBeta 下采样:activated [batch*channels, length*2] (f32) -> output [batch*channels, length]。
// out = Σ activated[src]*down_filter[tap]。oracle cpu/vae.rs:1017。
extern "C" __global__ void audio_snake_beta_down_f16(
    const float * __restrict__ activated,
    const __half * __restrict__ down_filter,
    __half * __restrict__ output,
    const unsigned int channels,
    const unsigned int length,
    const unsigned int filter,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int step = id % length;
    unsigned int rest = id / length;
    unsigned int channel = rest % channels;
    unsigned int item = rest / channels;
    int left = (int)filter / 2 - ((filter & 1u) == 0u ? 1 : 0);
    int top = (int)(length * 2) - 1;
    unsigned long long act_row = ((unsigned long long)item * channels + channel) * ((unsigned long long)length * 2);
    float value = 0.0f;
    for (unsigned int tap = 0; tap < filter; tap++) {
        int idx = (int)step * 2 + (int)tap - left;
        if (idx < 0) idx = 0;
        if (idx > top) idx = top;
        value += activated[act_row + (unsigned int)idx] * __half2float(down_filter[tap]);
    }
    output[id] = __float2half(value);
}

// elementwise × scalar。
extern "C" __global__ void scale_tensor_f16(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const float scale,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = __float2half(__half2float(input[id]) * scale);
}

// elementwise tanh(最终激活,PCM 有界化)。
extern "C" __global__ void tanh_f16(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = __float2half(tanhf(__half2float(input[id])));
}
"#;

use super::tensor::grid_1d;
use super::{CudaContext, CudaSliceF16, CudaTensor, LaunchConfig, PushKernelArg, THREADS};

/// 无仿射逐头 RMSNorm。输入 [rows, heads*head_dim],每 head 独立归一化。
pub fn rmsnorm_heads_unit_f16(ctx: &CudaContext, input: &CudaTensor, heads: usize, head_dim: usize, eps: f32) -> Result<CudaTensor, String> {
    if heads == 0 || head_dim == 0 || input.cols != heads * head_dim {
        return Err(format!("CUDA rmsnorm_heads_unit input=[{},{}], heads={heads}, head_dim={head_dim}", input.rows, input.cols));
    }
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("rmsnorm_heads_unit_f16")?;
    let cfg = LaunchConfig { grid_dim: ((input.rows * heads) as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (THREADS as usize * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(&output.slice).arg(&(heads as u32)).arg(&(head_dim as u32)).arg(&eps).launch(cfg).map_err(|e| format!("launch rmsnorm_heads_unit_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// output = input + update * scale[col]。scale 为 [columns] 向量(常驻 CudaWeight.data)。
pub fn scaled_residual_columns_f16(ctx: &CudaContext, input: &CudaTensor, update: &CudaTensor, scale: &CudaSliceF16) -> Result<CudaTensor, String> {
    if input.rows != update.rows || input.cols != update.cols || scale.len() != input.cols {
        return Err(format!("CUDA scaled_residual input=[{},{}] update=[{},{}] scale={}", input.rows, input.cols, update.rows, update.cols, scale.len()));
    }
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("scaled_residual_columns_f16")?;
    let cols = input.cols as u32;
    let count = input.len() as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(&update.slice).arg(scale).arg(&output.slice).arg(&cols).arg(&count).launch(grid_1d(input.len())).map_err(|e| format!("launch scaled_residual_columns_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// DiT patch 行 → VAE voxel 行重排 + 仿射。输出 [time*height*width, channels]。
pub fn unpatch_affine_f16(ctx: &CudaContext, input: &CudaTensor, scale: &CudaSliceF16, bias: &CudaSliceF16, shape: [usize; 3], patch: [usize; 3], channels: usize) -> Result<CudaTensor, String> {
    let (time, height, width) = (shape[0], shape[1], shape[2]);
    let (patch_t, patch_h, patch_w) = (patch[0], patch[1], patch[2]);
    if time % patch_t != 0 || height % patch_h != 0 || width % patch_w != 0 {
        return Err(format!("CUDA unpatch_affine shape={shape:?} 非整除 patch={patch:?}"));
    }
    let voxel_rows = time.checked_mul(height).and_then(|v| v.checked_mul(width)).ok_or("CUDA unpatch_affine 维度溢出")?;
    if scale.len() != channels || bias.len() != channels {
        return Err(format!("CUDA unpatch_affine scale={} bias={} channels={channels}", scale.len(), bias.len()));
    }
    let output = ctx.tensor_uninit(voxel_rows, channels)?;
    let func = ctx.function("unpatch_affine_f16")?;
    let count = (voxel_rows * channels) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(scale)
            .arg(bias)
            .arg(&output.slice)
            .arg(&(channels as u32))
            .arg(&(width as u32))
            .arg(&(height as u32))
            .arg(&(time as u32))
            .arg(&(patch_w as u32))
            .arg(&(patch_h as u32))
            .arg(&(patch_t as u32))
            .arg(&count)
            .launch(grid_1d(voxel_rows * channels))
            .map_err(|e| format!("launch unpatch_affine_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 每 sample 拼接 activation[input_rows] + weight[weight_rows],输出按 sample 交错。
pub fn concat_weight_rows_batched_f16(ctx: &CudaContext, input: &CudaTensor, weight: &CudaSliceF16, input_rows: usize, weight_rows: usize, batch: usize) -> Result<CudaTensor, String> {
    if batch == 0 || input.rows != batch * input_rows || weight.len() != weight_rows * input.cols {
        return Err(format!("CUDA concat_weight_rows_batched input=[{},{}] input_rows={input_rows} weight_rows={weight_rows} batch={batch}", input.rows, input.cols,));
    }
    let output = ctx.tensor_uninit(batch * (input_rows + weight_rows), input.cols)?;
    let func = ctx.function("concat_weight_rows_batched_f16")?;
    let count = (batch * (input_rows + weight_rows) * input.cols) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(weight)
            .arg(&output.slice)
            .arg(&(input_rows as u32))
            .arg(&(weight_rows as u32))
            .arg(&(input.cols as u32))
            .arg(&count)
            .launch(grid_1d(batch * (input_rows + weight_rows) * input.cols))
            .map_err(|e| format!("launch concat_weight_rows_batched_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 每 sample 取前 output_rows 行。输出按 sample 交错。
pub fn take_rows_batched_f16(ctx: &CudaContext, input: &CudaTensor, output_rows: usize, batch: usize) -> Result<CudaTensor, String> {
    let input_rows = input.rows.checked_div(batch).ok_or("CUDA take_rows_batched batch=0")?;
    if batch == 0 || output_rows > input_rows {
        return Err(format!("CUDA take_rows_batched input=[{},{}] output_rows={output_rows} batch={batch}", input.rows, input.cols));
    }
    let output = ctx.tensor_uninit(batch * output_rows, input.cols)?;
    let func = ctx.function("take_rows_batched_f16")?;
    let count = (batch * output_rows * input.cols) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(&output.slice)
            .arg(&(input_rows as u32))
            .arg(&(output_rows as u32))
            .arg(&(input.cols as u32))
            .arg(&count)
            .launch(grid_1d(batch * output_rows * input.cols))
            .map_err(|e| format!("launch take_rows_batched_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

// —— 音频 VAE 解码算子(M5)launcher ——

/// [batch*time, channels] → [batch*channels, time] 转置 + 每通道仿射。
pub fn audio_unpack_affine_f16(ctx: &CudaContext, input: &CudaTensor, scale: &CudaSliceF16, bias: &CudaSliceF16, batch: usize, time: usize, channels: usize) -> Result<CudaTensor, String> {
    if input.rows != batch * time || input.cols != channels || scale.len() != channels || bias.len() != channels {
        return Err(format!("CUDA audio_unpack_affine input=[{},{}] batch={batch} time={time} channels={channels} scale={} bias={}", input.rows, input.cols, scale.len(), bias.len()));
    }
    let output = ctx.tensor_uninit(batch * channels, time)?;
    let func = ctx.function("audio_unpack_affine_f16")?;
    let count = (batch * channels * time) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(scale)
            .arg(bias)
            .arg(&output.slice)
            .arg(&(batch as u32))
            .arg(&(channels as u32))
            .arg(&(time as u32))
            .arg(&count)
            .launch(grid_1d(batch * channels * time))
            .map_err(|e| format!("launch audio_unpack_affine_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 每行 weight-norm:g[r]/max(||v[r]||,1e-12) * v[r,:]。输出 [rows, cols]。
pub fn audio_weight_norm_f16(ctx: &CudaContext, weight_g: &CudaSliceF16, weight_v: &CudaSliceF16, rows: usize, cols: usize) -> Result<CudaSliceF16, String> {
    if weight_g.len() != rows || weight_v.len() != rows * cols {
        return Err(format!("CUDA audio_weight_norm g={} v={} rows={rows} cols={cols}", weight_g.len(), weight_v.len()));
    }
    let output = ctx.buffer_uninit::<half::f16>(rows * cols)?;
    let func = ctx.function("audio_weight_norm_f16")?;
    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (THREADS as usize * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream().launch_builder(&func).arg(weight_g).arg(weight_v).arg(&output).arg(&(cols as u32)).launch(cfg).map_err(|e| format!("launch audio_weight_norm_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 1D 卷积(stride=1),weight-norm 内联(g/bias 必填)。输出 [batch*out_ch, out_len]。
#[allow(clippy::too_many_arguments)]
pub fn audio_conv1d_f16(
    ctx: &CudaContext,
    input: &CudaTensor,
    weight_g: &CudaSliceF16,
    weight_v: &CudaSliceF16,
    bias: &CudaSliceF16,
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
) -> Result<CudaTensor, String> {
    let in_len = input.cols;
    if input.rows != batch * in_ch {
        return Err(format!("CUDA audio_conv1d input=[{},{}] batch={batch} in_ch={in_ch}", input.rows, input.cols));
    }
    if weight_v.len() != out_ch * in_ch * kernel || weight_g.len() != out_ch || bias.len() != out_ch {
        return Err(format!("CUDA audio_conv1d weight_g={} weight_v={} bias={} out_ch={out_ch} in_ch={in_ch} kernel={kernel}", weight_g.len(), weight_v.len(), bias.len()));
    }
    let effective = dilation * (kernel - 1) + 1;
    if in_len + 2 * padding < effective {
        return Err(format!("CUDA audio_conv1d in_len={in_len}+2*{padding} < effective={effective}"));
    }
    let out_len = in_len + 2 * padding - effective + 1;
    let output = ctx.tensor_uninit(batch * out_ch, out_len)?;
    let func = ctx.function("audio_conv1d_f16")?;
    let count = (batch * out_ch * out_len) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(weight_g)
            .arg(weight_v)
            .arg(bias)
            .arg(&output.slice)
            .arg(&(batch as u32))
            .arg(&(in_ch as u32))
            .arg(&(out_ch as u32))
            .arg(&(in_len as u32))
            .arg(&(out_len as u32))
            .arg(&(kernel as u32))
            .arg(&(dilation as u32))
            .arg(&(padding as u32))
            .arg(&count)
            .launch(grid_1d(batch * out_ch * out_len))
            .map_err(|e| format!("launch audio_conv1d_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 1D 卷积(stride=1),无 weight-norm(weight_v 直接用)。bias 必填(无 bias 由调用方零填)。
/// 输出 [batch*out_ch, out_len]。
#[allow(clippy::too_many_arguments)]
pub fn audio_conv1d_plain_f16(ctx: &CudaContext, input: &CudaTensor, weight_v: &CudaSliceF16, bias: &CudaSliceF16, batch: usize, in_ch: usize, out_ch: usize, kernel: usize, dilation: usize, padding: usize) -> Result<CudaTensor, String> {
    let in_len = input.cols;
    if input.rows != batch * in_ch {
        return Err(format!("CUDA audio_conv1d_plain input=[{},{}] batch={batch} in_ch={in_ch}", input.rows, input.cols));
    }
    if weight_v.len() != out_ch * in_ch * kernel || bias.len() != out_ch {
        return Err(format!("CUDA audio_conv1d_plain weight_v={} bias={} out_ch={out_ch} in_ch={in_ch} kernel={kernel}", weight_v.len(), bias.len()));
    }
    let effective = dilation * (kernel - 1) + 1;
    if in_len + 2 * padding < effective {
        return Err(format!("CUDA audio_conv1d_plain in_len={in_len}+2*{padding} < effective={effective}"));
    }
    let out_len = in_len + 2 * padding - effective + 1;
    let output = ctx.tensor_uninit(batch * out_ch, out_len)?;
    let func = ctx.function("audio_conv1d_plain_f16")?;
    let count = (batch * out_ch * out_len) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(weight_v)
            .arg(bias)
            .arg(&output.slice)
            .arg(&(batch as u32))
            .arg(&(in_ch as u32))
            .arg(&(out_ch as u32))
            .arg(&(in_len as u32))
            .arg(&(out_len as u32))
            .arg(&(kernel as u32))
            .arg(&(dilation as u32))
            .arg(&(padding as u32))
            .arg(&count)
            .launch(grid_1d(batch * out_ch * out_len))
            .map_err(|e| format!("launch audio_conv1d_plain_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 1D 转置卷积(内部先 weight-norm)。输出 [batch*out_ch, out_len]。
#[allow(clippy::too_many_arguments)]
pub fn audio_conv_transpose1d_f16(
    ctx: &CudaContext,
    input: &CudaTensor,
    weight_g: &CudaSliceF16,
    weight_v: &CudaSliceF16,
    bias: &CudaSliceF16,
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<CudaTensor, String> {
    let in_len = input.cols;
    if in_len == 0 || input.rows != batch * in_ch {
        return Err(format!("CUDA audio_conv_transpose1d input=[{},{}] batch={batch} in_ch={in_ch}", input.rows, input.cols));
    }
    if weight_v.len() != in_ch * out_ch * kernel || weight_g.len() != in_ch || bias.len() != out_ch {
        return Err(format!("CUDA audio_conv_transpose1d weight_g={} weight_v={} bias={} in_ch={in_ch} out_ch={out_ch} kernel={kernel}", weight_g.len(), weight_v.len(), bias.len()));
    }
    let out_len = (in_len - 1) * stride + kernel - 2 * padding;
    // weight 按 in_ch 归一化(norm 与输出 out_ch 不对齐,需预计算 temp)。
    let weight = audio_weight_norm_f16(ctx, weight_g, weight_v, in_ch, out_ch * kernel)?;
    let output = ctx.tensor_uninit(batch * out_ch, out_len)?;
    let func = ctx.function("audio_conv_transpose1d_f16")?;
    let count = (batch * out_ch * out_len) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(&weight)
            .arg(bias)
            .arg(&output.slice)
            .arg(&(batch as u32))
            .arg(&(in_ch as u32))
            .arg(&(out_ch as u32))
            .arg(&(in_len as u32))
            .arg(&(out_len as u32))
            .arg(&(kernel as u32))
            .arg(&(stride as u32))
            .arg(&(padding as u32))
            .arg(&count)
            .launch(grid_1d(batch * out_ch * out_len))
            .map_err(|e| format!("launch audio_conv_transpose1d_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 官方 alias-free upsample2→SnakeBeta→downsample2。输入/输出 [batch*channels, length]。
pub fn audio_snake_beta_f16(ctx: &CudaContext, input: &CudaTensor, alpha: &CudaSliceF16, beta: &CudaSliceF16, up_filter: &CudaSliceF16, down_filter: &CudaSliceF16, batch: usize, channels: usize) -> Result<CudaTensor, String> {
    let length = input.cols;
    if input.rows != batch * channels || alpha.len() != channels || beta.len() != channels || up_filter.len() != down_filter.len() || up_filter.is_empty() {
        return Err(format!("CUDA audio_snake_beta input=[{},{}] batch={batch} channels={channels} alpha={} beta={} up={} down={}", input.rows, input.cols, alpha.len(), beta.len(), up_filter.len(), down_filter.len()));
    }
    let filter = up_filter.len();
    let activated = ctx.buffer_uninit::<f32>(batch * channels * length * 2)?;
    let func_up = ctx.function("audio_snake_beta_up_f16")?;
    let count_up = batch * channels * length * 2;
    unsafe {
        ctx.stream()
            .launch_builder(&func_up)
            .arg(&input.slice)
            .arg(alpha)
            .arg(beta)
            .arg(up_filter)
            .arg(&activated)
            .arg(&(channels as u32))
            .arg(&(length as u32))
            .arg(&(filter as u32))
            .arg(&(count_up as u32))
            .launch(grid_1d(count_up))
            .map_err(|e| format!("launch audio_snake_beta_up_f16 失败: {e:?}"))?;
    }
    let output = ctx.tensor_uninit(batch * channels, length)?;
    let func_down = ctx.function("audio_snake_beta_down_f16")?;
    let count_down = batch * channels * length;
    unsafe {
        ctx.stream()
            .launch_builder(&func_down)
            .arg(&activated)
            .arg(down_filter)
            .arg(&output.slice)
            .arg(&(channels as u32))
            .arg(&(length as u32))
            .arg(&(filter as u32))
            .arg(&(count_down as u32))
            .launch(grid_1d(count_down))
            .map_err(|e| format!("launch audio_snake_beta_down_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// elementwise × scalar。
pub fn scale_tensor_f16(ctx: &CudaContext, input: &CudaTensor, scale: f32) -> Result<CudaTensor, String> {
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("scale_tensor_f16")?;
    let count = input.len() as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(&output.slice).arg(&scale).arg(&count).launch(grid_1d(input.len())).map_err(|e| format!("launch scale_tensor_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// elementwise tanh。
pub fn tanh_f16(ctx: &CudaContext, input: &CudaTensor) -> Result<CudaTensor, String> {
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("tanh_f16")?;
    let count = input.len() as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(&output.slice).arg(&count).launch(grid_1d(input.len())).map_err(|e| format!("launch tanh_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use super::*;
    use crate::kernel::cpu::vae as cpu;

    fn ctx() -> CudaContext {
        CudaContext::new_default().expect("CUDA 初始化")
    }

    /// host f32 → device f16 slice(对称 `diffusion::tests` 的 htod helper)。
    fn htod(ctx: &CudaContext, values: &[f32]) -> CudaSliceF16 {
        ctx.stream().clone_htod::<half::f16, _>(&values.iter().map(|v| half::f16::from_f32(*v)).collect::<Vec<_>>()).expect("htod 上传")
    }

    fn check_close(name: &str, actual: &[f32], expect: &[f32]) {
        let atol = 1e-2f32;
        let rtol = 1e-2f32;
        assert_eq!(actual.len(), expect.len(), "{name}: 长度 {} != {}", actual.len(), expect.len());
        for (i, (a, e)) in actual.iter().zip(expect).enumerate() {
            let err = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(err <= tol, "{name}[{i}] 偏差 {err:.4} 超阈值 {tol:.4}(actual={a:.4} expect={e:.4})");
        }
    }

    fn seq(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.07 - 0.35).collect()
    }

    #[test]
    fn rmsnorm_heads_unit_matches_oracle() {
        let ctx = ctx();
        let (rows, heads, head_dim) = (3usize, 4, 16);
        let input = seq(rows * heads * head_dim);
        let expect = cpu::rms_norm_heads_unit(&input, heads, head_dim, 1e-6).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, rows, heads * head_dim).unwrap();
        let out = rmsnorm_heads_unit_f16(&ctx, &gpu_in, heads, head_dim, 1e-6).unwrap();
        check_close("rmsnorm_heads_unit", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn scaled_residual_columns_matches_oracle() {
        let ctx = ctx();
        let (rows, cols) = (5usize, 7);
        let input = seq(rows * cols);
        let update: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.13 - 0.9).collect();
        let scale = seq(cols);
        let expect = cpu::scaled_residual(&input, &update, &scale, cols).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, rows, cols).unwrap();
        let gpu_up = ctx.tensor_from_f32(&update, rows, cols).unwrap();
        let gpu_scale = htod(&ctx, &scale);
        let out = scaled_residual_columns_f16(&ctx, &gpu_in, &gpu_up, &gpu_scale).unwrap();
        check_close("scaled_residual", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn unpatch_affine_matches_oracle() {
        // shape [t=2,h=4,w=4], patch [1,2,2], channels=3 → 8 patch 行 × 12 列。
        let ctx = ctx();
        let shape = [2usize, 4, 4];
        let patch = [1usize, 2, 2];
        let channels = 3usize;
        let grid = [shape[0] / patch[0], shape[1] / patch[1], shape[2] / patch[2]];
        let patch_rows = grid.into_iter().product::<usize>();
        let patch_cols = channels * patch.into_iter().product::<usize>();
        let input = seq(patch_rows * patch_cols);
        let scale = seq(channels);
        let bias = seq(channels);
        let expect = cpu::unpatch_affine(&input, &scale, &bias, shape, patch, channels).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, patch_rows, patch_cols).unwrap();
        let gpu_scale = htod(&ctx, &scale);
        let gpu_bias = htod(&ctx, &bias);
        let out = unpatch_affine_f16(&ctx, &gpu_in, &gpu_scale, &gpu_bias, shape, patch, channels).unwrap();
        check_close("unpatch_affine", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn layer_norm_matches_oracle() {
        let ctx = ctx();
        let (rows, cols) = (3usize, 32);
        let input = seq(rows * cols);
        let weight = seq(cols);
        let bias = seq(cols);
        let expect = cpu::layer_norm(&input, &weight, &bias, cols, 1e-5).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, rows, cols).unwrap();
        let gpu_w = htod(&ctx, &weight);
        let gpu_b = htod(&ctx, &bias);
        let out = crate::kernel::cuda::tensor::layer_norm_f16(&ctx, &gpu_in, &gpu_w, &gpu_b, 1e-5).unwrap();
        check_close("layer_norm", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn concat_weight_rows_batched_matches_reference() {
        // 每 sample 拼接 [activation(input_rows) | weight(weight_rows)],输出按 sample 交错。
        // 无 CPU oracle,内联 host 参考实现。
        let ctx = ctx();
        let (batch, input_rows, weight_rows, cols) = (2usize, 3, 2, 5);
        let input = seq(batch * input_rows * cols);
        let weight = seq(weight_rows * cols);
        let gpu_in = ctx.tensor_from_f32(&input, batch * input_rows, cols).unwrap();
        let gpu_w = htod(&ctx, &weight);
        let out = concat_weight_rows_batched_f16(&ctx, &gpu_in, &gpu_w, input_rows, weight_rows, batch).unwrap();
        let output_rows = input_rows + weight_rows;
        let mut expect = vec![0.0f32; batch * output_rows * cols];
        for sample in 0..batch {
            for row in 0..output_rows {
                for column in 0..cols {
                    let value = if row < input_rows { input[(sample * input_rows + row) * cols + column] } else { weight[(row - input_rows) * cols + column] };
                    expect[(sample * output_rows + row) * cols + column] = value;
                }
            }
        }
        check_close("concat_weight_rows_batched", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn take_rows_batched_matches_reference() {
        // 每 sample 取前 output_rows 行(丢弃尾部 register/suffix token)。无 CPU oracle,内联参考。
        let ctx = ctx();
        let (batch, input_rows, output_rows, cols) = (2usize, 5, 3, 4);
        let input = seq(batch * input_rows * cols);
        let gpu_in = ctx.tensor_from_f32(&input, batch * input_rows, cols).unwrap();
        let out = take_rows_batched_f16(&ctx, &gpu_in, output_rows, batch).unwrap();
        let mut expect = vec![0.0f32; batch * output_rows * cols];
        for sample in 0..batch {
            for row in 0..output_rows {
                for column in 0..cols {
                    expect[(sample * output_rows + row) * cols + column] = input[(sample * input_rows + row) * cols + column];
                }
            }
        }
        check_close("take_rows_batched", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn audio_unpack_affine_matches_oracle() {
        let ctx = ctx();
        let (batch, time, channels) = (2usize, 5, 3);
        let input = seq(batch * time * channels);
        let scale = seq(channels);
        let bias = seq(channels);
        let expect = cpu::audio_unpack_affine(&input, &scale, &bias, batch, time, channels).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, batch * time, channels).unwrap();
        let out = audio_unpack_affine_f16(&ctx, &gpu_in, &htod(&ctx, &scale), &htod(&ctx, &bias), batch, time, channels).unwrap();
        check_close("audio_unpack_affine", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn audio_weight_norm_matches_reference() {
        let ctx = ctx();
        let (rows, cols) = (3usize, 5);
        let weight_g = seq(rows);
        let weight_v: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.11 - 0.4).collect();
        let mut expect = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let src = &weight_v[r * cols..(r + 1) * cols];
            let norm = src.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            let factor = weight_g[r] / norm;
            for c in 0..cols {
                expect[r * cols + c] = src[c] * factor;
            }
        }
        let out = audio_weight_norm_f16(&ctx, &htod(&ctx, &weight_g), &htod(&ctx, &weight_v), rows, cols).unwrap();
        let got: Vec<f32> = ctx.stream().clone_dtoh::<half::f16, _>(&out).unwrap().iter().map(|v| v.to_f32()).collect();
        check_close("audio_weight_norm", &got, &expect);
    }

    #[test]
    fn audio_conv1d_matches_oracle() {
        let ctx = ctx();
        let (batch, in_ch, out_ch, in_len, kernel, dilation, padding) = (2usize, 3, 4, 9, 5, 2, 4);
        let input = seq(batch * in_ch * in_len);
        let weight_g = seq(out_ch);
        let weight_v: Vec<f32> = (0..out_ch * in_ch * kernel).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let bias = seq(out_ch);
        let expect = cpu::conv1d(&input, Some(&weight_g), &weight_v, Some(&bias), batch, in_ch, out_ch, in_len, kernel, 1, dilation, padding).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, batch * in_ch, in_len).unwrap();
        let out = audio_conv1d_f16(&ctx, &gpu_in, &htod(&ctx, &weight_g), &htod(&ctx, &weight_v), &htod(&ctx, &bias), batch, in_ch, out_ch, kernel, dilation, padding).unwrap();
        check_close("audio_conv1d", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn audio_conv1d_plain_matches_oracle() {
        // 无 weight-norm 的普通 conv(weight_g=None → factor=1,weight_v 直接用)。
        // 供 audio VAE 里 normalized=false 的 conv。oracle cpu::conv1d 的 weight_g=None 分支。
        let ctx = ctx();
        let (batch, in_ch, out_ch, in_len, kernel, dilation, padding) = (2usize, 3, 4, 9, 5, 2, 4);
        let input = seq(batch * in_ch * in_len);
        let weight_v: Vec<f32> = (0..out_ch * in_ch * kernel).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let bias = seq(out_ch);
        let expect = cpu::conv1d(&input, None, &weight_v, Some(&bias), batch, in_ch, out_ch, in_len, kernel, 1, dilation, padding).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, batch * in_ch, in_len).unwrap();
        let out = audio_conv1d_plain_f16(&ctx, &gpu_in, &htod(&ctx, &weight_v), &htod(&ctx, &bias), batch, in_ch, out_ch, kernel, dilation, padding).unwrap();
        check_close("audio_conv1d_plain", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn audio_conv_transpose1d_matches_oracle() {
        let ctx = ctx();
        let (batch, in_ch, out_ch, in_len, kernel, stride, padding) = (2usize, 3, 2, 5, 4, 2, 1);
        let input = seq(batch * in_ch * in_len);
        let weight_g = seq(in_ch);
        let weight_v: Vec<f32> = (0..in_ch * out_ch * kernel).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let bias = seq(out_ch);
        let expect = cpu::conv_transpose1d(&input, &weight_g, &weight_v, &bias, batch, in_ch, out_ch, in_len, kernel, stride, padding).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, batch * in_ch, in_len).unwrap();
        let out = audio_conv_transpose1d_f16(&ctx, &gpu_in, &htod(&ctx, &weight_g), &htod(&ctx, &weight_v), &htod(&ctx, &bias), batch, in_ch, out_ch, kernel, stride, padding).unwrap();
        check_close("audio_conv_transpose1d", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn audio_snake_beta_matches_oracle() {
        let ctx = ctx();
        let (batch, channels, length) = (2usize, 3, 7);
        let input = seq(batch * channels * length);
        let alpha = seq(channels);
        let beta = seq(channels);
        let up_filter: Vec<f32> = vec![0.1, 0.25, 0.25, 0.1];
        let down_filter: Vec<f32> = vec![0.1, 0.25, 0.25, 0.1];
        let expect = cpu::snake_beta(&input, &alpha, &beta, &up_filter, &down_filter, channels, length).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, batch * channels, length).unwrap();
        let out = audio_snake_beta_f16(&ctx, &gpu_in, &htod(&ctx, &alpha), &htod(&ctx, &beta), &htod(&ctx, &up_filter), &htod(&ctx, &down_filter), batch, channels).unwrap();
        check_close("audio_snake_beta", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn scale_tensor_matches_reference() {
        let ctx = ctx();
        let (rows, cols) = (3usize, 5);
        let input = seq(rows * cols);
        let gpu_in = ctx.tensor_from_f32(&input, rows, cols).unwrap();
        let out = scale_tensor_f16(&ctx, &gpu_in, 0.37).unwrap();
        let expect: Vec<f32> = input.iter().map(|v| v * 0.37).collect();
        check_close("scale_tensor", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn tanh_matches_reference() {
        let ctx = ctx();
        let (rows, cols) = (3usize, 5);
        let input = seq(rows * cols);
        let gpu_in = ctx.tensor_from_f32(&input, rows, cols).unwrap();
        let out = tanh_f16(&ctx, &gpu_in).unwrap();
        let expect: Vec<f32> = input.iter().map(|v| v.tanh()).collect();
        check_close("tanh", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }
}
