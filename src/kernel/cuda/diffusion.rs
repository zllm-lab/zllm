/// 本模块的 CUDA shader(DiT / DiffusionBackend 算子)。
///
/// 共用 helper 见 `mod.rs` 前导。`mod.rs` 的 `kernels_source()`
/// 把 mod.rs 的 `PREAMBLE_SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
///
/// 算法对齐 `kernel/cpu/vae.rs`(数值 oracle)与 `kernel/metal/diffusion.rs`/`metal/vae.rs`
/// (F16 模板)。f16 入出、f32 累加。
pub const SHADERS: &str = r#"
// output[id] = input[id] + bias[id % columns]。bias 为 [1, columns] 行向量,逐行广播。
// linear_bias(每个线性层后接)依赖此算子;DiT/VAE 共用。
extern "C" __global__ void add_row_bias_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ bias,
    __half * __restrict__ output,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = __float2half(__half2float(input[id]) + __half2float(bias[id % columns]));
}

// output = input * (1 + scale) + shift。shift/scale 为 [modulation_rows, columns]:
// modulation_rows==1 时按列广播,否则逐元素。
extern "C" __global__ void adaln_modulate_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ shift,
    const __half * __restrict__ scale,
    __half * __restrict__ output,
    const unsigned int columns,
    const unsigned int count,
    const unsigned int modulation_rows)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int modulation = modulation_rows == 1 ? id % columns : id;
    float x = __half2float(input[id]);
    output[id] = __float2half(x * (1.0f + __half2float(scale[modulation])) + __half2float(shift[modulation]));
}

// 分段广播 AdaLN:modulation = row_map[row] * columns + col。row_map[input_row] → 调制行。
extern "C" __global__ void adaln_modulate_segmented_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ shift,
    const __half * __restrict__ scale,
    const unsigned int * __restrict__ row_map,
    __half * __restrict__ output,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int modulation = row_map[id / columns] * columns + id % columns;
    float x = __half2float(input[id]);
    output[id] = __float2half(x * (1.0f + __half2float(scale[modulation])) + __half2float(shift[modulation]));
}

// 分段门控残差:output = residual + update * gate。gate 按 row_map 分段广播。
extern "C" __global__ void gated_residual_segmented_f16(
    const __half * __restrict__ residual,
    const __half * __restrict__ update,
    const __half * __restrict__ gate,
    const unsigned int * __restrict__ row_map,
    __half * __restrict__ output,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int modulation = row_map[id / columns] * columns + id % columns;
    output[id] = __float2half(__half2float(residual[id]) + __half2float(update[id]) * __half2float(gate[modulation]));
}

// f32 残差流的分段门控残差:residual 为 f32(残差可达 ~6e4,超 f16),update/gate 为 f16,
// 输出 f32。这是 DiT 残差累积路径——每层 hidden 在此以 f32 累加,避免溢出。
extern "C" __global__ void gated_residual_segmented_f32_from_f32(
    const float * __restrict__ residual,
    const __half * __restrict__ update,
    const __half * __restrict__ gate,
    const unsigned int * __restrict__ row_map,
    float * __restrict__ output,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int modulation = row_map[id / columns] * columns + id % columns;
    output[id] = residual[id] + __half2float(update[id]) * __half2float(gate[modulation]);
}

// 首层 bootstrap:residual 仍是 f16(packed_hidden,~50),输出 f32 以开启残差流 f32 路径。
extern "C" __global__ void gated_residual_segmented_f32_from_f16(
    const __half * __restrict__ residual,
    const __half * __restrict__ update,
    const __half * __restrict__ gate,
    const unsigned int * __restrict__ row_map,
    float * __restrict__ output,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int modulation = row_map[id / columns] * columns + id % columns;
    output[id] = __half2float(residual[id]) + __half2float(update[id]) * __half2float(gate[modulation]);
}

// f32 残差的逐行 RMSNorm:读 f32 input(残差,可达 ~5e5)、f16 weight,写 f32 output。
// f32 输出以把残差流 f32 传播给下游 adaln→linear(MLP down 投影可达 ~6e4,必须 f32)。
// 块归约同 rmsnorm_f16,只是 input/output 都是 float。
extern "C" __global__ void rmsnorm_residual_f32(
    const float * __restrict__ input,
    const __half * __restrict__ weight,
    float * __restrict__ output,
    const unsigned int columns,
    const float epsilon)
{
    extern __shared__ float warp_sums[];
    unsigned int row = blockIdx.x;
    unsigned int lane = threadIdx.x;
    unsigned int width = blockDim.x;
    unsigned long long offset = (unsigned long long)row * columns;

    float sum = 0.0f;
    for (unsigned int c = lane; c < columns; c += width) {
        float v = input[offset + c];
        sum += v * v;
    }
    for (int offset_w = 16; offset_w > 0; offset_w >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset_w);
    }
    if (lane % 32 == 0) warp_sums[lane / 32] = sum;
    __syncthreads();

    int num_warps = (width + 31) / 32;
    float total = 0.0f;
    if (lane < 32) {
        total = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, off);
        }
        if (lane == 0) warp_sums[0] = total;
    }
    __syncthreads();
    total = warp_sums[0];

    float scale = rsqrtf(total / float(columns) + epsilon);
    for (unsigned int c = lane; c < columns; c += width) {
        output[offset + c] = input[offset + c] * scale * __half2float(weight[c]);
    }
}

// 分段广播 AdaLN(f32 输入 → f32 输出):把残差流 f32 传播到 modulated,供 MLP linear。
// shift/scale 仍是 f16(modulation 投影输出,~O(1-5))。row_map[input_row] → 调制行。
extern "C" __global__ void adaln_modulate_segmented_f32(
    const float * __restrict__ input,
    const __half * __restrict__ shift,
    const __half * __restrict__ scale,
    const unsigned int * __restrict__ row_map,
    float * __restrict__ output,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int modulation = row_map[id / columns] * columns + id % columns;
    float x = input[id];
    output[id] = x * (1.0f + __half2float(scale[modulation])) + __half2float(shift[modulation]);
}

// gate_up [rows, 2*cols](f32)→ 激活 [rows, cols](f32):前 cols=gate,后 cols=up。
// output = silu(gate) * up。融合拆分+激活,把 f32 传到 MLP down 投影的输入。
extern "C" __global__ void split_gated_silu_f32(
    const float * __restrict__ input,
    float * __restrict__ output,
    const unsigned int cols,
    const unsigned int count)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    unsigned int row = gid / cols;
    unsigned int c = gid - row * cols;
    unsigned long long base = (unsigned long long)row * 2u * cols;
    float gate = input[base + c];
    float up = input[base + cols + c];
    output[gid] = (gate / (1.0f + expf(-gate))) * up;
}

// 分段门控残差(residual f32 + update f32 + gate f16 → f32):MLP 更新路径。
// mlp_update 来自 f32 linear(可达 ~6e4),故 update 也读 f32。gate 仍是 f16。
extern "C" __global__ void gated_residual_segmented_f32_f32update(
    const float * __restrict__ residual,
    const float * __restrict__ update,
    const __half * __restrict__ gate,
    const unsigned int * __restrict__ row_map,
    float * __restrict__ output,
    const unsigned int columns,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int modulation = row_map[id / columns] * columns + id % columns;
    output[id] = residual[id] + update[id] * __half2float(gate[modulation]);
}

// 逐元素 dtype 转换:残差流 f32 路径与 f16 权重/attention 桥接用。
extern "C" __global__ void cast_f16_to_f32(
    const __half * __restrict__ input,
    float * __restrict__ output,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = __half2float(input[id]);
}

extern "C" __global__ void cast_f32_to_f16(
    const float * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = __float2half(input[id]);
}

// adaln 输出切片:in[t][(m*chunks+c)*hidden+h] -> out[c][(t*modalities+m)*hidden+h]。
// 每个 chunk 单独启动(单独输出缓冲)。
extern "C" __global__ void modulation_chunks_f16(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int chunk_id,
    const unsigned int chunks,
    const unsigned int modalities,
    const unsigned int hidden,
    const unsigned int rows,
    const unsigned int input_cols,
    const unsigned int count)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    unsigned int output_row = gid / hidden;
    unsigned int h = gid - output_row * hidden;
    unsigned int time_row = output_row / modalities;
    unsigned int modality = output_row - time_row * modalities;
    unsigned int source = time_row * input_cols + (modality * chunks + chunk_id) * hidden + h;
    output[gid] = input[source];
}

// 逐头 RMSNorm(带 weight,无 (1+weight)):out = x * rsqrt(mean(x²)+eps) * weight[c]。
// 每 (row, head) 一组 block。对称 cpu/vae.rs:307;区别于 gemma 的 (1+weight)。
extern "C" __global__ void rmsnorm_heads_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int head_count,
    const unsigned int head_dim,
    const float eps)
{
    extern __shared__ float sums[];
    unsigned int lane = threadIdx.x;
    unsigned int group = blockIdx.x;
    unsigned long long begin = (unsigned long long)group * head_dim;
    float sum = 0.0f;
    for (unsigned int column = lane; column < head_dim; column += blockDim.x) {
        float value = __half2float(input[begin + column]);
        sum += value * value;
    }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    float scale = rsqrtf(sums[0] / (float)head_dim + eps);
    for (unsigned int column = lane; column < head_dim; column += blockDim.x) {
        output[begin + column] = __float2half(__half2float(input[begin + column]) * scale * __half2float(weight[column]));
    }
}

// Euler 速度更新:output = sample + scale * velocity。
extern "C" __global__ void flow_step_f16(
    const __half * __restrict__ sample,
    const __half * __restrict__ velocity,
    __half * __restrict__ output,
    const float scale,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = __float2half(__half2float(sample[id]) + scale * __half2float(velocity[id]));
}

// SiLU:output = x / (1 + exp(-x))。
extern "C" __global__ void silu_f16(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    float x = __half2float(input[id]);
    output[id] = __float2half(x / (1.0f + expf(-x)));
}

// 沿行拼接:output[gid] = gid < left_count ? left[gid] : right[gid - left_count]。
extern "C" __global__ void concat_rows_f16(
    const __half * __restrict__ left,
    const __half * __restrict__ right,
    __half * __restrict__ output,
    const unsigned int left_count,
    const unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    output[gid] = gid < left_count ? left[gid] : right[gid - left_count];
}

// 正弦时间步嵌入:half=dim/2, freq=exp(-ln(10000)*col/half), arg=t*freq。
// 一个线程写 cos 槽(col)与 sin 槽(half+col)。grid = batch * half。
extern "C" __global__ void timestep_embedding_f16(
    const float * __restrict__ timesteps,
    __half * __restrict__ output,
    const unsigned int batch,
    const unsigned int dim)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half = dim / 2;
    if (gid >= batch * half) return;
    unsigned int row = gid / half;
    unsigned int column = gid - row * half;
    float frequency = expf(-logf(10000.0f) * (float)column / (float)half);
    float value = timesteps[row] * frequency;
    output[row * dim + column] = __float2half(cosf(value));
    output[row * dim + half + column] = __float2half(sinf(value));
}

"#;

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaSlice;

use crate::diffusion::{ModulationSegment, modulation_row_map};

use super::tensor::grid_1d;
use super::{CudaContext, CudaSliceF16, CudaTensor, LaunchConfig, THREADS};

/// 逐行广播加偏置:output = input + bias(每行同一个 bias 向量)。
pub fn add_row_bias_f16(ctx: &CudaContext, input: &CudaTensor, bias: &CudaSliceF16) -> Result<CudaTensor, String> {
    if bias.len() != input.cols {
        return Err(format!("CUDA add_row_bias input=[{},{}] bias={}", input.rows, input.cols, bias.len()));
    }
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("add_row_bias_f16")?;
    let cols = input.cols as u32;
    let count = input.len() as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(bias).arg(&output.slice).arg(&cols).arg(&count).launch(grid_1d(input.len())).map_err(|e| format!("launch add_row_bias_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// output = input * (1 + scale) + shift。shift/scale 可按列广播([1,cols])或逐元素([rows,cols])。
pub fn adaln_modulate_f16(ctx: &CudaContext, input: &CudaTensor, shift: &CudaTensor, scale: &CudaTensor) -> Result<CudaTensor, String> {
    if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows || (shift.rows != 1 && shift.rows != input.rows) {
        return Err(format!("CUDA adaln_modulate input=[{},{}] shift=[{},{}] scale=[{},{}]", input.rows, input.cols, shift.rows, shift.cols, scale.rows, scale.cols));
    }
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("adaln_modulate_f16")?;
    let cols = input.cols as u32;
    let count = input.len() as u32;
    let modulation_rows = shift.rows as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(&shift.slice)
            .arg(&scale.slice)
            .arg(&output.slice)
            .arg(&cols)
            .arg(&count)
            .arg(&modulation_rows)
            .launch(grid_1d(input.len()))
            .map_err(|e| format!("launch adaln_modulate_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 分段广播 AdaLN。row_map 由 modulation_row_map(segments) 构建并上传。
pub fn adaln_modulate_segmented_f16(ctx: &CudaContext, input: &CudaTensor, shift: &CudaTensor, scale: &CudaTensor, segments: &[ModulationSegment]) -> Result<CudaTensor, String> {
    if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows {
        return Err(format!("CUDA adaln_modulate_segmented input=[{},{}] shift=[{},{}]", input.rows, input.cols, shift.rows, shift.cols));
    }
    let row_map = modulation_row_map(segments, input.rows, shift.rows).map_err(|e| format!("adaln_modulate_segmented row_map: {e}"))?;
    let row_map_gpu = ctx.upload_row_map(&row_map).map_err(|e| format!("上传 adaln row_map: {e:?}"))?;
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("adaln_modulate_segmented_f16")?;
    let cols = input.cols as u32;
    let count = input.len() as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(&shift.slice)
            .arg(&scale.slice)
            .arg(&*row_map_gpu)
            .arg(&output.slice)
            .arg(&cols)
            .arg(&count)
            .launch(grid_1d(input.len()))
            .map_err(|e| format!("launch adaln_modulate_segmented_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 分段门控残差:output = residual + update * gate。gate 按 row_map 分段广播。
pub fn gated_residual_segmented_f16(ctx: &CudaContext, residual: &CudaTensor, update: &CudaTensor, gate: &CudaTensor, segments: &[ModulationSegment]) -> Result<CudaTensor, String> {
    if residual.rows != update.rows || residual.cols != update.cols || gate.cols != residual.cols {
        return Err(format!("CUDA gated_residual_segmented residual=[{},{}] update=[{},{}] gate=[{},{}]", residual.rows, residual.cols, update.rows, update.cols, gate.rows, gate.cols));
    }
    let row_map = modulation_row_map(segments, residual.rows, gate.rows).map_err(|e| format!("gated_residual_segmented row_map: {e}"))?;
    let row_map_gpu = ctx.upload_row_map(&row_map).map_err(|e| format!("上传 gated row_map: {e:?}"))?;
    let output = ctx.tensor_uninit(residual.rows, residual.cols)?;
    let func = ctx.function("gated_residual_segmented_f16")?;
    let cols = residual.cols as u32;
    let count = residual.len() as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&residual.slice)
            .arg(&update.slice)
            .arg(&gate.slice)
            .arg(&*row_map_gpu)
            .arg(&output.slice)
            .arg(&cols)
            .arg(&count)
            .launch(grid_1d(residual.len()))
            .map_err(|e| format!("launch gated_residual_segmented_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// f32 残差分段门控残差:residual(f32) + update(f16) * gate(f16) → f32 残差张量。
/// 输出经 `new_f32_residual`(`slice_f32`=权威,`slice`=占位)。每层 hidden 在此以 f32 累加。
pub fn gated_residual_segmented_f32(ctx: &CudaContext, residual_f32: &CudaSlice<f32>, update: &CudaTensor, gate: &CudaTensor, segments: &[ModulationSegment], rows: usize, cols: usize) -> Result<CudaTensor, String> {
    if update.rows != rows || update.cols != cols || gate.cols != cols || gate.rows == 0 {
        return Err(format!("CUDA gated_residual_segmented_f32 rows={rows} cols={cols} update=[{},{}] gate=[{},{}]", update.rows, update.cols, gate.rows, gate.cols));
    }
    let row_map = modulation_row_map(segments, rows, gate.rows).map_err(|e| format!("gated_residual_segmented_f32 row_map: {e}"))?;
    let row_map_gpu = ctx.upload_row_map(&row_map).map_err(|e| format!("上传 gated f32 row_map: {e:?}"))?;
    let count = rows.checked_mul(cols).ok_or("CUDA gated_residual_segmented_f32 大小溢出")?;
    let out_f32 = ctx.buffer_uninit_f32(count)?;
    let func = ctx.function("gated_residual_segmented_f32_from_f32")?;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(residual_f32)
            .arg(&update.slice)
            .arg(&gate.slice)
            .arg(&*row_map_gpu)
            .arg(&out_f32)
            .arg(&(cols as u32))
            .arg(&(count as u32))
            .launch(grid_1d(count))
            .map_err(|e| format!("launch gated_residual_segmented_f32_from_f32 失败: {e:?}"))?;
    }
    let placeholder = ctx.placeholder_f16()?;
    Ok(CudaTensor::new_f32_residual(out_f32, placeholder, rows, cols))
}

/// 首层 bootstrap:residual 仍是 f16(packed_hidden),输出 f32 残差张量以开启 f32 路径。
pub fn gated_residual_segmented_f32_from_f16(ctx: &CudaContext, residual: &CudaTensor, update: &CudaTensor, gate: &CudaTensor, segments: &[ModulationSegment]) -> Result<CudaTensor, String> {
    let rows = residual.rows;
    let cols = residual.cols;
    if residual.slice_f32.is_some() || update.rows != rows || update.cols != cols || gate.cols != cols || gate.rows == 0 {
        return Err(format!("CUDA gated_residual_f32_from_f16 residual=[{rows},{cols}] update=[{},{}] gate=[{},{}]", update.rows, update.cols, gate.rows, gate.cols));
    }
    let row_map = modulation_row_map(segments, rows, gate.rows).map_err(|e| format!("gated_residual_f32_from_f16 row_map: {e}"))?;
    let row_map_gpu = ctx.upload_row_map(&row_map).map_err(|e| format!("上传 gated f32 row_map: {e:?}"))?;
    let count = rows.checked_mul(cols).ok_or("CUDA gated_residual_f32_from_f16 大小溢出")?;
    let out_f32 = ctx.buffer_uninit_f32(count)?;
    let func = ctx.function("gated_residual_segmented_f32_from_f16")?;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&residual.slice)
            .arg(&update.slice)
            .arg(&gate.slice)
            .arg(&*row_map_gpu)
            .arg(&out_f32)
            .arg(&(cols as u32))
            .arg(&(count as u32))
            .launch(grid_1d(count))
            .map_err(|e| format!("launch gated_residual_segmented_f32_from_f16 失败: {e:?}"))?;
    }
    let placeholder = ctx.placeholder_f16()?;
    Ok(CudaTensor::new_f32_residual(out_f32, placeholder, rows, cols))
}

/// f32 残差 RMSNorm:读 f32 残差(可达 ~5e5)+ f16 weight,写 f32 输出(传播 f32 到下游)。
/// 返回 f32 残差张量(slice_f32=权威,slice=占位)。
pub fn rmsnorm_residual_f32(ctx: &CudaContext, input_f32: &CudaSlice<f32>, weight: &CudaSliceF16, rows: usize, cols: usize, eps: f32) -> Result<CudaTensor, String> {
    if cols == 0 {
        return Err("CUDA rmsnorm_residual columns=0".to_string());
    }
    let out_f32 = ctx.buffer_uninit_f32(rows.checked_mul(cols).ok_or("CUDA rmsnorm_residual_f32 大小溢出")?)?;
    let func = ctx.function("rmsnorm_residual_f32")?;
    let shared_bytes = (THREADS as usize / 32) * std::mem::size_of::<f32>();
    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: shared_bytes as u32 };
    unsafe {
        ctx.stream().launch_builder(&func).arg(input_f32).arg(weight).arg(&out_f32).arg(&(cols as u32)).arg(&eps).launch(cfg).map_err(|e| format!("launch rmsnorm_residual_f32 失败: {e:?}"))?;
    }
    let placeholder = ctx.placeholder_f16()?;
    Ok(CudaTensor::new_f32_residual(out_f32, placeholder, rows, cols))
}

/// 分段广播 AdaLN(f32 输入 → f32 输出)。input 携带 slice_f32;shift/scale 为 f16。
pub fn adaln_modulate_segmented_f32(ctx: &CudaContext, input: &CudaTensor, shift: &CudaTensor, scale: &CudaTensor, segments: &[ModulationSegment]) -> Result<CudaTensor, String> {
    let input_f32 = input.slice_f32.as_ref().ok_or("CUDA adaln_modulate_segmented_f32 input 无 slice_f32")?;
    if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows {
        return Err(format!("CUDA adaln_modulate_segmented_f32 input=[{},{}] shift=[{},{}]", input.rows, input.cols, shift.rows, shift.cols));
    }
    let row_map = modulation_row_map(segments, input.rows, shift.rows).map_err(|e| format!("adaln_modulate_segmented_f32 row_map: {e}"))?;
    let row_map_gpu = ctx.upload_row_map(&row_map).map_err(|e| format!("上传 adaln f32 row_map: {e:?}"))?;
    let count = input.rows.checked_mul(input.cols).ok_or("CUDA adaln_modulate_segmented_f32 大小溢出")?;
    let out_f32 = ctx.buffer_uninit_f32(count)?;
    let func = ctx.function("adaln_modulate_segmented_f32")?;
    let cols = input.cols as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(input_f32)
            .arg(&shift.slice)
            .arg(&scale.slice)
            .arg(&*row_map_gpu)
            .arg(&out_f32)
            .arg(&cols)
            .arg(&(count as u32))
            .launch(grid_1d(count))
            .map_err(|e| format!("launch adaln_modulate_segmented_f32 失败: {e:?}"))?;
    }
    let placeholder = ctx.placeholder_f16()?;
    Ok(CudaTensor::new_f32_residual(out_f32, placeholder, input.rows, input.cols))
}

/// gate_up(f32)→ silu(gate)*up(f32):融合拆分+激活,输出 f32 供 MLP down 投影。
pub fn split_gated_silu_f32(ctx: &CudaContext, input: &CudaTensor, cols: usize) -> Result<CudaTensor, String> {
    let input_f32 = input.slice_f32.as_ref().ok_or("CUDA split_gated_silu_f32 input 无 slice_f32")?;
    if cols == 0 || input.cols != cols.checked_mul(2).ok_or("CUDA split_gated_silu_f32 cols 溢出")? {
        return Err(format!("CUDA split_gated_silu_f32 input cols={} 期望 {}", input.cols, cols * 2));
    }
    let count = input.rows.checked_mul(cols).ok_or("CUDA split_gated_silu_f32 大小溢出")?;
    let out_f32 = ctx.buffer_uninit_f32(count)?;
    let func = ctx.function("split_gated_silu_f32")?;
    let cols_u32 = cols as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(input_f32).arg(&out_f32).arg(&cols_u32).arg(&(count as u32)).launch(grid_1d(count)).map_err(|e| format!("launch split_gated_silu_f32 失败: {e:?}"))?;
    }
    let placeholder = ctx.placeholder_f16()?;
    Ok(CudaTensor::new_f32_residual(out_f32, placeholder, input.rows, cols))
}

/// 分段门控残差(residual f32 + update f32 + gate f16 → f32):MLP 更新路径。
pub fn gated_residual_segmented_f32_f32update(ctx: &CudaContext, residual_f32: &CudaSlice<f32>, update: &CudaTensor, gate: &CudaTensor, segments: &[ModulationSegment], rows: usize, cols: usize) -> Result<CudaTensor, String> {
    let update_f32 = update.slice_f32.as_ref().ok_or("CUDA gated_residual_f32_f32update update 无 slice_f32")?;
    if update.rows != rows || update.cols != cols || gate.cols != cols || gate.rows == 0 {
        return Err(format!("CUDA gated_residual_f32_f32update rows={rows} cols={cols} update=[{},{}] gate=[{},{}]", update.rows, update.cols, gate.rows, gate.cols));
    }
    let row_map = modulation_row_map(segments, rows, gate.rows).map_err(|e| format!("gated_residual_f32_f32update row_map: {e}"))?;
    let row_map_gpu = ctx.upload_row_map(&row_map).map_err(|e| format!("上传 gated f32update row_map: {e:?}"))?;
    let count = rows.checked_mul(cols).ok_or("CUDA gated_residual_f32_f32update 大小溢出")?;
    let out_f32 = ctx.buffer_uninit_f32(count)?;
    let func = ctx.function("gated_residual_segmented_f32_f32update")?;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(residual_f32)
            .arg(update_f32)
            .arg(&gate.slice)
            .arg(&*row_map_gpu)
            .arg(&out_f32)
            .arg(&(cols as u32))
            .arg(&(count as u32))
            .launch(grid_1d(count))
            .map_err(|e| format!("launch gated_residual_f32_f32update 失败: {e:?}"))?;
    }
    let placeholder = ctx.placeholder_f16()?;
    Ok(CudaTensor::new_f32_residual(out_f32, placeholder, rows, cols))
}

/// 逐元素 f16 → f32 设备转换(扩散 f32 linear 的权重转换用)。
pub fn cast_f16_to_f32(ctx: &CudaContext, input: &CudaSliceF16, count: usize) -> Result<CudaSlice<f32>, String> {
    let out_f32 = ctx.buffer_uninit_f32(count)?;
    let func = ctx.function("cast_f16_to_f32")?;
    unsafe {
        ctx.stream().launch_builder(&func).arg(input).arg(&out_f32).arg(&(count as u32)).launch(grid_1d(count)).map_err(|e| format!("launch cast_f16_to_f32 失败: {e:?}"))?;
    }
    Ok(out_f32)
}

/// 逐元素 f32 → f16 设备转换(扩散 attention 入口把 f32 qkv 转回 f16 用)。
pub fn cast_f32_to_f16(ctx: &CudaContext, input: &CudaTensor) -> Result<CudaTensor, String> {
    let input_f32 = input.slice_f32.as_ref().ok_or("CUDA cast_f32_to_f16 input 无 slice_f32")?;
    let count = input.rows.checked_mul(input.cols).ok_or("CUDA cast_f32_to_f16 大小溢出")?;
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("cast_f32_to_f16")?;
    unsafe {
        ctx.stream().launch_builder(&func).arg(input_f32).arg(&output.slice).arg(&(count as u32)).launch(grid_1d(count)).map_err(|e| format!("launch cast_f32_to_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 逐元素 f32 → f16 设备转换(从原始 f32 slice,用于 f32 linear 切到 f16 走 hgemm)。
pub fn cast_f32_to_f16_slice(ctx: &CudaContext, input: &cudarc::driver::safe::CudaSlice<f32>, count: usize) -> Result<cudarc::driver::safe::CudaSlice<half::f16>, String> {
    let output = ctx.buffer_uninit::<half::f16>(count)?;
    let func = ctx.function("cast_f32_to_f16")?;
    unsafe {
        ctx.stream().launch_builder(&func).arg(input).arg(&output).arg(&(count as u32)).launch(grid_1d(count)).map_err(|e| format!("launch cast_f32_to_f16_slice 失败: {e:?}"))?;
    }
    Ok(output)
}

/// adaln 输出切片为 chunks 个 [rows*modalities, hidden] 张量。每个 chunk 单独启动。
pub fn modulation_chunks_f16(ctx: &CudaContext, input: &CudaTensor, modalities: usize, chunks: usize, hidden: usize) -> Result<Vec<CudaTensor>, String> {
    let input_cols = modalities.checked_mul(chunks).and_then(|v| v.checked_mul(hidden)).ok_or("CUDA modulation_chunks 列数溢出")?;
    if input.cols != input_cols {
        return Err(format!("CUDA modulation_chunks input=[{},{}] 期望 cols={input_cols}", input.rows, input.cols));
    }
    let func = ctx.function("modulation_chunks_f16")?;
    let output_rows = input.rows * modalities;
    let output_elements = output_rows * hidden;
    let mut outputs = Vec::with_capacity(chunks);
    for chunk_id in 0..chunks {
        let output = ctx.tensor_uninit(output_rows, hidden)?;
        unsafe {
            ctx.stream()
                .launch_builder(&func)
                .arg(&input.slice)
                .arg(&output.slice)
                .arg(&(chunk_id as u32))
                .arg(&(chunks as u32))
                .arg(&(modalities as u32))
                .arg(&(hidden as u32))
                .arg(&(input.rows as u32))
                .arg(&(input_cols as u32))
                .arg(&(output_elements as u32))
                .launch(grid_1d(output_elements))
                .map_err(|e| format!("launch modulation_chunks_f16[{chunk_id}] 失败: {e:?}"))?;
        }
        outputs.push(output);
    }
    Ok(outputs)
}

/// 逐头 RMSNorm(带 weight,out = x * rsqrt(mean(x²)+eps) * weight[c])。
pub fn rmsnorm_heads_f16(ctx: &CudaContext, input: &CudaTensor, weight: &CudaSliceF16, head_count: usize, head_dim: usize, eps: f32) -> Result<CudaTensor, String> {
    if head_count == 0 || head_dim == 0 || input.cols != head_count * head_dim || weight.len() != head_dim {
        return Err(format!("CUDA rmsnorm_heads input=[{},{}] heads={head_count} head_dim={head_dim} weight={}", input.rows, input.cols, weight.len()));
    }
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("rmsnorm_heads_f16")?;
    let cfg = LaunchConfig { grid_dim: ((input.rows * head_count) as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (THREADS as usize * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(weight).arg(&output.slice).arg(&(head_count as u32)).arg(&(head_dim as u32)).arg(&eps).launch(cfg).map_err(|e| format!("launch rmsnorm_heads_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// Euler 速度更新:output = sample + scale * velocity。
pub fn flow_step_f16(ctx: &CudaContext, sample: &CudaTensor, velocity: &CudaTensor, scale: f32) -> Result<CudaTensor, String> {
    if sample.rows != velocity.rows || sample.cols != velocity.cols {
        return Err(format!("CUDA flow_step sample=[{},{}] velocity=[{},{}]", sample.rows, sample.cols, velocity.rows, velocity.cols));
    }
    let output = ctx.tensor_uninit(sample.rows, sample.cols)?;
    let func = ctx.function("flow_step_f16")?;
    let count = sample.len() as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&sample.slice).arg(&velocity.slice).arg(&output.slice).arg(&scale).arg(&count).launch(grid_1d(sample.len())).map_err(|e| format!("launch flow_step_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// SiLU:output = x / (1 + exp(-x))。
pub fn silu_f16(ctx: &CudaContext, input: &CudaTensor) -> Result<CudaTensor, String> {
    let output = ctx.tensor_uninit(input.rows, input.cols)?;
    let func = ctx.function("silu_f16")?;
    let count = input.len() as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(&output.slice).arg(&count).launch(grid_1d(input.len())).map_err(|e| format!("launch silu_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 沿行拼接:left ++ right → [left.rows + right.rows, cols]。
pub fn concat_rows_f16(ctx: &CudaContext, left: &CudaTensor, right: &CudaTensor) -> Result<CudaTensor, String> {
    if left.cols != right.cols {
        return Err(format!("CUDA concat_rows left cols={} right cols={}", left.cols, right.cols));
    }
    let output = ctx.tensor_uninit(left.rows + right.rows, left.cols)?;
    let func = ctx.function("concat_rows_f16")?;
    let left_count = left.len() as u32;
    let total = (left.len() + right.len()) as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&left.slice).arg(&right.slice).arg(&output.slice).arg(&left_count).arg(&total).launch(grid_1d(left.len() + right.len())).map_err(|e| format!("launch concat_rows_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 正弦时间步嵌入 → [timesteps.len(), dim]。
pub fn timestep_embedding_f16(ctx: &CudaContext, timesteps: &[f32], dim: usize) -> Result<CudaTensor, String> {
    if dim == 0 || dim % 2 != 0 {
        return Err(format!("CUDA timestep_embedding dim={dim} 必须为正偶数"));
    }
    let timesteps_gpu = ctx.stream().clone_htod::<f32, _>(timesteps).map_err(|e| format!("上传 timestep_embedding timesteps: {e:?}"))?;
    let output = ctx.tensor_uninit(timesteps.len(), dim)?;
    let func = ctx.function("timestep_embedding_f16")?;
    let half = dim / 2;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&timesteps_gpu)
            .arg(&output.slice)
            .arg(&(timesteps.len() as u32))
            .arg(&(dim as u32))
            .launch(grid_1d(timesteps.len() * half))
            .map_err(|e| format!("launch timestep_embedding_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use super::*;
    use crate::diffusion::modulation_row_map;
    use crate::kernel::cpu::vae as cpu;

    fn ctx() -> CudaContext {
        CudaContext::new_default().expect("CUDA 初始化")
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
    fn add_row_bias_matches_reference() {
        let ctx = ctx();
        let (rows, cols) = (3usize, 8);
        let input = seq(rows * cols);
        let bias = seq(cols);
        let expect: Vec<f32> = input.iter().enumerate().map(|(i, &x)| x + bias[i % cols]).collect();
        let gpu_in = ctx.tensor_from_f32(&input, rows, cols).unwrap();
        let bias_gpu = ctx.stream().clone_htod::<half::f16, _>(&bias.iter().map(|v| half::f16::from_f32(*v)).collect::<Vec<_>>()).unwrap();
        let out = add_row_bias_f16(&ctx, &gpu_in, &bias_gpu).unwrap();
        check_close("add_row_bias", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn adaln_modulate_matches_oracle() {
        let ctx = ctx();
        let (rows, cols) = (4usize, 8);
        for modulation_rows in [1usize, rows] {
            let input = seq(rows * cols);
            let mod_data = seq(modulation_rows * cols);
            let expect = cpu::adaln_modulate(&input, &mod_data, &mod_data, rows, cols).unwrap();
            let gpu_in = ctx.tensor_from_f32(&input, rows, cols).unwrap();
            let gpu_mod = ctx.tensor_from_f32(&mod_data, modulation_rows, cols).unwrap();
            let out = adaln_modulate_f16(&ctx, &gpu_in, &gpu_mod, &gpu_mod).unwrap();
            check_close(&format!("adaln_modulate[mod_rows={modulation_rows}]"), &ctx.tensor_to_f32(&out).unwrap(), &expect);
        }
    }

    fn two_segments() -> Vec<ModulationSegment> {
        // 6 行,前 4 行用调制行 0,后 2 行用调制行 1。
        vec![ModulationSegment { rows: 0..4, modulation_row: 0 }, ModulationSegment { rows: 4..6, modulation_row: 1 }]
    }

    #[test]
    fn adaln_modulate_segmented_matches_oracle() {
        let ctx = ctx();
        let (rows, cols) = (6usize, 8);
        let modulation_rows = 2usize;
        let input = seq(rows * cols);
        let mod_data = seq(modulation_rows * cols);
        let segments = two_segments();
        let row_map = modulation_row_map(&segments, rows, modulation_rows).unwrap();
        let expect = cpu::adaln_modulate_segmented(&input, &mod_data, &mod_data, rows, cols, &row_map).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, rows, cols).unwrap();
        let gpu_mod = ctx.tensor_from_f32(&mod_data, modulation_rows, cols).unwrap();
        let out = adaln_modulate_segmented_f16(&ctx, &gpu_in, &gpu_mod, &gpu_mod, &segments).unwrap();
        check_close("adaln_modulate_segmented", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn gated_residual_segmented_matches_oracle() {
        let ctx = ctx();
        let (rows, cols) = (6usize, 8);
        let modulation_rows = 2usize;
        let residual = seq(rows * cols);
        let update = seq(rows * cols);
        let gate = seq(modulation_rows * cols);
        let segments = two_segments();
        let row_map = modulation_row_map(&segments, rows, modulation_rows).unwrap();
        let expect = cpu::gated_residual_segmented(&residual, &update, &gate, rows, cols, &row_map).unwrap();
        let gpu_res = ctx.tensor_from_f32(&residual, rows, cols).unwrap();
        let gpu_upd = ctx.tensor_from_f32(&update, rows, cols).unwrap();
        let gpu_gate = ctx.tensor_from_f32(&gate, modulation_rows, cols).unwrap();
        let out = gated_residual_segmented_f16(&ctx, &gpu_res, &gpu_upd, &gpu_gate, &segments).unwrap();
        check_close("gated_residual_segmented", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn modulation_chunks_matches_oracle() {
        let ctx = ctx();
        let (rows, modalities, chunks, hidden) = (2usize, 2, 3, 4);
        let cols = modalities * chunks * hidden;
        let input = seq(rows * cols);
        let expect = cpu::modulation_chunks(&input, modalities, chunks, hidden).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, rows, cols).unwrap();
        let outs = modulation_chunks_f16(&ctx, &gpu_in, modalities, chunks, hidden).unwrap();
        assert_eq!(outs.len(), chunks);
        for (c, out) in outs.iter().enumerate() {
            check_close(&format!("modulation_chunks[{c}]"), &ctx.tensor_to_f32(out).unwrap(), &expect[c]);
        }
    }

    #[test]
    fn rmsnorm_heads_matches_oracle() {
        let ctx = ctx();
        let (rows, head_count, head_dim) = (3usize, 2, 4);
        let cols = head_count * head_dim;
        let input = seq(rows * cols);
        let weight = seq(head_dim);
        let expect = cpu::rmsnorm_heads(&input, &weight, head_count, head_dim, 1e-5).unwrap();
        let gpu_in = ctx.tensor_from_f32(&input, rows, cols).unwrap();
        let weight_gpu = ctx.stream().clone_htod::<half::f16, _>(&weight.iter().map(|v| half::f16::from_f32(*v)).collect::<Vec<_>>()).unwrap();
        let out = rmsnorm_heads_f16(&ctx, &gpu_in, &weight_gpu, head_count, head_dim, 1e-5).unwrap();
        check_close("rmsnorm_heads", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn flow_step_matches_reference() {
        let ctx = ctx();
        let (rows, cols) = (4usize, 8);
        let sample = seq(rows * cols);
        let velocity = seq(rows * cols);
        let scale = 0.37f32;
        let expect: Vec<f32> = sample.iter().zip(&velocity).map(|(&s, &v)| s + scale * v).collect();
        let gpu_s = ctx.tensor_from_f32(&sample, rows, cols).unwrap();
        let gpu_v = ctx.tensor_from_f32(&velocity, rows, cols).unwrap();
        let out = flow_step_f16(&ctx, &gpu_s, &gpu_v, scale).unwrap();
        check_close("flow_step", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn silu_matches_oracle() {
        let ctx = ctx();
        let n = 32usize;
        let input = seq(n);
        let expect = cpu::silu(&input);
        let gpu_in = ctx.tensor_from_f32(&input, 2, n / 2).unwrap();
        let out = silu_f16(&ctx, &gpu_in).unwrap();
        check_close("silu", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn concat_rows_matches_reference() {
        let ctx = ctx();
        let cols = 8usize;
        let left = seq(2 * cols);
        let right = seq(3 * cols);
        let mut expect = left.clone();
        expect.extend_from_slice(&right);
        let gpu_l = ctx.tensor_from_f32(&left, 2, cols).unwrap();
        let gpu_r = ctx.tensor_from_f32(&right, 3, cols).unwrap();
        let out = concat_rows_f16(&ctx, &gpu_l, &gpu_r).unwrap();
        check_close("concat_rows", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }

    #[test]
    fn timestep_embedding_matches_oracle() {
        let ctx = ctx();
        let timesteps = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let dim = 256usize;
        let expect = cpu::timestep_embedding(&timesteps, dim).unwrap();
        let out = timestep_embedding_f16(&ctx, &timesteps, dim).unwrap();
        check_close("timestep_embedding", &ctx.tensor_to_f32(&out).unwrap(), &expect);
    }
}
