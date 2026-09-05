/// 本模块的 CUDA shader(本文件用到的 kernel + 文件私有 `__device__` helper)。
///
/// 共用 helper 见 `mod.rs` 前导。`mod.rs` 的 `kernels_source()`
/// 把 mod.rs 的 `PREAMBLE_SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: add_f16, add_scaled_f16, silu_mul_f16, gelu_tanh_mul_f16, split_columns_f16, rmsnorm_f16, layer_norm_f16, apply_rope_partial_f16, select_row_f16, argmax_f16, split_interleaved_columns_f16, concat_columns_f16, apply_rope_prefix_f16
pub const SHADERS: &str = r#"
extern "C" __global__ void add_f16(
    const __half * __restrict__ a,
    const __half * __restrict__ b,
    __half * __restrict__ output,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = __float2half(__half2float(a[id]) + __half2float(b[id]));
}
extern "C" __global__ void add_scaled_f16(
    const __half * __restrict__ a,
    const __half * __restrict__ b,
    __half * __restrict__ output,
    const float scale,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = __float2half((__half2float(a[id]) + __half2float(b[id])) * scale);
}
extern "C" __global__ void silu_mul_f16(
    const __half * __restrict__ gate,
    const __half * __restrict__ up,
    __half * __restrict__ output,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    float g = __half2float(gate[id]);
    float u = __half2float(up[id]);
    // SwiGLU:silu(g) * u = (g * sigmoid(g)) * u = (g / (1 + e^-g)) * u。
    output[id] = __float2half((g / (1.0f + expf(-g))) * u);
}
extern "C" __global__ void gelu_tanh_mul_f16(
    const __half * __restrict__ gate,
    const __half * __restrict__ up,
    __half * __restrict__ output,
    const unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    float g = __half2float(gate[id]);
    float u = __half2float(up[id]);
    // 与 CPU gelu_tanh_mul 一致的 tanh 近似。
    const float c = 0.7978845608028654f;
    float gelu = 0.5f * g * (1.0f + tanhf(c * (g + 0.044715f * g * g * g)));
    output[id] = __float2half(gelu * u);
}
extern "C" __global__ void split_columns_f16(
    const __half * __restrict__ input,
    __half * __restrict__ left,
    __half * __restrict__ right,
    const unsigned int rows,
    const unsigned int left_cols,
    const unsigned int right_cols,
    const unsigned int total_cols)
{
    unsigned int total = rows * (left_cols > right_cols ? left_cols : right_cols);
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total) return;
    unsigned int row = id / (left_cols > right_cols ? left_cols : right_cols);
    unsigned int col_in_block = id % (left_cols > right_cols ? left_cols : right_cols);
    if (row >= rows) return;
    if (col_in_block < left_cols) {
        left[row * left_cols + col_in_block] = input[row * total_cols + col_in_block];
    }
    if (col_in_block < right_cols) {
        right[row * right_cols + col_in_block] = input[row * total_cols + left_cols + col_in_block];
    }
}
extern "C" __global__ void rmsnorm_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int columns,
    const float epsilon)
{
    extern __shared__ float warp_sums[];
    unsigned int row = blockIdx.x;
    unsigned int lane = threadIdx.x;
    unsigned int width = blockDim.x;
    unsigned long long offset = (unsigned long long)row * columns;

    // 1. 每线程累加一段列的 x²。
    float sum = 0.0f;
    for (unsigned int c = lane; c < columns; c += width) {
        float v = __half2float(input[offset + c]);
        sum += v * v;
    }

    // 2. block 内归约:先 warp shuffle 到每 warp 1 个值,再共享内存归约 warp 间。
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

    // 3. 写出:scale = rsqrt(mean(x²) + eps),output = x * scale * weight。
    float scale = rsqrtf(total / float(columns) + epsilon);
    for (unsigned int c = lane; c < columns; c += width) {
        output[offset + c] = __float2half(__half2float(input[offset + c]) * scale * __half2float(weight[c]));
    }
}
// 一次 launch 处理所有 (row, seg) 对的 (left + rmsnorm(right, weight)) * scale
// 等价 (left + right * rsqrt(mean(right²)+eps) * (1+weight)) * scale
// 替代 mod.rs:476 默认实现的 1 次 split + 2N 次 rmsnorm_add_scaled kernel launch
// (N=segments, 典型 42), 单 kernel 把 reduction 走 block shared mem + warp shuffle
// 输出写到 out_ptrs 数组指向的 N 个预分配 [rows, segment_columns] tensor
extern "C" __global__ void segmented_rmsnorm_add_scaled_f16(
    const __half * __restrict__ left,
    const __half * __restrict__ right,
    const __half * __restrict__ weight,
    __half * const * __restrict__ out_ptrs,
    const unsigned int rows,
    const unsigned int segments,
    const unsigned int segment_columns,
    const float epsilon,
    const float scale)
{
    extern __shared__ float warp_sums[];
    unsigned int row = blockIdx.x;
    unsigned int seg = blockIdx.y;
    unsigned int lane = threadIdx.x;
    if (row >= rows || seg >= segments) return;
    unsigned int width = blockDim.x;
    // 每行 layout: [seg0, seg1, ..., segN-1], 每段 segment_columns
    unsigned long long seg_offset = ((unsigned long long)row * segments + seg) * segment_columns;
    __half * __restrict__ out = out_ptrs[seg] + (unsigned long long)row * segment_columns;

    // 1. 算 rmsnorm 的 mean(right²) for this (row, seg)
    float sum_sq = 0.0f;
    for (unsigned int c = lane; c < segment_columns; c += width) {
        float v = __half2float(right[seg_offset + c]);
        sum_sq += v * v;
    }

    // 2. block reduce (warp shuffle + shared mem)
    for (int off = 16; off > 0; off >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
    }
    if (lane % 32 == 0) warp_sums[lane / 32] = sum_sq;
    __syncthreads();
    int num_warps = (width + 31) / 32;
    if (lane < 32) {
        float total = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, off);
        }
        if (lane == 0) warp_sums[0] = total;
    }
    __syncthreads();
    float total = warp_sums[0];
    float rsqrt_val = rsqrtf(total / float(segment_columns) + epsilon) * scale;

    // 3. 写 out = (left + right * rsqrt * (1 + weight))  (scale 已合并到 rsqrt)
    for (unsigned int c = lane; c < segment_columns; c += width) {
        float l = __half2float(left[seg_offset + c]);
        float r = __half2float(right[seg_offset + c]);
        float w = __half2float(weight[c]);
        out[c] = __float2half(l + r * rsqrt_val * (1.0f + w));
    }
}
// 每行一层 LayerNorm:out = (x-mean)*rsqrt(var+eps)*weight[c]+bias[c]。
extern "C" __global__ void layer_norm_f16(
    const __half * __restrict__ input,
    const __half * __restrict__ weight,
    const __half * __restrict__ bias,
    __half * __restrict__ output,
    const unsigned int cols,
    const float eps)
{
    extern __shared__ float shared[];
    float *sum_acc = shared;
    float *sqsum_acc = shared + blockDim.x;
    unsigned int row = blockIdx.x;
    unsigned int lane = threadIdx.x;
    unsigned long long base = (unsigned long long)row * cols;
    float sum = 0.0f, sqsum = 0.0f;
    for (unsigned int c = lane; c < cols; c += blockDim.x) {
        float v = __half2float(input[base + c]);
        sum += v;
        sqsum += v * v;
    }
    sum_acc[lane] = sum;
    sqsum_acc[lane] = sqsum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sum_acc[lane] += sum_acc[lane + stride];
            sqsum_acc[lane] += sqsum_acc[lane + stride];
        }
        __syncthreads();
    }
    float mean = sum_acc[0] / (float)cols;
    float variance = fmaxf(sqsum_acc[0] / (float)cols - mean * mean, 0.0f);
    float inv_std = rsqrtf(variance + eps);
    for (unsigned int c = lane; c < cols; c += blockDim.x) {
        float v = __half2float(input[base + c]);
        output[base + c] = __float2half((v - mean) * inv_std * __half2float(weight[c]) + __half2float(bias[c]));
    }
}
__device__ __forceinline__ void apply_rope_f16_impl(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int rows,
    const unsigned int columns,
    const unsigned int head_count,
    const unsigned int rotary_dim,
    const unsigned int position_offset,
    const __half * __restrict__ cos_data,
    const __half * __restrict__ sin_data,
    const bool prefix)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = rows * columns;
    if (idx >= count) return;

    unsigned int row = idx / columns;
    unsigned int column = idx - row * columns;
    unsigned int head_dim = columns / head_count;
    unsigned int head = column / head_dim;
    unsigned int head_column = column - head * head_dim;
    unsigned int rotary_start = prefix ? 0 : head_dim - rotary_dim;
    if (head_column < rotary_start || head_column >= rotary_start + rotary_dim) {
        output[idx] = input[idx];
        return;
    }

    unsigned int half_dim = rotary_dim >> 1;
    unsigned int component = head_column - rotary_start;
    unsigned int pair = component < half_dim ? component : component - half_dim;
    unsigned long long base = (unsigned long long)row * columns + (unsigned long long)head * head_dim + rotary_start;
    unsigned int angle = (position_offset + row) * half_dim + pair;
    float even = __half2float(input[base + pair]);
    float odd = __half2float(input[base + half_dim + pair]);
    float c = __half2float(cos_data[angle]);
    float s = __half2float(sin_data[angle]);
    output[idx] = __float2half(component < half_dim ? (even * c - odd * s) : (even * s + odd * c));
}
extern "C" __global__ void apply_rope_partial_f16(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int rows,
    const unsigned int columns,
    const unsigned int head_count,
    const unsigned int rotary_dim,
    const unsigned int position_offset,
    const __half * __restrict__ cos_data,
    const __half * __restrict__ sin_data)
{
    apply_rope_f16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, false);
}
extern "C" __global__ void select_row_f16(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int row,
    const unsigned int cols)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    output[col] = input[(unsigned long long)row * cols + col];
}
// argmax:屏蔽 excluded 中的 token(置 -inf)。excluded 为空指针 + 计数 0 时即普通 argmax。
// excluded 设备数组 + 计数;decode 文本生成禁止输出 image/video 等特殊 token 时由 runtime 传入(通常 ≤ 数个)。
extern "C" __global__ void add_f32_f16(const float *a, const __half *b, float *output, unsigned int count) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = a[id] + __half2float(b[id]);
}
extern "C" __global__ void add_f32_f32(const float *a, const float *b, float *output, unsigned int count) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = a[id] + b[id];
}
extern "C" __global__ void copy_f32(const float *input, float *output, unsigned int count) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] = input[id];
}
extern "C" __global__ void argmax_f16(
    const __half * __restrict__ input,
    unsigned int * __restrict__ output,
    const unsigned int cols,
    const unsigned int * __restrict__ excluded,
    const unsigned int excluded_count)
{
    extern __shared__ unsigned char smem[];
    float *vals = reinterpret_cast<float *>(smem);
    unsigned int *idxs = reinterpret_cast<unsigned int *>(smem + blockDim.x * sizeof(float));

    unsigned int tid = threadIdx.x;
    unsigned int width = blockDim.x;
    // NVRTC 不定义 INFINITY,用位模式构造负无穷(IEEE 754:0xff800000)。
    float NEG_INF = __int_as_float(0xff800000);
    float best_val = NEG_INF;
    unsigned int best_idx = 0;
    // 每线程跨步扫描,保留各自段内的 max 及其索引。
    for (unsigned int c = tid; c < cols; c += width) {
        float v = __half2float(input[c]);
        for (unsigned int e = 0; e < excluded_count; e++) {
            if (excluded[e] == c) { v = NEG_INF; break; }
        }
        if (v > best_val) {
            best_val = v;
            best_idx = c;
        }
    }

    // warp 内归约:max 优先,并列时取较小索引(保持稳定)。
    for (int off = 16; off > 0; off >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, off);
        unsigned int other_idx = __shfl_down_sync(0xffffffff, best_idx, off);
        if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }
    unsigned int num_warps = (width + 31) / 32;
    if (tid % 32 == 0) {
        vals[tid / 32] = best_val;
        idxs[tid / 32] = best_idx;
    }
    __syncthreads();

    if (tid < 32) {
        float w_val = (tid < num_warps) ? vals[tid] : NEG_INF;
        unsigned int w_idx = (tid < num_warps) ? idxs[tid] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, w_val, off);
            unsigned int oi = __shfl_down_sync(0xffffffff, w_idx, off);
            if (ov > w_val || (ov == w_val && oi < w_idx)) {
                w_val = ov;
                w_idx = oi;
            }
        }
        if (tid == 0) *output = w_idx;
    }
}
extern "C" __global__ void split_interleaved_columns_f16(const __half *input, __half *left, __half *right, unsigned int rows, unsigned int columns, unsigned int block_columns) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= rows * columns) return;
    unsigned int row = id / columns;
    unsigned int column = id - row * columns;
    unsigned int pair_columns = block_columns * 2;
    unsigned int pair = column / pair_columns;
    unsigned int in_pair = column - pair * pair_columns;
    unsigned int output_columns = columns / 2;
    unsigned int output_column = pair * block_columns + in_pair % block_columns;
    if (in_pair < block_columns) left[(unsigned long long)row * output_columns + output_column] = input[id];
    else right[(unsigned long long)row * output_columns + output_column] = input[id];
}
extern "C" __global__ void concat_columns_f16(const __half *left, const __half *right, __half *output, unsigned int rows, unsigned int left_columns, unsigned int right_columns) {
    unsigned int columns = left_columns + right_columns;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= rows * columns) return;
    unsigned int row = id / columns;
    unsigned int column = id - row * columns;
    output[id] = column < left_columns
        ? left[(unsigned long long)row * left_columns + column]
        : right[(unsigned long long)row * right_columns + column - left_columns];
}
extern "C" __global__ void apply_rope_prefix_f16(
    const __half * __restrict__ input,
    __half * __restrict__ output,
    const unsigned int rows,
    const unsigned int columns,
    const unsigned int head_count,
    const unsigned int rotary_dim,
    const unsigned int position_offset,
    const __half * __restrict__ cos_data,
    const __half * __restrict__ sin_data)
{
    apply_rope_f16_impl(input, output, rows, columns, head_count, rotary_dim, position_offset, cos_data, sin_data, true);
}
"#;

use super::{CudaContext, CudaSliceF16, CudaTensor, LaunchConfig, PushKernelArg, THREADS};

pub(super) fn grid_1d(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(THREADS), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }
}

// ===== elementwise 算子 =====

/// output = a + b。三者同形 [rows, cols]。
/// f32 残差流加法:任一侧 f32 即输出 f32(Laguna 残差流防 f16 悬崖)。
pub fn add_residual(ctx: &CudaContext, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!("CUDA add_residual shape [{},{}] vs [{},{}]", a.rows, a.cols, b.rows, b.cols));
    }
    let count = a.rows.checked_mul(a.cols).ok_or("CUDA add_residual 大小溢出")?;
    match (&a.slice_f32, &b.slice_f32) {
        (Some(left), Some(right)) => {
            let mut out = ctx.buffer_uninit_f32(count)?;
            let func = ctx.function("add_f32_f32")?;
            unsafe {
                ctx.stream().launch_builder(&func).arg(left).arg(right).arg(&mut out).arg(&(count as u32)).launch(grid_1d(count)).map_err(|e| format!("launch add_f32_f32: {e:?}"))?;
            }
            let placeholder = ctx.placeholder_f16()?;
            Ok(CudaTensor::new_f32_residual(out, placeholder, a.rows, a.cols))
        }
        (Some(left), None) => {
            let mut out = ctx.buffer_uninit_f32(count)?;
            let func = ctx.function("add_f32_f16")?;
            unsafe {
                ctx.stream().launch_builder(&func).arg(left).arg(&b.slice).arg(&mut out).arg(&(count as u32)).launch(grid_1d(count)).map_err(|e| format!("launch add_f32_f16: {e:?}"))?;
            }
            let placeholder = ctx.placeholder_f16()?;
            Ok(CudaTensor::new_f32_residual(out, placeholder, a.rows, a.cols))
        }
        (None, Some(right)) => {
            let mut out = ctx.buffer_uninit_f32(count)?;
            let func = ctx.function("add_f32_f16")?;
            unsafe {
                ctx.stream().launch_builder(&func).arg(right).arg(&a.slice).arg(&mut out).arg(&(count as u32)).launch(grid_1d(count)).map_err(|e| format!("launch add_f32_f16(swap): {e:?}"))?;
            }
            let placeholder = ctx.placeholder_f16()?;
            Ok(CudaTensor::new_f32_residual(out, placeholder, a.rows, a.cols))
        }
        (None, None) => add_f16(ctx, a, b),
    }
}

/// f32 设备切片拷贝(decode 累加器零拷贝不可行时的收尾)。
pub fn copy_f32_slice(ctx: &CudaContext, input: &cudarc::driver::safe::CudaSlice<f32>, count: usize) -> Result<cudarc::driver::safe::CudaSlice<f32>, String> {
    let mut out = ctx.buffer_uninit_f32(count)?;
    let func = ctx.function("copy_f32")?;
    unsafe {
        ctx.stream().launch_builder(&func).arg(input).arg(&mut out).arg(&(count as u32)).launch(grid_1d(count)).map_err(|e| format!("launch copy_f32: {e:?}"))?;
    }
    Ok(out)
}

pub fn add_f16(ctx: &CudaContext, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!("add shape 不匹配: a=[{},{}] b=[{},{}]", a.rows, a.cols, b.rows, b.cols));
    }
    let output = ctx.tensor_alloc(a.rows, a.cols)?;
    let func = ctx.function("add_f16")?;
    let count = (a.rows * a.cols) as u32;
    let cfg = grid_1d(a.rows * a.cols);
    unsafe {
        ctx.stream().launch_builder(&func).arg(&a.slice).arg(&b.slice).arg(&output.slice).arg(&count).launch(cfg).map_err(|e| format!("launch add_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

pub fn add_scaled_f16(ctx: &CudaContext, a: &CudaTensor, b: &CudaTensor, scale: f32) -> Result<CudaTensor, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!("add_scaled shape 不匹配: a=[{},{}] b=[{},{}]", a.rows, a.cols, b.rows, b.cols));
    }
    if !scale.is_finite() {
        return Err(format!("add_scaled scale={scale} 非法"));
    }
    let output = ctx.tensor_alloc(a.rows, a.cols)?;
    let func = ctx.function("add_scaled_f16")?;
    let count = (a.rows * a.cols) as u32;
    let cfg = grid_1d(a.rows * a.cols);
    unsafe {
        ctx.stream().launch_builder(&func).arg(&a.slice).arg(&b.slice).arg(&output.slice).arg(&scale).arg(&count).launch(cfg).map_err(|e| format!("launch add_scaled_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// output = silu(gate) * up。
pub fn silu_mul_f16(ctx: &CudaContext, gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor, String> {
    if gate.rows != up.rows || gate.cols != up.cols {
        return Err(format!("silu_mul shape 不匹配: gate=[{},{}] up=[{},{}]", gate.rows, gate.cols, up.rows, up.cols));
    }
    let output = ctx.tensor_alloc(gate.rows, gate.cols)?;
    let func = ctx.function("silu_mul_f16")?;
    let count = (gate.rows * gate.cols) as u32;
    let cfg = grid_1d(gate.rows * gate.cols);
    unsafe {
        ctx.stream().launch_builder(&func).arg(&gate.slice).arg(&up.slice).arg(&output.slice).arg(&count).launch(cfg).map_err(|e| format!("launch silu_mul_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// output = gelu_tanh(gate) * up。
pub fn gelu_tanh_mul_f16(ctx: &CudaContext, gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor, String> {
    if gate.rows != up.rows || gate.cols != up.cols {
        return Err(format!("gelu_tanh_mul shape 不匹配: gate=[{},{}] up=[{},{}]", gate.rows, gate.cols, up.rows, up.cols));
    }
    let output = ctx.tensor_alloc(gate.rows, gate.cols)?;
    let func = ctx.function("gelu_tanh_mul_f16")?;
    let count = (gate.rows * gate.cols) as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&gate.slice).arg(&up.slice).arg(&output.slice).arg(&count).launch(grid_1d(gate.rows * gate.cols)).map_err(|e| format!("launch gelu_tanh_mul_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 切分列:input [rows, total_cols] → (left [rows, left_cols], right [rows, right_cols])。
pub fn split_columns_f16(ctx: &CudaContext, input: &CudaTensor, left_cols: usize, right_cols: usize) -> Result<(CudaTensor, CudaTensor), String> {
    let total = left_cols + right_cols;
    if input.cols != total {
        return Err(format!("split_columns input cols={}，期望 left+right={total}", input.cols));
    }
    let left = ctx.tensor_alloc(input.rows, left_cols)?;
    let right = ctx.tensor_alloc(input.rows, right_cols)?;
    let func = ctx.function("split_columns_f16")?;
    let max_cols = left_cols.max(right_cols);
    let rows = input.rows as u32;
    let lc = left_cols as u32;
    let rc = right_cols as u32;
    let tc = total as u32;
    let cfg = grid_1d(input.rows * max_cols);
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(&left.slice).arg(&right.slice).arg(&rows).arg(&lc).arg(&rc).arg(&tc).launch(cfg).map_err(|e| format!("launch split_columns_f16 失败: {e:?}"))?;
    }
    Ok((left, right))
}

/// RMSNorm:每行独立归一化。weight 形状 [columns]。
pub fn rmsnorm_f16(ctx: &CudaContext, input: &CudaTensor, weight: &CudaSliceF16, eps: f32) -> Result<CudaTensor, String> {
    if input.cols == 0 {
        return Err("rmsnorm columns=0".to_string());
    }
    let output = ctx.tensor_alloc(input.rows, input.cols)?;
    let func = ctx.function("rmsnorm_f16")?;
    let columns = input.cols as u32;
    // block 内归约需要共享内存:warp_sums[blockDim.x/32] 个 float。
    let shared_bytes = (THREADS as usize / 32) * std::mem::size_of::<f32>();
    let cfg = LaunchConfig { grid_dim: (input.rows as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: shared_bytes as u32 };
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(weight).arg(&output.slice).arg(&columns).arg(&eps).launch(cfg).map_err(|e| format!("launch rmsnorm_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 分段 RMSNorm + 残差 + scale 融合:每 (row, seg) 独立算 rmsnorm(right) 然后
/// (left + rmsnorm(right) * weight) * scale 写到 output[seg]。一 kernel launch
/// 处理所有 (row, seg), 替代 mod.rs:476 默认实现的 1 次 split + 2N 次 rmsnorm_add_scaled
/// (N=segments, 典型 42) 的 kernel launch + 中间 tensor 分配。
pub fn segmented_rmsnorm_add_scaled_f16(ctx: &CudaContext, left: &CudaTensor, right: &CudaTensor, weight: &CudaSliceF16, segments: usize, segment_columns: usize, eps: f32, scale: f32) -> Result<Vec<CudaTensor>, String> {
    if segments == 0 || segment_columns == 0 || left.cols != segments * segment_columns || right.cols != segments * segment_columns {
        return Err(format!("segmented_rmsnorm_add_scaled shape 不兼容: left=[{},{}] right=[{},{}] segments={segments} segment_columns={segment_columns}", left.rows, left.cols, right.rows, right.cols));
    }
    if weight.len() != segment_columns {
        return Err(format!("segmented_rmsnorm_add_scaled weight 长度={} 期望 segment_columns={segment_columns}", weight.len()));
    }
    // 1. 预分配 N 个 [rows, segment_columns] 目标 tensor
    let mut out_tensors: Vec<CudaTensor> = Vec::with_capacity(segments);
    for _ in 0..segments {
        out_tensors.push(ctx.tensor_alloc(left.rows, segment_columns)?);
    }
    // 2. 拼指针数组: cudarc DevicePtr trait 拿 raw device pointer (CUdeviceptr = u64)
    use cudarc::driver::safe::DevicePtr;
    let mut ptrs: Vec<u64> = Vec::with_capacity(segments);
    let mut guards = Vec::with_capacity(segments); // 保留 SyncOnDrop 防异步竞态
    for t in &out_tensors {
        let (dev_ptr, sync_guard) = t.slice.device_ptr(ctx.stream());
        guards.push(sync_guard);
        ptrs.push(dev_ptr as usize as u64);
    }
    drop(guards); // 拿到 ptr 后立刻释放 (kernel launch 时 device memory 仍 owned by CudaTensor)
    let ptrs_dev = ctx.stream().clone_htod::<u64, _>(&ptrs).map_err(|e| format!("segmented_rmsnorm_add_scaled H2D ptrs: {e:?}"))?;
    // 3. 一次 kernel launch 写 N 个 segment
    let func = ctx.function("segmented_rmsnorm_add_scaled_f16")?;
    let shared_bytes = (THREADS as usize / 32) * std::mem::size_of::<f32>();
    let cfg = LaunchConfig { grid_dim: (left.rows as u32, segments as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: shared_bytes as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&left.slice)
            .arg(&right.slice)
            .arg(weight)
            .arg(&ptrs_dev)
            .arg(&(left.rows as u32))
            .arg(&(segments as u32))
            .arg(&(segment_columns as u32))
            .arg(&eps)
            .arg(&scale)
            .launch(cfg)
            .map_err(|e| format!("launch segmented_rmsnorm_add_scaled_f16 失败: {e:?}"))?;
    }
    Ok(out_tensors)
}

/// LayerNorm:每行独立归一化后施加 weight 与 bias。
pub fn layer_norm_f16(ctx: &CudaContext, input: &CudaTensor, weight: &CudaSliceF16, bias: &CudaSliceF16, eps: f32) -> Result<CudaTensor, String> {
    if input.cols == 0 || weight.len() != input.cols || bias.len() != input.cols {
        return Err(format!("CUDA layer_norm input=[{},{}] weight={} bias={}", input.rows, input.cols, weight.len(), bias.len()));
    }
    let output = ctx.tensor_alloc(input.rows, input.cols)?;
    let func = ctx.function("layer_norm_f16")?;
    let cfg = LaunchConfig { grid_dim: (input.rows as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (2 * THREADS as usize * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(weight).arg(bias).arg(&output.slice).arg(&(input.cols as u32)).arg(&eps).launch(cfg).map_err(|e| format!("launch layer_norm_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 把交错块 `[left, right]` 完全留在设备上拆成两组列。
pub fn split_interleaved_columns_f16(ctx: &CudaContext, input: &CudaTensor, block_columns: usize) -> Result<(CudaTensor, CudaTensor), String> {
    let pair_columns = block_columns.checked_mul(2).ok_or("CUDA interleaved split block 溢出")?;
    if block_columns == 0 || !input.cols.is_multiple_of(pair_columns) {
        return Err(format!("CUDA interleaved split cols={} block={block_columns} 非法", input.cols));
    }
    let output_columns = input.cols / 2;
    let left = ctx.tensor_alloc(input.rows, output_columns)?;
    let right = ctx.tensor_alloc(input.rows, output_columns)?;
    let func = ctx.function("split_interleaved_columns_f16")?;
    let rows = input.rows as u32;
    let columns = input.cols as u32;
    let block = block_columns as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(&left.slice)
            .arg(&right.slice)
            .arg(&rows)
            .arg(&columns)
            .arg(&block)
            .launch(grid_1d(input.rows * input.cols))
            .map_err(|e| format!("launch split_interleaved_columns_f16 失败: {e:?}"))?;
    }
    Ok((left, right))
}

/// 按行拼接两张设备 tensor。
pub fn concat_columns_f16(ctx: &CudaContext, left: &CudaTensor, right: &CudaTensor) -> Result<CudaTensor, String> {
    if left.rows != right.rows {
        return Err(format!("CUDA concat rows {} 与 {} 不一致", left.rows, right.rows));
    }
    let columns = left.cols.checked_add(right.cols).ok_or("CUDA concat columns 溢出")?;
    let output = ctx.tensor_alloc(left.rows, columns)?;
    let func = ctx.function("concat_columns_f16")?;
    let rows = left.rows as u32;
    let left_columns = left.cols as u32;
    let right_columns = right.cols as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&left.slice)
            .arg(&right.slice)
            .arg(&output.slice)
            .arg(&rows)
            .arg(&left_columns)
            .arg(&right_columns)
            .launch(grid_1d(left.rows * columns))
            .map_err(|e| format!("launch concat_columns_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// RoPE:对每 head 尾部 rotary_dim 维做旋转。
/// cos/sin 是 host f32,内部转 f16 上传。position_offset 是序列起始位置。
pub fn apply_rope_partial_f16(ctx: &CudaContext, input: &CudaTensor, head_count: usize, rotary_dim: usize, position_offset: usize, cos: &[f32], sin: &[f32]) -> Result<CudaTensor, String> {
    apply_rope_f16(ctx, input, head_count, rotary_dim, position_offset, cos, sin, false)
}

/// 对每个 head 的前 rotary_dim 维做 RoPE。
pub fn apply_rope_prefix_f16(ctx: &CudaContext, input: &CudaTensor, head_count: usize, rotary_dim: usize, position_offset: usize, cos: &[f32], sin: &[f32]) -> Result<CudaTensor, String> {
    apply_rope_f16(ctx, input, head_count, rotary_dim, position_offset, cos, sin, true)
}

#[allow(clippy::too_many_arguments)]
fn apply_rope_f16(ctx: &CudaContext, input: &CudaTensor, head_count: usize, rotary_dim: usize, position_offset: usize, cos: &[f32], sin: &[f32], prefix: bool) -> Result<CudaTensor, String> {
    let name = if prefix { "rope_prefix" } else { "rope" };
    if head_count == 0 || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || !input.cols.is_multiple_of(head_count) || rotary_dim > input.cols / head_count {
        return Err(format!("CUDA {name} shape input=[{},{}] heads={head_count} rotary={rotary_dim} 非法", input.rows, input.cols));
    }
    let half_dim = rotary_dim / 2;
    let table_begin = position_offset.checked_mul(half_dim).ok_or_else(|| format!("CUDA {name} table offset 溢出"))?;
    let table_end = position_offset.checked_add(input.rows).and_then(|rows| rows.checked_mul(half_dim)).ok_or_else(|| format!("CUDA {name} table end 溢出"))?;
    let cos = cos.get(table_begin..table_end).ok_or_else(|| format!("CUDA {name} cos={}，需要 {table_begin}..{table_end}", cos.len()))?;
    let sin = sin.get(table_begin..table_end).ok_or_else(|| format!("CUDA {name} sin={}，需要 {table_begin}..{table_end}", sin.len()))?;
    let output = ctx.tensor_alloc(input.rows, input.cols)?;
    let pipeline = if prefix { "apply_rope_prefix_f16" } else { "apply_rope_partial_f16" };
    let func = ctx.function(pipeline)?;
    let rows = u32::try_from(input.rows).map_err(|_| format!("CUDA {name} rows 超过 u32"))?;
    let columns = u32::try_from(input.cols).map_err(|_| format!("CUDA {name} columns 超过 u32"))?;
    let hc = u32::try_from(head_count).map_err(|_| format!("CUDA {name} head_count 超过 u32"))?;
    let rd = u32::try_from(rotary_dim).map_err(|_| format!("CUDA {name} rotary_dim 超过 u32"))?;
    let pos = 0u32;
    // 表窗口设备驻留缓存:同一窗口跨层复用,避免逐层 pageable 上传的重型负载 stall。
    let cos_gpu = ctx.rope_window_f16(cos).map_err(|e| format!("{name} cos 窗口上传失败: {e}"))?;
    let sin_gpu = ctx.rope_window_f16(sin).map_err(|e| format!("{name} sin 窗口上传失败: {e}"))?;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(&output.slice)
            .arg(&rows)
            .arg(&columns)
            .arg(&hc)
            .arg(&rd)
            .arg(&pos)
            .arg(&*cos_gpu)
            .arg(&*sin_gpu)
            .launch(grid_1d(input.rows * input.cols))
            .map_err(|e| format!("launch {pipeline} 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 取一行:output[1, cols] = input[row, cols]。
pub fn select_row_f16(ctx: &CudaContext, input: &CudaTensor, row: usize) -> Result<CudaTensor, String> {
    if row >= input.rows {
        return Err(format!("select_row {row} 越界(共 {} 行)", input.rows));
    }
    let output = ctx.tensor_alloc(1, input.cols)?;
    let func = ctx.function("select_row_f16")?;
    let r = row as u32;
    let cols = input.cols as u32;
    let cfg = grid_1d(input.cols);
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(&output.slice).arg(&r).arg(&cols).launch(cfg).map_err(|e| format!("launch select_row_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// argmax 核心:屏蔽 `excluded` 中的 token(置 -inf);空集即普通 argmax。
fn argmax_impl(ctx: &CudaContext, input: &CudaTensor, excluded: &[u32]) -> Result<u32, String> {
    if input.rows != 1 {
        return Err(format!("argmax 期望单行,实际 {} 行", input.rows));
    }
    let cols = input.cols;
    if excluded.iter().any(|&token| token as usize >= cols) {
        return Err(format!("argmax 禁止 token 越界: cols={cols}, excluded={excluded:?}"));
    }
    // 分配 1 个 u32 输出。
    let mut idx_gpu = ctx.stream().alloc_zeros::<u32>(1).map_err(|e| format!("argmax 分配输出失败: {e:?}"))?;
    // excluded 为空时传 dummy 槽 + 计数 0,kernel 不会读它;非空时上传设备数组(通常 ≤ 数个)。
    let excluded_host: &[u32] = if excluded.is_empty() { &[0] } else { excluded };
    let excluded_gpu = ctx.stream().clone_htod::<u32, _>(excluded_host).map_err(|e| format!("argmax excluded 上传失败: {e:?}"))?;
    let func = ctx.function("argmax_f16")?;
    let cols_u32 = cols as u32;
    let excluded_count = excluded.len() as u32;
    // block 内归约:共享内存 = (THREADS floats) + (THREADS uints)。
    let shared_bytes = (THREADS as usize) * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: shared_bytes as u32 };
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(&mut idx_gpu).arg(&cols_u32).arg(&excluded_gpu).arg(&excluded_count).launch(cfg).map_err(|e| format!("launch argmax_f16 失败: {e:?}"))?;
    }
    // 同步并回读。
    ctx.synchronize().map_err(|e| format!("argmax 同步失败: {e:?}"))?;
    let result = ctx.stream().clone_dtoh::<u32, _>(&idx_gpu).map_err(|e| format!("argmax 回读失败: {e:?}"))?;
    Ok(result[0])
}

/// argmax:返回最大值索引。input [1, cols]。
pub fn argmax_f16(ctx: &CudaContext, input: &CudaTensor) -> Result<u32, String> {
    argmax_impl(ctx, input, &[])
}

/// argmax 但屏蔽 `excluded` 中的 token(置 -inf)。decode 文本生成禁止输出
/// image/video 等特殊 token 时由 runtime 传入。
pub fn argmax_excluding_f16(ctx: &CudaContext, input: &CudaTensor, excluded: &[u32]) -> Result<u32, String> {
    argmax_impl(ctx, input, excluded)
}
