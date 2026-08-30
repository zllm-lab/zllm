/// 本模块的 CUDA shader(本文件用到的 kernel + 文件私有 `__device__` helper)。
///
/// 共用 helper 见 `mod.rs` 前导。`mod.rs` 的 `kernels_source()`
/// 把 mod.rs 的 `PREAMBLE_SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: gather_rows_f16, gather_rows_f32, moe_route_softmax_topk_f32, scatter_add_rows_f32, scatter_add_route_f32, f32_to_f16
pub const SHADERS: &str = r#"
extern "C" __global__ void gather_rows_f16(const __half *input, const unsigned int *rows, __half *output, unsigned int columns, unsigned int count) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) output[id] = input[(unsigned long long)rows[id / columns] * columns + id % columns];
}
extern "C" __global__ void gather_rows_f32(const float *input, const unsigned int *rows, float *output, unsigned int columns, unsigned int count) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) output[id] = input[(unsigned long long)rows[id / columns] * columns + id % columns];
}
extern "C" __global__ void moe_route_softmax_topk_f32(
    const float *logits,
    const float *bias,
    unsigned int *expert_ids,
    float *route_weights,
    unsigned int rows,
    unsigned int num_experts,
    unsigned int top_k,
    float scaling_factor,
    unsigned int normalize_selected)
{
    extern __shared__ float values[];
    unsigned int row = blockIdx.x;
    unsigned int lane = threadIdx.x;
    if (row >= rows) return;
    if (lane < num_experts) values[lane] = logits[(unsigned long long)row * num_experts + lane] + bias[lane];
    __syncthreads();
    if (lane != 0) return;

    float maximum = -3.402823466e+38F;
    for (unsigned int expert = 0; expert < num_experts; ++expert) maximum = fmaxf(maximum, values[expert]);
    float denominator = 0.0f;
    if (!normalize_selected) {
        for (unsigned int expert = 0; expert < num_experts; ++expert) denominator += expf(values[expert] - maximum);
    }
    for (unsigned int slot = 0; slot < top_k; ++slot) {
        unsigned int best_expert = 0;
        float best_value = -3.402823466e+38F;
        for (unsigned int expert = 0; expert < num_experts; ++expert) {
            float value = values[expert];
            if (value > best_value || (value == best_value && expert < best_expert)) {
                best_value = value;
                best_expert = expert;
            }
        }
        unsigned int route = row * top_k + slot;
        expert_ids[route] = best_expert;
        route_weights[route] = best_value;
        values[best_expert] = -3.402823466e+38F;
    }
    if (normalize_selected) {
        for (unsigned int slot = 0; slot < top_k; ++slot) denominator += expf(route_weights[row * top_k + slot] - maximum);
    }
    for (unsigned int slot = 0; slot < top_k; ++slot) {
        unsigned int route = row * top_k + slot;
        route_weights[route] = expf(route_weights[route] - maximum) / denominator * scaling_factor;
    }
}
extern "C" __global__ void scatter_add_rows_f32(
    const __half *input,
    const unsigned int *rows,
    const float *weights,
    float *output,
    unsigned int columns,
    unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned int input_row = id / columns;
    unsigned int column = id - input_row * columns;
    atomicAdd(output + (unsigned long long)rows[input_row] * columns + column, __half2float(input[id]) * weights[input_row]);
}
extern "C" __global__ void scatter_add_route_f32(
    const __half *input,
    const float *route_weights,
    float *output,
    unsigned int route,
    unsigned int count)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    output[id] += __half2float(input[id]) * route_weights[route];
}
extern "C" __global__ void f32_to_f16(float *input, __half *output, unsigned int count, unsigned int clear_input) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    float value = input[id];
    output[id] = __float2half(value);
    if (clear_input) input[id] = 0.0f;
}
"#;

use super::linear::cublas_matmul_control_f32;
use super::tensor::grid_1d;
use super::{CudaContext, CudaTensor, LaunchConfig, PushKernelArg, THREADS};
use cudarc::driver::safe::CudaSlice;

/// MoE 路由全程留在 GPU，只回传每个 token 的 Top-K 编号和权重。
pub fn moe_route_softmax_topk_device_f32(
    ctx: &CudaContext,
    input: &CudaTensor,
    weight: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    num_experts: usize,
    top_k: usize,
    scaling_factor: f32,
    normalize_selected: bool,
) -> Result<(Vec<u32>, CudaSlice<f32>), String> {
    if num_experts == 0 || num_experts > THREADS as usize || top_k == 0 || top_k > num_experts {
        return Err(format!("CUDA softmax router experts={num_experts} top_k={top_k} 超出单 block 能力"));
    }
    if bias.len() != num_experts {
        return Err(format!("CUDA router bias={}，期望 {num_experts}", bias.len()));
    }
    let logits = cublas_matmul_control_f32(ctx, input, weight, num_experts)?;
    let logits = logits.slice_f32.as_ref().ok_or("CUDA router F32 GEMM 未返回 F32 logits")?;
    let route_count = input.rows.checked_mul(top_k).ok_or("CUDA router route 数量溢出")?;
    let expert_ids = ctx.buffer_uninit::<u32>(route_count).map_err(|e| format!("CUDA router ids 分配失败: {e:?}"))?;
    let route_weights = ctx.buffer_uninit::<f32>(route_count).map_err(|e| format!("CUDA router weights 分配失败: {e:?}"))?;
    let func = ctx.function("moe_route_softmax_topk_f32")?;
    let rows = input.rows as u32;
    let experts = num_experts as u32;
    let selected = top_k as u32;
    let normalize_selected = u32::from(normalize_selected);
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(logits)
            .arg(bias)
            .arg(&expert_ids)
            .arg(&route_weights)
            .arg(&rows)
            .arg(&experts)
            .arg(&selected)
            .arg(&scaling_factor)
            .arg(&normalize_selected)
            .launch(LaunchConfig { grid_dim: (input.rows as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (num_experts * std::mem::size_of::<f32>()) as u32 })
            .map_err(|e| format!("launch moe_route_softmax_topk_f32 失败: {e:?}"))?;
    }
    let expert_ids = ctx.stream().clone_dtoh(&expert_ids).map_err(|e| format!("CUDA router ids 下载失败: {e:?}"))?;
    Ok((expert_ids, route_weights))
}

pub fn moe_route_softmax_topk_f32(
    ctx: &CudaContext,
    input: &CudaTensor,
    weight: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    num_experts: usize,
    top_k: usize,
    scaling_factor: f32,
    normalize_selected: bool,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let (expert_ids, route_weights) = moe_route_softmax_topk_device_f32(ctx, input, weight, bias, num_experts, top_k, scaling_factor, normalize_selected)?;
    let route_weights = ctx.stream().clone_dtoh(&route_weights).map_err(|e| format!("CUDA router weights 下载失败: {e:?}"))?;
    Ok((expert_ids, route_weights))
}

pub fn scatter_add_rows_f32(ctx: &CudaContext, output: &cudarc::driver::safe::CudaSlice<f32>, output_rows: usize, output_cols: usize, input: &CudaTensor, rows: &[u32], weights: &[f32]) -> Result<(), String> {
    if input.cols != output_cols || rows.len() != input.rows || weights.len() != input.rows {
        return Err(format!("CUDA MoE scatter shape output=[{output_rows},{output_cols}] input=[{},{}] rows={} weights={} 不一致", input.rows, input.cols, rows.len(), weights.len(),));
    }
    if rows.iter().any(|&row| row as usize >= output_rows) {
        return Err(format!("CUDA MoE scatter row 超出 output rows={output_rows}"));
    }
    let row_gpu = ctx.stream().clone_htod(rows).map_err(|e| format!("CUDA MoE scatter rows 上传失败: {e:?}"))?;
    let weight_gpu = ctx.stream().clone_htod(weights).map_err(|e| format!("CUDA MoE scatter weights 上传失败: {e:?}"))?;
    let func = ctx.function("scatter_add_rows_f32")?;
    let columns = output_cols as u32;
    let count = input.len() as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(&row_gpu).arg(&weight_gpu).arg(output).arg(&columns).arg(&count).launch(grid_1d(input.len())).map_err(|e| format!("launch scatter_add_rows_f32 失败: {e:?}"))?;
    }
    Ok(())
}

pub fn scatter_add_route_f32(ctx: &CudaContext, output: &cudarc::driver::safe::CudaSlice<f32>, input: &CudaTensor, route_weights: &cudarc::driver::safe::CudaSlice<f32>, route: usize) -> Result<(), String> {
    if input.rows != 1 || output.len() != input.cols {
        return Err(format!("CUDA MoE decode scatter shape output={} input=[{},{}] 不一致", output.len(), input.rows, input.cols,));
    }
    if route >= route_weights.len() {
        return Err(format!("CUDA MoE decode route={route} 越界 weights={}", route_weights.len()));
    }
    let func = ctx.function("scatter_add_route_f32")?;
    let route = route as u32;
    let count = input.cols as u32;
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(route_weights).arg(output).arg(&route).arg(&count).launch(grid_1d(input.cols)).map_err(|e| format!("launch scatter_add_route_f32 失败: {e:?}"))?;
    }
    Ok(())
}
pub fn f32_to_f16(ctx: &CudaContext, input: &cudarc::driver::safe::CudaSlice<f32>, rows: usize, cols: usize, clear_input: bool) -> Result<CudaTensor, String> {
    let count = rows.checked_mul(cols).ok_or("CUDA f32_to_f16 大小溢出")?;
    if input.len() != count {
        return Err(format!("CUDA f32_to_f16 input={}，期望 {count}", input.len()));
    }
    let output = ctx.tensor_uninit(rows, cols)?;
    let func = ctx.function("f32_to_f16")?;
    let count = count as u32;
    let clear_input = u32::from(clear_input);
    unsafe {
        ctx.stream().launch_builder(&func).arg(input).arg(&output.slice).arg(&count).arg(&clear_input).launch(grid_1d(count as usize)).map_err(|e| format!("launch f32_to_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 1D launch 配置:(n + 255) / 256 个 block,每 block 256 线程。
pub fn gather_rows_f16(ctx: &CudaContext, input: &CudaTensor, rows: &[u32]) -> Result<CudaTensor, String> {
    if rows.iter().any(|&row| row as usize >= input.rows) {
        return Err(format!("CUDA gather_rows_f16 row 超出 input rows={}", input.rows));
    }
    let indices = ctx.stream().clone_htod(rows).map_err(|error| format!("CUDA gather rows upload: {error:?}"))?;
    let output = ctx.tensor_uninit(rows.len(), input.cols)?;
    let func = ctx.function("gather_rows_f16")?;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(&indices)
            .arg(&output.slice)
            .arg(&(input.cols as u32))
            .arg(&(output.len() as u32))
            .launch(grid_1d(output.len()))
            .map_err(|error| format!("launch gather_rows_f16: {error:?}"))?;
    }
    Ok(output)
}

/// 从 f32 权威张量选择多行，保持输出为 f32 设备张量。
pub fn gather_rows_f32(ctx: &CudaContext, input: &CudaTensor, rows: &[u32]) -> Result<CudaTensor, String> {
    let input_f32 = input.slice_f32.as_ref().ok_or("CUDA gather_rows_f32 input 无 slice_f32")?;
    if rows.iter().any(|&row| row as usize >= input.rows) {
        return Err(format!("CUDA gather_rows_f32 row 超出 input rows={}", input.rows));
    }
    let indices = ctx.stream().clone_htod(rows).map_err(|error| format!("CUDA gather rows upload: {error:?}"))?;
    let count = rows.len().checked_mul(input.cols).ok_or("CUDA gather_rows_f32 大小溢出")?;
    let output = ctx.buffer_uninit_f32(count)?;
    let func = ctx.function("gather_rows_f32")?;
    unsafe {
        ctx.stream().launch_builder(&func).arg(input_f32).arg(&indices).arg(&output).arg(&(input.cols as u32)).arg(&(count as u32)).launch(grid_1d(count)).map_err(|error| format!("launch gather_rows_f32: {error:?}"))?;
    }
    Ok(CudaTensor::new_f32_residual(output, ctx.placeholder_f16()?, rows.len(), input.cols))
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use super::*;

    #[test]
    fn gather_rows_f32_preserves_device_dtype() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        let values = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data = ctx.stream().clone_htod(&values).expect("上传 F32 tensor");
        let input = CudaTensor::new_f32_residual(data, ctx.placeholder_f16().expect("分配占位"), 3, 2);
        let output = gather_rows_f32(&ctx, &input, &[2, 0]).expect("设备端选择行");

        assert!(output.slice_f32.is_some());
        assert_eq!(ctx.tensor_to_f32(&output).expect("回读结果"), [4.0, 5.0, 0.0, 1.0]);
    }
}
