//! 分组残差混合与膨胀因果卷积。维度均由调用者传入。

use super::{CudaContext, CudaTensor, LaunchConfig, PushKernelArg};
use crate::backend::cuda::CudaWeight;
use cudarc::driver::safe::CudaSlice;

pub const SHADERS: &str = r#"
extern "C" __global__ void repeat_groups_f32(const __half *x, float *out, unsigned int rows, unsigned int width, unsigned int groups) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width * groups) out[i] = __half2float(x[(i / (width * groups)) * width + i % width]);
}
// 左侧每 token 的向量复制到每组,右侧每组保留独立值,按组拼接。
extern "C" __global__ void concat_broadcast_groups_f16(const __half *left, const __half *right, __half *output, unsigned int rows, unsigned int width, unsigned int groups) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * groups * width * 2u) return;
    const unsigned int group = i / (2u * width), column = i % (2u * width);
    output[i] = column < width ? left[(group / groups) * width + column] : right[group * width + column - width];
}
extern "C" __global__ void grouped_norm_f16(const __half *x16, const float *x32, const float *w, __half *out, unsigned int width, unsigned int groups, float eps, unsigned int is_f32) {
    __shared__ float sum[256];
    unsigned int g = blockIdx.x, lane = threadIdx.x, base = g * width;
    float s = 0.0f;
    for (unsigned int c = lane; c < width; c += blockDim.x) { float v = is_f32 ? x32[base+c] : __half2float(x16[base+c]); s += v*v; }
    sum[lane] = s; __syncthreads();
    for (unsigned int d = 128; d; d >>= 1) { if (lane < d) sum[lane] += sum[lane+d]; __syncthreads(); }
    float inv = rsqrtf(sum[0] / width + eps);
    for (unsigned int c = lane; c < width; c += blockDim.x) out[base+c] = __float2half((is_f32 ? x32[base+c] : __half2float(x16[base+c])) * inv * w[(g % groups)*width+c]);
}
extern "C" __global__ void scaled_silu_f16(const __half *x, __half *out, unsigned int n, float scale) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = __half2float(x[i]) * scale; out[i] = __float2half(v / (1.0f + expf(-v))); }
}
extern "C" __global__ void sigmoid_group_mean_f16(const __half *x, const __half *gate, __half *out, unsigned int rows, unsigned int width, unsigned int groups) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    unsigned int base = (i / width) * width * groups + i % width;
    float sum = 0.0f;
    for (unsigned int g = 0; g < groups; ++g) { unsigned int j = base + g*width; sum += __half2float(x[j]) / (1.0f + expf(-__half2float(gate[j]))); }
    out[i] = __float2half(sum / groups);
}
extern "C" __global__ void sigmoid_group_residual_f32(const float *residual, const __half *update, const float *update32, const __half *gate, float *out, unsigned int rows, unsigned int width, unsigned int groups, unsigned int update_f32) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width * groups) return;
    unsigned int row = i / (width * groups), g = i / width % groups;
    float scale = 2.0f / (1.0f + expf(-__half2float(gate[row*groups+g]) / groups));
    unsigned int j = row*width+i%width;
    out[i] = residual[i] + scale * (update_f32 ? update32[j] : __half2float(update[j]));
}
extern "C" __global__ void signed_sqrt_group_gate_f16(const __half *key, const __half *query, const __half *value, __half *out, unsigned int width, unsigned int groups) {
    __shared__ float sum[256];
    unsigned int group = blockIdx.x, lane = threadIdx.x, base = group*width;
    float s = 0.0f;
    for (unsigned int c = lane; c < width; c += blockDim.x) s += __half2float(key[base+c]) * __half2float(query[base+c]);
    sum[lane] = s; __syncthreads();
    for (unsigned int d = 128; d; d >>= 1) { if (lane < d) sum[lane] += sum[lane+d]; __syncthreads(); }
    float dot = sum[0] * rsqrtf(float(width));
    float signed_root = dot == 0.0f ? 0.0f : copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    float gate = 1.0f / (1.0f + expf(-signed_root));
    for (unsigned int c = lane; c < width; c += blockDim.x) out[base+c] = __float2half(__half2float(value[(group/groups)*width+c]) * gate);
}
extern "C" __global__ void dilated_conv_residual_f32(const float *residual, const __half *gated, const __half *normalized, const __half *weight, float *state, float *out, unsigned int rows, unsigned int channels, unsigned int kernel, unsigned int dilation, float *checkpoints) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    unsigned int history = (kernel-1)*dilation;
    for (unsigned int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < kernel; ++k) {
            int from = int(row) - int((kernel-1-k)*dilation);
            float v = from < 0 ? state[(history+from)*channels+c] : __half2float(normalized[from*channels+c]);
            sum += v * __half2float(weight[c*kernel+k]);
        }
        unsigned int i = row*channels+c;
        out[i] = residual[i] + __half2float(gated[i]) + sum / (1.0f + expf(-sum));
        if (checkpoints) for (unsigned int t = 0; t < history; ++t) {
            const unsigned int from = t + row + 1u;
            checkpoints[((unsigned long long)row * history + t) * channels + c] = from < history ? state[from * channels + c] : __half2float(normalized[(from - history) * channels + c]);
        }
    }
    // 正序搬移确保较老的源状态在消费前不会被覆盖。
    for (unsigned int t = 0; t < history; ++t) {
        unsigned int from = t + rows;
        state[t*channels+c] = from < history ? state[from*channels+c] : __half2float(normalized[(from-history)*channels+c]);
    }
}
"#;

fn grid(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (n.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

fn f32_output(ctx: &CudaContext, rows: usize, cols: usize) -> Result<CudaTensor, String> {
    Ok(CudaTensor::new_f32_residual(ctx.buffer_uninit_f32(rows * cols)?, ctx.placeholder_f16()?, rows, cols))
}

pub fn repeat_groups(ctx: &CudaContext, input: &CudaTensor, groups: usize) -> Result<CudaTensor, String> {
    let out = f32_output(ctx, input.rows, input.cols * groups)?;
    let function = ctx.function("repeat_groups_f32")?;
    unsafe {
        ctx.stream()
            .launch_builder(&function)
            .arg(&input.slice)
            .arg(out.slice_f32.as_ref().unwrap())
            .arg(&(input.rows as u32))
            .arg(&(input.cols as u32))
            .arg(&(groups as u32))
            .launch(grid(input.len() * groups))
            .map_err(|e| format!("repeat_groups: {e:?}"))?;
    }
    Ok(out)
}

pub fn concat_broadcast_groups(ctx: &CudaContext, left: &CudaTensor, right: &CudaTensor, groups: usize) -> Result<CudaTensor, String> {
    if groups == 0 || left.cols == 0 || left.rows != right.rows || right.cols != groups * left.cols || left.slice_f32.is_some() || right.slice_f32.is_some() {
        return Err("concat broadcast groups shape/dtype 不匹配".into());
    }
    let output = ctx.tensor_uninit(left.rows * groups, left.cols * 2)?;
    let function = ctx.function("concat_broadcast_groups_f16")?;
    unsafe {
        ctx.stream()
            .launch_builder(&function)
            .arg(&left.slice)
            .arg(&right.slice)
            .arg(&output.slice)
            .arg(&(left.rows as u32))
            .arg(&(left.cols as u32))
            .arg(&(groups as u32))
            .launch(grid(output.len()))
            .map_err(|e| format!("concat broadcast groups: {e:?}"))?;
    }
    Ok(output)
}

pub fn norm(ctx: &CudaContext, input: &CudaTensor, weight: &CudaWeight, groups: usize, eps: f32) -> Result<CudaTensor, String> {
    if groups == 0 || !input.cols.is_multiple_of(groups) || weight.cols * weight.rows != input.cols {
        return Err(format!("grouped norm input={} groups={groups} weight={}x{}", input.cols, weight.rows, weight.cols));
    }
    let w = weight.data_f32.as_ref().ok_or("grouped norm 权重需要 F32")?;
    let out = ctx.tensor_alloc(input.rows, input.cols)?;
    let function = ctx.function("grouped_norm_f16")?;
    unsafe {
        ctx.stream()
            .launch_builder(&function)
            .arg(&input.slice)
            .arg(input.slice_f32.as_ref().unwrap_or(w))
            .arg(w)
            .arg(&out.slice)
            .arg(&((input.cols / groups) as u32))
            .arg(&(groups as u32))
            .arg(&eps)
            .arg(&u32::from(input.slice_f32.is_some()))
            .launch(LaunchConfig { grid_dim: ((input.rows * groups) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| format!("grouped norm: {e:?}"))?;
    }
    Ok(out)
}

pub fn scaled_silu(ctx: &CudaContext, input: &CudaTensor, scale: f32) -> Result<CudaTensor, String> {
    let out = ctx.tensor_alloc(input.rows, input.cols)?;
    let function = ctx.function("scaled_silu_f16")?;
    unsafe {
        ctx.stream().launch_builder(&function).arg(&input.slice).arg(&out.slice).arg(&(input.len() as u32)).arg(&scale).launch(grid(input.len())).map_err(|e| format!("scaled silu: {e:?}"))?;
    }
    Ok(out)
}

pub fn sigmoid_mean(ctx: &CudaContext, input: &CudaTensor, gate: &CudaTensor, groups: usize) -> Result<CudaTensor, String> {
    if groups == 0 || !input.cols.is_multiple_of(groups) || (input.rows, input.cols) != (gate.rows, gate.cols) {
        return Err("sigmoid group mean shape 不匹配".into());
    }
    let out = ctx.tensor_alloc(input.rows, input.cols / groups)?;
    let function = ctx.function("sigmoid_group_mean_f16")?;
    unsafe {
        ctx.stream()
            .launch_builder(&function)
            .arg(&input.slice)
            .arg(&gate.slice)
            .arg(&out.slice)
            .arg(&(out.rows as u32))
            .arg(&(out.cols as u32))
            .arg(&(groups as u32))
            .launch(grid(out.len()))
            .map_err(|e| format!("sigmoid group mean: {e:?}"))?;
    }
    Ok(out)
}

pub fn sigmoid_residual(ctx: &CudaContext, residual: &CudaTensor, update: &CudaTensor, gate: &CudaTensor) -> Result<CudaTensor, String> {
    let groups = gate.cols;
    if residual.rows != update.rows || gate.rows != update.rows || residual.cols != update.cols * groups {
        return Err("sigmoid group residual shape 不匹配".into());
    }
    let source = residual.slice_f32.as_ref().ok_or("group residual 需要 F32 残差")?;
    let out = f32_output(ctx, residual.rows, residual.cols)?;
    let function = ctx.function("sigmoid_group_residual_f32")?;
    unsafe {
        ctx.stream()
            .launch_builder(&function)
            .arg(source)
            .arg(&update.slice)
            .arg(update.slice_f32.as_ref().unwrap_or(source))
            .arg(&gate.slice)
            .arg(out.slice_f32.as_ref().unwrap())
            .arg(&(update.rows as u32))
            .arg(&(update.cols as u32))
            .arg(&(groups as u32))
            .arg(&u32::from(update.slice_f32.is_some()))
            .launch(grid(out.len()))
            .map_err(|e| format!("sigmoid group residual: {e:?}"))?;
    }
    Ok(out)
}

pub fn signed_sqrt_gate(ctx: &CudaContext, key: &CudaTensor, query: &CudaTensor, value: &CudaTensor, groups: usize) -> Result<CudaTensor, String> {
    if key.rows != value.rows || (key.rows, key.cols) != (query.rows, query.cols) || key.cols != value.cols * groups {
        return Err("signed sqrt gate shape 不匹配".into());
    }
    let out = ctx.tensor_alloc(key.rows, key.cols)?;
    let function = ctx.function("signed_sqrt_group_gate_f16")?;
    unsafe {
        ctx.stream()
            .launch_builder(&function)
            .arg(&key.slice)
            .arg(&query.slice)
            .arg(&value.slice)
            .arg(&out.slice)
            .arg(&(value.cols as u32))
            .arg(&(groups as u32))
            .launch(LaunchConfig { grid_dim: ((key.rows * groups) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| format!("signed sqrt gate: {e:?}"))?;
    }
    Ok(out)
}

pub fn dilated_conv_residual(
    ctx: &CudaContext,
    residual: &CudaTensor,
    gated: &CudaTensor,
    normalized: &CudaTensor,
    weight: &CudaWeight,
    state: &mut CudaSlice<f32>,
    dilation: usize,
    checkpoints: Option<&CudaSlice<f32>>,
) -> Result<CudaTensor, String> {
    let kernel = weight.cols;
    if kernel == 0
        || dilation == 0
        || weight.rows != residual.cols
        || state.len() != (kernel - 1) * dilation * residual.cols
        || (residual.rows, residual.cols) != (gated.rows, gated.cols)
        || (residual.rows, residual.cols) != (normalized.rows, normalized.cols)
    {
        return Err("dilated conv residual shape 不匹配".into());
    }
    use cudarc::driver::safe::DevicePtr;
    if checkpoints.is_some_and(|buffer| buffer.len() < residual.rows * state.len()) {
        return Err("PLE conv checkpoints 容量不足".into());
    }
    let checkpoint = checkpoints.map_or(0u64, |buffer| buffer.device_ptr(ctx.stream()).0);
    let source = residual.slice_f32.as_ref().ok_or("conv residual 需要 F32 残差")?;
    let out = f32_output(ctx, residual.rows, residual.cols)?;
    let function = ctx.function("dilated_conv_residual_f32")?;
    unsafe {
        ctx.stream()
            .launch_builder(&function)
            .arg(source)
            .arg(&gated.slice)
            .arg(&normalized.slice)
            .arg(&weight.data)
            .arg(state)
            .arg(out.slice_f32.as_ref().unwrap())
            .arg(&(residual.rows as u32))
            .arg(&(residual.cols as u32))
            .arg(&(kernel as u32))
            .arg(&(dilation as u32))
            .arg(&checkpoint)
            .launch(grid(residual.cols))
            .map_err(|e| format!("dilated conv residual: {e:?}"))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, BackendResources};

    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= 0.003 + e.abs() * 0.003, "index={index} actual={a} expected={e}");
        }
    }

    #[test]
    fn grouped_primitives_match_f32_reference() {
        let ctx = CudaContext::new_default().unwrap();
        let (rows, width, groups) = (5, 7, 3);
        let values: Vec<f32> = (0..rows * width).map(|i| (i as f32 - 17.0) / 16.0).collect();
        let input = ctx.tensor_from_f32(&values, rows, width).unwrap();
        let expanded = repeat_groups(&ctx, &input, groups).unwrap();
        let reference: Vec<f32> = values.chunks(width).flat_map(|row| (0..groups).flat_map(move |_| row.iter().copied())).collect();
        close(&ctx.tensor_to_f32(&expanded).unwrap(), &reference);
        let gamma: Vec<f32> = (0..width * groups).map(|i| 0.5 + i as f32 / 32.0).collect();
        let weight = ctx.prepare_f32(&gamma, 1, width * groups).unwrap();
        let normalized = norm(&ctx, &expanded, &weight, groups, 1e-6).unwrap();
        let expected: Vec<f32> = reference
            .chunks(width)
            .enumerate()
            .flat_map(|(g, row)| {
                let inv = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-6).sqrt().recip();
                row.iter().enumerate().map(|(c, v)| v * inv * gamma[g % groups * width + c]).collect::<Vec<_>>()
            })
            .collect();
        close(&ctx.tensor_to_f32(&normalized).unwrap(), &expected);
        let normalized_values = ctx.tensor_to_f32(&normalized).unwrap();
        let activated = scaled_silu(&ctx, &normalized, 1.0 / groups as f32).unwrap();
        let expected: Vec<f32> = normalized_values
            .iter()
            .map(|v| {
                let x = v / groups as f32;
                x / (1.0 + (-x).exp())
            })
            .collect();
        close(&ctx.tensor_to_f32(&activated).unwrap(), &expected);
        let mixed = sigmoid_mean(&ctx, &normalized, &normalized, groups).unwrap();
        let expected: Vec<f32> = (0..rows * width)
            .map(|i| {
                (0..groups)
                    .map(|g| {
                        let x = normalized_values[(i / width * groups + g) * width + i % width];
                        x / (1.0 + (-x).exp())
                    })
                    .sum::<f32>()
                    / groups as f32
            })
            .collect();
        close(&ctx.tensor_to_f32(&mixed).unwrap(), &expected);
        let gates = ctx.tensor_from_f32(&vec![0.0; rows * groups], rows, groups).unwrap();
        let combined = sigmoid_residual(&ctx, &expanded, &input, &gates).unwrap();
        close(&ctx.tensor_to_f32(&combined).unwrap(), &reference.iter().map(|v| v * 2.0).collect::<Vec<_>>());

        let gated = signed_sqrt_gate(&ctx, &normalized, &normalized, &input, groups).unwrap();
        let expected: Vec<f32> = (0..rows * groups)
            .flat_map(|g| {
                let dot = normalized_values[g * width..(g + 1) * width].iter().map(|v| v * v).sum::<f32>() / (width as f32).sqrt();
                let gate = 1.0 / (1.0 + (-dot.sqrt()).exp());
                values[g / groups * width..(g / groups + 1) * width].iter().map(move |v| v * gate).collect::<Vec<_>>()
            })
            .collect();
        close(&ctx.tensor_to_f32(&gated).unwrap(), &expected);
        let channels = width * groups;
        let (kernel, dilation) = (3, 2);
        let conv_values: Vec<f32> = (0..channels * kernel).map(|i| (i as f32 % 11.0 - 5.0) / 32.0).collect();
        let conv = ctx.prepare_f32(&conv_values, channels, kernel).unwrap();
        let history = (kernel - 1) * dilation;
        let mut state = ctx.stream().alloc_zeros::<f32>(history * channels).unwrap();
        let checkpoints = ctx.buffer_uninit::<f32>(rows * history * channels).unwrap();
        let out = dilated_conv_residual(&ctx, &expanded, &normalized, &normalized, &conv, &mut state, dilation, Some(&checkpoints)).unwrap();
        let recorded = ctx.stream().clone_dtoh(&checkpoints).unwrap();
        for row in 0..rows {
            for t in 0..history {
                for c in 0..channels {
                    let expected = (row + t + 1).checked_sub(history).map_or(0.0, |from| normalized_values[from * channels + c]);
                    assert_eq!(recorded[(row * history + t) * channels + c], expected, "PLE checkpoint row={row} history={t} channel={c}");
                }
            }
        }

        let expected: Vec<f32> = (0..rows * channels)
            .map(|i| {
                let (row, c) = (i / channels, i % channels);
                let sum = (0..kernel).filter_map(|k| row.checked_sub((kernel - 1 - k) * dilation).map(|from| normalized_values[from * channels + c] * conv_values[c * kernel + k])).sum::<f32>();
                reference[i] + normalized_values[i] + sum / (1.0 + (-sum).exp())
            })
            .collect();
        close(&ctx.tensor_to_f32(&out).unwrap(), &expected);
        close(&ctx.stream().clone_dtoh(&state).unwrap(), &normalized_values[(rows - history) * channels..]);
        // 分块边界须与整批一致,覆盖 rows < history 的状态移动。
        let mut state = ctx.stream().alloc_zeros::<f32>(history * channels).unwrap();
        let mut split = Vec::new();
        for (start, end) in [(0, 2), (2, 5)] {
            let base = ctx.tensor_from_f32(&values[start * width..end * width], end - start, width).unwrap();
            let residual = repeat_groups(&ctx, &base, groups).unwrap();
            let x = ctx.tensor_from_f32(&normalized_values[start * channels..end * channels], end - start, channels).unwrap();
            let out = dilated_conv_residual(&ctx, &residual, &x, &x, &conv, &mut state, dilation, None).unwrap();
            split.extend(ctx.tensor_to_f32(&out).unwrap());
        }
        close(&split, &expected);
    }

    #[test]
    fn group_concat_keeps_each_hidden_stream() {
        let ctx = CudaContext::new_default().unwrap();
        let (rows, width, groups) = (3usize, 37usize, 4usize);
        let left: Vec<f32> = (0..rows * width).map(|i| i as f32 / 16.0).collect();
        let right: Vec<f32> = (0..rows * groups * width).map(|i| -(i as f32) / 8.0).collect();
        let l = ctx.tensor_from_f32(&left, rows, width).unwrap();
        let r = ctx.tensor_from_f32(&right, rows, groups * width).unwrap();
        let actual = concat_broadcast_groups(&ctx, &l, &r, groups).and_then(|t| ctx.tensor_to_f32(&t)).unwrap();
        let mut expected = Vec::new();
        for row in 0..rows {
            for group in 0..groups {
                expected.extend_from_slice(&left[row * width..(row + 1) * width]);
                expected.extend_from_slice(&right[(row * groups + group) * width..(row * groups + group + 1) * width]);
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn sigmoid_and_silu_delta_gates_match_cpu() {
        use crate::attention::gated_delta_net::{GatedDeltaNetInputs, GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetWeightsRef, GdnOutputGate};
        use crate::{backend::cpu::CpuContext, kernel::cpu::CpuTensor};
        let ctx = CudaContext::new_default().unwrap();
        let cpu = CpuContext;
        for (output_gate, key_head_dim, value_head_dim) in [(GdnOutputGate::Silu, 4, 4), (GdnOutputGate::Sigmoid, 4, 4), (GdnOutputGate::Silu, 128, 96), (GdnOutputGate::Sigmoid, 128, 96)] {
            let spec = GatedDeltaNetSpec { key_heads: 2, value_heads: 4, key_head_dim, value_head_dim, conv_kernel: 3, rms_eps: 1e-6, output_gate };
            let (rows, columns) = (5, spec.conv_dim());
            let qkv: Vec<f32> = (0..rows * columns).map(|i| (i as f32 % 23.0 - 11.0) / 16.0).collect();
            let z: Vec<f32> = (0..rows * spec.value_dim()).map(|i| (i as f32 % 7.0 - 3.0) / 2.0).collect();
            let ab = vec![0.25; rows * spec.value_heads];
            let conv = vec![0.5; spec.conv_state_elements()];
            let alog = vec![-1.0; spec.value_heads];
            let dt = vec![0.1; spec.value_heads];
            let norm_weight = vec![1.0; spec.value_head_dim];
            let q = ctx.tensor_from_f32(&qkv, rows, columns).unwrap();
            let g = ctx.tensor_from_f32(&z, rows, spec.value_dim()).unwrap();
            let a = ctx.tensor_from_f32(&ab, rows, spec.value_heads).unwrap();
            let cw = ctx.prepare_f32(&conv, columns, spec.conv_kernel).unwrap();
            let aw = ctx.prepare_f32(&alog, 1, spec.value_heads).unwrap();
            let dw = ctx.prepare_f32(&dt, 1, spec.value_heads).unwrap();
            let nw = ctx.prepare_f32(&norm_weight, 1, spec.value_head_dim).unwrap();
            let mut state = ctx.allocate_gated_delta_net_storage(&spec).unwrap();
            state.enable_checkpoints(&ctx, rows).unwrap();
            let actual = ctx.gated_delta_net_fused(&mut state, GatedDeltaNetInputs { qkv: &q, z: &g, alpha: &a, beta: &a }, GatedDeltaNetWeightsRef { conv: &cw, a_log: &aw, dt_bias: &dw, norm: &nw }, &spec).unwrap();
            state.restore_checkpoint(&ctx, rows - 2).unwrap();
            let qr = ctx.select_row(&q, rows - 1).unwrap();
            let gr = ctx.select_row(&g, rows - 1).unwrap();
            let ar = ctx.select_row(&a, rows - 1).unwrap();
            let replay = ctx.gated_delta_net_fused(&mut state, GatedDeltaNetInputs { qkv: &qr, z: &gr, alpha: &ar, beta: &ar }, GatedDeltaNetWeightsRef { conv: &cw, a_log: &aw, dt_bias: &dw, norm: &nw }, &spec).unwrap();
            let expected_last = ctx.tensor_to_f32(&actual).unwrap();
            close(&ctx.tensor_to_f32(&replay).unwrap(), &expected_last[(rows - 1) * spec.value_dim()..]);
            let q = CpuTensor { data: qkv, rows, cols: columns };
            let g = CpuTensor { data: z, rows, cols: spec.value_dim() };
            let a = CpuTensor { data: ab, rows, cols: spec.value_heads };
            let cw = cpu.prepare_f32(&conv, columns, spec.conv_kernel).unwrap();
            let aw = cpu.prepare_f32(&alog, 1, spec.value_heads).unwrap();
            let dw = cpu.prepare_f32(&dt, 1, spec.value_heads).unwrap();
            let nw = cpu.prepare_f32(&norm_weight, 1, spec.value_head_dim).unwrap();
            let mut state = cpu.allocate_gated_delta_net_storage(&spec).unwrap();
            let expected = cpu.gated_delta_net_fused(&mut state, GatedDeltaNetInputs { qkv: &q, z: &g, alpha: &a, beta: &a }, GatedDeltaNetWeightsRef { conv: &cw, a_log: &aw, dt_bias: &dw, norm: &nw }, &spec).unwrap();
            close(&ctx.tensor_to_f32(&actual).unwrap(), &expected.data);
        }
    }
}
