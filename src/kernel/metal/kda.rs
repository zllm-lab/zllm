//! KDA Metal recurrent kernel。

/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: kda_recurrent_f16
// private helpers: kda_sigmoid, kda_softplus
pub const SHADERS: &str = r#"
inline float kda_sigmoid(float value)
{
    if (value >= 0.0f) return 1.0f / (1.0f + exp(-value));
    const float exponential = exp(value);
    return exponential / (1.0f + exponential);
}
inline float kda_softplus(float value)
{
    if (value > 20.0f) return value;
    if (value < -20.0f) return exp(value);
    return log(1.0f + exp(value));
}
kernel void kda_recurrent_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device const half *decay [[buffer(3)]],
    device const half *beta [[buffer(4)]],
    device const half *output_gate [[buffer(5)]],
    device const half *query_conv_weight [[buffer(6)]],
    device const half *key_conv_weight [[buffer(7)]],
    device const half *value_conv_weight [[buffer(8)]],
    device const float *a_log [[buffer(9)]],
    device const float *dt_bias [[buffer(10)]],
    device const half *output_norm_weight [[buffer(11)]],
    device float *conv_state [[buffer(12)]],
    device float *recurrent_state [[buffer(13)]],
    device half *output [[buffer(14)]],
    constant uint &input_rows [[buffer(15)]],
    constant uint &num_heads [[buffer(16)]],
    constant uint &head_dim [[buffer(17)]],
    constant uint &conv_kernel_size [[buffer(18)]],
    constant uint &gate_lower_bound_enabled [[buffer(19)]],
    constant float &gate_lower_bound [[buffer(20)]],
    constant uint &use_qk_l2norm [[buffer(21)]],
    constant float &output_norm_eps [[buffer(22)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint thread_count [[threads_per_threadgroup]])
{
    if (head >= num_heads) return;
    threadgroup float query_values[256];
    threadgroup float key_values[256];
    threadgroup float core_values[256];
    threadgroup float2 reductions[256];
    threadgroup float query_inverse;
    threadgroup float key_inverse;
    threadgroup float core_inverse;
    const uint projection_size = num_heads * head_dim;
    const uint channel = head * head_dim + lane;
    const uint history = conv_kernel_size - 1;
    const ulong conv_path_stride = ulong(projection_size) * history;
    const ulong recurrent_head_base = ulong(head) * head_dim * head_dim;
    const float query_scale = rsqrt(float(head_dim));
    const float a = exp(a_log[head]);

    for (uint token = 0; token < input_rows; ++token) {
        const ulong token_base = ulong(token) * projection_size;
        float query_item = 0.0f;
        float key_item = 0.0f;
        float value_item = 0.0f;
        if (lane < head_dim) {
            const ulong weight_base = ulong(channel) * conv_kernel_size;
            const ulong state_base = ulong(channel) * history;
            query_item = float(query[token_base + channel]) * float(query_conv_weight[weight_base + history]);
            key_item = float(key[token_base + channel]) * float(key_conv_weight[weight_base + history]);
            value_item = float(value[token_base + channel]) * float(value_conv_weight[weight_base + history]);
            for (uint index = 0; index < history; ++index) {
                query_item += conv_state[state_base + index] * float(query_conv_weight[weight_base + index]);
                key_item += conv_state[conv_path_stride + state_base + index] * float(key_conv_weight[weight_base + index]);
                value_item += conv_state[2 * conv_path_stride + state_base + index] * float(value_conv_weight[weight_base + index]);
            }
            for (uint index = 1; index < history; ++index) {
                conv_state[state_base + index - 1] = conv_state[state_base + index];
                conv_state[conv_path_stride + state_base + index - 1] = conv_state[conv_path_stride + state_base + index];
                conv_state[2 * conv_path_stride + state_base + index - 1] = conv_state[2 * conv_path_stride + state_base + index];
            }
            if (history != 0) {
                conv_state[state_base + history - 1] = float(query[token_base + channel]);
                conv_state[conv_path_stride + state_base + history - 1] = float(key[token_base + channel]);
                conv_state[2 * conv_path_stride + state_base + history - 1] = float(value[token_base + channel]);
            }
            query_item *= kda_sigmoid(query_item);
            key_item *= kda_sigmoid(key_item);
            value_item *= kda_sigmoid(value_item);
        }
        query_values[lane] = query_item;
        key_values[lane] = key_item;
        reductions[lane] = lane < head_dim ? float2(query_item * query_item, key_item * key_item) : float2(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint active = thread_count; active > 1;) {
            const uint half_active = (active + 1) >> 1;
            if (lane < half_active && lane + half_active < active) {
                reductions[lane] += reductions[lane + half_active];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            active = half_active;
        }
        if (lane == 0) {
            query_inverse = use_qk_l2norm != 0 ? rsqrt(reductions[0].x + 1.0e-6f) : 1.0f;
            key_inverse = use_qk_l2norm != 0 ? rsqrt(reductions[0].y + 1.0e-6f) : 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mixed = 0.0f;
        if (lane < head_dim) {
            float predicted = 0.0f;
            for (uint key_column = 0; key_column < head_dim; ++key_column) {
                const uint decay_column = head * head_dim + key_column;
                const float raw_gate = float(decay[token_base + decay_column]) + dt_bias[decay_column];
                const float log_decay = gate_lower_bound_enabled != 0
                    ? gate_lower_bound * kda_sigmoid(a * raw_gate)
                    : -a * kda_softplus(raw_gate);
                const ulong state_index = recurrent_head_base + ulong(key_column) * head_dim + lane;
                recurrent_state[state_index] *= exp(log_decay);
                predicted += recurrent_state[state_index] * key_values[key_column] * key_inverse;
            }
            const float beta_value = kda_sigmoid(float(beta[ulong(token) * num_heads + head]));
            const float delta = (value_item - predicted) * beta_value;
            for (uint key_column = 0; key_column < head_dim; ++key_column) {
                const ulong state_index = recurrent_head_base + ulong(key_column) * head_dim + lane;
                recurrent_state[state_index] += key_values[key_column] * key_inverse * delta;
                mixed += query_values[key_column] * query_inverse * query_scale * recurrent_state[state_index];
            }
        }
        core_values[lane] = mixed;
        reductions[lane] = lane < head_dim ? float2(mixed * mixed, 0.0f) : float2(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = thread_count >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) reductions[lane] += reductions[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) core_inverse = rsqrt(reductions[0].x / float(head_dim) + output_norm_eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < head_dim) {
            output[token_base + channel] = finite_f16(
                core_values[lane]
                * core_inverse
                * float(output_norm_weight[lane])
                * kda_sigmoid(float(output_gate[token_base + channel])));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

use crate::backend::metal::api as metal;

use crate::{
    attention::kda::KdaSpec,
    backend::metal::{MetalContext, MetalTensor, MetalTensorDType},
};

use super::{set_bytes, validate_u32};

#[allow(clippy::too_many_arguments)]
pub fn recurrent_tensor(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &MetalTensor,
    value: &MetalTensor,
    decay: &MetalTensor,
    beta: &MetalTensor,
    output_gate: &MetalTensor,
    query_conv_weight: &metal::Buffer,
    key_conv_weight: &metal::Buffer,
    value_conv_weight: &metal::Buffer,
    a_log: &metal::Buffer,
    a_log_len: usize,
    dt_bias: &metal::Buffer,
    dt_bias_len: usize,
    output_norm_weight: &metal::Buffer,
    conv_state: &metal::Buffer,
    recurrent_state: &metal::Buffer,
    spec: &KdaSpec,
) -> Result<MetalTensor, String> {
    spec.validate()?;
    let rows = query.rows;
    let projection_size = spec.projection_size();
    let projected = [key, value, decay, output_gate].iter().all(|tensor| tensor.rows == rows && tensor.cols == projection_size && tensor.dtype == MetalTensorDType::F16);
    if rows == 0 || query.cols != projection_size || query.dtype != MetalTensorDType::F16 || !projected || beta.rows != rows || beta.cols != spec.num_heads || beta.dtype != MetalTensorDType::F16 {
        return Err("Metal KDA input shape/dtype 与 spec 不一致".to_owned());
    }
    if spec.head_dim > 256 {
        return Err(format!("Metal KDA head_dim={} 超过单 threadgroup 上限 256", spec.head_dim));
    }
    let conv_weight_bytes = projection_size.checked_mul(spec.short_conv_kernel_size).and_then(|elements| elements.checked_mul(std::mem::size_of::<half::f16>())).ok_or_else(|| "Metal KDA conv weight 大小溢出".to_owned())? as u64;
    let output_norm_bytes = spec.head_dim.checked_mul(std::mem::size_of::<half::f16>()).ok_or_else(|| "Metal KDA output norm 大小溢出".to_owned())? as u64;
    let conv_state_bytes = spec.conv_state_elements().checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| "Metal KDA conv state 大小溢出".to_owned())? as u64;
    let recurrent_state_bytes = spec.recurrent_state_elements().checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| "Metal KDA recurrent state 大小溢出".to_owned())? as u64;
    if query_conv_weight.length() != conv_weight_bytes
        || key_conv_weight.length() != conv_weight_bytes
        || value_conv_weight.length() != conv_weight_bytes
        || a_log_len != spec.num_heads
        || dt_bias_len != projection_size
        || output_norm_weight.length() != output_norm_bytes
        || conv_state.length() != conv_state_bytes
        || recurrent_state.length() != recurrent_state_bytes
    {
        return Err("Metal KDA weight/state shape 与 spec 不一致".to_owned());
    }
    // 长度参数只核对元素数，kernel 按 float 读取，buffer 实际字节数必须够。
    let a_log_bytes = a_log_len as u64 * std::mem::size_of::<f32>() as u64;
    let dt_bias_bytes = dt_bias_len as u64 * std::mem::size_of::<f32>() as u64;
    if a_log.length() < a_log_bytes || dt_bias.length() < dt_bias_bytes {
        return Err(format!("Metal KDA a_log/dt_bias buffer 过小: a_log={}B(需 {a_log_bytes}B, {a_log_len} 个 f32) dt_bias={}B(需 {dt_bias_bytes}B, {dt_bias_len} 个 f32)", a_log.length(), dt_bias.length()));
    }

    let output = ctx.tensor_zeros(rows, projection_size);
    let input_rows = validate_u32("Metal KDA rows", rows)?;
    let num_heads = validate_u32("Metal KDA heads", spec.num_heads)?;
    let head_dim = validate_u32("Metal KDA head dim", spec.head_dim)?;
    let conv_kernel_size = validate_u32("Metal KDA conv kernel", spec.short_conv_kernel_size)?;
    let gate_lower_bound_enabled = u32::from(spec.gate_lower_bound.is_some());
    let gate_lower_bound = spec.gate_lower_bound.unwrap_or(0.0);
    let use_qk_l2norm = u32::from(spec.use_qk_l2norm);
    let threads = spec.head_dim.next_power_of_two().max(32);
    let pipeline = ctx.pipeline("kda_recurrent_f16")?;
    if pipeline.max_total_threads_per_threadgroup() < threads as u64 {
        return Err(format!("Metal KDA 需要 {threads} threads/threadgroup，设备仅支持 {}", pipeline.max_total_threads_per_threadgroup(),));
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&query.buffer), 0);
    encoder.set_buffer(1, Some(&key.buffer), 0);
    encoder.set_buffer(2, Some(&value.buffer), 0);
    encoder.set_buffer(3, Some(&decay.buffer), 0);
    encoder.set_buffer(4, Some(&beta.buffer), 0);
    encoder.set_buffer(5, Some(&output_gate.buffer), 0);
    encoder.set_buffer(6, Some(query_conv_weight), 0);
    encoder.set_buffer(7, Some(key_conv_weight), 0);
    encoder.set_buffer(8, Some(value_conv_weight), 0);
    encoder.set_buffer(9, Some(a_log), 0);
    encoder.set_buffer(10, Some(dt_bias), 0);
    encoder.set_buffer(11, Some(output_norm_weight), 0);
    encoder.set_buffer(12, Some(conv_state), 0);
    encoder.set_buffer(13, Some(recurrent_state), 0);
    encoder.set_buffer(14, Some(&output.buffer), 0);
    set_bytes(&encoder, 15, &input_rows);
    set_bytes(&encoder, 16, &num_heads);
    set_bytes(&encoder, 17, &head_dim);
    set_bytes(&encoder, 18, &conv_kernel_size);
    set_bytes(&encoder, 19, &gate_lower_bound_enabled);
    set_bytes(&encoder, 20, &gate_lower_bound);
    set_bytes(&encoder, 21, &use_qk_l2norm);
    set_bytes(&encoder, 22, &spec.output_norm_eps);
    encoder.dispatch_thread_groups(metal::MTLSize::new(spec.num_heads as u64, 1, 1), metal::MTLSize::new(threads as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={rows},heads={},head_dim={},conv={}", spec.num_heads, spec.head_dim, spec.short_conv_kernel_size,);
    let input_bytes = query.buffer.length()
        + key.buffer.length()
        + value.buffer.length()
        + decay.buffer.length()
        + beta.buffer.length()
        + output_gate.buffer.length()
        + query_conv_weight.length()
        + key_conv_weight.length()
        + value_conv_weight.length()
        + a_log.length()
        + dt_bias.length()
        + output_norm_weight.length()
        + conv_state.length()
        + recurrent_state.length();
    ctx.commit_and_wait_profiled(&command, "kda_recurrent_f16", &shape, input_bytes, output.buffer.length() + conv_state.length() + recurrent_state.length());
    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::{
        attention::kda::KdaSpec,
        backend::metal::MetalContext,
        kernel::{
            cpu::{CpuTensor, kda as cpu_kda},
            metal::kernels_source,
        },
    };

    use super::recurrent_tensor;

    #[test]
    fn 多token输出与cpu_oracle一致() {
        let spec = KdaSpec { num_heads: 1, head_dim: 2, short_conv_kernel_size: 2, use_full_rank_gate: true, gate_lower_bound: Some(-5.0), use_qk_l2norm: true, output_norm_eps: 1.0e-5 };
        let data = [1.0, 0.5, 0.2, 1.0];
        let decay_data = [0.1, -0.2, 0.3, 0.4];
        let beta_data = [0.0, 0.5];
        let gate_data = [0.2, -0.1, 0.4, 0.3];
        let conv_weight = [0.0, 1.0, 0.0, 1.0];
        let a_log = [0.0];
        let dt_bias = [0.0, 0.0];
        let norm_weight = [1.0, 1.0];
        let cpu_tensor = |data: &[f32], cols| CpuTensor { data: data.to_vec(), rows: 2, cols };
        let mut cpu_conv = [vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]];
        let mut cpu_recurrent = vec![0.0; 4];
        let [query_conv, key_conv, value_conv] = &mut cpu_conv;
        let expected = cpu_kda::recurrent(
            &cpu_tensor(&data, 2),
            &cpu_tensor(&data, 2),
            &cpu_tensor(&data, 2),
            &cpu_tensor(&decay_data, 2),
            &cpu_tensor(&beta_data, 1),
            &cpu_tensor(&gate_data, 2),
            query_conv,
            key_conv,
            value_conv,
            &mut cpu_recurrent,
            &conv_weight,
            &conv_weight,
            &conv_weight,
            &a_log,
            &dt_bias,
            &norm_weight,
            &spec,
        )
        .unwrap();

        let ctx = MetalContext::new(kernels_source()).unwrap();
        let input = ctx.tensor_from_f32(&data, 2, 2).unwrap();
        let decay = ctx.tensor_from_f32(&decay_data, 2, 2).unwrap();
        let beta = ctx.tensor_from_f32(&beta_data, 2, 1).unwrap();
        let gate = ctx.tensor_from_f32(&gate_data, 2, 2).unwrap();
        let conv_weight = ctx.tensor_from_f32(&conv_weight, 2, 2).unwrap();
        let norm_weight = ctx.tensor_from_f32(&norm_weight, 1, 2).unwrap();
        let a_log_buffer = f32_buffer(&ctx, &a_log);
        let dt_bias_buffer = f32_buffer(&ctx, &dt_bias);
        let conv_state = ctx.shared_buffer_zeros(spec.conv_state_elements() * std::mem::size_of::<f32>());
        let recurrent_state = ctx.shared_buffer_zeros(spec.recurrent_state_elements() * std::mem::size_of::<f32>());
        let actual = recurrent_tensor(
            &ctx,
            &input,
            &input,
            &input,
            &decay,
            &beta,
            &gate,
            &conv_weight.buffer,
            &conv_weight.buffer,
            &conv_weight.buffer,
            &a_log_buffer,
            a_log.len(),
            &dt_bias_buffer,
            dt_bias.len(),
            &norm_weight.buffer,
            &conv_state,
            &recurrent_state,
            &spec,
        )
        .unwrap();
        for (actual, expected) in ctx.tensor_to_f32(&actual).into_iter().zip(expected.data) {
            assert!((actual - expected).abs() <= 0.02, "actual={actual}, expected={expected}");
        }
    }

    #[test]
    fn kimi_head_dim128与cpu_oracle一致() {
        let spec = KdaSpec { num_heads: 2, head_dim: 128, short_conv_kernel_size: 4, use_full_rank_gate: true, gate_lower_bound: Some(-5.0), use_qk_l2norm: true, output_norm_eps: 1.0e-5 };
        let rows = 2;
        let projection = spec.projection_size();
        let values = |frequency: f32, scale: f32, len: usize| (0..len).map(|index| ((index as f32 + 1.0) * frequency).sin() * scale).collect::<Vec<_>>();
        let query = values(0.013, 0.2, rows * projection);
        let key = values(0.017, 0.15, rows * projection);
        let value = values(0.019, 0.1, rows * projection);
        let decay = values(0.007, 0.3, rows * projection);
        let beta = values(0.11, 0.4, rows * spec.num_heads);
        let gate = values(0.023, 0.2, rows * projection);
        let conv_weight = (0..projection * spec.short_conv_kernel_size).map(|index| if index % spec.short_conv_kernel_size == spec.short_conv_kernel_size - 1 { 1.0 } else { 0.01 }).collect::<Vec<_>>();
        let a_log = vec![0.0; spec.num_heads];
        let dt_bias = values(0.005, 0.1, projection);
        let norm_weight = vec![1.0; spec.head_dim];
        let cpu_tensor = |data: &[f32], cols| CpuTensor { data: data.to_vec(), rows, cols };
        let conv_elements = projection * (spec.short_conv_kernel_size - 1);
        let mut cpu_conv = [vec![0.0; conv_elements], vec![0.0; conv_elements], vec![0.0; conv_elements]];
        let mut cpu_recurrent = vec![0.0; spec.recurrent_state_elements()];
        let [query_conv, key_conv, value_conv] = &mut cpu_conv;
        let expected = cpu_kda::recurrent(
            &cpu_tensor(&query, projection),
            &cpu_tensor(&key, projection),
            &cpu_tensor(&value, projection),
            &cpu_tensor(&decay, projection),
            &cpu_tensor(&beta, spec.num_heads),
            &cpu_tensor(&gate, projection),
            query_conv,
            key_conv,
            value_conv,
            &mut cpu_recurrent,
            &conv_weight,
            &conv_weight,
            &conv_weight,
            &a_log,
            &dt_bias,
            &norm_weight,
            &spec,
        )
        .unwrap();

        let ctx = MetalContext::new(kernels_source()).unwrap();
        let query_tensor = ctx.tensor_from_f32(&query, rows, projection).unwrap();
        let key_tensor = ctx.tensor_from_f32(&key, rows, projection).unwrap();
        let value_tensor = ctx.tensor_from_f32(&value, rows, projection).unwrap();
        let decay_tensor = ctx.tensor_from_f32(&decay, rows, projection).unwrap();
        let beta_tensor = ctx.tensor_from_f32(&beta, rows, spec.num_heads).unwrap();
        let gate_tensor = ctx.tensor_from_f32(&gate, rows, projection).unwrap();
        let conv_weight_tensor = ctx.tensor_from_f32(&conv_weight, projection, spec.short_conv_kernel_size).unwrap();
        let norm_weight_tensor = ctx.tensor_from_f32(&norm_weight, 1, spec.head_dim).unwrap();
        let a_log_buffer = f32_buffer(&ctx, &a_log);
        let dt_bias_buffer = f32_buffer(&ctx, &dt_bias);
        let conv_state = ctx.shared_buffer_zeros(spec.conv_state_elements() * std::mem::size_of::<f32>());
        let recurrent_state = ctx.shared_buffer_zeros(spec.recurrent_state_elements() * std::mem::size_of::<f32>());
        let actual = recurrent_tensor(
            &ctx,
            &query_tensor,
            &key_tensor,
            &value_tensor,
            &decay_tensor,
            &beta_tensor,
            &gate_tensor,
            &conv_weight_tensor.buffer,
            &conv_weight_tensor.buffer,
            &conv_weight_tensor.buffer,
            &a_log_buffer,
            a_log.len(),
            &dt_bias_buffer,
            dt_bias.len(),
            &norm_weight_tensor.buffer,
            &conv_state,
            &recurrent_state,
            &spec,
        )
        .unwrap();
        for (actual, expected) in ctx.tensor_to_f32(&actual).into_iter().zip(expected.data) {
            let tolerance = 0.03 + 0.02 * expected.abs();
            assert!((actual - expected).abs() <= tolerance, "actual={actual}, expected={expected}, tolerance={tolerance}");
        }
    }

    fn f32_buffer(ctx: &MetalContext, values: &[f32]) -> crate::backend::metal::api::Buffer {
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
        ctx.shared_buffer(bytes)
    }
}
