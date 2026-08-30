//! 单 token RMSNorm、Q/K/V 投影、逐头 RMSNorm 与 RoPE 融合算子。
//!
//! 把 `rmsnorm -> Q/K/V gemv -> 逐头 GemmaRMSNorm -> RoPE` 的 ~11 次 kernel
//! launch 合并为一次。每个 threadgroup 负责一个 head 槽位(Q head 或 KV head),
//! hidden 的 RMS 在组内冗余计算(2560 个元素,远小于随后的权重流读取)。

pub const SHADERS: &str = r#"
kernel void fused_rmsnorm_qkv_head_norm_rope(
    device const half *hidden [[buffer(0)]],
    device const half *input_norm [[buffer(1)]],
    device const uint *q_packed [[buffer(2)]],
    device const uchar *q_scales [[buffer(3)]],
    device const uchar *q_biases [[buffer(4)]],
    device const uint *k_packed [[buffer(5)]],
    device const uchar *k_scales [[buffer(6)]],
    device const uchar *k_biases [[buffer(7)]],
    device const uint *v_packed [[buffer(8)]],
    device const uchar *v_scales [[buffer(9)]],
    device const uchar *v_biases [[buffer(10)]],
    device const float *q_norm [[buffer(11)]],
    device const float *k_norm [[buffer(12)]],
    device const half *cos_data [[buffer(13)]],
    device const half *sin_data [[buffer(14)]],
    device half *q_out [[buffer(15)]],
    device half *k_out [[buffer(16)]],
    device half *v_out [[buffer(17)]],
    constant uint &hidden_size [[buffer(18)]],
    constant uint &head_dim [[buffer(19)]],
    constant uint &rotary_dim [[buffer(20)]],
    constant float &eps [[buffer(21)]],
    constant uint &scale_dtype [[buffer(22)]],
    constant uint &group_size [[buffer(23)]],
    constant uint &head_count [[buffer(24)]],
    constant uint &kv_head_count [[buffer(25)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    // 尺寸上限由 host 校验;tgm 数组必须编译期定长。
    threadgroup half normed[4096];
    threadgroup float rows[512];
    threadgroup float post[512];
    threadgroup float hidden_sums[256];
    threadgroup float head_sums[256];

    // 1) hidden RMS + input norm(组内冗余计算,保留 float 精度)。
    // normed 以 half 存储:写入值全部经过 F16 边界,无损且 threadgroup 内存减半,
    // 避免大 TG 内存限制同时驻留的 threadgroup 数
    float square_sum = 0.0f;
    for (uint column = lane; column < hidden_size; column += 256) {
        const float value = float(hidden[column]);
        normed[column] = half(value);
        square_sum += value * value;
    }
    hidden_sums[lane] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) hidden_sums[lane] += hidden_sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(hidden_sums[0] / float(hidden_size) + eps);
    for (uint column = lane; column < hidden_size; column += 256) {
        // 与生产顺序路径一致:权重按 F16 取整后参与 norm,norm 结果过 F16 边界再进 gemv。
        // F32 直读会与顺序路径产生同方向的系统性 ULP 偏差,多层相干累积曾劣化生成质量。
        normed[column] = half(normed[column] * inv * input_norm[column]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) 角色分发:前 head_count 组是 Q head,随后 kv_head_count 组是 K,再后是 V。
    const bool is_query = group_id < head_count;
    const bool is_key = !is_query && group_id < head_count + kv_head_count;
    const uint head = is_query ? group_id : (is_key ? group_id - head_count : group_id - head_count - kv_head_count);
    device const uint *packed = is_query ? q_packed : (is_key ? k_packed : v_packed);
    device const uchar *scales = is_query ? q_scales : (is_key ? k_scales : v_scales);
    device const uchar *biases = is_query ? q_biases : (is_key ? k_biases : v_biases);
    const uint row = head * head_dim;

    // 3) 4-bit affine gemv:8 个 SIMD 组各认领一行子序列,列分块 + nibble 原位乘,
    // 与 mlx_affine_gemv_f16_u4 同构;normed 驻留 threadgroup,跨行重复读零成本。
    const uint simd_group = lane >> 5;
    const uint simd_lane = lane & 31;
    const uint values_per_thread = hidden_size % 16 == 0 ? 16 : 8;
    const uint block_size = values_per_thread * 32;
    const ulong packed_columns = (ulong(hidden_size) + 7) / 8;
    const ulong weight_bytes = packed_columns * 4;
    const ulong parameter_groups = hidden_size / group_size;
    for (uint offset = simd_group; offset < head_dim; offset += 8) {
        const uint gemv_row = row + offset;
        const device const ushort *weight_row = (const device ushort *)((device const uchar *)packed + ulong(gemv_row) * weight_bytes);
        float result = 0.0f;
        for (uint block = 0; block < hidden_size; block += block_size) {
            const uint begin = block + simd_lane * values_per_thread;
            const uint count = begin < hidden_size ? min(values_per_thread, hidden_size - begin) : 0;
            float input_values[16];
            float input_sum = 0.0f;
#pragma unroll
            for (uint index = 0; index < values_per_thread; index += 4) {
                if (index + 3 < count) {
                    const half4 values = *((threadgroup const half4 *)(normed + begin + index));
                    input_values[index] = float(values.x);
                    input_values[index + 1] = float(values.y) * 0.0625f;
                    input_values[index + 2] = float(values.z) * 0.00390625f;
                    input_values[index + 3] = float(values.w) * 0.000244140625f;
                    input_sum += float(values.x) + float(values.y) + float(values.z) + float(values.w);
                } else {
                    for (uint tail = index; tail < values_per_thread; ++tail) {
                        const float value = tail < count ? float(normed[begin + tail]) : 0.0f;
                        input_values[tail] = value / float(1u << ((tail & 3) * 4));
                        input_sum += value;
                    }
                }
            }
            if (count > 0) {
                float code_sum = 0.0f;
                if (values_per_thread == 16 && count == 16) {
                    const ushort4 packed_values = *((device const ushort4 *)(weight_row + (begin >> 2)));
#pragma unroll
                    for (uint index = 0; index < 4; ++index) {
                        const ushort packed_word = packed_values[index];
                        code_sum +=
                            input_values[index * 4] * float(packed_word & 0x000f) +
                            input_values[index * 4 + 1] * float(packed_word & 0x00f0) +
                            input_values[index * 4 + 2] * float(packed_word & 0x0f00) +
                            input_values[index * 4 + 3] * float(packed_word & 0xf000);
                    }
                } else {
                    for (uint index = 0; index < values_per_thread / 4; ++index) {
                        if (index * 4 >= count) break;
                        const ushort packed_word = weight_row[(begin >> 2) + index];
                        code_sum +=
                            input_values[index * 4] * float(packed_word & 0x000f) +
                            input_values[index * 4 + 1] * float(packed_word & 0x00f0) +
                            input_values[index * 4 + 2] * float(packed_word & 0x0f00) +
                            input_values[index * 4 + 3] * float(packed_word & 0xf000);
                    }
                }
                const ulong parameter = ulong(gemv_row) * parameter_groups + begin / group_size;
                result += w4a16_scale(scales, parameter, scale_dtype) * code_sum + input_sum * w4a16_scale(biases, parameter, scale_dtype);
            }
        }
        result = simd_sum(result);
        if (simd_lane == 0) {
            rows[offset] = float(half(result));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) 逐头 GemmaRMSNorm:weight_offset = 1(V 用全零 norm,即纯 RMS)。
    float head_square = 0.0f;
    for (uint offset = lane; offset < head_dim; offset += 256) {
        head_square += rows[offset] * rows[offset];
    }
    head_sums[lane] = head_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) head_sums[lane] += head_sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float head_inv = rsqrt(head_sums[0] / float(head_dim) + eps);
    for (uint offset = lane; offset < head_dim; offset += 256) {
        const float weight = is_query ? (q_norm[offset] + 1.0f) : (is_key ? (k_norm[offset] + 1.0f) : 1.0f);
        post[offset] = float(half(rows[offset] * head_inv * weight));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5) Split-half RoPE(与 apply_rope_prefix_f16 相同的角标与配对语义)。
    device half *output = is_query ? q_out : (is_key ? k_out : v_out);
    const uint rotary_half = rotary_dim >> 1;
    // 注意:这里不允许任何 early return。提前退出的线程与继续读取 tgm 的线程
    // 并存时,实测(Apple GPU)会出现跨调用非确定的结果;所有分支必须走到末尾。
    for (uint offset = lane; offset < head_dim; offset += 256) {
        const uint index = head * head_dim + offset;
        if (!is_query && !is_key || rotary_dim == 0 || offset >= rotary_dim) {
            output[index] = half(post[offset]);
        } else if (offset < rotary_half) {
            const float even = post[offset];
            const float odd = post[offset + rotary_half];
            const float c = float(cos_data[offset]);
            const float s = float(sin_data[offset]);
            output[index] = half(even * c - odd * s);
            output[head * head_dim + rotary_half + offset] = half(even * s + odd * c);
        }
    }
}
"#;

use crate::backend::metal::api::{Buffer, MTLSize};
use crate::backend::metal::{MetalContext, MetalTensor};

use super::set_bytes;

/// 标准布局 MLX affine 驻留权重视图(与 `MetalWeight::MlxAffine` pair_id == 0 分支一致)。
pub struct MlxAffineResident {
    pub packed: Buffer,
    pub scales: Buffer,
    pub biases: Buffer,
    pub scale_dtype: u32,
    pub bits: usize,
    pub group_size: usize,
    pub rows: usize,
    pub cols: usize,
}

impl MlxAffineResident {
    #[allow(clippy::too_many_arguments)]
    pub fn new(packed: Buffer, scales: Buffer, biases: Buffer, scale_dtype: u32, bits: usize, group_size: usize, rows: usize, cols: usize) -> Self {
        Self { packed, scales, biases, scale_dtype, bits, group_size, rows, cols }
    }
}

/// 把 `rmsnorm -> Q/K/V gemv(4-bit MLX affine 标准布局) -> 逐头 GemmaRMSNorm ->
/// Split-half RoPE` 融合为一次 launch。cos/sin 是**常驻全表 F16**,
/// `rope_offset` 为 position 行的字节偏移(见 `decode_rope_table_buffers`),
/// kernel 从行首直读,数值与逐 token 行上传逐位一致。norm 权重保持 F32 直读,
/// 避免 F16 量化在 42 层复合放大。
#[allow(clippy::too_many_arguments)]
pub fn fused_rmsnorm_qkv_head_norm_rope(
    ctx: &MetalContext,
    hidden: &MetalTensor,
    input_norm: &Buffer,
    q: &MlxAffineResident,
    k: &MlxAffineResident,
    v: &MlxAffineResident,
    query_norm: &Buffer,
    key_norm: &Buffer,
    cos_table: &Buffer,
    sin_table: &Buffer,
    rope_offset: u64,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
) -> Result<(MetalTensor, MetalTensor, MetalTensor), String> {
    if hidden.rows != 1 || hidden.cols == 0 || hidden.cols > 4096 {
        return Err(format!("融合 RMSNorm+QKV hidden=[{},{}] 需为单行且 <=4096 列", hidden.rows, hidden.cols));
    }
    let hidden_size = hidden.cols;
    for (name, weight) in [("q", q), ("k", k), ("v", v)] {
        if weight.bits != 4 || weight.cols != hidden_size || weight.scale_dtype > 2 || weight.group_size == 0 || !hidden_size.is_multiple_of(weight.group_size) {
            return Err(format!("融合 RMSNorm+QKV {name} weight=[{},{}] bits={} group={} 不兼容", weight.rows, weight.cols, weight.bits, weight.group_size));
        }
    }
    let q_rows = head_count.checked_mul(head_dim).ok_or("融合 RMSNorm+QKV head 溢出")?;
    let kv_rows = kv_head_count.checked_mul(head_dim).ok_or("融合 RMSNorm+QKV kv head 溢出")?;
    if q.rows != q_rows || k.rows != kv_rows || v.rows != kv_rows || head_dim == 0 || head_dim > 512 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Err(format!("融合 RMSNorm+QKV 形状不匹配 q={} k={} v={} heads={head_count}/{kv_head_count} head_dim={head_dim} rotary={rotary_dim}", q.rows, k.rows, v.rows));
    }
    let groups = head_count + 2 * kv_head_count;
    let query = ctx.tensor_zeros(1, q_rows);
    let key = ctx.tensor_zeros(1, kv_rows);
    let value = ctx.tensor_zeros(1, kv_rows);
    let hidden_u32 = super::validate_u32("融合 RMSNorm+QKV hidden", hidden_size)?;
    let head_dim_u32 = super::validate_u32("融合 RMSNorm+QKV head dim", head_dim)?;
    let rotary_u32 = super::validate_u32("融合 RMSNorm+QKV rotary", rotary_dim)?;
    let scale_dtype_u32 = super::validate_u32("融合 RMSNorm+QKV scale dtype", q.scale_dtype as usize)?;
    let group_u32 = super::validate_u32("融合 RMSNorm+QKV group", q.group_size)?;
    let heads_u32 = super::validate_u32("融合 RMSNorm+QKV heads", head_count)?;
    let kv_heads_u32 = super::validate_u32("融合 RMSNorm+QKV kv heads", kv_head_count)?;
    let pipeline = ctx.pipeline("fused_rmsnorm_qkv_head_norm_rope")?;
    if pipeline.max_total_threads_per_threadgroup() < 256 {
        return Err("融合 RMSNorm+QKV 需要至少 256 threads/threadgroup".to_owned());
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&hidden.buffer), 0);
    encoder.set_buffer(1, Some(input_norm), 0);
    encoder.set_buffer(2, Some(&q.packed), 0);
    encoder.set_buffer(3, Some(&q.scales), 0);
    encoder.set_buffer(4, Some(&q.biases), 0);
    encoder.set_buffer(5, Some(&k.packed), 0);
    encoder.set_buffer(6, Some(&k.scales), 0);
    encoder.set_buffer(7, Some(&k.biases), 0);
    encoder.set_buffer(8, Some(&v.packed), 0);
    encoder.set_buffer(9, Some(&v.scales), 0);
    encoder.set_buffer(10, Some(&v.biases), 0);
    encoder.set_buffer(11, Some(query_norm), 0);
    encoder.set_buffer(12, Some(key_norm), 0);
    encoder.set_buffer(13, Some(cos_table), rope_offset);
    encoder.set_buffer(14, Some(sin_table), rope_offset);
    encoder.set_buffer(15, Some(&query.buffer), 0);
    encoder.set_buffer(16, Some(&key.buffer), 0);
    encoder.set_buffer(17, Some(&value.buffer), 0);
    set_bytes(&encoder, 18, &hidden_u32);
    set_bytes(&encoder, 19, &head_dim_u32);
    set_bytes(&encoder, 20, &rotary_u32);
    set_bytes(&encoder, 21, &eps);
    set_bytes(&encoder, 22, &scale_dtype_u32);
    set_bytes(&encoder, 23, &group_u32);
    set_bytes(&encoder, 24, &heads_u32);
    set_bytes(&encoder, 25, &kv_heads_u32);
    encoder.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    let shape = format!("hidden={hidden_size},heads={head_count}+{kv_head_count},head_dim={head_dim},rotary={rotary_dim}");
    let read_bytes = hidden.buffer.length() + q.packed.length() + k.packed.length() + v.packed.length();
    ctx.commit_and_wait_profiled(&command, "fused_rmsnorm_qkv_head_norm_rope", &shape, read_bytes, query.buffer.length() + key.buffer.length() + value.buffer.length());
    Ok((query, key, value))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::backend::metal::MetalTensor;
    use half::{bf16, f16};

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }.to_vec()
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        let encoded: Vec<f16> = values.iter().map(|&value| f16::from_f32(value)).collect();
        let bytes = unsafe { std::slice::from_raw_parts(encoded.as_ptr().cast::<u8>(), std::mem::size_of_val(encoded.as_slice())) };
        bytes.to_vec()
    }

    fn bf16_word(value: f32) -> u16 {
        let bits = bf16::from_f32(value).to_le_bytes();
        u16::from_le_bytes(bits)
    }

    /// 融合 QKV+RoPE kernel 对 CPU oracle 的逐元素校验:
    /// 参考序列 = rmsnorm -> 逐行 affine gemv -> 逐头 GemmaRMSNorm(+1) -> split-half RoPE。
    #[test]
    fn fused_qkv_rope_matches_cpu_oracle() {
        if crate::backend::metal::api::Device::system_default().is_none() {
            return;
        }
        let hidden_size = 2560usize;
        let geometries: [(usize, usize, usize, usize); 2] = [(8, 2, 256, 256), (8, 2, 512, 512)];
        for (head_count, kv_head_count, head_dim, rotary_dim) in geometries {
            let fused_case = fused_qkv_rope_case(hidden_size, head_count, kv_head_count, head_dim, rotary_dim);
            match fused_case {
                Ok(()) => continue,
                Err(error) => panic!("geometry heads={head_count} kv={kv_head_count} head_dim={head_dim} rotary={rotary_dim}: {error}"),
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn fused_qkv_rope_case(hidden_size: usize, head_count: usize, kv_head_count: usize, head_dim: usize, rotary_dim: usize) -> Result<(), String> {
        let group_size = 64usize;
        let eps = 1.0e-6f32;
        let position = 7usize;
        let q_rows = head_count * head_dim;
        let kv_rows = kv_head_count * head_dim;

        let pattern = |offset: f32, step: f32| move |index: usize| ((index as f32 + offset) * step).sin();
        let hidden: Vec<f32> = (0..hidden_size).map(pattern(1.0, 0.031)).collect();
        let input_norm: Vec<f32> = (0..hidden_size).map(|index| 0.8 + 0.4 * ((index as f32 * 0.017).cos())).collect();
        let q_norm: Vec<f32> = (0..head_dim).map(|index| 0.1 * ((index as f32 * 0.05).sin())).collect();
        let k_norm: Vec<f32> = (0..head_dim).map(|index| 0.1 * ((index as f32 * 0.03).cos())).collect();
        let half = rotary_dim / 2;
        let cos: Vec<f32> = (0..(position + 1) * half).map(|index| ((index as f32 * 0.011) % 1.0).cos()).collect();
        let sin: Vec<f32> = (0..(position + 1) * half).map(|index| ((index as f32 * 0.013) % 1.0).sin()).collect();

        // 构造 4-bit affine 权重:codes/scales/biases 与 kernel 的解包语义一致。
        let build_weight = |rows: usize, seed: f32| -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<Vec<f32>>) {
            let groups = hidden_size / group_size;
            let mut packed = vec![0u8; rows * hidden_size.div_ceil(8) * 4];
            let mut scales = Vec::with_capacity(rows * groups * 2);
            let mut biases = Vec::with_capacity(rows * groups * 2);
            let mut decoded = vec![vec![0.0f32; hidden_size]; rows];
            for row in 0..rows {
                for group in 0..groups {
                    let scale = 1.0e-3 + ((row * 7 + group * 3) as f32 * seed).sin().abs() * 3.0e-3;
                    let bias = ((row * 11 + group * 5) as f32 * seed).sin() * 1.0e-3;
                    scales.extend_from_slice(&bf16_word(scale).to_le_bytes());
                    biases.extend_from_slice(&bf16_word(bias).to_le_bytes());
                    for element in 0..group_size {
                        let column = group * group_size + element;
                        let code = ((row * 31 + column * 17) % 16) as u32;
                        decoded[row][column] = scale * code as f32 + bias;
                        let word_index = row * hidden_size.div_ceil(8) * 4 + column / 8 * 4;
                        let word = u32::from_le_bytes([packed[word_index], packed[word_index + 1], packed[word_index + 2], packed[word_index + 3]]);
                        let shifted = word | (code << ((column % 8) * 4));
                        packed[word_index..word_index + 4].copy_from_slice(&shifted.to_le_bytes());
                    }
                }
            }
            (packed, scales, biases, decoded)
        };
        let (q_packed, q_scales, q_biases, q_decoded) = build_weight(q_rows, 0.019);
        let (k_packed, k_scales, k_biases, k_decoded) = build_weight(kv_rows, 0.023);
        let (v_packed, v_scales, v_biases, v_decoded) = build_weight(kv_rows, 0.029);

        // CPU oracle。
        let mean_square = hidden.iter().map(|&value| value * value).sum::<f32>() / hidden_size as f32;
        let inv = (mean_square + eps).sqrt().recip();
        let normed: Vec<f32> = hidden.iter().zip(&input_norm).map(|(&value, &weight)| value * inv * weight).collect();
        let head_norm = |rows: &[f32], weight: Option<&[f32]>| -> Vec<f32> {
            let mean = rows.iter().map(|&value| value * value).sum::<f32>() / head_dim as f32;
            let head_inv = (mean + eps).sqrt().recip();
            rows.iter().enumerate().map(|(index, &value)| value * head_inv * weight.map_or(1.0, |weight| weight[index] + 1.0)).collect()
        };
        let rope = |values: &[f32], _head: usize| -> Vec<f32> {
            let mut out = values.to_vec();
            for pair in 0..half {
                let even = values[pair];
                let odd = values[pair + half];
                let angle = position * half + pair;
                let c = cos[angle];
                let s = sin[angle];
                out[pair] = even * c - odd * s;
                out[pair + half] = even * s + odd * c;
            }
            out
        };
        let mut expected_q = vec![0.0f32; q_rows];
        let mut expected_k = vec![0.0f32; kv_rows];
        let mut expected_v = vec![0.0f32; kv_rows];
        for head in 0..head_count {
            let rows: Vec<f32> = (0..head_dim).map(|lane| q_decoded[head * head_dim + lane].iter().zip(&normed).map(|(weight, &value)| weight * value).sum()).collect();
            let normed_head = head_norm(&rows, Some(&q_norm));
            let roped = rope(&normed_head, head);
            for lane in 0..head_dim {
                expected_q[head * head_dim + lane] = roped[lane];
            }
        }
        for head in 0..kv_head_count {
            let k_rows_vec: Vec<f32> = (0..head_dim).map(|lane| k_decoded[head * head_dim + lane].iter().zip(&normed).map(|(weight, &value)| weight * value).sum()).collect();
            let v_rows_vec: Vec<f32> = (0..head_dim).map(|lane| v_decoded[head * head_dim + lane].iter().zip(&normed).map(|(weight, &value)| weight * value).sum()).collect();
            let k_head = head_norm(&k_rows_vec, Some(&k_norm));
            let v_head = head_norm(&v_rows_vec, None);
            let k_roped = rope(&k_head, head);
            for lane in 0..head_dim {
                expected_k[head * head_dim + lane] = k_roped[lane];
                expected_v[head * head_dim + lane] = v_head[lane];
            }
        }

        // GPU 融合 kernel。
        let ctx = MetalContext::new_default().unwrap();
        let hidden_buffer = ctx.shared_buffer(&f16_bytes(&hidden));
        let hidden_tensor = MetalTensor::new(hidden_buffer, 1, hidden_size);
        // 与生产一致:input_norm 走 F16 权重(kernel half 直读)
        let input_norm_buffer = ctx.shared_buffer(&f16_bytes(&input_norm));
        let q_norm_buffer = ctx.shared_buffer(&f32_bytes(&q_norm));
        let k_norm_buffer = ctx.shared_buffer(&f32_bytes(&k_norm));
        let cos_buffer = ctx.shared_buffer(&f16_bytes(&cos[position * half..(position + 1) * half]));
        let sin_buffer = ctx.shared_buffer(&f16_bytes(&sin[position * half..(position + 1) * half]));
        let upload = |bytes: &[u8]| ctx.shared_buffer(bytes);
        let q_weight = MlxAffineResident::new(upload(&q_packed), upload(&q_scales), upload(&q_biases), 0, 4, group_size, q_rows, hidden_size);
        let k_weight = MlxAffineResident::new(upload(&k_packed), upload(&k_scales), upload(&k_biases), 0, 4, group_size, kv_rows, hidden_size);
        let v_weight = MlxAffineResident::new(upload(&v_packed), upload(&v_scales), upload(&v_biases), 0, 4, group_size, kv_rows, hidden_size);
        let (query, key, value) =
            fused_rmsnorm_qkv_head_norm_rope(&ctx, &hidden_tensor, &input_norm_buffer, &q_weight, &k_weight, &v_weight, &q_norm_buffer, &k_norm_buffer, &cos_buffer, &sin_buffer, 0, head_count, kv_head_count, head_dim, rotary_dim, eps)
                .unwrap();

        let relative_l2 = |name: &str, expected: &[f32], actual: &[f32]| {
            let mut numerator = 0.0f64;
            let mut denominator = 0.0f64;
            for (want, got) in expected.iter().zip(actual) {
                numerator += (want - got) as f64 * (want - got) as f64;
                denominator += (*want) as f64 * (*want) as f64;
            }
            let relative = (numerator / denominator.max(1.0e-12)).sqrt();
            println!("[fused-qkv-norm-rope] {name} relative_l2={relative:.6} elements={}", expected.len());
            if relative >= 2.0e-2 { Err(format!("{name} relative_l2={relative} 超出容差")) } else { Ok(()) }
        };
        // 与真实顺序 Metal 路径逐段对比(这才是融合替换的对象)。
        use crate::backend::{Backend, GqaPrefillBackend};
        let input_norm_f16: Vec<f16> = input_norm.iter().map(|&value| f16::from_f32(value)).collect();
        let norm_weight = <MetalContext as crate::backend::BackendResources>::prepare_weight(&ctx, LinearWeight::F16(&input_norm_f16), 1, hidden_size).map_err(|error| error.to_string())?;
        let qk_norm_weight = |values: &[f32]| crate::backend::metal::MetalWeight::upload_f32(&ctx, values).unwrap();
        let q_norm_weight = qk_norm_weight(&q_norm);
        let k_norm_weight = qk_norm_weight(&k_norm);
        let v_norm_values = vec![0.0f32; head_dim];
        let v_norm_weight = qk_norm_weight(&v_norm_values);
        use crate::backend::LinearWeight;
        use crate::weight::format::quantization::{MlxAffineMatrix, QuantizedMatrixRef, ScaleDType};
        let q_matrix = MlxAffineMatrix::new(q_packed.clone(), q_scales.clone(), q_biases.clone(), ScaleDType::Bf16, 4, group_size, q_rows, hidden_size).map_err(|error| error.to_string())?;
        let q_linear = <MetalContext as crate::backend::BackendResources>::prepare_weight(&ctx, LinearWeight::Quantized(QuantizedMatrixRef::MlxAffine(&q_matrix)), q_rows, hidden_size).map_err(|error| error.to_string())?;
        let normed = ctx.rmsnorm(&hidden_tensor, &norm_weight, eps).map_err(|error| error.to_string())?;
        let q_gemv = ctx.linear(&normed, &q_linear).map_err(|error| error.to_string())?;
        let q_seq = ctx.gemma_rmsnorm_heads(&q_gemv, &q_norm_weight, head_count, head_dim, eps).map_err(|error| error.to_string())?;
        let q_seq = ctx.rope_prefix(&q_seq, head_count, rotary_dim, crate::attention::rope::RotaryLayout::SplitHalf, position, &cos, &sin).map_err(|error| error.to_string())?;
        let k_matrix = MlxAffineMatrix::new(k_packed.clone(), k_scales.clone(), k_biases.clone(), ScaleDType::Bf16, 4, group_size, kv_rows, hidden_size).map_err(|error| error.to_string())?;
        let k_linear = <MetalContext as crate::backend::BackendResources>::prepare_weight(&ctx, LinearWeight::Quantized(QuantizedMatrixRef::MlxAffine(&k_matrix)), kv_rows, hidden_size).map_err(|error| error.to_string())?;
        let k_gemv = ctx.linear(&normed, &k_linear).map_err(|error| error.to_string())?;
        let k_seq = ctx.gemma_rmsnorm_heads(&k_gemv, &k_norm_weight, kv_head_count, head_dim, eps).map_err(|error| error.to_string())?;
        let k_seq = ctx.rope_prefix(&k_seq, kv_head_count, rotary_dim, crate::attention::rope::RotaryLayout::SplitHalf, position, &cos, &sin).map_err(|error| error.to_string())?;
        let v_matrix = MlxAffineMatrix::new(v_packed.clone(), v_scales.clone(), v_biases.clone(), ScaleDType::Bf16, 4, group_size, kv_rows, hidden_size).map_err(|error| error.to_string())?;
        let v_linear = <MetalContext as crate::backend::BackendResources>::prepare_weight(&ctx, LinearWeight::Quantized(QuantizedMatrixRef::MlxAffine(&v_matrix)), kv_rows, hidden_size).map_err(|error| error.to_string())?;
        let v_gemv = ctx.linear(&normed, &v_linear).map_err(|error| error.to_string())?;
        let v_seq = ctx.gemma_rmsnorm_heads(&v_gemv, &v_norm_weight, kv_head_count, head_dim, eps).map_err(|error| error.to_string())?;
        let k_fused = ctx.tensor_to_f32(&key);
        let k_sequential = ctx.tensor_to_f32(&k_seq);
        let k_drift: f64 = k_fused.iter().zip(k_sequential.iter()).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
        let v_fused = ctx.tensor_to_f32(&value);
        let v_sequential = ctx.tensor_to_f32(&v_seq);
        let v_drift: f64 = v_fused.iter().zip(v_sequential.iter()).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
        println!("[seq-compare] k max abs diff={k_drift} v max abs diff={v_drift}");
        assert!(k_drift < 5.0e-2, "k drift={k_drift}");
        assert!(v_drift < 5.0e-2, "v drift={v_drift}");
        let q_fused = ctx.tensor_to_f32(&query);
        let q_sequential = ctx.tensor_to_f32(&q_seq);
        let seq_drift: f64 = q_fused.iter().zip(q_sequential.iter()).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
        println!("[seq-compare] fused-vs-sequential q max abs diff={seq_drift}");
        assert!(seq_drift < 5.0e-2, "fused vs sequential drift={seq_drift}");
        let _ = (&v_norm_weight, &k_norm_weight);
        relative_l2("q", &expected_q, &ctx.tensor_to_f32(&query))?;
        relative_l2("k", &expected_k, &ctx.tensor_to_f32(&key))?;
        relative_l2("v", &expected_v, &ctx.tensor_to_f32(&value))?;
        Ok(())
    }
}
