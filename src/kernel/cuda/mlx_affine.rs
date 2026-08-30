//! MLX `affine` groupwise 4-bit 量化设备内反量化算子。
//!
//! 对称 `kernel/metal/low_bit.rs` 的 `mlx_affine_*` kernel(M3 上 MLX 量化反量化路径)。
//! 权重保持 packed I32(每 int32 存 8 个无符号 INT4,LE),配合 per-group
//! `(scale, bias)` 在 kernel 内即时反量化,避免 host 展开为 f16(约 10× 显存 + HTOD)。
//!
//! 数据布局与 `weight/format/quantization.rs::MlxAffineMatrix::decode` 完全一致:
//! - `packed` 行优先 `[out_dim, k_dim/8]` 的 int32,第 `n,k` 个权重所在 word
//!   为 `packed[n*(k_dim/8) + k/8]`,nibble 位移 `(k&7)*4`,无符号值 `(word>>shift)&0xF`。
//! - `scales`/`biases` 行优先 `[out_dim, groups]`(`groups = k_dim/group_size`),每组一对标量。
//! - 解码值 `w = scale * code + bias`。
//!
//! 当前仅支持 bits=4(MLX affine 4-bit 默认)。bits=8 走 host decode 上传 f16 回退。
//!
//! 路径:
//! - decode(input.rows=1):`gemv_f16` / 融合 gate+up 的 `gated_silu_gemv_f16`
//!   (共用输入带宽,减半权重读取 + 单次 silu_mul)。
//! - prefill(input.rows>1):`dequant_f16` 整矩阵反量化为 f16 再喂 cuBLAS hgemm。

use cudarc::driver::safe::CudaSlice;

use super::{CudaContext, CudaSliceF16, CudaTensor, LaunchConfig, PushKernelArg, THREADS};

/// 本模块的 CUDA shader(本文件用到的 kernel + 文件私有 `__device__` helper)。
///
/// 共用 helper 见 `mod.rs` 前导。`mod.rs::kernels_source()` 把各模块 SHADERS 拼接。
// kernels: mlx_affine_gemv_f16, mlx_affine_gated_silu_gemv_f16, mlx_affine_dequant_f16
// private helpers: mlx_affine_scale
pub const SHADERS: &str = r#"
// 读取 per-group scale/bias。dtype: 0=BF16, 1=F16, 2=F32(对称 ScaleDType::metal_code)。
// BF16 是 fp32 的高 16 位,左移 16 位后按 int 位模式重解释即得 fp32。
__device__ __forceinline__ float mlx_affine_scale(const unsigned char *params, unsigned long long index, unsigned int dtype)
{
    if (dtype == 0u) {
        unsigned int bits = ((unsigned int)(reinterpret_cast<const unsigned short *>(params)[index])) << 16u;
        return __int_as_float(bits);
    }
    if (dtype == 1u) {
        return __half2float(reinterpret_cast<const __half *>(params)[index]);
    }
    return reinterpret_cast<const float *>(params)[index];
}

// Decode GEMV:单 token 输入。每个 block 算一个输出行 n,内部跨步累加后 warp/block 归约。
// 权重以 packed int32 即时反量化,scale/bias 先 tile 进 shared memory 复用。
extern "C" __global__ void mlx_affine_gemv_f16(
    const __half * __restrict__ input,
    const unsigned int * __restrict__ packed,
    const unsigned char * __restrict__ scales,
    const unsigned char * __restrict__ biases,
    __half * __restrict__ output,
    const unsigned int k_dim,
    const unsigned int out_dim,
    const unsigned int group_size,
    const unsigned int scale_dtype)
{
    const unsigned int n = blockIdx.x;
    if (n >= out_dim) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    extern __shared__ float shared[];
    const unsigned int groups = k_dim / group_size;
    float *scale_tile = shared;                       // groups 个 float
    float *bias_tile = shared + groups;               // groups 个 float
    float *partial = shared + 2u * groups;            // warp_count 个 float

    const unsigned int packed_cols = k_dim / 8u;
    const unsigned long long packed_row = (unsigned long long)n * packed_cols;
    const unsigned long long scale_row = (unsigned long long)n * groups;

    // 1. 把本行的 group (scale, bias) 装进 shared(每线程跨步搬运)。
    for (unsigned int g = tid; g < groups; g += blockDim.x) {
        scale_tile[g] = mlx_affine_scale(scales, scale_row + g, scale_dtype);
        bias_tile[g] = mlx_affine_scale(biases, scale_row + g, scale_dtype);
    }
    __syncthreads();

    // 2. 跨步累加 dot(input, weight_row)。每线程每次处理一个 int32 word(8 权重):
    //    packed word 只读一次、8 个 nibble 全用(消除 8 线程重复读同一 word 的冗余,
    //    权重带宽是 decode 瓶颈);input 用 int4(16B)一次取 8 个 half。
    //    group_size 是 8 的倍数 ⇒ 一个 word 完全落在同一组内,scale/bias 提到累加外只取一次。
    float acc = 0.0f;
    const unsigned int words_per_group = group_size / 8u;
    for (unsigned int w = tid; w < packed_cols; w += blockDim.x) {
        unsigned int word = packed[packed_row + w];
        float s = scale_tile[w / words_per_group];
        float b = bias_tile[w / words_per_group];
        int4 iv = reinterpret_cast<const int4 *>(input + w * 8u)[0];
        unsigned int ip[4] = { iv.x, iv.y, iv.z, iv.w };
        float local = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float x = __half2float(__ushort_as_half((unsigned short)((ip[j >> 1] >> ((j & 1) * 16u)) & 0xFFFFu)));
            int code = (int)((word >> (j * 4u)) & 0xFu);
            // MLX affine:w = scale * unsigned_code + bias。
            local += x * (s * (float)code + b);
        }
        acc += local;
    }

    // 3. warp 内 shuffle 归约,再 warp 间用 shared 归约。
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < warp_count) ? partial[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        }
        if (lane == 0) output[n] = __float2half(sum);
    }
}

// Decode 融合 gate/up GEMV + SiLU:output[n] = silu(gate_dot) * up_dot。
// 两个权重行共用同一段输入,相比两次独立 GEMV 减半输入带宽。
extern "C" __global__ void mlx_affine_gated_silu_gemv_f16(
    const __half * __restrict__ input,
    const unsigned int * __restrict__ gate_packed,
    const unsigned char * __restrict__ gate_scales,
    const unsigned char * __restrict__ gate_biases,
    const unsigned int * __restrict__ up_packed,
    const unsigned char * __restrict__ up_scales,
    const unsigned char * __restrict__ up_biases,
    __half * __restrict__ output,
    const unsigned int k_dim,
    const unsigned int out_dim,
    const unsigned int group_size,
    const unsigned int scale_dtype)
{
    const unsigned int n = blockIdx.x;
    if (n >= out_dim) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    extern __shared__ float shared[];
    const unsigned int groups = k_dim / group_size;
    float *gate_scale_tile = shared;                       // groups
    float *gate_bias_tile = shared + groups;               // groups
    float *up_scale_tile = shared + 2u * groups;          // groups
    float *up_bias_tile = shared + 3u * groups;            // groups
    float *partial = shared + 4u * groups;                 // 2 * warp_count(gate, up)
    const unsigned int packed_cols = k_dim / 8u;
    const unsigned long long packed_row = (unsigned long long)n * packed_cols;
    const unsigned long long scale_row = (unsigned long long)n * groups;

    for (unsigned int g = tid; g < groups; g += blockDim.x) {
        gate_scale_tile[g] = mlx_affine_scale(gate_scales, scale_row + g, scale_dtype);
        gate_bias_tile[g] = mlx_affine_scale(gate_biases, scale_row + g, scale_dtype);
        up_scale_tile[g] = mlx_affine_scale(up_scales, scale_row + g, scale_dtype);
        up_bias_tile[g] = mlx_affine_scale(up_biases, scale_row + g, scale_dtype);
    }
    __syncthreads();

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    // 同 gemv:每线程处理一个 int32 word(8 权重),gate/up 共用同一段 input(int4 一次取),
    // 各自 packed word 只读一次全用 8 nibble;scale/bias 提到累加外。
    const unsigned int words_per_group = group_size / 8u;
    for (unsigned int w = tid; w < packed_cols; w += blockDim.x) {
        unsigned int group = w / words_per_group;
        float gs = gate_scale_tile[group];
        float gb = gate_bias_tile[group];
        float us = up_scale_tile[group];
        float ub = up_bias_tile[group];
        int4 iv = reinterpret_cast<const int4 *>(input + w * 8u)[0];
        unsigned int ip[4] = { iv.x, iv.y, iv.z, iv.w };
        unsigned int gw = gate_packed[packed_row + w];
        unsigned int uw = up_packed[packed_row + w];
        float g_local = 0.0f;
        float u_local = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float x = __half2float(__ushort_as_half((unsigned short)((ip[j >> 1] >> ((j & 1) * 16u)) & 0xFFFFu)));
            int g_code = (int)((gw >> (j * 4u)) & 0xFu);
            int u_code = (int)((uw >> (j * 4u)) & 0xFu);
            g_local += x * (gs * (float)g_code + gb);
            u_local += x * (us * (float)u_code + ub);
        }
        gate_acc += g_local;
        up_acc += u_local;
    }

    for (int off = 16; off > 0; off >>= 1) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, off);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, off);
    }
    if (lane == 0) {
        partial[warp] = gate_acc;
        partial[warp_count + warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        float gs = (lane < warp_count) ? partial[lane] : 0.0f;
        float us = (lane < warp_count) ? partial[warp_count + lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            gs += __shfl_down_sync(0xffffffffu, gs, off);
            us += __shfl_down_sync(0xffffffffu, us, off);
        }
        if (lane == 0) {
            // silu(g) * u = (g / (1 + e^-g)) * u
            output[n] = __float2half((gs / (1.0f + expf(-gs))) * us);
        }
    }
}

// 整矩阵反量化为 f16(prefill 用):每线程负责一个 int32 word(8 元素),packed word 只读一次,
// 8 个 nibble 全用,输出按 int4(8 个 f16)合并写一次。
extern "C" __global__ void mlx_affine_dequant_f16(
    const unsigned int * __restrict__ packed,
    const unsigned char * __restrict__ scales,
    const unsigned char * __restrict__ biases,
    __half * __restrict__ output,
    const unsigned int k_dim,
    const unsigned int out_dim,
    const unsigned int group_size,
    const unsigned int scale_dtype)
{
    const unsigned int packed_cols = k_dim / 8u;
    const unsigned int groups = k_dim / group_size;
    const unsigned int words_per_group = group_size / 8u;
    const unsigned int total_words = out_dim * packed_cols;
    for (unsigned int w = blockIdx.x * blockDim.x + threadIdx.x; w < total_words; w += gridDim.x * blockDim.x) {
        unsigned int n = w / packed_cols;
        unsigned int lw = w - n * packed_cols;
        unsigned int word = packed[(unsigned long long)n * packed_cols + lw];
        float s = mlx_affine_scale(scales, (unsigned long long)n * groups + lw / words_per_group, scale_dtype);
        float b = mlx_affine_scale(biases, (unsigned long long)n * groups + lw / words_per_group, scale_dtype);
        __half2 pair[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int c0 = (int)((word >> ((j * 2) * 4u)) & 0xFu);
            int c1 = (int)((word >> ((j * 2 + 1) * 4u)) & 0xFu);
            pair[j] = __floats2half2_rn(s * (float)c0 + b, s * (float)c1 + b);
        }
        reinterpret_cast<int4 *>(output + ((unsigned long long)n * k_dim + lw * 8u))[0] = reinterpret_cast<int4 &>(pair);
    }
}
"#;

const GROUP_ALIGN: usize = 8;
const BITS: u32 = 4;
const VALUES_PER_WORD: usize = 32 / 4; // 8

// packed 以 I32 字节流上传(`CudaSlice<u8>`),kernel 内 reinterpret 为 `const unsigned int*`。
// device 分配 4 字节对齐,字节序与 host 一致(LE),读取安全。
fn validate_packed(k_dim: usize, out_dim: usize, packed: &CudaSlice<u8>) -> Result<(), String> {
    if !k_dim.is_multiple_of(GROUP_ALIGN) {
        return Err(format!("MLX affine k_dim={k_dim} 必须对齐 8"));
    }
    let expected = out_dim.checked_mul(k_dim / VALUES_PER_WORD).and_then(|n| n.checked_mul(4)).ok_or("MLX affine packed 大小溢出")?;
    if packed.len() != expected {
        return Err(format!("MLX affine packed bytes={}，期望 {expected}(out_dim={out_dim}, k_dim={k_dim})", packed.len()));
    }
    Ok(())
}

fn scale_bytes(scale_dtype: u32) -> usize {
    if scale_dtype == 2 { 4 } else { 2 }
}

fn validate_params(k_dim: usize, out_dim: usize, group_size: usize, scale_dtype: u32, params: &CudaSlice<u8>, name: &str) -> Result<(), String> {
    if group_size == 0 || !k_dim.is_multiple_of(group_size) {
        return Err(format!("MLX affine {name} group_size={group_size} 与 k_dim={k_dim} 不整除"));
    }
    // kernel 每线程处理一个 int32 word(8 权重),要求一个 word 不跨组 ⇒ group_size 必须是 8 的倍数。
    if !group_size.is_multiple_of(8) {
        return Err(format!("MLX affine {name} group_size={group_size} 必须是 8 的倍数(kernel 按 int32 word 反量化)"));
    }
    let expected = out_dim.checked_mul(k_dim / group_size).and_then(|n| n.checked_mul(scale_bytes(scale_dtype))).ok_or_else(|| format!("MLX affine {name} 参数大小溢出"))?;
    if params.len() != expected {
        return Err(format!("MLX affine {name} bytes={}，期望 {expected}", params.len()));
    }
    Ok(())
}

/// shared memory 字节数:scale_tile + bias_tile + warp 间归约缓冲。
fn gemv_shared_bytes(k_dim: usize, group_size: usize) -> Result<u32, String> {
    let groups = k_dim / group_size;
    let floats = groups.checked_mul(2).and_then(|n| n.checked_add(THREADS as usize / 32)).ok_or("MLX affine gemv shared 溢出")?;
    Ok((floats * std::mem::size_of::<f32>()) as u32)
}

/// Decode GEMV(单 token)。input `[1, k_dim]`,weight packed `[out_dim, k_dim]` → output `[1, out_dim]`。
#[allow(clippy::too_many_arguments)]
pub fn gemv_f16(ctx: &CudaContext, input: &CudaTensor, packed: &CudaSlice<u8>, scales: &CudaSlice<u8>, biases: &CudaSlice<u8>, out_dim: usize, group_size: usize, scale_dtype: u32, bits: u32) -> Result<CudaTensor, String> {
    if input.rows != 1 {
        return Err(format!("MLX affine GEMV 期望单行输入,实际 rows={}", input.rows));
    }
    if bits != BITS {
        return Err(format!("MLX affine GEMV 当前仅支持 bits=4，实际 bits={bits}"));
    }
    let k_dim = input.cols;
    validate_packed(k_dim, out_dim, packed)?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, scales, "scales")?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, biases, "biases")?;
    let output = ctx.tensor_uninit(1, out_dim)?;
    let func = ctx.function("mlx_affine_gemv_f16")?;
    let cfg = LaunchConfig { grid_dim: (out_dim as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: gemv_shared_bytes(k_dim, group_size)? };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(packed)
            .arg(scales)
            .arg(biases)
            .arg(&output.slice)
            .arg(&(k_dim as u32))
            .arg(&(out_dim as u32))
            .arg(&(group_size as u32))
            .arg(&scale_dtype)
            .launch(cfg)
            .map_err(|e| format!("launch mlx_affine_gemv_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// Decode 融合 gate/up GEMV + SiLU。两个 packed 权重同形 `[out_dim, k_dim]` → output `[1, out_dim]`。
#[allow(clippy::too_many_arguments)]
pub fn gated_silu_gemv_f16(
    ctx: &CudaContext,
    input: &CudaTensor,
    gate_packed: &CudaSlice<u8>,
    gate_scales: &CudaSlice<u8>,
    gate_biases: &CudaSlice<u8>,
    up_packed: &CudaSlice<u8>,
    up_scales: &CudaSlice<u8>,
    up_biases: &CudaSlice<u8>,
    out_dim: usize,
    group_size: usize,
    gate_scale_dtype: u32,
    up_scale_dtype: u32,
    bits: u32,
) -> Result<CudaTensor, String> {
    if input.rows != 1 {
        return Err(format!("MLX affine gated GEMV 期望单行输入,实际 rows={}", input.rows));
    }
    if bits != BITS {
        return Err(format!("MLX affine gated GEMV 当前仅支持 bits=4，实际 bits={bits}"));
    }
    // kernel 用单一 scale_dtype 同时解码 gate/up,两侧 dtype 不同会得到静默错误的结果。
    if gate_scale_dtype != up_scale_dtype {
        return Err(format!("MLX affine gated GEMV scale dtype 不一致: gate={gate_scale_dtype} up={up_scale_dtype}"));
    }
    let scale_dtype = gate_scale_dtype;
    let k_dim = input.cols;
    validate_packed(k_dim, out_dim, gate_packed)?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, gate_scales, "gate_scales")?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, gate_biases, "gate_biases")?;
    validate_packed(k_dim, out_dim, up_packed)?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, up_scales, "up_scales")?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, up_biases, "up_biases")?;
    let output = ctx.tensor_uninit(1, out_dim)?;
    let func = ctx.function("mlx_affine_gated_silu_gemv_f16")?;
    let groups = k_dim / group_size;
    let shared_floats = 4 * groups + 2 * (THREADS as usize / 32);
    let cfg = LaunchConfig { grid_dim: (out_dim as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(gate_packed)
            .arg(gate_scales)
            .arg(gate_biases)
            .arg(up_packed)
            .arg(up_scales)
            .arg(up_biases)
            .arg(&output.slice)
            .arg(&(k_dim as u32))
            .arg(&(out_dim as u32))
            .arg(&(group_size as u32))
            .arg(&scale_dtype)
            .launch(cfg)
            .map_err(|e| format!("launch mlx_affine_gated_silu_gemv_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 整矩阵反量化为 f16 行优先 `[out_dim, k_dim]`(prefill 喂 cuBLAS)。
#[allow(clippy::too_many_arguments)]
pub fn dequant_f16(ctx: &CudaContext, packed: &CudaSlice<u8>, scales: &CudaSlice<u8>, biases: &CudaSlice<u8>, out_dim: usize, k_dim: usize, group_size: usize, scale_dtype: u32, bits: u32) -> Result<CudaSliceF16, String> {
    if bits != BITS {
        return Err(format!("MLX affine dequant 当前仅支持 bits=4，实际 bits={bits}"));
    }
    validate_packed(k_dim, out_dim, packed)?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, scales, "scales")?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, biases, "biases")?;
    let elements = out_dim.checked_mul(k_dim).ok_or("MLX affine dequant 输出溢出")?;
    let output = ctx.buffer_uninit::<half::f16>(elements)?;
    let func = ctx.function("mlx_affine_dequant_f16")?;
    // 每线程处理一个 int32 word(8 元素);grid 按 word 数覆盖,block 用 THREADS。
    let words = out_dim.checked_mul(k_dim / VALUES_PER_WORD).ok_or("MLX affine dequant word 数溢出")?;
    let blocks = words.min(THREADS as usize * 1024).div_ceil(THREADS as usize).max(1) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(packed)
            .arg(scales)
            .arg(biases)
            .arg(&output)
            .arg(&(k_dim as u32))
            .arg(&(out_dim as u32))
            .arg(&(group_size as u32))
            .arg(&scale_dtype)
            .launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| format!("launch mlx_affine_dequant_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use super::*;
    use crate::weight::format::quantization::{MlxAffineMatrix, ScaleDType};

    fn check_close(name: &str, actual: &[f32], expect: &[f32]) {
        let atol = 1e-2f32;
        let rtol = 1e-2f32;
        for (i, (a, e)) in actual.iter().zip(expect).enumerate() {
            let err = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(err <= tol, "{name}[{i}] 偏差 {err:.4} 超阈值 {tol:.4}(actual={a:.4} expect={e:.4})");
        }
    }

    // 构造确定性 MLX affine 矩阵:权重按 (row,col) 生成无符号 int4,scale/bias 按组设定。
    // 解码公式:w = scale * code + bias。code ∈ [0, 15]。
    fn build_matrix(rows: usize, cols: usize, group_size: usize, dtype: ScaleDType) -> MlxAffineMatrix {
        let mut packed = vec![0u8; rows * (cols.div_ceil(8)) * 4];
        let mut scales = vec![0u8; rows * (cols / group_size) * dtype.bytes()];
        let mut biases = vec![0u8; rows * (cols / group_size) * dtype.bytes()];
        let groups = cols / group_size;
        for row in 0..rows {
            for col in 0..cols {
                // 无符号 code ∈ [0, 15]。
                let code: u32 = ((row * 13 + col * 5) % 16) as u32;
                let word_off = (row * (cols / 8) + col / 8) * 4;
                let word = u32::from_le_bytes(packed[word_off..word_off + 4].try_into().unwrap()) | (code << ((col % 8) * 4));
                packed[word_off..word_off + 4].copy_from_slice(&word.to_le_bytes());
            }
            for g in 0..groups {
                let scale = 0.05 * ((g + 1) as f32);
                let bias = -0.1 * (g as f32) + 0.02;
                let off = (row * groups + g) * dtype.bytes();
                match dtype {
                    ScaleDType::Bf16 => {
                        let s = half::bf16::from_f32(scale).to_le_bytes();
                        let b = half::bf16::from_f32(bias).to_le_bytes();
                        scales[off..off + 2].copy_from_slice(&s);
                        biases[off..off + 2].copy_from_slice(&b);
                    }
                    ScaleDType::F16 => {
                        let s = half::f16::from_f32(scale).to_le_bytes();
                        let b = half::f16::from_f32(bias).to_le_bytes();
                        scales[off..off + 2].copy_from_slice(&s);
                        biases[off..off + 2].copy_from_slice(&b);
                    }
                    ScaleDType::F32 => {
                        scales[off..off + 4].copy_from_slice(&scale.to_le_bytes());
                        biases[off..off + 4].copy_from_slice(&bias.to_le_bytes());
                    }
                }
            }
        }
        MlxAffineMatrix::new(packed, scales, biases, dtype, 4, group_size, rows, cols).unwrap()
    }

    #[test]
    fn gemv_matches_reference() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        for dtype in [ScaleDType::F16, ScaleDType::Bf16, ScaleDType::F32] {
            let rows = 64;
            let cols = 64;
            let group = 32;
            let matrix = build_matrix(rows, cols, group, dtype);
            let reference = matrix.decode().unwrap();

            let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.3).collect();
            let input_gpu = ctx.tensor_from_f32(&input, 1, cols).unwrap();

            // 上传 packed(I32 字节流,kernel 内 reinterpret 为 uint),scales/biases 为 u8。
            let packed_gpu = ctx.stream().clone_htod::<u8, _>(matrix.packed()).unwrap();
            let scales_gpu = ctx.stream().clone_htod::<u8, _>(matrix.scales()).unwrap();
            let biases_gpu = ctx.stream().clone_htod::<u8, _>(matrix.biases()).unwrap();

            let out_gpu = gemv_f16(&ctx, &input_gpu, &packed_gpu, &scales_gpu, &biases_gpu, rows, group, dtype.metal_code(), 4).unwrap();
            let out = ctx.tensor_to_f32(&out_gpu).unwrap();

            // CPU 参考:output[n] = Σ_k input[k] * reference[n*cols+k]。
            let mut expect = vec![0.0f32; rows];
            for n in 0..rows {
                for k in 0..cols {
                    expect[n] += input[k] * reference[n * cols + k];
                }
            }
            check_close(&format!("mlx_affine_gemv_{dtype:?}"), &out, &expect);
        }
    }

    #[test]
    fn dequant_matches_reference() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        let rows = 48;
        let cols = 64;
        let group = 32;
        let matrix = build_matrix(rows, cols, group, ScaleDType::F16);
        let reference = matrix.decode().unwrap();

        let packed_gpu = ctx.stream().clone_htod::<u8, _>(matrix.packed()).unwrap();
        let scales_gpu = ctx.stream().clone_htod::<u8, _>(matrix.scales()).unwrap();
        let biases_gpu = ctx.stream().clone_htod::<u8, _>(matrix.biases()).unwrap();
        let f16_buf = dequant_f16(&ctx, &packed_gpu, &scales_gpu, &biases_gpu, rows, cols, group, ScaleDType::F16.metal_code(), 4).unwrap();
        let host = ctx.stream().clone_dtoh::<half::f16, _>(&f16_buf).unwrap();
        let out: Vec<f32> = host.iter().map(|v| v.to_f32()).collect();
        check_close("mlx_affine_dequant", &out, &reference);
    }

    #[test]
    fn gated_silu_matches_reference() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        let rows = 64;
        let cols = 64;
        let group = 32;
        let gate = build_matrix(rows, cols, group, ScaleDType::F16);
        let up = build_matrix(rows, cols, group, ScaleDType::F16);
        let gate_ref = gate.decode().unwrap();
        let up_ref = up.decode().unwrap();

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.02 - 0.6).collect();
        let input_gpu = ctx.tensor_from_f32(&input, 1, cols).unwrap();

        let upload = |m: &MlxAffineMatrix| {
            let p = ctx.stream().clone_htod::<u8, _>(m.packed()).unwrap();
            let s = ctx.stream().clone_htod::<u8, _>(m.scales()).unwrap();
            let b = ctx.stream().clone_htod::<u8, _>(m.biases()).unwrap();
            (p, s, b)
        };
        let (gp, gs, gb) = upload(&gate);
        let (up_p, up_s, up_b) = upload(&up);

        let out_gpu = gated_silu_gemv_f16(&ctx, &input_gpu, &gp, &gs, &gb, &up_p, &up_s, &up_b, rows, group, ScaleDType::F16.metal_code(), ScaleDType::F16.metal_code(), 4).unwrap();
        let out = ctx.tensor_to_f32(&out_gpu).unwrap();

        let mut expect = vec![0.0f32; rows];
        for n in 0..rows {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for k in 0..cols {
                g += input[k] * gate_ref[n * cols + k];
                u += input[k] * up_ref[n * cols + k];
            }
            expect[n] = (g / (1.0 + (-g).exp())) * u;
        }
        check_close("mlx_affine_gated_silu", &out, &expect);
    }
}
