//! compressed-tensors W4A16(AWQ)设备内反量化算子。
//!
//! 对称 `kernel/metal/low_bit.rs` 的 `w4a16_*` kernel。权重保持 packed I32
//! (每 int32 存 8 个有符号 INT4,文件中存为 `code+8`,解码恢复 `[-8,7]`),
//! 配合 per-group scale(BF16/F16/F32)在 kernel 内即时反量化,避免 host 展开为 f16。
//!
//! 数据布局与 `weight/codec/groupwise.rs::decode_w4a16_matrix` 完全一致:
//! - `packed` 行优先 `[out_dim, k_dim/8]` 的 int32,第 `n,k` 个权重所在 word
//!   为 `packed[n*(k_dim/8) + k/8]`, nibble 位移 `(k&7)*4`, 有符号值 `((word>>shift)&0xF)-8`。
//! - `scales` 行优先 `[out_dim, groups]`(`groups = k_dim/group_size`),每组一个标量。
//! - 解码值 `w = float(code) * scale`。

use cudarc::driver::safe::CudaSlice;

use super::{CudaContext, CudaTensor, LaunchConfig, PushKernelArg, THREADS};

/// 本模块的 CUDA shader(本文件用到的 kernel + 文件私有 `__device__` helper)。
///
/// 共用 helper 见 `mod.rs` 前导。`mod.rs::kernels_source()` 把各模块 SHADERS 拼接。
// kernels: w4a16_gemv_f16, w4a16_gated_silu_gemv_f16, w4a16_dequant_f16
// private helpers: w4a16_scale
pub const SHADERS: &str = r#"
// 读取 per-group scale。dtype: 0=BF16, 1=F16, 2=F32(对称 ScaleDType::metal_code)。
// BF16 是 fp32 的高 16 位,左移 16 位后按 int 位模式重解释即得 fp32。
__device__ __forceinline__ float w4a16_scale(const unsigned char *scales, unsigned long long index, unsigned int dtype)
{
    if (dtype == 0u) {
        unsigned int bits = ((unsigned int)(reinterpret_cast<const unsigned short *>(scales)[index])) << 16u;
        return __int_as_float(bits);
    }
    if (dtype == 1u) {
        return __half2float(reinterpret_cast<const __half *>(scales)[index]);
    }
    return reinterpret_cast<const float *>(scales)[index];
}

// Decode GEMV:单 token 输入。每个 block 算一个输出行 n,内部跨步累加后 warp/block 归约。
// 权重以 packed int32 即时反量化,scale 先 tile 进 shared memory 复用。
extern "C" __global__ void w4a16_gemv_f16(
    const __half * __restrict__ input,
    const unsigned int * __restrict__ packed,
    const unsigned char * __restrict__ scales,
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
    float *scale_tile = shared;                       // groups 个 float
    float *partial = shared + (k_dim / group_size);   // warp_count 个 float

    const unsigned int groups = k_dim / group_size;
    const unsigned int packed_cols = k_dim / 8u;
    const unsigned long long packed_row = (unsigned long long)n * packed_cols;
    const unsigned long long scale_row = (unsigned long long)n * groups;

    // 1. 把本行的 group scale 装进 shared(每线程跨步搬运)。
    for (unsigned int g = tid; g < groups; g += blockDim.x) {
        scale_tile[g] = w4a16_scale(scales, scale_row + g, scale_dtype);
    }
    __syncthreads();

    // 2. 跨步累加 dot(input, weight_row)。每线程每次处理一个 int32 word(8 权重):
    //    packed word 只读一次、8 个 nibble 全用(消除旧版 8 线程重复读同一 word 的冗余,
    //    权重带宽是 decode 瓶颈);input 用 int4(16B)一次取 8 个 half。
    //    group_size 是 8 的倍数 ⇒ 一个 word 完全落在同一组内,scale 提到累加外只乘一次。
    float acc = 0.0f;
    const unsigned int words_per_group = group_size / 8u;
    for (unsigned int w = tid; w < packed_cols; w += blockDim.x) {
        unsigned int word = packed[packed_row + w];
        float s = scale_tile[w / words_per_group];
        int4 iv = reinterpret_cast<const int4 *>(input + w * 8u)[0];
        unsigned int ip[4] = { iv.x, iv.y, iv.z, iv.w };
        float local = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float x = __half2float(__ushort_as_half((unsigned short)((ip[j >> 1] >> ((j & 1) * 16u)) & 0xFFFFu)));
            int code = (int)((word >> (j * 4u)) & 0xFu) - 8;
            local += x * (float)code;
        }
        acc += local * s;
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
extern "C" __global__ void w4a16_gated_silu_gemv_f16(
    const __half * __restrict__ input,
    const unsigned int * __restrict__ gate_packed,
    const unsigned char * __restrict__ gate_scales,
    const unsigned int * __restrict__ up_packed,
    const unsigned char * __restrict__ up_scales,
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
    float *gate_tile = shared;                         // groups
    float *up_tile = shared + groups;                  // groups
    float *partial = shared + 2u * groups;             // 2 * warp_count(gate, up)
    const unsigned int packed_cols = k_dim / 8u;
    const unsigned long long packed_row = (unsigned long long)n * packed_cols;
    const unsigned long long scale_row = (unsigned long long)n * groups;

    for (unsigned int g = tid; g < groups; g += blockDim.x) {
        gate_tile[g] = w4a16_scale(gate_scales, scale_row + g, scale_dtype);
        up_tile[g] = w4a16_scale(up_scales, scale_row + g, scale_dtype);
    }
    __syncthreads();

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    // 同 gemv:每线程处理一个 int32 word(8 权重),gate/up 共用同一段 input(int4 一次取),
    // 各自 packed word 只读一次全用 8 nibble;scale 提到累加外。
    const unsigned int words_per_group = group_size / 8u;
    for (unsigned int w = tid; w < packed_cols; w += blockDim.x) {
        unsigned int group = w / words_per_group;
        float gs = gate_tile[group];
        float us = up_tile[group];
        int4 iv = reinterpret_cast<const int4 *>(input + w * 8u)[0];
        unsigned int ip[4] = { iv.x, iv.y, iv.z, iv.w };
        unsigned int gw = gate_packed[packed_row + w];
        unsigned int uw = up_packed[packed_row + w];
        float g_local = 0.0f;
        float u_local = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float x = __half2float(__ushort_as_half((unsigned short)((ip[j >> 1] >> ((j & 1) * 16u)) & 0xFFFFu)));
            g_local += x * (float)((int)((gw >> (j * 4u)) & 0xFu) - 8);
            u_local += x * (float)((int)((uw >> (j * 4u)) & 0xFu) - 8);
        }
        gate_acc += g_local * gs;
        up_acc += u_local * us;
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
extern "C" __global__ void w4a16_dequant_f16(
    const unsigned int * __restrict__ packed,
    const unsigned char * __restrict__ scales,
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
        float s = w4a16_scale(scales, (unsigned long long)n * groups + lw / words_per_group, scale_dtype);
        __half2 pair[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int c0 = (int)((word >> ((j * 2) * 4u)) & 0xFu) - 8;
            int c1 = (int)((word >> ((j * 2 + 1) * 4u)) & 0xFu) - 8;
            pair[j] = __floats2half2_rn((float)c0 * s, (float)c1 * s);
        }
        reinterpret_cast<int4 *>(output + ((unsigned long long)n * k_dim + lw * 8u))[0] = reinterpret_cast<int4 &>(pair);
    }
}
"#;

const GROUP_ALIGN: usize = 8;

// packed 以 I32 字节流上传(`CudaSlice<u8>`),kernel 内 reinterpret 为 `const unsigned int*`。
// device 分配 4 字节对齐,字节序与 host 一致(LE),读取安全。
fn validate_packed(k_dim: usize, out_dim: usize, packed: &CudaSlice<u8>) -> Result<(), String> {
    if !k_dim.is_multiple_of(GROUP_ALIGN) {
        return Err(format!("W4A16 k_dim={k_dim} 必须对齐 8"));
    }
    let expected = out_dim.checked_mul(k_dim / GROUP_ALIGN).and_then(|n| n.checked_mul(4)).ok_or("W4A16 packed 大小溢出")?;
    if packed.len() != expected {
        return Err(format!("W4A16 packed bytes={}，期望 {expected}(out_dim={out_dim}, k_dim={k_dim})", packed.len()));
    }
    Ok(())
}

fn scale_bytes(scale_dtype: u32) -> usize {
    if scale_dtype == 2 { 4 } else { 2 }
}

fn validate_scales(k_dim: usize, out_dim: usize, group_size: usize, scale_dtype: u32, scales: &CudaSlice<u8>) -> Result<(), String> {
    if group_size == 0 || !k_dim.is_multiple_of(group_size) {
        return Err(format!("W4A16 group_size={group_size} 与 k_dim={k_dim} 不整除"));
    }
    // kernel 每线程处理一个 int32 word(8 权重),要求一个 word 不跨组 ⇒ group_size 必须是 8 的倍数。
    if !group_size.is_multiple_of(8) {
        return Err(format!("W4A16 group_size={group_size} 必须是 8 的倍数(kernel 按 int32 word 反量化)"));
    }
    let expected = out_dim.checked_mul(k_dim / group_size).and_then(|n| n.checked_mul(scale_bytes(scale_dtype))).ok_or("W4A16 scale 大小溢出")?;
    if scales.len() != expected {
        return Err(format!("W4A16 scales={}，期望 {expected}", scales.len()));
    }
    Ok(())
}

/// shared memory 字节数:scale tile + warp 间归约缓冲。
fn gemv_shared_bytes(k_dim: usize, group_size: usize) -> Result<u32, String> {
    let groups = k_dim / group_size;
    let floats = groups.checked_add(THREADS as usize / 32).ok_or("W4A16 gemv shared 溢出")?;
    Ok((floats * std::mem::size_of::<f32>()) as u32)
}

/// Decode GEMV(单 token)。input `[1, k_dim]`,weight packed `[out_dim, k_dim]` → output `[1, out_dim]`。
#[allow(clippy::too_many_arguments)]
pub fn gemv_f16(ctx: &CudaContext, input: &CudaTensor, packed: &CudaSlice<u8>, scales: &CudaSlice<u8>, out_dim: usize, group_size: usize, scale_dtype: u32) -> Result<CudaTensor, String> {
    if input.rows != 1 {
        return Err(format!("W4A16 GEMV 期望单行输入,实际 rows={}", input.rows));
    }
    let k_dim = input.cols;
    validate_packed(k_dim, out_dim, packed)?;
    validate_scales(k_dim, out_dim, group_size, scale_dtype, scales)?;
    let output = ctx.tensor_uninit(1, out_dim)?;
    let func = ctx.function("w4a16_gemv_f16")?;
    let cfg = LaunchConfig { grid_dim: (out_dim as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: gemv_shared_bytes(k_dim, group_size)? };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(packed)
            .arg(scales)
            .arg(&output.slice)
            .arg(&(k_dim as u32))
            .arg(&(out_dim as u32))
            .arg(&(group_size as u32))
            .arg(&scale_dtype)
            .launch(cfg)
            .map_err(|e| format!("launch w4a16_gemv_f16 失败: {e:?}"))?;
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
    up_packed: &CudaSlice<u8>,
    up_scales: &CudaSlice<u8>,
    out_dim: usize,
    group_size: usize,
    gate_scale_dtype: u32,
    up_scale_dtype: u32,
) -> Result<CudaTensor, String> {
    if input.rows != 1 {
        return Err(format!("W4A16 gated GEMV 期望单行输入,实际 rows={}", input.rows));
    }
    // kernel 用单一 scale_dtype 同时解码 gate/up,两侧 dtype 不同会得到静默错误的结果。
    if gate_scale_dtype != up_scale_dtype {
        return Err(format!("W4A16 gated GEMV scale dtype 不一致: gate={gate_scale_dtype} up={up_scale_dtype}"));
    }
    let scale_dtype = gate_scale_dtype;
    let k_dim = input.cols;
    validate_packed(k_dim, out_dim, gate_packed)?;
    validate_packed(k_dim, out_dim, up_packed)?;
    validate_scales(k_dim, out_dim, group_size, scale_dtype, gate_scales)?;
    validate_scales(k_dim, out_dim, group_size, scale_dtype, up_scales)?;
    let output = ctx.tensor_uninit(1, out_dim)?;
    let func = ctx.function("w4a16_gated_silu_gemv_f16")?;
    let groups = k_dim / group_size;
    let shared_floats = 2 * groups + 2 * (THREADS as usize / 32);
    let cfg = LaunchConfig { grid_dim: (out_dim as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(gate_packed)
            .arg(gate_scales)
            .arg(up_packed)
            .arg(up_scales)
            .arg(&output.slice)
            .arg(&(k_dim as u32))
            .arg(&(out_dim as u32))
            .arg(&(group_size as u32))
            .arg(&scale_dtype)
            .launch(cfg)
            .map_err(|e| format!("launch w4a16_gated_silu_gemv_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 整矩阵反量化为 f16 行优先 `[out_dim, k_dim]`(prefill 喂 cuBLAS)。
#[allow(clippy::too_many_arguments)]
pub fn dequant_f16(ctx: &CudaContext, packed: &CudaSlice<u8>, scales: &CudaSlice<u8>, out_dim: usize, k_dim: usize, group_size: usize, scale_dtype: u32) -> Result<CudaSlice<half::f16>, String> {
    validate_packed(k_dim, out_dim, packed)?;
    validate_scales(k_dim, out_dim, group_size, scale_dtype, scales)?;
    let elements = out_dim.checked_mul(k_dim).ok_or("W4A16 dequant 输出溢出")?;
    let output = ctx.buffer_uninit::<half::f16>(elements)?;
    let func = ctx.function("w4a16_dequant_f16")?;
    // 每线程处理一个 int32 word(8 元素);grid 按 word 数覆盖,block 用 THREADS。
    let words = out_dim.checked_mul(k_dim / 8).ok_or("W4A16 dequant word 数溢出")?;
    let blocks = words.min(THREADS as usize * 1024).div_ceil(THREADS as usize).max(1) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(packed)
            .arg(scales)
            .arg(&output)
            .arg(&(k_dim as u32))
            .arg(&(out_dim as u32))
            .arg(&(group_size as u32))
            .arg(&scale_dtype)
            .launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| format!("launch w4a16_dequant_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use super::*;
    use crate::weight::format::quantization::{ScaleDType, W4A16Matrix};

    fn check_close(name: &str, actual: &[f32], expect: &[f32]) {
        let atol = 1e-2f32;
        let rtol = 1e-2f32;
        for (i, (a, e)) in actual.iter().zip(expect).enumerate() {
            let err = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(err <= tol, "{name}[{i}] 偏差 {err:.4} 超阈值 {tol:.4}(actual={a:.4} expect={e:.4})");
        }
    }

    // 构造确定性 W4A16 矩阵:权重按 (row,col) 生成有符号 int4,scale 按组设定。
    fn build_matrix(rows: usize, cols: usize, group_size: usize, dtype: ScaleDType) -> W4A16Matrix {
        let mut packed = vec![0u8; rows * (cols.div_ceil(8)) * 4];
        let mut scales = vec![0u8; rows * (cols / group_size) * dtype.bytes()];
        let groups = cols / group_size;
        for row in 0..rows {
            for col in 0..cols {
                // 有符号码 ∈ [-8,7],文件存 code+8 ∈ [0,15]。
                let code: i32 = (((row * 31 + col * 7) % 15) as i32) - 7;
                let stored = (code + 8) as u32 & 0xF;
                let word_off = (row * (cols / 8) + col / 8) * 4;
                let word = u32::from_le_bytes(packed[word_off..word_off + 4].try_into().unwrap()) | (stored << ((col % 8) * 4));
                packed[word_off..word_off + 4].copy_from_slice(&word.to_le_bytes());
            }
            for g in 0..groups {
                let scale = 0.125 * ((g + 1) as f32);
                let off = (row * groups + g) * dtype.bytes();
                match dtype {
                    ScaleDType::Bf16 => {
                        let b = half::bf16::from_f32(scale).to_le_bytes();
                        scales[off..off + 2].copy_from_slice(&b);
                    }
                    ScaleDType::F16 => {
                        let b = half::f16::from_f32(scale).to_le_bytes();
                        scales[off..off + 2].copy_from_slice(&b);
                    }
                    ScaleDType::F32 => scales[off..off + 4].copy_from_slice(&scale.to_le_bytes()),
                }
            }
        }
        W4A16Matrix::new(packed, scales, dtype, group_size, rows, cols).unwrap()
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

            // 上传 packed(I32 字节流,kernel 内 reinterpret 为 uint),scales 为 u8。
            let packed_gpu = ctx.stream().clone_htod::<u8, _>(matrix.packed()).unwrap();
            let scales_gpu = ctx.stream().clone_htod::<u8, _>(matrix.scales()).unwrap();

            let out_gpu = gemv_f16(&ctx, &input_gpu, &packed_gpu, &scales_gpu, rows, group, dtype.metal_code()).unwrap();
            let out = ctx.tensor_to_f32(&out_gpu).unwrap();

            // CPU 参考:output[n] = Σ_k input[k] * reference[n*cols+k]。
            let mut expect = vec![0.0f32; rows];
            for n in 0..rows {
                for k in 0..cols {
                    expect[n] += input[k] * reference[n * cols + k];
                }
            }
            check_close(&format!("w4a16_gemv_{dtype:?}"), &out, &expect);
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
        let f16_buf = dequant_f16(&ctx, &packed_gpu, &scales_gpu, rows, cols, group, ScaleDType::F16.metal_code()).unwrap();
        let host = ctx.stream().clone_dtoh::<half::f16, _>(&f16_buf).unwrap();
        let out: Vec<f32> = host.iter().map(|v| v.to_f32()).collect();
        check_close("w4a16_dequant", &out, &reference);
    }

    #[test]
    fn gated_silu_matches_reference() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        let rows = 64;
        let cols = 64;
        let group = 32;
        let gate = build_matrix(rows, cols, group, ScaleDType::F16);
        let gate_ref = gate.decode().unwrap();

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.02 - 0.6).collect();
        let input_gpu = ctx.tensor_from_f32(&input, 1, cols).unwrap();

        let upload = |m: &W4A16Matrix| {
            let p = ctx.stream().clone_htod::<u8, _>(m.packed()).unwrap();
            let s = ctx.stream().clone_htod::<u8, _>(m.scales()).unwrap();
            (p, s)
        };
        let (gp, gs) = upload(&gate);
        // gated kernel 要求 gate/up 同 dtype;这里用两组独立参考对比,故都用 F16。
        let up_f16 = build_matrix(rows, cols, group, ScaleDType::F16);
        let (up_p, up_s) = upload(&up_f16);
        let up_ref = up_f16.decode().unwrap();

        let out_gpu = gated_silu_gemv_f16(&ctx, &input_gpu, &gp, &gs, &up_p, &up_s, rows, group, ScaleDType::F16.metal_code(), ScaleDType::F16.metal_code()).unwrap();
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
        check_close("w4a16_gated_silu", &out, &expect);
    }
}
