//! compressed-tensors W8A16 group-wise 8-bit 设备内反量化算子。
//!
//! 服务 `LmHeadQuantization::Q8g128` 的 CUDA 路径:LM head 在 host 量化为
//! 每 128 列一组 scale 的对称 INT8(packed 存 `code + 128` 的 u8),保持 1 字节/元素
//! 常驻显存(相比 f16 减半),`linear()` 内即时反量化,避免 fallback 展开 1.5GB f16。
//!
//! 数据布局与 `weight/codec/groupwise.rs::decode_w8a16_matrix` 完全一致:
//! - `packed` 行优先 `[out_dim, k_dim]` 的 u8(行尾按 4 元素对齐到 u32 word,k_dim
//!   按 group 对齐时无填充),解码值 `w = (int8(byte) - 128) * scale`。
//! - `scales` 行优先 `[out_dim, groups]`(`groups = k_dim/group_size`),dtype BF16/F16/F32。
//!
//! 路径(对称 `w4a16.rs`):
//! - decode(input.rows=1):`gemv_f16`(lm_head 逐 token GEMV)。
//! - prefill(input.rows>1):`dequant_f16` 整矩阵反量化为 f16 再喂 cuBLAS hgemm。

use cudarc::driver::safe::CudaSlice;

use super::{CudaContext, CudaSliceF16, CudaTensor, LaunchConfig, PushKernelArg, THREADS};

// kernels: w8a16_gemv_f16, w8a16_dequant_f16
// private helpers: w8a16_scale
pub const SHADERS: &str = r#"
// 读取 per-group scale。dtype: 0=BF16, 1=F16, 2=F32(对称 ScaleDType::metal_code)。
__device__ __forceinline__ float w8a16_scale(const unsigned char *params, unsigned long long index, unsigned int dtype)
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
// 每线程每次处理一个 u32 word(4 个 INT8):word 只读一次、4 个字节全用;input 用
// int2(8B)一次取 4 个 half(word 起始 8B 对齐);group scale 先 tile 进 shared 复用。
extern "C" __global__ void w8a16_gemv_f16(
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
    const unsigned int groups = k_dim / group_size;
    float *scale_tile = shared;                       // groups 个 float
    float *partial = shared + groups;                 // warp_count 个 float

    const unsigned int packed_cols = k_dim / 4u;
    const unsigned long long packed_row = (unsigned long long)n * packed_cols;
    const unsigned long long scale_row = (unsigned long long)n * groups;

    // 1. 把本行的 group scale 装进 shared(每线程跨步搬运)。
    for (unsigned int g = tid; g < groups; g += blockDim.x) {
        scale_tile[g] = w8a16_scale(scales, scale_row + g, scale_dtype);
    }
    __syncthreads();

    // 2. 跨步累加 dot(input, weight_row)。group_size 是 8 的倍数 ⇒ 一个 word(4 元素)
    //    完全落在同一组内,scale 提到累加外只取一次。
    float acc = 0.0f;
    const unsigned int words_per_group = group_size / 4u;
    for (unsigned int w = tid; w < packed_cols; w += blockDim.x) {
        unsigned int word = packed[packed_row + w];
        float s = scale_tile[w / words_per_group];
        int2 iv = reinterpret_cast<const int2 *>(input + w * 4u)[0];
        unsigned int ip[2] = { iv.x, iv.y };
        float local = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x = __half2float(__ushort_as_half((unsigned short)((ip[j >> 1] >> ((j & 1) * 16u)) & 0xFFFFu)));
            int code = (int)((word >> (j * 8u)) & 0xFFu) - 128;
            local += x * ((float)code * s);
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

// 整矩阵反量化为 f16(prefill 用):每线程负责一个 u32 word(4 元素),输出按 int2
// (4 个 f16)合并写一次;group scale 直接解码不进 shared。
extern "C" __global__ void w8a16_dequant_f16(
    const unsigned int * __restrict__ packed,
    const unsigned char * __restrict__ scales,
    __half * __restrict__ output,
    const unsigned int k_dim,
    const unsigned int out_dim,
    const unsigned int group_size,
    const unsigned int scale_dtype)
{
    const unsigned int packed_cols = k_dim / 4u;
    const unsigned int groups = k_dim / group_size;
    const unsigned int total_words = out_dim * packed_cols;
    for (unsigned int w = blockIdx.x * blockDim.x + threadIdx.x; w < total_words; w += gridDim.x * blockDim.x) {
        unsigned int n = w / packed_cols;
        unsigned int lw = w - n * packed_cols;
        unsigned int word = packed[(unsigned long long)n * packed_cols + lw];
        float s = w8a16_scale(scales, (unsigned long long)n * groups + lw / (group_size / 4u), scale_dtype);
        __half2 pair[2];
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int c0 = (int)((word >> ((j * 2) * 8u)) & 0xFFu) - 128;
            int c1 = (int)((word >> ((j * 2 + 1) * 8u)) & 0xFFu) - 128;
            pair[j] = __floats2half2_rn((float)c0 * s, (float)c1 * s);
        }
        reinterpret_cast<int2 *>(output + ((unsigned long long)n * k_dim + lw * 4u))[0] = reinterpret_cast<int2 &>(pair);
    }
}
"#;

const WORD_ELEMENTS: usize = 4;

// packed 以 u8 字节流上传(`CudaSlice<u8>`),kernel 内 reinterpret 为 `const unsigned int*`。
// k_dim 按 4 对齐保证每行字节长度是 4 的倍数(group 对齐蕴含)。
fn validate_packed(k_dim: usize, out_dim: usize, packed: &CudaSlice<u8>) -> Result<(), String> {
    if !k_dim.is_multiple_of(WORD_ELEMENTS) {
        return Err(format!("W8A16 k_dim={k_dim} 必须对齐 {WORD_ELEMENTS}"));
    }
    let expected = out_dim.checked_mul(k_dim).ok_or("W8A16 packed 大小溢出")?;
    if packed.len() != expected {
        return Err(format!("W8A16 packed bytes={}，期望 {expected}(out_dim={out_dim}, k_dim={k_dim})", packed.len()));
    }
    Ok(())
}

fn scale_bytes(scale_dtype: u32) -> usize {
    if scale_dtype == 2 { 4 } else { 2 }
}

fn validate_params(k_dim: usize, out_dim: usize, group_size: usize, scale_dtype: u32, params: &CudaSlice<u8>, name: &str) -> Result<(), String> {
    if group_size == 0 || !k_dim.is_multiple_of(group_size) {
        return Err(format!("W8A16 {name} group_size={group_size} 与 k_dim={k_dim} 不整除"));
    }
    // kernel 每线程处理一个 u32 word(4 权重),要求一个 word 不跨组 ⇒ group_size 必须是 4 的倍数。
    if !group_size.is_multiple_of(WORD_ELEMENTS) {
        return Err(format!("W8A16 {name} group_size={group_size} 必须是 {WORD_ELEMENTS} 的倍数(kernel 按 u32 word 反量化)"));
    }
    let expected = out_dim.checked_mul(k_dim / group_size).and_then(|n| n.checked_mul(scale_bytes(scale_dtype))).ok_or_else(|| format!("W8A16 {name} 参数大小溢出"))?;
    if params.len() != expected {
        return Err(format!("W8A16 {name} bytes={}，期望 {expected}", params.len()));
    }
    Ok(())
}

/// Decode GEMV(单 token)。input `[1, k_dim]`,weight packed `[out_dim, k_dim]` → output `[1, out_dim]`。
pub fn gemv_f16(ctx: &CudaContext, input: &CudaTensor, packed: &CudaSlice<u8>, scales: &CudaSlice<u8>, out_dim: usize, group_size: usize, scale_dtype: u32) -> Result<CudaTensor, String> {
    if input.rows != 1 {
        return Err(format!("W8A16 GEMV 期望单行输入,实际 rows={}", input.rows));
    }
    let k_dim = input.cols;
    validate_packed(k_dim, out_dim, packed)?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, scales, "scales")?;
    let output = ctx.tensor_uninit(1, out_dim)?;
    let func = ctx.function("w8a16_gemv_f16")?;
    // shared: scale_tile(groups 个 float) + warp 间归约缓冲。
    let groups = k_dim / group_size;
    let shared_floats = groups + THREADS as usize / 32;
    let cfg = LaunchConfig { grid_dim: (out_dim as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
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
            .map_err(|e| format!("launch w8a16_gemv_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 整矩阵反量化为 f16 行优先 `[out_dim, k_dim]`(prefill 喂 cuBLAS)。
pub fn dequant_f16(ctx: &CudaContext, packed: &CudaSlice<u8>, scales: &CudaSlice<u8>, out_dim: usize, k_dim: usize, group_size: usize, scale_dtype: u32) -> Result<CudaSliceF16, String> {
    validate_packed(k_dim, out_dim, packed)?;
    validate_params(k_dim, out_dim, group_size, scale_dtype, scales, "scales")?;
    let elements = out_dim.checked_mul(k_dim).ok_or("W8A16 dequant 输出溢出")?;
    let output = ctx.buffer_uninit::<half::f16>(elements)?;
    let func = ctx.function("w8a16_dequant_f16")?;
    // 每线程处理一个 u32 word(4 元素);grid 按 word 数覆盖,block 用 THREADS。
    let words = out_dim.checked_mul(k_dim / WORD_ELEMENTS).ok_or("W8A16 dequant word 数溢出")?;
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
            .map_err(|e| format!("launch w8a16_dequant_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use super::*;
    use crate::weight::format::quantization::{ScaleDType, W8A16Matrix};

    fn check_close(name: &str, actual: &[f32], expect: &[f32]) {
        let atol = 1e-2f32;
        let rtol = 1e-2f32;
        for (i, (a, e)) in actual.iter().zip(expect).enumerate() {
            let err = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(err <= tol, "{name}[{i}] 偏差 {err:.4} 超阈值 {tol:.4}(actual={a:.4} expect={e:.4})");
        }
    }

    // 构造确定性 W8A16 矩阵:code 字节按 (row,col) 生成(存储态 +128,解码 (byte-128)×scale),
    // scale 按组设定。解码公式: w = (byte - 128) * scale。
    fn build_matrix(rows: usize, cols: usize, group_size: usize, dtype: ScaleDType) -> W8A16Matrix {
        let mut packed = vec![0u8; rows * cols];
        let mut scales = vec![0u8; rows * (cols / group_size) * dtype.bytes()];
        let groups = cols / group_size;
        for row in 0..rows {
            for col in 0..cols {
                packed[row * cols + col] = ((row * 7 + col * 13) % 256) as u8;
            }
            for g in 0..groups {
                let scale = 0.01 * ((g + 1) as f32);
                let off = (row * groups + g) * dtype.bytes();
                match dtype {
                    ScaleDType::Bf16 => scales[off..off + 2].copy_from_slice(&half::bf16::from_f32(scale).to_le_bytes()),
                    ScaleDType::F16 => scales[off..off + 2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes()),
                    ScaleDType::F32 => scales[off..off + 4].copy_from_slice(&scale.to_le_bytes()),
                }
            }
        }
        W8A16Matrix::new(packed, scales, dtype, group_size, rows, cols).unwrap()
    }

    #[test]
    fn gemv_matches_reference() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        for dtype in [ScaleDType::Bf16, ScaleDType::F16, ScaleDType::F32] {
            let rows = 64;
            let cols = 128;
            let group = 64;
            let matrix = build_matrix(rows, cols, group, dtype);
            let reference = matrix.decode().unwrap();

            let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.6).collect();
            let input_gpu = ctx.tensor_from_f32(&input, 1, cols).unwrap();

            let packed_gpu = ctx.stream().clone_htod::<u8, _>(matrix.packed()).unwrap();
            let scales_gpu = ctx.stream().clone_htod::<u8, _>(matrix.scales()).unwrap();
            let out_gpu = gemv_f16(&ctx, &input_gpu, &packed_gpu, &scales_gpu, rows, group, dtype.metal_code()).unwrap();
            let out = ctx.tensor_to_f32(&out_gpu).unwrap();

            let mut expect = vec![0.0f32; rows];
            for n in 0..rows {
                for k in 0..cols {
                    expect[n] += input[k] * reference[n * cols + k];
                }
            }
            check_close(&format!("w8a16_gemv_{dtype:?}"), &out, &expect);
        }
    }

    #[test]
    fn dequant_matches_reference() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        let rows = 48;
        let cols = 128;
        let group = 128;
        let matrix = build_matrix(rows, cols, group, ScaleDType::Bf16);
        let reference = matrix.decode().unwrap();

        let packed_gpu = ctx.stream().clone_htod::<u8, _>(matrix.packed()).unwrap();
        let scales_gpu = ctx.stream().clone_htod::<u8, _>(matrix.scales()).unwrap();
        let f16_buf = dequant_f16(&ctx, &packed_gpu, &scales_gpu, rows, cols, group, ScaleDType::Bf16.metal_code()).unwrap();
        let host = ctx.stream().clone_dtoh::<half::f16, _>(&f16_buf).unwrap();
        let out: Vec<f32> = host.iter().map(|v| v.to_f32()).collect();
        check_close("w8a16_dequant", &out, &reference);
    }
}
