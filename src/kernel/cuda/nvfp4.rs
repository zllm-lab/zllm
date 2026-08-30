//! NVIDIA ModelOpt NVFP4(block=16 E2M1 + E4M3 block scale + 全局 F32 scale)设备内反量化算子。
//!
//! Qwen3-14B NVFP4 checkpoint 的 Linear 权重保持 packed u8(每字节 2 个 E2M1 code)
//! 与 per-block E4M3 scale 常驻,kernel 内即时反量化,避免 host 展开为 f16(约 4× 显存 + HTOD)。
//!
//! 数据布局与 `kernel/cpu/nvfp4.rs`(CPU oracle)完全一致:
//! - `codes` 行优先 `[out_dim, k_dim/2]` 的 u8;第 `n,k` 个权重字节
//!   `codes[n*(k_dim/2) + k/2]`,偶数 k 取低 nibble、奇数 k 取高 nibble。
//! - `scales` 行优先 `[out_dim, k_dim/16]` 的 E4M3 字节,每 16 列一个 block scale。
//! - 解码值 `w = e2m1(code) * e4m3(block_scale) * global_scale`。
//!
//! 路径(对称 `mlx_affine.rs` / `w4a16.rs`):
//! - decode(input.rows=1):`gemv_f16` / 融合 gate+up 的 `gated_silu_gemv_f16`。
//! - prefill(input.rows>1):`dequant_f16` 整矩阵反量化为 f16 再喂 cuBLAS hgemm。

use cudarc::driver::safe::CudaSlice;

use super::{CudaContext, CudaSliceF16, CudaTensor, LaunchConfig, PushKernelArg, THREADS};

// kernels: nvfp4_gemv_f16, nvfp4_gated_silu_gemv_f16, nvfp4_dequant_f16
// private helpers: nvfp4_e2m1, nvfp4_e4m3
pub const SHADERS: &str = r#"
// E2M1 code(低 4 位)解码:3 位数值 LUT + 符号位,与 kernel/cpu/nvfp4.rs::decode_f4_e2m1 一致。
__device__ __forceinline__ float nvfp4_e2m1(unsigned int code)
{
    unsigned int value = code & 0x07u;
    unsigned int bits;
    if (value == 0u) {
        bits = 0u;
    } else if (value == 1u) {
        bits = 0x3F000000u;
    } else {
        bits = (((value >> 1) + 126u) << 23) | ((value & 1u) << 22);
    }
    bits |= (code & 0x08u) << 28;
    return __uint_as_float(bits);
}

// E4M3 字节软件解码,与 weight/codec/fp8.rs::decode_f8_e4m3、kernel/cuda/fp8.rs 逐位一致
// (sm_86 无 fp8 硬件转换指令)。
__device__ __forceinline__ float nvfp4_e4m3(unsigned int bits8)
{
    if (bits8 == 0x7Fu || bits8 == 0xFFu) {
        return nanf("");
    }
    unsigned int exponent = (bits8 >> 3) & 0x0Fu;
    unsigned int mantissa = bits8 & 0x07u;
    unsigned int bits = (exponent == 0u) ? (mantissa << 14) : (((exponent + 120u) << 23) | (mantissa << 20));
    bits |= (bits8 & 0x80u) << 24;
    return __uint_as_float(bits);
}

// Decode GEMV:单 token 输入。每个 block 算一个输出行 n,内部跨步累加后 warp/block 归约。
// 权重以 packed u32 即时反量化(word 只读一次、8 个 nibble 全用),block scale 先 tile 进
// shared(已乘 global_scale,主循环只做 LUT × scale)。
extern "C" __global__ void nvfp4_gemv_f16(
    const __half * __restrict__ input,
    const unsigned int * __restrict__ codes,
    const unsigned char * __restrict__ scales,
    __half * __restrict__ output,
    const float global_scale,
    const unsigned int k_dim,
    const unsigned int out_dim)
{
    const unsigned int n = blockIdx.x;
    if (n >= out_dim) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    extern __shared__ float shared[];
    const unsigned int blocks = k_dim / 16u;           // 每 16 列一个 scale block
    float *lut = shared;                               // 16 个 e2m1 解码值(动态索引的线程局部数组会落 local memory)
    float *scale_tile = shared + 16;                   // blocks 个 float(已含 global_scale)
    float *partial = shared + 16 + blocks;             // warp_count 个 float

    const unsigned int packed_cols = k_dim / 8u;       // 每行 u32 word 数(8 code/word)
    const unsigned long long code_row = (unsigned long long)n * packed_cols;
    const unsigned long long scale_row = (unsigned long long)n * blocks;

    // 1. shared LUT + 本行 block scale 解码进 shared(E4M3 × global_scale)。
    if (tid < 16u) {
        lut[tid] = nvfp4_e2m1(tid);
    }
    for (unsigned int b = tid; b < blocks; b += blockDim.x) {
        scale_tile[b] = nvfp4_e4m3(scales[scale_row + b]) * global_scale;
    }
    __syncthreads();

    // 2. 跨步累加 dot(input, weight_row)。每线程每次处理一个 u32 word(8 权重):
    //    word 只读一次、8 个 nibble 全用;一个 word(8 元素)完全落在同一个 block(16 元素)
    //    内,scale 只取一次;input 用 int4(16B)一次取 8 个 half。
    // 2. 每线程一次 int4 读 4 个 u32 word(32 code,恰为 2 个 scale block);k_dim 按 32
    //    对齐保证 word 四元组不跨 block、地址 16B 对齐;输入按 __half2 成对转换。
    const unsigned int quads = packed_cols >> 2u;
    const int4 *code_vec = reinterpret_cast<const int4 *>(codes + code_row);
    float acc = 0.0f;
    for (unsigned int q = tid; q < quads; q += blockDim.x) {
        int4 cv = code_vec[q];
        unsigned int words[4] = { cv.x, cv.y, cv.z, cv.w };
        float s0 = scale_tile[q << 1u];
        float s1 = scale_tile[(q << 1u) + 1u];
        float local = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned int word = words[i];
            float s = (i < 2) ? s0 : s1;
            __half2 pair[4];
            reinterpret_cast<int4 &>(pair) = reinterpret_cast<const int4 *>(input + ((q << 2u) + i) * 8u)[0];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                float2 xy = __half22float2(pair[p]);
                unsigned int base = p * 8u;
                local += xy.x * (lut[(word >> base) & 0xFu] * s);
                local += xy.y * (lut[(word >> (base + 4u)) & 0xFu] * s);
            }
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
// 两个权重行共用同一段输入(int4 一次取),各自 word 只读一次全用 8 nibble。
extern "C" __global__ void nvfp4_gated_silu_gemv_f16(
    const __half * __restrict__ input,
    const unsigned int * __restrict__ gate_codes,
    const unsigned char * __restrict__ gate_scales,
    const unsigned int * __restrict__ up_codes,
    const unsigned char * __restrict__ up_scales,
    __half * __restrict__ output,
    const float gate_global_scale,
    const float up_global_scale,
    const unsigned int k_dim,
    const unsigned int out_dim)
{
    const unsigned int n = blockIdx.x;
    if (n >= out_dim) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    extern __shared__ float shared[];
    const unsigned int blocks = k_dim / 16u;
    float *lut = shared;                               // 16 个 e2m1 解码值
    float *gate_scale_tile = shared + 16;              // blocks
    float *up_scale_tile = shared + 16 + blocks;       // blocks
    float *partial = shared + 16 + 2u * blocks;        // 2 * warp_count(gate, up)

    const unsigned int packed_cols = k_dim / 8u;
    const unsigned long long code_row = (unsigned long long)n * packed_cols;
    const unsigned long long scale_row = (unsigned long long)n * blocks;

    if (tid < 16u) {
        lut[tid] = nvfp4_e2m1(tid);
    }
    for (unsigned int b = tid; b < blocks; b += blockDim.x) {
        gate_scale_tile[b] = nvfp4_e4m3(gate_scales[scale_row + b]) * gate_global_scale;
        up_scale_tile[b] = nvfp4_e4m3(up_scales[scale_row + b]) * up_global_scale;
    }
    __syncthreads();

    const unsigned int quads = packed_cols >> 2u;
    const int4 *gate_vec = reinterpret_cast<const int4 *>(gate_codes + code_row);
    const int4 *up_vec = reinterpret_cast<const int4 *>(up_codes + code_row);
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (unsigned int q = tid; q < quads; q += blockDim.x) {
        int4 gv = gate_vec[q];
        int4 uv = up_vec[q];
        unsigned int gwords[4] = { gv.x, gv.y, gv.z, gv.w };
        unsigned int uwords[4] = { uv.x, uv.y, uv.z, uv.w };
        float gs0 = gate_scale_tile[q << 1u];
        float gs1 = gate_scale_tile[(q << 1u) + 1u];
        float us0 = up_scale_tile[q << 1u];
        float us1 = up_scale_tile[(q << 1u) + 1u];
        float g_local = 0.0f;
        float u_local = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float gs = (i < 2) ? gs0 : gs1;
            float us = (i < 2) ? us0 : us1;
            unsigned int gw = gwords[i];
            unsigned int uw = uwords[i];
            __half2 pair[4];
            reinterpret_cast<int4 &>(pair) = reinterpret_cast<const int4 *>(input + ((q << 2u) + i) * 8u)[0];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                float2 xy = __half22float2(pair[p]);
                unsigned int base = p * 8u;
                g_local += xy.x * (lut[(gw >> base) & 0xFu] * gs);
                g_local += xy.y * (lut[(gw >> (base + 4u)) & 0xFu] * gs);
                u_local += xy.x * (lut[(uw >> base) & 0xFu] * us);
                u_local += xy.y * (lut[(uw >> (base + 4u)) & 0xFu] * us);
            }
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

// 整矩阵反量化为 f16(prefill 用):每线程负责一个 u32 word(8 元素),word 只读一次、
// 8 个 nibble 全用,输出按 int4(8 个 f16)合并写一次;block scale 直接解码不进 shared。
extern "C" __global__ void nvfp4_dequant_f16(
    const unsigned int * __restrict__ codes,
    const unsigned char * __restrict__ scales,
    __half * __restrict__ output,
    const float global_scale,
    const unsigned int k_dim,
    const unsigned int out_dim)
{
    const unsigned int packed_cols = k_dim / 8u;
    const unsigned int blocks = k_dim / 16u;
    const unsigned int quads_per_row = packed_cols >> 2u;
    const unsigned int total_quads = out_dim * quads_per_row;
    for (unsigned int q = blockIdx.x * blockDim.x + threadIdx.x; q < total_quads; q += gridDim.x * blockDim.x) {
        unsigned int n = q / quads_per_row;
        unsigned int lq = q - n * quads_per_row;
        int4 cv = reinterpret_cast<const int4 *>(codes + (unsigned long long)n * packed_cols)[lq];
        unsigned int words[4] = { cv.x, cv.y, cv.z, cv.w };
        const unsigned long long scale_row = (unsigned long long)n * blocks;
        const unsigned long long out_base = (unsigned long long)n * k_dim + lq * 32u;
        float s0 = nvfp4_e4m3(scales[scale_row + (lq << 1u)]) * global_scale;
        float s1 = nvfp4_e4m3(scales[scale_row + (lq << 1u) + 1u]) * global_scale;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned int word = words[i];
            float s = (i < 2) ? s0 : s1;
            __half2 pair[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                unsigned int c0 = (word >> (((j * 2) >> 1) * 8u + ((j * 2) & 1u) * 4u)) & 0xFu;
                unsigned int c1 = (word >> (((j * 2 + 1) >> 1) * 8u + ((j * 2 + 1) & 1u) * 4u)) & 0xFu;
                pair[j] = __floats2half2_rn(nvfp4_e2m1(c0) * s, nvfp4_e2m1(c1) * s);
            }
            reinterpret_cast<int4 *>(output + out_base + i * 8u)[0] = reinterpret_cast<int4 &>(pair);
        }
    }
}
"#;

const BLOCK: usize = 16;

// GEMV 用 128 线程(4 warp):k_dim=5120 时每行 640 word,256 线程每线程只摊 ~2.5 word,
// warp/两级归约开销占比过高;128 线程摊到 ~5 word/线程,且每 SM 可驻留 12 个 block 维持满占用。
// kernel 按 blockDim 泛化,此处只影响 launch 配置。
const GEMV_THREADS: u32 = 128;

// codes 以 u8 字节流上传(`CudaSlice<u8>`),kernel 内 reinterpret 为 `const unsigned int*`。
// k_dim 按 16 对齐保证每行字节长度是 4 的倍数,word 读取安全。
fn validate_codes(k_dim: usize, out_dim: usize, codes: &CudaSlice<u8>) -> Result<(), String> {
    if !k_dim.is_multiple_of(32) {
        return Err(format!("NVFP4 k_dim={k_dim} 必须对齐 32(kernel 按 int4 读 4 word,恰为 2 个 block)"));
    }
    let expected = out_dim.checked_mul(k_dim / 2).ok_or("NVFP4 codes 大小溢出")?;
    if codes.len() != expected {
        return Err(format!("NVFP4 codes bytes={}，期望 {expected}(out_dim={out_dim}, k_dim={k_dim})", codes.len()));
    }
    Ok(())
}

fn validate_scales(k_dim: usize, out_dim: usize, scales: &CudaSlice<u8>) -> Result<(), String> {
    let expected = out_dim.checked_mul(k_dim / BLOCK).ok_or("NVFP4 scales 大小溢出")?;
    if scales.len() != expected {
        return Err(format!("NVFP4 scales bytes={}，期望 {expected}", scales.len()));
    }
    Ok(())
}

fn validate_global_scale(name: &str, scale: f32) -> Result<(), String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(format!("NVFP4 {name} global_scale={scale} 无效"));
    }
    Ok(())
}

/// Decode GEMV(单 token)。input `[1, k_dim]`,codes `[out_dim, k_dim/2]` → output `[1, out_dim]`。
pub fn gemv_f16(ctx: &CudaContext, input: &CudaTensor, codes: &CudaSlice<u8>, scales: &CudaSlice<u8>, global_scale: f32, out_dim: usize) -> Result<CudaTensor, String> {
    if input.rows != 1 {
        return Err(format!("NVFP4 GEMV 期望单行输入,实际 rows={}", input.rows));
    }
    let k_dim = input.cols;
    validate_codes(k_dim, out_dim, codes)?;
    validate_scales(k_dim, out_dim, scales)?;
    validate_global_scale("gemv", global_scale)?;
    let output = ctx.tensor_uninit(1, out_dim)?;
    let func = ctx.function("nvfp4_gemv_f16")?;
    // shared: scale_tile(blocks 个 float) + warp 间归约缓冲。
    let blocks = k_dim / BLOCK;
    let shared_floats = 16 + blocks + GEMV_THREADS as usize / 32;
    let cfg = LaunchConfig { grid_dim: (out_dim as u32, 1, 1), block_dim: (GEMV_THREADS, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream().launch_builder(&func).arg(&input.slice).arg(codes).arg(scales).arg(&output.slice).arg(&global_scale).arg(&(k_dim as u32)).arg(&(out_dim as u32)).launch(cfg).map_err(|e| format!("launch nvfp4_gemv_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// Decode 融合 gate/up GEMV + SiLU。两个 NVFP4 权重同形 `[out_dim, k_dim]` → output `[1, out_dim]`。
#[allow(clippy::too_many_arguments)]
pub fn gated_silu_gemv_f16(
    ctx: &CudaContext,
    input: &CudaTensor,
    gate_codes: &CudaSlice<u8>,
    gate_scales: &CudaSlice<u8>,
    up_codes: &CudaSlice<u8>,
    up_scales: &CudaSlice<u8>,
    gate_global_scale: f32,
    up_global_scale: f32,
    out_dim: usize,
) -> Result<CudaTensor, String> {
    if input.rows != 1 {
        return Err(format!("NVFP4 gated GEMV 期望单行输入,实际 rows={}", input.rows));
    }
    let k_dim = input.cols;
    validate_codes(k_dim, out_dim, gate_codes)?;
    validate_scales(k_dim, out_dim, gate_scales)?;
    validate_codes(k_dim, out_dim, up_codes)?;
    validate_scales(k_dim, out_dim, up_scales)?;
    validate_global_scale("gate", gate_global_scale)?;
    validate_global_scale("up", up_global_scale)?;
    let output = ctx.tensor_uninit(1, out_dim)?;
    let func = ctx.function("nvfp4_gated_silu_gemv_f16")?;
    let blocks = k_dim / BLOCK;
    let shared_floats = 16 + 2 * blocks + 2 * (GEMV_THREADS as usize / 32);
    let cfg = LaunchConfig { grid_dim: (out_dim as u32, 1, 1), block_dim: (GEMV_THREADS, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(gate_codes)
            .arg(gate_scales)
            .arg(up_codes)
            .arg(up_scales)
            .arg(&output.slice)
            .arg(&gate_global_scale)
            .arg(&up_global_scale)
            .arg(&(k_dim as u32))
            .arg(&(out_dim as u32))
            .launch(cfg)
            .map_err(|e| format!("launch nvfp4_gated_silu_gemv_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

/// 整矩阵反量化为 f16 行优先 `[out_dim, k_dim]`(prefill 喂 cuBLAS)。
pub fn dequant_f16(ctx: &CudaContext, codes: &CudaSlice<u8>, scales: &CudaSlice<u8>, global_scale: f32, out_dim: usize, k_dim: usize) -> Result<CudaSliceF16, String> {
    validate_codes(k_dim, out_dim, codes)?;
    validate_scales(k_dim, out_dim, scales)?;
    validate_global_scale("dequant", global_scale)?;
    let elements = out_dim.checked_mul(k_dim).ok_or("NVFP4 dequant 输出溢出")?;
    let output = ctx.buffer_uninit::<half::f16>(elements)?;
    let func = ctx.function("nvfp4_dequant_f16")?;
    // 每线程处理一个 u32 word(8 元素);grid 按 word 数覆盖,block 用 THREADS。
    let words = out_dim.checked_mul(k_dim / 8).ok_or("NVFP4 dequant word 数溢出")?;
    let blocks = words.min(THREADS as usize * 1024).div_ceil(THREADS as usize).max(1) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(codes)
            .arg(scales)
            .arg(&output)
            .arg(&global_scale)
            .arg(&(k_dim as u32))
            .arg(&(out_dim as u32))
            .launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| format!("launch nvfp4_dequant_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use super::*;
    use crate::weight::format::nvfp4::Nvfp4Matrix;

    fn check_close(name: &str, actual: &[f32], expect: &[f32]) {
        let atol = 1e-2f32;
        let rtol = 1e-2f32;
        for (i, (a, e)) in actual.iter().zip(expect).enumerate() {
            let err = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(err <= tol, "{name}[{i}] 偏差 {err:.4} 超阈值 {tol:.4}(actual={a:.4} expect={e:.4})");
        }
    }

    // 构造确定性 NVFP4 矩阵:nibble 按 (row,col) 生成(覆盖正负与全部量级),
    // block scale 在 {0.25, 0.5, 1.0, 2.0}(E4M3 精确可表示)间轮转。
    fn build_matrix(rows: usize, cols: usize, global_scale: f32) -> Nvfp4Matrix {
        let mut codes = vec![0u8; rows * cols / 2];
        let mut scales = vec![0u8; rows * cols / BLOCK];
        let scale_bytes = [0x28u8, 0x30, 0x38, 0x40];
        for row in 0..rows {
            for col in 0..cols {
                let nibble: u8 = ((row * 13 + col * 5) % 16) as u8;
                let offset = row * (cols / 2) + col / 2;
                if col % 2 == 0 {
                    codes[offset] |= nibble & 0x0f;
                } else {
                    codes[offset] |= nibble << 4;
                }
            }
            for block in 0..cols / BLOCK {
                scales[row * (cols / BLOCK) + block] = scale_bytes[block % 4];
            }
        }
        Nvfp4Matrix::new(codes, scales, global_scale, rows, cols).unwrap()
    }

    #[test]
    fn gemv_matches_reference() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        let rows = 64;
        let cols = 64;
        let matrix = build_matrix(rows, cols, 0.25);
        let reference = matrix.decode().unwrap();

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let input_gpu = ctx.tensor_from_f32(&input, 1, cols).unwrap();

        let codes_gpu = ctx.stream().clone_htod::<u8, _>(matrix.codes()).unwrap();
        let scales_gpu = ctx.stream().clone_htod::<u8, _>(matrix.scales()).unwrap();
        let out_gpu = gemv_f16(&ctx, &input_gpu, &codes_gpu, &scales_gpu, matrix.global_scale, rows).unwrap();
        let out = ctx.tensor_to_f32(&out_gpu).unwrap();

        let mut expect = vec![0.0f32; rows];
        for n in 0..rows {
            for k in 0..cols {
                expect[n] += input[k] * reference[n * cols + k];
            }
        }
        check_close("nvfp4_gemv", &out, &expect);
    }

    #[test]
    fn dequant_matches_reference() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        let rows = 48;
        let cols = 64;
        let matrix = build_matrix(rows, cols, 0.5);
        let reference = matrix.decode().unwrap();

        let codes_gpu = ctx.stream().clone_htod::<u8, _>(matrix.codes()).unwrap();
        let scales_gpu = ctx.stream().clone_htod::<u8, _>(matrix.scales()).unwrap();
        let f16_buf = dequant_f16(&ctx, &codes_gpu, &scales_gpu, matrix.global_scale, rows, cols).unwrap();
        let host = ctx.stream().clone_dtoh::<half::f16, _>(&f16_buf).unwrap();
        let out: Vec<f32> = host.iter().map(|v| v.to_f32()).collect();
        check_close("nvfp4_dequant", &out, &reference);
    }

    #[test]
    fn gated_silu_matches_reference() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        let rows = 64;
        let cols = 64;
        let gate = build_matrix(rows, cols, 0.25);
        let up = build_matrix(rows, cols, 0.5);
        let gate_ref = gate.decode().unwrap();
        let up_ref = up.decode().unwrap();

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.02 - 0.6).collect();
        let input_gpu = ctx.tensor_from_f32(&input, 1, cols).unwrap();

        let gate_codes = ctx.stream().clone_htod::<u8, _>(gate.codes()).unwrap();
        let gate_scales = ctx.stream().clone_htod::<u8, _>(gate.scales()).unwrap();
        let up_codes = ctx.stream().clone_htod::<u8, _>(up.codes()).unwrap();
        let up_scales = ctx.stream().clone_htod::<u8, _>(up.scales()).unwrap();

        let out_gpu = gated_silu_gemv_f16(&ctx, &input_gpu, &gate_codes, &gate_scales, &up_codes, &up_scales, gate.global_scale, up.global_scale, rows).unwrap();
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
        check_close("nvfp4_gated_silu", &out, &expect);
    }
}
