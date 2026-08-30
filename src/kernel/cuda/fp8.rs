//! per-tensor FP8 E4M3 设备内反量化算子。
//!
//! H3 curve+FP8 checkpoint 的 DiT 线性权重存为 raw E4M3 字节 + 单一 per-tensor F32 scale
//! (`crate::weight::PerTensorFp8Matrix`)。sm_86 无 fp8 硬件(Ampere 没有 fp8 tensor core、
//! 也无 `__nv_cvt_fp8_to_halfraw`),故采用与 W4A16 prefill 相同的策略:权重以 1 字节/元素
//! 常驻显存(相比 f16 减半 → 显存/HTOD 带宽均减半),`linear()` 内 dequant 为 f16 temp 再喂 cuBLAS。
//!
//! 不做寄存器内融合 GEMM:DiT 每步计算仅 ~0.14s(瓶颈是 HTOD 上传),手写 fp8 GEMM 几乎
//! 必然慢于 cuBLAS hgemm,反而劣化;fusion 的收益全在可忽略的计算上。对称 W4A16 prefill 路径。
//!
//! 解码与 `src/weight/codec/fp8.rs::decode_f8_e4m3` 逐位等价(软件 bit-decode)。

use cudarc::driver::safe::CudaSlice;

use super::{CudaContext, LaunchConfig, PushKernelArg, THREADS};

/// 本模块的 CUDA shader。共用 helper 见 `mod.rs` 前导。
// kernels: fp8_dequant_e4m3_f16
pub const SHADERS: &str = r#"
// FP8 E4M3 → f16 反量化(per-tensor scale)。一线程多元素(grid-stride)。
// bit 布局:[s eeee mmm];sign 位、4 位指数(bias 7)、3 位尾数。等价 decode_f8_e4m3:
//   e=0      → subnormal  sign * 2^-6 * mantissa/8
//   e=15,m=7 → NaN(e4m3fn 实际无 NaN,此处与 CPU codec 保持一致)
//   其余      → sign * 2^(e-7) * (1 + mantissa/8)
// 全程 f32 计算,最后 __float2half(×scale)(round-to-nearest-even)。
extern "C" __global__ void fp8_dequant_e4m3_f16(
    const unsigned char * __restrict__ codes,
    const float scale,
    __half * __restrict__ out,
    const unsigned int count)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        unsigned int bits = codes[i];
        float sign = (bits & 0x80u) ? -1.0f : 1.0f;
        unsigned int exponent = (bits >> 3) & 0x0fu;
        unsigned int mantissa = bits & 0x07u;
        float value;
        if (exponent == 0u) {
            value = sign * 0.015625f * ((float)mantissa * 0.125f);
        } else if (exponent == 15u && mantissa == 7u) {
            value = nanf("");
        } else {
            value = sign * ldexpf(1.0f + (float)mantissa * 0.125f, (int)exponent - 7);
        }
        out[i] = __float2half(value * scale);
    }
}
"#;

/// 把行优先 `[rows, cols]` 的 FP8 E4M3 codes 反量化为 f16(喂 cuBLAS)。
///
/// `count` = rows*cols。返回未初始化的 f16 buffer 并原地填充。空矩阵直接返回空 buffer。
pub fn fp8_dequant_f16(ctx: &CudaContext, codes: &CudaSlice<u8>, scale: f32, count: usize) -> Result<CudaSlice<half::f16>, String> {
    let output = ctx.buffer_uninit::<half::f16>(count)?;
    if count == 0 {
        return Ok(output);
    }
    let func = ctx.function("fp8_dequant_e4m3_f16")?;
    // grid-stride:grid 上限 1024 个 block(避免巨型 grid 的 launch 开销),每线程跨步覆盖。
    let blocks = count.min(THREADS as usize * 1024).div_ceil(THREADS as usize).max(1) as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(codes)
            .arg(&scale)
            .arg(&output)
            .arg(&(count as u32))
            .launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|e| format!("launch fp8_dequant_e4m3_f16 失败: {e:?}"))?;
    }
    Ok(output)
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use super::*;
    use crate::weight::codec::fp8::decode_f8_e4m3;

    fn check_close(name: &str, actual: &[f32], expect: &[f32]) {
        let atol = 1e-2f32;
        let rtol = 1e-2f32;
        for (i, (a, e)) in actual.iter().zip(expect).enumerate() {
            // e4m3fn 的 e=15,m=7(0x7F/0xFF)是 NaN:CPU codec 与 GPU kernel 都产出 NaN,视为匹配。
            if a.is_nan() && e.is_nan() {
                continue;
            }
            let err = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(err <= tol, "{name}[{i}] 偏差 {err:.4} 超阈值 {tol:.4}(actual={a:.4} expect={e:.4})");
        }
    }

    // 遍历全部 256 个 E4M3 字节 + 非平凡 scale,逐元素对照 CPU codec(decode_f8_e4m3 * scale)。
    #[test]
    fn fp8_dequant_matches_oracle() {
        let ctx = CudaContext::new_default().expect("CUDA 初始化");
        let cols = 256;
        let rows = 40;
        let scale = 0.5f32;
        // codes 覆盖 0..=255(每行循环一遍),exhaustive 验证 decode 表。
        let codes: Vec<u8> = (0..(rows * cols)).map(|i| (i % 256) as u8).collect();
        let reference: Vec<f32> = codes.iter().map(|&c| decode_f8_e4m3(c) * scale).collect();

        let codes_gpu = ctx.stream().clone_htod::<u8, _>(&codes).unwrap();
        let f16_buf = fp8_dequant_f16(&ctx, &codes_gpu, scale, rows * cols).unwrap();
        let host = ctx.stream().clone_dtoh::<half::f16, _>(&f16_buf).unwrap();
        let out: Vec<f32> = host.iter().map(|v| v.to_f32()).collect();
        check_close("fp8_dequant_e4m3", &out, &reference);
    }
}
