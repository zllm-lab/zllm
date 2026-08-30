//! CUDA 常驻权重类型。
//!
//! 对称 `backend/metal/resident.rs` 的 `MetalWeight`。
//! 密集权重以 f16 常驻;压缩格式(W4A16 AWQ、per-tensor FP8 E4M3)保持 packed/codes 态,
//! 在 `linear()` 内设备反量化(W4A16 → kernel 内 GEMV/dequant;FP8 → dequant f16 temp + cuBLAS)。

use cudarc::driver::safe::CudaSlice;
use half::f16;

/// compressed-tensors W4A16(AWQ)的设备 packed 视图。
///
/// 权重保持 I32 packed(8 INT4/word)与 per-group scale,不在 host 展开;
/// `linear()` 内部走 `kernel::cuda::w4a16` 即时反量化 GEMV/GEMM。
#[derive(Clone)]
pub struct CudaW4a16 {
    /// I32 字节流(8 INT4/word,LE);kernel 内 reinterpret 为 `const unsigned int*`。
    pub packed: CudaSlice<u8>,
    pub scales: CudaSlice<u8>,
    /// 0=BF16, 1=F16, 2=F32(对称 `ScaleDType::metal_code`)。
    pub scale_dtype: u32,
    pub group_size: usize,
}

/// per-tensor FP8 E4M3 权重的设备视图。行优先 `[rows, cols]` 的 raw 字节 + 单一 F32 scale。
///
/// 权重以 1 字节/元素常驻(相比 f16 减半 → 显存与 HTOD 带宽均减半);`linear()` 内部走
/// `kernel::cuda::fp8`(设备内 dequant 为 f16 temp 再喂 cuBLAS),对称 W4A16 的 prefill 路径。
#[derive(Clone)]
pub struct CudaFp8 {
    /// E4M3 codes,行优先 `[rows, cols]`。
    pub codes: CudaSlice<u8>,
    pub scale: f32,
}

/// MLX `affine` groupwise 矩阵(4-bit 量化)的设备 packed 视图。
///
/// 权重保持 packed I32(8 个无符号 INT4/word,bits=4) + 每组 (scale, bias),
/// 不在 host 展开;`linear()` 内部走 `kernel::cuda::mlx_affine` 即时反量化 GEMV/GEMM。
/// 对称 W4A16 路径,但解码公式是 `weight = scale * unsigned_code + bias`(`affine`)而
/// 不是 W4A16 的 `weight = scale * (signed_code - 8)`(对称)。
#[derive(Clone)]
pub struct CudaMlxAffine {
    /// I32 字节流(8 INT4/word,LE);kernel 内 reinterpret 为 `const unsigned int*`。
    pub packed: CudaSlice<u8>,
    pub scales: CudaSlice<u8>,
    pub biases: CudaSlice<u8>,
    /// 0=BF16, 1=F16, 2=F32(对称 `ScaleDType::metal_code`)。
    pub scale_dtype: u32,
    pub group_size: usize,
    /// 当前只支持 4(其他 bit 宽走 host decode 回退)。
    pub bits: u32,
}

/// NVIDIA ModelOpt NVFP4 矩阵的设备 packed 视图。
///
/// 权重保持 packed u8(2 E2M1/字节)+ per-block E4M3 scale + 全局 F32 scale,不在 host 展开;
/// `linear()` 内部走 `kernel::cuda::nvfp4` 即时反量化 GEMV/dequant。解码公式
/// `weight = e2m1(code) * e4m3(block_scale) * global_scale`。
#[derive(Clone)]
pub struct CudaNvfp4 {
    /// 行优先 `[rows, cols/2]` 的 packed 字节;kernel 内 reinterpret 为 `const unsigned int*`。
    pub codes: CudaSlice<u8>,
    /// 行优先 `[rows, cols/16]` 的 E4M3 block scale 字节。
    pub scales: CudaSlice<u8>,
    pub global_scale: f32,
}

/// compressed-tensors W8A16 group-wise 矩阵的设备 packed 视图。
///
/// 权重保持 1 字节/元素(存 `code + 128`)+ per-group scale,不在 host 展开;
/// `linear()` 内部走 `kernel::cuda::w8a16` 即时反量化 GEMV/dequant,
/// 解码公式 `weight = (int8(byte) - 128) * scale`。ConvRot 变体不走此路径。
#[derive(Clone)]
pub struct CudaW8a16 {
    pub packed: CudaSlice<u8>,
    pub scales: CudaSlice<u8>,
    /// 0=BF16, 1=F16, 2=F32(对称 `ScaleDType::metal_code`)。
    pub scale_dtype: u32,
    pub group_size: usize,
}

/// GGUF Q8_0/Q4_K/Q5_K/Q6_K 原始 block 的设备视图。
///
/// packed 权重不展开；prefill/decode 都由量化 kernel 直接解码并累加。
#[derive(Clone)]
pub struct CudaGgufPacked {
    pub codes: CudaSlice<u8>,
    pub tensor_type: u32,
}

/// 行优先 `[rows, cols]` 的 f16 常驻权重。`rows` 是输出维度(out_features),`cols` 是输入维度(in_features)。
///
/// `data` 始终是有效的 f16 兼容视图；`prepare_f32` 还在 `data_f32` 保存设备真值，
/// 供路由、状态衰减等控制路径直接读取。压缩权重的 `data` 是占位 buffer。
#[derive(Clone)]
pub struct CudaWeight {
    pub data: CudaSlice<f16>,
    pub data_f32: Option<CudaSlice<f32>>,
    pub rows: usize,
    pub cols: usize,
    pub w4a16: Option<CudaW4a16>,
    pub fp8: Option<CudaFp8>,
    pub mlx_affine: Option<CudaMlxAffine>,
    pub nvfp4: Option<CudaNvfp4>,
    pub w8a16: Option<CudaW8a16>,
    pub gguf_packed: Option<CudaGgufPacked>,
}

impl CudaWeight {
    pub fn new(data: CudaSlice<f16>, rows: usize, cols: usize) -> Self {
        Self { data, data_f32: None, rows, cols, w4a16: None, fp8: None, mlx_affine: None, nvfp4: None, w8a16: None, gguf_packed: None }
    }

    pub fn with_f32(data: CudaSlice<f16>, data_f32: CudaSlice<f32>, rows: usize, cols: usize) -> Self {
        Self { data, data_f32: Some(data_f32), rows, cols, w4a16: None, fp8: None, mlx_affine: None, nvfp4: None, w8a16: None, gguf_packed: None }
    }

    /// W4A16 packed 权重。`data` 用单元素占位(norm/elementwise 路径不会触及量化线性权重)。
    pub fn with_w4a16(placeholder: CudaSlice<f16>, w4a16: CudaW4a16, rows: usize, cols: usize) -> Self {
        Self { data: placeholder, data_f32: None, rows, cols, w4a16: Some(w4a16), fp8: None, mlx_affine: None, nvfp4: None, w8a16: None, gguf_packed: None }
    }

    /// per-tensor FP8 权重。`data` 用单元素占位(norm/elementwise 路径不会触及量化线性权重)。
    pub fn with_fp8(placeholder: CudaSlice<f16>, fp8: CudaFp8, rows: usize, cols: usize) -> Self {
        Self { data: placeholder, data_f32: None, rows, cols, w4a16: None, fp8: Some(fp8), mlx_affine: None, nvfp4: None, w8a16: None, gguf_packed: None }
    }

    /// MLX affine packed 权重。`data` 用单元素占位(norm/elementwise 路径不会触及量化线性权重)。
    pub fn with_mlx_affine(placeholder: CudaSlice<f16>, mlx_affine: CudaMlxAffine, rows: usize, cols: usize) -> Self {
        Self { data: placeholder, data_f32: None, rows, cols, w4a16: None, fp8: None, mlx_affine: Some(mlx_affine), nvfp4: None, w8a16: None, gguf_packed: None }
    }

    /// NVFP4 packed 权重。`data` 用单元素占位(norm/elementwise 路径不会触及量化线性权重)。
    pub fn with_nvfp4(placeholder: CudaSlice<f16>, nvfp4: CudaNvfp4, rows: usize, cols: usize) -> Self {
        Self { data: placeholder, data_f32: None, rows, cols, w4a16: None, fp8: None, mlx_affine: None, nvfp4: Some(nvfp4), w8a16: None, gguf_packed: None }
    }

    /// W8A16 packed 权重。`data` 用单元素占位(norm/elementwise 路径不会触及量化线性权重)。
    pub fn with_w8a16(placeholder: CudaSlice<f16>, w8a16: CudaW8a16, rows: usize, cols: usize) -> Self {
        Self { data: placeholder, data_f32: None, rows, cols, w4a16: None, fp8: None, mlx_affine: None, nvfp4: None, w8a16: Some(w8a16), gguf_packed: None }
    }

    pub fn with_gguf_packed(placeholder: CudaSlice<f16>, gguf_packed: CudaGgufPacked, rows: usize, cols: usize) -> Self {
        Self { data: placeholder, data_f32: None, rows, cols, w4a16: None, fp8: None, mlx_affine: None, nvfp4: None, w8a16: None, gguf_packed: Some(gguf_packed) }
    }
}
