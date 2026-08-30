//! Metal decode 常驻权重；上传发生在 token 循环之前。

use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::metal::api::Buffer;

use crate::{
    backend::LinearWeight,
    weight::{
        Fp8Matrix, PerTensorFp8Matrix,
        container::gguf::GgufMatrix,
        format::mxfp4::Mxfp4Matrix,
        format::mxfp8::Mxfp8Matrix,
        format::nvfp4::Nvfp4Matrix,
        format::quantization::{MlxAffineMatrix, QuantizedMatrixRef, W4A16Matrix, W8A16Matrix},
    },
};

use super::{MetalContext, MetalTensor};

static NEXT_MLX_PAIR_ID: AtomicU64 = AtomicU64::new(1);

pub enum MetalWeight {
    F16(MetalTensor),
    Fp8 {
        codes: Buffer,
        scale_inv: Buffer,
        rows: usize,
        cols: usize,
    },
    /// ComfyUI / 社区 per-tensor FP8 矩阵:codes 是 FP8 E4M3 bytes,
    /// scale 是单个 F32。GPU kernel 在 matmul 时按需 dequant,
    /// 不在 host 展开成 BF16 双倍体积。
    Fp8PerTensor {
        codes: Buffer,
        scale: Buffer,
        rows: usize,
        cols: usize,
    },
    Mxfp8 {
        codes: Buffer,
        scale_inv: Buffer,
        rows: usize,
        cols: usize,
    },
    Mxfp4 {
        packed: Buffer,
        scales: Buffer,
        rows: usize,
        cols: usize,
    },
    Nvfp4 {
        codes: Buffer,
        codes_offset: usize,
        scales: Buffer,
        scales_offset: usize,
        global_scale: Buffer,
        global_scale_offset: usize,
        rows: usize,
        cols: usize,
    },
    Gguf {
        blob: Buffer,
        tensor_type: u32,
        row_bytes: usize,
        rows: usize,
        cols: usize,
    },
    W4A16 {
        packed: Buffer,
        scales: Buffer,
        scale_dtype: u32,
        group_size: usize,
        rows: usize,
        cols: usize,
    },
    W8A16 {
        packed: Buffer,
        scales: Buffer,
        scale_dtype: u32,
        group_size: usize,
        rows: usize,
        cols: usize,
    },
    MlxAffine {
        packed: Buffer,
        scales: Buffer,
        biases: Buffer,
        scale_dtype: u32,
        bits: usize,
        group_size: usize,
        rows: usize,
        cols: usize,
        /// 非零时，三个 buffer 均按 pair_role 交错并由两个权重句柄共享。
        pair_id: u64,
        pair_role: u32,
    },
    F32 {
        buffer: Buffer,
        len: usize,
    },
}

impl MetalWeight {
    pub(crate) fn allocate_gguf(ctx: &MetalContext, weight: &GgufMatrix, rows: usize, cols: usize) -> Result<Self, String> {
        if weight.rows != rows || weight.columns != cols {
            return Err(format!("resident GGUF 权重形状 [{},{}] 与 [{rows},{cols}] 不符", weight.rows, weight.columns));
        }
        if !crate::weight::codec::ggml::supports_decode(weight.tensor_type.0) {
            return Err(format!("Metal 尚不支持 resident GGUF {} 权重", weight.tensor_type.name()));
        }
        let row_bytes = weight.tensor_type.storage_bytes(cols)?;
        let expected = row_bytes.checked_mul(rows).ok_or("resident GGUF 权重大小溢出")?;
        Ok(Self::Gguf { blob: ctx.try_shared_buffer_uninit(expected)?, tensor_type: weight.tensor_type.0, row_bytes, rows, cols })
    }

    pub(crate) fn fill_gguf(&self, weight: &GgufMatrix) -> Result<(), String> {
        let Self::Gguf { blob, tensor_type, row_bytes, rows, cols } = self else {
            return Err("Metal fill_gguf 目标不是 GGUF buffer".to_owned());
        };
        let expected = row_bytes.checked_mul(*rows).ok_or("resident GGUF 权重大小溢出")?;
        if weight.tensor_type.0 != *tensor_type || weight.rows != *rows || weight.columns != *cols || weight.storage_len() != expected {
            return Err(format!("Metal GGUF fill 布局不一致: {}=[{},{}]/{} bytes={}，目标=[{},{}]/{} bytes={expected}", weight.name, weight.rows, weight.columns, weight.tensor_type.name(), weight.storage_len(), rows, cols, tensor_type));
        }
        let output = unsafe { std::slice::from_raw_parts_mut(blob.contents().cast::<u8>(), expected) };
        weight.read_into(output)
    }

    pub(crate) fn upload_gguf_rows(ctx: &MetalContext, weight: &GgufMatrix, rows: &[u32]) -> Result<Self, String> {
        if rows.is_empty() || !crate::weight::codec::ggml::supports_decode(weight.tensor_type.0) {
            return Err(format!("Metal 不支持裁剪 GGUF {} rows={}", weight.tensor_type.name(), rows.len()));
        }
        let row_bytes = weight.tensor_type.storage_bytes(weight.columns)?;
        let bytes = rows.len().checked_mul(row_bytes).ok_or("Metal selected GGUF 权重大小溢出")?;
        let blob = ctx.try_shared_buffer_uninit(bytes)?;
        let output = unsafe { std::slice::from_raw_parts_mut(blob.contents().cast::<u8>(), bytes) };
        weight.read_rows_into(rows, output)?;
        Ok(Self::Gguf { blob, tensor_type: weight.tensor_type.0, row_bytes, rows: rows.len(), cols: weight.columns })
    }

    pub fn upload(ctx: &MetalContext, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self, String> {
        match weight {
            LinearWeight::F32(values) => {
                if values.len() != rows.checked_mul(cols).ok_or("resident F16 权重大小溢出")? {
                    return Err(format!("resident F16 权重长度 {} 与 [{rows},{cols}] 不符", values.len()));
                }
                Ok(Self::F16(ctx.tensor_from_f32(values, rows, cols)?))
            }
            LinearWeight::F16(values) => {
                if values.len() != rows.checked_mul(cols).ok_or("resident F16 权重大小溢出")? {
                    return Err(format!("resident F16 权重长度 {} 与 [{rows},{cols}] 不符", values.len()));
                }
                let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
                Ok(Self::F16(MetalTensor::new(ctx.try_shared_buffer(bytes)?, rows, cols)))
            }
            LinearWeight::Bf16Bytes(bytes) => {
                let expected = rows.checked_mul(cols).and_then(|elements| elements.checked_mul(2)).ok_or("resident BF16 权重大小溢出")?;
                if bytes.len() != expected {
                    return Err(format!("resident BF16 权重字节数 {} 与 [{rows},{cols}] 不符", bytes.len()));
                }
                let values: Vec<half::f16> = bytes.chunks_exact(2).map(|chunk| half::f16::from_f32(half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())).collect();
                let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values.as_slice())) };
                Ok(Self::F16(MetalTensor::new(ctx.try_shared_buffer(bytes)?, rows, cols)))
            }
            LinearWeight::Quantized(weight) => match weight {
                QuantizedMatrixRef::BlockFp8(weight) => {
                    if weight.rows != rows || weight.cols != cols || weight.block_shape() != (128, 128) {
                        return Err(format!("resident Block-FP8 shape [{},{}] block={:?}，期望 [{rows},{cols}] block=[128,128]", weight.rows, weight.cols, weight.block_shape()));
                    }
                    let scales = weight.scales().iter().map(|&scale| 2.0_f32.powi(scale as i32 - 127)).collect::<Vec<_>>();
                    let scale_bytes = unsafe { std::slice::from_raw_parts(scales.as_ptr().cast::<u8>(), std::mem::size_of_val(scales.as_slice())) };
                    Ok(Self::Fp8 { codes: ctx.try_shared_buffer(weight.codes())?, scale_inv: ctx.try_shared_buffer(scale_bytes)?, rows, cols })
                }
                QuantizedMatrixRef::Fp8(weight) => Self::upload_fp8(ctx, weight, rows, cols),
                QuantizedMatrixRef::PerTensorFp8(weight) => Self::upload_fp8_per_tensor(ctx, weight, rows, cols),
                QuantizedMatrixRef::Mxfp8(weight) => Self::upload_mxfp8(ctx, weight),
                QuantizedMatrixRef::Mxfp4(weight) => {
                    if weight.rows() != rows || weight.cols() != cols {
                        return Err(format!("resident MXFP4 权重形状 [{},{}] 与 [{rows},{cols}] 不符", weight.rows(), weight.cols()));
                    }
                    Self::upload_mxfp4(ctx, weight)
                }
                QuantizedMatrixRef::Nvfp4(weight) => {
                    if weight.rows != rows || weight.cols != cols {
                        return Err(format!("resident NVFP4 权重形状 [{},{}] 与 [{rows},{cols}] 不符", weight.rows, weight.cols));
                    }
                    Self::upload_nvfp4(ctx, weight)
                }
                QuantizedMatrixRef::W4A16(weight) => Self::upload_w4a16(ctx, weight, rows, cols),
                QuantizedMatrixRef::W8A16(weight) => Self::upload_w8a16(ctx, weight, rows, cols),
                QuantizedMatrixRef::MlxAffine(weight) => Self::upload_mlx_affine(ctx, weight, rows, cols),
                QuantizedMatrixRef::Gguf(weight) => {
                    let output = Self::allocate_gguf(ctx, weight, rows, cols)?;
                    output.fill_gguf(weight)?;
                    Ok(output)
                }
            },
        }
    }

    /// 标准权重只在上传时转换，SSD 数据和 weight 解析格式不受影响。
    pub fn upload_pair(ctx: &MetalContext, first: LinearWeight<'_>, second: LinearWeight<'_>, rows: usize, cols: usize) -> Result<(Self, Self), String> {
        match (first, second) {
            (LinearWeight::Quantized(QuantizedMatrixRef::MlxAffine(first)), LinearWeight::Quantized(QuantizedMatrixRef::MlxAffine(second)))
                if first.bits() == 8 && second.bits() == 8 && first.group_size() == 64 && second.group_size() == 64 && first.scale_dtype().metal_code() == 0 && second.scale_dtype().metal_code() == 0 =>
            {
                Self::upload_mlx_affine_pair(ctx, first, second, rows, cols)
            }
            (LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(first)), LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(second))) => {
                if first.rows != rows || first.columns != cols || second.rows != rows || second.columns != cols {
                    return Err(format!("Metal GGUF pair shape 不一致: {}=[{},{}]/{} {}=[{},{}]/{}", first.name, first.rows, first.columns, first.tensor_type.name(), second.name, second.rows, second.columns, second.tensor_type.name()));
                }
                // UD GGUF 允许 gate/up 使用不同量化类型；不能因为无法共享 pair
                // 布局而拒绝整个模型，分别上传后由通用 gated kernel 消费。
                if first.tensor_type != second.tensor_type {
                    return Ok((Self::upload(ctx, LinearWeight::gguf(first), rows, cols)?, Self::upload(ctx, LinearWeight::gguf(second), rows, cols)?));
                }
                let first_weight = Self::allocate_gguf(ctx, first, rows, cols)?;
                let second_weight = Self::allocate_gguf(ctx, second, rows, cols)?;
                std::thread::scope(|scope| {
                    let first_read = scope.spawn(|| first_weight.fill_gguf(first));
                    second_weight.fill_gguf(second)?;
                    first_read.join().map_err(|_| "Metal GGUF pair 读取线程 panic".to_owned())??;
                    Ok::<(), String>(())
                })?;
                Ok((first_weight, second_weight))
            }
            (first, second) => Ok((Self::upload(ctx, first, rows, cols)?, Self::upload(ctx, second, rows, cols)?)),
        }
    }

    fn upload_fp8(ctx: &MetalContext, weight: &Fp8Matrix, rows: usize, cols: usize) -> Result<Self, String> {
        if weight.rows != rows || weight.cols != cols {
            return Err(format!("resident FP8 权重形状 [{},{}] 与 [{rows},{cols}] 不符", weight.rows, weight.cols));
        }
        Ok(Self::Fp8 { codes: ctx.try_shared_buffer(&weight.codes)?, scale_inv: ctx.try_shared_buffer(&weight.scale_inv)?, rows, cols })
    }

    /// 上传 ComfyUI / 社区 per-tensor FP8 矩阵:codes 直接拷贝,
    /// scale 是单个 F32。GPU kernel 自己 dequant。
    fn upload_fp8_per_tensor(ctx: &MetalContext, weight: &PerTensorFp8Matrix, rows: usize, cols: usize) -> Result<Self, String> {
        if weight.rows != rows || weight.cols != cols {
            return Err(format!("resident per-tensor FP8 权重形状 [{},{}] 与 [{rows},{cols}] 不符", weight.rows, weight.cols));
        }
        Ok(Self::Fp8PerTensor { codes: ctx.try_resident_byte_weight_buffer(&weight.codes)?, scale: ctx.try_resident_byte_weight_buffer(&weight.scale.to_le_bytes())?, rows, cols })
    }

    pub fn upload_mxfp8(ctx: &MetalContext, weight: &Mxfp8Matrix) -> Result<Self, String> {
        Ok(Self::Mxfp8 { codes: ctx.try_shared_buffer(weight.codes())?, scale_inv: ctx.try_shared_buffer(weight.scale_inv())?, rows: weight.rows, cols: weight.cols })
    }

    pub fn upload_mxfp4(ctx: &MetalContext, weight: &Mxfp4Matrix) -> Result<Self, String> {
        Ok(Self::Mxfp4 { packed: ctx.try_shared_buffer(weight.packed())?, scales: ctx.try_shared_buffer(weight.scales())?, rows: weight.rows(), cols: weight.cols() })
    }

    pub fn upload_nvfp4(ctx: &MetalContext, weight: &Nvfp4Matrix) -> Result<Self, String> {
        Ok(Self::Nvfp4 {
            codes: ctx.try_shared_buffer(weight.codes())?,
            codes_offset: 0,
            scales: ctx.try_shared_buffer(weight.scales())?,
            scales_offset: 0,
            global_scale: ctx.try_shared_buffer(&weight.global_scale.to_le_bytes())?,
            global_scale_offset: 0,
            rows: weight.rows,
            cols: weight.cols,
        })
    }

    fn upload_w4a16(ctx: &MetalContext, weight: &W4A16Matrix, rows: usize, cols: usize) -> Result<Self, String> {
        if weight.rows != rows || weight.cols != cols {
            return Err(format!("resident W4A16 权重形状 [{},{}] 与 [{rows},{cols}] 不符", weight.rows, weight.cols));
        }
        Ok(Self::W4A16 { packed: ctx.try_shared_buffer(weight.packed())?, scales: ctx.try_shared_buffer(weight.scales())?, scale_dtype: weight.scale_dtype().metal_code(), group_size: weight.group_size(), rows, cols })
    }

    fn upload_w8a16(ctx: &MetalContext, weight: &W8A16Matrix, rows: usize, cols: usize) -> Result<Self, String> {
        if weight.rows != rows || weight.cols != cols {
            return Err(format!("resident W8A16 权重形状 [{},{}] 与 [{rows},{cols}] 不符", weight.rows, weight.cols));
        }
        Ok(Self::W8A16 { packed: ctx.try_shared_buffer(weight.packed())?, scales: ctx.try_shared_buffer(weight.scales())?, scale_dtype: weight.scale_dtype().metal_code(), group_size: weight.group_size(), rows, cols })
    }

    fn upload_mlx_affine(ctx: &MetalContext, weight: &MlxAffineMatrix, rows: usize, cols: usize) -> Result<Self, String> {
        if weight.rows != rows || weight.cols != cols {
            return Err(format!("resident MLX affine 权重形状 [{},{}] 与 [{rows},{cols}] 不符", weight.rows, weight.cols));
        }
        Ok(Self::MlxAffine {
            packed: ctx.try_shared_buffer(weight.packed())?,
            scales: ctx.try_shared_buffer(weight.scales())?,
            biases: ctx.try_shared_buffer(weight.biases())?,
            scale_dtype: weight.scale_dtype().metal_code(),
            bits: weight.bits(),
            group_size: weight.group_size(),
            rows,
            cols,
            pair_id: 0,
            pair_role: 0,
        })
    }

    fn upload_mlx_affine_pair(ctx: &MetalContext, first: &MlxAffineMatrix, second: &MlxAffineMatrix, rows: usize, cols: usize) -> Result<(Self, Self), String> {
        for (name, weight) in [("first", first), ("second", second)] {
            if weight.rows != rows || weight.cols != cols {
                return Err(format!("resident MLX affine {name} 成对权重形状 [{},{}] 与 [{rows},{cols}] 不符", weight.rows, weight.cols,));
            }
        }
        if first.bits() != second.bits() || first.group_size() != second.group_size() || first.scale_dtype().metal_code() != second.scale_dtype().metal_code() {
            return Err("resident MLX affine 成对权重的量化参数不一致".to_owned());
        }

        let packed_bytes = interleave_blocks(first.packed(), second.packed(), 16, "codes")?;
        let scales_bytes = interleave_blocks(first.scales(), second.scales(), 2, "scales")?;
        let biases_bytes = interleave_blocks(first.biases(), second.biases(), 2, "biases")?;
        let packed = ctx.try_shared_buffer(&packed_bytes)?;
        let scales = ctx.try_shared_buffer(&scales_bytes)?;
        let biases = ctx.try_shared_buffer(&biases_bytes)?;
        let pair_id = NEXT_MLX_PAIR_ID.fetch_add(1, Ordering::Relaxed);
        let make_weight = |pair_role| Self::MlxAffine {
            packed: packed.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
            scale_dtype: first.scale_dtype().metal_code(),
            bits: first.bits(),
            group_size: first.group_size(),
            rows,
            cols,
            pair_id,
            pair_role,
        };
        Ok((make_weight(0), make_weight(1)))
    }

    pub fn upload_f32(ctx: &MetalContext, values: &[f32]) -> Result<Self, String> {
        {
            let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
            Ok(Self::F32 { buffer: ctx.try_shared_buffer(bytes)?, len: values.len() })
        }
    }
}

fn interleave_blocks(first: &[u8], second: &[u8], block_bytes: usize, name: &str) -> Result<Vec<u8>, String> {
    if first.len() != second.len() || block_bytes == 0 || !first.len().is_multiple_of(block_bytes) {
        return Err(format!("resident MLX affine 成对 {name} 无法按 {block_bytes} 字节交错: first={}, second={}", first.len(), second.len(),));
    }
    let mut interleaved = Vec::with_capacity(first.len() * 2);
    for (first_block, second_block) in first.chunks_exact(block_bytes).zip(second.chunks_exact(block_bytes)) {
        interleaved.extend_from_slice(first_block);
        interleaved.extend_from_slice(second_block);
    }
    Ok(interleaved)
}
