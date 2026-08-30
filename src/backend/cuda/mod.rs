//! CUDA 后端资源与执行调度。
//!
//! 对称 `backend/metal/mod.rs` 与 `backend/cpu/mod.rs`。
//! `CudaContext` 只实现已有真实 kernel 支撑的 capability；模型 runtime
//! 按需要组合这些能力，没有实现的能力不提供占位实现。

pub mod context;
pub mod expert;
mod gated_delta_net;
pub mod kv_cache;
pub mod resident;
mod vae;

pub use context::{CudaContext, CudaContextOptions, CudaTensor};
pub use expert::{CudaMoeAccumulator, CudaMoeState, CudaPrefillExperts};
pub use gated_delta_net::CudaGatedDeltaNetStorage;
pub use kv_cache::CudaKvCache;
pub use resident::{CudaFp8, CudaGgufPacked, CudaMlxAffine, CudaNvfp4, CudaW4a16, CudaW8a16, CudaWeight};

use half::f16;

use crate::backend::{Backend, BackendError, BackendResources, LinearWeight, compute_error};
use crate::kernel::cuda as ops;
use crate::moe::Activation;
use crate::weight::format::quantization::QuantizedMatrixRef;

impl BackendResources for CudaContext {
    type Tensor = CudaTensor;
    type Weight = CudaWeight;
    type Cache = CudaKvCache;
    type LayerScope<'a>
        = ()
    where
        Self: 'a;

    fn layer_scope(&self) -> Self::LayerScope<'_> {}

    fn token_rows(&self, tensor: &CudaTensor) -> usize {
        tensor.rows
    }

    fn token_cols(&self, tensor: &CudaTensor) -> usize {
        tensor.cols
    }

    fn tensor_allocated_bytes(&self, tensor: &CudaTensor) -> u64 {
        tensor.slice_f32.as_ref().map_or_else(|| tensor.slice.len().saturating_mul(std::mem::size_of::<f16>()) as u64, |slice| slice.len().saturating_mul(std::mem::size_of::<f32>()) as u64)
    }

    fn begin_batch(&self) {}

    fn finish_batch(&self) {}

    fn prepare_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<CudaWeight, BackendError> {
        if let LinearWeight::Quantized(QuantizedMatrixRef::Gguf(matrix)) = weight
            && matches!(matrix.tensor_type.0, 8 | 12..=14)
        {
            if matrix.rows != rows || matrix.columns != cols {
                return Err(compute_error(format!("GGUF {} weight shape [{},{}]，期望 [{rows},{cols}]", matrix.name, matrix.rows, matrix.columns)));
            }
            let bytes = matrix.read_bytes().map_err(compute_error)?;
            let codes = self.stream().clone_htod::<u8, _>(&bytes).map_err(|e| compute_error(format!("CUDA GGUF {} 上传失败: {e:?}", matrix.name)))?;
            let placeholder = self.stream().alloc_zeros::<f16>(1).map_err(|e| compute_error(format!("CUDA GGUF 占位分配失败: {e:?}")))?;
            return Ok(CudaWeight::with_gguf_packed(placeholder, CudaGgufPacked { codes, tensor_type: matrix.tensor_type.0 }, rows, cols));
        }
        // MLX affine 4-bit:保持 packed I32 + per-group (scale, bias) 直接上传,
        // 设备内即时反量化(见 kernel::cuda::mlx_affine)。相比 f32→f16 上传省 ~10× 显存 + HTOD。
        if let LinearWeight::Quantized(QuantizedMatrixRef::MlxAffine(matrix)) = weight {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute_error(format!("MLX affine weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            if matrix.bits() != 4 {
                return Err(compute_error(format!("CUDA MLX affine 当前仅支持 bits=4，got bits={}", matrix.bits())));
            }
            let packed = self.stream().clone_htod::<u8, _>(matrix.packed()).map_err(|e| compute_error(format!("CUDA MLX affine packed 上传失败: {e:?}")))?;
            let scales = self.stream().clone_htod::<u8, _>(matrix.scales()).map_err(|e| compute_error(format!("CUDA MLX affine scales 上传失败: {e:?}")))?;
            let biases = self.stream().clone_htod::<u8, _>(matrix.biases()).map_err(|e| compute_error(format!("CUDA MLX affine biases 上传失败: {e:?}")))?;
            let placeholder = self.stream().alloc_zeros::<f16>(1).map_err(|e| compute_error(format!("CUDA MLX affine 占位分配失败: {e:?}")))?;
            return Ok(CudaWeight::with_mlx_affine(placeholder, CudaMlxAffine { packed, scales, biases, scale_dtype: matrix.scale_dtype().metal_code(), group_size: matrix.group_size(), bits: matrix.bits() as u32 }, rows, cols));
        }
        // W4A16(AWQ)保持 packed I32 + per-group scale 直接上传,设备内即时反量化(见 kernel::cuda::w4a16)。
        if let LinearWeight::Quantized(QuantizedMatrixRef::W4A16(matrix)) = weight {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute_error(format!("W4A16 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            let packed = self.stream().clone_htod::<u8, _>(matrix.packed()).map_err(|e| compute_error(format!("CUDA W4A16 packed 上传失败: {e:?}")))?;
            let scales = self.stream().clone_htod::<u8, _>(matrix.scales()).map_err(|e| compute_error(format!("CUDA W4A16 scales 上传失败: {e:?}")))?;
            // 占位 f16(norm/elementwise 路径不触及量化线性权重,此处仅需合法 CudaSlice)。
            let placeholder = self.stream().alloc_zeros::<f16>(1).map_err(|e| compute_error(format!("CUDA W4A16 占位分配失败: {e:?}")))?;
            return Ok(CudaWeight::with_w4a16(placeholder, CudaW4a16 { packed, scales, scale_dtype: matrix.scale_dtype().metal_code(), group_size: matrix.group_size() }, rows, cols));
        }
        // per-tensor FP8 E4M3(H3 curve+FP8 checkpoint):raw 字节 + 单 F32 scale 直接上传,
        // 设备内 dequant 为 f16 temp 再喂 cuBLAS(见 kernel::cuda::fp8)。相比 f16 减半显存/HTOD。
        if let LinearWeight::Quantized(QuantizedMatrixRef::PerTensorFp8(matrix)) = weight {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute_error(format!("FP8 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            let codes = self.stream().clone_htod::<u8, _>(&matrix.codes).map_err(|e| compute_error(format!("CUDA FP8 codes 上传失败: {e:?}")))?;
            let placeholder = self.stream().alloc_zeros::<f16>(1).map_err(|e| compute_error(format!("CUDA FP8 占位分配失败: {e:?}")))?;
            return Ok(CudaWeight::with_fp8(placeholder, CudaFp8 { codes, scale: matrix.scale }, rows, cols));
        }
        // NVFP4(ModelOpt):保持 packed u8 + E4M3 block scale + 全局 F32 scale 直接上传,
        // 设备内即时反量化(见 kernel::cuda::nvfp4)。相比 f16 省约 4× 显存 + HTOD。
        if let LinearWeight::Quantized(QuantizedMatrixRef::Nvfp4(matrix)) = weight {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute_error(format!("NVFP4 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            let codes = self.stream().clone_htod::<u8, _>(matrix.codes()).map_err(|e| compute_error(format!("CUDA NVFP4 codes 上传失败: {e:?}")))?;
            let scales = self.stream().clone_htod::<u8, _>(matrix.scales()).map_err(|e| compute_error(format!("CUDA NVFP4 scales 上传失败: {e:?}")))?;
            let placeholder = self.stream().alloc_zeros::<f16>(1).map_err(|e| compute_error(format!("CUDA NVFP4 占位分配失败: {e:?}")))?;
            return Ok(CudaWeight::with_nvfp4(placeholder, CudaNvfp4 { codes, scales, global_scale: matrix.global_scale }, rows, cols));
        }
        // W8A16 group-wise INT8(Q8g128 LM head / compressed-tensors):保持 1 字节/元素
        // + per-group scale 直接上传,设备内即时反量化(见 kernel::cuda::w8a16)。
        // ConvRot 变体解码不同,不走 packed kernel,落回下方通用 decode。
        if let LinearWeight::Quantized(QuantizedMatrixRef::W8A16(matrix)) = weight
            && matrix.convrot_group_size().is_none()
        {
            if matrix.rows != rows || matrix.cols != cols {
                return Err(compute_error(format!("W8A16 weight shape [{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols)));
            }
            let packed = self.stream().clone_htod::<u8, _>(matrix.packed()).map_err(|e| compute_error(format!("CUDA W8A16 packed 上传失败: {e:?}")))?;
            let scales = self.stream().clone_htod::<u8, _>(matrix.scales()).map_err(|e| compute_error(format!("CUDA W8A16 scales 上传失败: {e:?}")))?;
            let placeholder = self.stream().alloc_zeros::<f16>(1).map_err(|e| compute_error(format!("CUDA W8A16 占位分配失败: {e:?}")))?;
            return Ok(CudaWeight::with_w8a16(placeholder, CudaW8a16 { packed, scales, scale_dtype: matrix.scale_dtype().metal_code(), group_size: matrix.group_size() }, rows, cols));
        }
        // F16 密集:checkpoint 字节序即设备 f16 布局 → 直接上传。跳过旧路径的 f16→f32→f16 往返
        // (2 次全量元素转换 + 2 次大 Vec 分配)。DiT 注意力权重(qkv/out_proj/adaln)是 F16,
        // 流式 denoise 每步重 prepare 所有 50 层,这条路径是 prepare 主导开销。无损(f16↔f32 往返本就精确)。
        if let LinearWeight::F16(values) = weight {
            let expected = rows.checked_mul(cols).ok_or_else(|| compute_error("CUDA F16 weight 大小溢出"))?;
            if values.len() != expected {
                return Err(compute_error(format!("CUDA F16 weight 元素数 {}，期望 {expected}", values.len())));
            }
            // pinned upload 绕过 cudarc SyncOnDrop 在重型 stream 上的 ~30s stall。
            let mut slice = unsafe { self.stream().alloc::<f16>(values.len()) }.map_err(|e| compute_error(format!("CUDA F16 alloc 失败: {e:?}")))?;
            self.upload_f16_pinned(values, &mut slice).map_err(compute_error)?;
            return Ok(CudaWeight::new(slice, rows, cols));
        }
        // BF16 密集:一次 bf16→f16 直传(norm/token_refiner),跳过 f32 往返(省去全量 f32 Vec)。
        if let LinearWeight::Bf16Bytes(bytes) = weight {
            let expected = rows.checked_mul(cols).and_then(|elements| elements.checked_mul(2)).ok_or_else(|| compute_error("CUDA BF16 weight 大小溢出"))?;
            if bytes.len() != expected {
                return Err(compute_error(format!("CUDA BF16 weight 字节数 {}，期望 {expected}", bytes.len())));
            }
            let f16_data: Vec<f16> = bytes.chunks_exact(2).map(|chunk| half::f16::from_f32(half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())).collect();
            let mut slice = unsafe { self.stream().alloc::<f16>(f16_data.len()) }.map_err(|e| compute_error(format!("CUDA BF16 alloc 失败: {e:?}")))?;
            self.upload_f16_pinned(&f16_data, &mut slice).map_err(compute_error)?;
            return Ok(CudaWeight::new(slice, rows, cols));
        }
        // 其余密集/量化(F32 / 量化 fallback):CPU 解码为 f32 再转 f16 上传。
        let data: Vec<f32> = match weight {
            LinearWeight::F32(values) => values.to_vec(),
            LinearWeight::F16(_) | LinearWeight::Bf16Bytes(_) => unreachable!("F16/BF16 已在上方 if-let 直接上传"),
            LinearWeight::Quantized(matrix) => {
                if matrix.rows() != rows || matrix.cols() != cols {
                    return Err(compute_error(format!("{} weight shape [{},{}]，期望 [{rows},{cols}]", matrix.name(), matrix.rows(), matrix.cols())));
                }
                matrix.decode().map_err(compute_error)?
            }
        };
        let f16_data: Vec<f16> = data.iter().map(|v| f16::from_f32(*v)).collect();
        let mut slice = unsafe { self.stream().alloc::<f16>(f16_data.len()) }.map_err(|e| compute_error(format!("CUDA prepare_weight alloc 失败: {e:?}")))?;
        self.upload_f16_pinned(&f16_data, &mut slice).map_err(compute_error)?;
        Ok(CudaWeight::new(slice, rows, cols))
    }

    fn prepare_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<CudaWeight, BackendError> {
        if values.len() != rows * cols {
            return Err(compute_error(format!("CUDA prepare_f32 元素数 {}，期望 {}", values.len(), rows * cols)));
        }
        let f16_data: Vec<f16> = values.iter().map(|v| f16::from_f32(*v)).collect();
        let mut slice = unsafe { self.stream().alloc::<f16>(f16_data.len()) }.map_err(|e| compute_error(format!("CUDA prepare_f32 alloc 失败: {e:?}")))?;
        self.upload_f16_pinned(&f16_data, &mut slice).map_err(compute_error)?;
        let data_f32 = self.stream().clone_htod::<f32, _>(values).map_err(|e| compute_error(format!("CUDA prepare_f32 F32 上传失败: {e:?}")))?;
        Ok(CudaWeight::with_f32(slice, data_f32, rows, cols))
    }
}

impl Backend for CudaContext {
    fn linear(&self, input: &CudaTensor, weight: &CudaWeight) -> Result<CudaTensor, BackendError> {
        if input.cols != weight.cols {
            return Err(compute_error(format!("CUDA linear input cols={}，weight=[{},{}]", input.cols, weight.rows, weight.cols)));
        }
        if let Some(gguf) = &weight.gguf_packed {
            if input.slice_f32.is_some() {
                return Err(compute_error("CUDA GGUF K-quant linear 只接受 F16 activation"));
            }
            return ops::linear::gguf_kq_matmul_f16(self, input, &gguf.codes, weight.rows, gguf.tensor_type).map_err(compute_error);
        }
        // per-tensor FP8 E4M3(对称 W4A16 prefill):设备内 dequant 为 f16 temp,再按 input dtype 喂 cuBLAS。
        // 必须在 f32 分支前:fp8 权重的 data 是占位,f32 分支会误用占位。
        if let Some(fp8) = &weight.fp8 {
            let count = weight.rows.checked_mul(weight.cols).ok_or_else(|| compute_error("CUDA FP8 weight 大小溢出"))?;
            let f16_weight = ops::fp8::fp8_dequant_f16(self, &fp8.codes, fp8.scale, count).map_err(compute_error)?;
            if input.slice_f32.is_some() {
                return ops::linear::cublas_matmul_f32(self, input, &f16_weight, weight.rows).map_err(compute_error);
            }
            return ops::linear::cublas_matmul_f16(self, input, &f16_weight, weight.rows).map_err(compute_error);
        }
        // f32 扩散激活(DiT 残差流传播下来):f32 GEMM(Sgemm + 设备内 weight f16→f32),输出 f32。
        // 残差流可达 ~5e5,MLP down 投影输出可达 ~6e4,均超 f16,故扩散 linear 在 f32 路径输出 f32。
        if input.slice_f32.is_some() {
            return ops::linear::cublas_matmul_f32(self, input, &weight.data, weight.rows).map_err(compute_error);
        }
        // MLX affine 4-bit:decode GEMV / prefill dequant,共用设备内反量化路径,
        // 对称 W4A16 路径。MLX affine 输入是 f16 hidden(残差流不会传过来),走此分支。
        if let Some(m) = &weight.mlx_affine {
            if input.rows == 1 {
                return ops::mlx_affine::gemv_f16(self, input, &m.packed, &m.scales, &m.biases, weight.rows, m.group_size, m.scale_dtype, m.bits).map_err(compute_error);
            }
            let f16_weight = ops::mlx_affine::dequant_f16(self, &m.packed, &m.scales, &m.biases, weight.rows, weight.cols, m.group_size, m.scale_dtype, m.bits).map_err(compute_error)?;
            return ops::linear::cublas_matmul_f16(self, input, &f16_weight, weight.rows).map_err(compute_error);
        }
        // W4A16 packed:decode(单 token)走设备内 GEMV;prefill(多 token)反量化为 f16 再喂 cuBLAS。
        if let Some(w4a16) = &weight.w4a16 {
            if input.rows == 1 {
                return ops::w4a16::gemv_f16(self, input, &w4a16.packed, &w4a16.scales, weight.rows, w4a16.group_size, w4a16.scale_dtype).map_err(compute_error);
            }
            let f16_weight = ops::w4a16::dequant_f16(self, &w4a16.packed, &w4a16.scales, weight.rows, weight.cols, w4a16.group_size, w4a16.scale_dtype).map_err(compute_error)?;
            return ops::linear::cublas_matmul_f16(self, input, &f16_weight, weight.rows).map_err(compute_error);
        }
        // NVFP4 packed:decode 走设备内 GEMV / prefill 反量化 + cuBLAS,对称 W4A16 路径。
        if let Some(nvfp4) = &weight.nvfp4 {
            if input.rows == 1 {
                return ops::nvfp4::gemv_f16(self, input, &nvfp4.codes, &nvfp4.scales, nvfp4.global_scale, weight.rows).map_err(compute_error);
            }
            let f16_weight = ops::nvfp4::dequant_f16(self, &nvfp4.codes, &nvfp4.scales, nvfp4.global_scale, weight.rows, weight.cols).map_err(compute_error)?;
            return ops::linear::cublas_matmul_f16(self, input, &f16_weight, weight.rows).map_err(compute_error);
        }
        // W8A16 packed:decode 走设备内 GEMV / prefill 反量化 + cuBLAS,对称 W4A16 路径。
        if let Some(w8a16) = &weight.w8a16 {
            if input.rows == 1 {
                return ops::w8a16::gemv_f16(self, input, &w8a16.packed, &w8a16.scales, weight.rows, w8a16.group_size, w8a16.scale_dtype).map_err(compute_error);
            }
            let f16_weight = ops::w8a16::dequant_f16(self, &w8a16.packed, &w8a16.scales, weight.rows, weight.cols, w8a16.group_size, w8a16.scale_dtype).map_err(compute_error)?;
            return ops::linear::cublas_matmul_f16(self, input, &f16_weight, weight.rows).map_err(compute_error);
        }
        ops::linear::cublas_matmul_f16(self, input, &weight.data, weight.rows).map_err(compute_error)
    }

    fn rmsnorm(&self, input: &CudaTensor, weight: &CudaWeight, eps: f32) -> Result<CudaTensor, BackendError> {
        if weight.cols != input.cols || weight.rows != 1 {
            return Err(compute_error(format!("CUDA rmsnorm weight=[{},{}]，input cols={}", weight.rows, weight.cols, input.cols)));
        }
        // f32 残差流(DiT 主干 hidden,可达 ~5e5):读 f32、写 f32(把 f32 传播到下游 adaln/linear)。
        if let Some(input_f32) = &input.slice_f32 {
            return ops::diffusion::rmsnorm_residual_f32(self, input_f32, &weight.data, input.rows, input.cols, eps).map_err(compute_error);
        }
        ops::tensor::rmsnorm_f16(self, input, &weight.data, eps).map_err(compute_error)
    }

    fn gemma_rmsnorm(&self, input: &CudaTensor, weight: &CudaWeight, eps: f32) -> Result<CudaTensor, BackendError> {
        ops::attention::gemma_rmsnorm_heads_f16(self, input, &weight.data, 1, input.cols, eps).map_err(compute_error)
    }

    fn add_gemma_rmsnorm(&self, input: &CudaTensor, residual: &CudaTensor, weight: &CudaWeight, eps: f32) -> Result<CudaTensor, BackendError> {
        // 类 LLaMA 每层 1 次调用:hidden + mixed → post_attention_norm → ffn_input。
        // 设备内核一次 launch 同时完成 add + GemmaRMSNorm,省一份 sum 中间 tensor 的读写。
        ops::attention::gemma_rmsnorm_residual_heads_f16(self, input, residual, &weight.data, 1, input.cols, eps).map_err(compute_error)
    }

    fn layernorm_bias(&self, input: &CudaTensor, weight: &CudaWeight, bias: &CudaWeight, eps: f32) -> Result<CudaTensor, BackendError> {
        if weight.rows != 1 || weight.cols != input.cols || bias.rows != 1 || bias.cols != input.cols {
            return Err(compute_error(format!("CUDA layernorm_bias input=[{},{}] weight=[{},{}] bias=[{},{}]", input.rows, input.cols, weight.rows, weight.cols, bias.rows, bias.cols)));
        }
        // F32 残差流先在设备内收缩到 f16；旧 host 回退最终同样上传为 f16。
        let converted;
        let input = if input.slice_f32.is_some() {
            converted = ops::diffusion::cast_f32_to_f16(self, input).map_err(compute_error)?;
            &converted
        } else {
            input
        };
        ops::tensor::layer_norm_f16(self, input, &weight.data, &bias.data, eps).map_err(compute_error)
    }

    fn split_columns(&self, input: &CudaTensor, left_columns: usize) -> Result<(CudaTensor, CudaTensor), BackendError> {
        let right_columns = input.cols.checked_sub(left_columns).ok_or_else(|| compute_error(format!("CUDA split left={left_columns} 超过 cols={}", input.cols)))?;
        ops::tensor::split_columns_f16(self, input, left_columns, right_columns).map_err(compute_error)
    }

    fn split_interleaved_columns(&self, input: &CudaTensor, block_columns: usize) -> Result<(CudaTensor, CudaTensor), BackendError> {
        let pair_columns = block_columns.checked_mul(2).ok_or_else(|| compute_error("CUDA interleaved split block 溢出"))?;
        if block_columns == 0 || !input.cols.is_multiple_of(pair_columns) {
            return Err(compute_error(format!("CUDA interleaved split cols={} block={block_columns} 非法", input.cols)));
        }
        ops::tensor::split_interleaved_columns_f16(self, input, block_columns).map_err(compute_error)
    }

    fn concat_columns(&self, left: &CudaTensor, right: &CudaTensor) -> Result<CudaTensor, BackendError> {
        if left.rows != right.rows {
            return Err(compute_error(format!("CUDA concat rows {} 与 {} 不一致", left.rows, right.rows)));
        }
        ops::tensor::concat_columns_f16(self, left, right).map_err(compute_error)
    }

    fn rope(&self, input: &CudaTensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<CudaTensor, BackendError> {
        if layout != crate::attention::rope::RotaryLayout::SplitHalf {
            return Err(compute_error("CUDA 暂不支持 interleaved RoPE"));
        }
        ops::tensor::apply_rope_partial_f16(self, input, head_count, rotary_dim, position, cos, sin).map_err(compute_error)
    }

    fn rope_prefix(&self, input: &CudaTensor, head_count: usize, rotary_dim: usize, layout: crate::attention::rope::RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<CudaTensor, BackendError> {
        if layout != crate::attention::rope::RotaryLayout::SplitHalf {
            return Err(compute_error("CUDA 暂不支持 interleaved RoPE"));
        }
        ops::tensor::apply_rope_prefix_f16(self, input, head_count, rotary_dim, position, cos, sin).map_err(compute_error)
    }

    fn add(&self, left: &CudaTensor, right: &CudaTensor) -> Result<CudaTensor, BackendError> {
        ops::tensor::add_f16(self, left, right).map_err(compute_error)
    }

    fn add_scaled(&self, left: &CudaTensor, right: &CudaTensor, scale: f32) -> Result<CudaTensor, BackendError> {
        ops::tensor::add_scaled_f16(self, left, right, scale).map_err(compute_error)
    }

    fn sigmoid_gate(&self, input: &CudaTensor, gate: &CudaTensor) -> Result<CudaTensor, BackendError> {
        ops::attention::sigmoid_gate_f16(self, input, gate).map_err(compute_error)
    }

    fn select_row(&self, input: &CudaTensor, row: usize) -> Result<CudaTensor, BackendError> {
        if input.slice_f32.is_some() {
            return self.select_rows(input, &[u32::try_from(row).map_err(|_| compute_error("CUDA select_row index 超过 u32"))?]);
        }
        ops::tensor::select_row_f16(self, input, row).map_err(compute_error)
    }

    fn select_rows(&self, input: &CudaTensor, rows: &[u32]) -> Result<CudaTensor, BackendError> {
        if rows.is_empty() || rows.iter().any(|&row| row as usize >= input.rows) {
            return Err(compute_error(format!("CUDA select_rows 越界,input_rows={} rows={rows:?}", input.rows)));
        }
        if input.slice_f32.is_some() { ops::routing::gather_rows_f32(self, input, rows).map_err(compute_error) } else { ops::routing::gather_rows_f16(self, input, rows).map_err(compute_error) }
    }

    fn argmax(&self, input: &CudaTensor) -> Result<u32, BackendError> {
        ops::tensor::argmax_f16(self, input).map_err(compute_error)
    }

    fn argmax_excluding(&self, input: &CudaTensor, excluded: &[u32]) -> Result<u32, BackendError> {
        if excluded.is_empty() {
            return ops::tensor::argmax_f16(self, input).map_err(compute_error);
        }
        ops::tensor::argmax_excluding_f16(self, input, excluded).map_err(compute_error)
    }

    fn sample_top_p(&self, _input: &CudaTensor, _temperature: f32, _top_p: f32, _random: f32) -> Result<u32, BackendError> {
        // decode 路径才需要,MVP(prefill-only)暂不支持。
        Err(compute_error("CUDA sample_top_p 暂未实现(prefill-only 阶段)".to_string()))
    }

    fn gated_linear(&self, input: &CudaTensor, gate: &CudaWeight, up: &CudaWeight, activation: &Activation) -> Result<CudaTensor, BackendError> {
        // SiLU + 两个 W4A16 同规格权重 + decode(单 token):融合 gate/up GEMV,共用输入带宽。
        if matches!(activation, Activation::Silu) && input.rows == 1 && gate.rows == up.rows && gate.cols == up.cols && input.cols == gate.cols {
            if let (Some(g), Some(u)) = (&gate.w4a16, &up.w4a16) {
                if g.group_size == u.group_size && g.scale_dtype == u.scale_dtype {
                    return ops::w4a16::gated_silu_gemv_f16(self, input, &g.packed, &g.scales, &u.packed, &u.scales, gate.rows, g.group_size, g.scale_dtype, u.scale_dtype).map_err(compute_error);
                }
            }
        }
        // SiLU + 两个 MLX affine 4-bit 同规格权重 + decode(单 token):融合 gate/up GEMV + silu_mul。
        // gate/up 都是 MLX affine 4-bit,这里直接走设备内核,避免 host 解码 4B 权重。
        if matches!(activation, Activation::Silu) && input.rows == 1 && gate.rows == up.rows && gate.cols == up.cols && input.cols == gate.cols {
            if let (Some(g), Some(u)) = (&gate.mlx_affine, &up.mlx_affine) {
                if g.group_size == u.group_size && g.scale_dtype == u.scale_dtype && g.bits == u.bits {
                    return ops::mlx_affine::gated_silu_gemv_f16(self, input, &g.packed, &g.scales, &g.biases, &u.packed, &u.scales, &u.biases, gate.rows, g.group_size, g.scale_dtype, u.scale_dtype, g.bits).map_err(compute_error);
                }
            }
        }
        // 两个 MLX affine 4-bit 权重 + prefill(多 token):临时反量化后直接做双投影。
        // 临时权重不能进入跨层 cache，否则 64 层模型会长期保留数十 GiB 显存并循环抖动。
        if matches!(activation, Activation::Silu) && input.rows > 1 && gate.cols == up.cols && gate.rows == up.rows && input.cols == gate.cols {
            if let (Some(g), Some(u)) = (&gate.mlx_affine, &up.mlx_affine) {
                if g.group_size == u.group_size && g.scale_dtype == u.scale_dtype && g.bits == u.bits {
                    let gate_f16 = ops::mlx_affine::dequant_f16(self, &g.packed, &g.scales, &g.biases, gate.rows, gate.cols, g.group_size, g.scale_dtype, g.bits).map_err(compute_error)?;
                    let up_f16 = ops::mlx_affine::dequant_f16(self, &u.packed, &u.scales, &u.biases, up.rows, up.cols, u.group_size, u.scale_dtype, u.bits).map_err(compute_error)?;
                    let gate_output = ops::linear::cublas_matmul_f16(self, input, &gate_f16, gate.rows).map_err(compute_error)?;
                    let up_output = ops::linear::cublas_matmul_f16(self, input, &up_f16, up.rows).map_err(compute_error)?;
                    return ops::tensor::silu_mul_f16(self, &gate_output, &up_output).map_err(compute_error);
                }
            }
        }
        // SiLU + 两个 NVFP4 同规格权重 + decode(单 token):融合 gate/up GEMV + silu_mul。
        if matches!(activation, Activation::Silu) && input.rows == 1 && gate.rows == up.rows && gate.cols == up.cols && input.cols == gate.cols {
            if let (Some(g), Some(u)) = (&gate.nvfp4, &up.nvfp4) {
                return ops::nvfp4::gated_silu_gemv_f16(self, input, &g.codes, &g.scales, &u.codes, &u.scales, g.global_scale, u.global_scale, gate.rows).map_err(compute_error);
            }
        }
        // 两个 NVFP4 权重 + prefill(多 token):临时反量化后直接做双投影。
        // 临时权重不能进入跨层 cache,否则会长期保留数十 GiB 显存并循环抖动。
        if matches!(activation, Activation::Silu) && input.rows > 1 && gate.cols == up.cols && gate.rows == up.rows && input.cols == gate.cols {
            if let (Some(g), Some(u)) = (&gate.nvfp4, &up.nvfp4) {
                let gate_f16 = ops::nvfp4::dequant_f16(self, &g.codes, &g.scales, g.global_scale, gate.rows, gate.cols).map_err(compute_error)?;
                let up_f16 = ops::nvfp4::dequant_f16(self, &u.codes, &u.scales, u.global_scale, up.rows, up.cols).map_err(compute_error)?;
                let gate_output = ops::linear::cublas_matmul_f16(self, input, &gate_f16, gate.rows).map_err(compute_error)?;
                let up_output = ops::linear::cublas_matmul_f16(self, input, &up_f16, up.rows).map_err(compute_error)?;
                return ops::tensor::silu_mul_f16(self, &gate_output, &up_output).map_err(compute_error);
            }
        }
        // GGUF Q4_K prefill/decode:packed gate/up 共同解码并融合 SiLU，
        // 不生成反量化权重矩阵，也不保留两份投影输出。
        if matches!(activation, Activation::Silu) && gate.rows == up.rows && gate.cols == up.cols && input.cols == gate.cols {
            if let (Some(g), Some(u)) = (&gate.gguf_packed, &up.gguf_packed)
                && g.tensor_type == 12
                && u.tensor_type == 12
            {
                return ops::linear::gated_linear_q4_k_silu_f16(self, input, &g.codes, &u.codes, gate.rows).map_err(compute_error);
            }
        }
        // 默认语义:dual_linear(两次 linear)+ gated_activation。
        let (gate_out, up_out) = self.dual_linear(input, gate, up)?;
        self.gated_activation(&gate_out, &up_out, activation)
    }

    fn gated_activation(&self, gate: &CudaTensor, up: &CudaTensor, activation: &Activation) -> Result<CudaTensor, BackendError> {
        match activation {
            Activation::Silu => ops::tensor::silu_mul_f16(self, gate, up).map_err(compute_error),
            Activation::SiluClamped { .. } => Err(compute_error("CUDA gated_activation 限幅 SwiGLU 暂未实现".to_string())),
            Activation::Situ { .. } => Err(compute_error("CUDA gated_activation SiTU 暂未实现".to_string())),
            Activation::SwigluOai { .. } => Err(compute_error("CUDA gated_activation SwigluOai 暂未实现".to_string())),
            Activation::GeluTanh => ops::tensor::gelu_tanh_mul_f16(self, gate, up).map_err(compute_error),
        }
    }

    fn segmented_rmsnorm_add_scaled(&self, left: &CudaTensor, right: &CudaTensor, weight: &CudaWeight, segments: usize, segment_columns: usize, eps: f32, scale: f32) -> Result<Vec<CudaTensor>, BackendError> {
        if weight.cols != segment_columns || weight.rows != 1 {
            return Err(compute_error(format!("CUDA segmented_rmsnorm_add_scaled weight=[{},{}]，segment_columns={segment_columns}", weight.rows, weight.cols)));
        }
        ops::tensor::segmented_rmsnorm_add_scaled_f16(self, left, right, &weight.data, segments, segment_columns, eps, scale).map_err(compute_error)
    }

    fn split_gated_activation(&self, input: CudaTensor, left_columns: usize, activation: &Activation) -> Result<CudaTensor, BackendError> {
        // f32 扩散激活(MLP gate_up 来自 f32 linear):融合拆分+SiLU 输出 f32,传播到 down 投影。
        if matches!(activation, Activation::Silu) && input.slice_f32.is_some() {
            return ops::diffusion::split_gated_silu_f32(self, &input, left_columns).map_err(compute_error);
        }
        // 默认:f16 拆分 + 激活(LLM 与 f16 扩散路径)。
        let (gate, up) = self.split_columns(&input, left_columns)?;
        self.gated_activation(&gate, &up, activation)
    }
}
