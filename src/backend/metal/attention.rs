//! Metal Attention、KV cache 与 DSA 能力实现。

use crate::{
    attention::{dsa::DsaSpec, gqa::GqaSpec, mla::MlaSpec},
    backend::{Backend, BackendError, BackendResources, BlockAttentionBackend, DecodeBackend, DsaPrefillBackend, GqaPrefillBackend, LinearWeight, MlaPrefillBackend},
    kernel::metal as ops,
};

use super::{
    api::Buffer,
    context::{MetalContext, MetalTensor, MetalTensorDType},
    dsa, expect_resident_f32,
    kv_cache::MetalKvCache,
    resident::MetalWeight,
};

/// kv_b resident 权重统一转换为 KvBWeight。
/// `allow_w8a16=false` 用于 decode selected 内核：它尚未实现 W8A16 路径，必须拒绝。
fn kv_b_weight(kv_b: &MetalWeight, allow_w8a16: bool) -> Result<ops::mla::KvBWeight<'_>, BackendError> {
    match kv_b {
        MetalWeight::F16(weight) => Ok(ops::mla::KvBWeight::F16Resident(weight)),
        MetalWeight::Fp8 { codes, scale_inv, rows, cols } => Ok(ops::mla::KvBWeight::Fp8Resident { codes, scale_inv, rows: *rows, cols: *cols }),
        MetalWeight::Mxfp8 { .. } => Err(BackendError::Compute { msg: "MLA kv_b 暂不支持 MXFP8 resident".to_owned() }),
        MetalWeight::Mxfp4 { .. } => Err(BackendError::Compute { msg: "MLA kv_b 暂不支持 MXFP4 resident".to_owned() }),
        MetalWeight::Nvfp4 { .. } => Err(BackendError::Compute { msg: "MLA kv_b 不应使用 NVFP4 expert 权重".to_owned() }),
        MetalWeight::Gguf { .. } => Err(BackendError::Compute { msg: "MLA kv_b 暂不支持 GGUF resident".to_owned() }),
        MetalWeight::W4A16 { .. } => Err(BackendError::Compute { msg: "MLA kv_b 暂不支持 W4A16 resident".to_owned() }),
        MetalWeight::W8A16 { packed, scales, scale_dtype, group_size, rows, cols } if allow_w8a16 => Ok(ops::mla::KvBWeight::W8A16Resident { packed, scales, scale_dtype: *scale_dtype, group_size: *group_size, rows: *rows, cols: *cols }),
        MetalWeight::W8A16 { .. } => Err(BackendError::Compute { msg: "MLA kv_b 暂不支持 W8A16 resident".to_owned() }),
        MetalWeight::MlxAffine { .. } => Err(BackendError::Compute { msg: "MLA kv_b 暂不支持 MLX affine resident".to_owned() }),
        MetalWeight::F32 { .. } => Err(BackendError::Compute { msg: "MLA kv_b 不能使用 F32 常量".to_owned() }),
        MetalWeight::Fp8PerTensor { .. } => Err(BackendError::Compute { msg: "MLA kv_b 暂不支持 per-tensor FP8 resident".to_owned() }),
    }
}

impl MlaPrefillBackend for MetalContext {
    fn mla_prefill_attention(&self, query: &MetalTensor, latent: &MetalTensor, k_rope: &MetalTensor, kv_b: &MetalWeight, cache: Option<&mut MetalKvCache>, layer: usize, spec: &MlaSpec) -> Result<MetalTensor, BackendError> {
        if let Some(cache) = cache {
            let kv_b = kv_b_weight(kv_b, true)?;
            cache.append_layer_mla_tensor(self, layer, latent, k_rope).map_err(|msg| BackendError::Compute { msg })?;
            return ops::mla::mla_attention_with_cache_tensor(self, query, cache, kv_b, spec.num_heads, spec.qk_rope_head_dim, layer).map_err(|msg| BackendError::Compute { msg });
        }

        let kv = self.linear(latent, kv_b)?;
        let attention = ops::attention::mla_attention_tensor(self, query, &kv, k_rope, spec.num_heads, spec.qk_rope_head_dim).map_err(|msg| BackendError::Compute { msg })?;
        Ok(attention)
    }
}

impl DecodeBackend for MetalContext {
    type DsaState = dsa::MetalDsaState;

    fn prepare_mla_kv_b(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        match weight {
            // MLA/DSA 已有 Block-FP8/官方 FP8 设备路径，保持压缩态。
            LinearWeight::Quantized(matrix @ (crate::weight::format::quantization::QuantizedMatrixRef::BlockFp8(_) | crate::weight::format::quantization::QuantizedMatrixRef::Fp8(_))) => {
                self.prepare_weight(LinearWeight::Quantized(matrix), rows, cols)
            }
            // 没有原生 attention 算子的量化格式在模型加载阶段拒绝，禁止展开
            // 成 F16 掩盖算子缺口。
            LinearWeight::Quantized(matrix) => Err(BackendError::Compute { msg: format!("Metal MLA KV-B 缺少 {} 原生 attention 算子", matrix.name()) }),
            weight => self.prepare_weight(weight, rows, cols),
        }
    }

    fn append_mla(&self, cache: &mut MetalKvCache, layer: usize, latent: &MetalTensor, rope: &MetalTensor) -> Result<(), BackendError> {
        cache.append_layer_mla_tensor(self, layer, latent, rope).map_err(|msg| BackendError::Compute { msg })
    }

    fn mla_decode_attention(&self, query: &MetalTensor, cache: &MetalKvCache, kv_b: &MetalWeight, layer: usize, position: usize, spec: &MlaSpec) -> Result<MetalTensor, BackendError> {
        let kv_b = kv_b_weight(kv_b, true)?;
        ops::mla::mla_decode_attention(self, query, cache, kv_b, layer, position, spec.num_heads, spec.qk_nope_dim(), spec.qk_rope_head_dim, spec.value_dim()).map_err(|msg| BackendError::Compute { msg })
    }

    fn dsa_can_append(&self, state: &mut dsa::MetalDsaState, layer: usize, position: usize, spec: &DsaSpec) -> bool {
        state.can_update(layer, position, spec.head_dim, spec.top_k)
    }

    fn append_dsa_keys(&self, state: &mut dsa::MetalDsaState, layer: usize, position: usize, keys: &MetalTensor, _spec: &DsaSpec) -> Result<(), BackendError> {
        if position == 0 { state.append_prefill_keys(self, layer, keys).map_err(|msg| BackendError::Compute { msg }) } else { state.append_key(self, layer, position, keys).map_err(|msg| BackendError::Compute { msg }) }
    }

    fn dsa_select_topk(&self, state: &mut dsa::MetalDsaState, layer: usize, query: &MetalTensor, head_weights: &MetalTensor, spec: &DsaSpec) -> Result<(), BackendError> {
        state.select(self, layer, query, head_weights, spec.num_heads).map_err(|msg| BackendError::Compute { msg })
    }

    fn mla_decode_attention_selected(&self, query: &MetalTensor, cache: &MetalKvCache, kv_b: &MetalWeight, layer: usize, position: usize, mla: &MlaSpec, dsa: &DsaSpec, state: &dsa::MetalDsaState) -> Result<MetalTensor, BackendError> {
        if !state.selection_valid() || position <= dsa.top_k {
            return self.mla_decode_attention(query, cache, kv_b, layer, position, mla);
        }
        let kv_b = kv_b_weight(kv_b, false)?;
        ops::mla::mla_decode_attention_selected(self, query, cache, kv_b, layer, position, mla.num_heads, mla.qk_nope_dim(), mla.qk_rope_head_dim, mla.value_dim(), state.selection(), state.top_k())
            .map_err(|msg| BackendError::Compute { msg })
    }
}

#[cfg(test)]
mod quantized_kv_b_tests {
    use super::*;
    use crate::weight::format::{
        mxfp8::Mxfp8Matrix,
        quantization::{Fp8Matrix, ScaleDType, W4A16Matrix},
    };

    #[test]
    fn unsupported_native_kv_b_formats_are_rejected_during_prepare() {
        let context = MetalContext::new_default().unwrap();
        let (rows, cols) = (2usize, 32usize);
        let w4 = W4A16Matrix::new(vec![0; rows * cols.div_ceil(8) * 4], vec![0; rows * 2], ScaleDType::F16, cols, rows, cols).unwrap();
        let mxfp8 = Mxfp8Matrix::new(vec![0; rows * cols], vec![127; rows], rows, cols).unwrap();

        for weight in [LinearWeight::w4a16(&w4), LinearWeight::mxfp8(&mxfp8)] {
            let Err(error) = DecodeBackend::prepare_mla_kv_b(&context, weight, rows, cols) else {
                panic!("缺少原生 MLA 算子的量化格式不应装配成功");
            };
            assert!(error.to_string().contains("缺少") && error.to_string().contains("原生 attention 算子"));
        }
    }

    #[test]
    fn native_fp8_kv_b_stays_quantized() {
        let context = MetalContext::new_default().unwrap();
        let (rows, cols) = (2usize, 32usize);
        let fp8 = Fp8Matrix::new(vec![0; rows * cols], vec![0; 4], rows, cols).unwrap();
        let prepared = DecodeBackend::prepare_mla_kv_b(&context, LinearWeight::fp8(&fp8), rows, cols).unwrap();
        assert!(matches!(prepared, MetalWeight::Fp8 { rows: 2, cols: 32, .. }));
    }
}

impl DsaPrefillBackend for MetalContext {
    fn dsa_select_prefill(&self, state: &mut dsa::MetalDsaState, layer: usize, query: &MetalTensor, head_weights: &MetalTensor, spec: &DsaSpec) -> Result<(), BackendError> {
        state.select_prefill(self, layer, query, head_weights, spec.num_heads).map_err(|msg| BackendError::Compute { msg })
    }

    fn mla_prefill_attention_selected(
        &self,
        query: &MetalTensor,
        latent: &MetalTensor,
        k_rope: &MetalTensor,
        kv_b: &MetalWeight,
        cache: Option<&mut MetalKvCache>,
        layer: usize,
        mla: &MlaSpec,
        dsa: &DsaSpec,
        state: &dsa::MetalDsaState,
    ) -> Result<MetalTensor, BackendError> {
        let Some(selection) = state.prefill_selection(query.rows) else {
            return self.mla_prefill_attention(query, latent, k_rope, kv_b, cache, layer, mla);
        };
        if let Some(cache) = cache {
            cache.append_layer_mla_tensor(self, layer, latent, k_rope).map_err(|msg| BackendError::Compute { msg })?;
        }
        match kv_b {
            MetalWeight::F16(weight) => ops::mla::mla_prefill_attention_selected_f16(self, query, latent, k_rope, weight, selection, state.top_k(), mla).map_err(|msg| BackendError::Compute { msg }),
            MetalWeight::Fp8 { codes, scale_inv, rows, cols } => {
                ops::mla::mla_prefill_attention_selected_fp8(self, query, latent, k_rope, codes, scale_inv, *rows, *cols, selection, state.top_k(), mla).map_err(|msg| BackendError::Compute { msg })
            }
            MetalWeight::Mxfp8 { .. } => Err(BackendError::Compute { msg: "DSA prefill kv_b 暂不支持 MXFP8 resident".to_owned() }),
            MetalWeight::Mxfp4 { .. } => Err(BackendError::Compute { msg: "DSA prefill kv_b 暂不支持 MXFP4 resident".to_owned() }),
            MetalWeight::Nvfp4 { .. } => Err(BackendError::Compute { msg: "DSA prefill kv_b 不应使用 NVFP4 expert 权重".to_owned() }),
            MetalWeight::Gguf { .. } => Err(BackendError::Compute { msg: "DSA prefill kv_b 暂不支持 GGUF resident".to_owned() }),
            MetalWeight::W4A16 { .. } => Err(BackendError::Compute { msg: "DSA prefill kv_b 暂不支持 W4A16 resident".to_owned() }),
            MetalWeight::W8A16 { .. } => Err(BackendError::Compute { msg: "DSA prefill kv_b 暂不支持 W8A16 resident".to_owned() }),
            MetalWeight::MlxAffine { .. } => Err(BackendError::Compute { msg: "DSA prefill kv_b 暂不支持 MLX affine resident".to_owned() }),
            MetalWeight::F32 { .. } => Err(BackendError::Compute { msg: format!("DSA prefill kv_b 不能使用 F32 常量，top_k={}", dsa.top_k) }),
            MetalWeight::Fp8PerTensor { .. } => Err(BackendError::Compute { msg: "DSA prefill kv_b 暂不支持 per-tensor FP8 resident".to_owned() }),
        }
    }
}

impl BlockAttentionBackend for MetalContext {
    fn block_attention(&self, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, spec: &crate::attention::block::BlockAttentionSpec) -> Result<MetalTensor, BackendError> {
        ops::attention::block_attention_tensor(self, query, key, value, spec).map_err(|msg| BackendError::Compute { msg })
    }
}

/// 逐头 RMSNorm 的公共实现:`offset=0` 为普通 RMSNorm,`1.0` 为 GemmaRMSNorm
/// (kernel 用同一参数表达 gamma+1 编码)。
fn head_rmsnorm(ctx: &MetalContext, input: &MetalTensor, weight: &MetalWeight, head_count: usize, head_dim: usize, eps: f32, offset: f32, name: &str) -> Result<MetalTensor, BackendError> {
    let expected_columns = head_count.checked_mul(head_dim).ok_or_else(|| BackendError::Compute { msg: format!("{name} 维度溢出") })?;
    if input.cols != expected_columns {
        return Err(BackendError::Compute { msg: format!("{name} 输入列 {} 与 {head_count}x{head_dim} 不符", input.cols) });
    }
    let output = match weight {
        MetalWeight::F16(weight) => {
            let heads = input.reshape(input.rows * head_count, head_dim);
            ops::tensor::rmsnorm_tensor_resident_weight(ctx, &heads, weight, eps, offset).map_err(|msg| BackendError::Compute { msg })?
        }
        weight => {
            let (buffer, len) = expect_resident_f32(weight, "逐头 RMSNorm 需要 resident F16/F32 权重")?;
            if len != head_dim {
                return Err(BackendError::Compute { msg: format!("{name} F32 权重长度 {len}，期望 {head_dim}") });
            }
            let heads = input.reshape(input.rows * head_count, head_dim);
            if heads.dtype == MetalTensorDType::F16 {
                // F16 输入直读 F32 权重单 kernel:免掉两侧 f16↔f32 cast,数值与 cast 链一致
                ops::tensor::rmsnorm_f16_in_f32_weight_tensor_resident(ctx, &heads, buffer, len, eps, offset).map_err(|msg| BackendError::Compute { msg })?
            } else {
                let heads = ops::to_f32_tensor(ctx, &heads).map_err(|msg| BackendError::Compute { msg })?;
                let output = ops::tensor::rmsnorm_tensor_resident_f32_weight(ctx, &heads, buffer, len, eps, offset).map_err(|msg| BackendError::Compute { msg })?;
                ops::to_f16_tensor(ctx, &output).map_err(|msg| BackendError::Compute { msg })?
            }
        }
    };
    Ok(output.reshape(input.rows, input.cols))
}

fn rmsnorm_extract_affine<'a>(w: &'a MetalWeight, name: &str) -> Result<(&'a crate::backend::metal::api::Buffer, &'a crate::backend::metal::api::Buffer, &'a crate::backend::metal::api::Buffer, u32, usize, usize, usize), BackendError> {
    match w {
        MetalWeight::MlxAffine { packed, scales, biases, scale_dtype, bits, group_size, rows, cols, pair_id, .. } if *bits == 4 && *pair_id == 0 => Ok((packed, scales, biases, *scale_dtype, *group_size, *rows, *cols)),
        _ => Err(BackendError::Compute { msg: format!("rmsnorm 融合 {name} 需要 MlxAffine u4 权重") }),
    }
}

impl MetalContext {
    /// 融合 rmsnorm kernel 可直接消费的权重:未配对的 u4 MlxAffine。
    fn is_unpaired_u4(w: &MetalWeight) -> bool {
        matches!(w, MetalWeight::MlxAffine { bits: 4, pair_id: 0, .. })
    }
}

impl GqaPrefillBackend for MetalContext {
    fn rmsnorm_heads(&self, input: &MetalTensor, weight: &MetalWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<MetalTensor, BackendError> {
        head_rmsnorm(self, input, weight, head_count, head_dim, eps, 0.0, "逐头 RMSNorm")
    }

    fn gemma_rmsnorm_heads(&self, input: &MetalTensor, weight: &MetalWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<MetalTensor, BackendError> {
        head_rmsnorm(self, input, weight, head_count, head_dim, eps, 1.0, "逐头 GemmaRMSNorm")
    }

    fn gemma_rmsnorm_heads_f32(&self, input: &MetalTensor, weight: &MetalWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<MetalTensor, BackendError> {
        let expected_columns = head_count.checked_mul(head_dim).ok_or_else(|| BackendError::Compute { msg: "逐头 GemmaRMSNorm F32 维度溢出".to_owned() })?;
        if input.cols != expected_columns {
            return Err(BackendError::Compute { msg: format!("逐头 GemmaRMSNorm F32 输入列 {} 与 {head_count}x{head_dim} 不符", input.cols) });
        }
        // F16 权重走直通版(kernel 内 F32 累加),保持 F16 数据流
        if matches!(weight, MetalWeight::F16(_)) {
            return self.gemma_rmsnorm_heads(input, weight, head_count, head_dim, eps);
        }
        let input = ops::to_f32_tensor(self, input).map_err(|msg| BackendError::Compute { msg })?;
        let heads = input.reshape(input.rows * head_count, head_dim);
        let output = self.gemma_rmsnorm_f32(&heads, weight, eps)?;
        Ok(output.reshape(input.rows, input.cols))
    }

    fn gqa_prefill_attention(&self, query: &MetalTensor, key: &MetalTensor, value: &MetalTensor, spec: &GqaSpec) -> Result<MetalTensor, BackendError> {
        let output = ops::attention::gqa_prefill_attention_tensor(self, query, key, value, spec).map_err(|msg| BackendError::Compute { msg })?;
        Ok(output)
    }

    fn gqa_prefill_attention_cached(
        &self,
        cache: &mut MetalKvCache,
        layer: usize,
        position: usize,
        query: &MetalTensor,
        key: &MetalTensor,
        value: &MetalTensor,
        spec: &GqaSpec,
        _retain_full_cache: bool,
    ) -> Result<MetalTensor, BackendError> {
        if key.dtype != value.dtype || !matches!(key.dtype, MetalTensorDType::F16 | MetalTensorDType::Bf16) || (query.dtype != key.dtype && query.dtype != MetalTensorDType::F32) {
            return Err(BackendError::Compute { msg: format!("hybrid GQA dtype 不一致: Q={:?} K={:?} V={:?}", query.dtype, key.dtype, value.dtype) });
        }
        if (cache.is_hybrid_gqa() || cache.format() == crate::kv_cache::KvCacheFormat::Int8) && position == 0 && query.rows > 1 {
            let output = ops::attention::gqa_prefill_attention_tensor(self, query, key, value, spec).map_err(|msg| BackendError::Compute { msg })?;
            cache.append_layer_gqa_tensor(self, layer, position, key, value).map_err(|msg| BackendError::Compute { msg })?;
            return Ok(output);
        }
        // Int8 单行 decode 的小 KV 融合直通:量化+attention 一次 dispatch,省 kv_quantize 往返。
        if cache.format() == crate::kv_cache::KvCacheFormat::Int8 && !cache.is_hybrid_gqa() && query.rows == 1 {
            let view = cache.gqa_layer_view(layer).map_err(|msg| BackendError::Compute { msg })?;
            let first_visible = match spec.window {
                crate::attention::gqa::CausalWindow::Full => view.start,
                crate::attention::gqa::CausalWindow::Sliding { size } => view.start.max((position + 1).saturating_sub(size)),
            };
            let supported =
                view.capacity == 0 && view.rows == position && position + 1 - first_visible <= 512 && spec.head_dim <= 128 && spec.head_dim.is_multiple_of(32) && view.group_size.is_multiple_of(4) && spec.num_kv_heads * spec.head_dim <= 512;
            if supported {
                cache.reserve_layer_gqa_row(layer, position).map_err(|msg| BackendError::Compute { msg })?;
                return ops::attention::gqa_decode_attention_append_direct_q8_tensor(self, query, key, value, &view, position, spec).map_err(|msg| BackendError::Compute { msg });
            }
        }
        cache.append_layer_gqa_tensor(self, layer, position, key, value).map_err(|msg| BackendError::Compute { msg })?;
        let view = cache.gqa_layer_view(layer).map_err(|msg| BackendError::Compute { msg })?;
        ops::attention::gqa_prefill_attention_cached_tensor(self, query, &view, position, spec).map_err(|msg| BackendError::Compute { msg })
    }

    fn gqa_prefill_attention_cached_visible(
        &self,
        cache: &mut MetalKvCache,
        layer: usize,
        position: usize,
        query: &MetalTensor,
        key: &MetalTensor,
        value: &MetalTensor,
        spec: &GqaSpec,
        visible_ends: &[u32],
        _retain_full_cache: bool,
    ) -> Result<MetalTensor, BackendError> {
        cache.append_layer_gqa_tensor(self, layer, position, key, value).map_err(|msg| BackendError::Compute { msg })?;
        let view = cache.gqa_layer_view(layer).map_err(|msg| BackendError::Compute { msg })?;
        ops::attention::gqa_prefill_attention_cached_visible_tensor(self, query, &view, position, spec, visible_ends).map_err(|msg| BackendError::Compute { msg })
    }

    fn gqa_prefill_attention_cached_from(&self, cache: &MetalKvCache, source_layer: usize, position: usize, query: &MetalTensor, spec: &GqaSpec) -> Result<MetalTensor, BackendError> {
        let view = cache.gqa_layer_view(source_layer).map_err(|msg| BackendError::Compute { msg })?;
        let required_rows = position.checked_add(query.rows).ok_or_else(|| BackendError::Compute { msg: "GQA shared cache position 溢出".to_owned() })?;
        if view.rows < required_rows {
            return Err(BackendError::Compute { msg: format!("GQA shared cache L{source_layer} rows={}，需要 {required_rows}", view.rows) });
        }
        ops::attention::gqa_prefill_attention_cached_tensor(self, query, &view, position, spec).map_err(|msg| BackendError::Compute { msg })
    }

    fn fused_rmsnorm_qkv_head_norm_rope(
        &self,
        hidden: &MetalTensor,
        input_norm: &MetalWeight,
        query: &MetalWeight,
        key: &MetalWeight,
        value: &MetalWeight,
        query_norm: &MetalWeight,
        key_norm: &MetalWeight,
        cos: &[f32],
        sin: &[f32],
        head_count: usize,
        kv_head_count: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        eps: f32,
    ) -> Result<(MetalTensor, MetalTensor, MetalTensor), BackendError> {
        let resident = |weight: &MetalWeight| -> Result<ops::fused::MlxAffineResident, BackendError> {
            match weight {
                MetalWeight::MlxAffine { packed, scales, biases, scale_dtype, bits, group_size, rows, cols, pair_id, .. } if *pair_id == 0 => {
                    Ok(ops::fused::MlxAffineResident::new(packed.clone(), scales.clone(), biases.clone(), *scale_dtype, *bits, *group_size, *rows, *cols))
                }
                _ => Err(BackendError::Compute { msg: "融合 QKV 需要标准布局 MLX affine 权重".to_owned() }),
            }
        };
        let query_weight = resident(query)?;
        let key_weight = resident(key)?;
        let value_weight = resident(value)?;
        fn expect_norm<'a>(weight: &'a MetalWeight, name: &str, head_dim: usize) -> Result<&'a Buffer, BackendError> {
            let (buffer, len) = expect_resident_f32(weight, "融合 QKV head norm 权重需要 F32 布局")?;
            if len != head_dim {
                return Err(BackendError::Compute { msg: format!("融合 QKV {name} norm 权重长度 {len}，期望 {head_dim}") });
            }
            Ok(buffer)
        }
        // input_norm 与生产顺序路径一致按 F16 准备;融合 kernel 以 half 直读,
        // 保证 norm 权重的舍入语义与顺序路径逐位一致
        let input_norm = match input_norm {
            MetalWeight::F16(tensor) if tensor.cols == hidden.cols => &tensor.buffer,
            MetalWeight::F16(tensor) => {
                return Err(BackendError::Compute { msg: format!("融合 QKV input norm 长度 {}，期望 {}", tensor.cols, hidden.cols) });
            }
            _ => return Err(BackendError::Compute { msg: "融合 QKV input norm 权重需要 F16 布局".to_owned() }),
        };
        let query_norm = expect_norm(query_norm, "query", head_dim)?;
        let key_norm = expect_norm(key_norm, "key", head_dim)?;
        let rotary_half = rotary_dim / 2;
        let (cos_table, sin_table) = self.decode_rope_table_buffers(cos, sin, rotary_half).map_err(|msg| BackendError::Compute { msg })?;
        let rope_offset = (position * rotary_half * 2) as u64;
        let (query, key, value) =
            ops::fused::fused_rmsnorm_qkv_head_norm_rope(self, hidden, input_norm, &query_weight, &key_weight, &value_weight, query_norm, key_norm, &cos_table, &sin_table, rope_offset, head_count, kv_head_count, head_dim, rotary_dim, eps)
                .map_err(|msg| BackendError::Compute { msg })?;
        Ok((query, key, value))
    }

    fn qkv_head_norms(
        &self,
        query: &MetalTensor,
        key: &MetalTensor,
        value: &MetalTensor,
        query_norm: &MetalWeight,
        key_norm: &MetalWeight,
        value_norm: &MetalWeight,
        head_count: usize,
        kv_head_count: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(MetalTensor, MetalTensor, MetalTensor), BackendError> {
        fn expect_head_norm<'a>(w: &'a MetalWeight, n: &str, d: usize) -> Result<&'a crate::backend::metal::api::Buffer, BackendError> {
            let (buffer, len) = expect_resident_f32(w, "QKV 融合 head norm 需要 F32 布局")?;
            if len != d {
                return Err(BackendError::Compute { msg: format!("QKV 融合 head norm {n} 长度 {len}，期望 {d}") });
            }
            Ok(buffer)
        }
        let qn = expect_head_norm(query_norm, "query", head_dim)?;
        let kn = expect_head_norm(key_norm, "key", head_dim)?;
        let vn = expect_head_norm(value_norm, "value", head_dim)?;
        ops::tensor::rmsnorm_f16_heads_triple_encoder(self, query, key, value, qn, kn, vn, head_count, kv_head_count, head_dim, eps).map_err(|msg| BackendError::Compute { msg })
    }

    fn rmsnorm_triple_linear(&self, input: &MetalTensor, norm_weight: &MetalWeight, eps: f32, first: &MetalWeight, second: &MetalWeight, third: &MetalWeight) -> Result<(MetalTensor, MetalTensor, MetalTensor), BackendError> {
        // F16 norm 权重(rms_norm_f16_simd 要求)
        let MetalWeight::F16(norm_tensor) = norm_weight else {
            return Err(BackendError::Compute { msg: "rmsnorm_triple 需要 F16 norm 权重".to_owned() });
        };
        // 融合 kernel 只吃未配对的 u4 MlxAffine;混合精度层(如 8-bit 成对交错布局)
        // 回退 rmsnorm + triple_linear 通用路径,不能直接拒绝。
        if !(Self::is_unpaired_u4(first) && Self::is_unpaired_u4(second) && Self::is_unpaired_u4(third)) {
            let normed = self.rmsnorm(input, norm_weight, eps)?;
            return self.triple_linear(&normed, first, second, third);
        }
        // 三路 MlxAffine 权重
        let (fp, fs, fb, sd, gs, fr, fc) = rmsnorm_extract_affine(first, "first")?;
        let (sp, ss, sb, _, _, sr, _) = rmsnorm_extract_affine(second, "second")?;
        let (tp, ts, tb, _, _, tr, _) = rmsnorm_extract_affine(third, "third")?;
        // 混合精度层(上游 8-bit gemv 输出 Bf16)由融合路径的 bf16 直读 norm kernel
        // 原生消费,不再需要整行 cast 到 F16。
        if input.rows != 1 || !matches!(input.dtype, MetalTensorDType::F16 | MetalTensorDType::Bf16) || input.cols != fc {
            return Err(BackendError::Compute { msg: format!("rmsnorm_triple input=[{},{}] fc={fc} dtype={:?} 不兼容", input.rows, input.cols, input.dtype) });
        }
        ops::mlx::mlx_affine_rmsnorm_triple_gemv_f16_u4(self, input, &norm_tensor.buffer, eps, fp, fs, fb, fr, sp, ss, sb, sr, tp, ts, tb, tr, sd, gs, fc).map_err(|msg| BackendError::Compute { msg })
    }

    fn rmsnorm_gated_linear(&self, input: &MetalTensor, norm_weight: &MetalWeight, eps: f32, gate: &MetalWeight, up: &MetalWeight, activation: &crate::moe::Activation) -> Result<MetalTensor, BackendError> {
        let MetalWeight::F16(norm_tensor) = norm_weight else {
            return Err(BackendError::Compute { msg: "rmsnorm_gated 需要 F16 norm 权重".to_owned() });
        };
        // 同上:gate/up 非未配对 u4(如 8-bit 交错 pair)时回退通用路径。
        if !(Self::is_unpaired_u4(gate) && Self::is_unpaired_u4(up)) {
            let normed = self.rmsnorm(input, norm_weight, eps)?;
            return self.gated_linear(&normed, gate, up, activation);
        }
        let (gp, gs, gb, sd, grp, gr, gc) = rmsnorm_extract_affine(gate, "gate")?;
        let (up_p, up_s, up_b, _, _, ur, _) = rmsnorm_extract_affine(up, "up")?;
        // 同 rmsnorm_triple:Bf16 输入由融合路径的 bf16 直读 norm kernel 原生消费。
        if input.rows != 1 || !matches!(input.dtype, MetalTensorDType::F16 | MetalTensorDType::Bf16) || input.cols != gc || gr != ur {
            return Err(BackendError::Compute { msg: format!("rmsnorm_gated input=[{},{}] dtype={:?} 不兼容", input.rows, input.cols, input.dtype) });
        }
        ops::mlx::mlx_affine_rmsnorm_gated_gemv_f16_u4(self, input, &norm_tensor.buffer, eps, gp, gs, gb, up_p, up_s, up_b, sd, grp, gr, gc, activation).map_err(|msg| BackendError::Compute { msg })
    }

    fn gqa_prefill_attention_cached_from_visible(&self, cache: &MetalKvCache, source_layer: usize, position: usize, query: &MetalTensor, spec: &GqaSpec, visible_ends: &[u32]) -> Result<MetalTensor, BackendError> {
        let view = cache.gqa_layer_view(source_layer).map_err(|msg| BackendError::Compute { msg })?;
        ops::attention::gqa_prefill_attention_cached_visible_tensor(self, query, &view, position, spec, visible_ends).map_err(|msg| BackendError::Compute { msg })
    }
}
