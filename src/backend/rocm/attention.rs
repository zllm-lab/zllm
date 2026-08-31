use super::*;

static DSA_FUSED_PROLOGUE_PROFILE_DIAGNOSTIC: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn supports_dsa_fused_prologue(state: &RocmDsaState, keys: &RocmTensor, norm_weight: &RocmWeight, norm_bias: &RocmWeight) -> bool {
    state.supports_layernorm_rope(keys.rows, keys.cols)
        && keys.dtype == RocmTensorDType::F32
        && keys.layout == RocmTensorLayout::RowMajor
        && !norm_weight.resident_bf16()
        && !norm_bias.resident_bf16()
        && norm_weight.data().len() == keys.cols
        && norm_bias.data().len() == keys.cols
        && norm_weight.resident().is_some()
        && norm_bias.resident().is_some()
}

fn mla_output_element_bytes(query_rows: usize) -> usize {
    if query_rows > 1 { std::mem::size_of::<u16>() } else { std::mem::size_of::<f32>() }
}

// decode 写 F32，prefill 写 BF16；分段输出共用 buffer 时必须采用一致元素宽度。
fn common_mla_output_element_bytes(query_rows: impl IntoIterator<Item = usize>) -> Option<usize> {
    let mut query_rows = query_rows.into_iter();
    let element_bytes = mla_output_element_bytes(query_rows.next()?);
    query_rows.all(|rows| mla_output_element_bytes(rows) == element_bytes).then_some(element_bytes)
}

impl BlockAttentionBackend for RocmContext {
    fn block_attention(&self, query: &RocmTensor, key: &RocmTensor, value: &RocmTensor, spec: &crate::attention::block::BlockAttentionSpec) -> Result<RocmTensor, BackendError> {
        spec.validate(query.rows, key.rows).map_err(compute_error)?;
        let query_cols = spec.geometry.query_columns().map_err(compute_error)?;
        let kv_cols = spec.geometry.kv_columns().map_err(compute_error)?;
        if query.cols != query_cols || key.rows != value.rows || key.cols != kv_cols || value.cols != kv_cols {
            return Err(compute_error(format!("ROCm block attention shape 异常: Q=[{},{}] K=[{},{}] V=[{},{}]", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols)));
        }
        let query = self.tensor_as_f32(query.clone())?;
        let key = self.tensor_as_f32(key.clone())?;
        let value = self.tensor_as_f32(value.clone())?;
        let visible = spec.visible.iter().flat_map(|range| [u32::try_from(range.start), u32::try_from(range.end)]).collect::<Result<Vec<_>, _>>().map_err(|_| compute_error("ROCm block attention 可见区间超过 u32"))?;
        let output = ops::hip::try_block_attention_resident_f32(
            self.device_id,
            query.device.as_deref().ok_or_else(|| compute_error("ROCm block attention query 缺少 resident buffer"))?,
            key.device.as_deref().ok_or_else(|| compute_error("ROCm block attention key 缺少 resident buffer"))?,
            value.device.as_deref().ok_or_else(|| compute_error("ROCm block attention value 缺少 resident buffer"))?,
            &visible,
            query.rows,
            key.rows,
            spec.geometry.num_heads,
            spec.geometry.num_kv_heads,
            spec.geometry.head_dim,
            spec.score_scale,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, query.rows, query.cols))
    }

    fn block_attention_prefix_suffix(
        &self,
        query: &RocmTensor,
        prefix_key: &RocmTensor,
        prefix_value: &RocmTensor,
        suffix_key: &RocmTensor,
        suffix_value: &RocmTensor,
        spec: &crate::attention::block::BlockAttentionSpec,
    ) -> Result<RocmTensor, BackendError> {
        let prefix_rows = prefix_key.rows;
        let suffix_rows = suffix_key.rows;
        spec.validate(query.rows, prefix_rows + suffix_rows).map_err(compute_error)?;
        let query_cols = spec.geometry.query_columns().map_err(compute_error)?;
        let kv_cols = spec.geometry.kv_columns().map_err(compute_error)?;
        if query.cols != query_cols || prefix_key.cols != kv_cols || prefix_value.rows != prefix_rows || prefix_value.cols != kv_cols || suffix_key.cols != kv_cols || suffix_value.rows != suffix_rows || suffix_value.cols != kv_cols {
            return Err(compute_error("ROCm split block attention shape 异常"));
        }
        let query = self.tensor_as_f32(query.clone())?;
        let prefix_key = self.tensor_as_f32(prefix_key.clone())?;
        let prefix_value = self.tensor_as_f32(prefix_value.clone())?;
        let suffix_key = self.tensor_as_f32(suffix_key.clone())?;
        let suffix_value = self.tensor_as_f32(suffix_value.clone())?;
        let visible = spec.visible.iter().flat_map(|range| [u32::try_from(range.start), u32::try_from(range.end)]).collect::<Result<Vec<_>, _>>().map_err(|_| compute_error("ROCm split block attention 可见区间超过 u32"))?;
        fn resident<'a>(tensor: &'a RocmTensor, name: &str) -> Result<&'a ops::hip::DeviceBuffer, BackendError> {
            tensor.device.as_deref().ok_or_else(|| compute_error(format!("ROCm split block attention {name} 缺少 resident buffer")))
        }
        let output = ops::hip::try_block_attention_prefix_suffix_resident_f32(
            self.device_id,
            resident(&query, "query")?,
            resident(&prefix_key, "prefix key")?,
            resident(&prefix_value, "prefix value")?,
            resident(&suffix_key, "suffix key")?,
            resident(&suffix_value, "suffix value")?,
            &visible,
            query.rows,
            prefix_rows,
            suffix_rows,
            spec.geometry.num_heads,
            spec.geometry.num_kv_heads,
            spec.geometry.head_dim,
            spec.score_scale,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, query.rows, query.cols))
    }
}

fn ct_mla_weight(weight: &RocmWeight) -> Result<ops::hip::CtMlaWeightRef<'_>, BackendError> {
    let (packed, scales, scale_dtype, group_size, bits) = match weight.quantized() {
        Some(RocmQuantizedWeight::W4A16 { packed, scales, scale_dtype, group_size }) => (packed.as_ref(), scales.as_ref(), *scale_dtype, *group_size, 4),
        Some(RocmQuantizedWeight::W8A16 { packed, scales, scale_dtype, group_size }) => (packed.as_ref(), scales.as_ref(), *scale_dtype, *group_size, 8),
        Some(RocmQuantizedWeight::BlockFp8 { .. }) => return Err(compute_error("ROCm paged MLA kv_b 不支持 BlockFp8 量化")),
        Some(RocmQuantizedWeight::ConvRotInt8 { .. }) => return Err(compute_error("ROCm paged MLA kv_b 不支持 ConvRot 量化")),
        Some(RocmQuantizedWeight::GgufPacked { .. }) => return Err(compute_error("ROCm paged MLA kv_b 不支持 GGUF expert 预载布局")),
        Some(RocmQuantizedWeight::Mxfp4 { .. }) => return Err(compute_error("ROCm paged MLA kv_b 不支持 MXFP4 布局")),
        None => {
            let resident = weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm paged MLA kv_b 缺少 resident 权重"))?;
            if weight.resident_bf16() { (resident, resident, ScaleDType::Bf16, weight.cols, 16) } else { (resident, resident, ScaleDType::F32, weight.cols, 32) }
        }
    };
    let scale_dtype = match scale_dtype {
        ScaleDType::Bf16 => 0,
        ScaleDType::F16 => 1,
        ScaleDType::F32 => 2,
    };
    Ok(ops::hip::CtMlaWeightRef { packed, scales, rows: weight.rows, cols: weight.cols, group_size, scale_dtype, bits })
}

fn paged_mla_attention_into(
    context: &RocmContext,
    query: &RocmTensor,
    cache: &RocmKvCache,
    kv_b: &RocmWeight,
    layer: usize,
    spec: &crate::attention::mla::MlaSpec,
    dsa_state: Option<&RocmDsaState>,
    top_k: usize,
    output: &ops::hip::DeviceBuffer,
) -> Result<(), BackendError> {
    let cached = cache.paged_layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} ROCm paged MLA cache 尚未初始化")))?;
    if query.rows == 0 || query.rows > cached.rows {
        return Err(compute_error(format!("L{layer} ROCm paged MLA query rows={}，cache rows={}", query.rows, cached.rows,)));
    }
    let query_device = query.device.as_deref().ok_or_else(|| compute_error("ROCm paged MLA query 缺少 device buffer"))?;
    let table = cache.block_table.buffer().ok_or_else(|| compute_error("ROCm paged MLA 缺少 block table"))?;
    let query_start = cached.rows - query.rows;
    if let Some(hot) = &cached.cpu_hot {
        if query.rows != 1 || cached.committed_rows <= 2048 {
            return Err(compute_error(format!("L{layer} ROCm MLA hot query_rows={} hot_rows={} 非法", query.rows, cached.committed_rows)));
        }
        let state = dsa_state.ok_or_else(|| compute_error(format!("L{layer} ROCm MLA hot 缺少 DSA selection")))?;
        let host_selection = state.host_selection(1, query_start).ok_or_else(|| compute_error(format!("L{layer} ROCm MLA hot 缺少 host selection: start={query_start}")))?;
        let mut hot = hot.lock().map_err(|_| compute_error(format!("L{layer} ROCm MLA hot 锁中毒")))?;
        let selection = hot.prepare_selection(context.device_id, layer, host_selection, &cached.latent, cached.latent_scales.as_deref().expect("Q8 MLA hot 必有 scales"), &cached.rope)?;
        ops::hip::try_paged_mla_attention_ct_into(
            context.device_id,
            query_device,
            &cached.latent,
            cached.latent_scales.as_deref(),
            cached.latent_group_size,
            &cached.rope,
            table,
            Some(&selection),
            ct_mla_weight(kv_b)?,
            1,
            cached.committed_rows,
            cached.committed_rows - 1,
            spec.q_projection_size,
            spec.num_heads,
            spec.qk_rope_head_dim,
            host_selection.len(),
            ROCM_KV_BLOCK_SIZE,
            output,
            None,
        )
        .map_err(compute_error)?;
        if ops::hip::options().kernel_sync {
            ops::hip::synchronize_device(context.device_id, &format!("L{layer} ROCm paged MLA hot synchronize")).map_err(compute_error)?;
        }
        return Ok(());
    }
    let selection = dsa_state.and_then(|state| state.device_selection(query.rows, query_start));
    let selection_width = if selection.is_some() { dsa_state.map_or(top_k, RocmDsaState::selection_width) } else { top_k };
    ops::hip::try_paged_mla_attention_ct_into(
        context.device_id,
        query_device,
        &cached.latent,
        cached.latent_scales.as_deref(),
        cached.latent_group_size,
        &cached.rope,
        table,
        selection,
        ct_mla_weight(kv_b)?,
        query.rows,
        cached.rows,
        query_start,
        spec.q_projection_size,
        spec.num_heads,
        spec.qk_rope_head_dim,
        selection_width,
        ROCM_KV_BLOCK_SIZE,
        output,
        None,
    )
    .map_err(compute_error)?;
    if ops::hip::options().kernel_sync {
        ops::hip::synchronize_device(context.device_id, &format!("L{layer} ROCm paged MLA synchronize")).map_err(compute_error)?;
    }
    Ok(())
}

fn paged_mla_attention(
    context: &RocmContext,
    query: &RocmTensor,
    cache: &RocmKvCache,
    kv_b: &RocmWeight,
    layer: usize,
    spec: &crate::attention::mla::MlaSpec,
    dsa_state: Option<&RocmDsaState>,
    top_k: usize,
) -> Result<RocmTensor, BackendError> {
    let element_bytes = mla_output_element_bytes(query.rows);
    let output_bytes = query.rows.checked_mul(spec.q_projection_size).and_then(|elements| elements.checked_mul(element_bytes)).ok_or_else(|| compute_error("ROCm paged MLA output 大小溢出"))?;
    let output = ops::hip::DeviceBuffer::allocate_reusable(context.device_id, output_bytes).map_err(compute_error)?;
    paged_mla_attention_into(context, query, cache, kv_b, layer, spec, dsa_state, top_k, &output)?;
    Ok(device_tensor_with_dtype(output, query.rows, spec.q_projection_size, if query.rows > 1 { RocmTensorDType::Bf16 } else { RocmTensorDType::F32 }))
}

impl MlaPrefillBackend for RocmContext {
    fn mla_prefill_attention(
        &self,
        query: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        kv_b: &Self::Weight,
        cache: Option<&mut Self::Cache>,
        layer: usize,
        spec: &crate::attention::mla::MlaSpec,
    ) -> Result<Self::Tensor, BackendError> {
        let cache = cache.ok_or_else(|| compute_error(format!("L{layer} ROCm MLA prefill 必须提供 paged cache")))?;
        cache.append_mla(self, layer, latent, k_rope)?;
        paged_mla_attention(self, query, cache, kv_b, layer, spec, None, 0)
    }

    fn mla_prefill_attention_rope(
        &self,
        query: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        kv_b: &Self::Weight,
        cache: Option<&mut Self::Cache>,
        layer: usize,
        position: usize,
        cos: &[f32],
        sin: &[f32],
        spec: &crate::attention::mla::MlaSpec,
    ) -> Result<Self::Tensor, BackendError> {
        let cache = cache.ok_or_else(|| compute_error(format!("L{layer} ROCm MLA prefill 必须提供 paged cache")))?;
        <Self as DecodeBackend>::append_mla_rope(self, cache, layer, latent, k_rope, spec.qk_rope_head_dim, spec.rotary_layout, position, cos, sin)?;
        paged_mla_attention(self, query, cache, kv_b, layer, spec, None, 0)
    }
}

impl DecodeBackend for RocmContext {
    type DsaState = RocmDsaState;

    fn prepare_mla_kv_b(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        if let LinearWeight::Quantized(QuantizedMatrixRef::W8A16(matrix)) = weight {
            let mut values = vec![0.0_f32; checked_elements(rows, cols, "ROCm MLA KV-B")?];
            crate::weight::codec::groupwise::decode_w8a16_matrix(matrix.packed(), matrix.scales(), matrix.scale_dtype(), matrix.group_size(), rows, cols, &mut values).map_err(compute_error)?;
            let values = values.into_iter().map(half::f16::from_f32).collect::<Vec<_>>();
            return self.prepare_weight(LinearWeight::F16(&values), rows, cols);
        }
        self.prepare_weight(weight, rows, cols)
    }

    fn append_mla(&self, cache: &mut Self::Cache, layer: usize, latent: &Self::Tensor, rope: &Self::Tensor) -> Result<(), BackendError> {
        cache.append_mla(self, layer, latent, rope)
    }

    fn append_mla_rope(
        &self,
        cache: &mut Self::Cache,
        layer: usize,
        latent: &Self::Tensor,
        rope: &Self::Tensor,
        rotary_dim: usize,
        layout: crate::attention::rope::RotaryLayout,
        position: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(), BackendError> {
        if ops::hip::options().kv_f16 {
            let rope = self.rope(rope, 1, rotary_dim, layout, position, cos, sin)?;
            cache.append_mla(self, layer, latent, &rope)
        } else {
            cache.append_mla_rope(self, layer, latent, rope, rotary_dim, layout, position, cos, sin)
        }
    }

    fn mla_decode_attention(&self, query: &Self::Tensor, cache: &Self::Cache, kv_b: &Self::Weight, layer: usize, position: usize, spec: &crate::attention::mla::MlaSpec) -> Result<Self::Tensor, BackendError> {
        let _ = position;
        paged_mla_attention(self, query, cache, kv_b, layer, spec, None, 0)
    }

    fn dsa_can_append(&self, state: &mut Self::DsaState, layer: usize, position: usize, spec: &crate::attention::dsa::DsaSpec) -> bool {
        let _ = spec;
        state.can_append(layer, position)
    }

    fn append_dsa_keys(&self, state: &mut Self::DsaState, layer: usize, position: usize, keys: &Self::Tensor, spec: &crate::attention::dsa::DsaSpec) -> Result<(), BackendError> {
        let _ = spec;
        state.append(self, layer, position, keys)
    }

    fn supports_dsa_keys_layernorm_rope(&self, state: &Self::DsaState, keys: &Self::Tensor, norm_weight: &Self::Weight, norm_bias: &Self::Weight, spec: &crate::attention::dsa::DsaSpec) -> bool {
        let _ = spec;
        let state_supported = state.supports_layernorm_rope(keys.rows, keys.cols);
        let compatible = supports_dsa_fused_prologue(state, keys, norm_weight, norm_bias);
        if ops::hip::options().kernel_profile && !compatible && DSA_FUSED_PROLOGUE_PROFILE_DIAGNOSTIC.set(()).is_ok() {
            eprintln!(
                "[dsa-fused-prologue] active=false reason=preflight state_supported={state_supported} key=[{},{}] dtype={:?} layout={:?} weight_len={} bias_len={} weight_bf16={} bias_bf16={} weight_resident={} bias_resident={}",
                keys.rows,
                keys.cols,
                keys.dtype,
                keys.layout,
                norm_weight.data().len(),
                norm_bias.data().len(),
                norm_weight.resident_bf16(),
                norm_bias.resident_bf16(),
                norm_weight.resident().is_some(),
                norm_bias.resident().is_some(),
            );
        }
        compatible
    }

    fn append_dsa_keys_layernorm_rope(
        &self,
        state: &mut Self::DsaState,
        layer: usize,
        position: usize,
        keys: &Self::Tensor,
        norm_weight: &Self::Weight,
        norm_bias: &Self::Weight,
        eps: f32,
        cos: &[f32],
        sin: &[f32],
        spec: &crate::attention::dsa::DsaSpec,
    ) -> Result<bool, BackendError> {
        let compatible = supports_dsa_fused_prologue(state, keys, norm_weight, norm_bias);
        if !compatible {
            if ops::hip::options().kernel_profile && DSA_FUSED_PROLOGUE_PROFILE_DIAGNOSTIC.set(()).is_ok() {
                eprintln!(
                    "[dsa-fused-prologue] active=false reason=weight_or_tensor key=[{},{}] dtype={:?} layout={:?} weight_len={} bias_len={} weight_bf16={} bias_bf16={}",
                    keys.rows,
                    keys.cols,
                    keys.dtype,
                    keys.layout,
                    norm_weight.data().len(),
                    norm_bias.data().len(),
                    norm_weight.resident_bf16(),
                    norm_bias.resident_bf16(),
                );
            }
            return Ok(false);
        }
        let Some(norm_weight) = norm_weight.resident().map(Arc::as_ref) else { return Ok(false) };
        let Some(norm_bias) = norm_bias.resident().map(Arc::as_ref) else { return Ok(false) };
        let active = state.append_layernorm_rope(self, layer, position, keys, norm_weight, norm_bias, eps, spec.rope_dim, spec.rotary_layout, cos, sin)?;
        if ops::hip::options().kernel_profile && DSA_FUSED_PROLOGUE_PROFILE_DIAGNOSTIC.set(()).is_ok() {
            eprintln!("[dsa-fused-prologue] active={active} reason=state_contract key=[{},{}]", keys.rows, keys.cols);
        }
        Ok(active)
    }

    fn append_dsa_keys_gated(&self, state: &mut Self::DsaState, layer: usize, position: usize, keys: &Self::Tensor, gate: &Self::Tensor, spec: &crate::attention::dsa::DsaSpec) -> Result<(), BackendError> {
        state.append_gated(self, layer, position, keys, gate, spec.kpool)
    }

    fn dsa_select_topk(&self, state: &mut Self::DsaState, layer: usize, query: &Self::Tensor, head_weights: &Self::Tensor, spec: &crate::attention::dsa::DsaSpec) -> Result<(), BackendError> {
        if spec.kpool > 0 {
            return state.select_kpool(self, layer, query, head_weights, spec.kpool);
        }
        state.select(self, layer, query, head_weights)
    }

    fn dsa_select_topk_begin(&self, state: &mut Self::DsaState, layer: usize, query: &Self::Tensor, head_weights: &Self::Tensor, spec: &crate::attention::dsa::DsaSpec) -> Result<(), BackendError> {
        if spec.kpool > 0 {
            return state.select_kpool(self, layer, query, head_weights, spec.kpool);
        }
        state.select_begin(self, layer, query, head_weights)
    }

    fn dsa_select_topk_finish(&self, state: &mut Self::DsaState) -> Result<(), BackendError> {
        state.select_finish(self)
    }

    fn mla_decode_attention_selected(
        &self,
        query: &Self::Tensor,
        cache: &Self::Cache,
        kv_b: &Self::Weight,
        layer: usize,
        position: usize,
        mla: &crate::attention::mla::MlaSpec,
        dsa: &crate::attention::dsa::DsaSpec,
        state: &Self::DsaState,
    ) -> Result<Self::Tensor, BackendError> {
        let _ = position;
        paged_mla_attention(self, query, cache, kv_b, layer, mla, Some(state), dsa.top_k)
    }
}

impl DsaPrefillBackend for RocmContext {
    fn dsa_select_prefill(&self, state: &mut Self::DsaState, layer: usize, query: &Self::Tensor, head_weights: &Self::Tensor, spec: &crate::attention::dsa::DsaSpec) -> Result<(), BackendError> {
        if spec.kpool > 0 {
            return state.select_kpool(self, layer, query, head_weights, spec.kpool);
        }
        state.select(self, layer, query, head_weights)
    }

    fn mla_prefill_attention_selected(
        &self,
        query: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        kv_b: &Self::Weight,
        cache: Option<&mut Self::Cache>,
        layer: usize,
        mla: &crate::attention::mla::MlaSpec,
        dsa: &crate::attention::dsa::DsaSpec,
        state: &Self::DsaState,
    ) -> Result<Self::Tensor, BackendError> {
        let cache = cache.ok_or_else(|| compute_error(format!("L{layer} ROCm DSA prefill 必须提供 paged cache")))?;
        cache.append_mla(self, layer, latent, k_rope)?;
        paged_mla_attention(self, query, cache, kv_b, layer, mla, Some(state), dsa.top_k)
    }

    fn mla_prefill_attention_selected_rope(
        &self,
        query: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        kv_b: &Self::Weight,
        cache: Option<&mut Self::Cache>,
        layer: usize,
        position: usize,
        cos: &[f32],
        sin: &[f32],
        mla: &crate::attention::mla::MlaSpec,
        dsa: &crate::attention::dsa::DsaSpec,
        state: &Self::DsaState,
    ) -> Result<Self::Tensor, BackendError> {
        let cache = cache.ok_or_else(|| compute_error(format!("L{layer} ROCm DSA prefill 必须提供 paged cache")))?;
        <Self as DecodeBackend>::append_mla_rope(self, cache, layer, latent, k_rope, mla.qk_rope_head_dim, mla.rotary_layout, position, cos, sin)?;
        paged_mla_attention(self, query, cache, kv_b, layer, mla, Some(state), dsa.top_k)
    }

    fn mla_prefill_attention_selected_segmented(
        &self,
        query: &Self::Tensor,
        latent: &Self::Tensor,
        k_rope: &Self::Tensor,
        kv_b: &Self::Weight,
        layer: usize,
        mla: &crate::attention::mla::MlaSpec,
        dsa: &crate::attention::dsa::DsaSpec,
        segments: &mut [crate::backend::DsaPrefillSegment<'_, Self>],
    ) -> Result<Self::Tensor, BackendError> {
        let total_rows = segments.iter().try_fold(0usize, |rows, segment| rows.checked_add(segment.rows).ok_or_else(|| compute_error("ROCm segmented MLA rows 溢出")))?;
        if segments.is_empty() || total_rows != query.rows || latent.rows != query.rows || k_rope.rows != query.rows {
            return Err(compute_error(format!("ROCm segmented MLA rows={total_rows} Q/KV={}/{}/{}", query.rows, latent.rows, k_rope.rows,)));
        }
        let element_bytes = common_mla_output_element_bytes(segments.iter().map(|segment| segment.rows)).ok_or_else(|| compute_error("ROCm segmented MLA 不能混合 decode F32 与 prefill BF16 输出"))?;
        let row_bytes = mla.q_projection_size.checked_mul(element_bytes).ok_or_else(|| compute_error("ROCm segmented MLA row bytes 溢出"))?;
        let output_bytes = total_rows.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm segmented MLA output bytes 溢出"))?;
        let output = std::sync::Arc::new(ops::hip::DeviceBuffer::allocate_reusable(self.device_id, output_bytes).map_err(compute_error)?);
        let fused = segments.iter().all(|segment| {
            // 单 token decode 保留 split-Q8-WMMA 路径；128 行 prefill tile 会浪费绝大部分计算。
            if segment.rows <= 1 {
                return false;
            }
            let query_start = segment.cache.paged_layers.get(layer).and_then(Option::as_ref).map_or(0, |cached| cached.rows);
            segment.state.device_selection(segment.rows, query_start).is_some()
        }) && !ops::hip::options().force_dense_prefill
            && !ops::hip::options().sparse_prefill_heads4;
        if fused {
            let mut row_offset = 0usize;
            for segment in segments.iter_mut() {
                let segment_latent = crate::backend::SegmentedTensorBackend::slice_token_rows(self, latent, row_offset, segment.rows)?;
                let segment_rope = crate::backend::SegmentedTensorBackend::slice_token_rows(self, k_rope, row_offset, segment.rows)?;
                segment.cache.append_mla(self, layer, &segment_latent, &segment_rope)?;
                row_offset += segment.rows;
            }
            let mut batch_segments = Vec::with_capacity(segments.len());
            for segment in segments.iter() {
                let cached = segment.cache.paged_layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} ROCm segmented MLA cache 尚未初始化")))?;
                let table = segment.cache.block_table.buffer().ok_or_else(|| compute_error("ROCm segmented MLA 缺少 block table"))?;
                let query_start = cached.rows.checked_sub(segment.rows).ok_or_else(|| compute_error("ROCm segmented MLA query start 下溢"))?;
                let selection = segment.state.device_selection(segment.rows, query_start).ok_or_else(|| compute_error("ROCm segmented MLA 缺少 DSA selection"))?;
                batch_segments.push(ops::hip::CtMlaPrefillSegmentRef {
                    latent_cache: &cached.latent,
                    latent_scales: cached.latent_scales.as_deref(),
                    latent_group_size: cached.latent_group_size,
                    rope_cache: &cached.rope,
                    block_table: table,
                    selection,
                    query_rows: segment.rows,
                    context_rows: cached.rows,
                    query_start,
                });
            }
            let query_device = query.device.as_deref().ok_or_else(|| compute_error("ROCm segmented MLA query 缺少 device buffer"))?;
            ops::hip::try_paged_mla_attention_ct_segmented(self.device_id, query_device, ct_mla_weight(kv_b)?, &batch_segments, mla.q_projection_size, mla.num_heads, mla.qk_rope_head_dim, dsa.top_k, ROCM_KV_BLOCK_SIZE, &output)
                .map_err(compute_error)?;
            if ops::hip::options().kernel_sync {
                ops::hip::synchronize_device(self.device_id, &format!("L{layer} ROCm segmented MLA synchronize")).map_err(compute_error)?;
            }
        } else {
            let mut row_offset = 0usize;
            for segment in segments {
                let segment_query = crate::backend::SegmentedTensorBackend::slice_token_rows(self, query, row_offset, segment.rows)?;
                let segment_latent = crate::backend::SegmentedTensorBackend::slice_token_rows(self, latent, row_offset, segment.rows)?;
                let segment_rope = crate::backend::SegmentedTensorBackend::slice_token_rows(self, k_rope, row_offset, segment.rows)?;
                segment.cache.append_mla(self, layer, &segment_latent, &segment_rope)?;
                let byte_offset = row_offset.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm segmented MLA output offset 溢出"))?;
                let segment_bytes = segment.rows.checked_mul(row_bytes).ok_or_else(|| compute_error("ROCm segmented MLA segment bytes 溢出"))?;
                let segment_output = ops::hip::DeviceBuffer::view(output.clone(), byte_offset, segment_bytes).map_err(compute_error)?;
                paged_mla_attention_into(self, &segment_query, &*segment.cache, kv_b, layer, mla, Some(segment.state), dsa.top_k, &segment_output)?;
                row_offset += segment.rows;
            }
        }
        let merged = ops::hip::DeviceBuffer::view(output, 0, output_bytes).map_err(compute_error)?;
        Ok(device_tensor_with_dtype(merged, total_rows, mla.q_projection_size, if element_bytes == RocmTensorDType::Bf16.element_bytes() { RocmTensorDType::Bf16 } else { RocmTensorDType::F32 }))
    }
}

fn rmsnorm_heads(context: &RocmContext, input: &RocmTensor, weight: &RocmWeight, head_count: usize, head_dim: usize, eps: f32, gemma: bool) -> Result<RocmTensor, BackendError> {
    if input.cols != head_count.checked_mul(head_dim).ok_or_else(|| compute_error("Rocm GQA head 列维度溢出"))? {
        return Err(compute_error(format!("Rocm GQA head norm input cols={}，期望 {}×{}", input.cols, head_count, head_dim)));
    }
    let input = context.tensor_as_f32(input.clone())?;
    let input_device = input.device.as_deref().ok_or_else(|| compute_error("Rocm GQA head norm input 缺少 device buffer"))?;
    let weight_device = weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("Rocm GQA head norm weight 缺少 resident buffer"))?;
    let output =
        crate::kernel::rocm::hip::try_rmsnorm_resident_weight_to_f32(context.device_id, input_device, weight_device, input.rows.checked_mul(head_count).ok_or_else(|| compute_error("Rocm GQA head norm rows 溢出"))?, head_dim, eps, gemma)
            .map_err(compute_error)?;
    Ok(super::device_tensor_f32(output, input.rows, input.cols))
}

impl GqaPrefillBackend for RocmContext {
    fn rmsnorm_triple_linear(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, first: &Self::Weight, second: &Self::Weight, third: &Self::Weight) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), BackendError> {
        let short_w8 = input.rows > 0
            && input.rows <= 8
            && matches!(first.quantized(), Some(RocmQuantizedWeight::W8A16 { .. }))
            && matches!(second.quantized(), Some(RocmQuantizedWeight::W8A16 { .. }))
            && matches!(third.quantized(), Some(RocmQuantizedWeight::W8A16 { .. }));
        if short_w8 {
            let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm RMSNorm triple W8 input 缺少 resident buffer"))?;
            let weight_device = norm_weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm RMSNorm triple W8 weight 缺少 resident buffer"))?;
            let normalized = ops::hip::try_rmsnorm_resident_weight_to_bf16(self.device_id, input_device, weight_device, input.rows, input.cols, eps, false).map_err(compute_error)?;
            let normalized = device_tensor_bf16(normalized, input.rows, input.cols);
            return Ok((self.linear(&normalized, first)?, self.linear(&normalized, second)?, self.linear(&normalized, third)?));
        }
        let normalized = self.rmsnorm(input, norm_weight, eps)?;
        self.triple_linear(&normalized, first, second, third)
    }

    fn rmsnorm_gated_linear(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, gate: &Self::Weight, up: &Self::Weight, activation: &crate::moe::Activation) -> Result<Self::Tensor, BackendError> {
        let short_w8 = input.rows > 0 && input.rows <= 8 && matches!(gate.quantized(), Some(RocmQuantizedWeight::W8A16 { .. })) && matches!(up.quantized(), Some(RocmQuantizedWeight::W8A16 { .. }));
        if short_w8 {
            let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm RMSNorm gated W8 input 缺少 resident buffer"))?;
            let weight_device = norm_weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error("ROCm RMSNorm gated W8 weight 缺少 resident buffer"))?;
            let normalized = ops::hip::try_rmsnorm_resident_weight_to_bf16(self.device_id, input_device, weight_device, input.rows, input.cols, eps, false).map_err(compute_error)?;
            let normalized = device_tensor_bf16(normalized, input.rows, input.cols);
            let gate = self.linear(&normalized, gate)?;
            let up = self.linear(&normalized, up)?;
            return self.gated_activation(&gate, &up, activation);
        }
        let normalized = self.rmsnorm(input, norm_weight, eps)?;
        self.gated_linear(&normalized, gate, up, activation)
    }

    fn rmsnorm_heads(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        rmsnorm_heads(self, input, weight, head_count, head_dim, eps, false)
    }

    fn gemma_rmsnorm_heads(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        rmsnorm_heads(self, input, weight, head_count, head_dim, eps, true)
    }

    fn gqa_prefill_attention(&self, query: &Self::Tensor, key: &Self::Tensor, value: &Self::Tensor, spec: &crate::attention::gqa::GqaSpec) -> Result<Self::Tensor, BackendError> {
        let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA query 维度溢出"))?;
        let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA KV 维度溢出"))?;
        if query.rows != key.rows || query.rows != value.rows || query.cols != query_cols || key.cols != kv_cols || value.cols != kv_cols {
            return Err(compute_error(format!("Rocm GQA prefill shape 异常: Q=[{},{}] K=[{},{}] V=[{},{}]", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols)));
        }
        let query = self.tensor_as_f32(query.clone())?;
        let key = self.tensor_as_f32(key.clone())?;
        let value = self.tensor_as_f32(value.clone())?;
        let query_device = query.device.as_deref().ok_or_else(|| compute_error("Rocm GQA query 缺少 device buffer"))?;
        let key_device = key.device.as_deref().ok_or_else(|| compute_error("Rocm GQA key 缺少 device buffer"))?;
        let value_device = value.device.as_deref().ok_or_else(|| compute_error("Rocm GQA value 缺少 device buffer"))?;
        let output = if spec.head_dim == 128 {
            ops::hip::try_gqa_prefill_wmma_resident_f32(self.device_id, query_device, key_device, value_device, query.rows, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.score_scale)
        } else {
            ops::hip::try_gqa_prefill_resident_f32(self.device_id, query_device, key_device, value_device, query.rows, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.score_scale)
        }
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, query.rows, query.cols))
    }

    fn gqa_prefill_attention_cached(
        &self,
        cache: &mut Self::Cache,
        layer: usize,
        position: usize,
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        spec: &crate::attention::gqa::GqaSpec,
        retain_full_cache: bool,
    ) -> Result<Self::Tensor, BackendError> {
        let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA query 维度溢出"))?;
        let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA KV 维度溢出"))?;
        if query.rows != key.rows || query.rows != value.rows || query.cols != query_cols || key.cols != kv_cols || value.cols != kv_cols {
            return Err(compute_error(format!("Rocm GQA cached prefill shape 异常: Q=[{},{}] K=[{},{}] V=[{},{}]", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols)));
        }
        let key_f32 = self.tensor_as_f32(key.clone())?;
        let value_f32 = self.tensor_as_f32(value.clone())?;
        cache.append_gqa_rows(self, layer, &key_f32, &value_f32)?;
        let query_f32 = self.tensor_as_f32(query.clone())?;
        let query_device = query_f32.device.as_deref().ok_or_else(|| compute_error("Rocm GQA query 缺少 device buffer"))?;
        let (key_cached, value_cached, rows, cols) = cache.gqa_layer_buffers(layer)?;
        let expected_rows = position.checked_add(query.rows).ok_or_else(|| compute_error("Rocm GQA cache position 溢出"))?;
        if rows != expected_rows || cols != kv_cols {
            return Err(compute_error(format!("Rocm GQA L{layer} 设备缓存不完整: rows={rows}/{expected_rows} cols={cols}/{kv_cols}")));
        }
        if query.rows == 1 {
            let output = ops::hip::try_gqa_decode_cached_f32(self.device_id, query_device, &key_cached, &value_cached, rows, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.score_scale).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, 1, query_cols));
        }
        if matches!(spec.window, crate::attention::gqa::CausalWindow::Full) {
            // chunked prefill 的 Q 只含本 chunk，K/V 则是完整前缀。显式可见区间
            // 保持因果语义，并让 64K 路径全程留在设备，不建立等大的 host 镜像。
            let mut visible = Vec::with_capacity(query.rows * 2);
            for row in 0..query.rows {
                let end = position.checked_add(row).and_then(|value| value.checked_add(1)).and_then(|value| u32::try_from(value).ok()).ok_or_else(|| compute_error("Rocm GQA visible end 溢出 u32"))?;
                visible.extend_from_slice(&[0, end]);
            }
            let output =
                ops::hip::try_block_attention_resident_f32(self.device_id, query_device, &key_cached, &value_cached, &visible, query.rows, rows, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.score_scale).map_err(compute_error)?;
            return Ok(device_tensor_f32(output, query.rows, query_cols));
        }
        append_gqa_cache(cache, layer, position, key, value, spec, retain_full_cache)?;
        if query.rows > 1 {
            self.require_cpu_reference_fallback("cached GQA attention")?;
            let query_data = tensor_data(&query_f32)?;
            let cached = cache.gqa_layer(layer)?;
            let mut output = vec![0.0; query.rows * query_cols];
            crate::attention::gqa::prefill_attention_at_f32(&query_data, &cached.key, &cached.value, query.rows, cached.start, cached.rows, position, spec, &mut output).map_err(compute_error)?;
            return self.tensor_from_f32(output, query.rows, query_cols).map_err(compute_error);
        }
        unreachable!("单行已由设备 decode 分支处理")
    }

    fn gqa_prefill_attention_cached_visible(
        &self,
        cache: &mut Self::Cache,
        layer: usize,
        position: usize,
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        spec: &crate::attention::gqa::GqaSpec,
        visible_ends: &[u32],
        retain_full_cache: bool,
    ) -> Result<Self::Tensor, BackendError> {
        self.require_cpu_reference_fallback("cached visible GQA attention")?;
        let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA query 维度溢出"))?;
        let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA KV 维度溢出"))?;
        if query.rows != key.rows || query.rows != value.rows || query.cols != query_cols || key.cols != kv_cols || value.cols != kv_cols {
            return Err(compute_error(format!("Rocm GQA cached visible shape 异常: Q=[{},{}] K=[{},{}] V=[{},{}]", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols)));
        }
        if visible_ends.len() != query.rows {
            return Err(compute_error(format!("Rocm GQA visible_ends={}，期望 {}", visible_ends.len(), query.rows)));
        }
        append_gqa_cache(cache, layer, position, key, value, spec, retain_full_cache)?;
        let cached = cache.gqa_layer(layer)?;
        let query_data = tensor_data(query)?;
        let mut output = vec![0.0; query.rows * query_cols];
        crate::attention::gqa::prefill_attention_at_visible_f32(&query_data, &cached.key, &cached.value, query.rows, cached.start, cached.rows, position, spec, Some(visible_ends), &mut output).map_err(compute_error)?;
        self.tensor_from_f32(output, query.rows, query_cols).map_err(compute_error)
    }

    fn gqa_prefill_attention_cached_from(&self, cache: &Self::Cache, source_layer: usize, position: usize, query: &Self::Tensor, spec: &crate::attention::gqa::GqaSpec) -> Result<Self::Tensor, BackendError> {
        self.require_cpu_reference_fallback("shared cached GQA attention")?;
        let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA query 维度溢出"))?;
        let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA KV 维度溢出"))?;
        let cached = cache.gqa_layer(source_layer)?;
        let required_rows = position.checked_add(query.rows).ok_or_else(|| compute_error("Rocm GQA shared cache position 溢出"))?;
        if query.cols != query_cols || cached.cols != kv_cols || cached.rows < required_rows {
            return Err(compute_error(format!("Rocm GQA shared cache L{source_layer} 不完整: Q=[{},{}], cache=[{},{}], required_rows={required_rows}", query.rows, query.cols, cached.rows, cached.cols,)));
        }
        let query_data = tensor_data(query)?;
        let mut output = vec![0.0; query.rows * query_cols];
        crate::attention::gqa::prefill_attention_at_f32(&query_data, &cached.key, &cached.value, query.rows, cached.start, cached.rows, position, spec, &mut output).map_err(compute_error)?;
        self.tensor_from_f32(output, query.rows, query_cols).map_err(compute_error)
    }

    fn gqa_prefill_attention_cached_from_visible(&self, cache: &Self::Cache, source_layer: usize, position: usize, query: &Self::Tensor, spec: &crate::attention::gqa::GqaSpec, visible_ends: &[u32]) -> Result<Self::Tensor, BackendError> {
        self.require_cpu_reference_fallback("shared cached visible GQA attention")?;
        let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA query 维度溢出"))?;
        let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute_error("Rocm GQA KV 维度溢出"))?;
        let cached = cache.gqa_layer(source_layer)?;
        if visible_ends.len() != query.rows {
            return Err(compute_error(format!("Rocm GQA visible_ends={}，期望 {}", visible_ends.len(), query.rows)));
        }
        if query.cols != query_cols || cached.cols != kv_cols || cached.rows < position {
            return Err(compute_error(format!("Rocm GQA visible shared cache L{source_layer} 不完整: Q=[{},{}], cache=[{},{}]", query.rows, query.cols, cached.rows, cached.cols)));
        }
        let query_data = tensor_data(query)?;
        let mut output = vec![0.0; query.rows * query_cols];
        crate::attention::gqa::prefill_attention_at_visible_f32(&query_data, &cached.key, &cached.value, query.rows, cached.start, cached.rows, position, spec, Some(visible_ends), &mut output).map_err(compute_error)?;
        self.tensor_from_f32(output, query.rows, query_cols).map_err(compute_error)
    }
}

#[cfg(test)]
mod tests {
    use super::common_mla_output_element_bytes;

    #[test]
    fn segmented_mla输出精度按每个session的query_rows确定() {
        assert_eq!(common_mla_output_element_bytes([1, 1, 1]), Some(4));
        assert_eq!(common_mla_output_element_bytes([2, 2048, 4096]), Some(2));
        assert_eq!(common_mla_output_element_bytes([1, 2]), None);
        assert_eq!(common_mla_output_element_bytes([]), None);
    }
}
