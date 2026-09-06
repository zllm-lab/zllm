use super::*;

static DSA_FUSED_PROLOGUE_PROFILE_DIAGNOSTIC: std::sync::OnceLock<()> = std::sync::OnceLock::new();

pub(super) fn supports_dsa_fused_prologue(state: &RocmDsaState, keys: &RocmTensor, norm_weight: &RocmWeight, norm_bias: &RocmWeight) -> bool {
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

const MLA_HOT_PREFILL_TILE_ROWS: usize = 32;

fn tensor_row_view(tensor: &RocmTensor, row_start: usize, rows: usize, name: &str) -> Result<RocmTensor, BackendError> {
    if rows == 0 || row_start.checked_add(rows).is_none_or(|end| end > tensor.rows) {
        return Err(compute_error(format!("ROCm {name} row view start={row_start} rows={rows} 超过 {}", tensor.rows)));
    }
    let row_bytes = tensor.cols.checked_mul(tensor.dtype.element_bytes()).ok_or_else(|| compute_error(format!("ROCm {name} row bytes 溢出")))?;
    let offset = row_start.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("ROCm {name} row offset 溢出")))?;
    let bytes = rows.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("ROCm {name} view bytes 溢出")))?;
    let owner = tensor.device.as_ref().ok_or_else(|| compute_error(format!("ROCm {name} 缺少 device buffer")))?.clone();
    let view = ops::hip::DeviceBuffer::view(owner, offset, bytes).map_err(compute_error)?;
    Ok(device_tensor_with_dtype(view, rows, tensor.cols, tensor.dtype))
}

fn buffer_row_view(source: &Arc<ops::hip::DeviceBuffer>, row_start: usize, rows: usize, row_bytes: usize, name: &str) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
    let offset = row_start.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("ROCm {name} row offset 溢出")))?;
    let bytes = rows.checked_mul(row_bytes).ok_or_else(|| compute_error(format!("ROCm {name} view bytes 溢出")))?;
    if rows == 0 || offset.checked_add(bytes).is_none_or(|end| end > source.bytes()) {
        return Err(compute_error(format!("ROCm {name} row view start={row_start} rows={rows} stride={row_bytes} 超过 {}", source.bytes())));
    }
    Ok(Arc::new(ops::hip::DeviceBuffer::view(source.clone(), offset, bytes).map_err(compute_error)?))
}

// prefill 输出统一为 BF16；不能把最后一行单独切成 decode/F32 输出。
fn next_mla_hot_tile_rows(remaining: usize) -> usize {
    if remaining <= MLA_HOT_PREFILL_TILE_ROWS {
        return remaining;
    }
    if remaining == MLA_HOT_PREFILL_TILE_ROWS + 1 {
        return MLA_HOT_PREFILL_TILE_ROWS - 1;
    }
    MLA_HOT_PREFILL_TILE_ROWS
}

#[allow(clippy::too_many_arguments)]
fn cpu_prefill_mla_attention(
    context: &RocmContext,
    query: &RocmTensor,
    latent: &RocmTensor,
    k_rope: &RocmTensor,
    kv_b: &RocmWeight,
    cache: &mut RocmKvCache,
    layer: usize,
    position: usize,
    mla: &crate::attention::mla::MlaSpec,
    state: &RocmDsaState,
) -> Result<RocmTensor, BackendError> {
    let total_started = std::time::Instant::now();
    if query.rows <= 1 || query.rows != latent.rows || query.rows != k_rope.rows || query.cols != mla.q_projection_size || latent.cols != mla.kv_lora_rank || k_rope.cols != mla.qk_rope_head_dim {
        return Err(compute_error(format!("L{layer} CPU prefill MLA Q/KV/RoPE=[{},{}]/[{},{}]/[{},{}] 非法", query.rows, query.cols, latent.rows, latent.cols, k_rope.rows, k_rope.cols)));
    }
    let kv_b = kv_b.cpu_mla_data().ok_or_else(|| compute_error(format!("L{layer} CPU prefill MLA 缺少 kv_b host 权重")))?;
    if position != cache.mla_rows(layer) {
        return Err(compute_error(format!("L{layer} CPU prefill MLA position={position}，cache rows={}", cache.mla_rows(layer))));
    }
    cache.append_mla(context, layer, latent, k_rope)?;
    let append_submit_ms = total_started.elapsed().as_secs_f64() * 1e3;
    let history_rows = cache.mla_rows(layer);
    let selection = state.host_selection(query.rows, position).map(|tokens| (tokens, state.selection_width()));
    if history_rows > state.selection_width() && selection.is_none() {
        return Err(compute_error(format!("L{layer} CPU prefill MLA history={history_rows} 缺少 CPU DSA selection: rows={} start={position}", query.rows)));
    }
    let started = std::time::Instant::now();
    let (output, compute_ms, query_download_ms) = context.with_tensors_bf16_bits(&[query], |bits| {
        let query_download_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = std::time::Instant::now();
        let output = cache.with_cpu_mla_history(layer, |history| {
            let scales = history.latent_scales.as_deref().ok_or_else(|| compute_error(format!("L{layer} CPU prefill MLA history 不是 Q8")))?;
            crate::kernel::cpu::mla::mla_attention_absorbed_q8_history(mla, bits[0], query.rows, position, &history.latent, scales, history.latent_group_size, &history.rope, history.rows, kv_b, selection).map_err(compute_error)
        })?;
        let compute_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok((output, compute_ms, query_download_ms))
    })?;
    let started = std::time::Instant::now();
    let output = context.tensor_from_bf16_bits_streamed(output, query.rows, mla.num_heads * mla.value_dim())?;
    let upload_ms = started.elapsed().as_secs_f64() * 1e3;
    if ops::hip::options().mla_hot_trace {
        eprintln!(
            "[mla-cpu-prefill] device={} layer={layer} rows={} start={position} history={} selected={} cache_append_submit_ms={append_submit_ms:.3} query_d2h_ms={query_download_ms:.3} compute_ms={compute_ms:.3} upload_submit_ms={upload_ms:.3} total_ms={:.3}",
            context.device_id,
            query.rows,
            history_rows,
            selection.is_some(),
            total_started.elapsed().as_secs_f64() * 1e3,
        );
    }
    Ok(output)
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
        if query.rows > cached.committed_rows {
            return Err(compute_error(format!("L{layer} ROCm MLA hot query_rows={} 超过 hot_rows={}", query.rows, cached.committed_rows)));
        }
        let host_selection = dsa_state.and_then(|state| state.host_selection(query.rows, query_start));
        let mut hot = hot.lock().map_err(|_| compute_error(format!("L{layer} ROCm MLA hot 锁中毒")))?;
        let selection = host_selection
            .map(|selection| {
                hot.take_prefetched_selection(context.device_id, query_start)?
                    .map_or_else(|| hot.prepare_selection(context.device_id, layer, selection, &cached.latent, cached.latent_scales.as_deref().expect("Q8 MLA hot 必有 scales"), &cached.rope), Ok)
            })
            .transpose()?;
        let (context_rows, hot_query_start, selection_width) = match (selection.as_ref(), host_selection) {
            (Some(_), Some(selection)) => (cached.committed_rows, cached.committed_rows - query.rows, selection.len() / query.rows),
            (None, None) if cached.rows <= top_k && cached.rows <= cached.committed_rows => (cached.rows, query_start, top_k),
            _ => return Err(compute_error(format!("L{layer} ROCm MLA hot 缺少 host selection: rows={} start={query_start} context={}", query.rows, cached.rows))),
        };
        ops::hip::try_paged_mla_attention_ct_into(
            context.device_id,
            query_device,
            &cached.latent,
            cached.latent_scales.as_deref(),
            cached.latent_group_size,
            &cached.rope,
            table,
            selection.as_ref(),
            ct_mla_weight(kv_b)?,
            query.rows,
            context_rows,
            hot_query_start,
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

#[allow(clippy::too_many_arguments)]
fn paged_mla_attention_with_selection(
    context: &RocmContext,
    query: &RocmTensor,
    cache: &RocmKvCache,
    kv_b: &RocmWeight,
    layer: usize,
    spec: &crate::attention::mla::MlaSpec,
    selection: Option<&ops::hip::DeviceBuffer>,
    host_selection: Option<&[u32]>,
    selection_width: usize,
) -> Result<RocmTensor, BackendError> {
    let cached = cache.paged_layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} ROCm cooperative MLA cache 尚未初始化")))?;
    if query.rows == 0 || query.rows > cached.rows {
        return Err(compute_error(format!("L{layer} ROCm cooperative MLA cache/query 状态非法: query={} cache={} hot={}", query.rows, cached.rows, cached.cpu_hot.is_some())));
    }
    let query_device = query.device.as_deref().ok_or_else(|| compute_error("ROCm cooperative MLA query 缺少 device buffer"))?;
    let table = cache.block_table.buffer().ok_or_else(|| compute_error("ROCm cooperative MLA 缺少 block table"))?;
    let query_start = cached.rows - query.rows;
    let element_bytes = mla_output_element_bytes(query.rows);
    let output_bytes = query.rows.checked_mul(spec.q_projection_size).and_then(|elements| elements.checked_mul(element_bytes)).ok_or_else(|| compute_error("ROCm cooperative MLA output 大小溢出"))?;

    if let (Some(hot), Some(host_selection)) = (&cached.cpu_hot, host_selection)
        && query.rows > 1
        && cached.rows > cached.committed_rows
    {
        if query.rows > cached.committed_rows {
            return Err(compute_error(format!("L{layer} ROCm cooperative MLA hot query={} 超过 {}", query.rows, cached.committed_rows)));
        }
        if host_selection.len() != query.rows.checked_mul(selection_width).ok_or_else(|| compute_error("ROCm cooperative MLA selection 大小溢出"))? {
            return Err(compute_error(format!("L{layer} ROCm cooperative MLA host selection={}，期望 {}x{selection_width}", host_selection.len(), query.rows)));
        }
        let output = Arc::new(ops::hip::DeviceBuffer::allocate_reusable(context.device_id, output_bytes).map_err(compute_error)?);
        let mut row_start = 0usize;
        while row_start < query.rows {
            let tile_rows = next_mla_hot_tile_rows(query.rows - row_start);
            let tile_query = tensor_row_view(query, row_start, tile_rows, "cooperative MLA hot query")?;
            let selection_start = row_start * selection_width;
            let selection_end = selection_start + tile_rows * selection_width;
            let remapped = {
                let mut hot = hot.lock().map_err(|_| compute_error(format!("L{layer} ROCm cooperative MLA hot 锁中毒")))?;
                hot.prepare_selection(context.device_id, layer, &host_selection[selection_start..selection_end], &cached.latent, cached.latent_scales.as_deref().expect("Q8 MLA hot 必有 scales"), &cached.rope)?
            };
            let output_offset = row_start.checked_mul(spec.q_projection_size).and_then(|elements| elements.checked_mul(element_bytes)).ok_or_else(|| compute_error("ROCm cooperative MLA hot output offset 溢出"))?;
            let tile_bytes = tile_rows.checked_mul(spec.q_projection_size).and_then(|elements| elements.checked_mul(element_bytes)).ok_or_else(|| compute_error("ROCm cooperative MLA hot tile output 大小溢出"))?;
            let tile_output = ops::hip::DeviceBuffer::view(output.clone(), output_offset, tile_bytes).map_err(compute_error)?;
            ops::hip::try_paged_mla_attention_ct_into(
                context.device_id,
                tile_query.device.as_deref().expect("MLA hot tile query 必有 device"),
                &cached.latent,
                cached.latent_scales.as_deref(),
                cached.latent_group_size,
                &cached.rope,
                table,
                Some(&remapped),
                ct_mla_weight(kv_b)?,
                tile_rows,
                cached.committed_rows,
                cached.committed_rows - tile_rows,
                spec.q_projection_size,
                spec.num_heads,
                spec.qk_rope_head_dim,
                selection_width,
                ROCM_KV_BLOCK_SIZE,
                &tile_output,
                None,
            )
            .map_err(compute_error)?;
            row_start += tile_rows;
        }
        return Ok(device_tensor_with_arc(output, query.rows, spec.q_projection_size, RocmTensorDType::Bf16));
    }

    let remapped = if let Some(hot) = &cached.cpu_hot {
        if query.rows > cached.committed_rows {
            return Err(compute_error(format!("L{layer} ROCm cooperative MLA hot query={} 超过 {}", query.rows, cached.committed_rows)));
        }
        if let Some(host_selection) = host_selection {
            let mut hot = hot.lock().map_err(|_| compute_error(format!("L{layer} ROCm cooperative MLA hot 锁中毒")))?;
            Some(
                hot.take_prefetched_selection(context.device_id, query_start)?
                    .map_or_else(|| hot.prepare_selection(context.device_id, layer, host_selection, &cached.latent, cached.latent_scales.as_deref().expect("Q8 MLA hot 必有 scales"), &cached.rope), Ok)?,
            )
        } else {
            None
        }
    } else {
        None
    };
    let (selection, context_rows, query_start, selection_width) = if cached.cpu_hot.is_some() {
        match (remapped.as_ref(), host_selection) {
            (Some(remapped), Some(host_selection)) => (Some(remapped), cached.committed_rows, cached.committed_rows - query.rows, host_selection.len() / query.rows),
            (None, None) if cached.rows <= selection_width && cached.rows <= cached.committed_rows => (None, cached.rows, query_start, selection_width),
            _ => return Err(compute_error(format!("L{layer} ROCm cooperative MLA hot 缺少 host selection: rows={} start={query_start}", query.rows))),
        }
    } else {
        (selection, cached.rows, query_start, selection_width)
    };
    let output = ops::hip::DeviceBuffer::allocate_reusable(context.device_id, output_bytes).map_err(compute_error)?;
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
        context_rows,
        query_start,
        spec.q_projection_size,
        spec.num_heads,
        spec.qk_rope_head_dim,
        selection_width,
        ROCM_KV_BLOCK_SIZE,
        &output,
        None,
    )
    .map_err(compute_error)?;
    Ok(device_tensor_with_dtype(output, query.rows, spec.q_projection_size, if query.rows > 1 { RocmTensorDType::Bf16 } else { RocmTensorDType::F32 }))
}

#[allow(clippy::too_many_arguments)]
fn paged_mla_attention_shard(
    context: &RocmContext,
    query: &RocmTensor,
    cache: &RocmKvCache,
    kv_b: &RocmWeight,
    layer: usize,
    spec: &crate::attention::mla::MlaSpec,
    selection: Option<&ops::hip::DeviceBuffer>,
    selection_counts: Option<&ops::hip::DeviceBuffer>,
    selection_width: usize,
    parity: usize,
    query_start: usize,
) -> Result<ops::hip::PagedMlaShardAttention, BackendError> {
    let cached = cache.paged_layers.get(layer).and_then(Option::as_ref).ok_or_else(|| compute_error(format!("L{layer} ROCm shard MLA cache 尚未初始化")))?;
    if cache.ownership.parity() != Some(parity) || cached.cpu_hot.is_some() || query.rows == 0 || query_start.checked_add(query.rows).is_none_or(|end| end > cached.rows) {
        return Err(compute_error(format!("L{layer} ROCm shard MLA state 非法: ownership={:?} parity={parity} query={} start={query_start} rows={} hot={}", cache.ownership, query.rows, cached.rows, cached.cpu_hot.is_some())));
    }
    let table = cache.block_table.buffer().ok_or_else(|| compute_error("ROCm shard MLA 缺少 block table"))?;
    ops::hip::try_paged_mla_attention_ct_shard(
        context.device_id,
        query.device.as_deref().ok_or_else(|| compute_error("ROCm shard MLA query 缺少 device buffer"))?,
        &cached.latent,
        cached.latent_scales.as_deref(),
        cached.latent_group_size,
        &cached.rope,
        table,
        selection,
        selection_counts,
        ct_mla_weight(kv_b)?,
        query.rows,
        cached.rows,
        query_start,
        spec.q_projection_size,
        spec.num_heads,
        spec.qk_rope_head_dim,
        selection_width,
        ROCM_KV_BLOCK_SIZE,
        parity,
    )
    .map_err(compute_error)
}

fn project_local_sequence_partial(
    context: &RocmContext,
    shard: &ops::hip::PagedMlaShardAttention,
    local_stats: &ops::hip::DeviceBuffer,
    remote_stats: &ops::hip::DeviceBuffer,
    kv_b: &RocmWeight,
    o_proj: &RocmWeight,
    rows: usize,
    mla: &crate::attention::mla::MlaSpec,
    profile_pair: bool,
) -> Result<RocmTensor, BackendError> {
    if profile_pair {
        ops::hip::device_profile_scope_operator(context.device_id, "glm_pair_attn_kv_b").map_err(compute_error)?;
    }
    let attention_bytes = rows.checked_mul(mla.q_projection_size).and_then(|n| n.checked_mul(mla_output_element_bytes(rows))).ok_or_else(|| compute_error("ROCm shard attention 大小溢出"))?;
    let attention = ops::hip::DeviceBuffer::allocate_reusable(context.device_id, attention_bytes).map_err(compute_error)?;
    ops::hip::try_paged_mla_shard_scale_project_ct(context.device_id, &shard.weighted, local_stats, remote_stats, ct_mla_weight(kv_b)?, rows, mla.q_projection_size, mla.num_heads, mla.qk_rope_head_dim, &attention).map_err(compute_error)?;
    if profile_pair {
        ops::hip::device_profile_scope_operator(context.device_id, "glm_pair_attn_o_proj").map_err(compute_error)?;
    }
    let attention = device_tensor_with_dtype(attention, rows, mla.q_projection_size, if rows > 1 { RocmTensorDType::Bf16 } else { RocmTensorDType::F32 });
    context.tensor_as_f32(context.linear(&attention, o_proj)?)
}

fn stable_sequence_shard(shard: ops::hip::PagedMlaShardAttention) -> Result<ops::hip::PagedMlaShardAttention, BackendError> {
    let stable = |buffer: Arc<ops::hip::DeviceBuffer>| -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> { if buffer.is_async_allocated() { Ok(Arc::new(buffer.copy_to_stable_deferred().map_err(compute_error)?)) } else { Ok(buffer) } };
    Ok(ops::hip::PagedMlaShardAttention { weighted: stable(shard.weighted)?, stats: stable(shard.stats)? })
}

fn copy_sequence_shards_to_device(shards: &[&ops::hip::PagedMlaShardAttention], destination_device: i32, completion_device: i32) -> Result<Vec<ops::hip::PagedMlaShardAttention>, BackendError> {
    let sources = shards.iter().flat_map(|shard| [shard.weighted.clone(), shard.stats.clone()]).collect::<Vec<_>>();
    let outputs = ops::hip::DeviceBuffer::copy_stable_group_to_device_ordered_async_retained_by(&sources, destination_device, completion_device).map_err(compute_error)?;
    if outputs.len() != shards.len() * 2 {
        return Err(compute_error(format!("ROCm shard P2P 数量={}/{} 异常", outputs.len(), shards.len() * 2)));
    }
    let mut outputs = outputs.into_iter();
    Ok((0..shards.len()).map(|_| ops::hip::PagedMlaShardAttention { weighted: Arc::new(outputs.next().expect("shard weighted 已校验")), stats: Arc::new(outputs.next().expect("shard stats 已校验")) }).collect())
}

fn exchange_sequence_shards(
    left: &[&ops::hip::PagedMlaShardAttention],
    right: &[&ops::hip::PagedMlaShardAttention],
    completion_device: i32,
) -> Result<(Vec<ops::hip::PagedMlaShardAttention>, Vec<ops::hip::PagedMlaShardAttention>), BackendError> {
    let left_sources = left.iter().flat_map(|shard| [shard.weighted.clone(), shard.stats.clone()]).collect::<Vec<_>>();
    let right_sources = right.iter().flat_map(|shard| [shard.weighted.clone(), shard.stats.clone()]).collect::<Vec<_>>();
    let (left_outputs, right_outputs) = ops::hip::DeviceBuffer::exchange_stable_groups_ordered_async_retained_by(&left_sources, &right_sources, completion_device).map_err(compute_error)?;
    let collect = |outputs: Vec<ops::hip::DeviceBuffer>, expected: usize| -> Result<Vec<ops::hip::PagedMlaShardAttention>, BackendError> {
        if outputs.len() != expected * 2 {
            return Err(compute_error(format!("ROCm shard 双向 P2P 数量={}/{} 异常", outputs.len(), expected * 2)));
        }
        let mut outputs = outputs.into_iter();
        Ok((0..expected).map(|_| ops::hip::PagedMlaShardAttention { weighted: Arc::new(outputs.next().expect("shard weighted 已校验")), stats: Arc::new(outputs.next().expect("shard stats 已校验")) }).collect())
    };
    Ok((collect(left_outputs, left.len())?, collect(right_outputs, right.len())?))
}

#[allow(clippy::too_many_arguments)]
fn project_merged_sequence_heads(
    context: &RocmContext,
    local: &ops::hip::PagedMlaShardAttention,
    remote: &ops::hip::PagedMlaShardAttention,
    kv_b: &RocmWeight,
    o_proj: &RocmWeight,
    rows: usize,
    head_start: usize,
    head_count: usize,
    mla: &crate::attention::mla::MlaSpec,
) -> Result<RocmTensor, BackendError> {
    let columns = head_count.checked_mul(mla.q_projection_size / mla.num_heads).ok_or_else(|| compute_error("ROCm shard head attention columns 溢出"))?;
    if o_proj.cols != columns {
        return Err(compute_error(format!("ROCm shard o_proj cols={}，期望 head columns={columns}", o_proj.cols)));
    }
    let attention_bytes = rows.checked_mul(columns).and_then(|n| n.checked_mul(mla_output_element_bytes(rows))).ok_or_else(|| compute_error("ROCm shard head attention 大小溢出"))?;
    let attention = ops::hip::DeviceBuffer::allocate_reusable(context.device_id, attention_bytes).map_err(compute_error)?;
    ops::hip::try_paged_mla_shard_merge_project_heads_ct(
        context.device_id,
        &local.weighted,
        &remote.weighted,
        &local.stats,
        &remote.stats,
        ct_mla_weight(kv_b)?,
        rows,
        mla.q_projection_size,
        mla.num_heads,
        head_start,
        head_count,
        mla.qk_rope_head_dim,
        &attention,
    )
    .map_err(compute_error)?;
    let attention = device_tensor_with_dtype(attention, rows, columns, if rows > 1 { RocmTensorDType::Bf16 } else { RocmTensorDType::F32 });
    context.tensor_as_f32(context.linear(&attention, o_proj)?)
}

impl RocmContext {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_operator_mla_query_impl(
        &self,
        layer: usize,
        experts: &RocmPrefillExperts,
        normalized_q_lora: &RocmTensor,
        position: usize,
        cosine: &[f32],
        sine: &[f32],
        mla: &crate::attention::mla::MlaSpec,
    ) -> Result<bool, BackendError> {
        if normalized_q_lora.rows != 1 {
            return Ok(false);
        }
        let (operator, weights) = experts.operator_mla_layer(layer)?;
        let q_head_dim = mla.q_projection_size / mla.num_heads;
        let owner_heads = weights.owner_q_b.rows / q_head_dim;
        if normalized_q_lora.cols != mla.q_lora_rank
            || q_head_dim == 0
            || !weights.owner_q_b.rows.is_multiple_of(q_head_dim)
            || !weights.peer_q_b.rows.is_multiple_of(q_head_dim)
            || weights.owner_q_b.rows + weights.peer_q_b.rows != mla.q_projection_size
        {
            return Err(compute_error(format!("L{layer} operator query 预提交 shape 非法")));
        }

        // 现在记录 q_lora ready，不能等 KV 链排入同一 owner stream 后再记录，
        // 否则 peer q_b 会被无关的 kv_a/cache append 串在后面。
        let q_lora = self.tensor_to_stable_deferred(normalized_q_lora.clone())?;
        let q_source = q_lora.device.as_ref().ok_or_else(|| compute_error("operator query 预提交缺少 q_lora device buffer"))?.clone();
        let mut transferred = ops::hip::DeviceBuffer::copy_stable_group_to_device_ordered_async_retained_by(&[q_source.clone()], operator.context.device_id, self.device_id)
            .map_err(|error| compute_error(format!("L{layer} operator query 提前 owner->peer: {error}")))?;
        let peer_q_lora = device_tensor_with_dtype(transferred.remove(0), 1, mla.q_lora_rank, q_lora.dtype);
        operator.worker.retain_for_stage(vec![q_source])?;

        // owner q_b 放到本卡独立 stream，与随后主 stream 的 KV 前导并行。
        // 下游 attention 取 pending query 时再用 event 接回主 stream。
        let main_stream = ops::hip::active_compute_stream() as usize;
        let query_stream = ops::hip::cooperative_shared_stream(self.device_id).map_err(compute_error)?;
        ops::hip::order_stream_after(self.device_id, main_stream, query_stream).map_err(compute_error)?;
        ops::hip::activate_compute_stream(self.device_id, query_stream).map_err(compute_error)?;
        let owner_query = (|| {
            let query = self.linear(&q_lora, &weights.owner_q_b)?;
            let query = self.rope(&query, owner_heads, mla.qk_rope_head_dim, mla.rotary_layout, position, cosine, sine)?;
            self.tensor_to_stable_deferred(query)
        })();
        ops::hip::activate_compute_stream(self.device_id, main_stream).map_err(compute_error)?;
        let owner_query = owner_query?;
        experts.store_operator_mla_query(layer, position, super::expert::RocmOperatorPendingQuery { owner_query, peer_q_lora, owner_query_stream: query_stream })?;
        Ok(true)
    }

    pub(super) fn operator_mla_cache_append_impl(&self, layer: usize, experts: &RocmPrefillExperts, cache: &mut RocmKvCache, latent: &RocmTensor, k_rope: &RocmTensor, position: usize) -> Result<bool, BackendError> {
        let (operator, _) = experts.operator_mla_layer(layer)?;
        let peer = operator.context;
        let peer_cache = cache.ensure_operator_peer(self, &peer)?;
        // peer cache 初始化会切换当前 HIP device；这里只恢复 device，保留
        // scheduler 已选中的 background/latency stage stream。
        ops::hip::set_device(self.device_id).map_err(compute_error)?;
        let rows = latent.rows;
        if rows == 0 || k_rope.rows != rows {
            return Err(compute_error(format!("L{layer} operator cache-only MLA rows={rows}/{} 非法", k_rope.rows)));
        }
        if cache.operator_packed_kv_replication_enabled(layer) {
            cache.append_mla(self, layer, latent, k_rope)?;
            // 来源是 owner cache 的稳定 view；其生命周期另由 owner completion
            // 接管，目标直接写入 peer cache 最终 offset。
            let _ = cache.replicate_operator_mla_append(self, &peer, layer, position, rows)?;
            return Ok(true);
        }

        // BF16/CPU 诊断形态保留双端各自 append。默认 Q8 热路径不会走这里；
        // 两次提交仍由 stream event 串联，不做 device synchronize。
        let latent = self.tensor_to_stable_deferred(latent.clone())?;
        let k_rope = self.tensor_to_stable_deferred(k_rope.clone())?;
        let sources =
            [latent.device.as_ref().ok_or_else(|| compute_error("operator cache-only latent 缺少 device buffer"))?.clone(), k_rope.device.as_ref().ok_or_else(|| compute_error("operator cache-only rope 缺少 device buffer"))?.clone()];
        cache.append_mla(self, layer, &latent, &k_rope)?;
        let mut transferred = ops::hip::DeviceBuffer::copy_stable_group_to_device_ordered_async_retained_by(&sources, peer.device_id, self.device_id).map_err(compute_error)?;
        let peer_latent = device_tensor_with_dtype(transferred.remove(0), rows, latent.cols, latent.dtype);
        let peer_rope = device_tensor_with_dtype(transferred.remove(0), rows, k_rope.cols, k_rope.dtype);
        let owner_stream = ops::hip::active_compute_stream() as usize;
        let result = (|| {
            peer.activate().map_err(compute_error)?;
            peer_cache.lock().map_err(|_| compute_error(format!("L{layer} operator peer KV 锁中毒")))?.append_mla(&peer, layer, &peer_latent, &peer_rope)
        })();
        ops::hip::activate_compute_stream(self.device_id, owner_stream).map_err(compute_error)?;
        result?;
        Ok(true)
    }

    pub(super) fn finish_operator_mla_cache_submission_impl(&self, experts: &RocmPrefillExperts) -> Result<(), BackendError> {
        let Ok(operator) = experts.operator_peer() else { return Ok(()) };
        let owner_stream = ops::hip::compute_stream_for(self.device_id) as usize;
        let peer_stream = ops::hip::compute_stream_for(operator.context.device_id) as usize;
        ops::hip::order_device_stream_after(operator.context.device_id, peer_stream, self.device_id, owner_stream).map_err(compute_error)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn operator_mla_prefill_add_impl(
        &self,
        layer: usize,
        experts: &RocmPrefillExperts,
        normalized_q_lora: &RocmTensor,
        latent: &RocmTensor,
        k_rope: &RocmTensor,
        residual: &RocmTensor,
        cache: Option<&mut RocmKvCache>,
        dsa_state: Option<&RocmDsaState>,
        position: usize,
        cosine: &[f32],
        sine: &[f32],
        mla: &crate::attention::mla::MlaSpec,
        dsa: &crate::attention::dsa::DsaSpec,
    ) -> Result<RocmTensor, BackendError> {
        let cache = cache.ok_or_else(|| compute_error(format!("L{layer} operator MLA 必须提供 cache")))?;
        let (operator, weights) = experts.operator_mla_layer(layer)?;
        let q_head_dim = mla.q_projection_size / mla.num_heads;
        let kv_head_dim = mla.kv_projection_size / mla.num_heads;
        let owner_heads = weights.owner_q_b.rows / q_head_dim;
        let peer_heads = weights.peer_q_b.rows / q_head_dim;
        let owner_q = weights.owner_q_b.rows;
        let peer_q = weights.peer_q_b.rows;
        let owner_kv = weights.owner_kv_b.rows;
        let peer_kv = weights.peer_kv_b.rows;
        if normalized_q_lora.rows == 0
            || normalized_q_lora.rows != latent.rows
            || latent.rows != k_rope.rows
            || residual.rows != latent.rows
            || normalized_q_lora.cols != mla.q_lora_rank
            || latent.cols != mla.kv_lora_rank
            || k_rope.cols != mla.qk_rope_head_dim
            || q_head_dim == 0
            || kv_head_dim == 0
            || !owner_q.is_multiple_of(q_head_dim)
            || !peer_q.is_multiple_of(q_head_dim)
            || !owner_kv.is_multiple_of(kv_head_dim)
            || !peer_kv.is_multiple_of(kv_head_dim)
            || owner_heads + peer_heads != mla.num_heads
            || owner_q + peer_q != mla.q_projection_size
            || owner_kv + peer_kv != mla.kv_projection_size
            || weights.owner_o.cols != owner_q
            || weights.peer_o.cols != peer_q
            || weights.owner_o.rows != residual.cols
            || weights.peer_o.rows != residual.cols
        {
            return Err(compute_error(format!(
                "L{layer} operator MLA shape 非法: q=[{},{}] latent=[{},{}] rope=[{},{}] residual=[{},{}] heads={} q_proj={} kv_proj={}",
                normalized_q_lora.rows, normalized_q_lora.cols, latent.rows, latent.cols, k_rope.rows, k_rope.cols, residual.rows, residual.cols, mla.num_heads, mla.q_projection_size, mla.kv_projection_size,
            )));
        }
        let peer = operator.context;
        let worker = operator.worker.clone();
        let peer_cache = cache.ensure_operator_peer(self, &peer)?;
        let rows = normalized_q_lora.rows;
        let hidden = residual.cols;
        let pending_query = if rows == 1 { experts.take_operator_mla_query(layer, position)? } else { None };
        let presubmitted_query = pending_query.is_some();
        let (pending_owner_query, pending_peer_q_lora, pending_owner_query_stream) = match pending_query {
            Some(pending) => (Some(pending.owner_query), Some(pending.peer_q_lora), Some(pending.owner_query_stream)),
            None => (None, None, None),
        };
        let query_start = position;
        let selection = dsa_state.and_then(|state| state.device_selection_arc(rows, query_start));
        let host_selection_arc = dsa_state.and_then(|state| state.host_selection_arc(rows, query_start));
        let host_selection = host_selection_arc.as_deref().map(Vec::as_slice);
        let selection_width = if selection.is_some() || host_selection.is_some() { dsa_state.map_or(dsa.top_k, RocmDsaState::selection_width) } else { dsa.top_k };
        if host_selection.is_some() && selection.is_none() {
            return Err(compute_error(format!("L{layer} operator MLA 当前要求 DSA selection 常驻 device")));
        }
        let peer_selection = if let Some(source) = selection.as_ref() {
            if let Some(cached) = cache.cached_operator_selection(peer.device_id, source) {
                Some(cached)
            } else {
                // 显式 device pool 关闭时，DSA selection 可能来自
                // hipMallocAsync；先在 producer stream 上稳定化，再建立跨卡
                // event。cache 仍以原 Arc 为 identity，IndexShare 才能复用镜像。
                let stable = if source.is_async_allocated() { Arc::new(source.copy_to_stable_deferred().map_err(compute_error)?) } else { source.clone() };
                let copied = Arc::new(stable.copy_stable_to_device_ordered_async_retained_by(peer.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} operator selection owner->peer: {error}")))?);
                cache.cache_operator_selection(peer.device_id, source.clone(), copied.clone())?;
                Some(copied)
            }
        } else {
            None
        };

        let packed_kv_replica = cache.operator_packed_kv_replication_enabled(layer);
        let q_lora = if presubmitted_query { None } else { Some(self.tensor_to_stable_deferred(normalized_q_lora.clone())?) };
        let latent = if packed_kv_replica { latent.clone() } else { self.tensor_to_stable_deferred(latent.clone())? };
        // 默认路径只在 owner 旋转、量化一次 K-RoPE，再复制压缩 cache；兼容
        // 路径仍传 F32 latent/rope，由 peer 自己 append。
        let k_rope = self.rope(k_rope, 1, mla.qk_rope_head_dim, mla.rotary_layout, position, cosine, sine)?;
        let k_rope = if packed_kv_replica { k_rope } else { self.tensor_to_stable_deferred(k_rope)? };
        let mut peer_sources = Vec::new();
        if let Some(q_lora) = q_lora.as_ref() {
            peer_sources.push(q_lora.device.as_ref().ok_or_else(|| compute_error("operator MLA q_lora 缺少 device buffer"))?.clone());
        }
        if !packed_kv_replica {
            peer_sources.push(latent.device.as_ref().ok_or_else(|| compute_error("operator MLA latent 缺少 device buffer"))?.clone());
            peer_sources.push(k_rope.device.as_ref().ok_or_else(|| compute_error("operator MLA k_rope 缺少 device buffer"))?.clone());
        }
        let residual_replica = residual.replica.as_ref().filter(|replica| replica.device_id == peer.device_id && replica.device.bytes() == rows * hidden * replica.dtype.element_bytes()).cloned();
        let residual = f32_tensor(self, residual)?;
        let residual_device = residual.device.as_ref().ok_or_else(|| compute_error("operator MLA residual 缺少 F32 device buffer"))?.clone();
        let copy_residual = residual_replica.is_none();
        if copy_residual {
            let stable_residual = self.tensor_to_stable_deferred(residual.clone())?;
            peer_sources.push(stable_residual.device.ok_or_else(|| compute_error("operator MLA stable residual 缺少 device buffer"))?);
        }
        let (peer_cosine, peer_sine) = ops::hip::resident_rope_tables(peer.device_id, cosine, sine, mla.qk_rope_head_dim / 2, position..position + rows).map_err(compute_error)?;
        let peer_q_b = weights.peer_q_b.clone();
        let peer_kv_b = weights.peer_kv_b.clone();
        let peer_o = weights.peer_o.clone();
        let owner_mla = crate::attention::mla::MlaSpec {
            q_lora_rank: mla.q_lora_rank,
            kv_lora_rank: mla.kv_lora_rank,
            qk_rope_head_dim: mla.qk_rope_head_dim,
            q_projection_size: owner_q,
            kv_projection_size: owner_kv,
            num_heads: owner_heads,
            rope_theta: mla.rope_theta,
            rotary_layout: mla.rotary_layout,
        };
        let peer_mla = crate::attention::mla::MlaSpec { q_projection_size: peer_q, kv_projection_size: peer_kv, num_heads: peer_heads, ..owner_mla.clone() };
        let peer_host_selection = host_selection_arc.clone();
        let q_dtype = normalized_q_lora.dtype;
        let latent_dtype = latent.dtype;
        let rope_dtype = k_rope.dtype;
        // packed 路径只同步两次 host enqueue：peer query 已经排入，随后 compact
        // KV copy 与 peer attention 在同一 stream 上自然串联；没有 device wait。
        let (query_ready_sender, query_ready_receiver) = if packed_kv_replica {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            (Some(sender), Some(receiver))
        } else {
            (None, None)
        };
        let (kv_ready_sender, kv_ready_receiver) = if packed_kv_replica {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            (Some(sender), Some(receiver))
        } else {
            (None, None)
        };
        // P2P/event 由 owner submission 线程发起，使源 buffer 与 event 能被
        // owner stage completion 正确接管；peer worker 只提交计算 kernel。
        let transferred = if peer_sources.is_empty() {
            Vec::new()
        } else {
            ops::hip::DeviceBuffer::copy_stable_group_to_device_ordered_async_retained_by(&peer_sources, peer.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} operator MLA input owner->peer: {error}")))?
        };
        let ticket = worker.submit(move |peer| {
            peer.activate().map_err(compute_error)?;
            let mut transferred = transferred;
            let expected = usize::from(!presubmitted_query) + if packed_kv_replica { 0 } else { 2 } + usize::from(copy_residual);
            if transferred.len() != expected {
                return Err(compute_error(format!("L{layer} operator MLA peer 输入数量异常")));
            }
            let peer_q_lora = match pending_peer_q_lora {
                Some(query) => query,
                None => device_tensor_with_dtype(transferred.remove(0), rows, peer_mla.q_lora_rank, q_dtype),
            };
            let (peer_latent, peer_rope) = if packed_kv_replica {
                (None, None)
            } else {
                (Some(device_tensor_with_dtype(transferred.remove(0), rows, peer_mla.kv_lora_rank, latent_dtype)), Some(device_tensor_with_dtype(transferred.remove(0), rows, peer_mla.qk_rope_head_dim, rope_dtype)))
            };
            let peer_residual = match residual_replica {
                Some(residual) => {
                    let tensor = device_tensor_with_arc(residual.device, rows, hidden, residual.dtype);
                    let tensor = peer.tensor_to_stable_deferred(peer.tensor_as_f32(tensor)?)?;
                    tensor.device.ok_or_else(|| compute_error(format!("L{layer} operator MLA peer residual 缺失")))?
                }
                None => Arc::new(transferred.remove(0)),
            };
            let peer_query = peer.linear(&peer_q_lora, &peer_q_b)?;
            let peer_query_device = peer_query.device.as_deref().ok_or_else(|| compute_error("operator MLA peer query 缺少 device buffer"))?;
            let peer_query = ops::hip::try_rope_with_resident_tables_f32(peer.device_id, peer_query_device, rows, peer_q, peer_heads, peer_mla.qk_rope_head_dim, peer_mla.rotary_layout, position, &peer_cosine, &peer_sine, false)
                .map_err(compute_error)?;
            let peer_query = device_tensor_f32(peer_query, rows, peer_q);
            if let Some(sender) = query_ready_sender {
                sender.send(()).map_err(|_| compute_error(format!("L{layer} operator MLA query ready 通道提前关闭")))?;
                kv_ready_receiver.expect("packed KV 必有 ready receiver").recv().map_err(|_| compute_error(format!("L{layer} operator MLA packed KV ready 通道提前关闭")))?;
            }
            let mut cache = peer_cache.lock().map_err(|_| compute_error(format!("L{layer} operator peer KV 锁中毒")))?;
            if let (Some(peer_latent), Some(peer_rope)) = (peer_latent.as_ref(), peer_rope.as_ref()) {
                cache.append_mla(&peer, layer, peer_latent, peer_rope)?;
            }
            let attention = paged_mla_attention_with_selection(&peer, &peer_query, &cache, &peer_kv_b, layer, &peer_mla, peer_selection.as_deref(), peer_host_selection.as_deref().map(Vec::as_slice), selection_width)?;
            drop(cache);
            let partial = peer.tensor_to_stable_deferred(peer.tensor_as_f32(peer.linear(&attention, &peer_o)?)?)?;
            let partial = partial.device.ok_or_else(|| compute_error(format!("L{layer} operator peer partial 缺少 device buffer")))?;
            Ok((partial, peer_residual))
        })?;
        worker.retain_for_stage(peer_sources)?;

        // resident_rope_tables/cross-device copy 都可能把当前 host 线程留在
        // peer context；这里只恢复 device，不能把 owner 切回 context 默认流。
        ops::hip::set_device(self.device_id).map_err(compute_error)?;
        let owner_query = if packed_kv_replica {
            // peer query 与 owner append 并行；compact P2P 等 owner append 的 event，
            // 随后的 owner query 又可与 peer copy 重叠。
            cache.append_mla(self, layer, &latent, &k_rope)?;
            query_ready_receiver.expect("packed KV 必有 query receiver").recv().map_err(|_| compute_error(format!("L{layer} operator MLA peer query 提交失败")))?;
            let packed_sources = cache.replicate_operator_mla_append(self, &peer, layer, position, rows)?;
            worker.retain_for_stage(packed_sources)?;
            kv_ready_sender.expect("packed KV 必有 ready sender").send(()).map_err(|_| compute_error(format!("L{layer} operator MLA peer packed KV 等待失败")))?;
            match pending_owner_query {
                Some(query) => {
                    let main_stream = ops::hip::active_compute_stream() as usize;
                    ops::hip::order_stream_after(self.device_id, pending_owner_query_stream.expect("pending owner query 必有 stream"), main_stream).map_err(compute_error)?;
                    query
                }
                None => {
                    let query = self.linear(q_lora.as_ref().expect("普通 operator query 必有 q_lora"), &weights.owner_q_b)?;
                    self.rope(&query, owner_heads, mla.qk_rope_head_dim, mla.rotary_layout, position, cosine, sine)?
                }
            }
        } else {
            cache.append_mla(self, layer, &latent, &k_rope)?;
            match pending_owner_query {
                Some(query) => {
                    let main_stream = ops::hip::active_compute_stream() as usize;
                    ops::hip::order_stream_after(self.device_id, pending_owner_query_stream.expect("pending owner query 必有 stream"), main_stream).map_err(compute_error)?;
                    query
                }
                None => {
                    let query = self.linear(q_lora.as_ref().expect("普通 operator query 必有 q_lora"), &weights.owner_q_b)?;
                    self.rope(&query, owner_heads, mla.qk_rope_head_dim, mla.rotary_layout, position, cosine, sine)?
                }
            }
        };
        let owner_attention = paged_mla_attention_with_selection(self, &owner_query, cache, &weights.owner_kv_b, layer, &owner_mla, selection.as_deref(), host_selection, selection_width)?;
        let owner_partial = self.tensor_to_stable_deferred(self.tensor_as_f32(self.linear(&owner_attention, &weights.owner_o)?)?)?;
        let owner_partial = owner_partial.device.ok_or_else(|| compute_error("operator MLA owner partial 缺少 device buffer"))?;
        let (peer_partial, peer_residual) = ticket.wait()?;
        let (owner_output, peer_output) = ops::hip::DeviceBuffer::join_peer_partials_residual_ordered_async_retained_by(&owner_partial, &peer_partial, &residual_device, &peer_residual, rows * hidden, self.device_id)
            .map_err(|error| compute_error(format!("L{layer} operator MLA direct partial join: {error}")))?;
        worker.retain_for_stage(vec![owner_partial, peer_partial])?;
        let mut output = device_tensor_f32(owner_output, rows, hidden);
        output.replica = Some(RocmTensorReplica { device_id: peer.device_id, dtype: RocmTensorDType::F32, device: Arc::new(peer_output) });
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn cooperative_mla_prefill_add_impl(
        &self,
        layer: usize,
        experts: &RocmPrefillExperts,
        normalized_q_lora: &RocmTensor,
        latent: &RocmTensor,
        k_rope: &RocmTensor,
        residual: &RocmTensor,
        cache: Option<&mut RocmKvCache>,
        dsa_state: Option<&RocmDsaState>,
        position: usize,
        cosine: &[f32],
        sine: &[f32],
        mla: &crate::attention::mla::MlaSpec,
        dsa: &crate::attention::dsa::DsaSpec,
    ) -> Result<RocmTensor, BackendError> {
        let cache = cache.ok_or_else(|| compute_error(format!("L{layer} cooperative MLA 必须提供 cache")))?;
        let (peer, weights) = experts.cooperative_mla_layer(layer)?;
        let o_columns_valid =
            if weights.o_head_sharded() { weights.owner_o.cols.checked_add(weights.peer_o.cols) == Some(mla.q_projection_size) } else { weights.owner_o.cols == mla.q_projection_size && weights.peer_o.cols == mla.q_projection_size };
        if normalized_q_lora.rows == 0
            || normalized_q_lora.rows != latent.rows
            || latent.rows != k_rope.rows
            || residual.rows != latent.rows
            || normalized_q_lora.cols != mla.q_lora_rank
            || weights.owner_q_b.rows != mla.q_projection_size
            || weights.owner_q_b.cols != mla.q_lora_rank
            || latent.cols != mla.kv_lora_rank
            || k_rope.cols != mla.qk_rope_head_dim
            || residual.cols != weights.owner_o.rows
            || weights.owner_o_full.rows != residual.cols
            || weights.owner_o_full.cols != mla.q_projection_size
            || !mla.num_heads.is_multiple_of(2)
            || !o_columns_valid
        {
            return Err(compute_error(format!(
                "L{layer} cooperative MLA shape 非法: q=[{},{}] latent=[{},{}] rope=[{},{}] residual=[{},{}] heads={} q_proj={} kv_proj={}",
                normalized_q_lora.rows, normalized_q_lora.cols, latent.rows, latent.cols, k_rope.rows, k_rope.cols, residual.rows, residual.cols, mla.num_heads, mla.q_projection_size, mla.kv_projection_size,
            )));
        }
        let diagnose = normalized_q_lora.rows == 1;
        let trace_started = std::time::Instant::now();
        let mut trace_last = trace_started;
        let mut trace_step = |step: &str| {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(trace_last);
            trace_last = now;
            if diagnose && elapsed >= std::time::Duration::from_millis(5) {
                eprintln!("[rocm-pair-mla-step] layer={layer} position={position} step={step} wall_ms={:.3} total_ms={:.3}", elapsed.as_secs_f64() * 1000.0, now.duration_since(trace_started).as_secs_f64() * 1000.0,);
            }
        };
        // owner stage stream 在这里一次性交给 owner default stream；直到 attention
        // partial join 结束才交还，期间两卡各自顺序排队，不插入 host 同步。
        ops::hip::set_device(self.device_id).map_err(compute_error)?;
        let owner_stream = ops::hip::active_compute_stream() as usize;
        ops::hip::order_stream_after(self.device_id, owner_stream, 0).map_err(compute_error)?;
        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        trace_step("entry-handoff");

        // Prefill 按 query rows 均分：每行 q_b 只算一次。peer 只接收自己的
        // q_lora 半区，随后两边交换已经投影并完成 RoPE 的 query 半区。
        // decode 单行仍保留原来的完整 query 路径。
        let query_row_split = normalized_q_lora.rows > 1 && normalized_q_lora.rows.is_multiple_of(2);
        let half_query_rows = normalized_q_lora.rows / 2;
        let (owner_q_lora, q_transfer) = if query_row_split {
            let owner_q_lora = tensor_row_view(normalized_q_lora, 0, half_query_rows, "pair owner q_lora")?;
            let peer_q_lora = tensor_row_view(normalized_q_lora, half_query_rows, half_query_rows, "pair peer q_lora")?;
            let peer_q_lora = self.tensor_to_stable_deferred(peer_q_lora)?;
            let transfer = peer_q_lora.device.as_ref().ok_or_else(|| compute_error("pair peer q_lora 缺少 device buffer"))?.clone();
            (Some(owner_q_lora), transfer)
        } else {
            let stable_q = self.tensor_to_stable_deferred(normalized_q_lora.clone())?;
            let transfer = stable_q.device.as_ref().ok_or_else(|| compute_error("stable q 缺少 device buffer"))?.clone();
            (None, transfer)
        };
        trace_step("query-stabilize");
        cache.append_cooperative_mla_rope(self, &peer, layer, latent, k_rope, mla.qk_rope_head_dim, mla.rotary_layout, position, cosine, sine)?;
        trace_step("cache-append");
        let query_start = position;
        let selection_width =
            dsa_state.filter(|state| state.device_selection(normalized_q_lora.rows, query_start).is_some() || state.host_selection(normalized_q_lora.rows, query_start).is_some()).map_or(dsa.top_k, RocmDsaState::selection_width);
        let source_selection = dsa_state.and_then(|state| state.device_selection_arc(normalized_q_lora.rows, query_start));
        let cached_selection = source_selection
            .as_ref()
            .and_then(|selection| cache.cached_cooperative_selection(peer.device_id, selection))
            .map(|(owner, owner_counts, peer, peer_counts, width)| (owner.clone(), owner_counts.clone(), peer.clone(), peer_counts.clone(), width));
        let fresh_selection = if cached_selection.is_none() {
            source_selection.as_ref().map(|selection| ops::hip::try_split_paged_mla_selection_parity(self.device_id, selection, normalized_q_lora.rows, selection_width, ROCM_KV_BLOCK_SIZE).map_err(compute_error)).transpose()?
        } else {
            None
        };
        let mut sources = vec![q_transfer];
        if let Some(selection) = fresh_selection.as_ref() {
            sources.push(selection.peer.clone());
            sources.push(selection.peer_counts.clone());
        }
        trace_step("selection-prepare");

        let profile_pair = ops::hip::device_profile_enabled();
        peer.activate().map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_begin(peer.device_id, "glm_pair_attn_peer_input").map_err(compute_error)?;
        }
        let mut transferred = ops::hip::DeviceBuffer::copy_stable_group_to_device_ordered_async_retained_by(&sources, peer.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} MLA owner->peer: {error}")))?;
        trace_step("owner-to-peer");
        if transferred.len() != sources.len() {
            return Err(compute_error(format!("L{layer} cooperative MLA P2P 数量={}/{} 异常", transferred.len(), sources.len())));
        }
        let peer_q_lora = device_tensor_with_dtype(transferred.remove(0), if query_row_split { half_query_rows } else { normalized_q_lora.rows }, normalized_q_lora.cols, normalized_q_lora.dtype);
        let (owner_selection, owner_selection_counts, peer_selection, peer_selection_counts) = match (cached_selection, fresh_selection, source_selection) {
            (Some((owner, owner_counts, peer, peer_counts, width)), None, Some(_)) if width == selection_width => (Some(owner), Some(owner_counts), Some(peer), Some(peer_counts)),
            (None, Some(split), Some(source)) if transferred.len() == 2 => {
                let peer_selection = Arc::new(transferred.remove(0));
                let peer_counts = Arc::new(transferred.remove(0));
                let owner_selection = split.owner;
                let owner_counts = split.owner_counts;
                cache.cache_cooperative_selection(peer.device_id, source, owner_selection.clone(), owner_counts.clone(), peer_selection.clone(), peer_counts.clone(), selection_width)?;
                (Some(owner_selection), Some(owner_counts), Some(peer_selection), Some(peer_counts))
            }
            (None, None, None) if transferred.is_empty() => (None, None, None, None),
            _ => return Err(compute_error(format!("L{layer} cooperative MLA selection split/P2P 状态不一致"))),
        };

        if profile_pair {
            ops::hip::device_profile_scope_operator(peer.device_id, "glm_pair_attn_peer_query").map_err(compute_error)?;
        }
        let decode_q_b_shards = (normalized_q_lora.rows == 1).then(|| weights.decode_q_b_shards()).flatten();
        let query_head_count = if decode_q_b_shards.is_some() { mla.num_heads / 2 } else { mla.num_heads };
        let peer_q_b = decode_q_b_shards.map_or(&weights.peer_q_b, |(_, peer)| peer);
        let peer_query = peer.linear(&peer_q_lora, peer_q_b)?;
        let peer_query = peer.rope(&peer_query, query_head_count, mla.qk_rope_head_dim, mla.rotary_layout, if query_row_split { position + half_query_rows } else { position }, cosine, sine)?;
        let peer_query = peer.tensor_to_stable_deferred(peer_query)?;
        trace_step("peer-query");

        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_begin(self.device_id, "glm_pair_attn_owner_query").map_err(compute_error)?;
        }
        let owner_q_b = decode_q_b_shards.map_or(&weights.owner_q_b, |(owner, _)| owner);
        let owner_query = if let Some(owner_q_lora) = owner_q_lora.as_ref() {
            let query = self.linear(owner_q_lora, &weights.owner_q_b)?;
            self.rope(&query, mla.num_heads, mla.qk_rope_head_dim, mla.rotary_layout, position, cosine, sine)?
        } else {
            let query = self.linear(normalized_q_lora, owner_q_b)?;
            self.rope(&query, query_head_count, mla.qk_rope_head_dim, mla.rotary_layout, position, cosine, sine)?
        };
        let owner_query = self.tensor_to_stable_deferred(owner_query)?;
        trace_step("owner-query");
        if profile_pair {
            ops::hip::device_profile_scope_operator(self.device_id, "glm_pair_attn_owner_query_exchange").map_err(compute_error)?;
        }

        let (owner_query, peer_query) = if decode_q_b_shards.is_some() {
            // q_b 是 decode attention 的最大单个权重流量。两卡各算连续一半 head，
            // 再交换 32KiB 左右的投影结果；head 顺序不变，RoPE 可在交换前独立完成。
            let half_columns = mla.q_projection_size / 2;
            let owner_source = owner_query.device.as_ref().ok_or_else(|| compute_error("pair owner query head shard 缺少 device buffer"))?.clone();
            let peer_source = peer_query.device.as_ref().ok_or_else(|| compute_error("pair peer query head shard 缺少 device buffer"))?.clone();
            // 先在两张卡各自的 producer stream 记录 ready，再同时排入
            // 两个方向的 copy。连续两次单向 copy 会让 peer ready 落在
            // owner->peer 入向 copy 之后，把本可并行的 query half 交换串行化。
            let (mut owner_on_peer, mut peer_on_owner) =
                ops::hip::DeviceBuffer::exchange_stable_groups_ordered_async_retained_by(&[owner_source], &[peer_source], self.device_id).map_err(|error| compute_error(format!("L{layer} MLA query head exchange: {error}")))?;
            let owner_on_peer = owner_on_peer.pop().ok_or_else(|| compute_error(format!("L{layer} MLA owner query head->peer 缺失")))?;
            let peer_on_owner = peer_on_owner.pop().ok_or_else(|| compute_error(format!("L{layer} MLA peer query head->owner 缺失")))?;
            let owner_on_peer = device_tensor_f32(owner_on_peer, 1, half_columns);
            let peer_on_owner = device_tensor_f32(peer_on_owner, 1, half_columns);
            peer.activate().map_err(compute_error)?;
            let peer_full = peer.concat_columns(&owner_on_peer, &peer_query)?;
            ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            let owner_full = self.concat_columns(&owner_query, &peer_on_owner)?;
            trace_step("query-head-join");
            (owner_full, peer_full)
        } else {
            (owner_query, peer_query)
        };

        if query_row_split {
            // prefill MLA 最终以 BF16 query 进入 WMMA；跨卡先压成 BF16，把
            // query 表传输减半。两向传输各走本卡 transfer stream，与另一象限
            // 的本地 MLA 扫描重叠，helper 在消费远端象限前再显式汇合。
            peer.activate().map_err(compute_error)?;
            let peer_query_transfer = peer.tensor_to_stable_deferred(peer.tensor_as_bf16(peer_query.clone())?)?;
            ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            let owner_query_transfer = self.tensor_to_stable_deferred(self.tensor_as_bf16(owner_query.clone())?)?;
            let owner_query_source = owner_query_transfer.device.as_ref().ok_or_else(|| compute_error("pair owner BF16 query 缺少 device buffer"))?.clone();
            let peer_query_source = peer_query_transfer.device.as_ref().ok_or_else(|| compute_error("pair peer BF16 query 缺少 device buffer"))?.clone();
            let peer_transfer_stream = ops::hip::cooperative_peer_stream(peer.device_id).map_err(compute_error)?;
            let owner_transfer_stream = ops::hip::cooperative_peer_stream(self.device_id).map_err(compute_error)?;
            ops::hip::activate_compute_stream(peer.device_id, peer_transfer_stream).map_err(compute_error)?;
            let owner_query_on_peer =
                owner_query_source.copy_stable_to_device_ordered_async_on_streams_retained_by(peer.device_id, self.device_id, 0, peer_transfer_stream).map_err(|error| compute_error(format!("L{layer} MLA owner query->peer: {error}")))?;
            ops::hip::activate_compute_stream(self.device_id, owner_transfer_stream).map_err(compute_error)?;
            let peer_query_on_owner =
                peer_query_source.copy_stable_to_device_ordered_async_on_streams_retained_by(self.device_id, self.device_id, 0, owner_transfer_stream).map_err(|error| compute_error(format!("L{layer} MLA peer query->owner: {error}")))?;
            ops::hip::activate_compute_stream(peer.device_id, 0).map_err(compute_error)?;
            ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            let owner_query_on_peer = device_tensor_with_dtype(owner_query_on_peer, half_query_rows, mla.q_projection_size, RocmTensorDType::Bf16);
            let peer_query_on_owner = device_tensor_with_dtype(peer_query_on_owner, half_query_rows, mla.q_projection_size, RocmTensorDType::Bf16);
            return self.cooperative_mla_query_row_shards_finish(
                layer,
                &peer,
                &weights,
                cache,
                residual,
                &owner_query,
                &peer_query_on_owner,
                &owner_query_on_peer,
                &peer_query,
                owner_selection.as_ref(),
                owner_selection_counts.as_ref(),
                peer_selection.as_ref(),
                peer_selection_counts.as_ref(),
                selection_width,
                position,
                mla,
                owner_stream,
                profile_pair,
            );
        }

        // 两卡分别扫描自己的 parity block；prefill 的完整 query 由 token 行半区
        // 交换得到，decode 的完整 query 由 q_b head 半区交换得到。
        peer.activate().map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_end(peer.device_id).map_err(compute_error)?;
            ops::hip::device_profile_scope_begin(peer.device_id, "glm_pair_attn_peer_scan").map_err(compute_error)?;
        }
        let peer_cache = cache.cooperative_peer_cache(peer.device_id)?;
        let peer_shard = paged_mla_attention_shard(&peer, &peer_query, peer_cache, &weights.peer_kv_b, layer, mla, peer_selection.as_deref(), peer_selection_counts.as_deref(), selection_width, 1, position)?;
        let peer_shard = if weights.o_head_sharded() { stable_sequence_shard(peer_shard)? } else { peer_shard };
        trace_step("peer-attention");
        if profile_pair {
            ops::hip::device_profile_scope_end(peer.device_id).map_err(compute_error)?;
            ops::hip::device_profile_scope_begin(peer.device_id, "glm_pair_attn_peer_post").map_err(compute_error)?;
        }

        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
            ops::hip::device_profile_scope_begin(self.device_id, "glm_pair_attn_owner_scan").map_err(compute_error)?;
        }
        let owner_shard = paged_mla_attention_shard(self, &owner_query, cache, &weights.owner_kv_b, layer, mla, owner_selection.as_deref(), owner_selection_counts.as_deref(), selection_width, 0, position)?;
        let owner_shard = if weights.o_head_sharded() { stable_sequence_shard(owner_shard)? } else { owner_shard };
        trace_step("owner-attention");
        if profile_pair {
            ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
            ops::hip::device_profile_scope_begin(self.device_id, "glm_pair_attn_owner_post").map_err(compute_error)?;
        }
        if normalized_q_lora.rows == 1 && ops::hip::options().cooperative_mla_decode_replicated && weights.decode_replicated() {
            let (owner_o_full, peer_o_full) = weights.full_o().expect("decode_replicated 已校验完整 o_proj");
            let (mut owner_on_peer, mut peer_on_owner) = exchange_sequence_shards(&[&owner_shard], &[&peer_shard], self.device_id)?;
            let owner_on_peer = owner_on_peer.pop().expect("单 shard P2P 已校验");
            let peer_on_owner = peer_on_owner.pop().expect("单 shard P2P 已校验");

            let residual = f32_tensor(self, residual)?;
            let peer_residual = match residual.replica.as_ref() {
                Some(replica) if replica.device_id == peer.device_id && replica.dtype == RocmTensorDType::F32 && replica.device.bytes() == residual.rows * residual.cols * std::mem::size_of::<f32>() => {
                    RocmTensor { data: Vec::new(), rows: residual.rows, cols: residual.cols, dtype: RocmTensorDType::F32, layout: RocmTensorLayout::RowMajor, device: Some(replica.device.clone()), replica: None }
                }
                _ => {
                    let stable = self.tensor_to_stable_deferred(residual.clone())?;
                    let source = stable.device.as_ref().ok_or_else(|| compute_error("ROCm cooperative MLA residual 缺少 device buffer"))?;
                    let copied = source.copy_stable_to_device_ordered_async_retained_by(peer.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} MLA residual owner->peer: {error}")))?;
                    device_tensor_f32(copied, residual.rows, residual.cols)
                }
            };

            peer.activate().map_err(compute_error)?;
            // 两边都固定 owner shard 在前、peer shard 在后，保证 softmax merge
            // 与后续 o_proj 的归约顺序一致，peer hidden 才能作为精确副本延续。
            let peer_projected = project_merged_sequence_heads(&peer, &owner_on_peer, &peer_shard, &weights.peer_kv_b, peer_o_full, 1, 0, mla.num_heads, mla)?;
            let peer_output = peer.add(&peer_projected, &peer_residual)?;
            let peer_output_device = peer_output.device.as_ref().ok_or_else(|| compute_error("ROCm cooperative MLA peer output 缺少 device buffer"))?.clone();
            if profile_pair {
                ops::hip::device_profile_scope_end(peer.device_id).map_err(compute_error)?;
            }

            ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            let owner_projected = project_merged_sequence_heads(self, &owner_shard, &peer_on_owner, &weights.owner_kv_b, owner_o_full, 1, 0, mla.num_heads, mla)?;
            let mut output = self.add(&owner_projected, &residual)?;
            output.replica = Some(RocmTensorReplica { device_id: peer.device_id, dtype: RocmTensorDType::F32, device: peer_output_device });
            if profile_pair {
                ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
            }
            ops::hip::order_stream_after(self.device_id, 0, owner_stream).map_err(compute_error)?;
            ops::hip::activate_compute_stream(self.device_id, owner_stream).map_err(compute_error)?;
            trace_step("replicated-output");
            return Ok(output);
        }

        if normalized_q_lora.rows == 1 && weights.o_head_sharded() && ops::hip::options().cooperative_mla_decode_full_merge {
            // decode 已经必须把 peer 的 weighted/stats 送到 owner 完成稳定 softmax。
            // owner 直接物化全部 head 并执行完整 o_proj，删除 full-hidden partial
            // 的第二次 P2P 与归约；peer 不再做 attention 尾部投影。
            let mut peer_on_owner = copy_sequence_shards_to_device(&[&peer_shard], self.device_id, self.device_id)?;
            let peer_on_owner = peer_on_owner.pop().expect("单 shard P2P 已校验");
            if profile_pair {
                peer.activate().map_err(compute_error)?;
                ops::hip::device_profile_scope_end(peer.device_id).map_err(compute_error)?;
                ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            }
            let projected = project_merged_sequence_heads(self, &owner_shard, &peer_on_owner, &weights.owner_kv_b, &weights.owner_o_full, 1, 0, mla.num_heads, mla)?;
            trace_step("full-output-projection");
            let residual = f32_tensor(self, residual)?;
            let output = self.add(&projected, &residual)?;
            trace_step("residual-add");
            if profile_pair {
                ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
            }
            ops::hip::order_stream_after(self.device_id, 0, owner_stream).map_err(compute_error)?;
            ops::hip::activate_compute_stream(self.device_id, owner_stream).map_err(compute_error)?;
            trace_step("exit-handoff");
            return Ok(output);
        }

        let (local_partial, stable_peer_partial) = if weights.o_head_sharded() {
            // 两张卡先完成各自的 sequence shard，再各自通过 peer BAR 读取另一半
            // weighted latent，只物化自己的 head 半区并执行半列 o_proj。
            let (mut owner_on_peer, mut peer_on_owner) = exchange_sequence_shards(&[&owner_shard], &[&peer_shard], self.device_id)?;
            let owner_on_peer = owner_on_peer.pop().expect("单 shard P2P 已校验");
            let peer_on_owner = peer_on_owner.pop().expect("单 shard P2P 已校验");
            let head_count = mla.num_heads / 2;
            peer.activate().map_err(compute_error)?;
            let peer_partial = project_merged_sequence_heads(&peer, &peer_shard, &owner_on_peer, &weights.peer_kv_b, &weights.peer_o, normalized_q_lora.rows, head_count, head_count, mla)?;
            let stable_peer_partial = peer.tensor_to_stable_deferred(peer.tensor_as_bf16(peer_partial)?)?;
            let stable_peer_partial = stable_peer_partial.device.as_ref().ok_or_else(|| compute_error("ROCm cooperative MLA peer partial 缺少 device buffer"))?.clone();
            if profile_pair {
                ops::hip::device_profile_scope_end(peer.device_id).map_err(compute_error)?;
            }
            ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            let local_partial = project_merged_sequence_heads(self, &owner_shard, &peer_on_owner, &weights.owner_kv_b, &weights.owner_o, normalized_q_lora.rows, 0, head_count, mla)?;
            (local_partial, stable_peer_partial)
        } else {
            // GGUF 仍使用完整 o_proj：只交换 softmax stats，每卡把自己的
            // sequence contribution 算到 full-hidden 后再归约。
            let peer_stats = if peer_shard.stats.is_async_allocated() { Arc::new(peer_shard.stats.copy_to_stable_deferred().map_err(compute_error)?) } else { peer_shard.stats.clone() };
            let owner_stats = if owner_shard.stats.is_async_allocated() { Arc::new(owner_shard.stats.copy_to_stable_deferred().map_err(compute_error)?) } else { owner_shard.stats.clone() };
            let owner_stats_on_peer = owner_stats.copy_stable_to_device_ordered_async_retained_by(peer.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} MLA owner stats->peer: {error}")))?;
            let peer_stats_on_owner = peer_stats.copy_stable_to_device_ordered_async_retained_by(self.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} MLA peer stats->owner: {error}")))?;
            peer.activate().map_err(compute_error)?;
            let peer_partial = project_local_sequence_partial(&peer, &peer_shard, &peer_stats, &owner_stats_on_peer, &weights.peer_kv_b, &weights.peer_o, normalized_q_lora.rows, mla, profile_pair)?;
            let stable_peer_partial = peer.tensor_to_stable_deferred(peer.tensor_as_bf16(peer_partial)?)?;
            let stable_peer_partial = stable_peer_partial.device.as_ref().ok_or_else(|| compute_error("ROCm cooperative MLA peer partial 缺少 device buffer"))?.clone();
            if profile_pair {
                ops::hip::device_profile_scope_end(peer.device_id).map_err(compute_error)?;
            }
            ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            let local_partial = project_local_sequence_partial(self, &owner_shard, &owner_stats, &peer_stats_on_owner, &weights.owner_kv_b, &weights.owner_o, normalized_q_lora.rows, mla, profile_pair)?;
            (local_partial, stable_peer_partial)
        };
        trace_step("output-projection");
        if profile_pair {
            ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
            ops::hip::device_profile_scope_begin(self.device_id, "glm_pair_attn_join").map_err(compute_error)?;
        }
        let peer_partial_on_owner = stable_peer_partial.copy_stable_to_device_ordered_async_retained_by(self.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} MLA peer partial->owner: {error}")))?;
        trace_step("peer-partial-to-owner");

        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        let local_device = local_partial.device.as_deref().ok_or_else(|| compute_error("ROCm cooperative MLA local partial 缺少 device buffer"))?;
        let residual = f32_tensor(self, residual)?;
        let residual_device = residual.device.as_deref().ok_or_else(|| compute_error("ROCm cooperative MLA residual 缺少 device buffer"))?;
        let output = ops::hip::try_ct_cooperative_partial_join_f32(self.device_id, local_device, &peer_partial_on_owner, residual_device, residual.rows, residual.cols).map_err(compute_error)?;
        trace_step("partial-join");
        if profile_pair {
            ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
        }
        ops::hip::order_stream_after(self.device_id, 0, owner_stream).map_err(compute_error)?;
        ops::hip::activate_compute_stream(self.device_id, owner_stream).map_err(compute_error)?;
        trace_step("exit-handoff");
        Ok(device_tensor_f32(output, residual.rows, residual.cols))
    }

    #[allow(clippy::too_many_arguments)]
    fn cooperative_mla_query_row_shards_finish(
        &self,
        layer: usize,
        peer: &RocmContext,
        weights: &super::expert::RocmCooperativeMlaWeights,
        cache: &mut RocmKvCache,
        residual: &RocmTensor,
        owner_q0: &RocmTensor,
        owner_q1: &RocmTensor,
        peer_q0: &RocmTensor,
        peer_q1: &RocmTensor,
        owner_selection: Option<&Arc<ops::hip::DeviceBuffer>>,
        owner_selection_counts: Option<&Arc<ops::hip::DeviceBuffer>>,
        peer_selection: Option<&Arc<ops::hip::DeviceBuffer>>,
        peer_selection_counts: Option<&Arc<ops::hip::DeviceBuffer>>,
        selection_width: usize,
        position: usize,
        mla: &crate::attention::mla::MlaSpec,
        owner_stream: usize,
        profile_pair: bool,
    ) -> Result<RocmTensor, BackendError> {
        let rows = owner_q0.rows;
        if rows == 0
            || owner_q1.rows != rows
            || peer_q0.rows != rows
            || peer_q1.rows != rows
            || [owner_q0, owner_q1, peer_q0, peer_q1].iter().any(|query| query.cols != mla.q_projection_size)
            || owner_q0.dtype != RocmTensorDType::F32
            || peer_q1.dtype != RocmTensorDType::F32
            || owner_q1.dtype != RocmTensorDType::Bf16
            || peer_q0.dtype != RocmTensorDType::Bf16
        {
            return Err(compute_error(format!(
                "L{layer} cooperative MLA query row shard shape 非法: owner={:?}/{:?} peer={:?}/{:?}",
                (owner_q0.rows, owner_q0.cols, owner_q0.dtype),
                (owner_q1.rows, owner_q1.cols, owner_q1.dtype),
                (peer_q0.rows, peer_q0.cols, peer_q0.dtype),
                (peer_q1.rows, peer_q1.cols, peer_q1.dtype),
            )));
        }
        let selection_rows = match (owner_selection, owner_selection_counts, peer_selection, peer_selection_counts) {
            (Some(owner), Some(owner_counts), Some(peer_selection), Some(peer_counts)) => {
                let table_row_bytes = selection_width.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| compute_error("ROCm pair selection row bytes 溢出"))?;
                Some((
                    buffer_row_view(owner, 0, rows, table_row_bytes, "owner q0 selection")?,
                    buffer_row_view(owner_counts, 0, rows, std::mem::size_of::<u32>(), "owner q0 selection counts")?,
                    buffer_row_view(owner, rows, rows, table_row_bytes, "owner q1 selection")?,
                    buffer_row_view(owner_counts, rows, rows, std::mem::size_of::<u32>(), "owner q1 selection counts")?,
                    buffer_row_view(peer_selection, 0, rows, table_row_bytes, "peer q0 selection")?,
                    buffer_row_view(peer_counts, 0, rows, std::mem::size_of::<u32>(), "peer q0 selection counts")?,
                    buffer_row_view(peer_selection, rows, rows, table_row_bytes, "peer q1 selection")?,
                    buffer_row_view(peer_counts, rows, rows, std::mem::size_of::<u32>(), "peer q1 selection counts")?,
                ))
            }
            (None, None, None, None) => None,
            _ => return Err(compute_error(format!("L{layer} cooperative MLA selection row shard 状态不完整"))),
        };

        peer.activate().map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_operator(peer.device_id, "glm_pair_attn_peer_mla_q1").map_err(compute_error)?;
        }
        let peer_cache = cache.cooperative_peer_cache(peer.device_id)?;
        let peer_q1_shard = paged_mla_attention_shard(
            peer,
            peer_q1,
            peer_cache,
            &weights.peer_kv_b,
            layer,
            mla,
            selection_rows.as_ref().map(|selection| selection.6.as_ref()),
            selection_rows.as_ref().map(|selection| selection.7.as_ref()),
            selection_width,
            1,
            position + rows,
        )?;
        let peer_transfer_stream = ops::hip::cooperative_peer_stream(peer.device_id).map_err(compute_error)?;
        ops::hip::order_stream_after(peer.device_id, peer_transfer_stream, 0).map_err(compute_error)?;
        let peer_q0 = peer.tensor_as_f32(peer_q0.clone())?;
        if profile_pair {
            ops::hip::device_profile_scope_operator(peer.device_id, "glm_pair_attn_peer_mla_q0").map_err(compute_error)?;
        }
        let peer_q0_shard = paged_mla_attention_shard(
            peer,
            &peer_q0,
            peer_cache,
            &weights.peer_kv_b,
            layer,
            mla,
            selection_rows.as_ref().map(|selection| selection.4.as_ref()),
            selection_rows.as_ref().map(|selection| selection.5.as_ref()),
            selection_width,
            1,
            position,
        )?;
        let (peer_q0_shard, peer_q1_shard) = if weights.o_head_sharded() { (stable_sequence_shard(peer_q0_shard)?, stable_sequence_shard(peer_q1_shard)?) } else { (peer_q0_shard, peer_q1_shard) };
        if profile_pair {
            ops::hip::device_profile_scope_operator(peer.device_id, "glm_pair_attn_peer_stats_exchange").map_err(compute_error)?;
        }

        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_operator(self.device_id, "glm_pair_attn_owner_mla_q0").map_err(compute_error)?;
        }
        let owner_q0_shard = paged_mla_attention_shard(
            self,
            owner_q0,
            cache,
            &weights.owner_kv_b,
            layer,
            mla,
            selection_rows.as_ref().map(|selection| selection.0.as_ref()),
            selection_rows.as_ref().map(|selection| selection.1.as_ref()),
            selection_width,
            0,
            position,
        )?;
        let owner_transfer_stream = ops::hip::cooperative_peer_stream(self.device_id).map_err(compute_error)?;
        ops::hip::order_stream_after(self.device_id, owner_transfer_stream, 0).map_err(compute_error)?;
        let owner_q1 = self.tensor_as_f32(owner_q1.clone())?;
        if profile_pair {
            ops::hip::device_profile_scope_operator(self.device_id, "glm_pair_attn_owner_mla_q1").map_err(compute_error)?;
        }
        let owner_q1_shard = paged_mla_attention_shard(
            self,
            &owner_q1,
            cache,
            &weights.owner_kv_b,
            layer,
            mla,
            selection_rows.as_ref().map(|selection| selection.2.as_ref()),
            selection_rows.as_ref().map(|selection| selection.3.as_ref()),
            selection_width,
            0,
            position + rows,
        )?;
        let (owner_q0_shard, owner_q1_shard) = if weights.o_head_sharded() { (stable_sequence_shard(owner_q0_shard)?, stable_sequence_shard(owner_q1_shard)?) } else { (owner_q0_shard, owner_q1_shard) };
        if profile_pair {
            ops::hip::device_profile_scope_operator(self.device_id, "glm_pair_attn_owner_stats_exchange").map_err(compute_error)?;
        }

        if ops::hip::options().cooperative_mla_prefill_row_output
            && weights.o_head_sharded()
            && let Some((owner_o_full, peer_o_full)) = weights.full_o()
        {
            // q0 只在 owner 生成最终 hidden，q1 只在 peer 生成最终 hidden。
            // 每个方向只传对应行的 P/stats，避免两卡都为全部 token 做半列
            // o_proj，attention 边界也不再归约两份 full-row hidden partial。
            let (mut owner_q1_on_peer, mut peer_q0_on_owner) = exchange_sequence_shards(&[&owner_q1_shard], &[&peer_q0_shard], self.device_id)?;
            let owner_q1_on_peer = owner_q1_on_peer.pop().expect("q1 shard P2P 已校验");
            let peer_q0_on_owner = peer_q0_on_owner.pop().expect("q0 shard P2P 已校验");

            peer.activate().map_err(compute_error)?;
            let peer_q1_output = project_merged_sequence_heads(peer, &owner_q1_on_peer, &peer_q1_shard, &weights.peer_kv_b, peer_o_full, rows, 0, mla.num_heads, mla)?;
            let peer_q1_output = peer.tensor_to_stable_deferred(peer_q1_output)?;
            let peer_q1_source = peer_q1_output.device.as_ref().ok_or_else(|| compute_error("ROCm cooperative MLA peer q1 output 缺少 device buffer"))?;
            if profile_pair {
                ops::hip::device_profile_scope_end(peer.device_id).map_err(compute_error)?;
            }

            ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            let owner_q0_output = project_merged_sequence_heads(self, &owner_q0_shard, &peer_q0_on_owner, &weights.owner_kv_b, owner_o_full, rows, 0, mla.num_heads, mla)?;
            let peer_q1_on_owner = peer_q1_source.copy_stable_to_device_ordered_async_retained_by(self.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} MLA q1 output peer->owner: {error}")))?;
            let peer_q1_on_owner = device_tensor_f32(peer_q1_on_owner, rows, residual.cols);
            let projected = <RocmContext as crate::backend::SegmentedTensorBackend>::concat_token_rows(self, &[&owner_q0_output, &peer_q1_on_owner])?;
            let output = self.add(residual, &projected)?;
            if profile_pair {
                ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
            }
            ops::hip::order_stream_after(self.device_id, 0, owner_stream).map_err(compute_error)?;
            ops::hip::activate_compute_stream(self.device_id, owner_stream).map_err(compute_error)?;
            return Ok(output);
        }

        let (owner_partial, peer_partial) = if weights.o_head_sharded() {
            let owner_on_peer = copy_sequence_shards_to_device(&[&owner_q0_shard, &owner_q1_shard], peer.device_id, self.device_id)?;
            let peer_on_owner = copy_sequence_shards_to_device(&[&peer_q0_shard, &peer_q1_shard], self.device_id, self.device_id)?;
            let head_count = mla.num_heads / 2;
            peer.activate().map_err(compute_error)?;
            let peer_q0_partial = project_merged_sequence_heads(peer, &peer_q0_shard, &owner_on_peer[0], &weights.peer_kv_b, &weights.peer_o, rows, head_count, head_count, mla)?;
            let peer_q1_partial = project_merged_sequence_heads(peer, &peer_q1_shard, &owner_on_peer[1], &weights.peer_kv_b, &weights.peer_o, rows, head_count, head_count, mla)?;
            let peer_q0_partial = peer.tensor_as_bf16(peer_q0_partial)?;
            let peer_q1_partial = peer.tensor_as_bf16(peer_q1_partial)?;
            let peer_partial = <RocmContext as crate::backend::SegmentedTensorBackend>::concat_token_rows(peer, &[&peer_q0_partial, &peer_q1_partial])?;
            let peer_partial = peer.tensor_to_stable_deferred(peer_partial)?;
            let peer_partial = peer_partial.device.as_ref().ok_or_else(|| compute_error("ROCm cooperative MLA peer partial 缺少 device buffer"))?.clone();
            if profile_pair {
                ops::hip::device_profile_scope_end(peer.device_id).map_err(compute_error)?;
            }
            ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            let owner_q0_partial = project_merged_sequence_heads(self, &owner_q0_shard, &peer_on_owner[0], &weights.owner_kv_b, &weights.owner_o, rows, 0, head_count, mla)?;
            let owner_q1_partial = project_merged_sequence_heads(self, &owner_q1_shard, &peer_on_owner[1], &weights.owner_kv_b, &weights.owner_o, rows, 0, head_count, mla)?;
            let owner_partial = <RocmContext as crate::backend::SegmentedTensorBackend>::concat_token_rows(self, &[&owner_q0_partial, &owner_q1_partial])?;
            (owner_partial, peer_partial)
        } else {
            let stable_stats = |stats: &Arc<ops::hip::DeviceBuffer>| -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
                if stats.is_async_allocated() { Ok(Arc::new(stats.copy_to_stable_deferred().map_err(compute_error)?)) } else { Ok(stats.clone()) }
            };
            let owner_q0_stats = stable_stats(&owner_q0_shard.stats)?;
            let owner_q1_stats = stable_stats(&owner_q1_shard.stats)?;
            let peer_q0_stats = stable_stats(&peer_q0_shard.stats)?;
            let peer_q1_stats = stable_stats(&peer_q1_shard.stats)?;
            let owner_stats_on_peer = ops::hip::DeviceBuffer::copy_stable_group_to_device_ordered_async_retained_by(&[owner_q0_stats.clone(), owner_q1_stats.clone()], peer.device_id, self.device_id)
                .map_err(|error| compute_error(format!("L{layer} MLA owner stats->peer: {error}")))?;
            let peer_stats_on_owner = ops::hip::DeviceBuffer::copy_stable_group_to_device_ordered_async_retained_by(&[peer_q0_stats.clone(), peer_q1_stats.clone()], self.device_id, self.device_id)
                .map_err(|error| compute_error(format!("L{layer} MLA peer stats->owner: {error}")))?;
            if owner_stats_on_peer.len() != 2 || peer_stats_on_owner.len() != 2 {
                return Err(compute_error(format!("L{layer} MLA stats row shard P2P 数量异常")));
            }
            peer.activate().map_err(compute_error)?;
            let peer_q0_partial = project_local_sequence_partial(peer, &peer_q0_shard, &peer_q0_stats, &owner_stats_on_peer[0], &weights.peer_kv_b, &weights.peer_o, rows, mla, profile_pair)?;
            let peer_q1_partial = project_local_sequence_partial(peer, &peer_q1_shard, &peer_q1_stats, &owner_stats_on_peer[1], &weights.peer_kv_b, &weights.peer_o, rows, mla, profile_pair)?;
            let peer_q0_partial = peer.tensor_as_bf16(peer_q0_partial)?;
            let peer_q1_partial = peer.tensor_as_bf16(peer_q1_partial)?;
            let peer_partial = <RocmContext as crate::backend::SegmentedTensorBackend>::concat_token_rows(peer, &[&peer_q0_partial, &peer_q1_partial])?;
            let peer_partial = peer.tensor_to_stable_deferred(peer_partial)?;
            let peer_partial = peer_partial.device.as_ref().ok_or_else(|| compute_error("ROCm cooperative MLA peer partial 缺少 device buffer"))?.clone();
            if profile_pair {
                ops::hip::device_profile_scope_end(peer.device_id).map_err(compute_error)?;
            }
            ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
            let owner_q0_partial = project_local_sequence_partial(self, &owner_q0_shard, &owner_q0_stats, &peer_stats_on_owner[0], &weights.owner_kv_b, &weights.owner_o, rows, mla, profile_pair)?;
            let owner_q1_partial = project_local_sequence_partial(self, &owner_q1_shard, &owner_q1_stats, &peer_stats_on_owner[1], &weights.owner_kv_b, &weights.owner_o, rows, mla, profile_pair)?;
            let owner_partial = <RocmContext as crate::backend::SegmentedTensorBackend>::concat_token_rows(self, &[&owner_q0_partial, &owner_q1_partial])?;
            (owner_partial, peer_partial)
        };
        if profile_pair {
            ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
            ops::hip::device_profile_scope_begin(self.device_id, "glm_pair_attn_join_wait").map_err(compute_error)?;
        }
        let peer_partial_on_owner = peer_partial.copy_stable_to_device_ordered_async_retained_by(self.device_id, self.device_id).map_err(|error| compute_error(format!("L{layer} MLA peer partial->owner: {error}")))?;

        ops::hip::activate_compute_stream(self.device_id, 0).map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_operator(self.device_id, "glm_pair_attn_join_add").map_err(compute_error)?;
        }
        let owner_partial = owner_partial.device.as_deref().ok_or_else(|| compute_error("ROCm cooperative MLA owner partial 缺少 device buffer"))?;
        let residual = f32_tensor(self, residual)?;
        let residual_device = residual.device.as_deref().ok_or_else(|| compute_error("ROCm cooperative MLA residual 缺少 device buffer"))?;
        let output = ops::hip::try_ct_cooperative_partial_join_f32(self.device_id, owner_partial, &peer_partial_on_owner, residual_device, residual.rows, residual.cols).map_err(compute_error)?;
        if profile_pair {
            ops::hip::device_profile_scope_end(self.device_id).map_err(compute_error)?;
        }
        ops::hip::order_stream_after(self.device_id, 0, owner_stream).map_err(compute_error)?;
        ops::hip::activate_compute_stream(self.device_id, owner_stream).map_err(compute_error)?;
        Ok(device_tensor_f32(output, residual.rows, residual.cols))
    }
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
        let cpu_mla_data = if ops::hip::options().prefill_attention_cpu {
            let expected = checked_elements(rows, cols, "ROCm CPU MLA KV-B")?;
            let values = match weight {
                LinearWeight::F32(values) => values.to_vec(),
                LinearWeight::F16(values) => values.iter().map(|value| value.to_f32()).collect(),
                LinearWeight::Bf16Bytes(bytes) => {
                    if bytes.len() != expected.saturating_mul(2) {
                        return Err(compute_error(format!("ROCm CPU MLA KV-B BF16 bytes={}，期望 {}", bytes.len(), expected * 2)));
                    }
                    bytes.chunks_exact(2).map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect()
                }
                LinearWeight::Quantized(matrix) => {
                    if matrix.rows() != rows || matrix.cols() != cols {
                        return Err(compute_error(format!("ROCm CPU MLA KV-B {} shape=[{},{}]，期望 [{rows},{cols}]", matrix.name(), matrix.rows(), matrix.cols())));
                    }
                    matrix.decode().map_err(compute_error)?
                }
            };
            if values.len() != expected {
                return Err(compute_error(format!("ROCm CPU MLA KV-B 元素={}，期望 {expected}", values.len())));
            }
            Some(Arc::new(values))
        } else {
            None
        };
        // W8A16 直接驻留（absorb decode 有 W8G32 向量化臂、PV 有 W8G32 快路径，
        // 均为同值逐位/近无损）；此前 decode 成 F32 dense 驻留让 absorb/PV 多读
        // 4 倍字节（58.6MB/层 → 14.7MB/层）。prefill WMMA 走 paged_weight 通用
        // W8 反量化，正确但逐元素解码（TTFT 观察项）。
        let mut prepared = self.prepare_weight(weight, rows, cols)?;
        prepared.cpu_mla_data = cpu_mla_data;
        Ok(prepared)
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
    fn dsa_select_prefill_begin(&self, state: &mut Self::DsaState, layer: usize, query: &Self::Tensor, head_weights: &Self::Tensor, spec: &crate::attention::dsa::DsaSpec) -> Result<(), BackendError> {
        if query.rows == 1 && spec.kpool == 0 && (ops::hip::options().dsa_cpu_select || ops::hip::options().mla_cpu_hot_rows != 0) {
            state.select_begin(self, layer, query, head_weights)
        } else {
            self.dsa_select_prefill(state, layer, query, head_weights, spec)
        }
    }

    fn dsa_select_prefill(&self, state: &mut Self::DsaState, layer: usize, query: &Self::Tensor, head_weights: &Self::Tensor, spec: &crate::attention::dsa::DsaSpec) -> Result<(), BackendError> {
        if spec.kpool > 0 {
            return state.select_kpool(self, layer, query, head_weights, spec.kpool);
        }
        if ops::hip::options().prefill_attention_cpu && query.rows > 1 {
            return state.select_prefill_cpu(self, layer, query, head_weights);
        }
        state.select(self, layer, query, head_weights)
    }

    fn supports_dsa_prefill_selection_reuse(&self, state: &Self::DsaState, layer: usize, position: usize, rows: usize, spec: &crate::attention::dsa::DsaSpec) -> bool {
        spec.kpool == 0 && state.can_reuse_prefill_selection(layer, position, rows)
    }

    fn reuse_dsa_prefill_selection(&self, state: &mut Self::DsaState, layer: usize, position: usize, rows: usize, spec: &crate::attention::dsa::DsaSpec) -> Result<(), BackendError> {
        if spec.kpool != 0 {
            return Err(compute_error(format!("L{layer} ROCm DSA kpool selection 不能用于 MTP 迭代复用")));
        }
        state.reuse_prefill_selection(layer, position, rows)
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
        if ops::hip::options().prefill_attention_cpu && query.rows > 1 {
            let position = cache.mla_rows(layer);
            return cpu_prefill_mla_attention(self, query, latent, k_rope, kv_b, cache, layer, position, mla, state);
        }
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
        if ops::hip::options().prefill_attention_cpu && query.rows > 1 {
            let k_rope = self.rope(k_rope, 1, mla.qk_rope_head_dim, mla.rotary_layout, position, cos, sin)?;
            return cpu_prefill_mla_attention(self, query, latent, &k_rope, kv_b, cache, layer, position, mla, state);
        }
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
