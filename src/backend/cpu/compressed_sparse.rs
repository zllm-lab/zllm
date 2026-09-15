//! 压缩稀疏注意力 CPU oracle。

use crate::{
    attention::compressed_sparse::{
        CandidateSpec, CompressedBatch, CompressedGatedSegment, CompressedKvState, CompressedSelection, CompressedSparseAttentionSpec, CompressedSparseKernel, CompressedSparsePrefillSegment, CompressionStream, GatedPoolState,
        KvCompressionSpec, SharedCompressedBatch, SharedCsaOutput, V41Compressed, normalize_rope_compressed_f32,
    },
    backend::{
        BackendError, compute_error as compute,
        cpu::{CpuContext, CpuWeight},
    },
    kernel::cpu::CpuTensor,
};

pub struct CpuCompressedKvStorage {
    state: CompressedKvState,
    compressor: GatedPoolState,
    indexer: GatedPoolState,
}

fn one_row<'a>(tensor: &'a CpuTensor, name: &str) -> Result<&'a [f32], BackendError> {
    if tensor.rows != 1 {
        return Err(compute(format!("CPU CSA decode {name} 需要单行 tensor，实际 rows={}", tensor.rows)));
    }
    Ok(&tensor.data)
}

fn slice_rows(tensor: &CpuTensor, start: usize, rows: usize) -> Result<CpuTensor, BackendError> {
    let end = start.checked_add(rows).ok_or_else(|| compute("CPU segmented tensor rows 溢出"))?;
    if rows == 0 || end > tensor.rows {
        return Err(compute(format!("CPU segmented tensor rows={start}..{end} 超过 {}", tensor.rows)));
    }
    Ok(CpuTensor { data: tensor.data[start * tensor.cols..end * tensor.cols].to_vec(), rows, cols: tensor.cols })
}

impl CompressedSparseKernel for CpuContext {
    type SharedSelection = Vec<Vec<usize>>;
    type CompressedKvStorage = CpuCompressedKvStorage;

    fn allocate_compressed_kv(&self, spec: &CompressedSparseAttentionSpec) -> Result<Self::CompressedKvStorage, BackendError> {
        spec.validate().map_err(compute)?;
        Ok(CpuCompressedKvStorage { state: CompressedKvState::new(spec.window_size).map_err(compute)?, compressor: GatedPoolState::default(), indexer: GatedPoolState::default() })
    }

    fn compressed_rmsnorm_heads(&self, input: &CpuTensor, weight: &CpuWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<CpuTensor, BackendError> {
        let data = crate::kernel::cpu::vae::rmsnorm_heads(&input.data, weight.data(), head_count, head_dim, eps).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn compressed_sparse_store_recent(&self, storage: &mut Self::CompressedKvStorage, positions: &[usize], key: &CpuTensor, value: &CpuTensor) -> Result<(), BackendError> {
        if key.rows != positions.len() || value.rows != positions.len() || key.cols == 0 || value.cols != key.cols {
            return Err(compute(format!("CPU CSA recent store positions={} key=[{},{}] value=[{},{}]", positions.len(), key.rows, key.cols, value.rows, value.cols)));
        }
        storage.state.push_recent_batch(positions, &key.data, &value.data, key.cols).map_err(compute)
    }

    fn compress_gated(
        &self,
        storage: &mut Self::CompressedKvStorage,
        stream: CompressionStream,
        positions: &[usize],
        kv: &CpuTensor,
        gate: &CpuTensor,
        position_bias: &CpuWeight,
        norm: &CpuWeight,
        compression: KvCompressionSpec,
        width: usize,
        rotary_dim: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
    ) -> Result<CompressedBatch<CpuTensor>, BackendError> {
        if kv.rows != positions.len() || gate.rows != kv.rows || gate.cols != kv.cols {
            return Err(compute(format!("CPU compressor shape 非法: positions={} kv=[{},{}] gate=[{},{}]", positions.len(), kv.rows, kv.cols, gate.rows, gate.cols)));
        }
        let state = match stream {
            CompressionStream::Attention => &mut storage.compressor,
            CompressionStream::Indexer => &mut storage.indexer,
        };
        let (visible_positions, pooled) = state.push_f32(positions, &kv.data, &gate.data, position_bias.data(), compression.ratio, width, compression.overlap).map_err(compute)?;
        let values = normalize_rope_compressed_f32(&pooled, &visible_positions, compression.ratio, width, norm.data(), eps, rotary_dim, cos, sin).map_err(compute)?;
        let rows = visible_positions.len();
        Ok(CompressedBatch { visible_positions, values: CpuTensor { data: values, rows, cols: width } })
    }

    fn compress_gated_segmented(
        &self,
        segments: &mut [CompressedGatedSegment<'_, Self>],
        stream: CompressionStream,
        kv: &CpuTensor,
        gate: &CpuTensor,
        position_bias: &CpuWeight,
        norm: &CpuWeight,
        compression: KvCompressionSpec,
        width: usize,
        rotary_dim: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
    ) -> Result<Vec<CompressedBatch<CpuTensor>>, BackendError> {
        let mut offset = 0;
        let mut outputs = Vec::with_capacity(segments.len());
        for segment in segments {
            let rows = segment.positions.len();
            outputs.push(self.compress_gated(segment.storage, stream, segment.positions, &slice_rows(kv, offset, rows)?, &slice_rows(gate, offset, rows)?, position_bias, norm, compression, width, rotary_dim, cos, sin, eps)?);
            offset += rows;
        }
        if offset != kv.rows || offset != gate.rows {
            return Err(compute(format!("CPU segmented compressor rows={offset}，kv/gate={}/{}", kv.rows, gate.rows)));
        }
        Ok(outputs)
    }

    fn compressed_sparse_decode(
        &self,
        storage: &mut Self::CompressedKvStorage,
        position: usize,
        query: &CpuTensor,
        key: &CpuTensor,
        value: &CpuTensor,
        compressed_key: Option<&CpuTensor>,
        compressed_value: Option<&CpuTensor>,
        compressed_index_key: Option<&CpuTensor>,
        index_query: Option<&CpuTensor>,
        index_head_weights: Option<&CpuTensor>,
        sink: Option<&CpuWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<CpuTensor, BackendError> {
        let query = one_row(query, "query")?;
        let key = one_row(key, "key")?;
        let value = one_row(value, "value")?;
        match (compressed_key, compressed_value, compressed_index_key) {
            (Some(key), Some(value), Some(index_key)) => storage.state.push_compressed_indexed(position, one_row(key, "compressed_key")?, one_row(value, "compressed_value")?, one_row(index_key, "compressed_index_key")?).map_err(compute)?,
            (Some(key), Some(value), None) => storage.state.push_compressed(position, one_row(key, "compressed_key")?, one_row(value, "compressed_value")?).map_err(compute)?,
            (None, None, None) => {}
            _ => return Err(compute("CPU CSA compressed KV 输入必须成组出现")),
        }
        storage.state.push_recent(position, key, value).map_err(compute)?;
        let selected = match spec.compression.map(|compression| compression.selection) {
            Some(CompressedSelection::LearnedIndexer(indexer)) => Some(
                storage
                    .state
                    .selected_indices(
                        one_row(index_query.ok_or_else(|| compute("CPU CSA 缺少 index_query"))?, "index_query")?,
                        one_row(index_head_weights.ok_or_else(|| compute("CPU CSA 缺少 index head weights"))?, "index_head_weights")?,
                        &indexer,
                    )
                    .map_err(compute)?,
            ),
            _ => None,
        };
        let sink = sink.map(|weight| weight.data());
        let data = storage.state.attend_f32(query, spec.num_heads, spec.num_kv_heads, spec.head_dim, selected.as_deref(), sink).map_err(compute)?;
        Ok(CpuTensor { rows: 1, cols: data.len(), data })
    }

    #[allow(clippy::too_many_arguments)]
    fn compressed_sparse_prefill(
        &self,
        storage: &mut Self::CompressedKvStorage,
        positions: &[usize],
        causal_batch: bool,
        query: &CpuTensor,
        key: &CpuTensor,
        value: &CpuTensor,
        compressed_positions: Option<&[usize]>,
        compressed_key: Option<&CpuTensor>,
        compressed_value: Option<&CpuTensor>,
        compressed_index_key: Option<&CpuTensor>,
        index_query: Option<&CpuTensor>,
        index_head_weights: Option<&CpuTensor>,
        sink: Option<&CpuWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<CpuTensor, BackendError> {
        let rows = query.rows;
        if rows == 0 || query.cols != spec.num_heads * spec.head_dim || key.rows != rows || value.rows != rows || key.cols != spec.num_kv_heads * spec.head_dim || value.cols != key.cols {
            return Err(compute(format!("CPU CSA prefill 维度非法: rows={rows} query=[{},{}] key=[{},{}] heads={}/{} head_dim={}", query.rows, query.cols, key.rows, key.cols, spec.num_heads, spec.num_kv_heads, spec.head_dim,)));
        }
        // 合并历史 compressed + 本批 compressed，作为 attention 的压缩历史来源。
        let history_compressed_len = storage.state.compressed_len();
        let kv_width = spec.num_kv_heads * spec.head_dim;
        let mut all_compressed_positions: Vec<usize> = (0..history_compressed_len).map(|index| storage.state.compressed_position(index)).collect();
        let mut all_compressed_keys: Vec<f32> = (0..history_compressed_len).flat_map(|index| storage.state.compressed_key(index)).copied().collect();
        let mut all_compressed_values: Vec<f32> = (0..history_compressed_len).flat_map(|index| storage.state.compressed_value(index)).copied().collect();
        let learned_indexer = matches!(spec.compression.map(|compression| compression.selection), Some(CompressedSelection::LearnedIndexer(_)));
        let mut all_compressed_index_keys = learned_indexer.then(Vec::new);
        if let Some(index_keys) = &mut all_compressed_index_keys {
            for index in 0..history_compressed_len {
                let key = storage.state.compressed_index_key(index).ok_or_else(|| compute(format!("CPU CSA 历史 compressed row {index} 缺少 index key")))?;
                index_keys.extend_from_slice(key);
            }
        }
        let mut batch_compressed_rows = 0;
        if let (Some(cpositions), Some(ckey), Some(cvalue)) = (compressed_positions, compressed_key, compressed_value) {
            if ckey.rows != cpositions.len() || cvalue.rows != cpositions.len() || ckey.cols != kv_width || cvalue.cols != kv_width {
                return Err(compute(format!("CPU CSA prefill compressed 维度非法: positions={} key=[{},{}] value=[{},{}]", cpositions.len(), ckey.rows, ckey.cols, cvalue.rows, cvalue.cols,)));
            }
            batch_compressed_rows = cpositions.len();
            all_compressed_positions.extend_from_slice(cpositions);
            all_compressed_keys.extend_from_slice(&ckey.data);
            all_compressed_values.extend_from_slice(&cvalue.data);
            if let Some(index_key) = compressed_index_key {
                if index_key.rows != cpositions.len() {
                    return Err(compute(format!("CPU CSA prefill index_key 行数={} 与 compressed={} 不一致", index_key.rows, cpositions.len(),)));
                }
                let index_keys = all_compressed_index_keys.get_or_insert_with(Vec::new);
                index_keys.extend_from_slice(&index_key.data);
            }
        }
        // indexer 选择
        let indexer = match spec.compression.map(|compression| compression.selection) {
            Some(CompressedSelection::LearnedIndexer(indexer)) => {
                if index_query.is_none() || index_query.unwrap().rows != rows || index_head_weights.is_none() || index_head_weights.unwrap().rows != rows {
                    return Err(compute("CPU CSA prefill c4a 层缺少 index query/head weights 或行数不匹配"));
                }
                Some(indexer)
            }
            _ => None,
        };
        let sink = sink.map(|weight| weight.data());
        let data = storage
            .state
            .attend_batch_f32(
                &query.data,
                rows,
                positions,
                causal_batch,
                &key.data,
                &value.data,
                &all_compressed_positions,
                &all_compressed_keys,
                &all_compressed_values,
                all_compressed_index_keys.as_deref(),
                index_query.map(|t| t.data.as_slice()),
                index_head_weights.map(|t| t.data.as_slice()),
                spec.num_heads,
                spec.num_kv_heads,
                spec.head_dim,
                indexer,
                sink,
            )
            .map_err(compute)?;
        // 更新 cache：本批 compressed 追加到压缩历史，本批 recent 滑窗只保留最后 window_size 个
        if batch_compressed_rows > 0 {
            let cpositions = compressed_positions.unwrap();
            let ckey = compressed_key.unwrap();
            let cvalue = compressed_value.unwrap();
            if let Some(index_key) = compressed_index_key {
                storage.state.push_compressed_indexed_batch(cpositions, &ckey.data, &cvalue.data, &index_key.data, kv_width).map_err(compute)?;
            } else {
                storage.state.push_compressed_batch(cpositions, &ckey.data, &cvalue.data, kv_width).map_err(compute)?;
            }
        }
        storage.state.push_recent_batch(positions, &key.data, &value.data, kv_width).map_err(compute)?;
        let cols = spec.num_heads * spec.head_dim;
        Ok(CpuTensor { rows, cols, data })
    }

    fn compressed_sparse_prefill_segmented(
        &self,
        query: &CpuTensor,
        key: &CpuTensor,
        value: &CpuTensor,
        index_query: Option<&CpuTensor>,
        index_head_weights: Option<&CpuTensor>,
        segments: &mut [CompressedSparsePrefillSegment<'_, Self>],
        sink: Option<&CpuWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<CpuTensor, BackendError> {
        let mut offset = 0;
        let mut data = Vec::new();
        for segment in segments {
            let rows = segment.positions.len();
            let output = self.compressed_sparse_prefill(
                segment.storage,
                segment.positions,
                segment.causal_batch,
                &slice_rows(query, offset, rows)?,
                &slice_rows(key, offset, rows)?,
                &slice_rows(value, offset, rows)?,
                segment.compressed_positions,
                segment.compressed_key,
                segment.compressed_value,
                segment.compressed_index_key,
                index_query.map(|tensor| slice_rows(tensor, offset, rows)).transpose()?.as_ref(),
                index_head_weights.map(|tensor| slice_rows(tensor, offset, rows)).transpose()?.as_ref(),
                sink,
                spec,
            )?;
            data.extend(output.data);
            offset += rows;
        }
        if offset != query.rows || offset != key.rows || offset != value.rows {
            return Err(compute(format!("CPU segmented CSA rows={offset}，Q/K/V={}/{}/{}", query.rows, key.rows, value.rows)));
        }
        Ok(CpuTensor { data, rows: offset, cols: spec.num_heads * spec.head_dim })
    }

    fn compress_v41(
        &self,
        storage: &mut CpuCompressedKvStorage,
        positions: &[usize],
        kv: &CpuTensor,
        gate: Option<&CpuTensor>,
        norm: &CpuWeight,
        index_key_projection: Option<(&CpuWeight, &CpuWeight)>,
        compression: KvCompressionSpec,
        width: usize,
        rotary_dim: usize,
        index_rope_dim: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
    ) -> Result<V41Compressed<CpuTensor>, BackendError> {
        let ratio = compression.ratio;
        // ratio=1 官方无 gate:池化退化为逐行直通,不经过 pending 状态。
        let (visible_positions, pooled) = if ratio == 1 {
            if kv.rows != positions.len() || kv.cols != width {
                return Err(compute(format!("CPU V4.1 compressor(ratio=1) shape 非法: positions={} kv=[{},{}] width={width}", positions.len(), kv.rows, kv.cols)));
            }
            (positions.to_vec(), kv.data.clone())
        } else {
            let gate = gate.ok_or_else(|| compute("CPU V4.1 compressor ratio>1 缺少 gate"))?;
            if kv.rows != positions.len() || gate.rows != kv.rows || gate.cols != kv.cols || kv.cols != width {
                return Err(compute(format!("CPU V4.1 compressor shape 非法: positions={} kv=[{},{}] gate=[{},{}] width={width}", positions.len(), kv.rows, kv.cols, gate.rows, gate.cols)));
            }
            // V4.1 无 ape:位置偏置恒为 0。
            let bias = vec![0.0; ratio * width];
            storage.compressor.push_f32(positions, &kv.data, &gate.data, &bias, ratio, width, compression.overlap).map_err(compute)?
        };
        let attention_values = normalize_rope_compressed_f32(&pooled, &visible_positions, ratio, width, norm.data(), eps, rotary_dim, cos, sin).map_err(compute)?;
        let rows = visible_positions.len();
        let attention = CompressedBatch { visible_positions: visible_positions.clone(), values: CpuTensor { data: attention_values, rows, cols: width } };
        let index_key = index_key_projection
            .map(|(wk, k_norm)| {
                let (index_dim, data) = crate::attention::compressed_sparse::index_key_compressed_f32(&pooled, &visible_positions, ratio, width, wk.data(), k_norm.data(), eps, index_rope_dim, cos, sin).map_err(compute)?;
                Ok::<_, BackendError>(CompressedBatch { visible_positions: visible_positions.clone(), values: CpuTensor { data, rows, cols: index_dim } })
            })
            .transpose()?;
        Ok(V41Compressed { attention, index_key })
    }

    fn compressed_sparse_prefill_shared(
        &self,
        recent: &mut CpuCompressedKvStorage,
        shared: Option<&CpuCompressedKvStorage>,
        positions: &[usize],
        causal_batch: bool,
        query: &CpuTensor,
        key: &CpuTensor,
        value: &CpuTensor,
        batch: Option<SharedCompressedBatch<'_, CpuTensor>>,
        index_query: Option<&CpuTensor>,
        index_head_weights: Option<&CpuTensor>,
        preset_selection: Option<&Self::SharedSelection>,
        candidate: Option<CandidateSpec>,
        sink: Option<&CpuWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<SharedCsaOutput<CpuTensor, Self::SharedSelection>, BackendError> {
        let rows = query.rows;
        if rows == 0 || query.cols != spec.num_heads * spec.head_dim || key.rows != rows || value.rows != rows || key.cols != spec.num_kv_heads * spec.head_dim || value.cols != key.cols {
            return Err(compute(format!("CPU V4.1 CSA prefill 维度非法: rows={rows} query=[{},{}] key=[{},{}] heads={}/{} head_dim={}", query.rows, query.cols, key.rows, key.cols, spec.num_heads, spec.num_kv_heads, spec.head_dim)));
        }
        let indexer = match spec.compression.map(|compression| compression.selection) {
            Some(CompressedSelection::LearnedIndexer(dsa)) => Some(dsa),
            _ => None,
        };
        // 压缩历史权威:非源层读 shared,源层(shared=None)的 history 就在自己的 storage。
        let (all_positions, all_keys, all_values, all_index_keys) = shared_compressed_parts(shared.unwrap_or(recent), batch.as_ref(), spec, indexer.is_some())?;
        // selection:preset 复用;index 层在可见行上自算并发布(候选两阶段可选)。
        let mut published = None;
        let preset = if indexer.is_some() && preset_selection.is_some() {
            Some(preset_selection.expect("上方已校验").clone())
        } else if let Some(dsa) = indexer {
            let index_query = index_query.ok_or_else(|| compute("CPU V4.1 index 层缺少 index_query"))?;
            let index_head_weights = index_head_weights.ok_or_else(|| compute("CPU V4.1 index 层缺少 index head weights"))?;
            let index_keys = all_index_keys.as_deref().ok_or_else(|| compute("CPU V4.1 压缩历史缺少 index key"))?;
            let mut per_row = Vec::with_capacity(rows);
            for row in 0..rows {
                let query_position = if causal_batch { positions[row] } else { *positions.last().ok_or_else(|| compute("CPU V4.1 CSA 空 positions"))? };
                let visible: Vec<usize> = all_positions.iter().enumerate().filter(|entry| *entry.1 <= query_position).map(|(index, _)| index).collect();
                let query_width = dsa.num_heads * dsa.head_dim;
                let iq = &index_query.data[row * query_width..(row + 1) * query_width];
                let weights = &index_head_weights.data[row * dsa.num_heads..(row + 1) * dsa.num_heads];
                let mut visible_keys = Vec::with_capacity(visible.len() * dsa.head_dim);
                for &index in &visible {
                    visible_keys.extend_from_slice(&index_keys[index * dsa.head_dim..(index + 1) * dsa.head_dim]);
                }
                let local = match candidate {
                    Some(candidate) => crate::attention::compressed_sparse::topk_indexer_candidate_f32(iq, &visible_keys, weights, &dsa, dsa.top_k, candidate).map_err(compute)?,
                    None => crate::attention::compressed_sparse::topk_indexer_f32(iq, &visible_keys, weights, &dsa, dsa.top_k).map_err(compute)?,
                };
                per_row.push(local.into_iter().map(|local| visible[local]).collect());
            }
            published = Some(per_row.clone());
            Some(per_row)
        } else {
            None
        };
        let sink_data = sink.map(|weight| weight.data());
        let data = recent
            .state
            .attend_batch_preset_f32(
                &query.data,
                rows,
                positions,
                causal_batch,
                &key.data,
                &value.data,
                &all_positions,
                &all_keys,
                &all_values,
                all_index_keys.as_deref(),
                index_query.map(|tensor| tensor.data.as_slice()),
                index_head_weights.map(|tensor| tensor.data.as_slice()),
                spec.num_heads,
                spec.num_kv_heads,
                spec.head_dim,
                indexer,
                preset.as_deref(),
                sink_data,
            )
            .map_err(compute)?;
        // 写入:滑窗永远写本层;压缩历史只写源层(shared=None 时 recent 即源)。
        if shared.is_none()
            && let Some(batch) = &batch
        {
            let kv_width = spec.num_kv_heads * spec.head_dim;
            match batch.index_key {
                Some(index_key) => recent.state.push_compressed_indexed_batch(batch.visible_positions, &batch.key.data, &batch.value.data, &index_key.data, kv_width).map_err(compute)?,
                None => recent.state.push_compressed_batch(batch.visible_positions, &batch.key.data, &batch.value.data, kv_width).map_err(compute)?,
            }
        }
        let kv_width = spec.num_kv_heads * spec.head_dim;
        recent.state.push_recent_batch(positions, &key.data, &value.data, kv_width).map_err(compute)?;
        Ok(SharedCsaOutput { attended: CpuTensor { rows, cols: spec.num_heads * spec.head_dim, data }, selection: published })
    }

    fn compressed_sparse_decode_shared(
        &self,
        recent: &mut CpuCompressedKvStorage,
        shared: Option<&CpuCompressedKvStorage>,
        position: usize,
        query: &CpuTensor,
        key: &CpuTensor,
        value: &CpuTensor,
        batch: Option<SharedCompressedBatch<'_, CpuTensor>>,
        index_query: Option<&CpuTensor>,
        index_head_weights: Option<&CpuTensor>,
        preset_selection: Option<&Self::SharedSelection>,
        candidate: Option<CandidateSpec>,
        sink: Option<&CpuWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<SharedCsaOutput<CpuTensor, Self::SharedSelection>, BackendError> {
        let indexer = match spec.compression.map(|compression| compression.selection) {
            Some(CompressedSelection::LearnedIndexer(dsa)) => Some(dsa),
            _ => None,
        };
        let (all_positions, all_keys, all_values, all_index_keys) = shared_compressed_parts(shared.unwrap_or(recent), batch.as_ref(), spec, indexer.is_some())?;
        let mut published = None;
        let preset = if indexer.is_some() && preset_selection.is_some() {
            Some(preset_selection.expect("上方已校验").clone())
        } else if let Some(dsa) = indexer {
            let index_query = one_row(index_query.ok_or_else(|| compute("CPU V4.1 index 层缺少 index_query"))?, "index_query")?;
            let weights = one_row(index_head_weights.ok_or_else(|| compute("CPU V4.1 index 层缺少 index head weights"))?, "index_head_weights")?;
            let index_keys = all_index_keys.as_deref().ok_or_else(|| compute("CPU V4.1 压缩历史缺少 index key"))?;
            let visible: Vec<usize> = all_positions.iter().enumerate().filter(|entry| *entry.1 <= position).map(|(index, _)| index).collect();
            let mut visible_keys = Vec::with_capacity(visible.len() * dsa.head_dim);
            for &index in &visible {
                visible_keys.extend_from_slice(&index_keys[index * dsa.head_dim..(index + 1) * dsa.head_dim]);
            }
            let local = match candidate {
                Some(candidate) => crate::attention::compressed_sparse::topk_indexer_candidate_f32(index_query, &visible_keys, weights, &dsa, dsa.top_k, candidate).map_err(compute)?,
                None => crate::attention::compressed_sparse::topk_indexer_f32(index_query, &visible_keys, weights, &dsa, dsa.top_k).map_err(compute)?,
            };
            let row: Vec<usize> = local.into_iter().map(|local| visible[local]).collect();
            published = Some(vec![row.clone()]);
            Some(vec![row])
        } else {
            None
        };
        let query_row = one_row(query, "query")?;
        let key_row = one_row(key, "key")?;
        let value_row = one_row(value, "value")?;
        let sink_data = sink.map(|weight| weight.data());
        let data = recent
            .state
            .attend_batch_preset_f32(
                query_row,
                1,
                &[position],
                true,
                key_row,
                value_row,
                &all_positions,
                &all_keys,
                &all_values,
                all_index_keys.as_deref(),
                index_query.map(|tensor| &tensor.data[..]),
                index_head_weights.map(|tensor| &tensor.data[..]),
                spec.num_heads,
                spec.num_kv_heads,
                spec.head_dim,
                indexer,
                preset.as_deref(),
                sink_data,
            )
            .map_err(compute)?;
        if shared.is_none()
            && let Some(batch) = &batch
            && batch.visible_positions.len() == 1
        {
            let kv_width = spec.num_kv_heads * spec.head_dim;
            let key_row = one_row(&batch.key, "shared compressed key")?;
            let value_row = one_row(&batch.value, "shared compressed value")?;
            match batch.index_key {
                Some(index_key) => recent.state.push_compressed_indexed(batch.visible_positions[0], key_row, value_row, one_row(index_key, "shared compressed index key")?).map_err(compute)?,
                None => recent.state.push_compressed(batch.visible_positions[0], key_row, value_row).map_err(compute)?,
            }
        }
        recent.state.push_recent(position, key_row, value_row).map_err(compute)?;
        Ok(SharedCsaOutput { attended: CpuTensor { rows: 1, cols: data.len(), data }, selection: published })
    }
}

/// 合并压缩历史(源层 storage)与本批广播压缩项,返回行优先的 positions/keys/values
/// 与(需要时)index keys。
fn shared_compressed_parts(
    source: &CpuCompressedKvStorage,
    batch: Option<&SharedCompressedBatch<'_, CpuTensor>>,
    spec: &CompressedSparseAttentionSpec,
    need_index: bool,
) -> Result<(Vec<usize>, Vec<f32>, Vec<f32>, Option<Vec<f32>>), BackendError> {
    let history = &source.state;
    let kv_width = spec.num_kv_heads * spec.head_dim;
    let mut positions = Vec::with_capacity(history.compressed_len());
    let mut keys = Vec::with_capacity(history.compressed_len() * kv_width);
    let mut values = Vec::with_capacity(history.compressed_len() * kv_width);
    let mut index_keys = need_index.then(Vec::new);
    for row in 0..history.compressed_len() {
        positions.push(history.compressed_position(row));
        keys.extend_from_slice(history.compressed_key(row));
        values.extend_from_slice(history.compressed_value(row));
        if let Some(index_keys) = index_keys.as_mut() {
            let key = history.compressed_index_key(row).ok_or_else(|| compute(format!("CPU V4.1 压缩历史 row {row} 缺少 index key")))?;
            index_keys.extend_from_slice(key);
        }
    }
    if let Some(batch) = batch {
        if batch.key.rows != batch.visible_positions.len() || batch.value.rows != batch.visible_positions.len() || batch.key.cols != kv_width || batch.value.cols != kv_width {
            return Err(compute(format!("CPU V4.1 共享压缩批维度非法: positions={} key=[{},{}] value=[{},{}] kv_width={kv_width}", batch.visible_positions.len(), batch.key.rows, batch.key.cols, batch.value.rows, batch.value.cols)));
        }
        positions.extend_from_slice(batch.visible_positions);
        keys.extend_from_slice(&batch.key.data);
        values.extend_from_slice(&batch.value.data);
        if let Some(batch_index_key) = batch.index_key {
            if batch_index_key.rows != batch.visible_positions.len() {
                return Err(compute(format!("CPU V4.1 共享压缩批 index_key 行数={} 与 positions={} 不一致", batch_index_key.rows, batch.visible_positions.len())));
            }
            index_keys.get_or_insert_with(Vec::new).extend_from_slice(&batch_index_key.data);
        }
    }
    Ok((positions, keys, values, index_keys))
}
