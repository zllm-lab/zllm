//! 压缩稀疏注意力的 Metal resident capability。

use std::mem;

use half::f16;

use crate::{
    attention::compressed_sparse::{
        CompressedBatch, CompressedGatedSegment, CompressedSelection, CompressedSparseAttentionSpec, CompressedSparseKernel, CompressedSparsePrefillSegment, CompressionState, CompressionStream, KvCompressionSpec,
        compress_gated_segmented_fallback, compressed_sparse_prefill_segmented_fallback,
    },
    backend::{
        BackendError, compute_error as compute,
        metal::{MetalContext, MetalTensor, MetalTensorDType, MetalWeight, api::Buffer, expect_resident_f16},
    },
    kernel::metal as ops,
};

#[derive(Default)]
struct MetalGatedPoolState {
    state: CompressionState,
    ratio: usize,
    width: usize,
    channels: usize,
    overlap: bool,
    pending_key: Option<Buffer>,
    pending_gate: Option<Buffer>,
    overlap_key: Option<Buffer>,
    overlap_gate: Option<Buffer>,
}

impl MetalGatedPoolState {
    fn buffers(&mut self, ctx: &MetalContext, ratio: usize, width: usize, overlap: bool) -> Result<(&Buffer, &Buffer, &Buffer, &Buffer), BackendError> {
        let channels = if overlap { width.checked_mul(2).ok_or_else(|| compute("V4 compressor channels 溢出"))? } else { width };
        if self.pending_key.is_none() {
            self.ratio = ratio;
            self.width = width;
            self.channels = channels;
            self.overlap = overlap;
            let pending_bytes = ratio.checked_mul(channels).and_then(|n| n.checked_mul(mem::size_of::<f16>())).ok_or_else(|| compute("V4 compressor pending buffer 溢出"))?;
            self.pending_key = Some(ctx.shared_buffer_zeros(pending_bytes));
            self.pending_gate = Some(ctx.shared_buffer_zeros(pending_bytes));
            if overlap {
                let overlap_elements = ratio.checked_mul(width).ok_or_else(|| compute("V4 compressor overlap buffer 溢出"))?;
                self.overlap_key = Some(ctx.shared_buffer_zeros(overlap_elements * mem::size_of::<f16>()));
                self.overlap_gate = Some(ctx.shared_buffer_from_f32(&vec![f32::NEG_INFINITY; overlap_elements]));
            } else {
                self.overlap_key = Some(ctx.shared_buffer_zeros(mem::size_of::<f16>()));
                self.overlap_gate = Some(ctx.shared_buffer_zeros(mem::size_of::<f16>()));
            }
        } else if (self.ratio, self.width, self.channels, self.overlap) != (ratio, width, channels, overlap) {
            return Err(compute(format!("V4 compressor 状态规格改变: 原={}/{}/{}/{} 新={ratio}/{width}/{channels}/{overlap}", self.ratio, self.width, self.channels, self.overlap)));
        }
        Ok((
            self.pending_key.as_ref().expect("pending key 已初始化"),
            self.pending_gate.as_ref().expect("pending gate 已初始化"),
            self.overlap_key.as_ref().expect("overlap key 已初始化"),
            self.overlap_gate.as_ref().expect("overlap gate 已初始化"),
        ))
    }
}

pub struct MetalCompressedKvStorage {
    window_size: usize,
    kv_width: usize,
    recent_key: Buffer,
    recent_value: Buffer,
    recent_start: usize,
    recent_len: usize,
    recent_first_position: usize,
    next_recent_position: Option<usize>,
    compressed_key: Buffer,
    compressed_value: Buffer,
    compressed_index_key: Option<Buffer>,
    compressed_index_width: usize,
    compressed_positions: Vec<usize>,
    compressed_capacity: usize,
    compressor: MetalGatedPoolState,
    indexer: MetalGatedPoolState,
}

fn bytes<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), mem::size_of_val(values)) }
}

fn constant(weight: &MetalWeight, expected: usize, name: &str) -> Result<(Buffer, bool), BackendError> {
    match weight {
        MetalWeight::F16(tensor) if tensor.len() == expected && tensor.dtype == MetalTensorDType::F16 => Ok((tensor.buffer.clone(), false)),
        MetalWeight::F32 { buffer, len } if *len == expected => Ok((buffer.clone(), true)),
        MetalWeight::F16(tensor) => Err(compute(format!("V4 {name} F16 长度/dtype 非法: len={} dtype={:?} expected={expected}", tensor.len(), tensor.dtype))),
        MetalWeight::F32 { len, .. } => Err(compute(format!("V4 {name} F32 长度={len}，期望 {expected}"))),
        _ => Err(compute(format!("V4 {name} 必须是 F16/F32 resident 常量"))),
    }
}

impl MetalCompressedKvStorage {
    fn append_compressed(&mut self, ctx: &MetalContext, positions: &[usize], key: &MetalTensor, value: &MetalTensor, index_key: Option<&MetalTensor>) -> Result<(), BackendError> {
        if positions.is_empty() {
            return Ok(());
        }
        if key.rows != positions.len()
            || value.rows != positions.len()
            || key.cols != self.kv_width
            || value.cols != self.kv_width
            || positions.windows(2).any(|pair| pair[1] <= pair[0])
            || self.compressed_positions.last().is_some_and(|last| positions[0] <= *last)
        {
            return Err(compute(format!("V4 compressed append 非法: positions={positions:?} key=[{},{}] value=[{},{}] width={} history={:?}", key.rows, key.cols, value.rows, value.cols, self.kv_width, self.compressed_positions.last())));
        }
        let key = ops::to_f16_tensor(ctx, key).map_err(compute)?;
        let value = ops::to_f16_tensor(ctx, value).map_err(compute)?;
        let index_key = index_key.map(|tensor| ops::to_f16_tensor(ctx, tensor)).transpose().map_err(compute)?;
        if let Some(index) = &index_key {
            if index.rows != positions.len() || index.cols == 0 {
                return Err(compute(format!("V4 compressed index key=[{},{}] 与 rows={} 不符", index.rows, index.cols, positions.len())));
            }
            if self.compressed_index_width != 0 && self.compressed_index_width != index.cols {
                return Err(compute(format!("V4 compressed index width 从 {} 变为 {}", self.compressed_index_width, index.cols)));
            }
            if self.compressed_index_key.is_none() && !self.compressed_positions.is_empty() {
                return Err(compute("V4 compressed 历史缺少 index key"));
            }
        } else if self.compressed_index_key.is_some() {
            return Err(compute("V4 compressed append 缺少 index key"));
        }

        let required = self.compressed_positions.len().checked_add(positions.len()).ok_or_else(|| compute("V4 compressed rows 溢出"))?;
        let growing = required > self.compressed_capacity;
        let new_capacity = if growing { required.next_power_of_two() } else { self.compressed_capacity };
        let kv_bytes = new_capacity.checked_mul(self.kv_width).and_then(|n| n.checked_mul(mem::size_of::<f16>())).ok_or_else(|| compute("V4 compressed KV buffer 溢出"))?;
        let new_key = if growing { ctx.shared_buffer_uninit(kv_bytes) } else { self.compressed_key.clone() };
        let new_value = if growing { ctx.shared_buffer_uninit(kv_bytes) } else { self.compressed_value.clone() };
        let index_width = index_key.as_ref().map_or(self.compressed_index_width, |tensor| tensor.cols);
        let new_index = if index_width == 0 {
            None
        } else if growing || self.compressed_index_key.is_none() {
            Some(ctx.shared_buffer_uninit(new_capacity * index_width * mem::size_of::<f16>()))
        } else {
            self.compressed_index_key.clone()
        };

        let command = ctx.command_buffer();
        let blit = command.new_blit_command_encoder();
        let old_rows = self.compressed_positions.len();
        if growing && old_rows != 0 {
            let old_kv_bytes = (old_rows * self.kv_width * mem::size_of::<f16>()) as u64;
            blit.copy_from_buffer(&self.compressed_key, 0, &new_key, 0, old_kv_bytes);
            blit.copy_from_buffer(&self.compressed_value, 0, &new_value, 0, old_kv_bytes);
            if let (Some(old), Some(new)) = (&self.compressed_index_key, &new_index) {
                blit.copy_from_buffer(old, 0, new, 0, (old_rows * index_width * mem::size_of::<f16>()) as u64);
            }
        }
        let destination_offset = (old_rows * self.kv_width * mem::size_of::<f16>()) as u64;
        blit.copy_from_buffer(&key.buffer, 0, &new_key, destination_offset, key.buffer.length());
        blit.copy_from_buffer(&value.buffer, 0, &new_value, destination_offset, value.buffer.length());
        if let (Some(index), Some(destination)) = (&index_key, &new_index) {
            blit.copy_from_buffer(&index.buffer, 0, destination, (old_rows * index_width * mem::size_of::<f16>()) as u64, index.buffer.length());
        }
        blit.end_encoding();
        let shape = format!("rows={old_rows}+{} width={} index_width={index_width}", positions.len(), self.kv_width);
        ctx.commit_and_wait_profiled(&command, "csa_append_compressed", &shape, key.buffer.length() + value.buffer.length(), (key.buffer.length() + value.buffer.length()) * u64::from(growing) + key.buffer.length() + value.buffer.length());
        self.compressed_key = new_key;
        self.compressed_value = new_value;
        self.compressed_index_key = new_index;
        self.compressed_index_width = index_width;
        self.compressed_capacity = new_capacity;
        self.compressed_positions.extend_from_slice(positions);
        Ok(())
    }

    fn validate_recent_positions(&self, positions: &[usize]) -> Result<(), BackendError> {
        if positions.is_empty() || positions.windows(2).any(|pair| pair[1] != pair[0] + 1) || self.next_recent_position.is_some_and(|next| positions[0] != next) {
            return Err(compute(format!("V4 recent positions 非连续: positions={positions:?} next={:?}", self.next_recent_position)));
        }
        Ok(())
    }

    fn append_recent(&mut self, ctx: &MetalContext, positions: &[usize], key: &MetalTensor, value: &MetalTensor) -> Result<(), BackendError> {
        self.validate_recent_positions(positions)?;
        if key.rows != positions.len() || value.rows != positions.len() || key.cols != self.kv_width || value.cols != self.kv_width {
            return Err(compute(format!("V4 recent append 维度非法: positions={} key=[{},{}] value=[{},{}] width={}", positions.len(), key.rows, key.cols, value.rows, value.cols, self.kv_width)));
        }
        let row_bytes = self.kv_width * mem::size_of::<f16>();
        let command = ctx.command_buffer();
        let blit = command.new_blit_command_encoder();
        if positions.len() >= self.window_size {
            let source_row = positions.len() - self.window_size;
            let source_offset = (source_row * row_bytes) as u64;
            let copy_bytes = (self.window_size * row_bytes) as u64;
            blit.copy_from_buffer(&key.buffer, source_offset, &self.recent_key, 0, copy_bytes);
            blit.copy_from_buffer(&value.buffer, source_offset, &self.recent_value, 0, copy_bytes);
            self.recent_start = 0;
            self.recent_len = self.window_size;
            self.recent_first_position = positions[source_row];
        } else {
            let tail = (self.recent_start + self.recent_len) % self.window_size;
            let first_rows = positions.len().min(self.window_size - tail);
            let first_bytes = (first_rows * row_bytes) as u64;
            blit.copy_from_buffer(&key.buffer, 0, &self.recent_key, (tail * row_bytes) as u64, first_bytes);
            blit.copy_from_buffer(&value.buffer, 0, &self.recent_value, (tail * row_bytes) as u64, first_bytes);
            if first_rows < positions.len() {
                let second_bytes = ((positions.len() - first_rows) * row_bytes) as u64;
                blit.copy_from_buffer(&key.buffer, first_bytes, &self.recent_key, 0, second_bytes);
                blit.copy_from_buffer(&value.buffer, first_bytes, &self.recent_value, 0, second_bytes);
            }
            let overflow = (self.recent_len + positions.len()).saturating_sub(self.window_size);
            if self.recent_len == 0 {
                self.recent_first_position = positions[0];
            } else {
                self.recent_first_position += overflow;
            }
            self.recent_start = (self.recent_start + overflow) % self.window_size;
            self.recent_len = (self.recent_len + positions.len()).min(self.window_size);
        }
        blit.end_encoding();
        let shape = format!("rows={} width={} ring={}/{}", positions.len(), self.kv_width, self.recent_start, self.recent_len);
        ctx.commit_and_wait_profiled(&command, "csa_append_recent", &shape, key.buffer.length() + value.buffer.length(), (positions.len().min(self.window_size) * row_bytes * 2) as u64);
        self.next_recent_position = positions.last().map(|position| position + 1);
        Ok(())
    }

    fn visible_counts(&self, positions: &[usize]) -> Result<Vec<u32>, BackendError> {
        positions
            .iter()
            .map(|position| {
                let count = self.compressed_positions.partition_point(|compressed| compressed <= position);
                u32::try_from(count).map_err(|_| compute(format!("V4 compressed visible rows={count} 超过 u32")))
            })
            .collect()
    }
}

impl CompressedSparseKernel for MetalContext {
    type CompressedKvStorage = MetalCompressedKvStorage;

    fn allocate_compressed_kv(&self, spec: &CompressedSparseAttentionSpec) -> Result<Self::CompressedKvStorage, BackendError> {
        spec.validate().map_err(compute)?;
        let kv_width = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute("V4 Metal KV width 溢出"))?;
        let recent_bytes = spec.window_size.checked_mul(kv_width).and_then(|n| n.checked_mul(mem::size_of::<f16>())).ok_or_else(|| compute("V4 Metal recent KV buffer 溢出"))?;
        let compressed_capacity = spec.window_size;
        let compressed_bytes = compressed_capacity.checked_mul(kv_width).and_then(|n| n.checked_mul(mem::size_of::<f16>())).ok_or_else(|| compute("V4 Metal compressed KV buffer 溢出"))?;
        Ok(MetalCompressedKvStorage {
            window_size: spec.window_size,
            kv_width,
            recent_key: self.shared_buffer_zeros(recent_bytes),
            recent_value: self.shared_buffer_zeros(recent_bytes),
            recent_start: 0,
            recent_len: 0,
            recent_first_position: 0,
            next_recent_position: None,
            compressed_key: self.shared_buffer_zeros(compressed_bytes),
            compressed_value: self.shared_buffer_zeros(compressed_bytes),
            compressed_index_key: None,
            compressed_index_width: 0,
            compressed_positions: Vec::new(),
            compressed_capacity,
            compressor: MetalGatedPoolState::default(),
            indexer: MetalGatedPoolState::default(),
        })
    }

    fn compressed_rmsnorm_heads(&self, input: &MetalTensor, weight: &MetalWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<MetalTensor, BackendError> {
        if input.cols != head_count.checked_mul(head_dim).ok_or_else(|| compute("Metal V4 head RMSNorm columns 溢出"))? {
            return Err(compute(format!("Metal V4 head RMSNorm input=[{},{}] heads={head_count} dim={head_dim}", input.rows, input.cols)));
        }
        let weight = expect_resident_f16(weight, "Metal V4 head RMSNorm 需要 resident F16 权重")?;
        let heads = input.reshape(input.rows * head_count, head_dim);
        let output = ops::tensor::rmsnorm_tensor_resident_weight(self, &heads, weight, eps, 0.0).map_err(compute)?;
        Ok(output.reshape(input.rows, input.cols))
    }

    fn compress_gated_segmented(
        &self,
        segments: &mut [CompressedGatedSegment<'_, Self>],
        stream: CompressionStream,
        kv: &MetalTensor,
        gate: &MetalTensor,
        position_bias: &MetalWeight,
        norm: &MetalWeight,
        compression: KvCompressionSpec,
        width: usize,
        rotary_dim: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
    ) -> Result<Vec<CompressedBatch<MetalTensor>>, BackendError> {
        compress_gated_segmented_fallback(self, segments, stream, kv, gate, position_bias, norm, compression, width, rotary_dim, cos, sin, eps)
    }

    fn compressed_sparse_prefill_segmented(
        &self,
        query: &MetalTensor,
        key: &MetalTensor,
        value: &MetalTensor,
        index_query: Option<&MetalTensor>,
        index_head_weights: Option<&MetalTensor>,
        segments: &mut [CompressedSparsePrefillSegment<'_, Self>],
        sink: Option<&MetalWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<MetalTensor, BackendError> {
        compressed_sparse_prefill_segmented_fallback(self, query, key, value, index_query, index_head_weights, segments, sink, spec)
    }

    fn compressed_sparse_store_recent(&self, storage: &mut Self::CompressedKvStorage, positions: &[usize], key: &MetalTensor, value: &MetalTensor) -> Result<(), BackendError> {
        let key = ops::to_f16_tensor(self, key).map_err(compute)?;
        let value = ops::to_f16_tensor(self, value).map_err(compute)?;
        storage.append_recent(self, positions, &key, &value)
    }

    fn compress_gated(
        &self,
        storage: &mut Self::CompressedKvStorage,
        stream: CompressionStream,
        positions: &[usize],
        key: &MetalTensor,
        gate: &MetalTensor,
        position_bias: &MetalWeight,
        norm: &MetalWeight,
        compression: KvCompressionSpec,
        width: usize,
        rotary_dim: usize,
        cos: &[f32],
        sin: &[f32],
        eps: f32,
    ) -> Result<CompressedBatch<MetalTensor>, BackendError> {
        if positions.len() != key.rows || positions.is_empty() || positions.windows(2).any(|pair| pair[1] != pair[0] + 1) {
            return Err(compute(format!("V4 Metal compressor positions={positions:?} key rows={}", key.rows)));
        }
        let channels = if compression.overlap { width.checked_mul(2).ok_or_else(|| compute("V4 compressor channels 溢出"))? } else { width };
        if gate.rows != key.rows || key.cols != channels || gate.cols != channels || rotary_dim == 0 || rotary_dim > width || !rotary_dim.is_multiple_of(2) || cos.len() != sin.len() {
            return Err(compute(format!("V4 Metal compressor shape 非法: key=[{},{}] gate=[{},{}] width={width} rotary={rotary_dim}", key.rows, key.cols, gate.rows, gate.cols)));
        }
        let key = ops::to_f16_tensor(self, key).map_err(compute)?;
        let gate = ops::to_f16_tensor(self, gate).map_err(compute)?;
        let (position_bias, position_bias_f32) = constant(position_bias, compression.ratio * channels, "compressor position bias")?;
        let (norm, norm_f32) = constant(norm, width, "compressor norm")?;
        let state = match stream {
            CompressionStream::Attention => &mut storage.compressor,
            CompressionStream::Indexer => &mut storage.indexer,
        };
        let plan = state.state.plan(positions, compression.ratio).map_err(compute)?;
        let pending_rows = plan.pending_rows();
        let entry_start = plan.entry_start();
        let windows = plan.windows();
        let required_table = (entry_start + windows).checked_mul(compression.ratio).and_then(|rows| rows.checked_mul(rotary_dim / 2)).ok_or_else(|| compute("V4 compressor RoPE table offset 溢出"))?;
        if windows != 0 && required_table > cos.len() {
            return Err(compute(format!("V4 compressor RoPE table={}，需要 {required_table}", cos.len())));
        }
        let (pending_key, pending_gate, overlap_key, overlap_gate) = state.buffers(self, compression.ratio, width, compression.overlap)?;
        let cos = self.resident_f16_weight_buffer(cos);
        let sin = self.resident_f16_weight_buffer(sin);
        let (values, remaining) = ops::compressed_sparse::gated_compress(
            self,
            pending_key,
            pending_gate,
            pending_rows,
            &key,
            &gate,
            &position_bias,
            position_bias_f32,
            &norm,
            norm_f32,
            overlap_key,
            overlap_gate,
            compression.ratio,
            width,
            compression.overlap,
            entry_start,
            rotary_dim,
            &cos,
            &sin,
            eps,
        )
        .map_err(compute)?;
        state.state.commit(plan, remaining).map_err(compute)?;
        let visible_positions = plan.visible_positions();
        Ok(CompressedBatch { visible_positions, values })
    }

    fn compressed_sparse_decode(
        &self,
        storage: &mut Self::CompressedKvStorage,
        position: usize,
        query: &MetalTensor,
        key: &MetalTensor,
        value: &MetalTensor,
        compressed_key: Option<&MetalTensor>,
        compressed_value: Option<&MetalTensor>,
        compressed_index_key: Option<&MetalTensor>,
        index_query: Option<&MetalTensor>,
        index_head_weights: Option<&MetalTensor>,
        sink: Option<&MetalWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<MetalTensor, BackendError> {
        if query.rows != 1 || key.rows != 1 || value.rows != 1 {
            return Err(compute("Metal CSA decode 的 Q/K/V 必须是单行 tensor"));
        }
        let query = ops::to_f16_tensor(self, query).map_err(compute)?;
        let key = ops::to_f16_tensor(self, key).map_err(compute)?;
        let value = ops::to_f16_tensor(self, value).map_err(compute)?;
        match (compressed_key, compressed_value) {
            (Some(compressed_key), Some(compressed_value)) => storage.append_compressed(self, &[position], compressed_key, compressed_value, compressed_index_key)?,
            (None, None) if compressed_index_key.is_none() => {}
            _ => return Err(compute("Metal CSA compressed KV 输入必须成组出现")),
        }
        let positions = [position];
        storage.validate_recent_positions(&positions)?;
        let visible = storage.visible_counts(&positions)?;
        let visible_buffer = self.shared_buffer(bytes(&visible));
        let selection = select_history(self, storage, index_query, index_head_weights, spec, &visible, &visible_buffer)?;
        let (sink, sink_dtype) = sink_view(self, sink, spec.num_heads)?;
        let output = ops::compressed_sparse::attention(
            self,
            &query,
            &storage.compressed_key,
            &storage.compressed_value,
            &visible_buffer,
            selection.as_ref().map(|(buffer, top_k)| (buffer, *top_k)),
            &storage.recent_key,
            &storage.recent_value,
            storage.recent_start,
            storage.recent_len,
            storage.recent_first_position,
            &key,
            &value,
            position,
            true,
            &sink,
            sink_dtype,
            spec.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            spec.window_size,
        )
        .map_err(compute)?;
        storage.append_recent(self, &positions, &key, &value)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn compressed_sparse_prefill(
        &self,
        storage: &mut Self::CompressedKvStorage,
        positions: &[usize],
        causal_batch: bool,
        query: &MetalTensor,
        key: &MetalTensor,
        value: &MetalTensor,
        compressed_positions: Option<&[usize]>,
        compressed_key: Option<&MetalTensor>,
        compressed_value: Option<&MetalTensor>,
        compressed_index_key: Option<&MetalTensor>,
        index_query: Option<&MetalTensor>,
        index_head_weights: Option<&MetalTensor>,
        sink: Option<&MetalWeight>,
        spec: &CompressedSparseAttentionSpec,
    ) -> Result<MetalTensor, BackendError> {
        if positions.len() != query.rows || positions.is_empty() || positions.windows(2).any(|pair| pair[1] != pair[0] + 1) {
            return Err(compute(format!("Metal CSA prefill positions={positions:?} query rows={}", query.rows)));
        }
        let query = ops::to_f16_tensor(self, query).map_err(compute)?;
        let key = ops::to_f16_tensor(self, key).map_err(compute)?;
        let value = ops::to_f16_tensor(self, value).map_err(compute)?;
        match (compressed_positions, compressed_key, compressed_value) {
            (Some(compressed_positions), Some(compressed_key), Some(compressed_value)) => storage.append_compressed(self, compressed_positions, compressed_key, compressed_value, compressed_index_key)?,
            (None, None, None) if compressed_index_key.is_none() => {}
            _ => return Err(compute("Metal CSA prefill compressed positions/K/V 必须成组出现")),
        }
        storage.validate_recent_positions(positions)?;
        let visible = if causal_batch { storage.visible_counts(positions)? } else { storage.visible_counts(&vec![*positions.last().expect("positions 已非空"); positions.len()])? };
        let visible_buffer = self.shared_buffer(bytes(&visible));
        let selection = select_history(self, storage, index_query, index_head_weights, spec, &visible, &visible_buffer)?;
        let (sink, sink_dtype) = sink_view(self, sink, spec.num_heads)?;
        let output = ops::compressed_sparse::attention(
            self,
            &query,
            &storage.compressed_key,
            &storage.compressed_value,
            &visible_buffer,
            selection.as_ref().map(|(buffer, top_k)| (buffer, *top_k)),
            &storage.recent_key,
            &storage.recent_value,
            storage.recent_start,
            storage.recent_len,
            storage.recent_first_position,
            &key,
            &value,
            positions[0],
            causal_batch,
            &sink,
            sink_dtype,
            spec.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            spec.window_size,
        )
        .map_err(compute)?;
        storage.append_recent(self, positions, &key, &value)?;
        Ok(output)
    }
}

fn select_history(
    ctx: &MetalContext,
    storage: &MetalCompressedKvStorage,
    index_query: Option<&MetalTensor>,
    index_head_weights: Option<&MetalTensor>,
    spec: &CompressedSparseAttentionSpec,
    visible: &[u32],
    visible_buffer: &Buffer,
) -> Result<Option<(Buffer, usize)>, BackendError> {
    match spec.compression.map(|compression| compression.selection) {
        Some(CompressedSelection::LearnedIndexer(indexer)) => {
            let query = index_query.ok_or_else(|| compute("Metal CSA 缺少 index query"))?;
            let head_weights = index_head_weights.ok_or_else(|| compute("Metal CSA 缺少 index head weights"))?;
            if visible.iter().copied().max().unwrap_or(0) as usize <= indexer.top_k {
                return Ok(None);
            }
            let keys = storage.compressed_index_key.as_ref().ok_or_else(|| compute("Metal CSA compressed 历史缺少 index key"))?;
            let query = ops::to_f16_tensor(ctx, query).map_err(compute)?;
            let head_weights = ops::to_f16_tensor(ctx, head_weights).map_err(compute)?;
            let selection = ops::compressed_sparse::select_indexed_history(ctx, &query, keys, &head_weights, visible_buffer, storage.compressed_positions.len(), indexer.num_heads, indexer.head_dim, indexer.top_k).map_err(compute)?;
            Ok(Some((selection, indexer.top_k)))
        }
        Some(CompressedSelection::All) | None => Ok(None),
    }
}

fn sink_view(ctx: &MetalContext, sink: Option<&MetalWeight>, heads: usize) -> Result<(Buffer, u32), BackendError> {
    match sink {
        Some(weight) => {
            let (buffer, f32) = constant(weight, heads, "attention sink")?;
            Ok((buffer, u32::from(f32)))
        }
        None => Ok((ctx.shared_buffer_zeros(mem::size_of::<f16>()), 2)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{
        compressed_sparse::{CompressedKvState, GatedPoolState, normalize_rope_compressed_f32},
        dsa::DsaSpec,
        rope::{RopeSpec, RopeTable, RotaryLayout},
    };

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() <= tolerance, "index={index} actual={actual} expected={expected}");
        }
    }

    fn spec(compression: Option<KvCompressionSpec>) -> CompressedSparseAttentionSpec {
        CompressedSparseAttentionSpec {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            q_lora_rank: 2,
            output_groups: 1,
            output_lora_rank: 2,
            window_size: 2,
            rope: RopeSpec::Default { rotary_dim: 2, theta: 10_000.0 },
            compression,
            attention_sink: true,
        }
    }

    #[test]
    fn gated_compressor_matches_cpu_across_overlap_calls() {
        let ctx = MetalContext::new_default().unwrap();
        let compression = KvCompressionSpec {
            ratio: 4,
            overlap: true,
            selection: CompressedSelection::LearnedIndexer(DsaSpec { num_heads: 1, head_dim: 2, rope_dim: 2, top_k: 1, rotary_layout: RotaryLayout::Interleaved, kpool: 0, always_select_tail: false }),
        };
        let mut storage = ctx.allocate_compressed_kv(&spec(Some(compression))).unwrap();
        let position = MetalWeight::upload_f32(&ctx, &[0.0; 16]).unwrap();
        let norm = MetalWeight::F16(ctx.tensor_from_f32(&[1.0, 1.0], 1, 2).unwrap());
        let rope = RopeTable::precompute(16, 2, 10_000.0);
        let mut oracle = GatedPoolState::default();
        let mut begin = 0;
        for rows in [2, 2, 3, 1] {
            let positions = (begin..begin + rows).collect::<Vec<_>>();
            let key = (0..rows).flat_map(|row| [1.0 + (begin + row) as f32, 2.0, 10.0 + (begin + row) as f32, 20.0]).collect::<Vec<_>>();
            let gate = vec![0.0; key.len()];
            let actual = ctx
                .compress_gated(
                    &mut storage,
                    CompressionStream::Attention,
                    &positions,
                    &ctx.tensor_from_f32(&key, rows, 4).unwrap(),
                    &ctx.tensor_from_f32(&gate, rows, 4).unwrap(),
                    &position,
                    &norm,
                    compression,
                    2,
                    2,
                    &rope.cos,
                    &rope.sin,
                    1.0e-6,
                )
                .unwrap();
            let (visible, pooled) = oracle.push_f32(&positions, &key, &gate, &[0.0; 16], 4, 2, true).unwrap();
            let expected = normalize_rope_compressed_f32(&pooled, &visible, 4, 2, &[1.0, 1.0], 1.0e-6, 2, &rope.cos, &rope.sin).unwrap();
            assert_eq!(actual.visible_positions, visible);
            assert_close(&ctx.tensor_to_f32(&actual.values), &expected, 2.0e-2);
            begin += rows;
        }
    }

    #[test]
    fn resident_decode_reads_wrapped_recent_window() {
        let ctx = MetalContext::new_default().unwrap();
        let spec = spec(None);
        let mut storage = ctx.allocate_compressed_kv(&spec).unwrap();
        let sink = MetalWeight::upload_f32(&ctx, &[-1.0e9]).unwrap();
        let positions = [0, 1, 2];
        ctx.compressed_sparse_prefill(
            &mut storage,
            &positions,
            true,
            &ctx.tensor_from_f32(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 3, 2).unwrap(),
            &ctx.tensor_from_f32(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3, 2).unwrap(),
            &ctx.tensor_from_f32(&[10.0, 0.0, 20.0, 0.0, 0.0, 30.0], 3, 2).unwrap(),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&sink),
            &spec,
        )
        .unwrap();
        let actual = ctx
            .compressed_sparse_decode(
                &mut storage,
                3,
                &ctx.tensor_from_f32(&[1.0, 0.0], 1, 2).unwrap(),
                &ctx.tensor_from_f32(&[1.0, 0.0], 1, 2).unwrap(),
                &ctx.tensor_from_f32(&[40.0, 0.0], 1, 2).unwrap(),
                None,
                None,
                None,
                None,
                None,
                Some(&sink),
                &spec,
            )
            .unwrap();
        let mut oracle = CompressedKvState::new(2).unwrap();
        oracle.push_recent_batch(&positions, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[10.0, 0.0, 20.0, 0.0, 0.0, 30.0], 2).unwrap();
        oracle.push_recent(3, &[1.0, 0.0], &[40.0, 0.0]).unwrap();
        let expected = oracle.attend_f32(&[1.0, 0.0], 1, 1, 2, None, Some(&[-1.0e9])).unwrap();
        assert_close(&ctx.tensor_to_f32(&actual), &expected, 2.0e-2);
    }

    #[test]
    fn resident_prefill_attention_matches_cpu_oracle() {
        let ctx = MetalContext::new_default().unwrap();
        let spec = spec(None);
        let mut storage = ctx.allocate_compressed_kv(&spec).unwrap();
        let positions = [0, 1, 2];
        let query = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let key = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let value = [10.0, 0.0, 20.0, 0.0, 0.0, 30.0];
        let sink = MetalWeight::upload_f32(&ctx, &[-1.0e9]).unwrap();
        let actual = ctx
            .compressed_sparse_prefill(
                &mut storage,
                &positions,
                true,
                &ctx.tensor_from_f32(&query, 3, 2).unwrap(),
                &ctx.tensor_from_f32(&key, 3, 2).unwrap(),
                &ctx.tensor_from_f32(&value, 3, 2).unwrap(),
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&sink),
                &spec,
            )
            .unwrap();
        let expected = CompressedKvState::new(2).unwrap().attend_batch_f32(&query, 3, &positions, true, &key, &value, &[], &[], &[], None, None, None, 1, 1, 2, None, Some(&[-1.0e9])).unwrap();
        assert_close(&ctx.tensor_to_f32(&actual), &expected, 2.0e-2);
    }

    #[test]
    fn learned_indexer_and_compressed_attention_match_cpu() {
        let ctx = MetalContext::new_default().unwrap();
        let indexer = DsaSpec { num_heads: 1, head_dim: 2, rope_dim: 0, top_k: 1, rotary_layout: RotaryLayout::Interleaved, kpool: 0, always_select_tail: false };
        let compression = KvCompressionSpec { ratio: 4, overlap: true, selection: CompressedSelection::LearnedIndexer(indexer) };
        let spec = spec(Some(compression));
        let mut storage = ctx.allocate_compressed_kv(&spec).unwrap();
        let positions = [2];
        let query = [1.0, 0.0];
        let key = [1.0, 0.0];
        let value = [5.0, 0.0];
        let compressed_positions = [0, 1];
        let compressed_key = [1.0, 0.0, 1.0, 0.0];
        let compressed_value = [10.0, 0.0, 30.0, 0.0];
        let index_keys = [2.0, 0.0, 0.0, 2.0];
        let index_query = [0.0, 1.0];
        let head_weights = [1.0];
        let sink = MetalWeight::upload_f32(&ctx, &[-1.0e9]).unwrap();
        let actual = ctx
            .compressed_sparse_prefill(
                &mut storage,
                &positions,
                true,
                &ctx.tensor_from_f32(&query, 1, 2).unwrap(),
                &ctx.tensor_from_f32(&key, 1, 2).unwrap(),
                &ctx.tensor_from_f32(&value, 1, 2).unwrap(),
                Some(&compressed_positions),
                Some(&ctx.tensor_from_f32(&compressed_key, 2, 2).unwrap()),
                Some(&ctx.tensor_from_f32(&compressed_value, 2, 2).unwrap()),
                Some(&ctx.tensor_from_f32(&index_keys, 2, 2).unwrap()),
                Some(&ctx.tensor_from_f32(&index_query, 1, 2).unwrap()),
                Some(&ctx.tensor_from_f32(&head_weights, 1, 1).unwrap()),
                Some(&sink),
                &spec,
            )
            .unwrap();
        let expected = CompressedKvState::new(2)
            .unwrap()
            .attend_batch_f32(&query, 1, &positions, true, &key, &value, &compressed_positions, &compressed_key, &compressed_value, Some(&index_keys), Some(&index_query), Some(&head_weights), 1, 1, 2, Some(indexer), Some(&[-1.0e9]))
            .unwrap();
        assert_close(&ctx.tensor_to_f32(&actual), &expected, 2.0e-2);
    }
}
