//! ROCm compressed KV 的持久化快照编解码。

use super::*;

impl RocmCompressedKvStorage {
    pub fn download_snapshot(&self) -> Result<RocmCompressedKvSerde, BackendError> {
        if self.transaction.is_some() {
            return Err(compute("V4 ROCm terminal cache 仍有 speculative transaction"));
        }
        let scale_row_bytes = self.scale_row_bytes();
        let recent_bytes = self.recent_capacity.checked_mul(self.kv_width).ok_or_else(|| compute("V4 ROCm recent snapshot 溢出"))?;
        let recent_scale_bytes = self.recent_capacity.checked_mul(scale_row_bytes).ok_or_else(|| compute("V4 ROCm recent scale snapshot 溢出"))?;
        let compressed_rows = self.compressed_positions.len();
        let compressed_bytes = compressed_rows.checked_mul(self.kv_width).ok_or_else(|| compute("V4 ROCm compressed snapshot 溢出"))?;
        let compressed_scale_bytes = compressed_rows.checked_mul(scale_row_bytes).ok_or_else(|| compute("V4 ROCm compressed scale snapshot 溢出"))?;
        let recent_shared = Arc::ptr_eq(&self.recent_key, &self.recent_value) && Arc::ptr_eq(&self.recent_key_scales, &self.recent_value_scales);
        let compressed_shared = Arc::ptr_eq(&self.compressed_key, &self.compressed_value) && Arc::ptr_eq(&self.compressed_key_scales, &self.compressed_value_scales);
        Ok(RocmCompressedKvSerde {
            window_size: self.window_size,
            kv_width: self.kv_width,
            q8_group_size: self.q8_group_size,
            recent_capacity: self.recent_capacity,
            recent_start: self.recent_start,
            recent_len: self.recent_len,
            recent_first_position: self.recent_first_position,
            next_recent_position: self.next_recent_position,
            recent_shared,
            recent_key: download_buffer(&self.recent_key, recent_bytes)?,
            recent_key_scales: download_buffer(&self.recent_key_scales, recent_scale_bytes)?,
            recent_value: if recent_shared { Vec::new() } else { download_buffer(&self.recent_value, recent_bytes)? },
            recent_value_scales: if recent_shared { Vec::new() } else { download_buffer(&self.recent_value_scales, recent_scale_bytes)? },
            compressed_positions: self.compressed_positions.clone(),
            compressed_shared,
            compressed_key: download_buffer(&self.compressed_key, compressed_bytes)?,
            compressed_key_scales: download_buffer(&self.compressed_key_scales, compressed_scale_bytes)?,
            compressed_value: if compressed_shared { Vec::new() } else { download_buffer(&self.compressed_value, compressed_bytes)? },
            compressed_value_scales: if compressed_shared { Vec::new() } else { download_buffer(&self.compressed_value_scales, compressed_scale_bytes)? },
            compressed_index_width: self.compressed_index_width,
            compressed_index_key: self
                .compressed_index_key
                .as_ref()
                .map(|buffer| {
                    let bytes = compressed_rows.checked_mul(self.compressed_index_width).and_then(|value| value.checked_mul(mem::size_of::<f32>())).ok_or_else(|| compute("V4 ROCm index snapshot 溢出"))?;
                    download_buffer(buffer, bytes)
                })
                .transpose()?,
            compressor: download_gated_pool(&self.compressor)?,
            indexer: download_gated_pool(&self.indexer)?,
        })
    }

    pub fn upload_snapshot(&mut self, snapshot: RocmCompressedKvSerde) -> Result<(), BackendError> {
        if (snapshot.window_size, snapshot.kv_width, snapshot.q8_group_size) != (self.window_size, self.kv_width, self.q8_group_size) {
            return Err(compute(format!("V4 ROCm cache snapshot 规格不匹配: {}/{}/{} != {}/{}/{}", snapshot.window_size, snapshot.kv_width, snapshot.q8_group_size, self.window_size, self.kv_width, self.q8_group_size)));
        }
        let device_id = self.recent_key.device_id();
        let scale_row_bytes = self.scale_row_bytes();
        validate_buffer(&snapshot.recent_key, snapshot.recent_capacity, self.kv_width, "recent key")?;
        validate_buffer(&snapshot.recent_key_scales, snapshot.recent_capacity, scale_row_bytes, "recent scales")?;
        if snapshot.recent_len > snapshot.recent_capacity || snapshot.recent_start >= snapshot.recent_capacity.max(1) {
            return Err(compute("V4 ROCm recent snapshot 元数据非法"));
        }
        self.recent_key = upload_bytes(device_id, &snapshot.recent_key)?;
        self.recent_key_scales = upload_bytes(device_id, &snapshot.recent_key_scales)?;
        if snapshot.recent_shared {
            self.recent_value = self.recent_key.clone();
            self.recent_value_scales = self.recent_key_scales.clone();
        } else {
            validate_buffer(&snapshot.recent_value, snapshot.recent_capacity, self.kv_width, "recent value")?;
            validate_buffer(&snapshot.recent_value_scales, snapshot.recent_capacity, scale_row_bytes, "recent value scales")?;
            self.recent_value = upload_bytes(device_id, &snapshot.recent_value)?;
            self.recent_value_scales = upload_bytes(device_id, &snapshot.recent_value_scales)?;
        }
        self.recent_capacity = snapshot.recent_capacity;
        self.recent_start = snapshot.recent_start;
        self.recent_len = snapshot.recent_len;
        self.recent_first_position = snapshot.recent_first_position;
        self.next_recent_position = snapshot.next_recent_position;
        let compressed_rows = snapshot.compressed_positions.len();
        if snapshot.compressed_positions.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(compute("V4 ROCm compressed positions 非递增"));
        }
        let index_bytes = compressed_rows.checked_mul(snapshot.compressed_index_width).and_then(|value| value.checked_mul(mem::size_of::<f32>())).ok_or_else(|| compute("V4 ROCm compressed index 大小溢出"))?;
        match &snapshot.compressed_index_key {
            Some(bytes) if snapshot.compressed_index_width != 0 && bytes.len() == index_bytes => {}
            None if snapshot.compressed_index_width == 0 => {}
            _ => return Err(compute(format!("V4 ROCm compressed index width={} bytes={} rows={} 不一致", snapshot.compressed_index_width, snapshot.compressed_index_key.as_ref().map_or(0, Vec::len), compressed_rows))),
        }
        if compressed_rows != 0 {
            validate_buffer(&snapshot.compressed_key, compressed_rows, self.kv_width, "compressed key")?;
            validate_buffer(&snapshot.compressed_key_scales, compressed_rows, scale_row_bytes, "compressed scales")?;
            self.compressed_key = upload_bytes(device_id, &snapshot.compressed_key)?;
            self.compressed_key_scales = upload_bytes(device_id, &snapshot.compressed_key_scales)?;
            if snapshot.compressed_shared {
                self.compressed_value = self.compressed_key.clone();
                self.compressed_value_scales = self.compressed_key_scales.clone();
            } else {
                validate_buffer(&snapshot.compressed_value, compressed_rows, self.kv_width, "compressed value")?;
                validate_buffer(&snapshot.compressed_value_scales, compressed_rows, scale_row_bytes, "compressed value scales")?;
                self.compressed_value = upload_bytes(device_id, &snapshot.compressed_value)?;
                self.compressed_value_scales = upload_bytes(device_id, &snapshot.compressed_value_scales)?;
            }
            self.compressed_index_key = snapshot.compressed_index_key.as_ref().map(|bytes| upload_bytes(device_id, bytes)).transpose()?;
            self.compressed_capacity = compressed_rows;
        } else {
            self.compressed_index_key = None;
            self.compressed_capacity = self.compressed_capacity.max(1);
        }
        self.compressed_positions = snapshot.compressed_positions;
        self.compressed_index_width = snapshot.compressed_index_width;
        self.compressor = upload_gated_pool(device_id, snapshot.compressor)?;
        self.indexer = upload_gated_pool(device_id, snapshot.indexer)?;
        self.visible_scalar_value = None;
        self.visible_scalar = None;
        self.batch_key = None;
        self.batch_key_scales = None;
        self.batch_value = None;
        self.batch_value_scales = None;
        self.batch_capacity = 0;
        self.transaction = None;
        self.transaction_pool = None;
        Ok(())
    }
}

fn download_buffer(buffer: &Arc<ops::hip::DeviceBuffer>, bytes: usize) -> Result<Vec<u8>, BackendError> {
    if bytes > buffer.bytes() {
        return Err(compute(format!("V4 ROCm snapshot bytes={bytes} 超过 device buffer={}", buffer.bytes())));
    }
    let mut output = vec![0; bytes];
    if bytes != 0 {
        buffer.copy_to_host(&mut output).map_err(compute)?;
    }
    Ok(output)
}

fn upload_bytes(device_id: i32, bytes: &[u8]) -> Result<Arc<ops::hip::DeviceBuffer>, BackendError> {
    if bytes.is_empty() {
        return Err(compute("V4 ROCm 不能恢复空 device buffer"));
    }
    ops::hip::DeviceBuffer::upload(device_id, bytes).map(Arc::new).map_err(compute)
}

fn validate_buffer(bytes: &[u8], rows: usize, row_bytes: usize, name: &str) -> Result<(), BackendError> {
    let expected = rows.checked_mul(row_bytes).ok_or_else(|| compute(format!("V4 ROCm {name} 大小溢出")))?;
    if bytes.len() != expected {
        return Err(compute(format!("V4 ROCm {name} bytes={}，期望 {expected}", bytes.len())));
    }
    Ok(())
}

fn download_gated_pool(pool: &RocmGatedPoolState) -> Result<RocmGatedPoolSerde, BackendError> {
    let (next_position, entry_count, pending_rows) = pool.state.parts();
    Ok(RocmGatedPoolSerde {
        next_position,
        entry_count,
        pending_rows,
        ratio: pool.ratio,
        width: pool.width,
        channels: pool.channels,
        overlap: pool.overlap,
        pending_key: pool.pending_key.as_ref().map(|buffer| download_buffer(buffer, buffer.bytes())).transpose()?,
        pending_gate: pool.pending_gate.as_ref().map(|buffer| download_buffer(buffer, buffer.bytes())).transpose()?,
        overlap_key: pool.overlap_key.as_ref().map(|buffer| download_buffer(buffer, buffer.bytes())).transpose()?,
        overlap_gate: pool.overlap_gate.as_ref().map(|buffer| download_buffer(buffer, buffer.bytes())).transpose()?,
    })
}

fn upload_gated_pool(device_id: i32, pool: RocmGatedPoolSerde) -> Result<RocmGatedPoolState, BackendError> {
    Ok(RocmGatedPoolState {
        state: CompressionState::from_parts(pool.next_position, pool.entry_count, pool.pending_rows),
        ratio: pool.ratio,
        width: pool.width,
        channels: pool.channels,
        overlap: pool.overlap,
        pending_key: pool.pending_key.as_deref().map(|bytes| upload_bytes(device_id, bytes)).transpose()?,
        pending_gate: pool.pending_gate.as_deref().map(|bytes| upload_bytes(device_id, bytes)).transpose()?,
        overlap_key: pool.overlap_key.as_deref().map(|bytes| upload_bytes(device_id, bytes)).transpose()?,
        overlap_gate: pool.overlap_gate.as_deref().map(|bytes| upload_bytes(device_id, bytes)).transpose()?,
        rope: None,
    })
}
