//! ROCm compressed KV 的 speculative transaction 生命周期。

use super::*;

impl RocmCompressedKvStorage {
    pub(super) fn begin_transaction(&mut self, context: &RocmContext) -> Result<(), BackendError> {
        if self.transaction.is_some() {
            return Err(compute("V4 ROCm speculative cache transaction 已存在"));
        }
        let shared_recent = Arc::ptr_eq(&self.recent_key, &self.recent_value) && Arc::ptr_eq(&self.recent_key_scales, &self.recent_value_scales);
        let transaction = if let Some(mut transaction) = self.transaction_pool.take() {
            copy_required_buffer(context.device_id, &self.recent_key, &mut transaction.recent_key)?;
            copy_required_buffer(context.device_id, &self.recent_key_scales, &mut transaction.recent_key_scales)?;
            if shared_recent {
                transaction.recent_value = transaction.recent_key.clone();
                transaction.recent_value_scales = transaction.recent_key_scales.clone();
            } else {
                copy_required_buffer(context.device_id, &self.recent_value, &mut transaction.recent_value)?;
                copy_required_buffer(context.device_id, &self.recent_value_scales, &mut transaction.recent_value_scales)?;
            }
            transaction.rows.clear();
            transaction.recent_capacity = self.recent_capacity;
            transaction.recent_start = self.recent_start;
            transaction.recent_len = self.recent_len;
            transaction.recent_first_position = self.recent_first_position;
            transaction.next_recent_position = self.next_recent_position;
            transaction.compressed_len = self.compressed_positions.len();
            transaction.compressor = self.compressor.snapshot(context.device_id, Some(transaction.compressor))?;
            transaction.indexer = self.indexer.snapshot(context.device_id, Some(transaction.indexer))?;
            transaction.attention_replays.clear();
            transaction.compressor_replays.clear();
            transaction.indexer_replays.clear();
            transaction
        } else {
            let recent_key = clone_required_buffer(context.device_id, &self.recent_key)?;
            let recent_key_scales = clone_required_buffer(context.device_id, &self.recent_key_scales)?;
            let recent_value = if shared_recent { recent_key.clone() } else { clone_required_buffer(context.device_id, &self.recent_value)? };
            let recent_value_scales = if shared_recent { recent_key_scales.clone() } else { clone_required_buffer(context.device_id, &self.recent_value_scales)? };
            RocmCompressedTransaction {
                rows: Vec::new(),
                recent_key,
                recent_key_scales,
                recent_value,
                recent_value_scales,
                recent_capacity: self.recent_capacity,
                recent_start: self.recent_start,
                recent_len: self.recent_len,
                recent_first_position: self.recent_first_position,
                next_recent_position: self.next_recent_position,
                compressed_len: self.compressed_positions.len(),
                compressor: self.compressor.snapshot(context.device_id, None)?,
                indexer: self.indexer.snapshot(context.device_id, None)?,
                attention_replays: Vec::new(),
                compressor_replays: Vec::new(),
                indexer_replays: Vec::new(),
            }
        };
        self.transaction = Some(transaction);
        Ok(())
    }

    pub(super) fn commit_transaction(&mut self, context: &RocmContext, retained_rows: usize) -> Result<(), BackendError> {
        let mut transaction = self.transaction.take().ok_or_else(|| compute("V4 ROCm speculative cache transaction 不存在"))?;
        let rows = std::mem::take(&mut transaction.rows);
        if rows.is_empty() {
            return Err(compute("V4 ROCm speculative cache transaction 尚未写入 batch"));
        }
        if retained_rows > rows.len() {
            return Err(compute(format!("V4 ROCm speculative retained_rows={retained_rows} 超过 batch rows={}", rows.len())));
        }
        if retained_rows == rows.len() {
            transaction.attention_replays.clear();
            transaction.compressor_replays.clear();
            transaction.indexer_replays.clear();
            self.transaction_pool = Some(transaction);
            return Ok(());
        }
        std::mem::swap(&mut self.recent_key, &mut transaction.recent_key);
        std::mem::swap(&mut self.recent_key_scales, &mut transaction.recent_key_scales);
        std::mem::swap(&mut self.recent_value, &mut transaction.recent_value);
        std::mem::swap(&mut self.recent_value_scales, &mut transaction.recent_value_scales);
        std::mem::swap(&mut self.recent_capacity, &mut transaction.recent_capacity);
        std::mem::swap(&mut self.recent_start, &mut transaction.recent_start);
        std::mem::swap(&mut self.recent_len, &mut transaction.recent_len);
        std::mem::swap(&mut self.recent_first_position, &mut transaction.recent_first_position);
        std::mem::swap(&mut self.next_recent_position, &mut transaction.next_recent_position);
        self.compressed_positions.truncate(transaction.compressed_len);
        self.compressor.swap_snapshot(&mut transaction.compressor);
        self.indexer.swap_snapshot(&mut transaction.indexer);

        if retained_rows != 0 {
            let replay_compression = |state: &mut RocmGatedPoolState, replays: &mut Vec<RocmCompressionReplay>| -> Result<Vec<usize>, BackendError> {
                let mut remaining = retained_rows;
                let mut positions = Vec::new();
                for replay in replays.drain(..) {
                    let rows = remaining.min(replay.key.rows);
                    positions.extend(state.replay_prefix(context, replay, rows)?);
                    remaining -= rows;
                    if remaining == 0 {
                        break;
                    }
                }
                Ok(positions)
            };
            let compressor_positions = replay_compression(&mut self.compressor, &mut transaction.compressor_replays)?;
            let indexer_positions = replay_compression(&mut self.indexer, &mut transaction.indexer_replays)?;
            if !compressor_positions.is_empty() && !indexer_positions.is_empty() && compressor_positions != indexer_positions {
                return Err(compute(format!("V4 ROCm speculative compressor/indexer replay 位置不一致: {compressor_positions:?} / {indexer_positions:?}")));
            }
            if let Some(positions) = (!compressor_positions.is_empty()).then_some(compressor_positions).or_else(|| (!indexer_positions.is_empty()).then_some(indexer_positions)) {
                self.compressed_positions.extend(positions);
            }
            let mut remaining = retained_rows;
            for replay in transaction.attention_replays.drain(..) {
                let replay_rows = remaining.min(replay.positions.len());
                if replay_rows == 0 {
                    break;
                }
                let key = if replay_rows == replay.key.rows { replay.key } else { context.slice_token_rows(&replay.key, 0, replay_rows)? };
                let value = if replay_rows == replay.value.rows { replay.value } else { context.slice_token_rows(&replay.value, 0, replay_rows)? };
                self.quantize_batch(context, &key, &value)?;
                self.append_recent(&replay.positions[..replay_rows])?;
                remaining -= replay_rows;
            }
            if remaining != 0 {
                return Err(compute(format!("V4 ROCm speculative attention replay 缺少 {remaining} 行")));
            }
        }
        transaction.attention_replays.clear();
        transaction.compressor_replays.clear();
        transaction.indexer_replays.clear();
        self.transaction_pool = Some(transaction);
        Ok(())
    }

    pub(super) fn record_transaction_attention(&mut self, positions: &[usize], key: &RocmTensor, value: &RocmTensor) -> Result<(), BackendError> {
        let Some(transaction) = self.transaction.as_mut() else { return Ok(()) };
        if transaction.rows.last().is_some_and(|last| positions.first() != Some(&(last + 1))) {
            return Err(compute(format!("V4 ROCm speculative attention batch 不连续: previous={:?} next={:?}", transaction.rows.last(), positions.first())));
        }
        transaction.rows.extend_from_slice(positions);
        transaction.attention_replays.push(RocmAttentionReplay { positions: positions.to_vec(), key: key.clone(), value: value.clone() });
        Ok(())
    }
}

impl crate::backend::SpeculativeCacheBackend<RocmCompressedKvStorage> for RocmContext {
    fn begin_speculative_cache(&self, cache: &mut RocmCompressedKvStorage) -> Result<(), BackendError> {
        cache.begin_transaction(self)
    }

    fn commit_speculative_cache(&self, cache: &mut RocmCompressedKvStorage, commit: crate::backend::SpeculativeCacheCommit) -> Result<(), BackendError> {
        cache.commit_transaction(self, commit.retained_rows())
    }
}
