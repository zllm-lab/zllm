//! ROCm 扩散 Transformer capability；activation 全程保持 F32 device resident。

use std::sync::Arc;

use crate::{
    backend::{BackendError, DiffusionBackend},
    diffusion::{ModulationSegment, validate_modulation_segments},
    kernel::rocm::hip,
};

use super::{RocmContext, RocmTensor, RocmTensorDType, compute_error, device_tensor_bf16, device_tensor_f32, f32_tensor, resident_weight};

impl RocmContext {
    /// Ulysses source shard 在传输前完成 Q/K norm、RoPE 与 BF16 round。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_compact_qkv_heads_bf16(
        &self,
        tensor: &RocmTensor,
        query_norm: &super::RocmWeight,
        key_norm: &super::RocmWeight,
        total_heads: usize,
        heads: std::ops::Range<usize>,
        head_dim: usize,
        rotary_dim: usize,
        eps: f32,
        table_row_start: usize,
        cosine: &[f32],
        sine: &[f32],
    ) -> Result<RocmTensor, BackendError> {
        if tensor.dtype != RocmTensorDType::F32 || heads.start >= heads.end || heads.end > total_heads {
            return Err(compute_error(format!("ROCm prepare compact QKV dtype={:?} heads={heads:?}/{total_heads} 非法", tensor.dtype)));
        }
        let expected_cols = total_heads.checked_mul(head_dim).and_then(|value| value.checked_mul(3)).ok_or_else(|| compute_error("ROCm prepare compact QKV cols 溢出"))?;
        if tensor.cols != expected_cols {
            return Err(compute_error(format!("ROCm prepare compact QKV cols={}，期望 {expected_cols}", tensor.cols)));
        }
        let source = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm prepare compact QKV 缺少 device buffer"))?;
        if source.device_id() != self.device_id {
            return Err(compute_error(format!("ROCm prepare compact QKV device={}，当前 device={}", source.device_id(), self.device_id)));
        }
        let query_norm = query_norm.resident().map(Arc::as_ref).filter(|_| !query_norm.resident_bf16()).ok_or_else(|| compute_error("ROCm prepare compact QKV query norm 缺少 F32 resident weight"))?;
        let key_norm = key_norm.resident().map(Arc::as_ref).filter(|_| !key_norm.resident_bf16()).ok_or_else(|| compute_error("ROCm prepare compact QKV key norm 缺少 F32 resident weight"))?;
        let local_heads = heads.len();
        let output = hip::try_prepare_compact_attention_qkv_range_resident_bf16(
            self.device_id,
            source,
            query_norm,
            key_norm,
            tensor.rows,
            table_row_start,
            total_heads,
            heads.start,
            local_heads,
            head_dim,
            rotary_dim,
            eps,
            cosine,
            sine,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_bf16(output, tensor.rows, local_heads * head_dim * 3))
    }

    pub(crate) fn full_attention_prepared_qkv_bf16(&self, qkv: RocmTensor, head_count: usize, head_dim: usize, score_scale: f32) -> Result<RocmTensor, BackendError> {
        let columns = head_count.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm prepared QKV attention columns 溢出"))?;
        if qkv.dtype != RocmTensorDType::Bf16 || qkv.cols != columns.checked_mul(3).ok_or_else(|| compute_error("ROCm prepared QKV attention input columns 溢出"))? {
            return Err(compute_error(format!("ROCm prepared QKV attention input=[{},{}] dtype={:?} heads={head_count} head_dim={head_dim}", qkv.rows, qkv.cols, qkv.dtype)));
        }
        let rows = qkv.rows;
        let qkv = qkv.device.ok_or_else(|| compute_error("ROCm prepared QKV attention 缺少 device buffer"))?;
        let qkv = Arc::try_unwrap(qkv).map_err(|_| compute_error("ROCm prepared QKV attention device buffer 仍被共享"))?;
        let output = hip::try_full_attention_prepared_qkv_resident_bf16(self.device_id, qkv, rows, head_count, head_dim, score_scale).map_err(compute_error)?;
        Ok(device_tensor_f32(output, rows, columns))
    }

    pub(crate) fn compact_prepared_qkv_heads_bf16(&self, tensor: &RocmTensor, total_heads: usize, heads: std::ops::Range<usize>, head_dim: usize) -> Result<RocmTensor, BackendError> {
        if tensor.dtype != RocmTensorDType::Bf16 || heads.start >= heads.end || heads.end > total_heads {
            return Err(compute_error(format!("ROCm compact prepared QKV dtype={:?} heads={heads:?}/{total_heads} 非法", tensor.dtype)));
        }
        let expected_cols = total_heads.checked_mul(head_dim).and_then(|value| value.checked_mul(3)).ok_or_else(|| compute_error("ROCm compact prepared QKV cols 溢出"))?;
        if tensor.cols != expected_cols {
            return Err(compute_error(format!("ROCm compact prepared QKV cols={}，期望 {expected_cols}", tensor.cols)));
        }
        let source = tensor.device.as_deref().ok_or_else(|| compute_error("ROCm compact prepared QKV 缺少 device buffer"))?;
        let output = hip::try_compact_prepared_attention_qkv_range_resident_bf16(self.device_id, source, tensor.rows, total_heads, heads.start, heads.len(), head_dim).map_err(compute_error)?;
        Ok(device_tensor_bf16(output, tensor.rows, heads.len() * head_dim * 3))
    }
}

impl DiffusionBackend for RocmContext {
    fn transfer_tensor(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError> {
        self.tensor_on_device(tensor)
    }

    fn silu(&self, input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        let input = f32_tensor(self, input)?;
        let output = hip::try_silu_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm SiLU input 缺少 device buffer"))?,
            input.rows.checked_mul(input.cols).ok_or_else(|| compute_error("ROCm SiLU elements 溢出"))?,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn add_row_bias(&self, input: &Self::Tensor, bias: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        if bias.data().len() != input.cols {
            return Err(compute_error(format!("ROCm row bias={} 期望 {}", bias.data().len(), input.cols)));
        }
        let input = f32_tensor(self, input)?;
        let output = hip::try_add_row_bias_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm row bias input 缺少 device buffer"))?, resident_weight(bias, "row bias")?, input.rows, input.cols)
            .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn concat_rows(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        if left.cols != right.cols {
            return Err(compute_error(format!("ROCm row concat cols={}/{} 不一致", left.cols, right.cols)));
        }
        let left = f32_tensor(self, left)?;
        let right = f32_tensor(self, right)?;
        let left_elements = left.rows.checked_mul(left.cols).ok_or_else(|| compute_error("ROCm row concat left 大小溢出"))?;
        let right_elements = right.rows.checked_mul(right.cols).ok_or_else(|| compute_error("ROCm row concat right 大小溢出"))?;
        let rows = left.rows.checked_add(right.rows).ok_or_else(|| compute_error("ROCm row concat rows 溢出"))?;
        let output = hip::try_concat_rows_resident_f32(
            self.device_id,
            left.device.as_deref().ok_or_else(|| compute_error("ROCm row concat left 缺少 device buffer"))?,
            left_elements,
            right.device.as_deref().ok_or_else(|| compute_error("ROCm row concat right 缺少 device buffer"))?,
            right_elements,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, rows, left.cols))
    }

    fn flow_step(&self, sample: &Self::Tensor, velocity: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError> {
        if sample.rows != velocity.rows || sample.cols != velocity.cols {
            return Err(compute_error(format!("ROCm flow step shape=[{},{}]/[{},{}] 不一致", sample.rows, sample.cols, velocity.rows, velocity.cols,)));
        }
        let sample = f32_tensor(self, sample)?;
        let velocity = f32_tensor(self, velocity)?;
        let elements = sample.rows.checked_mul(sample.cols).ok_or_else(|| compute_error("ROCm flow step 大小溢出"))?;
        let velocity = hip::try_scale_tensor_resident_f32(self.device_id, velocity.device.as_deref().ok_or_else(|| compute_error("ROCm flow step velocity 缺少 device buffer"))?, elements, scale).map_err(compute_error)?;
        let output = hip::try_add_resident_f32(self.device_id, sample.device.as_deref().ok_or_else(|| compute_error("ROCm flow step sample 缺少 device buffer"))?, &velocity, elements, 1.0).map_err(compute_error)?;
        Ok(device_tensor_f32(output, sample.rows, sample.cols))
    }

    fn modulation_chunks(&self, input: &Self::Tensor, modalities: usize, chunks: usize, hidden: usize) -> Result<Vec<Self::Tensor>, BackendError> {
        let expected = modalities.checked_mul(chunks).and_then(|value| value.checked_mul(hidden)).ok_or_else(|| compute_error("ROCm modulation columns 溢出"))?;
        if input.cols != expected {
            return Err(compute_error(format!("ROCm modulation input cols={} 期望 {expected}", input.cols)));
        }
        let input = f32_tensor(self, input)?;
        let output =
            hip::try_modulation_chunks_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm modulation input 缺少 device buffer"))?, input.rows, modalities, chunks, hidden).map_err(compute_error)?;
        Ok(output.into_iter().map(|buffer| device_tensor_f32(buffer, input.rows * modalities, hidden)).collect())
    }

    fn rmsnorm_heads(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        let columns = head_count.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm per-head RMSNorm columns 溢出"))?;
        if input.cols != columns || weight.data().len() != head_dim {
            return Err(compute_error(format!("ROCm per-head RMSNorm input=[{},{}] weight={} heads={head_count} head_dim={head_dim}", input.rows, input.cols, weight.data().len(),)));
        }
        let input = f32_tensor(self, input)?;
        let output = hip::try_rmsnorm_resident_to_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm per-head RMSNorm input 缺少 device buffer"))?,
            weight.data(),
            input.rows.checked_mul(head_count).ok_or_else(|| compute_error("ROCm per-head RMSNorm rows 溢出"))?,
            head_dim,
            eps,
            false,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn full_attention_qkv(
        &self,
        qkv: Self::Tensor,
        query_norm: &Self::Weight,
        key_norm: &Self::Weight,
        head_count: usize,
        head_dim: usize,
        rotary_dim: usize,
        eps: f32,
        cosine: &[f32],
        sine: &[f32],
        score_scale: f32,
    ) -> Result<Self::Tensor, BackendError> {
        let columns = head_count.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm fused QKV attention columns 溢出"))?;
        if qkv.cols != columns.checked_mul(3).ok_or_else(|| compute_error("ROCm fused QKV attention input columns 溢出"))? || query_norm.data().len() != head_dim || key_norm.data().len() != head_dim {
            return Err(compute_error(format!("ROCm fused QKV attention input=[{},{}] q_norm={} k_norm={} heads={head_count} head_dim={head_dim}", qkv.rows, qkv.cols, query_norm.data().len(), key_norm.data().len())));
        }
        let rows = qkv.rows;
        let run = |context: &RocmContext, qkv: RocmTensor, total_heads: usize, head_start: usize, heads: usize| -> Result<RocmTensor, BackendError> {
            let local_columns = heads.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm fused QKV local columns 溢出"))?;
            let qkv = context.tensor_as_f32(qkv)?.device.ok_or_else(|| compute_error("ROCm fused QKV attention 缺少 device buffer"))?;
            let qkv = std::sync::Arc::try_unwrap(qkv).map_err(|_| compute_error("ROCm fused QKV attention device buffer 仍被共享"))?;
            let query_norm = query_norm.resident().map(Arc::as_ref).filter(|_| !query_norm.resident_bf16()).ok_or_else(|| compute_error("ROCm fused QKV query norm 缺少 F32 resident weight"))?;
            let key_norm = key_norm.resident().map(Arc::as_ref).filter(|_| !key_norm.resident_bf16()).ok_or_else(|| compute_error("ROCm fused QKV key norm 缺少 F32 resident weight"))?;
            let output =
                hip::try_full_attention_qkv_range_resident_weights_f32(context.device_id, qkv, query_norm, key_norm, rows, total_heads, head_start, heads, head_dim, rotary_dim, eps, cosine, sine, score_scale).map_err(compute_error)?;
            Ok(device_tensor_f32(output, rows, local_columns))
        };

        let peer_device = hip::options().attention_devices.as_deref().and_then(|devices| {
            let [first, second] = devices else { return None };
            if self.device_id != *first && self.device_id != *second {
                return None;
            }
            Some(if self.device_id == *first { *second } else { *first })
        });
        let Some(peer_device) = peer_device else {
            return run(self, qkv, head_count, 0, head_count);
        };
        if head_count < 2 || head_count % 2 != 0 {
            return Err(compute_error(format!("ROCm 双卡 attention 要求偶数 head，实际 {head_count}")));
        }

        let half_heads = head_count / 2;
        let half_columns = half_heads.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm 双卡 attention half columns 溢出"))?;
        let source = qkv.device.as_deref().ok_or_else(|| compute_error("ROCm 双卡 attention QKV 缺少 device buffer"))?;
        let compact = hip::try_compact_qkv_head_range_resident_f32(self.device_id, source, rows, head_count, half_heads, half_heads, head_dim).map_err(compute_error)?;
        // compact 与上游 QKV projection 在主卡同一 stream，完成后再进入 P2P。
        <RocmContext as crate::backend::BackendResources>::synchronize(self)?;
        let peer = RocmContext::new(peer_device).map_err(compute_error)?;
        let peer_qkv = compact.copy_stable_to_device_ready(peer_device).map_err(compute_error)?;
        let peer_qkv = device_tensor_f32(peer_qkv, rows, half_columns * 3);
        let (first, second) = std::thread::scope(|scope| {
            let second = scope.spawn(|| {
                let output = run(&peer, peer_qkv, half_heads, 0, half_heads)?;
                // async allocation 不能跨线程直接 P2P，先在创建它的 worker 内稳定化。
                let device = output.device.as_deref().ok_or_else(|| compute_error("ROCm 双卡 attention output 缺少 device buffer"))?;
                let stable = device.copy_to_stable().map_err(compute_error)?;
                Ok::<_, BackendError>(device_tensor_f32(stable, output.rows, output.cols))
            });
            let first = run(self, qkv, head_count, 0, half_heads);
            let second = second.join().map_err(|_| compute_error("ROCm 双卡 attention worker panic"))?;
            Ok::<_, BackendError>((first?, second?))
        })?;
        let second_device = second.device.as_deref().ok_or_else(|| compute_error("ROCm 双卡 attention stable output 缺少 device buffer"))?;
        let second = second_device.copy_stable_to_device_ready(self.device_id).map_err(compute_error)?;
        let second = device_tensor_f32(second, rows, half_columns);
        <RocmContext as crate::backend::Backend>::concat_columns(self, &first, &second)
    }

    fn full_attention(&self, query: Self::Tensor, key: Self::Tensor, value: Self::Tensor, head_count: usize, head_dim: usize, score_scale: f32) -> Result<Self::Tensor, BackendError> {
        if query.rows != key.rows || query.rows != value.rows || query.cols != key.cols || query.cols != value.cols || query.cols != head_count.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm full attention columns 溢出"))? {
            return Err(compute_error("ROCm full attention Q/K/V shape 不兼容"));
        }
        let rows = query.rows;
        let cols = query.cols;
        let query = self.tensor_as_f32(query)?.device.ok_or_else(|| compute_error("ROCm full attention query 缺少 device buffer"))?;
        let key = self.tensor_as_f32(key)?.device.ok_or_else(|| compute_error("ROCm full attention key 缺少 device buffer"))?;
        let value = self.tensor_as_f32(value)?.device.ok_or_else(|| compute_error("ROCm full attention value 缺少 device buffer"))?;
        // 宽 head 的长序列用 wave 归约；短序列保留低寄存器的 block 路径。
        let output = if head_dim <= 128 {
            let query = std::sync::Arc::try_unwrap(query).map_err(|_| compute_error("ROCm full attention query device buffer 仍被共享"))?;
            let key = std::sync::Arc::try_unwrap(key).map_err(|_| compute_error("ROCm full attention key device buffer 仍被共享"))?;
            let value = std::sync::Arc::try_unwrap(value).map_err(|_| compute_error("ROCm full attention value device buffer 仍被共享"))?;
            hip::try_full_attention_resident_f32(self.device_id, query, key, value, rows, head_count, head_dim, score_scale)
        } else if head_dim == 512 && rows >= 4096 {
            hip::try_full_attention_wide_resident_f32(self.device_id, &query, &key, &value, rows, head_count, head_dim, score_scale)
        } else {
            let mut visible = Vec::with_capacity(rows * 2);
            for _ in 0..rows {
                visible.extend_from_slice(&[0, u32::try_from(rows).map_err(|_| compute_error("ROCm full attention rows 超过 u32"))?]);
            }
            hip::try_block_attention_resident_f32(self.device_id, &query, &key, &value, &visible, rows, rows, head_count, head_count, head_dim, score_scale)
        }
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, rows, cols))
    }

    fn full_attention_batched(&self, query: Self::Tensor, key: Self::Tensor, value: Self::Tensor, batch: usize, rows: usize, head_count: usize, head_dim: usize, score_scale: f32) -> Result<Self::Tensor, BackendError> {
        let total_rows = batch.checked_mul(rows).ok_or_else(|| compute_error("ROCm batched full attention rows 溢出"))?;
        if batch == 0
            || query.rows != total_rows
            || query.rows != key.rows
            || query.rows != value.rows
            || query.cols != key.cols
            || query.cols != value.cols
            || query.cols != head_count.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm batched full attention columns 溢出"))?
        {
            return Err(compute_error("ROCm batched full attention Q/K/V shape 不兼容"));
        }
        let cols = query.cols;
        let query = self.tensor_as_f32(query)?.device.ok_or_else(|| compute_error("ROCm batched full attention query 缺少 device buffer"))?;
        let key = self.tensor_as_f32(key)?.device.ok_or_else(|| compute_error("ROCm batched full attention key 缺少 device buffer"))?;
        let value = self.tensor_as_f32(value)?.device.ok_or_else(|| compute_error("ROCm batched full attention value 缺少 device buffer"))?;
        let query = std::sync::Arc::try_unwrap(query).map_err(|_| compute_error("ROCm batched full attention query device buffer 仍被共享"))?;
        let key = std::sync::Arc::try_unwrap(key).map_err(|_| compute_error("ROCm batched full attention key device buffer 仍被共享"))?;
        let value = std::sync::Arc::try_unwrap(value).map_err(|_| compute_error("ROCm batched full attention value device buffer 仍被共享"))?;
        let output = hip::try_full_attention_batched_resident_f32(self.device_id, query, key, value, batch, rows, head_count, head_dim, score_scale).map_err(compute_error)?;
        Ok(device_tensor_f32(output, total_rows, cols))
    }

    fn varlen_attention(&self, query: Self::Tensor, key: Self::Tensor, value: Self::Tensor, query_offsets: &[usize], kv_offsets: &[usize], head_count: usize, head_dim: usize, score_scale: f32) -> Result<Self::Tensor, BackendError> {
        use crate::backend::{BlockAttentionBackend, SegmentedTensorBackend};
        let geometry = crate::attention::gqa::GqaGeometry { num_heads: head_count, num_kv_heads: head_count, head_dim };
        let spec = crate::attention::block::BlockAttentionSpec::varlen(geometry, query.rows, key.rows, query_offsets, kv_offsets, score_scale).map_err(compute_error)?;
        let columns = geometry.query_columns().map_err(compute_error)?;
        if query.cols != columns || key.cols != columns || value.cols != columns || key.rows != value.rows {
            return Err(compute_error(format!("ROCm varlen attention shape 不一致: Q=[{},{}] K=[{},{}] V=[{},{}] heads={head_count} dim={head_dim}", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols)));
        }
        if query_offsets != kv_offsets || head_dim > 128 || !head_dim.is_multiple_of(16) {
            return self.block_attention(&query, &key, &value, &spec);
        }
        // 相邻等长窗口共享一次 WMMA attention；边缘窗口保持各自边界，不做 padding 扩窗。
        let mut outputs = Vec::new();
        let mut first = 0;
        while first + 1 < query_offsets.len() {
            let rows = query_offsets[first + 1] - query_offsets[first];
            let mut end = first + 1;
            while end + 1 < query_offsets.len() && query_offsets[end + 1] - query_offsets[end] == rows {
                end += 1;
            }
            let total = query_offsets[end] - query_offsets[first];
            let q = self.slice_token_rows(&query, query_offsets[first], total)?;
            let k = self.slice_token_rows(&key, query_offsets[first], total)?;
            let v = self.slice_token_rows(&value, query_offsets[first], total)?;
            let output = if head_dim == 128 {
                let take = |tensor| -> Result<hip::DeviceBuffer, BackendError> {
                    let buffer = self.tensor_as_f32(tensor)?.device.ok_or_else(|| compute_error("ROCm varlen attention 缺少 device buffer"))?;
                    Arc::try_unwrap(buffer).map_err(|_| compute_error("ROCm varlen attention view 仍被共享"))
                };
                let output = hip::try_full_attention_batched_zero_margin_resident_f32(self.device_id, take(q)?, take(k)?, take(v)?, end - first, rows, head_count, score_scale).map_err(compute_error)?;
                device_tensor_f32(output, total, columns)
            } else {
                self.full_attention_batched(q, k, v, end - first, rows, head_count, head_dim, score_scale)?
            };
            outputs.push(output);
            first = end;
        }
        self.concat_token_rows(&outputs.iter().collect::<Vec<_>>())
    }

    fn adaln_modulate(&self, input: &Self::Tensor, shift: &Self::Tensor, scale: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows || (shift.rows != 1 && shift.rows != input.rows) {
            return Err(compute_error(format!("ROCm AdaLN shape 不兼容: input=[{},{}] shift=[{},{}] scale=[{},{}]", input.rows, input.cols, shift.rows, shift.cols, scale.rows, scale.cols)));
        }
        let input = f32_tensor(self, input)?;
        let shift = f32_tensor(self, shift)?;
        let scale = f32_tensor(self, scale)?;
        let output = hip::try_adaln_modulate_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm AdaLN input 缺少 device buffer"))?,
            shift.device.as_deref().ok_or_else(|| compute_error("ROCm AdaLN shift 缺少 device buffer"))?,
            scale.device.as_deref().ok_or_else(|| compute_error("ROCm AdaLN scale 缺少 device buffer"))?,
            input.rows,
            input.cols,
            shift.rows,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn diffusion_tensor_from_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        let expected = rows.checked_mul(cols).ok_or_else(|| compute_error("ROCm diffusion tensor 大小溢出"))?;
        if values.len() != expected {
            return Err(compute_error(format!("ROCm diffusion tensor values={}，期望 {expected}", values.len())));
        }
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * std::mem::size_of::<f32>()) };
        let output = hip::DeviceBuffer::upload(self.device_id, bytes).map_err(compute_error)?;
        Ok(device_tensor_f32(output, rows, cols))
    }

    fn timestep_embedding(&self, timesteps: &[f32], dim: usize) -> Result<Self::Tensor, BackendError> {
        let output = hip::try_timestep_embedding_resident_f32(self.device_id, timesteps, dim).map_err(compute_error)?;
        Ok(device_tensor_f32(output, timesteps.len(), dim))
    }

    fn adaln_modulate_segmented(&self, input: &Self::Tensor, shift: &Self::Tensor, scale: &Self::Tensor, segments: &[ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows {
            return Err(compute_error("ROCm segmented AdaLN shape 不兼容"));
        }
        validate_modulation_segments(segments, input.rows, shift.rows).map_err(compute_error)?;
        let input = f32_tensor(self, input)?;
        let shift = f32_tensor(self, shift)?;
        let scale = f32_tensor(self, scale)?;
        let descriptors = segments.iter().map(|segment| (segment.rows.start, segment.rows.end, segment.modulation_row)).collect::<Vec<_>>();
        let output = hip::try_adaln_modulate_segmented_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm segmented AdaLN input 缺少 device buffer"))?,
            shift.device.as_deref().ok_or_else(|| compute_error("ROCm segmented AdaLN shift 缺少 device buffer"))?,
            scale.device.as_deref().ok_or_else(|| compute_error("ROCm segmented AdaLN scale 缺少 device buffer"))?,
            input.rows,
            input.cols,
            shift.rows,
            &descriptors,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn rmsnorm_adaln_modulate_segmented(&self, input: &Self::Tensor, norm_weight: &Self::Weight, eps: f32, shift: &Self::Tensor, scale: &Self::Tensor, segments: &[ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows || norm_weight.data().len() != input.cols {
            return Err(compute_error(format!("ROCm fused RMSNorm AdaLN shape 不兼容: input=[{},{}] norm={} shift=[{},{}] scale=[{},{}]", input.rows, input.cols, norm_weight.data().len(), shift.rows, shift.cols, scale.rows, scale.cols)));
        }
        validate_modulation_segments(segments, input.rows, shift.rows).map_err(compute_error)?;
        let input = self.tensor_on_device(input.clone())?;
        let shift = f32_tensor(self, shift)?;
        let scale = f32_tensor(self, scale)?;
        let descriptors = segments.iter().map(|segment| (segment.rows.start, segment.rows.end, segment.modulation_row)).collect::<Vec<_>>();
        let output = hip::try_rmsnorm_adaln_segmented_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm fused RMSNorm AdaLN input 缺少 device buffer"))?,
            resident_weight(norm_weight, "fused RMSNorm AdaLN norm")?,
            shift.device.as_deref().ok_or_else(|| compute_error("ROCm fused RMSNorm AdaLN shift 缺少 device buffer"))?,
            scale.device.as_deref().ok_or_else(|| compute_error("ROCm fused RMSNorm AdaLN scale 缺少 device buffer"))?,
            input.rows,
            input.cols,
            shift.rows,
            eps,
            &descriptors,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn gated_residual_segmented(&self, residual: &Self::Tensor, update: &Self::Tensor, gate: &Self::Tensor, segments: &[ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        if residual.rows != update.rows || residual.cols != update.cols || gate.cols != residual.cols {
            return Err(compute_error("ROCm segmented gated residual shape 不兼容"));
        }
        validate_modulation_segments(segments, residual.rows, gate.rows).map_err(compute_error)?;
        let residual = f32_tensor(self, residual)?;
        let update = f32_tensor(self, update)?;
        let gate = f32_tensor(self, gate)?;
        let descriptors = segments.iter().map(|segment| (segment.rows.start, segment.rows.end, segment.modulation_row)).collect::<Vec<_>>();
        let output = hip::try_gated_residual_segmented_resident_f32(
            self.device_id,
            residual.device.as_deref().ok_or_else(|| compute_error("ROCm segmented residual 缺少 device buffer"))?,
            update.device.as_deref().ok_or_else(|| compute_error("ROCm segmented update 缺少 device buffer"))?,
            gate.device.as_deref().ok_or_else(|| compute_error("ROCm segmented gate 缺少 device buffer"))?,
            residual.rows,
            residual.cols,
            gate.rows,
            &descriptors,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, residual.rows, residual.cols))
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::{Backend, BackendResources, DiffusionBackend, LinearWeight, SegmentedTensorBackend};

    use super::RocmContext;

    #[test]
    #[ignore = "需要 ROCm device 0"]
    fn wide_full_attention_matches_cpu() {
        let context = RocmContext::new(0).unwrap();
        let cpu = crate::backend::cpu::CpuContext;
        for (rows, heads, dim) in [(1, 1, 129), (7, 3, 193), (8, 1, 256), (9, 1, 511), (33, 3, 512)] {
            let columns = heads * dim;
            let query = (0..rows * columns).map(|i| ((i * 17 % 251) as f32 - 125.0) / 31.0).collect::<Vec<_>>();
            let key = (0..rows * columns).map(|i| ((i * 29 % 257) as f32 - 128.0) / 43.0).collect::<Vec<_>>();
            let value = (0..rows * columns).map(|i| ((i * 11 % 263) as f32 - 131.0) / 67.0).collect::<Vec<_>>();
            let scale = (dim as f32).sqrt().recip();
            let expected = cpu
                .full_attention(cpu.diffusion_tensor_from_f32(&query, rows, columns).unwrap(), cpu.diffusion_tensor_from_f32(&key, rows, columns).unwrap(), cpu.diffusion_tensor_from_f32(&value, rows, columns).unwrap(), heads, dim, scale)
                .unwrap();
            let query = context.diffusion_tensor_from_f32(&query, rows, columns).unwrap();
            let key = context.diffusion_tensor_from_f32(&key, rows, columns).unwrap();
            let value = context.diffusion_tensor_from_f32(&value, rows, columns).unwrap();
            let actual = super::hip::try_full_attention_wide_resident_f32(0, query.device.as_deref().unwrap(), key.device.as_deref().unwrap(), value.device.as_deref().unwrap(), rows, heads, dim, scale).unwrap();
            let actual = actual.download_f32(rows * columns).unwrap();
            for (i, (&actual, &expected)) in actual.iter().zip(&expected.data).enumerate() {
                assert!(actual.is_finite() && (actual - expected).abs() <= 0.01 + 0.01 * expected.abs(), "wide attention rows={rows} heads={heads} dim={dim} index={i}: actual={actual} expected={expected}");
            }
        }
    }

    #[test]
    #[ignore = "需要 ROCm device 0"]
    fn h3_full_attention_is_invariant_to_head_partition() {
        let context = RocmContext::new(0).unwrap();
        let rows = 257;
        let heads = 56;
        let head_dim = 128;
        let values = (0..rows * heads * head_dim * 3).map(|index| ((index * 17 % 1009) as f32 - 504.0) / 257.0).collect::<Vec<_>>();
        let norm = vec![1.0; head_dim];
        let query_norm = context.prepare_weight(LinearWeight::F32(&norm), 1, head_dim).unwrap();
        let key_norm = context.prepare_weight(LinearWeight::F32(&norm), 1, head_dim).unwrap();
        let angles = (0..rows * head_dim / 2).map(|index| (index % 97) as f32 * 0.0078125).collect::<Vec<_>>();
        let cosine = angles.iter().map(|angle| angle.cos()).collect::<Vec<_>>();
        let sine = angles.iter().map(|angle| angle.sin()).collect::<Vec<_>>();

        let full_qkv = context.tensor_from_f32(values.clone(), rows, heads * head_dim * 3).unwrap();
        let full = context.full_attention_qkv(full_qkv, &query_norm, &key_norm, heads, head_dim, head_dim, 1e-6, &cosine, &sine, (head_dim as f32).sqrt().recip()).unwrap();
        let full = context.tensor_to_f32(&full).unwrap();

        let source = context.tensor_from_f32(values.clone(), rows, heads * head_dim * 3).unwrap();
        let mut parts = Vec::new();
        for start in (0..heads).step_by(14) {
            let qkv = context.compact_qkv_heads(&source, heads, start..start + 14, head_dim).unwrap();
            parts.push(context.full_attention_qkv(qkv, &query_norm, &key_norm, 14, head_dim, head_dim, 1e-6, &cosine, &sine, (head_dim as f32).sqrt().recip()).unwrap());
        }
        let mut partitioned = parts.remove(0);
        for part in parts {
            partitioned = context.concat_columns(&partitioned, &part).unwrap();
        }
        let partitioned = context.tensor_to_f32(&partitioned).unwrap();
        let max_error = full.iter().zip(&partitioned).map(|(left, right)| (left - right).abs()).fold(0.0f32, f32::max);
        assert!(max_error <= 1e-6, "H3 attention head partition max_error={max_error}");

        let source = context.tensor_from_f32(values, rows, heads * head_dim * 3).unwrap();
        let mut prepared_parts = Vec::new();
        for head_start in (0..heads).step_by(14) {
            let mut sequence_parts = Vec::new();
            for sequence in [0..113, 113..rows] {
                let qkv = context.slice_token_rows(&source, sequence.start, sequence.len()).unwrap();
                sequence_parts.push(
                    context
                        .prepare_compact_qkv_heads_bf16(
                            &qkv,
                            &query_norm,
                            &key_norm,
                            heads,
                            head_start..head_start + 14,
                            head_dim,
                            head_dim,
                            1e-6,
                            sequence.start,
                            &cosine,
                            &sine,
                        )
                        .unwrap(),
                );
            }
            let sequence_refs = sequence_parts.iter().collect::<Vec<_>>();
            let qkv = context.concat_token_rows(&sequence_refs).unwrap();
            prepared_parts.push(context.full_attention_prepared_qkv_bf16(qkv, 14, head_dim, (head_dim as f32).sqrt().recip()).unwrap());
        }
        let mut prepared = prepared_parts.remove(0);
        for part in prepared_parts {
            prepared = context.concat_columns(&prepared, &part).unwrap();
        }
        let prepared = context.tensor_to_f32(&prepared).unwrap();
        assert_eq!(full, prepared, "source BF16 QKV preparation 必须与目标端 preparation 逐位一致");
    }
}
