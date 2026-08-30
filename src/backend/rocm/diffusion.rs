//! ROCm 扩散 Transformer capability；activation 全程保持 F32 device resident。

use std::sync::Arc;

use crate::{
    backend::{BackendError, DiffusionBackend},
    diffusion::{ModulationSegment, validate_modulation_segments},
    kernel::rocm::hip,
};

use super::{RocmContext, RocmTensor, compute_error, device_tensor_f32, f32_tensor, resident_weight};

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
        let query = std::sync::Arc::try_unwrap(query).map_err(|_| compute_error("ROCm full attention query device buffer 仍被共享"))?;
        let key = std::sync::Arc::try_unwrap(key).map_err(|_| compute_error("ROCm full attention key device buffer 仍被共享"))?;
        let value = std::sync::Arc::try_unwrap(value).map_err(|_| compute_error("ROCm full attention value device buffer 仍被共享"))?;
        let output = hip::try_full_attention_resident_f32(self.device_id, query, key, value, rows, head_count, head_dim, score_scale).map_err(compute_error)?;
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
    use crate::backend::{Backend, BackendResources, DiffusionBackend, LinearWeight};

    use super::RocmContext;

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
        let cosine = vec![1.0; rows * head_dim / 2];
        let sine = vec![0.0; rows * head_dim / 2];

        let full_qkv = context.tensor_from_f32(values.clone(), rows, heads * head_dim * 3).unwrap();
        let full = context.full_attention_qkv(full_qkv, &query_norm, &key_norm, heads, head_dim, head_dim, 1e-6, &cosine, &sine, (head_dim as f32).sqrt().recip()).unwrap();
        let full = context.tensor_to_f32(&full).unwrap();

        let source = context.tensor_from_f32(values, rows, heads * head_dim * 3).unwrap();
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
    }
}
