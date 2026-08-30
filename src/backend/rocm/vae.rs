//! ROCm VAE/Diffusion capability；activation 全程保持 F32 device resident。

use crate::{
    backend::{BackendError, VaeBackend},
    kernel::rocm::hip,
    vae::{Conv1dSpec, Conv3dSpec, PixelShuffleSpec},
};

use super::{RocmContext, compute_error, device_tensor_bf16, device_tensor_f32, f32_tensor, resident_weight};

#[cfg(test)]
mod resident_tests {
    use super::*;
    use crate::backend::{BackendResources, DiffusionBackend, VaeBackend};

    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() < 1e-4, "index={index} actual={actual} expected={expected}");
        }
    }

    #[test]
    fn h3_video_vae_resident_ops_execute() {
        let Ok(context) = RocmContext::new(0) else { return };

        let input = context.tensor_from_f32(vec![1.0, 2.0, 3.0, 4.0], 1, 4).unwrap();
        let update = context.tensor_from_f32(vec![0.5, 1.0, 1.5, 2.0], 1, 4).unwrap();
        let scale = context.prepare_f32(&[2.0, 3.0, 4.0, 5.0], 1, 4).unwrap();
        let residual = context.scaled_residual(&input, &update, &scale).unwrap();
        close(&context.tensor_to_f32(&residual).unwrap(), &[2.0, 5.0, 9.0, 14.0]);

        let heads = context.rms_norm_heads_unit(&input, 2, 2, 0.0).unwrap();
        let root_2_5 = 2.5f32.sqrt();
        let root_12_5 = 12.5f32.sqrt();
        close(&context.tensor_to_f32(&heads).unwrap(), &[1.0 / root_2_5, 2.0 / root_2_5, 3.0 / root_12_5, 4.0 / root_12_5]);

        let norm_weight = context.prepare_f32(&[1.0, 1.0, 1.0, 1.0], 1, 4).unwrap();
        let norm_bias = context.prepare_f32(&[0.0, 0.0, 0.0, 0.0], 1, 4).unwrap();
        let normalized = context.layer_norm(&input, &norm_weight, &norm_bias, 0.0).unwrap();
        let root_1_25 = 1.25f32.sqrt();
        close(&context.tensor_to_f32(&normalized).unwrap(), &[-1.5 / root_1_25, -0.5 / root_1_25, 0.5 / root_1_25, 1.5 / root_1_25]);

        let patches = context.tensor_from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1, 8).unwrap();
        let latent_scale = context.prepare_f32(&[2.0, 3.0], 1, 2).unwrap();
        let latent_bias = context.prepare_f32(&[10.0, 20.0], 1, 2).unwrap();
        let voxels = context.unpatch_affine(&patches, &latent_scale, &latent_bias, [1, 2, 2], [1, 2, 2], 2).unwrap();
        close(&context.tensor_to_f32(&voxels).unwrap(), &[12.0, 35.0, 14.0, 38.0, 16.0, 41.0, 18.0, 44.0]);
    }

    #[test]
    fn h3_audio_vae_resident_ops_match_cpu() {
        let Ok(context) = RocmContext::new(0) else { return };

        let packed_values = vec![1.0, 2.0, 3.0, 4.0];
        let packed = context.tensor_from_f32(packed_values.clone(), 2, 2).unwrap();
        let scale_values = [2.0, 3.0];
        let bias_values = [10.0, 20.0];
        let scale = context.prepare_f32(&scale_values, 1, 2).unwrap();
        let bias = context.prepare_f32(&bias_values, 1, 2).unwrap();
        let unpacked = context.audio_unpack_affine(&packed, &scale, &bias, 1, 2, 2).unwrap();
        let expected = crate::kernel::cpu::vae::audio_unpack_affine(&packed_values, &scale_values, &bias_values, 1, 2, 2).unwrap();
        close(&context.tensor_to_f32(&unpacked).unwrap(), &expected);

        let input_values = vec![1.0, 2.0, 3.0];
        let input = context.tensor_from_f32(input_values.clone(), 1, 3).unwrap();
        let weight_g_values = [2.0];
        let weight_v_values = [1.0];
        let conv_bias_values = [0.5];
        let weight_g = context.prepare_f32(&weight_g_values, 1, 1).unwrap();
        let weight_v = context.prepare_f32(&weight_v_values, 1, 1).unwrap();
        let conv_bias = context.prepare_f32(&conv_bias_values, 1, 1).unwrap();
        let convolved = context.conv1d(&input, Some(&weight_g), &weight_v, Some(&conv_bias), &Conv1dSpec { batch: 1, input_channels: 1, output_channels: 1, kernel: 1, stride: 1, dilation: 1, padding: 0 }).unwrap();
        let expected = crate::kernel::cpu::vae::conv1d(&input_values, Some(&weight_g_values), &weight_v_values, Some(&conv_bias_values), 1, 1, 1, 3, 1, 1, 1, 0).unwrap();
        close(&context.tensor_to_f32(&convolved).unwrap(), &expected);

        let transpose_input_values = vec![1.0, 2.0];
        let transpose_input = context.tensor_from_f32(transpose_input_values.clone(), 1, 2).unwrap();
        let transpose_g_values = [2.0_f32.sqrt()];
        let transpose_v_values = [1.0, 1.0];
        let transpose_bias_values = [0.25];
        let transpose_g = context.prepare_f32(&transpose_g_values, 1, 1).unwrap();
        let transpose_v = context.prepare_f32(&transpose_v_values, 1, 2).unwrap();
        let transpose_bias = context.prepare_f32(&transpose_bias_values, 1, 1).unwrap();
        let transposed = context.conv_transpose1d(&transpose_input, &transpose_g, &transpose_v, &transpose_bias, &Conv1dSpec { batch: 1, input_channels: 1, output_channels: 1, kernel: 2, stride: 2, dilation: 1, padding: 0 }).unwrap();
        let expected = crate::kernel::cpu::vae::conv_transpose1d(&transpose_input_values, &transpose_g_values, &transpose_v_values, &transpose_bias_values, 1, 1, 1, 2, 2, 2, 0).unwrap();
        close(&context.tensor_to_f32(&transposed).unwrap(), &expected);

        let snake_values = vec![0.1, -0.2, 0.3];
        let snake_input = context.tensor_from_f32(snake_values.clone(), 1, 3).unwrap();
        let alpha_values = [0.0];
        let beta_values = [0.0];
        let filter_values = [0.0, 0.5, 0.5, 0.0];
        let alpha = context.prepare_f32(&alpha_values, 1, 1).unwrap();
        let beta = context.prepare_f32(&beta_values, 1, 1).unwrap();
        let up_filter = context.prepare_f32(&filter_values, 1, 4).unwrap();
        let down_filter = context.prepare_f32(&filter_values, 1, 4).unwrap();
        let activated = context.snake_beta(&snake_input, &alpha, &beta, &up_filter, &down_filter, 1).unwrap();
        let expected = crate::kernel::cpu::vae::snake_beta(&snake_values, &alpha_values, &beta_values, &filter_values, &filter_values, 1, 3).unwrap();
        close(&context.tensor_to_f32(&activated).unwrap(), &expected);

        let scaled = context.scale_tensor(&snake_input, 2.0).unwrap();
        close(&context.tensor_to_f32(&scaled).unwrap(), &[0.2, -0.4, 0.6]);
        let squashed = context.tanh(&snake_input).unwrap();
        close(&context.tensor_to_f32(&squashed).unwrap(), &snake_values.iter().map(|value| value.tanh()).collect::<Vec<_>>());
    }

    #[test]
    fn batched_full_attention_dim64_matches_independent_samples() {
        let Ok(context) = RocmContext::new(0) else { return };
        let rows = 2_049;
        let heads = 1;
        let head_dim = 64;
        let sample_elements = rows * heads * head_dim;
        let values = |phase: f32| (0..sample_elements).map(|index| ((index as f32 * 0.017 + phase).sin() * 0.25).clamp(-1.0, 1.0)).collect::<Vec<_>>();
        let q_samples = [values(0.1), values(0.7)];
        let k_samples = [values(1.3), values(1.9)];
        let v_samples = [values(2.5), values(3.1)];
        let combined = |samples: &[Vec<f32>; 2]| samples.iter().flatten().copied().collect::<Vec<_>>();
        let q = context.tensor_from_f32(combined(&q_samples), rows * 2, heads * head_dim).unwrap();
        let k = context.tensor_from_f32(combined(&k_samples), rows * 2, heads * head_dim).unwrap();
        let v = context.tensor_from_f32(combined(&v_samples), rows * 2, heads * head_dim).unwrap();
        let actual = context.full_attention_batched(q, k, v, 2, rows, heads, head_dim, (head_dim as f32).sqrt().recip()).unwrap();
        let actual = context.tensor_to_f32(&actual).unwrap();

        for sample in 0..2 {
            let q = context.tensor_from_f32(q_samples[sample].clone(), rows, heads * head_dim).unwrap();
            let k = context.tensor_from_f32(k_samples[sample].clone(), rows, heads * head_dim).unwrap();
            let v = context.tensor_from_f32(v_samples[sample].clone(), rows, heads * head_dim).unwrap();
            let expected = context.full_attention(q, k, v, heads, head_dim, (head_dim as f32).sqrt().recip()).unwrap();
            let expected = context.tensor_to_f32(&expected).unwrap();
            let start = sample * sample_elements;
            let max_error = actual[start..start + sample_elements].iter().zip(expected).map(|(actual, expected)| (actual - expected).abs()).fold(0.0f32, f32::max);
            assert!(max_error < 2.0e-3, "sample={sample} max_error={max_error}");
        }
    }
}

impl VaeBackend for RocmContext {
    fn vae_tensor_from_f32(&self, values: Vec<f32>, rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        self.tensor_from_f32(values, rows, cols).map_err(compute_error)
    }

    fn vae_tensor_to_f32(&self, input: &Self::Tensor) -> Result<Vec<f32>, BackendError> {
        self.tensor_to_f32(input)
    }

    fn layer_norm(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        if weight.data().len() != input.cols || bias.data().len() != input.cols {
            return Err(compute_error(format!("ROCm VAE LayerNorm weight={}/{} bias={}/{}", weight.data().len(), input.cols, bias.data().len(), input.cols,)));
        }
        let input = f32_tensor(self, input)?;
        let output = hip::try_layernorm_bias_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE LayerNorm input 缺少 device buffer"))?,
            resident_weight(weight, "VAE LayerNorm weight")?,
            resident_weight(bias, "VAE LayerNorm bias")?,
            input.rows,
            input.cols,
            eps,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn rms_norm_heads_unit(&self, input: &Self::Tensor, heads: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        if input.cols != heads * head_dim {
            return Err(compute_error(format!("ROCm VAE head RMSNorm cols={}，期望 {}x{}", input.cols, heads, head_dim)));
        }
        let input = f32_tensor(self, input)?;
        let output =
            hip::try_rmsnorm_heads_unit_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE head RMSNorm input 缺少 device buffer"))?, input.rows, heads, head_dim, eps).map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn rms_norm_rope_pair_unit(&self, query: &Self::Tensor, key: &Self::Tensor, heads: usize, head_dim: usize, rotary_dim: usize, eps: f32, cosine: &[f32], sine: &[f32]) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        if query.rows != key.rows || query.cols != key.cols || query.cols != heads.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm VAE QK norm+RoPE 列数溢出"))? {
            return Err(compute_error("ROCm VAE QK norm+RoPE shape 不匹配"));
        }
        let query = f32_tensor(self, query)?;
        let key = f32_tensor(self, key)?;
        let (query_output, key_output) = hip::try_rmsnorm_rope_pair_unit_resident_f32(
            self.device_id,
            query.device.as_deref().ok_or_else(|| compute_error("ROCm VAE QK norm+RoPE query 缺少 device buffer"))?,
            key.device.as_deref().ok_or_else(|| compute_error("ROCm VAE QK norm+RoPE key 缺少 device buffer"))?,
            query.rows,
            heads,
            head_dim,
            rotary_dim,
            eps,
            cosine,
            sine,
        )
        .map_err(compute_error)?;
        Ok((device_tensor_f32(query_output, query.rows, query.cols), device_tensor_f32(key_output, key.rows, key.cols)))
    }

    fn scaled_residual(&self, input: &Self::Tensor, update: &Self::Tensor, scale: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        if input.rows != update.rows || input.cols != update.cols || scale.data().len() != input.cols {
            return Err(compute_error("ROCm VAE scaled residual shape 不匹配"));
        }
        let input = f32_tensor(self, input)?;
        let update = f32_tensor(self, update)?;
        let output = hip::try_scaled_residual_columns_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE scaled residual input 缺少 device buffer"))?,
            update.device.as_deref().ok_or_else(|| compute_error("ROCm VAE scaled residual update 缺少 device buffer"))?,
            None,
            resident_weight(scale, "VAE scaled residual scale")?,
            input.rows.checked_mul(input.cols).ok_or_else(|| compute_error("ROCm VAE scaled residual 大小溢出"))?,
            input.cols,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn scaled_residual_bias(&self, input: &Self::Tensor, update: &Self::Tensor, bias: &Self::Weight, scale: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        if input.rows != update.rows || input.cols != update.cols || bias.data().len() != input.cols || scale.data().len() != input.cols {
            return Err(compute_error("ROCm VAE scaled residual+bias shape 不匹配"));
        }
        let input = f32_tensor(self, input)?;
        let update = f32_tensor(self, update)?;
        let output = hip::try_scaled_residual_columns_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE scaled residual+bias input 缺少 device buffer"))?,
            update.device.as_deref().ok_or_else(|| compute_error("ROCm VAE scaled residual+bias update 缺少 device buffer"))?,
            Some(resident_weight(bias, "VAE scaled residual bias")?),
            resident_weight(scale, "VAE scaled residual scale")?,
            input.rows.checked_mul(input.cols).ok_or_else(|| compute_error("ROCm VAE scaled residual+bias 大小溢出"))?,
            input.cols,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn split_gated_bias_activation(&self, input: Self::Tensor, bias: &Self::Weight, left_columns: usize, activation: &crate::moe::Activation) -> Result<Self::Tensor, BackendError> {
        let expected_columns = left_columns.checked_mul(2).ok_or_else(|| compute_error("ROCm VAE packed gated+bias 列数溢出"))?;
        if input.cols != expected_columns || bias.data().len() != expected_columns {
            return Err(compute_error(format!("ROCm VAE packed gated+bias cols={} bias={}，期望 {expected_columns}", input.cols, bias.data().len())));
        }
        let input_device = input.device.ok_or_else(|| compute_error("ROCm VAE packed gated+bias input 缺少 device buffer"))?;
        let input_device = std::sync::Arc::try_unwrap(input_device).map_err(|_| compute_error("ROCm VAE packed gated+bias 输入仍被共享，无法转移所有权"))?;
        let output = hip::try_split_gated_activation_owned_bf16(self.device_id, input_device, Some(resident_weight(bias, "VAE packed gated bias")?), input.rows, left_columns, activation).map_err(compute_error)?;
        Ok(device_tensor_bf16(output, input.rows, left_columns))
    }

    fn split_columns_bias(&self, input: &Self::Tensor, bias: &Self::Weight, left_columns: usize) -> Result<(Self::Tensor, Self::Tensor), BackendError> {
        if bias.data().len() != input.cols {
            return Err(compute_error(format!("ROCm VAE split+bias bias={}，期望 {}", bias.data().len(), input.cols)));
        }
        let input = f32_tensor(self, input)?;
        let right_columns = input.cols.checked_sub(left_columns).ok_or_else(|| compute_error(format!("ROCm VAE split+bias left={left_columns} 超过 cols={}", input.cols)))?;
        let (left, right) = hip::try_split_columns_bias_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE split+bias input 缺少 device buffer"))?,
            resident_weight(bias, "VAE split bias")?,
            input.rows,
            input.cols,
            left_columns,
        )
        .map_err(compute_error)?;
        Ok((device_tensor_f32(left, input.rows, left_columns), device_tensor_f32(right, input.rows, right_columns)))
    }

    fn split_three_columns_bias(&self, input: &Self::Tensor, bias: &Self::Weight, columns: usize) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), BackendError> {
        if input.cols != columns.checked_mul(3).ok_or_else(|| compute_error("ROCm VAE split3+bias 列数溢出"))? || bias.data().len() != input.cols {
            return Err(compute_error(format!("ROCm VAE split3+bias input={} bias={}，期望 3x{columns}", input.cols, bias.data().len())));
        }
        let input = f32_tensor(self, input)?;
        let (first, second, third) = hip::try_split_three_columns_bias_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE split3+bias input 缺少 device buffer"))?,
            resident_weight(bias, "VAE split3 bias")?,
            input.rows,
            input.cols,
            columns,
        )
        .map_err(compute_error)?;
        Ok((device_tensor_f32(first, input.rows, columns), device_tensor_f32(second, input.rows, columns), device_tensor_f32(third, input.rows, columns)))
    }

    fn unpatch_affine(&self, input: &Self::Tensor, scale: &Self::Weight, bias: &Self::Weight, shape: [usize; 3], patch: [usize; 3], channels: usize) -> Result<Self::Tensor, BackendError> {
        let input = f32_tensor(self, input)?;
        let output = hip::try_unpatch_affine_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE unpatch input 缺少 device buffer"))?,
            resident_weight(scale, "VAE latent scale")?,
            resident_weight(bias, "VAE latent bias")?,
            shape,
            patch,
            channels,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, shape.into_iter().product(), channels))
    }

    fn audio_unpack_affine(&self, input: &Self::Tensor, scale: &Self::Weight, bias: &Self::Weight, batch: usize, time: usize, channels: usize) -> Result<Self::Tensor, BackendError> {
        if input.rows != batch.checked_mul(time).ok_or_else(|| compute_error("ROCm audio unpack rows 溢出"))? || input.cols != channels {
            return Err(compute_error(format!("ROCm audio unpack input=[{},{}]，期望 [{},{}]", input.rows, input.cols, batch * time, channels)));
        }
        let input = f32_tensor(self, input)?;
        let output = hip::try_audio_unpack_affine_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm audio unpack input 缺少 device buffer"))?,
            resident_weight(scale, "audio latent scale")?,
            resident_weight(bias, "audio latent bias")?,
            batch,
            time,
            channels,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, batch * channels, time))
    }

    fn conv1d(&self, input: &Self::Tensor, weight_g: Option<&Self::Weight>, weight_v: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        let Conv1dSpec { batch, input_channels, output_channels, kernel, dilation, padding, .. } = *spec;
        if input.rows != batch.checked_mul(input_channels).ok_or_else(|| compute_error("ROCm audio Conv1D rows 溢出"))? {
            return Err(compute_error(format!("ROCm audio Conv1D input rows={}，期望 {}x{}", input.rows, batch, input_channels)));
        }
        let input = f32_tensor(self, input)?;
        let (output, output_length) = hip::try_audio_conv1d_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm audio Conv1D input 缺少 device buffer"))?,
            weight_g.map(|weight| resident_weight(weight, "audio Conv1D weight_g")).transpose()?,
            resident_weight(weight_v, "audio Conv1D weight_v")?,
            bias.map(|weight| resident_weight(weight, "audio Conv1D bias")).transpose()?,
            batch,
            input_channels,
            output_channels,
            input.cols,
            kernel,
            dilation,
            padding,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, batch * output_channels, output_length))
    }

    fn conv1d_strided(&self, input: &Self::Tensor, weight_g: Option<&Self::Weight>, weight_v: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        let Conv1dSpec { batch, input_channels, output_channels, kernel, stride, dilation, padding } = *spec;
        if input.rows != batch.checked_mul(input_channels).ok_or_else(|| compute_error("ROCm audio strided Conv1D rows 溢出"))? {
            return Err(compute_error(format!("ROCm audio strided Conv1D input rows={}，期望 {}x{}", input.rows, batch, input_channels)));
        }
        let input = f32_tensor(self, input)?;
        let (output, output_length) = hip::try_audio_conv1d_strided_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm audio strided Conv1D input 缺少 device buffer"))?,
            weight_g.map(|weight| resident_weight(weight, "audio strided Conv1D weight_g")).transpose()?,
            resident_weight(weight_v, "audio strided Conv1D weight_v")?,
            bias.map(|weight| resident_weight(weight, "audio strided Conv1D bias")).transpose()?,
            batch,
            input_channels,
            output_channels,
            input.cols,
            kernel,
            dilation,
            padding,
            stride,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, batch * output_channels, output_length))
    }

    fn snake(&self, input: &Self::Tensor, alpha: &Self::Weight, channels: usize) -> Result<Self::Tensor, BackendError> {
        if !input.rows.is_multiple_of(channels) || alpha.data().len() != channels {
            return Err(compute_error(format!("ROCm Snake input=[{},{}] alpha={}，channels={channels}", input.rows, input.cols, alpha.data().len())));
        }
        let input = f32_tensor(self, input)?;
        let output =
            hip::try_snake_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm Snake input 缺少 device buffer"))?, resident_weight(alpha, "Snake alpha")?, channels, input.cols).map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn vae_gelu(&self, input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        let input = f32_tensor(self, input)?;
        let elements = input.rows.checked_mul(input.cols).ok_or_else(|| compute_error("ROCm VAE GELU elements 溢出"))?;
        let output = hip::try_gelu_tanh_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE GELU input 缺少 device buffer"))?, elements).map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn channels_to_time(&self, input: &Self::Tensor, channels: usize) -> Result<Self::Tensor, BackendError> {
        if !input.rows.is_multiple_of(channels) {
            return Err(compute_error(format!("ROCm channels-to-time input=[{},{}]，channels={channels}", input.rows, input.cols)));
        }
        let batch = input.rows / channels;
        let input = f32_tensor(self, input)?;
        let output = hip::try_channels_to_time_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm channels-to-time input 缺少 device buffer"))?, batch, channels, input.cols).map_err(compute_error)?;
        Ok(device_tensor_f32(output, batch * input.cols, channels))
    }

    fn causal_attention(&self, query: &Self::Tensor, key: &Self::Tensor, value: &Self::Tensor, time: usize, heads: usize, head_dim: usize, score_scale: f32) -> Result<Self::Tensor, BackendError> {
        let batch = query.rows / time;
        let rows = batch.checked_mul(time).ok_or_else(|| compute_error("ROCm causal attention rows 溢出"))?;
        let cols = heads.checked_mul(head_dim).ok_or_else(|| compute_error("ROCm causal attention cols 溢出"))?;
        if query.rows != rows || key.rows != rows || value.rows != rows || query.cols != cols || key.cols != cols || value.cols != cols {
            return Err(compute_error(format!("ROCm causal attention Q/K/V shape 不兼容，期望 [{rows},{cols}]")));
        }
        let query = f32_tensor(self, query)?;
        let key = f32_tensor(self, key)?;
        let value = f32_tensor(self, value)?;
        let output = hip::try_causal_attention_resident_f32(
            self.device_id,
            query.device.as_deref().ok_or_else(|| compute_error("ROCm causal attention query 缺少 device buffer"))?,
            key.device.as_deref().ok_or_else(|| compute_error("ROCm causal attention key 缺少 device buffer"))?,
            value.device.as_deref().ok_or_else(|| compute_error("ROCm causal attention value 缺少 device buffer"))?,
            batch,
            time,
            heads,
            head_dim,
            score_scale,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, rows, cols))
    }

    fn encoder_conv3d(&self, input: &Self::Tensor, weight: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv3dSpec, spatial_pad_after: [usize; 2]) -> Result<Self::Tensor, BackendError> {
        let input_spatial = spec.input_shape.into_iter().product::<usize>();
        if input.rows != spec.input_channels || input.cols != input_spatial {
            return Err(compute_error(format!("ROCm encoder Conv3D input=[{},{}]，期望 [{},{}]", input.rows, input.cols, spec.input_channels, input_spatial)));
        }
        let mut output_shape = spec.output_shape().map_err(compute_error)?;
        for axis in 1..3 {
            let padded = spec.input_shape[axis]
                .checked_add(spec.padding[axis].checked_mul(2).ok_or_else(|| compute_error("ROCm encoder Conv3D padding 溢出"))?)
                .and_then(|value| value.checked_add(spatial_pad_after[axis - 1]))
                .ok_or_else(|| compute_error("ROCm encoder Conv3D padded shape 溢出"))?;
            output_shape[axis] = padded.checked_sub(spec.kernel[axis]).ok_or_else(|| compute_error("ROCm encoder Conv3D kernel 超过输入"))? / spec.stride[axis] + 1;
        }
        let input = f32_tensor(self, input)?;
        let output = hip::try_encoder_conv3d_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm encoder Conv3D input 缺少 device buffer"))?,
            resident_weight(weight, "encoder Conv3D weight")?,
            bias.map(|weight| resident_weight(weight, "encoder Conv3D bias")).transpose()?,
            spec.input_channels,
            spec.output_channels,
            spec.input_shape,
            spec.kernel,
            spec.stride,
            spec.padding,
            spatial_pad_after,
            spec.causal,
            true,
            output_shape,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, spec.output_channels, output_shape.into_iter().product()))
    }

    fn group_norm_time_isolated(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, time: usize, num_groups: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        let channels = weight.data().len();
        if channels == 0 || input.rows != channels || bias.data().len() != channels || time == 0 || !input.cols.is_multiple_of(time) {
            return Err(compute_error(format!("ROCm time-isolated GroupNorm input=[{},{}] weight={} bias={} time={time}", input.rows, input.cols, channels, bias.data().len())));
        }
        let spatial = input.cols / time;
        let input = f32_tensor(self, input)?;
        let output = hip::try_group_norm_time_isolated_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm time-isolated GroupNorm input 缺少 device buffer"))?,
            resident_weight(weight, "time-isolated GroupNorm weight")?,
            resident_weight(bias, "time-isolated GroupNorm bias")?,
            channels,
            time,
            spatial,
            num_groups,
            eps,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn conv_transpose1d(&self, input: &Self::Tensor, weight_g: &Self::Weight, weight_v: &Self::Weight, bias: &Self::Weight, spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        let Conv1dSpec { batch, input_channels, output_channels, kernel, stride, padding, .. } = *spec;
        if input.rows != batch.checked_mul(input_channels).ok_or_else(|| compute_error("ROCm audio ConvTranspose1D rows 溢出"))? {
            return Err(compute_error(format!("ROCm audio ConvTranspose1D input rows={}，期望 {}x{}", input.rows, batch, input_channels)));
        }
        let input = f32_tensor(self, input)?;
        let (output, output_length) = hip::try_audio_conv_transpose1d_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm audio ConvTranspose1D input 缺少 device buffer"))?,
            resident_weight(weight_g, "audio ConvTranspose1D weight_g")?,
            resident_weight(weight_v, "audio ConvTranspose1D weight_v")?,
            resident_weight(bias, "audio ConvTranspose1D bias")?,
            batch,
            input_channels,
            output_channels,
            input.cols,
            kernel,
            stride,
            padding,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, batch * output_channels, output_length))
    }

    fn snake_beta(&self, input: &Self::Tensor, alpha: &Self::Weight, beta: &Self::Weight, up_filter: &Self::Weight, down_filter: &Self::Weight, channels: usize) -> Result<Self::Tensor, BackendError> {
        if !input.rows.is_multiple_of(channels) || up_filter.data().len() != down_filter.data().len() {
            return Err(compute_error("ROCm audio SnakeBeta shape 不匹配"));
        }
        let batch = input.rows / channels;
        let input = f32_tensor(self, input)?;
        let output = hip::try_audio_snake_beta_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm audio SnakeBeta input 缺少 device buffer"))?,
            resident_weight(alpha, "audio SnakeBeta alpha")?,
            resident_weight(beta, "audio SnakeBeta beta")?,
            resident_weight(up_filter, "audio SnakeBeta up filter")?,
            resident_weight(down_filter, "audio SnakeBeta down filter")?,
            batch,
            channels,
            input.cols,
            up_filter.data().len(),
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn scale_tensor(&self, input: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError> {
        let input = f32_tensor(self, input)?;
        let elements = input.rows.checked_mul(input.cols).ok_or_else(|| compute_error("ROCm audio scale 大小溢出"))?;
        let output = hip::try_scale_tensor_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm audio scale input 缺少 device buffer"))?, elements, scale).map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn tanh(&self, input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        let input = f32_tensor(self, input)?;
        let elements = input.rows.checked_mul(input.cols).ok_or_else(|| compute_error("ROCm audio tanh 大小溢出"))?;
        let output = hip::try_tanh_tensor_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm audio tanh input 缺少 device buffer"))?, elements).map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn take_rows(&self, input: &Self::Tensor, rows: usize) -> Result<Self::Tensor, BackendError> {
        if rows == 0 || rows > input.rows {
            return Err(compute_error(format!("ROCm VAE take rows={rows} 超过 {}", input.rows)));
        }
        let input = f32_tensor(self, input)?;
        let output = hip::try_prefix_rows_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE take rows input 缺少 device buffer"))?, rows, input.cols).map_err(compute_error)?;
        Ok(device_tensor_f32(output, rows, input.cols))
    }

    fn take_rows_batched(&self, input: &Self::Tensor, rows: usize, batch: usize) -> Result<Self::Tensor, BackendError> {
        if batch == 0 || rows == 0 || input.rows % batch != 0 || rows > input.rows / batch {
            return Err(compute_error(format!("ROCm batched VAE take rows={rows} batch={batch} 与 input rows={} 不兼容", input.rows)));
        }
        let input = f32_tensor(self, input)?;
        let source_rows = input.rows / batch;
        let output = hip::try_take_rows_batched_resident_f32(self.device_id, input.device.as_deref().ok_or_else(|| compute_error("ROCm batched VAE take rows input 缺少 device buffer"))?, batch, source_rows, rows, input.cols)
            .map_err(compute_error)?;
        Ok(device_tensor_f32(output, batch * rows, input.cols))
    }

    fn concat_weight_rows(&self, input: &Self::Tensor, weight: &Self::Weight, rows: usize) -> Result<Self::Tensor, BackendError> {
        if rows == 0 || weight.data().len() != rows * input.cols {
            return Err(compute_error(format!("ROCm VAE weight concat weight={}，期望 {}x{}", weight.data().len(), rows, input.cols,)));
        }
        let input = f32_tensor(self, input)?;
        let output = hip::try_concat_rows_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm VAE weight concat input 缺少 device buffer"))?,
            input.rows.checked_mul(input.cols).ok_or_else(|| compute_error("ROCm VAE weight concat input 溢出"))?,
            resident_weight(weight, "VAE weight concat")?,
            rows.checked_mul(input.cols).ok_or_else(|| compute_error("ROCm VAE weight concat weight 溢出"))?,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows + rows, input.cols))
    }

    fn concat_weight_rows_batched(&self, input: &Self::Tensor, weight: &Self::Weight, rows: usize, batch: usize) -> Result<Self::Tensor, BackendError> {
        if batch == 0 || input.rows % batch != 0 || rows == 0 || weight.data().len() != rows * input.cols {
            return Err(compute_error(format!("ROCm batched VAE weight concat input_rows={} batch={batch} weight={}，期望 {}x{}", input.rows, weight.data().len(), rows, input.cols)));
        }
        let input = f32_tensor(self, input)?;
        let input_rows = input.rows / batch;
        let output = hip::try_concat_weight_rows_batched_resident_f32(
            self.device_id,
            input.device.as_deref().ok_or_else(|| compute_error("ROCm batched VAE weight concat input 缺少 device buffer"))?,
            resident_weight(weight, "batched VAE weight concat")?,
            batch,
            input_rows,
            rows,
            input.cols,
        )
        .map_err(compute_error)?;
        Ok(device_tensor_f32(output, batch * (input_rows + rows), input.cols))
    }

    fn conv3d(&self, input: &Self::Tensor, weight: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv3dSpec) -> Result<Self::Tensor, BackendError> {
        let input_spatial = spec.input_spatial().map_err(compute_error)?;
        let output_shape = spec.output_shape().map_err(compute_error)?;
        let output_spatial = output_shape.into_iter().product::<usize>();
        let kernel_elements = spec.kernel.into_iter().product::<usize>();
        if input.rows != spec.input_channels || input.cols != input_spatial {
            return Err(compute_error(format!("ROCm Conv3D input=[{},{}]，期望 [{},{}]", input.rows, input.cols, spec.input_channels, input_spatial)));
        }
        if weight.rows != spec.output_channels || weight.cols != spec.input_channels * kernel_elements {
            return Err(compute_error(format!("ROCm Conv3D weight=[{},{}] 与 {spec:?} 不兼容", weight.rows, weight.cols)));
        }
        if bias.is_some_and(|bias| bias.data().len() != spec.output_channels) {
            return Err(compute_error("ROCm Conv3D bias 长度与输出通道不一致"));
        }
        let input = f32_tensor(self, input)?;
        let input = input.device.as_deref().ok_or_else(|| compute_error("ROCm Conv3D input 缺少 device buffer"))?;
        let weight = resident_weight(weight, "Conv3D weight")?;
        let bias = bias.map(|bias| resident_weight(bias, "Conv3D bias")).transpose()?;
        let output =
            hip::try_conv3d_resident_f32(self.device_id, input, weight, bias, spec.input_channels, spec.output_channels, spec.input_shape, spec.kernel, spec.stride, spec.padding, spec.causal, output_shape).map_err(compute_error)?;
        Ok(device_tensor_f32(output, spec.output_channels, output_spatial))
    }

    fn group_norm(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, num_groups: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        if input.rows == 0 || input.cols == 0 || weight.data().len() != input.rows || bias.data().len() != input.rows {
            return Err(compute_error("ROCm VAE GroupNorm shape 不兼容"));
        }
        let input = f32_tensor(self, input)?;
        let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm GroupNorm input 缺少 device buffer"))?;
        let output = hip::try_group_norm_resident_f32(self.device_id, input_device, resident_weight(weight, "GroupNorm scale")?, resident_weight(bias, "GroupNorm bias")?, input.rows, input.cols, num_groups, eps).map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }

    fn pixel_shuffle(&self, input: &Self::Tensor, spec: &PixelShuffleSpec) -> Result<Self::Tensor, BackendError> {
        spec.validate().map_err(compute_error)?;
        let input_channels = spec.channels * spec.upscale * spec.upscale;
        if input.rows != input_channels || input.cols != spec.height * spec.width {
            return Err(compute_error(format!("ROCm pixel shuffle input=[{},{}]，期望 [{input_channels},{}]", input.rows, input.cols, spec.height * spec.width)));
        }
        let input = f32_tensor(self, input)?;
        let input_device = input.device.as_deref().ok_or_else(|| compute_error("ROCm pixel shuffle input 缺少 device buffer"))?;
        let output = hip::try_pixel_shuffle_resident_f32(self.device_id, input_device, spec.channels, spec.height, spec.width, spec.upscale).map_err(compute_error)?;
        Ok(device_tensor_f32(output, spec.channels, spec.height * spec.upscale * spec.width * spec.upscale))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::{Backend, BackendResources, DiffusionBackend, VaeBackend},
        diffusion::ModulationSegment,
        kernel::cpu,
        vae::{Conv3dSpec, PixelShuffleSpec},
    };

    use super::RocmContext;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() <= tolerance + tolerance * expected.abs(), "index {index}: actual={actual} expected={expected}");
        }
    }

    #[test]
    fn resident_vae_and_diffusion_match_cpu_oracle() {
        let context = RocmContext::new(0).expect("测试需要可用 ROCm device 0");

        let input = context.tensor_from_f32(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let shift = context.tensor_from_f32(vec![0.5, -0.5], 1, 2).unwrap();
        let scale = context.tensor_from_f32(vec![1.0, 0.0], 1, 2).unwrap();
        let output = context.adaln_modulate(&input, &shift, &scale).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &[2.5, 1.5, 6.5, 3.5], 1e-6);

        let input = context.tensor_from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4, 2).unwrap();
        let shift = context.tensor_from_f32(vec![0.5, -0.5, -1.0, 1.0], 2, 2).unwrap();
        let scale = context.tensor_from_f32(vec![1.0, 0.0, 0.0, 0.5], 2, 2).unwrap();
        let segments = [ModulationSegment { rows: 0..2, modulation_row: 1 }, ModulationSegment { rows: 2..4, modulation_row: 0 }];
        let output = context.adaln_modulate_segmented(&input, &shift, &scale, &segments).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &[0.0, 4.0, 2.0, 7.0, 10.5, 5.5, 14.5, 7.5], 1e-6);
        let norm = context.prepare_f32(&[0.75, 1.25], 1, 2).unwrap();
        let normalized = context.rmsnorm(&input, &norm, 1e-6).unwrap();
        let expected = context.adaln_modulate_segmented(&normalized, &shift, &scale, &segments).unwrap();
        let fused = context.rmsnorm_adaln_modulate_segmented(&input, &norm, 1e-6, &shift, &scale, &segments).unwrap();
        assert_close(&context.tensor_to_f32(&fused).unwrap(), &context.tensor_to_f32(&expected).unwrap(), 1e-5);
        let residual = context.tensor_from_f32(vec![1.0; 8], 4, 2).unwrap();
        let update = context.tensor_from_f32(vec![2.0; 8], 4, 2).unwrap();
        let gate = context.tensor_from_f32(vec![0.5, 1.0, 1.0, -1.0], 2, 2).unwrap();
        let output = context.gated_residual_segmented(&residual, &update, &gate, &segments).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &[3.0, -1.0, 3.0, -1.0, 2.0, 3.0, 2.0, 3.0], 1e-6);

        let output = context.timestep_embedding(&[0.0, 1.0], 4).unwrap();
        let expected = cpu::vae::timestep_embedding(&[0.0, 1.0], 4).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &expected, 1e-6);

        let input_values = vec![3.0, 4.0, 0.0, 5.0];
        let input = context.tensor_from_f32(input_values.clone(), 1, 4).unwrap();
        let weight = context.prepare_f32(&[1.0, 2.0], 1, 2).unwrap();
        let output = context.rmsnorm_heads(&input, &weight, 2, 2, 1e-6).unwrap();
        let expected = cpu::vae::rmsnorm_heads(&input_values, &[1.0, 2.0], 1, 2, 1e-6).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &expected, 1e-6);

        let mut query_values = vec![0.0; 3 * 16];
        query_values[0] = 1.0;
        query_values[16 + 1] = 1.0;
        query_values[32] = 1.0;
        query_values[32 + 1] = 1.0;
        let key_values = query_values.clone();
        let mut value_values = vec![0.0; 3 * 16];
        value_values[0] = 10.0;
        value_values[16 + 1] = 20.0;
        value_values[32] = 5.0;
        value_values[32 + 1] = 5.0;
        let query = context.tensor_from_f32(query_values.clone(), 3, 16).unwrap();
        let key = context.tensor_from_f32(key_values.clone(), 3, 16).unwrap();
        let value = context.tensor_from_f32(value_values.clone(), 3, 16).unwrap();
        let output = context.full_attention(query, key, value, 1, 16, 16.0_f32.sqrt().recip()).unwrap();
        let expected = cpu::vae::full_attention(&query_values, &key_values, &value_values, 1, 16, 16.0_f32.sqrt().recip()).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &expected, 1e-2);

        let input_values = vec![-1.0, 0.0, 1.0, 2.0];
        let input = context.tensor_from_f32(input_values.clone(), 2, 2).unwrap();
        let output = context.silu(&input).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &cpu::vae::silu(&input_values), 1e-6);
        let bias = context.prepare_f32(&[0.5, -0.5], 1, 2).unwrap();
        let output = context.add_row_bias(&input, &bias).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &[-0.5, -0.5, 1.5, 1.5], 1e-6);
        let right = context.tensor_from_f32(vec![3.0, 4.0], 1, 2).unwrap();
        let output = context.concat_rows(&input, &right).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], 1e-6);
        let velocity = context.tensor_from_f32(vec![2.0, -2.0, 4.0, -4.0], 2, 2).unwrap();
        let output = context.flow_step(&input, &velocity, 0.25).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &[-0.5, -0.5, 2.0, 1.0], 1e-6);
        let modulation_values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let modulation = context.tensor_from_f32(modulation_values, 1, 12).unwrap();
        let output = context.modulation_chunks(&modulation, 2, 3, 2).unwrap();
        let expected = [vec![0.0, 1.0, 6.0, 7.0], vec![2.0, 3.0, 8.0, 9.0], vec![4.0, 5.0, 10.0, 11.0]];
        for (output, expected) in output.iter().zip(expected) {
            assert_close(&context.tensor_to_f32(output).unwrap(), &expected, 1e-6);
        }

        let input_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = context.tensor_from_f32(input_values.clone(), 2, 4).unwrap();
        let norm_weight = context.prepare_f32(&[1.0, 1.0], 2, 1).unwrap();
        let norm_bias = context.prepare_f32(&[0.0, 0.0], 2, 1).unwrap();
        let output = context.group_norm(&input, &norm_weight, &norm_bias, 1, 1e-6).unwrap();
        let expected = cpu::vae::group_norm(&input_values, 2, 4, 1, 1e-6, &[1.0, 1.0], &[0.0, 0.0]);
        assert_close(&context.tensor_to_f32(&output).unwrap(), &expected, 1e-5);

        let input = context.tensor_from_f32(vec![1.0, 2.0], 1, 2).unwrap();
        let conv_weight = context.prepare_f32(&[0.0, 0.0, 1.0], 1, 3).unwrap();
        let conv = Conv3dSpec { input_channels: 1, output_channels: 1, input_shape: [2, 1, 1], kernel: [3, 1, 1], stride: [1, 1, 1], padding: [2, 0, 0], causal: true };
        let output = context.conv3d(&input, &conv_weight, None, &conv).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &[1.0, 2.0], 1e-6);

        let input = context.tensor_from_f32(vec![1.0, 2.0, 3.0, 4.0], 4, 1).unwrap();
        let shuffle = PixelShuffleSpec { channels: 1, height: 1, width: 1, upscale: 2 };
        let output = context.pixel_shuffle(&input, &shuffle).unwrap();
        assert_close(&context.tensor_to_f32(&output).unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }
}
