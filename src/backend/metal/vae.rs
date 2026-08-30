//! Metal VAE/Diffusion 算子能力实现。

use crate::{
    backend::{BackendError, DiffusionBackend, VaeBackend},
    kernel::metal as ops,
    vae::{Conv1dSpec, Conv3dSpec, PixelShuffleSpec},
};

use super::{context::MetalContext, expect_resident_f16 as expect_f16_weight, expect_resident_f16_opt as expect_f16_weight_opt};

impl VaeBackend for MetalContext {
    fn conv3d(&self, input: &Self::Tensor, weight: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv3dSpec) -> Result<Self::Tensor, BackendError> {
        let weight = expect_f16_weight(weight, "VAE Conv3D 需要 F16/BF16 dense 权重")?;
        let bias = expect_f16_weight_opt(bias, "VAE Conv3D bias 需要 F16/BF16 dense 权重")?;
        ops::vae::conv3d_tensor(self, input, weight, bias, spec).map_err(|msg| BackendError::Compute { msg })
    }

    fn group_norm(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, num_groups: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        let channels = input.rows;
        let spatial = input.cols;
        let w_tensor = expect_f16_weight(weight, "VAE group_norm 需要 F16 权重")?;
        let b_tensor = expect_f16_weight(bias, "VAE group_norm 需要 F16 权重")?;
        ops::vae::vae_group_norm_tensor(self, input, w_tensor, b_tensor, channels, spatial, num_groups, eps).map_err(|msg| BackendError::Compute { msg })
    }

    fn pixel_shuffle(&self, input: &Self::Tensor, spec: &PixelShuffleSpec) -> Result<Self::Tensor, BackendError> {
        ops::vae::pixel_shuffle_tensor(self, input, spec).map_err(|msg| BackendError::Compute { msg })
    }

    fn audio_unpack_affine(&self, input: &Self::Tensor, scale: &Self::Weight, bias: &Self::Weight, batch: usize, time: usize, channels: usize) -> Result<Self::Tensor, BackendError> {
        let scale_v = expect_f16_weight(scale, "audio_unpack_affine 需要 F16 scale/bias")?;
        let bias_v = expect_f16_weight(bias, "audio_unpack_affine 需要 F16 scale/bias")?;
        ops::vae::audio_unpack_affine_f16_tensor(self, input, scale_v, bias_v, batch, time, channels).map_err(|msg| BackendError::Compute { msg })
    }

    fn conv1d(&self, input: &Self::Tensor, weight_g: Option<&Self::Weight>, weight_v: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        let Conv1dSpec { batch, input_channels, output_channels, kernel, dilation, padding, .. } = *spec;
        let weight_v = expect_f16_weight(weight_v, "VAE conv1d weight_v 需要 F16")?;
        let weight_g = expect_f16_weight_opt(weight_g, "VAE conv1d weight_g 需要 F16")?;
        let bias = expect_f16_weight_opt(bias, "VAE conv1d bias 需要 F16")?;
        let columns = input_channels * kernel;
        let normalized = ops::vae::weight_norm_f16_tensor(self, weight_g, weight_v, output_channels, columns).map_err(|msg| BackendError::Compute { msg })?;
        ops::vae::conv1d_f16_tensor(self, input, &normalized, bias, batch, input_channels, output_channels, kernel, dilation, padding).map_err(|msg| BackendError::Compute { msg })
    }

    fn conv_transpose1d(&self, input: &Self::Tensor, weight_g: &Self::Weight, weight_v: &Self::Weight, bias: &Self::Weight, spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        let Conv1dSpec { batch, input_channels, output_channels, kernel, stride, padding, .. } = *spec;
        let weight_v = expect_f16_weight(weight_v, "VAE conv_transpose1d weight_v 需要 F16")?;
        let weight_g = expect_f16_weight(weight_g, "VAE conv_transpose1d weight_g 需要 F16")?;
        let bias = expect_f16_weight(bias, "VAE conv_transpose1d bias 需要 F16")?;
        // conv_transpose1d weight 布局 [input_channels, output_channels * kernel]
        let normalized = ops::vae::weight_norm_f16_tensor(self, Some(weight_g), weight_v, input_channels, output_channels * kernel).map_err(|msg| BackendError::Compute { msg })?;
        ops::vae::conv_transpose1d_f16_tensor(self, input, &normalized, bias, batch, input_channels, output_channels, kernel, stride, padding).map_err(|msg| BackendError::Compute { msg })
    }

    fn snake_beta(&self, input: &Self::Tensor, alpha: &Self::Weight, beta: &Self::Weight, up_filter: &Self::Weight, down_filter: &Self::Weight, channels: usize) -> Result<Self::Tensor, BackendError> {
        let alpha_v = expect_f16_weight(alpha, "snake_beta 需要 F16 alpha/beta/up_filter/down_filter")?;
        let beta_v = expect_f16_weight(beta, "snake_beta 需要 F16 alpha/beta/up_filter/down_filter")?;
        let up_v = expect_f16_weight(up_filter, "snake_beta 需要 F16 alpha/beta/up_filter/down_filter")?;
        let down_v = expect_f16_weight(down_filter, "snake_beta 需要 F16 alpha/beta/up_filter/down_filter")?;
        // snake_beta = up_filter 上采样 2x + down_filter 下采样 2x → 输出同 length(对照 CPU kernel/cpu/vae.rs:987)。
        // up 输出 [batch*channels, 2*input_length],然后 down 把 2x 折叠回 1x。batch = input.rows / channels。
        let upsampled = ops::vae::snake_beta_upsample_f16_tensor(self, input, alpha_v, beta_v, up_v, down_v, channels).map_err(|msg| BackendError::Compute { msg })?;
        let output = ops::vae::snake_beta_downsample_f16_tensor(self, &upsampled, alpha_v, beta_v, up_v, down_v, channels).map_err(|msg| BackendError::Compute { msg })?;
        Ok(output)
    }

    fn scale_tensor(&self, input: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError> {
        ops::vae::scale_tensor_f16_tensor(self, input, scale).map_err(|msg| BackendError::Compute { msg })
    }

    fn tanh(&self, input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        ops::vae::vae_tanh_f16_tensor(self, input).map_err(|msg| BackendError::Compute { msg })
    }
}

impl DiffusionBackend for MetalContext {
    fn adaln_modulate(&self, input: &Self::Tensor, shift: &Self::Tensor, scale: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        ops::vae::vae_adaln_modulate_tensor(self, input, shift, scale).map_err(|msg| BackendError::Compute { msg })
    }

    fn timestep_embedding(&self, timesteps: &[f32], dim: usize) -> Result<Self::Tensor, BackendError> {
        ops::vae::vae_timestep_embedding_tensor(self, timesteps, dim).map_err(|msg| BackendError::Compute { msg })
    }

    fn diffusion_tensor_from_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        // 与 tensor_from_f32 同语义:上传 F32 MetalTensor。下游 linear 内部 to_f16 转换
        // (tensor.rs:105),diffusion 专用 kernel 接受 F32。语义对齐 ROCm vae.rs:736。
        self.tensor_from_f32(values, rows, cols).map_err(|msg| BackendError::Compute { msg })
    }

    fn silu(&self, input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        ops::vae::vae_silu_tensor(self, input).map_err(|msg| BackendError::Compute { msg })
    }

    fn add_row_bias(&self, input: &Self::Tensor, bias: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        let bias_tensor = expect_f16_weight(bias, "diffusion row bias 需要 F16 权重")?;
        ops::diffusion::add_row_bias_tensor(self, input, bias_tensor).map_err(|msg| BackendError::Compute { msg })
    }

    fn concat_rows(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        ops::diffusion::concat_rows_tensor(self, left, right).map_err(|msg| BackendError::Compute { msg })
    }

    fn flow_step(&self, sample: &Self::Tensor, velocity: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError> {
        ops::diffusion::flow_step_tensor(self, sample, velocity, scale).map_err(|msg| BackendError::Compute { msg })
    }

    fn modulation_chunks(&self, input: &Self::Tensor, modalities: usize, chunks: usize, hidden: usize) -> Result<Vec<Self::Tensor>, BackendError> {
        ops::diffusion::modulation_chunks_tensor(self, input, modalities, chunks, hidden).map_err(|msg| BackendError::Compute { msg })
    }

    fn rmsnorm_heads(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        if input.cols != head_count * head_dim {
            return Err(BackendError::Compute { msg: format!("逐头 RMSNorm 输入列 {} 与 {head_count}x{head_dim} 不符", input.cols) });
        }
        let weight_tensor = expect_f16_weight(weight, "逐头 RMSNorm 需要 F16 权重")?;
        // 把 [rows, head_count*head_dim] reshape 成 [rows*head_count, head_dim],
        // 每行(=每 head)走标准 RMSNorm,再 reshape 回去。语义与 CPU 一致。
        let heads = input.reshape(input.rows * head_count, head_dim);
        let output = ops::tensor::rmsnorm_tensor_resident_weight(self, &heads, weight_tensor, eps, 0.0).map_err(|msg| BackendError::Compute { msg })?;
        Ok(output.reshape(input.rows, input.cols))
    }

    fn full_attention(&self, query: Self::Tensor, key: Self::Tensor, value: Self::Tensor, head_count: usize, head_dim: usize, score_scale: f32) -> Result<Self::Tensor, BackendError> {
        ops::diffusion::full_attention_tensor(self, &query, &key, &value, head_count, head_dim, score_scale).map_err(|msg| BackendError::Compute { msg })
    }

    fn adaln_modulate_segmented(&self, input: &Self::Tensor, shift: &Self::Tensor, scale: &Self::Tensor, segments: &[crate::diffusion::ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        let row_map = crate::diffusion::modulation_row_map(segments, input.rows, shift.rows).map_err(|msg| BackendError::Compute { msg })?;
        ops::diffusion::adaln_modulate_segmented_tensor(self, input, shift, scale, &row_map).map_err(|msg| BackendError::Compute { msg })
    }

    fn gated_residual_segmented(&self, residual: &Self::Tensor, update: &Self::Tensor, gate: &Self::Tensor, segments: &[crate::diffusion::ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        let row_map = crate::diffusion::modulation_row_map(segments, residual.rows, gate.rows).map_err(|msg| BackendError::Compute { msg })?;
        ops::diffusion::gated_residual_segmented_tensor(self, residual, update, gate, &row_map).map_err(|msg| BackendError::Compute { msg })
    }
}
