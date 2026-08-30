//! CPU VAE/Diffusion capability；作为 Metal/ROCm 的数值 oracle。

use crate::{
    backend::{BackendError, DiffusionBackend, VaeBackend, compute_error as compute},
    diffusion::{ModulationSegment, modulation_row_map},
    kernel::cpu::{self, CpuTensor},
    vae::{Conv1dSpec, Conv3dSpec, PixelShuffleSpec},
};

use super::{CpuContext, CpuWeight};

impl VaeBackend for CpuContext {
    fn vae_tensor_from_f32(&self, values: Vec<f32>, rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        let expected = rows.checked_mul(cols).ok_or_else(|| compute("CPU VAE tensor 大小溢出"))?;
        if values.len() != expected {
            return Err(compute(format!("CPU VAE tensor 元素 {}，期望 {expected}", values.len())));
        }
        Ok(CpuTensor { data: values, rows, cols })
    }

    fn vae_tensor_to_f32(&self, input: &Self::Tensor) -> Result<Vec<f32>, BackendError> {
        Ok(input.data.clone())
    }

    fn layer_norm(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::layer_norm(&input.data, weight.data(), bias.data(), input.cols, eps).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn rms_norm_heads_unit(&self, input: &Self::Tensor, heads: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        if input.cols != heads * head_dim {
            return Err(compute(format!("CPU VAE head RMSNorm cols={}，期望 {}x{}", input.cols, heads, head_dim)));
        }
        let data = cpu::vae::rms_norm_heads_unit(&input.data, heads, head_dim, eps).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn scaled_residual(&self, input: &Self::Tensor, update: &Self::Tensor, scale: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::scaled_residual(&input.data, &update.data, scale.data(), input.cols).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn unpatch_affine(&self, input: &Self::Tensor, scale: &Self::Weight, bias: &Self::Weight, shape: [usize; 3], patch: [usize; 3], channels: usize) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::unpatch_affine(&input.data, scale.data(), bias.data(), shape, patch, channels).map_err(compute)?;
        Ok(CpuTensor { data, rows: shape.into_iter().product(), cols: channels })
    }

    fn audio_unpack_affine(&self, input: &Self::Tensor, scale: &Self::Weight, bias: &Self::Weight, batch: usize, time: usize, channels: usize) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::audio_unpack_affine(&input.data, scale.data(), bias.data(), batch, time, channels).map_err(compute)?;
        Ok(CpuTensor { data, rows: batch * channels, cols: time })
    }

    fn conv1d(&self, input: &Self::Tensor, weight_g: Option<&Self::Weight>, weight_v: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        // stride=1 是 conv1d_strided 的特例,校验与执行统一走 strided 实现。
        self.conv1d_strided(input, weight_g, weight_v, bias, &Conv1dSpec { stride: 1, ..*spec })
    }

    fn conv1d_strided(&self, input: &Self::Tensor, weight_g: Option<&Self::Weight>, weight_v: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        let Conv1dSpec { batch, input_channels, output_channels, kernel, stride, dilation, padding } = *spec;
        if input.rows != batch * input_channels {
            return Err(compute(format!("CPU conv1d_strided input.rows={}，期望 batch({batch})*input_channels({input_channels})={}", input.rows, batch * input_channels)));
        }
        let effective = dilation * (kernel - 1) + 1;
        if input.cols + 2 * padding < effective {
            return Err(compute(format!("CPU conv1d_strided 感受野 {effective} 超过 input.cols({}) + 2*padding({padding})", input.cols)));
        }
        let output_length = (input.cols + 2 * padding - effective) / stride + 1;
        let data = cpu::vae::conv1d(&input.data, weight_g.map(CpuWeight::data), weight_v.data(), bias.map(CpuWeight::data), batch, input_channels, output_channels, input.cols, kernel, stride, dilation, padding).map_err(compute)?;
        Ok(CpuTensor { data, rows: batch * output_channels, cols: output_length })
    }

    fn snake(&self, input: &Self::Tensor, alpha: &Self::Weight, channels: usize) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::snake(&input.data, alpha.data(), channels, input.cols).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn channels_to_time(&self, input: &Self::Tensor, channels: usize) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::channels_to_time(&input.data, channels, input.cols).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: channels })
    }

    fn causal_attention(&self, query: &Self::Tensor, key: &Self::Tensor, value: &Self::Tensor, time: usize, heads: usize, head_dim: usize, score_scale: f32) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::causal_attention(&query.data, &key.data, &value.data, time, heads, head_dim, score_scale).map_err(compute)?;
        Ok(CpuTensor { data, rows: query.rows, cols: heads * head_dim })
    }

    fn conv_transpose1d(&self, input: &Self::Tensor, weight_g: &Self::Weight, weight_v: &Self::Weight, bias: &Self::Weight, spec: &Conv1dSpec) -> Result<Self::Tensor, BackendError> {
        let Conv1dSpec { batch, input_channels, output_channels, kernel, stride, padding, .. } = *spec;
        if input.rows != batch * input_channels {
            return Err(compute(format!("CPU conv_transpose1d input.rows={}，期望 batch({batch})*input_channels({input_channels})={}", input.rows, batch * input_channels)));
        }
        if input.cols == 0 || (input.cols - 1) * stride + kernel < 2 * padding {
            return Err(compute(format!("CPU conv_transpose1d 输出长度下溢: input.cols={} stride={stride} kernel={kernel} padding={padding}", input.cols)));
        }
        let output_length = (input.cols - 1) * stride + kernel - 2 * padding;
        let data = cpu::vae::conv_transpose1d(&input.data, weight_g.data(), weight_v.data(), bias.data(), batch, input_channels, output_channels, input.cols, kernel, stride, padding).map_err(compute)?;
        Ok(CpuTensor { data, rows: batch * output_channels, cols: output_length })
    }

    fn snake_beta(&self, input: &Self::Tensor, alpha: &Self::Weight, beta: &Self::Weight, up_filter: &Self::Weight, down_filter: &Self::Weight, channels: usize) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::snake_beta(&input.data, alpha.data(), beta.data(), up_filter.data(), down_filter.data(), channels, input.cols).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn scale_tensor(&self, input: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError> {
        Ok(CpuTensor { data: input.data.iter().map(|value| value * scale).collect(), rows: input.rows, cols: input.cols })
    }

    fn tanh(&self, input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(CpuTensor { data: input.data.iter().map(|value| value.tanh()).collect(), rows: input.rows, cols: input.cols })
    }

    fn take_rows(&self, input: &Self::Tensor, rows: usize) -> Result<Self::Tensor, BackendError> {
        if rows == 0 || rows > input.rows {
            return Err(compute(format!("CPU VAE take rows={rows} 超过 {}", input.rows)));
        }
        Ok(CpuTensor { data: input.data[..rows * input.cols].to_vec(), rows, cols: input.cols })
    }

    fn concat_weight_rows(&self, input: &Self::Tensor, weight: &Self::Weight, rows: usize) -> Result<Self::Tensor, BackendError> {
        if rows == 0 || weight.data().len() != rows * input.cols {
            return Err(compute(format!("CPU VAE weight concat weight={}，期望 {}x{}", weight.data().len(), rows, input.cols,)));
        }
        let mut data = Vec::with_capacity(input.data.len() + weight.data().len());
        data.extend_from_slice(&input.data);
        data.extend_from_slice(weight.data());
        Ok(CpuTensor { data, rows: input.rows + rows, cols: input.cols })
    }

    fn conv3d(&self, input: &Self::Tensor, weight: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv3dSpec) -> Result<Self::Tensor, BackendError> {
        let input_spatial = spec.input_spatial().map_err(compute)?;
        let output_spatial = spec.output_spatial().map_err(compute)?;
        let kernel_elements = spec.kernel.into_iter().product::<usize>();
        if input.rows != spec.input_channels || input.cols != input_spatial {
            return Err(compute(format!("CPU Conv3D input=[{},{}]，期望 [{},{}]", input.rows, input.cols, spec.input_channels, input_spatial)));
        }
        if weight.rows() != spec.output_channels || weight.cols() != spec.input_channels * kernel_elements {
            return Err(compute(format!("CPU Conv3D weight=[{},{}] 与 {spec:?} 不兼容", weight.rows(), weight.cols())));
        }
        if bias.is_some_and(|bias| bias.data().len() != spec.output_channels) {
            return Err(compute("CPU Conv3D bias 长度与输出通道不一致"));
        }
        let [depth, height, width] = spec.input_shape;
        let output = cpu::vae::conv3d(&input.data, spec.input_channels, depth, height, width, weight.data(), spec.output_channels, spec.kernel.into(), spec.stride.into(), spec.padding.into(), bias.map(CpuWeight::data), spec.causal);
        Ok(CpuTensor { data: output, rows: spec.output_channels, cols: output_spatial })
    }

    fn encoder_conv3d(&self, input: &Self::Tensor, weight: &Self::Weight, bias: Option<&Self::Weight>, spec: &Conv3dSpec, spatial_pad_after: [usize; 2]) -> Result<Self::Tensor, BackendError> {
        if spatial_pad_after == [0, 0] {
            return self.conv3d(input, weight, bias, spec);
        }
        let [depth, height, width] = spec.input_shape;
        let padded_height = height.checked_add(spatial_pad_after[0]).ok_or_else(|| compute("CPU VAE encoder padded height 溢出"))?;
        let padded_width = width.checked_add(spatial_pad_after[1]).ok_or_else(|| compute("CPU VAE encoder padded width 溢出"))?;
        let input_spatial = depth.checked_mul(height).and_then(|value| value.checked_mul(width)).ok_or_else(|| compute("CPU VAE encoder input shape 溢出"))?;
        if input.rows != spec.input_channels || input.cols != input_spatial {
            return Err(compute(format!("CPU encoder Conv3D input=[{},{}]，期望 [{},{}]", input.rows, input.cols, spec.input_channels, input_spatial)));
        }
        let padded_spatial = depth.checked_mul(padded_height).and_then(|value| value.checked_mul(padded_width)).ok_or_else(|| compute("CPU VAE encoder padded shape 溢出"))?;
        let mut data = vec![0.0; spec.input_channels.checked_mul(padded_spatial).ok_or_else(|| compute("CPU VAE encoder padded tensor 溢出"))?];
        for channel in 0..spec.input_channels {
            for time in 0..depth {
                for row in 0..height {
                    let source = channel * input_spatial + (time * height + row) * width;
                    let destination = channel * padded_spatial + (time * padded_height + row) * padded_width;
                    data[destination..destination + width].copy_from_slice(&input.data[source..source + width]);
                }
            }
        }
        let padded = CpuTensor { data, rows: spec.input_channels, cols: padded_spatial };
        let padded_spec =
            Conv3dSpec { input_channels: spec.input_channels, output_channels: spec.output_channels, input_shape: [depth, padded_height, padded_width], kernel: spec.kernel, stride: spec.stride, padding: spec.padding, causal: spec.causal };
        self.conv3d(&padded, weight, bias, &padded_spec)
    }

    fn group_norm(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, num_groups: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        if input.rows == 0 || input.cols == 0 || weight.data().len() != input.rows || bias.data().len() != input.rows {
            return Err(compute("CPU VAE GroupNorm shape 不兼容"));
        }
        let data = cpu::vae::group_norm(&input.data, input.rows, input.cols, num_groups, eps, weight.data(), bias.data());
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn group_norm_time_isolated(&self, input: &Self::Tensor, weight: &Self::Weight, bias: &Self::Weight, time: usize, num_groups: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        let channels = weight.data().len();
        if channels == 0 || input.rows != channels || bias.data().len() != channels || time == 0 || !input.cols.is_multiple_of(time) {
            return Err(compute(format!("CPU time-isolated GroupNorm input=[{},{}] weight={} bias={} time={time}", input.rows, input.cols, channels, bias.data().len())));
        }
        let spatial = input.cols / time;
        let mut frame = vec![0.0; channels * spatial];
        let mut data = vec![0.0; input.data.len()];
        for position in 0..time {
            for channel in 0..channels {
                let source = channel * input.cols + position * spatial;
                let destination = channel * spatial;
                frame[destination..destination + spatial].copy_from_slice(&input.data[source..source + spatial]);
            }
            let normalized = cpu::vae::group_norm(&frame, channels, spatial, num_groups, eps, weight.data(), bias.data());
            for channel in 0..channels {
                let source = channel * spatial;
                let destination = channel * input.cols + position * spatial;
                data[destination..destination + spatial].copy_from_slice(&normalized[source..source + spatial]);
            }
        }
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn pixel_shuffle(&self, input: &Self::Tensor, spec: &PixelShuffleSpec) -> Result<Self::Tensor, BackendError> {
        spec.validate().map_err(compute)?;
        let input_channels = spec.channels * spec.upscale * spec.upscale;
        if input.rows != input_channels || input.cols != spec.height * spec.width {
            return Err(compute(format!("CPU pixel shuffle input=[{},{}]，期望 [{input_channels},{}]", input.rows, input.cols, spec.height * spec.width)));
        }
        let data = cpu::vae::pixel_shuffle(&input.data, spec.channels, spec.height, spec.width, spec.upscale);
        Ok(CpuTensor { data, rows: spec.channels, cols: spec.height * spec.upscale * spec.width * spec.upscale })
    }
}

impl DiffusionBackend for CpuContext {
    fn silu(&self, input: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(CpuTensor { data: cpu::vae::silu(&input.data), rows: input.rows, cols: input.cols })
    }

    fn add_row_bias(&self, input: &Self::Tensor, bias: &Self::Weight) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::add_row_bias(&input.data, bias.data()).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn concat_rows(&self, left: &Self::Tensor, right: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        if left.cols != right.cols {
            return Err(compute(format!("CPU row concat cols={}/{} 不一致", left.cols, right.cols)));
        }
        let data = cpu::vae::concat_rows(&left.data, &right.data, left.cols).map_err(compute)?;
        Ok(CpuTensor { data, rows: left.rows + right.rows, cols: left.cols })
    }

    fn flow_step(&self, sample: &Self::Tensor, velocity: &Self::Tensor, scale: f32) -> Result<Self::Tensor, BackendError> {
        if sample.rows != velocity.rows || sample.cols != velocity.cols {
            return Err(compute(format!("CPU flow step shape=[{},{}]/[{},{}] 不一致", sample.rows, sample.cols, velocity.rows, velocity.cols,)));
        }
        let data = sample.data.iter().zip(&velocity.data).map(|(&sample, &velocity)| sample + scale * velocity).collect();
        Ok(CpuTensor { data, rows: sample.rows, cols: sample.cols })
    }

    fn modulation_chunks(&self, input: &Self::Tensor, modalities: usize, chunks: usize, hidden: usize) -> Result<Vec<Self::Tensor>, BackendError> {
        let data = cpu::vae::modulation_chunks(&input.data, modalities, chunks, hidden).map_err(compute)?;
        Ok(data.into_iter().map(|data| CpuTensor { data, rows: input.rows * modalities, cols: hidden }).collect())
    }

    fn rmsnorm_heads(&self, input: &Self::Tensor, weight: &Self::Weight, head_count: usize, head_dim: usize, eps: f32) -> Result<Self::Tensor, BackendError> {
        let data = cpu::vae::rmsnorm_heads(&input.data, weight.data(), head_count, head_dim, eps).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn full_attention(&self, query: Self::Tensor, key: Self::Tensor, value: Self::Tensor, head_count: usize, head_dim: usize, score_scale: f32) -> Result<Self::Tensor, BackendError> {
        if query.rows != key.rows || query.rows != value.rows || query.cols != key.cols || query.cols != value.cols {
            return Err(compute("CPU full attention Q/K/V shape 不一致"));
        }
        let data = cpu::vae::full_attention(&query.data, &key.data, &value.data, head_count, head_dim, score_scale).map_err(compute)?;
        Ok(CpuTensor { data, rows: query.rows, cols: query.cols })
    }

    fn adaln_modulate(&self, input: &Self::Tensor, shift: &Self::Tensor, scale: &Self::Tensor) -> Result<Self::Tensor, BackendError> {
        if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows || (shift.rows != 1 && shift.rows != input.rows) {
            return Err(compute(format!("CPU AdaLN shape 不兼容: input=[{},{}] shift=[{},{}] scale=[{},{}]", input.rows, input.cols, shift.rows, shift.cols, scale.rows, scale.cols)));
        }
        let data = cpu::vae::adaln_modulate(&input.data, &shift.data, &scale.data, input.rows, input.cols).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn timestep_embedding(&self, timesteps: &[f32], dim: usize) -> Result<Self::Tensor, BackendError> {
        if timesteps.is_empty() {
            return Err(compute("timestep embedding 输入不能为空"));
        }
        let data = cpu::vae::timestep_embedding(timesteps, dim).map_err(compute)?;
        Ok(CpuTensor { data, rows: timesteps.len(), cols: dim })
    }

    fn adaln_modulate_segmented(&self, input: &Self::Tensor, shift: &Self::Tensor, scale: &Self::Tensor, segments: &[ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        if shift.cols != input.cols || scale.cols != input.cols || shift.rows != scale.rows {
            return Err(compute("CPU segmented AdaLN shape 不兼容"));
        }
        let row_map = modulation_row_map(segments, input.rows, shift.rows).map_err(compute)?;
        let data = cpu::vae::adaln_modulate_segmented(&input.data, &shift.data, &scale.data, input.rows, input.cols, &row_map).map_err(compute)?;
        Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
    }

    fn gated_residual_segmented(&self, residual: &Self::Tensor, update: &Self::Tensor, gate: &Self::Tensor, segments: &[ModulationSegment]) -> Result<Self::Tensor, BackendError> {
        if residual.rows != update.rows || residual.cols != update.cols || gate.cols != residual.cols {
            return Err(compute("CPU segmented gated residual shape 不兼容"));
        }
        let row_map = modulation_row_map(segments, residual.rows, gate.rows).map_err(compute)?;
        let data = cpu::vae::gated_residual_segmented(&residual.data, &update.data, &gate.data, residual.rows, residual.cols, &row_map).map_err(compute)?;
        Ok(CpuTensor { data, rows: residual.rows, cols: residual.cols })
    }
}
