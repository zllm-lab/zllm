//! H3 音频 VAE 权重准备与 AMP 解码。

use super::prepare_h3_dit_weight;
use crate::{
    backend::{Backend, BackendError, VaeBackend},
    vae::Conv1dSpec,
    weight::model::h3_vae::{H3AudioActivationWeights, H3AudioAmpUnitWeights, H3AudioConvWeights, H3AudioGlobalWeights, H3AudioResBlockWeights, H3AudioStageWeights, H3AudioVaeSource},
};

fn prepare_audio_conv<B: Backend>(backend: &B, source: &H3AudioConvWeights) -> Result<H3AudioConvWeights<B::Weight>, BackendError> {
    Ok(H3AudioConvWeights {
        weight_g: source.weight_g.as_ref().map(|weight| prepare_h3_dit_weight(backend, weight)).transpose()?,
        weight_v: prepare_h3_dit_weight(backend, &source.weight_v)?,
        bias: source.bias.as_ref().map(|weight| prepare_h3_dit_weight(backend, weight)).transpose()?,
    })
}

fn prepare_audio_activation<B: Backend>(backend: &B, source: &H3AudioActivationWeights) -> Result<H3AudioActivationWeights<B::Weight>, BackendError> {
    Ok(H3AudioActivationWeights { alpha: prepare_h3_dit_weight(backend, &source.alpha)?, beta: prepare_h3_dit_weight(backend, &source.beta)? })
}

fn prepare_audio_unit<B: Backend>(backend: &B, source: &H3AudioAmpUnitWeights) -> Result<H3AudioAmpUnitWeights<B::Weight>, BackendError> {
    Ok(H3AudioAmpUnitWeights {
        activation1: prepare_audio_activation(backend, &source.activation1)?,
        conv1: prepare_audio_conv(backend, &source.conv1)?,
        activation2: prepare_audio_activation(backend, &source.activation2)?,
        conv2: prepare_audio_conv(backend, &source.conv2)?,
    })
}

pub fn prepare_audio_stage<B: Backend>(backend: &B, source: &H3AudioStageWeights) -> Result<H3AudioStageWeights<B::Weight>, BackendError> {
    let mut blocks = Vec::with_capacity(source.blocks.len());
    for block in &source.blocks {
        let mut units = Vec::with_capacity(block.units.len());
        for unit in &block.units {
            units.push(prepare_audio_unit(backend, unit)?);
        }
        blocks.push(H3AudioResBlockWeights { units });
    }
    Ok(H3AudioStageWeights { upsample: prepare_audio_conv(backend, &source.upsample)?, blocks })
}

pub fn prepare_audio_vae<B: VaeBackend>(backend: &B, source: &H3AudioVaeSource) -> Result<H3AudioGlobalWeights<B::Weight>, BackendError> {
    let source = source.load_decoder_global().map_err(BackendError::ExpertLoad)?;
    Ok(H3AudioGlobalWeights {
        latent_scale: prepare_h3_dit_weight(backend, &source.latent_scale)?,
        latent_bias: prepare_h3_dit_weight(backend, &source.latent_bias)?,
        input: prepare_audio_conv(backend, &source.input)?,
        conv_pre: prepare_audio_conv(backend, &source.conv_pre)?,
        up_filter: prepare_h3_dit_weight(backend, &source.up_filter)?,
        down_filter: prepare_h3_dit_weight(backend, &source.down_filter)?,
        activation_post: prepare_audio_activation(backend, &source.activation_post)?,
        conv_post: prepare_audio_conv(backend, &source.conv_post)?,
    })
}

pub fn audio_conv<B: VaeBackend>(backend: &B, input: &B::Tensor, weights: &H3AudioConvWeights<B::Weight>, spec: &Conv1dSpec) -> Result<B::Tensor, BackendError> {
    backend.conv1d(input, weights.weight_g.as_ref(), &weights.weight_v, weights.bias.as_ref(), spec)
}

pub fn audio_snake<B: VaeBackend>(backend: &B, input: &B::Tensor, weights: &H3AudioActivationWeights<B::Weight>, global: &H3AudioGlobalWeights<B::Weight>, _batch: usize, channels: usize) -> Result<B::Tensor, BackendError> {
    backend.snake_beta(input, &weights.alpha, &weights.beta, &global.up_filter, &global.down_filter, channels)
}

#[allow(clippy::too_many_arguments)]
pub fn audio_amp_block<B: VaeBackend>(
    backend: &B,
    input: &B::Tensor,
    weights: &H3AudioResBlockWeights<B::Weight>,
    global: &H3AudioGlobalWeights<B::Weight>,
    batch: usize,
    channels: usize,
    kernel: usize,
    dilations: &[usize],
) -> Result<B::Tensor, BackendError> {
    if weights.units.len() != dilations.len() {
        return Err(BackendError::Compute { msg: "H3 audio AMP unit/dilation 数量不匹配".to_owned() });
    }
    // 首个 unit 直接以 input 为残差基底，跳过 scale 1.0 的恒等拷贝。
    let mut hidden: Option<B::Tensor> = None;
    for (unit, &dilation) in weights.units.iter().zip(dilations) {
        let base = hidden.as_ref().unwrap_or(input);
        let activated = audio_snake(backend, base, &unit.activation1, global, batch, channels)?;
        let update = audio_conv(backend, &activated, &unit.conv1, &Conv1dSpec { batch, input_channels: channels, output_channels: channels, kernel, stride: 1, dilation, padding: dilation * (kernel - 1) / 2 })?;
        let activated = audio_snake(backend, &update, &unit.activation2, global, batch, channels)?;
        let update = audio_conv(backend, &activated, &unit.conv2, &Conv1dSpec { batch, input_channels: channels, output_channels: channels, kernel, stride: 1, dilation: 1, padding: (kernel - 1) / 2 })?;
        hidden = Some(backend.add(base, &update)?);
    }
    hidden.ok_or_else(|| BackendError::Compute { msg: "H3 audio AMP 没有 unit".to_owned() })
}

/// 输入为 `[stereo*time, 32]` DiT audio rows，输出为 `[stereo, time*800]` PCM F32。
pub fn decode_audio_vae<B: VaeBackend>(backend: &B, source: &H3AudioVaeSource, global: &H3AudioGlobalWeights<B::Weight>, latent: &B::Tensor, time: usize) -> Result<B::Tensor, BackendError> {
    let spec = crate::vae::H3AudioVaeSpec::standard();
    if backend.token_rows(latent) != spec.output_channels * time || backend.token_cols(latent) != spec.latent_channels {
        return Err(BackendError::Compute { msg: format!("H3 audio VAE latent=[{},{}]，期望 [{},{}]", backend.token_rows(latent), backend.token_cols(latent), spec.output_channels * time, spec.latent_channels,) });
    }
    let mut hidden = backend.audio_unpack_affine(latent, &global.latent_scale, &global.latent_bias, spec.output_channels, time, spec.latent_channels)?;
    hidden = audio_conv(backend, &hidden, &global.input, &Conv1dSpec { batch: spec.output_channels, input_channels: spec.latent_channels, output_channels: spec.latent_dim, kernel: 1, stride: 1, dilation: 1, padding: 0 })?;
    hidden = audio_conv(backend, &hidden, &global.conv_pre, &Conv1dSpec { batch: spec.output_channels, input_channels: spec.latent_dim, output_channels: spec.decoder_dim, kernel: 7, stride: 1, dilation: 1, padding: 3 })?;

    for stage in 0..spec.decoder_rates.len() {
        let _scope = backend.layer_scope();
        let source_weights = source.load_decoder_stage(stage).map_err(|error| BackendError::ExpertLoad(format!("读取 H3 audio VAE stage {stage} 失败: {error}",)))?;
        let weights = prepare_audio_stage(backend, &source_weights)?;
        let input_channels = spec.decoder_dim >> stage;
        let channels = input_channels >> 1;
        let rate = spec.decoder_rates[stage];
        let kernel = spec.decoder_kernels[stage];
        hidden = backend.conv_transpose1d(
            &hidden,
            weights.upsample.weight_g.as_ref().ok_or_else(|| BackendError::Compute { msg: "H3 audio upsample 缺少 weight_g".to_owned() })?,
            &weights.upsample.weight_v,
            weights.upsample.bias.as_ref().ok_or_else(|| BackendError::Compute { msg: "H3 audio upsample 缺少 bias".to_owned() })?,
            &Conv1dSpec { batch: spec.output_channels, input_channels, output_channels: channels, kernel, stride: rate, dilation: 1, padding: (kernel - rate) / 2 },
        )?;
        let mut sum = None;
        for (block, &block_kernel) in weights.blocks.iter().zip(&spec.resblock_kernels) {
            let output = audio_amp_block(backend, &hidden, block, global, spec.output_channels, channels, block_kernel, &spec.resblock_dilations)?;
            sum = Some(match sum {
                Some(sum) => backend.add(&sum, &output)?,
                None => output,
            });
        }
        hidden = backend.scale_tensor(&sum.ok_or_else(|| BackendError::Compute { msg: "H3 audio stage 没有 resblock".to_owned() })?, 1.0 / spec.resblock_kernels.len() as f32)?;
    }

    let channels = spec.decoder_dim >> spec.decoder_rates.len();
    hidden = audio_snake(backend, &hidden, &global.activation_post, global, spec.output_channels, channels)?;
    hidden = audio_conv(backend, &hidden, &global.conv_post, &Conv1dSpec { batch: spec.output_channels, input_channels: channels, output_channels: 1, kernel: 7, stride: 1, dilation: 1, padding: 3 })?;
    backend.tanh(&hidden)
}
