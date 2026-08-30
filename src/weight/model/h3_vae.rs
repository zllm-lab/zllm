//! H3 视频/音频 VAE 权重加载。
//!
//! H3-VAE 是时空因果卷积视频编解码器（空间 16× / 时间 4× 压缩）。
//! 权重 3 shard，10.4 GB。音频 VAE 单独存储。

use crate::{
    vae::{H3AudioVaeSpec, H3VideoVaeSpec},
    weight::container::safetensor::{SafetensorStore, TensorData},
};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// H3 视频 VAE 权重加载器。
pub struct H3VideoVaeSource {
    store: SafetensorStore,
    config: PathBuf,
}

/// H3 音频 VAE 权重加载器。
pub struct H3AudioVaeSource {
    store: SafetensorStore,
    config: PathBuf,
}

pub struct H3AudioConvWeights<T = TensorData> {
    pub weight_g: Option<T>,
    pub weight_v: T,
    pub bias: Option<T>,
}

pub struct H3VideoConvWeights<T = TensorData> {
    pub weight: T,
    pub bias: T,
}

pub struct H3VideoNormWeights<T = TensorData> {
    pub weight: T,
    pub bias: T,
}

pub struct H3VideoEncoderResnetWeights<T = TensorData> {
    pub norm1: H3VideoNormWeights<T>,
    pub conv1: H3VideoConvWeights<T>,
    pub norm2: H3VideoNormWeights<T>,
    pub conv2: H3VideoConvWeights<T>,
    pub shortcut: Option<H3VideoConvWeights<T>>,
}

pub struct H3VideoEncoderLevelWeights<T = TensorData> {
    pub blocks: Vec<H3VideoEncoderResnetWeights<T>>,
    pub downsample: Option<H3VideoConvWeights<T>>,
}

pub struct H3VideoEncoderGlobalWeights<T = TensorData> {
    pub latent_mean: TensorData,
    pub latent_std: TensorData,
    pub conv_in: H3VideoConvWeights<T>,
    pub norm_out: H3VideoNormWeights<T>,
    pub conv_out: H3VideoConvWeights<T>,
    pub quant_conv: H3VideoConvWeights<T>,
}

pub struct H3AudioSnakeWeights<T = TensorData> {
    pub alpha: T,
}

pub struct H3AudioEncoderUnitWeights<T = TensorData> {
    pub activation1: H3AudioSnakeWeights<T>,
    pub conv1: H3AudioConvWeights<T>,
    pub activation2: H3AudioSnakeWeights<T>,
    pub conv2: H3AudioConvWeights<T>,
}

pub struct H3AudioEncoderStageWeights<T = TensorData> {
    pub units: Vec<H3AudioEncoderUnitWeights<T>>,
    pub activation: H3AudioSnakeWeights<T>,
    pub downsample: H3AudioConvWeights<T>,
}

pub struct H3AudioPreBlockWeights<T = TensorData> {
    pub norm1_weight: T,
    pub norm1_bias: T,
    pub qkv_weight: T,
    pub qkv_bias: T,
    pub attention_output_weight: T,
    pub attention_output_bias: T,
    pub input_norm_weight: T,
    pub input_norm_bias: T,
    pub input_weight: T,
    pub input_bias: T,
    pub norm2_weight: T,
    pub norm2_bias: T,
    pub mlp_norm_weight: T,
    pub mlp_norm_bias: T,
    pub gate_weight: T,
    pub gate_bias: T,
    pub up_weight: T,
    pub up_bias: T,
    pub down_weight: T,
    pub down_bias: T,
}

pub struct H3AudioEncoderGlobalWeights<T = TensorData> {
    pub latent_mean: TensorData,
    pub latent_std: TensorData,
    pub input: H3AudioConvWeights<T>,
    pub final_activation: H3AudioSnakeWeights<T>,
    pub final_conv: H3AudioConvWeights<T>,
    pub pre_block: H3AudioPreBlockWeights<T>,
    pub mean_proj_weight: T,
    pub mean_proj_bias: T,
}

pub struct H3AudioActivationWeights<T = TensorData> {
    pub alpha: T,
    pub beta: T,
}

pub struct H3AudioAmpUnitWeights<T = TensorData> {
    pub activation1: H3AudioActivationWeights<T>,
    pub conv1: H3AudioConvWeights<T>,
    pub activation2: H3AudioActivationWeights<T>,
    pub conv2: H3AudioConvWeights<T>,
}

pub struct H3AudioResBlockWeights<T = TensorData> {
    pub units: Vec<H3AudioAmpUnitWeights<T>>,
}

pub struct H3AudioStageWeights<T = TensorData> {
    pub upsample: H3AudioConvWeights<T>,
    pub blocks: Vec<H3AudioResBlockWeights<T>>,
}

pub struct H3AudioGlobalWeights<T = TensorData> {
    pub latent_scale: T,
    pub latent_bias: T,
    pub input: H3AudioConvWeights<T>,
    pub conv_pre: H3AudioConvWeights<T>,
    pub up_filter: T,
    pub down_filter: T,
    pub activation_post: H3AudioActivationWeights<T>,
    pub conv_post: H3AudioConvWeights<T>,
}

pub struct H3VideoVaeGlobalWeights<T = TensorData> {
    pub latent_scale: T,
    pub latent_bias: T,
    pub post_quant_weight: T,
    pub post_quant_bias: T,
    pub input_weight: T,
    pub input_bias: T,
    pub register_tokens: T,
    pub zero_suffix: T,
    pub norm_weight: T,
    pub norm_bias: T,
    pub output_weight: T,
    pub output_bias: T,
}

pub struct H3VideoVaeBlockWeights<T = TensorData> {
    pub norm1: T,
    pub qkv_weight: T,
    pub qkv_bias: T,
    pub attention_output_weight: T,
    pub attention_output_bias: T,
    pub scale1: T,
    pub norm2: T,
    pub gate_up_weight: T,
    pub gate_up_bias: T,
    pub down_weight: T,
    pub down_bias: T,
    pub scale2: T,
}

/// PyTorch VAE 先把 QKV reshape 成 `[token, head, 3 * head_dim]` 再切分，
/// safetensors 的输出维因此按 head 交错。这里一次性改成 runtime 使用的连续 Q/K/V。
fn reorder_decoder_qkv(mut tensor: TensorData, hidden: usize, head_dim: usize) -> Result<TensorData, String> {
    if tensor.dtype != "F32" || tensor.shape.first() != Some(&(hidden * 3)) || !hidden.is_multiple_of(head_dim) {
        return Err(format!("H3 video VAE QKV {} dtype={} shape={:?} hidden={hidden} head_dim={head_dim} 非法", tensor.name, tensor.dtype, tensor.shape,));
    }
    let row_elements = tensor.shape.iter().skip(1).product::<usize>();
    let block_bytes = head_dim.checked_mul(row_elements).and_then(|value| value.checked_mul(std::mem::size_of::<f32>())).ok_or_else(|| format!("H3 video VAE QKV {} block 大小溢出", tensor.name))?;
    let heads = hidden / head_dim;
    let expected = heads.checked_mul(3).and_then(|value| value.checked_mul(block_bytes)).ok_or_else(|| format!("H3 video VAE QKV {} 大小溢出", tensor.name))?;
    if tensor.data.len() != expected {
        return Err(format!("H3 video VAE QKV {} bytes={}，期望 {expected}", tensor.name, tensor.data.len(),));
    }

    let mut reordered = vec![0u8; expected];
    for head in 0..heads {
        for qkv in 0..3 {
            let source = (head * 3 + qkv) * block_bytes;
            let target = (qkv * heads + head) * block_bytes;
            reordered[target..target + block_bytes].copy_from_slice(&tensor.data[source..source + block_bytes]);
        }
    }
    tensor.data = reordered;
    Ok(tensor)
}

impl H3VideoVaeSource {
    /// 打开 vae/ 目录。
    pub fn open(root: &Path) -> Result<Self, String> {
        let vae_dir = [root.join("video_vae/source"), root.join("FL2VA/video_vae/source"), root.join("vae")].into_iter().find(|path| path.is_dir()).ok_or_else(|| format!("H3 video VAE 目录不存在: {}", root.display(),))?;
        let store = SafetensorStore::open(&vae_dir).map_err(|e| format!("打开 H3 video VAE safetensors: {e}"))?;
        let config = vae_dir
            .parent()
            .map(|path| path.join("config.json"))
            .filter(|path| path.is_file())
            .or_else(|| Some(vae_dir.join("config.json")).filter(|path| path.is_file()))
            .ok_or_else(|| format!("H3 video VAE config.json 不存在: {}", vae_dir.display()))?;
        Ok(Self { store, config })
    }

    pub fn load_decoder_global(&self) -> Result<H3VideoVaeGlobalWeights, String> {
        let spec = H3VideoVaeSpec::standard();
        let hidden = spec.decoder_hidden_size;
        let output = 3 * spec.temporal_compression * spec.spatial_compression.pow(2);
        Ok(H3VideoVaeGlobalWeights {
            latent_scale: self.load_config_vector("latents_std", spec.latent_channels)?,
            latent_bias: self.load_config_vector("latents_mean", spec.latent_channels)?,
            post_quant_weight: self.load_reshape("post_quant_conv.weight", &[spec.latent_channels, spec.latent_channels, 1, 1, 1], &[spec.latent_channels, spec.latent_channels])?,
            post_quant_bias: self.load("post_quant_conv.bias", &[spec.latent_channels])?,
            input_weight: self.load("decoder.x_embedder.weight", &[hidden, spec.latent_channels])?,
            input_bias: self.load("decoder.x_embedder.bias", &[hidden])?,
            register_tokens: self.load_reshape("decoder.register_tokens", &[1, spec.decoder_register_tokens, hidden], &[spec.decoder_register_tokens, hidden])?,
            zero_suffix: f32_tensor("decoder.zero_suffix".to_owned(), vec![1, hidden], vec![0.0; hidden]),
            norm_weight: self.load("decoder.norm_out.weight", &[hidden])?,
            norm_bias: self.load("decoder.norm_out.bias", &[hidden])?,
            output_weight: self.load("decoder.proj_out.weight", &[output, hidden])?,
            output_bias: self.load("decoder.proj_out.bias", &[output])?,
        })
    }

    pub fn load_encoder_global(&self) -> Result<H3VideoEncoderGlobalWeights, String> {
        let spec = H3VideoVaeSpec::standard();
        let final_channels = *spec.encoder_channels.last().expect("H3 encoder channels 非空");
        Ok(H3VideoEncoderGlobalWeights {
            latent_mean: self.load_config_vector("latents_mean", spec.latent_channels)?,
            latent_std: self.load_config_vector("latents_std", spec.latent_channels)?,
            conv_in: self.video_conv("encoder.conv_in", spec.encoder_channels[0], 3, 3)?,
            norm_out: self.video_norm("encoder.norm_out", final_channels)?,
            conv_out: self.video_conv("encoder.conv_out", spec.latent_channels * 2, final_channels, 3)?,
            quant_conv: self.video_conv("quant_conv", spec.latent_channels * 2, spec.latent_channels * 2, 1)?,
        })
    }

    pub fn load_encoder_level(&self, level: usize) -> Result<H3VideoEncoderLevelWeights, String> {
        let spec = H3VideoVaeSpec::standard();
        let &channels = spec.encoder_channels.get(level).ok_or_else(|| format!("H3 video encoder level {level} 越界"))?;
        let mut input_channels = if level == 0 { channels } else { spec.encoder_channels[level - 1] };
        let mut blocks = Vec::with_capacity(spec.encoder_res_blocks);
        for block in 0..spec.encoder_res_blocks {
            let prefix = format!("encoder.down.{level}.block.{block}");
            blocks.push(H3VideoEncoderResnetWeights {
                norm1: self.video_norm(&format!("{prefix}.norm1"), input_channels)?,
                conv1: self.video_conv(&format!("{prefix}.conv1"), channels, input_channels, 3)?,
                norm2: self.video_norm(&format!("{prefix}.norm2"), channels)?,
                conv2: self.video_conv(&format!("{prefix}.conv2"), channels, channels, 3)?,
                shortcut: (input_channels != channels).then(|| self.video_conv(&format!("{prefix}.nin_shortcut"), channels, input_channels, 1)).transpose()?,
            });
            input_channels = channels;
        }
        let downsample = (spec.encoder_space_strides[level] * spec.encoder_time_strides[level] > 1).then(|| self.video_conv(&format!("encoder.down.{level}.downsample.conv"), channels, channels, 3)).transpose()?;
        Ok(H3VideoEncoderLevelWeights { blocks, downsample })
    }

    fn video_conv(&self, prefix: &str, output: usize, input: usize, kernel: usize) -> Result<H3VideoConvWeights, String> {
        Ok(H3VideoConvWeights { weight: self.load(&format!("{prefix}.weight"), &[output, input, kernel, kernel, kernel])?, bias: self.load(&format!("{prefix}.bias"), &[output])? })
    }

    fn video_norm(&self, prefix: &str, channels: usize) -> Result<H3VideoNormWeights, String> {
        Ok(H3VideoNormWeights { weight: self.load(&format!("{prefix}.weight"), &[channels])?, bias: self.load(&format!("{prefix}.bias"), &[channels])? })
    }

    pub fn load_decoder_block(&self, layer: usize) -> Result<H3VideoVaeBlockWeights, String> {
        let spec = H3VideoVaeSpec::standard();
        if layer >= spec.decoder_layers {
            return Err(format!("H3 video VAE decoder layer {layer} 越界"));
        }
        let prefix = format!("decoder.transformer_blocks.{layer}");
        let hidden = spec.decoder_hidden_size;
        let ffn = spec.decoder_ffn_size;
        let qkv_weight = reorder_decoder_qkv(self.load(&format!("{prefix}.attn.to_qkv.weight"), &[hidden * 3, hidden])?, hidden, spec.decoder_head_dim)?;
        let qkv_bias = reorder_decoder_qkv(self.load(&format!("{prefix}.attn.to_qkv.bias"), &[hidden * 3])?, hidden, spec.decoder_head_dim)?;
        Ok(H3VideoVaeBlockWeights {
            norm1: self.load(&format!("{prefix}.norm1.weight"), &[hidden])?,
            qkv_weight,
            qkv_bias,
            attention_output_weight: self.load(&format!("{prefix}.attn.to_out.weight"), &[hidden, hidden])?,
            attention_output_bias: self.load(&format!("{prefix}.attn.to_out.bias"), &[hidden])?,
            scale1: self.load(&format!("{prefix}.scale1"), &[hidden])?,
            norm2: self.load(&format!("{prefix}.norm2.weight"), &[hidden])?,
            gate_up_weight: self.load(&format!("{prefix}.ff.w1.weight"), &[ffn * 2, hidden])?,
            gate_up_bias: self.load(&format!("{prefix}.ff.w1.bias"), &[ffn * 2])?,
            down_weight: self.load(&format!("{prefix}.ff.w2.weight"), &[hidden, ffn])?,
            down_bias: self.load(&format!("{prefix}.ff.w2.bias"), &[hidden])?,
            scale2: self.load(&format!("{prefix}.scale2"), &[hidden])?,
        })
    }

    fn load(&self, name: &str, shape: &[usize]) -> Result<TensorData, String> {
        vae_load(&self.store, "VAE", name, shape)
    }

    fn load_reshape(&self, name: &str, source_shape: &[usize], shape: &[usize]) -> Result<TensorData, String> {
        vae_load_reshape(&self.store, "VAE", name, source_shape, shape)
    }

    fn load_config_vector(&self, name: &str, elements: usize) -> Result<TensorData, String> {
        vae_load_config_vector(&self.config, name, elements)
    }

    /// 列出所有 tensor 名（VAE 结构待权重确认后细化）。
    pub fn tensor_names(&self) -> Vec<String> {
        self.store.tensor_names()
    }

    /// 读一个 tensor 转 f32。
    pub fn load_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        vae_load_f32(&self.store, "VAE", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_qkv_is_reordered_from_head_interleaved_layout() {
        let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let tensor = TensorData { name: "qkv".to_owned(), dtype: "F32".to_owned(), shape: vec![12], data: values.iter().flat_map(|value| value.to_le_bytes()).collect() };
        let tensor = reorder_decoder_qkv(tensor, 4, 2).unwrap();
        let values = tensor.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect::<Vec<_>>();
        assert_eq!(values, vec![0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0]);
    }
}

impl H3AudioVaeSource {
    /// 打开 audio_vae/ 目录。
    pub fn open(root: &Path) -> Result<Self, String> {
        let audio_dir = [root.join("audio_vae"), root.join("FL2VA/audio_vae")].into_iter().find(|path| path.is_dir()).ok_or_else(|| format!("audio_vae 目录不存在: {}", root.display()))?;
        let store = SafetensorStore::open(&audio_dir).map_err(|e| format!("打开 H3 audio VAE safetensors: {e}"))?;
        Ok(Self { store, config: audio_dir.join("config.json") })
    }

    pub fn load_decoder_global(&self) -> Result<H3AudioGlobalWeights, String> {
        let spec = H3AudioVaeSpec::standard();
        Ok(H3AudioGlobalWeights {
            latent_scale: self.load_config_vector("latents_std", spec.latent_channels)?,
            latent_bias: self.load_config_vector("latents_mean", spec.latent_channels)?,
            input: self.conv("dec_in_proj", spec.latent_dim, spec.latent_channels, 1, false, true)?,
            conv_pre: self.conv("decoder.conv_pre", spec.decoder_dim, spec.latent_dim, 7, true, true)?,
            up_filter: self.load_reshape("decoder.activation_post.upsample.filter", &[1, 1, spec.alias_filter_kernel], &[spec.alias_filter_kernel])?,
            down_filter: self.load_reshape("decoder.activation_post.downsample.lowpass.filter", &[1, 1, spec.alias_filter_kernel], &[spec.alias_filter_kernel])?,
            activation_post: self.activation("decoder.activation_post", spec.decoder_dim >> spec.decoder_rates.len())?,
            conv_post: self.conv("decoder.conv_post", 1, spec.decoder_dim >> spec.decoder_rates.len(), 7, true, false)?,
        })
    }

    pub fn load_encoder_global(&self) -> Result<H3AudioEncoderGlobalWeights, String> {
        let spec = H3AudioVaeSpec::standard();
        let hidden = spec.latent_dim;
        let projected = spec.latent_channels;
        let qkv_bias = self.concat_tensors("pre_block.attn.qkv.bias", &[("pre_block.attn.q_bias", hidden), ("pre_block.attn.zero_k_bias", hidden), ("pre_block.attn.v_bias", hidden)])?;
        Ok(H3AudioEncoderGlobalWeights {
            latent_mean: self.load_config_vector("latents_mean", projected)?,
            latent_std: self.load_config_vector("latents_std", projected)?,
            input: self.conv("encoder.block.0", spec.encoder_dim, 1, 7, true, true)?,
            final_activation: self.encoder_snake("encoder.block.6", hidden)?,
            final_conv: self.conv("encoder.block.7", hidden, hidden, 3, true, true)?,
            pre_block: H3AudioPreBlockWeights {
                norm1_weight: self.load("pre_block.norm1.weight", &[hidden])?,
                norm1_bias: self.load("pre_block.norm1.bias", &[hidden])?,
                qkv_weight: self.load("pre_block.attn.qkv.weight", &[hidden * 3, hidden])?,
                qkv_bias,
                attention_output_weight: self.load("pre_block.attn.proj.weight", &[projected, projected])?,
                attention_output_bias: self.load("pre_block.attn.proj.bias", &[projected])?,
                input_norm_weight: self.load("pre_block.norm3.weight", &[hidden])?,
                input_norm_bias: self.load("pre_block.norm3.bias", &[hidden])?,
                input_weight: self.load("pre_block.proj.weight", &[projected, hidden])?,
                input_bias: self.load("pre_block.proj.bias", &[projected])?,
                norm2_weight: self.load("pre_block.norm2.weight", &[projected])?,
                norm2_bias: self.load("pre_block.norm2.bias", &[projected])?,
                mlp_norm_weight: self.load("pre_block.mlp.norm.weight", &[projected])?,
                mlp_norm_bias: self.load("pre_block.mlp.norm.bias", &[projected])?,
                gate_weight: self.load("pre_block.mlp.w0.weight", &[projected * 2, projected])?,
                gate_bias: self.load("pre_block.mlp.w0.bias", &[projected * 2])?,
                up_weight: self.load("pre_block.mlp.w1.weight", &[projected * 2, projected])?,
                up_bias: self.load("pre_block.mlp.w1.bias", &[projected * 2])?,
                down_weight: self.load("pre_block.mlp.w2.weight", &[projected, projected * 2])?,
                down_bias: self.load("pre_block.mlp.w2.bias", &[projected])?,
            },
            mean_proj_weight: self.load_reshape("mean_proj.weight", &[projected, projected, 1], &[projected, projected])?,
            mean_proj_bias: self.load("mean_proj.bias", &[projected])?,
        })
    }

    pub fn load_encoder_stage(&self, stage: usize) -> Result<H3AudioEncoderStageWeights, String> {
        let spec = H3AudioVaeSpec::standard();
        let &rate = spec.encoder_rates.get(stage).ok_or_else(|| format!("H3 audio encoder stage {stage} 越界"))?;
        let channels = spec.encoder_dim << stage;
        let output_channels = channels << 1;
        let prefix = format!("encoder.block.{}", stage + 1);
        let mut units = Vec::with_capacity(3);
        for unit in 0..3 {
            let unit_prefix = format!("{prefix}.block.{unit}.block");
            units.push(H3AudioEncoderUnitWeights {
                activation1: self.encoder_snake(&format!("{unit_prefix}.0"), channels)?,
                conv1: self.conv(&format!("{unit_prefix}.1"), channels, channels, 7, true, true)?,
                activation2: self.encoder_snake(&format!("{unit_prefix}.2"), channels)?,
                conv2: self.conv(&format!("{unit_prefix}.3"), channels, channels, 1, true, true)?,
            });
        }
        Ok(H3AudioEncoderStageWeights { units, activation: self.encoder_snake(&format!("{prefix}.block.3"), channels)?, downsample: self.conv(&format!("{prefix}.block.4"), output_channels, channels, rate * 2, true, true)? })
    }

    fn encoder_snake(&self, prefix: &str, channels: usize) -> Result<H3AudioSnakeWeights, String> {
        Ok(H3AudioSnakeWeights { alpha: self.load_reshape(&format!("{prefix}.alpha"), &[1, channels, 1], &[channels])? })
    }

    fn concat_tensors(&self, name: &str, parts: &[(&str, usize)]) -> Result<TensorData, String> {
        let mut values = Vec::new();
        for &(part, elements) in parts {
            values.extend(self.load(part, &[elements])?.to_f32()?);
        }
        Ok(f32_tensor(name.to_owned(), vec![values.len()], values))
    }

    pub fn load_decoder_stage(&self, stage: usize) -> Result<H3AudioStageWeights, String> {
        let spec = H3AudioVaeSpec::standard();
        if stage >= spec.decoder_rates.len() {
            return Err(format!("H3 audio VAE stage {stage} 越界"));
        }
        let input_channels = spec.decoder_dim >> stage;
        let channels = input_channels >> 1;
        let upsample = self.conv_transpose(&format!("decoder.ups.{stage}.0"), input_channels, channels, spec.decoder_kernels[stage])?;
        let mut blocks = Vec::with_capacity(spec.resblock_kernels.len());
        for block in 0..spec.resblock_kernels.len() {
            let index = stage * spec.resblock_kernels.len() + block;
            let kernel = spec.resblock_kernels[block];
            let mut units = Vec::with_capacity(spec.resblock_dilations.len());
            for unit in 0..spec.resblock_dilations.len() {
                let prefix = format!("decoder.resblocks.{index}");
                units.push(H3AudioAmpUnitWeights {
                    activation1: self.activation(&format!("{prefix}.activations.{}", unit * 2), channels)?,
                    conv1: self.conv(&format!("{prefix}.convs1.{unit}"), channels, channels, kernel, true, true)?,
                    activation2: self.activation(&format!("{prefix}.activations.{}", unit * 2 + 1), channels)?,
                    conv2: self.conv(&format!("{prefix}.convs2.{unit}"), channels, channels, kernel, true, true)?,
                });
            }
            blocks.push(H3AudioResBlockWeights { units });
        }
        Ok(H3AudioStageWeights { upsample, blocks })
    }

    fn conv(&self, prefix: &str, output: usize, input: usize, kernel: usize, normalized: bool, bias: bool) -> Result<H3AudioConvWeights, String> {
        Ok(H3AudioConvWeights {
            weight_g: normalized.then(|| self.load_reshape(&format!("{prefix}.weight_g"), &[output, 1, 1], &[output])).transpose()?,
            weight_v: self.load(&format!("{prefix}.{}", if normalized { "weight_v" } else { "weight" }), &[output, input, kernel])?,
            bias: bias.then(|| self.load(&format!("{prefix}.bias"), &[output])).transpose()?,
        })
    }

    fn conv_transpose(&self, prefix: &str, input: usize, output: usize, kernel: usize) -> Result<H3AudioConvWeights, String> {
        Ok(H3AudioConvWeights {
            weight_g: Some(self.load_reshape(&format!("{prefix}.weight_g"), &[input, 1, 1], &[input])?),
            weight_v: self.load(&format!("{prefix}.weight_v"), &[input, output, kernel])?,
            bias: Some(self.load(&format!("{prefix}.bias"), &[output])?),
        })
    }

    fn activation(&self, prefix: &str, channels: usize) -> Result<H3AudioActivationWeights, String> {
        Ok(H3AudioActivationWeights { alpha: self.load(&format!("{prefix}.act.alpha"), &[channels])?, beta: self.load(&format!("{prefix}.act.beta"), &[channels])? })
    }

    fn load(&self, name: &str, shape: &[usize]) -> Result<TensorData, String> {
        vae_load(&self.store, "Audio VAE", name, shape)
    }

    fn load_reshape(&self, name: &str, source_shape: &[usize], shape: &[usize]) -> Result<TensorData, String> {
        vae_load_reshape(&self.store, "Audio VAE", name, source_shape, shape)
    }

    fn load_config_vector(&self, name: &str, elements: usize) -> Result<TensorData, String> {
        vae_load_config_vector(&self.config, name, elements)
    }

    /// 列出所有 tensor 名。
    pub fn tensor_names(&self) -> Vec<String> {
        self.store.tensor_names()
    }

    /// 读一个 tensor 转 f32。
    pub fn load_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        vae_load_f32(&self.store, "Audio VAE", name)
    }
}

// 视频/音频 VAE 的加载、shape 校验与 config 读取逻辑完全一致,仅错误标签不同;共享私有 helper 消除两份拷贝。
fn vae_load(store: &SafetensorStore, label: &str, name: &str, shape: &[usize]) -> Result<TensorData, String> {
    let tensor = store.load(name).map_err(|error| format!("{label} {name}: {error}"))?;
    if tensor.dtype != "F32" || tensor.shape != shape {
        return Err(format!("{label} {name} dtype={} shape={:?}，期望 F32 {shape:?}", tensor.dtype, tensor.shape));
    }
    Ok(tensor)
}

fn vae_load_reshape(store: &SafetensorStore, label: &str, name: &str, source_shape: &[usize], shape: &[usize]) -> Result<TensorData, String> {
    let mut tensor = vae_load(store, label, name, source_shape)?;
    tensor.shape = shape.to_vec();
    Ok(tensor)
}

fn vae_load_config_vector(config: &Path, name: &str, elements: usize) -> Result<TensorData, String> {
    let bytes = fs::read(config).map_err(|error| format!("读取 {}: {error}", config.display()))?;
    let config_json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| format!("解析 {}: {error}", config.display()))?;
    let values = config_json
        .get(name)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} 缺少 {name}", config.display()))?
        .iter()
        .map(|value| value.as_f64().map(|value| value as f32).ok_or_else(|| format!("{name} 包含非数字")))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != elements {
        return Err(format!("{name}={}，期望 {elements}", values.len()));
    }
    Ok(f32_tensor(name.to_owned(), vec![elements], values))
}

fn vae_load_f32(store: &SafetensorStore, label: &str, name: &str) -> Result<Vec<f32>, String> {
    let tensor = store.load(name).map_err(|error| format!("{label} {name}: {error}"))?;
    tensor.to_f32()
}

fn f32_tensor(name: String, shape: Vec<usize>, values: Vec<f32>) -> TensorData {
    TensorData { name, dtype: "F32".to_owned(), shape, data: values.into_iter().flat_map(f32::to_le_bytes).collect() }
}
