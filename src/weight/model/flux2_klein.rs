//! FLUX.2 Klein 4B 官方 Diffusers safetensors 权重映射。

use std::{path::Path, sync::Arc};

use crate::{
    model_spec::flux2_klein::Flux2KleinConfig,
    weight::container::safetensor::{SafetensorStore, TensorData},
};

pub struct Flux2AttentionWeights<T = TensorData> {
    pub query: T,
    pub key: T,
    pub value: T,
    pub query_norm: T,
    pub key_norm: T,
    pub output: T,
}

pub struct Flux2MlpWeights<T = TensorData> {
    pub input: T,
    pub output: T,
}

pub struct Flux2DoubleBlockWeights<T = TensorData> {
    pub image_attention: Flux2AttentionWeights<T>,
    pub image_mlp: Flux2MlpWeights<T>,
    pub text_attention: Flux2AttentionWeights<T>,
    pub text_mlp: Flux2MlpWeights<T>,
}

pub struct Flux2SingleBlockWeights<T = TensorData> {
    pub input: T,
    pub query_norm: T,
    pub key_norm: T,
    pub output: T,
}

pub struct Flux2GlobalWeights<T = TensorData> {
    pub image_input: T,
    pub text_input: T,
    pub time_input: T,
    pub time_output: T,
    pub image_modulation: T,
    pub text_modulation: T,
    pub single_modulation: T,
    pub final_modulation: T,
    pub output: T,
}

pub struct Flux2VaeConv<T = TensorData> {
    pub weight: T,
    pub bias: T,
}

pub struct Flux2VaeResnet<T = TensorData> {
    pub norm1_weight: T,
    pub norm1_bias: T,
    pub conv1: Flux2VaeConv<T>,
    pub norm2_weight: T,
    pub norm2_bias: T,
    pub conv2: Flux2VaeConv<T>,
    pub shortcut: Option<Flux2VaeConv<T>>,
}

pub struct Flux2VaeAttention<T = TensorData> {
    pub norm_weight: T,
    pub norm_bias: T,
    pub query: Flux2VaeConv<T>,
    pub key: Flux2VaeConv<T>,
    pub value: Flux2VaeConv<T>,
    pub output: Flux2VaeConv<T>,
}

pub struct Flux2VaeWeights<T = TensorData> {
    pub batch_mean: T,
    pub batch_variance: T,
    pub encoder_input: Flux2VaeConv<T>,
    pub down_resnets: Vec<Vec<Flux2VaeResnet<T>>>,
    pub downsamplers: Vec<Flux2VaeConv<T>>,
    pub encoder_mid_resnets: Vec<Flux2VaeResnet<T>>,
    pub encoder_mid_attention: Flux2VaeAttention<T>,
    pub encoder_output_norm_weight: T,
    pub encoder_output_norm_bias: T,
    pub encoder_output: Flux2VaeConv<T>,
    pub quant: Flux2VaeConv<T>,
    pub post_quant: Flux2VaeConv<T>,
    pub input: Flux2VaeConv<T>,
    pub mid_resnets: Vec<Flux2VaeResnet<T>>,
    pub mid_attention: Flux2VaeAttention<T>,
    pub up_resnets: Vec<Vec<Flux2VaeResnet<T>>>,
    pub upsamplers: Vec<Flux2VaeConv<T>>,
    pub output_norm_weight: T,
    pub output_norm_bias: T,
    pub output: Flux2VaeConv<T>,
}

#[derive(Clone)]
pub struct Flux2VaeSource {
    store: Arc<SafetensorStore>,
}

impl Flux2VaeSource {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref();
        let file = if root.is_file() {
            root.to_owned()
        } else if root.join("vae/diffusion_pytorch_model.safetensors").is_file() {
            root.join("vae/diffusion_pytorch_model.safetensors")
        } else {
            root.join("diffusion_pytorch_model.safetensors")
        };
        let store = SafetensorStore::open_file(file)?;
        for marker in [
            "bn.running_mean",
            "encoder.conv_in.weight",
            "encoder.mid_block.attentions.0.to_q.weight",
            "quant_conv.weight",
            "post_quant_conv.weight",
            "decoder.conv_in.weight",
            "decoder.mid_block.attentions.0.to_q.weight",
            "decoder.up_blocks.3.resnets.2.conv2.weight",
            "decoder.conv_out.weight",
        ] {
            if !store.has(marker) {
                return Err(format!("FLUX.2 Klein VAE 缺少标记权重 {marker}"));
            }
        }
        Ok(Self { store: Arc::new(store) })
    }

    pub fn load(&self) -> Result<Flux2VaeWeights, String> {
        let encoder_channels = [128, 256, 512, 512];
        let mut down_resnets = Vec::with_capacity(4);
        let mut downsamplers = Vec::with_capacity(3);
        let mut encoder_input_channels = 128;
        for (block, &output_channels) in encoder_channels.iter().enumerate() {
            let mut resnets = Vec::with_capacity(2);
            for layer in 0..2 {
                let prefix = format!("encoder.down_blocks.{block}.resnets.{layer}");
                resnets.push(self.resnet(&prefix, encoder_input_channels, output_channels)?);
                encoder_input_channels = output_channels;
            }
            down_resnets.push(resnets);
            if block < 3 {
                downsamplers.push(self.conv(&format!("encoder.down_blocks.{block}.downsamplers.0.conv"), output_channels, output_channels, 3)?);
            }
        }
        let channels = [512, 512, 256, 128];
        let mut up_resnets = Vec::with_capacity(4);
        let mut upsamplers = Vec::with_capacity(3);
        let mut input_channels = 512;
        for (block, &output_channels) in channels.iter().enumerate() {
            let mut resnets = Vec::with_capacity(3);
            for layer in 0..3 {
                let prefix = format!("decoder.up_blocks.{block}.resnets.{layer}");
                resnets.push(self.resnet(&prefix, input_channels, output_channels)?);
                input_channels = output_channels;
            }
            up_resnets.push(resnets);
            if block < 3 {
                upsamplers.push(self.conv(&format!("decoder.up_blocks.{block}.upsamplers.0.conv"), output_channels, output_channels, 3)?);
            }
        }
        Ok(Flux2VaeWeights {
            batch_mean: self.tensor("bn.running_mean", &[128])?,
            batch_variance: self.tensor("bn.running_var", &[128])?,
            encoder_input: self.conv("encoder.conv_in", 128, 3, 3)?,
            down_resnets,
            downsamplers,
            encoder_mid_resnets: vec![self.resnet("encoder.mid_block.resnets.0", 512, 512)?, self.resnet("encoder.mid_block.resnets.1", 512, 512)?],
            encoder_mid_attention: self.attention("encoder.mid_block.attentions.0", 512)?,
            encoder_output_norm_weight: self.tensor("encoder.conv_norm_out.weight", &[512])?,
            encoder_output_norm_bias: self.tensor("encoder.conv_norm_out.bias", &[512])?,
            encoder_output: self.conv("encoder.conv_out", 64, 512, 3)?,
            quant: self.conv("quant_conv", 64, 64, 1)?,
            post_quant: self.conv("post_quant_conv", 32, 32, 1)?,
            input: self.conv("decoder.conv_in", 512, 32, 3)?,
            mid_resnets: vec![self.resnet("decoder.mid_block.resnets.0", 512, 512)?, self.resnet("decoder.mid_block.resnets.1", 512, 512)?],
            mid_attention: self.attention("decoder.mid_block.attentions.0", 512)?,
            up_resnets,
            upsamplers,
            output_norm_weight: self.tensor("decoder.conv_norm_out.weight", &[128])?,
            output_norm_bias: self.tensor("decoder.conv_norm_out.bias", &[128])?,
            output: self.conv("decoder.conv_out", 3, 128, 3)?,
        })
    }

    fn resnet(&self, prefix: &str, input: usize, output: usize) -> Result<Flux2VaeResnet, String> {
        Ok(Flux2VaeResnet {
            norm1_weight: self.tensor(&format!("{prefix}.norm1.weight"), &[input])?,
            norm1_bias: self.tensor(&format!("{prefix}.norm1.bias"), &[input])?,
            conv1: self.conv(&format!("{prefix}.conv1"), output, input, 3)?,
            norm2_weight: self.tensor(&format!("{prefix}.norm2.weight"), &[output])?,
            norm2_bias: self.tensor(&format!("{prefix}.norm2.bias"), &[output])?,
            conv2: self.conv(&format!("{prefix}.conv2"), output, output, 3)?,
            shortcut: (input != output).then(|| self.conv(&format!("{prefix}.conv_shortcut"), output, input, 1)).transpose()?,
        })
    }

    fn attention(&self, prefix: &str, channels: usize) -> Result<Flux2VaeAttention, String> {
        Ok(Flux2VaeAttention {
            norm_weight: self.tensor(&format!("{prefix}.group_norm.weight"), &[channels])?,
            norm_bias: self.tensor(&format!("{prefix}.group_norm.bias"), &[channels])?,
            query: self.linear(&format!("{prefix}.to_q"), channels)?,
            key: self.linear(&format!("{prefix}.to_k"), channels)?,
            value: self.linear(&format!("{prefix}.to_v"), channels)?,
            output: self.linear(&format!("{prefix}.to_out.0"), channels)?,
        })
    }

    fn linear(&self, prefix: &str, channels: usize) -> Result<Flux2VaeConv, String> {
        Ok(Flux2VaeConv { weight: self.tensor(&format!("{prefix}.weight"), &[channels, channels])?, bias: self.tensor(&format!("{prefix}.bias"), &[channels])? })
    }

    fn conv(&self, prefix: &str, output: usize, input: usize, kernel: usize) -> Result<Flux2VaeConv, String> {
        Ok(Flux2VaeConv { weight: self.tensor(&format!("{prefix}.weight"), &[output, input, kernel, kernel])?, bias: self.tensor(&format!("{prefix}.bias"), &[output])? })
    }

    fn tensor(&self, name: &str, shape: &[usize]) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        tensor.expect_shape(shape)?;
        if !matches!(tensor.dtype.as_str(), "BF16" | "F16" | "F32") {
            return Err(format!("FLUX.2 Klein VAE 权重 {name} dtype={} 不受支持", tensor.dtype));
        }
        Ok(tensor)
    }
}

#[derive(Clone)]
pub struct Flux2KleinSource {
    store: Arc<SafetensorStore>,
    config: Flux2KleinConfig,
}

impl Flux2KleinSource {
    /// `root` 可指向模型根目录、`transformer/` 目录或 transformer safetensors 文件。
    pub fn open(root: impl AsRef<Path>, config: Flux2KleinConfig) -> Result<Self, String> {
        config.validate()?;
        let root = root.as_ref();
        let store = if root.is_file() {
            SafetensorStore::open_file(root)?
        } else {
            let transformer = root.join("transformer");
            let diffusers_file = if transformer.is_dir() { transformer.join("diffusion_pytorch_model.safetensors") } else { root.join("diffusion_pytorch_model.safetensors") };
            if diffusers_file.is_file() {
                SafetensorStore::open_file(diffusers_file)?
            } else if transformer.is_dir() {
                SafetensorStore::open(transformer)?
            } else {
                SafetensorStore::open(root)?
            }
        };
        for marker in ["x_embedder.weight", "transformer_blocks.0.attn.to_q.weight", "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight", "proj_out.weight"] {
            if !store.has(marker) {
                return Err(format!("FLUX.2 Klein transformer 缺少标记权重 {marker}"));
            }
        }
        Ok(Self { store: Arc::new(store), config })
    }

    pub fn config(&self) -> &Flux2KleinConfig {
        &self.config
    }

    pub fn load_global(&self) -> Result<Flux2GlobalWeights, String> {
        let c = &self.config;
        Ok(Flux2GlobalWeights {
            image_input: self.load("x_embedder.weight", &[c.hidden_size, c.input_channels])?,
            text_input: self.load("context_embedder.weight", &[c.hidden_size, c.text_dim])?,
            time_input: self.load("time_guidance_embed.timestep_embedder.linear_1.weight", &[c.hidden_size, c.timestep_dim])?,
            time_output: self.load("time_guidance_embed.timestep_embedder.linear_2.weight", &[c.hidden_size, c.hidden_size])?,
            image_modulation: self.load("double_stream_modulation_img.linear.weight", &[c.hidden_size * 6, c.hidden_size])?,
            text_modulation: self.load("double_stream_modulation_txt.linear.weight", &[c.hidden_size * 6, c.hidden_size])?,
            single_modulation: self.load("single_stream_modulation.linear.weight", &[c.hidden_size * 3, c.hidden_size])?,
            final_modulation: self.load("norm_out.linear.weight", &[c.hidden_size * 2, c.hidden_size])?,
            output: self.load("proj_out.weight", &[c.input_channels, c.hidden_size])?,
        })
    }

    pub fn load_double_block(&self, layer: usize) -> Result<Flux2DoubleBlockWeights, String> {
        if layer >= self.config.double_layers {
            return Err(format!("FLUX.2 Klein double block {layer} 越界，共 {} 层", self.config.double_layers));
        }
        let prefix = format!("transformer_blocks.{layer}");
        Ok(Flux2DoubleBlockWeights {
            image_attention: self.load_attention(&format!("{prefix}.attn"), false)?,
            image_mlp: self.load_mlp(&format!("{prefix}.ff"))?,
            text_attention: self.load_attention(&format!("{prefix}.attn"), true)?,
            text_mlp: self.load_mlp(&format!("{prefix}.ff_context"))?,
        })
    }

    pub fn load_single_block(&self, layer: usize) -> Result<Flux2SingleBlockWeights, String> {
        if layer >= self.config.single_layers {
            return Err(format!("FLUX.2 Klein single block {layer} 越界，共 {} 层", self.config.single_layers));
        }
        let prefix = format!("single_transformer_blocks.{layer}.attn");
        let packed = self.config.hidden_size * 3 + self.config.mlp_hidden_size * 2;
        Ok(Flux2SingleBlockWeights {
            input: self.load(&format!("{prefix}.to_qkv_mlp_proj.weight"), &[packed, self.config.hidden_size])?,
            query_norm: self.load(&format!("{prefix}.norm_q.weight"), &[self.config.head_dim])?,
            key_norm: self.load(&format!("{prefix}.norm_k.weight"), &[self.config.head_dim])?,
            output: self.load(&format!("{prefix}.to_out.weight"), &[self.config.hidden_size, self.config.hidden_size + self.config.mlp_hidden_size])?,
        })
    }

    fn load_attention(&self, prefix: &str, context: bool) -> Result<Flux2AttentionWeights, String> {
        let c = &self.config;
        let (query, key, value, query_norm, key_norm, output) = if context {
            ("add_q_proj.weight", "add_k_proj.weight", "add_v_proj.weight", "norm_added_q.weight", "norm_added_k.weight", "to_add_out.weight")
        } else {
            ("to_q.weight", "to_k.weight", "to_v.weight", "norm_q.weight", "norm_k.weight", "to_out.0.weight")
        };
        Ok(Flux2AttentionWeights {
            query: self.load(&format!("{prefix}.{query}"), &[c.hidden_size, c.hidden_size])?,
            key: self.load(&format!("{prefix}.{key}"), &[c.hidden_size, c.hidden_size])?,
            value: self.load(&format!("{prefix}.{value}"), &[c.hidden_size, c.hidden_size])?,
            query_norm: self.load(&format!("{prefix}.{query_norm}"), &[c.head_dim])?,
            key_norm: self.load(&format!("{prefix}.{key_norm}"), &[c.head_dim])?,
            output: self.load(&format!("{prefix}.{output}"), &[c.hidden_size, c.hidden_size])?,
        })
    }

    fn load_mlp(&self, prefix: &str) -> Result<Flux2MlpWeights, String> {
        Ok(Flux2MlpWeights {
            input: self.load(&format!("{prefix}.linear_in.weight"), &[self.config.mlp_hidden_size * 2, self.config.hidden_size])?,
            output: self.load(&format!("{prefix}.linear_out.weight"), &[self.config.hidden_size, self.config.mlp_hidden_size])?,
        })
    }

    fn load(&self, name: &str, shape: &[usize]) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        tensor.expect_shape(shape)?;
        if !matches!(tensor.dtype.as_str(), "BF16" | "F16" | "F32") {
            return Err(format!("FLUX.2 Klein 权重 {name} dtype={} 不受支持", tensor.dtype));
        }
        Ok(tensor)
    }
}
