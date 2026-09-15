//! SeedVR2 标准 dense safetensors 的命名、形状校验与按需装配。

use std::{path::Path, sync::Arc};

use crate::{
    model_spec::seedvr2::{SeedVr2Config, SeedVr2VaeConfig},
    weight::container::safetensor::{SafetensorStore, TensorData},
};

pub struct SeedVr2Linear<T = TensorData> {
    pub weight: T,
    pub bias: T,
}
pub struct SeedVr2Modulation<T = TensorData> {
    pub shift: T,
    pub scale: T,
    pub gate: T,
}
pub struct SeedVr2AttentionWeights<T = TensorData> {
    /// 原始行顺序为 [Q, K, V]，每组内部 [head, dim]；无 QKV bias。
    pub qkv: T,
    pub query_norm: T,
    pub key_norm: T,
    pub output: SeedVr2Linear<T>,
}
pub struct SeedVr2MlpWeights<T = TensorData> {
    pub input: SeedVr2Linear<T>,
    pub output: SeedVr2Linear<T>,
}
pub struct SeedVr2StreamWeights<T = TensorData> {
    pub attention: SeedVr2AttentionWeights<T>,
    pub attention_modulation: SeedVr2Modulation<T>,
    pub mlp: SeedVr2MlpWeights<T>,
    pub mlp_modulation: SeedVr2Modulation<T>,
}
pub struct SeedVr2BlockWeights<T = TensorData> {
    pub video: SeedVr2StreamWeights<T>,
    pub text: SeedVr2StreamWeights<T>,
    pub rope_frequencies: T,
}
pub struct SeedVr2GlobalWeights<T = TensorData> {
    pub video_input: SeedVr2Linear<T>,
    pub text_input: SeedVr2Linear<T>,
    pub time_input: SeedVr2Linear<T>,
    pub time_hidden: SeedVr2Linear<T>,
    pub time_output: SeedVr2Linear<T>,
}
pub struct SeedVr2FixedEmbeddings<T = TensorData> {
    pub positive: T,
    pub negative: T,
}

pub struct SeedVr2VaeConv<T = TensorData> {
    pub weight: T,
    pub bias: T,
}
pub struct SeedVr2VaeNorm<T = TensorData> {
    pub weight: T,
    pub bias: T,
}
pub struct SeedVr2VaeResnet<T = TensorData> {
    pub norm1: SeedVr2VaeNorm<T>,
    pub conv1: SeedVr2VaeConv<T>,
    pub norm2: SeedVr2VaeNorm<T>,
    pub conv2: SeedVr2VaeConv<T>,
    pub shortcut: Option<SeedVr2VaeConv<T>>,
}
pub struct SeedVr2VaeAttention<T = TensorData> {
    pub group_norm: SeedVr2VaeNorm<T>,
    pub query: SeedVr2Linear<T>,
    pub key: SeedVr2Linear<T>,
    pub value: SeedVr2Linear<T>,
    pub output: SeedVr2Linear<T>,
}
pub struct SeedVr2VaeMid<T = TensorData> {
    pub resnets: [SeedVr2VaeResnet<T>; 2],
    pub attention: SeedVr2VaeAttention<T>,
}
pub struct SeedVr2VaeDownBlock<T = TensorData> {
    pub resnets: Vec<SeedVr2VaeResnet<T>>,
    pub downsample: Option<SeedVr2VaeConv<T>>,
}
pub struct SeedVr2VaeUpsample<T = TensorData> {
    /// 输出通道维原样保留 [x, y, z, c] 顺序，布局转换由执行层负责。
    pub upscale_conv: SeedVr2VaeConv<T>,
    pub conv: SeedVr2VaeConv<T>,
}
pub struct SeedVr2VaeUpBlock<T = TensorData> {
    pub resnets: Vec<SeedVr2VaeResnet<T>>,
    pub upsample: Option<SeedVr2VaeUpsample<T>>,
}
pub struct SeedVr2VaeEncoder<T = TensorData> {
    pub conv_in: SeedVr2VaeConv<T>,
    pub down_blocks: Vec<SeedVr2VaeDownBlock<T>>,
    pub mid: SeedVr2VaeMid<T>,
    pub conv_norm_out: SeedVr2VaeNorm<T>,
    pub conv_out: SeedVr2VaeConv<T>,
}
pub struct SeedVr2VaeDecoder<T = TensorData> {
    pub conv_in: SeedVr2VaeConv<T>,
    pub mid: SeedVr2VaeMid<T>,
    pub up_blocks: Vec<SeedVr2VaeUpBlock<T>>,
    pub conv_norm_out: SeedVr2VaeNorm<T>,
    pub conv_out: SeedVr2VaeConv<T>,
}
pub struct SeedVr2VaeWeights<T = TensorData> {
    pub encoder: SeedVr2VaeEncoder<T>,
    pub decoder: SeedVr2VaeDecoder<T>,
}

#[derive(Clone)]
pub struct SeedVr2DitSource {
    store: Arc<SafetensorStore>,
    config: SeedVr2Config,
}

impl SeedVr2DitSource {
    pub fn open(path: impl AsRef<Path>, config: SeedVr2Config) -> Result<Self, String> {
        config.validate()?;
        let store = open_store(path.as_ref(), "seedvr2_ema_7b_fp16.safetensors")?;
        // 复用装配顺序检查全部 header；此处不读取 7B 权重数据。
        let mut expected = 0;
        let mut check = |name: &str, shape: &[usize]| {
            validate_info(&store, name, shape)?;
            expected += 1;
            Ok(())
        };
        dit_global(&config, &mut check)?;
        for layer in 0..config.num_layers {
            dit_block(&config, layer, &mut check)?;
        }
        dit_final(&config, &mut check)?;
        check_tensor_count(&store, expected)?;
        Ok(Self { store: Arc::new(store), config })
    }

    pub fn config(&self) -> &SeedVr2Config {
        &self.config
    }
    pub fn load_global(&self) -> Result<SeedVr2GlobalWeights, String> {
        self.load_global_with(Ok)
    }
    pub fn load_block(&self, layer: usize) -> Result<SeedVr2BlockWeights, String> {
        self.load_block_with(layer, Ok)
    }
    pub fn load_final(&self) -> Result<SeedVr2Linear, String> {
        self.load_final_with(Ok)
    }

    /// 回调在每个张量读取后立即消费，调用者可自行准备设备权重，避免整层双份暂存。
    pub fn load_global_with<T>(&self, mut prepare: impl FnMut(TensorData) -> Result<T, String>) -> Result<SeedVr2GlobalWeights<T>, String> {
        dit_global(&self.config, &mut |n, s| prepare(load_tensor(&self.store, n, s)?))
    }
    pub fn load_block_with<T>(&self, layer: usize, mut prepare: impl FnMut(TensorData) -> Result<T, String>) -> Result<SeedVr2BlockWeights<T>, String> {
        if layer >= self.config.num_layers {
            return Err(format!("SeedVR2 block {layer} 越界，共 {} 层", self.config.num_layers));
        }
        dit_block(&self.config, layer, &mut |n, s| prepare(load_tensor(&self.store, n, s)?))
    }
    pub fn load_final_with<T>(&self, mut prepare: impl FnMut(TensorData) -> Result<T, String>) -> Result<SeedVr2Linear<T>, String> {
        dit_final(&self.config, &mut |n, s| prepare(load_tensor(&self.store, n, s)?))
    }

    /// 固定文本条件来自独立标准 safetensors 资产，未包含在 DiT checkpoint 中。
    pub fn load_fixed_embeddings(&self, path: impl AsRef<Path>) -> Result<SeedVr2FixedEmbeddings, String> {
        let store = SafetensorStore::open_file(path)?;
        Ok(SeedVr2FixedEmbeddings {
            positive: load_tensor(&store, "pos_emb", &[self.config.positive_text_tokens, self.config.text_dim])?,
            negative: load_tensor(&store, "neg_emb", &[self.config.negative_text_tokens, self.config.text_dim])?,
        })
    }
}

#[derive(Clone)]
pub struct SeedVr2VaeSource {
    store: Arc<SafetensorStore>,
    config: SeedVr2VaeConfig,
}

impl SeedVr2VaeSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let config = SeedVr2VaeConfig::standard();
        config.validate()?;
        let store = open_store(path.as_ref(), "ema_vae_fp16.safetensors")?;
        let mut expected = 0;
        let mut check = |name: &str, shape: &[usize]| {
            validate_info(&store, name, shape)?;
            expected += 1;
            Ok(())
        };
        vae_encoder(&config, &mut check)?;
        vae_decoder(&config, &mut check)?;
        check_tensor_count(&store, expected)?;
        Ok(Self { store: Arc::new(store), config })
    }
    pub fn config(&self) -> &SeedVr2VaeConfig {
        &self.config
    }
    pub fn load_encoder(&self) -> Result<SeedVr2VaeEncoder, String> {
        self.load_encoder_with(Ok)
    }
    pub fn load_decoder(&self) -> Result<SeedVr2VaeDecoder, String> {
        self.load_decoder_with(Ok)
    }
    pub fn load(&self) -> Result<SeedVr2VaeWeights, String> {
        Ok(SeedVr2VaeWeights { encoder: self.load_encoder()?, decoder: self.load_decoder()? })
    }
    pub fn load_encoder_with<T>(&self, mut prepare: impl FnMut(TensorData) -> Result<T, String>) -> Result<SeedVr2VaeEncoder<T>, String> {
        vae_encoder(&self.config, &mut |n, s| prepare(load_tensor(&self.store, n, s)?))
    }
    pub fn load_decoder_with<T>(&self, mut prepare: impl FnMut(TensorData) -> Result<T, String>) -> Result<SeedVr2VaeDecoder<T>, String> {
        vae_decoder(&self.config, &mut |n, s| prepare(load_tensor(&self.store, n, s)?))
    }
}

fn open_store(path: &Path, filename: &str) -> Result<SafetensorStore, String> {
    if path.is_file() {
        SafetensorStore::open_file(path)
    } else if path.join(filename).is_file() {
        SafetensorStore::open_file(path.join(filename))
    } else {
        SafetensorStore::open(path)
    }
}

fn check_dtype(name: &str, dtype: &str) -> Result<(), String> {
    if !matches!(dtype, "F16" | "BF16" | "F32") {
        return Err(format!("SeedVR2 权重 {name} dtype={dtype}，标准 dense 装配仅支持 F16/BF16/F32"));
    }
    Ok(())
}

fn validate_info(store: &SafetensorStore, name: &str, shape: &[usize]) -> Result<(), String> {
    let info = store.tensor_info(name)?;
    if info.shape != shape {
        return Err(format!("SeedVR2 权重 {name} shape={:?}，期望 {shape:?}", info.shape));
    }
    check_dtype(name, &info.dtype)
}

fn check_tensor_count(store: &SafetensorStore, expected: usize) -> Result<(), String> {
    let actual = store.tensor_names().len();
    if actual != expected {
        return Err(format!("SeedVR2 checkpoint 张量数={actual}，标准规格期望 {expected}；包含未装配权重"));
    }
    Ok(())
}

fn load_tensor(store: &SafetensorStore, name: &str, shape: &[usize]) -> Result<TensorData, String> {
    validate_info(store, name, shape)?;
    let tensor = store.load(name)?;
    tensor.expect_shape(shape)?;
    let bytes = if tensor.dtype == "F32" { 4 } else { 2 };
    let expected = shape.iter().try_fold(bytes, |n: usize, dim| n.checked_mul(*dim));
    if expected != Some(tensor.data.len()) {
        return Err(format!("SeedVR2 权重 {name} 数据字节数={}，shape={shape:?}/dtype={} 期望 {expected:?}", tensor.data.len(), tensor.dtype));
    }
    Ok(tensor)
}

fn linear<T>(load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>, name: &str, input: usize, output: usize) -> Result<SeedVr2Linear<T>, String> {
    Ok(SeedVr2Linear { weight: load(&format!("{name}.weight"), &[output, input])?, bias: load(&format!("{name}.bias"), &[output])? })
}

fn dit_global<T>(c: &SeedVr2Config, load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>) -> Result<SeedVr2GlobalWeights<T>, String> {
    Ok(SeedVr2GlobalWeights {
        video_input: linear(load, "vid_in.proj", c.input_channels * c.patch_volume(), c.hidden_size)?,
        text_input: linear(load, "txt_in", c.text_dim, c.hidden_size)?,
        time_input: linear(load, "emb_in.proj_in", c.timestep_dim, c.hidden_size)?,
        time_hidden: linear(load, "emb_in.proj_hid", c.hidden_size, c.hidden_size)?,
        time_output: linear(load, "emb_in.proj_out", c.hidden_size, c.embedding_dim)?,
    })
}

fn dit_final<T>(c: &SeedVr2Config, load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>) -> Result<SeedVr2Linear<T>, String> {
    linear(load, "vid_out.proj", c.hidden_size, c.output_channels * c.patch_volume())
}

fn dit_block<T>(c: &SeedVr2Config, layer: usize, load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>) -> Result<SeedVr2BlockWeights<T>, String> {
    let prefix = format!("blocks.{layer}");
    Ok(SeedVr2BlockWeights { video: dit_stream(c, &prefix, "vid", load)?, text: dit_stream(c, &prefix, "txt", load)?, rope_frequencies: load(&format!("{prefix}.attn.rope.rope.freqs"), &[c.rope_frequency_count()])? })
}

fn dit_stream<T>(c: &SeedVr2Config, prefix: &str, stream: &str, load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>) -> Result<SeedVr2StreamWeights<T>, String> {
    Ok(SeedVr2StreamWeights {
        attention: SeedVr2AttentionWeights {
            qkv: load(&format!("{prefix}.attn.proj_qkv.{stream}.weight"), &[c.hidden_size * 3, c.hidden_size])?,
            query_norm: load(&format!("{prefix}.attn.norm_q.{stream}.weight"), &[c.head_dim])?,
            key_norm: load(&format!("{prefix}.attn.norm_k.{stream}.weight"), &[c.head_dim])?,
            output: linear(load, &format!("{prefix}.attn.proj_out.{stream}"), c.hidden_size, c.hidden_size)?,
        },
        attention_modulation: dit_modulation(load, &format!("{prefix}.ada.{stream}.attn"), c.hidden_size)?,
        mlp: SeedVr2MlpWeights { input: linear(load, &format!("{prefix}.mlp.{stream}.proj_in"), c.hidden_size, c.mlp_hidden_size)?, output: linear(load, &format!("{prefix}.mlp.{stream}.proj_out"), c.mlp_hidden_size, c.hidden_size)? },
        mlp_modulation: dit_modulation(load, &format!("{prefix}.ada.{stream}.mlp"), c.hidden_size)?,
    })
}

fn dit_modulation<T>(load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>, prefix: &str, dim: usize) -> Result<SeedVr2Modulation<T>, String> {
    Ok(SeedVr2Modulation { shift: load(&format!("{prefix}_shift"), &[dim])?, scale: load(&format!("{prefix}_scale"), &[dim])?, gate: load(&format!("{prefix}_gate"), &[dim])? })
}

fn vae_conv<T>(load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>, name: &str, input: usize, output: usize, kernel: [usize; 3]) -> Result<SeedVr2VaeConv<T>, String> {
    Ok(SeedVr2VaeConv { weight: load(&format!("{name}.weight"), &[output, input, kernel[0], kernel[1], kernel[2]])?, bias: load(&format!("{name}.bias"), &[output])? })
}

fn vae_norm<T>(load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>, name: &str, channels: usize) -> Result<SeedVr2VaeNorm<T>, String> {
    Ok(SeedVr2VaeNorm { weight: load(&format!("{name}.weight"), &[channels])?, bias: load(&format!("{name}.bias"), &[channels])? })
}

fn vae_resnet<T>(load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>, prefix: &str, input: usize, output: usize) -> Result<SeedVr2VaeResnet<T>, String> {
    Ok(SeedVr2VaeResnet {
        norm1: vae_norm(load, &format!("{prefix}.norm1"), input)?,
        conv1: vae_conv(load, &format!("{prefix}.conv1"), input, output, [3, 3, 3])?,
        norm2: vae_norm(load, &format!("{prefix}.norm2"), output)?,
        conv2: vae_conv(load, &format!("{prefix}.conv2"), output, output, [3, 3, 3])?,
        shortcut: if input == output { None } else { Some(vae_conv(load, &format!("{prefix}.conv_shortcut"), input, output, [1, 1, 1])?) },
    })
}

fn vae_mid<T>(load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>, prefix: &str, channels: usize) -> Result<SeedVr2VaeMid<T>, String> {
    let attn = format!("{prefix}.attentions.0");
    Ok(SeedVr2VaeMid {
        resnets: [vae_resnet(load, &format!("{prefix}.resnets.0"), channels, channels)?, vae_resnet(load, &format!("{prefix}.resnets.1"), channels, channels)?],
        attention: SeedVr2VaeAttention {
            group_norm: vae_norm(load, &format!("{attn}.group_norm"), channels)?,
            query: linear(load, &format!("{attn}.to_q"), channels, channels)?,
            key: linear(load, &format!("{attn}.to_k"), channels, channels)?,
            value: linear(load, &format!("{attn}.to_v"), channels, channels)?,
            output: linear(load, &format!("{attn}.to_out.0"), channels, channels)?,
        },
    })
}

fn vae_encoder<T>(c: &SeedVr2VaeConfig, load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>) -> Result<SeedVr2VaeEncoder<T>, String> {
    let first = c.block_channels[0];
    let last = c.block_channels[3];
    let conv_in = vae_conv(load, "encoder.conv_in", c.in_channels, first, [3, 3, 3])?;
    let mut input = first;
    let mut down_blocks = Vec::with_capacity(c.block_channels.len());
    for (block, &output) in c.block_channels.iter().enumerate() {
        let prefix = format!("encoder.down_blocks.{block}");
        let mut resnets = Vec::with_capacity(c.layers_per_block);
        for layer in 0..c.layers_per_block {
            resnets.push(vae_resnet(load, &format!("{prefix}.resnets.{layer}"), input, output)?);
            input = output;
        }
        let downsample = if block + 1 == c.block_channels.len() { None } else { Some(vae_conv(load, &format!("{prefix}.downsamplers.0.conv"), output, output, [if block == 0 { 1 } else { 3 }, 3, 3])?) };
        down_blocks.push(SeedVr2VaeDownBlock { resnets, downsample });
    }
    Ok(SeedVr2VaeEncoder {
        conv_in,
        down_blocks,
        mid: vae_mid(load, "encoder.mid_block", last)?,
        conv_norm_out: vae_norm(load, "encoder.conv_norm_out", last)?,
        // checkpoint 同时保存 mean/logvar 的输出；取 mode 是运行时责任。
        conv_out: vae_conv(load, "encoder.conv_out", last, c.latent_channels * 2, [3, 3, 3])?,
    })
}

fn vae_decoder<T>(c: &SeedVr2VaeConfig, load: &mut impl FnMut(&str, &[usize]) -> Result<T, String>) -> Result<SeedVr2VaeDecoder<T>, String> {
    let first = c.block_channels[3];
    let last = c.block_channels[0];
    let conv_in = vae_conv(load, "decoder.conv_in", c.latent_channels, first, [3, 3, 3])?;
    let mid = vae_mid(load, "decoder.mid_block", first)?;
    let mut input = first;
    let mut up_blocks = Vec::with_capacity(c.block_channels.len());
    for (block, &output) in c.block_channels.iter().rev().enumerate() {
        let prefix = format!("decoder.up_blocks.{block}");
        let mut resnets = Vec::with_capacity(c.layers_per_block + 1);
        for layer in 0..=c.layers_per_block {
            resnets.push(vae_resnet(load, &format!("{prefix}.resnets.{layer}"), input, output)?);
            input = output;
        }
        let upsample = if block + 1 == c.block_channels.len() {
            None
        } else {
            let prefix = format!("{prefix}.upsamplers.0");
            Some(SeedVr2VaeUpsample {
                upscale_conv: vae_conv(load, &format!("{prefix}.upscale_conv"), output, output * if block < 2 { 8 } else { 4 }, [1, 1, 1])?,
                conv: vae_conv(load, &format!("{prefix}.conv"), output, output, [3, 3, 3])?,
            })
        };
        up_blocks.push(SeedVr2VaeUpBlock { resnets, upsample });
    }
    Ok(SeedVr2VaeDecoder { conv_in, mid, up_blocks, conv_norm_out: vae_norm(load, "decoder.conv_norm_out", last)?, conv_out: vae_conv(load, "decoder.conv_out", last, c.out_channels, [3, 3, 3])? })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::BTreeMap,
        sync::atomic::{AtomicUsize, Ordering},
    };

    fn dit_schema(c: &SeedVr2Config) -> BTreeMap<String, Vec<usize>> {
        let mut schema = BTreeMap::new();
        let mut record = |name: &str, shape: &[usize]| {
            assert!(schema.insert(name.to_owned(), shape.to_vec()).is_none(), "重复权重 {name}");
            Ok(())
        };
        dit_global(c, &mut record).unwrap();
        for layer in 0..c.num_layers {
            dit_block(c, layer, &mut record).unwrap();
        }
        dit_final(c, &mut record).unwrap();
        schema
    }

    #[test]
    fn complete_7b_checkpoint_schema() {
        let schema = dit_schema(&SeedVr2Config::standard_7b());
        assert_eq!(schema.len(), 1128);
        assert_eq!(schema["vid_in.proj.weight"], [3072, 132]);
        assert_eq!(schema["vid_out.proj.weight"], [64, 3072]);
        assert_eq!(schema["emb_in.proj_out.weight"], [18432, 3072]);
        assert_eq!(schema["blocks.35.attn.proj_qkv.vid.weight"], [9216, 3072]);
        assert_eq!(schema["blocks.35.attn.norm_k.txt.weight"], [128]);
        assert_eq!(schema["blocks.35.attn.rope.rope.freqs"], [10]);
        assert_eq!(schema["blocks.35.mlp.txt.proj_in.weight"], [12288, 3072]);
        assert!(!schema.contains_key("blocks.0.attn.proj_qkv.vid.bias"));
        assert!(!schema.contains_key("emb_scale.proj_in.weight"));
    }

    #[test]
    fn complete_vae_checkpoint_schema_and_sampler_shapes() {
        let c = SeedVr2VaeConfig::standard();
        let mut schema = BTreeMap::new();
        let mut record = |name: &str, shape: &[usize]| {
            assert!(schema.insert(name.to_owned(), shape.to_vec()).is_none(), "重复权重 {name}");
            Ok(())
        };
        let encoder = vae_encoder(&c, &mut record).unwrap();
        let decoder = vae_decoder(&c, &mut record).unwrap();
        assert_eq!(schema.len(), 250);
        assert_eq!(schema["encoder.conv_out.weight"], [32, 512, 3, 3, 3]);
        assert_eq!(schema["encoder.down_blocks.0.downsamplers.0.conv.weight"], [128, 128, 1, 3, 3]);
        assert_eq!(schema["encoder.down_blocks.1.downsamplers.0.conv.weight"], [256, 256, 3, 3, 3]);
        assert_eq!(schema["decoder.up_blocks.0.upsamplers.0.upscale_conv.weight"], [4096, 512, 1, 1, 1]);
        assert_eq!(schema["decoder.up_blocks.2.upsamplers.0.upscale_conv.weight"], [1024, 256, 1, 1, 1]);
        assert_eq!(schema["decoder.mid_block.attentions.0.to_q.weight"], [512, 512]);
        assert!(!schema.keys().any(|name| name.starts_with("quant_conv") || name.starts_with("post_quant_conv")));
        assert!(encoder.down_blocks[3].downsample.is_none());
        assert!(decoder.up_blocks[3].upsample.is_none());
        assert!(encoder.down_blocks.iter().all(|b| b.resnets.len() == 2));
        assert!(decoder.up_blocks.iter().all(|b| b.resnets.len() == 3));
    }

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new(schema: &BTreeMap<String, Vec<usize>>, dtype: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!("zllm-seedvr2-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            std::fs::create_dir_all(&dir).unwrap();
            let mut header = serde_json::Map::new();
            let mut offset = 0usize;
            for (name, shape) in schema {
                let end = offset + shape.iter().product::<usize>() * 2;
                header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}));
                offset = end;
            }
            let mut json = serde_json::to_vec(&header).unwrap();
            while !json.len().is_multiple_of(8) {
                json.push(b' ');
            }
            let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&json);
            // 每个 F16 都是 1.0，用于发现误把原始位模式当作 BF16 的隐式转换。
            for _ in 0..offset / 2 {
                bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
            }
            let file = dir.join("weights.safetensors");
            std::fs::write(&file, bytes).unwrap();
            Self(file)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(self.0.parent().unwrap());
        }
    }

    fn tiny_config() -> SeedVr2Config {
        SeedVr2Config {
            input_channels: 3,
            output_channels: 1,
            text_dim: 8,
            hidden_size: 12,
            num_layers: 2,
            num_heads: 1,
            head_dim: 12,
            mlp_hidden_size: 48,
            timestep_dim: 4,
            embedding_dim: 72,
            patch: [1, 1, 1],
            window: [1, 1, 1],
            norm_eps: 1.0e-5,
            positive_text_tokens: 2,
            negative_text_tokens: 3,
        }
    }

    #[test]
    fn dense_loader_preserves_storage_and_prepares_each_tensor() {
        let c = tiny_config();
        let fixture = Fixture::new(&dit_schema(&c), "F16");
        let source = SeedVr2DitSource::open(&fixture.0, c).unwrap();
        let global = source.load_global().unwrap();
        assert_eq!(global.video_input.weight.dtype, "F16");
        assert!(global.video_input.weight.to_f32().unwrap().iter().all(|v| *v == 1.0));
        let mut visited = Vec::new();
        let block = source
            .load_block_with(1, |tensor| {
                visited.push(tensor.name.clone());
                Ok(tensor.shape)
            })
            .unwrap();
        assert_eq!(visited.len(), 31);
        assert_eq!(block.video.attention.qkv, [36, 12]);
        assert_eq!(block.rope_frequencies, [1]);
        assert_eq!(source.load_final().unwrap().weight.shape, [1, 12]);
        assert!(source.load_block(2).is_err());
        let mut callbacks = 0;
        let failed = source.load_block_with::<()>(0, |_| {
            callbacks += 1;
            Err("模拟设备准备失败".to_owned())
        });
        assert!(matches!(failed, Err(ref e) if e == "模拟设备准备失败"));
        assert_eq!(callbacks, 1);
    }

    #[test]
    fn open_rejects_incomplete_shape_dtype_and_extra_weights() {
        let c = tiny_config();
        for kind in 0..4 {
            let mut schema = dit_schema(&c);
            match kind {
                0 => {
                    schema.remove("blocks.1.ada.txt.mlp_gate");
                }
                1 => {
                    schema.insert("blocks.1.attn.proj_qkv.vid.weight".to_owned(), vec![35, 12]);
                }
                2 => {}
                _ => {
                    schema.insert("unexpected.weight".to_owned(), vec![1]);
                }
            }
            let fixture = Fixture::new(&schema, if kind == 2 { "F8_E4M3" } else { "F16" });
            assert!(SeedVr2DitSource::open(&fixture.0, c.clone()).is_err(), "未拒绝坏 checkpoint kind={kind}");
        }
    }

    #[test]
    fn fixed_embeddings_are_explicit_independent_assets() {
        let c = tiny_config();
        let checkpoint = Fixture::new(&dit_schema(&c), "F16");
        let source = SeedVr2DitSource::open(&checkpoint.0, c).unwrap();
        let schema = BTreeMap::from([("pos_emb".to_owned(), vec![2, 8]), ("neg_emb".to_owned(), vec![3, 8])]);
        let asset = Fixture::new(&schema, "BF16");
        let embeddings = source.load_fixed_embeddings(&asset.0).unwrap();
        assert_eq!(embeddings.positive.shape, [2, 8]);
        assert_eq!(embeddings.negative.dtype, "BF16");
        assert!(source.load_fixed_embeddings(&checkpoint.0).is_err());
    }
}
