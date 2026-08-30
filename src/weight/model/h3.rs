//! MiniMax H3 DiT 官方 safetensors 权重读取。
//!
//! checkpoint 是 13 shard、BF16/F32 混合格式。Block 必须按层读取并尽快释放，
//! 不能把完整模型展开为 F32；那会让约 66GB 的权重膨胀到约 130GB host 内存。

use std::{path::Path, sync::Arc};

use crate::{
    model_spec::h3::H3Config,
    weight::{
        container::safetensor::{SafetensorStore, TensorData},
        format::quantization::W8A16Matrix,
    },
};

/// H3 checkpoint tensor；主干矩阵可保持 INT8 ConvRot 压缩态。
pub struct H3Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
    pub quantized: Option<W8A16Matrix>,
    pub qkv_interleaved: bool,
}

impl H3Tensor {
    fn dense(tensor: TensorData, qkv_interleaved: bool) -> Self {
        Self { name: tensor.name, dtype: tensor.dtype, shape: tensor.shape, data: tensor.data, quantized: None, qkv_interleaved }
    }

    fn scalar(name: &str) -> Self {
        Self { name: name.to_owned(), dtype: "F32".to_owned(), shape: vec![1], data: 0.0f32.to_le_bytes().to_vec(), quantized: None, qkv_interleaved: false }
    }
}

#[derive(Clone)]
pub struct H3TimeCurve {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct ComfyQuantMetadata {
    format: String,
    #[serde(default)]
    convrot: bool,
    #[serde(default)]
    convrot_groupsize: usize,
}

/// H3 主干与 token refiner 共用的 attention 权重。
pub struct H3AttentionWeights<T = H3Tensor> {
    pub qkv: T,
    pub q_norm: T,
    pub k_norm: T,
    pub output: T,
}

/// H3 SwiGLU MLP；fc1 已融合 gate/up。
pub struct H3MlpWeights<T = H3Tensor> {
    pub gate_up: T,
    pub down: T,
}

/// 单个 DiT block；调用者执行完当前层后应立即释放。
pub struct H3DitBlockWeights<T = H3Tensor> {
    pub norm1: T,
    pub norm2: T,
    pub adaln_weight: T,
    pub adaln_bias: T,
    pub attention: H3AttentionWeights<T>,
    pub mlp: H3MlpWeights<T>,
}

/// 单个输入 token refiner block。
pub struct H3TokenRefinerBlockWeights<T = H3Tensor> {
    pub norm1: T,
    pub norm2: T,
    pub attention: H3AttentionWeights<T>,
    pub mlp: H3MlpWeights<T>,
}

/// Patch、condition、时间步和 RoPE 权重；生命周期覆盖一次 DiT forward。
pub struct H3DitGlobalWeights<T = H3Tensor> {
    pub video_patch_weight: T,
    pub video_patch_bias: T,
    pub audio_patch_weight: T,
    pub audio_patch_bias: T,
    pub condition_weight: T,
    pub condition_bias: T,
    pub time_input_weight: T,
    pub time_input_bias: T,
    pub time_output_weight: T,
    pub time_output_bias: T,
    pub rope_inv_freq: T,
    pub time_curve: Option<H3TimeCurve>,
}

/// 最终 AdaLN 与视频/音频 velocity 输出投影。
pub struct H3DitFinalWeights<T = H3Tensor> {
    pub adaln_weight: T,
    pub adaln_bias: T,
    pub norm: T,
    pub video_output_weight: T,
    pub video_output_bias: T,
    pub audio_output_weight: T,
    pub audio_output_bias: T,
}

#[derive(Clone)]
pub struct H3DitSource {
    store: Arc<SafetensorStore>,
    config: H3Config,
}

impl H3DitSource {
    /// `root` 是 H3 的 `FL2VA` 目录，权重位于其 `transformer/` 子目录。
    pub fn open(root: impl AsRef<Path>, mut config: H3Config) -> Result<Self, String> {
        config.validate()?;
        let transformer_dir = root.as_ref().join("transformer");
        let store = SafetensorStore::open(&transformer_dir).map_err(|error| format!("打开 H3 transformer {} 失败: {error}", transformer_dir.display()))?;
        if !store.has("blocks.0.attn.qkv_proj.weight") || !store.has("final_layer.video_out.weight") {
            return Err("H3 transformer index 缺少主干或 final layer 标记权重".to_owned());
        }
        if store.has("adaln_t_table") {
            config.time_embed_dim = 8;
            config.validate()?;
        }
        Ok(Self { store: Arc::new(store), config })
    }

    pub fn config(&self) -> &H3Config {
        &self.config
    }

    pub fn load_global(&self) -> Result<H3DitGlobalWeights, String> {
        let config = &self.config;
        let (time_input_weight, time_input_bias, time_output_weight, time_output_bias, time_curve) = if self.store.has("adaln_t_table") {
            let table = self.load_dense("adaln_t_table", "F32", &[1025, config.time_embed_dim])?;
            let values = table.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 时间曲线"))).collect::<Vec<_>>();
            (
                H3Tensor::scalar("pruned.time_input_weight"),
                H3Tensor::scalar("pruned.time_input_bias"),
                H3Tensor::scalar("pruned.time_output_weight"),
                H3Tensor::scalar("pruned.time_output_bias"),
                Some(H3TimeCurve { rows: 1025, cols: config.time_embed_dim, values }),
            )
        } else {
            (
                self.load("time_embedder.proj_in.weight", "F32", &[config.time_embed_hidden_size, config.timestep_input_dim])?,
                self.load("time_embedder.proj_in.bias", "F32", &[config.time_embed_hidden_size])?,
                self.load("time_embedder.proj_out.weight", "F32", &[config.time_embed_dim, config.time_embed_hidden_size])?,
                self.load("time_embedder.proj_out.bias", "F32", &[config.time_embed_dim])?,
                None,
            )
        };
        Ok(H3DitGlobalWeights {
            video_patch_weight: self.load("video_patch_proj.weight", "F32", &[config.hidden_size, config.video_patch_dim()])?,
            video_patch_bias: self.load("video_patch_proj.bias", "F32", &[config.hidden_size])?,
            audio_patch_weight: self.load("audio_patch_proj.weight", "F32", &[config.hidden_size, config.audio_latent_channels])?,
            audio_patch_bias: self.load("audio_patch_proj.bias", "F32", &[config.hidden_size])?,
            condition_weight: self.load("condition_proj.weight", "BF16", &[config.hidden_size, config.text_dim])?,
            condition_bias: self.load("condition_proj.bias", "BF16", &[config.hidden_size])?,
            time_input_weight,
            time_input_bias,
            time_output_weight,
            time_output_bias,
            rope_inv_freq: self.load("rope.inv_freq", "F32", &[config.rope_inv_freq_len])?,
            time_curve,
        })
    }

    pub fn load_block(&self, layer: usize) -> Result<H3DitBlockWeights, String> {
        if layer >= self.config.num_layers {
            return Err(format!("H3 block {layer} 越界，num_layers={}", self.config.num_layers));
        }
        let prefix = format!("blocks.{layer}");
        Ok(H3DitBlockWeights {
            norm1: self.load(&format!("{prefix}.norm1.weight"), "BF16", &[self.config.hidden_size])?,
            norm2: self.load(&format!("{prefix}.norm2.weight"), "BF16", &[self.config.hidden_size])?,
            adaln_weight: self.load_linear(&format!("{prefix}.adaln_proj.linear.weight"), &[self.config.adaln_out_features, self.config.time_embed_dim])?,
            adaln_bias: self.load(&format!("{prefix}.adaln_proj.linear.bias"), "BF16", &[self.config.adaln_out_features])?,
            attention: self.load_attention(&format!("{prefix}.attn"))?,
            mlp: self.load_mlp(&format!("{prefix}.mlp"))?,
        })
    }

    pub fn load_token_refiner_block(&self, layer: usize) -> Result<H3TokenRefinerBlockWeights, String> {
        if layer >= self.config.token_refiner_num_layers {
            return Err(format!("H3 token refiner block {layer} 越界，layer_count={}", self.config.token_refiner_num_layers,));
        }
        let prefix = format!("token_refiner.blocks.{layer}");
        Ok(H3TokenRefinerBlockWeights {
            norm1: self.load(&format!("{prefix}.norm1.weight"), "BF16", &[self.config.hidden_size])?,
            norm2: self.load(&format!("{prefix}.norm2.weight"), "BF16", &[self.config.hidden_size])?,
            attention: self.load_attention(&format!("{prefix}.attn"))?,
            mlp: self.load_mlp(&format!("{prefix}.mlp"))?,
        })
    }

    pub fn load_token_refiner_final_norm(&self) -> Result<H3Tensor, String> {
        self.load("token_refiner.final_norm.weight", "BF16", &[self.config.hidden_size])
    }

    pub fn load_final(&self) -> Result<H3DitFinalWeights, String> {
        let config = &self.config;
        Ok(H3DitFinalWeights {
            adaln_weight: self.load_linear("final_layer.adaln_proj.linear.weight", &[config.final_adaln_out_features, config.time_embed_dim])?,
            adaln_bias: self.load("final_layer.adaln_proj.linear.bias", "BF16", &[config.final_adaln_out_features])?,
            norm: self.load("final_layer.norm.weight", "BF16", &[config.hidden_size])?,
            video_output_weight: self.load("final_layer.video_out.weight", "F32", &[config.video_patch_dim(), config.hidden_size])?,
            video_output_bias: self.load("final_layer.video_out.bias", "F32", &[config.video_patch_dim()])?,
            audio_output_weight: self.load("final_layer.audio_out.weight", "F32", &[config.audio_latent_channels, config.hidden_size])?,
            audio_output_bias: self.load("final_layer.audio_out.bias", "F32", &[config.audio_latent_channels])?,
        })
    }

    fn load_attention(&self, prefix: &str) -> Result<H3AttentionWeights, String> {
        let attention = self.config.attention_dim();
        Ok(H3AttentionWeights {
            qkv: self.load_linear(&format!("{prefix}.qkv_proj.weight"), &[attention * 3, self.config.hidden_size])?,
            q_norm: self.load(&format!("{prefix}.q_norm.weight"), "BF16", &[self.config.attention_head_dim])?,
            k_norm: self.load(&format!("{prefix}.k_norm.weight"), "BF16", &[self.config.attention_head_dim])?,
            output: self.load_linear(&format!("{prefix}.out_proj.weight"), &[self.config.hidden_size, attention])?,
        })
    }

    fn load_mlp(&self, prefix: &str) -> Result<H3MlpWeights, String> {
        Ok(H3MlpWeights {
            gate_up: self.load_linear(&format!("{prefix}.fc1.weight"), &[self.config.ffn_hidden_size * 2, self.config.hidden_size])?,
            down: self.load_linear(&format!("{prefix}.fc2.weight"), &[self.config.hidden_size, self.config.ffn_hidden_size])?,
        })
    }

    /// 线性层权重:BF16/F16/F32 走原路径;F8_E4M3(per-tensor `_weight_scale`)
    /// 就地 dequant 成 BF16 字节。两者最终都包成 BF16 H3Tensor,下游无感。
    fn load_linear(&self, name: &str, shape: &[usize]) -> Result<H3Tensor, String> {
        let metadata_name = name.strip_suffix(".weight").map_or_else(|| format!("{name}.comfy_quant"), |prefix| format!("{prefix}.comfy_quant"));
        // 只在 metadata 明确标注 ComfyUI int8 ConvRot 时才走 ConvRot 路径。
        // 我们的 local checkpoint 同时声明 comfy_quant metadata(`format: float8_e4m3fn`)
        // 但 weight 本身存的是 F8_E4M3 字节,不能套 ConvRot I8 路径。
        if self.store.has(&metadata_name) && self.is_comfy_int8_convrot(&metadata_name) {
            return self.load(name, "BF16", shape);
        }
        let tensor = self.store.load(name).map_err(|error| format!("H3 {name}: {error}"))?;
        let qkv_interleaved = !self.store.has("adaln_t_table");
        match tensor.dtype.as_str() {
            "BF16" | "F16" | "F32" => {
                if tensor.shape != shape {
                    return Err(format!("H3 {name} shape={:?} 期望 {shape:?}", tensor.shape));
                }
                Ok(H3Tensor::dense(tensor, qkv_interleaved))
            }
            "F8_E4M3" => {
                let [rows, cols] = match tensor.shape.as_slice() {
                    [rows, cols] => [*rows, *cols],
                    _ => return Err(format!("H3 FP8 {name} shape={:?} 必须 rank-2", tensor.shape)),
                };
                if [rows, cols] != shape {
                    return Err(format!("H3 FP8 {name} shape={:?} 期望 {shape:?}", tensor.shape));
                }
                let scale = self.load_fp8_scale(name)?;
                // per-tensor dequant → BF16 字节(行优先),与官方 BF16 checkpoint 布局一致。
                let mut bytes = Vec::with_capacity(tensor.data.len() * 2);
                for &code in &tensor.data {
                    let value = crate::weight::codec::fp8::decode_f8_e4m3(code) * scale;
                    bytes.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
                }
                Ok(H3Tensor::dense(TensorData { name: tensor.name.clone(), dtype: "BF16".to_owned(), shape: tensor.shape.clone(), data: bytes }, qkv_interleaved))
            }
            dtype => Err(format!("H3 线性权重 {name} dtype {dtype} 不支持")),
        }
    }

    /// 读 ComfyUI FP8 权重的 per-tensor scale(F32 标量)。
    /// 张量名约定:`{weight_name}_scale`(weight_name 已含 `.weight` 后缀)。
    fn load_fp8_scale(&self, name: &str) -> Result<f32, String> {
        let scale_name = format!("{name}_scale");
        let tensor = self.store.load(&scale_name).map_err(|error| format!("H3 FP8 {name} 的 {scale_name}: {error}"))?;
        if tensor.dtype != "F32" || tensor.data.len() != 4 {
            return Err(format!("H3 FP8 {scale_name} 期望 F32 标量,实际 dtype={} bytes={}", tensor.dtype, tensor.data.len()));
        }
        Ok(f32::from_le_bytes([tensor.data[0], tensor.data[1], tensor.data[2], tensor.data[3]]))
    }

    /// 仅当 metadata 声明 `format: int8_tensorwise` + `convrot: true` 时返回 true。
    /// 我们的 local checkpoint 也会附带 `comfy_quant` metadata(`format: float8_e4m3fn`),
    /// 但其 weight 主体是 F8_E4M3 字节,不能套 ConvRot 路径,必须 fall through 到 F8_E4M3 / BF16。
    fn is_comfy_int8_convrot(&self, metadata_name: &str) -> bool {
        match self.store.load(metadata_name) {
            Ok(tensor) if tensor.dtype == "U8" => match serde_json::from_slice::<ComfyQuantMetadata>(&tensor.data) {
                Ok(meta) => meta.format == "int8_tensorwise" && meta.convrot,
                Err(_) => false,
            },
            _ => false,
        }
    }

    fn load(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<H3Tensor, String> {
        let metadata_name = name.strip_suffix(".weight").map_or_else(|| format!("{name}.comfy_quant"), |prefix| format!("{prefix}.comfy_quant"));
        if self.store.has(&metadata_name) {
            let tensor = self.store.load(name).map_err(|error| format!("H3 {name}: {error}"))?;
            if tensor.dtype != "I8" || tensor.shape != shape {
                return Err(format!("H3 {name} ConvRot dtype={} shape={:?}，期望 I8 {shape:?}", tensor.dtype, tensor.shape));
            }
            let scale_name = format!("{name}_scale");
            let scales = self.store.load(&scale_name).map_err(|error| format!("H3 {scale_name}: {error}"))?;
            if scales.dtype != "F32" || scales.shape != [shape[0], 1] {
                return Err(format!("H3 {scale_name} dtype={} shape={:?}，期望 F32 [{},1]", scales.dtype, scales.shape, shape[0]));
            }
            let metadata = self.store.load(&metadata_name).map_err(|error| format!("H3 {metadata_name}: {error}"))?;
            if metadata.dtype != "U8" {
                return Err(format!("H3 {metadata_name} dtype={}，期望 U8", metadata.dtype));
            }
            let metadata: ComfyQuantMetadata = serde_json::from_slice(&metadata.data).map_err(|error| format!("解析 H3 {metadata_name} 失败: {error}"))?;
            if metadata.format != "int8_tensorwise" || !metadata.convrot {
                return Err(format!("H3 {metadata_name} format={} convrot={} 不受支持", metadata.format, metadata.convrot));
            }
            let matrix = W8A16Matrix::new_convrot(tensor.data, scales.data, metadata.convrot_groupsize, shape[0], shape[1])?;
            return Ok(H3Tensor { name: name.to_owned(), dtype: "I8".to_owned(), shape: shape.to_vec(), data: Vec::new(), quantized: Some(matrix), qkv_interleaved: false });
        }
        self.load_dense(name, dtype, shape)
    }

    fn load_dense(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<H3Tensor, String> {
        let tensor = self.store.load(name).map_err(|error| format!("H3 {name}: {error}"))?;
        if tensor.dtype != dtype && !(dtype == "BF16" && tensor.dtype == "F16") {
            return Err(format!("H3 {name} dtype={}，期望 {dtype}", tensor.dtype));
        }
        if tensor.shape != shape {
            return Err(format!("H3 {name} shape={:?}，期望 {shape:?}", tensor.shape));
        }
        Ok(H3Tensor::dense(tensor, !self.store.has("adaln_t_table")))
    }
}
