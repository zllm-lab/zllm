//! Gemma 4 官方 BF16 / compressed-tensors / MLX affine / GGUF 权重映射。

use std::path::{Path, PathBuf};

use half::bf16;

use crate::attention::gqa::{CausalWindow, GqaKvProjection};
use crate::model_spec::gemma4::Gemma4Config;
use crate::weight::container::gguf::{GgufMatrix, GgufReader, GgufValue};
use crate::weight::container::safetensor::{SafetensorStore, TensorData};
use crate::weight::format::compressed_tensors::W4A16CtSource;
use crate::weight::format::mlx_affine::MlxAffineSource;
use crate::weight::format::quantization::QuantizedMatrix;

const CT_TEXT_PREFIX: &str = "model.language_model";
const MLX_TEXT_PREFIX: &str = "language_model.model";

pub enum Gemma4Matrix {
    Dense(TensorData),
    Quantized(QuantizedMatrix),
}

impl Gemma4Matrix {
    pub fn rows(&self) -> usize {
        match self {
            Self::Dense(tensor) => tensor.shape.first().copied().unwrap_or(0),
            Self::Quantized(matrix) => matrix.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Dense(tensor) => tensor.shape.get(1).copied().unwrap_or(0),
            Self::Quantized(matrix) => matrix.cols(),
        }
    }
}

pub struct Gemma4AttentionWeights {
    pub q_proj: Gemma4Matrix,
    pub k_proj: Option<Gemma4Matrix>,
    pub v_proj: Option<Gemma4Matrix>,
    pub o_proj: Gemma4Matrix,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

pub struct Gemma4MlpWeights {
    pub gate_proj: Gemma4Matrix,
    pub up_proj: Gemma4Matrix,
    pub down_proj: Gemma4Matrix,
}

pub struct Gemma4PerLayerInputWeights {
    pub gate: Gemma4Matrix,
    pub projection: Gemma4Matrix,
    pub post_norm: Vec<f32>,
}

pub struct Gemma4PerLayerModelWeights {
    pub projection: Gemma4Matrix,
    pub projection_norm: Vec<f32>,
}

pub struct Gemma4LayerWeights {
    pub input_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub pre_feedforward_norm: Vec<f32>,
    pub post_feedforward_norm: Vec<f32>,
    pub layer_scalar: f32,
    pub attention: Gemma4AttentionWeights,
    pub mlp: Gemma4MlpWeights,
    pub per_layer_input: Option<Gemma4PerLayerInputWeights>,
}

enum Gemma4Source {
    Official(SafetensorStore),
    CompressedTensors(W4A16CtSource),
    MlxAffine(MlxAffineSource),
    /// GGUF 主文件（文本 backbone）；`path` 用于在同目录找 mmproj 多模态文件。
    Gguf {
        reader: GgufReader,
        path: PathBuf,
    },
}

pub struct Gemma4VisionWeights {
    pub patch_ln1_weight: Vec<f32>,
    pub patch_ln1_bias: Vec<f32>,
    pub patch_dense: TensorData,
    pub patch_dense_bias: Vec<f32>,
    pub patch_ln2_weight: Vec<f32>,
    pub patch_ln2_bias: Vec<f32>,
    pub position_embedding: Vec<f32>,
    pub position_norm_weight: Vec<f32>,
    pub position_norm_bias: Vec<f32>,
    pub projection: TensorData,
}

pub struct Gemma4VisionClippedLinearWeights {
    pub weight: TensorData,
    pub input_min: f32,
    pub input_max: f32,
    pub output_min: f32,
    pub output_max: f32,
}

pub struct Gemma4VisionEncoderLayerWeights {
    pub input_norm: Vec<f32>,
    pub query: Gemma4VisionClippedLinearWeights,
    pub query_norm: Vec<f32>,
    pub key: Gemma4VisionClippedLinearWeights,
    pub key_norm: Vec<f32>,
    pub value: Gemma4VisionClippedLinearWeights,
    pub output: Gemma4VisionClippedLinearWeights,
    pub attention_post_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub gate: Gemma4VisionClippedLinearWeights,
    pub up: Gemma4VisionClippedLinearWeights,
    pub down: Gemma4VisionClippedLinearWeights,
    pub ffn_post_norm: Vec<f32>,
}

pub struct Gemma4VisionEncoderWeights {
    pub patch_embedding: TensorData,
    pub position_embedding: Vec<f32>,
    pub layers: Vec<Gemma4VisionEncoderLayerWeights>,
    pub projection: TensorData,
}

pub struct Gemma4AudioWeights {
    pub projection: TensorData,
}

pub struct Gemma4MultimodalWeights {
    pub vision: Option<Gemma4VisionWeights>,
    pub vision_encoder: Option<Gemma4VisionEncoderWeights>,
    pub audio: Option<Gemma4AudioWeights>,
}

pub enum Gemma4OutputWeight {
    Dense(TensorData),
    Quantized(QuantizedMatrix),
}

/// gemma4-assistant(MTP 头)的结构参数,全部来自 mtp-*.gguf 元数据。
#[derive(Clone, Debug)]
pub struct Gemma4MtpConfig {
    pub layer_count: usize,
    pub hidden_size: usize,
    pub backbone_hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub local_num_kv_heads: usize,
    pub local_head_dim: usize,
    pub global_num_kv_heads: usize,
    pub global_head_dim: usize,
    pub sliding_window: usize,
    pub rms_eps: f32,
    pub local_rope_theta: f32,
    pub global_rope_theta: f32,
}

pub struct Gemma4MtpLayerWeights {
    pub attention_norm: Vec<f32>,
    pub query: GgufMatrix,
    pub query_norm: Vec<f32>,
    pub attention_output: GgufMatrix,
    pub attention_post_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: GgufMatrix,
    pub ffn_up: GgufMatrix,
    pub ffn_down: GgufMatrix,
    pub ffn_post_norm: Vec<f32>,
    pub layer_output_scale: f32,
}

pub struct Gemma4MtpWeights {
    pub config: Gemma4MtpConfig,
    pub token_embedding: GgufMatrix,
    pub output_norm: Vec<f32>,
    pub pre_projection: GgufMatrix,
    pub post_projection: GgufMatrix,
    pub layers: Vec<Gemma4MtpLayerWeights>,
    /// global 层的逐维 RoPE 频率因子(顶层张量 rope_freqs.weight;缺失为 None)。
    pub rope_freqs: Option<Vec<f32>>,
}

impl Gemma4MtpWeights {
    /// 打开 mtp-*.gguf(gemma4-assistant)。元数据缺失或形状不符时报错。
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let reader = GgufReader::open(path)?;
        let architecture = reader.metadata("general.architecture").and_then(GgufValue::as_str).unwrap_or_default();
        if architecture != "gemma4-assistant" {
            return Err(format!("{} 的 general.architecture={architecture}，期望 gemma4-assistant", path.display()));
        }
        let value = |suffix: &str| -> Result<u64, String> { reader.metadata(&format!("gemma4-assistant.{suffix}")).and_then(GgufValue::as_u64).ok_or_else(|| format!("mtp 元数据缺 gemma4-assistant.{suffix}")) };
        let layer_count = value("block_count")? as usize;
        let hidden_size = value("embedding_length")? as usize;
        let backbone_hidden_size = value("embedding_length_out")? as usize;
        let intermediate_size = value("feed_forward_length")? as usize;
        let token_embedding = reader.read_matrix("token_embd.weight")?;
        let vocab_size = ["vocab_count", "vocab_size"].iter().find_map(|suffix| reader.metadata(&format!("gemma4-assistant.{suffix}")).and_then(GgufValue::as_u64)).map(|value| value as usize).unwrap_or(token_embedding.rows);
        let num_heads = value("attention.head_count")? as usize;
        let sliding_window = value("attention.sliding_window")? as usize;
        let rms_eps = reader.metadata("gemma4-assistant.attention.layer_norm_rms_epsilon").and_then(GgufValue::as_f64).unwrap_or(1.0e-6) as f32;
        let local_rope_theta = reader.metadata("gemma4-assistant.rope.freq_base_swa").and_then(GgufValue::as_f64).unwrap_or(10_000.0) as f32;
        let global_rope_theta = reader.metadata("gemma4-assistant.rope.freq_base").and_then(GgufValue::as_f64).unwrap_or(1_000_000.0) as f32;
        // head_count_kv 数组:前 3 层 local、末层 global(与主干最后 4 层对齐);
        // key_length(标量,global)与 key_length_swa(标量,local)区分头维。
        let kv_heads: Vec<usize> = if let Some(values) = reader.metadata("gemma4-assistant.attention.head_count_kv").and_then(GgufValue::as_i64_array) {
            values.into_iter().map(|value| value as usize).collect()
        } else if let Some(value) = reader.metadata("gemma4-assistant.attention.head_count_kv").and_then(GgufValue::as_u64) {
            // 新版官方 E4B assistant GGUF 在各层 KV 头数相同时写标量，
            // local/global 的 head_dim 仍由 key_length_swa/key_length 区分。
            vec![value as usize; layer_count]
        } else {
            return Err("mtp 元数据缺 head_count_kv".to_owned());
        };
        let local_head_dim = value("attention.key_length_swa")? as usize;
        let global_head_dim = value("attention.key_length")? as usize;
        if kv_heads.len() != layer_count {
            return Err(format!("mtp head_count_kv 长度 {} 与 block_count={layer_count} 不符", kv_heads.len()));
        }
        let local_num_kv_heads = kv_heads[0];
        let global_num_kv_heads = *kv_heads.last().unwrap();
        let config = Gemma4MtpConfig {
            layer_count,
            hidden_size,
            backbone_hidden_size,
            intermediate_size,
            vocab_size,
            num_heads,
            local_num_kv_heads,
            local_head_dim,
            global_num_kv_heads,
            global_head_dim,
            sliding_window,
            rms_eps,
            local_rope_theta,
            global_rope_theta,
        };

        if token_embedding.rows != vocab_size || token_embedding.columns != hidden_size {
            return Err(format!("mtp token_embd [{},{}]，期望 [{vocab_size},{hidden_size}]", token_embedding.rows, token_embedding.columns));
        }
        let output_norm = reader.read_tensor_f32("output_norm.weight")?;
        let pre_projection = reader.read_matrix("nextn.pre_projection.weight")?;
        if pre_projection.rows != hidden_size || pre_projection.columns != 2 * backbone_hidden_size {
            return Err(format!("mtp pre_projection [{},{}]，期望 [{},{}]", pre_projection.rows, pre_projection.columns, hidden_size, 2 * backbone_hidden_size));
        }
        let post_projection = reader.read_matrix("nextn.post_projection.weight")?;
        if post_projection.rows != backbone_hidden_size || post_projection.columns != hidden_size {
            return Err(format!("mtp post_projection [{},{}]，期望 [{backbone_hidden_size},{hidden_size}]", post_projection.rows, post_projection.columns));
        }
        let mut layers = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let prefix = format!("blk.{layer}");
            let query = reader.read_matrix(&format!("{prefix}.attn_q.weight"))?;
            // assistant 的层序固定为 local...local/global；新版 E4B 在两类层的
            // KV 头数相同时把 head_count_kv 写成标量，不能再用头数区分层型。
            let head_dim = if layer + 1 == layer_count { config.global_head_dim } else { config.local_head_dim };
            if query.rows != num_heads * head_dim || query.columns != hidden_size {
                return Err(format!("mtp {prefix}.attn_q [{},{}] 与头布局不符", query.rows, query.columns));
            }
            layers.push(Gemma4MtpLayerWeights {
                attention_norm: reader.read_tensor_f32(&format!("{prefix}.attn_norm.weight"))?,
                query,
                query_norm: reader.read_tensor_f32(&format!("{prefix}.attn_q_norm.weight"))?,
                attention_output: reader.read_matrix(&format!("{prefix}.attn_output.weight"))?,
                attention_post_norm: reader.read_tensor_f32(&format!("{prefix}.post_attention_norm.weight"))?,
                ffn_norm: reader.read_tensor_f32(&format!("{prefix}.ffn_norm.weight"))?,
                ffn_gate: reader.read_matrix(&format!("{prefix}.ffn_gate.weight"))?,
                ffn_up: reader.read_matrix(&format!("{prefix}.ffn_up.weight"))?,
                ffn_down: reader.read_matrix(&format!("{prefix}.ffn_down.weight"))?,
                ffn_post_norm: reader.read_tensor_f32(&format!("{prefix}.post_ffw_norm.weight"))?,
                layer_output_scale: reader.read_tensor_f32(&format!("{prefix}.layer_output_scale.weight"))?.first().copied().unwrap_or(1.0),
            });
        }
        let rope_freqs = reader.tensor("rope_freqs.weight").map(|_| reader.read_tensor_f32("rope_freqs.weight")).transpose()?;
        Ok(Self { config, token_embedding, output_norm, pre_projection, post_projection, layers, rope_freqs })
    }
}

pub struct Gemma4Weights {
    source: Gemma4Source,
    config: Gemma4Config,
}

impl Gemma4Weights {
    pub fn open(root: impl AsRef<Path>, config: Gemma4Config) -> Result<Self, String> {
        let root = root.as_ref();
        let source = if let Some(path) = locate_gguf(root) {
            Gemma4Source::Gguf { reader: GgufReader::open(&path)?, path }
        } else if Self::is_mlx_affine(root)? {
            Gemma4Source::MlxAffine(MlxAffineSource::open(root)?)
        } else if is_compressed_tensors(root)? {
            Gemma4Source::CompressedTensors(W4A16CtSource::open(root)?)
        } else {
            Gemma4Source::Official(SafetensorStore::open(root)?)
        };
        Ok(Self { source, config })
    }

    /// GGUF 单文件或含 GGUF 的目录（排除 mmproj）；safetensors/MLX 目录返回 None。
    pub fn is_gguf(root: impl AsRef<Path>) -> bool {
        locate_gguf(root.as_ref()).is_some()
    }

    pub fn is_mlx_affine(root: impl AsRef<Path>) -> Result<bool, String> {
        let path = root.as_ref().join("config.json");
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).map_err(|error| format!("读取 {} 失败: {error}", path.display()))?).map_err(|error| format!("解析 {} 失败: {error}", path.display()))?;
        Ok(config.get("quantization").and_then(|value| value.get("mode")).and_then(serde_json::Value::as_str) == Some("affine"))
    }

    /// 按 config.json 实际维度选择模型规格:量化格式不决定模型大小,12B 同样存在 MLX affine 量化。
    /// GGUF 没有 config.json,改从 `gemma4.*` metadata 选规格并逐项校验超参。
    pub fn select_config(root: impl AsRef<Path>) -> Result<Gemma4Config, String> {
        if let Some(path) = locate_gguf(root.as_ref()) {
            let reader = GgufReader::open(&path)?;
            return Self::config_from_gguf(&reader);
        }
        let path = root.as_ref().join("config.json");
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).map_err(|error| format!("读取 {} 失败: {error}", path.display()))?).map_err(|error| format!("解析 {} 失败: {error}", path.display()))?;
        let text = config.get("text_config").unwrap_or(&config);
        let missing = |field: &str| format!("config.json 缺少 text_config.{field}");
        let hidden_size = text.get("hidden_size").and_then(serde_json::Value::as_u64).ok_or_else(|| missing("hidden_size"))? as usize;
        let layer_count = text.get("num_hidden_layers").and_then(serde_json::Value::as_u64).ok_or_else(|| missing("num_hidden_layers"))? as usize;
        for spec in [Gemma4Config::e4b(), Gemma4Config::standard_12b()] {
            if spec.hidden_size == hidden_size && spec.layer_count == layer_count {
                return Ok(spec);
            }
        }
        Err(format!("Gemma4 变体 hidden_size={hidden_size} layer_count={layer_count} 没有对应规格"))
    }

    /// GGUF `gemma4.*` metadata → 变体规格；逐项校验硬超参，不符直接报错列差异。
    fn config_from_gguf(reader: &GgufReader) -> Result<Gemma4Config, String> {
        reader.expect_metadata_str("general.architecture", "gemma4")?;
        let hidden_size = reader.metadata_u64("gemma4.embedding_length")? as usize;
        let layer_count = reader.metadata_u64("gemma4.block_count")? as usize;
        let Some(spec) = [Gemma4Config::e4b(), Gemma4Config::standard_12b()].into_iter().find(|spec| spec.hidden_size == hidden_size && spec.layer_count == layer_count) else {
            return Err(format!("Gemma 4 GGUF hidden_size={hidden_size} layer_count={layer_count} 没有对应规格"));
        };
        let expect_u64 = |key: &str, expected: usize| -> Result<(), String> {
            let actual = reader.metadata_u64(key)? as usize;
            if actual != expected {
                return Err(format!("GGUF metadata {key}={actual}，期望 {expected}"));
            }
            Ok(())
        };
        expect_u64("gemma4.feed_forward_length", spec.intermediate_size)?;
        expect_u64("gemma4.attention.head_count", spec.num_heads)?;
        expect_u64("gemma4.attention.key_length", spec.global_head_dim)?;
        expect_u64("gemma4.attention.value_length", spec.global_head_dim)?;
        expect_u64("gemma4.attention.key_length_swa", spec.local_head_dim)?;
        expect_u64("gemma4.attention.value_length_swa", spec.local_head_dim)?;
        expect_u64("gemma4.attention.sliding_window", spec.sliding_window)?;
        expect_u64("gemma4.attention.shared_kv_layers", spec.num_kv_shared_layers)?;
        expect_u64("gemma4.embedding_length_per_layer_input", spec.per_layer_input_size)?;
        let expect_f32 = |key: &str, expected: f32| -> Result<(), String> {
            let actual = reader.metadata(key).and_then(GgufValue::as_f64).ok_or_else(|| format!("GGUF metadata {key} 缺失或类型错误"))? as f32;
            if (actual - expected).abs() > expected.abs() * 1e-6 {
                return Err(format!("GGUF metadata {key}={actual}，期望 {expected}"));
            }
            Ok(())
        };
        expect_f32("gemma4.rope.freq_base", spec.global_rope_theta)?;
        expect_f32("gemma4.rope.freq_base_swa", spec.local_rope_theta)?;
        expect_f32("gemma4.attention.layer_norm_rms_epsilon", spec.rms_eps)?;
        expect_f32("gemma4.final_logit_softcapping", spec.final_logit_softcap)?;
        // llama.cpp 新版 E4B 在 local/global KV 头数相同的情况下写单个标量；
        // 12B 两类层不同，仍写逐层数组。
        let kv_heads = match reader.metadata("gemma4.attention.head_count_kv") {
            Some(GgufValue::Unsigned(value)) => vec![i64::try_from(*value).map_err(|_| "GGUF head_count_kv 超过 i64")?; layer_count],
            Some(value) => value.as_i64_array().ok_or("GGUF metadata gemma4.attention.head_count_kv 类型错误")?,
            None => return Err("GGUF metadata gemma4.attention.head_count_kv 缺失".to_owned()),
        };
        if kv_heads.len() != layer_count {
            return Err(format!("GGUF head_count_kv 长度 {}，期望 {layer_count}", kv_heads.len()));
        }
        for layer in 0..layer_count {
            let expected = config_kv_heads(&spec, layer)?;
            if kv_heads[layer] != expected as i64 {
                return Err(format!("GGUF layer {layer} kv_heads={}，期望 {expected}", kv_heads[layer]));
            }
        }
        // 多模态权重在同目录 mmproj（见 load_multimodal_weights）；config 保留变体默认
        // capability，是否加载成功由权重侧 None 表达。
        Ok(spec)
    }

    /// GGUF 来源时暴露 reader，供 node 取 tokenizer / detokenizer / chat template。
    pub fn gguf_reader(&self) -> Option<&GgufReader> {
        match &self.source {
            Gemma4Source::Gguf { reader, .. } => Some(reader),
            _ => None,
        }
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    pub fn embedding_rows_bf16(&self, token_ids: &[u32]) -> Result<Vec<u8>, String> {
        let rows: Vec<usize> = token_ids.iter().map(|&token| token as usize).collect();
        if let Some(token) = rows.iter().find(|&&token| token >= self.config.vocab_size) {
            return Err(format!("Gemma 4 token {token} 越界于 vocab_size={}", self.config.vocab_size));
        }
        let tensor = match &self.source {
            Gemma4Source::Official(source) => source.load_bf16_rows(&format!("{CT_TEXT_PREFIX}.embed_tokens.weight"), &rows)?,
            Gemma4Source::CompressedTensors(source) => source.load_bf16_rows(&format!("{CT_TEXT_PREFIX}.embed_tokens.weight"), &rows)?,
            Gemma4Source::MlxAffine(_) => return Err("MLX affine embedding 不是 BF16 tensor，请使用 embedding_rows_f32".into()),
            Gemma4Source::Gguf { .. } => return Err("GGUF embedding 是量化 tensor，请使用 embedding_rows_f32".into()),
        };
        if tensor.shape != [rows.len(), self.config.hidden_size] {
            return Err(format!("Gemma 4 embedding rows shape {:?}，期望 [{},{}]", tensor.shape, rows.len(), self.config.hidden_size));
        }
        Ok(tensor.data)
    }

    pub fn embedding_rows_f32(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let rows: Vec<usize> = token_ids.iter().map(|&token| token as usize).collect();
        if let Some(token) = rows.iter().find(|&&token| token >= self.config.vocab_size) {
            return Err(format!("Gemma 4 token {token} 越界于 vocab_size={}", self.config.vocab_size));
        }
        match &self.source {
            Gemma4Source::Official(_) | Gemma4Source::CompressedTensors(_) => {
                let bytes = self.embedding_rows_bf16(token_ids)?;
                Ok(bytes.chunks_exact(2).map(|bytes| bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect())
            }
            Gemma4Source::MlxAffine(source) => source.load_matrix_rows(&format!("{MLX_TEXT_PREFIX}.embed_tokens"), &rows)?.decode(),
            Gemma4Source::Gguf { reader, .. } => gguf_embedding_rows_f32(reader, "token_embd.weight", &rows),
        }
    }

    pub fn per_layer_embedding_rows_f32(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        if self.config.per_layer_input_size == 0 {
            return Ok(Vec::new());
        }
        let rows: Vec<usize> = token_ids.iter().map(|&token| token as usize).collect();
        if let Some(token) = rows.iter().find(|&&token| token >= self.config.vocab_size) {
            return Err(format!("Gemma 4 token {token} 越界于 vocab_size={}", self.config.vocab_size));
        }
        match &self.source {
            Gemma4Source::MlxAffine(source) => source.load_matrix_rows(&format!("{MLX_TEXT_PREFIX}.embed_tokens_per_layer"), &rows)?.decode(),
            Gemma4Source::Gguf { reader, .. } => gguf_embedding_rows_f32(reader, "per_layer_token_embd.weight", &rows),
            _ => Err("当前 Gemma 4 per-layer embedding 只存在于 MLX affine / GGUF checkpoint".into()),
        }
    }

    /// GGUF per-layer token embedding 的原始量化矩阵。Metal decode 重放用它
    /// 常驻 Q5_K blob，并在设备端按上一步 token id 直接抠行。
    pub fn load_per_layer_embedding_gguf(&self) -> Result<Option<GgufMatrix>, String> {
        if self.config.per_layer_input_size == 0 {
            return Ok(None);
        }
        let Gemma4Source::Gguf { reader, .. } = &self.source else {
            return Ok(None);
        };
        let matrix = reader.read_matrix("per_layer_token_embd.weight")?;
        let columns = self.config.layer_count.checked_mul(self.config.per_layer_input_size).ok_or("Gemma 4 per-layer embedding 维度溢出")?;
        if matrix.rows != self.config.vocab_size || matrix.columns != columns {
            return Err(format!("Gemma 4 per-layer embedding shape=[{},{}]，期望 [{},{}]", matrix.rows, matrix.columns, self.config.vocab_size, columns));
        }
        Ok(Some(matrix))
    }

    pub fn load_per_layer_model(&self) -> Result<Option<Gemma4PerLayerModelWeights>, String> {
        let size = self.config.per_layer_input_size;
        if size == 0 {
            return Ok(None);
        }
        let output_size = self.config.layer_count.checked_mul(size).ok_or("Gemma 4 per-layer projection 维度溢出")?;
        Ok(Some(Gemma4PerLayerModelWeights {
            projection: self.load_matrix(&format!("{MLX_TEXT_PREFIX}.per_layer_model_projection"), output_size, self.config.hidden_size)?,
            projection_norm: self.load_vector(&format!("{MLX_TEXT_PREFIX}.per_layer_projection_norm.weight"), size)?,
        }))
    }

    /// 输出投影独立于 embedding；只在准备常驻 LM head 时显式加载。
    pub fn load_lm_head(&self) -> Result<TensorData, String> {
        let Gemma4Source::CompressedTensors(source) = &self.source else {
            if let Gemma4Source::Official(source) = &self.source {
                let tensor = source.load(&format!("{CT_TEXT_PREFIX}.embed_tokens.weight"))?;
                tensor.expect_shape(&[self.config.vocab_size, self.config.hidden_size])?;
                return Ok(tensor);
            }
            return Err("MLX affine / GGUF checkpoint 使用量化 tied embedding，请使用 load_output_weight".into());
        };
        let tensor = source.load_tensor("lm_head.weight")?;
        tensor.expect_shape(&[self.config.vocab_size, self.config.hidden_size])?;
        Ok(tensor)
    }

    pub fn load_output_weight(&self) -> Result<Gemma4OutputWeight, String> {
        match &self.source {
            Gemma4Source::Official(_) | Gemma4Source::CompressedTensors(_) => self.load_lm_head().map(Gemma4OutputWeight::Dense),
            Gemma4Source::MlxAffine(source) => source.load_matrix(&format!("{MLX_TEXT_PREFIX}.embed_tokens")).map(QuantizedMatrix::MlxAffine).map(Gemma4OutputWeight::Quantized),
            // GGUF 通常 tied（无 output.weight）；有独立 output.weight 时优先用它。
            Gemma4Source::Gguf { reader, .. } => {
                let name = if reader.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
                reader.read_matrix(name).map(QuantizedMatrix::Gguf).map(Gemma4OutputWeight::Quantized)
            }
        }
    }

    pub fn final_norm(&self) -> Result<Vec<f32>, String> {
        self.load_vector(&format!("{}.norm.weight", self.text_prefix()), self.config.hidden_size)
    }

    pub fn load_layer(&self, layer: usize) -> Result<Gemma4LayerWeights, String> {
        let attention_spec = self.config.attention_spec(layer).map_err(|error| error.to_string())?;
        let layer_prefix = format!("{}.layers.{layer}", self.text_prefix());
        let attention_prefix = format!("{layer_prefix}.self_attn");
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let hybrid = attention_spec.hybrid;
        let query_size = hybrid.geometry.num_heads * hybrid.geometry.head_dim;
        let kv_size = hybrid.geometry.num_kv_heads * hybrid.geometry.head_dim;
        let v_base = format!("{attention_prefix}.v_proj");
        let kv_shared = self.config.kv_source_layer(layer).map_err(|error| error.to_string())?.is_some();
        let (k_proj, v_proj) = if kv_shared {
            (None, None)
        } else {
            let value = match hybrid.kv_projection {
                GqaKvProjection::Separate => Some(self.load_matrix(&v_base, kv_size, self.config.hidden_size)?),
                GqaKvProjection::KeyAsValue if self.has(&v_base) => {
                    return Err(format!("Gemma 4 full attention layer {layer} 声明 K=V，但 checkpoint 含独立 v_proj"));
                }
                GqaKvProjection::KeyAsValue => None,
            };
            (Some(self.load_matrix(&format!("{attention_prefix}.k_proj"), kv_size, self.config.hidden_size)?), value)
        };
        let per_layer_input = if self.config.per_layer_input_size == 0 {
            None
        } else {
            let size = self.config.per_layer_input_size;
            Some(Gemma4PerLayerInputWeights {
                gate: self.load_matrix(&format!("{layer_prefix}.per_layer_input_gate"), size, self.config.hidden_size)?,
                projection: self.load_matrix(&format!("{layer_prefix}.per_layer_projection"), self.config.hidden_size, size)?,
                post_norm: self.load_vector(&format!("{layer_prefix}.post_per_layer_input_norm.weight"), self.config.hidden_size)?,
            })
        };
        if k_proj.is_some() && matches!(hybrid.window, CausalWindow::Sliding { .. }) && v_proj.is_none() {
            return Err(format!("Gemma 4 sliding attention layer {layer} 缺少 v_proj"));
        }

        Ok(Gemma4LayerWeights {
            input_norm: self.load_vector(&format!("{layer_prefix}.input_layernorm.weight"), self.config.hidden_size)?,
            post_attention_norm: self.load_vector(&format!("{layer_prefix}.post_attention_layernorm.weight"), self.config.hidden_size)?,
            pre_feedforward_norm: self.load_vector(&format!("{layer_prefix}.pre_feedforward_layernorm.weight"), self.config.hidden_size)?,
            post_feedforward_norm: self.load_vector(&format!("{layer_prefix}.post_feedforward_layernorm.weight"), self.config.hidden_size)?,
            layer_scalar: self.load_vector(&format!("{layer_prefix}.layer_scalar"), 1)?[0],
            attention: Gemma4AttentionWeights {
                q_proj: self.load_matrix(&format!("{attention_prefix}.q_proj"), query_size, self.config.hidden_size)?,
                k_proj,
                v_proj,
                o_proj: self.load_matrix(&format!("{attention_prefix}.o_proj"), self.config.hidden_size, query_size)?,
                q_norm: self.load_vector(&format!("{attention_prefix}.q_norm.weight"), hybrid.geometry.head_dim)?,
                // 共享 KV 层（E4B 尾部 per-layer embedding 层）不含 K 权重，checkpoint 也没有 k_norm
                k_norm: if kv_shared { Vec::new() } else { self.load_vector(&format!("{attention_prefix}.k_norm.weight"), hybrid.geometry.head_dim)? },
            },
            mlp: Gemma4MlpWeights {
                gate_proj: self.load_matrix(&format!("{mlp_prefix}.gate_proj"), self.config.intermediate_size, self.config.hidden_size)?,
                up_proj: self.load_matrix(&format!("{mlp_prefix}.up_proj"), self.config.intermediate_size, self.config.hidden_size)?,
                down_proj: self.load_matrix(&format!("{mlp_prefix}.down_proj"), self.config.hidden_size, self.config.intermediate_size)?,
            },
            per_layer_input,
        })
    }

    pub fn load_multimodal_weights(&self) -> Result<Gemma4MultimodalWeights, String> {
        // GGUF 主文件只有文本 backbone；多模态在同目录 mmproj-*.gguf（llama.cpp gemma4uv/gemma4ua 命名）。
        if let Gemma4Source::Gguf { path, .. } = &self.source {
            return load_mmproj_weights(&self.config, path);
        }
        let mut vision = self
            .config
            .vision
            .filter(|config| config.encoder.is_none())
            .map(|config| {
                let patch_size = config.model_patch_size().map_err(|error| error.to_string())?;
                let patch_columns = patch_size.checked_mul(patch_size).and_then(|value| value.checked_mul(3)).ok_or("Gemma 4 vision patch columns 溢出")?;
                let prefix = "model.vision_embedder";
                Ok::<Gemma4VisionWeights, String>(Gemma4VisionWeights {
                    patch_ln1_weight: self.load_vector(&format!("{prefix}.patch_ln1.weight"), patch_columns)?,
                    patch_ln1_bias: self.load_vector(&format!("{prefix}.patch_ln1.bias"), patch_columns)?,
                    patch_dense: self.load_dense_tensor(&format!("{prefix}.patch_dense.weight"), &[config.embedding_size, patch_columns])?,
                    patch_dense_bias: self.load_vector(&format!("{prefix}.patch_dense.bias"), config.embedding_size)?,
                    patch_ln2_weight: self.load_vector(&format!("{prefix}.patch_ln2.weight"), config.embedding_size)?,
                    patch_ln2_bias: self.load_vector(&format!("{prefix}.patch_ln2.bias"), config.embedding_size)?,
                    position_embedding: self.load_tensor(&format!("{prefix}.pos_embedding"))?.to_f32()?,
                    position_norm_weight: self.load_vector(&format!("{prefix}.pos_norm.weight"), config.embedding_size)?,
                    position_norm_bias: self.load_vector(&format!("{prefix}.pos_norm.bias"), config.embedding_size)?,
                    projection: self.load_dense_tensor("model.embed_vision.embedding_projection.weight", &[self.config.hidden_size, config.embedding_size])?,
                })
            })
            .transpose()?;
        if let (Some(config), Some(weights)) = (self.config.vision, vision.as_mut()) {
            let expected = config.position_embedding_size.checked_mul(2).and_then(|value| value.checked_mul(config.embedding_size)).ok_or("Gemma 4 position embedding 大小溢出")?;
            if weights.position_embedding.len() != expected {
                return Err(format!("Gemma 4 position embedding values={}，期望 {expected}", weights.position_embedding.len()));
            }
            // 官方 safetensors 是交错布局 [pos][2][emb](vLLM pos_embedding[:, i, :]),
            // 统一 permute 成 GGUF mmproj 的分块布局 [2][pos][emb],与 encode_image 的
            // 分块索引约定一致(x 表在前、y 表在后)。
            let raw = weights.position_embedding.clone();
            let mut blocked = vec![0.0f32; expected];
            for position in 0..config.position_embedding_size {
                for axis in 0..2 {
                    let source = (position * 2 + axis) * config.embedding_size;
                    let target = (axis * config.position_embedding_size + position) * config.embedding_size;
                    blocked[target..target + config.embedding_size].copy_from_slice(&raw[source..source + config.embedding_size]);
                }
            }
            weights.position_embedding = blocked;
        }
        let vision_encoder = match (&self.source, self.config.vision) {
            (Gemma4Source::MlxAffine(source), Some(vision_config)) if vision_config.encoder.is_some() => {
                let encoder = vision_config.encoder.expect("已筛选 encoder");
                let embedding_size = vision_config.embedding_size;
                let head_dim = embedding_size / encoder.num_heads;
                let dense = |name: &str, shape: &[usize]| -> Result<TensorData, String> {
                    let tensor = source.load_tensor(name)?;
                    tensor.expect_shape(shape)?;
                    if !matches!(tensor.dtype.as_str(), "BF16" | "F16" | "F32") {
                        return Err(format!("{name} dtype={}，期望 dense BF16/F16/F32", tensor.dtype));
                    }
                    Ok(tensor)
                };
                let vector = |name: &str, len: usize| -> Result<Vec<f32>, String> {
                    let tensor = source.load_tensor(name)?;
                    if tensor.shape.iter().product::<usize>() != len {
                        return Err(format!("{name} shape {:?}，期望 {len} 个元素", tensor.shape));
                    }
                    tensor.to_f32()
                };
                let clipped = |name: &str, rows: usize, cols: usize| -> Result<Gemma4VisionClippedLinearWeights, String> {
                    let scalar = |suffix: &str| -> Result<f32, String> {
                        let full = format!("{name}.{suffix}");
                        vector(&full, 1)?.first().copied().ok_or_else(|| format!("{full} 为空"))
                    };
                    Ok(Gemma4VisionClippedLinearWeights {
                        weight: dense(&format!("{name}.linear.weight"), &[rows, cols])?,
                        input_min: scalar("input_min")?,
                        input_max: scalar("input_max")?,
                        output_min: scalar("output_min")?,
                        output_max: scalar("output_max")?,
                    })
                };
                let layers = (0..encoder.layer_count)
                    .map(|layer| {
                        let prefix = format!("vision_tower.encoder.layers.{layer}");
                        Ok(Gemma4VisionEncoderLayerWeights {
                            input_norm: vector(&format!("{prefix}.input_layernorm.weight"), embedding_size)?,
                            query: clipped(&format!("{prefix}.self_attn.q_proj"), embedding_size, embedding_size)?,
                            query_norm: vector(&format!("{prefix}.self_attn.q_norm.weight"), head_dim)?,
                            key: clipped(&format!("{prefix}.self_attn.k_proj"), embedding_size, embedding_size)?,
                            key_norm: vector(&format!("{prefix}.self_attn.k_norm.weight"), head_dim)?,
                            value: clipped(&format!("{prefix}.self_attn.v_proj"), embedding_size, embedding_size)?,
                            output: clipped(&format!("{prefix}.self_attn.o_proj"), embedding_size, embedding_size)?,
                            attention_post_norm: vector(&format!("{prefix}.post_attention_layernorm.weight"), embedding_size)?,
                            ffn_norm: vector(&format!("{prefix}.pre_feedforward_layernorm.weight"), embedding_size)?,
                            gate: clipped(&format!("{prefix}.mlp.gate_proj"), encoder.intermediate_size, embedding_size)?,
                            up: clipped(&format!("{prefix}.mlp.up_proj"), encoder.intermediate_size, embedding_size)?,
                            down: clipped(&format!("{prefix}.mlp.down_proj"), embedding_size, encoder.intermediate_size)?,
                            ffn_post_norm: vector(&format!("{prefix}.post_feedforward_layernorm.weight"), embedding_size)?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let patch_columns = vision_config.patch_size.checked_mul(vision_config.patch_size).and_then(|value| value.checked_mul(3)).ok_or("Gemma 4 vision patch columns 溢出")?;
                let position_embedding = source.load_tensor("vision_tower.patch_embedder.position_embedding_table")?.to_f32()?;
                let expected_positions = 2usize.checked_mul(vision_config.position_embedding_size).and_then(|value| value.checked_mul(embedding_size)).ok_or("Gemma 4 vision position embedding 大小溢出")?;
                if position_embedding.len() != expected_positions {
                    return Err(format!("vision_tower.patch_embedder.position_embedding_table 元素数 {}，期望 {expected_positions}", position_embedding.len()));
                }
                let projection_name = "embed_vision.embedding_projection";
                let projection = source.load_matrix(projection_name)?;
                if projection.rows != self.config.hidden_size || projection.cols != embedding_size {
                    return Err(format!("{projection_name} shape=[{},{}]，期望 [{},{}]", projection.rows, projection.cols, self.config.hidden_size, embedding_size));
                }
                let projection = TensorData { name: projection_name.to_owned(), dtype: "F32".to_owned(), shape: vec![self.config.hidden_size, embedding_size], data: projection.decode()?.into_iter().flat_map(f32::to_le_bytes).collect() };
                Some(Gemma4VisionEncoderWeights { patch_embedding: dense("vision_tower.patch_embedder.input_proj.weight", &[embedding_size, patch_columns])?, position_embedding, layers, projection })
            }
            _ => None,
        };
        let audio = self
            .config
            .audio
            .map(|config| Ok::<Gemma4AudioWeights, String>(Gemma4AudioWeights { projection: self.load_dense_tensor("model.embed_audio.embedding_projection.weight", &[self.config.hidden_size, config.embedding_size])? }))
            .transpose()?;
        Ok(Gemma4MultimodalWeights { vision, vision_encoder, audio })
    }

    /// 能力上报只看当前权重是否真的存在兼容视觉塔，不能由模型族规格代替。
    pub fn has_vision_weights(&self) -> bool {
        match &self.source {
            Gemma4Source::Gguf { path, .. } => locate_mmproj(&self.config, path).ok().flatten().is_some(),
            Gemma4Source::MlxAffine(source) if self.config.vision.is_some_and(|vision| vision.encoder.is_some()) => {
                source.has_tensor("vision_tower.patch_embedder.input_proj.weight") && source.has_matrix("embed_vision.embedding_projection")
            }
            _ if self.config.vision.is_some_and(|vision| vision.encoder.is_some()) => false,
            _ => self.has("model.vision_embedder.patch_dense"),
        }
    }

    fn load_matrix(&self, base: &str, rows: usize, cols: usize) -> Result<Gemma4Matrix, String> {
        let matrix = match &self.source {
            Gemma4Source::Official(source) => Gemma4Matrix::Dense(source.load(&format!("{base}.weight"))?),
            Gemma4Source::CompressedTensors(source) => Gemma4Matrix::Quantized(QuantizedMatrix::W4A16(source.load_matrix(base)?)),
            Gemma4Source::MlxAffine(source) => Gemma4Matrix::Quantized(QuantizedMatrix::MlxAffine(source.load_matrix(base)?)),
            Gemma4Source::Gguf { reader, .. } => {
                let name = gguf_tensor_name(base)?;
                let info = reader.tensor(&name).ok_or_else(|| format!("GGUF 无 tensor {name}"))?;
                // E4B 把 per-layer input 的小矩阵保留为 F32。GGUF resident
                // 量化 kernel 不处理 F32；走既有 dense 路径在装载时转换为 F16。
                if info.tensor_type.0 == 0 {
                    Gemma4Matrix::Dense(TensorData { name: name.clone(), dtype: "F32".to_owned(), shape: vec![rows, cols], data: reader.read_tensor(&name)? })
                } else {
                    Gemma4Matrix::Quantized(QuantizedMatrix::Gguf(reader.read_matrix(&name)?))
                }
            }
        };
        if matrix.rows() != rows || matrix.cols() != cols {
            return Err(format!("{base} shape=[{},{}]，期望 [{rows},{cols}]", matrix.rows(), matrix.cols()));
        }
        Ok(matrix)
    }

    fn load_vector(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        // GGUF 的 Gemma4 norm 与 HF 一样是裸增益（非零中心），原样读即可。
        if let Gemma4Source::Gguf { reader, .. } = &self.source {
            let values = reader.read_tensor_f32(&gguf_tensor_name(name)?)?;
            if values.len() != len {
                return Err(format!("{name} 元素数 {}，期望 {len}", values.len()));
            }
            return Ok(values);
        }
        let tensor = match &self.source {
            Gemma4Source::Official(source) => source.load(name)?,
            Gemma4Source::CompressedTensors(source) => source.load_tensor(name)?,
            Gemma4Source::MlxAffine(source) => source.load_tensor(name)?,
            Gemma4Source::Gguf { .. } => unreachable!("GGUF 已在上面返回"),
        };
        if tensor.shape.iter().product::<usize>() != len {
            return Err(format!("{name} shape {:?}，期望 {len} 个元素", tensor.shape));
        }
        tensor.to_f32()
    }

    fn text_prefix(&self) -> &'static str {
        match self.source {
            Gemma4Source::Official(_) => CT_TEXT_PREFIX,
            Gemma4Source::CompressedTensors(_) => CT_TEXT_PREFIX,
            Gemma4Source::MlxAffine(_) => MLX_TEXT_PREFIX,
            // GGUF 在权重边界统一把 HF 名翻译成 llama.cpp 名（gguf_tensor_name）。
            Gemma4Source::Gguf { .. } => CT_TEXT_PREFIX,
        }
    }

    fn has(&self, base: &str) -> bool {
        match &self.source {
            Gemma4Source::Official(source) => source.has(&format!("{base}.weight")),
            Gemma4Source::CompressedTensors(source) => source.has(base),
            Gemma4Source::MlxAffine(source) => source.has_matrix(base),
            Gemma4Source::Gguf { reader, .. } => gguf_tensor_name(base).is_ok_and(|name| reader.tensor(&name).is_some()),
        }
    }

    fn load_tensor(&self, name: &str) -> Result<TensorData, String> {
        match &self.source {
            Gemma4Source::Official(source) => source.load(name),
            Gemma4Source::CompressedTensors(source) => source.load_tensor(name),
            Gemma4Source::MlxAffine(source) => source.load_tensor(name),
            Gemma4Source::Gguf { .. } => Err(format!("GGUF 主文件不含 {name}；多模态权重从 mmproj 加载")),
        }
    }

    fn load_dense_tensor(&self, name: &str, shape: &[usize]) -> Result<TensorData, String> {
        let tensor = self.load_tensor(name)?;
        tensor.expect_shape(shape)?;
        if !matches!(tensor.dtype.as_str(), "BF16" | "F16" | "F32") {
            return Err(format!("{name} dtype={}，期望 dense BF16/F16/F32", tensor.dtype));
        }
        Ok(tensor)
    }
}

fn is_compressed_tensors(root: &Path) -> Result<bool, String> {
    let path = root.join("config.json");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).map_err(|error| format!("读取 {} 失败: {error}", path.display()))?).map_err(|error| format!("解析 {} 失败: {error}", path.display()))?;
    Ok(config.get("quantization_config").and_then(|value| value.get("quant_method")).and_then(serde_json::Value::as_str).is_some_and(|method| method.contains("compressed")))
}

/// 定位 GGUF 主文件：单文件要求 .gguf 后缀；目录委托 `GgufReader::locate`（排除 mmproj）。
fn locate_gguf(root: &Path) -> Option<PathBuf> {
    if root.is_file() {
        return (root.extension().is_some_and(|ext| ext == "gguf")).then(|| root.to_owned());
    }
    GgufReader::locate(root).ok()
}

/// 该层规格的 KV 头数（global 层 K=V 只有 1 头，local 层 8 头）。
fn config_kv_heads(config: &Gemma4Config, layer: usize) -> Result<usize, String> {
    Ok(config.attention_spec(layer).map_err(|error| error.to_string())?.hybrid.geometry.num_kv_heads)
}

/// HF/MLX 风格张量名 → llama.cpp GGUF 张量名。矩阵 base（无 .weight 后缀）与完整向量名都接受。
/// E4B per-layer 名对应 llama.cpp 的 per_layer_token_embd / per_layer_model_proj /
/// per_layer_proj_norm / blk.{i}.inp_gate / blk.{i}.proj / blk.{i}.post_norm。
fn gguf_tensor_name(name: &str) -> Result<String, String> {
    let stripped = name.strip_prefix(CT_TEXT_PREFIX).or_else(|| name.strip_prefix(MLX_TEXT_PREFIX)).and_then(|rest| rest.strip_prefix('.')).unwrap_or(name);
    let rename = match stripped {
        "embed_tokens.weight" | "embed_tokens" => return Ok("token_embd.weight".to_owned()),
        "norm.weight" => return Ok("output_norm.weight".to_owned()),
        "embed_tokens_per_layer" => return Ok("per_layer_token_embd.weight".to_owned()),
        "per_layer_model_projection" => return Ok("per_layer_model_proj.weight".to_owned()),
        "per_layer_projection_norm.weight" => return Ok("per_layer_proj_norm.weight".to_owned()),
        other => other,
    };
    let Some((index, leaf)) = rename.strip_prefix("layers.").and_then(|rest| rest.split_once('.')) else {
        return Err(format!("Gemma 4 张量 {name} 没有 GGUF 名称映射"));
    };
    let leaf = match leaf {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
        "self_attn.q_proj" | "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj" | "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj" | "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj" | "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj" | "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj" | "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj" | "mlp.down_proj.weight" => "ffn_down.weight",
        "layer_scalar" => "layer_output_scale.weight",
        "per_layer_input_gate" => "inp_gate.weight",
        "per_layer_projection" => "proj.weight",
        "post_per_layer_input_norm.weight" => "post_norm.weight",
        _ => return Err(format!("Gemma 4 张量 {name} 没有 GGUF 名称映射")),
    };
    Ok(format!("blk.{index}.{leaf}"))
}

/// GGUF embedding 按行懒解码成 f32（prefill chunk / 逐 token decode 共用）。
fn gguf_embedding_rows_f32(reader: &GgufReader, name: &str, rows: &[usize]) -> Result<Vec<f32>, String> {
    let matrix = reader.read_matrix(name)?;
    let row_bytes = matrix.tensor_type.storage_bytes(matrix.columns)?;
    let ids: Vec<u32> = rows.iter().map(|&row| row as u32).collect();
    let mut raw = vec![0u8; rows.len().checked_mul(row_bytes).ok_or("GGUF embedding rows 大小溢出")?];
    matrix.read_rows_into(&ids, &mut raw)?;
    let mut output = vec![0.0; rows.len() * matrix.columns];
    for (chunk, destination) in raw.chunks_exact(row_bytes).zip(output.chunks_exact_mut(matrix.columns)) {
        let decoded = crate::weight::codec::ggml::dequantize(matrix.tensor_type.0, chunk, matrix.columns)?;
        destination.copy_from_slice(&decoded);
    }
    Ok(output)
}

/// GGUF 主文件同目录的 mmproj-*.gguf（llama.cpp clip 命名，gemma4uv/gemma4ua unified projector）
/// 映射为 Gemma4 多模态权重；没有 mmproj 时返回纯文本（vision/audio = None）。
fn load_mmproj_weights(config: &Gemma4Config, gguf_path: &Path) -> Result<Gemma4MultimodalWeights, String> {
    let empty = Gemma4MultimodalWeights { vision: None, vision_encoder: None, audio: None };
    let Some(mmproj) = locate_mmproj(config, gguf_path)? else { return Ok(empty) };
    let reader = GgufReader::open(&mmproj)?;
    // mmproj 张量名以真实文件为准核对：gemma4uv = v.patch_embd / v.patch_norm.{1,2,3} /
    // v.position_embd / mm.input_projection；gemma4ua = mm.a.input_projection。
    let dense = |name: &str, shape: &[usize]| -> Result<TensorData, String> {
        let info = reader.tensor(name).ok_or_else(|| format!("mmproj 缺少 {name}"))?;
        let dtype = match info.tensor_type.0 {
            0 => "F32",
            1 => "F16",
            30 => "BF16",
            other => return Err(format!("mmproj {name} dtype={other}，期望 dense F32/F16/BF16")),
        };
        let actual: Vec<usize> = info.dims.iter().rev().copied().collect();
        if actual != shape {
            return Err(format!("mmproj {name} shape {actual:?}，期望 {shape:?}"));
        }
        Ok(TensorData { name: name.to_owned(), dtype: dtype.to_owned(), shape: shape.to_vec(), data: reader.read_tensor(name)? })
    };
    let dense_flattened = |name: &str, rows: usize, cols: usize| -> Result<TensorData, String> {
        let info = reader.tensor(name).ok_or_else(|| format!("mmproj 缺少 {name}"))?;
        let dtype = match info.tensor_type.0 {
            0 => "F32",
            1 => "F16",
            30 => "BF16",
            other => return Err(format!("mmproj {name} dtype={other}，期望 dense F32/F16/BF16")),
        };
        let actual: Vec<usize> = info.dims.iter().rev().copied().collect();
        if actual.first() != Some(&rows) || actual.iter().product::<usize>() != rows * cols {
            return Err(format!("mmproj {name} shape {actual:?}，不能展平为 [{rows}, {cols}]"));
        }
        Ok(TensorData { name: name.to_owned(), dtype: dtype.to_owned(), shape: vec![rows, cols], data: reader.read_tensor(name)? })
    };
    let vector = |name: &str, len: usize| -> Result<Vec<f32>, String> {
        let values = reader.read_tensor_f32(name)?;
        if values.len() != len {
            return Err(format!("mmproj {name} 元素数 {}，期望 {len}", values.len()));
        }
        Ok(values)
    };
    let vision = config
        .vision
        .filter(|vision| vision.encoder.is_none())
        .map(|vision_config| {
            let patch_size = vision_config.model_patch_size().map_err(|error| error.to_string())?;
            let patch_columns = patch_size.checked_mul(patch_size).and_then(|value| value.checked_mul(3)).ok_or("Gemma 4 vision patch columns 溢出")?;
            let embedding_size = vision_config.embedding_size;
            Ok::<Gemma4VisionWeights, String>(Gemma4VisionWeights {
                patch_ln1_weight: vector("v.patch_norm.1.weight", patch_columns)?,
                patch_ln1_bias: vector("v.patch_norm.1.bias", patch_columns)?,
                patch_dense: dense("v.patch_embd.weight", &[embedding_size, patch_columns])?,
                patch_dense_bias: vector("v.patch_embd.bias", embedding_size)?,
                patch_ln2_weight: vector("v.patch_norm.2.weight", embedding_size)?,
                patch_ln2_bias: vector("v.patch_norm.2.bias", embedding_size)?,
                position_embedding: reader.read_tensor_f32("v.position_embd.weight")?,
                position_norm_weight: vector("v.patch_norm.3.weight", embedding_size)?,
                position_norm_bias: vector("v.patch_norm.3.bias", embedding_size)?,
                projection: dense("mm.input_projection.weight", &[config.hidden_size, embedding_size])?,
            })
        })
        .transpose()?;
    let vision_encoder = config
        .vision
        .filter(|vision| vision.encoder.is_some())
        .map(|vision_config| {
            let encoder = vision_config.encoder.expect("已筛选 encoder");
            let patch_columns = vision_config.patch_size.checked_mul(vision_config.patch_size).and_then(|value| value.checked_mul(3)).ok_or("Gemma 4 vision patch columns 溢出")?;
            let head_dim = vision_config.embedding_size / encoder.num_heads;
            let clipped = |name: &str, rows: usize, cols: usize| -> Result<Gemma4VisionClippedLinearWeights, String> {
                let scalar = |suffix: &str, fallback: f32| -> Result<f32, String> {
                    let full = format!("{name}.{suffix}");
                    let Some(_) = reader.tensor(&full) else { return Ok(fallback) };
                    let values = reader.read_tensor_f32(&full)?;
                    values.first().copied().ok_or_else(|| format!("mmproj {full} 为空"))
                };
                Ok(Gemma4VisionClippedLinearWeights {
                    weight: dense(&format!("{name}.weight"), &[rows, cols])?,
                    input_min: scalar("input_min", f32::MIN)?,
                    input_max: scalar("input_max", f32::MAX)?,
                    output_min: scalar("output_min", f32::MIN)?,
                    output_max: scalar("output_max", f32::MAX)?,
                })
            };
            let layers = (0..encoder.layer_count)
                .map(|layer| {
                    let prefix = format!("v.blk.{layer}");
                    Ok(Gemma4VisionEncoderLayerWeights {
                        input_norm: vector(&format!("{prefix}.ln1.weight"), vision_config.embedding_size)?,
                        query: clipped(&format!("{prefix}.attn_q"), vision_config.embedding_size, vision_config.embedding_size)?,
                        query_norm: vector(&format!("{prefix}.attn_q_norm.weight"), head_dim)?,
                        key: clipped(&format!("{prefix}.attn_k"), vision_config.embedding_size, vision_config.embedding_size)?,
                        key_norm: vector(&format!("{prefix}.attn_k_norm.weight"), head_dim)?,
                        value: clipped(&format!("{prefix}.attn_v"), vision_config.embedding_size, vision_config.embedding_size)?,
                        output: clipped(&format!("{prefix}.attn_out"), vision_config.embedding_size, vision_config.embedding_size)?,
                        attention_post_norm: vector(&format!("{prefix}.attn_post_norm.weight"), vision_config.embedding_size)?,
                        ffn_norm: vector(&format!("{prefix}.ln2.weight"), vision_config.embedding_size)?,
                        gate: clipped(&format!("{prefix}.ffn_gate"), encoder.intermediate_size, vision_config.embedding_size)?,
                        up: clipped(&format!("{prefix}.ffn_up"), encoder.intermediate_size, vision_config.embedding_size)?,
                        down: clipped(&format!("{prefix}.ffn_down"), vision_config.embedding_size, encoder.intermediate_size)?,
                        ffn_post_norm: vector(&format!("{prefix}.ffn_post_norm.weight"), vision_config.embedding_size)?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok::<Gemma4VisionEncoderWeights, String>(Gemma4VisionEncoderWeights {
                // GGUF 保留卷积核四维 shape [out, channel, height, width]；
                // patch 提取后等价于行主序 dense [out, channel*height*width]。
                patch_embedding: dense_flattened("v.patch_embd.weight", vision_config.embedding_size, patch_columns)?,
                position_embedding: {
                    let values = reader.read_tensor_f32("v.position_embd.weight")?;
                    let expected = 2 * vision_config.position_embedding_size * vision_config.embedding_size;
                    if values.len() != expected {
                        return Err(format!("mmproj v.position_embd.weight 元素数 {}，期望 {expected}", values.len()));
                    }
                    values
                },
                layers,
                projection: dense("mm.input_projection.weight", &[config.hidden_size, vision_config.embedding_size])?,
            })
        })
        .transpose()?;
    let audio = config.audio.map(|audio_config| Ok::<Gemma4AudioWeights, String>(Gemma4AudioWeights { projection: dense("mm.a.input_projection.weight", &[config.hidden_size, audio_config.embedding_size])? })).transpose()?;
    Ok(Gemma4MultimodalWeights { vision, vision_encoder, audio })
}

/// 同目录可能同时存在 BF16/Q8 mmproj；当前视觉矩阵只支持 dense，确定性优先 BF16。
fn locate_mmproj(config: &Gemma4Config, gguf_path: &Path) -> Result<Option<PathBuf>, String> {
    let Some(directory) = gguf_path.parent() else { return Ok(None) };
    let mut candidates = std::fs::read_dir(directory)
        .map_err(|error| format!("读取 mmproj 目录 {}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("mmproj") && name.ends_with(".gguf")))
        .collect::<Vec<_>>();
    candidates.sort_by_key(|path| (!path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.contains("BF16")), path.clone()));
    for path in candidates {
        let reader = GgufReader::open(&path)?;
        let projector = reader.metadata("clip.vision.projector_type").and_then(GgufValue::as_str);
        let expected = config.vision.and_then(|vision| vision.encoder).map_or("gemma4uv", |_| "gemma4v");
        if projector != Some(expected) || !matches!(reader.metadata("clip.has_vision_encoder"), Some(GgufValue::Bool(true))) {
            continue;
        }
        let Some(projection) = reader.tensor("mm.input_projection.weight") else { continue };
        if !matches!(projection.tensor_type.0, 0 | 1 | 30) {
            continue;
        }
        return Ok(Some(path));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    /// MTP 头加载 smoke:`ZLLM_GEMMA4_MTP=/path/to/mtp.gguf cargo test --lib mtp_open -- --nocapture`
    #[test]
    fn mtp_open() {
        let Some(path) = std::env::var_os("ZLLM_GEMMA4_MTP").map(PathBuf::from) else { return };
        let weights = Gemma4MtpWeights::open(&path).expect("open mtp");
        let config = &weights.config;
        println!(
            "[mtp-open] layers={} hidden={} backbone={} vocab={} kv={}/{} dim={}/{} scale=[{:?}] rope_freqs_len={:?}",
            config.layer_count,
            config.hidden_size,
            config.backbone_hidden_size,
            config.vocab_size,
            config.local_num_kv_heads,
            config.global_num_kv_heads,
            config.local_head_dim,
            config.global_head_dim,
            weights.layers.iter().map(|layer| layer.layer_output_scale).collect::<Vec<_>>(),
            weights.rope_freqs.as_ref().map(Vec::len)
        );
        assert_eq!(config.layer_count, 4);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.backbone_hidden_size, 3840);
    }
    use super::*;

    /// 真实 GGUF smoke：`ZLLM_GEMMA4_GGUF=/path/to/gemma4.gguf cargo test --lib gguf_smoke -- --nocapture`。
    /// 无权重时跳过，保持 cargo test --lib 永远绿。
    #[test]
    fn gguf_smoke() {
        let Some(root) = std::env::var_os("ZLLM_GEMMA4_GGUF").map(PathBuf::from) else { return };
        let config = Gemma4Weights::select_config(&root).expect("select_config");
        println!("config: hidden={} layers={}", config.hidden_size, config.layer_count);
        let weights = Gemma4Weights::open(&root, config.clone()).expect("open");
        let layer0 = weights.load_layer(0).expect("layer 0");
        assert_eq!(layer0.attention.q_proj.rows(), config.num_heads * config.local_head_dim);
        assert_eq!(layer0.input_norm.len(), config.hidden_size);
        // 12B global 层 K=V 无独立 v_proj；E4B 始终保存独立 V。
        let layer5 = weights.load_layer(5).expect("layer 5");
        assert_eq!(layer5.attention.v_proj.is_none(), config.attention_k_eq_v);
        assert_eq!(layer5.attention.k_norm.len(), config.global_head_dim);
        let final_norm = weights.final_norm().expect("final_norm");
        assert_eq!(final_norm.len(), config.hidden_size);
        let embedding = weights.embedding_rows_f32(&[2, 106]).expect("embedding rows");
        assert_eq!(embedding.len(), 2 * config.hidden_size);
        let output = weights.load_output_weight().expect("output weight");
        let Gemma4OutputWeight::Quantized(matrix) = &output else { panic!("GGUF 输出头应是量化矩阵") };
        assert_eq!((matrix.rows(), matrix.cols()), (config.vocab_size, config.hidden_size));
        // 同目录有 mmproj-*.gguf 时多模态权重应能加载（12B unified：vision+audio projection）。
        let mm = weights.load_multimodal_weights().expect("multimodal");
        if let (Some(vision), Some(vision_cfg)) = (&mm.vision, config.vision) {
            assert_eq!(vision.projection.shape, vec![config.hidden_size, vision_cfg.embedding_size]);
            println!("mmproj vision/audio 加载成功");
        } else if let (Some(vision), Some(vision_cfg)) = (&mm.vision_encoder, config.vision) {
            assert_eq!(vision.layers.len(), vision_cfg.encoder.expect("encoder config").layer_count);
            assert_eq!(vision.projection.shape, vec![config.hidden_size, vision_cfg.embedding_size]);
            println!("Gemma4V mmproj encoder 加载成功");
        } else {
            println!("无 mmproj，纯文本模式");
        }
        println!("layer0 q_proj [{}x{}] embedding ok, output head ok", layer0.attention.q_proj.rows(), layer0.attention.q_proj.cols());
    }
}

#[cfg(test)]
mod backbone_nextn_tests {
    use super::{GgufReader, PathBuf};
    /// 检查主干 GGUF 是否携带 nextn(MTP 辅助)张量。
    /// `ZLLM_GEMMA4_GGUF=... cargo test --lib backbone_nextn_tensors -- --nocapture`
    #[test]
    fn backbone_nextn_tensors() {
        let Some(path) = std::env::var_os("ZLLM_GEMMA4_GGUF").map(PathBuf::from) else { return };
        let reader = GgufReader::open(&path).expect("open");
        let mut names: Vec<&str> = reader.tensors().iter().map(|tensor| tensor.name.as_str()).collect();
        names.sort();
        for name in names {
            if name.contains("nextn") || name.contains("mtp") || name.contains("assistant") {
                println!("[backbone-nextn] {name} dims={:?}", reader.tensor(name).unwrap().dims);
            }
        }
        println!("[backbone-nextn] 检查完毕");
    }
}
