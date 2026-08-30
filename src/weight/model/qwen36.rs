//! Qwen3.6-27B MLX affine 4-bit 文本权重映射。
//!
//! 这里只负责标准张量名、格式加载和依赖分组。层执行顺序属于
//! `runtime::qwen36`，设备驻留与 kernel 属于 backend。

use crate::model_spec::qwen36::{Qwen36AttentionKind, Qwen36Config};

use crate::weight::container::safetensor::TensorData;
use crate::weight::format::mlx_affine::MlxAffineSource;
use crate::weight::format::quantization::MlxAffineMatrix;

pub struct Qwen36DeltaNetWeights {
    pub qkv: MlxAffineMatrix,
    pub z: MlxAffineMatrix,
    pub a: MlxAffineMatrix,
    pub b: MlxAffineMatrix,
    pub conv: TensorData,
    pub a_log: TensorData,
    pub dt_bias: TensorData,
    pub norm: TensorData,
    pub output: MlxAffineMatrix,
}

pub struct Qwen36FullAttentionWeights {
    /// q_proj 的前半部分是 query，后半部分是逐元素输出门控。
    pub query_and_gate: MlxAffineMatrix,
    pub query_norm: TensorData,
    pub key: MlxAffineMatrix,
    pub key_norm: TensorData,
    pub value: MlxAffineMatrix,
    pub output: MlxAffineMatrix,
}

pub enum Qwen36TokenMixerWeights {
    GatedDeltaNet(Qwen36DeltaNetWeights),
    FullAttention(Qwen36FullAttentionWeights),
}

pub struct Qwen36MlpWeights {
    pub gate: MlxAffineMatrix,
    pub up: MlxAffineMatrix,
    pub down: MlxAffineMatrix,
}

pub struct Qwen36LayerWeights {
    pub input_norm: TensorData,
    pub token_mixer: Qwen36TokenMixerWeights,
    pub post_attention_norm: TensorData,
    pub mlp: Qwen36MlpWeights,
}

pub struct Qwen36Weights {
    source: MlxAffineSource,
    config: Qwen36Config,
}

impl Qwen36Weights {
    pub fn new(source: MlxAffineSource, config: Qwen36Config) -> Result<Self, String> {
        config.validate()?;
        let wrapper = Self { source, config };
        wrapper.validate_tensors()?;
        Ok(wrapper)
    }

    /// 校验所有 layer 与顶层张量都存在,坏 safetensors 在加载时即抛错,
    /// 而不是延迟到 kernel 阶段。轻量级 — 仅检查索引,不解码数据。
    fn validate_tensors(&self) -> Result<(), String> {
        if !self.source.has_matrix("language_model.model.embed_tokens") {
            return Err("Qwen3.6 safetensors 缺少 language_model.model.embed_tokens".to_owned());
        }
        if !self.source.has_tensor("language_model.model.norm.weight") {
            return Err("Qwen3.6 safetensors 缺少 language_model.model.norm.weight".to_owned());
        }
        if !self.source.has_matrix("language_model.lm_head") {
            return Err("Qwen3.6 safetensors 缺少 language_model.lm_head".to_owned());
        }
        for layer in 0..self.config.num_layers {
            let prefix = format!("language_model.model.layers.{layer}");
            if !self.source.has_tensor(&format!("{prefix}.input_layernorm.weight")) || !self.source.has_tensor(&format!("{prefix}.post_attention_layernorm.weight")) {
                return Err(format!("Qwen3.6 layer {layer} 缺少 input/post_attention layernorm"));
            }
            if !self.source.has_matrix(&format!("{prefix}.mlp.gate_proj")) || !self.source.has_matrix(&format!("{prefix}.mlp.up_proj")) || !self.source.has_matrix(&format!("{prefix}.mlp.down_proj")) {
                return Err(format!("Qwen3.6 layer {layer} 缺少 mlp gate/up/down"));
            }
            match self.layer_kind(layer).expect("layer 已 bounds-check") {
                Qwen36AttentionKind::FullAttention => {
                    for matrix in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] {
                        if !self.source.has_matrix(&format!("{prefix}.{matrix}")) {
                            return Err(format!("Qwen3.6 layer {layer} FullAttention 缺少 {matrix}"));
                        }
                    }
                    for tensor in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                        if !self.source.has_tensor(&format!("{prefix}.{tensor}")) {
                            return Err(format!("Qwen3.6 layer {layer} FullAttention 缺少 {tensor}"));
                        }
                    }
                }
                Qwen36AttentionKind::GatedDeltaNet => {
                    for matrix in ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.out_proj"] {
                        if !self.source.has_matrix(&format!("{prefix}.{matrix}")) {
                            return Err(format!("Qwen3.6 layer {layer} GatedDeltaNet 缺少 {matrix}"));
                        }
                    }
                    for tensor in ["linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.norm.weight"] {
                        if !self.source.has_tensor(&format!("{prefix}.{tensor}")) {
                            return Err(format!("Qwen3.6 layer {layer} GatedDeltaNet 缺少 {tensor}"));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn source(&self) -> &MlxAffineSource {
        &self.source
    }

    pub fn config(&self) -> &Qwen36Config {
        &self.config
    }

    pub fn embedding(&self) -> Result<MlxAffineMatrix, String> {
        self.matrix("language_model.model.embed_tokens")
    }

    /// 加载指定 token 行的嵌入并解量化为 F32(行主序 `[len, hidden]`)。
    /// 单 token 预填时仅解对应几行,避免解 248k × 5120 的完整表。
    pub fn embedding_rows_f32(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let rows = tokens
            .iter()
            .map(|&token| {
                let row = token as usize;
                (row < self.config.vocab_size).then_some(row).ok_or_else(|| format!("Qwen3.6 token {row} 超出 vocab {}", self.config.vocab_size))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let matrix = self.source.load_matrix_rows("language_model.model.embed_tokens", &rows).map_err(|error| format!("加载 Qwen3.6 embed_tokens 行失败: {error}"))?;
        if matrix.rows != rows.len() || matrix.cols != self.config.hidden_size {
            return Err(format!("Qwen3.6 embed_tokens 形状 [{},{}] 与期望 [{},{}] 不一致", matrix.rows, matrix.cols, rows.len(), self.config.hidden_size));
        }
        matrix.decode()
    }

    pub fn final_norm(&self) -> Result<TensorData, String> {
        self.tensor("language_model.model.norm.weight")
    }

    pub fn lm_head(&self) -> Result<MlxAffineMatrix, String> {
        self.matrix("language_model.lm_head")
    }

    pub fn layer(&self, layer: usize) -> Result<Qwen36LayerWeights, String> {
        let kind = self.layer_kind(layer)?;
        let prefix = format!("language_model.model.layers.{layer}");
        let token_mixer = match kind {
            Qwen36AttentionKind::GatedDeltaNet => Qwen36TokenMixerWeights::GatedDeltaNet(self.delta_net(&prefix)?),
            Qwen36AttentionKind::FullAttention => Qwen36TokenMixerWeights::FullAttention(self.full_attention(&prefix)?),
        };

        Ok(Qwen36LayerWeights {
            input_norm: self.tensor(&format!("{prefix}.input_layernorm.weight"))?,
            token_mixer,
            post_attention_norm: self.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            mlp: Qwen36MlpWeights { gate: self.matrix(&format!("{prefix}.mlp.gate_proj"))?, up: self.matrix(&format!("{prefix}.mlp.up_proj"))?, down: self.matrix(&format!("{prefix}.mlp.down_proj"))? },
        })
    }

    pub fn layer_kind(&self, layer: usize) -> Result<Qwen36AttentionKind, String> {
        if layer >= self.config.num_layers {
            return Err(format!("Qwen3.6 layer {} 越界，总层数 {}", layer, self.config.num_layers));
        }
        if (layer + 1).is_multiple_of(self.config.full_attention_interval) { Ok(Qwen36AttentionKind::FullAttention) } else { Ok(Qwen36AttentionKind::GatedDeltaNet) }
    }

    fn delta_net(&self, prefix: &str) -> Result<Qwen36DeltaNetWeights, String> {
        let attention = format!("{prefix}.linear_attn");
        Ok(Qwen36DeltaNetWeights {
            qkv: self.matrix(&format!("{attention}.in_proj_qkv"))?,
            z: self.matrix(&format!("{attention}.in_proj_z"))?,
            a: self.matrix(&format!("{attention}.in_proj_a"))?,
            b: self.matrix(&format!("{attention}.in_proj_b"))?,
            conv: self.tensor(&format!("{attention}.conv1d.weight"))?,
            a_log: self.tensor(&format!("{attention}.A_log"))?,
            dt_bias: self.tensor(&format!("{attention}.dt_bias"))?,
            norm: self.tensor(&format!("{attention}.norm.weight"))?,
            output: self.matrix(&format!("{attention}.out_proj"))?,
        })
    }

    fn full_attention(&self, prefix: &str) -> Result<Qwen36FullAttentionWeights, String> {
        let attention = format!("{prefix}.self_attn");
        Ok(Qwen36FullAttentionWeights {
            query_and_gate: self.matrix(&format!("{attention}.q_proj"))?,
            query_norm: self.tensor(&format!("{attention}.q_norm.weight"))?,
            key: self.matrix(&format!("{attention}.k_proj"))?,
            key_norm: self.tensor(&format!("{attention}.k_norm.weight"))?,
            value: self.matrix(&format!("{attention}.v_proj"))?,
            output: self.matrix(&format!("{attention}.o_proj"))?,
        })
    }

    fn matrix(&self, name: &str) -> Result<MlxAffineMatrix, String> {
        self.source.load_matrix(name).map_err(|error| format!("加载 Qwen3.6 affine matrix {name} 失败: {error}"))
    }

    fn tensor(&self, name: &str) -> Result<TensorData, String> {
        self.source.load_tensor(name).map_err(|error| format!("加载 Qwen3.6 tensor {name} 失败: {error}"))
    }
}
