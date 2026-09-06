//! K2-Horizon MoVA MoE 模型规格与 GGUF 权重映射。

#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
pub mod metal_replay;
pub mod node;

use crate::{
    attention::{
        AttentionSpec,
        gqa::{CausalWindow, GqaSpec},
    },
    moe::{
        Activation, FeedforwardSpec,
        dense_mlp::DenseMlpSpec,
        topk_moe::{ScoringFunc, TopkMoeSpec},
    },
    norm::NormSpec,
    runtime::{LayerId, LayerSpec, Model, ModelError},
    tokenizer::{Detokenizer, Tokenizer},
    weight::{
        container::gguf::{GgufMatrix, GgufReader, GgufValue},
        expert_source::{ExpertSource, ExpertSourceProvider, GgufExpertSource, GgufExpertWeights},
    },
};
use std::path::Path;

pub use crate::model_spec::k2_horizon::K2HorizonConfig;

impl K2HorizonConfig {
    pub fn gqa_spec(&self) -> GqaSpec {
        GqaSpec {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            rope_theta: self.rope_theta,
            use_qk_norm: false,
            window: CausalWindow::Full,
            score_scale: 1.0 / (self.head_dim as f32).sqrt(),
            output_gate: true,
        }
    }

    pub fn dense_spec(&self) -> DenseMlpSpec {
        DenseMlpSpec { intermediate_size: self.dense_intermediate_size, activation: Activation::Silu }
    }

    pub fn moe_spec(&self) -> TopkMoeSpec {
        TopkMoeSpec {
            num_experts: self.expert_count,
            top_k: self.expert_top_k,
            num_shared_experts: 1,
            scoring_func: ScoringFunc::SigmoidBias,
            normalize_selected: true,
            routed_scaling_factor: self.routed_scaling_factor,
            intermediate_size: self.expert_intermediate_size,
            shared_intermediate_size: self.shared_intermediate_size,
            activation: Activation::Silu,
        }
    }

    pub fn value_moe_spec(&self) -> TopkMoeSpec {
        TopkMoeSpec {
            num_experts: self.value_expert_count,
            top_k: self.value_expert_top_k,
            num_shared_experts: 0,
            scoring_func: ScoringFunc::SigmoidBias,
            normalize_selected: true,
            routed_scaling_factor: self.routed_scaling_factor,
            intermediate_size: self.num_kv_heads * self.head_dim,
            shared_intermediate_size: 0,
            activation: Activation::Silu,
        }
    }

    pub fn from_gguf(reader: &GgufReader) -> Result<Self, String> {
        let integer = |suffix: &str| reader.metadata_u64(&format!("k2-horizon.{suffix}")).map(|value| value as usize);
        let float = |suffix: &str| {
            let key = format!("k2-horizon.{suffix}");
            reader.metadata(&key).and_then(GgufValue::as_f64).map(|value| value as f32).ok_or_else(|| format!("GGUF metadata {key} 缺失或类型错误"))
        };
        let bool_value = |suffix: &str| {
            let key = format!("k2-horizon.{suffix}");
            match reader.metadata(&key) {
                Some(GgufValue::Bool(value)) => Ok(*value),
                _ => Err(format!("GGUF metadata {key} 缺失或类型错误")),
            }
        };
        if reader.metadata_u64("k2-horizon.expert_gating_func")? != 2 || !bool_value("expert_weights_norm")? {
            return Err("K2-Horizon 当前要求 sigmoid 路由且归一化选中专家权重".to_owned());
        }
        if reader.metadata_u64("k2-horizon.expert_shared_count")? != 1 || reader.metadata_u64("k2-horizon.moe_every_n_layers")? != 1 {
            return Err("K2-Horizon 当前要求一个 shared expert 且 dense 前缀后每层均为 MoE".to_owned());
        }
        let head_dim = integer("attention.key_length")?;
        if integer("attention.value_length")? != head_dim {
            return Err("K2-Horizon attention key/value head dim 不一致".to_owned());
        }
        let embedding = reader.tensor("token_embd.weight").ok_or("K2-Horizon GGUF 缺少 token_embd.weight")?;
        if embedding.dims.len() != 2 {
            return Err(format!("K2-Horizon token_embd.weight shape={:?} 不是矩阵", embedding.dims));
        }
        let config = Self {
            vocab_size: embedding.dims[1],
            hidden_size: integer("embedding_length")?,
            layer_count: integer("block_count")?,
            leading_dense_layer_count: integer("leading_dense_block_count")?,
            num_heads: integer("attention.head_count")?,
            num_kv_heads: integer("attention.head_count_kv")?,
            head_dim,
            max_position_embeddings: integer("context_length")?,
            rope_dim: integer("rope.dimension_count")?,
            rope_theta: float("rope.freq_base")?,
            rms_eps: float("attention.layer_norm_rms_epsilon")?,
            norm_groups: integer("attention.group_norm_groups")?,
            dense_intermediate_size: integer("feed_forward_length")?,
            expert_count: integer("expert_count")?,
            expert_top_k: integer("expert_used_count")?,
            expert_intermediate_size: integer("expert_feed_forward_length")?,
            shared_intermediate_size: integer("expert_shared_feed_forward_length")?,
            routed_scaling_factor: float("expert_weights_scale")?,
            value_expert_count: integer("attention.value_expert_count")?,
            value_expert_top_k: integer("attention.value_expert_used_count")?,
            bos_token_id: reader.metadata_u64("tokenizer.ggml.bos_token_id")? as u32,
            eos_token_id: reader.metadata_u64("tokenizer.ggml.eos_token_id")? as u32,
        };
        ensure_supported(&config)?;
        Ok(config)
    }
}

pub struct K2Horizon {
    config: K2HorizonConfig,
    layers: Vec<LayerSpec>,
}

impl K2Horizon {
    pub fn new(config: K2HorizonConfig) -> Result<Self, ModelError> {
        ensure_supported(&config).map_err(ModelError::InvalidArchitecture)?;
        let norm = || NormSpec::GroupedRms { eps: config.rms_eps, groups: config.norm_groups };
        let layers = (0..config.layer_count)
            .map(|layer| LayerSpec {
                attention: AttentionSpec::Gqa(config.gqa_spec()),
                feedforward: if layer < config.leading_dense_layer_count { FeedforwardSpec::Dense(config.dense_spec()) } else { FeedforwardSpec::TopkMoe(config.moe_spec()) },
                input_norm: norm(),
                post_attention_norm: norm(),
                post_norm: None,
            })
            .collect();
        Ok(Self { config, layers })
    }
}

impl Model for K2Horizon {
    type Config = K2HorizonConfig;
    fn config(&self) -> &Self::Config {
        &self.config
    }
    fn layer_count(&self) -> usize {
        self.config.layer_count
    }
    fn layer_spec(&self, layer: LayerId) -> Result<&LayerSpec, ModelError> {
        self.layers.get(layer).ok_or(ModelError::LayerOutOfRange { layer, layer_count: self.config.layer_count })
    }
}

pub fn ensure_supported(config: &K2HorizonConfig) -> Result<(), String> {
    if config.layer_count == 0 || config.leading_dense_layer_count == 0 || config.leading_dense_layer_count >= config.layer_count {
        return Err(format!("K2-Horizon layer_count={} leading_dense={} 非法", config.layer_count, config.leading_dense_layer_count));
    }
    if config.hidden_size == 0 || config.norm_groups == 0 || !config.hidden_size.is_multiple_of(config.norm_groups) || config.rms_eps <= 0.0 {
        return Err("K2-Horizon hidden/grouped RMSNorm 参数非法".to_owned());
    }
    config.gqa_spec().geometry().validate().map_err(|error| format!("K2-Horizon GQA: {error}"))?;
    if config.rope_dim == 0 || config.rope_dim > config.head_dim || config.expert_count == 0 || config.expert_top_k == 0 || config.expert_top_k > config.expert_count {
        return Err("K2-Horizon RoPE/FFN expert 参数非法".to_owned());
    }
    if config.value_expert_count == 0 || config.value_expert_top_k == 0 || config.value_expert_top_k > config.value_expert_count || config.routed_scaling_factor <= 0.0 {
        return Err("K2-Horizon value expert 参数非法".to_owned());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum K2ExpertProjection {
    Gate,
    Up,
    Down,
}

pub struct K2HorizonGguf {
    reader: GgufReader,
    config: K2HorizonConfig,
}

impl K2HorizonGguf {
    pub fn open(path: &Path) -> Result<Self, String> {
        let reader = GgufReader::open(&GgufReader::locate(path)?)?;
        reader.expect_metadata_str("general.architecture", "k2-horizon")?;
        let config = K2HorizonConfig::from_gguf(&reader)?;
        Ok(Self { reader, config })
    }
    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }
    pub fn config(&self) -> &K2HorizonConfig {
        &self.config
    }
    pub fn embedding_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.reader.embedding_rows("token_embd.weight", tokens, self.config.hidden_size, self.config.vocab_size)
    }
    pub fn final_norm(&self) -> Result<Vec<f32>, String> {
        self.reader.read_tensor_f32("output_norm.weight")
    }
    pub fn output_head(&self) -> Result<GgufMatrix, String> {
        self.reader.read_matrix("output.weight")
    }
    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        self.reader.bpe_tokenizer().map_err(|error| format!("K2-Horizon tokenizer: {error}"))
    }
    pub fn detokenizer(&self) -> Result<Detokenizer, String> {
        self.reader.bpe_detokenizer().map_err(|error| format!("K2-Horizon detokenizer: {error}"))
    }
    pub fn routed_expert(&self, layer: usize, expert: usize, projection: K2ExpertProjection) -> Result<GgufMatrix, String> {
        if layer < self.config.leading_dense_layer_count || layer >= self.config.layer_count || expert >= self.config.expert_count {
            return Err(format!("K2-Horizon expert 索引越界: layer={layer} expert={expert}"));
        }
        let projection = match projection {
            K2ExpertProjection::Gate => "gate",
            K2ExpertProjection::Up => "up",
            K2ExpertProjection::Down => "down",
        };
        self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_{projection}_exps.weight"), expert)
    }
}

impl GgufExpertSource for K2HorizonGguf {
    fn intermediate(&self) -> usize {
        self.config.expert_intermediate_size
    }
    fn hidden(&self) -> usize {
        self.config.hidden_size
    }
    fn load_expert_gguf(&self, layer: usize, expert: usize) -> Result<GgufExpertWeights, String> {
        Ok(GgufExpertWeights { gate: self.routed_expert(layer, expert, K2ExpertProjection::Gate)?, up: self.routed_expert(layer, expert, K2ExpertProjection::Up)?, down: self.routed_expert(layer, expert, K2ExpertProjection::Down)? })
    }
}

impl ExpertSourceProvider for K2HorizonGguf {
    fn source(&self, layer: usize) -> Result<ExpertSource<'_>, String> {
        if layer < self.config.leading_dense_layer_count || layer >= self.config.layer_count {
            return Err(format!("K2-Horizon L{layer} 没有 routed FFN experts"));
        }
        Ok(ExpertSource::Gguf(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_schedule_and_specs() {
        let config = K2HorizonConfig::standard_36b_a4b();
        assert!(K2Horizon::new(config).is_ok());
        assert_eq!(config.leading_dense_layer_count, 3);
        assert_eq!(config.moe_spec().scoring_func, ScoringFunc::SigmoidBias);
        assert_eq!(config.value_moe_spec().top_k, 4);
    }
}
