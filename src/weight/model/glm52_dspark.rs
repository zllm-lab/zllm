//! GLM-5.2 DSpark drafter checkpoint；执行规格由 `speculative` 定义。

use crate::speculative::{BlockDraftSpec, HiddenStateCapturePlan};
use crate::weight::container::safetensor::{SafetensorStore, TensorData};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct Glm52DsparkConfig {
    pub aux_hidden_state_layer_ids: Vec<usize>,
    pub block_size: usize,
    pub confidence_head_with_markov: bool,
    pub draft_vocab_size: usize,
    pub enable_confidence_head: bool,
    pub markov_head_type: String,
    pub markov_rank: usize,
    pub mask_token_id: u32,
    #[serde(default)]
    pub sliding_window_non_causal: bool,
    pub speculators_config: SpeculatorsConfig,
    pub transformer_layer_config: DraftTransformerConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpeculatorsConfig {
    pub proposal_methods: Vec<ProposalMethod>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProposalMethod {
    pub speculative_tokens: usize,
    pub verifier_accept_k: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DraftTransformerConfig {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: DraftRopeConfig,
    pub sliding_window: Option<usize>,
    pub vocab_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DraftRopeConfig {
    pub rope_theta: f32,
    pub rope_type: String,
}

impl Glm52DsparkConfig {
    pub fn read(root: &Path) -> Result<Self, String> {
        let path = root.join("config.json");
        let config: Self = serde_json::from_slice(&std::fs::read(&path).map_err(|error| format!("读取 {} 失败: {error}", path.display()))?).map_err(|error| format!("解析 {} 失败: {error}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn block_spec(&self) -> Result<BlockDraftSpec, String> {
        let proposal = self.speculators_config.proposal_methods.first().ok_or("DSpark config 缺少 proposal_methods")?;
        BlockDraftSpec::new(self.block_size, proposal.speculative_tokens, proposal.verifier_accept_k).map_err(|error| error.to_string())
    }

    pub fn capture_plan(&self, verifier_layer_count: usize) -> Result<HiddenStateCapturePlan, String> {
        let boundaries = self.aux_hidden_state_layer_ids.iter().map(|layer| layer.checked_add(1).ok_or("DSpark target layer boundary 溢出")).collect::<Result<Vec<_>, _>>()?;
        HiddenStateCapturePlan::new(boundaries, verifier_layer_count).map_err(|error| error.to_string())
    }

    fn validate(&self) -> Result<(), String> {
        let transformer = &self.transformer_layer_config;
        if self.aux_hidden_state_layer_ids.is_empty() {
            return Err("DSpark aux_hidden_state_layer_ids 不能为空".to_owned());
        }
        if self.aux_hidden_state_layer_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("DSpark aux_hidden_state_layer_ids 必须严格递增".to_owned());
        }
        if transformer.hidden_size == 0 || transformer.intermediate_size == 0 || transformer.head_dim == 0 {
            return Err("DSpark transformer 维度必须大于 0".to_owned());
        }
        if transformer.num_attention_heads == 0 || transformer.num_key_value_heads == 0 || transformer.num_hidden_layers != transformer.layer_types.len() {
            return Err("DSpark layer_types 数量或 KV head 数非法".to_owned());
        }
        if transformer.layer_types.iter().any(|kind| kind != "full_attention" && kind != "sliding_attention")
            || transformer.layer_types.iter().any(|kind| kind == "sliding_attention") && !transformer.sliding_window.is_some_and(|window| window > 0)
        {
            return Err(format!("DSpark attention layer_types/window 非法: {:?}/{:?}", transformer.layer_types, transformer.sliding_window));
        }
        if transformer.rope_parameters.rope_type != "default" || !transformer.rope_parameters.rope_theta.is_finite() || transformer.rope_parameters.rope_theta <= 0.0 {
            return Err("当前 DSpark checkpoint 仅支持合法的 default RoPE".to_owned());
        }
        if self.draft_vocab_size != transformer.vocab_size || usize::try_from(self.mask_token_id).unwrap_or(usize::MAX) >= self.draft_vocab_size {
            return Err("DSpark draft vocab 或 mask token 非法".to_owned());
        }
        if self.markov_rank == 0 || self.markov_head_type != "vanilla" {
            return Err(format!("当前 DSpark checkpoint 仅支持 vanilla Markov head，实际 type={} rank={}", self.markov_head_type, self.markov_rank));
        }
        if !self.enable_confidence_head || !self.confidence_head_with_markov {
            return Err("当前 DSpark checkpoint 要求启用含 Markov embedding 的 confidence head".to_owned());
        }
        self.block_spec()?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct Glm52DsparkCheckpoint {
    store: SafetensorStore,
    pub config: Glm52DsparkConfig,
}

pub struct DsparkLayerWeights {
    pub input_norm: TensorData,
    pub q_proj: TensorData,
    pub q_norm: TensorData,
    pub k_proj: TensorData,
    pub k_norm: TensorData,
    pub v_proj: TensorData,
    pub o_proj: TensorData,
    pub post_attention_norm: TensorData,
    pub gate_proj: TensorData,
    pub up_proj: TensorData,
    pub down_proj: TensorData,
}

pub struct DsparkTargetLayerWeights {
    pub k_proj: TensorData,
    pub k_norm: TensorData,
    pub v_proj: TensorData,
}

impl Glm52DsparkCheckpoint {
    pub fn open(root: &Path) -> Result<Self, String> {
        let config = Glm52DsparkConfig::read(root)?;
        let checkpoint = Self { store: SafetensorStore::open(root)?, config };
        checkpoint.validate_tensor_set()?;
        Ok(checkpoint)
    }

    pub fn load(&self, name: &str, shape: &[usize]) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        tensor.expect_shape(shape)?;
        Ok(tensor)
    }

    pub fn embedding_rows(&self, token_ids: &[u32]) -> Result<TensorData, String> {
        let rows = token_ids.iter().map(|&token| token as usize).collect::<Vec<_>>();
        self.store.load_bf16_rows("embed_tokens.weight", &rows)
    }

    pub fn markov_rows(&self, token_ids: &[u32]) -> Result<TensorData, String> {
        let rows = token_ids.iter().map(|&token| token as usize).collect::<Vec<_>>();
        self.store.load_bf16_rows("markov_head.markov_w1.weight", &rows)
    }

    pub fn confidence_head(&self) -> Result<(TensorData, TensorData), String> {
        let columns = self.config.transformer_layer_config.hidden_size.checked_add(self.config.markov_rank).ok_or("DSpark confidence head 列数溢出")?;
        Ok((self.load("confidence_head.proj.weight", &[1, columns])?, self.load("confidence_head.proj.bias", &[1])?))
    }

    pub fn aux_projection(&self, capture_index: usize) -> Result<TensorData, String> {
        let hidden = self.config.transformer_layer_config.hidden_size;
        if capture_index >= self.config.aux_hidden_state_layer_ids.len() {
            return Err(format!("DSpark aux projection index={capture_index} 越界"));
        }
        let start = capture_index.checked_mul(hidden).ok_or("DSpark aux projection 列偏移溢出")?;
        self.store.load_columns("fc.weight", start..start + hidden)
    }

    pub fn load_layer(&self, layer: usize) -> Result<DsparkLayerWeights, String> {
        let cfg = &self.config.transformer_layer_config;
        if layer >= cfg.num_hidden_layers {
            return Err(format!("DSpark layer {layer} 越界，总层数 {}", cfg.num_hidden_layers));
        }
        let h = cfg.hidden_size;
        let q = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let i = cfg.intermediate_size;
        let prefix = format!("layers.{layer}");
        Ok(DsparkLayerWeights {
            input_norm: self.load(&format!("{prefix}.input_layernorm.weight"), &[h])?,
            q_proj: self.load(&format!("{prefix}.self_attn.q_proj.weight"), &[q, h])?,
            q_norm: self.load(&format!("{prefix}.self_attn.q_norm.weight"), &[cfg.head_dim])?,
            k_proj: self.load(&format!("{prefix}.self_attn.k_proj.weight"), &[kv, h])?,
            k_norm: self.load(&format!("{prefix}.self_attn.k_norm.weight"), &[cfg.head_dim])?,
            v_proj: self.load(&format!("{prefix}.self_attn.v_proj.weight"), &[kv, h])?,
            o_proj: self.load(&format!("{prefix}.self_attn.o_proj.weight"), &[h, q])?,
            post_attention_norm: self.load(&format!("{prefix}.post_attention_layernorm.weight"), &[h])?,
            gate_proj: self.load(&format!("{prefix}.mlp.gate_proj.weight"), &[i, h])?,
            up_proj: self.load(&format!("{prefix}.mlp.up_proj.weight"), &[i, h])?,
            down_proj: self.load(&format!("{prefix}.mlp.down_proj.weight"), &[h, i])?,
        })
    }

    /// GPU 只维护 CPU drafter 的 target K/V 时只读取必要张量，避免启动阶段
    /// 把 Q/O/FFN 等数 GB proposal 权重从 SSD 搬过一遍再立即丢弃。
    pub fn load_target_layer(&self, layer: usize) -> Result<DsparkTargetLayerWeights, String> {
        let cfg = &self.config.transformer_layer_config;
        if layer >= cfg.num_hidden_layers {
            return Err(format!("DSpark layer {layer} 越界，总层数 {}", cfg.num_hidden_layers));
        }
        let h = cfg.hidden_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let prefix = format!("layers.{layer}.self_attn");
        Ok(DsparkTargetLayerWeights {
            k_proj: self.load(&format!("{prefix}.k_proj.weight"), &[kv, h])?,
            k_norm: self.load(&format!("{prefix}.k_norm.weight"), &[cfg.head_dim])?,
            v_proj: self.load(&format!("{prefix}.v_proj.weight"), &[kv, h])?,
        })
    }

    fn validate_tensor_set(&self) -> Result<(), String> {
        let cfg = &self.config;
        let transformer = &cfg.transformer_layer_config;
        let h = transformer.hidden_size;
        let v = cfg.draft_vocab_size;
        let r = cfg.markov_rank;
        let aux = cfg.aux_hidden_state_layer_ids.len().checked_mul(h).ok_or("DSpark aux hidden 维度溢出")?;
        for (name, shape) in [
            ("embed_tokens.weight", vec![v, h]),
            ("lm_head.weight", vec![v, h]),
            ("fc.weight", vec![h, aux]),
            ("hidden_norm.weight", vec![h]),
            ("norm.weight", vec![h]),
            ("markov_head.markov_w1.weight", vec![v, r]),
            ("markov_head.markov_w2.weight", vec![v, r]),
            ("confidence_head.proj.weight", vec![1, h + r]),
            ("confidence_head.proj.bias", vec![1]),
        ] {
            let tensor = self.store.tensor_info(name)?;
            if tensor.dtype != "BF16" {
                return Err(format!("{} dtype={}，期望 BF16", tensor.name, tensor.dtype));
            }
            if tensor.shape != shape {
                return Err(format!("{} shape {:?}，期望 {shape:?}", tensor.name, tensor.shape));
            }
        }
        let q = transformer.num_attention_heads * transformer.head_dim;
        let kv = transformer.num_key_value_heads * transformer.head_dim;
        let i = transformer.intermediate_size;
        for layer in 0..transformer.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            for (suffix, shape) in [
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![q, h]),
                ("self_attn.q_norm.weight", vec![transformer.head_dim]),
                ("self_attn.k_proj.weight", vec![kv, h]),
                ("self_attn.k_norm.weight", vec![transformer.head_dim]),
                ("self_attn.v_proj.weight", vec![kv, h]),
                ("self_attn.o_proj.weight", vec![h, q]),
                ("mlp.gate_proj.weight", vec![i, h]),
                ("mlp.up_proj.weight", vec![i, h]),
                ("mlp.down_proj.weight", vec![h, i]),
            ] {
                let name = format!("{prefix}.{suffix}");
                let tensor = self.store.tensor_info(&name)?;
                if tensor.dtype != "BF16" {
                    return Err(format!("{} dtype={}，期望 BF16", tensor.name, tensor.dtype));
                }
                if tensor.shape != shape {
                    return Err(format!("{} shape {:?}，期望 {shape:?}", tensor.name, tensor.shape));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_official_shape_config() {
        let config: Glm52DsparkConfig = serde_json::from_str(
            r#"{
            "aux_hidden_state_layer_ids":[8,23,39,55,70], "block_size":8,
            "confidence_head_with_markov":true, "draft_vocab_size":154880,
            "enable_confidence_head":true, "markov_head_type":"vanilla", "markov_rank":256,
            "mask_token_id":154856, "sliding_window_non_causal":false,
            "speculators_config":{"proposal_methods":[{"speculative_tokens":7,"verifier_accept_k":1}]},
            "transformer_layer_config":{"attention_bias":false,"head_dim":64,"hidden_size":6144,
              "intermediate_size":12288,"layer_types":["full_attention","full_attention","full_attention","full_attention","full_attention"],
              "num_attention_heads":64,"num_hidden_layers":5,"num_key_value_heads":64,"rms_norm_eps":0.00001,
              "rope_parameters":{"rope_theta":8000000.0,"rope_type":"default"},"sliding_window":null,"vocab_size":154880}
        }"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.block_spec().unwrap(), BlockDraftSpec { block_size: 8, speculative_tokens: 7, verifier_accept_k: 1 });
        assert_eq!(config.capture_plan(78).unwrap().boundaries(), [9, 24, 40, 56, 71]);
    }
}
