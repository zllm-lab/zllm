//! 经典 dense Qwen3(`Qwen3ForCausalLM`)ModelOpt NVFP4 权重映射。
//!
//! 只做张量命名、存在性与 shape 校验(nvidia/Qwen3-14B-NVFP4 的 HF safetensors 布局):
//! Linear 为 NVFP4(`.weight` U8 packed + `.weight_scale` F8_E4M3 block + `.weight_scale_2`
//! F32 全局 scale,见 `weight::format::nvfp4::load_modelopt_matrix`);norm/q_norm/k_norm/
//! embed_tokens/lm_head 为 BF16。LM head 独立(tie_word_embeddings=false)。

use std::path::Path;

use crate::model_spec::qwen3_vl::Qwen3VlConfig;

use super::{
    super::container::safetensor::{SafetensorStore, TensorData},
    super::format::nvfp4::load_modelopt_matrix,
    qwen3_vl::{Qwen3TextSource, Qwen3VlLayerWeights, Qwen3VlMatrix},
};

pub struct Qwen3Weights {
    store: SafetensorStore,
    config: Qwen3VlConfig,
}

impl Qwen3Weights {
    pub fn open(root: impl AsRef<Path>, config: Qwen3VlConfig) -> Result<Self, String> {
        config.validate()?;
        if config.vision.is_some() {
            return Err("经典 Qwen3 是纯文本模型，config 不应携带 vision 规格".to_owned());
        }
        let store = SafetensorStore::open(root)?;
        let weights = Self { store, config };
        weights.validate_tensors()?;
        Ok(weights)
    }

    /// 全量枚举校验张量存在性(header 级,不读数据);shape 强校验在 `load_layer` 完成。
    fn validate_tensors(&self) -> Result<(), String> {
        for layer in 0..self.config.layer_count {
            let prefix = format!("model.layers.{layer}");
            for name in [format!("{prefix}.input_layernorm.weight"), format!("{prefix}.post_attention_layernorm.weight"), format!("{prefix}.self_attn.q_norm.weight"), format!("{prefix}.self_attn.k_norm.weight")] {
                if !self.store.has(&name) {
                    return Err(format!("Qwen3-14B 权重缺少 {name}"));
                }
            }
            for base in [
                format!("{prefix}.self_attn.q_proj"),
                format!("{prefix}.self_attn.k_proj"),
                format!("{prefix}.self_attn.v_proj"),
                format!("{prefix}.self_attn.o_proj"),
                format!("{prefix}.mlp.gate_proj"),
                format!("{prefix}.mlp.up_proj"),
                format!("{prefix}.mlp.down_proj"),
            ] {
                for suffix in ["weight", "weight_scale", "weight_scale_2"] {
                    let name = format!("{base}.{suffix}");
                    if !self.store.has(&name) {
                        return Err(format!("Qwen3-14B 权重缺少 {name}"));
                    }
                }
            }
        }
        for name in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"] {
            if !self.store.has(name) {
                return Err(format!("Qwen3-14B 权重缺少 {name}"));
            }
        }
        Ok(())
    }

    pub fn embedding_rows_f32(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let rows = tokens
            .iter()
            .map(|&token| {
                let row = token as usize;
                (row < self.config.vocab_size).then_some(row).ok_or_else(|| format!("Qwen3-14B token {row} 超出 vocab {}", self.config.vocab_size))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tensor = self.store.load_bf16_rows("model.embed_tokens.weight", &rows)?;
        tensor.expect_shape(&[rows.len(), self.config.hidden_size])?;
        tensor.to_f32()
    }

    fn load_layer(&self, layer: usize) -> Result<Qwen3VlLayerWeights, String> {
        if layer >= self.config.layer_count {
            return Err(format!("Qwen3-14B layer {layer} 越界，共 {} 层", self.config.layer_count));
        }
        let prefix = format!("model.layers.{layer}");
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        let query_columns = self.config.num_heads * self.config.head_dim;
        let kv_columns = self.config.num_kv_heads * self.config.head_dim;
        Ok(Qwen3VlLayerWeights {
            input_norm: self.load_vector(&format!("{prefix}.input_layernorm.weight"), self.config.hidden_size)?,
            query: self.load_matrix(&format!("{attention}.q_proj"), query_columns, self.config.hidden_size)?,
            query_norm: self.load_vector(&format!("{attention}.q_norm.weight"), self.config.head_dim)?,
            key: self.load_matrix(&format!("{attention}.k_proj"), kv_columns, self.config.hidden_size)?,
            key_norm: self.load_vector(&format!("{attention}.k_norm.weight"), self.config.head_dim)?,
            value: self.load_matrix(&format!("{attention}.v_proj"), kv_columns, self.config.hidden_size)?,
            output: self.load_matrix(&format!("{attention}.o_proj"), self.config.hidden_size, query_columns)?,
            post_attention_norm: self.load_vector(&format!("{prefix}.post_attention_layernorm.weight"), self.config.hidden_size)?,
            gate: self.load_matrix(&format!("{mlp}.gate_proj"), self.config.intermediate_size, self.config.hidden_size)?,
            up: self.load_matrix(&format!("{mlp}.up_proj"), self.config.intermediate_size, self.config.hidden_size)?,
            down: self.load_matrix(&format!("{mlp}.down_proj"), self.config.hidden_size, self.config.intermediate_size)?,
        })
    }

    fn load_matrix(&self, base: &str, rows: usize, cols: usize) -> Result<Qwen3VlMatrix, String> {
        let matrix = load_modelopt_matrix(&self.store, base, rows, cols)?;
        if matrix.rows != rows || matrix.cols != cols {
            return Err(format!("Qwen3-14B matrix {base} shape=[{},{}]，期望 [{rows},{cols}]", matrix.rows, matrix.cols));
        }
        Ok(Qwen3VlMatrix::Nvfp4(matrix))
    }

    fn load_vector(&self, name: &str, size: usize) -> Result<Vec<f32>, String> {
        let tensor = self.store.load(name)?;
        tensor.expect_shape(&[size])?;
        tensor.to_f32()
    }
}

impl Qwen3TextSource for Qwen3Weights {
    fn text_layer(&self, layer: usize) -> Result<Qwen3VlLayerWeights, String> {
        self.load_layer(layer)
    }

    fn final_norm(&self) -> Result<Vec<f32>, String> {
        self.load_vector("model.norm.weight", self.config.hidden_size)
    }

    fn lm_head(&self) -> Result<TensorData, String> {
        let tensor = self.store.load("lm_head.weight")?;
        tensor.expect_shape(&[self.config.vocab_size, self.config.hidden_size])?;
        Ok(tensor)
    }
}
