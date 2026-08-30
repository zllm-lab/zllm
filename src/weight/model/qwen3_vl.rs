//! Qwen3-VL compressed-tensors 与视觉塔权重映射。

use std::path::Path;

use crate::model_spec::qwen3_vl::{Qwen3VlConfig, Qwen3VlVisionConfig};

use super::{
    super::container::safetensor::TensorData,
    super::format::compressed_tensors::W4A16CtSource,
    super::format::nvfp4::Nvfp4Matrix,
    super::format::quantization::W4A16Matrix,
    vision::{LayerNormWeights, LinearWeights},
};

pub enum Qwen3VlMatrix {
    Quantized(W4A16Matrix),
    Nvfp4(Nvfp4Matrix),
    Dense(TensorData),
}

impl Qwen3VlMatrix {
    pub fn rows(&self) -> usize {
        match self {
            Self::Quantized(matrix) => matrix.rows,
            Self::Nvfp4(matrix) => matrix.rows,
            Self::Dense(tensor) => tensor.shape[0],
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Quantized(matrix) => matrix.cols,
            Self::Nvfp4(matrix) => matrix.cols,
            Self::Dense(tensor) => tensor.shape[1],
        }
    }
}

pub struct Qwen3VlLayerWeights {
    pub input_norm: Vec<f32>,
    pub query: Qwen3VlMatrix,
    pub query_norm: Vec<f32>,
    pub key: Qwen3VlMatrix,
    pub key_norm: Vec<f32>,
    pub value: Qwen3VlMatrix,
    pub output: Qwen3VlMatrix,
    pub post_attention_norm: Vec<f32>,
    pub gate: Qwen3VlMatrix,
    pub up: Qwen3VlMatrix,
    pub down: Qwen3VlMatrix,
}

/// 经典 dense Qwen3 与 Qwen3-VL checkpoint 的统一文本权重源。
pub trait Qwen3TextSource {
    fn text_layer(&self, layer: usize) -> Result<Qwen3VlLayerWeights, String>;
    fn final_norm(&self) -> Result<Vec<f32>, String>;
    fn lm_head(&self) -> Result<TensorData, String>;
}

pub struct Qwen3VlVisionLayerWeights {
    pub input_norm: LayerNormWeights,
    pub qkv: LinearWeights,
    pub output: LinearWeights,
    pub post_attention_norm: LayerNormWeights,
    pub mlp_input: LinearWeights,
    pub mlp_output: LinearWeights,
}

pub struct Qwen3VlVisionMergerWeights {
    pub norm: LayerNormWeights,
    pub input: LinearWeights,
    pub output: LinearWeights,
    pub norm_after_merge: bool,
}

pub struct Qwen3VlWeights {
    source: W4A16CtSource,
    config: Qwen3VlConfig,
}

impl Qwen3VlWeights {
    pub fn open(root: impl AsRef<Path>, config: Qwen3VlConfig) -> Result<Self, String> {
        config.validate()?;
        let source = W4A16CtSource::open(root)?;
        if source.group_size() != 32 {
            return Err(format!("Qwen3-VL-32B AWQ 需要 compressed-tensors group_size=32，实际 {}", source.group_size(),));
        }
        if !source.has("model.language_model.layers.0.self_attn.q_proj") {
            return Err("Qwen3-VL 权重缺少首层量化 query projection".to_owned());
        }
        Ok(Self { source, config })
    }

    pub fn embedding_rows_f32(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let rows = tokens
            .iter()
            .map(|&token| {
                let row = token as usize;
                (row < self.config.vocab_size).then_some(row).ok_or_else(|| format!("Qwen3-VL token {row} 超出 vocab {}", self.config.vocab_size))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tensor = self.source.load_bf16_rows("model.language_model.embed_tokens.weight", &rows)?;
        tensor.expect_shape(&[rows.len(), self.config.hidden_size])?;
        tensor.to_f32()
    }

    pub fn final_norm(&self) -> Result<Vec<f32>, String> {
        self.load_vector("model.language_model.norm.weight", self.config.hidden_size)
    }

    pub fn lm_head(&self) -> Result<TensorData, String> {
        let tensor = self.source.load_tensor("lm_head.weight")?;
        tensor.expect_shape(&[self.config.vocab_size, self.config.hidden_size])?;
        Ok(tensor)
    }

    pub fn load_layer(&self, layer: usize) -> Result<Qwen3VlLayerWeights, String> {
        if layer >= self.config.layer_count {
            return Err(format!("Qwen3-VL layer {layer} 越界，共 {} 层", self.config.layer_count));
        }
        let prefix = format!("model.language_model.layers.{layer}");
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

    pub fn load_vision_patch_embedding(&self) -> Result<LinearWeights, String> {
        let vision = self.vision()?;
        let columns = 3usize.checked_mul(vision.temporal_patch_size).and_then(|value| value.checked_mul(vision.patch_size)).and_then(|value| value.checked_mul(vision.patch_size)).ok_or("Qwen3-VL patch embedding 维度溢出")?;
        let mut weight = self.source.load_tensor("model.visual.patch_embed.proj.weight")?;
        if weight.shape.first() != Some(&vision.hidden_size) || weight.shape.iter().product::<usize>() != vision.hidden_size * columns {
            return Err(format!("Qwen3-VL patch embedding shape {:?}，期望 [{},{}]", weight.shape, vision.hidden_size, columns));
        }
        weight.shape = vec![vision.hidden_size, columns];
        let bias = self.source.load_tensor("model.visual.patch_embed.proj.bias")?;
        bias.expect_shape(&[vision.hidden_size])?;
        Ok(LinearWeights { weight, bias: Some(bias) })
    }

    pub fn load_vision_position_embedding(&self) -> Result<TensorData, String> {
        let vision = self.vision()?;
        let tensor = self.source.load_tensor("model.visual.pos_embed.weight")?;
        tensor.expect_shape(&[vision.position_embeddings, vision.hidden_size])?;
        Ok(tensor)
    }

    pub fn load_vision_layer(&self, layer: usize) -> Result<Qwen3VlVisionLayerWeights, String> {
        let vision = self.vision()?;
        if layer >= vision.depth {
            return Err(format!("Qwen3-VL vision layer {layer} 越界，共 {} 层", vision.depth));
        }
        let hidden = vision.hidden_size;
        let prefix = format!("model.visual.blocks.{layer}");
        Ok(Qwen3VlVisionLayerWeights {
            input_norm: self.load_layer_norm(&format!("{prefix}.norm1"), hidden)?,
            qkv: self.load_linear(&format!("{prefix}.attn.qkv"), hidden * 3, hidden, true)?,
            output: self.load_linear(&format!("{prefix}.attn.proj"), hidden, hidden, true)?,
            post_attention_norm: self.load_layer_norm(&format!("{prefix}.norm2"), hidden)?,
            mlp_input: self.load_linear(&format!("{prefix}.mlp.linear_fc1"), vision.intermediate_size, hidden, true)?,
            mlp_output: self.load_linear(&format!("{prefix}.mlp.linear_fc2"), hidden, vision.intermediate_size, true)?,
        })
    }

    pub fn load_vision_merger(&self) -> Result<Qwen3VlVisionMergerWeights, String> {
        self.load_merger("model.visual.merger", false)
    }

    pub fn load_vision_deepstack_merger(&self, index: usize) -> Result<Qwen3VlVisionMergerWeights, String> {
        let vision = self.vision()?;
        if index >= vision.deepstack_visual_indexes.len() {
            return Err(format!("Qwen3-VL deepstack merger {index} 越界"));
        }
        self.load_merger(&format!("model.visual.deepstack_merger_list.{index}"), true)
    }

    fn vision(&self) -> Result<&Qwen3VlVisionConfig, String> {
        self.config.vision.as_ref().ok_or_else(|| "dense Qwen3 文本模型无视觉权重".to_owned())
    }

    fn load_merger(&self, prefix: &str, norm_after_merge: bool) -> Result<Qwen3VlVisionMergerWeights, String> {
        let vision = self.vision()?;
        let merged = vision.hidden_size * vision.spatial_merge_size.pow(2);
        let norm_size = if norm_after_merge { merged } else { vision.hidden_size };
        Ok(Qwen3VlVisionMergerWeights {
            norm: self.load_layer_norm(&format!("{prefix}.norm"), norm_size)?,
            input: self.load_linear(&format!("{prefix}.linear_fc1"), merged, merged, true)?,
            output: self.load_linear(&format!("{prefix}.linear_fc2"), self.config.hidden_size, merged, true)?,
            norm_after_merge,
        })
    }

    fn load_matrix(&self, base: &str, rows: usize, cols: usize) -> Result<Qwen3VlMatrix, String> {
        let matrix = if self.source.has(base) {
            Qwen3VlMatrix::Quantized(self.source.load_matrix(base)?)
        } else {
            let tensor = self.source.load_tensor(&format!("{base}.weight"))?;
            // Dense 分支下游按二维矩阵索引 shape[0]/shape[1],低秩 tensor 必须在加载边界报 Err 而非 panic。
            tensor.expect_shape(&[rows, cols])?;
            Qwen3VlMatrix::Dense(tensor)
        };
        if matrix.rows() != rows || matrix.cols() != cols {
            return Err(format!("Qwen3-VL matrix {base} shape=[{},{}]，期望 [{rows},{cols}]", matrix.rows(), matrix.cols(),));
        }
        Ok(matrix)
    }

    fn load_linear(&self, prefix: &str, rows: usize, cols: usize, bias: bool) -> Result<LinearWeights, String> {
        let weight = self.source.load_tensor(&format!("{prefix}.weight"))?;
        weight.expect_shape(&[rows, cols])?;
        let bias = bias.then(|| self.source.load_tensor(&format!("{prefix}.bias"))).transpose()?;
        if let Some(bias) = &bias {
            bias.expect_shape(&[rows])?;
        }
        Ok(LinearWeights { weight, bias })
    }

    fn load_layer_norm(&self, prefix: &str, size: usize) -> Result<LayerNormWeights, String> {
        let weight = self.source.load_tensor(&format!("{prefix}.weight"))?;
        let bias = self.source.load_tensor(&format!("{prefix}.bias"))?;
        weight.expect_shape(&[size])?;
        bias.expect_shape(&[size])?;
        Ok(LayerNormWeights { weight, bias })
    }

    fn load_vector(&self, name: &str, size: usize) -> Result<Vec<f32>, String> {
        let tensor = self.source.load_tensor(name)?;
        tensor.expect_shape(&[size])?;
        tensor.to_f32()
    }
}

impl Qwen3TextSource for Qwen3VlWeights {
    fn text_layer(&self, layer: usize) -> Result<Qwen3VlLayerWeights, String> {
        self.load_layer(layer)
    }

    fn final_norm(&self) -> Result<Vec<f32>, String> {
        Qwen3VlWeights::final_norm(self)
    }

    fn lm_head(&self) -> Result<TensorData, String> {
        Qwen3VlWeights::lm_head(self)
    }
}
