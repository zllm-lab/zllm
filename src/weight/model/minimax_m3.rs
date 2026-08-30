//! MiniMax-M3 官方 safetensors 权重读取。

use crate::model_spec::minimax_m3::MiniMaxM3Config;
use crate::weight::container::safetensor::{SafetensorStore, TensorData};
use crate::weight::expert_source::{Mxfp4ExpertSource, Mxfp4ExpertWeights};
use crate::weight::format::mxfp4::load_mxfp4_matrix;
use crate::weight::format::mxfp8::{Mxfp8Matrix, Mxfp8MatrixBufferMut};
use crate::weight::format::nvfp4::NvidiaNvfp4Experts;
use crate::weight::model::vision::{LayerNormWeights, LinearWeights, VisionAttentionWeights, VisionEncoderLayerWeights, VisionMlpWeights};
use half::f16;
use std::path::Path;

const EMBEDDING_NAME: &str = "language_model.model.embed_tokens.weight";

pub enum MiniMaxM3CoreMatrix {
    Mxfp8(Mxfp8Matrix),
}

pub struct MiniMaxM3DenseLayerWeights {
    pub input_norm: Vec<f32>,
    pub query: MiniMaxM3CoreMatrix,
    pub query_norm: Vec<f32>,
    pub key: MiniMaxM3CoreMatrix,
    pub key_norm: Vec<f32>,
    pub value: MiniMaxM3CoreMatrix,
    pub output: MiniMaxM3CoreMatrix,
    pub post_attention_norm: Vec<f32>,
    pub gate: MiniMaxM3CoreMatrix,
    pub up: MiniMaxM3CoreMatrix,
    pub down: MiniMaxM3CoreMatrix,
}

pub struct MiniMaxM3MoeLayerWeights {
    pub input_norm: Vec<f32>,
    pub query: MiniMaxM3CoreMatrix,
    pub query_norm: Vec<f32>,
    pub key: MiniMaxM3CoreMatrix,
    pub key_norm: Vec<f32>,
    pub value: MiniMaxM3CoreMatrix,
    pub output: MiniMaxM3CoreMatrix,
    pub post_attention_norm: Vec<f32>,
    pub router_weight: Vec<f32>,
    pub router_bias: Vec<f32>,
    pub shared_gate: MiniMaxM3CoreMatrix,
    pub shared_up: MiniMaxM3CoreMatrix,
    pub shared_down: MiniMaxM3CoreMatrix,
}

pub struct MiniMaxM3MoeExpertWeights {
    pub gate: Mxfp8Matrix,
    pub up: Mxfp8Matrix,
    pub down: Mxfp8Matrix,
}

/// AMD Quark OCP MXFP4 routed experts；core 仍由 `MiniMaxM3Weights` 独立提供。
#[derive(Clone)]
pub struct MiniMaxM3Mxfp4Experts {
    store: SafetensorStore,
    dense_layer_count: usize,
    layer_count: usize,
    expert_intermediate_size: usize,
    hidden_size: usize,
}

impl MiniMaxM3Mxfp4Experts {
    pub fn open(root: impl AsRef<Path>, config: &MiniMaxM3Config) -> Result<Self, String> {
        Self::open_with_shape(root.as_ref(), config.dense_layer_count, config.layer_count, config.expert_intermediate_size, config.hidden_size)
    }

    fn open_with_shape(root: &Path, dense_layer_count: usize, layer_count: usize, expert_intermediate_size: usize, hidden_size: usize) -> Result<Self, String> {
        let store = SafetensorStore::open(root).map_err(|error| format!("打开 MiniMax-M3 MXFP4 专家目录 {} 失败: {error}", root.display()))?;
        let marker = format!("language_model.model.layers.{dense_layer_count}.block_sparse_moe.experts.0.w1.weight_scale");
        if !store.has(&marker) {
            return Err(format!("MiniMax-M3 MXFP4 权重索引缺少 {marker}"));
        }
        Ok(Self { store, dense_layer_count, layer_count, expert_intermediate_size, hidden_size })
    }

    fn load_matrix(&self, base: &str, rows: usize, cols: usize) -> Result<crate::weight::format::mxfp4::Mxfp4Matrix, String> {
        load_mxfp4_matrix(&self.store, &format!("{base}.weight"), &format!("{base}.weight_scale"), rows, cols)
    }
}

pub struct MiniMaxM3Weights {
    store: SafetensorStore,
    mxfp4_experts: Option<MiniMaxM3Mxfp4Experts>,
    nvfp4_experts: Option<NvidiaNvfp4Experts>,
    vocab_size: usize,
    hidden_size: usize,
    dense_layer_count: usize,
    layer_count: usize,
    dense_intermediate_size: usize,
    num_experts: usize,
    expert_intermediate_size: usize,
    shared_intermediate_size: usize,
    head_dim: usize,
    query_size: usize,
    kv_size: usize,
    vision_hidden_size: usize,
    vision_intermediate_size: usize,
    vision_layer_count: usize,
    vision_patch_cols: usize,
    vision_projector_hidden_size: usize,
    vision_merge_size: usize,
}

impl MiniMaxM3Weights {
    pub fn open(root: impl AsRef<Path>, config: &MiniMaxM3Config) -> Result<Self, String> {
        let root = root.as_ref();
        let store = SafetensorStore::open(root).map_err(|error| format!("打开 MiniMax-M3 权重目录 {} 失败: {error}", root.display()))?;
        if !store.has(EMBEDDING_NAME) {
            return Err(format!("MiniMax-M3 权重索引缺少 embedding {EMBEDDING_NAME:?}"));
        }
        let nvfp4_marker = format!("language_model.model.layers.{}.block_sparse_moe.experts.0.w1.weight_scale_2", config.dense_layer_count);
        let nvfp4_experts = store.has(&nvfp4_marker).then(|| NvidiaNvfp4Experts::open_minimax_m3(root, config.expert_intermediate_size, config.hidden_size, config.num_experts)).transpose()?;
        Self::from_store(store, nvfp4_experts, config)
    }

    fn from_store(store: SafetensorStore, nvfp4_experts: Option<NvidiaNvfp4Experts>, config: &MiniMaxM3Config) -> Result<Self, String> {
        let vision_patch_cols = 3usize
            .checked_mul(config.vision.temporal_patch_size)
            .and_then(|value| value.checked_mul(config.vision.patch_size))
            .and_then(|value| value.checked_mul(config.vision.patch_size))
            .ok_or_else(|| "MiniMax-M3 vision patch 维度溢出".to_owned())?;
        Ok(Self {
            store,
            mxfp4_experts: None,
            nvfp4_experts,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            dense_layer_count: config.dense_layer_count,
            layer_count: config.layer_count,
            dense_intermediate_size: config.dense_intermediate_size,
            num_experts: config.num_experts,
            expert_intermediate_size: config.expert_intermediate_size,
            shared_intermediate_size: config.shared_intermediate_size,
            head_dim: config.head_dim,
            query_size: config.num_heads * config.head_dim,
            kv_size: config.num_kv_heads * config.head_dim,
            vision_hidden_size: config.vision.hidden_size,
            vision_intermediate_size: config.vision.intermediate_size,
            vision_layer_count: config.vision.layer_count,
            vision_patch_cols,
            vision_projector_hidden_size: config.vision.projector_hidden_size,
            vision_merge_size: config.vision.spatial_merge_size,
        })
    }

    pub fn embedding_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let data = self.embedding_rows_bf16(token_ids)?;
        crate::weight::container::safetensor::decode_to_f32("token_embd", "BF16", &data)
    }

    pub fn embedding_rows_bf16(&self, token_ids: &[u32]) -> Result<Vec<u8>, String> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows: Vec<usize> = token_ids
            .iter()
            .map(|&token| {
                let row = token as usize;
                (row < self.vocab_size).then_some(row).ok_or_else(|| format!("MiniMax-M3 token {row} 超出 vocab {}", self.vocab_size))
            })
            .collect::<Result<_, _>>()?;
        let tensor = self.store.load_bf16_rows(EMBEDDING_NAME, &rows)?;
        if tensor.shape.as_slice() != [rows.len(), self.hidden_size] {
            return Err(format!("MiniMax-M3 embedding shape {:?}，期望 [{},{}]", tensor.shape, rows.len(), self.hidden_size));
        }
        Ok(tensor.data)
    }

    pub fn final_norm(&self) -> Result<Vec<f32>, String> {
        self.load_bf16_vec("language_model.model.norm.weight", self.hidden_size)
    }

    pub fn lm_head(&self) -> Result<Vec<f16>, String> {
        self.load_f16_matrix("language_model.lm_head.weight", self.vocab_size, self.hidden_size)
    }

    pub fn nvfp4_experts(&self) -> Option<NvidiaNvfp4Experts> {
        self.nvfp4_experts.clone()
    }

    pub fn mxfp4_experts(&self) -> Option<MiniMaxM3Mxfp4Experts> {
        self.mxfp4_experts.clone()
    }

    /// 用独立 MXFP4 routed-expert 根目录覆盖 core 根目录中的专家格式。
    pub fn with_mxfp4_experts(mut self, root: impl AsRef<Path>) -> Result<Self, String> {
        self.mxfp4_experts = Some(MiniMaxM3Mxfp4Experts::open_with_shape(root.as_ref(), self.dense_layer_count, self.layer_count, self.expert_intermediate_size, self.hidden_size)?);
        Ok(self)
    }

    pub fn load_dense_layer(&self, layer: usize) -> Result<MiniMaxM3DenseLayerWeights, String> {
        if layer >= self.dense_layer_count {
            return Err(format!("MiniMax-M3 dense layer {layer} 越界，dense_layer_count={}", self.dense_layer_count));
        }
        let prefix = format!("language_model.model.layers.{layer}");
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        Ok(MiniMaxM3DenseLayerWeights {
            input_norm: self.load_bf16_vec(&format!("{prefix}.input_layernorm.weight"), self.hidden_size)?,
            query: self.load_core_matrix(&format!("{attention}.q_proj.weight"), self.query_size, self.hidden_size)?,
            query_norm: self.load_bf16_vec(&format!("{attention}.q_norm.weight"), self.head_dim)?,
            key: self.load_core_matrix(&format!("{attention}.k_proj.weight"), self.kv_size, self.hidden_size)?,
            key_norm: self.load_bf16_vec(&format!("{attention}.k_norm.weight"), self.head_dim)?,
            value: self.load_core_matrix(&format!("{attention}.v_proj.weight"), self.kv_size, self.hidden_size)?,
            output: self.load_core_matrix(&format!("{attention}.o_proj.weight"), self.hidden_size, self.query_size)?,
            post_attention_norm: self.load_bf16_vec(&format!("{prefix}.post_attention_layernorm.weight"), self.hidden_size)?,
            gate: self.load_core_matrix(&format!("{mlp}.gate_proj.weight"), self.dense_intermediate_size, self.hidden_size)?,
            up: self.load_core_matrix(&format!("{mlp}.up_proj.weight"), self.dense_intermediate_size, self.hidden_size)?,
            down: self.load_core_matrix(&format!("{mlp}.down_proj.weight"), self.hidden_size, self.dense_intermediate_size)?,
        })
    }

    pub fn load_moe_layer(&self, layer: usize) -> Result<MiniMaxM3MoeLayerWeights, String> {
        if layer < self.dense_layer_count || layer >= self.layer_count {
            return Err(format!("MiniMax-M3 MoE layer {layer} 越界，有效范围 {}..{}", self.dense_layer_count, self.layer_count));
        }
        let prefix = format!("language_model.model.layers.{layer}");
        let attention = format!("{prefix}.self_attn");
        let moe = format!("{prefix}.block_sparse_moe");
        Ok(MiniMaxM3MoeLayerWeights {
            input_norm: self.load_bf16_vec(&format!("{prefix}.input_layernorm.weight"), self.hidden_size)?,
            query: self.load_core_matrix(&format!("{attention}.q_proj.weight"), self.query_size, self.hidden_size)?,
            query_norm: self.load_bf16_vec(&format!("{attention}.q_norm.weight"), self.head_dim)?,
            key: self.load_core_matrix(&format!("{attention}.k_proj.weight"), self.kv_size, self.hidden_size)?,
            key_norm: self.load_bf16_vec(&format!("{attention}.k_norm.weight"), self.head_dim)?,
            value: self.load_core_matrix(&format!("{attention}.v_proj.weight"), self.kv_size, self.hidden_size)?,
            output: self.load_core_matrix(&format!("{attention}.o_proj.weight"), self.hidden_size, self.query_size)?,
            post_attention_norm: self.load_bf16_vec(&format!("{prefix}.post_attention_layernorm.weight"), self.hidden_size)?,
            router_weight: self.load_f32_matrix(&format!("{moe}.gate.weight"), self.num_experts, self.hidden_size)?,
            router_bias: self.load_f32_vec(&format!("{moe}.e_score_correction_bias"), self.num_experts)?,
            shared_gate: self.load_core_matrix(&format!("{moe}.shared_experts.gate_proj.weight"), self.shared_intermediate_size, self.hidden_size)?,
            shared_up: self.load_core_matrix(&format!("{moe}.shared_experts.up_proj.weight"), self.shared_intermediate_size, self.hidden_size)?,
            shared_down: self.load_core_matrix(&format!("{moe}.shared_experts.down_proj.weight"), self.hidden_size, self.shared_intermediate_size)?,
        })
    }

    /// 直接把命中的专家读入调用方提供的 MXFP8 存储。
    pub fn load_moe_expert_into(&self, layer: usize, expert: usize, gate: Mxfp8MatrixBufferMut<'_>, up: Mxfp8MatrixBufferMut<'_>, down: Mxfp8MatrixBufferMut<'_>) -> Result<(), String> {
        if layer < self.dense_layer_count || layer >= self.layer_count || expert >= self.num_experts {
            return Err(format!("MiniMax-M3 MoE expert 越界: layer={layer}, expert={expert}"));
        }
        let prefix = format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}");
        self.load_mxfp8_into(&format!("{prefix}.w1.weight"), self.expert_intermediate_size, self.hidden_size, gate)?;
        self.load_mxfp8_into(&format!("{prefix}.w3.weight"), self.expert_intermediate_size, self.hidden_size, up)?;
        self.load_mxfp8_into(&format!("{prefix}.w2.weight"), self.hidden_size, self.expert_intermediate_size, down)
    }

    pub fn load_vision_patch_embedding(&self) -> Result<LinearWeights, String> {
        let name = "vision_tower.vision_model.embeddings.patch_embedding.weight";
        let mut weight = self.official()?.load(name).map_err(|error| format!("读取 MiniMax-M3 权重 {name}: {error}"))?;
        let elements = weight.shape.iter().try_fold(1usize, |count, value| count.checked_mul(*value)).ok_or_else(|| format!("MiniMax-M3 权重 {name} shape 大小溢出"))?;
        let expected = self.vision_hidden_size * self.vision_patch_cols;
        if weight.shape.first() != Some(&self.vision_hidden_size) || elements != expected {
            return Err(format!("MiniMax-M3 权重 {name} shape {:?}，期望首维 {}、共 {} 个元素", weight.shape, self.vision_hidden_size, expected));
        }
        // Conv3D kernel 的后四维已经按输入 patch 的展平顺序连续存储。
        weight.shape = vec![self.vision_hidden_size, self.vision_patch_cols];
        Ok(LinearWeights { weight, bias: None })
    }

    pub fn load_vision_pre_norm(&self) -> Result<LayerNormWeights, String> {
        self.load_layer_norm("vision_tower.vision_model.pre_layrnorm", self.vision_hidden_size)
    }

    /// 每次只加载一层，避免把整个 Vision Tower 展开到内存。
    pub fn load_vision_layer(&self, layer: usize) -> Result<VisionEncoderLayerWeights, String> {
        if layer >= self.vision_layer_count {
            return Err(format!("MiniMax-M3 vision layer {layer} 越界，layer_count={}", self.vision_layer_count));
        }
        let prefix = format!("vision_tower.vision_model.encoder.layers.{layer}");
        let attention = format!("{prefix}.self_attn");
        Ok(VisionEncoderLayerWeights {
            input_norm: self.load_layer_norm(&format!("{prefix}.layer_norm1"), self.vision_hidden_size)?,
            attention: VisionAttentionWeights {
                query: self.load_linear(&format!("{attention}.q_proj"), self.vision_hidden_size, self.vision_hidden_size, true)?,
                key: self.load_linear(&format!("{attention}.k_proj"), self.vision_hidden_size, self.vision_hidden_size, true)?,
                value: self.load_linear(&format!("{attention}.v_proj"), self.vision_hidden_size, self.vision_hidden_size, true)?,
                output: self.load_linear(&format!("{attention}.out_proj"), self.vision_hidden_size, self.vision_hidden_size, true)?,
            },
            post_attention_norm: self.load_layer_norm(&format!("{prefix}.layer_norm2"), self.vision_hidden_size)?,
            mlp: VisionMlpWeights {
                input: self.load_linear(&format!("{prefix}.mlp.fc1"), self.vision_intermediate_size, self.vision_hidden_size, true)?,
                output: self.load_linear(&format!("{prefix}.mlp.fc2"), self.vision_hidden_size, self.vision_intermediate_size, true)?,
            },
        })
    }

    /// patch projector 与 merge projector 分开加载，缩短大矩阵生命周期。
    pub fn load_vision_patch_projector(&self) -> Result<VisionMlpWeights, String> {
        Ok(VisionMlpWeights {
            input: self.load_linear("multi_modal_projector.linear_1", self.vision_projector_hidden_size, self.vision_hidden_size, true)?,
            output: self.load_linear("multi_modal_projector.linear_2", self.hidden_size, self.vision_projector_hidden_size, true)?,
        })
    }

    pub fn load_vision_merge_projector(&self) -> Result<VisionMlpWeights, String> {
        let merged_hidden = self.hidden_size.checked_mul(self.vision_merge_size).and_then(|value| value.checked_mul(self.vision_merge_size)).ok_or_else(|| "MiniMax-M3 vision merged hidden 维度溢出".to_owned())?;
        Ok(VisionMlpWeights {
            input: self.load_linear("patch_merge_mlp.linear_1", self.vision_projector_hidden_size, merged_hidden, true)?,
            output: self.load_linear("patch_merge_mlp.linear_2", self.hidden_size, self.vision_projector_hidden_size, true)?,
        })
    }

    fn load_linear(&self, prefix: &str, rows: usize, cols: usize, bias: bool) -> Result<LinearWeights, String> {
        let weight = self.load_checked(&format!("{prefix}.weight"), &[rows, cols])?;
        let bias = bias.then(|| self.load_checked(&format!("{prefix}.bias"), &[rows])).transpose()?;
        Ok(LinearWeights { weight, bias })
    }

    fn load_layer_norm(&self, prefix: &str, size: usize) -> Result<LayerNormWeights, String> {
        Ok(LayerNormWeights { weight: self.load_checked(&format!("{prefix}.weight"), &[size])?, bias: self.load_checked(&format!("{prefix}.bias"), &[size])? })
    }

    fn load_checked(&self, name: &str, expected_shape: &[usize]) -> Result<TensorData, String> {
        let tensor = self.official()?.load(name).map_err(|error| format!("读取 MiniMax-M3 权重 {name}: {error}"))?;
        if tensor.shape != expected_shape {
            return Err(format!("MiniMax-M3 权重 {name} shape {:?}，期望 {expected_shape:?}", tensor.shape));
        }
        Ok(tensor)
    }

    fn load_bf16_vec(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let tensor = self.load_checked(name, &[len])?;
        if tensor.dtype != "BF16" || tensor.data.len() != len * 2 {
            return Err(format!("MiniMax-M3 权重 {name} dtype={} bytes={}，期望 BF16/{}", tensor.dtype, tensor.data.len(), len * 2));
        }
        crate::weight::container::safetensor::decode_to_f32(name, &tensor.dtype, &tensor.data)
    }

    fn load_f32_vec(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let tensor = self.load_checked(name, &[len])?;
        if tensor.dtype != "F32" || tensor.data.len() != len * 4 {
            return Err(format!("MiniMax-M3 权重 {name} dtype={} bytes={}，期望 F32/{}", tensor.dtype, tensor.data.len(), len * 4));
        }
        crate::weight::container::safetensor::decode_to_f32(name, &tensor.dtype, &tensor.data)
    }

    fn load_f32_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>, String> {
        let len = rows.checked_mul(cols).ok_or_else(|| format!("MiniMax-M3 权重 {name} shape 大小溢出"))?;
        let tensor = self.load_checked(name, &[rows, cols])?;
        if tensor.dtype != "F32" || tensor.data.len() != len * 4 {
            return Err(format!("MiniMax-M3 权重 {name} dtype={} bytes={}，期望 F32/{}", tensor.dtype, tensor.data.len(), len * 4));
        }
        crate::weight::container::safetensor::decode_to_f32(name, &tensor.dtype, &tensor.data)
    }

    fn load_f16_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<f16>, String> {
        let tensor = self.load_checked(name, &[rows, cols])?;
        let len = rows.checked_mul(cols).ok_or_else(|| format!("MiniMax-M3 权重 {name} shape 大小溢出"))?;
        if !matches!(tensor.dtype.as_str(), "BF16" | "F16") || tensor.data.len() != len * 2 {
            return Err(format!("MiniMax-M3 权重 {name} dtype={} bytes={}，期望 BF16/F16/{}", tensor.dtype, tensor.data.len(), len * 2));
        }
        tensor.to_f16()
    }

    fn load_mxfp8(&self, name: &str, rows: usize, cols: usize) -> Result<Mxfp8Matrix, String> {
        let weight = self.load_checked(name, &[rows, cols])?;
        if weight.dtype != "F8_E4M3" {
            return Err(format!("MiniMax-M3 权重 {name} dtype={}，期望 F8_E4M3", weight.dtype));
        }
        let scale_name = format!("{name}_scale_inv");
        let scale = self.load_checked(&scale_name, &[rows, cols / 32])?;
        if scale.dtype != "U8" {
            return Err(format!("MiniMax-M3 权重 {scale_name} dtype={}，期望 U8 E8M0", scale.dtype));
        }
        Mxfp8Matrix::new(weight.data, scale.data, rows, cols).map_err(|error| format!("{name}: {error}"))
    }

    fn load_mxfp8_into(&self, name: &str, rows: usize, cols: usize, destination: Mxfp8MatrixBufferMut<'_>) -> Result<(), String> {
        if destination.rows != rows || destination.cols != cols {
            return Err(format!("MiniMax-M3 权重 {name} destination shape [{},{}]，期望 [{rows},{cols}]", destination.rows, destination.cols));
        }
        let weight = self.official()?.load_into(name, destination.codes)?;
        if weight.dtype != "F8_E4M3" || weight.shape.as_slice() != [rows, cols] {
            return Err(format!("MiniMax-M3 权重 {name} dtype={} shape={:?}，期望 F8_E4M3/[{rows},{cols}]", weight.dtype, weight.shape));
        }
        let scale_name = format!("{name}_scale_inv");
        let scale = self.official()?.load_into(&scale_name, destination.scale_inv)?;
        if scale.dtype != "U8" || scale.shape.as_slice() != [rows, cols / 32] {
            return Err(format!("MiniMax-M3 权重 {scale_name} dtype={} shape={:?}，期望 U8/[{},{}]", scale.dtype, scale.shape, rows, cols / 32));
        }
        Ok(())
    }

    fn official(&self) -> Result<&SafetensorStore, String> {
        Ok(&self.store)
    }

    fn load_core_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<MiniMaxM3CoreMatrix, String> {
        self.load_mxfp8(name, rows, cols).map(MiniMaxM3CoreMatrix::Mxfp8)
    }
}

impl Mxfp4ExpertSource for MiniMaxM3Mxfp4Experts {
    fn intermediate(&self) -> usize {
        self.expert_intermediate_size
    }

    fn hidden(&self) -> usize {
        self.hidden_size
    }

    fn load_expert_mxfp4(&self, layer: usize, expert: usize) -> Result<Mxfp4ExpertWeights, String> {
        if layer < self.dense_layer_count || layer >= self.layer_count {
            return Err(format!("MiniMax-M3 MXFP4 expert layer {layer} 越界，有效范围 {}..{}", self.dense_layer_count, self.layer_count));
        }
        let prefix = format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}");
        Ok(Mxfp4ExpertWeights {
            gate: self.load_matrix(&format!("{prefix}.w1"), self.expert_intermediate_size, self.hidden_size)?,
            up: self.load_matrix(&format!("{prefix}.w3"), self.expert_intermediate_size, self.hidden_size)?,
            down: self.load_matrix(&format!("{prefix}.w2"), self.hidden_size, self.expert_intermediate_size)?,
        })
    }
}

impl crate::weight::expert_source::Mxfp8ExpertSource for MiniMaxM3Weights {
    fn load_expert_into(
        &self,
        layer: usize,
        expert: usize,
        gate: crate::weight::format::mxfp8::Mxfp8MatrixBufferMut<'_>,
        up: crate::weight::format::mxfp8::Mxfp8MatrixBufferMut<'_>,
        down: crate::weight::format::mxfp8::Mxfp8MatrixBufferMut<'_>,
    ) -> Result<(), String> {
        self.load_moe_expert_into(layer, expert, gate, up, down)
    }
}

impl crate::weight::expert_source::ExpertSourceProvider for MiniMaxM3Weights {
    fn source(&self, _layer: usize) -> Result<crate::weight::expert_source::ExpertSource<'_>, String> {
        if let Some(source) = &self.mxfp4_experts {
            Ok(crate::weight::expert_source::ExpertSource::Mxfp4(source))
        } else if let Some(source) = &self.nvfp4_experts {
            Ok(crate::weight::expert_source::ExpertSource::Nvfp4(source))
        } else {
            Ok(crate::weight::expert_source::ExpertSource::Mxfp8(self))
        }
    }
}
