//! GLM-5.2 权重加载与标准格式装配。

use crate::model_spec::glm52::{Glm52Config, is_indexer_layer};
use crate::tokenizer::{Detokenizer, Tokenizer};
use crate::weight::container::gguf::{GgmlType, GgufMatrix, GgufReader};
use crate::weight::container::safetensor::{SafetensorStore, TensorData};
use crate::weight::expert_source::{GgufExpertSource, GgufExpertWeights};
use crate::weight::format::compressed_tensors_hybrid::{CompressedTensorsSource, CtLinearWeight};
use crate::weight::format::nvfp4::NvidiaNvfp4Experts;
use crate::weight::format::quantization::Fp8Matrix;
use crate::weight::format::quantization::{ScaleDType, W8A16Matrix};
use half::{bf16, f16};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct Nvfp4CoreIndexer {
    pub wq_b: Vec<half::f16>,
    pub wk: Vec<half::f16>,
    pub weights_proj: Vec<half::f16>,
    pub k_norm_weight: Vec<half::f16>,
    pub k_norm_bias: Vec<half::f16>,
}

pub struct Nvfp4CoreDenseLayer {
    pub indexer: Option<Nvfp4CoreIndexer>,
    pub input_norm: Vec<half::f16>,
    pub q_a_proj: Vec<half::f16>,
    pub q_a_norm: Vec<half::f16>,
    pub q_b_proj: Vec<half::f16>,
    pub kv_a_proj: Vec<half::f16>,
    pub kv_a_norm: Vec<half::f16>,
    pub kv_b_proj: Vec<half::f16>,
    pub o_proj: Vec<half::f16>,
    pub post_attn_norm: Vec<half::f16>,
    pub gate_proj: Vec<half::f16>,
    pub up_proj: Vec<half::f16>,
    pub down_proj: Vec<half::f16>,
}

pub struct Nvfp4CoreMoeLayer {
    pub indexer: Option<Nvfp4CoreIndexer>,
    pub input_norm: Vec<half::f16>,
    pub q_a_proj: Vec<half::f16>,
    pub q_a_norm: Vec<half::f16>,
    pub q_b_proj: Vec<half::f16>,
    pub kv_a_proj: Vec<half::f16>,
    pub kv_a_norm: Vec<half::f16>,
    pub kv_b_proj: Vec<half::f16>,
    pub o_proj: Vec<half::f16>,
    pub post_attn_norm: Vec<half::f16>,
    pub router_weight: Vec<half::f16>,
    pub router_bias: Vec<half::f16>,
    pub shared_gate: Vec<half::f16>,
    pub shared_up: Vec<half::f16>,
    pub shared_down: Vec<half::f16>,
}

/// NVIDIA GLM-5.2 NVFP4 checkpoint：BF16 core + NVFP4 routed experts。
#[derive(Clone)]
pub struct NvidiaNvfp4Model {
    store: crate::weight::container::safetensor::SafetensorStore,
    cfg: Glm52Config,
    experts: NvidiaNvfp4Experts,
}

impl NvidiaNvfp4Model {
    pub fn open(root: &std::path::Path, cfg: Glm52Config) -> Result<Self, String> {
        let store = crate::weight::container::safetensor::SafetensorStore::open(root)?;
        let experts = NvidiaNvfp4Experts::open(root, cfg.expert_intermediate_size, cfg.hidden_size, cfg.expert_count)?;
        Ok(Self { store, cfg, experts })
    }

    pub fn experts(&self) -> NvidiaNvfp4Experts {
        self.experts.clone()
    }

    pub fn embedding_rows_bf16(&self, token_ids: &[u32]) -> Result<Vec<u8>, String> {
        let rows: Vec<usize> = token_ids.iter().map(|&token| token as usize).collect();
        let tensor = self.store.load_bf16_rows("model.embed_tokens.weight", &rows)?;
        if tensor.shape != [rows.len(), self.cfg.hidden_size] {
            return Err(format!("model.embed_tokens.weight rows shape 异常: {:?}", tensor.shape));
        }
        Ok(tensor.data)
    }

    pub fn final_norm(&self) -> Result<Vec<f32>, String> {
        self.load_f32("model.norm.weight", &[self.cfg.hidden_size])
    }

    pub fn lm_head(&self) -> Result<Vec<half::f16>, String> {
        self.load_f16("lm_head.weight", &[self.cfg.vocab_size, self.cfg.hidden_size])
    }

    pub fn load_dense_layer(&self, layer: usize) -> Result<Nvfp4CoreDenseLayer, String> {
        let h = self.cfg.hidden_size;
        let attn = format!("model.layers.{layer}.self_attn");
        let mlp = format!("model.layers.{layer}.mlp");
        Ok(Nvfp4CoreDenseLayer {
            indexer: self.load_indexer(layer)?,
            input_norm: self.load_f16(&format!("model.layers.{layer}.input_layernorm.weight"), &[h])?,
            q_a_proj: self.load_f16(&format!("{attn}.q_a_proj.weight"), &[self.cfg.q_lora_rank, h])?,
            q_a_norm: self.load_f16(&format!("{attn}.q_a_layernorm.weight"), &[self.cfg.q_lora_rank])?,
            q_b_proj: self.load_f16(&format!("{attn}.q_b_proj.weight"), &[self.cfg.q_projection_size, self.cfg.q_lora_rank])?,
            kv_a_proj: self.load_f16(&format!("{attn}.kv_a_proj_with_mqa.weight"), &[self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim, h])?,
            kv_a_norm: self.load_f16(&format!("{attn}.kv_a_layernorm.weight"), &[self.cfg.kv_lora_rank])?,
            kv_b_proj: self.load_f16(&format!("{attn}.kv_b_proj.weight"), &[self.cfg.kv_projection_size, self.cfg.kv_lora_rank])?,
            o_proj: self.load_f16(&format!("{attn}.o_proj.weight"), &[h, self.cfg.q_projection_size])?,
            post_attn_norm: self.load_f16(&format!("model.layers.{layer}.post_attention_layernorm.weight"), &[h])?,
            gate_proj: self.load_f16(&format!("{mlp}.gate_proj.weight"), &[self.cfg.dense_intermediate_size, h])?,
            up_proj: self.load_f16(&format!("{mlp}.up_proj.weight"), &[self.cfg.dense_intermediate_size, h])?,
            down_proj: self.load_f16(&format!("{mlp}.down_proj.weight"), &[h, self.cfg.dense_intermediate_size])?,
        })
    }

    pub fn load_moe_layer(&self, layer: usize) -> Result<Nvfp4CoreMoeLayer, String> {
        let h = self.cfg.hidden_size;
        let attn = format!("model.layers.{layer}.self_attn");
        let mlp = format!("model.layers.{layer}.mlp");
        Ok(Nvfp4CoreMoeLayer {
            indexer: self.load_indexer(layer)?,
            input_norm: self.load_f16(&format!("model.layers.{layer}.input_layernorm.weight"), &[h])?,
            q_a_proj: self.load_f16(&format!("{attn}.q_a_proj.weight"), &[self.cfg.q_lora_rank, h])?,
            q_a_norm: self.load_f16(&format!("{attn}.q_a_layernorm.weight"), &[self.cfg.q_lora_rank])?,
            q_b_proj: self.load_f16(&format!("{attn}.q_b_proj.weight"), &[self.cfg.q_projection_size, self.cfg.q_lora_rank])?,
            kv_a_proj: self.load_f16(&format!("{attn}.kv_a_proj_with_mqa.weight"), &[self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim, h])?,
            kv_a_norm: self.load_f16(&format!("{attn}.kv_a_layernorm.weight"), &[self.cfg.kv_lora_rank])?,
            kv_b_proj: self.load_f16(&format!("{attn}.kv_b_proj.weight"), &[self.cfg.kv_projection_size, self.cfg.kv_lora_rank])?,
            o_proj: self.load_f16(&format!("{attn}.o_proj.weight"), &[h, self.cfg.q_projection_size])?,
            post_attn_norm: self.load_f16(&format!("model.layers.{layer}.post_attention_layernorm.weight"), &[h])?,
            router_weight: self.load_f16(&format!("{mlp}.gate.weight"), &[self.cfg.expert_count, h])?,
            router_bias: self.load_f16(&format!("{mlp}.gate.e_score_correction_bias"), &[self.cfg.expert_count])?,
            shared_gate: self.load_f16(&format!("{mlp}.shared_experts.gate_proj.weight"), &[self.cfg.expert_intermediate_size, h])?,
            shared_up: self.load_f16(&format!("{mlp}.shared_experts.up_proj.weight"), &[self.cfg.expert_intermediate_size, h])?,
            shared_down: self.load_f16(&format!("{mlp}.shared_experts.down_proj.weight"), &[h, self.cfg.expert_intermediate_size])?,
        })
    }

    fn load_indexer(&self, layer: usize) -> Result<Option<Nvfp4CoreIndexer>, String> {
        if !is_indexer_layer(layer) {
            return Ok(None);
        }
        let prefix = format!("model.layers.{layer}.self_attn.indexer");
        let query_size = self.cfg.index_heads * self.cfg.index_head_dim;
        Ok(Some(Nvfp4CoreIndexer {
            wq_b: self.load_f16(&format!("{prefix}.wq_b.weight"), &[query_size, self.cfg.q_lora_rank])?,
            wk: self.load_f16(&format!("{prefix}.wk.weight"), &[self.cfg.index_head_dim, self.cfg.hidden_size])?,
            weights_proj: self.load_f16(&format!("{prefix}.weights_proj.weight"), &[self.cfg.index_heads, self.cfg.hidden_size])?,
            k_norm_weight: self.load_f16(&format!("{prefix}.k_norm.weight"), &[self.cfg.index_head_dim])?,
            k_norm_bias: self.load_f16(&format!("{prefix}.k_norm.bias"), &[self.cfg.index_head_dim])?,
        }))
    }

    fn load_f16(&self, name: &str, shape: &[usize]) -> Result<Vec<half::f16>, String> {
        let tensor = self.store.load(name)?;
        if tensor.shape != shape {
            return Err(format!("{name} shape {:?}，期望 {shape:?}", tensor.shape));
        }
        let values = tensor.to_f16()?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{name} 转换为 F16 后包含非有限值"));
        }
        Ok(values)
    }

    fn load_f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let tensor = self.store.load(name)?;
        if tensor.shape != shape {
            return Err(format!("{name} shape {:?}，期望 {shape:?}", tensor.shape));
        }
        tensor.to_f32()
    }
}

/// GLM-5.2 compressed-tensors dense 层；矩阵保持 checkpoint 原始存储。
pub struct CtDenseLayer {
    pub indexer: Option<CtIndexerWeights>,
    pub input_norm: Vec<f32>,
    pub q_a_proj: CtLinearWeight,
    pub q_a_norm: Vec<f32>,
    pub q_b_proj: CtLinearWeight,
    pub kv_a_proj: CtLinearWeight,
    pub kv_a_norm: Vec<f32>,
    pub kv_b_proj: CtLinearWeight,
    pub o_proj: CtLinearWeight,
    pub post_attn_norm: Vec<f32>,
    pub gate_proj: CtLinearWeight,
    pub up_proj: CtLinearWeight,
    pub down_proj: CtLinearWeight,
}

/// GLM-5.2 compressed-tensors MoE core，不含 routed expert。
pub struct CtMoeLayer {
    pub indexer: Option<CtIndexerWeights>,
    pub input_norm: Vec<f32>,
    pub q_a_proj: CtLinearWeight,
    pub q_a_norm: Vec<f32>,
    pub q_b_proj: CtLinearWeight,
    pub kv_a_proj: CtLinearWeight,
    pub kv_a_norm: Vec<f32>,
    pub kv_b_proj: CtLinearWeight,
    pub o_proj: CtLinearWeight,
    pub post_attn_norm: Vec<f32>,
    pub router_weight: Vec<f32>,
    pub router_bias: Vec<f32>,
    pub shared_gate: CtLinearWeight,
    pub shared_up: CtLinearWeight,
    pub shared_down: CtLinearWeight,
}

/// GLM-5.2 next-token prediction 层。
pub struct CtMtpLayer {
    pub embedding_norm: Vec<f32>,
    pub hidden_norm: Vec<f32>,
    pub input_projection: CtLinearWeight,
    pub layer: CtMoeLayer,
    pub output_norm: Vec<f32>,
}

/// GLM-5.2 DSA indexer 权重。
pub struct CtIndexerWeights {
    pub wq_b: CtLinearWeight,
    pub wk: CtLinearWeight,
    pub weights_proj: CtLinearWeight,
    pub k_norm_weight: Vec<f32>,
    pub k_norm_bias: Vec<f32>,
}

impl CompressedTensorsSource {
    pub fn load_dense_layer(&self, layer: usize) -> Result<CtDenseLayer, String> {
        let p = format!("model.layers.{layer}");
        let attn = format!("{p}.self_attn");
        let mlp = format!("{p}.mlp");
        Ok(CtDenseLayer {
            indexer: self.load_glm52_indexer(layer, &attn)?,
            input_norm: self.load_norm(&format!("{p}.input_layernorm.weight"))?,
            q_a_proj: self.load_linear(&format!("{attn}.q_a_proj.weight"))?,
            q_a_norm: self.load_norm(&format!("{attn}.q_a_layernorm.weight"))?,
            q_b_proj: self.load_linear(&format!("{attn}.q_b_proj.weight"))?,
            kv_a_proj: self.load_linear(&format!("{attn}.kv_a_proj_with_mqa.weight"))?,
            kv_a_norm: self.load_norm(&format!("{attn}.kv_a_layernorm.weight"))?,
            kv_b_proj: self.load_linear(&format!("{attn}.kv_b_proj.weight"))?,
            o_proj: self.load_linear(&format!("{attn}.o_proj.weight"))?,
            post_attn_norm: self.load_norm(&format!("{p}.post_attention_layernorm.weight"))?,
            gate_proj: self.load_linear(&format!("{mlp}.gate_proj.weight"))?,
            up_proj: self.load_linear(&format!("{mlp}.up_proj.weight"))?,
            down_proj: self.load_linear(&format!("{mlp}.down_proj.weight"))?,
        })
    }

    pub fn load_moe_layer(&self, layer: usize) -> Result<CtMoeLayer, String> {
        let p = format!("model.layers.{layer}");
        let attn = format!("{p}.self_attn");
        let mlp = format!("{p}.mlp");
        let shared = format!("{mlp}.shared_experts");
        Ok(CtMoeLayer {
            indexer: self.load_glm52_indexer(layer, &attn)?,
            input_norm: self.load_norm(&format!("{p}.input_layernorm.weight"))?,
            q_a_proj: self.load_linear(&format!("{attn}.q_a_proj.weight"))?,
            q_a_norm: self.load_norm(&format!("{attn}.q_a_layernorm.weight"))?,
            q_b_proj: self.load_linear(&format!("{attn}.q_b_proj.weight"))?,
            kv_a_proj: self.load_linear(&format!("{attn}.kv_a_proj_with_mqa.weight"))?,
            kv_a_norm: self.load_norm(&format!("{attn}.kv_a_layernorm.weight"))?,
            kv_b_proj: self.load_linear(&format!("{attn}.kv_b_proj.weight"))?,
            o_proj: self.load_linear(&format!("{attn}.o_proj.weight"))?,
            post_attn_norm: self.load_norm(&format!("{p}.post_attention_layernorm.weight"))?,
            router_weight: self.load_f32_vec(&format!("{mlp}.gate.weight"))?,
            router_bias: self.load_f32_vec(&format!("{mlp}.gate.e_score_correction_bias"))?,
            shared_gate: self.load_linear(&format!("{shared}.gate_proj.weight"))?,
            shared_up: self.load_linear(&format!("{shared}.up_proj.weight"))?,
            shared_down: self.load_linear(&format!("{shared}.down_proj.weight"))?,
        })
    }

    pub fn load_mtp_layer(&self, layer: usize) -> Result<CtMtpLayer, String> {
        let p = format!("model.layers.{layer}");
        Ok(CtMtpLayer {
            embedding_norm: self.load_norm(&format!("{p}.enorm.weight"))?,
            hidden_norm: self.load_norm(&format!("{p}.hnorm.weight"))?,
            input_projection: self.load_linear(&format!("{p}.eh_proj.weight"))?,
            layer: self.load_moe_layer(layer)?,
            output_norm: self.load_norm(&format!("{p}.shared_head.norm.weight"))?,
        })
    }

    /// IndexShare 语义由层配置决定；CT checkpoint 可能保留共享层的冗余
    /// indexer 张量，不能据此把 shared 层误判为 full 层。
    fn load_glm52_indexer(&self, layer: usize, attention: &str) -> Result<Option<CtIndexerWeights>, String> {
        if !is_indexer_layer(layer) {
            return Ok(None);
        }
        let base = format!("{attention}.indexer");
        if !self.store().has(&format!("{base}.wk.weight")) {
            return Ok(None);
        }
        Ok(Some(CtIndexerWeights {
            wq_b: self.load_linear(&format!("{base}.wq_b.weight"))?,
            wk: self.load_linear(&format!("{base}.wk.weight"))?,
            weights_proj: self.load_linear(&format!("{base}.weights_proj.weight"))?,
            k_norm_weight: self.load_norm(&format!("{base}.k_norm.weight"))?,
            k_norm_bias: self.load_norm(&format!("{base}.k_norm.bias"))?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_indexshare_ignores_redundant_shared_layer_tensors() {
        let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("zllm-glm52-ct-indexshare-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), br#"{"quantization_config":{"quant_method":"compressed-tensors","config_groups":{"group":{"weights":{"group_size":64}}}}}"#).unwrap();

        let tensor = "model.layers.3.self_attn.indexer.wk.weight";
        let header = serde_json::to_vec(&serde_json::json!({
            tensor: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        }))
        .unwrap();
        let mut safetensor = Vec::with_capacity(8 + header.len() + 4);
        safetensor.extend_from_slice(&(header.len() as u64).to_le_bytes());
        safetensor.extend_from_slice(&header);
        safetensor.extend_from_slice(&0.0_f32.to_le_bytes());
        std::fs::write(root.join("model.safetensors"), safetensor).unwrap();

        let source = CompressedTensorsSource::open(&root).unwrap();
        let indexer = source.load_glm52_indexer(3, "model.layers.3.self_attn").unwrap();
        assert!(indexer.is_none(), "L3 是 IndexShare 层，即使 checkpoint 保留冗余张量也必须复用上一层 selection");

        std::fs::remove_dir_all(root).unwrap();
    }
}

/// GLM-5.2 GGUF dense 层；量化矩阵保持 GGUF 原始存储，norm/router 为 F32。
pub struct GgufDenseLayer {
    pub indexer: Option<GgufIndexerWeights>,
    pub input_norm: Vec<f32>,
    pub q_a_proj: GgufMatrix,
    pub q_a_norm: Vec<f32>,
    pub q_b_proj: GgufMatrix,
    pub kv_a_proj: GgufMatrix,
    pub kv_a_norm: Vec<f32>,
    /// k_b/v_b 按 MLA fused 行布局([k_nope|v] 每 head 一段)直接重排为 W8A16
    /// (group 32 + F16 scale)——Q8_0 与 W8A16 数学同构(d×int8)，无损转换；
    /// absorb/prefill MLA kernel 走 CT 原生 W8 packed 快路径。
    pub kv_b_w8: W8A16Matrix,
    pub o_proj: GgufMatrix,
    pub post_attn_norm: Vec<f32>,
    pub gate_proj: GgufMatrix,
    pub up_proj: GgufMatrix,
    pub down_proj: GgufMatrix,
}

/// GLM-5.2 GGUF MoE core，不含 routed expert。
pub struct GgufMoeLayer {
    pub indexer: Option<GgufIndexerWeights>,
    pub input_norm: Vec<f32>,
    pub q_a_proj: GgufMatrix,
    pub q_a_norm: Vec<f32>,
    pub q_b_proj: GgufMatrix,
    pub kv_a_proj: GgufMatrix,
    pub kv_a_norm: Vec<f32>,
    /// k_b/v_b fused 行布局直接重排为 W8A16(group 32 + F16 scale)，与 dense 层同构。
    pub kv_b_w8: W8A16Matrix,
    pub o_proj: GgufMatrix,
    pub post_attn_norm: Vec<f32>,
    pub router_weight: Vec<f32>,
    pub router_bias: Vec<f32>,
    pub shared_gate: GgufMatrix,
    pub shared_up: GgufMatrix,
    pub shared_down: GgufMatrix,
}

/// GLM-5.2 GGUF next-token prediction 层(blk.{layer_count}.nextn.*)。
pub struct GgufMtpLayer {
    pub embedding_norm: Vec<f32>,
    pub hidden_norm: Vec<f32>,
    pub input_projection: GgufMatrix,
    pub layer: GgufMoeLayer,
    pub output_norm: Vec<f32>,
}

/// GLM-5.2 GGUF DSA indexer 权重。
pub struct GgufIndexerWeights {
    pub wq_b: GgufMatrix,
    pub wk: GgufMatrix,
    pub weights_proj: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub k_norm_bias: Vec<f32>,
}

/// GLM-5.2 GGUF 权重视图(arch=glm-dsa)。
///
/// 只负责标准 tensor 命名映射、shape/元数据校验与逐层加载；注意力与 MoE
/// 执行顺序由 `runtime::glm52` 拥有，量化类型由 backend capability 分发。
#[derive(Clone)]
pub struct Glm52Gguf {
    reader: Arc<GgufReader>,
    cfg: Glm52Config,
}

impl Glm52Gguf {
    pub fn open(path: impl AsRef<Path>, cfg: Glm52Config) -> Result<Self, String> {
        let path = GgufReader::locate(path.as_ref())?;
        let model = Self { reader: Arc::new(GgufReader::open(&path)?), cfg };
        model.validate_metadata()?;
        model.validate_tensors()?;
        Ok(model)
    }

    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        self.reader.bpe_tokenizer().map_err(|error| format!("构造 GLM-5.2 GGUF tokenizer: {error}"))
    }

    pub fn detokenizer(&self) -> Result<Detokenizer, String> {
        self.reader.bpe_detokenizer().map_err(|error| format!("构造 GLM-5.2 GGUF detokenizer: {error}"))
    }

    fn validate_metadata(&self) -> Result<(), String> {
        let cfg = &self.cfg;
        let q_head_dim = cfg.q_projection_size / cfg.num_heads;
        let nope = q_head_dim - cfg.qk_rope_head_dim;
        let v_dim = cfg.kv_projection_size / cfg.num_heads - nope;
        self.reader.expect_metadata_str("general.architecture", "glm-dsa")?;
        for (key, expected) in [
            // GGUF block_count 含 MTP 层(blk.{layer_count})。
            ("glm-dsa.block_count", cfg.layer_count + cfg.mtp_layer_count),
            ("glm-dsa.embedding_length", cfg.hidden_size),
            ("glm-dsa.feed_forward_length", cfg.dense_intermediate_size),
            ("glm-dsa.vocab_size", cfg.vocab_size),
            ("glm-dsa.attention.head_count", cfg.num_heads),
            ("glm-dsa.attention.q_lora_rank", cfg.q_lora_rank),
            ("glm-dsa.attention.kv_lora_rank", cfg.kv_lora_rank),
            ("glm-dsa.attention.key_length", cfg.kv_lora_rank + cfg.qk_rope_head_dim),
            ("glm-dsa.attention.key_length_mla", q_head_dim),
            ("glm-dsa.attention.value_length_mla", v_dim),
            ("glm-dsa.expert_count", cfg.expert_count),
            ("glm-dsa.expert_used_count", cfg.expert_top_k),
            ("glm-dsa.expert_feed_forward_length", cfg.expert_intermediate_size),
            ("glm-dsa.expert_shared_count", 1),
            ("glm-dsa.leading_dense_block_count", cfg.dense_layer_count),
            ("glm-dsa.attention.indexer.head_count", cfg.index_heads),
            ("glm-dsa.attention.indexer.key_length", cfg.index_head_dim),
            ("glm-dsa.attention.indexer.top_k", cfg.index_top_k),
            ("glm-dsa.nextn_predict_layers", cfg.mtp_layer_count),
        ] {
            self.reader.expect_metadata_u64(key, expected as u64)?;
        }
        Ok(())
    }

    fn validate_tensors(&self) -> Result<(), String> {
        let cfg = &self.cfg;
        let q_head_dim = cfg.q_projection_size / cfg.num_heads;
        let nope = q_head_dim - cfg.qk_rope_head_dim;
        let v_dim = cfg.kv_projection_size / cfg.num_heads - nope;
        self.expect("token_embd.weight", &[cfg.hidden_size, cfg.vocab_size])?;
        self.expect("output.weight", &[cfg.hidden_size, cfg.vocab_size])?;
        self.expect_f32("output_norm.weight", &[cfg.hidden_size])?;
        for layer in 0..cfg.layer_count {
            let prefix = format!("blk.{layer}");
            self.validate_layer_core(layer, &prefix, nope, v_dim)?;
            if layer < cfg.dense_layer_count {
                self.expect(&format!("{prefix}.ffn_gate.weight"), &[cfg.hidden_size, cfg.dense_intermediate_size])?;
                self.expect(&format!("{prefix}.ffn_up.weight"), &[cfg.hidden_size, cfg.dense_intermediate_size])?;
                self.expect(&format!("{prefix}.ffn_down.weight"), &[cfg.dense_intermediate_size, cfg.hidden_size])?;
            } else {
                self.validate_layer_moe(&prefix)?;
            }
        }
        if cfg.mtp_layer_count > 0 {
            // MTP 层是 blk.{layer_count} 上的完整 MoE 层 + nextn 附加张量。
            let prefix = format!("blk.{}", cfg.layer_count);
            self.validate_layer_core(cfg.layer_count, &prefix, nope, v_dim)?;
            self.validate_layer_moe(&prefix)?;
            self.expect_f32(&format!("{prefix}.nextn.enorm.weight"), &[cfg.hidden_size])?;
            self.expect_f32(&format!("{prefix}.nextn.hnorm.weight"), &[cfg.hidden_size])?;
            self.expect(&format!("{prefix}.nextn.eh_proj.weight"), &[cfg.hidden_size * 2, cfg.hidden_size])?;
            self.expect_f32(&format!("{prefix}.nextn.shared_head_norm.weight"), &[cfg.hidden_size])?;
        }
        Ok(())
    }

    /// attention 共有部分;`attn_k_b` 是吸收式转置存储(nope 最快维)，`attn_v_b` 是常规存储(kv_lora 最快维)。
    fn validate_layer_core(&self, layer: usize, prefix: &str, nope: usize, v_dim: usize) -> Result<(), String> {
        let cfg = &self.cfg;
        self.expect_f32(&format!("{prefix}.attn_norm.weight"), &[cfg.hidden_size])?;
        self.expect(&format!("{prefix}.attn_q_a.weight"), &[cfg.hidden_size, cfg.q_lora_rank])?;
        self.expect_f32(&format!("{prefix}.attn_q_a_norm.weight"), &[cfg.q_lora_rank])?;
        self.expect(&format!("{prefix}.attn_q_b.weight"), &[cfg.q_lora_rank, cfg.q_projection_size])?;
        self.expect(&format!("{prefix}.attn_kv_a_mqa.weight"), &[cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim])?;
        self.expect_f32(&format!("{prefix}.attn_kv_a_norm.weight"), &[cfg.kv_lora_rank])?;
        self.expect(&format!("{prefix}.attn_k_b.weight"), &[nope, cfg.kv_lora_rank, cfg.num_heads])?;
        self.expect(&format!("{prefix}.attn_v_b.weight"), &[cfg.kv_lora_rank, v_dim, cfg.num_heads])?;
        self.expect(&format!("{prefix}.attn_output.weight"), &[cfg.q_projection_size, cfg.hidden_size])?;
        self.expect_f32(&format!("{prefix}.ffn_norm.weight"), &[cfg.hidden_size])?;
        if is_indexer_layer(layer) {
            self.expect(&format!("{prefix}.indexer.attn_q_b.weight"), &[cfg.q_lora_rank, cfg.index_heads * cfg.index_head_dim])?;
            self.expect(&format!("{prefix}.indexer.attn_k.weight"), &[cfg.hidden_size, cfg.index_head_dim])?;
            self.expect_f32(&format!("{prefix}.indexer.proj.weight"), &[cfg.hidden_size, cfg.index_heads])?;
            self.expect_f32(&format!("{prefix}.indexer.k_norm.weight"), &[cfg.index_head_dim])?;
            self.expect_f32(&format!("{prefix}.indexer.k_norm.bias"), &[cfg.index_head_dim])?;
        }
        Ok(())
    }

    fn validate_layer_moe(&self, prefix: &str) -> Result<(), String> {
        let cfg = &self.cfg;
        self.expect_f32(&format!("{prefix}.ffn_gate_inp.weight"), &[cfg.hidden_size, cfg.expert_count])?;
        self.expect_f32(&format!("{prefix}.exp_probs_b.bias"), &[cfg.expert_count])?;
        self.expect(&format!("{prefix}.ffn_gate_shexp.weight"), &[cfg.hidden_size, cfg.expert_intermediate_size])?;
        self.expect(&format!("{prefix}.ffn_up_shexp.weight"), &[cfg.hidden_size, cfg.expert_intermediate_size])?;
        self.expect(&format!("{prefix}.ffn_down_shexp.weight"), &[cfg.expert_intermediate_size, cfg.hidden_size])?;
        self.expect(&format!("{prefix}.ffn_gate_exps.weight"), &[cfg.hidden_size, cfg.expert_intermediate_size, cfg.expert_count])?;
        self.expect(&format!("{prefix}.ffn_up_exps.weight"), &[cfg.hidden_size, cfg.expert_intermediate_size, cfg.expert_count])?;
        self.expect(&format!("{prefix}.ffn_down_exps.weight"), &[cfg.expert_intermediate_size, cfg.hidden_size, cfg.expert_count])?;
        Ok(())
    }

    fn expect(&self, name: &str, dims: &[usize]) -> Result<&crate::weight::container::gguf::GgufTensorInfo, String> {
        self.reader.expect_tensor(name, dims)
    }

    fn expect_f32(&self, name: &str, dims: &[usize]) -> Result<(), String> {
        let tensor = self.expect(name, dims)?;
        if tensor.tensor_type != GgmlType(0) {
            return Err(format!("{name} type={}，期望 F32", tensor.tensor_type.name()));
        }
        Ok(())
    }

    pub fn load_dense_layer(&self, layer: usize) -> Result<GgufDenseLayer, String> {
        let cfg = &self.cfg;
        let prefix = format!("blk.{layer}");
        Ok(GgufDenseLayer {
            indexer: self.load_indexer(layer, &prefix)?,
            input_norm: self.vector_f32(&format!("{prefix}.attn_norm.weight"), cfg.hidden_size)?,
            q_a_proj: self.matrix(&format!("{prefix}.attn_q_a.weight"))?,
            q_a_norm: self.vector_f32(&format!("{prefix}.attn_q_a_norm.weight"), cfg.q_lora_rank)?,
            q_b_proj: self.matrix(&format!("{prefix}.attn_q_b.weight"))?,
            kv_a_proj: self.matrix(&format!("{prefix}.attn_kv_a_mqa.weight"))?,
            kv_a_norm: self.vector_f32(&format!("{prefix}.attn_kv_a_norm.weight"), cfg.kv_lora_rank)?,
            kv_b_w8: self.fused_kv_b_w8(&prefix)?,
            o_proj: self.matrix(&format!("{prefix}.attn_output.weight"))?,
            post_attn_norm: self.vector_f32(&format!("{prefix}.ffn_norm.weight"), cfg.hidden_size)?,
            gate_proj: self.matrix(&format!("{prefix}.ffn_gate.weight"))?,
            up_proj: self.matrix(&format!("{prefix}.ffn_up.weight"))?,
            down_proj: self.matrix(&format!("{prefix}.ffn_down.weight"))?,
        })
    }

    pub fn load_moe_layer(&self, layer: usize) -> Result<GgufMoeLayer, String> {
        let prefix = format!("blk.{layer}");
        self.load_moe_layer_at(&prefix, layer)
    }

    /// MTP 内层 MoE 与普通 MoE 层同构；prefix 可指向 blk.78。
    fn load_moe_layer_at(&self, prefix: &str, layer: usize) -> Result<GgufMoeLayer, String> {
        let cfg = &self.cfg;
        Ok(GgufMoeLayer {
            indexer: self.load_indexer(layer, prefix)?,
            input_norm: self.vector_f32(&format!("{prefix}.attn_norm.weight"), cfg.hidden_size)?,
            q_a_proj: self.matrix(&format!("{prefix}.attn_q_a.weight"))?,
            q_a_norm: self.vector_f32(&format!("{prefix}.attn_q_a_norm.weight"), cfg.q_lora_rank)?,
            q_b_proj: self.matrix(&format!("{prefix}.attn_q_b.weight"))?,
            kv_a_proj: self.matrix(&format!("{prefix}.attn_kv_a_mqa.weight"))?,
            kv_a_norm: self.vector_f32(&format!("{prefix}.attn_kv_a_norm.weight"), cfg.kv_lora_rank)?,
            kv_b_w8: self.fused_kv_b_w8(prefix)?,
            o_proj: self.matrix(&format!("{prefix}.attn_output.weight"))?,
            post_attn_norm: self.vector_f32(&format!("{prefix}.ffn_norm.weight"), cfg.hidden_size)?,
            router_weight: self.vector_f32(&format!("{prefix}.ffn_gate_inp.weight"), cfg.expert_count * cfg.hidden_size)?,
            router_bias: self.vector_f32(&format!("{prefix}.exp_probs_b.bias"), cfg.expert_count)?,
            shared_gate: self.matrix(&format!("{prefix}.ffn_gate_shexp.weight"))?,
            shared_up: self.matrix(&format!("{prefix}.ffn_up_shexp.weight"))?,
            shared_down: self.matrix(&format!("{prefix}.ffn_down_shexp.weight"))?,
        })
    }

    pub fn load_mtp_layer(&self) -> Result<GgufMtpLayer, String> {
        let cfg = &self.cfg;
        let prefix = format!("blk.{}", cfg.layer_count);
        Ok(GgufMtpLayer {
            embedding_norm: self.vector_f32(&format!("{prefix}.nextn.enorm.weight"), cfg.hidden_size)?,
            hidden_norm: self.vector_f32(&format!("{prefix}.nextn.hnorm.weight"), cfg.hidden_size)?,
            input_projection: self.matrix(&format!("{prefix}.nextn.eh_proj.weight"))?,
            layer: self.load_moe_layer_at(&prefix, cfg.layer_count)?,
            output_norm: self.vector_f32(&format!("{prefix}.nextn.shared_head_norm.weight"), cfg.hidden_size)?,
        })
    }

    /// IndexShare 语义由 `is_indexer_layer` 决定，不按张量存在性探测：
    /// GGUF 导出可能携带共享层的冗余 indexer 权重，必须忽略。
    fn load_indexer(&self, layer: usize, prefix: &str) -> Result<Option<GgufIndexerWeights>, String> {
        if !is_indexer_layer(layer) {
            return Ok(None);
        }
        let cfg = &self.cfg;
        let base = format!("{prefix}.indexer");
        Ok(Some(GgufIndexerWeights {
            wq_b: self.matrix(&format!("{base}.attn_q_b.weight"))?,
            wk: self.matrix(&format!("{base}.attn_k.weight"))?,
            weights_proj: self.vector_f32(&format!("{base}.proj.weight"), cfg.index_heads * cfg.hidden_size)?,
            k_norm_weight: self.vector_f32(&format!("{base}.k_norm.weight"), cfg.index_head_dim)?,
            k_norm_bias: self.vector_f32(&format!("{base}.k_norm.bias"), cfg.index_head_dim)?,
        }))
    }

    fn fused_kv_b_w8(&self, prefix: &str) -> Result<W8A16Matrix, String> {
        let cfg = &self.cfg;
        let nope = cfg.q_projection_size / cfg.num_heads - cfg.qk_rope_head_dim;
        let v_dim = cfg.kv_projection_size / cfg.num_heads - nope;
        let head_dim = nope + v_dim;
        let lora = cfg.kv_lora_rank;
        let k_name = format!("{prefix}.attn_k_b.weight");
        let v_name = format!("{prefix}.attn_v_b.weight");
        let k = self.reader.read_tensor_f32(&k_name)?;
        let v = self.reader.read_tensor_f32(&v_name)?;
        if k.len() != nope * lora * cfg.num_heads || v.len() != v_dim * lora * cfg.num_heads {
            return Err(format!("{k_name}/{v_name} 元素数与 head 切分不符"));
        }
        let rows = cfg.kv_projection_size;
        // 对称量化到 group-32 INT8：q = clamp(round(x/d))，d = max|x|/127。
        let mut packed = vec![0_u8; rows * lora];
        let mut scales = vec![0_u8; rows * (lora / 32) * 2];
        let scale_writer = |scales: &mut [u8], row: usize, group: usize, value: f32| {
            let bits = half::f16::from_f32(value).to_le_bytes();
            let base = (row * (lora / 32) + group) * 2;
            scales[base] = bits[0];
            scales[base + 1] = bits[1];
        };
        // GGUF ne 顺序 [c, j, head]：index = c + dim*(j + lora*head)，j 维 stride=dim，必须逐点索引。
        let mut fused = vec![0.0f32; lora];
        for head in 0..cfg.num_heads {
            for c in 0..nope {
                for j in 0..lora {
                    fused[j] = k[c + nope * (j + lora * head)];
                }
                quantize_row_w8(&mut packed, &mut scales, head * head_dim + c, &fused, scale_writer);
            }
            for o in 0..v_dim {
                for j in 0..lora {
                    fused[j] = v[j + lora * (o + v_dim * head)];
                }
                quantize_row_w8(&mut packed, &mut scales, head * head_dim + nope + o, &fused, scale_writer);
            }
        }
        W8A16Matrix::new(packed, scales, ScaleDType::F16, 32, rows, lora).map_err(|error| format!("{k_name} W8 重排: {error}"))
    }

    fn matrix(&self, name: &str) -> Result<GgufMatrix, String> {
        self.reader.read_matrix(name)
    }

    fn vector_f32(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let values = self.reader.read_tensor_f32(name)?;
        if values.len() != len {
            return Err(format!("{name} 元素数 {}，期望 {len}", values.len()));
        }
        Ok(values)
    }
}

impl GgufExpertSource for Glm52Gguf {
    fn intermediate(&self) -> usize {
        self.cfg.expert_intermediate_size
    }

    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }

    fn load_expert_gguf(&self, layer: usize, expert: usize) -> Result<GgufExpertWeights, String> {
        // layer 上限含 MTP 层(blk.{layer_count})。
        if layer >= self.cfg.layer_count + self.cfg.mtp_layer_count || expert >= self.cfg.expert_count {
            return Err(format!("GLM-5.2 GGUF expert 越界: layer={layer}/{}, expert={expert}/{}", self.cfg.layer_count + self.cfg.mtp_layer_count, self.cfg.expert_count));
        }
        Ok(GgufExpertWeights {
            gate: self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_gate_exps.weight"), expert)?,
            up: self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_up_exps.weight"), expert)?,
            down: self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_down_exps.weight"), expert)?,
        })
    }
}

enum WeightSource {
    Official(SafetensorStore),
    Nvfp4(NvidiaNvfp4Model),
    CompressedTensors(CompressedTensorsSource),
    Gguf(Arc<Glm52Gguf>),
}

pub struct Glm52Weights {
    source: WeightSource,
    source_path: PathBuf,
    cfg: Glm52Config,
}

impl Glm52Weights {
    pub fn open(root: &Path, cfg: Glm52Config) -> Result<Self, String> {
        Self::open_official(root, cfg)
    }

    pub fn open_official(root: &Path, cfg: Glm52Config) -> Result<Self, String> {
        let store = SafetensorStore::open(root).map_err(|error| format!("打开官方权重 root {}: {error}", root.display()))?;
        Ok(Self { source: WeightSource::Official(store), source_path: canonical_source_path(root), cfg })
    }

    pub fn open_nvfp4(root: &Path, cfg: Glm52Config) -> Result<Self, String> {
        let model = NvidiaNvfp4Model::open(root, cfg.clone()).map_err(|error| format!("打开 NVIDIA NVFP4 权重 root {}: {error}", root.display()))?;
        Ok(Self { source: WeightSource::Nvfp4(model), source_path: canonical_source_path(root), cfg })
    }

    /// 打开 compressed-tensors 混合精度(W4A16+W8A16)权重。
    pub fn open_compressed_tensors(root: &Path, cfg: Glm52Config) -> Result<Self, String> {
        let source = CompressedTensorsSource::open(root).map_err(|error| format!("打开 compressed-tensors 权重 root {}: {error}", root.display()))?;
        Ok(Self { source: WeightSource::CompressedTensors(source), source_path: canonical_source_path(root), cfg })
    }

    /// 打开 GGUF(glm-dsa)权重;path 可指向单文件或多分片目录。
    pub fn open_gguf(path: &Path, cfg: Glm52Config) -> Result<Self, String> {
        let source = Glm52Gguf::open(path, cfg.clone()).map_err(|error| format!("打开 GLM-5.2 GGUF 权重 {}: {error}", path.display()))?;
        Ok(Self { source: WeightSource::Gguf(Arc::new(source)), source_path: canonical_source_path(path), cfg })
    }

    pub fn cfg(&self) -> &Glm52Config {
        &self.cfg
    }

    pub fn cache_source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn cache_quantization(&self) -> String {
        match &self.source {
            WeightSource::Official(_) => "safetensors:official".to_owned(),
            WeightSource::Nvfp4(_) => "safetensors:nvfp4".to_owned(),
            WeightSource::CompressedTensors(source) => format!("compressed-tensors:w4a16+w8a16:g{}", source.group_size()),
            WeightSource::Gguf(source) => {
                let mut types = source.reader().tensors().iter().map(|tensor| tensor.tensor_type.name()).collect::<Vec<_>>();
                types.sort_unstable();
                types.dedup();
                let name = source.reader().metadata("general.name").and_then(|value| value.as_str()).unwrap_or("unknown");
                format!("gguf:{name}:{}", types.join("+"))
            }
        }
    }

    pub fn source_is_official(&self) -> bool {
        matches!(self.source, WeightSource::Official(_))
    }

    pub fn source_is_nvfp4(&self) -> bool {
        matches!(self.source, WeightSource::Nvfp4(_))
    }

    pub fn source_is_ct(&self) -> bool {
        matches!(self.source, WeightSource::CompressedTensors(_))
    }

    pub fn source_is_gguf(&self) -> bool {
        matches!(self.source, WeightSource::Gguf(_))
    }

    /// 返回 GGUF source 的 clone(runtime 用于构造 expert source 与 tokenizer)。
    pub fn gguf_source(&self) -> Result<Arc<Glm52Gguf>, String> {
        match &self.source {
            WeightSource::Gguf(source) => Ok(source.clone()),
            _ => Err("gguf_source 只在 GGUF 权重源可用时调用".to_owned()),
        }
    }

    /// 返回 CT source 的引用(runtime 用于构造 expert source)。
    pub fn ct_source(&self) -> Result<CompressedTensorsSource, String> {
        match &self.source {
            WeightSource::CompressedTensors(source) => Ok(source.clone()),
            _ => Err("ct_source 只在 compressed-tensors 权重源可用时调用".to_owned()),
        }
    }

    fn official_name(&self, candidates: &[&str], usage: &str) -> Result<String, String> {
        let store = self.official()?;
        candidates.iter().copied().find(|name| store.has(name)).map(str::to_owned).ok_or_else(|| format!("{usage} 官方权重名未命中: {candidates:?}"))
    }

    fn load_official_tensor_by_names(&self, candidates: &[&str], usage: &str) -> Result<TensorData, String> {
        let name = self.official_name(candidates, usage)?;
        self.official()?.load(&name).map_err(|error| format!("读取官方权重 {name}: {error}"))
    }

    fn ct(&self) -> Result<&CompressedTensorsSource, String> {
        match &self.source {
            WeightSource::CompressedTensors(source) => Ok(source),
            _ => Err("当前权重实例未使用 compressed-tensors 源".to_owned()),
        }
    }

    pub fn nvfp4_experts(&self) -> Option<NvidiaNvfp4Experts> {
        match &self.source {
            WeightSource::Nvfp4(model) => Some(model.experts()),
            WeightSource::Official(_) | WeightSource::CompressedTensors(_) | WeightSource::Gguf(_) => None,
        }
    }

    pub fn embedding_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let bytes = self.embedding_rows_bf16(token_ids)?;
        crate::weight::container::safetensor::decode_to_f32("model.embed_tokens.weight", "BF16", &bytes)
    }

    pub fn embedding_rows_bf16(&self, token_ids: &[u32]) -> Result<Vec<u8>, String> {
        if let WeightSource::Gguf(source) = &self.source {
            let rows = source.reader().embedding_rows("token_embd.weight", token_ids, self.cfg.hidden_size, self.cfg.vocab_size)?;
            let mut bytes = Vec::with_capacity(rows.len() * 2);
            for value in rows {
                bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
            }
            return Ok(bytes);
        }
        if let WeightSource::Nvfp4(model) = &self.source {
            return model.embedding_rows_bf16(token_ids);
        }
        if let WeightSource::CompressedTensors(source) = &self.source {
            let rows: Vec<usize> = token_ids.iter().map(|&token| token as usize).collect();
            let tensor = source.load_bf16_rows("model.embed_tokens.weight", &rows)?;
            if tensor.shape.len() != 2 || tensor.shape[1] != self.cfg.hidden_size {
                return Err(format!("model.embed_tokens.weight shape 异常: {:?}，期望 [N, {}]", tensor.shape, self.cfg.hidden_size));
            }
            return Ok(tensor.data);
        }
        let rows: Vec<usize> = token_ids.iter().map(|&token| token as usize).collect();
        let name = self.official_name(&["model.embed_tokens.weight", "transformer.embedding.word_embeddings.weight"], "embedding")?;
        let tensor = self.official()?.load_bf16_rows(&name, &rows)?;
        if tensor.shape.len() != 2 || tensor.shape[1] != self.cfg.hidden_size {
            return Err(format!("embedding shape 异常: {:?}", tensor.shape));
        }
        Ok(tensor.data)
    }

    pub fn load_dense_layer(&self, layer: usize) -> Result<DenseLayerF32, String> {
        let h = self.cfg.hidden_size;
        let attn = format!("model.layers.{layer}.self_attn");
        let mlp = format!("model.layers.{layer}.mlp");
        Ok(DenseLayerF32 {
            indexer: self.load_official_indexer(layer, &attn)?,
            input_norm: self.load_official_f32_vec(&format!("model.layers.{layer}.input_layernorm.weight"), h)?,
            q_a_proj: self.load_official_fp8(&format!("{attn}.q_a_proj.weight"), self.cfg.q_lora_rank, h)?,
            q_a_norm: self.load_official_f32_vec(&format!("{attn}.q_a_layernorm.weight"), self.cfg.q_lora_rank)?,
            q_b_proj: self.load_official_fp8(&format!("{attn}.q_b_proj.weight"), self.cfg.q_projection_size, self.cfg.q_lora_rank)?,
            kv_a_proj: self.load_official_fp8(&format!("{attn}.kv_a_proj_with_mqa.weight"), self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim, h)?,
            kv_a_norm: self.load_official_f32_vec(&format!("{attn}.kv_a_layernorm.weight"), self.cfg.kv_lora_rank)?,
            kv_b_proj: self.load_official_fp8(&format!("{attn}.kv_b_proj.weight"), self.cfg.kv_projection_size, self.cfg.kv_lora_rank)?,
            o_proj: self.load_official_fp8(&format!("{attn}.o_proj.weight"), h, self.cfg.q_projection_size)?,
            post_attn_norm: self.load_official_f32_vec(&format!("model.layers.{layer}.post_attention_layernorm.weight"), h)?,
            gate_proj: self.load_official_fp8(&format!("{mlp}.gate_proj.weight"), self.cfg.dense_intermediate_size, h)?,
            up_proj: self.load_official_fp8(&format!("{mlp}.up_proj.weight"), self.cfg.dense_intermediate_size, h)?,
            down_proj: self.load_official_fp8(&format!("{mlp}.down_proj.weight"), h, self.cfg.dense_intermediate_size)?,
        })
    }

    pub fn load_dense_layer_nvfp4(&self, layer: usize) -> Result<Nvfp4CoreDenseLayer, String> {
        self.nvfp4()?.load_dense_layer(layer)
    }

    pub fn load_dense_layer_ct(&self, layer: usize) -> Result<CtDenseLayer, String> {
        self.ct()?.load_dense_layer(layer)
    }

    pub fn load_dense_layer_gguf(&self, layer: usize) -> Result<GgufDenseLayer, String> {
        self.gguf()?.load_dense_layer(layer)
    }

    pub fn load_moe_layer(&self, layer: usize) -> Result<MoeLayerF32, String> {
        let h = self.cfg.hidden_size;
        let attn = format!("model.layers.{layer}.self_attn");
        let mlp = format!("model.layers.{layer}.mlp");
        Ok(MoeLayerF32 {
            indexer: self.load_official_indexer(layer, &attn)?,
            input_norm: self.load_official_f32_vec(&format!("model.layers.{layer}.input_layernorm.weight"), h)?,
            q_a_proj: self.load_official_fp8(&format!("{attn}.q_a_proj.weight"), self.cfg.q_lora_rank, h)?,
            q_a_norm: self.load_official_f32_vec(&format!("{attn}.q_a_layernorm.weight"), self.cfg.q_lora_rank)?,
            q_b_proj: self.load_official_fp8(&format!("{attn}.q_b_proj.weight"), self.cfg.q_projection_size, self.cfg.q_lora_rank)?,
            kv_a_proj: self.load_official_fp8(&format!("{attn}.kv_a_proj_with_mqa.weight"), self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim, h)?,
            kv_a_norm: self.load_official_f32_vec(&format!("{attn}.kv_a_layernorm.weight"), self.cfg.kv_lora_rank)?,
            kv_b_proj: self.load_official_fp8(&format!("{attn}.kv_b_proj.weight"), self.cfg.kv_projection_size, self.cfg.kv_lora_rank)?,
            o_proj: self.load_official_fp8(&format!("{attn}.o_proj.weight"), h, self.cfg.q_projection_size)?,
            post_attn_norm: self.load_official_f32_vec(&format!("model.layers.{layer}.post_attention_layernorm.weight"), h)?,
            router_weight: self.load_official_f32_matrix(&format!("{mlp}.gate.weight"), self.cfg.expert_count, h)?,
            router_bias: self.load_official_f32_vec(&format!("{mlp}.gate.e_score_correction_bias"), self.cfg.expert_count)?,
            shared_gate: self.load_official_fp8(&format!("{mlp}.shared_experts.gate_proj.weight"), self.cfg.expert_intermediate_size, h)?,
            shared_up: self.load_official_fp8(&format!("{mlp}.shared_experts.up_proj.weight"), self.cfg.expert_intermediate_size, h)?,
            shared_down: self.load_official_fp8(&format!("{mlp}.shared_experts.down_proj.weight"), h, self.cfg.expert_intermediate_size)?,
        })
    }

    fn load_official_indexer(&self, layer: usize, attention: &str) -> Result<Option<Glm52IndexerWeights>, String> {
        if !is_indexer_layer(layer) {
            return Ok(None);
        }
        let base = format!("{attention}.indexer");
        let query_size = self.cfg.index_heads * self.cfg.index_head_dim;
        Ok(Some(Glm52IndexerWeights {
            wq_b: self.load_official_fp8(&format!("{base}.wq_b.weight"), query_size, self.cfg.q_lora_rank)?,
            wk: self.load_official_fp8(&format!("{base}.wk.weight"), self.cfg.index_head_dim, self.cfg.hidden_size)?,
            weights_proj: self.load_official_bf16_matrix(&format!("{base}.weights_proj.weight"), self.cfg.index_heads, self.cfg.hidden_size)?,
            k_norm_weight: self.load_official_f32_vec(&format!("{base}.k_norm.weight"), self.cfg.index_head_dim)?,
            k_norm_bias: self.load_official_f32_vec(&format!("{base}.k_norm.bias"), self.cfg.index_head_dim)?,
        }))
    }

    pub fn load_moe_layer_nvfp4(&self, layer: usize) -> Result<Nvfp4CoreMoeLayer, String> {
        self.nvfp4()?.load_moe_layer(layer)
    }

    pub fn load_moe_layer_ct(&self, layer: usize) -> Result<CtMoeLayer, String> {
        self.ct()?.load_moe_layer(layer)
    }

    pub fn load_moe_layer_gguf(&self, layer: usize) -> Result<GgufMoeLayer, String> {
        self.gguf()?.load_moe_layer(layer)
    }

    pub fn load_mtp_layer_gguf(&self) -> Result<GgufMtpLayer, String> {
        self.gguf()?.load_mtp_layer()
    }

    pub fn final_norm(&self) -> Result<Vec<f32>, String> {
        match &self.source {
            WeightSource::Official(_) => self.load_official_f32_vec_names(&["model.norm.weight", "transformer.encoder.final_layernorm.weight"], self.cfg.hidden_size),
            WeightSource::Nvfp4(model) => model.final_norm(),
            WeightSource::CompressedTensors(source) => source.load_norm("model.norm.weight"),
            WeightSource::Gguf(source) => source.reader().read_tensor_f32("output_norm.weight").map_err(|error| format!("读取 GGUF output_norm.weight: {error}")),
        }
    }

    pub fn lm_head(&self) -> Result<Vec<f32>, String> {
        if let WeightSource::CompressedTensors(source) = &self.source {
            return source.load_norm("lm_head.weight");
        }
        self.load_official_f32_matrix_names(&["transformer.output_layer.weight", "lm_head.weight", "model.output_layer.weight"], self.cfg.vocab_size, self.cfg.hidden_size)
    }

    /// LM Head 直接保持 BF16 字节，避免先展开为双倍大小的 F32 常驻副本。
    pub fn lm_head_bf16_bytes(&self) -> Result<Vec<u8>, String> {
        let expected = self.cfg.vocab_size.checked_mul(self.cfg.hidden_size).ok_or("lm_head 元素数溢出")?;
        if let WeightSource::Gguf(source) = &self.source {
            let values = source.reader().read_tensor_f32("output.weight").map_err(|error| format!("读取 GGUF output.weight: {error}"))?;
            if values.len() != expected {
                return Err(format!("GGUF output.weight 元素数 {}，期望 {expected}", values.len()));
            }
            let mut bytes = Vec::with_capacity(expected * 2);
            for value in values {
                bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
            }
            return Ok(bytes);
        }
        let tensor = match &self.source {
            WeightSource::Official(_) => self.load_official_tensor_by_names(&["transformer.output_layer.weight", "lm_head.weight", "model.output_layer.weight"], "LM Head")?,
            WeightSource::CompressedTensors(source) => source.load_tensor("lm_head.weight")?,
            // GGUF 分支已在上方提前返回。
            WeightSource::Gguf(_) => return Err("GGUF lm_head 已在专用分支处理".to_owned()),
            WeightSource::Nvfp4(model) => {
                let values = model.lm_head()?;
                if values.len() != expected {
                    return Err(format!("lm_head 元素数 {}，期望 {expected}", values.len()));
                }
                let mut bytes = Vec::with_capacity(expected * 2);
                for value in values {
                    bytes.extend_from_slice(&bf16::from_f32(value.to_f32()).to_bits().to_le_bytes());
                }
                return Ok(bytes);
            }
        };
        if tensor.shape.as_slice() != [self.cfg.vocab_size, self.cfg.hidden_size] {
            return Err(format!("lm_head shape={:?}，期望 [{},{}]", tensor.shape, self.cfg.vocab_size, self.cfg.hidden_size));
        }
        let mut bytes = Vec::with_capacity(expected * 2);
        match tensor.dtype.as_str() {
            "BF16" if tensor.data.len() == expected * 2 => return Ok(tensor.data),
            "F16" if tensor.data.len() == expected * 2 => {
                for value in tensor.data.chunks_exact(2) {
                    let value = f16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32();
                    bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
                }
            }
            "F32" if tensor.data.len() == expected * 4 => {
                for value in tensor.data.chunks_exact(4) {
                    let value = f32::from_le_bytes([value[0], value[1], value[2], value[3]]);
                    bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
                }
            }
            _ => return Err(format!("lm_head dtype={} bytes={} 与元素数 {expected} 不兼容", tensor.dtype, tensor.data.len())),
        }
        Ok(bytes)
    }

    /// CT LM head 保持文件精度直接转成 F16，避免先展开为双倍大小的 F32 常驻副本。
    pub fn lm_head_f16(&self) -> Result<Vec<half::f16>, String> {
        let WeightSource::CompressedTensors(source) = &self.source else {
            return self.lm_head().map(|values| values.into_iter().map(half::f16::from_f32).collect());
        };
        let tensor = source.load_tensor("lm_head.weight")?;
        let expected = self.cfg.vocab_size.checked_mul(self.cfg.hidden_size).ok_or("lm_head 元素数溢出")?;
        if tensor.shape != [self.cfg.vocab_size, self.cfg.hidden_size] {
            return Err(format!("lm_head.weight shape={:?}，期望 [{},{}]", tensor.shape, self.cfg.vocab_size, self.cfg.hidden_size));
        }
        let mut values = Vec::with_capacity(expected);
        match tensor.dtype.as_str() {
            "BF16" => {
                for bytes in tensor.data.chunks_exact(2) {
                    values.push(half::f16::from_f32(half::bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()));
                }
            }
            "F16" => {
                for bytes in tensor.data.chunks_exact(2) {
                    values.push(half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])));
                }
            }
            "F32" => {
                for bytes in tensor.data.chunks_exact(4) {
                    values.push(half::f16::from_f32(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])));
                }
            }
            dtype => return Err(format!("lm_head.weight dtype={dtype} 不支持转 F16")),
        }
        if values.len() != expected {
            return Err(format!("lm_head.weight 元素数 {}，期望 {expected}", values.len()));
        }
        Ok(values)
    }

    pub fn lm_head_nvfp4(&self) -> Result<Vec<f16>, String> {
        self.nvfp4()?.lm_head()
    }

    fn official(&self) -> Result<&SafetensorStore, String> {
        match &self.source {
            WeightSource::Official(store) => Ok(store),
            WeightSource::Nvfp4(_) | WeightSource::CompressedTensors(_) | WeightSource::Gguf(_) => Err("当前权重实例未使用官方 safetensors 源".to_owned()),
        }
    }

    fn nvfp4(&self) -> Result<&NvidiaNvfp4Model, String> {
        match &self.source {
            WeightSource::Nvfp4(model) => Ok(model),
            WeightSource::Official(_) | WeightSource::CompressedTensors(_) | WeightSource::Gguf(_) => Err("当前权重实例未使用 NVIDIA NVFP4 源".to_owned()),
        }
    }

    fn gguf(&self) -> Result<&Glm52Gguf, String> {
        match &self.source {
            WeightSource::Gguf(source) => Ok(source),
            _ => Err("当前权重实例未使用 GGUF 源".to_owned()),
        }
    }

    fn load_official_tensor(&self, name: &str) -> Result<TensorData, String> {
        self.load_official_tensor_by_names(std::slice::from_ref(&name), name)
    }

    fn load_official_f32_vec_names(&self, names: &[&str], len: usize) -> Result<Vec<f32>, String> {
        let name = self.official_name(names, "官方向量权重")?;
        let tensor = self.load_official_tensor_by_names(&[name.as_str()], &name)?;
        if tensor.shape.as_slice() != [len] {
            return Err(format!("{name} shape 异常: {:?}", tensor.shape));
        }
        decode_float_tensor(&name, &tensor, len)
    }

    fn load_official_f32_vec(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        self.load_official_f32_vec_names(&[name], len)
    }

    fn load_official_f32_matrix_names(&self, names: &[&str], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
        let name = self.official_name(names, "官方矩阵权重")?;
        let tensor = self.load_official_tensor_by_names(&[name.as_str()], &name)?;
        if tensor.shape.as_slice() != [rows, cols] {
            return Err(format!("{name} shape 异常: {:?}", tensor.shape));
        }
        let len = rows.checked_mul(cols).ok_or_else(|| format!("{name}: 矩阵大小溢出"))?;
        decode_float_tensor(&name, &tensor, len)
    }

    fn load_official_f32_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>, String> {
        self.load_official_f32_matrix_names(&[name], rows, cols)
    }

    fn load_official_bf16_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<u8>, String> {
        let tensor = self.load_official_tensor(name)?;
        if tensor.dtype != "BF16" || tensor.shape.as_slice() != [rows, cols] {
            return Err(format!("{name} dtype={} shape={:?}，期望 BF16/[{rows},{cols}]", tensor.dtype, tensor.shape));
        }
        Ok(tensor.data)
    }

    fn load_official_fp8(&self, name: &str, rows: usize, cols: usize) -> Result<Fp8Matrix, String> {
        let tensor = self.load_official_tensor(name)?;
        if tensor.dtype != "F8_E4M3" || tensor.shape.as_slice() != [rows, cols] {
            return Err(format!("{name} dtype={} shape={:?}，期望 F8_E4M3/[{rows},{cols}]", tensor.dtype, tensor.shape));
        }
        let scale_name = format!("{name}_scale_inv");
        let scale = self.load_official_tensor(&scale_name)?;
        let expected_scale = [rows.div_ceil(128), cols.div_ceil(128)];
        if scale.dtype != "F32" || scale.shape.as_slice() != expected_scale {
            return Err(format!("{scale_name} dtype={} shape={:?}，期望 F32/{expected_scale:?}", scale.dtype, scale.shape));
        }
        Fp8Matrix::new(tensor.data, scale.data, rows, cols).map_err(|error| format!("{name}: {error}"))
    }
}

fn canonical_source_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned())
}

fn decode_float_tensor(name: &str, tensor: &TensorData, len: usize) -> Result<Vec<f32>, String> {
    let values = crate::weight::container::safetensor::decode_to_f32(name, &tensor.dtype, &tensor.data)?;
    if values.len() != len {
        return Err(format!("{name} dtype={} bytes={} 与元素数 {len} 不兼容", tensor.dtype, tensor.data.len()));
    }
    Ok(values)
}

pub struct Glm52IndexerWeights {
    pub wq_b: Fp8Matrix,
    pub wk: Fp8Matrix,
    pub weights_proj: Vec<u8>,
    pub k_norm_weight: Vec<f32>,
    pub k_norm_bias: Vec<f32>,
}

pub struct DenseLayerF32 {
    pub indexer: Option<Glm52IndexerWeights>,
    pub input_norm: Vec<f32>,
    pub q_a_proj: Fp8Matrix,
    pub q_a_norm: Vec<f32>,
    pub q_b_proj: Fp8Matrix,
    pub kv_a_proj: Fp8Matrix,
    pub kv_a_norm: Vec<f32>,
    pub kv_b_proj: Fp8Matrix,
    pub o_proj: Fp8Matrix,
    pub post_attn_norm: Vec<f32>,
    pub gate_proj: Fp8Matrix,
    pub up_proj: Fp8Matrix,
    pub down_proj: Fp8Matrix,
}

pub struct MoeLayerF32 {
    pub indexer: Option<Glm52IndexerWeights>,
    pub input_norm: Vec<f32>,
    pub q_a_proj: Fp8Matrix,
    pub q_a_norm: Vec<f32>,
    pub q_b_proj: Fp8Matrix,
    pub kv_a_proj: Fp8Matrix,
    pub kv_a_norm: Vec<f32>,
    pub kv_b_proj: Fp8Matrix,
    pub o_proj: Fp8Matrix,
    pub post_attn_norm: Vec<f32>,
    pub router_weight: Vec<f32>,
    pub router_bias: Vec<f32>,
    pub shared_gate: Fp8Matrix,
    pub shared_up: Fp8Matrix,
    pub shared_down: Fp8Matrix,
}

/// 对称 group-32 INT8 量化一行(packed 用 CT 的 +128 偏移编码)。
fn quantize_row_w8(packed: &mut [u8], scales: &mut [u8], row: usize, values: &[f32], write_scale: impl Fn(&mut [u8], usize, usize, f32)) {
    let lora = values.len();
    for (group, chunk) in values.chunks(32).enumerate() {
        let max = chunk.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let d = if max > 0.0 { max / 127.0 } else { 1.0 };
        let base = row * lora + group * 32;
        for (index, value) in chunk.iter().enumerate() {
            let quant = (value / d).round().clamp(-127.0, 127.0) as i32;
            packed[base + index] = (quant as u8).wrapping_add(128);
        }
        write_scale(scales, row, group, d);
    }
}
