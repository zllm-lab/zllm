//! MoonshotAI/Kimi-K3-Instruct 官方 checkpoint 的张量命名。

use std::{path::Path, sync::Arc};

use crate::weight::container::safetensor::{SafetensorStore, TensorData};
use crate::weight::expert_source::{ExpertSource, ExpertSourceProvider, Mxfp4ExpertSource, Mxfp4ExpertWeights};
use crate::weight::format::mxfp4::{Mxfp4Matrix, load_mxfp4_matrix};

pub const EMBEDDING: &str = "language_model.model.embed_tokens.weight";
pub const FINAL_NORM: &str = "language_model.model.norm.weight";
pub const OUTPUT_ATTN_RES_NORM: &str = "language_model.model.output_attn_res_norm.weight";
pub const OUTPUT_ATTN_RES_PROJ: &str = "language_model.model.output_attn_res_proj.weight";
pub const LM_HEAD: &str = "language_model.lm_head.weight";

#[derive(Clone, Debug)]
pub struct KimiK3LayerNames {
    prefix: String,
}

impl KimiK3LayerNames {
    pub fn new(layer: usize) -> Self {
        Self { prefix: format!("language_model.model.layers.{layer}") }
    }

    fn tensor(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.prefix)
    }

    pub fn common(&self) -> LayerCommonNames {
        LayerCommonNames {
            input_norm: self.tensor("input_layernorm.weight"),
            post_attention_norm: self.tensor("post_attention_layernorm.weight"),
            attention_res_norm: self.tensor("self_attention_res_norm.weight"),
            attention_res_proj: self.tensor("self_attention_res_proj.weight"),
            mlp_res_norm: self.tensor("mlp_res_norm.weight"),
            mlp_res_proj: self.tensor("mlp_res_proj.weight"),
        }
    }

    pub fn kda(&self) -> KdaNames {
        KdaNames {
            q: KdaConvPathNames { projection: self.tensor("self_attn.q_proj.weight"), convolution: self.tensor("self_attn.q_conv1d.weight") },
            k: KdaConvPathNames { projection: self.tensor("self_attn.k_proj.weight"), convolution: self.tensor("self_attn.k_conv1d.weight") },
            v: KdaConvPathNames { projection: self.tensor("self_attn.v_proj.weight"), convolution: self.tensor("self_attn.v_conv1d.weight") },
            a_log: self.tensor("self_attn.A_log"),
            f_a_projection: self.tensor("self_attn.f_a_proj.weight"),
            f_b_projection: self.tensor("self_attn.f_b_proj.weight"),
            dt_bias: self.tensor("self_attn.dt_bias"),
            beta_projection: self.tensor("self_attn.b_proj.weight"),
            gate_projection: self.tensor("self_attn.g_proj.weight"),
            output_norm: self.tensor("self_attn.o_norm.weight"),
            output_projection: self.tensor("self_attn.o_proj.weight"),
        }
    }

    pub fn gated_mla(&self) -> GatedMlaNames {
        GatedMlaNames {
            query: MlaQueryNames { a_projection: self.tensor("self_attn.q_a_proj.weight"), a_norm: self.tensor("self_attn.q_a_layernorm.weight"), b_projection: self.tensor("self_attn.q_b_proj.weight") },
            key_value: MlaKeyValueNames { a_projection: self.tensor("self_attn.kv_a_proj_with_mqa.weight"), a_norm: self.tensor("self_attn.kv_a_layernorm.weight"), b_projection: self.tensor("self_attn.kv_b_proj.weight") },
            gate_projection: self.tensor("self_attn.g_proj.weight"),
            output_projection: self.tensor("self_attn.o_proj.weight"),
        }
    }

    pub fn dense_mlp(&self) -> DenseMlpNames {
        DenseMlpNames { gate_projection: self.tensor("mlp.gate_proj.weight"), up_projection: self.tensor("mlp.up_proj.weight"), down_projection: self.tensor("mlp.down_proj.weight") }
    }

    pub fn latent_moe(&self) -> LatentMoeNames {
        LatentMoeNames {
            router: self.tensor("block_sparse_moe.gate.weight"),
            correction_bias: self.tensor("block_sparse_moe.gate.e_score_correction_bias"),
            routed_down_projection: self.tensor("block_sparse_moe.routed_expert_down_proj.weight"),
            routed_norm: self.tensor("block_sparse_moe.routed_expert_norm.weight"),
            routed_up_projection: self.tensor("block_sparse_moe.routed_expert_up_proj.weight"),
            shared: DenseMlpNames {
                gate_projection: self.tensor("block_sparse_moe.shared_experts.gate_proj.weight"),
                up_projection: self.tensor("block_sparse_moe.shared_experts.up_proj.weight"),
                down_projection: self.tensor("block_sparse_moe.shared_experts.down_proj.weight"),
            },
        }
    }

    pub fn expert(&self, expert: usize) -> LatentExpertNames {
        let prefix = format!("{}.block_sparse_moe.experts.{expert}", self.prefix);
        LatentExpertNames { gate_projection: Mxfp4TensorNames::new(&prefix, "w1"), down_projection: Mxfp4TensorNames::new(&prefix, "w2"), up_projection: Mxfp4TensorNames::new(&prefix, "w3") }
    }
}

#[derive(Clone, Debug)]
pub struct LayerCommonNames {
    pub input_norm: String,
    pub post_attention_norm: String,
    pub attention_res_norm: String,
    pub attention_res_proj: String,
    pub mlp_res_norm: String,
    pub mlp_res_proj: String,
}

#[derive(Clone, Debug)]
pub struct KdaConvPathNames {
    pub projection: String,
    pub convolution: String,
}

#[derive(Clone, Debug)]
pub struct KdaNames {
    pub q: KdaConvPathNames,
    pub k: KdaConvPathNames,
    pub v: KdaConvPathNames,
    pub a_log: String,
    pub f_a_projection: String,
    pub f_b_projection: String,
    pub dt_bias: String,
    pub beta_projection: String,
    pub gate_projection: String,
    pub output_norm: String,
    pub output_projection: String,
}

#[derive(Clone, Debug)]
pub struct MlaQueryNames {
    pub a_projection: String,
    pub a_norm: String,
    pub b_projection: String,
}

#[derive(Clone, Debug)]
pub struct MlaKeyValueNames {
    pub a_projection: String,
    pub a_norm: String,
    pub b_projection: String,
}

#[derive(Clone, Debug)]
pub struct GatedMlaNames {
    pub query: MlaQueryNames,
    pub key_value: MlaKeyValueNames,
    pub gate_projection: String,
    pub output_projection: String,
}

#[derive(Clone, Debug)]
pub struct DenseMlpNames {
    pub gate_projection: String,
    pub up_projection: String,
    pub down_projection: String,
}

#[derive(Clone, Debug)]
pub struct LatentMoeNames {
    pub router: String,
    pub correction_bias: String,
    pub routed_down_projection: String,
    pub routed_norm: String,
    pub routed_up_projection: String,
    pub shared: DenseMlpNames,
}

#[derive(Clone, Debug)]
pub struct Mxfp4TensorNames {
    pub packed: String,
    pub scale: String,
}

impl Mxfp4TensorNames {
    fn new(prefix: &str, projection: &str) -> Self {
        Self { packed: format!("{prefix}.{projection}.weight_packed"), scale: format!("{prefix}.{projection}.weight_scale") }
    }
}

#[derive(Clone, Debug)]
pub struct LatentExpertNames {
    /// 官方 `w1`，即 gate projection。
    pub gate_projection: Mxfp4TensorNames,
    /// 官方 `w2`，即 down projection。
    pub down_projection: Mxfp4TensorNames,
    /// 官方 `w3`，即 up projection。
    pub up_projection: Mxfp4TensorNames,
}

pub struct KimiK3LayerCommonWeights {
    pub input_norm: TensorData,
    pub post_attention_norm: TensorData,
    pub attention_res_norm: TensorData,
    pub attention_res_projection: TensorData,
    pub mlp_res_norm: TensorData,
    pub mlp_res_projection: TensorData,
}

pub struct KimiK3KdaConvWeights {
    pub projection: TensorData,
    pub convolution: TensorData,
}

pub struct KimiK3KdaWeights {
    pub q: KimiK3KdaConvWeights,
    pub k: KimiK3KdaConvWeights,
    pub v: KimiK3KdaConvWeights,
    pub a_log: TensorData,
    pub f_a_projection: TensorData,
    pub f_b_projection: TensorData,
    pub dt_bias: TensorData,
    pub beta_projection: TensorData,
    pub gate_projection: TensorData,
    pub output_norm: TensorData,
    pub output_projection: TensorData,
}

pub struct KimiK3MlaQueryWeights {
    pub a_projection: TensorData,
    pub a_norm: TensorData,
    pub b_projection: TensorData,
}

pub struct KimiK3MlaKeyValueWeights {
    pub a_projection: TensorData,
    pub a_norm: TensorData,
    pub b_projection: TensorData,
}

pub struct KimiK3GatedMlaWeights {
    pub query: KimiK3MlaQueryWeights,
    pub key_value: KimiK3MlaKeyValueWeights,
    pub gate_projection: TensorData,
    pub output_projection: TensorData,
}

pub struct KimiK3DenseMlpWeights {
    pub gate_projection: TensorData,
    pub up_projection: TensorData,
    pub down_projection: TensorData,
}

pub struct KimiK3LatentMoeWeights {
    pub router: TensorData,
    pub correction_bias: TensorData,
    pub routed_down_projection: TensorData,
    pub routed_norm: TensorData,
    pub routed_up_projection: TensorData,
    pub shared: KimiK3DenseMlpWeights,
}

pub type KimiK3LatentExpertWeights = Mxfp4ExpertWeights;

#[derive(Clone)]
pub struct KimiK3ExpertSource {
    weights: KimiK3Weights,
    layer_count: usize,
    expert_count: usize,
    latent_size: usize,
    intermediate_size: usize,
}

/// K3 官方 safetensors 数据源；模型编排和设备资源生命周期由 runtime/backend 管理。
#[derive(Clone)]
pub struct KimiK3Weights {
    store: Arc<SafetensorStore>,
}

impl KimiK3Weights {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        Ok(Self { store: Arc::new(SafetensorStore::open(root.as_ref())?) })
    }

    pub fn expert_source(&self, layer_count: usize, expert_count: usize, latent_size: usize, intermediate_size: usize) -> Result<KimiK3ExpertSource, String> {
        if layer_count == 0 || expert_count == 0 || latent_size == 0 || intermediate_size == 0 {
            return Err(format!("Kimi K3 expert source 尺寸必须非零：layers={layer_count}, experts={expert_count}, latent={latent_size}, intermediate={intermediate_size}"));
        }
        Ok(KimiK3ExpertSource { weights: self.clone(), layer_count, expert_count, latent_size, intermediate_size })
    }

    pub fn embedding_rows_bf16(&self, token_ids: &[u32], vocab_size: usize, hidden_size: usize) -> Result<Vec<u8>, String> {
        let rows = token_ids.iter().map(|&token| token as usize).collect::<Vec<_>>();
        if let Some(token) = rows.iter().find(|&&token| token >= vocab_size) {
            return Err(format!("Kimi K3 token {token} 越界于 vocab_size={vocab_size}"));
        }
        let tensor = self.store.load_bf16_rows(EMBEDDING, &rows)?;
        if tensor.shape != [rows.len(), hidden_size] {
            return Err(format!("Kimi K3 embedding rows shape {:?}，期望 [{},{}]", tensor.shape, rows.len(), hidden_size));
        }
        Ok(tensor.data)
    }

    pub fn final_norm(&self) -> Result<TensorData, String> {
        self.load_bf16(FINAL_NORM)
    }

    pub fn output_attn_res(&self) -> Result<(TensorData, TensorData), String> {
        Ok((self.load_bf16(OUTPUT_ATTN_RES_NORM)?, self.load_bf16(OUTPUT_ATTN_RES_PROJ)?))
    }

    /// LM head 体积很大，只在 backend 准备常驻输出投影时显式读取。
    pub fn lm_head(&self) -> Result<TensorData, String> {
        self.load_bf16(LM_HEAD)
    }

    pub fn load_layer_common(&self, layer: usize) -> Result<KimiK3LayerCommonWeights, String> {
        let names = KimiK3LayerNames::new(layer).common();
        Ok(KimiK3LayerCommonWeights {
            input_norm: self.load_bf16(&names.input_norm)?,
            post_attention_norm: self.load_bf16(&names.post_attention_norm)?,
            attention_res_norm: self.load_bf16(&names.attention_res_norm)?,
            attention_res_projection: self.load_bf16(&names.attention_res_proj)?,
            mlp_res_norm: self.load_bf16(&names.mlp_res_norm)?,
            mlp_res_projection: self.load_bf16(&names.mlp_res_proj)?,
        })
    }

    pub fn load_kda(&self, layer: usize) -> Result<KimiK3KdaWeights, String> {
        let names = KimiK3LayerNames::new(layer).kda();
        Ok(KimiK3KdaWeights {
            q: self.load_kda_conv(names.q)?,
            k: self.load_kda_conv(names.k)?,
            v: self.load_kda_conv(names.v)?,
            a_log: self.load_f32(&names.a_log)?,
            f_a_projection: self.load_bf16(&names.f_a_projection)?,
            f_b_projection: self.load_bf16(&names.f_b_projection)?,
            dt_bias: self.load_f32(&names.dt_bias)?,
            beta_projection: self.load_bf16(&names.beta_projection)?,
            gate_projection: self.load_bf16(&names.gate_projection)?,
            output_norm: self.load_bf16(&names.output_norm)?,
            output_projection: self.load_bf16(&names.output_projection)?,
        })
    }

    pub fn load_gated_mla(&self, layer: usize) -> Result<KimiK3GatedMlaWeights, String> {
        let names = KimiK3LayerNames::new(layer).gated_mla();
        Ok(KimiK3GatedMlaWeights {
            query: KimiK3MlaQueryWeights { a_projection: self.load_bf16(&names.query.a_projection)?, a_norm: self.load_bf16(&names.query.a_norm)?, b_projection: self.load_bf16(&names.query.b_projection)? },
            key_value: KimiK3MlaKeyValueWeights { a_projection: self.load_bf16(&names.key_value.a_projection)?, a_norm: self.load_bf16(&names.key_value.a_norm)?, b_projection: self.load_bf16(&names.key_value.b_projection)? },
            gate_projection: self.load_bf16(&names.gate_projection)?,
            output_projection: self.load_bf16(&names.output_projection)?,
        })
    }

    pub fn load_dense_mlp(&self, layer: usize) -> Result<KimiK3DenseMlpWeights, String> {
        let names = KimiK3LayerNames::new(layer).dense_mlp();
        self.load_dense(names)
    }

    pub fn load_latent_moe(&self, layer: usize) -> Result<KimiK3LatentMoeWeights, String> {
        let names = KimiK3LayerNames::new(layer).latent_moe();
        Ok(KimiK3LatentMoeWeights {
            router: self.load_bf16(&names.router)?,
            correction_bias: self.load_bf16(&names.correction_bias)?,
            routed_down_projection: self.load_bf16(&names.routed_down_projection)?,
            routed_norm: self.load_bf16(&names.routed_norm)?,
            routed_up_projection: self.load_bf16(&names.routed_up_projection)?,
            shared: self.load_dense(names.shared)?,
        })
    }

    pub fn load_latent_expert(&self, layer: usize, expert: usize, latent_size: usize, intermediate_size: usize) -> Result<KimiK3LatentExpertWeights, String> {
        let names = KimiK3LayerNames::new(layer).expert(expert);
        Ok(KimiK3LatentExpertWeights {
            gate: self.load_mxfp4(names.gate_projection, intermediate_size, latent_size)?,
            down: self.load_mxfp4(names.down_projection, latent_size, intermediate_size)?,
            up: self.load_mxfp4(names.up_projection, intermediate_size, latent_size)?,
        })
    }

    fn load_kda_conv(&self, names: KdaConvPathNames) -> Result<KimiK3KdaConvWeights, String> {
        Ok(KimiK3KdaConvWeights { projection: self.load_bf16(&names.projection)?, convolution: self.load_bf16(&names.convolution)? })
    }

    fn load_dense(&self, names: DenseMlpNames) -> Result<KimiK3DenseMlpWeights, String> {
        Ok(KimiK3DenseMlpWeights { gate_projection: self.load_bf16(&names.gate_projection)?, up_projection: self.load_bf16(&names.up_projection)?, down_projection: self.load_bf16(&names.down_projection)? })
    }

    fn load_mxfp4(&self, names: Mxfp4TensorNames, rows: usize, cols: usize) -> Result<Mxfp4Matrix, String> {
        load_mxfp4_matrix(&self.store, &names.packed, &names.scale, rows, cols)
    }

    fn load_bf16(&self, name: &str) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        if tensor.dtype != "BF16" {
            return Err(format!("{} dtype={}，Kimi K3 官方 core 期望 BF16", tensor.name, tensor.dtype));
        }
        Ok(tensor)
    }

    fn load_f32(&self, name: &str) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        if tensor.dtype != "F32" {
            return Err(format!("{} dtype={}，Kimi K3 KDA 状态参数期望 F32", tensor.name, tensor.dtype));
        }
        Ok(tensor)
    }
}

impl Mxfp4ExpertSource for KimiK3ExpertSource {
    fn intermediate(&self) -> usize {
        self.intermediate_size
    }

    fn hidden(&self) -> usize {
        self.latent_size
    }

    fn load_expert_mxfp4(&self, layer: usize, expert: usize) -> Result<Mxfp4ExpertWeights, String> {
        if layer >= self.layer_count || expert >= self.expert_count {
            return Err(format!("Kimi K3 MXFP4 expert 越界：layer={layer}/{}, expert={expert}/{}", self.layer_count, self.expert_count));
        }
        self.weights.load_latent_expert(layer, expert, self.latent_size, self.intermediate_size)
    }
}

impl ExpertSourceProvider for KimiK3ExpertSource {
    fn source(&self, layer: usize) -> Result<ExpertSource<'_>, String> {
        if layer >= self.layer_count {
            return Err(format!("Kimi K3 MXFP4 expert source layer 越界：{layer}/{}", self.layer_count));
        }
        Ok(ExpertSource::Mxfp4(self))
    }
}
