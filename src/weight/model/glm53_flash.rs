//! zai-org/GLM-5.3-Flash(`glm5_next`)官方 checkpoint 的张量命名与装配。
//!
//! KDA 层权重保持 BF16/F32;MLA 投影与全部 FFN/MoE 为官方 FP8
//! (F8_E4M3 codes + F32 `weight_scale_inv`,[128,128] 块)。
//! mHC 小张量(`hc_*`)无 `.weight` 后缀,与层内其它命名不同。
//! 视觉塔(`model.visual.*`,glm5_next_vision)整体保持 BF16,
//! 不参与官方 FP8 量化。

use std::{path::Path, sync::Arc};

use crate::weight::{
    Fp8Matrix,
    container::safetensor::{SafetensorStore, TensorData},
};

pub const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
pub const FINAL_NORM: &str = "model.language_model.norm.weight";
pub const LM_HEAD: &str = "lm_head.weight";
pub const VISION_PATCH_EMBED: &str = "model.visual.patch_embed.proj.weight";
pub const VISION_POST_LAYERNORM: &str = "model.visual.post_layernorm.weight";
pub const VISION_DOWNSAMPLE: &str = "model.visual.downsample.weight";
pub const VISION_MERGER_PROJ: &str = "model.visual.merger.proj.weight";
pub const VISION_MERGER_NORM: &str = "model.visual.merger.post_projection_norm";

/// 视觉塔装配尺寸;head_dim = hidden_size / num_heads。
#[derive(Clone, Debug, Default)]
pub struct Glm53VisionDims {
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub intermediate_size: usize,
    pub projection_intermediate_size: usize,
    /// downsample/merger 输出宽,必须与语言模型 hidden_size 一致。
    pub out_hidden_size: usize,
}

/// 装配需要的维度,来自 runtime 的 Glm53FlashConfig;只保存校验必需的副本。
#[derive(Clone)]
pub struct Glm53FlashDims {
    pub hidden_size: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub q_projection_size: usize,
    pub kv_projection_size: usize,
    pub kda_projection_size: usize,
    pub kda_num_heads: usize,
    pub kda_head_dim: usize,
    pub kda_short_conv_kernel: usize,
    pub kda_decay_rank: usize,
    pub dense_intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub expert_count: usize,
    pub index_num_heads: usize,
    pub index_head_dim: usize,
    pub index_kpool: usize,
    pub vision: Glm53VisionDims,
}

#[derive(Clone, Debug)]
pub struct Glm53FlashLayerNames {
    prefix: String,
}

impl Glm53FlashLayerNames {
    pub fn new(layer: usize) -> Self {
        Self { prefix: format!("model.language_model.layers.{layer}") }
    }

    fn tensor(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.prefix)
    }

    pub fn common(&self) -> Glm53LayerCommonNames {
        Glm53LayerCommonNames { input_norm: self.tensor("input_layernorm.weight"), post_attention_norm: self.tensor("post_attention_layernorm.weight") }
    }

    pub fn mhc(&self) -> Glm53MhcNames {
        let sub = |role: &str| Glm53MhcTrioNames { function: self.tensor(&format!("hc_{role}_fn")), base: self.tensor(&format!("hc_{role}_base")), scale: self.tensor(&format!("hc_{role}_scale")) };
        Glm53MhcNames { attention: sub("attn"), feedforward: sub("ffn") }
    }

    pub fn kda(&self) -> Glm53KdaNames {
        let conv = |name: &str| Glm53KdaConvNames { projection: self.tensor(&format!("self_attn.{name}_proj.weight")), convolution: self.tensor(&format!("self_attn.{name}_conv1d.weight")) };
        Glm53KdaNames {
            query: conv("q"),
            key: conv("k"),
            value: conv("v"),
            a_log: self.tensor("self_attn.A_log"),
            f_a_projection: self.tensor("self_attn.f_a_proj.weight"),
            f_b_projection: self.tensor("self_attn.f_b_proj.weight"),
            dt_bias: self.tensor("self_attn.dt_bias"),
            beta_projection: self.tensor("self_attn.b_proj.weight"),
            gate_input_projection: self.tensor("self_attn.g_a_proj.weight"),
            gate_output_projection: self.tensor("self_attn.g_b_proj.weight"),
            output_norm: self.tensor("self_attn.o_norm.weight"),
            output_projection: self.tensor("self_attn.o_proj.weight"),
        }
    }

    pub fn dsa_mla(&self) -> Glm53DsaMlaNames {
        Glm53DsaMlaNames {
            query_a: self.tensor("self_attn.q_a_proj.weight"),
            query_norm: self.tensor("self_attn.q_a_layernorm.weight"),
            query_b: self.tensor("self_attn.q_b_proj.weight"),
            kv_a: self.tensor("self_attn.kv_a_proj_with_mqa.weight"),
            kv_norm: self.tensor("self_attn.kv_a_layernorm.weight"),
            kv_b: self.tensor("self_attn.kv_b_proj.weight"),
            output: self.tensor("self_attn.o_proj.weight"),
            indexer: Glm53IndexerNames {
                query_projection: self.tensor("self_attn.indexer.wq_b.weight"),
                key_projection: self.tensor("self_attn.indexer.wk.weight"),
                head_weights_projection: self.tensor("self_attn.indexer.weights_proj.weight"),
                key_norm_weight: self.tensor("self_attn.indexer.k_norm.weight"),
                key_norm_bias: self.tensor("self_attn.indexer.k_norm.bias"),
                kpool_ape: self.tensor("self_attn.indexer.index_kpool_compress_ape"),
                kpool_gate: self.tensor("self_attn.indexer.index_kpool_compress_gate"),
            },
        }
    }

    pub fn dense_mlp(&self) -> Glm53DenseMlpNames {
        Glm53DenseMlpNames { gate: self.tensor("mlp.gate_proj.weight"), up: self.tensor("mlp.up_proj.weight"), down: self.tensor("mlp.down_proj.weight") }
    }

    pub fn moe(&self) -> Glm53MoeNames {
        Glm53MoeNames { router: self.tensor("mlp.gate.weight"), correction_bias: self.tensor("mlp.gate.e_score_correction_bias"), shared: self.dense_mlp_prefix("mlp.shared_experts") }
    }

    fn dense_mlp_prefix(&self, prefix: &str) -> Glm53DenseMlpNames {
        Glm53DenseMlpNames { gate: self.tensor(&format!("{prefix}.gate_proj.weight")), up: self.tensor(&format!("{prefix}.up_proj.weight")), down: self.tensor(&format!("{prefix}.down_proj.weight")) }
    }

    pub fn expert(&self, expert: usize) -> Glm53DenseMlpNames {
        self.dense_mlp_prefix(&format!("mlp.experts.{expert}"))
    }

    /// MTP 层(layers.{mtp_layer}):DSA-MLA 与 MoE 命名与主层同构,
    /// 额外有 eh_proj/enorm/hnorm/shared_head.norm,且无 hc_*。
    pub fn mtp(&self) -> Glm53MtpNames {
        Glm53MtpNames { embedding_norm: self.tensor("enorm.weight"), hidden_norm: self.tensor("hnorm.weight"), input_projection: self.tensor("eh_proj.weight"), shared_head_norm: self.tensor("shared_head.norm.weight") }
    }
}

#[derive(Clone, Debug)]
pub struct Glm53MtpNames {
    pub embedding_norm: String,
    pub hidden_norm: String,
    pub input_projection: String,
    pub shared_head_norm: String,
}

/// 视觉塔块命名(`model.visual.blocks.{layer}`)。norm 为无 bias 的 RMSNorm,
/// attention 为 fused qkv + per-head q/k norm,MLP 为带 bias 的 SwiGLU。
#[derive(Clone, Debug)]
pub struct Glm53VisionLayerNames {
    prefix: String,
}

impl Glm53VisionLayerNames {
    pub fn new(layer: usize) -> Self {
        Self { prefix: format!("model.visual.blocks.{layer}") }
    }

    fn tensor(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.prefix)
    }

    pub fn block(&self) -> Glm53VisionBlockNames {
        Glm53VisionBlockNames {
            input_norm: self.tensor("norm1.weight"),
            post_attention_norm: self.tensor("norm2.weight"),
            attention: Glm53VisionAttentionNames {
                query_key_value: self.tensor("attn.qkv.weight"),
                query_key_value_bias: self.tensor("attn.qkv.bias"),
                query_norm: self.tensor("attn.q_norm.weight"),
                key_norm: self.tensor("attn.k_norm.weight"),
                output: self.tensor("attn.proj.weight"),
                output_bias: self.tensor("attn.proj.bias"),
            },
            mlp: Glm53VisionMlpNames {
                gate: self.tensor("mlp.gate_proj.weight"),
                gate_bias: self.tensor("mlp.gate_proj.bias"),
                up: self.tensor("mlp.up_proj.weight"),
                up_bias: self.tensor("mlp.up_proj.bias"),
                down: self.tensor("mlp.down_proj.weight"),
                down_bias: self.tensor("mlp.down_proj.bias"),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct Glm53VisionAttentionNames {
    pub query_key_value: String,
    pub query_key_value_bias: String,
    pub query_norm: String,
    pub key_norm: String,
    pub output: String,
    pub output_bias: String,
}

#[derive(Clone, Debug)]
pub struct Glm53VisionMlpNames {
    pub gate: String,
    pub gate_bias: String,
    pub up: String,
    pub up_bias: String,
    pub down: String,
    pub down_bias: String,
}

#[derive(Clone, Debug)]
pub struct Glm53VisionBlockNames {
    pub input_norm: String,
    pub post_attention_norm: String,
    pub attention: Glm53VisionAttentionNames,
    pub mlp: Glm53VisionMlpNames,
}

#[derive(Clone, Debug)]
pub struct Glm53LayerCommonNames {
    pub input_norm: String,
    pub post_attention_norm: String,
}

#[derive(Clone, Debug)]
pub struct Glm53MhcTrioNames {
    pub function: String,
    pub base: String,
    pub scale: String,
}

#[derive(Clone, Debug)]
pub struct Glm53MhcNames {
    pub attention: Glm53MhcTrioNames,
    pub feedforward: Glm53MhcTrioNames,
}

#[derive(Clone, Debug)]
pub struct Glm53KdaConvNames {
    pub projection: String,
    pub convolution: String,
}

#[derive(Clone, Debug)]
pub struct Glm53KdaNames {
    pub query: Glm53KdaConvNames,
    pub key: Glm53KdaConvNames,
    pub value: Glm53KdaConvNames,
    pub a_log: String,
    pub f_a_projection: String,
    pub f_b_projection: String,
    pub dt_bias: String,
    pub beta_projection: String,
    pub gate_input_projection: String,
    pub gate_output_projection: String,
    pub output_norm: String,
    pub output_projection: String,
}

#[derive(Clone, Debug)]
pub struct Glm53IndexerNames {
    pub query_projection: String,
    pub key_projection: String,
    pub head_weights_projection: String,
    pub key_norm_weight: String,
    pub key_norm_bias: String,
    /// kpool 池内绝对位置嵌入 [kpool, head_dim]。
    pub kpool_ape: String,
    /// kpool 池化 gate 投影 [head_dim, hidden]。
    pub kpool_gate: String,
}

#[derive(Clone, Debug)]
pub struct Glm53DsaMlaNames {
    pub query_a: String,
    pub query_norm: String,
    pub query_b: String,
    pub kv_a: String,
    pub kv_norm: String,
    pub kv_b: String,
    pub output: String,
    pub indexer: Glm53IndexerNames,
}

#[derive(Clone, Debug)]
pub struct Glm53DenseMlpNames {
    pub gate: String,
    pub up: String,
    pub down: String,
}

#[derive(Clone, Debug)]
pub struct Glm53MoeNames {
    pub router: String,
    pub correction_bias: String,
    pub shared: Glm53DenseMlpNames,
}

pub struct Glm53LayerCommonWeights {
    pub input_norm: TensorData,
    pub post_attention_norm: TensorData,
}

pub struct Glm53MhcTrio {
    /// function 是 [mixes, copies*hidden] 的 BF16 线性权重(实测 dtype,非 F32)。
    pub function: TensorData,
    pub base: Vec<f32>,
    pub scale: Vec<f32>,
}

pub struct Glm53MhcWeights {
    pub attention: Glm53MhcTrio,
    pub feedforward: Glm53MhcTrio,
}

pub struct Glm53KdaConvWeights {
    pub projection: TensorData,
    pub convolution: TensorData,
}

pub struct Glm53KdaLayerWeights {
    pub query: Glm53KdaConvWeights,
    pub key: Glm53KdaConvWeights,
    pub value: Glm53KdaConvWeights,
    pub a_log: Vec<f32>,
    pub f_a_projection: TensorData,
    pub f_b_projection: TensorData,
    pub dt_bias: Vec<f32>,
    pub beta_projection: TensorData,
    pub gate_input_projection: TensorData,
    pub gate_output_projection: TensorData,
    pub output_norm: TensorData,
    pub output_projection: TensorData,
}

pub struct Glm53DsaMlaLayerWeights {
    pub query_a: Fp8Matrix,
    pub query_norm: TensorData,
    pub query_b: Fp8Matrix,
    pub kv_a: Fp8Matrix,
    pub kv_norm: TensorData,
    /// kv_b 保持 BF16(实测:唯一未量化的 MLA 大投影,在 modules_to_not_convert 外的例外)。
    pub kv_b: TensorData,
    pub output: Fp8Matrix,
    pub indexer_query: TensorData,
    pub indexer_key: TensorData,
    pub indexer_head_weights: TensorData,
    pub indexer_key_norm_weight: TensorData,
    pub indexer_key_norm_bias: TensorData,
    pub indexer_kpool_ape: TensorData,
    pub indexer_kpool_gate: TensorData,
}

pub struct Glm53DenseMlpWeights {
    pub gate: Fp8Matrix,
    pub up: Fp8Matrix,
    pub down: Fp8Matrix,
}

pub struct Glm53MoeRouterWeights {
    pub router: Vec<f32>,
    pub correction_bias: Vec<f32>,
}

/// MTP 头:eh_proj 与三个 norm(BF16);transformer 层部分经
/// `load_dsa_mla`/`load_moe_router`/`load_shared_experts`/`load_expert` 复用。
pub struct Glm53MtpWeights {
    pub embedding_norm: TensorData,
    pub hidden_norm: TensorData,
    pub input_projection: TensorData,
    pub shared_head_norm: TensorData,
}

/// conv3d patch embed:kernel 与 stride 同为 (temporal,patch,patch),
/// 无重叠窗口,checkpoint 展平后直接按 rank-2 线性权重消费。
pub struct Glm53VisionPatchEmbedWeights {
    /// [hidden, 3*temporal*patch*patch] 的 BF16 线性权重。
    pub projection: TensorData,
    pub bias: TensorData,
}

pub struct Glm53VisionAttentionWeights {
    pub query_key_value: TensorData,
    pub query_key_value_bias: TensorData,
    pub query_norm: TensorData,
    pub key_norm: TensorData,
    pub output: TensorData,
    pub output_bias: TensorData,
}

/// gate/up 输入独立可并行;clamp 语义在激活 kernel(SiluClamped)内。
pub struct Glm53VisionMlpWeights {
    pub gate: TensorData,
    pub gate_bias: TensorData,
    pub up: TensorData,
    pub up_bias: TensorData,
    pub down: TensorData,
    pub down_bias: TensorData,
}

pub struct Glm53VisionBlockWeights {
    pub input_norm: TensorData,
    pub post_attention_norm: TensorData,
    pub attention: Glm53VisionAttentionWeights,
    pub mlp: Glm53VisionMlpWeights,
}

/// 视觉塔尾部:post RMSNorm → 2×2 downsample → merger。
/// downsample 的 conv2d 权重已在装载时置换为 [out, merge²*hidden] 线性布局,
/// 与 `merge_spatial` 的 (u,v) patch 行主序拼接一致。
pub struct Glm53VisionTailWeights {
    pub post_layernorm: TensorData,
    pub downsample: TensorData,
    pub downsample_bias: TensorData,
    pub merger_projection: TensorData,
    pub merger_norm_weight: TensorData,
    pub merger_norm_bias: TensorData,
    pub merger_gate: TensorData,
    pub merger_up: TensorData,
    pub merger_down: TensorData,
}

/// GLM-5.3-Flash 官方 safetensors 数据源;模型编排与设备资源由 runtime/backend 管理。
#[derive(Clone)]
pub struct Glm53FlashWeights {
    store: Arc<SafetensorStore>,
    dims: Glm53FlashDims,
}

impl Glm53FlashWeights {
    pub fn open(root: impl AsRef<Path>, dims: Glm53FlashDims) -> Result<Self, String> {
        if dims.hidden_size == 0 || dims.expert_count == 0 {
            return Err(format!("GLM-5.3-Flash 装配尺寸非法: hidden={} experts={}", dims.hidden_size, dims.expert_count));
        }
        let vision = &dims.vision;
        if vision.hidden_size == 0
            || vision.depth == 0
            || vision.num_heads == 0
            || vision.out_hidden_size == 0
            || !vision.hidden_size.is_multiple_of(vision.num_heads)
            || vision.patch_size == 0
            || vision.temporal_patch_size == 0
            || vision.spatial_merge_size == 0
        {
            return Err(format!("GLM-5.3-Flash 视觉装配尺寸非法: {vision:?}"));
        }
        Ok(Self { store: Arc::new(SafetensorStore::open(root.as_ref())?), dims })
    }

    pub fn dims(&self) -> &Glm53FlashDims {
        &self.dims
    }

    /// checkpoint 是否携带视觉塔张量;能力探测用,不装载任何权重。
    pub fn has_vision_tower(&self) -> bool {
        self.store.has(VISION_PATCH_EMBED)
    }

    /// kpool APE 平铺 f32([kpool * head_dim]),注入 DsaState 的 CPU 选择路径。
    pub fn kpool_ape_f32(&self, layer: usize) -> Result<Vec<f32>, String> {
        let names = Glm53FlashLayerNames::new(layer).dsa_mla();
        let tensor = self.load_bf16(&names.indexer.kpool_ape, self.dims.index_kpool, self.dims.index_head_dim)?;
        crate::weight::container::safetensor::decode_to_f32(&names.indexer.kpool_ape, "BF16", &tensor.data)
    }

    pub fn embedding_rows_bf16(&self, token_ids: &[u32], vocab_size: usize) -> Result<Vec<u8>, String> {
        let rows = token_ids.iter().map(|&token| token as usize).collect::<Vec<_>>();
        if let Some(token) = rows.iter().find(|&&token| token >= vocab_size) {
            return Err(format!("GLM-5.3-Flash token {token} 越界于 vocab_size={vocab_size}"));
        }
        let tensor = self.store.load_bf16_rows(EMBEDDING, &rows)?;
        if tensor.shape != [rows.len(), self.dims.hidden_size] {
            return Err(format!("GLM-5.3-Flash embedding rows shape {:?}，期望 [{},{}]", tensor.shape, rows.len(), self.dims.hidden_size));
        }
        Ok(tensor.data)
    }

    pub fn final_norm(&self) -> Result<TensorData, String> {
        self.load_bf16_vector(FINAL_NORM, self.dims.hidden_size)
    }

    /// LM head 体积很大,只在 backend 准备常驻输出投影时显式读取。
    pub fn lm_head(&self, vocab_size: usize) -> Result<TensorData, String> {
        self.load_bf16(LM_HEAD, vocab_size, self.dims.hidden_size)
    }

    pub fn load_layer_common(&self, layer: usize) -> Result<Glm53LayerCommonWeights, String> {
        let names = Glm53FlashLayerNames::new(layer).common();
        Ok(Glm53LayerCommonWeights { input_norm: self.load_bf16_vector(&names.input_norm, self.dims.hidden_size)?, post_attention_norm: self.load_bf16_vector(&names.post_attention_norm, self.dims.hidden_size)? })
    }

    /// mHC 投影。function 行数 = (2 + copies) * copies,base 同行数,scale 恒为 3;
    /// copies 由调用方传入,与 checkpoint 无关的部分在此校验一致性。
    pub fn load_mhc(&self, layer: usize, copies: usize) -> Result<Glm53MhcWeights, String> {
        let names = Glm53FlashLayerNames::new(layer).mhc();
        let mixes = (2 + copies) * copies;
        let trio = |names: &Glm53MhcTrioNames, label: &str| -> Result<Glm53MhcTrio, String> {
            Ok(Glm53MhcTrio {
                function: self.load_bf16(&names.function, mixes, copies * self.dims.hidden_size).map_err(|error| format!("({label}) {error}"))?,
                base: self.load_f32_vector(&names.base, mixes, label)?,
                scale: self.load_f32_vector(&names.scale, 3, label)?,
            })
        };
        Ok(Glm53MhcWeights { attention: trio(&names.attention, "attn")?, feedforward: trio(&names.feedforward, "ffn")? })
    }

    pub fn load_kda(&self, layer: usize) -> Result<Glm53KdaLayerWeights, String> {
        let names = Glm53FlashLayerNames::new(layer).kda();
        let heads = self.dims.kda_projection_size;
        let conv = |names: &Glm53KdaConvNames| -> Result<Glm53KdaConvWeights, String> {
            Ok(Glm53KdaConvWeights { projection: self.load_bf16(&names.projection, heads, self.dims.hidden_size)?, convolution: self.load_bf16_conv1d(&names.convolution, heads, self.dims.kda_short_conv_kernel)? })
        };
        Ok(Glm53KdaLayerWeights {
            query: conv(&names.query)?,
            key: conv(&names.key)?,
            value: conv(&names.value)?,
            a_log: self.load_small_f32(&names.a_log, "KDA a_log")?,
            f_a_projection: self.load_bf16(&names.f_a_projection, self.dims.kda_decay_rank, self.dims.hidden_size)?,
            f_b_projection: self.load_bf16(&names.f_b_projection, self.dims.kda_projection_size, self.dims.kda_decay_rank)?,
            dt_bias: self.load_small_f32(&names.dt_bias, "KDA dt_bias")?,
            beta_projection: self.load_bf16(&names.beta_projection, self.dims.kda_num_heads, self.dims.hidden_size)?,
            gate_input_projection: self.load_bf16(&names.gate_input_projection, self.dims.kda_decay_rank, self.dims.hidden_size)?,
            gate_output_projection: self.load_bf16(&names.gate_output_projection, self.dims.kda_projection_size, self.dims.kda_decay_rank)?,
            // o_norm 是一维 [head_dim],每 head 共享;非 [1, num_heads*head_dim]。
            output_norm: self.load_bf16_vector(&names.output_norm, self.dims.kda_head_dim)?,
            output_projection: self.load_bf16(&names.output_projection, self.dims.hidden_size, self.dims.kda_projection_size)?,
        })
    }

    pub fn load_dsa_mla(&self, layer: usize) -> Result<Glm53DsaMlaLayerWeights, String> {
        let names = Glm53FlashLayerNames::new(layer).dsa_mla();
        let dims = &self.dims;
        let index_query = dims.index_num_heads * dims.index_head_dim;
        Ok(Glm53DsaMlaLayerWeights {
            query_a: self.load_fp8(&names.query_a, dims.q_lora_rank, dims.hidden_size)?,
            query_norm: self.load_bf16_vector(&names.query_norm, dims.q_lora_rank)?,
            query_b: self.load_fp8(&names.query_b, dims.q_projection_size, dims.q_lora_rank)?,
            kv_a: self.load_fp8(&names.kv_a, dims.kv_lora_rank, dims.hidden_size)?,
            kv_norm: self.load_bf16_vector(&names.kv_norm, dims.kv_lora_rank)?,
            kv_b: self.load_bf16(&names.kv_b, dims.kv_projection_size, dims.kv_lora_rank)?,
            output: self.load_fp8(&names.output, dims.hidden_size, dims.q_projection_size)?,
            indexer_query: self.load_bf16(&names.indexer.query_projection, index_query, dims.q_lora_rank)?,
            indexer_key: self.load_bf16(&names.indexer.key_projection, dims.index_head_dim, dims.hidden_size)?,
            indexer_head_weights: self.load_bf16(&names.indexer.head_weights_projection, dims.index_num_heads, dims.hidden_size)?,
            indexer_key_norm_weight: self.load_bf16_vector(&names.indexer.key_norm_weight, dims.index_head_dim)?,
            indexer_key_norm_bias: self.load_bf16_vector(&names.indexer.key_norm_bias, dims.index_head_dim)?,
            indexer_kpool_ape: self.load_bf16(&names.indexer.kpool_ape, dims.index_kpool, dims.index_head_dim)?,
            indexer_kpool_gate: self.load_bf16(&names.indexer.kpool_gate, dims.index_head_dim, dims.hidden_size)?,
        })
    }

    pub fn load_dense_mlp(&self, layer: usize) -> Result<Glm53DenseMlpWeights, String> {
        let names = Glm53FlashLayerNames::new(layer).dense_mlp();
        self.load_dense_mlp_names(&names, self.dims.dense_intermediate_size)
    }

    pub fn load_moe_router(&self, layer: usize) -> Result<Glm53MoeRouterWeights, String> {
        let names = Glm53FlashLayerNames::new(layer).moe();
        Ok(Glm53MoeRouterWeights { router: self.decode_f32(&names.router, self.dims.expert_count, self.dims.hidden_size)?, correction_bias: self.load_f32_vector(&names.correction_bias, self.dims.expert_count, "MoE router bias")? })
    }

    pub fn load_shared_experts(&self, layer: usize) -> Result<Glm53DenseMlpWeights, String> {
        let names = Glm53FlashLayerNames::new(layer).moe().shared;
        self.load_dense_mlp_names(&names, self.dims.moe_intermediate_size)
    }

    pub fn load_expert(&self, layer: usize, expert: usize) -> Result<Glm53DenseMlpWeights, String> {
        if expert >= self.dims.expert_count {
            return Err(format!("GLM-5.3-Flash expert {expert} 越界于 {}", self.dims.expert_count));
        }
        let names = Glm53FlashLayerNames::new(layer).expert(expert);
        self.load_dense_mlp_names(&names, self.dims.moe_intermediate_size)
    }

    /// MTP 层号由调用方传入(通常 = layer_count);只加载 MTP 专属张量,
    /// DSA-MLA 与 MoE 部分用主层的 load_dsa_mla/load_moe_* 在同一层号上加载。
    pub fn load_mtp(&self, mtp_layer: usize) -> Result<Glm53MtpWeights, String> {
        let names = Glm53FlashLayerNames::new(mtp_layer).mtp();
        let hidden = self.dims.hidden_size;
        Ok(Glm53MtpWeights {
            embedding_norm: self.load_bf16_vector(&names.embedding_norm, hidden)?,
            hidden_norm: self.load_bf16_vector(&names.hidden_norm, hidden)?,
            input_projection: self.load_bf16(&names.input_projection, hidden, 2 * hidden)?,
            shared_head_norm: self.load_bf16_vector(&names.shared_head_norm, hidden)?,
        })
    }

    /// conv3d patch embed。safetensors 里的 [hidden, 3, temporal, patch, patch]
    /// 与 patch 张量列序 (c,t,y,x) 一致,直接按行展平为 rank-2。
    pub fn load_vision_patch_embedding(&self) -> Result<Glm53VisionPatchEmbedWeights, String> {
        let vision = &self.dims.vision;
        let patch = vision.patch_size;
        let cols = 3usize.checked_mul(vision.temporal_patch_size).and_then(|value| value.checked_mul(patch)).and_then(|value| value.checked_mul(patch)).ok_or_else(|| "GLM-5.3-Flash 视觉 patch embed 列数溢出".to_owned())?;
        let tensor = self.load_bf16_raw(VISION_PATCH_EMBED)?;
        let expected = [vision.hidden_size, 3, vision.temporal_patch_size, patch, patch];
        if tensor.shape != expected {
            return Err(format!("{VISION_PATCH_EMBED} shape={:?}，期望 {expected:?}", tensor.shape));
        }
        let projection = TensorData { name: tensor.name, dtype: tensor.dtype, shape: vec![vision.hidden_size, cols], data: tensor.data };
        Ok(Glm53VisionPatchEmbedWeights { projection, bias: self.load_bf16(&format!("{VISION_PATCH_EMBED}.bias"), 1, vision.hidden_size)? })
    }

    pub fn load_vision_layer(&self, layer: usize) -> Result<Glm53VisionBlockWeights, String> {
        if layer >= self.dims.vision.depth {
            return Err(format!("GLM-5.3-Flash 视觉层 {layer} 越界于 {}", self.dims.vision.depth));
        }
        let vision = &self.dims.vision;
        let hidden = vision.hidden_size;
        let head_dim = hidden / vision.num_heads;
        let names = Glm53VisionLayerNames::new(layer).block();
        let attention = Glm53VisionAttentionWeights {
            query_key_value: self.load_bf16(&names.attention.query_key_value, 3 * hidden, hidden)?,
            query_key_value_bias: self.load_bf16(&names.attention.query_key_value_bias, 1, 3 * hidden)?,
            query_norm: self.load_bf16(&names.attention.query_norm, 1, head_dim)?,
            key_norm: self.load_bf16(&names.attention.key_norm, 1, head_dim)?,
            output: self.load_bf16(&names.attention.output, hidden, hidden)?,
            output_bias: self.load_bf16(&names.attention.output_bias, 1, hidden)?,
        };
        let mlp = Glm53VisionMlpWeights {
            gate: self.load_bf16(&names.mlp.gate, vision.intermediate_size, hidden)?,
            gate_bias: self.load_bf16(&names.mlp.gate_bias, 1, vision.intermediate_size)?,
            up: self.load_bf16(&names.mlp.up, vision.intermediate_size, hidden)?,
            up_bias: self.load_bf16(&names.mlp.up_bias, 1, vision.intermediate_size)?,
            down: self.load_bf16(&names.mlp.down, hidden, vision.intermediate_size)?,
            down_bias: self.load_bf16(&names.mlp.down_bias, 1, hidden)?,
        };
        Ok(Glm53VisionBlockWeights { input_norm: self.load_bf16(&names.input_norm, 1, hidden)?, post_attention_norm: self.load_bf16(&names.post_attention_norm, 1, hidden)?, attention, mlp })
    }

    /// 视觉塔尾部:post RMSNorm、downsample(权重置换为线性布局)与 merger。
    pub fn load_vision_tail(&self) -> Result<Glm53VisionTailWeights, String> {
        let vision = &self.dims.vision;
        let hidden = vision.hidden_size;
        let out = vision.out_hidden_size;
        let merge = vision.spatial_merge_size;
        let downsample = self.load_bf16_raw(VISION_DOWNSAMPLE)?;
        let expected = [out, hidden, merge, merge];
        if downsample.shape != expected {
            return Err(format!("{VISION_DOWNSAMPLE} shape={:?}，期望 {expected:?}", downsample.shape));
        }
        let downsample = permute_conv_window_merge(&downsample, out, hidden, merge)?;
        Ok(Glm53VisionTailWeights {
            post_layernorm: self.load_bf16(VISION_POST_LAYERNORM, 1, hidden)?,
            downsample,
            downsample_bias: self.load_bf16(&format!("{VISION_DOWNSAMPLE}.bias"), 1, out)?,
            merger_projection: self.load_bf16(VISION_MERGER_PROJ, out, out)?,
            merger_norm_weight: self.load_bf16(&format!("{VISION_MERGER_NORM}.weight"), 1, out)?,
            merger_norm_bias: self.load_bf16(&format!("{VISION_MERGER_NORM}.bias"), 1, out)?,
            merger_gate: self.load_bf16("model.visual.merger.gate_proj.weight", vision.projection_intermediate_size, out)?,
            merger_up: self.load_bf16("model.visual.merger.up_proj.weight", vision.projection_intermediate_size, out)?,
            merger_down: self.load_bf16("model.visual.merger.down_proj.weight", out, vision.projection_intermediate_size)?,
        })
    }

    /// 仅校验 dtype 与总体积,shape 留给高维权重(conv)的调用方判定。
    fn load_bf16_raw(&self, name: &str) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        if tensor.dtype != "BF16" {
            return Err(format!("{name} dtype={}，期望 BF16", tensor.dtype));
        }
        Ok(tensor)
    }

    fn load_dense_mlp_names(&self, names: &Glm53DenseMlpNames, intermediate: usize) -> Result<Glm53DenseMlpWeights, String> {
        Ok(Glm53DenseMlpWeights {
            gate: self.load_fp8(&names.gate, intermediate, self.dims.hidden_size)?,
            up: self.load_fp8(&names.up, intermediate, self.dims.hidden_size)?,
            down: self.load_fp8(&names.down, self.dims.hidden_size, intermediate)?,
        })
    }

    /// 官方块级 FP8:F8_E4M3 codes + F32 `weight_scale_inv`(128×128 块)。
    fn load_fp8(&self, name: &str, rows: usize, cols: usize) -> Result<Fp8Matrix, String> {
        let tensor = self.store.load(name)?;
        if tensor.dtype != "F8_E4M3" || tensor.shape.as_slice() != [rows, cols] {
            return Err(format!("{name} dtype={} shape={:?}，期望 F8_E4M3/[{rows},{cols}]", tensor.dtype, tensor.shape));
        }
        let scale_name = format!("{name}_scale_inv");
        let scale = self.store.load(&scale_name)?;
        let expected_scale = [rows.div_ceil(128), cols.div_ceil(128)];
        if scale.dtype != "F32" || scale.shape.as_slice() != expected_scale {
            return Err(format!("{scale_name} dtype={} shape={:?}，期望 F32/{expected_scale:?}", scale.dtype, scale.shape));
        }
        Fp8Matrix::new(tensor.data, scale.data, rows, cols).map_err(|error| format!("{name}: {error}"))
    }

    fn load_bf16(&self, name: &str, rows: usize, cols: usize) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        if tensor.dtype != "BF16" || tensor.shape.as_slice() != [rows, cols] {
            return Err(format!("{name} dtype={} shape={:?}，期望 BF16/[{rows},{cols}]", tensor.dtype, tensor.shape));
        }
        Ok(tensor)
    }

    /// 一维 norm 向量(checkpoint 实测 shape 为 `(n,)` 而非 `[1,n]`)。
    fn load_bf16_vector(&self, name: &str, len: usize) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        if tensor.dtype != "BF16" || tensor.shape.as_slice() != [len] {
            return Err(format!("{name} dtype={} shape={:?}，期望 BF16/({len},)", tensor.dtype, tensor.shape));
        }
        Ok(tensor)
    }

    /// PyTorch Conv1d 权重 `[out, 1, kernel]`;按元素数校验,prepare 展平为 [out, kernel]。
    fn load_bf16_conv1d(&self, name: &str, out: usize, kernel: usize) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        let expected = [out, 1, kernel];
        if tensor.dtype != "BF16" || tensor.shape.as_slice() != expected {
            return Err(format!("{name} dtype={} shape={:?}，期望 BF16/{expected:?}", tensor.dtype, tensor.shape));
        }
        Ok(tensor)
    }

    /// 一维 F32 向量(mhc base/scale 实测 shape 为 `(n,)` 而非 `[n,1]`)。
    fn load_f32_vector(&self, name: &str, len: usize, label: &str) -> Result<Vec<f32>, String> {
        let tensor = self.store.load(name)?;
        if tensor.dtype != "F32" || tensor.shape.as_slice() != [len] {
            return Err(format!("{name} dtype={} shape={:?}，期望 F32/({len},)({label})", tensor.dtype, tensor.shape));
        }
        crate::weight::container::safetensor::decode_to_f32(name, &tensor.dtype, &tensor.data)
    }

    /// 小状态参数(a_log/dt_bias):行数未知维度由调用方校验,仅要求一维。
    fn load_small_f32(&self, name: &str, label: &str) -> Result<Vec<f32>, String> {
        let tensor = self.store.load(name)?;
        if tensor.dtype != "F32" || tensor.shape.len() != 1 {
            return Err(format!("{name} dtype={} shape={:?}，期望 F32 一维({label})", tensor.dtype, tensor.shape));
        }
        crate::weight::container::safetensor::decode_to_f32(name, &tensor.dtype, &tensor.data)
    }

    /// 路由权重允许 BF16 或 F32(官方 moe_router_dtype=float32,以 checkpoint 实际为准)。
    fn decode_f32(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>, String> {
        let tensor = self.store.load(name)?;
        if tensor.shape.as_slice() != [rows, cols] {
            return Err(format!("{name} dtype={} shape={:?}，期望 [{rows},{cols}]", tensor.dtype, tensor.shape));
        }
        crate::weight::container::safetensor::decode_to_f32(name, &tensor.dtype, &tensor.data)
    }
}

/// conv2d 窗口权重 [out, hidden, merge, merge] 置换为线性 [out, merge²*hidden]。
/// `merge_spatial` 拼出的行向量第 (u*merge+v)*hidden + c 个元素是 patch (u,v) 的
/// 第 c 通道,而 conv 权重按 (c,u,v) 展开;这里只挪 BF16 元素位置,不改数值。
fn permute_conv_window_merge(tensor: &TensorData, out: usize, hidden: usize, merge: usize) -> Result<TensorData, String> {
    let window = merge.checked_mul(merge).ok_or_else(|| format!("{} merge 溢出", tensor.name))?;
    let elements = out.checked_mul(hidden).and_then(|value| value.checked_mul(window)).ok_or_else(|| format!("{} 元素数量溢出", tensor.name))?;
    if tensor.data.len() != elements * 2 {
        return Err(format!("{} 字节数 {}，期望 {}", tensor.name, tensor.data.len(), elements * 2));
    }
    let mut data = vec![0u8; elements * 2];
    for output_row in 0..out {
        for input_channel in 0..hidden {
            for window_u in 0..merge {
                for window_v in 0..merge {
                    let source = ((output_row * hidden + input_channel) * merge + window_u) * merge + window_v;
                    let destination = output_row * hidden * window + (window_u * merge + window_v) * hidden + input_channel;
                    data[destination * 2..destination * 2 + 2].copy_from_slice(&tensor.data[source * 2..source * 2 + 2]);
                }
            }
        }
    }
    Ok(TensorData { name: tensor.name.clone(), dtype: tensor.dtype.clone(), shape: vec![out, hidden * window], data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample置换匹配conv窗口语义() {
        // out=1, hidden=2, merge=2:conv 权重按 (c,u,v) 展平为 0..8,
        // 等价线性权重应为 [0,4,1,5,2,6,3,7]。
        let values = (0..8u16).flat_map(|value| half::bf16::from_f32(f32::from(value)).to_le_bytes()).collect();
        let tensor = TensorData { name: "downsample".to_owned(), dtype: "BF16".to_owned(), shape: vec![1, 2, 2, 2], data: values };
        let permuted = permute_conv_window_merge(&tensor, 1, 2, 2).unwrap();
        assert_eq!(permuted.shape, vec![1, 8]);
        let decoded = permuted.to_f32().unwrap();
        assert_eq!(decoded, vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
    }

    #[test]
    fn 视觉层命名与官方一致() {
        let block = Glm53VisionLayerNames::new(3).block();
        assert_eq!(block.input_norm, "model.visual.blocks.3.norm1.weight");
        assert_eq!(block.attention.query_key_value, "model.visual.blocks.3.attn.qkv.weight");
        assert_eq!(block.attention.key_norm, "model.visual.blocks.3.attn.k_norm.weight");
        assert_eq!(block.mlp.gate_bias, "model.visual.blocks.3.mlp.gate_proj.bias");
        assert_eq!(block.mlp.down, "model.visual.blocks.3.mlp.down_proj.weight");
    }

    #[test]
    fn mtp归一化权重按输入角色装配() {
        let mtp = Glm53FlashLayerNames::new(45).mtp();
        assert_eq!(mtp.embedding_norm, "model.language_model.layers.45.enorm.weight");
        assert_eq!(mtp.hidden_norm, "model.language_model.layers.45.hnorm.weight");
        assert_eq!(mtp.shared_head_norm, "model.language_model.layers.45.shared_head.norm.weight");
    }
}
