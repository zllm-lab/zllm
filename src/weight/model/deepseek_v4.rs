//! DeepSeek-V4 官方 Safetensors 权重装配。
//!
//! Core FP8、routed MXFP4 与 dense BF16/F32 保持各自格式；本模块只负责名称、
//! shape 和生命周期，不包含模型执行或设备上传。

use std::{path::Path, sync::Arc};

use crate::{
    model_spec::deepseek_v4::{DeepSeekV4Config, DeepSeekV4RoutingSelection},
    tokenizer::{Detokenizer, Tokenizer},
    weight::{
        container::gguf::{GgmlType, GgufMatrix, GgufReader},
        container::safetensor::{SafetensorStore, TensorData},
        expert_source::{ExpertSource, ExpertSourceProvider, Mxfp4ExpertSource, Mxfp4ExpertWeights},
        expert_source::{GgufExpertSource, GgufExpertWeights},
        format::block_fp8::BlockFp8Matrix,
        format::mxfp4::load_mxfp4_matrix,
        format::mxfp8::Mxfp8Matrix,
    },
};

const FP8_BLOCK: usize = 128;

/// 核心线性矩阵的三种落地形态:官方 FP8(block 128)、Jundot/omlx MXFP8(block 32)、
/// 未量化 dense(omlx 对 wo_a 等模块保留 BF16)。
pub enum DeepSeekV4CoreMatrix {
    BlockFp8(BlockFp8Matrix),
    Mxfp8(Mxfp8Matrix),
    Dense(TensorData),
}

impl DeepSeekV4CoreMatrix {
    pub fn rows(&self) -> usize {
        match self {
            Self::BlockFp8(matrix) => matrix.rows,
            Self::Mxfp8(matrix) => matrix.rows,
            Self::Dense(tensor) => tensor.shape[0],
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::BlockFp8(matrix) => matrix.cols,
            Self::Mxfp8(matrix) => matrix.cols,
            Self::Dense(tensor) => tensor.shape[1],
        }
    }
}

/// V4.1 engram 层的小权重(k/q 投影与 wkv);embed 大表按需读取。
pub struct DeepSeekV4EngramWeights {
    pub k: TensorData,
    pub q: TensorData,
    pub wkv: DeepSeekV4CoreMatrix,
}

/// engram embed 行的两种量化形态。
pub enum DeepSeekV4EngramEmbedding {
    Mxfp8(Mxfp8Matrix),
    MlxAffine(crate::weight::format::quantization::MlxAffineMatrix),
}
const EMBEDDING: &str = "embed.weight";
const FINAL_NORM: &str = "norm.weight";
const LM_HEAD: &str = "head.weight";

/// 核心张量实际出现的 dtype；未知 dtype 显式报错而不是猜字节宽。
fn dtype_bytes(dtype: &str, name: &str) -> Result<usize, String> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" | "I32" | "U32" => Ok(4),
        "I64" => Ok(8),
        "F8_E4M3" | "F8_E8M0" | "U8" => Ok(1),
        _ => Err(format!("DeepSeek-V4 tensor {name} dtype={dtype} 字节宽未知")),
    }
}

#[derive(Clone, Debug)]
pub struct DeepSeekV4LayerNames {
    prefix: String,
}

impl DeepSeekV4LayerNames {
    /// `ns` 为空串(官方)或 "language_model"(omlx/Jundot 系 checkpoint)。
    pub fn new(layer: usize, ns: &str) -> Self {
        Self { prefix: format!("{ns}layers.{layer}") }
    }

    fn mtp(layer: usize, ns: &str) -> Self {
        Self { prefix: format!("{ns}mtp.{layer}") }
    }

    fn tensor(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.prefix)
    }

    fn attention(&self, suffix: &str) -> String {
        self.tensor(&format!("attn.{suffix}"))
    }

    fn feedforward(&self, suffix: &str) -> String {
        self.tensor(&format!("ffn.{suffix}"))
    }
}

pub struct DeepSeekV4HyperConnectionWeights {
    pub function: TensorData,
    pub base: TensorData,
    pub scale: TensorData,
}

pub struct DeepSeekV4QueryWeights {
    pub input_projection: DeepSeekV4CoreMatrix,
    pub norm: TensorData,
    pub output_projection: DeepSeekV4CoreMatrix,
}

pub struct DeepSeekV4KeyValueWeights {
    pub projection: DeepSeekV4CoreMatrix,
    pub norm: TensorData,
}

pub struct DeepSeekV4OutputWeights {
    pub input_projection: DeepSeekV4CoreMatrix,
    pub output_projection: DeepSeekV4CoreMatrix,
}

pub struct DeepSeekV4CompressorWeights {
    /// V4 的绝对位置嵌入;V4.1 compressor 无 ape(checkpoint 无此键)。
    pub position: Option<TensorData>,
    pub key_value_projection: TensorData,
    /// ratio>1 的 softmax 池化门;ratio=1 纯投影无门(checkpoint 无 wgate 键)。
    pub gate_projection: Option<TensorData>,
    pub norm: TensorData,
}

pub struct DeepSeekV4IndexerWeights {
    pub query_projection: DeepSeekV4CoreMatrix,
    pub head_weights_projection: TensorData,
    /// V4.1 跨层共享 indexer:K 投影与 norm 只落在组首层,共享层为 None。
    pub key_projection: Option<TensorData>,
    pub key_norm: Option<TensorData>,
    /// V4 的 indexer compressor;V4.1 indexer 无 compressor。
    pub compressor: Option<DeepSeekV4CompressorWeights>,
}

pub struct DeepSeekV4AttentionWeights {
    pub sink: TensorData,
    pub query: DeepSeekV4QueryWeights,
    pub key_value: DeepSeekV4KeyValueWeights,
    pub output: DeepSeekV4OutputWeights,
    pub compressor: Option<DeepSeekV4CompressorWeights>,
    pub indexer: Option<DeepSeekV4IndexerWeights>,
}

pub enum DeepSeekV4RouterWeights {
    TokenHash { weight: TensorData, token_to_experts: TensorData },
    ScoreTopK { weight: TensorData, correction_bias: TensorData, value_level_bias: Option<TensorData> },
}

pub struct DeepSeekV4SharedExpertWeights {
    pub gate: DeepSeekV4CoreMatrix,
    pub up: DeepSeekV4CoreMatrix,
    pub down: DeepSeekV4CoreMatrix,
}

pub struct DeepSeekV4AttentionBlockWeights {
    pub hyper_connection: DeepSeekV4HyperConnectionWeights,
    pub norm: TensorData,
    pub attention: DeepSeekV4AttentionWeights,
}

pub struct DeepSeekV4FeedforwardBlockWeights {
    pub hyper_connection: DeepSeekV4HyperConnectionWeights,
    pub norm: TensorData,
    pub router: DeepSeekV4RouterWeights,
    pub shared_expert: DeepSeekV4SharedExpertWeights,
}

pub struct DeepSeekV4LayerWeights {
    pub attention: DeepSeekV4AttentionBlockWeights,
    pub feedforward: DeepSeekV4FeedforwardBlockWeights,
}

pub struct DeepSeekV4HeadWeights {
    /// V4.1 head 无 hyper-connection 系数(checkpoint 无 hc_head_* 键),V4 必有。
    pub hyper_connection: Option<DeepSeekV4HyperConnectionWeights>,
    pub norm: TensorData,
    pub lm_head: TensorData,
}

/// DeepSeek-V4.1 视觉塔一层(BF16 dense;attention 带 bias,MLP 无 bias)。
/// `mlp_gate_up` 是官方 w1 的 [2*inter, dim] 融合矩阵,执行侧按列拆 gate/up。
pub struct DeepseekV41VisionBlockWeights {
    pub input_norm: TensorData,
    pub post_attention_norm: TensorData,
    pub query_key_value: TensorData,
    pub query_key_value_bias: TensorData,
    pub output: TensorData,
    pub output_bias: TensorData,
    pub mlp_gate_up: TensorData,
    pub mlp_down: TensorData,
}

pub struct DeepseekV41VisionPatchEmbedWeights {
    pub projection: TensorData,
    pub bias: TensorData,
}

/// aligner:3×3 空间合并后的 [r²*vision_dim, hidden] 双线性投影。
pub struct DeepseekV41AlignerWeights {
    pub input: TensorData,
    pub input_bias: TensorData,
    pub output: TensorData,
    pub output_bias: TensorData,
}

/// image span 首尾/换行位置的学习 embedding(与 LLM hidden 同宽)。
pub struct DeepseekV41ImageSpanEmbeddings {
    pub start: TensorData,
    pub newline: TensorData,
    pub end: TensorData,
}

#[derive(Clone)]
pub struct DeepSeekV4Weights {
    store: Arc<SafetensorStore>,
    config: DeepSeekV4Config,
    /// 顶层命名空间:""(官方)或 "language_model."(omlx/Jundot 系)。
    ns: &'static str,
}

/// 权重装配路径会做 `num_heads * head_dim / output_groups` 分组除法;
/// 在 open 前置校验,避免非法 Config 触发整数除零 panic
/// (CompressedSparseAttentionSpec::validate 只在 runtime 侧才调用,晚于装配)。
fn validate_head_groups(config: &DeepSeekV4Config) -> Result<(), String> {
    if config.output_groups == 0 || config.num_heads % config.output_groups != 0 {
        return Err(format!("DeepSeek-V4 output_groups={} 与 num_heads={} 不匹配", config.output_groups, config.num_heads));
    }
    Ok(())
}

impl DeepSeekV4Weights {
    pub fn open(root: impl AsRef<Path>, config: DeepSeekV4Config) -> Result<Self, String> {
        validate_head_groups(&config)?;
        if config.layer_count == 0 || config.expert_count == 0 {
            return Err("DeepSeek-V4 权重配置的层数和专家数必须非零".to_owned());
        }
        let store = Arc::new(SafetensorStore::open(root.as_ref())?);
        let ns = if store.has("language_model.embed.weight") { "language_model." } else { "" };
        Ok(Self { store, config, ns })
    }

    pub fn config(&self) -> &DeepSeekV4Config {
        &self.config
    }

    fn top(&self, suffix: &str) -> String {
        format!("{}{suffix}", self.ns)
    }

    /// 带命名空间的任意权重名(dspark checkpoint 等外部装配使用)。
    pub(crate) fn namespaced(&self, suffix: &str) -> String {
        format!("{}{}", self.ns, suffix)
    }
    /// 张量存在性检查(经命名空间);V4.1 按存在性区分 V4/V4.1 张量布局。
    pub fn has_namespaced(&self, suffix: &str) -> bool {
        self.store.has(&self.namespaced(suffix))
    }

    pub fn embedding_rows_bf16(&self, token_ids: &[u32]) -> Result<Vec<u8>, String> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows = token_ids
            .iter()
            .map(|&token| {
                let token = token as usize;
                (token < self.config.vocab_size).then_some(token).ok_or_else(|| format!("DeepSeek-V4 token {token} 越界于 vocab {}", self.config.vocab_size))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tensor = self.store.load_bf16_rows(&self.top(EMBEDDING), &rows)?;
        tensor.expect_shape(&[rows.len(), self.config.hidden_size])?;
        Ok(tensor.data)
    }

    pub fn final_norm(&self) -> Result<TensorData, String> {
        self.load_dense(&self.top(FINAL_NORM), &[self.config.hidden_size])
    }

    /// LM head 很大，只在 backend 准备输出投影时显式读取。
    pub fn lm_head(&self) -> Result<TensorData, String> {
        self.load_dense(&self.top(LM_HEAD), &[self.config.vocab_size, self.config.hidden_size])
    }

    pub fn output_head(&self) -> Result<DeepSeekV4HeadWeights, String> {
        let copies = self.config.hyper_connection_copies;
        let hidden = copies.checked_mul(self.config.hidden_size).ok_or("DeepSeek-V4 head hidden 维度溢出")?;
        let hyper_connection = if self.store.has(&self.top("hc_head_fn")) {
            Some(DeepSeekV4HyperConnectionWeights { function: self.load_f32(&self.top("hc_head_fn"), &[copies, hidden])?, base: self.load_f32(&self.top("hc_head_base"), &[copies])?, scale: self.load_f32(&self.top("hc_head_scale"), &[1])? })
        } else {
            None
        };
        Ok(DeepSeekV4HeadWeights { hyper_connection, norm: self.final_norm()?, lm_head: self.lm_head()? })
    }

    /// V4.1 checkpoint 以顶层 `vision.patch_embed.proj.weight` 标记视觉塔;
    /// 视觉张量不属于 language_model 命名空间。
    pub fn has_vision_tower(&self) -> bool {
        self.store.has("vision.patch_embed.proj.weight")
    }

    pub fn load_vision_patch_embed(&self, vision: &crate::model_spec::deepseek_v4::DeepseekV41VisionConfig) -> Result<DeepseekV41VisionPatchEmbedWeights, String> {
        let patch_columns = 3usize.checked_mul(vision.patch_size).and_then(|value| value.checked_mul(vision.patch_size)).ok_or("DeepSeek-V4.1 视觉 patch 列数溢出")?;
        Ok(DeepseekV41VisionPatchEmbedWeights { projection: self.load_dense("vision.patch_embed.proj.weight", &[vision.hidden_size, patch_columns])?, bias: self.load_dense("vision.patch_embed.proj.bias", &[vision.hidden_size])? })
    }

    pub fn load_vision_block(&self, layer: usize, vision: &crate::model_spec::deepseek_v4::DeepseekV41VisionConfig) -> Result<DeepseekV41VisionBlockWeights, String> {
        if layer >= vision.layer_count {
            return Err(format!("DeepSeek-V4.1 视觉层 {layer} 越界,layer_count={}", vision.layer_count));
        }
        let prefix = format!("vision.blocks.{layer}");
        let dim = vision.hidden_size;
        Ok(DeepseekV41VisionBlockWeights {
            input_norm: self.load_dense(&format!("{prefix}.norm1.weight"), &[dim])?,
            post_attention_norm: self.load_dense(&format!("{prefix}.norm2.weight"), &[dim])?,
            query_key_value: self.load_dense(&format!("{prefix}.attn.wqkv.weight"), &[3 * dim, dim])?,
            query_key_value_bias: self.load_dense(&format!("{prefix}.attn.wqkv.bias"), &[3 * dim])?,
            output: self.load_dense(&format!("{prefix}.attn.wo.weight"), &[dim, dim])?,
            output_bias: self.load_dense(&format!("{prefix}.attn.wo.bias"), &[dim])?,
            mlp_gate_up: self.load_dense(&format!("{prefix}.mlp.w1.weight"), &[2 * vision.intermediate_size, dim])?,
            mlp_down: self.load_dense(&format!("{prefix}.mlp.w2.weight"), &[dim, vision.intermediate_size])?,
        })
    }

    pub fn load_vision_final_norm(&self, vision: &crate::model_spec::deepseek_v4::DeepseekV41VisionConfig) -> Result<TensorData, String> {
        self.load_dense("vision.norm.weight", &[vision.hidden_size])
    }

    pub fn load_aligner(&self, vision: &crate::model_spec::deepseek_v4::DeepseekV41VisionConfig) -> Result<DeepseekV41AlignerWeights, String> {
        let input_columns = vision.downsample_ratio.checked_mul(vision.downsample_ratio).and_then(|merge| vision.hidden_size.checked_mul(merge)).ok_or("DeepSeek-V4.1 aligner 输入宽度溢出")?;
        let hidden = self.config.hidden_size;
        Ok(DeepseekV41AlignerWeights {
            input: self.load_dense("aligner.w1.weight", &[hidden, input_columns])?,
            input_bias: self.load_dense("aligner.w1.bias", &[hidden])?,
            output: self.load_dense("aligner.w2.weight", &[hidden, hidden])?,
            output_bias: self.load_dense("aligner.w2.bias", &[hidden])?,
        })
    }

    pub fn load_image_span_embeddings(&self) -> Result<DeepseekV41ImageSpanEmbeddings, String> {
        let hidden = self.config.hidden_size;
        Ok(DeepseekV41ImageSpanEmbeddings { start: self.load_dense("image_start", &[hidden])?, newline: self.load_dense("image_newline", &[hidden])?, end: self.load_dense("image_end", &[hidden])? })
    }

    pub fn load_layer(&self, layer: usize) -> Result<DeepSeekV4LayerWeights, String> {
        if layer >= self.config.layer_count {
            return Err(format!("DeepSeek-V4 layer {layer} 越界，layer_count={}", self.config.layer_count));
        }
        let names = DeepSeekV4LayerNames::new(layer, self.ns);
        let spec = self.config.compress_ratios[layer];
        let routing = if layer < self.config.hash_layer_count { DeepSeekV4RoutingSelection::TokenHash } else { DeepSeekV4RoutingSelection::ScoreTopK };
        self.load_named_layer(&names, spec, routing, self.config.expert_count)
    }

    pub fn load_mtp_layer(&self, layer: usize) -> Result<DeepSeekV4LayerWeights, String> {
        if layer >= self.config.mtp_layer_count {
            return Err(format!("DeepSeek-V4 MTP layer {layer} 越界，mtp_layer_count={}", self.config.mtp_layer_count));
        }
        let names = DeepSeekV4LayerNames::mtp(layer, self.ns);
        let spec = self.config.compress_ratios[self.config.layer_count + layer];
        self.load_named_layer(&names, spec, DeepSeekV4RoutingSelection::ScoreTopK, self.config.mtp_expert_count)
    }

    fn load_named_layer(&self, names: &DeepSeekV4LayerNames, spec: usize, routing: DeepSeekV4RoutingSelection, expert_count: usize) -> Result<DeepSeekV4LayerWeights, String> {
        let attention = DeepSeekV4AttentionBlockWeights {
            hyper_connection: self.load_hyper_connection(&names, "attn")?,
            norm: self.load_dense(&names.tensor("attn_norm.weight"), &[self.config.hidden_size])?,
            attention: DeepSeekV4AttentionWeights {
                sink: self.load_f32(&names.attention("attn_sink"), &[self.config.num_heads])?,
                query: DeepSeekV4QueryWeights {
                    input_projection: self.load_core_matrix(&names.attention("wq_a"), self.config.q_lora_rank, self.config.hidden_size)?,
                    norm: self.load_dense(&names.attention("q_norm.weight"), &[self.config.q_lora_rank])?,
                    output_projection: self.load_core_matrix(&names.attention("wq_b"), self.config.num_heads * self.config.head_dim, self.config.q_lora_rank)?,
                },
                key_value: DeepSeekV4KeyValueWeights {
                    projection: self.load_core_matrix(&names.attention("wkv"), self.config.head_dim, self.config.hidden_size)?,
                    norm: self.load_dense(&names.attention("kv_norm.weight"), &[self.config.head_dim])?,
                },
                output: DeepSeekV4OutputWeights {
                    input_projection: self.load_core_matrix(&names.attention("wo_a"), self.config.output_groups * self.config.output_lora_rank, self.config.num_heads * self.config.head_dim / self.config.output_groups)?,
                    output_projection: self.load_core_matrix(&names.attention("wo_b"), self.config.hidden_size, self.config.output_groups * self.config.output_lora_rank)?,
                },
                // V4:ratio!=0 必有 compressor 张量;V4.1:compressor/indexer 跨层共享,
                // 只落在组首层,按存在性装配(共享层的执行侧再借组首)。
                compressor: (spec != 0 && self.store.has(&names.tensor("attn.compressor.wkv.weight"))).then(|| self.load_compressor(&names, "attn.compressor", spec, self.config.head_dim)).transpose()?,
                // V4:ratio==4 层有 indexer;V4.1:ratio 1/2 的 index_source 层均有(探测张量)
                indexer: ((spec == 4 || spec == 2 || spec == 1) && self.store.has(&names.attention("indexer.wq_b.weight"))).then(|| self.load_indexer(&names, spec)).transpose()?,
            },
        };
        let router_weight = self.load_dense(&names.feedforward("gate.weight"), &[expert_count, self.config.hidden_size])?;
        let router = match routing {
            DeepSeekV4RoutingSelection::TokenHash => DeepSeekV4RouterWeights::TokenHash { weight: router_weight, token_to_experts: self.load_i32(&names.feedforward("gate.tid2eid"), &[self.config.vocab_size, self.config.expert_top_k])? },
            DeepSeekV4RoutingSelection::ScoreTopK => DeepSeekV4RouterWeights::ScoreTopK {
                weight: router_weight,
                correction_bias: self.load_f32(&names.feedforward("gate.bias"), &[expert_count])?,
                value_level_bias: self.config.router_value_level_bias.then(|| self.load_f32(&names.feedforward("gate.bias_vl"), &[expert_count])).transpose()?,
            },
        };
        Ok(DeepSeekV4LayerWeights {
            attention,
            feedforward: DeepSeekV4FeedforwardBlockWeights {
                hyper_connection: self.load_hyper_connection(&names, "ffn")?,
                norm: self.load_dense(&names.tensor("ffn_norm.weight"), &[self.config.hidden_size])?,
                router,
                shared_expert: DeepSeekV4SharedExpertWeights {
                    gate: self.load_core_matrix(&names.feedforward("shared_experts.w1"), self.config.expert_intermediate_size, self.config.hidden_size)?,
                    up: self.load_core_matrix(&names.feedforward("shared_experts.w3"), self.config.expert_intermediate_size, self.config.hidden_size)?,
                    down: self.load_core_matrix(&names.feedforward("shared_experts.w2"), self.config.hidden_size, self.config.expert_intermediate_size)?,
                },
            },
        })
    }

    pub fn expert_source(&self) -> DeepSeekV4ExpertSource {
        DeepSeekV4ExpertSource { weights: self.clone(), prefix: format!("{}layers", self.ns), layer_count: self.config.layer_count, expert_count: self.config.expert_count }
    }

    pub fn mtp_expert_source(&self) -> DeepSeekV4ExpertSource {
        DeepSeekV4ExpertSource { weights: self.clone(), prefix: format!("{}mtp", self.ns), layer_count: self.config.mtp_layer_count, expert_count: self.config.mtp_expert_count }
    }

    /// 估算一层非 routed-expert 权重的存储字节数，供 runtime 规划固定常驻层；
    /// 与 GGUF 版 `DeepSeekV4Gguf::layer_storage_bytes` 语义一致。
    pub fn layer_storage_bytes(&self, layer: usize) -> Result<usize, String> {
        if layer >= self.config.layer_count {
            return Err(format!("DeepSeek-V4 layer {layer} 越界，layer_count={}", self.config.layer_count));
        }
        let prefix = format!("{}layers.{layer}.", self.ns);
        self.store.tensor_names().iter().filter(|name| name.starts_with(&prefix) && !name.contains("ffn.experts.")).try_fold(0usize, |bytes, name| {
            let info = self.store.tensor_info(name)?;
            let elements = info.shape.iter().try_fold(1usize, |value, &dimension| value.checked_mul(dimension)).ok_or_else(|| format!("DeepSeek-V4 tensor {name} 元素数溢出"))?;
            bytes.checked_add(elements.checked_mul(dtype_bytes(&info.dtype, name)?).ok_or_else(|| format!("DeepSeek-V4 tensor {name} 字节数溢出"))?).ok_or_else(|| format!("DeepSeek-V4 layer {layer} 字节数溢出"))
        })
    }

    fn load_hyper_connection(&self, names: &DeepSeekV4LayerNames, sublayer: &str) -> Result<DeepSeekV4HyperConnectionWeights, String> {
        let copies = self.config.hyper_connection_copies;
        let mixes = (2 + copies).checked_mul(copies).ok_or("DeepSeek-V4 mHC mix 维度溢出")?;
        let hidden = copies.checked_mul(self.config.hidden_size).ok_or("DeepSeek-V4 mHC hidden 维度溢出")?;
        Ok(DeepSeekV4HyperConnectionWeights {
            function: self.load_f32(&names.tensor(&format!("hc_{sublayer}_fn")), &[mixes, hidden])?,
            base: self.load_f32(&names.tensor(&format!("hc_{sublayer}_base")), &[mixes])?,
            scale: self.load_f32(&names.tensor(&format!("hc_{sublayer}_scale")), &[3])?,
        })
    }

    fn load_compressor(&self, names: &DeepSeekV4LayerNames, suffix: &str, ratio: usize, head_dim: usize) -> Result<DeepSeekV4CompressorWeights, String> {
        let channels = if ratio == 4 { 2 * head_dim } else { head_dim };
        let ape_name = names.tensor(&format!("{suffix}.ape"));
        let gate_name = names.tensor(&format!("{suffix}.wgate.weight"));
        Ok(DeepSeekV4CompressorWeights {
            position: self.store.has(&ape_name).then(|| self.load_f32(&ape_name, &[ratio, channels])).transpose()?,
            key_value_projection: self.load_dense(&names.tensor(&format!("{suffix}.wkv.weight")), &[channels, self.config.hidden_size])?,
            gate_projection: (ratio > 1 && self.store.has(&gate_name)).then(|| self.load_dense(&gate_name, &[channels, self.config.hidden_size])).transpose()?,
            norm: self.load_dense(&names.tensor(&format!("{suffix}.norm.weight")), &[head_dim])?,
        })
    }

    fn load_indexer(&self, names: &DeepSeekV4LayerNames, ratio: usize) -> Result<DeepSeekV4IndexerWeights, String> {
        let key_name = names.attention("indexer.wk.weight");
        let key_norm_name = names.attention("indexer.k_norm.weight");
        let compressor_prefix = names.attention("indexer.compressor.wkv.weight");
        Ok(DeepSeekV4IndexerWeights {
            query_projection: self.load_core_matrix(&names.attention("indexer.wq_b"), self.config.index_heads * self.config.index_head_dim, self.config.q_lora_rank)?,
            head_weights_projection: self.load_dense(&names.attention("indexer.weights_proj.weight"), &[self.config.index_heads, self.config.hidden_size])?,
            key_projection: self.store.has(&key_name).then(|| self.load_dense(&key_name, &[self.config.index_head_dim, self.config.head_dim])).transpose()?,
            key_norm: self.store.has(&key_norm_name).then(|| self.load_dense(&key_norm_name, &[self.config.index_head_dim])).transpose()?,
            compressor: self.store.has(&compressor_prefix).then(|| self.load_compressor(names, "attn.indexer.compressor", ratio, self.config.index_head_dim)).transpose()?,
        })
    }

    /// V4.1 engram 层权重(384M 行 embed 大表不预装,经 `engram_embedding_rows` 按需读取)。
    pub fn load_engram(&self, layer: usize) -> Result<Option<DeepSeekV4EngramWeights>, String> {
        let Some(engram) = &self.config.engram else { return Ok(None) };
        if !engram.layer_ids.contains(&layer) {
            return Ok(None);
        }
        let names = DeepSeekV4LayerNames::new(layer, self.ns);
        let wkv_prefix = names.tensor("engram.wkv");
        let info = self.store.tensor_info(&format!("{wkv_prefix}.weight"))?;
        // Jundot mxfp8 存储形状为 U32 packed(逻辑列 = 存储列×4);官方 F8_E4M3/dense 即逻辑形状
        let rows = info.shape[0];
        let cols = if self.store.has(&format!("{wkv_prefix}.scales")) { info.shape[1] * 4 } else { info.shape[1] };
        Ok(Some(DeepSeekV4EngramWeights {
            k: self.load_dense(&names.tensor("engram.k_weight"), &[engram.max_ngram_size, self.config.hidden_size])?,
            q: self.load_dense(&names.tensor("engram.q_weight"), &[engram.max_ngram_size, self.config.hidden_size])?,
            wkv: self.load_core_matrix(&wkv_prefix, rows, cols)?,
        }))
    }

    /// 按需读取 engram embed 行:官方为 MXFP8(E4M3+E8M0 g32),Jundot 为 mlx-affine 4bit g32。
    pub fn engram_embedding_rows(&self, layer: usize, rows: &[usize]) -> Result<DeepSeekV4EngramEmbedding, String> {
        let engram = self.config.engram.as_ref().ok_or("DeepSeek-V4 配置无 engram")?.clone();
        if !engram.layer_ids.contains(&layer) {
            return Err(format!("DeepSeek-V4 layer {layer} 非 engram 层(engram_layer_ids={:?})", engram.layer_ids));
        }
        let dim = engram.head_dim;
        let names = DeepSeekV4LayerNames::new(layer, self.ns);
        let base = names.tensor("engram.embed");
        let weight = self.store.load_rows_mapped(&format!("{base}.weight"), rows)?;
        if self.store.has(&format!("{base}.scales")) {
            if weight.dtype != "U32" {
                return Err(format!("{base}.weight dtype={}，期望 U32(mlx-affine 4bit)", weight.dtype));
            }
            let scales = self.store.load_rows_mapped(&format!("{base}.scales"), rows)?;
            let biases = self.store.load_rows_mapped(&format!("{base}.biases"), rows)?;
            if scales.dtype != "BF16" || biases.dtype != "BF16" {
                return Err(format!("{base}.scales/biases dtype={}/{}，期望 BF16", scales.dtype, biases.dtype));
            }
            return crate::weight::format::quantization::MlxAffineMatrix::new(weight.data, scales.data, biases.data, crate::weight::format::quantization::ScaleDType::Bf16, 4, 32, rows.len(), dim)
                .map(DeepSeekV4EngramEmbedding::MlxAffine)
                .map_err(|error| format!("{base}: {error}"));
        }
        if weight.dtype != "F8_E4M3" {
            return Err(format!("{base}.weight dtype={}，期望 F8_E4M3(MXFP8)或 U32(affine)", weight.dtype));
        }
        let scales = self.store.load_rows_mapped(&format!("{base}.scale"), rows)?;
        if !matches!(scales.dtype.as_str(), "F8_E8M0" | "U8") {
            return Err(format!("{base}.scale dtype={}，期望 F8_E8M0/U8", scales.dtype));
        }
        Mxfp8Matrix::new(weight.data, scales.data, rows.len(), dim).map(DeepSeekV4EngramEmbedding::Mxfp8).map_err(|error| format!("{base}: {error}"))
    }

    #[cfg(unix)]
    pub fn prefetch_engram_embedding(&self, layer: usize) -> Result<usize, String> {
        let names = DeepSeekV4LayerNames::new(layer, self.ns);
        let base = names.tensor("engram.embed");
        let mut bytes = self.store.prefetch_mapped(&format!("{base}.weight"))?;
        for suffix in ["scale", "scales", "biases"] {
            let name = format!("{base}.{suffix}");
            if self.store.has(&name) {
                bytes = bytes.checked_add(self.store.prefetch_mapped(&name)?).ok_or("engram resident 字节数溢出")?;
            }
        }
        Ok(bytes)
    }

    /// - `{prefix}.scale` + 二维 block scale 形状 → 官方 V4/V4.1 block-FP8
    /// - `{prefix}.scale` + [rows, cols/32] E8M0 → 行独立 MXFP8(weight 为未打包 F8_E4M3)
    /// - `{prefix}.scales`(复数)→ omlx/Jundot MXFP8(weight 为 U32 packed)
    /// - 皆无 → 未量化 dense(omlx 的 wo_a 等)
    pub(crate) fn load_core_matrix(&self, prefix: &str, rows: usize, cols: usize) -> Result<DeepSeekV4CoreMatrix, String> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.scale");
        if self.store.has(&scale_name) {
            let info = self.store.tensor_info(&scale_name)?;
            let block128 = [rows.div_ceil(FP8_BLOCK), cols.div_ceil(FP8_BLOCK)];
            let block32 = [rows.div_ceil(32), cols.div_ceil(32)];
            // V4.1 官方权重明确使用 32x32 二维 scale。小矩阵的 scale shape
            // 可能与 128x128 相同，必须先按模型版本选择，不能只看 shape 猜布局。
            if self.config.engram.is_some() && info.shape == block32 {
                return self.load_block_fp8_with_block(prefix, rows, cols, 32).map(DeepSeekV4CoreMatrix::BlockFp8);
            }
            if info.shape == block128 {
                return self.load_block_fp8_with_block(prefix, rows, cols, FP8_BLOCK).map(DeepSeekV4CoreMatrix::BlockFp8);
            }
            if !cols.is_multiple_of(32) || info.shape != [rows, cols / 32] {
                return Err(format!("{scale_name} shape={:?}，既非二维 block-FP8 亦非行独立 MXFP8(32)", info.shape));
            }
            let weight = self.store.load(&weight_name)?;
            weight.expect_shape(&[rows, cols])?;
            if weight.dtype != "F8_E4M3" {
                return Err(format!("{weight_name} dtype={}，期望 F8_E4M3(MXFP8 codes)", weight.dtype));
            }
            let scales = self.store.load(&scale_name)?;
            if !matches!(scales.dtype.as_str(), "F8_E8M0" | "U8") {
                return Err(format!("{scale_name} dtype={}，期望 F8_E8M0/U8", scales.dtype));
            }
            return Mxfp8Matrix::new(weight.data, scales.data, rows, cols).map(DeepSeekV4CoreMatrix::Mxfp8).map_err(|error| format!("{prefix}: {error}"));
        }
        let scales_name = format!("{prefix}.scales");
        if self.store.has(&scales_name) {
            if !cols.is_multiple_of(32) {
                return Err(format!("{prefix} MXFP8 列数 {cols} 不是 32 的倍数"));
            }
            let weight = self.store.load(&weight_name)?;
            weight.expect_shape(&[rows, cols / 4])?;
            if weight.dtype != "U32" {
                return Err(format!("{weight_name} dtype={}，期望 U32(MXFP8 codes)", weight.dtype));
            }
            let scales = self.store.load(&scales_name)?;
            scales.expect_shape(&[rows, cols / 32])?;
            if scales.dtype != "U8" {
                return Err(format!("{scales_name} dtype={}，期望 U8(E8M0)", scales.dtype));
            }
            // U32 小端字节即 4 个连续列的 E4M3 code,与 MXFP8 行主序布局一致
            return Mxfp8Matrix::new(weight.data, scales.data, rows, cols).map(DeepSeekV4CoreMatrix::Mxfp8).map_err(|error| format!("{prefix}: {error}"));
        }
        let tensor = self.store.load(&weight_name)?;
        tensor.expect_shape(&[rows, cols])?;
        if !matches!(tensor.dtype.as_str(), "BF16" | "F16" | "F32") {
            return Err(format!("{weight_name} dtype={}，期望量化或 BF16/F16/F32 dense", tensor.dtype));
        }
        Ok(DeepSeekV4CoreMatrix::Dense(tensor))
    }

    pub(crate) fn load_block_fp8_with_block(&self, prefix: &str, rows: usize, cols: usize, block: usize) -> Result<BlockFp8Matrix, String> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.scale");
        let weight = self.store.load(&weight_name)?;
        weight.expect_shape(&[rows, cols])?;
        if weight.dtype != "F8_E4M3" {
            return Err(format!("{weight_name} dtype={}，期望 F8_E4M3", weight.dtype));
        }
        let scale = self.store.load(&scale_name)?;
        scale.expect_shape(&[rows.div_ceil(block), cols.div_ceil(block)])?;
        if !matches!(scale.dtype.as_str(), "F8_E8M0" | "U8") {
            return Err(format!("{scale_name} dtype={}，期望 F8_E8M0/U8", scale.dtype));
        }
        BlockFp8Matrix::new(weight.data, scale.data, rows, cols, block, block).map_err(|error| format!("{prefix}: {error}"))
    }

    pub(crate) fn load_dense(&self, name: &str, shape: &[usize]) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        tensor.expect_shape(shape)?;
        if !matches!(tensor.dtype.as_str(), "BF16" | "F16" | "F32") {
            return Err(format!("{name} dtype={}，期望 BF16/F16/F32", tensor.dtype));
        }
        Ok(tensor)
    }

    pub(crate) fn load_f32(&self, name: &str, shape: &[usize]) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        tensor.expect_shape(shape)?;
        if tensor.dtype != "F32" {
            return Err(format!("{name} dtype={}，期望 F32", tensor.dtype));
        }
        Ok(tensor)
    }

    fn load_i32(&self, name: &str, shape: &[usize]) -> Result<TensorData, String> {
        let tensor = self.store.load(name)?;
        tensor.expect_shape(shape)?;
        // 官方 tid2eid 发布为 I64；专家 id 值域极小，I32/I64 两种存储都接受，由 runtime 统一窄化。
        let element_bytes = match tensor.dtype.as_str() {
            "I32" => 4,
            "I64" => 8,
            _ => return Err(format!("{name} dtype={}，期望 I32/I64", tensor.dtype)),
        };
        let elements = shape.iter().product::<usize>();
        if tensor.data.len() != elements * element_bytes {
            return Err(format!("{name} bytes={} 与 dtype={} shape={shape:?} 不兼容", tensor.data.len(), tensor.dtype));
        }
        Ok(tensor)
    }
}

#[derive(Clone)]
pub struct DeepSeekV4ExpertSource {
    weights: DeepSeekV4Weights,
    /// 层名空间前缀,含顶层 ns,如 "layers"、"language_model.mtp"。
    prefix: String,
    layer_count: usize,
    expert_count: usize,
}

impl Mxfp4ExpertSource for DeepSeekV4ExpertSource {
    fn intermediate(&self) -> usize {
        self.weights.config.expert_intermediate_size
    }

    fn hidden(&self) -> usize {
        self.weights.config.hidden_size
    }

    fn load_expert_mxfp4(&self, layer: usize, expert: usize) -> Result<Mxfp4ExpertWeights, String> {
        if layer >= self.layer_count || expert >= self.expert_count {
            return Err(format!("DeepSeek-V4 MXFP4 expert 越界: prefix={} layer={layer}/{}, expert={expert}/{}", self.prefix, self.layer_count, self.expert_count));
        }
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.expert_intermediate_size;
        let merged = format!("{}.{layer}.ffn.experts.w1.weight", self.prefix);
        if self.weights.store.has(&merged) {
            // omlx/Jundot:专家合并为 rank-3 [E, rows, packed],按展平行区间切片
            let load = |suffix: &str, rows: usize, cols: usize| -> Result<crate::weight::format::mxfp4::Mxfp4Matrix, String> {
                let base = format!("{}.{layer}.ffn.experts.{suffix}", self.prefix);
                let weight = self.weights.store.load_rows(&format!("{base}.weight"), &(expert * rows..(expert + 1) * rows).collect::<Vec<_>>())?;
                if weight.dtype != "U32" {
                    return Err(format!("{base}.weight dtype={}，期望 U32(MXFP4 codes)", weight.dtype));
                }
                let scales = self.weights.store.load_rows(&format!("{base}.scales"), &(expert * rows..(expert + 1) * rows).collect::<Vec<_>>())?;
                if scales.dtype != "U8" {
                    return Err(format!("{base}.scales dtype={}，期望 U8(E8M0)", scales.dtype));
                }
                crate::weight::format::mxfp4::Mxfp4Matrix::new(rows, cols, weight.data, scales.data)
            };
            return Ok(Mxfp4ExpertWeights { gate: load("w1", intermediate, hidden)?, up: load("w3", intermediate, hidden)?, down: load("w2", hidden, intermediate)? });
        }
        let prefix = format!("{}.{layer}.ffn.experts.{expert}", self.prefix);
        Ok(Mxfp4ExpertWeights {
            gate: load_mxfp4_matrix(&self.weights.store, &format!("{prefix}.w1.weight"), &format!("{prefix}.w1.scale"), intermediate, hidden)?,
            up: load_mxfp4_matrix(&self.weights.store, &format!("{prefix}.w3.weight"), &format!("{prefix}.w3.scale"), intermediate, hidden)?,
            down: load_mxfp4_matrix(&self.weights.store, &format!("{prefix}.w2.weight"), &format!("{prefix}.w2.scale"), hidden, intermediate)?,
        })
    }
}

impl ExpertSourceProvider for DeepSeekV4ExpertSource {
    fn source(&self, layer: usize) -> Result<ExpertSource<'_>, String> {
        if layer >= self.layer_count {
            return Err(format!("DeepSeek-V4 expert source prefix={} layer {layer} 越界，layer_count={}", self.prefix, self.layer_count));
        }
        Ok(ExpertSource::Mxfp4(self))
    }
}

/// DeepSeek-V4 标准 GGUF 权重视图。
///
/// 这里只映射标准 tensor 名称、shape 与分片生命周期；具体注意力和 MoE 顺序仍由
/// `runtime::deepseek_v4` 拥有，量化类型由 backend capability 选择执行路径。
#[derive(Clone)]
pub struct DeepSeekV4Gguf {
    reader: Arc<GgufReader>,
    config: DeepSeekV4Config,
}

impl DeepSeekV4Gguf {
    pub fn open(path: impl AsRef<Path>, config: DeepSeekV4Config) -> Result<Self, String> {
        validate_head_groups(&config)?;
        let path = GgufReader::locate(path.as_ref())?;
        let model = Self { reader: Arc::new(GgufReader::open(&path)?), config };
        model.validate_metadata()?;
        model.validate_tensors()?;
        Ok(model)
    }

    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.reader.tensors().iter().any(|tensor| tensor.name == name)
    }

    /// 估算一层非 routed-expert 权重的 GGUF 存储字节数，供 runtime 规划固定常驻层。
    pub fn layer_storage_bytes(&self, layer: usize) -> Result<usize, String> {
        if layer >= self.config.layer_count {
            return Err(format!("DeepSeek-V4 GGUF layer {layer} 越界，layer_count={}", self.config.layer_count));
        }
        let prefix = format!("blk.{layer}.");
        self.reader
            .tensors()
            .iter()
            .filter(|tensor| tensor.name.starts_with(&prefix) && !tensor.name.contains("_exps.weight"))
            .try_fold(0usize, |bytes, tensor| bytes.checked_add(tensor.bytes).ok_or_else(|| format!("DeepSeek-V4 GGUF layer {layer} 字节数溢出")))
    }

    pub fn config(&self) -> &DeepSeekV4Config {
        &self.config
    }

    pub fn matrix(&self, name: &str) -> Result<GgufMatrix, String> {
        self.reader.read_matrix(name)
    }

    pub fn vector_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        self.reader.read_tensor_f32(name)
    }

    pub fn tensor_i32(&self, name: &str) -> Result<Vec<i32>, String> {
        self.reader.read_tensor_i32(name)
    }

    pub fn embedding_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        self.reader.embedding_rows("token_embd.weight", token_ids, self.config.hidden_size, self.config.vocab_size)
    }

    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        self.reader.bpe_tokenizer().map_err(|error| format!("构造 DeepSeek-V4 tokenizer: {error}"))
    }

    pub fn detokenizer(&self) -> Result<Detokenizer, String> {
        self.reader.bpe_detokenizer().map_err(|error| format!("构造 DeepSeek-V4 detokenizer: {error}"))
    }

    /// GGUF metadata 前缀与 architecture 同名:deepseek4 或 deepseek41(V4.1)。
    fn metadata_prefix(&self) -> Result<&'static str, String> {
        match self.reader.metadata("general.architecture").and_then(crate::weight::container::gguf::GgufValue::as_str) {
            Some("deepseek4") => Ok("deepseek4"),
            Some("deepseek41") => Ok("deepseek41"),
            Some(other) => Err(format!("DeepSeek-V4 GGUF 未知架构 {other}")),
            None => Err("GGUF metadata general.architecture 缺失或类型错误".to_owned()),
        }
    }

    fn validate_metadata(&self) -> Result<(), String> {
        let cfg = &self.config;
        let prefix = self.metadata_prefix()?;
        for (key, expected) in [
            ("block_count", cfg.layer_count),
            ("context_length", cfg.max_position_embeddings),
            ("embedding_length", cfg.hidden_size),
            ("attention.head_count", cfg.num_heads),
            ("attention.head_count_kv", cfg.num_kv_heads),
            ("attention.key_length", cfg.head_dim),
            ("attention.value_length", cfg.head_dim),
            ("attention.q_lora_rank", cfg.q_lora_rank),
            ("attention.output_group_count", cfg.output_groups),
            ("attention.output_lora_rank", cfg.output_lora_rank),
            ("attention.indexer.head_count", cfg.index_heads),
            ("attention.indexer.key_length", cfg.index_head_dim),
            ("attention.indexer.top_k", cfg.index_top_k),
            ("attention.sliding_window", cfg.sliding_window),
            ("expert_count", cfg.expert_count),
            ("expert_used_count", cfg.expert_top_k),
            ("expert_feed_forward_length", cfg.expert_intermediate_size),
            ("expert_shared_count", cfg.shared_expert_count),
            ("hash_layer_count", cfg.hash_layer_count),
            ("hyper_connection.count", cfg.hyper_connection_copies),
            ("hyper_connection.sinkhorn_iterations", cfg.hyper_connection_sinkhorn_iterations),
        ] {
            self.reader.expect_metadata_u64(&format!("{prefix}.{key}"), expected as u64)?;
        }
        let ratios = self
            .reader
            .metadata(&format!("{prefix}.attention.compress_ratios"))
            .and_then(|value| value.as_i64_array())
            .and_then(|values| values.into_iter().map(usize::try_from).collect::<Result<Vec<_>, _>>().ok())
            .ok_or_else(|| format!("GGUF metadata {prefix}.attention.compress_ratios 缺失或含负数"))?;
        if ratios != cfg.compress_ratios {
            return Err(format!("DeepSeek-V4 GGUF compress_ratios={ratios:?}，与 runtime 配置不一致"));
        }
        if let Some(engram) = &cfg.engram {
            if prefix != "deepseek41" {
                return Err("engram 配置仅 deepseek41 架构支持".to_owned());
            }
            for (key, expected) in [("engram.head_count", engram.n_heads), ("engram.key_length", engram.head_dim), ("engram.max_ngram_size", engram.max_ngram_size)] {
                self.reader.expect_metadata_u64(&format!("{prefix}.{key}"), expected as u64)?;
            }
        }
        Ok(())
    }

    fn validate_tensors(&self) -> Result<(), String> {
        let cfg = &self.config;
        let is_v41 = self.metadata_prefix()? == "deepseek41";
        self.expect("token_embd.weight", &[cfg.hidden_size, cfg.vocab_size])?;
        self.expect("output.weight", &[cfg.hidden_size, cfg.vocab_size])?;
        self.expect_f32("output_norm.weight", &[cfg.hidden_size])?;
        self.expect_f32("output_hc_base.weight", &[cfg.hyper_connection_copies])?;
        self.expect_f32("output_hc_fn.weight", &[cfg.hyper_connection_copies * cfg.hidden_size, cfg.hyper_connection_copies])?;
        self.expect_f32("output_hc_scale.weight", &[1])?;

        let mixes = (2 + cfg.hyper_connection_copies) * cfg.hyper_connection_copies;
        for layer in 0..cfg.layer_count {
            let prefix = format!("blk.{layer}");
            self.expect(&format!("{prefix}.attn_q_a.weight"), &[cfg.hidden_size, cfg.q_lora_rank])?;
            self.expect_f32(&format!("{prefix}.attn_q_a_norm.weight"), &[cfg.q_lora_rank])?;
            self.expect(&format!("{prefix}.attn_q_b.weight"), &[cfg.q_lora_rank, cfg.num_heads * cfg.head_dim])?;
            self.expect(&format!("{prefix}.attn_kv.weight"), &[cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim])?;
            self.expect_f32(&format!("{prefix}.attn_kv_a_norm.weight"), &[cfg.head_dim])?;
            self.expect(&format!("{prefix}.attn_output_a.weight"), &[cfg.num_heads * cfg.head_dim / cfg.output_groups, cfg.output_groups * cfg.output_lora_rank])?;
            self.expect(&format!("{prefix}.attn_output_b.weight"), &[cfg.output_groups * cfg.output_lora_rank, cfg.hidden_size])?;
            self.expect_f32(&format!("{prefix}.attn_sinks.weight"), &[cfg.num_heads])?;
            self.expect_f32(&format!("{prefix}.attn_norm.weight"), &[cfg.hidden_size])?;
            self.expect(&format!("{prefix}.ffn_gate_inp.weight"), &[cfg.hidden_size, cfg.expert_count])?;
            self.expect_f32(&format!("{prefix}.ffn_norm.weight"), &[cfg.hidden_size])?;
            self.expect(&format!("{prefix}.ffn_gate_shexp.weight"), &[cfg.hidden_size, cfg.expert_intermediate_size])?;
            self.expect(&format!("{prefix}.ffn_up_shexp.weight"), &[cfg.hidden_size, cfg.expert_intermediate_size])?;
            self.expect(&format!("{prefix}.ffn_down_shexp.weight"), &[cfg.expert_intermediate_size, cfg.hidden_size])?;
            self.expect(&format!("{prefix}.ffn_gate_exps.weight"), &[cfg.hidden_size, cfg.expert_intermediate_size, cfg.expert_count])?;
            self.expect(&format!("{prefix}.ffn_up_exps.weight"), &[cfg.hidden_size, cfg.expert_intermediate_size, cfg.expert_count])?;
            self.expect(&format!("{prefix}.ffn_down_exps.weight"), &[cfg.expert_intermediate_size, cfg.hidden_size, cfg.expert_count])?;
            for sublayer in ["attn", "ffn"] {
                self.expect_f32(&format!("{prefix}.hc_{sublayer}_base.weight"), &[mixes])?;
                self.expect_f32(&format!("{prefix}.hc_{sublayer}_fn.weight"), &[cfg.hyper_connection_copies * cfg.hidden_size, mixes])?;
                self.expect_f32(&format!("{prefix}.hc_{sublayer}_scale.weight"), &[3])?;
            }
            if layer < cfg.hash_layer_count {
                let table = self.expect(&format!("{prefix}.ffn_gate_tid2eid.weight"), &[cfg.expert_top_k, cfg.vocab_size])?;
                if table.tensor_type != GgmlType(26) {
                    return Err(format!("{} type={}，期望 I32", table.name, table.tensor_type.name()));
                }
            } else {
                self.expect_f32(&format!("{prefix}.exp_probs_b.bias"), &[cfg.expert_count])?;
                if cfg.router_value_level_bias {
                    self.expect_f32(&format!("{prefix}.exp_probs_b_vl.bias"), &[cfg.expert_count])?;
                }
            }
            if let Some(engram) = &cfg.engram {
                if let Some(slot) = engram.layer_ids.iter().position(|&id| id == layer) {
                    self.expect(&format!("{prefix}.engram_embd.weight"), &[engram.head_dim, engram.num_embeddings[slot]])?;
                    self.expect(&format!("{prefix}.engram_k.weight"), &[cfg.hidden_size, engram.max_ngram_size])?;
                    self.expect(&format!("{prefix}.engram_q.weight"), &[cfg.hidden_size, engram.max_ngram_size])?;
                    // engram_wkv 实测 [3*n_heads*head_dim, 100*head_dim];第二维 100× 的语义待官方代码确认
                    self.expect(&format!("{prefix}.engram_wkv.weight"), &[3 * engram.n_heads * engram.head_dim, 100 * engram.head_dim])?;
                }
            }
            let ratio = cfg.compress_ratios[layer];
            if is_v41 {
                // CSA2 的 indexer/compressor 跨层共享,张量只落在组首层(实测 L2/8/14 全套、
                // L20 缺 gate、L24/28/32/36 仅 q_b+proj),按存在性逐个校验;
                // V4.1 compressor 无 ape 张量,channels=head_dim。
                self.expect_optional(&format!("{prefix}.indexer.attn_q_b.weight"), &[cfg.q_lora_rank, cfg.index_heads * cfg.index_head_dim])?;
                self.expect_optional(&format!("{prefix}.indexer.proj.weight"), &[cfg.hidden_size, cfg.index_heads])?;
                self.expect_optional(&format!("{prefix}.indexer.attn_k.weight"), &[cfg.head_dim, cfg.index_head_dim])?;
                self.expect_optional_f32(&format!("{prefix}.indexer.k_norm.weight"), &[cfg.index_head_dim])?;
                self.expect_optional(&format!("{prefix}.attn_compressor_kv.weight"), &[cfg.hidden_size, cfg.head_dim])?;
                self.expect_optional(&format!("{prefix}.attn_compressor_gate.weight"), &[cfg.hidden_size, cfg.head_dim])?;
                self.expect_optional_f32(&format!("{prefix}.attn_compressor_norm.weight"), &[cfg.head_dim])?;
            } else {
                if ratio != 0 {
                    self.validate_compressor(&prefix, "attn_compressor", ratio, cfg.head_dim)?;
                }
                if ratio == 4 {
                    self.expect(&format!("{prefix}.indexer.attn_q_b.weight"), &[cfg.q_lora_rank, cfg.index_heads * cfg.index_head_dim])?;
                    self.expect(&format!("{prefix}.indexer.proj.weight"), &[cfg.hidden_size, cfg.index_heads])?;
                    self.validate_compressor(&prefix, "indexer_compressor", ratio, cfg.index_head_dim)?;
                }
            }
        }
        Ok(())
    }

    /// 张量存在才校验形状,返回是否存在;用于 V4.1 跨层共享的可选张量。
    fn expect_optional(&self, name: &str, dims: &[usize]) -> Result<bool, String> {
        let Some(tensor) = self.reader.tensors().iter().find(|tensor| tensor.name == name) else { return Ok(false) };
        if tensor.dims != dims {
            return Err(format!("{} dims={:?}，期望 {dims:?}", tensor.name, tensor.dims));
        }
        Ok(true)
    }

    fn expect_optional_f32(&self, name: &str, dims: &[usize]) -> Result<bool, String> {
        if self.expect_optional(name, dims)? {
            let tensor = self.reader.expect_tensor(name, dims)?;
            if tensor.tensor_type != GgmlType(0) {
                return Err(format!("{name} type={}，期望 F32", tensor.tensor_type.name()));
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn validate_compressor(&self, prefix: &str, name: &str, ratio: usize, width: usize) -> Result<(), String> {
        let channels = if ratio == 4 { 2 * width } else { width };
        self.expect_f32(&format!("{prefix}.{name}_ape.weight"), &[channels, ratio])?;
        self.expect(&format!("{prefix}.{name}_gate.weight"), &[self.config.hidden_size, channels])?;
        self.expect(&format!("{prefix}.{name}_kv.weight"), &[self.config.hidden_size, channels])?;
        self.expect_f32(&format!("{prefix}.{name}_norm.weight"), &[width])?;
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
}

impl GgufExpertSource for DeepSeekV4Gguf {
    fn intermediate(&self) -> usize {
        self.config.expert_intermediate_size
    }

    fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    fn load_expert_gguf(&self, layer: usize, expert: usize) -> Result<GgufExpertWeights, String> {
        if layer >= self.config.layer_count || expert >= self.config.expert_count {
            return Err(format!("DeepSeek-V4 GGUF expert 越界: layer={layer}/{}, expert={expert}/{}", self.config.layer_count, self.config.expert_count));
        }
        Ok(GgufExpertWeights {
            gate: self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_gate_exps.weight"), expert)?,
            up: self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_up_exps.weight"), expert)?,
            down: self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_down_exps.weight"), expert)?,
        })
    }
}

impl ExpertSourceProvider for DeepSeekV4Gguf {
    fn source(&self, layer: usize) -> Result<ExpertSource<'_>, String> {
        if layer >= self.config.layer_count {
            return Err(format!("DeepSeek-V4 GGUF expert source layer {layer} 越界，layer_count={}", self.config.layer_count));
        }
        Ok(ExpertSource::Gguf(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_names_match_official_checkpoint() {
        let names = DeepSeekV4LayerNames::new(2, "");
        assert_eq!(names.attention("indexer.wq_b.weight"), "layers.2.attn.indexer.wq_b.weight");
        assert_eq!(names.feedforward("gate.tid2eid"), "layers.2.ffn.gate.tid2eid");
        let omlx = DeepSeekV4LayerNames::new(2, "language_model.");
        assert_eq!(omlx.attention("wq_a.weight"), "language_model.layers.2.attn.wq_a.weight");
    }
}
