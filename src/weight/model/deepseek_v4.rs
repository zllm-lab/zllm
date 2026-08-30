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
    },
};

const FP8_BLOCK: usize = 128;
const EMBEDDING: &str = "embed.weight";
const FINAL_NORM: &str = "norm.weight";
const LM_HEAD: &str = "head.weight";

/// 核心张量实际出现的 dtype；未知 dtype 显式报错而不是猜字节宽。
fn dtype_bytes(dtype: &str, name: &str) -> Result<usize, String> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" | "I32" => Ok(4),
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
    pub fn new(layer: usize) -> Self {
        Self { prefix: format!("layers.{layer}") }
    }

    fn mtp(layer: usize) -> Self {
        Self { prefix: format!("mtp.{layer}") }
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
    pub input_projection: BlockFp8Matrix,
    pub norm: TensorData,
    pub output_projection: BlockFp8Matrix,
}

pub struct DeepSeekV4KeyValueWeights {
    pub projection: BlockFp8Matrix,
    pub norm: TensorData,
}

pub struct DeepSeekV4OutputWeights {
    pub input_projection: BlockFp8Matrix,
    pub output_projection: BlockFp8Matrix,
}

pub struct DeepSeekV4CompressorWeights {
    pub position: TensorData,
    pub key_value_projection: TensorData,
    pub gate_projection: TensorData,
    pub norm: TensorData,
}

pub struct DeepSeekV4IndexerWeights {
    pub query_projection: BlockFp8Matrix,
    pub head_weights_projection: TensorData,
    pub compressor: DeepSeekV4CompressorWeights,
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
    ScoreTopK { weight: TensorData, correction_bias: TensorData },
}

pub struct DeepSeekV4SharedExpertWeights {
    pub gate: BlockFp8Matrix,
    pub up: BlockFp8Matrix,
    pub down: BlockFp8Matrix,
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
    pub hyper_connection: DeepSeekV4HyperConnectionWeights,
    pub norm: TensorData,
    pub lm_head: TensorData,
}

#[derive(Clone)]
pub struct DeepSeekV4Weights {
    store: Arc<SafetensorStore>,
    config: DeepSeekV4Config,
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
        Ok(Self { store: Arc::new(SafetensorStore::open(root.as_ref())?), config })
    }

    pub fn config(&self) -> &DeepSeekV4Config {
        &self.config
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
        let tensor = self.store.load_bf16_rows(EMBEDDING, &rows)?;
        tensor.expect_shape(&[rows.len(), self.config.hidden_size])?;
        Ok(tensor.data)
    }

    pub fn final_norm(&self) -> Result<TensorData, String> {
        self.load_dense(FINAL_NORM, &[self.config.hidden_size])
    }

    /// LM head 很大，只在 backend 准备输出投影时显式读取。
    pub fn lm_head(&self) -> Result<TensorData, String> {
        self.load_dense(LM_HEAD, &[self.config.vocab_size, self.config.hidden_size])
    }

    pub fn output_head(&self) -> Result<DeepSeekV4HeadWeights, String> {
        let copies = self.config.hyper_connection_copies;
        let hidden = copies.checked_mul(self.config.hidden_size).ok_or("DeepSeek-V4 head hidden 维度溢出")?;
        Ok(DeepSeekV4HeadWeights {
            hyper_connection: DeepSeekV4HyperConnectionWeights { function: self.load_f32("hc_head_fn", &[copies, hidden])?, base: self.load_f32("hc_head_base", &[copies])?, scale: self.load_f32("hc_head_scale", &[1])? },
            norm: self.final_norm()?,
            lm_head: self.lm_head()?,
        })
    }

    pub fn load_layer(&self, layer: usize) -> Result<DeepSeekV4LayerWeights, String> {
        if layer >= self.config.layer_count {
            return Err(format!("DeepSeek-V4 layer {layer} 越界，layer_count={}", self.config.layer_count));
        }
        let names = DeepSeekV4LayerNames::new(layer);
        let spec = self.config.compress_ratios[layer];
        let routing = if layer < self.config.hash_layer_count { DeepSeekV4RoutingSelection::TokenHash } else { DeepSeekV4RoutingSelection::ScoreTopK };
        self.load_named_layer(&names, spec, routing)
    }

    pub fn load_mtp_layer(&self, layer: usize) -> Result<DeepSeekV4LayerWeights, String> {
        if layer >= self.config.mtp_layer_count {
            return Err(format!("DeepSeek-V4 MTP layer {layer} 越界，mtp_layer_count={}", self.config.mtp_layer_count));
        }
        let names = DeepSeekV4LayerNames::mtp(layer);
        let spec = self.config.compress_ratios[self.config.layer_count + layer];
        self.load_named_layer(&names, spec, DeepSeekV4RoutingSelection::ScoreTopK)
    }

    fn load_named_layer(&self, names: &DeepSeekV4LayerNames, spec: usize, routing: DeepSeekV4RoutingSelection) -> Result<DeepSeekV4LayerWeights, String> {
        let attention = DeepSeekV4AttentionBlockWeights {
            hyper_connection: self.load_hyper_connection(&names, "attn")?,
            norm: self.load_dense(&names.tensor("attn_norm.weight"), &[self.config.hidden_size])?,
            attention: DeepSeekV4AttentionWeights {
                sink: self.load_f32(&names.attention("attn_sink"), &[self.config.num_heads])?,
                query: DeepSeekV4QueryWeights {
                    input_projection: self.load_block_fp8(&names.attention("wq_a"), self.config.q_lora_rank, self.config.hidden_size)?,
                    norm: self.load_dense(&names.attention("q_norm.weight"), &[self.config.q_lora_rank])?,
                    output_projection: self.load_block_fp8(&names.attention("wq_b"), self.config.num_heads * self.config.head_dim, self.config.q_lora_rank)?,
                },
                key_value: DeepSeekV4KeyValueWeights {
                    projection: self.load_block_fp8(&names.attention("wkv"), self.config.head_dim, self.config.hidden_size)?,
                    norm: self.load_dense(&names.attention("kv_norm.weight"), &[self.config.head_dim])?,
                },
                output: DeepSeekV4OutputWeights {
                    input_projection: self.load_block_fp8(&names.attention("wo_a"), self.config.output_groups * self.config.output_lora_rank, self.config.num_heads * self.config.head_dim / self.config.output_groups)?,
                    output_projection: self.load_block_fp8(&names.attention("wo_b"), self.config.hidden_size, self.config.output_groups * self.config.output_lora_rank)?,
                },
                compressor: (spec != 0).then(|| self.load_compressor(&names, "attn.compressor", spec, self.config.head_dim)).transpose()?,
                indexer: (spec == 4).then(|| self.load_indexer(&names, spec)).transpose()?,
            },
        };
        let router_weight = self.load_dense(&names.feedforward("gate.weight"), &[self.config.expert_count, self.config.hidden_size])?;
        let router = match routing {
            DeepSeekV4RoutingSelection::TokenHash => DeepSeekV4RouterWeights::TokenHash { weight: router_weight, token_to_experts: self.load_i32(&names.feedforward("gate.tid2eid"), &[self.config.vocab_size, self.config.expert_top_k])? },
            DeepSeekV4RoutingSelection::ScoreTopK => DeepSeekV4RouterWeights::ScoreTopK { weight: router_weight, correction_bias: self.load_f32(&names.feedforward("gate.bias"), &[self.config.expert_count])? },
        };
        Ok(DeepSeekV4LayerWeights {
            attention,
            feedforward: DeepSeekV4FeedforwardBlockWeights {
                hyper_connection: self.load_hyper_connection(&names, "ffn")?,
                norm: self.load_dense(&names.tensor("ffn_norm.weight"), &[self.config.hidden_size])?,
                router,
                shared_expert: DeepSeekV4SharedExpertWeights {
                    gate: self.load_block_fp8(&names.feedforward("shared_experts.w1"), self.config.expert_intermediate_size, self.config.hidden_size)?,
                    up: self.load_block_fp8(&names.feedforward("shared_experts.w3"), self.config.expert_intermediate_size, self.config.hidden_size)?,
                    down: self.load_block_fp8(&names.feedforward("shared_experts.w2"), self.config.hidden_size, self.config.expert_intermediate_size)?,
                },
            },
        })
    }

    pub fn expert_source(&self) -> DeepSeekV4ExpertSource {
        DeepSeekV4ExpertSource { weights: self.clone(), namespace: "layers", layer_count: self.config.layer_count }
    }

    pub fn mtp_expert_source(&self) -> DeepSeekV4ExpertSource {
        DeepSeekV4ExpertSource { weights: self.clone(), namespace: "mtp", layer_count: self.config.mtp_layer_count }
    }

    /// 估算一层非 routed-expert 权重的存储字节数，供 runtime 规划固定常驻层；
    /// 与 GGUF 版 `DeepSeekV4Gguf::layer_storage_bytes` 语义一致。
    pub fn layer_storage_bytes(&self, layer: usize) -> Result<usize, String> {
        if layer >= self.config.layer_count {
            return Err(format!("DeepSeek-V4 layer {layer} 越界，layer_count={}", self.config.layer_count));
        }
        let prefix = format!("layers.{layer}.");
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
        Ok(DeepSeekV4CompressorWeights {
            position: self.load_f32(&names.tensor(&format!("{suffix}.ape")), &[ratio, channels])?,
            key_value_projection: self.load_dense(&names.tensor(&format!("{suffix}.wkv.weight")), &[channels, self.config.hidden_size])?,
            gate_projection: self.load_dense(&names.tensor(&format!("{suffix}.wgate.weight")), &[channels, self.config.hidden_size])?,
            norm: self.load_dense(&names.tensor(&format!("{suffix}.norm.weight")), &[head_dim])?,
        })
    }

    fn load_indexer(&self, names: &DeepSeekV4LayerNames, ratio: usize) -> Result<DeepSeekV4IndexerWeights, String> {
        Ok(DeepSeekV4IndexerWeights {
            query_projection: self.load_block_fp8(&names.attention("indexer.wq_b"), self.config.index_heads * self.config.index_head_dim, self.config.q_lora_rank)?,
            head_weights_projection: self.load_dense(&names.attention("indexer.weights_proj.weight"), &[self.config.index_heads, self.config.hidden_size])?,
            compressor: self.load_compressor(names, "attn.indexer.compressor", ratio, self.config.index_head_dim)?,
        })
    }

    pub(crate) fn load_block_fp8(&self, prefix: &str, rows: usize, cols: usize) -> Result<BlockFp8Matrix, String> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.scale");
        let weight = self.store.load(&weight_name)?;
        weight.expect_shape(&[rows, cols])?;
        if weight.dtype != "F8_E4M3" {
            return Err(format!("{weight_name} dtype={}，期望 F8_E4M3", weight.dtype));
        }
        let scale = self.store.load(&scale_name)?;
        scale.expect_shape(&[rows.div_ceil(FP8_BLOCK), cols.div_ceil(FP8_BLOCK)])?;
        if !matches!(scale.dtype.as_str(), "F8_E8M0" | "U8") {
            return Err(format!("{scale_name} dtype={}，期望 F8_E8M0/U8", scale.dtype));
        }
        BlockFp8Matrix::new(weight.data, scale.data, rows, cols, FP8_BLOCK, FP8_BLOCK).map_err(|error| format!("{prefix}: {error}"))
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
    namespace: &'static str,
    layer_count: usize,
}

impl Mxfp4ExpertSource for DeepSeekV4ExpertSource {
    fn intermediate(&self) -> usize {
        self.weights.config.expert_intermediate_size
    }

    fn hidden(&self) -> usize {
        self.weights.config.hidden_size
    }

    fn load_expert_mxfp4(&self, layer: usize, expert: usize) -> Result<Mxfp4ExpertWeights, String> {
        if layer >= self.layer_count || expert >= self.weights.config.expert_count {
            return Err(format!("DeepSeek-V4 MXFP4 expert 越界: namespace={} layer={layer}/{}, expert={expert}/{}", self.namespace, self.layer_count, self.weights.config.expert_count));
        }
        let prefix = format!("{}.{layer}.ffn.experts.{expert}", self.namespace);
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.expert_intermediate_size;
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
            return Err(format!("DeepSeek-V4 expert source namespace={} layer {layer} 越界，layer_count={}", self.namespace, self.layer_count));
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

    fn validate_metadata(&self) -> Result<(), String> {
        let cfg = &self.config;
        self.reader.expect_metadata_str("general.architecture", "deepseek4")?;
        for (key, expected) in [
            ("deepseek4.block_count", cfg.layer_count),
            ("deepseek4.context_length", cfg.max_position_embeddings),
            ("deepseek4.embedding_length", cfg.hidden_size),
            ("deepseek4.attention.head_count", cfg.num_heads),
            ("deepseek4.attention.head_count_kv", cfg.num_kv_heads),
            ("deepseek4.attention.key_length", cfg.head_dim),
            ("deepseek4.attention.value_length", cfg.head_dim),
            ("deepseek4.attention.q_lora_rank", cfg.q_lora_rank),
            ("deepseek4.attention.output_group_count", cfg.output_groups),
            ("deepseek4.attention.output_lora_rank", cfg.output_lora_rank),
            ("deepseek4.attention.indexer.head_count", cfg.index_heads),
            ("deepseek4.attention.indexer.key_length", cfg.index_head_dim),
            ("deepseek4.attention.indexer.top_k", cfg.index_top_k),
            ("deepseek4.attention.sliding_window", cfg.sliding_window),
            ("deepseek4.expert_count", cfg.expert_count),
            ("deepseek4.expert_used_count", cfg.expert_top_k),
            ("deepseek4.expert_feed_forward_length", cfg.expert_intermediate_size),
            ("deepseek4.expert_shared_count", cfg.shared_expert_count),
            ("deepseek4.hash_layer_count", cfg.hash_layer_count),
            ("deepseek4.hyper_connection.count", cfg.hyper_connection_copies),
            ("deepseek4.hyper_connection.sinkhorn_iterations", cfg.hyper_connection_sinkhorn_iterations),
        ] {
            self.reader.expect_metadata_u64(key, expected as u64)?;
        }
        let ratios = self
            .reader
            .metadata("deepseek4.attention.compress_ratios")
            .and_then(|value| value.as_i64_array())
            .and_then(|values| values.into_iter().map(usize::try_from).collect::<Result<Vec<_>, _>>().ok())
            .ok_or_else(|| "GGUF metadata deepseek4.attention.compress_ratios 缺失或含负数".to_owned())?;
        if ratios != cfg.compress_ratios {
            return Err(format!("DeepSeek-V4 GGUF compress_ratios={ratios:?}，与 runtime 配置不一致"));
        }
        Ok(())
    }

    fn validate_tensors(&self) -> Result<(), String> {
        let cfg = &self.config;
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
            }
            let ratio = cfg.compress_ratios[layer];
            if ratio != 0 {
                self.validate_compressor(&prefix, "attn_compressor", ratio, cfg.head_dim)?;
            }
            if ratio == 4 {
                self.expect(&format!("{prefix}.indexer.attn_q_b.weight"), &[cfg.q_lora_rank, cfg.index_heads * cfg.index_head_dim])?;
                self.expect(&format!("{prefix}.indexer.proj.weight"), &[cfg.hidden_size, cfg.index_heads])?;
                self.validate_compressor(&prefix, "indexer_compressor", ratio, cfg.index_head_dim)?;
            }
        }
        Ok(())
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
        let names = DeepSeekV4LayerNames::new(2);
        assert_eq!(names.attention("indexer.wq_b.weight"), "layers.2.attn.indexer.wq_b.weight");
        assert_eq!(names.feedforward("gate.tid2eid"), "layers.2.ffn.gate.tid2eid");
    }
}
