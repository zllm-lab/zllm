//! DeepSeek-V4-Flash 的平台无关运行时编排组件。
//!
//! 这里仅保存模型选择：普通层使用短上下文 RoPE，压缩层使用扩展 YaRN RoPE。
//! mHC/CSA 的数学语义在 `attention`，设备资源与 kernel 由 backend capability 拥有。

pub mod engram_cpu;
pub mod protocol;
pub mod vision;
#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
pub(crate) use protocol::resume_suffix as deepseek_v4_resume_suffix;
#[cfg(test)]
use protocol::{ASSISTANT, BOS, EOS, HIGH_REASONING_PREFIX, USER};
pub use protocol::{DeepSeekV4ChatMessage, chat_prompt as deepseek_v4_chat_prompt, chat_prompt_with_public_prefix as deepseek_v4_chat_prompt_with_public_prefix};

use crate::{
    attention::{
        compressed_sparse::{CompressedBatch, CompressedSparseKernel, CompressionStream, SharedCompressedBatch},
        hyper_connection::HyperConnectionKernel,
        rope::{RopeTable, RotaryLayout},
    },
    backend::{Backend, BackendError, ExpertPrefillBackend, LinearWeight},
    weight::{
        container::safetensor::TensorData,
        model::deepseek_v4::{DeepSeekV4AttentionWeights, DeepSeekV4Gguf, DeepSeekV4HyperConnectionWeights, DeepSeekV4RouterWeights, DeepSeekV4Weights},
    },
};

pub struct DeepSeekV4RopeTables {
    sliding: RopeTable,
    sliding_inverse_sin: Vec<f32>,
    compressed: RopeTable,
    compressed_inverse_sin: Vec<f32>,
}

#[derive(Clone, Copy)]
pub struct DeepSeekV4Rope<'a> {
    table: &'a RopeTable,
    inverse_sin: &'a [f32],
}

impl DeepSeekV4RopeTables {
    pub fn new(model: &DeepSeekV4, sequence_len: usize) -> Result<Self, BackendError> {
        let sliding_spec = model.layer_spec(0).map_err(|error| crate::runtime::compute_error(error.to_string()))?.attention.rope;
        let compressed_layer = (0..model.layer_count())
            .find_map(|layer| {
                let spec = model.layer_spec(layer).ok()?;
                spec.attention.compression.map(|_| spec.attention.rope)
            })
            .ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4 缺少压缩注意力层"))?;
        let sliding = RopeTable::from_spec(sequence_len, sliding_spec).map_err(crate::runtime::compute_error)?;
        let compressed = RopeTable::from_spec(sequence_len, compressed_layer).map_err(crate::runtime::compute_error)?;
        let sliding_inverse_sin = sliding.sin.iter().map(|value| -*value).collect();
        let compressed_inverse_sin = compressed.sin.iter().map(|value| -*value).collect();
        Ok(Self { sliding, sliding_inverse_sin, compressed, compressed_inverse_sin })
    }

    pub fn layer(&self, spec: &DeepSeekV4LayerSpec) -> DeepSeekV4Rope<'_> {
        if spec.attention.compression.is_some() { DeepSeekV4Rope { table: &self.compressed, inverse_sin: &self.compressed_inverse_sin } } else { DeepSeekV4Rope { table: &self.sliding, inverse_sin: &self.sliding_inverse_sin } }
    }
}

#[derive(Clone)]
struct PreparedHyperConnection<W> {
    input_norm: W,
    function: W,
    base: W,
    scale: W,
}

struct PreparedCompressor<W> {
    /// V4 的 ape 位置偏置;V4.1 compressor 无 ape。
    position: Option<W>,
    key_value: W,
    /// ratio>1 的 softmax 池化门;ratio=1 无。
    gate: Option<W>,
    norm: W,
}

struct PreparedIndexer<W> {
    query: W,
    head_weights: W,
    /// V4.1 跨层共享 indexer:K 投影与 norm 只落在组首(kv_source)层,共享层为 None。
    key_projection: Option<W>,
    key_norm: Option<W>,
    /// V4 的 indexer compressor;V4.1 indexer 无(压缩 latent 直接经 wk 投影)。
    compressor: Option<PreparedCompressor<W>>,
}

struct PreparedAttention<W> {
    sink: W,
    query_input: W,
    query_norm: W,
    query_output: W,
    query_head_norm: W,
    key_value: W,
    key_norm: W,
    output_inputs: Vec<W>,
    output: W,
    compressor: Option<PreparedCompressor<W>>,
    indexer: Option<PreparedIndexer<W>>,
}

enum PreparedRouter<W> {
    TokenHash { weight: W, zero_bias: W, token_to_experts: Vec<u32> },
    ScoreTopK { weight: W, correction_bias: W, value_level_bias: Option<W> },
}

/// 核心矩阵(host 侧三种量化形态)到 backend 权重视图的适配。
fn core_weight(matrix: &crate::weight::model::deepseek_v4::DeepSeekV4CoreMatrix) -> Result<LinearWeight<'_>, BackendError> {
    use crate::weight::model::deepseek_v4::DeepSeekV4CoreMatrix;
    Ok(match matrix {
        DeepSeekV4CoreMatrix::BlockFp8(matrix) => LinearWeight::block_fp8(matrix),
        DeepSeekV4CoreMatrix::Mxfp8(matrix) => LinearWeight::mxfp8(matrix),
        DeepSeekV4CoreMatrix::Dense(tensor) if tensor.dtype == "BF16" => LinearWeight::Bf16Bytes(&tensor.data),
        DeepSeekV4CoreMatrix::Dense(tensor) => return Err(crate::runtime::compute_error(format!("dense 核心矩阵 {} dtype={} 暂不支持 prepare", tensor.name, tensor.dtype))),
    })
}

/// output wo_a 的 grouped 准备:BlockFp8 走 backend 原生分组;dense BF16 按行区间切组。
fn prepare_grouped_core<B: Backend>(backend: &B, matrix: &crate::weight::model::deepseek_v4::DeepSeekV4CoreMatrix, groups: usize, rows_per_group: usize) -> Result<Vec<B::Weight>, BackendError> {
    use crate::weight::model::deepseek_v4::DeepSeekV4CoreMatrix;
    match matrix {
        DeepSeekV4CoreMatrix::BlockFp8(matrix) => backend.prepare_grouped_block_fp8(matrix, groups, rows_per_group),
        DeepSeekV4CoreMatrix::Dense(tensor) if tensor.dtype == "BF16" => {
            let cols = tensor.shape[1];
            let group_bytes = rows_per_group.checked_mul(cols).and_then(|v| v.checked_mul(2)).ok_or_else(|| crate::runtime::compute_error("grouped dense 切片字节数溢出"))?;
            if tensor.shape[0] != groups * rows_per_group || tensor.data.len() != groups * group_bytes {
                return Err(crate::runtime::compute_error(format!("grouped dense {} shape={:?} 与 groups={groups} rows={rows_per_group} 不匹配", tensor.name, tensor.shape)));
            }
            (0..groups).map(|group| backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data[group * group_bytes..(group + 1) * group_bytes]), rows_per_group, cols)).collect()
        }
        _ => Err(crate::runtime::compute_error("grouped 核心矩阵形态暂不支持(仅 BlockFp8/dense BF16)")),
    }
}

struct PreparedFeedforward<W> {
    hyper_connection: PreparedHyperConnection<W>,
    norm: W,
    router: PreparedRouter<W>,
    shared_gate: W,
    shared_up: W,
    shared_down: W,
}

pub struct DeepSeekV4PreparedLayer<W> {
    attention_hyper_connection: PreparedHyperConnection<W>,
    attention_norm: W,
    attention: PreparedAttention<W>,
    feedforward: PreparedFeedforward<W>,
}

/// 固定保留一组核心层；它拥有设备权重及其跨 token 生命周期。
/// 常驻规划只依赖每层存储字节数，对 GGUF 与官方 safetensors 来源一视同仁。
pub struct DeepSeekV4LayerCache<W> {
    layers: Vec<Option<DeepSeekV4PreparedLayer<W>>>,
    resident: Vec<bool>,
    planned_bytes: usize,
}

impl<W> DeepSeekV4LayerCache<W> {
    pub fn with_layer_bytes(layer_bytes: Vec<usize>, budget_bytes: usize) -> Self {
        let mut sizes = layer_bytes.into_iter().enumerate().collect::<Vec<_>>();
        // 每个常驻字节都能省下一次 token 读取；同预算优先覆盖更多较小层，减少文件访问次数。
        sizes.sort_unstable_by_key(|&(layer, bytes)| (bytes, layer));
        let mut resident = vec![false; sizes.len()];
        let mut planned_bytes = 0usize;
        for (layer, bytes) in sizes {
            let Some(next) = planned_bytes.checked_add(bytes) else { break };
            if next > budget_bytes {
                continue;
            }
            resident[layer] = true;
            planned_bytes = next;
        }
        Self { layers: (0..resident.len()).map(|_| None).collect(), resident, planned_bytes }
    }

    pub fn planned_layers(&self) -> usize {
        self.resident.iter().filter(|&&resident| resident).count()
    }

    pub fn loaded_layers(&self) -> usize {
        self.layers.iter().filter(|layer| layer.is_some()).count()
    }

    pub fn planned_bytes(&self) -> usize {
        self.planned_bytes
    }

    pub fn get(&self, layer: usize) -> Option<&DeepSeekV4PreparedLayer<W>> {
        self.layers.get(layer).and_then(|layer| layer.as_ref())
    }

    /// 插入常驻层并返回引用;超出预算规划(usize::MAX)的层也允许临时缓存,
    /// 由调用方在层结束后释放。
    pub fn put(&mut self, layer: usize, prepared: DeepSeekV4PreparedLayer<W>) -> &DeepSeekV4PreparedLayer<W> {
        assert!(layer < self.layers.len(), "layer cache 容量在构造时按层数固定");
        self.layers[layer] = Some(prepared);
        self.layers[layer].as_ref().expect("刚插入的层必然存在")
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod dspark_rocm;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_engine;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_node;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_stage;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_swap;

#[derive(Clone)]
pub struct DeepSeekV4OutputHead<W> {
    input_norm: W,
    /// V4 的 head hyper-connection;V4.1 head 无 hc 系数(checkpoint 无 hc_head_*)。
    hyper_connection: Option<PreparedHyperConnection<W>>,
    output: crate::runtime::output::OutputHead<W>,
}

pub fn prepare_deepseek_v4_layer<B: Backend>(backend: &B, source: &DeepSeekV4Weights, layer: usize) -> Result<DeepSeekV4PreparedLayer<B::Weight>, BackendError> {
    let config = source.config();
    let weights = source.load_layer(layer).map_err(crate::runtime::compute_error)?;
    prepare_deepseek_v4_loaded_layer(backend, config, &weights)
}

pub(crate) fn prepare_deepseek_v4_loaded_layer<B: Backend>(backend: &B, config: &DeepSeekV4Config, weights: &crate::weight::model::deepseek_v4::DeepSeekV4LayerWeights) -> Result<DeepSeekV4PreparedLayer<B::Weight>, BackendError> {
    let attention = prepare_attention(backend, config, &weights.attention.attention)?;
    let router = match &weights.feedforward.router {
        DeepSeekV4RouterWeights::TokenHash { weight, token_to_experts } => {
            let selected = tensor_i32(token_to_experts)?
                .into_iter()
                .map(|expert| u32::try_from(expert).ok().filter(|&expert| expert < config.expert_count as u32).ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 token hash expert id {expert} 越界"))))
                .collect::<Result<Vec<_>, _>>()?;
            PreparedRouter::TokenHash { weight: prepare_f32_tensor(backend, weight)?, zero_bias: backend.prepare_f32(&vec![0.0; config.expert_count], 1, config.expert_count)?, token_to_experts: selected }
        }
        DeepSeekV4RouterWeights::ScoreTopK { weight, correction_bias, value_level_bias } => PreparedRouter::ScoreTopK {
            weight: prepare_f32_tensor(backend, weight)?,
            correction_bias: prepare_f32_tensor(backend, correction_bias)?,
            value_level_bias: value_level_bias.as_ref().map(|bias| prepare_f32_tensor(backend, bias)).transpose()?,
        },
    };
    Ok(DeepSeekV4PreparedLayer {
        attention_hyper_connection: prepare_hyper_connection(backend, &weights.attention.hyper_connection)?,
        attention_norm: prepare_dense_tensor(backend, &weights.attention.norm)?,
        attention,
        feedforward: PreparedFeedforward {
            hyper_connection: prepare_hyper_connection(backend, &weights.feedforward.hyper_connection)?,
            norm: prepare_f32_tensor(backend, &weights.feedforward.norm)?,
            router,
            shared_gate: backend.prepare_weight(core_weight(&weights.feedforward.shared_expert.gate)?, config.expert_intermediate_size, config.hidden_size)?,
            shared_up: backend.prepare_weight(core_weight(&weights.feedforward.shared_expert.up)?, config.expert_intermediate_size, config.hidden_size)?,
            shared_down: backend.prepare_weight(core_weight(&weights.feedforward.shared_expert.down)?, config.hidden_size, config.expert_intermediate_size)?,
        },
    })
}

/// 从标准 GGUF 准备 DeepSeek-V4 层；runtime 编排与 Safetensors 路径共用同一份
/// `DeepSeekV4PreparedLayer`，因此格式差异不会渗入 attention/MoE 执行流程。
pub fn prepare_deepseek_v4_gguf_layer<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, layer: usize) -> Result<DeepSeekV4PreparedLayer<B::Weight>, BackendError> {
    let config = source.config();
    if layer >= config.layer_count {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 GGUF layer {layer} 越界，layer_count={}", config.layer_count)));
    }
    let prefix = format!("blk.{layer}");
    let matrix = |suffix: &str, rows: usize, cols: usize| prepare_gguf_matrix(backend, source, &format!("{prefix}.{suffix}"), rows, cols);
    let vector = |suffix: &str, columns: usize| prepare_gguf_vector(backend, source, &format!("{prefix}.{suffix}"), columns);
    let query_input = matrix("attn_q_a.weight", config.q_lora_rank, config.hidden_size)?;
    let query_norm = prepare_gguf_dense_vector(backend, source, &format!("{prefix}.attn_q_a_norm.weight"), config.q_lora_rank)?;
    let query_output = matrix("attn_q_b.weight", config.num_heads * config.head_dim, config.q_lora_rank)?;
    let key_value = matrix("attn_kv.weight", config.num_kv_heads * config.head_dim, config.hidden_size)?;
    let key_norm = prepare_gguf_dense_vector(backend, source, &format!("{prefix}.attn_kv_a_norm.weight"), config.head_dim)?;
    let output_a = source.matrix(&format!("{prefix}.attn_output_a.weight")).map_err(crate::runtime::compute_error)?;
    if output_a.rows != config.output_groups * config.output_lora_rank || output_a.columns != config.num_heads * config.head_dim / config.output_groups {
        return Err(crate::runtime::compute_error(format!("{} shape=[{},{}] 与 grouped output 不兼容", output_a.name, output_a.rows, output_a.columns)));
    }
    let output_inputs = (0..config.output_groups)
        .map(|group| {
            let part = output_a.row_slice(group * config.output_lora_rank, config.output_lora_rank).map_err(crate::runtime::compute_error)?;
            backend.prepare_weight(LinearWeight::gguf(&part), config.output_lora_rank, output_a.columns)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = matrix("attn_output_b.weight", config.hidden_size, config.output_groups * config.output_lora_rank)?;
    let ratio = config.compress_ratios[layer];
    // V4.1 的 compressor/indexer 跨层共享,GGUF 张量只落在组首层,按存在性装配
    let compressor = (ratio != 0 && source.has_tensor(&format!("{prefix}.attn_compressor_kv.weight"))).then(|| prepare_gguf_compressor(backend, source, &prefix, "attn_compressor", ratio, config.head_dim)).transpose()?;
    let indexer = ((ratio == 4 || ratio == 2 || ratio == 1) && source.has_tensor(&format!("{prefix}.indexer.attn_q_b.weight")))
        .then(|| {
            Ok(PreparedIndexer {
                query: matrix("indexer.attn_q_b.weight", config.index_heads * config.index_head_dim, config.q_lora_rank)?,
                head_weights: matrix("indexer.proj.weight", config.index_heads, config.hidden_size)?,
                key_projection: if source.has_tensor(&format!("{prefix}.indexer.wk.weight")) { Some(matrix("indexer.wk.weight", config.index_head_dim, config.head_dim)?) } else { None },
                key_norm: if source.has_tensor(&format!("{prefix}.indexer.k_norm.weight")) { Some(vector("indexer.k_norm.weight", config.index_head_dim)?) } else { None },
                compressor: if source.has_tensor(&format!("{prefix}.indexer_compressor_kv.weight")) { Some(prepare_gguf_compressor(backend, source, &prefix, "indexer_compressor", ratio, config.index_head_dim)?) } else { None },
            })
        })
        .transpose()?;
    // DeepSeek-V4 router 保留 F32，避免低精度改变 top-k 专家集合。
    let router_weight = prepare_gguf_f32_matrix(backend, source, &format!("{prefix}.ffn_gate_inp.weight"), config.expert_count, config.hidden_size)?;
    let router = if layer < config.hash_layer_count {
        let selected = source
            .tensor_i32(&format!("{prefix}.ffn_gate_tid2eid.weight"))
            .map_err(crate::runtime::compute_error)?
            .into_iter()
            .map(|expert| u32::try_from(expert).ok().filter(|&expert| expert < config.expert_count as u32).ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 token hash expert id {expert} 越界"))))
            .collect::<Result<Vec<_>, _>>()?;
        PreparedRouter::TokenHash { weight: router_weight, zero_bias: backend.prepare_f32(&vec![0.0; config.expert_count], 1, config.expert_count)?, token_to_experts: selected }
    } else {
        PreparedRouter::ScoreTopK {
            weight: router_weight,
            correction_bias: vector("exp_probs_b.bias", config.expert_count)?,
            value_level_bias: if config.router_value_level_bias { Some(vector("exp_probs_b_vl.bias", config.expert_count)?) } else { None },
        }
    };
    let (shared_gate, shared_up) = prepare_gguf_matrix_pair(backend, source, &format!("{prefix}.ffn_gate_shexp.weight"), &format!("{prefix}.ffn_up_shexp.weight"), config.expert_intermediate_size, config.hidden_size)?;
    Ok(DeepSeekV4PreparedLayer {
        attention_hyper_connection: prepare_gguf_hyper_connection(backend, source, &prefix, "attn")?,
        attention_norm: prepare_gguf_dense_vector(backend, source, &format!("{prefix}.attn_norm.weight"), config.hidden_size)?,
        attention: PreparedAttention {
            sink: vector("attn_sinks.weight", config.num_heads)?,
            query_input,
            query_norm,
            query_output,
            query_head_norm: prepare_unit_norm(backend, config.head_dim)?,
            key_value,
            key_norm,
            output_inputs,
            output,
            compressor,
            indexer,
        },
        feedforward: PreparedFeedforward {
            hyper_connection: prepare_gguf_hyper_connection(backend, source, &prefix, "ffn")?,
            norm: prepare_gguf_vector(backend, source, &format!("{prefix}.ffn_norm.weight"), config.hidden_size)?,
            router,
            shared_gate,
            shared_up,
            shared_down: matrix("ffn_down_shexp.weight", config.hidden_size, config.expert_intermediate_size)?,
        },
    })
}

fn prepare_gguf_matrix<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, name: &str, rows: usize, columns: usize) -> Result<B::Weight, BackendError> {
    let matrix = source.matrix(name).map_err(crate::runtime::compute_error)?;
    if matrix.rows != rows || matrix.columns != columns {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 GGUF {name} shape=[{},{}]，期望 [{rows},{columns}]", matrix.rows, matrix.columns)));
    }
    backend.prepare_weight(LinearWeight::gguf(&matrix), rows, columns)
}

fn prepare_gguf_matrix_pair<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, first_name: &str, second_name: &str, rows: usize, columns: usize) -> Result<(B::Weight, B::Weight), BackendError> {
    let first = source.matrix(first_name).map_err(crate::runtime::compute_error)?;
    let second = source.matrix(second_name).map_err(crate::runtime::compute_error)?;
    if first.rows != rows || first.columns != columns || second.rows != rows || second.columns != columns {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 GGUF pair {first_name}=[{},{}] {second_name}=[{},{}]，期望 [{rows},{columns}]", first.rows, first.columns, second.rows, second.columns)));
    }
    backend.prepare_weight_pair(LinearWeight::gguf(&first), LinearWeight::gguf(&second), rows, columns)
}

fn prepare_gguf_dense_matrix<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, name: &str, rows: usize, columns: usize) -> Result<B::Weight, BackendError> {
    let matrix = source.matrix(name).map_err(crate::runtime::compute_error)?;
    if matrix.rows != rows || matrix.columns != columns {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 GGUF {name} shape=[{},{}]，期望 [{rows},{columns}]", matrix.rows, matrix.columns)));
    }
    let values = matrix.decode().map_err(crate::runtime::compute_error)?;
    backend.prepare_weight(LinearWeight::F32(&values), rows, columns)
}

fn prepare_gguf_f32_matrix<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, name: &str, rows: usize, columns: usize) -> Result<B::Weight, BackendError> {
    let matrix = source.matrix(name).map_err(crate::runtime::compute_error)?;
    if matrix.rows != rows || matrix.columns != columns {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 GGUF {name} shape=[{},{}]，期望 [{rows},{columns}]", matrix.rows, matrix.columns)));
    }
    let values = matrix.decode().map_err(crate::runtime::compute_error)?;
    backend.prepare_f32(&values, rows, columns)
}

fn prepare_gguf_vector<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, name: &str, columns: usize) -> Result<B::Weight, BackendError> {
    let values = source.vector_f32(name).map_err(crate::runtime::compute_error)?;
    if values.len() != columns {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 GGUF {name} 长度={}，期望 {columns}", values.len())));
    }
    backend.prepare_f32(&values, 1, columns)
}

fn prepare_gguf_dense_vector<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, name: &str, columns: usize) -> Result<B::Weight, BackendError> {
    let values = source.vector_f32(name).map_err(crate::runtime::compute_error)?;
    if values.len() != columns {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 GGUF {name} 长度={}，期望 {columns}", values.len())));
    }
    backend.prepare_weight(LinearWeight::F32(&values), 1, columns)
}

fn prepare_gguf_compressor<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, prefix: &str, name: &str, ratio: usize, width: usize) -> Result<PreparedCompressor<B::Weight>, BackendError> {
    let channels = if ratio == 4 { 2 * width } else { width };
    let gate_name = format!("{prefix}.{name}_gate.weight");
    let (key_value, gate) = if source.has_tensor(&gate_name) {
        let (key_value, gate) = prepare_gguf_matrix_pair(backend, source, &format!("{prefix}.{name}_kv.weight"), &gate_name, channels, source.config().hidden_size)?;
        (key_value, Some(gate))
    } else {
        // V4.1 ratio=1 纯投影无 gate
        (prepare_gguf_matrix(backend, source, &format!("{prefix}.{name}_kv.weight"), channels, source.config().hidden_size)?, None)
    };
    // V4.1 compressor 无 ape 张量
    Ok(PreparedCompressor {
        position: if source.has_tensor(&format!("{prefix}.{name}_ape.weight")) { Some(prepare_gguf_vector(backend, source, &format!("{prefix}.{name}_ape.weight"), ratio * channels)?) } else { None },
        key_value,
        gate,
        norm: prepare_gguf_dense_vector(backend, source, &format!("{prefix}.{name}_norm.weight"), width)?,
    })
}

fn prepare_gguf_hyper_connection<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, prefix: &str, sublayer: &str) -> Result<PreparedHyperConnection<B::Weight>, BackendError> {
    let config = source.config();
    let mixes = (2 + config.hyper_connection_copies) * config.hyper_connection_copies;
    Ok(PreparedHyperConnection {
        input_norm: prepare_unit_norm(backend, config.hyper_connection_copies * config.hidden_size)?,
        function: prepare_gguf_dense_matrix(backend, source, &format!("{prefix}.hc_{sublayer}_fn.weight"), mixes, config.hyper_connection_copies * config.hidden_size)?,
        base: prepare_gguf_vector(backend, source, &format!("{prefix}.hc_{sublayer}_base.weight"), mixes)?,
        scale: prepare_gguf_vector(backend, source, &format!("{prefix}.hc_{sublayer}_scale.weight"), 3)?,
    })
}

pub fn prepare_deepseek_v4_gguf_output_head<B: Backend>(backend: &B, source: &DeepSeekV4Gguf) -> Result<DeepSeekV4OutputHead<B::Weight>, BackendError> {
    prepare_deepseek_v4_gguf_output_head_quantized(backend, source, crate::weight::LmHeadQuantization::Native)
}

/// 官方 safetensors 输出头（mHC + final norm + dense lm_head），与 GGUF 版共用 `DeepSeekV4OutputHead`。
pub fn prepare_deepseek_v4_official_output_head<B: Backend>(backend: &B, source: &DeepSeekV4Weights) -> Result<DeepSeekV4OutputHead<B::Weight>, BackendError> {
    let config = source.config();
    let head = source.output_head().map_err(crate::runtime::compute_error)?;
    let copies = config.hyper_connection_copies;
    let expanded = copies.checked_mul(config.hidden_size).ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4 output mHC hidden 宽度溢出"))?;
    let norm = tensor_f32_values(&head.norm)?;
    // BF16 走零拷贝字节路径；F16/F32 统一解到 F32，避免 lm_head 再分支出第二种设备格式。
    let lm_head_f32;
    let lm_head = match head.lm_head.dtype.as_str() {
        "BF16" => LinearWeight::Bf16Bytes(&head.lm_head.data),
        _ => {
            lm_head_f32 = tensor_f32_values(&head.lm_head)?;
            LinearWeight::F32(&lm_head_f32)
        }
    };
    Ok(DeepSeekV4OutputHead {
        // V4.1 head 无 hc:forward 末层已折叠成 [rows, hidden],norm 宽度取 hidden;
        // V4 保持展开态 rmsnorm 后再 head_reduce。
        input_norm: prepare_unit_norm(backend, if head.hyper_connection.is_some() { expanded } else { config.hidden_size })?,
        hyper_connection: head.hyper_connection.as_ref().map(|hc| prepare_hyper_connection(backend, hc)).transpose()?,
        output: crate::runtime::output::prepare_output_head_quantized(backend, &norm, lm_head, config.vocab_size, config.hidden_size, crate::weight::LmHeadQuantization::Native)?,
    })
}

fn tensor_f32_values(tensor: &TensorData) -> Result<Vec<f32>, BackendError> {
    let elements = tensor.shape.iter().try_fold(1usize, |value, &dimension| value.checked_mul(dimension)).ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 权重 {} 元素数溢出", tensor.name)))?;
    match tensor.dtype.as_str() {
        "F32" if tensor.data.len() == elements * 4 => Ok(tensor.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 权重块"))).collect()),
        "BF16" if tensor.data.len() == elements * 2 => Ok(tensor.data.chunks_exact(2).map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect()),
        "F16" if tensor.data.len() == elements * 2 => Ok(tensor.data.chunks_exact(2).map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect()),
        _ => Err(crate::runtime::compute_error(format!("DeepSeek-V4 输出权重 {} dtype={} bytes={} 与 shape={:?} 不兼容", tensor.name, tensor.dtype, tensor.data.len(), tensor.shape))),
    }
}

pub fn prepare_deepseek_v4_gguf_output_head_quantized<B: Backend>(backend: &B, source: &DeepSeekV4Gguf, quantization: crate::weight::LmHeadQuantization) -> Result<DeepSeekV4OutputHead<B::Weight>, BackendError> {
    let config = source.config();
    let copies = config.hyper_connection_copies;
    let expanded = copies.checked_mul(config.hidden_size).ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4 output mHC hidden 宽度溢出"))?;
    let final_norm = source.vector_f32("output_norm.weight").map_err(crate::runtime::compute_error)?;
    if final_norm.len() != config.hidden_size {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 output norm 长度={}，期望 {}", final_norm.len(), config.hidden_size)));
    }
    let lm_head = source.matrix("output.weight").map_err(crate::runtime::compute_error)?;
    if lm_head.rows != config.vocab_size || lm_head.columns != config.hidden_size {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 output head shape=[{},{}]，期望 [{},{}]", lm_head.rows, lm_head.columns, config.vocab_size, config.hidden_size)));
    }
    let hyper_connection = if source.has_tensor("output_hc_fn.weight") {
        Some(PreparedHyperConnection {
            input_norm: prepare_unit_norm(backend, expanded)?,
            function: prepare_gguf_dense_matrix(backend, source, "output_hc_fn.weight", copies, expanded)?,
            base: prepare_gguf_vector(backend, source, "output_hc_base.weight", copies)?,
            scale: prepare_gguf_vector(backend, source, "output_hc_scale.weight", 1)?,
        })
    } else {
        None
    };
    Ok(DeepSeekV4OutputHead {
        input_norm: prepare_unit_norm(backend, if hyper_connection.is_some() { expanded } else { config.hidden_size })?,
        hyper_connection,
        output: crate::runtime::output::prepare_output_head_quantized(backend, &final_norm, LinearWeight::gguf(&lm_head), config.vocab_size, config.hidden_size, quantization)?,
    })
}

pub fn deepseek_v4_token_output<B>(backend: &B, config: &DeepSeekV4Config, head: &DeepSeekV4OutputHead<B::Weight>, hidden: &B::Tensor) -> Result<crate::runtime::output::OutputResult<B::Tensor>, BackendError>
where
    B: HyperConnectionKernel,
{
    let reduced = deepseek_v4_output_hidden(backend, config, head, hidden)?;
    crate::runtime::output::last_token_output(
        backend,
        &head.output,
        &reduced,
        backend.token_rows(&reduced) - 1,
        &crate::runtime::output::OutputPlan { eps: config.rms_eps, norm: crate::runtime::output::OutputNorm::Rms, excluded_tokens: Vec::new() },
    )
}

pub(crate) fn deepseek_v4_output_hidden<B>(backend: &B, config: &DeepSeekV4Config, head: &DeepSeekV4OutputHead<B::Weight>, hidden: &B::Tensor) -> Result<B::Tensor, BackendError>
where
    B: HyperConnectionKernel,
{
    let normalized = backend.rmsnorm(hidden, &head.input_norm, config.rms_eps)?;
    match &head.hyper_connection {
        Some(hc) => {
            let mixes = backend.linear(&normalized, &hc.function)?;
            backend.hyper_connection_head_reduce(hidden, &mixes, &hc.base, &hc.scale, config.hyper_connection_copies, config.hyper_connection_eps)
        }
        // TODO(V4.1):head 无 hc 的 reduce 语义待官方 inference 代码确认,当前按 rmsnorm 直通
        None => Ok(normalized),
    }
}

pub fn deepseek_v4_token_outputs<B>(backend: &B, config: &DeepSeekV4Config, head: &DeepSeekV4OutputHead<B::Weight>, hidden: &B::Tensor) -> Result<crate::runtime::output::OutputResult<B::Tensor>, BackendError>
where
    B: HyperConnectionKernel,
{
    let reduced = deepseek_v4_output_hidden(backend, config, head, hidden)?;
    crate::runtime::output::token_output(backend, &head.output, &reduced, &crate::runtime::output::OutputPlan { eps: config.rms_eps, norm: crate::runtime::output::OutputNorm::Rms, excluded_tokens: Vec::new() })
}

pub fn allocate_deepseek_v4_caches<B: CompressedSparseKernel>(backend: &B, model: &DeepSeekV4) -> Result<Vec<B::CompressedKvStorage>, BackendError> {
    (0..model.layer_count()).map(|layer| backend.allocate_compressed_kv(&model.layer_spec(layer).map_err(|error| crate::runtime::compute_error(error.to_string()))?.attention)).collect()
}

#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_gguf_forward_cached<B>(
    backend: &B,
    model: &DeepSeekV4,
    source: &DeepSeekV4Gguf,
    caches: &mut [B::CompressedKvStorage],
    experts: &mut B::PrefillExperts,
    layer_cache: &mut DeepSeekV4LayerCache<B::Weight>,
    input: &B::Tensor,
    rope: &DeepSeekV4RopeTables,
    position: usize,
    token_ids: &[u32],
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + CompressedSparseKernel + HyperConnectionKernel,
    B::Tensor: Clone,
{
    deepseek_v4_forward_impl(backend, model, caches, experts, Some(layer_cache), input, rope, position, token_ids, None::<fn(usize, &[u32], &mut B::Tensor) -> Result<(), BackendError>>, |backend, layer| {
        prepare_deepseek_v4_gguf_layer(backend, source, layer)
    })
}

/// 官方 safetensors 权重（Core FP8 + 路由专家 MXFP4）的 forward 入口，编排与 GGUF 路径共用。
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_official_forward_cached<B>(
    backend: &B,
    model: &DeepSeekV4,
    source: &DeepSeekV4Weights,
    caches: &mut [B::CompressedKvStorage],
    experts: &mut B::PrefillExperts,
    layer_cache: &mut DeepSeekV4LayerCache<B::Weight>,
    input: &B::Tensor,
    rope: &DeepSeekV4RopeTables,
    position: usize,
    token_ids: &[u32],
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + CompressedSparseKernel + HyperConnectionKernel,
    B::Tensor: Clone,
{
    deepseek_v4_forward_impl(backend, model, caches, experts, Some(layer_cache), input, rope, position, token_ids, None::<fn(usize, &[u32], &mut B::Tensor) -> Result<(), BackendError>>, |backend, layer| {
        prepare_deepseek_v4_layer(backend, source, layer)
    })
}

#[allow(clippy::too_many_arguments)]
fn deepseek_v4_forward_impl<B, F, E>(
    backend: &B,
    model: &DeepSeekV4,
    caches: &mut [B::CompressedKvStorage],
    experts: &mut B::PrefillExperts,
    mut layer_cache: Option<&mut DeepSeekV4LayerCache<B::Weight>>,
    input: &B::Tensor,
    rope: &DeepSeekV4RopeTables,
    position: usize,
    token_ids: &[u32],
    mut engram: Option<E>,
    prepare: F,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + CompressedSparseKernel + HyperConnectionKernel,
    B::Tensor: Clone,
    F: Fn(&B, usize) -> Result<DeepSeekV4PreparedLayer<B::Weight>, BackendError>,
    E: FnMut(usize, &[u32], &mut B::Tensor) -> Result<(), BackendError>,
{
    let rows = backend.token_rows(input);
    let cached_layers = layer_cache.as_ref().map_or(model.layer_count(), |cache| cache.layers.len());
    if rows == 0 || rows != token_ids.len() || backend.token_cols(input) != model.config().hidden_size || caches.len() != model.layer_count() || cached_layers != model.layer_count() {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 forward 输入非法: shape=[{},{}] tokens={} caches={} layer_cache={cached_layers}", rows, backend.token_cols(input), token_ids.len(), caches.len())));
    }
    let positions = (position..position.checked_add(rows).ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4 position 溢出"))?).collect::<Vec<_>>();
    let result = (|| {
        let mut hidden = backend.hyper_connection_expand(input, model.config().hyper_connection_copies)?;
        let mut v41_shared = model.is_v41().then(V41SharedState::new);
        let last_layer = model.layer_count() - 1;
        for layer in 0..model.layer_count() {
            let _scope = backend.layer_scope();
            if rows == 1 {
                backend.begin_decode_batch();
            } else {
                backend.begin_batch();
            }
            let spec = model.layer_spec(layer).map_err(|error| crate::runtime::compute_error(error.to_string()))?;
            let prepared = if let Some(cache) = layer_cache.as_deref_mut()
                && cache.resident[layer]
            {
                if cache.layers[layer].is_none() {
                    cache.layers[layer] = Some(prepare(backend, layer)?);
                }
                cache.layers[layer].as_ref().ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 L{layer} 常驻权重未加载")))?
            } else {
                // 非常驻层的临时权重只活在本次层计算内,借用在层循环体内闭合。
                let prepared = prepare(backend, layer)?;
                hidden = deepseek_v4_forward_one_layer(backend, model, spec, &prepared, 0, caches, None, v41_shared.as_mut(), experts, layer, &hidden, rope, &positions, token_ids, layer == last_layer, engram.as_mut())?;
                backend.submit_batch();
                continue;
            };
            hidden = deepseek_v4_forward_one_layer(backend, model, spec, prepared, 0, caches, None, v41_shared.as_mut(), experts, layer, &hidden, rope, &positions, token_ids, layer == last_layer, engram.as_mut())?;
            backend.submit_batch();
        }
        Ok(hidden)
    })();
    backend.finish_batch();
    result
}

/// 单层前向的 V4/V4.1 分派;V4.1 在层入口调用 engram 注入。
#[allow(clippy::too_many_arguments)]
fn deepseek_v4_forward_one_layer<B, E>(
    backend: &B,
    model: &DeepSeekV4,
    spec: &DeepSeekV4LayerSpec,
    prepared: &DeepSeekV4PreparedLayer<B::Weight>,
    layer_start: usize,
    caches: &mut [B::CompressedKvStorage],
    source_cache: Option<&B::CompressedKvStorage>,
    v41_shared: Option<&mut V41SharedState<B>>,
    experts: &mut B::PrefillExperts,
    layer: usize,
    hidden: &B::Tensor,
    rope: &DeepSeekV4RopeTables,
    positions: &[usize],
    token_ids: &[u32],
    collapse_head: bool,
    mut engram: Option<E>,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + CompressedSparseKernel + HyperConnectionKernel,
    B::Tensor: Clone,
    E: FnMut(usize, &[u32], &mut B::Tensor) -> Result<(), BackendError>,
{
    if let Some(shared) = v41_shared {
        let mut hidden = (*hidden).clone();
        if let Some(engram) = engram.as_mut() {
            engram(layer, token_ids, &mut hidden)?;
        }
        deepseek_v4_prefill_layer_v41(backend, model, spec, prepared, layer_start, caches, source_cache, shared, experts, layer, &hidden, rope.layer(spec), positions, token_ids, true, collapse_head)
    } else {
        deepseek_v4_prefill_layer(backend, model.config(), spec, prepared, &mut caches[layer], experts, layer, hidden, rope.layer(spec), positions, token_ids, true, &spec.hyper_connection)
    }
}

fn prepare_attention<B: Backend>(backend: &B, config: &DeepSeekV4Config, weights: &DeepSeekV4AttentionWeights) -> Result<PreparedAttention<B::Weight>, BackendError> {
    Ok(PreparedAttention {
        sink: prepare_tensor(backend, &weights.sink)?,
        query_input: backend.prepare_weight(core_weight(&weights.query.input_projection)?, config.q_lora_rank, config.hidden_size)?,
        query_norm: prepare_dense_tensor(backend, &weights.query.norm)?,
        query_output: backend.prepare_weight(core_weight(&weights.query.output_projection)?, config.num_heads * config.head_dim, config.q_lora_rank)?,
        query_head_norm: prepare_unit_norm(backend, config.head_dim)?,
        key_value: backend.prepare_weight(core_weight(&weights.key_value.projection)?, config.num_kv_heads * config.head_dim, config.hidden_size)?,
        key_norm: prepare_dense_tensor(backend, &weights.key_value.norm)?,
        output_inputs: prepare_grouped_core(backend, &weights.output.input_projection, config.output_groups, config.output_lora_rank)?,
        output: backend.prepare_weight(core_weight(&weights.output.output_projection)?, config.hidden_size, config.output_groups * config.output_lora_rank)?,
        compressor: weights.compressor.as_ref().map(|compressor| prepare_compressor(backend, compressor)).transpose()?,
        indexer: weights
            .indexer
            .as_ref()
            .map(|indexer| {
                Ok(PreparedIndexer {
                    query: backend.prepare_weight(core_weight(&indexer.query_projection)?, config.index_heads * config.index_head_dim, config.q_lora_rank)?,
                    head_weights: prepare_dense_tensor(backend, &indexer.head_weights_projection)?,
                    key_projection: indexer.key_projection.as_ref().map(|wk| prepare_f32_tensor(backend, wk)).transpose()?,
                    key_norm: indexer.key_norm.as_ref().map(|norm| prepare_f32_tensor(backend, norm)).transpose()?,
                    compressor: indexer.compressor.as_ref().map(|compressor| prepare_compressor(backend, compressor)).transpose()?,
                })
            })
            .transpose()?,
    })
}

fn prepare_compressor<B: Backend>(backend: &B, weights: &crate::weight::model::deepseek_v4::DeepSeekV4CompressorWeights) -> Result<PreparedCompressor<B::Weight>, BackendError> {
    Ok(PreparedCompressor {
        position: weights.position.as_ref().map(|position| prepare_tensor(backend, position)).transpose()?,
        key_value: prepare_dense_tensor(backend, &weights.key_value_projection)?,
        gate: weights.gate_projection.as_ref().map(|gate| prepare_dense_tensor(backend, gate)).transpose()?,
        norm: prepare_dense_tensor(backend, &weights.norm)?,
    })
}

fn prepare_hyper_connection<B: Backend>(backend: &B, weights: &DeepSeekV4HyperConnectionWeights) -> Result<PreparedHyperConnection<B::Weight>, BackendError> {
    let hidden = weights.function.shape.get(1).copied().ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 mHC function {} 缺少 hidden 维度", weights.function.name)))?;
    Ok(PreparedHyperConnection { input_norm: prepare_unit_norm(backend, hidden)?, function: prepare_dense_tensor(backend, &weights.function)?, base: prepare_tensor(backend, &weights.base)?, scale: prepare_tensor(backend, &weights.scale)? })
}

fn prepare_unit_norm<B: Backend>(backend: &B, columns: usize) -> Result<B::Weight, BackendError> {
    backend.prepare_weight(LinearWeight::F32(&vec![1.0; columns]), 1, columns)
}

fn prepare_dense_tensor<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let (rows, columns) = match tensor.shape.as_slice() {
        [columns] => (1, *columns),
        [rows, columns] => (*rows, *columns),
        shape => return Err(crate::runtime::compute_error(format!("DeepSeek-V4 dense 权重 {} shape={shape:?} 必须是 rank-1/2", tensor.name))),
    };
    let elements = rows.checked_mul(columns).ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 dense 权重 {} 元素数溢出", tensor.name)))?;
    match tensor.dtype.as_str() {
        // 矩阵权重保持 BF16 原始字节驻留:decode GEMV 读字节减半,kernel 内
        // BF16→F32 解码逐位无损。rank-1 常量(norm 等)继续解 F32,供 constant()
        // 与 RMSNorm 消费。
        "BF16" if tensor.data.len() == elements * 2 => {
            if rows == 1 {
                let values = tensor.data.chunks_exact(2).map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect::<Vec<_>>();
                backend.prepare_f32(&values, rows, columns)
            } else {
                backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, columns)
            }
        }
        "F16" if tensor.data.len() == elements * 2 => {
            let values = tensor.data.chunks_exact(2).map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect::<Vec<_>>();
            backend.prepare_f32(&values, rows, columns)
        }
        "F32" if tensor.data.len() == elements * 4 => {
            let values = tensor.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 权重块"))).collect::<Vec<_>>();
            backend.prepare_weight(LinearWeight::F32(&values), rows, columns)
        }
        _ => Err(crate::runtime::compute_error(format!("DeepSeek-V4 dense 权重 {} dtype={} bytes={} 与 shape={:?} 不兼容", tensor.name, tensor.dtype, tensor.data.len(), tensor.shape))),
    }
}

fn prepare_tensor<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let (rows, columns) = match tensor.shape.as_slice() {
        [columns] => (1, *columns),
        [rows, columns] => (*rows, *columns),
        shape => return Err(crate::runtime::compute_error(format!("DeepSeek-V4 权重 {} shape={shape:?} 必须是 rank-1/2", tensor.name))),
    };
    let elements = rows.checked_mul(columns).ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 权重 {} 元素数溢出", tensor.name)))?;
    match tensor.dtype.as_str() {
        "BF16" if tensor.data.len() == elements * 2 => backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, columns),
        "F16" if tensor.data.len() == elements * 2 => {
            let values = tensor.data.chunks_exact(2).map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            backend.prepare_weight(LinearWeight::F16(&values), rows, columns)
        }
        "F32" if tensor.data.len() == elements * 4 => {
            let values = tensor.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 权重块"))).collect::<Vec<_>>();
            backend.prepare_f32(&values, rows, columns)
        }
        _ => Err(crate::runtime::compute_error(format!("DeepSeek-V4 权重 {} dtype={} bytes={} 与 shape={:?} 不兼容", tensor.name, tensor.dtype, tensor.data.len(), tensor.shape,))),
    }
}

fn prepare_f32_tensor<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let (rows, columns) = match tensor.shape.as_slice() {
        [columns] => (1, *columns),
        [rows, columns] => (*rows, *columns),
        shape => return Err(crate::runtime::compute_error(format!("DeepSeek-V4 F32 控制权重 {} shape={shape:?} 必须是 rank-1/2", tensor.name))),
    };
    let elements = rows.checked_mul(columns).ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 F32 控制权重 {} 元素数溢出", tensor.name)))?;
    let values: Vec<f32> = match tensor.dtype.as_str() {
        "BF16" if tensor.data.len() == elements * 2 => tensor.data.chunks_exact(2).map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect(),
        "F16" if tensor.data.len() == elements * 2 => tensor.data.chunks_exact(2).map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect(),
        "F32" if tensor.data.len() == elements * 4 => tensor.data.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 权重块"))).collect(),
        _ => return Err(crate::runtime::compute_error(format!("DeepSeek-V4 F32 控制权重 {} dtype={} bytes={} 与 shape={:?} 不兼容", tensor.name, tensor.dtype, tensor.data.len(), tensor.shape))),
    };
    backend.prepare_f32(&values, rows, columns)
}

fn tensor_i32(tensor: &TensorData) -> Result<Vec<i32>, BackendError> {
    let elements = tensor.shape.iter().try_fold(1usize, |value, &dimension| value.checked_mul(dimension)).ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 权重 {} 元素数溢出", tensor.name)))?;
    // 官方 tid2eid 是 I64；专家 id 值域远小于 i32，窄化失败说明数据本身异常。
    match tensor.dtype.as_str() {
        "I32" if tensor.data.len() == elements * 4 => Ok(tensor.data.chunks_exact(4).map(|bytes| i32::from_le_bytes(bytes.try_into().expect("I32 权重块"))).collect()),
        "I64" if tensor.data.len() == elements * 8 => tensor
            .data
            .chunks_exact(8)
            .map(|bytes| i32::try_from(i64::from_le_bytes(bytes.try_into().expect("I64 权重块"))).map_err(|_| crate::runtime::compute_error(format!("DeepSeek-V4 权重 {} 含超出 i32 的条目", tensor.name))))
            .collect(),
        _ => Err(crate::runtime::compute_error(format!("DeepSeek-V4 权重 {} dtype={} bytes={} 与 I32/I64 shape={:?} 不兼容", tensor.name, tensor.dtype, tensor.data.len(), tensor.shape))),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_prefill_layer<B>(
    backend: &B,
    config: &DeepSeekV4Config,
    spec: &DeepSeekV4LayerSpec,
    weights: &DeepSeekV4PreparedLayer<B::Weight>,
    cache: &mut B::CompressedKvStorage,
    experts: &mut B::PrefillExperts,
    layer: usize,
    hidden: &B::Tensor,
    rope: DeepSeekV4Rope<'_>,
    positions: &[usize],
    token_ids: &[u32],
    causal_batch: bool,
    hyper_connection: &crate::attention::hyper_connection::HyperConnectionSpec,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + crate::attention::compressed_sparse::CompressedSparseKernel + crate::attention::hyper_connection::HyperConnectionKernel,
{
    let rows = backend.token_rows(hidden);
    let expected_columns = config.hidden_size.checked_mul(hyper_connection.copies).ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4 mHC hidden 宽度溢出"))?;
    if rows == 0 || backend.token_cols(hidden) != expected_columns || positions.len() != rows || token_ids.len() != rows || positions.windows(2).any(|pair| pair[1] != pair[0] + 1) {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 L{layer} 输入 shape=[{},{}] positions={} tokens={} expected_cols={expected_columns}", rows, backend.token_cols(hidden), positions.len(), token_ids.len(),)));
    }
    backend.profile_device_operator("attention_hc")?;
    let hidden = hyper_connection_sublayer(backend, hidden, &weights.attention_hyper_connection, &weights.attention_norm, hyper_connection, config.rms_eps, false, "attention_merge", |input| {
        backend.profile_device_operator("attn_qkv")?;
        attention_prefill(backend, config, spec, &weights.attention, cache, input, rope, positions, causal_batch)
    })?;
    backend.profile_device_operator("ffn_hc")?;
    let (moe_label, merge_label) = match spec.routing {
        DeepSeekV4RoutingSelection::TokenHash => ("moe_hash", "ffn_merge_hash"),
        DeepSeekV4RoutingSelection::ScoreTopK => ("moe_score", "ffn_merge_score"),
    };
    hyper_connection_sublayer(backend, &hidden, &weights.feedforward.hyper_connection, &weights.feedforward.norm, hyper_connection, config.rms_eps, true, merge_label, |input| {
        backend.profile_device_operator(moe_label)?;
        feedforward_prefill(backend, spec, &weights.feedforward, experts, layer, input, token_ids, config.image_token_id)
    })
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(crate) struct DeepSeekV4PrefillSegment<'a, C> {
    pub cache: &'a mut C,
    pub rope: DeepSeekV4Rope<'a>,
    pub positions: &'a [usize],
    pub token_ids: &'a [u32],
    pub causal_batch: bool,
}

/// CSA/KV 状态按 session 独立推进，逐行算子与专家 GEMM 共享同一连续 batch。
#[allow(clippy::too_many_arguments)]
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(crate) fn deepseek_v4_prefill_layer_segmented<B>(
    backend: &B,
    config: &DeepSeekV4Config,
    spec: &DeepSeekV4LayerSpec,
    weights: &DeepSeekV4PreparedLayer<B::Weight>,
    experts: &mut B::PrefillExperts,
    layer: usize,
    hidden: &B::Tensor,
    segments: &mut [DeepSeekV4PrefillSegment<'_, B::CompressedKvStorage>],
    hyper_connection: &crate::attention::hyper_connection::HyperConnectionSpec,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + crate::attention::compressed_sparse::CompressedSparseKernel + crate::attention::hyper_connection::HyperConnectionKernel + crate::backend::SegmentedTensorBackend,
{
    let rows = backend.token_rows(hidden);
    let segment_rows = segments.iter().map(|segment| segment.token_ids.len()).sum::<usize>();
    let expected_columns = config.hidden_size.checked_mul(hyper_connection.copies).ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4 segmented mHC hidden 宽度溢出"))?;
    if rows == 0 || rows != segment_rows || backend.token_cols(hidden) != expected_columns {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 L{layer} segmented 输入 shape=[{},{}] segment_rows={segment_rows} expected_cols={expected_columns}", rows, backend.token_cols(hidden))));
    }
    for segment in segments.iter() {
        if segment.positions.len() != segment.token_ids.len() || segment.positions.is_empty() || segment.positions.windows(2).any(|pair| pair[1] != pair[0] + 1) {
            return Err(crate::runtime::compute_error(format!("DeepSeek-V4 L{layer} segmented positions={} tokens={}", segment.positions.len(), segment.token_ids.len())));
        }
    }

    backend.profile_device_operator("attention_hc")?;
    let attention_hc = &weights.attention_hyper_connection;
    let normalized_hidden = backend.rmsnorm(hidden, &attention_hc.input_norm, config.rms_eps)?;
    let mixes = backend.linear(&normalized_hidden, &attention_hc.function)?;
    let prepared = backend.hyper_connection_prepare_sublayer(hidden, &mixes, &attention_hc.base, &attention_hc.scale, hyper_connection)?;
    let normalized = backend.rmsnorm(&prepared.reduced, &weights.attention_norm, config.rms_eps)?;
    backend.profile_device_operator("attn_qkv")?;
    let attended = attention_prefill_segmented(backend, config, spec, &weights.attention, &normalized, segments)?;
    backend.profile_device_operator("attention_merge")?;
    let hidden = backend.hyper_connection_expand_scaled_add(&attended, &prepared.post, &prepared.residual, hyper_connection.copies)?;

    backend.profile_device_operator("ffn_hc")?;
    let feedforward_hc = &weights.feedforward.hyper_connection;
    let normalized_hidden = backend.rmsnorm(&hidden, &feedforward_hc.input_norm, config.rms_eps)?;
    let mixes = backend.linear(&normalized_hidden, &feedforward_hc.function)?;
    let prepared = backend.hyper_connection_prepare_sublayer(&hidden, &mixes, &feedforward_hc.base, &feedforward_hc.scale, hyper_connection)?;
    let normalized = backend.rmsnorm_f32(&prepared.reduced, &weights.feedforward.norm, config.rms_eps)?;
    let token_ids = segments.iter().flat_map(|segment| segment.token_ids.iter().copied()).collect::<Vec<_>>();
    backend.profile_device_operator("moe")?;
    let output = feedforward_prefill(backend, spec, &weights.feedforward, experts, layer, &normalized, &token_ids, config.image_token_id)?;
    backend.profile_device_operator("ffn_merge")?;
    backend.hyper_connection_expand_scaled_add(&output, &prepared.post, &prepared.residual, hyper_connection.copies)
}

#[allow(clippy::too_many_arguments)]
fn hyper_connection_sublayer<B, F>(
    backend: &B,
    hidden: &B::Tensor,
    weights: &PreparedHyperConnection<B::Weight>,
    norm: &B::Weight,
    spec: &crate::attention::hyper_connection::HyperConnectionSpec,
    rms_eps: f32,
    norm_f32: bool,
    merge_label: &'static str,
    sublayer: F,
) -> Result<B::Tensor, BackendError>
where
    B: crate::attention::hyper_connection::HyperConnectionKernel,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    let normalized_hidden = backend.rmsnorm(hidden, &weights.input_norm, rms_eps)?;
    let mixes = backend.linear(&normalized_hidden, &weights.function)?;
    let prepared = backend.hyper_connection_prepare_sublayer(hidden, &mixes, &weights.base, &weights.scale, spec)?;
    let normalized = if norm_f32 { backend.rmsnorm_f32(&prepared.reduced, norm, rms_eps)? } else { backend.rmsnorm(&prepared.reduced, norm, rms_eps)? };
    let output = sublayer(&normalized)?;
    backend.profile_device_operator(merge_label)?;
    backend.hyper_connection_expand_scaled_add(&output, &prepared.post, &prepared.residual, spec.copies)
}

#[allow(clippy::too_many_arguments)]
fn attention_prefill<B>(
    backend: &B,
    config: &DeepSeekV4Config,
    spec: &DeepSeekV4LayerSpec,
    weights: &PreparedAttention<B::Weight>,
    cache: &mut B::CompressedKvStorage,
    input: &B::Tensor,
    rope: DeepSeekV4Rope<'_>,
    positions: &[usize],
    causal_batch: bool,
) -> Result<B::Tensor, BackendError>
where
    B: crate::attention::compressed_sparse::CompressedSparseKernel,
{
    // query 低秩入口与 KV 投影独立读取同一 hidden；backend 可合并成一次提交。
    let (query_lora, key_value) = backend.dual_linear(input, &weights.query_input, &weights.key_value)?;
    let query_lora = backend.rmsnorm(&query_lora, &weights.query_norm, config.rms_eps)?;
    let query = backend.linear(&query_lora, &weights.query_output)?;
    let query = backend.compressed_rmsnorm_heads(&query, &weights.query_head_norm, config.num_heads, config.head_dim, config.rms_eps)?;
    let query = backend.rope(&query, config.num_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, &rope.table.sin)?;
    let key_value = backend.rmsnorm(&key_value, &weights.key_norm, config.rms_eps)?;
    let key_value = backend.rope(&key_value, config.num_kv_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, &rope.table.sin)?;

    backend.profile_device_operator("attn_compress")?;
    let compressed = match (spec.attention.compression, weights.compressor.as_ref()) {
        (Some(compression), Some(compressor)) => {
            let compressor_gate = compressor.gate.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 ratio=1 compressor(无 gate)的执行路径尚未接入"))?;
            let (kv, gate) = backend.dual_linear(input, &compressor.key_value, compressor_gate)?;
            let position = compressor.position.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 compressor 无 ape 的执行路径尚未接入"))?;
            Some(backend.compress_gated(cache, CompressionStream::Attention, positions, &kv, &gate, position, &compressor.norm, compression, config.head_dim, config.qk_rope_head_dim, &rope.table.cos, &rope.table.sin, config.rms_eps)?)
        }
        (None, None) => None,
        _ => return Err(crate::runtime::compute_error("DeepSeek-V4 compressor 规格与权重不一致")),
    };

    let mut index_query = None;
    let mut index_head_weights = None;
    let mut compressed_index = None;
    backend.profile_device_operator("attn_index")?;
    match (spec.attention.compression.map(|compression| compression.selection), weights.indexer.as_ref()) {
        (Some(crate::attention::compressed_sparse::CompressedSelection::LearnedIndexer(indexer_spec)), Some(indexer)) => {
            let query = backend.linear(&query_lora, &indexer.query)?;
            index_query = Some(backend.rope(&query, indexer_spec.num_heads, indexer_spec.rope_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, &rope.table.sin)?);
            index_head_weights = Some(backend.linear(input, &indexer.head_weights)?);
            let indexer_compressor = indexer.compressor.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 indexer 无 compressor(跨层共享 K)的执行路径尚未接入"))?;
            let indexer_position = indexer_compressor.position.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 compressor 无 ape 的执行路径尚未接入"))?;
            let indexer_gate = indexer_compressor.gate.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 ratio=1 compressor(无 gate)的执行路径尚未接入"))?;
            let (kv, gate) = backend.dual_linear(input, &indexer_compressor.key_value, indexer_gate)?;
            compressed_index = Some(backend.compress_gated(
                cache,
                CompressionStream::Indexer,
                positions,
                &kv,
                &gate,
                indexer_position,
                &indexer_compressor.norm,
                spec.attention.compression.expect("indexer 必须有 compression"),
                indexer_spec.head_dim,
                indexer_spec.rope_dim,
                &rope.table.cos,
                &rope.table.sin,
                config.rms_eps,
            )?);
        }
        (Some(crate::attention::compressed_sparse::CompressedSelection::LearnedIndexer(_)), None) | (_, Some(_)) => return Err(crate::runtime::compute_error("DeepSeek-V4 indexer 规格与权重不一致")),
        _ => {}
    }
    if let (Some(compressed), Some(compressed_index)) = (&compressed, &compressed_index)
        && compressed.visible_positions != compressed_index.visible_positions
    {
        return Err(crate::runtime::compute_error("DeepSeek-V4 attention/indexer compressor 窗口状态不一致"));
    }
    backend.profile_device_operator("attn_csa")?;
    let attended = backend.compressed_sparse_prefill(
        cache,
        positions,
        causal_batch,
        &query,
        &key_value,
        &key_value,
        compressed.as_ref().map(|batch| batch.visible_positions.as_slice()),
        compressed.as_ref().map(|batch| &batch.values),
        compressed.as_ref().map(|batch| &batch.values),
        compressed_index.as_ref().map(|batch| &batch.values),
        index_query.as_ref(),
        index_head_weights.as_ref(),
        Some(&weights.sink),
        &spec.attention,
    )?;
    backend.profile_device_operator("attn_out")?;
    let attended = backend.rope(&attended, config.num_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, rope.inverse_sin)?;
    let projected = backend.grouped_linear_columns(&attended, &weights.output_inputs)?;
    backend.linear(&projected, &weights.output)
}

/// V4.1 chunk 内跨层共享的压缩状态:源层(kv_source)发布本批压缩项与 index key,
/// index 层发布逐 query 行 selection,组内其余层只读消费。每个 prefill chunk /
/// decode step 开始时重建。
pub struct V41SharedState<B: CompressedSparseKernel> {
    attention: Option<CompressedBatch<B::Tensor>>,
    index_key: Option<CompressedBatch<B::Tensor>>,
    selection: Option<B::SharedSelection>,
}

impl<B: CompressedSparseKernel> V41SharedState<B> {
    pub fn new() -> Self {
        Self { attention: None, index_key: None, selection: None }
    }

    /// stage 边界只传递 backend-native selection；压缩 KV 已写入会话级 cache 表。
    pub fn with_selection(selection: Option<B::SharedSelection>) -> Self {
        Self { attention: None, index_key: None, selection }
    }

    pub fn into_selection(self) -> Option<B::SharedSelection> {
        self.selection
    }
}

impl<B: CompressedSparseKernel> Default for V41SharedState<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// V4.1 压缩层 attention:滑窗 KV 归本层,压缩历史组内共享,selection 由
/// index 层发布、其余层复用;候选块粗筛只作用于 candidate_source 之后的层。
/// `cache` 为本层存储(滑窗+压缩写);`source_cache` 为组源层引用——非 source 层
/// 必传(同 stage 切片或跨 stage 的全局表项,kernel 侧按 P2P 读其 buffer)。
#[allow(clippy::too_many_arguments)]
fn attention_prefill_v41<B>(
    backend: &B,
    model: &DeepSeekV4,
    spec: &DeepSeekV4LayerSpec,
    weights: &PreparedAttention<B::Weight>,
    cache: &mut B::CompressedKvStorage,
    source_cache: Option<&B::CompressedKvStorage>,
    shared: &mut V41SharedState<B>,
    layer: usize,
    input: &B::Tensor,
    rope: DeepSeekV4Rope<'_>,
    positions: &[usize],
    causal_batch: bool,
) -> Result<B::Tensor, BackendError>
where
    B: CompressedSparseKernel,
{
    let config = model.config();
    let compression = spec.attention.compression.ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4.1 L{layer} 缺少 compression 规格")))?;
    let indexer_spec = match compression.selection {
        CompressedSelection::LearnedIndexer(dsa) => Some(dsa),
        CompressedSelection::All => None,
    };
    let is_source = model.is_kv_source(layer);
    let is_index = model.is_index_layer(layer);
    if !is_source && source_cache.is_none() {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4.1 L{layer} 缺少组源层 cache(非 source 层必传 source_cache)")));
    }

    let (query_lora, key_value) = backend.dual_linear(input, &weights.query_input, &weights.key_value)?;
    let query_lora = backend.rmsnorm(&query_lora, &weights.query_norm, config.rms_eps)?;
    let query = backend.linear(&query_lora, &weights.query_output)?;
    let query = backend.compressed_rmsnorm_heads(&query, &weights.query_head_norm, config.num_heads, config.head_dim, config.rms_eps)?;
    let query = backend.rope(&query, config.num_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, &rope.table.sin)?;
    let key_value = backend.rmsnorm(&key_value, &weights.key_norm, config.rms_eps)?;
    let key_value = backend.rope(&key_value, config.num_kv_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, &rope.table.sin)?;

    if is_source {
        let compressor = weights.compressor.as_ref().ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4.1 L{layer} 源层缺少 compressor 权重")))?;
        let (pooled_input, gate) = match compressor.gate.as_ref() {
            Some(gate_weight) => backend.dual_linear(input, &compressor.key_value, gate_weight).map(|(kv, gate)| (kv, Some(gate)))?,
            None => (backend.linear(input, &compressor.key_value)?, None),
        };
        // 组首层(kv_source)拥有 wk/k_norm,从池化 latent 产 index key。
        let index_projection = weights.indexer.as_ref().and_then(|indexer| match (&indexer.key_projection, &indexer.key_norm) {
            (Some(wk), Some(k_norm)) => Some((wk, k_norm)),
            _ => None,
        });
        let index_rope_dim = indexer_spec.map_or(config.qk_rope_head_dim, |dsa| dsa.rope_dim);
        let compressed = backend.compress_v41(
            cache,
            positions,
            &pooled_input,
            gate.as_ref(),
            &compressor.norm,
            index_projection,
            compression,
            config.head_dim,
            config.qk_rope_head_dim,
            index_rope_dim,
            &rope.table.cos,
            &rope.table.sin,
            config.rms_eps,
        )?;
        shared.attention = Some(compressed.attention);
        shared.index_key = compressed.index_key;
    }

    let (index_query, index_head_weights) = if is_index && indexer_spec.is_some() {
        let indexer = weights.indexer.as_ref().ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4.1 L{layer} index 层缺少 indexer 权重")))?;
        let dsa = indexer_spec.expect("上方已校验");
        let index_query = backend.linear(&query_lora, &indexer.query)?;
        let index_query = backend.rope(&index_query, dsa.num_heads, dsa.rope_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, &rope.table.sin)?;
        let head_weights = backend.linear(input, &indexer.head_weights)?;
        (Some(index_query), Some(head_weights))
    } else {
        (None, None)
    };

    let batch =
        shared.attention.as_ref().map(|attention| SharedCompressedBatch { visible_positions: &attention.visible_positions, key: &attention.values, value: &attention.values, index_key: shared.index_key.as_ref().map(|index| &index.values) });
    let preset = (!is_index).then(|| shared.selection.as_ref()).flatten();
    let output = if is_source {
        backend.compressed_sparse_prefill_shared(
            cache,
            None,
            positions,
            causal_batch,
            &query,
            &key_value,
            &key_value,
            batch,
            index_query.as_ref(),
            index_head_weights.as_ref(),
            preset,
            model.candidate_spec_for(layer),
            Some(&weights.sink),
            &spec.attention,
        )?
    } else {
        backend.compressed_sparse_prefill_shared(
            cache,
            source_cache,
            positions,
            causal_batch,
            &query,
            &key_value,
            &key_value,
            batch,
            index_query.as_ref(),
            index_head_weights.as_ref(),
            preset,
            model.candidate_spec_for(layer),
            Some(&weights.sink),
            &spec.attention,
        )?
    };
    if let Some(selection) = output.selection {
        shared.selection = Some(selection);
    }
    backend.profile_device_operator("attn_out")?;
    let attended = backend.rope(&output.attended, config.num_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, rope.inverse_sin)?;
    let projected = backend.grouped_linear_columns(&attended, &weights.output_inputs)?;
    backend.linear(&projected, &weights.output)
}

/// `deepseek_v4_prefill_layer` 的 V4.1 形态:attention 走共享路径,ffn/mHC 不变。
/// `collapse_head` 为 true(最后一层)时,ffn 后用其 pre_mix 把展开态 hidden 折叠成
/// `[rows, hidden]`——官方 head 无 hc 系数,reduce 复用最后 ffn 的折叠系数。
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_prefill_layer_v41<B>(
    backend: &B,
    model: &DeepSeekV4,
    spec: &DeepSeekV4LayerSpec,
    weights: &DeepSeekV4PreparedLayer<B::Weight>,
    layer_start: usize,
    caches: &mut [B::CompressedKvStorage],
    source_cache: Option<&B::CompressedKvStorage>,
    shared: &mut V41SharedState<B>,
    experts: &mut B::PrefillExperts,
    layer: usize,
    hidden: &B::Tensor,
    rope: DeepSeekV4Rope<'_>,
    positions: &[usize],
    token_ids: &[u32],
    causal_batch: bool,
    collapse_head: bool,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + CompressedSparseKernel + HyperConnectionKernel,
{
    deepseek_v4_prefill_layer_v41_chained(backend, model, spec, weights, layer_start, caches, source_cache, shared, experts, layer, hidden, None, rope, positions, token_ids, causal_batch, collapse_head).map(|(hidden, _)| hidden)
}

/// 官方 mHC 语义：当前 attention 使用前一层 FFN 发布的 pre，当前 FFN 使用
/// 本层 attention 发布的 pre；本层 FFN 的 pre 随 hidden 交给下一层。
#[allow(clippy::too_many_arguments)]
pub(crate) fn deepseek_v4_prefill_layer_v41_chained<B>(
    backend: &B,
    model: &DeepSeekV4,
    spec: &DeepSeekV4LayerSpec,
    weights: &DeepSeekV4PreparedLayer<B::Weight>,
    layer_start: usize,
    caches: &mut [B::CompressedKvStorage],
    source_cache: Option<&B::CompressedKvStorage>,
    shared: &mut V41SharedState<B>,
    experts: &mut B::PrefillExperts,
    layer: usize,
    hidden: &B::Tensor,
    incoming_pre: Option<&B::Tensor>,
    rope: DeepSeekV4Rope<'_>,
    positions: &[usize],
    token_ids: &[u32],
    causal_batch: bool,
    collapse_head: bool,
) -> Result<(B::Tensor, B::Tensor), BackendError>
where
    B: ExpertPrefillBackend + CompressedSparseKernel + HyperConnectionKernel,
{
    let config = model.config();
    let rows = backend.token_rows(hidden);
    let expected_columns = config.hidden_size.checked_mul(spec.hyper_connection.copies).ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 mHC hidden 宽度溢出"))?;
    if rows == 0 || backend.token_cols(hidden) != expected_columns || positions.len() != rows || token_ids.len() != rows {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4.1 L{layer} 输入 shape=[{},{}] positions={} tokens={}", rows, backend.token_cols(hidden), positions.len(), token_ids.len())));
    }
    backend.profile_device_operator("attention_hc")?;
    let attention_hc = &weights.attention_hyper_connection;
    let normalized_hidden = backend.rmsnorm(hidden, &attention_hc.input_norm, config.rms_eps)?;
    let attention_mixes = backend.linear(&normalized_hidden, &attention_hc.function)?;
    let attention_split = backend.hyper_connection_split(&attention_mixes, &attention_hc.base, &attention_hc.scale, &spec.hyper_connection)?;
    let attention_residual = backend.hyper_connection_mix(hidden, &attention_split.combination, &spec.hyper_connection)?;
    let attention_input = backend.hyper_connection_reduce(hidden, incoming_pre.unwrap_or(&attention_split.pre), spec.hyper_connection.copies)?;
    let attention_input = backend.rmsnorm(&attention_input, &weights.attention_norm, config.rms_eps)?;
    backend.profile_device_operator("attn_qkv")?;
    let local = layer.checked_sub(layer_start).ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4.1 L{layer} 在 stage 起始 {layer_start} 之前")))?;
    let (head, tail) = caches.split_at_mut(local);
    let cache = tail.first_mut().ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4.1 L{layer} 超出本 stage cache 范围")))?;
    let attended = if spec.attention.compression.is_some() {
        // 外部 source_cache(跨 stage 全局表)优先;缺省时回退本 stage 切片(单 stage 形态)
        let source = source_cache.or_else(|| model.kv_source_of(layer).and_then(|source| source.checked_sub(layer_start)).and_then(|local| head.get(local)).map(|storage| storage as &B::CompressedKvStorage));
        attention_prefill_v41(backend, model, spec, &weights.attention, cache, source, shared, layer, &attention_input, rope, positions, causal_batch)?
    } else {
        attention_prefill(backend, config, spec, &weights.attention, cache, &attention_input, rope, positions, causal_batch)?
    };
    backend.profile_device_operator("attention_merge")?;
    let hidden = backend.hyper_connection_expand_scaled_add(&attended, &attention_split.post, &attention_residual, spec.hyper_connection.copies)?;
    backend.profile_device_operator("ffn_hc")?;
    let (moe_label, merge_label) = match spec.routing {
        DeepSeekV4RoutingSelection::TokenHash => ("moe_hash", "ffn_merge_hash"),
        DeepSeekV4RoutingSelection::ScoreTopK => ("moe_score", "ffn_merge_score"),
    };
    // FFN 使用 attention 发布的 pre；本层 FFN pre 留给下一层或最终 head。
    let feedforward_hc = &weights.feedforward.hyper_connection;
    let normalized_hidden = backend.rmsnorm(&hidden, &feedforward_hc.input_norm, config.rms_eps)?;
    let mixes = backend.linear(&normalized_hidden, &feedforward_hc.function)?;
    let feedforward_split = backend.hyper_connection_split(&mixes, &feedforward_hc.base, &feedforward_hc.scale, &spec.hyper_connection)?;
    let feedforward_residual = backend.hyper_connection_mix(&hidden, &feedforward_split.combination, &spec.hyper_connection)?;
    let feedforward_input = backend.hyper_connection_reduce(&hidden, &attention_split.pre, spec.hyper_connection.copies)?;
    let normalized = backend.rmsnorm_f32(&feedforward_input, &weights.feedforward.norm, config.rms_eps)?;
    backend.profile_device_operator(moe_label)?;
    let output = feedforward_prefill(backend, spec, &weights.feedforward, experts, layer, &normalized, token_ids, config.image_token_id)?;
    backend.profile_device_operator(merge_label)?;
    let expanded = backend.hyper_connection_expand_scaled_add(&output, &feedforward_split.post, &feedforward_residual, spec.hyper_connection.copies)?;
    if collapse_head {
        backend.profile_device_operator("head_reduce")?;
        let collapsed = backend.hyper_connection_reduce(&expanded, &feedforward_split.pre, spec.hyper_connection.copies)?;
        Ok((collapsed, feedforward_split.pre))
    } else {
        Ok((expanded, feedforward_split.pre))
    }
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn attention_prefill_segmented<B>(
    backend: &B,
    config: &DeepSeekV4Config,
    spec: &DeepSeekV4LayerSpec,
    weights: &PreparedAttention<B::Weight>,
    input: &B::Tensor,
    segments: &mut [DeepSeekV4PrefillSegment<'_, B::CompressedKvStorage>],
) -> Result<B::Tensor, BackendError>
where
    B: crate::attention::compressed_sparse::CompressedSparseKernel + crate::backend::SegmentedTensorBackend,
{
    let rope = segments.first().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4 segmented attention 不能为空"))?.rope;
    let token_segments = segments.iter().map(|segment| crate::backend::TokenSegment { position: segment.positions[0], rows: segment.positions.len() }).collect::<Vec<_>>();
    let (query_lora, key_value) = backend.dual_linear(input, &weights.query_input, &weights.key_value)?;
    let query_lora = backend.rmsnorm(&query_lora, &weights.query_norm, config.rms_eps)?;
    let query = backend.linear(&query_lora, &weights.query_output)?;
    let query = backend.compressed_rmsnorm_heads(&query, &weights.query_head_norm, config.num_heads, config.head_dim, config.rms_eps)?;
    let query = backend.rope_segmented(&query, config.num_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, false, &token_segments, &rope.table.cos, &rope.table.sin)?;
    let key_value = backend.rmsnorm(&key_value, &weights.key_norm, config.rms_eps)?;
    let key_value = backend.rope_segmented(&key_value, config.num_kv_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, false, &token_segments, &rope.table.cos, &rope.table.sin)?;

    backend.profile_device_operator("attn_compress")?;
    let compressed = match (spec.attention.compression, weights.compressor.as_ref()) {
        (Some(compression), Some(compressor)) => {
            let compressor_gate = compressor.gate.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 ratio=1 compressor(无 gate)的执行路径尚未接入"))?;
            let (kv, gate) = backend.dual_linear(input, &compressor.key_value, compressor_gate)?;
            let position = compressor.position.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 compressor 无 ape 的执行路径尚未接入"))?;
            let mut state_segments = segments.iter_mut().map(|segment| crate::attention::compressed_sparse::CompressedGatedSegment { storage: &mut *segment.cache, positions: segment.positions }).collect::<Vec<_>>();
            Some(backend.compress_gated_segmented(
                &mut state_segments,
                CompressionStream::Attention,
                &kv,
                &gate,
                position,
                &compressor.norm,
                compression,
                config.head_dim,
                config.qk_rope_head_dim,
                &rope.table.cos,
                &rope.table.sin,
                config.rms_eps,
            )?)
        }
        (None, None) => None,
        _ => return Err(crate::runtime::compute_error("DeepSeek-V4 compressor 规格与权重不一致")),
    };

    backend.profile_device_operator("attn_index")?;
    let mut index_query = None;
    let mut index_head_weights = None;
    let mut compressed_index = None;
    match (spec.attention.compression.map(|compression| compression.selection), weights.indexer.as_ref()) {
        (Some(crate::attention::compressed_sparse::CompressedSelection::LearnedIndexer(indexer_spec)), Some(indexer)) => {
            let query = backend.linear(&query_lora, &indexer.query)?;
            index_query = Some(backend.rope_segmented(&query, indexer_spec.num_heads, indexer_spec.rope_dim, RotaryLayout::Interleaved, false, &token_segments, &rope.table.cos, &rope.table.sin)?);
            index_head_weights = Some(backend.linear(input, &indexer.head_weights)?);
            let indexer_compressor = indexer.compressor.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 indexer 无 compressor(跨层共享 K)的执行路径尚未接入"))?;
            let indexer_position = indexer_compressor.position.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 compressor 无 ape 的执行路径尚未接入"))?;
            let indexer_gate = indexer_compressor.gate.as_ref().ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4.1 ratio=1 compressor(无 gate)的执行路径尚未接入"))?;
            let (kv, gate) = backend.dual_linear(input, &indexer_compressor.key_value, indexer_gate)?;
            let mut state_segments = segments.iter_mut().map(|segment| crate::attention::compressed_sparse::CompressedGatedSegment { storage: &mut *segment.cache, positions: segment.positions }).collect::<Vec<_>>();
            compressed_index = Some(backend.compress_gated_segmented(
                &mut state_segments,
                CompressionStream::Indexer,
                &kv,
                &gate,
                indexer_position,
                &indexer_compressor.norm,
                spec.attention.compression.expect("indexer 必须有 compression"),
                indexer_spec.head_dim,
                indexer_spec.rope_dim,
                &rope.table.cos,
                &rope.table.sin,
                config.rms_eps,
            )?);
        }
        (Some(crate::attention::compressed_sparse::CompressedSelection::LearnedIndexer(_)), None) | (_, Some(_)) => return Err(crate::runtime::compute_error("DeepSeek-V4 indexer 规格与权重不一致")),
        _ => {}
    }
    if let (Some(attention), Some(indexer)) = (&compressed, &compressed_index)
        && attention.iter().zip(indexer).any(|(attention, indexer)| attention.visible_positions != indexer.visible_positions)
    {
        return Err(crate::runtime::compute_error("DeepSeek-V4 segmented attention/indexer compressor 窗口状态不一致"));
    }
    backend.profile_device_operator("attn_csa")?;
    let mut sparse_segments = segments
        .iter_mut()
        .enumerate()
        .map(|(index, segment)| crate::attention::compressed_sparse::CompressedSparsePrefillSegment {
            storage: &mut *segment.cache,
            positions: segment.positions,
            causal_batch: segment.causal_batch,
            compressed_positions: compressed.as_ref().map(|batches| batches[index].visible_positions.as_slice()),
            compressed_key: compressed.as_ref().map(|batches| &batches[index].values),
            compressed_value: compressed.as_ref().map(|batches| &batches[index].values),
            compressed_index_key: compressed_index.as_ref().map(|batches| &batches[index].values),
        })
        .collect::<Vec<_>>();
    let attended = backend.compressed_sparse_prefill_segmented(&query, &key_value, &key_value, index_query.as_ref(), index_head_weights.as_ref(), &mut sparse_segments, Some(&weights.sink), &spec.attention)?;
    backend.profile_device_operator("attn_out")?;
    let attended = backend.rope_segmented(&attended, config.num_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, false, &token_segments, &rope.table.cos, rope.inverse_sin)?;
    let projected = backend.grouped_linear_columns(&attended, &weights.output_inputs)?;
    backend.linear(&projected, &weights.output)
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn deepseek_v4_store_attention_kv<B>(
    backend: &B,
    config: &DeepSeekV4Config,
    spec: &DeepSeekV4LayerSpec,
    weights: &PreparedAttention<B::Weight>,
    cache: &mut B::CompressedKvStorage,
    input: &B::Tensor,
    rope: DeepSeekV4Rope<'_>,
    positions: &[usize],
) -> Result<(), BackendError>
where
    B: crate::attention::compressed_sparse::CompressedSparseKernel,
{
    if spec.attention.compression.is_some() || weights.compressor.is_some() || weights.indexer.is_some() {
        return Err(crate::runtime::compute_error("DeepSeek-V4 draft KV-only prefill 不支持压缩历史"));
    }
    let key_value = backend.linear(input, &weights.key_value)?;
    let key_value = backend.rmsnorm(&key_value, &weights.key_norm, config.rms_eps)?;
    let key_value = backend.rope(&key_value, config.num_kv_heads, config.qk_rope_head_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, &rope.table.sin)?;
    backend.compressed_sparse_store_recent(cache, positions, &key_value, &key_value)
}

fn feedforward_prefill<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &DeepSeekV4LayerSpec,
    weights: &PreparedFeedforward<B::Weight>,
    experts: &mut B::PrefillExperts,
    layer: usize,
    input: &B::Tensor,
    token_ids: &[u32],
    image_token_id: Option<u32>,
) -> Result<B::Tensor, BackendError> {
    let selected;
    // V4.1 VL:noaux_tc_for_vl 对 image span 行改用 bias_vl 选专家(路由权重仍来自无偏分数)
    let image_rows;
    let (router_weight, router_bias, router_bias_vl, image_mask, selected_experts) = match &weights.router {
        PreparedRouter::TokenHash { weight, zero_bias, token_to_experts } => {
            let vocab_size = token_to_experts.len() / spec.feedforward.top_k;
            if token_ids.iter().any(|&token| token as usize >= vocab_size) {
                return Err(crate::runtime::compute_error(format!("DeepSeek-V4 L{layer} token hash id 越界: vocab={vocab_size} tokens={token_ids:?}",)));
            }
            selected = token_ids
                .iter()
                .flat_map(|&token| {
                    let start = token as usize * spec.feedforward.top_k;
                    token_to_experts[start..start + spec.feedforward.top_k].iter().copied()
                })
                .collect::<Vec<_>>();
            image_rows = Vec::new();
            (weight, zero_bias, None, None, Some(selected.as_slice()))
        }
        PreparedRouter::ScoreTopK { weight, correction_bias, value_level_bias } => match (value_level_bias.as_ref(), image_token_id) {
            (Some(bias), Some(image)) => {
                image_rows = token_ids.iter().map(|&token| token == image).collect::<Vec<_>>();
                let mask = image_rows.iter().any(|&row| row).then_some(image_rows.as_slice());
                (weight, correction_bias, Some(bias), mask, None)
            }
            _ => {
                image_rows = Vec::new();
                (weight, correction_bias, None, None, None)
            }
        },
    };
    let shared = [crate::moe::topk_moe::SharedExpertRef { gate: &weights.shared_gate, up: &weights.shared_up, down: &weights.shared_down, output_gate: None }];
    let moe = crate::moe::topk_moe::MoeFfnRef { router_weight, router_bias, shared_experts: &shared, selected_experts, router_bias_vl, image_rows: image_mask };
    // 生产路径不消费 host route trace；由 backend 根据批量规模选择 resident fused
    // 或通用 grouped 路径，模型层不再硬编码 decode/verify 分界。
    crate::moe::prefill::prefill_experts_untraced(backend, &spec.feedforward, &moe, layer, experts, input, None)
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn flash_uses_two_rope_domains() {
        let model = DeepSeekV4::flash();
        let tables = DeepSeekV4RopeTables::new(&model, 256).unwrap();
        assert!(std::ptr::eq(tables.layer(model.layer_spec(0).unwrap()).table, &tables.sliding));
        assert!(std::ptr::eq(tables.layer(model.layer_spec(2).unwrap()).table, &tables.compressed));
    }

    #[test]
    fn layer_cache_prefers_smaller_layers_within_budget() {
        // 层字节 [300, 100, 200]：预算 350 放下层 1、2，层 0 放不下；预算 0 全部按需加载。
        let cache = DeepSeekV4LayerCache::<()>::with_layer_bytes(vec![300, 100, 200], 350);
        assert_eq!(cache.planned_layers(), 2);
        assert_eq!(cache.planned_bytes(), 300);
        let empty = DeepSeekV4LayerCache::<()>::with_layer_bytes(vec![300, 100, 200], 0);
        assert_eq!(empty.planned_layers(), 0);
        assert_eq!(empty.planned_bytes(), 0);
    }
}

// DeepSeek-V4 模型规格。
//
// V4 不是 V3 MLA 的参数变体：每层维护四路 mHC hidden，注意力由滑窗和压缩
// 历史共同组成，前三层还使用 token→expert 固定表。这里保留完整差异，不把它
// 压入普通 [`super::LayerSpec`]，避免改变已有模型的 residual 与 cache 语义。

use crate::{
    attention::{
        compressed_sparse::{CompressedKvFormat, CompressedSelection, CompressedSparseAttentionSpec, KvCompressionSpec},
        dsa::DsaSpec,
        hyper_connection::HyperConnectionSpec,
        rope::RopeSpec,
    },
    moe::{
        Activation,
        topk_moe::{ScoringFunc, TopkMoeSpec},
    },
    norm::NormSpec,
    runtime::{LayerId, ModelError},
};

pub use crate::model_spec::deepseek_v4::DeepSeekV4RoutingSelection;

#[derive(Debug)]
pub struct DeepSeekV4LayerSpec {
    pub attention: CompressedSparseAttentionSpec,
    pub feedforward: TopkMoeSpec,
    pub attention_norm: NormSpec,
    pub feedforward_norm: NormSpec,
    pub hyper_connection: HyperConnectionSpec,
    pub routing: DeepSeekV4RoutingSelection,
}

pub use crate::model_spec::deepseek_v4::DeepSeekV4Config;

pub struct DeepSeekV4 {
    config: DeepSeekV4Config,
    layer_specs: Vec<DeepSeekV4LayerSpec>,
    /// V4.1(kv_source_layers 非空):压缩层 → 组源层(kvc 共享权威)。
    /// ratio=0 层与 V4 配置恒为 None。
    kv_source_of: Vec<Option<usize>>,
    /// V4.1:本层是否 index 层(自算并发布 selection)。
    index_layer: Vec<bool>,
}

impl DeepSeekV4 {
    pub fn new(config: DeepSeekV4Config) -> Result<Self, ModelError> {
        if config.layer_count == 0 || config.hidden_size == 0 || config.hash_layer_count > config.layer_count || config.compress_ratios.len() != config.layer_count + config.mtp_layer_count {
            return Err(ModelError::InvalidArchitecture(format!(
                "DeepSeek-V4 层配置非法: hidden={} layers={} hash={} mtp={} ratios={}",
                config.hidden_size,
                config.layer_count,
                config.hash_layer_count,
                config.mtp_layer_count,
                config.compress_ratios.len(),
            )));
        }
        if config.expert_count == 0
            || config.expert_top_k == 0
            || config.expert_top_k > config.expert_count
            || config.shared_expert_count != 1
            || config.expert_intermediate_size == 0
            || !config.routed_scaling_factor.is_finite()
            || config.routed_scaling_factor <= 0.0
            || !config.swiglu_limit.is_finite()
            || config.swiglu_limit <= 0.0
        {
            return Err(ModelError::InvalidArchitecture("DeepSeek-V4 MoE 配置非法".into()));
        }
        if config.hyper_connection_copies == 0 || config.hyper_connection_sinkhorn_iterations == 0 || !config.hyper_connection_eps.is_finite() || config.hyper_connection_eps <= 0.0 {
            return Err(ModelError::InvalidArchitecture("DeepSeek-V4 mHC 配置非法".into()));
        }
        HyperConnectionSpec { copies: config.hyper_connection_copies, sinkhorn_iterations: config.hyper_connection_sinkhorn_iterations, eps: config.hyper_connection_eps }.validate().map_err(ModelError::InvalidArchitecture)?;
        let layer_specs = (0..config.layer_count).map(|layer| Self::build_layer_spec(&config, layer)).collect::<Result<_, _>>()?;
        // V4.1 层映射:压缩层的组源 = 不超过本层的最近 kv_source;index 层集合独立给出。
        let v41 = !config.kv_source_layers.is_empty();
        if v41
            && (config.kv_source_layers.iter().any(|&layer| layer >= config.layer_count || config.compress_ratios[layer] == 0)
                || config.index_source_layers.iter().any(|&layer| layer >= config.layer_count || config.compress_ratios[layer] == 0)
                || config.candidate_source_layer.is_some_and(|layer| !config.index_source_layers.contains(&layer)))
        {
            return Err(ModelError::InvalidArchitecture(format!("DeepSeek-V4.1 层映射非法: kv_source={:?} index_source={:?} candidate={:?}", config.kv_source_layers, config.index_source_layers, config.candidate_source_layer)));
        }
        let kv_source_of = (0..config.layer_count).map(|layer| if !v41 || config.compress_ratios[layer] == 0 { None } else { config.kv_source_layers.iter().copied().filter(|&source| source <= layer).max() }).collect::<Vec<_>>();
        let index_layer = (0..config.layer_count).map(|layer| v41 && config.index_source_layers.contains(&layer)).collect::<Vec<_>>();
        Ok(Self { config, layer_specs, kv_source_of, index_layer })
    }

    pub fn flash() -> Self {
        Self::new(DeepSeekV4Config::flash()).expect("DeepSeek-V4-Flash 官方配置必须有效")
    }

    pub fn flash_v41() -> Self {
        Self::new(DeepSeekV4Config::flash_v41()).expect("DeepSeek-V4.1-Flash 官方配置必须有效")
    }

    /// V4.1 共享架构(kv_source_layers 非空)时返回压缩层的组源层。
    pub fn kv_source_of(&self, layer: LayerId) -> Option<usize> {
        self.kv_source_of.get(layer).copied().flatten()
    }

    pub fn is_kv_source(&self, layer: LayerId) -> bool {
        self.kv_source_of.get(layer).is_some_and(|source| *source == Some(layer))
    }

    pub fn is_index_layer(&self, layer: LayerId) -> bool {
        self.index_layer.get(layer).copied().unwrap_or(false)
    }

    /// V4.1:candidate_source 之后的 index 层在 top-k 前先做候选块粗筛。
    pub fn candidate_spec_for(&self, layer: LayerId) -> Option<crate::attention::compressed_sparse::CandidateSpec> {
        let config = &self.config;
        if !self.is_index_layer(layer) {
            return None;
        }
        let source = config.candidate_source_layer?;
        (layer > source).then_some(crate::attention::compressed_sparse::CandidateSpec { topk_blocks: config.candidate_topk_blocks, block_size: config.candidate_block_size })
    }

    pub fn is_v41(&self) -> bool {
        !self.config.kv_source_layers.is_empty()
    }

    /// 只缩减分数路由层的专家数；TokenHash 固定表宽度和 checkpoint
    /// 架构仍保持官方配置，避免把执行近似混入权重格式。
    pub fn with_score_expert_top_k(mut self, top_k: usize) -> Result<Self, ModelError> {
        if top_k == 0 || top_k > self.config.expert_top_k {
            return Err(ModelError::InvalidArchitecture(format!("DeepSeek-V4 score expert top-k={top_k} 非法，期望 1..={}", self.config.expert_top_k)));
        }
        for spec in &mut self.layer_specs {
            if spec.routing == DeepSeekV4RoutingSelection::ScoreTopK {
                spec.feedforward.top_k = top_k;
            }
        }
        Ok(self)
    }

    fn build_layer_spec(config: &DeepSeekV4Config, layer: LayerId) -> Result<DeepSeekV4LayerSpec, ModelError> {
        let ratio = config.compress_ratios[layer];
        let rope = if ratio == 0 {
            RopeSpec::Default { rotary_dim: config.qk_rope_head_dim, theta: config.rope_theta }
        } else {
            RopeSpec::Yarn {
                rotary_dim: config.qk_rope_head_dim,
                theta: config.compress_rope_theta,
                factor: config.rope_factor,
                original_context: config.original_position_embeddings,
                beta_fast: config.rope_beta_fast,
                beta_slow: config.rope_beta_slow,
            }
        };
        let compression = match ratio {
            0 => None,
            // V4 ratio=4(4:1,窗口重叠)与 V4.1 ratio=2(2:1,不重叠)都携带 LearnedIndexer 张量集。
            ratio @ (2 | 4) => Some(KvCompressionSpec {
                ratio,
                overlap: ratio == 4,
                selection: CompressedSelection::LearnedIndexer(DsaSpec {
                    num_heads: config.index_heads,
                    head_dim: config.index_head_dim,
                    rope_dim: config.qk_rope_head_dim,
                    top_k: config.index_top_k,
                    rotary_layout: crate::attention::rope::RotaryLayout::Interleaved,
                    kpool: 0,
                    always_select_tail: false,
                }),
            }),
            // V4.1 ratio=1:L20 投影 KV(1:1)共享给 L20-39,稀疏性全靠 LearnedIndexer topk。
            ratio if ratio >= 1 && !config.index_source_layers.is_empty() => Some(KvCompressionSpec {
                ratio,
                overlap: false,
                selection: CompressedSelection::LearnedIndexer(DsaSpec {
                    num_heads: config.index_heads,
                    head_dim: config.index_head_dim,
                    rope_dim: config.qk_rope_head_dim,
                    top_k: config.index_top_k,
                    rotary_layout: crate::attention::rope::RotaryLayout::Interleaved,
                    kpool: 0,
                    always_select_tail: false,
                }),
            }),
            ratio => Some(KvCompressionSpec { ratio, overlap: false, selection: CompressedSelection::All }),
        };
        let attention = CompressedSparseAttentionSpec {
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            q_lora_rank: config.q_lora_rank,
            output_groups: config.output_groups,
            output_lora_rank: config.output_lora_rank,
            window_size: config.sliding_window,
            rope,
            compression,
            attention_sink: true,
            kv_format: if config.kv_source_layers.is_empty() { CompressedKvFormat::Q8 } else { CompressedKvFormat::Fp8WindowFp4Compressed },
        };
        attention.validate().map_err(ModelError::InvalidArchitecture)?;
        let norm = NormSpec::Rms { eps: config.rms_eps };
        Ok(DeepSeekV4LayerSpec {
            attention,
            feedforward: TopkMoeSpec {
                num_experts: config.expert_count,
                top_k: config.expert_top_k,
                num_shared_experts: config.shared_expert_count,
                scoring_func: ScoringFunc::SqrtSoftplusBias,
                normalize_selected: true,
                routed_scaling_factor: config.routed_scaling_factor,
                intermediate_size: config.expert_intermediate_size,
                shared_intermediate_size: config.expert_intermediate_size,
                activation: Activation::SiluClamped { limit: config.swiglu_limit },
            },
            attention_norm: norm,
            feedforward_norm: NormSpec::Rms { eps: config.rms_eps },
            hyper_connection: HyperConnectionSpec { copies: config.hyper_connection_copies, sinkhorn_iterations: config.hyper_connection_sinkhorn_iterations, eps: config.hyper_connection_eps },
            routing: if layer < config.hash_layer_count { DeepSeekV4RoutingSelection::TokenHash } else { DeepSeekV4RoutingSelection::ScoreTopK },
        })
    }

    pub fn config(&self) -> &DeepSeekV4Config {
        &self.config
    }

    pub fn layer_count(&self) -> usize {
        self.config.layer_count
    }

    pub fn layer_spec(&self, layer: LayerId) -> Result<&DeepSeekV4LayerSpec, ModelError> {
        self.layer_specs.get(layer).ok_or(ModelError::LayerOutOfRange { layer, layer_count: self.config.layer_count })
    }

    pub fn mtp_compression_ratio(&self, mtp_layer: usize) -> Result<usize, ModelError> {
        self.config.compress_ratios.get(self.config.layer_count + mtp_layer).copied().ok_or(ModelError::LayerOutOfRange { layer: mtp_layer, layer_count: self.config.mtp_layer_count })
    }

    /// 单层 KV 每 token 的线性增长字节。K/V 共享一份常驻量化行；V4 用
    /// Q8g64，V4.1 compressed KV 用 E2M1+E4M3(group 16)，index K 用
    /// E2M1+E8M0(group 32)；compressed 行数按 ratio 随 token 折减。ratio=0
    /// 层只有有界 recent ring，线性增长为 0。recent ring、batch 工作区等每会话
    /// 固定开销不在线性项里，由节点上报时的 memory_reserve_bytes 覆盖。
    pub fn kv_bytes_per_layer_token(&self, layer: LayerId) -> usize {
        let attention = &self.layer_specs[layer].attention;
        let Some(compression) = attention.compression else { return 0 };
        let kv_width = attention.num_kv_heads * attention.head_dim;
        let row_bytes = match attention.kv_format {
            CompressedKvFormat::Q8 => {
                let group = [crate::kv_cache::DEFAULT_GROUP_SIZE, 32, 16, 8, 4, 2, 1].into_iter().find(|&group| attention.head_dim.is_multiple_of(group)).unwrap_or(1);
                kv_width + kv_width / group * 2
            }
            CompressedKvFormat::Fp8WindowFp4Compressed => kv_width / 2 + kv_width / 16,
        };
        let index_bytes = matches!(compression.selection, crate::attention::compressed_sparse::CompressedSelection::LearnedIndexer(_))
            .then_some(match attention.kv_format {
                CompressedKvFormat::Q8 => self.config.index_head_dim * 4,
                CompressedKvFormat::Fp8WindowFp4Compressed => self.config.index_head_dim / 2 + self.config.index_head_dim / 32,
            })
            .unwrap_or(0);
        row_bytes.div_ceil(compression.ratio) + index_bytes.div_ceil(compression.ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 官方high模板保留system与user() {
        let prompt = deepseek_v4_chat_prompt(
            [
                DeepSeekV4ChatMessage { role: "system", content: "system", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
                DeepSeekV4ChatMessage { role: "user", content: "question", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
            ],
            None,
            Some("enabled"),
            Some("high"),
        )
        .unwrap();
        assert_eq!(prompt, format!("{BOS}{HIGH_REASONING_PREFIX}system{USER}question{ASSISTANT}<think>"));
    }

    #[test]
    fn 官方chat模板显式关闭thinking() {
        let prompt = deepseek_v4_chat_prompt([DeepSeekV4ChatMessage { role: "user", content: "question", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] }], None, Some("disabled"), None).unwrap();
        assert_eq!(prompt, format!("{BOS}{USER}question{ASSISTANT}</think>"));
    }

    #[test]
    fn api默认不把私有thinking混入正文() {
        let prompt = deepseek_v4_chat_prompt([DeepSeekV4ChatMessage { role: "user", content: "question", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] }], None, None, Some("high")).unwrap();
        assert_eq!(prompt, format!("{BOS}{USER}question{ASSISTANT}</think>"));
    }

    #[test]
    fn kv每token字节按压缩比与indexer折算() {
        let model = DeepSeekV4::flash();
        // ratio=0：只有有界 recent ring，无线性增长。
        assert_eq!(model.kv_bytes_per_layer_token(0), 0);
        assert_eq!(model.kv_bytes_per_layer_token(1), 0);
        // ratio=4 + LearnedIndexer：共享 Q8 行 512+16 与 f32 index 行 128*4 各除以 4。
        assert_eq!(model.kv_bytes_per_layer_token(2), (512 + 16 + 128 * 4usize).div_ceil(4));
        // ratio=128 + All：只有共享 Q8 行参与折减。
        assert_eq!(model.kv_bytes_per_layer_token(3), (512usize + 16).div_ceil(128));
        // 全模型线性口径 ≈ 5.4 KiB/token，量级异常时这里会先报警。
        let total: usize = (0..model.layer_count()).map(|layer| model.kv_bytes_per_layer_token(layer)).sum();
        assert!((5000..6000).contains(&total), "flash 全模型 bytes/token={total}");

        let model = DeepSeekV4::flash_v41();
        // 官方 compressed 行：512/2 bytes E2M1 code + 512/16 bytes E4M3 scale；
        // index 行：128/2 bytes E2M1 code + 128/32 bytes E8M0 scale。
        assert_eq!(model.kv_bytes_per_layer_token(2), (512 / 2 + 512 / 16usize).div_ceil(2) + (128 / 2 + 128 / 32usize).div_ceil(2));
        assert_eq!(model.kv_bytes_per_layer_token(20), 512 / 2 + 512 / 16 + 128 / 2 + 128 / 32);
    }

    #[test]
    fn 官方工具模板位于system之后并合并developer_user_tool() {
        let prompt = deepseek_v4_chat_prompt(
            [
                DeepSeekV4ChatMessage { role: "system", content: "system", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
                DeepSeekV4ChatMessage { role: "developer", content: "developer", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
                DeepSeekV4ChatMessage { role: "user", content: "question", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
                DeepSeekV4ChatMessage { role: "tool", content: "result", reasoning_content: None, tool_calls: "", tool_call_id: Some("call_1"), tool_call_ids: &[] },
            ],
            Some("tools"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(prompt, format!("{BOS}system\n\ntools{USER}developer{USER}question\n\n<tool_result>result</tool_result>{ASSISTANT}</think>"));
    }

    #[test]
    fn 只有user时工具模板挂到合成system() {
        let prompt = deepseek_v4_chat_prompt([DeepSeekV4ChatMessage { role: "user", content: "question", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] }], Some("tools"), None, None).unwrap();
        assert_eq!(prompt, format!("{BOS}\n\ntools{USER}question{ASSISTANT}</think>"));
    }

    #[test]
    fn 公共工具块边界不包含首条用户消息() {
        let build = |question| {
            deepseek_v4_chat_prompt_with_public_prefix(
                [
                    DeepSeekV4ChatMessage { role: "system", content: "system", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
                    DeepSeekV4ChatMessage { role: "user", content: question, reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
                ],
                Some("tools"),
                None,
                None,
            )
            .unwrap()
        };
        let (first, first_boundary) = build("first");
        let (second, second_boundary) = build("second");
        let first_boundary = first_boundary.unwrap();
        let second_boundary = second_boundary.unwrap();
        assert_eq!(&first[..first_boundary], &second[..second_boundary]);
        assert_eq!(&first[..first_boundary], format!("{BOS}system\n\ntools"));
        assert_ne!(&first[first_boundary..], &second[second_boundary..]);
    }

    #[test]
    fn 没有工具说明时不创建公共工具块() {
        let (_, boundary) =
            deepseek_v4_chat_prompt_with_public_prefix([DeepSeekV4ChatMessage { role: "user", content: "question", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] }], None, None, None).unwrap();
        assert_eq!(boundary, None);
    }

    #[test]
    fn 官方多轮工具结果按调用id排序且不重复thinking边界() {
        let ids = vec!["call_b".to_owned(), "call_a".to_owned()];
        let calls = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"Read\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let prompt = deepseek_v4_chat_prompt(
            [
                DeepSeekV4ChatMessage { role: "system", content: "system", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
                DeepSeekV4ChatMessage { role: "user", content: "question", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
                DeepSeekV4ChatMessage { role: "assistant", content: "", reasoning_content: None, tool_calls: calls, tool_call_id: None, tool_call_ids: &ids },
                DeepSeekV4ChatMessage { role: "tool", content: "A", reasoning_content: None, tool_calls: "", tool_call_id: Some("call_a"), tool_call_ids: &[] },
                DeepSeekV4ChatMessage { role: "tool", content: "B", reasoning_content: None, tool_calls: "", tool_call_id: Some("call_b"), tool_call_ids: &[] },
                DeepSeekV4ChatMessage { role: "user", content: "continue", reasoning_content: None, tool_calls: "", tool_call_id: None, tool_call_ids: &[] },
            ],
            Some("tools"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(prompt, format!("{BOS}system\n\ntools{USER}question{ASSISTANT}</think>{calls}{EOS}{USER}<tool_result>B</tool_result>\n\n<tool_result>A</tool_result>\n\ncontinue{ASSISTANT}</think>"));
    }

    #[test]
    fn terminal续接只编码assistant之后的后缀() {
        let assistant = format!("{BOS}{USER}question{ASSISTANT}</think>answer{EOS}");
        let full = format!("{BOS}{USER}question{ASSISTANT}</think>answer{EOS}{USER}<tool_result>ok</tool_result>{ASSISTANT}</think>");
        assert_eq!(deepseek_v4_resume_suffix(&full, &assistant).unwrap(), format!("{EOS}{USER}<tool_result>ok</tool_result>{ASSISTANT}</think>"));
    }

    #[test]
    fn flash_preserves_official_attention_and_routing_pattern() {
        let model = DeepSeekV4::flash();
        assert_eq!(model.layer_count(), 43);
        assert_eq!(model.mtp_compression_ratio(0).unwrap(), 0);
        assert_eq!(model.mtp_compression_ratio(2).unwrap(), 0);
        assert_eq!(model.layer_spec(0).unwrap().routing, DeepSeekV4RoutingSelection::TokenHash);
        assert!(model.layer_spec(0).unwrap().attention.compression.is_none());
        assert!(matches!(model.layer_spec(2).unwrap().attention.compression.unwrap().selection, CompressedSelection::LearnedIndexer(_)));
        assert!(matches!(model.layer_spec(3).unwrap().attention.compression.unwrap().selection, CompressedSelection::All));
        assert_eq!(model.layer_spec(3).unwrap().routing, DeepSeekV4RoutingSelection::ScoreTopK);
    }

    #[test]
    fn flash_v41_layer_specs_match_official_pattern() {
        let model = DeepSeekV4::flash_v41();
        assert_eq!(model.layer_count(), 40);
        assert_eq!(model.mtp_compression_ratio(0).unwrap(), 0);
        // 全层 ScoreTopK(hash_layer_count=0)
        for layer in 0..40 {
            assert_eq!(model.layer_spec(layer).unwrap().routing, DeepSeekV4RoutingSelection::ScoreTopK, "layer {layer}");
        }
        // L0/1 无压缩
        assert!(model.layer_spec(0).unwrap().attention.compression.is_none());
        assert!(model.layer_spec(1).unwrap().attention.compression.is_none());
        // L2-39 ratio=2/1 全部 LearnedIndexer(CSA2:idxs 由 index_source 层算/共享)
        for layer in 2..20 {
            let compression = model.layer_spec(layer).unwrap().attention.compression.unwrap();
            assert_eq!(compression.ratio, 2);
            assert!(matches!(compression.selection, CompressedSelection::LearnedIndexer(_)), "layer {layer}");
        }
        for layer in 20..40 {
            let compression = model.layer_spec(layer).unwrap().attention.compression.unwrap();
            assert_eq!(compression.ratio, 1);
            assert!(matches!(compression.selection, CompressedSelection::LearnedIndexer(_)), "layer {layer}");
        }
    }

    #[test]
    fn score_expert_top_k_preserves_token_hash_width() {
        let model = DeepSeekV4::flash().with_score_expert_top_k(5).unwrap();
        assert_eq!(model.config().expert_top_k, 6);
        assert_eq!(model.layer_spec(0).unwrap().feedforward.top_k, 6);
        assert_eq!(model.layer_spec(3).unwrap().feedforward.top_k, 5);
    }
}

/// V4.1 共享架构的小尺寸全前向验证:压缩共享、index selection 复用、候选粗筛、
/// engram 注入与 head 折叠全部经真实 CPU kernel 路径跑通(权重为确定性 dummy)。
#[cfg(test)]
mod v41_forward_tests {
    use super::*;
    use crate::backend::BackendResources;
    use crate::backend::cpu::{CpuCompressedKvStorage, CpuContext, CpuPrefillExperts, CpuWeight};
    use crate::kernel::cpu::CpuTensor;
    use crate::model_spec::deepseek_v4::{DeepSeekV4Config, DeepSeekV4EngramConfig};
    use crate::runtime::deepseek_v4::engram_cpu::{EngramHashState, EngramLayerCpu, EngramTokenMap};
    use crate::weight::container::safetensor::TensorData;

    /// xorshift 确定性伪随机,输出 [-1, 1)。
    struct Lcg(u64);
    impl Lcg {
        fn bump(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn next(&mut self) -> f32 {
            (self.bump() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        fn vector(&mut self, len: usize) -> Vec<f32> {
            (0..len).map(|_| self.next()).collect()
        }
        fn tensor(&mut self, name: &str, shape: Vec<usize>) -> TensorData {
            let len = shape.iter().product::<usize>();
            let data = self.vector(len).into_iter().flat_map(f32::to_le_bytes).collect();
            TensorData { name: name.to_owned(), dtype: "F32".to_owned(), shape, data }
        }
    }

    fn small_v41_config() -> DeepSeekV4Config {
        DeepSeekV4Config {
            image_token_id: None,
            vocab_size: 128,
            hidden_size: 32,
            layer_count: 8,
            mtp_layer_count: 0,
            hash_layer_count: 0,
            max_position_embeddings: 256,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 16,
            q_lora_rank: 8,
            qk_rope_head_dim: 8,
            output_groups: 2,
            output_lora_rank: 8,
            sliding_window: 4,
            // L0/1 滑窗;L2-3 ratio=2(源 L2);L4-7 ratio=1(源 L4,共享 1:1 KV)。
            compress_ratios: vec![0, 0, 2, 2, 1, 1, 1, 1],
            rope_theta: 10_000.0,
            compress_rope_theta: 160_000.0,
            rope_factor: 4.0,
            original_position_embeddings: 64,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            index_heads: 2,
            index_head_dim: 8,
            index_top_k: 4,
            expert_count: 4,
            mtp_expert_count: 4,
            mtp_expert_top_k: 2,
            expert_top_k: 2,
            shared_expert_count: 1,
            expert_intermediate_size: 8,
            routed_scaling_factor: 1.5,
            swiglu_limit: 10.0,
            rms_eps: 1.0e-6,
            hyper_connection_copies: 2,
            hyper_connection_sinkhorn_iterations: 4,
            hyper_connection_eps: 1.0e-6,
            bos_token_id: 0,
            eos_token_ids: vec![1],
            engram: Some(DeepSeekV4EngramConfig {
                layer_ids: vec![1, 4],
                num_embeddings: vec![64; 2],
                max_ngram_size: 4,
                vocab_size: 1_000,
                n_heads: 8,
                // EngramLayerCpu 固定消费 24×256 的 embed 布局(ENGRAM_HEAD_DIM 常量)。
                head_dim: 256,
                pad_token_id: 2,
                compressed_vocab_size: 1_000,
            }),
            router_value_level_bias: false,
            kv_source_layers: vec![2, 4],
            index_source_layers: vec![2, 4, 6],
            candidate_source_layer: Some(4),
            candidate_topk_blocks: 2,
            candidate_block_size: 2,
        }
    }

    fn prepared_hyper_connection(ctx: &CpuContext, config: &DeepSeekV4Config, sublayer: &str, rng: &mut Lcg) -> PreparedHyperConnection<CpuWeight> {
        let copies = config.hyper_connection_copies;
        let mixes = (2 + copies) * copies;
        let hidden = copies * config.hidden_size;
        PreparedHyperConnection {
            input_norm: prepare_unit_norm(ctx, hidden).unwrap(),
            function: prepare_dense_tensor(ctx, &rng.tensor(&format!("hc_{sublayer}_fn"), vec![mixes, hidden])).unwrap(),
            base: prepare_dense_tensor(ctx, &rng.tensor(&format!("hc_{sublayer}_base"), vec![mixes])).unwrap(),
            scale: prepare_dense_tensor(ctx, &rng.tensor(&format!("hc_{sublayer}_scale"), vec![3])).unwrap(),
        }
    }

    fn prepared_layer(ctx: &CpuContext, model: &DeepSeekV4, layer: usize, rng: &mut Lcg) -> DeepSeekV4PreparedLayer<CpuWeight> {
        let config = model.config();
        let matrix = |rows: usize, cols: usize, name: &str, rng: &mut Lcg| ctx.prepare_f32(&rng.vector(rows * cols), rows, cols).unwrap();
        let ratio = config.compress_ratios[layer];
        let is_source = model.is_kv_source(layer);
        let is_index = model.is_index_layer(layer);
        let compressor = (ratio != 0 && is_source).then(|| PreparedCompressor {
            position: None,
            key_value: prepare_dense_tensor(ctx, &rng.tensor("compressor_wkv", vec![config.head_dim, config.hidden_size])).unwrap(),
            gate: (ratio > 1).then(|| prepare_dense_tensor(ctx, &rng.tensor("compressor_wgate", vec![config.head_dim, config.hidden_size])).unwrap()),
            norm: prepare_dense_tensor(ctx, &rng.tensor("compressor_norm", vec![config.head_dim])).unwrap(),
        });
        let indexer = (is_index && ratio != 0).then(|| PreparedIndexer {
            query: matrix(config.index_heads * config.index_head_dim, config.q_lora_rank, "indexer_q", rng),
            head_weights: matrix(config.index_heads, config.hidden_size, "indexer_w", rng),
            key_projection: is_source.then(|| prepare_dense_tensor(ctx, &rng.tensor("indexer_wk", vec![config.index_head_dim, config.head_dim])).unwrap()),
            key_norm: is_source.then(|| prepare_dense_tensor(ctx, &rng.tensor("indexer_k_norm", vec![config.index_head_dim])).unwrap()),
            compressor: None,
        });
        let attention = PreparedAttention {
            sink: matrix(1, config.num_heads, "sink", rng),
            query_input: matrix(config.q_lora_rank, config.hidden_size, "q_a", rng),
            query_norm: prepare_dense_tensor(ctx, &rng.tensor("q_norm", vec![config.q_lora_rank])).unwrap(),
            query_output: matrix(config.num_heads * config.head_dim, config.q_lora_rank, "q_b", rng),
            query_head_norm: prepare_unit_norm(ctx, config.head_dim).unwrap(),
            key_value: matrix(config.num_kv_heads * config.head_dim, config.hidden_size, "wkv", rng),
            key_norm: prepare_dense_tensor(ctx, &rng.tensor("k_norm", vec![config.num_kv_heads * config.head_dim])).unwrap(),
            output_inputs: (0..config.output_groups).map(|_| matrix(config.output_lora_rank, config.num_heads * config.head_dim / config.output_groups, "wo_a", rng)).collect(),
            output: matrix(config.hidden_size, config.output_groups * config.output_lora_rank, "wo_b", rng),
            compressor,
            indexer,
        };
        let router = PreparedRouter::ScoreTopK { weight: matrix(config.expert_count, config.hidden_size, "router", rng), correction_bias: matrix(1, config.expert_count, "router_bias", rng), value_level_bias: None };
        let feedforward = PreparedFeedforward {
            hyper_connection: prepared_hyper_connection(ctx, config, "ffn", rng),
            norm: prepare_dense_tensor(ctx, &rng.tensor("ffn_norm", vec![config.hidden_size])).unwrap(),
            router,
            shared_gate: matrix(config.expert_intermediate_size, config.hidden_size, "sh_gate", rng),
            shared_up: matrix(config.expert_intermediate_size, config.hidden_size, "sh_up", rng),
            shared_down: matrix(config.hidden_size, config.expert_intermediate_size, "sh_down", rng),
        };
        DeepSeekV4PreparedLayer {
            attention_hyper_connection: prepared_hyper_connection(ctx, config, "attn", rng),
            attention_norm: prepare_dense_tensor(ctx, &rng.tensor("attn_norm", vec![config.hidden_size])).unwrap(),
            attention,
            feedforward,
        }
    }

    fn resident_experts(config: &DeepSeekV4Config, rng: &mut Lcg) -> CpuPrefillExperts {
        let layers = (0..config.layer_count)
            .map(|_| {
                (0..config.expert_count)
                    .map(|_| (rng.vector(config.expert_intermediate_size * config.hidden_size), rng.vector(config.expert_intermediate_size * config.hidden_size), rng.vector(config.hidden_size * config.expert_intermediate_size)))
                    .collect()
            })
            .collect();
        CpuPrefillExperts::f32_resident(layers)
    }

    /// 测试用 engram:真实 hash 状态机 + 真实门控数学,embed 表缩成 64 行
    /// (hash 行号取模后从确定性伪随机生成,数学路径与生产一致)。
    struct TestEngram {
        hash: EngramHashState,
        layers: Vec<EngramLayerCpu>,
        layer_ids: Vec<usize>,
        embed_dim: usize,
        table_rows: usize,
        chunk_start: usize,
    }

    impl TestEngram {
        fn new(config: &DeepSeekV4Config, rng: &mut Lcg) -> Self {
            let engram = config.engram.clone().expect("测试配置必含 engram");
            let map = EngramTokenMap { map: (0..config.vocab_size as u32).map(|token| token % 1_000).collect(), pad_id: 2 };
            let layers = engram
                .layer_ids
                .iter()
                .map(|&layer| {
                    let weights = crate::weight::model::deepseek_v4::DeepSeekV4EngramWeights {
                        // EngramLayerCpu 按 [hc, dim] 消费 q/k(逐元素 q⊙k 后按 copy 分段)。
                        k: rng.tensor("engram_k", vec![config.hyper_connection_copies, config.hidden_size]),
                        q: rng.tensor("engram_q", vec![config.hyper_connection_copies, config.hidden_size]),
                        wkv: crate::weight::model::deepseek_v4::DeepSeekV4CoreMatrix::Dense(rng.tensor("engram_wkv", vec![config.hidden_size * (config.hyper_connection_copies + 1), 24 * engram.head_dim])),
                    };
                    EngramLayerCpu::prepare(&weights, config.hyper_connection_copies, config.hidden_size, config.rms_eps).unwrap()
                })
                .collect();
            Self { hash: EngramHashState::new(map), layers, layer_ids: engram.layer_ids, embed_dim: engram.head_dim, table_rows: engram.num_embeddings[0], chunk_start: 0 }
        }

        fn embed_row(&self, row: i64, out: &mut Vec<f32>) {
            let row = (row.unsigned_abs() % self.table_rows as u64) as u64;
            let mut rng = Lcg(row ^ 0x9E3779B97F4A7C15);
            out.extend((0..self.embed_dim).map(|_| rng.next()));
        }

        fn hook(&mut self, layer: usize, tokens: &[u32], tensor: &mut CpuTensor) -> Result<(), BackendError> {
            // 每 chunk 只推进一次 hash:第一个层调用时 push 整段。
            if layer == 0 {
                self.hash.prefill(tokens, 0);
            }
            let Some(slot) = self.layer_ids.iter().position(|&engram_layer| engram_layer == layer) else {
                return Ok(());
            };
            let stride = tensor.cols;
            let mut embed = Vec::with_capacity(self.embed_dim);
            for row in 0..tensor.rows {
                embed.clear();
                for hash_row in self.hash.hash_at(self.chunk_start + row, slot) {
                    self.embed_row(hash_row, &mut embed);
                }
                self.layers[slot].apply(&embed, &mut tensor.data[row * stride..(row + 1) * stride]).map_err(crate::runtime::compute_error)?;
            }
            Ok(())
        }
    }

    fn run_forward(
        model: &DeepSeekV4,
        engram: Option<&mut TestEngram>,
        tokens: &[u32],
        position: usize,
        caches: &mut [CpuCompressedKvStorage],
        experts: &mut CpuPrefillExperts,
        layers: &mut DeepSeekV4LayerCache<CpuWeight>,
        head: &DeepSeekV4OutputHead<CpuWeight>,
    ) -> u32 {
        let ctx = CpuContext;
        let rope = DeepSeekV4RopeTables::new(model, position + tokens.len()).unwrap();
        let hidden_size = model.config().hidden_size;
        let embedding = (0..tokens.len() * hidden_size)
            .map(|index| {
                let token = tokens[index / hidden_size] as f32;
                let column = (index % hidden_size) as f32;
                (token + column * 0.125) / 256.0
            })
            .collect::<Vec<_>>();
        let input = CpuTensor { data: embedding, rows: tokens.len(), cols: model.config().hidden_size };
        let hidden = deepseek_v4_forward_impl(
            &ctx,
            model,
            caches,
            experts,
            Some(layers),
            &input,
            &rope,
            position,
            tokens,
            engram.map(|engram| move |layer: usize, chunk_tokens: &[u32], tensor: &mut CpuTensor| engram.hook(layer, chunk_tokens, tensor)),
            |_, layer| Err(crate::runtime::compute_error(format!("resident 层不应重新 prepare: L{layer}"))),
        )
        .unwrap();
        let result = deepseek_v4_token_output(&ctx, model.config(), head, &hidden).unwrap();
        assert!((result.token_id as usize) < model.config().vocab_size);
        assert!(result.logits.data.iter().all(|value| value.is_finite()));
        result.token_id
    }

    #[test]
    fn v41小尺寸全前向覆盖压缩共享候选与engram() {
        let config = small_v41_config();
        let model = DeepSeekV4::new(config.clone()).unwrap();
        let ctx = CpuContext;
        let mut rng = Lcg(0x5EED_41FA);
        let layer_count = model.layer_count();
        let mut layer_cache = DeepSeekV4LayerCache { layers: (0..layer_count).map(|_| None).collect(), resident: vec![true; layer_count], planned_bytes: 0 };
        for layer in 0..layer_count {
            layer_cache.layers[layer] = Some(prepared_layer(&ctx, &model, layer, &mut rng));
        }
        let mut experts = resident_experts(&config, &mut rng);
        let head = DeepSeekV4OutputHead {
            input_norm: prepare_unit_norm(&ctx, config.hidden_size).unwrap(),
            hyper_connection: None,
            output: crate::runtime::output::prepare_output_head_quantized(
                &ctx,
                &vec![1.0; config.hidden_size],
                LinearWeight::F32(&rng.vector(config.vocab_size * config.hidden_size)),
                config.vocab_size,
                config.hidden_size,
                crate::weight::LmHeadQuantization::Native,
            )
            .unwrap(),
        };
        let mut caches = allocate_deepseek_v4_caches(&ctx, &model).unwrap();

        // engram 关闭:基线前向(prefill + 2 步 decode)。
        let prompt: Vec<u32> = vec![3, 17, 42, 5, 99, 8, 64, 21];
        let mut baseline = Vec::new();
        let mut token = run_forward(&model, None, &prompt, 0, &mut caches, &mut experts, &mut layer_cache, &head);
        baseline.push(token);
        for step in 1..=2 {
            token = run_forward(&model, None, std::slice::from_ref(&token), prompt.len() + step - 1, &mut caches, &mut experts, &mut layer_cache, &head);
            baseline.push(token);
        }

        // engram 打开:重新分配 cache,同权重重跑;engram 注入应改变输出。
        let mut engram_caches = allocate_deepseek_v4_caches(&ctx, &model).unwrap();
        let mut engram = TestEngram::new(&config, &mut rng);
        let mut with_engram = Vec::new();
        let mut token = run_forward(&model, Some(&mut engram), &prompt, 0, &mut engram_caches, &mut experts, &mut layer_cache, &head);
        with_engram.push(token);
        for step in 1..=2 {
            token = run_forward(&model, Some(&mut engram), std::slice::from_ref(&token), prompt.len() + step - 1, &mut engram_caches, &mut experts, &mut layer_cache, &head);
            with_engram.push(token);
        }
        assert_ne!(baseline, with_engram, "engram 注入必须改变前向输出");
    }
}
