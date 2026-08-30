//! DeepSeek-V4-Flash 的平台无关运行时编排组件。
//!
//! 这里仅保存模型选择：普通层使用短上下文 RoPE，压缩层使用扩展 YaRN RoPE。
//! mHC/CSA 的数学语义在 `attention`，设备资源与 kernel 由 backend capability 拥有。

pub mod protocol;
#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
pub(crate) use protocol::resume_suffix as deepseek_v4_resume_suffix;
#[cfg(test)]
use protocol::{ASSISTANT, BOS, EOS, HIGH_REASONING_PREFIX, USER};
pub use protocol::{DeepSeekV4ChatMessage, chat_prompt as deepseek_v4_chat_prompt, chat_prompt_with_public_prefix as deepseek_v4_chat_prompt_with_public_prefix};

use crate::{
    attention::{
        compressed_sparse::CompressedSparseKernel,
        compressed_sparse::CompressionStream,
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

struct PreparedHyperConnection<W> {
    input_norm: W,
    function: W,
    base: W,
    scale: W,
}

struct PreparedCompressor<W> {
    position: W,
    key_value: W,
    gate: W,
    norm: W,
}

struct PreparedIndexer<W> {
    query: W,
    head_weights: W,
    compressor: PreparedCompressor<W>,
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
    ScoreTopK { weight: W, correction_bias: W },
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

#[cfg(feature = "with-rocm")]
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
    function: W,
    base: W,
    scale: W,
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
        DeepSeekV4RouterWeights::ScoreTopK { weight, correction_bias } => PreparedRouter::ScoreTopK { weight: prepare_f32_tensor(backend, weight)?, correction_bias: prepare_f32_tensor(backend, correction_bias)? },
    };
    Ok(DeepSeekV4PreparedLayer {
        attention_hyper_connection: prepare_hyper_connection(backend, &weights.attention.hyper_connection)?,
        attention_norm: prepare_dense_tensor(backend, &weights.attention.norm)?,
        attention,
        feedforward: PreparedFeedforward {
            hyper_connection: prepare_hyper_connection(backend, &weights.feedforward.hyper_connection)?,
            norm: prepare_f32_tensor(backend, &weights.feedforward.norm)?,
            router,
            shared_gate: backend.prepare_weight(LinearWeight::block_fp8(&weights.feedforward.shared_expert.gate), config.expert_intermediate_size, config.hidden_size)?,
            shared_up: backend.prepare_weight(LinearWeight::block_fp8(&weights.feedforward.shared_expert.up), config.expert_intermediate_size, config.hidden_size)?,
            shared_down: backend.prepare_weight(LinearWeight::block_fp8(&weights.feedforward.shared_expert.down), config.hidden_size, config.expert_intermediate_size)?,
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
    let compressor = (ratio != 0).then(|| prepare_gguf_compressor(backend, source, &prefix, "attn_compressor", ratio, config.head_dim)).transpose()?;
    let indexer = (ratio == 4)
        .then(|| {
            Ok(PreparedIndexer {
                query: matrix("indexer.attn_q_b.weight", config.index_heads * config.index_head_dim, config.q_lora_rank)?,
                head_weights: matrix("indexer.proj.weight", config.index_heads, config.hidden_size)?,
                compressor: prepare_gguf_compressor(backend, source, &prefix, "indexer_compressor", ratio, config.index_head_dim)?,
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
        PreparedRouter::ScoreTopK { weight: router_weight, correction_bias: vector("exp_probs_b.bias", config.expert_count)? }
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
    let (key_value, gate) = prepare_gguf_matrix_pair(backend, source, &format!("{prefix}.{name}_kv.weight"), &format!("{prefix}.{name}_gate.weight"), channels, source.config().hidden_size)?;
    Ok(PreparedCompressor {
        position: prepare_gguf_vector(backend, source, &format!("{prefix}.{name}_ape.weight"), ratio * channels)?,
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
        input_norm: prepare_unit_norm(backend, expanded)?,
        function: prepare_f32_tensor(backend, &head.hyper_connection.function)?,
        base: prepare_f32_tensor(backend, &head.hyper_connection.base)?,
        scale: prepare_f32_tensor(backend, &head.hyper_connection.scale)?,
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
    Ok(DeepSeekV4OutputHead {
        input_norm: prepare_unit_norm(backend, expanded)?,
        function: prepare_gguf_dense_matrix(backend, source, "output_hc_fn.weight", copies, expanded)?,
        base: prepare_gguf_vector(backend, source, "output_hc_base.weight", copies)?,
        scale: prepare_gguf_vector(backend, source, "output_hc_scale.weight", 1)?,
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
    let mixes = backend.linear(&normalized, &head.function)?;
    backend.hyper_connection_head_reduce(hidden, &mixes, &head.base, &head.scale, config.hyper_connection_copies, config.hyper_connection_eps)
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
{
    deepseek_v4_forward_impl(backend, model, caches, experts, Some(layer_cache), input, rope, position, token_ids, |backend, layer| prepare_deepseek_v4_gguf_layer(backend, source, layer))
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
{
    deepseek_v4_forward_impl(backend, model, caches, experts, Some(layer_cache), input, rope, position, token_ids, |backend, layer| prepare_deepseek_v4_layer(backend, source, layer))
}

#[allow(clippy::too_many_arguments)]
fn deepseek_v4_forward_impl<B, F>(
    backend: &B,
    model: &DeepSeekV4,
    caches: &mut [B::CompressedKvStorage],
    experts: &mut B::PrefillExperts,
    mut layer_cache: Option<&mut DeepSeekV4LayerCache<B::Weight>>,
    input: &B::Tensor,
    rope: &DeepSeekV4RopeTables,
    position: usize,
    token_ids: &[u32],
    prepare: F,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + CompressedSparseKernel + HyperConnectionKernel,
    F: Fn(&B, usize) -> Result<DeepSeekV4PreparedLayer<B::Weight>, BackendError>,
{
    let rows = backend.token_rows(input);
    let cached_layers = layer_cache.as_ref().map_or(model.layer_count(), |cache| cache.layers.len());
    if rows == 0 || rows != token_ids.len() || backend.token_cols(input) != model.config().hidden_size || caches.len() != model.layer_count() || cached_layers != model.layer_count() {
        return Err(crate::runtime::compute_error(format!("DeepSeek-V4 forward 输入非法: shape=[{},{}] tokens={} caches={} layer_cache={cached_layers}", rows, backend.token_cols(input), token_ids.len(), caches.len())));
    }
    let positions = (position..position.checked_add(rows).ok_or_else(|| crate::runtime::compute_error("DeepSeek-V4 position 溢出"))?).collect::<Vec<_>>();
    let result = (|| {
        let mut hidden = backend.hyper_connection_expand(input, model.config().hyper_connection_copies)?;
        for (layer, kv_cache) in caches.iter_mut().enumerate() {
            let _scope = backend.layer_scope();
            if rows == 1 {
                backend.begin_decode_batch();
            } else {
                backend.begin_batch();
            }
            let spec = model.layer_spec(layer).map_err(|error| crate::runtime::compute_error(error.to_string()))?;
            if let Some(cache) = layer_cache.as_deref_mut()
                && cache.resident[layer]
            {
                if cache.layers[layer].is_none() {
                    cache.layers[layer] = Some(prepare(backend, layer)?);
                }
                let prepared = cache.layers[layer].as_ref().ok_or_else(|| crate::runtime::compute_error(format!("DeepSeek-V4 L{layer} 常驻权重未加载")))?;
                hidden = deepseek_v4_prefill_layer(backend, model.config(), spec, prepared, kv_cache, experts, layer, &hidden, rope.layer(spec), &positions, token_ids, true, &spec.hyper_connection)?;
            } else {
                let prepared = prepare(backend, layer)?;
                hidden = deepseek_v4_prefill_layer(backend, model.config(), spec, &prepared, kv_cache, experts, layer, &hidden, rope.layer(spec), &positions, token_ids, true, &spec.hyper_connection)?;
            }
            backend.submit_batch();
        }
        Ok(hidden)
    })();
    backend.finish_batch();
    result
}

fn prepare_attention<B: Backend>(backend: &B, config: &DeepSeekV4Config, weights: &DeepSeekV4AttentionWeights) -> Result<PreparedAttention<B::Weight>, BackendError> {
    Ok(PreparedAttention {
        sink: prepare_tensor(backend, &weights.sink)?,
        query_input: backend.prepare_weight(LinearWeight::block_fp8(&weights.query.input_projection), config.q_lora_rank, config.hidden_size)?,
        query_norm: prepare_dense_tensor(backend, &weights.query.norm)?,
        query_output: backend.prepare_weight(LinearWeight::block_fp8(&weights.query.output_projection), config.num_heads * config.head_dim, config.q_lora_rank)?,
        query_head_norm: prepare_unit_norm(backend, config.head_dim)?,
        key_value: backend.prepare_weight(LinearWeight::block_fp8(&weights.key_value.projection), config.num_kv_heads * config.head_dim, config.hidden_size)?,
        key_norm: prepare_dense_tensor(backend, &weights.key_value.norm)?,
        output_inputs: backend.prepare_grouped_block_fp8(&weights.output.input_projection, config.output_groups, config.output_lora_rank)?,
        output: backend.prepare_weight(LinearWeight::block_fp8(&weights.output.output_projection), config.hidden_size, config.output_groups * config.output_lora_rank)?,
        compressor: weights.compressor.as_ref().map(|compressor| prepare_compressor(backend, compressor)).transpose()?,
        indexer: weights
            .indexer
            .as_ref()
            .map(|indexer| {
                Ok(PreparedIndexer {
                    query: backend.prepare_weight(LinearWeight::block_fp8(&indexer.query_projection), config.index_heads * config.index_head_dim, config.q_lora_rank)?,
                    head_weights: prepare_dense_tensor(backend, &indexer.head_weights_projection)?,
                    compressor: prepare_compressor(backend, &indexer.compressor)?,
                })
            })
            .transpose()?,
    })
}

fn prepare_compressor<B: Backend>(backend: &B, weights: &crate::weight::model::deepseek_v4::DeepSeekV4CompressorWeights) -> Result<PreparedCompressor<B::Weight>, BackendError> {
    Ok(PreparedCompressor {
        position: prepare_tensor(backend, &weights.position)?,
        key_value: prepare_dense_tensor(backend, &weights.key_value_projection)?,
        gate: prepare_dense_tensor(backend, &weights.gate_projection)?,
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
        feedforward_prefill(backend, spec, &weights.feedforward, experts, layer, input, token_ids)
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
    let output = feedforward_prefill(backend, spec, &weights.feedforward, experts, layer, &normalized, &token_ids)?;
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
            let (kv, gate) = backend.dual_linear(input, &compressor.key_value, &compressor.gate)?;
            Some(backend.compress_gated(
                cache,
                CompressionStream::Attention,
                positions,
                &kv,
                &gate,
                &compressor.position,
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

    let mut index_query = None;
    let mut index_head_weights = None;
    let mut compressed_index = None;
    backend.profile_device_operator("attn_index")?;
    match (spec.attention.compression.map(|compression| compression.selection), weights.indexer.as_ref()) {
        (Some(crate::attention::compressed_sparse::CompressedSelection::LearnedIndexer(indexer_spec)), Some(indexer)) => {
            let query = backend.linear(&query_lora, &indexer.query)?;
            index_query = Some(backend.rope(&query, indexer_spec.num_heads, indexer_spec.rope_dim, RotaryLayout::Interleaved, positions[0], &rope.table.cos, &rope.table.sin)?);
            index_head_weights = Some(backend.linear(input, &indexer.head_weights)?);
            let (kv, gate) = backend.dual_linear(input, &indexer.compressor.key_value, &indexer.compressor.gate)?;
            compressed_index = Some(backend.compress_gated(
                cache,
                CompressionStream::Indexer,
                positions,
                &kv,
                &gate,
                &indexer.compressor.position,
                &indexer.compressor.norm,
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
            let (kv, gate) = backend.dual_linear(input, &compressor.key_value, &compressor.gate)?;
            let mut state_segments = segments.iter_mut().map(|segment| crate::attention::compressed_sparse::CompressedGatedSegment { storage: &mut *segment.cache, positions: segment.positions }).collect::<Vec<_>>();
            Some(backend.compress_gated_segmented(
                &mut state_segments,
                CompressionStream::Attention,
                &kv,
                &gate,
                &compressor.position,
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
            let (kv, gate) = backend.dual_linear(input, &indexer.compressor.key_value, &indexer.compressor.gate)?;
            let mut state_segments = segments.iter_mut().map(|segment| crate::attention::compressed_sparse::CompressedGatedSegment { storage: &mut *segment.cache, positions: segment.positions }).collect::<Vec<_>>();
            compressed_index = Some(backend.compress_gated_segmented(
                &mut state_segments,
                CompressionStream::Indexer,
                &kv,
                &gate,
                &indexer.compressor.position,
                &indexer.compressor.norm,
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
) -> Result<B::Tensor, BackendError> {
    let selected;
    let (router_weight, router_bias, selected_experts) = match &weights.router {
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
            (weight, zero_bias, Some(selected.as_slice()))
        }
        PreparedRouter::ScoreTopK { weight, correction_bias } => (weight, correction_bias, None),
    };
    let shared = [crate::moe::topk_moe::SharedExpertRef { gate: &weights.shared_gate, up: &weights.shared_up, down: &weights.shared_down, output_gate: None }];
    let moe = crate::moe::topk_moe::MoeFfnRef { router_weight, router_bias, shared_experts: &shared, selected_experts };
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
        compressed_sparse::{CompressedSelection, CompressedSparseAttentionSpec, KvCompressionSpec},
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
        Ok(Self { config, layer_specs })
    }

    pub fn flash() -> Self {
        Self::new(DeepSeekV4Config::flash()).expect("DeepSeek-V4-Flash 官方配置必须有效")
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
            4 => Some(KvCompressionSpec {
                ratio,
                overlap: true,
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

    /// 单层 KV 每 token 的线性增长字节（Q8g64 CSA 准入口径）。K/V 共享一份
    /// Q8 行加 u16 group scale；LearnedIndexer 压缩层每个 compressed 行另存
    /// index_head_dim 个 f32；compressed 行数按 ratio 随 token 折减。ratio=0
    /// 层只有有界 recent ring，线性增长为 0。recent ring、batch 工作区等每会话
    /// 固定开销不在线性项里，由节点上报时的 memory_reserve_bytes 覆盖。
    pub fn kv_bytes_per_layer_token(&self, layer: LayerId) -> usize {
        let attention = &self.layer_specs[layer].attention;
        let Some(compression) = attention.compression else { return 0 };
        let kv_width = attention.num_kv_heads * attention.head_dim;
        // 与 RocmCompressedKvStorage 的组宽选择保持一致：取能整除 head_dim 的最大组宽。
        let group = [crate::kv_cache::DEFAULT_GROUP_SIZE, 32, 16, 8, 4, 2, 1].into_iter().find(|&group| attention.head_dim.is_multiple_of(group)).unwrap_or(1);
        let row_bytes = kv_width + kv_width / group * 2;
        let index_bytes = matches!(compression.selection, crate::attention::compressed_sparse::CompressedSelection::LearnedIndexer(_)).then_some(self.config.index_head_dim * 4).unwrap_or(0);
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
    fn score_expert_top_k_preserves_token_hash_width() {
        let model = DeepSeekV4::flash().with_score_expert_top_k(5).unwrap();
        assert_eq!(model.config().expert_top_k, 6);
        assert_eq!(model.layer_spec(0).unwrap().feedforward.top_k, 6);
        assert_eq!(model.layer_spec(3).unwrap().feedforward.top_k, 5);
    }
}
