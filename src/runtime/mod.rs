//! 模型运行时：模型架构规格（Config/Spec）与平台无关的执行编排。
//!
//! 每个模型的 Config + LayerSpec + forward 编排直接放在模型文件；只有模型确实拥有
//! 平台组合或独立子能力时才使用同名目录，遵循“定义靠近使用点”。

pub mod chat_template;
pub mod deepseek_v4;
pub mod dspark;
pub mod expert_pipeline;
pub mod gemma4;
pub(crate) mod generation;
// generation_guard 供所有平台的 embedded 通用循环围栏接线（linux rocm node
// 另有 fence 采样联动）；不再是 linux 专属
pub(crate) mod generation_guard;
pub mod glm52;
pub mod glm53_flash;
pub mod h3;
#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
pub(crate) mod json_fence;
pub mod kimi_k3;
#[cfg(target_os = "macos")]
pub(crate) mod metal_node;
pub mod minimax_m3;
pub mod mistral;
// 向后兼容：旧 `mistral_small32` 模块名映射到 `mistral`。`MistralSmall32*` 类型在
// `runtime::mistral` 末尾以 `pub type` alias 形式保留，所有旧 import 继续工作。
pub mod mistral_small32 {
    pub use crate::runtime::mistral::*;
}
pub mod minicpm5;
pub mod multiplex;
pub mod node;
pub mod ornith;
pub mod output;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(crate) mod pipeline;
pub mod prefill;
mod prefill_admission;
mod prefill_scheduler;
pub mod qwen36;
pub mod qwen3_vl;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_chain;
pub mod session;
pub mod speculative;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(crate) mod stage_artifact;
pub mod tool;

pub const DEFAULT_DECODE_PREFETCH_COUNT: usize = 2;

/// 所有平台入口共用模型声明的上下文边界，避免 standalone/Node/backend 各自漏掉 0 或上限。
pub fn validate_max_sequence_length(model: &str, configured: usize, supported: usize) -> Result<(), String> {
    if configured == 0 || configured > supported {
        return Err(format!("{model} max_sequence_length={configured} 超出模型范围 1..={supported}"));
    }
    Ok(())
}

use crate::attention::AttentionSpec;
use crate::moe::FeedforwardSpec;
use crate::norm::NormSpec;

/// Token id。
pub type TokenId = u32;

/// 层 id。
pub type LayerId = usize;

/// 一层 Transformer 的架构规格。描述该层用什么注意力、什么前馈、什么归一化。
#[derive(Debug)]
pub struct LayerSpec {
    pub attention: AttentionSpec,
    pub feedforward: FeedforwardSpec,
    pub input_norm: NormSpec,
    pub post_attention_norm: NormSpec,
    pub post_norm: Option<NormSpec>,
}

/// 模型架构 trait：描述执行图，与硬件无关。
pub trait Model {
    type Config;

    fn config(&self) -> &Self::Config;

    fn layer_count(&self) -> usize;

    fn layer_spec(&self, layer: LayerId) -> Result<&LayerSpec, ModelError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("invalid model architecture: {0}")]
    InvalidArchitecture(String),
    #[error("layer {layer} out of range (layer_count={layer_count})")]
    LayerOutOfRange { layer: LayerId, layer_count: usize },
    #[error("model input is empty")]
    EmptyInput,
}

// ============================================================================
// GGUF 权重 prepare 共享函数 —— 所有 GGUF 模型 wrapper 复用。
// 桥接 weight::GgufReader 和 backend::Backend,放在 runtime 层(依赖方向合法)。
// ============================================================================

use crate::backend::{Backend, BackendError, LinearWeight};
use crate::weight::{
    ResidentWeightQuantization,
    codec::groupwise::{quantize_bf16_rows_w8a16, quantize_f16_rows_w8a16, quantize_f32_rows_w8a16},
    container::gguf::GgufReader,
    format::quantization::QuantizedMatrixRef,
};

/// 把标准矩阵转换为配置指定的 resident 格式，再交给 backend 准备。
/// 模型只提供来源、shape 和用途名，不复制格式转换或按平台分支。
pub fn prepare_resident_matrix<B: Backend>(backend: &B, source: LinearWeight<'_>, rows: usize, columns: usize, quantization: ResidentWeightQuantization, name: &str) -> Result<B::Weight, BackendError> {
    if quantization == ResidentWeightQuantization::Native {
        if let LinearWeight::Quantized(QuantizedMatrixRef::Gguf(matrix)) = source
            && matches!(matrix.tensor_type.0, 0 | 1)
        {
            let values = matrix.decode().map_err(|msg| compute_error(format!("解码 {name} {}: {msg}", matrix.tensor_type.name())))?;
            return backend.prepare_weight(LinearWeight::F32(&values), rows, columns);
        }
        return backend.prepare_weight(source, rows, columns);
    }
    if let LinearWeight::Quantized(QuantizedMatrixRef::W8A16(matrix)) = source
        && matrix.group_size() == 128
    {
        return backend.prepare_weight(source, rows, columns);
    }
    let started = std::time::Instant::now();
    let row_count = u32::try_from(rows).map_err(|_| compute_error(format!("{name} rows={rows} 超过 u32")))?;
    let selected_rows = (0..row_count).collect::<Vec<_>>();
    let matrix = match source {
        LinearWeight::Bf16Bytes(values) => quantize_bf16_rows_w8a16(values, rows, columns, &selected_rows, 128),
        LinearWeight::F16(values) => quantize_f16_rows_w8a16(values, rows, columns, &selected_rows, 128),
        LinearWeight::F32(values) => quantize_f32_rows_w8a16(values, rows, columns, &selected_rows, 128),
        LinearWeight::Quantized(matrix) => {
            let values = matrix.decode().map_err(|msg| compute_error(format!("解码 {name} {}: {msg}", matrix.name())))?;
            quantize_f32_rows_w8a16(&values, rows, columns, &selected_rows, 128)
        }
    }
    .map_err(|msg| compute_error(format!("量化 {name}: {msg}")))?;
    eprintln!("[resident-weight] name={name} quantization=q8g128 rows={rows} columns={columns} wall={:.3}s", started.elapsed().as_secs_f64());
    backend.prepare_weight(LinearWeight::w8a16(&matrix), rows, columns)
}

/// F32/F16 矩阵解量化后走 F32 路径,量化块保留给 backend resident 在线 dequant。
pub fn prepare_gguf_matrix<B: Backend>(backend: &B, reader: &GgufReader, name: &str) -> Result<B::Weight, BackendError> {
    let matrix = reader.read_matrix(name).map_err(gguf_weight_error)?;
    if matches!(matrix.tensor_type.0, 0 | 1) {
        let values = matrix.decode().map_err(gguf_weight_error)?;
        backend.prepare_weight(LinearWeight::F32(&values), matrix.rows, matrix.columns)
    } else {
        backend.prepare_weight(LinearWeight::gguf(&matrix), matrix.rows, matrix.columns)
    }
}

/// 同 shape 的相关 GGUF 权重一起交给 backend 准备；量化 backend 可据此选择
/// 不同但互补的 resident 布局，普通 backend 仍走默认的两次 prepare。
pub fn prepare_gguf_matrix_pair<B: Backend>(backend: &B, reader: &GgufReader, first_name: &str, second_name: &str) -> Result<(B::Weight, B::Weight), BackendError> {
    let first = reader.read_matrix(first_name).map_err(gguf_weight_error)?;
    let second = reader.read_matrix(second_name).map_err(gguf_weight_error)?;
    if first.rows != second.rows || first.columns != second.columns {
        return Err(compute_error(format!("GGUF 权重对 shape 不一致: {first_name}=[{},{}] {second_name}=[{},{}]", first.rows, first.columns, second.rows, second.columns)));
    }
    if matches!(first.tensor_type.0, 0 | 1) && matches!(second.tensor_type.0, 0 | 1) {
        let first_values = first.decode().map_err(gguf_weight_error)?;
        let second_values = second.decode().map_err(gguf_weight_error)?;
        backend.prepare_weight_pair(LinearWeight::F32(&first_values), LinearWeight::F32(&second_values), first.rows, first.columns)
    } else if !matches!(first.tensor_type.0, 0 | 1) && !matches!(second.tensor_type.0, 0 | 1) {
        backend.prepare_weight_pair(LinearWeight::gguf(&first), LinearWeight::gguf(&second), first.rows, first.columns)
    } else {
        Ok((prepare_gguf_matrix(backend, reader, first_name)?, prepare_gguf_matrix(backend, reader, second_name)?))
    }
}

/// 纯 F32 矩阵(decode 后丢弃 GGUF 量化形态;backend kernel 要求 F32 输入)。
pub fn prepare_gguf_f32_matrix<B: Backend>(backend: &B, reader: &GgufReader, name: &str) -> Result<B::Weight, BackendError> {
    require_dense_gguf_tensor(reader, name)?;
    let matrix = reader.read_matrix(name).map_err(gguf_weight_error)?;
    let values = matrix.decode().map_err(gguf_weight_error)?;
    backend.prepare_f32(&values, matrix.rows, matrix.columns)
}

/// F32 矩阵 decode 后转 F16 resident:Metal 走 F16 gemv 直通,避免 F32 权重路径
/// 每层 f16→f32 cast + MPS f32 matmul + f32→f16 回 cast(gemma4 同款模式)。
/// alpha/beta 经 softplus/sigmoid 有界化,F16 权重误差在递归状态可接受范围内。
pub fn prepare_gguf_f16_matrix<B: Backend>(backend: &B, reader: &GgufReader, name: &str) -> Result<B::Weight, BackendError> {
    require_dense_gguf_tensor(reader, name)?;
    let matrix = reader.read_matrix(name).map_err(gguf_weight_error)?;
    let values = matrix.decode().map_err(gguf_weight_error)?;
    let packed: Vec<half::f16> = values.iter().map(|&value| half::f16::from_f32(value)).collect();
    backend.prepare_weight(LinearWeight::F16(&packed), matrix.rows, matrix.columns)
}

/// GemmaRMS norm 向量(GGUF +1 存储还原为架构语义)。
pub fn prepare_gguf_gemma_vector<B: Backend>(backend: &B, reader: &GgufReader, name: &str) -> Result<B::Weight, BackendError> {
    require_dense_gguf_tensor(reader, name)?;
    let values = reader.gemma_norm_vector(name).map_err(gguf_weight_error)?;
    backend.prepare_gemma_f32(&values, 1, values.len())
}

/// GemmaRMS norm 向量的 F16 resident 版：Metal 的 F16-weight kernel 原生直通，
/// 避免 F32 权重路径每层两次 f16↔f32 cast（gemma4 同款模式）。
pub fn prepare_gguf_gemma_vector_f16<B: Backend>(backend: &B, reader: &GgufReader, name: &str) -> Result<B::Weight, BackendError> {
    require_dense_gguf_tensor(reader, name)?;
    let values = reader.gemma_norm_vector(name).map_err(gguf_weight_error)?;
    let packed: Vec<half::f16> = values.iter().map(|&value| half::f16::from_f32(value)).collect();
    backend.prepare_weight(crate::backend::LinearWeight::F16(&packed), 1, values.len())
}

/// F32 向量保留 dtype上传(backend kernel 要求 F32,如 dt_bias)。
pub fn prepare_gguf_f32_vector<B: Backend>(backend: &B, reader: &GgufReader, name: &str) -> Result<B::Weight, BackendError> {
    require_dense_gguf_tensor(reader, name)?;
    let values = reader.read_tensor_f32(name).map_err(gguf_weight_error)?;
    backend.prepare_f32(&values, 1, values.len())
}

/// SSM A_log 向量(GGUF -exp(A_log) 还原为 A_log)。
pub fn prepare_gguf_a_log_vector<B: Backend>(backend: &B, reader: &GgufReader, name: &str) -> Result<B::Weight, BackendError> {
    require_dense_gguf_tensor(reader, name)?;
    let values = reader.a_log_vector(name).map_err(gguf_weight_error)?;
    backend.prepare_f32(&values, 1, values.len())
}

/// 把任意可转 String 的错误包装成 `BackendError::Compute`，统一各模型的错误转换。
/// 取代散落在各 runtime 文件里的同名 `fn compute` 局部 helper。
pub fn compute_error(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}

fn gguf_weight_error(error: String) -> BackendError {
    compute_error(error)
}

fn require_dense_gguf_tensor(reader: &GgufReader, name: &str) -> Result<(), BackendError> {
    let tensor = reader.tensor(name).ok_or_else(|| compute_error(format!("GGUF 缺少 tensor {name}")))?;
    if !matches!(tensor.tensor_type.0, 0 | 1 | 30) {
        return Err(compute_error(format!("GGUF tensor {name}={} 是量化格式，但该模型位置没有原生量化算子", tensor.tensor_type.name())));
    }
    Ok(())
}

/// MTP 输入投影的公共前半段：shape 校验 → norm(embedding) → norm(hidden) → concat → input_projection。
///
/// GLM-5.2 / Qwen3.6 / Ornith 的 MTP 层前向共享此流程，仅中间的 attention/FFN 层不同。
/// `norm` 决定用 Rms 还是 GemmaRMS 归一化；`f32_norm` 为 true 时走精度敏感的 `*_f32` 变体
/// （Qwen3.6 / Ornith 的 MTP 在 Metal 上要求 F32 归一化）。
/// `hidden_size` 校验 embedding/hidden 的列宽，防止 concat 后宽度与 input_projection 不匹配。
#[allow(clippy::too_many_arguments)]
pub fn mtp_project<B: Backend>(
    backend: &B,
    embedding: &B::Tensor,
    hidden: &B::Tensor,
    embedding_norm: &B::Weight,
    hidden_norm: &B::Weight,
    input_projection: &B::Weight,
    hidden_size: usize,
    norm: NormSpec,
    f32_norm: bool,
) -> Result<B::Tensor, BackendError> {
    let rows = backend.token_rows(embedding);
    if rows == 0 || backend.token_rows(hidden) != rows || backend.token_cols(embedding) != hidden_size || backend.token_cols(hidden) != hidden_size {
        return Err(compute_error(format!("MTP 输入 shape 非法: embedding=[{},{}], hidden=[{},{}]，期望行数相同且列宽为 {hidden_size}", rows, backend.token_cols(embedding), backend.token_rows(hidden), backend.token_cols(hidden),)));
    }
    let (eps, is_gemma) = match norm {
        NormSpec::Rms { eps } => (eps, false),
        NormSpec::GemmaRms { eps } => (eps, true),
        NormSpec::AdaLn { .. } => return Err(compute_error("MTP 不支持 AdaLn 归一化")),
    };
    let normed_embedding = match (is_gemma, f32_norm) {
        (false, false) => backend.rmsnorm(embedding, embedding_norm, eps)?,
        (false, true) => backend.rmsnorm_f32(embedding, embedding_norm, eps)?,
        (true, false) => backend.gemma_rmsnorm(embedding, embedding_norm, eps)?,
        (true, true) => backend.gemma_rmsnorm_f32(embedding, embedding_norm, eps)?,
    };
    let normed_hidden = match (is_gemma, f32_norm) {
        (false, false) => backend.rmsnorm(hidden, hidden_norm, eps)?,
        (false, true) => backend.rmsnorm_f32(hidden, hidden_norm, eps)?,
        (true, false) => backend.gemma_rmsnorm(hidden, hidden_norm, eps)?,
        (true, true) => backend.gemma_rmsnorm_f32(hidden, hidden_norm, eps)?,
    };
    let fused = backend.concat_columns(&normed_embedding, &normed_hidden)?;
    backend.linear(&fused, input_projection)
}
