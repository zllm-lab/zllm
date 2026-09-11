//! MiniCPM5 风格 dense LLM 平台无关运行时。
//!
//! 当前已验证：MiniCPM5-1B-Instruct Q4_K_M GGUF。
//! 解码器形态 = Llama 风格：
//!   pre-norm → QKV proj → RoPE → causal GQA → out_proj + residual
//!           → pre-norm → SwiGLU → + residual
//!
//! 与 Mistral 的差异（架构本身几乎一致，主要在 chat template 与 EOS）：
//! - chat template 是 ChatML：`<s><|im_start|>user\n{...}<|im_end|>\n<|im_start|>assistant\n`
//!   （Mistral 是 `<s>[INST] {prompt} [/INST]`；两者都显式带 BOS）
//! - 1B GGUF 同时带 `token_embd.weight`(Q4_K)与 `output.weight`(Q6_K),untied;
//!   tied 时 `lm_head()` 复用 `token_embd.weight`,runtime 两种都接受
//! - EOS 从 GGUF metadata `tokenizer.ggml.eos_token_id` 读（默认 2 作 fallback）

pub mod protocol;
#[cfg(feature = "with-qnn")]
pub mod qnn;

use crate::{
    attention::{
        gqa::{CausalWindow, GqaSpec},
        rope::{RopeTable, RotaryLayout},
    },
    backend::{Backend, BackendError, GqaPrefillBackend, LinearWeight},
    moe::Activation,
    weight::model::minicpm5::{MiniCpm5Config, MiniCpm5LayerWeights, MiniCpm5Weights},
};

/// 单层 MiniCPM5 解码器已 prepare 的权重（backend-resident）。
pub struct MiniCpm5TextLayer<W> {
    pub input_norm: W,
    pub query: W,
    pub key: W,
    pub value: W,
    pub output: W,
    pub post_attention_norm: W,
    pub gate: W,
    pub up: W,
    pub down: W,
}

pub type MiniCpm5OutputHead<W> = super::output::OutputHead<W>;

/// 把 host `MiniCpm5LayerWeights` (GGUF lazy matrices) 喂给 backend → resident device weights。
pub fn prepare_minicpm5_layer<B: Backend>(backend: &B, weights: &MiniCpm5LayerWeights) -> Result<MiniCpm5TextLayer<B::Weight>, BackendError> {
    let rows = 1;
    Ok(MiniCpm5TextLayer {
        input_norm: backend.prepare_f32(&weights.input_norm, rows, weights.input_norm.len())?,
        query: backend.prepare_weight(LinearWeight::gguf(&weights.query), weights.query.rows, weights.query.columns)?,
        key: backend.prepare_weight(LinearWeight::gguf(&weights.key), weights.key.rows, weights.key.columns)?,
        value: backend.prepare_weight(LinearWeight::gguf(&weights.value), weights.value.rows, weights.value.columns)?,
        output: backend.prepare_weight(LinearWeight::gguf(&weights.output), weights.output.rows, weights.output.columns)?,
        post_attention_norm: backend.prepare_f32(&weights.post_attention_norm, rows, weights.post_attention_norm.len())?,
        gate: backend.prepare_weight(LinearWeight::gguf(&weights.gate), weights.gate.rows, weights.gate.columns)?,
        up: backend.prepare_weight(LinearWeight::gguf(&weights.up), weights.up.rows, weights.up.columns)?,
        down: backend.prepare_weight(LinearWeight::gguf(&weights.down), weights.down.rows, weights.down.columns)?,
    })
}

/// 准备 LM head（output_norm + output 矩阵）。`output.weight` 是 GGUF 量化矩阵（tied 时
/// 实际来自 `token_embd.weight`，但 matrix 本身一致）。
pub fn prepare_minicpm5_output_head<B: Backend>(backend: &B, config: &MiniCpm5Config, weights: &MiniCpm5Weights) -> Result<MiniCpm5OutputHead<B::Weight>, BackendError> {
    prepare_minicpm5_output_head_quantized(backend, config, weights, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_minicpm5_output_head_quantized<B: Backend>(backend: &B, config: &MiniCpm5Config, weights: &MiniCpm5Weights, quantization: crate::weight::LmHeadQuantization) -> Result<MiniCpm5OutputHead<B::Weight>, BackendError> {
    let norm = weights.final_norm().map_err(crate::runtime::compute_error)?;
    let lm_head = weights.lm_head().map_err(crate::runtime::compute_error)?;
    if lm_head.rows != config.vocab_size || lm_head.columns != config.hidden_size {
        return Err(crate::runtime::compute_error(format!("MiniCPM5 lm_head shape [{},{}]，期望 [{},{}]", lm_head.rows, lm_head.columns, config.vocab_size, config.hidden_size)));
    }
    super::output::prepare_output_head_quantized(backend, &norm, LinearWeight::gguf(&lm_head), config.vocab_size, config.hidden_size, quantization)
}

/// RoPE 表，按整个 prefill 序列一次性算好。
pub fn minicpm5_rope_table(config: &MiniCpm5Config, seq_len: usize) -> RopeTable {
    RopeTable::precompute(seq_len, config.head_dim, config.rope_theta)
}

/// Prefill 一整段 prompt 的 hidden states（最后一层输出），可选同时写入 KV cache。
///
/// 输入 hidden = embeddings（`weights.embedding_rows_f32(&tokens)` → 后端 tensor）。
/// 输出 hidden = 最后一层第 N 个 token 的 hidden（行数 = tokens.len()）。
/// `layers` 必须已按层 prepare 完毕（decode 每 token 复用，不能逐层重传：
/// 2026-08-26 实测逐 token 重传 24 层权重 = 12× 减速）。
///
/// `cache` = `Some` 时，KV 自 `position` 行起写入（要求 `cache` 已分配 ≥ position + tokens.len() 行）；
/// `cache` = `None` 时不写 KV，纯文本编码场景（FLUX.2 仅取 last hidden）走这条。
pub fn minicpm5_text_hidden<B: GqaPrefillBackend>(
    backend: &B,
    config: &MiniCpm5Config,
    layers: &[MiniCpm5TextLayer<B::Weight>],
    cache: Option<&mut B::Cache>,
    hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B::Tensor: Clone,
{
    Ok(minicpm5_text_hidden_with_captures(backend, config, layers, cache, hidden, rope, position, &[])?.0)
}

/// `minicpm5_text_hidden` 的捕获变体：额外返回 `capture_layers` 里各层（按层序号，
/// 取该层输出 hidden）的 [rows, hidden] 张量，供 DSpark drafter 投影 target 上下文。
pub fn minicpm5_text_hidden_with_captures<B: GqaPrefillBackend>(
    backend: &B,
    config: &MiniCpm5Config,
    layers: &[MiniCpm5TextLayer<B::Weight>],
    mut cache: Option<&mut B::Cache>,
    mut hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
    capture_layers: &[usize],
) -> Result<(B::Tensor, Vec<B::Tensor>), BackendError>
where
    B::Tensor: Clone,
{
    let positions = backend.token_rows(&hidden);
    if positions == 0 {
        return Err(crate::runtime::compute_error(format!("MiniCPM5 prefill positions={positions}，期望非零")));
    }
    if backend.token_cols(&hidden) != config.hidden_size {
        return Err(crate::runtime::compute_error(format!("MiniCPM5 prefill hidden cols={}，期望 {}", backend.token_cols(&hidden), config.hidden_size)));
    }
    if layers.len() != config.layer_count {
        return Err(crate::runtime::compute_error(format!("MiniCPM5 prepared layers={}，期望 {}", layers.len(), config.layer_count)));
    }
    let mut captures = Vec::with_capacity(capture_layers.len());
    backend.begin_batch();
    for (layer, prepared) in layers.iter().enumerate() {
        let _scope = backend.layer_scope();
        hidden = minicpm5_layer(backend, config, cache.as_deref_mut(), layer, prepared, &hidden, rope, position)?;
        if capture_layers.contains(&layer) {
            captures.push(hidden.clone());
        }
    }
    backend.finish_batch();
    Ok((hidden, captures))
}

/// 单 token 增量推理（prefill 后用），维护 KV cache。`layers` 语义同 `minicpm5_text_hidden`。
pub fn minicpm5_decode_round<B: GqaPrefillBackend>(
    backend: &B,
    config: &MiniCpm5Config,
    layers: &[MiniCpm5TextLayer<B::Weight>],
    cache: &mut B::Cache,
    hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    minicpm5_decode_round_impl(backend, config, layers, cache, hidden, rope, position, false)
}

/// `minicpm5_decode_round` 的流水线变体:结束只恢复 backend 状态、不同步 GPU,
/// 供逐 token 异步流水线把 gather/输出步接进同一提交窗口。
pub fn minicpm5_decode_round_deferred<B: GqaPrefillBackend>(
    backend: &B,
    config: &MiniCpm5Config,
    layers: &[MiniCpm5TextLayer<B::Weight>],
    cache: &mut B::Cache,
    hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    minicpm5_decode_round_impl(backend, config, layers, cache, hidden, rope, position, true)
}

fn minicpm5_decode_round_impl<B: GqaPrefillBackend>(
    backend: &B,
    config: &MiniCpm5Config,
    layers: &[MiniCpm5TextLayer<B::Weight>],
    cache: &mut B::Cache,
    mut hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
    deferred: bool,
) -> Result<B::Tensor, BackendError> {
    if backend.token_rows(&hidden) != 1 {
        return Err(crate::runtime::compute_error(format!("MiniCPM5 decode hidden rows={}，期望 1", backend.token_rows(&hidden))));
    }
    if layers.len() != config.layer_count {
        return Err(crate::runtime::compute_error(format!("MiniCPM5 prepared layers={}，期望 {}", layers.len(), config.layer_count)));
    }
    // decode 一轮一次 begin/finish:逐层 finish_batch 会各触发一次全 GPU 同步,
    // 1B 级模型每层 GPU 工作极小,24 次同步/token 曾把 decode 压到 1/3 速度。
    backend.begin_decode_batch();
    for (layer, prepared) in layers.iter().enumerate() {
        let _scope = backend.layer_scope();
        hidden = minicpm5_layer(backend, config, Some(cache), layer, prepared, &hidden, rope, position)?;
    }
    if deferred {
        backend.finish_batch_deferred();
    } else {
        backend.finish_batch();
    }
    Ok(hidden)
}

/// 一次性 prepare 全部层（常驻显存），供 prefill/decode 反复使用。
pub fn prepare_minicpm5_layers<B: Backend>(backend: &B, weights: &MiniCpm5Weights) -> Result<Vec<MiniCpm5TextLayer<B::Weight>>, BackendError> {
    (0..weights.config().layer_count)
        .map(|layer| {
            let source = weights.load_layer(layer).map_err(crate::runtime::compute_error)?;
            prepare_minicpm5_layer(backend, &source)
        })
        .collect()
}

/// 从 GGUF metadata `tokenizer.ggml.eos_token_id` 读出 EOS，缺失时回落 2（Llama 风格）。
/// tied embedding 模式下 `output.weight` 不存在，但 eos token id 与 tied 无关。
pub fn minicpm5_eos_token_id(weights: &MiniCpm5Weights) -> u32 {
    use crate::weight::container::gguf::GgufValue;
    weights.reader().metadata("tokenizer.ggml.eos_token_id").and_then(GgufValue::as_u64).map(|v| v as u32).unwrap_or(2)
}

/// MiniCPM5 ChatML 标准 prompt（不依赖 tokenizer chat template 解析，纯字面拼接）。
///
/// MiniCPM5-1B 标准模板：`<s><|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`
/// 当 `system` 非空时插入 system 段。**BOS `<s>` 必须有**：缺失时模型输出空白/乱token
/// （2026-08-26 实测，llama.cpp 无 BOS 直出同样为空）。
///
/// MiniCPM5 是双模式思考模型：assistant 段必须打开 `<think>` 块，否则模型把思考
/// 独白当正文裸写（没有 `</think>` 分界、长时间不收敛，表现为循环输出）。
/// `thinking=false` 时注入空 think 块（官方 no-think 约定），模型直接作答。
pub fn minicpm5_instruct_prompt(user: &str, system: Option<&str>, thinking: bool) -> String {
    let head = match system {
        Some(system) if !system.is_empty() => format!("<s><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"),
        _ => format!("<s><|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"),
    };
    if thinking { format!("{head}<think>\n") } else { format!("{head}<think>\n\n</think>\n\n") }
}

/// Prefill 后取最后一个 token 位置（prompt 末位）作为首个输出。
pub fn minicpm5_last_token_output<B: Backend>(backend: &B, config: &MiniCpm5Config, head: &MiniCpm5OutputHead<B::Weight>, hidden: &B::Tensor, _eos_token_ids: &[u32]) -> Result<super::output::OutputResult<B::Tensor>, BackendError> {
    // EOS 必须参与采样；生成生命周期负责截断且不把停止 token 发给客户端。
    super::output::last_token_output(backend, head, hidden, backend.token_rows(hidden) - 1, &super::output::OutputPlan { eps: config.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: Vec::new() })
}

/// decode 轮末尾取 hidden 第 0 行作为输出。
pub fn minicpm5_token_output<B: Backend>(backend: &B, config: &MiniCpm5Config, head: &MiniCpm5OutputHead<B::Weight>, hidden: &B::Tensor, _eos_token_ids: &[u32]) -> Result<super::output::OutputResult<B::Tensor>, BackendError> {
    super::output::token_output(backend, head, hidden, &super::output::OutputPlan { eps: config.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: Vec::new() })
}

/// 同上，但按 temperature/top_p 采样。
/// MiniCPM5-1B 纯贪心会陷入复读循环；且 eos/im_end 必须留在候选里，
/// 否则永远采不到停止符、只能跑满 max_tokens（表现为循环输出）。
pub fn minicpm5_sampled_token_output<B>(
    backend: &B,
    config: &MiniCpm5Config,
    head: &MiniCpm5OutputHead<B::Weight>,
    hidden: &B::Tensor,
    sampling: &crate::backend::TokenSampling,
) -> Result<super::output::OutputResult<B::Tensor>, BackendError>
where
    B: Backend,
{
    let plan = super::output::OutputPlan { eps: config.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: Vec::new() };
    let (input, logits) = super::output::norm_and_lm_head(backend, head, hidden, &plan)?;
    let token_id = backend.sample_top_p(&logits, sampling.temperature, sampling.top_p, sampling.random)?;
    Ok(super::output::OutputResult { input, logits, token_id })
}

/// MiniCPM5 单层 forward（Llama 风格，无 QK-norm，与 Mistral 同构）。
#[allow(clippy::too_many_arguments)]
fn minicpm5_layer<B: GqaPrefillBackend>(
    backend: &B,
    config: &MiniCpm5Config,
    cache: Option<&mut B::Cache>,
    layer: usize,
    weights: &MiniCpm5TextLayer<B::Weight>,
    hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    // 1. pre-norm + QKV projections
    let normed = backend.rmsnorm(hidden, &weights.input_norm, config.rms_eps)?;
    let (query, key) = backend.dual_linear(&normed, &weights.query, &weights.key)?;
    let value = backend.linear(&normed, &weights.value)?;
    // 2. RoPE on Q/K (Interleaved layout, full head_dim as rotary dim);decode 单行合并一次 dispatch
    let (query, key) = backend.rope_prefix_qk(&query, &key, config.num_heads, config.num_kv_heads, config.head_dim, RotaryLayout::Interleaved, position, &rope.cos, &rope.sin)?;
    // 3. Causal GQA attention (no QK-norm, MiniCPM5 默认)
    let spec = GqaSpec {
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        rope_dim: config.head_dim,
        rope_theta: config.rope_theta,
        use_qk_norm: false,
        window: CausalWindow::Full,
        score_scale: 1.0 / (config.head_dim as f32).sqrt(),
        output_gate: false,
    };
    let attention = match cache {
        Some(cache) => backend.gqa_prefill_attention_cached(cache, layer, position, &query, &key, &value, &spec, false)?,
        None => backend.gqa_prefill_attention(&query, &key, &value, &spec)?,
    };
    // 4. output projection + residual
    let attention_residual = backend.linear_add(&attention, &weights.output, hidden)?;
    // 5. pre-norm + SwiGLU + 残差(down 投影 epilogue 直接加回,省独立 add dispatch)
    let normed = backend.rmsnorm(&attention_residual, &weights.post_attention_norm, config.rms_eps)?;
    backend.gated_mlp_add_residual(&normed, &weights.gate, &weights.up, &weights.down, &Activation::Silu, &attention_residual)
}

// Node 适配器：组合 MiniCPM5 平台无关运行时与 Metal / CPU 后端资源，匹配 zllm 通用模式。
// macOS 走 Metal 路径，非 macOS 暴露占位 `NodeEngine` 提示当前平台不支持。
pub mod cpu_node;
#[cfg(target_os = "macos")]
pub mod dspark;
pub mod engine;
pub mod node;
