//! Mistral 风格 dense LLM 平台无关运行时。
//!
//! 当前已验证：Mistral-Small-3.2-24B-Instruct-2506（Q6_K / Q8_0 GGUF）。
//! 解码器形态 = Llama 风格：
//!   pre-norm → QKV proj → RoPE → causal GQA → out_proj + residual
//!           → pre-norm → SwiGLU → + residual
//!
//! 与 qwen3_vl 的差异（Mistral 默认无 QK-norm）：
//! - 无 QK-norm（`GqaSpec.use_qk_norm = false`）
//! - q/k/v 独立权重，通过 `triple_linear` 让 backend 选择是否合并 dispatch
//! - 无 vision tower（Mistral-3.2 文本分支；未来 VLM 接入预留 `vision.rs` 子模块）
//!
//! 模板：`src/runtime/qwen3_vl.rs::text_layer` (715-737)，删掉 `gemma_rmsnorm_heads`
//! 两行。

use crate::{
    attention::{
        gqa::{CausalWindow, GqaSpec},
        rope::{RopeSpec, RopeTable, RotaryLayout},
    },
    backend::{Backend, BackendError, GqaPrefillBackend, LinearWeight},
    moe::Activation,
    weight::model::mistral::{MistralConfig, MistralLayerWeights, MistralWeights},
};

/// 单层 Mistral 解码器已 prepare 的权重（backend-resident）。
pub struct MistralTextLayer<W> {
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

pub type MistralOutputHead<W> = super::output::OutputHead<W>;

/// 把 host `MistralLayerWeights` (GGUF lazy matrices) 喂给 backend → resident device weights。
pub fn prepare_mistral_layer<B: Backend>(backend: &B, weights: &MistralLayerWeights) -> Result<MistralTextLayer<B::Weight>, BackendError> {
    let rows = 1;
    Ok(MistralTextLayer {
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

pub fn prepare_mistral_layers<B: Backend>(backend: &B, weights: &MistralWeights) -> Result<Vec<MistralTextLayer<B::Weight>>, BackendError> {
    (0..weights.config().layer_count)
        .map(|layer| {
            let source = weights.load_layer(layer).map_err(crate::runtime::compute_error)?;
            prepare_mistral_layer(backend, &source)
        })
        .collect()
}

/// 准备 LM head（output_norm + output 矩阵）。`output.weight` 是 GGUF 量化矩阵。
pub fn prepare_mistral_output_head<B: Backend>(backend: &B, config: &MistralConfig, weights: &MistralWeights) -> Result<MistralOutputHead<B::Weight>, BackendError> {
    prepare_mistral_output_head_quantized(backend, config, weights, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_mistral_output_head_quantized<B: Backend>(backend: &B, config: &MistralConfig, weights: &MistralWeights, quantization: crate::weight::LmHeadQuantization) -> Result<MistralOutputHead<B::Weight>, BackendError> {
    let norm = weights.final_norm().map_err(crate::runtime::compute_error)?;
    let lm_head = weights.lm_head().map_err(crate::runtime::compute_error)?;
    if lm_head.rows != config.vocab_size || lm_head.columns != config.hidden_size {
        return Err(crate::runtime::compute_error(format!("Mistral lm_head shape [{},{}]，期望 [{},{}]", lm_head.rows, lm_head.columns, config.vocab_size, config.hidden_size)));
    }
    super::output::prepare_output_head_quantized(backend, &norm, LinearWeight::gguf(&lm_head), config.vocab_size, config.hidden_size, quantization)
}

/// RoPE 表，按整个 prefill 序列一次性算好。
pub fn mistral_rope_table(config: &MistralConfig, seq_len: usize) -> RopeTable {
    let spec = match config.rope_scaling {
        Some(scaling) => RopeSpec::Yarn { rotary_dim: config.head_dim, theta: config.rope_theta, factor: scaling.factor, original_context: scaling.original_context, beta_fast: scaling.beta_fast, beta_slow: scaling.beta_slow },
        None => RopeSpec::Default { rotary_dim: config.head_dim, theta: config.rope_theta },
    };
    RopeTable::from_spec(seq_len, spec).unwrap_or_else(|error| panic!("Mistral RoPE 参数非法: {error}"))
}

/// 单 token 增量推理的 RoPE 前缀表；常驻 engine 应预计算最大长度并复用，
/// 这个便捷入口只供一次性调用。
pub fn mistral_decode_rope_table(config: &MistralConfig, position: usize) -> RopeTable {
    mistral_rope_table(config, position + 1)
}

/// Prefill 一整段 prompt 的 hidden states（最后一层输出），可选同时写入 KV cache。
///
/// 输入 hidden = embeddings（`weights.embedding_rows_f32(&tokens)` → 后端 tensor）。
/// 输出 hidden = 最后一层第 N 个 token 的 hidden（行数 = tokens.len()）。
///
/// `cache` = `Some` 时，KV 自 `position` 行起写入（要求 `cache` 已分配 ≥ position + tokens.len() 行）；
/// `cache` = `None` 时不写 KV，纯文本编码场景（FLUX.2 仅取 last hidden）走这条。
pub fn mistral_text_hidden<B: GqaPrefillBackend>(
    backend: &B,
    config: &MistralConfig,
    layers: &[MistralTextLayer<B::Weight>],
    mut cache: Option<&mut B::Cache>,
    mut hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    let positions = backend.token_rows(&hidden);
    if positions == 0 {
        return Err(crate::runtime::compute_error(format!("Mistral prefill positions={positions}，期望非零")));
    }
    if backend.token_cols(&hidden) != config.hidden_size {
        return Err(crate::runtime::compute_error(format!("Mistral prefill hidden cols={}，期望 {}", backend.token_cols(&hidden), config.hidden_size)));
    }
    if layers.len() != config.layer_count {
        return Err(crate::runtime::compute_error(format!("Mistral prepared layers={}，期望 {}", layers.len(), config.layer_count)));
    }
    for (layer, weights) in layers.iter().enumerate() {
        let _scope = backend.layer_scope();
        let result = mistral_layer(backend, config, cache.as_deref_mut(), layer, weights, &hidden, rope, position);
        backend.finish_batch();
        hidden = result?;
    }
    Ok(hidden)
}

/// 单 token 增量推理（prefill 后用），维护 KV cache。
pub fn mistral_decode_round<B: GqaPrefillBackend>(
    backend: &B,
    config: &MistralConfig,
    layers: &[MistralTextLayer<B::Weight>],
    cache: &mut B::Cache,
    mut hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    if backend.token_rows(&hidden) != 1 {
        return Err(crate::runtime::compute_error(format!("Mistral decode hidden rows={}，期望 1", backend.token_rows(&hidden))));
    }
    if layers.len() != config.layer_count {
        return Err(crate::runtime::compute_error(format!("Mistral prepared layers={}，期望 {}", layers.len(), config.layer_count)));
    }
    for (layer, weights) in layers.iter().enumerate() {
        let _scope = backend.layer_scope();
        let result = mistral_layer(backend, config, Some(cache), layer, weights, &hidden, rope, position);
        backend.finish_batch();
        hidden = result?;
    }
    Ok(hidden)
}

/// Mistral `<s>` (BOS) token id = 1, `</s>` (EOS) token id = 2 (llama.cpp GGUF SentencePiece 标准)。
pub const MISTRAL_BOS_TOKEN_ID: u32 = 1;
pub const MISTRAL_EOS_TOKEN_ID: u32 = 2;

/// 输出尾段 norm 语义:groups=1 是标准 RMSNorm,K2-Horizon 为分组变体。
fn output_norm(config: &MistralConfig) -> super::output::OutputNorm {
    if config.norm_groups == 1 { super::output::OutputNorm::Rms } else { super::output::OutputNorm::GroupedRms { groups: config.norm_groups } }
}

/// 构造 Mistral-Small-3.2 instruct 标准 prompt（不依赖 tokenizer chat template 解析，纯字面拼接）。
///
/// Mistral-3.2 标准模板：`<s>[SYSTEM_PROMPT]<s>[INST] {prompt} [/INST]`
/// 这里采用单 BOS + 单 `[INST] ... [/INST]` 的最简形态，方便端到端跑通。
pub fn mistral_instruct_prompt(user: &str, system: Option<&str>) -> String {
    let system = system.unwrap_or("");
    if system.is_empty() { format!("<s>[INST] {user} [/INST]") } else { format!("<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST] {user} [/INST]") }
}

/// Mistral Metal/CUDA 共用的请求准备。当前模板只支持单轮 instruct；不支持的
/// tools 显式报错，不能静默丢掉 schema 后继续生成普通文本。
pub fn mistral_request_prompt(request: &serde_json::Value) -> Result<String, String> {
    let tools = crate::runtime::tool::request_tools(request)?;
    if crate::runtime::tool::ToolDialect::ChatmlJson.instructions(tools, request.get("tool_choice"))?.is_some() {
        return Err("Mistral 当前未接入工具调用协议".to_owned());
    }
    let messages = request.get("messages").and_then(serde_json::Value::as_array).ok_or("Mistral 请求缺少 messages")?;
    let mut system = Vec::new();
    let mut user = None;
    for message in messages {
        let role = message.get("role").and_then(serde_json::Value::as_str).ok_or("Mistral message.role 必须是字符串")?;
        let content = crate::runtime::session::text_content(message.get("content"))?;
        match role {
            "system" | "developer" if user.is_none() => system.push(content),
            "system" | "developer" => return Err("Mistral system/developer 消息必须位于 user 之前".to_owned()),
            "user" if user.is_none() => user = Some(content),
            "user" | "assistant" | "tool" => return Err("Mistral 当前模板只支持单轮 system/developer + user，请勿静默丢弃历史消息".to_owned()),
            other => return Err(format!("Mistral 不支持 message.role={other}")),
        }
    }
    let user = user.ok_or("Mistral 请求缺少 user 消息")?;
    let system = system.join("\n");
    Ok(mistral_instruct_prompt(&user, (!system.is_empty()).then_some(system.as_str())))
}

/// Prefill 后取最后一个 token 位置（prompt 末位）作为首个输出。
pub fn mistral_last_token_output<B: Backend>(backend: &B, config: &MistralConfig, head: &MistralOutputHead<B::Weight>, hidden: &B::Tensor) -> Result<super::output::OutputResult<B::Tensor>, BackendError> {
    super::output::last_token_output(backend, head, hidden, backend.token_rows(hidden) - 1, &super::output::OutputPlan { eps: config.rms_eps, norm: output_norm(config), excluded_tokens: Vec::new() })
}

/// decode 轮末尾取 hidden 第 0 行作为输出。
pub fn mistral_token_output<B: Backend>(backend: &B, config: &MistralConfig, head: &MistralOutputHead<B::Weight>, hidden: &B::Tensor) -> Result<super::output::OutputResult<B::Tensor>, BackendError> {
    super::output::token_output(backend, head, hidden, &super::output::OutputPlan { eps: config.rms_eps, norm: output_norm(config), excluded_tokens: Vec::new() })
}

/// Mistral 单层 forward（Llama 风格，无 QK-norm）。
#[allow(clippy::too_many_arguments)]
fn mistral_layer<B: GqaPrefillBackend>(
    backend: &B,
    config: &MistralConfig,
    cache: Option<&mut B::Cache>,
    layer: usize,
    weights: &MistralTextLayer<B::Weight>,
    hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    if backend.token_rows(hidden) == 1 {
        backend.begin_decode_batch();
    } else {
        backend.begin_batch();
    }
    // 1. pre-norm + QKV projections(K2-Horizon 为分组 RMSNorm,groups=1 时保持原路径)
    let normed = match config.norm_groups {
        1 => backend.rmsnorm(hidden, &weights.input_norm, config.rms_eps)?,
        groups => backend.grouped_rmsnorm(hidden, &weights.input_norm, config.rms_eps, groups)?,
    };
    let (query, key, value) = backend.triple_linear(&normed, &weights.query, &weights.key, &weights.value)?;
    // 2. RoPE on Q/K (SplitHalf layout, full head_dim as rotary dim)
    let query = backend.rope_prefix(&query, config.num_heads, config.head_dim, RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
    let key = backend.rope_prefix(&key, config.num_kv_heads, config.head_dim, RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
    // 3. Causal GQA attention (no QK-norm, Mistral 默认)
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
    // 5. pre-norm + SwiGLU + 残差:走 gated_mlp_add_residual 能力,支持 fused
    //    epilogue 的 backend 把残差加法融进 down 投影,省一次独立 dispatch。
    let normed = match config.norm_groups {
        1 => backend.rmsnorm(&attention_residual, &weights.post_attention_norm, config.rms_eps)?,
        groups => backend.grouped_rmsnorm(&attention_residual, &weights.post_attention_norm, config.rms_eps, groups)?,
    };
    backend.gated_mlp_add_residual(&normed, &weights.gate, &weights.up, &weights.down, &Activation::Silu, &attention_residual)
}

// ── 向后兼容 alias ───────────────────────────────────────────────────────
//
// 旧 `mistral_small32` 标识符通过 `pub type` 映射到新 `mistral` 模块符号。新代码直接用
// `MistralTextLayer` / `MistralOutputHead` / `prepare_mistral_layer`。
pub type MistralSmall32TextLayer<W> = MistralTextLayer<W>;
pub type MistralSmall32OutputHead<W> = MistralOutputHead<W>;

// Node 适配器：组合 Mistral 平台无关运行时与 Metal / CPU 后端资源，匹配 zllm 通用模式。
// macOS 走 Metal 路径,非 macOS 暴露占位 `NodeEngine` 提示当前平台不支持。
#[cfg(feature = "with-cuda")]
pub mod cuda_node;
pub mod node;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_prompt统一system与user且拒绝未实现工具协议() {
        let request = json!({"messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]});
        assert_eq!(mistral_request_prompt(&request).unwrap(), "<s>[SYSTEM_PROMPT]system[/SYSTEM_PROMPT][INST] hello [/INST]");

        let request = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "read"}}]
        });
        assert!(mistral_request_prompt(&request).unwrap_err().contains("工具调用"));
        assert!(mistral_request_prompt(&json!({"messages": [{"role": "user", "content": "hello"}], "tools": {}})).unwrap_err().contains("必须是数组"));
        assert!(mistral_request_prompt(&json!({"messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "second"}]})).unwrap_err().contains("只支持单轮"));
        assert!(mistral_request_prompt(&json!({"messages": [{"role": "user", "content": "hello"}], "tool_choice": "required"})).unwrap_err().contains("tools 不能为空"));
    }

    #[test]
    fn sequence_length不超过模型声明上限() {
        let config = MistralConfig {
            architecture: crate::weight::model::mistral::DenseGqaArchitecture::Mistral,
            layer_count: 1,
            hidden_size: 8,
            intermediate_size: 16,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 8,
            vocab_size: 32,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            rope_scaling: None,
            rms_eps: 1e-5,
            norm_groups: 1,
            eos_token_id: 2,
        };
        assert!(crate::runtime::validate_max_sequence_length("Mistral", 128, config.max_position_embeddings).is_ok());
        assert!(crate::runtime::validate_max_sequence_length("Mistral", 0, config.max_position_embeddings).is_err());
        assert!(crate::runtime::validate_max_sequence_length("Mistral", 129, config.max_position_embeddings).is_err());
    }
}
