//! Laguna 2.1 模型算法与执行组合。
//!
//! 结构:full/sliding 混合 GQA(滑窗层 ring KV)+ 逐头 softplus 门控 +
//! sigmoid-bias 路由 MoE(共享专家无门控)+ 前置 dense 层。
//! 平台无关算法在本文件;具体平台组合见同目录 `<backend>_*.rs`。

pub mod cpu;
#[cfg(feature = "with-cuda")]
pub mod cuda;
#[cfg(feature = "with-cuda")]
pub mod cuda_node;

use crate::{
    attention::{
        gqa::{CausalWindow, GqaSpec},
        rope::{RopeSpec, RopeTable, RotaryLayout},
    },
    backend::{Backend, BackendError, ExpertDecodeBackend, ExpertPrefillBackend, GqaPrefillBackend},
    moe::{
        Activation, FeedforwardSpec,
        dense_mlp::{DenseMlpSpec, DenseMlpWeightsRef, forward_observed},
        prefill::prefill_experts_observed,
        topk_moe::{MoeFfnRef, ScoringFunc, SharedExpertRef, TopkMoeSpec},
    },
    norm::NormSpec,
    runtime::{LayerId, LayerSpec, Model, ModelError},
    weight::expert_source::ExpertSourceProvider,
};

use super::expert_pipeline::{ExpertDecodePipeline, ExpertDecodeRequest};

pub use crate::model_spec::laguna::LagunaConfig;

// ============================================================================
// 模型规格:层类型、注意力/MoE spec 与 GGUF 映射。
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagunaLayerKind {
    FullAttention,
    SlidingAttention,
}

/// Laguna 2.1 全系 HF config 的 chat 结束 token(eos_token_id = [2, 24]);
/// GGUF 只带单 eos,映射时并入停止集。
pub const LAGUNA_CHAT_EOS_TOKEN: u32 = 24;

impl LagunaConfig {
    pub fn layer_kind(&self, layer: LayerId) -> Option<LagunaLayerKind> {
        if layer >= self.layer_count {
            return None;
        }
        if layer.is_multiple_of(self.full_attention_interval) { Some(LagunaLayerKind::FullAttention) } else { Some(LagunaLayerKind::SlidingAttention) }
    }

    pub fn num_heads(&self, layer: LayerId) -> Option<usize> {
        match self.layer_kind(layer)? {
            LagunaLayerKind::FullAttention => Some(self.full_num_heads),
            LagunaLayerKind::SlidingAttention => Some(self.sliding_num_heads),
        }
    }

    pub fn rope_spec(&self, layer: LayerId) -> Option<&RopeSpec> {
        match self.layer_kind(layer)? {
            LagunaLayerKind::FullAttention => Some(&self.full_rope),
            LagunaLayerKind::SlidingAttention => Some(&self.sliding_rope),
        }
    }

    pub fn gqa_spec(&self, layer: LayerId) -> Option<GqaSpec> {
        let (num_heads, window, rope_dim, rope_theta) = match self.layer_kind(layer)? {
            LagunaLayerKind::FullAttention => (self.full_num_heads, CausalWindow::Full, self.full_rope.rotary_dim(), self.full_rope.theta()),
            LagunaLayerKind::SlidingAttention => (self.sliding_num_heads, CausalWindow::Sliding { size: self.sliding_window }, self.sliding_rope.rotary_dim(), self.sliding_rope.theta()),
        };
        Some(GqaSpec { num_heads, num_kv_heads: self.num_kv_heads, head_dim: self.head_dim, rope_dim, rope_theta, use_qk_norm: true, window, score_scale: 1.0 / (self.head_dim as f32).sqrt(), output_gate: true })
    }

    pub fn moe_spec(&self) -> TopkMoeSpec {
        TopkMoeSpec {
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            num_shared_experts: 1,
            scoring_func: ScoringFunc::SigmoidBias,
            normalize_selected: true,
            routed_scaling_factor: self.routed_scaling_factor,
            intermediate_size: self.expert_intermediate_size,
            shared_intermediate_size: self.shared_intermediate_size,
            activation: Activation::Silu,
        }
    }

    pub fn dense_spec(&self) -> DenseMlpSpec {
        DenseMlpSpec { intermediate_size: self.dense_intermediate_size, activation: Activation::Silu }
    }

    /// 逐层 KV 存储容量;复用 CausalWindow::cache_capacity(sliding 层=窗口,ring 寻址)。
    pub fn kv_capacities(&self, max_sequence_len: usize) -> Vec<usize> {
        (0..self.layer_count).map(|layer| self.gqa_spec(layer).expect("层号已校验").window.cache_capacity(max_sequence_len)).collect()
    }

    /// 从 GGUF laguna 元数据构造配置;Laguna-S/XS 2.1 共用同一映射。
    ///
    /// GGUF 的 `attention.head_count` 是单一值(滑窗层头数),两类层的头数都
    /// 从 `attn_q` 张量维度推导(full 层采样 blk.0,sliding 层采样 blk.1)。
    pub fn from_gguf(reader: &crate::weight::container::gguf::GgufReader) -> Result<Self, String> {
        let value = |key: &str| reader.metadata_u64(key).map_err(|error| format!("Laguna GGUF {error}"));
        let float = |key: &str| match reader.metadata(key) {
            Some(crate::weight::container::gguf::GgufValue::Float(number)) => Ok(*number as f32),
            Some(crate::weight::container::gguf::GgufValue::Unsigned(number)) => Ok(*number as f32),
            _ => Err(format!("Laguna GGUF metadata {key} 缺失或不是数值")),
        };
        const GATING_SIGMOID: u64 = 2;
        if value("laguna.expert_gating_func")? != GATING_SIGMOID {
            return Err("Laguna GGUF expert_gating_func != 2(sigmoid 路由),当前只支持 sigmoid".to_owned());
        }
        let head_dim = value("laguna.attention.key_length")? as usize;
        if head_dim != value("laguna.attention.value_length")? as usize {
            return Err("Laguna GGUF attention.key_length != value_length".to_owned());
        }
        let query_heads = |layer: usize| -> Result<usize, String> {
            let name = format!("blk.{layer}.attn_q.weight");
            let tensor = reader.tensor(&name).ok_or_else(|| format!("Laguna GGUF 缺少 {name}"))?;
            let out_columns = tensor.dims.last().copied().ok_or_else(|| format!("Laguna GGUF {name} 维度缺失"))?;
            if out_columns.is_multiple_of(head_dim) { Ok(out_columns / head_dim) } else { Err(format!("Laguna GGUF {name} 输出列 {out_columns} 不是 head_dim={head_dim} 的整数倍")) }
        };
        let vocab_size = reader
            .tensor("token_embd.weight")
            .and_then(|tensor| tensor.dims.last().copied())
            .filter(|_| reader.tensor("token_embd.weight").is_some_and(|tensor| tensor.dims.len() == 2))
            .ok_or_else(|| "Laguna GGUF token_embd.weight 必须是 [hidden, vocab] 二维".to_owned())?;
        let rope_scaling = match reader.metadata("laguna.rope.scaling.type") {
            Some(crate::weight::container::gguf::GgufValue::String(text)) => text.as_str(),
            _ => "",
        };
        if rope_scaling != "yarn" {
            return Err(format!("Laguna GGUF rope.scaling.type={rope_scaling} 不是 yarn,当前只支持 yarn"));
        }
        let eos_from_gguf = value("tokenizer.ggml.eos_token_id")? as u32;
        let mut eos_token_ids = vec![eos_from_gguf, LAGUNA_CHAT_EOS_TOKEN];
        eos_token_ids.sort_unstable();
        eos_token_ids.dedup();
        Ok(Self {
            vocab_size,
            hidden_size: value("laguna.embedding_length")? as usize,
            layer_count: value("laguna.block_count")? as usize,
            leading_dense_layer_count: (value("laguna.leading_dense_block_count")? as usize).max(1),
            full_attention_interval: 4,
            full_num_heads: query_heads(0)?,
            sliding_num_heads: query_heads(1)?,
            num_kv_heads: value("laguna.attention.head_count_kv")? as usize,
            head_dim,
            sliding_window: value("laguna.attention.sliding_window")? as usize,
            max_position_embeddings: value("laguna.context_length")? as usize,
            full_rope: RopeSpec::Yarn {
                rotary_dim: value("laguna.rope.dimension_count")? as usize,
                theta: float("laguna.rope.freq_base")?,
                factor: float("laguna.rope.scaling.factor")?,
                original_context: value("laguna.rope.scaling.original_context_length")? as usize,
                beta_fast: float("laguna.rope.scaling.yarn_beta_fast")?,
                beta_slow: float("laguna.rope.scaling.yarn_beta_slow")?,
            },
            sliding_rope: RopeSpec::Default { rotary_dim: value("laguna.rope.dimension_count_swa")? as usize, theta: float("laguna.rope.freq_base_swa")? },
            rms_eps: float("laguna.attention.layer_norm_rms_epsilon")?,
            dense_intermediate_size: value("laguna.feed_forward_length")? as usize,
            num_experts: value("laguna.expert_count")? as usize,
            num_experts_per_tok: value("laguna.expert_used_count")? as usize,
            expert_intermediate_size: value("laguna.expert_feed_forward_length")? as usize,
            shared_intermediate_size: value("laguna.expert_shared_feed_forward_length")? as usize,
            routed_scaling_factor: float("laguna.expert_weights_scale")?,
            eos_token_ids,
        })
    }

    /// Laguna-S-2.1(118B-A8B)。unsloth GGUF 将上下文截到 256K(yarn factor 32)。
    pub fn standard_s() -> Self {
        Self {
            vocab_size: 100_352,
            hidden_size: 3_072,
            layer_count: 48,
            leading_dense_layer_count: 1,
            full_attention_interval: 4,
            full_num_heads: 48,
            sliding_num_heads: 72,
            num_kv_heads: 8,
            head_dim: 128,
            sliding_window: 512,
            max_position_embeddings: 262_144,
            full_rope: RopeSpec::Yarn { rotary_dim: 64, theta: 500_000.0, factor: 32.0, original_context: 8_192, beta_fast: 32.0, beta_slow: 1.0 },
            sliding_rope: RopeSpec::Default { rotary_dim: 128, theta: 10_000.0 },
            rms_eps: 1e-6,
            dense_intermediate_size: 12_288,
            num_experts: 256,
            num_experts_per_tok: 10,
            expert_intermediate_size: 1_024,
            shared_intermediate_size: 1_024,
            routed_scaling_factor: 2.5,
            eos_token_ids: vec![2, LAGUNA_CHAT_EOS_TOKEN],
        }
    }

    /// Laguna-XS-2.1(33B-A2.4B)。
    pub fn standard_xs() -> Self {
        Self {
            vocab_size: 100_352,
            hidden_size: 2_048,
            layer_count: 40,
            leading_dense_layer_count: 1,
            full_attention_interval: 4,
            full_num_heads: 48,
            sliding_num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            sliding_window: 512,
            max_position_embeddings: 262_144,
            full_rope: RopeSpec::Yarn { rotary_dim: 64, theta: 500_000.0, factor: 32.0, original_context: 8_192, beta_fast: 64.0, beta_slow: 1.0 },
            sliding_rope: RopeSpec::Default { rotary_dim: 128, theta: 10_000.0 },
            rms_eps: 1e-6,
            dense_intermediate_size: 8_192,
            num_experts: 256,
            num_experts_per_tok: 8,
            expert_intermediate_size: 512,
            shared_intermediate_size: 512,
            routed_scaling_factor: 2.5,
            eos_token_ids: vec![2, LAGUNA_CHAT_EOS_TOKEN],
        }
    }
}

pub struct Laguna {
    config: LagunaConfig,
    layer_specs: Vec<LayerSpec>,
}

impl Laguna {
    pub fn new(config: LagunaConfig) -> Result<Self, ModelError> {
        ensure_supported(&config).map_err(ModelError::InvalidArchitecture)?;
        let layer_specs = (0..config.layer_count).map(|layer| Self::build_layer_spec(&config, layer)).collect();
        Ok(Self { config, layer_specs })
    }

    fn build_layer_spec(config: &LagunaConfig, layer: LayerId) -> LayerSpec {
        let attention = crate::attention::AttentionSpec::Gqa(config.gqa_spec(layer).expect("build_layer_spec 层号已过校验"));
        let feedforward = if layer < config.leading_dense_layer_count { FeedforwardSpec::Dense(config.dense_spec()) } else { FeedforwardSpec::TopkMoe(config.moe_spec()) };
        LayerSpec { attention, feedforward, input_norm: NormSpec::Rms { eps: config.rms_eps }, post_attention_norm: NormSpec::Rms { eps: config.rms_eps }, post_norm: None }
    }
}

impl Model for Laguna {
    type Config = LagunaConfig;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn layer_count(&self) -> usize {
        self.config.layer_count
    }

    fn layer_spec(&self, layer: LayerId) -> Result<&LayerSpec, ModelError> {
        self.layer_specs.get(layer).ok_or(ModelError::LayerOutOfRange { layer, layer_count: self.config.layer_count })
    }
}

pub fn ensure_supported(config: &LagunaConfig) -> Result<(), String> {
    if config.layer_count < 2 || config.leading_dense_layer_count == 0 || config.leading_dense_layer_count >= config.layer_count {
        return Err(format!("Laguna layer_count={} / leading_dense={} 非法", config.layer_count, config.leading_dense_layer_count));
    }
    if config.full_attention_interval == 0 || config.layer_kind(0) != Some(LagunaLayerKind::FullAttention) {
        return Err("Laguna 要求第 0 层是 full attention".to_owned());
    }
    for layer in 0..config.layer_count {
        let Some(heads) = config.num_heads(layer) else { return Err(format!("Laguna L{layer} 头数缺失")) };
        if heads == 0 || config.num_kv_heads == 0 || !heads.is_multiple_of(config.num_kv_heads) {
            return Err(format!("Laguna L{layer} heads={heads} kv={} 不合法", config.num_kv_heads));
        }
        let Some(spec) = config.gqa_spec(layer) else { return Err(format!("Laguna L{layer} gqa spec 缺失")) };
        spec.geometry().validate().map_err(|error| format!("Laguna L{layer}: {error}"))?;
        spec.window.validate().map_err(|error| format!("Laguna L{layer}: {error}"))?;
        config.rope_spec(layer).expect("kind 已校验").validate().map_err(|error| format!("Laguna L{layer} rope: {error}"))?;
    }
    if config.sliding_window < 2 || config.rms_eps <= 0.0 || config.hidden_size == 0 || config.head_dim == 0 {
        return Err("Laguna window/rms_eps/hidden/head_dim 非法".to_owned());
    }
    if config.num_experts == 0 || config.num_experts_per_tok == 0 || config.num_experts_per_tok > config.num_experts || config.expert_intermediate_size == 0 || config.shared_intermediate_size == 0 {
        return Err(format!("Laguna MoE 参数非法: experts={} top-k={} inter={} shared={}", config.num_experts, config.num_experts_per_tok, config.expert_intermediate_size, config.shared_intermediate_size));
    }
    if config.dense_intermediate_size == 0 || config.vocab_size == 0 {
        return Err("Laguna dense intermediate/vocab 非法".to_owned());
    }
    Ok(())
}

/// full 与 sliding 两张 RoPE 表;按层取用,常驻整个会话。
pub struct LagunaRopeTables {
    full: RopeTable,
    sliding: RopeTable,
}

impl LagunaRopeTables {
    pub fn new(config: &LagunaConfig, sequence_len: usize) -> Result<Self, BackendError> {
        Ok(Self {
            full: RopeTable::from_spec(sequence_len, config.full_rope.clone()).map_err(|error| BackendError::Compute { msg: format!("Laguna full rope 表: {error}") })?,
            sliding: RopeTable::from_spec(sequence_len, config.sliding_rope.clone()).map_err(|error| BackendError::Compute { msg: format!("Laguna sliding rope 表: {error}") })?,
        })
    }

    pub fn layer(&self, kind: LagunaLayerKind) -> &RopeTable {
        match kind {
            LagunaLayerKind::FullAttention => &self.full,
            LagunaLayerKind::SlidingAttention => &self.sliding,
        }
    }
}

// ============================================================================
// 权重装载。
// ============================================================================

#[derive(Debug)]
pub struct LagunaAttention<W> {
    pub query: W,
    pub query_norm: W,
    pub key: W,
    pub key_norm: W,
    pub value: W,
    /// 逐头门控投影:attention 输出乘 softplus(gate) 标量(head_dim 广播)。
    pub gate: W,
    pub output: W,
}

#[derive(Debug)]
pub enum LagunaMlp<W> {
    Dense {
        gate: W,
        up: W,
        down: W,
    },
    Sparse {
        router_weight: W,
        /// sigmoid 路由的 e_score_correction_bias(GGUF exp_probs_b.bias)。
        router_bias: W,
        shared_gate: W,
        shared_up: W,
        shared_down: W,
    },
}

#[derive(Debug)]
pub struct LagunaLayer<W> {
    pub input_norm: W,
    pub attention: LagunaAttention<W>,
    pub post_attention_norm: W,
    pub mlp: LagunaMlp<W>,
}

pub fn prepare_laguna_layers<B: Backend>(backend: &B, source: &LagunaGguf) -> Result<Vec<LagunaLayer<B::Weight>>, BackendError> {
    (0..source.config().layer_count).map(|layer| prepare_laguna_layer(backend, source, layer)).collect()
}

pub fn prepare_laguna_layer<B: Backend>(backend: &B, source: &LagunaGguf, layer: usize) -> Result<LagunaLayer<B::Weight>, BackendError> {
    let prefix = format!("blk.{layer}");
    let reader = source.reader();
    let dense = layer < source.config().leading_dense_layer_count;
    let mlp = if dense {
        LagunaMlp::Dense {
            gate: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ffn_gate.weight"))?,
            up: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ffn_up.weight"))?,
            down: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ffn_down.weight"))?,
        }
    } else {
        LagunaMlp::Sparse {
            router_weight: super::prepare_gguf_f32_matrix(backend, reader, &format!("{prefix}.ffn_gate_inp.weight"))?,
            router_bias: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.exp_probs_b.bias"))?,
            shared_gate: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ffn_gate_shexp.weight"))?,
            shared_up: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ffn_up_shexp.weight"))?,
            shared_down: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.ffn_down_shexp.weight"))?,
        }
    };
    Ok(LagunaLayer {
        input_norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.attn_norm.weight"))?,
        attention: LagunaAttention {
            query: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_q.weight"))?,
            query_norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.attn_q_norm.weight"))?,
            key: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_k.weight"))?,
            key_norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.attn_k_norm.weight"))?,
            value: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_v.weight"))?,
            gate: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_gate.weight"))?,
            output: super::prepare_gguf_matrix(backend, reader, &format!("{prefix}.attn_output.weight"))?,
        },
        post_attention_norm: super::prepare_gguf_f32_vector(backend, reader, &format!("{prefix}.ffn_norm.weight"))?,
        mlp,
    })
}

pub type LagunaOutputHead<W> = super::output::OutputHead<W>;

pub fn prepare_laguna_output_quantized<B: Backend>(backend: &B, source: &LagunaGguf, quantization: crate::weight::LmHeadQuantization) -> Result<LagunaOutputHead<B::Weight>, BackendError> {
    let cfg = source.config();
    let norm = source.final_norm().map_err(|error| BackendError::Compute { msg: format!("Laguna final norm: {error}") })?;
    let head = source.output_head().map_err(|error| BackendError::Compute { msg: format!("Laguna output head: {error}") })?;
    super::output::prepare_output_head_quantized(backend, &norm, crate::backend::LinearWeight::gguf(&head), cfg.vocab_size, cfg.hidden_size, quantization)
}

pub fn laguna_token_output<B: Backend>(backend: &B, cfg: &LagunaConfig, head: &LagunaOutputHead<B::Weight>, hidden: &B::Tensor) -> Result<super::output::OutputResult<B::Tensor>, BackendError> {
    // f32 残差流收缩 f16 进 final norm/LM head(残差值域有界;LM head 是 K-quant 投影,与层入口同模式)。
    let hidden = backend.cast_f16(hidden)?;
    super::output::token_output(backend, head, &hidden, &super::output::OutputPlan { eps: cfg.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: Vec::new() })
}

// ============================================================================
// 执行:prefill 与 decode。
// ============================================================================

#[derive(Clone, Copy, Default)]
pub struct LagunaRuntimeOptions {
    pub expert_batch_size: Option<usize>,
}

/// 绑定一次模型加载期间保持不变的执行依赖。
pub struct LagunaRuntime<'a, B: Backend> {
    backend: &'a B,
    config: &'a LagunaConfig,
    layers: &'a [LagunaLayer<B::Weight>],
    rope: &'a LagunaRopeTables,
    moe: TopkMoeSpec,
    options: LagunaRuntimeOptions,
}

impl<'a, B: Backend> LagunaRuntime<'a, B> {
    pub fn new(backend: &'a B, config: &'a LagunaConfig, layers: &'a [LagunaLayer<B::Weight>], rope: &'a LagunaRopeTables, options: LagunaRuntimeOptions) -> Self {
        Self { backend, config, layers, rope, moe: config.moe_spec(), options }
    }

    /// 单层 attention:RMSNorm → QKV+gate 投影 → q/k 逐头 RMSNorm → RoPE(双表)→
    /// GQA(窗口语义由 spec.window + ring cache 承担)→ softplus 逐头门控 → 输出投影。
    fn attention(&self, cache: &mut B::Cache, layer: usize, weights: &LagunaLayer<B::Weight>, hidden: &B::Tensor, position: usize) -> Result<B::Tensor, BackendError>
    where
        B: GqaPrefillBackend,
    {
        let cfg = self.config;
        let kind = cfg.layer_kind(layer).expect("执行层号已过校验");
        let spec = cfg.gqa_spec(layer).expect("执行层号已过校验");
        let heads = cfg.num_heads(layer).expect("执行层号已过校验");
        let table = self.rope.layer(kind);
        // ZLLM_CUDA_NO_ATTENTION 消元开关:attention 整体旁路(全行拷贝直通)。
        if std::env::var_os("ZLLM_CUDA_NO_ATTENTION").is_some() {
            let rows = self.backend.token_rows(hidden);
            let passthrough = (0..rows as u32).collect::<Vec<_>>();
            return self.backend.select_rows(hidden, &passthrough);
        }
        // f32 残差流收缩 f16 进投影(值域有界;悬崖只在求和路径)。
        let hidden_f16 = self.backend.cast_f16(hidden)?;
        let normed = self.backend.rmsnorm(&hidden_f16, &weights.input_norm, cfg.rms_eps)?;
        let (query, key, value) = self.backend.triple_linear(&normed, &weights.attention.query, &weights.attention.key, &weights.attention.value)?;
        let gate = self.backend.linear(&normed, &weights.attention.gate)?;
        let query = self.backend.rmsnorm_heads(&query, &weights.attention.query_norm, heads, cfg.head_dim, cfg.rms_eps)?;
        let key_normed = self.backend.rmsnorm_heads(&key, &weights.attention.key_norm, cfg.num_kv_heads, cfg.head_dim, cfg.rms_eps)?;
        let query = self.backend.rope_prefix(&query, heads, spec.rope_dim, RotaryLayout::SplitHalf, position, &table.cos, &table.sin)?;
        let key = self.backend.rope_prefix(&key_normed, cfg.num_kv_heads, spec.rope_dim, RotaryLayout::SplitHalf, position, &table.cos, &table.sin)?;
        if std::env::var_os("ZLLM_CUDA_NAN_AUDIT").is_some() && std::env::var("ZLLM_CUDA_AUDIT_LAYER").map(|value| value.parse::<usize>() == Ok(layer)).unwrap_or(false) {
            let rows = |tensor: &B::Tensor, row: usize| {
                self.backend.debug_row_head_f32(tensor, row).map(|values| values.iter().map(|value| if value.is_nan() { "NaN".to_owned() } else { format!("{value:.4}") }).collect::<Vec<_>>().join(",")).unwrap_or_else(|_| "err".to_owned())
            };
            let full = |tensor: &B::Tensor, row: usize| {
                self.backend
                    .debug_row_full_f32(tensor, row)
                    .map(|(values, nan)| format!("[{}] NaN={}/{}", values.iter().map(|value| if value.is_nan() { "NaN".to_owned() } else { format!("{value:.3}") }).collect::<Vec<_>>().join(","), nan, values.len().max(3072 - 8) + 8))
                    .unwrap_or_else(|_| "err".to_owned())
            };
            eprintln!("[laguna-k-stage] L{layer} normed0={} k_lin0={} hidden0={}", full(&normed, 0), full(&key, 0), full(hidden, 0));
        }
        let attended = self.backend.gqa_prefill_attention_cached(cache, layer, position, &query, &key, &value, &spec, false)?;
        if std::env::var_os("ZLLM_CUDA_NAN_AUDIT").is_some() && std::env::var("ZLLM_CUDA_AUDIT_LAYER").map(|value| value.parse::<usize>() == Ok(layer)).unwrap_or(false) {
            let brief = |tensor: &B::Tensor| {
                self.backend
                    .debug_last_row_f32(tensor)
                    .map(|values| {
                        if values.iter().any(|value| value.is_nan()) {
                            format!("NaN {}/{}", values.iter().filter(|value| value.is_nan()).count(), values.len())
                        } else {
                            format!("ok {:.2}", values.iter().fold(0.0f32, |acc, value| acc.max(value.abs())))
                        }
                    })
                    .unwrap_or_else(|error| format!("err {error:?}"))
            };
            let head = |tensor: &B::Tensor| {
                self.backend.debug_row_head_f32(tensor, 0).map(|values| values.iter().map(|value| if value.is_nan() { "NaN".to_owned() } else { format!("{value:.2}") }).collect::<Vec<_>>().join(",")).unwrap_or_else(|_| "err".to_owned())
            };
            eprintln!(
                "[laguna-attn-audit] L{layer} normed={} q={} k={} v={} attended={} | row0: normed=[{}] k=[{}] v=[{}]",
                brief(&normed),
                brief(&query),
                brief(&key),
                brief(&value),
                brief(&attended),
                head(&normed),
                head(&key),
                head(&value)
            );
        }
        // ZLLM_CUDA_NO_GATE 消元开关:跳过 softplus 门控(结果不正确,只测污染路径)。
        let gated = if std::env::var_os("ZLLM_CUDA_NO_GATE").is_some() { attended } else { self.backend.softplus_gate(&attended, &gate)? };
        self.backend.linear(&gated, &weights.attention.output)
    }

    /// prefill 一段 chunk;`position` 是 chunk 起点。返回末行 hidden。
    pub fn prefill(&self, cache: &mut B::Cache, experts: &mut B::PrefillExperts, hidden: B::Tensor, position: usize) -> Result<B::Tensor, BackendError>
    where
        B: ExpertPrefillBackend + GqaPrefillBackend,
    {
        let rows = self.backend.token_rows(&hidden);
        if rows == 0 || self.backend.token_cols(&hidden) != self.config.hidden_size || self.layers.len() != self.config.layer_count {
            return Err(BackendError::Compute { msg: format!("Laguna prefill 输入不完整: layers={}/{}, hidden=[{},{}]", self.layers.len(), self.config.layer_count, rows, self.backend.token_cols(&hidden)) });
        }
        // ring KV 正确性约束:单个 chunk 不能超过滑窗容量,否则同 chunk 内
        // 槽位回绕会覆盖同 chunk 仍需读取的键(gemma4 ring KV 同款约束)。
        if rows > self.config.sliding_window {
            return Err(BackendError::Compute { msg: format!("Laguna prefill rows={} 超过滑窗 {}，必须分块", rows, self.config.sliding_window) });
        }
        if position.checked_add(rows).map_or(true, |end| end > self.config.max_position_embeddings) {
            return Err(BackendError::Compute { msg: format!("Laguna prefill position={position} rows={rows} 超过 max_position={}", self.config.max_position_embeddings) });
        }
        let mut hidden = hidden;
        let nan_audit = std::env::var_os("ZLLM_CUDA_NAN_AUDIT").is_some();
        let result = (|| {
            for (layer, weights) in self.layers.iter().enumerate() {
                let _scope = self.backend.layer_scope();
                self.backend.begin_batch();
                hidden = self.prefill_layer(cache, experts, layer, weights, hidden, position)?;
                self.backend.submit_batch();
                if std::env::var_os("ZLLM_CUDA_ROW0_AUDIT").is_some() {
                    // 调试插桩:逐层回读 hidden 首 token 行的 max-abs 增长曲线(扰动较大,独立开关)。
                    if let Ok((head, nan)) = self.backend.debug_row_full_f32(&hidden, 0) {
                        let max_abs = head.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
                        eprintln!("[laguna-row0-max] L{layer} max={max_abs:.2} nan8={nan}");
                    }
                }
            }
            Ok(hidden)
        })();
        self.backend.finish_batch();
        result
    }

    fn prefill_layer(&self, cache: &mut B::Cache, experts: &mut B::PrefillExperts, layer: usize, weights: &LagunaLayer<B::Weight>, hidden: B::Tensor, position: usize) -> Result<B::Tensor, BackendError>
    where
        B: ExpertPrefillBackend + GqaPrefillBackend,
    {
        let attention = self.attention(cache, layer, weights, &hidden, position)?;
        let residual = self.backend.add(&hidden, &attention)?;
        // ZLLM_CUDA_NO_MLP 消元开关:MLP 整体旁路(层退化为 attention-only)。
        if std::env::var_os("ZLLM_CUDA_NO_MLP").is_some() {
            let rows = self.backend.token_rows(&residual);
            let passthrough = (0..rows as u32).collect::<Vec<_>>();
            return self.backend.select_rows(&residual, &passthrough);
        }
        let residual_f16 = self.backend.cast_f16(&residual)?;
        let moe_input = self.backend.rmsnorm(&residual_f16, &weights.post_attention_norm, self.config.rms_eps)?;
        let feedforward = self.mlp_prefill(cache, experts, layer, weights, &moe_input)?;
        let output = self.backend.add(&residual, &feedforward)?;
        if std::env::var_os("ZLLM_CUDA_NAN_AUDIT").is_some() && std::env::var("ZLLM_CUDA_AUDIT_LAYER").map(|value| value.parse::<usize>() == Ok(layer)).unwrap_or(false) {
            if let (Ok(residual_values), Ok(moe_values), Ok(output_values)) = (self.backend.debug_last_row_f32(&residual), self.backend.debug_last_row_f32(&moe_input), self.backend.debug_last_row_f32(&output)) {
                let brief = |values: &[f32]| {
                    if values.iter().any(|value| value.is_nan()) {
                        format!("NaN {}/{}", values.iter().filter(|value| value.is_nan()).count(), values.len())
                    } else {
                        format!("ok max={:.3}", values.iter().fold(f32::NEG_INFINITY, |acc, value| acc.max(value.abs())))
                    }
                };
                let row0 =
                    |tensor: &B::Tensor| self.backend.debug_row_full_f32(tensor, 0).map(|(values, nan)| format!("max={:.2} nan={}", values.iter().fold(0.0f32, |acc, value| acc.max(value.abs())), nan)).unwrap_or_else(|_| "err".into());
                eprintln!(
                    "[laguna-layer-audit] L{layer} residual={} moe_in={} out={} | row0: residual=[{}] moe_in=[{}] out=[{}]",
                    brief(&residual_values),
                    brief(&moe_values),
                    brief(&output_values),
                    row0(&residual),
                    row0(&moe_input),
                    row0(&output)
                );
            }
        }
        Ok(output)
    }

    fn mlp_prefill(&self, _cache: &mut B::Cache, experts: &mut B::PrefillExperts, layer: usize, weights: &LagunaLayer<B::Weight>, moe_input: &B::Tensor) -> Result<B::Tensor, BackendError>
    where
        B: ExpertPrefillBackend + GqaPrefillBackend,
    {
        match &weights.mlp {
            LagunaMlp::Dense { gate, up, down } => forward_observed(self.backend, &self.config.dense_spec(), DenseMlpWeightsRef { gate, up, down }, moe_input, |_| {}),
            LagunaMlp::Sparse { router_weight, router_bias, shared_gate, shared_up, shared_down } => {
                let shared_full = [SharedExpertRef { gate: shared_gate, up: shared_up, down: shared_down, output_gate: None }];
                let shared: &[SharedExpertRef<B::Weight>] = if std::env::var_os("ZLLM_CUDA_NO_SHARED").is_some() { &[] } else { &shared_full };
                let result = prefill_experts_observed(self.backend, &self.moe, &MoeFfnRef { router_weight, router_bias, shared_experts: &shared, selected_experts: None }, layer, experts, moe_input, self.options.expert_batch_size, |_| {});
                if let (Ok(moe), true) = (&result, std::env::var("ZLLM_CUDA_AUDIT_LAYER").map(|value| value.parse::<usize>() == Ok(layer)).unwrap_or(false)) {
                    let pairs: Vec<String> = (0..moe.routing.top_k).map(|slot| format!("E{}:{:.3}", moe.routing.expert_ids[slot], moe.routing.weights[slot])).collect();
                    eprintln!("[laguna-route-audit] L{layer} row0 [{}]", pairs.join(" "));
                }
                result.map(|moe| moe.tensor)
            }
        }
    }

    /// decode 单 token;`position` 是该 token 的绝对位置,返回下一位置 hidden。
    pub fn decode<S>(&self, expert_sources: &S, expert_state: &mut ExpertDecodePipeline<B::MoeState>, cache: &mut B::Cache, hidden: B::Tensor, position: usize) -> Result<B::Tensor, BackendError>
    where
        B: ExpertDecodeBackend + GqaPrefillBackend,
        S: ExpertSourceProvider,
    {
        if self.backend.token_rows(&hidden) != 1 || self.backend.token_cols(&hidden) != self.config.hidden_size || self.layers.len() != self.config.layer_count {
            return Err(BackendError::Compute { msg: "Laguna decode 输入必须是完整层权重和一个 hidden token".to_owned() });
        }
        if position >= self.config.max_position_embeddings {
            return Err(BackendError::Compute { msg: format!("Laguna decode position={position} 超过上下文上限") });
        }
        let mut hidden = hidden;
        let result = (|| {
            // 整个 decode round 共用一个 batch,避免逐层拆 command buffer。
            self.backend.begin_decode_batch();
            for (layer, weights) in self.layers.iter().enumerate() {
                let _scope = self.backend.layer_scope();
                let attention = self.attention(cache, layer, weights, &hidden, position)?;
                let residual = self.backend.add(&hidden, &attention)?;
                let residual_f16 = self.backend.cast_f16(&residual)?;
                let moe_input = self.backend.rmsnorm(&residual_f16, &weights.post_attention_norm, self.config.rms_eps)?;
                let feedforward = match &weights.mlp {
                    LagunaMlp::Dense { gate, up, down } => forward_observed(self.backend, &self.config.dense_spec(), DenseMlpWeightsRef { gate, up, down }, &moe_input, |_| {})?,
                    LagunaMlp::Sparse { router_weight, router_bias, shared_gate, shared_up, shared_down } => {
                        let shared_full = [SharedExpertRef { gate: shared_gate, up: shared_up, down: shared_down, output_gate: None }];
                        let shared: &[SharedExpertRef<B::Weight>] = if std::env::var_os("ZLLM_CUDA_NO_SHARED").is_some() { &[] } else { &shared_full };
                        let source = expert_sources.source(layer).map_err(BackendError::ExpertLoad)?;
                        // 中间层预取下一层专家;最后一层传 None 结束本 token 的流水线边界。
                        let next_source = (layer + 1 < self.config.layer_count).then(|| expert_sources.source(layer + 1).map(|source| (layer + 1, source))).transpose().map_err(BackendError::ExpertLoad)?;
                        expert_state.decode(
                            self.backend,
                            &self.moe,
                            &MoeFfnRef { router_weight, router_bias, shared_experts: &shared, selected_experts: None },
                            ExpertDecodeRequest { layer, source, position, next: next_source },
                            &moe_input,
                        )?
                    }
                };
                hidden = self.backend.add(&residual, &feedforward)?;
                // decode 侧 nonfinite 审计:每层收尾检查残差流(同步回读会推迟计算流,
                // 可能掩盖 copy 流竞态,仅取证用)。
                if std::env::var_os("ZLLM_CUDA_NAN_AUDIT").is_some() {
                    if let Ok(values) = self.backend.debug_last_row_f32(&hidden) {
                        let bad = values.iter().filter(|value| !value.is_finite()).count();
                        if bad > 0 {
                            let max = values.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
                            eprintln!("[laguna-decode-nan] L{layer} nonfinite={}/{} max={max}", bad, values.len());
                        }
                    }
                }
            }
            self.backend.submit_batch();
            Ok(hidden)
        })();
        self.backend.finish_batch();
        result
    }
}

// ============================================================================
// LagunaGguf —— GGUF 权重 wrapper(模型知识在 runtime,格式能力委托 GgufReader)。
// impl GgufExpertSource/ExpertSourceProvider 为 MoE expert pipeline 提供数据源。
// ============================================================================

use crate::tokenizer::{Detokenizer, Tokenizer};
use crate::weight::container::gguf::{GgufMatrix, GgufReader};
use crate::weight::expert_source::{ExpertSource, GgufExpertSource, GgufExpertWeights};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagunaExpertProjection {
    Gate,
    Up,
    Down,
}

impl GgufExpertSource for LagunaGguf {
    fn intermediate(&self) -> usize {
        self.cfg.expert_intermediate_size
    }

    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }

    fn load_expert_gguf(&self, layer: usize, expert: usize) -> Result<GgufExpertWeights, String> {
        Ok(GgufExpertWeights {
            gate: self.routed_expert(layer, expert, LagunaExpertProjection::Gate)?,
            up: self.routed_expert(layer, expert, LagunaExpertProjection::Up)?,
            down: self.routed_expert(layer, expert, LagunaExpertProjection::Down)?,
        })
    }
}

impl ExpertSourceProvider for LagunaGguf {
    fn source(&self, layer: usize) -> Result<ExpertSource<'_>, String> {
        if layer >= self.cfg.layer_count {
            return Err(format!("Laguna expert source layer 越界: {layer} >= {}", self.cfg.layer_count));
        }
        Ok(ExpertSource::Gguf(self))
    }
}

pub struct LagunaGguf {
    reader: GgufReader,
    cfg: LagunaConfig,
}

impl LagunaGguf {
    pub fn open(path: &Path) -> Result<Self, String> {
        let reader = GgufReader::open(&GgufReader::locate(path)?)?;
        let cfg = LagunaConfig::from_gguf(&reader)?;
        let model = Self { reader, cfg };
        model.validate_metadata()?;
        Ok(model)
    }

    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    pub fn config(&self) -> &LagunaConfig {
        &self.cfg
    }

    pub fn embedding_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        self.reader.embedding_rows("token_embd.weight", token_ids, self.cfg.hidden_size, self.cfg.vocab_size)
    }

    pub fn final_norm(&self) -> Result<Vec<f32>, String> {
        self.reader.read_tensor_f32("output_norm.weight")
    }

    pub fn output_head(&self) -> Result<GgufMatrix, String> {
        self.reader.read_matrix("output.weight")
    }

    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        self.reader.bpe_tokenizer().map_err(|error| format!("Laguna tokenizer: {error}"))
    }

    pub fn detokenizer(&self) -> Result<Detokenizer, String> {
        self.reader.bpe_detokenizer().map_err(|error| format!("Laguna detokenizer: {error}"))
    }

    pub fn routed_expert(&self, layer: usize, expert: usize, projection: LagunaExpertProjection) -> Result<GgufMatrix, String> {
        if layer >= self.cfg.layer_count || expert >= self.cfg.num_experts {
            return Err(format!("Laguna expert 索引越界: layer={layer}/{}, expert={expert}/{}", self.cfg.layer_count, self.cfg.num_experts));
        }
        let proj = match projection {
            LagunaExpertProjection::Gate => "gate",
            LagunaExpertProjection::Up => "up",
            LagunaExpertProjection::Down => "down",
        };
        self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_{proj}_exps.weight"), expert)
    }

    fn validate_metadata(&self) -> Result<(), String> {
        self.reader.expect_metadata_str("general.architecture", "laguna")?;
        self.reader.expect_metadata_u64("laguna.block_count", self.cfg.layer_count as u64)?;
        Ok(())
    }
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn xs_layer_schedule_interleaves_full_and_sliding() {
        let cfg = LagunaConfig::standard_xs();
        assert_eq!(cfg.layer_kind(0), Some(LagunaLayerKind::FullAttention));
        assert_eq!(cfg.layer_kind(1), Some(LagunaLayerKind::SlidingAttention));
        assert_eq!(cfg.layer_kind(3), Some(LagunaLayerKind::SlidingAttention));
        assert_eq!(cfg.layer_kind(4), Some(LagunaLayerKind::FullAttention));
        assert_eq!(cfg.layer_kind(39), Some(LagunaLayerKind::SlidingAttention));
        assert_eq!(cfg.layer_kind(40), None);
        assert_eq!(cfg.layer_kind(0).is_some_and(|kind| cfg.num_heads(0) == Some(48)), true);
        assert_eq!(cfg.num_heads(1), Some(64));
    }

    #[test]
    fn s_uses_72_sliding_heads_and_dense_prefix() {
        let cfg = LagunaConfig::standard_s();
        assert_eq!(cfg.num_heads(0), Some(48));
        assert_eq!(cfg.num_heads(1), Some(72));
        assert_eq!(cfg.layer_kind(47), Some(LagunaLayerKind::SlidingAttention));
        let model = Laguna::new(cfg).expect("S 配置必须有效");
        assert!(matches!(model.layer_spec(0).unwrap().feedforward, FeedforwardSpec::Dense(_)));
        assert!(matches!(model.layer_spec(1).unwrap().feedforward, FeedforwardSpec::TopkMoe(_)));
    }

    #[test]
    fn kv_capacities_follow_layer_kinds() {
        let cfg = LagunaConfig::standard_xs();
        let capacities = cfg.kv_capacities(8192);
        assert_eq!(capacities.len(), 40);
        assert_eq!(capacities[0], 8192);
        assert_eq!(capacities[1], 512);
        assert_eq!(capacities[4], 8192);
    }

    #[test]
    fn moe_spec_matches_laguna_routing() {
        let spec = LagunaConfig::standard_xs().moe_spec();
        assert!(matches!(spec.scoring_func, ScoringFunc::SigmoidBias));
        assert_eq!(spec.num_experts, 256);
        assert_eq!(spec.top_k, 8);
        assert_eq!(spec.num_shared_experts, 1);
        assert_eq!(spec.routed_scaling_factor, 2.5);
        assert!(spec.normalize_selected);
    }

    #[test]
    fn ensure_supported_rejects_misaligned_heads() {
        let mut cfg = LagunaConfig::standard_xs();
        cfg.sliding_num_heads = 63;
        assert!(ensure_supported(&cfg).is_err());
    }
}
