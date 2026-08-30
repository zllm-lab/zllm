//! MiniMax-M3 的平台无关层编排。
//!
//! runtime 负责完整 prefill/decode 层循环、KV position、MoE pipeline 与输出头；
//! backend 只提供设备算子、cache 和专家资源能力。

pub mod vision;

use crate::attention::AttentionSpec;
use crate::attention::gqa::GqaSpec;
use crate::attention::msa::MsaSpec;
use crate::attention::rope::RopeTable;
use crate::backend::{Backend, BackendError, ExpertDecodeBackend, ExpertPrefillBackend, GqaPrefillBackend, LinearWeight};
use crate::moe::{
    Activation, FeedforwardSpec,
    dense_mlp::{self, DenseMlpSpec, DenseMlpWeightsRef},
    topk_moe::{MoeFfnRef, ScoringFunc, SharedExpertRef, TopkMoeSpec},
};
use crate::norm::NormSpec;
use crate::runtime::{LayerId, LayerSpec, Model, ModelError};
use crate::weight::expert_source::ExpertSourceProvider;
use crate::weight::model::minimax_m3::{MiniMaxM3CoreMatrix, MiniMaxM3DenseLayerWeights, MiniMaxM3MoeLayerWeights};

use super::expert_pipeline::ExpertDecodePipeline;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxM3PrefillLayerKind {
    Dense,
    Moe,
}

pub type MiniMaxM3TerminalState<B> = crate::kv_cache::terminal_cache::TerminalState<<B as crate::backend::BackendResources>::Cache, <B as crate::backend::BackendResources>::Tensor>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxM3CoreLinearRole {
    AttentionQkv,
    AttentionOutput,
    FeedForwardGateUp,
    FeedForwardDown,
}

pub struct DensePrefillLayer<W> {
    input_norm: W,
    query: W,
    query_norm: W,
    key: W,
    key_norm: W,
    value: W,
    output: W,
    post_attention_norm: W,
    gate: W,
    up: W,
    down: W,
}

pub struct DensePrefillOutput<T> {
    pub hidden: T,
    pub key: T,
    pub value: T,
}

pub struct MoePrefillLayer<W> {
    input_norm: W,
    query: W,
    query_norm: W,
    key: W,
    key_norm: W,
    value: W,
    output: W,
    post_attention_norm: W,
    router_weight: W,
    router_bias: W,
    shared: DenseFfn<W>,
}

pub type MoeDecodeLayer<W> = MoePrefillLayer<W>;

pub type MiniMaxM3OutputHead<W> = super::output::OutputHead<W>;

pub use crate::moe::DenseFfn;

pub struct MoePrefillOutput<T> {
    pub hidden: T,
    pub key: T,
    pub value: T,
    pub active_experts: Vec<usize>,
    pub expert_ids: Vec<u32>,
    pub route_weights: Vec<f32>,
}

pub fn prepare_dense_prefill_layer<B>(backend: &B, cfg: &MiniMaxM3Config, weights: &MiniMaxM3DenseLayerWeights) -> Result<DensePrefillLayer<B::Weight>, BackendError>
where
    B: Backend,
{
    Ok(DensePrefillLayer {
        input_norm: backend.prepare_f32(&weights.input_norm, 1, cfg.hidden_size)?,
        query: prepare_core_weight(backend, &weights.query)?,
        query_norm: backend.prepare_f32(&weights.query_norm, 1, cfg.head_dim)?,
        key: prepare_core_weight(backend, &weights.key)?,
        key_norm: backend.prepare_f32(&weights.key_norm, 1, cfg.head_dim)?,
        value: prepare_core_weight(backend, &weights.value)?,
        output: prepare_core_weight(backend, &weights.output)?,
        post_attention_norm: backend.prepare_f32(&weights.post_attention_norm, 1, cfg.hidden_size)?,
        gate: prepare_core_weight(backend, &weights.gate)?,
        up: prepare_core_weight(backend, &weights.up)?,
        down: prepare_core_weight(backend, &weights.down)?,
    })
}

pub fn prepare_moe_prefill_layer<B>(backend: &B, cfg: &MiniMaxM3Config, weights: &MiniMaxM3MoeLayerWeights) -> Result<MoePrefillLayer<B::Weight>, BackendError>
where
    B: Backend,
{
    if cfg.num_shared_experts != 1 {
        return Err(BackendError::Compute { msg: format!("MiniMax-M3 当前要求 1 个 shared expert，实际 {}", cfg.num_shared_experts) });
    }
    Ok(MoePrefillLayer {
        input_norm: backend.prepare_f32(&weights.input_norm, 1, cfg.hidden_size)?,
        query: prepare_core_weight(backend, &weights.query)?,
        query_norm: backend.prepare_f32(&weights.query_norm, 1, cfg.head_dim)?,
        key: prepare_core_weight(backend, &weights.key)?,
        key_norm: backend.prepare_f32(&weights.key_norm, 1, cfg.head_dim)?,
        value: prepare_core_weight(backend, &weights.value)?,
        output: prepare_core_weight(backend, &weights.output)?,
        post_attention_norm: backend.prepare_f32(&weights.post_attention_norm, 1, cfg.hidden_size)?,
        router_weight: backend.prepare_f32(&weights.router_weight, cfg.num_experts, cfg.hidden_size)?,
        router_bias: backend.prepare_f32(&weights.router_bias, 1, cfg.num_experts)?,
        shared: DenseFfn { gate: prepare_core_weight(backend, &weights.shared_gate)?, up: prepare_core_weight(backend, &weights.shared_up)?, down: prepare_core_weight(backend, &weights.shared_down)? },
    })
}

pub fn prepare_moe_decode_layer<B>(backend: &B, cfg: &MiniMaxM3Config, weights: &MiniMaxM3MoeLayerWeights) -> Result<MoeDecodeLayer<B::Weight>, BackendError>
where
    B: Backend,
{
    // MoeDecodeLayer 是 MoePrefillLayer 的 type alias,两层装配完全一致。
    prepare_moe_prefill_layer(backend, cfg, weights)
}

fn prepare_core_weight<B>(backend: &B, weight: &MiniMaxM3CoreMatrix) -> Result<B::Weight, BackendError>
where
    B: Backend,
{
    match weight {
        MiniMaxM3CoreMatrix::Mxfp8(weight) => backend.prepare_weight(LinearWeight::mxfp8(weight), weight.rows, weight.cols),
    }
}

pub fn prepare_output_head<B>(backend: &B, cfg: &MiniMaxM3Config, final_norm: &[f32], lm_head: LinearWeight<'_>) -> Result<MiniMaxM3OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    prepare_output_head_quantized(backend, cfg, final_norm, lm_head, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_output_head_quantized<B>(backend: &B, cfg: &MiniMaxM3Config, final_norm: &[f32], lm_head: LinearWeight<'_>, quantization: crate::weight::LmHeadQuantization) -> Result<MiniMaxM3OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    super::output::prepare_output_head_quantized(backend, final_norm, lm_head, cfg.vocab_size, cfg.hidden_size, quantization)
}

pub fn moe_expert_from_weights<W>(gate: W, up: W, down: W) -> DenseFfn<W> {
    DenseFfn { gate, up, down }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn attention_prefill<B>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    cache: &mut B::Cache,
    layer: usize,
    input_norm: &B::Weight,
    query_weight: &B::Weight,
    query_norm: &B::Weight,
    key_weight: &B::Weight,
    key_norm: &B::Weight,
    value_weight: &B::Weight,
    output_weight: &B::Weight,
    post_attention_norm: &B::Weight,
    hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<(B::Tensor, B::Tensor, B::Tensor, B::Tensor), BackendError>
where
    B: GqaPrefillBackend,
{
    attention_prefill_observed(backend, cfg, cache, layer, input_norm, query_weight, query_norm, key_weight, key_norm, value_weight, output_weight, post_attention_norm, hidden, rope, position, |_, _| {})
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn attention_prefill_observed<B, O>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    cache: &mut B::Cache,
    layer: usize,
    input_norm: &B::Weight,
    query_weight: &B::Weight,
    query_norm: &B::Weight,
    key_weight: &B::Weight,
    key_norm: &B::Weight,
    value_weight: &B::Weight,
    output_weight: &B::Weight,
    post_attention_norm: &B::Weight,
    hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
    mut observe: O,
) -> Result<(B::Tensor, B::Tensor, B::Tensor, B::Tensor), BackendError>
where
    B: GqaPrefillBackend,
    O: FnMut(MiniMaxM3CoreLinearRole, &B::Tensor),
{
    let spec = GqaSpec {
        num_heads: cfg.num_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        rope_dim: cfg.rope_dim,
        rope_theta: cfg.rope_theta,
        use_qk_norm: cfg.use_qk_norm,
        window: crate::attention::gqa::CausalWindow::Full,
        score_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        output_gate: false,
    };
    let normed = backend.gemma_rmsnorm(hidden, input_norm, cfg.rms_eps)?;
    observe(MiniMaxM3CoreLinearRole::AttentionQkv, &normed);
    let query = backend.linear(&normed, query_weight)?;
    let key = backend.linear(&normed, key_weight)?;
    let value = backend.linear(&normed, value_weight)?;
    let query = backend.gemma_rmsnorm_heads(&query, query_norm, cfg.num_heads, cfg.head_dim, cfg.rms_eps)?;
    let key = backend.gemma_rmsnorm_heads(&key, key_norm, cfg.num_kv_heads, cfg.head_dim, cfg.rms_eps)?;
    let query = backend.rope_prefix(&query, cfg.num_heads, cfg.rope_dim, crate::attention::rope::RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
    let key = backend.rope_prefix(&key, cfg.num_kv_heads, cfg.rope_dim, crate::attention::rope::RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
    let attention = backend.gqa_prefill_attention_cached(cache, layer, position, &query, &key, &value, &spec, false)?;
    observe(MiniMaxM3CoreLinearRole::AttentionOutput, &attention);
    let residual = backend.linear_add(&attention, output_weight, hidden)?;
    let normed = backend.gemma_rmsnorm_f32(&residual, post_attention_norm, cfg.rms_eps)?;
    observe(MiniMaxM3CoreLinearRole::FeedForwardGateUp, &normed);
    Ok((residual, normed, key, value))
}

#[allow(clippy::too_many_arguments)]
pub fn dense_layer_prefill<B>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    weights: &DensePrefillLayer<B::Weight>,
    cache: &mut B::Cache,
    layer: usize,
    hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<DensePrefillOutput<B::Tensor>, BackendError>
where
    B: GqaPrefillBackend,
{
    dense_layer_prefill_core_observed(backend, cfg, weights, cache, layer, hidden, rope, position, |_, _| {})
}

#[allow(clippy::too_many_arguments)]
pub fn dense_layer_prefill_core_observed<B, O>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    weights: &DensePrefillLayer<B::Weight>,
    cache: &mut B::Cache,
    layer: usize,
    hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
    mut observe: O,
) -> Result<DensePrefillOutput<B::Tensor>, BackendError>
where
    B: GqaPrefillBackend,
    O: FnMut(MiniMaxM3CoreLinearRole, &B::Tensor),
{
    ensure_sequence_supported(cfg, position, backend.token_rows(hidden))?;
    if backend.token_rows(hidden) == 1 {
        backend.begin_decode_batch();
    } else {
        backend.begin_batch();
    }
    let (attention_residual, normed, key, value) = attention_prefill_observed(
        backend,
        cfg,
        cache,
        layer,
        &weights.input_norm,
        &weights.query,
        &weights.query_norm,
        &weights.key,
        &weights.key_norm,
        &weights.value,
        &weights.output,
        &weights.post_attention_norm,
        hidden,
        rope,
        position,
        |role, tensor| observe(role, tensor),
    )?;
    let spec = DenseMlpSpec { intermediate_size: cfg.dense_intermediate_size, activation: Activation::SwigluOai { alpha: cfg.swiglu_alpha, limit: cfg.swiglu_limit } };
    let feedforward = dense_mlp::forward_observed(backend, &spec, DenseMlpWeightsRef { gate: &weights.gate, up: &weights.up, down: &weights.down }, &normed, |tensor| observe(MiniMaxM3CoreLinearRole::FeedForwardDown, tensor))?;
    let hidden = backend.add(&attention_residual, &feedforward)?;
    Ok(DensePrefillOutput { hidden, key, value })
}

#[allow(clippy::too_many_arguments)]
pub fn moe_layer_prefill<B>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    weights: &MoePrefillLayer<B::Weight>,
    experts: &mut B::PrefillExperts,
    cache: &mut B::Cache,
    layer: usize,
    hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<MoePrefillOutput<B::Tensor>, BackendError>
where
    B: ExpertPrefillBackend + GqaPrefillBackend,
{
    moe_layer_prefill_core_observed(backend, cfg, weights, experts, cache, layer, hidden, rope, position, |_, _| {})
}

#[allow(clippy::too_many_arguments)]
pub fn moe_layer_prefill_core_observed<B, O>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    weights: &MoePrefillLayer<B::Weight>,
    experts: &mut B::PrefillExperts,
    cache: &mut B::Cache,
    layer: usize,
    hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
    mut observe: O,
) -> Result<MoePrefillOutput<B::Tensor>, BackendError>
where
    B: ExpertPrefillBackend + GqaPrefillBackend,
    O: FnMut(MiniMaxM3CoreLinearRole, &B::Tensor),
{
    ensure_sequence_supported(cfg, position, backend.token_rows(hidden))?;
    backend.begin_batch();
    let (attention_residual, normed, key, value) = attention_prefill_observed(
        backend,
        cfg,
        cache,
        layer,
        &weights.input_norm,
        &weights.query,
        &weights.query_norm,
        &weights.key,
        &weights.key_norm,
        &weights.value,
        &weights.output,
        &weights.post_attention_norm,
        hidden,
        rope,
        position,
        |role, tensor| observe(role, tensor),
    )?;
    let shared = [SharedExpertRef { gate: &weights.shared.gate, up: &weights.shared.up, down: &weights.shared.down, output_gate: None }];
    let ffn = crate::moe::prefill::prefill_experts_observed(
        backend,
        &topk_moe_spec(cfg),
        &MoeFfnRef { router_weight: &weights.router_weight, router_bias: &weights.router_bias, shared_experts: &shared, selected_experts: None },
        layer,
        experts,
        &normed,
        None,
        |tensor| observe(MiniMaxM3CoreLinearRole::FeedForwardDown, tensor),
    )?;
    let active_experts = crate::moe::routing::active_experts_from_ids(&ffn.routing.expert_ids, cfg.num_experts).map_err(|msg| BackendError::Compute { msg })?;
    let hidden = backend.add(&attention_residual, &ffn.tensor)?;
    Ok(MoePrefillOutput { hidden, key, value, active_experts, expert_ids: ffn.routing.expert_ids, route_weights: ffn.routing.weights })
}

pub enum MiniMaxM3PrefillLayer<W> {
    Dense(DensePrefillLayer<W>),
    Moe(MoePrefillLayer<W>),
}

pub enum MiniMaxM3DecodeLayer<W> {
    Dense(DensePrefillLayer<W>),
    Moe(MoeDecodeLayer<W>),
}

pub struct MiniMaxM3RoundOutput<T> {
    pub hidden: T,
    pub output_input: T,
    pub logits: T,
    pub token_id: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn minimax_m3_prefill_hidden<B, L>(backend: &B, cfg: &MiniMaxM3Config, experts: &mut B::PrefillExperts, cache: &mut B::Cache, hidden: B::Tensor, rope: &RopeTable, position: usize, load_layer: L) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + GqaPrefillBackend,
    L: FnMut(usize, MiniMaxM3PrefillLayerKind) -> Result<MiniMaxM3PrefillLayer<B::Weight>, BackendError>,
{
    minimax_m3_prefill_hidden_core_observed(backend, cfg, experts, cache, hidden, rope, position, load_layer, |_, _, _| {})
}

#[allow(clippy::too_many_arguments)]
pub fn minimax_m3_prefill_hidden_core_observed<B, L, O>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    experts: &mut B::PrefillExperts,
    cache: &mut B::Cache,
    hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
    mut load_layer: L,
    mut observe: O,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + GqaPrefillBackend,
    L: FnMut(usize, MiniMaxM3PrefillLayerKind) -> Result<MiniMaxM3PrefillLayer<B::Weight>, BackendError>,
    O: FnMut(usize, MiniMaxM3CoreLinearRole, &B::Tensor),
{
    let token_count = backend.token_rows(&hidden);
    if token_count == 0 || backend.token_cols(&hidden) != cfg.hidden_size {
        return Err(BackendError::Compute { msg: format!("MiniMax-M3 prefill hidden=[{},{}]，期望 rows>0, cols={}", token_count, backend.token_cols(&hidden), cfg.hidden_size) });
    }
    ensure_sequence_supported(cfg, position, token_count)?;
    let result = super::prefill::run_full_token_layers(backend, token_count, cfg.layer_count, hidden, |layer, hidden| {
        let kind = if layer < cfg.dense_layer_count { MiniMaxM3PrefillLayerKind::Dense } else { MiniMaxM3PrefillLayerKind::Moe };
        match (kind, load_layer(layer, kind)?) {
            (MiniMaxM3PrefillLayerKind::Dense, MiniMaxM3PrefillLayer::Dense(weights)) => {
                Ok(dense_layer_prefill_core_observed(backend, cfg, &weights, cache, layer, &hidden, rope, position, |role, tensor| observe(layer, role, tensor))?.hidden)
            }
            (MiniMaxM3PrefillLayerKind::Moe, MiniMaxM3PrefillLayer::Moe(weights)) => {
                Ok(moe_layer_prefill_core_observed(backend, cfg, &weights, experts, cache, layer, &hidden, rope, position, |role, tensor| observe(layer, role, tensor))?.hidden)
            }
            _ => Err(BackendError::Compute { msg: format!("MiniMax-M3 L{layer} prefill layer 类型不匹配") }),
        }
    });
    backend.finish_batch();
    result
}

#[allow(clippy::too_many_arguments)]
pub fn minimax_m3_prefill_hidden_resident_observed<B, O>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    layers: &[MiniMaxM3DecodeLayer<B::Weight>],
    experts: &mut B::PrefillExperts,
    cache: &mut B::Cache,
    hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
    mut observe_route: O,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + GqaPrefillBackend,
    O: FnMut(usize, usize, usize, &[u32]) -> Result<(), BackendError>,
{
    let token_count = backend.token_rows(&hidden);
    if layers.len() != cfg.layer_count || token_count == 0 || backend.token_cols(&hidden) != cfg.hidden_size {
        return Err(BackendError::Compute { msg: format!("MiniMax-M3 resident prefill 输入不完整: layers={}/{}, hidden=[{},{}]", layers.len(), cfg.layer_count, token_count, backend.token_cols(&hidden),) });
    }
    ensure_sequence_supported(cfg, position, token_count)?;
    let result = super::prefill::run_full_token_layers(backend, token_count, cfg.layer_count, hidden, |layer, hidden| match &layers[layer] {
        MiniMaxM3DecodeLayer::Dense(weights) if layer < cfg.dense_layer_count => Ok(dense_layer_prefill(backend, cfg, weights, cache, layer, &hidden, rope, position)?.hidden),
        MiniMaxM3DecodeLayer::Moe(weights) if layer >= cfg.dense_layer_count => {
            let output = moe_layer_prefill(backend, cfg, weights, experts, cache, layer, &hidden, rope, position)?;
            observe_route(layer, token_count, cfg.num_experts_per_tok, &output.expert_ids)?;
            Ok(output.hidden)
        }
        _ => Err(BackendError::Compute { msg: format!("MiniMax-M3 L{layer} resident prefill layer 类型不匹配") }),
    });
    backend.finish_batch();
    result
}

#[allow(clippy::too_many_arguments)]
pub fn dense_layer_decode<B>(backend: &B, cfg: &MiniMaxM3Config, weights: &DensePrefillLayer<B::Weight>, cache: &mut B::Cache, layer: usize, hidden: &B::Tensor, rope: &RopeTable, position: usize) -> Result<B::Tensor, BackendError>
where
    B: GqaPrefillBackend,
{
    Ok(dense_layer_prefill(backend, cfg, weights, cache, layer, hidden, rope, position)?.hidden)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_layer_decode<B, S>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    weights: &MoeDecodeLayer<B::Weight>,
    cache: &mut B::Cache,
    layer: usize,
    expert_sources: &S,
    state: &mut ExpertDecodePipeline<B::MoeState>,
    hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertDecodeBackend + GqaPrefillBackend,
    S: ExpertSourceProvider,
{
    ensure_sequence_supported(cfg, position, 1)?;
    backend.begin_decode_batch();
    let (attention_residual, normed, _, _) =
        attention_prefill(backend, cfg, cache, layer, &weights.input_norm, &weights.query, &weights.query_norm, &weights.key, &weights.key_norm, &weights.value, &weights.output, &weights.post_attention_norm, hidden, rope, position)?;
    let shared = [SharedExpertRef { gate: &weights.shared.gate, up: &weights.shared.up, down: &weights.shared.down, output_gate: None }];
    let ffn_weights = MoeFfnRef { router_weight: &weights.router_weight, router_bias: &weights.router_bias, shared_experts: &shared, selected_experts: None };
    let source = expert_sources.source(layer).map_err(BackendError::ExpertLoad)?;
    let next_source = (layer + 1 < cfg.layer_count).then(|| expert_sources.source(layer + 1).map(|source| (layer + 1, source))).transpose().map_err(BackendError::ExpertLoad)?;
    let request = crate::runtime::expert_pipeline::ExpertDecodeRequest { layer, source, position, next: next_source };
    let feedforward = state.decode(backend, &topk_moe_spec(cfg), &ffn_weights, request, &normed)?;
    backend.add(&attention_residual, &feedforward)
}

#[allow(clippy::too_many_arguments)]
pub fn minimax_m3_decode_round<B, S>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    layers: &[MiniMaxM3DecodeLayer<B::Weight>],
    expert_sources: &S,
    state: &mut ExpertDecodePipeline<B::MoeState>,
    cache: &mut B::Cache,
    mut hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
    head: &MiniMaxM3OutputHead<B::Weight>,
) -> Result<MiniMaxM3RoundOutput<B::Tensor>, BackendError>
where
    B: ExpertDecodeBackend + GqaPrefillBackend,
    S: ExpertSourceProvider,
{
    if layers.len() != cfg.layer_count || backend.token_rows(&hidden) != 1 || backend.token_cols(&hidden) != cfg.hidden_size {
        return Err(BackendError::Compute { msg: format!("MiniMax-M3 decode 输入不完整: layers={}/{}, hidden=[{},{}]", layers.len(), cfg.layer_count, backend.token_rows(&hidden), backend.token_cols(&hidden)) });
    }
    ensure_sequence_supported(cfg, position, 1)?;
    let result = (|| {
        for (layer, weights) in layers.iter().enumerate() {
            let _layer_scope = backend.layer_scope();
            hidden = match weights {
                MiniMaxM3DecodeLayer::Dense(weights) if layer < cfg.dense_layer_count => dense_layer_decode(backend, cfg, weights, cache, layer, &hidden, rope, position)?,
                MiniMaxM3DecodeLayer::Moe(weights) if layer >= cfg.dense_layer_count => moe_layer_decode(backend, cfg, weights, cache, layer, expert_sources, state, &hidden, rope, position)?,
                _ => return Err(BackendError::Compute { msg: format!("MiniMax-M3 L{layer} decode layer 类型不匹配") }),
            };
            backend.submit_batch();
        }
        minimax_m3_token_output(backend, cfg, head, hidden)
    })();
    backend.finish_batch();
    result
}

#[allow(clippy::too_many_arguments)]
pub fn minimax_m3_decode_round_streamed<B, S, L>(
    backend: &B,
    cfg: &MiniMaxM3Config,
    expert_sources: &S,
    state: &mut ExpertDecodePipeline<B::MoeState>,
    cache: &mut B::Cache,
    mut hidden: B::Tensor,
    rope: &RopeTable,
    position: usize,
    head: &MiniMaxM3OutputHead<B::Weight>,
    mut load_layer: L,
) -> Result<MiniMaxM3RoundOutput<B::Tensor>, BackendError>
where
    B: ExpertDecodeBackend + GqaPrefillBackend,
    S: ExpertSourceProvider,
    L: FnMut(usize, MiniMaxM3PrefillLayerKind) -> Result<MiniMaxM3DecodeLayer<B::Weight>, BackendError>,
{
    if backend.token_rows(&hidden) != 1 || backend.token_cols(&hidden) != cfg.hidden_size {
        return Err(BackendError::Compute { msg: format!("MiniMax-M3 streamed decode hidden=[{},{}]，期望 [1,{}]", backend.token_rows(&hidden), backend.token_cols(&hidden), cfg.hidden_size) });
    }
    ensure_sequence_supported(cfg, position, 1)?;
    let result = (|| {
        for layer in 0..cfg.layer_count {
            let _layer_scope = backend.layer_scope();
            let kind = if layer < cfg.dense_layer_count { MiniMaxM3PrefillLayerKind::Dense } else { MiniMaxM3PrefillLayerKind::Moe };
            hidden = match (kind, load_layer(layer, kind)?) {
                (MiniMaxM3PrefillLayerKind::Dense, MiniMaxM3DecodeLayer::Dense(weights)) => dense_layer_decode(backend, cfg, &weights, cache, layer, &hidden, rope, position)?,
                (MiniMaxM3PrefillLayerKind::Moe, MiniMaxM3DecodeLayer::Moe(weights)) => moe_layer_decode(backend, cfg, &weights, cache, layer, expert_sources, state, &hidden, rope, position)?,
                _ => return Err(BackendError::Compute { msg: format!("MiniMax-M3 L{layer} streamed decode layer 类型不匹配") }),
            };
            backend.submit_batch();
        }
        minimax_m3_token_output(backend, cfg, head, hidden)
    })();
    backend.finish_batch();
    result
}

fn minimax_m3_token_output<B>(backend: &B, cfg: &MiniMaxM3Config, head: &MiniMaxM3OutputHead<B::Weight>, hidden: B::Tensor) -> Result<MiniMaxM3RoundOutput<B::Tensor>, BackendError>
where
    B: Backend,
{
    let output = super::output::token_output(backend, head, &hidden, &super::output::OutputPlan { eps: cfg.rms_eps, norm: super::output::OutputNorm::GemmaRms, excluded_tokens: Vec::new() })?;
    Ok(MiniMaxM3RoundOutput { hidden, output_input: output.input, logits: output.logits, token_id: output.token_id })
}

pub fn minimax_m3_output<B>(backend: &B, cfg: &MiniMaxM3Config, head: &MiniMaxM3OutputHead<B::Weight>, hidden: B::Tensor, last_row: usize) -> Result<MiniMaxM3RoundOutput<B::Tensor>, BackendError>
where
    B: Backend,
{
    let output = super::output::last_token_output(backend, head, &hidden, last_row, &super::output::OutputPlan { eps: cfg.rms_eps, norm: super::output::OutputNorm::GemmaRms, excluded_tokens: Vec::new() })?;
    Ok(MiniMaxM3RoundOutput { hidden, output_input: output.input, logits: output.logits, token_id: output.token_id })
}

fn topk_moe_spec(cfg: &MiniMaxM3Config) -> TopkMoeSpec {
    TopkMoeSpec {
        num_experts: cfg.num_experts,
        top_k: cfg.num_experts_per_tok,
        num_shared_experts: cfg.num_shared_experts,
        scoring_func: ScoringFunc::SigmoidBias,
        normalize_selected: true,
        routed_scaling_factor: cfg.routed_scaling_factor,
        intermediate_size: cfg.expert_intermediate_size,
        shared_intermediate_size: cfg.shared_intermediate_size,
        activation: Activation::SwigluOai { alpha: cfg.swiglu_alpha, limit: cfg.swiglu_limit },
    }
}

fn ensure_sequence_supported(cfg: &MiniMaxM3Config, position: usize, rows: usize) -> Result<(), BackendError> {
    let limit = cfg.msa_block_size.checked_mul(cfg.msa_topk_blocks).ok_or_else(|| BackendError::Compute { msg: "MiniMax-M3 MSA dense 等价长度溢出".to_owned() })?;
    let end = position.checked_add(rows).ok_or_else(|| BackendError::Compute { msg: "MiniMax-M3 position 溢出".to_owned() })?;
    if end > limit {
        return Err(BackendError::Compute { msg: format!("MiniMax-M3 稀疏 MSA 尚未接入：当前 dense GQA 支持 end_position <= {limit}，实际 {end}") });
    }
    Ok(())
}

/// MiniMax-M3 runtime 分阶段接入原因说明。
///
/// 先把阻塞项集中成可追踪消息,避免 "暂未接入" 提示缺少可执行信息。
pub fn ensure_supported(cfg: &MiniMaxM3Config) -> Result<(), BackendError> {
    let blockers = runtime_blockers(cfg);
    if blockers.is_empty() {
        return Ok(());
    }
    Err(BackendError::Compute { msg: format!("MiniMax-M3 运行时尚未接入：{}", blockers.join("；")) })
}

/// 生成当前配置下的接入阻塞项清单。
///
/// 这里不判断文件/路径是否存在,仅判断架构与能力面是否完整。
pub fn runtime_blockers(cfg: &MiniMaxM3Config) -> Vec<&'static str> {
    let mut blockers: Vec<&'static str> = Vec::new();

    if cfg.layer_count == 0 || cfg.dense_layer_count > cfg.layer_count || cfg.num_kv_heads == 0 || !cfg.num_heads.is_multiple_of(cfg.num_kv_heads) {
        blockers.push("配置层存在结构性不一致");
    }
    if cfg.eos_token_ids.is_empty() {
        blockers.push("eos_token_ids 尚未对齐官方 EOS token 列表");
    }
    blockers
}

pub use crate::model_spec::minimax_m3::{MiniMaxM3Config, MiniMaxM3VisionConfig};
pub struct MiniMaxM3 {
    config: MiniMaxM3Config,
    layer_specs: Vec<LayerSpec>,
}

impl MiniMaxM3 {
    pub fn new(config: MiniMaxM3Config) -> Result<Self, ModelError> {
        if config.layer_count == 0 || config.dense_layer_count > config.layer_count {
            return Err(ModelError::InvalidArchitecture("dense_layer_count > layer_count".into()));
        }
        if config.num_experts_per_tok == 0 || config.num_experts_per_tok > config.num_experts {
            return Err(ModelError::InvalidArchitecture("num_experts_per_tok 必须在 1..=num_experts".into()));
        }
        if !config.num_heads.is_multiple_of(config.num_kv_heads) {
            return Err(ModelError::InvalidArchitecture("num_heads 必须能被 num_kv_heads 整除".into()));
        }
        if config.vision.num_heads == 0 || !config.vision.hidden_size.is_multiple_of(config.vision.num_heads) {
            return Err(ModelError::InvalidArchitecture("vision hidden_size 必须能被 num_heads 整除".into()));
        }
        if config.vision.projection_dim != config.hidden_size {
            return Err(ModelError::InvalidArchitecture("vision projection_dim 必须等于语言模型 hidden_size".into()));
        }
        let layer_specs = (0..config.layer_count).map(|layer| Self::build_layer_spec(&config, layer)).collect();
        Ok(Self { config, layer_specs })
    }

    pub fn standard() -> Self {
        Self::new(MiniMaxM3Config::standard()).expect("MiniMax-M3 标准配置必须有效")
    }

    fn build_layer_spec(config: &MiniMaxM3Config, layer: LayerId) -> LayerSpec {
        let is_dense = layer < config.dense_layer_count;
        let gqa = GqaSpec {
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            rope_dim: config.rope_dim,
            rope_theta: config.rope_theta,
            use_qk_norm: config.use_qk_norm,
            window: crate::attention::gqa::CausalWindow::Full,
            score_scale: 1.0 / (config.head_dim as f32).sqrt(),
            output_gate: false,
        };
        let attention = if is_dense {
            AttentionSpec::Gqa(gqa)
        } else {
            AttentionSpec::Msa(MsaSpec { gqa, index_dim: config.msa_index_dim, num_index_heads: config.num_kv_heads, block_size: config.msa_block_size, topk_blocks: config.msa_topk_blocks, init_block: 0, local_block: 1 })
        };
        let activation = Activation::SwigluOai { alpha: config.swiglu_alpha, limit: config.swiglu_limit };
        let feedforward = if is_dense {
            FeedforwardSpec::Dense(DenseMlpSpec { intermediate_size: config.dense_intermediate_size, activation })
        } else {
            FeedforwardSpec::TopkMoe(TopkMoeSpec {
                num_experts: config.num_experts,
                top_k: config.num_experts_per_tok,
                num_shared_experts: config.num_shared_experts,
                scoring_func: ScoringFunc::SigmoidBias,
                normalize_selected: true,
                routed_scaling_factor: config.routed_scaling_factor,
                intermediate_size: config.expert_intermediate_size,
                shared_intermediate_size: config.shared_intermediate_size,
                activation,
            })
        };
        LayerSpec { attention, feedforward, input_norm: NormSpec::GemmaRms { eps: config.rms_eps }, post_attention_norm: NormSpec::GemmaRms { eps: config.rms_eps }, post_norm: None }
    }
}

impl Model for MiniMaxM3 {
    type Config = MiniMaxM3Config;

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
