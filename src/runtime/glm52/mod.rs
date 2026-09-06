//! GLM-5.2 模型算法与执行组合。

use std::time::Instant;

pub mod cpu;
pub mod dspark;
#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
#[cfg_attr(test, allow(dead_code))]
pub mod dspark_cpu;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod dspark_rocm;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
mod protocol;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_node;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_swap;
pub mod stage;
#[cfg(any(test, all(target_os = "linux", feature = "with-rocm")))]
pub mod tool;

use crate::{
    attention::dsa::{DsaSpec, DsaWeightsRef},
    attention::rope::RopeTable,
    attention::{
        AttentionSpec,
        mla::{MlaSpec, mla_causal_prefill_rope},
    },
    backend::{Backend, BackendError, DecodeBackend, DsaPrefillBackend, ExpertDecodeBackend, ExpertPrefillBackend, LinearWeight, SegmentedTensorBackend},
    moe::{
        Activation, FeedforwardSpec,
        dense_mlp::{DenseMlpSpec, DenseMlpWeightsRef},
        expert_predictor::ExpertRouteTrace,
        topk_moe::{MoeFfnRef, RoutedMoeInputs, RoutedMoeWeightsRef, ScoringFunc, SharedExpertRef, TopkMoeSpec},
    },
    norm::NormSpec,
    runtime::{LayerId, LayerSpec, Model, ModelError},
    weight::container::gguf::GgufMatrix,
    weight::format::compressed_tensors_hybrid::{CtLinearWeight, CtMatrix},
    weight::format::nvfp4::NvidiaNvfp4Experts,
    weight::format::official_fp8::OfficialExpertArchive,
    weight::model::glm52::{CtDenseLayer, CtIndexerWeights, CtMoeLayer, CtMtpLayer},
    weight::model::glm52::{DenseLayerF32, Glm52IndexerWeights, MoeLayerF32},
    weight::model::glm52::{GgufDenseLayer, GgufIndexerWeights, GgufMoeLayer, GgufMtpLayer},
    weight::model::glm52::{Nvfp4CoreDenseLayer, Nvfp4CoreIndexer, Nvfp4CoreMoeLayer},
    weight::{
        Glm52Weights,
        expert_source::{ExpertSourceProvider, Glm52ExpertSources},
    },
};

/// 按实际权重来源解析 tokenizer，优先使用量化权重目录。
pub fn tokenizer_path(model_dir: &std::path::Path, ct_root: Option<&std::path::Path>, nvfp4_root: Option<&std::path::Path>) -> std::path::PathBuf {
    ct_root.or(nvfp4_root).map(|root| root.join("tokenizer.json")).filter(|path| path.is_file()).unwrap_or_else(|| model_dir.join("tokenizer.json"))
}

/// 按 compressed-tensors → NVFP4 → GGUF → official 的优先级打开 GLM-5.2 权重。
pub fn open_weights(cfg: &Glm52Config, model_dir: &std::path::Path, ct_root: Option<&std::path::Path>, nvfp4_root: Option<&std::path::Path>, gguf_root: Option<&std::path::Path>) -> Result<Glm52Weights, String> {
    if let Some(root) = ct_root {
        eprintln!("[ct] 使用 compressed-tensors 权重: {}", root.display());
        let weights = Glm52Weights::open_compressed_tensors(root, cfg.clone());
        eprintln!("[ct] open 结果: {}", if weights.is_ok() { "OK" } else { "FAIL" });
        weights.map_err(|error| error.to_string())
    } else if let Some(root) = nvfp4_root {
        eprintln!("[nvfp4] 使用 NVFP4 权重: {}", root.display());
        Glm52Weights::open_nvfp4(root, cfg.clone()).map_err(|error| error.to_string())
    } else if let Some(root) = gguf_root {
        eprintln!("[gguf] 使用 GGUF 权重: {}", root.display());
        Glm52Weights::open_gguf(root, cfg.clone()).map_err(|error| error.to_string())
    } else {
        eprintln!("[official] 使用官方 safetensors 权重: {}", model_dir.display());
        Glm52Weights::open_official(model_dir, cfg.clone()).map_err(|error| error.to_string())
    }
}

fn decode_expert_sources(weights: &Glm52Weights, nvfp4_source: Option<&NvidiaNvfp4Experts>, model_dir: &std::path::Path, cfg: &Glm52Config) -> Result<Glm52ExpertSources, String> {
    if weights.source_is_ct() {
        Ok(Glm52ExpertSources::W4A16(weights.ct_source()?))
    } else if weights.source_is_gguf() {
        Ok(Glm52ExpertSources::Gguf(weights.gguf_source()?))
    } else if let Some(source) = nvfp4_source {
        Ok(Glm52ExpertSources::Nvfp4(source.clone()))
    } else {
        Ok(Glm52ExpertSources::Fp8(OfficialExpertArchive::open(model_dir, cfg.expert_intermediate_size, cfg.hidden_size, cfg.expert_count)?))
    }
}

/// 各执行路径共用的 DSA 规格,字段全部来自 cfg/mla;集中一处避免多处字面构造漂移。
pub(crate) fn glm52_dsa_spec(cfg: &Glm52Config, mla: &MlaSpec) -> DsaSpec {
    DsaSpec { num_heads: cfg.index_heads, head_dim: cfg.index_head_dim, rope_dim: mla.qk_rope_head_dim, top_k: cfg.index_top_k, rotary_layout: mla.rotary_layout, kpool: 0, always_select_tail: false }
}

fn glm52_topk_moe_spec(cfg: &Glm52Config) -> TopkMoeSpec {
    TopkMoeSpec {
        num_experts: cfg.expert_count,
        top_k: cfg.expert_top_k,
        num_shared_experts: 1,
        scoring_func: ScoringFunc::SigmoidBias,
        normalize_selected: true,
        routed_scaling_factor: cfg.routed_scaling_factor,
        intermediate_size: cfg.expert_intermediate_size,
        shared_intermediate_size: cfg.expert_intermediate_size,
        activation: Activation::Silu,
    }
}

/// MoE 路由规格与 shared expert 视图;`MoeFfnRef` 借用 shared 数组,只能在调用点装配。
fn glm52_moe_spec<'a, W>(cfg: &Glm52Config, shared_gate: &'a W, shared_up: &'a W, shared_down: &'a W) -> (TopkMoeSpec, [SharedExpertRef<'a, W>; 1]) {
    let shared = [SharedExpertRef { gate: shared_gate, up: shared_up, down: shared_down, output_gate: None }];
    (glm52_topk_moe_spec(cfg), shared)
}

/// 各后端 CLI 共用的 expert 路由统计输出。
pub(crate) fn print_expert_cache(state: &ExpertDecodePipeline<crate::moe::UncachedMoeState>) {
    let pipeline = state.stats();
    eprintln!("[expert-routing] policy=uncached routed={} prefetch={}", state.backend_state().routed_experts(), pipeline.prefetch_requested,);
}

/// 把单个 token id 增量 detokenize 并打印(不换行,立即 flush 保证流式可见)。
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "with-rocm")))]
pub(crate) fn print_token(detokenizer: &crate::tokenizer::Detokenizer, token: u32, skip_special_tokens: bool) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let bytes = detokenizer.decode_bytes(&[token], skip_special_tokens).map_err(|e| format!("detokenize {token}: {e}"))?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&bytes).map_err(|e| format!("print token: {e}"))?;
    stdout.flush().map_err(|e| format!("flush token: {e}"))?;
    Ok(())
}

use super::{expert_pipeline::ExpertDecodePipeline, prefill::run_full_token_layers_prefetched};

pub type Glm52TerminalState<B> = crate::kv_cache::terminal_cache::TerminalState<<B as crate::backend::BackendResources>::Cache, <B as crate::backend::BackendResources>::Tensor, <B as DecodeBackend>::DsaState>;

#[derive(Debug, Clone, Copy)]
pub enum Glm52PrefillLayerKind {
    Dense,
    Moe,
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_prefill_stage<B, S, P, F>(backend: &B, cfg: &Glm52Config, token_count: usize, start_layer: usize, end_layer: usize, hidden: B::Tensor, state: &mut S, mut prefetch_layer: P, mut run_layer: F) -> Result<B::Tensor, BackendError>
where
    B: Backend,
    P: FnMut(&mut S, usize, Glm52PrefillLayerKind) -> Result<(), BackendError>,
    F: FnMut(&mut S, usize, Glm52PrefillLayerKind, B::Tensor) -> Result<B::Tensor, BackendError>,
{
    glm52_prefill_stage_observed(backend, cfg, token_count, start_layer, end_layer, hidden, state, &mut prefetch_layer, &mut run_layer, |_, _| Ok(()))
}

/// 与 `glm52_prefill_stage` 相同，但在每层输出边界同步通知模型 adapter。
/// boundary=0 的 embedding 由调用方在进入本函数前处理；这里报告 layer+1。
#[allow(clippy::too_many_arguments)]
pub fn glm52_prefill_stage_observed<B, S, P, F, O>(
    backend: &B,
    cfg: &Glm52Config,
    token_count: usize,
    start_layer: usize,
    end_layer: usize,
    hidden: B::Tensor,
    state: &mut S,
    mut prefetch_layer: P,
    mut run_layer: F,
    mut observe: O,
) -> Result<B::Tensor, BackendError>
where
    B: Backend,
    P: FnMut(&mut S, usize, Glm52PrefillLayerKind) -> Result<(), BackendError>,
    F: FnMut(&mut S, usize, Glm52PrefillLayerKind, B::Tensor) -> Result<B::Tensor, BackendError>,
    O: FnMut(usize, &B::Tensor) -> Result<(), BackendError>,
{
    if start_layer > end_layer || end_layer > cfg.layer_count {
        return Err(BackendError::Compute { msg: format!("GLM-5.2 prefill 层区间非法: {start_layer}..{end_layer}, 总层数 {}", cfg.layer_count) });
    }
    let result = run_full_token_layers_prefetched(
        backend,
        token_count,
        end_layer,
        start_layer,
        hidden,
        state,
        |state, layer| {
            let kind = if layer < cfg.dense_layer_count { Glm52PrefillLayerKind::Dense } else { Glm52PrefillLayerKind::Moe };
            prefetch_layer(state, layer, kind)
        },
        |state, layer, hidden| {
            let kind = if layer < cfg.dense_layer_count { Glm52PrefillLayerKind::Dense } else { Glm52PrefillLayerKind::Moe };
            let output = run_layer(state, layer, kind, hidden)?;
            observe(layer + 1, &output)?;
            Ok(output)
        },
    );
    backend.finish_batch();
    result
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_prefill<B, S, P, F>(backend: &B, cfg: &Glm52Config, token_count: usize, start_layer: usize, hidden: B::Tensor, state: &mut S, prefetch_layer: P, run_layer: F) -> Result<B::Tensor, BackendError>
where
    B: Backend,
    P: FnMut(&mut S, usize, Glm52PrefillLayerKind) -> Result<(), BackendError>,
    F: FnMut(&mut S, usize, Glm52PrefillLayerKind, B::Tensor) -> Result<B::Tensor, BackendError>,
{
    glm52_prefill_stage(backend, cfg, token_count, start_layer, cfg.layer_count, hidden, state, prefetch_layer, run_layer)
}

#[derive(Clone)]
pub struct Glm52IndexerDecodeWeights<W> {
    pub wq_b: W,
    pub wk: W,
    pub weights_proj: W,
    pub k_norm_weight: W,
    pub k_norm_bias: W,
}

#[derive(Clone)]
pub struct Glm52DenseDecodeLayer<W> {
    pub indexer: Option<Glm52IndexerDecodeWeights<W>>,
    input_norm: W,
    q_a_proj: W,
    q_a_norm: W,
    q_b_proj: W,
    kv_a_proj: W,
    kv_a_norm: W,
    kv_b_proj: W,
    o_proj: W,
    post_attn_norm: W,
    gate_proj: W,
    up_proj: W,
    down_proj: W,
}

/// prefill 与 decode 共用同一份 resident 装配。
pub type Glm52DensePrefillLayer<W> = Glm52DenseDecodeLayer<W>;

#[derive(Clone)]
pub struct Glm52MoeDecodeLayer<W> {
    pub indexer: Option<Glm52IndexerDecodeWeights<W>>,
    input_norm: W,
    q_a_proj: W,
    q_a_norm: W,
    q_b_proj: W,
    kv_a_proj: W,
    kv_a_norm: W,
    kv_b_proj: W,
    o_proj: W,
    post_attn_norm: W,
    router_weight: W,
    router_bias: W,
    shared_gate: W,
    shared_up: W,
    shared_down: W,
}

pub enum Glm52DecodeLayer<W> {
    Dense(Glm52DenseDecodeLayer<W>),
    Moe(Glm52MoeDecodeLayer<W>),
}

pub struct Glm52Mtp<W> {
    embedding_norm: W,
    hidden_norm: W,
    input_projection: W,
    layer: Glm52MoeDecodeLayer<W>,
    output_norm: W,
}

/// prefill 与 decode 共用同一份 resident 装配。
pub type Glm52MoePrefillLayer<W> = Glm52MoeDecodeLayer<W>;

fn prepare_indexer<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, weights: &Glm52IndexerWeights) -> Result<Glm52IndexerDecodeWeights<B::Weight>, BackendError> {
    let query_size = cfg.index_heads * cfg.index_head_dim;
    Ok(Glm52IndexerDecodeWeights {
        wq_b: backend.prepare_weight(LinearWeight::fp8(&weights.wq_b), query_size, cfg.q_lora_rank)?,
        wk: backend.prepare_weight(LinearWeight::fp8(&weights.wk), cfg.index_head_dim, cfg.hidden_size)?,
        weights_proj: backend.prepare_weight(LinearWeight::Bf16Bytes(&weights.weights_proj), cfg.index_heads, cfg.hidden_size)?,
        k_norm_weight: backend.prepare_f32(&weights.k_norm_weight, 1, cfg.index_head_dim)?,
        k_norm_bias: backend.prepare_f32(&weights.k_norm_bias, 1, cfg.index_head_dim)?,
    })
}

fn prepare_nvfp4_indexer<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, weights: &Nvfp4CoreIndexer) -> Result<Glm52IndexerDecodeWeights<B::Weight>, BackendError> {
    let query_size = cfg.index_heads * cfg.index_head_dim;
    Ok(Glm52IndexerDecodeWeights {
        wq_b: backend.prepare_weight(LinearWeight::F16(&weights.wq_b), query_size, cfg.q_lora_rank)?,
        wk: backend.prepare_weight(LinearWeight::F16(&weights.wk), cfg.index_head_dim, cfg.hidden_size)?,
        weights_proj: backend.prepare_weight(LinearWeight::F16(&weights.weights_proj), cfg.index_heads, cfg.hidden_size)?,
        k_norm_weight: backend.prepare_weight(LinearWeight::F16(&weights.k_norm_weight), 1, cfg.index_head_dim)?,
        k_norm_bias: backend.prepare_weight(LinearWeight::F16(&weights.k_norm_bias), 1, cfg.index_head_dim)?,
    })
}

fn prepare_f16_f32<B: Backend>(backend: &B, values: &[half::f16], rows: usize, columns: usize) -> Result<B::Weight, BackendError> {
    let values = values.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
    backend.prepare_f32(&values, rows, columns)
}

pub fn prepare_dense_prefill_layer_nvfp4<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &Nvfp4CoreDenseLayer) -> Result<Glm52DensePrefillLayer<B::Weight>, BackendError> {
    Ok(Glm52DensePrefillLayer {
        indexer: weights.indexer.as_ref().map(|indexer| prepare_nvfp4_indexer(backend, cfg, indexer)).transpose()?,
        input_norm: backend.prepare_weight(LinearWeight::F16(&weights.input_norm), 1, cfg.hidden_size)?,
        q_a_proj: backend.prepare_weight(LinearWeight::F16(&weights.q_a_proj), mla.q_lora_rank, cfg.hidden_size)?,
        q_a_norm: backend.prepare_weight(LinearWeight::F16(&weights.q_a_norm), 1, mla.q_lora_rank)?,
        q_b_proj: backend.prepare_weight(LinearWeight::F16(&weights.q_b_proj), mla.q_projection_size, mla.q_lora_rank)?,
        kv_a_proj: backend.prepare_weight(LinearWeight::F16(&weights.kv_a_proj), mla.kv_a_out(), cfg.hidden_size)?,
        kv_a_norm: backend.prepare_weight(LinearWeight::F16(&weights.kv_a_norm), 1, mla.kv_lora_rank)?,
        kv_b_proj: backend.prepare_mla_kv_b(LinearWeight::F16(&weights.kv_b_proj), mla.kv_projection_size, mla.kv_lora_rank)?,
        o_proj: backend.prepare_weight(LinearWeight::F16(&weights.o_proj), cfg.hidden_size, mla.q_projection_size)?,
        post_attn_norm: backend.prepare_weight(LinearWeight::F16(&weights.post_attn_norm), 1, cfg.hidden_size)?,
        gate_proj: backend.prepare_weight(LinearWeight::F16(&weights.gate_proj), cfg.dense_intermediate_size, cfg.hidden_size)?,
        up_proj: backend.prepare_weight(LinearWeight::F16(&weights.up_proj), cfg.dense_intermediate_size, cfg.hidden_size)?,
        down_proj: backend.prepare_weight(LinearWeight::F16(&weights.down_proj), cfg.hidden_size, cfg.dense_intermediate_size)?,
    })
}

pub fn prepare_moe_prefill_layer_nvfp4<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &Nvfp4CoreMoeLayer) -> Result<Glm52MoePrefillLayer<B::Weight>, BackendError> {
    Ok(Glm52MoePrefillLayer {
        indexer: weights.indexer.as_ref().map(|indexer| prepare_nvfp4_indexer(backend, cfg, indexer)).transpose()?,
        input_norm: backend.prepare_weight(LinearWeight::F16(&weights.input_norm), 1, cfg.hidden_size)?,
        q_a_proj: backend.prepare_weight(LinearWeight::F16(&weights.q_a_proj), mla.q_lora_rank, cfg.hidden_size)?,
        q_a_norm: backend.prepare_weight(LinearWeight::F16(&weights.q_a_norm), 1, mla.q_lora_rank)?,
        q_b_proj: backend.prepare_weight(LinearWeight::F16(&weights.q_b_proj), mla.q_projection_size, mla.q_lora_rank)?,
        kv_a_proj: backend.prepare_weight(LinearWeight::F16(&weights.kv_a_proj), mla.kv_a_out(), cfg.hidden_size)?,
        kv_a_norm: backend.prepare_weight(LinearWeight::F16(&weights.kv_a_norm), 1, mla.kv_lora_rank)?,
        kv_b_proj: backend.prepare_mla_kv_b(LinearWeight::F16(&weights.kv_b_proj), mla.kv_projection_size, mla.kv_lora_rank)?,
        o_proj: backend.prepare_weight(LinearWeight::F16(&weights.o_proj), cfg.hidden_size, mla.q_projection_size)?,
        post_attn_norm: prepare_f16_f32(backend, &weights.post_attn_norm, 1, cfg.hidden_size)?,
        router_weight: prepare_f16_f32(backend, &weights.router_weight, cfg.expert_count, cfg.hidden_size)?,
        router_bias: prepare_f16_f32(backend, &weights.router_bias, 1, cfg.expert_count)?,
        shared_gate: backend.prepare_weight(LinearWeight::F16(&weights.shared_gate), cfg.expert_intermediate_size, cfg.hidden_size)?,
        shared_up: backend.prepare_weight(LinearWeight::F16(&weights.shared_up), cfg.expert_intermediate_size, cfg.hidden_size)?,
        shared_down: backend.prepare_weight(LinearWeight::F16(&weights.shared_down), cfg.hidden_size, cfg.expert_intermediate_size)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn glm52_prefill_indexer<B: DsaPrefillBackend + ExpertPrefillBackend>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    indexer: Option<&Glm52IndexerDecodeWeights<B::Weight>>,
    layer: usize,
    experts: Option<&B::PrefillExperts>,
    state: &mut B::DsaState,
    normalized_hidden: &B::Tensor,
    normalized_q_lora: &B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<(), BackendError> {
    let Some(indexer) = indexer else { return Ok(()) };
    let spec = glm52_dsa_spec(cfg, mla);
    backend.profile_device_operator("glm_index_key")?;
    prefill_select(
        backend,
        experts,
        state,
        normalized_hidden,
        normalized_q_lora,
        DsaWeightsRef { wq_b: &indexer.wq_b, wk: &indexer.wk, weights_proj: &indexer.weights_proj, k_norm_weight: &indexer.k_norm_weight, k_norm_bias: &indexer.k_norm_bias },
        layer,
        position,
        &rope.cos,
        &rope.sin,
        &spec,
    )
}

#[allow(clippy::too_many_arguments)]
fn glm52_prefill_indexer_segmented<B: DsaPrefillBackend + SegmentedTensorBackend>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    indexer: Option<&Glm52IndexerDecodeWeights<B::Weight>>,
    layer: usize,
    normalized_hidden: &B::Tensor,
    normalized_q_lora: &B::Tensor,
    rope: &RopeTable,
    reuse_selection: bool,
    segments: &mut [Glm52PrefillSegment<'_, B>],
) -> Result<(), BackendError> {
    let Some(indexer) = indexer else { return Ok(()) };
    let spec = glm52_dsa_spec(cfg, mla);
    if reuse_selection {
        if !segments.iter().all(|segment| backend.supports_dsa_prefill_selection_reuse(&*segment.dsa, layer, segment.position, segment.rows, &spec)) {
            return Err(BackendError::Compute { msg: format!("L{layer} MTP DSA selection 复用状态不连续") });
        }
        backend.profile_device_operator("glm_index_reuse")?;
        for segment in segments {
            backend.reuse_dsa_prefill_selection(&mut *segment.dsa, layer, segment.position, segment.rows, &spec)?;
        }
        return Ok(());
    }
    backend.profile_device_operator("glm_index_key")?;
    let key = backend.linear(normalized_hidden, &indexer.wk)?;
    let rope_segments = segments.iter().map(|segment| crate::backend::TokenSegment { position: segment.position, rows: segment.rows }).collect::<Vec<_>>();
    let fused = !segments.is_empty() && segments.iter().all(|segment| backend.supports_dsa_keys_layernorm_rope(&*segment.dsa, &key, &indexer.k_norm_weight, &indexer.k_norm_bias, &spec));
    if fused {
        let mut offset = 0usize;
        let segment_keys = segments
            .iter()
            .map(|segment| {
                let segment_key = backend.slice_token_rows(&key, offset, segment.rows);
                offset += segment.rows;
                segment_key
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (segment, segment_key) in segments.iter_mut().zip(segment_keys.iter()) {
            if !backend.append_dsa_keys_layernorm_rope(&mut *segment.dsa, layer, segment.position, segment_key, &indexer.k_norm_weight, &indexer.k_norm_bias, 1.0e-6, &rope.cos, &rope.sin, &spec)? {
                return Err(BackendError::Compute { msg: format!("GLM-5.2 segmented DSA L{layer} 融合预检后拒绝执行") });
            }
        }
    } else {
        let key = backend.layernorm_bias(&key, &indexer.k_norm_weight, &indexer.k_norm_bias, 1.0e-6)?;
        let key = backend.rope_segmented(&key, 1, spec.rope_dim, spec.rotary_layout, true, &rope_segments, &rope.cos, &rope.sin)?;
        let mut offset = 0usize;
        for segment in segments.iter_mut() {
            let segment_key = backend.slice_token_rows(&key, offset, segment.rows)?;
            backend.append_dsa_keys(&mut *segment.dsa, layer, segment.position, &segment_key, &spec)?;
            offset += segment.rows;
        }
    }
    backend.profile_device_operator("glm_index_query")?;
    let query = backend.linear(normalized_q_lora, &indexer.wq_b)?;
    let query = backend.rope_segmented(&query, spec.num_heads, spec.rope_dim, spec.rotary_layout, true, &rope_segments, &rope.cos, &rope.sin)?;
    let head_weights = backend.linear(normalized_hidden, &indexer.weights_proj)?;
    backend.profile_device_operator("glm_index_select")?;
    let mut offset = 0usize;
    for segment in segments {
        let segment_query = backend.slice_token_rows(&query, offset, segment.rows)?;
        let segment_weights = backend.slice_token_rows(&head_weights, offset, segment.rows)?;
        backend.dsa_select_prefill(&mut *segment.dsa, layer, &segment_query, &segment_weights, &spec)?;
        offset += segment.rows;
    }
    Ok(())
}

pub fn prepare_dense_prefill_layer<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &DenseLayerF32) -> Result<Glm52DensePrefillLayer<B::Weight>, BackendError> {
    Ok(Glm52DensePrefillLayer {
        indexer: weights.indexer.as_ref().map(|indexer| prepare_indexer(backend, cfg, indexer)).transpose()?,
        input_norm: backend.prepare_f32(&weights.input_norm, 1, cfg.hidden_size)?,
        q_a_proj: backend.prepare_weight(LinearWeight::fp8(&weights.q_a_proj), mla.q_lora_rank, cfg.hidden_size)?,
        q_a_norm: backend.prepare_f32(&weights.q_a_norm, 1, mla.q_lora_rank)?,
        q_b_proj: backend.prepare_weight(LinearWeight::fp8(&weights.q_b_proj), mla.q_projection_size, mla.q_lora_rank)?,
        kv_a_proj: backend.prepare_weight(LinearWeight::fp8(&weights.kv_a_proj), mla.kv_a_out(), cfg.hidden_size)?,
        kv_a_norm: backend.prepare_f32(&weights.kv_a_norm, 1, mla.kv_lora_rank)?,
        kv_b_proj: backend.prepare_mla_kv_b(LinearWeight::fp8(&weights.kv_b_proj), weights.kv_b_proj.rows, weights.kv_b_proj.cols)?,
        o_proj: backend.prepare_weight(LinearWeight::fp8(&weights.o_proj), cfg.hidden_size, mla.q_projection_size)?,
        post_attn_norm: backend.prepare_f32(&weights.post_attn_norm, 1, cfg.hidden_size)?,
        gate_proj: backend.prepare_weight(LinearWeight::fp8(&weights.gate_proj), cfg.dense_intermediate_size, cfg.hidden_size)?,
        up_proj: backend.prepare_weight(LinearWeight::fp8(&weights.up_proj), cfg.dense_intermediate_size, cfg.hidden_size)?,
        down_proj: backend.prepare_weight(LinearWeight::fp8(&weights.down_proj), cfg.hidden_size, cfg.dense_intermediate_size)?,
    })
}

pub(crate) fn prepare_ct_linear<B: crate::backend::Backend>(backend: &B, weight: &CtLinearWeight, rows: usize, cols: usize) -> Result<B::Weight, BackendError> {
    match weight {
        CtLinearWeight::Quantized(CtMatrix::W4(matrix)) => backend.prepare_weight(LinearWeight::w4a16(matrix), rows, cols),
        CtLinearWeight::Quantized(CtMatrix::W8(matrix)) => backend.prepare_weight(LinearWeight::w8a16(matrix), rows, cols),
        CtLinearWeight::Bf16(bytes) => backend.prepare_weight(LinearWeight::Bf16Bytes(bytes), rows, cols),
        CtLinearWeight::F32(values) => backend.prepare_weight(LinearWeight::F32(values), rows, cols),
    }
}

pub(crate) fn prepare_ct_kv_b<B: DecodeBackend>(backend: &B, weight: &CtLinearWeight, rows: usize, cols: usize) -> Result<B::Weight, BackendError> {
    match weight {
        CtLinearWeight::Quantized(CtMatrix::W4(matrix)) => backend.prepare_mla_kv_b(LinearWeight::w4a16(matrix), rows, cols),
        CtLinearWeight::Quantized(CtMatrix::W8(matrix)) => backend.prepare_mla_kv_b(LinearWeight::w8a16(matrix), rows, cols),
        CtLinearWeight::Bf16(bytes) => backend.prepare_mla_kv_b(LinearWeight::Bf16Bytes(bytes), rows, cols),
        CtLinearWeight::F32(values) => backend.prepare_mla_kv_b(LinearWeight::F32(values), rows, cols),
    }
}

fn prepare_ct_indexer<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, weights: &CtIndexerWeights) -> Result<Glm52IndexerDecodeWeights<B::Weight>, BackendError> {
    let query_size = cfg.index_heads * cfg.index_head_dim;
    Ok(Glm52IndexerDecodeWeights {
        wq_b: prepare_ct_linear(backend, &weights.wq_b, query_size, cfg.q_lora_rank)?,
        wk: prepare_ct_linear(backend, &weights.wk, cfg.index_head_dim, cfg.hidden_size)?,
        weights_proj: prepare_ct_linear(backend, &weights.weights_proj, cfg.index_heads, cfg.hidden_size)?,
        k_norm_weight: backend.prepare_f32(&weights.k_norm_weight, 1, cfg.index_head_dim)?,
        k_norm_bias: backend.prepare_f32(&weights.k_norm_bias, 1, cfg.index_head_dim)?,
    })
}

/// CT 版 dense prefill prepare：普通线性层保持压缩态；decode 常驻 KV-B 的物理布局由 backend 选择。
pub fn prepare_dense_prefill_layer_ct<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &CtDenseLayer, decode_resident: bool) -> Result<Glm52DensePrefillLayer<B::Weight>, BackendError> {
    Ok(Glm52DensePrefillLayer {
        indexer: weights.indexer.as_ref().map(|indexer| prepare_ct_indexer(backend, cfg, indexer)).transpose()?,
        input_norm: backend.prepare_f32(&weights.input_norm, 1, cfg.hidden_size)?,
        q_a_proj: prepare_ct_linear(backend, &weights.q_a_proj, mla.q_lora_rank, cfg.hidden_size)?,
        q_a_norm: backend.prepare_f32(&weights.q_a_norm, 1, mla.q_lora_rank)?,
        q_b_proj: prepare_ct_linear(backend, &weights.q_b_proj, mla.q_projection_size, mla.q_lora_rank)?,
        kv_a_proj: prepare_ct_linear(backend, &weights.kv_a_proj, mla.kv_a_out(), cfg.hidden_size)?,
        kv_a_norm: backend.prepare_f32(&weights.kv_a_norm, 1, mla.kv_lora_rank)?,
        kv_b_proj: if decode_resident { prepare_ct_kv_b(backend, &weights.kv_b_proj, mla.kv_projection_size, mla.kv_lora_rank)? } else { prepare_ct_linear(backend, &weights.kv_b_proj, mla.kv_projection_size, mla.kv_lora_rank)? },
        o_proj: prepare_ct_linear(backend, &weights.o_proj, cfg.hidden_size, mla.q_projection_size)?,
        post_attn_norm: backend.prepare_f32(&weights.post_attn_norm, 1, cfg.hidden_size)?,
        gate_proj: prepare_ct_linear(backend, &weights.gate_proj, cfg.dense_intermediate_size, cfg.hidden_size)?,
        up_proj: prepare_ct_linear(backend, &weights.up_proj, cfg.dense_intermediate_size, cfg.hidden_size)?,
        down_proj: prepare_ct_linear(backend, &weights.down_proj, cfg.hidden_size, cfg.dense_intermediate_size)?,
    })
}

/// CT 版 MoE prepare：router 保持原精度，shared expert 使用 W8A16。
pub fn prepare_moe_prefill_layer_ct<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &CtMoeLayer, decode_resident: bool) -> Result<Glm52MoePrefillLayer<B::Weight>, BackendError> {
    Ok(Glm52MoePrefillLayer {
        indexer: weights.indexer.as_ref().map(|indexer| prepare_ct_indexer(backend, cfg, indexer)).transpose()?,
        input_norm: backend.prepare_f32(&weights.input_norm, 1, cfg.hidden_size)?,
        q_a_proj: prepare_ct_linear(backend, &weights.q_a_proj, mla.q_lora_rank, cfg.hidden_size)?,
        q_a_norm: backend.prepare_f32(&weights.q_a_norm, 1, mla.q_lora_rank)?,
        q_b_proj: prepare_ct_linear(backend, &weights.q_b_proj, mla.q_projection_size, mla.q_lora_rank)?,
        kv_a_proj: prepare_ct_linear(backend, &weights.kv_a_proj, mla.kv_a_out(), cfg.hidden_size)?,
        kv_a_norm: backend.prepare_f32(&weights.kv_a_norm, 1, mla.kv_lora_rank)?,
        kv_b_proj: if decode_resident { prepare_ct_kv_b(backend, &weights.kv_b_proj, mla.kv_projection_size, mla.kv_lora_rank)? } else { prepare_ct_linear(backend, &weights.kv_b_proj, mla.kv_projection_size, mla.kv_lora_rank)? },
        o_proj: prepare_ct_linear(backend, &weights.o_proj, cfg.hidden_size, mla.q_projection_size)?,
        post_attn_norm: backend.prepare_f32(&weights.post_attn_norm, 1, cfg.hidden_size)?,
        router_weight: backend.prepare_f32(&weights.router_weight, cfg.expert_count, cfg.hidden_size)?,
        router_bias: backend.prepare_f32(&weights.router_bias, 1, cfg.expert_count)?,
        shared_gate: prepare_ct_linear(backend, &weights.shared_gate, cfg.expert_intermediate_size, cfg.hidden_size)?,
        shared_up: prepare_ct_linear(backend, &weights.shared_up, cfg.expert_intermediate_size, cfg.hidden_size)?,
        shared_down: prepare_ct_linear(backend, &weights.shared_down, cfg.hidden_size, cfg.expert_intermediate_size)?,
    })
}

pub fn prepare_glm52_mtp_ct<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &CtMtpLayer) -> Result<Glm52Mtp<B::Weight>, BackendError> {
    if cfg.mtp_layer_count != 1 {
        return Err(BackendError::Compute { msg: format!("GLM-5.2 当前要求恰好 1 个 MTP 层，实际 {}", cfg.mtp_layer_count) });
    }
    Ok(Glm52Mtp {
        embedding_norm: backend.prepare_f32(&weights.embedding_norm, 1, cfg.hidden_size)?,
        hidden_norm: backend.prepare_f32(&weights.hidden_norm, 1, cfg.hidden_size)?,
        input_projection: prepare_ct_linear(backend, &weights.input_projection, cfg.hidden_size, cfg.hidden_size * 2)?,
        layer: prepare_moe_prefill_layer_ct(backend, cfg, mla, &weights.layer, false)?,
        output_norm: backend.prepare_f32(&weights.output_norm, 1, cfg.hidden_size)?,
    })
}

fn prepare_gguf_linear<B: Backend>(backend: &B, matrix: &GgufMatrix, rows: usize, cols: usize) -> Result<B::Weight, BackendError> {
    backend.prepare_weight(LinearWeight::gguf(matrix), rows, cols)
}

fn prepare_indexer_gguf<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, weights: &GgufIndexerWeights) -> Result<Glm52IndexerDecodeWeights<B::Weight>, BackendError> {
    let query_size = cfg.index_heads * cfg.index_head_dim;
    // weights_proj 用 BF16 常驻:消费方的输入是 rmsnorm_quantized 的 BF16，
    // F32 resident 会落进要求 F32 输入的 sgemm 路径。
    let weights_proj_bf16 = weights.weights_proj.iter().map(|value| half::bf16::from_f32(*value).to_le_bytes().to_vec()).collect::<Vec<_>>().concat();
    Ok(Glm52IndexerDecodeWeights {
        wq_b: prepare_gguf_linear(backend, &weights.wq_b, query_size, cfg.q_lora_rank)?,
        wk: prepare_gguf_linear(backend, &weights.wk, cfg.index_head_dim, cfg.hidden_size)?,
        weights_proj: backend.prepare_weight(LinearWeight::Bf16Bytes(&weights_proj_bf16), cfg.index_heads, cfg.hidden_size)?,
        k_norm_weight: backend.prepare_f32(&weights.k_norm_weight, 1, cfg.index_head_dim)?,
        k_norm_bias: backend.prepare_f32(&weights.k_norm_bias, 1, cfg.index_head_dim)?,
    })
}

/// GGUF 版 dense prepare：量化矩阵保持 GGUF 打包，KV-B 的设备布局由 backend 选择。
pub fn prepare_dense_prefill_layer_gguf<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &GgufDenseLayer) -> Result<Glm52DensePrefillLayer<B::Weight>, BackendError> {
    Ok(Glm52DensePrefillLayer {
        indexer: weights.indexer.as_ref().map(|indexer| prepare_indexer_gguf(backend, cfg, indexer)).transpose()?,
        input_norm: backend.prepare_f32(&weights.input_norm, 1, cfg.hidden_size)?,
        q_a_proj: prepare_gguf_linear(backend, &weights.q_a_proj, mla.q_lora_rank, cfg.hidden_size)?,
        q_a_norm: backend.prepare_f32(&weights.q_a_norm, 1, mla.q_lora_rank)?,
        q_b_proj: prepare_gguf_linear(backend, &weights.q_b_proj, mla.q_projection_size, mla.q_lora_rank)?,
        kv_a_proj: prepare_gguf_linear(backend, &weights.kv_a_proj, mla.kv_a_out(), cfg.hidden_size)?,
        kv_a_norm: backend.prepare_f32(&weights.kv_a_norm, 1, mla.kv_lora_rank)?,
        kv_b_proj: backend.prepare_mla_kv_b(LinearWeight::w8a16(&weights.kv_b_w8), mla.kv_projection_size, mla.kv_lora_rank)?,
        o_proj: prepare_gguf_linear(backend, &weights.o_proj, cfg.hidden_size, mla.q_projection_size)?,
        post_attn_norm: backend.prepare_f32(&weights.post_attn_norm, 1, cfg.hidden_size)?,
        gate_proj: prepare_gguf_linear(backend, &weights.gate_proj, cfg.dense_intermediate_size, cfg.hidden_size)?,
        up_proj: prepare_gguf_linear(backend, &weights.up_proj, cfg.dense_intermediate_size, cfg.hidden_size)?,
        down_proj: prepare_gguf_linear(backend, &weights.down_proj, cfg.hidden_size, cfg.dense_intermediate_size)?,
    })
}

/// GGUF 版 MoE prepare：router 保持 F32，shared expert 保持 GGUF 打包。
pub fn prepare_moe_prefill_layer_gguf<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &GgufMoeLayer) -> Result<Glm52MoePrefillLayer<B::Weight>, BackendError> {
    Ok(Glm52MoePrefillLayer {
        indexer: weights.indexer.as_ref().map(|indexer| prepare_indexer_gguf(backend, cfg, indexer)).transpose()?,
        input_norm: backend.prepare_f32(&weights.input_norm, 1, cfg.hidden_size)?,
        q_a_proj: prepare_gguf_linear(backend, &weights.q_a_proj, mla.q_lora_rank, cfg.hidden_size)?,
        q_a_norm: backend.prepare_f32(&weights.q_a_norm, 1, mla.q_lora_rank)?,
        q_b_proj: prepare_gguf_linear(backend, &weights.q_b_proj, mla.q_projection_size, mla.q_lora_rank)?,
        kv_a_proj: prepare_gguf_linear(backend, &weights.kv_a_proj, mla.kv_a_out(), cfg.hidden_size)?,
        kv_a_norm: backend.prepare_f32(&weights.kv_a_norm, 1, mla.kv_lora_rank)?,
        kv_b_proj: backend.prepare_mla_kv_b(LinearWeight::w8a16(&weights.kv_b_w8), mla.kv_projection_size, mla.kv_lora_rank)?,
        o_proj: prepare_gguf_linear(backend, &weights.o_proj, cfg.hidden_size, mla.q_projection_size)?,
        post_attn_norm: backend.prepare_f32(&weights.post_attn_norm, 1, cfg.hidden_size)?,
        router_weight: backend.prepare_f32(&weights.router_weight, cfg.expert_count, cfg.hidden_size)?,
        router_bias: backend.prepare_f32(&weights.router_bias, 1, cfg.expert_count)?,
        shared_gate: backend.prepare_expert_weight(LinearWeight::gguf(&weights.shared_gate), cfg.expert_intermediate_size, cfg.hidden_size)?,
        shared_up: backend.prepare_expert_weight(LinearWeight::gguf(&weights.shared_up), cfg.expert_intermediate_size, cfg.hidden_size)?,
        shared_down: backend.prepare_expert_weight(LinearWeight::gguf(&weights.shared_down), cfg.hidden_size, cfg.expert_intermediate_size)?,
    })
}

pub fn prepare_glm52_mtp_gguf<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &GgufMtpLayer) -> Result<Glm52Mtp<B::Weight>, BackendError> {
    if cfg.mtp_layer_count != 1 {
        return Err(BackendError::Compute { msg: format!("GLM-5.2 当前要求恰好 1 个 MTP 层，实际 {}", cfg.mtp_layer_count) });
    }
    Ok(Glm52Mtp {
        embedding_norm: backend.prepare_f32(&weights.embedding_norm, 1, cfg.hidden_size)?,
        hidden_norm: backend.prepare_f32(&weights.hidden_norm, 1, cfg.hidden_size)?,
        input_projection: prepare_gguf_linear(backend, &weights.input_projection, cfg.hidden_size, cfg.hidden_size * 2)?,
        layer: prepare_moe_prefill_layer_gguf(backend, cfg, mla, &weights.layer)?,
        output_norm: backend.prepare_f32(&weights.output_norm, 1, cfg.hidden_size)?,
    })
}

/// 按权重来源(nvfp4/ct/gguf/official)加载并准备单层 dense 权重;
/// `ct_decode_resident` 仅对 compressed-tensors 生效(decode 常驻布局)，其余来源忽略。
pub fn load_prepare_dense_prefill_layer<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &Glm52Weights, layer: usize, ct_decode_resident: bool) -> Result<Glm52DensePrefillLayer<B::Weight>, BackendError> {
    if weights.source_is_nvfp4() {
        let layer_weights = weights.load_dense_layer_nvfp4(layer).map_err(|msg| BackendError::Compute { msg: format!("L{layer} dense NVFP4 core: {msg}") })?;
        prepare_dense_prefill_layer_nvfp4(backend, cfg, mla, &layer_weights)
    } else if weights.source_is_ct() {
        let layer_weights = weights.load_dense_layer_ct(layer).map_err(|msg| BackendError::Compute { msg: format!("L{layer} dense CT: {msg}") })?;
        prepare_dense_prefill_layer_ct(backend, cfg, mla, &layer_weights, ct_decode_resident)
    } else if weights.source_is_gguf() {
        let layer_weights = weights.load_dense_layer_gguf(layer).map_err(|msg| BackendError::Compute { msg: format!("L{layer} dense GGUF: {msg}") })?;
        prepare_dense_prefill_layer_gguf(backend, cfg, mla, &layer_weights)
    } else {
        let layer_weights = weights.load_dense_layer(layer).map_err(|msg| BackendError::Compute { msg: format!("L{layer} dense 权重: {msg}") })?;
        prepare_dense_prefill_layer(backend, cfg, mla, &layer_weights)
    }
}

/// 按权重来源(nvfp4/ct/gguf/official)加载并准备单层 MoE 权重;参数含义同 [`load_prepare_dense_prefill_layer`]。
pub fn load_prepare_moe_prefill_layer<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &Glm52Weights, layer: usize, ct_decode_resident: bool) -> Result<Glm52MoePrefillLayer<B::Weight>, BackendError> {
    if weights.source_is_nvfp4() {
        let layer_weights = weights.load_moe_layer_nvfp4(layer).map_err(|msg| BackendError::Compute { msg: format!("L{layer} MoE NVFP4 core: {msg}") })?;
        prepare_moe_prefill_layer_nvfp4(backend, cfg, mla, &layer_weights)
    } else if weights.source_is_ct() {
        let layer_weights = weights.load_moe_layer_ct(layer).map_err(|msg| BackendError::Compute { msg: format!("L{layer} MoE CT: {msg}") })?;
        prepare_moe_prefill_layer_ct(backend, cfg, mla, &layer_weights, ct_decode_resident)
    } else if weights.source_is_gguf() {
        let layer_weights = weights.load_moe_layer_gguf(layer).map_err(|msg| BackendError::Compute { msg: format!("L{layer} MoE GGUF: {msg}") })?;
        prepare_moe_prefill_layer_gguf(backend, cfg, mla, &layer_weights)
    } else {
        let layer_weights = weights.load_moe_layer(layer).map_err(|msg| BackendError::Compute { msg: format!("L{layer} MoE 权重: {msg}") })?;
        prepare_moe_prefill_layer(backend, cfg, mla, &layer_weights)
    }
}

pub fn prepare_glm52_decode_layers<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &Glm52Weights) -> Result<Vec<Glm52DecodeLayer<B::Weight>>, BackendError> {
    let mut layers = Vec::with_capacity(cfg.layer_count);
    for layer in 0..cfg.layer_count {
        if layer < cfg.dense_layer_count {
            let resident = load_prepare_dense_prefill_layer(backend, cfg, mla, weights, layer, false).map_err(|error| BackendError::Compute { msg: format!("准备 GLM-5.2 decode L{layer}: {error:?}") })?;
            layers.push(Glm52DecodeLayer::Dense(resident));
        } else {
            let resident = load_prepare_moe_prefill_layer(backend, cfg, mla, weights, layer, false).map_err(|error| BackendError::Compute { msg: format!("准备 GLM-5.2 decode L{layer}: {error:?}") })?;
            layers.push(Glm52DecodeLayer::Moe(resident));
        }
    }
    Ok(layers)
}

pub fn prepare_moe_prefill_layer<B: DecodeBackend>(backend: &B, cfg: &Glm52Config, mla: &MlaSpec, weights: &MoeLayerF32) -> Result<Glm52MoePrefillLayer<B::Weight>, BackendError> {
    Ok(Glm52MoePrefillLayer {
        indexer: weights.indexer.as_ref().map(|indexer| prepare_indexer(backend, cfg, indexer)).transpose()?,
        input_norm: backend.prepare_f32(&weights.input_norm, 1, cfg.hidden_size)?,
        q_a_proj: backend.prepare_weight(LinearWeight::fp8(&weights.q_a_proj), mla.q_lora_rank, cfg.hidden_size)?,
        q_a_norm: backend.prepare_f32(&weights.q_a_norm, 1, mla.q_lora_rank)?,
        q_b_proj: backend.prepare_weight(LinearWeight::fp8(&weights.q_b_proj), mla.q_projection_size, mla.q_lora_rank)?,
        kv_a_proj: backend.prepare_weight(LinearWeight::fp8(&weights.kv_a_proj), mla.kv_a_out(), cfg.hidden_size)?,
        kv_a_norm: backend.prepare_f32(&weights.kv_a_norm, 1, mla.kv_lora_rank)?,
        kv_b_proj: backend.prepare_mla_kv_b(LinearWeight::fp8(&weights.kv_b_proj), weights.kv_b_proj.rows, weights.kv_b_proj.cols)?,
        o_proj: backend.prepare_weight(LinearWeight::fp8(&weights.o_proj), cfg.hidden_size, mla.q_projection_size)?,
        post_attn_norm: backend.prepare_f32(&weights.post_attn_norm, 1, cfg.hidden_size)?,
        router_weight: backend.prepare_f32(&weights.router_weight, cfg.expert_count, cfg.hidden_size)?,
        router_bias: backend.prepare_f32(&weights.router_bias, 1, cfg.expert_count)?,
        shared_gate: backend.prepare_weight(LinearWeight::fp8(&weights.shared_gate), cfg.expert_intermediate_size, cfg.hidden_size)?,
        shared_up: backend.prepare_weight(LinearWeight::fp8(&weights.shared_up), cfg.expert_intermediate_size, cfg.hidden_size)?,
        shared_down: backend.prepare_weight(LinearWeight::fp8(&weights.shared_down), cfg.hidden_size, cfg.expert_intermediate_size)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_dense_prefill_layer<B: DsaPrefillBackend + ExpertPrefillBackend>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52DensePrefillLayer<B::Weight>,
    layer: usize,
    experts: Option<&B::PrefillExperts>,
    dsa_state: Option<&mut B::DsaState>,
    hidden: &B::Tensor,
    rope: &RopeTable,
    cache: Option<&mut B::Cache>,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    let diagnose = backend.token_rows(hidden) == 1;
    let total_start_us = diagnose.then(crate::runtime::prefill_scheduler::stage_trace_timestamp_us);
    let total_started = diagnose.then(Instant::now);
    let mut phase_started = diagnose.then(Instant::now);
    let mut mark_phase = || {
        phase_started.as_mut().map_or(0, |started| {
            let elapsed = started.elapsed().as_micros();
            *started = Instant::now();
            elapsed
        })
    };
    backend.begin_batch();
    backend.profile_device_operator("glm_attention")?;
    let begin_micros = mark_phase();
    let normed = backend.rmsnorm_quantized(hidden, &weights.input_norm, cfg.rms_eps)?;

    let q_a = backend.linear(&normed, &weights.q_a_proj)?;
    let input_micros = mark_phase();
    backend.profile_device_operator("glm_attn_query")?;
    // cooperative MLA 同时覆盖 prefill 与 decode：单行仍有完整的 q_b、
    // indexed MLA 和 o_proj，若只让 owner 执行，peer 会在整段 attention 空转。
    let parallel_mla = experts.is_some_and(|experts| backend.supports_parallel_mla_prefill(layer, experts));
    let cooperative_mla = experts.is_some_and(|experts| backend.supports_cooperative_mla_prefill(layer, experts));
    let paired_mla = parallel_mla || cooperative_mla;
    let mut dsa_state = dsa_state;
    let prepared = (|| -> Result<_, BackendError> {
        let (normalized_q_lora, query) = if weights.indexer.is_some() {
            let q_a = backend.rmsnorm_quantized(&q_a, &weights.q_a_norm, cfg.rms_eps)?;
            if parallel_mla {
                backend.begin_parallel_mla_query(layer, experts.expect("parallel dense MLA 必须提供 experts"), &q_a, position, &rope.cos, &rope.sin, mla)?;
            }
            if let Some(state) = dsa_state.as_deref_mut() {
                glm52_prefill_indexer(backend, cfg, mla, weights.indexer.as_ref(), layer, experts, state, &normed, &q_a, rope, position)?;
            }
            if paired_mla { (Some(q_a), None) } else { (None, Some(backend.linear(&q_a, &weights.q_b_proj)?)) }
        } else if paired_mla {
            let q_a = backend.rmsnorm_quantized(&q_a, &weights.q_a_norm, cfg.rms_eps)?;
            if parallel_mla {
                backend.begin_parallel_mla_query(layer, experts.expect("parallel dense MLA 必须提供 experts"), &q_a, position, &rope.cos, &rope.sin, mla)?;
            }
            (Some(q_a), None)
        } else {
            // IndexShare 层复用最近 full indexer 的 selection，不能退回 dense MLA。
            (None, Some(backend.rmsnorm_linear(&q_a, &weights.q_a_norm, cfg.rms_eps, &weights.q_b_proj)?))
        };
        let query_micros = mark_phase();

        backend.profile_device_operator("glm_attn_kv")?;
        let kv_a = backend.linear(&normed, &weights.kv_a_proj)?;
        let (latent, k_rope) = backend.split_columns(&kv_a, mla.kv_lora_rank)?;
        let latent = backend.rmsnorm(&latent, &weights.kv_a_norm, cfg.rms_eps)?;
        let kv_micros = mark_phase();
        Ok((normalized_q_lora, query, latent, k_rope, query_micros, kv_micros))
    })();
    let finish = dsa_state.as_deref_mut().map(|state| backend.dsa_select_topk_finish(state)).transpose();
    let (normalized_q_lora, query, latent, k_rope, query_micros, kv_micros) = prepared?;
    finish?;

    let dsa = glm52_dsa_spec(cfg, mla);
    backend.profile_device_operator("glm_attn_mla")?;
    let out = if parallel_mla {
        backend.parallel_mla_prefill_add(
            layer,
            experts.expect("parallel dense MLA 必须提供 experts"),
            normalized_q_lora.as_ref().expect("parallel dense MLA 必须保留 q_lora"),
            &latent,
            &k_rope,
            hidden,
            cache,
            dsa_state.as_deref(),
            position,
            &rope.cos,
            &rope.sin,
            mla,
            &dsa,
        )?
    } else if cooperative_mla {
        backend.cooperative_mla_prefill_add(
            layer,
            experts.expect("cooperative dense MLA 必须提供 experts"),
            normalized_q_lora.as_ref().expect("cooperative dense MLA 必须保留 q_lora"),
            &latent,
            &k_rope,
            hidden,
            cache,
            dsa_state.as_deref(),
            position,
            &rope.cos,
            &rope.sin,
            mla,
            &dsa,
        )?
    } else {
        let query = backend.rope(query.as_ref().expect("普通 dense MLA 必须生成完整 query"), mla.num_heads, mla.qk_rope_head_dim, mla.rotary_layout, position, &rope.cos, &rope.sin)?;
        let attention = mla_causal_prefill_rope(backend, &query, &latent, &k_rope, &weights.kv_b_proj, cache, layer, position, &rope.cos, &rope.sin, mla, &dsa, dsa_state)?;
        backend.linear_add(&attention, &weights.o_proj, hidden)?
    };
    let mla_micros = mark_phase();
    backend.profile_device_operator("glm_attn_out")?;
    let out_micros = mark_phase();

    backend.profile_device_operator("glm_ffn")?;
    let dense_spec = DenseMlpSpec { intermediate_size: cfg.dense_intermediate_size, activation: Activation::Silu };
    let dense_weights = DenseMlpWeightsRef { gate: &weights.gate_proj, up: &weights.up_proj, down: &weights.down_proj };
    let parallel_result = experts.map(|experts| backend.parallel_dense_mlp_rmsnorm_add(&dense_spec, dense_weights, layer, experts, &out, &weights.post_attn_norm, cfg.rms_eps)).transpose()?.flatten();
    let (result, parallel_ffn_micros, gate_micros, down_micros, residual_micros) = if let Some(result) = parallel_result {
        (result, mark_phase(), 0, 0, 0)
    } else {
        let mlp_normed = backend.rmsnorm_quantized(&out, &weights.post_attn_norm, cfg.rms_eps)?;
        backend.profile_device_operator("glm_dense_gate_up")?;
        let activated = backend.gated_linear(&mlp_normed, &weights.gate_proj, &weights.up_proj, &Activation::Silu)?;
        let gate_micros = mark_phase();
        backend.profile_device_operator("glm_dense_down")?;
        let down = backend.linear(&activated, &weights.down_proj)?;
        let down_micros = mark_phase();
        backend.profile_device_operator("glm_dense_residual")?;
        let result = backend.add(&out, &down)?;
        let residual_micros = mark_phase();
        (result, 0, gate_micros, down_micros, residual_micros)
    };
    if let Some(total_started) = total_started {
        let total_micros = total_started.elapsed().as_micros();
        if total_micros >= 100_000 {
            eprintln!(
                "[glm52-layer-host-slow] ts_us={} kind=dense layer={layer} position={position} total_ms={:.3} begin_ms={:.3} input_ms={:.3} query_ms={:.3} kv_ms={:.3} mla_ms={:.3} out_ms={:.3} parallel_ffn_ms={:.3} gate_ms={:.3} down_ms={:.3} residual_ms={:.3} complete_us={}",
                total_start_us.unwrap_or_default(),
                total_micros as f64 / 1000.0,
                begin_micros as f64 / 1000.0,
                input_micros as f64 / 1000.0,
                query_micros as f64 / 1000.0,
                kv_micros as f64 / 1000.0,
                mla_micros as f64 / 1000.0,
                out_micros as f64 / 1000.0,
                parallel_ffn_micros as f64 / 1000.0,
                gate_micros as f64 / 1000.0,
                down_micros as f64 / 1000.0,
                residual_micros as f64 / 1000.0,
                crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
            );
        }
    }
    Ok(result)
}

pub struct Glm52PrefillSegment<'a, B: DsaPrefillBackend> {
    pub position: usize,
    pub rows: usize,
    pub cache: &'a mut B::Cache,
    pub dsa: &'a mut B::DsaState,
}

struct Glm52PrefillAttentionWeights<'a, W> {
    indexer: Option<&'a Glm52IndexerDecodeWeights<W>>,
    input_norm: &'a W,
    q_a_proj: &'a W,
    q_a_norm: &'a W,
    q_b_proj: &'a W,
    kv_a_proj: &'a W,
    kv_a_norm: &'a W,
    kv_b_proj: &'a W,
    o_proj: &'a W,
}

#[allow(clippy::too_many_arguments)]
fn segmented_mla<B: DsaPrefillBackend + SegmentedTensorBackend>(
    backend: &B,
    query: &B::Tensor,
    latent: &B::Tensor,
    k_rope: &B::Tensor,
    kv_b: &B::Weight,
    layer: usize,
    mla: &MlaSpec,
    cfg: &Glm52Config,
    segments: &mut [Glm52PrefillSegment<'_, B>],
) -> Result<B::Tensor, BackendError> {
    let dsa = glm52_dsa_spec(cfg, mla);
    let mut backend_segments = segments.iter_mut().map(|segment| crate::backend::DsaPrefillSegment { rows: segment.rows, cache: &mut *segment.cache, state: &*segment.dsa }).collect::<Vec<_>>();
    backend.mla_prefill_attention_selected_segmented(query, latent, k_rope, kv_b, layer, mla, &dsa, &mut backend_segments)
}

#[allow(clippy::too_many_arguments)]
fn glm52_segmented_attention<B: DsaPrefillBackend + ExpertPrefillBackend + SegmentedTensorBackend>(
    backend: &B,
    experts: Option<&B::PrefillExperts>,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: Glm52PrefillAttentionWeights<'_, B::Weight>,
    layer: usize,
    hidden: &B::Tensor,
    rope: &RopeTable,
    reuse_dsa_selection: bool,
    segments: &mut [Glm52PrefillSegment<'_, B>],
) -> Result<B::Tensor, BackendError> {
    backend.begin_batch();
    let total_rows = segments.iter().try_fold(0usize, |rows, segment| rows.checked_add(segment.rows).ok_or_else(|| BackendError::Compute { msg: "GLM segmented rows 溢出".to_owned() }))?;
    if segments.is_empty() || total_rows != backend.token_rows(hidden) {
        return Err(BackendError::Compute { msg: format!("GLM segmented attention rows={total_rows}，hidden={}", backend.token_rows(hidden)) });
    }

    let parallel_mla = experts.is_some_and(|experts| backend.supports_parallel_mla_prefill(layer, experts));
    let cooperative_mla = experts.is_some_and(|experts| backend.supports_cooperative_mla_prefill(layer, experts));
    let paired_mla = parallel_mla || cooperative_mla;
    let normed = backend.rmsnorm_quantized(hidden, weights.input_norm, cfg.rms_eps)?;
    let q_a = backend.linear(&normed, weights.q_a_proj)?;
    backend.profile_device_operator("glm_attn_query")?;
    let prepared = (|| -> Result<_, BackendError> {
        let (normalized_q_lora, query) = if weights.indexer.is_some() {
            let q_a = backend.rmsnorm_quantized(&q_a, weights.q_a_norm, cfg.rms_eps)?;
            if paired_mla {
                if parallel_mla && total_rows == 1 {
                    let position = segments.first().expect("单行 parallel MLA 必须有 segment").position;
                    backend.begin_parallel_mla_query(layer, experts.expect("parallel MLA 必须提供 experts"), &q_a, position, &rope.cos, &rope.sin, mla)?;
                }
                if reuse_dsa_selection {
                    let spec = glm52_dsa_spec(cfg, mla);
                    if !segments.iter().all(|segment| backend.supports_dsa_prefill_selection_reuse(&*segment.dsa, layer, segment.position, segment.rows, &spec)) {
                        return Err(BackendError::Compute { msg: format!("L{layer} paired MTP DSA selection 复用状态不连续") });
                    }
                    backend.profile_device_operator("glm_index_reuse")?;
                    for segment in segments.iter_mut() {
                        backend.reuse_dsa_prefill_selection(&mut *segment.dsa, layer, segment.position, segment.rows, &spec)?;
                    }
                } else {
                    let mut offset = 0usize;
                    for segment in segments.iter_mut() {
                        let segment_hidden = backend.slice_token_rows(&normed, offset, segment.rows)?;
                        let segment_q = backend.slice_token_rows(&q_a, offset, segment.rows)?;
                        glm52_prefill_indexer(backend, cfg, mla, weights.indexer, layer, experts, &mut *segment.dsa, &segment_hidden, &segment_q, rope, segment.position)?;
                        offset += segment.rows;
                    }
                }
                (Some(q_a), None)
            } else {
                glm52_prefill_indexer_segmented(backend, cfg, mla, weights.indexer, layer, &normed, &q_a, rope, reuse_dsa_selection, segments)?;
                (None, Some(backend.linear(&q_a, weights.q_b_proj)?))
            }
        } else if paired_mla {
            let q_a = backend.rmsnorm_quantized(&q_a, weights.q_a_norm, cfg.rms_eps)?;
            if parallel_mla && total_rows == 1 {
                let position = segments.first().expect("单行 parallel MLA 必须有 segment").position;
                backend.begin_parallel_mla_query(layer, experts.expect("parallel MLA 必须提供 experts"), &q_a, position, &rope.cos, &rope.sin, mla)?;
            }
            (Some(q_a), None)
        } else {
            (None, Some(backend.rmsnorm_linear(&q_a, weights.q_a_norm, cfg.rms_eps, weights.q_b_proj)?))
        };
        backend.profile_device_operator("glm_attn_kv")?;
        let kv_a = backend.linear(&normed, weights.kv_a_proj)?;
        let (latent, k_rope) = backend.split_columns(&kv_a, mla.kv_lora_rank)?;
        let latent = backend.rmsnorm(&latent, weights.kv_a_norm, cfg.rms_eps)?;
        Ok((normalized_q_lora, query, latent, k_rope))
    })();
    let mut finish = Ok(());
    for segment in segments.iter_mut() {
        if let Err(error) = backend.dsa_select_topk_finish(segment.dsa) {
            finish = Err(error);
        }
    }
    let (normalized_q_lora, query, latent, k_rope) = prepared?;
    finish?;
    backend.profile_device_operator("glm_attn_mla")?;
    if paired_mla {
        let experts = experts.expect("paired segmented MLA 必须提供 experts");
        let normalized_q_lora = normalized_q_lora.as_ref().expect("paired segmented MLA 必须保留 q_lora");
        let dsa = glm52_dsa_spec(cfg, mla);
        let mut outputs = Vec::with_capacity(segments.len());
        let mut offset = 0usize;
        for segment in segments.iter_mut() {
            let segment_q = backend.slice_token_rows(normalized_q_lora, offset, segment.rows)?;
            let segment_latent = backend.slice_token_rows(&latent, offset, segment.rows)?;
            let segment_rope = backend.slice_token_rows(&k_rope, offset, segment.rows)?;
            let segment_hidden = backend.slice_token_rows(hidden, offset, segment.rows)?;
            outputs.push(if parallel_mla {
                backend.parallel_mla_prefill_add(layer, experts, &segment_q, &segment_latent, &segment_rope, &segment_hidden, Some(&mut *segment.cache), Some(&*segment.dsa), segment.position, &rope.cos, &rope.sin, mla, &dsa)?
            } else {
                backend.cooperative_mla_prefill_add(layer, experts, &segment_q, &segment_latent, &segment_rope, &segment_hidden, Some(&mut *segment.cache), Some(&*segment.dsa), segment.position, &rope.cos, &rope.sin, mla, &dsa)?
            });
            offset += segment.rows;
        }
        let outputs = outputs.iter().collect::<Vec<_>>();
        if parallel_mla {
            match backend.concat_parallel_stage_tensors(experts, &outputs)? {
                Some(output) => Ok(output),
                None => backend.concat_token_rows(&outputs),
            }
        } else {
            backend.concat_token_rows(&outputs)
        }
    } else {
        let query = query.as_ref().expect("普通 segmented MLA 必须生成 query");
        let rope_segments = segments.iter().map(|segment| crate::backend::TokenSegment { position: segment.position, rows: segment.rows }).collect::<Vec<_>>();
        let (query, k_rope) = backend.rope_segmented_pair(query, mla.num_heads, &k_rope, 1, mla.qk_rope_head_dim, mla.rotary_layout, false, &rope_segments, &rope.cos, &rope.sin)?;
        let attention = segmented_mla(backend, &query, &latent, &k_rope, weights.kv_b_proj, layer, mla, cfg, segments)?;
        backend.profile_device_operator("glm_attn_out")?;
        backend.linear_add(&attention, weights.o_proj, hidden)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_dense_prefill_layer_segmented<B: DsaPrefillBackend + ExpertPrefillBackend + SegmentedTensorBackend>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52DensePrefillLayer<B::Weight>,
    layer: usize,
    experts: &B::PrefillExperts,
    hidden: &B::Tensor,
    rope: &RopeTable,
    segments: &mut [Glm52PrefillSegment<'_, B>],
) -> Result<B::Tensor, BackendError> {
    backend.profile_device_operator("glm_attention")?;
    let out = glm52_segmented_attention(
        backend,
        Some(experts),
        cfg,
        mla,
        Glm52PrefillAttentionWeights {
            indexer: weights.indexer.as_ref(),
            input_norm: &weights.input_norm,
            q_a_proj: &weights.q_a_proj,
            q_a_norm: &weights.q_a_norm,
            q_b_proj: &weights.q_b_proj,
            kv_a_proj: &weights.kv_a_proj,
            kv_a_norm: &weights.kv_a_norm,
            kv_b_proj: &weights.kv_b_proj,
            o_proj: &weights.o_proj,
        },
        layer,
        hidden,
        rope,
        false,
        segments,
    )?;
    backend.profile_device_operator("glm_ffn")?;
    let spec = DenseMlpSpec { intermediate_size: cfg.dense_intermediate_size, activation: Activation::Silu };
    let dense_weights = DenseMlpWeightsRef { gate: &weights.gate_proj, up: &weights.up_proj, down: &weights.down_proj };
    if let Some(result) = backend.parallel_dense_mlp_rmsnorm_add(&spec, dense_weights, layer, experts, &out, &weights.post_attn_norm, cfg.rms_eps)? {
        Ok(result)
    } else {
        let mlp_normed = backend.rmsnorm_quantized(&out, &weights.post_attn_norm, cfg.rms_eps)?;
        backend.profile_device_operator("glm_dense_gate_up")?;
        let activated = backend.gated_linear(&mlp_normed, &weights.gate_proj, &weights.up_proj, &Activation::Silu)?;
        backend.profile_device_operator("glm_dense_down")?;
        let down = backend.linear(&activated, &weights.down_proj)?;
        backend.profile_device_operator("glm_dense_residual")?;
        backend.add(&out, &down)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_moe_prefill_layer_segmented<B: ExpertPrefillBackend + DsaPrefillBackend + SegmentedTensorBackend>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52MoePrefillLayer<B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    hidden: &B::Tensor,
    rope: &RopeTable,
    reuse_dsa_selection: bool,
    segments: &mut [Glm52PrefillSegment<'_, B>],
) -> Result<B::Tensor, BackendError> {
    backend.profile_device_operator("glm_attention")?;
    let out = glm52_segmented_attention(
        backend,
        Some(experts),
        cfg,
        mla,
        Glm52PrefillAttentionWeights {
            indexer: weights.indexer.as_ref(),
            input_norm: &weights.input_norm,
            q_a_proj: &weights.q_a_proj,
            q_a_norm: &weights.q_a_norm,
            q_b_proj: &weights.q_b_proj,
            kv_a_proj: &weights.kv_a_proj,
            kv_a_norm: &weights.kv_a_norm,
            kv_b_proj: &weights.kv_b_proj,
            o_proj: &weights.o_proj,
        },
        layer,
        hidden,
        rope,
        reuse_dsa_selection,
        segments,
    )?;
    backend.profile_device_operator("glm_ffn")?;
    let (spec, shared) = glm52_moe_spec(cfg, &weights.shared_gate, &weights.shared_up, &weights.shared_down);
    let ffn_weights = MoeFfnRef { router_weight: &weights.router_weight, router_bias: &weights.router_bias, shared_experts: &shared, selected_experts: None };
    let routed_weights = RoutedMoeWeightsRef { router: &weights.router_weight, bias: &weights.router_bias, selected_experts: None };
    if let Some(result) = backend.parallel_moe_rmsnorm_add(&spec, routed_weights, &shared, layer, experts, &out, &weights.post_attn_norm, cfg.rms_eps)? {
        Ok(result)
    } else if let Some((route_input, expert_input)) = backend.rmsnorm_quantized_pair(&out, &weights.post_attn_norm, cfg.rms_eps)? {
        crate::moe::prefill::prefill_experts_inputs_add_residual_untraced(backend, &spec, &ffn_weights, layer, experts, RoutedMoeInputs { route: &route_input, expert: &expert_input }, &out, None)
    } else {
        let mlp_normed = backend.rmsnorm_f32(&out, &weights.post_attn_norm, cfg.rms_eps)?;
        crate::moe::prefill::prefill_experts_add_residual_untraced(backend, &spec, &ffn_weights, layer, experts, &mlp_normed, &out, None)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_moe_prefill_layer<B: ExpertPrefillBackend + DsaPrefillBackend>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52MoePrefillLayer<B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    route_trace: Option<&mut ExpertRouteTrace>,
    dsa_state: Option<&mut B::DsaState>,
    hidden: &B::Tensor,
    rope: &RopeTable,
    cache: Option<&mut B::Cache>,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    let diagnose = backend.token_rows(hidden) == 1;
    let total_start_us = diagnose.then(crate::runtime::prefill_scheduler::stage_trace_timestamp_us);
    let total_started = diagnose.then(Instant::now);
    let mut phase_started = diagnose.then(Instant::now);
    let mut mark_phase = || {
        phase_started.as_mut().map_or(0, |started| {
            let elapsed = started.elapsed().as_micros();
            *started = Instant::now();
            elapsed
        })
    };
    backend.begin_batch();
    backend.profile_device_operator("glm_attention")?;
    let begin_micros = mark_phase();
    let single_row = backend.token_rows(hidden) == 1;
    let (normed, q_a, kv_a) = if weights.indexer.is_none() && single_row {
        // 单行 IndexShare 层不消费 normed 本身，直接让两路量化投影共享 BF16
        // RMSNorm 结果，避免先写 F32、再分别转换给 q/kv 投影。
        let (q_a, kv_a) = backend.rmsnorm_dual_linear(hidden, &weights.input_norm, cfg.rms_eps, &weights.q_a_proj, &weights.kv_a_proj)?;
        (None, q_a, kv_a)
    } else if weights.indexer.is_some()
        && single_row
        && let Some((precise, quantized)) = backend.rmsnorm_quantized_pair(hidden, &weights.input_norm, cfg.rms_eps)?
    {
        // Full-indexer 同时需要精确 F32 输入做 DSA，以及 BF16 输入做 W4 q/kv
        // 双投影；由同一次 RMSNorm 写出两种表示，避免 dual_linear 再量化一次。
        let (q_a, kv_a) = backend.dual_linear(&quantized, &weights.q_a_proj, &weights.kv_a_proj)?;
        (Some(precise), q_a, kv_a)
    } else {
        let normed = backend.rmsnorm_quantized(hidden, &weights.input_norm, cfg.rms_eps)?;
        let (q_a, kv_a) = backend.dual_linear(&normed, &weights.q_a_proj, &weights.kv_a_proj)?;
        (Some(normed), q_a, kv_a)
    };
    let input_micros = mark_phase();
    backend.profile_device_operator("glm_attn_query")?;
    // 单行也沿 query head 拆分；peer KV 由 cooperative append 同步推进。
    let parallel_mla = backend.supports_parallel_mla_prefill(layer, experts);
    let cooperative_mla = backend.supports_cooperative_mla_prefill(layer, experts);
    let paired_mla = parallel_mla || cooperative_mla;
    let mut dsa_state = dsa_state;
    let prepared = (|| -> Result<_, BackendError> {
        let (normalized_q_lora, query) = if weights.indexer.is_some() {
            let normed = normed.as_ref().expect("indexer 层必须保留归一化输入");
            let q_a = backend.rmsnorm_quantized(&q_a, &weights.q_a_norm, cfg.rms_eps)?;
            if parallel_mla {
                backend.begin_parallel_mla_query(layer, experts, &q_a, position, &rope.cos, &rope.sin, mla)?;
            }
            if let Some(state) = dsa_state.as_deref_mut() {
                glm52_prefill_indexer(backend, cfg, mla, weights.indexer.as_ref(), layer, Some(&*experts), state, normed, &q_a, rope, position)?;
            }
            if paired_mla { (Some(q_a), None) } else { (None, Some(backend.linear(&q_a, &weights.q_b_proj)?)) }
        } else if paired_mla {
            // IndexShare 只复用 owner 最近一次 selection；q_b 从这里开始才拆给 peer。
            let q_a = backend.rmsnorm_quantized(&q_a, &weights.q_a_norm, cfg.rms_eps)?;
            if parallel_mla {
                backend.begin_parallel_mla_query(layer, experts, &q_a, position, &rope.cos, &rope.sin, mla)?;
            }
            (Some(q_a), None)
        } else {
            // IndexShare 层复用最近 full indexer 的 selection，不能退回 dense MLA。
            (None, Some(backend.rmsnorm_linear(&q_a, &weights.q_a_norm, cfg.rms_eps, &weights.q_b_proj)?))
        };
        let query_micros = mark_phase();
        backend.profile_device_operator("glm_attn_kv")?;
        let (latent, k_rope) = backend.split_columns(&kv_a, mla.kv_lora_rank)?;
        let latent = backend.rmsnorm(&latent, &weights.kv_a_norm, cfg.rms_eps)?;
        let kv_micros = mark_phase();
        Ok((normalized_q_lora, query, latent, k_rope, query_micros, kv_micros))
    })();
    let finish = dsa_state.as_deref_mut().map(|state| backend.dsa_select_topk_finish(state)).transpose();
    let (normalized_q_lora, query, latent, k_rope, query_micros, kv_micros) = prepared?;
    finish?;
    let dsa = glm52_dsa_spec(cfg, mla);
    backend.profile_device_operator("glm_attn_mla")?;
    let out = if parallel_mla {
        backend.parallel_mla_prefill_add(layer, experts, normalized_q_lora.as_ref().expect("parallel MLA 必须保留 q_lora"), &latent, &k_rope, hidden, cache, dsa_state.as_deref(), position, &rope.cos, &rope.sin, mla, &dsa)?
    } else if cooperative_mla {
        backend.cooperative_mla_prefill_add(layer, experts, normalized_q_lora.as_ref().expect("cooperative MLA 必须保留 q_lora"), &latent, &k_rope, hidden, cache, dsa_state.as_deref(), position, &rope.cos, &rope.sin, mla, &dsa)?
    } else {
        let query = backend.rope(query.as_ref().expect("普通 MLA 必须生成完整 query"), mla.num_heads, mla.qk_rope_head_dim, mla.rotary_layout, position, &rope.cos, &rope.sin)?;
        let attention = mla_causal_prefill_rope(backend, &query, &latent, &k_rope, &weights.kv_b_proj, cache, layer, position, &rope.cos, &rope.sin, mla, &dsa, dsa_state)?;
        backend.linear_add(&attention, &weights.o_proj, hidden)?
    };
    let mla_micros = mark_phase();
    backend.profile_device_operator("glm_attn_out")?;
    let out_micros = mark_phase();

    backend.profile_device_operator("glm_ffn")?;
    let (spec, shared) = glm52_moe_spec(cfg, &weights.shared_gate, &weights.shared_up, &weights.shared_down);
    let ffn_weights = MoeFfnRef { router_weight: &weights.router_weight, router_bias: &weights.router_bias, shared_experts: &shared, selected_experts: None };
    let routed_weights = RoutedMoeWeightsRef { router: &weights.router_weight, bias: &weights.router_bias, selected_experts: None };
    let parallel_result = if route_trace.is_none() { backend.parallel_moe_rmsnorm_add(&spec, routed_weights, &shared, layer, experts, &out, &weights.post_attn_norm, cfg.rms_eps)? } else { None };
    let result = if let Some(result) = parallel_result {
        Ok(result)
    } else if let Some((route_input, expert_input)) = backend.rmsnorm_quantized_pair(&out, &weights.post_attn_norm, cfg.rms_eps)? {
        let inputs = RoutedMoeInputs { route: &route_input, expert: &expert_input };
        match route_trace {
            Some(trace) => {
                let ffn = crate::moe::prefill::prefill_experts_inputs(backend, &spec, &ffn_weights, layer, experts, inputs, None)?;
                trace.record_layer(layer, ffn.routing.rows, ffn.routing.top_k, &ffn.routing.expert_ids).map_err(BackendError::ExpertLoad)?;
                backend.add(&out, &ffn.tensor)
            }
            None => crate::moe::prefill::prefill_experts_inputs_add_residual_untraced(backend, &spec, &ffn_weights, layer, experts, inputs, &out, None),
        }
    } else {
        let mlp_normed = backend.rmsnorm_f32(&out, &weights.post_attn_norm, cfg.rms_eps)?;
        match route_trace {
            Some(trace) => {
                let ffn = crate::moe::prefill::prefill_experts(backend, &spec, &ffn_weights, layer, experts, &mlp_normed, None)?;
                trace.record_layer(layer, ffn.routing.rows, ffn.routing.top_k, &ffn.routing.expert_ids).map_err(BackendError::ExpertLoad)?;
                backend.add(&out, &ffn.tensor)
            }
            None => crate::moe::prefill::prefill_experts_add_residual_untraced(backend, &spec, &ffn_weights, layer, experts, &mlp_normed, &out, None),
        }
    }?;
    let moe_micros = mark_phase();
    if let Some(total_started) = total_started {
        let total_micros = total_started.elapsed().as_micros();
        if total_micros >= 100_000 {
            eprintln!(
                "[glm52-layer-host-slow] ts_us={} kind=moe layer={layer} position={position} total_ms={:.3} begin_ms={:.3} input_ms={:.3} query_ms={:.3} kv_ms={:.3} mla_ms={:.3} out_ms={:.3} moe_ms={:.3} complete_us={}",
                total_start_us.unwrap_or_default(),
                total_micros as f64 / 1000.0,
                begin_micros as f64 / 1000.0,
                input_micros as f64 / 1000.0,
                query_micros as f64 / 1000.0,
                kv_micros as f64 / 1000.0,
                mla_micros as f64 / 1000.0,
                out_micros as f64 / 1000.0,
                moe_micros as f64 / 1000.0,
                crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
            );
        }
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_dense_decode_layer<B: DecodeBackend>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52DenseDecodeLayer<B::Weight>,
    layer: usize,
    dsa_state: &mut B::DsaState,
    hidden: &B::Tensor,
    rope: &RopeTable,
    cache: &mut B::Cache,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    backend.begin_decode_batch();
    let dsa = glm52_dsa_spec(cfg, mla);
    let indexer = weights.indexer.as_ref().map(|w| DsaWeightsRef { wq_b: &w.wq_b, wk: &w.wk, weights_proj: &w.weights_proj, k_norm_weight: &w.k_norm_weight, k_norm_bias: &w.k_norm_bias });
    let out = mla_decode(
        backend,
        mla,
        &dsa,
        indexer,
        dsa_state,
        MlaDecodeWeights {
            input_norm: &weights.input_norm,
            q_a_proj: &weights.q_a_proj,
            q_a_norm: &weights.q_a_norm,
            q_b_proj: &weights.q_b_proj,
            kv_a_proj: &weights.kv_a_proj,
            kv_a_norm: &weights.kv_a_norm,
            kv_b_proj: &weights.kv_b_proj,
            o_proj: &weights.o_proj,
        },
        hidden,
        cache,
        layer,
        position,
        &rope.cos,
        &rope.sin,
        cfg.rms_eps,
    )?;
    let normed = backend.rmsnorm(&out, &weights.post_attn_norm, cfg.rms_eps)?;
    let spec = DenseMlpSpec { intermediate_size: cfg.dense_intermediate_size, activation: Activation::Silu };
    let ffn = crate::moe::dense_mlp::decode(backend, &spec, DenseMlpWeightsRef { gate: &weights.gate_proj, up: &weights.up_proj, down: &weights.down_proj }, &normed)?;
    backend.add(&out, &ffn)
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_decode_layer<B: ExpertDecodeBackend + DecodeBackend, S: ExpertSourceProvider>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52MoeDecodeLayer<B::Weight>,
    layer: usize,
    expert_sources: &S,
    state: &mut ExpertDecodePipeline<B::MoeState>,
    dsa_state: &mut B::DsaState,
    hidden: &B::Tensor,
    rope: &RopeTable,
    cache: &mut B::Cache,
    position: usize,
) -> Result<B::Tensor, BackendError> {
    backend.begin_decode_batch();
    let dsa = glm52_dsa_spec(cfg, mla);
    let indexer = weights.indexer.as_ref().map(|w| DsaWeightsRef { wq_b: &w.wq_b, wk: &w.wk, weights_proj: &w.weights_proj, k_norm_weight: &w.k_norm_weight, k_norm_bias: &w.k_norm_bias });
    let out = mla_decode(
        backend,
        mla,
        &dsa,
        indexer,
        dsa_state,
        MlaDecodeWeights {
            input_norm: &weights.input_norm,
            q_a_proj: &weights.q_a_proj,
            q_a_norm: &weights.q_a_norm,
            q_b_proj: &weights.q_b_proj,
            kv_a_proj: &weights.kv_a_proj,
            kv_a_norm: &weights.kv_a_norm,
            kv_b_proj: &weights.kv_b_proj,
            o_proj: &weights.o_proj,
        },
        hidden,
        cache,
        layer,
        position,
        &rope.cos,
        &rope.sin,
        cfg.rms_eps,
    )?;
    let (spec, shared) = glm52_moe_spec(cfg, &weights.shared_gate, &weights.shared_up, &weights.shared_down);
    let ffn_weights = MoeFfnRef { router_weight: &weights.router_weight, router_bias: &weights.router_bias, shared_experts: &shared, selected_experts: None };
    let source = expert_sources.source(layer).map_err(BackendError::ExpertLoad)?;
    let next_source = if layer + 1 < cfg.layer_count {
        let next_layer = layer + 1;
        Some((next_layer, expert_sources.source(next_layer).map_err(BackendError::ExpertLoad)?))
    } else {
        None
    };
    let request = crate::runtime::expert_pipeline::ExpertDecodeRequest { layer, source, position, next: next_source };
    let ffn = match backend.rmsnorm_quantized_pair(&out, &weights.post_attn_norm, cfg.rms_eps)? {
        Some((route, expert)) => state.decode_inputs(backend, &spec, &ffn_weights, request, RoutedMoeInputs { route: &route, expert: &expert })?,
        None => {
            let normed = backend.rmsnorm_f32(&out, &weights.post_attn_norm, cfg.rms_eps)?;
            state.decode(backend, &spec, &ffn_weights, request, &normed)?
        }
    };
    backend.add(&out, &ffn)
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_decode_layers<B, S>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    layers: &[Glm52DecodeLayer<B::Weight>],
    expert_sources: &S,
    expert_state: &mut ExpertDecodePipeline<B::MoeState>,
    dsa_state: &mut B::DsaState,
    mut hidden: B::Tensor,
    rope: &RopeTable,
    cache: &mut B::Cache,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertDecodeBackend + DecodeBackend,
    S: ExpertSourceProvider,
{
    let result = (|| {
        for (layer, layer_weights) in layers.iter().enumerate() {
            hidden = match layer_weights {
                Glm52DecodeLayer::Dense(layer_weights) => glm52_dense_decode_layer(backend, cfg, mla, layer_weights, layer, dsa_state, &hidden, rope, cache, position),
                Glm52DecodeLayer::Moe(layer_weights) => glm52_decode_layer(backend, cfg, mla, layer_weights, layer, expert_sources, expert_state, dsa_state, &hidden, rope, cache, position),
            }
            .map_err(|error| BackendError::Compute { msg: format!("GLM-5.2 decode L{layer} position={position}: {error:?}") })?;
            backend.submit_batch();
        }
        Ok(hidden)
    })();
    backend.finish_batch();
    result
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_mtp_decode<B, S>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52Mtp<B::Weight>,
    expert_sources: &S,
    expert_state: &mut ExpertDecodePipeline<B::MoeState>,
    dsa_state: &mut B::DsaState,
    cache: &mut B::Cache,
    token_embedding: &B::Tensor,
    target_hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertDecodeBackend + DecodeBackend,
    S: ExpertSourceProvider,
{
    let result = (|| {
        let fused = super::mtp_project(backend, token_embedding, target_hidden, &weights.embedding_norm, &weights.hidden_norm, &weights.input_projection, cfg.hidden_size, NormSpec::Rms { eps: cfg.rms_eps }, false)?;
        let hidden = glm52_decode_layer(backend, cfg, mla, &weights.layer, cfg.layer_count, expert_sources, expert_state, dsa_state, &fused, rope, cache, position)?;
        backend.rmsnorm(&hidden, &weights.output_norm, cfg.rms_eps)
    })();
    backend.finish_batch();
    result
}

/// MTP catch-up 与草稿共用因果多行路径。prompt/verify 用多行补齐真实 hidden，
/// draft 用单行递推；两者共享同一份 KV/DSA 逻辑长度。
#[allow(clippy::too_many_arguments)]
pub fn glm52_mtp_prefill<B>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52Mtp<B::Weight>,
    experts: &mut B::PrefillExperts,
    dsa_state: &mut B::DsaState,
    cache: &mut B::Cache,
    token_embedding: &B::Tensor,
    target_hidden: &B::Tensor,
    rope: &RopeTable,
    position: usize,
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + DsaPrefillBackend,
{
    let result = (|| {
        let fused = super::mtp_project(backend, token_embedding, target_hidden, &weights.embedding_norm, &weights.hidden_norm, &weights.input_projection, cfg.hidden_size, NormSpec::Rms { eps: cfg.rms_eps }, false)?;
        let hidden = glm52_moe_prefill_layer(backend, cfg, mla, &weights.layer, cfg.layer_count, experts, None, Some(dsa_state), &fused, rope, Some(cache), position)?;
        backend.rmsnorm(&hidden, &weights.output_norm, cfg.rms_eps)
    })();
    backend.finish_batch();
    result
}

/// 多会话 MTP L78：输入按 session 拼行，KV/DSA 仍按 segment 写回各自状态。
/// catch-up 与每个 draft depth 共用这条路径，使 4 路 × 1/4 行能直接形成
/// ROCm W8 WMMA 的有效 M。
#[allow(clippy::too_many_arguments)]
pub fn glm52_mtp_prefill_segmented<B>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52Mtp<B::Weight>,
    experts: &mut B::PrefillExperts,
    token_embedding: &B::Tensor,
    target_hidden: &B::Tensor,
    rope: &RopeTable,
    reuse_dsa_selection: bool,
    segments: &mut [Glm52PrefillSegment<'_, B>],
) -> Result<B::Tensor, BackendError>
where
    B: ExpertPrefillBackend + DsaPrefillBackend + SegmentedTensorBackend,
{
    let result = (|| {
        let fused = super::mtp_project(backend, token_embedding, target_hidden, &weights.embedding_norm, &weights.hidden_norm, &weights.input_projection, cfg.hidden_size, NormSpec::Rms { eps: cfg.rms_eps }, false)?;
        backend.profile_device_operator(if reuse_dsa_selection { "glm_mtp_index_share" } else { "glm_mtp_index_seed" })?;
        let hidden = glm52_moe_prefill_layer_segmented(backend, cfg, mla, &weights.layer, cfg.layer_count, experts, &fused, rope, reuse_dsa_selection, segments)?;
        // MTP L78 是独立的一层逻辑 stage；output norm/head 只消费 owner，
        // 在丢掉 replica 前记录 peer completion，下一轮才可安全复用工作区。
        backend.finish_parallel_stage_submission(experts, &hidden)?;
        backend.rmsnorm(&hidden, &weights.output_norm, cfg.rms_eps)
    })();
    backend.finish_batch();
    result
}

/// MTP 接受后的追赶只需要补齐 L78 的 KV/DSA；本轮 L78 输出不会参与后续
/// 计算，下一轮草稿直接使用 target 的 terminal hidden。省去 query、attention、
/// output projection、MoE 与 output norm，且保持下一次 decode 所需 cache 完整。
#[allow(clippy::too_many_arguments)]
pub fn glm52_mtp_cache_segmented<B>(
    backend: &B,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    weights: &Glm52Mtp<B::Weight>,
    experts: &B::PrefillExperts,
    token_embedding: &B::Tensor,
    target_hidden: &B::Tensor,
    rope: &RopeTable,
    segments: &mut [Glm52PrefillSegment<'_, B>],
) -> Result<(), BackendError>
where
    B: DsaPrefillBackend + ExpertPrefillBackend + SegmentedTensorBackend,
{
    let result = (|| {
        backend.begin_batch();
        let fused = super::mtp_project(backend, token_embedding, target_hidden, &weights.embedding_norm, &weights.hidden_norm, &weights.input_projection, cfg.hidden_size, NormSpec::Rms { eps: cfg.rms_eps }, false)?;
        let layer = &weights.layer;
        let normalized = backend.rmsnorm_quantized(&fused, &layer.input_norm, cfg.rms_eps)?;

        let kv_a = backend.linear(&normalized, &layer.kv_a_proj)?;
        let (latent, k_rope) = backend.split_columns(&kv_a, mla.kv_lora_rank)?;
        let latent = backend.rmsnorm(&latent, &layer.kv_a_norm, cfg.rms_eps)?;
        let rope_segments = segments.iter().map(|segment| crate::backend::TokenSegment { position: segment.position, rows: segment.rows }).collect::<Vec<_>>();
        let k_rope = backend.rope_segmented(&k_rope, 1, mla.qk_rope_head_dim, mla.rotary_layout, false, &rope_segments, &rope.cos, &rope.sin)?;
        let mut offset = 0_usize;
        for segment in segments.iter_mut() {
            let segment_latent = backend.slice_token_rows(&latent, offset, segment.rows)?;
            let segment_rope = backend.slice_token_rows(&k_rope, offset, segment.rows)?;
            if !backend.parallel_mla_cache_append(cfg.layer_count, experts, &mut *segment.cache, &segment_latent, &segment_rope, segment.position)? {
                backend.append_mla(&mut *segment.cache, cfg.layer_count, &segment_latent, &segment_rope)?;
            }
            offset += segment.rows;
        }
        backend.finish_parallel_mla_cache_submission(experts)?;

        if let Some(indexer) = layer.indexer.as_ref() {
            let dsa = glm52_dsa_spec(cfg, mla);
            let key = backend.linear(&normalized, &indexer.wk)?;
            let fused = !segments.is_empty() && segments.iter().all(|segment| backend.supports_dsa_keys_layernorm_rope(&*segment.dsa, &key, &indexer.k_norm_weight, &indexer.k_norm_bias, &dsa));
            if fused {
                let mut offset = 0_usize;
                let segment_keys = segments
                    .iter()
                    .map(|segment| {
                        let segment_key = backend.slice_token_rows(&key, offset, segment.rows);
                        offset += segment.rows;
                        segment_key
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for (segment, segment_key) in segments.iter_mut().zip(segment_keys.iter()) {
                    if !backend.append_dsa_keys_layernorm_rope(&mut *segment.dsa, cfg.layer_count, segment.position, segment_key, &indexer.k_norm_weight, &indexer.k_norm_bias, 1.0e-6, &rope.cos, &rope.sin, &dsa)? {
                        return Err(BackendError::Compute { msg: "GLM-5.2 segmented MTP DSA 融合预检后拒绝执行".to_owned() });
                    }
                }
            } else {
                let key = backend.layernorm_bias(&key, &indexer.k_norm_weight, &indexer.k_norm_bias, 1.0e-6)?;
                let key = backend.rope_segmented(&key, 1, dsa.rope_dim, dsa.rotary_layout, true, &rope_segments, &rope.cos, &rope.sin)?;
                let mut offset = 0_usize;
                for segment in segments.iter_mut() {
                    let segment_key = backend.slice_token_rows(&key, offset, segment.rows)?;
                    backend.append_dsa_keys(&mut *segment.dsa, cfg.layer_count, segment.position, &segment_key, &dsa)?;
                    offset += segment.rows;
                }
            }
        }
        Ok(())
    })();
    backend.finish_batch();
    result
}

pub type Glm52OutputHead<W> = super::output::OutputHead<W>;

pub fn prepare_glm52_output_head<B>(backend: &B, cfg: &Glm52Config, final_norm: &[f32], lm_head: LinearWeight<'_>) -> Result<Glm52OutputHead<B::Weight>, BackendError>
where
    B: crate::backend::Backend,
{
    prepare_glm52_output_head_quantized(backend, cfg, final_norm, lm_head, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_glm52_output_head_quantized<B>(backend: &B, cfg: &Glm52Config, final_norm: &[f32], lm_head: LinearWeight<'_>, quantization: crate::weight::LmHeadQuantization) -> Result<Glm52OutputHead<B::Weight>, BackendError>
where
    B: crate::backend::Backend,
{
    super::output::prepare_output_head_quantized(backend, final_norm, lm_head, cfg.vocab_size, cfg.hidden_size, quantization)
}

pub fn glm52_token_output<B>(backend: &B, cfg: &Glm52Config, head: &Glm52OutputHead<B::Weight>, hidden: &B::Tensor) -> Result<super::output::OutputResult<B::Tensor>, BackendError>
where
    B: crate::backend::Backend,
{
    super::output::token_output(backend, head, hidden, &super::output::OutputPlan { eps: cfg.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: Vec::new() })
}

pub fn glm52_sampled_token_ids<B>(backend: &B, cfg: &Glm52Config, head: &Glm52OutputHead<B::Weight>, hidden: &B::Tensor, sampling: &[crate::backend::TokenSampling]) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    super::output::sampled_token_ids(backend, head, hidden, &super::output::OutputPlan { eps: cfg.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: Vec::new() }, sampling)
}

pub fn glm52_sampled_token_ids_fenced<B>(
    backend: &B,
    cfg: &Glm52Config,
    head: &Glm52OutputHead<B::Weight>,
    hidden: &B::Tensor,
    sampling: &[crate::backend::TokenSampling],
    fences: &[crate::backend::TokenFence],
) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    super::output::sampled_token_ids_fenced(backend, head, hidden, &super::output::OutputPlan { eps: cfg.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: Vec::new() }, sampling, fences)
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(crate) fn glm52_normalize_target_hidden<B>(backend: &B, cfg: &Glm52Config, head: &Glm52OutputHead<B::Weight>, hidden: &B::Tensor) -> Result<B::Tensor, BackendError>
where
    B: Backend,
{
    super::output::normalize_hidden(backend, head, hidden, &super::output::OutputPlan { eps: cfg.rms_eps, norm: super::output::OutputNorm::Rms, excluded_tokens: Vec::new() })
}

pub fn glm52_mtp_token_output<B>(backend: &B, head: &Glm52OutputHead<B::Weight>, normalized_hidden: &B::Tensor) -> Result<u32, BackendError>
where
    B: crate::backend::Backend,
{
    super::output::normalized_token_id(backend, head, normalized_hidden, &[])
}

pub fn glm52_mtp_token_ids<B>(backend: &B, head: &Glm52OutputHead<B::Weight>, normalized_hidden: &B::Tensor) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    super::output::normalized_token_ids(backend, head, normalized_hidden, &[])
}

pub fn glm52_mtp_token_ids_fenced<B>(backend: &B, head: &Glm52OutputHead<B::Weight>, normalized_hidden: &B::Tensor, fences: &[crate::backend::TokenFence]) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    super::output::normalized_token_ids_fenced(backend, head, normalized_hidden, fences)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_select_begin<B: DecodeBackend>(
    backend: &B,
    state: &mut B::DsaState,
    hidden: &B::Tensor,
    q_lora: &B::Tensor,
    weights: DsaWeightsRef<'_, B::Weight>,
    layer: usize,
    position: usize,
    cos: &[f32],
    sin: &[f32],
    spec: &DsaSpec,
) -> Result<bool, BackendError> {
    if !backend.dsa_can_append(state, layer, position, spec) {
        return Ok(false);
    }
    let (key, head_weights) = if position < spec.top_k {
        (backend.linear(hidden, weights.wk)?, None)
    } else {
        let (key, head_weights) = backend.dual_linear(hidden, weights.wk, weights.weights_proj)?;
        (key, Some(head_weights))
    };
    if !backend.append_dsa_keys_layernorm_rope(state, layer, position, &key, weights.k_norm_weight, weights.k_norm_bias, 1.0e-6, cos, sin, spec)? {
        let key = backend.layernorm_bias(&key, weights.k_norm_weight, weights.k_norm_bias, 1.0e-6)?;
        let key = backend.rope_prefix(&key, 1, spec.rope_dim, spec.rotary_layout, position, cos, sin)?;
        backend.append_dsa_keys(state, layer, position, &key, spec)?;
    }
    if position < spec.top_k {
        return Ok(false);
    }
    let query = backend.linear(q_lora, weights.wq_b)?;
    let query = backend.rope_prefix(&query, spec.num_heads, spec.rope_dim, spec.rotary_layout, position, cos, sin)?;
    backend.dsa_select_topk_begin(state, layer, &query, &head_weights.expect("DSA head weights 已准备"), spec)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn prefill_select<B: DsaPrefillBackend + ExpertPrefillBackend>(
    backend: &B,
    experts: Option<&B::PrefillExperts>,
    state: &mut B::DsaState,
    hidden: &B::Tensor,
    q_lora: &B::Tensor,
    weights: DsaWeightsRef<'_, B::Weight>,
    layer: usize,
    position: usize,
    cos: &[f32],
    sin: &[f32],
    spec: &DsaSpec,
) -> Result<(), BackendError> {
    let diagnose = backend.token_rows(hidden) == 1;
    let mut phase_started = diagnose.then(Instant::now);
    let mut mark_phase = || {
        phase_started.as_mut().map_or(0, |started| {
            let elapsed = started.elapsed().as_micros();
            *started = Instant::now();
            elapsed
        })
    };
    let key = backend.linear(hidden, weights.wk)?;
    let key_micros = mark_phase();
    let cooperative_append = if let Some(experts) = experts { backend.cooperative_dsa_append_keys_layernorm_rope(layer, experts, state, position, &key, weights.k_norm_weight, weights.k_norm_bias, 1.0e-6, cos, sin, spec)? } else { false };
    if !cooperative_append && !backend.append_dsa_keys_layernorm_rope(state, layer, position, &key, weights.k_norm_weight, weights.k_norm_bias, 1.0e-6, cos, sin, spec)? {
        let key = backend.layernorm_bias(&key, weights.k_norm_weight, weights.k_norm_bias, 1.0e-6)?;
        let key = backend.rope_prefix(&key, 1, spec.rope_dim, spec.rotary_layout, position, cos, sin)?;
        backend.append_dsa_keys(state, layer, position, &key, spec)?;
    }
    let append_micros = mark_phase();
    if position.saturating_add(backend.token_rows(hidden)) <= spec.top_k {
        if key_micros >= 5_000 || append_micros >= 5_000 {
            eprintln!("[glm-indexer-step] layer={layer} position={position} key_ms={:.3} append_ms={:.3}", key_micros as f64 / 1000.0, append_micros as f64 / 1000.0,);
        }
        // 候选数尚未超过 Top-K 时 MLA 直接使用完整历史；head projection 没有
        // 消费者。单 token decode 不能继续沿用 prefill 的无条件投影。
        return Ok(());
    }
    backend.profile_device_operator("glm_index_query")?;
    let head_weights = backend.linear(hidden, weights.weights_proj)?;
    backend.profile_device_operator("glm_index_select")?;
    if let Some(experts) = experts
        && backend.cooperative_dsa_project_select_prefill(layer, experts, state, q_lora, weights.wq_b, &head_weights, position, cos, sin, spec)?
    {
        return Ok(());
    }
    let query = backend.linear(q_lora, weights.wq_b)?;
    let query = backend.rope_prefix(&query, spec.num_heads, spec.rope_dim, spec.rotary_layout, position, cos, sin)?;
    if let Some(experts) = experts
        && backend.cooperative_dsa_select_prefill(layer, experts, state, &query, &head_weights, spec)?
    {
        return Ok(());
    }
    backend.dsa_select_prefill_begin(state, layer, &query, &head_weights, spec)
}

/// MLA decode 权重视图。Dense 与 MoE 层只在矩阵存储格式上不同。
pub struct MlaDecodeWeights<'a, W> {
    pub input_norm: &'a W,
    pub q_a_proj: &'a W,
    pub q_a_norm: &'a W,
    pub q_b_proj: &'a W,
    pub kv_a_proj: &'a W,
    pub kv_a_norm: &'a W,
    pub kv_b_proj: &'a W,
    pub o_proj: &'a W,
}

/// MLA 单 token decode 算法流；设备资源、KV 布局和 kernel 选择由 Backend 实现。
#[allow(clippy::too_many_arguments)]
pub fn mla_decode<B: DecodeBackend>(
    backend: &B,
    spec: &MlaSpec,
    dsa_spec: &DsaSpec,
    dsa_weights: Option<DsaWeightsRef<'_, B::Weight>>,
    dsa_state: &mut B::DsaState,
    weights: MlaDecodeWeights<'_, B::Weight>,
    hidden: &B::Tensor,
    cache: &mut B::Cache,
    layer: usize,
    position: usize,
    cos: &[f32],
    sin: &[f32],
    eps: f32,
) -> Result<B::Tensor, BackendError> {
    let (query, kv_a, dsa_started) = if let Some(dsa_weights) = dsa_weights {
        let normed = backend.rmsnorm(hidden, weights.input_norm, eps)?;
        let (q_a, kv_a) = backend.dual_linear(&normed, weights.q_a_proj, weights.kv_a_proj)?;
        let q_a = backend.rmsnorm(&q_a, weights.q_a_norm, eps)?;
        let dsa_started = decode_select_begin(backend, dsa_state, &normed, &q_a, dsa_weights, layer, position, cos, sin, dsa_spec)?;
        let query = match backend.linear(&q_a, weights.q_b_proj) {
            Ok(query) => query,
            Err(error) => {
                if dsa_started {
                    let _ = backend.dsa_select_topk_finish(dsa_state);
                }
                return Err(error);
            }
        };
        (query, kv_a, dsa_started)
    } else {
        let (q_a, kv_a) = backend.rmsnorm_dual_linear(hidden, weights.input_norm, eps, weights.q_a_proj, weights.kv_a_proj)?;
        (backend.rmsnorm_linear(&q_a, weights.q_a_norm, eps, weights.q_b_proj)?, kv_a, false)
    };
    let prepared = (|| {
        let (latent, k_rope) = backend.split_columns(&kv_a, spec.kv_lora_rank)?;
        let latent = backend.rmsnorm(&latent, weights.kv_a_norm, eps)?;
        let query = backend.rope(&query, spec.num_heads, spec.qk_rope_head_dim, spec.rotary_layout, position, cos, sin)?;
        backend.append_mla_rope(cache, layer, &latent, &k_rope, spec.qk_rope_head_dim, spec.rotary_layout, position, cos, sin)?;
        Ok(query)
    })();
    let query = match prepared {
        Ok(query) => query,
        Err(error) => {
            if dsa_started {
                let _ = backend.dsa_select_topk_finish(dsa_state);
            }
            return Err(error);
        }
    };
    if dsa_started {
        backend.dsa_select_topk_finish(dsa_state)?;
    }
    let attention = backend.mla_decode_attention_selected(&query, cache, weights.kv_b_proj, layer, position + 1, spec, dsa_spec, dsa_state)?;
    let projected = backend.linear(&attention, weights.o_proj)?;
    backend.add(hidden, &projected)
}

// GLM-5.2 模型规格。
//
// 架构:MLA 注意力 + DeepSeekMoE 风格路由(256 专家 / top-8)+ RMSNorm + SwiGLU。
// 前 3 层 dense,后 75 层 MoE。

pub use crate::model_spec::glm52::{GLM52_LAYER_COUNT, Glm52Config, is_indexer_layer};

/// GLM-5.2 模型。持有 config 与每层预计算的 LayerSpec。
pub struct Glm52 {
    config: Glm52Config,
    layer_specs: Vec<LayerSpec>,
}

impl Glm52 {
    pub fn new(config: Glm52Config) -> Result<Self, ModelError> {
        if config.layer_count == 0 || config.dense_layer_count > config.layer_count {
            return Err(ModelError::InvalidArchitecture("dense_layer_count > layer_count".into()));
        }
        if config.expert_top_k == 0 || config.expert_top_k > config.expert_count {
            return Err(ModelError::InvalidArchitecture("expert_top_k 必须在 1..=expert_count".into()));
        }
        // MLA 维度减法与 RoPE 表构造依赖这些不变量，构造期一次性校验。
        Self::mla_spec(&config).validate().map_err(ModelError::InvalidArchitecture)?;
        let layer_specs = (0..config.layer_count).map(|layer| Self::build_layer_spec(&config, layer)).collect();
        Ok(Self { config, layer_specs })
    }

    fn mla_spec(config: &Glm52Config) -> MlaSpec {
        MlaSpec {
            q_lora_rank: config.q_lora_rank,
            kv_lora_rank: config.kv_lora_rank,
            qk_rope_head_dim: config.qk_rope_head_dim,
            q_projection_size: config.q_projection_size,
            kv_projection_size: config.kv_projection_size,
            num_heads: config.num_heads,
            rope_theta: config.rope_theta,
            rotary_layout: crate::attention::rope::RotaryLayout::Interleaved,
        }
    }

    pub fn standard() -> Self {
        Self::new(Glm52Config::standard()).expect("GLM-5.2 标准配置必须有效")
    }

    fn build_layer_spec(config: &Glm52Config, layer: LayerId) -> LayerSpec {
        let is_dense = layer < config.dense_layer_count;
        let attention = AttentionSpec::Mla(Self::mla_spec(config));
        let feedforward = if is_dense { FeedforwardSpec::Dense(DenseMlpSpec { intermediate_size: config.dense_intermediate_size, activation: Activation::Silu }) } else { FeedforwardSpec::TopkMoe(glm52_topk_moe_spec(config)) };
        LayerSpec { attention, feedforward, input_norm: NormSpec::Rms { eps: config.rms_eps }, post_attention_norm: NormSpec::Rms { eps: config.rms_eps }, post_norm: None }
    }
}

impl Model for Glm52 {
    type Config = Glm52Config;

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
