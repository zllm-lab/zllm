//! Ornith-1.0-35B 模型算法与执行组合。

pub mod cpu;
#[cfg(feature = "with-cuda")]
pub mod cuda;
#[cfg(feature = "with-cuda")]
pub mod cuda_node;
pub mod node;
pub mod options;
pub mod protocol;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_node;

use crate::{
    attention::{
        gated_delta_net::{GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetState},
        gqa::GqaSpec,
        hybrid::{DeltaNetWeights, FullAttentionWeights, HybridAttention, HybridAttentionOptions, HybridTokenMixer},
        rope::RopeTable,
    },
    backend::{Backend, BackendError, ExpertDecodeBackend, ExpertPrefillBackend, GqaPrefillBackend, LinearWeight},
    moe::{
        Activation, FeedforwardSpec,
        topk_moe::{MoeFfnRef, ScoringFunc, SharedExpertRef, TopkMoeSpec},
    },
    norm::NormSpec,
    runtime::{LayerId, LayerSpec, Model, ModelError},
    weight::expert_source::ExpertSourceProvider,
};

use super::expert_pipeline::{ExpertDecodePipeline, ExpertDecodeRequest};
use crate::moe::prefill::prefill_experts_observed;

#[derive(Debug)]
pub struct OrnithMoe<W> {
    pub router_weight: W,
    /// Softmax router 不使用 bias；保留零向量让现有通用 MoE capability 维持统一 shape。
    pub router_bias: W,
    pub shared_output_gate: W,
    pub shared_gate: W,
    pub shared_up: W,
    pub shared_down: W,
}

#[derive(Debug)]
pub struct OrnithLayer<W> {
    pub input_norm: W,
    pub token_mixer: HybridTokenMixer<W>,
    pub post_attention_norm: W,
    pub moe: OrnithMoe<W>,
}

#[derive(Debug)]
pub struct OrnithMtp<W> {
    pub embedding_norm: W,
    pub hidden_norm: W,
    pub input_projection: W,
    pub layer: OrnithLayer<W>,
    pub output_norm: W,
}

pub fn prepare_ornith_layers<B: Backend>(backend: &B, source: &OrnithGguf) -> Result<Vec<OrnithLayer<B::Weight>>, BackendError> {
    (0..source.config().layer_count).map(|layer| prepare_ornith_layer(backend, source, layer)).collect()
}

pub fn prepare_ornith_layer<B: Backend>(backend: &B, source: &OrnithGguf, layer: usize) -> Result<OrnithLayer<B::Weight>, BackendError> {
    let cfg = source.config();
    let kind = cfg.ornith_layer_kind(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
    let prefix = format!("blk.{layer}");
    let input_norm = prepare_gemma_f32_vector(backend, source, &format!("{prefix}.attn_norm.weight"))?;
    let post_attention_norm = prepare_gemma_f32_vector(backend, source, &format!("{prefix}.post_attention_norm.weight"))?;
    let token_mixer = match kind {
        OrnithLayerKind::FullAttention => HybridTokenMixer::FullAttention(FullAttentionWeights {
            query_gate: prepare_matrix(backend, source, &format!("{prefix}.attn_q.weight"))?,
            query_norm: prepare_gemma_f32_vector(backend, source, &format!("{prefix}.attn_q_norm.weight"))?,
            key: prepare_matrix(backend, source, &format!("{prefix}.attn_k.weight"))?,
            key_norm: prepare_gemma_f32_vector(backend, source, &format!("{prefix}.attn_k_norm.weight"))?,
            value: prepare_matrix(backend, source, &format!("{prefix}.attn_v.weight"))?,
            output: prepare_matrix(backend, source, &format!("{prefix}.attn_output.weight"))?,
        }),
        OrnithLayerKind::DeltaNet => HybridTokenMixer::DeltaNet(DeltaNetWeights {
            qkv: prepare_matrix(backend, source, &format!("{prefix}.attn_qkv.weight"))?,
            z: prepare_matrix(backend, source, &format!("{prefix}.attn_gate.weight"))?,
            alpha: prepare_f32_matrix(backend, source, &format!("{prefix}.ssm_alpha.weight"))?,
            beta: prepare_f32_matrix(backend, source, &format!("{prefix}.ssm_beta.weight"))?,
            conv: prepare_matrix(backend, source, &format!("{prefix}.ssm_conv1d.weight"))?,
            a_log: prepare_a_log_vector(backend, source, &format!("{prefix}.ssm_a"))?,
            dt_bias: prepare_f32_vector(backend, source, &format!("{prefix}.ssm_dt.bias"))?,
            norm: prepare_f32_vector(backend, source, &format!("{prefix}.ssm_norm.weight"))?,
            output: prepare_matrix(backend, source, &format!("{prefix}.ssm_out.weight"))?,
        }),
    };
    Ok(OrnithLayer {
        input_norm,
        token_mixer,
        post_attention_norm,
        moe: OrnithMoe {
            router_weight: prepare_f32_matrix(backend, source, &format!("{prefix}.ffn_gate_inp.weight"))?,
            router_bias: backend.prepare_f32(&vec![0.0; cfg.num_experts], cfg.num_experts, 1)?,
            shared_output_gate: prepare_f32_vector(backend, source, &format!("{prefix}.ffn_gate_inp_shexp.weight"))?,
            shared_gate: prepare_matrix(backend, source, &format!("{prefix}.ffn_gate_shexp.weight"))?,
            shared_up: prepare_matrix(backend, source, &format!("{prefix}.ffn_up_shexp.weight"))?,
            shared_down: prepare_matrix(backend, source, &format!("{prefix}.ffn_down_shexp.weight"))?,
        },
    })
}

pub fn prepare_ornith_output<B: Backend>(backend: &B, source: &OrnithGguf) -> Result<(B::Weight, B::Weight), BackendError> {
    prepare_ornith_output_quantized(backend, source, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_ornith_output_quantized<B: Backend>(backend: &B, source: &OrnithGguf, quantization: crate::weight::LmHeadQuantization) -> Result<(B::Weight, B::Weight), BackendError> {
    let norm = prepare_gemma_f32_vector(backend, source, "output_norm.weight")?;
    let output = source.output_head().map_err(crate::runtime::compute_error)?;
    let head = super::output::prepare_lm_head_weight(backend, LinearWeight::gguf(&output), source.config().vocab_size, source.config().hidden_size, quantization)?;
    Ok((norm, head))
}

pub type OrnithOutputHead<W> = super::output::OutputHead<W>;

pub fn prepare_ornith_output_head<B: Backend>(backend: &B, source: &OrnithGguf) -> Result<OrnithOutputHead<B::Weight>, BackendError> {
    prepare_ornith_output_head_quantized(backend, source, crate::weight::LmHeadQuantization::Native)
}

pub fn prepare_ornith_output_head_quantized<B: Backend>(backend: &B, source: &OrnithGguf, quantization: crate::weight::LmHeadQuantization) -> Result<OrnithOutputHead<B::Weight>, BackendError> {
    let final_norm = source.gemma_norm_vector("output_norm.weight").map_err(crate::runtime::compute_error)?;
    let output = source.output_head().map_err(crate::runtime::compute_error)?;
    let decoded_output = if output.tensor_type.0 == 0 || output.tensor_type.0 == 1 { Some(output.decode().map_err(crate::runtime::compute_error)?) } else { None };
    let lm_head = match decoded_output.as_deref() {
        Some(values) => LinearWeight::F32(values),
        None => LinearWeight::gguf(&output),
    };
    super::output::prepare_output_head_gemma_quantized(backend, &final_norm, lm_head, source.config().vocab_size, source.config().hidden_size, quantization)
}

pub fn ornith_token_output<B: Backend>(backend: &B, cfg: &OrnithConfig, head: &OrnithOutputHead<B::Weight>, hidden: &B::Tensor) -> Result<super::output::OutputResult<B::Tensor>, BackendError> {
    super::output::token_output(backend, head, hidden, &super::output::OutputPlan { eps: cfg.rms_eps, norm: super::output::OutputNorm::GemmaRms, excluded_tokens: Vec::new() })
}

pub fn prepare_ornith_mtp<B: Backend>(backend: &B, source: &OrnithGguf) -> Result<OrnithMtp<B::Weight>, BackendError> {
    if !source.has_mtp() {
        return Err(BackendError::Compute { msg: "Ornith GGUF 不含 MTP layer".to_owned() });
    }
    let cfg = source.config();
    let layer = cfg.layer_count;
    let prefix = format!("blk.{layer}");
    Ok(OrnithMtp {
        embedding_norm: prepare_gemma_f32_vector(backend, source, &format!("{prefix}.nextn.enorm.weight"))?,
        hidden_norm: prepare_gemma_f32_vector(backend, source, &format!("{prefix}.nextn.hnorm.weight"))?,
        input_projection: prepare_matrix(backend, source, &format!("{prefix}.nextn.eh_proj.weight"))?,
        output_norm: prepare_gemma_f32_vector(backend, source, &format!("{prefix}.nextn.shared_head_norm.weight"))?,
        layer: OrnithLayer {
            input_norm: prepare_gemma_f32_vector(backend, source, &format!("{prefix}.attn_norm.weight"))?,
            token_mixer: HybridTokenMixer::FullAttention(FullAttentionWeights {
                query_gate: prepare_matrix(backend, source, &format!("{prefix}.attn_q.weight"))?,
                query_norm: prepare_gemma_f32_vector(backend, source, &format!("{prefix}.attn_q_norm.weight"))?,
                key: prepare_matrix(backend, source, &format!("{prefix}.attn_k.weight"))?,
                key_norm: prepare_gemma_f32_vector(backend, source, &format!("{prefix}.attn_k_norm.weight"))?,
                value: prepare_matrix(backend, source, &format!("{prefix}.attn_v.weight"))?,
                output: prepare_matrix(backend, source, &format!("{prefix}.attn_output.weight"))?,
            }),
            post_attention_norm: prepare_gemma_f32_vector(backend, source, &format!("{prefix}.post_attention_norm.weight"))?,
            moe: OrnithMoe {
                router_weight: prepare_f32_matrix(backend, source, &format!("{prefix}.ffn_gate_inp.weight"))?,
                router_bias: backend.prepare_f32(&vec![0.0; cfg.num_experts], cfg.num_experts, 1)?,
                shared_output_gate: prepare_f32_vector(backend, source, &format!("{prefix}.ffn_gate_inp_shexp.weight"))?,
                shared_gate: prepare_matrix(backend, source, &format!("{prefix}.ffn_gate_shexp.weight"))?,
                shared_up: prepare_matrix(backend, source, &format!("{prefix}.ffn_up_shexp.weight"))?,
                shared_down: prepare_matrix(backend, source, &format!("{prefix}.ffn_down_shexp.weight"))?,
            },
        },
    })
}

fn prepare_matrix<B: Backend>(backend: &B, source: &OrnithGguf, name: &str) -> Result<B::Weight, BackendError> {
    super::prepare_gguf_matrix(backend, source.reader(), name)
}

fn prepare_f32_matrix<B: Backend>(backend: &B, source: &OrnithGguf, name: &str) -> Result<B::Weight, BackendError> {
    super::prepare_gguf_f32_matrix(backend, source.reader(), name)
}

fn prepare_gemma_f32_vector<B: Backend>(backend: &B, source: &OrnithGguf, name: &str) -> Result<B::Weight, BackendError> {
    super::prepare_gguf_gemma_vector(backend, source.reader(), name)
}

fn prepare_f32_vector<B: Backend>(backend: &B, source: &OrnithGguf, name: &str) -> Result<B::Weight, BackendError> {
    super::prepare_gguf_f32_vector(backend, source.reader(), name)
}

fn prepare_a_log_vector<B: Backend>(backend: &B, source: &OrnithGguf, name: &str) -> Result<B::Weight, BackendError> {
    super::prepare_gguf_a_log_vector(backend, source.reader(), name)
}

pub struct OrnithLayerOutput<T> {
    pub hidden: T,
    pub active_experts: Vec<usize>,
    pub expert_ids: Vec<u32>,
    pub route_weights: Vec<f32>,
}

#[derive(Clone, Copy, Default)]
pub struct OrnithRuntimeOptions {
    pub attention: HybridAttentionOptions,
    pub expert_batch_size: Option<usize>,
}

/// 绑定一次模型加载期间保持不变的执行依赖，并缓存由模型配置派生的算子规格。
///
/// `layers` 是 `[first_layer, first_layer + layers.len())` 的连续层切片；多卡分层
/// 部署时每个设备各持一个 runtime，只跑自己的层区间，层 id 仍用全模型编号。
pub struct OrnithRuntime<'a, B: Backend> {
    backend: &'a B,
    config: &'a OrnithConfig,
    layers: &'a [OrnithLayer<B::Weight>],
    first_layer: usize,
    rope: &'a RopeTable,
    attention: HybridAttention<'a, B>,
    moe: TopkMoeSpec,
    options: OrnithRuntimeOptions,
}

impl<'a, B: Backend> OrnithRuntime<'a, B> {
    pub fn new(backend: &'a B, config: &'a OrnithConfig, layers: &'a [OrnithLayer<B::Weight>], first_layer: usize, rope: &'a RopeTable, options: OrnithRuntimeOptions) -> Self {
        Self { backend, config, layers, first_layer, rope, attention: HybridAttention::new(backend, config.full_attention_spec(), config.gated_delta_net_spec(), config.rms_eps, rope, options.attention), moe: topk_moe_spec(config), options }
    }

    /// 将同一 sequence 的两类状态绑定到一个位置，避免在层调用链重复透传。
    pub fn at<'run>(&'run self, cache: &'run mut B::Cache, recurrent: &'run mut GatedDeltaNetState<B::GatedDeltaNetStorage>, position: usize) -> OrnithExecution<'run, 'a, B>
    where
        B: GqaPrefillBackend + GatedDeltaNetKernel,
    {
        OrnithExecution { runtime: self, cache, recurrent, position }
    }

    fn full_attention_mixer(&self, weights: &FullAttentionWeights<B::Weight>, cache: &mut B::Cache, cache_layer: usize, input: &B::Tensor, position: usize) -> Result<B::Tensor, BackendError>
    where
        B: GqaPrefillBackend,
    {
        self.attention.full(weights, cache, cache_layer, input, position)
    }

    pub fn mtp_prefill(
        &self,
        weights: &OrnithMtp<B::Weight>,
        experts: &mut B::PrefillExperts,
        cache: &mut B::Cache,
        token_embedding: &B::Tensor,
        target_hidden: &B::Tensor,
        position: usize,
    ) -> Result<OrnithLayerOutput<B::Tensor>, BackendError>
    where
        B: ExpertPrefillBackend + GqaPrefillBackend,
    {
        let rows = self.backend.token_rows(token_embedding);
        if rows == 0 || rows != self.backend.token_rows(target_hidden) || self.backend.token_cols(token_embedding) != self.config.hidden_size || self.backend.token_cols(target_hidden) != self.config.hidden_size {
            return Err(BackendError::Compute {
                msg: format!("Ornith MTP 输入 shape 非法: embedding=[{},{}], hidden=[{},{}]", rows, self.backend.token_cols(token_embedding), self.backend.token_rows(target_hidden), self.backend.token_cols(target_hidden),),
            });
        }
        ensure_sequence_supported(self.config, position, rows, self.rope)?;
        self.backend.begin_decode_batch();
        let result = (|| {
            let _scope = self.backend.layer_scope();
            let embedding = self.backend.gemma_rmsnorm_f32(token_embedding, &weights.embedding_norm, self.config.rms_eps)?;
            let hidden = self.backend.gemma_rmsnorm_f32(target_hidden, &weights.hidden_norm, self.config.rms_eps)?;
            let fused = self.backend.concat_columns(&embedding, &hidden)?;
            let fused = self.backend.linear(&fused, &weights.input_projection)?;
            let normed = self.backend.gemma_rmsnorm_f32(&fused, &weights.layer.input_norm, self.config.rms_eps)?;
            let HybridTokenMixer::FullAttention(attention_weights) = &weights.layer.token_mixer else {
                return Err(BackendError::Compute { msg: "Ornith MTP 必须使用 full attention".to_owned() });
            };
            let mixed = self.full_attention_mixer(attention_weights, cache, 0, &normed, position)?;
            let residual = self.backend.add(&fused, &mixed)?;
            let moe_input = self.backend.gemma_rmsnorm_f32(&residual, &weights.layer.post_attention_norm, self.config.rms_eps)?;
            let shared = [SharedExpertRef { gate: &weights.layer.moe.shared_gate, up: &weights.layer.moe.shared_up, down: &weights.layer.moe.shared_down, output_gate: Some(&weights.layer.moe.shared_output_gate) }];
            let moe = prefill_experts_observed(
                self.backend,
                &self.moe,
                &MoeFfnRef { router_weight: &weights.layer.moe.router_weight, router_bias: &weights.layer.moe.router_bias, shared_experts: &shared, selected_experts: None },
                self.config.layer_count,
                experts,
                &moe_input,
                self.options.expert_batch_size,
                |_| {},
            )?;
            let hidden = self.backend.add(&residual, &moe.tensor)?;
            let hidden = self.backend.gemma_rmsnorm_f32(&hidden, &weights.output_norm, self.config.rms_eps)?;
            let active_experts = active_experts(&moe.routing.expert_ids, self.config.num_experts)?;
            Ok(OrnithLayerOutput { hidden, active_experts, expert_ids: moe.routing.expert_ids, route_weights: moe.routing.weights })
        })();
        self.backend.finish_batch();
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn mtp_decode_open<S>(
        &self,
        weights: &OrnithMtp<B::Weight>,
        expert_sources: &S,
        expert_state: &mut ExpertDecodePipeline<B::MoeState>,
        cache: &mut B::Cache,
        token_embedding: &B::Tensor,
        target_hidden: &B::Tensor,
        position: usize,
    ) -> Result<B::Tensor, BackendError>
    where
        B: ExpertDecodeBackend + GqaPrefillBackend,
        S: ExpertSourceProvider,
    {
        ensure_sequence_supported(self.config, position, 1, self.rope)?;
        let _scope = self.backend.layer_scope();
        let fused = crate::runtime::mtp_project(
            self.backend,
            token_embedding,
            target_hidden,
            &weights.embedding_norm,
            &weights.hidden_norm,
            &weights.input_projection,
            self.config.hidden_size,
            NormSpec::GemmaRms { eps: self.config.rms_eps },
            true,
        )?;
        let normed = self.backend.gemma_rmsnorm_f32(&fused, &weights.layer.input_norm, self.config.rms_eps)?;
        let HybridTokenMixer::FullAttention(attention_weights) = &weights.layer.token_mixer else {
            return Err(BackendError::Compute { msg: "Ornith MTP 必须使用 full attention".to_owned() });
        };
        let mixed = self.full_attention_mixer(attention_weights, cache, 0, &normed, position)?;
        let residual = self.backend.add(&fused, &mixed)?;
        let moe_input = self.backend.gemma_rmsnorm_f32(&residual, &weights.layer.post_attention_norm, self.config.rms_eps)?;
        let shared = [SharedExpertRef { gate: &weights.layer.moe.shared_gate, up: &weights.layer.moe.shared_up, down: &weights.layer.moe.shared_down, output_gate: Some(&weights.layer.moe.shared_output_gate) }];
        let source = expert_sources.source(self.config.layer_count).map_err(BackendError::ExpertLoad)?;
        let feedforward = expert_state.decode(
            self.backend,
            &self.moe,
            &MoeFfnRef { router_weight: &weights.layer.moe.router_weight, router_bias: &weights.layer.moe.router_bias, shared_experts: &shared, selected_experts: None },
            ExpertDecodeRequest { layer: self.config.layer_count, source, position, next: None },
            &moe_input,
        )?;
        let hidden = self.backend.add(&residual, &feedforward)?;
        self.backend.gemma_rmsnorm_f32(&hidden, &weights.output_norm, self.config.rms_eps)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mtp_decode<S>(
        &self,
        weights: &OrnithMtp<B::Weight>,
        expert_sources: &S,
        expert_state: &mut ExpertDecodePipeline<B::MoeState>,
        cache: &mut B::Cache,
        token_embedding: &B::Tensor,
        target_hidden: &B::Tensor,
        position: usize,
    ) -> Result<B::Tensor, BackendError>
    where
        B: ExpertDecodeBackend + GqaPrefillBackend,
        S: ExpertSourceProvider,
    {
        self.backend.begin_decode_batch();
        let result = self.mtp_decode_open(weights, expert_sources, expert_state, cache, token_embedding, target_hidden, position);
        self.backend.finish_batch();
        result
    }

    /// 让 MTP 的 lm_head 与 greedy argmax 复用当前 decode batch，只在 argmax readback 时等待 CPU。
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_decode_then<S, R, F>(
        &self,
        weights: &OrnithMtp<B::Weight>,
        expert_sources: &S,
        expert_state: &mut ExpertDecodePipeline<B::MoeState>,
        cache: &mut B::Cache,
        token_embedding: &B::Tensor,
        target_hidden: &B::Tensor,
        position: usize,
        then: F,
    ) -> Result<R, BackendError>
    where
        B: ExpertDecodeBackend + GqaPrefillBackend,
        S: ExpertSourceProvider,
        F: FnOnce(&B::Tensor) -> Result<R, BackendError>,
    {
        self.backend.begin_decode_batch();
        let result = self.mtp_decode_open(weights, expert_sources, expert_state, cache, token_embedding, target_hidden, position).and_then(|hidden| then(&hidden));
        self.backend.finish_batch();
        result
    }
}

pub struct OrnithExecution<'run, 'model, B>
where
    B: GqaPrefillBackend + GatedDeltaNetKernel,
{
    runtime: &'run OrnithRuntime<'model, B>,
    cache: &'run mut B::Cache,
    recurrent: &'run mut GatedDeltaNetState<B::GatedDeltaNetStorage>,
    position: usize,
}

pub fn ornith_layer_kind(cfg: &OrnithConfig, layer: usize) -> Option<OrnithLayerKind> {
    cfg.ornith_layer_kind(layer)
}

/// 从模型层计划生成通用 cache 映射；runtime 决定哪些层需要 KV，backend 只消费映射。
pub fn kv_cache_layer_map(cfg: &OrnithConfig) -> Result<crate::kv_cache::KvCacheLayerMap, String> {
    crate::kv_cache::KvCacheLayerMap::from_cached_layers(cfg.layer_count, (0..cfg.layer_count).filter(|&layer| cfg.ornith_layer_kind(layer) == Some(OrnithLayerKind::FullAttention)))
}

fn topk_moe_spec(cfg: &OrnithConfig) -> TopkMoeSpec {
    TopkMoeSpec {
        num_experts: cfg.num_experts,
        top_k: cfg.num_experts_per_tok,
        num_shared_experts: cfg.num_shared_experts,
        scoring_func: ScoringFunc::Softmax,
        normalize_selected: true,
        routed_scaling_factor: cfg.routed_scaling_factor,
        intermediate_size: cfg.expert_intermediate_size,
        shared_intermediate_size: cfg.shared_intermediate_size,
        activation: Activation::Silu,
    }
}

pub fn ensure_supported(cfg: &OrnithConfig) -> Result<(), BackendError> {
    Ornith::new(cfg.clone()).map(|_| ()).map_err(|error| BackendError::Compute { msg: format!("Ornith 配置非法: {error:?}") })
}

fn ensure_sequence_supported(cfg: &OrnithConfig, position: usize, token_count: usize, rope: &RopeTable) -> Result<(), BackendError> {
    if token_count == 0 {
        return Err(BackendError::Compute { msg: "Ornith 输入 token 数不能为 0".to_owned() });
    }
    let end = position.checked_add(token_count).ok_or_else(|| BackendError::Compute { msg: "position + token_count 溢出".to_owned() })?;
    if end > rope.seq_len {
        return Err(BackendError::Compute { msg: format!("Ornith 需要位置 {end}，RoPE 只预计算到 {}", rope.seq_len) });
    }
    if cfg.hidden_size == 0 {
        return Err(BackendError::Compute { msg: "Ornith hidden_size 不能为 0".to_owned() });
    }
    Ok(())
}

fn ensure_mixer_matches_layer<W>(cfg: &OrnithConfig, layer: usize, mixer: &HybridTokenMixer<W>) -> Result<(), BackendError> {
    let kind = cfg.ornith_layer_kind(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
    if matches!((kind, mixer), (OrnithLayerKind::FullAttention, HybridTokenMixer::FullAttention(_)) | (OrnithLayerKind::DeltaNet, HybridTokenMixer::DeltaNet(_))) {
        Ok(())
    } else {
        Err(BackendError::Compute { msg: format!("Ornith L{layer} token mixer 与 layer type 不一致") })
    }
}

fn active_experts(expert_ids: &[u32], expert_count: usize) -> Result<Vec<usize>, BackendError> {
    crate::moe::routing::active_experts_from_ids(expert_ids, expert_count).map_err(|msg| BackendError::Compute { msg })
}

impl<B> OrnithExecution<'_, '_, B>
where
    B: GqaPrefillBackend + GatedDeltaNetKernel,
{
    fn token_mixer(&mut self, weights: &HybridTokenMixer<B::Weight>, layer: usize, input: &B::Tensor) -> Result<B::Tensor, BackendError> {
        ensure_mixer_matches_layer(self.runtime.config, layer, weights)?;
        match weights {
            HybridTokenMixer::FullAttention(weights) => self.runtime.full_attention_mixer(weights, self.cache, layer, input, self.position),
            HybridTokenMixer::DeltaNet(weights) => self.runtime.attention.delta(weights, self.recurrent, layer, input, self.position),
        }
    }

    fn attention_block(&mut self, weights: &OrnithLayer<B::Weight>, layer: usize, hidden: &B::Tensor) -> Result<(B::Tensor, B::Tensor), BackendError> {
        let normed = self.runtime.backend.gemma_rmsnorm_f32(hidden, &weights.input_norm, self.runtime.config.rms_eps)?;
        let mixed = self.token_mixer(&weights.token_mixer, layer, &normed)?;
        let residual = self.runtime.backend.add(hidden, &mixed)?;
        let moe_input = self.runtime.backend.gemma_rmsnorm_f32(&residual, &weights.post_attention_norm, self.runtime.config.rms_eps)?;
        Ok((residual, moe_input))
    }

    fn prefill_layer(&mut self, weights: &OrnithLayer<B::Weight>, experts: &mut B::PrefillExperts, layer: usize, hidden: &B::Tensor) -> Result<OrnithLayerOutput<B::Tensor>, BackendError>
    where
        B: ExpertPrefillBackend,
    {
        let (residual, moe_input) = self.attention_block(weights, layer, hidden)?;
        let shared = [SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }];
        let moe = prefill_experts_observed(
            self.runtime.backend,
            &self.runtime.moe,
            &MoeFfnRef { router_weight: &weights.moe.router_weight, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None },
            layer,
            experts,
            &moe_input,
            self.runtime.options.expert_batch_size,
            |_| {},
        )?;
        let active_experts = active_experts(&moe.routing.expert_ids, self.runtime.config.num_experts)?;
        Ok(OrnithLayerOutput { hidden: self.runtime.backend.add(&residual, &moe.tensor)?, active_experts, expert_ids: moe.routing.expert_ids, route_weights: moe.routing.weights })
    }

    pub fn prefill(self, experts: &mut B::PrefillExperts, hidden: B::Tensor) -> Result<B::Tensor, BackendError>
    where
        B: ExpertPrefillBackend,
    {
        self.prefill_observed(experts, hidden, |_, _| Ok(()))
    }

    pub fn prefill_observed<O>(mut self, experts: &mut B::PrefillExperts, hidden: B::Tensor, mut observe: O) -> Result<B::Tensor, BackendError>
    where
        B: ExpertPrefillBackend,
        O: FnMut(usize, &B::Tensor) -> Result<(), BackendError>,
    {
        let backend = self.runtime.backend;
        let config = self.runtime.config;
        let layers = self.runtime.layers;
        let first_layer = self.runtime.first_layer;
        let token_count = backend.token_rows(&hidden);
        if first_layer + layers.len() > config.layer_count || token_count == 0 || backend.token_cols(&hidden) != config.hidden_size {
            return Err(BackendError::Compute {
                msg: format!("Ornith prefill 输入不完整: layers=[{first_layer},{}), total={}, hidden=[{},{}]", first_layer + layers.len(), config.layer_count, token_count, backend.token_cols(&hidden))
            });
        }
        ensure_sequence_supported(config, self.position, token_count, self.runtime.rope)?;
        let result = (|| {
            let mut hidden = hidden;
            for (offset, weights) in layers.iter().enumerate() {
                let layer = first_layer + offset;
                let _scope = backend.layer_scope();
                backend.begin_batch();
                hidden = self.prefill_layer(weights, experts, layer, &hidden)?.hidden;
                if backend.token_rows(&hidden) != token_count || backend.token_cols(&hidden) != config.hidden_size {
                    return Err(BackendError::Compute { msg: format!("Ornith L{layer} prefill 输出 shape 异常") });
                }
                // scope 释放临时编码资源前先提交；同一 queue 保证下一层依赖顺序，无需在层边界等待。
                backend.submit_batch();
                observe(layer, &hidden)?;
            }
            Ok(hidden)
        })();
        backend.finish_batch();
        result
    }

    fn decode_layer<S>(&mut self, weights: &OrnithLayer<B::Weight>, layer: usize, expert_sources: &S, expert_state: &mut ExpertDecodePipeline<B::MoeState>, hidden: &B::Tensor) -> Result<B::Tensor, BackendError>
    where
        B: ExpertDecodeBackend,
        S: ExpertSourceProvider,
    {
        self.runtime.backend.begin_decode_batch();
        let (residual, moe_input) = self.attention_block(weights, layer, hidden)?;
        let shared = [SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }];
        let ffn = MoeFfnRef { router_weight: &weights.moe.router_weight, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
        let source = expert_sources.source(layer).map_err(BackendError::ExpertLoad)?;
        // 只在本设备的层区间内预取下一层专家；区间外属于别的设备，不提前装载。
        let next_source = (layer + 1 < self.runtime.first_layer + self.runtime.layers.len()).then(|| expert_sources.source(layer + 1).map(|source| (layer + 1, source))).transpose().map_err(BackendError::ExpertLoad)?;
        let request = ExpertDecodeRequest { layer, source, position: self.position, next: next_source };
        let feedforward = expert_state.decode(self.runtime.backend, &self.runtime.moe, &ffn, request, &moe_input)?;
        self.runtime.backend.add(&residual, &feedforward)
    }

    pub fn decode<S>(mut self, expert_sources: &S, expert_state: &mut ExpertDecodePipeline<B::MoeState>, hidden: B::Tensor) -> Result<B::Tensor, BackendError>
    where
        B: ExpertDecodeBackend,
        S: ExpertSourceProvider,
    {
        let backend = self.runtime.backend;
        let config = self.runtime.config;
        let layers = self.runtime.layers;
        let first_layer = self.runtime.first_layer;
        if first_layer + layers.len() > config.layer_count || backend.token_rows(&hidden) == 0 || backend.token_cols(&hidden) != config.hidden_size {
            return Err(BackendError::Compute {
                msg: format!("Ornith decode 输入不完整: layers=[{first_layer},{}), total={}, hidden=[{},{}]", first_layer + layers.len(), config.layer_count, backend.token_rows(&hidden), backend.token_cols(&hidden)),
            });
        }
        ensure_sequence_supported(config, self.position, backend.token_rows(&hidden), self.runtime.rope)?;
        let result = (|| {
            let _scope = backend.layer_scope();
            let mut hidden = hidden;
            for (offset, weights) in layers.iter().enumerate() {
                hidden = self.decode_layer(weights, first_layer + offset, expert_sources, expert_state, &hidden)?;
            }
            Ok(hidden)
        })();
        backend.finish_batch();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_runtime_accepts_model_configuration() {
        assert!(ensure_supported(&OrnithConfig::standard()).is_ok());
    }

    #[test]
    fn full_attention_cache_map_is_compact() {
        let map = kv_cache_layer_map(&OrnithConfig::standard()).unwrap();
        assert_eq!(map.logical_layer_count(), 40);
        assert_eq!(map.slot_count(), 10);
        assert_eq!(map.cache_slot(3), Ok(0));
        assert_eq!(map.cache_slot(39), Ok(9));
        assert!(map.cache_slot(0).is_err());
    }
}

// Ornith-1.0-35B 模型规格。
//
// 参数与层序列对齐 Qwen3.5-MoE text config；MTP 是主干之外的独立 draft layer。

use crate::attention::AttentionSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrnithLayerKind {
    FullAttention,
    DeltaNet,
}

pub use crate::model_spec::ornith::OrnithConfig;

impl OrnithConfig {
    pub fn ornith_layer_kind(&self, layer: LayerId) -> Option<OrnithLayerKind> {
        if layer >= self.layer_count {
            return None;
        }
        let ordinal = layer + 1;
        if ordinal.is_multiple_of(self.full_attention_interval) && ordinal / self.full_attention_interval <= self.full_attention_layers { Some(OrnithLayerKind::FullAttention) } else { Some(OrnithLayerKind::DeltaNet) }
    }

    pub fn full_attention_spec(&self) -> GqaSpec {
        GqaSpec {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            rope_theta: self.rope_theta,
            use_qk_norm: self.use_qk_norm,
            window: crate::attention::gqa::CausalWindow::Full,
            score_scale: 1.0 / (self.head_dim as f32).sqrt(),
            output_gate: true,
        }
    }

    pub fn gated_delta_net_spec(&self) -> GatedDeltaNetSpec {
        GatedDeltaNetSpec {
            key_heads: self.linear_key_heads,
            value_heads: self.linear_value_heads,
            key_head_dim: self.linear_key_head_dim,
            value_head_dim: self.linear_value_head_dim,
            conv_kernel: self.linear_conv_kernel,
            rms_eps: self.rms_eps,
        }
    }

    /// 从 GGUF qwen35moe 元数据构造配置；Ornith 家族 9B/35B/397B 共用同一映射。
    /// ssm.inner_size / state_size 得到 value 头数（1.0-35B: 4096/128=32，1.5-397B: 8192/128=64）。
    pub fn from_gguf(reader: &crate::weight::container::gguf::GgufReader) -> Result<Self, String> {
        let value = |key: &str| reader.metadata_u64(key).map_err(|error| format!("Ornith GGUF {error}"));
        let float = |key: &str| match reader.metadata(key) {
            Some(crate::weight::container::gguf::GgufValue::Float(value)) => Ok(*value as f32),
            Some(crate::weight::container::gguf::GgufValue::Unsigned(value)) => Ok(*value as f32),
            _ => Err(format!("Ornith GGUF metadata {key} 缺失或不是数值")),
        };
        let layer_count = value("qwen35moe.block_count")? as usize;
        let interval = value("qwen35moe.full_attention_interval")? as usize;
        let state_size = value("qwen35moe.ssm.state_size")? as usize;
        let inner_size = value("qwen35moe.ssm.inner_size")? as usize;
        let key_length = value("qwen35moe.attention.key_length")? as usize;
        let value_length = value("qwen35moe.attention.value_length")? as usize;
        if key_length != value_length {
            return Err(format!("Ornith GGUF attention.key_length({key_length}) != value_length({value_length})"));
        }
        if interval == 0 || state_size == 0 || !inner_size.is_multiple_of(state_size) {
            return Err(format!("Ornith GGUF ssm/full_attention 元数据非法: interval={interval}, state_size={state_size}, inner_size={inner_size}"));
        }
        // GGUF dims 为 [ne0=hidden, ne1=vocab]，词表大小在最后一维。
        let vocab_size = reader
            .tensor("token_embd.weight")
            .and_then(|tensor| tensor.dims.last().copied())
            .filter(|_| reader.tensor("token_embd.weight").is_some_and(|t| t.dims.len() == 2))
            .ok_or_else(|| "Ornith GGUF token_embd.weight 必须是 [hidden, vocab] 二维".to_owned())?;
        Ok(Self {
            vocab_size,
            hidden_size: value("qwen35moe.embedding_length")? as usize,
            layer_count,
            mtp_layer_count: 1,
            full_attention_interval: interval,
            full_attention_layers: layer_count.div_ceil(interval),
            num_heads: value("qwen35moe.attention.head_count")? as usize,
            num_kv_heads: value("qwen35moe.attention.head_count_kv")? as usize,
            head_dim: key_length,
            rope_dim: value("qwen35moe.rope.dimension_count")? as usize,
            rope_theta: float("qwen35moe.rope.freq_base")?,
            rms_eps: float("qwen35moe.attention.layer_norm_rms_epsilon")?,
            use_qk_norm: true,
            linear_key_heads: value("qwen35moe.ssm.group_count")? as usize,
            linear_value_heads: inner_size / state_size,
            linear_key_head_dim: state_size,
            linear_value_head_dim: state_size,
            linear_conv_kernel: value("qwen35moe.ssm.conv_kernel")? as usize,
            expert_intermediate_size: value("qwen35moe.expert_feed_forward_length")? as usize,
            shared_intermediate_size: value("qwen35moe.expert_shared_feed_forward_length")? as usize,
            num_experts: value("qwen35moe.expert_count")? as usize,
            num_experts_per_tok: value("qwen35moe.expert_used_count")? as usize,
            num_shared_experts: 1,
            routed_scaling_factor: 1.0,
            eos_token_ids: vec![value("tokenizer.ggml.eos_token_id")? as u32],
        })
    }

    pub fn standard() -> Self {
        Self {
            vocab_size: 248_320,
            hidden_size: 2_048,
            layer_count: 40,
            mtp_layer_count: 1,
            full_attention_interval: 4,
            full_attention_layers: 10,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim: 256,
            rope_dim: 64,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            use_qk_norm: true,
            linear_key_heads: 16,
            linear_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel: 4,
            expert_intermediate_size: 512,
            shared_intermediate_size: 512,
            num_experts: 256,
            num_experts_per_tok: 8,
            num_shared_experts: 1,
            routed_scaling_factor: 1.0,
            eos_token_ids: vec![248_046],
        }
    }
}

pub struct Ornith {
    config: OrnithConfig,
    layer_specs: Vec<LayerSpec>,
}

impl Ornith {
    pub fn new(config: OrnithConfig) -> Result<Self, ModelError> {
        if config.layer_count == 0 {
            return Err(ModelError::InvalidArchitecture("layer_count 不能为 0".into()));
        }
        if config.num_heads == 0 || config.num_kv_heads == 0 || !config.num_heads.is_multiple_of(config.num_kv_heads) {
            return Err(ModelError::InvalidArchitecture("num_heads 与 num_kv_heads 不合法".into()));
        }
        if config.head_dim == 0 || config.hidden_size == 0 || config.rms_eps <= 0.0 {
            return Err(ModelError::InvalidArchitecture("hidden_size/head_dim/rms_eps 不合法".into()));
        }
        if config.full_attention_layers == 0 || config.full_attention_interval == 0 {
            return Err(ModelError::InvalidArchitecture("full attention 参数不能为 0".into()));
        }
        if config.num_experts_per_tok == 0 || config.num_experts_per_tok > config.num_experts {
            return Err(ModelError::InvalidArchitecture("expert top-k 超界".into()));
        }
        if config.rope_dim == 0 || config.rope_dim > config.head_dim {
            return Err(ModelError::InvalidArchitecture("rope_dim 必须位于 full-attention head_dim 内".into()));
        }
        config.gated_delta_net_spec().validate().map_err(ModelError::InvalidArchitecture)?;

        let layer_specs = (0..config.layer_count).map(|layer| Self::build_layer_spec(&config, layer)).collect();
        Ok(Self { config, layer_specs })
    }

    pub fn standard() -> Self {
        Self::new(OrnithConfig::standard()).expect("Ornith 标准配置必须有效")
    }

    fn is_full_attention_layer(cfg: &OrnithConfig, layer: LayerId) -> bool {
        cfg.ornith_layer_kind(layer) == Some(OrnithLayerKind::FullAttention)
    }

    fn build_layer_spec(cfg: &OrnithConfig, layer: LayerId) -> LayerSpec {
        let is_full_attention = Self::is_full_attention_layer(cfg, layer);
        let attention = if is_full_attention { AttentionSpec::Gqa(cfg.full_attention_spec()) } else { AttentionSpec::GatedDeltaNet(cfg.gated_delta_net_spec()) };

        let feedforward = FeedforwardSpec::TopkMoe(TopkMoeSpec {
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            num_shared_experts: cfg.num_shared_experts,
            scoring_func: ScoringFunc::Softmax,
            normalize_selected: true,
            routed_scaling_factor: cfg.routed_scaling_factor,
            intermediate_size: cfg.expert_intermediate_size,
            shared_intermediate_size: cfg.shared_intermediate_size,
            activation: Activation::Silu,
        });

        LayerSpec { attention, feedforward, input_norm: NormSpec::GemmaRms { eps: cfg.rms_eps }, post_attention_norm: NormSpec::GemmaRms { eps: cfg.rms_eps }, post_norm: None }
    }
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn standard_layer_schedule_matches_qwen35() {
        let cfg = OrnithConfig::standard();
        assert_eq!(cfg.ornith_layer_kind(0), Some(OrnithLayerKind::DeltaNet));
        assert_eq!(cfg.ornith_layer_kind(2), Some(OrnithLayerKind::DeltaNet));
        assert_eq!(cfg.ornith_layer_kind(3), Some(OrnithLayerKind::FullAttention));
        assert_eq!(cfg.ornith_layer_kind(39), Some(OrnithLayerKind::FullAttention));
        assert_eq!(cfg.ornith_layer_kind(40), None);
        assert_eq!(cfg.mtp_layer_count, 1);
    }

    #[test]
    fn every_layer_uses_softmax_moe() {
        let model = Ornith::standard();
        for layer in 0..model.layer_count() {
            assert!(matches!(model.layer_spec(layer).unwrap().feedforward, FeedforwardSpec::TopkMoe(_)));
        }
    }
}

impl Model for Ornith {
    type Config = OrnithConfig;

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

// ============================================================================
// OrnithGguf —— GGUF 权重 wrapper(模型知识在 runtime,格式能力委托 GgufReader)。
// impl GgufExpertSource/ExpertSourceProvider 为 MoE expert pipeline 提供数据源。
// ============================================================================

use crate::tokenizer::{Detokenizer, Tokenizer};
use crate::weight::container::gguf::{GgufMatrix, GgufReader};
use crate::weight::expert_source::{ExpertSource, GgufExpertSource, GgufExpertWeights};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrnithExpertProjection {
    Gate,
    Up,
    Down,
}

impl GgufExpertSource for OrnithGguf {
    fn intermediate(&self) -> usize {
        self.cfg.expert_intermediate_size
    }

    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }

    fn load_expert_gguf(&self, layer: usize, expert: usize) -> Result<GgufExpertWeights, String> {
        Ok(GgufExpertWeights {
            gate: self.routed_expert(layer, expert, OrnithExpertProjection::Gate)?,
            up: self.routed_expert(layer, expert, OrnithExpertProjection::Up)?,
            down: self.routed_expert(layer, expert, OrnithExpertProjection::Down)?,
        })
    }
}

impl ExpertSourceProvider for OrnithGguf {
    fn source(&self, layer: usize) -> Result<ExpertSource<'_>, String> {
        let layer_count = self.cfg.layer_count + usize::from(self.has_mtp());
        if layer >= layer_count {
            return Err(format!("Ornith expert source layer 越界: {layer} >= {layer_count}"));
        }
        Ok(ExpertSource::Gguf(self))
    }
}

#[derive(Debug, Default, Clone)]
pub struct OrnithInventoryClass {
    pub tensors: usize,
    pub bytes: u64,
    pub elements: u64,
}

impl OrnithInventoryClass {
    pub fn f16_bytes(&self) -> u64 {
        self.elements * std::mem::size_of::<half::f16>() as u64
    }
}

#[derive(Debug, Default, Clone)]
pub struct OrnithTensorInventory {
    pub embedding_and_head: OrnithInventoryClass,
    pub core: OrnithInventoryClass,
    pub routed_experts: OrnithInventoryClass,
    pub mtp: OrnithInventoryClass,
    pub other: OrnithInventoryClass,
    pub quant_types: std::collections::BTreeMap<String, OrnithInventoryClass>,
}

impl OrnithTensorInventory {
    pub fn total_bytes(&self) -> u64 {
        self.embedding_and_head.bytes + self.core.bytes + self.routed_experts.bytes + self.mtp.bytes + self.other.bytes
    }
}

pub struct OrnithGguf {
    reader: GgufReader,
    cfg: OrnithConfig,
}

impl OrnithGguf {
    pub fn open(path: &Path) -> Result<Self, String> {
        let reader = GgufReader::open(&GgufReader::locate(path)?)?;
        let cfg = OrnithConfig::from_gguf(&reader)?;
        let model = Self { reader, cfg };
        model.validate_metadata()?;
        model.validate_tensors()?;
        Ok(model)
    }

    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }
    pub fn config(&self) -> &OrnithConfig {
        &self.cfg
    }

    pub fn has_mtp(&self) -> bool {
        self.reader.has_mtp(self.cfg.layer_count)
    }
    pub fn matrix(&self, name: &str) -> Result<GgufMatrix, String> {
        self.reader.read_matrix(name)
    }
    pub fn gemma_norm_vector(&self, name: &str) -> Result<Vec<f32>, String> {
        self.reader.gemma_norm_vector(name)
    }
    pub fn a_log_vector(&self, name: &str) -> Result<Vec<f32>, String> {
        self.reader.a_log_vector(name)
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
        self.reader.bpe_tokenizer().map_err(|e| format!("Ornith tokenizer: {e}"))
    }
    pub fn detokenizer(&self) -> Result<Detokenizer, String> {
        self.reader.bpe_detokenizer().map_err(|e| format!("Ornith detokenizer: {e}"))
    }

    pub fn routed_expert(&self, layer: usize, expert: usize, projection: OrnithExpertProjection) -> Result<GgufMatrix, String> {
        let layer_count = self.cfg.layer_count + usize::from(self.has_mtp());
        if layer >= layer_count || expert >= self.cfg.num_experts {
            return Err(format!("Ornith expert 索引越界: layer={layer}/{layer_count}, expert={expert}/{}", self.cfg.num_experts));
        }
        let proj = match projection {
            OrnithExpertProjection::Gate => "gate",
            OrnithExpertProjection::Up => "up",
            OrnithExpertProjection::Down => "down",
        };
        self.reader.read_matrix_slice(&format!("blk.{layer}.ffn_{proj}_exps.weight"), expert)
    }

    pub fn inventory(&self) -> OrnithTensorInventory {
        let mut inv = OrnithTensorInventory::default();
        for tensor in self.reader.tensors() {
            let class = if tensor.name == "token_embd.weight" || tensor.name == "output.weight" || tensor.name == "output_norm.weight" {
                &mut inv.embedding_and_head
            } else if tensor.name.starts_with("blk.40.") {
                &mut inv.mtp
            } else if tensor.name.contains("_exps.weight") {
                &mut inv.routed_experts
            } else if tensor.name.starts_with("blk.") {
                &mut inv.core
            } else {
                &mut inv.other
            };
            class.tensors += 1;
            class.bytes += tensor.bytes as u64;
            class.elements += tensor.dims.iter().product::<usize>() as u64;
            let quant = inv.quant_types.entry(tensor.tensor_type.name().to_owned()).or_default();
            quant.tensors += 1;
            quant.bytes += tensor.bytes as u64;
            quant.elements += tensor.dims.iter().product::<usize>() as u64;
        }
        inv
    }

    fn validate_metadata(&self) -> Result<(), String> {
        self.reader.expect_metadata_str("general.architecture", "qwen35moe")?;
        self.reader.expect_metadata_u64("qwen35moe.block_count", self.cfg.layer_count as u64).or_else(|_| self.reader.expect_metadata_u64("qwen35moe.block_count", (self.cfg.layer_count + 1) as u64))?;
        Ok(())
    }

    fn validate_tensors(&self) -> Result<(), String> {
        Ok(())
    }
}
#[cfg(target_os = "macos")]
pub mod metal;
