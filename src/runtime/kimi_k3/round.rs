//! K3 单层权重装配与 prefill/decode 分派。

use crate::{
    attention::AttentionSpec,
    attention::{
        attn_res::{AttnResBackend, AttnResState},
        kda::{KdaKernel, KdaState},
    },
    backend::{Backend, BackendError, DecodeBackend, ExpertDecodeBackend, ExpertPrefillBackend, MlaPrefillBackend},
    moe::{
        DenseFfn, FeedforwardSpec,
        latent_moe::{self, LatentMoeWeights},
    },
    runtime::LayerSpec,
    weight::{expert_source::ExpertSourceProvider, model::kimi_k3::KimiK3Weights},
};

use super::{
    layer::{GatedMlaLayerWeights, KdaLayerWeights, gated_mla_decode_layer, gated_mla_prefill_layer, kda_layer},
    prepare::{prepare_dense_mlp, prepare_gated_mla_layer, prepare_kda_layer, prepare_latent_moe},
};

pub enum PreparedAttention<W> {
    Kda(KdaLayerWeights<W>),
    GatedMla(GatedMlaLayerWeights<W>),
}

pub enum PreparedFeedforward<W> {
    Dense(DenseFfn<W>),
    LatentMoe(LatentMoeWeights<W>),
}

pub struct KimiK3PreparedLayer<W> {
    pub attention: PreparedAttention<W>,
    pub feedforward: PreparedFeedforward<W>,
}

pub fn load_prepare_layer<B: Backend>(backend: &B, source: &KimiK3Weights, layer: usize, spec: &LayerSpec) -> Result<KimiK3PreparedLayer<B::Weight>, BackendError> {
    let common = source.load_layer_common(layer).map_err(|msg| load_error(layer, "common", msg))?;
    let attention = match &spec.attention {
        AttentionSpec::Kda(_) => {
            let weights = source.load_kda(layer).map_err(|msg| load_error(layer, "KDA", msg))?;
            PreparedAttention::Kda(prepare_kda_layer(backend, &common, &weights)?)
        }
        AttentionSpec::GatedMla(_) => {
            let weights = source.load_gated_mla(layer).map_err(|msg| load_error(layer, "Gated MLA", msg))?;
            PreparedAttention::GatedMla(prepare_gated_mla_layer(backend, &common, &weights)?)
        }
        other => {
            return Err(BackendError::Compute { msg: format!("Kimi K3 L{layer} 不支持 attention {other:?}") });
        }
    };
    let feedforward = match &spec.feedforward {
        FeedforwardSpec::Dense(_) => {
            let weights = source.load_dense_mlp(layer).map_err(|msg| load_error(layer, "dense MLP", msg))?;
            PreparedFeedforward::Dense(prepare_dense_mlp(backend, &weights)?)
        }
        FeedforwardSpec::LatentTopkMoe(_) => {
            let weights = source.load_latent_moe(layer).map_err(|msg| load_error(layer, "Latent-MoE", msg))?;
            PreparedFeedforward::LatentMoe(prepare_latent_moe(backend, &weights)?)
        }
        other => {
            return Err(BackendError::Compute { msg: format!("Kimi K3 L{layer} 不支持 FFN {other:?}") });
        }
    };
    Ok(KimiK3PreparedLayer { attention, feedforward })
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_layer<B>(
    backend: &B,
    kda_state: &mut KdaState<B::KdaStorage>,
    cache: Option<&mut B::Cache>,
    attn_res_state: &mut AttnResState<B::Tensor>,
    layer: usize,
    position: usize,
    hidden: B::Tensor,
    weights: &KimiK3PreparedLayer<B::Weight>,
    spec: &LayerSpec,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
    experts: &mut B::PrefillExperts,
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + MlaPrefillBackend + AttnResBackend + ExpertPrefillBackend,
{
    match (&weights.attention, &spec.attention) {
        (PreparedAttention::Kda(attention_weights), AttentionSpec::Kda(attention)) => {
            kda_layer(backend, kda_state, attn_res_state, layer, position, hidden, attention_weights, attention, rms_eps, |input| prefill_feedforward(backend, layer, input, &weights.feedforward, spec, experts))
        }
        (PreparedAttention::GatedMla(attention_weights), AttentionSpec::GatedMla(attention)) => {
            gated_mla_prefill_layer(backend, cache, attn_res_state, layer, position, hidden, attention_weights, attention, rms_eps, cos, sin, |input| prefill_feedforward(backend, layer, input, &weights.feedforward, spec, experts))
        }
        _ => Err(BackendError::Compute { msg: format!("Kimi K3 L{layer} prepared attention 与 LayerSpec 不一致") }),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_layer<B, S>(
    backend: &B,
    kda_state: &mut KdaState<B::KdaStorage>,
    cache: &mut B::Cache,
    attn_res_state: &mut AttnResState<B::Tensor>,
    moe_state: &mut B::MoeState,
    expert_source: &S,
    layer: usize,
    position: usize,
    hidden: B::Tensor,
    weights: &KimiK3PreparedLayer<B::Weight>,
    spec: &LayerSpec,
    rms_eps: f32,
    cos: &[f32],
    sin: &[f32],
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + DecodeBackend + AttnResBackend + ExpertDecodeBackend,
    S: ExpertSourceProvider,
{
    match (&weights.attention, &spec.attention) {
        (PreparedAttention::Kda(attention_weights), AttentionSpec::Kda(attention)) => {
            kda_layer(backend, kda_state, attn_res_state, layer, position, hidden, attention_weights, attention, rms_eps, |input| decode_feedforward(backend, layer, input, &weights.feedforward, spec, expert_source, moe_state))
        }
        (PreparedAttention::GatedMla(attention_weights), AttentionSpec::GatedMla(attention)) => gated_mla_decode_layer(backend, cache, attn_res_state, layer, position, hidden, attention_weights, attention, rms_eps, cos, sin, |input| {
            decode_feedforward(backend, layer, input, &weights.feedforward, spec, expert_source, moe_state)
        }),
        _ => Err(BackendError::Compute { msg: format!("Kimi K3 L{layer} prepared attention 与 LayerSpec 不一致") }),
    }
}

fn prefill_feedforward<B: ExpertPrefillBackend>(backend: &B, layer: usize, input: &B::Tensor, weights: &PreparedFeedforward<B::Weight>, spec: &LayerSpec, experts: &mut B::PrefillExperts) -> Result<B::Tensor, BackendError> {
    match (weights, &spec.feedforward) {
        (PreparedFeedforward::Dense(weights), FeedforwardSpec::Dense(spec)) => dense(backend, input, weights, &spec.activation),
        (PreparedFeedforward::LatentMoe(weights), FeedforwardSpec::LatentTopkMoe(spec)) => Ok(latent_moe::prefill(backend, spec, weights, layer, experts, input)?.tensor),
        _ => Err(BackendError::Compute { msg: format!("Kimi K3 L{layer} prepared FFN 与 LayerSpec 不一致") }),
    }
}

fn decode_feedforward<B, S>(backend: &B, layer: usize, input: &B::Tensor, weights: &PreparedFeedforward<B::Weight>, spec: &LayerSpec, source: &S, state: &mut B::MoeState) -> Result<B::Tensor, BackendError>
where
    B: ExpertDecodeBackend,
    S: ExpertSourceProvider,
{
    match (weights, &spec.feedforward) {
        (PreparedFeedforward::Dense(weights), FeedforwardSpec::Dense(spec)) => dense(backend, input, weights, &spec.activation),
        (PreparedFeedforward::LatentMoe(weights), FeedforwardSpec::LatentTopkMoe(spec)) => {
            let source = source.source(layer).map_err(BackendError::ExpertLoad)?;
            latent_moe::decode(backend, spec, weights, layer, source, state, input, |_, _| Ok(None))
        }
        _ => Err(BackendError::Compute { msg: format!("Kimi K3 L{layer} prepared FFN 与 LayerSpec 不一致") }),
    }
}

fn dense<B: Backend>(backend: &B, input: &B::Tensor, weights: &DenseFfn<B::Weight>, activation: &crate::moe::Activation) -> Result<B::Tensor, BackendError> {
    let activated = backend.gated_linear(input, &weights.gate, &weights.up, activation)?;
    backend.linear(&activated, &weights.down)
}

fn load_error(layer: usize, part: &str, msg: String) -> BackendError {
    BackendError::Compute { msg: format!("Kimi K3 L{layer} 加载 {part}: {msg}") }
}
