//! Latent MoE 规格。

use crate::{
    backend::{Backend, BackendError, ExpertDecodeBackend, ExpertPrefillBackend},
    norm::NormSpec,
    weight::expert_source::ExpertSource,
};

use super::{
    prefill::{ExpertPrefillOutput, prefill_routed_experts},
    topk_moe::{RoutedMoeInputs, RoutedMoeWeightsRef, TopkMoeSpec, decode_routed_topk_moe},
};

/// 路由先基于原 hidden 计算，专家部分在投影后的 latent 空间执行。
/// 共享 MLP 独立读取原 hidden，因此不塞进 `TopkMoeSpec` 的 shared expert 布局。
#[derive(Debug)]
pub struct LatentTopkMoeSpec {
    pub routed: TopkMoeSpec,
    pub routed_hidden_size: usize,
    pub routed_output_norm: Option<NormSpec>,
    pub shared_intermediate_size: usize,
}

pub struct LatentMoeWeights<W> {
    pub router_weight: W,
    pub router_bias: W,
    /// 原 hidden → routed latent。
    pub routed_down_projection: W,
    /// routed experts 聚合后、升回 hidden 前的 norm。
    pub routed_norm: W,
    pub routed_up_projection: W,
    /// 一个合并宽度的 shared MLP，始终读取原 hidden。
    pub shared_gate: W,
    pub shared_up: W,
    pub shared_down: W,
}

fn validate(spec: &LatentTopkMoeSpec) -> Result<(), BackendError> {
    if spec.routed_hidden_size == 0 || spec.shared_intermediate_size == 0 {
        return Err(BackendError::Compute { msg: "LatentMoE routed/shared hidden 不能为 0".to_owned() });
    }
    if spec.routed.num_shared_experts != 0 || spec.routed.shared_intermediate_size != 0 {
        return Err(BackendError::Compute { msg: "LatentMoE shared MLP 必须位于外层，routed TopkMoeSpec 不能再包含 shared expert".to_owned() });
    }
    Ok(())
}

fn shared_mlp<B: Backend>(backend: &B, spec: &LatentTopkMoeSpec, weights: &LatentMoeWeights<B::Weight>, input: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let activated = backend.gated_linear(input, &weights.shared_gate, &weights.shared_up, &spec.routed.activation)?;
    backend.linear(&activated, &weights.shared_down)
}

fn finish_routed<B: Backend>(backend: &B, spec: &LatentTopkMoeSpec, weights: &LatentMoeWeights<B::Weight>, routed: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let routed = match spec.routed_output_norm {
        Some(NormSpec::Rms { eps }) => backend.rmsnorm(routed, &weights.routed_norm, eps)?,
        Some(NormSpec::GroupedRms { eps, groups }) => backend.grouped_rmsnorm(routed, &weights.routed_norm, eps, groups)?,
        Some(NormSpec::GemmaRms { eps }) => backend.gemma_rmsnorm(routed, &weights.routed_norm, eps)?,
        Some(NormSpec::AdaLn { .. }) => return Err(BackendError::Compute { msg: "AdaLN 不用于 latent MoE 路由输出".to_owned() }),
        None => return backend.linear(routed, &weights.routed_up_projection),
    };
    backend.linear(&routed, &weights.routed_up_projection)
}

pub fn merge_shared<B: Backend>(backend: &B, spec: &LatentTopkMoeSpec, weights: &LatentMoeWeights<B::Weight>, input: &B::Tensor, routed: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let routed = finish_routed(backend, spec, weights, routed)?;
    let shared = shared_mlp(backend, spec, weights, input)?;
    backend.add(&routed, &shared)
}

#[allow(clippy::too_many_arguments)]
pub fn prefill<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &LatentTopkMoeSpec,
    weights: &LatentMoeWeights<B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    input: &B::Tensor,
) -> Result<ExpertPrefillOutput<B::Tensor>, BackendError> {
    validate(spec)?;
    let expert_input = backend.linear(input, &weights.routed_down_projection)?;
    let routed_weights = super::topk_moe::RoutedMoeWeightsRef { router: &weights.router_weight, bias: &weights.router_bias, selected_experts: None };
    let routed_inputs = super::topk_moe::RoutedMoeInputs { route: input, expert: &expert_input };
    let routed = prefill_routed_experts(backend, &spec.routed, routed_weights, layer, experts, routed_inputs, None)?;
    let tensor = merge_shared(backend, spec, weights, input, &routed.tensor)?;
    Ok(ExpertPrefillOutput { tensor, routing: routed.routing })
}

#[allow(clippy::too_many_arguments)]
pub fn decode<'a, B, F>(backend: &B, spec: &LatentTopkMoeSpec, weights: &LatentMoeWeights<B::Weight>, layer: usize, source: ExpertSource<'_>, state: &mut B::MoeState, input: &B::Tensor, on_routed: F) -> Result<B::Tensor, BackendError>
where
    B: ExpertDecodeBackend,
    F: FnOnce(&[u16], &mut B::MoeState) -> Result<Option<crate::backend::ExpertPrefetchRequest<'a>>, BackendError>,
{
    validate(spec)?;
    let expert_input = backend.linear(input, &weights.routed_down_projection)?;
    let routed_weights = RoutedMoeWeightsRef { router: &weights.router_weight, bias: &weights.router_bias, selected_experts: None };
    let inputs = RoutedMoeInputs { route: input, expert: &expert_input };
    // LatentMoE 的 shared MLP 由 merge_shared 无条件执行，不依赖回调触发标志。
    let (routed, _) = decode_routed_topk_moe(backend, &spec.routed, routed_weights, layer, source, state, inputs, on_routed)?;
    merge_shared(backend, spec, weights, input, &routed)
}
