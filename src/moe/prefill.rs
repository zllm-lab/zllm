//! 平台无关的 MoE prefill 编排。

use crate::{
    backend::{BackendError, ExpertPrefillBackend, ExpertPrefillBatch, MoePrefillBackend, MoePrefillRouting},
    moe::topk_moe::{MoeFfnRef, RoutedMoeInputs, RoutedMoeWeightsRef, TopkMoeSpec},
};

pub struct ExpertPrefillOutput<T> {
    pub tensor: T,
    pub routing: MoePrefillRouting,
}

pub fn prefill_experts<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: &MoeFfnRef<'_, B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    input: &B::Tensor,
    expert_batch_size: Option<usize>,
) -> Result<ExpertPrefillOutput<B::Tensor>, BackendError> {
    prefill_experts_observed(backend, spec, weights, layer, experts, input, expert_batch_size, |_| {})
}

/// 不消费 host routing 的生产路径；backend 可将 route 与 expert 计算完整留在设备端。
pub fn prefill_experts_untraced<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: &MoeFfnRef<'_, B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    input: &B::Tensor,
    expert_batch_size: Option<usize>,
) -> Result<B::Tensor, BackendError> {
    prefill_experts_inner(backend, spec, weights, layer, experts, input, input, expert_batch_size, true, None, |_| {}).map(|(tensor, _)| tensor)
}

/// 路由保留精确 activation，shared/routed expert 复用同一份量化 activation。
pub fn prefill_experts_inputs<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: &MoeFfnRef<'_, B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    inputs: RoutedMoeInputs<'_, B::Tensor>,
    expert_batch_size: Option<usize>,
) -> Result<ExpertPrefillOutput<B::Tensor>, BackendError> {
    let (tensor, routing) = prefill_experts_inner(backend, spec, weights, layer, experts, inputs.route, inputs.expert, expert_batch_size, false, None, |_| {})?;
    let routing = routing.ok_or_else(|| BackendError::Compute { msg: "需要路由观测时 backend 未返回 host routing".to_owned() })?;
    Ok(ExpertPrefillOutput { tensor, routing })
}

pub fn prefill_experts_inputs_untraced<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: &MoeFfnRef<'_, B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    inputs: RoutedMoeInputs<'_, B::Tensor>,
    expert_batch_size: Option<usize>,
) -> Result<B::Tensor, BackendError> {
    prefill_experts_inner(backend, spec, weights, layer, experts, inputs.route, inputs.expert, expert_batch_size, true, None, |_| {}).map(|(tensor, _)| tensor)
}

pub fn prefill_experts_add_residual_untraced<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: &MoeFfnRef<'_, B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    input: &B::Tensor,
    residual: &B::Tensor,
    expert_batch_size: Option<usize>,
) -> Result<B::Tensor, BackendError> {
    prefill_experts_inner(backend, spec, weights, layer, experts, input, input, expert_batch_size, true, Some(residual), |_| {}).map(|(tensor, _)| tensor)
}

pub fn prefill_experts_inputs_add_residual_untraced<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: &MoeFfnRef<'_, B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    inputs: RoutedMoeInputs<'_, B::Tensor>,
    residual: &B::Tensor,
    expert_batch_size: Option<usize>,
) -> Result<B::Tensor, BackendError> {
    prefill_experts_inner(backend, spec, weights, layer, experts, inputs.route, inputs.expert, expert_batch_size, true, Some(residual), |_| {}).map(|(tensor, _)| tensor)
}

/// 观测 shared expert 的 down 输入；routed expert 调度与生产路径完全复用。
#[allow(clippy::too_many_arguments)]
pub fn prefill_experts_observed<B, O>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: &MoeFfnRef<'_, B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    input: &B::Tensor,
    expert_batch_size: Option<usize>,
    observe_shared_down_input: O,
) -> Result<ExpertPrefillOutput<B::Tensor>, BackendError>
where
    B: ExpertPrefillBackend,
    O: FnMut(&B::Tensor),
{
    let (tensor, routing) = prefill_experts_inner(backend, spec, weights, layer, experts, input, input, expert_batch_size, false, None, observe_shared_down_input)?;
    let routing = routing.ok_or_else(|| BackendError::Compute { msg: "需要路由观测时 backend 未返回 host routing".to_owned() })?;
    Ok(ExpertPrefillOutput { tensor, routing })
}

// 融合 epilogue 是可选能力；不支持时仍应保留 resident route，而不是退回 host routing。
fn resident_route_with_residual<T, E>(fused: Option<T>, shared: Option<&T>, residual: &T, fallback: impl FnOnce() -> Result<Option<T>, E>, mut add: impl FnMut(&T, &T) -> Result<T, E>) -> Result<Option<T>, E> {
    if fused.is_some() {
        return Ok(fused);
    }
    let Some(routed) = fallback()? else {
        return Ok(None);
    };
    let output = match shared {
        Some(shared) => add(shared, &routed)?,
        None => routed,
    };
    add(residual, &output).map(Some)
}

#[allow(clippy::too_many_arguments)]
fn prefill_experts_inner<B, O>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: &MoeFfnRef<'_, B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    route_input: &B::Tensor,
    expert_input: &B::Tensor,
    expert_batch_size: Option<usize>,
    allow_resident_route: bool,
    residual: Option<&B::Tensor>,
    observe_shared_down_input: O,
) -> Result<(B::Tensor, Option<MoePrefillRouting>), BackendError>
where
    B: ExpertPrefillBackend,
    O: FnMut(&B::Tensor),
{
    if backend.token_rows(route_input) != backend.token_rows(expert_input) {
        return Err(BackendError::Compute { msg: "MoE prefill route input 与 expert input 行数不一致".to_owned() });
    }
    let routed_weights = || RoutedMoeWeightsRef { router: weights.router_weight, bias: weights.router_bias, selected_experts: weights.selected_experts };
    let routed_inputs = || RoutedMoeInputs { route: route_input, expert: expert_input };
    if allow_resident_route
        && let Some(residual) = residual
        && let Some(fused) = backend.prefill_resident_moe_add(spec, routed_weights(), weights.shared_experts, layer, experts, routed_inputs(), residual)?
    {
        return Ok((fused, None));
    }
    backend.profile_device_operator("moe_shared")?;
    let shared_output = super::topk_moe::shared_experts_observed(backend, spec, weights.shared_experts, expert_input, observe_shared_down_input)?;
    backend.profile_device_operator("moe_routed")?;

    if allow_resident_route {
        let resident = match residual {
            Some(residual) => {
                let fused = backend.prefill_resident_routed_experts_add(spec, routed_weights(), layer, experts, routed_inputs(), shared_output.as_ref(), residual)?;
                resident_route_with_residual(fused, shared_output.as_ref(), residual, || backend.prefill_resident_routed_experts(spec, routed_weights(), layer, experts, routed_inputs()), |left, right| backend.add(left, right))?
            }
            None => {
                if let Some(shared) = shared_output.as_ref()
                    && let Some(fused) = backend.prefill_resident_routed_experts_add_shared(spec, routed_weights(), layer, experts, routed_inputs(), shared)?
                {
                    return Ok((fused, None));
                }
                backend.prefill_resident_routed_experts(spec, routed_weights(), layer, experts, routed_inputs())?
            }
        };
        if let Some(routed) = resident {
            backend.profile_device_operator("moe_epilogue")?;
            if residual.is_some() {
                return Ok((routed, None));
            }
            let tensor = match shared_output {
                Some(shared) => backend.add(&shared, &routed)?,
                None => routed,
            };
            return Ok((tensor, None));
        }
    }

    let routing = match weights.selected_experts {
        None => route(backend, spec, weights.router_weight, weights.router_bias, route_input)?,
        Some(selected) => route_selected(backend, spec, weights.router_weight, selected, route_input)?,
    };
    backend.profile_device_operator("moe_experts")?;
    let routed = execute_routed(backend, spec, layer, experts, expert_input, &routing, expert_batch_size)?;
    backend.profile_device_operator("moe_epilogue")?;
    let tensor = match shared_output {
        Some(shared) => backend.add(&shared, &routed)?,
        None => routed,
    };
    let tensor = match residual {
        Some(residual) => backend.add(residual, &tensor)?,
        None => tensor,
    };

    Ok((tensor, Some(routing)))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::resident_route_with_residual;

    #[test]
    fn residual融合不支持时回退resident_route() {
        let fallback_calls = Cell::new(0);
        let output = resident_route_with_residual(
            None,
            Some(&3),
            &10,
            || {
                fallback_calls.set(fallback_calls.get() + 1);
                Ok::<_, ()>(Some(4))
            },
            |left, right| Ok(left + right),
        )
        .unwrap();

        assert_eq!(output, Some(17));
        assert_eq!(fallback_calls.get(), 1);
    }

    #[test]
    fn residual融合命中时不重复执行resident_route() {
        let fallback_calls = Cell::new(0);
        let output = resident_route_with_residual(
            Some(17),
            Some(&3),
            &10,
            || {
                fallback_calls.set(fallback_calls.get() + 1);
                Ok::<_, ()>(Some(4))
            },
            |left, right| Ok(left + right),
        )
        .unwrap();

        assert_eq!(output, Some(17));
        assert_eq!(fallback_calls.get(), 0);
    }
}

/// routed-only prefill；允许 router 读取原 hidden，而专家读取独立 latent tensor。
pub fn prefill_routed_experts<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: RoutedMoeWeightsRef<'_, B::Weight>,
    layer: usize,
    experts: &mut B::PrefillExperts,
    inputs: RoutedMoeInputs<'_, B::Tensor>,
    expert_batch_size: Option<usize>,
) -> Result<ExpertPrefillOutput<B::Tensor>, BackendError> {
    if backend.token_rows(inputs.route) != backend.token_rows(inputs.expert) {
        return Err(BackendError::Compute { msg: "MoE prefill route input 与 expert input 行数不一致".to_owned() });
    }
    let routing = match weights.selected_experts {
        None => route(backend, spec, weights.router, weights.bias, inputs.route)?,
        Some(selected) => route_selected(backend, spec, weights.router, selected, inputs.route)?,
    };
    let tensor = execute_routed(backend, spec, layer, experts, inputs.expert, &routing, expert_batch_size)?;
    Ok(ExpertPrefillOutput { tensor, routing })
}

fn route<B: MoePrefillBackend>(backend: &B, spec: &TopkMoeSpec, router_weight: &B::Weight, router_bias: &B::Weight, input: &B::Tensor) -> Result<MoePrefillRouting, BackendError> {
    backend.moe_route(input, router_weight, router_bias, spec)
}

/// 固定专家路由：专家集合由调用方预先决定（如 hash 路由），router weight 只算权重。
fn route_selected<B: MoePrefillBackend>(backend: &B, spec: &TopkMoeSpec, router_weight: &B::Weight, selected_experts: &[u32], input: &B::Tensor) -> Result<MoePrefillRouting, BackendError> {
    backend.moe_route_selected(input, router_weight, selected_experts, spec)
}

fn execute_routed<B: ExpertPrefillBackend>(
    backend: &B,
    spec: &TopkMoeSpec,
    layer: usize,
    experts: &mut B::PrefillExperts,
    input: &B::Tensor,
    routing: &MoePrefillRouting,
    expert_batch_size: Option<usize>,
) -> Result<B::Tensor, BackendError> {
    let assignments = routing.grouped(spec.num_experts).map_err(|msg| BackendError::Compute { msg })?;
    match backend.prefill_routed_experts(spec, layer, experts, input, &assignments)? {
        Some(routed) => Ok(routed),
        None => {
            let mut active: Vec<_> = assignments.iter().enumerate().filter_map(|(expert, assigned)| (!assigned.is_empty()).then_some(expert)).collect();
            active.sort_unstable_by(|&left, &right| assignments[right].len().cmp(&assignments[left].len()).then_with(|| left.cmp(&right)));
            let batch_size = expert_batch_size.filter(|&size| size > 0).unwrap_or_else(|| active.len().max(1));
            let mut output = backend.moe_zeros(routing.rows, backend.token_cols(input))?;

            for chunk in active.chunks(batch_size) {
                let mut rows: Vec<Vec<u32>> = Vec::with_capacity(chunk.len());
                let mut weights: Vec<Vec<f32>> = Vec::with_capacity(chunk.len());
                for &expert in chunk {
                    rows.push(assignments[expert].iter().map(|&(row, _)| row).collect());
                    weights.push(assignments[expert].iter().map(|&(_, weight)| weight).collect());
                }
                let gathered = backend.moe_gather_rows_batch(input, &rows)?;
                let batch = chunk.iter().copied().zip(gathered).map(|(expert, input)| ExpertPrefillBatch { expert, input }).collect();
                let computed = backend.prefill_expert_batch(spec, layer, experts, batch)?;
                if computed.len() != rows.len() {
                    return Err(BackendError::Compute { msg: format!("L{layer} expert batch 返回 {} 项，期望 {} 项", computed.len(), rows.len()) });
                }
                backend.moe_scatter_add_rows_batch(&mut output, &computed, &rows, &weights)?;
            }
            backend.moe_finish(output)
        }
    }
}
