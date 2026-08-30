//! Top-k 路由 MoE 规格。

/// 路由评分函数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringFunc {
    /// Softmax 路由；归一化范围由 `TopkMoeSpec::normalize_selected` 决定。
    Softmax,
    /// Sigmoid + e_score_correction_bias(DeepSeek-V3 noaux_tc 风格,GLM-5.2、MiniMax-M3)。
    SigmoidBias,
    /// sqrt(softplus(logit)) + correction bias，DeepSeek-V4 score-based 层使用。
    SqrtSoftplusBias,
}

/// Top-k 路由 MoE 规格。
#[derive(Debug)]
pub struct TopkMoeSpec {
    /// 路由专家总数。
    pub num_experts: usize,
    /// 每 token 激活的专家数。
    pub top_k: usize,
    /// 共享专家数(总是激活,不参与路由)。0 表示无。
    pub num_shared_experts: usize,
    /// 路由评分函数。
    pub scoring_func: ScoringFunc,
    /// 是否在 Top-K 选择后对选中专家权重重新归一化。
    /// `false` 表示保留全部专家评分函数产生的概率质量。
    pub normalize_selected: bool,
    /// 路由后缩放系数(GLM-5.2 = 2.5, MiniMax-M3 = 2.0)。
    pub routed_scaling_factor: f32,
    /// 每(路由)专家 FFN 的中间维度。
    pub intermediate_size: usize,
    /// 共享专家 FFN 的中间维度(无共享专家时忽略)。
    pub shared_intermediate_size: usize,
    /// 激活函数。
    pub activation: super::dense_mlp::Activation,
}

/// 共享专家权重视图。
pub struct SharedExpertRef<'a, W: ?Sized> {
    pub gate: &'a W,
    pub up: &'a W,
    pub down: &'a W,
    /// Qwen3.5 风格 shared expert 输出门；`None` 表示直接合并。
    pub output_gate: Option<&'a W>,
}

/// Top-k MoE 权重视图；专家 archive 由 Backend 平台能力管理。
pub struct MoeFfnRef<'a, W: ?Sized> {
    pub router_weight: &'a W,
    pub router_bias: &'a W,
    pub shared_experts: &'a [SharedExpertRef<'a, W>],
    /// 预先选定的专家集合（每 token `top_k` 个）。`None` 走常规 score 路由；
    /// `Some` 走固定专家路由（router weight 只算权重，不改专家集合），
    /// 适用于 hash 路由等预先决定专家的场景（DeepSeek-V4 前若干层）。
    pub selected_experts: Option<&'a [u32]>,
}

#[derive(Clone, Copy)]
pub struct RoutedMoeWeightsRef<'a, W: ?Sized> {
    pub router: &'a W,
    pub bias: &'a W,
    pub selected_experts: Option<&'a [u32]>,
}

#[derive(Clone, Copy)]
pub struct RoutedMoeInputs<'a, T: ?Sized> {
    pub route: &'a T,
    pub expert: &'a T,
}

use crate::{
    backend::{Backend, BackendError, ExpertDecodeBackend},
    weight::expert_source::ExpertSource,
};

/// Shared expert 累加循环的公共实现；观测闭包与 `dense_mlp::forward_observed`
/// 同约定——生产路径传 no-op，量化/诊断路径观测每个 shared expert 的 down 输入。
pub fn shared_experts_observed<B, O>(backend: &B, spec: &TopkMoeSpec, shared_experts: &[SharedExpertRef<'_, B::Weight>], input: &B::Tensor, mut observe_down_input: O) -> Result<Option<B::Tensor>, BackendError>
where
    B: Backend,
    O: FnMut(&B::Tensor),
{
    let mut output = None;
    for shared in shared_experts {
        let activated = backend.gated_linear(input, shared.gate, shared.up, &spec.activation)?;
        observe_down_input(&activated);
        let down = backend.linear(&activated, shared.down)?;
        let down = match shared.output_gate {
            Some(weight) => backend.linear_sigmoid_gate(input, weight, &down)?,
            None => down,
        };
        output = Some(match output {
            Some(accumulator) => backend.add(&accumulator, &down)?,
            None => down,
        });
    }
    Ok(output)
}

/// Shared expert 的平台无关 decode 数据流。矩阵与激活由 backend capability 实现。
pub fn decode_shared_experts<B: Backend>(backend: &B, spec: &TopkMoeSpec, weights: &MoeFfnRef<'_, B::Weight>, input: &B::Tensor) -> Result<Option<B::Tensor>, BackendError> {
    let mut output = None;
    for shared in weights.shared_experts {
        let down = backend.gated_mlp(input, shared.gate, shared.up, shared.down, &spec.activation)?;
        let down = match shared.output_gate {
            Some(weight) => backend.linear_sigmoid_gate(input, weight, &down)?,
            None => down,
        };
        output = Some(match output {
            Some(accumulator) => backend.add(&accumulator, &down)?,
            None => down,
        });
    }
    Ok(output)
}

/// Top-k routed-only decode。`route_input` 决定专家，`expert_input` 进入专家矩阵。
/// 普通 MoE 两者相同，LatentMoE 分别使用原 hidden 与低维 latent。
///
/// `selected_experts` 为 `Some` 时走固定专家路由（hash 路由等预先决定专家的场景），
/// router weight 只算权重；为 `None` 走常规 score 路由。
///
/// 返回的 `bool` 标记 `on_routed` 是否被 backend 真实触发；调用方依赖该回调
/// 执行 shared experts 等副作用时，必须用标志区分"未触发"与"已触发"，
/// 不能把 backend 的静默跳过当成成功。
#[allow(clippy::too_many_arguments)]
pub fn decode_routed_topk_moe<'a, B, F>(
    backend: &B,
    spec: &TopkMoeSpec,
    weights: RoutedMoeWeightsRef<'_, B::Weight>,
    layer: usize,
    source: ExpertSource<'_>,
    state: &mut B::MoeState,
    inputs: RoutedMoeInputs<'_, B::Tensor>,
    on_routed: F,
) -> Result<(B::Tensor, bool), BackendError>
where
    B: ExpertDecodeBackend,
    F: FnOnce(&[u16], &mut B::MoeState) -> Result<Option<crate::backend::ExpertPrefetchRequest<'a>>, BackendError>,
{
    if backend.token_rows(inputs.route) != backend.token_rows(inputs.expert) {
        return Err(BackendError::Compute { msg: "MoE route input 与 expert input 行数不一致".to_owned() });
    }
    let resident_weights = RoutedMoeWeightsRef { router: weights.router, bias: weights.bias, selected_experts: weights.selected_experts };
    let resident_inputs = RoutedMoeInputs { route: inputs.route, expert: inputs.expert };
    if let Some((output, active)) = backend.decode_resident_routed_experts(spec, resident_weights, layer, source, state, resident_inputs)? {
        // `None` 表示 route 有意保持设备常驻；此时全量 resident backend 不需要预测反馈。
        if let Some(request) = on_routed(active.as_deref().unwrap_or(&[]), state)? {
            backend.prefetch_experts(spec, state, request)?;
        }
        return Ok((output, true));
    }
    let (routing, backend_routing) = match weights.selected_experts {
        None => backend.decode_route(inputs.route, weights.router, weights.bias, spec)?,
        Some(selected) => backend.decode_route_selected(inputs.route, weights.router, selected, spec)?,
    };
    let assignments = routing.grouped(spec.num_experts).map_err(|msg| BackendError::Compute { msg })?;
    let active = super::routing::active_experts(&assignments).map_err(|msg| BackendError::Compute { msg })?;
    let mut on_routed_fired = false;
    let output = backend.decode_routed_experts(spec, layer, source, state, inputs.expert, &assignments, &backend_routing, |state| {
        on_routed_fired = true;
        on_routed(&active, state)
    })?;
    Ok((output, on_routed_fired))
}
