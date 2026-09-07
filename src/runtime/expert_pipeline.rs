//! 后端无关的专家预测、预取与真实路由反馈流水线。

use crate::{
    backend::{BackendError, ExpertDecodeBackend, ExpertPrefetchRequest},
    moe::{
        expert_predictor::{ExpertPredictor, ExpertPredictorConfig, ExpertRouteTrace},
        topk_moe::{MoeFfnRef, RoutedMoeInputs, RoutedMoeWeightsRef, TopkMoeSpec, decode_routed_topk_moe, decode_shared_experts},
    },
    weight::expert_source::ExpertSource,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct ExpertPipelineStats {
    pub prefetch_requested: usize,
}

pub struct ExpertDecodePipeline<S> {
    backend_state: S,
    predictor: ExpertPredictor,
    first_layer: usize,
    position: Option<usize>,
    stats: ExpertPipelineStats,
}

/// 一次 routed expert 执行及其下一层预取边界。
#[derive(Clone, Copy)]
pub struct ExpertDecodeRequest<'a> {
    pub layer: usize,
    pub source: ExpertSource<'a>,
    pub position: usize,
    pub next: Option<(usize, ExpertSource<'a>)>,
}

fn predict_prefetch_request<'a>(predictor: &mut ExpertPredictor, stats: &mut ExpertPipelineStats, spec: &TopkMoeSpec, layer: usize, source: ExpertSource<'a>) -> Result<Option<ExpertPrefetchRequest<'a>>, BackendError> {
    let Some(prediction) = predictor.predict(layer, None).map_err(BackendError::ExpertLoad)? else {
        return Ok(None);
    };
    let mut candidates: Vec<(usize, f32)> = prediction.experts.into_iter().zip(prediction.priorities).filter_map(|(expert, priority)| (priority > 0.0).then_some((usize::from(expert), priority))).collect();
    candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    candidates.truncate(spec.top_k);
    let mut predicted: Vec<usize> = candidates.into_iter().map(|(expert, _)| expert).collect();
    predicted.sort_unstable();
    predicted.dedup();
    stats.prefetch_requested += predicted.len();
    Ok(Some(ExpertPrefetchRequest { layer, source, experts: predicted }))
}

fn start_prefetch_with_state<B, S>(predictor: &mut ExpertPredictor, stats: &mut ExpertPipelineStats, backend: &B, spec: &TopkMoeSpec, layer: usize, source: ExpertSource<'_>, state: &mut S) -> Result<(), BackendError>
where
    B: ExpertDecodeBackend<MoeState = S>,
{
    let Some(request) = predict_prefetch_request(predictor, stats, spec, layer, source)? else {
        return Ok(());
    };
    backend.prefetch_experts(spec, state, request)?;
    Ok(())
}

impl<S> ExpertDecodePipeline<S> {
    pub fn new(backend_state: S, predictor: ExpertPredictorConfig) -> Result<Self, String> {
        let first_layer = predictor.first_layer;
        Ok(Self { backend_state, predictor: ExpertPredictor::new(predictor)?, first_layer, position: None, stats: ExpertPipelineStats::default() })
    }

    pub fn backend_state(&self) -> &S {
        &self.backend_state
    }

    pub fn backend_state_mut(&mut self) -> &mut S {
        &mut self.backend_state
    }

    pub fn into_backend_state(self) -> S {
        self.backend_state
    }

    pub fn stats(&self) -> ExpertPipelineStats {
        self.stats
    }

    pub fn take_stats(&mut self) -> ExpertPipelineStats {
        std::mem::take(&mut self.stats)
    }

    pub fn train_prefill_routes(&mut self, routes: &ExpertRouteTrace) -> Result<(), String> {
        if self.position.is_some() {
            return Err("decode 进行中不能加载 prefill route trace".to_owned());
        }
        routes.replay(&mut self.predictor)
    }

    fn start_prefetch<B>(&mut self, backend: &B, spec: &TopkMoeSpec, layer: usize, source: ExpertSource<'_>) -> Result<(), BackendError>
    where
        B: ExpertDecodeBackend<MoeState = S>,
    {
        start_prefetch_with_state(&mut self.predictor, &mut self.stats, backend, spec, layer, source, &mut self.backend_state)
    }

    /// 请求级重置:position 边界回到 None(新 token 流从头开始)。
    pub fn reset_token_boundary(&mut self) {
        self.position = None;
    }

    pub fn decode<B>(&mut self, backend: &B, spec: &TopkMoeSpec, weights: &MoeFfnRef<'_, B::Weight>, request: ExpertDecodeRequest<'_>, input: &B::Tensor) -> Result<B::Tensor, BackendError>
    where
        B: ExpertDecodeBackend<MoeState = S>,
    {
        self.decode_inputs(backend, spec, weights, request, RoutedMoeInputs { route: input, expert: input })
    }

    /// router 保持精确输入，专家矩阵可消费同一次归一化产生的低精度输入。
    pub fn decode_inputs<B>(&mut self, backend: &B, spec: &TopkMoeSpec, weights: &MoeFfnRef<'_, B::Weight>, request: ExpertDecodeRequest<'_>, inputs: RoutedMoeInputs<'_, B::Tensor>) -> Result<B::Tensor, BackendError>
    where
        B: ExpertDecodeBackend<MoeState = S>,
    {
        let mut shared = None;
        let expert_input = inputs.expert;
        let routed_weights = RoutedMoeWeightsRef { router: weights.router_weight, bias: weights.router_bias, selected_experts: weights.selected_experts };
        let routed = self.decode_routed_with_ready(backend, spec, routed_weights, request, inputs, || {
            shared = decode_shared_experts(backend, spec, weights, expert_input)?;
            if shared.is_some() {
                backend.submit_batch();
            }
            Ok(())
        })?;
        match shared {
            Some(shared) => backend.add(&routed, &shared),
            None => Ok(routed),
        }
    }

    pub fn decode_routed<B>(&mut self, backend: &B, spec: &TopkMoeSpec, weights: RoutedMoeWeightsRef<'_, B::Weight>, request: ExpertDecodeRequest<'_>, inputs: RoutedMoeInputs<'_, B::Tensor>) -> Result<B::Tensor, BackendError>
    where
        B: ExpertDecodeBackend<MoeState = S>,
    {
        self.decode_routed_with_ready(backend, spec, weights, request, inputs, || Ok(()))
    }

    fn decode_routed_with_ready<B, R>(
        &mut self,
        backend: &B,
        spec: &TopkMoeSpec,
        weights: RoutedMoeWeightsRef<'_, B::Weight>,
        request: ExpertDecodeRequest<'_>,
        inputs: RoutedMoeInputs<'_, B::Tensor>,
        on_route_ready: R,
    ) -> Result<B::Tensor, BackendError>
    where
        B: ExpertDecodeBackend<MoeState = S>,
        R: FnOnce() -> Result<(), BackendError>,
    {
        let ExpertDecodeRequest { layer, source, position, next } = request;
        let observe_single_token = backend.token_rows(inputs.route) == 1;
        if self.position != Some(position) {
            if self.position.is_some() || layer != self.first_layer {
                return Err(BackendError::ExpertLoad(format!("expert pipeline token 边界错误: position={position},layer={layer}")));
            }
            self.predictor.begin_token().map_err(BackendError::ExpertLoad)?;
            self.position = Some(position);
            if observe_single_token {
                self.start_prefetch(backend, spec, layer, source)?;
            }
        }

        let predictor = &mut self.predictor;
        let stats = &mut self.stats;
        let active_position = &mut self.position;
        let mut on_route_ready = Some(on_route_ready);
        let (routed, route_ready_fired) = decode_routed_topk_moe(backend, spec, weights, layer, source, &mut self.backend_state, inputs, |active_experts, _state| {
            // backend 契约保证回调至多触发一次；一次都未触发时不能静默跳过
            // shared experts，由下方的 route_ready_fired 检查显式报错。
            if let Some(on_route_ready) = on_route_ready.take() {
                on_route_ready()?;
            }
            if observe_single_token && !active_experts.is_empty() {
                predictor.observe_route(layer, active_experts).map_err(|msg| BackendError::ExpertLoad(format!("decode L{layer} route 塌缩: active={active_experts:?}: {msg}")))?;
            }
            match next {
                Some((next_layer, next_source)) if observe_single_token => predict_prefetch_request(predictor, stats, spec, next_layer, next_source),
                Some(_) => Ok(None),
                None => {
                    predictor.finish_token().map_err(BackendError::ExpertLoad)?;
                    *active_position = None;
                    Ok(None)
                }
            }
        })?;
        if !route_ready_fired {
            return Err(BackendError::Compute { msg: format!("expert pipeline L{layer} backend 未触发 route ready 回调，shared experts 未执行") });
        }
        Ok(routed)
    }
}
