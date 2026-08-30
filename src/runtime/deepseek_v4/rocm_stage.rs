//! DeepSeek-V4 的 ROCm stage 组合。
//!
//! session、队列、completion 与 pipeline window 全部由 `runtime::prefill`
//! 提供；本模块只描述一个 DeepSeek stage 的共享权重、会话 CSA 和模型计算。

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use crate::{
    attention::{compressed_sparse::CompressedSparseKernel, hyper_connection::HyperConnectionKernel},
    backend::rocm::{RocmCompressedKvSerde, RocmCompressedKvStorage, RocmContext, RocmPrefillExperts, RocmTensor, RocmWeight},
    backend::{BackendError, BackendResources, SegmentedTensorBackend, SpeculativeCacheBackend, SpeculativeCacheCommit, StageExecutionBackend, StageTensorBackend},
    runtime::prefill::{SchedulerSessions, StageSchedulerConfig, StageSchedulerHandle, StageSchedulerOutput, StageWorkKind, drive_stage_scheduler_recoverable},
};

use super::{DeepSeekV4, DeepSeekV4Config, DeepSeekV4LayerCache, DeepSeekV4PrefillSegment, DeepSeekV4RopeTables, deepseek_v4_prefill_layer, deepseek_v4_prefill_layer_segmented};

pub struct DeepSeekV4StageState {
    backend: RocmContext,
    model: Arc<DeepSeekV4>,
    rope: Arc<DeepSeekV4RopeTables>,
    layer_start: usize,
    layer_cache: Arc<DeepSeekV4LayerCache<RocmWeight>>,
    experts: Arc<Mutex<RocmPrefillExperts>>,
    caches: Vec<RocmCompressedKvStorage>,
    capture_plan: Option<Arc<crate::runtime::speculative::HiddenStateCapturePlan>>,
    profile: bool,
}

pub struct DeepSeekV4StageCache {
    pub layer_start: usize,
    pub layers: Vec<RocmCompressedKvSerde>,
}

#[derive(Clone)]
pub struct DeepSeekV4StageTemplate {
    backend: RocmContext,
    model: Arc<DeepSeekV4>,
    rope: Arc<DeepSeekV4RopeTables>,
    layer_start: usize,
    layer_count: usize,
    layer_cache: Arc<DeepSeekV4LayerCache<RocmWeight>>,
    experts: Arc<Mutex<RocmPrefillExperts>>,
    capture_plan: Option<Arc<crate::runtime::speculative::HiddenStateCapturePlan>>,
    profile: bool,
}

impl DeepSeekV4StageState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: RocmContext,
        model: Arc<DeepSeekV4>,
        rope: Arc<DeepSeekV4RopeTables>,
        layer_start: usize,
        layer_cache: DeepSeekV4LayerCache<RocmWeight>,
        experts: RocmPrefillExperts,
        caches: Vec<RocmCompressedKvStorage>,
        capture_plan: Option<Arc<crate::runtime::speculative::HiddenStateCapturePlan>>,
        profile: bool,
    ) -> Self {
        Self { backend, model, rope, layer_start, layer_cache: Arc::new(layer_cache), experts: Arc::new(Mutex::new(experts)), caches, capture_plan, profile }
    }

    pub fn fresh_session(&self) -> Result<Self, BackendError> {
        self.template().open_session()
    }

    pub fn fork_session(&self) -> Result<Self, BackendError> {
        Ok(Self {
            backend: self.backend,
            model: self.model.clone(),
            rope: self.rope.clone(),
            layer_start: self.layer_start,
            layer_cache: self.layer_cache.clone(),
            experts: self.experts.clone(),
            caches: self.caches.iter().map(RocmCompressedKvStorage::fork_session).collect::<Result<_, _>>()?,
            capture_plan: self.capture_plan.clone(),
            profile: self.profile,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.caches.len()
    }

    pub fn reset_session(&mut self) {
        for cache in &mut self.caches {
            cache.reset_session();
        }
    }

    pub fn cache_allocated_bytes(&self) -> u64 {
        self.caches.iter().map(RocmCompressedKvStorage::allocated_bytes).sum()
    }

    pub fn cache_batch_allocated_bytes(&self) -> u64 {
        self.caches.iter().map(RocmCompressedKvStorage::batch_allocated_bytes).sum()
    }

    pub fn download_cache(&self) -> Result<DeepSeekV4StageCache, BackendError> {
        self.backend.activate().map_err(crate::backend::compute_error)?;
        Ok(DeepSeekV4StageCache { layer_start: self.layer_start, layers: self.caches.iter().map(RocmCompressedKvStorage::download_snapshot).collect::<Result<_, _>>()? })
    }

    pub fn upload_cache(&mut self, snapshot: DeepSeekV4StageCache) -> Result<(), BackendError> {
        if snapshot.layer_start != self.layer_start || snapshot.layers.len() != self.caches.len() {
            return Err(crate::backend::compute_error(format!("DeepSeek-V4 stage cache {}/{} 与当前 {}/{} 不匹配", snapshot.layer_start, snapshot.layers.len(), self.layer_start, self.caches.len())));
        }
        self.backend.activate().map_err(crate::backend::compute_error)?;
        for (cache, snapshot) in self.caches.iter_mut().zip(snapshot.layers) {
            cache.upload_snapshot(snapshot)?;
        }
        Ok(())
    }

    pub fn template(&self) -> DeepSeekV4StageTemplate {
        DeepSeekV4StageTemplate {
            backend: self.backend,
            model: self.model.clone(),
            rope: self.rope.clone(),
            layer_start: self.layer_start,
            layer_count: self.caches.len(),
            layer_cache: self.layer_cache.clone(),
            experts: self.experts.clone(),
            capture_plan: self.capture_plan.clone(),
            profile: self.profile,
        }
    }
}

impl DeepSeekV4StageTemplate {
    pub fn open_session(&self) -> Result<DeepSeekV4StageState, BackendError> {
        Ok(DeepSeekV4StageState {
            backend: self.backend,
            model: self.model.clone(),
            rope: self.rope.clone(),
            layer_start: self.layer_start,
            layer_cache: self.layer_cache.clone(),
            experts: self.experts.clone(),
            caches: allocate_stage_caches(&self.backend, &self.model, self.layer_start, self.layer_start + self.layer_count)?,
            capture_plan: self.capture_plan.clone(),
            profile: self.profile,
        })
    }
}

enum DeepSeekV4StageAction {
    Forward {
        _chunk_index: usize,
        tokens: Vec<u32>,
        hidden: RocmTensor,
        captures: Vec<RocmTensor>,
        capture: bool,
        speculative: bool,
        begin_speculative: bool,
        commit_speculative: Option<SpeculativeCacheCommit>,
        dspark_prefill_from: Option<usize>,
    },
    CommitSpeculative(SpeculativeCacheCommit),
}

pub struct DeepSeekV4StageValue {
    action: DeepSeekV4StageAction,
}

pub struct DeepSeekV4StageOutput {
    pub session: usize,
    pub position: usize,
    pub tokens: Vec<u32>,
    pub hidden: RocmTensor,
    pub captures: Vec<RocmTensor>,
    pub dspark_prefill_from: Option<usize>,
}

pub enum DeepSeekV4StageEvent {
    Forward(DeepSeekV4StageOutput),
    CommittedSpeculative { session: usize },
    Closed { session: usize, states: Vec<DeepSeekV4StageState> },
}

pub struct DeepSeekV4StagePipeline<'a> {
    scheduler: &'a StageSchedulerHandle<DeepSeekV4StageValue, DeepSeekV4StageState>,
    capture_count: usize,
}

pub struct DeepSeekV4ReadyForward {
    pub session: usize,
    pub chunk_index: usize,
    pub position: usize,
    pub tokens: Vec<u32>,
    pub hidden: RocmTensor,
    pub speculative: bool,
    pub begin_speculative: bool,
    pub commit_speculative: Option<SpeculativeCacheCommit>,
}

impl<'a> DeepSeekV4StagePipeline<'a> {
    pub fn new(scheduler: &'a StageSchedulerHandle<DeepSeekV4StageValue, DeepSeekV4StageState>, capture_count: usize) -> Self {
        Self { scheduler, capture_count }
    }

    pub fn push(&self, session: usize, chunk_index: usize, position: usize, tokens: Vec<u32>, hidden: RocmTensor, dspark_prefill_from: Option<usize>) -> Result<(), String> {
        self.scheduler
            .submit(
                session,
                position,
                DeepSeekV4StageValue {
                    action: DeepSeekV4StageAction::Forward {
                        _chunk_index: chunk_index,
                        tokens,
                        hidden,
                        captures: Vec::new(),
                        capture: dspark_prefill_from.is_some(),
                        speculative: false,
                        begin_speculative: false,
                        commit_speculative: None,
                        dspark_prefill_from,
                    },
                },
            )
            .map_err(stage_error)
    }

    pub fn push_dspark(&self, session: usize, chunk_index: usize, position: usize, tokens: Vec<u32>, hidden: RocmTensor, speculative: bool, commit_speculative: Option<SpeculativeCacheCommit>) -> Result<(), String> {
        self.scheduler
            .submit(
                session,
                position,
                DeepSeekV4StageValue {
                    action: DeepSeekV4StageAction::Forward { _chunk_index: chunk_index, tokens, hidden, captures: Vec::new(), capture: true, speculative, begin_speculative: speculative, commit_speculative, dspark_prefill_from: None },
                },
            )
            .map_err(stage_error)
    }

    pub fn push_dspark_ready(&self, forwards: Vec<DeepSeekV4ReadyForward>) -> Result<(), String> {
        for forward in forwards {
            let value = DeepSeekV4StageValue {
                action: DeepSeekV4StageAction::Forward {
                    _chunk_index: forward.chunk_index,
                    tokens: forward.tokens,
                    hidden: forward.hidden,
                    captures: Vec::new(),
                    capture: true,
                    speculative: forward.speculative,
                    begin_speculative: forward.begin_speculative,
                    commit_speculative: forward.commit_speculative,
                    dspark_prefill_from: None,
                },
            };
            self.scheduler.submit_ready_cohort(std::iter::once((forward.session, forward.position, value))).map_err(stage_error)?;
        }
        Ok(())
    }

    pub fn pull(&self) -> Result<DeepSeekV4StageOutput, String> {
        match self.pull_event()? {
            DeepSeekV4StageEvent::Forward(output) => Ok(output),
            DeepSeekV4StageEvent::CommittedSpeculative { session } => Err(format!("DeepSeek stage 意外收到 CommitSpeculative session={session}")),
            DeepSeekV4StageEvent::Closed { session, .. } => Err(format!("DeepSeek stage 意外收到 Closed session={session}")),
        }
    }

    pub fn pull_event(&self) -> Result<DeepSeekV4StageEvent, String> {
        loop {
            if let Some(event) = self.try_pull_event()? {
                return Ok(event);
            }
            std::thread::yield_now();
        }
    }

    pub fn try_pull_event(&self) -> Result<Option<DeepSeekV4StageEvent>, String> {
        loop {
            match self.scheduler.try_recv().map_err(stage_error)? {
                Some(StageSchedulerOutput::Work { session, position, value: DeepSeekV4StageValue { action: DeepSeekV4StageAction::Forward { tokens, hidden, captures, capture, dspark_prefill_from, .. } }, .. }) => {
                    let expected = if capture { self.capture_count } else { 0 };
                    if captures.len() != expected {
                        return Err(format!("DeepSeek stage captures={} 期望={expected}", captures.len()));
                    }
                    return Ok(Some(DeepSeekV4StageEvent::Forward(DeepSeekV4StageOutput { session, position, tokens, hidden, captures, dspark_prefill_from })));
                }
                Some(StageSchedulerOutput::Work { session, value: DeepSeekV4StageValue { action: DeepSeekV4StageAction::CommitSpeculative(_) }, .. }) => return Ok(Some(DeepSeekV4StageEvent::CommittedSpeculative { session })),
                Some(StageSchedulerOutput::Opened { .. }) => continue,
                Some(StageSchedulerOutput::Closed { session, states }) => return Ok(Some(DeepSeekV4StageEvent::Closed { session, states })),
                None => return Ok(None),
            }
        }
    }

    pub fn pull_ready_events(&self) -> Result<Vec<DeepSeekV4StageEvent>, String> {
        decode_ready_events(self.scheduler.recv_ready().map_err(stage_error)?, self.capture_count)
    }

    /// 动态 admission 循环不能等待一个尚未提交的 work；立即排空当前已完成事件，
    /// 没有结果时返回空集，让外层继续接收 tokenization 和新请求。
    pub fn try_pull_ready_events(&self) -> Result<Vec<DeepSeekV4StageEvent>, String> {
        decode_ready_events(self.scheduler.try_recv_ready().map_err(stage_error)?, self.capture_count)
    }

    pub fn submit_commit_speculative(&self, session: usize, commit: SpeculativeCacheCommit) -> Result<(), String> {
        self.scheduler.submit(session, 0, DeepSeekV4StageValue { action: DeepSeekV4StageAction::CommitSpeculative(commit) }).map_err(stage_error)
    }

    pub fn close(&self, session: usize) -> Result<(), String> {
        self.scheduler.close(session).map_err(stage_error)
    }
}

fn decode_ready_events(outputs: Vec<StageSchedulerOutput<DeepSeekV4StageValue, DeepSeekV4StageState>>, capture_count: usize) -> Result<Vec<DeepSeekV4StageEvent>, String> {
    let mut events = Vec::new();
    for output in outputs {
        match output {
            StageSchedulerOutput::Work { session, position, value: DeepSeekV4StageValue { action: DeepSeekV4StageAction::Forward { tokens, hidden, captures, capture, dspark_prefill_from, .. } }, .. } => {
                let expected = if capture { capture_count } else { 0 };
                if captures.len() != expected {
                    return Err(format!("DeepSeek stage captures={} 期望={expected}", captures.len()));
                }
                events.push(DeepSeekV4StageEvent::Forward(DeepSeekV4StageOutput { session, position, tokens, hidden, captures, dspark_prefill_from }));
            }
            StageSchedulerOutput::Work { session, value: DeepSeekV4StageValue { action: DeepSeekV4StageAction::CommitSpeculative(_) }, .. } => events.push(DeepSeekV4StageEvent::CommittedSpeculative { session }),
            // Opened 表示 scheduler 已经前进；本轮没有模型输出也必须返回外层，
            // 下一轮才会为新 session 提交第一块 prefill。
            StageSchedulerOutput::Opened { .. } => {}
            StageSchedulerOutput::Closed { session, states } => events.push(DeepSeekV4StageEvent::Closed { session, states }),
        }
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opened_only_ready_batch_returns_to_outer_scheduler() {
        let outputs: Vec<StageSchedulerOutput<DeepSeekV4StageValue, DeepSeekV4StageState>> = vec![StageSchedulerOutput::Opened { session: 0 }];
        assert!(decode_ready_events(outputs, 0).unwrap().is_empty());
    }
}

pub fn drive_deepseek_v4_stage_pipeline<D, R>(
    states: Vec<Vec<DeepSeekV4StageState>>,
    session_capacity: usize,
    cfg: &DeepSeekV4Config,
    pipeline_work_window: usize,
    decode_batch_limit: usize,
    profile: bool,
    drive: D,
) -> Result<(Result<R, BackendError>, SchedulerSessions<DeepSeekV4StageState>), BackendError>
where
    D: FnOnce(&StageSchedulerHandle<DeepSeekV4StageValue, DeepSeekV4StageState>) -> Result<R, BackendError>,
{
    let first = states.first().ok_or_else(|| compute("DeepSeek stage pipeline 没有 session"))?;
    let backends = first.iter().map(|state| state.backend).collect::<Vec<_>>();
    let work_backend = *backends.first().ok_or_else(|| compute("DeepSeek stage pipeline 没有 stage"))?;
    let free_bytes = backends.iter().filter_map(|backend| backend.stage_available_bytes().ok()).min().unwrap_or(0);
    let scheduler = StageSchedulerConfig {
        session_capacity,
        batch_work_limit: free_bytes.saturating_mul(3).saturating_div(4).max(cfg.hidden_size.saturating_mul(256)),
        execution_slots: 1,
        decode_execution_slots: 1,
        pipeline_work_window: pipeline_work_window.max(1),
        prefill_admission_burst: 1,
        decode_batch_limit: session_capacity.min(decode_batch_limit.max(1)),
        prefill_batch_limit: session_capacity,
        profile_completion: profile,
    };
    drive_stage_scheduler_recoverable(
        backends,
        states,
        scheduler,
        move |value| match &value.action {
            DeepSeekV4StageAction::Forward { hidden, speculative, .. } => work_backend.token_rows(hidden).saturating_mul(cfg.hidden_size).saturating_mul(if *speculative || work_backend.token_rows(hidden) == 1 { 16 } else { 256 }).max(1),
            DeepSeekV4StageAction::CommitSpeculative(_) => 1,
        },
        |value| match &value.action {
            DeepSeekV4StageAction::Forward { hidden, speculative, .. } if !*speculative && hidden.rows != 1 => StageWorkKind::Prefill,
            _ => StageWorkKind::Decode,
        },
        |_, states, stage, batch| run_stage_batch(states, stage, batch),
        drive,
    )
}

fn run_stage_batch(states: &mut [Option<DeepSeekV4StageState>], stage: usize, mut batch: Vec<(usize, usize, DeepSeekV4StageValue)>) -> Result<Vec<(usize, usize, DeepSeekV4StageValue)>, BackendError> {
    if batch.len() > 1 {
        let (forwards, serial): (Vec<_>, Vec<_>) = batch.drain(..).partition(|(_, _, value)| matches!(&value.action, DeepSeekV4StageAction::Forward { .. }));
        if forwards.len() > 1 {
            let mut output = run_stage_forward_cohort(states, stage, forwards)?;
            output.extend(run_stage_batch(states, stage, serial)?);
            return Ok(output);
        }
        batch = serial;
        batch.extend(forwards);
    }
    let mut output = Vec::with_capacity(batch.len());
    for (session, position, mut value) in batch {
        let state = states.get_mut(session).and_then(Option::as_mut).ok_or_else(|| compute(format!("DeepSeek stage={stage} session={session} 尚未打开")))?;
        if state.profile {
            state.backend.profile_stage_begin(true)?;
        }
        match &mut value.action {
            DeepSeekV4StageAction::CommitSpeculative(commit) => {
                for cache in &mut state.caches {
                    state.backend.commit_speculative_cache(cache, *commit)?;
                }
            }
            DeepSeekV4StageAction::Forward { tokens, hidden, captures, capture, begin_speculative, commit_speculative, .. } => {
                if let Some(commit) = commit_speculative {
                    for cache in &mut state.caches {
                        state.backend.commit_speculative_cache(cache, *commit)?;
                    }
                }
                if *begin_speculative {
                    for cache in &mut state.caches {
                        state.backend.begin_speculative_cache(cache)?;
                    }
                }
                state.backend.activate_stage()?;
                let decode = hidden.rows == 1;
                let mut current = if stage == 0 {
                    if state.profile {
                        state.backend.profile_device_operator("input_expand")?;
                    }
                    state.backend.hyper_connection_expand(hidden, state.model.config().hyper_connection_copies)?
                } else if decode {
                    state.backend.move_tensor_to_stage_ordered(hidden.clone())?
                } else {
                    state.backend.tensor_on_device(hidden.clone())?
                };
                let mut moved_captures = if stage == 0 {
                    std::mem::take(captures)
                } else if decode {
                    std::mem::take(captures).into_iter().map(|capture| state.backend.move_tensor_to_stage_ordered(capture)).collect::<Result<Vec<_>, _>>()?
                } else {
                    std::mem::take(captures).into_iter().map(|capture| state.backend.tensor_on_device(capture)).collect::<Result<Vec<_>, _>>()?
                };
                let positions = (position..position + tokens.len()).collect::<Vec<_>>();
                let mut experts = state.experts.lock().map_err(|_| compute(format!("DeepSeek stage={stage} expert 锁中毒")))?;
                for layer in state.layer_start..state.layer_start + state.caches.len() {
                    current = state.backend.tensor_as_f32(current)?;
                    let prepared = state.layer_cache.get(layer).ok_or_else(|| compute(format!("L{layer} 未预装配")))?;
                    let spec = state.model.layer_spec(layer).map_err(|error| compute(error.to_string()))?;
                    current = deepseek_v4_prefill_layer(
                        &state.backend,
                        state.model.config(),
                        spec,
                        prepared,
                        &mut state.caches[layer - state.layer_start],
                        &mut experts,
                        layer,
                        &current,
                        state.rope.layer(spec),
                        &positions,
                        tokens,
                        true,
                        &spec.hyper_connection,
                    )?;
                    if state.profile {
                        state.backend.profile_device_operator("layer_compact")?;
                    }
                    current = state.backend.tensor_as_bf16(current)?;
                    if *capture && state.capture_plan.as_ref().is_some_and(|plan| plan.captures_layer_output(layer)) {
                        let capture = state.backend.hyper_connection_mean(&current, state.model.config().hyper_connection_copies)?;
                        moved_captures.push(state.backend.tensor_as_bf16(capture)?);
                    }
                }
                if state.profile {
                    state.backend.profile_device_operator("stage_stabilize")?;
                }
                *hidden = state.backend.stabilize_stage_tensor(current)?;
                if state.profile {
                    state.backend.profile_device_operator("stage_captures")?;
                }
                *captures = moved_captures.into_iter().map(|capture| state.backend.stabilize_stage_tensor(capture)).collect::<Result<Vec<_>, _>>()?;
                if state.profile {
                    state.backend.profile_device_operator("stage_tail")?;
                }
            }
        }
        if state.profile {
            state.backend.profile_stage_end()?;
        }
        output.push((session, position, value));
    }
    Ok(output)
}

fn run_stage_forward_cohort(states: &mut [Option<DeepSeekV4StageState>], stage: usize, mut batch: Vec<(usize, usize, DeepSeekV4StageValue)>) -> Result<Vec<(usize, usize, DeepSeekV4StageValue)>, BackendError> {
    batch.sort_by_key(|(session, _, _)| *session);
    if batch.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(compute(format!("DeepSeek stage={stage} cohort 含重复 session")));
    }
    let wanted = batch.iter().map(|(session, _, _)| *session).collect::<HashSet<_>>();
    let mut active = states.iter_mut().enumerate().filter_map(|(session, state)| wanted.contains(&session).then(|| state.as_mut().map(|state| (session, state))).flatten()).collect::<Vec<_>>();
    if active.len() != batch.len() || active.iter().zip(&batch).any(|((session, _), (expected, _, _))| session != expected) {
        return Err(compute(format!("DeepSeek stage={stage} cohort session 未全部打开")));
    }

    let profile = active.iter().any(|(_, state)| state.profile);
    let backend = active[0].1.backend;
    if profile {
        backend.profile_stage_begin(true)?;
    }
    backend.activate_stage()?;
    let model = active[0].1.model.clone();
    let rope = active[0].1.rope.clone();
    let layer_start = active[0].1.layer_start;
    let layer_count = active[0].1.caches.len();
    let layer_cache = active[0].1.layer_cache.clone();
    let capture_plan = active[0].1.capture_plan.clone();
    let expert_pool = active[0].1.experts.clone();
    let mut row_counts = Vec::with_capacity(batch.len());
    let mut moved_inputs = Vec::with_capacity(batch.len());
    let mut moved_captures = Vec::with_capacity(batch.len());
    let mut positions = Vec::with_capacity(batch.len());
    let mut capture_outputs = Vec::with_capacity(batch.len());
    for ((_, state), (_, position, value)) in active.iter_mut().zip(batch.iter_mut()) {
        let DeepSeekV4StageAction::Forward { tokens, hidden, captures, capture, begin_speculative, commit_speculative, .. } = &mut value.action else { unreachable!("cohort 已检查 forward") };
        if hidden.rows != tokens.len() {
            return Err(compute(format!("DeepSeek stage={stage} cohort hidden rows={} tokens={}", hidden.rows, tokens.len())));
        }
        if let Some(commit) = commit_speculative {
            for cache in &mut state.caches {
                state.backend.commit_speculative_cache(cache, *commit)?;
            }
        }
        if *begin_speculative {
            for cache in &mut state.caches {
                state.backend.begin_speculative_cache(cache)?;
            }
        }
        row_counts.push(tokens.len());
        moved_inputs.push(if stage == 0 {
            hidden.clone()
        } else if hidden.rows == 1 {
            state.backend.move_tensor_to_stage_ordered(hidden.clone())?
        } else {
            state.backend.tensor_on_device(hidden.clone())?
        });
        moved_captures.push(if stage == 0 {
            std::mem::take(captures)
        } else if hidden.rows == 1 {
            std::mem::take(captures).into_iter().map(|capture| state.backend.move_tensor_to_stage_ordered(capture)).collect::<Result<Vec<_>, _>>()?
        } else {
            std::mem::take(captures).into_iter().map(|capture| state.backend.tensor_on_device(capture)).collect::<Result<Vec<_>, _>>()?
        });
        positions.push((*position..*position + tokens.len()).collect::<Vec<_>>());
        capture_outputs.push(*capture);
    }
    let input_refs = moved_inputs.iter().collect::<Vec<_>>();
    let mut current = backend.concat_token_rows(&input_refs)?;
    if stage == 0 {
        if profile {
            backend.profile_device_operator("input_expand")?;
        }
        current = backend.hyper_connection_expand(&current, model.config().hyper_connection_copies)?;
    }
    let mut experts = expert_pool.lock().map_err(|_| compute(format!("DeepSeek stage={stage} expert 锁中毒")))?;
    for layer in layer_start..layer_start + layer_count {
        current = backend.tensor_as_f32(current)?;
        let prepared = layer_cache.get(layer).ok_or_else(|| compute(format!("L{layer} 未预装配")))?;
        let spec = model.layer_spec(layer).map_err(|error| compute(error.to_string()))?;
        let mut segments = active
            .iter_mut()
            .zip(batch.iter())
            .zip(positions.iter())
            .map(|(((_, state), (_, _, value)), positions)| {
                let DeepSeekV4StageAction::Forward { tokens, .. } = &value.action else { unreachable!("cohort 已检查 forward") };
                DeepSeekV4PrefillSegment { cache: &mut state.caches[layer - layer_start], rope: rope.layer(spec), positions, token_ids: tokens, causal_batch: true }
            })
            .collect::<Vec<_>>();
        current = deepseek_v4_prefill_layer_segmented(&backend, model.config(), spec, prepared, &mut experts, layer, &current, &mut segments, &spec.hyper_connection)?;
        if profile {
            backend.profile_device_operator("layer_compact")?;
        }
        current = backend.tensor_as_bf16(current)?;
        if capture_plan.as_ref().is_some_and(|plan| plan.captures_layer_output(layer)) {
            let capture = backend.hyper_connection_mean(&current, model.config().hyper_connection_copies)?;
            let capture = backend.tensor_as_bf16(capture)?;
            let mut row_start = 0usize;
            for (index, &rows) in row_counts.iter().enumerate() {
                if capture_outputs[index] {
                    moved_captures[index].push(backend.slice_token_rows(&capture, row_start, rows)?);
                }
                row_start += rows;
            }
        }
    }
    if profile {
        backend.profile_device_operator("stage_stabilize")?;
    }
    let mut row_start = 0usize;
    for (index, ((_, state), (_, _, value))) in active.iter_mut().zip(batch.iter_mut()).enumerate() {
        let DeepSeekV4StageAction::Forward { hidden, captures, .. } = &mut value.action else { unreachable!("cohort 已检查 forward") };
        let split = backend.slice_token_rows(&current, row_start, row_counts[index])?;
        *hidden = state.backend.stabilize_stage_tensor(split)?;
        *captures = std::mem::take(&mut moved_captures[index]).into_iter().map(|capture| state.backend.stabilize_stage_tensor(capture)).collect::<Result<Vec<_>, _>>()?;
        row_start += row_counts[index];
    }
    if profile {
        backend.profile_stage_end()?;
    }
    Ok(batch)
}

pub fn allocate_stage_caches(context: &RocmContext, model: &DeepSeekV4, layer_start: usize, layer_end: usize) -> Result<Vec<RocmCompressedKvStorage>, BackendError> {
    (layer_start..layer_end)
        .map(|layer| {
            let spec = model.layer_spec(layer).map_err(|error| compute(error.to_string()))?;
            context.allocate_compressed_kv(&spec.attention)
        })
        .collect()
}

fn compute(message: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: message.into() }
}
fn stage_error(error: BackendError) -> String {
    format!("{error:?}")
}
