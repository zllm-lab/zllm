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
    backend::rocm::{RocmCompressedKvSerde, RocmCompressedKvStorage, RocmContext, RocmCsaSelection, RocmPrefillExperts, RocmTensor, RocmWeight},
    backend::{BackendError, BackendResources, SegmentedTensorBackend, SpeculativeCacheBackend, SpeculativeCacheCommit, StageExecutionBackend, StageTensorBackend},
    runtime::prefill::{SchedulerSessions, StageSchedulerConfig, StageSchedulerHandle, StageSchedulerOutput, StageWorkKind, drive_stage_scheduler_recoverable},
};

use super::{DeepSeekV4, DeepSeekV4Config, DeepSeekV4LayerCache, DeepSeekV4PrefillSegment, DeepSeekV4RopeTables, V41SharedState, deepseek_v4_prefill_layer, deepseek_v4_prefill_layer_segmented, deepseek_v4_prefill_layer_v41_chained};

/// V4.1 的会话级全层 CSA cache 表(40 层,各归属其 stage 的 device);
/// 跨 stage 的压缩历史经此共享,kernel 按 P2P 读归属 device 的 buffer。
pub type V41CacheTable = Arc<Vec<Mutex<RocmCompressedKvStorage>>>;

fn fork_v41_table(table: &V41CacheTable) -> Result<V41CacheTable, BackendError> {
    let forked = table.iter().map(|cache| cache.lock().map_err(|_| crate::backend::compute_error("V4.1 cache 表锁中毒"))?.fork_session()).collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::new(forked.into_iter().map(Mutex::new).collect::<Vec<_>>()))
}

/// 从同一组模板打开一个 session；V4.1 的全层 cache 表只 fork 一次，随后由
/// 全部 stage 共享。逐 stage 各自 fork 会让非源层读到另一张空表。
pub(super) fn open_stage_session(templates: &[DeepSeekV4StageTemplate]) -> Result<Vec<DeepSeekV4StageState>, BackendError> {
    let v41_caches = templates.first().and_then(|template| template.v41_caches.as_ref()).map(fork_v41_table).transpose()?;
    let engram = templates.first().and_then(|template| template.engram.as_ref()).map(|engram| engram.lock().map(|engram| Arc::new(Mutex::new(engram.fork()))).map_err(|_| crate::backend::compute_error("engram 锁中毒"))).transpose()?;
    templates
        .iter()
        .map(|template| {
            let mut state = template.open_session_with_v41(v41_caches.clone())?;
            state.engram = engram.clone();
            Ok(state)
        })
        .collect()
}

pub struct DeepSeekV4StageState {
    backend: RocmContext,
    model: Arc<DeepSeekV4>,
    rope: Arc<DeepSeekV4RopeTables>,
    layer_start: usize,
    layer_count: usize,
    layer_cache: Arc<DeepSeekV4LayerCache<RocmWeight>>,
    experts: Arc<Mutex<RocmPrefillExperts>>,
    caches: Vec<RocmCompressedKvStorage>,
    v41_caches: Option<V41CacheTable>,
    pub(crate) engram: Option<Arc<Mutex<super::engram_cpu::DeepSeekV4EngramCpu>>>,
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
    v41_caches: Option<V41CacheTable>,
    pub(crate) engram: Option<Arc<Mutex<super::engram_cpu::DeepSeekV4EngramCpu>>>,
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
        layer_count: usize,
        layer_cache: DeepSeekV4LayerCache<RocmWeight>,
        experts: RocmPrefillExperts,
        caches: Vec<RocmCompressedKvStorage>,
        v41_caches: Option<V41CacheTable>,
        engram: Option<Arc<Mutex<super::engram_cpu::DeepSeekV4EngramCpu>>>,
        capture_plan: Option<Arc<crate::runtime::speculative::HiddenStateCapturePlan>>,
        profile: bool,
    ) -> Self {
        Self { backend, model, rope, layer_start, layer_count, layer_cache: Arc::new(layer_cache), experts: Arc::new(Mutex::new(experts)), caches, v41_caches, engram, capture_plan, profile }
    }

    pub fn fresh_session(&self) -> Result<Self, BackendError> {
        self.template().open_session()
    }

    pub fn fork_session(&self) -> Result<Self, BackendError> {
        let v41_caches = self.v41_caches.as_ref().map(fork_v41_table).transpose()?;
        self.fork_session_with_v41(v41_caches)
    }

    pub(super) fn fork_v41_caches(&self) -> Result<Option<V41CacheTable>, BackendError> {
        self.v41_caches.as_ref().map(fork_v41_table).transpose()
    }

    pub(super) fn fork_session_with_v41(&self, v41_caches: Option<V41CacheTable>) -> Result<Self, BackendError> {
        Ok(Self {
            backend: self.backend,
            model: self.model.clone(),
            rope: self.rope.clone(),
            layer_start: self.layer_start,
            layer_count: self.layer_count,
            layer_cache: self.layer_cache.clone(),
            experts: self.experts.clone(),
            caches: self.caches.iter().map(RocmCompressedKvStorage::fork_session).collect::<Result<_, _>>()?,
            v41_caches,
            // engram 由 RocmSessionState 层统一 fork 一次并覆盖(全部 stage 共享一份)
            engram: self.engram.clone(),
            capture_plan: self.capture_plan.clone(),
            profile: self.profile,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn reset_session(&mut self) {
        for cache in &mut self.caches {
            cache.reset_session();
        }
        // V4.1:会话级 cache 表与 engram hash 一并重置
        if let Some(table) = &self.v41_caches {
            for cache in table.iter() {
                if let Ok(mut cache) = cache.lock() {
                    cache.reset_session();
                }
            }
        }
        if let Some(engram) = &self.engram {
            if let Ok(mut cpu) = engram.lock() {
                cpu.reset_hash();
            }
        }
    }

    fn begin_speculative(&mut self) -> Result<(), BackendError> {
        if self.layer_start == 0
            && let Some(engram) = &self.engram
        {
            engram.lock().map_err(|_| compute("engram 锁中毒"))?.begin_speculative().map_err(compute)?;
        }
        if let Some(table) = &self.v41_caches {
            for layer in self.layer_start..self.layer_start + self.layer_count {
                let mut cache = table[layer].lock().map_err(|_| compute(format!("V4.1 L{layer} cache 锁中毒")))?;
                self.backend.begin_speculative_cache(&mut cache)?;
            }
        } else {
            for cache in &mut self.caches {
                self.backend.begin_speculative_cache(cache)?;
            }
        }
        Ok(())
    }

    fn commit_speculative(&mut self, commit: SpeculativeCacheCommit) -> Result<(), BackendError> {
        if self.layer_start == 0
            && let Some(engram) = &self.engram
        {
            engram.lock().map_err(|_| compute("engram 锁中毒"))?.commit_speculative(commit.retained_rows()).map_err(compute)?;
        }
        if let Some(table) = &self.v41_caches {
            for layer in self.layer_start..self.layer_start + self.layer_count {
                let mut cache = table[layer].lock().map_err(|_| compute(format!("V4.1 L{layer} cache 锁中毒")))?;
                self.backend.commit_speculative_cache(&mut cache, commit)?;
            }
        } else {
            for cache in &mut self.caches {
                self.backend.commit_speculative_cache(cache, commit)?;
            }
        }
        Ok(())
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
            layer_count: self.layer_count,
            layer_cache: self.layer_cache.clone(),
            experts: self.experts.clone(),
            v41_caches: self.v41_caches.clone(),
            engram: self.engram.clone(),
            capture_plan: self.capture_plan.clone(),
            profile: self.profile,
        }
    }
}

impl DeepSeekV4StageTemplate {
    pub fn open_session(&self) -> Result<DeepSeekV4StageState, BackendError> {
        let v41_caches = self.v41_caches.as_ref().map(fork_v41_table).transpose()?;
        self.open_session_with_v41(v41_caches)
    }

    fn open_session_with_v41(&self, v41_caches: Option<V41CacheTable>) -> Result<DeepSeekV4StageState, BackendError> {
        Ok(DeepSeekV4StageState {
            backend: self.backend,
            model: self.model.clone(),
            rope: self.rope.clone(),
            layer_start: self.layer_start,
            layer_count: self.layer_count,
            layer_cache: self.layer_cache.clone(),
            experts: self.experts.clone(),
            caches: allocate_stage_caches(&self.backend, &self.model, self.layer_start, self.layer_start + self.layer_count)?,
            v41_caches,
            // 共享母版;新会话取出复用时经 reset_hash 清理序列状态
            engram: self.engram.clone(),
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
        /// V4.1 index 层发布后随 hidden 穿过 stage，供下一 index 层之前复用。
        v41_selection: Option<RocmCsaSelection>,
        /// 当前 work 最近一个 KV source 层的只读 cache 视图。跨 stage 只共享
        /// allocation，标量长度固定在本 work，不能回读已被上游推进的全局表。
        v41_source_cache: Option<(usize, RocmCompressedKvStorage)>,
        /// prefill 的两层 Engram WKV 在 stage0 提前投影，随工作项传到消费层。
        engram_batches: [Option<std::thread::JoinHandle<Result<super::engram_cpu::DeepSeekV4EngramBatch, String>>>; 2],
        /// 官方 mHC 中由前一子层发布、供下一子层折叠 hidden 的系数。
        v41_pre_mix: Option<RocmTensor>,
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
                        v41_selection: None,
                        v41_source_cache: None,
                        engram_batches: [None, None],
                        v41_pre_mix: None,
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
                    action: DeepSeekV4StageAction::Forward {
                        _chunk_index: chunk_index,
                        tokens,
                        hidden,
                        captures: Vec::new(),
                        capture: true,
                        speculative,
                        begin_speculative: speculative,
                        commit_speculative,
                        dspark_prefill_from: None,
                        v41_selection: None,
                        v41_source_cache: None,
                        engram_batches: [None, None],
                        v41_pre_mix: None,
                    },
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
                    v41_selection: None,
                    v41_source_cache: None,
                    engram_batches: [None, None],
                    v41_pre_mix: None,
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
        if forwards.len() > 1 && !states.iter().flatten().any(|state| state.model.is_v41()) {
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
                state.commit_speculative(*commit)?;
            }
            DeepSeekV4StageAction::Forward { tokens, hidden, captures, capture, begin_speculative, commit_speculative, v41_selection, v41_source_cache, engram_batches, v41_pre_mix, .. } => {
                if let Some(commit) = commit_speculative {
                    state.commit_speculative(*commit)?;
                }
                if *begin_speculative {
                    state.begin_speculative()?;
                }
                state.backend.activate_stage()?;
                let is_v41 = state.model.is_v41();
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
                let mut chained_pre = if !is_v41 {
                    None
                } else if stage == 0 {
                    let copies = state.model.config().hyper_connection_copies;
                    let mut identity = vec![0.0_f32; hidden.rows * copies];
                    for row in identity.chunks_exact_mut(copies) {
                        row[0] = 1.0;
                    }
                    Some(state.backend.stage_tensor_from_f32(identity, hidden.rows, copies)?)
                } else {
                    let incoming = std::mem::take(v41_pre_mix).ok_or_else(|| compute(format!("V4.1 stage={stage} 缺少前一 stage 的 pre_mix")))?;
                    Some(if decode { state.backend.move_tensor_to_stage_ordered(incoming)? } else { state.backend.tensor_on_device(incoming)? })
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
                let mut v41_shared = is_v41.then(|| V41SharedState::with_selection(std::mem::take(v41_selection)));
                if is_v41 && stage == 0 && state.engram.is_some() {
                    let engram = state.engram.as_ref().expect("上方已检查 engram");
                    let batches = {
                        let mut cpu = engram.lock().map_err(|_| compute("engram 锁中毒"))?;
                        for &token in tokens.iter() {
                            cpu.push_token(token);
                        }
                        let hash_len = cpu.hash_len();
                        if hash_len != position + tokens.len() {
                            return Err(compute(format!("engram hash 长度 {hash_len} 与 chunk 末尾 {} 不一致", position + tokens.len())));
                        }
                        [cpu.prepare_prefill_rows(0, position, tokens.len()), cpu.prepare_prefill_rows(1, position, tokens.len())]
                    };
                    for (slot, batch) in batches.into_iter().enumerate() {
                        let batch = batch.map_err(|error| compute(format!("engram L{}: {error}", if slot == 0 { 1 } else { 14 })))?;
                        engram_batches[slot] = Some(std::thread::Builder::new().name(format!("engram-project-{}", if slot == 0 { 1 } else { 14 })).spawn(move || batch.precompute()).map_err(|error| compute(format!("启动 engram projection: {error}")))?);
                    }
                }
                for layer in state.layer_start..state.layer_start + state.caches.len().max(if is_v41 { state.layer_count } else { 0 }) {
                    // V4.1 engram 在层入口注入(L1/L14):h D2H → CPU 门控 → H2D,不进显存
                    if is_v41 && (layer == 1 || layer == 14) && state.engram.is_some() {
                        let engram_started = std::time::Instant::now();
                        let engram = state.engram.as_ref().ok_or_else(|| compute("V4.1 缺少 engram 实例"))?;
                        let slot = if layer == 1 { 0 } else { 1 };
                        let rows = current.rows;
                        let (batch, precomputed) = if let Some(handle) = engram_batches[slot].take() {
                            (
                                handle.join().map_err(|_| compute(format!("engram L{layer} projection 线程 panic")))?.map_err(|error| compute(format!("engram L{layer} projection: {error}")))?,
                                true,
                            )
                        } else {
                            let mut cpu = engram.lock().map_err(|_| compute("engram 锁中毒"))?;
                            // VL:image span(全部位置共用 image_token_id)推入 DEAD
                            // 不参与 n-gram,且官方对这些行关闭 engram 门控直通。
                            let image_token = state.model.config().image_token_id;
                            let image_mask = tokens.iter().map(|&token| image_token.is_some_and(|id| token == id)).collect::<Vec<_>>();
                            if slot == 0 {
                                for (&token, image) in tokens.iter().zip(image_mask.iter()) {
                                    cpu.push_token(token, *image);
                                }
                            }
                            let hash_len = cpu.hash_len();
                            if hash_len != position + rows {
                                return Err(compute(format!("engram hash 长度 {hash_len} 与 chunk 末尾 {} 不一致", position + rows)));
                            }
                            (cpu.prepare_prefill_rows(slot, position, rows).map_err(|error| compute(format!("engram L{layer}: {error}")))?, false)
                        };
                        let prepare_done = std::time::Instant::now();
                        if precomputed && !decode {
                            let (projected, qk_weight, hc, dim, eps) = batch.into_projected().map_err(|error| compute(format!("engram L{layer}: {error}")))?;
                            current = state.backend.apply_engram_projected(current, projected, &qk_weight, rows, hc, dim, eps)?;
                            if state.profile {
                                let apply_done = std::time::Instant::now();
                                eprintln!(
                                    "[engram-cpu-profile] layer={layer} rows={rows} prepare_ms={:.3} gpu_apply_ms={:.3} wall_ms={:.3}",
                                    prepare_done.duration_since(engram_started).as_secs_f64() * 1000.0,
                                    apply_done.duration_since(prepare_done).as_secs_f64() * 1000.0,
                                    apply_done.duration_since(engram_started).as_secs_f64() * 1000.0,
                                );
                                state.backend.profile_device_operator("engram_projected")?;
                            }
                        } else {
                            let mut h = if rows >= 1024 {
                            state.backend.with_tensors_bf16_bits(&[&current], |segments| Ok(batch.expand_hidden_bf16(segments[0]))).map_err(|error| compute(format!("engram BF16 D2H: {error}")))?
                            } else {
                                current = state.backend.tensor_as_f32(current)?;
                                state.backend.tensor_to_f32(&current).map_err(|error| compute(format!("engram F32 D2H: {error}")))?
                            };
                            let d2h_done = std::time::Instant::now();
                            batch.apply(&mut h).map_err(|error| compute(format!("engram L{layer}: {error}")))?;
                            let apply_done = std::time::Instant::now();
                            current = state.backend.tensor_from_f32(h, rows, state.model.config().hidden_size * state.model.config().hyper_connection_copies).map_err(|error| compute(format!("engram H2D: {error}")))?;
                            if state.profile {
                                let h2d_done = std::time::Instant::now();
                                eprintln!(
                                    "[engram-cpu-profile] layer={layer} rows={rows} d2h_ms={:.3} prepare_ms={:.3} apply_ms={:.3} h2d_ms={:.3} wall_ms={:.3}",
                                    d2h_done.duration_since(prepare_done).as_secs_f64() * 1000.0,
                                    prepare_done.duration_since(engram_started).as_secs_f64() * 1000.0,
                                    apply_done.duration_since(d2h_done).as_secs_f64() * 1000.0,
                                    h2d_done.duration_since(apply_done).as_secs_f64() * 1000.0,
                                    h2d_done.duration_since(engram_started).as_secs_f64() * 1000.0,
                                );
                                state.backend.profile_device_operator("engram_cpu")?;
                            }
                        }
                    } else {
                        current = state.backend.tensor_as_f32(current)?;
                    }
                    let prepared = state.layer_cache.get(layer).ok_or_else(|| compute(format!("L{layer} 未预装配")))?;
                    let spec = state.model.layer_spec(layer).map_err(|error| compute(error.to_string()))?;
                    current = if is_v41 {
                        let table = state.v41_caches.as_ref().ok_or_else(|| compute("V4.1 缺少会话 cache 表"))?;
                        let mut cache_guard = table[layer].lock().map_err(|_| compute("V4.1 cache 锁中毒"))?;
                        // 非 source 的压缩层才加锁组源层(无压缩层不读源;source 层用自身);
                        // 源由 ≤layer 的最近 kv_source 发布,锁顺序与层循环一致(先大后小无并发死锁)
                        let source_guard;
                        let source_cache = if spec.attention.compression.is_some() && !state.model.is_kv_source(layer) {
                            let source_layer = state.model.kv_source_of(layer).ok_or_else(|| compute(format!("V4.1 L{layer} 缺少组源层映射")))?;
                            if source_layer == layer {
                                return Err(compute(format!("V4.1 L{layer} 非 source 层却引用自身为源(source 未正确发布)")));
                            }
                            if let Some((_, source)) = v41_source_cache.as_ref().filter(|(published_layer, _)| *published_layer == source_layer) {
                                Some(source)
                            } else {
                                source_guard = table[source_layer].lock().map_err(|_| compute("V4.1 source cache 锁中毒"))?;
                                Some(&*source_guard as &RocmCompressedKvStorage)
                            }
                        } else {
                            None
                        };
                        let (hidden, next_pre) = deepseek_v4_prefill_layer_v41_chained(
                            &state.backend,
                            &state.model,
                            spec,
                            prepared,
                            layer,
                            std::slice::from_mut(&mut *cache_guard),
                            source_cache,
                            v41_shared.as_mut().ok_or_else(|| compute("V4.1 缺少共享状态"))?,
                            &mut experts,
                            layer,
                            &current,
                            chained_pre.as_ref(),
                            state.rope.layer(spec),
                            &positions,
                            tokens,
                            true,
                            layer + 1 == state.model.layer_count(),
                        )?;
                        if state.model.is_kv_source(layer) {
                            *v41_source_cache = Some((layer, cache_guard.source_view(decode)));
                        }
                        chained_pre = Some(next_pre);
                        hidden
                    } else {
                        deepseek_v4_prefill_layer(
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
                        )?
                    };
                    if state.profile {
                        state.backend.profile_device_operator("layer_compact")?;
                    }
                    current = state.backend.tensor_as_bf16(current)?;
                    if *capture && state.capture_plan.as_ref().is_some_and(|plan| plan.captures_layer_output(layer)) {
                        let capture = state.backend.hyper_connection_mean(&current, state.model.config().hyper_connection_copies)?;
                        moved_captures.push(state.backend.tensor_as_bf16(capture)?);
                    }
                }
                *v41_selection = v41_shared.and_then(V41SharedState::into_selection).map(|selection| state.backend.stabilize_shared_selection(selection)).transpose()?;
                *v41_pre_mix = chained_pre.map(|pre| state.backend.stabilize_stage_tensor(pre)).transpose()?;
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
            state.commit_speculative(*commit)?;
        }
        if *begin_speculative {
            state.begin_speculative()?;
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
