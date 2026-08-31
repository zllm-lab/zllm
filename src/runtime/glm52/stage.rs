//! GLM-5.2 的 backend 无关 stage 编排。
//!
//! 本模块只描述模型层、session state 与 batch 执行。连续队列、completion 和
//! Open/Close 生命周期由 runtime::prefill 提供，设备迁移由 backend
//! capability 提供。

use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::{
    attention::{mla::MlaSpec, rope::RopeTable},
    backend::{BackendError, DsaStageBackend, ExpertPrefillBackend, StageTensorBackend},
    runtime::prefill::{SchedulerSessions, StageSchedulerConfig, StageSchedulerHandle, StageSchedulerOutput, StageWorkKind, TokenStreamBatchPoll, drive_stage_scheduler_with_batch_class, run_single_stage_chain},
    weight::Glm52Weights,
};

use super::{
    Glm52Config, Glm52DensePrefillLayer, Glm52MoePrefillLayer, Glm52PrefillSegment, glm52_dense_prefill_layer, glm52_dense_prefill_layer_segmented, glm52_moe_prefill_layer, glm52_moe_prefill_layer_segmented, glm52_prefill_stage_observed,
    load_prepare_dense_prefill_layer, load_prepare_moe_prefill_layer,
};

pub trait Glm52HiddenProjector<B: DsaStageBackend> {
    fn boundary(&self) -> usize;
    fn project_add(&self, backend: &B, hidden: &B::Tensor, accumulated: Option<B::Tensor>) -> Result<B::Tensor, BackendError>;
}

#[derive(Clone)]
pub enum Glm52PrefillLayer<W> {
    Dense(Glm52DensePrefillLayer<W>),
    Moe(Glm52MoePrefillLayer<W>),
}

pub struct Glm52StageState<B>
where
    B: DsaStageBackend + ExpertPrefillBackend,
{
    pub backend: B,
    pub layer_start: usize,
    pub layers: Arc<Vec<Glm52PrefillLayer<B::Weight>>>,
    pub experts: Arc<Mutex<B::PrefillExperts>>,
    pub cache: B::Cache,
    pub dsa: B::DsaState,
    /// 只描述该 session 在当前物理 stage 是否已经进入 decode；不持久化。
    pub decode_active: bool,
    pub hidden_projectors: Vec<Arc<dyn Glm52HiddenProjector<B> + Send + Sync>>,
}

impl<B> Glm52StageState<B>
where
    B: DsaStageBackend + ExpertPrefillBackend + Clone,
{
    pub fn fresh_session(&self, cfg: &Glm52Config, max_seq_len: usize) -> Result<Self, BackendError> {
        Ok(Self {
            backend: self.backend.clone(),
            layer_start: self.layer_start,
            layers: self.layers.clone(),
            experts: self.experts.clone(),
            cache: self.backend.new_stage_cache(cfg.layer_count, max_seq_len)?,
            dsa: self.backend.new_stage_dsa(cfg.layer_count, max_seq_len, cfg.index_head_dim, cfg.index_top_k)?,
            decode_active: false,
            hidden_projectors: self.hidden_projectors.clone(),
        })
    }

    pub fn reset_session(&mut self, cfg: &Glm52Config, max_seq_len: usize) -> Result<(), BackendError> {
        self.cache = self.backend.new_stage_cache(cfg.layer_count, max_seq_len)?;
        self.dsa = self.backend.new_stage_dsa(cfg.layer_count, max_seq_len, cfg.index_head_dim, cfg.index_top_k)?;
        self.decode_active = false;
        Ok(())
    }

    pub fn cache_allocated_bytes(&self) -> u64 {
        self.backend.stage_cache_allocated_bytes(&self.cache, &self.dsa)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_glm52_stage_states<B>(
    backends: &[B],
    layer_ends: &[usize],
    layer_start: usize,
    cfg: &Glm52Config,
    layers: Vec<Glm52PrefillLayer<B::Weight>>,
    experts: Vec<B::PrefillExperts>,
    max_seq_len: usize,
) -> Result<Vec<Glm52StageState<B>>, BackendError>
where
    B: DsaStageBackend + ExpertPrefillBackend + Clone,
{
    if backends.is_empty() || backends.len() != layer_ends.len() {
        return Err(BackendError::Compute { msg: format!("GLM stage backend/layer_ends 数量非法: {}/{}", backends.len(), layer_ends.len()) });
    }
    let mut groups = (0..backends.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    for (offset, layer) in layers.into_iter().enumerate() {
        let absolute_layer = layer_start + offset;
        let placement = layer_ends.iter().position(|&end| absolute_layer <= end).ok_or_else(|| BackendError::Compute { msg: format!("L{absolute_layer} 没有 distributed stage backend") })?;
        groups[placement].push(layer);
    }
    let mut experts = experts.into_iter();
    let mut next_layer = layer_start;
    let mut states = Vec::with_capacity(backends.len());
    for (backend, layers) in backends.iter().cloned().zip(groups) {
        let current_start = next_layer;
        next_layer += layers.len();
        let experts = experts.next().ok_or_else(|| BackendError::Compute { msg: "distributed stage 缺少 expert state".to_owned() })?;
        states.push(Glm52StageState {
            cache: backend.new_stage_cache(cfg.layer_count, max_seq_len)?,
            dsa: backend.new_stage_dsa(cfg.layer_count, max_seq_len, cfg.index_head_dim, cfg.index_top_k)?,
            decode_active: false,
            hidden_projectors: Vec::new(),
            backend,
            layer_start: current_start,
            layers: Arc::new(layers),
            experts: Arc::new(Mutex::new(experts)),
        });
    }
    if experts.next().is_some() {
        return Err(BackendError::Compute { msg: "distributed stage expert state 数量多于 backend".to_owned() });
    }
    Ok(states)
}

pub fn run_glm52_stage_chunk<B>(
    state: &mut Glm52StageState<B>,
    position: usize,
    hidden: B::Tensor,
    aux_hidden: &mut Option<B::Tensor>,
    aux_taps: &mut usize,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    rope: &RopeTable,
) -> Result<B::Tensor, BackendError>
where
    B: DsaStageBackend + ExpertPrefillBackend + Clone,
{
    let backend = state.backend.clone();
    let token_count = backend.token_rows(&hidden);
    let diagnose = token_count <= 16;
    let total_start_us = diagnose.then(crate::runtime::prefill_scheduler::stage_trace_timestamp_us);
    let total_started = diagnose.then(Instant::now);
    let layer_start = state.layer_start;
    let layer_end = layer_start + state.layers.len();
    backend.activate_stage().map_err(|error| BackendError::Compute { msg: format!("distributed stage {} 激活失败: {error:?}", backend.stage_label()) })?;
    let hidden = backend.move_tensor_to_stage(hidden)?;
    let layers = &state.layers;
    let cache = &mut state.cache;
    let dsa = &mut state.dsa;
    let mut experts = state.experts.lock().map_err(|_| BackendError::Compute { msg: format!("distributed stage {} expert 锁中毒", backend.stage_label()) })?;
    let projectors = &state.hidden_projectors;
    let setup_micros = total_started.map_or(0, |started| started.elapsed().as_micros());
    let mut layer_micros = Vec::with_capacity(usize::from(diagnose) * state.layers.len());
    let compute_started = diagnose.then(Instant::now);
    let hidden = glm52_prefill_stage_observed(
        &backend,
        cfg,
        token_count,
        layer_start,
        layer_end,
        hidden,
        &mut *experts,
        |_, _, _| Ok(()),
        |experts, layer, _kind, hidden| {
            let started = diagnose.then(Instant::now);
            let output = match &layers[layer - layer_start] {
                Glm52PrefillLayer::Dense(resident) => glm52_dense_prefill_layer(&backend, cfg, mla, resident, layer, Some(dsa), &hidden, rope, Some(cache), position),
                Glm52PrefillLayer::Moe(resident) => glm52_moe_prefill_layer(&backend, cfg, mla, resident, layer, experts, None, Some(dsa), &hidden, rope, Some(cache), position),
            }?;
            let output = backend.compact_stage_tensor(output)?;
            backend.profile_device_operator("glm_layer_tail")?;
            let retire_started = crate::runtime::prefill_scheduler::stage_event_trace_enabled().then(|| (Instant::now(), crate::runtime::prefill_scheduler::stage_trace_timestamp_us()));
            drop(hidden);
            if let Some((started, start_us)) = retire_started {
                let duration_us = started.elapsed().as_micros();
                if duration_us >= 20_000 {
                    crate::runtime::prefill_scheduler::record_stage_trace(format!(
                        "[stage-retire-trace] ts_us={start_us} phase=retire layer={layer} position={position} rows={token_count} duration_us={duration_us} complete_us={}",
                        crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                    ));
                }
            }
            if let Some(started) = started {
                layer_micros.push((layer, started.elapsed().as_micros()));
            }
            Ok(output)
        },
        |boundary, hidden| {
            for projector in projectors.iter().filter(|projector| projector.boundary() == boundary) {
                *aux_hidden = Some(projector.project_add(&backend, hidden, aux_hidden.take())?);
                *aux_taps += 1;
            }
            Ok(())
        },
    )?;
    let compute_micros = compute_started.map_or(0, |started| started.elapsed().as_micros());
    let aux_started = diagnose.then(Instant::now);
    if let Some(projected) = aux_hidden.take() {
        *aux_hidden = Some(backend.compact_stage_tensor(projected)?);
    }
    if let Some(total_started) = total_started {
        let total_micros = total_started.elapsed().as_micros();
        if total_micros >= 100_000 {
            let layers = layer_micros.iter().map(|(layer, micros)| format!("{layer}:{:.3}", *micros as f64 / 1000.0)).collect::<Vec<_>>().join(",");
            let layer_total = layer_micros.iter().map(|(_, micros)| *micros).sum::<u128>();
            eprintln!(
                "[glm52-stage-layers-slow] ts_us={} backend={} position={position} rows={token_count} layers={layer_start}..{layer_end} total_ms={:.3} setup_ms={:.3} layer_ms={:.3} finish_ms={:.3} aux_ms={:.3} per_layer=[{layers}] complete_us={}",
                total_start_us.unwrap_or_default(),
                backend.stage_label(),
                total_micros as f64 / 1000.0,
                setup_micros as f64 / 1000.0,
                layer_total as f64 / 1000.0,
                compute_micros.saturating_sub(layer_total) as f64 / 1000.0,
                aux_started.map_or(0, |started| started.elapsed().as_micros()) as f64 / 1000.0,
                crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
            );
        }
    }
    Ok(hidden)
}

pub struct Glm52StageValue<B>
where
    B: DsaStageBackend,
{
    pub decode: bool,
    pub verify: bool,
    pub truncate_to: Option<usize>,
    pub hidden: B::Tensor,
    pub selection: Option<B::DsaSelection>,
    pub aux_hidden: Option<B::Tensor>,
    pub aux_taps: usize,
}

#[derive(Clone, Copy)]
pub struct Glm52SchedulerPolicy {
    pub execution_slots: usize,
    pub decode_execution_slots: usize,
    pub pipeline_work_window: usize,
    pub prefill_admission_burst: usize,
    pub decode_batch_limit: usize,
    pub prefill_batch_limit: usize,
    pub profile_completion: bool,
}

impl Default for Glm52SchedulerPolicy {
    fn default() -> Self {
        Self { execution_slots: 1, decode_execution_slots: 1, pipeline_work_window: 8, prefill_admission_burst: 1, decode_batch_limit: 4, prefill_batch_limit: 4, profile_completion: false }
    }
}

impl<B> StageSchedulerHandle<Glm52StageValue<B>, Glm52StageState<B>>
where
    B: DsaStageBackend + ExpertPrefillBackend,
{
    pub fn submit_glm52(&self, session: usize, position: usize, decode: bool, hidden: B::Tensor, selection: Option<B::DsaSelection>) -> Result<(), BackendError> {
        self.submit_glm52_aux(session, position, decode, hidden, selection, None, 0)
    }

    pub fn submit_glm52_aux(&self, session: usize, position: usize, decode: bool, hidden: B::Tensor, selection: Option<B::DsaSelection>, aux_hidden: Option<B::Tensor>, aux_taps: usize) -> Result<(), BackendError> {
        self.submit(session, position, Glm52StageValue { decode, verify: false, truncate_to: decode.then_some(position), hidden, selection, aux_hidden, aux_taps })
    }

    pub fn submit_glm52_cohort(&self, cohort: u64, cohort_size: usize, session: usize, position: usize, hidden: B::Tensor, selection: Option<B::DsaSelection>) -> Result<(), BackendError> {
        self.submit_glm52_cohort_aux(cohort, cohort_size, session, position, hidden, selection, None, 0)
    }

    pub fn submit_glm52_cohort_aux(&self, cohort: u64, cohort_size: usize, session: usize, position: usize, hidden: B::Tensor, selection: Option<B::DsaSelection>, aux_hidden: Option<B::Tensor>, aux_taps: usize) -> Result<(), BackendError> {
        self.submit_cohort(cohort, cohort_size, std::iter::once((session, position, Glm52StageValue { decode: true, verify: false, truncate_to: Some(position), hidden, selection, aux_hidden, aux_taps })))
    }

    pub fn submit_glm52_wave_aux(&self, wave: u64, wave_size: usize, session: usize, position: usize, hidden: B::Tensor, selection: Option<B::DsaSelection>, aux_hidden: Option<B::Tensor>, aux_taps: usize) -> Result<(), BackendError> {
        self.submit_wave_member(wave, wave_size, session, position, Glm52StageValue { decode: true, verify: false, truncate_to: Some(position), hidden, selection, aux_hidden, aux_taps })
    }

    pub fn submit_glm52_verify(&self, session: usize, position: usize, hidden: B::Tensor, selection: Option<B::DsaSelection>) -> Result<(), BackendError> {
        self.submit_glm52_verify_aux(session, position, hidden, selection, None, 0)
    }

    pub fn submit_glm52_verify_aux(&self, session: usize, position: usize, hidden: B::Tensor, selection: Option<B::DsaSelection>, aux_hidden: Option<B::Tensor>, aux_taps: usize) -> Result<(), BackendError> {
        self.submit(session, position, Glm52StageValue { decode: false, verify: true, truncate_to: Some(position), hidden, selection, aux_hidden, aux_taps })
    }

    pub fn submit_glm52_verify_cohort(&self, cohort: u64, cohort_size: usize, session: usize, position: usize, hidden: B::Tensor, selection: Option<B::DsaSelection>) -> Result<(), BackendError> {
        self.submit_glm52_verify_cohort_aux(cohort, cohort_size, session, position, hidden, selection, None, 0)
    }

    pub fn submit_glm52_verify_cohort_aux(
        &self,
        cohort: u64,
        cohort_size: usize,
        session: usize,
        position: usize,
        hidden: B::Tensor,
        selection: Option<B::DsaSelection>,
        aux_hidden: Option<B::Tensor>,
        aux_taps: usize,
    ) -> Result<(), BackendError> {
        self.submit_cohort(cohort, cohort_size, std::iter::once((session, position, Glm52StageValue { decode: false, verify: true, truncate_to: Some(position), hidden, selection, aux_hidden, aux_taps })))
    }

    pub fn submit_glm52_verify_wave_aux(
        &self,
        wave: u64,
        wave_size: usize,
        session: usize,
        position: usize,
        hidden: B::Tensor,
        selection: Option<B::DsaSelection>,
        aux_hidden: Option<B::Tensor>,
        aux_taps: usize,
    ) -> Result<(), BackendError> {
        self.submit_wave_member(wave, wave_size, session, position, Glm52StageValue { decode: false, verify: true, truncate_to: Some(position), hidden, selection, aux_hidden, aux_taps })
    }
}

fn run_glm52_stage_batch<B>(
    states: &mut [Glm52StageState<B>],
    stage: usize,
    mut batch: Vec<(usize, usize, Glm52StageValue<B>)>,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    rope: &RopeTable,
) -> Result<Vec<(usize, usize, Glm52StageValue<B>)>, BackendError>
where
    B: DsaStageBackend + ExpertPrefillBackend + Clone,
{
    batch.sort_by_key(|(session, _, _)| *session);
    if batch.is_empty() || batch.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(BackendError::Compute { msg: "GLM stage microbatch 为空或 session 重复".to_owned() });
    }
    if batch.len() == 1 {
        let (session, position, mut value) = batch.pop().expect("长度已检查");
        let state = states.get_mut(session).ok_or_else(|| BackendError::Compute { msg: format!("GLM stage microbatch session={session} 越界") })?;
        state.backend.profile_stage_begin(true)?;
        if let Some(rows) = value.truncate_to {
            state.backend.truncate_stage_state(&mut state.cache, &mut state.dsa, rows).map_err(|error| BackendError::Compute { msg: format!("GLM stage={stage} session={session} position={position} truncate={rows}: {error:?}") })?;
        }
        state.backend.import_stage_selection(&mut state.dsa, value.selection.take())?;
        value.hidden = run_glm52_stage_chunk(state, position, value.hidden, &mut value.aux_hidden, &mut value.aux_taps, cfg, mla, rope)?;
        value.selection = state.backend.export_stage_selection(&state.dsa);
        state.backend.profile_stage_end()?;
        return Ok(vec![(session, position, value)]);
    }

    let first_session = batch[0].0;
    let first = states.get(first_session).ok_or_else(|| BackendError::Compute { msg: format!("GLM stage microbatch session={first_session} 越界") })?;
    let backend = first.backend.clone();
    let backend_label = backend.stage_label();
    let layer_start = first.layer_start;
    let layers = first.layers.clone();
    let experts = first.experts.clone();
    let projectors = first.hidden_projectors.clone();
    let layer_end = layer_start + layers.len();
    backend.profile_stage_begin(true)?;
    backend.activate_stage()?;

    let mut metadata = vec![None; states.len()];
    let mut hiddens = Vec::with_capacity(batch.len());
    let mut aux_hiddens = Vec::with_capacity(batch.len());
    for (session, position, mut value) in batch {
        let state = states.get_mut(session).ok_or_else(|| BackendError::Compute { msg: format!("GLM stage microbatch session={session} 越界") })?;
        if state.backend.stage_label() != backend_label || state.layer_start != layer_start || state.layers.len() != layers.len() {
            return Err(BackendError::Compute { msg: format!("GLM stage microbatch session={session} placement 不一致") });
        }
        if let Some(rows) = value.truncate_to {
            state.backend.truncate_stage_state(&mut state.cache, &mut state.dsa, rows).map_err(|error| BackendError::Compute { msg: format!("GLM batch stage={stage} session={session} position={position} truncate={rows}: {error:?}") })?;
        }
        backend.import_stage_selection(&mut state.dsa, value.selection.take())?;
        let hidden = backend.move_tensor_to_stage(value.hidden)?;
        let aux_hidden = value.aux_hidden.map(|hidden| backend.move_tensor_to_stage(hidden)).transpose()?;
        metadata[session] = Some((position, backend.token_rows(&hidden), value.decode, value.verify, value.truncate_to, value.aux_taps));
        hiddens.push(hidden);
        aux_hiddens.push(aux_hidden);
    }
    let refs = hiddens.iter().collect::<Vec<_>>();
    let mut hidden = backend.concat_token_rows(&refs)?;
    let present_aux = aux_hiddens.iter().filter(|hidden| hidden.is_some()).count();
    if present_aux != 0 && present_aux != aux_hiddens.len() {
        return Err(BackendError::Compute { msg: "GLM stage microbatch aux hidden 状态不一致".to_owned() });
    }
    let mut aux_hidden = if present_aux == 0 {
        None
    } else {
        let refs = aux_hiddens.iter().map(|hidden| hidden.as_ref().expect("数量已检查")).collect::<Vec<_>>();
        Some(backend.concat_token_rows(&refs)?)
    };
    let total_rows = backend.token_rows(&hidden);
    let mut experts = experts.lock().map_err(|_| BackendError::Compute { msg: format!("distributed batch stage {stage} expert 锁中毒") })?;
    for layer in layer_start..layer_end {
        let mut segments =
            states.iter_mut().enumerate().filter_map(|(session, state)| metadata[session].map(|(position, rows, _, _, _, _)| Glm52PrefillSegment { position, rows, cache: &mut state.cache, dsa: &mut state.dsa })).collect::<Vec<_>>();
        let output = match &layers[layer - layer_start] {
            Glm52PrefillLayer::Dense(resident) => glm52_dense_prefill_layer_segmented(&backend, cfg, mla, resident, layer, &hidden, rope, &mut segments),
            Glm52PrefillLayer::Moe(resident) => glm52_moe_prefill_layer_segmented(&backend, cfg, mla, resident, layer, &mut *experts, &hidden, rope, &mut segments),
        }
        .map_err(|error| BackendError::Compute { msg: format!("distributed batch stage={stage} backend={backend_label} layer={layer} rows={total_rows}: {error:?}") })?;
        hidden = backend.compact_stage_tensor(output)?;
        backend.profile_device_operator("glm_layer_tail")?;
        for projector in projectors.iter().filter(|projector| projector.boundary() == layer + 1) {
            aux_hidden = Some(projector.project_add(&backend, &hidden, aux_hidden.take())?);
            for metadata in metadata.iter_mut().flatten() {
                metadata.5 += 1;
            }
        }
    }
    aux_hidden = aux_hidden.map(|hidden| backend.compact_stage_tensor(hidden)).transpose()?;
    let mut offset = 0usize;
    let mut output = Vec::with_capacity(hiddens.len());
    for (session, meta) in metadata.into_iter().enumerate() {
        let Some((position, rows, decode, verify, truncate_to, aux_taps)) = meta else {
            continue;
        };
        let rows_hidden = backend.slice_token_rows(&hidden, offset, rows)?;
        let rows_aux_hidden = aux_hidden.as_ref().map(|hidden| backend.slice_token_rows(hidden, offset, rows)).transpose()?;
        offset += rows;
        output.push((session, position, Glm52StageValue { decode, verify, truncate_to, hidden: rows_hidden, selection: states[session].backend.export_stage_selection(&states[session].dsa), aux_hidden: rows_aux_hidden, aux_taps }));
    }
    backend.profile_stage_end()?;
    Ok(output)
}

fn run_glm52_dynamic_stage_batch<B>(
    states: &mut [Option<Glm52StageState<B>>],
    stage: usize,
    mut batch: Vec<(usize, usize, Glm52StageValue<B>)>,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    rope: &RopeTable,
) -> Result<Vec<(usize, usize, Glm52StageValue<B>)>, BackendError>
where
    B: DsaStageBackend + ExpertPrefillBackend + Clone,
{
    batch.sort_by_key(|(session, _, _)| *session);
    if batch.is_empty() || batch.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(BackendError::Compute { msg: "GLM dynamic stage microbatch 为空或 session 重复".to_owned() });
    }
    let diagnose = batch.iter().any(|(_, _, value)| value.decode || value.verify);
    let total_start_us = diagnose.then(crate::runtime::prefill_scheduler::stage_trace_timestamp_us);
    let total_started = diagnose.then(Instant::now);
    let position_min = batch.iter().map(|(_, position, _)| *position).min().unwrap_or(0);
    let position_max = batch.iter().map(|(_, position, _)| *position).max().unwrap_or(0);
    for (session, _, value) in &batch {
        if let Some(state) = states.get_mut(*session).and_then(Option::as_mut) {
            state.decode_active = value.decode || value.verify;
        }
    }
    let decode_parallelism = states.iter().filter(|state| state.as_ref().is_some_and(|state| state.decode_active)).count().max(1);
    let sessions = batch.iter().map(|(session, _, _)| *session).collect::<Vec<_>>();
    let mut compact = Vec::with_capacity(sessions.len());
    for &session in &sessions {
        let Some(slot) = states.get_mut(session) else {
            for (&restored_session, state) in sessions.iter().zip(compact.drain(..)) {
                states[restored_session] = Some(state);
            }
            return Err(BackendError::Compute { msg: format!("GLM dynamic stage session={session} 越界") });
        };
        let Some(state) = slot.take() else {
            for (&restored_session, state) in sessions.iter().zip(compact.drain(..)) {
                states[restored_session] = Some(state);
            }
            return Err(BackendError::Compute { msg: format!("GLM dynamic stage session={session} 尚未 Open") });
        };
        compact.push(state);
    }
    for state in &mut compact {
        state.backend.set_stage_decode_parallelism(&mut state.dsa, decode_parallelism);
    }
    let take_micros = total_started.map_or(0, |started| started.elapsed().as_micros());
    let move_started = diagnose.then(Instant::now);
    let mut ready_batch = Vec::with_capacity(batch.len());
    for (compact_session, (_, position, mut value)) in batch.into_iter().enumerate() {
        let backend = compact[compact_session].backend.clone();
        value.hidden = backend.move_tensor_to_stage_ordered(value.hidden)?;
        value.aux_hidden = value.aux_hidden.map(|hidden| backend.move_tensor_to_stage_ordered(hidden)).transpose()?;
        value.selection = value.selection.map(|selection| backend.move_selection_to_stage_ordered(selection)).transpose()?;
        ready_batch.push((compact_session, position, value));
    }
    let rows = ready_batch.iter().map(|(session, _, value)| compact[*session].backend.token_rows(&value.hidden)).sum::<usize>();
    let move_micros = move_started.map_or(0, |started| started.elapsed().as_micros());
    let stage_started = diagnose.then(Instant::now);
    let stage_result = run_glm52_stage_batch(&mut compact, stage, ready_batch, cfg, mla, rope);
    let stage_micros = stage_started.map_or(0, |started| started.elapsed().as_micros());
    let stabilize_started = diagnose.then(Instant::now);
    let result = stage_result.and_then(|output| {
        output
            .into_iter()
            .map(|(session, position, mut value)| {
                let backend = compact[session].backend.clone();
                value.hidden = backend.stabilize_stage_tensor(value.hidden)?;
                value.aux_hidden = value.aux_hidden.map(|hidden| backend.stabilize_stage_tensor(hidden)).transpose()?;
                value.selection = value.selection.map(|selection| backend.stabilize_stage_selection(selection)).transpose()?;
                Ok((session, position, value))
            })
            .collect::<Result<Vec<_>, BackendError>>()
    });
    let stabilize_micros = stabilize_started.map_or(0, |started| started.elapsed().as_micros());
    let restore_started = diagnose.then(Instant::now);
    for (&session, state) in sessions.iter().zip(compact.drain(..)) {
        states[session] = Some(state);
    }
    if let Some(total_started) = total_started {
        let total_micros = total_started.elapsed().as_micros();
        if total_micros >= 100_000 {
            eprintln!(
                "[glm52-stage-host-slow] ts_us={} stage={stage} sessions={} positions={position_min}..={position_max} rows={rows} total_ms={:.3} take_ms={:.3} move_ms={:.3} compute_ms={:.3} stabilize_ms={:.3} restore_ms={:.3} complete_us={}",
                total_start_us.unwrap_or_default(),
                sessions.len(),
                total_micros as f64 / 1000.0,
                take_micros as f64 / 1000.0,
                move_micros as f64 / 1000.0,
                stage_micros as f64 / 1000.0,
                stabilize_micros as f64 / 1000.0,
                restore_started.map_or(0, |started| started.elapsed().as_micros()) as f64 / 1000.0,
                crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
            );
        }
    }
    result.map(|mut output| {
        for item in &mut output {
            item.0 = sessions[item.0];
        }
        output
    })
}

pub fn run_glm52_stage_pipeline_stateful<B, I, O>(chunks: I, states: Vec<Glm52StageState<B>>, cfg: &Glm52Config, mla: &MlaSpec, rope: &RopeTable, policy: Glm52SchedulerPolicy, mut output: O) -> Result<Vec<Glm52StageState<B>>, BackendError>
where
    B: DsaStageBackend + ExpertPrefillBackend + Clone + Send + Sync,
    B::Tensor: Send,
    B::Weight: Send + Sync,
    B::Cache: Send,
    B::DsaState: Send,
    B::PrefillExperts: Send,
    I: IntoIterator<Item = Result<(usize, B::Tensor, Option<B::DsaSelection>), BackendError>>,
    O: FnMut(usize, B::Tensor, Option<B::DsaSelection>) -> Result<(), BackendError> + Send,
{
    let mut chunks = chunks.into_iter().collect::<Vec<_>>();
    if chunks.len() == 1 {
        let (position, hidden, selection) = chunks.pop().expect("长度已检查")?;
        let backends = states.iter().map(|state| state.backend.clone()).collect::<Vec<_>>();
        let (position, value, states) =
            run_single_stage_chain(&backends, states, position, Glm52StageValue { decode: false, verify: false, truncate_to: None, hidden, selection, aux_hidden: None, aux_taps: 0 }, |_, states, stage, batch| {
                run_glm52_dynamic_stage_batch(states, stage, batch, cfg, mla, rope)
            })?;
        output(position, value.hidden, value.selection)?;
        return Ok(states);
    }
    let mut chunks = chunks.into_iter();
    let ((), mut sessions) = drive_glm52_stream_stage_pipeline_stateful(vec![states], 1, cfg, mla, rope, policy, |scheduler| {
        let mut submitted = 0usize;
        for chunk in chunks.by_ref() {
            let (position, hidden, selection) = chunk?;
            scheduler.submit_glm52(0, position, false, hidden, selection)?;
            submitted += 1;
        }
        if submitted == 0 {
            return Err(BackendError::Compute { msg: "GLM stage pipeline 没有输入".to_owned() });
        }
        let mut completed = 0usize;
        while completed < submitted {
            match scheduler.try_recv()? {
                Some(StageSchedulerOutput::Work { session: 0, position, value, .. }) => {
                    output(position, value.hidden, value.selection)?;
                    completed += 1;
                }
                Some(StageSchedulerOutput::Work { session, .. }) => {
                    return Err(BackendError::Compute { msg: format!("GLM stage pipeline 收到越界 session={session}") });
                }
                Some(StageSchedulerOutput::Opened { session }) => {
                    return Err(BackendError::Compute { msg: format!("GLM stage pipeline 意外收到 Opened session={session}") });
                }
                Some(StageSchedulerOutput::Closed { session, .. }) => {
                    return Err(BackendError::Compute { msg: format!("GLM stage pipeline 意外收到 Closed session={session}") });
                }
                None => std::thread::yield_now(),
            }
        }
        Ok(())
    })?;
    sessions.pop().flatten().ok_or_else(|| BackendError::Compute { msg: "GLM stage pipeline 丢失 session state".to_owned() })
}

/// 连续流式 stage 流水线。输入闭包在专用 feeder 线程持续拉取并提交；输出在
/// drive 线程按完成顺序回调——两个方向并发，调用方在输出回调里产出 token、
/// 上游据此发来下一帧时不能让任何一侧阻塞另一侧。输入 Closed 或出错后等全部
/// 在途工作完成，再整体归还各 session 的 stage state。
pub fn run_glm52_stream_stage_pipeline_stateful<B, I, O>(
    chunks: I,
    states: Vec<Vec<Glm52StageState<B>>>,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    rope: &RopeTable,
    policy: Glm52SchedulerPolicy,
    mut output: O,
) -> Result<Vec<Vec<Glm52StageState<B>>>, BackendError>
where
    B: DsaStageBackend + ExpertPrefillBackend + Clone + Send + Sync,
    B::Tensor: Send,
    B::Weight: Send + Sync,
    B::Cache: Send,
    B::DsaState: Send,
    B::PrefillExperts: Send,
    I: FnMut() -> TokenStreamBatchPoll<(usize, usize, bool, B::Tensor, Option<B::DsaSelection>)> + Send,
    O: FnMut(usize, usize, bool, B::Tensor, Option<B::DsaSelection>) -> Result<(), BackendError> + Send,
{
    let session_count = states.len();
    if session_count == 0 {
        return Err(BackendError::Compute { msg: "GLM stream pipeline 没有 session".to_owned() });
    }
    let mut source = chunks;
    let ((), sessions) = drive_glm52_stream_stage_pipeline_stateful(states, session_count, cfg, mla, rope, policy, |scheduler| {
        // feeder 线程阻塞拉网络帧并经通道转发;提交与输出消费都在 drive 线程,
        // StageSchedulerHandle 的 Receiver 不是 Sync,不能跨线程共享。
        let (input_tx, input_rx) = std::sync::mpsc::channel::<Result<(usize, usize, bool, B::Tensor, Option<B::DsaSelection>), BackendError>>();
        let feeder_closed = std::thread::scope(|scope| {
            scope.spawn(move || {
                loop {
                    match source() {
                        TokenStreamBatchPoll::Ready(item) => {
                            if input_tx.send(item).is_err() {
                                return;
                            }
                        }
                        TokenStreamBatchPoll::Pending => std::thread::yield_now(),
                        TokenStreamBatchPoll::Closed => return,
                    }
                }
            });
            input_rx
        });
        let mut submitted = 0usize;
        let mut completed = 0usize;
        loop {
            // 输出优先:token 延迟决定上游下一帧的往返时间。
            if let Some(output_item) = scheduler.try_recv()? {
                match output_item {
                    StageSchedulerOutput::Work { session, position, value, .. } => {
                        output(session, position, value.decode, value.hidden, value.selection)?;
                        completed += 1;
                    }
                    StageSchedulerOutput::Opened { session } => {
                        return Err(BackendError::Compute { msg: format!("GLM stream pipeline 意外收到 Opened session={session}") });
                    }
                    StageSchedulerOutput::Closed { session, .. } => {
                        return Err(BackendError::Compute { msg: format!("GLM stream pipeline 意外收到 Closed session={session}") });
                    }
                }
                continue;
            }
            match feeder_closed.try_recv() {
                Ok(Ok((session, position, decode, hidden, selection))) => {
                    scheduler.submit_glm52(session, position, decode, hidden, selection)?;
                    submitted += 1;
                }
                Ok(Err(error)) => return Err(error),
                Err(std::sync::mpsc::TryRecvError::Empty) => std::thread::sleep(std::time::Duration::from_micros(50)),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    if completed == submitted {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            }
        }
        if submitted == 0 {
            return Err(BackendError::Compute { msg: "GLM stream pipeline 没有输入".to_owned() });
        }
        Ok(())
    })?;
    sessions.into_iter().map(|session| session.ok_or_else(|| BackendError::Compute { msg: "GLM stream pipeline 丢失 session state".to_owned() })).collect()
}

pub fn glm52_scheduler_config<B>(states: &[Vec<Glm52StageState<B>>], session_capacity: usize, cfg: &Glm52Config, policy: Glm52SchedulerPolicy) -> Result<StageSchedulerConfig, BackendError>
where
    B: DsaStageBackend + ExpertPrefillBackend,
{
    let first = states.first().ok_or_else(|| BackendError::Compute { msg: "GLM scheduler 没有初始 session".to_owned() })?;
    let free_bytes = first.iter().filter_map(|state| state.backend.stage_available_bytes().ok()).min().unwrap_or(0);
    let batch_work_limit = free_bytes.saturating_mul(3).saturating_div(4).max(cfg.hidden_size.saturating_mul(256));
    Ok(StageSchedulerConfig {
        session_capacity,
        batch_work_limit,
        execution_slots: policy.execution_slots.max(1).min(session_capacity),
        decode_execution_slots: policy.decode_execution_slots.max(1).min(session_capacity),
        pipeline_work_window: policy.pipeline_work_window,
        prefill_admission_burst: policy.prefill_admission_burst,
        decode_batch_limit: policy.decode_batch_limit.max(1).min(session_capacity),
        prefill_batch_limit: policy.prefill_batch_limit.max(1).min(session_capacity),
        profile_completion: policy.profile_completion,
    })
}

pub fn drive_glm52_stream_stage_pipeline_stateful<B, D, R>(
    states: Vec<Vec<Glm52StageState<B>>>,
    session_capacity: usize,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    rope: &RopeTable,
    policy: Glm52SchedulerPolicy,
    drive: D,
) -> Result<(R, SchedulerSessions<Glm52StageState<B>>), BackendError>
where
    B: DsaStageBackend + ExpertPrefillBackend + Clone + Send + Sync,
    B::Tensor: Send,
    B::Weight: Send + Sync,
    B::Cache: Send,
    B::DsaState: Send,
    B::PrefillExperts: Send,
    D: FnOnce(&StageSchedulerHandle<Glm52StageValue<B>, Glm52StageState<B>>) -> Result<R, BackendError>,
{
    let scheduler = glm52_scheduler_config(&states, session_capacity, cfg, policy)?;
    let first = states.first().ok_or_else(|| BackendError::Compute { msg: "GLM dynamic stream pipeline 没有 session".to_owned() })?;
    let backends = first.iter().map(|state| state.backend.clone()).collect::<Vec<_>>();
    let work_backend = backends.first().cloned().ok_or_else(|| BackendError::Compute { msg: "GLM dynamic stream pipeline 没有 stage".to_owned() })?;
    drive_stage_scheduler_with_batch_class(
        backends,
        states,
        scheduler,
        |value: &Glm52StageValue<B>| {
            // verify 行仍按 session 独立拥有 KV/DSA；这里只允许调度器把当前
            // 已经 ready 的不同 session 行组成 stage-local batch。各 stage 可
            // 独立拆分，不建立跨 stage cohort，也不等待后续行凑批。
            let bytes_per_element = if value.decode || value.verify { 16 } else { 256 };
            work_backend.token_rows(&value.hidden).saturating_mul(cfg.hidden_size).saturating_mul(bytes_per_element).max(1)
        },
        |value| {
            if value.decode || value.verify { StageWorkKind::Decode } else { StageWorkKind::Prefill }
        },
        |value| usize::from(work_backend.token_rows(&value.hidden) > 1),
        |_, states, stage, batch| run_glm52_dynamic_stage_batch(states, stage, batch, cfg, mla, rope),
        drive,
    )
}

pub fn last_token_row<B>(backend: &B, hidden: B::Tensor) -> Result<B::Tensor, BackendError>
where
    B: StageTensorBackend,
{
    let rows = backend.token_rows(&hidden);
    if rows == 0 {
        return Err(BackendError::Compute { msg: "stage hidden 没有行".to_owned() });
    }
    backend.slice_token_rows(&hidden, rows - 1, 1)
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_prefill_layers<B>(backends: &[B], layer_ends: &[usize], layer_start: usize, layer_end: usize, cfg: &Glm52Config, mla: &MlaSpec, weights: &Glm52Weights) -> Result<Vec<Glm52PrefillLayer<B::Weight>>, BackendError>
where
    B: DsaStageBackend + ExpertPrefillBackend,
{
    for backend in backends {
        backend.warmup_stage()?;
    }
    let mut layers = Vec::with_capacity(layer_end - layer_start);
    for layer in layer_start..layer_end {
        let placement = layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| BackendError::Compute { msg: format!("L{layer} 没有 prefill backend") })?;
        let backend = &backends[placement];
        backend.activate_stage()?;
        if layer < cfg.dense_layer_count {
            let resident = load_prepare_dense_prefill_layer(backend, cfg, mla, weights, layer, false).map_err(|error| BackendError::Compute { msg: format!("准备 prefill L{layer}: {error:?}") })?;
            layers.push(Glm52PrefillLayer::Dense(resident));
        } else {
            let resident = load_prepare_moe_prefill_layer(backend, cfg, mla, weights, layer, false).map_err(|error| BackendError::Compute { msg: format!("准备 prefill L{layer}: {error:?}") })?;
            layers.push(Glm52PrefillLayer::Moe(resident));
        }
    }
    Ok(layers)
}

fn weight_error(error: impl std::fmt::Debug) -> BackendError {
    BackendError::Compute { msg: format!("加载 GLM stage 权重: {error:?}") }
}

pub fn build_pp_prefill_chunk<B>(backend: &B, weights: &Glm52Weights, cfg: &Glm52Config, tokens: &[u32], position: usize) -> Result<B::Tensor, BackendError>
where
    B: StageTensorBackend,
{
    let values = weights.embedding_rows(tokens).map_err(weight_error)?;
    backend.stage_tensor_from_f32(values, tokens.len(), cfg.hidden_size).map_err(|error| BackendError::Compute { msg: format!("准备 position={position} embedding: {error:?}") })
}

pub fn build_pp_prefill_chunks<B>(backend: &B, weights: &Glm52Weights, cfg: &Glm52Config, tokens: &[u32], suffix_start: usize, policy: crate::runtime::prefill::AdaptiveChunkPolicy) -> Result<Vec<(usize, B::Tensor)>, BackendError>
where
    B: StageTensorBackend,
{
    if suffix_start >= tokens.len() {
        return Ok(Vec::new());
    }
    let mut chunks = Vec::new();
    let mut position = suffix_start;
    while position < tokens.len() {
        let chunk_size = policy.chunk_size(suffix_start, position);
        let end = (position + chunk_size).min(tokens.len());
        let embedding = build_pp_prefill_chunk(backend, weights, cfg, &tokens[position..end], position)?;
        chunks.push((position, embedding));
        position = end;
    }
    Ok(chunks)
}
