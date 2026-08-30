//! GLM-5.2 DSpark 的 ROCm 张量搬运与 capture projection 组合。

use std::{path::Path, sync::Arc, time::Instant};

use super::dspark::{Glm52DsparkBackend, Glm52DsparkCacheRuntime, Glm52DsparkDraftBatch, Glm52DsparkRuntime, prepare_dspark_matrix};
use crate::{
    backend::{
        Backend, BackendError,
        rocm::{RocmContext, RocmTensor, RocmWeight},
    },
    runtime::{
        dspark::DsparkTargetCache,
        glm52::stage::{Glm52HiddenProjector, Glm52StageState},
    },
    weight::{LmHeadQuantization, ResidentWeightQuantization, model::glm52_dspark::Glm52DsparkCheckpoint},
};

pub type RocmDsparkDraftBatch<'a> = Glm52DsparkDraftBatch<'a, RocmContext>;

/// GPU proposal 保留完整 runtime；CPU proposal 只保留 aux norm 与 target K/V
/// projector，避免同一份 drafter 的 Q/O/FFN/head 同时常驻 CPU 和 GPU。
pub enum RocmDsparkRuntime {
    Full(Glm52DsparkRuntime<RocmContext>),
    Cache(Glm52DsparkCacheRuntime<RocmContext>),
}

impl RocmDsparkRuntime {
    pub fn load(
        backend: &RocmContext,
        root: &Path,
        max_seq_len: usize,
        draft_tokens: usize,
        confidence_threshold: Option<f32>,
        lm_head_quantization: LmHeadQuantization,
        weight_quantization: ResidentWeightQuantization,
    ) -> Result<Self, BackendError> {
        Glm52DsparkRuntime::load(backend, root, max_seq_len, draft_tokens, confidence_threshold, lm_head_quantization, weight_quantization).map(Self::Full)
    }

    pub fn load_cache(backend: &RocmContext, root: &Path, max_seq_len: usize, weight_quantization: ResidentWeightQuantization) -> Result<Self, BackendError> {
        Glm52DsparkCacheRuntime::load(backend, root, max_seq_len, weight_quantization).map(Self::Cache)
    }

    pub fn capture_count(&self) -> usize {
        match self {
            Self::Full(runtime) => runtime.capture_count(),
            Self::Cache(runtime) => runtime.capture_count(),
        }
    }

    pub fn target_history_window(&self) -> Option<usize> {
        match self {
            Self::Full(runtime) => runtime.target_history_window(),
            Self::Cache(runtime) => runtime.target_history_window(),
        }
    }

    pub fn target_layer_count(&self) -> usize {
        match self {
            Self::Full(runtime) => runtime.target_layer_count(),
            Self::Cache(runtime) => runtime.target_layer_count(),
        }
    }

    pub fn target_columns(&self) -> usize {
        match self {
            Self::Full(runtime) => runtime.target_columns(),
            Self::Cache(runtime) => runtime.target_columns(),
        }
    }

    pub fn normalize_aux_hidden(&self, backend: &RocmContext, projected: &RocmTensor) -> Result<RocmTensor, BackendError> {
        match self {
            Self::Full(runtime) => runtime.normalize_aux_hidden(backend, projected),
            Self::Cache(runtime) => runtime.normalize_aux_hidden(backend, projected),
        }
    }

    pub fn warm_target_cache(&self, backend: &RocmContext, cache: &mut DsparkTargetCache<RocmTensor>, target_hidden: &RocmTensor, target_position: usize) -> Result<(), BackendError> {
        match self {
            Self::Full(runtime) => runtime.warm_target_cache(backend, cache, target_hidden, target_position),
            Self::Cache(runtime) => runtime.warm_target_cache(backend, cache, target_hidden, target_position),
        }
    }

    pub fn draft_batch(&self, backend: &RocmContext, batch: &mut [RocmDsparkDraftBatch<'_>]) -> Result<(), BackendError> {
        match self {
            Self::Full(runtime) => runtime.draft_batch(backend, batch),
            Self::Cache(_) => Err(BackendError::Compute { msg: "CPU DSpark cache runtime 不能执行 GPU proposal".to_owned() }),
        }
    }
}

impl Glm52DsparkBackend for RocmContext {
    fn dspark_tensor_from_bf16_bits(&self, values: Vec<u16>, rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        self.tensor_from_bf16_bits(values, rows, cols)
    }

    fn dspark_tensor_as_f32(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError> {
        self.tensor_as_f32(tensor)
    }

    fn dspark_tensor_to_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, BackendError> {
        self.tensor_to_f32(tensor)
    }

    fn dspark_argmax_add_rows(&self, logits: &Self::Tensor, rows: &[u32], bias: &Self::Tensor) -> Result<Vec<u32>, BackendError> {
        self.argmax_add_rows(logits, rows, bias)
    }

    fn dspark_profile_enabled(&self) -> bool {
        crate::kernel::rocm::hip::options().kernel_profile
    }
}

pub fn attach_dspark_projections(states: &mut [Glm52StageState<RocmContext>], root: &Path, verifier_layer_count: usize, quantization: ResidentWeightQuantization) -> Result<(), BackendError> {
    let checkpoint = Glm52DsparkCheckpoint::open(root).map_err(compute)?;
    let plan = checkpoint.config.capture_plan(verifier_layer_count).map_err(compute)?;
    for (index, &boundary) in plan.boundaries().iter().enumerate() {
        let Some(state) = states.iter_mut().find(|state| boundary > state.layer_start && boundary <= state.layer_start + state.layers.len()) else {
            continue;
        };
        let weight = prepare_dspark_matrix(&state.backend, checkpoint.aux_projection(index).map_err(compute)?, quantization)?;
        state.hidden_projectors.push(Arc::new(RocmDsparkProjection { boundary, weight }));
    }
    Ok(())
}

pub struct RocmDsparkProjection {
    pub boundary: usize,
    weight: RocmWeight,
}

impl RocmDsparkProjection {
    pub fn load(backend: &RocmContext, root: &Path, boundary: usize, quantization: ResidentWeightQuantization) -> Result<Option<Self>, BackendError> {
        let checkpoint = Glm52DsparkCheckpoint::open(root).map_err(compute)?;
        let Some(index) = checkpoint.config.aux_hidden_state_layer_ids.iter().position(|&candidate| candidate == boundary) else {
            return Ok(None);
        };
        Ok(Some(Self { boundary, weight: prepare_dspark_matrix(backend, checkpoint.aux_projection(index).map_err(compute)?, quantization)? }))
    }

    pub fn project_add(&self, backend: &RocmContext, hidden: &RocmTensor, accumulated: Option<&RocmTensor>) -> Result<RocmTensor, BackendError> {
        let projected = backend.linear(hidden, &self.weight)?;
        match accumulated {
            Some(accumulated) => backend.add(accumulated, &projected),
            None => Ok(projected),
        }
    }
}

impl Glm52HiddenProjector<RocmContext> for RocmDsparkProjection {
    fn boundary(&self) -> usize {
        self.boundary
    }

    fn project_add(&self, backend: &RocmContext, hidden: &RocmTensor, accumulated: Option<RocmTensor>) -> Result<RocmTensor, BackendError> {
        let trace_started = crate::runtime::prefill_scheduler::stage_event_trace_enabled().then(|| (Instant::now(), crate::runtime::prefill_scheduler::stage_trace_timestamp_us()));
        let linear_started = trace_started.map(|_| Instant::now());
        let projected = backend.linear(hidden, &self.weight)?;
        let linear_us = linear_started.map_or(0, |started| started.elapsed().as_micros());
        let (output, add_us, retire_us) = match accumulated {
            Some(accumulated) => {
                let add_started = trace_started.map(|_| Instant::now());
                let output = backend.add(&accumulated, &projected)?;
                let add_us = add_started.map_or(0, |started| started.elapsed().as_micros());
                let retire_started = trace_started.map(|_| Instant::now());
                drop(projected);
                drop(accumulated);
                let retire_us = retire_started.map_or(0, |started| started.elapsed().as_micros());
                (output, add_us, retire_us)
            }
            None => (projected, 0, 0),
        };
        if let Some((started, start_us)) = trace_started {
            let total_us = started.elapsed().as_micros();
            if total_us >= 20_000 {
                crate::runtime::prefill_scheduler::record_stage_trace(format!(
                    "[dspark-projection-trace] ts_us={start_us} phase=project boundary={} rows={} linear_us={linear_us} add_us={add_us} retire_us={retire_us} total_us={total_us} complete_us={}",
                    self.boundary,
                    hidden.rows,
                    crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                ));
            }
        }
        Ok(output)
    }
}

fn compute(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}
