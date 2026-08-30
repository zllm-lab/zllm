//! Kimi Delta Attention(KDA)规格、state 生命周期与 backend capability。

use crate::{
    attention::recurrent_state::RecurrentState,
    backend::{Backend, BackendError},
};

/// KDA 的架构常量。recurrent state 与短卷积 state 的存储由 backend 管理。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KdaSpec {
    pub num_heads: usize,
    pub head_dim: usize,
    pub short_conv_kernel_size: usize,
    pub use_full_rank_gate: bool,
    pub gate_lower_bound: Option<f32>,
    pub use_qk_l2norm: bool,
    pub output_norm_eps: f32,
}

impl KdaSpec {
    pub fn projection_size(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn recurrent_state_elements(&self) -> usize {
        self.num_heads * self.head_dim * self.head_dim
    }

    /// q/k/v 三条短卷积各保留 kernel_size - 1 个历史位置。
    pub fn conv_state_elements(&self) -> usize {
        3 * self.projection_size() * self.short_conv_kernel_size.saturating_sub(1)
    }

    pub fn validate(self) -> Result<(), String> {
        if self.num_heads == 0 || self.head_dim == 0 || self.short_conv_kernel_size == 0 {
            return Err("KDA 维度不能为 0".to_owned());
        }
        if !self.output_norm_eps.is_finite() || self.output_norm_eps <= 0.0 {
            return Err(format!("KDA output_norm_eps={} 非法", self.output_norm_eps));
        }
        if self.gate_lower_bound.is_some_and(|bound| !bound.is_finite() || bound >= 0.0) {
            return Err(format!("KDA gate_lower_bound={:?} 必须是有限负数", self.gate_lower_bound));
        }
        Ok(())
    }
}

pub struct KdaInputs<'a, T> {
    pub query: &'a T,
    pub key: &'a T,
    pub value: &'a T,
    pub decay: &'a T,
    pub beta: &'a T,
    pub output_gate: &'a T,
}

pub struct KdaWeightsRef<'a, W> {
    pub query_conv: &'a W,
    pub key_conv: &'a W,
    pub value_conv: &'a W,
    pub a_log: &'a W,
    pub dt_bias: &'a W,
    pub output_norm: &'a W,
}

pub trait KdaStorage {
    fn allocated_bytes(&self) -> usize;
}

pub trait KdaKernel: Backend {
    type KdaStorage: KdaStorage;

    fn allocate_kda_storage(&self, spec: &KdaSpec) -> Result<Self::KdaStorage, BackendError>;

    fn kda_fused(&self, storage: &mut Self::KdaStorage, inputs: KdaInputs<'_, Self::Tensor>, weights: KdaWeightsRef<'_, Self::Weight>, spec: &KdaSpec) -> Result<Self::Tensor, BackendError>;
}

pub struct KdaState<S> {
    inner: RecurrentState<S, KdaSpec>,
}

impl<S> KdaState<S> {
    pub fn new(layer_count: usize, spec: KdaSpec) -> Result<Self, BackendError> {
        spec.validate().map_err(|msg| BackendError::Compute { msg })?;
        Ok(Self { inner: RecurrentState::new("KDA", layer_count, spec) })
    }

    pub fn layer_count(&self) -> usize {
        self.inner.layer_count()
    }

    pub fn layer_position(&self, layer: usize) -> Option<usize> {
        self.inner.layer_position(layer)
    }

    /// speculative verify 的字节级快照支持:读/替换/回退某层 storage。
    pub fn layer_storage(&self, layer: usize) -> Option<&S> {
        self.inner.layer_storage(layer)
    }

    pub fn restore_layer(&mut self, layer: usize, position: usize, storage: S) -> Result<(), BackendError> {
        self.inner.restore_layer(layer, position, storage)
    }

    pub fn rewind_layer(&mut self, layer: usize, position: usize) -> Result<(), BackendError> {
        self.inner.rewind_layer(layer, position)
    }
}

impl<S: KdaStorage> KdaState<S> {
    pub fn allocated_bytes(&self) -> usize {
        self.inner.allocated_bytes(KdaStorage::allocated_bytes)
    }
}

pub fn kda<B: KdaKernel>(backend: &B, state: &mut KdaState<B::KdaStorage>, layer: usize, position: usize, inputs: KdaInputs<'_, B::Tensor>, weights: KdaWeightsRef<'_, B::Weight>, spec: &KdaSpec) -> Result<B::Tensor, BackendError> {
    spec.validate().map_err(|msg| BackendError::Compute { msg })?;
    let rows = backend.token_rows(inputs.query);
    let projection_size = spec.projection_size();
    let projection_shape_ok = |tensor: &B::Tensor| backend.token_rows(tensor) == rows && backend.token_cols(tensor) == projection_size;
    if rows == 0
        || !projection_shape_ok(inputs.key)
        || !projection_shape_ok(inputs.value)
        || !projection_shape_ok(inputs.decay)
        || !projection_shape_ok(inputs.output_gate)
        || backend.token_cols(inputs.query) != projection_size
        || backend.token_rows(inputs.beta) != rows
        || backend.token_cols(inputs.beta) != spec.num_heads
    {
        return Err(BackendError::Compute { msg: "KDA input shape 与 spec 不一致".to_owned() });
    }
    let layer_state = state.inner.layer_mut(layer, position, spec, || backend.allocate_kda_storage(spec))?;
    let output = backend.kda_fused(&mut layer_state.storage, inputs, weights, spec)?;
    if backend.token_rows(&output) != rows || backend.token_cols(&output) != projection_size {
        return Err(BackendError::Compute { msg: "KDA fused kernel 输出 shape 异常".to_owned() });
    }
    layer_state.position += rows;
    Ok(output)
}
