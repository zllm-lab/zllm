//! Gated DeltaNet 规格与 session state。
//!
//! recurrent state 与 full-attention KV cache 的更新语义完全不同，二者只共享
//! session 生命周期，不共享存储抽象。

use crate::{
    attention::recurrent_state::RecurrentState,
    backend::{Backend, BackendError},
};

/// GDN 输出门激活:Qwen3.5/3.6/Ornith 用 silu(z·sigmoid(z)),
/// Qwen4-Exp 改为纯 sigmoid(z)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnOutputGate {
    Silu,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatedDeltaNetSpec {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
    pub rms_eps: f32,
    pub output_gate: GdnOutputGate,
}

impl GatedDeltaNetSpec {
    pub fn key_dim(self) -> usize {
        self.key_heads * self.key_head_dim
    }

    pub fn value_dim(self) -> usize {
        self.value_heads * self.value_head_dim
    }

    pub fn conv_dim(self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }

    pub fn recurrent_elements(self) -> usize {
        self.value_heads * self.key_head_dim * self.value_head_dim
    }

    pub fn conv_state_elements(self) -> usize {
        self.conv_dim() * self.conv_kernel
    }

    pub fn validate(self) -> Result<(), String> {
        if self.key_heads == 0 || self.value_heads == 0 || self.key_head_dim == 0 || self.value_head_dim == 0 || self.conv_kernel == 0 {
            return Err("Gated DeltaNet 维度不能为 0".to_owned());
        }
        if !self.value_heads.is_multiple_of(self.key_heads) {
            return Err(format!("Gated DeltaNet value_heads={} 不能整除 key_heads={}", self.value_heads, self.key_heads));
        }
        if !self.rms_eps.is_finite() || self.rms_eps <= 0.0 {
            return Err(format!("Gated DeltaNet rms_eps={} 非法", self.rms_eps));
        }
        Ok(())
    }
}

/// Q/K head 扩展到 value head 时的物理布局。
///
/// 原始 HF/MLX safetensors 使用连续分组；llama.cpp 转换 GGUF 时重排成
/// 交错布局。布局必须跟权重来源绑定，不能只按模型名推断。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedDeltaNetHeadLayout {
    Grouped,
    Tiled,
}

impl GatedDeltaNetHeadLayout {
    pub fn key_head_for_value(self, spec: &GatedDeltaNetSpec, value_head: usize) -> usize {
        match self {
            Self::Grouped => value_head / (spec.value_heads / spec.key_heads),
            Self::Tiled => value_head % spec.key_heads,
        }
    }
}

#[derive(Clone, Copy)]
pub struct GatedDeltaNetInputs<'a, T> {
    pub qkv: &'a T,
    pub z: &'a T,
    pub alpha: &'a T,
    pub beta: &'a T,
}

#[derive(Clone, Copy)]
pub struct GatedDeltaNetWeightsRef<'a, W> {
    pub conv: &'a W,
    pub a_log: &'a W,
    pub dt_bias: &'a W,
    pub norm: &'a W,
}

pub trait GatedDeltaNetStorage {
    fn allocated_bytes(&self) -> usize;
}

pub trait GatedDeltaNetKernel: Backend {
    type GatedDeltaNetStorage: GatedDeltaNetStorage;

    fn allocate_gated_delta_net_storage(&self, spec: &GatedDeltaNetSpec) -> Result<Self::GatedDeltaNetStorage, BackendError>;

    fn gated_delta_net_fused(
        &self,
        storage: &mut Self::GatedDeltaNetStorage,
        inputs: GatedDeltaNetInputs<'_, Self::Tensor>,
        weights: GatedDeltaNetWeightsRef<'_, Self::Weight>,
        spec: &GatedDeltaNetSpec,
    ) -> Result<Self::Tensor, BackendError>;

    fn gated_delta_net_fused_layout(
        &self,
        storage: &mut Self::GatedDeltaNetStorage,
        inputs: GatedDeltaNetInputs<'_, Self::Tensor>,
        weights: GatedDeltaNetWeightsRef<'_, Self::Weight>,
        head_layout: GatedDeltaNetHeadLayout,
        spec: &GatedDeltaNetSpec,
    ) -> Result<Self::Tensor, BackendError> {
        match head_layout {
            GatedDeltaNetHeadLayout::Tiled => self.gated_delta_net_fused(storage, inputs, weights, spec),
            GatedDeltaNetHeadLayout::Grouped => Err(BackendError::Compute { msg: "当前 backend 未实现 grouped Gated DeltaNet head 布局".to_owned() }),
        }
    }
}

pub struct GatedDeltaNetState<S> {
    inner: RecurrentState<S, GatedDeltaNetSpec>,
    head_layout: GatedDeltaNetHeadLayout,
}

impl<S> GatedDeltaNetState<S> {
    pub fn new(layer_count: usize, spec: GatedDeltaNetSpec) -> Result<Self, BackendError> {
        Self::with_head_layout(layer_count, spec, GatedDeltaNetHeadLayout::Tiled)
    }

    pub fn with_head_layout(layer_count: usize, spec: GatedDeltaNetSpec, head_layout: GatedDeltaNetHeadLayout) -> Result<Self, BackendError> {
        spec.validate().map_err(|msg| BackendError::Compute { msg })?;
        Ok(Self { inner: RecurrentState::new("Gated DeltaNet", layer_count, spec), head_layout })
    }

    pub fn layer_count(&self) -> usize {
        self.inner.layer_count()
    }

    pub fn layer_position(&self, layer: usize) -> Option<usize> {
        self.inner.layer_position(layer)
    }

    pub fn head_layout(&self) -> GatedDeltaNetHeadLayout {
        self.head_layout
    }

    pub fn layer_storage(&self, layer: usize) -> Option<&S> {
        self.inner.layer_storage(layer)
    }

    pub fn layer_storage_mut(&mut self, layer: usize) -> Option<&mut S> {
        self.inner.layer_storage_mut(layer)
    }

    /// 快照恢复：注入层 storage 并把 position 推进到快照时的 token 数。
    pub fn restore_layer(&mut self, layer: usize, position: usize, storage: S) -> Result<(), BackendError> {
        self.inner.restore_layer(layer, position, storage)
    }

    /// 字节级快照恢复后把 position 对齐到快照时的 token 数。
    pub fn rewind_layer(&mut self, layer: usize, position: usize) -> Result<(), BackendError> {
        self.inner.rewind_layer(layer, position)
    }
}

impl<S: GatedDeltaNetStorage> GatedDeltaNetState<S> {
    pub fn allocated_bytes(&self) -> usize {
        self.inner.allocated_bytes(GatedDeltaNetStorage::allocated_bytes)
    }
}

pub fn gated_delta_net<B: GatedDeltaNetKernel>(
    backend: &B,
    state: &mut GatedDeltaNetState<B::GatedDeltaNetStorage>,
    layer: usize,
    position: usize,
    inputs: GatedDeltaNetInputs<'_, B::Tensor>,
    weights: GatedDeltaNetWeightsRef<'_, B::Weight>,
    spec: &GatedDeltaNetSpec,
) -> Result<B::Tensor, BackendError> {
    spec.validate().map_err(|msg| BackendError::Compute { msg })?;
    let rows = backend.token_rows(inputs.qkv);
    if rows == 0
        || backend.token_cols(inputs.qkv) != spec.conv_dim()
        || backend.token_rows(inputs.z) != rows
        || backend.token_cols(inputs.z) != spec.value_dim()
        || backend.token_rows(inputs.alpha) != rows
        || backend.token_cols(inputs.alpha) != spec.value_heads
        || backend.token_rows(inputs.beta) != rows
        || backend.token_cols(inputs.beta) != spec.value_heads
    {
        return Err(BackendError::Compute { msg: "Gated DeltaNet input shape 与 spec 不一致".to_owned() });
    }
    let head_layout = state.head_layout;
    let layer_state = state.inner.layer_mut(layer, position, spec, || backend.allocate_gated_delta_net_storage(spec))?;
    let output = backend.gated_delta_net_fused_layout(&mut layer_state.storage, inputs, weights, head_layout, spec)?;
    if backend.token_rows(&output) != rows || backend.token_cols(&output) != spec.value_dim() {
        return Err(BackendError::Compute { msg: "Gated DeltaNet fused kernel 输出 shape 异常".to_owned() });
    }
    layer_state.position += rows;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ornith_state_budget_matches_real_shape() {
        let spec = GatedDeltaNetSpec { key_heads: 16, value_heads: 32, key_head_dim: 128, value_head_dim: 128, conv_kernel: 4, rms_eps: 1e-6, output_gate: GdnOutputGate::Silu };
        assert_eq!(spec.key_dim(), 2_048);
        assert_eq!(spec.value_dim(), 4_096);
        assert_eq!(spec.conv_dim(), 8_192);
        assert_eq!(spec.recurrent_elements(), 524_288);
        assert_eq!(spec.conv_state_elements(), 32_768);
    }

    #[test]
    fn head_layout_matches_source_weight_order() {
        let spec = GatedDeltaNetSpec { key_heads: 16, value_heads: 32, key_head_dim: 128, value_head_dim: 128, conv_kernel: 4, rms_eps: 1e-6, output_gate: GdnOutputGate::Silu };
        let heads = [0, 1, 15, 16, 17, 31];
        assert_eq!(heads.map(|head| GatedDeltaNetHeadLayout::Grouped.key_head_for_value(&spec, head)), [0, 0, 7, 8, 8, 15]);
        assert_eq!(heads.map(|head| GatedDeltaNetHeadLayout::Tiled.key_head_for_value(&spec, head)), [0, 1, 15, 0, 1, 15]);
    }
}
