//! 单专家 dense FFN 规格与算法流。

#[derive(Debug)]
pub enum Activation {
    Silu,
    /// DeepSeek-V4 SwiGLU：gate 只限制上界，up 限制到 `[-limit, limit]`。
    SiluClamped {
        limit: f32,
    },
    /// Kimi SiTU:beta*tanh(gate/beta)*sigmoid(gate)，up 可选线性 tanh 限幅。
    Situ {
        beta: f32,
        linear_beta: Option<f32>,
    },
    SwigluOai {
        alpha: f32,
        limit: f32,
    },
    GeluTanh,
}

#[derive(Debug)]
pub struct DenseMlpSpec {
    pub intermediate_size: usize,
    pub activation: Activation,
}

use crate::backend::{Backend, BackendError};

pub struct DenseMlpWeightsRef<'a, W> {
    pub gate: &'a W,
    pub up: &'a W,
    pub down: &'a W,
}

/// Dense FFN 算法流：down(activation(gate) * up)。
pub fn decode<B: Backend>(backend: &B, spec: &DenseMlpSpec, weights: DenseMlpWeightsRef<'_, B::Weight>, input: &B::Tensor) -> Result<B::Tensor, BackendError> {
    backend.gated_mlp(input, weights.gate, weights.up, weights.down, &spec.activation)
}

/// 量化/诊断观测激活后的 down 输入；生产路径传入 no-op，不改变算法流。
pub fn forward_observed<B, O>(backend: &B, spec: &DenseMlpSpec, weights: DenseMlpWeightsRef<'_, B::Weight>, input: &B::Tensor, mut observe_down_input: O) -> Result<B::Tensor, BackendError>
where
    B: Backend,
    O: FnMut(&B::Tensor),
{
    let activated = backend.gated_linear(input, weights.gate, weights.up, &spec.activation)?;
    observe_down_input(&activated);
    backend.linear(&activated, weights.down)
}
