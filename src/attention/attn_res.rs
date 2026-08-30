//! AttnRes 的跨层残差选择规格与平台能力。

use crate::backend::{Backend, BackendError};

pub trait AttnResBackend: Backend {
    /// 对历史 block residual 与当前 prefix 做 RMS 打分和 softmax 加权。
    fn attn_res_mix(&self, current: &Self::Tensor, block_residuals: &[Self::Tensor], norm_weight: &Self::Weight, projection_weight: &Self::Weight, eps: f32) -> Result<Self::Tensor, BackendError>;
}

/// 一个 prefill/decode round 内的 block residual；不跨 token session 持久化。
pub struct AttnResState<T> {
    block_size: usize,
    next_layer: usize,
    residuals: Vec<T>,
}

impl<T> AttnResState<T> {
    pub fn new(block_size: usize) -> Result<Self, BackendError> {
        if block_size == 0 {
            return Err(BackendError::Compute { msg: "AttnRes block_size 不能为 0".to_owned() });
        }
        Ok(Self { block_size, next_layer: 0, residuals: Vec::new() })
    }

    pub fn begin_layer(&self, layer: usize) -> Result<(), BackendError> {
        if layer != self.next_layer {
            return Err(BackendError::Compute { msg: format!("AttnRes layer 不连续: next={}, input={layer}", self.next_layer) });
        }
        Ok(())
    }

    pub fn finish_layer(&mut self) {
        self.next_layer += 1;
    }

    pub fn next_layer(&self) -> usize {
        self.next_layer
    }

    pub fn is_block_start(&self, layer: usize) -> bool {
        layer.is_multiple_of(self.block_size)
    }

    pub fn push(&mut self, residual: T) {
        self.residuals.push(residual);
    }

    pub fn residuals(&self) -> &[T] {
        &self.residuals
    }

    pub fn block_count(&self) -> usize {
        self.residuals.len()
    }

    /// 完整模型轮次结束后释放跨层 scratch；不允许掩盖截断执行。
    pub fn finish_round(&mut self, layer_count: usize) -> Result<(), BackendError> {
        if self.next_layer != layer_count {
            return Err(BackendError::Compute { msg: format!("AttnRes round 未完成: processed={}, expected={layer_count}", self.next_layer) });
        }
        self.next_layer = 0;
        self.residuals.clear();
        Ok(())
    }
}

pub fn mix<B: AttnResBackend>(backend: &B, state: &AttnResState<B::Tensor>, current: &B::Tensor, norm_weight: &B::Weight, projection_weight: &B::Weight, eps: f32) -> Result<B::Tensor, BackendError> {
    backend.attn_res_mix(current, state.residuals(), norm_weight, projection_weight, eps)
}
