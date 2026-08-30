//! AttnRes CPU capability。

use crate::{
    attention::attn_res::AttnResBackend,
    backend::{
        BackendError,
        cpu::{CpuContext, CpuWeight},
    },
    kernel::cpu::{CpuTensor, attn_res},
};

impl AttnResBackend for CpuContext {
    fn attn_res_mix(&self, current: &CpuTensor, block_residuals: &[CpuTensor], norm_weight: &CpuWeight, projection_weight: &CpuWeight, eps: f32) -> Result<CpuTensor, BackendError> {
        if (norm_weight.rows(), norm_weight.cols()) != (1, current.cols) || (projection_weight.rows(), projection_weight.cols()) != (1, current.cols) {
            return Err(BackendError::Compute { msg: "CPU AttnRes weight shape 非法".to_owned() });
        }
        attn_res::mix(current, block_residuals, norm_weight.data(), projection_weight.data(), eps).map_err(|msg| BackendError::Compute { msg })
    }
}
