//! AttnRes Metal capability。

use crate::{
    attention::attn_res::AttnResBackend,
    backend::{
        BackendError,
        metal::{MetalContext, MetalTensor, MetalWeight},
    },
    kernel::metal as ops,
};

use super::expect_resident_f16;

impl AttnResBackend for MetalContext {
    fn attn_res_mix(&self, current: &MetalTensor, block_residuals: &[MetalTensor], norm_weight: &MetalWeight, projection_weight: &MetalWeight, eps: f32) -> Result<MetalTensor, BackendError> {
        let norm_weight = expect_resident_f16(norm_weight, "Metal AttnRes norm weight 需要 resident F16")?;
        let projection_weight = expect_resident_f16(projection_weight, "Metal AttnRes projection weight 需要 resident F16")?;
        if (norm_weight.rows, norm_weight.cols) != (1, current.cols) || (projection_weight.rows, projection_weight.cols) != (1, current.cols) {
            return Err(BackendError::Compute { msg: "Metal AttnRes weight shape 非法".to_owned() });
        }
        ops::attn_res::mix_tensor(self, current, block_residuals, norm_weight, projection_weight, eps).map_err(|msg| BackendError::Compute { msg })
    }
}
