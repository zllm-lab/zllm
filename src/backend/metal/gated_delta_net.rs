//! Gated DeltaNet Metal fused kernel capability。

use crate::backend::metal::api::Buffer;

use crate::{
    attention::gated_delta_net::{GatedDeltaNetInputs, GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetStorage, GatedDeltaNetWeightsRef},
    backend::BackendError,
    backend::metal::{MetalContext, MetalTensor, MetalWeight},
    kernel::metal as ops,
};

use super::{expect_resident_f16, expect_resident_f32};

pub struct MetalGatedDeltaNetStorage {
    conv: Buffer,
    recurrent: Buffer,
}

impl MetalGatedDeltaNetStorage {
    pub fn conv_buffer(&self) -> &Buffer {
        &self.conv
    }

    pub fn recurrent_buffer(&self) -> &Buffer {
        &self.recurrent
    }

    /// 快照恢复：从已拷贝好字节的 buffer 重建 storage。
    pub fn from_buffers(conv: Buffer, recurrent: Buffer) -> Self {
        Self { conv, recurrent }
    }
}

impl GatedDeltaNetStorage for MetalGatedDeltaNetStorage {
    fn allocated_bytes(&self) -> usize {
        self.conv.length() as usize + self.recurrent.length() as usize
    }
}

impl GatedDeltaNetKernel for MetalContext {
    type GatedDeltaNetStorage = MetalGatedDeltaNetStorage;

    fn allocate_gated_delta_net_storage(&self, spec: &GatedDeltaNetSpec) -> Result<Self::GatedDeltaNetStorage, BackendError> {
        Ok(MetalGatedDeltaNetStorage { conv: self.shared_buffer_zeros(spec.conv_state_elements() * std::mem::size_of::<f32>()), recurrent: self.shared_buffer_zeros(spec.recurrent_elements() * std::mem::size_of::<f32>()) })
    }

    fn gated_delta_net_fused(&self, storage: &mut Self::GatedDeltaNetStorage, inputs: GatedDeltaNetInputs<'_, MetalTensor>, weights: GatedDeltaNetWeightsRef<'_, MetalWeight>, spec: &GatedDeltaNetSpec) -> Result<MetalTensor, BackendError> {
        let GatedDeltaNetInputs { qkv, z, alpha, beta } = inputs;
        let GatedDeltaNetWeightsRef { conv: conv_weight, a_log, dt_bias, norm: norm_weight } = weights;
        let conv_weight = expect_resident_f16(conv_weight, "Metal Gated DeltaNet conv weight 需要 resident F16")?;
        let (a_log, a_log_len) = expect_resident_f32(a_log, "Metal Gated DeltaNet A_log 需要 resident F32")?;
        let (dt_bias, dt_bias_len) = expect_resident_f32(dt_bias, "Metal Gated DeltaNet dt_bias 需要 resident F32")?;
        let (norm_weight, norm_weight_len) = expect_resident_f32(norm_weight, "Metal Gated DeltaNet norm weight 需要 resident F32")?;
        ops::tensor::gated_delta_net_tensor(self, qkv, z, alpha, beta, conv_weight, a_log, a_log_len, dt_bias, dt_bias_len, norm_weight, norm_weight_len, &storage.conv, &storage.recurrent, spec).map_err(|msg| BackendError::Compute { msg })
    }
}
