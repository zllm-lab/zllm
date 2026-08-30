//! KDA Metal fused capability。

use crate::backend::metal::api::Buffer;

use crate::{
    attention::kda::{KdaInputs, KdaKernel, KdaSpec, KdaStorage, KdaWeightsRef},
    backend::{
        BackendError,
        metal::{MetalContext, MetalTensor, MetalWeight},
    },
    kernel::metal as ops,
};

use super::{expect_resident_f16, expect_resident_f32};

pub struct MetalKdaStorage {
    conv: Buffer,
    recurrent: Buffer,
}

impl KdaStorage for MetalKdaStorage {
    fn allocated_bytes(&self) -> usize {
        self.conv.length() as usize + self.recurrent.length() as usize
    }
}

impl KdaKernel for MetalContext {
    type KdaStorage = MetalKdaStorage;

    fn allocate_kda_storage(&self, spec: &KdaSpec) -> Result<Self::KdaStorage, BackendError> {
        Ok(MetalKdaStorage { conv: self.shared_buffer_zeros(spec.conv_state_elements() * std::mem::size_of::<f32>()), recurrent: self.shared_buffer_zeros(spec.recurrent_state_elements() * std::mem::size_of::<f32>()) })
    }

    fn kda_fused(&self, storage: &mut Self::KdaStorage, inputs: KdaInputs<'_, MetalTensor>, weights: KdaWeightsRef<'_, MetalWeight>, spec: &KdaSpec) -> Result<MetalTensor, BackendError> {
        let query_conv_weight = expect_resident_f16(weights.query_conv, "Metal KDA query conv weight 需要 resident F16")?;
        let key_conv_weight = expect_resident_f16(weights.key_conv, "Metal KDA key conv weight 需要 resident F16")?;
        let value_conv_weight = expect_resident_f16(weights.value_conv, "Metal KDA value conv weight 需要 resident F16")?;
        let (a_log, a_log_len) = expect_resident_f32(weights.a_log, "Metal KDA A_log 需要 resident F32")?;
        let (dt_bias, dt_bias_len) = expect_resident_f32(weights.dt_bias, "Metal KDA dt_bias 需要 resident F32")?;
        let output_norm_weight = expect_resident_f16(weights.output_norm, "Metal KDA output norm weight 需要 resident F16")?;
        ops::kda::recurrent_tensor(
            self,
            inputs.query,
            inputs.key,
            inputs.value,
            inputs.decay,
            inputs.beta,
            inputs.output_gate,
            &query_conv_weight.buffer,
            &key_conv_weight.buffer,
            &value_conv_weight.buffer,
            a_log,
            a_log_len,
            dt_bias,
            dt_bias_len,
            &output_norm_weight.buffer,
            &storage.conv,
            &storage.recurrent,
            spec,
        )
        .map_err(|msg| BackendError::Compute { msg })
    }
}
