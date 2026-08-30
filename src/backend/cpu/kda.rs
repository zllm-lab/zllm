//! KDA CPU fused capability。

use crate::{
    attention::kda::{KdaInputs, KdaKernel, KdaSpec, KdaStorage, KdaWeightsRef},
    backend::{
        BackendError,
        cpu::{CpuContext, CpuWeight},
    },
    kernel::cpu::{CpuTensor, kda},
};

pub struct CpuKdaStorage {
    query_conv: Vec<f32>,
    key_conv: Vec<f32>,
    value_conv: Vec<f32>,
    recurrent: Vec<f32>,
}

impl KdaStorage for CpuKdaStorage {
    fn allocated_bytes(&self) -> usize {
        (self.query_conv.len() + self.key_conv.len() + self.value_conv.len() + self.recurrent.len()) * std::mem::size_of::<f32>()
    }
}

impl KdaKernel for CpuContext {
    type KdaStorage = CpuKdaStorage;

    fn allocate_kda_storage(&self, spec: &KdaSpec) -> Result<Self::KdaStorage, BackendError> {
        let conv_elements = spec.projection_size() * spec.short_conv_kernel_size.saturating_sub(1);
        Ok(CpuKdaStorage { query_conv: vec![0.0; conv_elements], key_conv: vec![0.0; conv_elements], value_conv: vec![0.0; conv_elements], recurrent: vec![0.0; spec.recurrent_state_elements()] })
    }

    fn kda_fused(&self, storage: &mut Self::KdaStorage, inputs: KdaInputs<'_, CpuTensor>, weights: KdaWeightsRef<'_, CpuWeight>, spec: &KdaSpec) -> Result<CpuTensor, BackendError> {
        let projection_size = spec.projection_size();
        let conv_shape = (projection_size, spec.short_conv_kernel_size);
        if (weights.query_conv.rows(), weights.query_conv.cols()) != conv_shape
            || (weights.key_conv.rows(), weights.key_conv.cols()) != conv_shape
            || (weights.value_conv.rows(), weights.value_conv.cols()) != conv_shape
            || weights.a_log.data().len() != spec.num_heads
            || weights.dt_bias.data().len() != projection_size
            || weights.output_norm.data().len() != spec.head_dim
        {
            return Err(BackendError::Compute { msg: "CPU KDA weight shape 与 spec 不一致".to_owned() });
        }
        kda::recurrent(
            inputs.query,
            inputs.key,
            inputs.value,
            inputs.decay,
            inputs.beta,
            inputs.output_gate,
            &mut storage.query_conv,
            &mut storage.key_conv,
            &mut storage.value_conv,
            &mut storage.recurrent,
            weights.query_conv.data(),
            weights.key_conv.data(),
            weights.value_conv.data(),
            weights.a_log.data(),
            weights.dt_bias.data(),
            weights.output_norm.data(),
            spec,
        )
        .map_err(|msg| BackendError::Compute { msg })
    }
}
