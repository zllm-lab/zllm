//! Gated DeltaNet CUDA fused kernel capability。

use cudarc::driver::safe::CudaSlice;

use crate::{
    attention::gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetInputs, GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetStorage, GatedDeltaNetWeightsRef},
    backend::BackendError,
    backend::cuda::{CudaContext, CudaTensor, CudaWeight},
    kernel::cuda as ops,
};

pub struct CudaGatedDeltaNetStorage {
    conv: CudaSlice<f32>,
    recurrent: CudaSlice<f32>,
}

impl GatedDeltaNetStorage for CudaGatedDeltaNetStorage {
    fn allocated_bytes(&self) -> usize {
        (self.conv.len() + self.recurrent.len()) * std::mem::size_of::<f32>()
    }
}

impl GatedDeltaNetKernel for CudaContext {
    type GatedDeltaNetStorage = CudaGatedDeltaNetStorage;

    fn allocate_gated_delta_net_storage(&self, spec: &GatedDeltaNetSpec) -> Result<Self::GatedDeltaNetStorage, BackendError> {
        let conv = self.stream().alloc_zeros::<f32>(spec.conv_state_elements()).map_err(|error| BackendError::Compute { msg: format!("CUDA DeltaNet conv state 分配失败: {error:?}") })?;
        let recurrent = self.stream().alloc_zeros::<f32>(spec.recurrent_elements()).map_err(|error| BackendError::Compute { msg: format!("CUDA DeltaNet recurrent state 分配失败: {error:?}") })?;
        Ok(CudaGatedDeltaNetStorage { conv, recurrent })
    }

    fn gated_delta_net_fused(&self, storage: &mut Self::GatedDeltaNetStorage, inputs: GatedDeltaNetInputs<'_, CudaTensor>, weights: GatedDeltaNetWeightsRef<'_, CudaWeight>, spec: &GatedDeltaNetSpec) -> Result<CudaTensor, BackendError> {
        self.gated_delta_net_fused_layout(storage, inputs, weights, GatedDeltaNetHeadLayout::Tiled, spec)
    }

    fn gated_delta_net_fused_layout(
        &self,
        storage: &mut Self::GatedDeltaNetStorage,
        inputs: GatedDeltaNetInputs<'_, CudaTensor>,
        weights: GatedDeltaNetWeightsRef<'_, CudaWeight>,
        head_layout: GatedDeltaNetHeadLayout,
        spec: &GatedDeltaNetSpec,
    ) -> Result<CudaTensor, BackendError> {
        ops::attention::gated_delta_net_f16(self, inputs, weights, &storage.conv, &storage.recurrent, head_layout, spec).map_err(|msg| BackendError::Compute { msg })
    }
}
