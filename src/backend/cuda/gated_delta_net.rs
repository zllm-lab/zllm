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
    checkpoints: Option<(CudaSlice<f32>, CudaSlice<f32>, usize)>,
    checkpoint_rows: usize,
}

impl CudaGatedDeltaNetStorage {
    /// 验证批次按行保留设备状态,容量由 session 的草稿预算决定。
    pub fn enable_checkpoints(&mut self, ctx: &CudaContext, rows: usize) -> Result<(), String> {
        self.checkpoint_rows = 0;
        if rows == 0 {
            self.checkpoints = None;
            return Ok(());
        }
        let conv = rows.checked_mul(self.conv.len()).ok_or("GDN conv checkpoints 大小溢出")?;
        let recurrent = rows.checked_mul(self.recurrent.len()).ok_or("GDN recurrent checkpoints 大小溢出")?;
        self.checkpoints = Some((ctx.buffer_uninit(conv)?, ctx.buffer_uninit(recurrent)?, rows));
        Ok(())
    }

    /// row 是最近一次验证调用中最后一个接受输入的零基行号。
    pub fn restore_checkpoint(&mut self, ctx: &CudaContext, row: usize) -> Result<(), String> {
        let (conv, recurrent, rows) = self.checkpoints.as_ref().ok_or("GDN checkpoints 尚未启用")?;
        if row >= self.checkpoint_rows {
            return Err(format!("GDN checkpoint row={row} 超过已记录 {} 行(capacity={rows})", self.checkpoint_rows));
        }
        let conv_len = self.conv.len();
        let recurrent_len = self.recurrent.len();
        ctx.stream().memcpy_dtod(&conv.slice(row * conv_len..(row + 1) * conv_len), &mut self.conv).map_err(|e| format!("恢复 GDN conv: {e:?}"))?;
        ctx.stream().memcpy_dtod(&recurrent.slice(row * recurrent_len..(row + 1) * recurrent_len), &mut self.recurrent).map_err(|e| format!("恢复 GDN recurrent: {e:?}"))?;
        Ok(())
    }
}

impl GatedDeltaNetStorage for CudaGatedDeltaNetStorage {
    fn allocated_bytes(&self) -> usize {
        (self.conv.len() + self.recurrent.len() + self.checkpoints.as_ref().map_or(0, |(conv, recurrent, _)| conv.len() + recurrent.len())) * std::mem::size_of::<f32>()
    }
}

impl GatedDeltaNetKernel for CudaContext {
    type GatedDeltaNetStorage = CudaGatedDeltaNetStorage;

    fn allocate_gated_delta_net_storage(&self, spec: &GatedDeltaNetSpec) -> Result<Self::GatedDeltaNetStorage, BackendError> {
        let conv = self.stream().alloc_zeros::<f32>(spec.conv_state_elements()).map_err(|error| BackendError::Compute { msg: format!("CUDA DeltaNet conv state 分配失败: {error:?}") })?;
        let recurrent = self.stream().alloc_zeros::<f32>(spec.recurrent_elements()).map_err(|error| BackendError::Compute { msg: format!("CUDA DeltaNet recurrent state 分配失败: {error:?}") })?;
        Ok(CudaGatedDeltaNetStorage { conv, recurrent, checkpoints: None, checkpoint_rows: 0 })
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
        if storage.checkpoints.as_ref().is_some_and(|(_, _, rows)| inputs.qkv.rows > *rows) {
            return Err(crate::backend::compute_error("GDN 输入超过 checkpoint 容量"));
        }
        let checkpoints = storage.checkpoints.as_ref().map(|(conv, recurrent, _)| (conv, recurrent));
        let rows = inputs.qkv.rows;
        let output = ops::attention::gated_delta_net_f16(self, inputs, weights, &storage.conv, &storage.recurrent, head_layout, spec, checkpoints).map_err(|msg| BackendError::Compute { msg })?;
        storage.checkpoint_rows = if storage.checkpoints.is_some() { rows } else { 0 };
        Ok(output)
    }
}
