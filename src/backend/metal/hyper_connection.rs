//! mHC Metal capability；张量变换和 Sinkhorn 全程留在设备端。

use crate::{
    attention::hyper_connection::{HyperConnectionKernel, HyperConnectionSpec, HyperConnectionSplit},
    backend::{
        BackendError, compute_error as compute,
        metal::{MetalContext, MetalTensor, MetalWeight, api::Buffer},
    },
    kernel::metal::hyper_connection as ops,
};

impl HyperConnectionKernel for MetalContext {
    fn hyper_connection_expand(&self, hidden: &MetalTensor, copies: usize) -> Result<MetalTensor, BackendError> {
        ops::expand_tensor(self, hidden, copies).map_err(compute)
    }

    fn hyper_connection_reduce(&self, hidden: &MetalTensor, coefficients: &MetalTensor, copies: usize) -> Result<MetalTensor, BackendError> {
        ops::reduce_tensor(self, hidden, coefficients, copies).map_err(compute)
    }

    fn hyper_connection_expand_scaled(&self, hidden: &MetalTensor, coefficients: &MetalTensor, copies: usize) -> Result<MetalTensor, BackendError> {
        ops::expand_scaled_tensor(self, hidden, coefficients, copies).map_err(compute)
    }

    fn hyper_connection_mix(&self, hidden: &MetalTensor, matrix: &MetalTensor, spec: &HyperConnectionSpec) -> Result<MetalTensor, BackendError> {
        spec.validate().map_err(compute)?;
        ops::mix_tensor(self, hidden, matrix, spec.copies).map_err(compute)
    }

    fn hyper_connection_split(&self, mixes: &MetalTensor, base: &MetalWeight, scale: &MetalWeight, spec: &HyperConnectionSpec) -> Result<HyperConnectionSplit<MetalTensor>, BackendError> {
        spec.validate().map_err(compute)?;
        let copies = spec.copies;
        let rows = mixes.rows;
        let mix_columns = copies.checked_mul(copies.checked_add(2).ok_or_else(|| compute("Metal mHC copies 溢出"))?).ok_or_else(|| compute("Metal mHC mix columns 溢出"))?;
        let base = f32_weight_buffer(base, mix_columns, "mHC base")?;
        let scale = f32_weight_buffer(scale, 3, "mHC scale")?;
        let (pre, post, combination) = ops::split_tensor(self, mixes, base, scale, copies, spec.sinkhorn_iterations, spec.eps).map_err(compute)?;
        debug_assert_eq!((pre.rows, pre.cols), (rows, copies));
        Ok(HyperConnectionSplit { pre, post, combination })
    }

    fn hyper_connection_head_reduce(&self, hidden: &MetalTensor, mixes: &MetalTensor, base: &MetalWeight, scale: &MetalWeight, copies: usize, eps: f32) -> Result<MetalTensor, BackendError> {
        let base = f32_weight_buffer(base, copies, "output mHC base")?;
        let scale = f32_weight_buffer(scale, 1, "output mHC scale")?;
        ops::head_reduce_tensor(self, hidden, mixes, base, scale, copies, eps).map_err(compute)
    }
}

fn f32_weight_buffer<'a>(weight: &'a MetalWeight, expected: usize, name: &str) -> Result<&'a Buffer, BackendError> {
    match weight {
        MetalWeight::F32 { buffer, len } if *len == expected => Ok(buffer),
        MetalWeight::F32 { len, .. } => Err(compute(format!("Metal {name} 长度={len}，期望 {expected}"))),
        _ => Err(compute(format!("Metal {name} 必须是设备端 F32 常量"))),
    }
}
