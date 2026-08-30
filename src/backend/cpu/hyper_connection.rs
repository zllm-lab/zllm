//! mHC CPU reference capability。

use crate::{
    attention::hyper_connection::{HyperConnectionKernel, HyperConnectionSpec, HyperConnectionSplit, expand_f32, expand_scaled_f32, head_reduce_f32, mix_f32, reduce_f32, split_f32},
    backend::{
        BackendError, compute_error as compute,
        cpu::{CpuContext, CpuWeight},
    },
    kernel::cpu::CpuTensor,
};

impl HyperConnectionKernel for CpuContext {
    fn hyper_connection_collapse(&self, hidden: &CpuTensor, copies: usize) -> Result<CpuTensor, BackendError> {
        if copies == 0 || hidden.data.len() % copies != 0 {
            return Err(BackendError::Compute { msg: format!("mHC collapse 维度非法: len={} copies={copies}", hidden.data.len()) });
        }
        let width = hidden.data.len() / copies;
        let scale = 1.0 / copies as f32;
        let mut data = vec![0.0_f32; width];
        for chunk in hidden.data.chunks_exact(width) {
            for (out, value) in data.iter_mut().zip(chunk) {
                *out += value * scale;
            }
        }
        Ok(CpuTensor { data, rows: hidden.rows, cols: width })
    }

    fn hyper_connection_expand(&self, hidden: &CpuTensor, copies: usize) -> Result<CpuTensor, BackendError> {
        let mut data = Vec::with_capacity(hidden.data.len() * copies);
        for row in hidden.data.chunks_exact(hidden.cols) {
            data.extend(expand_f32(row, copies).map_err(compute)?);
        }
        Ok(CpuTensor { data, rows: hidden.rows, cols: hidden.cols * copies })
    }

    fn hyper_connection_reduce(&self, hidden: &CpuTensor, coefficients: &CpuTensor, copies: usize) -> Result<CpuTensor, BackendError> {
        if hidden.rows != coefficients.rows || !hidden.cols.is_multiple_of(copies) || coefficients.cols != copies {
            return Err(compute(format!("CPU mHC reduce shape 不一致: hidden={}x{} coefficients={}x{} copies={copies}", hidden.rows, hidden.cols, coefficients.rows, coefficients.cols,)));
        }
        let width = hidden.cols / copies;
        let mut data = Vec::with_capacity(hidden.rows * width);
        for row in 0..hidden.rows {
            data.extend(reduce_f32(&hidden.data[row * hidden.cols..(row + 1) * hidden.cols], &coefficients.data[row * copies..(row + 1) * copies], copies).map_err(compute)?);
        }
        Ok(CpuTensor { data, rows: hidden.rows, cols: width })
    }

    fn hyper_connection_expand_scaled(&self, hidden: &CpuTensor, coefficients: &CpuTensor, copies: usize) -> Result<CpuTensor, BackendError> {
        if hidden.rows != coefficients.rows || coefficients.cols != copies || hidden.cols == 0 {
            return Err(compute(format!("CPU mHC expand_scaled shape 非法: hidden={}x{} coefficients={}x{} copies={copies}", hidden.rows, hidden.cols, coefficients.rows, coefficients.cols,)));
        }
        let mut data = Vec::with_capacity(hidden.rows * hidden.cols * copies);
        for row in 0..hidden.rows {
            let hidden = &hidden.data[row * hidden.cols..(row + 1) * hidden.cols];
            let coefficients = &coefficients.data[row * copies..(row + 1) * copies];
            data.extend(expand_scaled_f32(hidden, coefficients, copies).map_err(compute)?);
        }
        Ok(CpuTensor { data, rows: hidden.rows, cols: hidden.cols * copies })
    }

    fn hyper_connection_mix(&self, hidden: &CpuTensor, matrix: &CpuTensor, spec: &HyperConnectionSpec) -> Result<CpuTensor, BackendError> {
        spec.validate().map_err(compute)?;
        if hidden.rows != matrix.rows || !hidden.cols.is_multiple_of(spec.copies) || matrix.cols != spec.copies * spec.copies {
            return Err(compute(format!("CPU mHC mix shape 不一致: hidden={}x{} matrix={}x{} copies={}", hidden.rows, hidden.cols, matrix.rows, matrix.cols, spec.copies,)));
        }
        let mut data = Vec::with_capacity(hidden.data.len());
        for row in 0..hidden.rows {
            data.extend(mix_f32(&hidden.data[row * hidden.cols..(row + 1) * hidden.cols], &matrix.data[row * matrix.cols..(row + 1) * matrix.cols], spec.copies).map_err(compute)?);
        }
        Ok(CpuTensor { data, rows: hidden.rows, cols: hidden.cols })
    }

    fn hyper_connection_split(&self, mixes: &CpuTensor, base: &CpuWeight, scale: &CpuWeight, spec: &HyperConnectionSpec) -> Result<HyperConnectionSplit<CpuTensor>, BackendError> {
        spec.validate().map_err(compute)?;
        let copies = spec.copies;
        let base = base.data();
        let scale = scale.data();
        let rows = mixes.rows;
        let split = split_f32(&mixes.data, base, scale, spec).map_err(compute)?;
        Ok(HyperConnectionSplit { pre: CpuTensor { data: split.pre, rows, cols: copies }, post: CpuTensor { data: split.post, rows, cols: copies }, combination: CpuTensor { data: split.combination, rows, cols: copies * copies } })
    }

    fn hyper_connection_head_reduce(&self, hidden: &CpuTensor, mixes: &CpuTensor, base: &CpuWeight, scale: &CpuWeight, copies: usize, eps: f32) -> Result<CpuTensor, BackendError> {
        let scale = scale.data();
        if scale.len() != 1 {
            return Err(compute(format!("CPU output mHC scale 长度={}，期望 1", scale.len())));
        }
        let data = head_reduce_f32(&hidden.data, &mixes.data, base.data(), scale[0], copies, eps).map_err(compute)?;
        Ok(CpuTensor { data, rows: hidden.rows, cols: hidden.cols / copies })
    }
}
