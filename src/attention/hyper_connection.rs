//! 多路残差连接的设备无关数学语义。
//!
//! mHC 的投影由 runtime 按模型权重编排；这里仅保存副本展开、Sinkhorn 约束和
//! 副本混合，CPU/Metal 可用各自 kernel 替换这些 reference 实现。

use crate::backend::{Backend, BackendError};

#[derive(Debug, Clone, Copy)]
pub struct HyperConnectionSpec {
    pub copies: usize,
    pub sinkhorn_iterations: usize,
    pub eps: f32,
}

pub struct HyperConnectionSplit<T> {
    pub pre: T,
    pub post: T,
    pub combination: T,
}

pub struct HyperConnectionPrepared<T> {
    pub residual: T,
    pub reduced: T,
    pub post: T,
}

impl HyperConnectionSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.copies == 0 || self.sinkhorn_iterations == 0 || !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(format!("mHC 配置非法: copies={} sinkhorn_iterations={} eps={}", self.copies, self.sinkhorn_iterations, self.eps,));
        }
        Ok(())
    }
}

pub fn expand_f32(hidden: &[f32], copies: usize) -> Result<Vec<f32>, String> {
    if copies == 0 || hidden.is_empty() {
        return Err(format!("mHC 展开维度非法: hidden={} copies={copies}", hidden.len()));
    }
    let mut expanded = Vec::with_capacity(hidden.len() * copies);
    for _ in 0..copies {
        expanded.extend_from_slice(hidden);
    }
    Ok(expanded)
}

pub fn expand_scaled_f32(hidden: &[f32], coefficients: &[f32], copies: usize) -> Result<Vec<f32>, String> {
    if copies == 0 || hidden.is_empty() || coefficients.len() != copies {
        return Err(format!("mHC 缩放展开维度非法: hidden={} coefficients={} copies={copies}", hidden.len(), coefficients.len(),));
    }
    let mut expanded = Vec::with_capacity(hidden.len() * copies);
    for coefficient in coefficients {
        expanded.extend(hidden.iter().map(|value| coefficient * value));
    }
    Ok(expanded)
}

pub fn reduce_f32(hidden: &[f32], coefficients: &[f32], copies: usize) -> Result<Vec<f32>, String> {
    if copies == 0 || hidden.is_empty() || !hidden.len().is_multiple_of(copies) || coefficients.len() != copies {
        return Err(format!("mHC 归并维度非法: hidden={} coefficients={} copies={copies}", hidden.len(), coefficients.len(),));
    }
    let width = hidden.len() / copies;
    let mut output = vec![0.0; width];
    for (copy, coefficient) in hidden.chunks_exact(width).zip(coefficients) {
        for (out, value) in output.iter_mut().zip(copy) {
            *out += coefficient * value;
        }
    }
    Ok(output)
}

/// DeepSeek-V4 output mHC：每个 copy 使用 sigmoid 系数归并，不做 softmax。
/// `rows` 从 `mixes.len() / copies` 推出，`hidden` 形状必须与之兼容。
pub fn head_reduce_f32(hidden: &[f32], mixes: &[f32], base: &[f32], scale: f32, copies: usize, eps: f32) -> Result<Vec<f32>, String> {
    if copies == 0 || hidden.is_empty() || mixes.len() == 0 || !mixes.len().is_multiple_of(copies) || base.len() != copies || !scale.is_finite() || !eps.is_finite() || eps <= 0.0 {
        return Err(format!("output mHC 维度非法: hidden={} mixes={} base={} scale={scale} copies={copies} eps={eps}", hidden.len(), mixes.len(), base.len()));
    }
    let rows = mixes.len() / copies;
    if !hidden.len().is_multiple_of(rows) || !(hidden.len() / rows).is_multiple_of(copies) {
        return Err(format!("output mHC hidden={} 不能按 rows={rows} copies={copies} 拆分", hidden.len()));
    }
    let expanded_width = hidden.len() / rows;
    let width = expanded_width / copies;
    let mut output = vec![0.0; rows * width];
    for row in 0..rows {
        for copy in 0..copies {
            let logit = mixes[row * copies + copy] * scale + base[copy];
            let coefficient = 1.0 / (1.0 + (-logit).exp()) + eps;
            let input = &hidden[row * expanded_width + copy * width..row * expanded_width + (copy + 1) * width];
            let target = &mut output[row * width..(row + 1) * width];
            for (target, value) in target.iter_mut().zip(input) {
                *target += coefficient * value;
            }
        }
    }
    Ok(output)
}

/// 把方阵 logits 约束成双随机混合矩阵。
fn sinkhorn_f32(logits: &[f32], size: usize, iterations: usize, eps: f32) -> Result<Vec<f32>, String> {
    if size == 0 || logits.len() != size * size || iterations == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(format!("mHC Sinkhorn 维度非法: logits={} size={size} iterations={iterations} eps={eps}", logits.len(),));
    }
    let mut matrix = Vec::with_capacity(logits.len());
    for row in logits.chunks_exact(size) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = row.iter().map(|value| (*value - maximum).exp()).sum::<f32>();
        matrix.extend(row.iter().map(|value| (*value - maximum).exp() / sum + eps));
    }
    for col in 0..size {
        let sum = (0..size).map(|row| matrix[row * size + col]).sum::<f32>();
        for row in 0..size {
            matrix[row * size + col] /= sum + eps;
        }
    }
    for _ in 1..iterations {
        for row in matrix.chunks_exact_mut(size) {
            let sum = row.iter().sum::<f32>();
            row.iter_mut().for_each(|value| *value /= sum + eps);
        }
        for col in 0..size {
            let sum = (0..size).map(|row| matrix[row * size + col]).sum::<f32>();
            for row in 0..size {
                matrix[row * size + col] /= sum + eps;
            }
        }
    }
    Ok(matrix)
}

pub fn mix_f32(hidden: &[f32], matrix: &[f32], copies: usize) -> Result<Vec<f32>, String> {
    if copies == 0 || hidden.is_empty() || !hidden.len().is_multiple_of(copies) || matrix.len() != copies * copies {
        return Err(format!("mHC 混合维度非法: hidden={} matrix={} copies={copies}", hidden.len(), matrix.len(),));
    }
    let width = hidden.len() / copies;
    let mut output = vec![0.0; hidden.len()];
    for out_copy in 0..copies {
        for in_copy in 0..copies {
            let coefficient = matrix[out_copy * copies + in_copy];
            for column in 0..width {
                output[out_copy * width + column] += coefficient * hidden[in_copy * width + column];
            }
        }
    }
    Ok(output)
}

/// 按官方 mHC function 投影布局拆出 pre、post 与双随机混合矩阵。
/// `rows` 从 `mixes.len() / mix_width` 推出。
pub fn split_f32(mixes: &[f32], base: &[f32], scale: &[f32], spec: &HyperConnectionSpec) -> Result<HyperConnectionSplit<Vec<f32>>, String> {
    spec.validate()?;
    let copies = spec.copies;
    let mix_width = (2 + copies) * copies;
    if base.len() != mix_width || scale.len() != 3 || !mixes.len().is_multiple_of(mix_width) {
        return Err(format!("mHC split 维度非法: mixes={} base={} scale={} copies={copies}", mixes.len(), base.len(), scale.len(),));
    }
    let rows = mixes.len() / mix_width;
    let mut pre = vec![0.0; rows * copies];
    let mut post = vec![0.0; rows * copies];
    let mut combination = vec![0.0; rows * copies * copies];
    for row in 0..rows {
        let mix = &mixes[row * mix_width..(row + 1) * mix_width];
        for index in 0..copies {
            let pre_logit = mix[index] * scale[0] + base[index];
            pre[row * copies + index] = 1.0 / (1.0 + (-pre_logit).exp()) + spec.eps;
            let post_offset = copies + index;
            let post_logit = mix[post_offset] * scale[1] + base[post_offset];
            post[row * copies + index] = 2.0 / (1.0 + (-post_logit).exp());
        }
        let combination_logits = (0..copies * copies)
            .map(|index| {
                let offset = 2 * copies + index;
                mix[offset] * scale[2] + base[offset]
            })
            .collect::<Vec<_>>();
        combination[row * copies * copies..(row + 1) * copies * copies].copy_from_slice(&sinkhorn_f32(&combination_logits, copies, spec.sinkhorn_iterations, spec.eps)?);
    }
    Ok(HyperConnectionSplit { pre, post, combination })
}

/// mHC 的设备能力只表达副本张量变换；模型投影和 residual 编排留在 runtime。
pub trait HyperConnectionKernel: Backend {
    fn hyper_connection_expand(&self, hidden: &Self::Tensor, copies: usize) -> Result<Self::Tensor, BackendError>;

    /// 把 copies 份残差流折叠回单份(无权重平均)。glm5_next 的输出收尾;
    /// backend 未实现时模型退路是构造全 1 系数走 reduce。
    fn hyper_connection_collapse(&self, hidden: &Self::Tensor, copies: usize) -> Result<Self::Tensor, BackendError> {
        let _ = (hidden, copies);
        Err(BackendError::Compute { msg: "backend 未实现 mHC collapse".to_owned() })
    }

    fn hyper_connection_reduce(&self, hidden: &Self::Tensor, coefficients: &Self::Tensor, copies: usize) -> Result<Self::Tensor, BackendError>;

    /// 使用 split 已约束好的双随机矩阵混合 copies，不重复执行 Sinkhorn。
    fn hyper_connection_mix(&self, hidden: &Self::Tensor, matrix: &Self::Tensor, spec: &HyperConnectionSpec) -> Result<Self::Tensor, BackendError>;

    /// 把子层输出按每个 mHC copy 的系数展开。
    fn hyper_connection_expand_scaled(&self, hidden: &Self::Tensor, coefficients: &Self::Tensor, copies: usize) -> Result<Self::Tensor, BackendError>;

    /// 展开后立即叠加 residual；backend 可融合，默认保持两步语义。
    fn hyper_connection_expand_scaled_add(&self, hidden: &Self::Tensor, coefficients: &Self::Tensor, residual: &Self::Tensor, copies: usize) -> Result<Self::Tensor, BackendError> {
        let expanded = self.hyper_connection_expand_scaled(hidden, coefficients, copies)?;
        self.add(residual, &expanded)
    }

    /// 把 function 投影拆成 pre、post 与 copy 混合矩阵。
    fn hyper_connection_split(&self, mixes: &Self::Tensor, base: &Self::Weight, scale: &Self::Weight, spec: &HyperConnectionSpec) -> Result<HyperConnectionSplit<Self::Tensor>, BackendError>;

    /// split 后的 mix/reduce 共用 hidden；backend 可融合并省略 pre/combination 中间张量。
    fn hyper_connection_prepare_sublayer(&self, hidden: &Self::Tensor, mixes: &Self::Tensor, base: &Self::Weight, scale: &Self::Weight, spec: &HyperConnectionSpec) -> Result<HyperConnectionPrepared<Self::Tensor>, BackendError> {
        let split = self.hyper_connection_split(mixes, base, scale, spec)?;
        let residual = self.hyper_connection_mix(hidden, &split.combination, spec)?;
        let reduced = self.hyper_connection_reduce(hidden, &split.pre, spec.copies)?;
        Ok(HyperConnectionPrepared { residual, reduced, post: split.post })
    }

    /// 输出层使用 sigmoid gate 直接归并 copies，与层内 Sinkhorn mHC 分开。
    fn hyper_connection_head_reduce(&self, hidden: &Self::Tensor, mixes: &Self::Tensor, base: &Self::Weight, scale: &Self::Weight, copies: usize, eps: f32) -> Result<Self::Tensor, BackendError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_produces_balanced_copy_mixing() {
        let matrix = sinkhorn_f32(&[2.0, -1.0, 0.5, 3.0], 2, 64, 1.0e-6).unwrap();
        for row in matrix.chunks_exact(2) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1.0e-4);
        }
        for column in 0..2 {
            assert!((matrix[column] + matrix[2 + column] - 1.0).abs() < 1.0e-4);
        }
    }

    #[test]
    fn identity_mix_keeps_each_copy() {
        let hidden = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mix_f32(&hidden, &[1.0, 0.0, 0.0, 1.0], 2).unwrap(), hidden);
    }

    #[test]
    fn reduce_uses_one_coefficient_per_copy() {
        assert_eq!(reduce_f32(&[1.0, 2.0, 5.0, 8.0], &[0.25, 0.75], 2).unwrap(), [4.0, 6.5]);
    }

    #[test]
    fn split_keeps_projection_layout_in_one_place() {
        let spec = HyperConnectionSpec { copies: 2, sinkhorn_iterations: 64, eps: 1.0e-6 };
        let split = split_f32(&[0.0; 8], &[0.0; 8], &[1.0; 3], &spec).unwrap();
        assert_eq!(split.pre, [0.500001, 0.500001]);
        assert_eq!(split.post, [1.0, 1.0]);
        assert!(split.combination.iter().all(|value| (*value - 0.5).abs() < 1.0e-6));
    }

    #[test]
    fn output_head_uses_independent_sigmoid_gates() {
        let output = head_reduce_f32(&[1.0, 2.0, 5.0, 8.0], &[0.0, 0.0], &[0.0, 0.0], 1.0, 2, 1.0e-6).unwrap();
        assert!((output[0] - 3.000_006).abs() < 1.0e-5);
        assert!((output[1] - 5.000_01).abs() < 1.0e-5);
    }
}
