//! AttnRes CPU reference 算子。

use crate::kernel::cpu::CpuTensor;

pub fn mix(current: &CpuTensor, block_residuals: &[CpuTensor], norm_weight: &[f32], projection_weight: &[f32], eps: f32) -> Result<CpuTensor, String> {
    if current.rows == 0 || current.cols == 0 || norm_weight.len() != current.cols || projection_weight.len() != current.cols {
        return Err("CPU AttnRes current/weight shape 非法".to_owned());
    }
    if block_residuals.iter().any(|tensor| tensor.rows != current.rows || tensor.cols != current.cols) {
        return Err("CPU AttnRes block residual shape 不一致".to_owned());
    }
    let candidate_count = block_residuals.len() + 1;
    let mut output = vec![0.0; current.data.len()];
    let mut scores = vec![0.0; candidate_count];
    for row in 0..current.rows {
        let candidates = block_residuals.iter().map(|tensor| tensor.row(row)).chain(std::iter::once(current.row(row)));
        let mut maximum = f32::NEG_INFINITY;
        for (candidate, score) in candidates.zip(&mut scores) {
            let variance = candidate.iter().map(|value| value * value).sum::<f32>() / current.cols as f32;
            let inv_rms = 1.0 / (variance + eps).sqrt();
            *score = candidate.iter().zip(norm_weight).zip(projection_weight).map(|((&value, &norm), &projection)| value * inv_rms * norm * projection).sum();
            maximum = maximum.max(*score);
        }
        let denominator = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - maximum).exp();
                *score
            })
            .sum::<f32>();
        for (candidate, &score) in block_residuals.iter().map(|tensor| tensor.row(row)).chain(std::iter::once(current.row(row))).zip(&scores) {
            let probability = score / denominator;
            for (column, &value) in candidate.iter().enumerate() {
                output[row * current.cols + column] += probability * value;
            }
        }
    }
    Ok(CpuTensor { data: output, rows: current.rows, cols: current.cols })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 零打分权重产生均匀混合() {
        let current = CpuTensor { data: vec![3.0, 5.0], rows: 1, cols: 2 };
        let residual = CpuTensor { data: vec![1.0, 3.0], rows: 1, cols: 2 };
        let output = mix(&current, &[residual], &[1.0, 1.0], &[0.0, 0.0], 1.0e-5).unwrap();
        assert_eq!(output.data, vec![2.0, 4.0]);
    }
}
