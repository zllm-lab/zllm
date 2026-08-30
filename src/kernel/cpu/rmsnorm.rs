//! RMSNorm。

use wide::f32x8;

const SIMD_LANES: usize = 8;

/// `out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]`。
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    rmsnorm_with_weight_offset(x, w, eps, 0.0, out, "RMSNorm");
}

/// `out[i] = x[i] * rsqrt(mean(x^2) + eps) * (1 + w[i])`。MiniMax-M3 用。
pub fn gemma_rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    rmsnorm_with_weight_offset(x, w, eps, 1.0, out, "GemmaRMSNorm");
}

pub(crate) fn rmsnorm_with_weight_offset(x: &[f32], w: &[f32], eps: f32, weight_offset: f32, out: &mut [f32], name: &str) {
    assert_eq!(x.len(), w.len(), "{name} input 与 weight 长度不一致");
    assert_eq!(x.len(), out.len(), "{name} input 与 output 长度不一致");
    if x.is_empty() {
        return;
    }
    let mean_sq = sum_of_squares(x) / x.len() as f32;
    let scalar_inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let inv_rms = f32x8::splat(scalar_inv_rms);
    let offset = f32x8::splat(weight_offset);
    for ((xc, wc), oc) in x.chunks_exact(SIMD_LANES).zip(w.chunks_exact(SIMD_LANES)).zip(out.chunks_exact_mut(SIMD_LANES)) {
        let x = f32x8::from(<[f32; SIMD_LANES]>::try_from(xc).unwrap());
        let weight = f32x8::from(<[f32; SIMD_LANES]>::try_from(wc).unwrap());
        let weight = if weight_offset == 0.0 { weight } else { offset + weight };
        let values: [f32; SIMD_LANES] = (x * inv_rms * weight).into();
        oc.copy_from_slice(&values);
    }
    let full = x.len() / SIMD_LANES * SIMD_LANES;
    for ((&x, &weight), output) in x[full..].iter().zip(&w[full..]).zip(&mut out[full..]) {
        let weight = if weight_offset == 0.0 { weight } else { weight_offset + weight };
        *output = x * scalar_inv_rms * weight;
    }
}

fn sum_of_squares(x: &[f32]) -> f32 {
    let mut acc = f32x8::splat(0.0);
    for c in x.chunks_exact(SIMD_LANES) {
        let v = f32x8::from(<[f32; SIMD_LANES]>::try_from(c).unwrap());
        acc += v * v;
    }
    let full = x.len() / SIMD_LANES * SIMD_LANES;
    acc.reduce_add() + x[full..].iter().map(|value| value * value).sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::{SIMD_LANES, gemma_rmsnorm, rmsnorm};

    #[test]
    fn gemma权重按一加gamma解释() {
        let input: Vec<f32> = (1..=SIMD_LANES).map(|value| value as f32).collect();
        let weight: Vec<f32> = (0..SIMD_LANES).map(|index| [0.0, 0.5, -0.5, 1.0][index % 4]).collect();
        let mut standard = vec![0.0; SIMD_LANES];
        let mut gemma = vec![0.0; SIMD_LANES];

        rmsnorm(&input, &weight, 1.0e-6, &mut standard);
        gemma_rmsnorm(&input, &weight, 1.0e-6, &mut gemma);

        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / SIMD_LANES as f32;
        let inv_rms = 1.0 / (mean_square + 1.0e-6).sqrt();
        for index in 0..SIMD_LANES {
            assert!((standard[index] - input[index] * inv_rms * weight[index]).abs() < 1.0e-6);
            assert!((gemma[index] - input[index] * inv_rms * (1.0 + weight[index])).abs() < 1.0e-6);
        }
    }

    #[test]
    fn 非八倍数长度覆盖标量尾部() {
        let input: Vec<f32> = (1..=11).map(|value| value as f32).collect();
        let weight: Vec<f32> = (0..11).map(|index| 0.5 + index as f32 * 0.1).collect();
        let mut standard = vec![f32::NAN; input.len()];
        let mut gemma = vec![f32::NAN; input.len()];
        rmsnorm(&input, &weight, 1.0e-6, &mut standard);
        gemma_rmsnorm(&input, &weight, 1.0e-6, &mut gemma);

        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
        let inv_rms = 1.0 / (mean_square + 1.0e-6).sqrt();
        for index in 0..input.len() {
            assert!((standard[index] - input[index] * inv_rms * weight[index]).abs() < 1.0e-5);
            assert!((gemma[index] - input[index] * inv_rms * (1.0 + weight[index])).abs() < 1.0e-5);
        }
    }
}
