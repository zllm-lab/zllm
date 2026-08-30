//! SiLU × up 算子(swiglu FFN 激活)。

use wide::f32x8;

const SIMD_LANES: usize = 8;

/// `out[i] = silu(gate[i]) * up[i]`。silu(x) = x * sigmoid(x)。
pub fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_same_lengths(gate, up, out, "SiLU");
    for ((g, u), o) in gate.chunks_exact(SIMD_LANES).zip(up.chunks_exact(SIMD_LANES)).zip(out.chunks_exact_mut(SIMD_LANES)) {
        let gv = f32x8::from(<[f32; SIMD_LANES]>::try_from(g).unwrap());
        let sv = sigmoid_v(gv);
        let uv = f32x8::from(<[f32; SIMD_LANES]>::try_from(u).unwrap());
        let values: [f32; SIMD_LANES] = (gv * sv * uv).into();
        o.copy_from_slice(&values);
    }
    let full = gate.len() / SIMD_LANES * SIMD_LANES;
    for ((&gate, &up), output) in gate[full..].iter().zip(&up[full..]).zip(&mut out[full..]) {
        *output = gate / (1.0 + (-gate).exp()) * up;
    }
}

/// DeepSeek-V4 SwiGLU：`silu(min(gate, limit)) * clamp(up, -limit, limit)`。
pub fn silu_clamped_mul(gate: &[f32], up: &[f32], limit: f32, out: &mut [f32]) {
    assert_same_lengths(gate, up, out, "限幅 SiLU");
    let lim = f32x8::splat(limit);
    let neg_lim = f32x8::splat(-limit);
    for ((g, u), o) in gate.chunks_exact(SIMD_LANES).zip(up.chunks_exact(SIMD_LANES)).zip(out.chunks_exact_mut(SIMD_LANES)) {
        let gv = f32x8::from(<[f32; SIMD_LANES]>::try_from(g).unwrap()).min(lim);
        let uv = f32x8::from(<[f32; SIMD_LANES]>::try_from(u).unwrap()).max(neg_lim).min(lim);
        let values: [f32; SIMD_LANES] = (gv * sigmoid_v(gv) * uv).into();
        o.copy_from_slice(&values);
    }
    let full = gate.len() / SIMD_LANES * SIMD_LANES;
    for ((&gate, &up), output) in gate[full..].iter().zip(&up[full..]).zip(&mut out[full..]) {
        let gate = gate.min(limit);
        *output = gate / (1.0 + (-gate).exp()) * up.clamp(-limit, limit);
    }
}

/// Kimi SiTU:`beta * tanh(gate / beta) * sigmoid(gate) * up`。
/// `linear_beta` 存在时，先把 up 变换为 `linear_beta * tanh(up / linear_beta)`。
pub fn situ_mul(gate: &[f32], up: &[f32], beta: f32, linear_beta: Option<f32>, out: &mut [f32]) {
    assert_same_lengths(gate, up, out, "SiTU");
    for ((&gate, &up), output) in gate.iter().zip(up).zip(out) {
        let up = linear_beta.map_or(up, |linear_beta| linear_beta * (up / linear_beta).tanh());
        *output = beta * (gate / beta).tanh() / (1.0 + (-gate).exp()) * up;
    }
}

/// MiniMax-M3 SwiGLU-OAI:`gate * sigmoid(alpha * gate) * (up + 1)`。
/// `gate` 只限制上界,`up` 限制到 `[-limit, limit]`。
pub fn swiglu_oai_mul(gate: &[f32], up: &[f32], alpha: f32, limit: f32, out: &mut [f32]) {
    assert_same_lengths(gate, up, out, "SwiGLU-OAI");
    let av = f32x8::splat(alpha);
    let lim = f32x8::splat(limit);
    let neg_lim = f32x8::splat(-limit);
    let one = f32x8::splat(1.0);
    for ((g, u), o) in gate.chunks_exact(SIMD_LANES).zip(up.chunks_exact(SIMD_LANES)).zip(out.chunks_exact_mut(SIMD_LANES)) {
        let gv = f32x8::from(<[f32; SIMD_LANES]>::try_from(g).unwrap()).min(lim);
        let uv = f32x8::from(<[f32; SIMD_LANES]>::try_from(u).unwrap()).max(neg_lim).min(lim);
        let values: [f32; SIMD_LANES] = (gv * sigmoid_v(av * gv) * (uv + one)).into();
        o.copy_from_slice(&values);
    }
    let full = gate.len() / SIMD_LANES * SIMD_LANES;
    for ((&gate, &up), output) in gate[full..].iter().zip(&up[full..]).zip(&mut out[full..]) {
        let gate = gate.min(limit);
        *output = gate / (1.0 + (-alpha * gate).exp()) * (up.clamp(-limit, limit) + 1.0);
    }
}

/// PyTorch tanh 近似 GELU × up，Gemma 4 dense MLP 使用。
pub fn gelu_tanh_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_same_lengths(gate, up, out, "GELU");
    const SCALE: f32 = 0.797_884_6;
    for ((&gate, &up), output) in gate.iter().zip(up).zip(out) {
        let gelu = 0.5 * gate * (1.0 + (SCALE * (gate + 0.044_715 * gate * gate * gate)).tanh());
        *output = gelu * up;
    }
}

fn assert_same_lengths(gate: &[f32], up: &[f32], out: &[f32], operator: &str) {
    assert_eq!(gate.len(), up.len(), "{operator} gate 与 up 长度不一致");
    assert_eq!(gate.len(), out.len(), "{operator} gate 与 output 长度不一致");
}

/// 非门控 tanh 近似 GELU，视觉 ViT MLP 使用。常数与 Metal `gelu_f16` 对齐(CPU 是 oracle)。
pub fn gelu(x: &[f32], out: &mut [f32]) {
    const SCALE: f32 = 0.797_884_6;
    const CUBIC: f32 = 0.044_715;
    let scale = f32x8::splat(SCALE);
    let cubic = f32x8::splat(CUBIC);
    let half = f32x8::splat(0.5);
    let one = f32x8::splat(1.0);
    for (xc, oc) in x.chunks_exact(SIMD_LANES).zip(out.chunks_exact_mut(SIMD_LANES)) {
        let v = f32x8::from(<[f32; SIMD_LANES]>::try_from(xc).unwrap());
        let inner = scale * (v + cubic * (v * v * v));
        let g: [f32; SIMD_LANES] = (half * v * (one + tanh_v(inner))).into();
        oc.copy_from_slice(&g);
    }
    for (xc, oc) in x[SIMD_LANES * (x.len() / SIMD_LANES)..].iter().copied().zip(&mut out[SIMD_LANES * (x.len() / SIMD_LANES)..]) {
        let inner = SCALE * (xc + CUBIC * (xc * xc * xc));
        *oc = 0.5 * xc * (1.0 + inner.tanh());
    }
}

/// 稳定 SIMD tanh:`1 - 2/(e^{2x}+1)`。大正负输入分别收敛到 ±1，不产生 NaN。
fn tanh_v(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let two = f32x8::splat(2.0);
    one - two / ((two * x).exp() + one)
}

fn sigmoid_v(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    one / (one + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::{SIMD_LANES, gelu, gelu_tanh_mul, silu_clamped_mul, situ_mul, swiglu_oai_mul};

    #[test]
    fn deepseek_swiglu限制gate和up但不平移up() {
        let gate = vec![20.0; SIMD_LANES];
        let up = vec![20.0; SIMD_LANES];
        let mut output = vec![0.0; SIMD_LANES];
        silu_clamped_mul(&gate, &up, 10.0, &mut output);
        let expected = 10.0 / (1.0 + (-10.0_f32).exp()) * 10.0;
        assert!((output[0] - expected).abs() < 1.0e-4);
    }

    #[test]
    fn situ按官方公式变换gate和up() {
        let gate = vec![0.0, 2.0, -3.0];
        let up = vec![1.0, 30.0, -30.0];
        let mut output = vec![0.0; gate.len()];
        situ_mul(&gate, &up, 4.0, Some(25.0), &mut output);
        for index in 0..gate.len() {
            let g: f32 = gate[index];
            let u: f32 = up[index];
            let expected = 4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp()) * 25.0 * (u / 25.0).tanh();
            assert!((output[index] - expected).abs() < 1.0e-5);
        }
    }

    #[test]
    fn swiglu_oai按官方公式限制gate和up() {
        let gate: Vec<f32> = (0..SIMD_LANES).map(|index| [10.0, -10.0, 2.0, -2.0][index % 4]).collect();
        let up: Vec<f32> = (0..SIMD_LANES).map(|index| [10.0, -10.0, 0.5, -0.5][index % 4]).collect();
        let mut output = vec![0.0; SIMD_LANES];

        swiglu_oai_mul(&gate, &up, 1.702, 7.0, &mut output);

        for index in 0..SIMD_LANES {
            let g = gate[index].min(7.0);
            let u = up[index].clamp(-7.0, 7.0);
            let expected = g / (1.0 + (-1.702 * g).exp()) * (u + 1.0);
            assert!((output[index] - expected).abs() < 1.0e-5);
        }
    }

    #[test]
    fn gelu_tanh乘up() {
        let gate = vec![0.0, 1.0, -1.0];
        let up = vec![2.0, 2.0, 2.0];
        let mut output = vec![0.0; 3];
        gelu_tanh_mul(&gate, &up, &mut output);
        assert_eq!(output[0], 0.0);
        assert!((output[1] - 1.682).abs() < 1.0e-3);
        assert!((output[2] + 0.317_310_5).abs() < 1.0e-3);
    }

    #[test]
    fn gelu按tanh近似且对称() {
        // 覆盖 SIMD 主路 + 标量尾部(长度 11)。
        let input: Vec<f32> = (-5..=5).map(|value| value as f32 * 0.3).collect();
        let mut output = vec![0.0; input.len()];
        gelu(&input, &mut output);
        for index in 0..input.len() {
            let x = input[index];
            let expected = 0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh());
            assert!((output[index] - expected).abs() < 1.0e-5, "index {index}: {x} -> {} vs {expected}", output[index]);
        }
        // tanh-GELU 近似关于原点不严格对称(gelu(-x) = gelu(x) - x)，校验该关系。
        for index in 0..input.len() {
            let x = input[index];
            if x.abs() < 1.0e-6 {
                continue;
            }
            let neg = output[input.len() - 1 - index];
            assert!((neg - (output[index] - x)).abs() < 1.0e-4);
        }
    }
}
