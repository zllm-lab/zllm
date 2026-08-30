//! RoPE 的平台无关预计算表。

/// RoPE 频率语义。Proportional RoPE 保持完整 head 配对布局，只把未启用频率设为 0。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeSpec {
    Default { rotary_dim: usize, theta: f32 },
    Proportional { head_dim: usize, theta: f32, active_fraction: f32 },
    Yarn { rotary_dim: usize, theta: f32, factor: f32, original_context: usize, beta_fast: f32, beta_slow: f32 },
}

impl RopeSpec {
    pub fn validate(&self) -> Result<(), String> {
        let (dim, theta) = (self.rotary_dim(), self.theta());
        if dim == 0 || !dim.is_multiple_of(2) || !theta.is_finite() || theta <= 0.0 {
            return Err(format!("RoPE 参数非法: dim={dim} theta={theta}"));
        }
        if let Self::Proportional { active_fraction, .. } = self
            && (!active_fraction.is_finite() || !(0.0..=1.0).contains(active_fraction))
        {
            return Err(format!("proportional RoPE active_fraction={active_fraction} 非法"));
        }
        if let Self::Yarn { factor, original_context, beta_fast, beta_slow, .. } = self
            && (!factor.is_finite() || *factor <= 0.0 || *original_context == 0 || !beta_fast.is_finite() || *beta_fast <= 0.0 || !beta_slow.is_finite() || *beta_slow <= 0.0)
        {
            return Err(format!("YaRN 参数非法: factor={factor} original_context={original_context} beta_fast={beta_fast} beta_slow={beta_slow}"));
        }
        Ok(())
    }

    pub fn rotary_dim(&self) -> usize {
        match self {
            Self::Default { rotary_dim, .. } | Self::Yarn { rotary_dim, .. } => *rotary_dim,
            Self::Proportional { head_dim, .. } => *head_dim,
        }
    }

    pub fn theta(&self) -> f32 {
        match self {
            Self::Default { theta, .. } | Self::Proportional { theta, .. } | Self::Yarn { theta, .. } => *theta,
        }
    }

    fn active_pairs(&self) -> usize {
        let half = self.rotary_dim() / 2;
        match self {
            Self::Default { .. } | Self::Yarn { .. } => half,
            Self::Proportional { head_dim, active_fraction, .. } => ((*active_fraction * *head_dim as f32) / 2.0).floor() as usize,
        }
    }
}

/// 预计算的 RoPE cos/sin 表 `[seq_len, rotary_dim/2]` 行优先。
#[derive(Clone)]
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub rotary_dim: usize,
    pub seq_len: usize,
}

impl RopeTable {
    pub fn precompute(seq_len: usize, rotary_dim: usize, theta: f32) -> Self {
        // 参数来自调用方模型配置；各模型 Config/LayerSpec 构造处的 validate
        // （如 MlaSpec::validate、HybridGqaLayerSpec::validate）已保证合法，
        // 这里 panic 时带上实际参数便于定位是哪个配置漏了校验。
        Self::from_spec(seq_len, RopeSpec::Default { rotary_dim, theta }).unwrap_or_else(|error| panic!("RopeTable::precompute 参数非法: {error}"))
    }

    pub fn from_spec(seq_len: usize, spec: RopeSpec) -> Result<Self, String> {
        spec.validate()?;
        let rotary_dim = spec.rotary_dim();
        let half = rotary_dim / 2;
        let active_pairs = spec.active_pairs();
        let theta = spec.theta();
        let freqs: Vec<f32> = (0..half)
            .map(|pair| {
                if pair >= active_pairs {
                    return 0.0;
                }
                let base = theta.powf(-2.0 * pair as f32 / rotary_dim as f32);
                let RopeSpec::Yarn { factor, original_context, beta_fast, beta_slow, .. } = spec else {
                    return base;
                };
                let correction = |rotations: f32| rotary_dim as f32 * (original_context as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln() / (2.0 * theta.ln());
                let low = correction(beta_fast).floor().max(0.0);
                let high = correction(beta_slow).ceil().min((rotary_dim - 1) as f32);
                let width = if low == high { 0.001 } else { high - low };
                let ramp = ((pair as f32 - low) / width).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                base * (smooth + (1.0 - smooth) / factor)
            })
            .collect();
        let mut cos = vec![0.0; seq_len * half];
        let mut sin = vec![0.0; seq_len * half];
        for pos in 0..seq_len {
            for (i, &freq) in freqs.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos[pos * half + i] = angle.cos();
                sin[pos * half + i] = angle.sin();
            }
        }
        Ok(Self { cos, sin, rotary_dim, seq_len })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryLayout {
    SplitHalf,
    Interleaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryPlacement {
    Prefix,
    Suffix,
}

/// 平台无关的 f32 RoPE reference。backend 只负责张量适配或使用设备 kernel 替代。
#[allow(clippy::too_many_arguments)]
pub fn apply_f32(input: &[f32], rows: usize, columns: usize, head_count: usize, rotary_dim: usize, position: usize, cos: &[f32], sin: &[f32], layout: RotaryLayout, placement: RotaryPlacement) -> Result<Vec<f32>, String> {
    if input.len() != rows.checked_mul(columns).ok_or("RoPE shape 溢出")? {
        return Err(format!("RoPE input 长度 {} 与 shape [{rows},{columns}] 不符", input.len()));
    }
    if head_count == 0 || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || !columns.is_multiple_of(head_count) {
        return Err(format!("RoPE 参数非法: shape=[{rows},{columns}] heads={head_count} rotary={rotary_dim}"));
    }
    if cos.len() != sin.len() {
        return Err(format!("RoPE cos/sin 长度不一致: {}/{}", cos.len(), sin.len()));
    }
    let head_dim = columns / head_count;
    if rotary_dim > head_dim {
        return Err(format!("RoPE rotary={rotary_dim} 超过 head_dim={head_dim}"));
    }
    let half = rotary_dim / 2;
    let required = position.checked_add(rows).and_then(|value| value.checked_mul(half)).ok_or("RoPE table offset 溢出")?;
    if required > cos.len() {
        return Err(format!("RoPE table 长度 {}，至少需要 {required}", cos.len()));
    }

    let mut output = input.to_vec();
    for row in 0..rows {
        let cos_row = &cos[(position + row) * half..(position + row + 1) * half];
        let sin_row = &sin[(position + row) * half..(position + row + 1) * half];
        for head in 0..head_count {
            let head_start = row * columns + head * head_dim;
            let start = match placement {
                RotaryPlacement::Prefix => head_start,
                RotaryPlacement::Suffix => head_start + head_dim - rotary_dim,
            };
            let source = &input[start..start + rotary_dim];
            for pair in 0..half {
                let (real_index, imaginary_index) = match layout {
                    RotaryLayout::SplitHalf => (pair, half + pair),
                    RotaryLayout::Interleaved => (pair * 2, pair * 2 + 1),
                };
                let real = source[real_index];
                let imaginary = source[imaginary_index];
                output[start + real_index] = real * cos_row[pair] - imaginary * sin_row[pair];
                output[start + imaginary_index] = imaginary * cos_row[pair] + real * sin_row[pair];
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_f32_rotates_split_half_layout() {
        let output = apply_f32(&[1.0, 2.0, 3.0, 4.0], 1, 4, 1, 4, 0, &[0.0, 0.0], &[1.0, 1.0], RotaryLayout::SplitHalf, RotaryPlacement::Prefix).unwrap();
        assert_eq!(output, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn apply_f32_rotates_interleaved_layout() {
        let output = apply_f32(&[1.0, 2.0, 3.0, 4.0], 1, 4, 1, 4, 0, &[0.0, 0.0], &[1.0, 1.0], RotaryLayout::Interleaved, RotaryPlacement::Prefix).unwrap();
        assert_eq!(output, vec![-2.0, 1.0, -4.0, 3.0]);
    }

    #[test]
    fn apply_f32_rejects_short_table() {
        assert!(apply_f32(&[0.0; 4], 1, 4, 1, 4, 1, &[1.0, 1.0], &[0.0, 0.0], RotaryLayout::SplitHalf, RotaryPlacement::Suffix).is_err());
    }

    #[test]
    fn proportional_rope_keeps_full_head_pairing() {
        let table = RopeTable::from_spec(2, RopeSpec::Proportional { head_dim: 8, theta: 10_000.0, active_fraction: 0.5 }).unwrap();
        assert_eq!(table.rotary_dim, 8);
        assert_eq!(&table.cos[6..8], &[1.0, 1.0]);
        assert_eq!(&table.sin[6..8], &[0.0, 0.0]);
    }

    #[test]
    fn yarn_preserves_low_frequency_and_interpolates_high_frequency() {
        let yarn = RopeTable::from_spec(2, RopeSpec::Yarn { rotary_dim: 64, theta: 160_000.0, factor: 16.0, original_context: 65_536, beta_fast: 32.0, beta_slow: 1.0 }).unwrap();
        let plain = RopeTable::precompute(2, 64, 160_000.0);
        assert_eq!(yarn.sin[32], plain.sin[32]);
        assert!(yarn.sin[63].abs() < plain.sin[63].abs());
    }
}
