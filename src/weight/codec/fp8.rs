//! FP8 E4M3 权重解码。

/// 把单个 FP8 E4M3 字节解码为 f32。
pub fn decode_f8_e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        sign * 2.0_f32.powi(-6) * mantissa as f32 / 8.0
    } else if exponent == 0x0f && mantissa == 0x07 {
        f32::NAN
    } else {
        sign * 2.0_f32.powi(exponent as i32 - 7) * (1.0 + mantissa as f32 / 8.0)
    }
}

/// FP8 块大小(官方 GLM-5.2 用 128)。
pub const FP8_BLOCK: usize = 128;

/// 把 FP8 权重 + scale_inv 解码为行优先 f32 矩阵 `[rows, cols]`。
/// scale_inv 是 `[ceil(rows/128), ceil(cols/128)]` 的 f32 行优先。
pub fn decode_fp8_matrix(codes: &[u8], scale_inv: &[u8], rows: usize, cols: usize, out: &mut [f32]) {
    let scale_cols = cols.div_ceil(FP8_BLOCK);
    for r in 0..rows {
        let scale_row = r / FP8_BLOCK;
        for c in 0..cols {
            let scale_col = c / FP8_BLOCK;
            let off = (scale_row * scale_cols + scale_col) * 4;
            let scale = f32::from_le_bytes([scale_inv[off], scale_inv[off + 1], scale_inv[off + 2], scale_inv[off + 3]]);
            out[r * cols + c] = decode_f8_e4m3(codes[r * cols + c]) * scale;
        }
    }
}

/// 把官方 block-FP8 直接解码成 BF16 bits，避免先构造同尺寸 F32 矩阵。
pub fn decode_fp8_matrix_bf16(codes: &[u8], scale_inv: &[u8], rows: usize, cols: usize, out: &mut [u16]) {
    let values = std::array::from_fn::<_, 256, _>(|bits| decode_f8_e4m3(bits as u8));
    let scale_cols = cols.div_ceil(FP8_BLOCK);
    for r in 0..rows {
        let scale_row = r / FP8_BLOCK;
        for scale_col in 0..scale_cols {
            let start = scale_col * FP8_BLOCK;
            let end = (start + FP8_BLOCK).min(cols);
            let off = (scale_row * scale_cols + scale_col) * 4;
            let scale = f32::from_le_bytes([scale_inv[off], scale_inv[off + 1], scale_inv[off + 2], scale_inv[off + 3]]);
            for c in start..end {
                out[r * cols + c] = half::bf16::from_f32(values[codes[r * cols + c] as usize] * scale).to_bits();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_decode_matches_f32_oracle() {
        let rows = 129_usize;
        let cols = 257_usize;
        let codes = (0..rows * cols).map(|index| (index & 255) as u8).collect::<Vec<_>>();
        let scales = (0..rows.div_ceil(FP8_BLOCK) * cols.div_ceil(FP8_BLOCK)).flat_map(|index| (0.25 + index as f32 * 0.125).to_le_bytes()).collect::<Vec<_>>();
        let mut f32_values = vec![0.0; rows * cols];
        decode_fp8_matrix(&codes, &scales, rows, cols, &mut f32_values);
        let mut bf16_values = vec![0_u16; rows * cols];
        decode_fp8_matrix_bf16(&codes, &scales, rows, cols, &mut bf16_values);
        for (actual, expected) in bf16_values.into_iter().zip(f32_values) {
            assert_eq!(actual, half::bf16::from_f32(expected).to_bits());
        }
    }
}
