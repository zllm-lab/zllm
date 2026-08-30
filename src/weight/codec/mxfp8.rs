//! MXFP8：E4M3 值与每 32 个连续输入元素一个 E8M0 scale。

use super::fp8::decode_f8_e4m3;

pub const MXFP8_BLOCK: usize = 32;

#[inline]
pub fn decode_e8m0(bits: u8) -> f32 {
    2.0_f32.powi(bits as i32 - 127)
}

pub fn decode_mxfp8_matrix(codes: &[u8], scale_inv: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let scale_cols = cols / MXFP8_BLOCK;
    for row in 0..rows {
        for block in 0..scale_cols {
            let scale = decode_e8m0(scale_inv[row * scale_cols + block]);
            let start = row * cols + block * MXFP8_BLOCK;
            for column in 0..MXFP8_BLOCK {
                output[start + column] = decode_f8_e4m3(codes[start + column]) * scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_is_power_of_two() {
        assert_eq!(decode_e8m0(126), 0.5);
        assert_eq!(decode_e8m0(127), 1.0);
        assert_eq!(decode_e8m0(128), 2.0);
    }

    #[test]
    fn matrix_uses_one_scale_per_32_values() {
        let codes = vec![0x38; MXFP8_BLOCK * 2];
        let mut output = vec![0.0; codes.len()];
        decode_mxfp8_matrix(&codes, &[127, 128], 1, MXFP8_BLOCK * 2, &mut output);
        assert!(output[..MXFP8_BLOCK].iter().all(|value| *value == 1.0));
        assert!(output[MXFP8_BLOCK..].iter().all(|value| *value == 2.0));
    }
}
