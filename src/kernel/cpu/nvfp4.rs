//! NVFP4 E2M1 权重解码；作为 Metal 路径的 CPU 数值 oracle。

use crate::weight::codec::fp8::decode_f8_e4m3;
use wide::f32x8;

pub const NVFP4_BLOCK: usize = 16;
const SIMD_LANES: usize = 8;

#[inline]
pub fn decode_f4_e2m1(code: u8) -> f32 {
    const LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = LEVELS[(code & 0x07) as usize];
    if code & 0x08 == 0 { value } else { -value }
}

pub fn decode_nvfp4_matrix(codes: &[u8], scales: &[u8], global_scale: f32, rows: usize, cols: usize, output: &mut [f32]) -> Result<(), String> {
    let elements = validate_nvfp4(codes, scales, rows, cols)?;
    if output.len() != elements {
        return Err(format!("NVFP4 output 长度不符: output={}/{elements}", output.len(),));
    }

    let blocks_per_row = cols / NVFP4_BLOCK;
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let output_base = row * cols + block * NVFP4_BLOCK;
            let code_base = output_base / 2;
            let scale = decode_f8_e4m3(scales[row * blocks_per_row + block]) * global_scale;
            for byte in 0..NVFP4_BLOCK / 2 {
                let packed = codes[code_base + byte];
                output[output_base + byte * 2] = decode_f4_e2m1(packed & 0x0f) * scale;
                output[output_base + byte * 2 + 1] = decode_f4_e2m1(packed >> 4) * scale;
            }
        }
    }
    Ok(())
}

/// 单 token 直接读取 packed NVFP4 权重，避免 decode 阶段展开完整 F32 矩阵。
pub fn matvec_nvfp4_matrix(codes: &[u8], scales: &[u8], global_scale: f32, rows: usize, cols: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    validate_nvfp4(codes, scales, rows, cols)?;
    if input.len() != cols || output.len() != rows {
        return Err(format!("NVFP4 GEMV shape 不符: input={}/{cols} output={}/{rows}", input.len(), output.len(),));
    }
    debug_assert!(NVFP4_BLOCK.is_multiple_of(SIMD_LANES));

    let blocks_per_row = cols / NVFP4_BLOCK;
    for (row, value) in output.iter_mut().enumerate() {
        let mut sum = f32x8::splat(0.0);
        for block in 0..blocks_per_row {
            let column = block * NVFP4_BLOCK;
            let weight_base = row * cols + column;
            let scale = decode_f8_e4m3(scales[row * blocks_per_row + block]) * global_scale;
            for lane_base in (0..NVFP4_BLOCK).step_by(SIMD_LANES) {
                let mut weights = [0.0_f32; SIMD_LANES];
                for (lane, weight) in weights.iter_mut().enumerate() {
                    let index = weight_base + lane_base + lane;
                    let packed = codes[index / 2];
                    let code = if index & 1 == 0 { packed & 0x0f } else { packed >> 4 };
                    *weight = decode_f4_e2m1(code) * scale;
                }
                let input = f32x8::from(<[f32; SIMD_LANES]>::try_from(&input[column + lane_base..column + lane_base + SIMD_LANES]).unwrap());
                sum = input.mul_add(f32x8::from(weights), sum);
            }
        }
        *value = sum.reduce_add();
    }
    Ok(())
}

fn validate_nvfp4(codes: &[u8], scales: &[u8], rows: usize, cols: usize) -> Result<usize, String> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(NVFP4_BLOCK) {
        return Err(format!("NVFP4 shape [{rows},{cols}] 无效，cols 必须按 {NVFP4_BLOCK} 对齐"));
    }
    let elements = rows.checked_mul(cols).ok_or_else(|| "NVFP4 矩阵大小溢出".to_owned())?;
    let expected_codes = elements / 2;
    let expected_scales = rows.checked_mul(cols / NVFP4_BLOCK).ok_or_else(|| "NVFP4 scale 大小溢出".to_owned())?;
    if codes.len() != expected_codes || scales.len() != expected_scales {
        return Err(format!("NVFP4 字节数不符: codes={}/{expected_codes} scales={}/{expected_scales}", codes.len(), scales.len(),));
    }
    Ok(elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_e2m1_levels_and_two_level_scale() {
        assert_eq!(decode_f4_e2m1(0x07), 6.0);
        assert_eq!(decode_f4_e2m1(0x0f), -6.0);

        let mut output = [0.0; NVFP4_BLOCK];
        let codes = [0x21; NVFP4_BLOCK / 2];
        let scales = [0x40]; // E4M3 2.0
        decode_nvfp4_matrix(&codes, &scales, 0.25, 1, NVFP4_BLOCK, &mut output).unwrap();
        assert_eq!(output[0], 0.25); // 0.5 * 2.0 * 0.25
        assert_eq!(output[1], 0.5); // 1.0 * 2.0 * 0.25
    }

    #[test]
    fn packed_gemv_matches_decoded_matrix() {
        let rows = 2;
        let cols = NVFP4_BLOCK;
        let codes: Vec<u8> = (0..rows * cols / 2).map(|index| ((index * 2 + 1) as u8 & 0x0f) | (((index * 2 + 2) as u8 & 0x0f) << 4)).collect();
        let scales = vec![0x38, 0x40]; // E4M3 1.0, 2.0
        let input: Vec<f32> = (0..cols).map(|index| index as f32 * 0.125 - 0.5).collect();
        let mut decoded = vec![0.0; rows * cols];
        decode_nvfp4_matrix(&codes, &scales, 0.25, rows, cols, &mut decoded).unwrap();
        let expected: Vec<f32> = decoded.chunks_exact(cols).map(|weight| weight.iter().zip(&input).map(|(weight, input)| weight * input).sum()).collect();
        let mut actual = vec![0.0; rows];
        matvec_nvfp4_matrix(&codes, &scales, 0.25, rows, cols, &input, &mut actual).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4, "actual={actual} expected={expected}");
        }
    }
}
