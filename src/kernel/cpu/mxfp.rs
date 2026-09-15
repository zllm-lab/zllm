//! MXFP4 / MXFP8 单 token 在线 GEMV,作为 GPU 路径的 CPU oracle 与 decode 低延迟路径。

use crate::weight::codec::fp8::decode_f8_e4m3;
use crate::weight::codec::mxfp8::{MXFP8_BLOCK, decode_e8m0};
use wide::f32x8;

use super::nvfp4::decode_f4_e2m1;

const SIMD_LANES: usize = 8;
const MXFP4_GROUP: usize = crate::weight::format::mxfp4::MXFP4_GROUP_SIZE;

/// 单 token 直接读取 packed MXFP4 权重,避免 decode 阶段展开完整 F32 矩阵。
pub fn matvec_mxfp4_matrix(packed: &[u8], scales: &[u8], rows: usize, cols: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(MXFP4_GROUP) {
        return Err(format!("MXFP4 shape [{rows},{cols}] 无效,cols 必须按 {MXFP4_GROUP} 对齐"));
    }
    let scale_cols = cols / MXFP4_GROUP;
    if packed.len() != rows * cols / 2 || scales.len() != rows * scale_cols {
        return Err(format!("MXFP4 字节数不符: packed={}/{}, scales={}/{}", packed.len(), rows * cols / 2, scales.len(), rows * scale_cols));
    }
    if input.len() != cols || output.len() != rows {
        return Err(format!("MXFP4 GEMV shape 不符: input={}/{cols} output={}/{rows}", input.len(), output.len()));
    }

    for row in 0..rows {
        let packed_row = &packed[row * (cols / 2)..(row + 1) * (cols / 2)];
        let scale_row = &scales[row * scale_cols..(row + 1) * scale_cols];
        let mut sum = f32x8::splat(0.0);
        for block in 0..scale_cols {
            let column = block * MXFP4_GROUP;
            let scale = decode_e8m0(scale_row[block]);
            for lane_base in (0..MXFP4_GROUP).step_by(SIMD_LANES) {
                let mut weights = [0.0_f32; SIMD_LANES];
                for (lane, weight) in weights.iter_mut().enumerate() {
                    // packed 低 nibble 对应偶数列,高 nibble 对应后一个奇数列
                    let index = column + lane_base + lane;
                    let byte = packed_row[index / 2];
                    let code = if index % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    *weight = decode_f4_e2m1(code) * scale;
                }
                let lanes = f32x8::from(<[f32; SIMD_LANES]>::try_from(&input[column + lane_base..column + lane_base + SIMD_LANES]).unwrap());
                sum = lanes.mul_add(f32x8::from(weights), sum);
            }
        }
        output[row] = sum.reduce_add();
    }
    Ok(())
}

/// 单 token 直接读取 MXFP8 权重,避免 decode 阶段展开完整 F32 矩阵。
pub fn matvec_mxfp8_matrix(codes: &[u8], scale_inv: &[u8], rows: usize, cols: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(MXFP8_BLOCK) {
        return Err(format!("MXFP8 shape [{rows},{cols}] 无效,cols 必须按 {MXFP8_BLOCK} 对齐"));
    }
    let scale_cols = cols / MXFP8_BLOCK;
    if codes.len() != rows * cols || scale_inv.len() != rows * scale_cols {
        return Err(format!("MXFP8 字节数不符: codes={}/{}, scales={}/{}", codes.len(), rows * cols, scale_inv.len(), rows * scale_cols));
    }
    if input.len() != cols || output.len() != rows {
        return Err(format!("MXFP8 GEMV shape 不符: input={}/{cols} output={}/{rows}", input.len(), output.len()));
    }

    for row in 0..rows {
        let code_row = &codes[row * cols..(row + 1) * cols];
        let scale_row = &scale_inv[row * scale_cols..(row + 1) * scale_cols];
        let mut sum = f32x8::splat(0.0);
        for block in 0..scale_cols {
            let column = block * MXFP8_BLOCK;
            let scale = decode_e8m0(scale_row[block]);
            for lane_base in (0..MXFP8_BLOCK).step_by(SIMD_LANES) {
                let mut weights = [0.0_f32; SIMD_LANES];
                for (lane, weight) in weights.iter_mut().enumerate() {
                    *weight = decode_f8_e4m3(code_row[column + lane_base + lane]) * scale;
                }
                let lanes = f32x8::from(<[f32; SIMD_LANES]>::try_from(&input[column + lane_base..column + lane_base + SIMD_LANES]).unwrap());
                sum = lanes.mul_add(f32x8::from(weights), sum);
            }
        }
        output[row] = sum.reduce_add();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::codec::mxfp8::decode_mxfp8_matrix;
    use crate::weight::format::mxfp4::Mxfp4Matrix;

    #[test]
    fn mxfp4_gemv_matches_decoded_matrix() {
        let rows = 2;
        let cols = MXFP4_GROUP * 2;
        let packed: Vec<u8> = (0..rows * cols / 2).map(|index| (((index + index / 16) % 8) as u8) | (((((index * 3 + index / 16 + 1) % 8) as u8) | 0x08) << 4)).collect();
        let scales: Vec<u8> = vec![125, 130, 127, 128];
        let input: Vec<f32> = (0..cols).map(|index| index as f32 * 0.125 - 2.0).collect();

        let matrix = Mxfp4Matrix::new(rows, cols, packed.clone(), scales.clone()).unwrap();
        let decoded = matrix.decode().unwrap();
        let expected: Vec<f32> = decoded.chunks_exact(cols).map(|weight| weight.iter().zip(&input).map(|(weight, input)| weight * input).sum()).collect();

        let mut actual = vec![0.0; rows];
        matvec_mxfp4_matrix(&packed, &scales, rows, cols, &input, &mut actual).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5, "actual={actual} expected={expected}");
        }
    }

    #[test]
    fn mxfp8_gemv_matches_decoded_matrix() {
        let rows = 2;
        let cols = MXFP8_BLOCK * 2;
        // 避开 E4M3 的 NaN/SNaN 编码,其余值覆盖正负与不同指数
        let codes: Vec<u8> = (0..rows * cols).map(|index| ((index * 17 + 3) % 0x70) as u8).collect();
        let scale_inv: Vec<u8> = vec![127, 128, 125, 130];
        let input: Vec<f32> = (0..cols).map(|index| index as f32 * 0.25 - 8.0).collect();

        let mut decoded = vec![0.0; rows * cols];
        decode_mxfp8_matrix(&codes, &scale_inv, rows, cols, &mut decoded);
        let expected: Vec<f32> = decoded.chunks_exact(cols).map(|weight| weight.iter().zip(&input).map(|(weight, input)| weight * input).sum()).collect();

        let mut actual = vec![0.0; rows];
        matvec_mxfp8_matrix(&codes, &scale_inv, rows, cols, &input, &mut actual).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5, "actual={actual} expected={expected}");
        }
    }
}
