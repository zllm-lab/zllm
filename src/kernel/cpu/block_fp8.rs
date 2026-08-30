//! Block-scaled FP8 E4M3(E8M0 scale) GEMM / GEMV,单精度 oracle。
//!
//! 输入布局对齐 `BlockFp8Matrix`(`src/weight/block_fp8.rs`):
//! - `codes` row-major `[rows, cols]`,每元素是 1 字节 E4M3 编码。
//! - `scales` row-major `[rows/block_rows, cols/block_cols]`,每元素 1 字节 E8M0 scale,
//!   `scale = 2^(byte - 127)`。
//!
//! decode 与 `BlockFp8Matrix::decode()` 逐位对齐——单测用后者做 ground truth。

use rayon::prelude::*;
use wide::f32x8;

use crate::weight::codec::fp8::decode_f8_e4m3;

const SIMD_LANES: usize = 8;

/// `output[row] = Σ_col dequant(row,col) · input[col]`。
///
/// `block_rows` / `block_cols` 是 scale 共享的行列块大小(DeepSeek-V4 固定 128)。
/// decode 单 token 主用例。
#[allow(clippy::too_many_arguments)]
pub fn matvec_block_fp8_matrix(codes: &[u8], scales: &[u8], block_rows: usize, block_cols: usize, rows: usize, cols: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    validate(codes, scales, block_rows, block_cols, rows, cols)?;
    if input.len() != cols || output.len() != rows {
        return Err(format!("BlockFp8 GEMV input/output={}/{}，期望 {cols}/{rows}", input.len(), output.len()));
    }
    // 按行并行:每行 dequant + dot 一次,block_cols 内部用 SIMD lane=8 累加。
    if cols.is_multiple_of(SIMD_LANES) {
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let scale_row = row / block_rows;
            let scale_cols = cols.div_ceil(block_cols);
            let blocks = cols.div_ceil(block_cols);
            let mut buffer = [0.0f32; SIMD_LANES];
            let mut sum = f32x8::splat(0.0);
            for block_col in 0..blocks {
                let scale_byte = scales[scale_row * scale_cols + block_col];
                let scale_v = f32x8::splat(f32::from_bits(((scale_byte as u32).wrapping_add(0x3f800000u32)) << 23));
                let start = block_col * block_cols;
                let end = (start + block_cols).min(cols);
                let words = (end - start) / SIMD_LANES;
                for word in 0..words {
                    let column = start + word * SIMD_LANES;
                    let code_base = row * cols + column;
                    for lane in 0..SIMD_LANES {
                        buffer[lane] = decode_f8_e4m3(codes[code_base + lane]);
                    }
                    let code_v = f32x8::from(buffer);
                    let input_v = f32x8::from(<[f32; SIMD_LANES]>::try_from(&input[column..column + SIMD_LANES]).expect("SIMD input"));
                    sum = (code_v * scale_v).mul_add(input_v, sum);
                }
            }
            *out = sum.reduce_add();
        });
    } else {
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let mut sum = 0.0_f32;
            let scale_row = row / block_rows;
            let scale_cols = cols.div_ceil(block_cols);
            let blocks = cols.div_ceil(block_cols);
            for block_col in 0..blocks {
                let scale_byte = scales[scale_row * scale_cols + block_col];
                let scale = f32::from_bits(((scale_byte as u32).wrapping_add(0x3f800000u32)) << 23);
                let start = block_col * block_cols;
                let end = (start + block_cols).min(cols);
                for column in start..end {
                    sum += decode_f8_e4m3(codes[row * cols + column]) * scale * input[column];
                }
            }
            *out = sum;
        });
    }
    Ok(())
}

fn validate(codes: &[u8], scales: &[u8], block_rows: usize, block_cols: usize, rows: usize, cols: usize) -> Result<(), String> {
    if rows == 0 || cols == 0 || block_rows == 0 || block_cols == 0 {
        return Err(format!("BlockFp8 shape=[{rows},{cols}] block=[{block_rows},{block_cols}] 非法"));
    }
    let code_bytes = rows.checked_mul(cols).ok_or_else(|| "BlockFp8 codes 大小溢出".to_owned())?;
    let scale_bytes = rows.div_ceil(block_rows).checked_mul(cols.div_ceil(block_cols)).ok_or_else(|| "BlockFp8 scales 大小溢出".to_owned())?;
    if codes.len() != code_bytes || scales.len() != scale_bytes {
        return Err(format!("BlockFp8 字节数 codes={}/{} scales={}/{}", codes.len(), code_bytes, scales.len(), scale_bytes));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::format::block_fp8::BlockFp8Matrix;

    fn build_matrix(rows: usize, cols: usize) -> BlockFp8Matrix {
        // 合成 codes + scales,使矩阵内容可控、易断言。
        // codes: row * 7 + col(取低 8 位)作为非负 byte;scale 字节设为 127 ⇒ scale=1.0。
        let codes: Vec<u8> = (0..rows * cols).map(|index: usize| ((index.wrapping_mul(37).wrapping_add(13)) % 256) as u8).collect();
        let scales: Vec<u8> = vec![127u8; rows.div_ceil(128) * cols.div_ceil(128)];
        BlockFp8Matrix::new(codes, scales, rows, cols, 128, 128).expect("BlockFp8 new")
    }

    fn input_for(cols: usize, seed: usize) -> Vec<f32> {
        (0..cols).map(|column| (column as f32 + seed as f32) * 0.013).collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} 长度不一致");
        for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            let diff = (a - e).abs();
            // 浮点累加 + FP8 量化 8-bit mantissa:1e-2 容忍已足够覆盖测试规模。
            assert!(diff < 1.0e-2, "{label} index={index} actual={a} expected={e} diff={diff}");
        }
    }

    #[test]
    fn matvec_matches_decoded_ground_truth() {
        let rows = 4;
        let cols = 128;
        let matrix = build_matrix(rows, cols);
        let input = input_for(cols, 0);
        // ground truth = dequant-then-dot
        let decoded = matrix.decode();
        let mut expected = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0_f32;
            for column in 0..cols {
                sum += decoded[row * cols + column] * input[column];
            }
            expected[row] = sum;
        }
        let mut actual = vec![0.0f32; rows];
        matvec_block_fp8_matrix(matrix.codes(), matrix.scales(), 128, 128, rows, cols, &input, &mut actual).expect("matvec");
        assert_close(&actual, &expected, "matvec");
    }

    #[test]
    fn matvec_handles_partial_blocks() {
        // cols 不是 128 倍数,触发 block 边界。
        let rows = 3;
        let cols = 160; // 1 block + 32 tail
        let matrix = build_matrix(rows, cols);
        let input = input_for(cols, 5);
        let decoded = matrix.decode();
        let mut expected = vec![0.0_f32; rows];
        for row in 0..rows {
            for column in 0..cols {
                expected[row] += decoded[row * cols + column] * input[column];
            }
        }
        let mut actual = vec![0.0_f32; rows];
        matvec_block_fp8_matrix(matrix.codes(), matrix.scales(), 128, 128, rows, cols, &input, &mut actual).expect("matvec");
        assert_close(&actual, &expected, "partial-block matvec");
    }

    #[test]
    fn matmul_respects_non_unit_scale() {
        // scale 字节非 127 时 oracle 仍然对齐。
        let rows = 2;
        let cols = 256;
        let codes: Vec<u8> = (0..rows * cols).map(|index: usize| ((index.wrapping_mul(53)) % 256) as u8).collect();
        // block (0,0) 设 scale=128(2.0),block (0,1) 设 scale=126(0.5)
        let mut scales = vec![127u8; rows.div_ceil(128) * cols.div_ceil(128)];
        scales[0] = 128;
        scales[1] = 126;
        let matrix = BlockFp8Matrix::new(codes, scales, rows, cols, 128, 128).expect("new");
        let input = input_for(cols, 1);
        let decoded = matrix.decode();
        let mut expected = vec![0.0_f32; rows];
        for row in 0..rows {
            for column in 0..cols {
                expected[row] += decoded[row * cols + column] * input[column];
            }
        }
        let mut actual = vec![0.0_f32; rows];
        matvec_block_fp8_matrix(matrix.codes(), matrix.scales(), 128, 128, rows, cols, &input, &mut actual).expect("matvec");
        assert_close(&actual, &expected, "non-unit-scale matvec");
    }
}
