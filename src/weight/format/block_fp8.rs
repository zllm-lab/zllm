//! 二维分块 FP8 权重：E4M3 codes + E8M0 scale。
//!
//! DeepSeek-V4 的普通线性层按 128×128 block 共享一个 E8M0 scale。该格式与
//! GLM 的 E4M3 + F32 scale_inv 不同，保持独立类型，避免 backend 猜测 scale
//! 的布局和 dtype。

use crate::weight::codec::{fp8::decode_f8_e4m3, mxfp8::decode_e8m0};

#[derive(Clone, Debug)]
pub struct BlockFp8Matrix {
    codes: Vec<u8>,
    scales: Vec<u8>,
    block_rows: usize,
    block_cols: usize,
    pub rows: usize,
    pub cols: usize,
}

impl BlockFp8Matrix {
    pub fn new(codes: Vec<u8>, scales: Vec<u8>, rows: usize, cols: usize, block_rows: usize, block_cols: usize) -> Result<Self, String> {
        if rows == 0 || cols == 0 || block_rows == 0 || block_cols == 0 {
            return Err(format!("block FP8 shape=[{rows},{cols}] block=[{block_rows},{block_cols}] 非法"));
        }
        let code_bytes = rows.checked_mul(cols).ok_or("block FP8 codes 大小溢出")?;
        let scale_bytes = rows.div_ceil(block_rows).checked_mul(cols.div_ceil(block_cols)).ok_or("block FP8 scales 大小溢出")?;
        if codes.len() != code_bytes || scales.len() != scale_bytes {
            return Err(format!("block FP8 字节数 codes={}/{} scales={}/{}", codes.len(), code_bytes, scales.len(), scale_bytes,));
        }
        Ok(Self { codes, scales, block_rows, block_cols, rows, cols })
    }

    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub fn block_shape(&self) -> (usize, usize) {
        (self.block_rows, self.block_cols)
    }

    pub fn decode(&self) -> Vec<f32> {
        let scale_cols = self.cols.div_ceil(self.block_cols);
        let mut output = vec![0.0; self.rows * self.cols];
        for row in 0..self.rows {
            for column in 0..self.cols {
                let scale = self.scales[row / self.block_rows * scale_cols + column / self.block_cols];
                let scale = decode_e8m0(scale);
                output[row * self.cols + column] = decode_f8_e4m3(self.codes[row * self.cols + column]) * scale;
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_e4m3_with_e8m0_block_scale() {
        let matrix = BlockFp8Matrix::new(vec![0x38, 0x38], vec![128], 1, 2, 128, 128).unwrap();
        assert_eq!(matrix.decode(), vec![2.0, 2.0]);
    }
}
