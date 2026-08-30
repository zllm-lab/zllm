//! 保持压缩态的 MXFP8 矩阵。

use crate::weight::codec::mxfp8::{MXFP8_BLOCK, decode_mxfp8_matrix};

pub struct Mxfp8MatrixBufferMut<'a> {
    pub codes: &'a mut [u8],
    pub scale_inv: &'a mut [u8],
    pub rows: usize,
    pub cols: usize,
}

pub struct Mxfp8Matrix {
    codes: Vec<u8>,
    scale_inv: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl Mxfp8Matrix {
    pub(crate) fn new(codes: Vec<u8>, scale_inv: Vec<u8>, rows: usize, cols: usize) -> Result<Self, String> {
        let (code_bytes, scale_bytes) = mxfp8_storage_lengths(rows, cols)?;
        if codes.len() != code_bytes || scale_inv.len() != scale_bytes {
            return Err(format!("MXFP8 字节数 codes={}/{} scale={}/{}", codes.len(), code_bytes, scale_inv.len(), scale_bytes));
        }
        Ok(Self { codes, scale_inv, rows, cols })
    }

    pub fn decode(&self) -> Vec<f32> {
        let mut output = vec![0.0; self.rows * self.cols];
        decode_mxfp8_matrix(&self.codes, &self.scale_inv, self.rows, self.cols, &mut output);
        output
    }

    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn scale_inv(&self) -> &[u8] {
        &self.scale_inv
    }
}

pub fn mxfp8_storage_lengths(rows: usize, cols: usize) -> Result<(usize, usize), String> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(MXFP8_BLOCK) {
        return Err(format!("MXFP8 shape [{rows},{cols}] 无效，cols 必须按 {MXFP8_BLOCK} 对齐"));
    }
    let code_bytes = rows.checked_mul(cols).ok_or_else(|| "MXFP8 矩阵大小溢出".to_owned())?;
    let scale_bytes = rows.checked_mul(cols / MXFP8_BLOCK).ok_or_else(|| "MXFP8 scale 大小溢出".to_owned())?;
    Ok((code_bytes, scale_bytes))
}
