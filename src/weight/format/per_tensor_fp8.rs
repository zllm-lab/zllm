//! Per-tensor FP8 E4M3 矩阵。
//!
//! ComfyUI / 社区 H3 curve+FP8 checkpoint 把整个线性层量化为:
//! - FP8 E4M3 codes(`rows × cols` 个字节)
//! - 单一 F32 per-tensor scale(`{weight_name}_scale`)
//!
//! 跟官方 GLM-5.2 / DeepSeek-V4 的 128×128 block-FP8(`Fp8Matrix`)布局互不兼容,
//! 所以独立 struct 而不是复用 `Fp8Matrix.scale_inv` 做 broadcast。

use crate::weight::codec::fp8::decode_f8_e4m3;

/// 单个 per-tensor FP8 E4M3 矩阵。
#[derive(Clone, Debug)]
pub struct PerTensorFp8Matrix {
    pub codes: Vec<u8>,
    pub scale: f32,
    pub rows: usize,
    pub cols: usize,
}

impl PerTensorFp8Matrix {
    /// `codes` 长度必须是 `rows * cols`,`scale` 是 per-tensor 标量。
    pub fn new(codes: Vec<u8>, scale: f32, rows: usize, cols: usize) -> Result<Self, String> {
        let expected = rows.checked_mul(cols).ok_or_else(|| "per-tensor FP8 矩阵大小溢出".to_owned())?;
        if codes.len() != expected {
            return Err(format!("per-tensor FP8 codes 字节数 {} 期望 {expected}", codes.len()));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("per-tensor FP8 scale={scale} 必须是正有限数"));
        }
        Ok(Self { codes, scale, rows, cols })
    }

    /// 跟 `src/weight/codec/fp8.rs::decode_f8_e4m3` 等价的 CPU oracle,
    /// 给单测 / fallback 当 ground truth。
    pub fn decode(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for &code in &self.codes {
            out.push(decode_f8_e4m3(code) * self.scale);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_codes(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        (0..rows * cols).map(|i| (i as u8).wrapping_add(seed)).collect()
    }

    #[test]
    fn decode_matches_per_byte_loop() {
        let rows = 4;
        let cols = 8;
        let scale = 0.75;
        let codes = build_codes(rows, cols, 1);
        let matrix = PerTensorFp8Matrix::new(codes.clone(), scale, rows, cols).unwrap();

        let decoded = matrix.decode();
        assert_eq!(decoded.len(), rows * cols);
        for (index, code) in codes.iter().enumerate() {
            let expected = decode_f8_e4m3(*code) * scale;
            assert!((decoded[index] - expected).abs() < 1.0e-6, "decode[{index}] mismatch");
        }
    }

    #[test]
    fn rejects_wrong_code_length() {
        let err = PerTensorFp8Matrix::new(vec![0u8; 7], 1.0, 2, 4).unwrap_err();
        assert!(err.contains("per-tensor FP8 codes 字节数"), "{err}");
    }

    #[test]
    fn rejects_non_positive_scale() {
        let err = PerTensorFp8Matrix::new(vec![0u8; 4], 0.0, 2, 2).unwrap_err();
        assert!(err.contains("scale"), "{err}");
        let err = PerTensorFp8Matrix::new(vec![0u8; 4], -1.0, 2, 2).unwrap_err();
        assert!(err.contains("scale"), "{err}");
        let err = PerTensorFp8Matrix::new(vec![0u8; 4], f32::NAN, 2, 2).unwrap_err();
        assert!(err.contains("scale"), "{err}");
    }
}
