//! compressed-tensors `mxfp4-pack-quantized` 权重格式。

use crate::weight::container::safetensor::{SafetensorStore, TensorData};

/// MXFP4 每组共享一个 E8M0 scale。
pub const MXFP4_GROUP_SIZE: usize = 32;

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// 按行保存的 MXFP4 矩阵。
///
/// `packed` 的低 nibble 对应偶数列，高 nibble 对应后一个奇数列；
/// `scales` 每 32 个连续列保存一个 bias-127 的 E8M0 指数。
#[derive(Clone, Debug)]
pub struct Mxfp4Matrix {
    rows: usize,
    cols: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
}

impl Mxfp4Matrix {
    pub fn new(rows: usize, cols: usize, packed: Vec<u8>, scales: Vec<u8>) -> Result<Self, String> {
        if cols == 0 || !cols.is_multiple_of(MXFP4_GROUP_SIZE) {
            return Err(format!("MXFP4 列数必须是 {MXFP4_GROUP_SIZE} 的正整数倍，实际为 {cols}"));
        }

        let packed_len = rows.checked_mul(cols / 2).ok_or_else(|| format!("MXFP4 矩阵尺寸溢出：{rows}x{cols}"))?;
        if packed.len() != packed_len {
            return Err(format!("MXFP4 packed 长度不匹配：矩阵 {rows}x{cols} 需要 {packed_len} bytes，实际为 {}", packed.len()));
        }

        let scale_len = rows.checked_mul(cols / MXFP4_GROUP_SIZE).ok_or_else(|| format!("MXFP4 scale 尺寸溢出：{rows}x{cols}"))?;
        if scales.len() != scale_len {
            return Err(format!("MXFP4 scale 长度不匹配：矩阵 {rows}x{cols} 需要 {scale_len} bytes，实际为 {}", scales.len()));
        }

        Ok(Self { rows, cols, packed, scales })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    /// reference 路径只解一行，避免为了校验或 CPU oracle 展开整块专家权重。
    pub fn dequantize_row(&self, row: usize, output: &mut [f32]) -> Result<(), String> {
        if row >= self.rows {
            return Err(format!("MXFP4 行越界：矩阵有 {} 行，请求第 {row} 行", self.rows));
        }
        if output.len() != self.cols {
            return Err(format!("MXFP4 输出行长度不匹配：期望 {}，实际为 {}", self.cols, output.len()));
        }

        let packed_row = &self.packed[row * (self.cols / 2)..(row + 1) * (self.cols / 2)];
        let scale_row = &self.scales[row * (self.cols / MXFP4_GROUP_SIZE)..(row + 1) * (self.cols / MXFP4_GROUP_SIZE)];
        for (col, value) in output.iter_mut().enumerate() {
            let byte = packed_row[col / 2];
            let code = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            let magnitude = E2M1[(code & 0x07) as usize];
            let signed = if code & 0x08 == 0 { magnitude } else { -magnitude };
            let exponent = scale_row[col / MXFP4_GROUP_SIZE] as i32 - 127;
            *value = signed * 2.0f32.powi(exponent);
        }
        Ok(())
    }

    /// 通用 backend 的正确性路径；生产 Metal kernel 直接消费 packed 与 scales。
    pub fn decode(&self) -> Result<Vec<f32>, String> {
        let len = self.rows.checked_mul(self.cols).ok_or_else(|| format!("MXFP4 解码尺寸溢出：{}x{}", self.rows, self.cols))?;
        let mut output = vec![0.0; len];
        for row in 0..self.rows {
            self.dequantize_row(row, &mut output[row * self.cols..(row + 1) * self.cols])?;
        }
        Ok(output)
    }
}

/// 从 compressed-tensors 的两个 safetensors 张量加载一块 MXFP4 矩阵。
pub fn load_mxfp4_matrix(store: &SafetensorStore, packed_name: &str, scale_name: &str, rows: usize, cols: usize) -> Result<Mxfp4Matrix, String> {
    let packed = expect_bytes(store.load(packed_name)?, &[rows, cols / 2], "MXFP4 packed", &["U8", "I8", "F4", "F4_E2M1", "F4_E2M1FN_X2"])?;
    let scales = expect_bytes(store.load(scale_name)?, &[rows, cols / MXFP4_GROUP_SIZE], "MXFP4 scale", &["U8", "F8_E8M0"])?;
    Mxfp4Matrix::new(rows, cols, packed, scales)
}

fn expect_bytes(tensor: TensorData, shape: &[usize], kind: &str, dtypes: &[&str]) -> Result<Vec<u8>, String> {
    if !dtypes.contains(&tensor.dtype.as_str()) {
        return Err(format!("{} dtype={}，{kind} 期望 {}", tensor.name, tensor.dtype, dtypes.join("/"),));
    }
    if tensor.shape != shape {
        return Err(format!("{} shape={:?}，{kind} 期望 {shape:?}", tensor.name, tensor.shape));
    }
    Ok(tensor.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_low_nibble_first_and_e8m0_scale() {
        let mut packed = vec![0; MXFP4_GROUP_SIZE / 2];
        packed[0] = 0x91;
        let matrix = Mxfp4Matrix::new(1, MXFP4_GROUP_SIZE, packed, vec![126]).unwrap();
        let mut row = vec![0.0; MXFP4_GROUP_SIZE];

        matrix.dequantize_row(0, &mut row).unwrap();

        assert_eq!(row[0], 0.25);
        assert_eq!(row[1], -0.25);
        assert!(row[2..].iter().all(|value| *value == 0.0));
    }
}
