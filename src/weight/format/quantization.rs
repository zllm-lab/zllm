//! 保持压缩态的线性权重，以及 backend 消费的统一量化视图。

use crate::weight::{
    codec::{
        fp8::{decode_fp8_matrix, decode_fp8_matrix_bf16},
        groupwise::{decode_w4a16_matrix, decode_w8a16_matrix},
    },
    container::gguf::GgufMatrix,
    format::mxfp8::Mxfp8Matrix,
    format::nvfp4::Nvfp4Matrix,
    format::per_tensor_fp8::PerTensorFp8Matrix,
};

/// 官方 `F8_E4M3 + F32 scale_inv` 矩阵。
pub struct Fp8Matrix {
    pub(crate) codes: Vec<u8>,
    pub(crate) scale_inv: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl Fp8Matrix {
    /// pub(crate) 供 crate 内测试构造 FP8 权重，生产代码从权重文件加载。
    pub(crate) fn new(codes: Vec<u8>, scale_inv: Vec<u8>, rows: usize, cols: usize) -> Result<Self, String> {
        let code_bytes = rows.checked_mul(cols).ok_or_else(|| "FP8 矩阵大小溢出".to_owned())?;
        if codes.len() != code_bytes {
            return Err(format!("FP8 codes 字节数 {} 期望 {code_bytes}", codes.len()));
        }
        let scale_bytes = rows.div_ceil(128).checked_mul(cols.div_ceil(128)).and_then(|n| n.checked_mul(4)).ok_or_else(|| "FP8 scale 大小溢出".to_owned())?;
        if scale_inv.len() != scale_bytes {
            return Err(format!("FP8 scale_inv 字节数 {} 期望 {scale_bytes}", scale_inv.len()));
        }
        Ok(Self { codes, scale_inv, rows, cols })
    }

    pub fn decode(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.rows * self.cols];
        decode_fp8_matrix(&self.codes, &self.scale_inv, self.rows, self.cols, &mut out);
        out
    }

    pub fn decode_bf16(&self) -> Vec<u16> {
        let mut out = vec![0_u16; self.rows * self.cols];
        decode_fp8_matrix_bf16(&self.codes, &self.scale_inv, self.rows, self.cols, &mut out);
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDType {
    Bf16,
    F16,
    F32,
}

impl ScaleDType {
    pub const fn bytes(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    pub const fn metal_code(self) -> u32 {
        match self {
            Self::Bf16 => 0,
            Self::F16 => 1,
            Self::F32 => 2,
        }
    }
}

/// compressed-tensors `pack-quantized` W4A16 矩阵。
///
/// `packed` 按每行连续 int32 存放，8 个带符号 INT4 权重占一个 int32；
/// 量化值在文件中加 8 存成无符号 nibble，解码时恢复到 `[-8, 7]`。
#[derive(Clone, Debug)]
pub struct W4A16Matrix {
    packed: Vec<u8>,
    scales: Vec<u8>,
    scale_dtype: ScaleDType,
    group_size: usize,
    pub rows: usize,
    pub cols: usize,
}

impl W4A16Matrix {
    pub fn new(packed: Vec<u8>, scales: Vec<u8>, scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize) -> Result<Self, String> {
        if rows == 0 || cols == 0 || group_size == 0 || !cols.is_multiple_of(group_size) {
            return Err(format!("W4A16 shape=[{rows},{cols}] group_size={group_size} 无效"));
        }
        let packed_bytes = rows.checked_mul(cols.div_ceil(8)).and_then(|words| words.checked_mul(4)).ok_or_else(|| "W4A16 packed 大小溢出".to_owned())?;
        let scale_bytes = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(scale_dtype.bytes())).ok_or_else(|| "W4A16 scale 大小溢出".to_owned())?;
        if packed.len() != packed_bytes || scales.len() != scale_bytes {
            return Err(format!("W4A16 字节数 packed={}/{packed_bytes} scales={}/{scale_bytes}", packed.len(), scales.len(),));
        }
        Ok(Self { packed, scales, scale_dtype, group_size, rows, cols })
    }

    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub fn scale_dtype(&self) -> ScaleDType {
        self.scale_dtype
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn decode(&self) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0; self.rows * self.cols];
        decode_w4a16_matrix(&self.packed, &self.scales, self.scale_dtype, self.group_size, self.rows, self.cols, &mut out)?;
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn slice_rows(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        slice_groupwise_rows(self.packed(), self.scales(), self.scale_dtype, self.group_size, self.rows, self.cols, 4, range).and_then(|(packed, scales, rows)| Self::new(packed, scales, self.scale_dtype, self.group_size, rows, self.cols))
    }

    pub(crate) fn slice_columns(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        slice_groupwise_columns(self.packed(), self.scales(), self.scale_dtype, self.group_size, self.rows, self.cols, 4, range)
            .and_then(|(packed, scales, cols)| Self::new(packed, scales, self.scale_dtype, self.group_size, self.rows, cols))
    }
}

/// compressed-tensors `pack-quantized` W8A16 矩阵。
///
/// 与 [`W4A16Matrix`] 对称,差别仅在打包密度:每 int32 存 4 个有符号 INT8(而非 8 个 INT4)。
/// Int4-Int8Mix 的 attention/shared 线性层用此格式(W8A16 group=128)。
#[derive(Clone, Debug)]
pub struct W8A16Matrix {
    packed: Vec<u8>,
    scales: Vec<u8>,
    scale_dtype: ScaleDType,
    group_size: usize,
    convrot_group_size: Option<usize>,
    pub rows: usize,
    pub cols: usize,
}

/// MLX `affine` groupwise 矩阵：`weight = scale * unsigned_code + bias`。
pub struct MlxAffineMatrix {
    packed: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
    scale_dtype: ScaleDType,
    bits: usize,
    group_size: usize,
    pub rows: usize,
    pub cols: usize,
}

impl MlxAffineMatrix {
    #[allow(clippy::too_many_arguments)]
    pub fn new(packed: Vec<u8>, scales: Vec<u8>, biases: Vec<u8>, scale_dtype: ScaleDType, bits: usize, group_size: usize, rows: usize, cols: usize) -> Result<Self, String> {
        if !matches!(bits, 4 | 8) || rows == 0 || group_size == 0 || !cols.is_multiple_of(group_size) {
            return Err(format!("MLX affine shape=[{rows},{cols}] bits={bits} group={group_size} 无效"));
        }
        let values_per_word = 32 / bits;
        let packed_bytes = rows.checked_mul(cols.div_ceil(values_per_word)).and_then(|n| n.checked_mul(4)).ok_or("MLX affine packed 大小溢出")?;
        let parameter_bytes = rows.checked_mul(cols / group_size).and_then(|n| n.checked_mul(scale_dtype.bytes())).ok_or("MLX affine 参数大小溢出")?;
        if packed.len() != packed_bytes || scales.len() != parameter_bytes || biases.len() != parameter_bytes {
            return Err(format!("MLX affine 字节数 packed={}/{packed_bytes} scales={}/{parameter_bytes} biases={}/{parameter_bytes}", packed.len(), scales.len(), biases.len(),));
        }
        Ok(Self { packed, scales, biases, scale_dtype, bits, group_size, rows, cols })
    }

    pub fn packed(&self) -> &[u8] {
        &self.packed
    }
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
    pub fn biases(&self) -> &[u8] {
        &self.biases
    }
    pub fn scale_dtype(&self) -> ScaleDType {
        self.scale_dtype
    }
    pub fn bits(&self) -> usize {
        self.bits
    }
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn decode(&self) -> Result<Vec<f32>, String> {
        let values_per_word = 32 / self.bits;
        let packed_columns = self.cols.div_ceil(values_per_word);
        let groups = self.cols / self.group_size;
        let mask = (1_u32 << self.bits) - 1;
        let parameter = |bytes: &[u8], index: usize| match self.scale_dtype {
            ScaleDType::Bf16 => half::bf16::from_le_bytes(bytes[index * 2..index * 2 + 2].try_into().expect("BF16 参数")).to_f32(),
            ScaleDType::F16 => half::f16::from_le_bytes(bytes[index * 2..index * 2 + 2].try_into().expect("F16 参数")).to_f32(),
            ScaleDType::F32 => f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().expect("F32 参数")),
        };
        let mut output = vec![0.0; self.rows * self.cols];
        for row in 0..self.rows {
            for column in 0..self.cols {
                let word_offset = (row * packed_columns + column / values_per_word) * 4;
                let word = u32::from_le_bytes(self.packed[word_offset..word_offset + 4].try_into().expect("U32 packed"));
                let code = (word >> ((column % values_per_word) * self.bits)) & mask;
                let parameter_index = row * groups + column / self.group_size;
                output[row * self.cols + column] = parameter(&self.scales, parameter_index) * code as f32 + parameter(&self.biases, parameter_index);
            }
        }
        Ok(output)
    }
}

pub enum QuantizedMatrix {
    W4A16(W4A16Matrix),
    MlxAffine(MlxAffineMatrix),
    Gguf(GgufMatrix),
}

impl QuantizedMatrix {
    pub fn as_ref(&self) -> QuantizedMatrixRef<'_> {
        match self {
            Self::W4A16(matrix) => QuantizedMatrixRef::W4A16(matrix),
            Self::MlxAffine(matrix) => QuantizedMatrixRef::MlxAffine(matrix),
            Self::Gguf(matrix) => QuantizedMatrixRef::Gguf(matrix),
        }
    }

    pub fn rows(&self) -> usize {
        self.as_ref().rows()
    }
    pub fn cols(&self) -> usize {
        self.as_ref().cols()
    }
}

impl W8A16Matrix {
    pub fn new(packed: Vec<u8>, scales: Vec<u8>, scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize) -> Result<Self, String> {
        if rows == 0 || cols == 0 || group_size == 0 || !cols.is_multiple_of(group_size) {
            return Err(format!("W8A16 shape=[{rows},{cols}] group_size={group_size} 无效"));
        }
        // 每 int32 存 4 个 INT8,packed_bytes = rows * cols.div_ceil(4) * 4 = rows * cols(向 4 对齐)。
        let packed_bytes = rows.checked_mul(cols.div_ceil(4)).and_then(|words| words.checked_mul(4)).ok_or_else(|| "W8A16 packed 大小溢出".to_owned())?;
        let scale_bytes = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(scale_dtype.bytes())).ok_or_else(|| "W8A16 scale 大小溢出".to_owned())?;
        if packed.len() != packed_bytes || scales.len() != scale_bytes {
            return Err(format!("W8A16 字节数 packed={}/{packed_bytes} scales={}/{scale_bytes}", packed.len(), scales.len(),));
        }
        Ok(Self { packed, scales, scale_dtype, group_size, convrot_group_size: None, rows, cols })
    }

    pub fn new_convrot(packed: Vec<u8>, scales: Vec<u8>, convrot_group_size: usize, rows: usize, cols: usize) -> Result<Self, String> {
        // 蝶形旋转以 stride×4 展开,group_size 必须是 4 的幂(2 的幂不够,如 8/32 会让蝶形越界)。
        let is_power_of_four = convrot_group_size.is_power_of_two() && convrot_group_size.trailing_zeros() % 2 == 0;
        if !is_power_of_four || !cols.is_multiple_of(convrot_group_size) {
            return Err(format!("INT8 ConvRot shape=[{rows},{cols}] group_size={convrot_group_size} 无效: 必须是 4 的幂且整除 cols"));
        }
        let mut matrix = Self::new(packed, scales, ScaleDType::F32, cols, rows, cols)?;
        matrix.convrot_group_size = Some(convrot_group_size);
        Ok(matrix)
    }

    pub fn convrot_group_size(&self) -> Option<usize> {
        self.convrot_group_size
    }

    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub fn scale_dtype(&self) -> ScaleDType {
        self.scale_dtype
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn decode(&self) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0; self.rows * self.cols];
        decode_w8a16_matrix(&self.packed, &self.scales, self.scale_dtype, self.group_size, self.rows, self.cols, &mut out)?;
        if let Some(group_size) = self.convrot_group_size {
            let normalization = (group_size as f32).sqrt().recip();
            for row in out.chunks_exact_mut(self.cols) {
                for group in row.chunks_exact_mut(group_size) {
                    let mut stride = 1;
                    while stride < group_size {
                        for base in (0..group_size).step_by(stride * 4) {
                            for offset in 0..stride {
                                let index = base + offset;
                                let [a, b, c, d] = [group[index], group[index + stride], group[index + stride * 2], group[index + stride * 3]];
                                group[index] = a + b + c - d;
                                group[index + stride] = a + b - c + d;
                                group[index + stride * 2] = a - b + c + d;
                                group[index + stride * 3] = -a + b + c + d;
                            }
                        }
                        stride *= 4;
                    }
                    group.iter_mut().for_each(|value| *value *= normalization);
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn slice_rows(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        if self.convrot_group_size.is_some() {
            return Err("ConvRot W8 不支持张量并行切片".to_owned());
        }
        slice_groupwise_rows(self.packed(), self.scales(), self.scale_dtype, self.group_size, self.rows, self.cols, 8, range).and_then(|(packed, scales, rows)| Self::new(packed, scales, self.scale_dtype, self.group_size, rows, self.cols))
    }

    pub(crate) fn slice_columns(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        if self.convrot_group_size.is_some() {
            return Err("ConvRot W8 不支持张量并行切片".to_owned());
        }
        slice_groupwise_columns(self.packed(), self.scales(), self.scale_dtype, self.group_size, self.rows, self.cols, 8, range)
            .and_then(|(packed, scales, cols)| Self::new(packed, scales, self.scale_dtype, self.group_size, self.rows, cols))
    }
}

fn slice_groupwise_rows(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, bits: usize, range: std::ops::Range<usize>) -> Result<(Vec<u8>, Vec<u8>, usize), String> {
    if range.start >= range.end || range.end > rows {
        return Err(format!("W{bits} row slice={range:?}/{rows} 非法"));
    }
    let values_per_word = 32 / bits;
    let packed_row_bytes = cols.div_ceil(values_per_word) * 4;
    let scale_row_bytes = cols / group_size * scale_dtype.bytes();
    Ok((packed[range.start * packed_row_bytes..range.end * packed_row_bytes].to_vec(), scales[range.start * scale_row_bytes..range.end * scale_row_bytes].to_vec(), range.len()))
}

fn slice_groupwise_columns(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, bits: usize, range: std::ops::Range<usize>) -> Result<(Vec<u8>, Vec<u8>, usize), String> {
    let values_per_word = 32 / bits;
    if range.start >= range.end || range.end > cols || !range.start.is_multiple_of(group_size) || !range.end.is_multiple_of(group_size) || !range.start.is_multiple_of(values_per_word) || !range.end.is_multiple_of(values_per_word) {
        return Err(format!("W{bits} column slice={range:?}/{cols} group={group_size} 非法"));
    }
    let source_packed_row_bytes = cols.div_ceil(values_per_word) * 4;
    let source_scale_row_bytes = cols / group_size * scale_dtype.bytes();
    let packed_start = range.start / values_per_word * 4;
    let packed_bytes = range.len() / values_per_word * 4;
    let scale_start = range.start / group_size * scale_dtype.bytes();
    let scale_bytes = range.len() / group_size * scale_dtype.bytes();
    let mut sliced_packed = Vec::with_capacity(rows * packed_bytes);
    let mut sliced_scales = Vec::with_capacity(rows * scale_bytes);
    for row in 0..rows {
        sliced_packed.extend_from_slice(&packed[row * source_packed_row_bytes + packed_start..row * source_packed_row_bytes + packed_start + packed_bytes]);
        sliced_scales.extend_from_slice(&scales[row * source_scale_row_bytes + scale_start..row * source_scale_row_bytes + scale_start + scale_bytes]);
    }
    Ok((sliced_packed, sliced_scales, range.len()))
}

/// Backend 准备 resident 权重时消费的统一量化视图。
///
/// 这里只统一“矩阵是什么”，不统一各格式的加载、expert archive 和预取策略。
#[derive(Clone, Copy)]
pub enum QuantizedMatrixRef<'a> {
    BlockFp8(&'a crate::weight::format::block_fp8::BlockFp8Matrix),
    Fp8(&'a Fp8Matrix),
    PerTensorFp8(&'a PerTensorFp8Matrix),
    Mxfp8(&'a Mxfp8Matrix),
    Mxfp4(&'a crate::weight::format::mxfp4::Mxfp4Matrix),
    Nvfp4(&'a Nvfp4Matrix),
    W4A16(&'a W4A16Matrix),
    W8A16(&'a W8A16Matrix),
    MlxAffine(&'a MlxAffineMatrix),
    Gguf(&'a GgufMatrix),
}

impl QuantizedMatrixRef<'_> {
    pub fn rows(self) -> usize {
        match self {
            Self::BlockFp8(matrix) => matrix.rows,
            Self::Fp8(matrix) => matrix.rows,
            Self::PerTensorFp8(matrix) => matrix.rows,
            Self::Mxfp8(matrix) => matrix.rows,
            Self::Mxfp4(matrix) => matrix.rows(),
            Self::Nvfp4(matrix) => matrix.rows,
            Self::W4A16(matrix) => matrix.rows,
            Self::W8A16(matrix) => matrix.rows,
            Self::MlxAffine(matrix) => matrix.rows,
            Self::Gguf(matrix) => matrix.rows,
        }
    }

    pub fn cols(self) -> usize {
        match self {
            Self::BlockFp8(matrix) => matrix.cols,
            Self::Fp8(matrix) => matrix.cols,
            Self::PerTensorFp8(matrix) => matrix.cols,
            Self::Mxfp8(matrix) => matrix.cols,
            Self::Mxfp4(matrix) => matrix.cols(),
            Self::Nvfp4(matrix) => matrix.cols,
            Self::W4A16(matrix) => matrix.cols,
            Self::W8A16(matrix) => matrix.cols,
            Self::MlxAffine(matrix) => matrix.cols,
            Self::Gguf(matrix) => matrix.columns,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::BlockFp8(_) => "Block-FP8",
            Self::Fp8(_) => "FP8",
            Self::PerTensorFp8(_) => "PerTensor-FP8",
            Self::Mxfp8(_) => "MXFP8",
            Self::Mxfp4(_) => "MXFP4",
            Self::Nvfp4(_) => "NVFP4",
            Self::W4A16(_) => "W4A16-CT",
            Self::W8A16(matrix) if matrix.convrot_group_size().is_some() => "INT8-ConvRot",
            Self::W8A16(_) => "W8A16-CT",
            Self::MlxAffine(_) => "MLX-affine",
            Self::Gguf(_) => "GGUF",
        }
    }

    pub fn decode(self) -> Result<Vec<f32>, String> {
        match self {
            Self::BlockFp8(matrix) => Ok(matrix.decode()),
            Self::Fp8(matrix) => Ok(matrix.decode()),
            Self::PerTensorFp8(matrix) => Ok(matrix.decode()),
            Self::Mxfp8(matrix) => Ok(matrix.decode()),
            Self::Mxfp4(matrix) => matrix.decode(),
            Self::Nvfp4(matrix) => matrix.decode(),
            Self::W4A16(matrix) => matrix.decode(),
            Self::W8A16(matrix) => matrix.decode(),
            Self::MlxAffine(matrix) => matrix.decode(),
            Self::Gguf(matrix) => matrix.decode(),
        }
    }
}
