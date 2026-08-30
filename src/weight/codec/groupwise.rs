//! compressed-tensors W4A16/W8A16 group-wise reference decode。

use half::{bf16, f16};
use rayon::prelude::*;

use crate::weight::format::quantization::{ScaleDType, W8A16Matrix};

#[allow(clippy::too_many_arguments)]
pub fn decode_w4a16_matrix(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, output: &mut [f32]) -> Result<(), String> {
    validate(packed, scales, scale_dtype, group_size, rows, cols, output.len(), 8, "W4A16")?;
    let packed_row_bytes = cols.div_ceil(8) * 4;
    decode_rows(scales, scale_dtype, group_size, rows, cols, output, |row, column| {
        let row = &packed[row * packed_row_bytes..(row + 1) * packed_row_bytes];
        let offset = column / 8 * 4;
        let word = u32::from_le_bytes(row[offset..offset + 4].try_into().expect("W4A16 word"));
        (((word >> ((column % 8) * 4)) & 0x0f) as i8) - 8
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn decode_w8a16_matrix(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, output: &mut [f32]) -> Result<(), String> {
    validate(packed, scales, scale_dtype, group_size, rows, cols, output.len(), 4, "W8A16")?;
    let packed_row_bytes = cols.div_ceil(4) * 4;
    decode_rows(scales, scale_dtype, group_size, rows, cols, output, |row, column| (i16::from(packed[row * packed_row_bytes + column]) - 128) as i8);
    Ok(())
}

/// 从 BF16 矩阵按调用方顺序抽行并量化为 compressed-tensors W8A16。
///
/// 输出头可传入完整词表，draft head 也可只抽取裁剪词表；直接读源行能避免
/// 先复制一份同等大小的 BF16 中间矩阵。各行独立量化，启动阶段并行执行。
pub fn quantize_bf16_rows_w8a16(values: &[u8], source_rows: usize, cols: usize, selected_rows: &[u32], group_size: usize) -> Result<W8A16Matrix, String> {
    let source_row_bytes = cols.checked_mul(2).ok_or("BF16 source row 大小溢出")?;
    let expected = source_rows.checked_mul(source_row_bytes).ok_or("BF16 source 矩阵大小溢出")?;
    if values.len() != expected {
        return Err(format!("BF16 source 字节数 {}，期望 {expected}", values.len()));
    }
    quantize_rows_w8a16(source_rows, cols, selected_rows, group_size, |row, column| {
        let offset = row * source_row_bytes + column * 2;
        bf16::from_le_bytes([values[offset], values[offset + 1]]).to_f32()
    })
}

/// 从 F16 矩阵直接逐行量化，避免为大词表 LM head 扩出整份 F32 中间矩阵。
pub fn quantize_f16_rows_w8a16(values: &[f16], source_rows: usize, cols: usize, selected_rows: &[u32], group_size: usize) -> Result<W8A16Matrix, String> {
    let expected = source_rows.checked_mul(cols).ok_or("F16 source 矩阵大小溢出")?;
    if values.len() != expected {
        return Err(format!("F16 source 元素数 {}，期望 {expected}", values.len()));
    }
    quantize_rows_w8a16(source_rows, cols, selected_rows, group_size, |row, column| values[row * cols + column].to_f32())
}

/// F32 源权重的通用 W8A16 行量化；非 BF16 checkpoint 与已解码标准格式共用。
pub fn quantize_f32_rows_w8a16(values: &[f32], source_rows: usize, cols: usize, selected_rows: &[u32], group_size: usize) -> Result<W8A16Matrix, String> {
    let expected = source_rows.checked_mul(cols).ok_or("F32 source 矩阵大小溢出")?;
    if values.len() != expected {
        return Err(format!("F32 source 元素数 {}，期望 {expected}", values.len()));
    }
    quantize_rows_w8a16(source_rows, cols, selected_rows, group_size, |row, column| values[row * cols + column])
}

fn quantize_rows_w8a16(source_rows: usize, cols: usize, selected_rows: &[u32], group_size: usize, value: impl Fn(usize, usize) -> f32 + Sync) -> Result<W8A16Matrix, String> {
    if source_rows == 0 || cols == 0 || selected_rows.is_empty() || group_size == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("source -> W8A16 shape=[{source_rows},{cols}] selected={} group={group_size} 无效", selected_rows.len()));
    }
    if let Some(row) = selected_rows.iter().find(|&&row| row as usize >= source_rows) {
        return Err(format!("selected row={row} 超出 source_rows={source_rows}"));
    }

    let packed_row_bytes = cols.div_ceil(4).checked_mul(4).ok_or("W8A16 packed row 大小溢出")?;
    let scale_row_bytes = (cols / group_size).checked_mul(2).ok_or("W8A16 scale row 大小溢出")?;
    let mut packed = vec![128_u8; selected_rows.len().checked_mul(packed_row_bytes).ok_or("W8A16 packed 大小溢出")?];
    let mut scales = vec![0_u8; selected_rows.len().checked_mul(scale_row_bytes).ok_or("W8A16 scales 大小溢出")?];
    packed.par_chunks_mut(packed_row_bytes).zip(scales.par_chunks_mut(scale_row_bytes)).zip(selected_rows.par_iter()).try_for_each(|((packed_row, scale_row), &source_row)| -> Result<(), String> {
        let source_row = source_row as usize;
        for group in 0..cols / group_size {
            let column_start = group * group_size;
            let mut maximum = 0.0_f32;
            for column in column_start..column_start + group_size {
                let value = value(source_row, column);
                if !value.is_finite() {
                    return Err(format!("source row={source_row} group={group} 包含非有限值"));
                }
                maximum = maximum.max(value.abs());
            }
            let scale = if maximum > 0.0 { maximum / 127.0 } else { 1.0 };
            scale_row[group * 2..group * 2 + 2].copy_from_slice(&bf16::from_f32(scale).to_le_bytes());
            for offset in 0..group_size {
                let value = value(source_row, column_start + offset);
                let quantized = (value / scale).round().clamp(-127.0, 127.0) as i16;
                packed_row[column_start + offset] = (quantized + 128) as u8;
            }
        }
        Ok(())
    })?;
    W8A16Matrix::new(packed, scales, ScaleDType::Bf16, group_size, selected_rows.len(), cols)
}

#[allow(clippy::too_many_arguments)]
fn decode_rows(scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, output: &mut [f32], code: impl Fn(usize, usize) -> i8) {
    let groups = cols / group_size;
    for row in 0..rows {
        for group in 0..groups {
            let scale = read_scale(scales, scale_dtype, row * groups + group);
            let begin = group * group_size;
            for column in begin..begin + group_size {
                output[row * cols + column] = f32::from(code(row, column)) * scale;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, output_elements: usize, values_per_word: usize, name: &str) -> Result<(), String> {
    if rows == 0 || cols == 0 || group_size == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("{name} shape=[{rows},{cols}] group_size={group_size} 无效"));
    }
    let packed_bytes = rows.checked_mul(cols.div_ceil(values_per_word)).and_then(|words| words.checked_mul(4)).ok_or_else(|| format!("{name} packed 大小溢出"))?;
    let scale_bytes = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(scale_dtype.bytes())).ok_or_else(|| format!("{name} scale 大小溢出"))?;
    let expected_output = rows.checked_mul(cols).ok_or_else(|| format!("{name} output 大小溢出"))?;
    if packed.len() != packed_bytes || scales.len() != scale_bytes || output_elements != expected_output {
        return Err(format!("{name} buffer packed={}/{packed_bytes} scales={}/{scale_bytes} output={output_elements}/{expected_output}", packed.len(), scales.len(),));
    }
    Ok(())
}

fn read_scale(scales: &[u8], dtype: ScaleDType, index: usize) -> f32 {
    let offset = index * dtype.bytes();
    match dtype {
        ScaleDType::Bf16 => bf16::from_le_bytes(scales[offset..offset + 2].try_into().expect("BF16 scale")).to_f32(),
        ScaleDType::F16 => f16::from_le_bytes(scales[offset..offset + 2].try_into().expect("F16 scale")).to_f32(),
        ScaleDType::F32 => f32::from_le_bytes(scales[offset..offset + 4].try_into().expect("F32 scale")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_row_selection_quantizes_in_requested_order() {
        let rows = [[-1.0, -0.5, 0.25, 1.0, -2.0, -1.0, 1.0, 2.0], [9.0; 8], [0.125, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0, -1.5]];
        let values = rows.into_iter().flatten().flat_map(|value| bf16::from_f32(value).to_le_bytes()).collect::<Vec<_>>();
        let matrix = quantize_bf16_rows_w8a16(&values, 3, 8, &[2, 0], 4).unwrap();
        let decoded = matrix.decode().unwrap();

        assert_eq!((matrix.rows, matrix.cols), (2, 8));
        for (actual, expected) in decoded[..8].iter().zip(rows[2]) {
            assert!((actual - expected).abs() < 0.02, "selected row 2: {actual} vs {expected}");
        }
        for (actual, expected) in decoded[8..].iter().zip(rows[0]) {
            assert!((actual - expected).abs() < 0.02, "selected row 0: {actual} vs {expected}");
        }
    }

    #[test]
    fn f16_rows_quantize_without_f32_matrix_copy() {
        let values = [f16::from_f32(-1.0), f16::from_f32(-0.5), f16::from_f32(0.5), f16::from_f32(1.0)];
        let matrix = quantize_f16_rows_w8a16(&values, 1, 4, &[0], 4).unwrap();
        let decoded = matrix.decode().unwrap();

        for (actual, expected) in decoded.iter().zip([-1.0, -0.5, 0.5, 1.0]) {
            assert!((actual - expected).abs() < 0.02, "{actual} vs {expected}");
        }
    }
}
