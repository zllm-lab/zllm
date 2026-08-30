//! 扩散/流匹配采样算法领域层。
//!
//! 与 attention/、moe/、norm.rs 平行的独立领域，保存调制段规格与
//! 平台无关的 reference 实现。
//! 具体的网络 forward 算子（matmul/attention/AdaLN）留在 backend/。

use std::ops::Range;

/// 连续 token 段引用某一行 modulation 参数。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModulationSegment {
    pub rows: Range<usize>,
    pub modulation_row: usize,
}

pub fn validate_modulation_segments(segments: &[ModulationSegment], rows: usize, modulation_rows: usize) -> Result<(), String> {
    if rows == 0 || modulation_rows == 0 || segments.is_empty() {
        return Err(format!("modulation segments 非法: rows={rows} modulation_rows={modulation_rows} segments={}", segments.len()));
    }
    let mut cursor = 0;
    for (index, segment) in segments.iter().enumerate() {
        if segment.rows.start != cursor || segment.rows.end <= segment.rows.start || segment.rows.end > rows || segment.modulation_row >= modulation_rows {
            return Err(format!("modulation segment {index} 非法: rows={:?} modulation_row={} cursor={cursor} total_rows={rows} modulation_rows={modulation_rows}", segment.rows, segment.modulation_row,));
        }
        cursor = segment.rows.end;
    }
    if cursor != rows {
        return Err(format!("modulation segments 只覆盖 {cursor}/{rows} 行"));
    }
    Ok(())
}

/// CPU reference 使用的逐行 modulation 索引。
pub fn modulation_row_map(segments: &[ModulationSegment], rows: usize, modulation_rows: usize) -> Result<Vec<u32>, String> {
    validate_modulation_segments(segments, rows, modulation_rows)?;
    let mut row_map = vec![0; rows];
    for segment in segments {
        let modulation_row = u32::try_from(segment.modulation_row).map_err(|_| "modulation row 超过 u32".to_owned())?;
        row_map[segment.rows.clone()].fill(modulation_row);
    }
    Ok(row_map)
}
