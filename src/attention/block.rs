//! 非对称块注意力规格与 CPU reference。

use std::{cell::RefCell, ops::Range};

use rayon::prelude::*;
use wide::f32x8;

use super::gqa::GqaGeometry;

#[derive(Debug, Clone, PartialEq)]
pub struct BlockAttentionSpec {
    pub geometry: GqaGeometry,
    pub score_scale: f32,
    /// 每个 query 行可见的 KV 半开区间。
    pub visible: Vec<Range<usize>>,
}

thread_local! {
    /// decode/DSpark 重复使用相同窗口，复用每线程 score 缓冲避免逐 head 分配。
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

impl BlockAttentionSpec {
    pub fn validate(&self, query_rows: usize, kv_rows: usize) -> Result<(), String> {
        self.geometry.validate()?;
        if !self.score_scale.is_finite() || self.score_scale <= 0.0 {
            return Err(format!("block attention score_scale={} 非法", self.score_scale));
        }
        if self.visible.len() != query_rows {
            return Err(format!("block attention visible={}，期望 query_rows={query_rows}", self.visible.len()));
        }
        for (query, range) in self.visible.iter().enumerate() {
            if range.is_empty() || range.end > kv_rows {
                return Err(format!("block attention query={query} 可见区间 {range:?} 超出 KV rows={kv_rows}"));
            }
        }
        Ok(())
    }

    pub fn full(geometry: GqaGeometry, query_rows: usize, kv_rows: usize, score_scale: f32) -> Self {
        Self { geometry, score_scale, visible: vec![0..kv_rows; query_rows] }
    }
}

pub fn attention_f32(query: &[f32], key: &[f32], value: &[f32], query_rows: usize, kv_rows: usize, spec: &BlockAttentionSpec) -> Result<Vec<f32>, String> {
    spec.validate(query_rows, kv_rows)?;
    let query_cols = spec.geometry.query_columns()?;
    let kv_cols = spec.geometry.kv_columns()?;
    if query.len() != query_rows * query_cols || key.len() != kv_rows * kv_cols || value.len() != key.len() {
        return Err(format!("block attention 数据长度非法: Q={} K={} V={}", query.len(), key.len(), value.len()));
    }
    let group = spec.geometry.group_size();
    let dim = spec.geometry.head_dim;
    let mut output = vec![0.0; query.len()];
    output.par_chunks_mut(dim).enumerate().for_each(|(task, out)| {
        let query_row = task / spec.geometry.num_heads;
        let head = task % spec.geometry.num_heads;
        let kv_head = head / group;
        let q = &query[query_row * query_cols + head * dim..query_row * query_cols + (head + 1) * dim];
        let range = spec.visible[query_row].clone();
        SCORES.with(|cell| {
            let scores = &mut *cell.borrow_mut();
            scores.clear();
            scores.reserve(range.len());
            let mut maximum = f32::NEG_INFINITY;
            for row in range.clone() {
                let k = &key[row * kv_cols + kv_head * dim..row * kv_cols + (kv_head + 1) * dim];
                let score = crate::kernel::cpu::matmul::dot(q, k) * spec.score_scale;
                maximum = maximum.max(score);
                scores.push(score);
            }
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - maximum).exp();
                    *score
                })
                .sum::<f32>();
            const LANES: usize = 8;
            for (row, &score) in range.zip(scores.iter()) {
                let probability = f32x8::splat(score / denominator);
                let v = &value[row * kv_cols + kv_head * dim..row * kv_cols + (kv_head + 1) * dim];
                let full = dim / LANES * LANES;
                for column in (0..full).step_by(LANES) {
                    let output = f32x8::from(<[f32; LANES]>::try_from(&out[column..column + LANES]).unwrap());
                    let value = f32x8::from(<[f32; LANES]>::try_from(&v[column..column + LANES]).unwrap());
                    let updated: [f32; LANES] = probability.mul_add(value, output).into();
                    out[column..column + LANES].copy_from_slice(&updated);
                }
                for column in full..dim {
                    out[column] += score / denominator * v[column];
                }
            }
        });
    });
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asymmetric_visibility_is_respected() {
        let spec = BlockAttentionSpec { geometry: GqaGeometry { num_heads: 1, num_kv_heads: 1, head_dim: 1 }, score_scale: 1.0, visible: vec![0..2, 1..3] };
        let output = attention_f32(&[0.0, 0.0], &[0.0, 0.0, 0.0], &[2.0, 6.0, 10.0], 2, 3, &spec).unwrap();
        assert_eq!(output, [4.0, 8.0]);
    }
}
