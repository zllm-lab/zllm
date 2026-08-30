//! MLA 的 CPU 实现。

use rayon::prelude::*;

use crate::{
    attention::mla::MlaSpec,
    kernel::cpu::{CpuTensor, blas, matmul::dot},
};
use wide::f32x8;

const SIMD_LANES: usize = 8;

/// MLA 前向（CPU）。
///
/// CPU backend 与显式 CPU fallback 共用此实现。
pub fn mla_attention_cpu(mla: &MlaSpec, q: &CpuTensor, kv: &CpuTensor, k_rope: &CpuTensor) -> CpuTensor {
    let n = q.rows;
    let q_head_dim = mla.q_head_dim();
    let qk_nope_dim = mla.qk_nope_dim();
    let kv_head_dim = mla.kv_head_dim();
    let value_dim = mla.value_dim();
    let num_heads = mla.num_heads;
    let q_proj = mla.q_projection_size;
    let kv_proj = mla.kv_projection_size;
    let rope_dim = mla.qk_rope_head_dim;

    // 小 batch 时 BLAS 收益有限,回落标量实现;selection 路径(backend DSA 选中
    // token 的稀疏 softmax)无标量门槛,由 backend 直接调用。
    if n >= 64
        && let Some(out) = mla_attention_blas(mla, q, kv, k_rope, None)
    {
        return out;
    }

    let mut out = CpuTensor { data: vec![0.0; n * q_proj], rows: n, cols: q_proj };
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    out.data.par_chunks_mut(q_proj).enumerate().for_each(|(i, out_row)| {
        let mut scores = vec![0.0_f32; i + 1];
        let full = value_dim / SIMD_LANES * SIMD_LANES;
        let mut accumulators = vec![f32x8::splat(0.0); full / SIMD_LANES];
        for h in 0..num_heads {
            accumulators.fill(f32x8::splat(0.0));
            let q_base = i * q_proj + h * q_head_dim;
            let q_nope = &q.data[q_base..q_base + qk_nope_dim];
            let q_rope = &q.data[q_base + qk_nope_dim..q_base + q_head_dim];

            let mut max_score = f32::NEG_INFINITY;
            for tok in 0..=i {
                let kv_base = tok * kv_proj + h * kv_head_dim;
                let kv_nope = &kv.data[kv_base..kv_base + qk_nope_dim];
                let krope = &k_rope.data[tok * rope_dim..(tok + 1) * rope_dim];
                let score = (dot(q_nope, kv_nope) + dot(q_rope, krope)) * scale;
                scores[tok] = score;
                max_score = max_score.max(score);
            }

            let mut sum = 0.0;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for score in &mut scores {
                *score /= sum;
            }

            let out_base = h * q_head_dim;
            for tok in 0..=i {
                let value = &kv.data[tok * kv_proj + h * kv_head_dim + qk_nope_dim..];
                let weight = f32x8::splat(scores[tok]);
                for (chunk, accumulator) in accumulators.iter_mut().enumerate() {
                    let start = chunk * SIMD_LANES;
                    let value = f32x8::from(<[f32; SIMD_LANES]>::try_from(&value[start..start + SIMD_LANES]).unwrap());
                    *accumulator = weight.mul_add(value, *accumulator);
                }
            }
            for (chunk, accumulator) in accumulators.iter().enumerate() {
                let values: [f32; SIMD_LANES] = (*accumulator).into();
                let start = out_base + chunk * SIMD_LANES;
                out_row[start..start + SIMD_LANES].copy_from_slice(&values);
            }
            for j in full..value_dim {
                let mut accumulator = 0.0;
                for tok in 0..=i {
                    let v_base = tok * kv_proj + h * kv_head_dim + qk_nope_dim;
                    accumulator += scores[tok] * kv.data[v_base + j];
                }
                out_row[out_base + j] = accumulator;
            }
        }
    });
    out
}

/// MLA BLAS 注意力(分 head 并行 + QUERY_TILE 分块 GEMM)。
///
/// `selection = Some((选中 token 表, top_k))` 时按行做选中 token 的稀疏 softmax
/// (DSA prefill 语义,选中表按 `[row * top_k, (row + 1) * top_k)` 分段);
/// `None` 时做常规因果 softmax。两种路径的输出布局分别与其标量实现保持一致:
/// selection 路径紧凑排布 `[rows, heads * value_dim]`,因果路径按
/// `q_projection_size` 步长排布。
pub fn mla_attention_blas(mla: &MlaSpec, q: &CpuTensor, kv: &CpuTensor, k_rope: &CpuTensor, selection: Option<(&[usize], usize)>) -> Option<CpuTensor> {
    const QUERY_TILE: usize = 256;
    let n = q.rows;
    if (n < 64 && selection.is_none()) || !blas::available() {
        return None;
    }
    // selection 表长度不足时无法安全分段,回落调用方的标量路径。
    if selection.is_some_and(|(rows, top_k)| rows.len() < n * top_k) {
        return None;
    }
    let q_head_dim = mla.q_head_dim();
    let qk_nope_dim = mla.qk_nope_dim();
    let kv_head_dim = mla.kv_head_dim();
    let value_dim = mla.value_dim();
    let q_proj = mla.q_projection_size;
    let kv_proj = mla.kv_projection_size;
    let rope_dim = mla.qk_rope_head_dim;
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    // 输出中每个 head 的起始步长:因果路径沿用 q_proj 布局,selection 路径紧凑 value 布局。
    let head_stride = if selection.is_some() { value_dim } else { q_head_dim };

    let heads: Vec<Option<Vec<f32>>> = (0..mla.num_heads)
        .into_par_iter()
        .map(|head| {
            let mut query = Vec::with_capacity(n * q_head_dim);
            let mut key = Vec::with_capacity(n * q_head_dim);
            let mut value = Vec::with_capacity(n * value_dim);
            for token in 0..n {
                let q_base = token * q_proj + head * q_head_dim;
                query.extend_from_slice(&q.data[q_base..q_base + q_head_dim]);
                let kv_base = token * kv_proj + head * kv_head_dim;
                key.extend_from_slice(&kv.data[kv_base..kv_base + qk_nope_dim]);
                key.extend_from_slice(&k_rope.data[token * rope_dim..(token + 1) * rope_dim]);
                value.extend_from_slice(&kv.data[kv_base + qk_nope_dim..kv_base + qk_nope_dim + value_dim]);
            }

            let mut output = vec![0.0_f32; n * value_dim];
            let mut scores = vec![0.0_f32; QUERY_TILE * n];
            let mut tile_output = vec![0.0_f32; QUERY_TILE * value_dim];
            for start in (0..n).step_by(QUERY_TILE) {
                let rows = QUERY_TILE.min(n - start);
                let keys = start + rows;
                let score_slice = &mut scores[..rows * keys];
                if !blas::sgemm_nt(rows, keys, q_head_dim, scale, &query[start * q_head_dim..], &key[..keys * q_head_dim], score_slice) {
                    return None;
                }
                for row in 0..rows {
                    let scores = &mut score_slice[row * keys..(row + 1) * keys];
                    match selection {
                        Some((selection, top_k)) => {
                            // 只对选中 token 做 softmax,其余位置置零。
                            let global_row = start + row;
                            let count = top_k.min(global_row + 1);
                            let selected = &selection[global_row * top_k..global_row * top_k + count];
                            let mut selected_scores = Vec::with_capacity(count);
                            let mut max_score = f32::NEG_INFINITY;
                            for &token in selected {
                                let score = scores[token];
                                max_score = max_score.max(score);
                                selected_scores.push(score);
                            }
                            scores.fill(0.0);
                            let mut sum = 0.0;
                            for score in &mut selected_scores {
                                *score = (*score - max_score).exp();
                                sum += *score;
                            }
                            for (slot, &token) in selected.iter().enumerate() {
                                scores[token] = selected_scores[slot] / sum;
                            }
                        }
                        None => {
                            // 因果 softmax:仅前 valid 个 key 有效。
                            let valid = start + row + 1;
                            let max_score = scores[..valid].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let mut sum = 0.0;
                            for score in &mut scores[..valid] {
                                *score = (*score - max_score).exp();
                                sum += *score;
                            }
                            for score in &mut scores[..valid] {
                                *score /= sum;
                            }
                            scores[valid..].fill(0.0);
                        }
                    }
                }
                let tile = &mut tile_output[..rows * value_dim];
                if !blas::sgemm_nn(rows, value_dim, keys, score_slice, &value[..keys * value_dim], tile) {
                    return None;
                }
                output[start * value_dim..(start + rows) * value_dim].copy_from_slice(tile);
            }
            Some(output)
        })
        .collect();
    let heads: Vec<Vec<f32>> = heads.into_iter().collect::<Option<Vec<_>>>()?;
    let out_cols = mla.num_heads * head_stride;
    let mut out = CpuTensor { data: vec![0.0; n * out_cols], rows: n, cols: out_cols };
    for (head, values) in heads.iter().enumerate() {
        for token in 0..n {
            let out_base = token * out_cols + head * head_stride;
            out.data[out_base..out_base + value_dim].copy_from_slice(&values[token * value_dim..(token + 1) * value_dim]);
        }
    }
    Some(out)
}

#[cfg(test)]
mod blas_tests {
    use super::*;

    #[test]
    fn causal_attention_uniform_scores() {
        let rows = 320;
        let spec = MlaSpec { q_lora_rank: 1, kv_lora_rank: 1, qk_rope_head_dim: 2, q_projection_size: 4, kv_projection_size: 4, num_heads: 1, rope_theta: 10_000.0, rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf };
        let query = CpuTensor { data: vec![0.0; rows * 4], rows, cols: 4 };
        let mut kv = CpuTensor { data: vec![0.0; rows * 4], rows, cols: 4 };
        for token in 0..rows {
            kv.data[token * 4 + 2] = token as f32;
            kv.data[token * 4 + 3] = 2.0 * token as f32;
        }
        let rope = CpuTensor { data: vec![0.0; rows * 2], rows, cols: 2 };
        let output = mla_attention_cpu(&spec, &query, &kv, &rope);
        for token in 0..rows {
            let mean = token as f32 * 0.5;
            assert!((output.data[token * 4] - mean).abs() < 1e-3);
            assert!((output.data[token * 4 + 1] - 2.0 * mean).abs() < 1e-3);
        }
    }
}
