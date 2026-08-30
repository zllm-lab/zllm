//! 视觉 ViT 算子:RoPE 与双向多头自注意力。
//!
//! 语义与 Metal `vision_rope_f16`/`vision_attention_f16`
//! (`kernel/metal/kernels.rs:6980`/`:7014`)逐位对齐——CPU 是跨后端 oracle。

use std::cell::RefCell;

use rayon::prelude::*;
use wide::f32x8;

use super::matmul::dot;

const SIMD_LANES: usize = 8;

/// 视觉 RoPE(GPT-NeoX rotate-half)。
///
/// `query`/`key` 行优先 `[rows, head_count*head_dim]`;`cos`/`sin` 行优先
/// `[rows, rotary_dim]`,跨 head 共享、按 within-head 下标 `d` 索引。
/// `d < rotary_dim/2` 配 `d + half`,`d >= half` 配 `d - half`;前半取负。
#[allow(clippy::too_many_arguments)]
pub fn vision_rope(query: &[f32], key: &[f32], cos: &[f32], sin: &[f32], head_count: usize, head_dim: usize, rotary_dim: usize, out_query: &mut [f32], out_key: &mut [f32]) {
    let row_stride = head_count * head_dim;
    if !query.len().is_multiple_of(row_stride) || !key.len().is_multiple_of(row_stride) {
        return;
    }
    let rows = query.len() / row_stride;
    let half = rotary_dim / 2;
    for row in 0..rows {
        let cos_row = &cos[row * rotary_dim..(row + 1) * rotary_dim];
        let sin_row = &sin[row * rotary_dim..(row + 1) * rotary_dim];
        for head in 0..head_count {
            let base = row * row_stride + head * head_dim;
            // rotary_dim 之后直接透传(本模型 rotary_dim == head_dim,该分支为空)。
            out_query[base + rotary_dim..base + head_dim].copy_from_slice(&query[base + rotary_dim..base + head_dim]);
            out_key[base + rotary_dim..base + head_dim].copy_from_slice(&key[base + rotary_dim..base + head_dim]);
            for d in 0..half {
                let paired = d + half;
                let (c, s) = (cos_row[d], sin_row[d]);
                out_query[base + d] = query[base + d].mul_add(c, -query[base + paired] * s);
                out_key[base + d] = key[base + d].mul_add(c, -key[base + paired] * s);
            }
            for d in half..rotary_dim {
                let paired = d - half;
                let (c, s) = (cos_row[d], sin_row[d]);
                out_query[base + d] = query[base + d].mul_add(c, query[base + paired] * s);
                out_key[base + d] = key[base + d].mul_add(c, key[base + paired] * s);
            }
        }
    }
}

thread_local! {
    /// 每 query 行复用的 score 缓冲,避免 rayon 任务内重复分配。
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// 双向多头自注意力(无 mask、无 KV cache)。
///
/// `query`/`key`/`value`/`out` 均为 `[rows, head_count*head_dim]`,head 布局
/// `[row, head, head_dim]`。每个 query 行对所有 key 全量打分、softmax、加权求和。
///
/// 按 query 行分块(`QUERY_BLOCK`):一个块内的多条 query 共享同一批 K/V——每个 K[k]/V[k]
/// 只读取一次,复用于整块 query。这把 K/V 的读取次数从 `rows` 降到 `rows/QUERY_BLOCK`,
/// 缓解大图(N 大)下 O(N²) 的内存带宽压力。rayon 并行 over query 块;score 内积复用
/// `matmul::dot`,value 累加走 f32x8。
pub fn vision_attention(query: &[f32], key: &[f32], value: &[f32], cols: usize, head_count: usize, out: &mut [f32]) {
    let head_dim = cols.checked_div(head_count).unwrap_or(0);
    vision_attention_scaled(query, key, value, cols, head_count, 1.0 / (head_dim as f32).sqrt(), out);
}

pub fn vision_attention_scaled(query: &[f32], key: &[f32], value: &[f32], cols: usize, head_count: usize, scale: f32, out: &mut [f32]) {
    if !query.len().is_multiple_of(cols) || !key.len().is_multiple_of(cols) || !value.len().is_multiple_of(cols) || out.len() != query.len() || head_count == 0 || !cols.is_multiple_of(head_count) {
        return;
    }
    let rows = query.len() / cols;
    let head_dim = cols / head_count;
    const QUERY_BLOCK: usize = 16;

    out.par_chunks_mut(cols * QUERY_BLOCK).enumerate().for_each(|(block_index, out_block)| {
        let block_rows = out_block.len() / cols;
        let block_start = block_index * QUERY_BLOCK;
        out_block.fill(0.0);

        // 紧凑 query 缓冲 [block_rows, head_dim]:把 strided 的 query 行聚到一起,提升点乘局部性。
        let mut q_buf = vec![0.0f32; block_rows * head_dim];
        // 每 query 行的 softmax 分母(exp 未归一,累加 value 时再除)。
        let mut sums = vec![0.0f32; block_rows];

        SCORES.with(|cell| {
            let mut scores = cell.borrow_mut();
            scores.resize(block_rows * rows, 0.0); // [block_rows, rows],query-major
            let scores = &mut scores[..];

            for head in 0..head_count {
                let head_off = head * head_dim;
                for (i, q_row) in (block_start..block_start + block_rows).enumerate() {
                    let base = q_row * cols + head_off;
                    q_buf[i * head_dim..(i + 1) * head_dim].copy_from_slice(&query[base..base + head_dim]);
                }
                // scores[i][k] = dot(q[i], K[k,head]) * scale。每个 K[k] 只读一次,复用于整块 query。
                for k in 0..rows {
                    let k_base = k * cols + head_off;
                    let k_vec = &key[k_base..k_base + head_dim];
                    for i in 0..block_rows {
                        scores[i * rows + k] = dot(&q_buf[i * head_dim..(i + 1) * head_dim], k_vec) * scale;
                    }
                }
                // 逐 query 行 softmax(max + exp + sum)。
                for i in 0..block_rows {
                    let row = &mut scores[i * rows..(i + 1) * rows];
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for score in row.iter_mut() {
                        let exp = (*score - max).exp();
                        *score = exp;
                        sum += exp;
                    }
                    sums[i] = sum;
                }
                // 加权求 value:每个 V[k] 只读一次,复用于整块 query 的累加(f32x8 saxpy)。
                for k in 0..rows {
                    let v_base = k * cols + head_off;
                    let v_vec = &value[v_base..v_base + head_dim];
                    for i in 0..block_rows {
                        let weight = scores[i * rows + k] / sums[i];
                        let out_head = &mut out_block[i * cols + head_off..i * cols + head_off + head_dim];
                        let wv = f32x8::splat(weight);
                        let mut vi = 0;
                        while vi + SIMD_LANES <= head_dim {
                            let v = f32x8::from(<[f32; SIMD_LANES]>::try_from(&v_vec[vi..vi + SIMD_LANES]).unwrap());
                            let o = f32x8::from(<[f32; SIMD_LANES]>::try_from(&out_head[vi..vi + SIMD_LANES]).unwrap());
                            let merged: [f32; SIMD_LANES] = (o + wv * v).into();
                            out_head[vi..vi + SIMD_LANES].copy_from_slice(&merged);
                            vi += SIMD_LANES;
                        }
                        while vi < head_dim {
                            out_head[vi] += weight * v_vec[vi];
                            vi += 1;
                        }
                    }
                }
            }
        });
    });
}

/// Gemma4V 二维 RoPE：每个 head 的前/后半分别对应 x/y 轴，各轴内部使用 NeoX rotate-half。
#[allow(clippy::too_many_arguments)]
pub fn vision_rope_2d(query: &[f32], key: &[f32], cos: &[f32], sin: &[f32], head_count: usize, head_dim: usize, out_query: &mut [f32], out_key: &mut [f32]) {
    let rows = query.len() / (head_count * head_dim);
    let axis_dim = head_dim / 2;
    let axis_half = axis_dim / 2;
    for row in 0..rows {
        for head in 0..head_count {
            let base = (row * head_count + head) * head_dim;
            for dimension in 0..head_dim {
                let axis = dimension / axis_dim;
                let within = dimension % axis_dim;
                let paired_within = if within < axis_half { within + axis_half } else { within - axis_half };
                let paired = axis * axis_dim + paired_within;
                let sign = if within < axis_half { -1.0 } else { 1.0 };
                let rope = row * head_dim + dimension;
                out_query[base + dimension] = query[base + dimension] * cos[rope] + sign * query[base + paired] * sin[rope];
                out_key[base + dimension] = key[base + dimension] * cos[rope] + sign * key[base + paired] * sin[rope];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_rope(query: &[f32], key: &[f32], cos: &[f32], sin: &[f32], rows: usize, head_count: usize, head_dim: usize, rotary_dim: usize, out_query: &mut [f32], out_key: &mut [f32]) {
        let half = rotary_dim / 2;
        let row_stride = head_count * head_dim;
        for row in 0..rows {
            for head in 0..head_count {
                let base = row * row_stride + head * head_dim;
                for d in 0..head_dim {
                    let index = base + d;
                    if d >= rotary_dim {
                        out_query[index] = query[index];
                        out_key[index] = key[index];
                        continue;
                    }
                    let paired = if d < half { d + half } else { d - half };
                    let sign = if d < half { -1.0 } else { 1.0 };
                    let c = cos[row * rotary_dim + d];
                    let s = sin[row * rotary_dim + d];
                    out_query[index] = query[index] * c + sign * query[base + paired] * s;
                    out_key[index] = key[index] * c + sign * key[base + paired] * s;
                }
            }
        }
    }

    #[test]
    fn vision_rope匹配朴素实现() {
        // head_dim=8(含尾部透传)、rotary_dim=6(奇数 half=3,覆盖两半与配对)。
        let (rows, heads, head_dim, rotary_dim) = (3, 2, 8, 6);
        let cols = heads * head_dim;
        let query: Vec<f32> = (0..rows * cols).map(|i| (i as f32 % 5.0) - 2.0).collect();
        let key = query.iter().map(|v| v + 0.5).collect::<Vec<_>>();
        let cos: Vec<f32> = (0..rows * rotary_dim).map(|i| (i as f32 * 0.1).cos()).collect();
        let sin: Vec<f32> = (0..rows * rotary_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut out = vec![0.0; rows * cols];
        let mut out_key = vec![0.0; rows * cols];
        let mut ref_out = vec![0.0; rows * cols];
        let mut ref_key = vec![0.0; rows * cols];
        vision_rope(&query, &key, &cos, &sin, heads, head_dim, rotary_dim, &mut out, &mut out_key);
        naive_rope(&query, &key, &cos, &sin, rows, heads, head_dim, rotary_dim, &mut ref_out, &mut ref_key);
        for index in 0..rows * cols {
            assert!((out[index] - ref_out[index]).abs() < 1.0e-5, "query {index}: {} vs {}", out[index], ref_out[index]);
            assert!((out_key[index] - ref_key[index]).abs() < 1.0e-5, "key {index}: {} vs {}", out_key[index], ref_key[index]);
        }
    }

    fn naive_attention(query: &[f32], key: &[f32], value: &[f32], rows: usize, cols: usize, head_count: usize, out: &mut [f32]) {
        let head_dim = cols / head_count;
        let scale = 1.0 / (head_dim as f32).sqrt();
        for q in 0..rows {
            for head in 0..head_count {
                let q_off = q * cols + head * head_dim;
                let scores: Vec<f32> = (0..rows)
                    .map(|k| {
                        let k_off = k * cols + head * head_dim;
                        query[q_off..q_off + head_dim].iter().zip(&key[k_off..k_off + head_dim]).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = weights.iter().sum();
                for d in 0..head_dim {
                    out[q_off + d] = (0..rows).map(|k| weights[k] / sum * value[k * cols + head * head_dim + d]).sum::<f32>();
                }
            }
        }
    }

    #[test]
    fn vision_attention匹配朴素双向softmax() {
        // head_dim=8(整除 SIMD_LANES),2 head,6 行。
        let (rows, heads) = (6, 2);
        let head_dim = 8;
        let cols = heads * head_dim;
        let query: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.37) % 3.0 - 1.5).collect();
        let key = (0..rows * cols).map(|i| (i as f32 * 0.21) % 3.0 - 1.5).collect::<Vec<_>>();
        let value = (0..rows * cols).map(|i| (i as f32 * 0.13) % 3.0 - 1.5).collect::<Vec<_>>();
        let mut out = vec![0.0; rows * cols];
        let mut reference = vec![0.0; rows * cols];
        vision_attention(&query, &key, &value, cols, heads, &mut out);
        naive_attention(&query, &key, &value, rows, cols, heads, &mut reference);
        for index in 0..rows * cols {
            let mag = reference[index].abs();
            assert!((out[index] - reference[index]).abs() < 1.0e-5 * (1.0 + mag), "{}: {} vs {}", index, out[index], reference[index]);
        }
    }
}
