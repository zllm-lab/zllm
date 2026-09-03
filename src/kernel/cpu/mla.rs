//! MLA 的 CPU 实现。

use half::bf16 as HalfBf16;
use rayon::prelude::*;

use crate::{
    attention::mla::MlaSpec,
    kernel::cpu::{CpuTensor, blas, matmul::dot},
};
use wide::f32x8;

const SIMD_LANES: usize = 8;

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

#[cfg(target_arch = "x86_64")]
unsafe impl<T> Send for SendPtr<T> {}

#[cfg(target_arch = "x86_64")]
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(target_arch = "x86_64")]
impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// GLM-5.2 dense prefill 的 BF16 直算路径。K-nope/V 保持 GPU kv_b 投影的原始
/// 布局，RoPE 仍只保存一份；每个 8-row×head 任务一次扫描同时服务 8 个 query，
/// 避免为 CPU 另建 256MiB F32 K/V 布局。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512bf16,fma")]
unsafe fn mla_attention_dense_bf16_avx512_impl(mla: &MlaSpec, query: &[u16], query_rows: usize, query_start: usize, projected_kv: &[u16], rope: &[u16], history_rows: usize) -> Vec<u16> {
    use std::arch::x86_64::*;

    debug_assert_eq!(query_start + query_rows, history_rows);
    const ROW_CHUNK: usize = 8;
    let q_head_dim = mla.q_head_dim();
    let qk_nope_dim = mla.qk_nope_dim();
    let kv_head_dim = mla.kv_head_dim();
    let value_dim = mla.value_dim();
    let output_cols = mla.num_heads * value_dim;
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    let chunks = query_rows.div_ceil(ROW_CHUNK);
    let tasks = chunks * mla.num_heads;
    let mut output = vec![0_u16; query_rows * output_cols];
    let output_ptr = SendPtr(output.as_mut_ptr());
    (0..tasks).into_par_iter().for_each(|task| unsafe {
        let chunk = task / mla.num_heads;
        let head = task % mla.num_heads;
        let row_base = chunk * ROW_CHUNK;
        let row_count = ROW_CHUNK.min(query_rows - row_base);
        let span = query_start + row_base + row_count;
        let mut scores = vec![0.0_f32; row_count * span];
        let mut maximum = [f32::NEG_INFINITY; ROW_CHUNK];
        for token in 0..span {
            let key_base = token * mla.kv_projection_size + head * kv_head_dim;
            let rope_base = token * mla.qk_rope_head_dim;
            let mut sums = [_mm512_setzero_ps(); ROW_CHUNK];
            for column in (0..qk_nope_dim).step_by(32) {
                let key = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(projected_kv.as_ptr().add(key_base + column).cast()));
                for row in 0..row_count {
                    if token >= query_start + row_base + row + 1 {
                        continue;
                    }
                    let query_base = (row_base + row) * mla.q_projection_size + head * q_head_dim;
                    let q = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(query.as_ptr().add(query_base + column).cast()));
                    sums[row] = _mm512_dpbf16_ps(sums[row], q, key);
                }
            }
            for column in (0..mla.qk_rope_head_dim).step_by(32) {
                let key = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(rope.as_ptr().add(rope_base + column).cast()));
                for row in 0..row_count {
                    if token >= query_start + row_base + row + 1 {
                        continue;
                    }
                    let query_base = (row_base + row) * mla.q_projection_size + head * q_head_dim + qk_nope_dim;
                    let q = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(query.as_ptr().add(query_base + column).cast()));
                    sums[row] = _mm512_dpbf16_ps(sums[row], q, key);
                }
            }
            for row in 0..row_count {
                if token >= query_start + row_base + row + 1 {
                    continue;
                }
                let score = _mm512_reduce_add_ps(sums[row]) * scale;
                scores[row * span + token] = score;
                maximum[row] = maximum[row].max(score);
            }
        }
        let mut inverse = [0.0_f32; ROW_CHUNK];
        for row in 0..row_count {
            let valid = query_start + row_base + row + 1;
            let row_scores = &mut scores[row * span..row * span + valid];
            let mut denominator = 0.0_f32;
            for score in row_scores {
                *score = (*score - maximum[row]).exp();
                denominator += *score;
            }
            inverse[row] = denominator.recip();
        }
        for column in (0..value_dim).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); ROW_CHUNK];
            for token in 0..span {
                let value_base = token * mla.kv_projection_size + head * kv_head_dim + qk_nope_dim + column;
                let bits = _mm256_loadu_si256(projected_kv.as_ptr().add(value_base).cast());
                let bits = _mm512_cvtepu16_epi32(bits);
                let value = _mm512_castsi512_ps(_mm512_slli_epi32(bits, 16));
                for row in 0..row_count {
                    let valid = query_start + row_base + row + 1;
                    if token >= valid {
                        continue;
                    }
                    let probability = _mm512_set1_ps(scores[row * span + token] * inverse[row]);
                    accumulators[row] = _mm512_fmadd_ps(probability, value, accumulators[row]);
                }
            }
            for row in 0..row_count {
                let mut values = [0.0_f32; 16];
                _mm512_storeu_ps(values.as_mut_ptr(), accumulators[row]);
                let destination = output_ptr.get().add((row_base + row) * output_cols + head * value_dim + column);
                for lane in 0..16 {
                    *destination.add(lane) = HalfBf16::from_f32(values[lane]).to_bits();
                }
            }
        }
    });
    output
}

/// DSA 选中历史后的 BF16 直算路径。一个任务负责一行的全部 head，让每个选中
/// token 的整行 KV 连续流过 CPU；否则按 row×head 拆任务会把同一份随机历史
/// 分散扫描 64 次，长上下文时主要消耗在内存访问而不是点积。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512bf16,fma")]
unsafe fn mla_attention_selected_bf16_avx512_impl(mla: &MlaSpec, query: &[u16], query_rows: usize, _query_start: usize, projected_kv: &[u16], rope: &[u16], selection: &[u32], width: usize) -> Vec<u16> {
    use std::arch::x86_64::*;

    const ROW_CHUNK: usize = 32;
    const HEAD_CHUNK: usize = 4;
    let q_head_dim = mla.q_head_dim();
    let qk_nope_dim = mla.qk_nope_dim();
    let kv_head_dim = mla.kv_head_dim();
    let value_dim = mla.value_dim();
    let output_cols = mla.num_heads * value_dim;
    let value_vectors = value_dim / 16;
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    let mut output = vec![0_u16; query_rows * output_cols];
    let output_ptr = SendPtr(output.as_mut_ptr());
    let row_chunks = query_rows.div_ceil(ROW_CHUNK);
    let head_chunks = mla.num_heads.div_ceil(HEAD_CHUNK);
    let entries = (0..row_chunks)
        .map(|chunk| {
            let row_base = chunk * ROW_CHUNK;
            let row_count = ROW_CHUNK.min(query_rows - row_base);
            // 高 32 位按 token 排序；低位保存 tile 内 row 与原 selection slot。
            // 相邻 query 的同一个 token 因而只读取一次完整 projected KV 行。
            let mut entries = Vec::with_capacity(row_count * width);
            for row in 0..row_count {
                for (slot, &token) in selection[(row_base + row) * width..(row_base + row + 1) * width].iter().enumerate() {
                    entries.push((u64::from(token) << 32) | ((row as u64) << 16) | slot as u64);
                }
            }
            entries.sort_unstable();
            entries
        })
        .collect::<Vec<_>>();

    (0..row_chunks * head_chunks).into_par_iter().for_each(|task| unsafe {
        let chunk = task / head_chunks;
        let head_base = task % head_chunks * HEAD_CHUNK;
        let head_count = HEAD_CHUNK.min(mla.num_heads - head_base);
        let row_base = chunk * ROW_CHUNK;
        let row_count = ROW_CHUNK.min(query_rows - row_base);
        let entries = &entries[chunk];
        let mut scores = vec![0.0_f32; row_count * head_count * width];
        let mut maxima = vec![f32::NEG_INFINITY; row_count * head_count];

        let mut group = 0;
        while group < entries.len() {
            let token = (entries[group] >> 32) as usize;
            let mut group_end = group + 1;
            while group_end < entries.len() && entries[group_end] >> 32 == entries[group] >> 32 {
                group_end += 1;
            }
            let rows = &entries[group..group_end];
            let projected_row = &projected_kv[token * mla.kv_projection_size..(token + 1) * mla.kv_projection_size];
            let rope_row = &rope[token * mla.qk_rope_head_dim..(token + 1) * mla.qk_rope_head_dim];
            for local_head in 0..head_count {
                let head = head_base + local_head;
                let kv_head = &projected_row[head * kv_head_dim..(head + 1) * kv_head_dim];
                let mut sums = [_mm512_setzero_ps(); ROW_CHUNK];
                for column in (0..qk_nope_dim).step_by(32) {
                    let k = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(kv_head.as_ptr().add(column).cast()));
                    for &entry in rows {
                        let row = ((entry >> 16) & 0xffff) as usize;
                        let query_head = query.as_ptr().add((row_base + row) * mla.q_projection_size + head * q_head_dim);
                        let q = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(query_head.add(column).cast()));
                        sums[row] = _mm512_dpbf16_ps(sums[row], q, k);
                    }
                }
                for column in (0..mla.qk_rope_head_dim).step_by(32) {
                    let k = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(rope_row.as_ptr().add(column).cast()));
                    for &entry in rows {
                        let row = ((entry >> 16) & 0xffff) as usize;
                        let query_head = query.as_ptr().add((row_base + row) * mla.q_projection_size + head * q_head_dim + qk_nope_dim);
                        let q = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(query_head.add(column).cast()));
                        sums[row] = _mm512_dpbf16_ps(sums[row], q, k);
                    }
                }
                for &entry in rows {
                    let row = ((entry >> 16) & 0xffff) as usize;
                    let slot = (entry & 0xffff) as usize;
                    let score = _mm512_reduce_add_ps(sums[row]) * scale;
                    scores[(row * head_count + local_head) * width + slot] = score;
                    let maximum = &mut maxima[row * head_count + local_head];
                    *maximum = maximum.max(score);
                }
            }
            group = group_end;
        }

        for row in 0..row_count {
            for local_head in 0..head_count {
                let index = row * head_count + local_head;
                let head_scores = &mut scores[index * width..(index + 1) * width];
                let mut denominator = 0.0_f32;
                for score in head_scores.iter_mut() {
                    *score = (*score - maxima[index]).exp();
                    denominator += *score;
                }
                let inverse = denominator.recip();
                for score in head_scores {
                    *score *= inverse;
                }
            }
        }

        let mut accumulators = vec![_mm512_setzero_ps(); row_count * head_count * value_vectors];
        group = 0;
        while group < entries.len() {
            let token = (entries[group] >> 32) as usize;
            let mut group_end = group + 1;
            while group_end < entries.len() && entries[group_end] >> 32 == entries[group] >> 32 {
                group_end += 1;
            }
            let rows = &entries[group..group_end];
            let projected_row = &projected_kv[token * mla.kv_projection_size..(token + 1) * mla.kv_projection_size];
            for local_head in 0..head_count {
                let head = head_base + local_head;
                let value = projected_row.as_ptr().add(head * kv_head_dim + qk_nope_dim);
                for vector in 0..value_vectors {
                    let bits = _mm256_loadu_si256(value.add(vector * 16).cast());
                    let bits = _mm512_cvtepu16_epi32(bits);
                    let value = _mm512_castsi512_ps(_mm512_slli_epi32(bits, 16));
                    for &entry in rows {
                        let row = ((entry >> 16) & 0xffff) as usize;
                        let slot = (entry & 0xffff) as usize;
                        let probability = _mm512_set1_ps(scores[(row * head_count + local_head) * width + slot]);
                        let accumulator = &mut accumulators[(row * head_count + local_head) * value_vectors + vector];
                        *accumulator = _mm512_fmadd_ps(probability, value, *accumulator);
                    }
                }
            }
            group = group_end;
        }
        for row in 0..row_count {
            for local_head in 0..head_count {
                let head = head_base + local_head;
                for vector in 0..value_vectors {
                    let mut values = [0.0_f32; 16];
                    _mm512_storeu_ps(values.as_mut_ptr(), accumulators[(row * head_count + local_head) * value_vectors + vector]);
                    let destination = output_ptr.get().add((row_base + row) * output_cols + head * value_dim + vector * 16);
                    for lane in 0..16 {
                        *destination.add(lane) = HalfBf16::from_f32(values[lane]).to_bits();
                    }
                }
            }
        }
    });
    output
}

pub fn mla_attention_dense_bf16(mla: &MlaSpec, query: &[u16], query_rows: usize, query_start: usize, projected_kv: &[u16], rope: &[u16], history_rows: usize) -> Option<Vec<u16>> {
    if query_rows == 0
        || query_start.checked_add(query_rows) != Some(history_rows)
        || query.len() != query_rows.checked_mul(mla.q_projection_size)?
        || projected_kv.len() != history_rows.checked_mul(mla.kv_projection_size)?
        || rope.len() != history_rows.checked_mul(mla.qk_rope_head_dim)?
        || mla.q_head_dim() != mla.value_dim()
        || !mla.qk_nope_dim().is_multiple_of(32)
        || !mla.qk_rope_head_dim.is_multiple_of(32)
        || !mla.value_dim().is_multiple_of(16)
    {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw") && std::arch::is_x86_feature_detected!("avx512bf16") && std::arch::is_x86_feature_detected!("fma") {
        return Some(unsafe { mla_attention_dense_bf16_avx512_impl(mla, query, query_rows, query_start, projected_kv, rope, history_rows) });
    }
    None
}

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

/// 已有 CPU history 上的追加 chunk 因果 MLA。与首段 prefill 不同，query 第 0 行
/// 的绝对位置是 `query_start`，GEMM 的 K/V 维度因此覆盖 prefix + 当前 tile。
pub fn mla_attention_blas_chunk(mla: &MlaSpec, q: &CpuTensor, kv: &CpuTensor, k_rope: &CpuTensor, query_start: usize) -> Option<CpuTensor> {
    const QUERY_TILE: usize = 128;
    let rows = q.rows;
    let history_rows = kv.rows;
    if rows == 0 || history_rows != query_start.checked_add(rows)? || k_rope.rows != history_rows || !blas::available() {
        return None;
    }
    let q_head_dim = mla.q_head_dim();
    let qk_nope_dim = mla.qk_nope_dim();
    let kv_head_dim = mla.kv_head_dim();
    let value_dim = mla.value_dim();
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    let heads = (0..mla.num_heads)
        .into_par_iter()
        .map(|head| {
            let mut query = Vec::with_capacity(rows * q_head_dim);
            for token in 0..rows {
                let base = token * mla.q_projection_size + head * q_head_dim;
                query.extend_from_slice(&q.data[base..base + q_head_dim]);
            }
            let mut key = Vec::with_capacity(history_rows * q_head_dim);
            let mut value = Vec::with_capacity(history_rows * value_dim);
            for token in 0..history_rows {
                let base = token * mla.kv_projection_size + head * kv_head_dim;
                key.extend_from_slice(&kv.data[base..base + qk_nope_dim]);
                key.extend_from_slice(&k_rope.data[token * mla.qk_rope_head_dim..(token + 1) * mla.qk_rope_head_dim]);
                value.extend_from_slice(&kv.data[base + qk_nope_dim..base + qk_nope_dim + value_dim]);
            }
            let mut output = vec![0.0_f32; rows * value_dim];
            let mut scores = vec![0.0_f32; QUERY_TILE * history_rows];
            let mut tile_output = vec![0.0_f32; QUERY_TILE * value_dim];
            for start in (0..rows).step_by(QUERY_TILE) {
                let tile_rows = QUERY_TILE.min(rows - start);
                let keys = query_start + start + tile_rows;
                let score_slice = &mut scores[..tile_rows * keys];
                if !blas::sgemm_nt(tile_rows, keys, q_head_dim, scale, &query[start * q_head_dim..], &key[..keys * q_head_dim], score_slice) {
                    return None;
                }
                for row in 0..tile_rows {
                    let valid = query_start + start + row + 1;
                    let row_scores = &mut score_slice[row * keys..(row + 1) * keys];
                    let maximum = row_scores[..valid].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut denominator = 0.0_f32;
                    for score in &mut row_scores[..valid] {
                        *score = (*score - maximum).exp();
                        denominator += *score;
                    }
                    for score in &mut row_scores[..valid] {
                        *score /= denominator;
                    }
                    row_scores[valid..].fill(0.0);
                }
                let tile = &mut tile_output[..tile_rows * value_dim];
                if !blas::sgemm_nn(tile_rows, value_dim, keys, score_slice, &value[..keys * value_dim], tile) {
                    return None;
                }
                output[start * value_dim..(start + tile_rows) * value_dim].copy_from_slice(tile);
            }
            Some(output)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Option<Vec<_>>>()?;
    let output_cols = mla.num_heads * value_dim;
    let mut output = CpuTensor { data: vec![0.0; rows * output_cols], rows, cols: output_cols };
    for (head, values) in heads.iter().enumerate() {
        for row in 0..rows {
            let destination = row * output_cols + head * value_dim;
            output.data[destination..destination + value_dim].copy_from_slice(&values[row * value_dim..(row + 1) * value_dim]);
        }
    }
    Some(output)
}

/// ROCm prefill offload 的 CPU attention。query、GPU 已展开的 KV 与 RoPE 都以
/// BF16 传输；历史由 CPU 持有，计算结果也直接压成 BF16 返回 GPU。
#[allow(clippy::too_many_arguments)]
pub fn mla_attention_selected_bf16_history(mla: &MlaSpec, query: &[u16], query_rows: usize, query_start: usize, projected_kv: &[u16], rope: &[u16], history_rows: usize, selection: Option<(&[u32], usize)>) -> Result<Vec<u16>, String> {
    let q_head_dim = mla.q_head_dim();
    let qk_nope_dim = mla.qk_nope_dim();
    let kv_head_dim = mla.kv_head_dim();
    let value_dim = mla.value_dim();
    let output_cols = mla.num_heads.checked_mul(value_dim).ok_or("CPU MLA output cols 溢出")?;
    if query_rows == 0
        || query_start.checked_add(query_rows) != Some(history_rows)
        || query.len() != query_rows.saturating_mul(mla.q_projection_size)
        || projected_kv.len() != history_rows.saturating_mul(mla.kv_projection_size)
        || rope.len() != history_rows.saturating_mul(mla.qk_rope_head_dim)
    {
        return Err(format!(
            "CPU MLA BF16 history shape 非法: query={}/{}x{} start={query_start} kv={}/{}x{} rope={}/{}x{}",
            query.len(),
            query_rows,
            mla.q_projection_size,
            projected_kv.len(),
            history_rows,
            mla.kv_projection_size,
            rope.len(),
            history_rows,
            mla.qk_rope_head_dim
        ));
    }
    if selection.is_some_and(|(tokens, width)| width == 0 || tokens.len() != query_rows.saturating_mul(width)) {
        return Err("CPU MLA selection shape 非法".to_owned());
    }
    if let Some((tokens, width)) = selection {
        for (row, selected) in tokens.chunks_exact(width).enumerate() {
            let valid_rows = query_start + row + 1;
            if let Some(&token) = selected.iter().find(|&&token| token as usize >= valid_rows) {
                return Err(format!("CPU MLA row={row} selection token={token} 超过 causal rows={valid_rows}"));
            }
        }
        #[cfg(target_arch = "x86_64")]
        if mla.q_head_dim() == mla.value_dim()
            && mla.qk_nope_dim().is_multiple_of(32)
            && mla.qk_rope_head_dim.is_multiple_of(32)
            && mla.value_dim().is_multiple_of(16)
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512bf16")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return Ok(unsafe { mla_attention_selected_bf16_avx512_impl(mla, query, query_rows, query_start, projected_kv, rope, tokens, width) });
        }
    }
    if selection.is_none() {
        let q = CpuTensor { data: query.iter().map(|&bits| HalfBf16::from_bits(bits).to_f32()).collect(), rows: query_rows, cols: mla.q_projection_size };
        let kv = CpuTensor { data: projected_kv.iter().map(|&bits| HalfBf16::from_bits(bits).to_f32()).collect(), rows: history_rows, cols: mla.kv_projection_size };
        let k_rope = CpuTensor { data: rope.iter().map(|&bits| HalfBf16::from_bits(bits).to_f32()).collect(), rows: history_rows, cols: mla.qk_rope_head_dim };
        if let Some(output) = mla_attention_blas_chunk(mla, &q, &kv, &k_rope, query_start) {
            return Ok(output.data.into_iter().map(|value| HalfBf16::from_f32(value).to_bits()).collect());
        }
    }
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    let mut output = vec![0_u16; query_rows.saturating_mul(output_cols)];
    output.par_chunks_mut(output_cols).enumerate().try_for_each(|(row, output)| -> Result<(), String> {
        let valid_rows = query_start + row + 1;
        let selected = selection.map(|(tokens, width)| &tokens[row * width..(row + 1) * width]);
        for head in 0..mla.num_heads {
            let query_base = row * mla.q_projection_size + head * q_head_dim;
            let query_nope = &query[query_base..query_base + qk_nope_dim];
            let query_rope = &query[query_base + qk_nope_dim..query_base + q_head_dim];
            let selected_count = selected.map_or(valid_rows, <[u32]>::len);
            let mut scores = Vec::with_capacity(selected_count);
            let mut maximum = f32::NEG_INFINITY;
            for slot in 0..selected_count {
                let token = selected.map_or(slot, |tokens| tokens[slot] as usize);
                if token >= valid_rows {
                    return Err(format!("CPU MLA row={row} selection token={token} 超过 causal rows={valid_rows}"));
                }
                let kv_base = token * mla.kv_projection_size + head * kv_head_dim;
                let mut score = 0.0_f32;
                for column in 0..qk_nope_dim {
                    score += HalfBf16::from_bits(query_nope[column]).to_f32() * HalfBf16::from_bits(projected_kv[kv_base + column]).to_f32();
                }
                let rope_base = token * mla.qk_rope_head_dim;
                for column in 0..mla.qk_rope_head_dim {
                    score += HalfBf16::from_bits(query_rope[column]).to_f32() * HalfBf16::from_bits(rope[rope_base + column]).to_f32();
                }
                score *= scale;
                maximum = maximum.max(score);
                scores.push(score);
            }
            let mut denominator = 0.0_f32;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            for value in 0..value_dim {
                let mut sum = 0.0_f32;
                for (slot, &score) in scores.iter().enumerate() {
                    let token = selected.map_or(slot, |tokens| tokens[slot] as usize);
                    let value_index = token * mla.kv_projection_size + head * kv_head_dim + qk_nope_dim + value;
                    sum += score * HalfBf16::from_bits(projected_kv[value_index]).to_f32();
                }
                output[head * value_dim + value] = HalfBf16::from_f32(sum / denominator).to_bits();
            }
        }
        Ok(())
    })?;
    Ok(output)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn matmul_nn_f32_avx512(m: usize, n: usize, k: usize, left: &[f32], right: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    for row in 0..m {
        let mut column = 0;
        while column + 16 <= n {
            let mut sum = _mm512_setzero_ps();
            for inner in 0..k {
                let value = _mm512_set1_ps(left[row * k + inner]);
                let weight = _mm512_loadu_ps(right.as_ptr().add(inner * n + column));
                sum = _mm512_fmadd_ps(value, weight, sum);
            }
            _mm512_storeu_ps(output.as_mut_ptr().add(row * n + column), sum);
            column += 16;
        }
        for column in column..n {
            output[row * n + column] = (0..k).map(|inner| left[row * k + inner] * right[inner * n + column]).sum();
        }
    }
}

fn matmul_nn_f32(m: usize, n: usize, k: usize, left: &[f32], right: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("fma") {
        return unsafe { matmul_nn_f32_avx512(m, n, k, left, right, output) };
    }
    output.fill(0.0);
    for row in 0..m {
        for inner in 0..k {
            let value = left[row * k + inner];
            for column in 0..n {
                output[row * n + column] += value * right[inner * n + column];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn matmul_nt_accumulate_f32_avx512(m: usize, n: usize, k: usize, alpha: f32, beta: f32, left: &[f32], right: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    for row in 0..m {
        for column in 0..n {
            let mut sum = _mm512_setzero_ps();
            let mut inner = 0;
            while inner + 16 <= k {
                let lhs = _mm512_loadu_ps(left.as_ptr().add(row * k + inner));
                let rhs = _mm512_loadu_ps(right.as_ptr().add(column * k + inner));
                sum = _mm512_fmadd_ps(lhs, rhs, sum);
                inner += 16;
            }
            let mut value = _mm512_reduce_add_ps(sum);
            for inner in inner..k {
                value += left[row * k + inner] * right[column * k + inner];
            }
            output[row * n + column] = alpha * value + beta * output[row * n + column];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_nt_accumulate_f32(m: usize, n: usize, k: usize, alpha: f32, beta: f32, left: &[f32], right: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("fma") {
        return unsafe { matmul_nt_accumulate_f32_avx512(m, n, k, alpha, beta, left, right, output) };
    }
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0_f32;
            for inner in 0..k {
                sum += left[row * k + inner] * right[column * k + inner];
            }
            output[row * n + column] = alpha * sum + beta * output[row * n + column];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn absorbed_attention_row_q8_scalar(
    mla: &MlaSpec,
    query: &[u16],
    row: usize,
    query_start: usize,
    absorbed_query: &[Vec<u16>],
    latent_codes: &[u8],
    latent_scales: &[f32],
    latent_group_size: usize,
    rope: &[u16],
    selected: Option<&[u32]>,
) -> Vec<f32> {
    let latent_dim = mla.kv_lora_rank;
    let rope_dim = mla.qk_rope_head_dim;
    let q_head_dim = mla.q_head_dim();
    let nope_dim = mla.qk_nope_dim();
    let groups = latent_dim / latent_group_size;
    let key_count = selected.map_or(query_start + row + 1, <[u32]>::len);
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    let mut output = vec![0.0_f32; mla.num_heads * latent_dim];
    for head in 0..mla.num_heads {
        let q_absorbed = &absorbed_query[head][row * latent_dim..(row + 1) * latent_dim];
        let q_rope_start = row * mla.q_projection_size + head * q_head_dim + nope_dim;
        let q_rope = &query[q_rope_start..q_rope_start + rope_dim];
        let mut scores = Vec::with_capacity(key_count);
        let mut maximum = f32::NEG_INFINITY;
        for slot in 0..key_count {
            let token = selected.map_or(slot, |tokens| tokens[slot] as usize);
            let mut score = 0.0_f32;
            for column in 0..latent_dim {
                let group = column / latent_group_size;
                let value = (latent_codes[token * latent_dim + column] as i8) as f32 * latent_scales[token * groups + group];
                score += HalfBf16::from_bits(q_absorbed[column]).to_f32() * value;
            }
            for column in 0..rope_dim {
                score += HalfBf16::from_bits(q_rope[column]).to_f32() * HalfBf16::from_bits(rope[token * rope_dim + column]).to_f32();
            }
            score *= scale;
            maximum = maximum.max(score);
            scores.push(score);
        }
        let mut denominator = 0.0_f32;
        for score in &mut scores {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        let output = &mut output[head * latent_dim..(head + 1) * latent_dim];
        for (slot, score) in scores.into_iter().enumerate() {
            let token = selected.map_or(slot, |tokens| tokens[slot] as usize);
            let probability = score / denominator;
            for column in 0..latent_dim {
                let group = column / latent_group_size;
                let value = (latent_codes[token * latent_dim + column] as i8) as f32 * latent_scales[token * groups + group];
                output[column] += probability * value;
            }
        }
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512bf16,fma")]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn absorbed_attention_row_q8_avx512(
    mla: &MlaSpec,
    query: &[u16],
    row: usize,
    query_start: usize,
    absorbed_query: &[Vec<u16>],
    latent_codes: &[u8],
    latent_scales: &[f32],
    latent_group_size: usize,
    rope: &[u16],
    selected: Option<&[u32]>,
) -> Vec<f32> {
    use std::arch::x86_64::*;

    const HEAD_CHUNK: usize = 8;
    let latent_dim = mla.kv_lora_rank;
    let rope_dim = mla.qk_rope_head_dim;
    let q_head_dim = mla.q_head_dim();
    let nope_dim = mla.qk_nope_dim();
    let groups = latent_dim / latent_group_size;
    let key_count = selected.map_or(query_start + row + 1, <[u32]>::len);
    let attention_scale = 1.0 / (q_head_dim as f32).sqrt();
    let mut scores = vec![0.0_f32; mla.num_heads * key_count];
    for head_base in (0..mla.num_heads).step_by(HEAD_CHUNK) {
        let head_count = HEAD_CHUNK.min(mla.num_heads - head_base);
        for slot in 0..key_count {
            let token = selected.map_or(slot, |tokens| tokens[slot] as usize);
            let mut sums = [_mm512_setzero_ps(); HEAD_CHUNK];
            for group in 0..groups {
                let scale = _mm512_set1_ps(latent_scales[token * groups + group]);
                for column in (group * latent_group_size..(group + 1) * latent_group_size).step_by(32) {
                    let low = _mm_loadu_si128(latent_codes.as_ptr().add(token * latent_dim + column).cast());
                    let high = _mm_loadu_si128(latent_codes.as_ptr().add(token * latent_dim + column + 16).cast());
                    let low = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(low)), scale);
                    let high = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(high)), scale);
                    let values = std::mem::transmute::<__m512bh, __m512i>(_mm512_cvtne2ps_pbh(high, low));
                    let values = std::mem::transmute::<__m512i, __m512bh>(values);
                    for local_head in 0..head_count {
                        let query = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(absorbed_query[head_base + local_head].as_ptr().add(row * latent_dim + column).cast()));
                        sums[local_head] = _mm512_dpbf16_ps(sums[local_head], query, values);
                    }
                }
            }
            for column in (0..rope_dim).step_by(32) {
                let key = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(rope.as_ptr().add(token * rope_dim + column).cast()));
                for local_head in 0..head_count {
                    let head = head_base + local_head;
                    let query_start = row * mla.q_projection_size + head * q_head_dim + nope_dim + column;
                    let query = std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(query.as_ptr().add(query_start).cast()));
                    sums[local_head] = _mm512_dpbf16_ps(sums[local_head], query, key);
                }
            }
            for local_head in 0..head_count {
                scores[(head_base + local_head) * key_count + slot] = _mm512_reduce_add_ps(sums[local_head]) * attention_scale;
            }
        }
    }
    for scores in scores.chunks_exact_mut(key_count) {
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0_f32;
        for score in scores.iter_mut() {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        for score in scores {
            *score /= denominator;
        }
    }

    let mut output = vec![0.0_f32; mla.num_heads * latent_dim];
    for head_base in (0..mla.num_heads).step_by(HEAD_CHUNK) {
        let head_count = HEAD_CHUNK.min(mla.num_heads - head_base);
        for column in (0..latent_dim).step_by(16) {
            let group = column / latent_group_size;
            let mut sums = [_mm512_setzero_ps(); HEAD_CHUNK];
            for slot in 0..key_count {
                let token = selected.map_or(slot, |tokens| tokens[slot] as usize);
                let codes = _mm_loadu_si128(latent_codes.as_ptr().add(token * latent_dim + column).cast());
                let scale = _mm512_set1_ps(latent_scales[token * groups + group]);
                let values = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(codes)), scale);
                for local_head in 0..head_count {
                    let probability = _mm512_set1_ps(scores[(head_base + local_head) * key_count + slot]);
                    sums[local_head] = _mm512_fmadd_ps(probability, values, sums[local_head]);
                }
            }
            for local_head in 0..head_count {
                _mm512_storeu_ps(output.as_mut_ptr().add((head_base + local_head) * latent_dim + column), sums[local_head]);
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn absorbed_attention_row_q8(
    mla: &MlaSpec,
    query: &[u16],
    row: usize,
    query_start: usize,
    absorbed_query: &[Vec<u16>],
    latent_codes: &[u8],
    latent_scales: &[f32],
    latent_group_size: usize,
    rope: &[u16],
    selected: Option<&[u32]>,
) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if mla.kv_lora_rank.is_multiple_of(32)
        && mla.qk_rope_head_dim.is_multiple_of(32)
        && latent_group_size.is_multiple_of(32)
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512bf16")
        && std::arch::is_x86_feature_detected!("fma")
    {
        return unsafe { absorbed_attention_row_q8_avx512(mla, query, row, query_start, absorbed_query, latent_codes, latent_scales, latent_group_size, rope, selected) };
    }
    absorbed_attention_row_q8_scalar(mla, query, row, query_start, absorbed_query, latent_codes, latent_scales, latent_group_size, rope, selected)
}

/// CPU prefill 的吸收式 MLA。历史只保存 decode 共用的 Q8G latent 与 BF16
/// RoPE；K 投影吸收到 query，softmax 后先聚合 latent，再通过 V 权重还原输出。
#[allow(clippy::too_many_arguments)]
pub fn mla_attention_absorbed_q8_history(
    mla: &MlaSpec,
    query: &[u16],
    query_rows: usize,
    query_start: usize,
    latent_codes: &[u8],
    latent_scale_bytes: &[u8],
    latent_group_size: usize,
    rope_bytes: &[u8],
    history_rows: usize,
    kv_b: &[f32],
    selection: Option<(&[u32], usize)>,
) -> Result<Vec<u16>, String> {
    let latent_dim = mla.kv_lora_rank;
    let rope_dim = mla.qk_rope_head_dim;
    let q_head_dim = mla.q_head_dim();
    let nope_dim = mla.qk_nope_dim();
    let kv_head_dim = mla.kv_head_dim();
    let value_dim = mla.value_dim();
    let heads = mla.num_heads;
    let groups = latent_dim.checked_div(latent_group_size.max(1)).ok_or("CPU absorbed MLA group 除零")?;
    let expected_scales = history_rows.checked_mul(groups).and_then(|elements| elements.checked_mul(2)).ok_or("CPU absorbed MLA scales 大小溢出")?;
    if query_rows == 0
        || query_start.checked_add(query_rows) != Some(history_rows)
        || latent_group_size == 0
        || !latent_dim.is_multiple_of(latent_group_size)
        || query.len() != query_rows.saturating_mul(mla.q_projection_size)
        || latent_codes.len() != history_rows.saturating_mul(latent_dim)
        || latent_scale_bytes.len() != expected_scales
        || rope_bytes.len() != history_rows.saturating_mul(rope_dim).saturating_mul(2)
        || kv_b.len() != mla.kv_projection_size.saturating_mul(latent_dim)
    {
        return Err(format!(
            "CPU absorbed MLA shape 非法: query={}/{}x{} start={query_start} latent={}/{}x{} scales={}/{} rope={}/{}x{} kv_b={}/{}x{} group={latent_group_size}",
            query.len(),
            query_rows,
            mla.q_projection_size,
            latent_codes.len(),
            history_rows,
            latent_dim,
            latent_scale_bytes.len(),
            expected_scales,
            rope_bytes.len(),
            history_rows,
            rope_dim,
            kv_b.len(),
            mla.kv_projection_size,
            latent_dim,
        ));
    }
    if let Some((tokens, width)) = selection {
        if width == 0 || tokens.len() != query_rows.saturating_mul(width) {
            return Err("CPU absorbed MLA selection shape 非法".to_owned());
        }
        for (row, tokens) in tokens.chunks_exact(width).enumerate() {
            let valid = query_start + row + 1;
            if let Some(&token) = tokens.iter().find(|&&token| token as usize >= valid) {
                return Err(format!("CPU absorbed MLA row={row} selection token={token} 超过 causal rows={valid}"));
            }
        }
    }

    let query_f32 = query.iter().map(|&bits| HalfBf16::from_bits(bits).to_f32()).collect::<Vec<_>>();
    let latent_scales = latent_scale_bytes.chunks_exact(2).map(|bytes| HalfBf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()).collect::<Vec<_>>();
    let rope = rope_bytes.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();

    // 每个 head 的 K 吸收是独立 GEMM；head-major 保证后续权重和输出投影连续。
    let absorbed_query = (0..heads)
        .into_par_iter()
        .map(|head| {
            let mut q_nope = Vec::with_capacity(query_rows * nope_dim);
            for row in 0..query_rows {
                let start = row * mla.q_projection_size + head * q_head_dim;
                q_nope.extend_from_slice(&query_f32[start..start + nope_dim]);
            }
            let weight_start = head * kv_head_dim * latent_dim;
            let weight = &kv_b[weight_start..weight_start + nope_dim * latent_dim];
            let mut output = vec![0.0_f32; query_rows * latent_dim];
            matmul_nn_f32(query_rows, latent_dim, nope_dim, &q_nope, weight, &mut output);
            output.into_iter().map(|value| HalfBf16::from_f32(value).to_bits()).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // selection 每行不同；自有 AVX-512 kernel 直接读取 Q8G history，不建立
    // F32 latent，也不把 2048 个随机 token gather 成临时矩阵。
    let weighted_latent = (0..query_rows)
        .into_par_iter()
        .map(|row| {
            let selected = selection.map(|(tokens, width)| &tokens[row * width..(row + 1) * width]);
            absorbed_attention_row_q8(mla, query, row, query_start, &absorbed_query, latent_codes, &latent_scales, latent_group_size, &rope, selected)
        })
        .collect::<Vec<_>>();

    let head_outputs = (0..heads)
        .into_par_iter()
        .map(|head| {
            let mut weighted = Vec::with_capacity(query_rows * latent_dim);
            for row in &weighted_latent {
                weighted.extend_from_slice(&row[head * latent_dim..(head + 1) * latent_dim]);
            }
            let weight_start = (head * kv_head_dim + nope_dim) * latent_dim;
            let weight = &kv_b[weight_start..weight_start + value_dim * latent_dim];
            let mut output = vec![0.0_f32; query_rows * value_dim];
            matmul_nt_accumulate_f32(query_rows, value_dim, latent_dim, 1.0, 0.0, &weighted, weight, &mut output);
            output
        })
        .collect::<Vec<_>>();
    let output_cols = heads.checked_mul(value_dim).ok_or("CPU absorbed MLA output cols 溢出")?;
    let mut output = vec![0_u16; query_rows * output_cols];
    output.par_chunks_mut(output_cols).enumerate().for_each(|(row, output)| {
        for head in 0..heads {
            let values = &head_outputs[head][row * value_dim..(row + 1) * value_dim];
            for (destination, &value) in output[head * value_dim..(head + 1) * value_dim].iter_mut().zip(values) {
                *destination = HalfBf16::from_f32(value).to_bits();
            }
        }
    });
    Ok(output)
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

    #[test]
    fn selected_bf16_history_uses_absolute_chunk_positions() {
        let spec = MlaSpec { q_lora_rank: 1, kv_lora_rank: 1, qk_rope_head_dim: 2, q_projection_size: 4, kv_projection_size: 4, num_heads: 1, rope_theta: 10_000.0, rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf };
        let bf16 = |values: &[f32]| values.iter().map(|&value| HalfBf16::from_f32(value).to_bits()).collect::<Vec<_>>();
        let query = bf16(&[0.0; 8]);
        let projected = bf16(&[0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 2.0, 20.0, 0.0, 0.0, 3.0, 30.0, 0.0, 0.0, 4.0, 40.0]);
        let rope = bf16(&[0.0; 8]);
        let output = mla_attention_selected_bf16_history(&spec, &query, 2, 2, &projected, &rope, 4, Some((&[0, 2, 1, 3], 2))).unwrap();
        let output = output.into_iter().map(|bits| HalfBf16::from_bits(bits).to_f32()).collect::<Vec<_>>();
        assert_eq!(output, vec![2.0, 20.0, 3.0, 30.0]);
    }

    #[test]
    fn selected_bf16_avx512_matches_uniform_attention() {
        let spec = MlaSpec { q_lora_rank: 1, kv_lora_rank: 1, qk_rope_head_dim: 32, q_projection_size: 128, kv_projection_size: 192, num_heads: 2, rope_theta: 10_000.0, rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf };
        let query = vec![0_u16; 2 * spec.q_projection_size];
        let mut projected = vec![0_u16; 4 * spec.kv_projection_size];
        for token in 0..4 {
            for head in 0..spec.num_heads {
                let value = HalfBf16::from_f32((token * 10 + head) as f32).to_bits();
                let start = token * spec.kv_projection_size + head * spec.kv_head_dim() + spec.qk_nope_dim();
                projected[start..start + spec.value_dim()].fill(value);
            }
        }
        let rope = vec![0_u16; 4 * spec.qk_rope_head_dim];
        let output = mla_attention_selected_bf16_history(&spec, &query, 2, 2, &projected, &rope, 4, Some((&[0, 2, 1, 3], 2))).unwrap();
        for row in 0..2 {
            for head in 0..spec.num_heads {
                let expected = HalfBf16::from_f32((row * 10 + 10 + head) as f32).to_bits();
                let start = row * spec.q_projection_size + head * spec.value_dim();
                assert!(output[start..start + spec.value_dim()].iter().all(|&value| value == expected));
            }
        }
    }

    #[test]
    fn absorbed_q8_history_matches_expanded_kv() {
        let spec = MlaSpec { q_lora_rank: 1, kv_lora_rank: 3, qk_rope_head_dim: 2, q_projection_size: 8, kv_projection_size: 8, num_heads: 2, rope_theta: 10_000.0, rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf };
        let bf16 = |values: &[f32]| values.iter().map(|&value| HalfBf16::from_f32(value).to_bits()).collect::<Vec<_>>();
        let query = bf16(&[0.5, -0.25, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, -0.5, 0.75, 0.0, 0.0, 0.25, -1.0, 0.0, 0.0]);
        let latent_values = [1.0, 2.0, -1.0, 2.0, -1.0, 1.0, -1.0, 1.0, 2.0, 1.0, -2.0, 1.0];
        let latent_codes = latent_values.iter().map(|&value| (value as i8) as u8).collect::<Vec<_>>();
        let scale = HalfBf16::from_f32(1.0).to_bits().to_le_bytes();
        let latent_scales = (0..4).flat_map(|_| scale).collect::<Vec<_>>();
        let rope_values = [0.0; 8];
        let rope = bf16(&rope_values);
        let rope_bytes = rope.iter().flat_map(|bits| bits.to_le_bytes()).collect::<Vec<_>>();
        let kv_b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
        let mut projected = Vec::with_capacity(4 * spec.kv_projection_size);
        for latent in latent_values.chunks_exact(spec.kv_lora_rank) {
            for weight in kv_b.chunks_exact(spec.kv_lora_rank) {
                projected.push(weight.iter().zip(latent).map(|(weight, value)| weight * value).sum::<f32>());
            }
        }
        let projected = bf16(&projected);
        let selection = [0, 2, 1, 3];
        let expected = mla_attention_selected_bf16_history(&spec, &query, 2, 2, &projected, &rope, 4, Some((&selection, 2))).unwrap();
        let actual = mla_attention_absorbed_q8_history(&spec, &query, 2, 2, &latent_codes, &latent_scales, 3, &rope_bytes, 4, &kv_b, Some((&selection, 2))).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            let actual = HalfBf16::from_bits(actual).to_f32();
            let expected = HalfBf16::from_bits(expected).to_f32();
            assert!((actual - expected).abs() < 0.02, "actual={actual} expected={expected}");
        }
    }

    #[test]
    fn absorbed_q8_avx512_matches_expanded_kv() {
        let spec = MlaSpec { q_lora_rank: 1, kv_lora_rank: 64, qk_rope_head_dim: 32, q_projection_size: 128, kv_projection_size: 128, num_heads: 2, rope_theta: 10_000.0, rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf };
        let bf16 = |values: &[f32]| values.iter().map(|&value| HalfBf16::from_f32(value).to_bits()).collect::<Vec<_>>();
        let query_values = (0..2 * spec.q_projection_size).map(|index| (index as i32 % 11 - 5) as f32 * 0.03125).collect::<Vec<_>>();
        let query = bf16(&query_values);
        let latent_codes = (0..4 * spec.kv_lora_rank).map(|index| ((index as i32 * 7 % 31) - 15) as i8 as u8).collect::<Vec<_>>();
        let scale_value = HalfBf16::from_f32(0.0625).to_bits();
        let latent_scales = (0..4).flat_map(|_| scale_value.to_le_bytes()).collect::<Vec<_>>();
        let rope_values = (0..4 * spec.qk_rope_head_dim).map(|index| (index as i32 % 9 - 4) as f32 * 0.015625).collect::<Vec<_>>();
        let rope = bf16(&rope_values);
        let rope_bytes = rope.iter().flat_map(|bits| bits.to_le_bytes()).collect::<Vec<_>>();
        let kv_b = (0..spec.kv_projection_size * spec.kv_lora_rank).map(|index| (index as i32 * 13 % 17 - 8) as f32 * 0.0078125).collect::<Vec<_>>();
        let latent = latent_codes.iter().map(|&code| (code as i8) as f32 * 0.0625).collect::<Vec<_>>();
        let mut projected = Vec::with_capacity(4 * spec.kv_projection_size);
        for latent in latent.chunks_exact(spec.kv_lora_rank) {
            for weight in kv_b.chunks_exact(spec.kv_lora_rank) {
                projected.push(weight.iter().zip(latent).map(|(weight, value)| weight * value).sum::<f32>());
            }
        }
        let projected = bf16(&projected);
        let selection = [0, 2, 1, 3];
        let expected = mla_attention_selected_bf16_history(&spec, &query, 2, 2, &projected, &rope, 4, Some((&selection, 2))).unwrap();
        let actual = mla_attention_absorbed_q8_history(&spec, &query, 2, 2, &latent_codes, &latent_scales, 64, &rope_bytes, 4, &kv_b, Some((&selection, 2))).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            let actual = HalfBf16::from_bits(actual).to_f32();
            let expected = HalfBf16::from_bits(expected).to_f32();
            assert!((actual - expected).abs() < 0.02, "actual={actual} expected={expected}");
        }
    }

    #[test]
    fn blas_chunk_uses_absolute_causal_prefix() {
        let spec = MlaSpec { q_lora_rank: 1, kv_lora_rank: 1, qk_rope_head_dim: 2, q_projection_size: 4, kv_projection_size: 4, num_heads: 1, rope_theta: 10_000.0, rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf };
        let query = CpuTensor { data: vec![0.0; 8], rows: 2, cols: 4 };
        let projected = CpuTensor { data: vec![0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 2.0, 20.0, 0.0, 0.0, 3.0, 30.0, 0.0, 0.0, 4.0, 40.0], rows: 4, cols: 4 };
        let rope = CpuTensor { data: vec![0.0; 8], rows: 4, cols: 2 };
        let Some(output) = mla_attention_blas_chunk(&spec, &query, &projected, &rope, 2) else { return };
        assert_eq!(output.data, vec![2.0, 20.0, 2.5, 25.0]);
    }
}
