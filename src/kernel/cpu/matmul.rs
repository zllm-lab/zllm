//! 矩阵乘 `y[N,K] = x[N,M] @ w[K,M]^T`。w 行优先。

use rayon::prelude::*;

use super::blas;
use wide::f32x8;

const SIMD_LANES: usize = 8;

pub fn matmul(x: &[f32], w: &[f32], n: usize, m: usize, k: usize, out: &mut [f32]) {
    const OUTPUT_CHUNK: usize = 16;

    if n == 1 {
        out[..k].par_chunks_mut(OUTPUT_CHUNK).enumerate().for_each(|(chunk_index, chunk)| {
            let output_start = chunk_index * OUTPUT_CHUNK;
            let mut offset = 0;
            while offset + 4 <= chunk.len() {
                let row = output_start + offset;
                let sums = dot4(x, &w[row * m..], m);
                chunk[offset..offset + 4].copy_from_slice(&sums);
                offset += 4;
            }
            while offset < chunk.len() {
                let row = output_start + offset;
                chunk[offset] = dot(x, &w[row * m..(row + 1) * m]);
                offset += 1;
            }
        });
        return;
    }

    // OpenBLAS 单线程微内核负责块内缓存复用，Rayon 负责块间并行。
    // 不能让每个并行块再启动 BLAS 线程，否则 attention/expert 嵌套时会过度订阅。
    if n >= 8 && m > 0 && k > 0 && blas::available() {
        let threads = rayon::current_num_threads().max(1);
        let rows_per_task = n.div_ceil(threads).clamp(8, 256);
        out[..n * k].par_chunks_mut(rows_per_task * k).enumerate().for_each(|(task, output)| {
            let row = task * rows_per_task;
            let rows = output.len() / k;
            let input = &x[row * m..(row + rows) * m];
            let loaded = blas::sgemm_nt(rows, k, m, 1.0, input, w, output);
            debug_assert!(loaded, "OpenBLAS 在矩阵执行期间不可用");
        });
        return;
    }

    matmul_tiled(x, w, n, m, k, out);
}

/// Cache-blocked GEMM for `n >= 8` without OpenBLAS。并行 over (n,k) tile,
/// 每 tile 在本地 [ni×ki] block 里按 m 分块累加。c 内层、r 外层的顺序让 weight 行
/// 切片跨 ni 次复用(进 L1);unsafe 直接写 out,跳过 collect+scatter 双 pass。
///
/// SAFETY 调用方须保证 `out` 已零初始化,且 (ii,kk) tile 之间无输出重叠。
fn matmul_tiled(x: &[f32], w: &[f32], n: usize, m: usize, k: usize, out: &mut [f32]) {
    const NB: usize = 32;
    const KB: usize = 32;
    // SAFETY: (ii, kk) 网格把 out 切成不重叠的 [ii..ii+ni) × [kk..kk+ki) 块,
    // par_iter 保证 worker 间互不踩踏;out 已零初始化时 `+= value` 等价于 `=`。
    // 指针在闭包外 cast 成 usize(par_iter 闭包须 Send,原始指针非 Send)。
    let x_addr = x.as_ptr() as usize;
    let w_addr = w.as_ptr() as usize;
    let out_addr = out.as_mut_ptr() as usize;
    unsafe {
        (0..n).step_by(NB).flat_map(|ii| (0..k).step_by(KB).map(move |kk| (ii, kk))).collect::<Vec<_>>().into_par_iter().for_each(|(ii, kk)| {
            let ni = NB.min(n - ii);
            let ki = KB.min(k - kk);
            let x_base = x_addr as *const f32;
            let out_base = out_addr as *mut f32;
            for c in 0..ki {
                let w_row = std::slice::from_raw_parts((w_addr as *const f32).add((kk + c) * m), m);
                let out_col = out_base.add(ii * k + kk + c);
                for r in 0..ni {
                    let xr = std::slice::from_raw_parts(x_base.add((ii + r) * m), m);
                    *out_col.add(r * k) += dot(xr, w_row);
                }
            }
        });
    }
}

/// 同一输入向量同时计算四个输出行，减少输入加载并隐藏 Broadwell 的 FMA 延迟。
fn dot4(x: &[f32], w: &[f32], m: usize) -> [f32; 4] {
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);
    let full = m / SIMD_LANES * SIMD_LANES;
    let mut i = 0;
    while i < full {
        let xv = f32x8::from(<[f32; SIMD_LANES]>::try_from(&x[i..i + SIMD_LANES]).unwrap());
        let w0 = f32x8::from(<[f32; SIMD_LANES]>::try_from(&w[i..i + SIMD_LANES]).unwrap());
        let w1 = f32x8::from(<[f32; SIMD_LANES]>::try_from(&w[m + i..m + i + SIMD_LANES]).unwrap());
        let w2 = f32x8::from(<[f32; SIMD_LANES]>::try_from(&w[2 * m + i..2 * m + i + SIMD_LANES]).unwrap());
        let w3 = f32x8::from(<[f32; SIMD_LANES]>::try_from(&w[3 * m + i..3 * m + i + SIMD_LANES]).unwrap());
        acc0 = xv.mul_add(w0, acc0);
        acc1 = xv.mul_add(w1, acc1);
        acc2 = xv.mul_add(w2, acc2);
        acc3 = xv.mul_add(w3, acc3);
        i += SIMD_LANES;
    }
    let mut sums = [acc0.reduce_add(), acc1.reduce_add(), acc2.reduce_add(), acc3.reduce_add()];
    while i < m {
        let xv = x[i];
        sums[0] += xv * w[i];
        sums[1] += xv * w[m + i];
        sums[2] += xv * w[2 * m + i];
        sums[3] += xv * w[3 * m + i];
        i += 1;
    }
    sums
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = f32x8::splat(0.0);
    let pair = a.len().min(b.len());
    let full = pair / SIMD_LANES * SIMD_LANES;
    let mut i = 0usize;
    while i < full {
        let a = f32x8::from(<[f32; SIMD_LANES]>::try_from(&a[i..i + SIMD_LANES]).unwrap());
        let b = f32x8::from(<[f32; SIMD_LANES]>::try_from(&b[i..i + SIMD_LANES]).unwrap());
        acc = a.mul_add(b, acc);
        i += SIMD_LANES;
    }
    let mut tail = 0.0;
    while i < pair {
        tail += a[i] * b[i];
        i += 1;
    }
    acc.reduce_add() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_works_like_sum_product() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [2.0_f32, 0.5, -1.0, 3.0];
        assert_eq!(dot(&a, &b), 12.0);
    }

    #[test]
    fn matmul_matches_rowwise_dot() {
        let x = [1.0_f32, 2.0];
        let w = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 2];
        matmul(&x, &w, 1, 2, 2, &mut out);
        assert_eq!(out, [5.0, 11.0]);
    }

    #[test]
    fn batched_matmul_matches_rowwise_dot() {
        let x: Vec<f32> = (0..24).map(|value| value as f32 * 0.25).collect();
        let w = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0_f32; 16];
        matmul(&x, &w, 8, 3, 2, &mut out);
        for row in 0..8 {
            assert_eq!(out[row * 2], dot(&x[row * 3..row * 3 + 3], &w[..3]));
            assert_eq!(out[row * 2 + 1], dot(&x[row * 3..row * 3 + 3], &w[3..]));
        }
    }
}
