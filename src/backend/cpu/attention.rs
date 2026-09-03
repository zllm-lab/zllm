//! CPU Attention cache、DSA 状态与执行能力。

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::cell::RefCell;

use crate::{
    attention::{
        dsa::{self, DsaSpec},
        gqa::GqaSpec,
        mla::MlaSpec,
    },
    backend::{Backend, BackendError, BlockAttentionBackend, DecodeBackend, DsaPrefillBackend, GqaPrefillBackend, MlaPrefillBackend, compute_error as compute},
    kernel::cpu::{
        CpuTensor,
        gqa::{gqa_prefill_attention, gqa_prefill_attention_at, gqa_prefill_attention_at_visible},
        matmul::dot,
    },
};

use super::context::{CpuContext, CpuWeight};

impl BlockAttentionBackend for CpuContext {
    fn block_attention(&self, query: &CpuTensor, key: &CpuTensor, value: &CpuTensor, spec: &crate::attention::block::BlockAttentionSpec) -> Result<CpuTensor, BackendError> {
        let data = block_attention_cpu(query, &[(key, value)], spec)?;
        Ok(CpuTensor { data, rows: query.rows, cols: query.cols })
    }

    fn block_attention_segments(&self, query: &CpuTensor, keys: &[&CpuTensor], values: &[&CpuTensor], spec: &crate::attention::block::BlockAttentionSpec) -> Result<CpuTensor, BackendError> {
        if keys.len() != values.len() || keys.is_empty() {
            return Err(compute(format!("CPU block attention segments K/V 数量={}/{}", keys.len(), values.len())));
        }
        let segments = keys.iter().copied().zip(values.iter().copied()).collect::<Vec<_>>();
        let data = block_attention_cpu(query, &segments, spec)?;
        Ok(CpuTensor { data, rows: query.rows, cols: query.cols })
    }

    fn block_attention_prefix_suffix(
        &self,
        query: &CpuTensor,
        prefix_key: &CpuTensor,
        prefix_value: &CpuTensor,
        suffix_key: &CpuTensor,
        suffix_value: &CpuTensor,
        spec: &crate::attention::block::BlockAttentionSpec,
    ) -> Result<CpuTensor, BackendError> {
        let data = block_attention_cpu(query, &[(prefix_key, prefix_value), (suffix_key, suffix_value)], spec)?;
        Ok(CpuTensor { data, rows: query.rows, cols: query.cols })
    }
}

pub(crate) fn block_attention_cpu(query: &CpuTensor, segments: &[(&CpuTensor, &CpuTensor)], spec: &crate::attention::block::BlockAttentionSpec) -> Result<Vec<f32>, BackendError> {
    let query_cols = spec.geometry.query_columns().map_err(compute)?;
    let kv_cols = spec.geometry.kv_columns().map_err(compute)?;
    let kv_rows = segments.iter().try_fold(0usize, |rows, (key, value)| {
        if key.rows != value.rows || key.cols != kv_cols || value.cols != kv_cols || key.data.len() != key.rows.saturating_mul(kv_cols) || value.data.len() != value.rows.saturating_mul(kv_cols) {
            return Err(compute(format!("CPU block attention KV shape 非法: K=[{},{}] V=[{},{}]", key.rows, key.cols, value.rows, value.cols)));
        }
        rows.checked_add(key.rows).ok_or_else(|| compute("CPU block attention KV rows 溢出"))
    })?;
    spec.validate(query.rows, kv_rows).map_err(compute)?;
    if query.cols != query_cols || query.data.len() != query.rows.saturating_mul(query_cols) {
        return Err(compute(format!("CPU block attention query=[{},{}]，期望 cols={query_cols}", query.rows, query.cols)));
    }
    #[cfg(target_arch = "x86_64")]
    if spec.geometry.head_dim.is_multiple_of(16) && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
        return Ok(unsafe { block_attention_avx512(query, segments, spec) });
    }
    let mut key = Vec::with_capacity(kv_rows * kv_cols);
    let mut value = Vec::with_capacity(kv_rows * kv_cols);
    for (segment_key, segment_value) in segments {
        key.extend_from_slice(&segment_key.data);
        value.extend_from_slice(&segment_value.data);
    }
    crate::attention::block::attention_f32(&query.data, &key, &value, query.rows, kv_rows, spec).map_err(compute)
}

/// 跨线程共享的原始指针:query 只读;output 的每个 (行块, head) 单元由
/// 唯一任务写入,互不重叠,以此声明 Send 是安全的。必须经 `get()` 取指针:
/// 闭包对 `.0` 的不相交字段捕获会绕过本包装直接捕获裸指针。
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct SendPtr<P>(P);

#[cfg(target_arch = "x86_64")]
unsafe impl<P> Send for SendPtr<P> {}

#[cfg(target_arch = "x86_64")]
unsafe impl<P> Sync for SendPtr<P> {}

#[cfg(target_arch = "x86_64")]
impl<P: Copy> SendPtr<P> {
    fn get(&self) -> P {
        self.0
    }
}

// 固定 team 的 worker / 主线程各自持有 score 缓冲,跨调用复用,
// 消除逐调用的 64KB 分配与首次触碰页错误。
#[cfg(target_arch = "x86_64")]
thread_local! {
    static WORKER_SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// head-major 融合块注意力:任务粒度 = 行块 × 单个 head,K/V 各扫描一次
/// 即可服务块内全部行(旧实现按 query 行重复扫同一 head 的 K/V)。
/// 行块按"可见区间彼此重叠"贪心分组(上限 8 行):同 session 的因果行
/// 共享一次扫描,不相交区间(跨 session)自然分块,避免一个块横跨多个
/// session 造成 K/V 重复扫描;softmax 权重保持每行独立,块内各行可见
/// 区间不一致时按行判断归属,正确性不依赖区间单调。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn block_attention_avx512(query: &CpuTensor, segments: &[(&CpuTensor, &CpuTensor)], spec: &crate::attention::block::BlockAttentionSpec) -> Vec<f32> {
    const ROW_CHUNK: usize = 8;
    let heads = spec.geometry.num_heads;
    let dim = spec.geometry.head_dim;
    let query_cols = heads * dim;
    let kv_cols = spec.geometry.num_kv_heads * dim;
    let group = spec.geometry.group_size();
    let rows = query.rows;
    let visible = &spec.visible;
    // (行基址, 联合区间起点, 联合区间终点) —— 区间不相交即开新块。
    let mut chunks = Vec::with_capacity(rows.div_ceil(ROW_CHUNK) + 1);
    {
        let mut base = 0usize;
        let mut union_start = usize::MAX;
        let mut union_end = 0usize;
        for row in 0..rows {
            let range = &visible[row];
            if row - base > 0 && (row - base == ROW_CHUNK || range.start >= union_end || range.end <= union_start) {
                chunks.push((base, union_start, union_end));
                base = row;
            }
            union_start = union_start.min(range.start);
            union_end = union_end.max(range.end);
        }
        if base < rows {
            chunks.push((base, union_start, union_end));
        }
    }
    // score 缓冲按最大联合跨度分配,每个线程复用于它分到的全部任务。
    let stride = chunks.iter().map(|&(_, start, end)| end.saturating_sub(start)).max().unwrap_or(1).max(1);
    let tasks = chunks.len() * heads;
    let threads = crate::kernel::cpu::allowed_parallelism().min(16).min(tasks.max(1));
    let tasks_per_thread = tasks.div_ceil(threads);
    let mut output = vec![0.0_f32; query.data.len()];
    let query_ptr = SendPtr(query.data.as_ptr());
    let output_ptr = SendPtr(output.as_mut_ptr());
    crate::kernel::cpu::team::team_execute(threads, move |thread| unsafe {
        let first_task = thread * tasks_per_thread;
        let task_end = tasks.min(first_task + tasks_per_thread);
        if first_task >= task_end {
            return;
        }
        let query_ptr = query_ptr.get();
        let output_ptr = output_ptr.get();
        WORKER_SCORES.with(|cell| {
            let scores = &mut *cell.borrow_mut();
            scores.resize(ROW_CHUNK * stride, 0.0);
            for task in first_task..task_end {
                let chunk = task / heads;
                let head = task % heads;
                let kv_head = head / group;
                let (row_base, span_start, span_end) = chunks[chunk];
                let row_limit = chunks.get(chunk + 1).map_or(rows, |&(next, _, _)| next).min(row_base + ROW_CHUNK);
                let row_count = row_limit - row_base;
                let mut starts = [0usize; ROW_CHUNK];
                let mut ends = [0usize; ROW_CHUNK];
                let mut queries = [std::ptr::null(); ROW_CHUNK];
                for row in 0..row_count {
                    let range = &visible[row_base + row];
                    starts[row] = range.start;
                    ends[row] = range.end;
                    queries[row] = query_ptr.add((row_base + row) * query_cols + head * dim);
                }
                // K pass:一次扫描同时累计块内各行的 score。块内全可见区间
                // (滑窗下占绝对多数)走行配对联合点积,水平归约减半。
                let mut maximum = [f32::NEG_INFINITY; ROW_CHUNK];
                let all_visible_start = starts[..row_count].iter().copied().max().unwrap_or(span_start);
                let all_visible_end = ends[..row_count].iter().copied().min().unwrap_or(span_start);
                let mut segment = 0usize;
                let mut segment_start = 0usize;
                for row_absolute in span_start..span_end {
                    while row_absolute >= segment_start + segments[segment].0.rows {
                        segment_start += segments[segment].0.rows;
                        segment += 1;
                    }
                    let key = segments[segment].0.data.as_ptr().add((row_absolute - segment_start) * kv_cols + kv_head * dim);
                    let offset = row_absolute - span_start;
                    if row_absolute >= all_visible_start && row_absolute < all_visible_end {
                        let mut row = 0usize;
                        while row + 1 < row_count {
                            let (left, right) = dot_pair_avx512(queries[row], queries[row + 1], key, dim);
                            let left = left * spec.score_scale;
                            let right = right * spec.score_scale;
                            maximum[row] = maximum[row].max(left);
                            maximum[row + 1] = maximum[row + 1].max(right);
                            scores[row * stride + offset] = left;
                            scores[(row + 1) * stride + offset] = right;
                            row += 2;
                        }
                        if row < row_count {
                            let score = dot_avx512(queries[row], key, dim) * spec.score_scale;
                            maximum[row] = maximum[row].max(score);
                            scores[row * stride + offset] = score;
                        }
                        continue;
                    }
                    // 因果端点分叉的尾部:逐行判断归属。
                    for row in 0..row_count {
                        if row_absolute < starts[row] || row_absolute >= ends[row] {
                            continue;
                        }
                        let score = dot_avx512(queries[row], key, dim) * spec.score_scale;
                        maximum[row] = maximum[row].max(score);
                        scores[row * stride + (row_absolute - span_start)] = score;
                    }
                }
                // softmax 每行独立:只读取该行可见区间内本次写入的 score;
                // 16 路 vector exp,分母按向量树归约。
                let mut inverse = [0.0_f32; ROW_CHUNK];
                for row in 0..row_count {
                    let offset = row * stride + (starts[row] - span_start);
                    let row_scores = &mut scores[offset..offset + (ends[row] - starts[row])];
                    let row_maximum = maximum[row];
                    let maximum = _mm512_set1_ps(row_maximum);
                    let mut sums = _mm512_setzero_ps();
                    let full = row_scores.len() / 16 * 16;
                    let (body, tail) = row_scores.split_at_mut(full);
                    for chunk in body.chunks_exact_mut(16) {
                        let value = _mm512_sub_ps(_mm512_loadu_ps(chunk.as_ptr()), maximum);
                        let value = exp16_avx512(value);
                        _mm512_storeu_ps(chunk.as_mut_ptr(), value);
                        sums = _mm512_add_ps(sums, value);
                    }
                    let mut denominator = _mm512_reduce_add_ps(sums);
                    for score in tail.iter_mut() {
                        *score = (*score - row_maximum).exp();
                        denominator += *score;
                    }
                    inverse[row] = denominator.recip();
                }
                // V pass 按输出列分块:8 行 × 2 个 zmm 累加器跨完整 history
                // 常驻寄存器。同一 V 向量只加载一次服务块内全部 query 行，输出
                // 最后只写一次；旧的 token-major 外积会为每个 history token 对
                // 输出做一次 RMW，并把同一 V 向量重复加载 row_count 次。
                for column in (0..dim).step_by(32) {
                    let paired = column + 32 <= dim;
                    let mut output0 = [_mm512_setzero_ps(); ROW_CHUNK];
                    let mut output1 = [_mm512_setzero_ps(); ROW_CHUNK];
                    let mut segment = 0usize;
                    let mut segment_start = 0usize;
                    for row_absolute in span_start..span_end {
                        while row_absolute >= segment_start + segments[segment].1.rows {
                            segment_start += segments[segment].1.rows;
                            segment += 1;
                        }
                        let value = segments[segment].1.data.as_ptr().add((row_absolute - segment_start) * kv_cols + kv_head * dim + column);
                        let value0 = _mm512_loadu_ps(value);
                        let value1 = paired.then(|| _mm512_loadu_ps(value.add(16)));
                        for row in 0..row_count {
                            if row_absolute < starts[row] || row_absolute >= ends[row] {
                                continue;
                            }
                            let weight = _mm512_set1_ps(scores[row * stride + (row_absolute - span_start)] * inverse[row]);
                            output0[row] = _mm512_fmadd_ps(weight, value0, output0[row]);
                            if let Some(value1) = value1 {
                                output1[row] = _mm512_fmadd_ps(weight, value1, output1[row]);
                            }
                        }
                    }
                    for row in 0..row_count {
                        let output = output_ptr.add((row_base + row) * query_cols + head * dim + column);
                        _mm512_storeu_ps(output, output0[row]);
                        if paired {
                            _mm512_storeu_ps(output.add(16), output1[row]);
                        }
                    }
                }
            }
        });
    });
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn dot_avx512(left: *const f32, right: *const f32, len: usize) -> f32 {
    let mut even = _mm512_setzero_ps();
    let mut odd = _mm512_setzero_ps();
    let mut column = 0usize;
    while column + 32 <= len {
        let left0 = unsafe { _mm512_loadu_ps(left.add(column)) };
        let right0 = unsafe { _mm512_loadu_ps(right.add(column)) };
        even = _mm512_fmadd_ps(left0, right0, even);
        let left1 = unsafe { _mm512_loadu_ps(left.add(column + 16)) };
        let right1 = unsafe { _mm512_loadu_ps(right.add(column + 16)) };
        odd = _mm512_fmadd_ps(left1, right1, odd);
        column += 32;
    }
    if column < len {
        let left = unsafe { _mm512_loadu_ps(left.add(column)) };
        let right = unsafe { _mm512_loadu_ps(right.add(column)) };
        even = _mm512_fmadd_ps(left, right, even);
    }
    _mm512_reduce_add_ps(_mm512_add_ps(even, odd))
}

/// 两行共享同一 K 行的联合点积:K 只读一遍,水平归约经组间折叠 +
/// 128 位组内水平加完成,替代逐 reduce_add 的长指令序列。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn horizontal512(value: __m512) -> f32 {
    // 组间折叠:每个 128 位组变为跨 4 组的逐 lane 和。
    let halved = _mm512_add_ps(value, _mm512_shuffle_f32x4(value, value, 0xEE));
    let folded = _mm512_add_ps(halved, _mm512_shuffle_f32x4(halved, halved, 0x55));
    // 低 128 位 [S0..S3] 即各 lane 跨组之和,再做组内水平加。
    let low = _mm512_castps512_ps128(folded);
    let pair = _mm_add_ps(low, _mm_shuffle_ps(low, low, 0x4E));
    _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0xB1)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn dot_pair_avx512(left: *const f32, right: *const f32, key: *const f32, len: usize) -> (f32, f32) {
    unsafe {
        let mut acc_left = _mm512_setzero_ps();
        let mut acc_right = _mm512_setzero_ps();
        for column in (0..len).step_by(16) {
            let key = _mm512_loadu_ps(key.add(column));
            acc_left = _mm512_fmadd_ps(_mm512_loadu_ps(left.add(column)), key, acc_left);
            acc_right = _mm512_fmadd_ps(_mm512_loadu_ps(right.add(column)), key, acc_right);
        }
        (horizontal512(acc_left), horizontal512(acc_right))
    }
}

/// softmax 用的 16 路 vector exp:n = round(x·log2e),r = x − n·ln2,
/// 六阶泰勒多项式 + scalef 还原。|r| ≤ ln2/2,相对误差约 1e-7,
/// 对 oracle 2e-5 阈值与 argmax 语义均无影响。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn exp16_avx512(values: __m512) -> __m512 {
    let log2e = _mm512_set1_ps(1.4426950408889634);
    let ln2 = _mm512_set1_ps(0.6931471805599453);
    let n = _mm512_roundscale_ps(_mm512_mul_ps(values, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let r = _mm512_fnmadd_ps(n, ln2, values);
    // exp(r) = 1 + r·(1 + r·(1/2 + r·(1/6 + r·(1/24 + r·(1/120 + r/720))))) Horner,
    // 注意两个常数 1:内层闭合多项式与外层 1。
    let one = _mm512_set1_ps(1.0);
    let mut poly = _mm512_set1_ps(1.0 / 720.0);
    poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(1.0 / 120.0));
    poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(1.0 / 24.0));
    poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(1.0 / 6.0));
    poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(0.5));
    poly = _mm512_fmadd_ps(poly, r, one);
    poly = _mm512_fmadd_ps(poly, r, one);
    _mm512_mul_ps(_mm512_scalef_ps(one, n), poly)
}

impl GqaPrefillBackend for CpuContext {
    fn rmsnorm_heads(&self, input: &CpuTensor, weight: &CpuWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<CpuTensor, BackendError> {
        rmsnorm_heads(input, weight, head_count, head_dim, eps, 0.0, "RMSNorm")
    }

    fn gemma_rmsnorm_heads(&self, input: &CpuTensor, weight: &CpuWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<CpuTensor, BackendError> {
        rmsnorm_heads(input, weight, head_count, head_dim, eps, 1.0, "GemmaRMSNorm")
    }

    fn gqa_prefill_attention(&self, query: &CpuTensor, key: &CpuTensor, value: &CpuTensor, spec: &GqaSpec) -> Result<CpuTensor, BackendError> {
        let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or_else(|| compute("GQA query 维度溢出"))?;
        let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute("GQA KV 维度溢出"))?;
        if query.rows != key.rows || query.rows != value.rows || query.cols != query_cols || key.cols != kv_cols || value.cols != kv_cols {
            return Err(compute(format!("GQA prefill shape 异常: Q=[{},{}] K=[{},{}] V=[{},{}]", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols)));
        }
        let mut output = CpuTensor { data: vec![0.0; query.rows * query_cols], rows: query.rows, cols: query_cols };
        gqa_prefill_attention(&query.data, &key.data, &value.data, spec, &mut output.data);
        Ok(output)
    }

    fn gqa_prefill_attention_cached(&self, cache: &mut CpuKvCache, layer: usize, position: usize, query: &CpuTensor, key: &CpuTensor, value: &CpuTensor, spec: &GqaSpec, retain_full_cache: bool) -> Result<CpuTensor, BackendError> {
        let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or_else(|| compute("GQA query 维度溢出"))?;
        let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute("GQA KV 维度溢出"))?;
        if query.rows != key.rows || query.rows != value.rows || query.cols != query_cols || key.cols != kv_cols || value.cols != kv_cols {
            return Err(compute(format!("GQA cached prefill shape 异常: Q=[{},{}] K=[{},{}] V=[{},{}]", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols)));
        }
        if position == 0 && query.rows > 1 && !matches!(spec.window, crate::attention::gqa::CausalWindow::Full) {
            let mut output = CpuTensor { data: vec![0.0; query.rows * query_cols], rows: query.rows, cols: query_cols };
            gqa_prefill_attention(&query.data, &key.data, &value.data, spec, &mut output.data);
            cache.append_gqa(layer, position, key, value, spec, retain_full_cache)?;
            return Ok(output);
        }
        if query.rows > 1 && !matches!(spec.window, crate::attention::gqa::CausalWindow::Full) {
            return Err(compute("滑窗 GQA 暂不支持从非零 position 追加多 token prefill"));
        }
        cache.append_gqa(layer, position, key, value, spec, retain_full_cache)?;
        let cached = cache.gqa_layer(layer)?;
        let mut output = CpuTensor { data: vec![0.0; query.rows * query_cols], rows: query.rows, cols: query_cols };
        gqa_prefill_attention_at(&query.data, &cached.key, &cached.value, cached.start, cached.rows, position, spec, &mut output.data);
        Ok(output)
    }

    fn gqa_prefill_attention_cached_visible(
        &self,
        cache: &mut CpuKvCache,
        layer: usize,
        position: usize,
        query: &CpuTensor,
        key: &CpuTensor,
        value: &CpuTensor,
        spec: &GqaSpec,
        visible_ends: &[u32],
        retain_full_cache: bool,
    ) -> Result<CpuTensor, BackendError> {
        if visible_ends.len() != query.rows {
            return Err(compute(format!("GQA visible_ends={}，期望 {}", visible_ends.len(), query.rows)));
        }
        cache.append_gqa(layer, position, key, value, spec, retain_full_cache)?;
        let cached = cache.gqa_layer(layer)?;
        let mut output = CpuTensor { data: vec![0.0; query.rows * query.cols], rows: query.rows, cols: query.cols };
        gqa_prefill_attention_at_visible(&query.data, &cached.key, &cached.value, cached.start, cached.rows, position, spec, Some(visible_ends), &mut output.data);
        Ok(output)
    }

    fn gqa_prefill_attention_cached_from(&self, cache: &CpuKvCache, source_layer: usize, position: usize, query: &CpuTensor, spec: &GqaSpec) -> Result<CpuTensor, BackendError> {
        let query_cols = spec.num_heads.checked_mul(spec.head_dim).ok_or_else(|| compute("GQA query 维度溢出"))?;
        let kv_cols = spec.num_kv_heads.checked_mul(spec.head_dim).ok_or_else(|| compute("GQA KV 维度溢出"))?;
        let cached = cache.gqa_layer(source_layer)?;
        let required_rows = position.checked_add(query.rows).ok_or_else(|| compute("GQA shared cache position 溢出"))?;
        if query.cols != query_cols || cached.cols != kv_cols || cached.rows < required_rows {
            return Err(compute(format!("GQA shared cache L{source_layer} 不完整: Q=[{},{}], cache=[{},{}], required_rows={required_rows}", query.rows, query.cols, cached.rows, cached.cols,)));
        }
        let mut output = CpuTensor { data: vec![0.0; query.rows * query_cols], rows: query.rows, cols: query_cols };
        gqa_prefill_attention_at(&query.data, &cached.key, &cached.value, cached.start, cached.rows, position, spec, &mut output.data);
        Ok(output)
    }

    fn gqa_prefill_attention_cached_from_visible(&self, cache: &CpuKvCache, source_layer: usize, position: usize, query: &CpuTensor, spec: &GqaSpec, visible_ends: &[u32]) -> Result<CpuTensor, BackendError> {
        let cached = cache.gqa_layer(source_layer)?;
        if visible_ends.len() != query.rows {
            return Err(compute(format!("GQA visible_ends={}，期望 {}", visible_ends.len(), query.rows)));
        }
        let mut output = CpuTensor { data: vec![0.0; query.rows * query.cols], rows: query.rows, cols: query.cols };
        gqa_prefill_attention_at_visible(&query.data, &cached.key, &cached.value, cached.start, cached.rows, position, spec, Some(visible_ends), &mut output.data);
        Ok(output)
    }
}

fn rmsnorm_heads(input: &CpuTensor, weight: &CpuWeight, head_count: usize, head_dim: usize, eps: f32, weight_offset: f32, name: &str) -> Result<CpuTensor, BackendError> {
    let expected_columns = head_count.checked_mul(head_dim).ok_or_else(|| compute("GQA head 维度溢出"))?;
    if head_count == 0 || head_dim == 0 || input.cols != expected_columns || weight.data.len() != head_dim {
        return Err(compute(format!("GQA head {name} shape 非法: input=[{},{}] weight={} heads={head_count} dim={head_dim}", input.rows, input.cols, weight.data.len())));
    }
    let data = crate::kernel::cpu::vae::rmsnorm_heads_with(&input.data, Some(&weight.data), head_count, head_dim, eps, weight_offset).map_err(compute)?;
    Ok(CpuTensor { data, rows: input.rows, cols: input.cols })
}

#[derive(Default)]
pub(crate) struct CpuMlaLayer {
    pub(crate) latent: Vec<f32>,
    pub(crate) rope: Vec<f32>,
    pub(crate) rows: usize,
    pub(crate) latent_cols: usize,
    pub(crate) rope_cols: usize,
}

#[derive(Default)]
pub(crate) struct CpuGqaLayer {
    pub(crate) key: Vec<f32>,
    pub(crate) value: Vec<f32>,
    pub(crate) start: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

pub struct CpuKvCache {
    pub(crate) layers: Vec<CpuMlaLayer>,
    pub(crate) gqa_layers: Vec<CpuGqaLayer>,
}

impl CpuKvCache {
    pub fn new(layer_count: usize) -> Self {
        Self { layers: (0..layer_count).map(|_| CpuMlaLayer::default()).collect(), gqa_layers: (0..layer_count).map(|_| CpuGqaLayer::default()).collect() }
    }

    pub(crate) fn append(&mut self, layer: usize, latent: &CpuTensor, rope: &CpuTensor) -> Result<(), BackendError> {
        if latent.rows != rope.rows {
            return Err(compute(format!("L{layer} MLA cache 行数不一致: latent={} rope={}", latent.rows, rope.rows)));
        }
        let cached = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if cached.rows == 0 {
            cached.latent_cols = latent.cols;
            cached.rope_cols = rope.cols;
        } else if cached.latent_cols != latent.cols || cached.rope_cols != rope.cols {
            return Err(compute(format!("L{layer} MLA cache 列数不一致: latent={}/{} rope={}/{}", latent.cols, cached.latent_cols, rope.cols, cached.rope_cols,)));
        }
        cached.latent.extend_from_slice(&latent.data);
        cached.rope.extend_from_slice(&rope.data);
        cached.rows += latent.rows;
        Ok(())
    }

    pub(crate) fn append_gqa(&mut self, layer: usize, position: usize, key: &CpuTensor, value: &CpuTensor, spec: &GqaSpec, retain_full_cache: bool) -> Result<(), BackendError> {
        if key.rows != value.rows || key.cols != value.cols {
            return Err(compute(format!("L{layer} GQA cache K=[{},{}] V=[{},{}] 不一致", key.rows, key.cols, value.rows, value.cols,)));
        }
        let cached = self.gqa_layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if cached.rows != position {
            return Err(compute(format!("L{layer} GQA cache position={position}，当前 rows={}", cached.rows)));
        }
        if cached.rows == 0 {
            cached.cols = key.cols;
        } else if cached.cols != key.cols {
            return Err(compute(format!("L{layer} GQA cache cols={}，输入 {}", cached.cols, key.cols)));
        }
        cached.key.extend_from_slice(&key.data);
        cached.value.extend_from_slice(&value.data);
        cached.rows = position.checked_add(key.rows).ok_or_else(|| compute("GQA cache position 溢出"))?;
        if !retain_full_cache && let crate::attention::gqa::CausalWindow::Sliding { size } = spec.window {
            let retained_rows = cached.key.len() / cached.cols;
            if retained_rows > size {
                let drop_rows = retained_rows - size;
                let drop_elements = drop_rows * cached.cols;
                cached.key.drain(..drop_elements);
                cached.value.drain(..drop_elements);
            }
            cached.start = cached.rows - cached.key.len() / cached.cols;
        }
        Ok(())
    }

    pub(crate) fn gqa_layer(&self, layer: usize) -> Result<&CpuGqaLayer, BackendError> {
        self.gqa_layers.get(layer).ok_or(BackendError::UnsupportedLayer { layer })
    }
}

pub struct CpuDsaState {
    keys: Vec<Vec<f32>>,
    /// kpool 池化 gate 缓存(每层 [tokens, head_dim]);kpool=0 时保持为空。
    gates: Vec<Vec<f32>>,
    /// 每层的池内 APE 权重 [kpool, head_dim];kpool=0 时保持为空。
    kpool_apes: Vec<Vec<f32>>,
    lengths: Vec<usize>,
    capacity: usize,
    head_dim: usize,
    top_k: usize,
    selection: Vec<usize>,
    selection_valid: bool,
    selection_rows: usize,
}

impl CpuDsaState {
    pub fn new(layer_count: usize, capacity: usize, head_dim: usize, top_k: usize) -> Result<Self, String> {
        if layer_count == 0 || capacity == 0 || head_dim == 0 || top_k == 0 || top_k > capacity {
            return Err(format!("DSA state 参数非法: layers={layer_count} capacity={capacity} head_dim={head_dim} top_k={top_k}"));
        }
        Ok(Self {
            keys: (0..layer_count).map(|_| Vec::new()).collect(),
            gates: (0..layer_count).map(|_| Vec::new()).collect(),
            kpool_apes: vec![Vec::new(); layer_count],
            lengths: vec![0; layer_count],
            capacity,
            head_dim,
            top_k,
            selection: Vec::new(),
            selection_valid: false,
            selection_rows: 0,
        })
    }

    /// 每行选择的槽位宽度:kpool 尾池直选时为 top_k + kpool - 1,否则 top_k。
    fn selection_width(&self, spec: &DsaSpec) -> usize {
        if spec.kpool > 0 && spec.always_select_tail { self.top_k + spec.kpool - 1 } else { self.top_k }
    }

    /// 注入某层的池内 APE 权重(平铺 [kpool * head_dim])。
    /// kpool 选择依赖它;装配时按层设置,select 前必须完成。
    pub fn set_kpool_ape(&mut self, layer: usize, ape: Vec<f32>) -> Result<(), String> {
        if layer >= self.lengths.len() {
            return Err(format!("kpool APE layer {layer} 越界于 {}", self.lengths.len()));
        }
        self.kpool_apes[layer] = ape;
        Ok(())
    }

    pub fn selection(&self) -> Option<&[usize]> {
        self.selection_valid.then_some(&self.selection)
    }

    pub(crate) fn can_append(&mut self, layer: usize, position: usize, spec: &DsaSpec) -> bool {
        self.selection_valid = false;
        self.selection_rows = 0;
        spec.head_dim == self.head_dim && spec.top_k == self.top_k && position < self.capacity && self.lengths.get(layer).copied() == Some(position)
    }

    pub(crate) fn append(&mut self, layer: usize, position: usize, keys: &CpuTensor) -> Result<(), BackendError> {
        if keys.cols != self.head_dim || position.checked_add(keys.rows).is_none_or(|end| end > self.capacity) {
            return Err(compute(format!("L{layer} DSA key=[{},{}] position={position}，capacity={} head_dim={}", keys.rows, keys.cols, self.capacity, self.head_dim,)));
        }
        let length = self.lengths.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if *length != position {
            return Err(compute(format!("L{layer} DSA position={position}，当前 length={length}")));
        }
        self.keys[layer].extend_from_slice(&keys.data);
        *length += keys.rows;
        Ok(())
    }

    /// kpool 打包追加:key 与 gate 同长度同步落盘,select 时二者配对池化。
    pub(crate) fn append_gated(&mut self, layer: usize, position: usize, keys: &CpuTensor, gate: &CpuTensor) -> Result<(), BackendError> {
        if gate.rows != keys.rows || gate.cols != self.head_dim {
            return Err(compute(format!("L{layer} kpool gate=[{},{}] 与 key=[{},{}] 不一致(head_dim={})", gate.rows, gate.cols, keys.rows, keys.cols, self.head_dim)));
        }
        if !self.gates[layer].is_empty() || !self.keys[layer].is_empty() {
            // append 已保证 keys 与 position 对齐;gate 只需同长追加。
            let expected = self.keys[layer].len();
            if self.gates[layer].len() != expected {
                return Err(compute(format!("L{layer} kpool gate 长度 {} 与 key 长度 {expected} 失配", self.gates[layer].len())));
            }
        }
        self.gates[layer].extend_from_slice(&gate.data);
        self.append(layer, position, keys)
    }

    /// kpool>0 时从打包状态池化并按池选择;返回 None 表示池数不足,走全量注意力。
    fn select_kpool(&mut self, layer: usize, query: &CpuTensor, head_weights: &CpuTensor, spec: &DsaSpec, query_row: usize, count: usize) -> Result<Option<Vec<usize>>, BackendError> {
        let rows = self.lengths.get(layer).copied().ok_or(BackendError::UnsupportedLayer { layer })?;
        if self.gates[layer].len() != rows * self.head_dim {
            return Err(compute(format!("L{layer} kpool gate 缓存长度 {} 与 rows={rows} 不匹配", self.gates[layer].len())));
        }
        let ape = self.kpool_apes.get(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if ape.len() != spec.kpool * spec.head_dim {
            return Err(compute(format!("L{layer} kpool APE 长度 {} 期望 {}(set_kpool_ape 未注入?)", ape.len(), spec.kpool * spec.head_dim)));
        }
        // 单流 CPU 无 padding,有效性全真;越界位由 pool_states 内部处理。
        let valid = vec![true; rows];
        let pools = dsa::pool_states(&self.keys[layer], &self.gates[layer], &valid, ape, spec.head_dim, spec.kpool).map_err(compute)?;
        if pools.pool_count() == 0 {
            return Ok(None);
        }
        let selected = dsa::select_rows_kpool(&pools, &query.data, query.rows, &head_weights.data, spec, &valid, query_row, count).map_err(compute)?;
        Ok(Some(selected))
    }

    pub(crate) fn select(&mut self, layer: usize, query: &CpuTensor, head_weights: &CpuTensor, spec: &DsaSpec) -> Result<(), BackendError> {
        let rows = *self.lengths.get(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if rows <= self.top_k {
            self.selection_valid = false;
            self.selection_rows = 0;
            return Ok(());
        }
        let selected = if spec.kpool > 0 {
            match self.select_kpool(layer, query, head_weights, spec, rows - 1, self.top_k)? {
                Some(selected) => selected,
                None => {
                    self.selection_valid = false;
                    self.selection_rows = 0;
                    return Ok(());
                }
            }
        } else {
            dsa::select_rows(&self.keys[layer], &query.data, query.rows, &head_weights.data, spec, rows - 1, self.top_k).map_err(compute)?
        };
        self.selection = selected;
        self.selection_valid = true;
        self.selection_rows = 1;
        Ok(())
    }

    pub(crate) fn select_prefill(&mut self, layer: usize, query: &CpuTensor, head_weights: &CpuTensor, spec: &DsaSpec) -> Result<(), BackendError> {
        let rows = *self.lengths.get(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if rows <= self.top_k || query.rows != rows {
            self.selection_valid = false;
            self.selection_rows = 0;
            return Ok(());
        }
        self.selection.clear();
        self.selection.reserve(rows * self.top_k);
        let width = self.selection_width(spec);
        for row in 0..rows {
            let count = width.min(row + 1);
            let selected = if spec.kpool > 0 {
                match self.select_kpool(layer, query, head_weights, spec, row, count)? {
                    Some(selected) => selected,
                    None => {
                        self.selection_valid = false;
                        self.selection_rows = 0;
                        return Ok(());
                    }
                }
            } else {
                dsa::select_rows(&self.keys[layer], &query.data, query.rows, &head_weights.data, spec, row, count).map_err(compute)?
            };
            self.selection.extend(selected);
            self.selection.resize((row + 1) * width, 0);
        }
        self.selection_valid = true;
        self.selection_rows = rows;
        Ok(())
    }
}

impl MlaPrefillBackend for CpuContext {
    fn mla_prefill_attention(&self, query: &CpuTensor, latent: &CpuTensor, k_rope: &CpuTensor, kv_b: &CpuWeight, cache: Option<&mut CpuKvCache>, layer: usize, spec: &MlaSpec) -> Result<CpuTensor, BackendError> {
        let kv = self.linear(latent, kv_b)?;
        if let Some(cache) = cache {
            cache.append(layer, latent, k_rope)?;
        }
        Ok(crate::kernel::cpu::mla::mla_attention_cpu(spec, query, &kv, k_rope))
    }
}

impl DecodeBackend for CpuContext {
    type DsaState = CpuDsaState;
    fn append_mla(&self, cache: &mut CpuKvCache, layer: usize, latent: &CpuTensor, rope: &CpuTensor) -> Result<(), BackendError> {
        cache.append(layer, latent, rope)
    }

    fn mla_decode_attention(&self, query: &CpuTensor, cache: &CpuKvCache, kv_b: &CpuWeight, layer: usize, position: usize, spec: &MlaSpec) -> Result<CpuTensor, BackendError> {
        mla_attention(self, query, cache, kv_b, layer, position, spec, None)
    }

    fn dsa_can_append(&self, state: &mut CpuDsaState, layer: usize, position: usize, spec: &DsaSpec) -> bool {
        state.can_append(layer, position, spec)
    }

    fn append_dsa_keys(&self, state: &mut CpuDsaState, layer: usize, position: usize, keys: &CpuTensor, _spec: &DsaSpec) -> Result<(), BackendError> {
        state.append(layer, position, keys)
    }

    fn append_dsa_keys_gated(&self, state: &mut CpuDsaState, layer: usize, position: usize, keys: &CpuTensor, gate: &CpuTensor, _spec: &DsaSpec) -> Result<(), BackendError> {
        state.append_gated(layer, position, keys, gate)
    }

    fn dsa_select_topk(&self, state: &mut CpuDsaState, layer: usize, query: &CpuTensor, head_weights: &CpuTensor, spec: &DsaSpec) -> Result<(), BackendError> {
        state.select(layer, query, head_weights, spec)
    }

    fn mla_decode_attention_selected(&self, query: &CpuTensor, cache: &CpuKvCache, kv_b: &CpuWeight, layer: usize, position: usize, mla: &MlaSpec, dsa: &DsaSpec, state: &CpuDsaState) -> Result<CpuTensor, BackendError> {
        let selection = if position > dsa.top_k { state.selection() } else { None };
        mla_attention(self, query, cache, kv_b, layer, position, mla, selection)
    }
}

impl DsaPrefillBackend for CpuContext {
    fn dsa_select_prefill(&self, state: &mut CpuDsaState, layer: usize, query: &CpuTensor, head_weights: &CpuTensor, spec: &DsaSpec) -> Result<(), BackendError> {
        state.select_prefill(layer, query, head_weights, spec)
    }

    fn mla_prefill_attention_selected(
        &self,
        query: &CpuTensor,
        latent: &CpuTensor,
        k_rope: &CpuTensor,
        kv_b: &CpuWeight,
        cache: Option<&mut CpuKvCache>,
        layer: usize,
        spec: &MlaSpec,
        dsa: &DsaSpec,
        state: &CpuDsaState,
    ) -> Result<CpuTensor, BackendError> {
        if !state.selection_valid || state.selection_rows != query.rows {
            return self.mla_prefill_attention(query, latent, k_rope, kv_b, cache, layer, spec);
        }
        let projected = self.linear(latent, kv_b)?;
        if let Some(cache) = cache {
            cache.append(layer, latent, k_rope)?;
        }
        let q_head_dim = spec.q_head_dim();
        let qk_nope_dim = spec.qk_nope_dim();
        let kv_head_dim = spec.kv_head_dim();
        let value_dim = spec.value_dim();
        let output_cols = spec.num_heads * value_dim;
        let scale = 1.0 / (q_head_dim as f32).sqrt();
        let mut output = CpuTensor { data: vec![0.0; query.rows * output_cols], rows: query.rows, cols: output_cols };
        let width = state.selection_width(dsa);
        if let Some(output) = crate::kernel::cpu::mla::mla_attention_blas(spec, query, &projected, k_rope, Some((state.selection.as_slice(), width))) {
            return Ok(output);
        }
        output.data.par_chunks_mut(output_cols).enumerate().for_each(|(row, output_row)| {
            let count = width.min(row + 1);
            let selected = &state.selection[row * width..row * width + count];
            let mut scores = vec![0.0_f32; count];
            for head in 0..spec.num_heads {
                let query_base = row * query.cols + head * q_head_dim;
                let mut maximum = f32::NEG_INFINITY;
                for (slot, &token) in selected.iter().enumerate() {
                    let kv_base = token * projected.cols + head * kv_head_dim;
                    let rope_base = token * k_rope.cols;
                    let score = (dot(&query.data[query_base..query_base + qk_nope_dim], &projected.data[kv_base..kv_base + qk_nope_dim])
                        + dot(&query.data[query_base + qk_nope_dim..query_base + q_head_dim], &k_rope.data[rope_base..rope_base + k_rope.cols]))
                        * scale;
                    maximum = maximum.max(score);
                    scores[slot] = score;
                }
                let mut denominator = 0.0_f32;
                for score in &mut scores {
                    *score = (*score - maximum).exp();
                    denominator += *score;
                }
                let output_base = head * value_dim;
                for (slot, &token) in selected.iter().enumerate() {
                    let weight = scores[slot] / denominator;
                    let value_base = token * projected.cols + head * kv_head_dim + qk_nope_dim;
                    for value in 0..value_dim {
                        output_row[output_base + value] += weight * projected.data[value_base + value];
                    }
                }
            }
        });
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn mla_attention(backend: &CpuContext, query: &CpuTensor, cache: &CpuKvCache, kv_b: &CpuWeight, layer: usize, position: usize, spec: &MlaSpec, selection: Option<&[usize]>) -> Result<CpuTensor, BackendError> {
    if query.rows != 1 || query.cols != spec.q_projection_size {
        return Err(compute(format!("L{layer} CPU MLA query=[{},{}]，期望 [1,{}]", query.rows, query.cols, spec.q_projection_size)));
    }
    let cached = cache.layers.get(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
    if position == 0 || position > cached.rows {
        return Err(compute(format!("L{layer} CPU MLA position={position}，cache rows={}", cached.rows)));
    }
    if cached.latent_cols != spec.kv_lora_rank || cached.rope_cols != spec.qk_rope_head_dim {
        return Err(compute(format!("L{layer} CPU MLA cache shape latent={} rope={}，期望 {}/{}", cached.latent_cols, cached.rope_cols, spec.kv_lora_rank, spec.qk_rope_head_dim)));
    }
    let indices: Vec<usize> = match selection {
        Some(selection) if !selection.is_empty() => selection.iter().copied().filter(|index| *index < position).collect(),
        _ => (0..position).collect(),
    };
    if indices.is_empty() {
        return Err(compute(format!("L{layer} CPU MLA selection 为空")));
    }
    let mut latent = CpuTensor { data: Vec::with_capacity(indices.len() * cached.latent_cols), rows: indices.len(), cols: cached.latent_cols };
    for &index in &indices {
        latent.data.extend_from_slice(&cached.latent[index * cached.latent_cols..(index + 1) * cached.latent_cols]);
    }
    let projected = backend.linear(&latent, kv_b)?;
    if projected.cols != spec.kv_projection_size {
        return Err(compute(format!("L{layer} CPU MLA kv_b 输出 {}，期望 {}", projected.cols, spec.kv_projection_size)));
    }

    let data = crate::attention::mla::reference_attention(&query.data, &projected.data, &cached.rope, cached.rope_cols, &indices, spec).map_err(compute)?;
    Ok(CpuTensor { data, rows: 1, cols: query.cols })
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::{attention::block::BlockAttentionSpec, attention::gqa::GqaGeometry};

    fn avx512_ready() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma")
    }

    fn tensor(rows: usize, cols: usize, wave: f32, scale: f32) -> CpuTensor {
        CpuTensor { data: (0..rows * cols).map(|index| ((index as f32) * wave).sin() * scale).collect(), rows, cols }
    }

    /// 按 DSpark 语义构造 batch:每 session 一段 history + 一段 noise K/V,
    /// query 行按 session-major 排列,可见区间为滑窗起点 + 因果递增终点。
    fn batched_case(sessions: usize, block_rows: usize, history: usize, window: usize, heads: usize, kv_heads: usize, scale: f32) -> (CpuTensor, Vec<CpuTensor>, Vec<CpuTensor>, BlockAttentionSpec) {
        let dim = 32;
        let query_cols = heads * dim;
        let kv_cols = kv_heads * dim;
        let query = tensor(sessions * block_rows, query_cols, 0.019, scale);
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut visible = Vec::with_capacity(sessions * block_rows);
        for session in 0..sessions {
            keys.push(tensor(history, kv_cols, 0.013 + session as f32 * 0.001, 1.0));
            values.push(tensor(history, kv_cols, 0.017 + session as f32 * 0.001, 1.0));
            keys.push(tensor(block_rows, kv_cols, 0.023, 1.0));
            values.push(tensor(block_rows, kv_cols, 0.029, 1.0));
            let base = session * (history + block_rows);
            let start = base + history.saturating_sub(window);
            for row in 0..block_rows {
                visible.push(start..base + history + row + 1);
            }
        }
        let spec = BlockAttentionSpec { geometry: GqaGeometry { num_heads: heads, num_kv_heads: kv_heads, head_dim: dim }, score_scale: 1.0 / (dim as f32).sqrt(), visible };
        (query, keys, values, spec)
    }

    /// 拼接 K/V 调 reference,与 segments 实现逐元素对照(NaN 视为相等)。
    fn compare_segments_with_reference(query: &CpuTensor, keys: &[CpuTensor], values: &[CpuTensor], spec: &BlockAttentionSpec, tolerance: f32) {
        let kv_cols = spec.geometry.kv_columns().unwrap();
        let mut key = Vec::new();
        let mut value = Vec::new();
        for (key_segment, value_segment) in keys.iter().zip(values) {
            key.extend_from_slice(&key_segment.data);
            value.extend_from_slice(&value_segment.data);
        }
        let kv_rows = key.len() / kv_cols;
        let expected = crate::attention::block::attention_f32(&query.data, &key, &value, query.rows, kv_rows, spec).unwrap();
        let key_refs = keys.iter().collect::<Vec<_>>();
        let value_refs = values.iter().collect::<Vec<_>>();
        let actual = CpuContext.block_attention_segments(query, &key_refs, &value_refs, spec).unwrap();
        for (index, (&actual, &expected)) in actual.data.iter().zip(&expected).enumerate() {
            if actual.is_nan() || expected.is_nan() {
                assert!(actual.is_nan() && expected.is_nan(), "index={index}: NaN 传播不一致 {actual} vs {expected}");
                continue;
            }
            assert!((actual - expected).abs() < tolerance * (1.0 + expected.abs()), "index={index}: {actual} != {expected}");
        }
    }

    #[test]
    fn avx512_block_attention_matches_reference() {
        if !avx512_ready() {
            return;
        }
        let query_rows = 3;
        let kv_rows = 7;
        let heads = 2;
        let dim = 64;
        let cols = heads * dim;
        let query = CpuTensor { data: (0..query_rows * cols).map(|index| (index as f32 * 0.017).sin()).collect(), rows: query_rows, cols };
        let key = CpuTensor { data: (0..kv_rows * cols).map(|index| (index as f32 * 0.013).cos()).collect(), rows: kv_rows, cols };
        let value = CpuTensor { data: (0..kv_rows * cols).map(|index| (index as f32 * 0.019).sin()).collect(), rows: kv_rows, cols };
        let spec = BlockAttentionSpec { geometry: GqaGeometry { num_heads: heads, num_kv_heads: heads, head_dim: dim }, score_scale: 1.0 / (dim as f32).sqrt(), visible: vec![0..3, 1..6, 2..7] };
        let expected = crate::attention::block::attention_f32(&query.data, &key.data, &value.data, query_rows, kv_rows, &spec).unwrap();
        let actual = CpuContext.block_attention(&query, &key, &value, &spec).unwrap();
        for (index, (&actual, &expected)) in actual.data.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() < 2.0e-5, "index={index}: {actual} != {expected}");
        }
    }

    #[test]
    fn fused_attention_matches_reference_across_batch_shapes() {
        if !avx512_ready() {
            return;
        }
        for &(sessions, block_rows, history, window, heads, kv_heads) in &[(1_usize, 8, 24, 16, 5, 5), (2, 8, 24, 16, 5, 5), (4, 8, 24, 16, 5, 5), (3, 5, 17, 12, 5, 5), (5, 3, 11, 8, 5, 5), (2, 8, 24, 16, 6, 3)] {
            // rows 非 8 倍数的用例让 8 行块跨 session 边界;6/3 组合覆盖 GQA 共享 kv head。
            let (query, keys, values, spec) = batched_case(sessions, block_rows, history, window, heads, kv_heads, 1.0);
            compare_segments_with_reference(&query, &keys, &values, &spec, 2.0e-5);
        }
    }

    #[test]
    fn fused_attention_handles_zero_and_extreme_inputs() {
        if !avx512_ready() {
            return;
        }
        // 全零输入:softmax 均匀,输出与 reference 同为 0。
        for &(sessions, block_rows) in &[(1_usize, 8_usize), (4, 8)] {
            let (query, keys, values, spec) = batched_case(sessions, block_rows, 24, 16, 5, 5, 0.0);
            compare_segments_with_reference(&query, &keys, &values, &spec, 2.0e-5);
        }
        // 极值:大数量级 Q/K 使 softmax 深度饱和,结果仍须与 reference 同阶一致。
        let (query, keys, values, spec) = batched_case(2, 8, 24, 16, 5, 5, 100.0);
        compare_segments_with_reference(&query, &keys, &values, &spec, 2.0e-5);
    }

    #[test]
    fn fused_attention_propagates_non_finite_like_reference() {
        if !avx512_ready() {
            return;
        }
        let (mut query, keys, values, spec) = batched_case(2, 8, 24, 16, 5, 5, 1.0);
        // 第 1 行 query 注入 NaN:该行输出两侧都必须 NaN,其余行仍与 reference 一致。
        let cols = query.cols;
        query.data[cols..2 * cols].fill(f32::NAN);
        compare_segments_with_reference(&query, &keys, &values, &spec, 2.0e-5);
    }
}
