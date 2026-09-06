/// 本模块的 Metal shader(本文件用到的 kernel + 文件私有 helper)。
///
/// 共用 helper 见 [`super::preamble`]。`mod.rs` 的 `kernels_source()`
/// 把 `preamble::SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: gated_delta_conv_f16, gated_delta_conv_prefill_f16, gated_delta_conv_state_f16, gated_delta_prefill_controls_f16, gated_delta_recurrent_prefill_simd8_f16, gated_delta_recurrent_legacy_f16, gated_delta_recurrent_rows_f16, gated_delta_norm_gate_f16
pub const SHADERS: &str = r#"
kernel void gated_delta_conv_f16(
    device const half *qkv [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *state [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    constant uint &kernel_size [[buffer(6)]],
    uint channel [[thread_position_in_grid]])
{
    if (channel >= channels) return;
    const ulong state_base = ulong(channel) * kernel_size;
    const ulong weight_base = ulong(channel) * kernel_size;
    for (uint row = 0; row < rows; ++row) {
        for (uint item = 1; item < kernel_size; ++item) {
            state[state_base + item - 1] = state[state_base + item];
        }
        state[state_base + kernel_size - 1] = float(qkv[ulong(row) * channels + channel]);
        float sum = 0.0f;
        for (uint item = 0; item < kernel_size; ++item) {
            sum += state[state_base + item] * float(weight[weight_base + item]);
        }
        output[ulong(row) * channels + channel] = finite_f16(sum / (1.0f + exp(-sum)));
    }
}
kernel void gated_delta_conv_prefill_f16(
    device const half *qkv [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const float *state [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    constant uint &kernel_size [[buffer(6)]],
    uint item [[thread_position_in_grid]])
{
    if (item >= rows * channels) return;
    const uint row = item / channels;
    const uint channel = item - row * channels;
    const ulong state_base = ulong(channel) * kernel_size;
    const ulong weight_base = ulong(channel) * kernel_size;
    float sum = 0.0f;
    for (uint tap = 0; tap < kernel_size; ++tap) {
        const uint history = row + tap + 1;
        const float value = history < kernel_size
            ? state[state_base + history]
            : float(qkv[ulong(history - kernel_size) * channels + channel]);
        sum += value * float(weight[weight_base + tap]);
    }
    output[ulong(row) * channels + channel] = finite_f16(sum / (1.0f + exp(-sum)));
}
kernel void gated_delta_conv_state_f16(
    device const half *qkv [[buffer(0)]],
    device float *state [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &kernel_size [[buffer(4)]],
    uint channel [[thread_position_in_grid]])
{
    if (channel >= channels) return;
    const ulong state_base = ulong(channel) * kernel_size;
    for (uint tap = 0; tap < kernel_size; ++tap) {
        const uint history = rows + tap;
        state[state_base + tap] = history < kernel_size
            ? state[state_base + history]
            : float(qkv[ulong(history - kernel_size) * channels + channel]);
    }
}
kernel void gated_delta_prefill_controls_f16(
    device const half *mixed [[buffer(0)]],
    device const half *alpha [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device float *controls [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &key_heads [[buffer(6)]],
    constant uint &value_heads [[buffer(7)]],
    constant uint &key_head_dim [[buffer(8)]],
    constant uint &value_head_dim [[buffer(9)]],
    uint index [[thread_position_in_grid]])
{
    const uint heads = max(key_heads, value_heads);
    if (index >= rows * heads) return;
    const uint row = index / heads;
    const uint head = index - row * heads;
    const uint key_dim = key_heads * key_head_dim;
    const uint value_dim = value_heads * value_head_dim;
    const uint conv_dim = key_dim * 2 + value_dim;
    const ulong mixed_row = ulong(row) * conv_dim;
    const ulong control_row = ulong(row) * (key_heads * 2 + value_heads);
    if (head < key_heads) {
        const ulong query_base = mixed_row + ulong(head) * key_head_dim;
        const ulong key_base = mixed_row + key_dim + ulong(head) * key_head_dim;
        float query_sum = 0.0f;
        float key_sum = 0.0f;
        for (uint key_column = 0; key_column < key_head_dim; ++key_column) {
            const float query = float(mixed[query_base + key_column]);
            const float key = float(mixed[key_base + key_column]);
            query_sum += query * query;
            key_sum += key * key;
        }
        controls[control_row + head * 2] = rsqrt(max(query_sum, 1.0e-12f)) * rsqrt(float(value_head_dim));
        controls[control_row + head * 2 + 1] = rsqrt(max(key_sum, 1.0e-12f));
    }
    if (head < value_heads) {
        const float step = float(alpha[ulong(row) * value_heads + head]) + dt_bias[head];
        const float softplus = step > 20.0f ? step : (step < -20.0f ? exp(step) : log(1.0f + exp(step)));
        controls[control_row + key_heads * 2 + head] = exp(-exp(a_log[head]) * softplus);
    }
}
// ============================================================================
// Gated DeltaNet chunked parallel prefill(FLA chunk 算法的 Metal 移植)。
//
// 每 chunk(BT=64)先并行算 WY 表示:下三角 (I+A)^-1 前代、k_cumdecay(=T@(k'βb))
// 与 k_cumsum(=T@(vβ));scan 拆成每 chunk 4 个跨 head batched kernel,chunk 间
// 依赖由 dispatch 顺序承担。与 simd8 串行 kernel 的逐 token 语义严格一致:
//   S_t = d_t·S_{t-1} + k'_tᵀ β_t (v_t − k'_t·d_t·S_{t-1}),o_t = q'_t·S_t
// ============================================================================
constant uint GDN_CHUNK = 64;

// Kernel A:chunk 内变换。一个 threadgroup 处理一个 (chunk, value_head)。
// tg_k 就地变成 w=k'βb、tg_v 变成 u=vβ,输出 f16 scratch 与 chunk 累积 decay。
kernel void gdn_chunk_transform_f16(
    device const half *mixed [[buffer(0)]],
    device const half *beta [[buffer(1)]],
    device const float *controls [[buffer(2)]],
    device half *scratch_decay [[buffer(3)]],
    device half *scratch_cumsum [[buffer(4)]],
    device float *scratch_b [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &key_heads [[buffer(7)]],
    constant uint &value_heads [[buffer(8)]],
    constant uint &key_head_dim [[buffer(9)]],
    constant uint &value_head_dim [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint width = 256;
    if (key_head_dim > 128 || value_head_dim > 128) return;
    const uint chunk = group.x;
    const uint value_head = group.y;
    if (chunk * GDN_CHUNK >= rows || value_head >= value_heads) return;
    const uint key_head = value_head % key_heads;
    const uint key_dim = key_heads * key_head_dim;
    const uint value_dim = value_heads * value_head_dim;
    const uint conv_dim = key_dim * 2 + value_dim;
    const uint chunk_begin = chunk * GDN_CHUNK;
    const uint chunk_rows = min(GDN_CHUNK, rows - chunk_begin);
    const uint control_stride = key_heads * 2 + value_heads;
    const uint kd = key_head_dim;
    const uint vd = value_head_dim;

    // tg 预算 32KB:只驻 T(64×64 f32=16KB) 与 b/row 快照;k/v 直读 global,
    // 单个 chunk 的 k/v 仅 32KB,L1/L2 热驻。
    threadgroup float tg_t[GDN_CHUNK * GDN_CHUNK];
    threadgroup float tg_b[GDN_CHUNK];
    threadgroup float tg_row[GDN_CHUNK];

    if (tid == 0) {
        float product = 1.0f;
        for (uint row = 0; row < GDN_CHUNK; ++row) {
            if (chunk_begin + row < rows) {
                product *= controls[ulong(chunk_begin + row) * control_stride + key_heads * 2 + value_head];
            }
            tg_b[row] = product;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A[i][j] = -β_j·(k'_i·k'_j)·b_i/b_j,严格下三角。
    for (uint index = tid; index < GDN_CHUNK * GDN_CHUNK; index += width) {
        const uint i = index / GDN_CHUNK;
        const uint j = index - i * GDN_CHUNK;
        float value = 0.0f;
        if (i > j && i < chunk_rows && j < chunk_rows) {
            // FLA: A = -(kβ_i·k_j)·b_i/b_j,β 在行(i)侧。
            const float beta_i = 1.0f / (1.0f + exp(-float(beta[ulong(chunk_begin + i) * value_heads + value_head])));
            const ulong key_base_i = ulong(chunk_begin + i) * conv_dim + key_dim + ulong(key_head) * kd;
            const ulong key_base_j = ulong(chunk_begin + j) * conv_dim + key_dim + ulong(key_head) * kd;
            const float key_scale_i = controls[ulong(chunk_begin + i) * control_stride + key_head * 2 + 1];
            const float key_scale_j = controls[ulong(chunk_begin + j) * control_stride + key_head * 2 + 1];
            float dot = 0.0f;
            for (uint column = 0; column < kd; ++column) {
                dot += float(mixed[key_base_i + column]) * key_scale_i * float(mixed[key_base_j + column]) * key_scale_j;
            }
            value = -beta_i * dot * (tg_b[i] / max(tg_b[j], 1.0e-20f));
        }
        tg_t[index] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 前代:new[i][j] = old[i][j] + Σ_m old[i][m]·T[m][j](FLA naive 递推);
    // 行快照避免读写竞态,barrier 由全组到达。
    for (uint i = 1; i < chunk_rows; ++i) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint m = 0; m < i; ++m) tg_row[m] = tg_t[i * GDN_CHUNK + m];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            float j0 = tg_t[i * GDN_CHUNK + tid];
            float j1 = tg_t[i * GDN_CHUNK + tid + 32];
            for (uint m = 0; m < i; ++m) {
                const float row_m = tg_row[m];
                j0 += row_m * tg_t[m * GDN_CHUNK + tid];
                j1 += row_m * tg_t[m * GDN_CHUNK + tid + 32];
            }
            if (tid < i) tg_t[i * GDN_CHUNK + tid] = j0;
            if (tid + 32 < i) tg_t[i * GDN_CHUNK + tid + 32] = j1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < chunk_rows) tg_t[tid * GDN_CHUNK + tid] += 1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // k_cumdecay[t][c] = Σ_{t'} T[t][t']·k'[t'][c]·β_{t'}·b_{t'}
    // k_cumsum[t][c]    = Σ_{t'} T[t][t']·v[t'][c]·β_{t'}
    const ulong decay_base = (ulong(chunk) * value_heads + value_head) * GDN_CHUNK * kd;
    const ulong cumsum_base = (ulong(chunk) * value_heads + value_head) * GDN_CHUNK * vd;
    for (uint index = tid; index < GDN_CHUNK * kd; index += width) {
        const uint t = index / kd;
        const uint column = index - t * kd;
        float sum = 0.0f;
        if (t < chunk_rows) {
            for (uint m = 0; m < chunk_rows; ++m) {
                const ulong key_base_m = ulong(chunk_begin + m) * conv_dim + key_dim + ulong(key_head) * kd;
                const float key_scale_m = controls[ulong(chunk_begin + m) * control_stride + key_head * 2 + 1];
                const float beta_m = 1.0f / (1.0f + exp(-float(beta[ulong(chunk_begin + m) * value_heads + value_head])));
                sum += tg_t[t * GDN_CHUNK + m] * float(mixed[key_base_m + column]) * key_scale_m * beta_m * tg_b[m];
            }
        }
        scratch_decay[decay_base + index] = half(sum);
    }
    for (uint index = tid; index < GDN_CHUNK * vd; index += width) {
        const uint t = index / vd;
        const uint column = index - t * vd;
        float sum = 0.0f;
        if (t < chunk_rows) {
            for (uint m = 0; m < chunk_rows; ++m) {
                const ulong value_base_m = ulong(chunk_begin + m) * conv_dim + key_dim * 2 + ulong(value_head) * vd + column;
                const float beta_m = 1.0f / (1.0f + exp(-float(beta[ulong(chunk_begin + m) * value_heads + value_head])));
                sum += tg_t[t * GDN_CHUNK + m] * float(mixed[value_base_m]) * beta_m;
            }
        }
        scratch_cumsum[cumsum_base + index] = half(sum);
    }
    if (tid < GDN_CHUNK) {
        const ulong b_base = (ulong(chunk) * value_heads + value_head) * GDN_CHUNK;
        scratch_b[b_base + tid] = tg_b[tid];
    }
}

// ============================================================================
// GDN chunked scan 重写(2026-08):跨 head batched simdgroup GEMM,chunk 间依赖
// 由 dispatch 顺序承担(单 kernel 内 barrier 串行链是旧实现两次实测慢于 simd8
// 的根因)。每 chunk 固定 4 步,数学与 simd8 逐 token 语义一致:
//   qk[t][m]    = tril((q'_t·k'_m)·b_t/b_m)
//   v_new[t][c] = k_cumsum[t][c] − k_cumdecay[t][·]·S     (原地写回 cumsum 槽)
//   o[t][c]     = (q'_t·b_t)·S + Σ_m qk[t][m]·v_new[m][c]
//   S ← b_end·S + Σ_t k'_t·(b_end/b_t)·v_new[t][c]        (state 原地)
// 四个 kernel 统一微结构:128 线程(4 SIMD group)算 32×32 输出 tile,每 SIMD
// group 16×16(2×2 个 8×8 矩阵);staging ≤ 20KB,grid.z = value_head 一次
// dispatch 覆盖全部 head 的 tile。
// ============================================================================

// 步骤 1:qk = tril((q'·k')·b_t/b_m),K = key_head_dim,输出 [64,64]/head。
kernel void gdn_chunk_qk_f16(
    device const half *mixed [[buffer(0)]],
    device const float *controls [[buffer(1)]],
    device const float *scratch_b [[buffer(2)]],
    device half *scratch_qk [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &key_heads [[buffer(5)]],
    constant uint &value_heads [[buffer(6)]],
    constant uint &key_head_dim [[buffer(7)]],
    constant uint &value_head_dim [[buffer(8)]],
    constant uint &chunk [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]])
{
    const uint kd = key_head_dim;
    const uint head = group.z;
    const uint key_head = head % key_heads;
    const uint conv_dim = key_heads * kd * 2 + value_heads * value_head_dim;
    const uint control_stride = key_heads * 2 + value_heads;
    const uint chunk_begin = chunk * GDN_CHUNK;
    const uint chunk_rows = min(GDN_CHUNK, rows - chunk_begin);
    const ulong b_base = (ulong(chunk) * value_heads + head) * GDN_CHUNK;
    const uint t_base = group.y * 32;
    const uint m_base = group.x * 32;
    const uint gi = simd_index & 1;
    const uint gj = simd_index >> 1;

    threadgroup half stage_a[32 * 128];
    threadgroup half stage_b[128 * 32];
    threadgroup float result[32 * 32];
    // stage_a[t][k] = q'_t[k];行 t 超出 chunk_rows 清零,杜绝 mixed 越界读。
    for (uint index = tid; index < 32 * kd; index += 128) {
        const uint t = index / kd;
        const uint k = index - t * kd;
        half value = 0.0h;
        if (t_base + t < chunk_rows) {
            const uint row = chunk_begin + t_base + t;
            const float scale = controls[ulong(row) * control_stride + key_head * 2];
            value = half(float(mixed[ulong(row) * conv_dim + ulong(key_head) * kd + k]) * scale);
        }
        stage_a[index] = value;
    }
    // stage_b[k][m] = k'_m[k](B 转置装载)。
    for (uint index = tid; index < kd * 32; index += 128) {
        const uint k = index / 32;
        const uint m = index - k * 32;
        half value = 0.0h;
        if (m_base + m < chunk_rows) {
            const uint row = chunk_begin + m_base + m;
            const float scale = controls[ulong(row) * control_stride + key_head * 2 + 1];
            value = half(float(mixed[ulong(row) * conv_dim + key_heads * kd + ulong(key_head) * kd + k]) * scale);
        }
        stage_b[index] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[2][2];
    for (uint mi = 0; mi < 2; ++mi) {
        for (uint nj = 0; nj < 2; ++nj) {
            acc[mi][nj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    simdgroup_half8x8 a[2];
    simdgroup_half8x8 b[2];
    for (uint k = 0; k < kd; k += 8) {
        for (uint mi = 0; mi < 2; ++mi) simdgroup_load(a[mi], stage_a + (gi * 16 + mi * 8) * kd + k, kd);
        for (uint nj = 0; nj < 2; ++nj) simdgroup_load(b[nj], stage_b + k * 32 + gj * 16 + nj * 8, 32);
        for (uint mi = 0; mi < 2; ++mi) {
            for (uint nj = 0; nj < 2; ++nj) simdgroup_multiply_accumulate(acc[mi][nj], a[mi], b[nj], acc[mi][nj]);
        }
    }
    for (uint mi = 0; mi < 2; ++mi) {
        for (uint nj = 0; nj < 2; ++nj) simdgroup_store(acc[mi][nj], result + (gi * 16 + mi * 8) * 32 + gj * 16 + nj * 8, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < 32 * 32; index += 128) {
        const uint t = t_base + index / 32;
        const uint m = m_base + index % 32;
        // 严格下三角(t ≥ m 蕴含 m < chunk_rows);b 单调不增,b_t/b_m ≤ 1 无溢出。
        float value = 0.0f;
        if (t >= m && t < chunk_rows) {
            value = result[index] * (scratch_b[b_base + t] / max(scratch_b[b_base + m], 1.0e-20f));
        }
        scratch_qk[(b_base + t) * GDN_CHUNK + m] = half(value);
    }
}

// 步骤 2:v_new = k_cumsum − k_cumdecay·S,K = key_head_dim。v_new 原地写回
// cumsum 槽(每个元素先读后写、仅属一个 tile,无跨组竞态);S 以 f16 staging。
kernel void gdn_chunk_vnew_f16(
    device const half *scratch_decay [[buffer(0)]],
    device const float *state [[buffer(1)]],
    device half *scratch_cumsum [[buffer(2)]],
    constant uint &value_heads [[buffer(3)]],
    constant uint &key_head_dim [[buffer(4)]],
    constant uint &value_head_dim [[buffer(5)]],
    constant uint &chunk [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]])
{
    const uint kd = key_head_dim;
    const uint vd = value_head_dim;
    const uint head = group.z;
    const uint t_base = group.y * 32;
    const uint c_base = group.x * 32;
    if (c_base >= vd) return;
    const ulong chunk_head = (ulong(chunk) * value_heads + head) * GDN_CHUNK;
    const ulong state_head = ulong(head) * kd * vd;
    const uint gi = simd_index & 1;
    const uint gj = simd_index >> 1;

    threadgroup half stage_a[32 * 128];
    threadgroup half stage_b[128 * 32];
    threadgroup float result[32 * 32];
    for (uint index = tid; index < 32 * kd; index += 128) {
        stage_a[index] = scratch_decay[(chunk_head + t_base + index / kd) * kd + index % kd];
    }
    for (uint index = tid; index < kd * 32; index += 128) {
        const uint k = index / 32;
        const uint c = c_base + index % 32;
        stage_b[index] = c < vd ? half(state[state_head + ulong(k) * vd + c]) : 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[2][2];
    for (uint mi = 0; mi < 2; ++mi) {
        for (uint nj = 0; nj < 2; ++nj) {
            acc[mi][nj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    simdgroup_half8x8 a[2];
    simdgroup_half8x8 b[2];
    for (uint k = 0; k < kd; k += 8) {
        for (uint mi = 0; mi < 2; ++mi) simdgroup_load(a[mi], stage_a + (gi * 16 + mi * 8) * kd + k, kd);
        for (uint nj = 0; nj < 2; ++nj) simdgroup_load(b[nj], stage_b + k * 32 + gj * 16 + nj * 8, 32);
        for (uint mi = 0; mi < 2; ++mi) {
            for (uint nj = 0; nj < 2; ++nj) simdgroup_multiply_accumulate(acc[mi][nj], a[mi], b[nj], acc[mi][nj]);
        }
    }
    for (uint mi = 0; mi < 2; ++mi) {
        for (uint nj = 0; nj < 2; ++nj) simdgroup_store(acc[mi][nj], result + (gi * 16 + mi * 8) * 32 + gj * 16 + nj * 8, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < 32 * 32; index += 128) {
        const uint t = t_base + index / 32;
        const uint c = c_base + index % 32;
        if (c < vd) {
            const ulong target = (chunk_head + t) * vd + c;
            scratch_cumsum[target] = half(float(scratch_cumsum[target]) - result[index]);
        }
    }
}

// 步骤 3:o = (q'·b_t)·S + tril(qk)·v_new,两轮 GEMM 共用同一累加器
// (K=64 的 intra 与 K=kd 的 inter),staging 复用同一块 threadgroup。
kernel void gdn_chunk_output_f16(
    device const half *mixed [[buffer(0)]],
    device const float *controls [[buffer(1)]],
    device const float *scratch_b [[buffer(2)]],
    device const half *scratch_qk [[buffer(3)]],
    device const half *scratch_vnew [[buffer(4)]],
    device const float *state [[buffer(5)]],
    device half *core [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &key_heads [[buffer(8)]],
    constant uint &value_heads [[buffer(9)]],
    constant uint &key_head_dim [[buffer(10)]],
    constant uint &value_head_dim [[buffer(11)]],
    constant uint &chunk [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]])
{
    const uint kd = key_head_dim;
    const uint vd = value_head_dim;
    const uint head = group.z;
    const uint key_head = head % key_heads;
    const uint conv_dim = key_heads * kd * 2 + value_heads * vd;
    const uint control_stride = key_heads * 2 + value_heads;
    const uint chunk_begin = chunk * GDN_CHUNK;
    const uint chunk_rows = min(GDN_CHUNK, rows - chunk_begin);
    const ulong b_base = (ulong(chunk) * value_heads + head) * GDN_CHUNK;
    const uint t_base = group.y * 32;
    const uint c_base = group.x * 32;
    if (c_base >= vd) return;
    const ulong state_head = ulong(head) * kd * vd;
    const uint gi = simd_index & 1;
    const uint gj = simd_index >> 1;

    threadgroup half stage_a[32 * 128];
    threadgroup half stage_b[128 * 32];
    threadgroup float result[32 * 32];
    simdgroup_float8x8 acc[2][2];
    for (uint mi = 0; mi < 2; ++mi) {
        for (uint nj = 0; nj < 2; ++nj) {
            acc[mi][nj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    simdgroup_half8x8 a[2];
    simdgroup_half8x8 b[2];

    // 第一轮:intra = qk @ v_new(K=64;qk 与 v_new 超出 chunk_rows 的行列已是 0)。
    for (uint index = tid; index < 32 * GDN_CHUNK; index += 128) {
        const uint t = index / GDN_CHUNK;
        const uint m = index - t * GDN_CHUNK;
        stage_a[index] = scratch_qk[(b_base + t_base + t) * GDN_CHUNK + m];
    }
    for (uint index = tid; index < GDN_CHUNK * 32; index += 128) {
        const uint m = index / 32;
        const uint c = c_base + index % 32;
        stage_b[index] = c < vd ? scratch_vnew[(b_base + m) * vd + c] : 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < GDN_CHUNK; k += 8) {
        for (uint mi = 0; mi < 2; ++mi) simdgroup_load(a[mi], stage_a + (gi * 16 + mi * 8) * GDN_CHUNK + k, GDN_CHUNK);
        for (uint nj = 0; nj < 2; ++nj) simdgroup_load(b[nj], stage_b + k * 32 + gj * 16 + nj * 8, 32);
        for (uint mi = 0; mi < 2; ++mi) {
            for (uint nj = 0; nj < 2; ++nj) simdgroup_multiply_accumulate(acc[mi][nj], a[mi], b[nj], acc[mi][nj]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 第二轮:inter += (q'·scale·b_t) @ S(K=kd)。
    for (uint index = tid; index < 32 * kd; index += 128) {
        const uint t = index / kd;
        const uint k = index - t * kd;
        half value = 0.0h;
        if (t_base + t < chunk_rows) {
            const uint row = chunk_begin + t_base + t;
            const float scale = controls[ulong(row) * control_stride + key_head * 2] * scratch_b[b_base + t_base + t];
            value = half(float(mixed[ulong(row) * conv_dim + ulong(key_head) * kd + k]) * scale);
        }
        stage_a[index] = value;
    }
    for (uint index = tid; index < kd * 32; index += 128) {
        const uint k = index / 32;
        const uint c = c_base + index % 32;
        stage_b[index] = c < vd ? half(state[state_head + ulong(k) * vd + c]) : 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < kd; k += 8) {
        for (uint mi = 0; mi < 2; ++mi) simdgroup_load(a[mi], stage_a + (gi * 16 + mi * 8) * kd + k, kd);
        for (uint nj = 0; nj < 2; ++nj) simdgroup_load(b[nj], stage_b + k * 32 + gj * 16 + nj * 8, 32);
        for (uint mi = 0; mi < 2; ++mi) {
            for (uint nj = 0; nj < 2; ++nj) simdgroup_multiply_accumulate(acc[mi][nj], a[mi], b[nj], acc[mi][nj]);
        }
    }
    for (uint mi = 0; mi < 2; ++mi) {
        for (uint nj = 0; nj < 2; ++nj) simdgroup_store(acc[mi][nj], result + (gi * 16 + mi * 8) * 32 + gj * 16 + nj * 8, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < 32 * 32; index += 128) {
        const uint t = t_base + index / 32;
        const uint c = c_base + index % 32;
        if (t < chunk_rows && c < vd) {
            core[ulong(chunk_begin + t) * (value_heads * vd) + ulong(head) * vd + c] = finite_f16(result[index]);
        }
    }
}

// 步骤 4:S ← b_end·S + (k'·(b_end/b))ᵀ @ v_new,K=64,输出 [kd,vd]/head。
// state 原地更新:b_end·S 项逐元素先读后写,每个元素只被一个线程触碰。
kernel void gdn_chunk_state_f16(
    device const half *mixed [[buffer(0)]],
    device const float *controls [[buffer(1)]],
    device const float *scratch_b [[buffer(2)]],
    device const half *scratch_vnew [[buffer(3)]],
    device float *state [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &key_heads [[buffer(6)]],
    constant uint &value_heads [[buffer(7)]],
    constant uint &key_head_dim [[buffer(8)]],
    constant uint &value_head_dim [[buffer(9)]],
    constant uint &chunk [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]])
{
    const uint kd = key_head_dim;
    const uint vd = value_head_dim;
    const uint head = group.z;
    const uint key_head = head % key_heads;
    const uint conv_dim = key_heads * kd * 2 + value_heads * vd;
    const uint control_stride = key_heads * 2 + value_heads;
    const uint chunk_begin = chunk * GDN_CHUNK;
    const uint chunk_rows = min(GDN_CHUNK, rows - chunk_begin);
    const ulong b_base = (ulong(chunk) * value_heads + head) * GDN_CHUNK;
    const uint k_base = group.y * 32;
    const uint c_base = group.x * 32;
    if (c_base >= vd) return;
    const ulong state_head = ulong(head) * kd * vd;
    const float b_end = scratch_b[b_base + chunk_rows - 1];
    const uint gi = simd_index & 1;
    const uint gj = simd_index >> 1;

    threadgroup half stage_a[32 * GDN_CHUNK];
    threadgroup half stage_b[GDN_CHUNK * 32];
    threadgroup float result[32 * 32];
    // stage_a[k][t] = k'_t[k]·(b_end/b_t) —— A 转置装载(m 维是 key 列)。
    for (uint index = tid; index < 32 * GDN_CHUNK; index += 128) {
        const uint k = index / GDN_CHUNK;
        const uint t = index - k * GDN_CHUNK;
        half value = 0.0h;
        if (t < chunk_rows) {
            const uint row = chunk_begin + t;
            const float scale = controls[ulong(row) * control_stride + key_head * 2 + 1];
            const float ratio = b_end / max(scratch_b[b_base + t], 1.0e-20f);
            value = half(float(mixed[ulong(row) * conv_dim + key_heads * kd + ulong(key_head) * kd + k_base + k]) * scale * ratio);
        }
        stage_a[index] = value;
    }
    for (uint index = tid; index < GDN_CHUNK * 32; index += 128) {
        const uint t = index / 32;
        const uint c = c_base + index % 32;
        stage_b[index] = c < vd ? scratch_vnew[(b_base + t) * vd + c] : 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[2][2];
    for (uint mi = 0; mi < 2; ++mi) {
        for (uint nj = 0; nj < 2; ++nj) {
            acc[mi][nj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    simdgroup_half8x8 a[2];
    simdgroup_half8x8 b[2];
    for (uint k = 0; k < GDN_CHUNK; k += 8) {
        for (uint mi = 0; mi < 2; ++mi) simdgroup_load(a[mi], stage_a + (gi * 16 + mi * 8) * GDN_CHUNK + k, GDN_CHUNK);
        for (uint nj = 0; nj < 2; ++nj) simdgroup_load(b[nj], stage_b + k * 32 + gj * 16 + nj * 8, 32);
        for (uint mi = 0; mi < 2; ++mi) {
            for (uint nj = 0; nj < 2; ++nj) simdgroup_multiply_accumulate(acc[mi][nj], a[mi], b[nj], acc[mi][nj]);
        }
    }
    for (uint mi = 0; mi < 2; ++mi) {
        for (uint nj = 0; nj < 2; ++nj) simdgroup_store(acc[mi][nj], result + (gi * 16 + mi * 8) * 32 + gj * 16 + nj * 8, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < 32 * 32; index += 128) {
        const uint k = index / 32;
        const uint c = c_base + index % 32;
        if (c < vd) {
            const ulong target = state_head + ulong(k_base + k) * vd + c;
            state[target] = state[target] * b_end + result[index];
        }
    }
}
kernel void gated_delta_recurrent_prefill_simd8_f16(
    device const half *mixed [[buffer(0)]],
    device const half *beta [[buffer(2)]],
    device float *state [[buffer(5)]],
    device half *output [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &key_heads [[buffer(8)]],
    constant uint &value_heads [[buffer(9)]],
    constant uint &key_head_dim [[buffer(10)]],
    constant uint &value_head_dim [[buffer(11)]],
    device const float *controls [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    constexpr uint tile_columns = 8;
    constexpr uint key_lanes = 32;
    const uint tiles_per_head = (value_head_dim + tile_columns - 1) / tile_columns;
    if (group >= value_heads * tiles_per_head) return;
    const uint lane = thread_index & 31;
    const uint value_head = group / tiles_per_head;
    const uint value_column = (group - value_head * tiles_per_head) * tile_columns;

    const uint key_head = value_head % key_heads;
    const uint key_dim = key_heads * key_head_dim;
    const uint value_dim = value_heads * value_head_dim;
    const uint conv_dim = key_dim * 2 + value_dim;
    const ulong state_head = ulong(value_head) * key_head_dim * value_head_dim;
    const uint keys_per_lane = key_head_dim / key_lanes;
    float4 local_state_lo[4];
    float4 local_state_hi[4];

    for (uint slot = 0; slot < keys_per_lane; ++slot) {
        const uint key_column = lane * keys_per_lane + slot;
        const ulong state_column = state_head + ulong(key_column) * value_head_dim + value_column;
        local_state_lo[slot] = float4(state[state_column], state[state_column + 1], state[state_column + 2], state[state_column + 3]);
        local_state_hi[slot] = float4(state[state_column + 4], state[state_column + 5], state[state_column + 6], state[state_column + 7]);
    }

    for (uint row = 0; row < rows; ++row) {
        const ulong mixed_row = ulong(row) * conv_dim;
        const ulong query_base = mixed_row + ulong(key_head) * key_head_dim;
        const ulong key_base = mixed_row + key_dim + ulong(key_head) * key_head_dim;
        const ulong control_row = ulong(row) * (key_heads * 2 + value_heads);
        float query_scale = lane == 0 ? controls[control_row + key_head * 2] : 0.0f;
        float key_scale = lane == 0 ? controls[control_row + key_head * 2 + 1] : 0.0f;
        float decay = lane == 0 ? controls[control_row + key_heads * 2 + value_head] : 0.0f;
        query_scale = simd_broadcast_first(query_scale);
        key_scale = simd_broadcast_first(key_scale);
        decay = simd_broadcast_first(decay);

        float beta_exp = 0.0f;
        if (lane == 0) beta_exp = exp(-float(beta[ulong(row) * value_heads + value_head]));
        beta_exp = simd_broadcast_first(beta_exp);
        const float beta_value = 1.0f / (1.0f + beta_exp);

        for (uint slot = 0; slot < keys_per_lane; ++slot) {
            local_state_lo[slot] *= decay;
            local_state_hi[slot] *= decay;
        }

        float4 memory_chunk_lo = 0.0f;
        float4 memory_chunk_hi = 0.0f;
        for (uint slot = 0; slot < keys_per_lane; ++slot) {
            const uint key_column = lane * keys_per_lane + slot;
            const float key = float(mixed[key_base + key_column]) * key_scale;
            memory_chunk_lo += local_state_lo[slot] * key;
            memory_chunk_hi += local_state_hi[slot] * key;
        }
        const float4 memory_lo = float4(simd_sum(memory_chunk_lo.x), simd_sum(memory_chunk_lo.y), simd_sum(memory_chunk_lo.z), simd_sum(memory_chunk_lo.w));
        const float4 memory_hi = float4(simd_sum(memory_chunk_hi.x), simd_sum(memory_chunk_hi.y), simd_sum(memory_chunk_hi.z), simd_sum(memory_chunk_hi.w));

        const ulong value_index = mixed_row + key_dim * 2 + ulong(value_head) * value_head_dim + value_column;
        float4 values_lo = 0.0f;
        float4 values_hi = 0.0f;
        if (lane == 0) {
            values_lo = float4(mixed[value_index], mixed[value_index + 1], mixed[value_index + 2], mixed[value_index + 3]);
            values_hi = float4(mixed[value_index + 4], mixed[value_index + 5], mixed[value_index + 6], mixed[value_index + 7]);
        }
        values_lo = simd_broadcast_first(values_lo);
        values_hi = simd_broadcast_first(values_hi);
        const float4 delta_lo = (values_lo - memory_lo) * beta_value;
        const float4 delta_hi = (values_hi - memory_hi) * beta_value;

        for (uint slot = 0; slot < keys_per_lane; ++slot) {
            const uint key_column = lane * keys_per_lane + slot;
            const float key = float(mixed[key_base + key_column]) * key_scale;
            local_state_lo[slot] += delta_lo * key;
            local_state_hi[slot] += delta_hi * key;
        }

        float4 result_chunk_lo = 0.0f;
        float4 result_chunk_hi = 0.0f;
        for (uint slot = 0; slot < keys_per_lane; ++slot) {
            const uint key_column = lane * keys_per_lane + slot;
            const float query = float(mixed[query_base + key_column]) * query_scale;
            result_chunk_lo += local_state_lo[slot] * query;
            result_chunk_hi += local_state_hi[slot] * query;
        }
        const float4 result_lo = float4(simd_sum(result_chunk_lo.x), simd_sum(result_chunk_lo.y), simd_sum(result_chunk_lo.z), simd_sum(result_chunk_lo.w));
        const float4 result_hi = float4(simd_sum(result_chunk_hi.x), simd_sum(result_chunk_hi.y), simd_sum(result_chunk_hi.z), simd_sum(result_chunk_hi.w));

        if (lane == 0) {
            const ulong output_index = ulong(row) * value_dim + ulong(value_head) * value_head_dim + value_column;
            output[output_index] = finite_f16(result_lo.x);
            output[output_index + 1] = finite_f16(result_lo.y);
            output[output_index + 2] = finite_f16(result_lo.z);
            output[output_index + 3] = finite_f16(result_lo.w);
            output[output_index + 4] = finite_f16(result_hi.x);
            output[output_index + 5] = finite_f16(result_hi.y);
            output[output_index + 6] = finite_f16(result_hi.z);
            output[output_index + 7] = finite_f16(result_hi.w);
        }
    }

    for (uint slot = 0; slot < keys_per_lane; ++slot) {
        const uint key_column = lane * keys_per_lane + slot;
        const ulong state_column = state_head + ulong(key_column) * value_head_dim + value_column;
        state[state_column] = local_state_lo[slot].x;
        state[state_column + 1] = local_state_lo[slot].y;
        state[state_column + 2] = local_state_lo[slot].z;
        state[state_column + 3] = local_state_lo[slot].w;
        state[state_column + 4] = local_state_hi[slot].x;
        state[state_column + 5] = local_state_hi[slot].y;
        state[state_column + 6] = local_state_hi[slot].z;
        state[state_column + 7] = local_state_hi[slot].w;
    }
}
kernel void gated_delta_recurrent_legacy_f16(
    device const half *mixed [[buffer(0)]],
    device const half *alpha [[buffer(1)]],
    device const half *beta [[buffer(2)]],
    device const float *a_log [[buffer(3)]],
    device const float *dt_bias [[buffer(4)]],
    device float *state [[buffer(5)]],
    device half *output [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &key_heads [[buffer(8)]],
    constant uint &value_heads [[buffer(9)]],
    constant uint &key_head_dim [[buffer(10)]],
    constant uint &value_head_dim [[buffer(11)]],
    uint value_head [[threadgroup_position_in_grid]],
    uint value_column [[thread_index_in_threadgroup]])
{
    if (value_head >= value_heads) return;
    // 对齐 GGML repeat 布局:Q/K head 整体重复,不是每个 head 连续分组。
    // decode 单 token 时 query/key 归一化系数、decay、beta 对全组 128 个
    // value_column 完全相同,逐 thread 重算是纯浪费:thread 0 算一次放
    // threadgroup memory,barrier 后全组共享(对标 llama.cpp shared x_dt/dA)。
    threadgroup float shared_controls[4];
    const uint key_head = value_head % key_heads;
    const uint key_dim = key_heads * key_head_dim;
    const uint value_dim = value_heads * value_head_dim;
    const uint conv_dim = key_dim * 2 + value_dim;
    const ulong state_head = ulong(value_head) * key_head_dim * value_head_dim;

    for (uint row = 0; row < rows; ++row) {
        const ulong mixed_row = ulong(row) * conv_dim;
        const ulong query_base = mixed_row + ulong(key_head) * key_head_dim;
        const ulong key_base = mixed_row + key_dim + ulong(key_head) * key_head_dim;
        if (value_column == 0) {
            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (uint key_column = 0; key_column < key_head_dim; ++key_column) {
                const float query = float(mixed[query_base + key_column]);
                const float key = float(mixed[key_base + key_column]);
                query_sum += query * query;
                key_sum += key * key;
            }
            shared_controls[0] = rsqrt(max(query_sum, 1.0e-12f)) * rsqrt(float(value_head_dim));
            shared_controls[1] = rsqrt(max(key_sum, 1.0e-12f));
            const float step = float(alpha[ulong(row) * value_heads + value_head]) + dt_bias[value_head];
            const float softplus = step > 20.0f ? step : (step < -20.0f ? exp(step) : log(1.0f + exp(step)));
            shared_controls[2] = exp(-exp(a_log[value_head]) * softplus);
            shared_controls[3] = 1.0f / (1.0f + exp(-float(beta[ulong(row) * value_heads + value_head])));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float query_scale = shared_controls[0];
        const float key_scale = shared_controls[1];
        const float decay = shared_controls[2];
        const float beta_value = shared_controls[3];
        if (value_column < value_head_dim) {
            const ulong value_index = mixed_row + key_dim * 2 + ulong(value_head) * value_head_dim + value_column;
            float memory = 0.0f;
            for (uint key_column = 0; key_column < key_head_dim; ++key_column) {
                // state 乘 decay 只进累加不写回;decayed 在 update 循环重算,
                // 每元素每 token 省一次 global 写(2 写 -> 1 写)。
                memory += state[state_head + ulong(key_column) * value_head_dim + value_column] * decay * float(mixed[key_base + key_column]) * key_scale;
            }
            const float delta = (float(mixed[value_index]) - memory) * beta_value;
            float result = 0.0f;
            for (uint key_column = 0; key_column < key_head_dim; ++key_column) {
                const ulong state_index = state_head + ulong(key_column) * value_head_dim + value_column;
                const float key = float(mixed[key_base + key_column]) * key_scale;
                const float updated = state[state_index] * decay + key * delta;
                state[state_index] = updated;
                result += updated * float(mixed[query_base + key_column]) * query_scale;
            }
            output[ulong(row) * value_dim + ulong(value_head) * value_head_dim + value_column] = finite_f16(result);
        }
        // rows>1 时防止下一轮 thread 0 覆写 shared_controls 与其他 thread 读竞争。
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// decode 取向重写(对标 llama.cpp kernel_gated_delta_net):state 在 token 循环
// 全程驻留寄存器,只在 kernel 首/尾各读写一次;lane 覆盖 key 维,行内归约用
// simd_sum(树形 ~5 步),取代 legacy 的 128 次串行依赖链。
// TG = 32 lane x columns_per_group(4) = 128 thread,负责一个 value_head 的 4 个
// value_column;grid = (value_head_dim/4, value_heads)。Qwen3.8 为例:1536 TG,
// 是 legacy(48 TG)的 32 倍并行度。
// state 布局仍是 GGML 契约 [key,value](行=key),lane 读 4B 跨行不合并,但
// 每 head 仅 64KB,同 head 32 个 TG 经 L2 吸收,DRAM 唯一流量不变。
kernel void gated_delta_recurrent_rows_f16(
    device const half *mixed [[buffer(0)]],
    device const half *alpha [[buffer(1)]],
    device const half *beta [[buffer(2)]],
    device const float *a_log [[buffer(3)]],
    device const float *dt_bias [[buffer(4)]],
    device float *state [[buffer(5)]],
    device half *output [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &key_heads [[buffer(8)]],
    constant uint &value_heads [[buffer(9)]],
    constant uint &key_head_dim [[buffer(10)]],
    constant uint &value_head_dim [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint column_slot [[simdgroup_index_in_threadgroup]])
{
    // 编译期常量化(dispatch 侧保证 key_head_dim%32==0 且 <=128,value_head_dim%4==0)
    const uint keys_per_lane = zllm_fc_u32_0;
    const uint value_head = group.y;
    const uint value_column = group.x * 4 + column_slot;
    const uint key_head = value_head % key_heads;
    const uint key_dim = key_heads * key_head_dim;
    const uint value_dim = value_heads * value_head_dim;
    const uint conv_dim = key_dim * 2 + value_dim;
    const ulong state_head = ulong(value_head) * key_head_dim * value_head_dim;

    float ls[4];
    #pragma unroll
    for (uint j = 0; j < 4; ++j) {
        if (j < keys_per_lane) {
            ls[j] = state[state_head + ulong(lane * keys_per_lane + j) * value_head_dim + value_column];
        }
    }
    for (uint row = 0; row < rows; ++row) {
        const ulong mixed_row = ulong(row) * conv_dim;
        const ulong query_base = mixed_row + ulong(key_head) * key_head_dim;
        const ulong key_base = mixed_row + key_dim + ulong(key_head) * key_head_dim;
        // q/k 归一化系数:lane 内部分和 + simd_sum;4 个 simdgroup 冗余各算一份(省跨组 barrier)
        float qn[4];
        float kn[4];
        float query_sum = 0.0f;
        float key_sum = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            if (j < keys_per_lane) {
                const uint key_column = lane * keys_per_lane + j;
                qn[j] = float(mixed[query_base + key_column]);
                kn[j] = float(mixed[key_base + key_column]);
                query_sum += qn[j] * qn[j];
                key_sum += kn[j] * kn[j];
            }
        }
        query_sum = simd_sum(query_sum);
        key_sum = simd_sum(key_sum);
        const float query_scale = rsqrt(max(query_sum, 1.0e-12f)) * rsqrt(float(value_head_dim));
        const float key_scale = rsqrt(max(key_sum, 1.0e-12f));
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            if (j < keys_per_lane) {
                qn[j] *= query_scale;
                kn[j] *= key_scale;
            }
        }
        const float step = float(alpha[ulong(row) * value_heads + value_head]) + dt_bias[value_head];
        const float softplus = step > 20.0f ? step : (step < -20.0f ? exp(step) : log(1.0f + exp(step)));
        const float decay = exp(-exp(a_log[value_head]) * softplus);
        const float beta_value = 1.0f / (1.0f + exp(-float(beta[ulong(row) * value_heads + value_head])));
        float memory = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            if (j < keys_per_lane) {
                ls[j] *= decay;
                memory += ls[j] * kn[j];
            }
        }
        memory = simd_sum(memory);
        const float delta = (float(mixed[mixed_row + key_dim * 2 + ulong(value_head) * value_head_dim + value_column]) - memory) * beta_value;
        float result = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            if (j < keys_per_lane) {
                ls[j] += kn[j] * delta;
                result += ls[j] * qn[j];
            }
        }
        result = simd_sum(result);
        if (lane == 0) {
            output[ulong(row) * value_dim + ulong(value_head) * value_head_dim + value_column] = finite_f16(result);
        }
    }
    #pragma unroll
    for (uint j = 0; j < 4; ++j) {
        if (j < keys_per_lane) {
            state[state_head + ulong(lane * keys_per_lane + j) * value_head_dim + value_column] = ls[j];
        }
    }
}
kernel void gated_delta_norm_gate_f16(
    device const half *input [[buffer(0)]],
    device const half *gate [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &value_heads [[buffer(4)]],
    constant uint &value_head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float sums[256];
    const uint value_head = group % value_heads;
    const uint row = group / value_heads;
    const ulong begin = ulong(row) * value_heads * value_head_dim + ulong(value_head) * value_head_dim;
    const float value = lane < value_head_dim ? float(input[begin + lane]) : 0.0f;
    sums[lane] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < value_head_dim) {
        const float inv_rms = rsqrt(sums[0] / float(value_head_dim) + eps);
        const float gate_value = float(gate[begin + lane]);
        const float silu_gate = gate_value / (1.0f + exp(-gate_value));
        output[begin + lane] = finite_f16(value * inv_rms * weight[lane] * silu_gate);
    }
}
kernel void softplus_gate_scaled_f16(
    device const half *input [[buffer(0)]],
    device const half *gate [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &gate_columns [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    constant float &gate_scale [[buffer(6)]],
    constant float &output_scale [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const uint column = index % columns;
    const uint gate_column = column / (columns / gate_columns);
    const uint gate_index = (index / columns) * gate_columns + gate_column;
    const float value = float(gate[gate_index]) * gate_scale;
    const float softplus = value > 20.0f ? value : (value < -20.0f ? exp(value) : log(1.0f + exp(value)));
    output[index] = finite_f16(float(input[index]) * softplus * output_scale);
}
"#;

use crate::backend::metal::api as metal;

use super::dense::{GatedActivation, validate_tensor};
use super::{Activation, GatedDeltaNetSpec, MTLSize, MetalContext, MetalTensor, MetalTensorDType, THREADS, launch_1d, launch_rows_with_pipeline, mem, set_bytes, validate_u32};

pub(crate) fn rmsnorm_tensor_resident_weight(ctx: &MetalContext, input: &MetalTensor, weight: &MetalTensor, eps: f32, weight_offset: f32) -> Result<MetalTensor, String> {
    if weight.rows * weight.cols != input.cols {
        return Err(format!("resident RMSNorm weight 长度 {} 与 input cols {} 不符", weight.rows * weight.cols, input.cols));
    }
    let output = match input.dtype {
        MetalTensorDType::F16 => ctx.tensor_zeros(input.rows, input.cols),
        MetalTensorDType::Bf16 => ctx.tensor_zeros_bf16(input.rows, input.cols),
        MetalTensorDType::F32 => ctx.tensor_zeros_f32(input.rows, input.cols),
    };
    let output_f16 = (input.dtype == MetalTensorDType::F32).then(|| ctx.tensor_zeros(input.rows, input.cols));
    let columns = validate_u32("cols", input.cols)?;
    let write_bytes = output.buffer.length() + output_f16.as_ref().map_or(0, |tensor| tensor.buffer.length());
    // F16 输入 + 列 4 对齐:单 SIMD group 版,消除树状归约 barrier(单行算子 ~27µs -> 延迟地板)
    if input.dtype == MetalTensorDType::F16 && columns.is_multiple_of(4) {
        let pipeline = ctx.pipeline("rms_norm_f16_simd")?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&weight.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(&encoder, 3, &columns);
        set_bytes(&encoder, 4, &eps);
        set_bytes(&encoder, 5, &weight_offset);
        encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new(32, 1, 1));
        encoder.end_encoding();
        let shape = format!("[{},{columns}]", input.rows);
        ctx.commit_and_wait_profiled(&command, "rms_norm_f16_simd", &shape, input.buffer.length() + weight.buffer.length(), write_bytes);
        return Ok(output);
    }
    let pipeline = match input.dtype {
        MetalTensorDType::F16 => "rms_norm_f16",
        MetalTensorDType::Bf16 => "rms_norm_bf16",
        MetalTensorDType::F32 => "rms_norm_f32_f16",
    };
    launch_rows_with_pipeline(ctx, pipeline, input.rows, input.cols, input.buffer.length() + weight.buffer.length(), write_bytes, |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&weight.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &eps);
        set_bytes(encoder, 5, &weight_offset);
        if let Some(output_f16) = &output_f16 {
            encoder.set_buffer(6, Some(&output_f16.buffer), 0);
        }
    })?;
    if let Some(output_f16) = &output_f16 {
        ctx.retain_f16_cast(&output, output_f16);
    }
    Ok(output)
}

pub(crate) fn rmsnorm_tensor_resident_f32_weight(ctx: &MetalContext, input: &MetalTensor, weight: &metal::Buffer, weight_len: usize, eps: f32, weight_offset: f32) -> Result<MetalTensor, String> {
    if input.dtype != MetalTensorDType::F32 || weight_len != input.cols {
        return Err(format!("F32 RMSNorm shape 不兼容: input=[{},{},{:?}] weight={weight_len}", input.rows, input.cols, input.dtype));
    }
    let output = ctx.tensor_zeros_f32(input.rows, input.cols);
    let output_f16 = ctx.tensor_zeros(input.rows, input.cols);
    let columns = validate_u32("cols", input.cols)?;
    // 列数 4 对齐时走单 SIMD group + float4 版本,长序列 prefill 带宽敏感。
    if columns.is_multiple_of(4) {
        let pipeline = ctx.pipeline("rms_norm_f32_weight_f32_f16_simd")?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(&encoder, 3, &columns);
        set_bytes(&encoder, 4, &eps);
        set_bytes(&encoder, 5, &weight_offset);
        encoder.set_buffer(6, Some(&output_f16.buffer), 0);
        encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new(32, 1, 1));
        encoder.end_encoding();
        let shape = format!("[{},{columns}]", input.rows);
        ctx.commit_and_wait_profiled(&command, "rms_norm_f32_weight_f32_f16", &shape, input.buffer.length() + weight.length(), output.buffer.length() + output_f16.buffer.length());
    } else {
        launch_rows_with_pipeline(ctx, "rms_norm_f32_weight_f32_f16", input.rows, input.cols, input.buffer.length() + weight.length(), output.buffer.length() + output_f16.buffer.length(), |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(weight), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            set_bytes(encoder, 3, &columns);
            set_bytes(encoder, 4, &eps);
            set_bytes(encoder, 5, &weight_offset);
            encoder.set_buffer(6, Some(&output_f16.buffer), 0);
        })?;
    }
    ctx.retain_f16_cast(&output, &output_f16);
    Ok(output)
}

/// F16 输入 + F32 权重的逐行 RMSNorm,输出仅 F16:消除 F32 权重路径两侧的
/// f16↔f32 cast 与多余的 F32 输出写出,数值与 cast 链逐位一致。
pub(crate) fn rmsnorm_f16_in_f32_weight_tensor_resident(ctx: &MetalContext, input: &MetalTensor, weight: &metal::Buffer, weight_len: usize, eps: f32, weight_offset: f32) -> Result<MetalTensor, String> {
    if input.dtype != MetalTensorDType::F16 || weight_len != input.cols {
        return Err(format!("F16-in F32-weight RMSNorm shape 不兼容: input=[{},{},{:?}] weight={weight_len}", input.rows, input.cols, input.dtype));
    }
    let output = ctx.tensor_zeros(input.rows, input.cols);
    let columns = validate_u32("cols", input.cols)?;
    if columns.is_multiple_of(4) {
        let pipeline = ctx.pipeline("rms_norm_f16_in_f32_weight_f16_simd")?;
        let command = ctx.command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(&encoder, 3, &columns);
        set_bytes(&encoder, 4, &eps);
        set_bytes(&encoder, 5, &weight_offset);
        encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new(32, 1, 1));
        encoder.end_encoding();
        let shape = format!("[{},{columns}]", input.rows);
        ctx.commit_and_wait_profiled(&command, "rms_norm_f16_in_f32_weight_f16", &shape, input.buffer.length() + weight.length(), output.buffer.length());
    } else {
        launch_rows_with_pipeline(ctx, "rms_norm_f16_in_f32_weight_f16", input.rows, input.cols, input.buffer.length() + weight.length(), output.buffer.length(), |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(weight), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            set_bytes(encoder, 3, &columns);
            set_bytes(encoder, 4, &eps);
            set_bytes(encoder, 5, &weight_offset);
        })?;
    }
    Ok(output)
}

/// K2-Horizon 分组 RMSNorm:F16 输入直读 F32 权重,输出 F16(层 norm 主路径)。
/// 单 simdgroup 逐行内核,组内 simd_sum 归约。
pub(crate) fn grouped_rmsnorm_f16_in_f32_weight_tensor_resident(ctx: &MetalContext, input: &MetalTensor, weight: &metal::Buffer, weight_len: usize, eps: f32, groups: usize) -> Result<MetalTensor, String> {
    if input.dtype != MetalTensorDType::F16 || weight_len != input.cols {
        return Err(format!("grouped RMSNorm shape 不兼容: input=[{},{},{:?}] weight={weight_len}", input.rows, input.cols, input.dtype));
    }
    if groups == 0 || !input.cols.is_multiple_of(groups) {
        return Err(format!("grouped RMSNorm groups={groups} 无法整除 cols={}", input.cols));
    }
    let output = ctx.tensor_zeros(input.rows, input.cols);
    let columns = validate_u32("cols", input.cols)?;
    let groups = validate_u32("groups", groups)?;
    let pipeline = ctx.pipeline("grouped_rms_norm_f16_in_f32_weight_f16_simd")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &columns);
    set_bytes(&encoder, 4, &eps);
    set_bytes(&encoder, 5, &groups);
    encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new((groups * 32) as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("[{},{columns}]", input.rows);
    ctx.commit_and_wait_profiled(&command, "grouped_rms_norm_f16_in_f32_weight_f16", &shape, input.buffer.length() + weight.length(), output.buffer.length());
    Ok(output)
}

/// K2-Horizon F32 residual 直接输出 F16 grouped RMSNorm，省去独立 cast。
pub(crate) fn grouped_rmsnorm_f32_in_f32_weight_to_f16_tensor_resident(ctx: &MetalContext, input: &MetalTensor, weight: &metal::Buffer, weight_len: usize, eps: f32, groups: usize) -> Result<MetalTensor, String> {
    if input.dtype != MetalTensorDType::F32 || weight_len != input.cols {
        return Err(format!("F32→F16 grouped RMSNorm shape 不兼容: input=[{},{},{:?}] weight={weight_len}", input.rows, input.cols, input.dtype));
    }
    if groups == 0 || !input.cols.is_multiple_of(groups) {
        return Err(format!("grouped RMSNorm groups={groups} 无法整除 cols={}", input.cols));
    }
    let output = ctx.tensor_zeros(input.rows, input.cols);
    let columns = validate_u32("cols", input.cols)?;
    let groups = validate_u32("groups", groups)?;
    let pipeline = ctx.pipeline("grouped_rms_norm_f32_in_f32_weight_f16_simd")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &columns);
    set_bytes(&encoder, 4, &eps);
    set_bytes(&encoder, 5, &groups);
    encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new((groups * 32) as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("[{},{}]", input.rows, input.cols);
    ctx.commit_and_wait_profiled(&command, "grouped_rms_norm_f32_in_f32_weight_f16", &shape, input.buffer.length() + weight.length(), output.buffer.length());
    Ok(output)
}

/// F32 residual + F16 投影，并直接生成下一子层的 grouped RMSNorm 输入。
pub fn add_f32_f16_grouped_rmsnorm_tensor_resident(ctx: &MetalContext, left: &MetalTensor, right: &MetalTensor, weight: &metal::Buffer, weight_len: usize, eps: f32, groups: usize) -> Result<(MetalTensor, MetalTensor), String> {
    validate_tensor("grouped RMSNorm add rhs", right, left.rows, left.cols)?;
    if left.dtype != MetalTensorDType::F32 || right.dtype != MetalTensorDType::F16 || weight_len != left.cols {
        return Err(format!("grouped RMSNorm add shape 不兼容: left=[{},{},{:?}] right={:?} weight={weight_len}", left.rows, left.cols, left.dtype, right.dtype));
    }
    if groups == 0 || !left.cols.is_multiple_of(groups) {
        return Err(format!("grouped RMSNorm add groups={groups} 无法整除 cols={}", left.cols));
    }
    let residual = ctx.tensor_zeros_f32(left.rows, left.cols);
    let normalized = ctx.tensor_zeros(left.rows, left.cols);
    let columns = validate_u32("cols", left.cols)?;
    let groups = validate_u32("groups", groups)?;
    let pipeline = ctx.pipeline("add_f32_f16_grouped_rms_norm_f32_weight_f16_simd")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&left.buffer), 0);
    encoder.set_buffer(1, Some(&right.buffer), 0);
    encoder.set_buffer(2, Some(weight), 0);
    encoder.set_buffer(3, Some(&residual.buffer), 0);
    encoder.set_buffer(4, Some(&normalized.buffer), 0);
    set_bytes(&encoder, 5, &columns);
    set_bytes(&encoder, 6, &eps);
    set_bytes(&encoder, 7, &groups);
    encoder.dispatch_thread_groups(MTLSize::new(left.rows as u64, 1, 1), MTLSize::new((groups * 32) as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("[{},{}]", left.rows, left.cols);
    ctx.commit_and_wait_profiled(&command, "add_f32_f16_grouped_rms_norm_f16", &shape, left.buffer.length() + right.buffer.length() + weight.length(), residual.buffer.length() + normalized.buffer.length());
    Ok((residual, normalized))
}

/// K2-Horizon 分组 RMSNorm 的 F32 路径(output head,norm_f32=true)。
pub(crate) fn grouped_rmsnorm_f32_tensor_resident(ctx: &MetalContext, input: &MetalTensor, weight: &metal::Buffer, weight_len: usize, eps: f32, groups: usize) -> Result<MetalTensor, String> {
    if input.dtype != MetalTensorDType::F32 || weight_len != input.cols {
        return Err(format!("F32 grouped RMSNorm shape 不兼容: input=[{},{},{:?}] weight={weight_len}", input.rows, input.cols, input.dtype));
    }
    if groups == 0 || !input.cols.is_multiple_of(groups) {
        return Err(format!("grouped RMSNorm groups={groups} 无法整除 cols={}", input.cols));
    }
    let output = ctx.tensor_zeros_f32(input.rows, input.cols);
    let columns = validate_u32("cols", input.cols)?;
    let groups = validate_u32("groups", groups)?;
    let pipeline = ctx.pipeline("grouped_rms_norm_f32_in_f32_weight_f32")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    set_bytes(&encoder, 3, &columns);
    set_bytes(&encoder, 4, &eps);
    set_bytes(&encoder, 5, &groups);
    encoder.dispatch_thread_groups(MTLSize::new(input.rows as u64, 1, 1), MTLSize::new((groups * 32) as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("[{},{columns}]", input.rows);
    ctx.commit_and_wait_profiled(&command, "grouped_rms_norm_f32_in_f32_weight_f32", &shape, input.buffer.length() + weight.length(), output.buffer.length());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn segmented_rmsnorm_add_scaled_tensor(ctx: &MetalContext, left: &MetalTensor, right: &MetalTensor, weight: &MetalTensor, segments: usize, segment_columns: usize, eps: f32, scale: f32) -> Result<MetalTensor, String> {
    let expected_columns = segments.checked_mul(segment_columns).ok_or("segmented RMSNorm 列数溢出")?;
    let operand_dtype = if left.dtype == right.dtype && matches!(left.dtype, MetalTensorDType::Bf16 | MetalTensorDType::F16) { left.dtype } else { MetalTensorDType::F32 };
    if segments == 0
        || segment_columns == 0
        || left.rows == 0
        || left.rows != right.rows
        || left.cols != expected_columns
        || right.cols != expected_columns
        || operand_dtype == MetalTensorDType::F32
        || weight.dtype != MetalTensorDType::F16
        || weight.len() != segment_columns
    {
        return Err(format!(
            "segmented RMSNorm shape 不兼容: left=[{},{},{:?}] right=[{},{},{:?}] weight=[{},{}] segments={segments} columns={segment_columns}",
            left.rows, left.cols, left.dtype, right.rows, right.cols, right.dtype, weight.rows, weight.cols,
        ));
    }
    if !eps.is_finite() || eps < 0.0 || !scale.is_finite() {
        return Err(format!("segmented RMSNorm 参数非法: columns={segment_columns} eps={eps} scale={scale}"));
    }

    let rows = validate_u32("segmented RMSNorm rows", left.rows)?;
    let segments_u32 = validate_u32("segmented RMSNorm segments", segments)?;
    let segment_columns_u32 = validate_u32("segmented RMSNorm columns", segment_columns)?;
    let total_columns = validate_u32("segmented RMSNorm total columns", expected_columns)?;
    let thread_count = segment_columns.next_power_of_two().clamp(32, THREADS);
    let simdgroups = validate_u32("segmented RMSNorm simdgroups", thread_count.div_ceil(32))?;
    let groups = left.rows.checked_mul(segments).ok_or("segmented RMSNorm group 数溢出")?;
    // F16 operand 走原生 kernel,免去每层 f16→bf16 的大张量 cast
    let (output, pipeline_name) = if operand_dtype == MetalTensorDType::F16 {
        (ctx.tensor_kernel_output(left.rows, expected_columns), "segmented_rmsnorm_add_scaled_f16")
    } else {
        (ctx.tensor_zeros_bf16(left.rows, expected_columns), "segmented_rmsnorm_add_scaled_bf16")
    };
    let pipeline = ctx.pipeline(pipeline_name)?;
    if pipeline.max_total_threads_per_threadgroup() < thread_count as u64 {
        return Err(format!("segmented RMSNorm 需要至少 {thread_count} threads/threadgroup"));
    }
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&left.buffer), 0);
    encoder.set_buffer(1, Some(&right.buffer), 0);
    encoder.set_buffer(2, Some(&weight.buffer), 0);
    encoder.set_buffer(3, Some(&output.buffer), 0);
    set_bytes(&encoder, 4, &rows);
    set_bytes(&encoder, 5, &segments_u32);
    set_bytes(&encoder, 6, &segment_columns_u32);
    set_bytes(&encoder, 7, &total_columns);
    set_bytes(&encoder, 8, &eps);
    set_bytes(&encoder, 9, &scale);
    set_bytes(&encoder, 10, &simdgroups);
    encoder.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(thread_count as u64, 1, 1));
    encoder.end_encoding();
    let shape = format!("rows={},segments={segments},columns={segment_columns}", left.rows);
    ctx.commit_and_wait_profiled(&command, pipeline_name, &shape, left.buffer.length() + right.buffer.length() + weight.buffer.length(), output.buffer.length());
    Ok(output)
}

pub fn gated_activation_tensor(ctx: &MetalContext, gate: &MetalTensor, up: &MetalTensor, activation: &Activation) -> Result<MetalTensor, String> {
    validate_tensor("gated activation up", up, gate.rows, gate.cols)?;
    if gate.dtype != up.dtype || !matches!(gate.dtype, MetalTensorDType::F16 | MetalTensorDType::Bf16) {
        return Err(format!("gated activation 要求相同的 F16/BF16 输入，实际 gate={:?} up={:?}", gate.dtype, up.dtype));
    }
    let params = GatedActivation::from_spec(activation)?;
    let count = validate_u32("count", gate.len())?;
    let (output, kernel) = match gate.dtype {
        MetalTensorDType::F16 => (ctx.tensor_zeros(gate.rows, gate.cols), "gated_activation_f16"),
        MetalTensorDType::Bf16 => (ctx.tensor_zeros_bf16(gate.rows, gate.cols), "gated_activation_bf16"),
        MetalTensorDType::F32 => unreachable!(),
    };
    let shape = format!("elements={}", gate.len());
    launch_1d(ctx, kernel, &shape, gate.len(), gate.buffer.length() + up.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&gate.buffer), 0);
        encoder.set_buffer(1, Some(&up.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &count);
        set_bytes(encoder, 4, &params.kind);
        set_bytes(encoder, 5, &params.alpha);
        set_bytes(encoder, 6, &params.limit);
    })?;
    Ok(output)
}

pub fn sigmoid_gate_tensor(ctx: &MetalContext, input: &MetalTensor, gate: &MetalTensor) -> Result<MetalTensor, String> {
    if input.rows != gate.rows || (gate.cols != 1 && gate.cols != input.cols) {
        return Err(format!("sigmoid gate shape input=[{},{}], gate=[{},{}]", input.rows, input.cols, gate.rows, gate.cols));
    }
    let columns = validate_u32("sigmoid gate columns", input.cols)?;
    let gate_columns = validate_u32("sigmoid gate columns", gate.cols)?;
    let count = validate_u32("sigmoid gate count", input.len())?;
    let (output, kernel) = if input.dtype == MetalTensorDType::F16 && gate.dtype == MetalTensorDType::F32 {
        (ctx.tensor_kernel_output_f32(input.rows, input.cols), "sigmoid_gate_f16_f32")
    } else if input.dtype == MetalTensorDType::F16 && gate.dtype == MetalTensorDType::F16 {
        (ctx.tensor_kernel_output(input.rows, input.cols), "sigmoid_gate_f16")
    } else {
        return Err(format!("sigmoid gate dtype 不支持: input={:?}, gate={:?}", input.dtype, gate.dtype));
    };
    let shape = format!("input=[{},{}],gate_cols={}", input.rows, input.cols, gate.cols);
    launch_1d(ctx, kernel, &shape, input.len(), input.buffer.length() + gate.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&gate.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &gate_columns);
        set_bytes(encoder, 5, &count);
    })?;
    Ok(output)
}

pub fn softplus_gate_scaled_tensor(ctx: &MetalContext, input: &MetalTensor, gate: &MetalTensor, gate_scale: f32, output_scale: f32) -> Result<MetalTensor, String> {
    if input.rows != gate.rows || gate.cols == 0 || !input.cols.is_multiple_of(gate.cols) {
        return Err(format!("scaled softplus gate shape input=[{},{}], gate=[{},{}] 不兼容", input.rows, input.cols, gate.rows, gate.cols));
    }
    if input.dtype != MetalTensorDType::F16 || gate.dtype != MetalTensorDType::F16 || !gate_scale.is_finite() || !output_scale.is_finite() {
        return Err(format!("scaled softplus gate 要求 F16 与有限缩放，input={:?} gate={:?} scales={gate_scale}/{output_scale}", input.dtype, gate.dtype));
    }
    let columns = validate_u32("scaled softplus columns", input.cols)?;
    let gate_columns = validate_u32("scaled softplus gate columns", gate.cols)?;
    let count = validate_u32("scaled softplus count", input.len())?;
    let output = ctx.tensor_kernel_output(input.rows, input.cols);
    let shape = format!("input=[{},{}],gate_cols={},scales={gate_scale}/{output_scale}", input.rows, input.cols, gate.cols);
    launch_1d(ctx, "softplus_gate_scaled_f16", &shape, input.len(), input.buffer.length() + gate.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&gate.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &gate_columns);
        set_bytes(encoder, 5, &count);
        set_bytes(encoder, 6, &gate_scale);
        set_bytes(encoder, 7, &output_scale);
    })?;
    Ok(output)
}

#[cfg(test)]
mod mixed_sigmoid_gate_tests {
    use super::*;

    #[test]
    fn f16_value_f32_gate_keeps_finite_values() {
        let ctx = MetalContext::new_default().unwrap();
        let input = ctx.tensor_from_f32(&[2.0, -4.0], 1, 2).unwrap();
        let gate = ctx.tensor_from_f32_preserve(&[0.0, 0.0], 1, 2).unwrap();
        let output = sigmoid_gate_tensor(&ctx, &input, &gate).unwrap();
        assert_eq!(output.dtype, MetalTensorDType::F32);
        assert_eq!(ctx.tensor_to_f32(&output), vec![1.0, -2.0]);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_net_tensor(
    ctx: &MetalContext,
    qkv: &MetalTensor,
    z: &MetalTensor,
    alpha: &MetalTensor,
    beta: &MetalTensor,
    conv_weight: &MetalTensor,
    a_log: &metal::Buffer,
    a_log_len: usize,
    dt_bias: &metal::Buffer,
    dt_bias_len: usize,
    norm_weight: &metal::Buffer,
    norm_weight_len: usize,
    conv_state: &metal::Buffer,
    recurrent_state: &metal::Buffer,
    spec: &GatedDeltaNetSpec,
) -> Result<MetalTensor, String> {
    // chunked parallel 路径(2026-08 batched GEMM 重写)真机 A/B 仍慢于 simd8
    // (+15~34% GDN GPU,见 docs/metal-qwen38-prefill-lessons.md §3),默认关闭;
    // ZLLM_GDN_CHUNKED=1 是诊断/复测开关。
    let enable_chunked = std::env::var_os("ZLLM_GDN_CHUNKED").is_some();
    gated_delta_net_tensor_impl(ctx, qkv, z, alpha, beta, conv_weight, a_log, a_log_len, dt_bias, dt_bias_len, norm_weight, norm_weight_len, conv_state, recurrent_state, spec, enable_chunked)
}

/// `enable_chunked=false` 时强制 simd8 串行路径,供一致性测试当 oracle。
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_net_tensor_impl(
    ctx: &MetalContext,
    qkv: &MetalTensor,
    z: &MetalTensor,
    alpha: &MetalTensor,
    beta: &MetalTensor,
    conv_weight: &MetalTensor,
    a_log: &metal::Buffer,
    a_log_len: usize,
    dt_bias: &metal::Buffer,
    dt_bias_len: usize,
    norm_weight: &metal::Buffer,
    norm_weight_len: usize,
    conv_state: &metal::Buffer,
    recurrent_state: &metal::Buffer,
    spec: &GatedDeltaNetSpec,
    enable_chunked: bool,
) -> Result<MetalTensor, String> {
    spec.validate()?;
    if qkv.rows == 0 || qkv.cols != spec.conv_dim() {
        return Err(format!("Metal Gated DeltaNet qkv=[{},{}]，期望 [rows,{}]", qkv.rows, qkv.cols, spec.conv_dim()));
    }
    validate_tensor("Gated DeltaNet z", z, qkv.rows, spec.value_dim())?;
    validate_tensor("Gated DeltaNet alpha", alpha, qkv.rows, spec.value_heads)?;
    validate_tensor("Gated DeltaNet beta", beta, qkv.rows, spec.value_heads)?;
    validate_tensor("Gated DeltaNet conv weight", conv_weight, spec.conv_dim(), spec.conv_kernel)?;
    if a_log_len != spec.value_heads || dt_bias_len != spec.value_heads || norm_weight_len != spec.value_head_dim {
        return Err(format!("Metal Gated DeltaNet weight shape 异常: A={a_log_len}, dt={dt_bias_len}, norm={norm_weight_len}"));
    }
    // 长度参数只核对元素数，kernel 按 f32 读取，buffer 实际字节数必须够。
    for (name, buffer, len) in [("a_log", a_log, a_log_len), ("dt_bias", dt_bias, dt_bias_len), ("norm_weight", norm_weight, norm_weight_len)] {
        let needed = len as u64 * mem::size_of::<f32>() as u64;
        if buffer.length() < needed {
            return Err(format!("Metal Gated DeltaNet {name} buffer 过小: {}B，需 {needed}B({len} 个 f32)", buffer.length()));
        }
    }
    if conv_state.length() as usize != spec.conv_state_elements() * mem::size_of::<f32>() || recurrent_state.length() as usize != spec.recurrent_elements() * mem::size_of::<f32>() {
        return Err("Metal Gated DeltaNet state buffer 大小与 spec 不一致".to_owned());
    }

    // Gated DeltaNet shader 的主数据面是 F16；控制投影可在此前保持 F32 累加，边界只转换一次。
    let qkv = super::to_f16_tensor(ctx, qkv)?;
    let z = super::to_f16_tensor(ctx, z)?;
    let alpha = super::to_f16_tensor(ctx, alpha)?;
    let beta = super::to_f16_tensor(ctx, beta)?;
    let conv_weight = super::to_f16_tensor(ctx, conv_weight)?;

    let rows = validate_u32("Gated DeltaNet rows", qkv.rows)?;
    let key_heads = validate_u32("Gated DeltaNet key heads", spec.key_heads)?;
    let value_heads = validate_u32("Gated DeltaNet value heads", spec.value_heads)?;
    let key_head_dim = validate_u32("Gated DeltaNet key head dim", spec.key_head_dim)?;
    let value_head_dim = validate_u32("Gated DeltaNet value head dim", spec.value_head_dim)?;
    let conv_dim = validate_u32("Gated DeltaNet conv dim", spec.conv_dim())?;
    let conv_kernel = validate_u32("Gated DeltaNet conv kernel", spec.conv_kernel)?;
    let recurrent_threads = spec.value_head_dim.next_power_of_two();
    if recurrent_threads > THREADS {
        return Err(format!("Metal Gated DeltaNet value_head_dim={} 超过 {THREADS}", spec.value_head_dim));
    }

    let mixed = ctx.tensor_kernel_output(qkv.rows, spec.conv_dim());
    let core = ctx.tensor_kernel_output(qkv.rows, spec.value_dim());
    let output = ctx.tensor_kernel_output(qkv.rows, spec.value_dim());
    // 单个 SIMDgroup 同时持有 8 列 state；Q/K 先归一化，memory/update 共用同一 F32 值。
    // bulk 路径(conv 批量 + controls + simd8 recurrent)与行数无关;simd8 比 legacy
    // 逐 token 路径并行度高 16 倍(value_head_dim/8 组),短 prompt 也受益。decode
    // 的单 token 输入保持 legacy 单次路径,不承担批量 kernel 的固定开销。
    let simd8_capable = spec.key_head_dim <= 128
        && spec.key_head_dim.is_multiple_of(32)
        && spec.value_head_dim.is_multiple_of(8)
        && spec.value_dim() * std::mem::size_of::<half::f16>() >= (spec.key_heads * 2 + spec.value_heads) * std::mem::size_of::<f32>();
    let bulk_prefill = qkv.rows > 1 && (simd8_capable || qkv.rows >= 512);
    let simd8_prefill = simd8_capable && bulk_prefill;
    let command = ctx.command_buffer();

    let conv_pipeline = ctx.pipeline(if bulk_prefill { "gated_delta_conv_prefill_f16" } else { "gated_delta_conv_f16" })?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&conv_pipeline);
    encoder.set_buffer(0, Some(&qkv.buffer), 0);
    encoder.set_buffer(1, Some(&conv_weight.buffer), 0);
    encoder.set_buffer(2, Some(conv_state), 0);
    encoder.set_buffer(3, Some(&mixed.buffer), 0);
    set_bytes(&encoder, 4, &rows);
    set_bytes(&encoder, 5, &conv_dim);
    set_bytes(&encoder, 6, &conv_kernel);
    let conv_threads = if bulk_prefill { qkv.rows * spec.conv_dim() } else { spec.conv_dim() };
    encoder.dispatch_threads(MTLSize::new(conv_threads as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
    encoder.end_encoding();

    if bulk_prefill {
        let state_pipeline = ctx.pipeline("gated_delta_conv_state_f16")?;
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&state_pipeline);
        encoder.set_buffer(0, Some(&qkv.buffer), 0);
        encoder.set_buffer(1, Some(conv_state), 0);
        set_bytes(&encoder, 2, &rows);
        set_bytes(&encoder, 3, &conv_dim);
        set_bytes(&encoder, 4, &conv_kernel);
        encoder.dispatch_threads(MTLSize::new(spec.conv_dim() as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
        encoder.end_encoding();
    }

    if simd8_prefill {
        let controls_pipeline = ctx.pipeline("gated_delta_prefill_controls_f16")?;
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&controls_pipeline);
        encoder.set_buffer(0, Some(&mixed.buffer), 0);
        encoder.set_buffer(1, Some(&alpha.buffer), 0);
        encoder.set_buffer(2, Some(a_log), 0);
        encoder.set_buffer(3, Some(dt_bias), 0);
        encoder.set_buffer(4, Some(&output.buffer), 0);
        set_bytes(&encoder, 5, &rows);
        set_bytes(&encoder, 6, &key_heads);
        set_bytes(&encoder, 7, &value_heads);
        set_bytes(&encoder, 8, &key_head_dim);
        set_bytes(&encoder, 9, &value_head_dim);
        encoder.dispatch_threads(MTLSize::new((qkv.rows * spec.key_heads.max(spec.value_heads)) as u64, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
        encoder.end_encoding();
    }

    // 长序列走 chunked parallel:WY 变换全 chunk 并行,scan 为每 chunk 4 步跨 head
    // batched GEMM。scratch 为 f16(28k 序列约 860MB,批结束即释放),短序列仍用
    // simd8 串行。
    let chunked_prefill = enable_chunked && simd8_prefill && qkv.rows >= 512 && spec.key_head_dim <= 128 && spec.value_head_dim <= 128;
    if chunked_prefill {
        let chunk: u64 = 64;
        let chunks = (qkv.rows as u64 + chunk - 1) / chunk;
        let heads = spec.value_heads as u64;
        let scratch_decay = ctx.cached_zero_buffer("gdn_decay", (chunks * heads * chunk * spec.key_head_dim as u64 * 2) as usize);
        let scratch_cumsum = ctx.cached_zero_buffer("gdn_cumsum", (chunks * heads * chunk * spec.value_head_dim as u64 * 2) as usize);
        let scratch_b = ctx.cached_zero_buffer("gdn_b", (chunks * heads * chunk * 4) as usize);
        let scratch_qk = ctx.cached_zero_buffer("gdn_qk", (chunks * heads * chunk * chunk * 2) as usize);
        let transform_pipeline = ctx.pipeline("gdn_chunk_transform_f16")?;
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&transform_pipeline);
        encoder.set_buffer(0, Some(&mixed.buffer), 0);
        encoder.set_buffer(1, Some(&beta.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        encoder.set_buffer(3, Some(&scratch_decay), 0);
        encoder.set_buffer(4, Some(&scratch_cumsum), 0);
        encoder.set_buffer(5, Some(&scratch_b), 0);
        set_bytes(&encoder, 6, &rows);
        set_bytes(&encoder, 7, &key_heads);
        set_bytes(&encoder, 8, &value_heads);
        set_bytes(&encoder, 9, &key_head_dim);
        set_bytes(&encoder, 10, &value_head_dim);
        encoder.dispatch_thread_groups(MTLSize::new(chunks, heads, 1), MTLSize::new(THREADS as u64, 1, 1));
        encoder.end_encoding();

        // scan:每 chunk 4 个全 GPU kernel(qk → v_new → output → state),chunk 间
        // 依赖由同一 encoder 内的 dispatch 顺序保证;不用单 kernel 内 barrier 承担
        // chunk 串行(旧实现两次实测慢于 simd8 的根因)。全部 dispatch 共用一个
        // encoder:边界不触发 GPU 排空,buffer 绑定只设一次。
        // controls 复用 output 槽(norm_gate 在 scan 之后才覆写它)。
        let qk_pipeline = ctx.pipeline("gdn_chunk_qk_f16")?;
        let vnew_pipeline = ctx.pipeline("gdn_chunk_vnew_f16")?;
        let output_pipeline = ctx.pipeline("gdn_chunk_output_f16")?;
        let state_pipeline = ctx.pipeline("gdn_chunk_state_f16")?;
        let value_tiles = (spec.value_head_dim as u64 + 31) / 32;
        let key_tiles = (spec.key_head_dim as u64 + 31) / 32;
        let scan_encoder = command.new_compute_command_encoder();
        for chunk_index in 0..chunks as u32 {
            scan_encoder.set_compute_pipeline_state(&qk_pipeline);
            scan_encoder.set_buffer(0, Some(&mixed.buffer), 0);
            scan_encoder.set_buffer(1, Some(&output.buffer), 0);
            scan_encoder.set_buffer(2, Some(&scratch_b), 0);
            scan_encoder.set_buffer(3, Some(&scratch_qk), 0);
            set_bytes(&scan_encoder, 4, &rows);
            set_bytes(&scan_encoder, 5, &key_heads);
            set_bytes(&scan_encoder, 6, &value_heads);
            set_bytes(&scan_encoder, 7, &key_head_dim);
            set_bytes(&scan_encoder, 8, &value_head_dim);
            set_bytes(&scan_encoder, 9, &chunk_index);
            scan_encoder.dispatch_thread_groups(MTLSize::new(2, 2, heads), MTLSize::new(128, 1, 1));

            scan_encoder.set_compute_pipeline_state(&vnew_pipeline);
            scan_encoder.set_buffer(0, Some(&scratch_decay), 0);
            scan_encoder.set_buffer(1, Some(recurrent_state), 0);
            scan_encoder.set_buffer(2, Some(&scratch_cumsum), 0);
            set_bytes(&scan_encoder, 3, &value_heads);
            set_bytes(&scan_encoder, 4, &key_head_dim);
            set_bytes(&scan_encoder, 5, &value_head_dim);
            set_bytes(&scan_encoder, 6, &chunk_index);
            scan_encoder.dispatch_thread_groups(MTLSize::new(value_tiles, 2, heads), MTLSize::new(128, 1, 1));

            scan_encoder.set_compute_pipeline_state(&output_pipeline);
            scan_encoder.set_buffer(0, Some(&mixed.buffer), 0);
            scan_encoder.set_buffer(1, Some(&output.buffer), 0);
            scan_encoder.set_buffer(2, Some(&scratch_b), 0);
            scan_encoder.set_buffer(3, Some(&scratch_qk), 0);
            scan_encoder.set_buffer(4, Some(&scratch_cumsum), 0);
            scan_encoder.set_buffer(5, Some(recurrent_state), 0);
            scan_encoder.set_buffer(6, Some(&core.buffer), 0);
            set_bytes(&scan_encoder, 7, &rows);
            set_bytes(&scan_encoder, 8, &key_heads);
            set_bytes(&scan_encoder, 9, &value_heads);
            set_bytes(&scan_encoder, 10, &key_head_dim);
            set_bytes(&scan_encoder, 11, &value_head_dim);
            set_bytes(&scan_encoder, 12, &chunk_index);
            scan_encoder.dispatch_thread_groups(MTLSize::new(value_tiles, 2, heads), MTLSize::new(128, 1, 1));

            scan_encoder.set_compute_pipeline_state(&state_pipeline);
            scan_encoder.set_buffer(0, Some(&mixed.buffer), 0);
            scan_encoder.set_buffer(1, Some(&output.buffer), 0);
            scan_encoder.set_buffer(2, Some(&scratch_b), 0);
            scan_encoder.set_buffer(3, Some(&scratch_cumsum), 0);
            scan_encoder.set_buffer(4, Some(recurrent_state), 0);
            set_bytes(&scan_encoder, 5, &rows);
            set_bytes(&scan_encoder, 6, &key_heads);
            set_bytes(&scan_encoder, 7, &value_heads);
            set_bytes(&scan_encoder, 8, &key_head_dim);
            set_bytes(&scan_encoder, 9, &value_head_dim);
            set_bytes(&scan_encoder, 10, &chunk_index);
            scan_encoder.dispatch_thread_groups(MTLSize::new(value_tiles, key_tiles, heads), MTLSize::new(128, 1, 1));
        }
        scan_encoder.end_encoding();
    } else if simd8_prefill {
        let recurrent_pipeline = ctx.pipeline("gated_delta_recurrent_prefill_simd8_f16")?;
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&recurrent_pipeline);
        encoder.set_buffer(0, Some(&mixed.buffer), 0);
        encoder.set_buffer(2, Some(&beta.buffer), 0);
        encoder.set_buffer(5, Some(recurrent_state), 0);
        encoder.set_buffer(6, Some(&core.buffer), 0);
        set_bytes(&encoder, 7, &rows);
        set_bytes(&encoder, 8, &key_heads);
        set_bytes(&encoder, 9, &value_heads);
        set_bytes(&encoder, 10, &key_head_dim);
        set_bytes(&encoder, 11, &value_head_dim);
        encoder.set_buffer(12, Some(&output.buffer), 0);
        encoder.dispatch_thread_groups(MTLSize::new((spec.value_heads * spec.value_head_dim.div_ceil(8)) as u64, 1, 1), MTLSize::new(32, 1, 1));
        encoder.end_encoding();
    } else {
        // decode 取向 rows kernel:state 驻留寄存器 + lanes-on-keys 归约,要求
        // key_head_dim 整除 32 且 <=128(寄存器 ls[4])、value_head_dim 整除 4;
        // 不满足回退 legacy 逐列 kernel。
        let rows_capable = spec.key_head_dim % 32 == 0 && spec.key_head_dim <= 128 && spec.value_head_dim % 4 == 0;
        let recurrent_pipeline = if rows_capable { ctx.pipeline_u32_constants("gated_delta_recurrent_rows_f16", &[(spec.key_head_dim / 32) as u32])? } else { ctx.pipeline("gated_delta_recurrent_legacy_f16")? };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&recurrent_pipeline);
        encoder.set_buffer(0, Some(&mixed.buffer), 0);
        encoder.set_buffer(1, Some(&alpha.buffer), 0);
        encoder.set_buffer(2, Some(&beta.buffer), 0);
        encoder.set_buffer(3, Some(a_log), 0);
        encoder.set_buffer(4, Some(dt_bias), 0);
        encoder.set_buffer(5, Some(recurrent_state), 0);
        encoder.set_buffer(6, Some(&core.buffer), 0);
        set_bytes(&encoder, 7, &rows);
        set_bytes(&encoder, 8, &key_heads);
        set_bytes(&encoder, 9, &value_heads);
        set_bytes(&encoder, 10, &key_head_dim);
        set_bytes(&encoder, 11, &value_head_dim);
        if rows_capable {
            encoder.dispatch_thread_groups(MTLSize::new((spec.value_head_dim / 4) as u64, spec.value_heads as u64, 1), MTLSize::new(32, 4, 1));
        } else {
            encoder.dispatch_thread_groups(MTLSize::new(spec.value_heads as u64, 1, 1), MTLSize::new(recurrent_threads as u64, 1, 1));
        }
        // decode 路径(recurrent→norm_gate 有数据依赖):memory_barrier 后同 encoder 接续 norm_gate,
        // 省 1 个 encoder 边界;bulk/simd8/chunked 路径仍走独立 encoder
        if !simd8_prefill && !chunked_prefill && !bulk_prefill {
            encoder.memory_barrier();
            let norm_pipeline = ctx.pipeline("gated_delta_norm_gate_f16")?;
            encoder.set_compute_pipeline_state(&norm_pipeline);
            encoder.set_buffer(0, Some(&core.buffer), 0);
            encoder.set_buffer(1, Some(&z.buffer), 0);
            encoder.set_buffer(2, Some(norm_weight), 0);
            encoder.set_buffer(3, Some(&output.buffer), 0);
            set_bytes(&encoder, 4, &value_heads);
            set_bytes(&encoder, 5, &value_head_dim);
            set_bytes(&encoder, 6, &spec.rms_eps);
            encoder.dispatch_thread_groups(MTLSize::new((qkv.rows * spec.value_heads) as u64, 1, 1), MTLSize::new(recurrent_threads as u64, 1, 1));
            encoder.end_encoding();
        } else {
            encoder.end_encoding();
        }
    }

    if simd8_prefill || chunked_prefill || bulk_prefill {
        let norm_pipeline = ctx.pipeline("gated_delta_norm_gate_f16")?;
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&norm_pipeline);
        encoder.set_buffer(0, Some(&core.buffer), 0);
        encoder.set_buffer(1, Some(&z.buffer), 0);
        encoder.set_buffer(2, Some(norm_weight), 0);
        encoder.set_buffer(3, Some(&output.buffer), 0);
        set_bytes(&encoder, 4, &value_heads);
        set_bytes(&encoder, 5, &value_head_dim);
        set_bytes(&encoder, 6, &spec.rms_eps);
        encoder.dispatch_thread_groups(MTLSize::new((qkv.rows * spec.value_heads) as u64, 1, 1), MTLSize::new(recurrent_threads as u64, 1, 1));
        encoder.end_encoding();
    }

    let shape = format!(
        "rows={},kh={},vh={},kd={},vd={},conv={},parallel_conv={bulk_prefill},ordered_recurrent={simd8_prefill},chunked={chunked_prefill}",
        qkv.rows, spec.key_heads, spec.value_heads, spec.key_head_dim, spec.value_head_dim, spec.conv_kernel
    );
    ctx.commit_and_wait_profiled(
        &command,
        "gated_delta_net_f16",
        &shape,
        qkv.buffer.length() + z.buffer.length() + alpha.buffer.length() + beta.buffer.length() + conv_weight.buffer.length() + a_log.length() + dt_bias.length() + norm_weight.length() + conv_state.length() + recurrent_state.length(),
        mixed.buffer.length() + core.buffer.length() + output.buffer.length() + conv_state.length() + recurrent_state.length(),
    );
    Ok(output)
}

/// F16 张量逐元素乘标量(gemma4 MTP layer_output_scale)。
pub fn mul_scalar_f16_tensor(ctx: &MetalContext, a: &MetalTensor, scale: f32) -> Result<MetalTensor, String> {
    if a.dtype != MetalTensorDType::F16 {
        return Err(format!("mul_scalar 需要 F16，实际 {:?}", a.dtype));
    }
    let count = validate_u32("count", a.len())?;
    let output = ctx.tensor_kernel_output(a.rows, a.cols);
    let shape = format!("elements={}", a.len());
    launch_1d(ctx, "mul_scalar_f16", &shape, count as usize, a.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&a.buffer), 0);
        set_bytes(&encoder, 1, &scale);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(&encoder, 3, &count);
    })?;
    Ok(output)
}

pub fn add_tensor(ctx: &MetalContext, a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor, String> {
    validate_tensor("add rhs", b, a.rows, a.cols)?;
    let count = validate_u32("count", a.len())?;
    let output = match (a.dtype, b.dtype) {
        (MetalTensorDType::Bf16, MetalTensorDType::Bf16) => ctx.tensor_kernel_output_bf16(a.rows, a.cols),
        (MetalTensorDType::F32, _) | (_, MetalTensorDType::F32) => ctx.tensor_kernel_output_f32(a.rows, a.cols),
        _ => ctx.tensor_kernel_output(a.rows, a.cols),
    };
    let shape = format!("elements={}", a.len());
    let (pipeline, total): (&str, u32) = match (a.dtype, b.dtype) {
        (MetalTensorDType::F16, MetalTensorDType::F16) if count.is_multiple_of(8) && count >= 4096 => ("add_f16_vec8", count / 8),
        (MetalTensorDType::F16, MetalTensorDType::F16) => ("add_f16", count),
        (MetalTensorDType::Bf16, MetalTensorDType::Bf16) => ("add_bf16", count),
        (MetalTensorDType::F32, MetalTensorDType::F16) => ("add_f32_f16", count),
        (MetalTensorDType::F16, MetalTensorDType::F32) => ("add_f16_f32", count),
        (MetalTensorDType::F32, MetalTensorDType::F32) => ("add_f32", count),
        _ => return Err(format!("add dtype 不兼容: {:?}/{:?}", a.dtype, b.dtype)),
    };
    launch_1d(ctx, pipeline, &shape, total as usize, a.buffer.length() + b.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&a.buffer), 0);
        encoder.set_buffer(1, Some(&b.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &count);
    })?;
    Ok(output)
}

pub fn add_scaled_tensor(ctx: &MetalContext, a: &MetalTensor, b: &MetalTensor, scale: f32) -> Result<MetalTensor, String> {
    validate_tensor("add_scaled rhs", b, a.rows, a.cols)?;
    if !scale.is_finite() {
        return Err(format!("add_scaled scale={scale} 非法"));
    }
    let count = validate_u32("add_scaled count", a.len())?;
    let output = match (a.dtype, b.dtype) {
        (MetalTensorDType::Bf16, MetalTensorDType::Bf16) => ctx.tensor_kernel_output_bf16(a.rows, a.cols),
        (MetalTensorDType::F32, _) | (_, MetalTensorDType::F32) => ctx.tensor_kernel_output_f32(a.rows, a.cols),
        _ => ctx.tensor_kernel_output(a.rows, a.cols),
    };
    let shape = format!("elements={},scale={scale}", a.len());
    let pipeline = match (a.dtype, b.dtype) {
        (MetalTensorDType::F16, MetalTensorDType::F16) => "add_scaled_f16",
        (MetalTensorDType::Bf16, MetalTensorDType::Bf16) => "add_scaled_bf16",
        (MetalTensorDType::F32, MetalTensorDType::F16) => "add_scaled_f32_f16",
        (MetalTensorDType::F16, MetalTensorDType::F32) => "add_scaled_f16_f32",
        (MetalTensorDType::F32, MetalTensorDType::F32) => "add_scaled_f32",
        _ => return Err(format!("add_scaled dtype 不兼容: {:?}/{:?}", a.dtype, b.dtype)),
    };
    launch_1d(ctx, pipeline, &shape, a.len(), a.buffer.length() + b.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&a.buffer), 0);
        encoder.set_buffer(1, Some(&b.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &scale);
        set_bytes(encoder, 4, &count);
    })?;
    Ok(output)
}

pub fn add_bias_tensor(ctx: &MetalContext, input: &MetalTensor, bias: &MetalTensor) -> Result<MetalTensor, String> {
    if bias.len() != input.cols {
        return Err(format!("bias 长度 {} 与 input columns {} 不符", bias.len(), input.cols));
    }
    let count = validate_u32("bias add count", input.len())?;
    let columns = validate_u32("bias add columns", input.cols)?;
    if bias.dtype != MetalTensorDType::F16 {
        return Err(format!("bias dtype={:?}，期望 F16", bias.dtype));
    }
    let (output, pipeline) = match input.dtype {
        MetalTensorDType::F16 => (ctx.tensor_kernel_output(input.rows, input.cols), "add_bias_f16"),
        MetalTensorDType::Bf16 => (ctx.tensor_kernel_output_bf16(input.rows, input.cols), "add_bias_bf16_f16"),
        MetalTensorDType::F32 => return Err("add_bias 暂不支持 F32 activation".to_owned()),
    };
    let shape = format!("input=[{},{}]", input.rows, input.cols);
    launch_1d(ctx, pipeline, &shape, input.len(), input.buffer.length() + bias.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&bias.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_bytes(encoder, 3, &columns);
        set_bytes(encoder, 4, &count);
    })?;
    Ok(output)
}

pub fn gelu_tensor(ctx: &MetalContext, input: &MetalTensor) -> Result<MetalTensor, String> {
    let count = validate_u32("GELU count", input.len())?;
    let output = ctx.tensor_kernel_output(input.rows, input.cols);
    let shape = format!("input=[{},{}]", input.rows, input.cols);
    launch_1d(ctx, "gelu_f16", &shape, input.len(), input.buffer.length(), output.buffer.length(), |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_bytes(encoder, 2, &count);
    })?;
    Ok(output)
}

#[cfg(test)]
mod f32_control_norm_tests {
    use super::*;

    #[test]
    fn f32_weight_rmsnorm_matches_cpu() {
        let ctx = MetalContext::new_default().unwrap();
        let input_values = [0.25_f32, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0];
        let weight_values = [0.75_f32, 1.0, 1.25, 1.5];
        let input = ctx.tensor_from_f32_preserve(&input_values, 2, 4).unwrap();
        let weight_bytes = unsafe { std::slice::from_raw_parts(weight_values.as_ptr().cast::<u8>(), std::mem::size_of_val(&weight_values)) };
        let weight = ctx.shared_buffer(weight_bytes);
        let actual = rmsnorm_tensor_resident_f32_weight(&ctx, &input, &weight, weight_values.len(), 1.0e-6, 0.0).unwrap();
        let actual = ctx.tensor_to_f32(&actual);
        for (row, values) in input_values.chunks_exact(4).enumerate() {
            let inv_rms = (values.iter().map(|value| value * value).sum::<f32>() / 4.0 + 1.0e-6).sqrt().recip();
            for column in 0..4 {
                let expected = values[column] * inv_rms * weight_values[column];
                assert!((actual[row * 4 + column] - expected).abs() <= 1.0e-5, "row={row} column={column} actual={} expected={expected}", actual[row * 4 + column]);
            }
        }
    }
}

#[cfg(test)]
mod gdn_chunked_tests {
    use super::*;

    fn pseudo_random(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*seed >> 8) as f32 / 8388608.0 - 0.5) * 0.4
    }

    /// chunked parallel 与已被 e2e 验证的 simd8 串行路径逐元素一致(输出与最终
    /// recurrent state)。两组形状:128/128 全整 tile;96/72 覆盖非整 32 列的
    /// masking 路径与 3×3 kd tile。
    #[test]
    fn gdn_chunked_matches_serial_simd8() {
        // 600 行 = 9 个整 chunk + 24 行尾 chunk;520 行 = 8 个整 chunk + 8 行尾。
        run_chunked_consistency(GatedDeltaNetSpec { key_heads: 2, value_heads: 4, key_head_dim: 128, value_head_dim: 128, conv_kernel: 4, rms_eps: 1.0e-5, output_gate: crate::attention::gated_delta_net::GdnOutputGate::Silu }, 600);
        run_chunked_consistency(GatedDeltaNetSpec { key_heads: 2, value_heads: 6, key_head_dim: 96, value_head_dim: 72, conv_kernel: 4, rms_eps: 1.0e-5, output_gate: crate::attention::gated_delta_net::GdnOutputGate::Silu }, 520);
    }

    /// decode 的逐 token 路径(rows kernel 或 legacy 回退)与 simd8 批量 oracle
    /// 一致:N 次 rows=1 顺序调用(conv/recurrent state 跨调用延续)对比一次
    /// rows=N simd8 调用,输出与最终 state 都必须在容差内。
    /// 128/128 走 rows kernel;96/72 走 legacy 回退(两条 decode kernel 都覆盖)。
    #[test]
    fn gdn_legacy_decode_matches_simd8() {
        run_decode_consistency(GatedDeltaNetSpec { key_heads: 2, value_heads: 4, key_head_dim: 128, value_head_dim: 128, conv_kernel: 4, rms_eps: 1.0e-5, output_gate: crate::attention::gated_delta_net::GdnOutputGate::Silu });
        run_decode_consistency(GatedDeltaNetSpec { key_heads: 2, value_heads: 6, key_head_dim: 96, value_head_dim: 72, conv_kernel: 4, rms_eps: 1.0e-5, output_gate: crate::attention::gated_delta_net::GdnOutputGate::Silu });
    }

    fn run_decode_consistency(spec: GatedDeltaNetSpec) {
        let rows = 8usize;
        let ctx = MetalContext::new_default().unwrap();
        spec.validate().unwrap();
        let conv_dim = spec.conv_dim();
        let value_dim = spec.value_dim();
        let mut seed = 0x5eed_77u32;
        let qkv_values: Vec<f32> = (0..rows * conv_dim).map(|_| pseudo_random(&mut seed)).collect();
        let z_values: Vec<f32> = (0..rows * value_dim).map(|_| pseudo_random(&mut seed)).collect();
        let alpha_values: Vec<f32> = (0..rows * spec.value_heads).map(|_| pseudo_random(&mut seed)).collect();
        let beta_values: Vec<f32> = (0..rows * spec.value_heads).map(|_| pseudo_random(&mut seed) + 0.6).collect();
        let conv_values: Vec<f32> = (0..conv_dim * spec.conv_kernel).map(|_| pseudo_random(&mut seed)).collect();
        let a_log_values: Vec<f32> = vec![(0.3f32).ln(); spec.value_heads];
        let dt_bias_values: Vec<f32> = vec![0.1; spec.value_heads];
        let norm_values: Vec<f32> = vec![1.0; spec.value_head_dim];
        let conv_weight = ctx.tensor_from_f32(&conv_values, conv_dim, spec.conv_kernel).unwrap();
        let a_log = ctx.shared_buffer(&a_log_values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dt_bias = ctx.shared_buffer(&dt_bias_values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let norm_weight = ctx.shared_buffer(&norm_values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());

        // simd8 oracle:一次 rows=N
        let simd8_conv_state = ctx.shared_buffer_zeros(spec.conv_state_elements() * 4);
        let simd8_recurrent = ctx.shared_buffer_zeros(spec.recurrent_elements() * 4);
        let simd8 = {
            let qkv = ctx.tensor_from_f32(&qkv_values, rows, conv_dim).unwrap();
            let z = ctx.tensor_from_f32(&z_values, rows, value_dim).unwrap();
            let alpha = ctx.tensor_from_f32(&alpha_values, rows, spec.value_heads).unwrap();
            let beta = ctx.tensor_from_f32(&beta_values, rows, spec.value_heads).unwrap();
            let output =
                gated_delta_net_tensor_impl(&ctx, &qkv, &z, &alpha, &beta, &conv_weight, &a_log, spec.value_heads, &dt_bias, spec.value_heads, &norm_weight, spec.value_head_dim, &simd8_conv_state, &simd8_recurrent, &spec, false).unwrap();
            ctx.read_f16_to_f32(&output.buffer, rows * value_dim)
        };

        // legacy decode:N 次 rows=1,state 跨调用延续
        let legacy_conv_state = ctx.shared_buffer_zeros(spec.conv_state_elements() * 4);
        let legacy_recurrent = ctx.shared_buffer_zeros(spec.recurrent_elements() * 4);
        let mut legacy = Vec::with_capacity(rows * value_dim);
        for row in 0..rows {
            let qkv = ctx.tensor_from_f32(&qkv_values[row * conv_dim..(row + 1) * conv_dim], 1, conv_dim).unwrap();
            let z = ctx.tensor_from_f32(&z_values[row * value_dim..(row + 1) * value_dim], 1, value_dim).unwrap();
            let alpha = ctx.tensor_from_f32(&alpha_values[row * spec.value_heads..(row + 1) * spec.value_heads], 1, spec.value_heads).unwrap();
            let beta = ctx.tensor_from_f32(&beta_values[row * spec.value_heads..(row + 1) * spec.value_heads], 1, spec.value_heads).unwrap();
            let output =
                gated_delta_net_tensor_impl(&ctx, &qkv, &z, &alpha, &beta, &conv_weight, &a_log, spec.value_heads, &dt_bias, spec.value_heads, &norm_weight, spec.value_head_dim, &legacy_conv_state, &legacy_recurrent, &spec, false).unwrap();
            legacy.extend_from_slice(&ctx.read_f16_to_f32(&output.buffer, value_dim));
        }
        ctx.synchronize();
        for index in 0..legacy.len() {
            let difference = (legacy[index] - simd8[index]).abs();
            assert!(difference < 5.0e-2 * (1.0 + simd8[index].abs()), "output index={index}: legacy={} simd8={}", legacy[index], simd8[index]);
        }
        let simd8_state = unsafe { std::slice::from_raw_parts(simd8_recurrent.contents() as *const f32, spec.recurrent_elements()) };
        let legacy_state = unsafe { std::slice::from_raw_parts(legacy_recurrent.contents() as *const f32, spec.recurrent_elements()) };
        for index in 0..spec.recurrent_elements() {
            let difference = (legacy_state[index] - simd8_state[index]).abs();
            assert!(difference < 5.0e-2 * (1.0 + simd8_state[index].abs()), "state index={index}: legacy={} simd8={}", legacy_state[index], simd8_state[index]);
        }
    }

    /// 单 kernel 微基准(诊断用,不随 CI 跑):legacy 逐列 kernel vs rows 寄存器
    fn run_chunked_consistency(spec: GatedDeltaNetSpec, rows: usize) {
        let ctx = MetalContext::new_default().unwrap();
        spec.validate().unwrap();
        let conv_dim = spec.conv_dim();
        let value_dim = spec.value_dim();
        let mut seed = 0x5eed_1234u32;
        let qkv_values: Vec<f32> = (0..rows * conv_dim).map(|_| pseudo_random(&mut seed)).collect();
        let z_values: Vec<f32> = (0..rows * value_dim).map(|_| pseudo_random(&mut seed)).collect();
        let alpha_values: Vec<f32> = (0..rows * spec.value_heads).map(|_| pseudo_random(&mut seed)).collect();
        let beta_values: Vec<f32> = (0..rows * spec.value_heads).map(|_| pseudo_random(&mut seed) + 0.6).collect();
        let conv_values: Vec<f32> = (0..conv_dim * spec.conv_kernel).map(|_| pseudo_random(&mut seed)).collect();
        let a_log_values: Vec<f32> = vec![(0.3f32).ln(); spec.value_heads];
        let dt_bias_values: Vec<f32> = vec![0.1; spec.value_heads];
        let norm_values: Vec<f32> = vec![1.0; spec.value_head_dim];

        let run = |chunked: bool, recurrent: &metal::Buffer| -> Result<Vec<f32>, String> {
            let qkv = ctx.tensor_from_f32(&qkv_values, rows, conv_dim)?;
            let z = ctx.tensor_from_f32(&z_values, rows, value_dim)?;
            let alpha = ctx.tensor_from_f32(&alpha_values, rows, spec.value_heads)?;
            let beta = ctx.tensor_from_f32(&beta_values, rows, spec.value_heads)?;
            let conv_weight = ctx.tensor_from_f32(&conv_values, conv_dim, spec.conv_kernel)?;
            let a_log = ctx.shared_buffer(&a_log_values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            let dt_bias = ctx.shared_buffer(&dt_bias_values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            let norm_weight = ctx.shared_buffer(&norm_values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            let conv_state = ctx.shared_buffer_zeros(spec.conv_state_elements() * 4);
            let output = gated_delta_net_tensor_impl(&ctx, &qkv, &z, &alpha, &beta, &conv_weight, &a_log, spec.value_heads, &dt_bias, spec.value_heads, &norm_weight, spec.value_head_dim, &conv_state, recurrent, &spec, chunked)?;
            Ok(ctx.read_f16_to_f32(&output.buffer, rows * value_dim))
        };
        let serial_state = ctx.shared_buffer_zeros(spec.recurrent_elements() * 4);
        let chunked_state = ctx.shared_buffer_zeros(spec.recurrent_elements() * 4);
        let serial = run(false, &serial_state).unwrap();
        let chunked = run(true, &chunked_state).unwrap();
        ctx.synchronize();
        let mut mismatches = 0usize;
        let mut row_errors = vec![0usize; rows];
        let mut col_errors = vec![0usize; value_dim];
        for index in 0..serial.len() {
            let difference = (serial[index] - chunked[index]).abs();
            if !(difference < 5.0e-2 * (1.0 + serial[index].abs())) {
                row_errors[index / value_dim] += 1;
                col_errors[index % value_dim] += 1;
                mismatches += 1;
            }
        }
        // scratch_b 是 fused 内部 buffer,测试侧无法读取;退而求其次:再跑一次 impl(拷贝 probe 不必要) ——
        // 直接跳过 probe 打印,靠错误分布判断。
        eprintln!("total mismatches: {mismatches}/{}", serial.len());
        let mut bad: Vec<usize> = Vec::new();
        for c in 0..value_dim {
            let index = value_dim + c;
            if (serial[index] - chunked[index]).abs() >= 5.0e-2 * (1.0 + serial[index].abs()) {
                bad.push(c);
            }
        }
        eprintln!("row1 head0 bad cols({}): {:?}", bad.len(), &bad[..bad.len().min(20)]);
        assert_eq!(mismatches, 0);
        // 最终 recurrent state 也必须一致(decode 依赖它继续推进)。
        let serial_state_values = unsafe { std::slice::from_raw_parts(serial_state.contents() as *const f32, spec.recurrent_elements()) };
        let chunked_state_values = unsafe { std::slice::from_raw_parts(chunked_state.contents() as *const f32, spec.recurrent_elements()) };
        for index in 0..spec.recurrent_elements() {
            let difference = (serial_state_values[index] - chunked_state_values[index]).abs();
            assert!(difference < 5.0e-2 * (1.0 + serial_state_values[index].abs()), "state index={index}: {} vs {}", serial_state_values[index], chunked_state_values[index]);
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod f16_in_f32_weight_tests {
    use super::super::{to_f16_tensor, to_f32_tensor};
    use super::*;

    /// F16 直读 F32 权重的新路径必须与「f16→f32 cast + F32 权重 kernel + f32→f16 cast」逐位一致
    #[test]
    fn f16_in_f32_weight_rmsnorm_matches_cast_chain() {
        if crate::backend::metal::api::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        // 256/512 覆盖 head_dim 常见 simd 路径,130 覆盖非 4 对齐的标量回退
        for columns in [256_usize, 512, 130] {
            let rows = 12usize;
            let input_values = (0..rows * columns).map(|index| ((index as f32) * 0.017).sin() * 2.5).collect::<Vec<_>>();
            let input = ctx.tensor_from_f32(&input_values, rows, columns).unwrap();
            let weight = (0..columns).map(|index| 0.7 + 0.3 * ((index as f32) * 0.031).cos()).collect::<Vec<f32>>();
            let weight_bytes = unsafe { std::slice::from_raw_parts(weight.as_ptr().cast::<u8>(), std::mem::size_of_val(weight.as_slice())) };
            let weight_buffer = ctx.shared_buffer(weight_bytes);
            let eps = 1.0e-6_f32;

            let casted = to_f32_tensor(&ctx, &input).unwrap();
            let legacy = rmsnorm_tensor_resident_f32_weight(&ctx, &casted, &weight_buffer, columns, eps, 1.0).unwrap();
            let legacy = to_f16_tensor(&ctx, &legacy).unwrap();
            let direct = rmsnorm_f16_in_f32_weight_tensor_resident(&ctx, &input, &weight_buffer, columns, eps, 1.0).unwrap();

            let expected = ctx.tensor_to_f32(&legacy);
            let actual = ctx.tensor_to_f32(&direct);
            for (index, (want, got)) in expected.iter().zip(&actual).enumerate() {
                assert_eq!(want, got, "columns={columns} index={index}: cast 链={want} 直通={got}");
            }
        }
    }

    #[test]
    fn f32_in_f32_weight_grouped_rmsnorm_to_f16_matches_cast_chain() {
        if crate::backend::metal::api::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let columns = 2560usize;
        let rows = 3usize;
        let groups = 2usize;
        let input_values = (0..rows * columns).map(|index| ((index as f32) * 0.013).sin() * 1.7).collect::<Vec<_>>();
        let input = ctx.tensor_from_f32_preserve(&input_values, rows, columns).unwrap();
        let weight = (0..columns).map(|index| 0.8 + 0.2 * ((index as f32) * 0.019).cos()).collect::<Vec<f32>>();
        let weight_bytes = unsafe { std::slice::from_raw_parts(weight.as_ptr().cast::<u8>(), std::mem::size_of_val(weight.as_slice())) };
        let weight_buffer = ctx.shared_buffer(weight_bytes);
        let expected = grouped_rmsnorm_f32_tensor_resident(&ctx, &input, &weight_buffer, columns, 1.0e-6, groups).unwrap();
        let expected = to_f16_tensor(&ctx, &expected).unwrap();
        let actual = grouped_rmsnorm_f32_in_f32_weight_to_f16_tensor_resident(&ctx, &input, &weight_buffer, columns, 1.0e-6, groups).unwrap();
        assert_eq!(ctx.tensor_to_f32(&actual), ctx.tensor_to_f32(&expected));
    }

    #[test]
    fn fused_add_grouped_rmsnorm_matches_separate_path() {
        if crate::backend::metal::api::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let columns = 2560usize;
        let left_values = (0..columns).map(|index| ((index as f32) * 0.011).sin()).collect::<Vec<_>>();
        let right_values = (0..columns).map(|index| ((index as f32) * 0.017).cos()).collect::<Vec<_>>();
        let left = ctx.tensor_from_f32_preserve(&left_values, 1, columns).unwrap();
        let right = ctx.tensor_from_f32(&right_values, 1, columns).unwrap();
        let weight = (0..columns).map(|index| 0.9 + 0.1 * ((index as f32) * 0.023).sin()).collect::<Vec<f32>>();
        let weight_bytes = unsafe { std::slice::from_raw_parts(weight.as_ptr().cast::<u8>(), std::mem::size_of_val(weight.as_slice())) };
        let weight_buffer = ctx.shared_buffer(weight_bytes);
        let expected_residual = add_tensor(&ctx, &left, &right).unwrap();
        let expected_normalized = grouped_rmsnorm_f32_in_f32_weight_to_f16_tensor_resident(&ctx, &expected_residual, &weight_buffer, columns, 1.0e-6, 2).unwrap();
        let (actual_residual, actual_normalized) = add_f32_f16_grouped_rmsnorm_tensor_resident(&ctx, &left, &right, &weight_buffer, columns, 1.0e-6, 2).unwrap();
        assert_eq!(ctx.tensor_to_f32(&actual_residual), ctx.tensor_to_f32(&expected_residual));
        assert_eq!(ctx.tensor_to_f32(&actual_normalized), ctx.tensor_to_f32(&expected_normalized));
    }
}

/// Q/K/V 三路逐头 GemmaRMSNorm 共享一个 encoder(3 dispatch 省 2 个 encoder 边界)。
/// 复用 rms_norm_f16_in_f32_weight_f16_simd(生产已验证 kernel)。
pub(crate) fn rmsnorm_f16_heads_triple_encoder(
    ctx: &MetalContext,
    query: &MetalTensor,
    key: &MetalTensor,
    value: &MetalTensor,
    query_norm: &metal::Buffer,
    key_norm: &metal::Buffer,
    value_norm: &metal::Buffer,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(MetalTensor, MetalTensor, MetalTensor), String> {
    if query.rows != 1 || key.rows != 1 || value.rows != 1 || query.dtype != MetalTensorDType::F16 {
        return Err("三路逐头 norm 需要 F16 单行输入".to_owned());
    }
    if head_dim == 0 || head_dim % 4 != 0 || head_dim > 512 {
        return Err(format!("三路逐头 norm head_dim={head_dim} 不兼容"));
    }
    if query.cols != head_count * head_dim || key.cols != kv_head_count * head_dim || value.cols != kv_head_count * head_dim {
        return Err("三路逐头 norm 列不匹配".to_owned());
    }
    let columns = validate_u32("三路逐头 norm cols", head_dim)?;
    let weight_offset: f32 = 1.0;
    let query_out = ctx.tensor_zeros(1, query.cols);
    let key_out = ctx.tensor_zeros(1, key.cols);
    let value_out = ctx.tensor_zeros(1, value.cols);
    let pipeline = ctx.pipeline("rms_norm_f16_in_f32_weight_f16_simd")?;
    let command = ctx.command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (input, weight, output, rows) in [(&query.buffer, query_norm, &query_out.buffer, head_count), (&key.buffer, key_norm, &key_out.buffer, kv_head_count), (&value.buffer, value_norm, &value_out.buffer, kv_head_count)] {
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        set_bytes(&encoder, 3, &columns);
        set_bytes(&encoder, 4, &eps);
        set_bytes(&encoder, 5, &weight_offset);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(32, 1, 1));
    }
    encoder.end_encoding();
    let shape = format!("heads={head_count}+2x{kv_head_count},dim={head_dim}");
    ctx.commit_and_wait_profiled(&command, "rmsnorm_f16_heads_triple", &shape, query.buffer.length() + key.buffer.length() + value.buffer.length(), query_out.buffer.length() + key_out.buffer.length() + value_out.buffer.length());
    Ok((query_out, key_out, value_out))
}
