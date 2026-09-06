//! QSA(Qwen Sparse Attention)稀疏路径 CUDA kernel。
//!
//! 语义源:runtime/qwen4exp/cpu.rs 的 reference(oracle)。三段:
//! 1. `qsa_pool_blocks_f32`:满 ratio-token 块均值池化 raw 索引 K → RMSNorm(k_norm)
//!    → SplitHalf RoPE(块起点位置,cos/sin 由调用方按 b*ratio 行收集上传);
//! 2. `qsa_score_select`:逐 query 行对全部块 ReLU 逐头点积求和打分,
//!    死块/tail 块 +1e9 偏置,按(分数降、cell 升)取 top (top_k+r-1) 个 cell,
//!    输出压缩索引表 + 计数(读侧免空转;块内 cell 同分,块按分数序展开
//!    与 reference 的稳定排序等价);
//! 3. `qsa_masked_attention_{q8g64,f16}`:在选中 cell 集上做在线 softmax GQA,
//!    结构镜像 kernel/cuda/attention.rs 的同名稠密 kernel,仅迭代源换成索引表。

use super::{CudaContext, CudaSliceF16, CudaTensor, LaunchConfig, PushKernelArg};
use cudarc::driver::safe::CudaSlice;

pub const SHADERS: &str = r#"
// 均值池化 ratio 行 F32 → RMSNorm → SplitHalf RoPE → F16。grid: 新块数,block: head_dim 线程。
extern "C" __global__ void qsa_pool_blocks_f32(
    const __half * __restrict__ raw,
    float * __restrict__ scratch,
    const __half * __restrict__ norm_weight,
    const float * __restrict__ cos_rows,
    const float * __restrict__ sin_rows,
    __half * __restrict__ pooled,
    unsigned int new_blocks,
    unsigned int head_dim,
    unsigned int ratio,
    unsigned int first_block,
    unsigned int half,
    float eps)
{
    extern __shared__ float row[];
    float *rotated = row + head_dim;
    __shared__ float squares[256];
    unsigned int block = blockIdx.x;
    if (block >= new_blocks) return;
    unsigned int column = threadIdx.x;
    unsigned long long base_raw = ((unsigned long long)first_block + block) * ratio * head_dim;
    float value = 0.0f;
    if (column < head_dim) {
        float sum = 0.0f;
        for (unsigned int member = 0; member < ratio; ++member) sum += __half2float(raw[base_raw + (unsigned long long)member * head_dim + column]);
        value = sum / (float)ratio;
        row[column] = value;
    }
    if (column < 256) squares[column] = 0.0f;
    __syncthreads();
    if (column < head_dim) atomicAdd(&squares[0], value * value);
    __syncthreads();
    float inv = 0.0f;
    if (column == 0) scratch[block] = rsqrtf(squares[0] / (float)head_dim + eps);
    __syncthreads();
    inv = scratch[block];
    // RoPE:每个 thread 处理一对 (c, c+half),双缓冲写 rotated,无循环内同步。
    if (column < half) {
        float even = row[column] * inv * __half2float(norm_weight[column]);
        float odd = row[column + half] * inv * __half2float(norm_weight[column + half]);
        float c_cos = cos_rows[(unsigned long long)block * half + column];
        float c_sin = sin_rows[(unsigned long long)block * half + column];
        rotated[column] = even * c_cos - odd * c_sin;
        rotated[column + half] = even * c_sin + odd * c_cos;
    }
    __syncthreads();
    if (column < head_dim) {
        pooled[((unsigned long long)first_block + block) * head_dim + column] = __float2half(rotated[column]);
    }
}

// 逐 query 行块打分 + top-W cell 选择。grid: query 行数,block: 256。
extern "C" __global__ void qsa_score_select(
    const __half * __restrict__ pooled,
    const __half * __restrict__ index_query,
    float * __restrict__ block_scores,
    unsigned int * __restrict__ selected,
    unsigned int * __restrict__ selected_count,
    unsigned int n_blocks_full,
    unsigned int n_cells,
    unsigned int dead_block,
    unsigned int query_position,
    unsigned int ratio,
    unsigned int top_k,
    unsigned int indexer_heads,
    unsigned int head_dim,
    unsigned int width)
{
    unsigned int row = blockIdx.x;
    unsigned int lane = threadIdx.x;
    unsigned int n_blocks = (n_cells + ratio - 1) / ratio;
    float *scores = block_scores + (unsigned long long)row * n_blocks;
    unsigned long long iq_base = (unsigned long long)row * indexer_heads * head_dim;
    // 打分:未满块记 0 分(靠偏置入选),满块 ReLU 逐头点积求和。
    for (unsigned int b = lane; b < n_blocks; b += blockDim.x) {
        float score = 0.0f;
        if (b < n_blocks_full) {
            const __half *k = pooled + (unsigned long long)b * head_dim;
            for (unsigned int h = 0; h < indexer_heads; ++h) {
                const __half *q = index_query + iq_base + (unsigned long long)h * head_dim;
                float dot = 0.0f;
                for (unsigned int c = 0; c < head_dim; ++c) dot += __half2float(k[c]) * __half2float(q[c]);
                score += dot > 0.0f ? dot : 0.0f;
            }
        }
        scores[b] = score;
    }
    __syncthreads();
    unsigned int tail_start = (query_position + 1) / ratio * ratio;
    unsigned int count = 0;
    unsigned int *out = selected + (unsigned long long)row * width;
    __shared__ float shared_max[256];
    __shared__ unsigned int shared_arg[256];
    while (count < width) {
        float local_max = -3.402823466e+38F;
        unsigned int local_arg = 0u;
        for (unsigned int b = lane; b < n_blocks; b += blockDim.x) {
            float value = scores[b];
            if (b == dead_block && n_blocks_full < n_blocks) value += 1e9f;
            if ((unsigned long long)b * ratio >= tail_start) value += 1e9f;
            if (value > local_max || (value == local_max && b < local_arg)) { local_max = value; local_arg = b; }
        }
        shared_max[lane] = local_max;
        shared_arg[lane] = local_arg;
        __syncthreads();
        for (unsigned int d = blockDim.x >> 1; d; d >>= 1) {
            if (lane < d) {
                float other = shared_max[lane + d];
                unsigned int other_arg = shared_arg[lane + d];
                if (other > shared_max[lane] || (other == shared_max[lane] && other_arg < shared_arg[lane])) {
                    shared_max[lane] = other;
                    shared_arg[lane] = other_arg;
                }
            }
            __syncthreads();
        }
        if (shared_max[0] <= -3.402823466e+38F) break;
        unsigned int block_start = shared_arg[0] * ratio;
        if (lane == 0) scores[shared_arg[0]] = -3.402823466e+38F;
        for (unsigned int c = block_start; c < block_start + ratio && count < width; ++c) {
            if (c >= n_cells || c > query_position) continue;
            if (lane == 0) out[count] = c;
            count++;
        }
        __syncthreads();
    }
    if (lane == 0) selected_count[row] = count;
}

// 选中 cell 集上的 GQA 在线 softmax(Q8G64 KV),逐 (row,head) block;
// uchar4 向量化 + query smem 预载。融合版见 _fused(数值调试中)。
extern "C" __global__ void qsa_masked_attention_q8g64(
    const __half * __restrict__ query,
    const signed char * __restrict__ key_codes,
    const float * __restrict__ key_scales,
    const signed char * __restrict__ value_codes,
    const float * __restrict__ value_scales,
    const unsigned int * __restrict__ selected,
    const unsigned int * __restrict__ selected_count,
    __half * __restrict__ output,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int width,
    float attention_scale)
{
    extern __shared__ float shared[];
    unsigned int block = blockIdx.x;
    unsigned int row = block / heads;
    unsigned int head = block - row * heads;
    unsigned int lane = threadIdx.x;
    unsigned int warp = lane >> 5;
    unsigned int warp_lane = lane & 31u;
    unsigned int warp_count = blockDim.x >> 5;
    unsigned int kv_head = head / (heads / kv_heads);
    unsigned int groups_per_row = (kv_heads * head_dim) >> 6;
    unsigned long long q_base = ((unsigned long long)row * heads + head) * head_dim;
    unsigned long long kv_head_column = (unsigned long long)kv_head * head_dim;
    const unsigned int *cells = selected + (unsigned long long)row * width;
    unsigned int count = selected_count[row];
    float *partial_maximum = shared;
    float *partial_denominator = shared + warp_count;
    float *partial_values = shared + warp_count * 2;
    float *query_cached = partial_values + warp_count * head_dim;
    for (unsigned int index = lane; index < warp_count * head_dim; index += blockDim.x) partial_values[index] = 0.0f;
    for (unsigned int dimension = lane; dimension < head_dim; dimension += blockDim.x) query_cached[dimension] = __half2float(query[q_base + dimension]);
    __syncthreads();

    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    for (unsigned int index = warp; index < count; index += warp_count) {
        unsigned int token = cells[index];
        unsigned long long kv_base = ((unsigned long long)token * kv_heads + kv_head) * head_dim;
        unsigned long long scale_base = (unsigned long long)token * groups_per_row + kv_head_column / 64u;
        const uchar4 * __restrict__ key_packed = (const uchar4 *)(key_codes + kv_base);
        const uchar4 * __restrict__ value_packed = (const uchar4 *)(value_codes + kv_base);
        float partial = 0.0f;
        for (unsigned int dimension4 = warp_lane; dimension4 < (head_dim >> 2); dimension4 += 32u) {
            float scale = key_scales[scale_base + ((dimension4 << 2) >> 6)];
            uchar4 packed = key_packed[dimension4];
            const float *q4 = query_cached + (dimension4 << 2);
            partial += (q4[0] * (float)((signed char)packed.x)
                      + q4[1] * (float)((signed char)packed.y)
                      + q4[2] * (float)((signed char)packed.z)
                      + q4[3] * (float)((signed char)packed.w)) * scale;
        }
        for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, offset);
        float old_scale = 0.0f;
        float new_scale = 0.0f;
        if (warp_lane == 0u) {
            float score = partial * attention_scale;
            float next_maximum = fmaxf(maximum, score);
            old_scale = maximum <= -3.402823466e+38F ? 0.0f : expf(maximum - next_maximum);
            new_scale = expf(score - next_maximum);
            denominator = denominator * old_scale + new_scale;
            maximum = next_maximum;
        }
        old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
        new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
        float *accumulator = partial_values + warp * head_dim;
        for (unsigned int dimension4 = warp_lane; dimension4 < (head_dim >> 2); dimension4 += 32u) {
            float scale = value_scales[scale_base + ((dimension4 << 2) >> 6)];
            uchar4 packed = value_packed[dimension4];
            float *slot = accumulator + (dimension4 << 2);
            slot[0] = slot[0] * old_scale + new_scale * (float)((signed char)packed.x) * scale;
            slot[1] = slot[1] * old_scale + new_scale * (float)((signed char)packed.y) * scale;
            slot[2] = slot[2] * old_scale + new_scale * (float)((signed char)packed.z) * scale;
            slot[3] = slot[3] * old_scale + new_scale * (float)((signed char)packed.w) * scale;
        }
    }
    if (warp_lane == 0u) {
        partial_maximum[warp] = maximum;
        partial_denominator[warp] = denominator;
    }
    __syncthreads();
    for (unsigned int dimension = lane; dimension < head_dim; dimension += blockDim.x) {
        float global_maximum = -3.402823466e+38F;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) global_maximum = fmaxf(global_maximum, partial_maximum[item]);
        }
        float global_denominator = 0.0f;
        float accumulator = 0.0f;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) {
                float rescale = expf(partial_maximum[item] - global_maximum);
                global_denominator += partial_denominator[item] * rescale;
                accumulator += partial_values[item * head_dim + dimension] * rescale;
            }
        }
        output[q_base + dimension] = __float2half(accumulator / (global_denominator < 1e-30f ? 1e-30f : global_denominator));
    }
}

// 选中 cell 集上的 GQA 融合 flash-tile(Q8G64 KV)。grid: (rows, kv_heads);
// block = heads_per_group × 32 线程,每 warp 负责该 kv 组的一个 query head。
// cell K/V 按 16 个一分幅协同装载进 smem(同行 12 个 head 共享一份 Q8 解码,
// 全局码流量降 12 倍),在线 softmax 逐 head 独立,尾段归一化写出。
extern "C" __global__ void qsa_masked_attention_q8g64_fused(
    const __half * __restrict__ query,
    const signed char * __restrict__ key_codes,
    const float * __restrict__ key_scales,
    const signed char * __restrict__ value_codes,
    const float * __restrict__ value_scales,
    const unsigned int * __restrict__ selected,
    const unsigned int * __restrict__ selected_count,
    __half * __restrict__ output,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int width,
    float attention_scale)
{
    extern __shared__ float shared[];
    unsigned int row = blockIdx.x;
    unsigned int kv_group = blockIdx.y;
    unsigned int heads_per_group = heads / kv_heads;
    unsigned int lane = threadIdx.x;
    unsigned int warp = lane >> 5;
    unsigned int warp_lane = lane & 31u;
    if (warp >= heads_per_group) return;
    unsigned int head = kv_group * heads_per_group + warp;
    unsigned int groups_per_row = (kv_heads * head_dim) >> 6;
    unsigned long long q_base = ((unsigned long long)row * heads + head) * head_dim;
    unsigned long long kv_head_column = (unsigned long long)kv_group * head_dim;
    const unsigned int *cells = selected + (unsigned long long)row * width;
    unsigned int count = selected_count[row];
    const unsigned int tile = 8u;
    // query_cache/accumulator 每 warp 一份:各 warp 负责不同 head 的 query,
    // 共享同一块会互相覆写(12 warp 数据竞争 → NaN → route 塌缩,实测踩中)。
    float *query_cache = shared + (unsigned long long)warp * head_dim;
    float *accumulator = shared + ((unsigned long long)warp + heads_per_group) * head_dim;
    float *tile_k = shared + 2u * (unsigned long long)heads_per_group * head_dim;
    float *tile_v = tile_k + tile * head_dim;
    float *tile_scales = tile_v + tile * head_dim;
    for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32u) {
        query_cache[dimension] = __half2float(query[q_base + dimension]);
        accumulator[dimension] = 0.0f;
    }
    __syncthreads();

    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    for (unsigned int chunk_start = 0; chunk_start < count; chunk_start += tile) {
        unsigned int chunk = tile < (count - chunk_start) ? tile : (count - chunk_start);
        for (unsigned int index = lane; index < chunk * head_dim; index += blockDim.x) {
            unsigned int local = index / head_dim, dimension = index - local * head_dim;
            unsigned int token = cells[chunk_start + local];
            unsigned long long kv_base = ((unsigned long long)token * kv_heads + kv_group) * head_dim;
            unsigned long long scale_base = (unsigned long long)token * groups_per_row + kv_head_column / 64u;
            tile_k[local * head_dim + dimension] = (float)key_codes[kv_base + dimension];
            tile_v[local * head_dim + dimension] = (float)value_codes[kv_base + dimension];
            if (dimension % 64u == 0u) {
                tile_scales[local * (head_dim >> 6) * 2u + (dimension >> 6)] = key_scales[scale_base + (dimension >> 6)];
                tile_scales[local * (head_dim >> 6) * 2u + (head_dim >> 6) + (dimension >> 6)] = value_scales[scale_base + (dimension >> 6)];
            }
        }
        __syncthreads();
        for (unsigned int local = 0; local < chunk; ++local) {
            const float *k = tile_k + local * head_dim;
            const float *ks = tile_scales + local * (head_dim >> 6) * 2u;
            float partial = 0.0f;
            for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32u) {
                partial += query_cache[dimension] * k[dimension] * ks[dimension >> 6];
            }
            for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, offset);
            float old_scale = 0.0f, new_scale = 0.0f;
            if (warp_lane == 0u) {
                float score = partial * attention_scale;
                float next_maximum = fmaxf(maximum, score);
                old_scale = maximum <= -3.402823466e+38F ? 0.0f : expf(maximum - next_maximum);
                new_scale = expf(score - next_maximum);
                denominator = denominator * old_scale + new_scale;
                maximum = next_maximum;
            }
            old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
            new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
            const float *v = tile_v + local * head_dim;
            const float *vs = tile_scales + local * (head_dim >> 6) * 2u + (head_dim >> 6);
            for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32u) {
                accumulator[dimension] = accumulator[dimension] * old_scale + new_scale * v[dimension] * vs[dimension >> 6];
            }
        }
        __syncthreads();
    }
    for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32u) {
        output[q_base + dimension] = __float2half(accumulator[dimension] / (denominator < 1e-30f ? 1e-30f : denominator));
    }
}

// F16 KV 形态的同构变体(oracle 对照/高精度模式)。
extern "C" __global__ void qsa_masked_attention_f16(
    const __half * __restrict__ query,
    const __half * __restrict__ key,
    const __half * __restrict__ value,
    const unsigned int * __restrict__ selected,
    const unsigned int * __restrict__ selected_count,
    __half * __restrict__ output,
    unsigned int heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int kv_columns,
    unsigned int width,
    float attention_scale)
{
    extern __shared__ float shared[];
    unsigned int block = blockIdx.x;
    unsigned int row = block / heads;
    unsigned int head = block - row * heads;
    unsigned int lane = threadIdx.x;
    unsigned int warp = lane >> 5;
    unsigned int warp_lane = lane & 31u;
    unsigned int warp_count = blockDim.x >> 5;
    unsigned int kv_head = head / (heads / kv_heads);
    unsigned long long q_base = ((unsigned long long)row * heads + head) * head_dim;
    unsigned long long kv_base_head = (unsigned long long)kv_head * head_dim;
    const unsigned int *cells = selected + (unsigned long long)row * width;
    unsigned int count = selected_count[row];
    float *partial_maximum = shared;
    float *partial_denominator = shared + warp_count;
    float *partial_values = shared + warp_count * 2;
    for (unsigned int index = lane; index < warp_count * head_dim; index += blockDim.x) partial_values[index] = 0.0f;
    __syncthreads();

    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    for (unsigned int index = warp; index < count; index += warp_count) {
        unsigned long long kv_base = (unsigned long long)cells[index] * kv_columns + kv_base_head;
        float partial = 0.0f;
        for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32u) {
            partial += __half2float(query[q_base + dimension]) * __half2float(key[kv_base + dimension]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, offset);
        float old_scale = 0.0f;
        float new_scale = 0.0f;
        if (warp_lane == 0u) {
            float score = partial * attention_scale;
            float next_maximum = fmaxf(maximum, score);
            old_scale = expf(maximum - next_maximum);
            new_scale = expf(score - next_maximum);
            denominator = denominator * old_scale + new_scale;
            maximum = next_maximum;
        }
        old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
        new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
        float *accumulator = partial_values + warp * head_dim;
        for (unsigned int dimension = warp_lane; dimension < head_dim; dimension += 32u) {
            accumulator[dimension] = accumulator[dimension] * old_scale + new_scale * __half2float(value[kv_base + dimension]);
        }
    }
    if (warp_lane == 0u) {
        partial_maximum[warp] = maximum;
        partial_denominator[warp] = denominator;
    }
    __syncthreads();
    for (unsigned int dimension = lane; dimension < head_dim; dimension += blockDim.x) {
        float global_maximum = -3.402823466e+38F;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) global_maximum = fmaxf(global_maximum, partial_maximum[item]);
        }
        float global_denominator = 0.0f;
        float accumulator = 0.0f;
        for (unsigned int item = 0; item < warp_count; ++item) {
            if (partial_denominator[item] > 0.0f) {
                float rescale = expf(partial_maximum[item] - global_maximum);
                global_denominator += partial_denominator[item] * rescale;
                accumulator += partial_values[item * head_dim + dimension] * rescale;
            }
        }
        output[q_base + dimension] = __float2half(accumulator / global_denominator);
    }
}
"#;

/// 池化新完成的块:raw F32 行 → norm+RoPE → pooled F16 行。
/// cos/sin 由调用方收集(块起点位置 b*ratio 的 RoPE 表行)上传。
#[allow(clippy::too_many_arguments)]
pub fn qsa_pool_blocks(
    ctx: &CudaContext,
    raw: &CudaSliceF16,
    pooled: &mut CudaTensor,
    norm_weight: &CudaSliceF16,
    cos_rows: &CudaSlice<f32>,
    sin_rows: &CudaSlice<f32>,
    first_block: usize,
    new_blocks: usize,
    head_dim: usize,
    ratio: usize,
    eps: f32,
) -> Result<(), String> {
    if new_blocks == 0 {
        return Ok(());
    }
    let half = 32usize;
    let scratch = ctx.stream().alloc_zeros::<f32>(new_blocks).map_err(|error| format!("qsa pool scratch: {error:?}"))?;
    let func = ctx.function("qsa_pool_blocks_f32")?;
    let cfg = LaunchConfig { grid_dim: (new_blocks as u32, 1, 1), block_dim: (head_dim.max(256) as u32, 1, 1), shared_mem_bytes: ((head_dim * 2) * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(raw)
            .arg(&scratch)
            .arg(norm_weight)
            .arg(cos_rows)
            .arg(sin_rows)
            .arg(&pooled.slice)
            .arg(&(new_blocks as u32))
            .arg(&(head_dim as u32))
            .arg(&(ratio as u32))
            .arg(&(first_block as u32))
            .arg(&(half as u32))
            .arg(&eps)
            .launch(cfg)
            .map_err(|error| format!("qsa pool launch: {error:?}"))?;
    }
    Ok(())
}

/// 逐 query 行打分 + top-W 选择;输出压缩 cell 索引表(rows × width)与计数。
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_select(
    ctx: &CudaContext,
    pooled: &CudaTensor,
    index_query: &CudaTensor,
    block_scores: &mut CudaSlice<f32>,
    selected: &mut CudaSlice<u32>,
    selected_count: &mut CudaSlice<u32>,
    rows: usize,
    n_cells: usize,
    dead_block: usize,
    position: usize,
    ratio: usize,
    top_k: usize,
    indexer_heads: usize,
    head_dim: usize,
    width: usize,
) -> Result<(), String> {
    let n_blocks_full = n_cells / ratio;
    let func = ctx.function("qsa_score_select")?;
    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (256 * 8) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&pooled.slice)
            .arg(&index_query.slice)
            .arg(block_scores)
            .arg(selected)
            .arg(selected_count)
            .arg(&(n_blocks_full as u32))
            .arg(&(n_cells as u32))
            .arg(&(dead_block as u32))
            .arg(&(position as u32))
            .arg(&(ratio as u32))
            .arg(&(top_k as u32))
            .arg(&(indexer_heads as u32))
            .arg(&(head_dim as u32))
            .arg(&(width as u32))
            .launch(cfg)
            .map_err(|error| format!("qsa select launch: {error:?}"))?;
    }
    Ok(())
}

/// 选中 cell 集上的掩码 GQA(Q8G64 KV)。
#[allow(clippy::too_many_arguments)]
pub fn qsa_masked_attention_q8g64(
    ctx: &CudaContext,
    query: &CudaTensor,
    key_codes: &CudaSlice<i8>,
    key_scales: &CudaSlice<f32>,
    value_codes: &CudaSlice<i8>,
    value_scales: &CudaSlice<f32>,
    selected: &CudaSlice<u32>,
    selected_count: &CudaSlice<u32>,
    spec: &crate::attention::gqa::GqaSpec,
    width: usize,
) -> Result<CudaTensor, String> {
    let query_cols = spec.num_heads * spec.head_dim;
    let output = ctx.tensor_alloc(query.rows, query_cols)?;
    let func = ctx.function("qsa_masked_attention_q8g64")?;
    let threads = 256;
    let warp_count = threads / 32;
    let shared_floats = warp_count * (spec.head_dim + 2);
    let cfg = LaunchConfig { grid_dim: ((query.rows * spec.num_heads) as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&query.slice)
            .arg(key_codes)
            .arg(key_scales)
            .arg(value_codes)
            .arg(value_scales)
            .arg(selected)
            .arg(selected_count)
            .arg(&output.slice)
            .arg(&(spec.num_heads as u32))
            .arg(&(spec.num_kv_heads as u32))
            .arg(&(spec.head_dim as u32))
            .arg(&(width as u32))
            .arg(&spec.score_scale)
            .launch(cfg)
            .map_err(|error| format!("qsa masked q8 launch: {error:?}"))?;
    }
    Ok(output)
}

/// 选中 cell 集上的掩码 GQA(F16 KV)。
#[allow(clippy::too_many_arguments)]
pub fn qsa_masked_attention_f16(
    ctx: &CudaContext,
    query: &CudaTensor,
    key: &CudaSliceF16,
    value: &CudaSliceF16,
    selected: &CudaSlice<u32>,
    selected_count: &CudaSlice<u32>,
    spec: &crate::attention::gqa::GqaSpec,
    width: usize,
) -> Result<CudaTensor, String> {
    let query_cols = spec.num_heads * spec.head_dim;
    let output = ctx.tensor_alloc(query.rows, query_cols)?;
    let func = ctx.function("qsa_masked_attention_f16")?;
    let threads = 256;
    let warp_count = threads / 32;
    let shared_floats = warp_count * (spec.head_dim + 2);
    let cfg = LaunchConfig { grid_dim: ((query.rows * spec.num_heads) as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32 };
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&query.slice)
            .arg(key)
            .arg(value)
            .arg(selected)
            .arg(selected_count)
            .arg(&output.slice)
            .arg(&(spec.num_heads as u32))
            .arg(&(spec.num_kv_heads as u32))
            .arg(&(spec.head_dim as u32))
            .arg(&((spec.num_kv_heads * spec.head_dim) as u32))
            .arg(&(width as u32))
            .arg(&spec.score_scale)
            .launch(cfg)
            .map_err(|error| format!("qsa masked f16 launch: {error:?}"))?;
    }
    Ok(output)
}
