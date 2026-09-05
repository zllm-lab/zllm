/// 本模块的 CUDA shader(本文件用到的 kernel + 文件私有 `__device__` helper)。
///
/// 共用 helper 见 `mod.rs` 前导。`mod.rs` 的 `kernels_source()`
/// 把 mod.rs 的 `PREAMBLE_SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: gguf_kq_matmul_f16, gguf_kq_gemv_f16, gated_linear_q4_k_silu_f16, gated_linear_q4_k_silu_rows8_f16, gated_linear_q4_0_silu_f16, gated_linear_q4_0_silu_rows8_f16, linear_q5_k_f16, linear_q5_k_accumulate_f32, linear_q6_k_f16, linear_q6_k_accumulate_f32
// private helpers: q4_k_scale_min, q4_k_block_dot_pair, q4_0_block_dot_pair, q5_k_block_dot, q5_k_row_dot, q6_k_block_dot, q6_k_row_dot
pub const SHADERS: &str = r#"
__device__ __forceinline__ void q4_k_scale_min(
    const unsigned char *scales,
    const unsigned int group,
    unsigned int *scale,
    unsigned int *minimum)
{
    if (group < 4) {
        *scale = scales[group] & 0x3f;
        *minimum = scales[group + 4] & 0x3f;
    } else {
        *scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
        *minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
}
__device__ __forceinline__ unsigned int gguf_kq_block_bytes(const unsigned int tensor_type)
{
    return tensor_type == 2 ? 18u : (tensor_type == 8 ? 34u : (tensor_type == 12 ? 144u : (tensor_type == 13 ? 176u : 210u)));
}
__device__ __forceinline__ unsigned int gguf_kq_block_elements(const unsigned int tensor_type)
{
    return tensor_type == 8 || tensor_type == 2 ? 32u : 256u;
}
__device__ __forceinline__ float gguf_kq_value(
    const unsigned char *row,
    const unsigned int tensor_type,
    const unsigned int column)
{
    const unsigned int block_bytes = gguf_kq_block_bytes(tensor_type);
    const unsigned int block_elements = gguf_kq_block_elements(tensor_type);
    const unsigned char *block = row + (unsigned long long)(column / block_elements) * block_bytes;
    if (tensor_type == 8) {
        return __half2float(*reinterpret_cast<const __half *>(block)) * float((signed char)block[2 + (column & 31)]);
    }
    // Q4_0:32 元素块 = f16 scale + 16 字节 nibble。字节 j 低 nibble 是第 j 个
    // 元素、高 nibble 是第 j+16 个元素(与 weight/codec/ggml.rs 的 decode_q4_0 一致)。
    if (tensor_type == 2) {
        const unsigned int local = column & 31;
        const unsigned char packed = block[2 + (local & 15)];
        const unsigned int nibble = local < 16 ? (packed & 15u) : (packed >> 4);
        return __half2float(*reinterpret_cast<const __half *>(block)) * float(int(nibble) - 8);
    }
    const unsigned int local = column & 255;
    if (tensor_type == 14) {
        const unsigned int half = local >> 7;
        const unsigned int remaining = local & 127;
        const unsigned int index = remaining & 31;
        const unsigned int slot = remaining >> 5;
        const unsigned char low = block[half * 64 + index + (slot & 1) * 32];
        const unsigned int nibble = slot < 2 ? (low & 15u) : (low >> 4);
        const unsigned int high = (block[128 + half * 32 + index] >> (slot * 2)) & 3u;
        const int quant = int(nibble | (high << 4)) - 32;
        const signed char scale = (signed char)block[192 + half * 8 + (index >> 4) + slot * 2];
        return __half2float(*reinterpret_cast<const __half *>(block + 208)) * float(scale) * float(quant);
    }
    const unsigned int group = local >> 5;
    const unsigned int index = local & 31;
    unsigned int scale, minimum;
    q4_k_scale_min(block + 4, group, &scale, &minimum);
    const unsigned int low_offset = tensor_type == 12 ? 16 : 48;
    const unsigned char packed = block[low_offset + (group >> 1) * 32 + index];
    unsigned int quant = (group & 1) == 0 ? (packed & 15u) : (packed >> 4);
    if (tensor_type == 13 && (block[16 + index] & (1u << group)) != 0) quant += 16;
    return __half2float(*reinterpret_cast<const __half *>(block)) * float(scale * quant)
         - __half2float(*reinterpret_cast<const __half *>(block + 2)) * float(minimum);
}
extern "C" __global__ void gguf_kq_matmul_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int input_rows,
    const unsigned int columns,
    const unsigned int output_rows,
    const unsigned int tensor_type)
{
    const unsigned int output_row = blockIdx.x;
    const unsigned int input_base = blockIdx.y * 8;
    const unsigned int lane = threadIdx.x;
    if (output_row >= output_rows) return;
    const unsigned long long row_bytes = (unsigned long long)(columns / gguf_kq_block_elements(tensor_type)) * gguf_kq_block_bytes(tensor_type);
    const unsigned char *row = weight + (unsigned long long)output_row * row_bytes;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        const float value = gguf_kq_value(row, tensor_type, column);
        #pragma unroll
        for (unsigned int local_row = 0; local_row < 8; ++local_row) {
            if (input_base + local_row < input_rows) {
                sums[local_row] += __half2float(input[(unsigned long long)(input_base + local_row) * columns + column]) * value;
            }
        }
    }
    __shared__ float partial[8][256];
    #pragma unroll
    for (unsigned int local_row = 0; local_row < 8; ++local_row) partial[local_row][lane] = sums[local_row];
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            #pragma unroll
            for (unsigned int local_row = 0; local_row < 8; ++local_row) partial[local_row][lane] += partial[local_row][lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned int local_row = 0; local_row < 8; ++local_row) {
            if (input_base + local_row < input_rows) {
                output[(unsigned long long)(input_base + local_row) * output_rows + output_row] = __float2half(partial[local_row][0]);
            }
        }
    }
}
extern "C" __global__ void gguf_kq_gemv_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int columns,
    const unsigned int output_rows,
    const unsigned int tensor_type)
{
    const unsigned int output_row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (output_row >= output_rows) return;
    const unsigned long long row_bytes = (unsigned long long)(columns / gguf_kq_block_elements(tensor_type)) * gguf_kq_block_bytes(tensor_type);
    const unsigned char *row = weight + (unsigned long long)output_row * row_bytes;
    float sum = 0.0f;
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        sum += __half2float(input[column]) * gguf_kq_value(row, tensor_type, column);
    }
    __shared__ float partial[256];
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[output_row] = __float2half(partial[0]);
}
// Q4_0 专用单行 GEMV:每 warp 独占一个输出行,lane 固定覆盖 32 元素块的 1 列,
// scale 每块解码一次;输入向量经 smem 广播(LM head 262144 行共享同一份 x)。
// 替代通用 gguf_kq_gemv 的逐元素解码(26 万行 × 逐元素 f16 scale 重读,~6% 带宽)。
extern "C" __global__ void linear_q4_0_gemv_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int columns,
    const unsigned int out_rows,
    const unsigned int blocks_per_row)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    extern __shared__ __half shared_input[];
    for (unsigned int i = threadIdx.x; i < columns; i += blockDim.x) shared_input[i] = input[i];
    __syncthreads();
    const unsigned int row = blockIdx.x * warp_count + warp;
    if (row >= out_rows) return;
    const unsigned char *row_bytes = weight + (unsigned long long)row * blocks_per_row * 18u;
    float acc = 0.0f;
    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned char *block_bytes = row_bytes + (unsigned long long)block * 18u;
        const float d = __half2float(*reinterpret_cast<const __half *>(block_bytes));
        const unsigned char packed = block_bytes[2u + (lane & 15u)];
        const unsigned int nibble = lane < 16u ? (packed & 15u) : (packed >> 4);
        acc += __half2float(shared_input[block * 32u + lane]) * d * float(int(nibble) - 8);
    }
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, offset);
    if (lane == 0) output[row] = __float2half(acc);
}

// ===== Q4_0 × Q8_1 MMVQ(参考 llama.cpp vecdotq/mmvq:batch ≤ 8 的解码/verify)=====
// 激活先整体量化成 Q8_1(每 32 元素块:32 个 int8 码 + f16 scale,36B 对齐);
// kernel 内权重 nibble 一次提取、对 K 份激活各做一条 dp4a(4 MAC/指令),
// -8 偏移用 dp4a(u, 0x01010101) 精确求和校正,全程整数点积。
struct __align__(4) q4_0_mmvq_q8_block {
    signed char codes[32];
    __half scale;
    __half pad;
};
extern "C" __global__ void q8_1_quantize_rows_f16(
    const __half * __restrict__ input,
    unsigned char * __restrict__ output,
    const unsigned int columns,
    const unsigned int rows)
{
    const unsigned int element = (blockIdx.x * blockDim.x + threadIdx.x) * 32u;
    const unsigned int row = element / columns;
    if (row >= rows) return;
    // 每 thread 独立量化一个 32 元素块:amax → scale=amax/127 → 码 = round(x/scale)。
    q4_0_mmvq_q8_block *block = reinterpret_cast<q4_0_mmvq_q8_block *>(output) + (unsigned long long)row * (columns / 32u) + (element % columns) / 32u;
    const __half *source = input + (unsigned long long)row * columns + element % columns;
    float amax = 0.0f;
    #pragma unroll
    for (unsigned int i = 0; i < 32u; ++i) amax = fmaxf(amax, fabsf(__half2float(source[i])));
    const float scale = amax / 127.0f;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    #pragma unroll
    for (unsigned int i = 0; i < 32u; ++i) block->codes[i] = (signed char)__float2int_rn(__half2float(source[i]) * inverse);
    block->scale = __float2half(scale);
    block->pad = __float2half(0.0f);
}
template <int K>
__device__ __forceinline__ void q4_0_q8_1_mmvq_body(
    const unsigned char * __restrict__ weight,
    const unsigned char * __restrict__ activation,
    __half * __restrict__ output,
    const unsigned int columns,
    const unsigned int out_rows)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    const unsigned int blocks_per_row = columns / 32u;
    const unsigned int blocks_per_cta = blockDim.x / 4u;
    // 每 4 个 thread 服务一个权重块:thread t 持有块内第 t 个 int(4 字节 = 低半
    // 4 元素 + 高半 4 元素),vi0/vi1 与激活码段 u0/u1 逐元素对齐。
    const unsigned char *row_bytes = weight + (unsigned long long)row * blocks_per_row * 18u;
    const q4_0_mmvq_q8_block *acts = reinterpret_cast<const q4_0_mmvq_q8_block *>(activation);
    float acc[K];
    #pragma unroll
    for (int j = 0; j < K; ++j) acc[j] = 0.0f;
    for (unsigned int block = (warp * 32u + lane) / 4u; block < blocks_per_row; block += blocks_per_cta) {
        const unsigned int slot = (warp * 32u + lane) & 3u;
        // 权重 code 区在 18B 块的 +2 偏移,int 载荷只有 2 字节对齐:用两个
        // ushort 组合(llama.cpp get_int_b2 同款约束,字节序保序)。
        const unsigned short *codes = reinterpret_cast<const unsigned short *>(row_bytes + (unsigned long long)block * 18u + 2u + slot * 4u);
        const unsigned int v = unsigned(codes[0]) | (unsigned(codes[1]) << 16);
        const unsigned int vi0 = v & 0x0F0F0F0Fu;
        const unsigned int vi1 = (v >> 4) & 0x0F0F0F0Fu;
        const float d = __half2float(*reinterpret_cast<const __half *>(row_bytes + (unsigned long long)block * 18u));
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const q4_0_mmvq_q8_block *act = acts + (unsigned long long)j * blocks_per_row + block;
            const int u0 = *reinterpret_cast<const int *>(act->codes + slot * 4u);
            const int u1 = *reinterpret_cast<const int *>(act->codes + 16u + slot * 4u);
            int sumi = 0;
            sumi = __dp4a(static_cast<int>(vi0), u0, sumi);
            sumi = __dp4a(static_cast<int>(vi1), u1, sumi);
            // -8 偏移校正:Σ覆盖码 = dp4a(u, 0x01010101)(覆盖本 thread 的 8 元素)。
            int covered = 0;
            covered = __dp4a(u0, static_cast<int>(0x01010101u), covered);
            covered = __dp4a(u1, static_cast<int>(0x01010101u), covered);
            const float act_scale = __half2float(act->scale);
            acc[j] += d * act_scale * (float(sumi) - 8.0f * float(covered));
        }
    }
    // warp 内归约后,跨 warp 经 smem 合并(全部 warp 共享同一输出行)。
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) acc[j] += __shfl_down_sync(0xffffffffu, acc[j], offset);
    }
    __shared__ float partial[8][8];
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < K; ++j) partial[j][warp] = acc[j];
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            float sum = lane < warp_count ? partial[j][lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (lane == 0) output[(unsigned long long)j * out_rows + row] = __float2half(sum);
        }
    }
}
// K=1..8 实例化导出(NVRTC 模块按名加载需要 extern "C" 符号)。
extern "C" __global__ void q4_0_q8_1_mmvq_k1_f16(const unsigned char *w, const unsigned char *a, __half *o, const unsigned int c, const unsigned int r) { q4_0_q8_1_mmvq_body<1>(w, a, o, c, r); }
extern "C" __global__ void q4_0_q8_1_mmvq_k2_f16(const unsigned char *w, const unsigned char *a, __half *o, const unsigned int c, const unsigned int r) { q4_0_q8_1_mmvq_body<2>(w, a, o, c, r); }
extern "C" __global__ void q4_0_q8_1_mmvq_k3_f16(const unsigned char *w, const unsigned char *a, __half *o, const unsigned int c, const unsigned int r) { q4_0_q8_1_mmvq_body<3>(w, a, o, c, r); }
extern "C" __global__ void q4_0_q8_1_mmvq_k4_f16(const unsigned char *w, const unsigned char *a, __half *o, const unsigned int c, const unsigned int r) { q4_0_q8_1_mmvq_body<4>(w, a, o, c, r); }
extern "C" __global__ void q4_0_q8_1_mmvq_k5_f16(const unsigned char *w, const unsigned char *a, __half *o, const unsigned int c, const unsigned int r) { q4_0_q8_1_mmvq_body<5>(w, a, o, c, r); }
extern "C" __global__ void q4_0_q8_1_mmvq_k6_f16(const unsigned char *w, const unsigned char *a, __half *o, const unsigned int c, const unsigned int r) { q4_0_q8_1_mmvq_body<6>(w, a, o, c, r); }
extern "C" __global__ void q4_0_q8_1_mmvq_k7_f16(const unsigned char *w, const unsigned char *a, __half *o, const unsigned int c, const unsigned int r) { q4_0_q8_1_mmvq_body<7>(w, a, o, c, r); }
extern "C" __global__ void q4_0_q8_1_mmvq_k8_f16(const unsigned char *w, const unsigned char *a, __half *o, const unsigned int c, const unsigned int r) { q4_0_q8_1_mmvq_body<8>(w, a, o, c, r); }

// ===== mma m16n8k32 映射探针(诊断用) =====
// host 填 A[16][32]/B[8][32] 已知 int8,按 mapping 方案填片段跑一发 mma,
// 回读 D[16][8] 供 CPU 对拍穷举正确片段布局。
extern "C" __global__ void mma_mapping_probe(
    const signed char * __restrict__ a_in,
    const signed char * __restrict__ b_in,
    int * __restrict__ d_out,
    const unsigned int a_map,
    const unsigned int b_map)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int group = lane >> 2;
    const unsigned int tig = lane & 3u;
    int a0, a1, a2, a3, b0, b1;
    if (a_map == 0u) {
        a0 = *reinterpret_cast<const int *>(a_in + group * 32u + tig * 8u);
        a1 = *reinterpret_cast<const int *>(a_in + group * 32u + tig * 8u + 4u);
        a2 = *reinterpret_cast<const int *>(a_in + (group + 8u) * 32u + tig * 8u);
        a3 = *reinterpret_cast<const int *>(a_in + (group + 8u) * 32u + tig * 8u + 4u);
    } else {
        a0 = *reinterpret_cast<const int *>(a_in + group * 32u + tig * 4u);
        a1 = *reinterpret_cast<const int *>(a_in + (group + 8u) * 32u + tig * 4u);
        a2 = *reinterpret_cast<const int *>(a_in + group * 32u + 16u + tig * 4u);
        a3 = *reinterpret_cast<const int *>(a_in + (group + 8u) * 32u + 16u + tig * 4u);
    }
    if (b_map == 0u) {
        b0 = *reinterpret_cast<const int *>(b_in + group * 32u + tig * 4u);
        b1 = *reinterpret_cast<const int *>(b_in + group * 32u + 16u + tig * 4u);
    } else {
        b0 = *reinterpret_cast<const int *>(b_in + tig * 32u + group * 4u);
        b1 = *reinterpret_cast<const int *>(b_in + (4u + tig) * 32u + group * 4u);
    }
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    // 原样转储 32 thread × 4 寄存器,映射由 host 对拍穷举。
    d_out[lane * 4u + 0u] = d0;
    d_out[lane * 4u + 1u] = d1;
    d_out[lane * 4u + 2u] = d2;
    d_out[lane * 4u + 3u] = d3;
}

// ===== Q4_0 × Q8_1 MMQ(参考 llama.cpp mmq 路径):W4A8 int8 tensor core GEMM =====
// prefill 多行替代"反量化成 f16 全量物化 + cuBLAS":直接吃 packed 权重
// (6.26GB 读一次,无 25GB f16 往返)。结构:
//   - 激活复用 q8_1_quantize_rows_f16 的 36B 块(codes[32] + f16 scale);
//   - warp 输出 16×8 tile,mma.sync.m16n8k32.s8 一个 k 块(32 元素)一发,
//     块尺度 d_w×d_a 在 s32 累加器外乘(PTX ISA 标准 m16n8k32 片段布局);
//   - 权重 nibble 用 __vsubss4 一条指令展开 4 个 (q-8) int8;
//   - block 8 warp 横向铺 64 列共享 16×32 激活 tile(smem ~2.6KB)。
extern "C" __global__ void q4_0_q8_1_mmq_s8(
    const unsigned char * __restrict__ weight,
    const unsigned char * __restrict__ activation, // q4_0_mmvq_q8_block[rows][blocks]
    __half * __restrict__ output,
    const unsigned int rows,        // M(激活行)
    const unsigned int out_rows,    // N(权重行)
    const unsigned int columns,     // K(3840 等,32 对齐)
    const float scale_output)       // 输出整体缩放(当前恒 1.0,留作 fused 挂点)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int group = lane >> 2;          // 0..7
    const unsigned int tig = lane & 3u;            // thread-in-group
    const unsigned int blocks = columns / 32u;
    const unsigned int m_base = blockIdx.y * 16u;  // 本 block 的 16 行激活
    const unsigned int n_base = blockIdx.x * 64u + warp * 8u; // 本 warp 的 8 列权重行

    __shared__ signed char a_codes[16][32];
    __shared__ float a_scale[16];
    // B tile 每 warp 独立(8 warp × 8 权重行 × 32 k)。
    __shared__ signed char b_codes[8][8][32];
    __shared__ float b_scale[8][8];

    // C 片段归属:行 {group, group+8} × 列 {tig*2, tig*2+1}。
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (unsigned int kb = 0; kb < blocks; ++kb) {
        // 合作加载激活 tile:每 thread 负责 (m, 8 字节) 片段。
        for (unsigned int index = threadIdx.x; index < 16u * 4u; index += blockDim.x) {
            const unsigned int m = index / 4u;
            const unsigned int seg = index - m * 4u; // 8B 段
            const unsigned int m_global = m_base + m;
            if (m_global < rows) {
                const q4_0_mmvq_q8_block *act = reinterpret_cast<const q4_0_mmvq_q8_block *>(activation) + (unsigned long long)m_global * blocks + kb;
                // 36B 块内 codes 只保证 4 字节对齐,用两个 int 读 8B。
                int *dst = reinterpret_cast<int *>(a_codes[m] + seg * 8u);
                const int *srcp = reinterpret_cast<const int *>(act->codes + seg * 8u);
                dst[0] = srcp[0];
                dst[1] = srcp[1];
                if (seg == 0u) a_scale[m] = __half2float(act->scale);
            } else {
                int *dst = reinterpret_cast<int *>(a_codes[m] + seg * 8u);
                dst[0] = 0;
                dst[1] = 0;
                if (seg == 0u) a_scale[m] = 0.0f;
            }
        }
        // 每 warp 各自 8 个权重行:32 thread = 8 行 × 4 段(4B codes)。
        {
            // 32 thread = 8 行 × 4 段;权重码区 2 字节对齐,ushort 组合读。
            const unsigned int n_local = lane >> 2;
            const unsigned int seg = lane & 3u;
            if (n_base + n_local < out_rows) {
                const unsigned char *block_bytes = weight + ((unsigned long long)(n_base + n_local) * blocks + kb) * 18u;
                b_scale[warp][n_local] = __half2float(*reinterpret_cast<const __half *>(block_bytes));
                const unsigned short *codes16 = reinterpret_cast<const unsigned short *>(block_bytes + 2u + seg * 4u);
                const unsigned int packed = unsigned(codes16[0]) | (unsigned(codes16[1]) << 16);
                // 低 nibble = 元素 seg*4+i;高 nibble = 元素 16+seg*4+i。
                *reinterpret_cast<unsigned int *>(b_codes[warp][n_local] + seg * 4u) = __vsubss4((packed >> 0) & 0x0F0F0F0Fu, 0x08080808u);
                *reinterpret_cast<unsigned int *>(b_codes[warp][n_local] + 16u + seg * 4u) = __vsubss4((packed >> 4) & 0x0F0F0F0Fu, 0x08080808u);
            } else {
                b_scale[warp][n_local] = 0.0f;
                *reinterpret_cast<unsigned int *>(b_codes[warp][n_local] + seg * 4u) = 0u;
                *reinterpret_cast<unsigned int *>(b_codes[warp][n_local] + 16u + seg * 4u) = 0u;
            }
        }
        __syncthreads();

        // m16n8k32 s8 片段(mma_mapping_probe 实测穷举确定):
        //   a0 = A[group][4*tig..+3], a1 = A[group+8][同 k], a2 = A[group][16+4*tig], a3 = A[group+8][16+4*tig]
        //   b0 = B(=W^T)[n=group][4*tig..], b1 = [n=group][16+4*tig..]
        //   D: 行{group, group+8} × 列{tig*2, tig*2+1}
        int a0 = *reinterpret_cast<const int *>(a_codes[group] + tig * 4u);
        int a1 = *reinterpret_cast<const int *>(a_codes[group + 8u] + tig * 4u);
        int a2 = *reinterpret_cast<const int *>(a_codes[group] + 16u + tig * 4u);
        int a3 = *reinterpret_cast<const int *>(a_codes[group + 8u] + 16u + tig * 4u);
        int b0 = *reinterpret_cast<const int *>(b_codes[warp][group] + tig * 4u);
        int b1 = *reinterpret_cast<const int *>(b_codes[warp][group] + 16u + tig * 4u);
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        // 块尺度:acc[r][c] += d_a[m_row] * d_w[n_col] * d。
        const float sa0 = a_scale[group] * b_scale[warp][tig * 2u];
        const float sa1 = a_scale[group] * b_scale[warp][tig * 2u + 1u];
        const float sa2 = a_scale[group + 8u] * b_scale[warp][tig * 2u];
        const float sa3 = a_scale[group + 8u] * b_scale[warp][tig * 2u + 1u];
        acc[0][0] += sa0 * float(d0);
        acc[0][1] += sa1 * float(d1);
        acc[1][0] += sa2 * float(d2);
        acc[1][1] += sa3 * float(d3);
        __syncthreads();
    }

    // 输出 [rows, out_rows] 行优先。
    const unsigned int out_col0 = n_base + tig * 2u;
    if (m_base + group < rows && out_col0 < out_rows) output[(unsigned long long)(m_base + group) * out_rows + out_col0] = __float2half(acc[0][0] * scale_output);
    if (m_base + group < rows && out_col0 + 1u < out_rows) output[(unsigned long long)(m_base + group) * out_rows + out_col0 + 1u] = __float2half(acc[0][1] * scale_output);
    if (m_base + group + 8u < rows && out_col0 < out_rows) output[(unsigned long long)(m_base + group + 8u) * out_rows + out_col0] = __float2half(acc[1][0] * scale_output);
    if (m_base + group + 8u < rows && out_col0 + 1u < out_rows) output[(unsigned long long)(m_base + group + 8u) * out_rows + out_col0 + 1u] = __float2half(acc[1][1] * scale_output);
}


__device__ __forceinline__ void q4_k_block_dot_pair(
    const unsigned char *gate_block,
    const unsigned char *up_block,
    const __half *input,
    float *gate_sum,
    float *up_sum)
{
    const unsigned int lane = threadIdx.x & 31;
    const float gate_d = __half2float(*reinterpret_cast<const __half *>(gate_block));
    const float gate_dmin = __half2float(*reinterpret_cast<const __half *>(gate_block + 2));
    const float up_d = __half2float(*reinterpret_cast<const __half *>(up_block));
    const float up_dmin = __half2float(*reinterpret_cast<const __half *>(up_block + 2));
    const unsigned char *gate_scales = gate_block + 4;
    const unsigned char *up_scales = up_block + 4;
    const unsigned char *gate_quants = gate_block + 16;
    const unsigned char *up_quants = up_block + 16;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    #pragma unroll
    for (unsigned int group = 0; group < 8; ++group) {
        unsigned int gate_scale, gate_min, up_scale, up_min;
        q4_k_scale_min(gate_scales, group, &gate_scale, &gate_min);
        q4_k_scale_min(up_scales, group, &up_scale, &up_min);
        const unsigned int source = (group >> 1) * 32 + lane;
        const unsigned char gate_packed = gate_quants[source];
        const unsigned char up_packed = up_quants[source];
        const unsigned int shift = (group & 1) * 4;
        const unsigned int gate_quant = (gate_packed >> shift) & 0x0f;
        const unsigned int up_quant = (up_packed >> shift) & 0x0f;
        const float x = __half2float(input[group * 32 + lane]);
        gate_acc += x * (gate_d * float(gate_scale * gate_quant) - gate_dmin * float(gate_min));
        up_acc += x * (up_d * float(up_scale * up_quant) - up_dmin * float(up_min));
    }
    *gate_sum = gate_acc;
    *up_sum = up_acc;
}
__device__ __forceinline__ void q4_k_block_dot_pair_rows8(
    const unsigned char *gate_block,
    const unsigned char *up_block,
    const __half *input,
    const unsigned int input_base,
    const unsigned int input_rows,
    const unsigned int input_columns,
    const unsigned int block_column,
    float *gate_sums,
    float *up_sums)
{
    const unsigned int lane = threadIdx.x & 31;
    const float gate_d = __half2float(*reinterpret_cast<const __half *>(gate_block));
    const float gate_dmin = __half2float(*reinterpret_cast<const __half *>(gate_block + 2));
    const float up_d = __half2float(*reinterpret_cast<const __half *>(up_block));
    const float up_dmin = __half2float(*reinterpret_cast<const __half *>(up_block + 2));
    const unsigned char *gate_scales = gate_block + 4;
    const unsigned char *up_scales = up_block + 4;
    const unsigned char *gate_quants = gate_block + 16;
    const unsigned char *up_quants = up_block + 16;
    #pragma unroll
    for (unsigned int group = 0; group < 8; ++group) {
        unsigned int gate_scale, gate_min, up_scale, up_min;
        q4_k_scale_min(gate_scales, group, &gate_scale, &gate_min);
        q4_k_scale_min(up_scales, group, &up_scale, &up_min);
        const unsigned int source = (group >> 1) * 32 + lane;
        const unsigned int shift = (group & 1) * 4;
        const unsigned int gate_quant = (gate_quants[source] >> shift) & 0x0f;
        const unsigned int up_quant = (up_quants[source] >> shift) & 0x0f;
        const float gate_value = gate_d * float(gate_scale * gate_quant) - gate_dmin * float(gate_min);
        const float up_value = up_d * float(up_scale * up_quant) - up_dmin * float(up_min);
        #pragma unroll
        for (unsigned int local_row = 0; local_row < 8; ++local_row) {
            if (input_base + local_row < input_rows) {
                const unsigned long long input_index = (unsigned long long)(input_base + local_row) * input_columns + block_column + group * 32 + lane;
                const float x = __half2float(input[input_index]);
                gate_sums[local_row] += x * gate_value;
                up_sums[local_row] += x * up_value;
            }
        }
    }
}
// Q4_0 双矩阵 block dot:lane 覆盖 32 元素块的 1 列,scale 每块只解码一次。
__device__ __forceinline__ void q4_0_block_dot_pair(
    const unsigned char *gate_block,
    const unsigned char *up_block,
    const __half *input,
    float *gate_sum,
    float *up_sum)
{
    const unsigned int lane = threadIdx.x & 31;
    const float gate_d = __half2float(*reinterpret_cast<const __half *>(gate_block));
    const float up_d = __half2float(*reinterpret_cast<const __half *>(up_block));
    const unsigned int shift = (lane >> 4) * 4;
    const unsigned char gate_packed = gate_block[2 + (lane & 15)];
    const unsigned char up_packed = up_block[2 + (lane & 15)];
    const float gate_value = gate_d * float(int((gate_packed >> shift) & 15) - 8);
    const float up_value = up_d * float(int((up_packed >> shift) & 15) - 8);
    const float x = __half2float(input[lane]);
    *gate_sum = x * gate_value;
    *up_sum = x * up_value;
}
__device__ __forceinline__ void q4_0_block_dot_pair_rows8(
    const unsigned char *gate_block,
    const unsigned char *up_block,
    const __half *input,
    const unsigned int input_base,
    const unsigned int input_rows,
    const unsigned int input_columns,
    const unsigned int block_column,
    float *gate_sums,
    float *up_sums)
{
    const unsigned int lane = threadIdx.x & 31;
    const float gate_d = __half2float(*reinterpret_cast<const __half *>(gate_block));
    const float up_d = __half2float(*reinterpret_cast<const __half *>(up_block));
    const unsigned int shift = (lane >> 4) * 4;
    const unsigned char gate_packed = gate_block[2 + (lane & 15)];
    const unsigned char up_packed = up_block[2 + (lane & 15)];
    const float gate_value = gate_d * float(int((gate_packed >> shift) & 15) - 8);
    const float up_value = up_d * float(int((up_packed >> shift) & 15) - 8);
    #pragma unroll
    for (unsigned int local_row = 0; local_row < 8; ++local_row) {
        if (input_base + local_row < input_rows) {
            const unsigned long long input_index = (unsigned long long)(input_base + local_row) * input_columns + block_column + lane;
            const float x = __half2float(input[input_index]);
            gate_sums[local_row] += x * gate_value;
            up_sums[local_row] += x * up_value;
        }
    }
}
__device__ __forceinline__ float q5_k_block_dot(
    const unsigned char *block,
    const __half *input)
{
    const unsigned int lane = threadIdx.x & 31;
    const float d = __half2float(*reinterpret_cast<const __half *>(block));
    const float dmin = __half2float(*reinterpret_cast<const __half *>(block + 2));
    const unsigned char *scales = block + 4;
    const unsigned char *high_bits = block + 16;
    const unsigned char *low_bits = block + 48;
    float total = 0.0f;
    #pragma unroll
    for (unsigned int group = 0; group < 8; ++group) {
        unsigned int scale, minimum;
        q4_k_scale_min(scales, group, &scale, &minimum);
        const unsigned int source = (group >> 1) * 32 + lane;
        const unsigned int shift = (group & 1) * 4;
        const unsigned int low = (low_bits[source] >> shift) & 0x0f;
        const unsigned int quant = low + ((high_bits[lane] & (1u << group)) ? 16u : 0u);
        const float value = d * float(scale * quant) - dmin * float(minimum);
        total += __half2float(input[group * 32 + lane]) * value;
    }
    return total;
}
__device__ __forceinline__ float q5_k_row_dot(
    const __half *input,
    const unsigned char *weight,
    const unsigned int col,
    const unsigned int row,
    const unsigned int blocks_per_row,
    float *partial)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    if (warp < blocks_per_row) {
        const unsigned long long block_index = (unsigned long long)col * blocks_per_row + warp;
        float sum = q5_k_block_dot(weight + block_index * 176, input + ((unsigned long long)row * blocks_per_row + warp) * 256);
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) partial[warp] = sum;
    }
    __syncthreads();
    float sum = 0.0f;
    if (warp == 0) {
        sum = lane < blocks_per_row ? partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    return sum;
}
__device__ __forceinline__ float q6_k_block_dot(
    const unsigned char *block,
    const __half *input)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned char *low_bits = block;
    const unsigned char *high_bits = block + 128;
    const signed char *scales = reinterpret_cast<const signed char *>(block + 192);
    const float d = __half2float(*reinterpret_cast<const __half *>(block + 208));
    float total = 0.0f;
    #pragma unroll
    for (unsigned int half = 0; half < 2; ++half) {
        const unsigned int low = half * 64;
        const unsigned int high = half * 32;
        const unsigned int scale = half * 8;
        const unsigned int target = half * 128;
        const unsigned int scale_index = lane >> 4;
        const unsigned int high_value = high_bits[high + lane];
        const int q1 = int((low_bits[low + lane] & 0x0f) | (((high_value >> 0) & 3) << 4)) - 32;
        const int q2 = int((low_bits[low + lane + 32] & 0x0f) | (((high_value >> 2) & 3) << 4)) - 32;
        const int q3 = int((low_bits[low + lane] >> 4) | (((high_value >> 4) & 3) << 4)) - 32;
        const int q4 = int((low_bits[low + lane + 32] >> 4) | (((high_value >> 6) & 3) << 4)) - 32;
        total += __half2float(input[target + lane]) * d * float(scales[scale + scale_index]) * float(q1);
        total += __half2float(input[target + lane + 32]) * d * float(scales[scale + scale_index + 2]) * float(q2);
        total += __half2float(input[target + lane + 64]) * d * float(scales[scale + scale_index + 4]) * float(q3);
        total += __half2float(input[target + lane + 96]) * d * float(scales[scale + scale_index + 6]) * float(q4);
    }
    return total;
}
__device__ __forceinline__ float q6_k_row_dot(
    const __half *input,
    const unsigned char *weight,
    const unsigned int col,
    const unsigned int row,
    const unsigned int blocks_per_row,
    float *partial)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    if (warp < blocks_per_row) {
        const unsigned long long block_index = (unsigned long long)col * blocks_per_row + warp;
        float sum = q6_k_block_dot(weight + block_index * 210, input + ((unsigned long long)row * blocks_per_row + warp) * 256);
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) partial[warp] = sum;
    }
    __syncthreads();
    float sum = 0.0f;
    if (warp == 0) {
        sum = lane < blocks_per_row ? partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    return sum;
}
extern "C" __global__ void gated_linear_q4_k_silu_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ gate,
    const unsigned char * __restrict__ up,
    __half * __restrict__ output,
    const unsigned int in_cols,
    const unsigned int out_cols,
    const unsigned int blocks_per_row)
{
    const unsigned int col = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (col >= out_cols) return;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    extern __shared__ float partial[];
    float gate_total = 0.0f;
    float up_total = 0.0f;
    for (unsigned int block = warp; block < blocks_per_row; block += warp_count) {
        const unsigned long long block_index = (unsigned long long)col * blocks_per_row + block;
        float gate_sum, up_sum;
        q4_k_block_dot_pair(
            gate + block_index * 144,
            up + block_index * 144,
            input + ((unsigned long long)row * blocks_per_row + block) * 256,
            &gate_sum,
            &up_sum);
        gate_total += gate_sum;
        up_total += up_sum;
    }
    if (warp < warp_count) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_total += __shfl_down_sync(0xffffffff, gate_total, offset);
            up_total += __shfl_down_sync(0xffffffff, up_total, offset);
        }
        if (lane == 0) {
            partial[warp] = gate_total;
            partial[warp_count + warp] = up_total;
        }
    }
    __syncthreads();
    if (warp == 0) {
        float gate_sum = lane < warp_count ? partial[lane] : 0.0f;
        float up_sum = lane < warp_count ? partial[warp_count + lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
            up_sum += __shfl_down_sync(0xffffffff, up_sum, offset);
        }
        if (lane == 0) {
            output[(unsigned long long)row * out_cols + col] = __float2half((gate_sum / (1.0f + expf(-gate_sum))) * up_sum);
        }
    }
}
extern "C" __global__ void gated_linear_q4_k_silu_rows8_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ gate,
    const unsigned char * __restrict__ up,
    __half * __restrict__ output,
    const unsigned int input_rows,
    const unsigned int input_columns,
    const unsigned int output_columns,
    const unsigned int blocks_per_row)
{
    const unsigned int output_column = blockIdx.x;
    const unsigned int input_base = blockIdx.y * 8;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    if (output_column >= output_columns) return;
    float gate_sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up_sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int block = warp; block < blocks_per_row; block += warp_count) {
        const unsigned long long weight_block = ((unsigned long long)output_column * blocks_per_row + block) * 144u;
        q4_k_block_dot_pair_rows8(
            gate + weight_block,
            up + weight_block,
            input,
            input_base,
            input_rows,
            input_columns,
            block * 256,
            gate_sums,
            up_sums);
    }
    #pragma unroll
    for (unsigned int local_row = 0; local_row < 8; ++local_row) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sums[local_row] += __shfl_down_sync(0xffffffff, gate_sums[local_row], offset);
            up_sums[local_row] += __shfl_down_sync(0xffffffff, up_sums[local_row], offset);
        }
    }
    __shared__ float partial[2][8][8];
    if (lane == 0) {
        #pragma unroll
        for (unsigned int local_row = 0; local_row < 8; ++local_row) {
            partial[0][local_row][warp] = gate_sums[local_row];
            partial[1][local_row][warp] = up_sums[local_row];
        }
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (unsigned int local_row = 0; local_row < 8; ++local_row) {
            float gate_sum = lane < warp_count ? partial[0][local_row][lane] : 0.0f;
            float up_sum = lane < warp_count ? partial[1][local_row][lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
                up_sum += __shfl_down_sync(0xffffffff, up_sum, offset);
            }
            if (lane == 0 && input_base + local_row < input_rows) {
                output[(unsigned long long)(input_base + local_row) * output_columns + output_column] = __float2half((gate_sum / (1.0f + expf(-gate_sum))) * up_sum);
            }
        }
    }
}
extern "C" __global__ void gated_linear_q4_0_silu_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ gate,
    const unsigned char * __restrict__ up,
    __half * __restrict__ output,
    const unsigned int in_cols,
    const unsigned int out_cols,
    const unsigned int blocks_per_row)
{
    const unsigned int col = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (col >= out_cols) return;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    extern __shared__ float partial[];
    float gate_total = 0.0f;
    float up_total = 0.0f;
    for (unsigned int block = warp; block < blocks_per_row; block += warp_count) {
        const unsigned long long block_index = (unsigned long long)col * blocks_per_row + block;
        float gate_sum, up_sum;
        q4_0_block_dot_pair(
            gate + block_index * 18,
            up + block_index * 18,
            input + ((unsigned long long)row * blocks_per_row + block) * 32,
            &gate_sum,
            &up_sum);
        gate_total += gate_sum;
        up_total += up_sum;
    }
    if (warp < warp_count) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_total += __shfl_down_sync(0xffffffff, gate_total, offset);
            up_total += __shfl_down_sync(0xffffffff, up_total, offset);
        }
        if (lane == 0) {
            partial[warp] = gate_total;
            partial[warp_count + warp] = up_total;
        }
    }
    __syncthreads();
    if (warp == 0) {
        float gate_sum = lane < warp_count ? partial[lane] : 0.0f;
        float up_sum = lane < warp_count ? partial[warp_count + lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
            up_sum += __shfl_down_sync(0xffffffff, up_sum, offset);
        }
        if (lane == 0) {
            output[(unsigned long long)row * out_cols + col] = __float2half((gate_sum / (1.0f + expf(-gate_sum))) * up_sum);
        }
    }
}
extern "C" __global__ void gated_linear_q4_0_silu_rows8_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ gate,
    const unsigned char * __restrict__ up,
    __half * __restrict__ output,
    const unsigned int input_rows,
    const unsigned int input_columns,
    const unsigned int output_columns,
    const unsigned int blocks_per_row)
{
    const unsigned int output_column = blockIdx.x;
    const unsigned int input_base = blockIdx.y * 8;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    if (output_column >= output_columns) return;
    float gate_sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up_sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int block = warp; block < blocks_per_row; block += warp_count) {
        const unsigned long long weight_block = ((unsigned long long)output_column * blocks_per_row + block) * 18u;
        q4_0_block_dot_pair_rows8(
            gate + weight_block,
            up + weight_block,
            input,
            input_base,
            input_rows,
            input_columns,
            block * 32,
            gate_sums,
            up_sums);
    }
    #pragma unroll
    for (unsigned int local_row = 0; local_row < 8; ++local_row) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sums[local_row] += __shfl_down_sync(0xffffffff, gate_sums[local_row], offset);
            up_sums[local_row] += __shfl_down_sync(0xffffffff, up_sums[local_row], offset);
        }
    }
    __shared__ float partial[2][8][8];
    if (lane == 0) {
        #pragma unroll
        for (unsigned int local_row = 0; local_row < 8; ++local_row) {
            partial[0][local_row][warp] = gate_sums[local_row];
            partial[1][local_row][warp] = up_sums[local_row];
        }
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (unsigned int local_row = 0; local_row < 8; ++local_row) {
            float gate_sum = lane < warp_count ? partial[0][local_row][lane] : 0.0f;
            float up_sum = lane < warp_count ? partial[1][local_row][lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
                up_sum += __shfl_down_sync(0xffffffff, up_sum, offset);
            }
            if (lane == 0 && input_base + local_row < input_rows) {
                output[(unsigned long long)(input_base + local_row) * output_columns + output_column] = __float2half((gate_sum / (1.0f + expf(-gate_sum))) * up_sum);
            }
        }
    }
}
// Q4_0 单矩阵 rows8 block dot:与 gated 版同构,去掉 gate/up 双路;prefill 每 8 个
// token 复用一次 weight 解码,替代通用 matmul 的逐元素 gguf_kq_value 重解码。
__device__ __forceinline__ void q4_0_block_dot_rows8(
    const unsigned char *block_bytes,
    const __half *input,
    const unsigned int input_base,
    const unsigned int input_rows,
    const unsigned int input_columns,
    const unsigned int block_column,
    float *sums)
{
    const unsigned int lane = threadIdx.x & 31;
    const float d = __half2float(*reinterpret_cast<const __half *>(block_bytes));
    const unsigned char packed = block_bytes[2u + (lane & 15u)];
    const unsigned int nibble = lane < 16u ? (packed & 15u) : (packed >> 4);
    const float value = d * float(int(nibble) - 8);
    #pragma unroll
    for (unsigned int local_row = 0; local_row < 8; ++local_row) {
        if (input_base + local_row < input_rows) {
            sums[local_row] += __half2float(input[(unsigned long long)(input_base + local_row) * input_columns + block_column + lane]) * value;
        }
    }
}
extern "C" __global__ void linear_q4_0_matmul_rows8_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int input_rows,
    const unsigned int input_columns,
    const unsigned int output_rows,
    const unsigned int blocks_per_row)
{
    const unsigned int output_row = blockIdx.x;
    const unsigned int input_base = blockIdx.y * 8;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = blockDim.x >> 5;
    if (output_row >= output_rows) return;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // 每 warp 认领行内一段连续 block(warp 间步进的跨行读会把 DRAM 打成碎片)。
    const unsigned int per_warp = (blocks_per_row + warp_count - 1u) / warp_count;
    const unsigned int block_begin = warp * per_warp;
    const unsigned int block_end = min(block_begin + per_warp, blocks_per_row);
    for (unsigned int block = block_begin; block < block_end; ++block) {
        q4_0_block_dot_rows8(
            weight + ((unsigned long long)output_row * blocks_per_row + block) * 18u,
            input,
            input_base,
            input_rows,
            input_columns,
            block * 32,
            sums);
    }
    #pragma unroll
    for (unsigned int local_row = 0; local_row < 8; ++local_row) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[local_row] += __shfl_down_sync(0xffffffff, sums[local_row], offset);
        }
    }
    __shared__ float partial[8][8];
    if (lane == 0) {
        #pragma unroll
        for (unsigned int local_row = 0; local_row < 8; ++local_row) {
            partial[local_row][warp] = sums[local_row];
        }
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (unsigned int local_row = 0; local_row < 8; ++local_row) {
            float sum = lane < warp_count ? partial[local_row][lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0 && input_base + local_row < input_rows) {
                output[(unsigned long long)(input_base + local_row) * output_rows + output_row] = __float2half(sum);
            }
        }
    }
}
// Q4_0 反量化到 f16:每线程连续展开 4 个元素(可能跨块,逐元素定位)。
// prefill 多 token 时先反量化一次再喂 cuBLAS tensor core
// (标量直算 kernel 在 1024-token chunk 上是 ALU 瓶颈)。
// 每线程一个完整 32 元素块:uchar4 向量读 16 字节码,零除法。
extern "C" __global__ void q4_0_dequant_f16(
    const unsigned char * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int blocks)
{
    for (unsigned int block = blockIdx.x * blockDim.x + threadIdx.x; block < blocks; block += gridDim.x * blockDim.x) {
        const unsigned char *b = weight + (unsigned long long)block * 18u;
        const float d = __half2float(*reinterpret_cast<const __half *>(b));
        // 块内 code 区起始地址只保证 2 字节对齐(18B 步进),用 uchar2 逐对读取。
        const uchar2 p[8] = {
            *reinterpret_cast<const uchar2 *>(b + 2u), *reinterpret_cast<const uchar2 *>(b + 4u),
            *reinterpret_cast<const uchar2 *>(b + 6u), *reinterpret_cast<const uchar2 *>(b + 8u),
            *reinterpret_cast<const uchar2 *>(b + 10u), *reinterpret_cast<const uchar2 *>(b + 12u),
            *reinterpret_cast<const uchar2 *>(b + 14u), *reinterpret_cast<const uchar2 *>(b + 16u),
        };
        __half *out = output + (unsigned long long)block * 32u;
        #pragma unroll
        for (unsigned int pair = 0; pair < 8u; ++pair) {
            const unsigned int base = pair * 2u;
            out[base] = __float2half(d * float(int(p[pair].x & 15u) - 8));
            out[base + 16u] = __float2half(d * float(int(p[pair].x >> 4) - 8));
            out[base + 1u] = __float2half(d * float(int(p[pair].y & 15u) - 8));
            out[base + 17u] = __float2half(d * float(int(p[pair].y >> 4) - 8));
        }
    }
}
extern "C" __global__ void linear_q5_k_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int out_cols,
    const unsigned int blocks_per_row)
{
    const unsigned int col = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (col >= out_cols) return;
    extern __shared__ float partial[];
    const float sum = q5_k_row_dot(input, weight, col, row, blocks_per_row, partial);
    if (threadIdx.x == 0) {
        output[(unsigned long long)row * out_cols + col] = __float2half(sum);
    }
}
extern "C" __global__ void linear_q5_k_accumulate_f32(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ weight,
    float * __restrict__ output,
    const float * __restrict__ route_weights,
    const unsigned int route,
    const unsigned int out_cols,
    const unsigned int blocks_per_row)
{
    const unsigned int col = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (col >= out_cols) return;
    extern __shared__ float partial[];
    const float sum = q5_k_row_dot(input, weight, col, row, blocks_per_row, partial);
    if (threadIdx.x == 0) {
        output[(unsigned long long)row * out_cols + col] += __half2float(__float2half(sum)) * route_weights[route];
    }
}
extern "C" __global__ void linear_q6_k_f16(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ weight,
    __half * __restrict__ output,
    const unsigned int out_cols,
    const unsigned int blocks_per_row)
{
    const unsigned int col = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (col >= out_cols) return;
    extern __shared__ float partial[];
    const float sum = q6_k_row_dot(input, weight, col, row, blocks_per_row, partial);
    if (threadIdx.x == 0) {
        output[(unsigned long long)row * out_cols + col] = __float2half(sum);
    }
}
extern "C" __global__ void linear_q6_k_accumulate_f32(
    const __half * __restrict__ input,
    const unsigned char * __restrict__ weight,
    float * __restrict__ output,
    const float * __restrict__ route_weights,
    const unsigned int route,
    const unsigned int out_cols,
    const unsigned int blocks_per_row)
{
    const unsigned int col = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (col >= out_cols) return;
    extern __shared__ float partial[];
    const float sum = q6_k_row_dot(input, weight, col, row, blocks_per_row, partial);
    if (threadIdx.x == 0) {
        output[(unsigned long long)row * out_cols + col] += __half2float(__float2half(sum)) * route_weights[route];
    }
}
"#;

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use cudarc::cublas::Gemm;

use super::diffusion::{cast_f16_to_f32, cast_f32_to_f16_slice};
use super::tensor::grid_1d;
use super::{CudaContext, CudaSliceF16, CudaTensor, GemmConfig, LaunchConfig, PushKernelArg, THREADS, sys};

/// cuBLAS 在每个 unique (M, N, K) shape 首次调用时做 heuristic search(~30s for M=19300)。
/// 重复 matmul shape 在多层之间保持一致,所以 warmup 一次性付出后,后续 100 次
/// (2 steps × 50 layers) 都走 cached tensor-core kernel(~3ms)。
/// 此 set 记录已 warm 的 shape (M, N, K) 防止重复 warm。
static HGEMM_WARMED: OnceLock<Mutex<HashSet<(i32, i32, i32, bool)>>> = OnceLock::new();
fn warmed_set() -> &'static Mutex<HashSet<(i32, i32, i32, bool)>> {
    HGEMM_WARMED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// 对给定 GEMM shape 做一次 hgemm warmup；模型 runtime 负责给出需要预热的 shape。
pub fn prewarm_hgemm(ctx: &CudaContext, output_columns: usize, rows: usize, input_columns: usize, output_f32: bool) {
    cublas_warmup_hgemm_ex(ctx, output_columns as i32, rows as i32, input_columns as i32, output_f32);
}

/// `c_is_f32`: warmup 输出 buffer 用 f32 还是 f16。cuBLAS 的 heuristic cache
/// key 包含输出 dtype;warmup 必须用与真实调用一致的 dtype,否则 cache miss。
/// 现在 cublas_matmul_f32_via_hgemm_ex 用 f32 output,其余走 f16。
fn cublas_warmup_hgemm_ex(ctx: &CudaContext, m: i32, n: i32, k: i32, c_is_f32: bool) {
    let key = (m, n, k, c_is_f32);
    {
        let warmed = warmed_set().lock().unwrap();
        if warmed.contains(&key) {
            return;
        }
    }
    let _wt = std::time::Instant::now();
    eprintln!("[warmup] start shape={:?} c_f32={}", key, c_is_f32);
    // 同步分配 + 同步释放,绕开 cudarc 的 cuMemAllocAsync pool。
    // 12GB 设备上预热 c_f32=true 的 (28672, 19348) gate_up 输出需要 2.2GB,
    // 若用 pool 路径预热,cudarc 不释放回 OS,后续 DiT 第一次 alloc 同尺寸会 OOM。
    let a_bytes = (k as u64 * m as u64 * 2) as usize;
    let b_bytes = (k as u64 * n as u64 * 2) as usize;
    let c_bytes = (m as u64 * n as u64 * if c_is_f32 { 4 } else { 2 }) as usize;
    let a_ptr = unsafe { cudarc::driver::result::malloc_sync(a_bytes) };
    let b_ptr = unsafe { cudarc::driver::result::malloc_sync(b_bytes) };
    let c_ptr = unsafe { cudarc::driver::result::malloc_sync(c_bytes) };
    let (a_ptr, b_ptr, c_ptr) = match (a_ptr, b_ptr, c_ptr) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => {
            // alloc 失败,放弃 warmup;记入 warmed 避免每次调用都重试 + 刷日志,
            // 真实 GEMM 路径自己会报 OOM。
            eprintln!("[warmup] alloc 失败 shape={key:?}，不再重试");
            if let Ok(p) = a_ptr {
                unsafe {
                    let _ = cudarc::driver::result::free_sync(p);
                }
            }
            if let Ok(p) = b_ptr {
                unsafe {
                    let _ = cudarc::driver::result::free_sync(p);
                }
            }
            if let Ok(p) = c_ptr {
                unsafe {
                    let _ = cudarc::driver::result::free_sync(p);
                }
            }
            warmed_set().lock().unwrap().insert(key);
            return;
        }
    };
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe {
        sys::cublasGemmEx(
            *ctx.blas_handle(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha as *const _ as *const std::ffi::c_void,
            a_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            k,
            b_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            k,
            &beta as *const _ as *const std::ffi::c_void,
            c_ptr as *mut _,
            if c_is_f32 { sys::cudaDataType_t::CUDA_R_32F } else { sys::cudaDataType_t::CUDA_R_16F },
            m,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    eprintln!("[warmup] status={:?}", status);
    // 同步等 heuristic 完工(否则 cuBLAS cache 还没写入就 free 不影响,但保险起见)
    let _ = ctx.synchronize();
    // 同步释放:不依赖 cudarc 的 Drop 路径。
    unsafe {
        let _ = cudarc::driver::result::free_sync(a_ptr);
        let _ = cudarc::driver::result::free_sync(b_ptr);
        let _ = cudarc::driver::result::free_sync(c_ptr);
    }
    eprintln!("[warmup] done shape={:?} wall_ms={:.1}", key, _wt.elapsed().as_secs_f64() * 1000.0);
    warmed_set().lock().unwrap().insert(key);
}

pub fn cublas_matmul_f16(ctx: &CudaContext, input: &CudaTensor, weight: &CudaSliceF16, out_cols: usize) -> Result<CudaTensor, String> {
    if weight.len() != out_cols * input.cols {
        return Err(format!("cuBLAS weight={}，期望 {}×{}", weight.len(), out_cols, input.cols));
    }
    // cuBLAS warmup:首次每个 unique (M, N, K) shape 会跑 ~30s heuristic;warmup 一次性
    // 避免每层首次调用都付 heuristic 代价。output_linear 同样可能进入 f16 路径,需要此保护。
    cublas_warmup_hgemm_ex(ctx, out_cols as i32, input.rows as i32, input.cols as i32, /*c_is_f32=*/ false);
    let mut output = ctx.tensor_alloc(input.rows, out_cols)?;
    use cudarc::driver::safe::{DevicePtr, DevicePtrMut};
    let stream = ctx.stream();
    let (weight_ptr, weight_sync) = weight.device_ptr(stream);
    let (input_ptr, input_sync) = input.slice.device_ptr(stream);
    let (output_ptr, output_sync) = output.slice.device_ptr_mut(stream);
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let status = unsafe {
        sys::cublasGemmEx(
            *ctx.blas_handle(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            out_cols as i32,
            input.rows as i32,
            input.cols as i32,
            &alpha as *const _ as *const std::ffi::c_void,
            weight_ptr as *const std::ffi::c_void,
            sys::cudaDataType_t::CUDA_R_16F,
            input.cols as i32,
            input_ptr as *const std::ffi::c_void,
            sys::cudaDataType_t::CUDA_R_16F,
            input.cols as i32,
            &beta as *const _ as *const std::ffi::c_void,
            output_ptr as *mut std::ffi::c_void,
            sys::cudaDataType_t::CUDA_R_16F,
            out_cols as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    drop(weight_sync);
    drop(input_sync);
    drop(output_sync);
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("cuBLAS F16-store/F32-accum GEMM 失败: status={status:?}"));
    }
    Ok(output)
}

/// 扩散 f32 激活的线性:f32 input(slice_f32)+ f16 weight → f32 输出。
/// cuBLAS handle 在 cudarc 0.19.8 是 pub(crate),无法走 mixed f16→f32 gemm_ex,
/// 故设备内把 f16 weight 转 f32 再 Sgemm(输出 f32,避免 down 投影 ~6e4 溢出)。
pub fn cublas_matmul_f32(ctx: &CudaContext, input: &CudaTensor, weight: &CudaSliceF16, out_cols: usize) -> Result<CudaTensor, String> {
    cublas_matmul_f32_via_hgemm_ex(ctx, input, weight, out_cols)
}

fn gguf_kq_layout(rows: usize, cols: usize, tensor_type: u32) -> Result<(usize, usize), String> {
    let block_bytes = match tensor_type {
        2 => 18,
        8 => 34,
        12 => 144,
        13 => 176,
        14 => 210,
        other => return Err(format!("CUDA GGUF K-quant 不支持 type={other}")),
    };
    let block_elements = if tensor_type == 8 || tensor_type == 2 { 32 } else { 256 };
    if rows == 0 || cols == 0 || !cols.is_multiple_of(block_elements) {
        return Err(format!("CUDA GGUF K-quant shape [{rows},{cols}] 非法"));
    }
    let elements = rows.checked_mul(cols).ok_or("CUDA GGUF K-quant 元素数溢出")?;
    let bytes = rows.checked_mul(cols / block_elements).and_then(|blocks| blocks.checked_mul(block_bytes)).ok_or("CUDA GGUF quant 字节数溢出")?;
    Ok((elements, bytes))
}

/// prefill/decode 都直接从 GGUF K-quant block 计算；同一行权重一次处理最多 8 个 token。
pub fn gguf_kq_matmul_f16(ctx: &CudaContext, input: &CudaTensor, weight: &cudarc::driver::safe::CudaSlice<u8>, out_rows: usize, tensor_type: u32) -> Result<CudaTensor, String> {
    let (_, expected) = gguf_kq_layout(out_rows, input.cols, tensor_type)?;
    if weight.len() != expected {
        return Err(format!("CUDA GGUF type={tensor_type} weight={}，期望 {expected}", weight.len()));
    }
    if tensor_type == 2 && (2..=8).contains(&input.rows) {
        // Q4_0 × Q8_1 MMVQ(llama.cpp 同款):2..8 行前向(MTP verify),权重只读
        // 一次,整数 dp4a 点积。decode(rows=1)保持 gemv,数值路径不变。
        return linear_q4_0_mmvq_f16(ctx, input, weight, out_rows);
    }
    if input.rows == 1 {
        let output = ctx.tensor_alloc(1, out_rows)?;
        let cols_u32 = input.cols as u32;
        let out_rows_u32 = out_rows as u32;
        if tensor_type == 2 {
            // Q4_0 专用 GEMV:warp=行 + smem 输入广播,LM head 等大行数矩阵的主力路径。
            let blocks_per_row = input.cols / 32;
            let func = ctx.function("linear_q4_0_gemv_f16")?;
            unsafe {
                ctx.stream()
                    .launch_builder(&func)
                    .arg(&input.slice)
                    .arg(weight)
                    .arg(&output.slice)
                    .arg(&cols_u32)
                    .arg(&out_rows_u32)
                    .arg(&(blocks_per_row as u32))
                    .launch(LaunchConfig { grid_dim: ((out_rows.div_ceil((THREADS / 32) as usize)) as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (input.cols * std::mem::size_of::<half::f16>()) as u32 })
                    .map_err(|error| format!("launch linear_q4_0_gemv_f16 失败: {error:?}"))?;
            }
            return Ok(output);
        }
        let func = ctx.function("gguf_kq_gemv_f16")?;
        unsafe {
            ctx.stream()
                .launch_builder(&func)
                .arg(&input.slice)
                .arg(weight)
                .arg(&output.slice)
                .arg(&cols_u32)
                .arg(&out_rows_u32)
                .arg(&tensor_type)
                .launch(LaunchConfig { grid_dim: (out_rows as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
                .map_err(|error| format!("launch gguf_kq_gemv_f16 失败: {error:?}"))?;
        }
        return Ok(output);
    }
    if tensor_type == 2 && input.rows > 16 && std::env::var_os("ZLLM_MMQ").is_none() {
        // Q4_0 大 chunk prefill:反量化一次成 f16 交 cuBLAS。MMQ(int8 mma,
        // linear_q4_0_mmq_s8)已数值验证但朴素分块仅 196 tok/s < 本路径 345,
        // 需 warp-tile/双缓冲工程后启用(ZLLM_MMQ=1 可强制对比)。
        let blocks = out_rows * (input.cols / 32);
        let mut f16_weight = ctx.buffer_uninit::<half::f16>(out_rows * input.cols)?;
        let func = ctx.function("q4_0_dequant_f16")?;
        let blocks_u32 = blocks as u32;
        unsafe {
            ctx.stream().launch_builder(&func).arg(weight).arg(&mut f16_weight).arg(&blocks_u32).launch(grid_1d(blocks.min(1 << 20))).map_err(|error| format!("launch q4_0_dequant_f16 失败: {error:?}"))?;
        }
        let result = cublas_matmul_f16(ctx, input, &f16_weight, out_rows);
        drop(f16_weight);
        return result;
    }
    if tensor_type == 2 && input.rows > 16 {
        return linear_q4_0_mmq_s8(ctx, input, weight, out_rows);
    }
    let output = ctx.tensor_alloc(input.rows, out_rows)?;
    if tensor_type == 2 {
        // Q4_0 小行数(MTP verify 等):直算 rows8 kernel,只读一次 packed 权重。
        let func = ctx.function("linear_q4_0_matmul_rows8_f16")?;
        let input_rows_u32 = input.rows as u32;
        let cols_u32 = input.cols as u32;
        let out_rows_u32 = out_rows as u32;
        let blocks_per_row_u32 = (input.cols / 32) as u32;
        unsafe {
            ctx.stream()
                .launch_builder(&func)
                .arg(&input.slice)
                .arg(weight)
                .arg(&output.slice)
                .arg(&input_rows_u32)
                .arg(&cols_u32)
                .arg(&out_rows_u32)
                .arg(&blocks_per_row_u32)
                .launch(LaunchConfig { grid_dim: (out_rows as u32, input.rows.div_ceil(8) as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
                .map_err(|error| format!("launch linear_q4_0_matmul_rows8_f16 失败: {error:?}(rows={}, cols={}, out_rows={})", input.rows, input.cols, out_rows))?;
        }
        return Ok(output);
    }
    let func = ctx.function("gguf_kq_matmul_f16")?;
    let input_rows_u32 = input.rows as u32;
    let cols_u32 = input.cols as u32;
    let out_rows_u32 = out_rows as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(weight)
            .arg(&output.slice)
            .arg(&input_rows_u32)
            .arg(&cols_u32)
            .arg(&out_rows_u32)
            .arg(&tensor_type)
            .launch(LaunchConfig { grid_dim: (out_rows as u32, input.rows.div_ceil(8) as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|error| format!("launch gguf_kq_matmul_f16 失败: {error:?}"))?;
    }
    Ok(output)
}

/// 精度敏感控制投影：F32 weight × F32 input，F32 累加并保留 F32 输出。
pub fn cublas_matmul_control_f32(ctx: &CudaContext, input: &CudaTensor, weight: &cudarc::driver::safe::CudaSlice<f32>, out_cols: usize) -> Result<CudaTensor, String> {
    if weight.len() != out_cols * input.cols {
        return Err(format!("CUDA control F32 weight={}，期望 {}×{}", weight.len(), out_cols, input.cols));
    }
    let converted_input = if input.slice_f32.is_none() { Some(cast_f16_to_f32(ctx, &input.slice, input.len())?) } else { None };
    let input_f32 = input.slice_f32.as_ref().or(converted_input.as_ref()).ok_or("CUDA control F32 input 不可用")?;
    let count = input.rows.checked_mul(out_cols).ok_or("CUDA control F32 输出溢出")?;
    let mut output = ctx.buffer_uninit_f32(count)?;
    unsafe {
        ctx.blas()
            .gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_cols as i32,
                    n: input.rows as i32,
                    k: input.cols as i32,
                    alpha: 1.0f32,
                    lda: input.cols as i32,
                    ldb: input.cols as i32,
                    beta: 0.0f32,
                    ldc: out_cols as i32,
                },
                weight,
                input_f32,
                &mut output,
            )
            .map_err(|error| format!("CUDA control cuBLAS Sgemm 失败: {error:?}"))?;
    }
    Ok(CudaTensor::new_f32_residual(output, ctx.placeholder_f16()?, input.rows, out_cols))
}

/// 扩散 f32 激活的线性:f32 input(slice_f32)+ f16 weight → f32 输出。
///
/// 走 cublasGemmEx(computeType=CUBLAS_COMPUTE_32F_FAST_TF32, weight=A=f16, input=B=f16,
/// output=C=f32):启用 TF32 tensor core,在 sm_86 (Ampere) 上 f16 输入走 tensor core 比纯
/// FP32 sgemm 快 ~10×。input f32 → f16 cast 是 lossy(>65504 饱和),但 mlp.gate_up_linear 输入
/// 来自 rmsnorm+adaln_modulate_segmented(归一后 O(1) × scale·shift O(1)),实测安全 f16。
/// weight 已经在 device 是 f16(dequant 后),无需再 cast。
pub fn cublas_matmul_f32_via_hgemm_ex(ctx: &CudaContext, input: &CudaTensor, weight: &CudaSliceF16, out_cols: usize) -> Result<CudaTensor, String> {
    let input_f32 = input.slice_f32.as_ref().ok_or("CUDA cublas_matmul_f32_via_hgemm_ex input 无 slice_f32")?;
    if weight.len() != out_cols * input.cols {
        return Err(format!("CUDA f32-via-tf32 weight={}，期望 {}×{}", weight.len(), out_cols, input.cols));
    }
    let input_count = input.rows.checked_mul(input.cols).ok_or("CUDA f32-via-tf32 input 溢出")?;
    // 0) cuBLAS warmup:首次每个 unique (M, N, K) shape 会跑 ~30s 的 heuristic search;
    //    warmup 一次后,后续同 shape 调用走 cached tensor-core kernel(~3ms)。H3 每层
    //    三次 matmul 的 shape 在 50 层完全一致,所以一次性 warmup 后,2 steps × 50 layers
    //    = 100 次后续调用都 fast。
    cublas_warmup_hgemm_ex(ctx, out_cols as i32, input.rows as i32, input.cols as i32, /*c_is_f32=*/ true);
    // 1) f32 → f16 cast(input)
    let input_f16 = cast_f32_to_f16_slice(ctx, input_f32, input_count)?;
    // 2) 分配 f32 output(CUBLAS_COMPUTE_32F 用 fp32 accumulation,避免 K=5376 累加溢出 f16)
    let count = input.rows.checked_mul(out_cols).ok_or("CUDA f32-via-tf32 输出溢出")?;
    let mut out_f32 = ctx.buffer_uninit_f32(count)?;
    use cudarc::driver::safe::{DevicePtr, DevicePtrMut};
    let stream = ctx.stream();
    let (weight_ptr, _w_sync) = weight.device_ptr(stream);
    let (input_ptr, _i_sync) = input_f16.device_ptr(stream);
    let (out_view, _o_sync) = (&mut out_f32 as &mut cudarc::driver::safe::CudaSlice<f32>).device_ptr_mut(stream);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe {
        sys::cublasGemmEx(
            *ctx.blas_handle(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            out_cols as i32,
            input.rows as i32,
            input.cols as i32,
            &alpha as *const _ as *const std::ffi::c_void,
            weight_ptr as *const std::ffi::c_void,
            sys::cudaDataType_t::CUDA_R_16F,
            input.cols as i32,
            input_ptr as *const std::ffi::c_void,
            sys::cudaDataType_t::CUDA_R_16F,
            input.cols as i32,
            &beta as *const _ as *const std::ffi::c_void,
            out_view as *mut std::ffi::c_void,
            sys::cudaDataType_t::CUDA_R_32F,
            out_cols as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    drop(_w_sync);
    drop(_i_sync);
    drop(_o_sync);
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("cuBLAS GemmEx hgemm 失败: status={status:?}"));
    }
    let placeholder = ctx.placeholder_f16()?;
    Ok(CudaTensor::new_f32_residual(out_f32, placeholder, input.rows, out_cols))
}

/// Q4_K gate/up 直接计算并融合 SiLU；prefill 以 8 行 tile 复用同一 packed block。
pub fn gated_linear_q4_k_silu_f16(ctx: &CudaContext, input: &CudaTensor, gate: &cudarc::driver::safe::CudaSlice<u8>, up: &cudarc::driver::safe::CudaSlice<u8>, out_cols: usize) -> Result<CudaTensor, String> {
    const QK_K: usize = 256;
    const Q4_K_BYTES: usize = 144;
    if !input.cols.is_multiple_of(QK_K) {
        return Err(format!("Q4_K gated GEMV 需要 columns 对齐 256，实际 [{},{}]", input.rows, input.cols));
    }
    let blocks_per_row = input.cols / QK_K;
    let expected = out_cols.checked_mul(blocks_per_row).and_then(|blocks| blocks.checked_mul(Q4_K_BYTES)).ok_or("Q4_K gated GEMV weight 大小溢出")?;
    if gate.len() != expected || up.len() != expected {
        return Err(format!("Q4_K gate={} up={}，期望 {expected}", gate.len(), up.len()));
    }
    let output = ctx.tensor_alloc(input.rows, out_cols)?;
    if input.rows > 1 {
        let func = ctx.function("gated_linear_q4_k_silu_rows8_f16")?;
        unsafe {
            ctx.stream()
                .launch_builder(&func)
                .arg(&input.slice)
                .arg(gate)
                .arg(up)
                .arg(&output.slice)
                .arg(&(input.rows as u32))
                .arg(&(input.cols as u32))
                .arg(&(out_cols as u32))
                .arg(&(blocks_per_row as u32))
                .launch(LaunchConfig { grid_dim: (out_cols as u32, input.rows.div_ceil(8) as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
                .map_err(|error| format!("launch gated_linear_q4_k_silu_rows8_f16 失败: {error:?}"))?;
        }
        return Ok(output);
    }
    let func = ctx.function("gated_linear_q4_k_silu_f16")?;
    let in_cols = input.cols as u32;
    let out_cols_u32 = out_cols as u32;
    let blocks_per_row_u32 = blocks_per_row as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(gate)
            .arg(up)
            .arg(&output.slice)
            .arg(&in_cols)
            .arg(&out_cols_u32)
            .arg(&blocks_per_row_u32)
            .launch(LaunchConfig { grid_dim: (out_cols as u32, input.rows as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (2 * (THREADS / 32) as usize * std::mem::size_of::<f32>()) as u32 })
            .map_err(|error| format!("launch gated_linear_q4_k_silu_f16 失败: {error:?}"))?;
    }
    Ok(output)
}

/// Q4_0 gate/up 直接计算并融合 SiLU;块为 32 元素 / 18 字节,结构对称
/// [`gated_linear_q4_k_silu_f16`],服务 Gemma4 QAT 官方权重。
/// Q4_0 × Q8_1 MMQ(int8 tensor core GEMM):prefill 大 chunk 专用,直接吃
/// packed 权重,无 f16 物化往返。激活复用 MMVQ 的 Q8_1 量化布局。
pub fn linear_q4_0_mmq_s8(ctx: &CudaContext, input: &CudaTensor, weight: &cudarc::driver::safe::CudaSlice<u8>, out_rows: usize) -> Result<CudaTensor, String> {
    if input.rows < 16 {
        return Err(format!("Q4_0 MMQ 面向 ≥16 行,实际 rows={}", input.rows));
    }
    if !input.cols.is_multiple_of(32) {
        return Err(format!("Q4_0 MMQ 需要 columns 对齐 32,实际 {}", input.cols));
    }
    let blocks = input.cols / 32;
    let expected = out_rows.checked_mul(blocks).and_then(|n| n.checked_mul(18)).ok_or("Q4_0 MMQ weight 溢出")?;
    if weight.len() != expected {
        return Err(format!("Q4_0 MMQ weight={}，期望 {expected}", weight.len()));
    }
    // 激活 Q8_1 量化(与 MMVQ 同布局)。
    let mut activation = ctx.buffer_uninit::<u8>(input.rows.checked_mul(blocks).and_then(|n| n.checked_mul(36)).ok_or("Q4_0 MMQ 激活溢出")?)?;
    let quantize = ctx.function("q8_1_quantize_rows_f16")?;
    let cols_u32 = input.cols as u32;
    let rows_u32 = input.rows as u32;
    unsafe {
        ctx.stream().launch_builder(&quantize).arg(&input.slice).arg(&mut activation).arg(&cols_u32).arg(&rows_u32).launch(grid_1d(input.rows * blocks)).map_err(|error| format!("launch q8_1_quantize_rows_f16 失败: {error:?}"))?;
    }
    let output = ctx.tensor_alloc(input.rows, out_rows)?;
    let func = ctx.function("q4_0_q8_1_mmq_s8")?;
    let out_rows_u32 = out_rows as u32;
    let one = 1.0f32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(weight)
            .arg(&activation)
            .arg(&output.slice)
            .arg(&rows_u32)
            .arg(&out_rows_u32)
            .arg(&cols_u32)
            .arg(&one)
            .launch(LaunchConfig { grid_dim: (out_rows.div_ceil(64) as u32, input.rows.div_ceil(16) as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|error| format!("launch q4_0_q8_1_mmq_s8 失败: {error:?}"))?;
    }
    Ok(output)
}

/// Q4_0 × Q8_1 MMVQ(参考 llama.cpp vecdotq/mmvq):激活先整体量化成 Q8_1,
/// kernel 内权重 nibble 一次提取、对 K 行激活各做 dp4a 整数点积;-8 偏移用
/// dp4a(u, 0x01010101) 精确校正。服务 2..8 行前向(MTP verify 的 K+1 行)。
/// 激活量化引入 ~0.4% 相对误差(与 Q8G64 KV 同级)。
pub fn linear_q4_0_mmvq_f16(ctx: &CudaContext, input: &CudaTensor, weight: &cudarc::driver::safe::CudaSlice<u8>, out_rows: usize) -> Result<CudaTensor, String> {
    let k = input.rows;
    if !(2..=8).contains(&k) {
        return Err(format!("Q4_0 MMVQ 面向 2..8 行,实际 rows={k}"));
    }
    if !input.cols.is_multiple_of(32) {
        return Err(format!("Q4_0 MMVQ 需要 columns 对齐 32,实际 {}", input.cols));
    }
    let blocks = input.cols / 32;
    let expected = out_rows.checked_mul(blocks).and_then(|blocks| blocks.checked_mul(18)).ok_or("Q4_0 MMVQ weight 溢出")?;
    if weight.len() != expected {
        return Err(format!("Q4_0 MMVQ weight={}，期望 {expected}", weight.len()));
    }
    // 激活量化:k × blocks 个 36B 块(32 码 + f16 scale + pad)。
    let mut activation = ctx.buffer_uninit::<u8>(k.checked_mul(blocks).and_then(|n| n.checked_mul(36)).ok_or("Q4_0 MMVQ 激活溢出")?)?;
    let quantize = ctx.function("q8_1_quantize_rows_f16")?;
    let cols_u32 = input.cols as u32;
    let rows_u32 = k as u32;
    unsafe {
        ctx.stream().launch_builder(&quantize).arg(&input.slice).arg(&mut activation).arg(&cols_u32).arg(&rows_u32).launch(grid_1d(k * blocks)).map_err(|error| format!("launch q8_1_quantize_rows_f16 失败: {error:?}"))?;
    }
    let output = ctx.tensor_alloc(k, out_rows)?;
    let name = match k {
        2 => "q4_0_q8_1_mmvq_k2_f16",
        3 => "q4_0_q8_1_mmvq_k3_f16",
        4 => "q4_0_q8_1_mmvq_k4_f16",
        5 => "q4_0_q8_1_mmvq_k5_f16",
        6 => "q4_0_q8_1_mmvq_k6_f16",
        7 => "q4_0_q8_1_mmvq_k7_f16",
        _ => "q4_0_q8_1_mmvq_k8_f16",
    };
    let func = ctx.function(name)?;
    let out_rows_u32 = out_rows as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(weight)
            .arg(&activation)
            .arg(&output.slice)
            .arg(&cols_u32)
            .arg(&out_rows_u32)
            .launch(LaunchConfig { grid_dim: (out_rows as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
            .map_err(|error| format!("launch {name} 失败: {error:?}"))?;
    }
    Ok(output)
}

pub fn gated_linear_q4_0_silu_f16(ctx: &CudaContext, input: &CudaTensor, gate: &cudarc::driver::safe::CudaSlice<u8>, up: &cudarc::driver::safe::CudaSlice<u8>, out_cols: usize) -> Result<CudaTensor, String> {
    const QK: usize = 32;
    const Q4_0_BYTES: usize = 18;
    if !input.cols.is_multiple_of(QK) {
        return Err(format!("Q4_0 gated GEMV 需要 columns 对齐 32，实际 [{},{}]", input.rows, input.cols));
    }
    let blocks_per_row = input.cols / QK;
    let expected = out_cols.checked_mul(blocks_per_row).and_then(|blocks| blocks.checked_mul(Q4_0_BYTES)).ok_or("Q4_0 gated GEMV weight 大小溢出")?;
    if gate.len() != expected || up.len() != expected {
        return Err(format!("Q4_0 gate={} up={}，期望 {expected}", gate.len(), up.len()));
    }
    let output = ctx.tensor_alloc(input.rows, out_cols)?;
    if input.rows > 1 {
        let func = ctx.function("gated_linear_q4_0_silu_rows8_f16")?;
        unsafe {
            ctx.stream()
                .launch_builder(&func)
                .arg(&input.slice)
                .arg(gate)
                .arg(up)
                .arg(&output.slice)
                .arg(&(input.rows as u32))
                .arg(&(input.cols as u32))
                .arg(&(out_cols as u32))
                .arg(&(blocks_per_row as u32))
                .launch(LaunchConfig { grid_dim: (out_cols as u32, input.rows.div_ceil(8) as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 })
                .map_err(|error| format!("launch gated_linear_q4_0_silu_rows8_f16 失败: {error:?}"))?;
        }
        return Ok(output);
    }
    let func = ctx.function("gated_linear_q4_0_silu_f16")?;
    let in_cols = input.cols as u32;
    let out_cols_u32 = out_cols as u32;
    let blocks_per_row_u32 = blocks_per_row as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(gate)
            .arg(up)
            .arg(&output.slice)
            .arg(&in_cols)
            .arg(&out_cols_u32)
            .arg(&blocks_per_row_u32)
            .launch(LaunchConfig { grid_dim: (out_cols as u32, input.rows as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (2 * (THREADS / 32) as usize * std::mem::size_of::<f32>()) as u32 })
            .map_err(|error| format!("launch gated_linear_q4_0_silu_f16 失败: {error:?}"))?;
    }
    Ok(output)
}

/// Q5_K 矩阵直接 GEMV，decode 不展开 down 权重。
pub fn linear_q5_k_f16(ctx: &CudaContext, input: &CudaTensor, weight: &cudarc::driver::safe::CudaSlice<u8>, out_cols: usize) -> Result<CudaTensor, String> {
    const QK_K: usize = 256;
    const Q5_K_BYTES: usize = 176;
    if !input.cols.is_multiple_of(QK_K) {
        return Err(format!("Q5_K GEMV 需要 columns 对齐 256，实际 [{},{}]", input.rows, input.cols));
    }
    let blocks_per_row = input.cols / QK_K;
    let threads = (blocks_per_row as u32 * 32).max(32);
    if threads > THREADS {
        return Err(format!("Q5_K GEMV blocks_per_row={blocks_per_row} 超过 {}", THREADS / 32));
    }
    let expected = out_cols.checked_mul(blocks_per_row).and_then(|blocks| blocks.checked_mul(Q5_K_BYTES)).ok_or("Q5_K GEMV weight 大小溢出")?;
    if weight.len() != expected {
        return Err(format!("Q5_K weight={}，期望 {expected}", weight.len()));
    }
    let output = ctx.tensor_alloc(input.rows, out_cols)?;
    let func = ctx.function("linear_q5_k_f16")?;
    let out_cols_u32 = out_cols as u32;
    let blocks_per_row_u32 = blocks_per_row as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(weight)
            .arg(&output.slice)
            .arg(&out_cols_u32)
            .arg(&blocks_per_row_u32)
            .launch(LaunchConfig { grid_dim: (out_cols as u32, input.rows as u32, 1), block_dim: (threads, 1, 1), shared_mem_bytes: (blocks_per_row * std::mem::size_of::<f32>()) as u32 })
            .map_err(|error| format!("launch linear_q5_k_f16 失败: {error:?}"))?;
    }
    Ok(output)
}

/// Q6_K 矩阵直接 GEMV，覆盖少数高精度 down 权重。
pub fn linear_q6_k_f16(ctx: &CudaContext, input: &CudaTensor, weight: &cudarc::driver::safe::CudaSlice<u8>, out_cols: usize) -> Result<CudaTensor, String> {
    const QK_K: usize = 256;
    const Q6_K_BYTES: usize = 210;
    if !input.cols.is_multiple_of(QK_K) {
        return Err(format!("Q6_K GEMV 需要 columns 对齐 256，实际 [{},{}]", input.rows, input.cols));
    }
    let blocks_per_row = input.cols / QK_K;
    let threads = (blocks_per_row as u32 * 32).max(32);
    if threads > THREADS {
        return Err(format!("Q6_K GEMV blocks_per_row={blocks_per_row} 超过 {}", THREADS / 32));
    }
    let expected = out_cols.checked_mul(blocks_per_row).and_then(|blocks| blocks.checked_mul(Q6_K_BYTES)).ok_or("Q6_K GEMV weight 大小溢出")?;
    if weight.len() != expected {
        return Err(format!("Q6_K weight={}，期望 {expected}", weight.len()));
    }
    let output = ctx.tensor_alloc(input.rows, out_cols)?;
    let func = ctx.function("linear_q6_k_f16")?;
    let out_cols_u32 = out_cols as u32;
    let blocks_per_row_u32 = blocks_per_row as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(weight)
            .arg(&output.slice)
            .arg(&out_cols_u32)
            .arg(&blocks_per_row_u32)
            .launch(LaunchConfig { grid_dim: (out_cols as u32, input.rows as u32, 1), block_dim: (threads, 1, 1), shared_mem_bytes: (blocks_per_row * std::mem::size_of::<f32>()) as u32 })
            .map_err(|error| format!("launch linear_q6_k_f16 失败: {error:?}"))?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn linear_qk_accumulate_f32(
    ctx: &CudaContext,
    input: &CudaTensor,
    weight: &cudarc::driver::safe::CudaSlice<u8>,
    output: &cudarc::driver::safe::CudaSlice<f32>,
    out_cols: usize,
    route_weights: &cudarc::driver::safe::CudaSlice<f32>,
    route: usize,
    block_bytes: usize,
    kernel: &str,
) -> Result<(), String> {
    const QK_K: usize = 256;
    if !input.cols.is_multiple_of(QK_K) {
        return Err(format!("{kernel} 需要 columns 对齐 256，实际 [{},{}]", input.rows, input.cols));
    }
    let blocks_per_row = input.cols / QK_K;
    let threads = (blocks_per_row as u32 * 32).max(32);
    if threads > THREADS {
        return Err(format!("{kernel} blocks_per_row={blocks_per_row} 超过 {}", THREADS / 32));
    }
    let expected = out_cols.checked_mul(blocks_per_row).and_then(|blocks| blocks.checked_mul(block_bytes)).ok_or_else(|| format!("{kernel} weight 大小溢出"))?;
    if weight.len() != expected || output.len() != input.rows * out_cols || route >= route_weights.len() {
        return Err(format!("{kernel} shape 不匹配: weight={}/{expected}, output={}/{}, route={route}/{}", weight.len(), output.len(), input.rows * out_cols, route_weights.len(),));
    }
    let func = ctx.function(kernel)?;
    let route = route as u32;
    let out_cols_u32 = out_cols as u32;
    let blocks_per_row_u32 = blocks_per_row as u32;
    unsafe {
        ctx.stream()
            .launch_builder(&func)
            .arg(&input.slice)
            .arg(weight)
            .arg(output)
            .arg(route_weights)
            .arg(&route)
            .arg(&out_cols_u32)
            .arg(&blocks_per_row_u32)
            .launch(LaunchConfig { grid_dim: (out_cols as u32, input.rows as u32, 1), block_dim: (threads, 1, 1), shared_mem_bytes: (blocks_per_row * std::mem::size_of::<f32>()) as u32 })
            .map_err(|error| format!("launch {kernel} 失败: {error:?}"))?;
    }
    Ok(())
}

pub fn linear_q5_k_accumulate_f32(
    ctx: &CudaContext,
    input: &CudaTensor,
    weight: &cudarc::driver::safe::CudaSlice<u8>,
    output: &cudarc::driver::safe::CudaSlice<f32>,
    out_cols: usize,
    route_weights: &cudarc::driver::safe::CudaSlice<f32>,
    route: usize,
) -> Result<(), String> {
    linear_qk_accumulate_f32(ctx, input, weight, output, out_cols, route_weights, route, 176, "linear_q5_k_accumulate_f32")
}

pub fn linear_q6_k_accumulate_f32(
    ctx: &CudaContext,
    input: &CudaTensor,
    weight: &cudarc::driver::safe::CudaSlice<u8>,
    output: &cudarc::driver::safe::CudaSlice<f32>,
    out_cols: usize,
    route_weights: &cudarc::driver::safe::CudaSlice<f32>,
    route: usize,
) -> Result<(), String> {
    linear_qk_accumulate_f32(ctx, input, weight, output, out_cols, route_weights, route, 210, "linear_q6_k_accumulate_f32")
}

#[cfg(all(test, target_os = "linux", feature = "with-cuda"))]
mod tests {
    use half::f16;

    use super::*;

    fn packed_blocks(tensor_type: u32, rows: usize) -> Vec<u8> {
        let block_bytes = match tensor_type {
            2 => 18,
            8 => 34,
            12 => 144,
            13 => 176,
            14 => 210,
            _ => unreachable!(),
        };
        let blocks_per_row = if tensor_type == 8 || tensor_type == 2 { 8 } else { 1 };
        let mut bytes = vec![0u8; rows * blocks_per_row * block_bytes];
        for block_index in 0..rows * blocks_per_row {
            let row = block_index / blocks_per_row;
            let block = &mut bytes[block_index * block_bytes..(block_index + 1) * block_bytes];
            if tensor_type == 2 {
                block[..2].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
                for (index, value) in block[2..].iter_mut().enumerate() {
                    *value = (index as i8).wrapping_mul(11).wrapping_add((row as i8).wrapping_mul(5)) as u8;
                }
            } else if tensor_type == 8 {
                block[..2].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
                for (index, value) in block[2..].iter_mut().enumerate() {
                    *value = (index as i8).wrapping_mul(7).wrapping_add((row as i8).wrapping_mul(3)) as u8;
                }
            } else if tensor_type == 14 {
                for (index, value) in block[..192].iter_mut().enumerate() {
                    *value = (index * 37 + row * 11) as u8;
                }
                for (index, value) in block[192..208].iter_mut().enumerate() {
                    *value = ((index as i8 % 9) - 4) as u8;
                }
                block[208..210].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
            } else {
                block[..2].copy_from_slice(&f16::from_f32(0.0625).to_le_bytes());
                block[2..4].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
                block[4..16].fill(0x11);
                for (index, value) in block[16..].iter_mut().enumerate() {
                    *value = (index * 29 + row * 7) as u8;
                }
            }
        }
        bytes
    }

    #[test]
    fn gguf_kq_matmul_matches_cpu_decode() {
        let ctx = CudaContext::new_default().expect("初始化 CUDA");
        let input_rows = 9;
        let columns = 256;
        let output_rows = 2;
        let input = (0..input_rows * columns).map(|index| f16::from_f32(((index * 17 % 101) as f32 - 50.0) / 64.0)).collect::<Vec<_>>();
        let input_device = ctx.stream().clone_htod(&input).expect("上传 input");
        let input_tensor = CudaTensor::new(input_device, input_rows, columns);
        for tensor_type in [2, 8, 12, 13, 14] {
            let packed = packed_blocks(tensor_type, output_rows);
            let decoded = crate::weight::codec::ggml::dequantize(tensor_type, &packed, output_rows * columns).expect("CPU decode");
            let weight = ctx.stream().clone_htod(&packed).expect("上传 packed weight");
            let actual = gguf_kq_matmul_f16(&ctx, &input_tensor, &weight, output_rows, tensor_type).and_then(|tensor| ctx.tensor_to_f32(&tensor)).expect("CUDA matmul");
            for row in 0..input_rows {
                for output_row in 0..output_rows {
                    let expected = (0..columns).map(|column| input[row * columns + column].to_f32() * decoded[output_row * columns + column]).sum::<f32>();
                    let value = actual[row * output_rows + output_row];
                    let tolerance = 0.05 + expected.abs() * 0.002;
                    assert!((value - expected).abs() <= tolerance, "type={tensor_type} row={row} output={output_row}: CUDA={value} CPU={expected} tolerance={tolerance}");
                }
            }
            let decode_input = ctx.stream().clone_htod(&input[..columns]).expect("上传 decode input");
            let decode_tensor = CudaTensor::new(decode_input, 1, columns);
            let decode_actual = gguf_kq_matmul_f16(&ctx, &decode_tensor, &weight, output_rows, tensor_type).and_then(|tensor| ctx.tensor_to_f32(&tensor)).expect("CUDA GEMV");
            for output_row in 0..output_rows {
                let expected = (0..columns).map(|column| input[column].to_f32() * decoded[output_row * columns + column]).sum::<f32>();
                let tolerance = 0.05 + expected.abs() * 0.002;
                assert!((decode_actual[output_row] - expected).abs() <= tolerance, "GEMV type={tensor_type} output={output_row}: CUDA={} CPU={expected}", decode_actual[output_row]);
            }
        }
    }

    #[test]
    fn gated_q4_k_prefill_and_decode_match_cpu_decode() {
        let ctx = CudaContext::new_default().expect("初始化 CUDA");
        let input_rows = 9;
        let columns = 512;
        let output_rows = 2;
        let input = (0..input_rows * columns).map(|index| f16::from_f32(((index * 13 % 89) as f32 - 44.0) / 96.0)).collect::<Vec<_>>();
        let packed = packed_blocks(12, output_rows * columns / 256);
        let decoded = crate::weight::codec::ggml::dequantize(12, &packed, output_rows * columns).expect("CPU decode");
        let input_device = ctx.stream().clone_htod(&input).expect("上传 input");
        let gate = ctx.stream().clone_htod(&packed).expect("上传 gate");
        let up = ctx.stream().clone_htod(&packed).expect("上传 up");
        let input_tensor = CudaTensor::new(input_device, input_rows, columns);
        let actual = gated_linear_q4_k_silu_f16(&ctx, &input_tensor, &gate, &up, output_rows).and_then(|tensor| ctx.tensor_to_f32(&tensor)).expect("CUDA gated Q4_K prefill");
        for token in 0..input_rows {
            for row in 0..output_rows {
                let dot = (0..columns).map(|column| input[token * columns + column].to_f32() * decoded[row * columns + column]).sum::<f32>();
                let expected = (dot / (1.0 + (-dot).exp())) * dot;
                let value = actual[token * output_rows + row];
                let tolerance = 0.1 + expected.abs() * 0.003;
                assert!((value - expected).abs() <= tolerance, "token={token} row={row}: CUDA={value} CPU={expected} tolerance={tolerance}");
            }
        }

        let columns = 5120;
        let input = (0..columns).map(|index| f16::from_f32(((index * 13 % 89) as f32 - 44.0) / 96.0)).collect::<Vec<_>>();
        let packed = packed_blocks(12, output_rows * columns / 256);
        let decoded = crate::weight::codec::ggml::dequantize(12, &packed, output_rows * columns).expect("CPU decode");
        let input_device = ctx.stream().clone_htod(&input).expect("上传 decode input");
        let gate = ctx.stream().clone_htod(&packed).expect("上传 decode gate");
        let up = ctx.stream().clone_htod(&packed).expect("上传 decode up");
        let input_tensor = CudaTensor::new(input_device, 1, columns);
        let actual = gated_linear_q4_k_silu_f16(&ctx, &input_tensor, &gate, &up, output_rows).and_then(|tensor| ctx.tensor_to_f32(&tensor)).expect("CUDA gated Q4_K decode");
        for row in 0..output_rows {
            let dot = (0..columns).map(|column| input[column].to_f32() * decoded[row * columns + column]).sum::<f32>();
            let expected = (dot / (1.0 + (-dot).exp())) * dot;
            let tolerance = 0.1 + expected.abs() * 0.003;
            assert!((actual[row] - expected).abs() <= tolerance, "decode row={row}: CUDA={} CPU={expected} tolerance={tolerance}", actual[row]);
        }
    }

    fn q4_0_blocks_probe(total_blocks: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; total_blocks * 18];
        for (index, chunk) in bytes.chunks_exact_mut(18).enumerate() {
            chunk[..2].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
            for (j, value) in chunk[2..].iter_mut().enumerate() {
                *value = (j as i8).wrapping_mul(11).wrapping_add((index as i8).wrapping_mul(5)) as u8;
            }
        }
        bytes
    }

    #[test]
    fn mma_mapping_probe_debug() {
        // 穷举 a_map×b_map 组合,对拍 CPU 点积定位正确的 m16n8k32 片段布局。
        let ctx = CudaContext::new_default().expect("CUDA");
        // A/B 用受限随机值,D 碰撞概率低;D 值唯一匹配可反解每个寄存器的 (r,c)。
        let mut seed = 12345u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) % 7) as i32 - 3
        };
        let a: Vec<i8> = (0..16 * 32).map(|_| rand() as i8).collect();
        let b: Vec<i8> = (0..8 * 32).map(|_| rand() as i8).collect();
        let mut cpu_d = vec![0i32; 16 * 8];
        for r in 0..16 {
            for c in 0..8 {
                cpu_d[r * 8 + c] = (0..32).map(|k| a[r * 32 + k] as i32 * b[c * 32 + k] as i32).sum();
            }
        }
        let a_gpu = ctx.stream().clone_htod(&a).expect("up a");
        let b_gpu = ctx.stream().clone_htod(&b).expect("up b");
        for a_map in 0..2u32 {
            for b_map in 0..2u32 {
                let mut d_gpu = ctx.buffer_uninit::<i32>(128).unwrap();
                let func = ctx.function("mma_mapping_probe").unwrap();
                unsafe {
                    ctx.stream().launch_builder(&func).arg(&a_gpu).arg(&b_gpu).arg(&mut d_gpu).arg(&a_map).arg(&b_map).launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }).unwrap();
                }
                let raw = ctx.stream().clone_dtoh::<i32, _>(&d_gpu).unwrap();
                // 对每个 (lane,reg) 找 CPU D 中唯一匹配位置,反解行/列分配。
                let mut row_of = [0usize; 4];
                let mut col_of = [0usize; 4];
                let mut ok = true;
                for reg in 0..4 {
                    let mut rows = std::collections::HashSet::new();
                    let mut cols = std::collections::HashSet::new();
                    for lane in 0..32 {
                        let value = raw[lane * 4 + reg];
                        let mut matches = vec![];
                        for r in 0..16 {
                            for c in 0..8 {
                                if cpu_d[r * 8 + c] == value {
                                    matches.push((r, c));
                                }
                            }
                        }
                        if matches.len() != 1 {
                            ok = false;
                            break;
                        }
                        rows.insert(matches[0].0);
                        cols.insert(matches[0].1);
                    }
                    if !ok {
                        break;
                    }
                    row_of[reg] = 0;
                    // 验证规律:reg 的 (r,c) 应由 lane 的 group/tig 决定
                    let consistent = (0..32).all(|lane| {
                        let value = raw[lane * 4 + reg];
                        let m = (0..16 * 8).filter(|&i| cpu_d[i] == value).collect::<Vec<_>>();
                        m.len() == 1
                    });
                    if !consistent {
                        ok = false;
                        break;
                    }
                    let _ = (&rows, &cols);
                }
                println!("[probe] a_map={a_map} b_map={b_map} 结构化匹配={}", if ok { "可解" } else { "碰撞" });
                // 直接验证常用 D 假设:reg{0,1}=行 group{,+8} 列 2tig+{0,1}
                let d_ok = (0..32).all(|lane| {
                    let g = lane >> 2;
                    let t = lane & 3;
                    raw[lane * 4] == cpu_d[g * 8 + 2 * t] && raw[lane * 4 + 1] == cpu_d[g * 8 + 2 * t + 1] && raw[lane * 4 + 2] == cpu_d[(g + 8) * 8 + 2 * t] && raw[lane * 4 + 3] == cpu_d[(g + 8) * 8 + 2 * t + 1]
                });
                if d_ok {
                    println!("[probe] *** a_map={a_map} b_map={b_map} D 映射 {0}=({{group}},2tig),{1}=({{group}},2tig+1),{2}=({{group}}+8,2tig),{3}=({{group}}+8,2tig+1) 全对 ***", 0, 1, 2, 3);
                }
            }
        }
    }

    #[test]
    fn q4_0_mmq_tiny_debug() {
        // 单块最小形状:打印 CUDA vs CPU 矩阵定位错位模式(诊断用)。
        let ctx = CudaContext::new_default().expect("CUDA");
        let columns = std::env::var("ZLLM_MMQ_COLS").ok().and_then(|v| v.parse().ok()).unwrap_or(32usize);
        let _ = std::env::var_os("ZLLM_MMQ_COLS64");
        let output_rows = std::env::var("ZLLM_MMQ_OUT").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize);
        let row_count = std::env::var("ZLLM_MMQ_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
        let input: Vec<f16> = (0..row_count * columns).map(|i| f16::from_f32((i % 17) as f32 * 0.25 - 2.0)).collect();
        let blocks_per_row = columns / 32;
        let mut packed = vec![0u8; output_rows * blocks_per_row * 18];
        for (index, block) in packed.chunks_exact_mut(18).enumerate() {
            let kb = index % blocks_per_row;
            block[..2].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
            for j in 0..16 {
                block[2 + j] = if std::env::var_os("ZLLM_MMQ_ZERO0").is_some() && kb == 0 { 8u8 } else { ((index * 3 + j) % 16) as u8 };
            }
        }
        let decoded = crate::weight::codec::ggml::dequantize(2, &packed, output_rows * columns).expect("decode");
        let input_device = ctx.stream().clone_htod(&input).expect("up");
        let weight = ctx.stream().clone_htod(&packed).expect("up");
        let input_tensor = CudaTensor::new(input_device, row_count, columns);
        let actual = gguf_kq_matmul_f16(&ctx, &input_tensor, &weight, output_rows, 2).and_then(|t| ctx.tensor_to_f32(&t)).expect("mmq");
        for m in 0..row_count.min(4) {
            let mut cpu_row = Vec::new();
            let mut gpu_row = Vec::new();
            for n in 0..8 {
                cpu_row.push((0..columns).map(|c| input[m * columns + c].to_f32() * decoded[n * columns + c]).sum::<f32>());
                gpu_row.push(actual[m * output_rows + n]);
            }
            println!("[mmq-debug] m={m} cpu={cpu_row:.3?}");
            println!("[mmq-debug] m={m} gpu={gpu_row:.3?}");
            for n in 0..8 {
                let tolerance = 0.05 + cpu_row[n].abs() * 0.02;
                assert!((gpu_row[n] - cpu_row[n]).abs() <= tolerance, "m={m} n={n}: gpu={} cpu={}", gpu_row[n], cpu_row[n]);
            }
        }
    }

    #[test]
    fn q4_0_mmq_matches_cpu_decode() {
        // prefill 形状:rows=32 走 int8 mma MMQ(Q8_1 激活量化 ~1% 误差)。
        let ctx = CudaContext::new_default().expect("初始化 CUDA");
        let columns = 3840;
        let output_rows = 64;
        let input = (0..32 * columns).map(|index| f16::from_f32(((index * 13 % 89) as f32 - 44.0) / 96.0)).collect::<Vec<_>>();
        let packed = q4_0_blocks_probe(output_rows * columns / 32);
        let decoded = crate::weight::codec::ggml::dequantize(2, &packed, output_rows * columns).expect("CPU decode");
        let input_device = ctx.stream().clone_htod(&input).expect("上传 input");
        let weight = ctx.stream().clone_htod(&packed).expect("上传 packed");
        let input_tensor = CudaTensor::new(input_device, 32, columns);
        let actual = gguf_kq_matmul_f16(&ctx, &input_tensor, &weight, output_rows, 2).and_then(|tensor| ctx.tensor_to_f32(&tensor)).expect("CUDA mmq");
        for row in 0..32 {
            for out in 0..output_rows {
                let expected = (0..columns).map(|column| input[row * columns + column].to_f32() * decoded[out * columns + column]).sum::<f32>();
                let tolerance = 0.1 + expected.abs() * 0.02;
                assert!((actual[row * output_rows + out] - expected).abs() <= tolerance, "row={row} out={out}: CUDA={} CPU={expected} tol={tolerance}", actual[row * output_rows + out]);
            }
        }
    }

    #[test]
    fn q4_0_small_rows_matmul_matches_cpu_decode() {
        // MTP verify 形状:rows=3 走 Q8_1 MMVQ(dp4a,激活量化 ~1% 误差)。
        let ctx = CudaContext::new_default().expect("初始化 CUDA");
        let columns = 3840;
        let output_rows = 64;
        let input = (0..3 * columns).map(|index| f16::from_f32(((index * 13 % 89) as f32 - 44.0) / 96.0)).collect::<Vec<_>>();
        let packed = q4_0_blocks_probe(output_rows * columns / 32);
        let decoded = crate::weight::codec::ggml::dequantize(2, &packed, output_rows * columns).expect("CPU decode");
        let input_device = ctx.stream().clone_htod(&input).expect("上传 input");
        let weight = ctx.stream().clone_htod(&packed).expect("上传 packed");
        let input_tensor = CudaTensor::new(input_device, 3, columns);
        let actual = gguf_kq_matmul_f16(&ctx, &input_tensor, &weight, output_rows, 2).and_then(|tensor| ctx.tensor_to_f32(&tensor)).expect("CUDA rows3 matmul");
        for row in 0..3 {
            for out in 0..output_rows {
                let expected = (0..columns).map(|column| input[row * columns + column].to_f32() * decoded[out * columns + column]).sum::<f32>();
                // rows=3 走 MMVQ:Q8_1 激活量化引入 ~1% 级相对误差。
                let tolerance = 0.1 + expected.abs() * 0.02;
                assert!((actual[row * output_rows + out] - expected).abs() <= tolerance, "row={row} out={out}: CUDA={} CPU={expected}", actual[row * output_rows + out]);
            }
        }
    }

    #[test]
    fn gated_q4_0_prefill_and_decode_match_cpu_decode() {
        let ctx = CudaContext::new_default().expect("初始化 CUDA");
        // Q4_0 按"总块数"构造(packed_blocks 的 rows 语义是 256 列行,Q4_0 行宽可变)。
        let q4_0_blocks = |total_blocks: usize| -> Vec<u8> {
            let mut bytes = vec![0u8; total_blocks * 18];
            for (index, chunk) in bytes.chunks_exact_mut(18).enumerate() {
                chunk[..2].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
                for (j, value) in chunk[2..].iter_mut().enumerate() {
                    *value = (j as i8).wrapping_mul(11).wrapping_add((index as i8).wrapping_mul(5)) as u8;
                }
            }
            bytes
        };
        let input_rows = 9;
        let columns = 384;
        let output_rows = 2;
        let input = (0..input_rows * columns).map(|index| f16::from_f32(((index * 13 % 89) as f32 - 44.0) / 96.0)).collect::<Vec<_>>();
        let packed = q4_0_blocks(output_rows * columns / 32);
        let decoded = crate::weight::codec::ggml::dequantize(2, &packed, output_rows * columns).expect("CPU decode");
        let input_device = ctx.stream().clone_htod(&input).expect("上传 input");
        let gate = ctx.stream().clone_htod(&packed).expect("上传 gate");
        let up = ctx.stream().clone_htod(&packed).expect("上传 up");
        let input_tensor = CudaTensor::new(input_device, input_rows, columns);
        let actual = gated_linear_q4_0_silu_f16(&ctx, &input_tensor, &gate, &up, output_rows).and_then(|tensor| ctx.tensor_to_f32(&tensor)).expect("CUDA gated Q4_0 prefill");
        for token in 0..input_rows {
            for row in 0..output_rows {
                let dot = (0..columns).map(|column| input[token * columns + column].to_f32() * decoded[row * columns + column]).sum::<f32>();
                let expected = (dot / (1.0 + (-dot).exp())) * dot;
                let value = actual[token * output_rows + row];
                let tolerance = 0.1 + expected.abs() * 0.003;
                assert!((value - expected).abs() <= tolerance, "token={token} row={row}: CUDA={value} CPU={expected} tolerance={tolerance}");
            }
        }

        // 真实 Gemma4 12B hidden 宽度:验证 120 个 32 元素块的 decode 路径。
        let columns = 3840;
        let input = (0..columns).map(|index| f16::from_f32(((index * 13 % 89) as f32 - 44.0) / 96.0)).collect::<Vec<_>>();
        let packed = q4_0_blocks(output_rows * columns / 32);
        let decoded = crate::weight::codec::ggml::dequantize(2, &packed, output_rows * columns).expect("CPU decode");
        let input_device = ctx.stream().clone_htod(&input).expect("上传 decode input");
        let gate = ctx.stream().clone_htod(&packed).expect("上传 decode gate");
        let up = ctx.stream().clone_htod(&packed).expect("上传 decode up");
        let input_tensor = CudaTensor::new(input_device, 1, columns);
        let actual = gated_linear_q4_0_silu_f16(&ctx, &input_tensor, &gate, &up, output_rows).and_then(|tensor| ctx.tensor_to_f32(&tensor)).expect("CUDA gated Q4_0 decode");
        for row in 0..output_rows {
            let dot = (0..columns).map(|column| input[column].to_f32() * decoded[row * columns + column]).sum::<f32>();
            let expected = (dot / (1.0 + (-dot).exp())) * dot;
            let tolerance = 0.1 + expected.abs() * 0.003;
            assert!((actual[row] - expected).abs() <= tolerance, "decode row={row}: CUDA={} CPU={expected} tolerance={tolerance}", actual[row]);
        }
    }
}
