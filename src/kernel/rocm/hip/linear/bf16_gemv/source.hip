typedef unsigned short __attribute__((ext_vector_type(2))) zllm_bf16x2;

extern "C" __global__ void bf16_gemv_f32(
    const void* input,
    const unsigned short* weight,
    float* output,
    unsigned int columns,
    unsigned int rows,
    unsigned int input_is_bf16)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows) return;
    const unsigned int token = blockIdx.y;
    const float* input_f32 = (const float*)input + (unsigned long long)token * columns;
    const unsigned short* input_bf16 = (const unsigned short*)input + (unsigned long long)token * columns;
    const unsigned short* row_weight = weight + (unsigned long long)row * columns;
    float sum = 0.0F;
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const unsigned long long packed = *reinterpret_cast<const unsigned long long*>(row_weight + column);
        if (input_is_bf16) {
            const unsigned long long x = *reinterpret_cast<const unsigned long long*>(input_bf16 + column);
            sum = fmaf(zllm_bf16_to_f32((unsigned short)x), zllm_bf16_to_f32((unsigned short)packed), sum);
            sum = fmaf(zllm_bf16_to_f32((unsigned short)(x >> 16)), zllm_bf16_to_f32((unsigned short)(packed >> 16)), sum);
            sum = fmaf(zllm_bf16_to_f32((unsigned short)(x >> 32)), zllm_bf16_to_f32((unsigned short)(packed >> 32)), sum);
            sum = fmaf(zllm_bf16_to_f32((unsigned short)(x >> 48)), zllm_bf16_to_f32((unsigned short)(packed >> 48)), sum);
        } else {
            const float4 x = *reinterpret_cast<const float4*>(input_f32 + column);
            sum = fmaf(x.x, zllm_bf16_to_f32((unsigned short)packed), sum);
            sum = fmaf(x.y, zllm_bf16_to_f32((unsigned short)(packed >> 16)), sum);
            sum = fmaf(x.z, zllm_bf16_to_f32((unsigned short)(packed >> 32)), sum);
            sum = fmaf(x.w, zllm_bf16_to_f32((unsigned short)(packed >> 48)), sum);
        }
    }
    for (; column < columns; column += 32) {
        const float x = input_is_bf16 ? zllm_bf16_to_f32(input_bf16[column]) : input_f32[column];
        sum = fmaf(x, zllm_bf16_to_f32(row_weight[column]), sum);
    }
    for (unsigned int delta = 16; delta; delta >>= 1) sum += __shfl_down(sum, delta, 32);
    if (lane == 0) output[(unsigned long long)token * rows + row] = sum;
}

// DSpark 等允许 BF16 activation 近似的单行 head：每次权重读取保持不变，
// F32 input 在寄存器中 RNE 到 BF16 后用 RDNA3 packed dot2 累加。
extern "C" __global__ void bf16_gemv_bfdot_f32(
    const float* input,
    const unsigned short* weight,
    float* output,
    unsigned int columns,
    unsigned int rows,
    unsigned int input_is_bf16)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows || input_is_bf16) return;
    const unsigned short* row_weight = weight + (unsigned long long)row * columns;
    float sum = 0.0f;
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const unsigned long long packed = *reinterpret_cast<const unsigned long long*>(row_weight + column);
        const float4 x = *reinterpret_cast<const float4*>(input + column);
        const zllm_bf16x2 x01 = { zllm_f32_to_bf16(x.x), zllm_f32_to_bf16(x.y) };
        const zllm_bf16x2 x23 = { zllm_f32_to_bf16(x.z), zllm_f32_to_bf16(x.w) };
        const zllm_bf16x2 w01 = { (unsigned short)packed, (unsigned short)(packed >> 16) };
        const zllm_bf16x2 w23 = { (unsigned short)(packed >> 32), (unsigned short)(packed >> 48) };
        sum = __builtin_amdgcn_fdot2_f32_bf16(x01, w01, sum, false);
        sum = __builtin_amdgcn_fdot2_f32_bf16(x23, w23, sum, false);
    }
    for (; column < columns; column += 32) {
        sum = fmaf(input[column], zllm_bf16_to_f32(row_weight[column]), sum);
    }
    for (unsigned int delta = 16; delta; delta >>= 1) sum += __shfl_down(sum, delta, 32);
    if (lane == 0) output[row] = sum;
}

extern "C" __global__ void f32_gemv_f32(
    const float* input,
    const float* weight,
    float* output,
    unsigned int columns,
    unsigned int rows)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows) return;
    const float* row_weight = weight + (unsigned long long)row * columns;
    float sum = 0.0F;
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const float4 x = *reinterpret_cast<const float4*>(input + column);
        const float4 w = *reinterpret_cast<const float4*>(row_weight + column);
        sum = fmaf(x.x, w.x, sum);
        sum = fmaf(x.y, w.y, sum);
        sum = fmaf(x.z, w.z, sum);
        sum = fmaf(x.w, w.w, sum);
    }
    for (; column < columns; column += 32) sum = fmaf(input[column], row_weight[column], sum);
    for (unsigned int delta = 16; delta; delta >>= 1) sum += __shfl_down(sum, delta, 32);
    if (lane == 0) output[row] = sum;
}

// 小批量 F32 GEMV：mHC 等窄输出投影共享一次权重读取，避免 rows>1
// 退化为逐 token 单独 GEMV。每个 token 的 FMA 与 wave32 归约顺序保持不变。
extern "C" __global__ void f32_gemv_small_n_f32(
    const float* input,
    const float* weight,
    float* output,
    unsigned int columns,
    unsigned int rows,
    unsigned int input_rows)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows || input_rows == 0 || input_rows > 8) return;
    const float* row_weight = weight + (unsigned long long)row * columns;
    float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const float4 w = *reinterpret_cast<const float4*>(row_weight + column);
        for (unsigned int token = 0; token < input_rows; ++token) {
            const float4 x = *reinterpret_cast<const float4*>(input + (unsigned long long)token * columns + column);
            sums[token] = fmaf(x.x, w.x, sums[token]);
            sums[token] = fmaf(x.y, w.y, sums[token]);
            sums[token] = fmaf(x.z, w.z, sums[token]);
            sums[token] = fmaf(x.w, w.w, sums[token]);
        }
    }
    for (; column < columns; column += 32) {
        const float w = row_weight[column];
        for (unsigned int token = 0; token < input_rows; ++token) {
            sums[token] = fmaf(input[(unsigned long long)token * columns + column], w, sums[token]);
        }
    }
    for (unsigned int delta = 16; delta; delta >>= 1) {
        for (unsigned int token = 0; token < input_rows; ++token) sums[token] += __shfl_down(sums[token], delta, 32);
    }
    if (lane == 0) {
        for (unsigned int token = 0; token < input_rows; ++token) output[(unsigned long long)token * rows + row] = sums[token];
    }
}

// 小输出行、长列 F32 GEMV 的 split-K 路径:mHC function 投影等 [~24, 16384]
// 形状在单行 grid 下只有 rows 个 wave,96 CU 基本空转,纯延迟主导。把列维
// 切成 splits 段后每 wave 处理 (row, segment),partial 落盘再按段序合并。
// 段内 FMA 与归约顺序与 f32_gemv_f32 相同;段间合并顺序为 split 升序。
extern "C" __global__ void f32_gemv_splitk_f32(
    const float* input,
    const float* weight,
    float* partial,
    unsigned int columns,
    unsigned int rows,
    unsigned int splits,
    unsigned int segment)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int index = blockIdx.x * 8 + wave;
    const unsigned int row = index / splits;
    const unsigned int split = index - row * splits;
    if (row >= rows) return;
    const unsigned int begin = split * segment;
    const unsigned int end = begin + segment < columns ? begin + segment : columns;
    const float* row_weight = weight + (unsigned long long)row * columns;
    float sum = 0.0F;
    unsigned int column = begin + lane * 4U;
    for (; column + 3U < end; column += 128U) {
        const float4 x = *reinterpret_cast<const float4*>(input + column);
        const float4 w = *reinterpret_cast<const float4*>(row_weight + column);
        sum = fmaf(x.x, w.x, sum);
        sum = fmaf(x.y, w.y, sum);
        sum = fmaf(x.z, w.z, sum);
        sum = fmaf(x.w, w.w, sum);
    }
    for (; column < end; column += 32U) sum = fmaf(input[column], row_weight[column], sum);
    for (unsigned int delta = 16; delta; delta >>= 1) sum += __shfl_down(sum, delta, 32);
    if (lane == 0) partial[(unsigned long long)split * rows + row] = sum;
}

// split-K 段和合并:每 wave 一行,lane 按 split 递增累加后 wave 归约。
extern "C" __global__ void f32_gemv_splitk_reduce_f32(
    const float* partial,
    float* output,
    unsigned int rows,
    unsigned int splits)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows) return;
    float sum = 0.0F;
    for (unsigned int split = lane; split < splits; split += 32U) sum += partial[(unsigned long long)split * rows + row];
    for (unsigned int delta = 16; delta; delta >>= 1) sum += __shfl_down(sum, delta, 32);
    if (lane == 0) output[row] = sum;
}

// Decode 中两路投影独立读取同一个 hidden。把两组行拼成一个 grid 后只需
// 一次 launch，单行内部的 FMA 与归约顺序仍与 f32_gemv_f32 完全相同。
extern "C" __global__ void f32_dual_gemv_f32(
    const float* input,
    const float* first_weight,
    const float* second_weight,
    float* output,
    unsigned int columns,
    unsigned int first_rows,
    unsigned int second_rows)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= first_rows + second_rows) return;
    const bool first = row < first_rows;
    const unsigned int matrix_row = first ? row : row - first_rows;
    const float* weight = first ? first_weight : second_weight;
    const float* row_weight = weight + (unsigned long long)matrix_row * columns;
    float sum = 0.0F;
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const float4 x = *reinterpret_cast<const float4*>(input + column);
        const float4 w = *reinterpret_cast<const float4*>(row_weight + column);
        sum = fmaf(x.x, w.x, sum);
        sum = fmaf(x.y, w.y, sum);
        sum = fmaf(x.z, w.z, sum);
        sum = fmaf(x.w, w.w, sum);
    }
    for (; column < columns; column += 32) sum = fmaf(input[column], row_weight[column], sum);
    for (unsigned int delta = 16; delta; delta >>= 1) sum += __shfl_down(sum, delta, 32);
    if (lane == 0) output[row] = sum;
}

// 单 token、同输入的两路 BF16 权重 GEMV 合并成一次提交;单行 FMA 与归约
// 顺序仍与 bf16_gemv_f32 完全相同,权重解码逐位无损。
extern "C" __global__ void bf16_dual_gemv_f32(
    const float* input,
    const unsigned short* first_weight,
    const unsigned short* second_weight,
    float* output,
    unsigned int columns,
    unsigned int first_rows,
    unsigned int second_rows)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= first_rows + second_rows) return;
    const bool first = row < first_rows;
    const unsigned int matrix_row = first ? row : row - first_rows;
    const unsigned short* weight = first ? first_weight : second_weight;
    const unsigned short* row_weight = weight + (unsigned long long)matrix_row * columns;
    float sum = 0.0F;
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const unsigned long long packed = *reinterpret_cast<const unsigned long long*>(row_weight + column);
        const float4 x = *reinterpret_cast<const float4*>(input + column);
        sum = fmaf(x.x, zllm_bf16_to_f32((unsigned short)packed), sum);
        sum = fmaf(x.y, zllm_bf16_to_f32((unsigned short)(packed >> 16)), sum);
        sum = fmaf(x.z, zllm_bf16_to_f32((unsigned short)(packed >> 32)), sum);
        sum = fmaf(x.w, zllm_bf16_to_f32((unsigned short)(packed >> 48)), sum);
    }
    for (; column < columns; column += 32) sum = fmaf(input[column], zllm_bf16_to_f32(row_weight[column]), sum);
    for (unsigned int delta = 16; delta; delta >>= 1) sum += __shfl_down(sum, delta, 32);
    if (lane == 0) output[row] = sum;
}

// MTP verify 固定两行。单独展开后只保留两个累加器，并去掉动态 token 循环；
// 两行仍共享一次大权重读取，各自的 FMA/归约顺序与单行 kernel 完全相同。
extern "C" __global__ void bf16_gemv_f32_rows2(
    const void* input,
    const unsigned short* weight,
    float* output,
    unsigned int columns,
    unsigned int rows,
    unsigned int input_rows,
    unsigned int input_is_bf16)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows || input_rows != 2) return;
    const float* input_f32 = (const float*)input;
    const unsigned short* input_bf16 = (const unsigned short*)input;
    const unsigned short* row_weight = weight + (unsigned long long)row * columns;
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const unsigned long long packed = *reinterpret_cast<const unsigned long long*>(row_weight + column);
        const float w0 = zllm_bf16_to_f32((unsigned short)packed);
        const float w1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
        const float w2 = zllm_bf16_to_f32((unsigned short)(packed >> 32));
        const float w3 = zllm_bf16_to_f32((unsigned short)(packed >> 48));
        if (input_is_bf16) {
            const unsigned long long x0 = *reinterpret_cast<const unsigned long long*>(input_bf16 + column);
            const unsigned long long x1 = *reinterpret_cast<const unsigned long long*>(input_bf16 + (unsigned long long)columns + column);
            sum0 = fmaf(zllm_bf16_to_f32((unsigned short)x0), w0, sum0);
            sum0 = fmaf(zllm_bf16_to_f32((unsigned short)(x0 >> 16)), w1, sum0);
            sum0 = fmaf(zllm_bf16_to_f32((unsigned short)(x0 >> 32)), w2, sum0);
            sum0 = fmaf(zllm_bf16_to_f32((unsigned short)(x0 >> 48)), w3, sum0);
            sum1 = fmaf(zllm_bf16_to_f32((unsigned short)x1), w0, sum1);
            sum1 = fmaf(zllm_bf16_to_f32((unsigned short)(x1 >> 16)), w1, sum1);
            sum1 = fmaf(zllm_bf16_to_f32((unsigned short)(x1 >> 32)), w2, sum1);
            sum1 = fmaf(zllm_bf16_to_f32((unsigned short)(x1 >> 48)), w3, sum1);
        } else {
            const float4 x0 = *reinterpret_cast<const float4*>(input_f32 + column);
            const float4 x1 = *reinterpret_cast<const float4*>(input_f32 + (unsigned long long)columns + column);
            sum0 = fmaf(x0.x, w0, sum0);
            sum0 = fmaf(x0.y, w1, sum0);
            sum0 = fmaf(x0.z, w2, sum0);
            sum0 = fmaf(x0.w, w3, sum0);
            sum1 = fmaf(x1.x, w0, sum1);
            sum1 = fmaf(x1.y, w1, sum1);
            sum1 = fmaf(x1.z, w2, sum1);
            sum1 = fmaf(x1.w, w3, sum1);
        }
    }
    for (; column < columns; column += 32) {
        const float w = zllm_bf16_to_f32(row_weight[column]);
        const float x0 = input_is_bf16 ? zllm_bf16_to_f32(input_bf16[column]) : input_f32[column];
        const float x1 = input_is_bf16 ? zllm_bf16_to_f32(input_bf16[(unsigned long long)columns + column]) : input_f32[(unsigned long long)columns + column];
        sum0 = fmaf(x0, w, sum0);
        sum1 = fmaf(x1, w, sum1);
    }
    for (unsigned int delta = 16; delta; delta >>= 1) {
        sum0 += __shfl_down(sum0, delta, 32);
        sum1 += __shfl_down(sum1, delta, 32);
    }
    if (lane == 0) {
        output[row] = sum0;
        output[(unsigned long long)rows + row] = sum1;
    }
}

// 双波次 verify 固定三行。与 rows2 一样展开 token 维，避免
// rows4 的第四个累加器和动态循环，每行 FMA/归约顺序不变。
extern "C" __global__ void bf16_gemv_f32_rows3(
    const void* input,
    const unsigned short* weight,
    float* output,
    unsigned int columns,
    unsigned int rows,
    unsigned int input_rows,
    unsigned int input_is_bf16)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows || input_rows != 3) return;
    const float* input_f32 = (const float*)input;
    const unsigned short* input_bf16 = (const unsigned short*)input;
    const unsigned short* row_weight = weight + (unsigned long long)row * columns;
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const unsigned long long packed = *reinterpret_cast<const unsigned long long*>(row_weight + column);
        const float w0 = zllm_bf16_to_f32((unsigned short)packed);
        const float w1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
        const float w2 = zllm_bf16_to_f32((unsigned short)(packed >> 32));
        const float w3 = zllm_bf16_to_f32((unsigned short)(packed >> 48));
        if (input_is_bf16) {
            const unsigned long long x0 = *reinterpret_cast<const unsigned long long*>(input_bf16 + column);
            const unsigned long long x1 = *reinterpret_cast<const unsigned long long*>(input_bf16 + (unsigned long long)columns + column);
            const unsigned long long x2 = *reinterpret_cast<const unsigned long long*>(input_bf16 + (unsigned long long)columns * 2 + column);
            sum0 = fmaf(zllm_bf16_to_f32((unsigned short)x0), w0, sum0);
            sum0 = fmaf(zllm_bf16_to_f32((unsigned short)(x0 >> 16)), w1, sum0);
            sum0 = fmaf(zllm_bf16_to_f32((unsigned short)(x0 >> 32)), w2, sum0);
            sum0 = fmaf(zllm_bf16_to_f32((unsigned short)(x0 >> 48)), w3, sum0);
            sum1 = fmaf(zllm_bf16_to_f32((unsigned short)x1), w0, sum1);
            sum1 = fmaf(zllm_bf16_to_f32((unsigned short)(x1 >> 16)), w1, sum1);
            sum1 = fmaf(zllm_bf16_to_f32((unsigned short)(x1 >> 32)), w2, sum1);
            sum1 = fmaf(zllm_bf16_to_f32((unsigned short)(x1 >> 48)), w3, sum1);
            sum2 = fmaf(zllm_bf16_to_f32((unsigned short)x2), w0, sum2);
            sum2 = fmaf(zllm_bf16_to_f32((unsigned short)(x2 >> 16)), w1, sum2);
            sum2 = fmaf(zllm_bf16_to_f32((unsigned short)(x2 >> 32)), w2, sum2);
            sum2 = fmaf(zllm_bf16_to_f32((unsigned short)(x2 >> 48)), w3, sum2);
        } else {
            const float4 x0 = *reinterpret_cast<const float4*>(input_f32 + column);
            const float4 x1 = *reinterpret_cast<const float4*>(input_f32 + (unsigned long long)columns + column);
            const float4 x2 = *reinterpret_cast<const float4*>(input_f32 + (unsigned long long)columns * 2 + column);
            sum0 = fmaf(x0.x, w0, sum0);
            sum0 = fmaf(x0.y, w1, sum0);
            sum0 = fmaf(x0.z, w2, sum0);
            sum0 = fmaf(x0.w, w3, sum0);
            sum1 = fmaf(x1.x, w0, sum1);
            sum1 = fmaf(x1.y, w1, sum1);
            sum1 = fmaf(x1.z, w2, sum1);
            sum1 = fmaf(x1.w, w3, sum1);
            sum2 = fmaf(x2.x, w0, sum2);
            sum2 = fmaf(x2.y, w1, sum2);
            sum2 = fmaf(x2.z, w2, sum2);
            sum2 = fmaf(x2.w, w3, sum2);
        }
    }
    for (; column < columns; column += 32) {
        const float w = zllm_bf16_to_f32(row_weight[column]);
        const float x0 = input_is_bf16 ? zllm_bf16_to_f32(input_bf16[column]) : input_f32[column];
        const float x1 = input_is_bf16 ? zllm_bf16_to_f32(input_bf16[(unsigned long long)columns + column]) : input_f32[(unsigned long long)columns + column];
        const float x2 = input_is_bf16 ? zllm_bf16_to_f32(input_bf16[(unsigned long long)columns * 2 + column]) : input_f32[(unsigned long long)columns * 2 + column];
        sum0 = fmaf(x0, w, sum0);
        sum1 = fmaf(x1, w, sum1);
        sum2 = fmaf(x2, w, sum2);
    }
    for (unsigned int delta = 16; delta; delta >>= 1) {
        sum0 += __shfl_down(sum0, delta, 32);
        sum1 += __shfl_down(sum1, delta, 32);
        sum2 += __shfl_down(sum2, delta, 32);
    }
    if (lane == 0) {
        output[row] = sum0;
        output[(unsigned long long)rows + row] = sum1;
        output[(unsigned long long)rows * 2 + row] = sum2;
    }
}

// Verify 的 3–4 行读取同一份大权重。每个 block 共享一次 weight load，
// 各 token 保留与单行 kernel 完全相同的 FMA/归约顺序。
extern "C" __global__ void bf16_gemv_f32_rows4(
    const void* input,
    const unsigned short* weight,
    float* output,
    unsigned int columns,
    unsigned int rows,
    unsigned int input_rows,
    unsigned int input_is_bf16)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows) return;
    const unsigned int token_base = blockIdx.y * 4;
    const unsigned int token_count = min(4u, input_rows - token_base);
    const float* input_f32 = (const float*)input + (unsigned long long)token_base * columns;
    const unsigned short* input_bf16 = (const unsigned short*)input + (unsigned long long)token_base * columns;
    const unsigned short* row_weight = weight + (unsigned long long)row * columns;
    float sums[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const unsigned long long packed = *reinterpret_cast<const unsigned long long*>(row_weight + column);
        const float w0 = zllm_bf16_to_f32((unsigned short)packed);
        const float w1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
        const float w2 = zllm_bf16_to_f32((unsigned short)(packed >> 32));
        const float w3 = zllm_bf16_to_f32((unsigned short)(packed >> 48));
        for (unsigned int token = 0; token < token_count; ++token) {
            if (input_is_bf16) {
                const unsigned long long x = *reinterpret_cast<const unsigned long long*>(input_bf16 + (unsigned long long)token * columns + column);
                sums[token] = fmaf(zllm_bf16_to_f32((unsigned short)x), w0, sums[token]);
                sums[token] = fmaf(zllm_bf16_to_f32((unsigned short)(x >> 16)), w1, sums[token]);
                sums[token] = fmaf(zllm_bf16_to_f32((unsigned short)(x >> 32)), w2, sums[token]);
                sums[token] = fmaf(zllm_bf16_to_f32((unsigned short)(x >> 48)), w3, sums[token]);
            } else {
                const float4 x = *reinterpret_cast<const float4*>(input_f32 + (unsigned long long)token * columns + column);
                sums[token] = fmaf(x.x, w0, sums[token]);
                sums[token] = fmaf(x.y, w1, sums[token]);
                sums[token] = fmaf(x.z, w2, sums[token]);
                sums[token] = fmaf(x.w, w3, sums[token]);
            }
        }
    }
    for (; column < columns; column += 32) {
        const float w = zllm_bf16_to_f32(row_weight[column]);
        for (unsigned int token = 0; token < token_count; ++token) {
            const float x = input_is_bf16 ? zllm_bf16_to_f32(input_bf16[(unsigned long long)token * columns + column]) : input_f32[(unsigned long long)token * columns + column];
            sums[token] = fmaf(x, w, sums[token]);
        }
    }
    for (unsigned int delta = 16; delta; delta >>= 1) {
        for (unsigned int token = 0; token < token_count; ++token) {
            sums[token] += __shfl_down(sums[token], delta, 32);
        }
    }
    if (lane == 0) {
        for (unsigned int token = 0; token < token_count; ++token) {
            output[(unsigned long long)(token_base + token) * rows + row] = sums[token];
        }
    }
}
// Decode 的 5–8 行读取同一份大权重；逐 token 的 FMA 与归约顺序不变，
// 只把原本两个 rows4 block 的重复权重读取合成一次。
extern "C" __global__ void bf16_gemv_f32_rows8(
    const void* input,
    const unsigned short* weight,
    float* output,
    unsigned int columns,
    unsigned int rows,
    unsigned int input_rows,
    unsigned int input_is_bf16)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows) return;
    const unsigned int token_base = blockIdx.y * 8;
    const unsigned int token_count = min(8u, input_rows - token_base);
    const float* input_f32 = (const float*)input + (unsigned long long)token_base * columns;
    const unsigned short* input_bf16 = (const unsigned short*)input + (unsigned long long)token_base * columns;
    const unsigned short* row_weight = weight + (unsigned long long)row * columns;
    float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    unsigned int column = lane * 4;
    for (; column + 3 < columns; column += 128) {
        const unsigned long long packed = *reinterpret_cast<const unsigned long long*>(row_weight + column);
        const float w0 = zllm_bf16_to_f32((unsigned short)packed);
        const float w1 = zllm_bf16_to_f32((unsigned short)(packed >> 16));
        const float w2 = zllm_bf16_to_f32((unsigned short)(packed >> 32));
        const float w3 = zllm_bf16_to_f32((unsigned short)(packed >> 48));
        for (unsigned int token = 0; token < token_count; ++token) {
            if (input_is_bf16) {
                const unsigned long long x = *reinterpret_cast<const unsigned long long*>(input_bf16 + (unsigned long long)token * columns + column);
                sums[token] = fmaf(zllm_bf16_to_f32((unsigned short)x), w0, sums[token]);
                sums[token] = fmaf(zllm_bf16_to_f32((unsigned short)(x >> 16)), w1, sums[token]);
                sums[token] = fmaf(zllm_bf16_to_f32((unsigned short)(x >> 32)), w2, sums[token]);
                sums[token] = fmaf(zllm_bf16_to_f32((unsigned short)(x >> 48)), w3, sums[token]);
            } else {
                const float4 x = *reinterpret_cast<const float4*>(input_f32 + (unsigned long long)token * columns + column);
                sums[token] = fmaf(x.x, w0, sums[token]);
                sums[token] = fmaf(x.y, w1, sums[token]);
                sums[token] = fmaf(x.z, w2, sums[token]);
                sums[token] = fmaf(x.w, w3, sums[token]);
            }
        }
    }
    for (; column < columns; column += 32) {
        const float w = zllm_bf16_to_f32(row_weight[column]);
        for (unsigned int token = 0; token < token_count; ++token) {
            const float x = input_is_bf16 ? zllm_bf16_to_f32(input_bf16[(unsigned long long)token * columns + column]) : input_f32[(unsigned long long)token * columns + column];
            sums[token] = fmaf(x, w, sums[token]);
        }
    }
    for (unsigned int delta = 16; delta; delta >>= 1) {
        for (unsigned int token = 0; token < token_count; ++token) {
            sums[token] += __shfl_down(sums[token], delta, 32);
        }
    }
    if (lane == 0) {
        for (unsigned int token = 0; token < token_count; ++token) {
            output[(unsigned long long)(token_base + token) * rows + row] = sums[token];
        }
    }
}
extern "C" __global__ void bf16_gemv_bfdot_f32_rows8(
    const void* input,
    const unsigned short* weight,
    float* output,
    unsigned int columns,
    unsigned int rows,
    unsigned int input_rows,
    unsigned int input_is_bf16) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int wave = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * 8 + wave;
    if (row >= rows) {
        return;
    }

    const unsigned int token_base = blockIdx.y * 8;
    const unsigned int token_count = min(8u, input_rows - token_base);
    const float* input_f32 = static_cast<const float*>(input) + token_base * columns;
    const unsigned short* row_weight = weight + row * columns;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    unsigned int column = lane * 4;
    for (; column + 4 <= columns; column += 32 * 4) {
        const zllm_bf16x2 w01 =
            *reinterpret_cast<const zllm_bf16x2*>(row_weight + column);
        const zllm_bf16x2 w23 =
            *reinterpret_cast<const zllm_bf16x2*>(row_weight + column + 2);
        for (unsigned int token = 0; token < token_count; ++token) {
            const float4 x = *reinterpret_cast<const float4*>(
                input_f32 + token * columns + column);
            const zllm_bf16x2 x01 = {
                zllm_f32_to_bf16(x.x), zllm_f32_to_bf16(x.y)};
            const zllm_bf16x2 x23 = {
                zllm_f32_to_bf16(x.z), zllm_f32_to_bf16(x.w)};
            sums[token] = __builtin_amdgcn_fdot2_f32_bf16(x01, w01, sums[token], false);
            sums[token] = __builtin_amdgcn_fdot2_f32_bf16(x23, w23, sums[token], false);
        }
    }
    for (; column < columns; column += 32) {
        const float w = zllm_bf16_to_f32(row_weight[column]);
        for (unsigned int token = 0; token < token_count; ++token) {
            sums[token] += input_f32[token * columns + column] * w;
        }
    }

    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        for (unsigned int token = 0; token < token_count; ++token) {
            sums[token] += __shfl_down(sums[token], offset);
        }
    }
    if (lane == 0) {
        for (unsigned int token = 0; token < token_count; ++token) {
            output[(unsigned long long)(token_base + token) * rows + row] = sums[token];
        }
    }
}
