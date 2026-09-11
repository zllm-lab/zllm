#include <hip/hip_runtime.h>

__device__ float weight_value(const void* values, unsigned long long index, unsigned bf16) {
    return bf16 ? __uint_as_float((unsigned)((const unsigned short*)values)[index] << 16) : ((const float*)values)[index];
}

extern "C" __global__ void grouped_block_conv_f32(const float* input, const float* delta, const void* base, float* out,
    unsigned rows, unsigned cols, unsigned block_rows, unsigned groups, unsigned taps, unsigned side, unsigned bf16) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (unsigned long long)rows * cols) return;
    unsigned row = index / cols, col = index % cols;
    float value = 0;
    for (unsigned tap = 0; tap < taps && tap <= row % block_rows; ++tap) {
        float coefficient = weight_value(base, ((unsigned long long)side * taps + tap) * cols + col, bf16)
            + delta[(((unsigned long long)row * 2 + side) * taps + tap) * groups + col / (cols / groups)];
        value += coefficient * input[(unsigned long long)(row - tap) * cols + col];
    }
    out[index] = value;
}

// 以已选 pair 作上界逐次归约，避免修改 logits 或维护词表大小的 mask。
extern "C" __global__ void candidate_topk_f32(const float* logits, unsigned* ids, float* unary, unsigned vocab, unsigned k) {
    __shared__ float scores[256];
    __shared__ unsigned tokens[256];
    __shared__ unsigned invalid;
    unsigned lane = threadIdx.x, row = blockIdx.x;
    if (lane == 0) invalid = 0;
    __syncthreads();
    float upper = __uint_as_float(0x7f800000u);
    unsigned upper_id = 0;
    for (unsigned choice = 0; choice < k; ++choice) {
        float best = -__uint_as_float(0x7f800000u);
        unsigned token = ~0u;
        for (unsigned col = lane; col < vocab; col += 256) {
            float value = logits[(unsigned long long)row * vocab + col];
            if (isnan(value) || value == __uint_as_float(0x7f800000u)) atomicOr(&invalid, 1u);
            if (!isfinite(value) || value > upper || (value == upper && col <= upper_id)) continue;
            if (value > best || (value == best && col < token)) { best = value; token = col; }
        }
        scores[lane] = best; tokens[lane] = token;
        __syncthreads();
        for (unsigned stride = 128; stride; stride >>= 1) {
            if (lane < stride && (scores[lane + stride] > scores[lane] ||
                (scores[lane + stride] == scores[lane] && tokens[lane + stride] < tokens[lane]))) {
                scores[lane] = scores[lane + stride]; tokens[lane] = tokens[lane + stride];
            }
            __syncthreads();
        }
        upper = scores[0]; upper_id = tokens[0];
        if (lane == 0) { ids[row*k+choice] = invalid ? ~0u : upper_id; unary[row*k+choice] = upper; }
        __syncthreads();
    }
}

// 每轮只读取所选前驱；K×rank 留在设备上，回传值仅为最终 token IDs。
extern "C" __global__ void candidate_greedy_path_f32(const unsigned* ids, const float* unary, const float* gate,
    const void* predecessor, const void* successor, unsigned* out, unsigned rows, unsigned rank, unsigned k, unsigned anchor, unsigned pred_bf16, unsigned succ_bf16) {
    __shared__ float scores[256];
    __shared__ unsigned tokens[256];
    __shared__ unsigned invalid;
    unsigned lane = threadIdx.x, previous = anchor;
    if (lane == 0) invalid = 0;
    __syncthreads();
    for (unsigned row = 0; row < rows; ++row) {
        float best = -__uint_as_float(0x7f800000u);
        unsigned token = ~0u;
        for (unsigned c = lane; c < k; c += 256) {
            unsigned candidate = ids[row*k+c];
            if (candidate == ~0u) continue;
            float score = unary[row*k+c];
            for (unsigned d = 0; d < rank; ++d) {
                score += weight_value(predecessor, (unsigned long long)previous*rank+d, pred_bf16)
                    * gate[(unsigned long long)row*rank+d] * weight_value(successor, (unsigned long long)candidate*rank+d, succ_bf16);
            }
            if (!isfinite(score)) atomicOr(&invalid, 1u);
            if (score > best || (score == best && candidate < token)) { best = score; token = candidate; }
        }
        scores[lane] = best; tokens[lane] = token;
        __syncthreads();
        for (unsigned stride = 128; stride; stride >>= 1) {
            if (lane < stride && (scores[lane+stride] > scores[lane] || (scores[lane+stride] == scores[lane] && tokens[lane+stride] < tokens[lane]))) {
                scores[lane] = scores[lane+stride]; tokens[lane] = tokens[lane+stride];
            }
            __syncthreads();
        }
        unsigned next = invalid ? ~0u : tokens[0];
        if (lane == 0) out[row] = next;
        if (next == ~0u) {
            for (unsigned tail = row+1+lane; tail < rows; tail += 256) out[tail] = ~0u;
            return;
        }
        previous = next;
        __syncthreads();
    }
}
