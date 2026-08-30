/// 本模块的 CUDA shader(本文件用到的 kernel + 文件私有 `__device__` helper)。
///
/// 共用 helper 见 `mod.rs` 前导。`mod.rs` 的 `kernels_source()`
/// 把 mod.rs 的 `PREAMBLE_SHADERS` 与各模块的 `SHADERS` 拼成完整字符串。
// kernels: gguf_kq_matmul_f16, gguf_kq_gemv_f16, gated_linear_q4_k_silu_f16, gated_linear_q4_k_silu_rows8_f16, linear_q5_k_f16, linear_q5_k_accumulate_f32, linear_q6_k_f16, linear_q6_k_accumulate_f32
// private helpers: q4_k_scale_min, q4_k_block_dot_pair, q5_k_block_dot, q5_k_row_dot, q6_k_block_dot, q6_k_row_dot
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
    return tensor_type == 8 ? 34u : (tensor_type == 12 ? 144u : (tensor_type == 13 ? 176u : 210u));
}
__device__ __forceinline__ unsigned int gguf_kq_block_elements(const unsigned int tensor_type)
{
    return tensor_type == 8 ? 32u : 256u;
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
    let mut output = ctx.tensor_uninit(input.rows, out_cols)?;
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
        8 => 34,
        12 => 144,
        13 => 176,
        14 => 210,
        other => return Err(format!("CUDA GGUF K-quant 不支持 type={other}")),
    };
    let block_elements = if tensor_type == 8 { 32 } else { 256 };
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
    if input.rows == 1 {
        let output = ctx.tensor_uninit(1, out_rows)?;
        let func = ctx.function("gguf_kq_gemv_f16")?;
        let cols_u32 = input.cols as u32;
        let out_rows_u32 = out_rows as u32;
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
    let output = ctx.tensor_uninit(input.rows, out_rows)?;
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
    let output = ctx.tensor_uninit(input.rows, out_cols)?;
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
    let output = ctx.tensor_uninit(input.rows, out_cols)?;
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
    let output = ctx.tensor_uninit(input.rows, out_cols)?;
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
            8 => 34,
            12 => 144,
            13 => 176,
            14 => 210,
            _ => unreachable!(),
        };
        let blocks_per_row = if tensor_type == 8 { 8 } else { 1 };
        let mut bytes = vec![0u8; rows * blocks_per_row * block_bytes];
        for block_index in 0..rows * blocks_per_row {
            let row = block_index / blocks_per_row;
            let block = &mut bytes[block_index * block_bytes..(block_index + 1) * block_bytes];
            if tensor_type == 8 {
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
        for tensor_type in [8, 12, 13, 14] {
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
}
