#include <rocwmma/rocwmma.hpp>

extern "C" __global__ void fill_zero_f32(float* output, unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = 0.0f;
}

extern "C" __global__ void fill_u32(unsigned int* output, unsigned int value, unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = value;
}

extern "C" __global__ void add_scaled_f32(
    const float* left,
    const float* right,
    float* output,
    unsigned int elements,
    float scale)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = (left[index] + right[index]) * scale;
}

extern "C" __global__ void subtract_f32(
    const float* left,
    const float* right,
    float* output,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = left[index] - right[index];
}

extern "C" __global__ void relative_l1_delta_partial_f32(
    const float* output,
    const float* input,
    const float* previous,
    float* partial,
    unsigned int elements)
{
    const unsigned int lane = threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int index = blockIdx.x * blockDim.x + lane;
    float numerator = 0.0f;
    float denominator = 0.0f;
    while (index < elements) {
        const float current = output[index] - input[index];
        const float old_value = previous[index];
        numerator += fabsf(current - old_value);
        denominator += fabsf(old_value);
        index += stride;
    }
    __shared__ float numerator_partial[256];
    __shared__ float denominator_partial[256];
    numerator_partial[lane] = numerator;
    denominator_partial[lane] = denominator;
    __syncthreads();
    for (unsigned int width = blockDim.x >> 1; width > 0; width >>= 1) {
        if (lane < width) {
            numerator_partial[lane] += numerator_partial[lane + width];
            denominator_partial[lane] += denominator_partial[lane + width];
        }
        __syncthreads();
    }
    if (lane == 0) {
        partial[blockIdx.x * 2] = numerator_partial[0];
        partial[blockIdx.x * 2 + 1] = denominator_partial[0];
    }
}

extern "C" __global__ void add_bf16_f32(
    const unsigned short* left,
    const float* right,
    float* output,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float left_value = __uint_as_float((unsigned int)left[index] << 16);
        output[index] = left_value + right[index];
    }
}

extern "C" __global__ void validate_finite_f32(
    const void* input,
    unsigned int* invalid,
    unsigned long long offset,
    unsigned int elements,
    unsigned int input_is_bf16)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned long long input_index = offset + index;
    const float value = input_is_bf16
        ? __uint_as_float((unsigned int)((const unsigned short*)input)[input_index] << 16)
        : ((const float*)input)[input_index];
    if (!isfinite(value) || fabsf(value) > 1.0e10f) atomicExch(invalid, 1u);
}

extern "C" __global__ void rmsnorm_to_f32(
    const void* input,
    const float* weight,
    void* output,
    unsigned short* quantized_output,
    unsigned int rows,
    unsigned int columns,
    float eps,
    unsigned int gemma,
    unsigned int input_is_bf16,
    unsigned int output_is_bf16)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    const unsigned long long base = (unsigned long long)row * columns;
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        const float value = input_is_bf16
            ? __uint_as_float((unsigned int)((const unsigned short*)input)[base + column] << 16)
            : ((const float*)input)[base + column];
        sum += value * value;
    }
    const unsigned int wave_lane = lane % warpSize;
    const unsigned int wave = lane / warpSize;
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    __shared__ float partial[8];
    if (wave_lane == 0) partial[wave] = sum;
    __syncthreads();
    if (wave == 0) {
        sum = wave_lane < blockDim.x / warpSize ? partial[wave_lane] : 0.0f;
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
        if (wave_lane == 0) partial[0] = sum;
    }
    __syncthreads();
    const float inverse = rsqrtf(partial[0] / (float)columns + eps);
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        const float gain = gemma ? 1.0f + weight[column] : weight[column];
        const float value = input_is_bf16
            ? __uint_as_float((unsigned int)((const unsigned short*)input)[base + column] << 16)
            : ((const float*)input)[base + column];
        const float normalized = value * inverse * gain;
        if (output_is_bf16) {
            ((unsigned short*)output)[base + column] = zllm_f32_to_bf16(normalized);
        } else {
            ((float*)output)[base + column] = normalized;
            if (quantized_output) {
                quantized_output[base + column] = zllm_f32_to_bf16(normalized);
            }
        }
    }
}

extern "C" __global__ void rmsnorm_adaln_segment_f32(
    const void* input,
    const float* weight,
    const float* shift,
    const float* scale,
    float* output,
    unsigned int start_row,
    unsigned int row_count,
    unsigned int columns,
    unsigned int modulation_row,
    float eps,
    unsigned int input_is_bf16)
{
    const unsigned int local_row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (local_row >= row_count) return;
    const unsigned int row = start_row + local_row;
    const unsigned long long base = (unsigned long long)row * columns;
    float sum = 0.0f;
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        const float value = input_is_bf16
            ? __uint_as_float((unsigned int)((const unsigned short*)input)[base + column] << 16)
            : ((const float*)input)[base + column];
        sum += value * value;
    }
    const unsigned int wave_lane = lane % warpSize;
    const unsigned int wave = lane / warpSize;
    for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
    __shared__ float partial[8];
    if (wave_lane == 0) partial[wave] = sum;
    __syncthreads();
    if (wave == 0) {
        sum = wave_lane < blockDim.x / warpSize ? partial[wave_lane] : 0.0f;
        for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
        if (wave_lane == 0) partial[0] = sum;
    }
    __syncthreads();
    const float inverse = rsqrtf(partial[0] / (float)columns + eps);
    const unsigned long long modulation = (unsigned long long)modulation_row * columns;
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        const float value = input_is_bf16
            ? __uint_as_float((unsigned int)((const unsigned short*)input)[base + column] << 16)
            : ((const float*)input)[base + column];
        const float normalized = value * inverse * weight[column];
        output[base + column] = normalized * (1.0f + scale[modulation + column]) + shift[modulation + column];
    }
}

extern "C" __global__ void layernorm_bias_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    unsigned int rows,
    unsigned int columns,
    float eps)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows) return;
    const unsigned long long base = (unsigned long long)row * columns;
    float sum = 0.0f;
    float square = 0.0f;
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        const float value = input[base + column];
        sum += value;
        square += value * value;
    }
    __shared__ float partial_sum[256];
    __shared__ float partial_square[256];
    partial_sum[lane] = sum;
    partial_square[lane] = square;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial_sum[lane] += partial_sum[lane + stride];
            partial_square[lane] += partial_square[lane + stride];
        }
        __syncthreads();
    }
    const float mean = partial_sum[0] / (float)columns;
    const float variance = fmaxf(partial_square[0] / (float)columns - mean * mean, 0.0f);
    const float inverse = rsqrtf(variance + eps);
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        output[base + column] = (input[base + column] - mean) * inverse * weight[column] + bias[column];
    }
}

extern "C" __global__ void rmsnorm_heads_unit_f32(
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int heads,
    unsigned int head_dim,
    float eps)
{
    const unsigned int item = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (item >= rows * heads) return;
    const unsigned long long base = (unsigned long long)item * head_dim;
    float square = 0.0f;
    for (unsigned int column = lane; column < head_dim; column += blockDim.x) {
        const float value = input[base + column];
        square += value * value;
    }
    __shared__ float partial[256];
    partial[lane] = square;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(partial[0] / (float)head_dim + eps);
    for (unsigned int column = lane; column < head_dim; column += blockDim.x) {
        output[base + column] = input[base + column] * inverse;
    }
}

extern "C" __global__ void rmsnorm_rope_pair_unit_f32(
    const float* query,
    const float* key,
    const float* cosine,
    const float* sine,
    float* query_output,
    float* key_output,
    unsigned int rows,
    unsigned int heads,
    unsigned int head_dim,
    unsigned int rotary_dim,
    float eps)
{
    const unsigned int item = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (item >= rows * heads) return;
    const unsigned int row = item / heads;
    const unsigned long long base = (unsigned long long)item * head_dim;
    __shared__ float query_values[256];
    __shared__ float key_values[256];
    __shared__ float query_squares[256];
    __shared__ float key_squares[256];
    float query_value = 0.0f;
    float key_value = 0.0f;
    if (lane < head_dim) {
        query_value = query[base + lane];
        key_value = key[base + lane];
    }
    query_squares[lane] = query_value * query_value;
    key_squares[lane] = key_value * key_value;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            query_squares[lane] += query_squares[lane + stride];
            key_squares[lane] += key_squares[lane + stride];
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        query_values[lane] = query_value * rsqrtf(query_squares[0] / (float)head_dim + eps);
        key_values[lane] = key_value * rsqrtf(key_squares[0] / (float)head_dim + eps);
    }
    __syncthreads();
    if (lane < head_dim) {
        float rotated_query = query_values[lane];
        float rotated_key = key_values[lane];
        if (lane < rotary_dim) {
            const unsigned int half = rotary_dim / 2u;
            const unsigned int pair = lane % half;
            const float c = cosine[(unsigned long long)row * half + pair];
            const float s = sine[(unsigned long long)row * half + pair];
            const float query_real = query_values[pair];
            const float query_imaginary = query_values[half + pair];
            const float key_real = key_values[pair];
            const float key_imaginary = key_values[half + pair];
            rotated_query = lane < half ? query_real * c - query_imaginary * s : query_imaginary * c + query_real * s;
            rotated_key = lane < half ? key_real * c - key_imaginary * s : key_imaginary * c + key_real * s;
        }
        query_output[base + lane] = rotated_query;
        key_output[base + lane] = rotated_key;
    }
}

extern "C" __global__ void scaled_residual_columns_f32(
    const float* input,
    const float* update,
    const float* bias,
    const float* scale,
    float* output,
    unsigned int elements,
    unsigned int columns,
    unsigned int has_bias)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const unsigned int column = index % columns;
        output[index] = input[index] + (update[index] + (has_bias ? bias[column] : 0.0f)) * scale[column];
    }
}

extern "C" __global__ void unpatch_affine_f32(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    unsigned int channels,
    unsigned int time,
    unsigned int height,
    unsigned int width,
    unsigned int patch_t,
    unsigned int patch_h,
    unsigned int patch_w,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int channel = index % channels;
    const unsigned int voxel = index / channels;
    const unsigned int x = voxel % width;
    const unsigned int y = (voxel / width) % height;
    const unsigned int t = voxel / (height * width);
    const unsigned int grid_h = height / patch_h;
    const unsigned int grid_w = width / patch_w;
    const unsigned int patch_row = ((t / patch_t) * grid_h + y / patch_h) * grid_w + x / patch_w;
    const unsigned int patch_columns = channels * patch_t * patch_h * patch_w;
    const unsigned int patch_column = (((channel * patch_t + t % patch_t) * patch_h + y % patch_h) * patch_w) + x % patch_w;
    output[index] = input[(unsigned long long)patch_row * patch_columns + patch_column] * scale[channel] + bias[channel];
}

extern "C" __global__ void sigmoid_gate_f32(
    const float* input,
    const float* gate,
    float* output,
    unsigned int columns,
    unsigned int gate_columns,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int row = index / columns;
    const unsigned int gate_index = gate_columns == 1 ? row : index;
    output[index] = input[index] / (1.0f + expf(-gate[gate_index]));
}

extern "C" __global__ void split_interleaved_columns_f32(
    const float* input,
    float* left,
    float* right,
    unsigned int columns,
    unsigned int block_columns,
    unsigned int output_elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    const unsigned int output_columns = columns / 2;
    const unsigned int row = index / output_columns;
    const unsigned int output_column = index - row * output_columns;
    const unsigned int block = output_column / block_columns;
    const unsigned int offset = output_column - block * block_columns;
    const unsigned long long input_base = (unsigned long long)row * columns + block * block_columns * 2 + offset;
    left[index] = input[input_base];
    right[index] = input[input_base + block_columns];
}

extern "C" __global__ void gated_activation_f32(
    const float* gate,
    const float* up,
    float* output,
    unsigned int elements,
    unsigned int activation,
    float alpha,
    float beta,
    float limit,
    float linear_beta,
    unsigned int has_linear_beta)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float g = gate[index];
    float u = up[index];
    float value;
    if (activation == 0) {
        value = g / (1.0f + expf(-g)) * u;
    } else if (activation == 1) {
        g = fminf(g, limit);
        u = fmaxf(-limit, fminf(u, limit));
        value = g / (1.0f + expf(-g)) * u;
    } else if (activation == 2) {
        if (has_linear_beta) u = linear_beta * tanhf(u / linear_beta);
        value = beta * tanhf(g / beta) / (1.0f + expf(-g)) * u;
    } else if (activation == 3) {
        g = fminf(g, limit);
        u = fmaxf(-limit, fminf(u, limit));
        value = g / (1.0f + expf(-alpha * g)) * (u + 1.0f);
    } else {
        const float inner = 0.7978846f * (g + 0.044715f * g * g * g);
        value = 0.5f * g * (1.0f + tanhf(inner)) * u;
    }
    output[index] = value;
}

extern "C" __global__ void gated_activation_bf16(
    const float* gate,
    const float* up,
    unsigned short* output,
    unsigned int elements,
    unsigned int activation,
    float alpha,
    float beta,
    float limit,
    float linear_beta,
    unsigned int has_linear_beta)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float g = gate[index];
    float u = up[index];
    float value;
    if (activation == 0) {
        value = g / (1.0f + expf(-g)) * u;
    } else if (activation == 1) {
        g = fminf(g, limit);
        u = fmaxf(-limit, fminf(u, limit));
        value = g / (1.0f + expf(-g)) * u;
    } else if (activation == 2) {
        if (has_linear_beta) u = linear_beta * tanhf(u / linear_beta);
        value = beta * tanhf(g / beta) / (1.0f + expf(-g)) * u;
    } else if (activation == 3) {
        g = fminf(g, limit);
        u = fmaxf(-limit, fminf(u, limit));
        value = g / (1.0f + expf(-alpha * g)) * (u + 1.0f);
    } else {
        const float inner = 0.7978846f * (g + 0.044715f * g * g * g);
        value = 0.5f * g * (1.0f + tanhf(inner)) * u;
    }
    output[index] = zllm_f32_to_bf16(value);
}

extern "C" __global__ void split_gated_activation_bf16(
    const float* input,
    const float* bias,
    unsigned short* output,
    unsigned int rows,
    unsigned int columns,
    unsigned int has_bias,
    unsigned int activation,
    float alpha,
    float beta,
    float limit,
    float linear_beta,
    unsigned int has_linear_beta)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = rows * columns;
    if (index >= elements) return;
    const unsigned int row = index / columns;
    const unsigned int column = index - row * columns;
    const unsigned long long base = (unsigned long long)row * columns * 2;
    float g = input[base + column] + (has_bias ? bias[column] : 0.0f);
    float u = input[base + columns + column] + (has_bias ? bias[columns + column] : 0.0f);
    float value;
    if (activation == 0) {
        value = g / (1.0f + expf(-g)) * u;
    } else if (activation == 1) {
        g = fminf(g, limit);
        u = fmaxf(-limit, fminf(u, limit));
        value = g / (1.0f + expf(-g)) * u;
    } else if (activation == 2) {
        if (has_linear_beta) u = linear_beta * tanhf(u / linear_beta);
        value = beta * tanhf(g / beta) / (1.0f + expf(-g)) * u;
    } else if (activation == 3) {
        g = fminf(g, limit);
        u = fmaxf(-limit, fminf(u, limit));
        value = g / (1.0f + expf(-alpha * g)) * (u + 1.0f);
    } else {
        const float inner = 0.7978846f * (g + 0.044715f * g * g * g);
        value = 0.5f * g * (1.0f + tanhf(inner)) * u;
    }
    output[index] = zllm_f32_to_bf16(value);
}

extern "C" __global__ void split_columns_f32(
    const float* input,
    const float* bias,
    float* left,
    float* middle,
    float* right,
    unsigned int rows,
    unsigned int columns,
    unsigned int left_columns,
    unsigned int middle_columns,
    unsigned int has_bias,
    unsigned int has_middle)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = rows * columns;
    if (index >= elements) return;
    const unsigned int row = index / columns;
    const unsigned int column = index - row * columns;
    const float value = input[index] + (has_bias ? bias[column] : 0.0f);
    if (column < left_columns) {
        left[(unsigned long long)row * left_columns + column] = value;
    } else if (has_middle && column < left_columns + middle_columns) {
        middle[(unsigned long long)row * middle_columns + column - left_columns] = value;
    } else {
        const unsigned int right_base = left_columns + (has_middle ? middle_columns : 0);
        const unsigned int right_columns = columns - right_base;
        right[(unsigned long long)row * right_columns + column - right_base] = value;
    }
}

extern "C" __global__ void concat_columns_f32(
    const float* left,
    const float* right,
    float* output,
    unsigned int rows,
    unsigned int left_columns,
    unsigned int right_columns)
{
    const unsigned int columns = left_columns + right_columns;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = rows * columns;
    if (index >= elements) return;
    const unsigned int row = index / columns;
    const unsigned int column = index - row * columns;
    output[index] = column < left_columns
        ? left[(unsigned long long)row * left_columns + column]
        : right[(unsigned long long)row * right_columns + column - left_columns];
}

extern "C" __global__ void compact_qkv_head_range_f32(
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int total_head_count,
    unsigned int head_start,
    unsigned int head_count,
    unsigned int head_dim)
{
    const unsigned int local_columns = head_count * head_dim;
    const unsigned int output_columns = local_columns * 3u;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = rows * output_columns;
    if (index >= elements) return;
    const unsigned int row = index / output_columns;
    const unsigned int column = index - row * output_columns;
    const unsigned int section = column / local_columns;
    const unsigned int local_column = column - section * local_columns;
    const unsigned int total_columns = total_head_count * head_dim;
    output[index] = input[(unsigned long long)row * (total_columns * 3u)
        + section * total_columns + head_start * head_dim + local_column];
}
