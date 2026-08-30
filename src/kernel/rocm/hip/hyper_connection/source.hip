extern "C" __global__ void mhc_expand_f32(
    const float* __restrict__ hidden,
    float* __restrict__ output,
    const unsigned int columns,
    const unsigned int copies,
    const unsigned long long count)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const unsigned long long output_columns = (unsigned long long)columns * copies;
    const unsigned long long row = gid / output_columns;
    const unsigned int column = (unsigned int)(gid - row * output_columns);
    output[gid] = hidden[row * columns + (column % columns)];
}

extern "C" __global__ void mhc_reduce_f32(
    const float* __restrict__ hidden,
    const float* __restrict__ coefficients,
    float* __restrict__ output,
    const unsigned int width,
    const unsigned int copies,
    const unsigned long long count)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const unsigned long long row = gid / width;
    const unsigned int column = (unsigned int)(gid - row * width);
    const unsigned long long hidden_row = row * copies * width;
    const unsigned long long coefficient_row = row * copies;
    float sum = 0.0f;
    for (unsigned int copy = 0; copy < copies; ++copy) {
        sum += coefficients[coefficient_row + copy] * hidden[hidden_row + (unsigned long long)copy * width + column];
    }
    output[gid] = sum;
}

extern "C" __global__ void mhc_mean_f32(
    const float* __restrict__ hidden,
    float* __restrict__ output,
    const unsigned int width,
    const unsigned int copies,
    const unsigned long long count)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const unsigned long long row = gid / width;
    const unsigned int column = (unsigned int)(gid - row * width);
    const unsigned long long hidden_row = row * copies * width;
    float sum = 0.0f;
    for (unsigned int copy = 0; copy < copies; ++copy) {
        sum += hidden[hidden_row + (unsigned long long)copy * width + column];
    }
    output[gid] = sum / (float)copies;
}

extern "C" __global__ void mhc_expand_scaled_f32(
    const float* __restrict__ hidden,
    const float* __restrict__ coefficients,
    float* __restrict__ output,
    const unsigned int width,
    const unsigned int copies,
    const unsigned long long count)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const unsigned long long output_columns = (unsigned long long)width * copies;
    const unsigned long long row = gid / output_columns;
    const unsigned int column = (unsigned int)(gid - row * output_columns);
    const unsigned int copy = column / width;
    const unsigned int source_column = column - copy * width;
    output[gid] = coefficients[row * copies + copy] * hidden[row * width + source_column];
}

__device__ __forceinline__ float zllm_mhc_mul_add_rounded_f32(float left, float right, float addend)
{
    float product;
    float result;
    // 原路径跨两个 kernel，在乘法后必然舍入。显式 ISA 防止融合后被收缩成 FMA。
    asm volatile("v_mul_f32_e32 %0, %1, %2" : "=v"(product) : "v"(left), "v"(right));
    asm volatile("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(addend), "v"(product));
    return result;
}

extern "C" __global__ void mhc_expand_scaled_add_f32(
    const float* __restrict__ hidden,
    const float* __restrict__ coefficients,
    const float* __restrict__ residual,
    float* __restrict__ output,
    const unsigned int width,
    const unsigned int copies,
    const unsigned long long count)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const unsigned long long output_columns = (unsigned long long)width * copies;
    const unsigned long long row = gid / output_columns;
    const unsigned int column = (unsigned int)(gid - row * output_columns);
    const unsigned int copy = column / width;
    const unsigned int source_column = column - copy * width;
    output[gid] = zllm_mhc_mul_add_rounded_f32(coefficients[row * copies + copy], hidden[row * width + source_column], residual[gid]);
}

extern "C" __global__ void mhc_mix_f32(
    const float* __restrict__ hidden,
    const float* __restrict__ matrix,
    float* __restrict__ output,
    const unsigned int width,
    const unsigned int copies,
    const unsigned long long count)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const unsigned long long output_columns = (unsigned long long)width * copies;
    const unsigned long long row = gid / output_columns;
    const unsigned int column = (unsigned int)(gid - row * output_columns);
    const unsigned int output_copy = column / width;
    const unsigned int hidden_column = column - output_copy * width;
    const unsigned long long hidden_row = row * output_columns;
    const unsigned long long matrix_row = row * copies * copies + (unsigned long long)output_copy * copies;
    float sum = 0.0f;
    for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
        sum += matrix[matrix_row + input_copy] * hidden[hidden_row + (unsigned long long)input_copy * width + hidden_column];
    }
    output[gid] = sum;
}

__device__ __forceinline__ void zllm_mhc_split_row(
    const float* __restrict__ mixes,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ combination,
    const unsigned int copies,
    const unsigned int iterations,
    const float eps)
{
    for (unsigned int copy = 0; copy < copies; ++copy) {
        const float pre_logit = mixes[copy] * scale[0] + base[copy];
        pre[copy] = 1.0f / (1.0f + expf(-pre_logit)) + eps;
        const unsigned int post_offset = copies + copy;
        const float post_logit = mixes[post_offset] * scale[1] + base[post_offset];
        post[copy] = 2.0f / (1.0f + expf(-post_logit));
    }

    for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) {
        float maximum = __int_as_float(0xff800000u);
        for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
            const unsigned int index = output_copy * copies + input_copy;
            const unsigned int offset = 2 * copies + index;
            const float logit = mixes[offset] * scale[2] + base[offset];
            maximum = fmaxf(maximum, logit);
        }
        float sum = 0.0f;
        for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
            const unsigned int index = output_copy * copies + input_copy;
            const unsigned int offset = 2 * copies + index;
            const float logit = mixes[offset] * scale[2] + base[offset];
            const float value = expf(logit - maximum);
            combination[index] = value;
            sum += value;
        }
        for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
            const unsigned int index = output_copy * copies + input_copy;
            combination[index] = combination[index] / sum + eps;
        }
    }
    for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
        float sum = 0.0f;
        for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) {
            sum += combination[output_copy * copies + input_copy];
        }
        for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) {
            const unsigned int index = output_copy * copies + input_copy;
            combination[index] /= sum + eps;
        }
    }
    for (unsigned int iteration = 1; iteration < iterations; ++iteration) {
        for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) {
            float sum = 0.0f;
            for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
                sum += combination[output_copy * copies + input_copy];
            }
            for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
                const unsigned int index = output_copy * copies + input_copy;
                combination[index] /= sum + eps;
            }
        }
        for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
            float sum = 0.0f;
            for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) {
                sum += combination[output_copy * copies + input_copy];
            }
            for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) {
                const unsigned int index = output_copy * copies + input_copy;
                combination[index] /= sum + eps;
            }
        }
    }
}

// 每行一个线程:pre/post sigmoid 与完整 Sinkhorn 都在 thread-local 内完成。
extern "C" __global__ void mhc_split_f32(
    const float* __restrict__ mixes,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ combination,
    const unsigned int copies,
    const unsigned int iterations,
    const float eps,
    const unsigned int rows)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const unsigned int mix_columns = (2u + copies) * copies;
    zllm_mhc_split_row(mixes + (unsigned long long)row * mix_columns, base, scale, pre + (unsigned long long)row * copies, post + (unsigned long long)row * copies, combination + (unsigned long long)row * copies * copies, copies, iterations, eps);
}

extern "C" __global__ void mhc_prepare_coefficients_f32(
    const float* __restrict__ mixes,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float* __restrict__ coefficients,
    const unsigned int copies,
    const unsigned int iterations,
    const float eps,
    const unsigned int rows)
{
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int mix_columns = (2u + copies) * copies;
    mixes += (unsigned long long)row * mix_columns;
    float* pre_output = coefficients + (unsigned long long)row * copies;
    float* post = coefficients + (unsigned long long)(rows + row) * copies;
    float* combination_output = coefficients + (unsigned long long)2u * rows * copies + (unsigned long long)row * copies * copies;
    __shared__ float pre[16];
    __shared__ float combination[16 * 16];
    // 单 token 也不能把全部 sigmoid/softmax/Sinkhorn 串在一个线程上；按 copy
    // 行列展开到 block，保留每行内部的确定性归约顺序。
    if (threadIdx.x < copies) {
        const unsigned int copy = threadIdx.x;
        const float pre_logit = mixes[copy] * scale[0] + base[copy];
        pre[copy] = 1.0f / (1.0f + expf(-pre_logit)) + eps;
        const unsigned int post_offset = copies + copy;
        const float post_logit = mixes[post_offset] * scale[1] + base[post_offset];
        post[copy] = 2.0f / (1.0f + expf(-post_logit));
    }
    const unsigned int cell = threadIdx.x;
    if (cell < copies * copies) {
        const unsigned int offset = 2u * copies + cell;
        combination[cell] = mixes[offset] * scale[2] + base[offset];
    }
    __syncthreads();
    if (threadIdx.x < copies) {
        const unsigned int output_copy = threadIdx.x;
        float maximum = __int_as_float(0xff800000u);
        for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) maximum = fmaxf(maximum, combination[output_copy * copies + input_copy]);
        float sum = 0.0f;
        for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
            const unsigned int index = output_copy * copies + input_copy;
            const float value = expf(combination[index] - maximum);
            combination[index] = value;
            sum += value;
        }
        for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) {
            const unsigned int index = output_copy * copies + input_copy;
            combination[index] = combination[index] / sum + eps;
        }
    }
    __syncthreads();
    if (threadIdx.x < copies) {
        const unsigned int input_copy = threadIdx.x;
        float sum = 0.0f;
        for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) sum += combination[output_copy * copies + input_copy];
        for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) combination[output_copy * copies + input_copy] /= sum + eps;
    }
    __syncthreads();
    for (unsigned int iteration = 1; iteration < iterations; ++iteration) {
        if (threadIdx.x < copies) {
            const unsigned int output_copy = threadIdx.x;
            float sum = 0.0f;
            for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) sum += combination[output_copy * copies + input_copy];
            for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) combination[output_copy * copies + input_copy] /= sum + eps;
        }
        __syncthreads();
        if (threadIdx.x < copies) {
            const unsigned int input_copy = threadIdx.x;
            float sum = 0.0f;
            for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) sum += combination[output_copy * copies + input_copy];
            for (unsigned int output_copy = 0; output_copy < copies; ++output_copy) combination[output_copy * copies + input_copy] /= sum + eps;
        }
        __syncthreads();
    }
    if (threadIdx.x < copies) pre_output[threadIdx.x] = pre[threadIdx.x];
    if (threadIdx.x < copies * copies) combination_output[threadIdx.x] = combination[threadIdx.x];
}

extern "C" __global__ void mhc_prepare_outputs_f32(
    const float* __restrict__ hidden,
    const float* __restrict__ coefficients,
    float* __restrict__ residual,
    float* __restrict__ reduced,
    const unsigned int width,
    const unsigned int copies,
    const unsigned int rows)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long expanded = (unsigned long long)width * copies;
    const unsigned long long row_elements = expanded + width;
    if (gid >= (unsigned long long)rows * row_elements) return;
    const unsigned int row = (unsigned int)(gid / row_elements);
    const unsigned int index = (unsigned int)(gid - (unsigned long long)row * row_elements);
    hidden += (unsigned long long)row * expanded;
    const float* pre = coefficients + (unsigned long long)row * copies;
    const float* combination = coefficients + (unsigned long long)2u * rows * copies + (unsigned long long)row * copies * copies;
    if (index < expanded) {
        const unsigned int output_copy = index / width;
        const unsigned int column = index - output_copy * width;
        float sum = 0.0f;
        for (unsigned int input_copy = 0; input_copy < copies; ++input_copy) sum += combination[output_copy * copies + input_copy] * hidden[input_copy * width + column];
        residual[(unsigned long long)row * expanded + index] = sum;
    } else {
        const unsigned int column = index - expanded;
        float sum = 0.0f;
        for (unsigned int copy = 0; copy < copies; ++copy) sum += pre[copy] * hidden[copy * width + column];
        reduced[(unsigned long long)row * width + column] = sum;
    }
}

extern "C" __global__ void mhc_head_reduce_f32(
    const float* __restrict__ hidden,
    const float* __restrict__ mixes,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float* __restrict__ output,
    const unsigned int width,
    const unsigned int copies,
    const float eps,
    const unsigned long long count)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const unsigned long long row = gid / width;
    const unsigned int column = (unsigned int)(gid - row * width);
    const unsigned long long hidden_row = row * copies * width;
    const unsigned long long mix_row = row * copies;
    float sum = 0.0f;
    for (unsigned int copy = 0; copy < copies; ++copy) {
        const float logit = mixes[mix_row + copy] * scale[0] + base[copy];
        const float coefficient = 1.0f / (1.0f + expf(-logit)) + eps;
        sum += coefficient * hidden[hidden_row + (unsigned long long)copy * width + column];
    }
    output[gid] = sum;
}
