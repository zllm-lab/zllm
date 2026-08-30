extern "C" __global__ void scale_tensor_f32(const float* input, float* output, unsigned int elements, float scale)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index] * scale;
}

extern "C" __global__ void tanh_tensor_f32(const float* input, float* output, unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = tanhf(input[index]);
}

extern "C" __global__ void timestep_embedding_f32(
    const float* timesteps,
    float* output,
    unsigned int batch,
    unsigned int dim)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = dim / 2;
    if (index >= batch * half) return;
    const unsigned int row = index / half;
    const unsigned int column = index - row * half;
    const float frequency = expf(-logf(10000.0f) * (float)column / (float)half);
    const float value = timesteps[row] * frequency;
    output[(unsigned long long)row * dim + column] = cosf(value);
    output[(unsigned long long)row * dim + half + column] = sinf(value);
}

extern "C" __global__ void group_norm_f32(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    unsigned int channels,
    unsigned int spatial,
    unsigned int groups,
    float eps)
{
    const unsigned int group = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (group >= groups) return;
    const unsigned int channels_per_group = channels / groups;
    const unsigned int group_elements = channels_per_group * spatial;
    const unsigned long long begin = (unsigned long long)group * group_elements;
    float sum = 0.0f;
    float square = 0.0f;
    for (unsigned int index = lane; index < group_elements; index += blockDim.x) {
        const float value = input[begin + index];
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
    const float mean = partial_sum[0] / (float)group_elements;
    const float variance = fmaxf(partial_square[0] / (float)group_elements - mean * mean, 0.0f);
    const float inverse = rsqrtf(variance + eps);
    for (unsigned int index = lane; index < group_elements; index += blockDim.x) {
        const unsigned int channel = group * channels_per_group + index / spatial;
        output[begin + index] = (input[begin + index] - mean) * inverse * scale[channel] + bias[channel];
    }
}

extern "C" __global__ void group_norm_time_isolated_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    unsigned int channels,
    unsigned int time,
    unsigned int spatial,
    unsigned int groups,
    float epsilon
) {
    const unsigned int task = blockIdx.x;
    const unsigned int t = task / groups;
    const unsigned int group = task - t * groups;
    if (t >= time || group >= groups || groups == 0 || channels % groups != 0) return;

    const unsigned int channels_per_group = channels / groups;
    const unsigned int count = channels_per_group * spatial;
    __shared__ float reduction[256];

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        const unsigned int local_channel = i / spatial;
        const unsigned int position = i - local_channel * spatial;
        const unsigned int channel = group * channels_per_group + local_channel;
        sum += input[channel * time * spatial + t * spatial + position];
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        __syncthreads();
    }
    const float mean = reduction[0] / (float)count;

    float square_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        const unsigned int local_channel = i / spatial;
        const unsigned int position = i - local_channel * spatial;
        const unsigned int channel = group * channels_per_group + local_channel;
        const float value = input[channel * time * spatial + t * spatial + position] - mean;
        square_sum += value * value;
    }
    reduction[threadIdx.x] = square_sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        __syncthreads();
    }
    const float inv_std = rsqrtf(reduction[0] / (float)count + epsilon);

    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        const unsigned int local_channel = i / spatial;
        const unsigned int position = i - local_channel * spatial;
        const unsigned int channel = group * channels_per_group + local_channel;
        const unsigned int index = channel * time * spatial + t * spatial + position;
        output[index] = (input[index] - mean) * inv_std * weight[channel] + bias[channel];
    }
}

extern "C" __global__ void snake_f32(
    const float* input,
    const float* alpha,
    float* output,
    unsigned int channels,
    unsigned int columns,
    unsigned int elements
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int channel = (index / columns) % channels;
    const float a = fmaxf(fabsf(alpha[channel]), 1.0e-6f);
    const float sine = sinf(a * input[index]);
    output[index] = input[index] + sine * sine / a;
}

extern "C" __global__ void gelu_tanh_f32(
    const float* input,
    float* output,
    unsigned int elements
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float x = input[index];
    output[index] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void channels_to_time_f32(
    const float* input,
    float* output,
    unsigned int batch,
    unsigned int channels,
    unsigned int time,
    unsigned int elements
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int channel = index % channels;
    const unsigned int temporal = index / channels;
    const unsigned int step = temporal % time;
    const unsigned int batch_index = temporal / time;
    output[index] = input[((unsigned long long)batch_index * channels + channel) * time + step];
}

extern "C" __global__ void causal_attention_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    unsigned int batch,
    unsigned int time,
    unsigned int heads,
    unsigned int head_dim,
    float scale
) {
    const unsigned int task = blockIdx.x;
    const unsigned int head = task % heads;
    const unsigned int temporal = task / heads;
    const unsigned int query = temporal % time;
    const unsigned int batch_index = temporal / time;
    if (batch_index >= batch || query >= time || head >= heads || head_dim > 256) return;
    const unsigned long long batch_offset = (unsigned long long)batch_index * time * heads * head_dim;
    __shared__ float reduction[256];
    __shared__ float control[4];
    const unsigned int lane = threadIdx.x;
    float accumulated = 0.0f;
    if (lane == 0) {
        control[0] = -3.402823466e+38f;
        control[1] = 0.0f;
    }
    __syncthreads();
    for (unsigned int key_index = 0; key_index <= query; ++key_index) {
        float partial = 0.0f;
        if (lane < head_dim) {
            const unsigned long long q_index = batch_offset + query * heads * head_dim + head * head_dim + lane;
            const unsigned long long k_index = batch_offset + key_index * heads * head_dim + head * head_dim + lane;
            partial = q[q_index] * k[k_index];
        }
        reduction[lane] = partial;
        __syncthreads();
        for (unsigned int offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) reduction[lane] += reduction[lane + offset];
            __syncthreads();
        }
        if (lane == 0) {
            const float score = reduction[0] * scale;
            const float maximum = fmaxf(control[0], score);
            const float old_scale = expf(control[0] - maximum);
            const float key_scale = expf(score - maximum);
            control[0] = maximum;
            control[1] = control[1] * old_scale + key_scale;
            control[2] = old_scale;
            control[3] = key_scale;
        }
        __syncthreads();
        if (lane < head_dim) {
            const unsigned long long v_index = batch_offset + key_index * heads * head_dim + head * head_dim + lane;
            accumulated = accumulated * control[2] + v[v_index] * control[3];
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        const unsigned long long out_index = batch_offset + query * heads * head_dim + head * head_dim + lane;
        output[out_index] = accumulated / control[1];
    }
}
