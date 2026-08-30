extern "C" __global__ void audio_unpack_affine_f32(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    unsigned int batch,
    unsigned int time,
    unsigned int channels,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int step = index % time;
    const unsigned int row = index / time;
    const unsigned int channel = row % channels;
    const unsigned int item = row / channels;
    output[index] = input[((unsigned long long)item * time + step) * channels + channel] * scale[channel] + bias[channel];
}

extern "C" __global__ void audio_weight_norm_f32(
    const float* weight_g,
    const float* weight_v,
    float* output,
    unsigned int rows,
    unsigned int columns)
{
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows) return;
    const unsigned long long base = (unsigned long long)row * columns;
    float square = 0.0f;
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        const float value = weight_v[base + column];
        square += value * value;
    }
    __shared__ float partial[256];
    partial[lane] = square;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float factor = weight_g[row] * rsqrtf(fmaxf(partial[0], 1.0e-24f));
    for (unsigned int column = lane; column < columns; column += blockDim.x) {
        output[base + column] = weight_v[base + column] * factor;
    }
}

extern "C" __global__ void audio_conv1d_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    unsigned int batch,
    unsigned int input_channels,
    unsigned int output_channels,
    unsigned int input_length,
    unsigned int output_length,
    unsigned int kernel,
    unsigned int dilation,
    unsigned int padding,
    unsigned int stride,
    unsigned int has_bias,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int step = index % output_length;
    const unsigned int row = index / output_length;
    const unsigned int output_channel = row % output_channels;
    const unsigned int item = row / output_channels;
    float value = has_bias ? bias[output_channel] : 0.0f;
    for (unsigned int input_channel = 0; input_channel < input_channels; ++input_channel) {
        for (unsigned int tap = 0; tap < kernel; ++tap) {
            const int source = (int)(step * stride) + (int)(tap * dilation) - (int)padding;
            if (source >= 0 && source < (int)input_length) {
                value += input[((unsigned long long)item * input_channels + input_channel) * input_length + (unsigned int)source]
                    * weight[((unsigned long long)output_channel * input_channels + input_channel) * kernel + tap];
            }
        }
    }
    output[index] = value;
}
extern "C" __global__ void audio_conv_transpose1d_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    unsigned int batch,
    unsigned int input_channels,
    unsigned int output_channels,
    unsigned int input_length,
    unsigned int output_length,
    unsigned int kernel,
    unsigned int stride,
    unsigned int padding,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int step = index % output_length;
    const unsigned int row = index / output_length;
    const unsigned int output_channel = row % output_channels;
    const unsigned int item = row / output_channels;
    float value = bias[output_channel];
    for (unsigned int input_channel = 0; input_channel < input_channels; ++input_channel) {
        for (unsigned int tap = 0; tap < kernel; ++tap) {
            const unsigned int shifted = step + padding;
            if (shifted >= tap && (shifted - tap) % stride == 0) {
                const unsigned int source = (shifted - tap) / stride;
                if (source < input_length) {
                    value += input[((unsigned long long)item * input_channels + input_channel) * input_length + source]
                        * weight[((unsigned long long)input_channel * output_channels + output_channel) * kernel + tap];
                }
            }
        }
    }
    output[index] = value;
}

extern "C" __global__ void alias_upsample2_f32(
    const float* input,
    const float* filter,
    float* output,
    unsigned int batch,
    unsigned int channels,
    unsigned int length,
    unsigned int filter_kernel,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int step = index % (length * 2);
    const unsigned int row = index / (length * 2);
    const unsigned int pad = filter_kernel / 2 - 1;
    const unsigned int crop = pad * 2 + (filter_kernel - 2) / 2;
    const unsigned int raw = step + crop;
    float value = 0.0f;
    for (unsigned int tap = 0; tap < filter_kernel; ++tap) {
        if (raw >= tap && (raw - tap) % 2 == 0) {
            const unsigned int padded = (raw - tap) / 2;
            if (padded < length + 2 * pad) {
                const unsigned int shifted = padded - pad;
                const unsigned int source = padded < pad ? 0 : (shifted < length ? shifted : length - 1);
                value += input[(unsigned long long)row * length + source] * filter[tap] * 2.0f;
            }
        }
    }
    output[index] = value;
}

extern "C" __global__ void snake_beta_f32(
    const float* input,
    const float* alpha,
    const float* beta,
    float* output,
    unsigned int channels,
    unsigned int length,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int channel = (index / length) % channels;
    const float a = expf(alpha[channel]);
    const float b = fmaxf(expf(beta[channel]), 1.0e-9f);
    const float value = input[index];
    const float periodic = sinf(value * a);
    output[index] = value + periodic * periodic / b;
}

extern "C" __global__ void alias_downsample2_f32(
    const float* input,
    const float* filter,
    float* output,
    unsigned int length,
    unsigned int filter_kernel,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int step = index % length;
    const unsigned int row = index / length;
    const unsigned int left = filter_kernel / 2 - (filter_kernel % 2 == 0 ? 1 : 0);
    float value = 0.0f;
    for (unsigned int tap = 0; tap < filter_kernel; ++tap) {
        const int source = (int)(step * 2 + tap) - (int)left;
        const unsigned int high_length = length * 2;
        const unsigned int clamped = source < 0 ? 0 : ((unsigned int)source < high_length ? (unsigned int)source : high_length - 1);
        value += input[(unsigned long long)row * length * 2 + clamped] * filter[tap];
    }
    output[index] = value;
}
