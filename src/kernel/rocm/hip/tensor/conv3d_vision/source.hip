extern "C" __global__ void pack_conv3d_single_frame_weight_bf16(
    const float* weight,
    unsigned short* packed,
    unsigned int in_channels,
    unsigned int out_channels,
    unsigned int kernel_depth,
    unsigned int kernel_height,
    unsigned int kernel_width,
    unsigned int temporal_filter,
    unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int output_channel = index % out_channels;
    const unsigned int k = index / out_channels;
    const unsigned int kernel_spatial = kernel_height * kernel_width;
    const unsigned int input_channel = k / kernel_spatial;
    const unsigned int kernel_offset = k - input_channel * kernel_spatial;
    const unsigned int filter_y = kernel_offset / kernel_width;
    const unsigned int filter_x = kernel_offset - filter_y * kernel_width;
    const unsigned long long source =
        (((unsigned long long)output_channel * in_channels + input_channel) * kernel_depth + temporal_filter)
        * kernel_height * kernel_width + filter_y * kernel_width + filter_x;
    packed[index] = zllm_f32_to_bf16(weight[source]);
}

// Ref2VA 的参考图时间深度恒为 1。每个 block 让 8 个 wave 共享 16 个空间位置，
// 各 wave 计算 16 个输出通道；隐式 im2col 只存在于 LDS，不制造全图展开缓冲。
extern "C" __global__ void conv3d_single_frame_wmma_f32(
    const float* input,
    const unsigned short* weight,
    const float* bias,
    float* output,
    unsigned int in_channels,
    unsigned int out_channels,
    unsigned int in_depth,
    unsigned int in_height,
    unsigned int in_width,
    unsigned int kernel_depth,
    unsigned int kernel_height,
    unsigned int kernel_width,
    unsigned int stride_depth,
    unsigned int stride_height,
    unsigned int stride_width,
    unsigned int has_bias,
    unsigned int pad_depth,
    unsigned int pad_height,
    unsigned int pad_width,
    unsigned int pad_after_height,
    unsigned int pad_after_width,
    unsigned int causal,
    unsigned int reflect_spatial,
    unsigned int out_depth,
    unsigned int out_height,
    unsigned int out_width)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;

    __shared__ unsigned short shared_input[16 * 16];
    __shared__ unsigned short shared_weight[8][16 * 16];
    __shared__ float shared_output[8][16 * 16];

    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int spatial = out_height * out_width;
    const unsigned int position_tiles = (spatial + 15) / 16;
    const unsigned int position_tile = blockIdx.x % position_tiles;
    const unsigned int output_group = blockIdx.x / position_tiles;
    const unsigned int output_channel_base = (output_group * 8 + wave) * 16;
    const unsigned int position_base = position_tile * 16;
    const unsigned int effective_k = in_channels * kernel_height * kernel_width;

    CFragment accumulator;
    fill_fragment(accumulator, 0.0f);
    for (unsigned int k_base = 0; k_base < effective_k; k_base += 16) {
        const unsigned int input_index = threadIdx.x;
        const unsigned int local_position = input_index / 16;
        const unsigned int local_k = input_index - local_position * 16;
        const unsigned int position = position_base + local_position;
        const unsigned int k = k_base + local_k;
        float input_value = 0.0f;
        if (position < spatial && k < effective_k) {
            const unsigned int output_y = position / out_width;
            const unsigned int output_x = position - output_y * out_width;
            const unsigned int kernel_spatial = kernel_height * kernel_width;
            const unsigned int input_channel = k / kernel_spatial;
            const unsigned int kernel_offset = k - input_channel * kernel_spatial;
            const unsigned int filter_y = kernel_offset / kernel_width;
            const unsigned int filter_x = kernel_offset - filter_y * kernel_width;
            int input_y = (int)(output_y * stride_height + filter_y) - (int)pad_height;
            int input_x = (int)(output_x * stride_width + filter_x) - (int)pad_width;
            if (reflect_spatial) {
                if (in_height > 1) {
                    if (input_y < 0) input_y = -input_y;
                    if (input_y >= (int)in_height) input_y = 2 * (int)in_height - input_y - 2;
                } else {
                    input_y = 0;
                }
                if (in_width > 1) {
                    if (input_x < 0) input_x = -input_x;
                    if (input_x >= (int)in_width) input_x = 2 * (int)in_width - input_x - 2;
                } else {
                    input_x = 0;
                }
            }
            if (input_y >= 0 && input_y < (int)in_height && input_x >= 0 && input_x < (int)in_width) {
                input_value = input[((unsigned long long)input_channel * in_depth) * in_height * in_width
                    + (unsigned int)input_y * in_width + (unsigned int)input_x];
            }
        }
        shared_input[input_index] = zllm_f32_to_bf16(input_value);

        for (unsigned int weight_index = lane; weight_index < 16 * 16; weight_index += warpSize) {
            const unsigned int local_k_row = weight_index / 16;
            const unsigned int local_output_channel = weight_index - local_k_row * 16;
            const unsigned int weight_k = k_base + local_k_row;
            const unsigned int output_channel = output_channel_base + local_output_channel;
            shared_weight[wave][weight_index] =
                weight_k < effective_k && output_channel < out_channels
                    ? weight[(unsigned long long)weight_k * out_channels + output_channel]
                    : 0;
        }
        __syncthreads();

        AFragment input_fragment;
        BFragment weight_fragment;
        load_matrix_sync(input_fragment, (const bfloat16_t*)shared_input, 16);
        load_matrix_sync(weight_fragment, (const bfloat16_t*)shared_weight[wave], 16);
        mma_sync(accumulator, input_fragment, weight_fragment, accumulator);
        __syncthreads();
    }

    store_matrix_sync(shared_output[wave], accumulator, 16, mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < 16 * 16; index += warpSize) {
        const unsigned int local_position = index / 16;
        const unsigned int local_output_channel = index - local_position * 16;
        const unsigned int position = position_base + local_position;
        const unsigned int output_channel = output_channel_base + local_output_channel;
        if (position < spatial && output_channel < out_channels) {
            output[(unsigned long long)output_channel * spatial + position] =
                shared_output[wave][index] + (has_bias ? bias[output_channel] : 0.0f);
        }
    }
}

extern "C" __global__ void conv3d_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    unsigned int in_channels,
    unsigned int out_channels,
    unsigned int in_depth,
    unsigned int in_height,
    unsigned int in_width,
    unsigned int kernel_depth,
    unsigned int kernel_height,
    unsigned int kernel_width,
    unsigned int stride_depth,
    unsigned int stride_height,
    unsigned int stride_width,
    unsigned int has_bias,
    unsigned int pad_depth,
    unsigned int pad_height,
    unsigned int pad_width,
    unsigned int pad_after_height,
    unsigned int pad_after_width,
    unsigned int causal,
    unsigned int reflect_spatial,
    unsigned int out_depth,
    unsigned int out_height,
    unsigned int out_width)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = out_channels * out_depth * out_height * out_width;
    if (index >= total) return;
    unsigned int remainder = index;
    const unsigned int ow = remainder % out_width;
    remainder /= out_width;
    const unsigned int oh = remainder % out_height;
    remainder /= out_height;
    const unsigned int od = remainder % out_depth;
    const unsigned int oc = remainder / out_depth;
    float value = has_bias ? bias[oc] : 0.0f;
    for (unsigned int ic = 0; ic < in_channels; ++ic) {
        for (unsigned int fd = 0; fd < kernel_depth; ++fd) {
            const int temporal_padding = (int)pad_depth;
            const int id = (int)(od * stride_depth + fd) - temporal_padding;
            if (id < 0 || id >= (int)in_depth) continue;
            for (unsigned int fh = 0; fh < kernel_height; ++fh) {
                int ih = (int)(oh * stride_height + fh) - (int)pad_height;
                if (reflect_spatial) {
                    if (in_height > 1) {
                        if (ih < 0) ih = -ih;
                        if (ih >= (int)in_height) ih = 2 * (int)in_height - ih - 2;
                    } else {
                        ih = 0;
                    }
                }
                if (ih < 0 || ih >= (int)in_height) continue;
                for (unsigned int fw = 0; fw < kernel_width; ++fw) {
                    int iw = (int)(ow * stride_width + fw) - (int)pad_width;
                    if (reflect_spatial) {
                        if (in_width > 1) {
                            if (iw < 0) iw = -iw;
                            if (iw >= (int)in_width) iw = 2 * (int)in_width - iw - 2;
                        } else {
                            iw = 0;
                        }
                    }
                    if (iw < 0 || iw >= (int)in_width) continue;
                    const unsigned long long input_index =
                        ((unsigned long long)ic * in_depth + (unsigned int)id) * in_height * in_width
                        + (unsigned int)ih * in_width + (unsigned int)iw;
                    const unsigned long long weight_index =
                        (((unsigned long long)oc * in_channels + ic) * kernel_depth + fd)
                        * kernel_height * kernel_width + fh * kernel_width + fw;
                    value += input[input_index] * weight[weight_index];
                }
            }
        }
    }
    output[index] = value;
}

extern "C" __global__ void vision_rope_f32(
    const float* query,
    const float* key,
    const float* cosine,
    const float* sine,
    float* query_output,
    float* key_output,
    unsigned int heads,
    unsigned int head_dim,
    unsigned int rotary_dim,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned int dimension = index % head_dim;
    if (dimension >= rotary_dim) {
        query_output[index] = query[index];
        key_output[index] = key[index];
        return;
    }
    const unsigned int row = index / (heads * head_dim);
    const unsigned int half_dim = rotary_dim / 2;
    const unsigned int paired = dimension < half_dim ? dimension + half_dim : dimension - half_dim;
    const unsigned long long head_base = (unsigned long long)(index / head_dim) * head_dim;
    const float rotated_query = dimension < half_dim ? -query[head_base + paired] : query[head_base + paired];
    const float rotated_key = dimension < half_dim ? -key[head_base + paired] : key[head_base + paired];
    const unsigned long long rope_index = (unsigned long long)row * rotary_dim + dimension;
    query_output[index] = query[index] * cosine[rope_index] + rotated_query * sine[rope_index];
    key_output[index] = key[index] * cosine[rope_index] + rotated_key * sine[rope_index];
}

extern "C" __global__ void scatter_rows_f32(
    const float* source,
    float* destination,
    unsigned int start_row,
    unsigned int columns,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) destination[(unsigned long long)start_row * columns + index] = source[index];
}

extern "C" __global__ void vision_attention_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* cosine,
    const float* sine,
    float* output,
    unsigned int rows,
    unsigned int heads,
    unsigned int head_dim,
    unsigned int rotary_dim
) {
    const unsigned int tile = blockIdx.x / heads;
    const unsigned int head = blockIdx.x - tile * heads;
    const unsigned int first_query = tile * 4;
    const unsigned int lane = threadIdx.x;
    if (first_query >= rows || head >= heads || head_dim > 256) return;

    __shared__ float reduction[4][256];
    __shared__ float control[4][4];
    float accumulated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float rotated_query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < head_dim) {
        for (unsigned int q = 0; q < 4; ++q) {
            const unsigned int query_row = first_query + q;
            if (query_row >= rows) continue;
            const unsigned long long query_base = ((unsigned long long)query_row * heads + head) * head_dim;
            rotated_query[q] = query[query_base + lane];
            if (lane < rotary_dim) {
                const unsigned int half_dim = rotary_dim / 2;
                const unsigned int paired = lane < half_dim ? lane + half_dim : lane - half_dim;
                const float paired_value = lane < half_dim ? -query[query_base + paired] : query[query_base + paired];
                const unsigned long long rope_index = (unsigned long long)query_row * rotary_dim + lane;
                rotated_query[q] = rotated_query[q] * cosine[rope_index] + paired_value * sine[rope_index];
            }
        }
    }
    if (lane < 4) {
        control[lane][0] = -3.402823466e+38f;
        control[lane][1] = 0.0f;
    }
    __syncthreads();

    for (unsigned int key_row = 0; key_row < rows; ++key_row) {
        float rotated_key = 0.0f;
        float value_lane = 0.0f;
        if (lane < head_dim) {
            const unsigned long long key_base = ((unsigned long long)key_row * heads + head) * head_dim;
            rotated_key = key[key_base + lane];
            value_lane = value[key_base + lane];
            if (lane < rotary_dim) {
                const unsigned int half_dim = rotary_dim / 2;
                const unsigned int paired = lane < half_dim ? lane + half_dim : lane - half_dim;
                const float paired_value = lane < half_dim ? -key[key_base + paired] : key[key_base + paired];
                const unsigned long long rope_index = (unsigned long long)key_row * rotary_dim + lane;
                rotated_key = rotated_key * cosine[rope_index] + paired_value * sine[rope_index];
            }
        }

        for (unsigned int q = 0; q < 4; ++q) {
            const unsigned int query_row = first_query + q;
            float partial = 0.0f;
            if (query_row < rows && lane < head_dim) {
                partial = rotated_query[q] * rotated_key;
            }
            reduction[q][lane] = partial;
        }
        __syncthreads();
        for (unsigned int offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) {
                for (unsigned int q = 0; q < 4; ++q) reduction[q][lane] += reduction[q][lane + offset];
            }
            __syncthreads();
        }
        if (lane < 4 && first_query + lane < rows) {
            const float score = reduction[lane][0] * rsqrtf((float)head_dim);
            const float maximum = fmaxf(control[lane][0], score);
            const float old_scale = expf(control[lane][0] - maximum);
            const float key_scale = expf(score - maximum);
            control[lane][0] = maximum;
            control[lane][1] = control[lane][1] * old_scale + key_scale;
            control[lane][2] = old_scale;
            control[lane][3] = key_scale;
        }
        __syncthreads();
        for (unsigned int q = 0; q < 4; ++q) {
            if (first_query + q < rows && lane < head_dim) accumulated[q] = accumulated[q] * control[q][2] + value_lane * control[q][3];
        }
        __syncthreads();
    }

    if (lane < head_dim) {
        for (unsigned int q = 0; q < 4; ++q) {
            const unsigned int query_row = first_query + q;
            if (query_row < rows) {
                const unsigned long long output_index = ((unsigned long long)query_row * heads + head) * head_dim + lane;
                output[output_index] = accumulated[q] / control[q][1];
            }
        }
    }
}

extern "C" __global__ void pixel_shuffle_f32(
    const float* input,
    float* output,
    unsigned int channels,
    unsigned int height,
    unsigned int width,
    unsigned int upscale)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int out_height = height * upscale;
    const unsigned int out_width = width * upscale;
    const unsigned int total = channels * out_height * out_width;
    if (index >= total) return;
    unsigned int remainder = index;
    const unsigned int ow = remainder % out_width;
    remainder /= out_width;
    const unsigned int oh = remainder % out_height;
    const unsigned int channel = remainder / out_height;
    const unsigned int input_channel = channel * upscale * upscale + (oh % upscale) * upscale + ow % upscale;
    output[index] = input[((unsigned long long)input_channel * height + oh / upscale) * width + ow / upscale];
}
