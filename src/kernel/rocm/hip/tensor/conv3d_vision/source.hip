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

// 把逻辑时间前缀映射到紧凑历史；首片以当前首帧代替历史，完全不物化拼接。
template <bool SplitTemporal>
__device__ __forceinline__ float zllm_conv3d_input(
    const float* input, const float* history, unsigned int depth, unsigned int head,
    unsigned int channel, unsigned int time, unsigned long long plane, unsigned long long position)
{
    if constexpr (SplitTemporal) {
        const unsigned int current_depth = depth - head;
        if (time < head) {
            return history ? history[((unsigned long long)channel * head + time) * plane + position]
                           : input[(unsigned long long)channel * current_depth * plane + position];
        }
        return input[((unsigned long long)channel * current_depth + time - head) * plane + position];
    }
    return input[((unsigned long long)channel * depth + time) * plane + position];
}

// 多帧卷积以LDS保存当前K片段，避免完整im2col以及每个输出通道重复读取全局输入。
// M/N/K分块只决定调度，模型尺寸、padding与stride全部由参数传入。
template <bool SplitTemporal>
__device__ __forceinline__ void zllm_conv3d_tiled_f32(
    const float* input, const float* weight, const float* bias, float* output,
    unsigned int in_channels, unsigned int out_channels,
    unsigned int in_depth, unsigned int in_height, unsigned int in_width,
    unsigned int kernel_depth, unsigned int kernel_height, unsigned int kernel_width,
    unsigned int stride_depth, unsigned int stride_height, unsigned int stride_width,
    unsigned int has_bias, unsigned int pad_depth, unsigned int pad_height, unsigned int pad_width,
    unsigned int pad_after_height, unsigned int pad_after_width,
    unsigned int causal, unsigned int reflect_spatial,
    unsigned int out_depth, unsigned int out_height, unsigned int out_width,
    const float* history, unsigned int head_frames,
    unsigned int channel_tile = 64, unsigned int channel_offset = 0)
{
    __shared__ float a[16][16];
    __shared__ float b[64][16];
    const unsigned int spatial = out_depth * out_height * out_width;
    const unsigned int position_tiles = (spatial + 15) / 16;
    const unsigned int position_base = (blockIdx.x % position_tiles) * 16;
    const unsigned int channel_base = (blockIdx.x / position_tiles) * channel_tile + channel_offset;
    const unsigned int m = threadIdx.x % 16;
    const unsigned int n = threadIdx.x / 16;
    const unsigned int taps = kernel_depth * kernel_height * kernel_width;
    const unsigned int k_count = in_channels * taps;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int k_base = 0; k_base < k_count; k_base += 16) {
        // 256线程各载入一个A元素；每个输出通道复用同一组16个空间位置。
        const unsigned int load_m = threadIdx.x / 16;
        const unsigned int load_k = threadIdx.x % 16;
        const unsigned int position = position_base + load_m;
        const unsigned int k = k_base + load_k;
        float value = 0.0f;
        if (position < spatial && k < k_count) {
            const unsigned int ic = k / taps;
            const unsigned int tap = k - ic * taps;
            const unsigned int fd = tap / (kernel_height * kernel_width);
            const unsigned int fh = (tap / kernel_width) % kernel_height;
            const unsigned int fw = tap % kernel_width;
            const unsigned int od = position / (out_height * out_width);
            const unsigned int oh = (position / out_width) % out_height;
            const unsigned int ow = position % out_width;
            const int id = (int)(od * stride_depth + fd) - (int)pad_depth;
            int ih = (int)(oh * stride_height + fh) - (int)pad_height;
            int iw = (int)(ow * stride_width + fw) - (int)pad_width;
            if (reflect_spatial) {
                if (in_height > 1) {
                    if (ih < 0) ih = -ih;
                    if (ih >= (int)in_height) ih = 2 * (int)in_height - ih - 2;
                } else ih = 0;
                if (in_width > 1) {
                    if (iw < 0) iw = -iw;
                    if (iw >= (int)in_width) iw = 2 * (int)in_width - iw - 2;
                } else iw = 0;
            }
            if (id >= 0 && id < (int)in_depth && ih >= 0 && ih < (int)in_height && iw >= 0 && iw < (int)in_width) {
                value = zllm_conv3d_input<SplitTemporal>(input, history, in_depth, head_frames, ic, (unsigned int)id,
                    (unsigned long long)in_height * in_width, (unsigned long long)(unsigned int)ih * in_width + (unsigned int)iw);
            }
        }
        a[load_m][load_k] = value;
        for (unsigned int i = threadIdx.x; i < 64 * 16; i += 256) {
            const unsigned int local_n = i / 16;
            const unsigned int local_k = i % 16;
            const unsigned int oc = channel_base + local_n;
            const unsigned int rk = k_base + local_k;
            b[local_n][local_k] = oc < out_channels && rk < k_count
                ? weight[(unsigned long long)oc * k_count + rk] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int kk = 0; kk < 16; ++kk) {
            const float av = a[m][kk];
            #pragma unroll
            for (unsigned int j = 0; j < 4; ++j) sums[j] = fmaf(av, b[n + j * 16][kk], sums[j]);
        }
        __syncthreads();
    }
    const unsigned int position = position_base + m;
    #pragma unroll
    for (unsigned int j = 0; j < 4; ++j) {
        const unsigned int oc = channel_base + n + j * 16;
        if (position < spatial && oc < out_channels)
            output[(unsigned long long)oc * spatial + position] = sums[j] + (has_bias ? bias[oc] : 0.0f);
    }
}
extern "C" __global__ void conv3d_tiled_f32(
    const float* input, const float* weight, const float* bias, float* output,
    unsigned int in_channels, unsigned int out_channels,
    unsigned int in_depth, unsigned int in_height, unsigned int in_width,
    unsigned int kernel_depth, unsigned int kernel_height, unsigned int kernel_width,
    unsigned int stride_depth, unsigned int stride_height, unsigned int stride_width,
    unsigned int has_bias, unsigned int pad_depth, unsigned int pad_height, unsigned int pad_width,
    unsigned int pad_after_height, unsigned int pad_after_width,
    unsigned int causal, unsigned int reflect_spatial,
    unsigned int out_depth, unsigned int out_height, unsigned int out_width)
{
    zllm_conv3d_tiled_f32<false>(input, weight, bias, output, in_channels, out_channels, in_depth, in_height, in_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width, has_bias, pad_depth, pad_height, pad_width, pad_after_height, pad_after_width, causal, reflect_spatial, out_depth, out_height, out_width, nullptr, 0);
}

// 只在已验证的wave32架构导出；其他设备的host调度保留F32入口。
#if !defined(__HIP_DEVICE_COMPILE__) || (defined(__gfx1100__) && ROCWMMA_WAVE32_MODE)
extern "C" __global__ __launch_bounds__(256) void conv3d_history_wmma_f16_acc_f32(
    const float* input, const float* weight, const float* bias, float* output,
    unsigned int in_channels, unsigned int out_channels,
    unsigned int in_depth, unsigned int in_height, unsigned int in_width,
    unsigned int kernel_depth, unsigned int kernel_height, unsigned int kernel_width,
    unsigned int stride_depth, unsigned int stride_height, unsigned int stride_width,
    unsigned int has_bias, unsigned int pad_depth, unsigned int pad_height, unsigned int pad_width,
    unsigned int pad_after_height, unsigned int pad_after_width,
    unsigned int causal, unsigned int reflect_spatial,
    unsigned int out_depth, unsigned int out_height, unsigned int out_width,
    const float* history, unsigned int head_frames)
{
    using namespace rocwmma;
    using A = fragment<matrix_a, 16, 16, 16, float16_t, row_major>;
    using B = fragment<matrix_b, 16, 16, 16, float16_t, col_major>;
    using C = fragment<accumulator, 16, 16, 16, float>;
    __shared__ __align__(16) _Float16 a[256];
    __shared__ __align__(16) _Float16 b[8][256];
    __shared__ __align__(16) float result[8][256];
    __shared__ unsigned int needs_f32;
    if (threadIdx.x == 0) needs_f32 = 0;
    __syncthreads();
    const unsigned int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
    const unsigned long long spatial = (unsigned long long)out_depth * out_height * out_width;
    const unsigned long long position_tiles = (spatial + 15) / 16;
    const unsigned long long position_base = (blockIdx.x % position_tiles) * 16;
    const unsigned int channel_base = (blockIdx.x / position_tiles * 8 + wave) * 16;
    const unsigned int taps = kernel_depth * kernel_height * kernel_width;
    const unsigned int k_count = in_channels * taps;
    C sum;
    fill_fragment(sum, 0.0f);
    for (unsigned int begin = 0; begin < k_count; begin += 16) {
        const unsigned int k = begin + threadIdx.x % 16;
        const unsigned long long position = position_base + threadIdx.x / 16;
        float value = 0.0f;
        if (position < spatial && k < k_count) {
            const unsigned int channel = k / taps, tap = k % taps;
            const unsigned int ft = tap / (kernel_height * kernel_width);
            const unsigned int fy = (tap / kernel_width) % kernel_height, fx = tap % kernel_width;
            const unsigned int ot = position / ((unsigned long long)out_height * out_width);
            const unsigned int oy = (position / out_width) % out_height, ox = position % out_width;
            const long long time = (long long)ot * stride_depth + ft - pad_depth;
            const long long y = (long long)oy * stride_height + fy - pad_height;
            const long long x = (long long)ox * stride_width + fx - pad_width;
            if (time >= 0 && time < in_depth && y >= 0 && y < in_height && x >= 0 && x < in_width)
                value = zllm_conv3d_input<true>(input, history, in_depth, head_frames, channel,
                    (unsigned int)time, (unsigned long long)in_height * in_width, (unsigned long long)y * in_width + x);
        }
        // 只转换本次K片段；原权重保持F32常驻，不新增全权重或im2col副本。
        // 范围外或非有限输入必须保留F32语义，例如0*65536不能变成0*Inf。
        if (!isfinite(value) || fabsf(value) > 65504.0f) atomicExch(&needs_f32, 1u);
        a[threadIdx.x] = (_Float16)value;
        for (unsigned int index = lane; index < 256; index += warpSize) {
            const unsigned int channel = channel_base + index / 16, wk = begin + index % 16;
            b[wave][index] = (_Float16)(channel < out_channels && wk < k_count ? weight[(unsigned long long)channel * k_count + wk] : 0.0f);
        }
        __syncthreads();
        A aa; B bb;
        load_matrix_sync(aa, reinterpret_cast<const float16_t*>(a), 16);
        load_matrix_sync(bb, reinterpret_cast<const float16_t*>(b[wave]), 16);
        mma_sync(sum, aa, bb, sum);
        __syncthreads();
    }
    if (needs_f32) {
        // 每个WMMA block覆盖128输出通道，按两个原N64块重算；默认F32路径
        // 的索引和累加顺序不变。所有线程统一分支，沿用原helper的LDS屏障。
        zllm_conv3d_tiled_f32<true>(input, weight, bias, output, in_channels, out_channels, in_depth, in_height, in_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width, has_bias, pad_depth, pad_height, pad_width, pad_after_height, pad_after_width, causal, reflect_spatial, out_depth, out_height, out_width, history, head_frames, 128, 0);
        zllm_conv3d_tiled_f32<true>(input, weight, bias, output, in_channels, out_channels, in_depth, in_height, in_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width, has_bias, pad_depth, pad_height, pad_width, pad_after_height, pad_after_width, causal, reflect_spatial, out_depth, out_height, out_width, history, head_frames, 128, 64);
        return;
    }
    store_matrix_sync(result[wave], sum, 16, mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < 256; index += warpSize) {
        const unsigned long long position = position_base + index / 16;
        const unsigned int channel = channel_base + index % 16;
        if (position < spatial && channel < out_channels)
            output[(unsigned long long)channel * spatial + position] = result[wave][index] + (has_bias ? bias[channel] : 0.0f);
    }
}
#endif

extern "C" __global__ void conv3d_history_tiled_f32(
    const float* input, const float* weight, const float* bias, float* output,
    unsigned int in_channels, unsigned int out_channels,
    unsigned int in_depth, unsigned int in_height, unsigned int in_width,
    unsigned int kernel_depth, unsigned int kernel_height, unsigned int kernel_width,
    unsigned int stride_depth, unsigned int stride_height, unsigned int stride_width,
    unsigned int has_bias, unsigned int pad_depth, unsigned int pad_height, unsigned int pad_width,
    unsigned int pad_after_height, unsigned int pad_after_width,
    unsigned int causal, unsigned int reflect_spatial,
    unsigned int out_depth, unsigned int out_height, unsigned int out_width,
    const float* history, unsigned int head_frames)
{
    zllm_conv3d_tiled_f32<true>(input, weight, bias, output, in_channels, out_channels, in_depth, in_height, in_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width, has_bias, pad_depth, pad_height, pad_width, pad_after_height, pad_after_width, causal, reflect_spatial, out_depth, out_height, out_width, history, head_frames);
}

template <bool SplitTemporal>
__device__ __forceinline__ void zllm_conv3d_f32(
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
    unsigned int out_width,
    const float* history, unsigned int head_frames)
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
                    const unsigned long long weight_index =
                        (((unsigned long long)oc * in_channels + ic) * kernel_depth + fd)
                        * kernel_height * kernel_width + fh * kernel_width + fw;
                    value += zllm_conv3d_input<SplitTemporal>(input, history, in_depth, head_frames, ic, (unsigned int)id,
                        (unsigned long long)in_height * in_width, (unsigned long long)(unsigned int)ih * in_width + (unsigned int)iw) * weight[weight_index];
                }
            }
        }
    }
    output[index] = value;
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
    zllm_conv3d_f32<false>(input, weight, bias, output, in_channels, out_channels, in_depth, in_height, in_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width, has_bias, pad_depth, pad_height, pad_width, pad_after_height, pad_after_width, causal, reflect_spatial, out_depth, out_height, out_width, nullptr, 0);
}

extern "C" __global__ void conv3d_history_f32(
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
    unsigned int out_width,
    const float* history, unsigned int head_frames)
{
    zllm_conv3d_f32<true>(input, weight, bias, output, in_channels, out_channels, in_depth, in_height, in_width, kernel_depth, kernel_height, kernel_width, stride_depth, stride_height, stride_width, has_bias, pad_depth, pad_height, pad_width, pad_after_height, pad_after_width, causal, reflect_spatial, out_depth, out_height, out_width, history, head_frames);
}

// 每线程拥有一个通道/空间点并按时间升序复制。后片源t=D+目标t，D>=1，
// 因此即便output复用history，也不会覆盖其它线程或本线程尚未读取的历史。
extern "C" __global__ void conv3d_history_tail_f32(
    const float* input, const float* history, float* output,
    unsigned int current_depth, unsigned int head_frames, unsigned int keep_frames,
    unsigned long long plane, unsigned long long elements)
{
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements / keep_frames) return;
    const unsigned int channel = index / plane;
    const unsigned long long position = index % plane;
    for (unsigned int t = 0; t < keep_frames; ++t) {
        const unsigned int time = current_depth + head_frames - keep_frames + t;
        output[((unsigned long long)channel * keep_frames + t) * plane + position] =
            zllm_conv3d_input<true>(input, history, current_depth + head_frames,
                head_frames, channel, time, plane, position);
    }
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

#if !defined(__HIP_DEVICE_COMPILE__) || (defined(__gfx1100__) && ROCWMMA_WAVE32_MODE)
// 3×3×3、stride1 的通用卷积展开；空间工作区固定，不按整段视频增长。
extern "C" __global__ void conv3d_history_columns_f16(
 const float* input,const float* history,_Float16* columns,unsigned int* invalid,
 unsigned current_depth,unsigned height,unsigned width,unsigned channels,unsigned head,
 unsigned long long start,unsigned n)
{
 const unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(index>=(unsigned long long)channels*27*n)return;
 const unsigned k=index/n;const unsigned long long pos=start+index%n,plane=(unsigned long long)height*width;
 const unsigned channel=k/27,tap=k%27,time=pos/plane+tap/9;
 const long long y=(long long)((pos/width)%height)+(int)((tap/3)%3)-1;
 const long long x=(long long)(pos%width)+(int)(tap%3)-1;
 float value=0.0f;
 if(y>=0&&y<height&&x>=0&&x<width&&time<current_depth+head)
  value=zllm_conv3d_input<true>(input,history,current_depth+head,head,channel,time,plane,(unsigned long long)y*width+x);
 if(!isfinite(value)||fabsf(value)>65504.0f)atomicOr(invalid,1u);
 columns[index]=(_Float16)value;
}
// 逐点卷积只转换当前空间块，不展开邻域。
extern "C" __global__ void conv3d_pointwise_columns_f16(
    const float* input, const float* history, _Float16* columns, unsigned int* invalid,
    unsigned int channels, unsigned long long spatial, unsigned long long start, unsigned int n)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)channels * n) return;
    const float value = input[(i / n) * spatial + start + i % n];
    if (!isfinite(value) || fabsf(value) > 65504.0f) atomicOr(invalid, 1u);
    columns[i] = (_Float16)value;
}
extern "C" __global__ void conv3d_columns_gemm_f16(const _Float16* x,const float* w,const float* bias,float* out,unsigned co,unsigned k,unsigned n,unsigned long long start,unsigned long long spatial,unsigned has_bias) {
 using namespace rocwmma;
 using A=fragment<matrix_a,16,16,16,float16_t,row_major>;
 using B=fragment<matrix_b,16,16,16,float16_t,row_major>;
 using C=fragment<accumulator,16,16,16,float>;
 __shared__ _Float16 sa[64*32],sb[32*64];
 const unsigned wave=threadIdx.x/32,lane=threadIdx.x%32;
 const unsigned tiles=(n+63)/64;
 const unsigned oc0=(blockIdx.x/tiles)*64, pos0=(blockIdx.x%tiles)*64;
 const unsigned oc=oc0+(wave/2)*16, pos=pos0+(wave%2)*32;
 C sum[2];fill_fragment(sum[0],0.f);fill_fragment(sum[1],0.f);
 for(unsigned i=threadIdx.x;i<64*32;i+=256){unsigned r=i/32,c=i%32;sa[i]=(oc0+r<co&&c<k)?(_Float16)w[(unsigned long long)(oc0+r)*k+c]:(_Float16)0;}
 for(unsigned i=threadIdx.x;i<32*64;i+=256){unsigned r=i/64,c=i%64;sb[i]=(r<k&&pos0+c<n)?x[(unsigned long long)r*n+pos0+c]:(_Float16)0;}
 __syncthreads();
 for(unsigned base=0;base<k;base+=32){
  // 下一K块先读入寄存器；当前WMMA完成后才写LDS，隐藏global load延迟。
  float next_a[8];_Float16 next_b[8];
  #pragma unroll
  for(unsigned j=0;j<8;++j){unsigned i=threadIdx.x+j*256,r=i/32,c=i%32;next_a[j]=(oc0+r<co&&base+32+c<k)?w[(unsigned long long)(oc0+r)*k+base+32+c]:0.f;}
  #pragma unroll
  for(unsigned j=0;j<8;++j){unsigned i=threadIdx.x+j*256,r=i/64,c=i%64;next_b[j]=(base+32+r<k&&pos0+c<n)?x[(unsigned long long)(base+32+r)*n+pos0+c]:(_Float16)0;}
  #pragma unroll
  for(unsigned part=0;part<32;part+=16){
   A a;load_matrix_sync(a,(const float16_t*)(sa+(wave/2)*16*32+part),32);
   #pragma unroll
   for(unsigned t=0;t<2;++t){B b;load_matrix_sync(b,(const float16_t*)(sb+part*64+(wave%2)*32+t*16),64);mma_sync(sum[t],a,b,sum[t]);}
  }
  __syncthreads();
  #pragma unroll
  for(unsigned j=0;j<8;++j){unsigned i=threadIdx.x+j*256;sa[i]=(_Float16)next_a[j];sb[i]=next_b[j];}
  __syncthreads();
 }
 #pragma unroll
 for(unsigned t=0;t<2;++t){
  #pragma unroll
  for(unsigned e=0;e<C::num_elements;++e){unsigned r=oc+e*2+(lane>>4),c=pos+t*16+(lane&15);if(r<co&&c<n)out[(unsigned long long)r*spatial+start+c]=sum[t][e]+(has_bias?bias[r]:0.0f);}
 }
}

#endif
