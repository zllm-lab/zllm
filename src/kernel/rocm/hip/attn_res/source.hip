// AttnRes mix kernel:每个 block 处理一行,threadIdx.x 分列 stride 循环。
// 先逐候选算 sum_squares 与 sum(value * norm * projection),树形归约后得到
// score = projection * rsqrt(sum_squares / columns + eps),再 softmax 加权。
extern "C" __global__ void zllm_attn_res_mix_f32(
    const float* const* candidates,
    const float* norm_weight,
    const float* projection_weight,
    float* output,
    const unsigned int rows,
    const unsigned int columns,
    const unsigned int candidate_count,
    const float eps)
{
    const unsigned int row = blockIdx.x;
    if (row >= rows || candidate_count == 0 || candidate_count > 32) return;
    const unsigned int lane = threadIdx.x;
    const unsigned int thread_count = blockDim.x;
    __shared__ float squares[256];
    __shared__ float projections[256];
    __shared__ float probabilities[32];

    for (unsigned int candidate = 0; candidate < candidate_count; ++candidate)
    {
        const float* base = candidates[candidate] + (size_t)row * columns;
        float sum_squares = 0.0F;
        float projection = 0.0F;
        for (unsigned int column = lane; column < columns; column += thread_count)
        {
            const float value = base[column];
            sum_squares += value * value;
            projection += value * norm_weight[column] * projection_weight[column];
        }
        squares[lane] = sum_squares;
        projections[lane] = projection;
        __syncthreads();
        for (unsigned int stride = thread_count >> 1; stride > 0; stride >>= 1)
        {
            if (lane < stride)
            {
                squares[lane] += squares[lane + stride];
                projections[lane] += projections[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0)
            probabilities[candidate] = projections[0] * rsqrtf(squares[0] / (float)columns + eps);
        __syncthreads();
    }

    if (lane == 0)
    {
        float maximum = probabilities[0];
        for (unsigned int candidate = 1; candidate < candidate_count; ++candidate)
            maximum = fmaxf(maximum, probabilities[candidate]);
        float denominator = 0.0F;
        for (unsigned int candidate = 0; candidate < candidate_count; ++candidate)
        {
            probabilities[candidate] = expf(probabilities[candidate] - maximum);
            denominator += probabilities[candidate];
        }
        const float inverse = 1.0F / denominator;
        for (unsigned int candidate = 0; candidate < candidate_count; ++candidate)
            probabilities[candidate] *= inverse;
    }
    __syncthreads();

    for (unsigned int column = lane; column < columns; column += thread_count)
    {
        float value = 0.0F;
        for (unsigned int candidate = 0; candidate < candidate_count; ++candidate)
            value += probabilities[candidate] * candidates[candidate][(size_t)row * columns + column];
        output[(size_t)row * columns + column] = value;
    }
}
