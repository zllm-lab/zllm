extern "C" __global__ void mla_attention_f32(
    const float* query,
    const float* kv,
    const float* rope,
    const unsigned int* selection,
    float* output,
    unsigned int query_rows,
    unsigned int key_rows,
    unsigned int q_projection,
    unsigned int kv_projection,
    unsigned int head_count,
    unsigned int q_head_dim,
    unsigned int kv_head_dim,
    unsigned int rope_dim,
    unsigned int nope_dim,
    unsigned int value_dim,
    unsigned int top_k,
    unsigned int causal,
    unsigned int selected
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (row >= query_rows || head >= head_count) return;

    unsigned int count = selected ? top_k : key_rows;
    if (causal) {
        const unsigned int causal_count = row + 1;
        if (count > causal_count) count = causal_count;
    }
    if (count == 0) return;

    const float* q = query + row * q_projection + head * q_head_dim;
    const float scale = rsqrtf((float)q_head_dim);
    __shared__ float partial[1024];
    __shared__ float shared_probability;
    __shared__ float shared_denominator;
    float maximum = -3.402823466e+38F;
    for (unsigned int candidate = 0; candidate < count; ++candidate) {
        const unsigned int token = selected ? selection[row * top_k + candidate] : candidate;
        float contribution = 0.0F;
        if (token < key_rows && lane < q_head_dim) {
            const float* key = kv + token * kv_projection + head * kv_head_dim;
            const float* key_rope = rope + token * rope_dim;
            contribution = lane < nope_dim
                ? q[lane] * key[lane]
                : q[lane] * key_rope[lane - nope_dim];
        }
        partial[lane] = contribution;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) partial[lane] += partial[lane + stride];
            __syncthreads();
        }
        if (lane == 0 && token < key_rows) maximum = fmaxf(maximum, partial[0] * scale);
        __syncthreads();
    }

    float denominator = 0.0F;
    float weighted = 0.0F;
    for (unsigned int candidate = 0; candidate < count; ++candidate) {
        const unsigned int token = selected ? selection[row * top_k + candidate] : candidate;
        float contribution = 0.0F;
        if (token < key_rows && lane < q_head_dim) {
            const float* key = kv + token * kv_projection + head * kv_head_dim;
            const float* key_rope = rope + token * rope_dim;
            contribution = lane < nope_dim
                ? q[lane] * key[lane]
                : q[lane] * key_rope[lane - nope_dim];
        }
        partial[lane] = contribution;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) partial[lane] += partial[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            shared_probability = token < key_rows ? expf(partial[0] * scale - maximum) : 0.0F;
            denominator += shared_probability;
        }
        __syncthreads();
        if (token < key_rows && lane < value_dim) {
            const float* key = kv + token * kv_projection + head * kv_head_dim;
            weighted += shared_probability * key[nope_dim + lane];
        }
        __syncthreads();
    }
    if (lane == 0) shared_denominator = denominator;
    __syncthreads();
    if (lane < q_head_dim) {
        output[row * q_projection + head * q_head_dim + lane] =
            lane < value_dim ? weighted / shared_denominator : 0.0F;
    }
}

extern "C" __global__ void gqa_prefill_f32(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    unsigned int rows,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float score_scale)
{
    const unsigned int item = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const unsigned int row = item / num_heads;
    const unsigned int head = item % num_heads;
    if (row >= rows) return;
    const unsigned int kv_head = head / (num_heads / num_kv_heads);
    const unsigned long long query_base = ((unsigned long long)row * num_heads + head) * head_dim;
    __shared__ float partial[1024];
    __shared__ float probability;
    __shared__ float maximum;
    __shared__ float denominator;
    if (lane == 0) {
        maximum = -3.402823466e+38F;
        denominator = 0.0F;
    }
    __syncthreads();

    for (unsigned int token = 0; token <= row; ++token) {
        const unsigned long long key_base = ((unsigned long long)token * num_kv_heads + kv_head) * head_dim;
        partial[lane] = lane < head_dim ? query[query_base + lane] * key[key_base + lane] : 0.0F;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) partial[lane] += partial[lane + stride];
            __syncthreads();
        }
        if (lane == 0) maximum = fmaxf(maximum, partial[0] * score_scale);
        __syncthreads();
    }

    float weighted = 0.0F;
    for (unsigned int token = 0; token <= row; ++token) {
        const unsigned long long key_base = ((unsigned long long)token * num_kv_heads + kv_head) * head_dim;
        partial[lane] = lane < head_dim ? query[query_base + lane] * key[key_base + lane] : 0.0F;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) partial[lane] += partial[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            probability = expf(partial[0] * score_scale - maximum);
            denominator += probability;
        }
        __syncthreads();
        if (lane < head_dim) weighted += probability * value[key_base + lane];
        __syncthreads();
    }
    if (lane < head_dim) output[query_base + lane] = weighted / denominator;
}
