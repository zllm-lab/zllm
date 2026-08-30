extern "C" __global__ void stable_radix_topk_u32(
    const unsigned int* score_keys,
    const unsigned int* visible_counts,
    unsigned int* selection,
    unsigned int query_rows,
    unsigned int score_stride,
    unsigned int query_start,
    unsigned int top_k)
{
    const unsigned int row = blockIdx.x;
    if (row >= query_rows) return;
    const unsigned int causal_visible = query_start + row + 1u;
    const unsigned int requested_visible = visible_counts != nullptr ? visible_counts[row] : causal_visible;
    const unsigned int visible = min(requested_visible, score_stride);
    const unsigned int target_count = min(top_k, visible);
    if (target_count == 0u) return;
    const unsigned long long row_base = (unsigned long long)row * score_stride;
    __shared__ unsigned int histogram[256];
    __shared__ unsigned int offsets[256];
    __shared__ unsigned int prefix;
    __shared__ unsigned int prefix_mask;
    __shared__ unsigned int rank;
    __shared__ unsigned int threshold;
    __shared__ unsigned int greater_count;
    __shared__ unsigned int first_equal;

    if (threadIdx.x == 0u) {
        prefix = 0u;
        prefix_mask = 0u;
        rank = target_count - 1u;
    }
    __syncthreads();

    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[threadIdx.x] = 0u;
        __syncthreads();
        for (unsigned int token = threadIdx.x; token < visible; token += blockDim.x) {
            const unsigned int key = score_keys[row_base + token];
            if ((key & prefix_mask) == prefix) atomicAdd(&histogram[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            for (int bucket = 255; bucket >= 0; --bucket) {
                const unsigned int count = histogram[(unsigned int)bucket];
                if (rank < count) {
                    prefix |= (unsigned int)bucket << shift;
                    prefix_mask |= 255u << shift;
                    break;
                }
                rank -= count;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        threshold = prefix;
        first_equal = 0xffffffffu;
    }
    __syncthreads();

    // 每个线程处理连续 token 区间，线程前缀和因此保持历史位置稳定顺序。
    const unsigned int begin = ((unsigned long long)visible * threadIdx.x) / blockDim.x;
    const unsigned int end = ((unsigned long long)visible * (threadIdx.x + 1u)) / blockDim.x;
    unsigned int count = 0u;
    unsigned int local_first_equal = 0xffffffffu;
    for (unsigned int token = begin; token < end; ++token) {
        const unsigned int key = score_keys[row_base + token];
        count += key > threshold;
        if (key == threshold && local_first_equal == 0xffffffffu) local_first_equal = token;
    }
    if (local_first_equal != 0xffffffffu) atomicMin(&first_equal, local_first_equal);
    offsets[threadIdx.x] = count;
    __syncthreads();
    if (threadIdx.x == 0u) {
        unsigned int offset = 0u;
        for (unsigned int thread = 0u; thread < blockDim.x; ++thread) {
            const unsigned int thread_count = offsets[thread];
            offsets[thread] = offset;
            offset += thread_count;
        }
        greater_count = offset;
    }
    __syncthreads();

    unsigned int output = offsets[threadIdx.x];
    for (unsigned int token = begin; token < end; ++token) {
        if (score_keys[row_base + token] > threshold) {
            selection[(unsigned long long)row * top_k + output++] = token;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        unsigned int output = greater_count;
        if (output + 1u == target_count) {
            selection[(unsigned long long)row * top_k + output] = first_equal;
        } else {
            for (unsigned int token = 0u; token < visible && output < target_count; ++token) {
                if (score_keys[row_base + token] == threshold) {
                    selection[(unsigned long long)row * top_k + output++] = token;
                }
            }
        }
    }
}
