extern "C" __global__ void select_rows_to_f32(
    const void* input,
    const unsigned int* selected,
    float* output,
    unsigned int selected_count,
    unsigned int columns,
    unsigned int input_is_bf16)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = selected_count * columns;
    if (index >= elements) return;
    const unsigned int source = selected[index / columns] * columns + index % columns;
    output[index] = input_is_bf16
        ? __uint_as_float((unsigned int)((const unsigned short*)input)[source] << 16)
        : ((const float*)input)[source];
}

extern "C" __global__ void argmax_rows_excluding_f32(
    const float* input,
    unsigned int columns,
    const unsigned int* excluded,
    unsigned int excluded_count,
    unsigned int* output)
{
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x;
    input += (unsigned long long)row * columns;
    float best_value = -3.402823466e+38F;
    unsigned int best_index = 0xffffffffu;
    for (unsigned int index = lane; index < columns; index += blockDim.x) {
        bool skip = false;
        for (unsigned int item = 0; item < excluded_count; ++item) {
            if (excluded[item] == index) {
                skip = true;
                break;
            }
        }
        const float value = input[index];
        if (!skip && value == value &&
            (best_index == 0xffffffffu || value > best_value ||
             (value == best_value && index < best_index))) {
            best_value = value;
            best_index = index;
        }
    }
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    values[lane] = best_value;
    indices[lane] = best_index;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other_value = values[lane + stride];
            const unsigned int other_index = indices[lane + stride];
            if (other_index != 0xffffffffu &&
                (indices[lane] == 0xffffffffu || other_value > values[lane] ||
                 (other_value == values[lane] && other_index < indices[lane]))) {
                values[lane] = other_value;
                indices[lane] = other_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) output[row] = indices[0];
}


// 大词表若只用一个 block，每个线程会串行扫描数百个 token。先按 shard 并行
// 归约，再让每行一个 block 合并少量 partial，避免让词表宽度限制并行度。
extern "C" __global__ void argmax_rows_excluding_partial_f32(
    const float* input,
    unsigned int columns,
    const unsigned int* excluded,
    unsigned int excluded_count,
    float* partial_values,
    unsigned int* partial_indices,
    unsigned int partial_count)
{
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x / partial_count;
    const unsigned int shard = blockIdx.x - row * partial_count;
    input += (unsigned long long)row * columns;
    float best_value = -3.402823466e+38F;
    unsigned int best_index = 0xffffffffu;
    for (unsigned int index = shard * blockDim.x + lane;
         index < columns;
         index += partial_count * blockDim.x) {
        bool skip = false;
        for (unsigned int item = 0; item < excluded_count; ++item) {
            if (excluded[item] == index) {
                skip = true;
                break;
            }
        }
        const float value = input[index];
        if (!skip && value == value &&
            (best_index == 0xffffffffu || value > best_value ||
             (value == best_value && index < best_index))) {
            best_value = value;
            best_index = index;
        }
    }
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    values[lane] = best_value;
    indices[lane] = best_index;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other_value = values[lane + stride];
            const unsigned int other_index = indices[lane + stride];
            if (other_index != 0xffffffffu &&
                (indices[lane] == 0xffffffffu || other_value > values[lane] ||
                 (other_value == values[lane] && other_index < indices[lane]))) {
                values[lane] = other_value;
                indices[lane] = other_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        const unsigned int partial = row * partial_count + shard;
        partial_values[partial] = values[0];
        partial_indices[partial] = indices[0];
    }
}

// 多行版 input[selected_rows] + bias 的分片 argmax，避免物化 select/add 临时张量。
extern "C" __global__ void argmax_add_rows_partial_f32(
    const float* logits,
    const unsigned int* selected_rows,
    const float* bias,
    unsigned int columns,
    float* partial_values,
    unsigned int* partial_indices,
    unsigned int partial_count)
{
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x / partial_count;
    const unsigned int shard = blockIdx.x - row * partial_count;
    const float* logit = logits + (unsigned long long)selected_rows[row] * columns;
    const float* bias_row = bias + (unsigned long long)row * columns;
    float best_value = -3.402823466e+38F;
    unsigned int best_index = 0xffffffffu;
    for (unsigned int index = shard * blockDim.x + lane;
         index < columns;
         index += partial_count * blockDim.x) {
        const float value = logit[index] + bias_row[index];
        if (value == value &&
            (best_index == 0xffffffffu || value > best_value ||
             (value == best_value && index < best_index))) {
            best_value = value;
            best_index = index;
        }
    }
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    values[lane] = best_value;
    indices[lane] = best_index;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other_value = values[lane + stride];
            const unsigned int other_index = indices[lane + stride];
            if (other_index != 0xffffffffu &&
                (indices[lane] == 0xffffffffu || other_value > values[lane] ||
                 (other_value == values[lane] && other_index < indices[lane]))) {
                values[lane] = other_value;
                indices[lane] = other_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        const unsigned int partial = row * partial_count + shard;
        partial_values[partial] = values[0];
        partial_indices[partial] = indices[0];
    }
}

extern "C" __global__ void argmax_rows_merge_partial_f32(
    const float* partial_values,
    const unsigned int* partial_indices,
    unsigned int partial_count,
    unsigned int* output)
{
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x;
    const unsigned int base = row * partial_count;
    float best_value = -3.402823466e+38F;
    unsigned int best_index = 0xffffffffu;
    for (unsigned int partial = lane; partial < partial_count; partial += blockDim.x) {
        const unsigned int index = partial_indices[base + partial];
        const float value = partial_values[base + partial];
        if (index != 0xffffffffu &&
            (best_index == 0xffffffffu || value > best_value ||
             (value == best_value && index < best_index))) {
            best_value = value;
            best_index = index;
        }
    }
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    values[lane] = best_value;
    indices[lane] = best_index;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other_value = values[lane + stride];
            const unsigned int other_index = indices[lane + stride];
            if (other_index != 0xffffffffu &&
                (indices[lane] == 0xffffffffu || other_value > values[lane] ||
                 (other_value == values[lane] && other_index < indices[lane]))) {
                values[lane] = other_value;
                indices[lane] = other_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) output[row] = indices[0];
}

extern "C" __global__ void sample_top_p_rows_excluding_f32(
    const float* input,
    float* probabilities,
    unsigned int columns,
    const float* sampling,
    const unsigned int* excluded,
    unsigned int excluded_count,
    unsigned int* output)
{
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x;
    input += (unsigned long long)row * columns;
    probabilities += (unsigned long long)row * columns;
    const float temperature = sampling[row * 3];
    const float top_p = sampling[row * 3 + 1];
    const float random = sampling[row * 3 + 2];
    __shared__ float minimums[1024];
    __shared__ float maximums[1024];
    __shared__ float sums[1024];
    __shared__ unsigned int indices[1024];
    __shared__ float state[4];

    if (temperature == 0.0F) {
        float best_value = -3.402823466e+38F;
        unsigned int best_index = 0xffffffffu;
        for (unsigned int index = lane; index < columns; index += blockDim.x) {
            bool skip = false;
            for (unsigned int item = 0; item < excluded_count; ++item) {
                if (excluded[item] == index) { skip = true; break; }
            }
            const float value = input[index];
            if (!skip && value == value && (best_index == 0xffffffffu || value > best_value || (value == best_value && index < best_index))) {
                best_value = value;
                best_index = index;
            }
        }
        maximums[lane] = best_value;
        indices[lane] = best_index;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                const float other_value = maximums[lane + stride];
                const unsigned int other_index = indices[lane + stride];
                if (other_index != 0xffffffffu && (indices[lane] == 0xffffffffu || other_value > maximums[lane] || (other_value == maximums[lane] && other_index < indices[lane]))) {
                    maximums[lane] = other_value;
                    indices[lane] = other_index;
                }
            }
            __syncthreads();
        }
        if (lane == 0) output[row] = indices[0];
        return;
    }

    float local_minimum = 3.402823466e+38F;
    float local_maximum = -3.402823466e+38F;
    for (unsigned int index = lane; index < columns; index += blockDim.x) {
        bool skip = false;
        for (unsigned int item = 0; item < excluded_count; ++item) {
            if (excluded[item] == index) { skip = true; break; }
        }
        if (skip) continue;
        const float value = input[index] / temperature;
        if (value == value) {
            local_minimum = fminf(local_minimum, value);
            local_maximum = fmaxf(local_maximum, value);
        }
    }
    minimums[lane] = local_minimum;
    maximums[lane] = local_maximum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            minimums[lane] = fminf(minimums[lane], minimums[lane + stride]);
            maximums[lane] = fmaxf(maximums[lane], maximums[lane + stride]);
        }
        __syncthreads();
    }
    const float max_logit = maximums[0];
    float sum = 0.0F;
    for (unsigned int index = lane; index < columns; index += blockDim.x) {
        bool skip = false;
        for (unsigned int item = 0; item < excluded_count; ++item) {
            if (excluded[item] == index) { skip = true; break; }
        }
        const float mass = skip ? -1.0F : expf(input[index] / temperature - max_logit);
        probabilities[index] = mass;
        if (!skip) sum += mass;
    }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        state[0] = minimums[0];
        state[1] = maximums[0];
        state[2] = top_p * sums[0];
    }
    __syncthreads();
    // 一次扫描同时完成连续两轮二分：middle 与随后可能访问的 lower/upper
    // 共用同一个 expf。阈值构造和各自的归约顺序不变，12 轮等价原 24 轮。
    for (unsigned int iteration = 0; iteration < 12; ++iteration) {
        const float middle = (state[0] + state[1]) * 0.5F;
        const float lower = (state[0] + middle) * 0.5F;
        const float upper = (middle + state[1]) * 0.5F;
        float lower_mass = 0.0F;
        float middle_mass = 0.0F;
        float upper_mass = 0.0F;
        for (unsigned int index = lane; index < columns; index += blockDim.x) {
            const float mass = probabilities[index];
            const float value = input[index] / temperature;
            if (mass >= 0.0F) {
                if (value >= lower) lower_mass += mass;
                if (value >= middle) middle_mass += mass;
                if (value >= upper) upper_mass += mass;
            }
        }
        minimums[lane] = lower_mass;
        sums[lane] = middle_mass;
        maximums[lane] = upper_mass;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                minimums[lane] += minimums[lane + stride];
                sums[lane] += sums[lane + stride];
                maximums[lane] += maximums[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0) {
            if (maximums[0] >= state[2]) {
                state[0] = upper;
            } else if (sums[0] >= state[2]) {
                state[0] = middle;
                state[1] = upper;
            } else if (minimums[0] >= state[2]) {
                state[0] = lower;
                state[1] = middle;
            } else {
                state[1] = lower;
            }
        }
        __syncthreads();
    }
    float threshold_candidate = 3.402823466e+38F;
    for (unsigned int index = lane; index < columns; index += blockDim.x) {
        const float value = input[index] / temperature;
        if (probabilities[index] >= 0.0F && value >= state[0]) threshold_candidate = fminf(threshold_candidate, value);
    }
    minimums[lane] = threshold_candidate;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) minimums[lane] = fminf(minimums[lane], minimums[lane + stride]);
        __syncthreads();
    }
    if (lane == 0) state[0] = minimums[0];
    __syncthreads();

    const unsigned int begin = (unsigned int)(((unsigned long long)columns * lane) / blockDim.x);
    const unsigned int end = (unsigned int)(((unsigned long long)columns * (lane + 1)) / blockDim.x);
    float allowed_mass = 0.0F;
    for (unsigned int index = begin; index < end; ++index) {
        const float value = input[index] / temperature;
        const float mass = probabilities[index];
        if (mass >= 0.0F && value >= state[0]) allowed_mass += mass;
    }
    sums[lane] = allowed_mass;
    __syncthreads();
    if (lane == 0) {
        float total = 0.0F;
        for (unsigned int index = 0; index < blockDim.x; ++index) total += sums[index];
        float target = random * total;
        unsigned int selected_lane = 0;
        for (; selected_lane + 1 < blockDim.x && target >= sums[selected_lane]; ++selected_lane) target -= sums[selected_lane];
        state[1] = (float)selected_lane;
        state[2] = target;
    }
    __syncthreads();
    if (lane == (unsigned int)state[1]) {
        float cumulative = 0.0F;
        unsigned int selected = 0xffffffffu;
        for (unsigned int index = begin; index < end; ++index) {
            const float value = input[index] / temperature;
            const float mass = probabilities[index];
            if (mass < 0.0F || value < state[0]) continue;
            cumulative += mass;
            selected = index;
            if (cumulative >= state[2]) break;
        }
        output[row] = selected;
    }
}

extern "C" __global__ void adaln_modulate_f32(
    const float* input,
    const float* shift,
    const float* scale,
    float* output,
    unsigned int elements,
    unsigned int columns,
    unsigned int modulation_rows)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int modulation = modulation_rows == 1 ? index % columns : index;
    output[index] = input[index] * (1.0f + scale[modulation]) + shift[modulation];
}

extern "C" __global__ void adaln_modulate_segment_f32(
    const float* input,
    const float* shift,
    const float* scale,
    float* output,
    unsigned int start_row,
    unsigned int row_count,
    unsigned int columns,
    unsigned int modulation_row)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = row_count * columns;
    if (index >= elements) return;
    const unsigned long long target = (unsigned long long)start_row * columns + index;
    const unsigned int modulation = modulation_row * columns + index % columns;
    output[target] = input[target] * (1.0f + scale[modulation]) + shift[modulation];
}

extern "C" __global__ void gated_residual_segment_f32(
    const float* residual,
    const float* update,
    const float* gate,
    float* output,
    unsigned int start_row,
    unsigned int row_count,
    unsigned int columns,
    unsigned int modulation_row)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = row_count * columns;
    if (index >= elements) return;
    const unsigned long long target = (unsigned long long)start_row * columns + index;
    const unsigned int modulation = modulation_row * columns + index % columns;
    output[target] = residual[target] + update[target] * gate[modulation];
}
