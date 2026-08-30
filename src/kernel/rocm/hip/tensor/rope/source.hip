extern "C" __global__ void rope_f32(
    const float* input,
    const float* cosine,
    const float* sine,
    float* output,
    unsigned int rows,
    unsigned int columns,
    unsigned int head_count,
    unsigned int rotary_dim,
    unsigned int position,
    unsigned int split_half,
    unsigned int prefix)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = rows * columns;
    if (index >= elements) return;
    const unsigned int row = index / columns;
    const unsigned int column = index - row * columns;
    const unsigned int head_dim = columns / head_count;
    const unsigned int local = column % head_dim;
    const unsigned int start = prefix ? 0 : head_dim - rotary_dim;
    if (local < start || local >= start + rotary_dim) {
        output[index] = input[index];
        return;
    }
    const unsigned int rotary = local - start;
    const unsigned int half = rotary_dim / 2;
    const unsigned int pair = split_half ? rotary % half : rotary / 2;
    const unsigned int real_local = split_half ? start + pair : start + pair * 2;
    const unsigned int imag_local = split_half ? start + half + pair : real_local + 1;
    const unsigned int head_start = column - local;
    const float real = input[(unsigned long long)row * columns + head_start + real_local];
    const float imaginary = input[(unsigned long long)row * columns + head_start + imag_local];
    const float c = cosine[(unsigned long long)(position + row) * half + pair];
    const float s = sine[(unsigned long long)(position + row) * half + pair];
    output[index] = local == real_local ? real * c - imaginary * s : imaginary * c + real * s;
}

extern "C" __global__ void rope_segmented_pair_f32(
    const float* query,
    const float* key,
    const float* cosine,
    const float* sine,
    const unsigned int* row_positions,
    float* query_output,
    float* key_output,
    unsigned int rows,
    unsigned int query_columns,
    unsigned int query_head_count,
    unsigned int key_columns,
    unsigned int key_head_count,
    unsigned int rotary_dim,
    unsigned int split_half)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int query_elements = rows * query_columns;
    const unsigned int key_elements = rows * key_columns;
    if (index >= query_elements + key_elements) return;
    const bool is_key = index >= query_elements;
    const unsigned int tensor_index = is_key ? index - query_elements : index;
    const unsigned int columns = is_key ? key_columns : query_columns;
    const unsigned int head_count = is_key ? key_head_count : query_head_count;
    const float* input = is_key ? key : query;
    float* output = is_key ? key_output : query_output;
    const unsigned int row = tensor_index / columns;
    const unsigned int column = tensor_index - row * columns;
    const unsigned int head_dim = columns / head_count;
    const unsigned int local = column % head_dim;
    const unsigned int start = head_dim - rotary_dim;
    if (local < start) {
        output[tensor_index] = input[tensor_index];
        return;
    }
    const unsigned int rotary = local - start;
    const unsigned int half = rotary_dim / 2;
    const unsigned int pair = split_half ? rotary % half : rotary / 2;
    const unsigned int real_local = split_half ? start + pair : start + pair * 2;
    const unsigned int imag_local = split_half ? start + half + pair : real_local + 1;
    const unsigned int head_start = column - local;
    const float real = input[(unsigned long long)row * columns + head_start + real_local];
    const float imaginary = input[(unsigned long long)row * columns + head_start + imag_local];
    const unsigned int position = row_positions[row];
    const float c = cosine[(unsigned long long)position * half + pair];
    const float s = sine[(unsigned long long)position * half + pair];
    output[tensor_index] = local == real_local ? real * c - imaginary * s : imaginary * c + real * s;
}
