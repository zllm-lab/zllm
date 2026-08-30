extern "C" __global__ void silu_f32(const float* input, float* output, unsigned int elements)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float value = input[index];
        output[index] = value / (1.0f + expf(-value));
    }
}

extern "C" __global__ void add_row_bias_f32(
    const float* input,
    const float* bias,
    float* output,
    unsigned int elements,
    unsigned int columns)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index] + bias[index % columns];
}

extern "C" __global__ void modulation_chunk_f32(
    const float* input,
    float* output,
    unsigned int elements,
    unsigned int modalities,
    unsigned int chunks,
    unsigned int hidden,
    unsigned int chunk)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int output_row = index / hidden;
    const unsigned int column = index - output_row * hidden;
    const unsigned int row = output_row / modalities;
    const unsigned int modality = output_row - row * modalities;
    const unsigned int input_columns = modalities * chunks * hidden;
    const unsigned long long source = (unsigned long long)row * input_columns + (modality * chunks + chunk) * hidden + column;
    output[index] = input[source];
}
