// V4.1 engram 融合算子,分两步:host 完成 hash 查表与 embed 行解码并上传;
// gemv 把 embed × wkv(BF16)写进 scratch;apply 逐 (row, copy) 门控就地修正 h。
// hidden 维度(V4.1 为 5120)超过单 block shared 上限,key/value 全部走 global。
extern "C" __global__ void engram_gemv_bf16(
    const unsigned short* wkv_bf16,
    const float* embed,
    float* kv_out,
    unsigned int kv_rows,
    unsigned int kv_cols)
{
    const unsigned int row = blockIdx.x;
    const unsigned int output = blockIdx.y * blockDim.x + threadIdx.x;
    if (output >= kv_rows) return;
    const unsigned short* weights = wkv_bf16 + (unsigned long long)output * kv_cols;
    const float* values = embed + (unsigned long long)row * kv_cols;
    float dot = 0.0f;
    for (unsigned int column = 0; column < kv_cols; ++column) {
        // BF16 位型左移 16 即 F32 位型。
        dot += __uint_as_float(((unsigned int)weights[column]) << 16) * values[column];
    }
    kv_out[(unsigned long long)row * kv_rows + output] = dot;
}

extern "C" __global__ void engram_gate_apply_f32(
    const float* kv_out,
    const float* qk_weight,
    float* h,
    unsigned int hc,
    unsigned int dim,
    unsigned int kv_rows,
    float eps)
{
    const unsigned int row = blockIdx.y;
    const unsigned int copy = blockIdx.x;
    if (row >= gridDim.y || copy >= hc) return;
    __shared__ float reduced[1024];
    const unsigned int lane = threadIdx.x;
    const unsigned long long kv_base = (unsigned long long)row * kv_rows;
    const unsigned long long h_base = (unsigned long long)row * hc * dim + (unsigned long long)copy * dim;
    float partial = 0.0f;
    for (unsigned int d = lane; d < dim; d += blockDim.x) {
        const float h_value = h[h_base + d];
        partial += h_value * h_value;
    }
    reduced[lane] = partial;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) reduced[lane] += reduced[lane + stride];
        __syncthreads();
    }
    __shared__ float h_energy_shared;
    __shared__ float k_energy_shared;
    __shared__ float gate_shared;
    if (lane == 0) h_energy_shared = reduced[0] / (float)dim;
    __syncthreads();
    partial = 0.0f;
    for (unsigned int d = lane; d < dim; d += blockDim.x) {
        const float key = kv_out[kv_base + (unsigned long long)copy * dim + d];
        partial += key * key;
    }
    reduced[lane] = partial;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) reduced[lane] += reduced[lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        k_energy_shared = reduced[0] / (float)dim;
        // dot = Σ(h·(q⊙k)) · rstd / √dim;gate = sigmoid(copysign(√max(|dot|,1e-6), dot))。
        float dot = 0.0f;
        for (unsigned int d = 0; d < dim; ++d) {
            dot += h[h_base + d] * qk_weight[(unsigned long long)copy * dim + d] * kv_out[kv_base + (unsigned long long)copy * dim + d];
        }
        const float rstd = rsqrtf(h_energy_shared + eps) * rsqrtf(k_energy_shared + eps);
        dot *= rstd / sqrtf((float)dim);
        const float magnitude = fmaxf(fabsf(dot), 1.0e-6f);
        gate_shared = 1.0f / (1.0f + expf(-copysignf(sqrtf(magnitude), dot)));
    }
    __syncthreads();
    for (unsigned int d = lane; d < dim; d += blockDim.x) {
        h[h_base + d] += gate_shared * kv_out[kv_base + (unsigned long long)hc * dim + d];
    }
}
