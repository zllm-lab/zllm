#include <hip/hip_bfloat16.h>

__device__ __forceinline__ float zllm_gdn_sigmoid(float value)
{
    return value >= 0.0F ? 1.0F / (1.0F + expf(-value)) : expf(value) / (1.0F + expf(value));
}

__device__ __forceinline__ float zllm_gdn_softplus(float value)
{
    if (value > 20.0F) return value;
    if (value < -20.0F) return expf(value);
    return value > 0.0F ? value + log1pf(expf(-value)) : log1pf(expf(value));
}

// GatedDeltaNet fused kernel:单 block 按 token 顺序执行,threadIdx.x 并行处理
// value_head_dim 列。每个 value_head 的 delta rule 由 thread 循环处理。
//
// layout:
//   conv_state[conv_dim * conv_kernel]         — shift register,按 [channel][slot] 排布
//   recurrent[value_heads * key_head_dim * value_head_dim] — 按 [vh][kc][vc] 排布
//   qkv[rows * conv_dim]                       — 按行排布,mixed 的前 key_dim 是 q,接下来 key_dim 是 k,最后 value_dim 是 v
//   z[rows * value_dim],alpha[rows * value_heads],beta[rows * value_heads]
//   norm[value_head_dim]                       — per-head 共享 norm 权重
//   output[rows * value_dim]
extern "C" __global__ void zllm_gdn_fused_f32(
    float* conv_state,
    float* recurrent,
    const float* qkv,
    const float* z_tensor,
    const float* alpha,
    const float* beta,
    const float* conv_weight,
    const float* a_log,
    const float* dt_bias,
    const float* norm_weight,
    float* output,
    const unsigned int rows,
    const unsigned int key_dim,
    const unsigned int value_dim,
    const unsigned int conv_dim,
    const unsigned int conv_kernel,
    const unsigned int key_heads,
    const unsigned int value_heads,
    const unsigned int key_head_dim,
    const unsigned int value_head_dim,
    const unsigned int grouped_heads,
    const float rms_eps)
{
    const unsigned int tid = threadIdx.x;
    const float query_scale = 1.0F / sqrtf((float)key_head_dim);

    // 共享内存:临时存储当前 token 的 query / key / core / mixed
    extern __shared__ float shared[];
    float* mixed = shared;                                   // conv_dim
    float* query = mixed + conv_dim;                         // key_dim
    float* key = query + key_dim;                            // key_dim

    for (unsigned int token = 0; token < rows; ++token)
    {
        const float* qkv_row = qkv + (size_t)token * conv_dim;

        // 阶段 A:conv shift register + SiLU
        for (unsigned int channel = tid; channel < conv_dim; channel += blockDim.x)
        {
            const unsigned int begin = channel * conv_kernel;
            const unsigned int end = begin + conv_kernel;
            for (unsigned int slot = begin; slot + 1 < end; ++slot)
                conv_state[slot] = conv_state[slot + 1];
            conv_state[end - 1] = qkv_row[channel];
            float sum = 0.0F;
            for (unsigned int slot = 0; slot < conv_kernel; ++slot)
                sum += conv_state[begin + slot] * conv_weight[begin + slot];
            mixed[channel] = sum * zllm_gdn_sigmoid(sum);
        }
        __syncthreads();

        // 阶段 B:Q/K L2 归一化
        for (unsigned int channel = tid; channel < key_dim; channel += blockDim.x)
        {
            query[channel] = mixed[channel];
            key[channel] = mixed[key_dim + channel];
        }
        __syncthreads();

        for (unsigned int head = tid; head < key_heads; head += blockDim.x)
        {
            const unsigned int begin = head * key_head_dim;
            float q_sq = 0.0F;
            float k_sq = 0.0F;
            for (unsigned int i = 0; i < key_head_dim; ++i)
            {
                q_sq += query[begin + i] * query[begin + i];
                k_sq += key[begin + i] * key[begin + i];
            }
            const float q_inv = rsqrtf(fmaxf(q_sq, 1e-12F));
            const float k_inv = rsqrtf(fmaxf(k_sq, 1e-12F));
            for (unsigned int i = 0; i < key_head_dim; ++i)
            {
                query[begin + i] *= q_inv * query_scale;
                key[begin + i] *= k_inv;
            }
        }
        __syncthreads();

        // 阶段 C:delta rule 递归 + 阶段 D:RMSNorm + gate
        const float* alpha_row = alpha + (size_t)token * value_heads;
        const float* beta_row = beta + (size_t)token * value_heads;
        const float* z_row = z_tensor + (size_t)token * value_dim;

        const unsigned int recurrent_head_stride = key_head_dim * value_head_dim;

        for (unsigned int value_head = 0; value_head < value_heads; ++value_head)
        {
            const unsigned int key_head = grouped_heads ? value_head / (value_heads / key_heads) : value_head % key_heads;
            const float decay = expf(-expf(a_log[value_head]) * zllm_gdn_softplus(alpha_row[value_head] + dt_bias[value_head]));
            const float beta_gate = zllm_gdn_sigmoid(beta_row[value_head]);

            const float* q_head = query + key_head * key_head_dim;
            const float* k_head = key + key_head * key_head_dim;
            const float* v_head = mixed + key_dim * 2 + value_head * value_head_dim;
            float* recurrent_head = recurrent + (size_t)value_head * recurrent_head_stride;

            // 衰减整个 head 状态
            for (unsigned int element = tid; element < recurrent_head_stride; element += blockDim.x)
                recurrent_head[element] *= decay;
            __syncthreads();

            // delta rule + 读出 output + RMSNorm,gate
            for (unsigned int vc = tid; vc < value_head_dim; vc += blockDim.x)
            {
                // memory = sum_kc(recurrent[kc*value_head_dim + vc] * k[kc])
                float memory = 0.0F;
                for (unsigned int kc = 0; kc < key_head_dim; ++kc)
                    memory += recurrent_head[kc * value_head_dim + vc] * k_head[kc];
                float delta_val = (v_head[vc] - memory) * beta_gate;
                float core = 0.0F;
                for (unsigned int kc = 0; kc < key_head_dim; ++kc)
                {
                    float* state = &recurrent_head[kc * value_head_dim + vc];
                    *state += k_head[kc] * delta_val;
                    core += *state * q_head[kc];
                }
                // RMSNorm(单元素 variance,head 内 reduce)
                // 注意:这里只算了一个 vc 的贡献,需要整个 head 的 variance。
                // 暂存 core 到 output,后面统一做 RMSNorm。
                output[(size_t)token * value_dim + value_head * value_head_dim + vc] = core;
            }
            __syncthreads();

            // RMSNorm over value_head_dim + gate
            // 用单个 thread 做 per-head RMSNorm(128 个元素,sum 可接受)
            if (tid == 0)
            {
                const unsigned int out_base = (size_t)token * value_dim + value_head * value_head_dim;
                float variance = 0.0F;
                for (unsigned int vc = 0; vc < value_head_dim; ++vc)
                {
                    float core = output[out_base + vc];
                    variance += core * core;
                }
                variance /= (float)value_head_dim;
                float inv_rms = 1.0F / sqrtf(variance + rms_eps);
                for (unsigned int vc = 0; vc < value_head_dim; ++vc)
                {
                    unsigned int index = out_base + vc;
                    float core = output[index];
                    // z_row 已含 token 偏移,这里必须用 head 内相对索引,
                    // 否则 token 偏移被叠加两次,读到后续 token 的 z。
                    float z = z_row[value_head * value_head_dim + vc];
                    output[index] = core * inv_rms * norm_weight[vc] * z * zllm_gdn_sigmoid(z);
                }
            }
            __syncthreads();
        }
    }
}
