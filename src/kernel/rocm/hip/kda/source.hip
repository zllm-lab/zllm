__device__ __forceinline__ float zllm_kda_sigmoid(float value)
{
    return value >= 0.0F ? 1.0F / (1.0F + expf(-value)) : expf(value) / (1.0F + expf(value));
}
__device__ __forceinline__ float zllm_kda_softplus(float value)
{
    if (value > 20.0F) return value;
    if (value < -20.0F) return expf(value);
    return value > 0.0F ? value + log1pf(expf(-value)) : log1pf(expf(value));
}

// KDA fused kernel:每个 block 处理一个 head,blockIdx.x = head,threadIdx.x = lane
// 对应 value column(head 同时覆盖 key column 维度,该维度在 kernel 内串行)。
//
// layout:
//   conv_state[3 * projection_size * (kernel-1)] — 三段 [q|k|v],每段 [channel][slot] shift register
//   recurrent[heads * head_dim * head_dim]       — [head][key_column][value_column]
//   q/k/v/decay/gate[rows * projection_size],beta[rows * heads]
//   {q,k,v}_conv_weight[projection_size * kernel],a_log[heads],dt_bias[projection_size],o_norm[head_dim]
//   output[rows * projection_size]
extern "C" __global__ void zllm_kda_fused_f32(
    float* conv_state,
    float* recurrent,
    const float* query,
    const float* key,
    const float* value,
    const float* decay,
    const float* beta,
    const float* output_gate,
    const float* query_conv_weight,
    const float* key_conv_weight,
    const float* value_conv_weight,
    const float* a_log,
    const float* dt_bias,
    const float* output_norm_weight,
    float* output,
    const unsigned int rows,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const unsigned int conv_kernel_size,
    const unsigned int gate_lower_bound_enabled,
    const float gate_lower_bound,
    const unsigned int use_qk_l2norm,
    const float output_norm_eps)
{
    const unsigned int head = blockIdx.x;
    if (head >= num_heads) return;
    const unsigned int lane = threadIdx.x;
    const unsigned int thread_count = blockDim.x;
    __shared__ float query_values[256];
    __shared__ float key_values[256];
    __shared__ float core_values[256];
    __shared__ float q_squares[256];
    __shared__ float k_squares[256];
    __shared__ float core_squares[256];
    __shared__ float query_inverse;
    __shared__ float key_inverse;
    __shared__ float core_inverse;
    const unsigned int projection_size = num_heads * head_dim;
    const size_t channel = (size_t)head * head_dim + lane;
    const unsigned int history = conv_kernel_size - 1;
    const size_t conv_path_stride = (size_t)projection_size * history;
    const size_t recurrent_head_base = (size_t)head * head_dim * head_dim;
    const float query_scale = rsqrtf((float)head_dim);
    const float a = expf(a_log[head]);

    for (unsigned int token = 0; token < rows; ++token)
    {
        const size_t token_base = (size_t)token * projection_size;
        float query_item = 0.0F;
        float key_item = 0.0F;
        float value_item = 0.0F;
        if (lane < head_dim)
        {
            const size_t weight_base = channel * conv_kernel_size;
            const size_t state_base = channel * history;
            query_item = query[token_base + channel] * query_conv_weight[weight_base + history];
            key_item = key[token_base + channel] * key_conv_weight[weight_base + history];
            value_item = value[token_base + channel] * value_conv_weight[weight_base + history];
            for (unsigned int index = 0; index < history; ++index)
            {
                query_item += conv_state[state_base + index] * query_conv_weight[weight_base + index];
                key_item += conv_state[conv_path_stride + state_base + index] * key_conv_weight[weight_base + index];
                value_item += conv_state[2 * conv_path_stride + state_base + index] * value_conv_weight[weight_base + index];
            }
            for (unsigned int index = 1; index < history; ++index)
            {
                conv_state[state_base + index - 1] = conv_state[state_base + index];
                conv_state[conv_path_stride + state_base + index - 1] = conv_state[conv_path_stride + state_base + index];
                conv_state[2 * conv_path_stride + state_base + index - 1] = conv_state[2 * conv_path_stride + state_base + index];
            }
            if (history != 0)
            {
                conv_state[state_base + history - 1] = query[token_base + channel];
                conv_state[conv_path_stride + state_base + history - 1] = key[token_base + channel];
                conv_state[2 * conv_path_stride + state_base + history - 1] = value[token_base + channel];
            }
            query_item *= zllm_kda_sigmoid(query_item);
            key_item *= zllm_kda_sigmoid(key_item);
            value_item *= zllm_kda_sigmoid(value_item);
        }
        query_values[lane] = query_item;
        key_values[lane] = key_item;
        q_squares[lane] = lane < head_dim ? query_item * query_item : 0.0F;
        k_squares[lane] = lane < head_dim ? key_item * key_item : 0.0F;
        __syncthreads();
        for (unsigned int stride = thread_count >> 1; stride > 0; stride >>= 1)
        {
            if (lane < stride)
            {
                q_squares[lane] += q_squares[lane + stride];
                k_squares[lane] += k_squares[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0)
        {
            query_inverse = use_qk_l2norm != 0 ? rsqrtf(q_squares[0] + 1.0e-6F) : 1.0F;
            key_inverse = use_qk_l2norm != 0 ? rsqrtf(k_squares[0] + 1.0e-6F) : 1.0F;
        }
        __syncthreads();

        float mixed = 0.0F;
        if (lane < head_dim)
        {
            float predicted = 0.0F;
            for (unsigned int key_column = 0; key_column < head_dim; ++key_column)
            {
                const unsigned int decay_column = head * head_dim + key_column;
                const float raw_gate = decay[token_base + decay_column] + dt_bias[decay_column];
                const float log_decay = gate_lower_bound_enabled != 0
                    ? gate_lower_bound * zllm_kda_sigmoid(a * raw_gate)
                    : -a * zllm_kda_softplus(raw_gate);
                const size_t state_index = recurrent_head_base + (size_t)key_column * head_dim + lane;
                recurrent[state_index] *= expf(log_decay);
                predicted += recurrent[state_index] * key_values[key_column] * key_inverse;
            }
            const float beta_value = zllm_kda_sigmoid(beta[(size_t)token * num_heads + head]);
            const float delta = (value_item - predicted) * beta_value;
            for (unsigned int key_column = 0; key_column < head_dim; ++key_column)
            {
                const size_t state_index = recurrent_head_base + (size_t)key_column * head_dim + lane;
                recurrent[state_index] += key_values[key_column] * key_inverse * delta;
                mixed += query_values[key_column] * query_inverse * query_scale * recurrent[state_index];
            }
        }
        core_values[lane] = mixed;
        core_squares[lane] = lane < head_dim ? mixed * mixed : 0.0F;
        __syncthreads();
        for (unsigned int stride = thread_count >> 1; stride > 0; stride >>= 1)
        {
            if (lane < stride) core_squares[lane] += core_squares[lane + stride];
            __syncthreads();
        }
        if (lane == 0) core_inverse = rsqrtf(core_squares[0] / (float)head_dim + output_norm_eps);
        __syncthreads();
        if (lane < head_dim)
        {
            output[token_base + channel] = core_values[lane]
                * core_inverse
                * output_norm_weight[lane]
                * zllm_kda_sigmoid(output_gate[token_base + channel]);
        }
        // 下一 token 复用共享数组,必须等所有 lane 写完 output 再进入下一轮。
        __syncthreads();
    }
}

// KDA chunked kernel(WY 表示):grid.x = num_heads * (head_dim / 32) 个 block,
// 每 block 拥有 (head, 32 个 value 列);state 切片常驻 LDS,chunk 间按序推进。
// chunk 内把逐 token delta rule 改写为 (I+L)^{-1} 前向替代,读出/状态推进用
// 矩阵形式;RMSNorm+gate 由 epilogue kernel 跨 tile 完成。数学推导与 CPU
// reference `recurrent_chunked` 一致,pad 规则:key 维 +2、value 维 +1 错开 bank。
extern "C" __global__ void zllm_kda_chunked_f32(
    float* conv_state,
    float* recurrent,
    const float* query,
    const float* key,
    const float* value,
    const float* decay,
    const float* beta,
    const float* output_gate,
    const float* query_conv_weight,
    const float* key_conv_weight,
    const float* value_conv_weight,
    const float* a_log,
    const float* dt_bias,
    float* mixed,
    const unsigned int rows,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const unsigned int conv_kernel_size,
    const unsigned int gate_lower_bound_enabled,
    const float gate_lower_bound,
    const unsigned int use_qk_l2norm)
{
    const unsigned int CHUNK = 8;   // 与 host 侧 KDA_CHUNK 常量一致
    const unsigned int TILE = 32;   // 与 host 侧 KDA_VALUE_TILE 常量一致
    const unsigned int tid = threadIdx.x;
    const unsigned int tiles = head_dim / TILE;
    const unsigned int head = blockIdx.x / tiles;
    const unsigned int tile = blockIdx.x % tiles;
    const unsigned int v0 = tile * TILE;
    const unsigned int projection_size = num_heads * head_dim;
    const unsigned int history = conv_kernel_size - 1;
    const unsigned int key_stride = head_dim + 2;
    const unsigned int val_stride = TILE + 1;
    extern __shared__ float shared[];
    float* state = shared;                               // [head_dim][val_stride]
    float* kk = state + head_dim * val_stride;           // [CHUNK][key_stride] 归一化 key
    float* qq = kk + CHUNK * key_stride;                 // [CHUNK][key_stride] 归一化 query
    float* cum = qq + CHUNK * key_stride;                // [CHUNK][key_stride] chunk 内累计 log decay
    float* dk = cum + CHUNK * key_stride;                // [CHUNK][key_stride] k̂·e^{cum}
    float* dq = dk + CHUNK * key_stride;                 // [CHUNK][key_stride] q̂·e^{cum}
    float* u = dq + CHUNK * key_stride;                  // [CHUNK][val_stride] rhs 原地前向替代
    float* vv = u + CHUNK * val_stride;                  // [CHUNK][val_stride] SiLU 后 value
    float* tbeta = vv + CHUNK * val_stride;              // [CHUNK][CHUNK] β_s·KᵀK 衰减内积
    float* aq = tbeta + CHUNK * CHUNK;                   // [CHUNK][CHUNK] 读出用 QKᵀ 衰减内积
    float* rawq = aq + CHUNK * CHUNK;                    // [3][head_dim] 短卷积原始输入窗口
    float* rawk = rawq + 3 * head_dim;                   // [3][head_dim]
    float* rawv = rawk + 3 * head_dim;                   // [3][TILE]
    float* red = rawv + 3 * TILE;                        // [2][blockDim.x] L2 归约 scratch
    float* betas = red + 2 * blockDim.x;                 // [CHUNK]

    const float a = expf(a_log[head]);
    const float query_scale = rsqrtf((float)head_dim);
    const size_t channel = (size_t)head * head_dim + tid;
    const size_t conv_path_stride = (size_t)projection_size * history;
    const size_t recurrent_head_base = (size_t)head * head_dim * head_dim;
    const size_t wq_base = channel * conv_kernel_size;

    for (unsigned int index = tid; index < head_dim * TILE; index += blockDim.x)
    {
        const unsigned int j = index / TILE;
        const unsigned int c = index % TILE;
        state[j * val_stride + c] = recurrent[recurrent_head_base + (size_t)j * head_dim + v0 + c];
    }
    if (tid < head_dim)
    {
        for (unsigned int i = 0; i < history; ++i)
        {
            rawq[i * head_dim + tid] = conv_state[channel * history + i];
            rawk[i * head_dim + tid] = conv_state[conv_path_stride + channel * history + i];
        }
    }
    if (tid < TILE)
    {
        const size_t v_channel = (size_t)head * head_dim + v0 + tid;
        for (unsigned int i = 0; i < history; ++i)
            rawv[i * TILE + tid] = conv_state[2 * conv_path_stride + v_channel * history + i];
    }
    __syncthreads();

    const unsigned int chunk_count = (rows + CHUNK - 1) / CHUNK;
    for (unsigned int chunk_index = 0; chunk_index < chunk_count; ++chunk_index)
    {
        const unsigned int chunk_start = chunk_index * CHUNK;
        const unsigned int chunk_len = rows - chunk_start < CHUNK ? rows - chunk_start : CHUNK;
        float cum_run = 0.0F;
        // phase a:逐 token 准备。conv 窗口是移位寄存器,token 必须串行;
        // 其余(q/k/v 投影、SiLU、L2、累计 decay)按 lane 并行。
        for (unsigned int s = 0; s < chunk_len; ++s)
        {
            const size_t token_base = (size_t)(chunk_start + s) * projection_size;
            float q_item = 0.0F;
            float k_item = 0.0F;
            if (tid < head_dim)
            {
                q_item = query[token_base + channel] * query_conv_weight[wq_base + history];
                k_item = key[token_base + channel] * key_conv_weight[wq_base + history];
                for (unsigned int i = 0; i < history; ++i)
                {
                    q_item += rawq[i * head_dim + tid] * query_conv_weight[wq_base + i];
                    k_item += rawk[i * head_dim + tid] * key_conv_weight[wq_base + i];
                }
                q_item *= zllm_kda_sigmoid(q_item);
                k_item *= zllm_kda_sigmoid(k_item);
                const float raw_gate = decay[token_base + channel] + dt_bias[channel];
                const float log_decay = gate_lower_bound_enabled != 0
                    ? gate_lower_bound * zllm_kda_sigmoid(a * raw_gate)
                    : -a * zllm_kda_softplus(raw_gate);
                cum_run += log_decay;
            }
            if (tid < TILE)
            {
                const size_t v_channel = (size_t)head * head_dim + v0 + tid;
                const size_t vw_base = v_channel * conv_kernel_size;
                float v_item = value[token_base + v_channel] * value_conv_weight[vw_base + history];
                for (unsigned int i = 0; i < history; ++i)
                    v_item += rawv[i * TILE + tid] * value_conv_weight[vw_base + i];
                vv[s * val_stride + tid] = v_item * zllm_kda_sigmoid(v_item);
            }
            float q_inv = 1.0F;
            float k_inv = 1.0F;
            if (use_qk_l2norm != 0)
            {
                red[tid] = tid < head_dim ? q_item * q_item : 0.0F;
                red[blockDim.x + tid] = tid < head_dim ? k_item * k_item : 0.0F;
                __syncthreads();
                for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
                {
                    if (tid < stride)
                    {
                        red[tid] += red[tid + stride];
                        red[blockDim.x + tid] += red[blockDim.x + tid + stride];
                    }
                    __syncthreads();
                }
                q_inv = rsqrtf(red[0] + 1.0e-6F);
                k_inv = rsqrtf(red[blockDim.x] + 1.0e-6F);
                __syncthreads();
            }
            if (tid < head_dim)
            {
                qq[s * key_stride + tid] = q_item * q_inv;
                const float k_hat = k_item * k_inv;
                kk[s * key_stride + tid] = k_hat;
                cum[s * key_stride + tid] = cum_run;
                const float decayed = expf(cum_run);
                dk[s * key_stride + tid] = k_hat * decayed;
                dq[s * key_stride + tid] = q_item * q_inv * decayed;
                for (unsigned int i = 1; i < history; ++i)
                {
                    rawq[(i - 1) * head_dim + tid] = rawq[i * head_dim + tid];
                    rawk[(i - 1) * head_dim + tid] = rawk[i * head_dim + tid];
                }
                if (history != 0)
                {
                    rawq[(history - 1) * head_dim + tid] = query[token_base + channel];
                    rawk[(history - 1) * head_dim + tid] = key[token_base + channel];
                }
            }
            if (tid < TILE)
            {
                for (unsigned int i = 1; i < history; ++i)
                    rawv[(i - 1) * TILE + tid] = rawv[i * TILE + tid];
                if (history != 0)
                {
                    const size_t v_channel = (size_t)head * head_dim + v0 + tid;
                    rawv[(history - 1) * TILE + tid] = value[token_base + v_channel];
                }
            }
            if (tid == 0)
                betas[s] = zllm_kda_sigmoid(beta[(size_t)(chunk_start + s) * num_heads + head]);
            __syncthreads();
        }
        // phase b:L[s][r] = β_s Σ_j k̂_r[j] k̂_s[j] e^{cum_s[j] − cum_r[j]}(r<s)。
        {
            const unsigned int pair_count = chunk_len * (chunk_len - 1) / 2;
            for (unsigned int p = tid; p < pair_count; p += blockDim.x)
            {
                unsigned int s = (unsigned int)((sqrtf(8.0F * p + 1.0F) + 1.0F) * 0.5F);
                while (s * (s - 1) / 2 > p) --s;
                while ((s + 1) * s / 2 <= p) ++s;
                const unsigned int r = p - s * (s - 1) / 2;
                float sum = 0.0F;
                for (unsigned int j = 0; j < head_dim; ++j)
                    sum += kk[r * key_stride + j] * kk[s * key_stride + j]
                        * expf(cum[s * key_stride + j] - cum[r * key_stride + j]);
                tbeta[s * CHUNK + r] = betas[s] * sum;
            }
        }
        __syncthreads();
        // phase c:rhs[s][c] = β_s (ṽ_s[c] − Σ_j state[j][c] dk_s[j]),存入 u 原地解。
        for (unsigned int item = tid; item < chunk_len * TILE; item += blockDim.x)
        {
            const unsigned int s = item / TILE;
            const unsigned int c = item % TILE;
            float q0 = 0.0F;
            for (unsigned int j = 0; j < head_dim; ++j)
                q0 += state[j * val_stride + c] * dk[s * key_stride + j];
            u[s * val_stride + c] = betas[s] * (vv[s * val_stride + c] - q0);
        }
        __syncthreads();
        // phase d:前向替代 U = (I+L)^{-1} rhs。行间串行,列内并行。
        for (unsigned int s = 1; s < chunk_len; ++s)
        {
            if (tid < TILE)
            {
                float acc = u[s * val_stride + tid];
                for (unsigned int r = 0; r < s; ++r)
                    acc -= tbeta[s * CHUNK + r] * u[r * val_stride + tid];
                u[s * val_stride + tid] = acc;
            }
            __syncthreads();
        }
        // phase e1:aq[s][r] = Σ_j q̂_s[j] k̂_r[j] e^{cum_s[j] − cum_r[j]}(r≤s,对角无衰减)。
        {
            const unsigned int entry_count = chunk_len * (chunk_len + 1) / 2;
            for (unsigned int e = tid; e < entry_count; e += blockDim.x)
            {
                unsigned int s = (unsigned int)((sqrtf(8.0F * e + 1.0F) - 1.0F) * 0.5F);
                while (s * (s + 1) / 2 > e) --s;
                while ((s + 1) * (s + 2) / 2 <= e) ++s;
                const unsigned int r = e - s * (s + 1) / 2;
                float sum = 0.0F;
                for (unsigned int j = 0; j < head_dim; ++j)
                {
                    const float weight = r == s ? 1.0F : expf(cum[s * key_stride + j] - cum[r * key_stride + j]);
                    sum += qq[s * key_stride + j] * kk[r * key_stride + j] * weight;
                }
                aq[s * CHUNK + r] = sum;
            }
        }
        __syncthreads();
        // phase e2:mixed = scale·(state 读出 + Σ_{r≤s} aq·U),写 global 由 epilogue 归一。
        for (unsigned int item = tid; item < chunk_len * TILE; item += blockDim.x)
        {
            const unsigned int s = item / TILE;
            const unsigned int c = item % TILE;
            float inter = 0.0F;
            for (unsigned int j = 0; j < head_dim; ++j)
                inter += state[j * val_stride + c] * dq[s * key_stride + j];
            float intra = 0.0F;
            for (unsigned int r = 0; r <= s; ++r)
                intra += aq[s * CHUNK + r] * u[r * val_stride + c];
            mixed[(size_t)(chunk_start + s) * projection_size + (size_t)head * head_dim + v0 + c] = query_scale * (inter + intra);
        }
        __syncthreads();
        // phase f:state 推进 S ← e^{cum_end} ⊙ S + Σ_s (k̂_s ⊙ e^{cum_end − cum_s}) u_sᵀ。
        {
            const unsigned int last = (chunk_len - 1) * key_stride;
            for (unsigned int index = tid; index < head_dim * TILE; index += blockDim.x)
            {
                const unsigned int j = index / TILE;
                const unsigned int c = index % TILE;
                const float cum_end = cum[last + j];
                float acc = state[j * val_stride + c] * expf(cum_end);
                for (unsigned int s = 0; s < chunk_len; ++s)
                    acc += kk[s * key_stride + j] * expf(cum_end - cum[s * key_stride + j]) * u[s * val_stride + c];
                state[j * val_stride + c] = acc;
            }
        }
        // 末个 chunk 把 conv 移位寄存器与 state 写回持久 buffer;
        // q/k 段由 tile 0 写(各 tile 值相同,避免并发同址),v 段各 tile 写自己。
        if (chunk_index == chunk_count - 1)
        {
            if (tid < head_dim && tile == 0)
            {
                for (unsigned int i = 0; i < history; ++i)
                {
                    conv_state[channel * history + i] = rawq[i * head_dim + tid];
                    conv_state[conv_path_stride + channel * history + i] = rawk[i * head_dim + tid];
                }
            }
            if (tid < TILE)
            {
                const size_t v_channel = (size_t)head * head_dim + v0 + tid;
                for (unsigned int i = 0; i < history; ++i)
                    conv_state[2 * conv_path_stride + v_channel * history + i] = rawv[i * TILE + tid];
            }
        }
        __syncthreads();
    }
    for (unsigned int index = tid; index < head_dim * TILE; index += blockDim.x)
    {
        const unsigned int j = index / TILE;
        const unsigned int c = index % TILE;
        recurrent[recurrent_head_base + (size_t)j * head_dim + v0 + c] = state[j * val_stride + c];
    }
}

// chunked epilogue:每个 (token, head) 一个 block,跨 value tile 完成
// RMSNorm + output_norm·sigmoid(gate)。
extern "C" __global__ void zllm_kda_chunked_epilogue_f32(
    const float* mixed,
    const float* output_gate,
    const float* output_norm_weight,
    float* output,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const unsigned int projection_size,
    const float output_norm_eps)
{
    const unsigned int token = blockIdx.x / num_heads;
    const unsigned int head = blockIdx.x % num_heads;
    const size_t index = (size_t)token * projection_size + (size_t)head * head_dim + threadIdx.x;
    const float value = threadIdx.x < head_dim ? mixed[index] : 0.0F;
    __shared__ float squares[256];
    squares[threadIdx.x] = value * value;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride) squares[threadIdx.x] += squares[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x < head_dim)
        output[index] = value * rsqrtf(squares[0] / (float)head_dim + output_norm_eps)
            * output_norm_weight[threadIdx.x] * zllm_kda_sigmoid(output_gate[index]);
}
