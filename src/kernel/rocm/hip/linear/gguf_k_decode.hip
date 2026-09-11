// K-quant 小批次向量路径独立编译，使 wave64 不改变其他 WMMA 的 wave32 布局。
#include <hip/hip_runtime.h>

typedef unsigned short __attribute__((ext_vector_type(2))) zllm_ct_bf16x2;

struct GgufKqMeta {
    unsigned long long gate;
    unsigned long long up;
    unsigned long long down;
    unsigned int gate_type;
    unsigned int up_type;
    unsigned int down_type;
};

__device__ __forceinline__ float gguf_kq_f16(const unsigned char* bytes)
{
    // 标准 GGUF 的 half scale 按两字节对齐。硬件转换直接处理次正规数，
    // 避免每个量化块循环移位；仅保留原实现对 Inf/NaN payload 的位模式。
    union { unsigned short bits; _Float16 number; } value;
    value.bits = *(const unsigned short*)bytes;
    union { unsigned int bits; float number; } result;
    result.number = (float)value.number;
    const unsigned int expanded = ((unsigned int)(value.bits & 0x8000u) << 16u)
        | 0x7f800000u | ((unsigned int)(value.bits & 0x3ffu) << 13u);
    result.bits = (value.bits & 0x7c00u) == 0x7c00u ? expanded : result.bits;
    return result.number;
}

__device__ __forceinline__ unsigned int gguf_kq_block_bytes(const unsigned int tensor_type)
{
    return zllm_gguf_block_bytes(tensor_type);
}

__device__ __forceinline__ const unsigned char* gguf_kq_row(
    const unsigned long long weight, const unsigned int tensor_type,
    const unsigned int row, const unsigned int columns)
{
    return (const unsigned char*)weight
        + (unsigned long long)row * (columns >> 8) * gguf_kq_block_bytes(tensor_type);
}

// rows=2 时只需在两个 top-k 小数组之间配对，无需引入通用排序 kernel。
// 返回 >=0 表示第一行 assignment 对应的第二行 assignment，-1 表示不重复，
// -2 表示当前第二行 assignment 已由第一行负责。
__device__ __forceinline__ int gguf_rows2_peer_assignment(
    const unsigned int* route_slots,
    const unsigned int assignment,
    const unsigned int top_k)
{
    const unsigned int expert = route_slots[assignment];
    if (assignment < top_k) {
        for (unsigned int index = top_k; index < top_k * 2u; ++index) {
            if (route_slots[index] == expert) return (int)index;
        }
        return -1;
    }
    for (unsigned int index = 0u; index < top_k; ++index) {
        if (route_slots[index] == expert) return -2;
    }
    return -1;
}

__device__ __forceinline__ void gguf_k4_group_scale_min(
    const unsigned char* block,
    const unsigned int group,
    unsigned int& group_scale,
    unsigned int& minimum)
{
    const unsigned char* scales = block + 4u;
    if (group < 4u) {
        group_scale = scales[group] & 63u;
        minimum = scales[group + 4u] & 63u;
    } else {
        group_scale = (scales[group + 4u] & 15u)
            | ((scales[group - 4u] >> 6u) << 4u);
        minimum = (scales[group + 4u] >> 4u)
            | ((scales[group] >> 6u) << 4u);
    }
}

__device__ __forceinline__ zllm_ct_bf16x2 gguf_q4k_pair_bf16(
    const unsigned char* block,
    const unsigned int group,
    const unsigned int within,
    const float d,
    const float dmin,
    const unsigned int group_scale,
    const unsigned int minimum)
{
    const unsigned int shift = 4u * (group & 1u);
    const unsigned int packed_pair = *(const unsigned short*)(
        block + 16u + (group >> 1u) * 32u + within);
    const unsigned int first_quant = (packed_pair >> shift) & 15u;
    const unsigned int second_quant
        = (packed_pair >> (8u + shift)) & 15u;
    return {
        zllm_f32_to_bf16(
            d * (float)(group_scale * first_quant)
                - dmin * (float)minimum),
        zllm_f32_to_bf16(
            d * (float)(group_scale * second_quant)
                - dmin * (float)minimum)
    };
}

// Q4_K down 同样以两个 accumulator 模拟原 16-lane 归约的 offset=8，
// 同一组 d/scale/min 只解析一次，同时把每个 block 的输出行数翻倍。
__device__ __forceinline__ zllm_ct_bf16x2 gguf_q5k_pair_bf16(const unsigned char* block, unsigned int group, unsigned int within, float d, float dmin, unsigned int scale, unsigned int minimum) {
    const unsigned int shift=4u*(group&1u);
    const unsigned int low=*(const unsigned short*)(block+48u+(group>>1u)*32u+within);
    const unsigned int high=*(const unsigned short*)(block+16u+within);
    const unsigned int first=((low>>shift)&15u)+(((high>>group)&1u)<<4u);
    const unsigned int second=((low>>(8u+shift))&15u)+(((high>>(8u+group))&1u)<<4u);
    return {zllm_f32_to_bf16(d*(float)(scale*first)-dmin*(float)minimum),zllm_f32_to_bf16(d*(float)(scale*second)-dmin*(float)minimum)};
}

// 两组累加器保留原 16 lane 的求和次序，八 lane 同时处理两对输入。
// 编译期确定布局，避免每个量化块内反复分支并保留两套解码指令。
template <unsigned int TensorType>
__device__ __forceinline__ void gguf_fused_gate_up_k4_k5_impl(
const void* input, const GgufKqMeta* metas, const unsigned int* route_ids, const unsigned int* route_slots,
unsigned short* activated, unsigned int assignments, unsigned int top_k, unsigned int hidden,
unsigned int intermediate, unsigned int input_is_bf16) {
    (void)route_ids;
    const unsigned int assignment=blockIdx.y;
    if(assignment>=assignments || !input_is_bf16)return;
    // 每对相邻行独立复用专家权重，合批后仍保留相同累加顺序；奇数尾行单算。
    const unsigned int pair_base=(assignment/(2u*top_k))*(2u*top_k);
    int peer=pair_base+2u*top_k<=assignments?gguf_rows2_peer_assignment(route_slots+pair_base,assignment-pair_base,top_k):-1;
    if(peer>=0)peer+=(int)pair_base;
    if(peer==-2)return;
    const unsigned int lane=threadIdx.x&7u;
    const unsigned int row=blockIdx.x*(blockDim.x/8u)+(threadIdx.x>>3u);
    if(row>=intermediate)return;
    const GgufKqMeta meta=metas[route_slots[assignment]];
    const unsigned int type=TensorType;
    const unsigned int bb=type==12u?144u:176u;
    const unsigned char* gate=gguf_kq_row(meta.gate,type,row,hidden);
    const unsigned char* up=gguf_kq_row(meta.up,type,row,hidden);
    const unsigned short* token=(const unsigned short*)input+(unsigned long long)(assignment/top_k)*hidden;
    float g0=0.f,g1=0.f,u0=0.f,u1=0.f;
    const unsigned short* peer_token=(const unsigned short*)input+(unsigned long long)(peer>=0?peer/top_k:0)*hidden;
    float pg0=0.f,pg1=0.f,pu0=0.f,pu1=0.f;
    for(unsigned int bi=0;bi<hidden/256u;++bi){
        const unsigned char* gb=gate+(unsigned long long)bi*bb;
        const unsigned char* ub=up+(unsigned long long)bi*bb;
        float gd=gguf_kq_f16(gb),gm=gguf_kq_f16(gb+2u);
        float ud=gguf_kq_f16(ub),um=gguf_kq_f16(ub+2u);
        #pragma unroll
        for(unsigned int group=0;group<8u;++group){
            unsigned int gs=0,gmin=0,us=0,umin=0;
            gguf_k4_group_scale_min(gb,group,gs,gmin);gguf_k4_group_scale_min(ub,group,us,umin);
            // half 的 11 位有效数乘 6-bit scale 和 4-bit code 可由 F32 精确表示。
            // Q4 先合并组 scale/min，两次 pair 解码只需乘 code；单位参数会被编译期消去。
            const float gds=gd*(float)gs,gms=gm*(float)gmin;
            const float uds=ud*(float)us,ums=um*(float)umin;
            const unsigned int within=lane*2u;
            const unsigned int col=bi*256u+group*32u+within;
            const zllm_ct_bf16x2 x0={token[col],token[col+1u]},x1={token[col+16u],token[col+17u]};
            const zllm_ct_bf16x2 gw0=type==12u?gguf_q4k_pair_bf16(gb,group,within,gds,gms,1u,1u):gguf_q5k_pair_bf16(gb,group,within,gd,gm,gs,gmin);
            const zllm_ct_bf16x2 uw0=type==12u?gguf_q4k_pair_bf16(ub,group,within,uds,ums,1u,1u):gguf_q5k_pair_bf16(ub,group,within,ud,um,us,umin);
            const zllm_ct_bf16x2 gw1=type==12u?gguf_q4k_pair_bf16(gb,group,within+16u,gds,gms,1u,1u):gguf_q5k_pair_bf16(gb,group,within+16u,gd,gm,gs,gmin);
            const zllm_ct_bf16x2 uw1=type==12u?gguf_q4k_pair_bf16(ub,group,within+16u,uds,ums,1u,1u):gguf_q5k_pair_bf16(ub,group,within+16u,ud,um,us,umin);
            g0+=__builtin_amdgcn_fdot2_f32_bf16(x0,gw0,0.f,false);u0+=__builtin_amdgcn_fdot2_f32_bf16(x0,uw0,0.f,false);
            g1+=__builtin_amdgcn_fdot2_f32_bf16(x1,gw1,0.f,false);u1+=__builtin_amdgcn_fdot2_f32_bf16(x1,uw1,0.f,false);
            if(peer>=0){
                const zllm_ct_bf16x2 px0={peer_token[col],peer_token[col+1u]},px1={peer_token[col+16u],peer_token[col+17u]};
                pg0+=__builtin_amdgcn_fdot2_f32_bf16(px0,gw0,0.f,false);pu0+=__builtin_amdgcn_fdot2_f32_bf16(px0,uw0,0.f,false);
                pg1+=__builtin_amdgcn_fdot2_f32_bf16(px1,gw1,0.f,false);pu1+=__builtin_amdgcn_fdot2_f32_bf16(px1,uw1,0.f,false);
            }
        }
    }
    g0+=g1;u0+=u1;
    for(unsigned int offset=4u;offset;offset>>=1u){g0+=__shfl_down(g0,offset,8);u0+=__shfl_down(u0,offset,8);}
    if(lane==0u)activated[(unsigned long long)assignment*intermediate+row]=zllm_f32_to_bf16(g0/(1.f+expf(-g0))*u0);
    if(peer>=0){
        pg0+=pg1;pu0+=pu1;
        for(unsigned int offset=4u;offset;offset>>=1u){pg0+=__shfl_down(pg0,offset,8);pu0+=__shfl_down(pu0,offset,8);}
        if(lane==0u)activated[(unsigned long long)peer*intermediate+row]=zllm_f32_to_bf16(pg0/(1.f+expf(-pg0))*pu0);
    }
}

extern "C" __global__ void gguf_fused_gate_up_q4_k_bf16(
const void* input, const GgufKqMeta* metas, const unsigned int* route_ids, const unsigned int* route_slots,
unsigned short* activated, unsigned int assignments, unsigned int top_k, unsigned int hidden,
unsigned int intermediate, unsigned int input_is_bf16) {
    gguf_fused_gate_up_k4_k5_impl<12u>(input,metas,route_ids,route_slots,activated,assignments,top_k,hidden,intermediate,input_is_bf16);
}

extern "C" __global__ void gguf_fused_gate_up_q5_k_bf16(
const void* input, const GgufKqMeta* metas, const unsigned int* route_ids, const unsigned int* route_slots,
unsigned short* activated, unsigned int assignments, unsigned int top_k, unsigned int hidden,
unsigned int intermediate, unsigned int input_is_bf16) {
    gguf_fused_gate_up_k4_k5_impl<13u>(input,metas,route_ids,route_slots,activated,assignments,top_k,hidden,intermediate,input_is_bf16);
}

template <unsigned int TensorType>
__device__ __forceinline__ void gguf_fused_down_k4_k5_impl(
    const unsigned short* activated,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    const float* route_weights,
    float* output,
    const unsigned int tokens,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate)
{
    (void)route_ids;
    const unsigned int token = blockIdx.y;
    if (token >= tokens) return;
    const unsigned int lane = threadIdx.x & 7u;
    const unsigned int subgroup = threadIdx.x >> 3u;
    const unsigned int output_row = blockIdx.x * 32u + subgroup;
    if (output_row >= hidden) return;
    const unsigned int within = lane * 2u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (unsigned int k = 0u; k < top_k; ++k) {
        const unsigned int assignment = token * top_k + k;
        const GgufKqMeta meta = metas[route_slots[assignment]];
        const unsigned char* down_row = gguf_kq_row(
            meta.down, TensorType, output_row, intermediate);
        const unsigned short* input = activated
            + (unsigned long long)assignment * intermediate;
        const float route_weight = route_weights[assignment];
        for (unsigned int block_index = 0u;
             block_index < intermediate / 256u;
             ++block_index) {
            const unsigned char* block
                = down_row + (unsigned long long)block_index * (TensorType == 12u ? 144u : 176u);
            float d = (TensorType == 13u || lane == 0u) ? gguf_kq_f16(block) : 0.0f;
            float dmin = (TensorType == 13u || lane == 0u) ? gguf_kq_f16(block + 2u) : 0.0f;
            // Q5 的重复地址读由缓存合并，避免每组 scale 的跨 lane 等待。
            if (TensorType == 12u) {
                d = __shfl(d, 0, 8);
                dmin = __shfl(dmin, 0, 8);
            }
            const unsigned short* block_input
                = input + block_index * 256u;
            #pragma unroll
            for (unsigned int group = 0u; group < 8u; ++group) {
                unsigned int group_scale = 0u;
                unsigned int minimum = 0u;
                if (TensorType == 13u || lane == 0u) {
                    gguf_k4_group_scale_min(
                        block, group, group_scale, minimum);
                }
                if (TensorType == 12u) {
                    group_scale = __shfl(group_scale, 0, 8);
                    minimum = __shfl(minimum, 0, 8);
                }
                const unsigned int column0 = group * 32u + within;
                const unsigned int column1 = column0 + 16u;
                const zllm_ct_bf16x2 input0 = {
                    block_input[column0], block_input[column0 + 1u]
                };
                const zllm_ct_bf16x2 input1 = {
                    block_input[column1], block_input[column1 + 1u]
                };
                sum0 += __builtin_amdgcn_fdot2_f32_bf16(
                    input0, (TensorType == 12u ? gguf_q4k_pair_bf16(
                        block, group, within, d, dmin,
                        group_scale, minimum) : gguf_q5k_pair_bf16(
                        block, group, within, d, dmin,
                        group_scale, minimum)),
                    0.0f, false) * route_weight;
                sum1 += __builtin_amdgcn_fdot2_f32_bf16(
                    input1, (TensorType == 12u ? gguf_q4k_pair_bf16(
                        block, group, within + 16u, d, dmin,
                        group_scale, minimum) : gguf_q5k_pair_bf16(
                        block, group, within + 16u, d, dmin,
                        group_scale, minimum)),
                    0.0f, false) * route_weight;
            }
        }
    }
    sum0 += sum1;
    for (unsigned int offset = 4u; offset; offset >>= 1u)
        sum0 += __shfl_down(sum0, offset, 8);
    if (lane == 0u)
        output[(unsigned long long)token * hidden + output_row] = sum0;
}

extern "C" __global__ void gguf_fused_down_q4_k_f32(
    const unsigned short* activated,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    const float* route_weights,
    float* output,
    const unsigned int tokens,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate)
{
    gguf_fused_down_k4_k5_impl<12u>(activated, metas, route_ids, route_slots, route_weights, output, tokens, top_k, hidden, intermediate);
}

extern "C" __global__ void gguf_fused_down_q5_k_f32(
    const unsigned short* activated,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    const float* route_weights,
    float* output,
    const unsigned int tokens,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate)
{
    (void)route_ids;
    gguf_fused_down_k4_k5_impl<13u>(
        activated, metas, route_ids, route_slots, route_weights, output,
        tokens, top_k, hidden, intermediate);
}
