// grid-stride + 4 路展开:peer 读是 PCIe 延迟主导,单线程 16B 的在途字节
// 不足以打满带宽;固定大 grid 让每线程搬运多个不连续向量,叠加展开提升
// 内存级并行。
extern "C" __global__ void zllm_peer_copy_f32x4(
    const float4* source,
    float4* target,
    const unsigned int vectors)
{
    const unsigned int total = gridDim.x * blockDim.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int step = total * 4;
    while (index + 3u * total < vectors)
    {
        const unsigned int i0 = index;
        const unsigned int i1 = index + total;
        const unsigned int i2 = index + 2 * total;
        const unsigned int i3 = index + 3 * total;
        const float4 v0 = source[i0];
        const float4 v1 = source[i1];
        const float4 v2 = source[i2];
        const float4 v3 = source[i3];
        target[i0] = v0;
        target[i1] = v1;
        target[i2] = v2;
        target[i3] = v3;
        index += step;
    }
    for (; index < vectors; index += total)
        target[index] = source[index];
}

// 三段增量共用一个 CTA，省去逐段 launch；大小由调用方传入，保持原始字节。
extern "C" __global__ void zllm_peer_copy3_f32x4(
    const float4* source0,
    const float4* source1,
    const float4* source2,
    float4* target0,
    float4* target1,
    float4* target2,
    const unsigned int vectors0,
    const unsigned int vectors1,
    const unsigned int vectors2)
{
    unsigned int index = threadIdx.x;
    if (index < vectors0) {
        target0[index] = source0[index];
    } else if ((index -= vectors0) < vectors1) {
        target1[index] = source1[index];
    } else if ((index -= vectors1) < vectors2) {
        target2[index] = source2[index];
    }
}

// 两卡 partial 已在各自 stream 上完成后，直接通过 PCIe BAR 读取对端值，
// 在目标卡一次完成 partial 归并和 residual add。这样两边仍各自留下同值
// hidden，但不再为 24KiB decode hidden 额外物化 P2P 副本。
extern "C" __global__ void zllm_peer_join_residual_f32x4(
    const float4* local_partial,
    const float4* peer_partial,
    const float4* residual,
    float4* output,
    const unsigned int vectors)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vectors) return;
    const float4 local = local_partial[index];
    const float4 peer = peer_partial[index];
    const float4 skip = residual[index];
    output[index] = make_float4(
        skip.x + (local.x + peer.x),
        skip.y + (local.y + peer.y),
        skip.z + (local.z + peer.z),
        skip.w + (local.w + peer.w));
}
