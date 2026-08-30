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
