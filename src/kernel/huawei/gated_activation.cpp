#include "kernel_operator.h"

// Qwen3.5 dense MLP 的 fused SiLU(gate) * up。每个 AIV core 处理连续分片，
// resident FP16 张量只在 GM/UB 间流动，不生成 host 中间结果。
class KernelGatedActivation {
public:
    __aicore__ inline KernelGatedActivation() {}

    __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR output,
                                uint32_t batch, uint32_t columns)
    {
        elements_ = batch * columns;
        gate_.SetGlobalBuffer((__gm__ half*)gate, elements_);
        up_.SetGlobalBuffer((__gm__ half*)up, elements_);
        output_.SetGlobalBuffer((__gm__ half*)output, elements_);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t tile_elements = 128;
        constexpr uint32_t tile_bytes = tile_elements * sizeof(half);
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t core_elements = elements_ / cores;
        const uint32_t core_offset = core * core_elements;

        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> gate_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> up_queue;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
        pipe.InitBuffer(gate_queue, 1, tile_bytes);
        pipe.InitBuffer(up_queue, 1, tile_bytes);
        pipe.InitBuffer(output_queue, 1, tile_bytes);

        for (uint32_t offset = 0; offset < core_elements; offset += tile_elements) {
            auto gate_local = gate_queue.AllocTensor<half>();
            auto up_local = up_queue.AllocTensor<half>();
            AscendC::DataCopy(gate_local, gate_[core_offset + offset], tile_elements);
            AscendC::DataCopy(up_local, up_[core_offset + offset], tile_elements);
            gate_queue.EnQue(gate_local);
            up_queue.EnQue(up_local);
            gate_local = gate_queue.DeQue<half>();
            up_local = up_queue.DeQue<half>();

            auto output_local = output_queue.AllocTensor<half>();
            Compute(output_local, gate_local, up_local);
            output_queue.EnQue(output_local);
            gate_queue.FreeTensor(gate_local);
            up_queue.FreeTensor(up_local);
            output_local = output_queue.DeQue<half>();
            AscendC::DataCopy(output_[core_offset + offset], output_local, tile_elements);
            output_queue.FreeTensor(output_local);
        }
    }

    __aicore__ inline void Compute(const AscendC::LocalTensor<half>& output,
                                   const AscendC::LocalTensor<half>& gate,
                                   const AscendC::LocalTensor<half>& up)
    {
        __ubuf__ half* output_ptr = (__ubuf__ half*)output.GetPhyAddr();
        __ubuf__ half* gate_ptr = (__ubuf__ half*)gate.GetPhyAddr();
        __ubuf__ half* up_ptr = (__ubuf__ half*)up.GetPhyAddr();
        __VEC_SCOPE__
        {
            // dav-l210 每个 B16 vector register 覆盖 64 个 half。CreateAddrReg
            // 的第二个参数才是硬件步长；此前传 0 导致两个 repeat 都覆写前半 tile。
            constexpr uint16_t register_elements = 64;
            for (uint16_t repeat = 0;
                 repeat <= get_vloopn_bound_b16(128); ++repeat) {
                AscendC::MicroAPI::RegTensor<half> gate_reg;
                AscendC::MicroAPI::RegTensor<half> up_reg;
                AscendC::MicroAPI::RegTensor<half> sigmoid_reg;
                AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<half>();
                AscendC::MicroAPI::AddrReg zero_offset =
                    AscendC::MicroAPI::CreateAddrReg<half>(repeat, register_elements);
                AscendC::MicroAPI::DataCopy<half,
                    AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        gate_reg, gate_ptr, zero_offset);
                AscendC::MicroAPI::DataCopy<half,
                    AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        up_reg, up_ptr, zero_offset);
                AscendC::MicroAPI::Muls<half, half,
                    AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        sigmoid_reg, gate_reg, (half)-1.0f, mask);
                AscendC::MicroAPI::Exp<half,
                    AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        sigmoid_reg, sigmoid_reg, mask);
                AscendC::MicroAPI::Adds<half, half,
                    AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        sigmoid_reg, sigmoid_reg, (half)1.0f, mask);
                AscendC::MicroAPI::Div<half,
                    AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        sigmoid_reg, gate_reg, sigmoid_reg, mask);
                AscendC::MicroAPI::Mul<half,
                    AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        sigmoid_reg, sigmoid_reg, up_reg, mask);
                AscendC::MicroAPI::DataCopy<half,
                    AscendC::MicroAPI::StoreDist::DIST_NORM_B16>(
                        output_ptr, sigmoid_reg, zero_offset, mask);
            }
        }
    }

private:
    AscendC::GlobalTensor<half> gate_;
    AscendC::GlobalTensor<half> up_;
    AscendC::GlobalTensor<half> output_;
    uint32_t elements_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR gate, GM_ADDR up, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGatedActivation op;
    op.Init(gate, up, output, tiling_data.batch, tiling_data.columns);
    op.Process();
}
