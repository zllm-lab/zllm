#include "kernel_operator.h"

// Qwen3.5 attention 输出门：output = input * sigmoid(gate)。每个 AIV core
// 处理连续分片，两个输入和结果始终保留在 NPU tensor 中。
class KernelSigmoidGate {
public:
    __aicore__ inline KernelSigmoidGate() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR gate, GM_ADDR output,
                                uint32_t batch, uint32_t columns)
    {
        elements_ = batch * columns;
        input_.SetGlobalBuffer((__gm__ half*)input, elements_);
        gate_.SetGlobalBuffer((__gm__ half*)gate, elements_);
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
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> gate_queue;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
        pipe.InitBuffer(input_queue, 1, tile_bytes);
        pipe.InitBuffer(gate_queue, 1, tile_bytes);
        pipe.InitBuffer(output_queue, 1, tile_bytes);

        for (uint32_t offset = 0; offset < core_elements; offset += tile_elements) {
            auto input_local = input_queue.AllocTensor<half>();
            auto gate_local = gate_queue.AllocTensor<half>();
            AscendC::DataCopy(input_local, input_[core_offset + offset], tile_elements);
            AscendC::DataCopy(gate_local, gate_[core_offset + offset], tile_elements);
            input_queue.EnQue(input_local);
            gate_queue.EnQue(gate_local);
            input_local = input_queue.DeQue<half>();
            gate_local = gate_queue.DeQue<half>();

            auto output_local = output_queue.AllocTensor<half>();
            Compute(output_local, input_local, gate_local);
            output_queue.EnQue(output_local);
            input_queue.FreeTensor(input_local);
            gate_queue.FreeTensor(gate_local);
            output_local = output_queue.DeQue<half>();
            AscendC::DataCopy(output_[core_offset + offset], output_local, tile_elements);
            output_queue.FreeTensor(output_local);
        }
    }

    __aicore__ inline void Compute(const AscendC::LocalTensor<half>& output,
                                   const AscendC::LocalTensor<half>& input,
                                   const AscendC::LocalTensor<half>& gate)
    {
        __ubuf__ half* output_ptr = (__ubuf__ half*)output.GetPhyAddr();
        __ubuf__ half* input_ptr = (__ubuf__ half*)input.GetPhyAddr();
        __ubuf__ half* gate_ptr = (__ubuf__ half*)gate.GetPhyAddr();
        __VEC_SCOPE__
        {
            // dav-l210 每个 B16 vector register 覆盖 64 个 half。CreateAddrReg
            // 的第二个参数才是硬件步长；此前传 0 导致两个 repeat 都覆写前半 tile。
            constexpr uint16_t register_elements = 64;
            for (uint16_t repeat = 0;
                 repeat <= get_vloopn_bound_b16(128); ++repeat) {
                AscendC::MicroAPI::RegTensor<half> input_reg;
                AscendC::MicroAPI::RegTensor<half> gate_reg;
                AscendC::MicroAPI::RegTensor<half> denominator_reg;
                AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<half>();
                AscendC::MicroAPI::AddrReg zero_offset =
                    AscendC::MicroAPI::CreateAddrReg<half>(repeat, register_elements);
                AscendC::MicroAPI::DataCopy<half,
                    AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        input_reg, input_ptr, zero_offset);
                AscendC::MicroAPI::DataCopy<half,
                    AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        gate_reg, gate_ptr, zero_offset);
                AscendC::MicroAPI::Muls<half, half,
                    AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        denominator_reg, gate_reg, (half)-1.0f, mask);
                AscendC::MicroAPI::Exp<half,
                    AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        denominator_reg, denominator_reg, mask);
                AscendC::MicroAPI::Adds<half, half,
                    AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        denominator_reg, denominator_reg, (half)1.0f, mask);
                AscendC::MicroAPI::Div<half,
                    AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        denominator_reg, input_reg, denominator_reg, mask);
                AscendC::MicroAPI::DataCopy<half,
                    AscendC::MicroAPI::StoreDist::DIST_NORM_B16>(
                        output_ptr, denominator_reg, zero_offset, mask);
            }
        }
    }

private:
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> gate_;
    AscendC::GlobalTensor<half> output_;
    uint32_t elements_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR input, GM_ADDR gate, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSigmoidGate op;
    op.Init(input, gate, output, tiling_data.batch, tiling_data.columns);
    op.Process();
}
