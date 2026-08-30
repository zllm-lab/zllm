#include "kernel_operator.h"

// chunk 内 query-key Gram 必须保持因果性。每个 task 是连续的 16x16，
// 上三角直接在 AIV UB 中清零，输出继续留在 NPU 供 scan Cube 读取。
class KernelGdnQkCausalMask {
public:
    __aicore__ inline KernelGdnQkCausalMask() {}

    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR scales, GM_ADDR output, uint32_t tasks)
    {
        tasks_ = tasks;
        input_.SetGlobalBuffer((__gm__ half*)input, tasks * elements);
        scales_.SetGlobalBuffer((__gm__ half*)scales, tasks * side * 2);
        output_.SetGlobalBuffer((__gm__ half*)output, tasks * elements);
    }

    __aicore__ inline void Process()
    {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> input_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> scale_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> ratio_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> broadcast_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_buffer;
        pipe.InitBuffer(input_buffer, elements * sizeof(half));
        pipe.InitBuffer(output_buffer, elements * sizeof(half));
        pipe.InitBuffer(scale_buffer, side * 2 * sizeof(half));
        pipe.InitBuffer(ratio_buffer, side * sizeof(half));
        pipe.InitBuffer(broadcast_buffer, side * sizeof(half));
        pipe.InitBuffer(offset_buffer, side * sizeof(uint16_t));
        auto input_local = input_buffer.Get<half>();
        auto output_local = output_buffer.Get<half>();
        auto scales_local = scale_buffer.Get<half>();
        auto ratio = ratio_buffer.Get<half>();
        auto broadcast = broadcast_buffer.Get<half>();
        auto offsets = offset_buffer.Get<uint16_t>();
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        for (uint32_t task = core; task < tasks_; task += cores) {
            const uint32_t base = task * elements;
            AscendC::DataCopy(input_local, input_[base], elements);
            AscendC::DataCopy(scales_local, scales_[task * side * 2],
                              side * 2);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Duplicate(output_local, static_cast<half>(0.0f),
                               elements);
            for (uint32_t row = 0; row < side; ++row) {
                const uint32_t count = row + 1;
                AscendC::Duplicate(
                    offsets, static_cast<uint16_t>(row * sizeof(half)),
                    count);
                AscendC::Gather(broadcast, scales_local[side], offsets, 0,
                                count);
                AscendC::Sub(ratio, broadcast, scales_local[side], count);
                AscendC::Exp(ratio, ratio, count);
                AscendC::Mul(output_local[row * side],
                             input_local[row * side], ratio, count);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(output_[base], output_local, elements);
        }
    }

private:
    static constexpr uint32_t side = 16;
    static constexpr uint32_t elements = side * side;
    AscendC::GlobalTensor<half> input_, scales_, output_;
    uint32_t tasks_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR input, GM_ADDR scales, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    (void)workspace;
    KernelGdnQkCausalMask op;
    op.Init(input, scales, output, tiling_data.tasks);
    op.Process();
}
