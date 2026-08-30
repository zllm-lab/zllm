#include "kernel_operator.h"

// prefill 结束后只保留最后一个 token。输入/输出都由 HIAI 管理，复制在 NPU
// 上完成，host 不读取中间 hidden。
class KernelSelectLastRow {
public:
    __aicore__ inline KernelSelectLastRow() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                uint32_t rows, uint32_t columns)
    {
        rows_ = rows;
        columns_ = columns;
        input_.SetGlobalBuffer((__gm__ half*)input, rows * columns);
        output_.SetGlobalBuffer((__gm__ half*)output, columns);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t tile = 128;
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tile_buffer;
        pipe.InitBuffer(tile_buffer, 2 * tile * sizeof(half));
        auto tile_local = tile_buffer.Get<half>();
        const event_t load_to_store[2] = {
            static_cast<event_t>(
                pipe.FetchEventID(AscendC::HardEvent::MTE2_MTE3)),
            static_cast<event_t>(
                pipe.FetchEventID(AscendC::HardEvent::MTE2_MTE3))};
        const event_t store_to_load[2] = {
            static_cast<event_t>(
                pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2)),
            static_cast<event_t>(
                pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2))};
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t input_base = (rows_ - 1) * columns_;
        const uint32_t tiles = columns_ / tile;
        uint32_t ordinal = 0;
        for (uint32_t index = core; index < tiles;
             index += cores, ++ordinal) {
            const uint32_t slot = ordinal & 1;
            if (ordinal >= 2) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                    store_to_load[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                    store_to_load[slot]);
            }
            const uint32_t column = index * tile;
            const uint32_t local = slot * tile;
            AscendC::DataCopy(
                tile_local[local], input_[input_base + column], tile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                load_to_store[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                load_to_store[slot]);
            AscendC::DataCopy(output_[column], tile_local[local], tile);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::GlobalTensor<half> input_, output_;
    uint32_t rows_ = 0;
    uint32_t columns_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSelectLastRow op;
    op.Init(input, output, tiling_data.rows, tiling_data.columns);
    op.Process();
}
