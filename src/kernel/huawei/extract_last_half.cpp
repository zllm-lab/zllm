#include "kernel_operator.h"

// 分段 GDN 把结果原位保存在已消费的 QKV 后半区。这个 kernel 只做
// NPU 内部的连续列抽取，不让 host 接触中间 hidden。
class KernelExtractLastHalf {
public:
    __aicore__ inline KernelExtractLastHalf() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                uint32_t rows, uint32_t input_columns,
                                uint32_t output_columns)
    {
        rows_ = rows;
        input_columns_ = input_columns;
        output_columns_ = output_columns;
        input_.SetGlobalBuffer((__gm__ half*)input, rows * input_columns);
        output_.SetGlobalBuffer((__gm__ half*)output, rows * output_columns);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t tile = 128;
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tile_buffer;
        pipe.InitBuffer(tile_buffer, 2 * tile * sizeof(half));
        auto tile_local = tile_buffer.Get<half>();
        const event_t load_to_store[2] = {
            static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_MTE3)),
            static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_MTE3))};
        const event_t store_to_load[2] = {
            static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2)),
            static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2))};

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        // tiles_per_row 向下取整会丢尾部列：output_columns 必须是 128 的整数倍。
        // host 侧由固定 shape 的 OMC（1024x8192->4096）与 gated_delta_net_fused
        // 入口的 Qwen3.5-4B spec 校验保证，kernel 内无法返回错误，只能注释约束。
        const uint32_t tiles_per_row = output_columns_ / tile;
        const uint32_t total_tiles = rows_ * tiles_per_row;
        const uint32_t input_offset = input_columns_ - output_columns_;
        uint32_t ordinal = 0;
        for (uint32_t linear = core; linear < total_tiles;
             linear += cores, ++ordinal) {
            const uint32_t slot = ordinal & 1;
            if (ordinal >= 2) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(store_to_load[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(store_to_load[slot]);
            }
            const uint32_t row = linear / tiles_per_row;
            const uint32_t column = (linear - row * tiles_per_row) * tile;
            const uint32_t local = slot * tile;
            AscendC::DataCopy(tile_local[local],
                              input_[row * input_columns_ + input_offset + column],
                              tile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(load_to_store[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(load_to_store[slot]);
            AscendC::DataCopy(output_[row * output_columns_ + column],
                              tile_local[local], tile);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::GlobalTensor<half> input_, output_;
    uint32_t rows_ = 0, input_columns_ = 0, output_columns_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelExtractLastHalf op;
    op.Init(input, output, tiling_data.rows, tiling_data.input_columns,
            tiling_data.output_columns);
    op.Process();
}
