#include "kernel_operator.h"

// 把 NPU 内短生命周期的 ND FP16 权重重排为 Cube 16x16 tile。
// 输入来自量化 decode 的图内输出，不读取 sidecar，也不经过 CPU。
class KernelFp16NdPack {
public:
    __aicore__ inline KernelFp16NdPack() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR tiles,
                                uint32_t rows, uint32_t columns)
    {
        rows_ = rows;
        columns_ = columns;
        column_tiles_ = columns / cube_tile;
        input_.SetGlobalBuffer((__gm__ half*)input, rows * columns);
        tiles_.SetGlobalBuffer((__gm__ half*)tiles, rows * columns);
    }

    __aicore__ inline void Process()
    {
        if (rows_ % cube_tile != 0 || columns_ % cube_tile != 0) {
            return;
        }

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tile_buffer;
        pipe.InitBuffer(tile_buffer, 2 * cube_elements * sizeof(half));
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
        const AscendC::DataCopyParams gather(
            cube_tile, 1, columns_ / cube_tile - 1, 0);
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t tiles = rows_ / cube_tile * column_tiles_;
        uint32_t ordinal = 0;
        for (uint32_t tile = core; tile < tiles;
             tile += cores, ++ordinal) {
            const uint32_t slot = ordinal & 1;
            if (ordinal >= 2) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                    store_to_load[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                    store_to_load[slot]);
            }
            const uint32_t row_tile = tile / column_tiles_;
            const uint32_t column_tile = tile % column_tiles_;
            const uint32_t source =
                row_tile * cube_tile * columns_ + column_tile * cube_tile;
            const uint32_t local = slot * cube_elements;
            AscendC::DataCopy(tile_local[local], input_[source], gather);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                load_to_store[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                load_to_store[slot]);
            AscendC::DataCopy(tiles_[tile * cube_elements],
                              tile_local[local], cube_elements);
        }
    }

private:
    static constexpr uint32_t cube_tile = 16;
    static constexpr uint32_t cube_elements = cube_tile * cube_tile;

    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> tiles_;
    uint32_t rows_ = 0;
    uint32_t columns_ = 0;
    uint32_t column_tiles_ = 0;
};

extern "C" __global__ __aicore__ void fp16_nd_pack(
    GM_ADDR input, GM_ADDR tiles, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFp16NdPack op;
    op.Init(input, tiles, tiling_data.rows, tiling_data.columns);
    op.Process();
}
