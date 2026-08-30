#include "kernel_operator.h"

// Q8_0 当前层权重展开为与 Q5/Q6 相同的 16x16 Cube tile 布局。
class KernelQ8DecodeTiles {
public:
    __aicore__ inline KernelQ8DecodeTiles() {}

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR input,
                                GM_ADDR weight_tiles, GM_ADDR input_tiles,
                                uint32_t rows, uint32_t columns, uint32_t batch)
    {
        rows_ = rows;
        columns_ = columns;
        batch_ = batch;
        blocks_per_row_ = columns / 32;
        column_tiles_ = columns / 16;
        weight_bytes_ = (__gm__ uint8_t*)weight;
        weight_half_.SetGlobalBuffer((__gm__ half*)weight,
                                     rows * blocks_per_row_ * 17);
        input_.SetGlobalBuffer((__gm__ half*)input, batch * columns);
        weight_tiles_.SetGlobalBuffer((__gm__ half*)weight_tiles, rows * columns);
        input_tiles_.SetGlobalBuffer((__gm__ half*)input_tiles,
                                     ((batch + 15) / 16) * 16 * columns);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t blocks = rows_ * blocks_per_row_;
        for (uint32_t block = core; block < blocks; block += cores) {
            const uint32_t row = block / blocks_per_row_;
            const uint32_t column_block = block % blocks_per_row_;
            const uint32_t base = block * 34;
            const float d = (float)weight_half_.GetValue(block * 17);
            for (uint32_t index = 0; index < 32; ++index) {
                const uint32_t column = column_block * 32 + index;
                const uint32_t tile = (row / 16) * column_tiles_ + column / 16;
                const uint32_t destination = tile * 256 + (row % 16) * 16 + column % 16;
                weight_tiles_.SetValue(destination,
                    (half)(d * (float)((int8_t)weight_bytes_[base + 2 + index])));
            }
        }
        const uint32_t elements = ((batch_ + 15) / 16) * 16 * columns_;
        for (uint32_t element = core; element < elements; element += cores) {
            const uint32_t tile = element / 256;
            const uint32_t within = element % 256;
            const uint32_t token_tile = tile / column_tiles_;
            const uint32_t column_tile = tile % column_tiles_;
            const uint32_t token = token_tile * 16 + within / 16;
            const uint32_t lane = within % 16;
            const uint32_t column = column_tile * 16 + lane;
            input_tiles_.SetValue(element, token < batch_
                ? input_.GetValue(token * columns_ + column) : (half)0.0f);
        }
    }

private:
    __gm__ uint8_t* weight_bytes_;
    AscendC::GlobalTensor<half> weight_half_;
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> weight_tiles_;
    AscendC::GlobalTensor<half> input_tiles_;
    uint32_t rows_, columns_, batch_, blocks_per_row_, column_tiles_;
};

extern "C" __global__ __aicore__ void q8_decode(
    GM_ADDR weight, GM_ADDR input, GM_ADDR weight_tiles, GM_ADDR input_tiles,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQ8DecodeTiles op;
    op.Init(weight, input, weight_tiles, input_tiles, tiling_data.rows,
            tiling_data.columns, tiling_data.batch);
    op.Process();
}
