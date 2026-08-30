#include "kernel_operator.h"

// Q6_K 只在当前层首次使用时展开成 Cube tile；后续 chunk 直接复用输出。
class KernelQ6DecodeTiles {
public:
    __aicore__ inline KernelQ6DecodeTiles() {}

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR input,
                                GM_ADDR weight_tiles, GM_ADDR input_tiles,
                                uint32_t rows, uint32_t columns, uint32_t batch)
    {
        rows_ = rows;
        columns_ = columns;
        batch_ = batch;
        blocks_per_row_ = columns / 256;
        column_tiles_ = columns / 16;
        weight_bytes_ = (__gm__ uint8_t*)weight;
        weight_half_.SetGlobalBuffer((__gm__ half*)weight,
                                     rows * blocks_per_row_ * 105);
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
            DecodeBlock(block);
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
    __aicore__ inline void Store(uint32_t row, uint32_t column, half value)
    {
        const uint32_t tile = (row / 16) * column_tiles_ + column / 16;
        const uint32_t destination = tile * 256 + (row % 16) * 16 + column % 16;
        weight_tiles_.SetValue(destination, value);
    }

    __aicore__ inline void DecodeBlock(uint32_t block)
    {
        const uint32_t row = block / blocks_per_row_;
        const uint32_t column_block = block % blocks_per_row_;
        const uint32_t base = block * 210;
        const float d = (float)weight_half_.GetValue(block * 105 + 104);
        for (uint32_t half_block = 0; half_block < 2; ++half_block) {
            const uint32_t low_base = base + half_block * 64;
            const uint32_t high_base = base + 128 + half_block * 32;
            const uint32_t scale_base = base + 192 + half_block * 8;
            for (uint32_t index = 0; index < 32; ++index) {
                const uint32_t scale_index = index / 16;
                const uint8_t high = weight_bytes_[high_base + index];
                const int32_t q1 = ((weight_bytes_[low_base + index] & 0x0f) | (((high >> 0) & 3) << 4)) - 32;
                const int32_t q2 = ((weight_bytes_[low_base + index + 32] & 0x0f) | (((high >> 2) & 3) << 4)) - 32;
                const int32_t q3 = ((weight_bytes_[low_base + index] >> 4) | (((high >> 4) & 3) << 4)) - 32;
                const int32_t q4 = ((weight_bytes_[low_base + index + 32] >> 4) | (((high >> 6) & 3) << 4)) - 32;
                const uint32_t column = column_block * 256 + half_block * 128 + index;
                Store(row, column, (half)(d * (float)((int8_t)weight_bytes_[scale_base + scale_index]) * (float)q1));
                Store(row, column + 32, (half)(d * (float)((int8_t)weight_bytes_[scale_base + scale_index + 2]) * (float)q2));
                Store(row, column + 64, (half)(d * (float)((int8_t)weight_bytes_[scale_base + scale_index + 4]) * (float)q3));
                Store(row, column + 96, (half)(d * (float)((int8_t)weight_bytes_[scale_base + scale_index + 6]) * (float)q4));
            }
        }
    }

    __gm__ uint8_t* weight_bytes_;
    AscendC::GlobalTensor<half> weight_half_;
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> weight_tiles_;
    AscendC::GlobalTensor<half> input_tiles_;
    uint32_t rows_, columns_, batch_, blocks_per_row_, column_tiles_;
};

extern "C" __global__ __aicore__ void q6_decode(
    GM_ADDR weight, GM_ADDR input, GM_ADDR weight_tiles, GM_ADDR input_tiles,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQ6DecodeTiles op;
    op.Init(weight, input, weight_tiles, input_tiles, tiling_data.rows,
            tiling_data.columns, tiling_data.batch);
    op.Process();
}
