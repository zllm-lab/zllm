#include "kernel_operator.h"

// Q8_0 decode + matvec kernel（纯标量 AIV 架构，和 Q5_K 相同）。
// Q8_0 块：34 字节 / 32 元素。d(f16) + 32 个 signed int8。
class KernelQ80Matvec {
public:
    __aicore__ inline KernelQ80Matvec() {}

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR input, GM_ADDR output,
                                uint32_t rows, uint32_t columns, uint32_t batch)
    {
        rows_ = rows;
        columns_ = columns;
        blocks_per_row_ = columns / 32;
        batch_ = batch;
        weight_bytes_ = (__gm__ uint8_t*)weight;
        input_.SetGlobalBuffer((__gm__ half*)input, batch * columns);
        output_.SetGlobalBuffer((__gm__ half*)output, batch * rows);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        for (uint32_t row = core; row < rows_; row += cores) {
            float sums[8] = {};
            for (uint32_t blk = 0; blk < blocks_per_row_; ++blk) {
                const uint32_t block = row * blocks_per_row_ + blk;
                const uint32_t base = block * 34;
                const float d = (float)((__gm__ half*)weight_bytes_)[block * 17];
                for (uint32_t i = 0; i < 32; ++i) {
                    // 一个量化权重同时服务整个 prefill 小批次，避免为每个 token
                    // 重扫 9010 GM 中的整行权重。
                    const float weight = d * (float)((int8_t)weight_bytes_[base + 2 + i]);
                    for (uint32_t token = 0; token < batch_; ++token) {
                        const uint32_t input_base = token * columns_ + blk * 32;
                        sums[token] += weight * (float)input_.GetValue(input_base + i);
                    }
                }
            }
            for (uint32_t token = 0; token < batch_; ++token) {
                output_.SetValue(token * rows_ + row, (half)sums[token]);
            }
        }
    }

private:
    __gm__ uint8_t* weight_bytes_;
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> output_;
    uint32_t rows_, columns_, batch_;
    uint32_t blocks_per_row_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR weight, GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQ80Matvec op;
    op.Init(weight, input, output, tiling_data.rows, tiling_data.columns,
            tiling_data.batch);
    op.Process();
}
