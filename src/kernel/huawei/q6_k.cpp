#include "kernel_operator.h"

// GGUF Q6_K 的标量 decode + matvec kernel（和 Q5_K 相同的纯 AIV 架构）。
// Q6_K 块：210 字节 / 256 元素。low_bits[128](4位) + high_bits[64](2位)
// + scales[16](signed int8) + d(f16)。无 dmin。
class KernelQ6KMatvec {
public:
    __aicore__ inline KernelQ6KMatvec() {}

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR input, GM_ADDR output,
                                uint32_t rows, uint32_t columns, uint32_t batch)
    {
        rows_ = rows;
        columns_ = columns;
        blocks_per_row_ = columns / 256;
        batch_ = batch;
        weight_bytes_ = (__gm__ uint8_t*)weight;
        weight_half_.SetGlobalBuffer((__gm__ half*)weight,
                                     rows * blocks_per_row_ * 105);
        input_.SetGlobalBuffer((__gm__ half*)input, batch * columns);
        output_.SetGlobalBuffer((__gm__ half*)output, batch * rows);
    }

    __aicore__ inline void Process()
    {
        const uint32_t block_index = AscendC::GetBlockIdx();
        const uint32_t block_count = AscendC::GetBlockNum();
        for (uint32_t row = block_index; row < rows_; row += block_count) {
            float sums[8] = {};
            for (uint32_t column_block = 0; column_block < blocks_per_row_; ++column_block) {
                const uint32_t block = row * blocks_per_row_ + column_block;
                const uint32_t base = block * 210;
                // d(f16) 在块内字节偏移 208 = half 索引 104。
                const float d = (float)weight_half_.GetValue(block * 105 + 104);
                for (uint32_t hb = 0; hb < 2; ++hb) {
                    const uint32_t low_base = base + hb * 64;
                    const uint32_t high_base = base + 128 + hb * 32;
                    const uint32_t scale_base = base + 192 + hb * 8;
                    for (uint32_t index = 0; index < 32; ++index) {
                        const uint32_t scale_index = index / 16;
                        const uint8_t high_value = weight_bytes_[high_base + index];
                        const int32_t q1 = ((weight_bytes_[low_base + index] & 0x0f) | (((high_value >> 0) & 3) << 4)) - 32;
                        const int32_t q2 = ((weight_bytes_[low_base + index + 32] & 0x0f) | (((high_value >> 2) & 3) << 4)) - 32;
                        const int32_t q3 = ((weight_bytes_[low_base + index] >> 4) | (((high_value >> 4) & 3) << 4)) - 32;
                        const int32_t q4 = ((weight_bytes_[low_base + index + 32] >> 4) | (((high_value >> 6) & 3) << 4)) - 32;
                        const float s1 = d * (float)((int8_t)weight_bytes_[scale_base + scale_index]);
                        const float s2 = d * (float)((int8_t)weight_bytes_[scale_base + scale_index + 2]);
                        const float s3 = d * (float)((int8_t)weight_bytes_[scale_base + scale_index + 4]);
                        const float s4 = d * (float)((int8_t)weight_bytes_[scale_base + scale_index + 6]);
                        for (uint32_t token = 0; token < batch_; ++token) {
                            const uint32_t input_base = token * columns_ + column_block * 256 + hb * 128;
                            sums[token] += s1 * (float)q1 * (float)input_.GetValue(input_base + index);
                            sums[token] += s2 * (float)q2 * (float)input_.GetValue(input_base + index + 32);
                            sums[token] += s3 * (float)q3 * (float)input_.GetValue(input_base + index + 64);
                            sums[token] += s4 * (float)q4 * (float)input_.GetValue(input_base + index + 96);
                        }
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
    AscendC::GlobalTensor<half> weight_half_;
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> output_;
    uint32_t rows_;
    uint32_t columns_;
    uint32_t blocks_per_row_;
    uint32_t batch_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR weight, GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQ6KMatvec op;
    op.Init(weight, input, output, tiling_data.rows, tiling_data.columns,
            tiling_data.batch);
    op.Process();
}
