#include "kernel_operator.h"

// AIV 负责 Q5_K 解码与 Cube 分块布局；AIC 只消费 FP16 tile，避免在
// dav-l210 的 Cube 核里执行不受支持的标量 GM 访问。
class KernelQ5KDecode {
public:
    __aicore__ inline KernelQ5KDecode() {}

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR input,
                                GM_ADDR weight_tiles, GM_ADDR input_tiles,
                                uint32_t rows, uint32_t columns, uint32_t batch)
    {
        rows_ = rows;
        columns_ = columns;
        batch_ = batch;
        blocks_per_row_ = columns / q5_block;
        column_tiles_ = columns / cube_tile;
        weight_bytes_ = (__gm__ uint8_t*)weight;
        weight_half_.SetGlobalBuffer((__gm__ half*)weight,
                                     rows * blocks_per_row_ * q5_half_elements);
        input_.SetGlobalBuffer((__gm__ half*)input, batch * columns);
        weight_tiles_.SetGlobalBuffer((__gm__ half*)weight_tiles, rows * columns);
        input_tiles_.SetGlobalBuffer((__gm__ half*)input_tiles,
                                     ((batch + cube_tile - 1) / cube_tile) *
                                         cube_tile * columns);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t blocks = rows_ * blocks_per_row_;
        for (uint32_t block = core; block < blocks; block += cores) {
            DecodeBlock(block);
        }

        const uint32_t padded_input_elements =
            ((batch_ + cube_tile - 1) / cube_tile) * cube_tile * columns_;
        for (uint32_t element = core; element < padded_input_elements;
             element += cores) {
            const uint32_t tile = element / cube_elements;
            const uint32_t within_tile = element % cube_elements;
            const uint32_t token_tile = tile / column_tiles_;
            const uint32_t column_tile = tile % column_tiles_;
            const uint32_t token =
                token_tile * cube_tile + within_tile / cube_tile;
            const uint32_t column =
                column_tile * cube_tile + within_tile % cube_tile;
            const half value = token < batch_
                                   ? input_.GetValue(token * columns_ + column)
                                   : (half)0.0f;
            input_tiles_.SetValue(element, value);
        }
    }

private:
    __aicore__ inline void DecodeBlock(uint32_t block)
    {
        const uint32_t row = block / blocks_per_row_;
        const uint32_t column_block = block % blocks_per_row_;
        const uint32_t base = block * q5_bytes;
        const float d = (float)weight_half_.GetValue(block * q5_half_elements);
        const float dmin =
            (float)weight_half_.GetValue(block * q5_half_elements + 1);

        for (uint32_t group = 0; group < 8; ++group) {
            uint8_t scale;
            uint8_t minimum;
            if (group < 4) {
                scale = weight_bytes_[base + 4 + group] & 0x3f;
                minimum = weight_bytes_[base + 8 + group] & 0x3f;
            } else {
                scale = (weight_bytes_[base + 8 + group] & 0x0f) |
                        ((weight_bytes_[base + group] >> 6) << 4);
                minimum = (weight_bytes_[base + 8 + group] >> 4) |
                          ((weight_bytes_[base + 4 + group] >> 6) << 4);
            }
            const uint8_t high_mask = (uint8_t)(1u << group);
            const uint32_t low_base = base + 48 + (group / 2) * 32;
            const int32_t scale_value = (int32_t)scale;
            const int32_t minimum_value = (int32_t)minimum;

            for (uint32_t index = 0; index < 32; ++index) {
                const uint8_t packed = weight_bytes_[low_base + index];
                const uint8_t high_byte = weight_bytes_[base + 16 + index];
                const uint8_t low =
                    group % 2 == 0 ? packed & 0x0f : packed >> 4;
                const uint8_t high =
                    (high_byte & high_mask) != 0 ? 16 : 0;
                const int32_t quant = (int32_t)low + (int32_t)high;
                const float decoded =
                    d * (float)scale_value * (float)quant -
                    dmin * (float)minimum_value;

                const uint32_t local_column = group * 32 + index;
                const uint32_t column_tile =
                    column_block * (q5_block / cube_tile) +
                    local_column / cube_tile;
                const uint32_t lane = local_column % cube_tile;
                const uint32_t tile =
                    (row / cube_tile) * column_tiles_ + column_tile;
                const uint32_t destination =
                    tile * cube_elements + (row % cube_tile) * cube_tile + lane;
                weight_tiles_.SetValue(destination, (half)decoded);
            }
        }
    }

    static constexpr uint32_t cube_tile = 16;
    static constexpr uint32_t cube_elements = cube_tile * cube_tile;
    static constexpr uint32_t q5_block = 256;
    static constexpr uint32_t q5_bytes = 176;
    static constexpr uint32_t q5_half_elements = q5_bytes / sizeof(half);

    __gm__ uint8_t* weight_bytes_;
    AscendC::GlobalTensor<half> weight_half_;
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> weight_tiles_;
    AscendC::GlobalTensor<half> input_tiles_;
    uint32_t rows_;
    uint32_t columns_;
    uint32_t batch_;
    uint32_t blocks_per_row_;
    uint32_t column_tiles_;
};

extern "C" __global__ __aicore__ void q5_decode(
    GM_ADDR weight, GM_ADDR input, GM_ADDR weight_tiles, GM_ADDR input_tiles,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQ5KDecode op;
    op.Init(weight, input, weight_tiles, input_tiles, tiling_data.rows,
            tiling_data.columns, tiling_data.batch);
    op.Process();
}
