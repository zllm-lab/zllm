#include "kernel_operator.h"

// Prefill 先把当前权重解码成 Cube tile，同时把任意长度的 token batch
// 排成 16x16 tile；两个输出只在同一 NPU 图内传给 Cube GEMM。
class KernelQ5DecodeTiles {
public:
    __aicore__ inline KernelQ5DecodeTiles() {}

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
        // 输入打包图使用 16 行哑权重。整 16-token tile 可直接用二维 DMA
        // 从 ND 跨行采集到连续 UB，再一次写成 Cube tile，避免逐元素 GM 访问。
        if (rows_ == cube_tile && batch_ % cube_tile == 0) {
            PackInputDma();
            return;
        }

        const uint32_t blocks = rows_ * blocks_per_row_;
        for (uint32_t block = core; block < blocks; block += cores) {
            DecodeBlock(block);
        }

        if (batch_ % cube_tile == 0) {
            PackInputDma();
            return;
        }

        const uint32_t padded_batch =
            ((batch_ + cube_tile - 1) / cube_tile) * cube_tile;
        const uint32_t elements = padded_batch * columns_;
        for (uint32_t element = core; element < elements; element += cores) {
            const uint32_t tile = element / cube_elements;
            const uint32_t within = element % cube_elements;
            const uint32_t token_tile = tile / column_tiles_;
            const uint32_t column_tile = tile % column_tiles_;
            const uint32_t token = token_tile * cube_tile + within / cube_tile;
            const uint32_t column = column_tile * cube_tile + within % cube_tile;
            input_tiles_.SetValue(element, token < batch_
                ? input_.GetValue(token * columns_ + column) : (half)0.0f);
        }
    }

private:
    __aicore__ inline void PackInputDma()
    {
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
        const uint32_t token_tiles = batch_ / cube_tile;
        const uint32_t tiles = token_tiles * column_tiles_;
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
            const uint32_t token_tile = tile / column_tiles_;
            const uint32_t column_tile = tile % column_tiles_;
            const uint32_t source =
                token_tile * cube_tile * columns_ +
                column_tile * cube_tile;
            const uint32_t local = slot * cube_elements;
            AscendC::DataCopy(tile_local[local], input_[source], gather);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                load_to_store[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                load_to_store[slot]);
            AscendC::DataCopy(
                input_tiles_[tile * cube_elements], tile_local[local],
                cube_elements);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

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
                const uint8_t low = group % 2 == 0 ? packed & 0x0f : packed >> 4;
                const uint8_t high = (high_byte & high_mask) != 0 ? 16 : 0;
                const int32_t quant = (int32_t)low + (int32_t)high;
                const float decoded =
                    d * (float)scale_value * (float)quant -
                    dmin * (float)minimum_value;
                const uint32_t local_column = group * 32 + index;
                const uint32_t column_tile =
                    column_block * (q5_block / cube_tile) +
                    local_column / cube_tile;
                const uint32_t tile =
                    (row / cube_tile) * column_tiles_ + column_tile;
                const uint32_t destination =
                    tile * cube_elements + (row % cube_tile) * cube_tile +
                    local_column % cube_tile;
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
    uint32_t rows_, columns_, batch_, blocks_per_row_, column_tiles_;
};

extern "C" __global__ __aicore__ void q5_decode(
    GM_ADDR weight, GM_ADDR input, GM_ADDR weight_tiles, GM_ADDR input_tiles,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQ5DecodeTiles op;
    op.Init(weight, input, weight_tiles, input_tiles, tiling_data.rows,
            tiling_data.columns, tiling_data.batch);
    op.Process();
}
