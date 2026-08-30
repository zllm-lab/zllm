#include "kernel_operator.h"

// Q5_K packed 权重直接在 AIV 上按 4 行 x 2 block 解码成临时 ND FP16；
// 后续 MatMul 在同一 NPU 图内消费该张量。量化权重始终是图的常驻输入，
// 不依赖磁盘 FP16 sidecar，也不把中间结果返回 CPU。
class KernelQ5DecodeNd {
public:
    __aicore__ inline KernelQ5DecodeNd() {}

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR input,
                                GM_ADDR decoded, GM_ADDR input_tiles,
                                uint32_t rows, uint32_t columns,
                                uint32_t batch)
    {
        rows_ = rows;
        columns_ = columns;
        batch_ = batch;
        blocks_per_row_ = columns / q5_block;
        column_tiles_ = columns / output_tile;
        row_half_elements_ = blocks_per_row_ * q5_bytes / sizeof(half);
        weight_.SetGlobalBuffer((__gm__ half*)weight,
                                rows * row_half_elements_);
        input_.SetGlobalBuffer((__gm__ half*)input, batch * columns);
        decoded_.SetGlobalBuffer((__gm__ half*)decoded, rows * columns);
        input_tiles_.SetGlobalBuffer(
            (__gm__ half*)input_tiles,
            ((batch + output_tile - 1) / output_tile) *
                output_tile * columns);
    }

    __aicore__ inline void Process()
    {
        if (rows_ % output_tile != 0 || columns_ % q5_block != 0 ||
            row_half_elements_ % 16 != 0 ||
            blocks_per_row_ % block_batch != 0) {
            return;
        }

        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> weight_queue;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
        AscendC::TBuf<AscendC::TPosition::VECCALC> byte_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> half_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_buffer;
        pipe.InitBuffer(weight_queue, 1,
                        output_tile * row_half_elements_ * sizeof(half));
        pipe.InitBuffer(output_queue, 1, tile_elements * sizeof(half));
        pipe.InitBuffer(byte_buffer, byte_vectors * tile_elements);
        pipe.InitBuffer(half_buffer,
                        half_vectors * tile_elements * sizeof(half));
        pipe.InitBuffer(offset_buffer,
                        offset_vectors * tile_elements * sizeof(uint16_t));
        AscendC::TBuf<AscendC::TPosition::VECCALC> input_tile_buffer;
        pipe.InitBuffer(input_tile_buffer,
                        2 * cube_elements * sizeof(half));

        auto bytes = byte_buffer.Get<uint8_t>();
        auto high = bytes[0 * tile_elements];
        auto scale = bytes[1 * tile_elements];
        auto minimum = bytes[2 * tile_elements];
        auto temporary = bytes[3 * tile_elements];
        auto q_even_mask = bytes[4 * tile_elements];
        auto q_odd_mask = bytes[5 * tile_elements];
        auto high_mask = bytes[6 * tile_elements];
        auto scale_mask = bytes[7 * tile_elements];
        auto minimum_first_mask = bytes[8 * tile_elements];
        auto second_half_mask = bytes[9 * tile_elements];

        auto halves = half_buffer.Get<half>();
        auto high_half = halves[0 * tile_elements];
        auto high_factor = halves[1 * tile_elements];
        auto scale_half = halves[2 * tile_elements];
        auto minimum_half = halves[3 * tile_elements];
        auto d = halves[4 * tile_elements];
        auto dmin = halves[5 * tile_elements];

        auto index = offset_buffer.Get<uint16_t>();
        auto offsets = index[0 * tile_elements];
        auto code_indices = index[1 * tile_elements];
        auto high_indices = index[2 * tile_elements];
        auto scale_indices = index[3 * tile_elements];
        auto minimum_indices = index[4 * tile_elements];
        auto scale_high_indices = index[5 * tile_elements];
        auto minimum_high_indices = index[6 * tile_elements];
        PrepareIndices(offsets, code_indices, high_indices, scale_indices,
                       minimum_indices, scale_high_indices,
                       minimum_high_indices, q_even_mask, q_odd_mask,
                       high_mask, scale_mask, minimum_first_mask,
                       second_half_mask, high_factor);

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t row_tiles = rows_ / output_tile;
        for (uint32_t row_tile = core; row_tile < row_tiles;
             row_tile += cores) {
            const uint32_t first_row = row_tile * output_tile;
            auto packed_tile = weight_queue.AllocTensor<half>();
            AscendC::DataCopy(
                packed_tile, weight_[first_row * row_half_elements_],
                output_tile * row_half_elements_);
            weight_queue.EnQue(packed_tile);
            packed_tile = weight_queue.DeQue<half>();

            for (uint32_t row_base = 0; row_base < output_tile;
                 row_base += row_batch) {
                auto packed_half = packed_tile[row_base * row_half_elements_];
                auto packed = packed_half.ReinterpretCast<uint8_t>();
                for (uint32_t block = 0; block < blocks_per_row_;
                     block += block_batch) {
                    auto output = output_queue.AllocTensor<half>();
                    const uint32_t base = block * q5_bytes;
                    AscendC::Gather(d, packed_half[base / sizeof(half)],
                                    offsets, 0, tile_elements);
                    AscendC::Gather(
                        dmin, packed_half[base / sizeof(half) + 1], offsets,
                        0, tile_elements);
                    DecodeCodesTile(
                        packed, base, output, high, temporary, q_even_mask,
                        q_odd_mask, high_mask, high_half, high_factor,
                        code_indices, high_indices);
                    DecodeScaleMinTile(
                        packed, base, scale, minimum, temporary, scale_mask,
                        minimum_first_mask, second_half_mask, scale_indices,
                        minimum_indices, scale_high_indices,
                        minimum_high_indices);
                    AscendC::Cast(scale_half, scale,
                                  AscendC::RoundMode::CAST_NONE,
                                  tile_elements);
                    AscendC::Cast(minimum_half, minimum,
                                  AscendC::RoundMode::CAST_NONE,
                                  tile_elements);
                    AscendC::Mul(output, output, scale_half, tile_elements);
                    AscendC::Mul(output, output, d, tile_elements);
                    AscendC::Mul(minimum_half, minimum_half, dmin,
                                 tile_elements);
                    AscendC::Sub(output, output, minimum_half,
                                 tile_elements);
                    output_queue.EnQue(output);
                    output = output_queue.DeQue<half>();
                    for (uint32_t local_row = 0; local_row < row_batch;
                         ++local_row) {
                        const uint32_t destination =
                            (first_row + row_base + local_row) * columns_ +
                            block * q5_block;
                        AscendC::DataCopy(
                            decoded_[destination],
                            output[local_row * row_span], row_span);
                    }
                    output_queue.FreeTensor(output);
                }
            }
            weight_queue.FreeTensor(packed_tile);
        }

        // 整 16-token tile 用二维 DMA 打包。decode 的 batch=1 尾 tile
        // 沿用已在 9010 验证过的标量补零路径，避免 UB 上跨流水同步。
        if (batch_ % output_tile == 0) {
            auto tile_local = input_tile_buffer.Get<half>();
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
                output_tile, 1, columns_ / output_tile - 1, 0);
            const uint32_t token_tiles = batch_ / output_tile;
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
                    token_tile * output_tile * columns_ +
                    column_tile * output_tile;
                const uint32_t local = slot * cube_elements;
                AscendC::DataCopy(tile_local[local], input_[source], gather);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                    load_to_store[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                    load_to_store[slot]);
                AscendC::DataCopy(input_tiles_[tile * cube_elements],
                                  tile_local[local], cube_elements);
            }
            return;
        }

        const uint32_t padded_batch =
            ((batch_ + output_tile - 1) / output_tile) * output_tile;
        const uint32_t elements = padded_batch * columns_;
        for (uint32_t element = core; element < elements;
             element += cores) {
            const uint32_t tile = element / cube_elements;
            const uint32_t within = element % cube_elements;
            const uint32_t token_tile = tile / column_tiles_;
            const uint32_t column_tile = tile % column_tiles_;
            const uint32_t token =
                token_tile * output_tile + within / output_tile;
            const uint32_t column =
                column_tile * output_tile + within % output_tile;
            input_tiles_.SetValue(
                element, token < batch_
                    ? input_.GetValue(token * columns_ + column)
                    : (half)0.0f);
        }
    }

private:
    __aicore__ inline void PrepareIndices(
        AscendC::LocalTensor<uint16_t> offsets,
        AscendC::LocalTensor<uint16_t> code_indices,
        AscendC::LocalTensor<uint16_t> high_indices,
        AscendC::LocalTensor<uint16_t> scale_indices,
        AscendC::LocalTensor<uint16_t> minimum_indices,
        AscendC::LocalTensor<uint16_t> scale_high_indices,
        AscendC::LocalTensor<uint16_t> minimum_high_indices,
        AscendC::LocalTensor<uint8_t> q_even_mask,
        AscendC::LocalTensor<uint8_t> q_odd_mask,
        AscendC::LocalTensor<uint8_t> high_mask,
        AscendC::LocalTensor<uint8_t> scale_mask,
        AscendC::LocalTensor<uint8_t> minimum_first_mask,
        AscendC::LocalTensor<uint8_t> second_half_mask,
        AscendC::LocalTensor<half> high_factor)
    {
        const uint32_t row_bytes = row_half_elements_ * sizeof(half);
        for (uint32_t local_row = 0; local_row < row_batch; ++local_row) {
            const uint32_t row_offset = local_row * row_bytes;
            const uint32_t row_element = local_row * row_span;
            for (uint32_t block_lane = 0; block_lane < block_batch;
                 ++block_lane) {
                const uint32_t lane_offset =
                    row_offset + block_lane * q5_bytes;
                const uint32_t lane_element =
                    row_element + block_lane * q5_block;
                AscendC::Duplicate(offsets[lane_element],
                                   static_cast<uint16_t>(lane_offset),
                                   q5_block);
                for (uint32_t group = 0; group < 8; ++group) {
                    const uint32_t element =
                        lane_element + group * group_elements;
                    AscendC::CreateVecIndex(
                        code_indices[element].ReinterpretCast<int16_t>(),
                        static_cast<int16_t>(
                            lane_offset + (group / 2) * group_elements),
                        group_elements);
                    AscendC::CreateVecIndex(
                        high_indices[element].ReinterpretCast<int16_t>(),
                        static_cast<int16_t>(lane_offset), group_elements);
                    AscendC::Duplicate(
                        scale_indices[element],
                        static_cast<uint16_t>(
                            lane_offset + (group < 4 ? group : group + 4)),
                        group_elements);
                    AscendC::Duplicate(
                        minimum_indices[element],
                        static_cast<uint16_t>(lane_offset + group + 4),
                        group_elements);
                    AscendC::Duplicate(
                        scale_high_indices[element],
                        static_cast<uint16_t>(
                            lane_offset + (group < 4 ? 0 : group - 4)),
                        group_elements);
                    AscendC::Duplicate(
                        minimum_high_indices[element],
                        static_cast<uint16_t>(
                            lane_offset + (group < 4 ? 0 : group)),
                        group_elements);
                    const bool even = (group & 1) == 0;
                    const bool first = group < 4;
                    AscendC::Duplicate(
                        q_even_mask[element],
                        static_cast<uint8_t>(even ? 0x0f : 0),
                        group_elements);
                    AscendC::Duplicate(
                        q_odd_mask[element],
                        static_cast<uint8_t>(even ? 0 : 0x0f),
                        group_elements);
                    AscendC::Duplicate(high_mask[element],
                                       static_cast<uint8_t>(1u << group),
                                       group_elements);
                    AscendC::Duplicate(
                        scale_mask[element],
                        static_cast<uint8_t>(first ? 0x3f : 0x0f),
                        group_elements);
                    AscendC::Duplicate(
                        minimum_first_mask[element],
                        static_cast<uint8_t>(first ? 0x3f : 0),
                        group_elements);
                    AscendC::Duplicate(
                        second_half_mask[element],
                        static_cast<uint8_t>(first ? 0 : 0xff),
                        group_elements);
                    AscendC::Duplicate(
                        high_factor[element],
                        static_cast<half>(
                            group == 0 ? 16.0f :
                            group == 1 ? 8.0f :
                            group == 2 ? 4.0f :
                            group == 3 ? 2.0f :
                            group == 4 ? 1.0f :
                            group == 5 ? 0.5f :
                            group == 6 ? 0.25f : 0.125f),
                        group_elements);
                }
            }
        }
    }

    __aicore__ inline void DecodeCodesTile(
        const AscendC::LocalTensor<uint8_t>& packed, uint32_t base,
        AscendC::LocalTensor<half> q_half,
        AscendC::LocalTensor<uint8_t> high,
        AscendC::LocalTensor<uint8_t> temporary,
        AscendC::LocalTensor<uint8_t> q_even_mask,
        AscendC::LocalTensor<uint8_t> q_odd_mask,
        AscendC::LocalTensor<uint8_t> high_mask,
        AscendC::LocalTensor<half> high_half,
        AscendC::LocalTensor<half> high_factor,
        AscendC::LocalTensor<uint16_t> code_indices,
        AscendC::LocalTensor<uint16_t> high_indices)
    {
        auto q = temporary;
        AscendC::Gather(q, packed[base + 48], code_indices, 0,
                        tile_elements);
        // high 暂存低 nibble 的右移结果，随后会被 qh Gather 覆盖。
        AscendC::ShiftRight<uint8_t, false>(
            high, q, static_cast<uint8_t>(4), tile_elements);
        AscendC::And(q, q, q_even_mask, tile_elements);
        AscendC::And(high, high, q_odd_mask, tile_elements);
        AscendC::Or(q, q, high, tile_elements);

        AscendC::Gather(high, packed[base + 16], high_indices, 0,
                        tile_elements);
        AscendC::And(high, high, high_mask, tile_elements);
        AscendC::Cast(q_half, q, AscendC::RoundMode::CAST_NONE,
                      tile_elements);
        AscendC::Cast(high_half, high, AscendC::RoundMode::CAST_NONE,
                      tile_elements);
        AscendC::Mul(high_half, high_half, high_factor, tile_elements);
        AscendC::Add(q_half, q_half, high_half, tile_elements);
    }

    __aicore__ inline void DecodeScaleMinTile(
        const AscendC::LocalTensor<uint8_t>& packed, uint32_t base,
        AscendC::LocalTensor<uint8_t> scale,
        AscendC::LocalTensor<uint8_t> minimum,
        AscendC::LocalTensor<uint8_t> temporary,
        AscendC::LocalTensor<uint8_t> scale_mask,
        AscendC::LocalTensor<uint8_t> minimum_first_mask,
        AscendC::LocalTensor<uint8_t> second_half_mask,
        AscendC::LocalTensor<uint16_t> scale_indices,
        AscendC::LocalTensor<uint16_t> minimum_indices,
        AscendC::LocalTensor<uint16_t> scale_high_indices,
        AscendC::LocalTensor<uint16_t> minimum_high_indices)
    {
        AscendC::Gather(scale, packed[base + 4], scale_indices, 0,
                        tile_elements);
        AscendC::And(scale, scale, scale_mask, tile_elements);
        AscendC::Gather(temporary, packed[base + 4], scale_high_indices, 0,
                        tile_elements);
        AscendC::ShiftRight<uint8_t, false>(
            temporary, temporary, static_cast<uint8_t>(6), tile_elements);
        AscendC::ShiftLeft<uint8_t, false>(
            temporary, temporary, static_cast<uint8_t>(4), tile_elements);
        AscendC::And(temporary, temporary, second_half_mask, tile_elements);
        AscendC::Or(scale, scale, temporary, tile_elements);

        AscendC::Gather(minimum, packed[base + 4], minimum_indices, 0,
                        tile_elements);
        AscendC::ShiftRight<uint8_t, false>(
            temporary, minimum, static_cast<uint8_t>(4), tile_elements);
        AscendC::And(minimum, minimum, minimum_first_mask, tile_elements);
        AscendC::And(temporary, temporary, second_half_mask, tile_elements);
        AscendC::Or(minimum, minimum, temporary, tile_elements);
        AscendC::Gather(temporary, packed[base + 4], minimum_high_indices, 0,
                        tile_elements);
        AscendC::ShiftRight<uint8_t, false>(
            temporary, temporary, static_cast<uint8_t>(6), tile_elements);
        AscendC::ShiftLeft<uint8_t, false>(
            temporary, temporary, static_cast<uint8_t>(4), tile_elements);
        AscendC::And(temporary, temporary, second_half_mask, tile_elements);
        AscendC::Or(minimum, minimum, temporary, tile_elements);
    }

    static constexpr uint32_t q5_block = 256;
    static constexpr uint32_t q5_bytes = 176;
    static constexpr uint32_t group_elements = 32;
    static constexpr uint32_t output_tile = 16;
    static constexpr uint32_t row_batch = 4;
    static constexpr uint32_t block_batch = 2;
    static constexpr uint32_t row_span = block_batch * q5_block;
    static constexpr uint32_t tile_elements = row_batch * row_span;
    static constexpr uint32_t cube_elements = output_tile * output_tile;
    static constexpr uint32_t byte_vectors = 10;
    static constexpr uint32_t half_vectors = 6;
    static constexpr uint32_t offset_vectors = 7;

    AscendC::GlobalTensor<half> weight_;
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> decoded_;
    AscendC::GlobalTensor<half> input_tiles_;
    uint32_t rows_ = 0;
    uint32_t columns_ = 0;
    uint32_t batch_ = 0;
    uint32_t blocks_per_row_ = 0;
    uint32_t column_tiles_ = 0;
    uint32_t row_half_elements_ = 0;
};

extern "C" __global__ __aicore__ void q5_decode(
    GM_ADDR weight, GM_ADDR input, GM_ADDR weight_tiles, GM_ADDR input_tiles,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQ5DecodeNd op;
    op.Init(weight, input, weight_tiles, input_tiles, tiling_data.rows,
            tiling_data.columns, tiling_data.batch);
    op.Process();
}
