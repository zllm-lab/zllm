#include "kernel_operator.h"

// Decode 的 batch=1 直接消费 GGUF Q5_K。packed row 和 input 由 TQue 搬到
// UB；nibble/high-bit、scale/min 解包、F32 dot 与归约全部留在 V 流水，
// 不生成完整 FP16 权重，也不混用 UB 标量读取。
class KernelQ5KGemv {
public:
    __aicore__ inline KernelQ5KGemv() {}

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR input, GM_ADDR output,
                                uint32_t rows, uint32_t columns,
                                uint32_t batch)
    {
        rows_ = rows;
        columns_ = columns;
        batch_ = batch;
        blocks_per_row_ = columns / q5_block;
        row_half_elements_ = blocks_per_row_ * q5_bytes / sizeof(half);
        weight_.SetGlobalBuffer((__gm__ half*)weight,
                                rows * row_half_elements_);
        input_.SetGlobalBuffer((__gm__ half*)input, columns);
        output_.SetGlobalBuffer((__gm__ half*)output, rows);
    }

    __aicore__ inline void Process()
    {
        if (batch_ != 1 || rows_ % output_tile != 0 ||
            row_half_elements_ % 16 != 0 || columns_ % 16 != 0 ||
            blocks_per_row_ % block_batch != 0 ||
            block_batch % reduction_batch != 0) {
            return;
        }

        if constexpr (diagnostic_stage != 0) {
            ProcessDiagnostic();
            return;
        }

        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> weight_queue;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
        AscendC::TBuf<AscendC::TPosition::VECCALC> byte_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> half_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> float_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_buffer;
        const uint32_t partials_per_row =
            blocks_per_row_ / reduction_batch;
        const uint32_t block_partial_elements =
            row_batch * partials_per_row * accumulator_stride;
        const uint32_t reduce_elements =
            block_partial_elements +
            output_tile * accumulator_stride + group_elements;
        const uint32_t raw_offset_bytes =
            7 * tile_elements * sizeof(uint16_t) +
            output_tile * sizeof(uint32_t) +
            partials_per_row * sizeof(uint32_t);
        const uint32_t aligned_offset_bytes =
            (raw_offset_bytes + vector_bytes - 1) & ~(vector_bytes - 1);
        pipe.InitBuffer(input_queue, 1, columns_ * sizeof(half));
        pipe.InitBuffer(weight_queue, 1,
                        output_tile * row_half_elements_ * sizeof(half));
        pipe.InitBuffer(output_queue, 1, output_tile * sizeof(half));
        pipe.InitBuffer(byte_buffer, byte_vectors * tile_elements);
        pipe.InitBuffer(half_buffer,
                        half_vectors * tile_elements * sizeof(half));
        pipe.InitBuffer(float_buffer,
                        (row_span + group_elements) * sizeof(float));
        pipe.InitBuffer(reduce_buffer, reduce_elements * sizeof(float));
        pipe.InitBuffer(offset_buffer, aligned_offset_bytes);

        auto input = input_queue.AllocTensor<half>();
        AscendC::DataCopy(input, input_, columns_);
        input_queue.EnQue(input);
        input = input_queue.DeQue<half>();

        auto bytes = byte_buffer.Get<uint8_t>();
        auto q = bytes[0 * tile_elements];
        auto high = bytes[1 * tile_elements];
        auto scale = bytes[2 * tile_elements];
        auto minimum = bytes[3 * tile_elements];
        auto temporary = bytes[4 * tile_elements];
        auto q_even_mask = bytes[5 * tile_elements];
        auto q_odd_mask = bytes[6 * tile_elements];
        auto high_mask = bytes[7 * tile_elements];
        auto scale_mask = bytes[8 * tile_elements];
        auto minimum_first_mask = bytes[9 * tile_elements];
        auto second_half_mask = bytes[10 * tile_elements];

        auto halves = half_buffer.Get<half>();
        auto q_half = halves[0 * tile_elements];
        auto high_half = halves[1 * tile_elements];
        auto high_factor = halves[2 * tile_elements];
        auto scale_half = halves[3 * tile_elements];
        auto minimum_half = halves[4 * tile_elements];
        auto d = halves[5 * tile_elements];
        auto dmin = halves[6 * tile_elements];

        auto floats = float_buffer.Get<float>();
        auto products = floats[0];
        auto compact_results = floats[row_span];

        auto reductions = reduce_buffer.Get<float>();
        auto block_partials = reductions[0];
        auto accumulators = reductions[block_partial_elements];
        auto reduction =
            reductions[block_partial_elements +
                       output_tile * accumulator_stride];

        auto index = offset_buffer.Get<uint16_t>();
        auto offsets = index[0 * tile_elements];
        auto code_indices = index[1 * tile_elements];
        auto high_indices = index[2 * tile_elements];
        auto scale_indices = index[3 * tile_elements];
        auto minimum_indices = index[4 * tile_elements];
        auto scale_high_indices = index[5 * tile_elements];
        auto minimum_high_indices = index[6 * tile_elements];
        auto result_indices =
            index[7 * tile_elements].ReinterpretCast<uint32_t>();
        auto partial_indices =
            index[7 * tile_elements + 2 * output_tile]
                .ReinterpretCast<uint32_t>();
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
        auto signed_result_indices =
            result_indices.ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(signed_result_indices,
                                static_cast<int32_t>(0), output_tile);
        AscendC::Muls(signed_result_indices, signed_result_indices,
                      static_cast<int32_t>(accumulator_stride *
                                           sizeof(float)),
                      output_tile);
        auto signed_partial_indices =
            partial_indices.ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(signed_partial_indices,
                                static_cast<int32_t>(0), partials_per_row);
        AscendC::Muls(signed_partial_indices, signed_partial_indices,
                      static_cast<int32_t>(accumulator_stride *
                                           sizeof(float)),
                      partials_per_row);

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t row_tiles = rows_ / output_tile;
        for (uint32_t row_tile = core; row_tile < row_tiles;
             row_tile += cores) {
            auto output = output_queue.AllocTensor<half>();
            const uint32_t first_row = row_tile * output_tile;
            auto packed_tile = weight_queue.AllocTensor<half>();
            AscendC::DataCopy(
                packed_tile, weight_[first_row * row_half_elements_],
                output_tile * row_half_elements_);
            weight_queue.EnQue(packed_tile);
            packed_tile = weight_queue.DeQue<half>();
            for (uint32_t row_base = 0; row_base < output_tile;
                 row_base += row_batch) {
                auto packed_half =
                    packed_tile[row_base * row_half_elements_];
                auto packed = packed_half.ReinterpretCast<uint8_t>();
                for (uint32_t block = 0; block < blocks_per_row_;
                     block += block_batch) {
                    const uint32_t base = block * q5_bytes;
                    AscendC::Gather(d, packed_half[base / sizeof(half)],
                                    offsets, 0, tile_elements);
                    AscendC::Gather(
                        dmin, packed_half[base / sizeof(half) + 1],
                        offsets, 0, tile_elements);
                    DecodeCodesTile(
                        packed, base, q, high, temporary, q_even_mask,
                        q_odd_mask, high_mask, q_half, high_half,
                        high_factor, code_indices, high_indices);
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
                    AscendC::Mul(q_half, q_half, scale_half, tile_elements);
                    AscendC::Mul(q_half, q_half, d, tile_elements);
                    AscendC::Mul(minimum_half, minimum_half, dmin,
                                 tile_elements);
                    AscendC::Sub(q_half, q_half, minimum_half,
                                 tile_elements);
                    const uint32_t column = block * q5_block;
                    for (uint32_t local_row = 0; local_row < row_batch;
                         ++local_row) {
                        const uint32_t element = local_row * row_span;
                        // high_half 在 Q5 high-bit 合并后即可复用为 FP16
                        // product；一批相邻 Q5 block 合并为一次 F32 归约，
                        // 减少 Cast/Reduce 和后续 partial gather 的数量。
                        AscendC::Mul(high_half[element], input[column],
                                     q_half[element], row_span);
                        for (uint32_t reduction_lane = 0;
                             reduction_lane < reductions_per_batch;
                             ++reduction_lane) {
                            const uint32_t lane_element =
                                element + reduction_lane * reduction_span;
                            auto partial = block_partials[
                                (local_row * partials_per_row +
                                 block / reduction_batch + reduction_lane) *
                                accumulator_stride];
                            AscendC::Cast(products, high_half[lane_element],
                                          AscendC::RoundMode::CAST_NONE,
                                          reduction_span);
                            AscendC::ReduceSum(partial, products, reduction,
                                               reduction_span);
                        }
                    }
                }
                for (uint32_t local_row = 0; local_row < row_batch;
                     ++local_row) {
                    auto row_partials = block_partials[
                        local_row * partials_per_row *
                        accumulator_stride];
                    AscendC::Gather(compact_results, row_partials,
                                    partial_indices, 0, partials_per_row);
                    auto accumulator = accumulators[
                        (row_base + local_row) * accumulator_stride];
                    AscendC::ReduceSum(accumulator, compact_results,
                                       reduction, partials_per_row);
                }
            }
            weight_queue.FreeTensor(packed_tile);
            AscendC::Gather(compact_results, accumulators, result_indices, 0,
                            output_tile);
            AscendC::Cast(output, compact_results,
                          AscendC::RoundMode::CAST_NONE, output_tile);
            output_queue.EnQue(output);
            output = output_queue.DeQue<half>();
            AscendC::DataCopy(output_[first_row], output, output_tile);
            output_queue.FreeTensor(output);
        }
        input_queue.FreeTensor(input);
    }

private:
    __aicore__ inline void ProcessDiagnostic()
    {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> weight_queue;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
        AscendC::TBuf<AscendC::TPosition::VECCALC> byte_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> half_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_buffer;
        pipe.InitBuffer(input_queue, 1, columns_ * sizeof(half));
        pipe.InitBuffer(weight_queue, 1,
                        row_half_elements_ * sizeof(half));
        pipe.InitBuffer(output_queue, 1, output_tile * sizeof(half));
        pipe.InitBuffer(byte_buffer, byte_vectors * vector_bytes);
        pipe.InitBuffer(half_buffer, 2 * group_elements * sizeof(half));
        pipe.InitBuffer(offset_buffer, diagnostic_offset_bytes);

        auto input = input_queue.AllocTensor<half>();
        AscendC::DataCopy(input, input_, columns_);
        input_queue.EnQue(input);
        input = input_queue.DeQue<half>();

        auto byte_workspace = byte_buffer.Get<uint8_t>();
        auto q = byte_workspace[0 * group_elements];
        auto high = byte_workspace[1 * group_elements];
        auto scale = byte_workspace[2 * group_elements];
        auto minimum = byte_workspace[3 * group_elements];
        auto temporary = byte_workspace[4 * group_elements];
        auto mask = byte_workspace[5 * group_elements];
        auto half_workspace = half_buffer.Get<half>();
        auto d = half_workspace[0 * group_elements];
        auto dmin = half_workspace[1 * group_elements];
        auto offset_workspace = offset_buffer.Get<uint16_t>();
        auto offsets = offset_workspace[0];
        auto indices = offset_workspace[group_elements];
        auto signed_indices = indices.ReinterpretCast<int16_t>();
        AscendC::Duplicate(offsets, static_cast<uint16_t>(0),
                           group_elements);
        AscendC::CreateVecIndex(signed_indices, static_cast<int16_t>(0),
                                group_elements);

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t row_tiles = rows_ / output_tile;
        for (uint32_t row_tile = core; row_tile < row_tiles;
             row_tile += cores) {
            auto output = output_queue.AllocTensor<half>();
            const uint32_t first_row = row_tile * output_tile;
            for (uint32_t row_in_tile = 0; row_in_tile < output_tile;
                 ++row_in_tile) {
                auto packed = weight_queue.AllocTensor<half>();
                AscendC::DataCopy(
                    packed,
                    weight_[(first_row + row_in_tile) * row_half_elements_],
                    row_half_elements_);
                weight_queue.EnQue(packed);
                packed = weight_queue.DeQue<half>();
                auto packed_bytes = packed.ReinterpretCast<uint8_t>();
                for (uint32_t block = 0; block < blocks_per_row_; ++block) {
                    const uint32_t base = block * q5_bytes;
                    AscendC::Gather(d, packed[base / sizeof(half)], offsets,
                                    0, group_elements);
                    AscendC::Gather(dmin,
                                    packed[base / sizeof(half) + 1], offsets,
                                    0, group_elements);
                    for (uint32_t group = 0; group < 8; ++group) {
                        DecodeCodes(packed_bytes, base, group, q, high, mask,
                                    indices);
                        DecodeScaleMin(packed_bytes, base, group, scale,
                                       minimum, temporary, mask, offsets);
                    }
                }
                weight_queue.FreeTensor(packed);
            }
            AscendC::Duplicate(output, static_cast<half>(0.0f), output_tile);
            output_queue.EnQue(output);
            output = output_queue.DeQue<half>();
            AscendC::DataCopy(output_[first_row], output, output_tile);
            output_queue.FreeTensor(output);
        }
        input_queue.FreeTensor(input);
    }

    __aicore__ inline void DecodeCodesTile(
        const AscendC::LocalTensor<uint8_t>& packed, uint32_t base,
        AscendC::LocalTensor<uint8_t> q,
        AscendC::LocalTensor<uint8_t> high,
        AscendC::LocalTensor<uint8_t> temporary,
        AscendC::LocalTensor<uint8_t> q_even_mask,
        AscendC::LocalTensor<uint8_t> q_odd_mask,
        AscendC::LocalTensor<uint8_t> high_mask,
        AscendC::LocalTensor<half> q_half,
        AscendC::LocalTensor<half> high_half,
        AscendC::LocalTensor<half> high_factor,
        AscendC::LocalTensor<uint16_t> code_indices,
        AscendC::LocalTensor<uint16_t> high_indices)
    {
        AscendC::Gather(q, packed[base + 48], code_indices, 0,
                        tile_elements);
        AscendC::ShiftRight<uint8_t, false>(
            temporary, q, static_cast<uint8_t>(4), tile_elements);
        AscendC::And(q, q, q_even_mask, tile_elements);
        AscendC::And(temporary, temporary, q_odd_mask, tile_elements);
        AscendC::Or(q, q, temporary, tile_elements);

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

    __aicore__ inline void DecodeCodesBlock(
        const AscendC::LocalTensor<uint8_t>& packed, uint32_t base,
        AscendC::LocalTensor<uint8_t> q,
        AscendC::LocalTensor<uint8_t> high,
        AscendC::LocalTensor<uint8_t> high_mask,
        AscendC::LocalTensor<uint8_t> low_mask,
        AscendC::LocalTensor<uint16_t> code_indices,
        AscendC::LocalTensor<uint16_t> high_indices)
    {
        // 一个 Gather 同时展开 8 个 group 的低 nibble 源；奇数组先右移，
        // 再统一保留低 4 bit。索引是相对 packed[base + 48] 的字节偏移。
        AscendC::Gather(q, packed[base + 48], code_indices, 0, q5_block);
        for (uint32_t group = 1; group < 8; group += 2) {
            const uint32_t offset = group * group_elements;
            AscendC::ShiftRight<uint8_t, false>(
                q[offset], q[offset], static_cast<uint8_t>(4),
                group_elements);
        }
        AscendC::And(q, q, low_mask, q5_block);

        // qh 的同一 32 字节被 8 个 group 复用。先一次 Gather 复制成 256
        // 元素，再按 group mask/shift，把对应 bit 统一移动到 bit 4。
        AscendC::Gather(high, packed[base + 16], high_indices, 0, q5_block);
        AscendC::And(high, high, high_mask, q5_block);
        for (uint32_t group = 0; group < 8; ++group) {
            const uint32_t offset = group * group_elements;
            if (group < 4) {
                AscendC::ShiftLeft<uint8_t, false>(
                    high[offset], high[offset],
                    static_cast<uint8_t>(4 - group), group_elements);
            } else if (group > 4) {
                AscendC::ShiftRight<uint8_t, false>(
                    high[offset], high[offset],
                    static_cast<uint8_t>(group - 4), group_elements);
            }
        }
        AscendC::Or(q, q, high, q5_block);
    }

    __aicore__ inline void DecodeScaleMinBlock(
        const AscendC::LocalTensor<uint8_t>& packed, uint32_t base,
        AscendC::LocalTensor<uint8_t> scale,
        AscendC::LocalTensor<uint8_t> minimum,
        AscendC::LocalTensor<uint8_t> temporary,
        AscendC::LocalTensor<uint8_t> low_mask,
        AscendC::LocalTensor<uint16_t> scale_indices,
        AscendC::LocalTensor<uint16_t> minimum_indices,
        AscendC::LocalTensor<uint16_t> scale_high_indices,
        AscendC::LocalTensor<uint16_t> minimum_high_indices)
    {
        constexpr uint32_t half_block = q5_block / 2;
        // scales[0..11] 只有 12 字节且 block 起点交替落在 16 字节边界；
        // Gather 同时解决非对齐读取和每个标量复制 32 次的问题。
        AscendC::Gather(scale, packed[base + 4], scale_indices, 0,
                        q5_block);
        AscendC::Duplicate(temporary, static_cast<uint8_t>(0x3f),
                           half_block);
        AscendC::And(scale, scale, temporary, half_block);
        AscendC::And(scale[half_block], scale[half_block], low_mask,
                     half_block);
        AscendC::Gather(temporary[half_block], packed[base + 4],
                        scale_high_indices, 0, half_block);
        AscendC::ShiftRight<uint8_t, false>(
            temporary[half_block], temporary[half_block],
            static_cast<uint8_t>(6), half_block);
        AscendC::ShiftLeft<uint8_t, false>(
            temporary[half_block], temporary[half_block],
            static_cast<uint8_t>(4), half_block);
        AscendC::Or(scale[half_block], scale[half_block],
                    temporary[half_block], half_block);

        AscendC::Gather(minimum, packed[base + 4], minimum_indices, 0,
                        q5_block);
        AscendC::Duplicate(temporary, static_cast<uint8_t>(0x3f),
                           half_block);
        AscendC::And(minimum, minimum, temporary, half_block);
        AscendC::ShiftRight<uint8_t, false>(
            minimum[half_block], minimum[half_block],
            static_cast<uint8_t>(4), half_block);
        AscendC::Gather(temporary[half_block], packed[base + 4],
                        minimum_high_indices, 0, half_block);
        AscendC::ShiftRight<uint8_t, false>(
            temporary[half_block], temporary[half_block],
            static_cast<uint8_t>(6), half_block);
        AscendC::ShiftLeft<uint8_t, false>(
            temporary[half_block], temporary[half_block],
            static_cast<uint8_t>(4), half_block);
        AscendC::Or(minimum[half_block], minimum[half_block],
                    temporary[half_block], half_block);
    }

    __aicore__ inline void DecodeCodes(
        const AscendC::LocalTensor<uint8_t>& packed, uint32_t base,
        uint32_t group, AscendC::LocalTensor<uint8_t> q,
        AscendC::LocalTensor<uint8_t> high,
        AscendC::LocalTensor<uint8_t> mask,
        AscendC::LocalTensor<uint16_t> indices)
    {
        const uint32_t low_base = base + 48 + (group / 2) * group_elements;
        // dav-l210 的普通向量位运算要求 UB 源地址 32 字节对齐；GGUF
        // Q5_K 的 176-byte block 会让字段交替落在 16-byte 边界。先用
        // Gather 搬到对齐 workspace，否则离线编译成功、真机报 hwts 0x2。
        AscendC::Gather(q, packed[low_base], indices, 0, group_elements);
        if ((group & 1) == 0) {
            AscendC::Duplicate(mask, static_cast<uint8_t>(0x0f),
                               group_elements);
            AscendC::And(q, q, mask, group_elements);
        } else {
            AscendC::ShiftRight<uint8_t, false>(
                q, q, static_cast<uint8_t>(4), group_elements);
        }
        AscendC::Gather(high, packed[base + 16], indices, 0,
                        group_elements);
        AscendC::Duplicate(mask, static_cast<uint8_t>(1u << group),
                           group_elements);
        AscendC::And(high, high, mask, group_elements);
        if (group < 4) {
            AscendC::ShiftLeft<uint8_t, false>(
                high, high, static_cast<uint8_t>(4 - group),
                group_elements);
        } else if (group > 4) {
            AscendC::ShiftRight<uint8_t, false>(
                high, high, static_cast<uint8_t>(group - 4),
                group_elements);
        }
        AscendC::Or(q, q, high, group_elements);
    }

    __aicore__ inline void DecodeScaleMin(
        const AscendC::LocalTensor<uint8_t>& packed, uint32_t base,
        uint32_t group, AscendC::LocalTensor<uint8_t> scale,
        AscendC::LocalTensor<uint8_t> minimum,
        AscendC::LocalTensor<uint8_t> temporary,
        AscendC::LocalTensor<uint8_t> mask,
        AscendC::LocalTensor<uint16_t> offsets)
    {
        if (group < 4) {
            AscendC::Gather(scale, packed[base + 4 + group], offsets, 0,
                            group_elements);
            AscendC::Gather(minimum, packed[base + 8 + group], offsets, 0,
                            group_elements);
            AscendC::Duplicate(mask, static_cast<uint8_t>(0x3f),
                               group_elements);
            AscendC::And(scale, scale, mask, group_elements);
            AscendC::And(minimum, minimum, mask, group_elements);
            return;
        }

        AscendC::Gather(scale, packed[base + 8 + group], offsets, 0,
                        group_elements);
        AscendC::Gather(temporary, packed[base + group], offsets, 0,
                        group_elements);
        AscendC::Duplicate(mask, static_cast<uint8_t>(0x0f),
                           group_elements);
        AscendC::And(scale, scale, mask, group_elements);
        AscendC::ShiftRight<uint8_t, false>(
            temporary, temporary, static_cast<uint8_t>(6), group_elements);
        AscendC::ShiftLeft<uint8_t, false>(
            temporary, temporary, static_cast<uint8_t>(4), group_elements);
        AscendC::Or(scale, scale, temporary, group_elements);

        AscendC::Gather(minimum, packed[base + 8 + group], offsets, 0,
                        group_elements);
        AscendC::Gather(temporary, packed[base + 4 + group], offsets, 0,
                        group_elements);
        AscendC::ShiftRight<uint8_t, false>(
            minimum, minimum, static_cast<uint8_t>(4), group_elements);
        AscendC::ShiftRight<uint8_t, false>(
            temporary, temporary, static_cast<uint8_t>(6), group_elements);
        AscendC::ShiftLeft<uint8_t, false>(
            temporary, temporary, static_cast<uint8_t>(4), group_elements);
        AscendC::Or(minimum, minimum, temporary, group_elements);
    }

    static constexpr uint32_t q5_block = 256;
    static constexpr uint32_t q5_bytes = 176;
    static constexpr uint32_t group_elements = 32;
    static constexpr uint32_t output_tile = 16;
    static constexpr uint32_t row_batch = 4;
    static constexpr uint32_t block_batch = 2;
    static constexpr uint32_t reduction_batch = 2;
    static constexpr uint32_t row_span = block_batch * q5_block;
    static constexpr uint32_t reduction_span =
        reduction_batch * q5_block;
    static constexpr uint32_t reductions_per_batch =
        block_batch / reduction_batch;
    static constexpr uint32_t tile_elements = row_batch * row_span;
    static constexpr uint32_t vector_bytes = 32;
    static constexpr uint32_t byte_vectors = 11;
    static constexpr uint32_t half_vectors = 7;
    static constexpr uint32_t accumulator_stride = 8;
    static constexpr uint32_t diagnostic_offset_bytes =
        2 * group_elements * sizeof(uint16_t);
    // 真机逐段定位 hwts 异常；0 为完整 kernel。
    static constexpr uint32_t diagnostic_stage = 0;

    AscendC::GlobalTensor<half> weight_;
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> output_;
    uint32_t rows_ = 0;
    uint32_t columns_ = 0;
    uint32_t batch_ = 0;
    uint32_t blocks_per_row_ = 0;
    uint32_t row_half_elements_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR weight, GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQ5KGemv op;
    op.Init(weight, input, output, tiling_data.rows, tiling_data.columns,
            tiling_data.batch);
    op.Process();
}
