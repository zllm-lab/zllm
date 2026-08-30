#include "kernel_operator.h"

// 只给一条 Q5 投影使用 1-byte 量化码；另一条 Q5 与 Q6 保持 compact，
// 三份 Const 合计仍低于 Kirin 9010 的 64 MiB 单图访问窗口。
class KernelQ5KDenseGemv {
public:
    __aicore__ inline KernelQ5KDenseGemv() {}

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR input, GM_ADDR output,
                                uint32_t rows, uint32_t columns,
                                uint32_t batch)
    {
        rows_ = rows;
        columns_ = columns;
        batch_ = batch;
        blocks_per_row_ = columns / q5_block;
        row_half_elements_ =
            blocks_per_row_ * dense_bytes / sizeof(half);
        weight_.SetGlobalBuffer((__gm__ half*)weight,
                                rows * row_half_elements_);
        input_.SetGlobalBuffer((__gm__ half*)input, columns);
        output_.SetGlobalBuffer((__gm__ half*)output, rows);
    }

    __aicore__ inline void Process()
    {
        // 布局约束:batch=1、rows 为 16(output_tile)的倍数、
        // columns 同时被 256(q5_block) 与 5*256=1280(block_batch*q5_block) 整除、
        // 行 half 数对齐 16。kernel 没有错误通道,这里只能静默返回且
        // 不会写任何输出;权重上传侧必须在装配期完成同样的校验,
        // 否则下游会读到未初始化的输出 buffer。
        if (batch_ != 1 || rows_ % output_tile != 0 ||
            columns_ % q5_block != 0 ||
            blocks_per_row_ % block_batch != 0 ||
            row_half_elements_ % 16 != 0) {
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
        const uint32_t partials_per_row = blocks_per_row_ / block_batch;
        const uint32_t block_partial_elements =
            output_tile * partials_per_row * accumulator_stride;
        const uint32_t reduce_elements =
            block_partial_elements + output_tile * accumulator_stride +
            reduction_scratch_elements;
        const uint32_t raw_offset_bytes =
            3 * tile_elements * sizeof(uint16_t) +
            output_tile * sizeof(uint32_t) +
            partials_per_row * sizeof(uint32_t);
        const uint32_t aligned_offset_bytes =
            (raw_offset_bytes + vector_bytes - 1) & ~(vector_bytes - 1);
        pipe.InitBuffer(input_queue, 1, columns_ * sizeof(half));
        pipe.InitBuffer(weight_queue, 1,
                        output_tile * row_half_elements_ * sizeof(half));
        pipe.InitBuffer(output_queue, 1, output_tile * sizeof(half));
        pipe.InitBuffer(byte_buffer, tile_elements);
        pipe.InitBuffer(half_buffer, 3 * tile_elements * sizeof(half));
        pipe.InitBuffer(float_buffer, row_span * sizeof(float));
        pipe.InitBuffer(reduce_buffer, reduce_elements * sizeof(float));
        pipe.InitBuffer(offset_buffer, aligned_offset_bytes);

        auto input = input_queue.AllocTensor<half>();
        AscendC::DataCopy(input, input_, columns_);
        input_queue.EnQue(input);
        input = input_queue.DeQue<half>();

        auto quant = byte_buffer.Get<uint8_t>();
        auto halves = half_buffer.Get<half>();
        auto values = halves[0 * tile_elements];
        auto scales = halves[1 * tile_elements];
        auto biases = halves[2 * tile_elements];
        auto products = float_buffer.Get<float>();
        auto reductions = reduce_buffer.Get<float>();
        auto block_partials = reductions[0];
        auto accumulators = reductions[block_partial_elements];
        auto reduction = reductions[
            block_partial_elements + output_tile * accumulator_stride];

        auto index = offset_buffer.Get<uint16_t>();
        auto code_indices = index[0 * tile_elements];
        auto scale_indices = index[1 * tile_elements];
        auto bias_indices = index[2 * tile_elements];
        auto result_indices =
            index[3 * tile_elements].ReinterpretCast<uint32_t>();
        auto partial_indices =
            index[3 * tile_elements + 2 * output_tile]
                .ReinterpretCast<uint32_t>();

        for (uint32_t block_lane = 0; block_lane < block_batch;
             ++block_lane) {
            const uint32_t block_offset = block_lane * dense_bytes;
            const uint32_t block_element = block_lane * q5_block;
            for (uint32_t group = 0; group < groups_per_block; ++group) {
                const uint32_t element =
                    block_element + group * group_elements;
                AscendC::CreateVecIndex(
                    code_indices[element].ReinterpretCast<int16_t>(),
                    static_cast<int16_t>(
                        block_offset + code_offset +
                        group * group_elements),
                    group_elements);
                AscendC::Duplicate(
                    scale_indices[element],
                    static_cast<uint16_t>(block_offset + group * 2),
                    group_elements);
                AscendC::Duplicate(
                    bias_indices[element],
                    static_cast<uint16_t>(
                        block_offset + bias_offset + group * 2),
                    group_elements);
            }
        }

        auto signed_result_indices =
            result_indices.ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(signed_result_indices,
                                static_cast<int32_t>(0), output_tile);
        AscendC::Muls(signed_result_indices, signed_result_indices,
                      static_cast<int32_t>(
                          accumulator_stride * sizeof(float)),
                      output_tile);
        auto signed_partial_indices =
            partial_indices.ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(signed_partial_indices,
                                static_cast<int32_t>(0), partials_per_row);
        AscendC::Muls(signed_partial_indices, signed_partial_indices,
                      static_cast<int32_t>(
                          accumulator_stride * sizeof(float)),
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
            for (uint32_t local_row = 0; local_row < output_tile;
                 ++local_row) {
                auto packed_half =
                    packed_tile[local_row * row_half_elements_];
                auto packed = packed_half.ReinterpretCast<uint8_t>();
                for (uint32_t block = 0; block < blocks_per_row_;
                     block += block_batch) {
                    const uint32_t base = block * dense_bytes;
                    AscendC::Gather(quant, packed[base], code_indices, 0,
                                    tile_elements);
                    AscendC::Gather(scales,
                                    packed_half[base / sizeof(half)],
                                    scale_indices, 0, tile_elements);
                    AscendC::Gather(biases,
                                    packed_half[base / sizeof(half)],
                                    bias_indices, 0, tile_elements);
                    AscendC::Cast(values, quant,
                                  AscendC::RoundMode::CAST_NONE,
                                  tile_elements);
                    AscendC::Mul(values, values, scales, tile_elements);
                    AscendC::Sub(values, values, biases, tile_elements);
                    const uint32_t column = block * q5_block;
                    AscendC::Mul(biases, input[column], values, row_span);
                    AscendC::Cast(products, biases,
                                  AscendC::RoundMode::CAST_NONE, row_span);
                    auto partial = block_partials[
                        (local_row * partials_per_row +
                         block / block_batch) * accumulator_stride];
                    AscendC::ReduceSum(partial, products, reduction,
                                       row_span);
                }
                auto row_partials = block_partials[
                    local_row * partials_per_row * accumulator_stride];
                AscendC::Gather(products, row_partials, partial_indices, 0,
                                partials_per_row);
                auto accumulator =
                    accumulators[local_row * accumulator_stride];
                AscendC::ReduceSum(accumulator, products, reduction,
                                   partials_per_row);
            }
            weight_queue.FreeTensor(packed_tile);
            AscendC::Gather(products, accumulators, result_indices, 0,
                            output_tile);
            AscendC::Cast(output, products,
                          AscendC::RoundMode::CAST_NONE, output_tile);
            output_queue.EnQue(output);
            output = output_queue.DeQue<half>();
            AscendC::DataCopy(output_[first_row], output, output_tile);
            output_queue.FreeTensor(output);
        }
        input_queue.FreeTensor(input);
    }

private:
    static constexpr uint32_t q5_block = 256;
    static constexpr uint32_t groups_per_block = 8;
    static constexpr uint32_t group_elements = 32;
    static constexpr uint32_t dense_bytes = 288;
    static constexpr uint32_t bias_offset = 16;
    static constexpr uint32_t code_offset = 32;
    // GM DataCopy 的输出必须保持 32-byte 粒度；同时用 5-block 工作集
    // 把 UB 压到约 73 KiB，避免 10-block dense task 的 HWTS 0x2。
    static constexpr uint32_t output_tile = 16;
    static constexpr uint32_t block_batch = 5;
    static constexpr uint32_t row_span = block_batch * q5_block;
    static constexpr uint32_t tile_elements = row_span;
    static constexpr uint32_t vector_bytes = 32;
    static constexpr uint32_t accumulator_stride = 8;
    static constexpr uint32_t reduction_scratch_elements = 32;

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
    KernelQ5KDenseGemv op;
    op.Init(weight, input, output, tiling_data.rows, tiling_data.columns,
            tiling_data.batch);
    op.Process();
}
