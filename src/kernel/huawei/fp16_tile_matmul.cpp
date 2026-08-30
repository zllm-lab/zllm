#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"

// 注意：本文件与 fp16_matmul.cpp 导出同名 extern "C" 符号
// fp16_tile_matmul，二者是同一算子的两个候选实现，编译脚本每次只拷贝
// 其中一份（compile_q5_resident_tiles_kirin9010.sh 默认
// FP16_MATMUL_KERNEL=fp16_tile_matmul.cpp 使用本文件，配
// q5_decode_tiles.cpp），禁止同时编译。
// 本实现的 L0A 输入 tile 按 m-major 打包：(m_tile * k_tiles + k_tile)，
// block 为 256x128x32。
// 每个任务负责一个 token tile 与一个输出行 tile。prefill 因此可以一次
// 提交完整序列，不再把 1024 token 拆成几十次同步 HIAI Process。
class KernelFp16TileMatmul {
public:
    __aicore__ inline KernelFp16TileMatmul() {}

    __aicore__ inline void Init(GM_ADDR weight_tiles, GM_ADDR input_tiles,
                                GM_ADDR output, uint32_t rows,
                                uint32_t columns, uint32_t batch)
    {
        rows_ = rows;
        columns_ = columns;
        batch_ = batch;
        row_tiles_ = rows / cube_tile;
        column_tiles_ = columns / cube_tile;
        batch_tiles_ = (batch + cube_tile - 1) / cube_tile;
        weight_.SetGlobalBuffer((__gm__ half*)weight_tiles, rows * columns);
        input_.SetGlobalBuffer((__gm__ half*)input_tiles,
                               batch_tiles_ * cube_tile * columns);
        output_.SetGlobalBuffer((__gm__ half*)output, batch * rows);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t input_slot_elements = block_m * block_k;
        constexpr uint32_t weight_slot_elements = block_n * block_k;
        constexpr uint32_t result_elements = block_m * block_n;
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::A2> input_buffer;
        AscendC::TBuf<AscendC::TPosition::B2> weight_buffer;
        AscendC::TBuf<AscendC::TPosition::CO1> result_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_buffer;
        AscendC::TBuf<AscendC::TPosition::C1> fix_workspace_buffer;
        pipe.InitBuffer(input_buffer,
                        2 * input_slot_elements * sizeof(half));
        pipe.InitBuffer(weight_buffer,
                        2 * weight_slot_elements * sizeof(half));
        pipe.InitBuffer(result_buffer, result_elements * sizeof(half));
        pipe.InitBuffer(output_buffer, result_elements * sizeof(half));
        pipe.InitBuffer(fix_workspace_buffer, 2048);

        auto input_local = input_buffer.Get<half>();
        auto weight_local = weight_buffer.Get<half>();
        auto result_local = result_buffer.Get<half>();
        auto output_local = output_buffer.Get<half>();
        auto fix_workspace = fix_workspace_buffer.Get<uint64_t>();
        const AscendC::LoadData2DParams load(0, 1, 1, 0, 0, false, 0);
        const event_t load_to_mm[2] = {
            static_cast<event_t>(
                pipe.FetchEventID(AscendC::HardEvent::MTE2_M)),
            static_cast<event_t>(
                pipe.FetchEventID(AscendC::HardEvent::MTE2_M))};
        const event_t mm_to_load[2] = {
            static_cast<event_t>(
                pipe.FetchEventID(AscendC::HardEvent::M_MTE2)),
            static_cast<event_t>(
                pipe.FetchEventID(AscendC::HardEvent::M_MTE2))};
        const auto mm_to_fix = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::M_FIX));
        const auto fix_to_gm = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::FIX_MTE3));

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t batch_blocks =
            (batch_ + block_m - 1) / block_m;
        const uint32_t row_blocks = (rows_ + block_n - 1) / block_n;
        const uint32_t tasks = batch_blocks * row_blocks;
        for (uint32_t task = core; task < tasks; task += cores) {
            const uint32_t batch_block = task / row_blocks;
            const uint32_t row_block = task % row_blocks;
            const uint32_t token_start = batch_block * block_m;
            const uint32_t row_start = row_block * block_n;
            const uint32_t valid_m = batch_ - token_start < block_m
                                         ? batch_ - token_start
                                         : block_m;
            const uint32_t valid_n = rows_ - row_start < block_n
                                         ? rows_ - row_start
                                         : block_n;
            const uint32_t m_tiles =
                (valid_m + cube_tile - 1) / cube_tile;
            const uint32_t n_tiles =
                (valid_n + cube_tile - 1) / cube_tile;
            const uint32_t matrix_m = m_tiles * cube_tile;
            const uint32_t matrix_n = n_tiles * cube_tile;
            const uint32_t k_tiles_per_block = block_k / cube_tile;
            const uint32_t column_blocks =
                (column_tiles_ + k_tiles_per_block - 1) /
                k_tiles_per_block;
            for (uint32_t column_block = 0;
                 column_block < column_blocks; ++column_block) {
                const uint32_t slot = column_block & 1;
                if (column_block >= 2) {
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(
                        mm_to_load[slot]);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(
                        mm_to_load[slot]);
                }
                const uint32_t input_slot = slot * input_slot_elements;
                const uint32_t weight_slot = slot * weight_slot_elements;
                const uint32_t column_tile_start =
                    column_block * k_tiles_per_block;
                const uint32_t k_tiles =
                    column_tiles_ - column_tile_start < k_tiles_per_block
                        ? column_tiles_ - column_tile_start
                        : k_tiles_per_block;
                for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    const uint32_t column_tile =
                        column_tile_start + k_tile;
                    for (uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
                        const uint32_t input_tile =
                            (token_start / cube_tile + m_tile) *
                                column_tiles_ +
                            column_tile;
                        AscendC::LoadData(
                            input_local[input_slot +
                                        (m_tile * k_tiles + k_tile) *
                                            cube_elements],
                            input_[input_tile * cube_elements], load);
                    }
                    for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
                        const uint32_t weight_tile =
                            (row_start / cube_tile + n_tile) *
                                column_tiles_ +
                            column_tile;
                        AscendC::LoadData(
                            weight_local[weight_slot +
                                         (k_tile * n_tiles + n_tile) *
                                             cube_elements],
                            weight_[weight_tile * cube_elements], load);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(
                    load_to_mm[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(
                    load_to_mm[slot]);

                AscendC::MmadParams mm;
                mm.SetM(matrix_m);
                mm.SetN(matrix_n);
                mm.SetK(k_tiles * cube_tile);
                mm.SetCmatrixInitVal(column_block == 0);
                AscendC::Mmad(result_local, input_local[input_slot],
                              weight_local[weight_slot], mm);
            }

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);
            AscendC::FixpipeParams<half> fix;
            fix.SetNSize(matrix_n);
            fix.SetMSize(matrix_m);
            AscendC::Fixpipe(output_local, result_local, fix_workspace, fix);
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);

            for (uint32_t local_token = 0; local_token < valid_m;
                 ++local_token) {
                const uint32_t token = token_start + local_token;
                const uint32_t m_tile = local_token / cube_tile;
                const uint32_t m_lane = local_token % cube_tile;
                for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
                    const uint32_t source =
                        (n_tile * m_tiles + m_tile) * cube_elements +
                        m_lane * cube_tile;
                    AscendC::DataCopy(
                        output_[token * rows_ + row_start +
                                n_tile * cube_tile],
                        output_local[source], cube_tile);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    static constexpr uint32_t cube_tile = 16;
    static constexpr uint32_t cube_elements = cube_tile * cube_tile;
    static constexpr uint32_t block_m = 256;
    static constexpr uint32_t block_n = 128;
    static constexpr uint32_t block_k = 32;

    AscendC::GlobalTensor<half> weight_;
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> output_;
    uint32_t rows_, columns_, batch_, row_tiles_, column_tiles_, batch_tiles_;
};

extern "C" __global__ __aicore__ void fp16_tile_matmul(
    GM_ADDR weight_tiles, GM_ADDR input_tiles, GM_ADDR shape_input,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFp16TileMatmul op;
    op.Init(weight_tiles, input_tiles, output, tiling_data.rows,
            tiling_data.columns, tiling_data.batch);
    op.Process();
}
