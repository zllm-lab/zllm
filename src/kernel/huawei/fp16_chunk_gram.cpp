#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"

// 输入由前置 Vector 算子排成 task-major、K-tile-major 的 16x16 Cube tile。
// 每个 task 独立计算 [16,K] @ [16,K]^T，避免把无关 head 拼进同一个
// 大矩阵后改变 B2 分形布局。
class KernelFp16ChunkGram {
public:
    __aicore__ inline KernelFp16ChunkGram() {}

    __aicore__ inline void Init(GM_ADDR left_tiles, GM_ADDR right_tiles,
                                GM_ADDR output, uint32_t tasks,
                                uint32_t columns)
    {
        tasks_ = tasks;
        columns_ = columns;
        const uint32_t input_elements = tasks * cube_tile * columns;
        left_.SetGlobalBuffer((__gm__ half*)left_tiles, input_elements);
        right_.SetGlobalBuffer((__gm__ half*)right_tiles, input_elements);
        output_.SetGlobalBuffer((__gm__ half*)output,
                                tasks * cube_elements);
    }

    __aicore__ inline void Process()
    {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::A2> left_buffer;
        AscendC::TBuf<AscendC::TPosition::B2> right_buffer;
        AscendC::TBuf<AscendC::TPosition::CO1> result_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_buffer;
        AscendC::TBuf<AscendC::TPosition::C1> fix_workspace_buffer;
        const uint32_t matrix_elements = cube_tile * columns_;
        pipe.InitBuffer(left_buffer, matrix_elements * sizeof(half));
        pipe.InitBuffer(right_buffer, matrix_elements * sizeof(half));
        pipe.InitBuffer(result_buffer, cube_elements * sizeof(half));
        pipe.InitBuffer(output_buffer, cube_elements * sizeof(half));
        pipe.InitBuffer(fix_workspace_buffer, 2048);

        auto left_local = left_buffer.Get<half>();
        auto right_local = right_buffer.Get<half>();
        auto result_local = result_buffer.Get<half>();
        auto output_local = output_buffer.Get<half>();
        auto fix_workspace = fix_workspace_buffer.Get<uint64_t>();
        const AscendC::LoadData2DParams load(0, 1, 1, 0, 0, false, 0);
        const auto load_to_mm = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::MTE2_M));
        const auto mm_to_fix = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::M_FIX));
        const auto fix_to_gm = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::FIX_MTE3));

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t column_tiles = columns_ / cube_tile;
        for (uint32_t task = core; task < tasks_; task += cores) {
            const uint32_t task_base = task * matrix_elements;
            for (uint32_t column_tile = 0; column_tile < column_tiles;
                 ++column_tile) {
                const uint32_t tile_offset = column_tile * cube_elements;
                AscendC::LoadData(left_local[tile_offset],
                                  left_[task_base + tile_offset], load);
                AscendC::LoadData(right_local[tile_offset],
                                  right_[task_base + tile_offset], load);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(load_to_mm);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(load_to_mm);

            AscendC::MmadParams mm;
            mm.SetM(cube_tile);
            mm.SetN(cube_tile);
            mm.SetK(columns_);
            mm.SetCmatrixInitVal(true);
            AscendC::Mmad(result_local, left_local, right_local, mm);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);

            AscendC::FixpipeParams<half> fix;
            fix.SetNSize(cube_tile);
            fix.SetMSize(cube_tile);
            AscendC::Fixpipe(output_local, result_local, fix_workspace, fix);
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);
            AscendC::DataCopy(output_[task * cube_elements], output_local,
                              cube_elements);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    static constexpr uint32_t cube_tile = 16;
    static constexpr uint32_t cube_elements = cube_tile * cube_tile;

    AscendC::GlobalTensor<half> left_;
    AscendC::GlobalTensor<half> right_;
    AscendC::GlobalTensor<half> output_;
    uint32_t tasks_;
    uint32_t columns_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR left_tiles, GM_ADDR right_tiles, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFp16ChunkGram op;
    op.Init(left_tiles, right_tiles, output, tiling_data.tasks,
            tiling_data.columns);
    op.Process();
}
