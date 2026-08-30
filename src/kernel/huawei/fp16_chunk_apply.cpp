#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"

// 每个 task 计算 [16,16] inverse @ [16,128] K/V block。
// inverse 是普通 16x16 Cube tile，vectors 按 N-tile-major/N×K
// 存放，output 按 N-tile-major/M×N 存放；Fixpipe 结果可以
// 原样留给后续 chunk state Cube 算子，不经 CPU。
class KernelFp16ChunkApply {
public:
    __aicore__ inline KernelFp16ChunkApply() {}

    __aicore__ inline void Init(GM_ADDR inverse, GM_ADDR vectors,
                                GM_ADDR output, uint32_t tasks,
                                uint32_t columns)
    {
        tasks_ = tasks;
        columns_ = columns;
        inverse_.SetGlobalBuffer((__gm__ half*)inverse,
                                 tasks * cube_elements);
        vectors_.SetGlobalBuffer((__gm__ half*)vectors,
                                 tasks * cube_tile * columns);
        output_.SetGlobalBuffer((__gm__ half*)output,
                                tasks * cube_tile * columns);
    }

    __aicore__ inline void Process()
    {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::A2> inverse_buffer;
        AscendC::TBuf<AscendC::TPosition::B2> vector_buffer;
        AscendC::TBuf<AscendC::TPosition::CO1> result_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_buffer;
        AscendC::TBuf<AscendC::TPosition::C1> fix_workspace_buffer;
        const uint32_t vector_elements = cube_tile * columns_;
        pipe.InitBuffer(inverse_buffer, cube_elements * sizeof(half));
        pipe.InitBuffer(vector_buffer, vector_elements * sizeof(half));
        pipe.InitBuffer(result_buffer, vector_elements * sizeof(half));
        pipe.InitBuffer(output_buffer, vector_elements * sizeof(half));
        pipe.InitBuffer(fix_workspace_buffer, 2048);

        auto inverse_local = inverse_buffer.Get<half>();
        auto vector_local = vector_buffer.Get<half>();
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
        const uint32_t vector_elements_per_task = cube_tile * columns_;
        for (uint32_t task = core; task < tasks_; task += cores) {
            AscendC::LoadData(inverse_local,
                              inverse_[task * cube_elements], load);
            const uint32_t vector_base = task * vector_elements_per_task;
            for (uint32_t column_tile = 0; column_tile < column_tiles;
                 ++column_tile) {
                const uint32_t tile_offset = column_tile * cube_elements;
                AscendC::LoadData(vector_local[tile_offset],
                                  vectors_[vector_base + tile_offset], load);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(load_to_mm);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(load_to_mm);

            AscendC::MmadParams mm;
            mm.SetM(cube_tile);
            mm.SetN(columns_);
            mm.SetK(cube_tile);
            mm.SetCmatrixInitVal(true);
            AscendC::Mmad(result_local, inverse_local, vector_local, mm);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);

            AscendC::FixpipeParams<half> fix;
            fix.SetNSize(columns_);
            fix.SetMSize(cube_tile);
            AscendC::Fixpipe(output_local, result_local, fix_workspace, fix);
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);
            AscendC::DataCopy(output_[vector_base], output_local,
                              vector_elements_per_task);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    static constexpr uint32_t cube_tile = 16;
    static constexpr uint32_t cube_elements = cube_tile * cube_tile;

    AscendC::GlobalTensor<half> inverse_;
    AscendC::GlobalTensor<half> vectors_;
    AscendC::GlobalTensor<half> output_;
    uint32_t tasks_;
    uint32_t columns_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR inverse, GM_ADDR vectors, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFp16ChunkApply op;
    op.Init(inverse, vectors, output, tiling_data.tasks,
            tiling_data.columns);
    op.Process();
}
