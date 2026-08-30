#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"

// 只验证 dav-l210 的硬件 Cube 数据通路。16 是 Cube 的硬件分形尺寸，
// 不是模型维度；完整矩阵在此基础上按 tiling 循环。
extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR weight, GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    constexpr uint32_t tile = 16;
    constexpr uint32_t elements = tile * tile;
    constexpr uint32_t bytes = elements * sizeof(half);

    AscendC::GlobalTensor<half> weight_global;
    AscendC::GlobalTensor<half> input_global;
    AscendC::GlobalTensor<half> output_global;
    weight_global.SetGlobalBuffer((__gm__ half*)weight, elements);
    input_global.SetGlobalBuffer((__gm__ half*)input, elements);
    output_global.SetGlobalBuffer((__gm__ half*)output, elements);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A2> a_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> b_buffer;
    AscendC::TBuf<AscendC::TPosition::CO1> c_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> output_buffer;
    AscendC::TBuf<AscendC::TPosition::C1> fix_workspace;
    pipe.InitBuffer(a_buffer, bytes);
    pipe.InitBuffer(b_buffer, bytes);
    pipe.InitBuffer(c_buffer, bytes);
    pipe.InitBuffer(output_buffer, bytes);
    pipe.InitBuffer(fix_workspace, 2048);

    auto a = a_buffer.Get<half>();
    auto b = b_buffer.Get<half>();
    const AscendC::LoadData2DParams load(0, 1, 1, 0, 0, false, 0);
    AscendC::LoadData(a, input_global, load);
    AscendC::LoadData(b, weight_global, load);

    // 两个输入都由 MTE2 从 GM 直接送入 L0，统一等待后才能启动 Cube。
    // TQue<B2> 会误按标准 B1→B2 路径等待 MTE1，不能用于这条直达路径。
    const auto load_to_mm = static_cast<event_t>(
        pipe.FetchEventID(AscendC::HardEvent::MTE2_M));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(load_to_mm);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(load_to_mm);

    auto c = c_buffer.Get<half>();
    AscendC::MmadParams mm;
    mm.SetM(tile);
    mm.SetN(tile);
    mm.SetK(tile);
    mm.SetCmatrixInitVal(true);
    AscendC::Mmad(c, a, b, mm);

    const auto mm_to_fix = static_cast<event_t>(
        pipe.FetchEventID(AscendC::HardEvent::M_FIX));
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);

    auto output_local = output_buffer.Get<half>();
    auto fix_local = fix_workspace.Get<uint64_t>();
    AscendC::FixpipeParams<half> fix;
    fix.SetNSize(tile);
    fix.SetMSize(tile);
    AscendC::Fixpipe(output_local, c, fix_local, fix);

    const auto fix_to_gm = static_cast<event_t>(
        pipe.FetchEventID(AscendC::HardEvent::FIX_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);
    AscendC::DataCopy(output_global, output_local, elements);
}
