#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"

// 以 H=S^T 保存 recurrent state，使 chunk scan 的三步都能由 Cube 完成：
//   v_new^T = H @ (-W^T) + U^T
//   o^T     = H @ Q_b^T + v_new^T @ QK^T
//   H_next  = b_end * H + v_new^T @ R
// 每个 core 顺序推进若干 value head 的全部 chunk；chunk 内的中间结果只写
// NPU GM，下一步直接 MTE2 回 Cube，不经过 CPU。
class KernelFp16ChunkScan {
public:
    __aicore__ inline KernelFp16ChunkScan() {}

    __aicore__ inline void Init(
        GM_ADDR negative_w, GM_ADDR u, GM_ADDR q_b, GM_ADDR qk,
        GM_ADDR r_transposed, GM_ADDR b_identity, GM_ADDR identity,
        GM_ADDR state, GM_ADDR output_transposed, GM_ADDR output_state,
        uint32_t heads, uint32_t chunks)
    {
        heads_ = heads;
        chunks_ = chunks;
        const uint32_t tasks = heads * chunks;
        negative_w_.SetGlobalBuffer((__gm__ half*)negative_w,
                                    tasks * short_matrix_elements);
        u_.SetGlobalBuffer((__gm__ half*)u,
                           tasks * short_matrix_elements);
        q_b_.SetGlobalBuffer((__gm__ half*)q_b,
                             tasks * short_matrix_elements);
        qk_.SetGlobalBuffer((__gm__ half*)qk,
                            tasks * cube_elements);
        r_transposed_.SetGlobalBuffer((__gm__ half*)r_transposed,
                                      tasks * short_matrix_elements);
        b_identity_.SetGlobalBuffer((__gm__ half*)b_identity,
                                    tasks * cube_elements);
        identity_.SetGlobalBuffer((__gm__ half*)identity, state_elements);
        state_.SetGlobalBuffer((__gm__ half*)state,
                               heads * state_elements);
        output_transposed_.SetGlobalBuffer(
            (__gm__ half*)output_transposed,
            tasks * short_matrix_elements);
        output_state_.SetGlobalBuffer((__gm__ half*)output_state,
                                      heads * state_elements);
    }

    __aicore__ inline void Process()
    {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::A2> state_buffer;
        AscendC::TBuf<AscendC::TPosition::B2> right_buffer;
        AscendC::TBuf<AscendC::TPosition::B2> small_right_buffer;
        AscendC::TBuf<AscendC::TPosition::CO1> result_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> final_output_buffer;
        AscendC::TBuf<AscendC::TPosition::C1> fix_workspace_buffer;
        pipe.InitBuffer(state_buffer, state_elements * sizeof(half));
        pipe.InitBuffer(right_buffer,
                        short_matrix_elements * sizeof(half));
        pipe.InitBuffer(small_right_buffer,
                        state_update_k_tiles * cube_elements * sizeof(half));
        pipe.InitBuffer(result_buffer,
                        short_matrix_elements * sizeof(half));
        pipe.InitBuffer(output_buffer,
                        short_matrix_elements * sizeof(half));
        pipe.InitBuffer(final_output_buffer,
                        short_matrix_elements * sizeof(half));
        pipe.InitBuffer(fix_workspace_buffer, 2048);

        auto state_local = state_buffer.Get<half>();
        auto right_local = right_buffer.Get<half>();
        auto small_right_local = small_right_buffer.Get<half>();
        auto result_local = result_buffer.Get<half>();
        auto output_local = output_buffer.Get<half>();
        auto final_output_local = final_output_buffer.Get<half>();
        auto fix_workspace = fix_workspace_buffer.Get<uint64_t>();
        const AscendC::LoadData2DParams load(0, 1, 1, 0, 0, false, 0);
        const auto load_to_mm = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::MTE2_M));
        const auto mm_to_load = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::M_MTE2));
        const auto mm_to_fix = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::M_FIX));
        const auto fix_to_mm = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::FIX_M));
        const auto fix_to_gm = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::FIX_MTE3));
        const auto store_to_load = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        const auto vector_to_store = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        for (uint32_t head = core; head < heads_; head += cores) {
            const uint32_t state_base = head * state_elements;
            for (uint32_t chunk = 0; chunk < chunks_; ++chunk) {
                const uint32_t task = head * chunks_ + chunk;
                const uint32_t short_base = task * short_matrix_elements;

                // H：A2 使用 M-tile-major/K-tile-major。
                for (uint32_t tile = 0; tile < state_tiles; ++tile) {
                    if (chunk == 0) {
                        AscendC::LoadData(
                            state_local[tile * cube_elements],
                            state_[state_base + tile * cube_elements], load);
                    } else {
                        AscendC::LoadData(
                            state_local[tile * cube_elements],
                            output_state_[state_base + tile * cube_elements],
                            load);
                    }
                }
                for (uint32_t tile = 0; tile < state_side_tiles; ++tile) {
                    AscendC::LoadData(
                        right_local[tile * cube_elements],
                        negative_w_[short_base + tile * cube_elements], load);
                }
                LoadReady(load_to_mm);

                // H @ (-W^T)。negative_w 的 GM tile 内容是 [T,K]，
                // B2 把它解释成 [K,T]。
                AscendC::MmadParams large_mm;
                large_mm.SetM(state_side);
                large_mm.SetN(chunk_tokens);
                large_mm.SetK(state_side);
                large_mm.SetCmatrixInitVal(true);
                AscendC::Mmad(result_local, state_local, right_local,
                              large_mm);
                MmadReadyForLoad(mm_to_load);

                // 用完整 I128 @ U^T 一次加入 CO1。9010 不支持对
                // CO1 子矩阵做多次偏移累加；identity 复用 state A2 缓冲，
                // v_new 固定后再从 GM 重载 H。
                for (uint32_t tile = 0; tile < state_tiles; ++tile) {
                    AscendC::LoadData(
                        state_local[tile * cube_elements],
                        identity_[tile * cube_elements], load);
                }
                for (uint32_t tile = 0; tile < state_side_tiles; ++tile) {
                    AscendC::LoadData(
                        right_local[tile * cube_elements],
                        u_[short_base + tile * cube_elements], load);
                }
                LoadReady(load_to_mm);
                large_mm.SetCmatrixInitVal(false);
                AscendC::Mmad(result_local, state_local, right_local,
                              large_mm);
                MmadReadyForLoad(mm_to_load);
                // 最终 output tensor 的当前 task 先暂存 v_new；最终值保留
                // 在 UB，state 更新完成后原位覆盖，不需要第三个输出。
                FixShort(output_local, result_local, fix_workspace,
                         mm_to_fix);
                FixReady(fix_to_mm, fix_to_gm);
                AscendC::DataCopy(output_transposed_[short_base], output_local,
                                  short_matrix_elements);
                StoreReadyForLoad(store_to_load);
                for (uint32_t tile = 0; tile < state_tiles; ++tile) {
                    if (chunk == 0) {
                        AscendC::LoadData(
                            state_local[tile * cube_elements],
                            state_[state_base + tile * cube_elements], load);
                    } else {
                        AscendC::LoadData(
                            state_local[tile * cube_elements],
                            output_state_[state_base + tile * cube_elements],
                            load);
                    }
                }
                for (uint32_t k_tile = 0; k_tile < state_side_tiles;
                     ++k_tile) {
                    AscendC::LoadData(
                        right_local[k_tile * cube_elements],
                        q_b_[short_base + k_tile * cube_elements], load);
                }
                LoadReady(load_to_mm);

                // 9010 只能稳定在同一 CO1 上累加同形 K=128。
                // 第二项把 v_new/QK 放在首 tile，其余 tile 由
                // I128 的非对角零块 padding，得到等价的 K=128。
                large_mm.SetCmatrixInitVal(true);
                AscendC::Mmad(result_local, state_local, right_local,
                              large_mm);
                MmadReadyForLoad(mm_to_load);
                for (uint32_t m_tile = 0;
                     m_tile < state_side_tiles; ++m_tile) {
                    AscendC::LoadData(
                        state_local[m_tile * state_side_tiles *
                                    cube_elements],
                        output_transposed_[short_base +
                                           m_tile * cube_elements], load);
                }
                AscendC::LoadData(right_local,
                                  qk_[task * cube_elements], load);
                for (uint32_t k_tile = 1; k_tile < state_side_tiles;
                     ++k_tile) {
                    AscendC::LoadData(
                        right_local[k_tile * cube_elements],
                        identity_[cube_elements], load);
                }
                LoadReady(load_to_mm);
                large_mm.SetCmatrixInitVal(false);
                AscendC::Mmad(result_local, state_local, right_local,
                              large_mm);
                MmadReadyForLoad(mm_to_load);
                FixShort(final_output_local, result_local, fix_workspace,
                         mm_to_fix);
                FixReady(fix_to_mm, fix_to_gm);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Muls(final_output_local, final_output_local,
                              static_cast<half>(0.0625f),
                              short_matrix_elements);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                    vector_to_store);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                    vector_to_store);
                // H_next 的每个 128x16 N tile 用一次 K=32 Mmad 完成：
                // [H[:,n], v_new^T] @ [bI16; R[:,n]]。9010 的短矩阵
                // 二次 CO1 累加会丢掉第二项，合并后也少一次同步。
                AscendC::MmadParams short_mm;
                short_mm.SetM(state_side);
                short_mm.SetN(chunk_tokens);
                short_mm.SetK(state_update_k);
                for (uint32_t n_tile = 0; n_tile < state_side_tiles;
                     ++n_tile) {
                    for (uint32_t m_tile = 0;
                         m_tile < state_side_tiles; ++m_tile) {
                        const uint32_t source_tile =
                            m_tile * state_side_tiles + n_tile;
                        const uint32_t state_tile =
                            m_tile * state_update_k_tiles;
                        if (chunk == 0) {
                            AscendC::LoadData(
                                state_local[state_tile * cube_elements],
                                state_[state_base +
                                       source_tile * cube_elements], load);
                        } else {
                            AscendC::LoadData(
                                state_local[state_tile * cube_elements],
                                output_state_[state_base +
                                              source_tile * cube_elements],
                                load);
                        }
                        AscendC::LoadData(
                            state_local[(state_tile + 1) * cube_elements],
                            output_transposed_[short_base +
                                               m_tile * cube_elements], load);
                    }
                    AscendC::LoadData(
                        small_right_local,
                        b_identity_[task * cube_elements], load);
                    AscendC::LoadData(
                        small_right_local[cube_elements],
                        r_transposed_[short_base +
                                      n_tile * cube_elements], load);
                    LoadReady(load_to_mm);
                    short_mm.SetCmatrixInitVal(true);
                    AscendC::Mmad(result_local, state_local,
                                  small_right_local, short_mm);
                    MmadReadyForLoad(mm_to_load);
                    FixShort(output_local, result_local, fix_workspace,
                             mm_to_fix);
                    FixReady(fix_to_mm, fix_to_gm);
                    for (uint32_t m_tile = 0;
                         m_tile < state_side_tiles; ++m_tile) {
                        const uint32_t destination_tile =
                            m_tile * state_side_tiles + n_tile;
                        AscendC::DataCopy(
                            output_state_[state_base +
                                          destination_tile * cube_elements],
                            output_local[m_tile * cube_elements],
                            cube_elements);
                    }
                    StoreReadyForLoad(store_to_load);
                }
                AscendC::DataCopy(output_transposed_[short_base],
                                  final_output_local,
                                  short_matrix_elements);
                StoreReadyForLoad(store_to_load);
            }

        }
    }

private:
    __aicore__ inline void FixShort(
        AscendC::LocalTensor<half> output,
        AscendC::LocalTensor<half> result,
        AscendC::LocalTensor<uint64_t> workspace,
        event_t mm_to_fix)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mm_to_fix);
        AscendC::FixpipeParams<half> fix;
        fix.SetNSize(chunk_tokens);
        fix.SetMSize(state_side);
        AscendC::Fixpipe(output, result, workspace, fix);
    }

    __aicore__ inline void LoadReady(event_t event)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(event);
    }

    __aicore__ inline void MmadReadyForLoad(event_t event)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event);
    }

    __aicore__ inline void FixReady(event_t fix_to_mm, event_t fix_to_gm)
    {
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fix_to_mm);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(fix_to_mm);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE3>(fix_to_gm);
    }

    __aicore__ inline void StoreReadyForLoad(event_t event)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }

    static constexpr uint32_t cube_tile = 16;
    static constexpr uint32_t cube_elements = cube_tile * cube_tile;
    static constexpr uint32_t chunk_tokens = 16;
    static constexpr uint32_t state_side = 128;
    static constexpr uint32_t state_side_tiles = state_side / cube_tile;
    static constexpr uint32_t state_update_k_tiles = 2;
    static constexpr uint32_t state_update_k =
        state_update_k_tiles * cube_tile;
    static constexpr uint32_t state_tiles =
        state_side_tiles * state_side_tiles;
    static constexpr uint32_t state_elements = state_side * state_side;
    static constexpr uint32_t short_matrix_elements =
        state_side * chunk_tokens;

    AscendC::GlobalTensor<half> negative_w_;
    AscendC::GlobalTensor<half> u_;
    AscendC::GlobalTensor<half> q_b_;
    AscendC::GlobalTensor<half> qk_;
    AscendC::GlobalTensor<half> r_transposed_;
    AscendC::GlobalTensor<half> b_identity_;
    AscendC::GlobalTensor<half> identity_;
    AscendC::GlobalTensor<half> state_;
    AscendC::GlobalTensor<half> output_transposed_;
    AscendC::GlobalTensor<half> output_state_;
    uint32_t heads_;
    uint32_t chunks_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR negative_w, GM_ADDR u, GM_ADDR q_b, GM_ADDR qk,
    GM_ADDR r_transposed, GM_ADDR b_identity, GM_ADDR identity,
    GM_ADDR state, GM_ADDR output_transposed, GM_ADDR output_state,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    (void)workspace;
    KernelFp16ChunkScan op;
    op.Init(negative_w, u, q_b, qk, r_transposed, b_identity,
            identity, state, output_transposed, output_state,
            tiling_data.heads, tiling_data.chunks);
    op.Process();
}
