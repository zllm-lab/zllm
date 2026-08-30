#include "kernel_operator.h"

// Qwen3.5 full-attention 的 q_proj 按 head 交错保存 query 与 output gate。
// 每个 core 搬运若干完整 head，拆分后的两个张量继续常驻 NPU。
class KernelSplitInterleaved {
public:
    __aicore__ inline KernelSplitInterleaved() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR left, GM_ADDR right,
                                uint32_t batch, uint32_t columns,
                                uint32_t block_columns)
    {
        batch_ = batch;
        columns_ = columns;
        block_columns_ = block_columns;
        output_columns_ = columns_ / 2;
        heads_ = columns_ / (block_columns_ * 2);
        input_.SetGlobalBuffer((__gm__ half*)input, batch_ * columns_);
        left_.SetGlobalBuffer((__gm__ half*)left, batch_ * output_columns_);
        right_.SetGlobalBuffer((__gm__ half*)right, batch_ * output_columns_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t bytes = block_columns_ * sizeof(half);
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> query_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> gate_queue;
        pipe.InitBuffer(query_queue, 1, bytes);
        pipe.InitBuffer(gate_queue, 1, bytes);
        const event_t load_to_store = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        const event_t store_to_load = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t total_heads = batch_ * heads_;
        for (uint32_t linear_head = core; linear_head < total_heads;
             linear_head += cores) {
            const uint32_t row = linear_head / heads_;
            const uint32_t head = linear_head - row * heads_;
            const uint32_t input_base = row * columns_ + head * block_columns_ * 2;
            const uint32_t output_base = row * output_columns_ + head * block_columns_;

            auto query = query_queue.AllocTensor<half>();
            auto gate = gate_queue.AllocTensor<half>();
            AscendC::DataCopy(query, input_[input_base], block_columns_);
            AscendC::DataCopy(gate, input_[input_base + block_columns_], block_columns_);
            query_queue.EnQue(query);
            gate_queue.EnQue(gate);
            query = query_queue.DeQue<half>();
            gate = gate_queue.DeQue<half>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(load_to_store);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(load_to_store);
            AscendC::DataCopy(left_[output_base], query, block_columns_);
            AscendC::DataCopy(right_[output_base], gate, block_columns_);
            // B1 每个 core 会处理多个 head；必须等 MTE3 写回完成，才能释放并
            // 复用这一块 UB，否则前一 head 会落到后一 head 的输出地址。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(store_to_load);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(store_to_load);
            query_queue.FreeTensor(query);
            gate_queue.FreeTensor(gate);
        }
    }

private:
    AscendC::GlobalTensor<half> input_, left_, right_;
    uint32_t batch_ = 0, columns_ = 0, block_columns_ = 0;
    uint32_t output_columns_ = 0, heads_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR input, GM_ADDR left, GM_ADDR right, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSplitInterleaved op;
    op.Init(input, left, right, tiling_data.batch, tiling_data.columns,
            tiling_data.block_columns);
    op.Process();
}
