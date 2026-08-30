#include "kernel_operator.h"

// Qwen3.5 Q/K 的逐 head Gemma RMSNorm。输入张量保持 [batch, heads * 256]
// 的 resident 布局，每个 AIV core 处理若干完整 head，不经过 host 重排。
class KernelGemmaRmsNormHeads {
public:
    __aicore__ inline KernelGemmaRmsNormHeads() {}

    __aicore__ inline void Init(GM_ADDR gamma, GM_ADDR input, GM_ADDR output,
                                uint32_t head_dim, uint32_t head_rows)
    {
        head_dim_ = head_dim;
        head_rows_ = head_rows;
        gamma_.SetGlobalBuffer((__gm__ half*)gamma, head_dim_);
        input_.SetGlobalBuffer((__gm__ half*)input, head_rows_ * head_dim_);
        output_.SetGlobalBuffer((__gm__ half*)output, head_rows_ * head_dim_);
    }

    __aicore__ inline void Process(float reciprocal_value)
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t bytes = head_dim_ * sizeof(half);
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> gamma_queue;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
        AscendC::TBuf<AscendC::TPosition::VECCALC> square_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> sum_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> reciprocal_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_buf;
        pipe.InitBuffer(input_queue, 1, bytes);
        pipe.InitBuffer(gamma_queue, 1, bytes);
        pipe.InitBuffer(output_queue, 1, bytes);
        pipe.InitBuffer(square_buf, bytes);
        pipe.InitBuffer(sum_buf, 32);
        pipe.InitBuffer(work_buf, 32);
        pipe.InitBuffer(reciprocal_buf, bytes);
        pipe.InitBuffer(offset_buf, head_dim_ * sizeof(uint16_t));

        for (uint32_t row = core; row < head_rows_; row += cores) {
            auto input_local = input_queue.AllocTensor<half>();
            auto gamma_local = gamma_queue.AllocTensor<half>();
            AscendC::DataCopy(input_local, input_[row * head_dim_], head_dim_);
            AscendC::DataCopy(gamma_local, gamma_, head_dim_);
            input_queue.EnQue(input_local);
            gamma_queue.EnQue(gamma_local);
            input_local = input_queue.DeQue<half>();
            gamma_local = gamma_queue.DeQue<half>();

            auto output_local = output_queue.AllocTensor<half>();
            auto square_local = square_buf.Get<half>();
            auto sum_local = sum_buf.Get<half>();
            AscendC::Mul(square_local, input_local, input_local, head_dim_);
            AscendC::ReduceSum(sum_local, square_local, work_buf.Get<half>(), head_dim_);
            // 先乘 1/N 再加 eps：rsqrt(mean + eps)，eps 不能参与均值缩放。
            // eps 硬编码 1e-6：tiling 没有 eps 通道，host 入口
            // （huawei.rs gemma_rmsnorm_heads）已拒绝 eps != 1e-6 的调用。
            AscendC::Muls(sum_local, sum_local, static_cast<half>(reciprocal_value), 1);
            AscendC::Adds(sum_local, sum_local, static_cast<half>(1.0e-6f), 1);
            AscendC::Rsqrt(sum_local, sum_local, 1);

            auto reciprocal_local = reciprocal_buf.Get<half>();
            auto offset_local = offset_buf.Get<uint16_t>();
            AscendC::Duplicate(offset_local, static_cast<uint16_t>(0), head_dim_);
            AscendC::Gather(reciprocal_local, sum_local, offset_local, 0, head_dim_);
            // prepare_gemma_f32 已把零中心 gamma 转成设备侧 (1 + gamma)。
            AscendC::Mul(output_local, input_local, reciprocal_local, head_dim_);
            AscendC::Mul(output_local, output_local, gamma_local, head_dim_);
            output_queue.EnQue(output_local);
            input_queue.FreeTensor(input_local);
            gamma_queue.FreeTensor(gamma_local);
            output_local = output_queue.DeQue<half>();
            AscendC::DataCopy(output_[row * head_dim_], output_local, head_dim_);
            output_queue.FreeTensor(output_local);
        }
    }

private:
    AscendC::GlobalTensor<half> gamma_;
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> output_;
    uint32_t head_dim_ = 0;
    uint32_t head_rows_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR gamma, GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmaRmsNormHeads op;
    op.Init(gamma, input, output, tiling_data.head_dim, tiling_data.head_rows);
    op.Process(tiling_data.reciprocal);
}
