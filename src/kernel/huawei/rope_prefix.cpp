#include "kernel_operator.h"

// Qwen3.5 SplitHalf prefix RoPE。每个 AIV core 处理若干完整 head，Q/K 与结果
// 都保持 resident；host 只上传当前 token 小批次对应的 cos/sin 行。
class KernelRopePrefix {
public:
    __aicore__ inline KernelRopePrefix() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR cosine, GM_ADDR sine,
                                GM_ADDR output, uint32_t batch, uint32_t columns,
                                uint32_t head_dim, uint32_t rotary_dim)
    {
        batch_ = batch;
        columns_ = columns;
        head_dim_ = head_dim;
        rotary_dim_ = rotary_dim;
        half_ = rotary_dim / 2;
        heads_ = columns / head_dim;
        input_.SetGlobalBuffer((__gm__ half*)input, batch * columns);
        cosine_.SetGlobalBuffer((__gm__ half*)cosine, batch * half_);
        sine_.SetGlobalBuffer((__gm__ half*)sine, batch * half_);
        output_.SetGlobalBuffer((__gm__ half*)output, batch * columns);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t head_rows = batch_ * heads_;
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> rotary_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> tail_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> cosine_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> sine_queue;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
        AscendC::TBuf<AscendC::TPosition::VECCALC> temporary_buf;
        pipe.InitBuffer(rotary_queue, 1, rotary_dim_ * sizeof(half));
        pipe.InitBuffer(tail_queue, 1, (head_dim_ - rotary_dim_) * sizeof(half));
        pipe.InitBuffer(cosine_queue, 1, half_ * sizeof(half));
        pipe.InitBuffer(sine_queue, 1, half_ * sizeof(half));
        pipe.InitBuffer(output_queue, 1, rotary_dim_ * sizeof(half));
        pipe.InitBuffer(temporary_buf, half_ * sizeof(half));

        for (uint32_t head_row = core; head_row < head_rows; head_row += cores) {
            const uint32_t row = head_row / heads_;
            const uint32_t input_offset = head_row * head_dim_;
            auto rotary = rotary_queue.AllocTensor<half>();
            auto tail = tail_queue.AllocTensor<half>();
            auto cosine = cosine_queue.AllocTensor<half>();
            auto sine = sine_queue.AllocTensor<half>();
            AscendC::DataCopy(rotary, input_[input_offset], rotary_dim_);
            AscendC::DataCopy(tail, input_[input_offset + rotary_dim_], head_dim_ - rotary_dim_);
            AscendC::DataCopy(cosine, cosine_[row * half_], half_);
            AscendC::DataCopy(sine, sine_[row * half_], half_);
            rotary_queue.EnQue(rotary);
            tail_queue.EnQue(tail);
            cosine_queue.EnQue(cosine);
            sine_queue.EnQue(sine);
            rotary = rotary_queue.DeQue<half>();
            tail = tail_queue.DeQue<half>();
            cosine = cosine_queue.DeQue<half>();
            sine = sine_queue.DeQue<half>();

            auto rotated = output_queue.AllocTensor<half>();
            auto temporary = temporary_buf.Get<half>();
            AscendC::Mul(rotated, rotary, cosine, half_);
            AscendC::Mul(temporary, rotary[half_], sine, half_);
            AscendC::Sub(rotated, rotated, temporary, half_);
            AscendC::Mul(rotated[half_], rotary[half_], cosine, half_);
            AscendC::Mul(temporary, rotary, sine, half_);
            AscendC::Add(rotated[half_], rotated[half_], temporary, half_);
            output_queue.EnQue(rotated);
            rotary_queue.FreeTensor(rotary);
            cosine_queue.FreeTensor(cosine);
            sine_queue.FreeTensor(sine);
            rotated = output_queue.DeQue<half>();
            AscendC::DataCopy(output_[input_offset], rotated, rotary_dim_);
            AscendC::DataCopy(output_[input_offset + rotary_dim_], tail,
                              head_dim_ - rotary_dim_);
            output_queue.FreeTensor(rotated);
            tail_queue.FreeTensor(tail);
        }
    }

private:
    AscendC::GlobalTensor<half> input_;
    AscendC::GlobalTensor<half> cosine_;
    AscendC::GlobalTensor<half> sine_;
    AscendC::GlobalTensor<half> output_;
    uint32_t batch_ = 0;
    uint32_t columns_ = 0;
    uint32_t head_dim_ = 0;
    uint32_t rotary_dim_ = 0;
    uint32_t half_ = 0;
    uint32_t heads_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR input, GM_ADDR cosine, GM_ADDR sine, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelRopePrefix op;
    op.Init(input, cosine, sine, output, tiling_data.batch, tiling_data.columns,
            tiling_data.head_dim, tiling_data.rotary_dim);
    op.Process();
}
