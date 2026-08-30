#include "kernel_operator.h"

// 对每个 16-token chunk 求 (I + A)^-1。Gram、门控缩放、递推中间值和
// 输出都保持 FP16；这里故意不引入 F32 oracle 路径，用来验证 FP16 本身
// 可以正确表达 chunked Gated DeltaNet 的短递推。
class KernelFp16ChunkSolve {
public:
    __aicore__ inline KernelFp16ChunkSolve() {}

    __aicore__ inline void Init(GM_ADDR gram, GM_ADDR scales,
                                GM_ADDR identity, GM_ADDR output,
                                uint32_t tasks)
    {
        tasks_ = tasks;
        gram_.SetGlobalBuffer((__gm__ half*)gram, tasks * matrix_elements);
        scales_.SetGlobalBuffer((__gm__ half*)scales,
                                tasks * scale_elements);
        identity_.SetGlobalBuffer((__gm__ half*)identity, matrix_elements);
        output_.SetGlobalBuffer((__gm__ half*)output,
                                tasks * matrix_elements);
    }

    __aicore__ inline void Process()
    {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> gram_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> scale_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> identity_queue;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
        AscendC::TBuf<AscendC::TPosition::VECCALC> lower_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> inverse_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> broadcast_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> product_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_buffer;
        pipe.InitBuffer(gram_queue, 1, matrix_elements * sizeof(half));
        pipe.InitBuffer(scale_queue, 1, scale_elements * sizeof(half));
        pipe.InitBuffer(identity_queue, 1,
                        matrix_elements * sizeof(half));
        pipe.InitBuffer(output_queue, 1, matrix_elements * sizeof(half));
        pipe.InitBuffer(lower_buffer, matrix_elements * sizeof(half));
        pipe.InitBuffer(inverse_buffer, matrix_elements * sizeof(half));
        pipe.InitBuffer(broadcast_buffer, tile * sizeof(half));
        pipe.InitBuffer(product_buffer, tile * sizeof(half));
        pipe.InitBuffer(offset_buffer, tile * sizeof(uint16_t));
        auto lower = lower_buffer.Get<half>();
        auto inverse = inverse_buffer.Get<half>();
        auto broadcast = broadcast_buffer.Get<half>();
        auto product = product_buffer.Get<half>();
        auto offsets = offset_buffer.Get<uint16_t>();

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        auto identity_local = identity_queue.AllocTensor<half>();
        AscendC::DataCopy(identity_local, identity_, matrix_elements);
        identity_queue.EnQue(identity_local);
        identity_local = identity_queue.DeQue<half>();
        for (uint32_t task = core; task < tasks_; task += cores) {
            const uint32_t matrix_base = task * matrix_elements;
            const uint32_t scale_base = task * scale_elements;
            auto gram_local = gram_queue.AllocTensor<half>();
            auto scales_local = scale_queue.AllocTensor<half>();
            AscendC::DataCopy(gram_local, gram_[matrix_base], matrix_elements);
            AscendC::DataCopy(scales_local, scales_[scale_base], scale_elements);
            gram_queue.EnQue(gram_local);
            scale_queue.EnQue(scales_local);
            gram_local = gram_queue.DeQue<half>();
            scales_local = scale_queue.DeQue<half>();

            AscendC::Duplicate(inverse, static_cast<half>(0.0f),
                               matrix_elements);
            AscendC::Duplicate(lower, static_cast<half>(0.0f),
                               matrix_elements);
            for (uint32_t row = 1; row < tile; ++row) {
                // lower[i,j] = gram[i,j] * beta[i] *
                // exp(log_b[i] - log_b[j])，只计算严格下三角。
                // 这样不再分别保存会下溢的 b 与会溢出的 1/b。
                AscendC::Duplicate(
                    offsets, static_cast<uint16_t>(row * sizeof(half)), row);
                AscendC::Gather(broadcast, scales_local, offsets, 0, row);
                AscendC::Mul(lower[row * tile], gram_local[row * tile],
                             broadcast, row);
                AscendC::Gather(product, scales_local[tile], offsets, 0,
                                row);
                AscendC::Sub(product, product, scales_local[tile], row);
                AscendC::Exp(product, product, row);
                AscendC::Mul(lower[row * tile], lower[row * tile], product,
                             row);
            }

            // 先求严格下三角，再用一次对齐 Vector Add 加入单位阵。
            // 这避免 9010 对非 32-byte 对齐地址执行单元素写，
            // 后续 Cube W/U 可以直接消费完整 inverse。
            for (uint32_t row = 0; row < tile; ++row) {
                if (row != 0) {
                    AscendC::Muls(inverse[row * tile], lower[row * tile],
                                  static_cast<half>(-1.0f), row);
                }
                for (uint32_t inner = 0; inner < row; ++inner) {
                    AscendC::Duplicate(
                        offsets,
                        static_cast<uint16_t>(inner * sizeof(half)), tile);
                    AscendC::Gather(broadcast, lower[row * tile], offsets,
                                    0, tile);
                    AscendC::Muls(broadcast, broadcast,
                                  static_cast<half>(-1.0f), tile);
                    AscendC::Mul(product, inverse[inner * tile], broadcast,
                                 tile);
                    AscendC::Add(inverse[row * tile], inverse[row * tile],
                                 product, tile);
                }
            }
            auto output_local = output_queue.AllocTensor<half>();
            AscendC::Add(output_local, inverse, identity_local,
                         matrix_elements);
            output_queue.EnQue(output_local);
            output_local = output_queue.DeQue<half>();
            AscendC::DataCopy(output_[matrix_base], output_local,
                              matrix_elements);
            output_queue.FreeTensor(output_local);
            gram_queue.FreeTensor(gram_local);
            scale_queue.FreeTensor(scales_local);
        }
        identity_queue.FreeTensor(identity_local);
    }

private:
    static constexpr uint32_t tile = 16;
    static constexpr uint32_t matrix_elements = tile * tile;
    static constexpr uint32_t scale_elements = tile * 2;

    AscendC::GlobalTensor<half> gram_;
    AscendC::GlobalTensor<half> scales_;
    AscendC::GlobalTensor<half> identity_;
    AscendC::GlobalTensor<half> output_;
    uint32_t tasks_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR gram, GM_ADDR scales, GM_ADDR identity, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFp16ChunkSolve op;
    op.Init(gram, scales, identity, output, tiling_data.tasks);
    op.Process();
}
