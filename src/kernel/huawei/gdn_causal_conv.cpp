#include "kernel_operator.h"

// Qwen3.5-4B Gated DeltaNet 的 1024-token depthwise causal conv + SiLU。
// 每个 core 独占连续 channel tile，4-tap state 在 UB 中推进完整序列，
// 只在开始/结束访问 state；输出直接留在 NPU tensor 中交给 chunk prepare。
class KernelGdnCausalConv {
public:
    __aicore__ inline KernelGdnCausalConv() {}

    __aicore__ inline void Init(
        GM_ADDR qkv, GM_ADDR weight, GM_ADDR state, GM_ADDR mixed,
        uint32_t batch, uint32_t channels, uint32_t kernel_size)
    {
        batch_ = batch;
        channels_ = channels;
        kernel_size_ = kernel_size;
        qkv_.SetGlobalBuffer((__gm__ half*)qkv, batch * channels);
        weight_.SetGlobalBuffer((__gm__ half*)weight,
                                channels * kernel_size);
        state_.SetGlobalBuffer((__gm__ half*)state,
                               channels * kernel_size);
        mixed_.SetGlobalBuffer((__gm__ half*)mixed, batch * channels);
    }

    __aicore__ inline void Process()
    {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> interleaved_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> weight_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> state_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> current_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> work_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_buffer;
        pipe.InitBuffer(interleaved_buffer,
                        channel_tile * max_kernel * sizeof(half));
        pipe.InitBuffer(weight_buffer,
                        channel_tile * max_kernel * sizeof(half));
        pipe.InitBuffer(state_buffer,
                        channel_tile * max_kernel * sizeof(half));
        pipe.InitBuffer(current_buffer, channel_tile * sizeof(half));
        pipe.InitBuffer(output_buffer, channel_tile * sizeof(half));
        pipe.InitBuffer(work_buffer, channel_tile * sizeof(half));
        pipe.InitBuffer(offset_buffer, channel_tile * sizeof(uint16_t));

        auto interleaved = interleaved_buffer.Get<half>();
        auto weights = weight_buffer.Get<half>();
        auto states = state_buffer.Get<half>();
        auto current = current_buffer.Get<half>();
        auto output = output_buffer.Get<half>();
        auto work = work_buffer.Get<half>();
        auto offsets = offset_buffer.Get<uint16_t>();
        auto signed_offsets = offsets.ReinterpretCast<int16_t>();
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t tiles = channels_ / channel_tile;

        for (uint32_t tile = core; tile < tiles; tile += cores) {
            const uint32_t channel_begin = tile * channel_tile;
            const uint32_t parameter_begin = channel_begin * kernel_size_;
            AscendC::DataCopy(interleaved, weight_[parameter_begin],
                              channel_tile * kernel_size_);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t slot = 0; slot < kernel_size_; ++slot) {
                AscendC::CreateVecIndex(signed_offsets,
                                        static_cast<int16_t>(0),
                                        channel_tile);
                AscendC::Muls(signed_offsets, signed_offsets,
                              static_cast<int16_t>(kernel_size_ *
                                                   sizeof(half)),
                              channel_tile);
                AscendC::Adds(signed_offsets, signed_offsets,
                              static_cast<int16_t>(slot * sizeof(half)),
                              channel_tile);
                AscendC::Gather(weights[slot * channel_tile], interleaved,
                                offsets, 0, channel_tile);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(interleaved, state_[parameter_begin],
                              channel_tile * kernel_size_);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t slot = 0; slot < kernel_size_; ++slot) {
                AscendC::CreateVecIndex(signed_offsets,
                                        static_cast<int16_t>(0),
                                        channel_tile);
                AscendC::Muls(signed_offsets, signed_offsets,
                              static_cast<int16_t>(kernel_size_ *
                                                   sizeof(half)),
                              channel_tile);
                AscendC::Adds(signed_offsets, signed_offsets,
                              static_cast<int16_t>(slot * sizeof(half)),
                              channel_tile);
                AscendC::Gather(states[slot * channel_tile], interleaved,
                                offsets, 0, channel_tile);
            }

            for (uint32_t token = 0; token < batch_; ++token) {
                AscendC::Duplicate(output, static_cast<half>(0.0f),
                                   channel_tile);
                for (uint32_t slot = 0; slot < kernel_size_; ++slot) {
                    const int32_t source_token =
                        static_cast<int32_t>(token + slot) -
                        static_cast<int32_t>(kernel_size_ - 1);
                    if (source_token < 0) {
                        const uint32_t state_slot = static_cast<uint32_t>(
                            source_token + static_cast<int32_t>(kernel_size_));
                        AscendC::Adds(
                            current, states[state_slot * channel_tile],
                            static_cast<half>(0.0f), channel_tile);
                    } else {
                        AscendC::DataCopy(
                            current,
                            qkv_[static_cast<uint32_t>(source_token) *
                                     channels_ + channel_begin],
                            channel_tile);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Mul(work, current,
                                 weights[slot * channel_tile], channel_tile);
                    AscendC::Add(output, output, work, channel_tile);
                }
                AscendC::Muls(work, output, static_cast<half>(-1.0f),
                              channel_tile);
                AscendC::Exp(work, work, channel_tile);
                AscendC::Adds(work, work, static_cast<half>(1.0f),
                              channel_tile);
                AscendC::Div(output, output, work, channel_tile);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(
                    mixed_[token * channels_ + channel_begin], output,
                    channel_tile);
            }

            // 末 4 个输入直接成为下一次 decode 的 channel-major state。
            // 使用已经在标量 B1 kernel 验证过的 GM GetValue/SetValue，避免
            // 9010 对 UB 标量重排生成不稳定指令。
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t channel = 0; channel < channel_tile; ++channel) {
                for (uint32_t slot = 0; slot < kernel_size_; ++slot) {
                    state_.SetValue(
                        parameter_begin + channel * kernel_size_ + slot,
                        qkv_.GetValue(
                            (batch_ - kernel_size_ + slot) * channels_ +
                            channel_begin + channel));
                }
            }
        }
    }

private:
    static constexpr uint32_t channel_tile = 128;
    static constexpr uint32_t max_kernel = 4;
    AscendC::GlobalTensor<half> qkv_, weight_, state_, mixed_;
    uint32_t batch_ = 0;
    uint32_t channels_ = 0;
    uint32_t kernel_size_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR qkv, GM_ADDR weight, GM_ADDR state, GM_ADDR mixed,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    (void)workspace;
    KernelGdnCausalConv op;
    op.Init(qkv, weight, state, mixed, tiling_data.batch,
            tiling_data.channels, tiling_data.kernel_size);
    op.Process();
}
