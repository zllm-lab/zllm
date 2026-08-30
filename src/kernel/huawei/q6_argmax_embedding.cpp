#include "kernel_operator.h"

// 把输出选择与下一 token 的 Q6_K embedding lookup 合成一个 NPU kernel。
// decode 热路径只把 token id 作为控制结果返回 host；下一轮 hidden 已经是
// resident AiTensor，不再从 GGUF 读取、CPU 反量化或重新上传 embedding。
class KernelQ6ArgmaxEmbedding {
public:
    __aicore__ inline KernelQ6ArgmaxEmbedding() {}

    __aicore__ inline void Init(
        GM_ADDR logits, GM_ADDR embedding, GM_ADDR excluded,
        GM_ADDR hidden, GM_ADDR token, uint32_t vocab, uint32_t columns,
        uint32_t excluded_count)
    {
        vocab_ = vocab;
        columns_ = columns;
        excluded_count_ = excluded_count;
        blocks_per_row_ = columns / 256;
        logits_.SetGlobalBuffer((__gm__ half*)logits, vocab);
        embedding_bytes_ = (__gm__ uint8_t*)embedding;
        embedding_half_.SetGlobalBuffer(
            (__gm__ half*)embedding, vocab * blocks_per_row_ * 105);
        excluded_.SetGlobalBuffer((__gm__ int32_t*)excluded, excluded_count);
        hidden_.SetGlobalBuffer((__gm__ half*)hidden, columns);
        token_.SetGlobalBuffer((__gm__ int32_t*)token, 1);
    }

    __aicore__ inline void Process()
    {
        float best = -3.402823466e+38f;
        uint32_t best_index = 0;
        bool found = false;
        for (uint32_t index = 0; index < vocab_; ++index) {
            bool skip = false;
            for (uint32_t excluded_index = 0;
                 excluded_index < excluded_count_; ++excluded_index) {
                if ((uint32_t)excluded_.GetValue(excluded_index) == index) {
                    skip = true;
                    break;
                }
            }
            const float value = (float)logits_.GetValue(index);
            if (!skip && (!found || value > best)) {
                found = true;
                best = value;
                best_index = index;
            }
        }
        if (!found) {
            token_.SetValue(0, -1);
            return;
        }
        token_.SetValue(0, (int32_t)best_index);

        const uint32_t row_block = best_index * blocks_per_row_;
        for (uint32_t column_block = 0; column_block < blocks_per_row_;
             ++column_block) {
            const uint32_t block = row_block + column_block;
            const uint32_t base = block * 210;
            const float d = (float)embedding_half_.GetValue(block * 105 + 104);
            for (uint32_t half_block = 0; half_block < 2; ++half_block) {
                const uint32_t low_base = base + half_block * 64;
                const uint32_t high_base = base + 128 + half_block * 32;
                const uint32_t scale_base = base + 192 + half_block * 8;
                const uint32_t output_base =
                    column_block * 256 + half_block * 128;
                for (uint32_t index = 0; index < 32; ++index) {
                    const uint32_t scale_index = index / 16;
                    const uint8_t high = embedding_bytes_[high_base + index];
                    const int32_t q1 =
                        ((embedding_bytes_[low_base + index] & 0x0f) |
                         (((high >> 0) & 3) << 4)) - 32;
                    const int32_t q2 =
                        ((embedding_bytes_[low_base + index + 32] & 0x0f) |
                         (((high >> 2) & 3) << 4)) - 32;
                    const int32_t q3 =
                        ((embedding_bytes_[low_base + index] >> 4) |
                         (((high >> 4) & 3) << 4)) - 32;
                    const int32_t q4 =
                        ((embedding_bytes_[low_base + index + 32] >> 4) |
                         (((high >> 6) & 3) << 4)) - 32;
                    const float s1 = d * (float)((int8_t)
                        embedding_bytes_[scale_base + scale_index]);
                    const float s2 = d * (float)((int8_t)
                        embedding_bytes_[scale_base + scale_index + 2]);
                    const float s3 = d * (float)((int8_t)
                        embedding_bytes_[scale_base + scale_index + 4]);
                    const float s4 = d * (float)((int8_t)
                        embedding_bytes_[scale_base + scale_index + 6]);
                    hidden_.SetValue(output_base + index, (half)(s1 * q1));
                    hidden_.SetValue(output_base + index + 32, (half)(s2 * q2));
                    hidden_.SetValue(output_base + index + 64, (half)(s3 * q3));
                    hidden_.SetValue(output_base + index + 96, (half)(s4 * q4));
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<half> logits_;
    __gm__ uint8_t* embedding_bytes_;
    AscendC::GlobalTensor<half> embedding_half_;
    AscendC::GlobalTensor<int32_t> excluded_;
    AscendC::GlobalTensor<half> hidden_;
    AscendC::GlobalTensor<int32_t> token_;
    uint32_t vocab_ = 0;
    uint32_t columns_ = 0;
    uint32_t excluded_count_ = 0;
    uint32_t blocks_per_row_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR logits, GM_ADDR embedding, GM_ADDR excluded,
    GM_ADDR hidden, GM_ADDR token, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQ6ArgmaxEmbedding op;
    op.Init(logits, embedding, excluded, hidden, token,
            tiling_data.vocab, tiling_data.columns,
            tiling_data.excluded_count);
    op.Process();
}
