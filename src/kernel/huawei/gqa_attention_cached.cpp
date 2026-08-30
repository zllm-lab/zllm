#include "kernel_operator.h"

// Qwen3.5 4B 的 fused GQA：把当前 K/V 追加到各层常驻 cache，并完成 causal
// attention。每个 core 独占 query head；score、稳定 softmax 和 value 加权都
// 留在 UB 的向量流水，避免每个 score 往返 GM 或进入标量流水。
//
// 警告：kernel 内硬编码 Qwen3.5-4B 形状——query/output 行宽 4096（16 头 x
// head_dim 256）、KV 行宽 1024（4 KV 头）、GQA 分组 4、attention scale
// 0.0625（1/sqrt(256)）。接入新模型前必须把这些维度参数化（tiling 传入），
// 并同步放宽 host 入口 huawei.rs gqa_prefill_attention_cached 的 shape 校验。
class KernelGqaAttentionCached {
public:
    __aicore__ inline KernelGqaAttentionCached() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                GM_ADDR key_cache, GM_ADDR value_cache,
                                GM_ADDR position, GM_ADDR score_scratch, GM_ADDR output,
                                uint32_t batch, uint32_t cache_rows)
    {
        batch_ = batch;
        cache_rows_ = cache_rows;
        query_.SetGlobalBuffer((__gm__ half*)query, batch_ * 4096);
        key_.SetGlobalBuffer((__gm__ half*)key, batch_ * 1024);
        value_.SetGlobalBuffer((__gm__ half*)value, batch_ * 1024);
        key_cache_.SetGlobalBuffer((__gm__ half*)key_cache, cache_rows_ * 4096);
        value_cache_.SetGlobalBuffer((__gm__ half*)value_cache, cache_rows_ * 4096);
        position_.SetGlobalBuffer((__gm__ uint16_t*)position, 1);
        token_range_.SetGlobalBuffer((__gm__ uint16_t*)score_scratch, 3);
        score_offsets_.SetGlobalBuffer(
            (__gm__ uint16_t*)score_scratch + 16, cache_rows_);
        output_.SetGlobalBuffer((__gm__ half*)output, batch_ * 4096);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t position = position_.GetValue(0);
        const uint32_t token_begin = token_range_.GetValue(0);
        const uint32_t token_count = token_range_.GetValue(1);
        const uint32_t diagnostic_stage = token_range_.GetValue(2);
        const uint32_t token_end = token_begin + token_count < batch_
                                       ? token_begin + token_count
                                       : batch_;
        const uint32_t head_rows = (token_end - token_begin) * 16;

        // 先把移动数据与 attention 数学分开验证。正式执行固定使用 stage=10；
        // 诊断 stage 只由真机探针设置，不进入模型执行路径。
        if (diagnostic_stage < 2) {
            AscendC::TPipe diagnostic_pipe;
            AscendC::TQue<AscendC::QuePosition::VECIN, 1> query_queue;
            AscendC::TQue<AscendC::QuePosition::VECIN, 1> key_queue;
            AscendC::TQue<AscendC::QuePosition::VECIN, 1> value_queue;
            AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
            diagnostic_pipe.InitBuffer(query_queue, 1, 256 * sizeof(half));
            diagnostic_pipe.InitBuffer(output_queue, 1, 256 * sizeof(half));
            if (diagnostic_stage == 1) {
                diagnostic_pipe.InitBuffer(key_queue, 1, 256 * sizeof(half));
                diagnostic_pipe.InitBuffer(value_queue, 1, 256 * sizeof(half));
            }
            for (uint32_t head_row = core; head_row < head_rows;
                 head_row += cores) {
                const uint32_t row = token_begin + head_row / 16;
                const uint32_t query_head = head_row % 16;
                if (diagnostic_stage == 1 && row == token_begin) {
                    const uint32_t kv_head = query_head / 4;
                    for (uint32_t append_row = token_begin;
                         append_row < token_end; ++append_row) {
                        const uint32_t source = append_row * 1024 + kv_head * 256;
                        const uint32_t destination =
                            (position + append_row) * 4096 + query_head * 256;
                        auto key_local = key_queue.AllocTensor<half>();
                        auto value_local = value_queue.AllocTensor<half>();
                        AscendC::DataCopy(key_local, key_[source], 256);
                        AscendC::DataCopy(value_local, value_[source], 256);
                        key_queue.EnQue(key_local);
                        value_queue.EnQue(value_local);
                        key_local = key_queue.DeQue<half>();
                        value_local = value_queue.DeQue<half>();
                        AscendC::DataCopy(key_cache_[destination], key_local, 256);
                        AscendC::DataCopy(value_cache_[destination], value_local, 256);
                        key_queue.FreeTensor(key_local);
                        value_queue.FreeTensor(value_local);
                    }
                }
                auto query_local = query_queue.AllocTensor<half>();
                AscendC::DataCopy(
                    query_local, query_[row * 4096 + query_head * 256], 256);
                query_queue.EnQue(query_local);
                query_local = query_queue.DeQue<half>();
                auto output_local = output_queue.AllocTensor<half>();
                AscendC::DataCopy(output_local, query_local, 256);
                output_queue.EnQue(output_local);
                query_queue.FreeTensor(query_local);
                output_local = output_queue.DeQue<half>();
                AscendC::DataCopy(
                    output_[row * 4096 + query_head * 256], output_local, 256);
                output_queue.FreeTensor(output_local);
            }
            return;
        }

        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> query_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> key_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> value_queue;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
        AscendC::TBuf<AscendC::TPosition::VECCALC> product_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> sum_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_work_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> weight_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> weighted_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> score_slots_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> scores_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> score_work_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> score_offsets_buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> zero_offsets_buf;
        pipe.InitBuffer(query_queue, 1, 256 * sizeof(half));
        pipe.InitBuffer(key_queue, 1, 256 * sizeof(half));
        pipe.InitBuffer(value_queue, 1, 256 * sizeof(half));
        pipe.InitBuffer(output_queue, 1, 256 * sizeof(half));
        pipe.InitBuffer(product_buf, 256 * sizeof(half));
        pipe.InitBuffer(sum_buf, 16 * sizeof(half));
        pipe.InitBuffer(reduce_work_buf, 32);
        pipe.InitBuffer(weight_buf, 256 * sizeof(half));
        pipe.InitBuffer(weighted_buf, 256 * sizeof(half));
        // ReduceSum 的单值按 32-byte 对齐槽保存，再一次 Gather 成紧凑 score。
        pipe.InitBuffer(score_slots_buf, cache_rows_ * 16 * sizeof(half));
        pipe.InitBuffer(scores_buf, cache_rows_ * sizeof(half));
        pipe.InitBuffer(score_work_buf, cache_rows_ * sizeof(half));
        pipe.InitBuffer(score_offsets_buf, cache_rows_ * sizeof(uint16_t));
        pipe.InitBuffer(zero_offsets_buf, cache_rows_ * sizeof(uint16_t));
        const event_t load_to_store = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        const event_t store_to_load = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        if (diagnostic_stage == 2) {
            return;
        }

        auto score_offsets = score_offsets_buf.Get<uint16_t>();
        auto zero_offsets = zero_offsets_buf.Get<uint16_t>();
        AscendC::DataCopy(score_offsets, score_offsets_, cache_rows_);
        AscendC::Duplicate(
            zero_offsets, static_cast<uint16_t>(0), cache_rows_);
        AscendC::PipeBarrier<PIPE_ALL>();

        if (diagnostic_stage == 3) {
            return;
        }

        for (uint32_t head_row = core; head_row < head_rows; head_row += cores) {
            const uint32_t row = token_begin + head_row / 16;
            const uint32_t query_head = head_row % 16;
            const uint32_t kv_head = query_head / 4;
            if (row == token_begin) {
                for (uint32_t append_row = token_begin;
                     append_row < token_end;
                     ++append_row) {
                    const uint32_t source =
                        append_row * 1024 + kv_head * 256;
                    const uint32_t destination =
                        (position + append_row) * 4096 + query_head * 256;
                    auto key_local = key_queue.AllocTensor<half>();
                    auto value_local = value_queue.AllocTensor<half>();
                    AscendC::DataCopy(key_local, key_[source], 256);
                    AscendC::DataCopy(value_local, value_[source], 256);
                    key_queue.EnQue(key_local);
                    value_queue.EnQue(value_local);
                    key_local = key_queue.DeQue<half>();
                    value_local = value_queue.DeQue<half>();
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                        load_to_store);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                        load_to_store);
                    AscendC::DataCopy(
                        key_cache_[destination], key_local, 256);
                    AscendC::DataCopy(
                        value_cache_[destination], value_local, 256);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                        store_to_load);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                        store_to_load);
                    key_queue.FreeTensor(key_local);
                    value_queue.FreeTensor(value_local);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            auto query_local = query_queue.AllocTensor<half>();
            AscendC::DataCopy(
                query_local, query_[row * 4096 + query_head * 256], 256);
            query_queue.EnQue(query_local);
            query_local = query_queue.DeQue<half>();

            const uint32_t visible = position + row + 1;
            if (diagnostic_stage == 4) {
                auto key_local = key_queue.AllocTensor<half>();
                AscendC::DataCopy(
                    key_local,
                    key_cache_[query_head * 256], 256);
                key_queue.EnQue(key_local);
                key_local = key_queue.DeQue<half>();
                auto output_local = output_queue.AllocTensor<half>();
                AscendC::DataCopy(output_local, key_local, 256);
                output_queue.EnQue(output_local);
                query_queue.FreeTensor(query_local);
                key_queue.FreeTensor(key_local);
                output_local = output_queue.DeQue<half>();
                AscendC::DataCopy(
                    output_[row * 4096 + query_head * 256], output_local, 256);
                output_queue.FreeTensor(output_local);
                continue;
            }
            auto product = product_buf.Get<half>();
            auto sum_local = sum_buf.Get<half>();
            auto reduce_work = reduce_work_buf.Get<half>();
            auto score_slots = score_slots_buf.Get<half>();
            auto scores = scores_buf.Get<half>();
            auto score_work = score_work_buf.Get<half>();
            for (uint32_t cached = 0; cached < visible; ++cached) {
                auto key_local = key_queue.AllocTensor<half>();
                AscendC::DataCopy(
                    key_local,
                    key_cache_[cached * 4096 + query_head * 256], 256);
                key_queue.EnQue(key_local);
                key_local = key_queue.DeQue<half>();
                if (diagnostic_stage >= 5) {
                    AscendC::Mul(product, query_local, key_local, 256);
                }
                if (diagnostic_stage >= 6) {
                    AscendC::ReduceSum(sum_local, product, reduce_work, 256);
                    AscendC::Muls(
                        sum_local, sum_local, static_cast<half>(0.0625f), 1);
                    AscendC::Gather(
                        score_slots[cached * 16], sum_local, zero_offsets,
                        0, 1);
                }
                key_queue.FreeTensor(key_local);
            }
            if (diagnostic_stage >= 7) {
                AscendC::Gather(
                    scores, score_slots, score_offsets, 0, visible);
            }
            if (diagnostic_stage >= 8) {
                AscendC::ReduceMax(
                    sum_local, scores, reduce_work, visible);
                AscendC::Gather(
                    score_work, sum_local, zero_offsets, 0, visible);
                AscendC::Sub(scores, scores, score_work, visible);
                AscendC::Exp(scores, scores, visible);
                AscendC::ReduceSum(
                    sum_local, scores, reduce_work, visible);
                AscendC::Gather(
                    score_work, sum_local, zero_offsets, 0, visible);
                AscendC::Div(scores, scores, score_work, visible);
            }

            if (diagnostic_stage < 4) {
                auto output_local = output_queue.AllocTensor<half>();
                AscendC::DataCopy(output_local, query_local, 256);
                output_queue.EnQue(output_local);
                query_queue.FreeTensor(query_local);
                output_local = output_queue.DeQue<half>();
                AscendC::DataCopy(
                    output_[row * 4096 + query_head * 256], output_local, 256);
                output_queue.FreeTensor(output_local);
                continue;
            }

            if (diagnostic_stage == 5) {
                auto output_local = output_queue.AllocTensor<half>();
                AscendC::DataCopy(output_local, product, 256);
                output_queue.EnQue(output_local);
                query_queue.FreeTensor(query_local);
                output_local = output_queue.DeQue<half>();
                AscendC::DataCopy(
                    output_[row * 4096 + query_head * 256], output_local, 256);
                output_queue.FreeTensor(output_local);
                continue;
            }

            // stage 6 直接观察 ReduceSum，stage 7/8 分别观察 compact score
            // 和 softmax；三段输出都在 NPU 内广播，不经过标量流水。
            if (diagnostic_stage < 9) {
                auto output_local = output_queue.AllocTensor<half>();
                if (diagnostic_stage == 6) {
                    AscendC::Gather(output_local, sum_local, zero_offsets, 0, 256);
                } else {
                    AscendC::Gather(output_local, scores, zero_offsets, 0, 256);
                }
                output_queue.EnQue(output_local);
                query_queue.FreeTensor(query_local);
                output_local = output_queue.DeQue<half>();
                AscendC::DataCopy(
                    output_[row * 4096 + query_head * 256], output_local, 256);
                output_queue.FreeTensor(output_local);
                continue;
            }

            auto output_local = output_queue.AllocTensor<half>();
            auto weight_local = weight_buf.Get<half>();
            auto weighted = weighted_buf.Get<half>();
            AscendC::Duplicate(output_local, static_cast<half>(0.0f), 256);
            for (uint32_t cached = 0; cached < visible; ++cached) {
                auto value_local = value_queue.AllocTensor<half>();
                AscendC::DataCopy(
                    value_local,
                    value_cache_[cached * 4096 + query_head * 256], 256);
                value_queue.EnQue(value_local);
                value_local = value_queue.DeQue<half>();
                if (diagnostic_stage == 9) {
                    AscendC::DataCopy(output_local, value_local, 256);
                    value_queue.FreeTensor(value_local);
                    break;
                }
                AscendC::Gather(
                    weight_local, scores, zero_offsets,
                    cached * sizeof(half), 256);
                AscendC::Mul(weighted, value_local, weight_local, 256);
                AscendC::Add(output_local, output_local, weighted, 256);
                value_queue.FreeTensor(value_local);
            }
            output_queue.EnQue(output_local);
            query_queue.FreeTensor(query_local);
            output_local = output_queue.DeQue<half>();
            AscendC::DataCopy(
                output_[row * 4096 + query_head * 256], output_local, 256);
            output_queue.FreeTensor(output_local);
        }
    }

private:
    AscendC::GlobalTensor<half> query_, key_, value_, key_cache_, value_cache_;
    AscendC::GlobalTensor<uint16_t> position_, token_range_, score_offsets_;
    AscendC::GlobalTensor<half> output_;
    uint32_t batch_ = 0;
    uint32_t cache_rows_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR key_cache,
    GM_ADDR value_cache, GM_ADDR position, GM_ADDR score_scratch,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGqaAttentionCached op;
    op.Init(query, key, value, key_cache, value_cache, position, score_scratch, output,
            tiling_data.batch, tiling_data.cache_rows);
    op.Process();
}
