#include "kernel_operator.h"

// Qwen3.5-4B 的 1024-token Gated DeltaNet 前处理。每个 core 独占两个
// key head 及其映射的四个 value head，causal conv state 因而可在 UB 中
// 连续推进；chunk 之间没有 GM state 往返。输出直接采用后续 Cube kernel
// 的 A2/B2 tile 布局，不经过 CPU 重排。
class KernelGdnChunkPrepare {
public:
    __aicore__ inline KernelGdnChunkPrepare() {}

    __aicore__ inline void Init(
        GM_ADDR mixed, GM_ADDR alpha, GM_ADDR beta,
        GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR identity,
        GM_ADDR keys, GM_ADDR scales, GM_ADDR negative_vectors,
        GM_ADDR cumsum_vectors, GM_ADDR q_b, GM_ADDR k_inv_b,
        GM_ADDR r_transposed, GM_ADDR b_identity)
    {
        mixed_.SetGlobalBuffer((__gm__ half*)mixed, rows * conv_dim);
        alpha_.SetGlobalBuffer((__gm__ half*)alpha, rows * value_heads);
        beta_.SetGlobalBuffer((__gm__ half*)beta, rows * value_heads);
        a_log_.SetGlobalBuffer((__gm__ half*)a_log, value_heads);
        dt_bias_.SetGlobalBuffer((__gm__ half*)dt_bias, value_heads);
        identity_.SetGlobalBuffer((__gm__ half*)identity, tile_elements);
        keys_.SetGlobalBuffer((__gm__ half*)keys,
                              tasks * short_elements);
        scales_.SetGlobalBuffer((__gm__ half*)scales,
                                tasks * scale_elements);
        negative_vectors_.SetGlobalBuffer((__gm__ half*)negative_vectors,
                                          tasks * short_elements);
        cumsum_vectors_.SetGlobalBuffer((__gm__ half*)cumsum_vectors,
                                        tasks * short_elements);
        q_b_.SetGlobalBuffer((__gm__ half*)q_b,
                             tasks * short_elements);
        k_inv_b_.SetGlobalBuffer((__gm__ half*)k_inv_b,
                                 tasks * short_elements);
        r_transposed_.SetGlobalBuffer((__gm__ half*)r_transposed,
                                      tasks * short_elements);
        b_identity_.SetGlobalBuffer((__gm__ half*)b_identity,
                                    tasks * tile_elements);
    }

    __aicore__ inline void Normalize(
        AscendC::LocalTensor<half> vector, float scale,
        AscendC::LocalTensor<half> square,
        AscendC::LocalTensor<half> sum,
        AscendC::LocalTensor<half> reduce_work,
        AscendC::LocalTensor<half> broadcast,
        AscendC::LocalTensor<uint16_t> offsets)
    {
        AscendC::Mul(square, vector, vector, side);
        AscendC::ReduceSum(sum, square, reduce_work, side);
        AscendC::Adds(sum, sum, static_cast<half>(1.0e-6f), 1);
        AscendC::Rsqrt(sum, sum, 1);
        AscendC::Duplicate(offsets, static_cast<uint16_t>(0), side);
        AscendC::Gather(broadcast, sum, offsets, 0, side);
        AscendC::Muls(broadcast, broadcast, static_cast<half>(scale), side);
        AscendC::Mul(vector, vector, broadcast, side);
    }

    __aicore__ inline void CopyPackedTk(
        AscendC::LocalTensor<half> destination,
        AscendC::LocalTensor<half> source, float scale,
        AscendC::LocalTensor<uint16_t> packing_offsets)
    {
        AscendC::Gather(destination, source, packing_offsets, 0,
                        short_elements);
        if (scale != 1.0f) {
            AscendC::Muls(destination, destination,
                          static_cast<half>(scale), short_elements);
        }
    }

    __aicore__ inline void CopyPackedB(
        AscendC::LocalTensor<half> destination,
        AscendC::LocalTensor<half> source,
        AscendC::LocalTensor<half> row_scale,
        AscendC::LocalTensor<half> broadcast,
        AscendC::LocalTensor<uint16_t> packing_offsets,
        AscendC::LocalTensor<uint16_t> scale_offsets)
    {
        AscendC::Gather(destination, source, packing_offsets, 0,
                        short_elements);
        AscendC::Gather(broadcast, row_scale, scale_offsets, 0,
                        short_elements);
        AscendC::Mul(destination, destination, broadcast, short_elements);
    }

    __aicore__ inline void CopyPackedTkRows(
        AscendC::LocalTensor<half> destination,
        AscendC::LocalTensor<half> source,
        AscendC::LocalTensor<half> row_scale,
        AscendC::LocalTensor<half> broadcast,
        AscendC::LocalTensor<uint16_t> packing_offsets,
        AscendC::LocalTensor<uint16_t> scale_offsets)
    {
        AscendC::Gather(destination, source, packing_offsets, 0,
                        short_elements);
        AscendC::Gather(broadcast, row_scale, scale_offsets, 0,
                        short_elements);
        AscendC::Mul(destination, destination, broadcast, short_elements);
    }

    __aicore__ inline uint32_t ChannelBase(uint32_t key_head,
                                            uint32_t group) const
    {
        if (group == 0) return key_head * side;
        if (group == 1) return key_dim + key_head * side;
        const uint32_t value_head =
            group == 2 ? key_head : key_head + key_heads;
        return key_dim * 2 + value_head * side;
    }

    __aicore__ inline void Process()
    {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
        AscendC::TBuf<AscendC::TPosition::VECCALC> chunk_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> packed_keys_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> packed_negative_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> packed_cumsum_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> packed_q_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> packed_k_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> packed_r_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> scale_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> identity_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> transpose_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> square_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> sum_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_work_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> broadcast_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> packing_offset_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> alpha_block_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> beta_block_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> parameter_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> control_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> control_rows_buffer;

        pipe.InitBuffer(input_queue, 1, side * sizeof(half));
        pipe.InitBuffer(chunk_buffer,
                        groups * tile * side * sizeof(half));
        pipe.InitBuffer(packed_keys_buffer, short_elements * sizeof(half));
        pipe.InitBuffer(packed_negative_buffer,
                        short_elements * sizeof(half));
        pipe.InitBuffer(packed_cumsum_buffer,
                        short_elements * sizeof(half));
        pipe.InitBuffer(packed_q_buffer, short_elements * sizeof(half));
        pipe.InitBuffer(packed_k_buffer, short_elements * sizeof(half));
        pipe.InitBuffer(packed_r_buffer, short_elements * sizeof(half));
        pipe.InitBuffer(scale_buffer, scale_elements * sizeof(half));
        pipe.InitBuffer(identity_buffer, tile_elements * sizeof(half));
        pipe.InitBuffer(transpose_buffer, tile_elements * sizeof(half));
        pipe.InitBuffer(square_buffer, side * sizeof(half));
        pipe.InitBuffer(sum_buffer, 32);
        pipe.InitBuffer(reduce_work_buffer, 32);
        pipe.InitBuffer(broadcast_buffer, short_elements * sizeof(half));
        pipe.InitBuffer(offset_buffer, side * sizeof(uint16_t));
        pipe.InitBuffer(packing_offset_buffer,
                        packing_tables * short_elements * sizeof(uint16_t));
        pipe.InitBuffer(alpha_block_buffer,
                        tile * value_heads * sizeof(half));
        pipe.InitBuffer(beta_block_buffer,
                        tile * value_heads * sizeof(half));
        pipe.InitBuffer(parameter_buffer,
                        value_heads * 2 * sizeof(half));
        pipe.InitBuffer(control_buffer,
                        tile * control_vectors * sizeof(half));
        pipe.InitBuffer(control_rows_buffer,
                        tile_elements * 2 * sizeof(half));

        auto chunk_data = chunk_buffer.Get<half>();
        auto packed_keys = packed_keys_buffer.Get<half>();
        auto packed_negative = packed_negative_buffer.Get<half>();
        auto packed_cumsum = packed_cumsum_buffer.Get<half>();
        auto packed_q = packed_q_buffer.Get<half>();
        auto packed_k = packed_k_buffer.Get<half>();
        auto packed_r = packed_r_buffer.Get<half>();
        auto scale_local = scale_buffer.Get<half>();
        auto identity_local = identity_buffer.Get<half>();
        auto transpose_local = transpose_buffer.Get<half>();
        auto square_local = square_buffer.Get<half>();
        auto sum_local = sum_buffer.Get<half>();
        auto reduce_work = reduce_work_buffer.Get<half>();
        auto broadcast_local = broadcast_buffer.Get<half>();
        auto offsets_local = offset_buffer.Get<uint16_t>();
        auto packing_offsets = packing_offset_buffer.Get<uint16_t>();
        auto tk_offsets = packing_offsets;
        auto b_offsets = packing_offsets[short_elements];
        auto tk_scale_offsets = packing_offsets[short_elements * 2];
        auto b_scale_offsets = packing_offsets[short_elements * 3];
        auto alpha_block = alpha_block_buffer.Get<half>();
        auto beta_block = beta_block_buffer.Get<half>();
        auto parameters = parameter_buffer.Get<half>();
        auto controls = control_buffer.Get<half>();
        auto alpha_values = controls;
        auto beta_values = controls[tile];
        auto cumulative = controls[tile * 2];
        auto control_work = controls[tile * 3];
        auto row_scale = controls[tile * 5];
        auto control_rows = control_rows_buffer.Get<half>();
        auto cumulative_rows = control_rows;
        auto row_offsets =
            control_rows[tile_elements].ReinterpretCast<uint16_t>();
        const auto vector_to_store = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
        const auto store_to_vector = static_cast<event_t>(
            pipe.FetchEventID(AscendC::HardEvent::MTE3_V));

        AscendC::DataCopy(parameters, a_log_, value_heads);
        AscendC::DataCopy(parameters[value_heads], dt_bias_, value_heads);
        AscendC::DataCopy(identity_local, identity_, tile_elements);
        AscendC::PipeBarrier<PIPE_ALL>();

        // 四张 byte-offset 表只构造一次，后续所有 task 用整块 Gather
        // 完成 TK/B2 重排和行 scale 广播，避免每个输出执行 128 次
        // 16-element 小向量指令。
        auto signed_tk = tk_offsets.ReinterpretCast<int16_t>();
        auto signed_b = b_offsets.ReinterpretCast<int16_t>();
        auto signed_b_scale = b_scale_offsets.ReinterpretCast<int16_t>();
        for (uint32_t column_tile = 0; column_tile < side / tile;
             ++column_tile) {
            for (uint32_t row = 0; row < tile; ++row) {
                const uint32_t segment =
                    column_tile * tile_elements + row * tile;
                const int16_t source = static_cast<int16_t>(
                    row * side + column_tile * tile);
                AscendC::CreateVecIndex(signed_tk[segment], source, tile);
                AscendC::Muls(signed_tk[segment], signed_tk[segment],
                              static_cast<int16_t>(sizeof(half)), tile);
                AscendC::Duplicate(
                    tk_scale_offsets[segment],
                    static_cast<uint16_t>(row * sizeof(half)), tile);
            }
            for (uint32_t column = 0; column < tile; ++column) {
                const uint32_t segment =
                    column_tile * tile_elements + column * tile;
                AscendC::CreateVecIndex(signed_b[segment],
                                        static_cast<int16_t>(0), tile);
                AscendC::Muls(
                    signed_b[segment], signed_b[segment],
                    static_cast<int16_t>(side * sizeof(half)), tile);
                AscendC::Adds(
                    signed_b[segment], signed_b[segment],
                    static_cast<int16_t>(
                        (column_tile * tile + column) * sizeof(half)),
                    tile);
                AscendC::CreateVecIndex(
                    signed_b_scale[segment], static_cast<int16_t>(0), tile);
                AscendC::Muls(
                    signed_b_scale[segment], signed_b_scale[segment],
                    static_cast<int16_t>(sizeof(half)), tile);
            }
        }

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        for (uint32_t key_head = core; key_head < key_heads;
             key_head += cores) {
            const uint32_t value_head_0 = key_head;
            const uint32_t value_head_1 = key_head + key_heads;
            for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                for (uint32_t row = 0; row < tile; ++row) {
                    const uint32_t token = chunk * tile + row;
                    for (uint32_t group = 0; group < groups; ++group) {
                        const uint32_t channel_base =
                            ChannelBase(key_head, group);
                        auto input = input_queue.AllocTensor<half>();
                        AscendC::DataCopy(
                            input,
                            mixed_[token * conv_dim + channel_base],
                            side);
                        input_queue.EnQue(input);
                        input = input_queue.DeQue<half>();
                        auto destination =
                            chunk_data[(group * tile + row) * side];
                        AscendC::Adds(destination, input,
                                      static_cast<half>(0.0f), side);
                        input_queue.FreeTensor(input);
                    }
                    Normalize(chunk_data[row * side], query_scale,
                              square_local, sum_local, reduce_work,
                              broadcast_local, offsets_local);
                    Normalize(chunk_data[(tile + row) * side], 1.0f,
                              square_local, sum_local, reduce_work,
                              broadcast_local, offsets_local);
                }
                const uint32_t control_base =
                    chunk * tile * value_heads;
                AscendC::DataCopy(alpha_block, alpha_[control_base],
                                  tile * value_heads);
                AscendC::DataCopy(beta_block, beta_[control_base],
                                  tile * value_heads);
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t mapped = 0; mapped < 2; ++mapped) {
                    const uint32_t value_head =
                        mapped == 0 ? value_head_0 : value_head_1;
                    const uint32_t task = value_head * chunks + chunk;
                    const uint32_t short_base = task * short_elements;
                    const uint32_t value_group = 2 + mapped;

                    // 控制量按 16 token 一次向量化。prefix product 放在
                    // 16x16 对齐行中递推，规避 9010 的非对齐 Vector 写。
                    auto signed_offsets =
                        offsets_local.ReinterpretCast<int16_t>();
                    AscendC::CreateVecIndex(
                        signed_offsets, static_cast<int16_t>(0), tile);
                    AscendC::Muls(
                        signed_offsets, signed_offsets,
                        static_cast<int16_t>(value_heads * sizeof(half)),
                        tile);
                    AscendC::Adds(
                        signed_offsets, signed_offsets,
                        static_cast<int16_t>(value_head * sizeof(half)),
                        tile);
                    AscendC::Gather(alpha_values, alpha_block,
                                    offsets_local, 0, tile);
                    AscendC::Gather(beta_values, beta_block,
                                    offsets_local, 0, tile);

                    AscendC::Duplicate(
                        offsets_local,
                        static_cast<uint16_t>(value_head * sizeof(half)),
                        tile);
                    AscendC::Gather(
                        broadcast_local, parameters[value_heads],
                        offsets_local, 0, tile);
                    AscendC::Add(alpha_values, alpha_values,
                                 broadcast_local, tile);
                    AscendC::Exp(control_work, alpha_values, tile);
                    AscendC::Adds(control_work, control_work,
                                  static_cast<half>(1.0f), tile);
                    AscendC::Ln(alpha_values, control_work, tile);
                    AscendC::Gather(broadcast_local, parameters,
                                    offsets_local, 0, tile);
                    AscendC::Exp(broadcast_local, broadcast_local, tile);
                    AscendC::Mul(control_work, alpha_values,
                                 broadcast_local, tile);
                    AscendC::Muls(control_work, control_work,
                                  static_cast<half>(-1.0f), tile);
                    // 直接累计 log(decay)。真实模型一个 16-token chunk 的
                    // decay 乘积可低至 1e-20，FP16 保存 b 与 1/b 会下溢/
                    // 上溢；log_b 始终落在可表示范围。
                    AscendC::Adds(cumulative, control_work,
                                  static_cast<half>(0.0f), tile);

                    for (uint32_t row = 0; row < tile; ++row) {
                        AscendC::Duplicate(
                            row_offsets[row * tile],
                            static_cast<uint16_t>(row * sizeof(half)),
                            tile);
                    }
                    AscendC::Gather(cumulative_rows, cumulative,
                                    row_offsets, 0, tile_elements);
                    for (uint32_t row = 1; row < tile; ++row) {
                        AscendC::Add(cumulative_rows[row * tile],
                                     cumulative_rows[row * tile],
                                     cumulative_rows[(row - 1) * tile],
                                     tile);
                    }

                    AscendC::CreateVecIndex(
                        signed_offsets, static_cast<int16_t>(0), tile);
                    AscendC::Muls(
                        signed_offsets, signed_offsets,
                        static_cast<int16_t>(tile * sizeof(half)), tile);
                    AscendC::Gather(cumulative, cumulative_rows,
                                    offsets_local, 0, tile);
                    AscendC::Muls(beta_values, beta_values,
                                  static_cast<half>(-1.0f), tile);
                    AscendC::Exp(beta_values, beta_values, tile);
                    AscendC::Adds(beta_values, beta_values,
                                  static_cast<half>(1.0f), tile);
                    AscendC::Reciprocal(beta_values, beta_values, tile);
                    AscendC::Adds(scale_local, beta_values,
                                  static_cast<half>(0.0f), tile);
                    AscendC::Adds(scale_local[tile], cumulative,
                                  static_cast<half>(0.0f), tile);
                    if (diagnostic_stage == 1) {
                        AscendC::DataCopy(
                            scales_[task * scale_elements], scale_local,
                            scale_elements);
                        continue;
                    }
                    CopyPackedTk(packed_keys,
                                 chunk_data[tile * side], 1.0f, tk_offsets);
                    AscendC::Exp(row_scale, cumulative, tile);
                    CopyPackedTkRows(packed_q, chunk_data[0], row_scale,
                                     broadcast_local, tk_offsets,
                                     tk_scale_offsets);
                    // 真实 GDN 的 H/Q 都很小，9010 Cube 会把大量 FP16
                    // subnormal 乘积冲零。Q_b 用 2^4 精确放大；qk Gram
                    // 随之同比放大，scan 在两项累加后统一还原。
                    AscendC::Muls(packed_q, packed_q,
                                  static_cast<half>(16.0f), short_elements);
                    // 第六个输出改为未衰减的 query*16；与 keys 做 Gram 后，
                    // causal-mask AIV 用 log_b 差稳定地补上 b_i/b_j。
                    CopyPackedTk(packed_k, chunk_data[0], 16.0f,
                                 tk_offsets);
                    if (diagnostic_stage == 2) {
                        AscendC::DataCopy(keys_[short_base], packed_keys,
                                          short_elements);
                        AscendC::DataCopy(
                            scales_[task * scale_elements], scale_local,
                            scale_elements);
                        AscendC::DataCopy(q_b_[short_base], packed_q,
                                          short_elements);
                        AscendC::DataCopy(k_inv_b_[short_base], packed_k,
                                          short_elements);
                        continue;
                    }

                    AscendC::Exp(row_scale, cumulative, tile);
                    AscendC::Mul(row_scale, row_scale, beta_values, tile);
                    AscendC::Muls(row_scale, row_scale,
                                  static_cast<half>(-1.0f), tile);
                    CopyPackedB(packed_negative,
                                chunk_data[tile * side], row_scale,
                                broadcast_local, b_offsets, b_scale_offsets);
                    CopyPackedB(
                        packed_cumsum,
                        chunk_data[(value_group * tile) * side],
                        beta_values, broadcast_local, b_offsets,
                        b_scale_offsets);
                    AscendC::Duplicate(
                        offsets_local,
                        static_cast<uint16_t>((tile - 1) * sizeof(half)),
                        tile);
                    AscendC::Gather(broadcast_local, cumulative,
                                    offsets_local, 0, tile);
                    AscendC::Sub(row_scale, broadcast_local, cumulative,
                                 tile);
                    AscendC::Exp(row_scale, row_scale, tile);
                    CopyPackedB(packed_r,
                                chunk_data[tile * side], row_scale,
                                broadcast_local, b_offsets, b_scale_offsets);
                    if (diagnostic_stage == 3) {
                        AscendC::DataCopy(keys_[short_base], packed_keys,
                                          short_elements);
                        AscendC::DataCopy(
                            scales_[task * scale_elements], scale_local,
                            scale_elements);
                        AscendC::DataCopy(negative_vectors_[short_base],
                                          packed_negative, short_elements);
                        AscendC::DataCopy(cumsum_vectors_[short_base],
                                          packed_cumsum, short_elements);
                        AscendC::DataCopy(q_b_[short_base], packed_q,
                                          short_elements);
                        AscendC::DataCopy(k_inv_b_[short_base], packed_k,
                                          short_elements);
                        AscendC::DataCopy(r_transposed_[short_base], packed_r,
                                          short_elements);
                        continue;
                    }

                    AscendC::Duplicate(
                        offsets_local,
                        static_cast<uint16_t>((tile - 1) * sizeof(half)),
                        tile);
                    AscendC::Gather(broadcast_local, cumulative,
                                    offsets_local, 0, tile);
                    AscendC::Exp(broadcast_local, broadcast_local, tile);
                    for (uint32_t row = 0; row < tile; ++row) {
                        AscendC::Mul(transpose_local[row * tile],
                                     identity_local[row * tile],
                                     broadcast_local, tile);
                    }

                    // 最后一个 packed_r tile 和 b_identity 紧邻 GM
                    // 写回，显式等待 Vector 完成。
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                        vector_to_store);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                        vector_to_store);

                    AscendC::DataCopy(keys_[short_base], packed_keys,
                                      short_elements);
                    AscendC::DataCopy(
                        scales_[task * scale_elements], scale_local,
                        scale_elements);
                    AscendC::DataCopy(negative_vectors_[short_base],
                                      packed_negative, short_elements);
                    AscendC::DataCopy(cumsum_vectors_[short_base],
                                      packed_cumsum, short_elements);
                    AscendC::DataCopy(q_b_[short_base], packed_q,
                                      short_elements);
                    AscendC::DataCopy(k_inv_b_[short_base], packed_k,
                                      short_elements);
                    AscendC::DataCopy(r_transposed_[short_base], packed_r,
                                      short_elements);
                    AscendC::DataCopy(
                        b_identity_[task * tile_elements], transpose_local,
                        tile_elements);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                        store_to_vector);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                        store_to_vector);
                }
            }

        }
    }

private:
    static constexpr uint32_t rows = 1024;
    static constexpr uint32_t key_heads = 16;
    static constexpr uint32_t value_heads = 32;
    static constexpr uint32_t side = 128;
    static constexpr uint32_t key_dim = key_heads * side;
    static constexpr uint32_t value_dim = value_heads * side;
    static constexpr uint32_t conv_dim = key_dim * 2 + value_dim;
    static constexpr uint32_t tile = 16;
    static constexpr uint32_t chunks = rows / tile;
    static constexpr uint32_t tasks = value_heads * chunks;
    static constexpr uint32_t groups = 4;
    static constexpr uint32_t control_vectors = 6;
    static constexpr uint32_t packing_tables = 4;
    static constexpr uint32_t tile_elements = tile * tile;
    static constexpr uint32_t short_elements = tile * side;
    static constexpr uint32_t scale_elements = tile * 2;
    static constexpr float query_scale = 0.08838834764831845f;
    static constexpr uint32_t diagnostic_stage = 0;

    AscendC::GlobalTensor<half> mixed_, alpha_, beta_;
    AscendC::GlobalTensor<half> a_log_, dt_bias_, identity_;
    AscendC::GlobalTensor<half> keys_, scales_, negative_vectors_;
    AscendC::GlobalTensor<half> cumsum_vectors_, q_b_, k_inv_b_;
    AscendC::GlobalTensor<half> r_transposed_, b_identity_;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR mixed, GM_ADDR alpha, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR identity,
    GM_ADDR keys, GM_ADDR scales, GM_ADDR negative_vectors,
    GM_ADDR cumsum_vectors, GM_ADDR q_b, GM_ADDR k_inv_b,
    GM_ADDR r_transposed, GM_ADDR b_identity,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KernelGdnChunkPrepare op;
    op.Init(mixed, alpha, beta, a_log, dt_bias, identity,
            keys, scales, negative_vectors, cumsum_vectors, q_b, k_inv_b,
            r_transposed, b_identity);
    op.Process();
}
