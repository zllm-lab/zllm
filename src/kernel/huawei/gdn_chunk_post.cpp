#include "kernel_operator.h"

// chunk scan 输出为每个 value head 的 [chunk, 128, 16] Cube tile 布局。
// 本 kernel 在 NPU 上直接还原成 [token, 4096]，完成逐 head RMSNorm 和
// SiLU(z) 门控；scan 结果不经过 host，也不产生 CPU 重排。
class KernelGdnChunkPost {
public:
    __aicore__ inline KernelGdnChunkPost() {}

    __aicore__ inline void Init(
        GM_ADDR packed, GM_ADDR z, GM_ADDR norm, GM_ADDR packed_state,
        GM_ADDR output, GM_ADDR output_state,
        uint32_t batch, uint32_t heads, uint32_t head_dim,
        uint32_t chunk_tokens, float eps, float reciprocal)
    {
        batch_ = batch;
        heads_ = heads;
        head_dim_ = head_dim;
        chunk_tokens_ = chunk_tokens;
        chunks_ = batch / chunk_tokens;
        eps_ = eps;
        reciprocal_ = reciprocal;
        packed_.SetGlobalBuffer(
            (__gm__ half*)packed, heads * chunks_ * head_dim * chunk_tokens);
        z_.SetGlobalBuffer((__gm__ half*)z, batch * heads * head_dim);
        norm_.SetGlobalBuffer((__gm__ half*)norm, head_dim);
        packed_state_.SetGlobalBuffer((__gm__ half*)packed_state,
                                      heads * head_dim * head_dim);
        output_.SetGlobalBuffer((__gm__ half*)output,
                                batch * heads * head_dim);
        output_state_.SetGlobalBuffer((__gm__ half*)output_state,
                                      heads * head_dim * head_dim);
    }

    __aicore__ inline void Process()
    {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tile_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> square_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> sum_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_work_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> reciprocal_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> z_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> gate_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> norm_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> offset_buffer;
        pipe.InitBuffer(tile_buffer, tile_elements * sizeof(half));
        pipe.InitBuffer(output_buffer, max_head_dim * sizeof(half));
        pipe.InitBuffer(square_buffer, max_head_dim * sizeof(half));
        pipe.InitBuffer(sum_buffer, 32);
        pipe.InitBuffer(reduce_work_buffer, 32);
        pipe.InitBuffer(reciprocal_buffer, max_head_dim * sizeof(half));
        pipe.InitBuffer(z_buffer, max_head_dim * sizeof(half));
        pipe.InitBuffer(gate_buffer, max_head_dim * sizeof(half));
        pipe.InitBuffer(norm_buffer, max_head_dim * sizeof(half));
        pipe.InitBuffer(offset_buffer, max_head_dim * sizeof(uint16_t));

        auto tile_local = tile_buffer.Get<half>();
        auto output_local = output_buffer.Get<half>();
        auto square_local = square_buffer.Get<half>();
        auto sum_local = sum_buffer.Get<half>();
        auto reduce_work = reduce_work_buffer.Get<half>();
        auto reciprocal_local = reciprocal_buffer.Get<half>();
        auto z_local = z_buffer.Get<half>();
        auto gate_local = gate_buffer.Get<half>();
        auto norm_local = norm_buffer.Get<half>();
        auto offsets = offset_buffer.Get<uint16_t>();
        auto signed_offsets = offsets.ReinterpretCast<int16_t>();
        AscendC::DataCopy(norm_local, norm_, head_dim_);
        AscendC::PipeBarrier<PIPE_ALL>();

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t head_tiles = head_dim_ / cube_tile;
        const uint32_t task_elements = head_dim_ * chunk_tokens_;
        const uint32_t row_stride = heads_ * head_dim_;
        for (uint32_t head = core; head < heads_; head += cores) {
            for (uint32_t chunk = 0; chunk < chunks_; ++chunk) {
                const uint32_t task = head * chunks_ + chunk;
                const uint32_t packed_base = task * task_elements;
                for (uint32_t token_in_chunk = 0;
                     token_in_chunk < chunk_tokens_; ++token_in_chunk) {
                    AscendC::CreateVecIndex(
                        signed_offsets, static_cast<int16_t>(0), cube_tile);
                    AscendC::Muls(
                        signed_offsets, signed_offsets,
                        static_cast<int16_t>(chunk_tokens_ * sizeof(half)),
                        cube_tile);
                    AscendC::Adds(
                        signed_offsets, signed_offsets,
                        static_cast<int16_t>(token_in_chunk * sizeof(half)),
                        cube_tile);
                    for (uint32_t head_tile = 0; head_tile < head_tiles;
                         ++head_tile) {
                        AscendC::DataCopy(
                            tile_local,
                            packed_[packed_base +
                                    head_tile * tile_elements],
                            tile_elements);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::Gather(
                            output_local[head_tile * cube_tile], tile_local,
                            offsets, 0, cube_tile);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Mul(square_local, output_local, output_local,
                                 head_dim_);
                    AscendC::ReduceSum(sum_local, square_local, reduce_work,
                                       head_dim_);
                    AscendC::Muls(
                        sum_local, sum_local,
                        static_cast<half>(reciprocal_), 1);
                    AscendC::Adds(sum_local, sum_local,
                                  static_cast<half>(eps_), 1);
                    AscendC::Rsqrt(sum_local, sum_local, 1);
                    AscendC::Duplicate(offsets, static_cast<uint16_t>(0),
                                       head_dim_);
                    AscendC::Gather(reciprocal_local, sum_local, offsets, 0,
                                    head_dim_);
                    AscendC::Mul(output_local, output_local,
                                 reciprocal_local, head_dim_);
                    AscendC::Mul(output_local, output_local, norm_local,
                                 head_dim_);

                    const uint32_t token = chunk * chunk_tokens_ +
                                           token_in_chunk;
                    const uint32_t output_base =
                        token * row_stride + head * head_dim_;
                    AscendC::DataCopy(z_local, z_[output_base], head_dim_);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Muls(gate_local, z_local,
                                  static_cast<half>(-1.0f), head_dim_);
                    AscendC::Exp(gate_local, gate_local, head_dim_);
                    AscendC::Adds(gate_local, gate_local,
                                  static_cast<half>(1.0f), head_dim_);
                    AscendC::Div(gate_local, z_local, gate_local,
                                 head_dim_);
                    AscendC::Mul(output_local, output_local, gate_local,
                                 head_dim_);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopy(output_[output_base], output_local,
                                      head_dim_);
                }
            }

            // scan state 是 [valueTile,keyTile,valueInner,keyInner]；B1
            // decode 需要连续 [value,key]。AIV 每次搬一段连续 keyInner，
            // 组合完整 row 后直接写 resident output_state。
            const uint32_t state_head_base = head * head_dim_ * head_dim_;
            for (uint32_t value = 0; value < head_dim_; ++value) {
                const uint32_t value_tile = value / cube_tile;
                const uint32_t value_inner = value % cube_tile;
                for (uint32_t key_tile = 0; key_tile < head_tiles;
                     ++key_tile) {
                    const uint32_t source =
                        state_head_base +
                        (value_tile * head_tiles + key_tile) *
                            tile_elements +
                        value_inner * cube_tile;
                    AscendC::DataCopy(
                        output_local[key_tile * cube_tile],
                        packed_state_[source], cube_tile);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(
                    output_state_[state_head_base + value * head_dim_],
                    output_local, head_dim_);
            }
        }
    }

private:
    static constexpr uint32_t cube_tile = 16;
    static constexpr uint32_t tile_elements = cube_tile * cube_tile;
    static constexpr uint32_t max_head_dim = 128;
    AscendC::GlobalTensor<half> packed_, z_, norm_, packed_state_;
    AscendC::GlobalTensor<half> output_, output_state_;
    uint32_t batch_ = 0;
    uint32_t heads_ = 0;
    uint32_t head_dim_ = 0;
    uint32_t chunk_tokens_ = 0;
    uint32_t chunks_ = 0;
    float eps_ = 0.0f;
    float reciprocal_ = 0.0f;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR packed, GM_ADDR z, GM_ADDR norm, GM_ADDR packed_state,
    GM_ADDR output, GM_ADDR output_state,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    (void)workspace;
    KernelGdnChunkPost op;
    op.Init(packed, z, norm, packed_state, output, output_state,
            tiling_data.batch,
            tiling_data.heads, tiling_data.head_dim,
            tiling_data.chunk_tokens, tiling_data.eps,
            tiling_data.reciprocal);
    op.Process();
}
