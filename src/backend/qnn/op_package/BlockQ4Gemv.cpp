// block-32 Q4 GEMV 的 HTP 自定义算子；标量版先用于接口与数值门禁。
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_BlockQ4Gemv);

#if defined(__hexagon__)
static inline void l2fetch_contiguous(const void *address, size_t bytes) {
    auto *next = static_cast<const uint8_t *>(address);
    while (bytes) {
        const uint16_t width = std::min<size_t>(bytes, 32768);
        const uint64_t control = (static_cast<uint64_t>(width) << 32) |
                                 (static_cast<uint64_t>(width) << 16) | 1;
        __asm__ volatile("l2fetch(%0,%1)" : : "r"(next), "r"(control));
        next += width;
        bytes -= width;
    }
}
#endif

GraphStatus block_q4_gemv(Tensor &output,
                          const Tensor &activation,
                          const Tensor &codes,
                          const Tensor &scales) {
    const size_t rows = activation.dim(2);
    const size_t inner = activation.dim(3);
    const size_t columns = output.dim(3);
    if (inner % 32 != 0 || codes.valid_storage_bytes() * 2 != inner * columns ||
        scales.valid_storage_elements() != inner / 32 * columns) {
        return GraphStatus::ErrorFatal;
    }

    const SIdx origin[4] = {0, 0, 0, 0};
    const auto *x = static_cast<const uint16_t *>(activation.element_ptr(4, origin));
    const auto *packed = static_cast<const uint8_t *>(codes.element_ptr(4, origin));
    const auto *weight_scale = static_cast<const uint8_t *>(scales.element_ptr(4, origin));
    auto *y = static_cast<uint16_t *>(output.element_ptr(4, origin));
    const int32_t x_zero = activation.interface_offset();
    const float x_scale = activation.interface_scale();
    const float weight_scale_step = scales.interface_scale();
    const int32_t y_zero = output.interface_offset();
    const float y_recip = 1.0f / output.interface_scale();

    for (size_t row = 0; row < rows; ++row) {
#if defined(__hexagon__)
        if (inner % 32 == 0 && columns % 32 == 0) {
            auto *scratch = static_cast<uint8_t *>(__builtin_alloca(inner + 255));
            auto *x8 = reinterpret_cast<int8_t *>(
                (reinterpret_cast<uintptr_t>(scratch) + 127) & ~uintptr_t{127});
            for (size_t k = 0; k < inner; ++k) {
                const int32_t centered = static_cast<int32_t>(x[row * inner + k]) - x_zero;
                const int32_t rounded = centered >= 0 ? (centered + 128) / 256
                                                      : -((-centered + 128) / 256);
                x8[k] = static_cast<int8_t>(
                    std::clamp(rounded, int32_t{-128}, int32_t{127}));
            }
            const HVX_Vector nibble_mask = Q6_Vb_vsplat_R(15);
            const HVX_Vector sign_bit = Q6_Vb_vsplat_R(8);
            alignas(128) int32_t lanes[32];
            alignas(128) int32_t sums[32];
            const size_t blocks = inner / 32;
            l2fetch_contiguous(packed, blocks * 512);
            l2fetch_contiguous(weight_scale, blocks * 32);
            for (size_t output_block = 0; output_block < columns / 32; ++output_block) {
                if (output_block + 1 < columns / 32) {
                    l2fetch_contiguous(packed + (output_block + 1) * blocks * 512,
                                       blocks * 512);
                    l2fetch_contiguous(weight_scale + (output_block + 1) * blocks * 32,
                                       blocks * 32);
                }
                std::fill(std::begin(sums), std::end(sums), 0);
                for (size_t block = 0; block < blocks; ++block) {
                    auto products = Q6_V_vzero();
                    int32_t correction = 0;
                    const auto *block_codes = packed +
                        (output_block * blocks + block) * 512;
                    for (size_t group = 0; group < 8; group += 2) {
                        const auto packed_vector = *reinterpret_cast<const HVX_Vector *>(
                            block_codes + group * 64);
                    auto low = Q6_V_vand_VV(packed_vector, nibble_mask);
                    auto high = Q6_Vub_vlsr_VubR(packed_vector, 4);
                        const auto weights0 = Q6_V_vxor_VV(low, sign_bit);
                        const auto weights1 = Q6_V_vxor_VV(high, sign_bit);
                        int32_t activation0, activation1;
                        memcpy(&activation0, x8 + block * 32 + group * 4, sizeof(activation0));
                        memcpy(&activation1, x8 + block * 32 + (group + 1) * 4, sizeof(activation1));
                        products = Q6_Vw_vrmpyacc_VwVubRb(products, weights0, activation0);
                        products = Q6_Vw_vrmpyacc_VwVubRb(products, weights1, activation1);
                        for (size_t local = 0; local < 8; ++local) {
                            correction += 8 * x8[block * 32 + group * 4 + local];
                        }
                    }
                    *reinterpret_cast<HVX_Vector *>(lanes) = products;
                    const auto *block_scales = weight_scale +
                        (output_block * blocks + block) * 32;
                    for (size_t lane = 0; lane < 32; ++lane) {
                        sums[lane] += (lanes[lane] - correction) * block_scales[lane];
                    }
                }
                for (size_t lane = 0; lane < 32; ++lane) {
                    const int32_t quantized = static_cast<int32_t>(std::nearbyint(
                        sums[lane] * (x_scale * 256.0f * weight_scale_step) * y_recip)) + y_zero;
                    y[row * columns + output_block * 32 + lane] = static_cast<uint16_t>(
                        std::clamp(quantized, int32_t{0}, int32_t{65535}));
                }
            }
            continue;
        }
#endif
        for (size_t column = 0; column < columns; ++column) {
            float sum = 0.0f;
            for (size_t k = 0; k < inner; ++k) {
                const size_t output_block = column / 32;
                const size_t block = k / 32;
                const size_t group = (k % 32) / 4;
                const size_t lane = column % 32;
                const size_t local = k % 4;
                const size_t nibble = ((((output_block * (inner / 32) + block) * 8 + group) * 32 + lane) * 4 + local);
                const uint8_t byte = packed[nibble / 2];
                int8_t weight = (nibble & 1) ? (byte >> 4) : (byte & 15);
                if (weight >= 8) weight -= 16;
                sum += (static_cast<int32_t>(x[row * inner + k]) - x_zero) * weight *
                       weight_scale[(output_block * (inner / 32) + block) * 32 + lane] * weight_scale_step;
            }
            const int32_t quantized = static_cast<int32_t>(
                                          std::nearbyint(sum * x_scale * y_recip)) +
                                      y_zero;
            y[row * columns + column] = static_cast<uint16_t>(
                std::clamp(quantized, int32_t{0}, int32_t{65535}));
        }
    }
    return GraphStatus::Success;
}

DEF_TENSOR_PROPERTIES(Op("BlockQ4Gemv", "activation", "codes", "scales"),
                      Flat("*", "activation", "codes", "scales"),
                      MainMemory("*", "activation", "codes", "scales"))
DEF_PACKAGE_OP_AND_COST_AND_FLAGS(block_q4_gemv, "BlockQ4Gemv", FAST, Flags::RESOURCE_HVX)

END_PKG_OP_DEFINITION(PKG_BlockQ4Gemv);
