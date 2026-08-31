//! Metal shader 前导:`#include`、function constants,以及被多个功能模块
//! 共用的 inline helper。
//!
//! Metal 运行时把所有 `SHADERS` 拼成单一翻译单元,因此共用 helper 只能在此
//! 定义一次;各功能文件的 `SHADERS` 只放它自己的 kernel 与文件私有 helper。
//!
// shared helpers: finite_f16, decode_f8_e4m3, gated_activation_value, zllm_bf16_to_f32, zllm_f32_to_bf16, w4a16_scale
pub const SHADERS: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

constant uint zllm_fc_u32_0 [[function_constant(0)]];
constant uint zllm_fc_u32_1 [[function_constant(1)]];
constant uint zllm_fc_u32_2 [[function_constant(2)]];
constant uint zllm_fc_u32_3 [[function_constant(3)]];
constant uint zllm_fc_u32_4 [[function_constant(4)]];
constant uint zllm_fc_u32_5 [[function_constant(5)]];
constant uint zllm_fc_u32_6 [[function_constant(6)]];
constant uint zllm_fc_u32_7 [[function_constant(7)]];

inline half finite_f16(float value) {
    return half(clamp(value, -65504.0f, 65504.0f));
}
inline float decode_f8_e4m3(uchar bits) {
    uint exponent = (bits >> 3) & 0x0f;
    uint mantissa = bits & 0x07;
    if (exponent == 0) {
        const float sign = (bits & 0x80) == 0 ? 1.0f : -1.0f;
        return sign * float(mantissa) * 0.001953125f;
    }
    if (exponent == 0x0f && mantissa == 0x07) return NAN;
    const uint f32_bits = (uint(bits & 0x80) << 24)
        | ((exponent + 120) << 23)
        | (mantissa << 20);
    return as_type<float>(f32_bits);
}
inline float gated_activation_value(float gate, float up, uint activation_kind, float alpha, float limit);
inline float gated_activation_value(
    float gate,
    float up,
    uint activation_kind,
    float alpha,
    float limit)
{
    if (activation_kind == 4) {
        gate = min(gate, limit);
        up = clamp(up, -limit, limit);
        return gate / (1.0f + exp(-gate)) * up;
    }
    if (activation_kind == 3) {
        const float situ = alpha * tanh(gate / alpha) / (1.0f + exp(-gate));
        if (limit > 0.0f) up = limit * tanh(up / limit);
        return situ * up;
    }
    if (activation_kind == 2) {
        if (gate >= 10.0f) return gate * up;
        if (gate <= -10.0f) return 0.0f;
        const float x = 0.7978845608f * (gate + 0.044715f * gate * gate * gate);
        return 0.5f * gate * (1.0f + tanh(x)) * up;
    }
    if (activation_kind == 1) {
        gate = min(gate, limit);
        up = clamp(up, -limit, limit);
        return gate / (1.0f + exp(-alpha * gate)) * (up + 1.0f);
    }
    return gate / (1.0f + exp(-gate)) * up;
}
inline float zllm_bf16_to_f32(ushort value) {
    return as_type<float>(uint(value) << 16);
}
inline ushort zllm_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    const uint bias = 0x7fffu + ((bits >> 16) & 1u);
    return ushort((bits + bias) >> 16);
}
inline float w4a16_scale(device const uchar *scales, ulong index, uint scale_dtype)
{
    if (scale_dtype == 0) {
        const ushort bits = reinterpret_cast<device const ushort *>(scales)[index];
        return as_type<float>(uint(bits) << 16);
    }
    if (scale_dtype == 1) {
        return float(reinterpret_cast<device const half *>(scales)[index]);
    }
    return reinterpret_cast<device const float *>(scales)[index];
}
"#;
