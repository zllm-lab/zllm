    unsigned long long gate;
    unsigned long long up;
    unsigned long long down;
    unsigned int gate_type;
    unsigned int up_type;
    unsigned int down_type;
};

union GgufWmmaScratch {
    unsigned short lut[8][16 * 32];
    float output[8][16 * 16];
};

__device__ __forceinline__ float gguf_kq_f16(const unsigned char* bytes)
{
    const unsigned int bits = unsigned(bytes[0]) | (unsigned(bytes[1]) << 8);
    const unsigned int sign = (bits & 0x8000) << 16;
    unsigned int exponent = (bits >> 10) & 0x1f;
    unsigned int mantissa = bits & 0x03ff;
    unsigned int value;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            exponent = 113;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            value = sign | (exponent << 23) | ((mantissa & 0x03ff) << 13);
        }
    } else if (exponent == 31) {
        value = sign | 0x7f800000 | (mantissa << 13);
    } else {
        value = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    union { unsigned int bits; float number; } converted;
    converted.bits = value;
    return converted.number;
}

__device__ __forceinline__ unsigned int gguf_kq_block_bytes(const unsigned int tensor_type)
{
    return zllm_gguf_block_bytes(tensor_type);
}

__device__ __forceinline__ const unsigned char* gguf_kq_row(
    const unsigned long long weight, const unsigned int tensor_type,
    const unsigned int row, const unsigned int columns)
{
    return (const unsigned char*)weight
        + (unsigned long long)row * (columns >> 8) * gguf_kq_block_bytes(tensor_type);
}

// 行基址已知时的块内解码(fused kernel 逐行循环复用基址)。
__device__ __forceinline__ float gguf_kq_value_at(
    const unsigned char* row_base, const unsigned int tensor_type,
    const unsigned int column, const unsigned int block_bytes)
{
    const unsigned char* block = row_base + (unsigned long long)(column >> 8) * block_bytes;
    const unsigned int local = column & 255;
    if (tensor_type == 8u) {
        const unsigned char* q8 = block + (local >> 5) * 34u;
        return gguf_kq_f16(q8) * (float)(signed char)q8[2u + (local & 31u)];
    }
    if (tensor_type == 11u || tensor_type == 21u || tensor_type == 23u)
        return zllm_gguf_extra_value(row_base, tensor_type, column);
    if (tensor_type == 14u) {
        const unsigned int half = local >> 7;
        const unsigned int slot = (local & 127) >> 5;
        const unsigned int index = local & 31;
        const unsigned char low = block[half * 64 + index + (slot & 1u) * 32];
        const unsigned int nibble = (slot < 2u) ? (low & 0x0fu) : (low >> 4);
        const unsigned int high = (block[128 + half * 32 + index] >> (slot * 2)) & 3u;
        const int quant = (int)(nibble | (high << 4)) - 32;
        const signed char group_scale = (signed char)block[192 + half * 8 + (index >> 4) + slot * 2];
        return gguf_kq_f16(block + 208) * (float)group_scale * (float)quant;
    }
    const unsigned int group = local >> 5;
    const unsigned char* scales = block + 4;
    unsigned int scale;
    unsigned int minimum;
    if (group < 4u) {
        scale = scales[group] & 0x3f;
        minimum = scales[group + 4] & 0x3f;
    } else {
        scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
    const unsigned char packed = block[(tensor_type == 12u ? 16u : 48u) + (group >> 1) * 32 + (local & 31)];
    unsigned int quant = (group & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
    if (tensor_type == 13u && (block[16 + (local & 31)] & (1u << group)) != 0u) quant += 16u;
    return gguf_kq_f16(block) * (float)(scale * quant) - gguf_kq_f16(block + 2) * (float)minimum;
}

// 半个 wave 协作读取同一行连续 16 个 K，量化码全局读取保持连续；每组只由
// lane 0 解析一次 scale，再在 16-lane 子组广播。相比 fragment 原生的
// “16 行各读 8 列”布局，这里以少量 LDS 换掉跨大步长的离散读。
__device__ __forceinline__ unsigned short gguf_kq_segment16_bf16(
    const unsigned char* row_base,
    const unsigned int tensor_type,
    const unsigned int column,
    const unsigned int block_bytes,
    const unsigned int lane16)
{
    const unsigned char* block = row_base
        + (unsigned long long)(column >> 8) * block_bytes;
    const unsigned int local = column & 255u;
    if (tensor_type == 8u) {
        const unsigned char* q8 = block + (local >> 5) * 34u;
        float d = lane16 == 0u ? gguf_kq_f16(q8) : 0.0f;
        d = __shfl(d, 0, 16);
        return zllm_f32_to_bf16(
            d * (float)(signed char)q8[2u + (local & 31u)]);
    }
    if (tensor_type == 23u) {
        // IQ4_XS 每 32 个权重共享 d×6-bit scale；连续 16 列只解析一次，
        // 各 lane 直接读取连续 nibble 并查常量码本。
        const unsigned int group = local >> 5;
        float d = 0.0f;
        if (lane16 == 0u) {
            const unsigned int scales_h = (unsigned int)block[2]
                | ((unsigned int)block[3] << 8u);
            const unsigned int scales_l = block[4u + (group >> 1u)];
            const unsigned int scale
                = ((scales_l >> (4u * (group & 1u))) & 15u)
                | (((scales_h >> (2u * group)) & 3u) << 4u);
            d = gguf_kq_f16(block) * ((float)scale - 32.0f);
        }
        d = __shfl(d, 0, 16);
        const unsigned int within = local & 31u;
        const unsigned char packed = block[
            8u + group * 16u + (within & 15u)];
        const unsigned int code = within < 16u
            ? packed & 15u : packed >> 4u;
        return zllm_f32_to_bf16(d * (float)zllm_iq4nl[code]);
    }
    if (tensor_type == 21u) {
        // IQ3_S 同样以 32 列共享 scale；码本索引/符号仍由各 lane 独立
        // 解析，但 f16 与 scale 不再为每个权重重复加载和转换。
        const unsigned int group = local >> 5;
        float d = 0.0f;
        if (lane16 == 0u) {
            const unsigned int scale_byte = block[106u + (group >> 1u)];
            const unsigned int scale = (group & 1u) != 0u
                ? scale_byte >> 4u : scale_byte & 15u;
            d = gguf_kq_f16(block) * (1.0f + 2.0f * (float)scale);
        }
        d = __shfl(d, 0, 16);
        const unsigned int within = local & 31u;
        const unsigned int tuple = within >> 3u;
        const unsigned int tuple_lane = within & 7u;
        const unsigned int qh = block[66u + group];
        const unsigned int q = block[
            2u + group * 8u + 2u * tuple + (tuple_lane >= 4u)];
        const unsigned int shift = tuple_lane < 4u
            ? 8u - 2u * tuple : 7u - 2u * tuple;
        const unsigned int grid_index = q | ((qh << shift) & 256u);
        const unsigned int magnitude = (zllm_iq3s_grid[grid_index]
            >> (8u * (tuple_lane & 3u))) & 255u;
        const float sign = (block[74u + group * 4u + tuple]
            & (1u << tuple_lane)) != 0u ? -1.0f : 1.0f;
        return zllm_f32_to_bf16(d * (float)magnitude * sign);
    }
    if (tensor_type == 14u) {
        const unsigned int half = local >> 7;
        const unsigned int slot = (local & 127u) >> 5;
        const unsigned int index = local & 31u;
        float d = lane16 == 0u
            ? gguf_kq_f16(block + 208u) * (float)(signed char)block[
                192u + half * 8u + (index >> 4) + slot * 2u]
            : 0.0f;
        d = __shfl(d, 0, 16);
        const unsigned char low = block[
            half * 64u + index + (slot & 1u) * 32u];
        const unsigned int nibble = slot < 2u ? (low & 15u) : (low >> 4);
        const unsigned int high = (block[128u + half * 32u + index]
            >> (slot * 2u)) & 3u;
        return zllm_f32_to_bf16(
            d * (float)((int)(nibble | (high << 4u)) - 32));
    }
    if (tensor_type == 12u || tensor_type == 13u) {
        const unsigned int group = local >> 5;
        const unsigned char* scales = block + 4u;
        float d = 0.0f;
        float dmin = 0.0f;
        if (lane16 == 0u) {
            unsigned int scale;
            unsigned int minimum;
            if (group < 4u) {
                scale = scales[group] & 63u;
                minimum = scales[group + 4u] & 63u;
            } else {
                scale = (scales[group + 4u] & 15u)
                    | ((scales[group - 4u] >> 6u) << 4u);
                minimum = (scales[group + 4u] >> 4u)
                    | ((scales[group] >> 6u) << 4u);
            }
            d = gguf_kq_f16(block) * (float)scale;
            dmin = gguf_kq_f16(block + 2u) * (float)minimum;
        }
        d = __shfl(d, 0, 16);
        dmin = __shfl(dmin, 0, 16);
        const unsigned char packed = block[
            (tensor_type == 12u ? 16u : 48u)
            + (group >> 1u) * 32u + (local & 31u)];
        unsigned int quant = (group & 1u) == 0u
            ? (packed & 15u) : (packed >> 4u);
        if (tensor_type == 13u
            && (block[16u + (local & 31u)] & (1u << group)) != 0u)
            quant += 16u;
        return zllm_f32_to_bf16(d * (float)quant - dmin);
    }
    return zllm_f32_to_bf16(gguf_kq_value_at(
        row_base, tensor_type, column, block_bytes));
}

// IQ3_S 的一个 grid 编码 4 个连续权重。leader 一次读取 packed grid 与
// 符号字节并直接展开到 LDS，避免同一份元数据被 4 个 lane 重复读取。
__device__ __forceinline__ void gguf_iq3s_store4_bf16(
    unsigned short* output,
    const unsigned char* row_base,
    const unsigned int column)
{
    const unsigned char* block = row_base
        + (unsigned long long)(column >> 8u) * 110u;
    const unsigned int local = column & 255u;
    const unsigned int group = local >> 5u;
    const unsigned int within = local & 31u;
    const unsigned int tuple = within >> 3u;
    const unsigned int tuple_lane = within & 7u;
    const unsigned int scale_byte = block[106u + (group >> 1u)];
    const unsigned int scale = (group & 1u) != 0u
        ? scale_byte >> 4u : scale_byte & 15u;
    const float d = gguf_kq_f16(block) * (1.0f + 2.0f * (float)scale);
    const unsigned int qh = block[66u + group];
    const unsigned int q = block[
        2u + group * 8u + 2u * tuple + (tuple_lane >= 4u)];
    const unsigned int shift = tuple_lane < 4u
        ? 8u - 2u * tuple : 7u - 2u * tuple;
    const unsigned int packed_grid
        = zllm_iq3s_grid[q | ((qh << shift) & 256u)];
    const unsigned int signs = block[74u + group * 4u + tuple];
    #pragma unroll
    for (unsigned int item = 0u; item < 4u; ++item) {
        const unsigned int magnitude = (packed_grid >> (8u * item)) & 255u;
        const float sign = (signs & (1u << (tuple_lane + item))) != 0u
            ? -1.0f : 1.0f;
        output[item] = zllm_f32_to_bf16(d * (float)magnitude * sign);
    }
}

// BF16 decode 以相邻两个权重为一组。IQ3_S / IQ4_XS 的两个值共享
// block scale，在这里只解析一次；通用格式才回退到两次单值解码。
__device__ __forceinline__ zllm_ct_bf16x2 gguf_kq_pair_bf16(
    const unsigned char* row, unsigned int type, unsigned int column,
    unsigned int block_bytes)
{
    const unsigned char* block = row
        + (unsigned long long)(column >> 8u) * block_bytes;
    const unsigned int local = column & 255u;
    if (type == 23u) {
        const unsigned int group = local >> 5u;
        const unsigned int within = local & 31u;
        const unsigned int scales_h = (unsigned int)block[2]
            | ((unsigned int)block[3] << 8u);
        const unsigned int scales_l = block[4u + (group >> 1u)];
        const unsigned int group_scale
            = ((scales_l >> (4u * (group & 1u))) & 15u)
            | (((scales_h >> (2u * group)) & 3u) << 4u);
        const float scale = gguf_kq_f16(block)
            * ((float)group_scale - 32.0f);
        const unsigned char first = block[
            8u + group * 16u + (within & 15u)];
        const unsigned char second = block[
            8u + group * 16u + ((within + 1u) & 15u)];
        const unsigned int first_code = within < 16u
            ? first & 15u : first >> 4u;
        const unsigned int second_code = within < 16u
            ? second & 15u : second >> 4u;
        return {
            zllm_f32_to_bf16(scale * (float)zllm_iq4nl[first_code]),
            zllm_f32_to_bf16(scale * (float)zllm_iq4nl[second_code])
        };
    }
    if (type == 21u) {
        const unsigned int group = local >> 5u;
        const unsigned int within = local & 31u;
        const unsigned int tuple = within >> 3u;
        const unsigned int tuple_lane = within & 7u;
        const unsigned int scale_byte = block[106u + (group >> 1u)];
        const unsigned int group_scale = (group & 1u) != 0u
            ? scale_byte >> 4u : scale_byte & 15u;
        const float scale = gguf_kq_f16(block)
            * (1.0f + 2.0f * (float)group_scale);
        const unsigned int qh = block[66u + group];
        const unsigned int q = block[
            2u + group * 8u + 2u * tuple + (tuple_lane >= 4u)];
        const unsigned int shift = tuple_lane < 4u
            ? 8u - 2u * tuple : 7u - 2u * tuple;
        const unsigned int grid
            = zllm_iq3s_grid[q | ((qh << shift) & 256u)];
        const unsigned int signs = block[74u + group * 4u + tuple];
        const float first_sign = (signs & (1u << tuple_lane)) != 0u
            ? -1.0f : 1.0f;
        const float second_sign = (signs & (2u << tuple_lane)) != 0u
            ? -1.0f : 1.0f;
        return {
            zllm_f32_to_bf16(
                scale * (float)((grid >> (8u * (tuple_lane & 3u))) & 255u)
                    * first_sign),
            zllm_f32_to_bf16(
                scale * (float)((grid >> (8u * ((tuple_lane + 1u) & 3u))) & 255u)
                    * second_sign)
        };
    }
    return {
        zllm_f32_to_bf16(
            gguf_kq_value_at(row, type, column, block_bytes)),
        zllm_f32_to_bf16(
            gguf_kq_value_at(row, type, column + 1u, block_bytes))
    };
}

// IQ4_XS decode 专用 packed 路径：16-lane 子组的相邻两个 lane 分别消费
// 同一个 32-weight group 的低/高 nibble，每 lane 一次处理连续 16 个 K。
// 与通用 pair 路径保持相同的“scale×code 后过 BF16”边界，只把 block header
// 解析从每 2 个权重一次降为每 16 个权重一次，并把标量 byte load 合成 uint2。
__device__ __forceinline__ float gguf_iq4xs_group_scale(
    const unsigned char* block, const unsigned int group)
{
    const unsigned int scales_h = (unsigned int)block[2]
        | ((unsigned int)block[3] << 8u);
    const unsigned int scales_l = block[4u + (group >> 1u)];
    const unsigned int group_scale
        = ((scales_l >> (4u * (group & 1u))) & 15u)
        | (((scales_h >> (2u * group)) & 3u) << 4u);
    return gguf_kq_f16(block) * ((float)group_scale - 32.0f);
}

__device__ __forceinline__ zllm_ct_bf16x2 gguf_iq4xs_pair_bf16(
    const unsigned int word,
    const unsigned int nibble_shift,
    const unsigned int pair_shift,
    const float scale)
{
    const unsigned int codes = word >> nibble_shift;
    return {
        zllm_f32_to_bf16(
            scale * (float)zllm_iq4nl[(codes >> pair_shift) & 15u]),
        zllm_f32_to_bf16(
            scale * (float)zllm_iq4nl[(codes >> (pair_shift + 8u)) & 15u])
    };
}

__device__ __forceinline__ void gguf_iq4xs_gate_up_word8(
    const unsigned short* input,
    const uint2 gate_words,
    const uint2 up_words,
    const unsigned int nibble_shift,
    const float gate_scale,
    const float up_scale,
    float& gate_sum,
    float& up_sum)
{
    zllm_ct_bf16x2 x01;
    zllm_ct_bf16x2 x23;
    zllm_ct_bf16x2 x45;
    zllm_ct_bf16x2 x67;
    zllm_ct_load_bf16x8(input, x01, x23, x45, x67);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, gguf_iq4xs_pair_bf16(
            gate_words.x, nibble_shift, 0u, gate_scale),
        gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, gguf_iq4xs_pair_bf16(
            up_words.x, nibble_shift, 0u, up_scale),
        up_sum, false);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, gguf_iq4xs_pair_bf16(
            gate_words.x, nibble_shift, 16u, gate_scale),
        gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, gguf_iq4xs_pair_bf16(
            up_words.x, nibble_shift, 16u, up_scale),
        up_sum, false);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, gguf_iq4xs_pair_bf16(
            gate_words.y, nibble_shift, 0u, gate_scale),
        gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, gguf_iq4xs_pair_bf16(
            up_words.y, nibble_shift, 0u, up_scale),
        up_sum, false);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x67, gguf_iq4xs_pair_bf16(
            gate_words.y, nibble_shift, 16u, gate_scale),
        gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x67, gguf_iq4xs_pair_bf16(
            up_words.y, nibble_shift, 16u, up_scale),
        up_sum, false);
}

__device__ __forceinline__ float gguf_iq4xs_word8_dot(
    const unsigned short* input,
    const uint2 words,
    const unsigned int nibble_shift,
    const float scale,
    float sum)
{
    zllm_ct_bf16x2 x01;
    zllm_ct_bf16x2 x23;
    zllm_ct_bf16x2 x45;
    zllm_ct_bf16x2 x67;
    zllm_ct_load_bf16x8(input, x01, x23, x45, x67);
    sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, gguf_iq4xs_pair_bf16(words.x, nibble_shift, 0u, scale),
        sum, false);
    sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, gguf_iq4xs_pair_bf16(words.x, nibble_shift, 16u, scale),
        sum, false);
    sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, gguf_iq4xs_pair_bf16(words.y, nibble_shift, 0u, scale),
        sum, false);
    return __builtin_amdgcn_fdot2_f32_bf16(
        x67, gguf_iq4xs_pair_bf16(words.y, nibble_shift, 16u, scale),
        sum, false);
}

// IQ3_S 每个 grid 编码连续 4 个 magnitude；一次展开两个 grid 即覆盖一个
// 8-weight tuple。packed gate/up 让 scale、qh、sign 和 grid lookup 都按 tuple
// 复用，避免通用 pair 路径为相邻权重重复解析同一份元数据。
__device__ __forceinline__ float gguf_iq3s_group_scale(
    const unsigned char* block, const unsigned int group)
{
    const unsigned int scale_byte = block[106u + (group >> 1u)];
    const unsigned int group_scale = (group & 1u) != 0u
        ? scale_byte >> 4u : scale_byte & 15u;
    return gguf_kq_f16(block) * (1.0f + 2.0f * (float)group_scale);
}

__device__ __forceinline__ zllm_ct_bf16x2 gguf_iq3s_pair_bf16(
    const unsigned int grid,
    const unsigned int magnitude_shift,
    const unsigned int signs,
    const unsigned int sign_shift,
    const float scale)
{
    const float first_sign
        = (signs & (1u << sign_shift)) != 0u ? -1.0f : 1.0f;
    const float second_sign
        = (signs & (2u << sign_shift)) != 0u ? -1.0f : 1.0f;
    return {
        zllm_f32_to_bf16(
            scale * (float)((grid >> magnitude_shift) & 255u) * first_sign),
        zllm_f32_to_bf16(
            scale * (float)((grid >> (magnitude_shift + 8u)) & 255u)
                * second_sign)
    };
}

__device__ __forceinline__ void gguf_iq3s_gate_up_tuple8(
    const unsigned short* input,
    const unsigned char* gate_block,
    const unsigned char* up_block,
    const unsigned int group,
    const unsigned int tuple,
    const float gate_scale,
    const float up_scale,
    float& gate_sum,
    float& up_sum)
{
    const unsigned int gate_qh = gate_block[66u + group];
    const unsigned int up_qh = up_block[66u + group];
    const unsigned int q_offset = 2u + group * 8u + tuple * 2u;
    const unsigned int gate_grid0 = zllm_iq3s_grid[
        (unsigned int)gate_block[q_offset]
            | (((gate_qh >> (tuple * 2u)) & 1u) << 8u)];
    const unsigned int gate_grid1 = zllm_iq3s_grid[
        (unsigned int)gate_block[q_offset + 1u]
            | (((gate_qh >> (tuple * 2u + 1u)) & 1u) << 8u)];
    const unsigned int up_grid0 = zllm_iq3s_grid[
        (unsigned int)up_block[q_offset]
            | (((up_qh >> (tuple * 2u)) & 1u) << 8u)];
    const unsigned int up_grid1 = zllm_iq3s_grid[
        (unsigned int)up_block[q_offset + 1u]
            | (((up_qh >> (tuple * 2u + 1u)) & 1u) << 8u)];
    const unsigned int gate_signs = gate_block[74u + group * 4u + tuple];
    const unsigned int up_signs = up_block[74u + group * 4u + tuple];
    zllm_ct_bf16x2 x01;
    zllm_ct_bf16x2 x23;
    zllm_ct_bf16x2 x45;
    zllm_ct_bf16x2 x67;
    zllm_ct_load_bf16x8(input, x01, x23, x45, x67);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, gguf_iq3s_pair_bf16(
            gate_grid0, 0u, gate_signs, 0u, gate_scale),
        gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, gguf_iq3s_pair_bf16(
            up_grid0, 0u, up_signs, 0u, up_scale),
        up_sum, false);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, gguf_iq3s_pair_bf16(
            gate_grid0, 16u, gate_signs, 2u, gate_scale),
        gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, gguf_iq3s_pair_bf16(
            up_grid0, 16u, up_signs, 2u, up_scale),
        up_sum, false);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, gguf_iq3s_pair_bf16(
            gate_grid1, 0u, gate_signs, 4u, gate_scale),
        gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, gguf_iq3s_pair_bf16(
            up_grid1, 0u, up_signs, 4u, up_scale),
        up_sum, false);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x67, gguf_iq3s_pair_bf16(
            gate_grid1, 16u, gate_signs, 6u, gate_scale),
        gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x67, gguf_iq3s_pair_bf16(
            up_grid1, 16u, up_signs, 6u, up_scale),
        up_sum, false);
}

// 单元素完整反量化(标量路径)。与 codec::ggml 的 decode_q4_k/q5_k/q6_k 逐位一致。
__device__ __forceinline__ float gguf_kq_value(
    const unsigned long long weight, const unsigned int tensor_type,
    const unsigned int row, const unsigned int columns, const unsigned int column)
{
    return gguf_kq_value_at(
        gguf_kq_row(weight, tensor_type, row, columns),
        tensor_type, column, gguf_kq_block_bytes(tensor_type));
}

// tile 内单列 code 提取(WMMA LUT 路径)。
__device__ __forceinline__ unsigned int gguf_kq_code(
    const unsigned long long weight, const unsigned int tensor_type,
    const unsigned int row, const unsigned int columns, const unsigned int column)
{
    const unsigned char* block = gguf_kq_row(weight, tensor_type, row, columns)
        + (unsigned long long)(column >> 8) * gguf_kq_block_bytes(tensor_type);
    const unsigned int local = column & 255;
    if (tensor_type == 14u) {
        const unsigned int half = local >> 7;
        const unsigned int slot = (local & 127) >> 5;
        const unsigned int index = local & 31;
        const unsigned char low = block[half * 64 + index + (slot & 1u) * 32];
        const unsigned int nibble = (slot < 2u) ? (low & 0x0fu) : (low >> 4);
        const unsigned int high = (block[128 + half * 32 + index] >> (slot * 2)) & 3u;
        return nibble | (high << 4);
    }
    const unsigned int group = local >> 5;
    const unsigned char packed = block[(tensor_type == 12u ? 16u : 48u) + (group >> 1) * 32 + (local & 31)];
    const unsigned int nibble = (group & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
    if (tensor_type == 13u && (block[16 + (local & 31)] & (1u << group)) != 0u) return nibble + 16u;
    return nibble;
}

// 为 (行, tile) 构建 LUT：tile 的 16 列共享同一 32 元素组，反量化值域只有 code→bf16。
// 仅服务 gate/up(Q4_K/Q5_K，码本 ≤32)；Q6_K 只出现在 down，走标量 gguf_kq_value。
__device__ __forceinline__ void gguf_kq_fill_lut(
    unsigned short* lut_row,
    const unsigned long long weight, const unsigned int tensor_type,
    const unsigned int row, const unsigned int columns, const unsigned int column_base)
{
    const unsigned char* block = gguf_kq_row(weight, tensor_type, row, columns)
        + (unsigned long long)(column_base >> 8) * gguf_kq_block_bytes(tensor_type);
    const unsigned int local = column_base & 255;
    const unsigned int group = local >> 5;
    const unsigned char* scales = block + 4;
    unsigned int scale;
    unsigned int minimum;
    if (group < 4u) {
        scale = scales[group] & 0x3f;
        minimum = scales[group + 4] & 0x3f;
    } else {
        scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
    const float d = gguf_kq_f16(block);
    const float dmin = gguf_kq_f16(block + 2);
    if (tensor_type == 12u) {
        #pragma unroll
        for (unsigned int code = 0; code < 16; ++code)
            lut_row[code] = zllm_f32_to_bf16(d * (float)(scale * code) - dmin * (float)minimum);
    } else {
        #pragma unroll
        for (unsigned int code = 0; code < 32; ++code) {
            const unsigned int quant = (code & 15u) + 16u * (code >> 4);
            lut_row[code] = zllm_f32_to_bf16(d * (float)(scale * quant) - dmin * (float)minimum);
        }
    }
}

// ===== GGUF fused decode experts(小批次：route_count×top_k ≤ 64) =====
// 对应 CT 的 decode fused 路径：token-major，无路由分组 kernel，activated 常驻 workspace。

// 段一：每个 16-lane 子组计算一个 intermediate 维，一个 block 覆盖
// 16 维。旧路径让整个 block 同时持有 8 维 gate/up 累加并用 64 KiB
// LDS 归约，导致寄存器与 occupancy 双重受限。
extern "C" __global__ void gguf_fused_gate_up_f32(
    const void* input,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    unsigned short* activated,
    const unsigned int assignments,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate,
    const unsigned int input_is_bf16)
{
    const unsigned int assignment = blockIdx.y;
    if (assignment >= assignments) return;
    const unsigned int token = assignment / top_k;
    const GgufKqMeta meta = metas[route_slots[assignment]];
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * 16u + subgroup;
    if (output_row >= intermediate) return;
    const unsigned char* gate_row
        = gguf_kq_row(meta.gate, meta.gate_type, output_row, hidden);
    const unsigned char* up_row
        = gguf_kq_row(meta.up, meta.up_type, output_row, hidden);
    const unsigned int gate_bb = gguf_kq_block_bytes(meta.gate_type);
    const unsigned int up_bb = gguf_kq_block_bytes(meta.up_type);
    const unsigned long long input_base = (unsigned long long)token * hidden;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (input_is_bf16) {
        for (unsigned int column = lane * 2u;
             column + 1u < hidden;
             column += 32u) {
            const zllm_ct_bf16x2 x = {
                ((const unsigned short*)input)[input_base + column],
                ((const unsigned short*)input)[input_base + column + 1u]
            };
            gate_sum += __builtin_amdgcn_fdot2_f32_bf16(
                x, gguf_kq_pair_bf16(
                    gate_row, meta.gate_type, column, gate_bb),
                0.0f, false);
            up_sum += __builtin_amdgcn_fdot2_f32_bf16(
                x, gguf_kq_pair_bf16(
                    up_row, meta.up_type, column, up_bb),
                0.0f, false);
        }
    } else {
        for (unsigned int column = lane; column < hidden; column += 16u) {
            const float x = ((const float*)input)[input_base + column];
            gate_sum += x * gguf_kq_value_at(
                gate_row, meta.gate_type, column, gate_bb);
            up_sum += x * gguf_kq_value_at(
                up_row, meta.up_type, column, up_bb);
        }
    }
    for (unsigned int offset = 8u; offset; offset >>= 1u) {
        gate_sum += __shfl_down(gate_sum, offset, 16);
        up_sum += __shfl_down(up_sum, offset, 16);
    }
    if (lane == 0) {
        const float gate = gate_sum / (1.0f + expf(-gate_sum));
        activated[(unsigned long long)assignment * intermediate + output_row]
            = zllm_f32_to_bf16(gate * up_sum);
    }
}

// 段二：每个 16-lane 子组计算一个 hidden 维，跨 top_k 固定顺序
// 寄存器累加，无原子。
extern "C" __global__ void gguf_fused_down_f32(
    const unsigned short* activated,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    const float* route_weights,
    float* output,
    const unsigned int tokens,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate)
{
    const unsigned int token = blockIdx.y;
    if (token >= tokens) return;
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * 16u + subgroup;
    if (output_row >= hidden) return;
    float sum = 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) {
        const unsigned int assignment = token * top_k + k;
        const float route_weight = route_weights[assignment];
        const GgufKqMeta meta = metas[route_slots[assignment]];
        const unsigned int down_bb = gguf_kq_block_bytes(meta.down_type);
        const unsigned char* down_row = gguf_kq_row(
            meta.down, meta.down_type, output_row, intermediate);
        const unsigned long long input_base
            = (unsigned long long)assignment * intermediate;
        for (unsigned int column = lane * 2u;
             column + 1u < intermediate;
             column += 32u) {
            const zllm_ct_bf16x2 x = {
                activated[input_base + column],
                activated[input_base + column + 1u]
            };
            sum += __builtin_amdgcn_fdot2_f32_bf16(
                x, gguf_kq_pair_bf16(
                    down_row, meta.down_type, column, down_bb),
                0.0f, false) * route_weight;
        }
    }
    for (unsigned int offset = 8u; offset; offset >>= 1u)
        sum += __shfl_down(sum, offset, 16);
    if (lane == 0)
        output[(unsigned long long)token * hidden + output_row] = sum;
}

extern "C" __global__ void gguf_fused_gate_up_iq3s_f32(
    const void* input,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    unsigned short* activated,
    const unsigned int assignments,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate,
    const unsigned int input_is_bf16)
{
    const unsigned int assignment = blockIdx.y;
    if (assignment >= assignments || input_is_bf16 == 0u) return;
    const unsigned int token = assignment / top_k;
    const GgufKqMeta meta = metas[route_slots[assignment]];
    // 一个 lane 消费 IQ3_S 的完整 32-weight group，避免相邻 lane
    // 重复读取同一 scale/qh，并把一个 block 的输出行数翻倍。
    const unsigned int lane = threadIdx.x & 7u;
    const unsigned int subgroup = threadIdx.x >> 3u;
    const unsigned int output_row = blockIdx.x * 32u + subgroup;
    if (output_row >= intermediate) return;
    const unsigned char* gate_row = gguf_kq_row(
        meta.gate, 21u, output_row, hidden);
    const unsigned char* up_row = gguf_kq_row(
        meta.up, 21u, output_row, hidden);
    const unsigned int group = lane;
    const unsigned int column_in_block = group * 32u;
    const unsigned long long input_base = (unsigned long long)token * hidden;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (unsigned int block_index = 0u;
         block_index < hidden / 256u;
         ++block_index) {
        const unsigned char* gate_block
            = gate_row + (unsigned long long)block_index * 110u;
        const unsigned char* up_block
            = up_row + (unsigned long long)block_index * 110u;
        const unsigned short* block_input = (const unsigned short*)input
            + input_base + block_index * 256u + column_in_block;
        const float gate_scale = gguf_iq3s_group_scale(gate_block, group);
        const float up_scale = gguf_iq3s_group_scale(up_block, group);
        gguf_iq3s_gate_up_tuple8(
            block_input, gate_block, up_block, group, 0u,
            gate_scale, up_scale, gate_sum, up_sum);
        gguf_iq3s_gate_up_tuple8(
            block_input + 8u, gate_block, up_block, group, 1u,
            gate_scale, up_scale, gate_sum, up_sum);
        gguf_iq3s_gate_up_tuple8(
            block_input + 16u, gate_block, up_block, group, 2u,
            gate_scale, up_scale, gate_sum, up_sum);
        gguf_iq3s_gate_up_tuple8(
            block_input + 24u, gate_block, up_block, group, 3u,
            gate_scale, up_scale, gate_sum, up_sum);
    }
    for (unsigned int offset = 4u; offset; offset >>= 1u) {
        gate_sum += __shfl_down(gate_sum, offset, 8);
        up_sum += __shfl_down(up_sum, offset, 8);
    }
    if (lane == 0u) {
        const float gate = gate_sum / (1.0f + expf(-gate_sum));
        activated[(unsigned long long)assignment * intermediate + output_row]
            = zllm_f32_to_bf16(gate * up_sum);
    }
}

extern "C" __global__ void gguf_fused_gate_up_iq4xs_f32(
    const void* input,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    unsigned short* activated,
    const unsigned int assignments,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate,
    const unsigned int input_is_bf16)
{
    const unsigned int assignment = blockIdx.y;
    if (assignment >= assignments || input_is_bf16 == 0u) return;
    const unsigned int token = assignment / top_k;
    const GgufKqMeta meta = metas[route_slots[assignment]];
    // 单 lane 消费一个完整 32-column quant group，scale 只解析一次；
    // 8-lane 子组仍保持每个 quant group 的固定归约顺序。
    const unsigned int lane = threadIdx.x & 7u;
    const unsigned int subgroup = threadIdx.x >> 3u;
    const unsigned int output_row = blockIdx.x * 32u + subgroup;
    if (output_row >= intermediate) return;
    const unsigned char* gate_row = gguf_kq_row(
        meta.gate, 23u, output_row, hidden);
    const unsigned char* up_row = gguf_kq_row(
        meta.up, 23u, output_row, hidden);
    const unsigned int group = lane;
    const unsigned int column_in_block = group * 32u;
    const unsigned long long input_base = (unsigned long long)token * hidden;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (unsigned int block_index = 0u;
         block_index < hidden / 256u;
         ++block_index) {
        const unsigned char* gate_block
            = gate_row + (unsigned long long)block_index * 136u;
        const unsigned char* up_block
            = up_row + (unsigned long long)block_index * 136u;
        const unsigned char* gate_codes = gate_block + 8u + group * 16u;
        const unsigned char* up_codes = up_block + 8u + group * 16u;
        const unsigned short* block_input = (const unsigned short*)input
            + input_base + block_index * 256u + column_in_block;
        const float gate_scale = gguf_iq4xs_group_scale(gate_block, group);
        const float up_scale = gguf_iq4xs_group_scale(up_block, group);
        gguf_iq4xs_gate_up_word8(
            block_input,
            *reinterpret_cast<const uint2*>(gate_codes),
            *reinterpret_cast<const uint2*>(up_codes),
            0u, gate_scale, up_scale, gate_sum, up_sum);
        gguf_iq4xs_gate_up_word8(
            block_input + 8u,
            *reinterpret_cast<const uint2*>(gate_codes + 8u),
            *reinterpret_cast<const uint2*>(up_codes + 8u),
            0u, gate_scale, up_scale, gate_sum, up_sum);
        gguf_iq4xs_gate_up_word8(
            block_input + 16u,
            *reinterpret_cast<const uint2*>(gate_codes),
            *reinterpret_cast<const uint2*>(up_codes),
            4u, gate_scale, up_scale, gate_sum, up_sum);
        gguf_iq4xs_gate_up_word8(
            block_input + 24u,
            *reinterpret_cast<const uint2*>(gate_codes + 8u),
            *reinterpret_cast<const uint2*>(up_codes + 8u),
            4u, gate_scale, up_scale, gate_sum, up_sum);
    }
    for (unsigned int offset = 4u; offset; offset >>= 1u) {
        gate_sum += __shfl_down(gate_sum, offset, 8);
        up_sum += __shfl_down(up_sum, offset, 8);
    }
    if (lane == 0u) {
        const float gate = gate_sum / (1.0f + expf(-gate_sum));
        activated[(unsigned long long)assignment * intermediate + output_row]
            = zllm_f32_to_bf16(gate * up_sum);
    }
}

extern "C" __global__ void gguf_fused_down_iq4xs_f32(
    const unsigned short* activated,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    const float* route_weights,
    float* output,
    const unsigned int tokens,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate)
{
    const unsigned int token = blockIdx.y;
    if (token >= tokens) return;
    // 与 gate/up 使用同一 8-lane 映射，避免一组 scale 被相邻两 lane 重复解析。
    const unsigned int lane = threadIdx.x & 7u;
    const unsigned int subgroup = threadIdx.x >> 3u;
    const unsigned int output_row = blockIdx.x * 32u + subgroup;
    if (output_row >= hidden) return;
    const unsigned int group = lane;
    const unsigned int column_in_block = group * 32u;
    float sum = 0.0f;
    for (unsigned int k = 0u; k < top_k; ++k) {
        const unsigned int assignment = token * top_k + k;
        const GgufKqMeta meta = metas[route_slots[assignment]];
        const unsigned char* down_row = gguf_kq_row(
            meta.down, 23u, output_row, intermediate);
        const unsigned short* input = activated
            + (unsigned long long)assignment * intermediate;
        float route_sum = 0.0f;
        for (unsigned int block_index = 0u;
             block_index < intermediate / 256u;
             ++block_index) {
            const unsigned char* block
                = down_row + (unsigned long long)block_index * 136u;
            const unsigned char* codes = block + 8u + group * 16u;
            const unsigned short* block_input
                = input + block_index * 256u + column_in_block;
            const float scale = gguf_iq4xs_group_scale(block, group);
            route_sum = gguf_iq4xs_word8_dot(
                block_input,
                *reinterpret_cast<const uint2*>(codes),
                0u, scale, route_sum);
            route_sum = gguf_iq4xs_word8_dot(
                block_input + 8u,
                *reinterpret_cast<const uint2*>(codes + 8u),
                0u, scale, route_sum);
            route_sum = gguf_iq4xs_word8_dot(
                block_input + 16u,
                *reinterpret_cast<const uint2*>(codes),
                4u, scale, route_sum);
            route_sum = gguf_iq4xs_word8_dot(
                block_input + 24u,
                *reinterpret_cast<const uint2*>(codes + 8u),
                4u, scale, route_sum);
        }
        sum += route_sum * route_weights[assignment];
    }
    for (unsigned int offset = 4u; offset; offset >>= 1u)
        sum += __shfl_down(sum, offset, 8);
    if (lane == 0u)
        output[(unsigned long long)token * hidden + output_row] = sum;
}

__device__ __forceinline__ zllm_ct_bf16x2 gguf_q8_0_pair_bf16(
    const unsigned int word,
    const unsigned int shift,
    const float scale)
{
    return {
        zllm_f32_to_bf16(scale * (float)(signed char)(word >> shift)),
        zllm_f32_to_bf16(scale * (float)(signed char)(word >> (shift + 8u)))
    };
}

__device__ __forceinline__ uint2 gguf_q8_0_load8(
    const unsigned char* codes)
{
    return {
        (unsigned int)codes[0]
            | ((unsigned int)codes[1] << 8u)
            | ((unsigned int)codes[2] << 16u)
            | ((unsigned int)codes[3] << 24u),
        (unsigned int)codes[4]
            | ((unsigned int)codes[5] << 8u)
            | ((unsigned int)codes[6] << 16u)
            | ((unsigned int)codes[7] << 24u)
    };
}

__device__ __forceinline__ float gguf_q8_0_word8_dot(
    const unsigned short* input,
    const uint2 words,
    const float scale,
    float sum)
{
    zllm_ct_bf16x2 x01;
    zllm_ct_bf16x2 x23;
    zllm_ct_bf16x2 x45;
    zllm_ct_bf16x2 x67;
    zllm_ct_load_bf16x8(input, x01, x23, x45, x67);
    sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, gguf_q8_0_pair_bf16(words.x, 0u, scale), sum, false);
    sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, gguf_q8_0_pair_bf16(words.x, 16u, scale), sum, false);
    sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, gguf_q8_0_pair_bf16(words.y, 0u, scale), sum, false);
    return __builtin_amdgcn_fdot2_f32_bf16(
        x67, gguf_q8_0_pair_bf16(words.y, 16u, scale), sum, false);
}

__device__ __forceinline__ void gguf_q8_0_gate_up_word8(
    const unsigned short* input,
    const uint2 gate_words,
    const uint2 up_words,
    const float gate_scale,
    const float up_scale,
    float& gate_sum,
    float& up_sum)
{
    zllm_ct_bf16x2 x01;
    zllm_ct_bf16x2 x23;
    zllm_ct_bf16x2 x45;
    zllm_ct_bf16x2 x67;
    zllm_ct_load_bf16x8(input, x01, x23, x45, x67);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, gguf_q8_0_pair_bf16(gate_words.x, 0u, gate_scale), gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x01, gguf_q8_0_pair_bf16(up_words.x, 0u, up_scale), up_sum, false);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, gguf_q8_0_pair_bf16(gate_words.x, 16u, gate_scale), gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x23, gguf_q8_0_pair_bf16(up_words.x, 16u, up_scale), up_sum, false);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, gguf_q8_0_pair_bf16(gate_words.y, 0u, gate_scale), gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x45, gguf_q8_0_pair_bf16(up_words.y, 0u, up_scale), up_sum, false);
    gate_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x67, gguf_q8_0_pair_bf16(gate_words.y, 16u, gate_scale), gate_sum, false);
    up_sum = __builtin_amdgcn_fdot2_f32_bf16(
        x67, gguf_q8_0_pair_bf16(up_words.y, 16u, up_scale), up_sum, false);
}

extern "C" __global__ void gguf_fused_gate_up_q8_0_f32(
    const void* input,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    unsigned short* activated,
    const unsigned int assignments,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate,
    const unsigned int input_is_bf16)
{
    const unsigned int assignment = blockIdx.y;
    if (assignment >= assignments || input_is_bf16 == 0u) return;
    const unsigned int token = assignment / top_k;
    const GgufKqMeta meta = metas[route_slots[assignment]];
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * 16u + subgroup;
    if (output_row >= intermediate) return;
    const unsigned char* gate_row = gguf_kq_row(
        meta.gate, 8u, output_row, hidden);
    const unsigned char* up_row = gguf_kq_row(
        meta.up, 8u, output_row, hidden);
    const unsigned int group = lane >> 1u;
    const unsigned int half = lane & 1u;
    const unsigned int column_in_block = group * 32u + half * 16u;
    const unsigned long long input_base = (unsigned long long)token * hidden;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (unsigned int block_index = 0u;
         block_index < hidden / 256u;
         ++block_index) {
        const unsigned char* gate_q8 = gate_row
            + (unsigned long long)block_index * 272u + group * 34u;
        const unsigned char* up_q8 = up_row
            + (unsigned long long)block_index * 272u + group * 34u;
        const unsigned short* block_input = (const unsigned short*)input
            + input_base + block_index * 256u + column_in_block;
        const float gate_scale = gguf_kq_f16(gate_q8);
        const float up_scale = gguf_kq_f16(up_q8);
        const unsigned char* gate_codes = gate_q8 + 2u + half * 16u;
        const unsigned char* up_codes = up_q8 + 2u + half * 16u;
        gguf_q8_0_gate_up_word8(
            block_input,
            gguf_q8_0_load8(gate_codes),
            gguf_q8_0_load8(up_codes),
            gate_scale, up_scale, gate_sum, up_sum);
        gguf_q8_0_gate_up_word8(
            block_input + 8u,
            gguf_q8_0_load8(gate_codes + 8u),
            gguf_q8_0_load8(up_codes + 8u),
            gate_scale, up_scale, gate_sum, up_sum);
    }
    for (unsigned int offset = 8u; offset; offset >>= 1u) {
        gate_sum += __shfl_down(gate_sum, offset, 16);
        up_sum += __shfl_down(up_sum, offset, 16);
    }
    if (lane == 0u) {
        const float gate = gate_sum / (1.0f + expf(-gate_sum));
        activated[(unsigned long long)assignment * intermediate + output_row]
            = zllm_f32_to_bf16(gate * up_sum);
    }
}

extern "C" __global__ void gguf_fused_down_q8_0_f32(
    const unsigned short* activated,
    const GgufKqMeta* metas,
    const unsigned int* route_ids,
    const unsigned int* route_slots,
    const float* route_weights,
    float* output,
    const unsigned int tokens,
    const unsigned int top_k,
    const unsigned int hidden,
    const unsigned int intermediate)
{
    const unsigned int token = blockIdx.y;
    if (token >= tokens) return;
    const unsigned int lane = threadIdx.x & 15u;
    const unsigned int subgroup = threadIdx.x >> 4u;
    const unsigned int output_row = blockIdx.x * 16u + subgroup;
    if (output_row >= hidden) return;
    const unsigned int group = lane >> 1u;
    const unsigned int half = lane & 1u;
    const unsigned int column_in_block = group * 32u + half * 16u;
    float sum = 0.0f;
    for (unsigned int k = 0u; k < top_k; ++k) {
        const unsigned int assignment = token * top_k + k;
        const GgufKqMeta meta = metas[route_slots[assignment]];
        const unsigned char* down_row = gguf_kq_row(
            meta.down, 8u, output_row, intermediate);
        const unsigned short* input = activated
            + (unsigned long long)assignment * intermediate;
        float route_sum = 0.0f;
        for (unsigned int block_index = 0u;
             block_index < intermediate / 256u;
             ++block_index) {
            const unsigned char* q8 = down_row
                + (unsigned long long)block_index * 272u + group * 34u;
            const unsigned char* codes = q8 + 2u + half * 16u;
            const unsigned short* block_input = input
                + block_index * 256u + column_in_block;
            const float scale = gguf_kq_f16(q8);
            route_sum = gguf_q8_0_word8_dot(
                block_input,
                gguf_q8_0_load8(codes), scale, route_sum);
            route_sum = gguf_q8_0_word8_dot(
                block_input + 8u,
                gguf_q8_0_load8(codes + 8u), scale, route_sum);
        }
        sum += route_sum * route_weights[assignment];
    }
    for (unsigned int offset = 8u; offset; offset >>= 1u)
        sum += __shfl_down(sum, offset, 16);
    if (lane == 0u)
        output[(unsigned long long)token * hidden + output_row] = sum;
}
extern "C" __global__ void gguf_group_decode_routes(
    const unsigned int* route_ids,
    const float* route_weights,
    const GgufKqMeta* expert_metas,
    unsigned int* grouped_tokens,
    float* grouped_weights,
    unsigned int* route_to_grouped,
    unsigned int* grouped_offsets,
    GgufKqMeta* grouped_metas,
    unsigned int* grouped_experts,
    unsigned int route_count,
    unsigned int top_k,
    unsigned int expert_count,
    unsigned int slot_count)
{
    __shared__ unsigned int counts[256];
    __shared__ unsigned int expert_slots[256];
    __shared__ unsigned int cursors[256];
    const unsigned int lane = threadIdx.x;
    if (lane < 256u) {
        counts[lane] = 0u;
        expert_slots[lane] = 0xffffffffu;
        cursors[lane] = 0u;
    }
    __syncthreads();
    for (unsigned int route = lane; route < route_count; route += blockDim.x) {
        const unsigned int expert = route_ids[route];
        if (expert < expert_count) atomicAdd(&counts[expert], 1u);
    }
    __syncthreads();
    if (lane == 0u) {
        unsigned int active = 0u;
        unsigned int offset = 0u;
        for (unsigned int expert = 0u; expert < expert_count; ++expert) {
            const unsigned int count = counts[expert];
            if (count == 0u) continue;
            expert_slots[expert] = active;
            grouped_offsets[active] = offset;
            grouped_metas[active] = expert_metas[expert];
            cursors[active] = offset;
            offset += count;
            ++active;
        }
        for (unsigned int slot = active; slot < slot_count; ++slot) {
            grouped_offsets[slot] = offset;
            grouped_metas[slot] = expert_metas[0];
            cursors[slot] = offset;
        }
        grouped_offsets[slot_count] = offset;
    }
    __syncthreads();
    for (unsigned int expert = lane; expert < expert_count; expert += blockDim.x) {
        const unsigned int slot = expert_slots[expert];
        if (slot == 0xffffffffu) continue;
        unsigned int grouped = grouped_offsets[slot];
        for (unsigned int route = 0u; route < route_count; ++route) {
            if (route_ids[route] != expert) continue;
            grouped_tokens[grouped] = route / top_k;
            grouped_weights[grouped] = route_weights[route];
            route_to_grouped[route] = grouped;
            grouped_experts[grouped] = slot;
            ++grouped;
        }
    }
}

// gate/up 融合 WMMA：结构对齐 ct_grouped_gate_up_wmma_bf16，仅权重解码换成 GGUF LUT。
extern "C" __global__ void gguf_gate_up_wmma_bf16(
    const unsigned short* input,
    const unsigned int* token_ids,
    const GgufKqMeta* experts,
    const unsigned int* route_offsets,
    unsigned short* activated,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int expert_base,
    unsigned int expert_count)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    static_assert(BFragment::num_elements == 8, "GFX11 GGUF gate/up B fragment lane layout 已改变");
    const unsigned int expert = expert_base + blockIdx.z;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int projection = wave >> 2;
    const unsigned int tile_wave = wave & 3u;
    const unsigned int first_input_base = blockIdx.y * 128;
    const unsigned int output_base = (blockIdx.x * 4 + tile_wave) * 16;
    if (blockIdx.z >= expert_count || wave >= 8u) return;
    const unsigned int route_begin = route_offsets[expert];
    const unsigned int input_rows = route_offsets[expert + 1] - route_begin;
    // 与 CT 不同：不设 input_rows>=4 门槛——不足 16 行由 staging 零填充，
    // 单 token(MTP draft)与稀疏 expert 也必须产出 activated，否则 down 读到未初始化内存。
    if (first_input_base >= input_rows) return;
    const GgufKqMeta& meta = experts[expert];
    const unsigned long long weight = projection == 0u ? meta.gate : meta.up;
    const unsigned int tensor_type = projection == 0u ? meta.gate_type : meta.up_type;
    __shared__ unsigned short input_tiles[16][16 * 16];
    __shared__ unsigned short weight_tiles[8][16 * 16];
    __shared__ GgufWmmaScratch scratch;
    // route 数只有设备侧 compact 后才知道。与 CT grouped kernel 相同，host
    // 只发一个 y tile，由同一个 expert/output tile 顺序覆盖全部 route。
    for (unsigned int input_base = first_input_base;
         input_base < input_rows;
         input_base += gridDim.y * 128u) {
    CFragment accumulator_fragments[8];
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (input_base + tile * 16 < input_rows)
            fill_fragment(accumulator_fragments[tile], 0.0f);
    }
    const unsigned int stage_row = (threadIdx.x >> 2) & 15u;
    const unsigned int stage_chunk = threadIdx.x & 3u;
    const unsigned int stage_group = threadIdx.x >> 6;
    uint2 stage_data[4];
    unsigned int stage_valid[4] = {0u, 0u, 0u, 0u};
    auto stage_prefetch = [&](unsigned int prefetch_base) {
        #pragma unroll
        for (unsigned int half = 0; half < 4; ++half) {
            const unsigned int tile = stage_group + half * 4;
            const unsigned int column_offset = (half >> 1u) * 16u;
            const unsigned int route_row
                = input_base + (tile & 7u) * 16u + stage_row;
            stage_valid[half] = route_row < input_rows ? 1u : 0u;
            const unsigned int token = stage_valid[half]
                ? token_ids[route_begin + route_row] : 0u;
            stage_data[half] = stage_valid[half]
                ? *reinterpret_cast<const uint2*>(
                    input + (unsigned long long)token * input_columns
                    + prefetch_base + column_offset + stage_chunk * 4u)
                : make_uint2(0u, 0u);
        }
    };
    stage_prefetch(0);
    for (unsigned int column_base = 0; column_base < input_columns; column_base += 32u) {
        #pragma unroll
        for (unsigned int half = 0; half < 4; ++half) {
            const unsigned int tile = stage_group + half * 4u;
            if (stage_valid[half]) {
                *reinterpret_cast<uint2*>(
                    &input_tiles[tile][stage_row * 16u + stage_chunk * 4u])
                    = stage_data[half];
            }
        }
        if (column_base + 32u < input_columns)
            stage_prefetch(column_base + 32u);
        __syncthreads();
        #pragma unroll
        for (unsigned int segment = 0; segment < 2u; ++segment) {
            const unsigned int segment_column_base = column_base + segment * 16u;
            if (segment_column_base >= input_columns) break;
            BFragment weight_fragment;
            const unsigned int lane16 = lane & 15u;
            if (tensor_type == 21u) {
                const unsigned int output_row = output_base + lane16;
                if (output_row < output_rows) {
                    const unsigned int column
                        = segment_column_base + (lane >> 4u) * 8u;
                    const unsigned char* block
                        = gguf_kq_row(weight, tensor_type, output_row, input_columns)
                        + (unsigned long long)(column >> 8u) * 110u;
                    const unsigned int local = column & 255u;
                    const unsigned int group = local >> 5u;
                    const unsigned int scale_byte
                        = block[106u + (group >> 1u)];
                    const unsigned int group_scale = (group & 1u) != 0u
                        ? scale_byte >> 4u : scale_byte & 15u;
                    const float scale = gguf_kq_f16(block)
                        * (1.0f + 2.0f * (float)group_scale);
                    const unsigned int qh = block[66u + group];
                    #pragma unroll
                    for (unsigned int grid = 0u; grid < 2u; ++grid) {
                        const unsigned int within = (local & 31u) + grid * 4u;
                        const unsigned int tuple = within >> 3u;
                        const unsigned int tuple_lane = within & 7u;
                        const unsigned int q = block[
                            2u + group * 8u + 2u * tuple + (tuple_lane >= 4u)];
                        const unsigned int shift = tuple_lane < 4u
                            ? 8u - 2u * tuple : 7u - 2u * tuple;
                        const unsigned int packed_grid
                            = zllm_iq3s_grid[q | ((qh << shift) & 256u)];
                        const unsigned int signs
                            = block[74u + group * 4u + tuple];
                        #pragma unroll
                        for (unsigned int item = 0u; item < 4u; ++item) {
                            const unsigned int magnitude
                                = (packed_grid >> (8u * item)) & 255u;
                            const float sign = (signs
                                & (1u << (tuple_lane + item))) != 0u
                                ? -1.0f : 1.0f;
                            weight_fragment[grid * 4u + item].data
                                = zllm_f32_to_bf16(
                                    scale * (float)magnitude * sign);
                        }
                    }
                } else {
                    #pragma unroll
                    for (unsigned int element = 0u; element < 8u; ++element)
                        weight_fragment[element].data = 0u;
                }
            } else {
                if (tensor_type == 21u) {
                    #pragma unroll
                    for (unsigned int row_half = 0u; row_half < 2u; ++row_half) {
                        const unsigned int tile_row = row_half * 8u + lane / 4u;
                        const unsigned int tile_column = (lane & 3u) * 4u;
                        const unsigned int output_row = output_base + tile_row;
                        if (output_row < output_rows) {
                            gguf_iq3s_store4_bf16(
                                &weight_tiles[wave][tile_row * 16u + tile_column],
                                gguf_kq_row(weight, tensor_type, output_row, input_columns),
                                segment_column_base + tile_column);
                        }
                    }
                } else {
                    #pragma unroll
                    for (unsigned int row_pair = 0u; row_pair < 16u; row_pair += 2u) {
                        const unsigned int tile_row = row_pair + (lane >> 4u);
                        const unsigned int output_row = output_base + tile_row;
                        weight_tiles[wave][tile_row * 16u + lane16]
                            = output_row < output_rows
                                ? gguf_kq_segment16_bf16(
                                    gguf_kq_row(weight, tensor_type, output_row, input_columns),
                                    tensor_type, segment_column_base + lane16,
                                    gguf_kq_block_bytes(tensor_type), lane16)
                                : 0u;
                    }
                }
                __syncwarp();
                load_matrix_sync(
                    weight_fragment, (const bfloat16_t*)weight_tiles[wave], 16);
            }
            #pragma unroll
            for (unsigned int tile = 0; tile < 8u; ++tile) {
                if (input_base + tile * 16u < input_rows) {
                    AFragment input_fragment;
                    load_matrix_sync(
                        input_fragment,
                        (const bfloat16_t*)input_tiles[segment * 8u + tile], 16);
                    mma_sync(
                        accumulator_fragments[tile], input_fragment,
                        weight_fragment, accumulator_fragments[tile]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int tile = 0; tile < 8; ++tile) {
        if (input_base + tile * 16 < input_rows)
            store_matrix_sync(
                scratch.output[wave], accumulator_fragments[tile], 16, mem_row_major);
        __syncthreads();
        if (wave < 4u && input_base + tile * 16 < input_rows) {
            for (unsigned int index = lane; index < 16 * 16; index += warpSize) {
                const unsigned int local_row = input_base + tile * 16 + index / 16;
                const unsigned int output_row = output_base + index % 16;
                if (local_row < input_rows && output_row < output_rows) {
                    const unsigned long long output_index
                        = (unsigned long long)(route_begin + local_row) * output_rows + output_row;
                    const float gate = scratch.output[wave][index];
                    const float up = scratch.output[wave + 4u][index];
                    activated[output_index]
                        = zllm_f32_to_bf16(gate / (1.0f + expf(-gate)) * up);
                }
            }
        }
        __syncthreads();
    }
    }
}

// 多行 prefill down 复用 CT grouped WMMA 的执行结构：每个 active expert/output
// tile 只启动一个 block，block 内顺序覆盖该 expert 的全部 route。旧 token-major
// kernel 会为每个 token×8 hidden 发一个 256-thread block，2048 行时单层超过
// 157 万个 workgroup；这里降为 expert×ceil(hidden/128)。
template <bool Iq4Xs>
__device__ __forceinline__ void gguf_grouped_down_impl(
    const unsigned short* activated,
    const unsigned int* token_ids,
    const float* grouped_weights,
    const GgufKqMeta* experts,
    const unsigned int* route_offsets,
    float* output,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int expert_base,
    unsigned int expert_count)
{
    using namespace rocwmma;
    using AFragment = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
    using BFragment = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
    using CFragment = fragment<accumulator, 16, 16, 16, float>;
    static_assert(BFragment::num_elements == 8, "GFX11 GGUF down B fragment lane layout 已改变");
    const unsigned int expert = expert_base + blockIdx.z;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave = threadIdx.x / warpSize;
    const unsigned int output_base = (blockIdx.x * 8u + wave) * 16u;
    if (blockIdx.z >= expert_count || wave >= 8u) return;
    const unsigned int route_begin = route_offsets[expert];
    const unsigned int input_rows = route_offsets[expert + 1u] - route_begin;
    if (input_rows == 0u) return;
    const GgufKqMeta& meta = experts[expert];
    const unsigned long long weight = meta.down;
    const unsigned int tensor_type = Iq4Xs ? 23u : meta.down_type;
    __shared__ unsigned short input_tiles[16][16 * 16];
    __shared__ unsigned short weight_tiles[Iq4Xs ? 1 : 8][16 * 16];
    __shared__ GgufWmmaScratch scratch;

    for (unsigned int input_base = 0u; input_base < input_rows; input_base += 128u) {
        CFragment accumulator_fragments[8];
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            if (input_base + tile * 16u < input_rows)
                fill_fragment(accumulator_fragments[tile], 0.0f);
        }
        const unsigned int stage_row = (threadIdx.x >> 2) & 15u;
        const unsigned int stage_chunk = threadIdx.x & 3u;
        const unsigned int stage_group = threadIdx.x >> 6;
        uint2 stage_data[4];
        unsigned int stage_valid[4] = {0u, 0u, 0u, 0u};
        auto stage_prefetch = [&](unsigned int column_base) {
            #pragma unroll
            for (unsigned int half = 0; half < 4; ++half) {
                const unsigned int tile = stage_group + half * 4u;
                const unsigned int column_offset = (half >> 1u) * 16u;
                const unsigned int route_row
                    = input_base + (tile & 7u) * 16u + stage_row;
                stage_valid[half] = route_row < input_rows ? 1u : 0u;
                stage_data[half] = stage_valid[half]
                    ? *reinterpret_cast<const uint2*>(
                        activated + (unsigned long long)(route_begin + route_row) * input_columns
                        + column_base + column_offset + stage_chunk * 4u)
                    : make_uint2(0u, 0u);
            }
        };
        stage_prefetch(0u);
        for (unsigned int column_base = 0u; column_base < input_columns; column_base += 32u) {
            #pragma unroll
            for (unsigned int half = 0; half < 4; ++half) {
                const unsigned int tile = stage_group + half * 4u;
                if (stage_valid[half]) {
                    *reinterpret_cast<uint2*>(
                        &input_tiles[tile][stage_row * 16u + stage_chunk * 4u])
                        = stage_data[half];
                }
            }
            if (column_base + 32u < input_columns)
                stage_prefetch(column_base + 32u);
            __syncthreads();
            const unsigned int lane16 = lane & 15u;
            if (Iq4Xs || tensor_type == 23u) {
                const unsigned int output_row = output_base + lane16;
                const unsigned int lane_column = (lane >> 4u) * 8u;
                unsigned long long packed = 0u;
                float scale = 0.0f;
                if (output_row < output_rows) {
                    const unsigned char* block
                        = gguf_kq_row(weight, tensor_type, output_row, input_columns)
                        + (unsigned long long)(column_base >> 8u) * 136u;
                    const unsigned int group = (column_base & 255u) >> 5u;
                    const unsigned int scales_h = (unsigned int)block[2]
                        | ((unsigned int)block[3] << 8u);
                    const unsigned int scales_l = block[4u + (group >> 1u)];
                    const unsigned int group_scale
                        = ((scales_l >> (4u * (group & 1u))) & 15u)
                        | (((scales_h >> (2u * group)) & 3u) << 4u);
                    scale = gguf_kq_f16(block) * ((float)group_scale - 32.0f);
                    packed = *reinterpret_cast<const unsigned long long*>(
                        block + 8u + group * 16u + lane_column);
                }
                #pragma unroll
                for (unsigned int segment = 0u; segment < 2u; ++segment) {
                    BFragment weight_fragment;
                    #pragma unroll
                    for (unsigned int element = 0u; element < 8u; ++element) {
                        const unsigned int byte
                            = (packed >> (8u * element)) & 255u;
                        const unsigned int code = segment == 0u
                            ? byte & 15u : byte >> 4u;
                        weight_fragment[element].data = zllm_f32_to_bf16(
                            scale * (float)zllm_iq4nl[code]);
                    }
                    #pragma unroll
                    for (unsigned int tile = 0; tile < 8u; ++tile) {
                        if (input_base + tile * 16u < input_rows) {
                            AFragment input_fragment;
                            load_matrix_sync(
                                input_fragment,
                                (const bfloat16_t*)input_tiles[segment * 8u + tile], 16);
                            mma_sync(
                                accumulator_fragments[tile], input_fragment,
                                weight_fragment, accumulator_fragments[tile]);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (unsigned int segment = 0; segment < 2u; ++segment) {
                    const unsigned int segment_column_base = column_base + segment * 16u;
                    if (segment_column_base >= input_columns) break;
                    BFragment weight_fragment;
                    #pragma unroll
                    for (unsigned int row_pair = 0u; row_pair < 16u; row_pair += 2u) {
                        const unsigned int tile_row = row_pair + (lane >> 4u);
                        const unsigned int output_row = output_base + tile_row;
                        weight_tiles[wave][tile_row * 16u + lane16]
                            = output_row < output_rows
                            ? gguf_kq_segment16_bf16(
                                gguf_kq_row(weight, tensor_type, output_row, input_columns),
                                tensor_type, segment_column_base + lane16,
                                gguf_kq_block_bytes(tensor_type), lane16)
                            : 0u;
                    }
                    __syncwarp();
                    load_matrix_sync(
                        weight_fragment,
                        (const bfloat16_t*)weight_tiles[wave], 16);
                    #pragma unroll
                    for (unsigned int tile = 0; tile < 8u; ++tile) {
                        if (input_base + tile * 16u < input_rows) {
                            AFragment input_fragment;
                            load_matrix_sync(
                                input_fragment,
                                (const bfloat16_t*)input_tiles[segment * 8u + tile], 16);
                            mma_sync(
                                accumulator_fragments[tile], input_fragment,
                                weight_fragment, accumulator_fragments[tile]);
                        }
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (unsigned int tile = 0; tile < 8; ++tile) {
            if (input_base + tile * 16u >= input_rows) continue;
            store_matrix_sync(scratch.output[wave], accumulator_fragments[tile], 16, mem_row_major);
            __syncwarp();
            for (unsigned int index = lane; index < 16u * 16u; index += warpSize) {
                const unsigned int local_row = input_base + tile * 16u + index / 16u;
                const unsigned int output_row = output_base + index % 16u;
                if (local_row < input_rows && output_row < output_rows) {
                    const unsigned int route = route_begin + local_row;
                    atomicAdd(
                        &output[(unsigned long long)token_ids[route] * output_rows + output_row],
                        scratch.output[wave][index] * grouped_weights[route]);
                }
            }
            __syncwarp();
        }
    }
}

extern "C" __global__ void gguf_grouped_down_f32(
    const unsigned short* activated,
    const unsigned int* token_ids,
    const float* grouped_weights,
    const GgufKqMeta* experts,
    const unsigned int* route_offsets,
    float* output,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int expert_base,
    unsigned int expert_count)
{
    gguf_grouped_down_impl<false>(activated, token_ids, grouped_weights,
        experts, route_offsets, output, input_columns, output_rows,
        expert_base, expert_count);
}

extern "C" __global__ void gguf_grouped_down_iq4xs_f32(
    const unsigned short* activated,
    const unsigned int* token_ids,
    const float* grouped_weights,
    const GgufKqMeta* experts,
    const unsigned int* route_offsets,
    float* output,
    unsigned int input_columns,
    unsigned int output_rows,
    unsigned int expert_base,
    unsigned int expert_count)
{
    gguf_grouped_down_impl<true>(activated, token_ids, grouped_weights,
        experts, route_offsets, output, input_columns, output_rows,
        expert_base, expert_count);
}
