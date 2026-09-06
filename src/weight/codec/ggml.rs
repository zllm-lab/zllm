//! GGML block quant 布局与 reference decode。

const QK_K: usize = 256;
const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

pub fn iq2s_grid() -> &'static [u64; 1024] {
    &super::iq2s_grid::IQ2S_GRID
}

pub fn iq3xxs_grid() -> &'static [u32; 256] {
    &super::iq3xxs_grid::IQ3XXS_GRID
}

pub fn ksigns_iq2xs() -> &'static [u8; 128] {
    &super::iq3xxs_grid::KSIGNS_IQ2XS
}

pub fn iq3s_grid() -> &'static [u32; 512] {
    &super::iq3xxs_grid::IQ3S_GRID
}

pub fn iq2xs_grid() -> &'static [u64; 512] {
    &super::iq3xxs_grid::IQ2XS_GRID
}

/// Metal/CPU 共享的量化类型白名单：这些类型既有 CPU reference dequant 也有 Metal kernel。
/// 新增量化类型时只需改这一处，所有 backend 的 dispatch 自动跟随。
pub fn supports_decode(tensor_type: u32) -> bool {
    matches!(tensor_type, 2 | 8 | 11 | 12 | 13 | 14 | 17 | 18 | 20 | 21 | 22 | 23 | 30 | 39)
}

pub(crate) fn block_layout(tensor_type: u32) -> Result<(usize, usize), String> {
    match tensor_type {
        0 => Ok((1, 4)),
        1 => Ok((1, 2)),
        2 => Ok((32, 18)),
        3 => Ok((32, 20)),
        6 => Ok((32, 22)),
        7 => Ok((32, 24)),
        8 => Ok((32, 34)),
        9 => Ok((32, 36)),
        10 => Ok((QK_K, 84)),
        11 => Ok((QK_K, 110)),
        12 => Ok((QK_K, 144)),
        13 => Ok((QK_K, 176)),
        14 => Ok((QK_K, 210)),
        15 => Ok((QK_K, 292)),
        16 => Ok((QK_K, 66)),
        17 => Ok((QK_K, 74)),
        18 => Ok((QK_K, 98)),
        19 => Ok((QK_K, 50)),
        20 => Ok((32, 18)),
        21 => Ok((QK_K, 110)),
        22 => Ok((QK_K, 82)),
        23 => Ok((QK_K, 136)),
        24 => Ok((1, 1)),
        25 => Ok((1, 2)),
        26 => Ok((1, 4)),
        27 | 28 => Ok((1, 8)),
        29 => Ok((QK_K, 56)),
        30 => Ok((1, 2)),
        34 => Ok((QK_K, 54)),
        35 => Ok((QK_K, 66)),
        39 => Ok((32, 17)),
        40 => Ok((64, 36)),
        41 => Ok((128, 18)),
        42 => Ok((64, 18)),
        other => Err(format!("尚不支持 GGML tensor type {other}")),
    }
}

pub(crate) fn decode_block(tensor_type: u32, bytes: &[u8], output: &mut [f32]) -> Result<(), String> {
    let (elements, block_bytes) = block_layout(tensor_type)?;
    if bytes.len() != block_bytes || output.len() != elements {
        return Err(format!("GGML block 布局不匹配: type={tensor_type} bytes={}/{} output={}/{}", bytes.len(), block_bytes, output.len(), elements));
    }
    match tensor_type {
        0 => output[0] = f32::from_le_bytes(bytes.try_into().expect("F32 block")),
        1 => output[0] = half::f16::from_bits(u16::from_le_bytes(bytes.try_into().expect("F16 block"))).to_f32(),
        30 => output[0] = f32::from_bits((u16::from_le_bytes(bytes.try_into().expect("BF16 block")) as u32) << 16),
        39 => decode_mxfp4(bytes, output.try_into().expect("MXFP4 block")),
        2 => decode_q4_0(bytes, output.try_into().expect("Q4_0 block")),
        6 => decode_q5_0(bytes, output.try_into().expect("Q5_0 block")),
        7 => decode_q5_1(bytes, output.try_into().expect("Q5_1 block")),
        8 => decode_q8_0(bytes, output.try_into().expect("Q8_0 block")),
        11 => decode_q3_k(bytes, output.try_into().expect("Q3_K block")),
        12 => decode_q4_k(bytes, output.try_into().expect("Q4_K block")),
        13 => decode_q5_k(bytes, output.try_into().expect("Q5_K block")),
        14 => decode_q6_k(bytes, output.try_into().expect("Q6_K block")),
        22 => decode_iq2_s(bytes, output.try_into().expect("IQ2_S block")),
        18 => decode_iq3_xxs(bytes, output.try_into().expect("IQ3_XXS block")),
        21 => decode_iq3_s(bytes, output.try_into().expect("IQ3_S block")),
        17 => decode_iq2_xs(bytes, output.try_into().expect("IQ2_XS block")),
        20 => decode_iq4_nl(bytes, output.try_into().expect("IQ4_NL block")),
        23 => decode_iq4_xs(bytes, output.try_into().expect("IQ4_XS block")),
        other => return Err(format!("尚不支持 GGML tensor type {other} 解码")),
    }
    Ok(())
}

fn decode_q4_0(block: &[u8], output: &mut [f32; 32]) {
    let scale = half::f16::from_bits(u16::from_le_bytes(block[..2].try_into().expect("Q4_0 scale"))).to_f32();
    for index in 0..16 {
        let pair = block[2 + index];
        output[index] = ((pair & 0x0f) as i32 - 8) as f32 * scale;
        output[index + 16] = ((pair >> 4) as i32 - 8) as f32 * scale;
    }
}

fn decode_q5_0(block: &[u8], output: &mut [f32; 32]) {
    let scale = read_f16(block, 0);
    let high = u32::from_le_bytes(block[2..6].try_into().expect("Q5_0 high bits"));
    for index in 0..32 {
        let packed = block[6 + index % 16];
        let low = if index < 16 { packed & 15 } else { packed >> 4 };
        output[index] = scale * (((low as u32 | (((high >> index) & 1) << 4)) as i32 - 16) as f32);
    }
}

fn decode_q5_1(block: &[u8], output: &mut [f32; 32]) {
    let scale = read_f16(block, 0);
    let minimum = read_f16(block, 2);
    let high = u32::from_le_bytes(block[4..8].try_into().expect("Q5_1 high bits"));
    for index in 0..16 {
        let low = block[8 + index];
        output[index] = scale * ((low & 15) as u32 | (((high >> index) & 1) << 4)) as f32 + minimum;
        output[index + 16] = scale * ((low >> 4) as u32 | (((high >> (index + 16)) & 1) << 4)) as f32 + minimum;
    }
}

fn decode_mxfp4(block: &[u8], output: &mut [f32; 32]) {
    // GGML 保存 doubled E2M1 查表值，因此共享 E8M0 scale 需要再除以 2。
    let scale = match block[0] {
        0 => f32::from_bits(0x0020_0000),
        1 => f32::from_bits(0x0040_0000),
        exponent => f32::from_bits((exponent as u32 - 1) << 23),
    };
    for index in 0..16 {
        let pair = block[1 + index];
        output[index] = KVALUES_MXFP4[(pair & 0x0f) as usize] as f32 * scale;
        output[index + 16] = KVALUES_MXFP4[(pair >> 4) as usize] as f32 * scale;
    }
}

pub fn dequantize(tensor_type: u32, bytes: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    let (block_elements, block_bytes) = block_layout(tensor_type)?;
    if !elements.is_multiple_of(block_elements) {
        return Err(format!("GGML tensor type {tensor_type} 元素数 {elements} 未按 block {block_elements} 对齐"));
    }
    let expected = elements.checked_div(block_elements).and_then(|blocks| blocks.checked_mul(block_bytes)).ok_or_else(|| "GGML tensor 字节数溢出".to_owned())?;
    if bytes.len() != expected {
        return Err(format!("GGML tensor type {tensor_type} 字节数不匹配: 期望 {expected}，实际 {}", bytes.len()));
    }
    let mut output = vec![0.0; elements];
    for (block, values) in bytes.chunks_exact(block_bytes).zip(output.chunks_exact_mut(block_elements)) {
        decode_block(tensor_type, block, values)?;
    }
    Ok(output)
}

/// 把现有 GGML block 流式重编码为 Q4_K；只保留一个 256 元素 scratch，避免加载时展开整块矩阵。
pub fn requantize_q4_k(tensor_type: u32, bytes: &[u8], elements: usize) -> Result<Vec<u8>, String> {
    let (source_elements, source_bytes) = block_layout(tensor_type)?;
    if !matches!(tensor_type, 8 | 12 | 13) || !elements.is_multiple_of(QK_K) || !QK_K.is_multiple_of(source_elements) {
        return Err(format!("GGML type={tensor_type} elements={elements} 不能重编码为 Q4_K"));
    }
    let expected = elements / source_elements * source_bytes;
    if bytes.len() != expected {
        return Err(format!("GGML type={tensor_type} 输入字节 {}，期望 {expected}", bytes.len()));
    }
    if tensor_type == 12 {
        return Ok(bytes.to_vec());
    }
    let mut output = vec![0_u8; elements / QK_K * 144];
    let mut values = [0_f32; QK_K];
    for (block_index, target) in output.chunks_exact_mut(144).enumerate() {
        let first = block_index * (QK_K / source_elements);
        for source in 0..QK_K / source_elements {
            let offset = (first + source) * source_bytes;
            decode_block(tensor_type, &bytes[offset..offset + source_bytes], &mut values[source * source_elements..(source + 1) * source_elements])?;
        }
        encode_q4_k(&values, target);
    }
    Ok(output)
}

fn encode_q4_k(values: &[f32; QK_K], block: &mut [u8]) {
    let mut scales = [0_f32; 8];
    let mut minimums = [0_f32; 8];
    for group in 0..8 {
        let values = &values[group * 32..(group + 1) * 32];
        let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        minimums[group] = (-minimum).max(0.0);
        scales[group] = (maximum + minimums[group]).max(0.0) / 15.0;
    }
    let d = half::f16::from_f32(scales.iter().copied().fold(0_f32, f32::max) / 63.0).to_f32();
    let dmin = half::f16::from_f32(minimums.iter().copied().fold(0_f32, f32::max) / 63.0).to_f32();
    block.fill(0);
    block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
    let mut scale_codes = [0_u8; 8];
    let mut minimum_codes = [0_u8; 8];
    for group in 0..8 {
        scale_codes[group] = if d == 0.0 { 0 } else { (scales[group] / d).round().clamp(0.0, 63.0) as u8 };
        minimum_codes[group] = if dmin == 0.0 { 0 } else { (minimums[group] / dmin).round().clamp(0.0, 63.0) as u8 };
    }
    for group in 0..4 {
        block[4 + group] = (scale_codes[group] & 0x3f) | ((scale_codes[group + 4] >> 4) << 6);
        block[8 + group] = (minimum_codes[group] & 0x3f) | ((minimum_codes[group + 4] >> 4) << 6);
        block[12 + group] = (scale_codes[group + 4] & 0x0f) | ((minimum_codes[group + 4] & 0x0f) << 4);
    }
    for group in 0..8 {
        let scale = d * scale_codes[group] as f32;
        let minimum = dmin * minimum_codes[group] as f32;
        for index in 0..32 {
            let quant = if scale == 0.0 { 0 } else { ((values[group * 32 + index] + minimum) / scale).round().clamp(0.0, 15.0) as u8 };
            let target = 16 + (group / 2) * 32 + index;
            if group.is_multiple_of(2) {
                block[target] = quant;
            } else {
                block[target] |= quant << 4;
            }
        }
    }
}

fn decode_q8_0(block: &[u8], output: &mut [f32; 32]) {
    let d = read_f16(block, 0);
    for (value, quant) in output.iter_mut().zip(&block[2..]) {
        *value = d * (*quant as i8) as f32;
    }
}

// IQ4_NL: d(f16) + qs[16]，每字节低 nibble 是前 16 值、高 nibble 是后 16 值，查 KVALUES_IQ4NL 码本。
fn decode_iq4_nl(block: &[u8], output: &mut [f32; 32]) {
    let d = read_f16(block, 0);
    let quants = &block[2..18];
    for (j, &q) in quants.iter().enumerate() {
        output[j] = d * KVALUES_IQ4NL[(q & 0x0f) as usize] as f32;
        output[j + 16] = d * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
    }
}

fn decode_iq4_xs(block: &[u8], output: &mut [f32; QK_K]) {
    let d = read_f16(block, 0);
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let quants = &block[8..136];
    for ib in 0..8 {
        let ls = ((scales_l[ib / 2] as usize >> (4 * (ib % 2))) & 0x0f) | (((scales_h >> (2 * ib)) & 0x03) as usize) << 4;
        let dl = d * (ls as f32 - 32.0);
        let source = ib * 16;
        for j in 0..16 {
            let q = quants[source + j];
            output[ib * 32 + j] = dl * KVALUES_IQ4NL[(q & 0x0f) as usize] as f32;
            output[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
        }
    }
}

fn decode_iq2_s(block: &[u8], output: &mut [f32; QK_K]) {
    let d = read_f16(block, 0);
    let quants = &block[2..66];
    let high_bits = &block[66..74];
    let scales = &block[74..82];

    for group in 0..8 {
        let scale = scales[group];
        let group_scales = [d * (0.5 + (scale & 0x0f) as f32) * 0.25, d * (0.5 + (scale >> 4) as f32) * 0.25];
        let quant_base = group * 4;
        let sign_base = 32 + group * 4;
        for vector in 0..4 {
            let high = (((high_bits[group] as u16) << (8 - 2 * vector)) & 0x0300) as usize;
            let grid_index = quants[quant_base + vector] as usize | high;
            let grid = super::iq2s_grid::IQ2S_GRID[grid_index];
            let signs = quants[sign_base + vector];
            let scale = group_scales[vector / 2];
            for lane in 0..8 {
                let magnitude = ((grid >> (8 * lane)) & 0xff) as f32;
                let sign = if signs & (1 << lane) != 0 { -1.0 } else { 1.0 };
                output[group * 32 + vector * 8 + lane] = scale * magnitude * sign;
            }
        }
    }
}

/// IQ3_XXS: 256 权重 / 98 字节。d(2B) + qs[0..64] 码本索引 + qs[64..96] 缩放与符号。
fn decode_iq3_xxs(block: &[u8], output: &mut [f32; QK_K]) {
    let d = read_f16(block, 0);
    let qs = &block[2..98];
    let grid = super::iq3xxs_grid::IQ3XXS_GRID;
    let signs_table = super::iq3xxs_grid::KSIGNS_IQ2XS;

    for ib32 in 0..8 {
        // qs[64 + 4*ib32 ..] 读小端 u32:高 4 位缩放 + 4 个 7 位符号索引
        let scales_base = 64 + ib32 * 4;
        let aux32 = u32::from_le_bytes([qs[scales_base], qs[scales_base + 1], qs[scales_base + 2], qs[scales_base + 3]]);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;

        let qs_base = ib32 * 8;
        let out_base = ib32 * 32;
        for l in 0..4 {
            let signs = signs_table[((aux32 >> (7 * l)) & 127) as usize];
            let grid1 = grid[qs[qs_base + 2 * l] as usize].to_le_bytes();
            let grid2 = grid[qs[qs_base + 2 * l + 1] as usize].to_le_bytes();
            for j in 0..4 {
                let sign1 = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                let sign2 = if signs & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                output[out_base + l * 8 + j] = db * grid1[j] as f32 * sign1;
                output[out_base + l * 8 + j + 4] = db * grid2[j] as f32 * sign2;
            }
        }
    }
}

/// IQ3_S: 256 权重 / 110 字节。d(2B) + qs[64] + qh[8] + signs[32] + scales[4]。
fn decode_iq3_s(block: &[u8], output: &mut [f32; QK_K]) {
    let d = read_f16(block, 0);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    let grid = super::iq3xxs_grid::IQ3S_GRID;
    let kmask = super::iq3xxs_grid::KMASK_IQ2XS;

    let mut yi = 0usize;
    let mut qs_off = 0usize;
    let mut qh_off = 0usize;
    let mut sg_off = 0usize;
    for sc_off in 0..4 {
        // 每个 scale 字节编码两个 ib32 的 4-bit scale
        for &(db, qh_idx) in &[(d * (1.0 + 2.0 * (scales[sc_off] & 0xf) as f32), qh_off), (d * (1.0 + 2.0 * (scales[sc_off] >> 4) as f32), qh_off + 1)] {
            let qh_val = qh[qh_idx] as u32;
            for l in 0..4 {
                let qs0 = qs[qs_off + 2 * l] as usize;
                let qs1 = qs[qs_off + 2 * l + 1] as usize;
                let g1 = grid[qs0 | (((qh_val << (8 - 2 * l)) & 256) as usize)].to_le_bytes();
                let g2 = grid[qs1 | (((qh_val << (7 - 2 * l)) & 256) as usize)].to_le_bytes();
                let signs_byte = signs[sg_off + l];
                for j in 0..4 {
                    let s1 = if signs_byte & kmask[j] != 0 { -1.0 } else { 1.0 };
                    let s2 = if signs_byte & kmask[j + 4] != 0 { -1.0 } else { 1.0 };
                    output[yi + j] = db * g1[j] as f32 * s1;
                    output[yi + j + 4] = db * g2[j] as f32 * s2;
                }
                yi += 8;
            }
            qs_off += 8;
            sg_off += 4;
        }
        qh_off += 2;
    }
}

/// IQ2_XS: 256 权重 / 74 字节。d(2B) + qs[32 × u16 = 64B] + scales[8]。
fn decode_iq2_xs(block: &[u8], output: &mut [f32; QK_K]) {
    let d = read_f16(block, 0);
    let grid = super::iq3xxs_grid::IQ2XS_GRID;
    let ksigns = super::iq3xxs_grid::KSIGNS_IQ2XS;
    let kmask = super::iq3xxs_grid::KMASK_IQ2XS;

    let mut yi = 0usize;
    for ib32 in 0..8 {
        let db0 = d * (0.5 + (block[66 + ib32] & 0xf) as f32) * 0.25;
        let db1 = d * (0.5 + (block[66 + ib32] >> 4) as f32) * 0.25;
        for l in 0..4 {
            let q_offset = 2 + (4 * ib32 + l) * 2;
            let q = u16::from_le_bytes([block[q_offset], block[q_offset + 1]]);
            let grid_idx = (q & 0x1ff) as usize;
            let signs_byte = ksigns[(q >> 9) as usize];
            let g = grid[grid_idx].to_le_bytes();
            let db = if l < 2 { db0 } else { db1 };
            for j in 0..8 {
                let s = if signs_byte & kmask[j] != 0 { -1.0 } else { 1.0 };
                output[yi + j] = db * g[j] as f32 * s;
            }
            yi += 8;
        }
    }
}

fn decode_q3_k(block: &[u8], output: &mut [f32; QK_K]) {
    let hmask = &block[..32];
    let quants = &block[32..96];
    let scales = &block[96..108];
    let d = read_f16(block, 108);

    for group in 0..16 {
        let low = if group < 8 { scales[group] & 0x0f } else { scales[group - 8] >> 4 };
        let high = (scales[8 + group % 4] >> (2 * (group / 4))) & 0x03;
        let scale = (low | (high << 4)) as i32 - 32;
        let half = group / 8;
        let pair = group % 8;
        let source = half * 32 + (pair % 2) * 16;
        let mask_source = (pair % 2) * 16;
        let shift = 2 * (pair / 2);
        let mask = 1u8 << (group / 2);
        for index in 0..16 {
            let low_bits = ((quants[source + index] >> shift) & 0x03) as i32;
            let quant = low_bits - if hmask[mask_source + index] & mask == 0 { 4 } else { 0 };
            output[group * 16 + index] = d * scale as f32 * quant as f32;
        }
    }
}

fn decode_q4_k(block: &[u8], output: &mut [f32; QK_K]) {
    let d = read_f16(block, 0);
    let dmin = read_f16(block, 2);
    let scales = &block[4..16];
    let quants = &block[16..144];

    for group in 0..8 {
        let (scale, min) = scale_min_k4(group, scales);
        let source = (group / 2) * 32;
        for index in 0..32 {
            let packed = quants[source + index];
            let quant = if group % 2 == 0 { packed & 0x0f } else { packed >> 4 };
            output[group * 32 + index] = d * scale as f32 * quant as f32 - dmin * min as f32;
        }
    }
}

fn decode_q5_k(block: &[u8], output: &mut [f32; QK_K]) {
    let d = read_f16(block, 0);
    let dmin = read_f16(block, 2);
    let scales = &block[4..16];
    let high_bits = &block[16..48];
    let low_bits = &block[48..176];

    for group in 0..8 {
        let (scale, min) = scale_min_k4(group, scales);
        let source = (group / 2) * 32;
        let high_mask = 1u8 << group;
        for index in 0..32 {
            let packed = low_bits[source + index];
            let low = if group % 2 == 0 { packed & 0x0f } else { packed >> 4 };
            let quant = low + if high_bits[index] & high_mask != 0 { 16 } else { 0 };
            output[group * 32 + index] = d * scale as f32 * quant as f32 - dmin * min as f32;
        }
    }
}

fn decode_q6_k(block: &[u8], output: &mut [f32; QK_K]) {
    let low_bits = &block[..128];
    let high_bits = &block[128..192];
    let scales = &block[192..208];
    let d = read_f16(block, 208);

    for half in 0..2 {
        let low = half * 64;
        let high = half * 32;
        let scale = half * 8;
        let target = half * 128;
        for index in 0..32 {
            let scale_index = index / 16;
            let high_value = high_bits[high + index];
            let q1 = ((low_bits[low + index] & 0x0f) | ((high_value & 3) << 4)) as i32 - 32;
            let q2 = ((low_bits[low + index + 32] & 0x0f) | (((high_value >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((low_bits[low + index] >> 4) | (((high_value >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((low_bits[low + index + 32] >> 4) | (((high_value >> 6) & 3) << 4)) as i32 - 32;
            output[target + index] = d * (scales[scale + scale_index] as i8) as f32 * q1 as f32;
            output[target + index + 32] = d * (scales[scale + scale_index + 2] as i8) as f32 * q2 as f32;
            output[target + index + 64] = d * (scales[scale + scale_index + 4] as i8) as f32 * q3 as f32;
            output[target + index + 96] = d * (scales[scale + scale_index + 6] as i8) as f32 * q4 as f32;
        }
    }
}

fn scale_min_k4(group: usize, scales: &[u8]) -> (u8, u8) {
    if group < 4 { (scales[group] & 0x3f, scales[group + 4] & 0x3f) } else { ((scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4), (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)) }
}

fn read_f16(bytes: &[u8], offset: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q5_0_preserves_signed_fifth_bit() {
        let mut block = [0u8; 22];
        block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        block[2..6].copy_from_slice(&0xffff0000u32.to_le_bytes());
        for i in 0..16 {
            block[6 + i] = i as u8 | ((15 - i) as u8) << 4;
        }
        let result = dequantize(6, &block, 32).unwrap();
        for i in 0..16 {
            assert_eq!(result[i], (i as f32 - 16.0) * 0.5);
            assert_eq!(result[16 + i], (15 - i) as f32 * 0.5);
        }
    }

    #[test]
    fn q5_1_preserves_unsigned_fifth_bit_and_minimum() {
        let mut block = [0u8; 24];
        block[..2].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(-3.0).to_le_bytes());
        block[4..8].copy_from_slice(&0xffff0000u32.to_le_bytes());
        for index in 0..16 {
            block[8 + index] = (index | (index << 4)) as u8;
        }
        let values = dequantize(7, &block, 32).unwrap();
        for (index, value) in values.iter().enumerate() {
            assert_eq!(*value, index as f32 * 0.25 - 3.0);
        }
    }

    #[test]
    fn decodes_iq4_nl_block_layout() {
        let mut block = vec![0u8; 18];
        block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        block[2] = 0x0f; // j=0 低 nibble 15 → 113，j=16 高 nibble 0 → -127
        block[3] = 0x98; // j=1 低 nibble 8 → 1，j=17 高 nibble 9 → 13
        let output = dequantize(20, &block, 32).unwrap();
        assert_eq!(output[0], 0.5 * 113.0);
        assert_eq!(output[16], 0.5 * -127.0);
        assert_eq!(output[1], 0.5 * 1.0);
        assert_eq!(output[17], 0.5 * 13.0);
        // 其余字节为 0 → nibble 0 → 码本首值 -127
        assert!(output[2..16].iter().chain(&output[18..]).all(|value| *value == 0.5 * -127.0));
    }

    #[test]
    fn decodes_plain_and_q8_0() {
        assert_eq!(dequantize(1, &[0x00, 0x3c, 0x00, 0xc0], 2).unwrap(), vec![1.0, -2.0]);
        let mut block = vec![0u8; 34];
        block[..2].copy_from_slice(&0x4000u16.to_le_bytes());
        block[2] = (-3i8) as u8;
        block[3] = 4;
        let output = dequantize(8, &block, 32).unwrap();
        assert_eq!(&output[..2], &[-6.0, 8.0]);
    }

    #[test]
    fn requantizes_q8_0_to_standard_q4_k_layout() {
        let mut source = vec![0_u8; 8 * 34];
        for (block, chunk) in source.chunks_exact_mut(34).enumerate() {
            chunk[..2].copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
            for (index, quant) in chunk[2..].iter_mut().enumerate() {
                *quant = ((block * 17 + index * 7) as i32 % 255 - 127) as i8 as u8;
            }
        }
        let expected = dequantize(8, &source, QK_K).unwrap();
        let packed = requantize_q4_k(8, &source, QK_K).unwrap();
        assert_eq!(packed.len(), 144);
        let actual = dequantize(12, &packed, QK_K).unwrap();
        let max_error = actual.iter().zip(&expected).map(|(actual, expected)| (actual - expected).abs()).fold(0_f32, f32::max);
        assert!(max_error < 0.3, "Q8_0 -> Q4_K max_error={max_error}");
    }

    #[test]
    fn decodes_bf16_and_standard_mxfp4_layout() {
        assert_eq!(dequantize(30, &1.5f32.to_bits().to_le_bytes()[2..], 1).unwrap(), vec![1.5]);

        let mut block = vec![0u8; 17];
        block[0] = 127;
        // GGML 一个 byte 的低 nibble 是前半区 j，高 nibble 是后半区 j+16。
        block[1] = 0x92;
        let output = dequantize(39, &block, 32).unwrap();
        assert_eq!(output[0], 1.0);
        assert_eq!(output[16], -0.5);
        assert!(output[1..16].iter().chain(&output[17..]).all(|value| *value == 0.0));

        // E8M0 的 0/1 编码是 2^-128/2^-127；覆盖普通 pow 路径最容易漏掉的 subnormal。
        block[0] = 0;
        block[1] = 0x01;
        assert_eq!(dequantize(39, &block, 32).unwrap()[0].to_bits(), 0x0020_0000);
        block[0] = 1;
        assert_eq!(dequantize(39, &block, 32).unwrap()[0].to_bits(), 0x0040_0000);
    }

    #[test]
    fn decodes_k_quant_block_layouts() {
        let mut q3 = vec![0u8; 110];
        q3[0] = 1;
        q3[32] = 3;
        q3[96] = 1;
        q3[104] = 2;
        q3[108..].copy_from_slice(&0x3c00u16.to_le_bytes());
        assert_eq!(dequantize(11, &q3, QK_K).unwrap()[0], 3.0);

        let mut q4 = vec![0u8; 144];
        q4[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        q4[4] = 1;
        q4[16] = 2;
        assert_eq!(dequantize(12, &q4, QK_K).unwrap()[0], 2.0);

        let mut q5 = vec![0u8; 176];
        q5[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        q5[4] = 1;
        q5[16] = 1;
        q5[48] = 2;
        assert_eq!(dequantize(13, &q5, QK_K).unwrap()[0], 18.0);

        let mut q6 = vec![0u8; 210];
        q6[192] = 1;
        q6[208..].copy_from_slice(&0x3c00u16.to_le_bytes());
        assert_eq!(dequantize(14, &q6, QK_K).unwrap()[0], -32.0);
    }

    #[test]
    fn decodes_iq2_s_grid_scale_and_sign() {
        let mut block = vec![0u8; 82];
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[34] = 1;
        let output = dequantize(22, &block, QK_K).unwrap();
        assert_eq!(&output[..2], &[-1.0, 1.0]);
    }

    #[test]
    fn decodes_iq4_xs_non_zero() {
        let mut block = vec![0u8; 136];
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[2..4].copy_from_slice(&2u16.to_le_bytes());
        block[4] = 1;
        block[8] = 0xF0;
        let output = dequantize(23, &block, QK_K).unwrap();
        assert_eq!(output[0], -127.0);
        assert_eq!(output[16], 113.0);
    }

    #[test]
    fn decodes_iq3_s_zero_block() {
        // d=1.0, 全零 block: iq3s_grid[0] 的值决定输出
        let mut block = vec![0u8; 110];
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        let output = dequantize(21, &block, QK_K).unwrap();
        // scales 全零 → db = d*(1+0) = 1.0, grid[0] 值乘 db
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decodes_iq2_xs_zero_block() {
        let mut block = vec![0u8; 74];
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        let output = dequantize(17, &block, QK_K).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decodes_iq3_xxs_zero_block() {
        // d=1.0, 全零 qs: 每个 grid 查 iq3xxs_grid[0]=0x04040404 → 值 4.0,
        // aux32=0 → db = 1.0*(0.5+0)*0.5 = 0.25, signs=ksigns[0]=0(全正)
        let mut block = vec![0u8; 98];
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        let output = dequantize(18, &block, QK_K).unwrap();
        assert!((output[0] - 0.25 * 4.0).abs() < 1e-6);
        assert!((output[4] - 0.25 * 4.0).abs() < 1e-6);
    }

    #[test]
    fn decodes_iq3_xxs_nonzero_scale_and_signs() {
        // 设 aux32 高 4 位 = 1 (scale=1), grid index = 0, signs 索引 = 1
        let mut block = vec![0u8; 98];
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        // qs[0..8] = 码本索引,qs[64..96] = 缩放+符号。设 ib32=0 的 aux32:
        // 高 4 位 = 1 (db = 1.0*(0.5+1)*0.5 = 0.75)
        // 低 7 位 signs 索引(l=0) = 1 → ksigns[1] = 129 = 0b10000001 (bit0 和 bit7 取反)
        let aux32: u32 = (1u32 << 28) | 1; // scale=1, signs_idx_l0=1
        block[66..70].copy_from_slice(&aux32.to_le_bytes()); // qs[64..68] = ib32=0 的 scales_and_signs
        let output = dequantize(18, &block, QK_K).unwrap();
        let db = 0.75f32;
        let grid_val = 4.0f32; // iq3xxs_grid[0] = 0x04040404 → 每字节 = 4
        // l=0, grid1=grid2=4.0, signs=129 → bit0 取反 → output[0] = -db*4, output[1..3] = +db*4
        assert!((output[0] - (-db * grid_val)).abs() < 1e-6);
        assert!((output[1] - (db * grid_val)).abs() < 1e-6);
    }
}
