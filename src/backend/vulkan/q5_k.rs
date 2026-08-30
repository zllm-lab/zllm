use super::super::BackendError;

pub const BLOCK_BYTES: usize = 176;
const BLOCK_VALUES: usize = 256;
pub const PREFILL_TOKEN_TILE: usize = 4;

pub fn validate_weight(packed: &[u8], rows: usize, cols: usize) -> Result<(), BackendError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(BLOCK_VALUES) {
        return Err(BackendError::Compute { msg: format!("Q5_K weight shape 无效: rows={rows}, cols={cols}") });
    }
    let expected = rows * (cols / BLOCK_VALUES) * BLOCK_BYTES;
    if packed.len() != expected {
        return Err(BackendError::Compute { msg: format!("Q5_K 权重字节数错误: 期望 {expected}, 实际 {}", packed.len()) });
    }
    Ok(())
}

pub const SHADER: &str = r#"
struct Params { rows: u32, cols: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> partial: array<f32, 64>;

fn byte_at(word: u32, byte: u32) -> u32 {
    return (weights[word] >> (byte * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let blocks = params.cols / 256u;
    var sum = 0.0;
    let index = local.x % 32u;
    let first_q_group = local.x / 32u;
    var block = 0u;
    loop {
        if block >= blocks { break; }
        let base = (row * blocks + block) * 44u;
        let factors = unpack2x16float(weights[base]);
        var slot = 0u;
        loop {
            if slot >= 4u { break; }
            let q_group = first_q_group + slot * 2u;
            var scale: u32;
            var minimum: u32;
            if q_group < 4u {
                scale = byte_at(base + 1u, q_group) & 63u;
                minimum = byte_at(base + 2u, q_group) & 63u;
            } else {
                let byte = q_group - 4u;
                scale = (byte_at(base + 3u, byte) & 15u) | ((byte_at(base + 1u, byte) >> 6u) << 4u);
                minimum = (byte_at(base + 3u, byte) >> 4u) | ((byte_at(base + 2u, byte) >> 6u) << 4u);
            }
            let low_byte = byte_at(base + 12u + slot * 8u + index / 4u, index % 4u);
            let low = select(low_byte & 15u, low_byte >> 4u, first_q_group == 1u);
            let high_byte = byte_at(base + 4u + index / 4u, index % 4u);
            let quant = low + select(0u, 16u, (high_byte & (1u << q_group)) != 0u);
            let column = block * 256u + local.x + slot * 64u;
            sum += (factors.x * f32(scale * quant) - factors.y * f32(minimum)) * input[group.y * params.cols + column];
            slot += 1u;
        }
        block += 1u;
    }
    partial[local.x] = sum;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if local.x < stride { partial[local.x] += partial[local.x + stride]; }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    if local.x == 0u { output[group.y * params.rows + row] = partial[0]; }
}
"#;

pub const GATED_SHADER: &str = r#"
struct Params { rows: u32, cols: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> gate_weights: array<u32>;
@group(0) @binding(1) var<storage, read> up_weights: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> gate_partial: array<f32, 64>;
var<workgroup> up_partial: array<f32, 64>;

fn gate_byte(word: u32, byte: u32) -> u32 { return (gate_weights[word] >> (byte * 8u)) & 255u; }
fn up_byte(word: u32, byte: u32) -> u32 { return (up_weights[word] >> (byte * 8u)) & 255u; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let blocks = params.cols / 256u;
    var gate_sum = 0.0;
    var up_sum = 0.0;
    let index = local.x % 32u;
    let first_q_group = local.x / 32u;
    var block = 0u;
    loop {
        if block >= blocks { break; }
        let base = (row * blocks + block) * 44u;
        let gate_factors = unpack2x16float(gate_weights[base]);
        let up_factors = unpack2x16float(up_weights[base]);
        var slot = 0u;
        loop {
            if slot >= 4u { break; }
            let q_group = first_q_group + slot * 2u;
            var gate_scale: u32; var gate_minimum: u32;
            var up_scale: u32; var up_minimum: u32;
            if q_group < 4u {
                gate_scale = gate_byte(base + 1u, q_group) & 63u; gate_minimum = gate_byte(base + 2u, q_group) & 63u;
                up_scale = up_byte(base + 1u, q_group) & 63u; up_minimum = up_byte(base + 2u, q_group) & 63u;
            } else {
                let byte = q_group - 4u;
                gate_scale = (gate_byte(base + 3u, byte) & 15u) | ((gate_byte(base + 1u, byte) >> 6u) << 4u);
                gate_minimum = (gate_byte(base + 3u, byte) >> 4u) | ((gate_byte(base + 2u, byte) >> 6u) << 4u);
                up_scale = (up_byte(base + 3u, byte) & 15u) | ((up_byte(base + 1u, byte) >> 6u) << 4u);
                up_minimum = (up_byte(base + 3u, byte) >> 4u) | ((up_byte(base + 2u, byte) >> 6u) << 4u);
            }
            let gate_low_byte = gate_byte(base + 12u + slot * 8u + index / 4u, index % 4u);
            let up_low_byte = up_byte(base + 12u + slot * 8u + index / 4u, index % 4u);
            let gate_low = select(gate_low_byte & 15u, gate_low_byte >> 4u, first_q_group == 1u);
            let up_low = select(up_low_byte & 15u, up_low_byte >> 4u, first_q_group == 1u);
            let gate_high = gate_byte(base + 4u + index / 4u, index % 4u);
            let up_high = up_byte(base + 4u + index / 4u, index % 4u);
            let gate_quant = gate_low + select(0u, 16u, (gate_high & (1u << q_group)) != 0u);
            let up_quant = up_low + select(0u, 16u, (up_high & (1u << q_group)) != 0u);
            let column = block * 256u + local.x + slot * 64u;
            let x = input[group.y * params.cols + column];
            gate_sum += (gate_factors.x * f32(gate_scale * gate_quant) - gate_factors.y * f32(gate_minimum)) * x;
            up_sum += (up_factors.x * f32(up_scale * up_quant) - up_factors.y * f32(up_minimum)) * x;
            slot += 1u;
        }
        block += 1u;
    }
    gate_partial[local.x] = gate_sum; up_partial[local.x] = up_sum;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if local.x < stride { gate_partial[local.x] += gate_partial[local.x + stride]; up_partial[local.x] += up_partial[local.x + stride]; }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    if local.x == 0u {
        let gate = gate_partial[0];
        output[group.y * params.rows + row] = gate / (1.0 + exp(-gate)) * up_partial[0];
    }
}
"#;

/// Adreno prefill：一个 64-lane subgroup 解码一次权重，同时消费 4 行输入。
pub const PREFILL_SHADER: &str = r#"
struct Params { rows: u32, cols: u32, input_rows: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn byte_at(word: u32, byte: u32) -> u32 {
    return (weights[word] >> (byte * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let token_base = group.y * 4u;
    let blocks = params.cols / 256u;
    let index = local.x % 32u;
    let first_q_group = local.x / 32u;
    var sums: array<f32, 4>;
    for (var block = 0u; block < blocks; block += 1u) {
        let base = (row * blocks + block) * 44u;
        let factors = unpack2x16float(weights[base]);
        for (var slot = 0u; slot < 4u; slot += 1u) {
            let q_group = first_q_group + slot * 2u;
            var scale: u32;
            var minimum: u32;
            if q_group < 4u {
                scale = byte_at(base + 1u, q_group) & 63u;
                minimum = byte_at(base + 2u, q_group) & 63u;
            } else {
                let byte = q_group - 4u;
                scale = (byte_at(base + 3u, byte) & 15u) | ((byte_at(base + 1u, byte) >> 6u) << 4u);
                minimum = (byte_at(base + 3u, byte) >> 4u) | ((byte_at(base + 2u, byte) >> 6u) << 4u);
            }
            let low_byte = byte_at(base + 12u + slot * 8u + index / 4u, index % 4u);
            let low = select(low_byte & 15u, low_byte >> 4u, first_q_group == 1u);
            let high_byte = byte_at(base + 4u + index / 4u, index % 4u);
            let quant = low + select(0u, 16u, (high_byte & (1u << q_group)) != 0u);
            let value = factors.x * f32(scale * quant) - factors.y * f32(minimum);
            let column = block * 256u + local.x + slot * 64u;
            for (var token = 0u; token < 4u; token += 1u) {
                if token_base + token < params.input_rows {
                    sums[token] += value * input[(token_base + token) * params.cols + column];
                }
            }
        }
    }
    for (var token = 0u; token < 4u; token += 1u) {
        sums[token] = subgroupAdd(sums[token]);
    }
    if local.x == 0u {
        for (var token = 0u; token < 4u; token += 1u) {
            if token_base + token < params.input_rows {
                output[(token_base + token) * params.rows + row] = sums[token];
            }
        }
    }
}
"#;

pub const PREFILL_GATED_SHADER: &str = r#"
struct Params { rows: u32, cols: u32, input_rows: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> gate_weights: array<u32>;
@group(0) @binding(1) var<storage, read> up_weights: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn gate_byte(word: u32, byte: u32) -> u32 { return (gate_weights[word] >> (byte * 8u)) & 255u; }
fn up_byte(word: u32, byte: u32) -> u32 { return (up_weights[word] >> (byte * 8u)) & 255u; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let token_base = group.y * 4u;
    let blocks = params.cols / 256u;
    let index = local.x % 32u;
    let first_q_group = local.x / 32u;
    var gate_sums: array<f32, 4>;
    var up_sums: array<f32, 4>;
    for (var block = 0u; block < blocks; block += 1u) {
        let base = (row * blocks + block) * 44u;
        let gate_factors = unpack2x16float(gate_weights[base]);
        let up_factors = unpack2x16float(up_weights[base]);
        for (var slot = 0u; slot < 4u; slot += 1u) {
            let q_group = first_q_group + slot * 2u;
            var gate_scale: u32; var gate_minimum: u32;
            var up_scale: u32; var up_minimum: u32;
            if q_group < 4u {
                gate_scale = gate_byte(base + 1u, q_group) & 63u; gate_minimum = gate_byte(base + 2u, q_group) & 63u;
                up_scale = up_byte(base + 1u, q_group) & 63u; up_minimum = up_byte(base + 2u, q_group) & 63u;
            } else {
                let byte = q_group - 4u;
                gate_scale = (gate_byte(base + 3u, byte) & 15u) | ((gate_byte(base + 1u, byte) >> 6u) << 4u);
                gate_minimum = (gate_byte(base + 3u, byte) >> 4u) | ((gate_byte(base + 2u, byte) >> 6u) << 4u);
                up_scale = (up_byte(base + 3u, byte) & 15u) | ((up_byte(base + 1u, byte) >> 6u) << 4u);
                up_minimum = (up_byte(base + 3u, byte) >> 4u) | ((up_byte(base + 2u, byte) >> 6u) << 4u);
            }
            let gate_low_byte = gate_byte(base + 12u + slot * 8u + index / 4u, index % 4u);
            let up_low_byte = up_byte(base + 12u + slot * 8u + index / 4u, index % 4u);
            let gate_low = select(gate_low_byte & 15u, gate_low_byte >> 4u, first_q_group == 1u);
            let up_low = select(up_low_byte & 15u, up_low_byte >> 4u, first_q_group == 1u);
            let gate_high = gate_byte(base + 4u + index / 4u, index % 4u);
            let up_high = up_byte(base + 4u + index / 4u, index % 4u);
            let gate_quant = gate_low + select(0u, 16u, (gate_high & (1u << q_group)) != 0u);
            let up_quant = up_low + select(0u, 16u, (up_high & (1u << q_group)) != 0u);
            let gate_value = gate_factors.x * f32(gate_scale * gate_quant) - gate_factors.y * f32(gate_minimum);
            let up_value = up_factors.x * f32(up_scale * up_quant) - up_factors.y * f32(up_minimum);
            let column = block * 256u + local.x + slot * 64u;
            for (var token = 0u; token < 4u; token += 1u) {
                if token_base + token < params.input_rows {
                    let x = input[(token_base + token) * params.cols + column];
                    gate_sums[token] += gate_value * x;
                    up_sums[token] += up_value * x;
                }
            }
        }
    }
    for (var token = 0u; token < 4u; token += 1u) {
        gate_sums[token] = subgroupAdd(gate_sums[token]);
        up_sums[token] = subgroupAdd(up_sums[token]);
    }
    if local.x == 0u {
        for (var token = 0u; token < 4u; token += 1u) {
            if token_base + token < params.input_rows {
                let gate = gate_sums[token];
                output[(token_base + token) * params.rows + row] = gate / (1.0 + exp(-gate)) * up_sums[token];
            }
        }
    }
}
"#;

pub const GEMM_SHADER: &str = r#"
struct Params { rows: u32, cols: u32, input_rows: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn word_at(row_group: u32, block: u32, word: u32, row_lane: u32, blocks: u32) -> u32 {
    return weights[((row_group * blocks + block) * 44u + word) * 8u + row_lane];
}
fn byte_at(row_group: u32, block: u32, word: u32, byte: u32, row_lane: u32, blocks: u32) -> u32 {
    return (word_at(row_group, block, word, row_lane, blocks) >> (byte * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row_group = group.x;
    let token_base = group.y * 8u;
    let row_lane = local.x & 7u;
    let k_lane = local.x >> 3u;
    let row = row_group * 8u + row_lane;
    let blocks = params.cols / 256u;
    var sums: array<f32, 8>;
    for (var block = 0u; block < blocks; block += 1u) {
        let factors = unpack2x16float(word_at(row_group, block, 0u, row_lane, blocks));
        for (var q_group = 0u; q_group < 8u; q_group += 1u) {
            var scale: u32;
            var minimum: u32;
            if q_group < 4u {
                scale = byte_at(row_group, block, 1u, q_group, row_lane, blocks) & 63u;
                minimum = byte_at(row_group, block, 2u, q_group, row_lane, blocks) & 63u;
            } else {
                let byte = q_group - 4u;
                scale = (byte_at(row_group, block, 3u, byte, row_lane, blocks) & 15u) | ((byte_at(row_group, block, 1u, byte, row_lane, blocks) >> 6u) << 4u);
                minimum = (byte_at(row_group, block, 3u, byte, row_lane, blocks) >> 4u) | ((byte_at(row_group, block, 2u, byte, row_lane, blocks) >> 6u) << 4u);
            }
            for (var part = 0u; part < 4u; part += 1u) {
                let index = k_lane + part * 8u;
                let low_byte = byte_at(row_group, block, 12u + (q_group / 2u) * 8u + index / 4u, index % 4u, row_lane, blocks);
                let low = select(low_byte & 15u, low_byte >> 4u, (q_group & 1u) != 0u);
                let high_byte = byte_at(row_group, block, 4u + index / 4u, index % 4u, row_lane, blocks);
                let quant = low + select(0u, 16u, (high_byte & (1u << q_group)) != 0u);
                let value = factors.x * f32(scale * quant) - factors.y * f32(minimum);
                let column = block * 256u + q_group * 32u + index;
                for (var token = 0u; token < 8u; token += 1u) {
                    if token_base + token < params.input_rows {
                        sums[token] += value * input[(token_base + token) * params.cols + column];
                    }
                }
            }
        }
    }
    for (var token = 0u; token < 8u; token += 1u) {
        sums[token] += subgroupShuffleXor(sums[token], 8u);
        sums[token] += subgroupShuffleXor(sums[token], 16u);
        sums[token] += subgroupShuffleXor(sums[token], 32u);
    }
    if local.x < 8u && row < params.rows {
        for (var token = 0u; token < 8u; token += 1u) {
            if token_base + token < params.input_rows {
                output[(token_base + token) * params.rows + row] = sums[token];
            }
        }
    }
}
"#;

pub const Q8_1_SHADER: &str = r#"
requires packed_4x8_integer_dot_product;
struct Params { rows: u32, cols: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input_codes: array<u32>;
@group(0) @binding(2) var<storage, read> input_scales: array<f32>;
@group(0) @binding(3) var<storage, read> input_sums: array<i32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
var<workgroup> partial: array<f32, 32>;

fn byte_at(word: u32, byte: u32) -> u32 { return (weights[word] >> (byte * 8u)) & 255u; }

fn packed_weight(base: u32, q_group: u32, word: u32) -> u32 {
    let lows = weights[base + 12u + (q_group / 2u) * 8u + word];
    let highs = weights[base + 4u + word];
    var packed = 0u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
        let low_byte = (lows >> (lane * 8u)) & 255u;
        let low = select(low_byte & 15u, low_byte >> 4u, (q_group & 1u) != 0u);
        let high_byte = (highs >> (lane * 8u)) & 255u;
        let quant = i32(low + select(0u, 16u, (high_byte & (1u << q_group)) != 0u)) - 16;
        packed |= (u32(quant) & 255u) << (lane * 8u);
    }
    return packed;
}

fn group_scale(base: u32, q_group: u32) -> u32 {
    if q_group < 4u { return byte_at(base + 1u, q_group) & 63u; }
    let byte = q_group - 4u;
    return (byte_at(base + 3u, byte) & 15u) | ((byte_at(base + 1u, byte) >> 6u) << 4u);
}

fn group_minimum(base: u32, q_group: u32) -> u32 {
    if q_group < 4u { return byte_at(base + 2u, q_group) & 63u; }
    let byte = q_group - 4u;
    return (byte_at(base + 3u, byte) >> 4u) | ((byte_at(base + 2u, byte) >> 6u) << 4u);
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = group.x;
    let input_row = group.y;
    let local = local_id.x;
    let word = local & 7u;
    let first_group = local / 8u;
    let blocks = params.cols / 256u;
    let input_groups_per_row = params.cols / 32u;
    var sum = 0.0;
    for (var block = 0u; block < blocks; block += 1u) {
        let base = (row * blocks + block) * 44u;
        let factors = unpack2x16float(weights[base]);
        for (var half = 0u; half < 2u; half += 1u) {
            let q_group = first_group + half * 4u;
            let input_group = input_row * input_groups_per_row + block * 8u + q_group;
            let input_word = input_codes[input_group * 8u + word];
            let dot = dot4I8Packed(packed_weight(base, q_group, word), input_word);
            let scale = f32(group_scale(base, q_group));
            let minimum = f32(group_minimum(base, q_group));
            let input_sum = input_sums[input_group];
            // dot 由 8 个线程分别覆盖 4 列；整组的中心偏移与 minimum 修正只累加一次。
            let correction = select(0, input_sum, word == 0u);
            sum += input_scales[input_group] * (factors.x * scale * f32(dot + 16 * correction) - factors.y * minimum * f32(correction));
        }
    }
    partial[local] = sum;
    workgroupBarrier();
    var stride = 16u;
    loop {
        if local < stride { partial[local] += partial[local + stride]; }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    if local == 0u { output[input_row * params.rows + row] = partial[0]; }
}
"#;
