use super::super::BackendError;

pub const BLOCK_BYTES: usize = 136;
const BLOCK_VALUES: usize = 256;
pub const PREFILL_TOKEN_TILE: usize = 4;

pub fn validate_weight(packed: &[u8], rows: usize, cols: usize) -> Result<(), BackendError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(BLOCK_VALUES) {
        return Err(BackendError::Compute { msg: format!("IQ4_XS weight shape 无效: rows={rows}, cols={cols}") });
    }
    let expected = rows * (cols / BLOCK_VALUES) * BLOCK_BYTES;
    if packed.len() != expected {
        return Err(BackendError::Compute { msg: format!("IQ4_XS 权重字节数错误: 期望 {expected}, 实际 {}", packed.len()) });
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

const IQ4_NL = array<i32, 16>(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

fn byte_at(word: u32, byte: u32) -> u32 {
    return (weights[word] >> (byte * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let blocks = params.cols / 256u;
    let index = local.x & 31u;
    let first_sub_block = local.x >> 5u;
    var sum = 0.0;
    var block = 0u;
    loop {
        if block >= blocks { break; }
        let base = (row * blocks + block) * 34u;
        let header = weights[base];
        let d = unpack2x16float(header).x;
        let scales_h = header >> 16u;
        var slot = 0u;
        loop {
            if slot >= 4u { break; }
            let sub_block = first_sub_block + slot * 2u;
            let scales_l = byte_at(base + 1u, sub_block >> 1u);
            let low = select(scales_l & 15u, scales_l >> 4u, (sub_block & 1u) != 0u);
            let scale_code = low | (((scales_h >> (2u * sub_block)) & 3u) << 4u);
            let quant_index = index & 15u;
            let quant_byte = byte_at(base + 2u + sub_block * 4u + quant_index / 4u, quant_index & 3u);
            let quant = select(quant_byte & 15u, quant_byte >> 4u, index >= 16u);
            let column = block * 256u + sub_block * 32u + index;
            let value = d * (f32(scale_code) - 32.0) * f32(IQ4_NL[quant]);
            sum += value * input[group.y * params.cols + column];
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

const IQ4_NL = array<i32, 16>(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

fn gate_byte(word: u32, byte: u32) -> u32 { return (gate_weights[word] >> (byte * 8u)) & 255u; }
fn up_byte(word: u32, byte: u32) -> u32 { return (up_weights[word] >> (byte * 8u)) & 255u; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let blocks = params.cols / 256u;
    let index = local.x & 31u;
    let first_sub_block = local.x >> 5u;
    var gate_sum = 0.0;
    var up_sum = 0.0;
    var block = 0u;
    loop {
        if block >= blocks { break; }
        let base = (row * blocks + block) * 34u;
        let gate_header = gate_weights[base];
        let up_header = up_weights[base];
        let gate_d = unpack2x16float(gate_header).x;
        let up_d = unpack2x16float(up_header).x;
        let gate_scales_h = gate_header >> 16u;
        let up_scales_h = up_header >> 16u;
        var slot = 0u;
        loop {
            if slot >= 4u { break; }
            let sub_block = first_sub_block + slot * 2u;
            let gate_scales_l = gate_byte(base + 1u, sub_block >> 1u);
            let up_scales_l = up_byte(base + 1u, sub_block >> 1u);
            let gate_low = select(gate_scales_l & 15u, gate_scales_l >> 4u, (sub_block & 1u) != 0u);
            let up_low = select(up_scales_l & 15u, up_scales_l >> 4u, (sub_block & 1u) != 0u);
            let gate_scale = gate_low | (((gate_scales_h >> (2u * sub_block)) & 3u) << 4u);
            let up_scale = up_low | (((up_scales_h >> (2u * sub_block)) & 3u) << 4u);
            let quant_index = index & 15u;
            let gate_quant_byte = gate_byte(base + 2u + sub_block * 4u + quant_index / 4u, quant_index & 3u);
            let up_quant_byte = up_byte(base + 2u + sub_block * 4u + quant_index / 4u, quant_index & 3u);
            let gate_quant = select(gate_quant_byte & 15u, gate_quant_byte >> 4u, index >= 16u);
            let up_quant = select(up_quant_byte & 15u, up_quant_byte >> 4u, index >= 16u);
            let column = block * 256u + sub_block * 32u + index;
            let x = input[group.y * params.cols + column];
            gate_sum += gate_d * (f32(gate_scale) - 32.0) * f32(IQ4_NL[gate_quant]) * x;
            up_sum += up_d * (f32(up_scale) - 32.0) * f32(IQ4_NL[up_quant]) * x;
            slot += 1u;
        }
        block += 1u;
    }
    gate_partial[local.x] = gate_sum;
    up_partial[local.x] = up_sum;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if local.x < stride {
            gate_partial[local.x] += gate_partial[local.x + stride];
            up_partial[local.x] += up_partial[local.x + stride];
        }
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

const IQ4_NL = array<i32, 16>(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

fn byte_at(word: u32, byte: u32) -> u32 {
    return (weights[word] >> (byte * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let token_base = group.y * 4u;
    let blocks = params.cols / 256u;
    let index = local.x & 31u;
    let first_sub_block = local.x >> 5u;
    var sums: array<f32, 4>;
    for (var block = 0u; block < blocks; block += 1u) {
        let base = (row * blocks + block) * 34u;
        let header = weights[base];
        let d = unpack2x16float(header).x;
        let scales_h = header >> 16u;
        for (var slot = 0u; slot < 4u; slot += 1u) {
            let sub_block = first_sub_block + slot * 2u;
            let scales_l = byte_at(base + 1u, sub_block >> 1u);
            let low = select(scales_l & 15u, scales_l >> 4u, (sub_block & 1u) != 0u);
            let scale_code = low | (((scales_h >> (2u * sub_block)) & 3u) << 4u);
            let quant_index = index & 15u;
            let quant_byte = byte_at(base + 2u + sub_block * 4u + quant_index / 4u, quant_index & 3u);
            let quant = select(quant_byte & 15u, quant_byte >> 4u, index >= 16u);
            let column = block * 256u + sub_block * 32u + index;
            let value = d * (f32(scale_code) - 32.0) * f32(IQ4_NL[quant]);
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

const IQ4_NL = array<i32, 16>(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

fn gate_byte(word: u32, byte: u32) -> u32 { return (gate_weights[word] >> (byte * 8u)) & 255u; }
fn up_byte(word: u32, byte: u32) -> u32 { return (up_weights[word] >> (byte * 8u)) & 255u; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let token_base = group.y * 4u;
    let blocks = params.cols / 256u;
    let index = local.x & 31u;
    let first_sub_block = local.x >> 5u;
    var gate_sums: array<f32, 4>;
    var up_sums: array<f32, 4>;
    for (var block = 0u; block < blocks; block += 1u) {
        let base = (row * blocks + block) * 34u;
        let gate_header = gate_weights[base];
        let up_header = up_weights[base];
        let gate_d = unpack2x16float(gate_header).x;
        let up_d = unpack2x16float(up_header).x;
        let gate_scales_h = gate_header >> 16u;
        let up_scales_h = up_header >> 16u;
        for (var slot = 0u; slot < 4u; slot += 1u) {
            let sub_block = first_sub_block + slot * 2u;
            let gate_scales_l = gate_byte(base + 1u, sub_block >> 1u);
            let up_scales_l = up_byte(base + 1u, sub_block >> 1u);
            let gate_low = select(gate_scales_l & 15u, gate_scales_l >> 4u, (sub_block & 1u) != 0u);
            let up_low = select(up_scales_l & 15u, up_scales_l >> 4u, (sub_block & 1u) != 0u);
            let gate_scale = gate_low | (((gate_scales_h >> (2u * sub_block)) & 3u) << 4u);
            let up_scale = up_low | (((up_scales_h >> (2u * sub_block)) & 3u) << 4u);
            let quant_index = index & 15u;
            let gate_quant_byte = gate_byte(base + 2u + sub_block * 4u + quant_index / 4u, quant_index & 3u);
            let up_quant_byte = up_byte(base + 2u + sub_block * 4u + quant_index / 4u, quant_index & 3u);
            let gate_quant = select(gate_quant_byte & 15u, gate_quant_byte >> 4u, index >= 16u);
            let up_quant = select(up_quant_byte & 15u, up_quant_byte >> 4u, index >= 16u);
            let gate_value = gate_d * (f32(gate_scale) - 32.0) * f32(IQ4_NL[gate_quant]);
            let up_value = up_d * (f32(up_scale) - 32.0) * f32(IQ4_NL[up_quant]);
            let column = block * 256u + sub_block * 32u + index;
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

const IQ4_NL = array<i32, 16>(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

fn word_at(row_group: u32, block: u32, word: u32, row_lane: u32, blocks: u32) -> u32 {
    return weights[((row_group * blocks + block) * 34u + word) * 8u + row_lane];
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
        let header = word_at(row_group, block, 0u, row_lane, blocks);
        let d = unpack2x16float(header).x;
        let scales_h = header >> 16u;
        for (var sub_block = 0u; sub_block < 8u; sub_block += 1u) {
            let scales_l = byte_at(row_group, block, 1u, sub_block >> 1u, row_lane, blocks);
            let low = select(scales_l & 15u, scales_l >> 4u, (sub_block & 1u) != 0u);
            let scale_code = low | (((scales_h >> (2u * sub_block)) & 3u) << 4u);
            for (var part = 0u; part < 4u; part += 1u) {
                let index = k_lane + part * 8u;
                let quant_index = index & 15u;
                let quant_byte = byte_at(row_group, block, 2u + sub_block * 4u + quant_index / 4u, quant_index % 4u, row_lane, blocks);
                let quant = select(quant_byte & 15u, quant_byte >> 4u, index >= 16u);
                let value = d * (f32(scale_code) - 32.0) * f32(IQ4_NL[quant]);
                let column = block * 256u + sub_block * 32u + index;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_standard_iq4_xs_layout() {
        assert!(validate_weight(&vec![0; BLOCK_BYTES * 2], 2, 256).is_ok());
        assert!(validate_weight(&vec![0; BLOCK_BYTES], 1, 255).is_err());
        assert!(validate_weight(&vec![0; BLOCK_BYTES - 1], 1, 256).is_err());
    }
}
