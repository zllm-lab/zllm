use super::super::BackendError;

pub const BLOCK_BYTES: usize = 210;
const BLOCK_VALUES: usize = 256;
pub const PREFILL_TOKEN_TILE: usize = 4;

pub fn validate_weight(packed: &[u8], rows: usize, cols: usize) -> Result<(), BackendError> {
    validate(packed, rows, cols, BLOCK_BYTES, BLOCK_VALUES, "Q6_K")
}

fn validate(packed: &[u8], rows: usize, cols: usize, block_bytes: usize, block_values: usize, name: &str) -> Result<(), BackendError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(block_values) {
        return Err(BackendError::Compute { msg: format!("{name} weight shape 无效: rows={rows}, cols={cols}") });
    }
    let expected = rows * (cols / block_values) * block_bytes;
    if packed.len() != expected {
        return Err(BackendError::Compute { msg: format!("{name} 权重字节数错误: 期望 {expected}, 实际 {}", packed.len()) });
    }
    Ok(())
}

pub const SHADER: &str = r#"
struct Params { rows: u32, cols: u32, output_offset: u32, output_stride: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> partial: array<f32, 64>;

fn byte_at(offset: u32) -> u32 {
    return (weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}

fn signed_byte(value: u32) -> i32 {
    return i32(value << 24u) >> 24;
}

fn half_at(offset: u32) -> f32 {
    let pair = unpack2x16float(weights[offset / 4u]);
    return select(pair.x, pair.y, (offset % 4u) == 2u);
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
        let base = (row * blocks + block) * 210u;
        let d = half_at(base + 208u);
        var slot = 0u;
        loop {
            if slot >= 4u { break; }
            let half = slot / 2u;
            let q_group = first_q_group + (slot % 2u) * 2u;
            let low_offset = base + half * 64u + index + select(0u, 32u, first_q_group == 1u);
            let low_byte = byte_at(low_offset);
            let low = select(low_byte & 15u, low_byte >> 4u, (slot & 1u) == 1u);
            let high_byte = byte_at(base + 128u + half * 32u + index);
            let quant = i32(low | (((high_byte >> (q_group * 2u)) & 3u) << 4u)) - 32;
            let scale_index = half * 8u + q_group * 2u + index / 16u;
            let scale = signed_byte(byte_at(base + 192u + scale_index));
            let column = block * 256u + local.x + slot * 64u;
            sum += d * f32(scale * quant) * input[group.y * params.cols + column];
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
    if local.x == 0u {
        output[group.y * params.output_stride + params.output_offset + row] = partial[0];
    }
}
"#;

/// Q6_K 只在 prefill 使用该路径；output offset/stride 继续支持大权重分块常驻。
pub const PREFILL_SHADER: &str = r#"
struct Params { input_rows: u32, cols: u32, output_offset: u32, output_stride: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn byte_at(offset: u32) -> u32 {
    return (weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}

fn signed_byte(value: u32) -> i32 {
    return i32(value << 24u) >> 24;
}

fn half_at(offset: u32) -> f32 {
    let pair = unpack2x16float(weights[offset / 4u]);
    return select(pair.x, pair.y, (offset % 4u) == 2u);
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
        let base = (row * blocks + block) * 210u;
        let d = half_at(base + 208u);
        for (var slot = 0u; slot < 4u; slot += 1u) {
            let half = slot / 2u;
            let q_group = first_q_group + (slot % 2u) * 2u;
            let low_offset = base + half * 64u + index + select(0u, 32u, first_q_group == 1u);
            let low_byte = byte_at(low_offset);
            let low = select(low_byte & 15u, low_byte >> 4u, (slot & 1u) == 1u);
            let high_byte = byte_at(base + 128u + half * 32u + index);
            let quant = i32(low | (((high_byte >> (q_group * 2u)) & 3u) << 4u)) - 32;
            let scale_index = half * 8u + q_group * 2u + index / 16u;
            let scale = signed_byte(byte_at(base + 192u + scale_index));
            let value = d * f32(scale * quant);
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
                output[(token_base + token) * params.output_stride + params.output_offset + row] = sums[token];
            }
        }
    }
}
"#;

/// Q6_K block 为 210 bytes，行边界不按 u32 对齐。每个 invocation 收集 4 个
/// 输出行的同一 byte，避免并发写同一个目标 word。
pub const PACK_GEMM_SHADER: &str = r#"
struct Params { rows: u32, bytes_per_row: u32, padded_rows: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn source_byte(offset: u32) -> u32 {
    return (source[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row_groups = params.padded_rows / 8u;
    let words = row_groups * params.bytes_per_row * 2u;
    if id.x >= words { return; }
    let half = id.x & 1u;
    let byte_index = (id.x >> 1u) % params.bytes_per_row;
    let row_group = (id.x >> 1u) / params.bytes_per_row;
    var value = 0u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
        let row = row_group * 8u + half * 4u + lane;
        if row < params.rows {
            value |= source_byte(row * params.bytes_per_row + byte_index) << (lane * 8u);
        }
    }
    packed[id.x] = value;
}
"#;

pub const GEMM_SHADER: &str = r#"
struct Params {
    rows: u32, cols: u32, input_rows: u32, output_offset: u32,
    output_stride: u32, _pad0: u32, _pad1: u32, _pad2: u32,
}
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn byte_at(row_group: u32, offset: u32, row_lane: u32, bytes_per_row: u32) -> u32 {
    let word = weights[(row_group * bytes_per_row + offset) * 2u + row_lane / 4u];
    return (word >> ((row_lane % 4u) * 8u)) & 255u;
}
fn signed_byte(value: u32) -> i32 { return i32(value << 24u) >> 24; }
fn half_at(row_group: u32, offset: u32, row_lane: u32, bytes_per_row: u32) -> f32 {
    let bits = byte_at(row_group, offset, row_lane, bytes_per_row) | (byte_at(row_group, offset + 1u, row_lane, bytes_per_row) << 8u);
    return unpack2x16float(bits).x;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row_group = group.x;
    let token_base = group.y * 8u;
    let row_lane = local.x & 7u;
    let k_lane = local.x >> 3u;
    let row = row_group * 8u + row_lane;
    let blocks = params.cols / 256u;
    let bytes_per_row = blocks * 210u;
    var sums: array<f32, 8>;
    for (var block = 0u; block < blocks; block += 1u) {
        let base = block * 210u;
        let d = half_at(row_group, base + 208u, row_lane, bytes_per_row);
        for (var half = 0u; half < 2u; half += 1u) {
            for (var q_group = 0u; q_group < 4u; q_group += 1u) {
                for (var part = 0u; part < 4u; part += 1u) {
                    let index = k_lane + part * 8u;
                    let low_offset = base + half * 64u + index + select(0u, 32u, (q_group & 1u) != 0u);
                    let low_byte = byte_at(row_group, low_offset, row_lane, bytes_per_row);
                    let low = select(low_byte & 15u, low_byte >> 4u, q_group >= 2u);
                    let high_byte = byte_at(row_group, base + 128u + half * 32u + index, row_lane, bytes_per_row);
                    let quant = i32(low | (((high_byte >> (q_group * 2u)) & 3u) << 4u)) - 32;
                    let scale_index = half * 8u + q_group * 2u + index / 16u;
                    let scale = signed_byte(byte_at(row_group, base + 192u + scale_index, row_lane, bytes_per_row));
                    let value = d * f32(scale * quant);
                    let column = block * 256u + half * 128u + q_group * 32u + index;
                    for (var token = 0u; token < 8u; token += 1u) {
                        if token_base + token < params.input_rows {
                            sums[token] += value * input[(token_base + token) * params.cols + column];
                        }
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
                output[(token_base + token) * params.output_stride + params.output_offset + row] = sums[token];
            }
        }
    }
}
"#;

pub const EMBEDDING_SHADER: &str = r#"
struct Params { row: u32, cols: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
fn byte_at(offset: u32) -> u32 { return (weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u; }
fn signed_byte(value: u32) -> i32 { return i32(value << 24u) >> 24; }
fn half_at(offset: u32) -> f32 {
    let pair = unpack2x16float(weights[offset / 4u]);
    return select(pair.x, pair.y, (offset % 4u) == 2u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let column = id.x;
    if column >= params.cols { return; }
    let blocks = params.cols / 256u;
    let block = column / 256u;
    let lane = column % 256u;
    let base = (params.row * blocks + block) * 210u;
    let half = lane / 128u;
    let position = lane % 128u;
    let q_group = position / 32u;
    let index = position % 32u;
    let low_offset = base + half * 64u + index + select(0u, 32u, (q_group & 1u) == 1u);
    let low_byte = byte_at(low_offset);
    let low = select(low_byte & 15u, low_byte >> 4u, q_group >= 2u);
    let high_byte = byte_at(base + 128u + half * 32u + index);
    let quant = i32(low | (((high_byte >> (q_group * 2u)) & 3u) << 4u)) - 32;
    let scale_index = half * 8u + q_group * 2u + index / 16u;
    let scale = signed_byte(byte_at(base + 192u + scale_index));
    output[column] = half_at(base + 208u) * f32(scale * quant);
}
"#;
