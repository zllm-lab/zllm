use super::super::BackendError;

const BLOCK_BYTES: usize = 34;
const BLOCK_VALUES: usize = 32;

pub fn validate_weight(packed: &[u8], rows: usize, cols: usize) -> Result<(), BackendError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(BLOCK_VALUES) {
        return Err(BackendError::Compute { msg: format!("Q8_0 weight shape 无效: rows={rows}, cols={cols}") });
    }
    let expected = rows * (cols / BLOCK_VALUES) * BLOCK_BYTES;
    if packed.len() != expected {
        return Err(BackendError::Compute { msg: format!("Q8_0 权重字节数错误: 期望 {expected}, 实际 {}", packed.len()) });
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
    let blocks = params.cols / 32u;
    var sum = 0.0;
    var column = local.x;
    loop {
        if column >= params.cols { break; }
        let block = column / 32u;
        let lane = column % 32u;
        let base = (row * blocks + block) * 34u;
        let d = half_at(base);
        sum += d * f32(signed_byte(byte_at(base + 2u + lane))) * input[group.y * params.cols + column];
        column += 64u;
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
