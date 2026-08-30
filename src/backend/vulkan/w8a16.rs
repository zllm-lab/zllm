//! W8A16 group-wise 对称量化 GEMV 与 embedding lookup。

use crate::backend::BackendError;
use crate::weight::format::quantization::ScaleDType;

pub fn validate_weight(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize) -> Result<(), BackendError> {
    if rows == 0 || cols == 0 || group_size == 0 || !cols.is_multiple_of(group_size) || scale_dtype != ScaleDType::Bf16 {
        return Err(BackendError::Compute { msg: format!("Vulkan W8A16 需要 BF16 scale 且 shape/group 有效: [{rows},{cols}] group={group_size} scale={scale_dtype:?}") });
    }
    let expected_packed = rows.checked_mul(cols).ok_or_else(|| BackendError::Compute { msg: "Vulkan W8A16 packed 大小溢出".to_owned() })?;
    let expected_scales = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(2)).ok_or_else(|| BackendError::Compute { msg: "Vulkan W8A16 scales 大小溢出".to_owned() })?;
    if packed.len() != expected_packed || scales.len() != expected_scales {
        return Err(BackendError::Compute { msg: format!("Vulkan W8A16 字节数 packed={}/{} scales={}/{}", packed.len(), expected_packed, scales.len(), expected_scales) });
    }
    Ok(())
}

pub const SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    group_size: u32,
    output_offset: u32,
    output_rows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> partial: array<f32, 64>;

fn code_at(index: u32) -> f32 {
    let code = (weights[index / 4u] >> ((index % 4u) * 8u)) & 255u;
    return f32(i32(code) - 128);
}

fn scale_at(index: u32) -> f32 {
    let word = scales[index / 2u];
    let bits = select(word & 65535u, word >> 16u, (index & 1u) == 1u);
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let groups_per_row = params.cols / params.group_size;
    var sum = 0.0;
    var weight_group = 0u;
    loop {
        if weight_group >= groups_per_row { break; }
        let scale = scale_at(row * groups_per_row + weight_group);
        let group_base = weight_group * params.group_size;
        var offset = local.x;
        loop {
            if offset >= params.group_size { break; }
            let column = group_base + offset;
            sum += code_at(row * params.cols + column) * scale * input[group.y * params.cols + column];
            offset += 64u;
        }
        weight_group += 1u;
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
        output[group.y * params.output_rows + params.output_offset + row] = partial[0];
    }
}
"#;

pub const EMBEDDING_SHADER: &str = r#"
struct Params { row: u32, cols: u32, group_size: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn code_at(index: u32) -> f32 {
    let code = (weights[index / 4u] >> ((index % 4u) * 8u)) & 255u;
    return f32(i32(code) - 128);
}

fn scale_at(index: u32) -> f32 {
    let word = scales[index / 2u];
    let bits = select(word & 65535u, word >> 16u, (index & 1u) == 1u);
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let column = id.x;
    if column >= params.cols { return; }
    let groups_per_row = params.cols / params.group_size;
    output[column] = code_at(params.row * params.cols + column) * scale_at(params.row * groups_per_row + column / params.group_size);
}
"#;
