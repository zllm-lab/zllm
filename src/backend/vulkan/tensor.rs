pub const RMSNORM_SHADER: &str = r#"
struct Params { cols: u32, eps: f32, add_one: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let offset = row * params.cols;
    var square_sum = 0.0;
    var column = local.x;
    loop {
        if column >= params.cols { break; }
        let value = input[offset + column];
        square_sum += value * value;
        column += 128u;
    }
    partial[local.x] = square_sum;
    workgroupBarrier();
    var stride = 64u;
    loop {
        if local.x < stride { partial[local.x] += partial[local.x + stride]; }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    let inverse_rms = inverseSqrt(partial[0] / f32(params.cols) + params.eps);
    column = local.x;
    loop {
        if column >= params.cols { break; }
        let gamma = weight[column] + select(0.0, 1.0, params.add_one != 0u);
        output[offset + column] = input[offset + column] * inverse_rms * gamma;
        column += 128u;
    }
}
"#;

pub const ADD_GEMMA_RMSNORM_SHADER: &str = r#"
struct Params { cols: u32, eps: f32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> normalized: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
var<workgroup> partial: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let offset = group.x * params.cols;
    var square_sum = 0.0;
    var column = local.x;
    loop {
        if column >= params.cols { break; }
        let value = left[offset + column] + right[offset + column];
        square_sum += value * value;
        column += 128u;
    }
    partial[local.x] = square_sum;
    workgroupBarrier();
    var stride = 64u;
    loop {
        if local.x < stride { partial[local.x] += partial[local.x + stride]; }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    let inverse_rms = inverseSqrt(partial[0] / f32(params.cols) + params.eps);
    column = local.x;
    loop {
        if column >= params.cols { break; }
        let value = left[offset + column] + right[offset + column];
        residual[offset + column] = value;
        normalized[offset + column] = value * inverse_rms * (weight[column] + 1.0);
        column += 128u;
    }
}
"#;

pub const ELEMENTWISE_SHADER: &str = r#"
struct Params { elements: u32, value: f32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.elements { output[id.x] = left[id.x] + right[id.x]; }
}

@compute @workgroup_size(256)
fn add_scaled(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.elements { output[id.x] = (left[id.x] + right[id.x]) * params.value; }
}

@compute @workgroup_size(256)
fn sigmoid_gate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.elements { output[id.x] = left[id.x] / (1.0 + exp(-right[id.x])); }
}

@compute @workgroup_size(256)
fn silu_mul(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.elements {
        let gate = left[id.x];
        output[id.x] = gate / (1.0 + exp(-gate)) * right[id.x];
    }
}
"#;
