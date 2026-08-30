pub const F32_SHADER: &str = r#"
struct Params { rows: u32, cols: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    var sum = 0.0;
    var column = local.x;
    loop {
        if column >= params.cols { break; }
        sum += weight[row * params.cols + column] * input[group.y * params.cols + column];
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
