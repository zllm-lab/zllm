pub const CONCAT_SHADER: &str = r#"
struct Params { rows: u32, left_cols: u32, right_cols: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_cols = params.left_cols + params.right_cols;
    let elements = params.rows * output_cols;
    if id.x >= elements { return; }
    let row = id.x / output_cols;
    let column = id.x % output_cols;
    if column < params.left_cols {
        output[id.x] = left[row * params.left_cols + column];
    } else {
        output[id.x] = right[row * params.right_cols + column - params.left_cols];
    }
}
"#;

pub const SLICE_SHADER: &str = r#"
struct Params { rows: u32, input_cols: u32, start: u32, output_cols: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let elements = params.rows * params.output_cols;
    if id.x >= elements { return; }
    let row = id.x / params.output_cols;
    let column = id.x % params.output_cols;
    output[id.x] = input[row * params.input_cols + params.start + column];
}
"#;

pub const INTERLEAVED_SHADER: &str = r#"
struct Params { rows: u32, input_cols: u32, block_cols: u32, parity: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_cols = params.input_cols / 2u;
    let elements = params.rows * output_cols;
    if id.x >= elements { return; }
    let row = id.x / output_cols;
    let column = id.x % output_cols;
    let block = column / params.block_cols;
    let inside = column % params.block_cols;
    let source = (block * 2u + params.parity) * params.block_cols + inside;
    output[id.x] = input[row * params.input_cols + source];
}
"#;
