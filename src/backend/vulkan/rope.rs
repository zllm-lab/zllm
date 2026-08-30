pub const SHADER: &str = r#"
struct Params { rows: u32, cols: u32, heads: u32, rotary_layout: u32, rotary_dim: u32, position: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> cosine: array<f32>;
@group(0) @binding(2) var<storage, read> sine: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let elements = params.rows * params.cols;
    if id.x >= elements { return; }
    let row = id.x / params.cols;
    let column = id.x % params.cols;
    let head_dim = params.cols / params.heads;
    let inside = column % head_dim;
    if inside >= params.rotary_dim {
        output[id.x] = input[id.x];
        return;
    }
    let half = params.rotary_dim / 2u;
    var pair: u32;
    var imaginary: bool;
    var real_inside: u32;
    var imaginary_inside: u32;
    if params.rotary_layout == 0u {
        pair = inside % half;
        imaginary = inside >= half;
        real_inside = pair;
        imaginary_inside = half + pair;
    } else {
        pair = inside / 2u;
        imaginary = (inside & 1u) != 0u;
        real_inside = pair * 2u;
        imaginary_inside = real_inside + 1u;
    }
    let head_start = id.x - inside;
    let real = input[head_start + real_inside];
    let imag = input[head_start + imaginary_inside];
    let table = (params.position + row) * half + pair;
    let c = cosine[table];
    let s = sine[table];
    output[id.x] = select(real * c - imag * s, imag * c + real * s, imaginary);
}
"#;
