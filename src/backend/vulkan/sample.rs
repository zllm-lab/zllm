pub const ARGMAX_SHADER: &str = r#"
struct Params { elements: u32, excluded: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> excluded: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> values: array<f32, 256>;
var<workgroup> indices: array<u32, 256>;

fn allowed(index: u32) -> bool {
    for (var i = 0u; i < params.excluded; i += 1u) {
        if excluded[i] == index { return false; }
    }
    return true;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    var best_value = -3.402823466e+38;
    var best_index = 0xffffffffu;
    var index = local.x;
    loop {
        if index >= params.elements { break; }
        let value = logits[index];
        if allowed(index) && (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
        index += 256u;
    }
    values[local.x] = best_value;
    indices[local.x] = best_index;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if local.x < stride {
            let other_value = values[local.x + stride];
            let other_index = indices[local.x + stride];
            if other_value > values[local.x] || (other_value == values[local.x] && other_index < indices[local.x]) {
                values[local.x] = other_value;
                indices[local.x] = other_index;
            }
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    if local.x == 0u { output[0] = indices[0]; }
}
"#;
