pub const GROUP_SIZE: usize = 32;

pub const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> sums: array<i32>;
var<workgroup> maximums: array<f32, 32>;
var<workgroup> quantized: array<i32, 32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let local = local_id.x;
    let base = group.x * 32u;
    maximums[local] = abs(input[base + local]);
    workgroupBarrier();
    var stride = 16u;
    loop {
        if local < stride { maximums[local] = max(maximums[local], maximums[local + stride]); }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    let scale = max(maximums[0] / 127.0, 1.0e-12);
    quantized[local] = i32(clamp(round(input[base + local] / scale), -127.0, 127.0));
    workgroupBarrier();
    if local < 8u {
        var packed = 0u;
        for (var lane = 0u; lane < 4u; lane += 1u) {
            packed |= (u32(quantized[local * 4u + lane]) & 255u) << (lane * 8u);
        }
        codes[group.x * 8u + local] = packed;
    }
    stride = 16u;
    loop {
        if local < stride { quantized[local] += quantized[local + stride]; }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    if local == 0u { scales[group.x] = scale; sums[group.x] = quantized[0]; }
}
"#;
