pub const MIN_TOKEN_ROWS: usize = 16;
pub const TOKEN_TILE: usize = 8;
pub const ROW_TILE: usize = 8;

/// 大 chunk 临时重排：每个 u32 在 8 个相邻输出行之间交错，后续 8×8 GEMM
/// 可让相邻 lane 合并读取同一个量化字段。
pub const PACK_SHADER: &str = r#"
struct Params { rows: u32, words_per_row: u32, padded_rows: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= params.padded_rows * params.words_per_row { return; }
    let row = index / params.words_per_row;
    let word = index % params.words_per_row;
    let row_group = row / 8u;
    let row_lane = row % 8u;
    let destination = (row_group * params.words_per_row + word) * 8u + row_lane;
    if row < params.rows {
        packed[destination] = source[row * params.words_per_row + word];
    } else {
        packed[destination] = 0u;
    }
}
"#;
