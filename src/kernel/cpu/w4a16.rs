//! compressed-tensors W4A16 group-wise symmetric INT4 kernel(SIMD 主路 + 标量兜底)。
//! 同文件还包含对称的 W8A16:每 int32 打包 4 个 INT8(而非 8 个 INT4),
//! `read_w8a16_code` 取字节后减 128 恢复 `[-128, 127]`;scale 与 W4A16 完全一致。

use std::cell::RefCell;

use half::{bf16, f16};
use rayon::prelude::*;
use wide::f32x8;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub use crate::weight::codec::groupwise::decode_w4a16_matrix;
pub use crate::weight::codec::groupwise::decode_w8a16_matrix;
use crate::weight::format::quantization::ScaleDType;

use super::matmul::dot;

const SIMD_LANES: usize = 8;
#[cfg(target_arch = "x86_64")]
const VNNI_OUTPUT_TILE: usize = 16;
#[cfg(target_arch = "x86_64")]
const VNNI_INPUT_TILE: usize = 8;

/// AVX512-VNNI 专用 W8×A8 布局：每 4 个 K 元素交错 16 个输出行，
/// 使一条 VPDPBUSD 同时产生 16 个输出累加器。原始权重仍保持 unsigned
/// `q + 128`，内核用 `128 * sum(qx)` 精确消除零点项。
#[derive(Clone, Debug)]
pub struct W8A8VnniMatrix {
    #[cfg(target_arch = "x86_64")]
    packed: Vec<u8>,
    #[cfg(target_arch = "x86_64")]
    scales: Vec<f32>,
    group_size: usize,
    rows: usize,
    cols: usize,
}

impl W8A8VnniMatrix {
    #[cfg(target_arch = "x86_64")]
    pub fn try_repack(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize) -> Result<Option<Self>, String> {
        if group_size.is_multiple_of(4) && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("fma") {
            validate_groupwise("W8A16", 4, packed, scales, scale_dtype, group_size, rows, cols, rows.checked_mul(cols).ok_or_else(|| "W8A8 repack 大小溢出".to_owned())?)?;
            return Ok(Some(Self::repack(packed, scales, scale_dtype, group_size, rows, cols)?));
        }
        Ok(None)
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn try_repack(_packed: &[u8], _scales: &[u8], _scale_dtype: ScaleDType, _group_size: usize, _rows: usize, _cols: usize) -> Result<Option<Self>, String> {
        Ok(None)
    }

    #[cfg(target_arch = "x86_64")]
    fn repack(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize) -> Result<Self, String> {
        let row_tiles = rows.div_ceil(VNNI_OUTPUT_TILE);
        let groups = cols / group_size;
        let tile_bytes = cols.checked_mul(VNNI_OUTPUT_TILE).ok_or("W8A8 tile 大小溢出")?;
        let packed_bytes = row_tiles.checked_mul(tile_bytes).ok_or("W8A8 packed 大小溢出")?;
        // resident 权重逐块全量扫描:先建议 THP 再首次触碰,让大矩阵缺页
        // 直接落在 2MB 页,减少稳态扫描的 TLB 压力。
        let mut interleaved = Vec::<u8>::with_capacity(packed_bytes);
        super::advise_huge_pages_region(interleaved.as_mut_ptr().cast(), packed_bytes);
        interleaved.resize(packed_bytes, 128_u8);
        let source_row_bytes = cols.div_ceil(4) * 4;
        let threads = super::allowed_parallelism().min(16).min(row_tiles.max(1));
        let tiles_per_thread = row_tiles.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_index, destination) in interleaved.chunks_mut(tiles_per_thread * tile_bytes).enumerate() {
                let first_tile = chunk_index * tiles_per_thread;
                scope.spawn(move || {
                    for (local_tile, destination) in destination.chunks_mut(tile_bytes).enumerate() {
                        let tile = first_tile + local_tile;
                        for group in 0..groups {
                            for k4 in 0..group_size / 4 {
                                let output = (group * (group_size / 4) + k4) * VNNI_OUTPUT_TILE * 4;
                                let column = group * group_size + k4 * 4;
                                for lane in 0..VNNI_OUTPUT_TILE {
                                    let row = tile * VNNI_OUTPUT_TILE + lane;
                                    if row < rows {
                                        let source = row * source_row_bytes + column;
                                        destination[output + lane * 4..output + lane * 4 + 4].copy_from_slice(&packed[source..source + 4]);
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
        let scale_bytes = row_tiles * groups * VNNI_OUTPUT_TILE;
        let mut interleaved_scales = Vec::<f32>::with_capacity(scale_bytes);
        super::advise_huge_pages_region(interleaved_scales.as_mut_ptr().cast(), scale_bytes * std::mem::size_of::<f32>());
        interleaved_scales.resize(scale_bytes, 0.0_f32);
        for tile in 0..row_tiles {
            for group in 0..groups {
                for lane in 0..VNNI_OUTPUT_TILE {
                    let row = tile * VNNI_OUTPUT_TILE + lane;
                    if row < rows {
                        interleaved_scales[(tile * groups + group) * VNNI_OUTPUT_TILE + lane] = read_scale(scales, scale_dtype, row * groups + group);
                    }
                }
            }
        }
        Ok(Self { packed: interleaved, scales: interleaved_scales, group_size, rows, cols })
    }

    pub fn matmul(&self, n_inputs: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        if n_inputs == 0 || input.len() != n_inputs.saturating_mul(self.cols) || output.len() != n_inputs.saturating_mul(self.rows) {
            return Err(format!("W8A8 VNNI shape n_inputs={n_inputs} cols={} rows={}", self.cols, self.rows));
        }
        #[cfg(target_arch = "x86_64")]
        {
            let quantized = quantize_w8a8_input(input, n_inputs, self.cols, self.group_size)?;
            return self.matmul_quantized(n_inputs, &quantized, output);
        }
        #[cfg(not(target_arch = "x86_64"))]
        Err("W8A8 VNNI 仅支持 x86_64".to_owned())
    }

    /// 输入已按本矩阵的 cols/group 布局完成 A8 量化时直接复用;
    /// 共享同一 encoder 的 Q/K/V 与 gate/up 只量化一次。
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn matmul_quantized(&self, n_inputs: usize, input: &QuantizedW8A8Input, output: &mut [f32]) -> Result<(), String> {
        #[cfg(target_arch = "x86_64")]
        {
            let groups = self.cols / self.group_size;
            if n_inputs == 0 || input.values.len() != n_inputs.saturating_mul(self.cols) || input.scales.len() != n_inputs.saturating_mul(groups) || output.len() != n_inputs.saturating_mul(self.rows) {
                return Err(format!("W8A8 VNNI quantized shape n_inputs={n_inputs} cols={} rows={} groups={groups}", self.cols, self.rows));
            }
            let row_tiles = self.rows.div_ceil(VNNI_OUTPUT_TILE);
            // 每个任务至少包含两个 16-row tile；小矩阵避免创建比工作更多的线程。
            let threads = super::allowed_parallelism().min(16).min(row_tiles.div_ceil(2).max(1));
            let tiles_per_thread = row_tiles.div_ceil(threads);
            let packed_tile_bytes = self.cols * VNNI_OUTPUT_TILE;
            let scale_tile_elements = groups * VNNI_OUTPUT_TILE;
            // 固定 team 分发:直接写 [n_inputs, rows] 最终输出,不经中间
            // tiled 缓冲(大 vocab 头的逐次 20MB 分配与串行回拷是显著开销)。
            // 裸指针跨线程由 SendCell 声明,worker 各写独占的 tile 列段;
            // 大输出建议 THP,降低 worker 分散写与后续 argmax 读的 TLB 压力。
            let rows = self.rows;
            super::advise_huge_pages_region(output.as_mut_ptr().cast(), output.len() * std::mem::size_of::<f32>());
            let destination = super::team::SendCell(output.as_mut_ptr());
            super::team::team_execute(threads, move |thread| unsafe {
                let base = destination.get();
                let first_tile = thread * tiles_per_thread;
                let count = row_tiles.saturating_sub(first_tile).min(tiles_per_thread);
                if count == 0 {
                    return;
                }
                let output = std::slice::from_raw_parts_mut(base, n_inputs * rows);
                let packed = &self.packed[first_tile * packed_tile_bytes..(first_tile + count) * packed_tile_bytes];
                let scales = &self.scales[first_tile * scale_tile_elements..(first_tile + count) * scale_tile_elements];
                matmul_w8a8_vnni_tiles(packed, scales, input, self.group_size, self.cols, rows, n_inputs, first_tile, count, output);
            });
            return Ok(());
        }
        #[cfg(not(target_arch = "x86_64"))]
        Err("W8A8 VNNI 仅支持 x86_64".to_owned())
    }

    /// 供共享 encoder 的调用方判断多路权重能否复用同一份量化输入。
    pub fn quant_layout(&self) -> (usize, usize) {
        (self.cols, self.group_size)
    }

    /// 共享量化输入的多路矩阵一次 team 分发:把各矩阵的 output tile
    /// 合并成一个连续 work range,消除逐矩阵的唤醒与 barrier(Q/K/V、
    /// gate/up 各省 2/1 次)。tile 内计算与单矩阵调用逐位一致,输出
    /// bit-exact;tile 属于哪个 worker 不影响任何跨 tile 累加。
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn matmul_quantized_many(matrices: &[&Self], n_inputs: usize, input: &QuantizedW8A8Input, outputs: &mut [&mut [f32]]) -> Result<(), String> {
        #[cfg(target_arch = "x86_64")]
        {
            if matrices.is_empty() || matrices.len() != outputs.len() {
                return Err(format!("W8A8 VNNI many matrices={} outputs={}", matrices.len(), outputs.len()));
            }
            // (首 tile 全局编号, tile 数, 目标输出) 逐矩阵校验并汇总。
            let mut jobs = Vec::with_capacity(matrices.len());
            let mut destinations = Vec::with_capacity(matrices.len());
            let mut total_tiles = 0usize;
            for (matrix, output) in matrices.iter().zip(outputs.iter_mut()) {
                let groups = matrix.cols / matrix.group_size;
                if n_inputs == 0 || input.values.len() != n_inputs.saturating_mul(matrix.cols) || input.scales.len() != n_inputs.saturating_mul(groups) || output.len() != n_inputs.saturating_mul(matrix.rows) {
                    return Err(format!("W8A8 VNNI many shape n_inputs={n_inputs} cols={} rows={} groups={groups}", matrix.cols, matrix.rows));
                }
                let row_tiles = matrix.rows.div_ceil(VNNI_OUTPUT_TILE);
                jobs.push((total_tiles, row_tiles));
                destinations.push(super::team::SendCell(output.as_mut_ptr()));
                total_tiles += row_tiles;
            }
            let threads = super::allowed_parallelism().min(16).min(total_tiles.div_ceil(2).max(1));
            let tiles_per_thread = total_tiles.div_ceil(threads);
            super::team::team_execute(threads, move |thread| unsafe {
                let mut tile = thread * tiles_per_thread;
                let end = total_tiles.min(tile + tiles_per_thread);
                while tile < end {
                    // 定位全局 tile 所属矩阵;矩阵数量 ≤ 3,线性扫描即可。
                    let (index, local) = jobs.iter().enumerate().rev().find(|&(_, &(first, _))| tile >= first).map(|(index, &(first, _))| (index, tile - first)).expect("tile 总在某个矩阵内");
                    let matrix = matrices[index];
                    let row_tiles = jobs[index].1;
                    let count = (end - tile).min(row_tiles - local);
                    let packed_tile_bytes = matrix.cols * VNNI_OUTPUT_TILE;
                    let scale_tile_elements = (matrix.cols / matrix.group_size) * VNNI_OUTPUT_TILE;
                    let output = std::slice::from_raw_parts_mut(destinations[index].get(), n_inputs * matrix.rows);
                    let packed = &matrix.packed[local * packed_tile_bytes..(local + count) * packed_tile_bytes];
                    let scales = &matrix.scales[local * scale_tile_elements..(local + count) * scale_tile_elements];
                    matmul_w8a8_vnni_tiles(packed, scales, input, matrix.group_size, matrix.cols, matrix.rows, n_inputs, local, count, output);
                    tile += count;
                }
            });
            return Ok(());
        }
        #[cfg(not(target_arch = "x86_64"))]
        Err("W8A8 VNNI 仅支持 x86_64".to_owned())
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) struct QuantizedW8A8Input {
    values: Vec<i8>,
    scales: Vec<f32>,
    corrections: Vec<i32>,
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn quantize_w8a8_input(input: &[f32], rows: usize, cols: usize, group_size: usize) -> Result<QuantizedW8A8Input, String> {
    #[cfg(target_arch = "x86_64")]
    if group_size.is_multiple_of(16) && is_x86_feature_detected!("avx512f") {
        return unsafe { quantize_w8a8_input_avx512(input, rows, cols, group_size) };
    }
    quantize_w8a8_input_scalar(input, rows, cols, group_size)
}

/// AVX-512 版 A8 量化:除法保持与标量一致的 IEEE 结果,取整用最近偶
/// (仅精确半值与标量 half-away 不同;oracle 与内核共用本组函数的定义)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn quantize_w8a8_input_avx512(input: &[f32], rows: usize, cols: usize, group_size: usize) -> Result<QuantizedW8A8Input, String> {
    let groups = cols / group_size;
    let mut values = vec![0_i8; input.len()];
    let mut scales = vec![0.0_f32; rows * groups];
    let mut corrections = vec![0_i32; rows * groups];
    for row in 0..rows {
        for group in 0..groups {
            let begin = row * cols + group * group_size;
            let source = &input[begin..begin + group_size];
            let destination = unsafe { values.as_mut_ptr().add(begin) };
            let mut maximum = _mm512_setzero_ps();
            let mut unordered = 0_u16;
            for chunk in source.chunks_exact(16) {
                let value = unsafe { _mm512_loadu_ps(chunk.as_ptr()) };
                unordered |= !_mm512_cmp_ps_mask(value, value, _CMP_ORD_Q);
                maximum = _mm512_max_ps(maximum, _mm512_abs_ps(value));
            }
            let maximum = _mm512_reduce_max_ps(maximum);
            if unordered != 0 || !maximum.is_finite() {
                return Err(format!("W8A8 input row={row} group={group} 包含非有限值"));
            }
            let scale = if maximum > 0.0 { maximum / 127.0 } else { 1.0 };
            let scale = _mm512_set1_ps(scale);
            let lower = _mm512_set1_ps(-127.0);
            let upper = _mm512_set1_ps(127.0);
            let mut sum = _mm512_setzero_si512();
            for (index, chunk) in source.chunks_exact(16).enumerate() {
                let value = unsafe { _mm512_loadu_ps(chunk.as_ptr()) };
                // cvtps_epi32 自带最近偶取整,与标量 round 仅在精确半值处不同。
                let code = _mm512_div_ps(value, scale);
                let code = _mm512_min_ps(_mm512_max_ps(code, lower), upper);
                let integers = _mm512_cvtps_epi32(code);
                sum = _mm512_add_epi32(sum, integers);
                let packed = _mm512_cvtsepi32_epi8(integers);
                unsafe { _mm_storeu_si128(destination.add(index * 16).cast(), packed) };
            }
            scales[row * groups + group] = if maximum > 0.0 { maximum / 127.0 } else { 1.0 };
            corrections[row * groups + group] = 128 * _mm512_reduce_add_epi32(sum);
        }
    }
    Ok(QuantizedW8A8Input { values, scales, corrections })
}

#[cfg(target_arch = "x86_64")]
fn quantize_w8a8_input_scalar(input: &[f32], rows: usize, cols: usize, group_size: usize) -> Result<QuantizedW8A8Input, String> {
    let groups = cols / group_size;
    let mut values = vec![0_i8; input.len()];
    let mut scales = vec![0.0_f32; rows * groups];
    let mut corrections = vec![0_i32; rows * groups];
    for row in 0..rows {
        for group in 0..groups {
            let begin = row * cols + group * group_size;
            let source = &input[begin..begin + group_size];
            let mut maximum = 0.0_f32;
            for &value in source {
                if !value.is_finite() {
                    return Err(format!("W8A8 input row={row} group={group} 包含非有限值"));
                }
                maximum = maximum.max(value.abs());
            }
            let scale = if maximum > 0.0 { maximum / 127.0 } else { 1.0 };
            let mut sum = 0_i32;
            for (destination, &value) in values[begin..begin + group_size].iter_mut().zip(source) {
                let code = (value / scale).round().clamp(-127.0, 127.0) as i8;
                *destination = code;
                sum += i32::from(code);
            }
            scales[row * groups + group] = scale;
            corrections[row * groups + group] = 128 * sum;
        }
    }
    Ok(QuantizedW8A8Input { values, scales, corrections })
}

/// 直接写最终 `[n_inputs, rows]` 输出布局;`first_tile..first_tile+row_tiles`
/// 是本 worker 独占的输出列段,行尾残块用 masked store。
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,fma")]
unsafe fn matmul_w8a8_vnni_tiles(packed: &[u8], weight_scales: &[f32], input: &QuantizedW8A8Input, group_size: usize, cols: usize, rows: usize, n_inputs: usize, first_tile: usize, row_tiles: usize, output: &mut [f32]) {
    let groups = cols / group_size;
    let tile_bytes = cols * VNNI_OUTPUT_TILE;
    let out = output.as_mut_ptr();
    for tile in 0..row_tiles {
        let global_tile = first_tile + tile;
        let packed = unsafe { packed.as_ptr().add(tile * tile_bytes) };
        let weight_scales = unsafe { weight_scales.as_ptr().add(tile * groups * VNNI_OUTPUT_TILE) };
        for input_start in (0..n_inputs).step_by(VNNI_INPUT_TILE) {
            let count = VNNI_INPUT_TILE.min(n_inputs - input_start);
            let mut sums = [_mm512_setzero_ps(); VNNI_INPUT_TILE];
            for group in 0..groups {
                let mut dots = [_mm512_setzero_si512(); VNNI_INPUT_TILE];
                let packed_group = unsafe { packed.add(group * group_size * VNNI_OUTPUT_TILE) };
                for k4 in 0..group_size / 4 {
                    let weights = unsafe { _mm512_loadu_si512(packed_group.add(k4 * VNNI_OUTPUT_TILE * 4).cast()) };
                    for offset in 0..count {
                        let input_row = input_start + offset;
                        let begin = input_row * cols + group * group_size + k4 * 4;
                        let bytes = unsafe { std::slice::from_raw_parts(input.values.as_ptr().add(begin).cast::<u8>(), 4) };
                        let activation = _mm512_set1_epi32(i32::from_le_bytes(bytes.try_into().expect("W8A8 activation x4")));
                        dots[offset] = _mm512_dpbusd_epi32(dots[offset], weights, activation);
                    }
                }
                let scales = unsafe { _mm512_loadu_ps(weight_scales.add(group * VNNI_OUTPUT_TILE)) };
                for offset in 0..count {
                    let input_row = input_start + offset;
                    let correction = _mm512_set1_epi32(input.corrections[input_row * groups + group]);
                    let dot = _mm512_cvtepi32_ps(_mm512_sub_epi32(dots[offset], correction));
                    let activation_scale = _mm512_set1_ps(input.scales[input_row * groups + group]);
                    sums[offset] = _mm512_fmadd_ps(dot, _mm512_mul_ps(scales, activation_scale), sums[offset]);
                }
            }
            let valid = VNNI_OUTPUT_TILE.min(rows - global_tile * VNNI_OUTPUT_TILE);
            for offset in 0..count {
                let destination = unsafe { out.add((input_start + offset) * rows + global_tile * VNNI_OUTPUT_TILE) };
                if valid == VNNI_OUTPUT_TILE {
                    unsafe { _mm512_storeu_ps(destination, sums[offset]) };
                } else {
                    let mask = ((1_u32 << valid) - 1) as u16;
                    unsafe { _mm512_mask_storeu_ps(destination, mask, sums[offset]) };
                }
            }
        }
    }
}

thread_local! {
    /// 批量 matmul 中复用的反量化行缓冲(每线程一个),避免逐行重复分配。
    static W_ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[allow(clippy::too_many_arguments)]
pub fn matvec_w4a16_matrix(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    validate_groupwise("W4A16", 8, packed, scales, scale_dtype, group_size, rows, cols, rows.checked_mul(cols).ok_or_else(|| "W4A16 大小溢出".to_owned())?)?;
    if input.len() != cols || output.len() != rows {
        return Err(format!("W4A16 GEMV input/output={}/{}，期望 {cols}/{rows}", input.len(), output.len()));
    }
    let packed_row_bytes = cols.div_ceil(8) * 4;
    let groups = cols / group_size;
    // group_size 是 SIMD_LANES 倍数时走 SIMD:每 int32 word 解出 8 个 nibble,
    // 一次性乘 scale 与 input。否则标量兜底,保持通用性。
    if group_size.is_multiple_of(SIMD_LANES) {
        output.par_iter_mut().enumerate().for_each(|(row, output)| {
            let packed_row = &packed[row * packed_row_bytes..(row + 1) * packed_row_bytes];
            let mut acc = f32x8::splat(0.0);
            let mut codes = [0.0f32; SIMD_LANES];
            for group in 0..groups {
                let scale = f32x8::splat(read_scale(scales, scale_dtype, row * groups + group));
                let start = group * group_size;
                for word in 0..group_size / SIMD_LANES {
                    let column = start + word * SIMD_LANES;
                    let offset = column / 8 * 4;
                    let word = u32::from_le_bytes(packed_row[offset..offset + 4].try_into().expect("W4A16 word"));
                    for lane in 0..SIMD_LANES {
                        codes[lane] = (((word >> (lane * 4)) & 0x0f) as i8 - 8) as f32;
                    }
                    let code = f32x8::from(codes);
                    let input_v = f32x8::from(<[f32; SIMD_LANES]>::try_from(&input[column..column + SIMD_LANES]).unwrap());
                    acc = (code * scale).mul_add(input_v, acc);
                }
            }
            *output = acc.reduce_add();
        });
    } else {
        output.par_iter_mut().enumerate().for_each(|(row, output)| {
            let packed_row = &packed[row * packed_row_bytes..(row + 1) * packed_row_bytes];
            let mut sum = 0.0;
            for group in 0..groups {
                let scale = read_scale(scales, scale_dtype, row * groups + group);
                let start = group * group_size;
                for column in start..start + group_size {
                    sum += f32::from(read_code(packed_row, column)) * scale * input[column];
                }
            }
            *output = sum;
        });
    }
    Ok(())
}

/// 批量 W4A16 矩阵乘:`output[n,row] = Σ_col dequant(row,col) · input[n,col]`。
///
/// `input` 为 `[n_inputs, cols]`,`output` 为 `[n_inputs, rows]`。prefill 多 token 时,
/// 每个权重行只反量化一次并复用于全部 n_inputs 个输入,避免逐 token 重读打包权重
/// (否则权重会被读 n_inputs 遍,前处理长 prompt 时成为带宽瓶颈)。
/// decode(n_inputs==1)仍走 `matvec_w4a16_matrix`,无中间缓冲。
#[allow(clippy::too_many_arguments)]
pub fn matmul_w4a16_matrix(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, n_inputs: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    validate_groupwise("W4A16", 8, packed, scales, scale_dtype, group_size, rows, cols, rows.checked_mul(cols).ok_or_else(|| "W4A16 大小溢出".to_owned())?)?;
    if n_inputs == 0 || input.len() != n_inputs.checked_mul(cols).ok_or_else(|| "W4A16 matmul input 溢出".to_owned())? || output.len() != n_inputs.checked_mul(rows).ok_or_else(|| "W4A16 matmul output 溢出".to_owned())? {
        return Err(format!("W4A16 matmul shape n_inputs={n_inputs} cols={cols} rows={rows}"));
    }
    // 中间结果按 [rows, n_inputs] 布局:并行 over rows,每行写入连续的 n_inputs 个元素。
    let mut transposed = vec![0.0f32; rows * n_inputs];
    let packed_row_bytes = cols.div_ceil(8) * 4;
    transposed.par_chunks_mut(n_inputs).enumerate().for_each(|(row, row_out)| {
        let packed_row = &packed[row * packed_row_bytes..(row + 1) * packed_row_bytes];
        W_ROW.with(|cell| {
            let w = &mut *cell.borrow_mut();
            w.resize(cols, 0.0);
            decode_row_w4a16(packed_row, scales, scale_dtype, group_size, row, cols, w);
            for n in 0..n_inputs {
                row_out[n] = dot(w, &input[n * cols..(n + 1) * cols]);
            }
        });
    });
    // 转回 [n_inputs, rows],供 backend 的行优先输出。
    for row in 0..rows {
        let src = row * n_inputs;
        for n in 0..n_inputs {
            output[n * rows + row] = transposed[src + n];
        }
    }
    Ok(())
}

/// 反量化单行权重到 `w`。group_size 是 SIMD_LANES 倍数时走 SIMD(每 word 解 8 nibble)。
fn decode_row_w4a16(packed_row: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, row: usize, cols: usize, w: &mut [f32]) {
    let groups = cols / group_size;
    if group_size.is_multiple_of(SIMD_LANES) {
        let mut codes = [0.0f32; SIMD_LANES];
        for group in 0..groups {
            let scale = f32x8::splat(read_scale(scales, scale_dtype, row * groups + group));
            let start = group * group_size;
            for word in 0..group_size / SIMD_LANES {
                let column = start + word * SIMD_LANES;
                let offset = column / 8 * 4;
                let word = u32::from_le_bytes(packed_row[offset..offset + 4].try_into().expect("W4A16 word"));
                for lane in 0..SIMD_LANES {
                    codes[lane] = (((word >> (lane * 4)) & 0x0f) as i8 - 8) as f32;
                }
                let scaled: [f32; SIMD_LANES] = (f32x8::from(codes) * scale).into();
                w[column..column + SIMD_LANES].copy_from_slice(&scaled);
            }
        }
    } else {
        for group in 0..groups {
            let scale = read_scale(scales, scale_dtype, row * groups + group);
            let start = group * group_size;
            for column in start..start + group_size {
                w[column] = f32::from(read_code(packed_row, column)) * scale;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_groupwise(kind: &str, values_per_word: usize, packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, output_elements: usize) -> Result<(), String> {
    if rows == 0 || cols == 0 || group_size == 0 || !cols.is_multiple_of(group_size) {
        return Err(format!("{kind} shape=[{rows},{cols}] group_size={group_size} 无效"));
    }
    let packed_bytes = rows.checked_mul(cols.div_ceil(values_per_word)).and_then(|words| words.checked_mul(4)).ok_or_else(|| format!("{kind} packed 大小溢出"))?;
    let scale_bytes = rows.checked_mul(cols / group_size).and_then(|count| count.checked_mul(scale_dtype.bytes())).ok_or_else(|| format!("{kind} scale 大小溢出"))?;
    let expected_output = rows.checked_mul(cols).ok_or_else(|| format!("{kind} output 大小溢出"))?;
    if packed.len() != packed_bytes || scales.len() != scale_bytes || output_elements != expected_output {
        return Err(format!("{kind} buffer packed={}/{packed_bytes} scales={}/{scale_bytes} output={output_elements}/{expected_output}", packed.len(), scales.len(),));
    }
    Ok(())
}

fn read_code(packed_row: &[u8], column: usize) -> i8 {
    let word_offset = column / 8 * 4;
    let word = u32::from_le_bytes(packed_row[word_offset..word_offset + 4].try_into().expect("W4A16 word"));
    (((word >> ((column % 8) * 4)) & 0x0f) as i8) - 8
}

fn read_scale(scales: &[u8], dtype: ScaleDType, index: usize) -> f32 {
    let offset = index * dtype.bytes();
    match dtype {
        ScaleDType::Bf16 => bf16::from_le_bytes(scales[offset..offset + 2].try_into().expect("BF16 scale")).to_f32(),
        ScaleDType::F16 => f16::from_le_bytes(scales[offset..offset + 2].try_into().expect("F16 scale")).to_f32(),
        ScaleDType::F32 => f32::from_le_bytes(scales[offset..offset + 4].try_into().expect("F32 scale")),
    }
}

/// W8A16 GEMV:`output[row] = sum_col w[row,col] * input[col]`,解码与累加融合。
/// `packed`:按行连续存放的 int32 数组,每 int32 存 4 个有符号 INT8。
#[allow(clippy::too_many_arguments)]
pub fn matvec_w8a16_matrix(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    validate_groupwise("W8A16", 4, packed, scales, scale_dtype, group_size, rows, cols, rows.checked_mul(cols).ok_or_else(|| "W8A16 大小溢出".to_owned())?)?;
    if input.len() != cols || output.len() != rows {
        return Err(format!("W8A16 GEMV input/output={}/{}，期望 {cols}/{rows}", input.len(), output.len()));
    }
    #[cfg(target_arch = "x86_64")]
    if group_size.is_multiple_of(16) && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        unsafe { matvec_w8a16_avx512(packed, scales, scale_dtype, group_size, rows, cols, input, output) };
        return Ok(());
    }
    let packed_row_bytes = cols.div_ceil(4) * 4;
    let groups = cols / group_size;
    output.par_iter_mut().enumerate().for_each(|(row, output)| {
        let packed = &packed[row * packed_row_bytes..(row + 1) * packed_row_bytes];
        let mut acc = f32x8::splat(0.0);
        let mut codes = [0.0f32; SIMD_LANES];
        for group in 0..groups {
            let scale = f32x8::splat(read_scale(scales, scale_dtype, row * groups + group));
            let start = group * group_size;
            for column in (start..start + group_size).step_by(SIMD_LANES) {
                for lane in 0..SIMD_LANES {
                    codes[lane] = f32::from(read_w8a16_code(packed, column + lane));
                }
                let weight = f32x8::from(codes) * scale;
                let input = f32x8::from(<[f32; SIMD_LANES]>::try_from(&input[column..column + SIMD_LANES]).unwrap());
                acc = weight.mul_add(input, acc);
            }
        }
        *output = acc.reduce_add();
    });
    Ok(())
}

/// W8A16 小行数矩阵乘。每个输出权重行同时服务最多 8 个输入行，
/// DSpark 的 block rows=8 时只扫描一次权重；多 session 再按 8 行分片。
#[allow(clippy::too_many_arguments)]
pub fn matmul_w8a16_matrix(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, n_inputs: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    validate_groupwise("W8A16", 4, packed, scales, scale_dtype, group_size, rows, cols, rows.checked_mul(cols).ok_or_else(|| "W8A16 大小溢出".to_owned())?)?;
    if n_inputs == 0 || input.len() != n_inputs.checked_mul(cols).ok_or_else(|| "W8A16 matmul input 溢出".to_owned())? || output.len() != n_inputs.checked_mul(rows).ok_or_else(|| "W8A16 matmul output 溢出".to_owned())? {
        return Err(format!("W8A16 matmul shape n_inputs={n_inputs} cols={cols} rows={rows}"));
    }
    #[cfg(target_arch = "x86_64")]
    if group_size.is_multiple_of(16) && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        unsafe { matmul_w8a16_avx512(packed, scales, scale_dtype, group_size, rows, cols, n_inputs, input, output) };
        return Ok(());
    }
    const INPUT_TILE: usize = 8;
    let packed_row_bytes = cols.div_ceil(4) * 4;
    let groups = cols / group_size;
    let mut transposed = vec![0.0f32; rows * n_inputs];
    transposed.par_chunks_mut(n_inputs).enumerate().for_each(|(row, row_out)| {
        let packed = &packed[row * packed_row_bytes..(row + 1) * packed_row_bytes];
        let mut codes = [0.0f32; SIMD_LANES];
        for input_start in (0..n_inputs).step_by(INPUT_TILE) {
            let tile = INPUT_TILE.min(n_inputs - input_start);
            let mut accumulators = [f32x8::splat(0.0); INPUT_TILE];
            for group in 0..groups {
                let scale = f32x8::splat(read_scale(scales, scale_dtype, row * groups + group));
                let start = group * group_size;
                for column in (start..start + group_size).step_by(SIMD_LANES) {
                    for lane in 0..SIMD_LANES {
                        codes[lane] = f32::from(read_w8a16_code(packed, column + lane));
                    }
                    let weight = f32x8::from(codes) * scale;
                    for (offset, accumulator) in accumulators[..tile].iter_mut().enumerate() {
                        let begin = (input_start + offset) * cols + column;
                        let input = f32x8::from(<[f32; SIMD_LANES]>::try_from(&input[begin..begin + SIMD_LANES]).unwrap());
                        *accumulator = weight.mul_add(input, *accumulator);
                    }
                }
            }
            for offset in 0..tile {
                row_out[input_start + offset] = accumulators[offset].reduce_add();
            }
        }
    });
    for row in 0..rows {
        for n in 0..n_inputs {
            output[n * rows + row] = transposed[row * n_inputs + n];
        }
    }
    Ok(())
}

/// compressed-tensors W8 每个字节已加 128；翻转符号位即可恢复同一
/// bit pattern 的 i8。一次扩展 16 个值，避免标量逐元素转 f32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load_w8a16_f32x16(packed: *const u8, scale: f32) -> __m512 {
    let raw = unsafe { _mm_loadu_si128(packed.cast()) };
    let signed = _mm_xor_si128(raw, _mm_set1_epi8(i8::MIN));
    let integers = _mm512_cvtepi8_epi32(signed);
    _mm512_mul_ps(_mm512_cvtepi32_ps(integers), _mm512_set1_ps(scale))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn matvec_w8a16_avx512(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, _rows: usize, cols: usize, input: &[f32], output: &mut [f32]) {
    let packed_row_bytes = cols.div_ceil(4) * 4;
    let groups = cols / group_size;
    output.par_iter_mut().enumerate().for_each(|(row, output)| {
        let packed = &packed[row * packed_row_bytes..(row + 1) * packed_row_bytes];
        let mut acc = _mm512_setzero_ps();
        for group in 0..groups {
            let scale = read_scale(scales, scale_dtype, row * groups + group);
            let start = group * group_size;
            for column in (start..start + group_size).step_by(16) {
                let weight = unsafe { load_w8a16_f32x16(packed.as_ptr().add(column), scale) };
                let input = unsafe { _mm512_loadu_ps(input.as_ptr().add(column)) };
                acc = _mm512_add_ps(acc, _mm512_mul_ps(weight, input));
            }
        }
        *output = _mm512_reduce_add_ps(acc);
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn matmul_w8a16_avx512(packed: &[u8], scales: &[u8], scale_dtype: ScaleDType, group_size: usize, rows: usize, cols: usize, n_inputs: usize, input: &[f32], output: &mut [f32]) {
    const INPUT_TILE: usize = 8;
    let packed_row_bytes = cols.div_ceil(4) * 4;
    let groups = cols / group_size;
    let mut transposed = vec![0.0f32; rows * n_inputs];
    transposed.par_chunks_mut(n_inputs).enumerate().for_each(|(row, row_out)| {
        let packed = &packed[row * packed_row_bytes..(row + 1) * packed_row_bytes];
        for input_start in (0..n_inputs).step_by(INPUT_TILE) {
            let tile = INPUT_TILE.min(n_inputs - input_start);
            let mut accumulators = [_mm512_setzero_ps(); INPUT_TILE];
            for group in 0..groups {
                let scale = read_scale(scales, scale_dtype, row * groups + group);
                let start = group * group_size;
                for column in (start..start + group_size).step_by(16) {
                    let weight = unsafe { load_w8a16_f32x16(packed.as_ptr().add(column), scale) };
                    for (offset, accumulator) in accumulators[..tile].iter_mut().enumerate() {
                        let begin = (input_start + offset) * cols + column;
                        let input = unsafe { _mm512_loadu_ps(input.as_ptr().add(begin)) };
                        *accumulator = _mm512_add_ps(*accumulator, _mm512_mul_ps(weight, input));
                    }
                }
            }
            for offset in 0..tile {
                row_out[input_start + offset] = _mm512_reduce_add_ps(accumulators[offset]);
            }
        }
    });
    for row in 0..rows {
        for n in 0..n_inputs {
            output[n * rows + row] = transposed[row * n_inputs + n];
        }
    }
}

/// 从 packed 行读取第 `column` 个 INT8 值。
/// compressed-tensors 在 pack 前加 128 转为无符号范围。
fn read_w8a16_code(packed_row: &[u8], column: usize) -> i8 {
    let byte_offset = column / 4 * 4 + column % 4;
    (i16::from(packed_row[byte_offset]) - 128) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_and_matvec_match() {
        let rows = 2;
        let cols = 32;
        let mut packed = vec![0_u8; rows * cols / 2];
        for row in 0..rows {
            for column in 0..cols {
                let code = ((column % 16) as i8 - 8) as i32;
                let nibble = (code + 8) as u32;
                let word_offset = (row * cols.div_ceil(8) + column / 8) * 4;
                let mut word = u32::from_le_bytes(packed[word_offset..word_offset + 4].try_into().unwrap());
                word |= nibble << ((column % 8) * 4);
                packed[word_offset..word_offset + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let scales: Vec<u8> = [0.5_f32, 2.0].into_iter().flat_map(|value| value.to_le_bytes()).collect();
        let mut decoded = vec![0.0; rows * cols];
        decode_w4a16_matrix(&packed, &scales, ScaleDType::F32, 32, rows, cols, &mut decoded).unwrap();
        let input = vec![1.0; cols];
        let mut actual = vec![0.0; rows];
        matvec_w4a16_matrix(&packed, &scales, ScaleDType::F32, 32, rows, cols, &input, &mut actual).unwrap();
        for row in 0..rows {
            let expected: f32 = decoded[row * cols..(row + 1) * cols].iter().sum();
            assert!((actual[row] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn batched_matmul_matches_per_token_matvec() {
        // [3, 32] 权重,group=32,4 个输入行:批量 matmul 必须与逐 token matvec 逐元素一致。
        let rows = 3;
        let cols = 32;
        let group_size = 32;
        let mut packed = vec![0_u8; rows * cols / 2];
        for row in 0..rows {
            for column in 0..cols {
                let code = (((column + row) % 16) as i32) - 8;
                let nibble = (code + 8) as u32;
                let word_offset = (row * cols.div_ceil(8) + column / 8) * 4;
                let mut word = u32::from_le_bytes(packed[word_offset..word_offset + 4].try_into().unwrap());
                word |= nibble << ((column % 8) * 4);
                packed[word_offset..word_offset + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let scales: Vec<u8> = [0.5_f32, 1.0, 2.0].into_iter().flat_map(|value| value.to_le_bytes()).collect();
        let n_inputs = 4;
        let input: Vec<f32> = (0..n_inputs * cols).map(|index| (index as f32 * 0.1) - 1.0).collect();

        let mut batched = vec![0.0f32; n_inputs * rows];
        matmul_w4a16_matrix(&packed, &scales, ScaleDType::F32, group_size, rows, cols, n_inputs, &input, &mut batched).unwrap();

        for n in 0..n_inputs {
            let mut single = vec![0.0f32; rows];
            matvec_w4a16_matrix(&packed, &scales, ScaleDType::F32, group_size, rows, cols, &input[n * cols..(n + 1) * cols], &mut single).unwrap();
            for row in 0..rows {
                assert!((batched[n * rows + row] - single[row]).abs() < 1.0e-5, "n={n} row={row}: {} vs {}", batched[n * rows + row], single[row]);
            }
        }
    }

    #[test]
    fn w8a16_decode_and_matvec_match() {
        let rows = 2;
        let cols = 32;
        // 每 int32 存 4 个 INT8,行内按字节连续。
        let mut packed = vec![0_u8; rows * cols];
        for row in 0..rows {
            for column in 0..cols {
                let value = (column % 128) as i8 - 64; // [-64, 63]
                let byte_offset = row * cols + column;
                packed[byte_offset] = (i16::from(value) + 128) as u8;
            }
        }
        let scales: Vec<u8> = [0.5_f32, 2.0].into_iter().flat_map(|value| value.to_le_bytes()).collect();
        let mut decoded = vec![0.0; rows * cols];
        decode_w8a16_matrix(&packed, &scales, ScaleDType::F32, 32, rows, cols, &mut decoded).unwrap();
        let input = vec![1.0; cols];
        let mut actual = vec![0.0; rows];
        matvec_w8a16_matrix(&packed, &scales, ScaleDType::F32, 32, rows, cols, &input, &mut actual).unwrap();
        for row in 0..rows {
            let expected: f32 = decoded[row * cols..(row + 1) * cols].iter().sum();
            assert!((actual[row] - expected).abs() < 1e-4, "row {row}: {actual:?} vs {expected}");
        }
    }

    #[test]
    fn w8a16_batched_matmul_matches_matvec() {
        let rows = 7;
        let cols = 256;
        let group_size = 128;
        let mut packed = vec![0_u8; rows * cols];
        for row in 0..rows {
            for column in 0..cols {
                packed[row * cols + column] = (((row * 17 + column * 13) % 255) as i16 - 127 + 128) as u8;
            }
        }
        let scales = (0..rows * (cols / group_size)).flat_map(|index| (0.002_f32 * (index + 1) as f32).to_le_bytes()).collect::<Vec<_>>();
        let n_inputs = 11;
        let input = (0..n_inputs * cols).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
        let mut actual = vec![0.0; n_inputs * rows];
        matmul_w8a16_matrix(&packed, &scales, ScaleDType::F32, group_size, rows, cols, n_inputs, &input, &mut actual).unwrap();
        for n in 0..n_inputs {
            let mut expected = vec![0.0; rows];
            matvec_w8a16_matrix(&packed, &scales, ScaleDType::F32, group_size, rows, cols, &input[n * cols..(n + 1) * cols], &mut expected).unwrap();
            for row in 0..rows {
                assert!((actual[n * rows + row] - expected[row]).abs() < 1.0e-4, "n={n} row={row}: {} != {}", actual[n * rows + row], expected[row]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn w8a8_vector_quantizer_matches_scalar_codes() {
        let rows = 6;
        let cols = 256;
        let group_size = 128;
        let mut input = (0..rows * cols).map(|index| ((index as f32 * 0.017).sin() * 3.0).clamp(-2.7, 2.7)).collect::<Vec<_>>();
        // 覆盖全零组(maximum=0 → scale=1)与极值饱和。
        input[..cols].fill(0.0);
        input[cols..cols + 16].fill(1.0e30);
        let vector = quantize_w8a8_input(&input, rows, cols, group_size).unwrap();
        let scalar = quantize_w8a8_input_scalar(&input, rows, cols, group_size).unwrap();
        assert_eq!(vector.values, scalar.values);
        assert_eq!(vector.scales, scalar.scales);
        assert_eq!(vector.corrections, scalar.corrections);
        // 非有限值必须被拒绝。
        input[0] = f32::NAN;
        assert!(quantize_w8a8_input(&input, rows, cols, group_size).is_err());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn w8a8_vnni_matches_scalar_quantized_dot() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") || !is_x86_feature_detected!("avx512vnni") || !is_x86_feature_detected!("fma") {
            return;
        }
        let rows = 19;
        let cols = 256;
        let group_size = 128;
        let groups = cols / group_size;
        let packed = (0..rows * cols).map(|index| ((index * 29 + 17) % 255 + 1) as u8).collect::<Vec<_>>();
        let scales = (0..rows * groups).flat_map(|index| (0.001_f32 * (index + 1) as f32).to_le_bytes()).collect::<Vec<_>>();
        let input_rows = 11;
        let input = (0..input_rows * cols).map(|index| ((index as f32 * 0.017).sin() * 3.0).clamp(-2.7, 2.7)).collect::<Vec<_>>();
        let matrix = W8A8VnniMatrix::try_repack(&packed, &scales, ScaleDType::F32, group_size, rows, cols).unwrap().unwrap();
        let quantized = quantize_w8a8_input(&input, input_rows, cols, group_size).unwrap();
        let mut actual = vec![0.0_f32; input_rows * rows];
        matrix.matmul(input_rows, &input, &mut actual).unwrap();
        for input_row in 0..input_rows {
            for row in 0..rows {
                let mut expected = 0.0_f32;
                for group in 0..groups {
                    let mut dot = 0_i32;
                    for column in group * group_size..(group + 1) * group_size {
                        let weight = i32::from(packed[row * cols + column]) - 128;
                        dot += weight * i32::from(quantized.values[input_row * cols + column]);
                    }
                    expected += dot as f32 * read_scale(&scales, ScaleDType::F32, row * groups + group) * quantized.scales[input_row * groups + group];
                }
                let value = actual[input_row * rows + row];
                assert!((value - expected).abs() < 2.0e-3, "input={input_row} row={row}: {value} != {expected}");
            }
        }
    }
}
