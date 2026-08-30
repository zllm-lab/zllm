//! GGML block quant CPU 并行 matvec kernel。

use rayon::prelude::*;

use crate::weight::codec::ggml;

type DotFn = unsafe fn(&[f32], &[f32]) -> f32;
type QuantDotFn = unsafe fn(&[u8], &[f32]) -> f32;
type QuantPairDotFn = unsafe fn(&[u8], &[u8], &[f32]) -> (f32, f32);

unsafe fn dot_f32_scalar(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(left, right)| left * right).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut index = 0usize;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    while index + 32 <= left.len() {
        unsafe {
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(left.as_ptr().add(index)), _mm256_loadu_ps(right.as_ptr().add(index)), sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(left.as_ptr().add(index + 8)), _mm256_loadu_ps(right.as_ptr().add(index + 8)), sum1);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(left.as_ptr().add(index + 16)), _mm256_loadu_ps(right.as_ptr().add(index + 16)), sum2);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(left.as_ptr().add(index + 24)), _mm256_loadu_ps(right.as_ptr().add(index + 24)), sum3);
        }
        index += 32;
    }
    let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let halves = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps::<1>(sum));
    let pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    let mut total = _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs)));
    while index < left.len() {
        total += left[index] * right[index];
        index += 1;
    }
    total
}

fn direct_dot() -> DotFn {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return dot_f32_avx2;
    }
    dot_f32_scalar
}

#[cfg(target_arch = "x86_64")]
fn q4_k_scale_min(group: usize, scales: &[u8]) -> (u8, u8) {
    if group < 4 { (scales[group] & 0x3f, scales[group + 4] & 0x3f) } else { ((scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4), (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8_0_dot_avx2(block: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let scale = _mm256_set1_ps(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
    let quants = unsafe { _mm256_loadu_si256(block.as_ptr().add(2).cast()) };
    let halves = [_mm256_castsi256_si128(quants), _mm256_extracti128_si256::<1>(quants)];
    let mut total = _mm256_setzero_ps();
    for (half, bytes) in halves.into_iter().enumerate() {
        let chunks = [bytes, _mm_srli_si128::<8>(bytes)];
        for (chunk, bytes) in chunks.into_iter().enumerate() {
            let quant = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            let weight = _mm256_mul_ps(quant, scale);
            let offset = half * 16 + chunk * 8;
            let activation = unsafe { _mm256_loadu_ps(input.as_ptr().add(offset)) };
            total = _mm256_fmadd_ps(activation, weight, total);
        }
    }
    let halves = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps::<1>(total));
    let pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_k_dot_avx2(block: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let nibble_mask = _mm256_set1_epi8(15);
    let mut total = _mm256_setzero_ps();
    for group in 0..8 {
        let (scale, minimum) = q4_k_scale_min(group, &block[4..16]);
        let packed = unsafe { _mm256_loadu_si256(block.as_ptr().add(16 + (group / 2) * 32).cast()) };
        let quants = if group.is_multiple_of(2) { _mm256_and_si256(packed, nibble_mask) } else { _mm256_and_si256(_mm256_srli_epi16::<4>(packed), nibble_mask) };
        let halves = [_mm256_castsi256_si128(quants), _mm256_extracti128_si256::<1>(quants)];
        let scale = _mm256_set1_ps(d * scale as f32);
        let minimum = _mm256_set1_ps(dmin * minimum as f32);
        for (half, bytes) in halves.into_iter().enumerate() {
            let chunks = [bytes, _mm_srli_si128::<8>(bytes)];
            for (chunk, bytes) in chunks.into_iter().enumerate() {
                let quant = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
                let weight = _mm256_sub_ps(_mm256_mul_ps(quant, scale), minimum);
                let offset = group * 32 + half * 16 + chunk * 8;
                let activation = unsafe { _mm256_loadu_ps(input.as_ptr().add(offset)) };
                total = _mm256_fmadd_ps(activation, weight, total);
            }
        }
    }
    let halves = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps::<1>(total));
    let pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_k_pair_dot_avx2(first: &[u8], second: &[u8], input: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;

    let first_d = half::f16::from_le_bytes([first[0], first[1]]).to_f32();
    let first_dmin = half::f16::from_le_bytes([first[2], first[3]]).to_f32();
    let second_d = half::f16::from_le_bytes([second[0], second[1]]).to_f32();
    let second_dmin = half::f16::from_le_bytes([second[2], second[3]]).to_f32();
    let nibble_mask = _mm256_set1_epi8(15);
    let mut first_total = _mm256_setzero_ps();
    let mut second_total = _mm256_setzero_ps();
    for group in 0..8 {
        let (first_scale, first_minimum) = q4_k_scale_min(group, &first[4..16]);
        let (second_scale, second_minimum) = q4_k_scale_min(group, &second[4..16]);
        let first_packed = unsafe { _mm256_loadu_si256(first.as_ptr().add(16 + (group / 2) * 32).cast()) };
        let second_packed = unsafe { _mm256_loadu_si256(second.as_ptr().add(16 + (group / 2) * 32).cast()) };
        let first_quants = if group.is_multiple_of(2) { _mm256_and_si256(first_packed, nibble_mask) } else { _mm256_and_si256(_mm256_srli_epi16::<4>(first_packed), nibble_mask) };
        let second_quants = if group.is_multiple_of(2) { _mm256_and_si256(second_packed, nibble_mask) } else { _mm256_and_si256(_mm256_srli_epi16::<4>(second_packed), nibble_mask) };
        let first_halves = [_mm256_castsi256_si128(first_quants), _mm256_extracti128_si256::<1>(first_quants)];
        let second_halves = [_mm256_castsi256_si128(second_quants), _mm256_extracti128_si256::<1>(second_quants)];
        let first_scale = _mm256_set1_ps(first_d * first_scale as f32);
        let first_minimum = _mm256_set1_ps(first_dmin * first_minimum as f32);
        let second_scale = _mm256_set1_ps(second_d * second_scale as f32);
        let second_minimum = _mm256_set1_ps(second_dmin * second_minimum as f32);
        for half in 0..2 {
            let first_chunks = [first_halves[half], _mm_srli_si128::<8>(first_halves[half])];
            let second_chunks = [second_halves[half], _mm_srli_si128::<8>(second_halves[half])];
            for chunk in 0..2 {
                let first_quant = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(first_chunks[chunk]));
                let second_quant = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(second_chunks[chunk]));
                let first_weight = _mm256_sub_ps(_mm256_mul_ps(first_quant, first_scale), first_minimum);
                let second_weight = _mm256_sub_ps(_mm256_mul_ps(second_quant, second_scale), second_minimum);
                let offset = group * 32 + half * 16 + chunk * 8;
                let activation = unsafe { _mm256_loadu_ps(input.as_ptr().add(offset)) };
                first_total = _mm256_fmadd_ps(activation, first_weight, first_total);
                second_total = _mm256_fmadd_ps(activation, second_weight, second_total);
            }
        }
    }
    unsafe { (horizontal_sum_avx2(first_total), horizontal_sum_avx2(second_total)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let halves = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps::<1>(value));
    let pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs)))
}

fn direct_quant_dot(tensor_type: u32) -> Option<QuantDotFn> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return match tensor_type {
            8 => Some(q8_0_dot_avx2),
            12 => Some(q4_k_dot_avx2),
            _ => None,
        };
    }
    let _ = tensor_type;
    None
}

fn direct_quant_pair_dot(tensor_type: u32) -> Option<QuantPairDotFn> {
    #[cfg(target_arch = "x86_64")]
    if tensor_type == 12 && std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return Some(q4_k_pair_dot_avx2);
    }
    let _ = tensor_type;
    None
}

pub fn matvec(tensor_type: u32, bytes: &[u8], rows: usize, columns: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    if input.len() != columns || output.len() != rows {
        return Err(format!("GGML matvec shape 不匹配: matrix=[{rows}, {columns}], input={}, output={}", input.len(), output.len()));
    }
    let (block_elements, block_bytes) = ggml::block_layout(tensor_type)?;
    if !columns.is_multiple_of(block_elements) {
        return Err(format!("GGML matvec columns={columns} 未按 block {block_elements} 对齐"));
    }
    let row_bytes = columns / block_elements * block_bytes;
    let expected = row_bytes.checked_mul(rows).ok_or_else(|| "GGML matvec 字节数溢出".to_owned())?;
    if bytes.len() != expected {
        return Err(format!("GGML matvec 字节数不匹配: 期望 {expected}，实际 {}", bytes.len()));
    }
    let dot = direct_dot();
    let quant_dot = direct_quant_dot(tensor_type);
    bytes.par_chunks_exact(row_bytes).zip(output.par_iter_mut()).try_for_each(|(row, target)| {
        let mut decoded = [0.0_f32; 256];
        let mut sum = 0.0;
        for (block, input) in row.chunks_exact(block_bytes).zip(input.chunks_exact(block_elements)) {
            if let Some(quant_dot) = quant_dot {
                sum += unsafe { quant_dot(block, input) };
            } else {
                let values = &mut decoded[..block_elements];
                ggml::decode_block(tensor_type, block, values)?;
                sum += unsafe { dot(values, input) };
            }
        }
        *target = sum;
        Ok(())
    })
}

/// matvec + residual:在每个 output 元素上累加同位置 residual;与 matvec 后续接
/// `for r,o in zip(residual, &output) o += r` 的两步行为在 f32 域等价(无中间
/// f16 量化)。fused GGML matvec + GPU residual epilogue 的 CPU oracle。
pub fn matvec_residual(tensor_type: u32, bytes: &[u8], rows: usize, columns: usize, input: &[f32], output: &mut [f32], residual: &[f32]) -> Result<(), String> {
    if residual.len() != rows {
        return Err(format!("GGML matvec_residual residual 长度={} 与 rows={rows} 不匹配", residual.len()));
    }
    matvec(tensor_type, bytes, rows, columns, input, output)?;
    for (target, add) in output.iter_mut().zip(residual.iter()) {
        *target += *add;
    }
    Ok(())
}

/// GGML decode 门控投影：单次权重行调度完成 gate/up 点积和 SiLU。
pub fn gated_silu_matvec(tensor_type: u32, gate: &[u8], up: &[u8], rows: usize, columns: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    if input.len() != columns || output.len() != rows {
        return Err(format!("GGML gated matvec shape 不匹配: weight=[{rows},{columns}] input={} output={}", input.len(), output.len()));
    }
    let (block_elements, block_bytes) = ggml::block_layout(tensor_type)?;
    if !columns.is_multiple_of(block_elements) {
        return Err(format!("GGML gated matvec columns={columns} 未按 block {block_elements} 对齐"));
    }
    let row_bytes = columns / block_elements * block_bytes;
    let expected = rows.checked_mul(row_bytes).ok_or_else(|| "GGML gated matvec 字节数溢出".to_owned())?;
    if gate.len() != expected || up.len() != expected {
        return Err(format!("GGML gated matvec gate={} up={}，期望 {expected}", gate.len(), up.len()));
    }
    let dot = direct_dot();
    let quant_dot = direct_quant_dot(tensor_type);
    let quant_pair_dot = direct_quant_pair_dot(tensor_type);
    gate.par_chunks_exact(row_bytes).zip(up.par_chunks_exact(row_bytes)).zip(output.par_iter_mut()).try_for_each(|((gate_row, up_row), target)| {
        let mut gate_decoded = [0.0_f32; 256];
        let mut up_decoded = [0.0_f32; 256];
        let mut gate_sum = 0.0;
        let mut up_sum = 0.0;
        for ((gate_block, up_block), input) in gate_row.chunks_exact(block_bytes).zip(up_row.chunks_exact(block_bytes)).zip(input.chunks_exact(block_elements)) {
            if let Some(quant_pair_dot) = quant_pair_dot {
                let (gate, up) = unsafe { quant_pair_dot(gate_block, up_block, input) };
                gate_sum += gate;
                up_sum += up;
            } else if let Some(quant_dot) = quant_dot {
                gate_sum += unsafe { quant_dot(gate_block, input) };
                up_sum += unsafe { quant_dot(up_block, input) };
            } else {
                let gate_values = &mut gate_decoded[..block_elements];
                let up_values = &mut up_decoded[..block_elements];
                ggml::decode_block(tensor_type, gate_block, gate_values)?;
                ggml::decode_block(tensor_type, up_block, up_values)?;
                gate_sum += unsafe { dot(gate_values, input) };
                up_sum += unsafe { dot(up_values, input) };
            }
        }
        *target = (gate_sum / (1.0 + (-gate_sum).exp())) * up_sum;
        Ok::<(), String>(())
    })
}

/// GGML prefill matmul：每个 packed 权重 block 只解码一次，复用于全部 token。
pub fn matmul(tensor_type: u32, bytes: &[u8], rows: usize, columns: usize, input_rows: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    if input_rows == 0 || input.len() != input_rows * columns || output.len() != input_rows * rows {
        return Err(format!("GGML matmul shape 不匹配: weight=[{rows},{columns}] input=[{input_rows},{}] output={}", input.len() / input_rows.max(1), output.len()));
    }
    let (block_elements, block_bytes) = ggml::block_layout(tensor_type)?;
    if !columns.is_multiple_of(block_elements) {
        return Err(format!("GGML matmul columns={columns} 未按 block {block_elements} 对齐"));
    }
    let row_bytes = columns / block_elements * block_bytes;
    if bytes.len() != rows.checked_mul(row_bytes).ok_or_else(|| "GGML matmul 字节数溢出".to_owned())? {
        return Err(format!("GGML matmul weight 字节数不匹配: {}", bytes.len()));
    }
    // 每个量化 block 只展开到 256 元素栈 scratch；权重矩阵始终保持 packed。
    // 显式 AVX2/FMA 避免迭代器归约为保持顺序而退化成标量点积。
    let dot = direct_dot();
    let mut row_major = vec![0.0f32; rows * input_rows];
    bytes.par_chunks_exact(row_bytes).zip(row_major.par_chunks_exact_mut(input_rows)).try_for_each(|(weight_row, sums)| {
        let mut decoded = [0.0_f32; 256];
        for (block_index, block) in weight_row.chunks_exact(block_bytes).enumerate() {
            let values = &mut decoded[..block_elements];
            ggml::decode_block(tensor_type, block, values)?;
            let column = block_index * block_elements;
            for (token, sum) in sums.iter_mut().enumerate() {
                let input = &input[token * columns + column..token * columns + column + block_elements];
                *sum += unsafe { dot(values, input) };
            }
        }
        Ok::<(), String>(())
    })?;
    for (row, sums) in row_major.chunks_exact(input_rows).enumerate() {
        for (token, value) in sums.iter().enumerate() {
            output[token * rows + row] = *value;
        }
    }
    Ok(())
}

/// GGML gate/up packed prefill：同一轮遍历中完成两路直算与 SiLU，避免两份
/// `[tokens, intermediate]` 中间输出。每次仍只解码当前 block 到栈 scratch。
pub fn gated_silu_matmul(tensor_type: u32, gate: &[u8], up: &[u8], rows: usize, columns: usize, input_rows: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    if input_rows == 0 || input.len() != input_rows * columns || output.len() != input_rows * rows {
        return Err(format!("GGML gated matmul shape 不匹配: weight=[{rows},{columns}] input_rows={input_rows} input={} output={}", input.len(), output.len()));
    }
    let (block_elements, block_bytes) = ggml::block_layout(tensor_type)?;
    if !columns.is_multiple_of(block_elements) {
        return Err(format!("GGML gated matmul columns={columns} 未按 block {block_elements} 对齐"));
    }
    let row_bytes = columns / block_elements * block_bytes;
    let expected = rows.checked_mul(row_bytes).ok_or_else(|| "GGML gated matmul 字节数溢出".to_owned())?;
    if gate.len() != expected || up.len() != expected {
        return Err(format!("GGML gated matmul gate={} up={}，期望 {expected}", gate.len(), up.len()));
    }
    let dot = direct_dot();
    let mut row_major = vec![0.0f32; rows * input_rows];
    let mut up_row_major = vec![0.0f32; rows * input_rows];
    gate.par_chunks_exact(row_bytes).zip(up.par_chunks_exact(row_bytes)).zip(row_major.par_chunks_exact_mut(input_rows).zip(up_row_major.par_chunks_exact_mut(input_rows))).try_for_each(|((gate_row, up_row), (gate_sums, up_sums))| {
        let mut gate_decoded = [0.0_f32; 256];
        let mut up_decoded = [0.0_f32; 256];
        for (block_index, (gate_block, up_block)) in gate_row.chunks_exact(block_bytes).zip(up_row.chunks_exact(block_bytes)).enumerate() {
            let gate_values = &mut gate_decoded[..block_elements];
            let up_values = &mut up_decoded[..block_elements];
            ggml::decode_block(tensor_type, gate_block, gate_values)?;
            ggml::decode_block(tensor_type, up_block, up_values)?;
            let column = block_index * block_elements;
            for token in 0..input_rows {
                let input = &input[token * columns + column..token * columns + column + block_elements];
                gate_sums[token] += unsafe { dot(gate_values, input) };
                up_sums[token] += unsafe { dot(up_values, input) };
            }
        }
        for token in 0..input_rows {
            let gate = gate_sums[token];
            gate_sums[token] = (gate / (1.0 + (-gate).exp())) * up_sums[token];
        }
        Ok::<(), String>(())
    })?;
    for (row, values) in row_major.chunks_exact(input_rows).enumerate() {
        for (token, value) in values.iter().enumerate() {
            output[token * rows + row] = *value;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::{gated_silu_matmul, gated_silu_matvec, matmul, matvec, matvec_residual};

    fn packed_blocks(tensor_type: u32, rows: usize) -> Vec<u8> {
        let block_bytes = match tensor_type {
            8 => 34,
            12 => 144,
            13 => 176,
            14 => 210,
            23 => 136,
            _ => unreachable!(),
        };
        let blocks_per_row = if tensor_type == 8 { 8 } else { 1 };
        let mut bytes = vec![0u8; rows * blocks_per_row * block_bytes];
        for block_index in 0..rows * blocks_per_row {
            let row = block_index / blocks_per_row;
            let block = &mut bytes[block_index * block_bytes..(block_index + 1) * block_bytes];
            if tensor_type == 8 {
                block[..2].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
                for (index, value) in block[2..].iter_mut().enumerate() {
                    *value = (index as i8).wrapping_mul(7).wrapping_add((row as i8).wrapping_mul(3)) as u8;
                }
            } else if tensor_type == 14 {
                for (index, value) in block[..192].iter_mut().enumerate() {
                    *value = (index * 37 + row * 11) as u8;
                }
                for (index, value) in block[192..208].iter_mut().enumerate() {
                    *value = ((index as i8 % 9) - 4) as u8;
                }
                block[208..210].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
            } else if tensor_type == 23 {
                // IQ4_XS:d(f16) + scales_h(u16le) + scales_l(8 bytes) + qs(128 bytes) = 136 字节
                block[..2].copy_from_slice(&f16::from_f32(0.0625).to_le_bytes());
                block[2..4].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
                for (index, value) in block[4..136].iter_mut().enumerate() {
                    *value = (index * 23 + row * 5) as u8;
                }
            } else {
                block[..2].copy_from_slice(&f16::from_f32(0.0625).to_le_bytes());
                block[2..4].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
                block[4..16].fill(0x11);
                for (index, value) in block[16..].iter_mut().enumerate() {
                    *value = (index * 29 + row * 7) as u8;
                }
            }
        }
        bytes
    }

    #[test]
    fn quantized_matvec_keeps_rows_separate() {
        let mut bytes = vec![0u8; 68];
        bytes[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        bytes[2..34].fill(1);
        bytes[34..36].copy_from_slice(&0x4000u16.to_le_bytes());
        bytes[36..68].fill(1);
        let mut output = [0.0; 2];
        matvec(8, &bytes, 2, 32, &[1.0; 32], &mut output).unwrap();
        assert_eq!(output, [32.0, 64.0]);
    }

    #[test]
    fn quantized_matvec_matches_scalar_decode() {
        let columns = 256;
        let output_rows = 2;
        let input = (0..columns).map(|index| ((index * 17 % 101) as f32 - 50.0) / 64.0).collect::<Vec<_>>();
        for tensor_type in [8, 12, 13, 14] {
            let packed = packed_blocks(tensor_type, output_rows);
            let decoded = crate::weight::codec::ggml::dequantize(tensor_type, &packed, output_rows * columns).unwrap();
            let mut actual = vec![0.0; output_rows];
            matvec(tensor_type, &packed, output_rows, columns, &input, &mut actual).unwrap();
            for row in 0..output_rows {
                let expected = (0..columns).map(|column| input[column] * decoded[row * columns + column]).sum::<f32>();
                let tolerance = 1.0e-3 + expected.abs() * 1.0e-5;
                assert!((actual[row] - expected).abs() <= tolerance, "type={tensor_type} row={row}: actual={} expected={expected}", actual[row]);
            }
        }
    }

    #[test]
    fn quantized_matmul_reuses_weight_decode_across_tokens() {
        let mut bytes = vec![0u8; 68];
        bytes[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        bytes[2..34].fill(1);
        bytes[34..36].copy_from_slice(&0x4000u16.to_le_bytes());
        bytes[36..68].fill(1);
        let input = [[1.0; 32], [0.5; 32]].concat();
        let mut output = [0.0; 4];
        matmul(8, &bytes, 2, 32, 2, &input, &mut output).unwrap();
        assert_eq!(output, [32.0, 64.0, 16.0, 32.0]);
    }

    #[test]
    fn quantized_matmul_matches_scalar_decode() {
        let input_rows = 9;
        let columns = 256;
        let output_rows = 2;
        let input = (0..input_rows * columns).map(|index| ((index * 17 % 101) as f32 - 50.0) / 64.0).collect::<Vec<_>>();
        for tensor_type in [8, 12, 13, 14] {
            let packed = packed_blocks(tensor_type, output_rows);
            let decoded = crate::weight::codec::ggml::dequantize(tensor_type, &packed, output_rows * columns).unwrap();
            let mut actual = vec![0.0; input_rows * output_rows];
            matmul(tensor_type, &packed, output_rows, columns, input_rows, &input, &mut actual).unwrap();
            for token in 0..input_rows {
                for row in 0..output_rows {
                    let expected = (0..columns).map(|column| input[token * columns + column] * decoded[row * columns + column]).sum::<f32>();
                    let value = actual[token * output_rows + row];
                    let tolerance = 1.0e-3 + expected.abs() * 1.0e-5;
                    assert!((value - expected).abs() <= tolerance, "type={tensor_type} token={token} row={row}: actual={value} expected={expected}");
                }
            }
        }
    }

    #[test]
    fn gated_q4_k_matmul_matches_two_direct_projections() {
        let input_rows = 9;
        let columns = 512;
        let output_rows = 2;
        let input = (0..input_rows * columns).map(|index| ((index * 13 % 89) as f32 - 44.0) / 96.0).collect::<Vec<_>>();
        let packed = packed_blocks(12, output_rows * columns / 256);
        let mut gate = vec![0.0; input_rows * output_rows];
        let mut up = vec![0.0; input_rows * output_rows];
        matmul(12, &packed, output_rows, columns, input_rows, &input, &mut gate).unwrap();
        matmul(12, &packed, output_rows, columns, input_rows, &input, &mut up).unwrap();
        let mut actual = vec![0.0; input_rows * output_rows];
        gated_silu_matmul(12, &packed, &packed, output_rows, columns, input_rows, &input, &mut actual).unwrap();
        for ((actual, gate), up) in actual.iter().zip(gate).zip(up) {
            let expected = (gate / (1.0 + (-gate).exp())) * up;
            assert_eq!(*actual, expected);
        }
    }

    #[test]
    fn gated_q4_k_matvec_matches_two_direct_projections() {
        let columns = 512;
        let output_rows = 2;
        let input = (0..columns).map(|index| ((index * 13 % 89) as f32 - 44.0) / 96.0).collect::<Vec<_>>();
        let packed = packed_blocks(12, output_rows * columns / 256);
        let mut gate = vec![0.0; output_rows];
        let mut up = vec![0.0; output_rows];
        matvec(12, &packed, output_rows, columns, &input, &mut gate).unwrap();
        matvec(12, &packed, output_rows, columns, &input, &mut up).unwrap();
        let mut actual = vec![0.0; output_rows];
        gated_silu_matvec(12, &packed, &packed, output_rows, columns, &input, &mut actual).unwrap();
        for ((actual, gate), up) in actual.iter().zip(gate).zip(up) {
            let expected = (gate / (1.0 + (-gate).exp())) * up;
            assert_eq!(*actual, expected);
        }
    }

    /// `matvec_residual` 与"先 `matvec` 再逐元素加 residual"两步行为在 f32 域完全一致。
    /// 锁定 GPU fused `gguf_gemv_iq4xs_add_f16` kernel 的 CPU oracle。覆盖 IQ4_XS (23)
    /// 主用例:多输出行 + 非平凡 residual。
    #[test]
    fn matvec_residual_matches_unfused_add() {
        let columns = 512;
        let output_rows = 4;
        let input: Vec<f32> = (0..columns).map(|index| ((index * 19 % 97) as f32 - 48.0) / 80.0).collect();
        let packed = packed_blocks(23, output_rows * columns / 256);
        let residual: Vec<f32> = (0..output_rows).map(|row| (row as f32 - 1.5) * 0.073).collect();
        // 路径 1:fused matvec_residual
        let mut fused = vec![0.0_f32; output_rows];
        matvec_residual(23, &packed, output_rows, columns, &input, &mut fused, &residual).unwrap();
        // 路径 2:matvec 后接独立 add
        let mut unfused = vec![0.0_f32; output_rows];
        matvec(23, &packed, output_rows, columns, &input, &mut unfused).unwrap();
        for (slot, add) in unfused.iter_mut().zip(residual.iter()) {
            *slot += *add;
        }
        for (row, (fused, unfused)) in fused.iter().zip(unfused.iter()).enumerate() {
            let tolerance = 1.0e-4 + fused.abs() * 1.0e-5;
            assert!((fused - unfused).abs() <= tolerance, "row={row}: fused={fused} unfused={unfused}");
        }
    }

    /// residual 长度错配必须报错,不让 fused kernel 在 GPU 端写出 buffer overflow。
    #[test]
    fn matvec_residual_rejects_shape_mismatch() {
        let columns = 256;
        let output_rows = 2;
        let input = vec![0.5_f32; columns];
        let packed = packed_blocks(23, output_rows * columns / 256);
        let mut output = vec![0.0_f32; output_rows];
        let residual_bad = vec![0.0_f32; output_rows + 1];
        let err = matvec_residual(23, &packed, output_rows, columns, &input, &mut output, &residual_bad).unwrap_err();
        assert!(err.contains("不匹配"), "unexpected error: {err}");
    }
}
