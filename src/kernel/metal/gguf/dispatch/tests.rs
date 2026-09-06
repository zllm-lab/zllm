//! GGUF Metal dispatch 正确性测试。

use super::*;

#[cfg(test)]
mod q6k_gemv_tests {
    use super::*;

    #[test]
    fn q4_0_gemv_matches_cpu_dequant() {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, columns, row_bytes) = (5, 256, 144);
        let input_values = (0..columns).map(|index| ((index as f32 + 1.0) * 0.013).sin()).collect::<Vec<_>>();
        let rounded_input = input_values.iter().map(|&value| f16::from_f32(value).to_f32()).collect::<Vec<_>>();
        let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let mut weights = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..columns / 32 {
                let bytes = &mut weights[row * row_bytes + block * 18..row * row_bytes + (block + 1) * 18];
                bytes[..2].copy_from_slice(&f16::from_f32(0.03125 + row as f32 * 0.00390625).to_bits().to_le_bytes());
                for (index, value) in bytes[2..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(29).wrapping_add(row * 11).wrapping_add(block * 7) as u8;
                }
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 2, row_bytes, rows, columns).unwrap();
        let actual = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(2, &weights, rows * columns).unwrap();
        for row in 0..rows {
            let expected = decoded[row * columns..(row + 1) * columns].iter().zip(&rounded_input).map(|(weight, input)| weight * input).sum::<f32>();
            assert!((actual[row] - expected).abs() <= 0.02, "row={row} actual={} expected={expected}", actual[row]);
        }
    }

    /// 多行(5 行,跨 8 行批边界内)走 q4_0 专用 gemv 的 8 输入行批路径,
    /// 与 CPU dequant 点积对拍;覆盖 MTP verify / 短 prefill 的 multirow 分支。
    #[test]
    fn q4_0_multirow_gemv_matches_cpu_dequant() {
        let ctx = MetalContext::new_default().unwrap();
        let (input_rows, rows, columns, row_bytes) = (5, 19, 384, 216);
        let input_values = (0..input_rows * columns).map(|index| ((index as f32 + 1.0) * 0.0061).sin()).collect::<Vec<_>>();
        let rounded_input = input_values.iter().map(|&value| f16::from_f32(value).to_f32()).collect::<Vec<_>>();
        let input = ctx.tensor_from_f32(&input_values, input_rows, columns).unwrap();
        let mut weights = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..columns / 32 {
                let bytes = &mut weights[row * row_bytes + block * 18..row * row_bytes + (block + 1) * 18];
                bytes[..2].copy_from_slice(&f16::from_f32(0.03125 + row as f32 * 0.0009765625).to_bits().to_le_bytes());
                for (index, value) in bytes[2..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(29).wrapping_add(row * 11).wrapping_add(block * 7) as u8;
                }
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 2, row_bytes, rows, columns).unwrap();
        let actual = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(2, &weights, rows * columns).unwrap();
        for input_row in 0..input_rows {
            for row in 0..rows {
                let expected = decoded[row * columns..(row + 1) * columns].iter().zip(&rounded_input[input_row * columns..(input_row + 1) * columns]).map(|(weight, input)| weight * input).sum::<f32>();
                let value = actual[input_row * rows + row];
                assert!((value - expected).abs() <= 0.1, "input_row={input_row} row={row} actual={value} expected={expected}");
            }
        }
    }

    #[test]
    fn q4k_three_input_rows_matches_cpu_dequant() {
        let ctx = MetalContext::new_default().unwrap();
        let (input_rows, rows, columns, row_bytes) = (3, 7, 512, 288);
        let input_values = (0..input_rows * columns).map(|index| ((index as f32 + 1.0) * 0.0047).sin()).collect::<Vec<_>>();
        let rounded_input = input_values.iter().map(|&value| f16::from_f32(value).to_f32()).collect::<Vec<_>>();
        let input = ctx.tensor_from_f32(&input_values, input_rows, columns).unwrap();
        let mut weights = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..columns / 256 {
                let bytes = &mut weights[row * row_bytes + block * 144..row * row_bytes + (block + 1) * 144];
                bytes[..2].copy_from_slice(&f16::from_f32(0.001953125 + row as f32 * 0.0001220703125).to_bits().to_le_bytes());
                bytes[2..4].copy_from_slice(&f16::from_f32(0.0009765625 + block as f32 * 0.0001220703125).to_bits().to_le_bytes());
                for (index, value) in bytes[4..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(31).wrapping_add(row * 13).wrapping_add(block * 7) as u8;
                }
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 12, row_bytes, rows, columns).unwrap();
        let actual = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(12, &weights, rows * columns).unwrap();
        for input_row in 0..input_rows {
            for row in 0..rows {
                let expected = decoded[row * columns..(row + 1) * columns].iter().zip(&rounded_input[input_row * columns..(input_row + 1) * columns]).map(|(weight, input)| weight * input).sum::<f32>();
                let value = actual[input_row * rows + row];
                assert!((value - expected).abs() <= 0.1, "input_row={input_row} row={row} actual={value} expected={expected}");
            }
        }
    }

    #[test]
    fn q6k_gemv_matches_dequantized_dot() {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, columns, row_bytes) = (3, 256, 210);
        let input_values = (0..columns).map(|index| ((index as f32 + 1.0) * 0.0078125).sin()).collect::<Vec<_>>();
        let rounded_input = input_values.iter().map(|&value| f16::from_f32(value).to_f32()).collect::<Vec<_>>();
        let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let mut weights = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            let block = &mut weights[row * row_bytes..(row + 1) * row_bytes];
            for (index, value) in block[..192].iter_mut().enumerate() {
                *value = (index.wrapping_mul(37).wrapping_add(row * 11)) as u8;
            }
            for (index, value) in block[192..208].iter_mut().enumerate() {
                *value = ((index as i8 % 7) - 3 + row as i8) as u8;
            }
            block[208..210].copy_from_slice(&f16::from_f32(0.001953125).to_bits().to_le_bytes());
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 14, row_bytes, rows, columns).unwrap();
        let actual = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(14, &weights, rows * columns).unwrap();
        for row in 0..rows {
            let expected = decoded[row * columns..(row + 1) * columns].iter().zip(&rounded_input).map(|(weight, input)| weight * input).sum::<f32>();
            assert!((actual[row] - expected).abs() <= 0.02, "row={row} actual={} expected={expected}", actual[row]);
        }
    }

    /// 设备端 gather:抠出的行必须与 CPU dequant 逐元素一致(流水线正确性的源头)。
    #[test]
    fn q6k_gather_row_matches_cpu_dequant() {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, columns, row_bytes) = (7, 512, 420);
        let mut weights = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..columns / 256 {
                let bytes = &mut weights[row * row_bytes + block * 210..row * row_bytes + (block + 1) * 210];
                for (index, value) in bytes[..192].iter_mut().enumerate() {
                    *value = (index.wrapping_mul(41).wrapping_add(row * 13).wrapping_add(block * 7)) as u8;
                }
                for (index, value) in bytes[192..208].iter_mut().enumerate() {
                    *value = ((index as i32 + row as i32 + block as i32) % 11 - 5) as i8 as u8;
                }
                bytes[208..210].copy_from_slice(&f16::from_f32(0.001953125).to_bits().to_le_bytes());
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let decoded = crate::weight::codec::ggml::dequantize(14, &weights, rows * columns).unwrap();
        for row in 0..rows {
            let id = ctx.shared_buffer(&(row as u32).to_le_bytes());
            let gathered = gguf_gather_row_q6k_tensor(&ctx, &id, &blob, rows, columns, row_bytes).unwrap();
            let actual = ctx.tensor_to_f32(&gathered);
            for (column, (&a, &e)) in actual.iter().zip(&decoded[row * columns..(row + 1) * columns]).enumerate() {
                assert!((a - e).abs() <= 1e-2 * (1.0 + e.abs()), "row={row} column={column} gather={a} cpu={e}");
            }
        }
    }

    /// Q4_K gather + embedding scale:必须与 CPU decode_q4_k + gemma4_embedding_rows
    /// 的双 BF16 舍入路径逐位一致(gemma4 decode 闭环的正确性源头)。
    /// id 集中放一个 buffer 按字节偏移取,顺带覆盖 id_offset 寻址。
    #[test]
    fn q4k_gather_row_matches_cpu_embedding_rows() {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, columns, row_bytes) = (7, 512, 288);
        let mut weights = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..columns / 256 {
                let bytes = &mut weights[row * row_bytes + block * 144..row * row_bytes + (block + 1) * 144];
                bytes[0..2].copy_from_slice(&f16::from_f32(0.001953125 + 0.000244140625 * (row + block) as f32).to_bits().to_le_bytes());
                bytes[2..4].copy_from_slice(&f16::from_f32(0.0009765625 * (row + 1) as f32).to_bits().to_le_bytes());
                for (index, value) in bytes[4..16].iter_mut().enumerate() {
                    *value = (index.wrapping_mul(29).wrapping_add(row * 17).wrapping_add(block * 11)) as u8;
                }
                for (index, value) in bytes[16..144].iter_mut().enumerate() {
                    *value = (index.wrapping_mul(43).wrapping_add(row * 7).wrapping_add(block * 5)) as u8;
                }
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        // 与 gemma4 生产路径一致:scale 先经 bf16 舍入(sqrt(hidden_size)=sqrt(3840))
        let scale = half::bf16::from_f32(3840f32.sqrt()).to_f32();
        let decoded = crate::weight::codec::ggml::dequantize(12, &weights, rows * columns).unwrap();
        let order = [5usize, 0, 6, 2, 4, 1, 3];
        let ids: Vec<u8> = order.iter().flat_map(|row| (*row as u32).to_le_bytes()).collect();
        let id_buffer = ctx.shared_buffer(&ids);
        for (slot, &row) in order.iter().enumerate() {
            let gathered = gguf_gather_row_q4k_tensor_offset(&ctx, &id_buffer, (slot * mem::size_of::<u32>()) as u64, &blob, rows, columns, row_bytes, scale).unwrap();
            let actual = ctx.tensor_to_f32(&gathered);
            for (column, (&a, &e)) in actual.iter().zip(&decoded[row * columns..(row + 1) * columns]).enumerate() {
                let expected = f16::from_f32(half::bf16::from_f32(half::bf16::from_f32(e).to_f32() * scale).to_f32()).to_f32();
                assert_eq!(a, expected, "row={row} column={column} gather={a} cpu={expected}");
            }
        }
    }

    #[test]
    fn q5k_gather_row_matches_cpu_per_layer_embedding() {
        let ctx = MetalContext::new_default().unwrap();
        let (rows, columns, row_bytes) = (5, 512, 352);
        let mut weights = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..columns / 256 {
                let bytes = &mut weights[row * row_bytes + block * 176..row * row_bytes + (block + 1) * 176];
                bytes[0..2].copy_from_slice(&f16::from_f32(0.001953125 + row as f32 * 0.0001220703125).to_bits().to_le_bytes());
                bytes[2..4].copy_from_slice(&f16::from_f32(0.0009765625 + block as f32 * 0.0001220703125).to_bits().to_le_bytes());
                for (index, value) in bytes[4..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(37).wrapping_add(row * 13).wrapping_add(block * 7) as u8;
                }
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let decoded = crate::weight::codec::ggml::dequantize(13, &weights, rows * columns).unwrap();
        let scale = half::bf16::from_f32(256f32.sqrt()).to_f32();
        let ids: Vec<u8> = [4u32, 1, 3, 0, 2].into_iter().flat_map(u32::to_le_bytes).collect();
        let id_buffer = ctx.shared_buffer(&ids);
        for (slot, row) in [4usize, 1, 3, 0, 2].into_iter().enumerate() {
            let gathered = gguf_gather_row_q5k_tensor_offset(&ctx, &id_buffer, (slot * mem::size_of::<u32>()) as u64, &blob, rows, columns, row_bytes, scale).unwrap();
            let actual = ctx.tensor_to_f32(&gathered);
            for (column, (&a, &e)) in actual.iter().zip(&decoded[row * columns..(row + 1) * columns]).enumerate() {
                let expected = f16::from_f32(half::bf16::from_f32(half::bf16::from_f32(e).to_f32() * scale).to_f32()).to_f32();
                assert_eq!(a, expected, "row={row} column={column} gather={a} cpu={expected}");
            }
        }
    }
}

#[cfg(test)]
mod gguf_fused_gemm_tests {
    use super::*;

    fn encode_q3k_block(block: &mut [u8], seed: usize) {
        for (index, value) in block[..96].iter_mut().enumerate() {
            *value = (index.wrapping_mul(37).wrapping_add(seed * 11)) as u8;
        }
        for (index, value) in block[96..108].iter_mut().enumerate() {
            // Q3_K scales: 12 byte packed 6-bit + 2 byte d (f16)
            *value = ((index as i8 + seed as i8).wrapping_mul(19) % 13) as u8;
        }
        block[108..110].copy_from_slice(&f16::from_f32(0.00390625).to_bits().to_le_bytes());
    }

    fn encode_q6k_block(block: &mut [u8], seed: usize) {
        for (index, value) in block[..192].iter_mut().enumerate() {
            *value = (index.wrapping_mul(41).wrapping_add(seed * 13)) as u8;
        }
        for (index, value) in block[192..208].iter_mut().enumerate() {
            *value = ((index as i8 + seed as i8).wrapping_mul(7) % 11) as u8;
        }
        block[208..210].copy_from_slice(&f16::from_f32(0.001953125).to_bits().to_le_bytes());
    }

    #[test]
    fn iq3s_block_dequant_prefill_matches_reference() {
        let ctx = MetalContext::new_default().unwrap();
        let m = 32;
        let columns = 512;
        let n_rows = 12;
        let row_bytes = 220; // 512 / 256 * 110 = 2 IQ3_S blocks
        let input_values: Vec<f32> = (0..(m * columns)).map(|i| ((i as f32 * 0.0037) + 0.09).cos()).collect();
        let rounded_input: Vec<f32> = input_values.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let input = ctx.tensor_from_f32(&input_values, m, columns).unwrap();
        let d_bytes = f16::from_f32(0.015625).to_le_bytes();
        let mut weights = vec![0u8; n_rows * row_bytes];
        let mut seed = 0x85ebca6bu32;
        for byte in weights.chunks_exact_mut(4) {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            byte.copy_from_slice(&seed.to_le_bytes());
        }
        for row in 0..n_rows {
            for block in 0..2 {
                let base = row * row_bytes + block * 110;
                weights[base] = d_bytes[0];
                weights[base + 1] = d_bytes[1];
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 21, row_bytes, n_rows, columns).unwrap();
        let actual_flat = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(21, &weights, n_rows * columns).unwrap();
        for mi in 0..m {
            for ni in 0..n_rows {
                let expected: f32 = (0..columns).map(|k| decoded[ni * columns + k] * rounded_input[mi * columns + k]).sum();
                let actual_val = actual_flat[mi * n_rows + ni];
                let err = (actual_val - expected).abs();
                assert!(err <= 0.05 + expected.abs() * 0.01, "iq3s block dequant m={mi} n={ni} actual={actual_val} expected={expected} err={err}");
            }
        }
    }

    #[test]
    fn iq4xs_block_dequant_prefill_matches_reference() {
        // 32 行起走 fused GEMM(64×64 tile,stage 内反量化+simdgroup MMA)
        let ctx = MetalContext::new_default().unwrap();
        let m = 32;
        let columns = 512;
        let n_rows = 12;
        let row_bytes = 272; // 512 / 256 * 136 = 2 IQ4_XS blocks
        let input_values: Vec<f32> = (0..(m * columns)).map(|i| ((i as f32 * 0.0041) + 0.05).sin()).collect();
        let rounded_input: Vec<f32> = input_values.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let input = ctx.tensor_from_f32(&input_values, m, columns).unwrap();
        let d_bytes = f16::from_f32(0.03125).to_le_bytes();
        let mut weights = vec![0u8; n_rows * row_bytes];
        let mut seed = 0x9e3779b9u32;
        for byte in weights.chunks_exact_mut(4) {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            byte.copy_from_slice(&seed.to_le_bytes());
        }
        for row in 0..n_rows {
            for block in 0..2 {
                let base = row * row_bytes + block * 136;
                weights[base] = d_bytes[0];
                weights[base + 1] = d_bytes[1];
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 23, row_bytes, n_rows, columns).unwrap();
        let actual_flat = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(23, &weights, n_rows * columns).unwrap();
        for mi in 0..m {
            for ni in 0..n_rows {
                let expected: f32 = (0..columns).map(|k| decoded[ni * columns + k] * rounded_input[mi * columns + k]).sum();
                let actual_val = actual_flat[mi * n_rows + ni];
                // 期望是 CPU f32 反量化直乘,GPU 先物化 f16 再 GEMM,误差以相对阈值衡量
                let err = (actual_val - expected).abs();
                assert!(err <= 0.05 + expected.abs() * 0.01, "iq4xs block dequant m={mi} n={ni} actual={actual_val} expected={expected} err={err}");
            }
        }
    }

    #[test]
    fn iq4xs_fused_gemm_matches_dequantized_matmul() {
        // 58 行:64 行 tile 的边界(6 行填充),覆盖 fused 路径的 M/N 越界守卫;
        // 权重行也取 70 覆盖 N 方向的第二 tile 与不满块。
        let ctx = MetalContext::new_default().unwrap();
        let m = 58;
        let columns = 512;
        let n_rows = 70;
        let row_bytes = 272; // 512 / 256 * 136
        let input_values: Vec<f32> = (0..(m * columns)).map(|i| ((i as f32 * 0.0043) + 0.09).cos()).collect();
        let rounded_input: Vec<f32> = input_values.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let input = ctx.tensor_from_f32(&input_values, m, columns).unwrap();
        // 随机字节会生成 0..63 的 ib32 scale,dl 可达 ±32d;真实权重的组合
        // scale 远小于该上界。d 取 0.002 让 |w| ≤ ~0.7,f16 staging 的相对
        // 误差与既有 fused 测试同量级,阈值保持 0.05+1% 不放水。
        let d_bytes = f16::from_f32(0.002).to_le_bytes();
        let mut weights = vec![0u8; n_rows * row_bytes];
        let mut seed = 0x9e3779b9u32;
        for byte in weights.chunks_exact_mut(4) {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            byte.copy_from_slice(&seed.to_le_bytes());
        }
        for row in 0..n_rows {
            for block in 0..2 {
                let base = row * row_bytes + block * 136;
                weights[base] = d_bytes[0];
                weights[base + 1] = d_bytes[1];
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 23, row_bytes, n_rows, columns).unwrap();
        let actual_flat = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(23, &weights, n_rows * columns).unwrap();
        let mut error_squared = 0.0f32;
        let mut reference_squared = 0.0f32;
        for mi in 0..m {
            for ni in 0..n_rows {
                let expected: f32 = (0..columns).map(|k| decoded[ni * columns + k] * rounded_input[mi * columns + k]).sum();
                let actual_val = actual_flat[mi * n_rows + ni];
                let err = (actual_val - expected).abs();
                error_squared += err * err;
                reference_squared += expected * expected;
                assert!(err <= 0.05 + expected.abs() * 0.01, "fused iq4xs m={mi} n={ni} actual={actual_val} expected={expected} err={err}");
            }
        }
        println!("[iq4xs-fused-oracle] rel_l2={}", (error_squared / reference_squared).sqrt());
    }

    #[test]
    fn iq3s_fused_gemm_matches_dequantized_matmul() {
        // 58 行覆盖 64 行 tile 的 M 边界;权重 70 行覆盖 N 方向第二 tile。
        let ctx = MetalContext::new_default().unwrap();
        let m = 58;
        let columns = 512;
        let n_rows = 70;
        let row_bytes = 220; // 512 / 256 * 110
        let input_values: Vec<f32> = (0..(m * columns)).map(|i| ((i as f32 * 0.0031) + 0.07).sin()).collect();
        let rounded_input: Vec<f32> = input_values.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let input = ctx.tensor_from_f32(&input_values, m, columns).unwrap();
        // 随机 scale(6 位)使 db 可达 63d;真实权重组合远小于此,d 取小值
        // 把 f16 staging 误差压回与既有 fused 测试同量级。
        let d_bytes = f16::from_f32(0.002).to_le_bytes();
        let mut weights = vec![0u8; n_rows * row_bytes];
        let mut seed = 0x85ebca6bu32;
        for byte in weights.chunks_exact_mut(4) {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            byte.copy_from_slice(&seed.to_le_bytes());
        }
        for row in 0..n_rows {
            for block in 0..2 {
                let base = row * row_bytes + block * 110;
                weights[base] = d_bytes[0];
                weights[base + 1] = d_bytes[1];
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 21, row_bytes, n_rows, columns).unwrap();
        let actual_flat = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(21, &weights, n_rows * columns).unwrap();
        let mut error_squared = 0.0f32;
        let mut reference_squared = 0.0f32;
        for mi in 0..m {
            for ni in 0..n_rows {
                let expected: f32 = (0..columns).map(|k| decoded[ni * columns + k] * rounded_input[mi * columns + k]).sum();
                let actual_val = actual_flat[mi * n_rows + ni];
                let err = (actual_val - expected).abs();
                error_squared += err * err;
                reference_squared += expected * expected;
                assert!(err <= 0.05 + expected.abs() * 0.01, "fused iq3s m={mi} n={ni} actual={actual_val} expected={expected} err={err}");
            }
        }
        println!("[iq3s-fused-oracle] rel_l2={}", (error_squared / reference_squared).sqrt());
    }

    #[test]
    fn q3k_fused_gemm_matches_dequantized_matmul() {
        let ctx = MetalContext::new_default().unwrap();
        let m = 8;
        let columns = 512;
        let n_rows = 12;
        let row_bytes = 220; // 512 / 256 * 110 = 2 Q3K blocks
        let input_values: Vec<f32> = (0..(m * columns)).map(|i| ((i as f32 * 0.0031) + 0.13).sin()).collect();
        let rounded_input: Vec<f32> = input_values.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let input = ctx.tensor_from_f32(&input_values, m, columns).unwrap();
        let mut weights = vec![0u8; n_rows * row_bytes];
        for row in 0..n_rows {
            let block_a = &mut weights[row * row_bytes..row * row_bytes + 110];
            encode_q3k_block(block_a, row * 2);
            let block_b = &mut weights[row * row_bytes + 110..row * row_bytes + 220];
            encode_q3k_block(block_b, row * 2 + 1);
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 11, row_bytes, n_rows, columns).unwrap();
        let actual_flat = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(11, &weights, n_rows * columns).unwrap();
        for mi in 0..m {
            for ni in 0..n_rows {
                let expected: f32 = (0..columns).map(|k| decoded[ni * columns + k] * rounded_input[mi * columns + k]).sum();
                let actual_val = actual_flat[mi * n_rows + ni];
                let err = (actual_val - expected).abs();
                assert!(err <= 0.05, "fused q3k m={mi} n={ni} actual={actual_val} expected={expected} err={err}");
            }
        }
    }

    #[test]
    fn q6k_fused_gemm_matches_dequantized_matmul() {
        let ctx = MetalContext::new_default().unwrap();
        let m = 8;
        let columns = 512;
        let n_rows = 12;
        let row_bytes = 420; // 512 / 256 * 210 = 2 Q6K blocks
        let input_values: Vec<f32> = (0..(m * columns)).map(|i| ((i as f32 * 0.0029) + 0.07).cos()).collect();
        let rounded_input: Vec<f32> = input_values.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let input = ctx.tensor_from_f32(&input_values, m, columns).unwrap();
        let mut weights = vec![0u8; n_rows * row_bytes];
        for row in 0..n_rows {
            let block_a = &mut weights[row * row_bytes..row * row_bytes + 210];
            encode_q6k_block(block_a, row * 2);
            let block_b = &mut weights[row * row_bytes + 210..row * row_bytes + 420];
            encode_q6k_block(block_b, row * 2 + 1);
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 14, row_bytes, n_rows, columns).unwrap();
        let actual_flat = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(14, &weights, n_rows * columns).unwrap();
        for mi in 0..m {
            for ni in 0..n_rows {
                let expected: f32 = (0..columns).map(|k| decoded[ni * columns + k] * rounded_input[mi * columns + k]).sum();
                let actual_val = actual_flat[mi * n_rows + ni];
                let err = (actual_val - expected).abs();
                assert!(err <= 0.05, "fused q6k m={mi} n={ni} actual={actual_val} expected={expected} err={err}");
            }
        }
    }

    #[test]
    fn iq4nl_fused_gemm_matches_dequantized_matmul() {
        let ctx = MetalContext::new_default().unwrap();
        let m = 8;
        let columns = 512;
        let n_rows = 12;
        let row_bytes = 288; // 512 / 32 * 18
        let input_values: Vec<f32> = (0..(m * columns)).map(|i| ((i as f32 * 0.0037) + 0.11).sin()).collect();
        let rounded_input: Vec<f32> = input_values.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let input = ctx.tensor_from_f32(&input_values, m, columns).unwrap();
        let mut weights = vec![0u8; n_rows * row_bytes];
        for row in 0..n_rows {
            for block in 0..columns / 32 {
                let offset = row * row_bytes + block * 18;
                weights[offset..offset + 2].copy_from_slice(&f16::from_f32(0.015625 + row as f32 * 0.0009765625).to_bits().to_le_bytes());
                for (index, value) in weights[offset + 2..offset + 18].iter_mut().enumerate() {
                    *value = index.wrapping_mul(29).wrapping_add(row * 11).wrapping_add(block * 7) as u8;
                }
            }
        }
        let blob = ctx.resident_byte_weight_buffer(&weights);
        let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 20, row_bytes, n_rows, columns).unwrap();
        let actual_flat = ctx.tensor_to_f32(&actual);
        let decoded = crate::weight::codec::ggml::dequantize(20, &weights, n_rows * columns).unwrap();
        let mut error_squared = 0.0f32;
        let mut reference_squared = 0.0f32;
        let mut max_error = 0.0f32;
        for mi in 0..m {
            for ni in 0..n_rows {
                let expected: f32 = (0..columns).map(|k| decoded[ni * columns + k] * rounded_input[mi * columns + k]).sum();
                let actual_val = actual_flat[mi * n_rows + ni];
                let err = (actual_val - expected).abs();
                error_squared += err * err;
                reference_squared += expected * expected;
                max_error = max_error.max(err);
                assert!(err <= 0.05 + expected.abs() * 0.01, "fused iq4nl m={mi} n={ni} actual={actual_val} expected={expected} err={err}");
            }
        }
        println!("[iq4nl-oracle] rel_l2={} max_error={max_error}", (error_squared / reference_squared).sqrt());
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
mod gguf_quant_tests {
    use super::*;

    pub(super) fn block(tensor_type: u32) -> Vec<u8> {
        let mut bytes = vec![0; metal_block_layout(tensor_type).unwrap().1];
        match tensor_type {
            2 => {
                // Q4_0 平铺 block:d(2B) + qs[16B nibble 对] = 18B per 32 weights
                bytes[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                for (index, value) in bytes[2..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(13) as u8;
                }
            }
            8 => {
                bytes[..2].copy_from_slice(&0x3000u16.to_le_bytes());
                for (index, value) in bytes[2..].iter_mut().enumerate() {
                    *value = (index as i8 - 16) as u8;
                }
            }
            11 => {
                for (index, value) in bytes[..108].iter_mut().enumerate() {
                    *value = index.wrapping_mul(37) as u8;
                }
                bytes[108..].copy_from_slice(&0x2400u16.to_le_bytes());
            }
            12 | 13 => {
                bytes[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                bytes[2..4].copy_from_slice(&0x2000u16.to_le_bytes());
                for (index, value) in bytes[4..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(29) as u8;
                }
            }
            14 => {
                for (index, value) in bytes[..208].iter_mut().enumerate() {
                    *value = index.wrapping_mul(13) as u8;
                }
                bytes[208..].copy_from_slice(&0x2400u16.to_le_bytes());
            }
            22 => {
                bytes[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                for (index, value) in bytes[2..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(11) as u8;
                }
            }
            18 => {
                bytes[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                for (index, value) in bytes[2..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(17) as u8;
                }
            }
            23 => {
                bytes[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                bytes[2..4].copy_from_slice(&2u16.to_le_bytes());
                bytes[4] = 1;
                for (index, value) in bytes[8..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(19) as u8;
                }
            }
            20 => {
                // IQ4_NL 平铺 block:d(2B) + qs[16B] = 18B per 32 weights
                bytes[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                for (index, value) in bytes[2..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(7) as u8;
                }
            }
            21 => {
                bytes[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                for (index, value) in bytes[2..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(23) as u8;
                }
            }
            17 => {
                bytes[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                for (index, value) in bytes[2..].iter_mut().enumerate() {
                    *value = index.wrapping_mul(31) as u8;
                }
            }
            _ => unreachable!(),
        }
        bytes
    }

    fn bf16_row(columns: usize) -> Vec<u8> {
        (0..columns).flat_map(|index| half::bf16::from_f32((index as f32 - 8.0) / 16.0).to_le_bytes()).collect()
    }

    fn mxfp4_row() -> Vec<u8> {
        let mut bytes = vec![0u8; 17];
        bytes[0] = 127;
        for (index, value) in bytes[1..].iter_mut().enumerate() {
            *value = ((index + 1) as u8 & 7) | ((((index + 3) as u8 & 7) | 8) << 4);
        }
        bytes
    }

    #[test]
    fn iq3_dual_gemv_matches_separate_dispatches() {
        let ctx = MetalContext::new_default().unwrap();
        let columns = 256usize;
        let first_rows = 13usize;
        let second_rows = 13usize;
        let first_row = block(18);
        let second_row = block(21);
        let first_bytes = (0..first_rows).flat_map(|_| first_row.iter().copied()).collect::<Vec<_>>();
        let second_bytes = (0..second_rows).flat_map(|_| second_row.iter().copied()).collect::<Vec<_>>();
        let first_blob = ctx.resident_byte_weight_buffer(&first_bytes);
        let second_blob = ctx.resident_byte_weight_buffer(&second_bytes);
        let input_values = (0..columns).map(|index| ((index as f32) * 0.03125).sin()).collect::<Vec<_>>();
        let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
        let expected_first = gguf_matmul_tensor_resident(&ctx, &input, &first_blob, 18, first_row.len(), first_rows, columns).unwrap();
        let expected_second = gguf_matmul_tensor_resident(&ctx, &input, &second_blob, 21, second_row.len(), second_rows, columns).unwrap();
        let (actual_first, actual_second) = gguf_dual_gemv_iq3_tensor(&ctx, &input, &first_blob, 18, first_row.len(), first_rows, &second_blob, 21, second_row.len(), second_rows, columns).unwrap();
        assert_eq!(ctx.tensor_to_f32(&actual_first), ctx.tensor_to_f32(&expected_first));
        assert_eq!(ctx.tensor_to_f32(&actual_second), ctx.tensor_to_f32(&expected_second));
    }

    #[test]
    fn gguf_bf16和mxfp4_matvec_matches_cpu() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        for (tensor_type, columns, bytes) in [(30, 32, bf16_row(32)), (39, 32, mxfp4_row())] {
            let input: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.125).cos()).collect();
            let rounded: Vec<f32> = input.iter().map(|&value| f16::from_f32(value).to_f32()).collect();
            let mut expected = [0.0];
            crate::kernel::cpu::ggml_quant::matvec(tensor_type, &bytes, 1, columns, &rounded, &mut expected).unwrap();
            let input = ctx.tensor_from_f32(&input, 1, columns).unwrap();
            let blob = ctx.shared_buffer(&bytes);
            let output = gguf_matmul_tensor_resident(&ctx, &input, &blob, tensor_type, bytes.len(), 1, columns).unwrap();
            let actual = ctx.read_f16_to_f32(&output.buffer, 1)[0];
            let tolerance = 0.05 + expected[0].abs() * 0.005;
            assert!((actual - expected[0]).abs() <= tolerance, "type={tensor_type}: actual={actual}, expected={}", expected[0]);
        }
    }

    #[test]
    fn gguf_quantized_matvec_matches_cpu() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        for tensor_type in [8, 11, 12, 13, 14, 17, 18, 20, 21, 22, 23] {
            let columns = if matches!(tensor_type, 8 | 20) { 32 } else { 256 };
            let input: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.03125).sin()).collect();
            let rounded: Vec<f32> = input.iter().map(|&value| f16::from_f32(value).to_f32()).collect();
            let bytes = block(tensor_type);
            let mut expected = [0.0];
            crate::kernel::cpu::ggml_quant::matvec(tensor_type, &bytes, 1, columns, &rounded, &mut expected).unwrap();
            let input = ctx.tensor_from_f32(&input, 1, columns).unwrap();
            let blob = ctx.shared_buffer(&bytes);
            let output = gguf_matmul_tensor_resident(&ctx, &input, &blob, tensor_type, bytes.len(), 1, columns).unwrap();
            let actual = ctx.read_f16_to_f32(&output.buffer, 1)[0];
            let tolerance = 0.1 + expected[0].abs() * 0.003;
            assert!((actual - expected[0]).abs() <= tolerance, "type={tensor_type}: actual={actual}, expected={}", expected[0]);
        }
    }

    #[test]
    fn gguf_quantized_multirow_matches_cpu() {
        let ctx = MetalContext::new_default().unwrap();
        // 11 行跨 grid.y 的 8 行批边界(第二批只有 3 行);15..18 覆盖 MiniCPM5
        // 在 17 行观察到的 prefill 分叉(2026-08-26 诊断)
        for rows in [3, 11, 15, 16, 17, 18, GGUF_PREFILL_MPS_ROWS] {
            for tensor_type in [8, 11, 12, 13, 14, 17, 18, 20, 21, 22, 23] {
                let columns = if matches!(tensor_type, 8 | 20) { 32 } else { 256 };
                let input: Vec<f32> = (0..rows * columns).map(|index| ((index % columns) as f32 * 0.03125).sin() * (1.0 + (index / columns) as f32 * 0.25)).collect();
                let rounded: Vec<f32> = input.iter().map(|&value| f16::from_f32(value).to_f32()).collect();
                let bytes = block(tensor_type);
                let mut expected = vec![0.0; rows];
                for row in 0..rows {
                    crate::kernel::cpu::ggml_quant::matvec(tensor_type, &bytes, 1, columns, &rounded[row * columns..(row + 1) * columns], &mut expected[row..row + 1]).unwrap();
                }
                let input = ctx.tensor_from_f32(&input, rows, columns).unwrap();
                let blob = ctx.shared_buffer(&bytes);
                let output = gguf_matmul_tensor_resident(&ctx, &input, &blob, tensor_type, bytes.len(), 1, columns).unwrap();
                let actual = ctx.read_f16_to_f32(&output.buffer, rows);
                for (actual, expected) in actual.iter().zip(expected) {
                    let tolerance = 0.1 + expected.abs() * 0.003;
                    assert!((actual - expected).abs() <= tolerance, "rows={rows}, type={tensor_type}: actual={actual}, expected={expected}");
                }
            }
        }
    }

    #[test]
    fn gguf_gated_gemv_matches_unfused() {
        let ctx = MetalContext::new_default().unwrap();
        for (tensor_type, columns, bytes) in [
            (8, 32, block(8)),
            (2, 32, block(2)),
            (11, 256, block(11)),
            (12, 256, block(12)),
            (13, 256, block(13)),
            (14, 256, block(14)),
            (20, 32, block(20)),
            (21, 256, block(21)),
            (22, 256, block(22)),
            (23, 256, block(23)),
            (30, 32, bf16_row(32)),
            (39, 32, mxfp4_row()),
        ] {
            let input: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.03125).sin()).collect();
            let input = ctx.tensor_from_f32(&input, 1, columns).unwrap();
            let gate_blob = ctx.shared_buffer(&bytes);
            let up_blob = ctx.shared_buffer(&bytes);
            let gate = gguf_matmul_tensor_resident(&ctx, &input, &gate_blob, tensor_type, bytes.len(), 1, columns).unwrap();
            let up = gguf_matmul_tensor_resident(&ctx, &input, &up_blob, tensor_type, bytes.len(), 1, columns).unwrap();
            let expected = gated_activation_tensor(&ctx, &gate, &up, &Activation::Silu).unwrap();
            let actual = gguf_gated_gemv_tensor_resident(&ctx, &input, &gate_blob, tensor_type, bytes.len(), 1, columns, &up_blob, tensor_type, bytes.len(), 1, columns, &Activation::Silu).unwrap();
            let actual = ctx.read_f16_to_f32(&actual.buffer, 1);
            let expected = ctx.read_f16_to_f32(&expected.buffer, 1);
            if tensor_type == 20 {
                // iq4nl fused(每 lane 整 block)与 unfused(16 lane x 16 值)累加
                // 顺序不同,F16 输出允许 1 ulp 级差异
                assert!((actual[0] - expected[0]).abs() <= 1e-2 * (1.0 + expected[0].abs()), "type={tensor_type}: actual={} expected={}", actual[0], expected[0]);
            } else {
                assert_eq!(actual, expected, "type={tensor_type}",);
            }
        }
    }

    #[test]
    fn gated_iq3s_multirow_gemv_matches_cpu() {
        // 3..8 行曾是 float2 动态索引的越界 UB;9..31 行曾落入 gemm_rows 慢路径。
        // 覆盖两组 grid.y 批次与权重行 tile 边界(130 行 = 16 组 + 尾组 2 行)。
        let ctx = MetalContext::new_default().unwrap();
        let columns = 512;
        let weight_rows = 130;
        let row_bytes = metal_block_layout(21).unwrap().1 * (columns / 256);
        let matrix = |seed: u8| -> (Vec<u8>, Vec<f32>) {
            let mut bytes = Vec::with_capacity(weight_rows * row_bytes);
            for row in 0..weight_rows {
                for chunk in 0..columns / 256 {
                    let mut one = block(21);
                    // 行/块间变化,任何行偏移/子块映射错误都会被逐行对比发现
                    one[2 + (row + chunk) % 64] ^= ((row * 31 + chunk) as u8).wrapping_mul(37) | seed;
                    bytes.extend_from_slice(&one);
                }
            }
            let decoded = crate::weight::codec::ggml::dequantize(21, &bytes, weight_rows * columns).unwrap();
            (bytes, decoded)
        };
        let (gate_bytes, gate_decoded) = matrix(1);
        let (up_bytes, up_decoded) = matrix(3);
        let gate_blob = ctx.shared_buffer(&gate_bytes);
        let up_blob = ctx.shared_buffer(&up_bytes);
        for input_rows in [3usize, 5, 11, 16] {
            let input_values: Vec<f32> = (0..input_rows * columns).map(|index| ((index % columns) as f32 * 0.03125).sin() * (1.0 + (index / columns) as f32 * 0.25)).collect();
            let input = ctx.tensor_from_f32(&input_values, input_rows, columns).unwrap();
            let actual = gguf_gated_gemv_tensor_resident(&ctx, &input, &gate_blob, 21, row_bytes, weight_rows, columns, &up_blob, 21, row_bytes, weight_rows, columns, &Activation::Silu).unwrap();
            let actual_values = ctx.read_f16_to_f32(&actual.buffer, input_rows * weight_rows);
            for ir in 0..input_rows {
                for row in 0..weight_rows {
                    let dot = |decoded: &[f32]| -> f32 { decoded[row * columns..(row + 1) * columns].iter().zip(&input_values[ir * columns..(ir + 1) * columns]).map(|(weight, value)| weight * value).sum() };
                    let gate = dot(&gate_decoded);
                    let up = dot(&up_decoded);
                    let expected = gate / (1.0 + (-gate).exp()) * up;
                    let value = actual_values[ir * weight_rows + row];
                    assert!((value - expected).abs() < 0.05 * (1.0 + expected.abs()), "input_rows={input_rows} row={row}: {value} vs {expected}");
                }
            }
        }
    }

    #[test]
    fn iq_packed_gemv_matches_cpu_decode() {
        let ctx = MetalContext::new_default().unwrap();
        for tensor_type in [13u32, 21, 23] {
            let columns = 512; // 2 块
            let rows = 130; // 非 4 整除,覆盖 packed 行边界
            // iq4xs 多行输入(11 行批内截断、31 行跨 8 行批边界)与 130 权重行 tile 的组合
            let input_row_counts: &[usize] = if tensor_type == 23 { &[1, 11, 31] } else { &[1] };
            for &input_rows in input_row_counts {
                let row_bytes = metal_block_layout(tensor_type).unwrap().1 * (columns / 256);
                let mut bytes = Vec::with_capacity(rows * row_bytes);
                for row in 0..rows {
                    for chunk in 0..columns / 256 {
                        let mut one = block(tensor_type);
                        // 行/块间变化,任何行偏移/子块映射错误都会被逐行对比发现
                        one[2 + (row + chunk) % 64] ^= ((row * 31 + chunk) as u8).wrapping_mul(37) | 1;
                        if tensor_type == 13 {
                            // Q5_K 的 dummy d/dmin 会把 F16 输出推到饱和,缩到小 scale。
                            one[0] = 0x11;
                            one[1] = 0x11;
                            one[2] = 0x11;
                            one[3] = 0x11;
                        }
                        bytes.extend_from_slice(&one);
                    }
                }
                let input_values: Vec<f32> = (0..input_rows * columns).map(|index| ((index % columns) as f32 * 0.03125).sin() * (1.0 + (index / columns) as f32 * 0.25)).collect();
                let input = ctx.tensor_from_f32(&input_values, input_rows, columns).unwrap();
                let blob = ctx.shared_buffer(&bytes);
                let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, tensor_type, row_bytes, rows, columns).unwrap();
                let decoded = crate::weight::codec::ggml::dequantize(tensor_type, &bytes, rows * columns).unwrap();
                let actual_values = ctx.read_f16_to_f32(&actual.buffer, input_rows * rows);
                for ir in 0..input_rows {
                    for row in 0..rows {
                        let expected: f32 = decoded[row * columns..(row + 1) * columns].iter().zip(&input_values[ir * columns..(ir + 1) * columns]).map(|(weight, value)| weight * value).sum();
                        let value = actual_values[ir * rows + row];
                        assert!((value - expected).abs() < 0.05 * (1.0 + expected.abs()), "type={tensor_type} input_rows={input_rows} row={row}: {value} vs {expected}");
                    }
                }
            }
        }
    }

    #[test]
    fn iq4nl_gemv_matches_cpu_decode() {
        let ctx = MetalContext::new_default().unwrap();
        let columns = 512; // 16 个 32 值平铺 block,覆盖跨 block 边界
        let rows = 130; // 非 8 整除,覆盖 packed 行 tile 边界
        let row_bytes = metal_block_layout(20).unwrap().1 * (columns / 32);
        let mut bytes = Vec::with_capacity(rows * row_bytes);
        for row in 0..rows {
            for chunk in 0..columns / 32 {
                let mut one = block(20);
                // 行/块间变化,任何行偏移/block 映射错误都会被逐行对比发现
                one[2 + (row + chunk) % 16] ^= ((row * 31 + chunk) as u8).wrapping_mul(37) | 1;
                bytes.extend_from_slice(&one);
            }
        }
        // 1 行走 1r 专用 kernel;11/31 行走多行 gemv(批内截断与跨 8 行批边界)
        for input_rows in [1usize, 11, 31] {
            let input_values: Vec<f32> = (0..input_rows * columns).map(|index| ((index % columns) as f32 * 0.03125).sin() * (1.0 + (index / columns) as f32 * 0.25)).collect();
            let input = ctx.tensor_from_f32(&input_values, input_rows, columns).unwrap();
            let blob = ctx.shared_buffer(&bytes);
            let actual = gguf_matmul_tensor_resident(&ctx, &input, &blob, 20, row_bytes, rows, columns).unwrap();
            let decoded = crate::weight::codec::ggml::dequantize(20, &bytes, rows * columns).unwrap();
            let actual_values = ctx.read_f16_to_f32(&actual.buffer, input_rows * rows);
            for ir in 0..input_rows {
                for row in 0..rows {
                    let expected: f32 = decoded[row * columns..(row + 1) * columns].iter().zip(&input_values[ir * columns..(ir + 1) * columns]).map(|(weight, value)| weight * value).sum();
                    let value = actual_values[ir * rows + row];
                    assert!((value - expected).abs() < 0.05 * (1.0 + expected.abs()), "input_rows={input_rows} row={row}: {value} vs {expected}");
                }
            }
        }
    }

    #[test]
    fn gguf_indexed_experts_match_ordered_accumulation() {
        let ctx = MetalContext::new_default().unwrap();
        let ids = [0u32, 1];
        let route_weights = [0.25f32, 0.75];
        let top_k = ids.len();
        let id_buffer = ctx.shared_buffer(as_bytes(&ids));
        let route_buffer = ctx.shared_buffer(as_bytes(&route_weights));

        for tensor_type in [8, 11, 12, 13, 14, 17, 18, 21, 22, 23, 30, 39] {
            let columns = if matches!(tensor_type, 8 | 30 | 39) {
                32
            } else if tensor_type == 11 {
                512
            } else {
                256
            };
            let rows = columns;
            let input_values = (0..columns).map(|index| (index as f32 * 0.03125).sin()).collect::<Vec<_>>();
            let input = ctx.tensor_from_f32(&input_values, 1, columns).unwrap();
            let first_block = match tensor_type {
                30 => bf16_row(columns),
                39 => mxfp4_row(),
                11 => [block(tensor_type), block(tensor_type)].concat(),
                _ => block(tensor_type),
            };
            let mut second_block = first_block.clone();
            let changed = second_block.len() / 2;
            second_block[changed] ^= 0x5a;
            let expert_blocks = [first_block, second_block];
            let expert_matrices = expert_blocks.iter().map(|block| (0..rows).flat_map(|_| block.iter().copied()).collect::<Vec<_>>()).collect::<Vec<_>>();
            let packed = expert_matrices.iter().flat_map(|matrix| matrix.iter().copied()).collect::<Vec<_>>();
            let row_bytes = expert_blocks[0].len();
            let gate = MetalWeight::Gguf { blob: ctx.resident_byte_weight_buffer(&packed), tensor_type, row_bytes, rows, cols: columns };
            let up = MetalWeight::Gguf { blob: ctx.resident_byte_weight_buffer(&packed), tensor_type, row_bytes, rows, cols: columns };
            let down = MetalWeight::Gguf { blob: ctx.resident_byte_weight_buffer(&packed), tensor_type, row_bytes, rows, cols: columns };
            let actual = gguf_indexed_experts_tensor_resident(&ctx, &input, &gate, &up, &down, &id_buffer, &route_buffer, top_k, &Activation::Silu).unwrap();

            let expected_accumulator = crate::kernel::metal::moe::moe_accumulator_zeros(&ctx, 1, columns).unwrap();
            for expert in 0..top_k {
                let blob = ctx.resident_byte_weight_buffer(&expert_matrices[expert]);
                let activated = gguf_gated_gemv_tensor_resident(&ctx, &input, &blob, tensor_type, row_bytes, rows, columns, &blob, tensor_type, row_bytes, rows, columns, &Activation::Silu).unwrap();
                let projected = gguf_matmul_tensor_resident(&ctx, &activated, &blob, tensor_type, row_bytes, rows, columns).unwrap();
                crate::kernel::metal::moe::scatter_add_rows_weighted_f32(&ctx, &expected_accumulator, &projected, &[0], &[route_weights[expert]]).unwrap();
            }
            let expected = crate::kernel::metal::moe::finish_moe_accumulator(&ctx, expected_accumulator, None).unwrap();
            let actual_values = ctx.read_f16_to_f32(&actual.buffer, columns);
            let expected_values = ctx.read_f16_to_f32(&expected.buffer, columns);
            if matches!(tensor_type, 11 | 12 | 22) {
                assert_eq!(actual_values, expected_values, "tensor_type={tensor_type}");
            } else {
                for (column, (actual, expected)) in actual_values.into_iter().zip(expected_values).enumerate() {
                    let error = (actual - expected).abs();
                    assert!(error <= 1e-2 * (1.0 + expected.abs()), "tensor_type={tensor_type} column={column}: actual={actual} expected={expected} error={error}");
                }
            }
        }
    }
}

#[cfg(test)]
mod decode_cast_cache_tests {
    use super::*;

    #[test]
    fn repeated_decode_cast_reuses_buffer() {
        let ctx = MetalContext::new_default().unwrap();
        let input = ctx.tensor_from_f32_preserve(&[1.0, -2.0, 3.0, -4.0], 1, 4).unwrap();
        ctx.set_deferred_waits(true);
        let first = to_f16_tensor(&ctx, &input).unwrap();
        let second = to_f16_tensor(&ctx, &input).unwrap();
        assert_eq!(first.buffer.contents(), second.buffer.contents());
        ctx.synchronize();
        ctx.set_deferred_waits(false);
    }
}

#[cfg(test)]
mod gqa_split_kv_tests {
    use super::*;

    #[test]
    fn gqa_split_kv_matches_cpu() {
        let ctx = MetalContext::new_default().unwrap();
        let (kv_rows, head_count, kv_head_count, head_dim) = (513, 8, 1, 64);
        let query_values: Vec<f32> = (0..head_count * head_dim).map(|index| (index as f32 * 0.03125).sin()).collect();
        let kv_elements = kv_rows * kv_head_count * head_dim;
        let key_values: Vec<f32> = (0..kv_elements).map(|index| (index as f32 * 0.001).cos() * 0.25).collect();
        let value_values: Vec<f32> = (0..kv_elements).map(|index| (index as f32 * 0.003).sin() * 0.5).collect();
        let rounded_query: Vec<f32> = query_values.iter().map(|&value| f16::from_f32(value).to_f32()).collect();
        let rounded_key: Vec<f32> = key_values.iter().map(|&value| f16::from_f32(value).to_f32()).collect();
        let rounded_value: Vec<f32> = value_values.iter().map(|&value| f16::from_f32(value).to_f32()).collect();
        let query = ctx.tensor_from_f32(&query_values, 1, head_count * head_dim).unwrap();
        let key = ctx.tensor_from_f32(&key_values, kv_rows, kv_head_count * head_dim).unwrap();
        let value = ctx.tensor_from_f32(&value_values, kv_rows, kv_head_count * head_dim).unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = crate::kernel::metal::attention::gqa_decode_attention_split_kv_buffers(&ctx, &query, &key.buffer, 0, &value.buffer, 0, kv_rows, 0, 0, head_count, kv_head_count, head_dim, scale, false, None).unwrap();
        let actual = ctx.tensor_to_f32(&output);

        let heads_per_kv = head_count / kv_head_count;
        for head in 0..head_count {
            let kv_head = head / heads_per_kv;
            let query_base = head * head_dim;
            let mut scores = vec![0.0f32; kv_rows];
            for (token, score) in scores.iter_mut().enumerate() {
                let key_base = (token * kv_head_count + kv_head) * head_dim;
                *score = (0..head_dim).map(|dimension| rounded_query[query_base + dimension] * rounded_key[key_base + dimension]).sum::<f32>() * scale;
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|&score| (score - maximum).exp()).collect();
            let denominator: f32 = weights.iter().sum();
            for dimension in 0..head_dim {
                let expected = weights
                    .iter()
                    .enumerate()
                    .map(|(token, &weight)| {
                        let value_base = (token * kv_head_count + kv_head) * head_dim;
                        weight * rounded_value[value_base + dimension]
                    })
                    .sum::<f32>()
                    / denominator;
                let actual = actual[query_base + dimension];
                let tolerance = 0.01 + expected.abs() * 0.01;
                assert!((actual - expected).abs() <= tolerance, "head={head},dim={dimension}: actual={actual}, expected={expected}");
            }
        }
    }
}

#[cfg(test)]
mod gated_activation_tests {
    use super::*;

    #[test]
    fn gelu_tanh_extreme_is_finite_and_matches_cpu() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let ctx = MetalContext::new_default().unwrap();
        let gate = [-57.0, -10.0, -1.0, 0.0, 1.0, 10.0, 57.0];
        let up = [3.0; 7];
        let gate_tensor = ctx.tensor_from_f32(&gate, 1, gate.len()).unwrap();
        let up_tensor = ctx.tensor_from_f32(&up, 1, up.len()).unwrap();
        let output = gated_activation_tensor(&ctx, &gate_tensor, &up_tensor, &Activation::GeluTanh).unwrap();
        let actual = ctx.tensor_to_f32(&output);
        let mut expected = [0.0; 7];
        crate::kernel::cpu::silu::gelu_tanh_mul(&gate, &up, &mut expected);
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(actual.is_finite(), "GELU-tanh 极值输出必须有限");
            let expected = f16::from_f32(expected).to_f32();
            assert!((actual - expected).abs() <= 0.125, "actual={actual}, expected={expected}");
        }
    }
}
#[cfg(all(test, target_os = "macos"))]
mod rows2_gemv_tests {
    use super::*;
    use crate::backend::metal::MetalContext;

    /// multirow gemv 逐行 CPU 参照(rows=1..4):无歧义判定行级正确性。
    #[test]
    fn iq4nl_multirow_rows_scan_cpu() {
        let ctx = MetalContext::new_default().expect("metal");
        let columns = 256usize;
        let row_bytes = columns / 32 * 18;
        let table = [-127.0f32, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0];
        let weight_rows = 8usize;
        let mut weight = Vec::with_capacity(weight_rows * row_bytes);
        let mut weight_f32 = vec![0.0f32; weight_rows * columns];
        for row in 0..weight_rows {
            for block in 0..columns / 32 {
                weight.extend(0x3C00u16.to_le_bytes());
                for j in 0..16usize {
                    let low = (block + j + row) % 16;
                    let high = (block + j + row + 5) % 16;
                    weight.push(((high as u8) << 4) | low as u8);
                    weight_f32[row * columns + block * 32 + j] = table[low];
                    weight_f32[row * columns + block * 32 + 16 + j] = table[high];
                }
            }
        }
        let blob = ctx.shared_buffer(&weight);
        let input_f32: Vec<f32> = (0..4 * columns).map(|index| ((index % 11) as f32 - 5.0) / 5.0).collect();
        for rows in 1usize..=4 {
            let input = ctx.tensor_from_f32(&input_f32[..rows * columns], rows, columns).expect("input");
            let output = ctx.tensor_to_f32(&gguf_matmul_tensor_resident(&ctx, &input, &blob, 20, row_bytes, weight_rows, columns).expect("gemv"));
            for ir in 0..rows {
                for wrow in [0usize, 3, 7] {
                    let expected: f32 = (0..columns).map(|k| input_f32[ir * columns + k] * weight_f32[wrow * columns + k]).sum();
                    let actual = output[ir * weight_rows + wrow];
                    assert!((actual - expected).abs() < expected.abs() * 0.01 + 0.5, "rows={rows} 行 {ir} 权重 {wrow}: {actual} vs CPU {expected}");
                }
            }
            println!("[gemv-scan] rows={rows} 逐行对照通过");
        }
    }

    /// gated iq4nl fused gemv 的 rows=1/2 逐行 CPU 参照:rows=2 的行 1 曾是未初始化
    /// 内存(kernel 只读写行 0)——verify rows=2 行 1 数值错的根因(gemma4-mtp-verify-rows2-bug)。
    #[test]
    fn gated_iq4nl_two_row_gemv_matches_cpu() {
        let ctx = MetalContext::new_default().expect("metal");
        let columns = 256usize;
        let row_bytes = columns / 32 * 18;
        let table = [-127.0f32, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0];
        let weight_rows = 8usize;
        let matrix = |seed: usize| -> (Vec<u8>, Vec<f32>) {
            let mut weight = Vec::with_capacity(weight_rows * row_bytes);
            let mut weight_f32 = vec![0.0f32; weight_rows * columns];
            for row in 0..weight_rows {
                for block in 0..columns / 32 {
                    weight.extend(0x3C00u16.to_le_bytes());
                    for j in 0..16usize {
                        let low = (block + j + row + seed) % 16;
                        let high = (block + j + row + seed + 5) % 16;
                        weight.push(((high as u8) << 4) | low as u8);
                        weight_f32[row * columns + block * 32 + j] = table[low];
                        weight_f32[row * columns + block * 32 + 16 + j] = table[high];
                    }
                }
            }
            (weight, weight_f32)
        };
        let (gate_bytes, gate_f32) = matrix(0);
        let (up_bytes, up_f32) = matrix(7);
        let gate_blob = ctx.shared_buffer(&gate_bytes);
        let up_blob = ctx.shared_buffer(&up_bytes);
        // 小输入让 gate 落在 gelu 有效区间(|gate|<10 与截断分支都覆盖到)
        let input_f32: Vec<f32> = (0..2 * columns).map(|index| ((index % 11) as f32 - 5.0) / 50.0).collect();
        let rounded: Vec<f32> = input_f32.iter().map(|&value| f16::from_f32(value).to_f32()).collect();
        for rows in 1usize..=2 {
            let input = ctx.tensor_from_f32(&input_f32[..rows * columns], rows, columns).expect("input");
            let output = gguf_gated_gemv_tensor_resident(&ctx, &input, &gate_blob, 20, row_bytes, weight_rows, columns, &up_blob, 20, row_bytes, weight_rows, columns, &Activation::GeluTanh).expect("gated");
            let actual = ctx.tensor_to_f32(&output);
            for ir in 0..rows {
                for wrow in 0..weight_rows {
                    let gate: f32 = (0..columns).map(|k| rounded[ir * columns + k] * gate_f32[wrow * columns + k]).sum();
                    let up: f32 = (0..columns).map(|k| rounded[ir * columns + k] * up_f32[wrow * columns + k]).sum();
                    let expected = if gate >= 10.0 {
                        gate * up
                    } else if gate <= -10.0 {
                        0.0
                    } else {
                        0.5 * gate * (1.0 + (0.797_884_56 * (gate + 0.044_715 * gate * gate * gate)).tanh()) * up
                    };
                    let value = actual[ir * weight_rows + wrow];
                    assert!((value - expected).abs() < 0.05 * (1.0 + expected.abs()), "rows={rows} 行 {ir} 权重 {wrow}: {value} vs CPU {expected}");
                }
            }
            println!("[gated-scan] rows={rows} 逐行对照通过");
        }
        // 未多行化的类型 rows=2 必须拒绝(此前静默放行,输出行 1 是未初始化内存)
        let two_row = ctx.tensor_from_f32(&input_f32[..2 * columns], 2, columns).expect("input");
        let xs_row_bytes = columns / 256 * 136;
        let xs_blob = ctx.shared_buffer(&vec![0u8; weight_rows * xs_row_bytes]);
        assert!(gguf_gated_gemv_tensor_resident(&ctx, &two_row, &xs_blob, 23, xs_row_bytes, weight_rows, columns, &xs_blob, 23, xs_row_bytes, weight_rows, columns, &Activation::Silu).is_err(), "iq4xs rows=2 应拒绝 fused gated");
    }
}

#[cfg(all(test, target_os = "macos"))]
mod norm_rows2_tests {
    #[test]
    fn gemma_heads_norm_rows2_matches_split() {
        let ctx = crate::backend::metal::MetalContext::new_default().expect("metal");
        let (heads, dim) = (16usize, 256usize);
        let values: Vec<f32> = (0..2 * heads * dim).map(|i| ((i % 29) as f32 - 14.0) / 7.0).collect();
        let norm_w: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 % 7.0) * 0.1).collect();
        let two = ctx.tensor_from_f32(&values, 2, heads * dim).unwrap();
        let one0 = ctx.tensor_from_f32(&values[..heads * dim], 1, heads * dim).unwrap();
        let one1 = ctx.tensor_from_f32(&values[heads * dim..], 1, heads * dim).unwrap();
        let weight = <crate::backend::metal::MetalContext as crate::backend::BackendResources>::prepare_f32(&ctx, &norm_w, 1, dim).unwrap();
        let out2 = crate::backend::GqaPrefillBackend::gemma_rmsnorm_heads(&ctx, &two, &weight, heads, dim, 1.0e-6).unwrap();
        let out0 = crate::backend::GqaPrefillBackend::gemma_rmsnorm_heads(&ctx, &one0, &weight, heads, dim, 1.0e-6).unwrap();
        let out1 = crate::backend::GqaPrefillBackend::gemma_rmsnorm_heads(&ctx, &one1, &weight, heads, dim, 1.0e-6).unwrap();
        let v2 = ctx.tensor_to_f32(&out2);
        let v0 = ctx.tensor_to_f32(&out0);
        let v1 = ctx.tensor_to_f32(&out1);
        for i in 0..heads * dim {
            assert!((v2[i] - v0[i]).abs() < 0.01 + v0[i].abs() * 0.001, "rows=2 行 0[{i}]: {:+} vs {:+}", v2[i], v0[i]);
            assert!((v2[heads * dim + i] - v1[i]).abs() < 0.01 + v1[i].abs() * 0.001, "rows=2 行 1[{i}]: {:+} vs {:+}", v2[heads * dim + i], v1[i]);
        }
        println!("[norm-scan] gemma_rmsnorm_heads rows=2 逐行对照通过");
    }
}
