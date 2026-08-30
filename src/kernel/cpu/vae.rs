//! VAE 算子 CPU reference 实现。
//!
//! Conv3D / GroupNorm / SiLU 的正确性 reference，用于验证 Metal kernel。
//! 性能不优（纯标量循环），但保证数值正确。

use std::borrow::Cow;

use rayon::prelude::*;
use wide::f32x8;

use super::matmul::{dot, matmul};

const SIMD_LANES: usize = 8;

/// 时空 3D 卷积(causal padding)。
///
/// 输入/输出布局:`[channels, depth, height, width]`。
/// weight 布局:`[out_channels, in_channels, kd, kh, kw]`(行优先展开为
/// `[out_channels, in_channels*kd*kh*kw]`)。
///
/// 实现:对每个输出位置把感受野 patch 展开成 im2col 列(每个 patch 长度
/// `in_channels*kd*kh*kw`,与 weight 行同序),再一次性 `weight × im2col` GEMM。
/// 相比逐输出通道重读输入的标量循环,patch 只 gather 一次,MAC 走 BLAS。
#[allow(clippy::too_many_arguments)]
pub fn conv3d(
    input: &[f32],
    in_channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    weight: &[f32],
    out_channels: usize,
    kernel: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    bias: Option<&[f32]>,
    causal: bool,
) -> Vec<f32> {
    let (kd, kh, kw) = kernel;
    let (sd, sh, sw) = stride;
    let (pd, ph, pw) = padding;

    let time_padding = if causal { pd } else { pd * 2 };
    let out_depth = (depth + time_padding - kd) / sd + 1;
    let out_height = (height + ph * 2 - kh) / sh + 1;
    let out_width = (width + pw * 2 - kw) / sw + 1;
    let taps = in_channels.checked_mul(kd).and_then(|v| v.checked_mul(kh)).and_then(|v| v.checked_mul(kw)).expect("Conv3D taps 溢出");
    let out_positions = out_depth.checked_mul(out_height).and_then(|v| v.checked_mul(out_width)).expect("Conv3D out positions 溢出");
    let hw = height * width;
    let channel_stride = depth * hw;

    // 预计算: 每个 output_position 的 kd*kh*kw 个 input offsets。越界 tap 标记为
    // usize::MAX 作为哨兵,gather 时查表直接跳过;同一 (od, oh, ow) 下 27 个内核位置
    // 对所有 in_channels 共享,故只算一份。offset 表足够紧凑可整块缓存。
    let taps_per_pos = kd * kh * kw;
    let mut offsets = vec![usize::MAX; out_positions * taps_per_pos];
    let ohw = out_height * out_width;
    let depth_signed = depth as isize;
    let height_signed = height as isize;
    let width_signed = width as isize;
    offsets.par_chunks_mut(taps_per_pos).enumerate().for_each(|(pos, row)| {
        let od = pos / ohw;
        let oh = (pos / out_width) % out_height;
        let ow = pos % out_width;
        for fk in 0..kd {
            let id = od as isize * sd as isize + fk as isize - pd as isize;
            for fh in 0..kh {
                let ih = oh as isize * sh as isize + fh as isize - ph as isize;
                for fw in 0..kw {
                    let iw = ow as isize * sw as isize + fw as isize - pw as isize;
                    let t = (fk * kh + fh) * kw + fw;
                    row[t] = if id >= 0 && id < depth_signed && ih >= 0 && ih < height_signed && iw >= 0 && iw < width_signed { (id as usize) * hw + (ih as usize) * width + (iw as usize) } else { usize::MAX };
                }
            }
        }
    });

    // im2col:[out_positions, in_channels * taps_per_pos] 行优先,每行一个输出位置的 patch。
    // 越界 tap 查 usize::MAX 哨兵 → 写 0;in_channels 间 stride 共享一次。
    let mut cols = vec![0.0_f32; out_positions.checked_mul(taps).expect("Conv3D im2col 溢出")];
    cols.par_chunks_mut(taps).enumerate().for_each(|(pos, patch)| {
        let off_pos = &offsets[pos * taps_per_pos..pos * taps_per_pos + taps_per_pos];
        for ic in 0..in_channels {
            let base = ic * channel_stride;
            let base_tap = ic * taps_per_pos;
            for t in 0..taps_per_pos {
                let off = off_pos[t];
                patch[base_tap + t] = if off == usize::MAX { 0.0 } else { input[base + off] };
            }
        }
    });

    // output[oc, pos] = Σ_tap weight[oc, tap] · cols[pos, tap],走 BLAS/分块 matmul。
    let mut output = vec![0.0_f32; out_channels.checked_mul(out_positions).expect("Conv3D 输出溢出")];
    matmul(weight, &cols, out_channels, taps, out_positions, &mut output);

    // bias 与 matmul 结果融合(顺序可换;行内 par 让 out[] 写入分摊到 rayon)。
    if let Some(b) = bias {
        output.par_chunks_mut(out_positions).enumerate().for_each(|(oc, row)| {
            let bias_value = b[oc];
            for value in row.iter_mut() {
                *value += bias_value;
            }
        });
    }
    output
}

/// 标量 Conv3D reference(逐输出位置、逐 tap 累加),仅用于校验 im2col 路径。
#[cfg(test)]
fn conv3d_scalar(
    input: &[f32],
    in_channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    weight: &[f32],
    out_channels: usize,
    kernel: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    bias: Option<&[f32]>,
    causal: bool,
) -> Vec<f32> {
    let (kd, kh, kw) = kernel;
    let (sd, sh, sw) = stride;
    let (pd, ph, pw) = padding;

    let time_padding = if causal { pd } else { pd * 2 };
    let out_depth = (depth + time_padding - kd) / sd + 1;
    let out_height = (height + ph * 2 - kh) / sh + 1;
    let out_width = (width + pw * 2 - kw) / sw + 1;

    let out_size = out_channels * out_depth * out_height * out_width;
    let mut output = vec![0.0_f32; out_size];

    output.par_chunks_mut(out_depth * out_height * out_width).enumerate().for_each(|(oc, out_slice)| {
        for od in 0..out_depth {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut acc = 0.0_f32;
                    for ic in 0..in_channels {
                        for fk in 0..kd {
                            let id = (od * sd + fk) as isize - pd as isize;
                            if id < 0 || id >= depth as isize {
                                continue;
                            }
                            for fh in 0..kh {
                                let ih = (oh * sh + fh) as isize - ph as isize;
                                if ih < 0 || ih >= height as isize {
                                    continue;
                                }
                                for fw in 0..kw {
                                    let iw = (ow * sw + fw) as isize - pw as isize;
                                    if iw < 0 || iw >= width as isize {
                                        continue;
                                    }
                                    let w_idx = ((oc * in_channels + ic) * kd + fk) * kh * kw + fh * kw + fw;
                                    let i_idx = (ic * depth + id as usize) * height * width + ih as usize * width + iw as usize;
                                    acc += input[i_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    if let Some(b) = bias {
                        acc += b[oc];
                    }
                    out_slice[od * out_height * out_width + oh * out_width + ow] = acc;
                }
            }
        }
    });

    output
}

/// GroupNorm:按通道分组归一化。输入布局 `[channels, spatial]`,每 `num_groups` 个
/// 连续通道为一组。rayon 并行 over groups,f32x8 做 mean/var 归约与逐通道归一化。
pub fn group_norm(
    input: &[f32],
    channels: usize,
    spatial: usize, // depth × height × width
    num_groups: usize,
    eps: f32,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    assert!(channels.is_multiple_of(num_groups), "channels {channels} 不能被 num_groups {num_groups} 整除");
    let group_size = channels / num_groups;
    let group_elements = group_size * spatial;
    let mut output = vec![0.0_f32; input.len()];

    output.par_chunks_mut(group_elements).enumerate().for_each(|(g, out_group)| {
        let group = &input[g * group_elements..(g + 1) * group_elements];
        let mean = sum_v(group) / group_elements as f32;
        let mean_v = f32x8::splat(mean);
        let mut acc = f32x8::splat(0.0);
        for chunk in group.chunks_exact(SIMD_LANES) {
            let v = f32x8::from(<[f32; SIMD_LANES]>::try_from(chunk).unwrap());
            let diff = v - mean_v;
            acc += diff * diff;
        }
        let mut var = acc.reduce_add();
        for &value in &group[group.len() / SIMD_LANES * SIMD_LANES..] {
            let diff = value - mean;
            var += diff * diff;
        }
        let rstd = 1.0 / (var / group_elements as f32 + eps).sqrt();
        let rstd_v = f32x8::splat(rstd);
        for c in 0..group_size {
            let channel = g * group_size + c;
            let weight_v = f32x8::splat(weight[channel]);
            let bias_v = f32x8::splat(bias[channel]);
            let base = c * spatial;
            let channel_in = &group[base..base + spatial];
            let channel_out = &mut out_group[base..base + spatial];
            let mut si = 0;
            while si + SIMD_LANES <= spatial {
                let v = f32x8::from(<[f32; SIMD_LANES]>::try_from(&channel_in[si..si + SIMD_LANES]).unwrap());
                let normalized: [f32; SIMD_LANES] = ((v - mean_v) * rstd_v * weight_v + bias_v).into();
                channel_out[si..si + SIMD_LANES].copy_from_slice(&normalized);
                si += SIMD_LANES;
            }
            while si < spatial {
                channel_out[si] = (channel_in[si] - mean) * rstd * weight[channel] + bias[channel];
                si += 1;
            }
        }
    });

    output
}

/// f32x8 求和(含标量尾)。
fn sum_v(slice: &[f32]) -> f32 {
    let mut acc = f32x8::splat(0.0);
    for chunk in slice.chunks_exact(SIMD_LANES) {
        acc += f32x8::from(<[f32; SIMD_LANES]>::try_from(chunk).unwrap());
    }
    let mut sum = acc.reduce_add();
    for &value in &slice[slice.len() / SIMD_LANES * SIMD_LANES..] {
        sum += value;
    }
    sum
}

/// SiLU 激活:`x * sigmoid(x)`,f32x8。
pub fn silu(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; input.len()];
    let one = f32x8::splat(1.0);
    for (chunk, out_chunk) in input.chunks_exact(SIMD_LANES).zip(output.chunks_exact_mut(SIMD_LANES)) {
        let v = f32x8::from(<[f32; SIMD_LANES]>::try_from(chunk).unwrap());
        let sigmoid = one / (one + (-v).exp());
        let value: [f32; SIMD_LANES] = (v * sigmoid).into();
        out_chunk.copy_from_slice(&value);
    }
    for (&value, out) in input[input.len() / SIMD_LANES * SIMD_LANES..].iter().zip(&mut output[input.len() / SIMD_LANES * SIMD_LANES..]) {
        *out = value / (1.0 + (-value).exp());
    }
    output
}

/// Pixel Shuffle：`[channels*r^2, height, width] -> [channels, height*r, width*r]`。
pub fn pixel_shuffle(input: &[f32], channels: usize, height: usize, width: usize, upscale: usize) -> Vec<f32> {
    let out_height = height * upscale;
    let out_width = width * upscale;
    let mut output = vec![0.0; channels * out_height * out_width];
    for channel in 0..channels {
        for out_h in 0..out_height {
            for out_w in 0..out_width {
                let in_channel = channel * upscale * upscale + (out_h % upscale) * upscale + out_w % upscale;
                let input_index = (in_channel * height + out_h / upscale) * width + out_w / upscale;
                let output_index = (channel * out_height + out_h) * out_width + out_w;
                output[output_index] = input[input_index];
            }
        }
    }
    output
}

/// AdaLN 支持逐行参数或单行参数对全部 token 广播。
pub fn adaln_modulate(input: &[f32], shift: &[f32], scale: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let elements = rows.checked_mul(cols).ok_or("AdaLN shape 溢出")?;
    if input.len() != elements || shift.len() != scale.len() || (shift.len() != cols && shift.len() != elements) {
        return Err(format!("AdaLN shape 不兼容: input={} shift={} scale={} rows={rows} cols={cols}", input.len(), shift.len(), scale.len()));
    }
    Ok((0..elements)
        .map(|index| {
            let modulation = if shift.len() == cols { index % cols } else { index };
            input[index] * (1.0 + scale[modulation]) + shift[modulation]
        })
        .collect())
}

/// per-head RMSNorm 单一实现:`weight = None` 时不乘缩放(unit 版),
/// `weight_offset` 用于 Gemma 式 `(w + offset)`。eps 校验只在带 weight 的
/// 加权路径做,与合并前两个入口的校验行为一致。
pub fn rmsnorm_heads_with(input: &[f32], weight: Option<&[f32]>, head_count: usize, head_dim: usize, eps: f32, weight_offset: f32) -> Result<Vec<f32>, String> {
    let columns = head_count.checked_mul(head_dim).ok_or("per-head RMSNorm columns 溢出")?;
    if head_count == 0 || head_dim == 0 || !input.len().is_multiple_of(columns) || weight.is_some_and(|weight| weight.len() != head_dim) || weight.is_some() && (!eps.is_finite() || eps <= 0.0) {
        return Err(format!("per-head RMSNorm shape 不兼容: input={} weight={} heads={head_count} head_dim={head_dim} eps={eps}", input.len(), weight.map_or(0, <[f32]>::len)));
    }
    let mut output = vec![0.0; input.len()];
    // head 内 sum-of-squares 与缩放按 8 通道 SIMD;块结构与 rmsnorm.rs 一致。
    for (source, target) in input.chunks_exact(head_dim).zip(output.chunks_exact_mut(head_dim)) {
        let lanes = head_dim / 8 * 8;
        let mut squares = f32x8::splat(0.0);
        for chunk in source.chunks_exact(8) {
            let value = f32x8::from(<[f32; 8]>::try_from(chunk).unwrap());
            squares += value * value;
        }
        let mut sum = squares.reduce_add();
        for &value in &source[lanes..] {
            sum += value * value;
        }
        let scalar_inverse = (sum / head_dim as f32 + eps).sqrt().recip();
        let inverse = f32x8::splat(scalar_inverse);
        let offset = f32x8::splat(weight_offset);
        for (chunk_index, (chunk, target)) in source.chunks_exact(8).zip(target.chunks_exact_mut(8)).enumerate() {
            let value = f32x8::from(<[f32; 8]>::try_from(chunk).unwrap());
            let scaled: [f32; 8] = match weight {
                Some(weight) => {
                    let weight = f32x8::from(<[f32; 8]>::try_from(&weight[chunk_index * 8..chunk_index * 8 + 8]).unwrap());
                    (value * inverse * (weight + offset)).into()
                }
                None => (value * inverse).into(),
            };
            target.copy_from_slice(&scaled);
        }
        for column in lanes..head_dim {
            target[column] = source[column] * scalar_inverse * weight.map_or(1.0, |weight| weight[column] + weight_offset);
        }
    }
    Ok(output)
}

pub fn rmsnorm_heads(input: &[f32], weight: &[f32], head_count: usize, head_dim: usize, eps: f32) -> Result<Vec<f32>, String> {
    rmsnorm_heads_with(input, Some(weight), head_count, head_dim, eps, 0.0)
}

pub fn full_attention(query: &[f32], key: &[f32], value: &[f32], head_count: usize, head_dim: usize, score_scale: f32) -> Result<Vec<f32>, String> {
    let columns = head_count.checked_mul(head_dim).ok_or("full attention columns 溢出")?;
    if head_count == 0 || head_dim == 0 || !query.len().is_multiple_of(columns) || query.len() != key.len() || query.len() != value.len() || !score_scale.is_finite() || score_scale <= 0.0 {
        return Err(format!("full attention shape 不兼容: q={} k={} v={} heads={head_count} head_dim={head_dim} scale={score_scale}", query.len(), key.len(), value.len(),));
    }
    let rows = query.len() / columns;
    let elements = query.len();
    let mut output = vec![0.0; elements];
    let mut scores = vec![0.0; rows];
    for query_row in 0..rows {
        for head in 0..head_count {
            let query_base = query_row * columns + head * head_dim;
            let mut maximum = f32::NEG_INFINITY;
            for key_row in 0..rows {
                let key_base = key_row * columns + head * head_dim;
                let score = query[query_base..query_base + head_dim].iter().zip(&key[key_base..key_base + head_dim]).map(|(left, right)| left * right).sum::<f32>() * score_scale;
                scores[key_row] = score;
                maximum = maximum.max(score);
            }
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - maximum).exp();
                    *score
                })
                .sum::<f32>();
            for (key_row, &score) in scores.iter().enumerate() {
                let probability = score / denominator;
                let value_base = key_row * columns + head * head_dim;
                for column in 0..head_dim {
                    output[query_base + column] += probability * value[value_base + column];
                }
            }
        }
    }
    Ok(output)
}

pub fn add_row_bias(input: &[f32], bias: &[f32]) -> Result<Vec<f32>, String> {
    let cols = bias.len();
    if cols == 0 || !input.len().is_multiple_of(cols) {
        return Err(format!("row bias shape 不兼容: input={} bias={cols}", input.len()));
    }
    Ok(input.iter().enumerate().map(|(index, value)| value + bias[index % cols]).collect())
}

pub fn concat_rows(left: &[f32], right: &[f32], cols: usize) -> Result<Vec<f32>, String> {
    if cols == 0 || !left.len().is_multiple_of(cols) || !right.len().is_multiple_of(cols) {
        return Err(format!("row concat shape 不兼容: left={} right={} cols={cols}", left.len(), right.len()));
    }
    let mut output = Vec::with_capacity(left.len() + right.len());
    output.extend_from_slice(left);
    output.extend_from_slice(right);
    Ok(output)
}

pub fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32], cols: usize, eps: f32) -> Result<Vec<f32>, String> {
    if cols == 0 || !input.len().is_multiple_of(cols) || weight.len() != cols || bias.len() != cols || eps <= 0.0 {
        return Err(format!("VAE LayerNorm shape 不兼容: input={} weight={} bias={} cols={cols} eps={eps}", input.len(), weight.len(), bias.len()));
    }
    let rows = input.len() / cols;
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let start = row * cols;
        let values = &input[start..start + cols];
        let mean = values.iter().sum::<f32>() / cols as f32;
        let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / cols as f32;
        let inverse = (variance + eps).sqrt().recip();
        for column in 0..cols {
            output[start + column] = (values[column] - mean) * inverse * weight[column] + bias[column];
        }
    }
    Ok(output)
}

pub fn modulation_chunks(input: &[f32], modalities: usize, chunks: usize, hidden: usize) -> Result<Vec<Vec<f32>>, String> {
    let input_cols = modalities.checked_mul(chunks).and_then(|value| value.checked_mul(hidden)).ok_or("modulation chunks columns 溢出")?;
    if modalities == 0 || chunks == 0 || hidden == 0 || !input.len().is_multiple_of(input_cols) {
        return Err(format!("modulation chunks shape 不兼容: input={} modalities={modalities} chunks={chunks} hidden={hidden}", input.len()));
    }
    let rows = input.len() / input_cols;
    let output_elements = rows.checked_mul(modalities).and_then(|value| value.checked_mul(hidden)).ok_or("modulation chunks output 溢出")?;
    let mut output = (0..chunks).map(|_| vec![0.0; output_elements]).collect::<Vec<_>>();
    for row in 0..rows {
        for modality in 0..modalities {
            for chunk in 0..chunks {
                let source = row * input_cols + (modality * chunks + chunk) * hidden;
                let target = (row * modalities + modality) * hidden;
                output[chunk][target..target + hidden].copy_from_slice(&input[source..source + hidden]);
            }
        }
    }
    Ok(output)
}

pub fn adaln_modulate_segmented(input: &[f32], shift: &[f32], scale: &[f32], rows: usize, cols: usize, row_map: &[u32]) -> Result<Vec<f32>, String> {
    let elements = rows.checked_mul(cols).ok_or("segmented AdaLN shape 溢出")?;
    if cols == 0 || input.len() != elements || shift.len() != scale.len() || !shift.len().is_multiple_of(cols) || row_map.len() != rows {
        return Err("segmented AdaLN shape 不兼容".to_owned());
    }
    let modulation_rows = shift.len() / cols;
    if row_map.iter().any(|&row| row as usize >= modulation_rows) {
        return Err("segmented AdaLN row map 越界".to_owned());
    }
    Ok((0..elements)
        .map(|index| {
            let column = index % cols;
            let modulation = row_map[index / cols] as usize * cols + column;
            input[index] * (1.0 + scale[modulation]) + shift[modulation]
        })
        .collect())
}

pub fn gated_residual_segmented(residual: &[f32], update: &[f32], gate: &[f32], rows: usize, cols: usize, row_map: &[u32]) -> Result<Vec<f32>, String> {
    let elements = rows.checked_mul(cols).ok_or("segmented gated residual shape 溢出")?;
    if cols == 0 || residual.len() != elements || update.len() != elements || !gate.len().is_multiple_of(cols) || row_map.len() != rows {
        return Err("segmented gated residual shape 不兼容".to_owned());
    }
    let modulation_rows = gate.len() / cols;
    if row_map.iter().any(|&row| row as usize >= modulation_rows) {
        return Err("segmented gated residual row map 越界".to_owned());
    }
    Ok((0..elements)
        .map(|index| {
            let modulation = row_map[index / cols] as usize * cols + index % cols;
            residual[index] + update[index] * gate[modulation]
        })
        .collect())
}

pub fn timestep_embedding(timesteps: &[f32], dim: usize) -> Result<Vec<f32>, String> {
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(format!("timestep embedding dim {dim} 必须是非零偶数"));
    }
    let half = dim / 2;
    let mut output = vec![0.0; timesteps.len() * dim];
    for (row, &timestep) in timesteps.iter().enumerate() {
        for index in 0..half {
            let frequency = (-10000.0_f32.ln() * index as f32 / half as f32).exp();
            let value = timestep * frequency;
            output[row * dim + index] = value.cos();
            output[row * dim + half + index] = value.sin();
        }
    }
    Ok(output)
}

/// Snake 激活:`x + sin²(α·x)/α`,逐通道 α(钳到 ≥1e-6)。输入 `[batch*channels, time]`。
/// 与 ROCm `snake_f32` 逐位对齐(CPU 是 oracle)。
pub fn snake(input: &[f32], alpha: &[f32], channels: usize, time: usize) -> Result<Vec<f32>, String> {
    let row_size = channels.checked_mul(time).ok_or_else(|| "Snake 行大小溢出".to_owned())?;
    if channels == 0 || time == 0 || !input.len().is_multiple_of(row_size) || alpha.len() != channels {
        return Err(format!("Snake input={} channels={channels} time={time}", input.len()));
    }
    let mut output = vec![0.0_f32; input.len()];
    output.par_chunks_mut(time).enumerate().for_each(|(row, out_row)| {
        let channel = row % channels;
        let a = alpha[channel].abs().max(1.0e-6);
        let in_row = &input[row * time..(row + 1) * time];
        for (index, &value) in in_row.iter().enumerate() {
            let sine = (a * value).sin();
            out_row[index] = value + sine * sine / a;
        }
    });
    Ok(output)
}

/// `[batch*channels, time] -> [batch*time, channels]`,供音频 encoder 进入 attention。
/// `batch` 从 `input.len() / (channels * time)` 推出。
pub fn channels_to_time(input: &[f32], channels: usize, time: usize) -> Result<Vec<f32>, String> {
    let row_size = channels.checked_mul(time).ok_or_else(|| "channels-to-time 行大小溢出".to_owned())?;
    if channels == 0 || time == 0 || !input.len().is_multiple_of(row_size) {
        return Err(format!("channels-to-time input={} channels={channels} time={time}", input.len()));
    }
    let batch = input.len() / row_size;
    let mut output = vec![0.0_f32; input.len()];
    for batch_index in 0..batch {
        let in_batch = batch_index * channels * time;
        let out_batch = batch_index * time * channels;
        for channel in 0..channels {
            let in_ch = in_batch + channel * time;
            let out_ch = out_batch + channel;
            for timestep in 0..time {
                output[out_ch + timestep * channels] = input[in_ch + timestep];
            }
        }
    }
    Ok(output)
}

/// 因果多头自注意力:每个 (batch, query_time, head) 只 attend key_time <= query_time。
/// Q/K/V 布局 `[batch*time, heads*head_dim]`(行 = batch*time+t)。在线 softmax,与
/// ROCm `causal_attention_f32` 对齐。`batch` 从 `query.len() / (time * heads * head_dim)` 推出。
#[allow(clippy::too_many_arguments)]
pub fn causal_attention(query: &[f32], key: &[f32], value: &[f32], time: usize, heads: usize, head_dim: usize, score_scale: f32) -> Result<Vec<f32>, String> {
    let cols = heads.checked_mul(head_dim).ok_or_else(|| "causal attention cols 溢出".to_owned())?;
    let row_size = time.checked_mul(cols).ok_or_else(|| "causal attention 行大小溢出".to_owned())?;
    if time == 0 || heads == 0 || head_dim == 0 || !query.len().is_multiple_of(row_size) || query.len() != key.len() || query.len() != value.len() {
        return Err(format!("causal attention Q/K/V len={}，期望按 time={time} heads={heads} head_dim={head_dim} 拆分", query.len()));
    }
    let mut output = vec![0.0_f32; query.len()];
    output.par_chunks_mut(cols).enumerate().for_each(|(row, out_row)| {
        let query_time = row % time;
        let batch_index = row / time;
        // 批 stride 在 key_time 循环内复用,只 key_time 变;Q 行 base 同样复用。
        let key_head_base = batch_index * time * cols;
        let q_row_base = row * cols;
        let mut accumulator = vec![0.0_f32; head_dim];
        for head in 0..heads {
            let q = &query[q_row_base + head * head_dim..q_row_base + (head + 1) * head_dim];
            let mut maximum = f32::NEG_INFINITY;
            let mut denominator = 0.0_f32;
            for value in accumulator.iter_mut() {
                *value = 0.0;
            }
            for key_time in 0..=query_time {
                let key_base = key_head_base + key_time * cols + head * head_dim;
                let score = dot(q, &key[key_base..key_base + head_dim]) * score_scale;
                let new_maximum = maximum.max(score);
                let rescale = (maximum - new_maximum).exp();
                let weight = (score - new_maximum).exp();
                denominator = denominator * rescale + weight;
                for d in 0..head_dim {
                    accumulator[d] = accumulator[d] * rescale + weight * value[key_base + d];
                }
                maximum = new_maximum;
            }
            let inv = 1.0 / denominator;
            let out_base = head * head_dim;
            for d in 0..head_dim {
                out_row[out_base + d] = accumulator[d] * inv;
            }
        }
    });
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv3d_identity() {
        // 1×1×3×3 × 1×1×1×1×1 kernel=1 → identity
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = vec![1.0]; // [1,1,1,1,1]
        let out = conv3d(&input, 1, 1, 3, 3, &weight, 1, (1, 1, 1), (1, 1, 1), (0, 0, 0), None, false);
        assert_eq!(out, input);
    }

    #[test]
    fn group_norm_basic() {
        // 2 channels, 4 elements each, 1 group → 全局归一化
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![0.0, 0.0];
        let out = group_norm(&input, 2, 4, 1, 1e-6, &weight, &bias);
        let mean = 4.5;
        let rstd = 1.0 / (input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 8.0 + 1e-6).sqrt();
        for (i, &v) in input.iter().enumerate() {
            assert!((out[i] - (v - mean) * rstd).abs() < 1e-4, "idx {}: {} vs {}", i, out[i], (v - mean) * rstd);
        }
    }

    #[test]
    fn silu_correct() {
        assert!((silu(&[0.0])[0]).abs() < 1e-6);
        assert!((silu(&[1.0])[0] - 0.7310).abs() < 1e-3);
        assert!((silu(&[-1.0])[0] + 0.2689).abs() < 1e-3);
    }

    #[test]
    fn causal_conv3d_keeps_depth_and_never_reads_future() {
        let input = vec![1.0, 2.0];
        let weight = vec![0.0, 0.0, 1.0];
        let output = conv3d(&input, 1, 2, 1, 1, &weight, 1, (3, 1, 1), (1, 1, 1), (2, 0, 0), None, true);
        assert_eq!(output, input);
    }

    #[test]
    fn conv3d_im2col_matches_scalar() {
        // 2 输入通道、3 输出通道、3×3×3 kernel、causal、padding(2,1,1)、带 bias,覆盖多通道与边界。
        let (in_channels, out_channels) = (2, 3);
        let (depth, height, width) = (2, 4, 4);
        let input: Vec<f32> = (0..in_channels * depth * height * width).map(|index| (index as f32 * 0.37) % 3.0 - 1.5).collect();
        let taps = in_channels * 3 * 3 * 3;
        let weight: Vec<f32> = (0..out_channels * taps).map(|index| (index as f32 * 0.13) % 2.0 - 1.0).collect();
        let bias: Vec<f32> = vec![0.1, -0.2, 0.3];
        let expected = conv3d_scalar(&input, in_channels, depth, height, width, &weight, out_channels, (3, 3, 3), (1, 1, 1), (2, 1, 1), Some(&bias), true);
        let actual = conv3d(&input, in_channels, depth, height, width, &weight, out_channels, (3, 3, 3), (1, 1, 1), (2, 1, 1), Some(&bias), true);
        assert_eq!(expected.len(), actual.len());
        for (index, (reference, value)) in expected.iter().zip(&actual).enumerate() {
            assert!((reference - value).abs() < 1.0e-4, "conv3d[{index}]: scalar={reference} im2col={value}");
        }
    }

    #[test]
    fn group_norm_multi_group_matches_formula() {
        // 4 通道 / 2 组,spatial=3,校验并行 over groups 与逐通道 affine。
        let (channels, groups, spatial) = (4, 2, 3);
        let input: Vec<f32> = (0..channels * spatial).map(|index| index as f32 * 0.5 - 3.0).collect();
        let weight = vec![1.0, 1.0, 2.0, 0.5];
        let bias = vec![0.0, 0.1, -0.1, 0.2];
        let output = group_norm(&input, channels, spatial, groups, 1.0e-6, &weight, &bias);
        let group_size = channels / groups;
        let group_elements = group_size * spatial;
        for group in 0..groups {
            let slice = &input[group * group_elements..(group + 1) * group_elements];
            let mean = slice.iter().sum::<f32>() / group_elements as f32;
            let var = slice.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / group_elements as f32;
            let rstd = 1.0 / (var + 1.0e-6).sqrt();
            for channel_in_group in 0..group_size {
                let channel = group * group_size + channel_in_group;
                for s in 0..spatial {
                    let index = channel * spatial + s;
                    let expected = (input[index] - mean) * rstd * weight[channel] + bias[channel];
                    assert!((output[index] - expected).abs() < 1.0e-4, "group_norm[{index}]: {} vs {expected}", output[index]);
                }
            }
        }
    }

    #[test]
    fn pixel_shuffle_maps_channel_offsets_to_space() {
        assert_eq!(pixel_shuffle(&[1.0, 2.0, 3.0, 4.0], 1, 1, 1, 2), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn adaln_broadcasts_one_modulation_row() {
        let output = adaln_modulate(&[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5], &[1.0, 0.0], 2, 2).unwrap();
        assert_eq!(output, vec![2.5, 1.5, 6.5, 3.5]);
    }

    #[test]
    fn timestep_embedding_supports_batches() {
        let output = timestep_embedding(&[0.0, 1.0], 4).unwrap();
        assert_eq!(output.len(), 8);
        assert_eq!(&output[..4], &[1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn segmented_adaln_and_gate_follow_row_map() {
        let row_map = [1, 1, 0, 0];
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shift = [0.5, -0.5, -1.0, 1.0];
        let scale = [1.0, 0.0, 0.0, 0.5];
        assert_eq!(adaln_modulate_segmented(&input, &shift, &scale, 4, 2, &row_map).unwrap(), vec![0.0, 4.0, 2.0, 7.0, 10.5, 5.5, 14.5, 7.5],);
        assert_eq!(gated_residual_segmented(&[1.0; 8], &[2.0; 8], &[0.5, 1.0, 1.0, -1.0], 4, 2, &row_map).unwrap(), vec![3.0, -1.0, 3.0, -1.0, 2.0, 3.0, 2.0, 3.0],);
    }

    #[test]
    fn per_head_rmsnorm_is_independent() {
        let output = rmsnorm_heads(&[3.0, 4.0, 0.0, 5.0], &[1.0, 2.0], 1, 2, 1e-6).unwrap();
        let first_inverse = (12.5_f32 + 1e-6).sqrt().recip();
        let second_inverse = (12.5_f32 + 1e-6).sqrt().recip();
        assert!((output[0] - 3.0 * first_inverse).abs() < 1e-6);
        assert!((output[1] - 8.0 * first_inverse).abs() < 1e-6);
        assert!((output[2] - 0.0 * second_inverse).abs() < 1e-6);
        assert!((output[3] - 10.0 * second_inverse).abs() < 1e-6);
    }

    #[test]
    fn full_attention_is_bidirectional() {
        let output = full_attention(&[1.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0, 1.0], &[10.0, 0.0, 0.0, 20.0], 1, 2, 1.0).unwrap();
        let diagonal = 1.0_f32.exp() / (1.0_f32.exp() + 1.0);
        assert!((output[0] - 10.0 * diagonal).abs() < 1e-6);
        assert!((output[1] - 20.0 * (1.0 - diagonal)).abs() < 1e-6);
        assert!((output[2] - 10.0 * (1.0 - diagonal)).abs() < 1e-6);
        assert!((output[3] - 20.0 * diagonal).abs() < 1e-6);
    }

    #[test]
    fn row_bias_and_modulation_chunks_match_adaln_layout() {
        assert_eq!(add_row_bias(&[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5]).unwrap(), vec![1.5, 1.5, 3.5, 3.5]);
        assert_eq!(concat_rows(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0], 2).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let input = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(modulation_chunks(&input, 2, 3, 2).unwrap(), vec![vec![0.0, 1.0, 6.0, 7.0], vec![2.0, 3.0, 8.0, 9.0], vec![4.0, 5.0, 10.0, 11.0],]);
    }

    #[test]
    fn conv1d_strided_downsamples() {
        // 1 batch / 1 channel, length 6, kernel 3, stride 2, padding 1 → output_length 3。
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight_v = vec![1.0, 1.0, 1.0];
        let out = conv1d(&input, None, &weight_v, None, 1, 1, 1, 6, 3, 2, 1, 1).unwrap();
        assert_eq!(out, vec![3.0, 9.0, 15.0]);
    }

    #[test]
    fn snake_matches_formula() {
        // 1 batch, 2 channels, time 3。
        let input = vec![0.0, 1.0, 2.0, -1.0, 0.5, 3.0];
        let alpha = vec![1.0, 2.0];
        let out = snake(&input, &alpha, 2, 3).unwrap();
        for (index, &value) in input.iter().enumerate() {
            let channel = (index / 3) % 2;
            let a = alpha[channel].abs().max(1.0e-6);
            let sine = (a * value).sin();
            assert!((out[index] - (value + sine * sine / a)).abs() < 1.0e-5, "snake[{index}]");
        }
    }

    #[test]
    fn channels_to_time_transposes() {
        // batch 1, channels 2, time 3。
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = channels_to_time(&input, 2, 3).unwrap();
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn causal_attention_is_causal_and_matches_naive() {
        let (time, heads, head_dim) = (3_usize, 1, 2);
        let cols = heads * head_dim;
        let query = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let key = query.clone();
        let value = vec![10.0, 0.0, 0.0, 20.0, 30.0, 40.0];
        let scale = 1.0_f32;
        let out = causal_attention(&query, &key, &value, time, heads, head_dim, scale).unwrap();
        for query_time in 0..time {
            let q = &query[query_time * cols..query_time * cols + head_dim];
            let scores: Vec<f32> = (0..=query_time).map(|key_time| q.iter().zip(&key[key_time * cols..key_time * cols + head_dim]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
            let sum: f32 = weights.iter().sum();
            for d in 0..head_dim {
                let expected: f32 = (0..=query_time).map(|key_time| weights[key_time] / sum * value[key_time * cols + d]).sum::<f32>();
                assert!((out[query_time * cols + d] - expected).abs() < 1.0e-4, "causal[{query_time},{d}] {} vs {expected}", out[query_time * cols + d]);
            }
        }
        // 因果性:query 0 只看 key 0(value [10,0])→ out[0]=[10,0]。
        assert!((out[0] - 10.0).abs() < 1.0e-4 && out[1].abs() < 1.0e-4);
    }
}

pub fn rms_norm_heads_unit(input: &[f32], heads: usize, head_dim: usize, eps: f32) -> Result<Vec<f32>, String> {
    rmsnorm_heads_with(input, None, heads, head_dim, eps, 0.0)
}

pub fn scaled_residual(input: &[f32], update: &[f32], scale: &[f32], columns: usize) -> Result<Vec<f32>, String> {
    if columns == 0 || input.len() != update.len() || !input.len().is_multiple_of(columns) || scale.len() != columns {
        return Err("VAE scaled residual shape 不匹配".to_owned());
    }
    Ok(input.iter().zip(update).enumerate().map(|(index, (input, update))| input + update * scale[index % columns]).collect())
}

pub fn unpatch_affine(input: &[f32], scale: &[f32], bias: &[f32], shape: [usize; 3], patch: [usize; 3], channels: usize) -> Result<Vec<f32>, String> {
    if shape.contains(&0) || patch.contains(&0) || channels == 0 || (0..3).any(|axis| !shape[axis].is_multiple_of(patch[axis])) || scale.len() != channels || bias.len() != channels {
        return Err("VAE unpatch affine 参数非法".to_owned());
    }
    let grid = [shape[0] / patch[0], shape[1] / patch[1], shape[2] / patch[2]];
    let expected = grid.into_iter().product::<usize>() * channels * patch.into_iter().product::<usize>();
    if input.len() != expected {
        return Err(format!("VAE unpatch affine input={}，期望 {expected}", input.len()));
    }
    let mut output = vec![0.0; shape.into_iter().product::<usize>() * channels];
    for time in 0..shape[0] {
        for height in 0..shape[1] {
            for width in 0..shape[2] {
                let row = (time * shape[1] + height) * shape[2] + width;
                let patch_row = ((time / patch[0]) * grid[1] + height / patch[1]) * grid[2] + width / patch[2];
                for channel in 0..channels {
                    let column = (((channel * patch[0] + time % patch[0]) * patch[1] + height % patch[1]) * patch[2]) + width % patch[2];
                    output[row * channels + channel] = input[patch_row * channels * patch.into_iter().product::<usize>() + column] * scale[channel] + bias[channel];
                }
            }
        }
    }
    Ok(output)
}

pub fn audio_unpack_affine(input: &[f32], scale: &[f32], bias: &[f32], batch: usize, time: usize, channels: usize) -> Result<Vec<f32>, String> {
    if input.len() != batch * time * channels || scale.len() != channels || bias.len() != channels {
        return Err("audio unpack affine shape 不匹配".to_owned());
    }
    let mut output = vec![0.0; input.len()];
    for item in 0..batch {
        for channel in 0..channels {
            for step in 0..time {
                output[(item * channels + channel) * time + step] = input[(item * time + step) * channels + channel] * scale[channel] + bias[channel];
            }
        }
    }
    Ok(output)
}

fn weight_norm<'a>(weight_g: Option<&[f32]>, weight_v: &'a [f32], rows: usize, columns: usize) -> Result<Cow<'a, [f32]>, String> {
    if weight_v.len() != rows * columns || weight_g.is_some_and(|weight| weight.len() != rows) {
        return Err("audio weight norm shape 不匹配".to_owned());
    }
    // 无 g 缩放时直接借用 weight_v,避免整块拷贝。
    let Some(weight_g) = weight_g else { return Ok(Cow::Borrowed(weight_v)) };
    let mut output = vec![0.0; weight_v.len()];
    for row in 0..rows {
        let source = &weight_v[row * columns..(row + 1) * columns];
        let factor = weight_g[row] / source.iter().map(|value| value * value).sum::<f32>().sqrt().max(1.0e-12);
        for (target, source) in output[row * columns..(row + 1) * columns].iter_mut().zip(source) {
            *target = source * factor;
        }
    }
    Ok(Cow::Owned(output))
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    input: &[f32],
    weight_g: Option<&[f32]>,
    weight_v: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    input_length: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
) -> Result<Vec<f32>, String> {
    let effective = dilation * (kernel - 1) + 1;
    if stride == 0 {
        return Err("audio Conv1D stride 为 0".to_owned());
    }
    let output_length = (input_length + 2 * padding - effective) / stride + 1;
    if input.len() != batch * input_channels * input_length || bias.is_some_and(|bias| bias.len() != output_channels) {
        return Err("audio Conv1D shape 不匹配".to_owned());
    }
    let weight = weight_norm(weight_g, weight_v, output_channels, input_channels * kernel)?;
    let mut output = vec![0.0; batch * output_channels * output_length];
    for item in 0..batch {
        for out_channel in 0..output_channels {
            for out_step in 0..output_length {
                let mut sum = bias.map_or(0.0, |bias| bias[out_channel]);
                for in_channel in 0..input_channels {
                    for tap in 0..kernel {
                        let source = out_step * stride + tap * dilation;
                        if source >= padding && source - padding < input_length {
                            sum += input[(item * input_channels + in_channel) * input_length + source - padding] * weight[(out_channel * input_channels + in_channel) * kernel + tap];
                        }
                    }
                }
                output[(item * output_channels + out_channel) * output_length + out_step] = sum;
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    input: &[f32],
    weight_g: &[f32],
    weight_v: &[f32],
    bias: &[f32],
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    input_length: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>, String> {
    let output_length = (input_length - 1) * stride + kernel - 2 * padding;
    if input.len() != batch * input_channels * input_length || bias.len() != output_channels {
        return Err("audio ConvTranspose1D shape 不匹配".to_owned());
    }
    let weight = weight_norm(Some(weight_g), weight_v, input_channels, output_channels * kernel)?;
    let mut output = vec![0.0; batch * output_channels * output_length];
    for item in 0..batch {
        for out_channel in 0..output_channels {
            for out_step in 0..output_length {
                let mut sum = bias[out_channel];
                for in_channel in 0..input_channels {
                    for tap in 0..kernel {
                        let shifted = out_step + padding;
                        if shifted >= tap && (shifted - tap).is_multiple_of(stride) {
                            let source = (shifted - tap) / stride;
                            if source < input_length {
                                sum += input[(item * input_channels + in_channel) * input_length + source] * weight[(in_channel * output_channels + out_channel) * kernel + tap];
                            }
                        }
                    }
                }
                output[(item * output_channels + out_channel) * output_length + out_step] = sum;
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn snake_beta(input: &[f32], alpha: &[f32], beta: &[f32], up_filter: &[f32], down_filter: &[f32], channels: usize, length: usize) -> Result<Vec<f32>, String> {
    let filter = up_filter.len();
    let row_size = channels.checked_mul(length).ok_or_else(|| "SnakeBeta 行大小溢出".to_owned())?;
    if channels == 0 || length == 0 || !input.len().is_multiple_of(row_size) || alpha.len() != channels || beta.len() != channels || filter == 0 || down_filter.len() != filter {
        return Err("audio SnakeBeta shape 不匹配".to_owned());
    }
    let batch = input.len() / row_size;
    let pad = filter / 2 - 1;
    let crop = pad * 2 + (filter - 2) / 2;
    let mut activated = vec![0.0; input.len() * 2];
    for item in 0..batch {
        for channel in 0..channels {
            let a = alpha[channel].exp();
            let b = beta[channel].exp().max(1.0e-9);
            for step in 0..length * 2 {
                let raw = step + crop;
                let mut value = 0.0;
                for tap in 0..filter {
                    if raw >= tap && (raw - tap).is_multiple_of(2) {
                        let padded = (raw - tap) / 2;
                        if padded < length + 2 * pad {
                            let source = padded.saturating_sub(pad).min(length - 1);
                            value += input[(item * channels + channel) * length + source] * up_filter[tap] * 2.0;
                        }
                    }
                }
                activated[(item * channels + channel) * length * 2 + step] = value + (value * a).sin().powi(2) / b;
            }
        }
    }
    let mut output = vec![0.0; input.len()];
    let left = filter / 2 - usize::from(filter.is_multiple_of(2));
    for item in 0..batch {
        for channel in 0..channels {
            for step in 0..length {
                let mut value = 0.0;
                for tap in 0..filter {
                    let source = (step * 2 + tap).saturating_sub(left).min(length * 2 - 1);
                    value += activated[(item * channels + channel) * length * 2 + source] * down_filter[tap];
                }
                output[(item * channels + channel) * length + step] = value;
            }
        }
    }
    Ok(output)
}
