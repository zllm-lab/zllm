//! SeedVR2 7B 的窗口布局、双流 DiT 与单步复原算法。
//!
//! 窗口顺序与局部坐标来自 ByteDance SeedVR2（Apache-2.0）。模型算法
//! 保留在此处；设备驻留、算子与八卡交换由同目录 ROCm 组合负责。

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_node;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_pipeline;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_vae;
pub mod vae;

use std::ops::Range;

use crate::{
    backend::{Backend, BackendError, DiffusionBackend, LinearWeight, VaeBackend},
    diffusion::ModulationSegment,
    model_spec::seedvr2::SeedVr2Config,
    weight::{
        container::safetensor::TensorData,
        model::seedvr2::{SeedVr2AttentionWeights, SeedVr2BlockWeights, SeedVr2GlobalWeights, SeedVr2Linear, SeedVr2MlpWeights, SeedVr2Modulation, SeedVr2StreamWeights},
    },
};

fn error(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedVr2Request {
    pub model: String,
    #[serde(default)]
    pub input_video: std::path::PathBuf,
    /// 网站将已鉴权的产物随任务传到空闲节点，不依赖两机共享文件路径。
    #[serde(default)]
    pub input_video_data: Option<String>,
    pub width: usize,
    pub height: usize,
    #[serde(default = "default_frames")]
    pub frames: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
}
fn default_frames() -> usize {
    360
}
fn default_seed() -> u64 {
    666
}

impl SeedVr2Request {
    pub fn validate(&self) -> Result<(), String> {
        if self.model != "SeedVR2-7B" {
            return Err("model 必须为 SeedVR2-7B".to_owned());
        }
        if let Some(data) = &self.input_video_data {
            if !self.input_video.as_os_str().is_empty() || !data.starts_with("data:video/mp4;base64,") || data.len() > 70 * 1024 * 1024 {
                return Err("input_video_data 必须为不超过 50MB 的 MP4 data URL，且不能同时指定 input_video".to_owned());
            }
        } else if self.input_video.as_os_str().is_empty() || !self.input_video.is_absolute() {
            return Err("input_video 必须为节点可读的绝对文件路径".to_owned());
        }
        if self.frames == 0 || self.frames > 360 || self.width < 64 || self.height < 64 || !self.width.is_multiple_of(16) || !self.height.is_multiple_of(16) || self.width.checked_mul(self.height).is_none_or(|n| n > 16_777_216) {
            return Err("SeedVR2 要求 1..=360 帧、宽高为至少 64 的 16 倍数且面积不超过 16MP".to_owned());
        }
        Ok(())
    }
}

/// 本地可复现的标准正态输入；固定外部 noise 的验收走 denoise_with_noise，
/// 不把不同随机数发生器的同名 seed 误写成逐位对齐。
pub fn gaussian_noise(elements: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        (((z ^ (z >> 31)) >> 40) as f32 + 0.5) / 16_777_216.0
    };
    let mut values = Vec::with_capacity(elements);
    while values.len() < elements {
        let radius = (-2.0 * uniform().ln()).sqrt();
        let angle = std::f32::consts::TAU * uniform();
        values.push(radius * angle.cos());
        if values.len() < elements {
            values.push(radius * angle.sin());
        }
    }
    values
}

pub struct SeedVr2PreparedGlobal<W> {
    pub weights: SeedVr2GlobalWeights<W>,
    pub unit_norm: W,
}

pub struct SeedVr2PreparedStream<W, T> {
    pub attention: SeedVr2AttentionWeights<W>,
    pub mlp: SeedVr2MlpWeights<W>,
    pub attention_modulation: SeedVr2Modulation<T>,
    pub mlp_modulation: SeedVr2Modulation<T>,
}

pub struct SeedVr2PreparedBlock<W, T> {
    pub video: SeedVr2PreparedStream<W, T>,
    pub text: SeedVr2PreparedStream<W, T>,
}

pub(crate) fn prepare_tensor<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let (rows, cols) = match tensor.shape.as_slice() {
        [cols] => (1, *cols),
        [rows, cols] => (*rows, *cols),
        shape => return Err(error(format!("SeedVR2 权重 {} shape={shape:?} 不是向量/矩阵", tensor.name))),
    };
    match tensor.dtype.as_str() {
        "BF16" => backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, cols),
        "F16" => {
            if rows > 1 && cols > 1 {
                // 官方 F16 checkpoint 的 DiT 使用 BF16 autocast；显式准备同一
                // 计算精度，避免 ROCm 通用 F16 loader 展开 F32、占双倍显存并走 SGEMM。
                let bytes = tensor.data.chunks_exact(2).flat_map(|b| half::bf16::from_f32(half::f16::from_le_bytes([b[0], b[1]]).to_f32()).to_le_bytes()).collect::<Vec<_>>();
                return backend.prepare_weight(LinearWeight::Bf16Bytes(&bytes), rows, cols);
            }
            let values = tensor.data.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]])).collect::<Vec<_>>();
            backend.prepare_weight(LinearWeight::F16(&values), rows, cols)
        }
        "F32" => backend.prepare_f32(&tensor.to_f32().map_err(error)?, rows, cols),
        dtype => Err(error(format!("SeedVR2 权重 {} 不支持 dtype={dtype}", tensor.name))),
    }
}

pub fn prepare_linear<B: Backend>(backend: &B, source: &SeedVr2Linear) -> Result<SeedVr2Linear<B::Weight>, BackendError> {
    Ok(SeedVr2Linear { weight: prepare_tensor(backend, &source.weight)?, bias: prepare_tensor(backend, &source.bias)? })
}

pub fn linear_bias<B: DiffusionBackend>(backend: &B, input: &B::Tensor, weights: &SeedVr2Linear<B::Weight>) -> Result<B::Tensor, BackendError> {
    backend.add_row_bias(&backend.linear(input, &weights.weight)?, &weights.bias)
}

pub fn prepare_global<B: Backend>(backend: &B, config: &SeedVr2Config, source: &SeedVr2GlobalWeights) -> Result<SeedVr2PreparedGlobal<B::Weight>, BackendError> {
    config.validate().map_err(error)?;
    Ok(SeedVr2PreparedGlobal {
        weights: SeedVr2GlobalWeights {
            video_input: prepare_linear(backend, &source.video_input)?,
            text_input: prepare_linear(backend, &source.text_input)?,
            time_input: prepare_linear(backend, &source.time_input)?,
            time_hidden: prepare_linear(backend, &source.time_hidden)?,
            time_output: prepare_linear(backend, &source.time_output)?,
        },
        unit_norm: backend.prepare_f32(&vec![1.0; config.hidden_size], 1, config.hidden_size)?,
    })
}

/// 官方 embedding 顺序为所有 sin 后所有 cos；不能复用顺序相反的 H3 时间嵌入。
pub fn timestep<B: DiffusionBackend>(backend: &B, config: &SeedVr2Config, global: &SeedVr2PreparedGlobal<B::Weight>, value: f32) -> Result<B::Tensor, BackendError> {
    if !value.is_finite() {
        return Err(error("SeedVR2 timestep 必须为有限数"));
    }
    let half = config.timestep_dim / 2;
    let angles = (0..half).map(|i| value * (-10_000.0f32.ln() * i as f32 / half as f32).exp()).collect::<Vec<_>>();
    let values = angles.iter().map(|x| x.sin()).chain(angles.iter().map(|x| x.cos())).collect::<Vec<_>>();
    let input = backend.diffusion_tensor_from_f32(&values, 1, config.timestep_dim)?;
    let hidden = backend.silu(&linear_bias(backend, &input, &global.weights.time_input)?)?;
    let hidden = backend.silu(&linear_bias(backend, &hidden, &global.weights.time_hidden)?)?;
    linear_bias(backend, &hidden, &global.weights.time_output)
}

fn modulation_values(embedding: &[f32], source: &SeedVr2Modulation, layer: usize, hidden: usize) -> Result<SeedVr2Modulation<Vec<f32>>, BackendError> {
    if layer > 1 || embedding.len() != hidden * 6 {
        return Err(error("SeedVR2 AdaSingle embedding 维度不匹配"));
    }
    let combine = |tensor: &TensorData, group: usize, offset: f32| -> Result<Vec<f32>, BackendError> {
        let values = tensor.to_f32().map_err(error)?;
        if values.len() != hidden {
            return Err(error(format!("SeedVR2 AdaSingle {} 长度={}，期望 {hidden}", tensor.name, values.len())));
        }
        Ok(values.into_iter().enumerate().map(|(channel, v)| v + embedding[channel * 6 + layer * 3 + group] + offset).collect())
    };
    // AdaSingle 的 scale 已含基值，通用 AdaLN 算子另加 1，故在准备小向量时减去。
    Ok(SeedVr2Modulation { shift: combine(&source.shift, 0, 0.0)?, scale: combine(&source.scale, 1, -1.0)?, gate: combine(&source.gate, 2, 0.0)? })
}

fn prepare_stream<B: DiffusionBackend>(backend: &B, source: &SeedVr2StreamWeights, embedding: &[f32], hidden: usize) -> Result<SeedVr2PreparedStream<B::Weight, B::Tensor>, BackendError> {
    let modulation = |source, layer| -> Result<_, BackendError> {
        let values = modulation_values(embedding, source, layer, hidden)?;
        Ok(SeedVr2Modulation { shift: backend.diffusion_tensor_from_f32(&values.shift, 1, hidden)?, scale: backend.diffusion_tensor_from_f32(&values.scale, 1, hidden)?, gate: backend.diffusion_tensor_from_f32(&values.gate, 1, hidden)? })
    };
    Ok(SeedVr2PreparedStream {
        attention: SeedVr2AttentionWeights {
            qkv: prepare_tensor(backend, &source.attention.qkv)?,
            query_norm: prepare_tensor(backend, &source.attention.query_norm)?,
            key_norm: prepare_tensor(backend, &source.attention.key_norm)?,
            output: prepare_linear(backend, &source.attention.output)?,
        },
        mlp: SeedVr2MlpWeights { input: prepare_linear(backend, &source.mlp.input)?, output: prepare_linear(backend, &source.mlp.output)? },
        attention_modulation: modulation(&source.attention_modulation, 0)?,
        mlp_modulation: modulation(&source.mlp_modulation, 1)?,
    })
}

pub fn prepare_block<B: DiffusionBackend>(backend: &B, config: &SeedVr2Config, source: &SeedVr2BlockWeights, embedding: &[f32]) -> Result<SeedVr2PreparedBlock<B::Weight, B::Tensor>, BackendError> {
    Ok(SeedVr2PreparedBlock { video: prepare_stream(backend, &source.video, embedding, config.hidden_size)?, text: prepare_stream(backend, &source.text, embedding, config.hidden_size)? })
}

pub fn stream_qkv<B: DiffusionBackend>(backend: &B, config: &SeedVr2Config, unit_norm: &B::Weight, input: &B::Tensor, weights: &SeedVr2PreparedStream<B::Weight, B::Tensor>) -> Result<B::Tensor, BackendError> {
    let modulation = &weights.attention_modulation;
    let norm = backend.rmsnorm_adaln_modulate_segmented(input, unit_norm, config.norm_eps, &modulation.shift, &modulation.scale, &[ModulationSegment { rows: 0..backend.token_rows(input), modulation_row: 0 }])?;
    backend.linear(&norm, &weights.attention.qkv)
}

/// attention 已按原序列恢复到本 rank，MLP 与残差继续在相同 shard 上执行。
pub fn stream_finish<B: DiffusionBackend + VaeBackend>(
    backend: &B,
    config: &SeedVr2Config,
    unit_norm: &B::Weight,
    input: &B::Tensor,
    attention: &B::Tensor,
    weights: &SeedVr2PreparedStream<B::Weight, B::Tensor>,
) -> Result<B::Tensor, BackendError> {
    let segments = [ModulationSegment { rows: 0..backend.token_rows(input), modulation_row: 0 }];
    let attention = linear_bias(backend, attention, &weights.attention.output)?;
    let residual = backend.gated_residual_segmented(input, &attention, &weights.attention_modulation.gate, &segments)?;
    let modulation = &weights.mlp_modulation;
    let norm = backend.rmsnorm_adaln_modulate_segmented(&residual, unit_norm, config.norm_eps, &modulation.shift, &modulation.scale, &segments)?;
    let update = linear_bias(backend, &backend.vae_gelu(&linear_bias(backend, &norm, &weights.mlp.input)?)?, &weights.mlp.output)?;
    backend.gated_residual_segmented(&residual, &update, &modulation.gate, &segments)
}

#[derive(Debug)]
pub struct SeedVr2Window {
    pub shape: [usize; 3],
    pub video: Range<usize>,
    pub joint: Range<usize>,
}

/// 索引只依赖形状，整轮 36 层复用普通/移位两份布局。
#[derive(Debug)]
pub struct SeedVr2WindowPlan {
    pub video_rows: usize,
    pub text_rows: usize,
    pub partition: Vec<u32>,
    pub reverse: Vec<u32>,
    pub windows: Vec<SeedVr2Window>,
    pub joint_indices: Vec<u32>,
    pub video_indices: Vec<u32>,
    pub text_indices: Vec<Vec<u32>>,
    pub cosine: Vec<f32>,
    pub sine: Vec<f32>,
}

impl SeedVr2WindowPlan {
    pub fn new(shape: [usize; 3], text_rows: usize, window: [usize; 3], shifted: bool, frequencies: &[f32]) -> Result<Self, String> {
        if shape.contains(&0) || window.contains(&0) || text_rows == 0 || frequencies.is_empty() || frequencies.iter().any(|x| !x.is_finite()) {
            return Err(format!("SeedVR2 窗口参数非法: shape={shape:?} text={text_rows} window={window:?}"));
        }
        let video_rows = shape.into_iter().try_fold(1usize, |n, d| n.checked_mul(d)).ok_or("SeedVR2 视频行数溢出")?;
        if video_rows.checked_add(text_rows).is_none_or(|n| n > u32::MAX as usize) {
            return Err("SeedVR2 窗口索引超出 u32".to_owned());
        }
        let [t, h, w] = shape;
        // 官方窗口锚点为 patch 后 45×80；Python round 使用 ties-to-even。
        let scale = (3600.0 / (h as f64 * w as f64)).sqrt();
        let wh = ((h as f64 * scale).round_ties_even() as usize).div_ceil(window[1]).max(1);
        let ww = ((w as f64 * scale).round_ties_even() as usize).div_ceil(window[2]).max(1);
        let wt = t.min(30).div_ceil(window[0]);
        let intervals = |dim: usize, span: usize| -> Vec<Range<usize>> {
            let shift = if shifted && span < dim { 0.5 } else { 0.0 };
            let count = if shift > 0.0 { ((dim as f64 - shift) / span as f64).ceil() as usize + 1 } else { dim.div_ceil(span) };
            (0..count)
                .filter_map(|i| {
                    let start = (((i as f64 - shift) * span as f64).trunc().max(0.0) as usize).min(dim);
                    let end = (((i as f64 - shift + 1.0) * span as f64).trunc().max(0.0) as usize).min(dim);
                    (end > start).then_some(start..end)
                })
                .collect()
        };
        let ts = intervals(t, wt);
        let hs = intervals(h, wh);
        let ws = intervals(w, ww);
        let mut result = Self {
            video_rows,
            text_rows,
            partition: Vec::with_capacity(video_rows),
            reverse: vec![u32::MAX; video_rows],
            windows: Vec::new(),
            joint_indices: Vec::new(),
            video_indices: Vec::with_capacity(video_rows),
            text_indices: Vec::new(),
            cosine: Vec::new(),
            sine: Vec::new(),
        };
        // window 遍历为 W/H/T，window 内为 T/H/W，二者不能互换。
        for xr in &ws {
            for yr in &hs {
                for tr in &ts {
                    let local_shape = [tr.len(), yr.len(), xr.len()];
                    let video_start = result.partition.len();
                    let joint_start = result.joint_indices.len();
                    for ti in tr.clone() {
                        for yi in yr.clone() {
                            for xi in xr.clone() {
                                let original = (ti * h + yi) * w + xi;
                                let packed = result.partition.len();
                                if result.reverse[original] != u32::MAX {
                                    return Err(format!("SeedVR2 窗口重复覆盖视频行 {original}"));
                                }
                                result.reverse[original] = packed as u32;
                                result.partition.push(original as u32);
                                result.video_indices.push(u32::try_from(result.joint_indices.len()).map_err(|_| "SeedVR2 joint 索引溢出")?);
                                result.joint_indices.push(packed as u32);
                                for (coordinate, length) in [ti - tr.start, yi - yr.start, xi - xr.start].into_iter().zip(local_shape) {
                                    let position = if length == 1 { -1.0 } else { 2.0 * coordinate as f32 / (length - 1) as f32 - 1.0 };
                                    for &frequency in frequencies {
                                        let angle = position * frequency;
                                        result.cosine.push(angle.cos());
                                        result.sine.push(angle.sin());
                                    }
                                }
                            }
                        }
                    }
                    let mut text_indices = Vec::with_capacity(text_rows);
                    for text in 0..text_rows {
                        text_indices.push(u32::try_from(result.joint_indices.len()).map_err(|_| "SeedVR2 joint 索引溢出")?);
                        result.joint_indices.push((video_rows + text) as u32);
                    }
                    result.text_indices.push(text_indices);
                    result.windows.push(SeedVr2Window { shape: local_shape, video: video_start..result.partition.len(), joint: joint_start..result.joint_indices.len() });
                }
            }
        }
        if result.partition.len() != video_rows || result.reverse.contains(&u32::MAX) {
            return Err("SeedVR2 窗口未完整覆盖视频".to_owned());
        }
        Ok(result)
    }
}

/// `[T,H,W,C] -> [T/pt,H/ph,W/pw,pt*ph*pw*C]`。
/// 仅在输入/最终输出边界使用，层间激活始终由 backend 持有。
pub fn patchify(input: &[f32], shape: [usize; 3], channels: usize, patch: [usize; 3]) -> Result<Vec<f32>, String> {
    patch_reorder(input, shape, channels, patch, false)
}

pub fn unpatchify(input: &[f32], shape: [usize; 3], channels: usize, patch: [usize; 3]) -> Result<Vec<f32>, String> {
    patch_reorder(input, shape, channels, patch, true)
}

fn patch_reorder(input: &[f32], shape: [usize; 3], channels: usize, patch: [usize; 3], inverse: bool) -> Result<Vec<f32>, String> {
    let expected = shape.into_iter().chain([channels]).try_fold(1usize, |n, d| n.checked_mul(d));
    if shape.contains(&0) || patch.contains(&0) || channels == 0 || expected != Some(input.len()) || shape.into_iter().zip(patch).any(|(s, p)| !s.is_multiple_of(p)) {
        return Err(format!("SeedVR2 patch shape={shape:?} channels={channels} patch={patch:?} elements={} 不匹配", input.len()));
    }
    let [t, h, w] = shape;
    let [pt, ph, pw] = patch;
    let mut result = vec![0.0; input.len()];
    let mut packed = 0;
    for ti in (0..t).step_by(pt) {
        for yi in (0..h).step_by(ph) {
            for xi in (0..w).step_by(pw) {
                for dt in 0..pt {
                    for dy in 0..ph {
                        for dx in 0..pw {
                            let voxel = (((ti + dt) * h + yi + dy) * w + xi + dx) * channels;
                            if inverse {
                                result[voxel..voxel + channels].copy_from_slice(&input[packed..packed + channels]);
                            } else {
                                result[packed..packed + channels].copy_from_slice(&input[voxel..voxel + channels]);
                            }
                            packed += channels;
                        }
                    }
                }
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ada_single_uses_interleaved_time_channels_and_existing_scale_base() {
        let tensor = |v: &[f32]| TensorData { name: "ada".into(), dtype: "F32".into(), shape: vec![v.len()], data: v.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let source = SeedVr2Modulation { shift: tensor(&[1., 2.]), scale: tensor(&[1., 1.]), gate: tensor(&[0.5, 0.25]) };
        let time = [10., 11., 12., 13., 14., 15., 20., 21., 22., 23., 24., 25.];
        let attn = modulation_values(&time, &source, 0, 2).unwrap();
        assert_eq!(attn.shift, [11., 22.]);
        assert_eq!(attn.scale, [11., 21.]);
        assert_eq!(attn.gate, [12.5, 22.25]);
        let mlp = modulation_values(&time, &source, 1, 2).unwrap();
        assert_eq!(mlp.shift, [14., 25.]);
        assert_eq!(mlp.scale, [14., 24.]);
        assert_eq!(mlp.gate, [15.5, 25.25]);
    }

    #[test]
    fn official_33_frame_windows_cover_and_invert_without_padding() {
        for (shifted, count, joint) in [(false, 147, 153_678), (true, 196, 156_520)] {
            let plan = SeedVr2WindowPlan::new([9, 96, 168], 58, [4, 3, 3], shifted, &[1.0; 10]).unwrap();
            assert_eq!(plan.windows.len(), count);
            assert_eq!(plan.joint_indices.len(), joint);
            for (original, &packed) in plan.reverse.iter().enumerate() {
                assert_eq!(plan.partition[packed as usize] as usize, original);
            }
            assert_eq!(plan.cosine.len(), plan.video_rows * 30);
            assert!(plan.windows.iter().all(|w| w.joint.len() == w.video.len() + 58));
        }
    }

    #[test]
    fn shifted_window_local_rope_restarts_and_singleton_is_minus_one() {
        let plan = SeedVr2WindowPlan::new([2, 2, 2], 1, [4, 3, 3], true, &[0.7]).unwrap();
        for window in &plan.windows {
            assert!((plan.cosine[window.video.start * 3] - (-0.7f32).cos()).abs() < 1e-6);
            assert!((plan.sine[window.video.start * 3] - (-0.7f32).sin()).abs() < 1e-6);
        }
    }

    #[test]
    fn patch_layout_matches_channel_last_spatial_order() {
        let input: Vec<f32> = (0..48).map(|x| x as f32).collect();
        let packed = patchify(&input, [2, 2, 4], 3, [1, 2, 2]).unwrap();
        assert_eq!(&packed[..12], &[0., 1., 2., 3., 4., 5., 12., 13., 14., 15., 16., 17.]);
        assert_eq!(unpatchify(&packed, [2, 2, 4], 3, [1, 2, 2]).unwrap(), input);
        assert!(patchify(&input, [2, 2, 4], 3, [1, 3, 2]).is_err());
    }
}
