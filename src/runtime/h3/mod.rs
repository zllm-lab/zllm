//! H3 视频生成 pipeline：编排 VAE 编码 → DiT Flow Matching 采样 → VAE 解码。
//!
//! 与其他模型 runtime 平行，但不套用自回归 token 循环。
//! Flow Matching 采样循环（Euler 法）是本模块的核心编排逻辑。

use std::ops::Range;

use crate::backend::{Backend, BackendError, DiffusionBackend, LinearWeight};
use crate::diffusion::ModulationSegment;
use crate::moe::Activation;
use crate::weight::{
    container::safetensor::TensorData,
    model::h3::{H3AttentionWeights, H3DitBlockWeights, H3DitFinalWeights, H3DitGlobalWeights, H3DitSource, H3MlpWeights, H3Tensor, H3TokenRefinerBlockWeights},
};

/// H3 已解码并常驻 host 内存的连续 DiT 层权重。
pub struct HostLayerCache<W> {
    layers: Vec<W>,
}

impl<W> HostLayerCache<W> {
    pub fn load<E>(layer_count: usize, mut load: impl FnMut(usize) -> Result<W, E>) -> Result<Self, E> {
        let layers = (0..layer_count).map(&mut load).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, W> {
        self.layers.iter()
    }
}

/// H3 按需加载的 host DiT 层权重；每个 chunk 用完即释放。
pub struct HostStreamedLayers<'a, W> {
    count: usize,
    next_layer: usize,
    loader: Box<dyn FnMut(usize) -> Result<W, String> + 'a>,
}

impl<'a, W> HostStreamedLayers<'a, W> {
    pub fn new<F>(count: usize, loader: F) -> Self
    where
        F: FnMut(usize) -> Result<W, String> + 'a,
    {
        Self { count, next_layer: 0, loader: Box::new(loader) }
    }

    pub fn layer_count(&self) -> usize {
        self.count
    }

    pub fn rewind(&mut self) {
        self.next_layer = 0;
    }

    pub fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<Vec<W>>, String> {
        if chunk_size == 0 {
            return Err("H3 layer chunk_size 不能为 0".to_owned());
        }
        if self.next_layer >= self.count {
            return Ok(None);
        }
        let upper = (self.next_layer + chunk_size).min(self.count);
        let mut chunk = Vec::with_capacity(upper - self.next_layer);
        for layer in self.next_layer..upper {
            chunk.push((self.loader)(layer)?);
        }
        self.next_layer = upper;
        Ok(Some(chunk))
    }
}

const FRAME_PER_TOKEN: [usize; 5] = [1, 4, 4, 4, 4];
const FRAME_RESCALE: f32 = 5.0 / 3.0;
pub const VISUAL_CONDITION_TIMESTEP: f32 = 0.999;

/// H3 latent 的确定性标准正态噪声；连续调用保持同一随机序列。
pub struct H3GaussianNoise {
    state: u64,
    spare: Option<f32>,
}

impl H3GaussianNoise {
    pub fn new(seed: u64) -> Self {
        Self { state: seed, spare: None }
    }

    fn uniform(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (((value >> 40) as u32) as f32 + 0.5) / (1u32 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f32::consts::TAU * self.uniform();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    pub fn values(&mut self, elements: usize) -> Vec<f32> {
        (0..elements).map(|_| self.normal()).collect()
    }
}

/// H3 packed sequence 中的连续段类型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3SegmentKind {
    Text,
    VideoCondition,
    AudioCondition,
    Audio,
    Video,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3PackedSegment {
    pub rows: Range<usize>,
    pub kind: H3SegmentKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3KeyframeAnchor {
    First,
    Last,
}

/// 一次 H3 forward 的静态 packed-token 布局。
///
/// `position_ids` 按 `[time, height, width]` 排列，直接用于生成 3 轴 MM-RoPE。
#[derive(Clone, Debug)]
pub struct H3PackedLayout {
    pub seq_len: usize,
    pub position_ids: Vec<[f32; 3]>,
    pub segments: Vec<H3PackedSegment>,
    pub audio_rows: Range<usize>,
    pub video_rows: Range<usize>,
    /// patch 后的 `[time, height, width]`。
    pub video_grid: [usize; 3],
}

/// H3 Ulysses 中一个 rank 持有的连续 sequence/head 区间。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3UlyssesShard {
    pub sequence: Range<usize>,
    pub heads: Range<usize>,
}

/// 一次 H3 DiT forward 的 Ulysses 静态切分。
///
/// sequence 允许不整除 rank 数；attention head 必须整除，避免在逐层热路径中
/// 引入 padding、mask 和额外布局分支。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3UlyssesPlan {
    pub sequence_len: usize,
    pub head_count: usize,
    pub shards: Vec<H3UlyssesShard>,
}

impl H3UlyssesPlan {
    pub fn new(sequence_len: usize, head_count: usize, ranks: usize) -> Result<Self, String> {
        if sequence_len == 0 || head_count == 0 {
            return Err("H3 Ulysses sequence/head 不能为 0".to_owned());
        }
        if !matches!(ranks, 1 | 2 | 4 | 8) {
            return Err(format!("H3 Ulysses ranks={ranks} 不受支持，必须是 1/2/4/8"));
        }
        if sequence_len < ranks {
            return Err(format!("H3 Ulysses sequence={sequence_len} 小于 ranks={ranks}"));
        }
        if !head_count.is_multiple_of(ranks) {
            return Err(format!("H3 Ulysses heads={head_count} 不能整除 ranks={ranks}"));
        }
        let base_rows = sequence_len / ranks;
        let extra_rows = sequence_len % ranks;
        let heads_per_rank = head_count / ranks;
        let mut row = 0;
        let shards = (0..ranks)
            .map(|rank| {
                let rows = base_rows + usize::from(rank < extra_rows);
                let sequence = row..row + rows;
                row += rows;
                H3UlyssesShard { sequence, heads: rank * heads_per_rank..(rank + 1) * heads_per_rank }
            })
            .collect();
        Ok(Self { sequence_len, head_count, shards })
    }

    /// 将完整 sequence 上连续的 AdaLN 段裁到指定 rank，并把行号平移为
    /// rank-local 坐标。这样除 attention 外的算子都只处理本地 token。
    pub fn local_modulation_segments(&self, rank: usize, segments: &[ModulationSegment]) -> Result<Vec<ModulationSegment>, String> {
        let shard = self.shards.get(rank).ok_or_else(|| format!("H3 Ulysses rank={rank} 越界，ranks={}", self.shards.len()))?;
        let mut cursor = 0;
        for (index, segment) in segments.iter().enumerate() {
            if segment.rows.start != cursor || segment.rows.end <= segment.rows.start || segment.rows.end > self.sequence_len {
                return Err(format!("H3 Ulysses modulation segment {index} 非法: rows={:?} cursor={cursor} sequence={}", segment.rows, self.sequence_len));
            }
            cursor = segment.rows.end;
        }
        if cursor != self.sequence_len {
            return Err(format!("H3 Ulysses modulation segments 只覆盖 {cursor}/{} 行", self.sequence_len));
        }

        let local = segments
            .iter()
            .filter_map(|segment| {
                let start = segment.rows.start.max(shard.sequence.start);
                let end = segment.rows.end.min(shard.sequence.end);
                (start < end).then(|| ModulationSegment { rows: start - shard.sequence.start..end - shard.sequence.start, modulation_row: segment.modulation_row })
            })
            .collect::<Vec<_>>();
        if local.first().is_none_or(|segment| segment.rows.start != 0) || local.last().is_none_or(|segment| segment.rows.end != shard.sequence.len()) {
            return Err(format!("H3 Ulysses rank={rank} modulation 未覆盖 local rows={}", shard.sequence.len()));
        }
        Ok(local)
    }
}

/// 一次 forward 中视频/音频实际使用的时间及音频 velocity 缩放。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H3TimeState {
    pub video: f32,
    pub audio: f32,
    pub audio_velocity_scale: f32,
}

/// AdaLN 只为唯一时间步计算 embedding，再由连续段引用对应 modality row。
#[derive(Clone, Debug)]
pub struct H3ModulationPlan {
    pub unique_timesteps: Vec<f32>,
    pub segments: Vec<ModulationSegment>,
}

/// 在同一 base grid 上从一个 sigma shift 映射到另一个。
pub fn time_shift_sigma(sigma: f32, from_shift: f32, to_shift: f32) -> f32 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * base / (1.0 + (to_shift - 1.0) * base)
}

/// `d(sigma_to) / d(sigma_from)`，用来把音频 ODE 映射到视频采样器的 sigma 轴。
pub fn time_shift_slope(sigma: f32, from_shift: f32, to_shift: f32) -> f32 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * (1.0 + (from_shift - 1.0) * base).powi(2) / (from_shift * (1.0 + (to_shift - 1.0) * base).powi(2))
}

pub fn h3_time_state(sigma_video: f32, config: &H3Config) -> Result<H3TimeState, String> {
    if !sigma_video.is_finite() || !(0.0..=1.0).contains(&sigma_video) {
        return Err(format!("H3 video sigma={sigma_video} 必须在 [0,1]"));
    }
    config.validate()?;
    let sigma_video = sigma_video.max(1e-6);
    let sigma_audio = time_shift_sigma(sigma_video, config.sigma_shift_video, config.sigma_shift_audio);
    Ok(H3TimeState { video: 1.0 - sigma_video, audio: 1.0 - sigma_audio, audio_velocity_scale: time_shift_slope(sigma_video, config.sigma_shift_video, config.sigma_shift_audio) })
}

/// 构造 T2VA/FL2VA 共用的 `[text | condition | audio | video]` 布局。
#[allow(clippy::too_many_arguments)]
pub fn build_packed_layout(text_len: usize, latent_t: usize, latent_h: usize, latent_w: usize, audio_t: usize, patch_size: [usize; 3], keyframes: &[H3KeyframeAnchor]) -> Result<H3PackedLayout, String> {
    if text_len == 0 || latent_t == 0 || latent_h == 0 || latent_w == 0 || audio_t == 0 || patch_size.contains(&0) {
        return Err("H3 packed layout 维度必须非零".to_owned());
    }
    if !latent_t.is_multiple_of(patch_size[0]) || !latent_h.is_multiple_of(patch_size[1]) || !latent_w.is_multiple_of(patch_size[2]) {
        return Err(format!("H3 latent [{latent_t},{latent_h},{latent_w}] 不能被 patch {patch_size:?} 整除",));
    }
    if keyframes.iter().filter(|&&anchor| anchor == H3KeyframeAnchor::First).count() > 1 || keyframes.iter().filter(|&&anchor| anchor == H3KeyframeAnchor::Last).count() > 1 {
        return Err("H3 FL2VA first/last keyframe 不能重复".to_owned());
    }

    let video_grid = [latent_t / patch_size[0], latent_h / patch_size[1], latent_w / patch_size[2]];
    let (frame_grid, width_grid) = spatial_frame_grid(latent_h, latent_w, patch_size[1], patch_size[2]);
    let mut position_ids = Vec::new();
    let mut segments = Vec::new();

    let text_start = position_ids.len();
    position_ids.extend((0..text_len).map(|token| [token as f32, 0.0, 0.0]));
    segments.push(H3PackedSegment { rows: text_start..position_ids.len(), kind: H3SegmentKind::Text });

    let video_spans = video_time_spans(video_grid[0]);
    for anchor in keyframes {
        let time = match anchor {
            H3KeyframeAnchor::First => text_len as f32,
            H3KeyframeAnchor::Last => text_len as f32 + video_spans.iter().sum::<f32>() - FRAME_RESCALE,
        };
        let start = position_ids.len();
        position_ids.extend(frame_grid.iter().map(|&[height, width]| [time, height, width]));
        segments.push(H3PackedSegment { rows: start..position_ids.len(), kind: H3SegmentKind::VideoCondition });
    }

    let cursor = text_len as f32;
    let audio_start = position_ids.len();
    for channel in 0..2 {
        let width = if channel == 0 { width_grid[0] } else { width_grid[width_grid.len() - 1] };
        position_ids.extend((0..audio_t).map(|frame| [cursor + frame as f32, 0.0, width]));
    }
    let audio_rows = audio_start..position_ids.len();
    segments.push(H3PackedSegment { rows: audio_rows.clone(), kind: H3SegmentKind::Audio });

    let video_start = position_ids.len();
    let mut time = cursor;
    for span in video_spans {
        position_ids.extend(frame_grid.iter().map(|&[height, width]| [time, height, width]));
        time += span;
    }
    let video_rows = video_start..position_ids.len();
    segments.push(H3PackedSegment { rows: video_rows.clone(), kind: H3SegmentKind::Video });

    Ok(H3PackedLayout { seq_len: position_ids.len(), position_ids, segments, audio_rows, video_rows, video_grid })
}

pub fn build_modulation_plan(layout: &H3PackedLayout, time: H3TimeState) -> H3ModulationPlan {
    let condition_time = time.video.max(VISUAL_CONDITION_TIMESTEP);
    let mut unique_timesteps = vec![time.video, time.audio];
    if layout.segments.iter().any(|segment| matches!(segment.kind, H3SegmentKind::VideoCondition | H3SegmentKind::AudioCondition)) {
        unique_timesteps.push(condition_time);
    }
    unique_timesteps.sort_unstable_by(f32::total_cmp);
    unique_timesteps.dedup_by(|left, right| left.to_bits() == right.to_bits());

    let segments = layout
        .segments
        .iter()
        .map(|segment| {
            let (timestep, modality) = match segment.kind {
                H3SegmentKind::Text => (time.video, 1),
                H3SegmentKind::VideoCondition => (condition_time, 0),
                H3SegmentKind::AudioCondition => (condition_time, 2),
                H3SegmentKind::Audio => (time.audio, 2),
                H3SegmentKind::Video => (time.video, 0),
            };
            let time_row = unique_timesteps.iter().position(|value| value.to_bits() == timestep.to_bits()).expect("modulation 时间步来自 unique_timesteps");
            ModulationSegment { rows: segment.rows.clone(), modulation_row: time_row * 3 + modality }
        })
        .collect();
    H3ModulationPlan { unique_timesteps, segments }
}

/// 将 `[time,height,width]` 三轴坐标展开为 partial split-half RoPE 的 cos/sin 表。
///
/// 每轴使用同一份 `inv_freq`，因此半维度为 `3 * inv_freq.len()`。
pub fn mm_rope_tables(position_ids: &[[f32; 3]], inv_freq: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
    if position_ids.is_empty() || inv_freq.is_empty() || position_ids.iter().flatten().any(|value| !value.is_finite()) || inv_freq.iter().any(|value| !value.is_finite()) {
        return Err("H3 MM-RoPE position/inv_freq 必须是非空有限数".to_owned());
    }
    let half = inv_freq.len().checked_mul(3).ok_or("H3 MM-RoPE 维度溢出")?;
    let elements = position_ids.len().checked_mul(half).ok_or("H3 MM-RoPE table 大小溢出")?;
    let mut cosine = Vec::with_capacity(elements);
    let mut sine = Vec::with_capacity(elements);
    for position in position_ids {
        for axis in position {
            for frequency in inv_freq {
                let angle = axis * frequency;
                cosine.push(angle.cos());
                sine.push(angle.sin());
            }
        }
    }
    Ok((cosine, sine))
}

/// H3 的无 mask self-attention：fused QKV 之后才分支，Q/K 共用同一 MM-RoPE 表。
#[allow(clippy::too_many_arguments)]
pub fn attention<B: DiffusionBackend>(backend: &B, config: &H3Config, input: &B::Tensor, weights: &H3AttentionWeights<B::Weight>, cosine: &[f32], sine: &[f32]) -> Result<B::Tensor, BackendError> {
    if backend.token_cols(input) != config.hidden_size {
        return Err(BackendError::Compute { msg: format!("H3 attention input cols={} 期望 {}", backend.token_cols(input), config.hidden_size,) });
    }
    let attention_dim = config.attention_dim();
    let qkv = backend.linear(input, &weights.qkv)?;
    if backend.token_cols(&qkv) != attention_dim * 3 {
        return Err(BackendError::Compute { msg: format!("H3 QKV cols={} 期望 {}", backend.token_cols(&qkv), attention_dim * 3,) });
    }
    let attended = backend.full_attention_qkv(
        qkv,
        &weights.q_norm,
        &weights.k_norm,
        config.num_attention_heads,
        config.attention_head_dim,
        config.rope_inv_freq_len * 6,
        config.qk_norm_eps,
        cosine,
        sine,
        (config.attention_head_dim as f32).sqrt().recip(),
    )?;
    backend.linear(&attended, &weights.output)
}

/// Token refiner 使用独立的无 mask self-attention，不应用主干 MM-RoPE。
pub fn token_refiner_attention<B: DiffusionBackend>(backend: &B, config: &H3Config, input: &B::Tensor, weights: &H3AttentionWeights<B::Weight>) -> Result<B::Tensor, BackendError> {
    if backend.token_cols(input) != config.hidden_size {
        return Err(BackendError::Compute { msg: format!("H3 token refiner attention cols={} 期望 {}", backend.token_cols(input), config.hidden_size,) });
    }
    let attention_dim = config.attention_dim();
    let qkv = backend.linear(input, &weights.qkv)?;
    if backend.token_cols(&qkv) != attention_dim * 3 {
        return Err(BackendError::Compute { msg: format!("H3 token refiner QKV cols={} 期望 {}", backend.token_cols(&qkv), attention_dim * 3,) });
    }
    let (query, key_value) = backend.split_columns(&qkv, attention_dim)?;
    let (key, value) = backend.split_columns(&key_value, attention_dim)?;
    let query = backend.rmsnorm_heads(&query, &weights.q_norm, config.num_attention_heads, config.attention_head_dim, config.qk_norm_eps)?;
    let key = backend.rmsnorm_heads(&key, &weights.k_norm, config.num_attention_heads, config.attention_head_dim, config.qk_norm_eps)?;
    let attended = backend.full_attention(query, key, value, config.num_attention_heads, config.attention_head_dim, (config.attention_head_dim as f32).sqrt().recip())?;
    backend.linear(&attended, &weights.output)
}

fn linear_bias<B: DiffusionBackend>(backend: &B, input: &B::Tensor, weight: &B::Weight, bias: &B::Weight) -> Result<B::Tensor, BackendError> {
    let output = backend.linear(input, weight)?;
    backend.add_row_bias(&output, bias)
}

/// H3 官方权重只在准备期从 safetensors 的 BF16/F32 表示进入 backend。
/// BF16 保持原始字节，不在 host 展开成 F32。
trait H3WeightData {
    fn name(&self) -> &str;
    fn dtype(&self) -> &str;
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[u8];
    fn quantized(&self) -> Option<&crate::weight::format::quantization::W8A16Matrix> {
        None
    }
}

impl H3WeightData for H3Tensor {
    fn name(&self) -> &str {
        &self.name
    }
    fn dtype(&self) -> &str {
        &self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> &[u8] {
        &self.data
    }
    fn quantized(&self) -> Option<&crate::weight::format::quantization::W8A16Matrix> {
        self.quantized.as_ref()
    }
}

impl H3WeightData for TensorData {
    fn name(&self) -> &str {
        &self.name
    }
    fn dtype(&self) -> &str {
        &self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> &[u8] {
        &self.data
    }
}

fn prepare_h3_dit_weight<B: Backend, T: H3WeightData>(backend: &B, tensor: &T) -> Result<B::Weight, BackendError> {
    let name = tensor.name();
    let dtype = tensor.dtype();
    let shape = tensor.shape();
    let data = tensor.data();
    let (rows, cols) = match shape {
        [columns] => (1, *columns),
        [rows, dimensions @ ..] if !dimensions.is_empty() => {
            let cols = dimensions.iter().try_fold(1usize, |count, value| count.checked_mul(*value)).ok_or_else(|| BackendError::Compute { msg: format!("H3 权重 {name} shape={shape:?} 大小溢出") })?;
            (*rows, cols)
        }
        shape => return Err(BackendError::Compute { msg: format!("H3 权重 {name} shape={shape:?} 非法") }),
    };
    if let Some(matrix) = tensor.quantized() {
        return backend.prepare_weight(LinearWeight::w8a16(matrix), rows, cols);
    }
    let elements = rows.checked_mul(cols).ok_or_else(|| BackendError::Compute { msg: format!("H3 权重 {name} shape={shape:?} 大小溢出") })?;
    match dtype {
        "BF16" if data.len() == elements * 2 => backend.prepare_weight(LinearWeight::Bf16Bytes(data), rows, cols),
        "F16" if data.len() == elements * 2 => {
            let values = data.chunks_exact(2).map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            backend.prepare_weight(LinearWeight::F16(&values), rows, cols)
        }
        "F32" if data.len() == elements * 4 => {
            // H3 的 F32 线性权重只有 audio/video patch_proj(patch embedding)。
            // 转 F16:Metal linear/add_row_bias 只接 F16;ROCm/CPU 也兼容 F16。
            // patch embedding 对 F16 精度不敏感(VAE/patch 原生即 F16 量级)。
            let values = data.chunks_exact(4).map(|bytes| half::f16::from_f32(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))).collect::<Vec<_>>();
            backend.prepare_weight(LinearWeight::F16(&values), rows, cols)
        }
        _ => Err(BackendError::Compute { msg: format!("H3 权重 {name} dtype={dtype} bytes={} 与 shape={shape:?} 不兼容", data.len()) }),
    }
}

/// Video VAE checkpoint 为 F32；二维线性权重转 BF16 后由 WMMA 计算并保持 F32 累加。
fn prepare_h3_video_vae_weight<B: Backend, T: H3WeightData>(backend: &B, tensor: &T) -> Result<B::Weight, BackendError> {
    let shape = tensor.shape();
    if tensor.dtype() != "F32" || tensor.quantized().is_some() || shape.len() < 2 {
        return prepare_h3_dit_weight(backend, tensor);
    }
    let rows = shape[0];
    let cols = shape[1..].iter().try_fold(1usize, |count, value| count.checked_mul(*value)).ok_or_else(|| BackendError::Compute { msg: format!("H3 Video VAE 权重 {} shape={shape:?} 大小溢出", tensor.name()) })?;
    let elements = rows.checked_mul(cols).ok_or_else(|| BackendError::Compute { msg: format!("H3 Video VAE 权重 {} 大小溢出", tensor.name()) })?;
    if rows <= 1 || cols <= 1 || !rows.is_multiple_of(16) || !cols.is_multiple_of(16) || tensor.data().len() != elements * 4 {
        return prepare_h3_dit_weight(backend, tensor);
    }
    let mut bytes = Vec::with_capacity(elements * 2);
    for value in tensor.data().chunks_exact(4) {
        let value = f32::from_le_bytes([value[0], value[1], value[2], value[3]]);
        bytes.extend_from_slice(&half::bf16::from_f32(value).to_le_bytes());
    }
    backend.prepare_weight(LinearWeight::Bf16Bytes(&bytes), rows, cols)
}

fn prepare_qkv_weight<B: Backend>(backend: &B, qkv: &H3Tensor, q_norm: &H3Tensor) -> Result<B::Weight, BackendError> {
    if qkv.quantized.is_some() || !qkv.qkv_interleaved {
        return prepare_h3_dit_weight(backend, qkv);
    }
    let (rows, cols) = match qkv.shape.as_slice() {
        [rows, cols] => (*rows, *cols),
        shape => return Err(BackendError::Compute { msg: format!("H3 QKV 权重 {} shape={shape:?} 必须是 rank-2", qkv.name,) }),
    };
    let head_dim = match q_norm.shape.as_slice() {
        [head_dim] if *head_dim > 0 => *head_dim,
        shape => return Err(BackendError::Compute { msg: format!("H3 Q norm 权重 {} shape={shape:?} 非法", q_norm.name,) }),
    };
    let per_head = 3usize.checked_mul(head_dim).ok_or_else(|| BackendError::Compute { msg: "H3 QKV per-head 大小溢出".to_owned() })?;
    if qkv.dtype != "BF16" || rows % per_head != 0 {
        return Err(BackendError::Compute { msg: format!("H3 QKV 权重 {} dtype={} shape={:?} 不是 per-head 交错 BF16", qkv.name, qkv.dtype, qkv.shape,) });
    }
    let heads = rows / per_head;
    let row_bytes = cols.checked_mul(2).ok_or_else(|| BackendError::Compute { msg: "H3 QKV row bytes 溢出".to_owned() })?;
    let group_bytes = head_dim.checked_mul(row_bytes).ok_or_else(|| BackendError::Compute { msg: "H3 QKV head bytes 溢出".to_owned() })?;
    if qkv.data.len() != rows.saturating_mul(row_bytes) {
        return Err(BackendError::Compute { msg: format!("H3 QKV 权重 {} bytes={} 与 shape={:?} 不兼容", qkv.name, qkv.data.len(), qkv.shape,) });
    }
    let mut data = vec![0u8; qkv.data.len()];
    for head in 0..heads {
        for branch in 0..3 {
            let source = (head * 3 + branch) * group_bytes;
            let target = (branch * heads + head) * group_bytes;
            data[target..target + group_bytes].copy_from_slice(&qkv.data[source..source + group_bytes]);
        }
    }
    prepare_h3_dit_weight(backend, &H3Tensor { name: qkv.name.clone(), dtype: qkv.dtype.clone(), shape: qkv.shape.clone(), data, quantized: None, qkv_interleaved: false })
}

fn prepare_attention_weights<B: Backend>(backend: &B, source: &H3AttentionWeights) -> Result<H3AttentionWeights<B::Weight>, BackendError> {
    Ok(H3AttentionWeights {
        qkv: prepare_qkv_weight(backend, &source.qkv, &source.q_norm)?,
        q_norm: prepare_h3_dit_weight(backend, &source.q_norm)?,
        k_norm: prepare_h3_dit_weight(backend, &source.k_norm)?,
        output: prepare_h3_dit_weight(backend, &source.output)?,
    })
}

fn prepare_mlp_weights<B: Backend>(backend: &B, source: &H3MlpWeights) -> Result<H3MlpWeights<B::Weight>, BackendError> {
    Ok(H3MlpWeights { gate_up: prepare_h3_dit_weight(backend, &source.gate_up)?, down: prepare_h3_dit_weight(backend, &source.down)? })
}

fn prepare_block_weights<B: Backend>(backend: &B, source: &H3DitBlockWeights) -> Result<H3DitBlockWeights<B::Weight>, BackendError> {
    Ok(H3DitBlockWeights {
        norm1: prepare_h3_dit_weight(backend, &source.norm1)?,
        norm2: prepare_h3_dit_weight(backend, &source.norm2)?,
        adaln_weight: prepare_h3_dit_weight(backend, &source.adaln_weight)?,
        adaln_bias: prepare_h3_dit_weight(backend, &source.adaln_bias)?,
        attention: prepare_attention_weights(backend, &source.attention)?,
        mlp: prepare_mlp_weights(backend, &source.mlp)?,
    })
}

fn prepare_token_refiner_weights<B: Backend>(backend: &B, source: &H3TokenRefinerBlockWeights) -> Result<H3TokenRefinerBlockWeights<B::Weight>, BackendError> {
    Ok(H3TokenRefinerBlockWeights {
        norm1: prepare_h3_dit_weight(backend, &source.norm1)?,
        norm2: prepare_h3_dit_weight(backend, &source.norm2)?,
        attention: prepare_attention_weights(backend, &source.attention)?,
        mlp: prepare_mlp_weights(backend, &source.mlp)?,
    })
}

fn prepare_final_weights<B: Backend>(backend: &B, source: &H3DitFinalWeights) -> Result<H3DitFinalWeights<B::Weight>, BackendError> {
    Ok(H3DitFinalWeights {
        adaln_weight: prepare_h3_dit_weight(backend, &source.adaln_weight)?,
        adaln_bias: prepare_h3_dit_weight(backend, &source.adaln_bias)?,
        norm: prepare_h3_dit_weight(backend, &source.norm)?,
        video_output_weight: prepare_h3_dit_weight(backend, &source.video_output_weight)?,
        video_output_bias: prepare_h3_dit_weight(backend, &source.video_output_bias)?,
        audio_output_weight: prepare_h3_dit_weight(backend, &source.audio_output_weight)?,
        audio_output_bias: prepare_h3_dit_weight(backend, &source.audio_output_bias)?,
    })
}

/// 一次 H3 forward 常驻的设备权重；RoPE 频率保留 host F32，供每个请求生成位置表。
pub struct H3PreparedGlobal<W> {
    pub weights: H3DitGlobalWeights<W>,
    pub rope_inv_freq: Vec<f32>,
}

pub fn prepare_dit_global<B: Backend>(backend: &B, source: &H3DitGlobalWeights) -> Result<H3PreparedGlobal<B::Weight>, BackendError> {
    if source.rope_inv_freq.dtype != "F32" {
        return Err(BackendError::Compute { msg: format!("H3 RoPE 权重 {} dtype={}，期望 F32", source.rope_inv_freq.name, source.rope_inv_freq.dtype) });
    }
    let elements =
        source.rope_inv_freq.shape.iter().try_fold(1usize, |elements, &dimension| elements.checked_mul(dimension)).ok_or_else(|| BackendError::Compute { msg: format!("H3 RoPE 权重 {} shape 溢出", source.rope_inv_freq.name) })?;
    let expected_bytes = elements.checked_mul(4).ok_or_else(|| BackendError::Compute { msg: format!("H3 RoPE 权重 {} 字节数溢出", source.rope_inv_freq.name) })?;
    if source.rope_inv_freq.data.len() != expected_bytes {
        return Err(BackendError::Compute { msg: format!("H3 RoPE 权重 {} bytes={}，期望 {expected_bytes}", source.rope_inv_freq.name, source.rope_inv_freq.data.len()) });
    }
    let rope_inv_freq = source.rope_inv_freq.data.chunks_exact(4).map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])).collect();
    let weights = H3DitGlobalWeights {
        video_patch_weight: prepare_h3_dit_weight(backend, &source.video_patch_weight)?,
        video_patch_bias: prepare_h3_dit_weight(backend, &source.video_patch_bias)?,
        audio_patch_weight: prepare_h3_dit_weight(backend, &source.audio_patch_weight)?,
        audio_patch_bias: prepare_h3_dit_weight(backend, &source.audio_patch_bias)?,
        condition_weight: prepare_h3_dit_weight(backend, &source.condition_weight)?,
        condition_bias: prepare_h3_dit_weight(backend, &source.condition_bias)?,
        time_input_weight: prepare_h3_dit_weight(backend, &source.time_input_weight)?,
        time_input_bias: prepare_h3_dit_weight(backend, &source.time_input_bias)?,
        time_output_weight: prepare_h3_dit_weight(backend, &source.time_output_weight)?,
        time_output_bias: prepare_h3_dit_weight(backend, &source.time_output_bias)?,
        rope_inv_freq: prepare_h3_dit_weight(backend, &source.rope_inv_freq)?,
        time_curve: source.time_curve.clone(),
    };
    Ok(H3PreparedGlobal { weights, rope_inv_freq })
}

pub fn time_embedding<B: DiffusionBackend>(backend: &B, config: &H3Config, timesteps: &[f32], weights: &H3DitGlobalWeights<B::Weight>) -> Result<B::Tensor, BackendError> {
    if let Some(curve) = &weights.time_curve {
        let mut values = Vec::with_capacity(timesteps.len() * curve.cols);
        for &timestep in timesteps {
            let position = timestep.clamp(0.0, 1.0) * (curve.rows - 1) as f32;
            let lower = (position.floor() as usize).min(curve.rows - 2);
            let fraction = position - lower as f32;
            for column in 0..curve.cols {
                let left = curve.values[lower * curve.cols + column];
                let right = curve.values[(lower + 1) * curve.cols + column];
                values.push(left + (right - left) * fraction);
            }
        }
        return backend.diffusion_tensor_from_f32(&values, timesteps.len(), curve.cols);
    }
    let embedding = backend.timestep_embedding(timesteps, config.timestep_input_dim)?;
    let hidden = linear_bias(backend, &embedding, &weights.time_input_weight, &weights.time_input_bias)?;
    let hidden = backend.silu(&hidden)?;
    linear_bias(backend, &hidden, &weights.time_output_weight, &weights.time_output_bias)
}

pub fn mlp<B: DiffusionBackend>(backend: &B, config: &H3Config, input: &B::Tensor, weights: &H3MlpWeights<B::Weight>) -> Result<B::Tensor, BackendError> {
    let gate_up = backend.linear(input, &weights.gate_up)?;
    if backend.token_cols(&gate_up) != config.ffn_hidden_size * 2 {
        return Err(BackendError::Compute { msg: format!("H3 MLP gate/up cols={} 期望 {}", backend.token_cols(&gate_up), config.ffn_hidden_size * 2,) });
    }
    let activated = backend.split_gated_activation(gate_up, config.ffn_hidden_size, &Activation::Silu)?;
    backend.linear(&activated, &weights.down)
}

pub fn token_refiner_block<B: DiffusionBackend>(backend: &B, config: &H3Config, hidden: &B::Tensor, weights: &H3TokenRefinerBlockWeights<B::Weight>) -> Result<B::Tensor, BackendError> {
    let normalized = backend.rmsnorm(hidden, &weights.norm1, config.norm_eps)?;
    let update = token_refiner_attention(backend, config, &normalized, &weights.attention)?;
    let hidden = backend.add(hidden, &update)?;
    let normalized = backend.rmsnorm(&hidden, &weights.norm2, config.norm_eps)?;
    let update = mlp(backend, config, &normalized, &weights.mlp)?;
    backend.add(&hidden, &update)
}

pub struct H3AdaLnParameters<T> {
    pub shift_msa: T,
    pub scale_msa: T,
    pub gate_msa: T,
    pub shift_mlp: T,
    pub scale_mlp: T,
    pub gate_mlp: T,
}

fn block_adaln<B: DiffusionBackend>(backend: &B, config: &H3Config, time_embedding: &B::Tensor, weights: &H3DitBlockWeights<B::Weight>) -> Result<H3AdaLnParameters<B::Tensor>, BackendError> {
    let projected = if config.time_embed_dim == 8 {
        linear_bias(backend, time_embedding, &weights.adaln_weight, &weights.adaln_bias)?
    } else {
        let activated = backend.silu(time_embedding)?;
        linear_bias(backend, &activated, &weights.adaln_weight, &weights.adaln_bias)?
    };
    let chunks: [B::Tensor; 6] = backend.modulation_chunks(&projected, 3, 6, config.hidden_size)?.try_into().map_err(|chunks: Vec<B::Tensor>| BackendError::Compute { msg: format!("H3 AdaLN chunks={} 期望 6", chunks.len()) })?;
    let [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = chunks;
    Ok(H3AdaLnParameters { shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp })
}

#[allow(clippy::too_many_arguments)]
pub fn dit_block<B: DiffusionBackend>(
    backend: &B,
    config: &H3Config,
    hidden: &B::Tensor,
    time_embedding: &B::Tensor,
    segments: &[ModulationSegment],
    weights: &H3DitBlockWeights<B::Weight>,
    cosine: &[f32],
    sine: &[f32],
) -> Result<B::Tensor, BackendError> {
    let modulation = block_adaln(backend, config, time_embedding, weights)?;
    let normalized = backend.rmsnorm_adaln_modulate_segmented(hidden, &weights.norm1, config.norm_eps, &modulation.shift_msa, &modulation.scale_msa, segments)?;
    let update = attention(backend, config, &normalized, &weights.attention, cosine, sine)?;
    let hidden = backend.gated_residual_segmented(hidden, &update, &modulation.gate_msa, segments)?;
    let normalized = backend.rmsnorm_adaln_modulate_segmented(&hidden, &weights.norm2, config.norm_eps, &modulation.shift_mlp, &modulation.scale_mlp, segments)?;
    let update = mlp(backend, config, &normalized, &weights.mlp)?;
    backend.gated_residual_segmented(&hidden, &update, &modulation.gate_mlp, segments)
}

/// H3 主干最终只输出视频和音频 velocity；文本与条件 token 不进入输出 head。
pub struct H3VelocityOutput<T> {
    pub video: T,
    pub audio: T,
    /// 音频 sigma 映射相对视频采样轴的导数，由采样器应用到 audio velocity。
    pub audio_velocity_scale: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn dit_final_layer<B: DiffusionBackend>(
    backend: &B,
    config: &H3Config,
    hidden: &B::Tensor,
    time_embedding: &B::Tensor,
    segments: &[ModulationSegment],
    audio_rows: Range<usize>,
    video_rows: Range<usize>,
    weights: &H3DitFinalWeights<B::Weight>,
    audio_velocity_scale: f32,
) -> Result<H3VelocityOutput<B::Tensor>, BackendError> {
    let projected = if config.time_embed_dim == 8 {
        linear_bias(backend, time_embedding, &weights.adaln_weight, &weights.adaln_bias)?
    } else {
        let activated = backend.silu(time_embedding)?;
        linear_bias(backend, &activated, &weights.adaln_weight, &weights.adaln_bias)?
    };
    let chunks: [B::Tensor; 2] = backend.modulation_chunks(&projected, 1, 2, config.hidden_size)?.try_into().map_err(|chunks: Vec<B::Tensor>| BackendError::Compute { msg: format!("H3 final AdaLN chunks={} 期望 2", chunks.len(),) })?;
    let [shift, scale] = chunks;
    let final_segments = segments.iter().map(|segment| ModulationSegment { rows: segment.rows.clone(), modulation_row: segment.modulation_row / 3 }).collect::<Vec<_>>();
    let normalized = backend.rmsnorm(hidden, &weights.norm, config.final_norm_eps)?;
    let normalized = backend.adaln_modulate_segmented(&normalized, &shift, &scale, &final_segments)?;
    let audio_indices = row_indices(audio_rows, "audio")?;
    let video_indices = row_indices(video_rows, "video")?;
    let audio_hidden = backend.select_rows(&normalized, &audio_indices)?;
    let video_hidden = backend.select_rows(&normalized, &video_indices)?;
    let audio = linear_bias(backend, &audio_hidden, &weights.audio_output_weight, &weights.audio_output_bias)?;
    let video = linear_bias(backend, &video_hidden, &weights.video_output_weight, &weights.video_output_bias)?;
    Ok(H3VelocityOutput { video, audio, audio_velocity_scale })
}

fn row_indices(rows: Range<usize>, kind: &str) -> Result<Vec<u32>, BackendError> {
    if rows.is_empty() {
        return Err(BackendError::Compute { msg: format!("H3 {kind} 输出行为空") });
    }
    rows.map(|row| u32::try_from(row).map_err(|_| BackendError::Compute { msg: format!("H3 {kind} row={row} 超出 u32",) })).collect()
}

/// 尚未投影的 H3 四类输入。视频条件按 keyframe 时间顺序排列。
pub struct H3PackedInputs<'a, T> {
    pub text: &'a T,
    pub video_conditions: &'a [T],
    pub audio: &'a T,
    pub video: &'a T,
}

pub enum H3ConditionInput<'a, T> {
    Video(&'a T),
    Audio(&'a T),
}

fn prepare_dit_static_prefix_conditioned<B: DiffusionBackend>(
    backend: &B,
    source: &H3DitSource,
    global: &H3PreparedGlobal<B::Weight>,
    text_input: &B::Tensor,
    conditions: &[H3ConditionInput<'_, B::Tensor>],
) -> Result<B::Tensor, BackendError> {
    let config = source.config();
    let expect_cols = |name: &str, tensor: &B::Tensor, expected: usize| {
        let actual = backend.token_cols(tensor);
        if actual == expected { Ok(()) } else { Err(BackendError::Compute { msg: format!("H3 {name} cols={actual} 期望 {expected}") }) }
    };
    expect_cols("text", text_input, config.text_dim)?;
    for (index, condition) in conditions.iter().enumerate() {
        match condition {
            H3ConditionInput::Video(tensor) => expect_cols(&format!("video condition {index}"), tensor, config.video_patch_dim())?,
            H3ConditionInput::Audio(tensor) => expect_cols(&format!("audio condition {index}"), tensor, config.audio_latent_channels)?,
        }
    }

    let mut text = linear_bias(backend, text_input, &global.weights.condition_weight, &global.weights.condition_bias)?;
    for layer in 0..config.token_refiner_num_layers {
        let _scope = backend.layer_scope();
        let source_weights = source.load_token_refiner_block(layer).map_err(|error| BackendError::ExpertLoad(format!("读取 H3 token refiner block {layer} 失败: {error}")))?;
        let weights = prepare_token_refiner_weights(backend, &source_weights)?;
        backend.begin_batch();
        let result = token_refiner_block(backend, config, &text, &weights);
        backend.finish_stream_chunk();
        text = result?;
    }
    let _scope = backend.layer_scope();
    let final_norm = source.load_token_refiner_final_norm().map_err(|error| BackendError::ExpertLoad(format!("读取 H3 token refiner final norm 失败: {error}")))?;
    let final_norm = prepare_h3_dit_weight(backend, &final_norm)?;
    backend.begin_batch();
    let result = (|| {
        let mut prefix = backend.rmsnorm(&text, &final_norm, config.final_norm_eps)?;
        for condition in conditions {
            let condition = match condition {
                H3ConditionInput::Video(tensor) => linear_bias(backend, tensor, &global.weights.video_patch_weight, &global.weights.video_patch_bias)?,
                H3ConditionInput::Audio(tensor) => linear_bias(backend, tensor, &global.weights.audio_patch_weight, &global.weights.audio_patch_bias)?,
            };
            prefix = backend.concat_rows(&prefix, &condition)?;
        }
        Ok(prefix)
    })();
    backend.finish_batch();
    result
}

fn pack_dit_inputs_with_prefix<B: DiffusionBackend>(
    backend: &B,
    source: &H3DitSource,
    global: &H3PreparedGlobal<B::Weight>,
    prefix: &B::Tensor,
    audio_input: &B::Tensor,
    video_input: &B::Tensor,
    layout: &H3PackedLayout,
) -> Result<B::Tensor, BackendError> {
    let config = source.config();
    if backend.token_rows(prefix) != layout.audio_rows.start || backend.token_cols(prefix) != config.hidden_size {
        return Err(BackendError::Compute { msg: format!("H3 static prefix=[{},{}]，期望 [{},{}]", backend.token_rows(prefix), backend.token_cols(prefix), layout.audio_rows.start, config.hidden_size) });
    }
    if backend.token_cols(audio_input) != config.audio_latent_channels || backend.token_cols(video_input) != config.video_patch_dim() {
        return Err(BackendError::Compute { msg: format!("H3 dynamic input cols audio={} video={}，期望 {}/{}", backend.token_cols(audio_input), backend.token_cols(video_input), config.audio_latent_channels, config.video_patch_dim()) });
    }
    let audio = linear_bias(backend, audio_input, &global.weights.audio_patch_weight, &global.weights.audio_patch_bias)?;
    let mut packed = backend.concat_rows(prefix, &audio)?;
    let video = linear_bias(backend, video_input, &global.weights.video_patch_weight, &global.weights.video_patch_bias)?;
    packed = backend.concat_rows(&packed, &video)?;
    if backend.token_rows(&packed) != layout.seq_len || backend.token_cols(&packed) != config.hidden_size {
        return Err(BackendError::Compute { msg: format!("H3 packed input=[{},{}]，期望 [{},{}]", backend.token_rows(&packed), backend.token_cols(&packed), layout.seq_len, config.hidden_size) });
    }
    Ok(packed)
}

#[allow(clippy::too_many_arguments)]
fn pack_dit_inputs_conditioned<B: DiffusionBackend>(
    backend: &B,
    source: &H3DitSource,
    global: &H3PreparedGlobal<B::Weight>,
    text_input: &B::Tensor,
    conditions: &[H3ConditionInput<'_, B::Tensor>],
    audio_input: &B::Tensor,
    video_input: &B::Tensor,
    layout: &H3PackedLayout,
) -> Result<B::Tensor, BackendError> {
    let prefix = prepare_dit_static_prefix_conditioned(backend, source, global, text_input, conditions)?;
    pack_dit_inputs_with_prefix(backend, source, global, &prefix, audio_input, video_input, layout)
}

pub fn pack_dit_inputs<B: DiffusionBackend>(backend: &B, source: &H3DitSource, global: &H3PreparedGlobal<B::Weight>, inputs: H3PackedInputs<'_, B::Tensor>, layout: &H3PackedLayout) -> Result<B::Tensor, BackendError> {
    let conditions = inputs.video_conditions.iter().map(H3ConditionInput::Video).collect::<Vec<_>>();
    pack_dit_inputs_conditioned(backend, source, global, inputs.text, &conditions, inputs.audio, inputs.video, layout)
}

/// H3 官方 scheduler 的 sigma 网格。`steps` 是包含终点 0 的网格点数，
/// 因此实际 DiT 调用次数是去重后的 `len - 1`。
pub fn h3_sigma_schedule(steps: usize, shift: f32) -> Result<Vec<f32>, String> {
    if steps < 2 || !shift.is_finite() || shift <= 0.0 {
        return Err(format!("H3 scheduler steps={steps} shift={shift} 非法"));
    }
    let denominator = (steps - 1) as f32;
    let delta = -1.0f32 / denominator;
    let half = steps / 2;
    let mut sigmas = Vec::with_capacity(steps);
    for index in 0..steps {
        // 与 torch.linspace 的 float32 两端对称 FMA 顺序一致，避免 shift 后产生不同的 ULP。
        let base = if index < half { delta.mul_add(index as f32, 1.0) } else { (-delta).mul_add((steps - 1 - index) as f32, 0.0) };
        let sigma = shift * base / (1.0 + (shift - 1.0) * base);
        if sigmas.last().copied() != Some(sigma) {
            sigmas.push(sigma);
        }
    }
    if sigmas.len() < 2 || sigmas.last().copied() != Some(0.0) {
        return Err("H3 scheduler shift 后没有有效终点".to_owned());
    }
    Ok(sigmas)
}

pub struct H3DenoiseLatents<T> {
    pub video: T,
    pub audio: T,
}

/// 官方 `MiniMaxH3Scheduler(eta=0)`：data-ward velocity 的 Euler 更新。
fn h3_euler_step<B: DiffusionBackend>(backend: &B, sample: &B::Tensor, velocity: &B::Tensor, sigma: f32, sigma_next: f32) -> Result<B::Tensor, BackendError> {
    backend.flow_step(sample, velocity, sigma - sigma_next)
}

/// Guidance-distilled H3 双调度采样。每个网格区间只执行一次 DiT forward。
#[allow(clippy::too_many_arguments)]
pub fn denoise<B: DiffusionBackend>(
    backend: &B,
    source: &H3DitSource,
    global: &H3PreparedGlobal<B::Weight>,
    text: &B::Tensor,
    video_conditions: &[B::Tensor],
    latents: H3DenoiseLatents<B::Tensor>,
    layout: &H3PackedLayout,
    num_inference_steps: usize,
) -> Result<H3DenoiseLatents<B::Tensor>, BackendError> {
    let conditions = video_conditions.iter().map(H3ConditionInput::Video).collect::<Vec<_>>();
    denoise_conditioned(backend, source, global, text, &conditions, latents, layout, num_inference_steps)
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_conditioned<B: DiffusionBackend>(
    backend: &B,
    source: &H3DitSource,
    global: &H3PreparedGlobal<B::Weight>,
    text: &B::Tensor,
    conditions: &[H3ConditionInput<'_, B::Tensor>],
    mut latents: H3DenoiseLatents<B::Tensor>,
    layout: &H3PackedLayout,
    num_inference_steps: usize,
) -> Result<H3DenoiseLatents<B::Tensor>, BackendError> {
    let video_sigmas = h3_sigma_schedule(num_inference_steps, source.config().sigma_shift_video).map_err(|msg| BackendError::Compute { msg })?;
    let audio_sigmas = h3_sigma_schedule(num_inference_steps, source.config().sigma_shift_audio).map_err(|msg| BackendError::Compute { msg })?;
    if audio_sigmas.len() != video_sigmas.len() {
        return Err(BackendError::Compute { msg: format!("H3 video/audio sigma 网格长度不同：{}/{}", video_sigmas.len(), audio_sigmas.len()) });
    }
    let prefix = prepare_dit_static_prefix_conditioned(backend, source, global, text, conditions)?;
    for index in 0..video_sigmas.len() - 1 {
        let hidden = pack_dit_inputs_with_prefix(backend, source, global, &prefix, &latents.audio, &latents.video, layout)?;
        let output = dit_backbone_forward(backend, source, global, hidden, layout, video_sigmas[index])?;
        latents.video = h3_euler_step(backend, &latents.video, &output.video, video_sigmas[index], video_sigmas[index + 1])?;
        latents.audio = h3_euler_step(backend, &latents.audio, &output.audio, audio_sigmas[index], audio_sigmas[index + 1])?;
    }
    Ok(latents)
}

/// `[N,C,T,H,W] -> [N*t*h*w,C*pt*ph*pw]`。
pub fn patchify_video(latent: &[f32], shape: [usize; 5], patch: [usize; 3]) -> Result<Vec<f32>, String> {
    if shape.contains(&0) || patch.contains(&0) {
        return Err("H3 patchify shape 不能含 0".to_owned());
    }
    if !shape[2].is_multiple_of(patch[0]) || !shape[3].is_multiple_of(patch[1]) || !shape[4].is_multiple_of(patch[2]) {
        return Err(format!("H3 video shape {shape:?} 不能被 patch {patch:?} 整除"));
    }
    let elements = checked_product(&shape, "H3 video")?;
    if latent.len() != elements {
        return Err(format!("H3 video 数据长度={} 期望 {elements}", latent.len()));
    }
    let [batch, channels, frames, height, width] = shape;
    let [pt, ph, pw] = patch;
    let [out_t, out_h, out_w] = [frames / pt, height / ph, width / pw];
    let patch_dim = channels * pt * ph * pw;
    let mut rows = vec![0.0; elements];
    for n in 0..batch {
        for t in 0..out_t {
            for h in 0..out_h {
                for w in 0..out_w {
                    let row = ((n * out_t + t) * out_h + h) * out_w + w;
                    for c in 0..channels {
                        for kt in 0..pt {
                            for kh in 0..ph {
                                for kw in 0..pw {
                                    let column = ((c * pt + kt) * ph + kh) * pw + kw;
                                    let source = ((((n * channels + c) * frames + t * pt + kt) * height + h * ph + kh) * width) + w * pw + kw;
                                    rows[row * patch_dim + column] = latent[source];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(rows)
}

/// `patchify_video` 的逆变换。
pub fn unpatchify_video(rows: &[f32], shape: [usize; 5], patch: [usize; 3]) -> Result<Vec<f32>, String> {
    let mut latent = vec![0.0; checked_product(&shape, "H3 video")?];
    if rows.len() != latent.len() || shape.contains(&0) || patch.contains(&0) || !shape[2].is_multiple_of(patch[0]) || !shape[3].is_multiple_of(patch[1]) || !shape[4].is_multiple_of(patch[2]) {
        return Err(format!("H3 unpatchify rows={} shape={shape:?} patch={patch:?} 不兼容", rows.len()));
    }
    let [batch, channels, frames, height, width] = shape;
    let [pt, ph, pw] = patch;
    let [out_t, out_h, out_w] = [frames / pt, height / ph, width / pw];
    let patch_dim = channels * pt * ph * pw;
    for n in 0..batch {
        for t in 0..out_t {
            for h in 0..out_h {
                for w in 0..out_w {
                    let row = ((n * out_t + t) * out_h + h) * out_w + w;
                    for c in 0..channels {
                        for kt in 0..pt {
                            for kh in 0..ph {
                                for kw in 0..pw {
                                    let column = ((c * pt + kt) * ph + kh) * pw + kw;
                                    let target = ((((n * channels + c) * frames + t * pt + kt) * height + h * ph + kh) * width) + w * pw + kw;
                                    latent[target] = rows[row * patch_dim + column];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(latent)
}

/// `[C,stereo,T] -> [stereo*T,C]`，行按声道优先。
pub fn pack_audio(latent: &[f32], channels: usize, stereo: usize, frames: usize) -> Result<Vec<f32>, String> {
    let elements = checked_product(&[channels, stereo, frames], "H3 audio")?;
    if channels == 0 || stereo == 0 || frames == 0 || latent.len() != elements {
        return Err(format!("H3 audio 数据长度={} 期望 {elements}", latent.len()));
    }
    let mut rows = vec![0.0; elements];
    for channel in 0..stereo {
        for frame in 0..frames {
            let row = channel * frames + frame;
            for latent_channel in 0..channels {
                rows[row * channels + latent_channel] = latent[(latent_channel * stereo + channel) * frames + frame];
            }
        }
    }
    Ok(rows)
}

pub fn unpack_audio(rows: &[f32], channels: usize, stereo: usize) -> Result<Vec<f32>, String> {
    if channels == 0 || stereo == 0 || !rows.len().is_multiple_of(channels * stereo) {
        return Err(format!("H3 audio rows={} 与 channels={channels} stereo={stereo} 不兼容", rows.len()));
    }
    let frames = rows.len() / (channels * stereo);
    let mut latent = vec![0.0; rows.len()];
    for channel in 0..stereo {
        for frame in 0..frames {
            let row = channel * frames + frame;
            for latent_channel in 0..channels {
                latent[(latent_channel * stereo + channel) * frames + frame] = rows[row * channels + latent_channel];
            }
        }
    }
    Ok(latent)
}

fn checked_product(shape: &[usize], what: &str) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |size, &dim| size.checked_mul(dim).ok_or_else(|| format!("{what} shape 大小溢出")))
}

fn spatial_axis(dim: usize, patch: usize, square_area: f32) -> Vec<f32> {
    let ratio = dim as f32 / square_area;
    let count = dim / patch;
    (0..count).map(|index| (index as f32 * (ratio / count as f32) + (1.0 - ratio) / 2.0) * 32.0).collect()
}

fn spatial_frame_grid(height: usize, width: usize, patch_h: usize, patch_w: usize) -> (Vec<[f32; 2]>, Vec<f32>) {
    let square_area = (height as f32 * width as f32).sqrt();
    let height_axis = spatial_axis(height, patch_h, square_area);
    let width_axis = spatial_axis(width, patch_w, square_area);
    let frame = height_axis.iter().flat_map(|&h| width_axis.iter().map(move |&w| [h, w])).collect();
    (frame, width_axis)
}

fn video_time_spans(frames: usize) -> Vec<f32> {
    (0..frames).map(|frame| FRAME_RESCALE * FRAME_PER_TOKEN[frame % FRAME_PER_TOKEN.len()] as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_patch_rows_round_trip() {
        let shape = [1, 2, 2, 4, 4];
        let source = (0..64).map(|value| value as f32).collect::<Vec<_>>();
        let rows = patchify_video(&source, shape, [1, 2, 2]).unwrap();
        assert_eq!(&rows[..8], &[0.0, 1.0, 4.0, 5.0, 32.0, 33.0, 36.0, 37.0]);
        assert_eq!(unpatchify_video(&rows, shape, [1, 2, 2]).unwrap(), source);
    }

    #[test]
    fn audio_rows_are_channel_major_and_reversible() {
        // [latent_channel=2, stereo=2, frames=3]
        let source = vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0];
        let rows = pack_audio(&source, 2, 2, 3).unwrap();
        assert_eq!(rows, vec![0.0, 20.0, 1.0, 21.0, 2.0, 22.0, 10.0, 30.0, 11.0, 31.0, 12.0, 32.0]);
        assert_eq!(unpack_audio(&rows, 2, 2).unwrap(), source);
    }

    #[test]
    fn fl2va_layout_matches_official_segment_order() {
        let layout = build_packed_layout(3, 2, 4, 4, 2, [1, 2, 2], &[H3KeyframeAnchor::First, H3KeyframeAnchor::Last]).unwrap();
        assert_eq!(layout.seq_len, 23);
        assert_eq!(layout.audio_rows, 11..15);
        assert_eq!(layout.video_rows, 15..23);
        assert_eq!(layout.segments.iter().map(|segment| segment.kind).collect::<Vec<_>>(), vec![H3SegmentKind::Text, H3SegmentKind::VideoCondition, H3SegmentKind::VideoCondition, H3SegmentKind::Audio, H3SegmentKind::Video,]);
        assert_eq!(layout.position_ids[3][0], 3.0);
        assert!((layout.position_ids[7][0] - (3.0 + 20.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn dual_schedule_builds_three_modality_rows_per_time() {
        let config = H3Config::standard();
        let time = h3_time_state(0.5, &config).unwrap();
        assert!((time.video - 0.5).abs() < 1e-6);
        assert!((time.audio - 0.8).abs() < 1e-6);
        let layout = build_packed_layout(3, 2, 4, 4, 2, [1, 2, 2], &[H3KeyframeAnchor::First]).unwrap();
        let plan = build_modulation_plan(&layout, time);
        assert_eq!(plan.unique_timesteps.len(), 3);
        assert_eq!(plan.segments.iter().map(|segment| segment.modulation_row).collect::<Vec<_>>(), vec![1, 6, 5, 0]);
    }

    #[test]
    fn official_scheduler_uses_twenty_grid_points_and_nineteen_forwards() {
        let sigmas = h3_sigma_schedule(20, 12.0).unwrap();
        assert_eq!(sigmas.len(), 20);
        assert_eq!(sigmas[0].to_bits(), 0x3f80_0000);
        assert_eq!(sigmas[1].to_bits(), 0x3f7e_d1fe);
        assert_eq!(sigmas[18].to_bits(), 0x3ecc_cccd);
        assert_eq!(sigmas[19].to_bits(), 0);
        assert!(sigmas.windows(2).all(|pair| pair[1] < pair[0]));
    }

    #[test]
    fn mm_rope_tables_follow_time_height_width_order() {
        let (cosine, sine) = mm_rope_tables(&[[1.0, 2.0, 3.0]], &[1.0, 2.0]).unwrap();
        let angles = [1.0_f32, 2.0, 2.0, 4.0, 3.0, 6.0];
        for index in 0..angles.len() {
            assert!((cosine[index] - angles[index].cos()).abs() < 1e-6);
            assert!((sine[index] - angles[index].sin()).abs() < 1e-6);
        }
    }

    #[test]
    fn standard_matches_official_patch_and_attention_shapes() {
        let config = H3Config::standard();
        config.validate().unwrap();
        assert_eq!(config.attention_dim(), 7168);
        assert_eq!(config.video_patch_dim(), 96);
    }

    #[test]
    fn ulysses_plan_balances_sequence_and_partitions_heads() {
        let plan = H3UlyssesPlan::new(11, 56, 4).unwrap();
        assert_eq!(plan.shards, vec![H3UlyssesShard { sequence: 0..3, heads: 0..14 }, H3UlyssesShard { sequence: 3..6, heads: 14..28 }, H3UlyssesShard { sequence: 6..9, heads: 28..42 }, H3UlyssesShard { sequence: 9..11, heads: 42..56 },]);
        assert_eq!(plan.shards.iter().map(|shard| shard.sequence.len()).sum::<usize>(), 11);
    }

    #[test]
    fn ulysses_plan_rejects_unsupported_rank_or_head_split() {
        assert!(H3UlyssesPlan::new(8, 56, 3).unwrap_err().contains("1/2/4/8"));
        assert!(H3UlyssesPlan::new(8, 10, 4).unwrap_err().contains("不能整除"));
        assert!(H3UlyssesPlan::new(2, 56, 4).unwrap_err().contains("小于"));
    }

    #[test]
    fn ulysses_plan_slices_modulation_to_rank_local_rows() {
        let plan = H3UlyssesPlan::new(11, 56, 4).unwrap();
        let segments =
            vec![ModulationSegment { rows: 0..2, modulation_row: 1 }, ModulationSegment { rows: 2..5, modulation_row: 6 }, ModulationSegment { rows: 5..8, modulation_row: 5 }, ModulationSegment { rows: 8..11, modulation_row: 0 }];
        assert_eq!(plan.local_modulation_segments(1, &segments).unwrap(), vec![ModulationSegment { rows: 0..2, modulation_row: 6 }, ModulationSegment { rows: 2..3, modulation_row: 5 }]);
        assert_eq!(plan.local_modulation_segments(3, &segments).unwrap(), vec![ModulationSegment { rows: 0..2, modulation_row: 0 }]);
    }
}

// MiniMax H3 全模态生成模型规格。
//
pub use crate::model_spec::h3::H3Config;

mod audio_vae;
mod conditioning;
#[cfg(feature = "with-cuda")]
mod cuda;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod node;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
mod rocm;
mod staging;
mod video_vae;
pub use audio_vae::*;
pub use conditioning::*;
#[cfg(feature = "with-cuda")]
pub use cuda::*;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub use rocm::*;
pub use staging::*;
pub use video_vae::*;
