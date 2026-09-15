//! SeedVR2 VAE 的原生模型编排。卷积为 C×THW，DiT latent 为 THW×C。
//!
//! 时间缓存仅属于一次 encode/decode 调用；分片之间传尾部，调用结束即释放。
//! 所有 activation 操作经已有 backend capability，生产路径不回读 CPU。

use crate::{
    backend::{Backend, BackendError, BackendResources, LinearWeight, SegmentedTensorBackend, VaeBackend},
    model_spec::seedvr2::SeedVr2VaeConfig,
    vae::{Conv3dSpec, PixelShuffleSpec},
    weight::{
        container::safetensor::TensorData,
        model::seedvr2::{SeedVr2Linear, SeedVr2VaeAttention, SeedVr2VaeConv, SeedVr2VaeDecoder, SeedVr2VaeDownBlock, SeedVr2VaeEncoder, SeedVr2VaeMid, SeedVr2VaeNorm, SeedVr2VaeResnet, SeedVr2VaeSource},
    },
};

fn error(message: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: message.into() }
}

fn batch<B: BackendResources, T>(b: &B, f: impl FnOnce() -> Result<T, BackendError>) -> Result<T, BackendError> {
    b.begin_batch();
    let result = f();
    // 回调或算子失败时同样恢复后端批次状态。
    b.finish_batch();
    result
}

fn product(shape: [usize; 3]) -> Result<usize, BackendError> {
    if shape.contains(&0) {
        return Err(error(format!("SeedVR2 VAE shape含零维度: {shape:?}")));
    }
    shape.into_iter().try_fold(1usize, |n, v| n.checked_mul(v).ok_or_else(|| error("SeedVR2 VAE shape溢出")))
}

/// 一个样本的时空张量；RGB为C×THW，公开latent为THW×C，shape始终为[T,H,W]。
pub struct VaeVideo<T> {
    pub tensor: T,
    pub shape: [usize; 3],
}

pub struct PreparedUpsample<W> {
    /// 每个时间子相位独立的 C→Cxy 投影，输出行顺序为(c,x,y)。
    projections: Vec<SeedVr2Linear<W>>,
    conv: SeedVr2VaeConv<W>,
}

pub struct PreparedUpBlock<W> {
    resnets: Vec<SeedVr2VaeResnet<W>>,
    upsample: Option<PreparedUpsample<W>>,
}

pub struct PreparedDecoder<W> {
    conv_in: SeedVr2VaeConv<W>,
    mid: SeedVr2VaeMid<W>,
    up_blocks: Vec<PreparedUpBlock<W>>,
    conv_norm_out: SeedVr2VaeNorm<W>,
    conv_out: SeedVr2VaeConv<W>,
}

pub struct PreparedVae<W> {
    pub config: SeedVr2VaeConfig,
    pub encoder: SeedVr2VaeEncoder<W>,
    pub decoder: PreparedDecoder<W>,
}

fn prepare_tensor<B: Backend>(backend: &B, value: &TensorData) -> Result<B::Weight, BackendError> {
    let (&rows, suffix) = value.shape.split_first().ok_or_else(|| error(format!("VAE权重{}为空shape", value.name)))?;
    let cols = suffix.iter().try_fold(1usize, |n, &v| n.checked_mul(v).ok_or_else(|| error("VAE权重shape溢出")))?;
    // 保留checkpoint数值；具体平台自行选择计算表示，不借用H3的权重量化规则。
    let values = value.to_f32().map_err(BackendError::ExpertLoad)?;
    backend.prepare_weight(LinearWeight::F32(&values), rows, cols)
}

fn prepare_conv<B: Backend>(b: &B, w: &SeedVr2VaeConv) -> Result<SeedVr2VaeConv<B::Weight>, BackendError> {
    Ok(SeedVr2VaeConv { weight: prepare_tensor(b, &w.weight)?, bias: prepare_tensor(b, &w.bias)? })
}

fn prepare_linear<B: Backend>(b: &B, w: &SeedVr2Linear) -> Result<SeedVr2Linear<B::Weight>, BackendError> {
    Ok(SeedVr2Linear { weight: prepare_tensor(b, &w.weight)?, bias: prepare_tensor(b, &w.bias)? })
}

fn prepare_norm<B: Backend>(b: &B, w: &SeedVr2VaeNorm) -> Result<SeedVr2VaeNorm<B::Weight>, BackendError> {
    Ok(SeedVr2VaeNorm { weight: prepare_tensor(b, &w.weight)?, bias: prepare_tensor(b, &w.bias)? })
}

fn prepare_resnet<B: Backend>(b: &B, w: &SeedVr2VaeResnet) -> Result<SeedVr2VaeResnet<B::Weight>, BackendError> {
    Ok(SeedVr2VaeResnet { norm1: prepare_norm(b, &w.norm1)?, conv1: prepare_conv(b, &w.conv1)?, norm2: prepare_norm(b, &w.norm2)?, conv2: prepare_conv(b, &w.conv2)?, shortcut: w.shortcut.as_ref().map(|w| prepare_conv(b, w)).transpose()? })
}

fn prepare_mid<B: Backend>(b: &B, w: &SeedVr2VaeMid) -> Result<SeedVr2VaeMid<B::Weight>, BackendError> {
    let a = &w.attention;
    Ok(SeedVr2VaeMid {
        resnets: [prepare_resnet(b, &w.resnets[0])?, prepare_resnet(b, &w.resnets[1])?],
        attention: SeedVr2VaeAttention { group_norm: prepare_norm(b, &a.group_norm)?, query: prepare_linear(b, &a.query)?, key: prepare_linear(b, &a.key)?, value: prepare_linear(b, &a.value)?, output: prepare_linear(b, &a.output)? },
    })
}

/// 原行序(x,y,z,c)只在准备时重排；各投影仍读取同一输入，没有中间activation依赖。
fn prepare_upscale<B: Backend>(b: &B, w: &SeedVr2VaeConv, channels: usize, temporal: usize) -> Result<Vec<SeedVr2Linear<B::Weight>>, BackendError> {
    let rows = channels.checked_mul(4).ok_or_else(|| error("VAE upscale channels溢出"))?;
    if w.weight.shape != [rows * temporal, channels, 1, 1, 1] || w.bias.shape != [rows * temporal] {
        return Err(error(format!("VAE upscale权重{} shape={:?}，期望 [{},{channels},1,1,1]", w.weight.name, w.weight.shape, rows * temporal)));
    }
    let weight = w.weight.to_f32().map_err(BackendError::ExpertLoad)?;
    let bias = w.bias.to_f32().map_err(BackendError::ExpertLoad)?;
    (0..temporal)
        .map(|z| {
            let mut matrix = Vec::with_capacity(rows * channels);
            let mut offset = Vec::with_capacity(rows);
            for c in 0..channels {
                for x in 0..2 {
                    for y in 0..2 {
                        let row = ((x * 2 + y) * temporal + z) * channels + c;
                        matrix.extend_from_slice(&weight[row * channels..(row + 1) * channels]);
                        offset.push(bias[row]);
                    }
                }
            }
            Ok(SeedVr2Linear { weight: b.prepare_weight(LinearWeight::F32(&matrix), rows, channels)?, bias: b.prepare_weight(LinearWeight::F32(&offset), rows, 1)? })
        })
        .collect()
}

pub fn prepare_vae<B: VaeBackend>(b: &B, source: &SeedVr2VaeSource) -> Result<PreparedVae<B::Weight>, BackendError> {
    let config = source.config().clone();
    let e = source.load_encoder().map_err(BackendError::ExpertLoad)?;
    let d = source.load_decoder().map_err(BackendError::ExpertLoad)?;
    prepare_vae_weights(b, config, &e, &d)
}

fn prepare_vae_weights<B: VaeBackend>(b: &B, config: SeedVr2VaeConfig, e: &SeedVr2VaeEncoder, d: &SeedVr2VaeDecoder) -> Result<PreparedVae<B::Weight>, BackendError> {
    config.validate().map_err(error)?;
    if e.down_blocks.len() != config.block_channels.len()
        || d.up_blocks.len() != config.block_channels.len()
        || e.down_blocks.iter().any(|w| w.resnets.len() != config.layers_per_block)
        || d.up_blocks.iter().any(|w| w.resnets.len() != config.layers_per_block + 1)
    {
        return Err(error("SeedVR2 VAE block/resnet数与配置不一致"));
    }
    let encoder = SeedVr2VaeEncoder {
        conv_in: prepare_conv(b, &e.conv_in)?,
        down_blocks: e
            .down_blocks
            .iter()
            .map(|w| Ok(SeedVr2VaeDownBlock { resnets: w.resnets.iter().map(|w| prepare_resnet(b, w)).collect::<Result<_, _>>()?, downsample: w.downsample.as_ref().map(|w| prepare_conv(b, w)).transpose()? }))
            .collect::<Result<_, BackendError>>()?,
        mid: prepare_mid(b, &e.mid)?,
        conv_norm_out: prepare_norm(b, &e.conv_norm_out)?,
        conv_out: prepare_conv(b, &e.conv_out)?,
    };
    let decoder = PreparedDecoder {
        conv_in: prepare_conv(b, &d.conv_in)?,
        mid: prepare_mid(b, &d.mid)?,
        up_blocks: d
            .up_blocks
            .iter()
            .enumerate()
            .map(|(i, w)| {
                Ok(PreparedUpBlock {
                    resnets: w.resnets.iter().map(|w| prepare_resnet(b, w)).collect::<Result<_, _>>()?,
                    upsample: w
                        .upsample
                        .as_ref()
                        .map(|u| Ok(PreparedUpsample { projections: prepare_upscale(b, &u.upscale_conv, config.block_channels[config.block_channels.len() - 1 - i], if i < 2 { 2 } else { 1 })?, conv: prepare_conv(b, &u.conv)? }))
                        .transpose()?,
                })
            })
            .collect::<Result<_, BackendError>>()?,
        conv_norm_out: prepare_norm(b, &d.conv_norm_out)?,
        conv_out: prepare_conv(b, &d.conv_out)?,
    };
    Ok(PreparedVae { config, encoder, decoder })
}

fn transpose<B: VaeBackend>(b: &B, tensor: &B::Tensor) -> Result<B::Tensor, BackendError> {
    b.channels_to_time(tensor, b.token_rows(tensor))
}

fn validate_video<B: BackendResources>(b: &B, v: &VaeVideo<B::Tensor>, channels: usize) -> Result<(), BackendError> {
    if b.token_rows(&v.tensor) != channels || b.token_cols(&v.tensor) != product(v.shape)? {
        return Err(error(format!("SeedVR2 VAE tensor=[{},{}]，期望 [{channels},{}] shape={:?}", b.token_rows(&v.tensor), b.token_cols(&v.tensor), product(v.shape)?, v.shape)));
    }
    Ok(())
}

/// cache是紧凑C×尾部THW；不能直接保存ROCm slice view，否则会持有整个输入allocation。
pub(crate) struct VaeStageCache<T> {
    tails: Vec<Option<T>>,
    cursor: usize,
    first: bool,
}

impl<T> VaeStageCache<T> {
    pub(crate) fn new() -> Self {
        Self { tails: Vec::new(), cursor: 0, first: true }
    }
    fn next_chunk(&mut self) {
        self.cursor = 0;
        self.first = false;
    }
}

#[allow(clippy::too_many_arguments)]
fn conv<B: VaeBackend + SegmentedTensorBackend>(
    b: &B,
    x: &VaeVideo<B::Tensor>,
    w: &SeedVr2VaeConv<B::Weight>,
    output_channels: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
    spatial_padding: usize,
    pad_after: [usize; 2],
    cache: &mut VaeStageCache<B::Tensor>,
) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let channels = b.token_rows(&x.tensor);
    validate_video(b, x, channels)?;
    let index = cache.cursor;
    cache.cursor += 1;
    if index == cache.tails.len() {
        cache.tails.push(None);
    }
    let previous = cache.tails[index].take();
    if !cache.first && kernel[0] > stride[0] && previous.is_none() {
        return Err(error(format!("SeedVR2 VAE缺少第{index}层时间cache")));
    }
    let plane = x.shape[1] * x.shape[2];
    let spec = Conv3dSpec { input_channels: channels, output_channels, input_shape: x.shape, kernel, stride, padding: [0, spatial_padding, spatial_padding], causal: false };
    let history_frames = previous.as_ref().map(|tail| b.token_cols(tail) / plane);
    let (_, _, shape) = crate::vae::conv3d_history_layout(&spec, history_frames, pad_after).map_err(error)?;
    // 后端直接读current/history并生成紧凑尾部，模型不再物化整份时间拼接。
    let (tensor, tail) = b.conv3d_with_history(&x.tensor, previous, &w.weight, Some(&w.bias), &spec, pad_after)?;
    cache.tails[index] = tail;
    Ok(VaeVideo { tensor, shape })
}

fn norm_silu<B: VaeBackend>(b: &B, x: &VaeVideo<B::Tensor>, w: &SeedVr2VaeNorm<B::Weight>, c: &SeedVr2VaeConfig) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let norm = b.group_norm_time_isolated(&x.tensor, &w.weight, &w.bias, x.shape[0], c.norm_groups, c.norm_eps)?;
    Ok(VaeVideo { tensor: b.silu(&norm)?, shape: x.shape })
}

fn resnet<B: VaeBackend + SegmentedTensorBackend>(b: &B, x: VaeVideo<B::Tensor>, w: &SeedVr2VaeResnet<B::Weight>, channels: usize, c: &SeedVr2VaeConfig, cache: &mut VaeStageCache<B::Tensor>) -> Result<VaeVideo<B::Tensor>, BackendError> {
    // 赋值立即释放上一份activation；连续let遮蔽会把旧GPU tensor留到作用域末。
    let mut h = norm_silu(b, &x, &w.norm1, c)?;
    h = conv(b, &h, &w.conv1, channels, [3; 3], [1; 3], 1, [0; 2], cache)?;
    h = norm_silu(b, &h, &w.norm2, c)?;
    h = conv(b, &h, &w.conv2, channels, [3; 3], [1; 3], 1, [0; 2], cache)?;
    let mut x = x;
    if let Some(w) = &w.shortcut {
        x = conv(b, &x, w, channels, [1; 3], [1; 3], 0, [0; 2], cache)?;
    }
    if x.shape != h.shape {
        return Err(error("SeedVR2 VAE residual时空shape不一致"));
    }
    Ok(VaeVideo { tensor: b.add(&x.tensor, &h.tensor)?, shape: x.shape })
}

fn linear<B: VaeBackend>(b: &B, x: &B::Tensor, w: &SeedVr2Linear<B::Weight>) -> Result<B::Tensor, BackendError> {
    b.add_row_bias(&b.linear(x, &w.weight)?, &w.bias)
}

fn mid<B: VaeBackend + SegmentedTensorBackend>(b: &B, x: VaeVideo<B::Tensor>, w: &SeedVr2VaeMid<B::Weight>, c: &SeedVr2VaeConfig, cache: &mut VaeStageCache<B::Tensor>) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let channels = *c.block_channels.last().ok_or_else(|| error("VAE block_channels为空"))?;
    let mut x = resnet(b, x, &w.resnets[0], channels, c, cache)?;
    x = attention(b, x, &w.attention, c)?;
    resnet(b, x, &w.resnets[1], channels, c, cache)
}

fn attention<B: VaeBackend + SegmentedTensorBackend>(b: &B, x: VaeVideo<B::Tensor>, a: &SeedVr2VaeAttention<B::Weight>, c: &SeedVr2VaeConfig) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let channels = *c.block_channels.last().ok_or_else(|| error("VAE block_channels为空"))?;
    let norm = b.group_norm_time_isolated(&x.tensor, &a.group_norm.weight, &a.group_norm.bias, x.shape[0], c.norm_groups, c.norm_eps)?;
    let tokens = transpose(b, &norm)?;
    let spatial = x.shape[1] * x.shape[2];
    let mut outputs = Vec::with_capacity(x.shape[0]);
    for t in 0..x.shape[0] {
        let frame = b.slice_token_rows(&tokens, t * spatial, spatial)?;
        let q = linear(b, &frame, &a.query)?;
        let k = linear(b, &frame, &a.key)?;
        let v = linear(b, &frame, &a.value)?;
        // 每帧全空间KV，单head；不得把时间拼入同一个attention序列。
        let attended = b.full_attention(q, k, v, 1, channels, (channels as f32).sqrt().recip())?;
        outputs.push(linear(b, &attended, &a.output)?);
    }
    let refs = outputs.iter().collect::<Vec<_>>();
    let update = transpose(b, &b.concat_token_rows(&refs)?)?;
    Ok(VaeVideo { tensor: b.add(&x.tensor, &update)?, shape: x.shape })
}

fn upscale<B: VaeBackend + SegmentedTensorBackend>(b: &B, x: VaeVideo<B::Tensor>, w: &PreparedUpsample<B::Weight>, first: bool) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let channels = b.token_rows(&x.tensor);
    let shape = x.shape;
    let temporal = w.projections.len();
    if !(temporal == 1 || temporal == 2) {
        return Err(error("SeedVR2 VAE上采样时间倍率非法"));
    }
    let h = shape[1].checked_mul(2).ok_or_else(|| error("VAE upsample height溢出"))?;
    let width = shape[2].checked_mul(2).ok_or_else(|| error("VAE upsample width溢出"))?;
    let shuffle = PixelShuffleSpec { channels, height: shape[0] * shape[1], width: shape[2], upscale: 2 };
    // 上采样投影本就是1×1×1卷积；保持通道布局，省去投影前后的整幅转置。
    let spec = Conv3dSpec { input_channels: channels, output_channels: channels * 4, input_shape: shape, kernel: [1; 3], stride: [1; 3], padding: [0; 3], causal: false };
    if temporal == 1 {
        let projected = b.conv3d(&x.tensor, &w.projections[0].weight, Some(&w.projections[0].bias), &spec)?;
        return Ok(VaeVideo { tensor: b.pixel_shuffle(&projected, &shuffle)?, shape: [shape[0], h, width] });
    }
    let mut phases = Vec::with_capacity(temporal);
    for projection in &w.projections {
        let projected = b.conv3d(&x.tensor, &projection.weight, Some(&projection.bias), &spec)?;
        let shuffled = b.pixel_shuffle(&projected, &shuffle)?;
        drop(projected);
        phases.push(transpose(b, &shuffled)?);
    }
    drop(x);
    let mut frames = Vec::with_capacity(shape[0] * temporal);
    for t in 0..shape[0] {
        for (z, phase) in phases.iter().enumerate() {
            // 上游remove_head保留第0帧，删除初片第1帧；不是丢掉首帧。
            if first && temporal == 2 && t == 0 && z == 1 {
                continue;
            }
            frames.push(b.slice_token_rows(phase, t * h * width, h * width)?);
        }
    }
    let depth = frames.len();
    let refs = frames.iter().collect::<Vec<_>>();
    let joined = b.concat_token_rows(&refs)?;
    drop(refs);
    drop(frames);
    drop(phases);
    Ok(VaeVideo { tensor: transpose(b, &joined)?, shape: [depth, h, width] })
}

fn encode_chunk<B: VaeBackend + SegmentedTensorBackend>(b: &B, x: VaeVideo<B::Tensor>, w: &PreparedVae<B::Weight>, cache: &mut VaeStageCache<B::Tensor>) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let c = &w.config;
    let e = &w.encoder;
    let mut x = conv(b, &x, &e.conv_in, c.block_channels[0], [3; 3], [1; 3], 1, [0; 2], cache)?;
    for (i, block) in e.down_blocks.iter().enumerate() {
        let channels = c.block_channels[i];
        for weights in &block.resnets {
            x = resnet(b, x, weights, channels, c, cache)?;
        }
        if let Some(weights) = &block.downsample {
            let temporal = if i == 0 { 1 } else { 2 };
            x = conv(b, &x, weights, channels, [if temporal == 1 { 1 } else { 3 }, 3, 3], [temporal, 2, 2], 0, [1; 2], cache)?;
        }
    }
    x = mid(b, x, &e.mid, c, cache)?;
    x = norm_silu(b, &x, &e.conv_norm_out, c)?;
    x = conv(b, &x, &e.conv_out, c.latent_channels * 2, [3; 3], [1; 3], 1, [0; 2], cache)?;
    // 后验mode只使用前半mean；本模型没有quant_conv，也不采样posterior噪声。
    let tensor = b.slice_token_rows(&x.tensor, 0, c.latent_channels)?;
    Ok(VaeVideo { tensor, shape: x.shape })
}

fn decode_chunk<B: VaeBackend + SegmentedTensorBackend>(b: &B, x: VaeVideo<B::Tensor>, w: &PreparedVae<B::Weight>, cache: &mut VaeStageCache<B::Tensor>) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let c = &w.config;
    let d = &w.decoder;
    let mut x = conv(b, &x, &d.conv_in, *c.block_channels.last().unwrap(), [3; 3], [1; 3], 1, [0; 2], cache)?;
    x = mid(b, x, &d.mid, c, cache)?;
    for (i, block) in d.up_blocks.iter().enumerate() {
        let channels = c.block_channels[c.block_channels.len() - 1 - i];
        for weights in &block.resnets {
            x = resnet(b, x, weights, channels, c, cache)?;
        }
        if let Some(up) = &block.upsample {
            x = upscale(b, x, up, cache.first)?;
            x = conv(b, &x, &up.conv, channels, [3; 3], [1; 3], 1, [0; 2], cache)?;
        }
    }
    x = norm_silu(b, &x, &d.conv_norm_out, c)?;
    conv(b, &x, &d.conv_out, c.out_channels, [3; 3], [1; 3], 1, [0; 2], cache)
}

/// 每段只保存自身卷积的tail，必须按同一视频chunk顺序调用；新视频创建新的cache。
/// 段0接RGB C×THW，段7输出已scale的THW×latent，其余段交换C×THW。
pub(crate) fn encode_stage<B: VaeBackend + SegmentedTensorBackend>(b: &B, stage: usize, input: VaeVideo<B::Tensor>, weights: &PreparedVae<B::Weight>, cache: &mut VaeStageCache<B::Tensor>) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let result = batch(b, || {
        let c = &weights.config;
        let e = &weights.encoder;
        let mut x = input;
        match stage {
            0 => {
                validate_video(b, &x, c.in_channels)?;
                x = conv(b, &x, &e.conv_in, c.block_channels[0], [3; 3], [1; 3], 1, [0; 2], cache)?;
                x = resnet(b, x, &e.down_blocks[0].resnets[0], c.block_channels[0], c, cache)?;
            }
            1 => {
                x = resnet(b, x, &e.down_blocks[0].resnets[1], c.block_channels[0], c, cache)?;
                let w = e.down_blocks[0].downsample.as_ref().ok_or_else(|| error("VAE encoder stage1缺少downsample"))?;
                x = conv(b, &x, w, c.block_channels[0], [1, 3, 3], [1, 2, 2], 0, [1; 2], cache)?;
            }
            2..=4 => {
                let i = stage - 1;
                let channels = c.block_channels[i];
                for w in &e.down_blocks[i].resnets {
                    x = resnet(b, x, w, channels, c, cache)?;
                }
                if let Some(w) = &e.down_blocks[i].downsample {
                    x = conv(b, &x, w, channels, [3; 3], [2; 3], 0, [1; 2], cache)?;
                }
            }
            5 => {
                x = resnet(b, x, &e.mid.resnets[0], c.block_channels[3], c, cache)?;
            }
            6 => {
                x = attention(b, x, &e.mid.attention, c)?;
                x = resnet(b, x, &e.mid.resnets[1], c.block_channels[3], c, cache)?;
            }
            7 => {
                x = norm_silu(b, &x, &e.conv_norm_out, c)?;
                x = conv(b, &x, &e.conv_out, c.latent_channels * 2, [3; 3], [1; 3], 1, [0; 2], cache)?;
                let mean = b.slice_token_rows(&x.tensor, 0, c.latent_channels)?;
                x.tensor = b.scale_tensor(&transpose(b, &mean)?, c.scaling_factor)?;
            }
            _ => return Err(error(format!("VAE encoder stage={stage}不在0..8"))),
        }
        Ok(x)
    });
    if result.is_ok() {
        cache.next_chunk();
    }
    result
}

/// decoder末端按resnet分卡，避免全分辨率的六层tail集中在同一GPU。
/// 段0接已scale的THW×latent；其余段及最终RGB均为C×THW。
pub(crate) fn decode_stage<B: VaeBackend + SegmentedTensorBackend>(b: &B, stage: usize, input: VaeVideo<B::Tensor>, weights: &PreparedVae<B::Weight>, cache: &mut VaeStageCache<B::Tensor>) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let result = batch(b, || {
        let c = &weights.config;
        let d = &weights.decoder;
        let mut x = input;
        match stage {
            0..=1 => {
                if stage == 0 {
                    if b.token_rows(&x.tensor) != product(x.shape)? || b.token_cols(&x.tensor) != c.latent_channels {
                        return Err(error("VAE decoder stage0 latent矩阵shape不匹配"));
                    }
                    x.tensor = transpose(b, &b.scale_tensor(&x.tensor, c.scaling_factor.recip())?)?;
                    x = conv(b, &x, &d.conv_in, c.block_channels[3], [3; 3], [1; 3], 1, [0; 2], cache)?;
                    x = mid(b, x, &d.mid, c, cache)?;
                }
                let i = stage;
                let channels = c.block_channels[3 - i];
                let start = if stage == 1 { 2 } else { 0 };
                for w in &d.up_blocks[i].resnets[start..] {
                    x = resnet(b, x, w, channels, c, cache)?;
                }
                let up = d.up_blocks[i].upsample.as_ref().ok_or_else(|| error(format!("VAE decoder stage{stage}缺少upsample")))?;
                x = upscale(b, x, up, cache.first)?;
                x = conv(b, &x, &up.conv, channels, [3; 3], [1; 3], 1, [0; 2], cache)?;
                if stage == 0 {
                    // 此时已处于下一组的分辨率；前移两个残差块均衡八卡计算量。
                    for w in &d.up_blocks[1].resnets[..2] {
                        x = resnet(b, x, w, c.block_channels[2], c, cache)?;
                    }
                }
            }
            2 => {
                for w in &d.up_blocks[2].resnets[..2] {
                    x = resnet(b, x, w, c.block_channels[1], c, cache)?;
                }
            }
            3 => {
                x = resnet(b, x, &d.up_blocks[2].resnets[2], c.block_channels[1], c, cache)?;
                let up = d.up_blocks[2].upsample.as_ref().ok_or_else(|| error("VAE decoder stage3缺少upsample"))?;
                x = upscale(b, x, up, cache.first)?;
                x = conv(b, &x, &up.conv, c.block_channels[1], [3; 3], [1; 3], 1, [0; 2], cache)?;
            }
            4..=6 => {
                x = resnet(b, x, &d.up_blocks[3].resnets[stage - 4], c.block_channels[0], c, cache)?;
            }
            7 => {
                x = norm_silu(b, &x, &d.conv_norm_out, c)?;
                x = conv(b, &x, &d.conv_out, c.out_channels, [3; 3], [1; 3], 1, [0; 2], cache)?;
            }
            _ => return Err(error(format!("VAE decoder stage={stage}不在0..8"))),
        }
        Ok(x)
    });
    if result.is_ok() {
        cache.next_chunk();
    }
    result
}

fn visit_slices<B: VaeBackend + SegmentedTensorBackend>(
    b: &B,
    x: VaeVideo<B::Tensor>,
    w: &PreparedVae<B::Weight>,
    split: usize,
    encode: bool,
    mut emit: impl FnMut(usize, VaeVideo<B::Tensor>) -> Result<(), BackendError>,
) -> Result<[usize; 3], BackendError> {
    if split == 0 {
        return Err(error("SeedVR2 VAE时间分片不能为0"));
    }
    let tokens = transpose(b, &x.tensor)?;
    let plane = x.shape[1] * x.shape[2];
    let mut offset = 0;
    let mut shape = [0; 3];
    let mut cache = VaeStageCache::new();
    while offset < x.shape[0] {
        let count = (split + usize::from(cache.first)).min(x.shape[0] - offset);
        let part = b.slice_token_rows(&tokens, offset * plane, count * plane)?;
        let part = VaeVideo { tensor: transpose(b, &part)?, shape: [count, x.shape[1], x.shape[2]] };
        let out = if encode { encode_chunk(b, part, w, &mut cache)? } else { decode_chunk(b, part, w, &mut cache)? };
        if shape[0] != 0 && shape[1..] != out.shape[1..] {
            return Err(error("SeedVR2 VAE分片输出空间shape改变"));
        }
        let output_start = shape[0];
        shape = [shape[0] + out.shape[0], out.shape[1], out.shape[2]];
        emit(output_start, out)?;
        offset += count;
        cache.next_chunk();
    }
    // 返回前释放全部逐层尾cache；下一批/另一阶段不共享状态。
    drop(cache);
    Ok(shape)
}

fn sliced<B: VaeBackend + SegmentedTensorBackend>(b: &B, x: VaeVideo<B::Tensor>, w: &PreparedVae<B::Weight>, split: usize, encode: bool) -> Result<VaeVideo<B::Tensor>, BackendError> {
    let mut outputs = Vec::new();
    let shape = visit_slices(b, x, w, split, encode, |_, out| {
        outputs.push(transpose(b, &out.tensor)?);
        Ok(())
    })?;
    let refs = outputs.iter().collect::<Vec<_>>();
    Ok(VaeVideo { tensor: transpose(b, &b.concat_token_rows(&refs)?)?, shape })
}

fn validate_encode<B: VaeBackend>(b: &B, input: &VaeVideo<B::Tensor>, c: &SeedVr2VaeConfig) -> Result<(), BackendError> {
    c.validate().map_err(error)?;
    validate_video(b, input, c.in_channels)?;
    if !(input.shape[0] - 1).is_multiple_of(c.temporal_downsample_factor) || input.shape[1..].iter().any(|v| !v.is_multiple_of(c.spatial_downsample_factor)) {
        return Err(error(format!("SeedVR2 VAE输入{:?}不满足T=4n+1/空间8倍数", input.shape)));
    }
    if c.shifting_factor != 0.0 {
        return Err(error("当前SeedVR2 VAE原生路径要求已核实的shift=0"));
    }
    Ok(())
}

/// 回调逐片接收已scale的THW×C latent；start是latent时间坐标，shape是本片时空尺寸。
/// 回调返回错误会立即结束，所有逐层cache随本调用释放。
pub fn encode_stream<B: VaeBackend + SegmentedTensorBackend>(
    b: &B,
    input: VaeVideo<B::Tensor>,
    weights: &PreparedVae<B::Weight>,
    mut emit: impl FnMut(usize, VaeVideo<B::Tensor>) -> Result<(), BackendError>,
) -> Result<[usize; 3], BackendError> {
    batch(b, || {
        validate_encode(b, &input, &weights.config)?;
        visit_slices(b, input, weights, weights.config.temporal_downsample_factor, true, |start, out| {
            let tokens = transpose(b, &out.tensor)?;
            emit(start, VaeVideo { tensor: b.scale_tensor(&tokens, weights.config.scaling_factor)?, shape: out.shape })
        })
    })
}

/// 回调逐片接收C×THW RGB；start是输出帧坐标，初片5帧、后续每片4帧。
pub fn decode_stream<B: VaeBackend + SegmentedTensorBackend>(
    b: &B,
    latent: VaeVideo<B::Tensor>,
    weights: &PreparedVae<B::Weight>,
    emit: impl FnMut(usize, VaeVideo<B::Tensor>) -> Result<(), BackendError>,
) -> Result<[usize; 3], BackendError> {
    batch(b, || {
        let c = &weights.config;
        c.validate().map_err(error)?;
        if b.token_rows(&latent.tensor) != product(latent.shape)? || b.token_cols(&latent.tensor) != c.latent_channels || c.shifting_factor != 0.0 {
            return Err(error("SeedVR2 VAE latent shape/shift不符合checkpoint契约"));
        }
        let tensor = b.scale_tensor(&latent.tensor, c.scaling_factor.recip())?;
        let input = VaeVideo { tensor: transpose(b, &tensor)?, shape: latent.shape };
        visit_slices(b, input, weights, 1, false, emit)
    })
}

/// RGB 输入已归一化到[-1,1]；输出是供DiT使用的THW×16 latent。
pub fn encode<B: VaeBackend + SegmentedTensorBackend>(b: &B, input: VaeVideo<B::Tensor>, weights: &PreparedVae<B::Weight>) -> Result<VaeVideo<B::Tensor>, BackendError> {
    batch(b, || {
        let c = &weights.config;
        validate_encode(b, &input, c)?;
        let output = sliced(b, input, weights, c.temporal_downsample_factor, true)?;
        let tokens = transpose(b, &output.tensor)?;
        Ok(VaeVideo { tensor: b.scale_tensor(&tokens, c.scaling_factor)?, shape: output.shape })
    })
}

/// 输入THW×16 latent；输出C×THW RGB，保留原模型[-1,1]值域。
pub fn decode<B: VaeBackend + SegmentedTensorBackend>(b: &B, latent: VaeVideo<B::Tensor>, weights: &PreparedVae<B::Weight>) -> Result<VaeVideo<B::Tensor>, BackendError> {
    batch(b, || {
        let c = &weights.config;
        c.validate().map_err(error)?;
        if b.token_rows(&latent.tensor) != product(latent.shape)? || b.token_cols(&latent.tensor) != c.latent_channels || !c.scaling_factor.is_finite() || c.scaling_factor <= 0.0 || c.shifting_factor != 0.0 {
            return Err(error("SeedVR2 VAE latent shape/scale/shift不符合checkpoint契约"));
        }
        let tensor = b.scale_tensor(&latent.tensor, c.scaling_factor.recip())?;
        let input = VaeVideo { tensor: transpose(b, &tensor)?, shape: latent.shape };
        sliced(b, input, weights, 1, false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::cpu::CpuContext,
        weight::model::seedvr2::{SeedVr2VaeUpBlock, SeedVr2VaeUpsample},
    };

    fn tensor(name: &str, shape: &[usize], values: Vec<f32>) -> TensorData {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        TensorData { name: name.to_owned(), dtype: "F32".to_owned(), shape: shape.to_vec(), data: values.into_iter().flat_map(f32::to_le_bytes).collect() }
    }

    fn weights(input: usize, output: usize, kernel: [usize; 3]) -> SeedVr2VaeConv {
        let n = input * output * kernel.into_iter().product::<usize>();
        SeedVr2VaeConv {
            weight: tensor("test.conv.weight", &[output, input, kernel[0], kernel[1], kernel[2]], (0..n).map(|i| (i as f32 * 0.31).sin() * 0.05).collect()),
            bias: tensor("test.conv.bias", &[output], (0..output).map(|i| i as f32 * 0.07).collect()),
        }
    }

    fn norm(channels: usize) -> SeedVr2VaeNorm {
        SeedVr2VaeNorm { weight: tensor("test.norm.weight", &[channels], vec![1.0; channels]), bias: tensor("test.norm.bias", &[channels], vec![0.0; channels]) }
    }

    fn block(input: usize, output: usize) -> SeedVr2VaeResnet {
        SeedVr2VaeResnet { norm1: norm(input), conv1: weights(input, output, [3; 3]), norm2: norm(output), conv2: weights(output, output, [3; 3]), shortcut: (input != output).then(|| weights(input, output, [1; 3])) }
    }

    fn middle(channels: usize) -> SeedVr2VaeMid {
        let linear = || {
            let w = weights(channels, channels, [1; 3]);
            SeedVr2Linear { weight: tensor("test.linear.weight", &[channels, channels], w.weight.to_f32().unwrap()), bias: w.bias }
        };
        SeedVr2VaeMid { resnets: [block(channels, channels), block(channels, channels)], attention: SeedVr2VaeAttention { group_norm: norm(channels), query: linear(), key: linear(), value: linear(), output: linear() } }
    }

    fn small_model() -> PreparedVae<crate::backend::cpu::CpuWeight> {
        let mut c = SeedVr2VaeConfig::standard();
        c.in_channels = 1;
        c.out_channels = 1;
        c.latent_channels = 1;
        c.block_channels = [2, 2, 2, 2];
        c.norm_groups = 1;
        let e = SeedVr2VaeEncoder {
            conv_in: weights(1, 2, [3; 3]),
            down_blocks: (0..4).map(|i| SeedVr2VaeDownBlock { resnets: vec![block(2, 2), block(2, 2)], downsample: (i < 3).then(|| weights(2, 2, [if i == 0 { 1 } else { 3 }, 3, 3])) }).collect(),
            mid: middle(2),
            conv_norm_out: norm(2),
            conv_out: weights(2, 2, [3; 3]),
        };
        let d = SeedVr2VaeDecoder {
            conv_in: weights(1, 2, [3; 3]),
            mid: middle(2),
            up_blocks: (0..4)
                .map(|i| SeedVr2VaeUpBlock {
                    resnets: vec![block(2, 2), block(2, 2), block(2, 2)],
                    upsample: (i < 3).then(|| SeedVr2VaeUpsample { upscale_conv: weights(2, 2 * 4 * if i < 2 { 2 } else { 1 }, [1; 3]), conv: weights(2, 2, [3; 3]) }),
                })
                .collect(),
            conv_norm_out: norm(2),
            conv_out: weights(2, 1, [3; 3]),
        };
        prepare_vae_weights(&CpuContext, c, &e, &d).unwrap()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(x.is_finite() && y.is_finite() && (x - y).abs() <= 2.0e-5 + y.abs() * 2.0e-5, "index={i} actual={x} expected={y}");
        }
    }

    #[test]
    fn matrix_transpose_has_correct_non_square_metadata_and_inverse() {
        let b = CpuContext;
        let x = b.vae_tensor_from_f32((0..15).map(|i| i as f32).collect(), 3, 5).unwrap();
        let y = transpose(&b, &x).unwrap();
        assert_eq!((y.rows, y.cols), (5, 3));
        let z = transpose(&b, &y).unwrap();
        assert_eq!((z.rows, z.cols), (3, 5));
        assert_eq!(z.data, x.data);
        assert!(b.channels_to_time(&x, 2).is_err());
    }

    #[test]
    fn causal_conv_chunk_boundary_matches_whole_and_replicates_first_frame() {
        let b = CpuContext;
        let w = prepare_conv(&b, &SeedVr2VaeConv { weight: tensor("causal", &[1, 1, 3, 1, 1], vec![1.0, 2.0, 3.0]), bias: tensor("bias", &[1], vec![0.5]) }).unwrap();
        for stride in [1, 2] {
            let values = [2., 4., 7., 8., 11., 13., 17., 19., 23.];
            let x = VaeVideo { tensor: b.vae_tensor_from_f32(values.to_vec(), 1, 9).unwrap(), shape: [9, 1, 1] };
            let whole = conv(&b, &x, &w, 1, [3, 1, 1], [stride, 1, 1], 0, [0; 2], &mut VaeStageCache::new()).unwrap();
            assert_eq!(whole.tensor.data[0], 12.5);
            let mut cache = VaeStageCache::new();
            let mut pieces = Vec::new();
            for (start, length) in [(0, 5), (5, 4)] {
                let x = VaeVideo { tensor: b.vae_tensor_from_f32(values[start..start + length].to_vec(), 1, length).unwrap(), shape: [length, 1, 1] };
                let part = conv(&b, &x, &w, 1, [3, 1, 1], [stride, 1, 1], 0, [0; 2], &mut cache).unwrap();
                pieces.extend_from_slice(&part.tensor.data);
                cache.next_chunk();
            }
            assert_eq!(pieces, whole.tensor.data);
            assert_eq!(cache.tails[0].as_ref().unwrap().data.len(), 3 - stride);
        }
    }

    #[test]
    fn learned_shuffle_preserves_xyzc_order_and_removes_second_frame_only() {
        let b = CpuContext;
        let channels = 2;
        let shape = [3, 2, 3];
        let input = (0..channels * 18).map(|i| i as f32 * 0.1).collect::<Vec<_>>();
        for temporal in [1, 2] {
            let source = weights(channels, channels * 4 * temporal, [1; 3]);
            let raw_w = source.weight.to_f32().unwrap();
            let raw_b = source.bias.to_f32().unwrap();
            let up = PreparedUpsample { projections: prepare_upscale(&b, &source, channels, temporal).unwrap(), conv: prepare_conv(&b, &weights(channels, channels, [3; 3])).unwrap() };
            for first in [false, true] {
                let x = VaeVideo { tensor: b.vae_tensor_from_f32(input.clone(), channels, 18).unwrap(), shape };
                let y = upscale(&b, x, &up, first).unwrap();
                let depth = 3 * temporal - usize::from(first && temporal == 2);
                let mut expected = vec![0.0; channels * depth * 4 * 6];
                for c in 0..channels {
                    for t in 0..3 {
                        for z in 0..temporal {
                            let original = t * temporal + z;
                            if first && temporal == 2 && original == 1 {
                                continue;
                            }
                            let ot = original - usize::from(first && temporal == 2 && original > 1);
                            for h in 0..2 {
                                for x in 0..2 {
                                    for w in 0..3 {
                                        for yy in 0..2 {
                                            let row = ((x * 2 + yy) * temporal + z) * channels + c;
                                            let mut value = raw_b[row];
                                            for ic in 0..channels {
                                                value += raw_w[row * channels + ic] * input[ic * 18 + t * 6 + h * 3 + w];
                                            }
                                            expected[((c * depth + ot) * 4 + h * 2 + x) * 6 + w * 2 + yy] = value;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                assert_eq!(y.shape, [depth, 4, 6]);
                close(&y.tensor.data, &expected);
            }
        }
    }

    #[test]
    fn full_encoder_decoder_slicing_matches_whole_sequence() {
        let b = CpuContext;
        let model = small_model();
        let input = (0..9 * 8 * 8).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        let video = || VaeVideo { tensor: b.vae_tensor_from_f32(input.clone(), 1, 9 * 8 * 8).unwrap(), shape: [9, 8, 8] };
        let split = sliced(&b, video(), &model, 4, true).unwrap();
        let whole = sliced(&b, video(), &model, 9, true).unwrap();
        assert_eq!(split.shape, [3, 1, 1]);
        close(&split.tensor.data, &whole.tensor.data);
        let normalized = encode(&b, video(), &model).unwrap();
        assert_eq!((normalized.tensor.rows, normalized.tensor.cols), (3, 1));
        close(&normalized.tensor.data, &split.tensor.data.iter().map(|x| x * model.config.scaling_factor).collect::<Vec<_>>());
        let latent = || VaeVideo { tensor: b.vae_tensor_from_f32(split.tensor.data.clone(), 1, 3).unwrap(), shape: [3, 1, 1] };
        let decoded_split = sliced(&b, latent(), &model, 1, false).unwrap();
        let decoded_whole = sliced(&b, latent(), &model, 3, false).unwrap();
        assert_eq!(decoded_split.shape, [9, 8, 8]);
        close(&decoded_split.tensor.data, &decoded_whole.tensor.data);
        let decoded = decode(&b, normalized, &model).unwrap();
        assert_eq!(decoded.shape, [9, 8, 8]);
        close(&decoded.tensor.data, &decoded_split.tensor.data);
    }

    #[test]
    fn streaming_offsets_single_frame_and_callback_error() {
        let b = CpuContext;
        let model = small_model();
        for frames in [1, 5, 9] {
            let values = (0..frames * 64).map(|i| (i as f32 * 0.021).cos()).collect::<Vec<_>>();
            let video = || VaeVideo { tensor: b.vae_tensor_from_f32(values.clone(), 1, frames * 64).unwrap(), shape: [frames, 8, 8] };
            let expected = encode(&b, video(), &model).unwrap();
            let mut encoded = Vec::new();
            let mut encode_offsets = Vec::new();
            let shape = encode_stream(&b, video(), &model, |offset, chunk| {
                encode_offsets.push((offset, chunk.shape[0]));
                encoded.extend(chunk.tensor.data);
                Ok(())
            })
            .unwrap();
            assert_eq!(shape, expected.shape);
            close(&encoded, &expected.tensor.data);
            assert_eq!(
                encode_offsets,
                match frames {
                    1 => vec![(0, 1)],
                    5 => vec![(0, 2)],
                    _ => vec![(0, 2), (2, 1)],
                }
            );
            let latent = || VaeVideo { tensor: b.vae_tensor_from_f32(encoded.clone(), shape[0], 1).unwrap(), shape };
            let expected = decode(&b, latent(), &model).unwrap();
            let mut decoded = Vec::new();
            let mut decode_offsets = Vec::new();
            let output_shape = decode_stream(&b, latent(), &model, |offset, chunk| {
                decode_offsets.push((offset, chunk.shape[0]));
                decoded.extend(chunk.tensor.data);
                Ok(())
            })
            .unwrap();
            assert_eq!(output_shape, [frames, 8, 8]);
            close(&decoded, &expected.tensor.data);
            assert_eq!(
                decode_offsets,
                match frames {
                    1 => vec![(0, 1)],
                    5 => vec![(0, 5)],
                    _ => vec![(0, 5), (5, 4)],
                }
            );
            let mut calls = 0;
            let failure = decode_stream(&b, latent(), &model, |_, _| {
                calls += 1;
                Err(error("callback oracle"))
            })
            .unwrap_err();
            assert_eq!(calls, 1);
            assert!(failure.to_string().contains("callback oracle"));
            // 错误后的新调用从首片开始，不能继承上一次中断的尾缓存。
            close(&decode(&b, latent(), &model).unwrap().tensor.data, &expected.tensor.data);
        }
    }

    #[test]
    fn eight_stages_keep_independent_causal_cache_and_match_serial_chunks() {
        let b = CpuContext;
        let model = small_model();
        for frames in [1usize, 5, 9] {
            let values = (0..frames * 64).map(|i| (i as f32 * 0.037).sin()).collect::<Vec<_>>();
            let expected = encode(&b, VaeVideo { tensor: b.vae_tensor_from_f32(values.clone(), 1, frames * 64).unwrap(), shape: [frames, 8, 8] }, &model).unwrap();
            let mut caches = (0..8).map(|_| VaeStageCache::new()).collect::<Vec<_>>();
            let mut encoded = Vec::new();
            let mut start = 0;
            while start < frames {
                let count = (if start == 0 { 5 } else { 4 }).min(frames - start);
                let mut x = VaeVideo { tensor: b.vae_tensor_from_f32(values[start * 64..(start + count) * 64].to_vec(), 1, count * 64).unwrap(), shape: [count, 8, 8] };
                for (stage, cache) in caches.iter_mut().enumerate() {
                    x = encode_stage(&b, stage, x, &model, cache).unwrap();
                }
                encoded.extend(x.tensor.data);
                start += count;
            }
            assert_eq!(encoded, expected.tensor.data);
            assert_eq!(caches.iter().map(|c| c.tails.len()).sum::<usize>(), 25);
            let shape = expected.shape;
            let expected = decode(&b, expected, &model).unwrap();
            let mut caches = (0..8).map(|_| VaeStageCache::new()).collect::<Vec<_>>();
            let mut decoded = Vec::new();
            let mut start = 0;
            while start < shape[0] {
                let count = (if start == 0 { 2 } else { 1 }).min(shape[0] - start);
                let mut x = VaeVideo { tensor: b.vae_tensor_from_f32(encoded[start..start + count].to_vec(), count, 1).unwrap(), shape: [count, 1, 1] };
                for (stage, cache) in caches.iter_mut().enumerate() {
                    x = decode_stage(&b, stage, x, &model, cache).unwrap();
                }
                decoded.extend(x.tensor.data);
                start += count;
            }
            assert_eq!(decoded, expected.tensor.data);
            assert_eq!(caches.iter().map(|c| c.tails.len()).sum::<usize>(), 33);
            let dummy = || VaeVideo { tensor: b.vae_tensor_from_f32(vec![0.0], 1, 1).unwrap(), shape: [1; 3] };
            assert!(encode_stage(&b, 8, dummy(), &model, &mut VaeStageCache::new()).is_err());
            assert!(decode_stage(&b, 8, dummy(), &model, &mut VaeStageCache::new()).is_err());
        }
    }

    #[test]
    fn microframe_pipeline_downsample_pairs_and_upsample_frames_match_whole() {
        use crate::kernel::cpu::CpuTensor;
        use std::collections::VecDeque;
        let b = CpuContext;
        let model = small_model();
        let split = |x: VaeVideo<CpuTensor>, tokens: bool| {
            let plane = x.shape[1] * x.shape[2];
            let values = if tokens { x.tensor } else { transpose(&b, &x.tensor).unwrap() };
            (0..x.shape[0])
                .map(|t| {
                    let frame = b.slice_token_rows(&values, t * plane, plane).unwrap();
                    VaeVideo { tensor: if tokens { frame } else { transpose(&b, &frame).unwrap() }, shape: [1, x.shape[1], x.shape[2]] }
                })
                .collect::<VecDeque<_>>()
        };
        let join = |frames: Vec<VaeVideo<CpuTensor>>, tokens: bool| {
            let shape = [frames.len(), frames[0].shape[1], frames[0].shape[2]];
            let values = frames.into_iter().map(|x| if tokens { x.tensor } else { transpose(&b, &x.tensor).unwrap() }).collect::<Vec<_>>();
            let refs = values.iter().collect::<Vec<_>>();
            let tensor = b.concat_token_rows(&refs).unwrap();
            VaeVideo { tensor: if tokens { tensor } else { transpose(&b, &tensor).unwrap() }, shape }
        };
        for count in [1usize, 5, 9, 17] {
            let input = (0..count * 64).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            let make_input = || VaeVideo { tensor: b.vae_tensor_from_f32(input.clone(), 1, count * 64).unwrap(), shape: [count, 8, 8] };
            let expected = encode(&b, make_input(), &model).unwrap();
            let mut frames = split(make_input(), false);
            for stage in 0..8 {
                let mut cache = VaeStageCache::new();
                let mut outputs = VecDeque::new();
                while !frames.is_empty() {
                    let needed = if matches!(stage, 2 | 3) && !cache.first { 2 } else { 1 };
                    assert!(frames.len() >= needed, "stage{stage}有未配对尾帧");
                    let part = join(frames.drain(..needed).collect(), false);
                    let out = encode_stage(&b, stage, part, &model, &mut cache).unwrap();
                    outputs.extend(split(out, stage == 7));
                }
                frames = outputs;
            }
            let actual = join(frames.into_iter().collect(), true);
            assert_eq!(actual.shape, expected.shape);
            close(&actual.tensor.data, &expected.tensor.data);
            let mut frames = split(actual, true);
            for stage in 0..8 {
                let mut cache = VaeStageCache::new();
                let mut outputs = VecDeque::new();
                while let Some(frame) = frames.pop_front() {
                    let out = decode_stage(&b, stage, frame, &model, &mut cache).unwrap();
                    outputs.extend(split(out, false));
                }
                frames = outputs;
            }
            let actual = join(frames.into_iter().collect(), false);
            let expected = decode(&b, expected, &model).unwrap();
            assert_eq!(actual.shape, expected.shape);
            close(&actual.tensor.data, &expected.tensor.data);
        }
    }
}
