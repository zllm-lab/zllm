//! FLUX.2 Klein 4B 双流/单流 DiT 与 Flow Matching 采样。

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm;

use std::ops::Range;

use crate::{
    attention::rope::RotaryLayout,
    backend::{Backend, BackendError, DiffusionBackend, LinearWeight, VaeBackend},
    diffusion::ModulationSegment,
    model_spec::flux2_klein::{Flux2KleinConfig, Flux2KleinVaeConfig},
    moe::Activation,
    vae::{Conv3dSpec, PixelShuffleSpec},
    weight::{
        container::safetensor::TensorData,
        model::flux2_klein::{Flux2AttentionWeights, Flux2DoubleBlockWeights, Flux2GlobalWeights, Flux2KleinSource, Flux2MlpWeights, Flux2SingleBlockWeights, Flux2VaeAttention, Flux2VaeConv, Flux2VaeResnet, Flux2VaeSource},
    },
};

pub struct Flux2PreparedGlobal<W> {
    pub weights: Flux2GlobalWeights<W>,
    pub unit_norm_weight: W,
    pub unit_norm_bias: W,
}

pub struct Flux2PositionTables {
    pub cosine: Vec<f32>,
    pub sine: Vec<f32>,
    pub text_rows: usize,
    pub image_rows: usize,
    pub output_image_rows: usize,
}

struct DoubleModulation<T> {
    attention_shift: T,
    attention_scale: T,
    attention_gate: T,
    mlp_shift: T,
    mlp_scale: T,
    mlp_gate: T,
}

struct SingleModulation<T> {
    shift: T,
    scale: T,
    gate: T,
}

fn prepare_tensor<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let (rows, cols) = match tensor.shape.as_slice() {
        [cols] => (1, *cols),
        [rows, cols] => (*rows, *cols),
        shape => return Err(BackendError::Compute { msg: format!("FLUX.2 Klein 权重 {} shape={shape:?} 不是向量或矩阵", tensor.name) }),
    };
    match tensor.dtype.as_str() {
        "BF16" => backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, cols),
        "F16" => {
            let values = tensor.data.chunks_exact(2).map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]])).collect::<Vec<_>>();
            backend.prepare_weight(LinearWeight::F16(&values), rows, cols)
        }
        "F32" => {
            let values = tensor.to_f32().map_err(|msg| BackendError::Compute { msg })?;
            backend.prepare_weight(LinearWeight::F32(&values), rows, cols)
        }
        dtype => Err(BackendError::Compute { msg: format!("FLUX.2 Klein 权重 {} dtype={dtype} 不受支持", tensor.name) }),
    }
}

fn prepare_attention<B: Backend>(backend: &B, source: &Flux2AttentionWeights) -> Result<Flux2AttentionWeights<B::Weight>, BackendError> {
    Ok(Flux2AttentionWeights {
        query: prepare_tensor(backend, &source.query)?,
        key: prepare_tensor(backend, &source.key)?,
        value: prepare_tensor(backend, &source.value)?,
        query_norm: prepare_tensor(backend, &source.query_norm)?,
        key_norm: prepare_tensor(backend, &source.key_norm)?,
        output: prepare_tensor(backend, &source.output)?,
    })
}

fn prepare_mlp<B: Backend>(backend: &B, source: &Flux2MlpWeights) -> Result<Flux2MlpWeights<B::Weight>, BackendError> {
    Ok(Flux2MlpWeights { input: prepare_tensor(backend, &source.input)?, output: prepare_tensor(backend, &source.output)? })
}

fn prepare_double_block<B: Backend>(backend: &B, source: &Flux2DoubleBlockWeights) -> Result<Flux2DoubleBlockWeights<B::Weight>, BackendError> {
    Ok(Flux2DoubleBlockWeights {
        image_attention: prepare_attention(backend, &source.image_attention)?,
        image_mlp: prepare_mlp(backend, &source.image_mlp)?,
        text_attention: prepare_attention(backend, &source.text_attention)?,
        text_mlp: prepare_mlp(backend, &source.text_mlp)?,
    })
}

fn prepare_single_block<B: Backend>(backend: &B, source: &Flux2SingleBlockWeights) -> Result<Flux2SingleBlockWeights<B::Weight>, BackendError> {
    Ok(Flux2SingleBlockWeights {
        input: prepare_tensor(backend, &source.input)?,
        query_norm: prepare_tensor(backend, &source.query_norm)?,
        key_norm: prepare_tensor(backend, &source.key_norm)?,
        output: prepare_tensor(backend, &source.output)?,
    })
}

pub fn prepare_global<B: Backend>(backend: &B, source: &Flux2GlobalWeights) -> Result<Flux2PreparedGlobal<B::Weight>, BackendError> {
    let hidden = source.image_input.shape.first().copied().ok_or_else(|| BackendError::Compute { msg: "FLUX.2 Klein image input shape 为空".to_owned() })?;
    Ok(Flux2PreparedGlobal {
        weights: Flux2GlobalWeights {
            image_input: prepare_tensor(backend, &source.image_input)?,
            text_input: prepare_tensor(backend, &source.text_input)?,
            time_input: prepare_tensor(backend, &source.time_input)?,
            time_output: prepare_tensor(backend, &source.time_output)?,
            image_modulation: prepare_tensor(backend, &source.image_modulation)?,
            text_modulation: prepare_tensor(backend, &source.text_modulation)?,
            single_modulation: prepare_tensor(backend, &source.single_modulation)?,
            final_modulation: prepare_tensor(backend, &source.final_modulation)?,
            output: prepare_tensor(backend, &source.output)?,
        },
        unit_norm_weight: backend.prepare_f32(&vec![1.0; hidden], 1, hidden)?,
        unit_norm_bias: backend.prepare_f32(&vec![0.0; hidden], 1, hidden)?,
    })
}

/// 构造官方 `[time, height, width, text-index]` 四轴位置。
pub fn position_ids(text_rows: usize, latent_height: usize, latent_width: usize) -> Result<(Vec<[f32; 4]>, Vec<[f32; 4]>), String> {
    if text_rows == 0 || latent_height == 0 || latent_width == 0 {
        return Err("FLUX.2 Klein text/latent 网格必须非零".to_owned());
    }
    let text = (0..text_rows).map(|index| [0.0, 0.0, 0.0, index as f32]).collect();
    let image = (0..latent_height).flat_map(|height| (0..latent_width).map(move |width| [0.0, height as f32, width as f32, 0.0])).collect();
    Ok((text, image))
}

/// 将四轴位置展开成相邻偶奇配对的 RoPE 表；行顺序固定为 `[text, image]`。
pub fn position_tables(config: &Flux2KleinConfig, text: &[[f32; 4]], image: &[[f32; 4]]) -> Result<Flux2PositionTables, String> {
    position_tables_conditioned(config, text, image, image.len())
}

/// reference image rows 追加在 output rows 后；最终投影只保留 output 前缀。
pub fn position_tables_conditioned(config: &Flux2KleinConfig, text: &[[f32; 4]], image: &[[f32; 4]], output_image_rows: usize) -> Result<Flux2PositionTables, String> {
    config.validate()?;
    if text.is_empty() || image.is_empty() || output_image_rows == 0 || output_image_rows > image.len() || text.iter().chain(image).flatten().any(|value| !value.is_finite()) {
        return Err("FLUX.2 Klein position ids 必须非空且为有限数".to_owned());
    }
    let half = config.head_dim / 2;
    let rows = text.len().checked_add(image.len()).ok_or("FLUX.2 Klein position rows 溢出")?;
    let mut cosine = Vec::with_capacity(rows * half);
    let mut sine = Vec::with_capacity(rows * half);
    for position in text.iter().chain(image) {
        for (axis, &dim) in config.axes_dim.iter().enumerate() {
            for pair in 0..dim / 2 {
                let frequency = config.rope_theta.powf(-2.0 * pair as f32 / dim as f32);
                let angle = position[axis] * frequency;
                cosine.push(angle.cos());
                sine.push(angle.sin());
            }
        }
    }
    Ok(Flux2PositionTables { cosine, sine, text_rows: text.len(), image_rows: image.len(), output_image_rows })
}

fn chunks<B: DiffusionBackend>(backend: &B, projected: &B::Tensor, count: usize, hidden: usize) -> Result<Vec<B::Tensor>, BackendError> {
    backend.modulation_chunks(projected, 1, count, hidden)
}

fn double_modulation<B: DiffusionBackend>(backend: &B, activated: &B::Tensor, weight: &B::Weight, hidden: usize) -> Result<DoubleModulation<B::Tensor>, BackendError> {
    let values: [B::Tensor; 6] =
        chunks(backend, &backend.linear(activated, weight)?, 6, hidden)?.try_into().map_err(|values: Vec<_>| BackendError::Compute { msg: format!("FLUX.2 Klein double modulation chunks={}，期望 6", values.len()) })?;
    let [attention_shift, attention_scale, attention_gate, mlp_shift, mlp_scale, mlp_gate] = values;
    Ok(DoubleModulation { attention_shift, attention_scale, attention_gate, mlp_shift, mlp_scale, mlp_gate })
}

fn single_modulation<B: DiffusionBackend>(backend: &B, activated: &B::Tensor, weight: &B::Weight, hidden: usize) -> Result<SingleModulation<B::Tensor>, BackendError> {
    let values: [B::Tensor; 3] =
        chunks(backend, &backend.linear(activated, weight)?, 3, hidden)?.try_into().map_err(|values: Vec<_>| BackendError::Compute { msg: format!("FLUX.2 Klein single modulation chunks={}，期望 3", values.len()) })?;
    let [shift, scale, gate] = values;
    Ok(SingleModulation { shift, scale, gate })
}

fn unit_layer_norm<B: Backend>(backend: &B, global: &Flux2PreparedGlobal<B::Weight>, input: &B::Tensor, eps: f32) -> Result<B::Tensor, BackendError> {
    backend.layernorm_bias(input, &global.unit_norm_weight, &global.unit_norm_bias, eps)
}

fn gated_residual<B: DiffusionBackend>(backend: &B, residual: &B::Tensor, update: &B::Tensor, gate: &B::Tensor) -> Result<B::Tensor, BackendError> {
    backend.gated_residual_segmented(residual, update, gate, &[ModulationSegment { rows: 0..backend.token_rows(residual), modulation_row: 0 }])
}

fn mlp<B: DiffusionBackend>(backend: &B, config: &Flux2KleinConfig, input: &B::Tensor, weights: &Flux2MlpWeights<B::Weight>) -> Result<B::Tensor, BackendError> {
    let packed = backend.linear(input, &weights.input)?;
    let activated = backend.split_gated_activation(packed, config.mlp_hidden_size, &Activation::Silu)?;
    backend.linear(&activated, &weights.output)
}

fn qkv<B: DiffusionBackend>(backend: &B, config: &Flux2KleinConfig, input: &B::Tensor, weights: &Flux2AttentionWeights<B::Weight>) -> Result<(B::Tensor, B::Tensor, B::Tensor), BackendError> {
    let (query, key, value) = backend.triple_linear(input, &weights.query, &weights.key, &weights.value)?;
    let query = backend.rmsnorm_heads(&query, &weights.query_norm, config.num_heads, config.head_dim, config.norm_eps)?;
    let key = backend.rmsnorm_heads(&key, &weights.key_norm, config.num_heads, config.head_dim, config.norm_eps)?;
    Ok((query, key, value))
}

fn select_range<B: Backend>(backend: &B, input: &B::Tensor, rows: Range<usize>) -> Result<B::Tensor, BackendError> {
    let indices = rows.map(|row| u32::try_from(row).map_err(|_| BackendError::Compute { msg: format!("FLUX.2 Klein row={row} 超出 u32") })).collect::<Result<Vec<_>, _>>()?;
    backend.select_rows(input, &indices)
}

#[allow(clippy::too_many_arguments)]
fn double_block<B: DiffusionBackend>(
    backend: &B,
    config: &Flux2KleinConfig,
    global: &Flux2PreparedGlobal<B::Weight>,
    image: &B::Tensor,
    text: &B::Tensor,
    image_mod: &DoubleModulation<B::Tensor>,
    text_mod: &DoubleModulation<B::Tensor>,
    weights: &Flux2DoubleBlockWeights<B::Weight>,
    positions: &Flux2PositionTables,
) -> Result<(B::Tensor, B::Tensor), BackendError> {
    let image_norm = unit_layer_norm(backend, global, image, config.norm_eps)?;
    let image_norm = backend.adaln_modulate(&image_norm, &image_mod.attention_shift, &image_mod.attention_scale)?;
    let text_norm = unit_layer_norm(backend, global, text, config.norm_eps)?;
    let text_norm = backend.adaln_modulate(&text_norm, &text_mod.attention_shift, &text_mod.attention_scale)?;
    let (image_q, image_k, image_v) = qkv(backend, config, &image_norm, &weights.image_attention)?;
    let (text_q, text_k, text_v) = qkv(backend, config, &text_norm, &weights.text_attention)?;
    let query = backend.concat_rows(&text_q, &image_q)?;
    let key = backend.concat_rows(&text_k, &image_k)?;
    let value = backend.concat_rows(&text_v, &image_v)?;
    let (query, key) = backend.rope_pair_prefix(query, key, config.num_heads, config.head_dim, RotaryLayout::Interleaved, 0, &positions.cosine, &positions.sine)?;
    let attention = backend.full_attention(query, key, value, config.num_heads, config.head_dim, 1.0 / (config.head_dim as f32).sqrt())?;
    let text_attention = select_range(backend, &attention, 0..positions.text_rows)?;
    let image_attention = select_range(backend, &attention, positions.text_rows..positions.text_rows + positions.image_rows)?;

    let image_update = backend.linear(&image_attention, &weights.image_attention.output)?;
    let image = gated_residual(backend, image, &image_update, &image_mod.attention_gate)?;
    let image_norm = unit_layer_norm(backend, global, &image, config.norm_eps)?;
    let image_norm = backend.adaln_modulate(&image_norm, &image_mod.mlp_shift, &image_mod.mlp_scale)?;
    let image_update = mlp(backend, config, &image_norm, &weights.image_mlp)?;
    let image = gated_residual(backend, &image, &image_update, &image_mod.mlp_gate)?;

    let text_update = backend.linear(&text_attention, &weights.text_attention.output)?;
    let text = gated_residual(backend, text, &text_update, &text_mod.attention_gate)?;
    let text_norm = unit_layer_norm(backend, global, &text, config.norm_eps)?;
    let text_norm = backend.adaln_modulate(&text_norm, &text_mod.mlp_shift, &text_mod.mlp_scale)?;
    let text_update = mlp(backend, config, &text_norm, &weights.text_mlp)?;
    let text = gated_residual(backend, &text, &text_update, &text_mod.mlp_gate)?;
    Ok((image, text))
}

fn single_block<B: DiffusionBackend>(
    backend: &B,
    config: &Flux2KleinConfig,
    global: &Flux2PreparedGlobal<B::Weight>,
    hidden: &B::Tensor,
    modulation: &SingleModulation<B::Tensor>,
    weights: &Flux2SingleBlockWeights<B::Weight>,
    positions: &Flux2PositionTables,
) -> Result<B::Tensor, BackendError> {
    let normalized = unit_layer_norm(backend, global, hidden, config.norm_eps)?;
    let normalized = backend.adaln_modulate(&normalized, &modulation.shift, &modulation.scale)?;
    let packed = backend.linear(&normalized, &weights.input)?;
    let (qkv, mlp_packed) = backend.split_columns(&packed, config.hidden_size * 3)?;
    let (query, key_value) = backend.split_columns(&qkv, config.hidden_size)?;
    let (key, value) = backend.split_columns(&key_value, config.hidden_size)?;
    let query = backend.rmsnorm_heads(&query, &weights.query_norm, config.num_heads, config.head_dim, config.norm_eps)?;
    let key = backend.rmsnorm_heads(&key, &weights.key_norm, config.num_heads, config.head_dim, config.norm_eps)?;
    let (query, key) = backend.rope_pair_prefix(query, key, config.num_heads, config.head_dim, RotaryLayout::Interleaved, 0, &positions.cosine, &positions.sine)?;
    let attention = backend.full_attention(query, key, value, config.num_heads, config.head_dim, 1.0 / (config.head_dim as f32).sqrt())?;
    let activated = backend.split_gated_activation(mlp_packed, config.mlp_hidden_size, &Activation::Silu)?;
    let joined = backend.concat_columns(&attention, &activated)?;
    let update = backend.linear(&joined, &weights.output)?;
    gated_residual(backend, hidden, &update, &modulation.gate)
}

/// 单 batch Transformer forward；输入 image 为 `[latent_h*latent_w, 128]`，text 为 `[tokens, 7680]`。
pub fn forward<B: DiffusionBackend>(backend: &B, source: &Flux2KleinSource, global: &Flux2PreparedGlobal<B::Weight>, image: &B::Tensor, text: &B::Tensor, timestep: f32, positions: &Flux2PositionTables) -> Result<B::Tensor, BackendError> {
    let config = source.config();
    if backend.token_rows(image) != positions.image_rows || backend.token_cols(image) != config.input_channels || backend.token_rows(text) != positions.text_rows || backend.token_cols(text) != config.text_dim {
        return Err(BackendError::Compute {
            msg: format!(
                "FLUX.2 Klein input image=[{},{}] text=[{},{}]，期望 [{},{}] / [{},{}]",
                backend.token_rows(image),
                backend.token_cols(image),
                backend.token_rows(text),
                backend.token_cols(text),
                positions.image_rows,
                config.input_channels,
                positions.text_rows,
                config.text_dim
            ),
        });
    }
    if !timestep.is_finite() {
        return Err(BackendError::Compute { msg: format!("FLUX.2 Klein timestep={timestep} 非法") });
    }
    backend.begin_batch();
    let initial = (|| {
        let time = backend.timestep_embedding(&[timestep * 1_000.0], config.timestep_dim)?;
        let time = backend.linear(&time, &global.weights.time_input)?;
        let time = backend.silu(&time)?;
        let time = backend.linear(&time, &global.weights.time_output)?;
        let activated = backend.silu(&time)?;
        let image_mod = double_modulation(backend, &activated, &global.weights.image_modulation, config.hidden_size)?;
        let text_mod = double_modulation(backend, &activated, &global.weights.text_modulation, config.hidden_size)?;
        let single_mod = single_modulation(backend, &activated, &global.weights.single_modulation, config.hidden_size)?;
        let image = backend.linear(image, &global.weights.image_input)?;
        let text = backend.linear(text, &global.weights.text_input)?;
        Ok((time, image_mod, text_mod, single_mod, image, text))
    })();
    backend.finish_batch();
    let (time, image_mod, text_mod, single_mod, mut image, mut text) = initial?;

    for layer in 0..config.double_layers {
        let _scope = backend.layer_scope();
        let source_weights = source.load_double_block(layer).map_err(BackendError::ExpertLoad)?;
        let weights = prepare_double_block(backend, &source_weights)?;
        backend.begin_batch();
        let result = double_block(backend, config, global, &image, &text, &image_mod, &text_mod, &weights, positions);
        backend.finish_stream_chunk();
        (image, text) = result?;
    }
    let mut hidden = backend.concat_rows(&text, &image)?;
    for layer in 0..config.single_layers {
        let _scope = backend.layer_scope();
        let source_weights = source.load_single_block(layer).map_err(BackendError::ExpertLoad)?;
        let weights = prepare_single_block(backend, &source_weights)?;
        backend.begin_batch();
        let result = single_block(backend, config, global, &hidden, &single_mod, &weights, positions);
        backend.finish_stream_chunk();
        hidden = result?;
    }
    backend.begin_batch();
    let result = (|| {
        let image = select_range(backend, &hidden, positions.text_rows..positions.text_rows + positions.output_image_rows)?;
        let modulation: [B::Tensor; 2] = chunks(backend, &backend.linear(&backend.silu(&time)?, &global.weights.final_modulation)?, 2, config.hidden_size)?
            .try_into()
            .map_err(|values: Vec<_>| BackendError::Compute { msg: format!("FLUX.2 Klein final modulation chunks={}，期望 2", values.len()) })?;
        // 最终 AdaLayerNormContinuous 的权重顺序是 scale、shift，与 block modulation 不同。
        let [scale, shift] = modulation;
        let image = unit_layer_norm(backend, global, &image, config.norm_eps)?;
        let image = backend.adaln_modulate(&image, &shift, &scale)?;
        backend.linear(&image, &global.weights.output)
    })();
    backend.finish_batch();
    result
}

pub fn schedule(num_steps: usize, image_sequence_len: usize) -> Result<Vec<f32>, String> {
    if num_steps == 0 || image_sequence_len == 0 {
        return Err("FLUX.2 Klein steps 与 image sequence length 必须非零".to_owned());
    }
    let a1 = 8.738_095e-5_f32;
    let b1 = 1.898_333_3_f32;
    let a2 = 0.000_169_27_f32;
    let b2 = 0.456_666_65_f32;
    let length = image_sequence_len as f32;
    let mu = if image_sequence_len > 4_300 {
        a2 * length + b2
    } else {
        let m200 = a2 * length + b2;
        let m10 = a1 * length + b1;
        let slope = (m200 - m10) / 190.0;
        slope * num_steps as f32 + (m200 - 200.0 * slope)
    };
    let shift = mu.exp();
    Ok((0..=num_steps)
        .map(|index| {
            let timestep = 1.0 - index as f32 / num_steps as f32;
            shift * timestep / (1.0 - timestep + shift * timestep)
        })
        .collect())
}

pub fn denoise<B: DiffusionBackend>(
    backend: &B,
    source: &Flux2KleinSource,
    global: &Flux2PreparedGlobal<B::Weight>,
    text: &B::Tensor,
    mut image: B::Tensor,
    positions: &Flux2PositionTables,
    num_steps: usize,
) -> Result<B::Tensor, BackendError> {
    let timesteps = schedule(num_steps, positions.output_image_rows).map_err(|msg| BackendError::Compute { msg })?;
    for pair in timesteps.windows(2) {
        let velocity = forward(backend, source, global, &image, text, pair[0], positions)?;
        image = backend.flow_step(&image, &velocity, pair[1] - pair[0])?;
    }
    Ok(image)
}

pub fn denoise_conditioned<B: DiffusionBackend>(
    backend: &B,
    source: &Flux2KleinSource,
    global: &Flux2PreparedGlobal<B::Weight>,
    text: &B::Tensor,
    reference: &B::Tensor,
    mut image: B::Tensor,
    positions: &Flux2PositionTables,
    num_steps: usize,
) -> Result<B::Tensor, BackendError> {
    if backend.token_rows(reference) + backend.token_rows(&image) != positions.image_rows {
        return Err(BackendError::Compute { msg: "FLUX.2 reference/output rows 与 position rows 不一致".to_owned() });
    }
    let timesteps = schedule(num_steps, positions.output_image_rows).map_err(|msg| BackendError::Compute { msg })?;
    for pair in timesteps.windows(2) {
        let conditioned = backend.concat_rows(&image, reference)?;
        let velocity = forward(backend, source, global, &conditioned, text, pair[0], positions)?;
        image = backend.flow_step(&image, &velocity, pair[1] - pair[0])?;
    }
    Ok(image)
}

pub struct Flux2PreparedVae<W> {
    batch_mean: Vec<f32>,
    batch_std: Vec<f32>,
    batch_shift: W,
    batch_weight: W,
    unpatch_scale: W,
    unpatch_bias: W,
    encoder_input: Flux2VaeConv<W>,
    down_resnets: Vec<Vec<Flux2VaeResnet<W>>>,
    downsamplers: Vec<Flux2VaeConv<W>>,
    encoder_mid_resnets: Vec<Flux2VaeResnet<W>>,
    encoder_mid_attention: Flux2VaeAttention<W>,
    encoder_output_norm_weight: W,
    encoder_output_norm_bias: W,
    encoder_output: Flux2VaeConv<W>,
    quant: Flux2VaeConv<W>,
    post_quant: Flux2VaeConv<W>,
    input: Flux2VaeConv<W>,
    mid_resnets: Vec<Flux2VaeResnet<W>>,
    mid_attention: Flux2VaeAttention<W>,
    up_resnets: Vec<Vec<Flux2VaeResnet<W>>>,
    upsamplers: Vec<Flux2VaeConv<W>>,
    nearest: Vec<W>,
    output_norm_weight: W,
    output_norm_bias: W,
    output: Flux2VaeConv<W>,
}

fn prepare_vae_tensor<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let rows = tensor.shape.first().copied().ok_or_else(|| BackendError::Compute { msg: format!("FLUX.2 VAE 权重 {} shape 为空", tensor.name) })?;
    let elements = tensor.shape.iter().try_fold(1usize, |size, dim| size.checked_mul(*dim)).ok_or_else(|| BackendError::Compute { msg: format!("FLUX.2 VAE 权重 {} shape 溢出", tensor.name) })?;
    let cols = elements.checked_div(rows).filter(|value| *value > 0).ok_or_else(|| BackendError::Compute { msg: format!("FLUX.2 VAE 权重 {} shape={:?} 非法", tensor.name, tensor.shape) })?;
    let values = tensor.to_f32().map_err(|msg| BackendError::Compute { msg })?;
    backend.prepare_f32(&values, rows, cols)
}

fn prepare_vae_conv<B: Backend>(backend: &B, source: &Flux2VaeConv) -> Result<Flux2VaeConv<B::Weight>, BackendError> {
    Ok(Flux2VaeConv { weight: prepare_vae_tensor(backend, &source.weight)?, bias: prepare_vae_tensor(backend, &source.bias)? })
}

fn prepare_vae_resnet<B: Backend>(backend: &B, source: &Flux2VaeResnet) -> Result<Flux2VaeResnet<B::Weight>, BackendError> {
    Ok(Flux2VaeResnet {
        norm1_weight: prepare_vae_tensor(backend, &source.norm1_weight)?,
        norm1_bias: prepare_vae_tensor(backend, &source.norm1_bias)?,
        conv1: prepare_vae_conv(backend, &source.conv1)?,
        norm2_weight: prepare_vae_tensor(backend, &source.norm2_weight)?,
        norm2_bias: prepare_vae_tensor(backend, &source.norm2_bias)?,
        conv2: prepare_vae_conv(backend, &source.conv2)?,
        shortcut: source.shortcut.as_ref().map(|weight| prepare_vae_conv(backend, weight)).transpose()?,
    })
}

fn prepare_vae_attention<B: Backend>(backend: &B, source: &Flux2VaeAttention) -> Result<Flux2VaeAttention<B::Weight>, BackendError> {
    Ok(Flux2VaeAttention {
        norm_weight: prepare_vae_tensor(backend, &source.norm_weight)?,
        norm_bias: prepare_vae_tensor(backend, &source.norm_bias)?,
        query: prepare_vae_conv(backend, &source.query)?,
        key: prepare_vae_conv(backend, &source.key)?,
        value: prepare_vae_conv(backend, &source.value)?,
        output: prepare_vae_conv(backend, &source.output)?,
    })
}

pub fn prepare_vae<B: VaeBackend>(backend: &B, source: &Flux2VaeSource) -> Result<Flux2PreparedVae<B::Weight>, BackendError> {
    let config = Flux2KleinVaeConfig::standard();
    let source = source.load().map_err(BackendError::ExpertLoad)?;
    let mean = source.batch_mean.to_f32().map_err(|msg| BackendError::Compute { msg })?;
    let variance = source.batch_variance.to_f32().map_err(|msg| BackendError::Compute { msg })?;
    let scale = variance.iter().map(|value| (value + config.batch_norm_eps).sqrt()).collect::<Vec<_>>();
    let mut batch_weight = vec![0.0; config.packed_channels * config.packed_channels];
    for (channel, value) in scale.iter().copied().enumerate() {
        batch_weight[channel * config.packed_channels + channel] = value;
    }
    let nearest = source
        .upsamplers
        .iter()
        .map(|conv| {
            let channels = conv.bias.shape[0];
            let mut values = vec![0.0; channels * 4 * channels];
            for output in 0..channels * 4 {
                values[output * channels + output / 4] = 1.0;
            }
            backend.prepare_f32(&values, channels * 4, channels)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Flux2PreparedVae {
        batch_mean: mean.clone(),
        batch_std: scale.clone(),
        batch_shift: backend.prepare_f32(&mean, 1, config.packed_channels)?,
        batch_weight: backend.prepare_f32(&batch_weight, config.packed_channels, config.packed_channels)?,
        unpatch_scale: backend.prepare_f32(&vec![1.0; config.latent_channels], 1, config.latent_channels)?,
        unpatch_bias: backend.prepare_f32(&vec![0.0; config.latent_channels], 1, config.latent_channels)?,
        encoder_input: prepare_vae_conv(backend, &source.encoder_input)?,
        down_resnets: source.down_resnets.iter().map(|block| block.iter().map(|weight| prepare_vae_resnet(backend, weight)).collect::<Result<_, _>>()).collect::<Result<_, _>>()?,
        downsamplers: source.downsamplers.iter().map(|weight| prepare_vae_conv(backend, weight)).collect::<Result<_, _>>()?,
        encoder_mid_resnets: source.encoder_mid_resnets.iter().map(|weight| prepare_vae_resnet(backend, weight)).collect::<Result<_, _>>()?,
        encoder_mid_attention: prepare_vae_attention(backend, &source.encoder_mid_attention)?,
        encoder_output_norm_weight: prepare_vae_tensor(backend, &source.encoder_output_norm_weight)?,
        encoder_output_norm_bias: prepare_vae_tensor(backend, &source.encoder_output_norm_bias)?,
        encoder_output: prepare_vae_conv(backend, &source.encoder_output)?,
        quant: prepare_vae_conv(backend, &source.quant)?,
        post_quant: prepare_vae_conv(backend, &source.post_quant)?,
        input: prepare_vae_conv(backend, &source.input)?,
        mid_resnets: source.mid_resnets.iter().map(|weight| prepare_vae_resnet(backend, weight)).collect::<Result<_, _>>()?,
        mid_attention: prepare_vae_attention(backend, &source.mid_attention)?,
        up_resnets: source.up_resnets.iter().map(|block| block.iter().map(|weight| prepare_vae_resnet(backend, weight)).collect::<Result<_, _>>()).collect::<Result<_, _>>()?,
        upsamplers: source.upsamplers.iter().map(|weight| prepare_vae_conv(backend, weight)).collect::<Result<_, _>>()?,
        nearest,
        output_norm_weight: prepare_vae_tensor(backend, &source.output_norm_weight)?,
        output_norm_bias: prepare_vae_tensor(backend, &source.output_norm_bias)?,
        output: prepare_vae_conv(backend, &source.output)?,
    })
}

fn vae_conv2d<B: VaeBackend>(backend: &B, input: &B::Tensor, weight: &Flux2VaeConv<B::Weight>, input_channels: usize, output_channels: usize, height: usize, width: usize, kernel: usize) -> Result<B::Tensor, BackendError> {
    backend.conv3d(
        input,
        &weight.weight,
        Some(&weight.bias),
        &Conv3dSpec { input_channels, output_channels, input_shape: [1, height, width], kernel: [1, kernel, kernel], stride: [1, 1, 1], padding: [0, kernel / 2, kernel / 2], causal: false },
    )
}

fn vae_resnet<B: VaeBackend>(backend: &B, input: &B::Tensor, weight: &Flux2VaeResnet<B::Weight>, input_channels: usize, output_channels: usize, height: usize, width: usize, config: &Flux2KleinVaeConfig) -> Result<B::Tensor, BackendError> {
    let hidden = backend.group_norm(input, &weight.norm1_weight, &weight.norm1_bias, config.norm_groups, config.norm_eps)?;
    let hidden = backend.silu(&hidden)?;
    let hidden = vae_conv2d(backend, &hidden, &weight.conv1, input_channels, output_channels, height, width, 3)?;
    let hidden = backend.group_norm(&hidden, &weight.norm2_weight, &weight.norm2_bias, config.norm_groups, config.norm_eps)?;
    let hidden = backend.silu(&hidden)?;
    let hidden = vae_conv2d(backend, &hidden, &weight.conv2, output_channels, output_channels, height, width, 3)?;
    match &weight.shortcut {
        Some(shortcut) => {
            let residual = vae_conv2d(backend, input, shortcut, input_channels, output_channels, height, width, 1)?;
            backend.flow_step(&residual, &hidden, 1.0)
        }
        None => backend.flow_step(input, &hidden, 1.0),
    }
}

fn vae_attention<B: VaeBackend>(backend: &B, input: &B::Tensor, weight: &Flux2VaeAttention<B::Weight>, channels: usize, spatial: usize, config: &Flux2KleinVaeConfig) -> Result<B::Tensor, BackendError> {
    let normalized = backend.group_norm(input, &weight.norm_weight, &weight.norm_bias, config.norm_groups, config.norm_eps)?;
    let tokens = backend.channels_to_time(&normalized, channels)?;
    let linear = |weight: &Flux2VaeConv<B::Weight>| -> Result<B::Tensor, BackendError> { backend.add_row_bias(&backend.linear(&tokens, &weight.weight)?, &weight.bias) };
    let query = linear(&weight.query)?;
    let key = linear(&weight.key)?;
    let value = linear(&weight.value)?;
    let attended = backend.full_attention(query, key, value, 1, channels, 1.0 / (channels as f32).sqrt())?;
    let output = backend.add_row_bias(&backend.linear(&attended, &weight.output.weight)?, &weight.output.bias)?;
    let output = backend.channels_to_time(&output, spatial)?;
    backend.flow_step(input, &output, 1.0)
}

/// 把 channel-major RGB `[-1,1]` 编码为 Transformer 使用的 `[H/16*W/16,128]` reference latent。
pub fn encode_vae<B: VaeBackend>(backend: &B, vae: &Flux2PreparedVae<B::Weight>, image: &B::Tensor, height: usize, width: usize) -> Result<B::Tensor, BackendError> {
    let config = Flux2KleinVaeConfig::standard();
    config.validate().map_err(|msg| BackendError::Compute { msg })?;
    if height == 0 || width == 0 || !height.is_multiple_of(16) || !width.is_multiple_of(16) || backend.token_rows(image) != 3 || backend.token_cols(image) != height * width {
        return Err(BackendError::Compute { msg: format!("FLUX.2 VAE encoder image=[{},{}] size={}x{} 必须是 channel-major RGB 且宽高为 16 的倍数", backend.token_rows(image), backend.token_cols(image), width, height) });
    }
    let mut hidden = vae_conv2d(backend, image, &vae.encoder_input, 3, 128, height, width, 3)?;
    let mut channels = 128;
    let mut current_height = height;
    let mut current_width = width;
    for (block, &output_channels) in config.block_channels.iter().enumerate() {
        for weight in &vae.down_resnets[block] {
            hidden = vae_resnet(backend, &hidden, weight, channels, output_channels, current_height, current_width, &config)?;
            channels = output_channels;
        }
        if block < vae.downsamplers.len() {
            hidden = backend.encoder_conv3d_zero_pad(
                &hidden,
                &vae.downsamplers[block].weight,
                Some(&vae.downsamplers[block].bias),
                &Conv3dSpec { input_channels: channels, output_channels: channels, input_shape: [1, current_height, current_width], kernel: [1, 3, 3], stride: [1, 2, 2], padding: [0, 0, 0], causal: false },
                [1, 1],
            )?;
            current_height /= 2;
            current_width /= 2;
        }
    }
    hidden = vae_resnet(backend, &hidden, &vae.encoder_mid_resnets[0], 512, 512, current_height, current_width, &config)?;
    hidden = vae_attention(backend, &hidden, &vae.encoder_mid_attention, 512, current_height * current_width, &config)?;
    hidden = vae_resnet(backend, &hidden, &vae.encoder_mid_resnets[1], 512, 512, current_height, current_width, &config)?;
    hidden = backend.group_norm(&hidden, &vae.encoder_output_norm_weight, &vae.encoder_output_norm_bias, config.norm_groups, config.norm_eps)?;
    hidden = backend.silu(&hidden)?;
    hidden = vae_conv2d(backend, &hidden, &vae.encoder_output, 512, 64, current_height, current_width, 3)?;
    hidden = vae_conv2d(backend, &hidden, &vae.quant, 64, 64, current_height, current_width, 1)?;
    let mean = backend.take_rows(&hidden, config.latent_channels)?;
    let mean = backend.vae_tensor_to_f32(&mean)?;
    let packed_height = current_height / config.patch[0];
    let packed_width = current_width / config.patch[1];
    let mut packed = vec![0.0f32; packed_height * packed_width * config.packed_channels];
    for y in 0..packed_height {
        for x in 0..packed_width {
            let row = y * packed_width + x;
            for channel in 0..config.latent_channels {
                for patch_y in 0..config.patch[0] {
                    for patch_x in 0..config.patch[1] {
                        let packed_channel = (channel * config.patch[0] + patch_y) * config.patch[1] + patch_x;
                        let source = channel * current_height * current_width + (y * config.patch[0] + patch_y) * current_width + x * config.patch[1] + patch_x;
                        packed[row * config.packed_channels + packed_channel] = (mean[source] - vae.batch_mean[packed_channel]) / vae.batch_std[packed_channel];
                    }
                }
            }
        }
    }
    backend.vae_tensor_from_f32(packed, packed_height * packed_width, config.packed_channels)
}

/// 解码 `[latent_height*latent_width, 128]` packed latent，返回 channel-major RGB `[3, image_height*image_width]`。
pub fn decode_vae<B: VaeBackend>(backend: &B, vae: &Flux2PreparedVae<B::Weight>, latent: &B::Tensor, latent_height: usize, latent_width: usize) -> Result<B::Tensor, BackendError> {
    let config = Flux2KleinVaeConfig::standard();
    config.validate().map_err(|msg| BackendError::Compute { msg })?;
    if backend.token_rows(latent) != latent_height * latent_width || backend.token_cols(latent) != config.packed_channels {
        return Err(BackendError::Compute { msg: format!("FLUX.2 VAE latent=[{},{}]，期望 [{},{}]", backend.token_rows(latent), backend.token_cols(latent), latent_height * latent_width, config.packed_channels) });
    }
    let latent = backend.add_row_bias(&backend.linear(latent, &vae.batch_weight)?, &vae.batch_shift)?;
    let height = latent_height * config.patch[0];
    let width = latent_width * config.patch[1];
    let latent = backend.unpatch_affine(&latent, &vae.unpatch_scale, &vae.unpatch_bias, [1, height, width], [1, config.patch[0], config.patch[1]], config.latent_channels)?;
    let latent = backend.channels_to_time(&latent, height * width)?;
    let mut hidden = vae_conv2d(backend, &latent, &vae.post_quant, 32, 32, height, width, 1)?;
    hidden = vae_conv2d(backend, &hidden, &vae.input, 32, 512, height, width, 3)?;
    hidden = vae_resnet(backend, &hidden, &vae.mid_resnets[0], 512, 512, height, width, &config)?;
    hidden = vae_attention(backend, &hidden, &vae.mid_attention, 512, height * width, &config)?;
    hidden = vae_resnet(backend, &hidden, &vae.mid_resnets[1], 512, 512, height, width, &config)?;
    let block_channels = [512, 512, 256, 128];
    let mut channels = 512;
    let mut current_height = height;
    let mut current_width = width;
    for (block, &output_channels) in block_channels.iter().enumerate() {
        for weight in &vae.up_resnets[block] {
            hidden = vae_resnet(backend, &hidden, weight, channels, output_channels, current_height, current_width, &config)?;
            channels = output_channels;
        }
        if block < vae.upsamplers.len() {
            let expanded = backend.conv3d(
                &hidden,
                &vae.nearest[block],
                None,
                &Conv3dSpec { input_channels: channels, output_channels: channels * 4, input_shape: [1, current_height, current_width], kernel: [1, 1, 1], stride: [1, 1, 1], padding: [0, 0, 0], causal: false },
            )?;
            hidden = backend.pixel_shuffle(&expanded, &PixelShuffleSpec { channels, height: current_height, width: current_width, upscale: 2 })?;
            current_height *= 2;
            current_width *= 2;
            hidden = vae_conv2d(backend, &hidden, &vae.upsamplers[block], channels, channels, current_height, current_width, 3)?;
        }
    }
    hidden = backend.group_norm(&hidden, &vae.output_norm_weight, &vae.output_norm_bias, config.norm_groups, config.norm_eps)?;
    hidden = backend.silu(&hidden)?;
    vae_conv2d(backend, &hidden, &vae.output, 128, 3, current_height, current_width, 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{BackendResources, cpu::CpuContext},
        kernel::cpu::CpuTensor,
    };

    fn tiny_config() -> Flux2KleinConfig {
        Flux2KleinConfig { input_channels: 4, text_dim: 6, hidden_size: 8, double_layers: 1, single_layers: 1, num_heads: 1, head_dim: 8, mlp_hidden_size: 24, axes_dim: [2, 2, 2, 2], rope_theta: 2_000.0, timestep_dim: 4, norm_eps: 1.0e-6 }
    }

    fn weight(backend: &CpuContext, rows: usize, cols: usize, seed: usize) -> <CpuContext as BackendResources>::Weight {
        let values = (0..rows * cols).map(|index| (((index + seed) % 17) as f32 - 8.0) * 0.01).collect::<Vec<_>>();
        backend.prepare_f32(&values, rows, cols).unwrap()
    }

    fn attention(backend: &CpuContext, config: &Flux2KleinConfig, seed: usize) -> Flux2AttentionWeights<<CpuContext as BackendResources>::Weight> {
        Flux2AttentionWeights {
            query: weight(backend, config.hidden_size, config.hidden_size, seed),
            key: weight(backend, config.hidden_size, config.hidden_size, seed + 1),
            value: weight(backend, config.hidden_size, config.hidden_size, seed + 2),
            query_norm: backend.prepare_f32(&vec![1.0; config.head_dim], 1, config.head_dim).unwrap(),
            key_norm: backend.prepare_f32(&vec![1.0; config.head_dim], 1, config.head_dim).unwrap(),
            output: weight(backend, config.hidden_size, config.hidden_size, seed + 3),
        }
    }

    fn mlp_weights(backend: &CpuContext, config: &Flux2KleinConfig, seed: usize) -> Flux2MlpWeights<<CpuContext as BackendResources>::Weight> {
        Flux2MlpWeights { input: weight(backend, config.mlp_hidden_size * 2, config.hidden_size, seed), output: weight(backend, config.hidden_size, config.mlp_hidden_size, seed + 1) }
    }

    fn global(backend: &CpuContext, config: &Flux2KleinConfig) -> Flux2PreparedGlobal<<CpuContext as BackendResources>::Weight> {
        Flux2PreparedGlobal {
            weights: Flux2GlobalWeights {
                image_input: weight(backend, config.hidden_size, config.input_channels, 1),
                text_input: weight(backend, config.hidden_size, config.text_dim, 2),
                time_input: weight(backend, config.hidden_size, config.timestep_dim, 3),
                time_output: weight(backend, config.hidden_size, config.hidden_size, 4),
                image_modulation: weight(backend, config.hidden_size * 6, config.hidden_size, 5),
                text_modulation: weight(backend, config.hidden_size * 6, config.hidden_size, 6),
                single_modulation: weight(backend, config.hidden_size * 3, config.hidden_size, 7),
                final_modulation: weight(backend, config.hidden_size * 2, config.hidden_size, 8),
                output: weight(backend, config.input_channels, config.hidden_size, 9),
            },
            unit_norm_weight: backend.prepare_f32(&vec![1.0; config.hidden_size], 1, config.hidden_size).unwrap(),
            unit_norm_bias: backend.prepare_f32(&vec![0.0; config.hidden_size], 1, config.hidden_size).unwrap(),
        }
    }

    #[test]
    fn position_tables_follow_four_axis_interleaved_layout() {
        let mut config = Flux2KleinConfig::klein_4b();
        config.hidden_size = 8;
        config.num_heads = 1;
        config.head_dim = 8;
        config.mlp_hidden_size = 24;
        config.axes_dim = [2, 2, 2, 2];
        let (text, image) = position_ids(2, 1, 2).unwrap();
        let table = position_tables(&config, &text, &image).unwrap();
        assert_eq!(table.cosine.len(), 4 * 4);
        assert_eq!(&text, &[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]);
        assert_eq!(&image, &[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]);
        assert!((table.cosine[7] - 1.0_f32.cos()).abs() < 1.0e-6);
        assert!((table.cosine[14] - 1.0_f32.cos()).abs() < 1.0e-6);
    }

    #[test]
    fn conditioned_positions_keep_output_prefix_separate() {
        let mut config = tiny_config();
        config.axes_dim = [2, 2, 2, 2];
        let (text, mut image) = position_ids(2, 1, 2).unwrap();
        image.extend([[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 1.0, 0.0]]);
        let table = position_tables_conditioned(&config, &text, &image, 2).unwrap();
        assert_eq!(table.text_rows, 2);
        assert_eq!(table.image_rows, 4);
        assert_eq!(table.output_image_rows, 2);
    }

    #[test]
    fn official_schedule_has_strict_endpoints() {
        let timesteps = schedule(4, 4_096).unwrap();
        assert_eq!(timesteps.len(), 5);
        assert_eq!(timesteps[0], 1.0);
        assert_eq!(timesteps[4], 0.0);
        assert!(timesteps.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn cpu_double_and_single_blocks_form_a_finite_closed_path() {
        let backend = CpuContext;
        let config = tiny_config();
        config.validate().unwrap();
        let global = global(&backend, &config);
        let (text_ids, image_ids) = position_ids(2, 1, 2).unwrap();
        let positions = position_tables(&config, &text_ids, &image_ids).unwrap();
        let image = CpuTensor { data: (0..16).map(|index| index as f32 * 0.01).collect(), rows: 2, cols: 8 };
        let text = CpuTensor { data: (0..16).map(|index| index as f32 * -0.01).collect(), rows: 2, cols: 8 };
        let modulation = |value| DoubleModulation {
            attention_shift: CpuTensor { data: vec![value; 8], rows: 1, cols: 8 },
            attention_scale: CpuTensor { data: vec![0.0; 8], rows: 1, cols: 8 },
            attention_gate: CpuTensor { data: vec![0.5; 8], rows: 1, cols: 8 },
            mlp_shift: CpuTensor { data: vec![value; 8], rows: 1, cols: 8 },
            mlp_scale: CpuTensor { data: vec![0.0; 8], rows: 1, cols: 8 },
            mlp_gate: CpuTensor { data: vec![0.5; 8], rows: 1, cols: 8 },
        };
        let double_weights =
            Flux2DoubleBlockWeights { image_attention: attention(&backend, &config, 10), image_mlp: mlp_weights(&backend, &config, 20), text_attention: attention(&backend, &config, 30), text_mlp: mlp_weights(&backend, &config, 40) };
        let (image, text) = double_block(&backend, &config, &global, &image, &text, &modulation(0.01), &modulation(-0.01), &double_weights, &positions).unwrap();
        let hidden = backend.concat_rows(&text, &image).unwrap();
        let single_weights = Flux2SingleBlockWeights {
            input: weight(&backend, config.hidden_size * 3 + config.mlp_hidden_size * 2, config.hidden_size, 50),
            query_norm: backend.prepare_f32(&vec![1.0; config.head_dim], 1, config.head_dim).unwrap(),
            key_norm: backend.prepare_f32(&vec![1.0; config.head_dim], 1, config.head_dim).unwrap(),
            output: weight(&backend, config.hidden_size, config.hidden_size + config.mlp_hidden_size, 60),
        };
        let output = single_block(
            &backend,
            &config,
            &global,
            &hidden,
            &SingleModulation { shift: CpuTensor { data: vec![0.0; 8], rows: 1, cols: 8 }, scale: CpuTensor { data: vec![0.0; 8], rows: 1, cols: 8 }, gate: CpuTensor { data: vec![0.5; 8], rows: 1, cols: 8 } },
            &single_weights,
            &positions,
        )
        .unwrap();
        assert_eq!((output.rows, output.cols), (4, 8));
        assert!(output.data.iter().all(|value| value.is_finite()));
    }
}
