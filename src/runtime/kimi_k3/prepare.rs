//! K3 checkpoint 权重到 backend resident 权重的薄适配层。

use crate::{
    backend::{Backend, BackendError, LinearWeight},
    moe::{DenseFfn, latent_moe::LatentMoeWeights},
    runtime::kimi_k3::KimiK3Config,
    runtime::output::{self, OutputHead},
    weight::{
        container::safetensor::TensorData,
        format::mxfp4::Mxfp4Matrix,
        model::kimi_k3::{KimiK3DenseMlpWeights, KimiK3GatedMlaWeights, KimiK3KdaConvWeights, KimiK3KdaWeights, KimiK3LatentMoeWeights, KimiK3LayerCommonWeights, KimiK3Weights},
    },
};

use super::layer::{AttnResWeights, GatedMlaLayerWeights, GatedMlaOutputPath, GatedMlaWeights, KdaConvPath, KdaDecayPath, KdaLayerWeights, KdaOutputPath, KdaWeights, MlaKvPath, MlaQueryPath};

pub struct KimiK3PreparedOutput<W> {
    pub attention_residual: AttnResWeights<W>,
    pub head: OutputHead<W>,
}

/// LM head 很大，调用方应只在确定其驻留策略后显式执行。
pub fn load_prepare_output<B: Backend>(backend: &B, source: &KimiK3Weights, config: &KimiK3Config) -> Result<KimiK3PreparedOutput<B::Weight>, BackendError> {
    load_prepare_output_quantized(backend, source, config, crate::weight::LmHeadQuantization::Native)
}

pub fn load_prepare_output_quantized<B: Backend>(backend: &B, source: &KimiK3Weights, config: &KimiK3Config, quantization: crate::weight::LmHeadQuantization) -> Result<KimiK3PreparedOutput<B::Weight>, BackendError> {
    let (attention_norm, attention_projection) = source.output_attn_res().map_err(|msg| output_load_error("output AttnRes", msg))?;
    let final_norm = source.final_norm().map_err(|msg| output_load_error("final norm", msg))?;
    let lm_head = source.lm_head().map_err(|msg| output_load_error("LM head", msg))?;
    let attention_residual = prepare_residual(backend, &attention_norm, &attention_projection)?;
    let head = output::prepare_output_head_weight_quantized(backend, LinearWeight::Bf16Bytes(&final_norm.data), LinearWeight::Bf16Bytes(&lm_head.data), config.vocab_size, config.hidden_size, quantization)?;
    Ok(KimiK3PreparedOutput { attention_residual, head })
}

pub fn prepare_kda_layer<B: Backend>(backend: &B, common: &KimiK3LayerCommonWeights, attention: &KimiK3KdaWeights) -> Result<KdaLayerWeights<B::Weight>, BackendError> {
    Ok(KdaLayerWeights {
        input_norm: prepare_bf16_tensor(backend, &common.input_norm)?,
        attention_residual: prepare_residual(backend, &common.attention_res_norm, &common.attention_res_projection)?,
        attention: prepare_kda(backend, attention)?,
        post_attention_norm: prepare_f32_tensor(backend, &common.post_attention_norm)?,
        mlp_residual: prepare_residual(backend, &common.mlp_res_norm, &common.mlp_res_projection)?,
    })
}

pub fn prepare_gated_mla_layer<B: Backend>(backend: &B, common: &KimiK3LayerCommonWeights, attention: &KimiK3GatedMlaWeights) -> Result<GatedMlaLayerWeights<B::Weight>, BackendError> {
    Ok(GatedMlaLayerWeights {
        input_norm: prepare_bf16_tensor(backend, &common.input_norm)?,
        attention_residual: prepare_residual(backend, &common.attention_res_norm, &common.attention_res_projection)?,
        attention: prepare_gated_mla(backend, attention)?,
        post_attention_norm: prepare_f32_tensor(backend, &common.post_attention_norm)?,
        mlp_residual: prepare_residual(backend, &common.mlp_res_norm, &common.mlp_res_projection)?,
    })
}

pub fn prepare_dense_mlp<B: Backend>(backend: &B, source: &KimiK3DenseMlpWeights) -> Result<DenseFfn<B::Weight>, BackendError> {
    Ok(DenseFfn { gate: prepare_bf16_tensor(backend, &source.gate_projection)?, up: prepare_bf16_tensor(backend, &source.up_projection)?, down: prepare_bf16_tensor(backend, &source.down_projection)? })
}

pub fn prepare_latent_moe<B: Backend>(backend: &B, source: &KimiK3LatentMoeWeights) -> Result<LatentMoeWeights<B::Weight>, BackendError> {
    Ok(LatentMoeWeights {
        router_weight: prepare_f32_tensor(backend, &source.router)?,
        router_bias: prepare_f32_tensor(backend, &source.correction_bias)?,
        routed_down_projection: prepare_bf16_tensor(backend, &source.routed_down_projection)?,
        routed_norm: prepare_bf16_tensor(backend, &source.routed_norm)?,
        routed_up_projection: prepare_bf16_tensor(backend, &source.routed_up_projection)?,
        shared_gate: prepare_bf16_tensor(backend, &source.shared.gate_projection)?,
        shared_up: prepare_bf16_tensor(backend, &source.shared.up_projection)?,
        shared_down: prepare_bf16_tensor(backend, &source.shared.down_projection)?,
    })
}

pub fn prepare_kda<B: Backend>(backend: &B, source: &KimiK3KdaWeights) -> Result<KdaWeights<B::Weight>, BackendError> {
    Ok(KdaWeights {
        query: prepare_kda_conv(backend, &source.q)?,
        key: prepare_kda_conv(backend, &source.k)?,
        value: prepare_kda_conv(backend, &source.v)?,
        decay: KdaDecayPath {
            first_projection: prepare_bf16_tensor(backend, &source.f_a_projection)?,
            second_projection: prepare_bf16_tensor(backend, &source.f_b_projection)?,
            a_log: prepare_f32_tensor(backend, &source.a_log)?,
            dt_bias: prepare_f32_tensor(backend, &source.dt_bias)?,
        },
        beta_projection: prepare_bf16_tensor(backend, &source.beta_projection)?,
        output: KdaOutputPath { gate_projection: prepare_bf16_tensor(backend, &source.gate_projection)?, norm: prepare_bf16_tensor(backend, &source.output_norm)?, projection: prepare_bf16_tensor(backend, &source.output_projection)? },
    })
}

pub fn prepare_gated_mla<B: Backend>(backend: &B, source: &KimiK3GatedMlaWeights) -> Result<GatedMlaWeights<B::Weight>, BackendError> {
    Ok(GatedMlaWeights {
        query: MlaQueryPath {
            first_projection: prepare_bf16_tensor(backend, &source.query.a_projection)?,
            norm: prepare_bf16_tensor(backend, &source.query.a_norm)?,
            second_projection: prepare_bf16_tensor(backend, &source.query.b_projection)?,
        },
        kv: MlaKvPath {
            first_projection: prepare_bf16_tensor(backend, &source.key_value.a_projection)?,
            norm: prepare_bf16_tensor(backend, &source.key_value.a_norm)?,
            second_projection: prepare_bf16_tensor(backend, &source.key_value.b_projection)?,
        },
        output: GatedMlaOutputPath { gate_projection: prepare_bf16_tensor(backend, &source.gate_projection)?, projection: prepare_bf16_tensor(backend, &source.output_projection)? },
    })
}

fn prepare_residual<B: Backend>(backend: &B, norm: &TensorData, projection: &TensorData) -> Result<AttnResWeights<B::Weight>, BackendError> {
    Ok(AttnResWeights { norm: prepare_bf16_tensor(backend, norm)?, projection: prepare_bf16_tensor(backend, projection)? })
}

fn prepare_kda_conv<B: Backend>(backend: &B, source: &KimiK3KdaConvWeights) -> Result<KdaConvPath<B::Weight>, BackendError> {
    Ok(KdaConvPath { projection: prepare_bf16_tensor(backend, &source.projection)?, convolution: prepare_bf16_tensor(backend, &source.convolution)? })
}

/// PyTorch linear 使用 `[out,in]`；卷积权重保留首维并展平其余维度。
pub fn prepare_bf16_tensor<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let (&first, rest) = tensor.shape.split_first().ok_or_else(|| crate::runtime::compute_error(format!("{} 缺少 tensor shape", tensor.name)))?;
    let (rows, cols) = if rest.is_empty() {
        (1, first)
    } else {
        let cols = rest.iter().try_fold(1usize, |size, &dim| size.checked_mul(dim).ok_or_else(|| crate::runtime::compute_error(format!("{} tensor shape {:?} 溢出", tensor.name, tensor.shape))))?;
        (first, cols)
    };
    prepare_bf16(backend, tensor, rows, cols)
}

pub fn prepare_f32_tensor<B: Backend>(backend: &B, tensor: &TensorData) -> Result<B::Weight, BackendError> {
    let (&first, rest) = tensor.shape.split_first().ok_or_else(|| crate::runtime::compute_error(format!("{} 缺少 tensor shape", tensor.name)))?;
    let cols = rest.iter().try_fold(1usize, |size, &dim| size.checked_mul(dim).ok_or_else(|| crate::runtime::compute_error(format!("{} tensor shape {:?} 溢出", tensor.name, tensor.shape))))?;
    let (rows, cols) = if rest.is_empty() { (1, first) } else { (first, cols) };
    let elements = rows.checked_mul(cols).ok_or_else(|| crate::runtime::compute_error(format!("{} resident shape [{rows},{cols}] 溢出", tensor.name)))?;
    let values = tensor.to_f32().map_err(crate::runtime::compute_error)?;
    if values.len() != elements {
        return Err(crate::runtime::compute_error(format!("{} 解码元素数 {}，resident [{rows},{cols}] 需要 {elements}", tensor.name, values.len())));
    }
    backend.prepare_f32(&values, rows, cols)
}

/// BF16 tensor 保持原始字节直到 backend 准备 resident 权重。
pub fn prepare_bf16<B: Backend>(backend: &B, tensor: &TensorData, rows: usize, cols: usize) -> Result<B::Weight, BackendError> {
    if tensor.dtype != "BF16" {
        return Err(crate::runtime::compute_error(format!("{} dtype={}，K3 resident 权重期望 BF16", tensor.name, tensor.dtype)));
    }
    let elements = rows.checked_mul(cols).ok_or_else(|| crate::runtime::compute_error(format!("{} resident shape [{rows},{cols}] 溢出", tensor.name)))?;
    let actual_elements = tensor.shape.iter().try_fold(1usize, |size, &dim| size.checked_mul(dim).ok_or_else(|| crate::runtime::compute_error(format!("{} tensor shape {:?} 溢出", tensor.name, tensor.shape))))?;
    if actual_elements != elements {
        return Err(crate::runtime::compute_error(format!("{} shape {:?} 有 {actual_elements} 个元素，resident [{rows},{cols}] 需要 {elements}", tensor.name, tensor.shape)));
    }
    let expected_bytes = elements.checked_mul(2).ok_or_else(|| crate::runtime::compute_error(format!("{} BF16 字节数溢出", tensor.name)))?;
    if tensor.data.len() != expected_bytes {
        return Err(crate::runtime::compute_error(format!("{} BF16 字节数 {}，期望 {expected_bytes}", tensor.name, tensor.data.len())));
    }
    backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, cols)
}

/// MXFP4 直接交给 backend；CPU oracle 可展开，Metal 保持 packed 并原生执行。
pub fn prepare_mxfp4<B: Backend>(backend: &B, matrix: &Mxfp4Matrix, rows: usize, cols: usize) -> Result<B::Weight, BackendError> {
    if matrix.rows() != rows || matrix.cols() != cols {
        return Err(crate::runtime::compute_error(format!("K3 MXFP4 shape [{},{}]，期望 [{rows},{cols}]", matrix.rows(), matrix.cols())));
    }
    backend.prepare_weight(LinearWeight::mxfp4(matrix), rows, cols)
}

pub fn prepare_mxfp4_tensor<B: Backend>(backend: &B, matrix: &Mxfp4Matrix) -> Result<B::Weight, BackendError> {
    prepare_mxfp4(backend, matrix, matrix.rows(), matrix.cols())
}

fn output_load_error(part: &str, msg: String) -> BackendError {
    BackendError::Compute { msg: format!("Kimi K3 加载 {part}: {msg}") }
}
