//! SenseVoice-Small 平台无关运行时（SAN-M encoder + CTC 头）。
//!
//! 结构（对照 FunASR `sense_voice/model.py`，2026-09-08 钉死）：
//! - 输入行 = [lang, event, emo, textnorm] 4 个 query + LFR 帧（前端已乘 √512 加 PE）
//! - 层 0（encoders0.0，560→512）：attention 分支无残差；其余 69 层标准残差
//! - 每层：LayerNorm → 融合 QKV 线性 → 全局 MHA + FSMN depthwise 记忆分支
//!   （`linear_out(attn) + (v + depthwise(v))`）→ 残差；LayerNorm → ReLU FFN → 残差
//! - 第 50 层后 after_norm，第 70 层后 tp_norm，最后 CTC 线性头出逐帧 logits
//!
//! ASR encoder 单遍前向、无跨 token 复用，权重按层惰性加载（load → prepare →
//! forward → drop），常驻内存只有单层 ~13MB，而不是整个 234M 模型。

pub mod cpu;
pub mod decode;
pub mod frontend;

use crate::backend::{Backend, BackendError, DiffusionBackend, LinearWeight};
use crate::model_spec::sensevoice::SenseVoiceConfig;
use crate::weight::model::sensevoice::{SenseVoiceLayerWeights, SenseVoiceWeights};

pub use crate::model_spec::sensevoice::{SenseVoiceLanguage, SenseVoiceTextNorm};

/// 单层已 prepare 的权重（backend-resident）。
pub struct SenseVoiceLayer<W> {
    pub norm1_weight: W,
    pub norm1_bias: W,
    pub qkv: W,
    pub qkv_bias: W,
    pub attn_out: W,
    pub attn_out_bias: W,
    pub fsmn: W,
    pub norm2_weight: W,
    pub norm2_bias: W,
    pub ffn_up: W,
    pub ffn_up_bias: W,
    pub ffn_down: W,
    pub ffn_down_bias: W,
}

/// host 层权重 → backend 常驻权重。`first` 层 norm1/qkv 输入是 560 维。
pub fn prepare_sensevoice_layer<B: Backend>(backend: &B, weights: &SenseVoiceLayerWeights, config: &SenseVoiceConfig, first: bool) -> Result<SenseVoiceLayer<B::Weight>, BackendError> {
    let attention_dim = config.attention_heads * config.head_dim;
    let input_dim = if first { config.encoder_input_dim() } else { config.d_model };
    let rows = 1;
    let bias = |values: &[f32]| backend.prepare_f32(values, rows, values.len());
    Ok(SenseVoiceLayer {
        norm1_weight: backend.prepare_f32(&weights.norm1_weight, rows, input_dim)?,
        norm1_bias: backend.prepare_f32(&weights.norm1_bias, rows, input_dim)?,
        qkv: backend.prepare_weight(LinearWeight::F32(&weights.qkv_weight), 3 * attention_dim, input_dim)?,
        qkv_bias: bias(&weights.qkv_bias)?,
        attn_out: backend.prepare_weight(LinearWeight::F32(&weights.attn_out_weight), attention_dim, attention_dim)?,
        attn_out_bias: bias(&weights.attn_out_bias)?,
        fsmn: backend.prepare_f32(&weights.fsmn_weight, rows, config.fsmn_kernel * attention_dim)?,
        norm2_weight: backend.prepare_f32(&weights.norm2_weight, rows, config.d_model)?,
        norm2_bias: backend.prepare_f32(&weights.norm2_bias, rows, config.d_model)?,
        ffn_up: backend.prepare_weight(LinearWeight::F32(&weights.ffn_up_weight), config.ffn_dim, config.d_model)?,
        ffn_up_bias: bias(&weights.ffn_up_bias)?,
        ffn_down: backend.prepare_weight(LinearWeight::F32(&weights.ffn_down_weight), config.d_model, config.ffn_dim)?,
        ffn_down_bias: bias(&weights.ffn_down_bias)?,
    })
}

/// SAN-M 单层 forward。`residual` 为 false 时 attention 分支不加残差（首层 560→512）。
pub fn sensevoice_layer<B: Backend + DiffusionBackend>(backend: &B, config: &SenseVoiceConfig, layer: &SenseVoiceLayer<B::Weight>, residual: bool, hidden: B::Tensor) -> Result<B::Tensor, BackendError> {
    let attention_dim = config.attention_heads * config.head_dim;
    let normed = backend.layernorm_bias(&hidden, &layer.norm1_weight, &layer.norm1_bias, config.norm_eps)?;
    let qkv = backend.linear(&normed, &layer.qkv)?;
    let qkv = backend.add_row_bias(&qkv, &layer.qkv_bias)?;
    let (query, key_value) = backend.split_columns(&qkv, attention_dim)?;
    let (key, value) = backend.split_columns(&key_value, attention_dim)?;
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    // FSMN 记忆分支：v + depthwise(v)，sanm_shfit=0 对称零填充；先算记忆再消费 value
    let padding = config.fsmn_padding();
    let memory = backend.depthwise_conv1d(&value, &layer.fsmn, config.fsmn_kernel, padding, padding)?;
    let memory = backend.add(&memory, &value)?;
    let attention = backend.full_attention(query, key, value, config.attention_heads, config.head_dim, scale)?;
    let attention = backend.linear(&attention, &layer.attn_out)?;
    let attention = backend.add_row_bias(&attention, &layer.attn_out_bias)?;
    let update = backend.add(&attention, &memory)?;
    let hidden = if residual { backend.add(&hidden, &update)? } else { update };

    let normed = backend.layernorm_bias(&hidden, &layer.norm2_weight, &layer.norm2_bias, config.norm_eps)?;
    let mut projected = backend.linear(&normed, &layer.ffn_up)?;
    projected = backend.add_row_bias(&projected, &layer.ffn_up_bias)?;
    projected = backend.relu(&projected)?;
    projected = backend.linear(&projected, &layer.ffn_down)?;
    projected = backend.add_row_bias(&projected, &layer.ffn_down_bias)?;
    backend.add(&hidden, &projected)
}

/// 完整 encoder：流式逐层（load → prepare → forward → drop），
/// 第 `layer_count` 层前插入 after_norm，末尾 tp_norm + CTC 头。
/// 输入 `[rows][encoder_input_dim]`，输出逐帧 logits `[rows][vocab_size]`。
pub fn sensevoice_encode<B: Backend + DiffusionBackend>(backend: &B, config: &SenseVoiceConfig, weights: &SenseVoiceWeights, mut hidden: B::Tensor) -> Result<B::Tensor, BackendError> {
    if backend.token_cols(&hidden) != config.encoder_input_dim() {
        return Err(crate::runtime::compute_error(format!("SenseVoice encoder 输入 cols={}，期望 {}", backend.token_cols(&hidden), config.encoder_input_dim())));
    }
    if backend.token_rows(&hidden) <= config.query_prefix {
        return Err(crate::runtime::compute_error(format!("SenseVoice encoder 输入 rows={}，必须大于 {} 个控制位", backend.token_rows(&hidden), config.query_prefix)));
    }
    let final_norm = |hidden: B::Tensor, tp: bool| -> Result<B::Tensor, BackendError> {
        let (weight, bias) = weights.final_norm(tp).map_err(crate::runtime::compute_error)?;
        let weight = backend.prepare_f32(&weight, 1, config.d_model)?;
        let bias = backend.prepare_f32(&bias, 1, config.d_model)?;
        backend.layernorm_bias(&hidden, &weight, &bias, config.norm_eps)
    };
    backend.begin_batch();
    for index in 0..config.total_layer_count() {
        if index == config.layer_count {
            hidden = final_norm(hidden, false)?;
        }
        let _scope = backend.layer_scope();
        let source = weights.load_layer(index).map_err(crate::runtime::compute_error)?;
        let layer = prepare_sensevoice_layer(backend, &source, config, index == 0)?;
        hidden = sensevoice_layer(backend, config, &layer, index != 0, hidden)?;
    }
    hidden = final_norm(hidden, true)?;
    let (ctc_weight, ctc_bias) = weights.ctc_head().map_err(crate::runtime::compute_error)?;
    let ctc_weight = backend.prepare_weight(LinearWeight::F32(&ctc_weight), weights.vocab_size(), config.d_model)?;
    let ctc_bias = backend.prepare_f32(&ctc_bias, 1, weights.vocab_size())?;
    let logits = backend.linear(&hidden, &ctc_weight)?;
    let logits = backend.add_row_bias(&logits, &ctc_bias)?;
    backend.finish_batch();
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuContext;
    use crate::backend::VaeBackend;
    use crate::model_spec::sensevoice::SenseVoiceConfig;

    /// 与朴素标量 reference 对拍单层 SAN-M forward，验证算子组合与顺序。
    /// reference 独立实现 LayerNorm / 线性 / 多头 softmax attention / depthwise 卷积。
    #[test]
    fn sanm_layer_matches_scalar_reference() {
        let config = SenseVoiceConfig::standard();
        let attention_dim = config.attention_heads * config.head_dim;
        let input_dim = config.encoder_input_dim();
        let rows = 3usize;
        let input = random_vec(0.5, 0.0, rows * input_dim);
        let layer = SenseVoiceLayerWeights {
            norm1_weight: random_vec(0.2, 0.9, input_dim),
            norm1_bias: random_vec(0.1, 0.0, input_dim),
            qkv_weight: random_vec(0.1, 0.0, 3 * attention_dim * input_dim),
            qkv_bias: random_vec(0.1, 0.0, 3 * attention_dim),
            attn_out_weight: random_vec(0.1, 0.0, attention_dim * attention_dim),
            attn_out_bias: random_vec(0.1, 0.0, attention_dim),
            fsmn_weight: random_vec(0.1, 0.0, config.fsmn_kernel * attention_dim),
            norm2_weight: random_vec(0.2, 0.9, config.d_model),
            norm2_bias: random_vec(0.1, 0.0, config.d_model),
            ffn_up_weight: random_vec(0.1, 0.0, config.ffn_dim * config.d_model),
            ffn_up_bias: random_vec(0.1, 0.0, config.ffn_dim),
            ffn_down_weight: random_vec(0.1, 0.0, config.d_model * config.ffn_dim),
            ffn_down_bias: random_vec(0.1, 0.0, config.d_model),
        };
        let backend = CpuContext;
        let prepared = prepare_sensevoice_layer(&backend, &layer, &config, true).unwrap();
        let tensor = backend.vae_tensor_from_f32(input.clone(), rows, input_dim).unwrap();
        let output = sensevoice_layer(&backend, &config, &prepared, false, tensor).unwrap();
        let actual = backend.vae_tensor_to_f32(&output).unwrap();
        let expected = scalar_reference(&config, &layer, &input, rows);
        // 随机大权重下幅值可达百级，f32x8 分块求和与标量顺序差异用相对容差
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 1e-3 + expected.abs() * 1e-4;
            assert!((actual - expected).abs() < tolerance, "index={index} actual={actual:.6} expected={expected:.6}");
        }
    }

    thread_local! {
        static RANDOM: std::cell::RefCell<Lcg> = std::cell::RefCell::new(Lcg(0x5EED_0001));
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        }
    }

    fn random_vec(scale: f32, offset: f32, len: usize) -> Vec<f32> {
        RANDOM.with(|cell| {
            let mut random = cell.borrow_mut();
            (0..len).map(|_| random.next() * scale + offset).collect()
        })
    }

    /// 朴素实现：out = norm2 后 FFN 残差加在 (attention 无残差分支) 上（首层语义）。
    #[allow(clippy::too_many_lines)]
    fn scalar_reference(config: &SenseVoiceConfig, layer: &SenseVoiceLayerWeights, input: &[f32], rows: usize) -> Vec<f32> {
        let attention_dim = config.attention_heads * config.head_dim;
        let input_dim = config.encoder_input_dim();
        let layernorm = |row: &[f32], weight: &[f32], bias: &[f32]| -> Vec<f32> {
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            let variance = row.iter().map(|&value| (value - mean) * (value - mean)).sum::<f32>() / row.len() as f32;
            let deviation = (variance + config.norm_eps).sqrt();
            row.iter().zip(weight).zip(bias).map(|((&value, &scale), &shift)| (value - mean) / deviation * scale + shift).collect()
        };
        let linear = |row: &[f32], weight: &[f32], bias: &[f32], out: usize| -> Vec<f32> {
            let cols = row.len();
            (0..out)
                .map(|o| {
                    let mut sum = bias[o];
                    for c in 0..cols {
                        sum += row[c] * weight[o * cols + c];
                    }
                    sum
                })
                .collect()
        };
        // attention + FSMN（无残差，首层）。先算全部行的 Q/K/V，再做跨行 softmax。
        let mut update = vec![0.0f32; rows * attention_dim];
        let mut query_rows = Vec::with_capacity(rows);
        let mut key_rows = Vec::with_capacity(rows);
        let mut value_rows = Vec::with_capacity(rows);
        for row in 0..rows {
            let normed = layernorm(&input[row * input_dim..(row + 1) * input_dim], &layer.norm1_weight, &layer.norm1_bias);
            let qkv = linear(&normed, &layer.qkv_weight, &layer.qkv_bias, 3 * attention_dim);
            query_rows.push(qkv[..attention_dim].to_vec());
            key_rows.push(qkv[attention_dim..2 * attention_dim].to_vec());
            value_rows.push(qkv[2 * attention_dim..].to_vec());
        }
        let padding = config.fsmn_padding();
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        for row in 0..rows {
            let query = &query_rows[row];
            for head in 0..config.attention_heads {
                let offset = head * config.head_dim;
                let mut scores = vec![0.0f32; rows];
                for other in 0..rows {
                    let mut dot = 0.0;
                    for d in 0..config.head_dim {
                        dot += query[offset + d] * key_rows[other][offset + d];
                    }
                    scores[other] = dot * scale;
                }
                let peak = scores.iter().cloned().fold(f32::MIN, f32::max);
                let total: f32 = scores.iter().map(|&score| (score - peak).exp()).sum();
                for d in 0..config.head_dim {
                    let mut mixed = 0.0;
                    for other in 0..rows {
                        mixed += (scores[other] - peak).exp() / total * value_rows[other][offset + d];
                    }
                    update[row * attention_dim + offset + d] = mixed;
                }
            }
            let attention = linear(&update[row * attention_dim..(row + 1) * attention_dim], &layer.attn_out_weight, &layer.attn_out_bias, attention_dim);
            // depthwise(v) + v
            for c in 0..attention_dim {
                let mut conv = 0.0;
                for tap in 0..config.fsmn_kernel {
                    let source = row + tap;
                    if source < padding || source >= padding + rows {
                        continue;
                    }
                    conv += value_rows[source - padding][c] * layer.fsmn_weight[tap * attention_dim + c];
                }
                update[row * attention_dim + c] = attention[c] + conv + value_rows[row][c];
            }
        }
        // FFN 残差（norm2 在 update 上）
        let mut output = vec![0.0f32; rows * config.d_model];
        for row in 0..rows {
            let normed = layernorm(&update[row * config.d_model..(row + 1) * config.d_model], &layer.norm2_weight, &layer.norm2_bias);
            let mut hidden = linear(&normed, &layer.ffn_up_weight, &layer.ffn_up_bias, config.ffn_dim);
            for value in &mut hidden {
                *value = value.max(0.0);
            }
            let projected = linear(&hidden, &layer.ffn_down_weight, &layer.ffn_down_bias, config.d_model);
            for c in 0..config.d_model {
                output[row * config.d_model + c] = update[row * config.d_model + c] + projected[c];
            }
        }
        output
    }
}
