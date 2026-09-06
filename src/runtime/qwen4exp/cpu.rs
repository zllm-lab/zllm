//! Qwen4-Exp × CPU reference 组合(后续 CUDA lazy experts 路径的 oracle)。
//!
//! 逐算子对照 llama.cpp `qwen4exp.cpp`:
//! - Hyper-Connection 四流残差:分组 RMSNorm → 低秩 sigmoid 读门 → 均值混合;
//!   写门 = 2·sigmoid(inject/streams)。
//! - QSA:索引器 raw-K 入缓存 → 满 4-token 块均值池化 → RMSNorm+RoPE →
//!   ReLU 逐头点积求和 → tail/死块 1e9 偏置 → top(width) cell → 稠密 GQA 掩码。
//! - PLE:n-gram 哈希行 gather → key/value 投影 → 门控注回 → 膨胀深度卷积。
//! - GDN 复用 hybrid 底座(输出门 sigmoid 由 spec 携带)。

use crate::{
    attention::{
        gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetState},
        hybrid::{HybridAttention, HybridAttentionOptions},
        rope::{RopeTable, RotaryLayout},
    },
    backend::{
        Backend, BackendError, BackendResources, GqaPrefillBackend,
        cpu::{CpuContext, CpuGatedDeltaNetStorage, CpuPrefillExperts, CpuWeight},
    },
    kernel::cpu::{CpuTensor, sigmoid},
    moe::{
        UncachedMoeState,
        expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
        prefill::prefill_experts_observed,
        topk_moe::{MoeFfnRef, SharedExpertRef},
    },
    runtime::{
        expert_pipeline::{ExpertDecodePipeline, ExpertDecodeRequest},
        qwen4exp::{self, Qwen4ExpGguf, Qwen4ExpHyperConnection, Qwen4ExpLayer, Qwen4ExpMixer, Qwen4ExpPle},
    },
    weight::expert_source::ExpertSourceProvider,
};
use std::{path::Path, sync::Arc, time::Instant};

/// QSA 层自有的 K/V 与索引器 K 缓存(单序列追加式,reference 语义)。
struct QsaCache {
    key: Vec<f32>,
    value: Vec<f32>,
    index_key: Vec<f32>,
}

/// PLE 卷积历史 + 会话 token 序列(供 n-gram 前驱)。
struct PleState {
    conv_history: Vec<f32>,
    tokens: Vec<u32>,
}

pub struct Qwen4ExpCpuState {
    qsa: Vec<Option<QsaCache>>,
    delta: GatedDeltaNetState<CpuGatedDeltaNetStorage>,
    ple: PleState,
}

impl Qwen4ExpCpuState {
    pub fn new(cfg: &qwen4exp::Qwen4ExpConfig) -> Result<Self, String> {
        let ple_history = cfg.hc_dim() * cfg.ple.as_ref().map_or(0, |ple| ple.conv_history());
        Ok(Self {
            qsa: (0..cfg.num_layers).map(|layer| cfg.is_full_attention(layer).then(|| QsaCache { key: Vec::new(), value: Vec::new(), index_key: Vec::new() })).collect(),
            delta: GatedDeltaNetState::with_head_layout(cfg.num_layers, cfg.gated_delta_net_spec(), GatedDeltaNetHeadLayout::Tiled).map_err(|error| format!("Qwen4-Exp GDN state: {error:?}"))?,
            ple: PleState { conv_history: vec![0.0; ple_history], tokens: Vec::new() },
        })
    }
}

// ============================================================================
// Hyper-Connection 数学(f32 reference)。
// ============================================================================

/// 分组 RMSNorm:每流独立归一化后乘以跨流 gamma `[hc_dim]`。
fn grouped_rmsnorm_scale(input: &[f32], rows: usize, hidden: usize, streams: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    let hc_dim = streams * hidden;
    let mut output = vec![0.0; rows * hc_dim];
    for row in 0..rows {
        for stream in 0..streams {
            let base = row * hc_dim + stream * hidden;
            let variance = input[base..base + hidden].iter().map(|value| value * value).sum::<f32>() / hidden as f32;
            let inv_rms = 1.0 / (variance + eps).sqrt();
            for column in 0..hidden {
                output[base + column] = input[base + column] * inv_rms * weight[stream * hidden + column];
            }
        }
    }
    output
}

/// HC mix:分组 RMSNorm → 低秩门 → 门控均值混合;返回 (mixed, inject)。
fn hc_mix(backend: &CpuContext, cfg: &qwen4exp::Qwen4ExpConfig, residual: &CpuTensor, hc: &Qwen4ExpHyperConnection<CpuWeight>) -> Result<(CpuTensor, CpuTensor), BackendError> {
    let streams = cfg.hyper_connection.streams;
    let hidden = cfg.hidden_size;
    let rows = residual.rows;
    let xn = grouped_rmsnorm_scale(&residual.data, rows, hidden, streams, &hc.norm.data(), cfg.rms_norm_eps);
    let xn = CpuTensor { data: xn, rows, cols: cfg.hc_dim() };
    let lo = backend.linear(&xn, &hc.down)?;
    let lo = CpuTensor {
        data: lo
            .data
            .iter()
            .map(|value| {
                let scaled = value / streams as f32;
                sigmoid(scaled) * scaled
            })
            .collect(),
        rows,
        cols: cfg.hyper_connection.low_rank,
    };
    let gate = backend.linear(&lo, &hc.up)?.data;
    let mut mixed = vec![0.0; rows * hidden];
    for row in 0..rows {
        for stream in 0..streams {
            let base = row * cfg.hc_dim() + stream * hidden;
            for column in 0..hidden {
                mixed[row * hidden + column] += xn.data[base + column] * sigmoid(gate[base + column]);
            }
        }
        for value in &mut mixed[row * hidden..(row + 1) * hidden] {
            *value /= streams as f32;
        }
    }
    let inject = match &hc.inject {
        Some(weight) => backend.linear(&xn, weight)?,
        None => CpuTensor { data: vec![0.0; rows * streams], rows, cols: streams },
    };
    Ok((CpuTensor { data: mixed, rows, cols: hidden }, inject))
}

/// HC combine:写门 = 2·sigmoid(inject/streams),残差按流加权累加块输出。
fn hc_combine(cfg: &qwen4exp::Qwen4ExpConfig, residual: &CpuTensor, block: &CpuTensor, inject: &CpuTensor) -> CpuTensor {
    let streams = cfg.hyper_connection.streams;
    let hidden = cfg.hidden_size;
    let mut output = residual.data.clone();
    for row in 0..residual.rows {
        for stream in 0..streams {
            let weight = 2.0 * sigmoid(inject.data[row * streams + stream] / streams as f32);
            let base = row * cfg.hc_dim() + stream * hidden;
            for column in 0..hidden {
                output[base + column] += block.data[row * hidden + column] * weight;
            }
        }
    }
    CpuTensor { data: output, rows: residual.rows, cols: cfg.hc_dim() }
}

// ============================================================================
// QSA(Qwen Sparse Attention)reference。
// ============================================================================

/// 选择 top-width 个可见 cell;返回 cell 位图。tail 块与死块(未满块)偏置
/// 1e9 恒被选中——与 llama.cpp set_input_qsa 的 bias 语义一致。
fn qsa_select_cell_mask(block_score: &[f32], dead_block: usize, n_cells: usize, query_position: usize, ratio: usize, top_k: usize) -> Vec<bool> {
    let tail_start = (query_position + 1) / ratio * ratio;
    let width = n_cells.min(top_k + ratio - 1);
    let mut cell_score = vec![f32::NEG_INFINITY; n_cells];
    for cell in 0..=query_position.min(n_cells - 1) {
        let block = cell / ratio;
        let bias = if block == dead_block || block * ratio >= tail_start { 1e9 } else { 0.0 };
        cell_score[cell] = block_score[block.min(block_score.len() - 1)] + bias;
    }
    let mut order = (0..n_cells).collect::<Vec<_>>();
    order.sort_by(|a, b| cell_score[*b].partial_cmp(&cell_score[*a]).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.cmp(b)));
    let mut mask = vec![false; n_cells];
    for cell in order.into_iter().take(width) {
        if cell_score[cell].is_finite() {
            mask[cell] = true;
        }
    }
    mask
}

/// 稠密 GQA over 选中 cell(集合外 -inf),逐 query 行计算。
fn qsa_gqa_masked(out: &mut [f32], query: &[f32], key: &[f32], value: &[f32], mask: &[bool], cfg: &qwen4exp::Qwen4ExpConfig) {
    let head_dim = cfg.head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = cfg.num_kv_heads * head_dim;
    for head in 0..cfg.num_attention_heads {
        let kv_head = head / (cfg.num_attention_heads / cfg.num_kv_heads);
        let q = &query[head * head_dim..(head + 1) * head_dim];
        let mut scores = Vec::with_capacity(mask.len());
        let mut max_score = f32::NEG_INFINITY;
        for (cell, visible) in mask.iter().enumerate() {
            if !visible {
                continue;
            }
            let offset = cell * kv_stride + kv_head * head_dim;
            let score = q.iter().zip(&key[offset..offset + head_dim]).map(|(a, b)| a * b).sum::<f32>() * scale;
            max_score = max_score.max(score);
            scores.push((cell, score));
        }
        if scores.is_empty() {
            continue;
        }
        let sum = scores.iter().map(|(_, score)| (score - max_score).exp()).sum::<f32>();
        for column in 0..head_dim {
            out[head * head_dim + column] = 0.0;
        }
        for (cell, score) in &scores {
            let probability = (score - max_score).exp() / sum;
            let offset = cell * kv_stride + kv_head * head_dim;
            for column in 0..head_dim {
                out[head * head_dim + column] += probability * value[offset + column];
            }
        }
    }
}

// ============================================================================
// PLE reference。
// ============================================================================

fn ple_forward(
    backend: &CpuContext,
    cfg: &qwen4exp::Qwen4ExpConfig,
    ple_cfg: &crate::model_spec::qwen4exp::Qwen4ExpPleConfig,
    weights: &Qwen4ExpPle<CpuWeight>,
    gguf: &Qwen4ExpGguf,
    state: &mut PleState,
    residual: &CpuTensor,
    tokens: &[u32],
) -> Result<CpuTensor, BackendError> {
    let streams = cfg.hyper_connection.streams;
    let hidden = cfg.hidden_size;
    let rows = tokens.len();
    let indices = qwen4exp::ple_row_indices(ple_cfg, tokens, &state.tokens);
    let emb = gguf.ple_rows_f32(ple_cfg, &indices).map_err(|error| BackendError::Compute { msg: format!("PLE 行 gather: {error}") })?;
    let emb = CpuTensor { data: emb, rows, cols: hidden };

    let key = backend.linear(&emb, &weights.key)?;
    let value = backend.linear(&emb, &weights.value)?;
    let key = grouped_rmsnorm_scale(&key.data, rows, hidden, streams, &weights.norm_key.data(), cfg.rms_norm_eps);
    let query = grouped_rmsnorm_scale(&residual.data, rows, hidden, streams, &weights.norm_query.data(), cfg.rms_norm_eps);

    // 逐流点积 → 符号 sqrt 门 → value 广播到各流
    let hc_dim = cfg.hc_dim();
    let mut gated = vec![0.0; rows * hc_dim];
    for row in 0..rows {
        for stream in 0..streams {
            let base = row * hc_dim + stream * hidden;
            let dot = (0..hidden).map(|column| key[base + column] * query[base + column]).sum::<f32>() / (hidden as f32).sqrt();
            let signed_root = if dot == 0.0 { 0.0 } else { dot.signum() * dot.abs().max(1e-6).sqrt() };
            let gate = sigmoid(signed_root);
            for column in 0..hidden {
                gated[base + column] = value.data[row * hidden + column] * gate;
            }
        }
    }
    let normalized = grouped_rmsnorm_scale(&gated, rows, hidden, streams, &weights.norm_conv.data(), cfg.rms_norm_eps);

    // 膨胀深度因果卷积:out[c,t] = Σ_k w[k,c]·x[c, t-(kern-1-k)·dil]
    let dilation = ple_cfg.ngram_size;
    let kernel = ple_cfg.conv_kernel;
    let history = ple_cfg.conv_history();
    let conv_weight = weights.conv1d.data();
    let mut padded = state.conv_history.clone();
    padded.extend_from_slice(&normalized);
    let total_rows = padded.len() / hc_dim;
    let mut conv_out = vec![0.0; rows * hc_dim];
    for row in 0..rows {
        for channel in 0..hc_dim {
            let mut sum = 0.0;
            for tap in 0..kernel {
                let back = (kernel - 1 - tap) * dilation;
                let source = history + row - back;
                if source < total_rows && back <= history + row {
                    sum += padded[source * hc_dim + channel] * conv_weight[channel * kernel + tap];
                }
            }
            conv_out[row * hc_dim + channel] = sigmoid(sum) * sum;
        }
    }
    // 回写末 history 列卷积历史
    let keep_from = total_rows.saturating_sub(history) * hc_dim;
    state.conv_history.copy_from_slice(&padded[keep_from..]);
    state.tokens.extend_from_slice(tokens);

    let mut output = residual.data.clone();
    for ((output, gated), conv) in output.iter_mut().zip(&gated).zip(&conv_out) {
        *output += gated + conv;
    }
    Ok(CpuTensor { data: output, rows, cols: hc_dim })
}

// ============================================================================
// 前向组合。
// ============================================================================

type MlpFn<'a> = dyn FnMut(usize, &CpuTensor) -> Result<CpuTensor, BackendError> + 'a;

struct Qwen4ExpCpuRuntime<'a> {
    backend: &'a CpuContext,
    cfg: &'a qwen4exp::Qwen4ExpConfig,
    gguf: &'a Qwen4ExpGguf,
    attention: HybridAttention<'a, CpuContext>,
    rope: &'a RopeTable,
}

impl<'a> Qwen4ExpCpuRuntime<'a> {
    fn new(backend: &'a CpuContext, cfg: &'a qwen4exp::Qwen4ExpConfig, gguf: &'a Qwen4ExpGguf, rope: &'a RopeTable) -> Self {
        Self { backend, cfg, gguf, attention: HybridAttention::new(backend, cfg.qsa_spec(), cfg.gated_delta_net_spec(), cfg.rms_norm_eps, rope, HybridAttentionOptions::default()), rope }
    }

    /// QSA 层:索引器选择 + 掩码 GQA;K/V/idxK 追加进 state。
    fn sparse_attention(&self, state: &mut Qwen4ExpCpuState, layer: usize, weights: &Qwen4ExpLayer<CpuWeight>, input: &CpuTensor, position: usize) -> Result<CpuTensor, BackendError> {
        let Qwen4ExpMixer::SparseAttention { attention, indexer } = &weights.mixer else {
            return Err(BackendError::Compute { msg: format!("Qwen4-Exp L{layer} 不是 QSA 层") });
        };
        let cfg = self.cfg;
        let rows = input.rows;
        let query_gate = self.backend.linear(input, &attention.query_gate)?;
        let (query, gate) = self.backend.split_interleaved_columns(&query_gate, cfg.head_dim)?;
        let (key, value) = self.backend.dual_linear(input, &attention.key, &attention.value)?;
        let query = self.backend.rmsnorm_heads(&query, &attention.query_norm, cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps)?;
        let key = self.backend.rmsnorm_heads(&key, &attention.key_norm, cfg.num_kv_heads, cfg.head_dim, cfg.rms_norm_eps)?;
        let query = self.backend.rope_prefix(&query, cfg.num_attention_heads, cfg.rope_dim, RotaryLayout::SplitHalf, position, &self.rope.cos, &self.rope.sin)?;
        let key = self.backend.rope_prefix(&key, cfg.num_kv_heads, cfg.rope_dim, RotaryLayout::SplitHalf, position, &self.rope.cos, &self.rope.sin)?;

        // 索引器:q 走 norm+rope;k 存 raw(池化发生在 norm/rope 之前)
        let index_key_raw = self.backend.linear(input, &indexer.k_proj)?;
        let index_query = self.backend.linear(input, &indexer.q_proj)?;
        let index_query = self.backend.rmsnorm_heads(&index_query, &indexer.q_norm, cfg.indexer.head_count, cfg.indexer.head_dim, cfg.rms_norm_eps)?;
        let index_query = self.backend.rope_prefix(&index_query, cfg.indexer.head_count, cfg.rope_dim, RotaryLayout::SplitHalf, position, &self.rope.cos, &self.rope.sin)?;

        let qsa = state.qsa[layer].as_mut().expect("QSA 层必有缓存");
        let prev_cells = qsa.index_key.len() / cfg.indexer.head_dim;
        qsa.index_key.extend_from_slice(&index_key_raw.data);
        qsa.key.extend_from_slice(&key.data);
        qsa.value.extend_from_slice(&value.data);
        let n_cells = prev_cells + rows;
        let ratio = cfg.indexer.compress_ratio;
        let n_full = n_cells / ratio;
        let n_blocks = n_cells.div_ceil(ratio);
        let dead_block = if n_full < n_blocks { n_full } else { n_blocks - 1 };

        // 满块均值池化 → RMSNorm → RoPE(块起点位置:表行 b 取基础表行 b·ratio)
        let head_dim = cfg.indexer.head_dim;
        let mut pooled = vec![0.0f32; n_full * head_dim];
        for block in 0..n_full {
            for member in 0..ratio {
                let source = (block * ratio + member) * head_dim;
                for column in 0..head_dim {
                    pooled[block * head_dim + column] += qsa.index_key[source + column];
                }
            }
            for value in &mut pooled[block * head_dim..(block + 1) * head_dim] {
                *value /= ratio as f32;
            }
        }
        let pooled = if n_full > 0 {
            let tensor = CpuTensor { data: pooled, rows: n_full, cols: head_dim };
            let normed = self.backend.rmsnorm_heads(&tensor, &indexer.k_norm, 1, head_dim, cfg.rms_norm_eps)?;
            let half = cfg.rope_dim / 2;
            let mut cos = Vec::with_capacity(n_full * half);
            let mut sin = Vec::with_capacity(n_full * half);
            for block in 0..n_full {
                let base = block * ratio * half;
                cos.extend_from_slice(&self.rope.cos[base..base + half]);
                sin.extend_from_slice(&self.rope.sin[base..base + half]);
            }
            self.backend.rope_prefix(&normed, 1, cfg.rope_dim, RotaryLayout::SplitHalf, 0, &cos, &sin)?.data
        } else {
            pooled
        };

        let mut attended = vec![0.0; rows * cfg.num_attention_heads * cfg.head_dim];
        let q_width = cfg.num_attention_heads * cfg.head_dim;
        for row in 0..rows {
            let query_position = position + row;
            // 逐块打分:ReLU 逐头点积求和(死块得 0 分,靠 1e9 偏置入选)
            let mut block_score = vec![0.0; n_blocks];
            for block in 0..n_blocks {
                if block == dead_block && n_full < n_blocks {
                    continue;
                }
                let k = &pooled[block * head_dim..(block + 1) * head_dim];
                let mut score = 0.0;
                for head in 0..cfg.indexer.head_count {
                    let base = (row * cfg.indexer.head_count + head) * head_dim;
                    let dot = k.iter().zip(&index_query.data[base..base + head_dim]).map(|(a, b)| a * b).sum::<f32>();
                    score += dot.max(0.0);
                }
                block_score[block] = score;
            }
            let mask = qsa_select_cell_mask(&block_score, dead_block, n_cells, query_position, ratio, cfg.indexer.top_k);
            let q = &query.data[row * q_width..(row + 1) * q_width];
            let out = &mut attended[row * q_width..(row + 1) * q_width];
            qsa_gqa_masked(out, q, &qsa.key, &qsa.value, &mask, cfg);
        }
        let attended = CpuTensor { data: attended, rows, cols: q_width };
        let gated = self.backend.sigmoid_gate(&attended, &gate)?;
        self.backend.linear(&gated, &attention.output)
    }

    fn run_layer(&self, state: &mut Qwen4ExpCpuState, mlp: &mut MlpFn, layer: usize, weights: &Qwen4ExpLayer<CpuWeight>, residual: CpuTensor, tokens: &[u32], position: usize) -> Result<CpuTensor, BackendError> {
        let cfg = self.cfg;
        let mut residual = residual;
        if let Some(ple) = &weights.ple {
            let ple_cfg = cfg.ple.as_ref().expect("PLE 层存在时配置必有 PLE");
            residual = ple_forward(self.backend, cfg, ple_cfg, ple, self.gguf, &mut state.ple, &residual, tokens)?;
        }
        let (mixed, inject) = hc_mix(self.backend, cfg, &residual, &weights.hc_attn)?;
        let mixed = match &weights.mixer {
            Qwen4ExpMixer::SparseAttention { .. } => self.sparse_attention(state, layer, weights, &mixed, position)?,
            Qwen4ExpMixer::Delta(weights) => self.attention.delta(weights, &mut state.delta, layer, &mixed, position)?,
        };
        residual = hc_combine(cfg, &residual, &mixed, &inject);
        let (mixed, inject) = hc_mix(self.backend, cfg, &residual, &weights.hc_ffn)?;
        let feedforward = mlp(layer, &mixed)?;
        residual = hc_combine(cfg, &residual, &feedforward, &inject);
        Ok(residual)
    }
}

/// 残差初始化:embedding 复制 hc 流份。
fn expand_streams(embedding: &CpuTensor, cfg: &qwen4exp::Qwen4ExpConfig) -> CpuTensor {
    let streams = cfg.hyper_connection.streams;
    let mut residual = Vec::with_capacity(embedding.rows * cfg.hc_dim());
    for row in 0..embedding.rows {
        for _ in 0..streams {
            residual.extend_from_slice(&embedding.data[row * cfg.hidden_size..(row + 1) * cfg.hidden_size]);
        }
    }
    CpuTensor { data: residual, rows: embedding.rows, cols: cfg.hc_dim() }
}

fn argmax_token(logits: &CpuTensor) -> u32 {
    logits.data.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(id, _)| id as u32).unwrap_or(0)
}

pub fn run(gguf_path: &Path, prompt: &str, max_seq_len: usize, decode_steps: usize) -> Result<(), Box<dyn std::error::Error>> {
    let weights = Arc::new(Qwen4ExpGguf::open(gguf_path)?);
    let cfg = weights.config().clone();
    let tokenizer = weights.tokenizer()?;
    let tokens = tokenizer.tokenize(prompt.as_bytes());
    eprintln!("[qwen4exp-input] tokens={tokens:?}");
    if tokens.is_empty() || tokens.len().saturating_add(decode_steps) > max_seq_len {
        return Err(format!("Qwen4-Exp prompt tokens={}，max_seq_len={max_seq_len}", tokens.len()).into());
    }

    let backend = CpuContext;
    let prepare_started = Instant::now();
    let layers = qwen4exp::prepare_qwen4exp_layers(&backend, weights.as_ref()).map_err(|error| format!("准备 Qwen4-Exp CPU 层: {error:?}"))?;
    let lm_head = crate::runtime::prepare_gguf_matrix(&backend, weights.reader(), "output.weight").map_err(|error| format!("准备 Qwen4-Exp lm_head: {error:?}"))?;
    let output_hc = Qwen4ExpHyperConnection {
        norm: crate::runtime::prepare_gguf_f32_vector(&backend, weights.reader(), "output_hc_norm.weight")?,
        down: crate::runtime::prepare_gguf_matrix(&backend, weights.reader(), "output_hc_down.weight")?,
        up: crate::runtime::prepare_gguf_matrix(&backend, weights.reader(), "output_hc_up.weight")?,
        inject: None,
    };
    eprintln!("[qwen4exp-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);
    let runtime = Qwen4ExpCpuRuntime::new(&backend, &cfg, weights.as_ref(), &rope);
    let mut state = Qwen4ExpCpuState::new(&cfg)?;
    let moe_spec = cfg.moe_spec();

    // prefill:CpuPrefillExperts 批量专家路径
    let expert_source: Arc<dyn crate::weight::expert_source::GgufExpertSource> = weights.clone();
    let mut prefill_experts = CpuPrefillExperts::gguf(expert_source);
    let mut prefill_moe = |layer: usize, input: &CpuTensor| -> Result<CpuTensor, BackendError> {
        let weights = &layers[layer];
        let shared = [SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }];
        let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
        prefill_experts_observed(&backend, &moe_spec, &reference, layer, &mut prefill_experts, input, None, |_| {}).map(|moe| moe.tensor)
    };

    let chunk_size = 64usize;
    let prefill_started = Instant::now();
    let mut hidden = None;
    for (index, chunk) in tokens.chunks(chunk_size).enumerate() {
        let embedding = CpuTensor { data: weights.embedding_rows(chunk)?, rows: chunk.len(), cols: cfg.hidden_size };
        let mut residual = expand_streams(&embedding, &cfg);
        backend.begin_batch();
        for (layer, weights) in layers.iter().enumerate() {
            residual = runtime.run_layer(&mut state, &mut prefill_moe, layer, weights, residual, chunk, index * chunk_size)?;
        }
        backend.finish_batch();
        hidden = Some(residual);
    }
    let hidden = hidden.expect("tokens 非空已校验");
    eprintln!("[qwen4exp-prefill] tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64());
    let residual = CpuTensor { data: hidden.row(hidden.rows - 1).to_vec(), rows: 1, cols: cfg.hc_dim() };

    // 最终 HC mix 即输出 norm
    let logits_of = |residual: &CpuTensor| -> Result<CpuTensor, BackendError> {
        let (mixed, _) = hc_mix(&backend, &cfg, residual, &output_hc)?;
        backend.linear(&mixed, &lm_head)
    };
    let mut current = argmax_token(&logits_of(&residual)?);
    eprintln!("[qwen4exp-first-token] id={current}");
    if decode_steps == 0 {
        return Ok(());
    }

    // decode:ExpertDecodePipeline 流式专家路径
    let mut expert_state = ExpertDecodePipeline::new(
        UncachedMoeState::default(),
        ExpertPredictorConfig { first_layer: 0, layer_count: cfg.num_layers, expert_count: cfg.num_experts, routed_top_k: cfg.num_experts_per_tok, prefetch_count: 0, weights: ExpertPredictorWeights::default() },
    )?;
    let detokenizer = weights.detokenizer()?;
    let gguf = weights.as_ref();
    let decode_started = Instant::now();
    crate::runtime::generation::write_token(&detokenizer, current, true)?;
    let mut generated = 1usize;
    for position in tokens.len()..tokens.len() + decode_steps.saturating_sub(1) {
        if cfg.eos_token_ids.contains(&current) {
            break;
        }
        let input = CpuTensor { data: weights.embedding_rows(&[current])?, rows: 1, cols: cfg.hidden_size };
        let mut step_residual = expand_streams(&input, &cfg);
        let mut decode_moe = |layer: usize, moe_input: &CpuTensor| -> Result<CpuTensor, BackendError> {
            let weights = &layers[layer];
            let shared = [SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }];
            let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
            let source = gguf.source(layer).map_err(BackendError::ExpertLoad)?;
            let next = (layer + 1 < cfg.num_layers).then(|| gguf.source(layer + 1).map(|source| (layer + 1, source))).transpose().map_err(BackendError::ExpertLoad)?;
            expert_state.decode(&backend, &moe_spec, &reference, ExpertDecodeRequest { layer, source, position, next }, moe_input)
        };
        backend.begin_batch();
        for (layer, weights) in layers.iter().enumerate() {
            step_residual = runtime.run_layer(&mut state, &mut decode_moe, layer, weights, step_residual, &[current], position)?;
        }
        backend.finish_batch();
        current = argmax_token(&logits_of(&step_residual)?);
        eprintln!("[qwen4exp-cpu-token] position={position} id={current} elapsed={:.6}s", decode_started.elapsed().as_secs_f64());
        generated += 1;
        crate::runtime::generation::write_token(&detokenizer, current, true)?;
        if cfg.eos_token_ids.contains(&current) {
            break;
        }
    }
    eprintln!("[qwen4exp-summary] generated={} wall={:.3}s", generated, decode_started.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_mix_matches_scaled_silu_and_sigmoid_equations() {
        let backend = CpuContext;
        let mut cfg = qwen4exp::Qwen4ExpConfig::standard_flash_next();
        cfg.hidden_size = 2;
        cfg.hyper_connection.streams = 2;
        cfg.hyper_connection.low_rank = 1;
        let residual = CpuTensor { data: vec![1.0, 2.0, -3.0, 4.0], rows: 1, cols: 4 };
        let hc =
            Qwen4ExpHyperConnection { norm: backend.prepare_f32(&[1.0; 4], 1, 4).unwrap(), down: backend.prepare_f32(&[1.0, 0.0, 0.0, 0.0], 1, 4).unwrap(), up: backend.prepare_f32(&[1.0, -1.0, 2.0, -2.0], 4, 1).unwrap(), inject: None };
        let (actual, _) = hc_mix(&backend, &cfg, &residual, &hc).unwrap();
        let norm = [(2.5 + cfg.rms_norm_eps).sqrt(), (12.5 + cfg.rms_norm_eps).sqrt()];
        let xn = [1.0 / norm[0], 2.0 / norm[0], -3.0 / norm[1], 4.0 / norm[1]];
        let scaled = xn[0] / 2.0;
        let lo = scaled / (1.0 + (-scaled).exp());
        for column in 0..2 {
            let coefficients = [1.0, -1.0, 2.0, -2.0];
            let expected = (0..2)
                .map(|stream| {
                    let index = stream * 2 + column;
                    xn[index] / (1.0 + (-coefficients[index] * lo).exp())
                })
                .sum::<f32>()
                / 2.0;
            assert!((actual.data[column] - expected).abs() < 1e-6);
        }
    }
}
