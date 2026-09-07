//! 完整文本模型 CUDA 组合:packed 专家驻留主存,显存 LRU 按需上传。
//! 首版限制在 QSA 全可见的上下文区间,超过索引器预算明确拒绝。

use super::{Qwen4ExpConfig, Qwen4ExpGguf, Qwen4ExpHyperConnection, Qwen4ExpLayer, Qwen4ExpMixer, Qwen4ExpPle};
use crate::{
    attention::{
        gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetState},
        hybrid::{HybridAttention, HybridAttentionOptions},
        rope::{RopeTable, RotaryLayout},
    },
    backend::{
        Backend, GqaPrefillBackend,
        cuda::{CudaContext, CudaGatedDeltaNetStorage, CudaKvCache, CudaPrefillExperts, CudaTensor, CudaWeight},
    },
    kernel::cuda::grouped,
    moe::{
        expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
        prefill::prefill_experts_untraced,
        topk_moe::{MoeFfnRef, SharedExpertRef},
    },
    runtime::expert_pipeline::{ExpertDecodePipeline, ExpertDecodeRequest},
    weight::expert_source::{ExpertSourceProvider, GgufExpertSource},
};
use cudarc::driver::safe::CudaSlice;
use std::{path::Path, sync::Arc, time::Instant};

#[derive(Clone)]
pub struct CudaOptions {
    pub max_seq_len: usize,
    pub decode_steps: usize,
    pub prefill_chunk_size: usize,
    pub expert_cache_bytes: usize,
    pub expert_prefetch_count: usize,
    /// 分段同步诊断,不用于吞吐结果。
    pub profile: bool,
    pub pin_experts: bool,
    pub mtp_weights: Option<std::path::PathBuf>,
    pub mtp_steps: usize,
    pub mtp_cache_bytes: usize,
    pub mtp_min_confidence: f32,
    pub frequency_cache: bool,
    pub expert_transfer_group: usize,
    /// Q8G64 KV(64 元素组 i8 码 + f32 scale);false 走 F16 KV。
    pub q8_kv: bool,
}

fn profile_stage(ctx: &CudaContext, enabled: bool, start: &mut Instant, total: &mut f64) -> Result<(), crate::backend::BackendError> {
    if enabled {
        ctx.synchronize().map_err(|error| crate::backend::compute_error(format!("CUDA profile 同步: {error:?}")))?;
        *total += start.elapsed().as_secs_f64();
        *start = Instant::now();
    }
    Ok(())
}

pub(super) fn hc_mix(ctx: &CudaContext, cfg: &Qwen4ExpConfig, residual: &CudaTensor, weights: &Qwen4ExpHyperConnection<CudaWeight>) -> Result<(CudaTensor, Option<CudaTensor>), crate::backend::BackendError> {
    let op = crate::backend::compute_error;
    let groups = cfg.hyper_connection.streams;
    let normalized = grouped::norm(ctx, residual, &weights.norm, groups, cfg.rms_norm_eps).map_err(op)?;
    let low = ctx.linear(&normalized, &weights.down)?;
    let low = grouped::scaled_silu(ctx, &low, 1.0 / groups as f32).map_err(op)?;
    let gate = ctx.linear(&low, &weights.up)?;
    let mixed = grouped::sigmoid_mean(ctx, &normalized, &gate, groups).map_err(op)?;
    let inject = weights.inject.as_ref().map(|weight| ctx.linear(&normalized, weight)).transpose()?;
    Ok((mixed, inject))
}

pub(super) fn selected_rows(ctx: &CudaContext, source: &Qwen4ExpGguf, name: &str, ids: &[u32], rows: usize, columns: usize) -> Result<CudaTensor, crate::backend::BackendError> {
    let op = crate::backend::compute_error;
    let matrix = source.reader().read_matrix(name).map_err(op)?;
    static CALL_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if std::env::var_os("ZLLM_Q4E_AUDIT").is_some() {
        eprintln!("[selected_rows #{count}] name={name} ids={} rows={rows} cols={columns} gguf_cols={} type={}", ids.len(), matrix.columns, matrix.tensor_type.0);
    }
    // packed 按 GGUF 矩阵实际列宽分配;调用方 columns 与 GGUF columns 不一致
    // 时 read_rows_into 会越界写穿 Vec(堆损坏根因),显式拒绝。
    if matrix.columns != columns {
        return Err(op(format!("selected_rows {name}: GGUF columns={} 与调用方 columns={columns} 不一致", matrix.columns)));
    }
    let mut packed = vec![0u8; matrix.tensor_type.storage_bytes(matrix.columns).map_err(op)? * ids.len()];
    matrix.read_rows_into(ids, &mut packed).map_err(op)?;
    crate::kernel::cuda::linear::gguf_rows_f16(ctx, &packed, rows, columns, matrix.tensor_type.0).map_err(op)
}

fn ple(
    ctx: &CudaContext,
    cfg: &Qwen4ExpConfig,
    gguf: &Qwen4ExpGguf,
    weights: &Qwen4ExpPle<CudaWeight>,
    state: &mut CudaSlice<f32>,
    checkpoints: Option<&CudaSlice<f32>>,
    residual: &CudaTensor,
    tokens: &[u32],
    prev: &[u32],
) -> Result<CudaTensor, crate::backend::BackendError> {
    let op = crate::backend::compute_error;
    let ple = cfg.ple.as_ref().expect("PLE 权重与配置同时存在");
    let indices = super::ple_row_indices(ple, tokens, prev);
    // PLE 表列宽是 head_dim(160),不是 hidden_size(2560);走 GGUF 的
    // 专用行 gather(按 head 拼接),不能用 selected_rows(按整行 gather)。
    let ple_rows = gguf.ple_rows_f32(ple, &indices).map_err(op)?;
    let packed: Vec<half::f16> = ple_rows.iter().map(|&v| half::f16::from_f32(v)).collect();
    let slice = ctx.stream().clone_htod(&packed).map_err(|e| op(format!("PLE 行上传: {e:?}")))?;
    let input = CudaTensor::new(slice, tokens.len(), cfg.hidden_size);
    let key = ctx.linear(&input, &weights.key)?;
    let value = ctx.linear(&input, &weights.value)?;
    let groups = cfg.hyper_connection.streams;
    let key = grouped::norm(ctx, &key, &weights.norm_key, groups, cfg.rms_norm_eps).map_err(op)?;
    let query = grouped::norm(ctx, residual, &weights.norm_query, groups, cfg.rms_norm_eps).map_err(op)?;
    let gated = grouped::signed_sqrt_gate(ctx, &key, &query, &value, groups).map_err(op)?;
    let normalized = grouped::norm(ctx, &gated, &weights.norm_conv, groups, cfg.rms_norm_eps).map_err(op)?;
    grouped::dilated_conv_residual(ctx, residual, &gated, &normalized, &weights.conv1d, state, ple.ngram_size, checkpoints).map_err(op)
}

/// QSA 稀疏注意力(语义源:cpu.rs reference)。逐步:
/// 投影(主 GQA + 索引器)→ raw 索引 K 追加 → 新满块池化(norm+RoPE)→
/// 逐行打分选择 top (top_k+r-1) cell → 掩码 GQA(Q8G64/F16 按缓存形态)。
/// QSA 全可见区间(短序列)与稠密 GQA 精确等价,无需分支。
#[allow(clippy::too_many_arguments)]
pub(super) fn sparse_attention(
    ctx: &CudaContext,
    cfg: &Qwen4ExpConfig,
    weights: &crate::attention::hybrid::FullAttentionWeights<CudaWeight>,
    indexer: &super::Qwen4ExpIndexer<CudaWeight>,
    cache: &mut CudaKvCache,
    qsa: &mut QsaState,
    layer: usize,
    rope: &RopeTable,
    input: &CudaTensor,
    position: usize,
) -> Result<CudaTensor, crate::backend::BackendError> {
    let op = crate::backend::compute_error;
    let rows = input.rows;
    let query_gate = ctx.linear(input, &weights.query_gate)?;
    let (query, gate) = ctx.split_interleaved_columns(&query_gate, cfg.head_dim)?;
    let (key, value) = ctx.dual_linear(input, &weights.key, &weights.value)?;
    let query = ctx.rmsnorm_heads(&query, &weights.query_norm, cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps)?;
    let key = ctx.rmsnorm_heads(&key, &weights.key_norm, cfg.num_kv_heads, cfg.head_dim, cfg.rms_norm_eps)?;
    let query = ctx.rope_prefix(&query, cfg.num_attention_heads, cfg.rope_dim, RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
    let key = ctx.rope_prefix(&key, cfg.num_kv_heads, cfg.rope_dim, RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
    // 索引器:q 走 norm+rope;k 存 raw(池化发生在 norm/rope 之前)
    let index_key_raw = ctx.linear(input, &indexer.k_proj)?;
    let index_query = ctx.linear(input, &indexer.q_proj)?;
    let index_query = ctx.rmsnorm_heads(&index_query, &indexer.q_norm, cfg.indexer.head_count, cfg.indexer.head_dim, cfg.rms_norm_eps)?;
    let index_query = ctx.rope_prefix(&index_query, cfg.indexer.head_count, cfg.rope_dim, RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;

    // 主 KV 追加(内部按 Q8G64/F16 量化);raw 索引 K 追加到层缓冲偏移处
    cache.append(ctx, layer, position, &key, &value)?;
    let qsa_layer = &mut qsa.layers[layer];
    let head_dim = cfg.indexer.head_dim;
    let ratio = cfg.indexer.compress_ratio;
    ctx.copy_dtod_at(&index_key_raw.slice, &mut qsa_layer.raw.slice, qsa_layer.raw_rows * head_dim).map_err(op)?;
    let n_cells = qsa_layer.raw_rows + rows;
    qsa_layer.raw_rows = n_cells;
    // 新满块池化:位置 = first_block..n_full(块起点 RoPE 表行由宿主收集上传)
    let n_full = n_cells / ratio;
    let new_blocks = n_full.saturating_sub(qsa_layer.pooled_blocks);
    if new_blocks > 0 {
        let first_block = qsa_layer.pooled_blocks;
        let half = cfg.rope_dim / 2;
        let mut cos = Vec::with_capacity(new_blocks * half);
        let mut sin = Vec::with_capacity(new_blocks * half);
        for block in first_block..n_full {
            let base = block * ratio * half;
            cos.extend_from_slice(&rope.cos[base..base + half]);
            sin.extend_from_slice(&rope.sin[base..base + half]);
        }
        let mut cos_gpu = ctx.stream().alloc_zeros::<f32>(cos.len()).map_err(|e| op(format!("cos 分配: {e:?}")))?;
        let mut sin_gpu = ctx.stream().alloc_zeros::<f32>(sin.len()).map_err(|e| op(format!("sin 分配: {e:?}")))?;
        ctx.upload_f32_pinned(&cos, &mut cos_gpu).map_err(|e| op(format!("cos 上传: {e:?}")))?;
        ctx.upload_f32_pinned(&sin, &mut sin_gpu).map_err(|e| op(format!("sin 上传: {e:?}")))?;
        crate::kernel::cuda::qsa::qsa_pool_blocks(ctx, &qsa_layer.raw.slice, &mut qsa_layer.pooled, &indexer.k_norm.data, &cos_gpu, &sin_gpu, first_block, new_blocks, head_dim, ratio, cfg.rms_norm_eps).map_err(op)?;
        qsa_layer.pooled_blocks = n_full;
    }
    // 逐行选择:输出压缩索引表(rows × width)+ 计数 + 块分数暂存
    let width = n_cells.min(cfg.indexer.top_k + ratio - 1);
    let n_blocks = n_cells.div_ceil(ratio);
    let dead_block = if n_full < n_blocks { n_full } else { n_blocks - 1 };
    let mut block_scores = ctx.buffer_uninit_f32(rows * n_blocks).map_err(op)?;
    let mut selected = ctx.stream().alloc_zeros::<u32>(rows * width).map_err(|e| op(format!("selected 分配: {e:?}")))?;
    let mut selected_count = ctx.stream().alloc_zeros::<u32>(rows).map_err(|e| op(format!("count 分配: {e:?}")))?;
    crate::kernel::cuda::qsa::qsa_score_select(ctx, &qsa_layer.pooled, &index_query, &mut block_scores, &mut selected, &mut selected_count, rows, n_cells, dead_block, position, ratio, cfg.indexer.top_k, cfg.indexer.head_count, head_dim, width).map_err(op)?;
    // 掩码 GQA:按缓存形态分流(Q8G64 / F16)
    let kv_layer = cache.layer(layer)?;
    let attended = if let Some(q8) = kv_layer.q8.as_ref() {
        crate::kernel::cuda::qsa::qsa_masked_attention_q8g64(ctx, &query, &q8.key_codes, &q8.key_scales, &q8.value_codes, &q8.value_scales, &selected, &selected_count, &cfg.qsa_spec(), width).map_err(op)?
    } else {
        crate::kernel::cuda::qsa::qsa_masked_attention_f16(ctx, &query, &kv_layer.key, &kv_layer.value, &selected, &selected_count, &cfg.qsa_spec(), width).map_err(op)?
    };
    let gated = ctx.sigmoid_gate(&attended, &gate)?;
    ctx.linear(&gated, &weights.output)
}

/// QSA 预算内所有可见 cell 必然入选,这一区间与因果 GQA 精确等价。
pub(super) fn full_attention(
    ctx: &CudaContext,
    cfg: &Qwen4ExpConfig,
    weights: &crate::attention::hybrid::FullAttentionWeights<CudaWeight>,
    cache: &mut CudaKvCache,
    rope: &RopeTable,
    layer: usize,
    input: &CudaTensor,
    position: usize,
) -> Result<CudaTensor, crate::backend::BackendError> {
    let query_gate = ctx.linear(input, &weights.query_gate)?;
    let (query, gate) = ctx.split_interleaved_columns(&query_gate, cfg.head_dim)?;
    let (key, value) = ctx.dual_linear(input, &weights.key, &weights.value)?;
    let query = ctx.rmsnorm_heads(&query, &weights.query_norm, cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps)?;
    let key = ctx.rmsnorm_heads(&key, &weights.key_norm, cfg.num_kv_heads, cfg.head_dim, cfg.rms_norm_eps)?;
    let query = ctx.rope_prefix(&query, cfg.num_attention_heads, cfg.rope_dim, RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
    let key = ctx.rope_prefix(&key, cfg.num_kv_heads, cfg.rope_dim, RotaryLayout::SplitHalf, position, &rope.cos, &rope.sin)?;
    let attended = ctx.gqa_prefill_attention_cached(cache, layer, position, &query, &key, &value, &cfg.qsa_spec(), false)?;
    let gated = ctx.sigmoid_gate(&attended, &gate)?;
    ctx.linear(&gated, &weights.output)
}

pub(super) struct CudaState {
    delta: GatedDeltaNetState<CudaGatedDeltaNetStorage>,
    cache: CudaKvCache,
    ple_state: CudaSlice<f32>,
    ple_checkpoints: Option<CudaSlice<f32>>,
    previous: Vec<u32>,
    qsa: QsaState,
}

/// 每层原始索引 K(F32)与池化块缓存。raw 覆写追加;pooled 块只增写,
/// truncate 只回退游标(块内容依赖的 token 截断后重池化由下次追加覆盖)。
struct QsaState {
    layers: Vec<QsaLayer>,
}

struct QsaLayer {
    raw: CudaTensor,
    raw_rows: usize,
    pooled: CudaTensor,
    pooled_blocks: usize,
}

impl CudaState {
    pub(super) fn new(ctx: &CudaContext, cfg: &Qwen4ExpConfig, max_seq_len: usize, q8_kv: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let columns = cfg.num_kv_heads * cfg.head_dim;
        let cache = if q8_kv { CudaKvCache::new_q8g64(cfg.num_layers, max_seq_len, columns)? } else { CudaKvCache::new(cfg.num_layers, max_seq_len, columns) };
        let layers = (0..cfg.num_layers)
            .map(|layer| {
                if !cfg.is_full_attention(layer) {
                    let placeholder = ctx.placeholder_f16().map_err(crate::backend::compute_error)?;
                    let empty = CudaTensor { slice: placeholder.clone(), slice_f32: None, rows: 0, cols: 0 };
                    return Ok(QsaLayer { raw: empty.clone(), raw_rows: 0, pooled: empty, pooled_blocks: 0 });
                }
                let head_dim = cfg.indexer.head_dim;
                let raw = ctx.tensor_zeros(max_seq_len, head_dim).map_err(crate::backend::compute_error)?;
                let blocks = max_seq_len / cfg.indexer.compress_ratio;
                let pooled = ctx.tensor_zeros(blocks, head_dim).map_err(crate::backend::compute_error)?;
                Ok(QsaLayer { raw, raw_rows: 0, pooled, pooled_blocks: 0 })
            })
            .collect::<Result<Vec<_>, crate::backend::BackendError>>()?;
        Ok(Self {
            delta: GatedDeltaNetState::with_head_layout(cfg.num_layers, cfg.gated_delta_net_spec(), GatedDeltaNetHeadLayout::Tiled)?,
            cache,
            ple_state: ctx.stream().alloc_zeros::<f32>(cfg.ple.as_ref().map_or(0, |ple| ple.conv_history() * cfg.hc_dim()))?,
            ple_checkpoints: None,
            previous: Vec::new(),
            qsa: QsaState { layers },
        })
    }

    pub(super) fn enable_checkpoints(&mut self, ctx: &CudaContext, rows: usize) -> Result<(), String> {
        for layer in 0..self.delta.layer_count() {
            if let Some(storage) = self.delta.layer_storage_mut(layer) {
                storage.enable_checkpoints(ctx, rows)?;
            }
        }
        self.ple_checkpoints = Some(ctx.buffer_uninit(rows.checked_mul(self.ple_state.len()).ok_or("PLE checkpoint 大小溢出")?)?);
        Ok(())
    }

    pub(super) fn retain_prefix(&mut self, ctx: &CudaContext, position: usize, retained: usize) -> Result<(), Box<dyn std::error::Error>> {
        if retained == 0 {
            return Err("target verify 必须至少保留 anchor".into());
        }
        for layer in 0..self.delta.layer_count() {
            if let Some(storage) = self.delta.layer_storage_mut(layer) {
                storage.restore_checkpoint(ctx, retained - 1)?;
                self.delta.rewind_layer(layer, position + retained)?;
            }
        }
        let checkpoints = self.ple_checkpoints.as_ref().ok_or("PLE checkpoints 尚未启用")?;
        let count = self.ple_state.len();
        ctx.stream().memcpy_dtod(&checkpoints.slice((retained - 1) * count..retained * count), &mut self.ple_state)?;
        self.cache.truncate(position + retained);
        self.previous.truncate(position + retained);
        // QSA 游标同步回退:raw 截到保留边界,块游标重算;截断边界块的内容
        // 由下次追加的池化 kernel 按 first_block 覆写。
        let boundary = position + retained;
        for layer in self.qsa.layers.iter_mut() {
            if layer.raw_rows > boundary {
                layer.raw_rows = boundary;
            }
            let blocks = layer.raw_rows / 4;
            if layer.pooled_blocks > blocks {
                layer.pooled_blocks = blocks;
            }
        }
        Ok(())
    }

    pub(super) fn forward(
        &mut self,
        ctx: &CudaContext,
        cfg: &Qwen4ExpConfig,
        source: &Qwen4ExpGguf,
        layers: &[Qwen4ExpLayer<CudaWeight>],
        rope: &RopeTable,
        profile: bool,
        chunk: &[u32],
        position: usize,
        mlp: &mut dyn FnMut(usize, &CudaTensor) -> Result<CudaTensor, crate::backend::BackendError>,
    ) -> Result<CudaTensor, Box<dyn std::error::Error>> {
        let attention = HybridAttention::new(ctx, cfg.qsa_spec(), cfg.gated_delta_net_spec(), cfg.rms_norm_eps, rope, HybridAttentionOptions::default());
        let mut stage_wall = [0.0; 6];
        let mut stage_start = Instant::now();
        let embedding = selected_rows(ctx, source, "token_embd.weight", chunk, chunk.len(), cfg.hidden_size)?;
        let mut residual = grouped::repeat_groups(ctx, &embedding, cfg.hyper_connection.streams)?;
        for (layer, weights) in layers.iter().enumerate() {
            if let Some(weights) = &weights.ple {
                residual = ple(ctx, cfg, source, weights, &mut self.ple_state, self.ple_checkpoints.as_ref(), &residual, chunk, &self.previous)?;
            }
            profile_stage(ctx, profile, &mut stage_start, &mut stage_wall[0])?;
            let (mixed, inject) = hc_mix(ctx, cfg, &residual, &weights.hc_attn)?;
            profile_stage(ctx, profile, &mut stage_start, &mut stage_wall[1])?;
            let mixed = match &weights.mixer {
                Qwen4ExpMixer::Delta(weights) => attention.delta(weights, &mut self.delta, layer, &mixed, position)?,
                Qwen4ExpMixer::SparseAttention { attention, indexer } => {
                    // 短上下文走 dense(QSA 全可见区间与因果 GQA 精确等价):
                    // sparse kernel 的 Q8 掩码路径存在形状相关越界嫌疑
                    // (MTP node 二请求 token probability 崩溃的排除法分支)。
                    let n_cells = position + chunk.len();
                    if n_cells <= cfg.indexer.top_k {
                        full_attention(ctx, cfg, attention, &mut self.cache, &rope, layer, &mixed, position)?
                    } else {
                        sparse_attention(ctx, cfg, attention, indexer, &mut self.cache, &mut self.qsa, layer, &rope, &mixed, position)?
                    }
                }
            };
            profile_stage(ctx, profile, &mut stage_start, &mut stage_wall[2])?;
            residual = grouped::sigmoid_residual(ctx, &residual, &mixed, inject.as_ref().unwrap())?;
            let (mixed, inject) = hc_mix(ctx, cfg, &residual, &weights.hc_ffn)?;
            profile_stage(ctx, profile, &mut stage_start, &mut stage_wall[3])?;
            let mixed = mlp(layer, &mixed)?;
            profile_stage(ctx, profile, &mut stage_start, &mut stage_wall[4])?;
            residual = grouped::sigmoid_residual(ctx, &residual, &mixed, inject.as_ref().unwrap())?;
            profile_stage(ctx, profile, &mut stage_start, &mut stage_wall[5])?;
        }
        if profile {
            eprintln!(
                "[qwen4exp-profile-forward] position={position} rows={} ple={:.6} hc_attn={:.6} attention={:.6} hc_ffn={:.6} moe={:.6} combine={:.6}",
                chunk.len(),
                stage_wall[0],
                stage_wall[1],
                stage_wall[2],
                stage_wall[3],
                stage_wall[4],
                stage_wall[5]
            );
        }
        self.previous.extend_from_slice(chunk);
        Ok(residual)
    }
}

pub fn run(path: &Path, prompt: &str, options: CudaOptions) -> Result<(), Box<dyn std::error::Error>> {
    let mut source = Qwen4ExpGguf::open(path)?;
    let cfg = source.config().clone();
    if options.expert_transfer_group > cfg.num_experts_per_tok || (options.expert_transfer_group > 0 && (!options.pin_experts || options.profile)) {
        return Err("专家分组上传要求 pinned 驻留、关闭分段计时,组大小不超过 top_k".into());
    }
    let tokens = source.tokenizer()?.tokenize(prompt.as_bytes());
    if tokens.is_empty() || tokens.len().saturating_add(options.decode_steps) > options.max_seq_len || options.prefill_chunk_size == 0 {
        return Err(format!(
            "Qwen4-Exp CUDA 要求 0 < prompt+decode <= max_seq_len,实际 prompt={} decode={} max_seq_len={} chunk={}",
            tokens.len(),
            options.decode_steps,
            options.max_seq_len,
            options.prefill_chunk_size
        )
        .into());
    }
    let host_bytes: usize = source.reader().tensors().iter().filter(|t| t.name.contains("_exps.weight")).map(|t| t.bytes).sum();
    // 启动时拒绝任何会落入 F16 专家展开的格式,使这条路线始终保持原始 packed 编码。
    for tensor in source.reader().tensors().iter().filter(|t| t.name.contains("_exps.weight")) {
        let supported = if tensor.name.contains("ffn_down_exps") { matches!(tensor.tensor_type.0, 7 | 8) } else { matches!(tensor.tensor_type.0, 12 | 13) };
        if !supported {
            return Err(format!("Qwen4-Exp CUDA 原生专家路径不支持 {} type={}", tensor.name, tensor.tensor_type.0).into());
        }
    }
    if std::env::var_os("ZLLM_CUDA_F16_EXPERTS").is_some() {
        return Err("Qwen4-Exp 原生专家路径不能启用旧 F16 展开诊断开关".into());
    }
    #[cfg(target_os = "linux")]
    {
        let mem = std::fs::read_to_string("/proc/meminfo")?;
        let available = mem.lines().find_map(|line| line.strip_prefix("MemAvailable:").and_then(|value| value.split_whitespace().next()).and_then(|value| value.parse::<usize>().ok())).ok_or("无法读取 MemAvailable")? * 1024;
        if available < host_bytes + 8 * 1024 * 1024 * 1024 {
            // 警告而非拒绝:上一个 pinned 进程退出后 cuMemFreeHost 把页还给
            // 驱动 host 池而非 OS,MemAvailable 长期偏低但新 cuMemAllocHost
            // 可复用池页(2026-09-07 实测)。真实不足由逐矩阵分配的
            // CUDA OOM 错误兜底中止。
            eprintln!("[qwen4exp-meminfo-warn] 专家驻留 {} bytes,MemAvailable 仅 {} bytes(可能为驱动 pinned 池保留,继续加载)", host_bytes, available);
        }
    }
    let started = Instant::now();
    let ctx = CudaContext::new_default()?;
    if options.pin_experts {
        // 驱动自有 pinned 驻留(与 node 同路径);MTP 草稿专家保持堆驻留。
        let bytes = source.make_experts_resident_extern(&mut |len| ctx.alloc_pinned_source_bytes(len))?;
        for backing in source.resident_extern_backings() {
            ctx.retain_pinned_source(&backing)?;
        }
        eprintln!("[qwen4exp-host-pinned] bytes={bytes}");
    } else {
        source.make_experts_resident()?;
    }
    let source = Arc::new(source);
    eprintln!("[qwen4exp-resident-ready] bytes={host_bytes} wall={:.3}s", started.elapsed().as_secs_f64());
    let mut mtp_source = options.mtp_weights.as_ref().map(|path| super::cuda_mtp::MtpSource::open(path, &cfg).map(Arc::new)).transpose()?;
    if options.pin_experts && let Some(source) = mtp_source.as_mut().and_then(Arc::get_mut) {
        // 草稿专家同样走驱动 pinned 驻留:草稿 forward 的 miss 上传
        // DMA 直读,免槽环 memcpy(draft_wall 主要构成)。
        let bytes = source.make_experts_resident_extern(&mut |len| ctx.alloc_pinned_source_bytes(len))?;
        for backing in source.resident_extern_backings() {
            ctx.retain_pinned_source(&backing)?;
        }
        eprintln!("[qwen4exp-mtp-pinned] bytes={bytes}");
    }
    if mtp_source.is_some() && (options.mtp_steps == 0 || options.mtp_steps > 8 || !options.mtp_min_confidence.is_finite() || !(0.0..=1.0).contains(&options.mtp_min_confidence)) {
        return Err("MTP 草稿长度要求 1..=8, confidence 要求 0..=1".into());
    }
    let layers = super::prepare_qwen4exp_layers(&ctx, &source)?;
    let output_hc = Qwen4ExpHyperConnection {
        norm: crate::runtime::prepare_gguf_f32_vector(&ctx, source.reader(), "output_hc_norm.weight")?,
        down: crate::runtime::prepare_gguf_matrix(&ctx, source.reader(), "output_hc_down.weight")?,
        up: crate::runtime::prepare_gguf_matrix(&ctx, source.reader(), "output_hc_up.weight")?,
        inject: None,
    };
    let lm_head = crate::runtime::prepare_gguf_matrix(&ctx, source.reader(), "output.weight")?;
    let mut mtp = mtp_source.map(|source| super::cuda_mtp::Mtp::new(&ctx, source, options.max_seq_len, options.mtp_cache_bytes)).transpose()?;
    let rope = RopeTable::precompute(options.max_seq_len, cfg.rope_dim, cfg.rope_theta);
    ctx.rope_window_f16(&rope.cos)?;
    ctx.rope_window_f16(&rope.sin)?;
    let mut state = CudaState::new(&ctx, &cfg, options.max_seq_len, options.q8_kv)?;
    let expert_source: Arc<dyn GgufExpertSource> = source.clone();
    let mut prefill_experts = CudaPrefillExperts::gguf(expert_source, options.expert_cache_bytes);
    prefill_experts.reserve_arena(&ctx)?;
    prefill_experts.set_frequency_cache(options.frequency_cache);
    prefill_experts.set_transfer_group(options.expert_transfer_group);
    ctx.synchronize()?;
    eprintln!("[qwen4exp-cuda-ready] layers={} host_bytes={host_bytes} vram_free={} cache_budget={} load_wall={:.3}s", layers.len(), ctx.device().mem_get_info()?.0, options.expert_cache_bytes, started.elapsed().as_secs_f64());
    let spec = cfg.moe_spec();
    fn moe_ref(weights: &Qwen4ExpLayer<CudaWeight>) -> SharedExpertRef<'_, CudaWeight> {
        SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }
    }
    let prefill_started = Instant::now();
    let mut hidden = None;
    let mut mtp_previous = if mtp.is_some() { Some(CudaTensor::new_f32_residual(ctx.stream().alloc_zeros::<f32>(cfg.hc_dim())?, ctx.placeholder_f16()?, 1, cfg.hc_dim())) } else { None };
    for (chunk_index, chunk) in tokens.chunks(options.prefill_chunk_size).enumerate() {
        hidden = Some(state.forward(&ctx, &cfg, &source, &layers, &rope, options.profile, chunk, chunk_index * options.prefill_chunk_size, &mut |layer, input| {
            let weights = &layers[layer];
            let shared = [moe_ref(weights)];
            let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
            prefill_experts_untraced(&ctx, &spec, &reference, layer, &mut prefill_experts, input, None)
        })?);
        if let Some(mtp) = &mut mtp {
            for (row, &token) in chunk.iter().enumerate() {
                mtp.forward(&ctx, &source, &rope, mtp_previous.as_ref().unwrap(), token, chunk_index * options.prefill_chunk_size + row)?;
                mtp_previous = Some(ctx.select_row(hidden.as_ref().unwrap(), row)?);
            }
        }
    }
    ctx.synchronize()?;
    let prefill_wall = prefill_started.elapsed().as_secs_f64();
    eprintln!("[qwen4exp-cuda-prefill] tokens={} wall={prefill_wall:.6}s tok_per_s={:.3}", tokens.len(), tokens.len() as f64 / prefill_wall);
    let hidden = hidden.unwrap();
    let mut hidden = ctx.select_row(&hidden, hidden.rows - 1)?;
    let mut expert_state = ExpertDecodePipeline::new(
        prefill_experts.into_decode_state(),
        ExpertPredictorConfig { first_layer: 0, layer_count: cfg.num_layers, expert_count: cfg.num_experts, routed_top_k: cfg.num_experts_per_tok, prefetch_count: options.expert_prefetch_count, weights: ExpertPredictorWeights::default() },
    )?;
    expert_state.backend_state_mut().profile = options.profile;
    if let Some(mtp) = mtp {
        return run_mtp_decode(&ctx, &cfg, &source, &layers, &rope, &output_hc, &lm_head, &options, state, expert_state, mtp, hidden, tokens.len(), prefill_wall);
    }
    let detokenizer = source.detokenizer()?;
    let decode_started = Instant::now();
    let mut first_token_wall = 0.0;
    let mut generated = 0;
    for step in 0..options.decode_steps.max(1) {
        let head_start = Instant::now();
        let (mixed, _) = hc_mix(&ctx, &cfg, &hidden, &output_hc)?;
        let logits = ctx.linear(&mixed, &lm_head)?;
        let token = ctx.argmax(&logits)?;
        if options.profile {
            let state = expert_state.backend_state_mut();
            eprintln!("[qwen4exp-profile-head] step={step} wall={:.6} expert_upload_cumulative={:.6} expert_compute_cumulative={:.6}", head_start.elapsed().as_secs_f64(), state.upload_wall, state.compute_wall);
        }
        if step == 0 {
            first_token_wall = decode_started.elapsed().as_secs_f64();
        }
        let expert_stats = expert_state.backend_state_mut();
        eprintln!("[qwen4exp-cuda-token] step={step} id={token} elapsed={:.6}s loaded={} uploaded_bytes={}", decode_started.elapsed().as_secs_f64(), expert_stats.loaded, expert_stats.uploaded_bytes);
        if options.decode_steps == 0 {
            break;
        }
        crate::runtime::generation::write_token(&detokenizer, token, true)?;
        generated += 1;
        if cfg.eos_token_ids.contains(&token) || generated == options.decode_steps {
            break;
        }
        let position = tokens.len() + step;
        hidden = state.forward(&ctx, &cfg, &source, &layers, &rope, options.profile, &[token], position, &mut |layer, input| {
            let weights = &layers[layer];
            let shared = [moe_ref(weights)];
            let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
            let next = (layer + 1 < cfg.num_layers).then(|| source.source(layer + 1).map(|source| (layer + 1, source))).transpose().map_err(crate::backend::BackendError::ExpertLoad)?;
            expert_state.decode(&ctx, &spec, &reference, ExpertDecodeRequest { layer, source: source.source(layer).map_err(crate::backend::BackendError::ExpertLoad)?, position, next }, input)
        })?;
    }
    ctx.synchronize()?;
    let decode_wall = decode_started.elapsed().as_secs_f64() - first_token_wall;
    eprintln!(
        "[qwen4exp-cuda-summary] prompt_tokens={} generated={generated} ttft={:.6}s decode_rounds={} decode_wall={decode_wall:.6}s decode_tok_per_s={:.3} cache_bytes={}",
        tokens.len(),
        prefill_wall + first_token_wall,
        generated.saturating_sub(1),
        generated.saturating_sub(1) as f64 / decode_wall.max(f64::EPSILON),
        expert_state.backend_state_mut().resident_bytes()
    );
    Ok(())
}

fn run_mtp_decode(
    ctx: &CudaContext,
    cfg: &Qwen4ExpConfig,
    source: &Arc<Qwen4ExpGguf>,
    layers: &[Qwen4ExpLayer<CudaWeight>],
    rope: &RopeTable,
    output_hc: &Qwen4ExpHyperConnection<CudaWeight>,
    lm_head: &CudaWeight,
    options: &CudaOptions,
    mut state: CudaState,
    mut experts: ExpertDecodePipeline<crate::backend::cuda::CudaMoeState>,
    mut mtp: super::cuda_mtp::Mtp,
    mut hidden: CudaTensor,
    prompt_tokens: usize,
    prefill_wall: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let detokenizer = source.detokenizer()?;
    let started = Instant::now();
    state.enable_checkpoints(ctx, options.mtp_steps + 1)?;
    let target_sample = |hidden: &CudaTensor| -> Result<u32, Box<dyn std::error::Error>> {
        let (mixed, _) = hc_mix(ctx, cfg, hidden, output_hc)?;
        Ok(ctx.argmax(&ctx.linear(&mixed, lm_head)?)?)
    };
    let mut anchor = target_sample(&hidden)?;
    let first_token_wall = started.elapsed().as_secs_f64();
    eprintln!("[qwen4exp-cuda-token] step=0 id={anchor} elapsed={first_token_wall:.6}s");
    let mut generated = 0;
    if options.decode_steps > 0 {
        crate::runtime::generation::write_token(&detokenizer, anchor, true)?;
        generated = 1;
    }
    let mut position = prompt_tokens;
    let initial_uploaded_bytes = experts.backend_state_mut().uploaded_bytes;
    let initial_loaded = experts.backend_state_mut().loaded;
    let mut proposed = 0;
    let mut accepted = 0;
    let mut rounds = 0;
    let mut draft_wall = 0.0;
    let mut verify_wall = 0.0;
    let mut catchup_wall = 0.0;
    let mut verify_experts = CudaPrefillExperts::gguf(source.clone(), options.expert_cache_bytes);
    let spec = cfg.moe_spec();
    while generated < options.decode_steps && !cfg.eos_token_ids.contains(&anchor) {
        let draft_start = Instant::now();
        let count = options.mtp_steps.min(options.decode_steps - generated - 1);
        let mut candidates = Vec::with_capacity(count);
        let mut draft_hidden = ctx.select_row(&hidden, 0)?;
        let mut token = anchor;
        mtp.truncate(position);
        for step in 0..count {
            draft_hidden = mtp.forward(ctx, source, rope, &draft_hidden, token, position + step)?;
            let (candidate, confidence) = mtp.sample(ctx, &draft_hidden, lm_head)?;
            eprintln!("[qwen4exp-mtp-draft] position={} id={candidate} confidence={confidence:.5}", position + step);
            if confidence < options.mtp_min_confidence {
                break;
            }
            token = candidate;
            candidates.push(token);
            if cfg.eos_token_ids.contains(&token) {
                break;
            }
        }
        let draft_elapsed = draft_start.elapsed().as_secs_f64();
        draft_wall += draft_elapsed;
        proposed += candidates.len();
        let mut inputs = vec![anchor];
        inputs.extend_from_slice(&candidates);
        let verify_start = Instant::now();
        let verified = if inputs.len() == 1 {
            state.forward(ctx, cfg, source, layers, rope, options.profile, &inputs, position, &mut |layer, input| {
                let weights = &layers[layer];
                let shared = [SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }];
                let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
                let next = (layer + 1 < cfg.num_layers).then(|| source.source(layer + 1).map(|source| (layer + 1, source))).transpose().map_err(crate::backend::BackendError::ExpertLoad)?;
                experts.decode(ctx, &spec, &reference, ExpertDecodeRequest { layer, source: source.source(layer).map_err(crate::backend::BackendError::ExpertLoad)?, position, next }, input)
            })?
        } else {
            // prefill/verify 与 decode 共享同一显存专家缓存,阶段切换不重新上传热专家。
            verify_experts.swap_decode_state(experts.backend_state_mut());
            let result = state.forward(ctx, cfg, source, layers, rope, options.profile, &inputs, position, &mut |layer, input| {
                let weights = &layers[layer];
                let shared = [SharedExpertRef { gate: &weights.moe.shared_gate, up: &weights.moe.shared_up, down: &weights.moe.shared_down, output_gate: Some(&weights.moe.shared_output_gate) }];
                let reference = MoeFfnRef { router_weight: &weights.moe.router, router_bias: &weights.moe.router_bias, shared_experts: &shared, selected_experts: None };
                prefill_experts_untraced(ctx, &spec, &reference, layer, &mut verify_experts, input, None)
            });
            verify_experts.swap_decode_state(experts.backend_state_mut());
            result?
        };
        // 验证的输出头共用同一 packed 矩阵,小批次一次读权重完成全部候选评分。
        let (mixed, _) = hc_mix(ctx, cfg, &verified, output_hc)?;
        let logits = ctx.linear(&mixed, lm_head)?;
        let mut target_tokens = Vec::with_capacity(inputs.len());
        for row in 0..inputs.len() {
            target_tokens.push(ctx.argmax(&ctx.select_row(&logits, row)?)?);
        }
        let verification = crate::runtime::speculative::verify_samples(&target_tokens, &candidates, &cfg.eos_token_ids)?;
        let verify_elapsed = verify_start.elapsed().as_secs_f64();
        verify_wall += verify_elapsed;
        accepted += verification.accepted_drafts;
        rounds += 1;
        if verification.retained_rows < inputs.len() {
            state.retain_prefix(ctx, position, verification.retained_rows)?;
        }
        hidden = ctx.select_row(&verified, verification.retained_rows - 1)?;
        for &token in &verification.tokens {
            eprintln!("[qwen4exp-cuda-token] step={generated} id={token} elapsed={:.6}s", started.elapsed().as_secs_f64());
            crate::runtime::generation::write_token(&detokenizer, token, true)?;
            generated += 1;
        }
        anchor = verification.pending_token();
        let catchup_start = Instant::now();
        if generated < options.decode_steps && !verification.eos {
            // anchor 的草稿 KV 已使用真实上一位置 hidden;后续位置必须用
            // 本轮 target hidden 重写,不能继续保留自动回归草稿的近似 hidden。
            mtp.truncate(position + 1);
            for row in 1..verification.retained_rows {
                let previous = ctx.select_row(&verified, row - 1)?;
                mtp.forward(ctx, source, rope, &previous, inputs[row], position + row)?;
            }
        }
        ctx.synchronize()?;
        let catchup_elapsed = catchup_start.elapsed().as_secs_f64();
        catchup_wall += catchup_elapsed;
        eprintln!(
            "[qwen4exp-mtp-round] round={rounds} position={position} proposed={} accepted={} emitted={} draft={draft_elapsed:.6} verify={verify_elapsed:.6} catchup={catchup_elapsed:.6} uploaded_bytes={}",
            candidates.len(),
            verification.accepted_drafts,
            verification.tokens.len(),
            experts.backend_state_mut().uploaded_bytes
        );
        position += verification.retained_rows;
        if verification.eos {
            break;
        }
    }
    ctx.synchronize()?;
    let elapsed = started.elapsed().as_secs_f64() - first_token_wall;
    let throughput = generated.saturating_sub(1) as f64 / elapsed.max(f64::EPSILON);
    let stats = experts.backend_state_mut();
    eprintln!(
        "[qwen4exp-mtp-summary] generated={generated} rounds={rounds} proposed={proposed} accepted={accepted} acceptance={:.4} effective_tok_per_s={throughput:.3} draft_wall={draft_wall:.6} verify_wall={verify_wall:.6} catchup_wall={catchup_wall:.6} decode_uploaded_bytes={} decode_loaded={}",
        accepted as f64 / (proposed as f64).max(1.0),
        stats.uploaded_bytes - initial_uploaded_bytes,
        stats.loaded - initial_loaded
    );
    eprintln!(
        "[qwen4exp-cuda-summary] prompt_tokens={prompt_tokens} generated={generated} ttft={:.6}s decode_rounds={} decode_wall={elapsed:.6}s decode_tok_per_s={throughput:.3} cache_bytes={}",
        prefill_wall + first_token_wall,
        generated.saturating_sub(1),
        experts.backend_state_mut().resident_bytes()
    );
    Ok(())
}
