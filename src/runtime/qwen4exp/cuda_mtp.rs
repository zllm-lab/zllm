//! shared MTP 的一层草稿网络。原始专家驻留主存,只借用 target embedding/output。

use super::{Qwen4ExpConfig, Qwen4ExpGguf, Qwen4ExpHyperConnection, Qwen4ExpLayer, Qwen4ExpMixer};
use crate::{
    attention::rope::RopeTable,
    backend::{
        Backend,
        cuda::{CudaContext, CudaKvCache, CudaMoeState, CudaTensor, CudaWeight},
    },
    kernel::cuda::{diffusion::cast_f16_to_f32, grouped},
    moe::{
        expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
        topk_moe::{MoeFfnRef, SharedExpertRef},
    },
    runtime::{
        expert_pipeline::{ExpertDecodePipeline, ExpertDecodeRequest},
        prepare_gguf_f32_vector, prepare_gguf_matrix,
    },
    weight::{
        container::gguf::{GgufReader, GgufValue},
        expert_source::{ExpertSource, GgufExpertSource, GgufExpertWeights},
    },
};
use std::{path::Path, sync::Arc};

pub(super) struct MtpSource {
    pub reader: GgufReader,
    pub experts: Vec<GgufExpertWeights>,
    cfg: Qwen4ExpConfig,
}

impl MtpSource {
    pub fn open(path: &Path, cfg: &Qwen4ExpConfig) -> Result<Self, String> {
        let reader = GgufReader::open(&GgufReader::locate(path)?)?;
        reader.expect_metadata_str("general.architecture", "qwen4exp")?;
        if reader.metadata_u64("qwen4exp.nextn_predict_layers")? != 1
            || reader.metadata_u64("qwen4exp.block_count")? as usize != cfg.num_layers + 1
            || !matches!(reader.metadata("qwen4exp.nextn_shared_target_tensors"), Some(GgufValue::Bool(true)))
        {
            return Err("Qwen4-Exp MTP 需要与 target 层数匹配的单层 shared GGUF".into());
        }
        for (key, expected) in [
            ("embedding_length", cfg.hidden_size),
            ("expert_count", cfg.num_experts),
            ("expert_used_count", cfg.num_experts_per_tok),
            ("expert_feed_forward_length", cfg.expert_intermediate_size),
            ("attention.head_count", cfg.num_attention_heads),
            ("attention.head_count_kv", cfg.num_kv_heads),
            ("attention.key_length", cfg.head_dim),
            ("hyper_connection.count", cfg.hyper_connection.streams),
            ("hyper_connection.low_rank", cfg.hyper_connection.low_rank),
        ] {
            let actual = reader.metadata_u64(&format!("qwen4exp.{key}"))? as usize;
            if actual != expected {
                return Err(format!("MTP {key}={actual} 与 target {expected} 不一致"));
            }
        }
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for expert in 0..cfg.num_experts {
            let prefix = format!("blk.{}", cfg.num_layers);
            let weights = GgufExpertWeights {
                gate: reader.read_matrix_slice(&format!("{prefix}.ffn_gate_exps.weight"), expert)?,
                up: reader.read_matrix_slice(&format!("{prefix}.ffn_up_exps.weight"), expert)?,
                down: reader.read_matrix_slice(&format!("{prefix}.ffn_down_exps.weight"), expert)?,
            };
            if weights.gate.tensor_type.0 != 12 || weights.up.tensor_type.0 != 12 || !matches!(weights.down.tensor_type.0, 7 | 8) {
                return Err(format!("MTP E{expert} 原生专家格式不支持"));
            }
            weights.gate.bytes()?;
            weights.up.bytes()?;
            weights.down.bytes()?;
            experts.push(weights);
        }
        Ok(Self { reader, experts, cfg: cfg.clone() })
    }
}

impl GgufExpertSource for MtpSource {
    fn intermediate(&self) -> usize {
        self.cfg.expert_intermediate_size
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn load_expert_gguf(&self, layer: usize, expert: usize) -> Result<GgufExpertWeights, String> {
        if layer != self.cfg.num_layers {
            return Err(format!("MTP expert layer={layer}, 期望 {}", self.cfg.num_layers));
        }
        self.experts.get(expert).cloned().ok_or_else(|| format!("MTP expert={expert} 超出 {}", self.experts.len()))
    }
}

pub(super) struct Mtp {
    source: Arc<MtpSource>,
    weights: Qwen4ExpLayer<CudaWeight>,
    enorm: CudaWeight,
    hnorm: CudaWeight,
    eh_proj: CudaWeight,
    head: Qwen4ExpHyperConnection<CudaWeight>,
    cache: CudaKvCache,
    experts: ExpertDecodePipeline<CudaMoeState>,
}

impl Mtp {
    pub fn new(ctx: &CudaContext, source: Arc<MtpSource>, max_seq_len: usize, cache_bytes: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = &source.cfg;
        let weights = super::prepare_layer(ctx, &source.reader, cfg, cfg.num_layers, true)?;
        let prefix = format!("blk.{}.nextn", cfg.num_layers);
        let enorm = prepare_gguf_f32_vector(ctx, &source.reader, &format!("{prefix}.enorm.weight"))?;
        let hnorm = prepare_gguf_f32_vector(ctx, &source.reader, &format!("{prefix}.hnorm.weight"))?;
        let eh_proj = prepare_gguf_matrix(ctx, &source.reader, &format!("{prefix}.eh_proj.weight"))?;
        let head = super::prepare_hc(ctx, &source.reader, &format!("{prefix}.hc_head"), false)?;
        let cache = CudaKvCache::new(1, max_seq_len, cfg.num_kv_heads * cfg.head_dim);
        let mut expert_state = CudaMoeState::new(cache_bytes);
        expert_state.reserve_arena(ctx)?;
        let experts = ExpertDecodePipeline::new(
            expert_state,
            ExpertPredictorConfig { first_layer: cfg.num_layers, layer_count: 1, expert_count: cfg.num_experts, routed_top_k: cfg.num_experts_per_tok, prefetch_count: 0, weights: ExpertPredictorWeights::default() },
        )?;
        Ok(Self { source, weights, enorm, hnorm, eh_proj, head, cache, experts })
    }

    pub fn truncate(&mut self, position: usize) {
        self.cache.truncate(position);
    }

    /// token[position] 与 target/draft 的上一位置原始 HC hidden 配对。
    pub fn forward(&mut self, ctx: &CudaContext, target: &Qwen4ExpGguf, rope: &RopeTable, hidden: &CudaTensor, token: u32, position: usize) -> Result<CudaTensor, Box<dyn std::error::Error>> {
        let cfg = &self.source.cfg;
        let groups = cfg.hyper_connection.streams;
        if hidden.rows != 1 || hidden.cols != cfg.hc_dim() {
            return Err("MTP 输入必须是一行原始 HC hidden".into());
        }
        let embedding = super::cuda::selected_rows(ctx, target, "token_embd.weight", &[token], 1, cfg.hidden_size)?;
        let embedding = ctx.rmsnorm(&embedding, &self.enorm, cfg.rms_norm_eps)?;
        let normalized = grouped::norm(ctx, hidden, &self.hnorm, groups, cfg.rms_norm_eps)?;
        // 每流分别拼接和投影;先池化 HC 会丢失草稿网络训练时使用的残差。
        let joined = grouped::concat_broadcast_groups(ctx, &embedding, &normalized, groups)?;
        let projected = ctx.linear(&joined, &self.eh_proj)?;
        let projected = cast_f16_to_f32(ctx, &projected.slice, projected.len())?;
        let mut residual = CudaTensor::new_f32_residual(projected, ctx.placeholder_f16()?, 1, cfg.hc_dim());
        let (mixed, inject) = super::cuda::hc_mix(ctx, cfg, &residual, &self.weights.hc_attn)?;
        let Qwen4ExpMixer::SparseAttention { attention, .. } = &self.weights.mixer else {
            return Err("MTP 必须采用全注意力".into());
        };
        let mixed = super::cuda::full_attention(ctx, cfg, attention, &mut self.cache, rope, 0, &mixed, position)?;
        residual = grouped::sigmoid_residual(ctx, &residual, &mixed, inject.as_ref().unwrap())?;
        let (mixed, inject) = super::cuda::hc_mix(ctx, cfg, &residual, &self.weights.hc_ffn)?;
        let moe = &self.weights.moe;
        let shared = [SharedExpertRef { gate: &moe.shared_gate, up: &moe.shared_up, down: &moe.shared_down, output_gate: Some(&moe.shared_output_gate) }];
        let weights = MoeFfnRef { router_weight: &moe.router, router_bias: &moe.router_bias, shared_experts: &shared, selected_experts: None };
        let output = self.experts.decode(ctx, &cfg.moe_spec(), &weights, ExpertDecodeRequest { layer: cfg.num_layers, source: ExpertSource::Gguf(self.source.as_ref()), position, next: None }, &mixed)?;
        Ok(grouped::sigmoid_residual(ctx, &residual, &output, inject.as_ref().unwrap())?)
    }

    pub fn sample(&self, ctx: &CudaContext, hidden: &CudaTensor, target_head: &CudaWeight) -> Result<(u32, f32), Box<dyn std::error::Error>> {
        let (mixed, _) = super::cuda::hc_mix(ctx, &self.source.cfg, hidden, &self.head)?;
        let logits = ctx.linear(&mixed, target_head)?;
        let token = ctx.argmax(&logits)?;
        let probability = crate::kernel::cuda::tensor::token_probability_f16(ctx, &logits, token)?;
        Ok((token, probability))
    }
}
