//! GLM-5.2 DSpark 权重装配与执行；ROCm/CPU 只提供张量搬运能力。

use std::{path::Path, time::Instant};

use crate::runtime::dspark::{DsparkBackbone, DsparkLayer, DsparkTargetCache, DsparkTargetLayer, DsparkTargetProjector, DsparkTargetSegment, backbone_forward, backbone_forward_batch, warm_target_cache, warm_target_cache_projected};
use crate::{
    attention::rope::RopeTable,
    backend::{Backend, BackendError, BlockAttentionBackend, GqaPrefillBackend, LinearWeight, SegmentedTensorBackend},
    runtime::{output::prepare_lm_head_weight, prepare_resident_matrix, speculative::SpeculativeBlock},
    weight::{LmHeadQuantization, ResidentWeightQuantization, container::safetensor::TensorData, model::glm52_dspark::Glm52DsparkCheckpoint},
};

pub trait Glm52DsparkBackend: Backend + BlockAttentionBackend + GqaPrefillBackend + SegmentedTensorBackend {
    fn dspark_tensor_from_bf16_bits(&self, values: Vec<u16>, rows: usize, cols: usize) -> Result<Self::Tensor, BackendError>;
    fn dspark_tensor_as_f32(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError>;
    fn dspark_tensor_to_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, BackendError>;
    fn dspark_argmax_add_rows(&self, logits: &Self::Tensor, rows: &[u32], bias: &Self::Tensor) -> Result<Vec<u32>, BackendError>;
    fn dspark_profile_enabled(&self) -> bool {
        false
    }
}

pub struct Glm52DsparkRuntime<B: Glm52DsparkBackend> {
    checkpoint: Glm52DsparkCheckpoint,
    aux_norm: B::Weight,
    backbone: DsparkBackbone<B::Weight>,
    lm_head: B::Weight,
    markov_projection: B::Weight,
    confidence_projection: Option<B::Weight>,
    confidence_bias: f32,
    confidence_threshold: Option<f32>,
    draft_tokens: usize,
    rope: RopeTable,
}

/// CPU proposal 模式在 GPU 上保留的最小 cache runtime。aux capture projection
/// 由各 target stage 独立持有；这里仅有最终 RMSNorm 与三层 target K/V 投影。
pub struct Glm52DsparkCacheRuntime<B: Glm52DsparkBackend> {
    capture_count: usize,
    aux_norm: B::Weight,
    target: DsparkTargetProjector<B::Weight>,
    rope: RopeTable,
}

pub struct Glm52DsparkDraftBatch<'a, B: Glm52DsparkBackend> {
    pub cache: &'a mut DsparkTargetCache<B::Tensor>,
    pub anchor: u32,
    pub target_hidden: &'a B::Tensor,
    pub target_position: usize,
    pub block_position: usize,
    pub minimum_drafts: usize,
    pub drafts: Vec<u32>,
}

impl<B: Glm52DsparkBackend> Glm52DsparkCacheRuntime<B> {
    pub fn load(backend: &B, root: &Path, max_seq_len: usize, weight_quantization: ResidentWeightQuantization) -> Result<Self, BackendError> {
        let checkpoint = Glm52DsparkCheckpoint::open(root).map_err(compute)?;
        let cfg = &checkpoint.config;
        let transformer = &cfg.transformer_layer_config;
        let h = transformer.hidden_size;
        let mut layers = Vec::with_capacity(transformer.num_hidden_layers);
        for layer in 0..transformer.num_hidden_layers {
            let weights = checkpoint.load_target_layer(layer).map_err(compute)?;
            layers.push(DsparkTargetLayer {
                k_proj: prepare_dspark_matrix(backend, weights.k_proj, weight_quantization)?,
                k_norm: prepare(backend, weights.k_norm)?,
                v_proj: prepare_dspark_matrix(backend, weights.v_proj, weight_quantization)?,
                sliding_window: (transformer.layer_types[layer] == "sliding_attention").then_some(transformer.sliding_window).flatten(),
            });
        }
        Ok(Self {
            capture_count: cfg.aux_hidden_state_layer_ids.len(),
            aux_norm: prepare(backend, checkpoint.load("hidden_norm.weight", &[h]).map_err(compute)?)?,
            target: DsparkTargetProjector { layers, kv_head_count: transformer.num_key_value_heads, head_dim: transformer.head_dim, rms_eps: transformer.rms_norm_eps },
            rope: RopeTable::precompute(max_seq_len + cfg.block_size, transformer.head_dim, transformer.rope_parameters.rope_theta),
        })
    }

    pub fn capture_count(&self) -> usize {
        self.capture_count
    }

    pub fn target_history_window(&self) -> Option<usize> {
        self.target.target_history_window()
    }

    pub fn target_layer_count(&self) -> usize {
        self.target.layers.len()
    }

    pub fn target_columns(&self) -> usize {
        self.target.kv_head_count.saturating_mul(self.target.head_dim)
    }

    pub fn normalize_aux_hidden(&self, backend: &B, projected: &B::Tensor) -> Result<B::Tensor, BackendError> {
        backend.rmsnorm(projected, &self.aux_norm, self.target.rms_eps)
    }

    pub fn warm_target_cache(&self, backend: &B, cache: &mut DsparkTargetCache<B::Tensor>, target_hidden: &B::Tensor, target_position: usize) -> Result<(), BackendError> {
        warm_target_cache_projected(backend, &self.target, cache, target_hidden, target_position, &self.rope.cos, &self.rope.sin)
    }
}

impl<B: Glm52DsparkBackend> Glm52DsparkRuntime<B> {
    pub fn capture_count(&self) -> usize {
        self.checkpoint.config.aux_hidden_state_layer_ids.len()
    }

    pub fn target_history_window(&self) -> Option<usize> {
        self.backbone.target_history_window()
    }

    pub fn target_layer_count(&self) -> usize {
        self.backbone.layers.len()
    }

    pub fn target_columns(&self) -> usize {
        self.backbone.kv_head_count.saturating_mul(self.backbone.head_dim)
    }

    pub fn load(backend: &B, root: &Path, max_seq_len: usize, draft_tokens: usize, confidence_threshold: Option<f32>, lm_head_quantization: LmHeadQuantization, weight_quantization: ResidentWeightQuantization) -> Result<Self, BackendError> {
        let checkpoint = Glm52DsparkCheckpoint::open(root).map_err(compute)?;
        let cfg = &checkpoint.config;
        let block = cfg.block_spec().map_err(compute)?;
        if draft_tokens == 0 || draft_tokens > block.speculative_tokens {
            return Err(compute(format!("DSpark draft_tokens={draft_tokens} 超出 checkpoint block={}", block.speculative_tokens)));
        }
        if confidence_threshold.is_some_and(|threshold| !threshold.is_finite() || !(0.0..=1.0).contains(&threshold)) {
            return Err(compute(format!("DSpark confidence threshold={confidence_threshold:?} 非法")));
        }
        let transformer = &cfg.transformer_layer_config;
        let h = transformer.hidden_size;
        let mut layers = Vec::with_capacity(transformer.num_hidden_layers);
        for layer in 0..transformer.num_hidden_layers {
            let weights = checkpoint.load_layer(layer).map_err(compute)?;
            layers.push(DsparkLayer {
                input_norm: prepare(backend, weights.input_norm)?,
                q_proj: prepare_dspark_matrix(backend, weights.q_proj, weight_quantization)?,
                q_norm: prepare(backend, weights.q_norm)?,
                k_proj: prepare_dspark_matrix(backend, weights.k_proj, weight_quantization)?,
                k_norm: prepare(backend, weights.k_norm)?,
                v_proj: prepare_dspark_matrix(backend, weights.v_proj, weight_quantization)?,
                o_proj: prepare_dspark_matrix(backend, weights.o_proj, weight_quantization)?,
                post_attention_norm: prepare(backend, weights.post_attention_norm)?,
                gate_proj: prepare_dspark_matrix(backend, weights.gate_proj, weight_quantization)?,
                up_proj: prepare_dspark_matrix(backend, weights.up_proj, weight_quantization)?,
                down_proj: prepare_dspark_matrix(backend, weights.down_proj, weight_quantization)?,
                sliding_window: (transformer.layer_types[layer] == "sliding_attention").then_some(transformer.sliding_window).flatten(),
                sliding_window_non_causal: cfg.sliding_window_non_causal,
            });
        }
        let backbone = DsparkBackbone {
            layers,
            output_norm: prepare(backend, checkpoint.load("norm.weight", &[h]).map_err(compute)?)?,
            head_count: transformer.num_attention_heads,
            kv_head_count: transformer.num_key_value_heads,
            head_dim: transformer.head_dim,
            rms_eps: transformer.rms_norm_eps,
        };
        let lm_head = checkpoint.load("lm_head.weight", &[cfg.draft_vocab_size, h]).map_err(compute)?;
        let lm_head = prepare_lm_head_weight(backend, LinearWeight::Bf16Bytes(&lm_head.data), cfg.draft_vocab_size, h, lm_head_quantization)?;
        let (confidence_projection, confidence_bias) = match confidence_threshold {
            Some(_) => {
                let (projection, bias) = checkpoint.confidence_head().map_err(compute)?;
                let [low, high] = bias.data.as_slice() else {
                    return Err(compute(format!("DSpark confidence bias 字节数={}，期望 2", bias.data.len())));
                };
                (Some(prepare(backend, projection)?), half::bf16::from_le_bytes([*low, *high]).to_f32())
            }
            None => (None, 0.0),
        };
        Ok(Self {
            aux_norm: prepare(backend, checkpoint.load("hidden_norm.weight", &[h]).map_err(compute)?)?,
            lm_head,
            markov_projection: prepare_dspark_matrix(backend, checkpoint.load("markov_head.markov_w2.weight", &[cfg.draft_vocab_size, cfg.markov_rank]).map_err(compute)?, weight_quantization)?,
            confidence_projection,
            confidence_bias,
            confidence_threshold,
            draft_tokens,
            rope: RopeTable::precompute(max_seq_len + cfg.block_size, transformer.head_dim, transformer.rope_parameters.rope_theta),
            checkpoint,
            backbone,
        })
    }

    pub fn normalize_aux_hidden(&self, backend: &B, projected: &B::Tensor) -> Result<B::Tensor, BackendError> {
        backend.rmsnorm(projected, &self.aux_norm, self.backbone.rms_eps)
    }

    pub fn warm_target_cache(&self, backend: &B, cache: &mut DsparkTargetCache<B::Tensor>, target_hidden: &B::Tensor, target_position: usize) -> Result<(), BackendError> {
        warm_target_cache(backend, &self.backbone, cache, target_hidden, target_position, &self.rope.cos, &self.rope.sin)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draft_block(&self, backend: &B, cache: &mut DsparkTargetCache<B::Tensor>, anchor: u32, target_hidden: &B::Tensor, target_position: usize, block_position: usize, minimum_drafts: usize) -> Result<SpeculativeBlock, BackendError> {
        let profile = backend.dspark_profile_enabled();
        let mut phase_started = profile.then(Instant::now);
        let mut phase_micros = [0_u128; 5];
        let draft_count = self.draft_tokens;
        // causal block 的行独立:可见行 0..draft_count 的输出只依赖噪声行
        // 0..draft_count,行 draft_count..block_size 不被任何消费方读取
        // (lm_head/markov/confidence 都只取前 draft_count 行)。按 draft_count
        // 行计算与完整 block 在被消费的行上逐位一致,draft 长度不变。
        let mut input_ids = vec![self.checkpoint.config.mask_token_id; draft_count];
        input_ids[0] = anchor;
        let noise = upload_rows(backend, self.checkpoint.embedding_rows(&input_ids).map_err(compute)?)?;
        if profile {
            backend.synchronize()?;
            phase_micros[0] = phase_started.take().unwrap().elapsed().as_micros();
            phase_started = Some(Instant::now());
        }
        let hidden = backbone_forward(backend, &self.backbone, cache, noise, target_hidden, target_position, block_position, &self.rope.cos, &self.rope.sin)?;
        if profile {
            backend.synchronize()?;
            phase_micros[1] = phase_started.take().unwrap().elapsed().as_micros();
            phase_started = Some(Instant::now());
        }
        let base_logits = backend.linear(&hidden, &self.lm_head)?;
        if profile {
            backend.synchronize()?;
            phase_micros[2] = phase_started.take().unwrap().elapsed().as_micros();
            phase_started = Some(Instant::now());
        }
        let mut previous = anchor;
        let mut drafts = Vec::with_capacity(draft_count);
        let mut markov_rows = self.confidence_projection.as_ref().map(|_| Vec::with_capacity(draft_count));
        for row in 0..draft_count {
            let embedding = upload_rows(backend, self.checkpoint.markov_rows(&[previous]).map_err(compute)?)?;
            let bias = backend.linear(&embedding, &self.markov_projection)?;
            let row = u32::try_from(row).map_err(|_| compute("DSpark logits row 超过 u32"))?;
            let token = backend.dspark_argmax_add_rows(&base_logits, &[row], &bias)?.pop().ok_or_else(|| compute("DSpark add+argmax 没有返回 token"))?;
            drafts.push(token);
            if let Some(rows) = &mut markov_rows {
                rows.push(embedding);
            }
            previous = token;
        }
        if profile {
            backend.synchronize()?;
            phase_micros[3] = phase_started.take().unwrap().elapsed().as_micros();
            phase_started = Some(Instant::now());
        }
        let confidences = match markov_rows {
            Some(rows) => self.confidence_scores(backend, &hidden, &rows, 1, draft_count)?,
            None => vec![Vec::new()],
        };
        if profile {
            backend.synchronize()?;
            phase_micros[4] = phase_started.take().unwrap().elapsed().as_micros();
            eprintln!(
                "[glm52-dspark-draft-phases] batch=1 input_ms={:.3} backbone_ms={:.3} head_ms={:.3} markov_ms={:.3} confidence_ms={:.3}",
                phase_micros[0] as f64 / 1000.0,
                phase_micros[1] as f64 / 1000.0,
                phase_micros[2] as f64 / 1000.0,
                phase_micros[3] as f64 / 1000.0,
                phase_micros[4] as f64 / 1000.0,
            );
        }
        let block = SpeculativeBlock::new(anchor, drafts, confidences.into_iter().next().unwrap_or_default())?;
        match self.confidence_threshold {
            Some(threshold) => block.retain_confident_prefix_at_least(threshold, minimum_drafts.min(self.draft_tokens)),
            None => Ok(block),
        }
    }

    pub fn draft_batch(&self, backend: &B, batch: &mut [Glm52DsparkDraftBatch<'_, B>]) -> Result<(), BackendError> {
        if batch.is_empty() {
            return Ok(());
        }
        if batch.len() == 1 {
            let item = &mut batch[0];
            item.drafts = self.draft_block(backend, item.cache, item.anchor, item.target_hidden, item.target_position, item.block_position, item.minimum_drafts)?.drafts;
            return Ok(());
        }
        for item in batch.iter_mut() {
            item.drafts.clear();
        }
        // 与 draft_block 相同:causal block 只需计算前 draft_tokens 行,
        // 行 draft_tokens..block_size 不被消费,按行独立性逐位等价。
        let draft_count = self.draft_tokens;
        let mut input_ids = vec![self.checkpoint.config.mask_token_id; batch.len() * draft_count];
        for (session, item) in batch.iter().enumerate() {
            input_ids[session * draft_count] = item.anchor;
        }
        let profile = backend.dspark_profile_enabled();
        let mut phase_started = profile.then(Instant::now);
        let mut phase_micros = [0_u128; 5];
        let noise = upload_rows(backend, self.checkpoint.embedding_rows(&input_ids).map_err(compute)?)?;
        if profile {
            backend.synchronize()?;
            phase_micros[0] = phase_started.take().unwrap().elapsed().as_micros();
            phase_started = Some(Instant::now());
        }
        let mut segments = batch.iter_mut().map(|item| DsparkTargetSegment { cache: &mut *item.cache, hidden: item.target_hidden, position: item.target_position, block_position: item.block_position }).collect::<Vec<_>>();
        let hidden = backbone_forward_batch(backend, &self.backbone, &mut segments, draft_count, noise, &self.rope.cos, &self.rope.sin)?;
        if profile {
            backend.synchronize()?;
            phase_micros[1] = phase_started.take().unwrap().elapsed().as_micros();
            phase_started = Some(Instant::now());
        }
        let base_logits = backend.linear(&hidden, &self.lm_head)?;
        if profile {
            backend.synchronize()?;
            phase_micros[2] = phase_started.take().unwrap().elapsed().as_micros();
            phase_started = Some(Instant::now());
        }
        let mut previous = batch.iter().map(|item| item.anchor).collect::<Vec<_>>();
        let mut markov_rows = self.confidence_projection.as_ref().map(|_| Vec::with_capacity(self.draft_tokens));
        for depth in 0..self.draft_tokens {
            let embedding = upload_rows(backend, self.checkpoint.markov_rows(&previous).map_err(compute)?)?;
            let bias = backend.linear(&embedding, &self.markov_projection)?;
            let rows = (0..batch.len()).map(|session| u32::try_from(session * draft_count + depth).map_err(|_| compute("DSpark batch logits row 超过 u32"))).collect::<Result<Vec<_>, _>>()?;
            previous = backend.dspark_argmax_add_rows(&base_logits, &rows, &bias)?;
            if previous.len() != batch.len() {
                return Err(compute(format!("DSpark batch argmax rows={}，期望 {}", previous.len(), batch.len())));
            }
            for (item, &token) in batch.iter_mut().zip(&previous) {
                item.drafts.push(token);
            }
            if let Some(rows) = &mut markov_rows {
                rows.push(embedding);
            }
        }
        if profile {
            backend.synchronize()?;
            phase_micros[3] = phase_started.take().unwrap().elapsed().as_micros();
            phase_started = Some(Instant::now());
        }
        let confidences = match markov_rows {
            Some(rows) => self.confidence_scores(backend, &hidden, &rows, batch.len(), draft_count)?,
            None => vec![Vec::new(); batch.len()],
        };
        if profile {
            backend.synchronize()?;
            phase_micros[4] = phase_started.take().unwrap().elapsed().as_micros();
            eprintln!(
                "[glm52-dspark-draft-phases] batch={} input_ms={:.3} backbone_ms={:.3} head_ms={:.3} markov_ms={:.3} confidence_ms={:.3}",
                batch.len(),
                phase_micros[0] as f64 / 1000.0,
                phase_micros[1] as f64 / 1000.0,
                phase_micros[2] as f64 / 1000.0,
                phase_micros[3] as f64 / 1000.0,
                phase_micros[4] as f64 / 1000.0,
            );
        }
        for (item, confidences) in batch.iter_mut().zip(confidences) {
            let block = SpeculativeBlock::new(item.anchor, std::mem::take(&mut item.drafts), confidences)?;
            item.drafts = match self.confidence_threshold {
                Some(threshold) => block.retain_confident_prefix_at_least(threshold, item.minimum_drafts.min(self.draft_tokens))?.drafts,
                None => block.drafts,
            };
        }
        Ok(())
    }

    /// confidence head 只做一次 `[session * K, hidden + markov_rank]` 投影和一次回读。
    /// backbone hidden 是 session-major，Markov 行是 depth-major；这里在设备上重排后
    /// 再合批，避免 confidence 路径退回逐 session 执行。
    fn confidence_scores(&self, backend: &B, hidden: &B::Tensor, markov_depths: &[B::Tensor], sessions: usize, block_size: usize) -> Result<Vec<Vec<f32>>, BackendError> {
        let Some(projection) = self.confidence_projection.as_ref() else {
            return Ok(vec![Vec::new(); sessions]);
        };
        if sessions == 0 || markov_depths.len() != self.draft_tokens || markov_depths.iter().any(|rows| backend.token_rows(rows) != sessions) {
            return Err(compute(format!("DSpark confidence batch 非法: sessions={sessions} depths={} draft_tokens={}", markov_depths.len(), self.draft_tokens)));
        }
        let total_rows = sessions.checked_mul(self.draft_tokens).ok_or_else(|| compute("DSpark confidence rows 溢出"))?;
        let mut hidden_rows = Vec::with_capacity(sessions);
        let mut markov_rows = Vec::with_capacity(total_rows);
        for session in 0..sessions {
            hidden_rows.push(backend.slice_token_rows(hidden, session * block_size, self.draft_tokens)?);
            for rows in markov_depths {
                markov_rows.push(backend.slice_token_rows(rows, session, 1)?);
            }
        }
        let hidden_refs = hidden_rows.iter().collect::<Vec<_>>();
        let hidden = backend.dspark_tensor_as_f32(backend.concat_token_rows(&hidden_refs)?)?;
        let markov_refs = markov_rows.iter().collect::<Vec<_>>();
        let markov = backend.dspark_tensor_as_f32(backend.concat_token_rows(&markov_refs)?)?;
        let features = backend.concat_columns(&hidden, &markov)?;
        let raw = backend.dspark_tensor_to_f32(&backend.linear(&features, projection)?)?;
        if raw.len() != total_rows {
            return Err(compute(format!("DSpark confidence 输出={}，期望 {total_rows}", raw.len())));
        }
        Ok(raw.chunks_exact(self.draft_tokens).map(|rows| rows.iter().map(|&raw| confidence_probability(raw, self.confidence_bias)).collect()).collect())
    }
}

fn confidence_probability(raw: f32, bias: f32) -> f32 {
    let raw = raw + bias;
    if raw >= 0.0 {
        1.0 / (1.0 + (-raw).exp())
    } else {
        let exp = raw.exp();
        exp / (1.0 + exp)
    }
}

fn prepare<B: Glm52DsparkBackend>(backend: &B, tensor: TensorData) -> Result<B::Weight, BackendError> {
    if tensor.dtype != "BF16" || tensor.shape.is_empty() || tensor.shape.len() > 2 {
        return Err(compute(format!("DSpark {} dtype/shape 不支持: {} {:?}", tensor.name, tensor.dtype, tensor.shape)));
    }
    let (rows, cols) = if tensor.shape.len() == 1 { (1, tensor.shape[0]) } else { (tensor.shape[0], tensor.shape[1]) };
    backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, cols)
}

pub(super) fn prepare_dspark_matrix<B: Glm52DsparkBackend>(backend: &B, tensor: TensorData, quantization: ResidentWeightQuantization) -> Result<B::Weight, BackendError> {
    if tensor.dtype != "BF16" || tensor.shape.len() != 2 {
        return Err(compute(format!("DSpark matrix {} dtype/shape 不支持: {} {:?}", tensor.name, tensor.dtype, tensor.shape)));
    }
    prepare_resident_matrix(backend, LinearWeight::Bf16Bytes(&tensor.data), tensor.shape[0], tensor.shape[1], quantization, &tensor.name)
}

fn upload_rows<B: Glm52DsparkBackend>(backend: &B, tensor: TensorData) -> Result<B::Tensor, BackendError> {
    if tensor.dtype != "BF16" || tensor.shape.len() != 2 {
        return Err(compute(format!("DSpark row tensor {} 非 BF16 rank-2", tensor.name)));
    }
    let values = tensor.data.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect();
    backend.dspark_tensor_from_bf16_bits(values, tensor.shape[0], tensor.shape[1])
}

fn compute(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}

#[cfg(test)]
mod tests {
    use super::confidence_probability;

    #[test]
    fn confidence_sigmoid_is_stable_and_includes_bias() {
        assert_eq!(confidence_probability(0.0, 0.0), 0.5);
        assert!(confidence_probability(100.0, 0.0).is_finite());
        assert!(confidence_probability(-100.0, 0.0).is_finite());
        assert!(confidence_probability(0.0, 1.0) > confidence_probability(0.0, 0.0));
    }
}
