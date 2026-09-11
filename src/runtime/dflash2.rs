//! DFlash2 的设备无关草稿编排；增量 draft KV 由 GLM 会话持有，
//! target 模型的 KV/DSA 状态不进入本网络。

pub mod cpu;
pub use crate::model_spec::dflash2::Dflash2Config;
use crate::runtime::dspark::{DsparkTargetCache, update_target_cache_weights};
use crate::{
    attention::{
        block::BlockAttentionSpec,
        gqa::GqaGeometry,
        rope::{RopeTable, RotaryLayout},
    },
    backend::{Backend, BackendError, BackendResources, BlockAttentionBackend, BlockConvolutionBackend, GqaPrefillBackend, LinearWeight, SegmentedTensorBackend},
    moe::Activation,
    runtime::speculative::HiddenStateCapturePlan,
    weight::{container::safetensor::TensorData, model::dflash2::Dflash2Checkpoint},
};

struct Convolution<W> {
    projection: W,
    base: W,
}
struct Attention<W> {
    q: W,
    q_norm: W,
    k: W,
    k_norm: W,
    v: W,
    out: W,
}
struct Layer<W> {
    input_norm: W,
    attention: Attention<W>,
    attention_conv: Convolution<W>,
    post_attention_norm: W,
    gate: W,
    up: W,
    down: W,
    mlp_conv: Convolution<W>,
}

pub struct Dflash2<W> {
    config: Dflash2Config,
    capture_projections: Vec<W>,
    hidden_norm: W,
    layers: Vec<Layer<W>>,
    output_norm: W,
    selector_projection: W,
    predecessor: W,
    successor: W,
}

fn compute(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}

impl<W> Dflash2<W> {
    pub fn load<B: BackendResources<Weight = W>>(backend: &B, checkpoint: &Dflash2Checkpoint) -> Result<Self, BackendError> {
        Self::load_with_capture(backend, checkpoint, true)
    }

    /// 分布式服务的 FC 由各 target stage 持有，drafter 不复制六份投影。
    pub fn load_drafter<B: BackendResources<Weight = W>>(backend: &B, checkpoint: &Dflash2Checkpoint) -> Result<Self, BackendError> {
        Self::load_with_capture(backend, checkpoint, false)
    }

    fn load_with_capture<B: BackendResources<Weight = W>>(backend: &B, checkpoint: &Dflash2Checkpoint, capture: bool) -> Result<Self, BackendError> {
        let config = checkpoint.config().clone();
        let prepare = |tensor: TensorData| {
            let columns = *tensor.shape.last().ok_or_else(|| compute(format!("DFlash2 {} 缺少 shape", tensor.name)))?;
            let rows = tensor.shape[..tensor.shape.len() - 1].iter().product();
            backend.prepare_weight(LinearWeight::Bf16Bytes(&tensor.data), rows, columns)
        };
        let weight = |name: &str| prepare(checkpoint.load(name).map_err(compute)?);
        let capture_projections = (0..if capture { config.target_layer_ids.len() } else { 0 }).map(|i| prepare(checkpoint.capture_projection(i).map_err(compute)?)).collect::<Result<_, _>>()?;
        let mut layers = Vec::with_capacity(config.layer_count);
        for index in 0..config.layer_count {
            let load = |suffix: &str| weight(&format!("layers.{index}.{suffix}"));
            let convolution = |name: &str| -> Result<_, BackendError> { Ok(Convolution { projection: load(&format!("{name}.kernel_projection.weight"))?, base: load(&format!("{name}.base_kernel"))? }) };
            layers.push(Layer {
                input_norm: load("input_layernorm.weight")?,
                attention: Attention {
                    q: load("self_attn.q_proj.weight")?,
                    q_norm: load("self_attn.q_norm.weight")?,
                    k: load("self_attn.k_proj.weight")?,
                    k_norm: load("self_attn.k_norm.weight")?,
                    v: load("self_attn.v_proj.weight")?,
                    out: load("self_attn.o_proj.weight")?,
                },
                attention_conv: convolution("attention_conv")?,
                post_attention_norm: load("post_attention_layernorm.weight")?,
                gate: load("mlp.gate_proj.weight")?,
                up: load("mlp.up_proj.weight")?,
                down: load("mlp.down_proj.weight")?,
                mlp_conv: convolution("mlp_conv")?,
            });
        }
        Ok(Self {
            config,
            capture_projections,
            hidden_norm: weight("hidden_norm.weight")?,
            layers,
            output_norm: weight("norm.weight")?,
            selector_projection: weight("candidate_selector.hidden_projection.weight")?,
            predecessor: weight("candidate_selector.predecessor_codebook")?,
            successor: weight("candidate_selector.successor_codebook")?,
        })
    }

    pub fn config(&self) -> &Dflash2Config {
        &self.config
    }

    pub fn capture_plan(&self, target_layers: usize, target_hidden_size: usize, target_vocab_size: usize) -> Result<HiddenStateCapturePlan, BackendError> {
        let c = &self.config;
        if (target_layers, target_hidden_size, target_vocab_size) != (c.target_layer_count, c.hidden_size, c.vocab_size) {
            return Err(compute(format!("DFlash2 target 不兼容: layers/hidden/vocab={target_layers}/{target_hidden_size}/{target_vocab_size}，期望 {}/{}/{}", c.target_layer_count, c.hidden_size, c.vocab_size)));
        }
        HiddenStateCapturePlan::new(c.target_layer_ids.iter().map(|id| id + 1).collect(), target_layers).map_err(compute)
    }

    /// 顺序与 capture_plan 一致；FC 列切片的和等价于拼接后做一次线性投影。
    pub fn project_target<B: Backend<Weight = W>>(&self, backend: &B, captures: &[&B::Tensor]) -> Result<B::Tensor, BackendError> {
        if captures.is_empty() || captures.len() != self.capture_projections.len() {
            return Err(compute(format!("DFlash2 captures={}，期望 {}", captures.len(), self.capture_projections.len())));
        }
        let rows = backend.token_rows(captures[0]);
        let mut projected = None;
        for (index, (hidden, weight)) in captures.iter().zip(&self.capture_projections).enumerate() {
            if backend.token_rows(hidden) != rows || backend.token_cols(hidden) != self.config.hidden_size {
                return Err(compute(format!("DFlash2 capture {index} shape={}x{}，期望 {rows}x{}", backend.token_rows(hidden), backend.token_cols(hidden), self.config.hidden_size)));
            }
            let next = backend.linear(hidden, weight)?;
            projected = Some(match projected {
                Some(previous) => backend.add(&previous, &next)?,
                None => next,
            });
        }
        backend.rmsnorm(&projected.expect("capture 非空已由 config 校验"), &self.hidden_norm, self.config.rms_eps)
    }

    /// 输入行 0 是已经选定但尚未 target forward 的 anchor，后续行是 mask embedding。
    pub fn block_token_ids(&self, anchor: u32) -> Result<Vec<u32>, BackendError> {
        if anchor as usize >= self.config.vocab_size {
            return Err(compute(format!("DFlash2 anchor={anchor} 超出 vocab={}", self.config.vocab_size)));
        }
        let mut ids = vec![self.config.mask_token_id; self.config.block_size];
        ids[0] = anchor;
        Ok(ids)
    }

    /// 重算最近 target window 的 K/V，作为设备实现的 oracle；不维护推测 cache。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_hidden<B>(&self, backend: &B, noise: B::Tensor, target: &B::Tensor, target_position: usize, block_position: usize, rope: &RopeTable) -> Result<B::Tensor, BackendError>
    where
        B: BlockAttentionBackend<Weight = W> + BlockConvolutionBackend + GqaPrefillBackend + SegmentedTensorBackend,
    {
        let mut cache = DsparkTargetCache::new();
        self.forward_cached(backend, &mut cache, noise, target, target_position, block_position, rope)
    }

    pub fn normalize_target<B: Backend<Weight = W>>(&self, backend: &B, hidden: &B::Tensor) -> Result<B::Tensor, BackendError> {
        backend.rmsnorm(hidden, &self.hidden_norm, self.config.rms_eps)
    }

    pub fn warm_target_cache<B>(&self, backend: &B, cache: &mut DsparkTargetCache<B::Tensor>, target: &B::Tensor, position: usize, rope: &RopeTable) -> Result<(), BackendError>
    where
        B: GqaPrefillBackend<Weight = W> + SegmentedTensorBackend,
    {
        let c = &self.config;
        let end = position.checked_add(backend.token_rows(target)).ok_or_else(|| compute("DFlash2 cache position 溢出"))?;
        if backend.token_cols(target) != c.hidden_size || end > rope.seq_len || rope.rotary_dim != c.head_dim {
            return Err(compute("DFlash2 target cache shape/rope 不兼容"));
        }
        if cache.layers.is_empty() {
            cache.layers = (0..c.layer_count).map(|_| None).collect();
        }
        if cache.layers.len() != c.layer_count {
            return Err(compute("DFlash2 target cache 层数不匹配"));
        }
        for (layer, slot) in self.layers.iter().zip(&mut cache.layers) {
            let a = &layer.attention;
            update_target_cache_weights(backend, &a.k, &a.k_norm, &a.v, Some(c.sliding_window), Some(c.sliding_window), c.kv_head_count, c.head_dim, c.rms_eps, slot, target, position, &rope.cos, &rope.sin)?;
        }
        Ok(())
    }

    /// cache 只提交 target 的真实 hidden；当前块的 mask K/V 是短命 suffix。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached<B>(&self, backend: &B, cache: &mut DsparkTargetCache<B::Tensor>, mut noise: B::Tensor, target: &B::Tensor, target_position: usize, block_position: usize, rope: &RopeTable) -> Result<B::Tensor, BackendError>
    where
        B: BlockAttentionBackend<Weight = W> + BlockConvolutionBackend + GqaPrefillBackend + SegmentedTensorBackend,
    {
        let c = &self.config;
        let context_rows = backend.token_rows(target);
        let end = block_position.checked_add(c.block_size).ok_or_else(|| compute("DFlash2 block position 溢出"))?;
        if backend.token_rows(&noise) != c.block_size
            || backend.token_cols(&noise) != c.hidden_size
            || backend.token_cols(target) != c.hidden_size
            || target_position.checked_add(context_rows) != Some(block_position)
            || target_position > block_position.saturating_sub(c.sliding_window - 1)
            || end > c.max_position_embeddings
            || rope.rotary_dim != c.head_dim
            || rope.seq_len < end
            || rope.cos.len() != rope.seq_len.saturating_mul(c.head_dim / 2)
            || rope.sin.len() != rope.cos.len()
        {
            return Err(compute(format!(
                "DFlash2 forward shape/range 非法: noise={}x{} target={context_rows}x{}@{target_position} block={block_position}..{end} rope={}x{}",
                backend.token_rows(&noise),
                backend.token_cols(&noise),
                backend.token_cols(target),
                rope.seq_len,
                rope.rotary_dim
            )));
        }
        if cache.covered_range(backend) != Some(target_position..block_position) {
            self.warm_target_cache(backend, cache, target, target_position, rope)?;
        }
        let spec = attention_spec(c, context_rows);
        for (index, layer) in self.layers.iter().enumerate() {
            let a = &layer.attention;
            let residual = noise;
            let normalized = backend.rmsnorm(&residual, &layer.input_norm, c.rms_eps)?;
            let (input, delta) = self.conv_prepare(backend, &normalized, &layer.attention_conv)?;
            let query = backend.linear(&input, &a.q)?;
            let query = backend.rmsnorm_heads(&query, &a.q_norm, c.head_count, c.head_dim, c.rms_eps)?;
            let query = backend.rope(&query, c.head_count, c.head_dim, RotaryLayout::SplitHalf, block_position, &rope.cos, &rope.sin)?;
            // target 已融合并归一化，不再经过各层 input_norm 或块内卷积。
            let cached = cache.layers.get(index).and_then(Option::as_ref).ok_or_else(|| compute("DFlash2 cache 未完成预热"))?;
            if cached.start_position != target_position || backend.token_rows(&cached.key) != context_rows || backend.token_rows(&cached.value) != context_rows {
                return Err(compute("DFlash2 cache 层区间不一致"));
            }
            let noise_key = backend.linear(&input, &a.k)?;
            let noise_key = backend.rmsnorm_heads(&noise_key, &a.k_norm, c.kv_head_count, c.head_dim, c.rms_eps)?;
            let noise_key = backend.rope(&noise_key, c.kv_head_count, c.head_dim, RotaryLayout::SplitHalf, block_position, &rope.cos, &rope.sin)?;
            let noise_value = backend.linear(&input, &a.v)?;
            let attention = backend.block_attention_prefix_suffix(&query, &cached.key, &cached.value, &noise_key, &noise_value, &spec)?;
            let attention = backend.linear(&attention, &a.out)?;
            let attention = self.conv_finish(backend, &attention, &delta, &layer.attention_conv)?;
            let residual = backend.add(&residual, &attention)?;
            let normalized = backend.rmsnorm(&residual, &layer.post_attention_norm, c.rms_eps)?;
            let (input, delta) = self.conv_prepare(backend, &normalized, &layer.mlp_conv)?;
            let up = backend.linear(&input, &layer.up)?;
            let activated = backend.linear_gated_activation(&input, &layer.gate, &up, &Activation::Silu)?;
            let down = backend.linear(&activated, &layer.down)?;
            let down = self.conv_finish(backend, &down, &delta, &layer.mlp_conv)?;
            noise = backend.add(&residual, &down)?;
        }
        backend.rmsnorm(&noise, &self.output_norm, c.rms_eps)
    }

    pub fn selector_weights(&self) -> (&W, &W, &W) {
        (&self.selector_projection, &self.predecessor, &self.successor)
    }

    fn conv_prepare<B: BlockConvolutionBackend<Weight = W>>(&self, backend: &B, input: &B::Tensor, weights: &Convolution<W>) -> Result<(B::Tensor, B::Tensor), BackendError> {
        let delta = backend.linear(input, &weights.projection)?;
        let c = &self.config;
        let hidden = backend.grouped_block_conv(input, &delta, &weights.base, c.block_size, c.conv_group_size, c.conv_kernel_size, 0)?;
        Ok((hidden, delta))
    }

    fn conv_finish<B: BlockConvolutionBackend<Weight = W>>(&self, backend: &B, input: &B::Tensor, delta: &B::Tensor, weights: &Convolution<W>) -> Result<B::Tensor, BackendError> {
        let c = &self.config;
        backend.grouped_block_conv(input, delta, &weights.base, c.block_size, c.conv_group_size, c.conv_kernel_size, 1)
    }
}

fn attention_spec(c: &Dflash2Config, context_rows: usize) -> BlockAttentionSpec {
    let geometry = GqaGeometry { num_heads: c.head_count, num_kv_heads: c.kv_head_count, head_dim: c.head_dim };
    let end = context_rows + c.block_size;
    // HF 非因果滑窗为 |query_position-key_position| < window；每行左界不同。
    let visible = (0..c.block_size)
        .map(|row| {
            let position = context_rows + row;
            position.saturating_sub(c.sliding_window - 1)..position.saturating_add(c.sliding_window).min(end)
        })
        .collect();
    BlockAttentionSpec { geometry, score_scale: 1.0 / (c.head_dim as f32).sqrt(), visible }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn noncausal_window_moves_per_query() {
        let mut config = Dflash2Config::glm53();
        config.block_size = 3;
        config.sliding_window = 3;
        assert_eq!(attention_spec(&config, 4).visible, [2..7, 3..7, 4..7]);
        assert_eq!(attention_spec(&config, 0).visible, [0..3, 0..3, 0..3]);
    }
}
