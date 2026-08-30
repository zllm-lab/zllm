//! FullAttention 与 Gated DeltaNet 混合层的共享执行算法。

use crate::{
    attention::{
        gated_delta_net::{GatedDeltaNetInputs, GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetState, GatedDeltaNetWeightsRef, gated_delta_net},
        gqa::GqaSpec,
        rope::{RopeTable, RotaryLayout},
    },
    backend::{Backend, BackendError, GqaPrefillBackend},
};

#[derive(Debug)]
pub struct FullAttentionWeights<W> {
    pub query_gate: W,
    pub query_norm: W,
    pub key: W,
    pub key_norm: W,
    pub value: W,
    pub output: W,
}

#[derive(Debug)]
pub struct DeltaNetWeights<W> {
    pub qkv: W,
    pub z: W,
    pub alpha: W,
    pub beta: W,
    pub conv: W,
    pub a_log: W,
    pub dt_bias: W,
    pub norm: W,
    pub output: W,
}

#[derive(Debug)]
pub enum HybridTokenMixer<W> {
    FullAttention(FullAttentionWeights<W>),
    DeltaNet(DeltaNetWeights<W>),
}

#[derive(Clone, Copy, Default)]
pub struct HybridAttentionOptions {
    pub precise_prefill: bool,
}

/// 模型加载期间不变的混合注意力规格与执行依赖。
pub struct HybridAttention<'a, B: Backend> {
    backend: &'a B,
    full: GqaSpec,
    delta: GatedDeltaNetSpec,
    rms_eps: f32,
    rope: &'a RopeTable,
    options: HybridAttentionOptions,
}

impl<'a, B: Backend> HybridAttention<'a, B> {
    pub fn new(backend: &'a B, full: GqaSpec, delta: GatedDeltaNetSpec, rms_eps: f32, rope: &'a RopeTable, options: HybridAttentionOptions) -> Self {
        Self { backend, full, delta, rms_eps, rope, options }
    }

    /// q_proj 同时产生 query 与 output gate，随后执行 GQA 和输出投影。
    pub fn full(&self, weights: &FullAttentionWeights<B::Weight>, cache: &mut B::Cache, cache_layer: usize, input: &B::Tensor, position: usize) -> Result<B::Tensor, BackendError>
    where
        B: GqaPrefillBackend,
    {
        let precise_prefill = self.backend.token_rows(input) > 1 && self.options.precise_prefill;
        let query_gate = if precise_prefill { self.backend.linear_f32(input, &weights.query_gate)? } else { self.backend.linear(input, &weights.query_gate)? };
        let (query, gate) = self.backend.split_interleaved_columns(&query_gate, self.full.head_dim)?;
        let (key, value) = self.backend.dual_linear(input, &weights.key, &weights.value)?;
        let query = if precise_prefill {
            self.backend.gemma_rmsnorm_heads_f32(&query, &weights.query_norm, self.full.num_heads, self.full.head_dim, self.rms_eps)?
        } else {
            self.backend.gemma_rmsnorm_heads(&query, &weights.query_norm, self.full.num_heads, self.full.head_dim, self.rms_eps)?
        };
        let key = self.backend.gemma_rmsnorm_heads(&key, &weights.key_norm, self.full.num_kv_heads, self.full.head_dim, self.rms_eps)?;
        let query = self.backend.rope_prefix(&query, self.full.num_heads, self.full.rope_dim, RotaryLayout::SplitHalf, position, &self.rope.cos, &self.rope.sin)?;
        let key = self.backend.rope_prefix(&key, self.full.num_kv_heads, self.full.rope_dim, RotaryLayout::SplitHalf, position, &self.rope.cos, &self.rope.sin)?;
        let attention = self.backend.gqa_prefill_attention_cached(cache, cache_layer, position, &query, &key, &value, &self.full, false)?;
        let attention = self.backend.sigmoid_gate(&attention, &gate)?;
        self.backend.linear(&attention, &weights.output)
    }

    /// 短卷积、delta rule 递归、门控 RMSNorm 与输出投影。
    pub fn delta(&self, weights: &DeltaNetWeights<B::Weight>, state: &mut GatedDeltaNetState<B::GatedDeltaNetStorage>, layer: usize, input: &B::Tensor, position: usize) -> Result<B::Tensor, BackendError>
    where
        B: GatedDeltaNetKernel,
    {
        let (qkv, z) = self.backend.dual_linear(input, &weights.qkv, &weights.z)?;
        let (alpha, beta) = self.backend.dual_linear(input, &weights.alpha, &weights.beta)?;
        let mixed = gated_delta_net(
            self.backend,
            state,
            layer,
            position,
            GatedDeltaNetInputs { qkv: &qkv, z: &z, alpha: &alpha, beta: &beta },
            GatedDeltaNetWeightsRef { conv: &weights.conv, a_log: &weights.a_log, dt_bias: &weights.dt_bias, norm: &weights.norm },
            &self.delta,
        )?;
        self.backend.linear(&mixed, &weights.output)
    }

    /// 只推进 recurrent state 的 GDN 前进(DSpark 部分接受后的重放)。
    /// z 门控与 output 投影只决定层输出、不影响 conv/delta-rule state,
    /// 重放的 hidden 直接取 verify 输出行;传零 z 占位省掉两份投影的权重读。
    pub fn delta_advance_state(&self, weights: &DeltaNetWeights<B::Weight>, state: &mut GatedDeltaNetState<B::GatedDeltaNetStorage>, layer: usize, input: &B::Tensor, position: usize, zero_z: &B::Tensor) -> Result<(), BackendError>
    where
        B: GatedDeltaNetKernel,
    {
        let qkv = self.backend.linear(input, &weights.qkv)?;
        let (alpha, beta) = self.backend.dual_linear(input, &weights.alpha, &weights.beta)?;
        gated_delta_net(
            self.backend,
            state,
            layer,
            position,
            GatedDeltaNetInputs { qkv: &qkv, z: zero_z, alpha: &alpha, beta: &beta },
            GatedDeltaNetWeightsRef { conv: &weights.conv, a_log: &weights.a_log, dt_bias: &weights.dt_bias, norm: &weights.norm },
            &self.delta,
        )?;
        Ok(())
    }
}
