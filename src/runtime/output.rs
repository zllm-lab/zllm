//! 模型输出阶段的设备侧公共编排。

use crate::{
    backend::{Backend, BackendError, LinearWeight, SegmentedTensorBackend, TokenFence, TokenSampling},
    weight::LmHeadQuantization,
};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl SamplingConfig {
    pub const fn greedy(seed: u64) -> Self {
        Self { temperature: 0.0, top_p: 1.0, seed }
    }

    pub fn validate(self) -> Result<Self, String> {
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) || !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(format!("采样参数非法: temperature={} top_p={}", self.temperature, self.top_p));
        }
        Ok(self)
    }

    pub const fn is_greedy(self) -> bool {
        self.temperature == 0.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingState {
    config: SamplingConfig,
    state: u64,
}

impl SamplingState {
    pub fn new(config: SamplingConfig) -> Result<Self, String> {
        let config = config.validate()?;
        Ok(Self { config, state: config.seed })
    }

    pub fn next(&mut self) -> TokenSampling {
        if self.config.temperature == 0.0 {
            return TokenSampling { temperature: 0.0, top_p: self.config.top_p, random: 0.0 };
        }
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        let mantissa = (value >> 40) as u32;
        let random = mantissa as f32 * (1.0 / 16_777_216.0);
        TokenSampling { temperature: self.config.temperature, top_p: self.config.top_p, random }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OutputNorm {
    Rms,
    GemmaRms,
}

/// 模型输出尾段(norm + lm_head + 取 token)的公共参数。
/// 各模型 Config 提供一份构造,调用方不再逐个传 eps/norm/excluded。
#[derive(Debug, Clone)]
pub struct OutputPlan {
    pub eps: f32,
    pub norm: OutputNorm,
    pub excluded_tokens: Vec<u32>,
}

#[derive(Clone)]
pub struct OutputHead<W> {
    norm: W,
    lm_head: W,
    norm_f32: bool,
}

impl<W> OutputHead<W> {
    /// 常驻 lm_head 权重;MiniCPM5 异步流水线用它做设备端 embedding gather(tied)。
    pub fn lm_head(&self) -> &W {
        &self.lm_head
    }

    pub(crate) fn lm_head_mut(&mut self) -> &mut W {
        &mut self.lm_head
    }
}

impl<W: Clone> OutputHead<W> {
    #[cfg(all(target_os = "linux", feature = "with-rocm"))]
    pub(crate) fn clone_with_prepared_norm(&self, norm: W, norm_f32: bool) -> Self {
        Self { norm, lm_head: self.lm_head.clone(), norm_f32 }
    }
}

/// FR-Spec 只裁剪 MTP draft 的 LM head。target 仍持有完整 OutputHead，
/// draft 不命中时由 target 原分布纠正，因此不会改变最终采样分布。
pub struct DraftHead<W> {
    lm_head: W,
    token_ids: Vec<u32>,
    inverse_ids: Vec<u32>,
}

impl<W> DraftHead<W> {
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn contains_token(&self, token: u32) -> bool {
        self.inverse_ids.get(token as usize).is_some_and(|&reduced| reduced != u32::MAX)
    }

    fn reduced_fence(&self, fence: &TokenFence) -> TokenFence {
        let forced = fence.forced().and_then(|token| self.inverse_ids.get(token as usize).copied().filter(|&reduced| reduced != u32::MAX));
        TokenFence::constrained(forced, fence.excluded().iter().filter_map(|&token| self.inverse_ids.get(token as usize).copied().filter(|&reduced| reduced != u32::MAX)))
    }

    fn original_token(&self, reduced: u32) -> Result<u32, BackendError> {
        self.token_ids.get(reduced as usize).copied().ok_or_else(|| BackendError::Compute { msg: format!("draft head 返回越界 reduced token={reduced}，vocab={}", self.token_ids.len()) })
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DraftVocabularyFile {
    model: String,
    full_vocab_size: usize,
    token_ids: Vec<u32>,
}

pub fn load_draft_vocabulary(path: &Path, model: &str, full_vocab_size: usize, eos_token_ids: &[u32]) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("读取 draft vocabulary {}: {error}", path.display()))?;
    let vocabulary: DraftVocabularyFile = serde_json::from_slice(&bytes).map_err(|error| format!("解析 draft vocabulary {}: {error}", path.display()))?;
    if vocabulary.model != model || vocabulary.full_vocab_size != full_vocab_size {
        return Err(format!("draft vocabulary 元数据不匹配: model={}/{} full_vocab={}/{}", vocabulary.model, model, vocabulary.full_vocab_size, full_vocab_size));
    }
    if let Some(missing) = eos_token_ids.iter().find(|token| !vocabulary.token_ids.contains(token)) {
        return Err(format!("draft vocabulary 缺少 EOS token={missing}"));
    }
    Ok(vocabulary.token_ids)
}

pub struct OutputResult<T> {
    /// final norm 后、LM Head 前的真实输入；量化校准可直接观测，不复制输出流程。
    pub input: T,
    pub logits: T,
    pub token_id: u32,
}

pub fn prepare_output_head<B>(backend: &B, norm: &[f32], lm_head: LinearWeight<'_>, vocab_size: usize, hidden_size: usize) -> Result<OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    prepare_output_head_quantized(backend, norm, lm_head, vocab_size, hidden_size, LmHeadQuantization::Native)
}

pub fn prepare_output_head_quantized<B>(backend: &B, norm: &[f32], lm_head: LinearWeight<'_>, vocab_size: usize, hidden_size: usize, quantization: LmHeadQuantization) -> Result<OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    prepare_output_head_weight_quantized(backend, LinearWeight::F32(norm), lm_head, vocab_size, hidden_size, quantization)
}

/// 准备使用 GemmaRMSNorm 语义的输出头；norm 传入零中心 gamma。
pub fn prepare_output_head_gemma<B>(backend: &B, norm: &[f32], lm_head: LinearWeight<'_>, vocab_size: usize, hidden_size: usize) -> Result<OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    prepare_output_head_gemma_quantized(backend, norm, lm_head, vocab_size, hidden_size, LmHeadQuantization::Native)
}

pub fn prepare_output_head_gemma_quantized<B>(backend: &B, norm: &[f32], lm_head: LinearWeight<'_>, vocab_size: usize, hidden_size: usize, quantization: LmHeadQuantization) -> Result<OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    let norm = backend.prepare_gemma_f32(norm, 1, hidden_size)?;
    Ok(OutputHead { norm, lm_head: prepare_lm_head_weight(backend, lm_head, vocab_size, hidden_size, quantization)?, norm_f32: true })
}

pub fn prepare_output_head_weight<B>(backend: &B, norm: LinearWeight<'_>, lm_head: LinearWeight<'_>, vocab_size: usize, hidden_size: usize) -> Result<OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    prepare_output_head_weight_quantized(backend, norm, lm_head, vocab_size, hidden_size, LmHeadQuantization::Native)
}

pub fn prepare_output_head_weight_quantized<B>(backend: &B, norm: LinearWeight<'_>, lm_head: LinearWeight<'_>, vocab_size: usize, hidden_size: usize, quantization: LmHeadQuantization) -> Result<OutputHead<B::Weight>, BackendError>
where
    B: Backend,
{
    let (norm, norm_f32) = match norm {
        LinearWeight::F32(values) => (backend.prepare_f32(values, 1, hidden_size)?, true),
        norm => (backend.prepare_weight(norm, 1, hidden_size)?, false),
    };
    Ok(OutputHead { norm, lm_head: prepare_lm_head_weight(backend, lm_head, vocab_size, hidden_size, quantization)?, norm_f32 })
}

/// 把任意标准来源的 LM head 转成统一配置指定的驻留格式。
///
/// 模型 runtime 只负责提供原始矩阵和 shape；格式转换与 backend 准备顺序在这里
/// 唯一实现，避免各模型复制量化逻辑或按平台分支。
pub fn prepare_lm_head_weight<B>(backend: &B, lm_head: LinearWeight<'_>, vocab_size: usize, hidden_size: usize, quantization: LmHeadQuantization) -> Result<B::Weight, BackendError>
where
    B: Backend,
{
    super::prepare_resident_matrix(backend, lm_head, vocab_size, hidden_size, quantization, "lm-head")
}

pub fn prepare_draft_head<B>(backend: &B, lm_head: LinearWeight<'_>, vocab_size: usize, hidden_size: usize, token_ids: Vec<u32>) -> Result<DraftHead<B::Weight>, BackendError>
where
    B: Backend,
{
    if token_ids.is_empty() || token_ids.len() > vocab_size {
        return Err(BackendError::Compute { msg: format!("draft vocabulary 大小非法: {}，full={vocab_size}", token_ids.len()) });
    }
    let mut inverse_ids = vec![u32::MAX; vocab_size];
    for (reduced, &token) in token_ids.iter().enumerate() {
        let inverse = inverse_ids.get_mut(token as usize).ok_or_else(|| BackendError::Compute { msg: format!("draft vocabulary token={token} 超出 full vocab={vocab_size}") })?;
        if *inverse != u32::MAX {
            return Err(BackendError::Compute { msg: format!("draft vocabulary token={token} 重复") });
        }
        *inverse = reduced as u32;
    }
    let lm_head = backend.prepare_weight_rows(lm_head, vocab_size, hidden_size, &token_ids)?;
    Ok(DraftHead { lm_head, token_ids, inverse_ids })
}

pub fn last_token_output<B>(backend: &B, head: &OutputHead<B::Weight>, hidden: &B::Tensor, row: usize, plan: &OutputPlan) -> Result<OutputResult<B::Tensor>, BackendError>
where
    B: Backend,
{
    let hidden = backend.select_row(hidden, row)?;
    token_output(backend, head, &hidden, plan)
}

/// 输出尾段的公共前半程：按 plan 的 norm 语义归一化后过 LM head。
/// token_output / token_ids / sampled_token_ids 共用，避免 4 路 match 逐字重复。
pub(crate) fn norm_and_lm_head<B>(backend: &B, head: &OutputHead<B::Weight>, hidden: &B::Tensor, plan: &OutputPlan) -> Result<(B::Tensor, B::Tensor), BackendError>
where
    B: Backend,
{
    let hidden = match (plan.norm, head.norm_f32) {
        (OutputNorm::Rms, false) => backend.rmsnorm(hidden, &head.norm, plan.eps)?,
        (OutputNorm::GemmaRms, false) => backend.gemma_rmsnorm(hidden, &head.norm, plan.eps)?,
        (OutputNorm::Rms, true) => backend.rmsnorm_f32(hidden, &head.norm, plan.eps)?,
        (OutputNorm::GemmaRms, true) => backend.gemma_rmsnorm_f32(hidden, &head.norm, plan.eps)?,
    };
    let logits = backend.linear(&hidden, &head.lm_head)?;
    Ok((hidden, logits))
}

pub fn token_output<B>(backend: &B, head: &OutputHead<B::Weight>, hidden: &B::Tensor, plan: &OutputPlan) -> Result<OutputResult<B::Tensor>, BackendError>
where
    B: Backend,
{
    let (hidden, logits) = norm_and_lm_head(backend, head, hidden, plan)?;
    let token_id = backend.argmax_excluding(&logits, &plan.excluded_tokens)?;
    Ok(OutputResult { input: hidden, logits, token_id })
}

/// 多个已完成 session 共用一次 output norm / LM head，再逐行取 token。
pub fn token_ids<B>(backend: &B, head: &OutputHead<B::Weight>, hidden: &B::Tensor, plan: &OutputPlan) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    let rows = backend.token_rows(hidden);
    if rows == 0 {
        return Err(BackendError::Compute { msg: "output token batch 不能为空".to_owned() });
    }
    let (_, logits) = norm_and_lm_head(backend, head, hidden, plan)?;
    backend.argmax_rows_excluding(&logits, &plan.excluded_tokens)
}

/// 多个 session 共用 output norm / LM head，再按各自策略逐行采样。
pub fn sampled_token_ids<B>(backend: &B, head: &OutputHead<B::Weight>, hidden: &B::Tensor, plan: &OutputPlan, sampling: &[TokenSampling]) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    let rows = backend.token_rows(hidden);
    if rows == 0 || rows != sampling.len() {
        return Err(BackendError::Compute { msg: format!("output sampling shape 不兼容: rows={rows} sampling={}", sampling.len()) });
    }
    let (_, logits) = norm_and_lm_head(backend, head, hidden, plan)?;
    backend.sample_rows_excluding(&logits, sampling, &plan.excluded_tokens)
}

/// 多个 session 共用 output head，但每行可应用独立生成围栏。
pub fn sampled_token_ids_fenced<B>(backend: &B, head: &OutputHead<B::Weight>, hidden: &B::Tensor, plan: &OutputPlan, sampling: &[TokenSampling], fences: &[TokenFence]) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    let rows = backend.token_rows(hidden);
    if rows == 0 || rows != sampling.len() || rows != fences.len() {
        return Err(BackendError::Compute { msg: format!("output fenced sampling shape 不兼容: rows={rows} sampling={} fences={}", sampling.len(), fences.len()) });
    }
    let (_, logits) = norm_and_lm_head(backend, head, hidden, plan)?;
    if plan.excluded_tokens.is_empty() {
        backend.sample_rows_fenced(&logits, sampling, fences)
    } else {
        let fences = fences.iter().map(|fence| TokenFence::constrained(fence.forced(), plan.excluded_tokens.iter().copied().chain(fence.excluded().iter().copied()))).collect::<Vec<_>>();
        backend.sample_rows_fenced(&logits, sampling, &fences)
    }
}

/// 对已完成模型专用 output norm 的 hidden 复用同一 LM head。
pub fn normalized_token_id<B>(backend: &B, head: &OutputHead<B::Weight>, hidden: &B::Tensor, excluded_tokens: &[u32]) -> Result<u32, BackendError>
where
    B: Backend,
{
    let logits = backend.linear(hidden, &head.lm_head)?;
    backend.argmax_excluding(&logits, excluded_tokens)
}

/// 多个已完成模型专用 output norm 的 session 共用一次 LM head。
pub fn normalized_token_ids<B>(backend: &B, head: &OutputHead<B::Weight>, hidden: &B::Tensor, excluded_tokens: &[u32]) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    if backend.token_rows(hidden) == 0 {
        return Err(BackendError::Compute { msg: "normalized output batch 不能为空".to_owned() });
    }
    let logits = backend.linear(hidden, &head.lm_head)?;
    backend.argmax_rows_excluding(&logits, excluded_tokens)
}

pub fn normalized_token_ids_fenced<B>(backend: &B, head: &OutputHead<B::Weight>, hidden: &B::Tensor, fences: &[TokenFence]) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    if backend.token_rows(hidden) == 0 || backend.token_rows(hidden) != fences.len() {
        return Err(BackendError::Compute { msg: format!("normalized fenced output shape 不兼容: rows={} fences={}", backend.token_rows(hidden), fences.len()) });
    }
    let logits = backend.linear(hidden, &head.lm_head)?;
    backend.argmax_rows_fenced(&logits, fences)
}

pub fn normalized_draft_token_id<B>(backend: &B, head: &DraftHead<B::Weight>, hidden: &B::Tensor, excluded_tokens: &[u32]) -> Result<u32, BackendError>
where
    B: Backend,
{
    let logits = backend.linear(hidden, &head.lm_head)?;
    let fence = head.reduced_fence(&TokenFence::excluding(excluded_tokens.iter().copied()));
    head.original_token(backend.argmax_excluding(&logits, fence.excluded())?)
}

pub fn normalized_draft_token_ids_fenced<B>(backend: &B, head: &DraftHead<B::Weight>, hidden: &B::Tensor, fences: &[TokenFence]) -> Result<Vec<u32>, BackendError>
where
    B: SegmentedTensorBackend,
{
    if backend.token_rows(hidden) == 0 || backend.token_rows(hidden) != fences.len() {
        return Err(BackendError::Compute { msg: format!("normalized draft fenced shape 不兼容: rows={} fences={}", backend.token_rows(hidden), fences.len()) });
    }
    let logits = backend.linear(hidden, &head.lm_head)?;
    let fences = fences.iter().map(|fence| head.reduced_fence(fence)).collect::<Vec<_>>();
    backend.argmax_rows_fenced(&logits, &fences)?.into_iter().map(|token| head.original_token(token)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::cpu::CpuContext, kernel::cpu::CpuTensor};

    #[test]
    fn batched_token_ids_match_row_outputs() -> Result<(), BackendError> {
        let backend = CpuContext;
        let head = prepare_output_head(&backend, &[1.0, 0.5, 2.0], LinearWeight::F32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0]), 4, 3)?;
        let hidden = CpuTensor { data: vec![4.0, 1.0, 0.0, 0.0, 4.0, 1.0, -2.0, -1.0, -4.0], rows: 3, cols: 3 };
        let excluded = [2];
        let plan = OutputPlan { eps: 1.0e-6, norm: OutputNorm::Rms, excluded_tokens: excluded.to_vec() };
        let batched = token_ids(&backend, &head, &hidden, &plan)?;
        let row_outputs = (0..hidden.rows).map(|row| last_token_output(&backend, &head, &hidden, row, &plan).map(|output| output.token_id)).collect::<Result<Vec<_>, _>>()?;

        assert_eq!(batched, row_outputs);
        let normalized = normalized_token_ids(&backend, &head, &hidden, &excluded)?;
        let normalized_rows = (0..hidden.rows).map(|row| backend.slice_token_rows(&hidden, row, 1).and_then(|hidden| normalized_token_id(&backend, &head, &hidden, &excluded))).collect::<Result<Vec<_>, _>>()?;
        assert_eq!(normalized, normalized_rows);
        Ok(())
    }

    #[test]
    fn q8_lm_head_uses_common_backend_path() -> Result<(), BackendError> {
        let backend = CpuContext;
        let mut lm_head = vec![0.0_f32; 2 * 128];
        lm_head[0] = 2.0;
        lm_head[128] = -2.0;
        let head = prepare_output_head_quantized(&backend, &vec![1.0; 128], LinearWeight::F32(&lm_head), 2, 128, LmHeadQuantization::Q8g128)?;
        let mut hidden = vec![0.0; 128];
        hidden[0] = 4.0;
        let output = token_output(&backend, &head, &CpuTensor { data: hidden, rows: 1, cols: 128 }, &OutputPlan { eps: 1.0e-6, norm: OutputNorm::Rms, excluded_tokens: Vec::new() })?;

        assert_eq!(output.token_id, 0);
        Ok(())
    }

    #[test]
    fn sampling_state_is_reproducible_and_bounded() {
        let config = SamplingConfig { temperature: 0.6, top_p: 0.95, seed: 42 };
        let mut first = SamplingState::new(config).unwrap();
        let mut second = SamplingState::new(config).unwrap();
        let values = (0..16).map(|_| first.next()).collect::<Vec<_>>();
        assert_eq!(values, (0..16).map(|_| second.next()).collect::<Vec<_>>());
        assert!(values.iter().all(|value| (0.0..1.0).contains(&value.random)));
        assert!(values.windows(2).any(|pair| pair[0].random != pair[1].random));
    }

    #[test]
    fn batched_sampling_matches_individual_rows() -> Result<(), BackendError> {
        let backend = CpuContext;
        let head = prepare_output_head(&backend, &[1.0, 0.5, 2.0], LinearWeight::F32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0]), 4, 3)?;
        let hidden = CpuTensor { data: vec![4.0, 1.0, 0.0, 0.0, 4.0, 1.0, -2.0, -1.0, -4.0], rows: 3, cols: 3 };
        let sampling = [TokenSampling { temperature: 0.6, top_p: 0.95, random: 0.1 }, TokenSampling { temperature: 0.0, top_p: 1.0, random: 0.0 }, TokenSampling { temperature: 1.0, top_p: 0.8, random: 0.9 }];
        let excluded = [2];
        let plan = OutputPlan { eps: 1.0e-6, norm: OutputNorm::Rms, excluded_tokens: excluded.to_vec() };
        let batched = sampled_token_ids(&backend, &head, &hidden, &plan, &sampling)?;
        let individual = (0..hidden.rows)
            .map(|row| {
                let hidden = backend.slice_token_rows(&hidden, row, 1)?;
                sampled_token_ids(&backend, &head, &hidden, &plan, &sampling[row..row + 1]).map(|tokens| tokens[0])
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        assert_eq!(batched, individual);
        Ok(())
    }

    #[test]
    fn fenced_sampling_applies_each_row_independently() -> Result<(), BackendError> {
        let backend = CpuContext;
        let head = prepare_output_head(&backend, &[1.0, 1.0], LinearWeight::F32(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0]), 3, 2)?;
        let hidden = CpuTensor { data: vec![3.0, 1.0, 3.0, 1.0], rows: 2, cols: 2 };
        let sampling = [TokenSampling { temperature: 0.0, top_p: 1.0, random: 0.0 }; 2];
        let fences = [TokenFence::default(), TokenFence::excluding([0])];
        let tokens = sampled_token_ids_fenced(&backend, &head, &hidden, &OutputPlan { eps: 1.0e-6, norm: OutputNorm::Rms, excluded_tokens: Vec::new() }, &sampling, &fences)?;
        assert_eq!(tokens, vec![0, 1]);
        Ok(())
    }

    #[test]
    fn draft_head_maps_tokens_and_fences_back_to_full_vocabulary() -> Result<(), BackendError> {
        let backend = CpuContext;
        let full = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let head = prepare_draft_head(&backend, LinearWeight::F32(&full), 4, 2, vec![3, 0, 2])?;
        assert!(head.contains_token(0));
        assert!(!head.contains_token(1));
        let hidden = CpuTensor { data: vec![0.0, 1.0, 1.0, 0.0], rows: 2, cols: 2 };
        let tokens = normalized_draft_token_ids_fenced(&backend, &head, &hidden, &[TokenFence::excluding([3]), TokenFence::excluding([2])])?;
        assert_eq!(tokens, vec![0, 0]);
        Ok(())
    }
}
