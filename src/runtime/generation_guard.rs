//! 请求级生成围栏：在采样前阻断已知后继，并在 token 提交前识别失控循环。
//!
//! 本模块不解释模型协议。调用方决定检测到循环后是结束请求，还是用模型自身的
//! reasoning-end token 收口；speculative 路径通过 clone 后逐 token 推进临时状态。

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::fmt;
use std::sync::Arc;

use crate::backend::TokenFence;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
use crate::tokenizer::Detokenizer;

const HISTORY_WINDOW: usize = 4096;
const NUMBER_SIGNATURE: u64 = 0x9e37_79b9_7f4a_7c15;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
const SPACE_SIGNATURE: u64 = 0x517c_c1b7_2722_0a95;

/// 请求级生成围栏程序。实现只负责从语法状态生成禁止 token 集合，并在最终
/// token 提交后推进状态；模型 runtime 与 backend 不感知具体协议。
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub trait TokenFenceProgram {
    fn fence(&self) -> TokenFence;
    fn advance(&mut self, token: u32);
}

impl<P: TokenFenceProgram> TokenFenceProgram for Option<P> {
    fn fence(&self) -> TokenFence {
        self.as_ref().map(TokenFenceProgram::fence).unwrap_or_default()
    }

    fn advance(&mut self, token: u32) {
        if let Some(program) = self.as_mut() {
            program.advance(token);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopKind {
    RepeatedToken,
    RepeatedPattern,
    SemanticRestart,
}

impl fmt::Display for LoopKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::RepeatedToken => "repeated_token",
            Self::RepeatedPattern => "repeated_pattern",
            Self::SemanticRestart => "semantic_restart",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopRecovery {
    /// speculative verify 应保留的行数。
    pub retained_rows: usize,
    pub kind: LoopKind,
    /// true 表示最后一个输出已改写为调用方给出的 reasoning-end token。
    pub forced_end: bool,
}

/// 通用请求级生成围栏。精确重复使用原 token，语义重启检测使用调用方预先构造的
/// token 文本签名；未提供签名时仍保留全部精确检测能力。
#[derive(Clone)]
pub struct GenerationGuard<P> {
    inner: P,
    enabled: bool,
    protected: Vec<u32>,
    segment_end_tokens: Vec<u32>,
    restart_markers: Vec<u32>,
    semantic_restart_sequences: Vec<Vec<u32>>,
    semantic_restart_positions: Vec<usize>,
    signatures: Option<Arc<[u64]>>,
    /// 已经出现过的 segment 边界 token(如 reasoning-end)。它再次被采样只会
    /// 制造未配对的协议标签,进入排除集;围栏与预算强制收口直接改写候选,
    /// 不经过采样,不受影响。
    closed_boundaries: Vec<u32>,
    tokens: Vec<u32>,
    normalized: Vec<u64>,
}

impl<P> GenerationGuard<P> {
    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    pub fn new(inner: P, enabled: bool, protected: impl IntoIterator<Item = u32>) -> Self {
        let mut protected = protected.into_iter().collect::<Vec<_>>();
        protected.sort_unstable();
        protected.dedup();
        Self {
            inner,
            enabled,
            protected,
            segment_end_tokens: Vec::new(),
            restart_markers: Vec::new(),
            semantic_restart_sequences: Vec::new(),
            semantic_restart_positions: Vec::new(),
            signatures: None,
            closed_boundaries: Vec::new(),
            tokens: Vec::new(),
            normalized: Vec::new(),
        }
    }

    /// marker 和 token 文本签名由模型 runtime 在请求边界提供；通用生成层不保存
    /// 模型专有 token id 或词表规则。
    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    pub fn with_restart_markers(mut self, markers: impl IntoIterator<Item = u32>) -> Self {
        self.restart_markers.extend(markers);
        self.restart_markers.sort_unstable();
        self.restart_markers.dedup();
        self.semantic_restart_sequences.extend(self.restart_markers.iter().map(|&token| vec![token]));
        self.semantic_restart_sequences.sort_unstable();
        self.semantic_restart_sequences.dedup();
        self
    }

    /// 宽泛的“If there / Could it be”等句式只能作为组合证据，不能直接进入禁止
    /// token 集合，否则会破坏正常的证明和反例分析。
    #[cfg(all(target_os = "linux", feature = "with-rocm"))]
    pub fn with_semantic_restart_sequences(mut self, sequences: impl IntoIterator<Item = Vec<u32>>) -> Self {
        self.semantic_restart_sequences.extend(sequences.into_iter().filter(|sequence| !sequence.is_empty()));
        self.semantic_restart_sequences.sort_unstable();
        self.semantic_restart_sequences.dedup();
        self
    }

    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    pub fn with_token_signatures(mut self, signatures: Arc<[u64]>) -> Self {
        self.signatures = Some(signatures);
        self
    }

    /// reasoning-end 等协议边界会清空本段历史，避免 thinking 中的重复证据污染
    /// 最终答案；inner 协议状态仍照常推进。
    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    pub fn with_segment_end_tokens(mut self, tokens: impl IntoIterator<Item = u32>) -> Self {
        self.segment_end_tokens.extend(tokens);
        self.segment_end_tokens.sort_unstable();
        self.segment_end_tokens.dedup();
        self
    }

    /// 在候选 token 正式提交前逐行模拟围栏。thinking 中把触发循环的最后一行
    /// 改成 reasoning-end；最终答案没有 end token 时保留触发行并让调用方硬停止。
    pub fn recover(&self, candidates: &mut Vec<u32>, reasoning_end: Option<u32>) -> Option<LoopRecovery>
    where
        P: TokenFenceProgram + Clone,
    {
        if !self.enabled {
            return None;
        }
        let mut probe = self.clone();
        let mut reasoning_end = reasoning_end;
        for index in 0..candidates.len() {
            probe.advance(candidates[index]);
            if reasoning_end == Some(candidates[index]) {
                reasoning_end = None;
                continue;
            }
            let Some(kind) = probe.detected_loop() else { continue };
            candidates.truncate(index + 1);
            if let Some(end) = reasoning_end {
                candidates[index] = end;
            }
            return Some(LoopRecovery { retained_rows: index + 1, kind, forced_end: reasoning_end.is_some() });
        }
        None
    }

    fn signature(&self, token: u32) -> u64 {
        self.signatures.as_ref().and_then(|signatures| signatures.get(token as usize)).copied().unwrap_or(u64::from(token))
    }

    fn detected_loop(&self) -> Option<LoopKind> {
        if repeated_tail(&self.tokens, 1, 10) {
            return Some(LoopKind::RepeatedToken);
        }
        if (2..=256).any(|width| repeated_tail(&self.tokens, width, 5)) {
            return Some(LoopKind::RepeatedPattern);
        }
        self.semantic_restart_loop().then_some(LoopKind::SemanticRestart)
    }

    fn semantic_restart_loop(&self) -> bool {
        let window_start = self.tokens.len().saturating_sub(HISTORY_WINDOW);
        if self.signatures.is_none() || self.semantic_restart_positions.iter().rev().take_while(|&&position| position >= window_start).count() < 8 {
            return false;
        }
        [(24, 2), (12, 3), (8, 4)].into_iter().any(|(width, prior)| repeated_suffix_anchor(&self.normalized, width, prior).is_some())
    }

    fn restart_count(&self) -> usize {
        self.tokens.iter().rev().take(HISTORY_WINDOW).filter(|token| self.restart_markers.binary_search(token).is_ok()).count()
    }

    fn repeated_next(&self) -> Option<u32> {
        let len = self.tokens.len();
        if len >= 9 && self.tokens[len - 9..].iter().all(|&token| token == self.tokens[len - 1]) {
            return self.tokens.last().copied();
        }
        let contiguous = (2..=(len / 4).min(256)).find_map(|width| repeated_tail(&self.tokens, width, 4).then_some(self.tokens[len - width]));
        contiguous.or_else(|| self.repeated_anchor_next())
    }

    fn repeated_anchor_next(&self) -> Option<u32> {
        for (width, prior_occurrences) in [(24, 3), (12, 4), (8, 5)] {
            if let Some(start) = repeated_suffix_anchor(&self.tokens, width, prior_occurrences) {
                let suffix_start = self.tokens.len() - width;
                let suffix = &self.tokens[suffix_start..];
                let mut next = None;
                let mut occurrences = 0;
                let mut previous_end = 0;
                for candidate_start in start..=suffix_start - width {
                    if candidate_start < previous_end || &self.tokens[candidate_start..candidate_start + width] != suffix {
                        continue;
                    }
                    let candidate = self.tokens[candidate_start + width];
                    if next.is_some_and(|next| next != candidate) {
                        break;
                    }
                    next = Some(candidate);
                    occurrences += 1;
                    previous_end = candidate_start + width;
                    if occurrences == prior_occurrences {
                        return next;
                    }
                }
            }
        }
        None
    }

    fn repeated_restart_markers(&self) -> impl Iterator<Item = u32> + '_ {
        let repeated = self.restart_count() >= 12;
        self.restart_markers.iter().copied().filter(move |token| repeated && self.protected.binary_search(token).is_err())
    }
}

impl<P: TokenFenceProgram> TokenFenceProgram for GenerationGuard<P> {
    fn fence(&self) -> TokenFence {
        let inner = self.inner.fence();
        let repeated = self.enabled.then(|| self.repeated_next()).flatten().filter(|token| self.protected.binary_search(token).is_err());
        let restart_markers = self.enabled.then(|| self.repeated_restart_markers()).into_iter().flatten();
        TokenFence::constrained(inner.forced(), inner.excluded().iter().copied().chain(repeated).chain(restart_markers).chain(self.closed_boundaries.iter().copied()))
    }

    fn advance(&mut self, token: u32) {
        self.inner.advance(token);
        if self.segment_end_tokens.binary_search(&token).is_ok() {
            if let Err(position) = self.closed_boundaries.binary_search(&token) {
                self.closed_boundaries.insert(position, token);
            }
            self.tokens.clear();
            self.normalized.clear();
            self.semantic_restart_positions.clear();
            return;
        }
        self.tokens.push(token);
        self.normalized.push(self.signature(token));
        if self.semantic_restart_sequences.iter().any(|sequence| self.tokens.ends_with(sequence)) {
            self.semantic_restart_positions.push(self.tokens.len());
        }
    }
}

fn repeated_tail<T: PartialEq>(tokens: &[T], width: usize, count: usize) -> bool {
    let Some(total) = width.checked_mul(count) else { return false };
    if width == 0 || tokens.len() < total {
        return false;
    }
    let suffix = &tokens[tokens.len() - width..];
    (2..=count).all(|offset| suffix == &tokens[tokens.len() - offset * width..tokens.len() - (offset - 1) * width])
}

/// 返回扫描窗口起点；调用方可继续检查各次锚点的后继 token。
fn repeated_suffix_anchor<T: PartialEq>(tokens: &[T], width: usize, prior_occurrences: usize) -> Option<usize> {
    let len = tokens.len();
    if width == 0 || len < width * (prior_occurrences + 1) {
        return None;
    }
    let suffix_start = len - width;
    let suffix = &tokens[suffix_start..];
    let start = suffix_start.saturating_sub(HISTORY_WINDOW);
    let mut occurrences = 0;
    let mut previous_end = 0;
    for candidate_start in start..=suffix_start - width {
        if candidate_start < previous_end || &tokens[candidate_start..candidate_start + width] != suffix {
            continue;
        }
        occurrences += 1;
        previous_end = candidate_start + width;
        if occurrences == prior_occurrences {
            return Some(start);
        }
    }
    None
}

/// 为语义重启检测构造稳定签名。数字片段折叠为同一值，空白折叠为同一值，
/// 普通文本只做 ASCII 大小写归一化；构造发生在模型启动期，不进入 decode 热路径。
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub fn build_token_signatures(detokenizer: &Detokenizer, vocab_size: usize) -> Arc<[u64]> {
    (0..vocab_size)
        .map(|id| {
            let Ok(bytes) = detokenizer.decode_bytes(&[id as u32], false) else { return id as u64 };
            text_signature(&bytes, id as u64)
        })
        .collect::<Vec<_>>()
        .into()
}

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
fn text_signature(bytes: &[u8], fallback: u64) -> u64 {
    let start = bytes.iter().position(|byte| !byte.is_ascii_whitespace()).unwrap_or(bytes.len());
    let end = bytes.iter().rposition(|byte| !byte.is_ascii_whitespace()).map_or(start, |index| index + 1);
    let trimmed = &bytes[start..end];
    if trimmed.is_empty() {
        return SPACE_SIGNATURE;
    }
    let numeric = trimmed.iter().any(u8::is_ascii_digit) && trimmed.iter().all(|byte| byte.is_ascii_digit() || byte.is_ascii_whitespace() || b".,+-eExX^*/()%_:[]{}".contains(byte));
    if numeric {
        return NUMBER_SIGNATURE;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes.iter().map(u8::to_ascii_lowercase) {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
    }
    if hash == NUMBER_SIGNATURE || hash == SPACE_SIGNATURE { fallback } else { hash }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct OpenFence;

    impl TokenFenceProgram for OpenFence {
        fn fence(&self) -> TokenFence {
            TokenFence::default()
        }

        fn advance(&mut self, _token: u32) {}
    }

    #[test]
    fn optional_program_is_an_open_passthrough_when_absent() {
        let mut program: Option<OpenFence> = None;
        assert!(program.fence().is_open());
        program.advance(7);
        assert!(program.fence().is_open());
    }

    #[test]
    fn exact_pattern_softly_excludes_only_the_continuation() {
        let mut guard = GenerationGuard::new(OpenFence, true, []);
        for token in [10, 20, 10, 20, 10, 20, 10, 20] {
            guard.advance(token);
        }
        assert_eq!(guard.fence().excluded(), &[10]);
        guard.advance(30);
        assert!(guard.fence().is_open());
    }

    #[test]
    fn single_token_run_is_stopped_before_tenth_copy() {
        let mut guard = GenerationGuard::new(OpenFence, true, []);
        for _ in 0..9 {
            guard.advance(7);
        }
        assert_eq!(guard.fence().excluded(), &[7]);

        let mut candidates = vec![7];
        assert_eq!(guard.recover(&mut candidates, Some(99)), Some(LoopRecovery { retained_rows: 1, kind: LoopKind::RepeatedToken, forced_end: true }));
        assert_eq!(candidates, [99]);
    }

    #[test]
    fn protected_continuation_remains_available() {
        let mut guard = GenerationGuard::new(OpenFence, true, [10]);
        for token in [10, 20, 10, 20, 10, 20] {
            guard.advance(token);
        }
        assert!(guard.fence().is_open());
    }

    #[test]
    fn final_answer_loop_reports_hard_recovery_without_rewrite() {
        let mut guard = GenerationGuard::new(OpenFence, true, []);
        for token in [10, 20, 10, 20, 10, 20, 10, 20] {
            guard.advance(token);
        }
        let mut candidates = vec![10, 20];
        assert_eq!(guard.recover(&mut candidates, None), Some(LoopRecovery { retained_rows: 2, kind: LoopKind::RepeatedPattern, forced_end: false }));
        assert_eq!(candidates, [10, 20]);
    }

    #[test]
    fn 三份长结构块不触发循环保护() {
        let mut guard = GenerationGuard::new(OpenFence, true, []);
        let block = (100..112).collect::<Vec<_>>();
        for token in block.iter().cycle().take(block.len() * 3) {
            guard.advance(*token);
        }
        assert!(guard.fence().is_open());

        let mut candidates = vec![300];
        assert_eq!(guard.recover(&mut candidates, None), None);
        assert_eq!(candidates, [300]);
    }

    #[test]
    fn normalized_restart_loop_closes_thinking() {
        let mut signatures = (0_u64..512).collect::<Vec<_>>();
        signatures[300] = NUMBER_SIGNATURE;
        signatures[301] = NUMBER_SIGNATURE;
        let mut guard = GenerationGuard::new(OpenFence, true, []).with_restart_markers([70]).with_token_signatures(signatures.into());
        let anchors = [[100, 101, 300, 103, 104, 105, 106, 107], [100, 101, 301, 103, 104, 105, 106, 107]];
        for index in 0_usize..4 {
            guard.advance(70);
            guard.advance(200 + index as u32);
            guard.advance(70);
            for token in anchors[index % 2] {
                guard.advance(token);
            }
        }
        guard.advance(70);
        guard.advance(250);
        guard.advance(70);
        let mut candidates = anchors[1].to_vec();
        let recovery = guard.recover(&mut candidates, Some(99)).unwrap();
        assert_eq!(recovery.kind, LoopKind::SemanticRestart);
        assert!(recovery.forced_end);
        assert_eq!(candidates.last(), Some(&99));
    }

    #[test]
    fn segment_end_clears_reasoning_history() {
        let mut guard = GenerationGuard::new(OpenFence, true, []).with_segment_end_tokens([99]);
        for token in [10, 20, 10, 20, 10, 20, 99] {
            guard.advance(token);
        }
        // 段历史已清空(循环证据不残留),边界 token 自身进入排除集防止复发
        assert_eq!(guard.fence().excluded(), &[99]);
        let mut candidates = vec![30];
        assert_eq!(guard.recover(&mut candidates, None), None);
    }

    #[test]
    fn segment_end_token_is_banned_after_first_occurrence() {
        // 首次出现必须放行(server 靠它切分 reasoning),之后进入排除集,
        // 防止答案阶段再采样出未配对的 </think>。
        let mut guard = GenerationGuard::new(OpenFence, true, []).with_segment_end_tokens([99]);
        assert!(guard.fence().is_open());
        guard.advance(98);
        assert!(guard.fence().is_open());
        guard.advance(99);
        assert!(guard.fence().excluded().contains(&99));
    }

    #[test]
    fn guard_can_be_disabled() {
        let mut guard = GenerationGuard::new(OpenFence, false, []);
        for _ in 0..12 {
            guard.advance(7);
        }
        assert!(guard.fence().is_open());
        let mut candidates = vec![7];
        assert_eq!(guard.recover(&mut candidates, Some(99)), None);
    }
}
