//! 模型无关的推测解码候选、验证与统计；drafter 网络和 cache 留在具体模型。

use crate::backend::{BackendError, SpeculativeCacheCommit};
pub use crate::speculative::{BlockDraftSpec, HiddenStateCapturePlan};

#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeBlock {
    pub anchor: u32,
    pub drafts: Vec<u32>,
    pub confidences: Vec<f32>,
}

impl SpeculativeBlock {
    pub fn new(anchor: u32, drafts: Vec<u32>, confidences: Vec<f32>) -> Result<Self, BackendError> {
        if !confidences.is_empty() && confidences.len() != drafts.len() {
            return Err(BackendError::Compute { msg: format!("推测块 confidence={} 与 drafts={} 数量不一致", confidences.len(), drafts.len()) });
        }
        if confidences.iter().any(|confidence| !confidence.is_finite() || !(0.0..=1.0).contains(confidence)) {
            return Err(BackendError::Compute { msg: "推测块 confidence 必须是 [0,1] 内的有限值".to_owned() });
        }
        Ok(Self { anchor, drafts, confidences })
    }

    /// 保留首个低置信度候选之前的连续前缀；threshold=0 等价于固定整块验证。
    pub fn retain_confident_prefix(self, threshold: f32) -> Result<Self, BackendError> {
        self.retain_confident_prefix_at_least(threshold, 0)
    }

    /// confidence 前缀不足时至少保留 `minimum_drafts` 行，供流水线按真实缺口补足
    /// target 工作；最低行数由模型执行器决定，confidence 规则仍负责其余候选。
    pub fn retain_confident_prefix_at_least(mut self, threshold: f32, minimum_drafts: usize) -> Result<Self, BackendError> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(BackendError::Compute { msg: format!("推测块 confidence threshold={threshold} 非法") });
        }
        if minimum_drafts > self.drafts.len() {
            return Err(BackendError::Compute { msg: format!("推测块 minimum_drafts={minimum_drafts} 超过 drafts={}", self.drafts.len()) });
        }
        let confident = self.confidences.iter().position(|&confidence| confidence < threshold).unwrap_or(self.drafts.len());
        let keep = confident.max(minimum_drafts);
        self.drafts.truncate(keep);
        self.confidences.truncate(keep);
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeVerification {
    pub tokens: Vec<u32>,
    pub accepted_drafts: usize,
    pub retained_rows: usize,
    pub eos: bool,
}

impl SpeculativeVerification {
    /// verify 的最后一个输出是下一轮尚未前向的 pending token；此前输出可立即提交。
    pub fn pending_token(&self) -> u32 {
        *self.tokens.last().expect("推测验证必须至少产出一个 token")
    }

    pub fn emitted_tokens(&self) -> &[u32] {
        &self.tokens[..self.tokens.len() - 1]
    }

    pub const fn cache_commit(&self) -> SpeculativeCacheCommit {
        SpeculativeCacheCommit::retaining(self.retained_rows)
    }

    /// verifier 从 `position` 写入 `retained_rows` 行后，cache 应提交到的绝对长度。
    pub fn cache_end(&self, position: usize) -> Result<usize, BackendError> {
        position.checked_add(self.retained_rows).ok_or_else(|| BackendError::Compute { msg: format!("推测验证 cache end 溢出: position={position} retained_rows={}", self.retained_rows) })
    }
}

/// target 按请求参数采样每个草稿位置；相等时接受草稿，首个失配直接提交
/// target 样本，全命中时追加 target bonus。greedy 是同一算法的退化情况。
pub fn verify_samples(target_tokens: &[u32], drafts: &[u32], eos_tokens: &[u32]) -> Result<SpeculativeVerification, BackendError> {
    if target_tokens.len() != drafts.len().saturating_add(1) {
        return Err(BackendError::Compute { msg: format!("推测验证 shape 非法: target={} drafts={}", target_tokens.len(), drafts.len()) });
    }
    let mut tokens = Vec::with_capacity(target_tokens.len());
    let mut accepted_drafts = 0;
    for (index, &draft) in drafts.iter().enumerate() {
        let target = target_tokens[index];
        let accepted = draft == target;
        accepted_drafts += usize::from(accepted);
        let token = if accepted { draft } else { target };
        tokens.push(token);
        if eos_tokens.contains(&token) {
            return Ok(SpeculativeVerification { accepted_drafts, retained_rows: index + 1, tokens, eos: true });
        }
        if !accepted {
            return Ok(SpeculativeVerification { accepted_drafts, retained_rows: index + 1, tokens, eos: false });
        }
    }
    let bonus = target_tokens[drafts.len()];
    tokens.push(bonus);
    Ok(SpeculativeVerification { accepted_drafts, retained_rows: target_tokens.len(), eos: eos_tokens.contains(&bonus), tokens })
}

/// 验证推测块的非终局前缀。这里每个 target 输出都对应一个尚未执行的 draft，
/// 全部命中时不追加 bonus，调用方可以继续提交同一推测块的下一段。
pub fn verify_samples_prefix(target_tokens: &[u32], drafts: &[u32], eos_tokens: &[u32]) -> Result<SpeculativeVerification, BackendError> {
    if target_tokens.len() != drafts.len() {
        return Err(BackendError::Compute { msg: format!("推测前缀验证 shape 非法: target={} drafts={}", target_tokens.len(), drafts.len()) });
    }
    let mut tokens = Vec::with_capacity(target_tokens.len());
    let mut accepted_drafts = 0;
    for (index, (&target, &draft)) in target_tokens.iter().zip(drafts).enumerate() {
        let accepted = draft == target;
        accepted_drafts += usize::from(accepted);
        let token = if accepted { draft } else { target };
        tokens.push(token);
        if eos_tokens.contains(&token) {
            return Ok(SpeculativeVerification { accepted_drafts, retained_rows: index + 1, tokens, eos: true });
        }
        if !accepted {
            return Ok(SpeculativeVerification { accepted_drafts, retained_rows: index + 1, tokens, eos: false });
        }
    }
    Ok(SpeculativeVerification { accepted_drafts, retained_rows: target_tokens.len(), tokens, eos: false })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub rounds: u64,
    pub proposed: u64,
    pub accepted: u64,
    pub emitted: u64,
    pub target_forwards: u64,
}

impl SpeculativeStats {
    pub fn record(&mut self, proposed: usize, verification: &SpeculativeVerification) {
        self.rounds = self.rounds.saturating_add(1);
        self.proposed = self.proposed.saturating_add(proposed as u64);
        self.accepted = self.accepted.saturating_add(verification.accepted_drafts as u64);
        self.emitted = self.emitted.saturating_add(verification.tokens.len() as u64);
        self.target_forwards = self.target_forwards.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_stop_at_first_mismatch() {
        let result = verify_samples(&[10, 21, 30, 40], &[10, 20, 30], &[]).unwrap();
        assert_eq!(result.tokens, [10, 21]);
        assert_eq!(result.accepted_drafts, 1);
        assert_eq!(result.retained_rows, 2);
        assert!(!result.eos);
    }

    #[test]
    fn samples_all_accept_add_bonus() {
        let result = verify_samples(&[10, 20, 30, 40], &[10, 20, 30], &[]).unwrap();
        assert_eq!(result.tokens, [10, 20, 30, 40]);
        assert_eq!(result.accepted_drafts, 3);
        assert_eq!(result.retained_rows, 4);
    }

    #[test]
    fn prefix_accepts_all_without_bonus() {
        let result = verify_samples_prefix(&[10, 20, 30], &[10, 20, 30], &[]).unwrap();
        assert_eq!(result.tokens, [10, 20, 30]);
        assert_eq!(result.accepted_drafts, 3);
        assert_eq!(result.retained_rows, 3);
        assert!(!result.eos);
    }

    #[test]
    fn prefix_stops_at_first_mismatch() {
        let result = verify_samples_prefix(&[10, 21, 30], &[10, 20, 30], &[]).unwrap();
        assert_eq!(result.tokens, [10, 21]);
        assert_eq!(result.accepted_drafts, 1);
        assert_eq!(result.retained_rows, 2);
        assert!(!result.eos);
    }

    #[test]
    fn samples_eos_commits_only_visible_prefix() {
        let result = verify_samples(&[10, 20, 30], &[10, 20], &[20]).unwrap();
        assert_eq!(result.tokens, [10, 20]);
        assert_eq!(result.accepted_drafts, 2);
        assert_eq!(result.retained_rows, 2);
        assert!(result.eos);
    }

    #[test]
    fn block_rejects_misaligned_confidence() {
        assert!(SpeculativeBlock::new(1, vec![2, 3], vec![0.5]).is_err());
        assert!(SpeculativeBlock::new(1, vec![2], vec![f32::NAN]).is_err());
    }

    #[test]
    fn block_keeps_contiguous_confident_prefix() {
        let block = SpeculativeBlock::new(1, vec![2, 3, 4], vec![0.8, 0.4, 0.9]).unwrap().retain_confident_prefix(0.5).unwrap();
        assert_eq!(block.drafts, [2]);
        assert_eq!(block.confidences, [0.8]);
        assert!(SpeculativeBlock::new(1, vec![2], vec![0.5]).unwrap().retain_confident_prefix(f32::NAN).is_err());
    }

    #[test]
    fn block_keeps_supply_floor_beyond_confidence_prefix() {
        let block = SpeculativeBlock::new(1, vec![2, 3, 4], vec![0.8, 0.4, 0.9]).unwrap().retain_confident_prefix_at_least(0.5, 2).unwrap();
        assert_eq!(block.drafts, [2, 3]);
        assert_eq!(block.confidences, [0.8, 0.4]);
        assert!(SpeculativeBlock::new(1, vec![2], vec![0.5]).unwrap().retain_confident_prefix_at_least(0.5, 2).is_err());
    }

    #[test]
    fn block_spec_rejects_impossible_sizes() {
        assert!(BlockDraftSpec::new(8, 7, 1).is_ok());
        assert!(BlockDraftSpec::new(0, 0, 1).is_err());
        assert!(BlockDraftSpec::new(7, 8, 1).is_err());
        assert!(BlockDraftSpec::new(8, 7, 0).is_err());
    }

    #[test]
    fn hidden_capture_uses_model_output_boundaries() {
        let plan = HiddenStateCapturePlan::new(vec![70, 8, 23, 39, 55], 78).unwrap();
        assert_eq!(plan.boundaries(), [8, 23, 39, 55, 70]);
        assert!(plan.captures_layer_output(7));
        assert!(plan.captures_layer_output(69));
        assert!(!plan.captures_layer_output(70));
        assert!(!plan.captures_embedding());
    }

    #[test]
    fn stats_record_one_target_forward_per_round() {
        let verification = verify_samples(&[10, 21, 30], &[10, 20], &[]).unwrap();
        let mut stats = SpeculativeStats::default();
        stats.record(2, &verification);
        assert_eq!(stats, SpeculativeStats { rounds: 1, proposed: 2, accepted: 1, emitted: 2, target_forwards: 1 });
    }

    #[test]
    fn verification_describes_commit_boundary() {
        let verification = verify_samples(&[10, 21, 30], &[10, 20], &[]).unwrap();
        assert_eq!(verification.emitted_tokens(), [10]);
        assert_eq!(verification.pending_token(), 21);
        assert_eq!(verification.cache_commit().retained_rows(), 2);
        assert_eq!(verification.cache_end(7).unwrap(), 9);
    }
}
