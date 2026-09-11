//! CPU 草稿 oracle。只共享 target 的输入 embedding 行与输出头，不持有 target cache。

use super::{Dflash2, compute};
use crate::{
    attention::rope::RopeTable,
    backend::{
        Backend, BackendError, SegmentedTensorBackend,
        cpu::{CpuContext, CpuWeight},
    },
    kernel::cpu::CpuTensor,
    runtime::speculative::SpeculativeBlock,
};

impl Dflash2<CpuWeight> {
    /// noise 必须来自 target 对 block_token_ids 的 embedding；草稿仅消费末尾 mask 行。
    /// draft_tokens 只缩短验证前缀，始终计算完整的非因果块。
    #[allow(clippy::too_many_arguments)]
    pub fn draft_greedy(
        &self,
        backend: &CpuContext,
        noise: CpuTensor,
        normalized_target: &CpuTensor,
        anchor: u32,
        target_position: usize,
        block_position: usize,
        target_lm_head: &CpuWeight,
        draft_tokens: usize,
        rope: &RopeTable,
    ) -> Result<SpeculativeBlock, BackendError> {
        let c = &self.config;
        self.block_token_ids(anchor)?;
        if draft_tokens == 0 || draft_tokens >= c.block_size || target_lm_head.rows() != c.vocab_size || target_lm_head.cols() != c.hidden_size {
            return Err(compute(format!("DFlash2 drafts={draft_tokens} 或 target LM head={}x{} 不兼容 block={} vocab={} hidden={}", target_lm_head.rows(), target_lm_head.cols(), c.block_size, c.vocab_size, c.hidden_size)));
        }
        let hidden = self.forward_hidden(backend, noise, normalized_target, target_position, block_position, rope)?;
        // 行 0 是 anchor 的表示；候选位从行 1 开始，不使用 AR 的 logits 偏移。
        let hidden = backend.slice_token_rows(&hidden, 1, draft_tokens)?;
        let logits = backend.linear(&hidden, target_lm_head)?;
        let gate = backend.linear(&hidden, &self.selector_projection)?;
        let tokens = greedy_selector(&logits, &gate, &self.predecessor, &self.successor, anchor, c.selector_top_k)?;
        SpeculativeBlock::new(anchor, tokens, Vec::new())
    }
}

/// greedy 只计算所选前驱的一行 transition，数学上等价于预计算 K×K 后走同一路径。
/// 相同分数按较小 token ID 决定；避免依赖不同设备 top-k 的未定义 tie 顺序。
pub fn greedy_selector(logits: &CpuTensor, gate: &CpuTensor, predecessor: &CpuWeight, successor: &CpuWeight, anchor: u32, top_k: usize) -> Result<Vec<u32>, BackendError> {
    let vocab = logits.cols;
    let rank = gate.cols;
    let matrix_shape = |weight: &CpuWeight| weight.rows() == vocab && weight.cols() == rank && weight.data().len() == vocab.saturating_mul(rank);
    if vocab == 0
        || rank == 0
        || top_k == 0
        || top_k > vocab
        || anchor as usize >= vocab
        || vocab > u32::MAX as usize
        || logits.rows != gate.rows
        || logits.data.len() != logits.rows.saturating_mul(vocab)
        || gate.data.len() != gate.rows.saturating_mul(rank)
        || !matrix_shape(predecessor)
        || !matrix_shape(successor)
    {
        return Err(compute(format!("DFlash2 selector shape 非法: logits={}x{vocab} gate={}x{rank} anchor={anchor} top_k={top_k}", logits.rows, gate.rows)));
    }
    let mut previous = anchor as usize;
    let mut path = Vec::with_capacity(logits.rows);
    for row in 0..logits.rows {
        let unary = logits.row(row);
        if unary.iter().any(|v| v.is_nan() || *v == f32::INFINITY) || gate.row(row).iter().any(|v| !v.is_finite()) {
            return Err(compute(format!("DFlash2 selector row={row} 包含非法 logits/gate")));
        }
        let mut candidates: Vec<usize> = (0..vocab).collect();
        let order = |a: &usize, b: &usize| unary[*b].total_cmp(&unary[*a]).then(a.cmp(b));
        if top_k < vocab {
            candidates.select_nth_unstable_by(top_k, order);
        }
        candidates.truncate(top_k);
        let mut best = None;
        for candidate in candidates {
            if unary[candidate] == f32::NEG_INFINITY {
                continue;
            }
            let mut score = unary[candidate];
            for (column, &context) in gate.row(row).iter().enumerate() {
                score += predecessor.data()[previous * rank + column] * context * successor.data()[candidate * rank + column];
            }
            if !score.is_finite() {
                return Err(compute(format!("DFlash2 selector row={row} candidate={candidate} score={score} 非法")));
            }
            if best.is_none_or(|(token, value)| score > value || score == value && candidate < token) {
                best = Some((candidate, score));
            }
        }
        previous = best.ok_or_else(|| compute(format!("DFlash2 selector row={row} 没有有限候选")))?.0;
        path.push(previous as u32);
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendResources;

    #[test]
    fn selector_uses_chosen_predecessor_and_anchor() {
        let cpu = CpuContext;
        let pred = cpu.prepare_f32(&[1., -2., 0.], 3, 1).unwrap();
        let succ = cpu.prepare_f32(&[0., 3., 100.], 3, 1).unwrap();
        let logits = CpuTensor { data: vec![1., 0., -4., 1., 0., -4.], rows: 2, cols: 3 };
        let gate = CpuTensor { data: vec![1., 1.], rows: 2, cols: 1 };
        // token 2 的 transition 再高也不能越过 unary top-k；第二行依赖第一行的选择。
        assert_eq!(greedy_selector(&logits, &gate, &pred, &succ, 0, 2).unwrap(), [1, 0]);
        assert_eq!(greedy_selector(&logits, &gate, &pred, &succ, 1, 2).unwrap(), [0, 1]);
    }

    #[test]
    fn selector_ties_mask_and_invalid_values() {
        let cpu = CpuContext;
        let codes = cpu.prepare_f32(&[0.; 3], 3, 1).unwrap();
        let mut logits = CpuTensor { data: vec![f32::NEG_INFINITY, 1., 1.], rows: 1, cols: 3 };
        let gate = CpuTensor { data: vec![1.], rows: 1, cols: 1 };
        assert_eq!(greedy_selector(&logits, &gate, &codes, &codes, 0, 2).unwrap(), [1]);
        logits.data.fill(f32::NEG_INFINITY);
        assert!(greedy_selector(&logits, &gate, &codes, &codes, 0, 2).is_err());
        logits.data[1] = f32::NAN;
        assert!(greedy_selector(&logits, &gate, &codes, &codes, 0, 2).is_err());
        assert!(greedy_selector(&logits, &gate, &codes, &codes, 0, 4).is_err());
    }

    #[test]
    fn full_draft_matches_upstream_cpu_reference() {
        use crate::weight::model::dflash2::{
            Dflash2Checkpoint,
            tests::{fixture, tiny_config},
        };
        let config = tiny_config();
        let files = fixture(&config);
        let cpu = CpuContext;
        let model = Dflash2::load(&cpu, &Dflash2Checkpoint::open(&files.0).unwrap()).unwrap();
        assert_eq!(model.capture_plan(4, 4, 5).unwrap().boundaries(), [1, 3]);
        assert!(model.capture_plan(4, 4, 6).is_err());
        assert!(model.block_token_ids(5).is_err());
        let captures: Vec<_> = (0..2).map(|j| CpuTensor { data: (0..12).map(|i| (((i * 3 + j * 5) % 13) as f32 - 6.) / 8.).collect(), rows: 3, cols: 4 }).collect();
        let projected = model.project_target(&cpu, &[&captures[0], &captures[1]]).unwrap();
        let noise = || CpuTensor { data: model.block_token_ids(1).unwrap().iter().flat_map(|&token| (0..4).map(move |d| ((((token as usize * 4 + d) * 3) % 11) as f32 - 5.) / 8.)).collect(), rows: 3, cols: 4 };
        let rope = RopeTable::precompute(7, 2, config.rope_theta);
        let hidden = model.forward_hidden(&cpu, noise(), &projected, 1, 4, &rope).unwrap();
        // z-lab/dflash model.py blob 7821d66a7b932b4e0f15381a39f76ba1d7eceb8a，
        // Transformers 5.7.0/PyTorch CPU F32；相同 BF16 可精确表示的小权重。
        let expected_hidden = [-1.3723154, -0.019937964, 0.5028671, 1.531767, -0.34606624, 0.7816557, 1.5107033, -1.1288353, -0.30274412, 1.0865308, 1.5222632, -0.86877483];
        let expected_projected = [0., -1.7133405, 0.8605643, -0.7184349, 1.784499, -0.43621087, 0.252804, -1.2491493, 1.0739343, 1.0960318, -1.2396649, -0.8153946];
        for (label, actual, expected) in [("hidden", &hidden.data, &expected_hidden), ("projected", &projected.data, &expected_projected)] {
            for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                assert!((a - b).abs() < 1e-5, "{label}[{i}]: {a} != {b}");
            }
        }
        let head = cpu.prepare_f32(&(0..20).map(|i| ((i * 5 % 13) as f32 - 6.) / 8.).collect::<Vec<_>>(), 5, 4).unwrap();
        let draft = model.draft_greedy(&cpu, noise(), &projected, 1, 1, 4, &head, 2, &rope).unwrap();
        assert_eq!(draft.drafts, [4, 4]);
        assert_eq!(model.draft_greedy(&cpu, noise(), &projected, 1, 1, 4, &head, 1, &rope).unwrap().drafts, draft.drafts[..1]);
        // 按绝对位置丢弃不可见历史不改变输出；少留一行必须报错。
        let recent = cpu.slice_token_rows(&projected, 1, 2).unwrap();
        let cropped = model.forward_hidden(&cpu, noise(), &recent, 2, 4, &rope).unwrap();
        for (a, b) in hidden.data.iter().zip(&cropped.data) {
            assert!((a - b).abs() < 1e-5);
        }
        let short = cpu.slice_token_rows(&projected, 2, 1).unwrap();
        assert!(model.forward_hidden(&cpu, noise(), &short, 3, 4, &rope).is_err());
        assert!(model.forward_hidden(&cpu, noise(), &projected, 0, 4, &rope).is_err());
    }
}
