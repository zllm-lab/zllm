//! SenseVoice CTC 贪心解码。
//!
//! FunASR / sherpa-onnx 一致的顺序：先在完整逐帧 argmax 序列上合并连续重复，
//! 再剔除 blank（blank 是重复分隔符，所以 [A, blank, A] 保留两个 A），
//! 最后跳过前 4 个控制位预测（lang/event/emo/textnorm），拼接词表得到文本。

/// 逐帧 argmax id → 转写 token 序列。
pub fn ctc_greedy(ids: &[u32], blank: u32, control_prefix: usize) -> Vec<u32> {
    let mut collapsed: Vec<u32> = Vec::with_capacity(ids.len());
    for &id in ids {
        if collapsed.last() != Some(&id) {
            collapsed.push(id);
        }
    }
    let mut tokens: Vec<u32> = collapsed.into_iter().filter(|&id| id != blank).collect();
    if tokens.len() > control_prefix {
        tokens.drain(..control_prefix);
    } else {
        tokens.clear();
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::ctc_greedy;

    #[test]
    fn collapses_repeats_and_blanks() {
        // blank=0：[5,5,0,6,6,5] → [5,6,5]
        assert_eq!(ctc_greedy(&[5, 5, 0, 6, 6, 5], 0, 0), vec![5, 6, 5]);
        // [A, blank, A]：blank 分隔的重复都保留
        assert_eq!(ctc_greedy(&[7, 0, 7], 0, 0), vec![7, 7]);
    }

    #[test]
    fn skips_control_prefix_after_collapse() {
        // 前 4 个控制位（1,2,3,4）被剔除；重复跨越控制边界时先合并
        assert_eq!(ctc_greedy(&[1, 2, 3, 4, 9, 9, 0, 8], 0, 4), vec![9, 8]);
        // 全 blank / 少于前缀长度 → 空
        assert_eq!(ctc_greedy(&[0, 0, 0], 0, 4), Vec::<u32>::new());
    }
}
