//! CPU GQA causal prefill attention。

use crate::attention::gqa::{GqaSpec, prefill_attention_at_visible_with_dot_f32};

use super::matmul::dot;

#[allow(clippy::too_many_arguments)]
pub fn gqa_prefill_attention(query: &[f32], key: &[f32], value: &[f32], spec: &GqaSpec, output: &mut [f32]) {
    let query_tokens = gqa_query_tokens(query, spec);
    gqa_prefill_attention_at(query, key, value, 0, query_tokens, 0, spec, output);
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_prefill_attention_at(query: &[f32], key: &[f32], value: &[f32], kv_start: usize, kv_end: usize, position: usize, spec: &GqaSpec, output: &mut [f32]) {
    gqa_prefill_attention_at_visible(query, key, value, kv_start, kv_end, position, spec, None, output);
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_prefill_attention_at_visible(query: &[f32], key: &[f32], value: &[f32], kv_start: usize, kv_end: usize, position: usize, spec: &GqaSpec, visible_ends: Option<&[u32]>, output: &mut [f32]) {
    prefill_attention_at_visible_with_dot_f32(query, key, value, kv_start, kv_end, position, spec, visible_ends, output, dot).expect("CPU GQA shape 已由 backend 校验");
}

/// query 行数 = query.len() / query_cols(backend 已校验 query.len() 能被 query_cols 整除)。
fn gqa_query_tokens(query: &[f32], spec: &GqaSpec) -> usize {
    let query_cols = spec.num_heads.checked_mul(spec.head_dim).expect("CPU GQA query cols 溢出");
    query.len() / query_cols
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_attention_does_not_read_future_values() {
        let spec = GqaSpec { num_heads: 1, num_kv_heads: 1, head_dim: 1, rope_dim: 1, rope_theta: 1.0, use_qk_norm: false, window: crate::attention::gqa::CausalWindow::Full, score_scale: 1.0, output_gate: false };
        let query = vec![0.0; 2];
        let key = vec![0.0; 2];
        let value = vec![2.0, 6.0];
        let mut output = vec![0.0; 2];
        gqa_prefill_attention(&query, &key, &value, &spec, &mut output);
        assert_eq!(output, vec![2.0, 4.0]);
    }

    #[test]
    fn append_prefill_reads_prefix_values() {
        let spec = GqaSpec { num_heads: 1, num_kv_heads: 1, head_dim: 1, rope_dim: 1, rope_theta: 1.0, use_qk_norm: false, window: crate::attention::gqa::CausalWindow::Full, score_scale: 1.0, output_gate: false };
        let query = vec![0.0];
        let key = vec![0.0, 0.0];
        let value = vec![2.0, 6.0];
        let mut output = vec![0.0];
        gqa_prefill_attention_at(&query, &key, &value, 0, 2, 1, &spec, &mut output);
        assert_eq!(output, vec![4.0]);
    }

    #[test]
    fn sliding_window_discards_old_values() {
        let spec = GqaSpec { num_heads: 1, num_kv_heads: 1, head_dim: 1, rope_dim: 1, rope_theta: 1.0, use_qk_norm: false, window: crate::attention::gqa::CausalWindow::Sliding { size: 2 }, score_scale: 1.0, output_gate: false };
        let mut output = vec![0.0; 3];
        gqa_prefill_attention(&[0.0; 3], &[0.0; 3], &[2.0, 6.0, 10.0], &spec, &mut output);
        assert_eq!(output, vec![2.0, 4.0, 8.0]);
    }

    #[test]
    fn visual_block_reads_future_values() {
        let spec = GqaSpec { num_heads: 1, num_kv_heads: 1, head_dim: 1, rope_dim: 1, rope_theta: 1.0, use_qk_norm: false, window: crate::attention::gqa::CausalWindow::Sliding { size: 8 }, score_scale: 1.0, output_gate: false };
        let mut output = vec![0.0; 3];
        gqa_prefill_attention_at_visible(&[0.0; 3], &[0.0; 3], &[2.0, 6.0, 10.0], 0, 3, 0, &spec, Some(&[3, 3, 3]), &mut output);
        assert_eq!(output, vec![6.0, 6.0, 6.0]);
    }
}
