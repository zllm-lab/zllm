//! 多头潜在注意力(MLA)规格。GLM-5.2、DeepSeek-V3 使用。

/// MLA 规格。字段全是架构常量,与硬件无关。
#[derive(Debug, Clone)]
pub struct MlaSpec {
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub q_projection_size: usize,
    pub kv_projection_size: usize,
    pub num_heads: usize,
    pub rope_theta: f32,
    pub rotary_layout: super::rope::RotaryLayout,
}

/// 带输出门的 MLA 规格。Kimi-K3 可独立控制输出门和 RoPE。
#[derive(Debug, Clone)]
pub struct GatedMlaSpec {
    pub mla: MlaSpec,
    pub output_gate: bool,
    pub use_rope: bool,
}

impl MlaSpec {
    /// 构造期校验：保证 hot path 的 `qk_nope_dim()`/`value_dim()` 减法不下溢，
    /// 以及下游 `RopeTable::precompute` 的参数合法。模型在构造 LayerSpec/Config
    /// 时调用一次；hot path 不再重复检查。
    pub fn validate(&self) -> Result<(), String> {
        if self.num_heads == 0 || !self.q_projection_size.is_multiple_of(self.num_heads) || !self.kv_projection_size.is_multiple_of(self.num_heads) {
            return Err(format!("MLA head 维度非法: q_proj={} kv_proj={} heads={}", self.q_projection_size, self.kv_projection_size, self.num_heads));
        }
        // qk_rope_head_dim == 0 是 nope-only 变体(GLM-5.3-Flash):
        // MLA 本体无 rope 分量,位置信息由 DSA indexer 的独立 rope 承担。
        if self.qk_rope_head_dim > 0 && (!self.qk_rope_head_dim.is_multiple_of(2) || !self.rope_theta.is_finite() || self.rope_theta <= 0.0) {
            return Err(format!("MLA RoPE 参数非法: rope_dim={} theta={}", self.qk_rope_head_dim, self.rope_theta));
        }
        let q_head_dim = self.q_head_dim();
        if self.qk_rope_head_dim > q_head_dim {
            return Err(format!("MLA rope dim {} 超过 query head dim {q_head_dim}", self.qk_rope_head_dim));
        }
        let qk_nope_dim = q_head_dim - self.qk_rope_head_dim;
        let kv_head_dim = self.kv_head_dim();
        if qk_nope_dim > kv_head_dim {
            return Err(format!("MLA nope dim {qk_nope_dim} 超过 KV head dim {kv_head_dim}"));
        }
        Ok(())
    }

    /// 每 head 的 query 维度 = q_projection_size / num_heads(前 nope + 后 rope)。
    pub fn q_head_dim(&self) -> usize {
        self.q_projection_size / self.num_heads
    }
    /// q/k 点积的 nope 部分(不参与 RoPE)。
    pub fn qk_nope_dim(&self) -> usize {
        self.q_head_dim() - self.qk_rope_head_dim
    }
    /// 每 head 的 kv 维度 = kv_projection_size / num_heads(前 nope + 后 value)。
    pub fn kv_head_dim(&self) -> usize {
        self.kv_projection_size / self.num_heads
    }
    /// 每 head 的 value 维度(kv_head_dim - qk_nope_dim)。
    pub fn value_dim(&self) -> usize {
        self.kv_head_dim() - self.qk_nope_dim()
    }
    /// kv_a_proj 输出维度 = kv_lora_rank + qk_rope_head_dim(latent + rope)。
    pub fn kv_a_out(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }
}

/// 已完成 KV 投影后的 MLA attention f32 reference。
/// `projected_kv` 的行与 `rope_indices` 一一对应，rope 从完整 cache 中按索引读取。
pub fn reference_attention(query: &[f32], projected_kv: &[f32], rope_cache: &[f32], rope_columns: usize, rope_indices: &[usize], spec: &MlaSpec) -> Result<Vec<f32>, String> {
    if spec.num_heads == 0 || !spec.q_projection_size.is_multiple_of(spec.num_heads) || !spec.kv_projection_size.is_multiple_of(spec.num_heads) {
        return Err("MLA head 维度非法".to_owned());
    }
    if query.len() != spec.q_projection_size {
        return Err(format!("MLA query 长度 {}，期望 {}", query.len(), spec.q_projection_size));
    }
    if rope_indices.is_empty() {
        return Err("MLA selection 为空".to_owned());
    }
    let projected_len = rope_indices.len().checked_mul(spec.kv_projection_size).ok_or("MLA projected KV 长度溢出")?;
    if projected_kv.len() != projected_len {
        return Err(format!("MLA projected KV 长度 {}，期望 {projected_len}", projected_kv.len()));
    }
    // nope-only(rope_columns == 0)时 rope_cache 为空,点积退化为纯 nope 部分。
    if rope_columns != spec.qk_rope_head_dim || (rope_columns > 0 && !rope_cache.len().is_multiple_of(rope_columns)) {
        return Err(format!("MLA rope cache shape 异常: len={} cols={rope_columns}", rope_cache.len()));
    }
    let rope_rows = if rope_columns == 0 { usize::MAX } else { rope_cache.len() / rope_columns };
    if let Some(index) = rope_indices.iter().copied().find(|&index| index >= rope_rows) {
        return Err(format!("MLA rope index {index} 越界，rows={rope_rows}"));
    }

    let q_head_dim = spec.q_head_dim();
    if spec.qk_rope_head_dim > q_head_dim {
        return Err(format!("MLA rope dim {} 超过 query head dim {q_head_dim}", spec.qk_rope_head_dim));
    }
    let qk_nope_dim = spec.qk_nope_dim();
    let kv_head_dim = spec.kv_head_dim();
    if qk_nope_dim > kv_head_dim {
        return Err(format!("MLA nope dim {qk_nope_dim} 超过 KV head dim {kv_head_dim}"));
    }
    let value_dim = spec.value_dim();
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    let mut output = vec![0.0; spec.q_projection_size];

    for head in 0..spec.num_heads {
        let query_base = head * q_head_dim;
        let query_nope = &query[query_base..query_base + qk_nope_dim];
        let query_rope = &query[query_base + qk_nope_dim..query_base + q_head_dim];
        let mut scores = Vec::with_capacity(rope_indices.len());
        let mut maximum = f32::NEG_INFINITY;
        for (selected, &token) in rope_indices.iter().enumerate() {
            let kv_base = selected * spec.kv_projection_size + head * kv_head_dim;
            let key_nope = &projected_kv[kv_base..kv_base + qk_nope_dim];
            let key_rope = &rope_cache[token * rope_columns..(token + 1) * rope_columns];
            let score = (dot(query_nope, key_nope) + dot(query_rope, key_rope)) * scale;
            maximum = maximum.max(score);
            scores.push(score);
        }
        let mut denominator = 0.0;
        for score in &mut scores {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        for value in 0..value_dim {
            let mut sum = 0.0;
            for (selected, score) in scores.iter().enumerate() {
                let value_index = selected * spec.kv_projection_size + head * kv_head_dim + qk_nope_dim + value;
                sum += score / denominator * projected_kv[value_index];
            }
            output[head * q_head_dim + value] = sum;
        }
    }
    Ok(output)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(left, right)| left * right).sum()
}

#[allow(clippy::too_many_arguments)]
pub fn mla_causal_prefill<B: crate::backend::DsaPrefillBackend>(
    backend: &B,
    query: &B::Tensor,
    latent: &B::Tensor,
    k_rope: &B::Tensor,
    kv_b_proj: &B::Weight,
    cache: Option<&mut B::Cache>,
    layer: usize,
    mla: &MlaSpec,
    dsa: &super::dsa::DsaSpec,
    dsa_state: Option<&mut B::DsaState>,
) -> Result<B::Tensor, crate::backend::BackendError> {
    match dsa_state {
        Some(state) => backend.mla_prefill_attention_selected(query, latent, k_rope, kv_b_proj, cache, layer, mla, dsa, state),
        None => backend.mla_prefill_attention(query, latent, k_rope, kv_b_proj, cache, layer, mla),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn mla_causal_prefill_rope<B: crate::backend::DsaPrefillBackend>(
    backend: &B,
    query: &B::Tensor,
    latent: &B::Tensor,
    k_rope: &B::Tensor,
    kv_b_proj: &B::Weight,
    cache: Option<&mut B::Cache>,
    layer: usize,
    position: usize,
    cos: &[f32],
    sin: &[f32],
    mla: &MlaSpec,
    dsa: &super::dsa::DsaSpec,
    dsa_state: Option<&mut B::DsaState>,
) -> Result<B::Tensor, crate::backend::BackendError> {
    match dsa_state {
        Some(state) => backend.mla_prefill_attention_selected_rope(query, latent, k_rope, kv_b_proj, cache, layer, position, cos, sin, mla, dsa, state),
        None => backend.mla_prefill_attention_rope(query, latent, k_rope, kv_b_proj, cache, layer, position, cos, sin, mla),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_attention_single_token_returns_value() {
        let spec = MlaSpec { q_lora_rank: 2, kv_lora_rank: 2, qk_rope_head_dim: 2, q_projection_size: 4, kv_projection_size: 4, num_heads: 1, rope_theta: 10_000.0, rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf };
        let output = reference_attention(&[1.0, 0.0, 1.0, 0.0], &[1.0, 0.0, 2.0, 4.0], &[1.0, 0.0], 2, &[0], &spec).unwrap();
        assert_eq!(output, vec![2.0, 4.0, 0.0, 0.0]);
    }
}
