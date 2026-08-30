//! MiniMax 稀疏注意力(MSA)规格。MiniMax-M3 长上下文使用。
//!
//! 论文:arXiv:2606.13392。
//! 每个 query 仅看 `topk_blocks × block_size = 16 × 128 = 2048` 个 KV token,
//! 与上下文长度无关。

/// MSA 规格。
#[derive(Debug)]
pub struct MsaSpec {
    /// GQA 部分(query head / KV head / head dim)。
    pub gqa: super::gqa::GqaSpec,
    /// Index Branch 的 head 维度(d_idx)。
    pub index_dim: usize,
    /// Index Branch 的 query head 数(等于 KV head 数)。
    pub num_index_heads: usize,
    /// KV 块大小(B_k)。
    pub block_size: usize,
    /// 每 query 选 top-k 个块(k)。
    pub topk_blocks: usize,
    /// 强制包含的初始块数(attention sink)。
    pub init_block: usize,
    /// 强制包含的 local 块数(query 自身所在块)。
    pub local_block: usize,
}
