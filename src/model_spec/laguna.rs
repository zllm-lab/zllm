//! Laguna 模型架构配置(Poolside Laguna-S/XS 2.1)。
//!
//! 混合注意力 MoE:full attention 与 sliding-window GQA 按 1:4 交错,
//! full 层走 YaRN partial rotary,sliding 层走默认 RoPE;attention 输出带
//! 逐头 softplus 门控;MoE 为 sigmoid+bias 路由 + 共享专家。

use crate::attention::rope::RopeSpec;

#[derive(Debug, Clone)]
pub struct LagunaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub layer_count: usize,
    /// 前置 dense MLP 层数(Laguna = 1,即第 0 层;其余层为 sparse MoE)。
    pub leading_dense_layer_count: usize,
    /// full attention 交错间隔(Laguna = 4:layer 0,4,8.. 为 full,其余滑窗)。
    pub full_attention_interval: usize,
    /// full attention 层 query 头数(S=48,XS=48);GGUF 只存单一 head_count(滑窗层值),
    /// 映射时从 attn_q 张量维度逐类推导。
    pub full_num_heads: usize,
    /// sliding attention 层 query 头数(S=72,XS=64)。
    pub sliding_num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    /// full 层 RoPE(YaRN + partial rotary);GGUF 的 yarn_attn_factor 恒为 1.0,
    /// llama.cpp 生态按无 mscale 推理,这里保持一致。
    pub full_rope: RopeSpec,
    /// sliding 层 RoPE(默认 theta 10k,整 head_dim 旋转)。
    pub sliding_rope: RopeSpec,
    pub rms_eps: f32,
    /// dense MLP 中间维度(仅前置 dense 层使用)。
    pub dense_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub expert_intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    /// HF config eos_token_id = [2, 24];GGUF 只带单 eos,映射时并入。
    pub eos_token_ids: Vec<u32>,
}
