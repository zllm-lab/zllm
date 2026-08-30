//! GLM-5.2 架构配置。

/// GLM-5.2 仅在前 3 层及之后每 4 层启用一次 indexer。
pub const fn is_indexer_layer(layer: usize) -> bool {
    layer < 3 || (layer >= 6 && (layer - 6).is_multiple_of(4))
}

/// 官方权重的解码器层数，不含 MTP。
pub const GLM52_LAYER_COUNT: usize = 78;

#[derive(Debug, Clone)]
pub struct Glm52Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub layer_count: usize,
    pub mtp_layer_count: usize,
    pub dense_layer_count: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub q_projection_size: usize,
    pub kv_projection_size: usize,
    pub num_heads: usize,
    pub rope_theta: f32,
    pub expert_count: usize,
    pub expert_top_k: usize,
    pub expert_intermediate_size: usize,
    pub dense_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub rms_eps: f32,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_top_k: usize,
    pub eos_token_ids: Vec<u32>,
}

impl Glm52Config {
    pub fn standard() -> Self {
        Self {
            vocab_size: 154_880,
            hidden_size: 6_144,
            layer_count: GLM52_LAYER_COUNT,
            mtp_layer_count: 1,
            dense_layer_count: 3,
            q_lora_rank: 2_048,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            q_projection_size: 16_384,
            kv_projection_size: 28_672,
            num_heads: 64,
            rope_theta: 8_000_000.0,
            expert_count: 256,
            expert_top_k: 8,
            expert_intermediate_size: 2_048,
            dense_intermediate_size: 12_288,
            routed_scaling_factor: 2.5,
            rms_eps: 1.0e-5,
            index_heads: 32,
            index_head_dim: 128,
            index_top_k: 2048,
            eos_token_ids: vec![154_820, 154_827, 154_829],
        }
    }
}
