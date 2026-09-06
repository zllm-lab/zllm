//! K2-Horizon MoVA MoE 模型架构配置。

#[derive(Debug, Clone, Copy)]
pub struct K2HorizonConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub layer_count: usize,
    pub leading_dense_layer_count: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub norm_groups: usize,
    pub dense_intermediate_size: usize,
    pub expert_count: usize,
    pub expert_top_k: usize,
    pub expert_intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub value_expert_count: usize,
    pub value_expert_top_k: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl K2HorizonConfig {
    /// K2-Horizon-MoVA-36B-A4B 官方结构；GGUF 装载时仍逐字段校验。
    pub fn standard_36b_a4b() -> Self {
        Self {
            vocab_size: 250_624,
            hidden_size: 2_560,
            layer_count: 48,
            leading_dense_layer_count: 3,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_position_embeddings: 524_288,
            rope_dim: 128,
            rope_theta: 10_000_000.0,
            rms_eps: 1.0e-6,
            norm_groups: 2,
            dense_intermediate_size: 6_144,
            expert_count: 100,
            expert_top_k: 8,
            expert_intermediate_size: 768,
            shared_intermediate_size: 768,
            routed_scaling_factor: 2.5,
            value_expert_count: 64,
            value_expert_top_k: 4,
            bos_token_id: 0,
            eos_token_id: 1,
        }
    }
}
