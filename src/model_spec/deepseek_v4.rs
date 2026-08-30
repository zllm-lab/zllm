//! DeepSeek-V4 平台与执行无关的架构配置。

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4RoutingSelection {
    TokenHash,
    ScoreTopK,
}

#[derive(Debug, Clone)]
pub struct DeepSeekV4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub layer_count: usize,
    pub mtp_layer_count: usize,
    pub hash_layer_count: usize,
    pub max_position_embeddings: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub output_groups: usize,
    pub output_lora_rank: usize,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub original_position_embeddings: usize,
    pub rope_beta_fast: f32,
    pub rope_beta_slow: f32,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_top_k: usize,
    pub expert_count: usize,
    pub expert_top_k: usize,
    pub shared_expert_count: usize,
    pub expert_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub swiglu_limit: f32,
    pub rms_eps: f32,
    pub hyper_connection_copies: usize,
    pub hyper_connection_sinkhorn_iterations: usize,
    pub hyper_connection_eps: f32,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
}

impl DeepSeekV4Config {
    /// 官方 DeepSeek-V4-Flash 配置。末尾三个 compression ratio 属于 MTP 层。
    pub fn flash() -> Self {
        let layer_count = 43;
        let mtp_layer_count = 3;
        let mut compress_ratios = Vec::with_capacity(layer_count + mtp_layer_count);
        compress_ratios.extend([0, 0]);
        compress_ratios.extend((2..layer_count).map(|layer| if layer % 2 == 0 { 4 } else { 128 }));
        compress_ratios.extend(std::iter::repeat_n(0, mtp_layer_count));
        Self {
            vocab_size: 129_280,
            hidden_size: 4_096,
            layer_count,
            mtp_layer_count,
            hash_layer_count: 3,
            max_position_embeddings: 1_048_576,
            num_heads: 64,
            num_kv_heads: 1,
            head_dim: 512,
            q_lora_rank: 1_024,
            qk_rope_head_dim: 64,
            output_groups: 8,
            output_lora_rank: 1_024,
            sliding_window: 128,
            compress_ratios,
            rope_theta: 10_000.0,
            compress_rope_theta: 160_000.0,
            rope_factor: 16.0,
            original_position_embeddings: 65_536,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            index_heads: 64,
            index_head_dim: 128,
            index_top_k: 512,
            expert_count: 256,
            expert_top_k: 6,
            shared_expert_count: 1,
            expert_intermediate_size: 2_048,
            routed_scaling_factor: 1.5,
            swiglu_limit: 10.0,
            rms_eps: 1.0e-6,
            hyper_connection_copies: 4,
            hyper_connection_sinkhorn_iterations: 20,
            hyper_connection_eps: 1.0e-6,
            bos_token_id: 0,
            eos_token_ids: vec![1],
        }
    }
}
