//! MiniMax-M3 平台与执行无关的架构配置。

// MiniMax-M3 模型规格。
//
// 架构:前 3 层 GQA+dense FFN，后 57 层 MSA+sigmoid-bias MoE(128 专家/top4)，
// 使用 SwiGLU-OAI 与 Gemma-style RMSNorm。

#[derive(Debug)]
pub struct MiniMaxM3VisionConfig {
    pub hidden_size: usize,
    pub layer_count: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub projection_dim: usize,
    pub projector_hidden_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub max_aspect_ratio: f64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

/// MiniMax-M3 架构配置。所有字段直接暴露。
#[derive(Debug)]
pub struct MiniMaxM3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub layer_count: usize,
    pub dense_layer_count: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub use_qk_norm: bool,
    /// dense 层 FFN 中间维度。
    pub dense_intermediate_size: usize,
    /// 路由专家 FFN 中间维度。
    pub expert_intermediate_size: usize,
    /// 共享专家 FFN 中间维度。
    pub shared_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_shared_experts: usize,
    pub routed_scaling_factor: f32,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    // MSA(阶段 5 启用)
    pub msa_index_dim: usize,
    pub msa_block_size: usize,
    pub msa_topk_blocks: usize,
    pub eos_token_ids: Vec<u32>,
    pub vision: MiniMaxM3VisionConfig,
}

impl MiniMaxM3Config {
    /// 官方 MiniMax-M3 默认配置。
    pub fn standard() -> Self {
        Self {
            vocab_size: 200_064,
            hidden_size: 6_144,
            layer_count: 60,
            dense_layer_count: 3,
            num_heads: 64,
            num_kv_heads: 4,
            head_dim: 128,
            rope_dim: 64,
            rope_theta: 5_000_000.0,
            rms_eps: 1.0e-6,
            use_qk_norm: true,
            dense_intermediate_size: 12_288,
            expert_intermediate_size: 3_072,
            shared_intermediate_size: 3_072,
            num_experts: 128,
            num_experts_per_tok: 4,
            num_shared_experts: 1,
            routed_scaling_factor: 2.0,
            swiglu_alpha: 1.702,
            swiglu_limit: 7.0,
            msa_index_dim: 128,
            msa_block_size: 128,
            msa_topk_blocks: 16,
            eos_token_ids: vec![200_020],
            vision: MiniMaxM3VisionConfig {
                hidden_size: 1_280,
                layer_count: 32,
                num_heads: 16,
                intermediate_size: 5_120,
                projection_dim: 6_144,
                projector_hidden_size: 6_144,
                patch_size: 14,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                layer_norm_eps: 1.0e-5,
                rope_theta: 10_000.0,
                min_pixels: 4 * 28 * 28,
                max_pixels: 451_584,
                max_aspect_ratio: 200.0,
                image_mean: [0.48145466, 0.4578275, 0.40821073],
                image_std: [0.26862954, 0.261_302_6, 0.275_777_1],
            },
        }
    }
}
