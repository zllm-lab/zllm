//! Qwen3.6 文本与视觉架构配置。

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen36AttentionKind {
    GatedDeltaNet,
    FullAttention,
}

#[derive(Clone, Debug)]
pub struct Qwen36VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub position_embeddings: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub output_hidden_size: usize,
    pub rope_theta: f32,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub max_aspect_ratio: f64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Qwen36Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub full_attention_interval: usize,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_size: usize,
    pub mtp_layers: usize,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub vision: Qwen36VisionConfig,
}
