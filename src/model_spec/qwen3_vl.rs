//! Qwen3/Qwen3-VL 文本与视觉架构配置。

#[derive(Clone, Debug)]
pub struct Qwen3VlVisionConfig {
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
    pub deepstack_visual_indexes: Vec<usize>,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub max_aspect_ratio: f64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Qwen3VlConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_count: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    pub rms_eps: f32,
    pub max_position_embeddings: usize,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub vision: Option<Qwen3VlVisionConfig>,
}
