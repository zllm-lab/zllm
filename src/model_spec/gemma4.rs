//! Gemma 4 文本与多模态架构配置。

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4VisionEncoderConfig {
    pub layer_count: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4VisionConfig {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub embedding_size: usize,
    pub position_embedding_size: usize,
    pub default_soft_tokens: usize,
    /// `None` 是 12B unified patch projector；`Some` 是 E4B Gemma4V ViT。
    pub encoder: Option<Gemma4VisionEncoderConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4AudioConfig {
    pub sample_rate: usize,
    pub samples_per_token: usize,
    pub embedding_size: usize,
}

#[derive(Clone, Debug)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_count: usize,
    pub num_heads: usize,
    pub local_num_kv_heads: usize,
    pub local_head_dim: usize,
    pub global_num_kv_heads: usize,
    pub global_head_dim: usize,
    pub sliding_window: usize,
    pub local_rope_theta: f32,
    pub global_rope_theta: f32,
    pub global_rope_fraction: f32,
    pub max_position_embeddings: usize,
    pub rms_eps: f32,
    pub final_logit_softcap: f32,
    pub per_layer_input_size: usize,
    pub num_kv_shared_layers: usize,
    pub attention_k_eq_v: bool,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub audio_token_id: u32,
    pub begin_image_token_id: u32,
    pub end_image_token_id: u32,
    pub begin_audio_token_id: u32,
    pub end_audio_token_id: u32,
    pub vision: Option<Gemma4VisionConfig>,
    pub audio: Option<Gemma4AudioConfig>,
}
