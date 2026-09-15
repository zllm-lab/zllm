//! DeepSeek-V4 平台与执行无关的架构配置。

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4RoutingSelection {
    TokenHash,
    ScoreTopK,
}

/// DeepSeek-V4.1 引入的 engram n-gram 记忆表配置。
///
/// 表按层成对出现(V4.1-Flash 为 L1/L14),`num_embeddings` 逐层给出;
/// 官方 16M n-gram vocab 经压缩映射到 `compressed_vocab_size`。
#[derive(Debug, Clone)]
pub struct DeepSeekV4EngramConfig {
    pub layer_ids: Vec<usize>,
    pub num_embeddings: Vec<usize>,
    pub max_ngram_size: usize,
    pub vocab_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub pad_token_id: u32,
    pub compressed_vocab_size: usize,
}

/// DeepSeek-V4.1-Flash 视觉塔配置(官方 vision_config 纯常量)。
///
/// ViT 输出经 downsample_ratio×downsample_ratio 空间合并与 aligner 双线性
/// 投影到语言模型 hidden_size;`image_token_id` 是 image span 内所有位置
/// (含 start/newline/end)在 input_ids 中共用的原始 token id。
#[derive(Debug, Clone)]
pub struct DeepseekV41VisionConfig {
    pub layer_count: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub rope_theta: f32,
    pub downsample_ratio: usize,
    /// 单张图(含换行与首尾标记)最多占用的语言模型 token 数。
    pub max_image_tokens: usize,
    /// 像素面积下限;低于该值先等比放大再取整到 patch 网格。
    pub min_pixels: usize,
    pub image_token_id: u32,
}

impl DeepseekV41VisionConfig {
    /// 官方 DeepSeek-V4.1-Flash vision_config。
    pub fn standard() -> Self {
        Self { layer_count: 32, hidden_size: 1_024, num_heads: 16, intermediate_size: 2_816, patch_size: 14, rope_theta: 10_000.0, downsample_ratio: 3, max_image_tokens: 1_024, min_pixels: 295_936, image_token_id: 129_264 }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.layer_count == 0 || self.hidden_size == 0 || self.num_heads == 0 || self.patch_size == 0 || self.downsample_ratio == 0 {
            return Err("DeepSeek-V4.1 视觉配置的层数/维度/头数/patch/downsample 必须非零".to_owned());
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(format!("DeepSeek-V4.1 视觉 hidden_size={} 不能被 num_heads={} 整除", self.hidden_size, self.num_heads));
        }
        if self.hidden_size / self.num_heads % 4 != 0 {
            return Err(format!("DeepSeek-V4.1 视觉 head_dim={} 不能被 4 整除(2D RoPE h/w 各半)", self.hidden_size / self.num_heads));
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) || self.max_image_tokens < 2 || self.min_pixels == 0 {
            return Err("DeepSeek-V4.1 视觉 rope/max_image_tokens/min_pixels 配置非法".to_owned());
        }
        Ok(())
    }
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
    /// MTP 层 routed expert 数;V4.1 为 128(主干 384),V4 与主干一致。
    pub mtp_expert_count: usize,
    /// MTP/DSpark 层每 token 激活专家数;V4.1 为 3,与主干 top_k 一致时也不冗余。
    pub mtp_expert_top_k: usize,
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
    /// V4.1 engram 记忆表;V4 为 None。
    pub engram: Option<DeepSeekV4EngramConfig>,
    /// V4.1 路由在 correction bias 之外携带 value-level bias(exp_probs_b_vl)。
    pub router_value_level_bias: bool,
    /// V4.1 多模态:image span 所有位置共用的 token id(`<｜deepseek_image｜>`);
    /// V4 无视觉,置 None。engram 用它把 span 推成 DEAD、输出侧拦截生成。
    pub image_token_id: Option<u32>,
    /// CSA2:自产压缩 KV 的层(其余层读最近 source 的共享 cache);V4 每层自产,置空。
    pub kv_source_layers: Vec<usize>,
    /// CSA2:自算 indexer topk 的层(其余层复用最近 source 发布的 idxs)。
    pub index_source_layers: Vec<usize>,
    /// CSA2 候选粗筛源层(L20);<0 语义用 None 之外的 usize::MAX 表示关闭。
    pub candidate_source_layer: Option<usize>,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
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
            mtp_expert_count: 256,
            mtp_expert_top_k: 6,
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
            engram: None,
            router_value_level_bias: false,
            image_token_id: None,
            kv_source_layers: Vec::new(),
            index_source_layers: Vec::new(),
            candidate_source_layer: None,
            candidate_topk_blocks: 0,
            candidate_block_size: 0,
        }
    }

    /// 官方 DeepSeek-V4.1-Flash 配置。末尾三个 compression ratio 属于 MTP 层;
    /// ratio 语义:0=无压缩,1=滑窗全选,2=LearnedIndexer 2:1 压缩。
    pub fn flash_v41() -> Self {
        let layer_count = 40;
        let mtp_layer_count = 3;
        let mut compress_ratios = Vec::with_capacity(layer_count + mtp_layer_count);
        compress_ratios.extend([0, 0]);
        compress_ratios.extend(std::iter::repeat_n(2, 18));
        compress_ratios.extend(std::iter::repeat_n(1, 20));
        compress_ratios.extend(std::iter::repeat_n(0, mtp_layer_count));
        Self {
            vocab_size: 129_280,
            hidden_size: 5_120,
            layer_count,
            mtp_layer_count,
            hash_layer_count: 0,
            max_position_embeddings: 1_048_576,
            num_heads: 64,
            num_kv_heads: 1,
            head_dim: 512,
            q_lora_rank: 1_280,
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
            index_heads: 32,
            index_head_dim: 128,
            index_top_k: 512,
            expert_count: 384,
            mtp_expert_count: 128,
            mtp_expert_top_k: 3,
            expert_top_k: 6,
            shared_expert_count: 1,
            expert_intermediate_size: 2_304,
            routed_scaling_factor: 1.5,
            swiglu_limit: 10.0,
            rms_eps: 1.0e-20,
            hyper_connection_copies: 4,
            hyper_connection_sinkhorn_iterations: 20,
            hyper_connection_eps: 1.0e-6,
            bos_token_id: 0,
            eos_token_ids: vec![1],
            engram: Some(DeepSeekV4EngramConfig {
                layer_ids: vec![1, 14],
                num_embeddings: vec![384_006_168, 384_016_682],
                max_ngram_size: 4,
                vocab_size: 16_000_000,
                n_heads: 8,
                head_dim: 256,
                pad_token_id: 2,
                compressed_vocab_size: 99_092,
            }),
            router_value_level_bias: true,
            image_token_id: Some(129_264),
            kv_source_layers: vec![2, 8, 14, 20],
            index_source_layers: vec![2, 8, 14, 20, 24, 28, 32, 36],
            // ROCm 候选块粗筛 kernel 未接入(见 compressed_sparse 显式报错);
            // 先关闭走单级 topk(数学等价的稀疏选择),接入后恢复 Some(20)/2048/8。
            candidate_source_layer: None,
            candidate_topk_blocks: 0,
            candidate_block_size: 0,
        }
    }
}
