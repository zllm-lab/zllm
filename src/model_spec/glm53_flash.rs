//! GLM-5.3-Flash 模型架构配置。

#[derive(Clone)]
pub struct Glm53FlashConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub layer_count: usize,
    pub dense_layer_count: usize,
    /// 官方 checkpoint 追加的 MTP 层数(layers.{layer_count} 起),0 表示不装载。
    /// MTP 层为 DSA-MLA + MoE,无 mHC;供投机解码使用。
    pub mtp_layer_count: usize,
    pub max_position_embeddings: usize,
    // KDA 线性注意力(34 层)。
    pub kda_num_heads: usize,
    pub kda_head_dim: usize,
    pub kda_short_conv_kernel_size: usize,
    pub kda_gate_lower_bound: f32,
    /// decay(f_a/f_b)与输出门(g_a/g_b)共享的低秩宽度,权重 shape 实测 128。
    pub kda_decay_rank: usize,
    // nope MLA + DSA 索引(11 层)。
    /// 使用 DSA-MLA 的层号(0-based,官方 `full_attn_layers` 实测:层 2 无 MLA
    /// 投影张量,证明官方列表按 0-based 索引);其余层为 KDA。
    pub full_attention_layers: Vec<usize>,
    pub num_attention_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub value_head_dim: usize,
    pub index_num_heads: usize,
    pub index_head_dim: usize,
    pub index_top_k: usize,
    /// indexer 自身 rope 维度;实测 indexer 无 rope,位置感知走 kpool APE,固定 0。
    pub index_rope_head_dim: usize,
    /// indexer key 池化宽度(config index_kpool=4):每 4 token 压缩成 1 个索引条目。
    /// 压缩(APE+gate)语义在 backend DSA kernel 内实现,spec 仅透传。
    pub index_kpool: usize,
    /// 不完整尾池的原始 token 直接追加进选择结果(官方 index_kpool_always_select_tail)。
    pub index_kpool_always_select_tail: bool,
    // FFN / MoE。
    pub expert_count: usize,
    pub expert_top_k: usize,
    pub expert_intermediate_size: usize,
    pub dense_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    // mHC 残差。
    pub hyper_connection_copies: usize,
    pub hyper_connection_sinkhorn_iterations: usize,
    pub hyper_connection_eps: f32,
    pub rms_eps: f32,
    pub eos_token_ids: Vec<u32>,
}

impl Glm53FlashConfig {
    /// 官方 GLM-5.3-Flash 默认配置(来自 zai-org/GLM-5.3-Flash config.json)。
    pub fn standard() -> Self {
        Self {
            vocab_size: 154_880,
            hidden_size: 4_096,
            layer_count: 45,
            dense_layer_count: 3,
            mtp_layer_count: 1,
            max_position_embeddings: 1_048_576,
            kda_num_heads: 64,
            kda_head_dim: 128,
            kda_short_conv_kernel_size: 4,
            kda_gate_lower_bound: -5.0,
            kda_decay_rank: 128,
            full_attention_layers: (1..=11).map(|index| index * 4 - 1).collect(),
            num_attention_heads: 64,
            q_lora_rank: 1_536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 256,
            value_head_dim: 256,
            index_num_heads: 32,
            index_head_dim: 128,
            index_top_k: 2_048,
            index_rope_head_dim: 0,
            index_kpool: 4,
            index_kpool_always_select_tail: true,
            expert_count: 288,
            expert_top_k: 8,
            expert_intermediate_size: 2_048,
            dense_intermediate_size: 12_288,
            routed_scaling_factor: 2.5,
            hyper_connection_copies: 4,
            hyper_connection_sinkhorn_iterations: 20,
            hyper_connection_eps: 1.0e-6,
            rms_eps: 1.0e-5,
            eos_token_ids: vec![154_820, 154_827, 154_829],
        }
    }
}

/// 视觉塔配置；包含官方 vision config 与 processor 默认值。
#[derive(Debug)]
pub struct Glm53FlashVisionConfig {
    pub layer_count: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub intermediate_size: usize,
    pub projection_intermediate_size: usize,
    pub out_hidden_size: usize,
    pub swiglu_limit: f32,
    pub rms_norm_eps: f32,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub max_aspect_ratio: f64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Glm53FlashVisionConfig {
    pub fn standard() -> Self {
        Self {
            layer_count: 24,
            hidden_size: 1_024,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            intermediate_size: 4_096,
            projection_intermediate_size: 10_240,
            out_hidden_size: 4_096,
            swiglu_limit: 10.0,
            rms_norm_eps: 1.0e-5,
            layer_norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            min_pixels: 12_544,
            max_pixels: 9_633_792,
            max_aspect_ratio: 200.0,
            image_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            image_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let head_dim = self.hidden_size.checked_div(self.num_heads).unwrap_or(0);
        let valid = self.hidden_size > 0
            && self.layer_count > 0
            && self.num_heads > 0
            && self.hidden_size.is_multiple_of(self.num_heads)
            && head_dim >= 4
            && head_dim.is_multiple_of(4)
            && self.patch_size > 0
            && self.temporal_patch_size > 0
            && self.spatial_merge_size > 0
            && self.intermediate_size > 0
            && self.projection_intermediate_size > 0
            && self.out_hidden_size > 0
            && self.swiglu_limit.is_finite()
            && self.swiglu_limit > 0.0
            && self.rope_theta.is_finite()
            && self.rope_theta > 0.0
            && self.min_pixels > 0
            && self.min_pixels <= self.max_pixels
            && self.image_std.iter().all(|value| value.is_finite() && *value != 0.0);
        if valid { Ok(()) } else { Err(format!("GLM-5.3-Flash 视觉配置非法: {self:?}")) }
    }
}
