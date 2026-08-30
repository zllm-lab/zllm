//! GLM-5.3-Flash(`glm5_next`)平台无关模型规格。
//!
//! 45 层混合注意力:34 层 KDA 线性注意力 + 11 层 nope MLA(配 DSA 稀疏索引);
//! 前 3 层 dense FFN,其余 42 层 288+1 专家 MoE;层残差使用 mHC。
//! KDA 层权重保持 BF16,MLA 投影与全部 FFN/MoE 为官方 FP8([128,128] 块)。

pub mod execute;
pub mod layer;
pub mod prepare;
pub mod protocol;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_engine;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_node;
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub mod rocm_stage;
pub mod session;
pub mod vision;

pub use crate::model_spec::glm53_flash::Glm53FlashConfig;
use crate::{
    attention::{AttentionSpec, dsa::DsaSpec, hyper_connection::HyperConnectionSpec, kda::KdaSpec, mla::MlaSpec},
    moe::{
        Activation, FeedforwardSpec,
        dense_mlp::DenseMlpSpec,
        topk_moe::{ScoringFunc, TopkMoeSpec},
    },
    norm::NormSpec,
    runtime::{LayerId, LayerSpec, Model, ModelError},
};

pub struct Glm53Flash {
    config: Glm53FlashConfig,
    layer_specs: Vec<LayerSpec>,
}

impl Glm53Flash {
    pub fn new(config: Glm53FlashConfig) -> Result<Self, ModelError> {
        Self::validate(&config)?;
        let layer_specs = (0..config.layer_count).map(|layer| Self::build_layer_spec(&config, layer)).collect();
        Ok(Self { config, layer_specs })
    }

    pub fn standard() -> Self {
        Self::new(Glm53FlashConfig::standard()).expect("GLM-5.3-Flash 标准配置必须有效")
    }

    fn validate(config: &Glm53FlashConfig) -> Result<(), ModelError> {
        if config.layer_count == 0 || config.dense_layer_count > config.layer_count {
            return Err(ModelError::InvalidArchitecture("dense_layer_count 必须在 0..=layer_count".into()));
        }
        if config.mtp_layer_count > 1 {
            return Err(ModelError::InvalidArchitecture("mtp_layer_count 当前只支持 0 或 1".into()));
        }
        if config.expert_top_k == 0 || config.expert_top_k > config.expert_count {
            return Err(ModelError::InvalidArchitecture("expert_top_k 必须在 1..=expert_count".into()));
        }
        let mut full_layers = config.full_attention_layers.clone();
        full_layers.sort_unstable();
        if full_layers.iter().any(|&layer| layer == 0 || layer > config.layer_count) || full_layers.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ModelError::InvalidArchitecture("full_attention_layers 必须是唯一的 0-based 有效层号".into()));
        }
        // nope MLA 与 KDA 的维度减法依赖这些不变量,构造期一次性校验。
        Self::mla_spec(config).validate().map_err(ModelError::InvalidArchitecture)?;
        Self::kda_spec(config).validate().map_err(ModelError::InvalidArchitecture)?;
        Self::hyper_connection_spec(config).validate().map_err(ModelError::InvalidArchitecture)?;
        Ok(())
    }

    /// nope-only MLA:无 rope 分量,q/kv 投影只含 nope(+value) 部分。
    pub fn mla_spec(config: &Glm53FlashConfig) -> MlaSpec {
        MlaSpec {
            q_lora_rank: config.q_lora_rank,
            kv_lora_rank: config.kv_lora_rank,
            qk_rope_head_dim: 0,
            q_projection_size: config.num_attention_heads * config.qk_nope_head_dim,
            kv_projection_size: config.num_attention_heads * (config.qk_nope_head_dim + config.value_head_dim),
            num_heads: config.num_attention_heads,
            rope_theta: 10_000.0,
            rotary_layout: crate::attention::rope::RotaryLayout::Interleaved,
        }
    }

    pub fn dsa_spec(config: &Glm53FlashConfig) -> DsaSpec {
        DsaSpec {
            num_heads: config.index_num_heads,
            head_dim: config.index_head_dim,
            rope_dim: config.index_rope_head_dim,
            top_k: config.index_top_k,
            rotary_layout: crate::attention::rope::RotaryLayout::Interleaved,
            kpool: config.index_kpool,
            always_select_tail: config.index_kpool_always_select_tail,
        }
    }

    /// KDA:输出门为 g_a→g_b 两步因子化,不是 K3 的单矩阵全秩门,
    /// 由层编排(`layer::glm53_kda_attention`)展开,不影响 kernel spec。
    pub fn kda_spec(config: &Glm53FlashConfig) -> KdaSpec {
        KdaSpec {
            num_heads: config.kda_num_heads,
            head_dim: config.kda_head_dim,
            short_conv_kernel_size: config.kda_short_conv_kernel_size,
            use_full_rank_gate: false,
            gate_lower_bound: Some(config.kda_gate_lower_bound),
            use_qk_l2norm: true,
            output_norm_eps: config.rms_eps,
        }
    }

    pub fn hyper_connection_spec(config: &Glm53FlashConfig) -> HyperConnectionSpec {
        HyperConnectionSpec { copies: config.hyper_connection_copies, sinkhorn_iterations: config.hyper_connection_sinkhorn_iterations, eps: config.hyper_connection_eps }
    }

    pub fn moe_spec(config: &Glm53FlashConfig) -> TopkMoeSpec {
        TopkMoeSpec {
            num_experts: config.expert_count,
            top_k: config.expert_top_k,
            num_shared_experts: 1,
            scoring_func: ScoringFunc::SigmoidBias,
            normalize_selected: true,
            routed_scaling_factor: config.routed_scaling_factor,
            intermediate_size: config.expert_intermediate_size,
            shared_intermediate_size: config.expert_intermediate_size,
            activation: Activation::Silu,
        }
    }

    /// 层号(0-based)是否为 DSA-MLA 稀疏层;否则为 KDA 线性层。
    pub fn is_full_attention(&self, layer: LayerId) -> bool {
        self.config.full_attention_layers.contains(&layer)
    }

    fn build_layer_spec(config: &Glm53FlashConfig, layer: LayerId) -> LayerSpec {
        let attention = if config.full_attention_layers.contains(&layer) { AttentionSpec::Mla(Self::mla_spec(config)) } else { AttentionSpec::Kda(Self::kda_spec(config)) };
        let feedforward =
            if layer < config.dense_layer_count { FeedforwardSpec::Dense(DenseMlpSpec { intermediate_size: config.dense_intermediate_size, activation: Activation::Silu }) } else { FeedforwardSpec::TopkMoe(Self::moe_spec(config)) };
        LayerSpec { attention, feedforward, input_norm: NormSpec::Rms { eps: config.rms_eps }, post_attention_norm: NormSpec::Rms { eps: config.rms_eps }, post_norm: None }
    }
}

impl Model for Glm53Flash {
    type Config = Glm53FlashConfig;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn layer_count(&self) -> usize {
        self.config.layer_count
    }

    fn layer_spec(&self, layer: LayerId) -> Result<&LayerSpec, ModelError> {
        self.layer_specs.get(layer).ok_or(ModelError::LayerOutOfRange { layer, layer_count: self.config.layer_count })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard层表与官方配置一致() {
        let model = Glm53Flash::standard();
        let mut kda = 0;
        let mut mla = 0;
        let mut dense = 0;
        for layer in 0..model.layer_count() {
            let spec = model.layer_spec(layer).unwrap();
            match &spec.attention {
                AttentionSpec::Kda(_) => kda += 1,
                AttentionSpec::Mla(_) => mla += 1,
                other => panic!("GLM-5.3-Flash 出现非预期 attention: {other:?}"),
            }
            match &spec.feedforward {
                FeedforwardSpec::Dense(_) => dense += 1,
                FeedforwardSpec::TopkMoe(_) => {}
                other => panic!("GLM-5.3-Flash 出现非预期 feedforward: {other:?}"),
            }
        }
        assert_eq!((kda, mla, dense), (34, 11, 3));
        // DSA 层落在官方 full_attn_layers(0-based 的 3,7,...,43)。
        assert!(model.is_full_attention(3));
        assert!(model.is_full_attention(43));
        assert!(!model.is_full_attention(2));
        assert!(!model.is_full_attention(44));
    }

    #[test]
    fn nope_mla维度自洽() {
        let config = Glm53FlashConfig::standard();
        let mla = Glm53Flash::mla_spec(&config);
        mla.validate().unwrap();
        assert_eq!(mla.q_head_dim(), 256);
        assert_eq!(mla.qk_nope_dim(), 256);
        assert_eq!(mla.value_dim(), 256);
        // kv_a_proj_with_mqa 输出即纯 512 latent,无 rope 段。
        assert_eq!(mla.kv_a_out(), 512);
    }

    #[test]
    fn 重复层号被拒绝() {
        let mut config = Glm53FlashConfig::standard();
        config.full_attention_layers = vec![3, 3, 7];
        assert!(Glm53Flash::new(config).is_err());
    }
}
