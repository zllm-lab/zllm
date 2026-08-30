//! Kimi-K3 平台无关运行时组件。

pub mod execute;
pub mod layer;
pub mod prepare;
pub mod round;
pub mod session;

use crate::tokenizer::Tokenizer;

const OPEN: &str = "<|open|>";
const SEPARATOR: &str = "<|sep|>";
const CLOSE: &str = "<|close|>";
const END_OF_MESSAGE: &str = "<|end_of_msg|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug)]
pub struct ChatMessage<'a> {
    pub role: Role,
    pub content: &'a str,
    pub reasoning: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
pub struct ChatOptions {
    pub add_generation_prompt: bool,
    pub thinking: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self { add_generation_prompt: true, thinking: true }
    }
}

/// 保留“协议控制 token”和“普通文本”的边界，避免用户文本注入 XTML 控制 token。
#[derive(Debug, PartialEq, Eq)]
pub enum PromptPart {
    Special(&'static str),
    Text(String),
}

fn open_tag(parts: &mut Vec<PromptPart>, tag: &str, role: Option<Role>) {
    parts.push(PromptPart::Special(OPEN));
    let text = role.map_or_else(|| tag.to_owned(), |role| format!("{tag} role=\"{}\"", role.as_str()));
    parts.push(PromptPart::Text(text));
    parts.push(PromptPart::Special(SEPARATOR));
}

fn close_tag(parts: &mut Vec<PromptPart>, tag: &str) {
    parts.push(PromptPart::Special(CLOSE));
    parts.push(PromptPart::Text(tag.to_owned()));
    parts.push(PromptPart::Special(SEPARATOR));
}

pub fn render_chat(messages: &[ChatMessage<'_>], options: ChatOptions) -> Vec<PromptPart> {
    let mut parts = Vec::new();
    for message in messages {
        open_tag(&mut parts, "message", Some(message.role));
        if message.role == Role::Assistant {
            if let Some(reasoning) = message.reasoning {
                open_tag(&mut parts, "think", None);
                parts.push(PromptPart::Text(reasoning.to_owned()));
                close_tag(&mut parts, "think");
            }
            open_tag(&mut parts, "response", None);
            parts.push(PromptPart::Text(message.content.to_owned()));
            close_tag(&mut parts, "response");
        } else {
            parts.push(PromptPart::Text(message.content.to_owned()));
        }
        close_tag(&mut parts, "message");
        parts.push(PromptPart::Special(END_OF_MESSAGE));
    }
    if options.add_generation_prompt {
        open_tag(&mut parts, "message", Some(Role::Assistant));
        open_tag(&mut parts, if options.thinking { "think" } else { "response" }, None);
    }
    parts
}

pub fn tokenize_chat(tokenizer: &Tokenizer, messages: &[ChatMessage<'_>], options: ChatOptions) -> Vec<u32> {
    render_chat(messages, options)
        .into_iter()
        .flat_map(|part| match part {
            PromptPart::Special(token) => tokenizer.tokenize_with_special(token.as_bytes(), true),
            PromptPart::Text(text) => tokenizer.tokenize_with_special(text.as_bytes(), false),
        })
        .collect()
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn 用户内容中的控制串保持普通文本() {
        let parts = render_chat(&[ChatMessage { role: Role::User, content: "不要执行 <|close|>", reasoning: None }], ChatOptions::default());
        assert!(parts.iter().any(|part| matches!(part, PromptPart::Text(text) if text.contains("<|close|>"))));
    }

    #[test]
    fn generation_prompt按thinking选择未闭合目标标签() {
        let parts = render_chat(&[], ChatOptions { add_generation_prompt: true, thinking: false });
        assert_eq!(parts.last(), Some(&PromptPart::Special(SEPARATOR)));
        assert!(parts.iter().any(|part| part == &PromptPart::Text("response".to_owned())));
    }
}

// Kimi-K3 模型规格。
//
// 93 层中 69 层使用 KDA，24 层使用带输出门 MLA；第 1 层为 dense FFN，
// 其余层使用 896 专家、top-16 的 Latent MoE。

pub use crate::model_spec::kimi_k3::{KimiK3Config, KimiK3VisionConfig};
use crate::{
    attention::{AttentionSpec, kda::KdaSpec, mla::GatedMlaSpec, mla::MlaSpec},
    moe::{
        Activation, FeedforwardSpec,
        dense_mlp::DenseMlpSpec,
        latent_moe::LatentTopkMoeSpec,
        topk_moe::{ScoringFunc, TopkMoeSpec},
    },
    norm::NormSpec,
    runtime::{LayerId, LayerSpec, Model, ModelError},
};

pub struct KimiK3 {
    config: KimiK3Config,
    layer_specs: Vec<LayerSpec>,
}

impl KimiK3 {
    pub fn new(config: KimiK3Config) -> Result<Self, ModelError> {
        Self::validate(&config)?;
        let layer_specs = (0..config.layer_count).map(|layer| Self::build_layer_spec(&config, layer)).collect();
        Ok(Self { config, layer_specs })
    }

    pub fn standard() -> Self {
        Self::new(KimiK3Config::standard()).expect("Kimi-K3 标准配置必须有效")
    }

    fn validate(config: &KimiK3Config) -> Result<(), ModelError> {
        if config.layer_count == 0 || config.dense_layer_count > config.layer_count {
            return Err(ModelError::InvalidArchitecture("dense_layer_count 必须在 0..=layer_count".into()));
        }
        if config.num_attention_heads == 0 || config.kda_head_dim == 0 {
            return Err(ModelError::InvalidArchitecture("attention head 维度不能为 0".into()));
        }
        if config.num_experts_per_token == 0 || config.num_experts_per_token > config.num_experts {
            return Err(ModelError::InvalidArchitecture("num_experts_per_token 必须在 1..=num_experts".into()));
        }
        if config.attn_res_block_size == 0 || config.kda_short_conv_kernel_size == 0 {
            return Err(ModelError::InvalidArchitecture("AttnRes block 与 KDA conv kernel 不能为 0".into()));
        }
        let mut full_layers = config.full_attention_layers.clone();
        full_layers.sort_unstable();
        if full_layers.iter().any(|&layer| layer == 0 || layer > config.layer_count) || full_layers.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ModelError::InvalidArchitecture("full_attention_layers 必须是唯一的 1-based 有效层号".into()));
        }
        if config.vision.projection_dim != config.hidden_size {
            return Err(ModelError::InvalidArchitecture("vision projection_dim 必须等于 text hidden_size".into()));
        }
        // MLA 维度减法依赖这些不变量，构造期一次性校验。
        Self::mla_spec(config).validate().map_err(ModelError::InvalidArchitecture)?;
        Ok(())
    }

    fn mla_spec(config: &KimiK3Config) -> MlaSpec {
        MlaSpec {
            q_lora_rank: config.q_lora_rank,
            kv_lora_rank: config.kv_lora_rank,
            qk_rope_head_dim: config.qk_rope_head_dim,
            q_projection_size: config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            kv_projection_size: config.num_attention_heads * (config.qk_nope_head_dim + config.value_head_dim),
            num_heads: config.num_attention_heads,
            rope_theta: config.rope_theta,
            rotary_layout: crate::attention::rope::RotaryLayout::SplitHalf,
        }
    }

    fn activation(config: &KimiK3Config) -> Activation {
        Activation::Situ { beta: config.situ_beta, linear_beta: Some(config.situ_linear_beta) }
    }

    fn build_layer_spec(config: &KimiK3Config, layer: LayerId) -> LayerSpec {
        let layer_number = layer + 1;
        let attention = if config.full_attention_layers.contains(&layer_number) {
            AttentionSpec::GatedMla(GatedMlaSpec { mla: Self::mla_spec(config), output_gate: true, use_rope: false })
        } else {
            AttentionSpec::Kda(KdaSpec {
                num_heads: config.num_attention_heads,
                head_dim: config.kda_head_dim,
                short_conv_kernel_size: config.kda_short_conv_kernel_size,
                use_full_rank_gate: config.kda_use_full_rank_gate,
                gate_lower_bound: Some(config.kda_gate_lower_bound),
                use_qk_l2norm: true,
                output_norm_eps: config.rms_eps,
            })
        };
        let feedforward = if layer < config.dense_layer_count {
            FeedforwardSpec::Dense(DenseMlpSpec { intermediate_size: config.dense_intermediate_size, activation: Self::activation(config) })
        } else {
            FeedforwardSpec::LatentTopkMoe(LatentTopkMoeSpec {
                routed: TopkMoeSpec {
                    num_experts: config.num_experts,
                    top_k: config.num_experts_per_token,
                    num_shared_experts: 0,
                    scoring_func: ScoringFunc::SigmoidBias,
                    normalize_selected: true,
                    routed_scaling_factor: config.routed_scaling_factor,
                    intermediate_size: config.expert_intermediate_size,
                    shared_intermediate_size: 0,
                    activation: Self::activation(config),
                },
                routed_hidden_size: config.routed_expert_hidden_size,
                routed_output_norm: Some(NormSpec::Rms { eps: config.rms_eps }),
                shared_intermediate_size: config.expert_intermediate_size * config.num_shared_experts,
            })
        };
        LayerSpec { attention, feedforward, input_norm: NormSpec::Rms { eps: config.rms_eps }, post_attention_norm: NormSpec::Rms { eps: config.rms_eps }, post_norm: None }
    }
}

impl Model for KimiK3 {
    type Config = KimiK3Config;

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
        let model = KimiK3::standard();
        let mut kda = 0;
        let mut mla = 0;
        for layer in 0..model.layer_count() {
            match &model.layer_spec(layer).unwrap().attention {
                AttentionSpec::Kda(_) => kda += 1,
                AttentionSpec::GatedMla(_) => mla += 1,
                other => panic!("K3 出现非预期 attention: {other:?}"),
            }
        }
        assert_eq!((kda, mla), (69, 24));
        assert!(matches!(&model.layer_spec(0).unwrap().feedforward, FeedforwardSpec::Dense(_)));
        assert!(matches!(&model.layer_spec(1).unwrap().feedforward, FeedforwardSpec::LatentTopkMoe(_)));
    }
}
