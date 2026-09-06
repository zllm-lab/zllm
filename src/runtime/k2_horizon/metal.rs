//! K2-Horizon 与 Metal backend 的具体组合。

use super::{K2HorizonConfig, K2HorizonGguf};
use crate::{
    attention::rope::{RopeSpec, RopeTable, RotaryLayout},
    backend::metal::{MetalContext, MetalKvCache, MetalMoeDecodeState, MetalPrefillExperts, MetalTensor, MetalWeight},
    backend::{Backend, BackendError, BackendResources, GqaPrefillBackend, LinearWeight},
    moe::{
        prefill::prefill_experts_observed,
        topk_moe::{MoeFfnRef, RoutedMoeInputs, SharedExpertRef},
    },
    runtime::{
        expert_pipeline::{ExpertDecodePipeline, ExpertDecodeRequest},
        output::{OutputHead, OutputNorm, OutputPlan, OutputResult},
    },
    weight::expert_source::ExpertSourceProvider,
};

pub enum K2Value<W> {
    Dense(W),
    Routed { router: W, bias: W, experts: W },
}

pub enum K2Mlp<W> {
    Dense { gate: W, up: W, down: W },
    Sparse { router: W, bias: W, shared_gate: W, shared_up: W, shared_down: W },
}

pub struct K2Layer<W> {
    pub(super) input_norm: W,
    pub(super) query: W,
    pub(super) key: W,
    pub(super) value: K2Value<W>,
    pub(super) attention_gate: W,
    pub(super) output: W,
    pub(super) ffn_norm: W,
    pub(super) mlp: K2Mlp<W>,
}

pub type K2OutputHead = OutputHead<MetalWeight>;

pub fn rope_table(config: &K2HorizonConfig, sequence_len: usize) -> Result<RopeTable, String> {
    RopeTable::from_spec(sequence_len, RopeSpec::Default { rotary_dim: config.rope_dim, theta: config.rope_theta }).map_err(|error| format!("K2-Horizon RoPE: {error}"))
}

pub fn prepare_layers(context: &MetalContext, source: &K2HorizonGguf) -> Result<Vec<K2Layer<MetalWeight>>, BackendError> {
    (0..source.config().layer_count).map(|layer| prepare_layer(context, source, layer)).collect()
}

fn prepare_layer(context: &MetalContext, source: &K2HorizonGguf, layer: usize) -> Result<K2Layer<MetalWeight>, BackendError> {
    let reader = source.reader();
    let prefix = format!("blk.{layer}");
    let dense = layer < source.config().leading_dense_layer_count;
    let value = if dense {
        K2Value::Dense(crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.attn_v.weight"))?)
    } else {
        let matrix = reader.read_matrix_array(&format!("{prefix}.attn_v_exps.weight")).map_err(crate::runtime::compute_error)?;
        K2Value::Routed {
            // 路由矩阵很小；第三方 GGUF 将其量化为 IQ3_S，局部解量化为
            // F32 resident 可保持路由打分稳定，也不放宽公共 F32 装载契约。
            router: prepare_router_f32(context, reader, &format!("{prefix}.attn_v_gate.weight"))?,
            bias: crate::runtime::prepare_gguf_f32_vector(context, reader, &format!("{prefix}.attn_v_gate.bias"))?,
            experts: context.prepare_weight(LinearWeight::gguf(&matrix), matrix.rows, matrix.columns)?,
        }
    };
    let mlp = if dense {
        K2Mlp::Dense {
            gate: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.ffn_gate.weight"))?,
            up: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.ffn_up.weight"))?,
            down: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.ffn_down.weight"))?,
        }
    } else {
        K2Mlp::Sparse {
            router: crate::runtime::prepare_gguf_f32_matrix(context, reader, &format!("{prefix}.ffn_gate_inp.weight"))?,
            bias: crate::runtime::prepare_gguf_f32_vector(context, reader, &format!("{prefix}.exp_probs_b.bias"))?,
            shared_gate: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.ffn_gate_shexp.weight"))?,
            shared_up: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.ffn_up_shexp.weight"))?,
            shared_down: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.ffn_down_shexp.weight"))?,
        }
    };
    Ok(K2Layer {
        input_norm: crate::runtime::prepare_gguf_f32_vector(context, reader, &format!("{prefix}.attn_norm.weight"))?,
        query: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.attn_q.weight"))?,
        key: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.attn_k.weight"))?,
        value,
        attention_gate: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.attn_gate.weight"))?,
        output: crate::runtime::prepare_gguf_matrix(context, reader, &format!("{prefix}.attn_output.weight"))?,
        ffn_norm: crate::runtime::prepare_gguf_f32_vector(context, reader, &format!("{prefix}.ffn_norm.weight"))?,
        mlp,
    })
}

fn prepare_router_f32(context: &MetalContext, reader: &crate::weight::container::gguf::GgufReader, name: &str) -> Result<MetalWeight, BackendError> {
    let matrix = reader.read_matrix(name).map_err(crate::runtime::compute_error)?;
    let values = matrix.decode().map_err(crate::runtime::compute_error)?;
    context.prepare_f32(&values, matrix.rows, matrix.columns)
}

pub fn prepare_output(context: &MetalContext, source: &K2HorizonGguf, quantization: crate::weight::LmHeadQuantization) -> Result<K2OutputHead, BackendError> {
    let norm = source.final_norm().map_err(crate::runtime::compute_error)?;
    let head = source.output_head().map_err(crate::runtime::compute_error)?;
    crate::runtime::output::prepare_output_head_quantized(context, &norm, LinearWeight::gguf(&head), source.config().vocab_size, source.config().hidden_size, quantization)
}

pub struct K2MetalRuntime<'a> {
    context: &'a MetalContext,
    config: &'a K2HorizonConfig,
    layers: &'a [K2Layer<MetalWeight>],
    rope: &'a RopeTable,
}

impl<'a> K2MetalRuntime<'a> {
    pub fn new(context: &'a MetalContext, config: &'a K2HorizonConfig, layers: &'a [K2Layer<MetalWeight>], rope: &'a RopeTable) -> Self {
        Self { context, config, layers, rope }
    }

    fn routed_value(&self, input: &MetalTensor, router: &MetalWeight, bias: &MetalWeight, experts: &MetalWeight) -> Result<MetalTensor, BackendError> {
        let (MetalWeight::F32 { buffer: router, len: router_len }, MetalWeight::F32 { buffer: bias, len: bias_len }) = (router, bias) else {
            return Err(BackendError::Compute { msg: "K2-Horizon MoVA router 必须是 F32 resident 权重".to_owned() });
        };
        let cfg = self.config;
        let (expert_ids, route_weights) = if input.rows == 1 {
            crate::kernel::metal::moe::moe_router_sigmoid_decode_resident_f32(self.context, input, router, *router_len, bias, *bias_len, cfg.value_expert_count, cfg.value_expert_top_k, cfg.routed_scaling_factor)
                .map_err(|msg| BackendError::Compute { msg })?
        } else {
            let routing = crate::kernel::metal::moe::moe_router_tensor_resident_f32(self.context, input, router, *router_len, bias, *bias_len, cfg.value_expert_count, cfg.value_expert_top_k, cfg.routed_scaling_factor, 1)
                .map_err(|msg| BackendError::Compute { msg })?;
            (routing.expert_ids_buffer, routing.weights_buffer)
        };
        crate::kernel::metal::gguf::gguf_routed_value_iq3s_tensor_resident(self.context, input, experts, &expert_ids, &route_weights, cfg.value_expert_count, cfg.value_expert_top_k, cfg.num_kv_heads * cfg.head_dim)
            .map_err(|msg| BackendError::Compute { msg })
    }

    fn attention_projection(&self, cache: &mut MetalKvCache, layer: usize, weights: &K2Layer<MetalWeight>, normed: &MetalTensor, position: usize) -> Result<MetalTensor, BackendError> {
        let cfg = self.config;
        let (query, gate) = self.context.dual_linear(normed, &weights.query, &weights.attention_gate)?;
        let key = self.context.linear(normed, &weights.key)?;
        let value = match &weights.value {
            K2Value::Dense(value) => self.context.linear(normed, value)?,
            K2Value::Routed { router, bias, experts } => self.routed_value(normed, router, bias, experts)?,
        };
        let query = self.context.rope_prefix(&query, cfg.num_heads, cfg.rope_dim, RotaryLayout::SplitHalf, position, &self.rope.cos, &self.rope.sin)?;
        let key = self.context.rope_prefix(&key, cfg.num_kv_heads, cfg.rope_dim, RotaryLayout::SplitHalf, position, &self.rope.cos, &self.rope.sin)?;
        let attended = self.context.gqa_prefill_attention_cached(cache, layer, position, &query, &key, &value, &cfg.gqa_spec(), false)?;
        let gated = crate::kernel::metal::tensor::softplus_gate_scaled_tensor(self.context, &attended, &gate, std::f32::consts::LN_2, std::f32::consts::LOG2_E).map_err(|msg| BackendError::Compute { msg })?;
        self.context.linear(&gated, &weights.output)
    }

    fn attention(&self, cache: &mut MetalKvCache, layer: usize, weights: &K2Layer<MetalWeight>, hidden: &MetalTensor, position: usize) -> Result<MetalTensor, BackendError> {
        let normed = self.context.grouped_rmsnorm(hidden, &weights.input_norm, self.config.rms_eps, self.config.norm_groups)?;
        let projected = self.attention_projection(cache, layer, weights, &normed, position)?;
        self.context.add(hidden, &projected)
    }

    fn add_grouped_norm(&self, residual: &MetalTensor, projected: &MetalTensor, weight: &MetalWeight) -> Result<(MetalTensor, MetalTensor), BackendError> {
        let MetalWeight::F32 { buffer, len } = weight else {
            return Err(BackendError::Compute { msg: "K2-Horizon grouped RMSNorm 权重必须是 F32 resident".to_owned() });
        };
        crate::kernel::metal::tensor::add_f32_f16_grouped_rmsnorm_tensor_resident(self.context, residual, projected, buffer, *len, self.config.rms_eps, self.config.norm_groups).map_err(|msg| BackendError::Compute { msg })
    }

    fn prefill_layer(&self, cache: &mut MetalKvCache, experts: &mut MetalPrefillExperts, layer: usize, hidden: MetalTensor, position: usize) -> Result<MetalTensor, BackendError> {
        let weights = &self.layers[layer];
        let residual = self.attention(cache, layer, weights, &hidden, position)?;
        let input = self.context.grouped_rmsnorm(&residual, &weights.ffn_norm, self.config.rms_eps, self.config.norm_groups)?;
        match &weights.mlp {
            K2Mlp::Dense { gate, up, down } => self.context.gated_mlp_add_residual(&input, gate, up, down, &crate::moe::Activation::Silu, &residual),
            K2Mlp::Sparse { router, bias, shared_gate, shared_up, shared_down } => {
                let shared = [SharedExpertRef { gate: shared_gate, up: shared_up, down: shared_down, output_gate: None }];
                let output =
                    prefill_experts_observed(self.context, &self.config.moe_spec(), &MoeFfnRef { router_weight: router, router_bias: bias, shared_experts: &shared, selected_experts: None }, layer, experts, &input, None, |_| {})?.tensor;
                self.context.add(&residual, &output)
            }
        }
    }

    pub fn prefill(&self, cache: &mut MetalKvCache, experts: &mut MetalPrefillExperts, mut hidden: MetalTensor, position: usize) -> Result<MetalTensor, BackendError> {
        let rows = hidden.rows;
        if rows == 0 || hidden.cols != self.config.hidden_size || self.layers.len() != self.config.layer_count || position.checked_add(rows).is_none_or(|end| end > self.config.max_position_embeddings) {
            return Err(BackendError::Compute { msg: format!("K2-Horizon prefill 输入非法: position={position} hidden=[{},{}] layers={}", hidden.rows, hidden.cols, self.layers.len()) });
        }
        let result = (|| {
            for layer in 0..self.layers.len() {
                let _scope = self.context.layer_scope();
                self.context.begin_batch();
                hidden = self.prefill_layer(cache, experts, layer, hidden, position)?;
                self.context.submit_batch();
            }
            Ok(hidden)
        })();
        self.context.finish_batch();
        result
    }

    pub fn decode<S>(&self, source: &S, experts: &mut ExpertDecodePipeline<MetalMoeDecodeState>, cache: &mut MetalKvCache, mut hidden: MetalTensor, position: usize) -> Result<MetalTensor, BackendError>
    where
        S: ExpertSourceProvider,
    {
        if hidden.rows != 1 || hidden.cols != self.config.hidden_size || position >= self.config.max_position_embeddings {
            return Err(BackendError::Compute { msg: format!("K2-Horizon decode 输入非法: position={position} hidden=[{},{}]", hidden.rows, hidden.cols) });
        }
        let result = (|| {
            self.context.begin_decode_batch();
            let mut normed = self.context.grouped_rmsnorm(&hidden, &self.layers[0].input_norm, self.config.rms_eps, self.config.norm_groups)?;
            for (layer, weights) in self.layers.iter().enumerate() {
                let _scope = self.context.layer_scope();
                let attention = self.attention_projection(cache, layer, weights, &normed, position)?;
                let (residual, input) = self.add_grouped_norm(&hidden, &attention, &weights.ffn_norm)?;
                let output = match &weights.mlp {
                    K2Mlp::Dense { gate, up, down } => {
                        let activated = self.context.gated_linear(&input, gate, up, &crate::moe::Activation::Silu)?;
                        self.context.linear(&activated, down)?
                    }
                    K2Mlp::Sparse { router, bias, shared_gate, shared_up, shared_down } => {
                        let shared = [SharedExpertRef { gate: shared_gate, up: shared_up, down: shared_down, output_gate: None }];
                        let expert_source = source.source(layer).map_err(BackendError::ExpertLoad)?;
                        let next = (layer + 1 < self.config.layer_count).then(|| source.source(layer + 1).map(|next| (layer + 1, next))).transpose().map_err(BackendError::ExpertLoad)?;
                        experts.decode_inputs(
                            self.context,
                            &self.config.moe_spec(),
                            &MoeFfnRef { router_weight: router, router_bias: bias, shared_experts: &shared, selected_experts: None },
                            ExpertDecodeRequest { layer, source: expert_source, position, next },
                            RoutedMoeInputs { route: &input, expert: &input },
                        )?
                    }
                };
                if let Some(next_weights) = self.layers.get(layer + 1) {
                    (hidden, normed) = self.add_grouped_norm(&residual, &output, &next_weights.input_norm)?;
                } else {
                    hidden = self.context.add(&residual, &output)?;
                }
            }
            self.context.submit_batch();
            Ok(hidden)
        })();
        self.context.finish_batch();
        result
    }
}

pub fn token_output(context: &MetalContext, config: &K2HorizonConfig, head: &K2OutputHead, hidden: &MetalTensor) -> Result<OutputResult<MetalTensor>, BackendError> {
    crate::runtime::output::token_output(context, head, hidden, &OutputPlan { eps: config.rms_eps, norm: OutputNorm::GroupedRms { groups: config.norm_groups }, excluded_tokens: Vec::new() })
}
