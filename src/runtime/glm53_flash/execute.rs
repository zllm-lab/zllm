//! GLM-5.3-Flash 完整 Transformer core round 与贪心生成循环。
//!
//! 每轮流程:embedding → mHC 展开 → 45 层(KDA/DSA-MLA 交替,各包两层 mHC)
//! → copies 折叠回 hidden 宽;final RMSNorm + lm_head 由 OutputHead 执行。

use crate::{
    attention::{AttentionSpec, hyper_connection::HyperConnectionKernel, hyper_connection::HyperConnectionSpec, kda::KdaKernel},
    backend::{Backend, BackendError, ExpertDecodeBackend, ExpertPrefillBackend},
    moe::{
        Activation, DenseFfn,
        topk_moe::{MoeFfnRef, RoutedMoeInputs, RoutedMoeWeightsRef, SharedExpertRef, decode_routed_topk_moe, decode_shared_experts},
    },
    runtime::{
        Model,
        glm53_flash::{
            Glm53Flash,
            layer::{Glm53AttentionWeights, Glm53LayerWeights, glm53_collapse_hidden, glm53_dsa_mla_layer, glm53_expand_hidden, glm53_kda_layer},
            prepare::{Glm53MoeLayer, Glm53PreparedLayer, Glm53PreparedOutput},
        },
        output::{OutputNorm, OutputPlan, last_token_output},
    },
    weight::{expert_source::ExpertSourceProvider, model::glm53_flash::Glm53FlashWeights},
};

use super::session::Glm53FlashSessionState;

/// 从已 tokenize 的 prompt 开始贪心生成直到 EOS。
/// `embedding` 只负责把指定 token rows 放到目标 backend,不参与模型编排。
#[allow(clippy::too_many_arguments)]
pub fn generate_to_eos<B, S, F>(
    backend: &B,
    model: &Glm53Flash,
    weights: &Glm53FlashWeights,
    session: &mut Glm53FlashSessionState<B::KdaStorage, B::Cache, B::DsaState>,
    prefill_experts: &mut B::PrefillExperts,
    moe_state: &mut B::MoeState,
    expert_source: &S,
    prepared_output: &Glm53PreparedOutput<B::Weight>,
    prompt_tokens: &[u32],
    prefill_chunk_size: usize,
    sequence_capacity: usize,
    excluded_tokens: &[u32],
    mut embedding: F,
) -> Result<Vec<u32>, BackendError>
where
    B: KdaKernel + HyperConnectionKernel + crate::backend::DsaPrefillBackend + ExpertPrefillBackend + ExpertDecodeBackend,
    S: ExpertSourceProvider,
    F: FnMut(&[u32]) -> Result<B::Tensor, BackendError>,
{
    if prompt_tokens.is_empty() {
        return Err(BackendError::Compute { msg: "GLM-5.3-Flash prompt token 不能为空".to_owned() });
    }
    if prefill_chunk_size == 0 {
        return Err(BackendError::Compute { msg: "GLM-5.3-Flash prefill chunk 不能为 0".to_owned() });
    }
    if prompt_tokens.len() > sequence_capacity {
        return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash prompt tokens={} 超过 sequence capacity={sequence_capacity}", prompt_tokens.len()) });
    }

    let mut last_hidden = None;
    let mut last_rows = 0;
    for chunk in prompt_tokens.chunks(prefill_chunk_size) {
        last_rows = chunk.len();
        let hidden = embedding(chunk)?;
        if backend.token_rows(&hidden) != chunk.len() {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash embedding rows={}，期望 {}", backend.token_rows(&hidden), chunk.len()) });
        }
        last_hidden = Some(prefill_core_round(backend, model, weights, session, hidden, prefill_experts)?);
    }

    let plan = OutputPlan { eps: model.config().rms_eps, norm: OutputNorm::Rms, excluded_tokens: excluded_tokens.to_vec() };
    let mut token = last_token_output(backend, &prepared_output.head, last_hidden.as_ref().expect("非空 prompt 至少产生一个 prefill chunk"), last_rows - 1, &plan)?.token_id;
    let eos = &model.config().eos_token_ids;
    let mut generated = Vec::new();

    loop {
        generated.push(token);
        if eos.contains(&token) {
            return Ok(generated);
        }
        let consumed = prompt_tokens.len().checked_add(generated.len() - 1).ok_or_else(|| BackendError::Compute { msg: "GLM-5.3-Flash sequence position 溢出".to_owned() })?;
        if consumed >= sequence_capacity {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash 达到 sequence capacity={sequence_capacity} 仍未生成 EOS；generated={}", generated.len()) });
        }
        let hidden = embedding(&[token])?;
        token = last_token_output(backend, &prepared_output.head, &decode_core_round(backend, model, weights, session, moe_state, expert_source, hidden)?, 0, &plan)?.token_id;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_core_round<B>(
    backend: &B,
    model: &Glm53Flash,
    weights: &Glm53FlashWeights,
    session: &mut Glm53FlashSessionState<B::KdaStorage, B::Cache, B::DsaState>,
    hidden: B::Tensor,
    experts: &mut B::PrefillExperts,
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + HyperConnectionKernel + crate::backend::DsaPrefillBackend + ExpertPrefillBackend,
{
    let start_position = session.next_position();
    session.begin_round(start_position)?;
    let token_count = backend.token_rows(&hidden);
    if token_count == 0 {
        return Err(BackendError::Compute { msg: "GLM-5.3-Flash prefill 不能输入空序列".to_owned() });
    }
    let config = model.config();
    let hyper_connection = Glm53Flash::hyper_connection_spec(config);
    let mla = Glm53Flash::mla_spec(config);
    let dsa = Glm53Flash::dsa_spec(config);
    let result = (|| {
        let (kda_state, cache, dsa_state) = session.states_mut();
        let mut hidden = glm53_expand_hidden(backend, &hidden, hyper_connection.copies)?;
        for layer in 0..model.layer_count() {
            let spec = model.layer_spec(layer).map_err(|error| layer_error(layer, error.to_string()))?;
            let prepared = super::prepare::prepare_layer(backend, weights, config, model, layer)?;
            let core = match &prepared {
                Glm53PreparedLayer::Dense(prepared) => &prepared.core,
                Glm53PreparedLayer::Moe(prepared) => &prepared.core,
            };
            hidden = execute_layer(
                backend,
                model,
                &hyper_connection,
                (&mla, &dsa),
                layer,
                start_position,
                hidden,
                core,
                &spec.attention,
                |input| match &prepared {
                    Glm53PreparedLayer::Dense(prepared) => dense_ffn(backend, &prepared.feedforward, input),
                    Glm53PreparedLayer::Moe(prepared) => prefill_moe(backend, model, prepared, layer, experts, input),
                },
                kda_state,
                Some(cache),
                dsa_state,
            )?;
        }
        glm53_collapse_hidden(backend, &hidden, hyper_connection.copies)
    })();
    finish_round(session, start_position, token_count, result)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_core_round<B, S>(
    backend: &B,
    model: &Glm53Flash,
    weights: &Glm53FlashWeights,
    session: &mut Glm53FlashSessionState<B::KdaStorage, B::Cache, B::DsaState>,
    moe_state: &mut B::MoeState,
    expert_source: &S,
    hidden: B::Tensor,
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + HyperConnectionKernel + crate::backend::DsaPrefillBackend + ExpertDecodeBackend,
    S: ExpertSourceProvider,
{
    if backend.token_rows(&hidden) != 1 {
        return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash decode 每轮必须恰好一个 token，实际 rows={}", backend.token_rows(&hidden)) });
    }
    let position = session.next_position();
    session.begin_round(position)?;
    let config = model.config();
    let hyper_connection = Glm53Flash::hyper_connection_spec(config);
    let mla = Glm53Flash::mla_spec(config);
    let dsa = Glm53Flash::dsa_spec(config);
    let result = (|| {
        let (kda_state, cache, dsa_state) = session.states_mut();
        let mut hidden = glm53_expand_hidden(backend, &hidden, hyper_connection.copies)?;
        for layer in 0..model.layer_count() {
            let spec = model.layer_spec(layer).map_err(|error| layer_error(layer, error.to_string()))?;
            let prepared = super::prepare::prepare_layer(backend, weights, config, model, layer)?;
            let core = match &prepared {
                Glm53PreparedLayer::Dense(prepared) => &prepared.core,
                Glm53PreparedLayer::Moe(prepared) => &prepared.core,
            };
            hidden = execute_layer(
                backend,
                model,
                &hyper_connection,
                (&mla, &dsa),
                layer,
                position,
                hidden,
                core,
                &spec.attention,
                |input| match &prepared {
                    Glm53PreparedLayer::Dense(prepared) => dense_ffn(backend, &prepared.feedforward, input),
                    Glm53PreparedLayer::Moe(prepared) => decode_moe(backend, model, prepared, layer, expert_source, moe_state, input),
                },
                kda_state,
                Some(cache),
                dsa_state,
            )?;
        }
        glm53_collapse_hidden(backend, &hidden, hyper_connection.copies)
    })();
    finish_round(session, position, 1, result)
}

/// 一个层的统一分派:attention 类型(KDA/DSA-MLA)与 FFN(dense/MoE)独立选择,
/// FFN 经闭包注入以隔离 prefill/decode 的 expert 通道。
#[allow(clippy::too_many_arguments)]
fn execute_layer<'a, B, F>(
    backend: &B,
    model: &Glm53Flash,
    hyper_connection: &HyperConnectionSpec,
    specs: (&crate::attention::mla::MlaSpec, &crate::attention::dsa::DsaSpec),
    layer: usize,
    position: usize,
    hidden: B::Tensor,
    core: &'a Glm53LayerWeights<B::Weight>,
    attention_spec: &'a AttentionSpec,
    feedforward: F,
    kda_state: &mut crate::attention::kda::KdaState<B::KdaStorage>,
    cache: Option<&mut B::Cache>,
    dsa_state: &mut B::DsaState,
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + HyperConnectionKernel + crate::backend::DsaPrefillBackend,
    F: FnOnce(&B::Tensor) -> Result<B::Tensor, BackendError>,
{
    let (mla, dsa) = specs;
    match (&core.attention, attention_spec) {
        (Glm53AttentionWeights::Kda(_), AttentionSpec::Kda(kda)) => {
            let config = model.config();
            glm53_kda_layer(backend, kda_state, hyper_connection, core, kda, layer, position, hidden, config.rms_eps, |input| feedforward(input))
        }
        (Glm53AttentionWeights::DsaMla(_), AttentionSpec::Mla(_)) => {
            let config = model.config();
            glm53_dsa_mla_layer(backend, cache, dsa_state, hyper_connection, core, mla, dsa, layer, position, hidden, config.rms_eps, &[], &[], |input| feedforward(input))
        }
        _ => Err(BackendError::Compute { msg: format!("GLM-5.3-Flash L{layer} prepared attention 与 LayerSpec 不一致") }),
    }
}

fn dense_ffn<B: Backend>(backend: &B, weights: &DenseFfn<B::Weight>, input: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let activated = backend.gated_linear(input, &weights.gate, &weights.up, &Activation::Silu)?;
    backend.linear(&activated, &weights.down)
}

fn prefill_moe<B: ExpertPrefillBackend>(backend: &B, model: &Glm53Flash, prepared: &Glm53MoeLayer<B::Weight>, layer: usize, experts: &mut B::PrefillExperts, input: &B::Tensor) -> Result<B::Tensor, BackendError> {
    let spec = Glm53Flash::moe_spec(model.config());
    let shared = [SharedExpertRef { gate: &prepared.shared.gate, up: &prepared.shared.up, down: &prepared.shared.down, output_gate: None }];
    let weights = MoeFfnRef { router_weight: &prepared.router_weight, router_bias: &prepared.router_bias, shared_experts: &shared, selected_experts: None };
    crate::moe::prefill::prefill_experts_untraced(backend, &spec, &weights, layer, experts, input, None)
}

fn decode_moe<B, S>(backend: &B, model: &Glm53Flash, prepared: &Glm53MoeLayer<B::Weight>, layer: usize, source: &S, state: &mut B::MoeState, input: &B::Tensor) -> Result<B::Tensor, BackendError>
where
    B: ExpertDecodeBackend,
    S: ExpertSourceProvider,
{
    let spec = Glm53Flash::moe_spec(model.config());
    let shared = [SharedExpertRef { gate: &prepared.shared.gate, up: &prepared.shared.up, down: &prepared.shared.down, output_gate: None }];
    let weights = MoeFfnRef { router_weight: &prepared.router_weight, router_bias: &prepared.router_bias, shared_experts: &shared, selected_experts: None };
    let shared_output = decode_shared_experts(backend, &spec, &weights, input)?;
    let expert = source.source(layer).map_err(BackendError::ExpertLoad)?;
    let routed_weights = RoutedMoeWeightsRef { router: &prepared.router_weight, bias: &prepared.router_bias, selected_experts: None };
    let (mut output, _) = decode_routed_topk_moe(backend, &spec, routed_weights, layer, expert, state, RoutedMoeInputs { route: input, expert: input }, |_, _| Ok(None))?;
    if let Some(shared_output) = shared_output {
        output = backend.add(&output, &shared_output)?;
    }
    Ok(output)
}

fn finish_round<S, C, D, T>(session: &mut Glm53FlashSessionState<S, C, D>, position: usize, token_count: usize, result: Result<T, BackendError>) -> Result<T, BackendError> {
    match result {
        Ok(output) => {
            session.commit(position, token_count)?;
            Ok(output)
        }
        Err(error) => {
            session.poison();
            Err(error)
        }
    }
}

fn layer_error(layer: usize, msg: String) -> BackendError {
    BackendError::Compute { msg: format!("GLM-5.3-Flash L{layer} model spec: {msg}") }
}
