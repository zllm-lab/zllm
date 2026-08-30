//! K3 完整 Transformer core round；输出头由调用方在成功后执行。

use crate::{
    attention::{
        attn_res::{self, AttnResBackend, AttnResState},
        kda::KdaKernel,
    },
    backend::{Backend, BackendError, DecodeBackend, ExpertDecodeBackend, ExpertPrefillBackend, MlaPrefillBackend},
    runtime::output::{OutputNorm, OutputPlan, last_token_output},
    runtime::{Model, kimi_k3::KimiK3},
    weight::{expert_source::ExpertSourceProvider, model::kimi_k3::KimiK3Weights},
};

use super::{
    layer::AttnResWeights,
    prepare::KimiK3PreparedOutput,
    round::{decode_layer, load_prepare_layer, prefill_layer},
    session::KimiK3SessionState,
};

/// 从已经完成 XTML tokenize 的 prompt 开始，贪心生成直到 EOS。
/// `embedding` 只负责把指定 token rows 放到目标 backend，不参与模型编排。
#[allow(clippy::too_many_arguments)]
pub fn generate_to_eos<B, S, F>(
    backend: &B,
    model: &KimiK3,
    weights: &KimiK3Weights,
    session: &mut KimiK3SessionState<B::KdaStorage, B::Cache>,
    prefill_experts: &mut B::PrefillExperts,
    moe_state: &mut B::MoeState,
    expert_source: &S,
    prepared_output: &KimiK3PreparedOutput<B::Weight>,
    prompt_tokens: &[u32],
    prefill_chunk_size: usize,
    sequence_capacity: usize,
    excluded_tokens: &[u32],
    mut embedding: F,
) -> Result<Vec<u32>, BackendError>
where
    B: KdaKernel + MlaPrefillBackend + DecodeBackend + AttnResBackend + ExpertPrefillBackend + ExpertDecodeBackend,
    S: ExpertSourceProvider,
    F: FnMut(&[u32]) -> Result<B::Tensor, BackendError>,
{
    if prompt_tokens.is_empty() {
        return Err(crate::runtime::compute_error("Kimi K3 prompt token 不能为空"));
    }
    if prefill_chunk_size == 0 {
        return Err(crate::runtime::compute_error("Kimi K3 prefill chunk 不能为 0"));
    }
    if prompt_tokens.len() > sequence_capacity {
        return Err(crate::runtime::compute_error(format!("Kimi K3 prompt tokens={} 超过 sequence capacity={sequence_capacity}", prompt_tokens.len(),)));
    }

    let mut last_hidden = None;
    let mut last_rows = 0;
    for chunk in prompt_tokens.chunks(prefill_chunk_size) {
        last_rows = chunk.len();
        let hidden = embedding(chunk)?;
        if backend.token_rows(&hidden) != chunk.len() {
            return Err(crate::runtime::compute_error(format!("Kimi K3 embedding rows={}，期望 {}", backend.token_rows(&hidden), chunk.len(),)));
        }
        last_hidden = Some(prefill_core_round(backend, model, weights, session, hidden, &prepared_output.attention_residual, prefill_experts)?);
    }

    let mut token = last_token_output(
        backend,
        &prepared_output.head,
        last_hidden.as_ref().expect("非空 prompt 至少产生一个 prefill chunk"),
        last_rows - 1,
        &OutputPlan { eps: model.config().rms_eps, norm: OutputNorm::Rms, excluded_tokens: excluded_tokens.to_vec() },
    )?
    .token_id;
    let eos = &model.config().eos_token_ids;
    let mut generated = Vec::new();

    loop {
        generated.push(token);
        if eos.contains(&token) {
            return Ok(generated);
        }
        let consumed = prompt_tokens.len().checked_add(generated.len() - 1).ok_or_else(|| crate::runtime::compute_error("Kimi K3 sequence position 溢出"))?;
        if consumed >= sequence_capacity {
            return Err(crate::runtime::compute_error(format!("Kimi K3 达到 sequence capacity={sequence_capacity} 仍未生成 EOS；generated={}", generated.len(),)));
        }
        let hidden = embedding(&[token])?;
        token = last_token_output(
            backend,
            &prepared_output.head,
            &decode_core_round(backend, model, weights, session, moe_state, expert_source, hidden, &prepared_output.attention_residual)?,
            0,
            &OutputPlan { eps: model.config().rms_eps, norm: OutputNorm::Rms, excluded_tokens: excluded_tokens.to_vec() },
        )?
        .token_id;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_core_round<B>(
    backend: &B,
    model: &KimiK3,
    weights: &KimiK3Weights,
    session: &mut KimiK3SessionState<B::KdaStorage, B::Cache>,
    mut hidden: B::Tensor,
    output_attn_res: &AttnResWeights<B::Weight>,
    experts: &mut B::PrefillExperts,
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + MlaPrefillBackend + AttnResBackend + ExpertPrefillBackend,
{
    let start_position = session.next_position();
    session.begin_round(start_position)?;
    let token_count = backend.token_rows(&hidden);
    if token_count == 0 {
        return Err(BackendError::Compute { msg: "Kimi K3 prefill 不能输入空序列".to_owned() });
    }
    let mut attn_res_state = AttnResState::new(model.config().attn_res_block_size)?;
    let result = (|| {
        let (kda_state, cache) = session.states_mut();
        for layer in 0..model.layer_count() {
            let spec = model.layer_spec(layer).map_err(|error| model_error(layer, error.to_string()))?;
            hidden = with_layer_batch(backend, || {
                let prepared = load_prepare_layer(backend, weights, layer, spec)?;
                prefill_layer(backend, kda_state, Some(cache), &mut attn_res_state, layer, start_position, hidden, &prepared, spec, model.config().rms_eps, &[], &[], experts)
            })?;
        }
        hidden = attn_res::mix(backend, &attn_res_state, &hidden, &output_attn_res.norm, &output_attn_res.projection, model.config().rms_eps)?;
        attn_res_state.finish_round(model.layer_count())?;
        Ok(hidden)
    })();
    finish_session_round(session, start_position, token_count, result)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_core_round<B, S>(
    backend: &B,
    model: &KimiK3,
    weights: &KimiK3Weights,
    session: &mut KimiK3SessionState<B::KdaStorage, B::Cache>,
    moe_state: &mut B::MoeState,
    expert_source: &S,
    mut hidden: B::Tensor,
    output_attn_res: &AttnResWeights<B::Weight>,
) -> Result<B::Tensor, BackendError>
where
    B: KdaKernel + DecodeBackend + AttnResBackend + ExpertDecodeBackend,
    S: ExpertSourceProvider,
{
    if backend.token_rows(&hidden) != 1 {
        return Err(BackendError::Compute { msg: format!("Kimi K3 decode 每轮必须恰好一个 token，实际 rows={}", backend.token_rows(&hidden)) });
    }
    let position = session.next_position();
    session.begin_round(position)?;
    let mut attn_res_state = AttnResState::new(model.config().attn_res_block_size)?;
    let result = (|| {
        let (kda_state, cache) = session.states_mut();
        for layer in 0..model.layer_count() {
            let spec = model.layer_spec(layer).map_err(|error| model_error(layer, error.to_string()))?;
            hidden = with_layer_batch(backend, || {
                let prepared = load_prepare_layer(backend, weights, layer, spec)?;
                decode_layer(backend, kda_state, cache, &mut attn_res_state, moe_state, expert_source, layer, position, hidden, &prepared, spec, model.config().rms_eps, &[], &[])
            })?;
        }
        hidden = attn_res::mix(backend, &attn_res_state, &hidden, &output_attn_res.norm, &output_attn_res.projection, model.config().rms_eps)?;
        attn_res_state.finish_round(model.layer_count())?;
        Ok(hidden)
    })();
    finish_session_round(session, position, 1, result)
}

fn with_layer_batch<B: Backend, T>(backend: &B, execute: impl FnOnce() -> Result<T, BackendError>) -> Result<T, BackendError> {
    backend.begin_batch();
    let result = execute();
    backend.finish_batch();
    result
}

fn finish_session_round<S, C, T>(session: &mut KimiK3SessionState<S, C>, position: usize, token_count: usize, result: Result<T, BackendError>) -> Result<T, BackendError> {
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

fn model_error(layer: usize, msg: String) -> BackendError {
    BackendError::Compute { msg: format!("Kimi K3 L{layer} model spec: {msg}") }
}
