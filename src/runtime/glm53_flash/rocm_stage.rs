//! GLM-5.3-Flash ROCm tail stage:接收 head 的 mHC 展开态并完成 L44 与采样。
//! MTP 启用时承担 speculative 状态机:K+1 行 verify、greedy 前缀接受、
//! 拒绝回退(KDA 字节快照 + retained 前缀重放)与 MTP 递归 draft。

use crate::{
    backend::{BackendError, rocm::RocmTensor},
    config::{BackendConfig, StageModelConfig, StageProcessConfig, StageTransportConfig},
    runtime::speculative::{SpeculativeStats, verify_samples},
    server::stage_transport::{RequestId, StageMessage, StageTransport},
};
use std::time::Instant;

use super::{
    Glm53FlashConfig,
    rocm_engine::{Engine, Options},
};

/// tail 的 speculative 会话状态;MTP 关闭时 `enabled=false` 走单 token 路径。
struct Speculative {
    enabled: bool,
    /// 已确认未 forward 的 anchor(下轮 Verify 行 0 输入)。
    anchor: u32,
    /// anchor 前一位置的主干 collapse hidden(MTP 移位输入)。
    anchor_hidden: Option<RocmTensor>,
    /// 下轮 Verify 行 1.. 的递归 draft 输入。
    drafts: Vec<u32>,
    /// 配置的递归深度上限。
    draft_tokens: usize,
    stats: SpeculativeStats,
    /// depth[i] 表示第 i+1 个 draft 被 target 接受的轮数。
    accepted_depths: [u64; 8],
    verify_micros: u128,
    rollback_micros: u128,
    rebuild_micros: u128,
}

impl Speculative {
    fn disabled() -> Self {
        Self { enabled: false, anchor: 0, anchor_hidden: None, drafts: Vec::new(), draft_tokens: 0, stats: SpeculativeStats::default(), accepted_depths: [0; 8], verify_micros: 0, rollback_micros: 0, rebuild_micros: 0 }
    }
}

pub fn run(config: StageProcessConfig) -> Result<(), Box<dyn std::error::Error>> {
    let BackendConfig::Rocm(backend) = config.backend else { unreachable!("配置校验已保证 ROCm") };
    let StageModelConfig::Glm53Flash(model) = config.model else { unreachable!("入口已按模型分发") };
    let StageTransportConfig::Listen { iroh } = config.transport else { unreachable!("配置校验已保证 tail listen") };
    let mut engine = Engine::load(&Options {
        weights_directory: model.weights_directory,
        devices: backend.devices,
        layer_ends: model.layers.device_layer_ends,
        layer_start: model.layers.start,
        layer_end: model.layers.end,
        max_sequence_length: model.max_sequence_length,
        prefill_chunk_size: 1,
        mtp: model.mtp,
    })?;
    let mtp = model.mtp;
    let mtp_draft_tokens = model.mtp_draft_tokens;
    let memory = engine.device_memory()?;
    let listener = StageTransport::bind(iroh.runtime()?).map_err(|error| format!("启动 GLM-5.3-Flash tail: {error}"))?;
    let mut link = listener.accept().map_err(|error| format!("接收 GLM-5.3-Flash head: {error}"))?;
    link.send_device_memory(&memory, Some(1))?;
    let mut active = None;
    let mut speculative = Speculative::disabled();
    let mut mtp_prompt: Option<Vec<u32>> = None;
    loop {
        let frame = match link.recv() {
            Ok(frame) => frame,
            Err(error) => {
                eprintln!("[glm53-tail-reconnect] {error}");
                link.accept_reconnect()?;
                engine.reset().map_err(backend_error)?;
                active = None;
                speculative = Speculative::disabled();
                link.send_device_memory(&memory, Some(1))?;
                continue;
            }
        };
        match frame.message {
            StageMessage::Open { tail_sampling, .. } => {
                if !tail_sampling {
                    return Err("GLM-5.3-Flash tail 必须负责输出头与采样".into());
                }
                engine.reset().map_err(backend_error)?;
                active = Some(frame.request_id);
                // MTP 只有在 head 显式发送 MtpContext 时才启用；多模态
                // checkpoint 的 head 不发送该消息，因此始终走主模型解码。
                speculative = Speculative::disabled();
                mtp_prompt = None;
                link.send_ready(frame.request_id, 0)?;
            }
            StageMessage::MtpContext { prompt_tokens, max_decode, draft_tokens } => {
                require_active(active, frame.request_id)?;
                if !mtp {
                    return Err("GLM-5.3-Flash tail 未启用 MTP 却收到 MtpContext".into());
                }
                if mtp_prompt.is_some() {
                    return Err("GLM-5.3-Flash tail 重复收到 MtpContext".into());
                }
                if max_decode == 0 || draft_tokens != mtp_draft_tokens {
                    return Err(format!("GLM-5.3-Flash MTP context max_decode={max_decode} drafts={draft_tokens}，配置 drafts={mtp_draft_tokens}").into());
                }
                speculative.enabled = true;
                speculative.draft_tokens = draft_tokens;
                mtp_prompt = Some(prompt_tokens);
            }
            StageMessage::Prefill { position, rows, cols, values, selection, aux_values, aux_taps } => {
                require_active(active, frame.request_id)?;
                if !selection.is_empty() || !aux_values.is_empty() || aux_taps != 0 {
                    return Err("GLM-5.3-Flash stage boundary 只传 mHC hidden".into());
                }
                let request_id = frame.request_id;
                // MTP 的 prompt tokens 由 head 在 Prefill 前发 MtpContext 提供。
                let prompt_tokens = if speculative.enabled { mtp_prompt.take() } else { None };
                if speculative.enabled && prompt_tokens.is_none() {
                    return Err("GLM-5.3-Flash tail prefill 前未收到 MtpContext".into());
                }
                let mut first = Some((position, rows, cols, values));
                let mut done_tokens = None;
                let (hidden, rows) = engine
                    .prefill_hidden_pipeline_with_mtp(
                        || {
                            if let Some(input) = first.take() {
                                return Ok(Some(input));
                            }
                            let frame = link.recv().map_err(|msg| BackendError::Compute { msg })?;
                            if frame.request_id != request_id {
                                return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash tail prefill request={} 期望={request_id}", frame.request_id) });
                            }
                            match frame.message {
                                StageMessage::Prefill { position, rows, cols, values, selection, aux_values, aux_taps } => {
                                    if !selection.is_empty() || !aux_values.is_empty() || aux_taps != 0 {
                                        return Err(BackendError::Compute { msg: "GLM-5.3-Flash stage boundary 只传 mHC hidden".to_owned() });
                                    }
                                    Ok(Some((position, rows, cols, values)))
                                }
                                StageMessage::PrefillDone { tokens } => {
                                    done_tokens = Some(tokens);
                                    Ok(None)
                                }
                                other => Err(BackendError::Compute { msg: format!("GLM-5.3-Flash tail prefill 期望 Prefill/PrefillDone，实际 {other:?}") }),
                            }
                        },
                        prompt_tokens,
                    )
                    .map_err(backend_error)?;
                let tokens = done_tokens.ok_or("GLM-5.3-Flash tail pipeline 未收到 PrefillDone")?;
                if engine.position() != tokens {
                    return Err(format!("GLM-5.3-Flash tail prefill tokens={tokens}，当前 position={}", engine.position()).into());
                }
                let token = engine.sample_hidden(&hidden, rows - 1).map_err(backend_error)?;
                if speculative.enabled {
                    // MTP 移位:cache 位置 tokens-1 吃 (t_n, h_{n-1})，随后递归 K 步。
                    let hidden_row = engine.collapse_row(&hidden, rows - 1).map_err(backend_error)?;
                    let drafts = if is_eos(token) { Vec::new() } else { engine.mtp_drafts(token, &hidden_row, tokens - 1, speculative.draft_tokens).map_err(backend_error)? };
                    speculative.anchor = token;
                    speculative.anchor_hidden = Some(hidden_row);
                    speculative.drafts = drafts.clone();
                    link.send_speculative(request_id, &[token], 0, &drafts, is_eos(token))?;
                } else {
                    link.send_token(request_id, token, is_eos(token))?;
                }
            }
            StageMessage::PrefillDone { .. } => return Err("GLM-5.3-Flash tail 在首个 Prefill 前收到 PrefillDone".into()),
            StageMessage::Decode { position, cols, values, selection, aux_values, aux_taps, .. } => {
                require_active(active, frame.request_id)?;
                if !selection.is_empty() || !aux_values.is_empty() || aux_taps != 0 {
                    return Err("GLM-5.3-Flash decode boundary 只传 mHC hidden".into());
                }
                let hidden = engine.forward_hidden_bits(position, 1, cols, values).map_err(backend_error)?;
                let token = engine.sample_hidden(&hidden, 0).map_err(backend_error)?;
                link.send_token(frame.request_id, token, is_eos(token))?;
            }
            StageMessage::Verify { position, rows, cols, values, selection, aux_values, aux_taps, .. } => {
                require_active(active, frame.request_id)?;
                if !speculative.enabled {
                    return Err("GLM-5.3-Flash tail 未启用 MTP 却收到 Verify".into());
                }
                if !selection.is_empty() || !aux_values.is_empty() || aux_taps != 0 {
                    return Err("GLM-5.3-Flash verify boundary 只传 mHC hidden".into());
                }
                if rows != speculative.drafts.len() + 1 {
                    return Err(format!("GLM-5.3-Flash MTP verify rows={rows}，drafts={} 期望 K+1", speculative.drafts.len()).into());
                }
                let verify_inputs = std::iter::once(speculative.anchor).chain(speculative.drafts.iter().copied()).collect::<Vec<_>>();
                let verify_started = Instant::now();
                let hidden = engine.verify_bits_collected(position, rows, cols, values).map_err(backend_error)?;
                let samples = engine.sample_all_rows(&hidden, rows).map_err(backend_error)?;
                speculative.verify_micros += verify_started.elapsed().as_micros();
                let outcome = verify_samples(&samples, &speculative.drafts, &Glm53FlashConfig::standard().eos_token_ids).map_err(backend_error)?;
                speculative.stats.record(speculative.drafts.len(), &outcome);
                for depth in 0..outcome.accepted_drafts {
                    speculative.accepted_depths[depth] += 1;
                }
                let request_id = frame.request_id;
                if outcome.eos {
                    // 请求即将 Delete/reset，不为不可见后缀做回退和下一轮 draft。
                    link.send_speculative(request_id, &outcome.tokens, outcome.retained_rows, &[], true)?;
                } else {
                    if outcome.retained_rows < rows {
                        let rollback_started = Instant::now();
                        engine.rollback_verify(position + outcome.retained_rows).map_err(backend_error)?;
                        speculative.rollback_micros += rollback_started.elapsed().as_micros();
                    } else {
                        engine.commit_verify();
                    }
                    let anchor = *outcome.tokens.last().expect("verify 至少确认一个 token");
                    let previous_hidden = speculative.anchor_hidden.as_ref().ok_or("GLM-5.3-Flash MTP verify 缺少 anchor hidden")?;
                    let rebuild_started = Instant::now();
                    let (drafts, anchor_hidden) = engine.mtp_rebuild_drafts(&verify_inputs, &hidden, previous_hidden, position, outcome.retained_rows, anchor, speculative.draft_tokens).map_err(backend_error)?;
                    speculative.rebuild_micros += rebuild_started.elapsed().as_micros();
                    speculative.anchor = anchor;
                    speculative.anchor_hidden = Some(anchor_hidden);
                    speculative.drafts = drafts.clone();
                    link.send_speculative(request_id, &outcome.tokens, outcome.retained_rows, &drafts, false)?;
                }
            }
            StageMessage::Delete => {
                require_active(active, frame.request_id)?;
                engine.reset().map_err(backend_error)?;
                if speculative.enabled && speculative.stats.rounds > 0 {
                    let stats = speculative.stats;
                    eprintln!(
                        "[glm53-mtp] rounds={} proposed={} accepted={} rate={:.1}% emitted={} mean={:.2} depths={:?} verify_ms={:.1} rollback_ms={:.1} rebuild_ms={:.1}",
                        stats.rounds,
                        stats.proposed,
                        stats.accepted,
                        stats.accepted as f64 * 100.0 / stats.proposed.max(1) as f64,
                        stats.emitted,
                        stats.emitted as f64 / stats.rounds as f64,
                        &speculative.accepted_depths[..speculative.draft_tokens],
                        speculative.verify_micros as f64 / 1000.0,
                        speculative.rollback_micros as f64 / 1000.0,
                        speculative.rebuild_micros as f64 / 1000.0,
                    );
                }
                speculative = if mtp { Speculative { enabled: true, ..Speculative::disabled() } } else { Speculative::disabled() };
                active = None;
            }
            other => return Err(format!("GLM-5.3-Flash tail 不支持 stage message: {other:?}").into()),
        }
    }
}

fn is_eos(token: u32) -> bool {
    Glm53FlashConfig::standard().eos_token_ids.contains(&token)
}

fn require_active(active: Option<RequestId>, request: RequestId) -> Result<(), Box<dyn std::error::Error>> {
    if active == Some(request) { Ok(()) } else { Err(format!("GLM-5.3-Flash tail request={request} 与 active={active:?} 不一致").into()) }
}

fn backend_error(error: crate::backend::BackendError) -> String {
    format!("{error:?}")
}
