//! MiniCPM5-2B + DSpark 推测解码评测：drafter 一次提整块，目标批量验证，
//! 贪心语义 = 与目标逐行 argmax 严格前缀匹配 + bonus token。
//!
//! 复用：`runtime::dspark`（模型无关 drafter）、`minicpm5_text_hidden_with_captures`
//! （带 hidden 捕获的目标前向）、`minicpm5_token_output`（目标输出步）、
//! `MetalKvCache::truncate`（接受前缀后的 KV 回滚）。

use super::metal_session::MiniCpm5MetalSession;
use crate::{
    backend::{
        Backend,
        metal::{MetalContext, MetalTensor},
    },
    runtime::{
        dspark::DsparkTargetCache,
        minicpm5::{dspark::Minicpm5DsparkRuntime, minicpm5_rope_table, minicpm5_text_hidden_with_captures, minicpm5_token_output},
        speculative::SpeculativeBlock,
    },
};
use std::path::Path;
use std::time::Instant;

const QUESTIONS: &[&str] = &[
    "世界上最高的山峰",
    "请记住我的名字叫小石，只回答记住了。",
    "计算17加25，并解释步骤。",
    "写一个Python函数，返回列表中的最大值。",
    "Translate into English: 今天下午我们一起去图书馆。",
];

fn no_think_prompt(question: &str) -> String {
    format!("<s><|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

fn is_stop(token: u32) -> bool {
    token == 1 || token == 130073
}

/// 纯贪心基线（同一 Metal 会话，返回 token 流与 decode 墙钟，不含 prefill）。
fn plain_greedy(session: &MiniCpm5MetalSession, prompt: &str, max_tokens: usize) -> Result<(Vec<u32>, f64), String> {
    let ids = session.tokenize(prompt);
    let mut sequence = session.prefill(ids)?;
    let mut tokens = Vec::new();
    let started = Instant::now();
    for _ in 0..max_tokens {
        let token = session.next_token(&sequence, None)?;
        if is_stop(token) {
            break;
        }
        tokens.push(token);
        session.decode_token(&mut sequence, token)?;
    }
    Ok((tokens, started.elapsed().as_secs_f64()))
}

/// 推测解码一轮的验证：[anchor + drafts] 批量过目标，逐行 argmax；
/// 返回 (predicted[rows], verify 批 hidden 捕获)。
#[allow(clippy::too_many_arguments)]
fn verify_block(
    ctx: &MetalContext,
    session: &MiniCpm5MetalSession,
    draft: &Minicpm5DsparkRuntime,
    cache: &mut crate::backend::metal::MetalKvCache,
    rope: &crate::attention::rope::RopeTable,
    rows: &[u32],
    position: usize,
) -> Result<(Vec<u32>, Vec<MetalTensor>), String> {
    let config = session.config();
    let embedding = session.weights().embedding_rows_f32(rows)?;
    let input = ctx.tensor_from_f32(&embedding, rows.len(), config.hidden_size)?;
    let (hidden, captures) = minicpm5_text_hidden_with_captures(
        ctx,
        config,
        session.layers(),
        Some(cache),
        input,
        rope,
        position,
        draft.capture_layers(),
    )
    .map_err(|error| format!("MiniCPM5 verify 前向: {error:?}"))?;
    let head = session.output_head();
    let mut predicted = Vec::with_capacity(rows.len());
    for row in 0..rows.len() {
        let hidden_row = ctx.select_row(&hidden, row).map_err(|error| format!("verify select_row: {error:?}"))?;
        let token = minicpm5_token_output(ctx, config, head, &hidden_row, &[])
            .map_err(|error| format!("verify 输出步: {error:?}"))?
            .token_id;
        predicted.push(token);
    }
    Ok((predicted, captures))
}

fn run_speculative(
    session: &MiniCpm5MetalSession,
    draft: &Minicpm5DsparkRuntime,
    prompt: &str,
    max_tokens: usize,
) -> Result<serde_json::Value, String> {
    let ctx = session.context();
    let config = session.config();
    let ids = session.tokenize(prompt);
    let mut cache = session.allocate_cache()?;
    let mut draft_cache = DsparkTargetCache::new();
    let rope = minicpm5_rope_table(config, session_max_seq(session));

    // prefill + 5 层捕获 → drafter 上下文预热
    let embedding = session.weights().embedding_rows_f32(&ids)?;
    let input = ctx.tensor_from_f32(&embedding, ids.len(), config.hidden_size)?;
    let (hidden, captures) = minicpm5_text_hidden_with_captures(
        ctx,
        config,
        session.layers(),
        Some(&mut cache),
        input,
        &rope,
        0,
        draft.capture_layers(),
    )
    .map_err(|error| format!("MiniCPM5 prefill: {error:?}"))?;
    let aux = draft.project_captures(ctx, &captures).map_err(|error| format!("DSpark 投影: {error:?}"))?;
    draft.extend_cache(ctx, &mut draft_cache, &aux, 0).map_err(|error| format!("DSpark 预热: {error:?}"))?;
    let mut last_aux = ctx.select_row(&aux, ids.len() - 1).map_err(|error| format!("DSpark warm 行: {error:?}"))?;
    let mut warm_position = ids.len() - 1;

    // 首 token 由目标自身给出
    let last = ctx.select_row(&hidden, ids.len() - 1).map_err(|error| format!("select last: {error:?}"))?;
    let mut anchor = minicpm5_token_output(ctx, config, session.output_head(), &last, &[])
        .map_err(|error| format!("首 token: {error:?}"))?
        .token_id;
    let mut tokens: Vec<u32> = vec![anchor];
    let mut sequence_tokens = ids.clone(); // KV 覆盖 0..ids.len()；anchor 尚未入 KV

    let mut rounds = 0usize;
    let mut accepted_total = 0usize;
    let mut accepted_histogram = [0usize; 8];
    let draft_started = Instant::now();
    while tokens.len() < max_tokens && !is_stop(anchor) {
        let position = sequence_tokens.len();
        // drafter：anchor + mask 块
        let mut noise_ids = vec![draft.spec.mask_token_id; draft.draft_tokens];
        noise_ids[0] = anchor;
        let noise_embedding = session.weights().embedding_rows_f32(&noise_ids)?;
        let noise = ctx.tensor_from_f32(&noise_embedding, draft.draft_tokens, config.hidden_size)?;
        let block: SpeculativeBlock = draft
            .draft_block(ctx, session.output_head().lm_head(), &mut draft_cache, anchor, noise, &last_aux, warm_position, position)
            .map_err(|error| format!("DSpark draft: {error:?}"))?;
        // 目标批量验证 [anchor + drafts]
        let mut rows = vec![anchor];
        rows.extend_from_slice(&block.drafts);
        let (predicted, captures) = verify_block(ctx, session, draft, &mut cache, &rope, &rows, position)?;
        let mut accepted = 0usize;
        while accepted < block.drafts.len() && block.drafts[accepted] == predicted[accepted] {
            accepted += 1;
        }
        let bonus = predicted[accepted];
        let committed = accepted + 1;
        accepted_histogram[accepted.min(7)] += 1;
        accepted_total += accepted;
        rounds += 1;
        // KV 回滚到接受前缀；drafter 上下文推进 committed 行
        cache.truncate(position + committed);
        let aux = draft.project_captures(ctx, &captures).map_err(|error| format!("DSpark 投影: {error:?}"))?;
        let warm_rows: Vec<u32> = (0..committed as u32).collect();
        let warm = ctx.select_rows(&aux, &warm_rows).map_err(|error| format!("DSpark warm 选行: {error:?}"))?;
        draft.extend_cache(ctx, &mut draft_cache, &warm, position).map_err(|error| format!("DSpark 推进: {error:?}"))?;
        last_aux = ctx.select_row(&aux, committed - 1).map_err(|error| format!("DSpark warm 行: {error:?}"))?;
        warm_position = position + committed - 1;
        // 提交：drafts[..accepted] + bonus；停止符不计数（与基线口径一致）。
        let mut stopped = false;
        for token in &block.drafts[..accepted] {
            if is_stop(*token) {
                stopped = true;
                break;
            }
            tokens.push(*token);
            sequence_tokens.push(*token);
        }
        if stopped {
            break;
        }
        if is_stop(bonus) {
            break;
        }
        tokens.push(bonus);
        sequence_tokens.push(bonus);
        anchor = bonus;
    }
    let decode_seconds = draft_started.elapsed().as_secs_f64();
    Ok(serde_json::json!({
        "tokens": tokens,
        "token_count": tokens.len(),
        "rounds": rounds,
        "accepted_total": accepted_total,
        "accepted_per_round": if rounds > 0 { accepted_total as f64 / rounds as f64 } else { 0.0 },
        "emitted_per_round": if rounds > 0 { tokens.len() as f64 / rounds as f64 } else { 0.0 },
        "accepted_histogram": accepted_histogram,
        "decode_seconds": decode_seconds,
        "tok_s": if decode_seconds > 0.0 { tokens.len() as f64 / decode_seconds } else { 0.0 },
    }))
}

fn session_max_seq(session: &MiniCpm5MetalSession) -> usize {
    session.max_seq_len()
}

/// 入口：基线贪心 vs DSpark 推测解码，同一 Metal 会话同一 prompt。
pub fn run_dspark_eval(model: &Path, draft_directory: &Path, max_tokens: usize) -> Result<serde_json::Value, String> {
    let session = MiniCpm5MetalSession::load_with_replay(model, 4096, false, true, crate::weight::LmHeadQuantization::Native)?;
    let draft = Minicpm5DsparkRuntime::load(session.context(), draft_directory, 4096).map_err(|error| format!("DSpark drafter: {error:?}"))?;
    let mut results = Vec::new();
    for question in QUESTIONS {
        let prompt = no_think_prompt(question);
        let (baseline_tokens, baseline_seconds) = plain_greedy(&session, &prompt, max_tokens)?;
        let speculative = run_speculative(&session, &draft, &prompt, max_tokens)?;
        let baseline_matches = baseline_tokens == speculative["tokens"].as_array().map(|tokens| tokens.iter().filter_map(|token| token.as_u64().map(|id| id as u32)).collect::<Vec<u32>>()).unwrap_or_default();
        results.push(serde_json::json!({
            "question": question,
            "baseline": { "tokens": baseline_tokens, "token_count": baseline_tokens.len(), "tok_s": if baseline_seconds > 0.0 { baseline_tokens.len() as f64 / baseline_seconds } else { 0.0 }, "decode_seconds": baseline_seconds },
            "speculative": speculative,
            "tokens_equal_baseline": baseline_matches,
        }));
    }
    Ok(serde_json::json!({ "results": results }))
}
