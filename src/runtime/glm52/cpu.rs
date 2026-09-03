//! GLM-5.2 × CPU reference 组合。

use crate::{
    attention::{AttentionSpec, mla::MlaSpec, rope::RopeTable},
    backend::cpu::{CpuContext, CpuDsaState, CpuKvCache, CpuPrefillExperts, CpuWeight},
    kernel::cpu::{CpuTensor, matmul::matmul, rmsnorm},
    moe::{
        UncachedMoeState,
        expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
    },
    runtime::{
        Model,
        expert_pipeline::ExpertDecodePipeline,
        glm52::{Glm52, Glm52Config, Glm52DecodeLayer, Glm52PrefillLayerKind, glm52_decode_layers, glm52_dense_prefill_layer, glm52_moe_prefill_layer, glm52_prefill, prepare_glm52_decode_layers, print_expert_cache},
    },
    tokenizer::{Detokenizer, Tokenizer},
    weight::{Glm52Weights, expert_source::Glm52ExpertSources},
};
use std::time::Instant;

pub fn run(
    model_dir: &std::path::Path,
    prompt: &str,
    max_seq_len: usize,
    decode_steps: usize,
    expert_cache_gib: Option<usize>,
    expert_prefetch_count: Option<usize>,
    nvfp4_root: Option<&std::path::Path>,
    ct_root: Option<&std::path::Path>,
    gguf_root: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Glm52Config::standard();
    let tokenizer_path = crate::runtime::glm52::tokenizer_path(model_dir, ct_root, nvfp4_root);
    let tokenizer = if tokenizer_path.is_file() {
        Tokenizer::new(&tokenizer_path)?
    } else if let Some(root) = gguf_root {
        Glm52Weights::open_gguf(root, cfg.clone())?.gguf_source()?.tokenizer()?
    } else {
        Tokenizer::new(&tokenizer_path)?
    };
    let tokens = tokenizer.tokenize(prompt.as_bytes());
    if tokens.is_empty() {
        return Err("prompt 不能为空".into());
    }
    if tokens.len() > max_seq_len {
        return Err(format!("prompt {} tokens 超过 max_seq_len {}", tokens.len(), max_seq_len).into());
    }

    let weights = crate::runtime::glm52::open_weights(&cfg, model_dir, ct_root, nvfp4_root, gguf_root)?;
    let model = Glm52::standard();
    let mla = match &model.layer_spec(0)?.attention {
        AttentionSpec::Mla(spec) => spec.clone(),
        _ => return Err("GLM-5.2 dense 层不是 MLA".into()),
    };
    let experts_dir = model_dir.join("experts");
    let nvfp4_source = weights.nvfp4_experts().map(|source| source.with_archive_dir(&experts_dir));
    // decode_expert_sources 延迟到 decode 需要 时才构造(CT + decode_steps=0 时不需要)。
    // 见下方 decode 循环前的构造。
    let backend = CpuContext;
    let rope = RopeTable::precompute(max_seq_len, mla.qk_rope_head_dim, mla.rope_theta);
    let mut cache = CpuKvCache::new(cfg.layer_count);
    let mut dsa_state = CpuDsaState::new(cfg.layer_count, max_seq_len, cfg.index_head_dim, cfg.index_top_k)?;
    let mut hidden = CpuTensor { data: weights.embedding_rows(&tokens)?, rows: tokens.len(), cols: cfg.hidden_size };

    let prefill_started = Instant::now();
    let mut prefill_experts = ();
    hidden = glm52_prefill(
        &backend,
        &cfg,
        tokens.len(),
        0,
        hidden,
        &mut prefill_experts,
        |_, _, _| Ok(()),
        |_, layer, kind, hidden| {
            let layer_started = Instant::now();
            let output = match kind {
                Glm52PrefillLayerKind::Dense => {
                    let resident = crate::runtime::glm52::load_prepare_dense_prefill_layer(&backend, &cfg, &mla, &weights, layer, false)?;
                    glm52_dense_prefill_layer(&backend, &cfg, &mla, &resident, layer, None, Some(&mut dsa_state), &hidden, &rope, Some(&mut cache), 0)?
                }
                Glm52PrefillLayerKind::Moe => {
                    let resident = crate::runtime::glm52::load_prepare_moe_prefill_layer(&backend, &cfg, &mla, &weights, layer, false)?;
                    let mut experts = if let Some(source) = &nvfp4_source {
                        CpuPrefillExperts::nvfp4(source.clone())
                    } else if weights.source_is_ct() {
                        let ct_source = weights.ct_source().map_err(crate::backend::BackendError::ExpertLoad)?;
                        CpuPrefillExperts::ct(ct_source)
                    } else if weights.source_is_gguf() {
                        let gguf_source = weights.gguf_source().map_err(crate::backend::BackendError::ExpertLoad)?;
                        CpuPrefillExperts::gguf(gguf_source)
                    } else {
                        CpuPrefillExperts::fp8(model_dir, cfg.expert_intermediate_size, cfg.hidden_size, cfg.expert_count).map_err(crate::backend::BackendError::ExpertLoad)?
                    };
                    glm52_moe_prefill_layer(&backend, &cfg, &mla, &resident, layer, &mut experts, None, Some(&mut dsa_state), &hidden, &rope, Some(&mut cache), 0)?
                }
            };
            eprintln!("[prefill-layer] layer={layer} rows={} wall={:.3}s", output.rows, layer_started.elapsed().as_secs_f64());
            Ok(output)
        },
    )
    .map_err(|error| format!("CPU full-token prefill: {error:?}"))?;
    eprintln!("[prefill] tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64());

    let last_hidden = hidden.row(tokens.len() - 1).to_vec();
    hidden = CpuTensor { data: last_hidden, rows: 1, cols: cfg.hidden_size };
    if decode_steps == 0 {
        return Ok(());
    }

    // decode_expert_sources:decode 需要 时才构造(decode_steps>0 保证到这里)。
    let decode_expert_sources = super::decode_expert_sources(&weights, nvfp4_source.as_ref(), model_dir, &cfg)?;

    let prepare_started = Instant::now();
    let layers = if decode_steps > 1 { prepare_glm52_decode_layers(&backend, &cfg, &mla, &weights).map_err(|error| format!("准备 CPU decode 层: {error:?}"))? } else { Vec::new() };
    eprintln!("[decode-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());
    let expert_prefetch_count = expert_prefetch_count.unwrap_or(crate::runtime::DEFAULT_DECODE_PREFETCH_COUNT);
    let mut moe_state = ExpertDecodePipeline::new(
        UncachedMoeState::default(),
        ExpertPredictorConfig {
            first_layer: cfg.dense_layer_count,
            layer_count: cfg.layer_count - cfg.dense_layer_count,
            expert_count: cfg.expert_count,
            routed_top_k: cfg.expert_top_k,
            prefetch_count: expert_prefetch_count,
            weights: ExpertPredictorWeights::default(),
        },
    )?;
    if expert_cache_gib.is_some() {
        eprintln!("[expert-routing] backend=CPU policy=uncached --expert-cache-gib ignored");
    }
    eprintln!("[expert-routing] backend=CPU policy=uncached prefetch={expert_prefetch_count}");

    let final_norm = weights.final_norm()?;
    let lm_head = if weights.source_is_nvfp4() { weights.lm_head_nvfp4()?.into_iter().map(|value| value.to_f32()).collect() } else { weights.lm_head()? };
    let detokenizer = if tokenizer_path.is_file() { Detokenizer::load(&tokenizer_path)? } else { weights.gguf_source()?.detokenizer()? };
    crate::runtime::generation::run_generation(
        &mut hidden,
        crate::runtime::generation::GenerationLimits { prompt_tokens: tokens.len(), max_tokens: decode_steps, max_sequence_length: max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |hidden, _| Ok::<_, Box<dyn std::error::Error>>(argmax(&logits(hidden.row(0), &final_norm, &lm_head, &cfg))),
        |token, _| crate::runtime::generation::write_token(&detokenizer, token, true),
        |hidden, token, position, step| {
            let decode_started = Instant::now();
            *hidden = forward_token(token, position, &backend, &cfg, &mla, &layers, &weights, &decode_expert_sources, &rope, &mut cache, &mut dsa_state, &mut moe_state)?;
            eprintln!("\n[decode {}] position={} wall={:.3}s", step + 1, position, decode_started.elapsed().as_secs_f64());
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    print_expert_cache(&moe_state);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_token(
    token: u32,
    position: usize,
    backend: &CpuContext,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    layers: &[Glm52DecodeLayer<CpuWeight>],
    weights: &Glm52Weights,
    expert_sources: &Glm52ExpertSources,
    rope: &RopeTable,
    cache: &mut CpuKvCache,
    dsa_state: &mut CpuDsaState,
    moe_state: &mut ExpertDecodePipeline<UncachedMoeState>,
) -> Result<CpuTensor, Box<dyn std::error::Error>> {
    let hidden = CpuTensor { data: weights.embedding_rows(&[token])?, rows: 1, cols: cfg.hidden_size };
    glm52_decode_layers(backend, cfg, mla, layers, expert_sources, moe_state, dsa_state, hidden, rope, cache, position).map_err(|error| format!("CPU decode position={position}: {error:?}").into())
}

fn logits(hidden: &[f32], norm_weight: &[f32], lm_head: &[f32], cfg: &Glm52Config) -> Vec<f32> {
    let mut normalized = vec![0.0; cfg.hidden_size];
    rmsnorm(hidden, norm_weight, cfg.rms_eps, &mut normalized);
    let mut logits = vec![0.0; cfg.vocab_size];
    matmul(&normalized, lm_head, 1, cfg.hidden_size, cfg.vocab_size, &mut logits);
    logits
}

fn argmax(values: &[f32]) -> u32 {
    values.iter().enumerate().max_by(|(_, left), (_, right)| left.total_cmp(right)).map_or(0, |(index, _)| index as u32)
}
