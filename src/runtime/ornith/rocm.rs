//! Ornith × ROCm standalone 组合。

use std::{sync::Arc, time::Instant};

use crate::{
    attention::{gated_delta_net::GatedDeltaNetState, hybrid::HybridAttentionOptions, rope::RopeTable},
    backend::{
        Backend,
        rocm::{RocmContext, RocmGatedDeltaNetStorage, RocmKvCache, RocmPrefillExperts},
    },
    config::OrnithStandaloneModelConfig,
    moe::{
        UncachedMoeState,
        expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
    },
    runtime::{
        expert_pipeline::ExpertDecodePipeline,
        ornith::{self, OrnithGguf, OrnithRuntimeOptions},
    },
    weight::expert_source::GgufExpertSource,
};

pub fn run(backend: &RocmContext, model: OrnithStandaloneModelConfig) -> Result<(), Box<dyn std::error::Error>> {
    let weights = Arc::new(OrnithGguf::open(&model.weights)?);
    let cfg = weights.config().clone();
    ornith::ensure_supported(&cfg).map_err(|err| format!("Ornith runtime 尚不支持: {err:?}"))?;
    let tokenizer = weights.tokenizer()?;
    let prompt = if model.generation.prompt == "[gMASK]<|user|>你好<|assistant|>" { "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n" } else { model.generation.prompt.as_str() };
    let tokens = tokenizer.tokenize(prompt.as_bytes());
    if tokens.is_empty() || tokens.len() > model.generation.max_sequence_length {
        return Err(format!("Ornith prompt tokens={}，max_seq_len={}", tokens.len(), model.generation.max_sequence_length).into());
    }
    let layers = ornith::prepare_ornith_layers(backend, weights.as_ref()).map_err(|error| format!("准备 Ornith ROCm 层: {error:?}"))?;
    let output_head = ornith::prepare_ornith_output_head_quantized(backend, weights.as_ref(), model.lm_head_quantization).map_err(|error| format!("准备 Ornith output: {error:?}"))?;
    let rope = RopeTable::precompute(model.generation.max_sequence_length, cfg.rope_dim, cfg.rope_theta);
    let runtime =
        ornith::OrnithRuntime::new(backend, &cfg, &layers, 0, &rope, OrnithRuntimeOptions { attention: HybridAttentionOptions { precise_prefill: model.execution.precise_gqa_prefill }, expert_batch_size: model.execution.expert_batch_size });
    let mut cache = RocmKvCache::new(cfg.layer_count);
    let mut delta_state = GatedDeltaNetState::<RocmGatedDeltaNetStorage>::new(cfg.layer_count, cfg.gated_delta_net_spec()).map_err(|error| format!("创建 Ornith ROCm DeltaNet state: {error:?}"))?;
    let expert_source: Arc<dyn GgufExpertSource> = weights.clone();
    let mut prefill_experts = RocmPrefillExperts::gguf(expert_source);
    let hidden = backend.tensor_from_f32(weights.embedding_rows(&tokens)?, tokens.len(), cfg.hidden_size)?;
    let prefill_started = Instant::now();
    let hidden = runtime.at(&mut cache, &mut delta_state, 0).prefill(&mut prefill_experts, hidden).map_err(|error| format!("Ornith ROCm prefill: {error:?}"))?;
    eprintln!("[ornith-prefill] tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64());
    let mut hidden = backend.select_row(&hidden, tokens.len() - 1).map_err(|error| format!("选择 Ornith prefill 末行: {error:?}"))?;
    if model.generation.decode_steps == 0 {
        return Ok(());
    }

    let backend_state = UncachedMoeState::default();
    let mut expert_state = ExpertDecodePipeline::new(
        backend_state,
        ExpertPredictorConfig { first_layer: 0, layer_count: cfg.layer_count, expert_count: cfg.num_experts, routed_top_k: cfg.num_experts_per_tok, prefetch_count: 0, weights: ExpertPredictorWeights::default() },
    )?;
    let detokenizer = weights.detokenizer()?;
    crate::runtime::generation::run_generation(
        &mut hidden,
        crate::runtime::generation::GenerationLimits { prompt_tokens: tokens.len(), max_tokens: model.generation.decode_steps, max_sequence_length: model.generation.max_sequence_length, eos_tokens: &cfg.eos_token_ids },
        |hidden, _| Ok::<_, Box<dyn std::error::Error>>(ornith::ornith_token_output(backend, &cfg, &output_head, hidden).map_err(|error| format!("Ornith output: {error:?}"))?.token_id),
        |token, _| crate::runtime::generation::write_token(&detokenizer, token, true),
        |hidden, token, position, _| {
            let input = backend.tensor_from_f32(weights.embedding_rows(&[token])?, 1, cfg.hidden_size)?;
            *hidden = runtime.at(&mut cache, &mut delta_state, position).decode(weights.as_ref(), &mut expert_state, input).map_err(|error| format!("Ornith ROCm decode position={position}: {error:?}"))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    Ok(())
}
