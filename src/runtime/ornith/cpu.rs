//! Ornith × CPU reference 组合。

use crate::{
    attention::{gated_delta_net::GatedDeltaNetState, rope::RopeTable},
    backend::{
        Backend,
        cpu::{CpuContext, CpuGatedDeltaNetStorage, CpuKvCache, CpuPrefillExperts},
    },
    kernel::cpu::CpuTensor,
    moe::{
        UncachedMoeState,
        expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
    },
    runtime::{
        expert_pipeline::ExpertDecodePipeline,
        ornith::{self, OrnithGguf},
    },
    weight::expert_source::GgufExpertSource,
};
use std::{sync::Arc, time::Instant};

pub fn run(model_dir: &std::path::Path, prompt: &str, max_seq_len: usize, decode_steps: usize, runtime_options: crate::runtime::ornith::OrnithRuntimeOptions) -> Result<(), Box<dyn std::error::Error>> {
    let weights = Arc::new(OrnithGguf::open(&model_dir)?);
    let cfg = weights.config().clone();
    ornith::ensure_supported(&cfg).map_err(|err| format!("Ornith runtime 尚不支持: {err:?}"))?;
    let tokenizer = weights.tokenizer()?;
    let prompt = if prompt == "[gMASK]<|user|>你好<|assistant|>" { "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n" } else { prompt };
    let tokens = tokenizer.tokenize(prompt.as_bytes());
    if tokens.is_empty() || tokens.len() > max_seq_len {
        return Err(format!("Ornith prompt tokens={}，max_seq_len={}", tokens.len(), max_seq_len).into());
    }

    let backend = CpuContext;
    let prepare_started = Instant::now();
    let layers = ornith::prepare_ornith_layers(&backend, weights.as_ref()).map_err(|error| format!("准备 Ornith CPU 层: {error:?}"))?;
    let (final_norm, output_head) = ornith::prepare_ornith_output(&backend, weights.as_ref()).map_err(|error| format!("准备 Ornith output: {error:?}"))?;
    eprintln!("[ornith-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);
    let runtime = ornith::OrnithRuntime::new(&backend, &cfg, &layers, 0, &rope, runtime_options);
    let mut cache = CpuKvCache::new(cfg.layer_count);
    let mut delta_state = GatedDeltaNetState::<CpuGatedDeltaNetStorage>::new(cfg.layer_count, cfg.gated_delta_net_spec()).map_err(|error| format!("创建 Ornith DeltaNet state: {error:?}"))?;
    let expert_source: Arc<dyn GgufExpertSource> = weights.clone();
    let mut prefill_experts = CpuPrefillExperts::gguf(expert_source);
    let hidden = CpuTensor { data: weights.embedding_rows(&tokens)?, rows: tokens.len(), cols: cfg.hidden_size };
    let prefill_started = Instant::now();
    let hidden = runtime.at(&mut cache, &mut delta_state, 0).prefill(&mut prefill_experts, hidden).map_err(|error| format!("Ornith CPU prefill: {error:?}"))?;
    eprintln!("[ornith-prefill] tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64());
    let mut hidden = CpuTensor { data: hidden.row(tokens.len() - 1).to_vec(), rows: 1, cols: cfg.hidden_size };
    if decode_steps == 0 {
        return Ok(());
    }

    let mut expert_state = ExpertDecodePipeline::new(
        UncachedMoeState::default(),
        ExpertPredictorConfig { first_layer: 0, layer_count: cfg.layer_count, expert_count: cfg.num_experts, routed_top_k: cfg.num_experts_per_tok, prefetch_count: 0, weights: ExpertPredictorWeights::default() },
    )?;
    let detokenizer = weights.detokenizer()?;
    let stats = crate::runtime::generation::run_generation(
        &mut hidden,
        crate::runtime::generation::GenerationLimits { prompt_tokens: tokens.len(), max_tokens: decode_steps, max_sequence_length: max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |hidden, _| {
            let normalized = backend.gemma_rmsnorm_f32(hidden, &final_norm, cfg.rms_eps).map_err(|error| format!("Ornith output norm: {error:?}"))?;
            let logits = backend.linear(&normalized, &output_head).map_err(|error| format!("Ornith output head: {error:?}"))?;
            backend.argmax(&logits).map_err(|error| format!("Ornith argmax: {error:?}").into())
        },
        |token, _| crate::runtime::generation::write_token(&detokenizer, token, true),
        |hidden, token, position, _| {
            let input = CpuTensor { data: weights.embedding_rows(&[token])?, rows: 1, cols: cfg.hidden_size };
            *hidden = runtime.at(&mut cache, &mut delta_state, position).decode(weights.as_ref(), &mut expert_state, input).map_err(|error| format!("Ornith CPU decode position={position}: {error:?}"))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    eprintln!("[ornith-summary] prompt_tokens={} generated_tokens={} generation_wall={:.3}s tok_per_s={:.3}", tokens.len(), stats.generated_tokens, stats.elapsed.as_secs_f64(), stats.tokens_per_second(),);
    Ok(())
}
