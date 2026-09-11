//! Ornith × CUDA standalone 资源组合。

use crate::backend::cuda::CudaContext;

pub fn run(ctx: &CudaContext, model: crate::config::OrnithStandaloneModelConfig) -> Result<(), Box<dyn std::error::Error>> {
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    generate(ctx, model, &[], &cancelled, &mut |_, text| {
        print!("{text}");
        true
    })?;
    println!();
    Ok(())
}

/// Ornith CUDA 直接生成边界；standalone、console 与 Node 共用同一条 expert streaming 路径。
pub fn generate(
    ctx: &CudaContext,
    model: crate::config::OrnithStandaloneModelConfig,
    stops: &[String],
    cancellation: &std::sync::atomic::AtomicBool,
    on_token: &mut dyn FnMut(u32, String) -> bool,
) -> Result<crate::runtime::session::GenerationSummary, Box<dyn std::error::Error>> {
    use crate::{
        attention::hybrid::HybridAttentionOptions,
        attention::{gated_delta_net::GatedDeltaNetState, rope::RopeTable},
        backend::Backend,
        backend::cuda::{CudaGatedDeltaNetStorage, CudaKvCache, CudaPrefillExperts},
        moe::expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
        runtime::{
            expert_pipeline::ExpertDecodePipeline,
            ornith,
            ornith::{OrnithConfig, OrnithGguf, OrnithRuntimeOptions},
            session::GenerationOutput,
        },
        weight::expert_source::GgufExpertSource,
    };
    use std::{sync::Arc, time::Instant};

    let prompt = model.generation.prompt;
    let max_seq_len = model.generation.max_sequence_length;
    let decode_steps = model.generation.decode_steps;
    let expert_cache_gib = Some(model.execution.expert_cache_gib as f64);
    let expert_prefetch_count = model.execution.expert_prefetch_count.unwrap_or(0).min(OrnithConfig::standard().num_experts);
    let runtime_options = OrnithRuntimeOptions { attention: HybridAttentionOptions { precise_prefill: model.execution.precise_gqa_prefill }, expert_batch_size: model.execution.expert_batch_size };

    let weights = Arc::new(OrnithGguf::open(&model.weights)?);
    let cfg = weights.config().clone();
    ornith::ensure_supported(&cfg).map_err(|error| format!("Ornith 配置不受支持: {error:?}"))?;
    let tokenizer = weights.tokenizer()?;
    let tokens = tokenizer.tokenize(prompt.as_bytes());
    if tokens.is_empty() || tokens.len() > max_seq_len {
        return Err(format!("Ornith prompt tokens={}，max_seq_len={max_seq_len}", tokens.len()).into());
    }
    eprintln!("[ornith-cuda] model={} prompt_tokens={} decode_steps={decode_steps}", model.weights.display(), tokens.len());

    let prepare_started = Instant::now();
    let layers = ornith::prepare_ornith_layers(ctx, weights.as_ref()).map_err(|error| format!("准备 Ornith CUDA 层: {error:?}"))?;
    let (final_norm, output_head) = ornith::prepare_ornith_output_quantized(ctx, weights.as_ref(), model.lm_head_quantization).map_err(|error| format!("准备 Ornith output: {error:?}"))?;
    ctx.synchronize()?;
    eprintln!("[ornith-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    let mut cache = if model.execution.kv_cache_format == crate::config::KvCacheFormat::F16 {
        CudaKvCache::new(cfg.layer_count, max_seq_len, cfg.num_kv_heads * cfg.head_dim)
    } else {
        CudaKvCache::new_q8g64(cfg.layer_count, max_seq_len, cfg.num_kv_heads * cfg.head_dim)?
    };
    let mut delta_state = GatedDeltaNetState::<CudaGatedDeltaNetStorage>::new(cfg.layer_count, cfg.gated_delta_net_spec()).map_err(|error| format!("创建 Ornith CUDA DeltaNet state: {error:?}"))?;
    let expert_cache_bytes = match expert_cache_gib {
        Some(gib) => (gib * 1024.0 * 1024.0 * 1024.0) as usize,
        None => {
            let (free, total) = ctx.device().mem_get_info()?;
            // cudarc 延迟回收可能短暂保留上轮 expert，额外余量用于吸收分配尖峰。
            let reserve = (total / 16).max(512 * 1024 * 1024) + 128 * 1024 * 1024;
            let bytes = free.saturating_sub(reserve);
            if bytes == 0 {
                return Err(format!("CUDA 剩余显存 {:.2} GiB 不足以创建 expert cache", free as f64 / 1073741824.0).into());
            }
            eprintln!("[ornith-resource] expert_cache=auto cache_gib={:.2} free_gib={:.2} reserve_gib={:.2}", bytes as f64 / 1073741824.0, free as f64 / 1073741824.0, reserve as f64 / 1073741824.0,);
            bytes
        }
    };
    let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);
    let runtime = ornith::OrnithRuntime::new(ctx, &cfg, &layers, 0, &rope, runtime_options);
    let expert_source: Arc<dyn GgufExpertSource> = weights.clone();
    let mut prefill_experts = CudaPrefillExperts::gguf(expert_source, expert_cache_bytes);
    let embedding = weights.embedding_rows(&tokens)?;
    let hidden = ctx.tensor_from_f32(&embedding, tokens.len(), cfg.hidden_size)?;
    let prefill_started = Instant::now();
    let hidden = runtime.at(&mut cache, &mut delta_state, 0).prefill(&mut prefill_experts, hidden).map_err(|error| format!("Ornith CUDA prefill: {error:?}"))?;
    ctx.synchronize()?;
    eprintln!("[ornith-prefill] tokens={} wall={:.3}s kv_mib={:.1} delta_mib={:.1}", tokens.len(), prefill_started.elapsed().as_secs_f64(), cache.allocated_bytes() as f64 / 1048576.0, delta_state.allocated_bytes() as f64 / 1048576.0,);
    if decode_steps == 0 {
        return Ok(crate::runtime::session::GenerationSummary { finish_reason: "length".to_owned(), prompt_tokens: tokens.len(), completion_tokens: 0, cache: None, tool_calls: Vec::new() });
    }

    let backend_state = prefill_experts.into_decode_state();
    let mut expert_state = ExpertDecodePipeline::new(
        backend_state,
        ExpertPredictorConfig { first_layer: 0, layer_count: cfg.layer_count, expert_count: cfg.num_experts, routed_top_k: cfg.num_experts_per_tok, prefetch_count: expert_prefetch_count, weights: ExpertPredictorWeights::default() },
    )?;
    let mut hidden = ctx.select_row(&hidden, tokens.len() - 1).map_err(|error| format!("选择 Ornith 最后 token: {error:?}"))?;
    let detokenizer = weights.detokenizer()?;
    let generation_started = Instant::now();
    let mut output = GenerationOutput::new(stops);
    for step in 0..decode_steps {
        if cancellation.load(std::sync::atomic::Ordering::Acquire) {
            output.cancel();
            break;
        }
        let started = Instant::now();
        let normalized = ctx.gemma_rmsnorm_f32(&hidden, &final_norm, cfg.rms_eps).map_err(|error| format!("Ornith output norm: {error:?}"))?;
        let logits = ctx.linear(&normalized, &output_head).map_err(|error| format!("Ornith LM head: {error:?}"))?;
        let token = ctx.argmax(&logits).map_err(|error| format!("Ornith argmax: {error:?}"))?;
        eprintln!("[ornith-token] step={step} id={token} wall={:.3}s", started.elapsed().as_secs_f64());
        if cfg.eos_token_ids.contains(&token) {
            output.stop();
            break;
        }
        let bytes = crate::runtime::tool::decode_output_token(&detokenizer, token)?;
        if !output.push(&bytes, |chunk| on_token(token, chunk)) {
            break;
        }
        if output.completion_tokens() == decode_steps {
            break;
        }
        let position = tokens.len() + step;
        if position >= max_seq_len {
            break;
        }
        let input = ctx.tensor_from_f32(&weights.embedding_rows(&[token])?, 1, cfg.hidden_size)?;
        let decode_started = Instant::now();
        hidden = runtime.at(&mut cache, &mut delta_state, position).decode(weights.as_ref(), &mut expert_state, input).map_err(|error| format!("Ornith CUDA decode position={position}: {error:?}"))?;
        eprintln!("[ornith-decode-submit] step={step} submit_wall={:.3}s expert_cache_gib={:.2}", decode_started.elapsed().as_secs_f64(), expert_state.backend_state_mut().resident_bytes() as f64 / 1073741824.0,);
    }
    output.finish(|chunk| on_token(0, chunk));
    let elapsed = generation_started.elapsed().as_secs_f64();
    eprintln!("[ornith-summary] prompt_tokens={} generated_tokens={} generation_wall={elapsed:.3}s tok_per_s={:.3}", tokens.len(), output.completion_tokens(), output.completion_tokens() as f64 / elapsed.max(f64::EPSILON),);
    Ok(output.summary(tokens.len()))
}
