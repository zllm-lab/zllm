//! Laguna × CUDA standalone 资源组合。

use crate::backend::cuda::CudaContext;

pub fn run(ctx: &CudaContext, model: crate::config::LagunaStandaloneModelConfig) -> Result<(), Box<dyn std::error::Error>> {
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    generate(ctx, model, &[], &cancelled, &mut |_, text| {
        print!("{text}");
        true
    })?;
    println!();
    Ok(())
}

/// Laguna CUDA 直接生成边界;standalone、console 与 Node 共用同一条 expert streaming 路径。
#[allow(clippy::too_many_arguments)]
pub fn generate(
    ctx: &CudaContext,
    model: crate::config::LagunaStandaloneModelConfig,
    stops: &[String],
    cancellation: &std::sync::atomic::AtomicBool,
    on_token: &mut dyn FnMut(u32, String) -> bool,
) -> Result<crate::runtime::session::GenerationSummary, Box<dyn std::error::Error>> {
    use crate::{
        backend::{
            Backend,
            cuda::{CudaKvCache, CudaPrefillExperts},
        },
        moe::expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
        runtime::{
            expert_pipeline::ExpertDecodePipeline,
            laguna::{self, LagunaGguf, LagunaRopeTables, LagunaRuntimeOptions},
            session::GenerationOutput,
        },
        weight::expert_source::GgufExpertSource,
    };
    use std::{sync::Arc, time::Instant};

    let prompt = model.generation.prompt;
    let max_seq_len = model.generation.max_sequence_length;
    let decode_steps = model.generation.decode_steps;
    let expert_cache_gib = Some(model.execution.expert_cache_gib as f64);
    let runtime_options = LagunaRuntimeOptions { expert_batch_size: model.execution.expert_batch_size };

    let weights = Arc::new(LagunaGguf::open(&model.weights)?);
    let cfg = weights.config().clone();
    laguna::ensure_supported(&cfg).map_err(|error| format!("Laguna 配置不受支持: {error}"))?;
    let tokenizer = weights.tokenizer()?;
    let tokens = tokenizer.tokenize(prompt.as_bytes());
    if tokens.is_empty() || tokens.len() > max_seq_len {
        return Err(format!("Laguna prompt tokens={}，max_seq_len={max_seq_len}", tokens.len()).into());
    }
    eprintln!("[laguna-cuda] model={} prompt_tokens={} decode_steps={decode_steps}", model.weights.display(), tokens.len());

    let prepare_started = Instant::now();
    let layers = laguna::prepare_laguna_layers(ctx, weights.as_ref()).map_err(|error| format!("准备 Laguna CUDA 层: {error:?}"))?;
    let output_head = laguna::prepare_laguna_output_quantized(ctx, weights.as_ref(), model.lm_head_quantization).map_err(|error| format!("准备 Laguna output: {error:?}"))?;
    ctx.synchronize()?;
    eprintln!("[laguna-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    // 滑窗层按窗口容量 ring 存储(512),full 层按序列上限;f16 KV。
    let mut cache = CudaKvCache::new_with_capacities(cfg.layer_count, max_seq_len, cfg.num_kv_heads * cfg.head_dim, cfg.kv_capacities(max_seq_len))?;
    let expert_cache_bytes = match expert_cache_gib {
        Some(gib) => (gib * 1024.0 * 1024.0 * 1024.0) as usize,
        None => {
            let (free, total) = ctx.device().mem_get_info()?;
            let reserve = (total / 16).max(512 * 1024 * 1024) + 128 * 1024 * 1024;
            let bytes = free.saturating_sub(reserve);
            if bytes == 0 {
                return Err(format!("CUDA 剩余显存 {:.2} GiB 不足以创建 expert cache", free as f64 / 1073741824.0).into());
            }
            bytes
        }
    };
    let rope = LagunaRopeTables::new(&cfg, max_seq_len)?;
    let runtime = laguna::LagunaRuntime::new(ctx, &cfg, &layers, &rope, runtime_options);
    let expert_source: Arc<dyn GgufExpertSource> = weights.clone();
    let mut prefill_experts = CudaPrefillExperts::gguf(expert_source, expert_cache_bytes);
    let prefill_started = Instant::now();
    // 滑窗 ring KV 约束:chunk ≤ 512(窗口大小),按块推进。
    // ZLLM_LAGUNA_PREFILL_CHUNK=1 为消元开关:全部 linear 走单行 gemv 参考路径。
    let chunk_size = std::env::var("ZLLM_LAGUNA_PREFILL_CHUNK").ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(512).min(cfg.sliding_window);
    let mut hidden = None;
    for (index, chunk) in tokens.chunks(chunk_size).enumerate() {
        let position = index * chunk_size;
        let embedding = weights.embedding_rows(chunk)?;
        let input = ctx.tensor_from_f32(&embedding, chunk.len(), cfg.hidden_size)?;
        let output = runtime.prefill(&mut cache, &mut prefill_experts, input, position).map_err(|error| format!("Laguna CUDA prefill position={position}: {error:?}"))?;
        hidden = Some(output);
    }
    let hidden = hidden.expect("tokens 非空已校验");
    eprintln!("[laguna-prefill] tokens={} wall={:.3}s kv_mib={:.1}", tokens.len(), prefill_started.elapsed().as_secs_f64(), cache.allocated_bytes() as f64 / 1048576.0);
    if decode_steps == 0 {
        return Ok(crate::runtime::session::GenerationSummary { finish_reason: "length".to_owned(), prompt_tokens: tokens.len(), completion_tokens: 0, cache: None, tool_calls: Vec::new() });
    }

    let backend_state = prefill_experts.into_decode_state();
    let mut expert_state = ExpertDecodePipeline::new(
        backend_state,
        ExpertPredictorConfig {
            first_layer: cfg.leading_dense_layer_count,
            layer_count: cfg.layer_count,
            expert_count: cfg.num_experts,
            routed_top_k: cfg.num_experts_per_tok,
            prefetch_count: model.execution.expert_prefetch_count.unwrap_or(0),
            weights: ExpertPredictorWeights::default(),
        },
    )?;
    let mut hidden = ctx.select_row(&hidden, tokens.len() - 1).map_err(|error| format!("选择 Laguna 最后 token: {error:?}"))?;
    let detokenizer = weights.detokenizer()?;
    let generation_started = Instant::now();
    let mut output = GenerationOutput::new(stops);
    for step in 0..decode_steps {
        if cancellation.load(std::sync::atomic::Ordering::Acquire) {
            output.cancel();
            break;
        }
        let started = Instant::now();
        let logits = laguna::laguna_token_output(ctx, &cfg, &output_head, &hidden).map_err(|error| format!("Laguna output: {error:?}"))?;
        let token = ctx.argmax(&logits.logits).map_err(|error| format!("Laguna argmax: {error:?}"))?;
        eprintln!("[laguna-token] step={step} id={token} wall={:.3}s", started.elapsed().as_secs_f64());
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
        hidden = runtime.decode(weights.as_ref(), &mut expert_state, &mut cache, input, position).map_err(|error| format!("Laguna CUDA decode position={position}: {error:?}"))?;
        eprintln!(
            "[laguna-decode-submit] step={step} submit_wall={:.3}s expert_cache_gib={:.2} loaded={}",
            decode_started.elapsed().as_secs_f64(),
            expert_state.backend_state_mut().resident_bytes() as f64 / 1073741824.0,
            expert_state.backend_state_mut().loaded
        );
    }
    output.finish(|chunk| on_token(0, chunk));
    let elapsed = generation_started.elapsed().as_secs_f64();
    eprintln!("[laguna-summary] prompt_tokens={} generated_tokens={} generation_wall={elapsed:.3}s tok_per_s={:.3}", tokens.len(), output.completion_tokens(), output.completion_tokens() as f64 / elapsed.max(f64::EPSILON),);
    Ok(output.summary(tokens.len()))
}
