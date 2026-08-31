//! Ornith × Metal standalone 组合。

use crate::{
    attention::{gated_delta_net::GatedDeltaNetState, rope::RopeTable},
    backend::{
        Backend, BackendError, BackendResources,
        metal::{MetalContext, MetalGatedDeltaNetStorage, MetalKvCache, MetalPrefillExperts},
    },
    config::{KvCacheFormat, MetalBackendConfig},
    weight::expert_source::GgufExpertSource,
};
use std::{path::Path, sync::Arc};

use crate::runtime::ornith::{self, OrnithGguf};

pub fn run(model_path: &Path, prompt: &str, max_seq_len: usize, decode_steps: usize, execution: crate::config::OrnithStandaloneExecutionConfig, backend: &MetalBackendConfig) -> Result<(), Box<dyn std::error::Error>> {
    let weights = Arc::new(OrnithGguf::open(model_path)?);
    let cfg = weights.config().clone();
    ornith::ensure_supported(&cfg).map_err(|err| format!("Ornith runtime 尚不支持: {err:?}"))?;
    let expert_prefetch_count = execution.expert_prefetch_count.unwrap_or(0).min(cfg.num_experts);
    let tokenizer = weights.tokenizer()?;
    let prompt = if prompt == "[gMASK]<|user|>你好<|assistant|>" { "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n" } else { prompt };
    let tokens = tokenizer.tokenize(prompt.as_bytes());
    if tokens.is_empty() || tokens.len() > max_seq_len {
        return Err(format!("Ornith prompt tokens={}，max_seq_len={max_seq_len}", tokens.len()).into());
    }

    println!("backend: Metal, model: Ornith, prompt: {} tokens", tokens.len());
    let ctx_owner = MetalContext::new_default_with_replay(backend.replay).map_err(|error| format!("MetalContext 初始化失败: {error}"))?;
    let ctx = &ctx_owner;
    let prepare_started = std::time::Instant::now();
    let layers = ornith::prepare_ornith_layers(ctx, weights.as_ref()).map_err(|error| format!("准备 Ornith Metal 层: {error:?}"))?;
    eprintln!("[ornith-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    let attention = crate::attention::AttentionSpec::Gqa(cfg.full_attention_spec());
    let cache_spec = crate::kv_cache::KvCacheSpec::from_attention(&attention).map_err(|error| format!("Ornith KV cache spec: {error}"))?;
    let cache_layers = ornith::kv_cache_layer_map(&cfg).map_err(|error| format!("Ornith KV cache layer map: {error}"))?;
    let kv_f16 = execution.kv_cache_format == KvCacheFormat::F16;
    let mut cache = if kv_f16 { MetalKvCache::new_f16_mapped(ctx, cache_spec.clone(), cache_layers, max_seq_len) } else { MetalKvCache::new_mapped(ctx, cache_spec.clone(), cache_layers, max_seq_len) }
        .map_err(|error| format!("Ornith KV cache: {error}"))?;
    eprintln!("[ornith-kv-cache] format={} logical_layers={} slots={} allocated_mib={:.2}", if kv_f16 { "F16" } else { "Q8G64" }, cache.layer_count(), cache.cache_slot_count(), cache.allocated_bytes() as f64 / (1024.0 * 1024.0),);
    let mut delta_state = GatedDeltaNetState::<MetalGatedDeltaNetStorage>::new(cfg.layer_count, cfg.gated_delta_net_spec()).map_err(|error| format!("Ornith DeltaNet state: {error:?}"))?;
    let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);
    let expert_source: Arc<dyn GgufExpertSource> = weights.clone();
    let mut prefill_experts =
        if decode_steps == 0 { MetalPrefillExperts::gguf(expert_source) } else { MetalPrefillExperts::gguf_resident(expert_source, crate::backend::metal::MetalMoeDecodeState::with_gguf_cache_gib(execution.expert_cache_gib)?) };
    let embedding = weights.embedding_rows(&tokens)?;
    // hidden/residual 是跨层累加状态，必须保持 F32；QKV/KV 仍由各算子按短生命周期和 cache 格式管理。
    let hidden = ctx.tensor_from_f32_preserve(&embedding, tokens.len(), cfg.hidden_size).map_err(|error| format!("上传 Ornith embedding: {error}"))?;
    let profile_prefill_gpu = backend.profile_prefill_gpu;
    if profile_prefill_gpu {
        ctx.reset_gpu_stats();
    }
    let prefill_started = std::time::Instant::now();
    let trace_layer_dir = backend.trace_layer_directory.clone();
    if let Some(directory) = &trace_layer_dir {
        std::fs::create_dir_all(directory).map_err(|error| format!("创建 Ornith layer trace 目录失败: {error}"))?;
        let begin = embedding.len() - cfg.hidden_size;
        let bytes = embedding[begin..].iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        std::fs::write(directory.join("input.f32"), bytes).map_err(|error| format!("写入 Ornith input trace 失败: {error}"))?;
    }
    let trace_layers = trace_layer_dir.is_some() || backend.trace_layers;
    let runtime = ornith::OrnithRuntime::new(
        ctx,
        &cfg,
        &layers,
        0,
        &rope,
        crate::runtime::ornith::OrnithRuntimeOptions { attention: crate::attention::hybrid::HybridAttentionOptions { precise_prefill: execution.precise_gqa_prefill }, expert_batch_size: execution.expert_batch_size },
    );
    let hidden = runtime
        .at(&mut cache, &mut delta_state, 0)
        .prefill_observed(&mut prefill_experts, hidden, |layer, tensor| {
            if !trace_layers {
                return Ok(());
            }
            let row = ctx.token_rows(tensor).saturating_sub(1);
            let values = ctx.tensor_row_to_f32(tensor, row).map_err(|msg| BackendError::Compute { msg })?;
            let sum = values.iter().map(|&value| value as f64).sum::<f64>();
            let l2 = values.iter().map(|&value| (value as f64) * (value as f64)).sum::<f64>().sqrt();
            let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let first = values.iter().take(8).map(|value| format!("{value:.9}")).collect::<Vec<_>>().join(",");
            eprintln!("[zllm-layer] layer={layer} rows={} cols={} sum={sum:.9} l2={l2:.9} min={minimum:.9} max={maximum:.9} first={first}", ctx.token_rows(tensor), ctx.token_cols(tensor));
            if let Some(directory) = &trace_layer_dir {
                let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
                std::fs::write(directory.join(format!("layer-{layer:02}.f32")), bytes).map_err(|error| BackendError::Compute { msg: format!("写入 Ornith L{layer} trace 失败: {error}") })?;
            }
            Ok(())
        })
        .map_err(|error| format!("Ornith Metal prefill: {error:?}"))?;
    eprintln!("[ornith-prefill] tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64());
    if profile_prefill_gpu {
        let gpu = ctx.gpu_stats();
        eprintln!("[ornith-prefill-gpu] gpu={:.3}s commands={} submit_wait={:.3}s gaps={:.3}s tail={:.3}s", gpu.seconds, gpu.command_buffers, gpu.submit_wait_seconds, gpu.inter_command_gap_seconds, gpu.completion_tail_seconds,);
        for operator in ctx.gpu_profile().into_iter().take(20) {
            eprintln!("  ornith prefill gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape,);
        }
    }
    if decode_steps == 0 {
        return Ok(());
    }

    let (final_norm, output_head) = ornith::prepare_ornith_output(ctx, weights.as_ref()).map_err(|error| format!("准备 Ornith output: {error:?}"))?;
    let mtp_enabled = execution.mtp;
    let mut mtp_session = if weights.has_mtp() && mtp_enabled && decode_steps > 1 {
        let started = std::time::Instant::now();
        let mtp_weights = ornith::prepare_ornith_mtp(ctx, weights.as_ref()).map_err(|error| format!("准备 Ornith MTP: {error:?}"))?;
        let mut mtp_cache = if kv_f16 { MetalKvCache::new_f16(ctx, cache_spec.clone(), 1, max_seq_len) } else { MetalKvCache::new(ctx, cache_spec.clone(), 1, max_seq_len) }.map_err(|error| format!("Ornith MTP KV cache: {error}"))?;
        let mtp_source: Arc<dyn GgufExpertSource> = weights.clone();
        let mut mtp_prefill_experts = MetalPrefillExperts::gguf_resident(mtp_source, crate::backend::metal::MetalMoeDecodeState::new().expect("Metal MTP decode state 初始化不分配外部资源"));
        if tokens.len() > 1 {
            let normalized_hidden = ctx.gemma_rmsnorm(&hidden, &final_norm, cfg.rms_eps).map_err(|error| format!("Ornith MTP target hidden norm: {error:?}"))?;
            let target_rows = (0..tokens.len() - 1).map(|row| row as u32).collect::<Vec<_>>();
            let target_hidden = ctx.select_rows(&normalized_hidden, &target_rows).map_err(|error| format!("Ornith MTP target hidden shift: {error:?}"))?;
            let shifted_embedding = weights.embedding_rows(&tokens[1..])?;
            let shifted_embedding = ctx.tensor_from_f32_preserve(&shifted_embedding, tokens.len() - 1, cfg.hidden_size).map_err(|error| format!("上传 Ornith MTP shifted embedding: {error}"))?;
            runtime.mtp_prefill(&mtp_weights, &mut mtp_prefill_experts, &mut mtp_cache, &shifted_embedding, &target_hidden, 0).map_err(|error| format!("Ornith MTP prefill: {error:?}"))?;
        }
        let mtp_backend_state = mtp_prefill_experts.take_gguf_decode_state().unwrap_or_else(|| crate::backend::metal::MetalMoeDecodeState::new().expect("Metal MTP decode state 初始化不分配外部资源"));
        let mtp_expert_state = crate::runtime::expert_pipeline::ExpertDecodePipeline::new(
            mtp_backend_state,
            crate::moe::expert_predictor::ExpertPredictorConfig {
                first_layer: cfg.layer_count,
                layer_count: cfg.mtp_layer_count,
                expert_count: cfg.num_experts,
                routed_top_k: cfg.num_experts_per_tok,
                prefetch_count: 0,
                weights: crate::moe::expert_predictor::ExpertPredictorWeights::default(),
            },
        )?;
        ctx.synchronize();
        eprintln!("[ornith-mtp-prepare] prompt_cache={} wall={:.3}s", tokens.len().saturating_sub(1), started.elapsed().as_secs_f64(),);
        Some((mtp_weights, mtp_cache, mtp_expert_state))
    } else {
        None
    };
    // 在 11 GiB expert 常驻前收缩 prefill 输出，避免整段 hidden 跨资源峰值存活。
    let mut hidden = ctx.select_row(&hidden, tokens.len() - 1).map_err(|error| format!("Ornith 选择最后 token: {error:?}"))?;
    let mut backend_state = prefill_experts.take_gguf_decode_state().unwrap_or_else(|| crate::backend::metal::MetalMoeDecodeState::new().expect("Metal decode state 初始化不分配外部资源"));
    if !execution.lazy_experts {
        let started = std::time::Instant::now();
        let (resident_experts, resident_bytes) = backend_state.preload_gguf_experts(ctx, weights.as_ref(), cfg.layer_count, cfg.num_experts).map_err(|error| format!("Ornith expert 全量常驻: {error:?}"))?;
        eprintln!("[ornith-expert-preload] experts={resident_experts} resident_gib={:.2} wall={:.3}s", resident_bytes as f64 / (1024.0 * 1024.0 * 1024.0), started.elapsed().as_secs_f64(),);
    }
    let mut expert_state = crate::runtime::expert_pipeline::ExpertDecodePipeline::new(
        backend_state,
        crate::moe::expert_predictor::ExpertPredictorConfig {
            first_layer: 0,
            layer_count: cfg.layer_count,
            expert_count: cfg.num_experts,
            routed_top_k: cfg.num_experts_per_tok,
            prefetch_count: expert_prefetch_count,
            weights: crate::moe::expert_predictor::ExpertPredictorWeights::default(),
        },
    )?;
    let detokenizer = weights.detokenizer()?;
    if let Some((mtp_weights, mut mtp_cache, mut mtp_expert_state)) = mtp_session.take() {
        let decode_started = std::time::Instant::now();
        let mut generated = 0usize;
        let mut drafted = 0usize;
        let mut accepted = 0usize;
        let mut target_rounds = 0usize;
        let mut target_position = tokens.len();
        let mut mtp_position = tokens.len().saturating_sub(1);
        while generated < decode_steps {
            let normalized = ctx.gemma_rmsnorm_f32(&hidden, &final_norm, cfg.rms_eps).map_err(|error| format!("Ornith output norm: {error:?}"))?;
            let logits = ctx.linear(&normalized, &output_head).map_err(|error| format!("Ornith output head: {error:?}"))?;
            let token = ctx.argmax(&logits).map_err(|error| format!("Ornith argmax: {error:?}"))?;
            if backend.trace_tokens {
                eprintln!("[ornith-decode] step={generated} token={token}");
            }
            if cfg.eos_token_ids.contains(&token) {
                break;
            }
            crate::runtime::generation::write_token(&detokenizer, token, false)?;
            generated += 1;
            use std::io::Write;
            std::io::stdout().flush().ok();
            if generated >= decode_steps || target_position >= max_seq_len {
                break;
            }

            let token_embedding = weights.embedding_rows(&[token])?;
            let token_embedding = ctx.tensor_from_f32_preserve(&token_embedding, 1, cfg.hidden_size).map_err(|error| format!("上传 Ornith MTP decode embedding: {error}"))?;
            let (draft, draft_hidden) = runtime
                .mtp_decode_then(&mtp_weights, weights.as_ref(), &mut mtp_expert_state, &mut mtp_cache, &token_embedding, &normalized, mtp_position, |draft_hidden| {
                    let draft_logits = ctx.linear(draft_hidden, &output_head)?;
                    Ok((ctx.argmax(&draft_logits)?, draft_hidden.clone()))
                })
                .map_err(|error| format!("Ornith MTP decode/head position={mtp_position}: {error:?}"))?;
            mtp_position += 1;
            drafted += 1;

            if target_position + 2 > max_seq_len {
                hidden = runtime.at(&mut cache, &mut delta_state, target_position).decode(weights.as_ref(), &mut expert_state, token_embedding).map_err(|error| format!("Ornith Metal decode position={target_position}: {error:?}"))?;
                target_position += 1;
                target_rounds += 1;
                continue;
            }

            let first_hidden = runtime
                .at(&mut cache, &mut delta_state, target_position)
                .decode(weights.as_ref(), &mut expert_state, token_embedding.clone())
                .map_err(|error| format!("Ornith MTP target verify first position={target_position}: {error:?}"))?;
            target_rounds += 1;
            let first_normalized = ctx.gemma_rmsnorm(&first_hidden, &final_norm, cfg.rms_eps).map_err(|error| format!("Ornith MTP verify output norm: {error:?}"))?;
            let first_logits = ctx.linear(&first_normalized, &output_head).map_err(|error| format!("Ornith MTP verify output head: {error:?}"))?;
            let target_token = ctx.argmax(&first_logits).map_err(|error| format!("Ornith MTP verify argmax: {error:?}"))?;
            if backend.trace_tokens {
                eprintln!("[ornith-mtp-verify] next_step={} draft={draft} target={target_token} accepted={}", generated, target_token == draft,);
            }
            if target_token == draft {
                let draft_embedding = weights.embedding_rows(&[draft])?;
                let draft_embedding = ctx.tensor_from_f32_preserve(&draft_embedding, 1, cfg.hidden_size).map_err(|error| format!("上传 Ornith MTP verify draft embedding: {error}"))?;
                let second_hidden = runtime
                    .at(&mut cache, &mut delta_state, target_position + 1)
                    .decode(weights.as_ref(), &mut expert_state, draft_embedding)
                    .map_err(|error| format!("Ornith MTP target verify second position={}: {error:?}", target_position + 1))?;
                target_rounds += 1;
                hidden = second_hidden;
                target_position += 2;
                accepted += 1;
                if cfg.eos_token_ids.contains(&draft) {
                    break;
                }
                crate::runtime::generation::write_token(&detokenizer, draft, false)?;
                generated += 1;
                std::io::stdout().flush().ok();
                if generated >= decode_steps || target_position >= max_seq_len {
                    break;
                }
                let draft_embedding = weights.embedding_rows(&[draft])?;
                let draft_embedding = ctx.tensor_from_f32_preserve(&draft_embedding, 1, cfg.hidden_size).map_err(|error| format!("上传 Ornith MTP accepted embedding: {error}"))?;
                runtime.mtp_decode(&mtp_weights, weights.as_ref(), &mut mtp_expert_state, &mut mtp_cache, &draft_embedding, &draft_hidden, mtp_position).map_err(|error| format!("Ornith MTP advance position={mtp_position}: {error:?}"))?;
                mtp_position += 1;
            } else {
                hidden = first_hidden;
                target_position += 1;
            }
        }
        println!();
        let seconds = decode_started.elapsed().as_secs_f64();
        eprintln!(
            "[ornith-mtp-decode] tokens={generated} drafts={drafted} accepted={accepted} acceptance={:.1}% target_rounds={target_rounds} wall={seconds:.3}s throughput={:.3} tok/s",
            accepted as f64 * 100.0 / drafted.max(1) as f64,
            generated as f64 / seconds.max(f64::EPSILON),
        );
        return Ok(());
    }
    let mut generation_state = (hidden, None::<std::time::Instant>);
    let stats = crate::runtime::generation::run_generation(
        &mut generation_state,
        crate::runtime::generation::GenerationLimits { prompt_tokens: tokens.len(), max_tokens: decode_steps, max_sequence_length: max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |state, step| {
            let profile_decode = backend.profile_decode;
            if profile_decode {
                ctx.reset_gpu_stats();
            }
            state.1 = Some(std::time::Instant::now());
            let normalized = ctx.gemma_rmsnorm_f32(&state.0, &final_norm, cfg.rms_eps).map_err(|error| format!("Ornith output norm: {error:?}"))?;
            let logits = ctx.linear(&normalized, &output_head).map_err(|error| format!("Ornith output head: {error:?}"))?;
            if backend.trace_logits {
                let values = ctx.tensor_to_f32(&logits);
                let mut top = values.iter().copied().enumerate().collect::<Vec<_>>();
                top.select_nth_unstable_by(5, |left, right| right.1.total_cmp(&left.1));
                top[..5].sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
                eprintln!("[ornith-logits] token271={:.6} eos={:.6} delta={:.6} top={:?}", values[271], values[248046], values[271] - values[248046], &top[..5]);
            }
            let token = ctx.argmax(&logits).map_err(|error| format!("Ornith argmax: {error:?}"))?;
            if backend.trace_tokens {
                eprintln!("[ornith-decode] step={step} token={token}");
            }
            Ok::<_, Box<dyn std::error::Error>>(token)
        },
        |token, _| crate::runtime::generation::write_token(&detokenizer, token, false),
        |state, token, position, step| {
            let embedding = weights.embedding_rows(&[token])?;
            let input = ctx.tensor_from_f32_preserve(&embedding, 1, cfg.hidden_size).map_err(|error| format!("上传 Ornith decode embedding: {error}"))?;
            state.0 = runtime.at(&mut cache, &mut delta_state, position).decode(weights.as_ref(), &mut expert_state, input).map_err(|error| format!("Ornith Metal decode position={position}: {error:?}"))?;
            if backend.profile_decode {
                let token_started = state.1.take().expect("select_token 已设置计时");
                let expert_stats = expert_state.backend_state_mut().take_stats();
                let gpu = ctx.gpu_stats();
                eprintln!(
                    "[ornith-decode-profile] step={step} wall={:.3}s gpu={:.3}s commands={} hits={} misses={} read={:.3}s upload={:.3}s resident={} resident_gib={:.2}",
                    token_started.elapsed().as_secs_f64(),
                    gpu.seconds,
                    gpu.command_buffers,
                    expert_stats.cache_hits,
                    expert_stats.cache_misses,
                    expert_stats.sync_expert_read_seconds,
                    expert_stats.expert_upload_seconds,
                    expert_stats.resident_experts,
                    expert_stats.resident_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                if matches!(step, 0 | 15 | 31 | 47 | 62) {
                    for operator in ctx.gpu_profile().into_iter().take(12) {
                        eprintln!("  ornith gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape,);
                    }
                }
            }
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    eprintln!("[ornith-decode-summary] tokens={} wall={:.3}s throughput={:.3} tok/s", stats.generated_tokens, stats.elapsed.as_secs_f64(), stats.tokens_per_second(),);
    Ok(())
}
