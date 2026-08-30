//! Qwen3.6 × CUDA standalone 资源组合。

use crate::backend::cuda::CudaContext;

/// Qwen3.6-27B 纯文本入口:safetensors + MLX affine 4bit。
///
/// 用法: `zllm-rt-cuda --model qwen36 MODEL_DIR [--prompt TEXT] [--max-seq-len N] [--decode-steps N]`
///
/// MlxAffine 矩阵保持 packed 布局，decode 走 CUDA 原生即时反量化 kernel。
pub fn run(ctx: &CudaContext, model: crate::config::Qwen36StandaloneModelConfig) -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        attention::hybrid::HybridAttentionOptions,
        attention::{
            gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetState},
            rope::RopeTable,
        },
        backend::cuda::{CudaGatedDeltaNetStorage, CudaKvCache},
        runtime::qwen36::{Qwen36Config, Qwen36Runtime, prepare_qwen36_layers, prepare_qwen36_output_head_quantized, qwen36_token_output},
        weight::{format::mlx_affine::MlxAffineSource, model::qwen36::Qwen36Weights},
    };
    use std::time::Instant;

    let prompt = model.generation.prompt;
    let max_seq_len = model.generation.max_sequence_length;
    let decode_steps = model.generation.decode_steps;
    let cfg = Qwen36Config::standard_27b();
    crate::runtime::validate_max_sequence_length("Qwen3.6", max_seq_len, cfg.max_position_embeddings)?;

    let model_dir = model.weights_directory;
    let source = MlxAffineSource::open(&model_dir).map_err(|error| format!("打开 Qwen3.6 MLX affine safetensors: {error}"))?;
    let weights = Qwen36Weights::new(source, cfg.clone()).map_err(|error| format!("加载 Qwen3.6 权重: {error}"))?;
    let (tokenizer, detokenizer) = crate::tokenizer::load_bpe_directory(&model_dir)?;
    let prompt_text = crate::runtime::qwen36::chat_prompt(&prompt);
    let token_ids = tokenizer.tokenize(prompt_text.as_bytes());
    if token_ids.is_empty() {
        return Err("Qwen3.6 prompt tokenize 为空".into());
    }
    if token_ids.len() > max_seq_len {
        return Err(format!("prompt tokens={} 超过 max_seq_len={max_seq_len}", token_ids.len()).into());
    }
    eprintln!("[qwen36-cuda] model={} prompt_tokens={} max_seq_len={max_seq_len} decode_steps={decode_steps} MLX-affine 4bit packed", model_dir.display(), token_ids.len());

    let prepare_started = Instant::now();
    let layers = prepare_qwen36_layers(ctx, &weights).map_err(|error| format!("准备 Qwen3.6 CUDA 层: {error:?}"))?;
    let output_head = prepare_qwen36_output_head_quantized(ctx, &weights, model.lm_head_quantization).map_err(|error| format!("准备 Qwen3.6 output head: {error:?}"))?;
    ctx.synchronize()?;
    eprintln!("[qwen36-cuda] resident_layers={} preload={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    let kv_columns = cfg.num_kv_heads * cfg.head_dim;
    let mut cache = if model.execution.kv_cache_format == crate::config::KvCacheFormat::F16 { CudaKvCache::new(cfg.num_layers, max_seq_len, kv_columns) } else { CudaKvCache::new_q8g64(cfg.num_layers, max_seq_len, kv_columns)? };
    let mut delta_state = GatedDeltaNetState::<CudaGatedDeltaNetStorage>::with_head_layout(cfg.num_layers, cfg.gated_delta_net_spec(), GatedDeltaNetHeadLayout::Grouped).map_err(|error| format!("Qwen3.6 CUDA DeltaNet state: {error:?}"))?;

    // 纯文本三轴位置一致，预计算完整标准 RoPE 供 prefill/decode 共同复用。
    let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);
    let runtime = Qwen36Runtime::new(ctx, &cfg, &layers, &rope, HybridAttentionOptions { precise_prefill: model.execution.precise_gqa_prefill });

    let embedding = weights.embedding_rows_f32(&token_ids)?;
    let hidden = ctx.tensor_from_f32(&embedding, token_ids.len(), cfg.hidden_size).map_err(|error| format!("上传 Qwen3.6 embedding: {error}"))?;

    let prefill_started = Instant::now();
    let hidden = runtime.at(&mut cache, &mut delta_state, 0).prefill(hidden).map_err(|error| format!("Qwen3.6 CUDA prefill: {error:?}"))?;
    ctx.synchronize()?;
    eprintln!("[qwen36-cuda] prefill wall={:.3}s kv_cache={:.2} MiB delta_state={:.2} MiB", prefill_started.elapsed().as_secs_f64(), cache.allocated_bytes() as f64 / 1048576.0, delta_state.allocated_bytes() as f64 / 1048576.0);

    let mut output = qwen36_token_output(ctx, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3.6 CUDA first token: {error:?}"))?;
    if decode_steps == 0 {
        return Ok(());
    }

    let mut generated = Vec::with_capacity(decode_steps);
    let stats = crate::runtime::generation::run_generation(
        &mut output,
        crate::runtime::generation::GenerationLimits { prompt_tokens: token_ids.len(), max_tokens: decode_steps, max_sequence_length: max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |output, _| Ok::<_, Box<dyn std::error::Error>>(output.token_id),
        |token, _| {
            generated.push(token);
            Ok::<_, Box<dyn std::error::Error>>(())
        },
        |output, token, position, _| {
            let embedding = weights.embedding_rows_f32(&[token])?;
            let hidden = ctx.tensor_from_f32(&embedding, 1, cfg.hidden_size).map_err(|error| format!("上传 Qwen3.6 decode embedding: {error}"))?;
            let hidden = runtime.at(&mut cache, &mut delta_state, position).decode(hidden).map_err(|error| format!("Qwen3.6 CUDA decode position={position}: {error:?}"))?;
            *output = qwen36_token_output(ctx, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3.6 CUDA output position={position}: {error:?}"))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    let answer = detokenizer.decode_bytes(&generated, true)?;
    println!("{}", String::from_utf8_lossy(&answer));
    eprintln!("[qwen36-cuda] generated={} decode_rounds={} wall={:.3}s throughput={:.3} tok/s", stats.generated_tokens, stats.decode_rounds, stats.elapsed.as_secs_f64(), stats.decode_tokens_per_second());
    Ok(())
}
