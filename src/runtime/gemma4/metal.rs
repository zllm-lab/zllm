//! Gemma 4 × Metal standalone 组合。

use crate::{
    backend::{
        LinearWeight,
        metal::{MetalContext, MetalKvCache, MetalWeight, report_metal_resource_plan},
    },
    config::{Gemma4ExecutionConfig, MetalBackendConfig},
    tokenizer::Tokenizer,
    vision::RgbImage,
    weight::model::gemma4::{Gemma4OutputWeight, Gemma4Weights},
};
use half::{bf16, f16};
use std::path::{Path, PathBuf};

use crate::runtime::gemma4::{
    Gemma4, Gemma4Config, Gemma4OutputHead, Gemma4PerLayerModel, Gemma4RopeTables, gemma4_decode_round, gemma4_embedding_rows, gemma4_last_token_output, gemma4_per_layer_embedding_rows, gemma4_per_layer_inputs, gemma4_prefill_hidden,
    gemma4_prefill_hidden_with_visibility, gemma4_token_output,
    multimodal::{gemma4_multimodal_embedding, gemma4_multimodal_input, prepare_gemma4_multimodal_model},
    prepare_gemma4_layers, prepare_gemma4_output_head_quantized, prepare_gemma4_per_layer_model,
};

pub fn run(
    model_path: &Path,
    prompt: &str,
    max_seq_len: usize,
    decode_steps: usize,
    image_paths: &[PathBuf],
    video_frame_paths: &[PathBuf],
    audio_paths: &[PathBuf],
    execution: Gemma4ExecutionConfig,
    backend: &MetalBackendConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mlx_affine = Gemma4Weights::is_mlx_affine(model_path)?;
    let model = Gemma4::new(Gemma4Weights::select_config(model_path)?).map_err(|error| format!("Gemma4 规格无效: {error:?}"))?;
    let cfg = model.config();
    crate::runtime::validate_max_sequence_length("Gemma4", max_seq_len, cfg.max_position_embeddings)?;
    let weights = Gemma4Weights::open(model_path, cfg.clone())?;
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = Tokenizer::new(&tokenizer_path)?;
    let detokenizer = crate::tokenizer::Detokenizer::load(&tokenizer_path)?;
    let user = if prompt == "[gMASK]<|user|>你好<|assistant|>" { "你好" } else { prompt };
    let images = image_paths.iter().map(RgbImage::open).collect::<Result<Vec<_>, _>>()?;
    let video_frames = video_frame_paths.iter().map(RgbImage::open).collect::<Result<Vec<_>, _>>()?;
    let audio_samples = audio_paths.iter().map(|path| crate::audio::read_wav_16khz(path)).collect::<Result<Vec<_>, _>>()?;
    let multimodal = (!images.is_empty() || !video_frames.is_empty() || !audio_samples.is_empty()).then(|| gemma4_multimodal_input(&tokenizer, cfg, user, &images, &video_frames, &audio_samples)).transpose()?;
    let prompt = if prompt.contains("<|turn>") { prompt.to_owned() } else { format!("<bos><|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>") };
    let tokens = multimodal.as_ref().map_or_else(|| tokenizer.tokenize(prompt.as_bytes()), |input| input.token_ids.clone());
    if tokens.is_empty() || tokens.len() > max_seq_len {
        return Err(format!("Gemma4 prompt tokens={}，max_seq_len={max_seq_len}", tokens.len()).into());
    }

    println!(
        "backend: Metal, model: Gemma4 {}L/{}h{}, prompt: {} tokens, images={}, video_frames={}, audio={}",
        cfg.layer_count,
        cfg.hidden_size,
        if mlx_affine { "-MLX" } else { "" },
        tokens.len(),
        images.len(),
        video_frames.len(),
        audio_samples.len(),
    );
    let ctx_owner = MetalContext::new_default_with_replay(backend.replay).map_err(|error| format!("MetalContext 初始化失败: {error}"))?;
    if let Some(operations) = backend.decode_batch_operations {
        ctx_owner.set_decode_batch_max_operations(operations);
    }
    let ctx = &ctx_owner;
    let mut cache = MetalKvCache::new_hybrid_gqa(ctx, model.hybrid_gqa().clone(), max_seq_len).map_err(|error| format!("Gemma4 hybrid KV cache: {error}"))?;
    report_metal_resource_plan(ctx, model.layer_count(), max_seq_len, cache.allocated_bytes(), 0, 0)?;
    let rope = Gemma4RopeTables::new(cfg, max_seq_len).map_err(|error| format!("Gemma4 RoPE: {error:?}"))?;
    let prepare_started = std::time::Instant::now();
    let layers = prepare_gemma4_layers(ctx, &model, &weights).map_err(|error| format!("准备 Gemma4 Metal 层: {error:?}"))?;
    let per_layer_model = prepare_gemma4_per_layer_model(ctx, cfg, &weights).map_err(|error| format!("准备 Gemma4 per-layer model: {error:?}"))?;
    let multimodal_model = multimodal
        .as_ref()
        .map(|_| {
            let source = weights.load_multimodal_weights().map_err(|error| format!("加载 Gemma4 multimodal 权重: {error}"))?;
            prepare_gemma4_multimodal_model(ctx, cfg, &source).map_err(|error| format!("准备 Gemma4 multimodal 权重: {error:?}"))
        })
        .transpose()?;
    eprintln!("[gemma4-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    let embedding_scale = bf16::from_f32(cfg.embedding_scale()).to_f32();
    let prefill_chunk_size = execution.prefill_chunk_size;
    let profile_prefill_gpu = backend.profile_prefill_gpu;
    if profile_prefill_gpu {
        ctx.reset_gpu_stats();
    }
    let prefill_started = std::time::Instant::now();
    let mut hidden = None;
    let mut position = 0usize;
    while position < tokens.len() {
        // chunk 准备链同样要进延迟批次,否则逐算子 commit+同步
        crate::backend::BackendResources::begin_batch(ctx);
        let preferred_end = position.saturating_add(prefill_chunk_size).min(tokens.len());
        let end = multimodal.as_ref().map_or(preferred_end, |input| input.chunk_end(position, preferred_end));
        let chunk = &tokens[position..end];
        let embedding_tokens = multimodal.as_ref().map_or(chunk, |input| &input.embedding_token_ids[position..end]);
        let embedding = gemma4_embedding_rows(&weights, embedding_tokens, cfg.hidden_size, embedding_scale)?;
        let chunk_hidden = match (multimodal.as_ref(), multimodal_model.as_ref()) {
            (Some(input), Some(model)) => gemma4_multimodal_embedding(ctx, cfg, model, input, &embedding, position..end).map_err(|error| format!("Gemma4 multimodal embedding: {error:?}"))?,
            _ => ctx.tensor_from_f32(&embedding, chunk.len(), cfg.hidden_size).map_err(|error| format!("上传 Gemma4 embedding: {error}"))?,
        };
        let per_layer_inputs = gemma4_metal_per_layer_inputs(ctx, cfg, &weights, per_layer_model.as_ref(), &chunk_hidden, embedding_tokens)?;
        let visual_visibility = multimodal.as_ref().filter(|input| input.chunk_has_visual_visibility(position..end));
        hidden = Some(
            match visual_visibility {
                Some(input) => gemma4_prefill_hidden_with_visibility(ctx, &mut cache, &model, &layers, &rope, chunk_hidden, per_layer_inputs.as_deref(), position, &input.visible_ends[position..end]),
                None => gemma4_prefill_hidden(ctx, &mut cache, &model, &layers, &rope, chunk_hidden, per_layer_inputs.as_deref(), position),
            }
            .map_err(|error| format!("Gemma4 Metal prefill position={position}: {error:?}"))?,
        );
        position = end;
    }
    let hidden = hidden.ok_or("Gemma4 prefill 缺少 token")?;
    ctx.synchronize();
    eprintln!("[gemma4-prefill] tokens={} chunks={} chunk_size={} wall={:.3}s", tokens.len(), tokens.len().div_ceil(prefill_chunk_size), prefill_chunk_size, prefill_started.elapsed().as_secs_f64(),);
    if profile_prefill_gpu {
        let gpu = ctx.gpu_stats();
        eprintln!("[gemma4-prefill-gpu] gpu={:.3}s commands={} submit_wait={:.3}s gaps={:.3}s tail={:.3}s", gpu.seconds, gpu.command_buffers, gpu.submit_wait_seconds, gpu.inter_command_gap_seconds, gpu.completion_tail_seconds,);
        for operator in ctx.gpu_profile().into_iter().take(20) {
            eprintln!("  gemma4 prefill gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape,);
        }
    }
    if decode_steps == 0 {
        return Ok(());
    }

    let output_head = prepare_gemma4_metal_output_head(ctx, cfg, &weights, crate::weight::LmHeadQuantization::Native)?;

    // 平铺转录重放:整步命令表录制一次,逐 token 重编码执行(诊断/基建路径)
    if ctx.replay_enabled() {
        let mut replay = crate::runtime::gemma4::metal_replay::Gemma4DecodeReplay::record(ctx, &mut cache, &model, &layers, &rope, &weights, per_layer_model.as_ref(), &output_head)?;
        eprintln!("[gemma4-replay] commands={}", replay.command_count());
        // 算子消融计时分解:按 (threads, grid) 模式分类,逐级剔除测真实成本。
        // 输出无效,仅计时;谓词按 dump 的每层 17-op 周期归纳。
        if execution.replay_ablation {
            // 三档依次连续推进 position(单次 record,KV 游标跨档连续,避免重置)
            type Op = crate::backend::metal::api::RecordedComputeOp;
            let steps = decode_steps.clamp(16, 32);
            let tiers_spec: [(&str, fn(&Op) -> bool); 3] =
                [("full(全部 813 op)", |_| true), ("gemv+attn(剔 x32 norm 与 segmented)", |op: &Op| op.threads.width == 64 || (op.threads.width == 256 && op.groups.width > 1)), ("gemv-only(仅 6 gemv/层)", |op: &Op| op.threads.width == 64)];
            let first = gemma4_last_token_output(ctx, cfg, &output_head, &hidden, hidden.rows - 1).map_err(|error| format!("Gemma4 output head: {error:?}"))?;
            let mut token = first.token_id;
            let mut position = tokens.len();
            for (name, keep) in tiers_spec.iter() {
                let kept = replay.ops().iter().filter(|op| keep(op)).count();
                let started = std::time::Instant::now();
                for _ in 0..steps {
                    token = replay.step_filtered(token, position, keep).map_err(|error| format!("消融重放失败: {error}"))?;
                    position += 1;
                }
                eprintln!("[gemma4-ablation] {name}: kept={kept} 平均 {:.2} ms/步", started.elapsed().as_secs_f64() / steps as f64 * 1.0e3);
            }
            println!();
            return Ok(());
        }
        let first = gemma4_last_token_output(ctx, cfg, &output_head, &hidden, hidden.rows - 1).map_err(|error| format!("Gemma4 output head: {error:?}"))?;
        let _ = crate::runtime::generation::write_token(&detokenizer, first.token_id, false);
        let decode_started = std::time::Instant::now();
        let mut token = first.token_id;
        let mut generated = 0usize;
        for step in 0..decode_steps {
            if cfg.eos_token_ids.contains(&token) {
                break;
            }
            token = replay.step(token, tokens.len() + step)?;
            if cfg.eos_token_ids.contains(&token) {
                break;
            }
            let _ = crate::runtime::generation::write_token(&detokenizer, token, false);
            generated += 1;
        }
        println!();
        eprintln!("[gemma4-decode] tokens={generated} replay wall={:.3}s", decode_started.elapsed().as_secs_f64());
        return Ok(());
    }

    let profile_decode = backend.profile_decode;
    let mut decode_seconds = 0.0f64;
    let mut output = gemma4_last_token_output(ctx, cfg, &output_head, &hidden, hidden.rows - 1).map_err(|error| format!("Gemma4 output head: {error:?}"))?;
    let stats = crate::runtime::generation::run_generation(
        &mut output,
        crate::runtime::generation::GenerationLimits { prompt_tokens: tokens.len(), max_tokens: decode_steps, max_sequence_length: max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |output, _| Ok::<_, Box<dyn std::error::Error>>(output.token_id),
        |token, _| crate::runtime::generation::write_token(&detokenizer, token, false),
        |output, token, position, step| {
            if profile_decode {
                ctx.reset_gpu_stats();
                ctx.reset_decode_cpu_breakdown();
            }
            let round_started = std::time::Instant::now();
            // PLE 链 40+ 小算子必须在延迟批次内,否则逐算子 commit+同步白付 ~10ms
            crate::backend::BackendResources::begin_decode_batch(ctx);
            let embedding = gemma4_embedding_rows(&weights, &[token], cfg.hidden_size, embedding_scale)?;
            let input = ctx.tensor_from_f32(&embedding, 1, cfg.hidden_size).map_err(|error| format!("上传 Gemma4 decode embedding: {error}"))?;
            let embed_seconds = round_started.elapsed().as_secs_f64();
            let per_layer_started = std::time::Instant::now();
            let per_layer_inputs = gemma4_metal_per_layer_inputs(ctx, cfg, &weights, per_layer_model.as_ref(), &input, &[token])?;
            let ple_seconds = per_layer_started.elapsed().as_secs_f64();
            let prep_seconds = round_started.elapsed().as_secs_f64();
            let encode_started = std::time::Instant::now();
            let hidden = gemma4_decode_round(ctx, &mut cache, &model, &layers, &rope, input, per_layer_inputs.as_deref(), position).map_err(|error| format!("Gemma4 Metal decode position={position}: {error:?}"))?;
            let round_encode_seconds = encode_started.elapsed().as_secs_f64();
            *output = gemma4_token_output(ctx, cfg, &output_head, &hidden).map_err(|error| format!("Gemma4 output head: {error:?}"))?;
            let round_seconds = round_started.elapsed().as_secs_f64();
            decode_seconds += round_seconds;
            if profile_decode {
                let gpu = ctx.gpu_stats();
                let (alloc_ns, commit_ns) = ctx.decode_cpu_breakdown();
                eprintln!(
                    "[gemma4-decode-profile] step={step} wall={:.3}s gpu={:.3}s embed={:.3}s ple={:.3}s round={:.3}s head={:.3}s commands={} | cpu: alloc={:.1}ms commit={:.1}ms 其余={:.1}ms",
                    round_seconds,
                    gpu.seconds,
                    embed_seconds,
                    ple_seconds,
                    round_encode_seconds,
                    round_seconds - prep_seconds - round_encode_seconds,
                    gpu.command_buffers,
                    alloc_ns as f64 / 1.0e6,
                    commit_ns as f64 / 1.0e6,
                    (round_seconds * 1.0e3) - alloc_ns as f64 / 1.0e6 - commit_ns as f64 / 1.0e6,
                );
                eprintln!("[gemma4-decode-gpu] 算子={:.1}ms 间隙={:.1}ms 提交等待={:.1}ms 尾部={:.1}ms", gpu.seconds * 1.0e3, gpu.inter_command_gap_seconds * 1.0e3, gpu.submit_wait_seconds * 1.0e3, gpu.completion_tail_seconds * 1.0e3);
                if matches!(step, 0 | 15 | 31 | 47 | 62) {
                    for operator in ctx.gpu_profile().into_iter().take(12) {
                        eprintln!("  gemma4 gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape,);
                    }
                }
            }
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    eprintln!("[gemma4-decode] tokens={} recurrent_rounds={} wall={decode_seconds:.3}s throughput={:.3} tok/s", stats.generated_tokens, stats.decode_rounds, stats.decode_rounds as f64 / decode_seconds.max(f64::EPSILON),);
    Ok(())
}

/// 由权重目录构造 Metal 输出头；量化与 BF16/F16/F32 dense LM head 都支持。
#[cfg(target_os = "macos")]
/// 平台组合只负责把模型 runtime 准备好的 PLE 行上传到 Metal。
pub fn gemma4_metal_per_layer_inputs(
    ctx: &MetalContext,
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    model: Option<&Gemma4PerLayerModel<MetalWeight>>,
    hidden: &crate::backend::metal::MetalTensor,
    tokens: &[u32],
) -> Result<Option<Vec<crate::backend::metal::MetalTensor>>, String> {
    if cfg.per_layer_input_size == 0 {
        return Ok(None);
    }
    let values = gemma4_per_layer_embedding_rows(cfg, weights, tokens)?;
    let columns = cfg.layer_count.checked_mul(cfg.per_layer_input_size).ok_or("Gemma4 per-layer embedding 列数溢出")?;
    let token_inputs = ctx.tensor_from_f32(&values, tokens.len(), columns).map_err(|error| format!("上传 Gemma4 per-layer embedding: {error}"))?;
    gemma4_per_layer_inputs(ctx, cfg, model, hidden, Some(token_inputs)).map_err(|error| format!("准备 Gemma4 per-layer inputs: {error:?}"))
}

pub fn prepare_gemma4_metal_output_head(ctx: &MetalContext, cfg: &Gemma4Config, weights: &Gemma4Weights, quantization: crate::weight::LmHeadQuantization) -> Result<Gemma4OutputHead<MetalWeight>, String> {
    let final_norm = weights.final_norm()?;
    let output_weight = weights.load_output_weight()?;
    match output_weight {
        Gemma4OutputWeight::Quantized(weight) => prepare_gemma4_output_head_quantized(ctx, cfg, &final_norm, LinearWeight::Quantized(weight.as_ref()), quantization),
        Gemma4OutputWeight::Dense(lm_head) if lm_head.dtype == "BF16" => {
            let values: Vec<f16> = lm_head.data.chunks_exact(2).map(|bytes| f16::from_f32(bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())).collect();
            prepare_gemma4_output_head_quantized(ctx, cfg, &final_norm, LinearWeight::F16(&values), quantization)
        }
        Gemma4OutputWeight::Dense(lm_head) if lm_head.dtype == "F16" => {
            let values: Vec<f16> = lm_head.data.chunks_exact(2).map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]))).collect();
            prepare_gemma4_output_head_quantized(ctx, cfg, &final_norm, LinearWeight::F16(&values), quantization)
        }
        Gemma4OutputWeight::Dense(lm_head) if lm_head.dtype == "F32" => {
            let values: Vec<f32> = lm_head.data.chunks_exact(4).map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])).collect();
            prepare_gemma4_output_head_quantized(ctx, cfg, &final_norm, LinearWeight::F32(&values), quantization)
        }
        Gemma4OutputWeight::Dense(lm_head) => return Err(format!("Gemma4 LM head dtype={} 暂不支持", lm_head.dtype)),
    }
    .map_err(|error| format!("准备 Gemma4 output head: {error:?}"))
}
