//! Qwen3.6 × Metal standalone 组合。

use crate::{
    attention::{gated_delta_net::GatedDeltaNetState, rope::RopeTable},
    backend::{
        Backend,
        metal::{MetalContext, MetalGatedDeltaNetStorage, MetalKvCache},
    },
    config::{KvCacheFormat, MetalBackendConfig, Qwen36ExecutionConfig},
    weight::container::gguf::GgufReader,
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub fn run(
    model_path: &Path,
    prompt: &str,
    max_seq_len: usize,
    decode_steps: usize,
    image_paths: &[PathBuf],
    video_paths: &[PathBuf],
    execution: Qwen36ExecutionConfig,
    backend: &MetalBackendConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = crate::runtime::qwen36::Qwen36Config::standard_27b();
    let located = GgufReader::locate(model_path)?;
    let reader = GgufReader::open(&located)?;
    reader.expect_metadata_str("general.architecture", "qwen35")?;
    reader.expect_metadata_u64("qwen35.embedding_length", cfg.hidden_size as u64)?;
    // GGUF 可能带 MTP block(block_count = num_layers + mtp_layers)，prefill/decode 只执行前 num_layers 层。
    let block_count = reader.metadata_u64("qwen35.block_count")?;
    if block_count != cfg.num_layers as u64 && block_count != (cfg.num_layers + cfg.mtp_layers) as u64 {
        return Err(format!("GGUF metadata qwen35.block_count={block_count}，期望 {} 或 {}", cfg.num_layers, cfg.num_layers + cfg.mtp_layers).into());
    }
    let weights = Arc::new(reader);
    let has_visuals = !image_paths.is_empty() || !video_paths.is_empty();
    // 有图像或视频输入时加载 mmproj 视觉 GGUF(同目录下 mmproj-*.gguf)
    let mmproj = if has_visuals {
        let mmproj_path = std::fs::read_dir(model_path.is_dir().then(|| model_path).unwrap_or_else(|| model_path.parent().unwrap_or(Path::new("."))))
            .map_err(|error| format!("查找 mmproj 失败: {error}"))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("mmproj") && n.ends_with(".gguf")))
            .ok_or("未找到 mmproj-*.gguf 视觉权重文件")?;
        eprintln!("[qwen36-vision] mmproj: {}", mmproj_path.display());
        Some(Arc::new(GgufReader::open(&mmproj_path)?))
    } else {
        None
    };
    let tokenizer = weights.bpe_tokenizer()?;
    // 多模态:预处理图像/视频 → 渲染 placeholder → tokenize + M-RoPE 位置。
    // 纯文本路径保持原样:统一走 chat 模板包装。
    let multimodal = if has_visuals {
        use crate::runtime::qwen3_vl::Qwen3VlImageProcessor;
        use crate::vision::{ImageProcessor, RgbImage};

        let processor = Qwen3VlImageProcessor::new(&cfg.vision.to_qwen3vl())?;
        let mut visuals = Vec::with_capacity(image_paths.len() + video_paths.len());
        for path in image_paths {
            let image = RgbImage::open(path)?;
            let tensor = processor.preprocess(&image)?;
            eprintln!("[qwen36-vision] image={}x{} patches={} visual_tokens={}", image.width, image.height, tensor.rows, tensor.visual_token_count()?);
            visuals.push(crate::runtime::qwen36::Qwen36Visual { kind: crate::runtime::qwen36::Qwen36VisualKind::Image, tensor });
        }
        for path in video_paths {
            let frames = crate::vision::video::decode_video(path)?;
            let (frames, _) = crate::vision::video::qwen_video_frames(&frames)?;
            let tensor = processor.preprocess_video(&frames)?;
            eprintln!("[qwen36-vision] video={} frames={} patches={} visual_tokens={}", path.display(), frames.len(), tensor.rows, tensor.visual_token_count()?);
            visuals.push(crate::runtime::qwen36::Qwen36Visual { kind: crate::runtime::qwen36::Qwen36VisualKind::Video, tensor });
        }
        let input = crate::runtime::qwen36::qwen36_multimodal_input(&tokenizer, &cfg, prompt, &visuals)?;
        eprintln!("[qwen36-vision] visuals={} tokens={} rope_delta={}", input.visual_ranges.len(), input.token_ids.len(), input.rope_delta);
        Some((input, visuals))
    } else {
        None
    };
    let tokens = match &multimodal {
        Some((input, _)) => input.token_ids.clone(),
        None => tokenizer.tokenize(crate::runtime::qwen36::chat_prompt(prompt).as_bytes()),
    };
    if tokens.is_empty() || tokens.len() > max_seq_len {
        return Err(format!("Qwen3.6 prompt tokens={}，max_seq_len={max_seq_len}", tokens.len()).into());
    }

    println!("backend: Metal, model: Qwen3.6-27B, prompt: {} tokens", tokens.len());
    let ctx_owner = MetalContext::new_default().map_err(|error| format!("MetalContext 初始化失败: {error}"))?;
    let ctx = &ctx_owner;

    let prepare_started = std::time::Instant::now();
    let layers = crate::runtime::qwen36::prepare_qwen36_gguf_layers(ctx, weights.as_ref(), &cfg).map_err(|error| format!("准备 Qwen3.6 Metal 层: {error:?}"))?;
    eprintln!("[qwen36-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    let attention = crate::attention::AttentionSpec::Gqa(cfg.full_attention_spec());
    let cache_spec = crate::kv_cache::KvCacheSpec::from_attention(&attention).map_err(|error| format!("Qwen3.6 KV cache spec: {error}"))?;
    // 每 4 层才有 1 层 FullAttention 需要 KV cache
    let full_attention_layers: Vec<usize> = (0..cfg.num_layers).filter(|&layer| (layer + 1) % cfg.full_attention_interval == 0).collect();
    let cache_layers = crate::kv_cache::KvCacheLayerMap::from_cached_layers(cfg.num_layers, full_attention_layers.into_iter()).map_err(|error| format!("Qwen3.6 KV cache layer map: {error}"))?;
    let kv_f16 = execution.kv_cache_format == KvCacheFormat::F16;
    let mut cache = if kv_f16 { MetalKvCache::new_f16_mapped(ctx, cache_spec.clone(), cache_layers, max_seq_len) } else { MetalKvCache::new_mapped(ctx, cache_spec.clone(), cache_layers, max_seq_len) }
        .map_err(|error| format!("Qwen3.6 KV cache: {error}"))?;
    eprintln!("[qwen36-kv-cache] format={} logical_layers={} slots={} allocated_mib={:.2}", if kv_f16 { "F16" } else { "Q8G64" }, cache.layer_count(), cache.cache_slot_count(), cache.allocated_bytes() as f64 / (1024.0 * 1024.0),);
    let mut delta_state = GatedDeltaNetState::<MetalGatedDeltaNetStorage>::new(cfg.num_layers, cfg.gated_delta_net_spec()).map_err(|error| format!("Qwen3.6 DeltaNet state: {error:?}"))?;
    // 多模态用 M-RoPE 三轴表(位置已按 HF 语义压缩)，纯文本保持顺序表。
    let rope_owner = match &multimodal {
        Some((input, _)) => crate::runtime::qwen36::qwen36_mrope_table(&cfg, &input.position_ids).map_err(|error| format!("Qwen3.6 M-RoPE 表: {error}"))?,
        None => RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta),
    };
    let attention_options = crate::attention::hybrid::HybridAttentionOptions { precise_prefill: execution.precise_gqa_prefill };
    let runtime = crate::runtime::qwen36::Qwen36Runtime::new(ctx, &cfg, &layers, &rope_owner, attention_options);

    let embedding = weights.embedding_rows("token_embd.weight", &tokens, cfg.hidden_size, cfg.vocab_size)?;
    let mut hidden = ctx.tensor_from_f32(&embedding, tokens.len(), cfg.hidden_size).map_err(|error| format!("上传 Qwen3.6 embedding: {error}"))?;

    // 多模态:逐 visual ViT encode → scatter 到对应 placeholder token 行
    if let Some(((input, visuals), mmproj_reader)) = multimodal.as_ref().zip(mmproj.as_ref()) {
        let vision_started = std::time::Instant::now();
        let patch = crate::runtime::qwen36::prepare_qwen36_patch_embedding(ctx, mmproj_reader, &cfg).map_err(|error| format!("Qwen3.6 patch embed: {error:?}"))?;
        let position_embedding = mmproj_reader.read_tensor_f32("v.position_embd.weight").map_err(|error| format!("Qwen3.6 position embedding: {error}"))?;
        let vision_layers = crate::runtime::qwen36::prepare_qwen36_vision_layers(ctx, mmproj_reader, &cfg).map_err(|error| format!("Qwen3.6 vision layers: {error:?}"))?;
        let merger = crate::runtime::qwen36::prepare_qwen36_vision_merger(ctx, mmproj_reader, &cfg).map_err(|error| format!("Qwen3.6 vision merger: {error:?}"))?;
        use crate::backend::VisionBackend;
        for (index, (range, visual)) in input.visual_ranges.iter().zip(visuals).enumerate() {
            let vision_embedding = crate::runtime::qwen36::qwen36_encode_image(ctx, &cfg, &patch, &position_embedding, &vision_layers, &merger, &visual.tensor).map_err(|error| format!("Qwen3.6 visual {index} encode: {error:?}"))?;
            ctx.scatter_rows(&mut hidden, range.start, &vision_embedding).map_err(|error| format!("Qwen3.6 visual {index} scatter: {error:?}"))?;
        }
        eprintln!("[qwen36-vision-encode] visuals={} wall={:.3}s", visuals.len(), vision_started.elapsed().as_secs_f64());
    }

    let profile_prefill_gpu = backend.profile_prefill_gpu;
    if profile_prefill_gpu {
        ctx.reset_gpu_stats();
    }
    let prefill_started = std::time::Instant::now();
    let mut hidden = runtime.at(&mut cache, &mut delta_state, 0).prefill(hidden).map_err(|error| format!("Qwen3.6 Metal prefill: {error:?}"))?;
    eprintln!("[qwen36-prefill] tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64());
    if profile_prefill_gpu {
        let gpu = ctx.gpu_stats();
        eprintln!("[qwen36-prefill-gpu] gpu={:.3}s commands={} submit_wait={:.3}s gaps={:.3}s tail={:.3}s", gpu.seconds, gpu.command_buffers, gpu.submit_wait_seconds, gpu.inter_command_gap_seconds, gpu.completion_tail_seconds);
        for operator in ctx.gpu_profile().into_iter().take(20) {
            eprintln!("  qwen36 prefill gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
        }
    }

    if decode_steps == 0 {
        return Ok(());
    }

    let (final_norm, output_head) = crate::runtime::qwen36::prepare_qwen36_gguf_output(ctx, weights.as_ref()).map_err(|error| format!("准备 Qwen3.6 output: {error:?}"))?;
    let detokenizer = weights.bpe_detokenizer()?;

    // 取最后一个 token 的 hidden 用于 decode
    hidden = ctx.select_row(&hidden, tokens.len() - 1).map_err(|error| format!("Qwen3.6 选择最后 token: {error:?}"))?;

    let profile_decode = backend.profile_decode;
    // 多模态 decode 后三个轴重新一致，只需按 rope_delta 预计算一张常驻顺序表；
    // 不能每 token 重建 0..position，否则生成阶段会退化为 O(n²)。
    let rope_delta = multimodal.as_ref().map(|(input, _)| input.rope_delta);
    let decode_rope = rope_delta.map(|delta| crate::runtime::qwen36::qwen36_decode_rope_table(&cfg, max_seq_len - 1, delta).map_err(|error| format!("Qwen3.6 decode rope: {error}"))).transpose()?;
    let mut generation_state = (hidden, None::<std::time::Instant>);
    let stats = crate::runtime::generation::run_generation(
        &mut generation_state,
        crate::runtime::generation::GenerationLimits { prompt_tokens: tokens.len(), max_tokens: decode_steps, max_sequence_length: max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |state, step| {
            if profile_decode {
                ctx.reset_gpu_stats();
            }
            state.1 = Some(std::time::Instant::now());
            let normalized = ctx.gemma_rmsnorm_f32(&state.0, &final_norm, cfg.rms_norm_eps).map_err(|error| format!("Qwen3.6 output norm: {error:?}"))?;
            let logits = ctx.linear(&normalized, &output_head).map_err(|error| format!("Qwen3.6 output head: {error:?}"))?;
            let token = ctx.argmax(&logits).map_err(|error| format!("Qwen3.6 argmax: {error:?}"))?;
            if backend.trace_tokens {
                eprintln!("[qwen36-decode] step={step} token={token}");
            }
            Ok::<_, Box<dyn std::error::Error>>(token)
        },
        |token, _| crate::runtime::generation::write_token(&detokenizer, token, false),
        |state, token, position, step| {
            let embedding = weights.embedding_rows("token_embd.weight", &[token], cfg.hidden_size, cfg.vocab_size)?;
            let input = ctx.tensor_from_f32(&embedding, 1, cfg.hidden_size).map_err(|error| format!("上传 Qwen3.6 decode embedding: {error}"))?;
            state.0 = match &decode_rope {
                Some(rope) => crate::runtime::qwen36::Qwen36Runtime::new(ctx, &cfg, &layers, rope, attention_options)
                    .at(&mut cache, &mut delta_state, position)
                    .decode(input)
                    .map_err(|error| format!("Qwen3.6 Metal decode position={position}: {error:?}"))?,
                None => runtime.at(&mut cache, &mut delta_state, position).decode(input).map_err(|error| format!("Qwen3.6 Metal decode position={position}: {error:?}"))?,
            };
            if profile_decode {
                let gpu = ctx.gpu_stats();
                eprintln!("[qwen36-decode-profile] step={step} wall={:.3}s gpu={:.3}s commands={}", state.1.take().expect("select_token 已设置计时").elapsed().as_secs_f64(), gpu.seconds, gpu.command_buffers);
                for operator in ctx.gpu_profile().into_iter().take(15) {
                    eprintln!("  qwen36 decode gpu {:>8.3} ms x{} | {} {}", operator.gpu_seconds * 1.0e3, operator.calls, operator.operator, operator.shape);
                }
            }
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    eprintln!("[qwen36-decode-summary] tokens={} wall={:.3}s throughput={:.3} tok/s", stats.generated_tokens, stats.elapsed.as_secs_f64(), stats.tokens_per_second(),);
    Ok(())
}
