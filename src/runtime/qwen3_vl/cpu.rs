//! Qwen3-VL × CPU reference 组合,以及经典 dense Qwen3-14B 的 CPU oracle 入口。

use crate::{
    backend::{
        VisionBackend,
        cpu::{CpuContext, CpuKvCache},
    },
    kernel::cpu::CpuTensor,
    runtime::qwen3_vl::{
        Qwen3VlConfig, prepare_qwen3_vl_output_head, prepare_qwen3_vl_text_layer, qwen3_vl_decode_rope_table, qwen3_vl_decode_round_resident, qwen3_vl_encode_image, qwen3_vl_last_token_output, qwen3_vl_mrope_table,
        qwen3_vl_prefill_resident, qwen3_vl_single_image_input, qwen3_vl_token_output,
    },
    tokenizer::load_bpe_directory,
    vision::RgbImage,
    weight::model::qwen3_vl::Qwen3VlWeights,
};
use rayon::prelude::*;
use std::time::Instant;

/// 经典 dense Qwen3-14B(NVFP4)CPU oracle 入口,对拍 `cuda::run` 的 greedy 序列。
///
/// 用法: `zllm-qwen3-cpu --model DIR [--prompt TEXT] [--max-tokens N] [--max-seq-len N] [--lm-head-q8g128]`。
/// NVFP4/W8A16 矩阵在 `CpuContext::prepare_weight` 解码为 f32 常驻(14B 约 53GB 内存)。
pub fn run_dense(model_dir: &std::path::Path, prompt: &str, max_seq_len: usize, decode_steps: usize, lm_head_quantization: crate::weight::LmHeadQuantization) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Qwen3VlConfig::dense_14b();
    if max_seq_len > cfg.max_position_embeddings {
        return Err(format!("max_seq_len={max_seq_len} 超过模型上限 {}", cfg.max_position_embeddings).into());
    }
    let weights = crate::weight::model::qwen3::Qwen3Weights::open(model_dir, cfg.clone())?;
    let (tokenizer, detokenizer) = load_bpe_directory(model_dir)?;
    let prompt_text = crate::runtime::qwen3_vl::chat_prompt(prompt);
    let token_ids = tokenizer.tokenize(prompt_text.as_bytes());
    if token_ids.is_empty() || token_ids.len() > max_seq_len {
        return Err(format!("Qwen3-14B prompt tokens={} 非法(max_seq_len={max_seq_len})", token_ids.len()).into());
    }
    eprintln!("[qwen3-cpu] model={} prompt_tokens={} max_seq_len={max_seq_len} lm_head={lm_head_quantization:?}", model_dir.display(), token_ids.len());

    let backend = CpuContext;
    let mut cache = CpuKvCache::new(cfg.layer_count);

    let prepare_started = Instant::now();
    let resident = (0..cfg.layer_count).map(|layer| prepare_qwen3_vl_text_layer(&backend, &weights, layer).map_err(|error| format!("准备 Qwen3-14B L{layer}: {error:?}"))).collect::<Result<Vec<_>, _>>()?;
    let output_head = crate::runtime::qwen3_vl::prepare_qwen3_vl_output_head_quantized(&backend, &cfg, &weights, lm_head_quantization).map_err(|error| format!("准备 Qwen3-14B output head: {error:?}"))?;
    eprintln!("[qwen3-cpu] resident_layers={} decode_preload={:.3}s", resident.len(), prepare_started.elapsed().as_secs_f64());

    let embedding = weights.embedding_rows_f32(&token_ids)?;
    let hidden = CpuTensor { data: embedding, rows: token_ids.len(), cols: cfg.hidden_size };
    let positions: [Vec<usize>; 3] = std::array::from_fn(|_| (0..token_ids.len()).collect());
    let rope = qwen3_vl_mrope_table(&cfg, &positions)?;

    let prefill_started = Instant::now();
    let hidden = crate::runtime::qwen3_vl::qwen3_vl_text_prefill(&backend, &cfg, &weights, &resident, &mut cache, hidden, &rope, 0).map_err(|error| format!("Qwen3-14B CPU prefill: {error:?}"))?;
    eprintln!("[qwen3-cpu] prefill wall={:.3}s", prefill_started.elapsed().as_secs_f64());

    let mut output = qwen3_vl_last_token_output(&backend, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3-14B CPU first token: {error:?}"))?;

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
            let hidden = CpuTensor { data: embedding, rows: 1, cols: cfg.hidden_size };
            let rope = qwen3_vl_decode_rope_table(&cfg, position, 0)?;
            let hidden = qwen3_vl_decode_round_resident(&backend, &cfg, &weights, &resident, &mut cache, hidden, &rope, position).map_err(|error| format!("Qwen3-14B CPU decode position={position}: {error:?}"))?;
            *output = qwen3_vl_token_output(&backend, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3-14B CPU output position={position}: {error:?}"))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    let answer = detokenizer.decode_bytes(&generated, true)?;
    println!("{}", String::from_utf8_lossy(&answer));
    eprintln!("[qwen3-cpu] tokens={:?} generated={} decode_rounds={} throughput={:.3} tok/s", generated, stats.generated_tokens, stats.decode_rounds, stats.decode_tokens_per_second());
    Ok(())
}

pub fn run(model_dir: &std::path::Path, prompt: &str, image_path: &std::path::Path, max_seq_len: usize, decode_steps: usize) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Qwen3VlConfig::instruct_32b();
    if max_seq_len > cfg.max_position_embeddings {
        return Err(format!("max_seq_len={} 超过模型上限 {}", max_seq_len, cfg.max_position_embeddings).into());
    }
    let weights = Qwen3VlWeights::open(&model_dir, cfg.clone())?;
    let (tokenizer, detokenizer) = load_bpe_directory(&model_dir)?;
    let image = RgbImage::open(image_path)?;
    let input = qwen3_vl_single_image_input(&tokenizer, &cfg, &prompt, &image)?;
    if input.token_ids.len() > max_seq_len {
        return Err(format!("prompt tokens={} 超过 max_seq_len={}", input.token_ids.len(), max_seq_len).into());
    }
    eprintln!("[qwen3-vl] model={} image={}x{} patches={} visual_tokens={} prompt_tokens={} W4A16 group=32", model_dir.display(), image.width, image.height, input.image.rows, input.image_tokens.len(), input.token_ids.len());

    let backend = CpuContext;
    let mut cache = CpuKvCache::new(cfg.layer_count);

    // 视觉编码(前处理):全部在 CPU 完成,不占用显卡。
    let vision_started = Instant::now();
    let vision_config = cfg.vision.as_ref().ok_or("dense Qwen3 文本模型不支持视觉输入")?;
    let vision = qwen3_vl_encode_image(&backend, vision_config, &weights, &input.image).map_err(|error| format!("Qwen3-VL vision: {error:?}"))?;
    eprintln!("[qwen3-vl] vision wall={:.3}s deepstack={}", vision_started.elapsed().as_secs_f64(), vision.deepstack.len());

    let embedding = weights.embedding_rows_f32(&input.token_ids)?;
    let mut hidden = CpuTensor { data: embedding, rows: input.token_ids.len(), cols: cfg.hidden_size };
    backend.scatter_rows(&mut hidden, input.image_tokens.start, &vision.embedding).map_err(|error| format!("写入 Qwen3-VL image embedding: {error:?}"))?;
    let rope = qwen3_vl_mrope_table(&cfg, &input.position_ids)?;

    // 预加载全部文本层(resident),decode 时不再逐层解码 W4A16 权重。
    let prepare_started = Instant::now();
    let resident = (0..cfg.layer_count).into_par_iter().map(|layer| prepare_qwen3_vl_text_layer(&backend, &weights, layer).map_err(|error| format!("准备 Qwen3-VL L{layer}: {error:?}"))).collect::<Result<Vec<_>, _>>()?;
    eprintln!("[qwen3-vl] resident_layers={} preload={:.3}s", resident.len(), prepare_started.elapsed().as_secs_f64());

    let prefill_started = Instant::now();
    let hidden = qwen3_vl_prefill_resident(&backend, &cfg, &weights, &resident, &mut cache, hidden, &rope, &input.image_tokens, &vision.deepstack).map_err(|error| format!("Qwen3-VL prefill: {error:?}"))?;
    eprintln!("[qwen3-vl] prefill wall={:.3}s", prefill_started.elapsed().as_secs_f64());

    let output_head = prepare_qwen3_vl_output_head(&backend, &cfg, &weights).map_err(|error| format!("准备 Qwen3-VL output head: {error:?}"))?;
    let mut output = qwen3_vl_last_token_output(&backend, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3-VL first token: {error:?}"))?;

    // --decode-steps 控制生成长度(包含 prefill 给出的首个 token)。
    let max_tokens = decode_steps.max(1);
    let stats = crate::runtime::generation::run_generation(
        &mut output,
        crate::runtime::generation::GenerationLimits { prompt_tokens: input.token_ids.len(), max_tokens, max_sequence_length: max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |output, _| Ok::<_, Box<dyn std::error::Error>>(output.token_id),
        |token, _| crate::runtime::generation::write_token(&detokenizer, token, true),
        |output, token, position, _| {
            let embedding = weights.embedding_rows_f32(&[token])?;
            let hidden = CpuTensor { data: embedding, rows: 1, cols: cfg.hidden_size };
            let rope = qwen3_vl_decode_rope_table(&cfg, position, input.rope_delta)?;
            let hidden = qwen3_vl_decode_round_resident(&backend, &cfg, &weights, &resident, &mut cache, hidden, &rope, position).map_err(|error| format!("Qwen3-VL decode position={position}: {error:?}"))?;
            *output = qwen3_vl_token_output(&backend, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3-VL output position={position}: {error:?}"))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    eprintln!("[qwen3-vl] generated={} decode_rounds={} wall={:.3}s throughput={:.3} tok/s", stats.generated_tokens, stats.decode_rounds, stats.elapsed.as_secs_f64(), stats.decode_tokens_per_second());
    Ok(())
}
