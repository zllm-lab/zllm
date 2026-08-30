//! Qwen3.6 × CPU reference 组合。

use crate::{
    attention::{
        gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetState},
        rope::RopeTable,
    },
    backend::cpu::{CpuContext, CpuGatedDeltaNetStorage, CpuKvCache},
    kernel::cpu::CpuTensor,
    runtime::qwen36::{Qwen36Config, Qwen36Runtime, chat_prompt, prepare_qwen36_layers, prepare_qwen36_output_head, qwen36_token_output},
    tokenizer::load_bpe_directory,
    weight::{format::mlx_affine::MlxAffineSource, model::qwen36::Qwen36Weights},
};
use std::time::Instant;

/// Qwen3.6-27B 纯文本入口:safetensors + MLX affine 4bit → CPU reference。
/// 与 Qwen3-VL CPU 入口共享标准 BPE 目录加载与 GGUF 之外的 M-RoPE 工具。
pub fn run(model_dir: &std::path::Path, prompt: &str, max_seq_len: usize, decode_steps: usize, attention: crate::attention::hybrid::HybridAttentionOptions) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Qwen36Config::standard_27b();
    crate::runtime::validate_max_sequence_length("Qwen3.6", max_seq_len, cfg.max_position_embeddings)?;
    let source = MlxAffineSource::open(&model_dir).map_err(|error| format!("打开 Qwen3.6 MLX affine safetensors: {error}"))?;
    let weights = Qwen36Weights::new(source, cfg.clone()).map_err(|error| format!("加载 Qwen3.6 权重: {error}"))?;
    let (tokenizer, detokenizer) = load_bpe_directory(&model_dir)?;
    let prompt_text = chat_prompt(&prompt);
    let token_ids = tokenizer.tokenize(prompt_text.as_bytes());
    if token_ids.is_empty() {
        return Err("Qwen3.6 prompt tokenize 为空".into());
    }
    if token_ids.len() > max_seq_len {
        return Err(format!("prompt tokens={} 超过 max_seq_len={}", token_ids.len(), max_seq_len).into());
    }
    eprintln!("[qwen36] model={} prompt_tokens={} max_seq_len={} MLX-affine 4bit", model_dir.display(), token_ids.len(), max_seq_len);

    let backend = CpuContext;
    let mut cache = CpuKvCache::new(cfg.num_layers);
    let mut delta_state = GatedDeltaNetState::<CpuGatedDeltaNetStorage>::with_head_layout(cfg.num_layers, cfg.gated_delta_net_spec(), GatedDeltaNetHeadLayout::Grouped).map_err(|error| format!("Qwen3.6 DeltaNet state: {error:?}"))?;

    // 预加载所有 64 层到常驻权重(MLX affine 矩阵在 prepare_weight 内 dequant 到 f32)。
    // decode 时不再逐层反量化,直接复用 prepared weight。
    let prepare_started = Instant::now();
    let layers = prepare_qwen36_layers(&backend, &weights).map_err(|error| format!("准备 Qwen3.6 CPU 层: {error:?}"))?;
    eprintln!("[qwen36] resident_layers={} preload={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    // 纯文本三轴位置一致，预计算完整标准 RoPE 供 prefill/decode 共同复用。
    let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);
    let runtime = Qwen36Runtime::new(&backend, &cfg, &layers, &rope, attention);

    let embedding = weights.embedding_rows_f32(&token_ids)?;
    let hidden = CpuTensor { data: embedding, rows: token_ids.len(), cols: cfg.hidden_size };

    let prefill_started = Instant::now();
    let hidden = runtime.at(&mut cache, &mut delta_state, 0).prefill(hidden).map_err(|error| format!("Qwen3.6 CPU prefill: {error:?}"))?;
    eprintln!("[qwen36] prefill wall={:.3}s", prefill_started.elapsed().as_secs_f64());

    let output_head = prepare_qwen36_output_head(&backend, &weights).map_err(|error| format!("准备 Qwen3.6 output head: {error:?}"))?;
    let mut output = qwen36_token_output(&backend, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3.6 first token: {error:?}"))?;

    if decode_steps == 0 {
        return Ok(());
    }
    let stats = crate::runtime::generation::run_generation(
        &mut output,
        crate::runtime::generation::GenerationLimits { prompt_tokens: token_ids.len(), max_tokens: decode_steps, max_sequence_length: max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |output, _| Ok::<_, Box<dyn std::error::Error>>(output.token_id),
        |token, _| crate::runtime::generation::write_token(&detokenizer, token, true),
        |output, token, position, _| {
            let embedding = weights.embedding_rows_f32(&[token])?;
            let hidden = CpuTensor { data: embedding, rows: 1, cols: cfg.hidden_size };
            let hidden = runtime.at(&mut cache, &mut delta_state, position).decode(hidden).map_err(|error| format!("Qwen3.6 CPU decode position={position}: {error:?}"))?;
            *output = qwen36_token_output(&backend, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3.6 output position={position}: {error:?}"))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    eprintln!("[qwen36] generated={} decode_rounds={} wall={:.3}s throughput={:.3} tok/s", stats.generated_tokens, stats.decode_rounds, stats.elapsed.as_secs_f64(), stats.decode_tokens_per_second());
    Ok(())
}
