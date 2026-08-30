//! Qwen3.6 × ROCm standalone 组合。

use std::time::Instant;

use crate::{
    attention::{
        gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetState},
        hybrid::HybridAttentionOptions,
        rope::RopeTable,
    },
    backend::{
        Backend,
        rocm::{RocmContext, RocmGatedDeltaNetStorage, RocmKvCache},
    },
    config::Qwen36StandaloneModelConfig,
    runtime::qwen36::{self, Qwen36Config},
    weight::{format::mlx_affine::MlxAffineSource, model::qwen36::Qwen36Weights},
};

pub fn run(backend: &RocmContext, model: Qwen36StandaloneModelConfig) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Qwen36Config::standard_27b();
    let weights = Qwen36Weights::new(MlxAffineSource::open(&model.weights_directory)?, cfg.clone())?;

    // Qwen3.6 权重没有内置 tokenizer,读目录下 tokenizer.json。
    let tokenizer_path = model.weights_directory.join("tokenizer.json");
    let tokens = if tokenizer_path.exists() {
        let tokenizer = crate::tokenizer::Tokenizer::new(&tokenizer_path)?;
        tokenizer.tokenize(crate::runtime::qwen36::chat_prompt(&model.generation.prompt).as_bytes())
    } else {
        return Err(format!("Qwen3.6 需要 tokenizer.json 在权重目录下: {}", model.weights_directory.display()).into());
    };
    if tokens.is_empty() || tokens.len() > model.generation.max_sequence_length {
        return Err(format!("Qwen3.6 prompt tokens={}，max_seq_len={}", tokens.len(), model.generation.max_sequence_length).into());
    }

    let hidden = backend.tensor_from_f32(weights.embedding_rows_f32(&tokens)?, tokens.len(), cfg.hidden_size)?;

    let layers = qwen36::prepare_qwen36_layers(backend, &weights).map_err(|error| format!("准备 Qwen3.6 ROCm 层: {error:?}"))?;
    let output_head = qwen36::prepare_qwen36_output_head_quantized(backend, &weights, model.lm_head_quantization).map_err(|error| format!("准备 Qwen3.6 output: {error:?}"))?;
    let rope = RopeTable::precompute(model.generation.max_sequence_length, cfg.rope_dim, cfg.rope_theta);
    let runtime = qwen36::Qwen36Runtime::new(backend, &cfg, &layers, &rope, HybridAttentionOptions { precise_prefill: model.execution.precise_gqa_prefill });
    let mut cache = RocmKvCache::new(cfg.num_layers);
    let mut delta_state =
        GatedDeltaNetState::<RocmGatedDeltaNetStorage>::with_head_layout(cfg.num_layers, cfg.gated_delta_net_spec(), GatedDeltaNetHeadLayout::Grouped).map_err(|error| format!("创建 Qwen3.6 ROCm DeltaNet state: {error:?}"))?;

    let prefill_started = Instant::now();
    let hidden = runtime.at(&mut cache, &mut delta_state, 0).prefill(hidden).map_err(|error| format!("Qwen3.6 ROCm prefill: {error:?}"))?;
    eprintln!("[qwen36-prefill] tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64());

    let mut hidden = backend.select_row(&hidden, tokens.len() - 1).map_err(|error| format!("选择 Qwen3.6 prefill 末行: {error:?}"))?;
    if model.generation.decode_steps == 0 {
        return Ok(());
    }

    // 简单 detokenizer:从目录加载。
    let detokenizer_path = model.weights_directory.join("tokenizer.json");
    let detokenizer = crate::tokenizer::Detokenizer::load(&detokenizer_path)?;
    crate::runtime::generation::run_generation(
        &mut hidden,
        crate::runtime::generation::GenerationLimits { prompt_tokens: tokens.len(), max_tokens: model.generation.decode_steps, max_sequence_length: model.generation.max_sequence_length, eos_tokens: &cfg.eos_token_ids },
        |hidden, _| Ok::<_, Box<dyn std::error::Error>>(qwen36::qwen36_token_output(backend, &cfg, &output_head, hidden).map_err(|error| format!("Qwen3.6 output: {error:?}"))?.token_id),
        |token, _| crate::runtime::generation::write_token(&detokenizer, token, true),
        |hidden, token, position, _| {
            let input = backend.tensor_from_f32(weights.embedding_rows_f32(&[token])?, 1, cfg.hidden_size)?;
            *hidden = runtime.at(&mut cache, &mut delta_state, position).decode(input).map_err(|error| format!("Qwen3.6 ROCm decode position={position}: {error:?}"))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    Ok(())
}
