//! Laguna × CPU reference 组合(CUDA 的 oracle)。

use crate::{
    backend::cpu::{CpuContext, CpuKvCache, CpuPrefillExperts},
    moe::{
        UncachedMoeState,
        expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
    },
    runtime::{
        expert_pipeline::ExpertDecodePipeline,
        laguna::{self, LagunaGguf, LagunaRopeTables, LagunaRuntimeOptions},
    },
    weight::expert_source::GgufExpertSource,
};
use std::{sync::Arc, time::Instant};

pub fn run(model_dir: &std::path::Path, prompt: &str, max_seq_len: usize, decode_steps: usize, runtime_options: LagunaRuntimeOptions) -> Result<(), Box<dyn std::error::Error>> {
    let weights = Arc::new(LagunaGguf::open(model_dir)?);
    let cfg = weights.config().clone();
    laguna::ensure_supported(&cfg).map_err(|error| format!("Laguna runtime 尚不支持: {error}"))?;
    let tokenizer = weights.tokenizer()?;
    let tokens = tokenizer.tokenize(prompt.as_bytes());
    if tokens.is_empty() || tokens.len() > max_seq_len {
        return Err(format!("Laguna prompt tokens={}，max_seq_len={max_seq_len}", tokens.len()).into());
    }

    let backend = CpuContext;
    let prepare_started = Instant::now();
    let layers = laguna::prepare_laguna_layers(&backend, weights.as_ref()).map_err(|error| format!("准备 Laguna CPU 层: {error:?}"))?;
    let output_head = laguna::prepare_laguna_output_quantized(&backend, weights.as_ref(), crate::weight::LmHeadQuantization::Native).map_err(|error| format!("准备 Laguna output: {error:?}"))?;
    eprintln!("[laguna-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());

    let rope = LagunaRopeTables::new(&cfg, max_seq_len)?;
    let runtime = laguna::LagunaRuntime::new(&backend, &cfg, &layers, &rope, runtime_options);
    // CPU KV 为 dense 引用实现;滑窗语义由 spec.window 在注意力可见区间内保证。
    let mut cache = CpuKvCache::new(cfg.layer_count);
    let expert_source: Arc<dyn GgufExpertSource> = weights.clone();
    let mut prefill_experts = CpuPrefillExperts::gguf(expert_source);
    let prefill_started = Instant::now();
    let chunk_size = cfg.sliding_window.min(512);
    let mut hidden = None;
    for (index, chunk) in tokens.chunks(chunk_size).enumerate() {
        let input = crate::kernel::cpu::CpuTensor { data: weights.embedding_rows(chunk)?, rows: chunk.len(), cols: cfg.hidden_size };
        let output = runtime.prefill(&mut cache, &mut prefill_experts, input, index * chunk_size).map_err(|error| format!("Laguna CPU prefill: {error:?}"))?;
        hidden = Some(output);
    }
    let hidden = hidden.expect("tokens 非空已校验");
    eprintln!("[laguna-prefill] tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64());
    let mut hidden = crate::kernel::cpu::CpuTensor { data: hidden.row(tokens.len() - 1).to_vec(), rows: 1, cols: cfg.hidden_size };
    if decode_steps == 0 {
        return Ok(());
    }

    let mut expert_state = ExpertDecodePipeline::new(
        UncachedMoeState::default(),
        ExpertPredictorConfig { first_layer: cfg.leading_dense_layer_count, layer_count: cfg.layer_count, expert_count: cfg.num_experts, routed_top_k: cfg.num_experts_per_tok, prefetch_count: 0, weights: ExpertPredictorWeights::default() },
    )?;
    let detokenizer = weights.detokenizer()?;
    let stats = crate::runtime::generation::run_generation(
        &mut hidden,
        crate::runtime::generation::GenerationLimits { prompt_tokens: tokens.len(), max_tokens: decode_steps, max_sequence_length: max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |hidden, _| {
            let output = laguna::laguna_token_output(&backend, &cfg, &output_head, hidden).map_err(|error| format!("Laguna output: {error:?}"))?;
            Ok(output.token_id)
        },
        |token, _| {
            eprintln!("[laguna-cpu-token] id={token}");
            crate::runtime::generation::write_token(&detokenizer, token, true)
        },
        |hidden, token, position, _| {
            let input = crate::kernel::cpu::CpuTensor { data: weights.embedding_rows(&[token])?, rows: 1, cols: cfg.hidden_size };
            *hidden = runtime.decode(weights.as_ref(), &mut expert_state, &mut cache, input, position).map_err(|error| format!("Laguna CPU decode position={position}: {error:?}"))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    eprintln!("[laguna-summary] prompt_tokens={} generated_tokens={} generation_wall={:.3}s tok_per_s={:.3}", tokens.len(), stats.generated_tokens, stats.elapsed.as_secs_f64(), stats.tokens_per_second(),);
    Ok(())
}

#[cfg(test)]
mod real_weight_tests {
    use super::*;

    /// 真机 CPU oracle:与 CUDA [laguna-token] 逐 id 对比。手工运行:
    /// `LAGUNA_GGUF=...shard1.gguf cargo test --release --lib laguna::cpu -- --ignored --nocapture`
    #[test]
    #[ignore = "需要真机 GGUF 权重,CI 不可依赖"]
    fn cpu_oracle_renders_same_prompt_as_server() {
        let path = std::path::PathBuf::from(std::env::var("LAGUNA_GGUF").expect("LAGUNA_GGUF 指向 shard1 路径"));
        let reader = crate::weight::container::gguf::GgufReader::open(&path).expect("打开 GGUF");
        let template = reader.metadata("tokenizer.chat_template").and_then(crate::weight::container::gguf::GgufValue::as_str).expect("chat template");
        let mut compiled = crate::runtime::chat_template::ChatTemplate::new(template).expect("编译模板");
        let prompt = compiled.render(&serde_json::json!({"messages": [{"role": "user", "content": "Write a Rust function that reverses a string."}], "enable_thinking": false})).expect("渲染 prompt");
        run(&path, &prompt, 4096, 10, LagunaRuntimeOptions::default()).expect("CPU 前向");
    }
}
