//! 经典 dense Qwen3 × CUDA 组合(nvidia/Qwen3-14B-NVFP4 文本入口)。
//!
//! 用法(tools 入口 `qwen3_cuda`):`qwen3_cuda MODEL_DIR [--prompt TEXT] [--max-seq-len N]
//! [--decode-steps N] [--lm-head-q8g128]`。
//!
//! NVFP4 矩阵保持 packed u8 + E4M3 block scale 常驻,decode 走 CUDA 原生即时反量化
//! kernel(见 `kernel::cuda::nvfp4`);embedding 走 host 按行上传(不占显存)。
//! 纯文本三轴等位的 M-RoPE 即标准 RoPE,直接复用 `qwen3_vl_mrope_table`。

use std::path::Path;

use crate::backend::cuda::CudaContext;

pub fn run(ctx: &CudaContext, model_dir: &Path, prompt: &str, max_seq_len: usize, decode_steps: usize, lm_head_quantization: crate::weight::LmHeadQuantization, kv_q8g64: bool) -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        backend::cuda::CudaKvCache,
        runtime::qwen3_vl::{
            Qwen3VlConfig, chat_prompt, prepare_qwen3_vl_output_head_quantized, prepare_qwen3_vl_text_layer, qwen3_vl_decode_rope_table, qwen3_vl_decode_round_resident, qwen3_vl_last_token_output, qwen3_vl_mrope_table,
            qwen3_vl_text_prefill, qwen3_vl_token_output,
        },
        weight::model::qwen3::Qwen3Weights,
    };
    use std::time::Instant;

    if max_seq_len == 0 {
        return Err("--max-seq-len 必须 > 0".into());
    }
    let cfg = Qwen3VlConfig::dense_14b();
    if max_seq_len > cfg.max_position_embeddings {
        return Err(format!("max_seq_len={max_seq_len} 超过模型上限 {}", cfg.max_position_embeddings).into());
    }

    let weights = Qwen3Weights::open(model_dir, cfg.clone()).map_err(|error| format!("加载 Qwen3-14B NVFP4 权重: {error}"))?;
    let (tokenizer, detokenizer) = crate::tokenizer::load_bpe_directory(model_dir)?;
    let prompt_text = chat_prompt(prompt);
    let token_ids = tokenizer.tokenize(prompt_text.as_bytes());
    if token_ids.is_empty() {
        return Err("Qwen3-14B prompt tokenize 为空".into());
    }
    if token_ids.len() > max_seq_len {
        return Err(format!("prompt tokens={} 超过 max_seq_len={max_seq_len}", token_ids.len()).into());
    }
    eprintln!("[qwen3-cuda] model={} prompt_tokens={} max_seq_len={max_seq_len} decode_steps={decode_steps} NVFP4 lm_head={lm_head_quantization:?}", model_dir.display(), token_ids.len());

    let prepare_started = Instant::now();
    let mut resident = Vec::with_capacity(cfg.layer_count);
    for layer in 0..cfg.layer_count {
        resident.push(prepare_qwen3_vl_text_layer(ctx, &weights, layer).map_err(|error| format!("准备 Qwen3-14B L{layer}: {error:?}"))?);
    }
    let output_head = prepare_qwen3_vl_output_head_quantized(ctx, &cfg, &weights, lm_head_quantization).map_err(|error| format!("准备 Qwen3-14B output head: {error:?}"))?;
    ctx.synchronize()?;
    eprintln!("[qwen3-cuda] resident_layers={} preload={:.3}s", resident.len(), prepare_started.elapsed().as_secs_f64());

    // 用已知的 prompt token 数预热点全部 prefill GEMM 形状:cuBLAS 对每个 unique
    // (n, m, k) 首次调用的 heuristic ~20-160ms,不预热会计入 prefill 计时(长驻服务
    // 形状复用无此成本,单发入口把开销移到 prepare 阶段)。lm_head 的 GEMV 是 m=1。
    let rows = token_ids.len();
    for (n, k) in [(cfg.num_heads * cfg.head_dim, cfg.hidden_size), (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size), (cfg.intermediate_size, cfg.hidden_size), (cfg.hidden_size, cfg.intermediate_size)] {
        crate::kernel::cuda::linear::prewarm_hgemm(ctx, n, rows, k, false);
    }
    crate::kernel::cuda::linear::prewarm_hgemm(ctx, cfg.vocab_size, 1, cfg.hidden_size, false);

    let kv_columns = cfg.num_kv_heads * cfg.head_dim;
    let mut cache = if kv_q8g64 { CudaKvCache::new_q8g64(cfg.layer_count, max_seq_len, kv_columns).map_err(|error| format!("Q8G64 KV cache: {error}"))? } else { CudaKvCache::new(cfg.layer_count, max_seq_len, kv_columns) };
    eprintln!("[qwen3-cuda] kv_cache format={} columns={kv_columns}", cache.format());

    let embedding = weights.embedding_rows_f32(&token_ids)?;
    let hidden = ctx.tensor_from_f32(&embedding, token_ids.len(), cfg.hidden_size).map_err(|error| format!("上传 Qwen3-14B embedding: {error}"))?;

    // 纯文本三轴等位:一次构建 [0, n) 的标准 RoPE 表供 prefill 使用。
    let positions = (0..token_ids.len()).collect::<Vec<_>>();
    let rope = qwen3_vl_mrope_table(&cfg, &[positions.clone(), positions.clone(), positions])?;

    let prefill_started = Instant::now();
    let hidden = qwen3_vl_text_prefill(ctx, &cfg, &weights, &resident, &mut cache, hidden, &rope, 0).map_err(|error| format!("Qwen3-14B CUDA prefill: {error:?}"))?;
    ctx.synchronize()?;
    eprintln!("[qwen3-cuda] prefill wall={:.3}s kv_cache={:.2} MiB", prefill_started.elapsed().as_secs_f64(), cache.allocated_bytes() as f64 / 1048576.0);

    let mut output = qwen3_vl_last_token_output(ctx, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3-14B CUDA first token: {error:?}"))?;
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
            let hidden = ctx.tensor_from_f32(&embedding, 1, cfg.hidden_size).map_err(|error| format!("上传 Qwen3-14B decode embedding: {error}"))?;
            // rope_delta=0:纯文本位置即序列位置,单行 decode 表三轴同值。
            let rope = qwen3_vl_decode_rope_table(&cfg, position, 0)?;
            let hidden = qwen3_vl_decode_round_resident(ctx, &cfg, &weights, &resident, &mut cache, hidden, &rope, position).map_err(|error| format!("Qwen3-14B CUDA decode position={position}: {error:?}"))?;
            *output = qwen3_vl_token_output(ctx, &cfg, &output_head, &hidden).map_err(|error| format!("Qwen3-14B CUDA output position={position}: {error:?}"))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    let answer = detokenizer.decode_bytes(&generated, true)?;
    println!("{}", String::from_utf8_lossy(&answer));
    eprintln!("[qwen3-cuda] generated={} decode_rounds={} wall={:.3}s throughput={:.3} tok/s", stats.generated_tokens, stats.decode_rounds, stats.elapsed.as_secs_f64(), stats.decode_tokens_per_second());
    Ok(())
}
