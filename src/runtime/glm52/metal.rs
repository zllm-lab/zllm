//! GLM-5.2 × Metal standalone 组合。

use crate::{
    attention::rope::RopeTable,
    backend::{
        BackendResources, LinearWeight,
        metal::{MetalContext, MetalGpuProfile, MetalKvCache, MetalPrefillExperts, MetalWeight, merge_gpu_profiles, report_metal_resource_plan},
    },
    config::{Glm52StandaloneModelConfig, KvCacheFormat, MetalBackendConfig},
    kernel::cpu::CpuTensor,
    runtime::Model,
    tokenizer::Tokenizer,
};
use std::path::{Path, PathBuf};

use super::{
    Glm52, Glm52Config, Glm52DenseDecodeLayer, Glm52MoeDecodeLayer, Glm52PrefillLayerKind, glm52_decode_layer, glm52_dense_decode_layer, glm52_dense_prefill_layer, glm52_moe_prefill_layer, glm52_prefill, glm52_token_output,
    prepare_dense_prefill_layer, prepare_dense_prefill_layer_nvfp4, prepare_glm52_output_head_quantized, prepare_moe_prefill_layer, prepare_moe_prefill_layer_nvfp4, print_token,
};

pub fn run(model: Glm52StandaloneModelConfig, backend: MetalBackendConfig) -> Result<(), Box<dyn std::error::Error>> {
    let execution = model.execution;
    let lm_head_quantization = model.lm_head_quantization;
    let model_dir = model.weights_directory;
    let prompt = model.generation.prompt;
    let max_seq_len = model.generation.max_sequence_length;
    let decode_steps = model.generation.decode_steps;
    let kv_cache_dump = model.kv_cache_dump;
    let kv_cache_load = model.kv_cache_load;
    let nvfp4_root = model.nvfp4_directory;
    let ct_root = model.compressed_tensors_directory;
    let gguf_root = model.gguf_directory;
    if gguf_root.is_some() {
        return Err("GLM-5.2 Metal 入口暂不支持 GGUF 权重，请使用 ROCm 入口".into());
    }
    println!("backend: Metal");
    if let (Some(dump), Some(load)) = (&kv_cache_dump, &kv_cache_load) {
        if dump != load {
            return Err("续跑时 --kv-cache-dump 与 --kv-cache-load 必须指向同一目录".into());
        }
    }

    let ctx_owner = MetalContext::new_default_with_replay(backend.replay).map_err(|e| format!("MetalContext 初始化失败: {e}"))?;
    let ctx = &ctx_owner;

    let model_dir = PathBuf::from(&model_dir);
    let quant_root = ct_root.as_ref().or(nvfp4_root.as_ref());
    let tok_path = super::tokenizer_path(&model_dir, ct_root.as_deref(), nvfp4_root.as_deref());
    let expert_key = if let Some(root) = ct_root.as_ref() { format!("ct-int4-int8:{}", root.display()) } else { nvfp4_root.as_ref().map_or_else(|| "official-fp8".to_owned(), |root| format!("nvfp4:{}", root.display())) };
    let core_path = quant_root.unwrap_or(&model_dir);
    let model_key = format!("{}|{expert_key}|rope=interleaved", std::fs::canonicalize(core_path).unwrap_or_else(|_| core_path.clone()).display(),);
    let experts_dir = model_dir.join("experts");

    println!("加载 tokenizer: {}", tok_path.display());
    let tokenizer = Tokenizer::new(&tok_path)?;

    let token_ids = tokenizer.tokenize(prompt.as_bytes());
    let n_tokens = token_ids.len();
    if n_tokens == 0 {
        return Err("prompt 不能为空".into());
    }
    println!("prompt: {n_tokens} tokens, {} bytes", prompt.len());
    if n_tokens <= 64 {
        println!("prompt text: {prompt:?}");
        println!("token ids: {token_ids:?}");
    } else {
        println!("token ids: first={:?} last={:?}", &token_ids[..8], &token_ids[n_tokens - 8..]);
    }
    if n_tokens > max_seq_len {
        return Err(format!("prompt {n_tokens} 超过 max_seq_len {max_seq_len}(用 --max-seq-len 调整)").into());
    }

    let cfg = Glm52Config::standard();
    let weights = super::open_weights(&cfg, &model_dir, ct_root.as_deref(), nvfp4_root.as_deref(), gguf_root.as_deref())?;
    let nvfp4_source = weights.nvfp4_experts().map(|source| source.with_archive_dir(&experts_dir));
    let decode_expert_sources = super::decode_expert_sources(&weights, nvfp4_source.as_ref(), &model_dir, &cfg)?;
    let model = Glm52::standard();
    let mla = match &model.layer_spec(0)?.attention {
        crate::attention::AttentionSpec::Mla(m) => m.clone(),
        _ => return Err("dense 层应为 MLA".into()),
    };

    let layer_end = cfg.layer_count - 1;
    let execution_layer_count = cfg.layer_count;

    let n = token_ids.len();
    let loading_prefill = kv_cache_load.is_some();
    let mut hidden = if loading_prefill {
        CpuTensor { data: Vec::new(), rows: 0, cols: cfg.hidden_size }
    } else {
        let embedding = weights.embedding_rows(&token_ids).map_err(|e| format!("embedding: {e}"))?;
        CpuTensor { data: embedding, rows: n, cols: cfg.hidden_size }
    };
    let mut metal_hidden = if loading_prefill { None } else { Some(ctx.tensor_from_f32(&hidden.data, hidden.rows, hidden.cols).map_err(|error| format!("上传 embedding: {error}"))?) };

    let rope_table = RopeTable::precompute(max_seq_len, mla.qk_rope_head_dim, mla.rope_theta);

    // device-resident 路径独享:跨层共享的 MLA KV cache。
    // capacity = max_seq_len,prefill 占 [0, n_tokens)、decode 续写不超过此上限。
    // 长上下文(如 1M)时占用 = layer_count × max_seq_len × ~656B,可能超 GPU 内存 ——
    // 那种场景需要 oscar 2bit 压缩或 sparse attention(后续范围)。
    let spec = crate::kv_cache::KvCacheSpec::from_attention(&model.layer_spec(0)?.attention).map_err(|e| format!("KV cache spec: {e}"))?;
    let kv_f16 = execution.kv_cache_format == KvCacheFormat::F16;
    let mut kv_cache = if kv_f16 { MetalKvCache::new_f16(&ctx, spec, execution_layer_count, max_seq_len) } else { MetalKvCache::new(&ctx, spec, execution_layer_count, max_seq_len) }
        .map_err(|e| format!("{} KV cache 分配: {e}", if kv_f16 { "F16" } else { "INT8" }))?;
    println!("{} KV cache: {} 层 × capacity {} token(当前 prompt {}),占用 {:.1} MB", if kv_f16 { "F16" } else { "INT8" }, execution_layer_count, max_seq_len, n_tokens, kv_cache.allocated_bytes() as f64 / (1024.0 * 1024.0));
    let expert_arena_bytes = 0;
    let dsa_layer_count = (0..execution_layer_count).filter(|&layer| crate::runtime::glm52::is_indexer_layer(layer)).count();
    let dsa_cache_bytes = dsa_layer_count
        .saturating_mul(max_seq_len)
        .saturating_mul(cfg.index_head_dim)
        .saturating_mul(std::mem::size_of::<half::f16>())
        .saturating_add(max_seq_len.saturating_mul(std::mem::size_of::<u32>()))
        .saturating_add(cfg.index_top_k.saturating_mul(std::mem::size_of::<u32>()))
        .saturating_add(2 * std::mem::size_of::<u32>());
    report_metal_resource_plan(&ctx, execution_layer_count, max_seq_len, kv_cache.allocated_bytes(), dsa_cache_bytes, expert_arena_bytes)?;
    let mut dsa_state = crate::backend::metal::dsa::MetalDsaState::new(&ctx, execution_layer_count, max_seq_len, cfg.index_head_dim, cfg.index_top_k).map_err(|e| format!("DSA state: {e}"))?;

    let prefill_started = std::time::Instant::now();
    let mut layer_stats = Vec::new();
    let mut total_gpu_seconds = 0.0;
    let mut total_command_buffers = 0u64;
    let mut prefill_routes = crate::moe::expert_predictor::ExpertRouteTrace::new(cfg.dense_layer_count, cfg.layer_count - cfg.dense_layer_count, cfg.expert_count, cfg.expert_top_k)?;
    let mut total_gpu_operators: Vec<MetalGpuProfile> = Vec::new();
    let mut prefill_start_layer = 0usize;
    if let Some(dir) = kv_cache_load.as_deref() {
        let load_started = std::time::Instant::now();
        let (loaded_hidden, completed_layer) = load_prefill_state(dir, &mut kv_cache, layer_end, &token_ids, cfg.hidden_size, &model_key)?;
        hidden = loaded_hidden;
        prefill_start_layer = completed_layer + 1;
        metal_hidden = Some(ctx.tensor_from_f32(&hidden.data, hidden.rows, hidden.cols).map_err(|error| format!("上传 checkpoint hidden: {error}"))?);
        let indexer_layers: Vec<usize> = (0..=completed_layer).filter(|&layer| crate::runtime::glm52::is_indexer_layer(layer)).collect();
        let loaded = dsa_state.load_checkpoint(&ctx, dir, n_tokens, &indexer_layers, mla.qk_rope_head_dim, mla.rope_theta).map_err(|e| format!("DSA checkpoint: {e}"))?;
        if loaded != indexer_layers.len() {
            eprintln!("DSA indexer cache 加载 {loaded}/{} 层，缺失层保持 dense attention fallback", indexer_layers.len());
        }
        let route_path = dir.join("expert_routes.bin");
        if route_path.is_file() {
            prefill_routes = crate::moe::expert_predictor::ExpertRouteTrace::from_bytes(&std::fs::read(&route_path)?)?;
        }
        println!("Prefill checkpoint: 已完成 L{completed_layer}，从 L{prefill_start_layer} 继续，加载耗时 {:.3}s", load_started.elapsed().as_secs_f64());
    }
    let mut prefill_experts = ();
    if prefill_start_layer < execution_layer_count {
        let initial_hidden = metal_hidden.take().ok_or("Metal prefill 缺少初始 hidden")?;
        metal_hidden = Some(
            glm52_prefill(
                ctx,
                &cfg,
                n_tokens,
                prefill_start_layer,
                initial_hidden,
                &mut prefill_experts,
                |_, _, _| Ok(()),
                |_, layer, kind, device_hidden| {
                    ctx.reset_gpu_stats();
                    let now = std::time::Instant::now();
                    let output = match kind {
                        Glm52PrefillLayerKind::Dense => {
                            let resident = if weights.source_is_nvfp4() {
                                let w = weights.load_dense_layer_nvfp4(layer).map_err(|msg| crate::backend::BackendError::Compute { msg: format!("L{layer} NVFP4 dense core: {msg}") })?;
                                prepare_dense_prefill_layer_nvfp4(ctx, &cfg, &mla, &w)?
                            } else if weights.source_is_ct() {
                                let w = weights.load_dense_layer_ct(layer).map_err(|msg| crate::backend::BackendError::Compute { msg: format!("L{layer} CT dense core: {msg}") })?;
                                crate::runtime::glm52::prepare_dense_prefill_layer_ct(ctx, &cfg, &mla, &w, false)?
                            } else {
                                let w = weights.load_dense_layer(layer).map_err(|msg| crate::backend::BackendError::Compute { msg: format!("L{layer} dense 权重: {msg}") })?;
                                prepare_dense_prefill_layer(ctx, &cfg, &mla, &w)?
                            };
                            glm52_dense_prefill_layer(ctx, &cfg, &mla, &resident, layer, None, Some(&mut dsa_state), &device_hidden, &rope_table, Some(&mut kv_cache), 0)?
                        }
                        Glm52PrefillLayerKind::Moe => {
                            let resident = if weights.source_is_nvfp4() {
                                let w = weights.load_moe_layer_nvfp4(layer).map_err(|msg| crate::backend::BackendError::Compute { msg: format!("L{layer} NVFP4 MoE core: {msg}") })?;
                                prepare_moe_prefill_layer_nvfp4(ctx, &cfg, &mla, &w)?
                            } else if weights.source_is_ct() {
                                let w = weights.load_moe_layer_ct(layer).map_err(|msg| crate::backend::BackendError::Compute { msg: format!("L{layer} CT MoE core: {msg}") })?;
                                crate::runtime::glm52::prepare_moe_prefill_layer_ct(ctx, &cfg, &mla, &w, false)?
                            } else {
                                let w = weights.load_moe_layer(layer).map_err(|msg| crate::backend::BackendError::Compute { msg: format!("L{layer} 原始 FP8 MoE core: {msg}") })?;
                                prepare_moe_prefill_layer(ctx, &cfg, &mla, &w)?
                            };
                            let mut experts = if weights.source_is_ct() {
                                println!("layer {layer} experts: compressed-tensors W4A16");
                                MetalPrefillExperts::w4a16(std::sync::Arc::new(weights.ct_source().map_err(crate::backend::BackendError::ExpertLoad)?))
                            } else if let Some(source) = &nvfp4_source {
                                println!("layer {layer} experts: NVFP4 safetensors");
                                MetalPrefillExperts::nvfp4(source.clone())
                            } else {
                                println!("layer {layer} experts: official FP8 safetensors");
                                MetalPrefillExperts::fp8(&model_dir, cfg.expert_intermediate_size, cfg.hidden_size, cfg.expert_count).map_err(crate::backend::BackendError::ExpertLoad)?
                            };
                            glm52_moe_prefill_layer(ctx, &cfg, &mla, &resident, layer, &mut experts, Some(&mut prefill_routes), Some(&mut dsa_state), &device_hidden, &rope_table, Some(&mut kv_cache), 0)?
                        }
                    };
                    let wall_seconds = now.elapsed().as_secs_f64();
                    let gpu_stats = ctx.gpu_stats();
                    let (gpu_seconds, command_buffers, gpu_operators) = (gpu_stats.seconds, gpu_stats.command_buffers, ctx.gpu_profile());
                    merge_gpu_profiles(&mut total_gpu_operators, &gpu_operators);
                    total_gpu_seconds += gpu_seconds;
                    total_command_buffers += command_buffers;
                    println!("layer {layer}: wall={wall_seconds:.3}s gpu={gpu_seconds:.3}s commands={command_buffers} | device=[{},{}]", output.rows, output.cols);
                    for profile in &gpu_operators {
                        println!("  gpu {:>8.3} ms x{} {:>12} read {:>12} write | {} {}", profile.gpu_seconds * 1.0e3, profile.calls, profile.estimated_read_bytes, profile.estimated_write_bytes, profile.operator, profile.shape);
                    }
                    layer_stats.push(serde_json::json!({ "layer": layer, "wall_seconds": wall_seconds, "gpu_seconds": gpu_seconds, "command_buffers": command_buffers, "gpu_operators": gpu_operators }));
                    if let Some(dir) = kv_cache_dump.as_deref() {
                        let checkpoint_started = std::time::Instant::now();
                        let checkpoint_hidden = ctx.tensor_to_f32(&output);
                        dump_prefill_layer_state(dir, &kv_cache, &dsa_state, &prefill_routes, layer, &token_ids, &checkpoint_hidden, output.rows, cfg.hidden_size, &model_key)
                            .map_err(|error| crate::backend::BackendError::Compute { msg: format!("L{layer} checkpoint: {error}") })?;
                        println!("layer {layer} checkpoint: {:.3}s", checkpoint_started.elapsed().as_secs_f64());
                    }
                    Ok(output)
                },
            )
            .map_err(|error| format!("Metal full-token prefill: {error:?}"))?,
        );
    }

    if let Some(device_hidden) = metal_hidden.as_ref() {
        hidden = CpuTensor { data: ctx.tensor_row_to_f32(device_hidden, device_hidden.rows - 1).map_err(|error| format!("下载 prefill 最后一行: {error}"))?, rows: 1, cols: device_hidden.cols };
    }

    let elapsed_seconds = prefill_started.elapsed().as_secs_f64();
    println!("prefill: wall={elapsed_seconds:.3}s gpu={total_gpu_seconds:.3}s commands={total_command_buffers}");

    if let Some(dir) = kv_cache_dump.as_deref() {
        println!("Prefill checkpoint 完成到 L{layer_end}: {}", dir.display());
    }

    if decode_steps > 0 {
        let default_prefetch_count = if weights.source_is_ct() { 0 } else { crate::runtime::DEFAULT_DECODE_PREFETCH_COUNT };
        let decode_prefetch_count = backend.decode_prefetch_count.filter(|&count| count <= cfg.expert_count).unwrap_or(default_prefetch_count);
        println!("decode expert prefetch count: {decode_prefetch_count}");
        let final_norm_w = weights.final_norm()?;
        let mut decode_cfg = cfg.clone();
        decode_cfg.layer_count = execution_layer_count;
        let mut decode_state = if execution_layer_count > cfg.dense_layer_count {
            let backend_state = crate::backend::metal::MetalMoeDecodeState::new()?;
            Some(crate::runtime::expert_pipeline::ExpertDecodePipeline::new(
                backend_state,
                crate::moe::expert_predictor::ExpertPredictorConfig {
                    first_layer: cfg.dense_layer_count,
                    layer_count: execution_layer_count - cfg.dense_layer_count,
                    expert_count: cfg.expert_count,
                    routed_top_k: cfg.expert_top_k,
                    prefetch_count: decode_prefetch_count,
                    weights: crate::moe::expert_predictor::ExpertPredictorWeights { request_frequency: 0.0, temporal_transition: 0.01, spatial_transition: 1.0, future_router: 0.0 },
                },
            )?)
        } else {
            None
        };
        if let Some(decode_state) = decode_state.as_mut() {
            if let Some(dir) = kv_cache_load.as_deref() {
                let route_path = dir.join("expert_routes.bin");
                match std::fs::read(&route_path) {
                    Ok(bytes) => {
                        let routes = crate::moe::expert_predictor::ExpertRouteTrace::from_bytes(&bytes)?;
                        decode_state.train_prefill_routes(&routes)?;
                        println!("expert predictor: 已加载 prefill 路由历史 {} ({:.1} MiB)", route_path.display(), bytes.len() as f64 / (1024.0 * 1024.0));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        eprintln!("expert predictor: checkpoint 无 expert_routes.bin，decode 将冷启动");
                    }
                    Err(error) => {
                        return Err(format!("读取 prefill expert route trace {}: {error}", route_path.display()).into());
                    }
                }
            } else if prefill_routes.is_complete() {
                decode_state.train_prefill_routes(&prefill_routes)?;
            }
        }
        enum MetalDecodeLayerWeights {
            Dense(Glm52DenseDecodeLayer<MetalWeight>),
            Moe(Glm52MoeDecodeLayer<MetalWeight>),
        }
        // CT core 全量常驻会在 24 GiB UMA 机器上触发压缩与 swap。前缀层使用
        // 连续 resident slot，其余层流式加载；没有 LRU，也不改变模型执行顺序。
        let ct_resident_layer_count = if weights.source_is_ct() {
            backend
                .decode_resident_layers
                .unwrap_or_else(|| {
                    const GIB: u64 = 1024 * 1024 * 1024;
                    match ctx.device.recommended_max_working_set_size() {
                        bytes if bytes >= 32 * GIB => layer_end + 1,
                        bytes if bytes >= 16 * GIB => 40,
                        _ => 0,
                    }
                })
                .min(layer_end + 1)
        } else {
            layer_end + 1
        };
        let stream_ct_decode_core = weights.source_is_ct() && ct_resident_layer_count < layer_end + 1;
        let prefetch_ct_decode_core = stream_ct_decode_core && backend.prefetch_decode_core;
        let mut metal_decode_layer_weights = Vec::new();
        let started = std::time::Instant::now();
        if decode_steps > 1 {
            for layer in 0..=layer_end {
                if layer >= ct_resident_layer_count {
                    break;
                }
                if layer < cfg.dense_layer_count {
                    let resident = if weights.source_is_nvfp4() {
                        let weight = weights.load_dense_layer_nvfp4(layer).map_err(|e| format!("decode layer {layer} NVFP4 core: {e}"))?;
                        prepare_dense_prefill_layer_nvfp4(ctx, &cfg, &mla, &weight)
                    } else if weights.source_is_ct() {
                        let weight = weights.load_dense_layer_ct(layer).map_err(|e| format!("decode layer {layer} CT dense core: {e}"))?;
                        crate::runtime::glm52::prepare_dense_prefill_layer_ct(ctx, &cfg, &mla, &weight, true)
                    } else {
                        let weight = weights.load_dense_layer(layer).map_err(|e| format!("decode layer {layer} 原始 dense 权重: {e}"))?;
                        prepare_dense_prefill_layer(ctx, &cfg, &mla, &weight)
                    }
                    .map_err(|e| format!("decode layer {layer} resident: {e:?}"))?;
                    metal_decode_layer_weights.push(MetalDecodeLayerWeights::Dense(resident));
                } else {
                    let resident = if weights.source_is_nvfp4() {
                        let weight = weights.load_moe_layer_nvfp4(layer).map_err(|e| format!("decode layer {layer} NVFP4 core: {e}"))?;
                        prepare_moe_prefill_layer_nvfp4(ctx, &cfg, &mla, &weight)
                    } else if weights.source_is_ct() {
                        let weight = weights.load_moe_layer_ct(layer).map_err(|e| format!("decode layer {layer} CT MoE core: {e}"))?;
                        crate::runtime::glm52::prepare_moe_prefill_layer_ct(ctx, &cfg, &mla, &weight, true)
                    } else {
                        let weight = weights.load_moe_layer(layer).map_err(|e| format!("decode layer {layer} 原始 FP8 core: {e}"))?;
                        prepare_moe_prefill_layer(ctx, &cfg, &mla, &weight)
                    }
                    .map_err(|e| format!("decode layer {layer} resident: {e:?}"))?;
                    metal_decode_layer_weights.push(MetalDecodeLayerWeights::Moe(resident));
                }
            }
        }
        let loaded_layers = metal_decode_layer_weights.len();
        eprintln!(
            "[decode weight preload] seconds={:.3} layers={loaded_layers} mode={}",
            started.elapsed().as_secs_f64(),
            if stream_ct_decode_core && loaded_layers > 0 {
                "hybrid"
            } else if stream_ct_decode_core {
                "stream"
            } else {
                "resident"
            },
        );
        let detokenizer = crate::tokenizer::Detokenizer::load(&tok_path)?;
        {
            let output_head = if weights.source_is_nvfp4() {
                let lm_head = weights.lm_head_nvfp4().map_err(|e| format!("lm_head NVFP4 core: {e}"))?;
                prepare_glm52_output_head_quantized(ctx, &cfg, &final_norm_w, LinearWeight::F16(&lm_head), lm_head_quantization)
            } else if weights.source_is_ct() && !backend.lm_head_f32 {
                let lm_head = weights.lm_head_f16().map_err(|e| format!("lm_head CT F16: {e}"))?;
                prepare_glm52_output_head_quantized(ctx, &cfg, &final_norm_w, LinearWeight::F16(&lm_head), lm_head_quantization)
            } else {
                let lm_head = weights.lm_head().map_err(|e| format!("lm_head 官方权重: {e}"))?;
                prepare_glm52_output_head_quantized(ctx, &cfg, &final_norm_w, LinearWeight::F32(&lm_head), lm_head_quantization)
            }
            .map_err(|e| format!("output head 准备: {e:?}"))?;
            let _ = metal_hidden.take().ok_or("decode 缺少 metal_hidden")?;
            let mut h = ctx.tensor_from_f32(hidden.row(hidden.rows - 1), 1, cfg.hidden_size).map_err(|e| format!("decode 起点 hidden 上传: {e}"))?;
            let dsa_state = &mut dsa_state;
            ctx.reset_gpu_stats();
            let decode_started = std::time::Instant::now();
            let output = glm52_token_output(ctx, &cfg, &output_head, &h).map_err(|e| format!("output head: {e:?}"))?;
            let mut prev_token = output.token_id;
            eprintln!("[decode step 0] token={prev_token}");
            print_decode_profile(ctx, 0, decode_started.elapsed().as_secs_f64());
            print_token(&detokenizer, prev_token, true)?;
            use std::io::Write;
            std::io::stdout().flush().ok();

            for step in 1..decode_steps {
                if cfg.eos_token_ids.contains(&prev_token) {
                    println!("\n[EOS at step {step}]");
                    break;
                }
                let position = n_tokens + step - 1;
                if position >= max_seq_len {
                    println!("\n[达到 max_seq_len {max_seq_len},停止]");
                    break;
                }
                ctx.reset_gpu_stats();
                let decode_started = std::time::Instant::now();
                let embedding_bf16 = weights.embedding_rows_bf16(&[prev_token]).map_err(|e| format!("decode embedding: {e}"))?;
                h = ctx.tensor_from_bf16_row(&embedding_bf16, cfg.hidden_size).map_err(|e| format!("decode emb 上传: {e}"))?;

                // 逐层 decode(dense L0-L2 + MoE L3-L77)。
                let mut decode_weight_load_seconds = 0.0f64;
                std::thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
                    let mut prefetched_core = None;
                    if prefetch_ct_decode_core {
                        let first_layer = ct_resident_layer_count.max(cfg.dense_layer_count);
                        let source = &weights;
                        let (sender, receiver) = std::sync::mpsc::sync_channel(0);
                        scope.spawn(move || {
                            for layer in first_layer..=layer_end {
                                let loaded = source.load_moe_layer_ct(layer).map_err(|error| format!("decode layer {layer} CT MoE core: {error}"));
                                if sender.send((layer, loaded)).is_err() {
                                    break;
                                }
                            }
                        });
                        prefetched_core = Some(receiver);
                    }
                    for layer in 0..=layer_end {
                        let _layer_scope = ctx.layer_scope();
                        if weights.source_is_ct() && layer >= metal_decode_layer_weights.len() {
                            let load_started = std::time::Instant::now();
                            if layer < cfg.dense_layer_count {
                                let weight = weights.load_dense_layer_ct(layer).map_err(|e| format!("decode layer {layer} CT dense core: {e}"))?;
                                let resident = crate::runtime::glm52::prepare_dense_prefill_layer_ct(ctx, &cfg, &mla, &weight, true).map_err(|e| format!("decode layer {layer} CT dense upload: {e:?}"))?;
                                decode_weight_load_seconds += load_started.elapsed().as_secs_f64();
                                h = glm52_dense_decode_layer(ctx, &cfg, &mla, &resident, layer, &mut *dsa_state, &h, &rope_table, &mut kv_cache, position).map_err(|e| format!("decode layer {layer}: {e:?}"))?;
                            } else {
                                let weight = if let Some(receiver) = prefetched_core.as_ref() {
                                    let (loaded_layer, weight) = receiver.recv().map_err(|_| format!("decode layer {layer} CT core 预读线程提前结束"))?;
                                    if loaded_layer != layer {
                                        return Err(format!("decode CT core 顺序不一致: 预读 L{loaded_layer}，执行 L{layer}").into());
                                    }
                                    weight?
                                } else {
                                    weights.load_moe_layer_ct(layer).map_err(|e| format!("decode layer {layer} CT MoE core: {e}"))?
                                };
                                let resident = crate::runtime::glm52::prepare_moe_prefill_layer_ct(ctx, &cfg, &mla, &weight, true).map_err(|e| format!("decode layer {layer} CT MoE upload: {e:?}"))?;
                                decode_weight_load_seconds += load_started.elapsed().as_secs_f64();
                                h = glm52_decode_layer(ctx, &decode_cfg, &mla, &resident, layer, &decode_expert_sources, decode_state.as_mut().expect("MoE decode state 已初始化"), &mut *dsa_state, &h, &rope_table, &mut kv_cache, position)
                                    .map_err(|e| format!("decode layer {layer}: {e:?}"))?;
                            }
                        } else {
                            let layer_weight = metal_decode_layer_weights.get(layer).ok_or_else(|| format!("decode layer {layer} resident 权重缺失"))?;
                            match layer_weight {
                                MetalDecodeLayerWeights::Dense(w) => {
                                    h = glm52_dense_decode_layer(ctx, &cfg, &mla, w, layer, &mut *dsa_state, &h, &rope_table, &mut kv_cache, position).map_err(|e| format!("decode layer {layer}: {e:?}"))?;
                                }
                                MetalDecodeLayerWeights::Moe(w) => {
                                    h = glm52_decode_layer(ctx, &decode_cfg, &mla, w, layer, &decode_expert_sources, decode_state.as_mut().expect("MoE decode state 已初始化"), &mut *dsa_state, &h, &rope_table, &mut kv_cache, position)
                                        .map_err(|e| format!("decode layer {layer}: {e:?}"))?;
                                }
                            }
                        }
                    }
                    Ok(())
                })?;
                let output = glm52_token_output(ctx, &cfg, &output_head, &h).map_err(|e| format!("output head: {e:?}"))?;
                prev_token = output.token_id;
                eprintln!("[decode step {step}] token={prev_token}");
                if let Some(decode_state) = decode_state.as_mut() {
                    let pipeline_stats = decode_state.take_stats();
                    let expert_stats = decode_state.backend_state_mut().take_stats();
                    eprintln!(
                        "[decode expert cache {step}] hits={} misses={} prefetch_requested={} prefetch_hits={} prefetch_elapsed={:.3}s prefetch_wait={:.3}s sync_read={:.3}s sync_wait={:.3}s upload={:.3}s",
                        expert_stats.cache_hits,
                        expert_stats.cache_misses,
                        pipeline_stats.prefetch_requested,
                        expert_stats.prefetch_hits,
                        expert_stats.prefetch_elapsed_seconds,
                        expert_stats.prefetch_wait_seconds,
                        expert_stats.sync_expert_read_seconds,
                        expert_stats.sync_expert_wait_seconds,
                        expert_stats.expert_upload_seconds,
                    );
                }
                eprintln!("[decode weight load {step}] seconds={decode_weight_load_seconds:.3}");
                print_decode_profile(ctx, step, decode_started.elapsed().as_secs_f64());
                print_token(&detokenizer, prev_token, true)?;
                std::io::stdout().flush().ok();
            }
            println!();
        }
    }

    let (mn, mx, mean) = stats(&hidden.data);
    println!("\nL0-L{layer_end} 最后 token 输出: min={mn:.4} max={mx:.4} mean={mean:.4}");
    println!("最后 token 前 8 维:{:?}", &hidden.row(hidden.rows - 1)[..8]);
    if let Some(path) = backend.prefill_report.as_deref() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let report = serde_json::json!({
            "prompt_tokens": n_tokens,
            "prompt_bytes": prompt.len(),
            "last_layer": layer_end,
            "backend": "Metal",
            "mla_backend": "metal",
            "matmul_backend": "simdgroup",
            "wall_seconds": elapsed_seconds,
            "gpu_seconds": total_gpu_seconds,
            "command_buffers": total_command_buffers,
            "gpu_operators": total_gpu_operators,
            "output_min": mn,
            "output_max": mx,
            "output_mean": mean,
            "layers": layer_stats,
        });
        std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
        println!("report={}", path.display());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(serde::Serialize, serde::Deserialize)]
struct PrefillStateManifest {
    version: u32,
    model_key: String,
    cache_format: String,
    last_layer: usize,
    prompt_tokens: usize,
    hidden_size: usize,
    #[serde(default = "default_hidden_rows")]
    hidden_rows: usize,
    token_ids: Vec<u32>,
}

#[cfg(target_os = "macos")]
fn default_hidden_rows() -> usize {
    1
}

#[cfg(target_os = "macos")]
fn dump_prefill_layer_state(
    dir: &Path,
    cache: &MetalKvCache,
    dsa_state: &crate::backend::metal::dsa::MetalDsaState,
    prefill_routes: &crate::moe::expert_predictor::ExpertRouteTrace,
    layer: usize,
    token_ids: &[u32],
    hidden: &[f32],
    hidden_rows: usize,
    hidden_size: usize,
    model_key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if hidden.len() != hidden_rows * hidden_size || hidden_rows != token_ids.len() {
        return Err(format!("dump hidden shape [{},{}]，期望 [{},{}]", hidden_rows, hidden_size, token_ids.len(), hidden_size).into());
    }
    std::fs::create_dir_all(dir)?;
    if cache.layer_len(layer) != token_ids.len() {
        return Err(format!("L{layer} cache 长度 {}，期望 prompt {}", cache.layer_len(layer), token_ids.len()).into());
    }
    let kv_bytes = cache.dump_layer_int8(dir.join(format!("layer{layer:03}.kv")), layer).map_err(|error| format!("dump L{layer}: {error}"))?;
    let dsa_bytes = dsa_state.dump_layer_checkpoint(dir, layer).map_err(|error| format!("dump DSA L{layer}: {error}"))?;
    if layer >= 3 {
        write_atomic(&dir.join("expert_routes.bin"), &prefill_routes.to_bytes()?)?;
    }
    let (hidden_bytes, hidden_max_error, hidden_rmse) = quantize_grouped_i8(hidden, hidden_rows, hidden_size, crate::kv_cache::DEFAULT_GROUP_SIZE)?;
    write_atomic(&dir.join("hidden.i8g64"), &hidden_bytes)?;
    let manifest = PrefillStateManifest { version: 3, model_key: model_key.to_owned(), cache_format: "mla-int8-g64".to_owned(), last_layer: layer, prompt_tokens: token_ids.len(), hidden_size, hidden_rows, token_ids: token_ids.to_vec() };
    write_atomic(&dir.join("manifest.json"), &serde_json::to_vec_pretty(&manifest)?)?;
    println!(
        "checkpoint L{layer}: KV {:.1} MiB, DSA {:.1} MiB, hidden {:.1} MiB INT8(max_err={hidden_max_error:.5},rmse={hidden_rmse:.6})",
        kv_bytes as f64 / (1024.0 * 1024.0),
        dsa_bytes as f64 / (1024.0 * 1024.0),
        hidden_bytes.len() as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn load_prefill_state(dir: &Path, cache: &mut MetalKvCache, last_layer: usize, token_ids: &[u32], hidden_size: usize, model_key: &str) -> Result<(CpuTensor, usize), Box<dyn std::error::Error>> {
    let manifest: PrefillStateManifest = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    if manifest.version != 1 && manifest.version != 2 && manifest.version != 3 {
        return Err(format!("checkpoint version {}，当前只支持 1/2/3", manifest.version).into());
    }
    if manifest.model_key != model_key {
        return Err(format!("checkpoint 模型不匹配: {} != {model_key}", manifest.model_key).into());
    }
    let old_f16 = manifest.version < 3 && manifest.cache_format == "mla-f16";
    let int8_checkpoint = manifest.version == 3 && manifest.cache_format == "mla-int8-g64";
    if !old_f16 && !int8_checkpoint {
        return Err(format!("checkpoint cache format {} 与 version {} 不匹配", manifest.cache_format, manifest.version).into());
    }
    if manifest.last_layer > last_layer || (manifest.version == 1 && manifest.last_layer != last_layer) {
        return Err(format!("checkpoint 到 L{}，当前模型结束于 L{last_layer}", manifest.last_layer).into());
    }
    if manifest.prompt_tokens != token_ids.len() || manifest.token_ids != token_ids {
        return Err("checkpoint prompt token IDs 与当前 prompt 不一致".into());
    }
    if manifest.hidden_size != hidden_size {
        return Err(format!("checkpoint hidden_size {}，当前为 {hidden_size}", manifest.hidden_size).into());
    }
    let mut total_bytes = 0usize;
    for layer in 0..=manifest.last_layer {
        total_bytes += cache.load_layer(dir.join(format!("layer{layer:03}.kv")), layer).map_err(|error| format!("load L{layer}: {error}"))?;
        if cache.layer_len(layer) != token_ids.len() {
            return Err(format!("L{layer} cache 长度 {}，期望 {}", cache.layer_len(layer), token_ids.len()).into());
        }
    }
    let data = if int8_checkpoint {
        let path = dir.join("hidden.i8g64");
        dequantize_grouped_i8(&std::fs::read(&path)?, manifest.hidden_rows, hidden_size, crate::kv_cache::DEFAULT_GROUP_SIZE).map_err(|error| format!("{}: {error}", path.display()))?
    } else {
        let hidden_path = if manifest.version == 1 { dir.join("last_hidden.f16le") } else { dir.join("hidden.f16le") };
        let hidden_bytes = std::fs::read(&hidden_path)?;
        let expected_hidden_bytes = manifest.hidden_rows * hidden_size * std::mem::size_of::<u16>();
        if hidden_bytes.len() != expected_hidden_bytes {
            return Err(format!("{} 大小 {}，期望 {expected_hidden_bytes}", hidden_path.display(), hidden_bytes.len()).into());
        }
        hidden_bytes.chunks_exact(2).map(|bytes| half::f16::from_bits(u16::from_le_bytes(bytes.try_into().expect("2 字节 chunk"))).to_f32()).collect()
    };
    if old_f16 {
        let mut migrated_bytes = 0usize;
        for layer in 0..=manifest.last_layer {
            migrated_bytes += cache.dump_layer_int8(dir.join(format!("layer{layer:03}.kv")), layer).map_err(|error| format!("迁移 L{layer}: {error}"))?;
        }
        let (hidden_bytes, max_error, rmse) = quantize_grouped_i8(&data, manifest.hidden_rows, hidden_size, crate::kv_cache::DEFAULT_GROUP_SIZE)?;
        write_atomic(&dir.join("hidden.i8g64"), &hidden_bytes)?;
        let migrated_manifest = PrefillStateManifest {
            version: 3,
            model_key: manifest.model_key.clone(),
            cache_format: "mla-int8-g64".to_owned(),
            last_layer: manifest.last_layer,
            prompt_tokens: manifest.prompt_tokens,
            hidden_size: manifest.hidden_size,
            hidden_rows: manifest.hidden_rows,
            token_ids: manifest.token_ids.clone(),
        };
        write_atomic(&dir.join("manifest.json"), &serde_json::to_vec_pretty(&migrated_manifest)?)?;
        let old_hidden = if manifest.version == 1 { dir.join("last_hidden.f16le") } else { dir.join("hidden.f16le") };
        if old_hidden.exists() {
            std::fs::remove_file(&old_hidden)?;
        }
        println!("checkpoint 已迁移 INT8: KV {:.1} MiB, hidden {:.1} MiB(max_err={max_error:.5},rmse={rmse:.6})", migrated_bytes as f64 / (1024.0 * 1024.0), hidden_bytes.len() as f64 / (1024.0 * 1024.0));
    }
    println!("KV cache load 完成: {} 层，{:.1} MiB", manifest.last_layer + 1, total_bytes as f64 / (1024.0 * 1024.0));
    Ok((CpuTensor { data, rows: manifest.hidden_rows, cols: hidden_size }, manifest.last_layer))
}

#[cfg(target_os = "macos")]
fn quantize_grouped_i8(values: &[f32], rows: usize, columns: usize, group_size: usize) -> Result<(Vec<u8>, f32, f64), Box<dyn std::error::Error>> {
    if rows.checked_mul(columns) != Some(values.len()) || group_size == 0 || columns % group_size != 0 {
        return Err(format!("INT8 shape [{rows},{columns}] group={group_size} 非法").into());
    }
    let groups_per_row = columns / group_size;
    let mut bytes = vec![0u8; values.len() + rows * groups_per_row * 2];
    let (codes, scales) = bytes.split_at_mut(values.len());
    let mut max_error = 0.0f32;
    let mut squared_error = 0.0f64;
    for row in 0..rows {
        for group in 0..groups_per_row {
            let begin = row * columns + group * group_size;
            let source = &values[begin..begin + group_size];
            if source.iter().any(|value| !value.is_finite()) {
                return Err(format!("INT8 hidden [{row},{}] 含非有限值", group * group_size).into());
            }
            let maximum = source.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()));
            let scale = if maximum > 0.0 { maximum / 127.0 } else { 1.0 };
            let stored_scale = half::f16::from_f32(scale);
            let scale_offset = (row * groups_per_row + group) * 2;
            scales[scale_offset..scale_offset + 2].copy_from_slice(&stored_scale.to_bits().to_le_bytes());
            let inverse_scale = 1.0 / scale;
            for (offset, &value) in source.iter().enumerate() {
                let code = (value * inverse_scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                codes[begin + offset] = code as u8;
                let error = value - code as f32 * stored_scale.to_f32();
                max_error = max_error.max(error.abs());
                squared_error += f64::from(error) * f64::from(error);
            }
        }
    }
    Ok((bytes, max_error, (squared_error / values.len() as f64).sqrt()))
}

#[cfg(target_os = "macos")]
fn dequantize_grouped_i8(bytes: &[u8], rows: usize, columns: usize, group_size: usize) -> Result<Vec<f32>, String> {
    if group_size == 0 || columns % group_size != 0 {
        return Err(format!("INT8 shape [{rows},{columns}] group={group_size} 非法"));
    }
    let values = rows.checked_mul(columns).ok_or("INT8 hidden elements 溢出")?;
    let groups_per_row = columns / group_size;
    let expected = values.checked_add(rows * groups_per_row * 2).ok_or("INT8 hidden bytes 溢出")?;
    if bytes.len() != expected {
        return Err(format!("INT8 hidden 大小 {}，期望 {expected}", bytes.len()));
    }
    let (codes, scales) = bytes.split_at(values);
    let mut output = vec![0.0f32; values];
    for row in 0..rows {
        for column in 0..columns {
            let scale_index = row * groups_per_row + column / group_size;
            let scale_offset = scale_index * 2;
            let scale = half::f16::from_bits(u16::from_le_bytes([scales[scale_offset], scales[scale_offset + 1]])).to_f32();
            output[row * columns + column] = codes[row * columns + column] as i8 as f32 * scale;
        }
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    std::fs::rename(tmp, path)
}

#[cfg(target_os = "macos")]
fn print_decode_profile(ctx: &MetalContext, step: usize, wall_seconds: f64) {
    let stats = ctx.gpu_stats();
    eprintln!(
        "[decode perf {step}] wall={wall_seconds:.6}s submit_wait={:.6}s gpu_resident={:.6}s inter_command_gap={:.6}s completion_tail={:.6}s commands={}",
        stats.submit_wait_seconds, stats.seconds, stats.inter_command_gap_seconds, stats.completion_tail_seconds, stats.command_buffers
    );
    for profile in ctx.gpu_profile() {
        eprintln!("  decode gpu {:>8.3} ms x{} | {} {}", profile.gpu_seconds * 1.0e3, profile.calls, profile.operator, profile.shape);
    }
}

#[cfg(target_os = "macos")]
fn stats(data: &[f32]) -> (f32, f32, f32) {
    let mn = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    (mn, mx, mean)
}
