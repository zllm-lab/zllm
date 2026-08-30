//! Qwen3.5-4B GGUF × Android Vulkan standalone 组合。

use std::{path::Path, time::Instant};

use crate::{
    attention::{
        gated_delta_net::{GatedDeltaNetInputs, GatedDeltaNetState, GatedDeltaNetWeightsRef, gated_delta_net},
        hybrid::{HybridAttentionOptions, HybridTokenMixer},
        rope::RopeTable,
    },
    backend::{
        Backend, BackendError, BackendResources,
        vulkan::{VulkanContext, VulkanGatedDeltaNetStorage, VulkanKvCache, VulkanTensor, VulkanWeight},
    },
    runtime::qwen36::{Qwen36Config, Qwen36Runtime, Qwen36RuntimeLayer, chat_prompt, prepare_qwen36_gguf_layer, prepare_qwen36_gguf_output},
    weight::container::gguf::GgufReader,
};

/// XT2451-4 上 256 行会越过 Vulkan 工作集预算，128 行可稳定保留全部常驻权重。
/// chunk 只限制 activation 生命周期；量化 GEMM 的内部 tile 与权重驻留不变。
const RESIDENT_PREFILL_CHUNK_ROWS: usize = 128;

pub fn run(model_path: &Path, prompt: &str, max_seq_len: usize, decode_steps: usize, resident: bool, profile: bool) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Qwen36Config::standard_4b();
    crate::runtime::validate_max_sequence_length("Qwen3.6", max_seq_len, cfg.max_position_embeddings)?;
    let reader = GgufReader::open(&GgufReader::locate(model_path)?)?;
    reader.expect_metadata_str("general.architecture", "qwen35")?;
    reader.expect_metadata_u64("qwen35.embedding_length", cfg.hidden_size as u64)?;
    let tokenizer = reader.bpe_tokenizer()?;
    let detokenizer = reader.bpe_detokenizer()?;
    let tokens = tokenizer.tokenize(chat_prompt(prompt).as_bytes());
    if tokens.is_empty() || tokens.len() > max_seq_len {
        return Err(format!("prompt tokens={}，max_seq_len={max_seq_len}", tokens.len()).into());
    }

    let backend = VulkanContext::new()?;
    eprintln!("[qwen35-vulkan] adapter={} packed_i8_dot={} prompt_tokens={} weights={}", backend.adapter_name(), backend.supports_packed_i8_dot(), tokens.len(), if resident { "resident" } else { "stream" });
    if resident {
        return run_resident(&backend, &reader, &detokenizer, &cfg, &tokens, max_seq_len, decode_steps, profile);
    }
    let prepare_started = Instant::now();
    let (final_norm, output_head) = prepare_qwen36_gguf_output(&backend, &reader).map_err(|error| format!("准备 Vulkan output: {error:?}"))?;
    eprintln!("[qwen35-vulkan] output_head prepare={:.3}s", prepare_started.elapsed().as_secs_f64());

    let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);
    let runtime = Qwen36Runtime::new(&backend, &cfg, &[], &rope, HybridAttentionOptions { precise_prefill: true });
    let mut cache = VulkanKvCache::new_q8(cfg.num_layers, max_seq_len)?;
    // llama.cpp 的 Qwen3.5 GGUF 将 value head 按 tiled 布局排列。
    let mut delta_state = GatedDeltaNetState::<VulkanGatedDeltaNetStorage>::new(cfg.num_layers, cfg.gated_delta_net_spec())?;
    let embedding = reader.embedding_rows("token_embd.weight", &tokens, cfg.hidden_size, cfg.vocab_size)?;
    let mut hidden = backend.tensor_from_f32(&embedding, tokens.len(), cfg.hidden_size)?;
    let prefill_started = Instant::now();
    let first_prepare_started = Instant::now();
    let mut weights = prepare_qwen36_gguf_layer(&backend, &reader, &cfg, 0).map_err(|error| format!("准备 Vulkan L0: {error:?}"))?;
    let mut prefill_prepare = first_prepare_started.elapsed();
    let mut prefill_submit = std::time::Duration::ZERO;
    let mut prefill_wait = std::time::Duration::ZERO;
    for layer in 0..cfg.num_layers {
        let submit_started = Instant::now();
        hidden = runtime.prepared_layer(&mut cache, &mut delta_state, 0, layer, &weights, hidden).map_err(|error| format!("Vulkan prefill L{layer}: {error:?}"))?;
        let submit_elapsed = submit_started.elapsed();
        prefill_submit += submit_elapsed;
        // 当前层已经提交到 GPU 后再准备下一层，让文件读取与当前层计算重叠。
        let prepare_started = Instant::now();
        let next_weights = (layer + 1 < cfg.num_layers).then(|| prepare_qwen36_gguf_layer(&backend, &reader, &cfg, layer + 1).map_err(|error| format!("准备 Vulkan L{}: {error:?}", layer + 1))).transpose()?;
        let prepare_elapsed = prepare_started.elapsed();
        if next_weights.is_some() {
            prefill_prepare += prepare_elapsed;
        }
        let wait_started = Instant::now();
        backend.synchronize()?;
        let wait_elapsed = wait_started.elapsed();
        prefill_wait += wait_elapsed;
        if let Some(next_weights) = next_weights {
            weights = next_weights;
        }
        eprintln!("[qwen35-vulkan] prefill layer={}/{} next_prepare={:.3}s submit={:.3}s wait={:.3}s", layer + 1, cfg.num_layers, prepare_elapsed.as_secs_f64(), submit_elapsed.as_secs_f64(), wait_elapsed.as_secs_f64());
    }
    eprintln!("[qwen35-vulkan] prefill={:.3}s prepare={:.3}s submit={:.3}s wait={:.3}s", prefill_started.elapsed().as_secs_f64(), prefill_prepare.as_secs_f64(), prefill_submit.as_secs_f64(), prefill_wait.as_secs_f64());
    let mut hidden = backend.select_row(&hidden, tokens.len() - 1)?;

    for step in 0..decode_steps {
        let output_started = Instant::now();
        let normalized = backend.gemma_rmsnorm(&hidden, &final_norm, cfg.rms_norm_eps)?;
        let logits = backend.linear(&normalized, &output_head)?;
        let token = backend.argmax(&logits)?;
        backend.synchronize()?;
        let output_elapsed = output_started.elapsed();
        crate::runtime::generation::write_token(&detokenizer, token, false)?;
        if cfg.eos_token_ids.contains(&token) || step + 1 == decode_steps {
            break;
        }
        let position = tokens.len() + step;
        let input = backend.embedding_row(&output_head, token)?;
        hidden = input;
        let decode_started = Instant::now();
        let first_prepare_started = Instant::now();
        let mut weights = prepare_qwen36_gguf_layer(&backend, &reader, &cfg, 0).map_err(|error| format!("准备 Vulkan decode L0: {error:?}"))?;
        let mut decode_prepare = first_prepare_started.elapsed();
        let mut decode_submit = std::time::Duration::ZERO;
        let mut decode_wait = std::time::Duration::ZERO;
        for layer in 0..cfg.num_layers {
            let submit_started = Instant::now();
            hidden = runtime.prepared_layer(&mut cache, &mut delta_state, position, layer, &weights, hidden).map_err(|error| format!("Vulkan decode position={position} L{layer}: {error:?}"))?;
            decode_submit += submit_started.elapsed();
            let prepare_started = Instant::now();
            let next_weights = (layer + 1 < cfg.num_layers).then(|| prepare_qwen36_gguf_layer(&backend, &reader, &cfg, layer + 1).map_err(|error| format!("准备 Vulkan decode L{}: {error:?}", layer + 1))).transpose()?;
            if next_weights.is_some() {
                decode_prepare += prepare_started.elapsed();
            }
            let wait_started = Instant::now();
            backend.synchronize()?;
            decode_wait += wait_started.elapsed();
            if let Some(next_weights) = next_weights {
                weights = next_weights;
            }
        }
        eprintln!(
            "[qwen35-vulkan] decode step={} total={:.3}s prepare={:.3}s submit={:.3}s wait={:.3}s output={:.3}s",
            step + 1,
            decode_started.elapsed().as_secs_f64(),
            decode_prepare.as_secs_f64(),
            decode_submit.as_secs_f64(),
            decode_wait.as_secs_f64(),
            output_elapsed.as_secs_f64()
        );
    }
    backend.synchronize()?;
    println!();
    Ok(())
}

fn run_resident(
    backend: &VulkanContext,
    reader: &GgufReader,
    detokenizer: &crate::tokenizer::Detokenizer,
    cfg: &Qwen36Config,
    tokens: &[u32],
    max_seq_len: usize,
    decode_steps: usize,
    profile: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let prepare_started = Instant::now();
    let mut resident_bytes = 0_u64;
    let mut layers = Vec::with_capacity(cfg.num_layers);
    for layer in 0..cfg.num_layers {
        let weights = prepare_qwen36_gguf_layer(backend, reader, cfg, layer).map_err(|error| format!("准备 resident Vulkan L{layer}: {error:?}"))?;
        resident_bytes += layer_bytes(&weights);
        layers.push(weights);
        eprintln!("[qwen35-vulkan-resident] layer={}/{} resident={:.2}GiB", layer + 1, cfg.num_layers, resident_bytes as f64 / 1_073_741_824.0);
    }
    let (final_norm, output_head) = prepare_qwen36_gguf_output(backend, reader).map_err(|error| format!("准备 resident Vulkan output: {error:?}"))?;
    resident_bytes += final_norm.allocated_bytes() + output_head.allocated_bytes();
    eprintln!("[qwen35-vulkan-resident] ready={:.2}GiB prepare={:.3}s", resident_bytes as f64 / 1_073_741_824.0, prepare_started.elapsed().as_secs_f64());

    let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);
    let runtime = Qwen36Runtime::new(backend, cfg, &layers, &rope, HybridAttentionOptions { precise_prefill: true });
    let mut cache = VulkanKvCache::new_q8(cfg.num_layers, max_seq_len)?;
    let mut delta_state = GatedDeltaNetState::<VulkanGatedDeltaNetStorage>::new(cfg.num_layers, cfg.gated_delta_net_spec())?;
    let prefill_started = Instant::now();
    let mut final_chunk = None;
    for (chunk, chunk_tokens) in tokens.chunks(RESIDENT_PREFILL_CHUNK_ROWS).enumerate() {
        let position = chunk * RESIDENT_PREFILL_CHUNK_ROWS;
        let chunk_started = Instant::now();
        let embedding = reader.embedding_rows("token_embd.weight", chunk_tokens, cfg.hidden_size, cfg.vocab_size)?;
        let mut hidden = backend.tensor_from_f32(&embedding, chunk_tokens.len(), cfg.hidden_size)?;
        if profile {
            backend.take_submit_profile();
            for (layer, weights) in layers.iter().enumerate() {
                let layer_started = Instant::now();
                hidden = profile_prefill_layer(&runtime, &mut cache, &mut delta_state, position, layer, &hidden)?;
                let kind = match &weights.token_mixer {
                    HybridTokenMixer::FullAttention(_) => "full",
                    HybridTokenMixer::DeltaNet(_) => "delta",
                };
                eprintln!(
                    "[qwen35-vulkan-prefill-profile] position={position} layer={layer} kind={kind} mlp={}/{}/{} wall={:.6}s",
                    weight_kind(&weights.mlp.gate),
                    weight_kind(&weights.mlp.up),
                    weight_kind(&weights.mlp.down),
                    layer_started.elapsed().as_secs_f64(),
                );
            }
        } else {
            hidden = runtime.at(&mut cache, &mut delta_state, position).prefill(hidden)?;
        }
        backend.synchronize()?;
        eprintln!("[qwen35-vulkan-resident] prefill_chunk={}/{} position={position} rows={} wall={:.3}s", chunk + 1, tokens.len().div_ceil(RESIDENT_PREFILL_CHUNK_ROWS), chunk_tokens.len(), chunk_started.elapsed().as_secs_f64(),);
        if position + chunk_tokens.len() == tokens.len() {
            final_chunk = Some(hidden);
        }
    }
    eprintln!("[qwen35-vulkan-resident] prefill={:.3}s", prefill_started.elapsed().as_secs_f64());
    let final_chunk = final_chunk.ok_or("resident prefill 没有输入 chunk")?;
    let mut hidden = backend.select_row(&final_chunk, (tokens.len() - 1) % RESIDENT_PREFILL_CHUNK_ROWS)?;
    for step in 0..decode_steps {
        backend.take_submit_profile();
        backend.take_decode_resource_profile();
        let round_started = Instant::now();
        let normalized = backend.gemma_rmsnorm(&hidden, &final_norm, cfg.rms_norm_eps)?;
        let logits = backend.linear(&normalized, &output_head)?;
        let token = backend.argmax(&logits)?;
        let output_elapsed = round_started.elapsed();
        let (output_logical, output_driver, output_queue_wall) = backend.take_submit_profile();
        crate::runtime::generation::write_token(detokenizer, token, false)?;
        if cfg.eos_token_ids.contains(&token) || step + 1 == decode_steps {
            break;
        }
        let position = tokens.len() + step;
        let input = backend.embedding_row(&output_head, token)?;
        let mut model_submit = std::time::Duration::ZERO;
        let mut model_wait = std::time::Duration::ZERO;
        if profile {
            hidden = input;
            for (layer, weights) in layers.iter().enumerate() {
                let layer_started = Instant::now();
                let submit_started = Instant::now();
                hidden = runtime.prepared_layer(&mut cache, &mut delta_state, position, layer, weights, hidden)?;
                model_submit += submit_started.elapsed();
                let wait_started = Instant::now();
                backend.synchronize()?;
                model_wait += wait_started.elapsed();
                let kind = match &weights.token_mixer {
                    HybridTokenMixer::FullAttention(_) => "full",
                    HybridTokenMixer::DeltaNet(_) => "delta",
                };
                let mlp = if matches!((&weights.mlp.gate, &weights.mlp.up), (VulkanWeight::Iq4Xs { .. }, VulkanWeight::Iq4Xs { .. })) { "iq4_xs" } else { "other" };
                eprintln!("[qwen35-vulkan-profile] step={} layer={} kind={kind} mlp={mlp} wall={:.6}s", step + 1, layer, layer_started.elapsed().as_secs_f64());
            }
        } else {
            let submit_started = Instant::now();
            hidden = runtime.at(&mut cache, &mut delta_state, position).decode(input)?;
            model_submit = submit_started.elapsed();
            let wait_started = Instant::now();
            backend.synchronize()?;
            model_wait = wait_started.elapsed();
        }
        let (model_logical, model_driver, model_queue_wall) = backend.take_submit_profile();
        let [output_hits, output_misses, uniform_hits, uniform_misses, bind_hits, bind_misses] = backend.take_decode_resource_profile();
        eprintln!(
            "[qwen35-vulkan-resident] decode step={} wall={:.3}s output={:.3}s model={:.3}s submit={:.3}s wait={:.3}s queues={output_logical}:{output_driver}/{model_logical}:{model_driver} queue_wall={:.3}/{:.3}s reuse=o{output_hits}/{output_misses},u{uniform_hits}/{uniform_misses},b{bind_hits}/{bind_misses}",
            step + 1,
            round_started.elapsed().as_secs_f64(),
            output_elapsed.as_secs_f64(),
            (round_started.elapsed() - output_elapsed).as_secs_f64(),
            model_submit.as_secs_f64(),
            model_wait.as_secs_f64(),
            output_queue_wall.as_secs_f64(),
            model_queue_wall.as_secs_f64()
        );
    }
    backend.synchronize()?;
    println!();
    Ok(())
}

fn layer_bytes(layer: &Qwen36RuntimeLayer<VulkanWeight>) -> u64 {
    let mixer = match &layer.token_mixer {
        HybridTokenMixer::FullAttention(weights) => {
            weights.query_gate.allocated_bytes() + weights.query_norm.allocated_bytes() + weights.key.allocated_bytes() + weights.key_norm.allocated_bytes() + weights.value.allocated_bytes() + weights.output.allocated_bytes()
        }
        HybridTokenMixer::DeltaNet(weights) => {
            weights.qkv.allocated_bytes()
                + weights.z.allocated_bytes()
                + weights.alpha.allocated_bytes()
                + weights.beta.allocated_bytes()
                + weights.conv.allocated_bytes()
                + weights.a_log.allocated_bytes()
                + weights.dt_bias.allocated_bytes()
                + weights.norm.allocated_bytes()
                + weights.output.allocated_bytes()
        }
    };
    layer.input_norm.allocated_bytes() + layer.post_attention_norm.allocated_bytes() + layer.mlp.gate.allocated_bytes() + layer.mlp.up.allocated_bytes() + layer.mlp.down.allocated_bytes() + mixer
}

fn profile_prefill_layer(
    runtime: &Qwen36Runtime<'_, VulkanContext>,
    cache: &mut VulkanKvCache,
    recurrent: &mut GatedDeltaNetState<VulkanGatedDeltaNetStorage>,
    position: usize,
    layer: usize,
    hidden: &VulkanTensor,
) -> Result<VulkanTensor, BackendError> {
    let backend = runtime.backend;
    let weights = runtime.layers.get(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
    backend.begin_batch();
    let result = (|| {
        let mut stage_started = Instant::now();
        let normed = backend.gemma_rmsnorm_f32(hidden, &weights.input_norm, runtime.config.rms_norm_eps)?;
        stage_started = finish_prefill_profile_stage(backend, position, layer, "input_norm", stage_started)?;
        let mixed = match &weights.token_mixer {
            HybridTokenMixer::FullAttention(weights) => {
                let mixed = runtime.attention.full(weights, cache, layer, &normed, position)?;
                stage_started = finish_prefill_profile_stage(backend, position, layer, "mixer_full", stage_started)?;
                mixed
            }
            HybridTokenMixer::DeltaNet(weights) => {
                let (qkv, z) = backend.dual_linear(&normed, &weights.qkv, &weights.z)?;
                stage_started = finish_prefill_profile_stage(backend, position, layer, "mixer_qkv_z", stage_started)?;
                let (alpha, beta) = backend.dual_linear(&normed, &weights.alpha, &weights.beta)?;
                stage_started = finish_prefill_profile_stage(backend, position, layer, "mixer_alpha_beta", stage_started)?;
                let spec = runtime.config.gated_delta_net_spec();
                let core = gated_delta_net(
                    backend,
                    recurrent,
                    layer,
                    position,
                    GatedDeltaNetInputs { qkv: &qkv, z: &z, alpha: &alpha, beta: &beta },
                    GatedDeltaNetWeightsRef { conv: &weights.conv, a_log: &weights.a_log, dt_bias: &weights.dt_bias, norm: &weights.norm },
                    &spec,
                )?;
                stage_started = finish_prefill_profile_stage(backend, position, layer, "mixer_core", stage_started)?;
                let mixed = backend.linear(&core, &weights.output)?;
                stage_started = finish_prefill_profile_stage(backend, position, layer, "mixer_output", stage_started)?;
                mixed
            }
        };
        let (residual, ffn_input) = backend.add_gemma_rmsnorm_pair(hidden, &mixed, &weights.post_attention_norm, runtime.config.rms_norm_eps)?;
        stage_started = finish_prefill_profile_stage(backend, position, layer, "post_norm", stage_started)?;
        let output = backend.gated_mlp_add_residual(&ffn_input, &weights.mlp.gate, &weights.mlp.up, &weights.mlp.down, &runtime.mlp.activation, &residual)?;
        stage_started = finish_prefill_profile_stage(backend, position, layer, "ffn", stage_started)?;
        Ok(output)
    })();
    backend.finish_batch();
    result
}

fn finish_prefill_profile_stage(backend: &VulkanContext, position: usize, layer: usize, stage: &str, started: Instant) -> Result<Instant, BackendError> {
    let submit_started = Instant::now();
    backend.submit_batch();
    let submit_elapsed = submit_started.elapsed();
    let wait_started = Instant::now();
    backend.synchronize()?;
    let wait_elapsed = wait_started.elapsed();
    let (logical, driver, queue_wall) = backend.take_submit_profile();
    eprintln!(
        "[qwen35-vulkan-prefill-stage] position={position} layer={layer} stage={stage} wall={:.6}s submit={:.6}s wait={:.6}s queues={logical}:{driver} queue_wall={:.6}s",
        started.elapsed().as_secs_f64(),
        submit_elapsed.as_secs_f64(),
        wait_elapsed.as_secs_f64(),
        queue_wall.as_secs_f64(),
    );
    Ok(Instant::now())
}

fn weight_kind(weight: &VulkanWeight) -> &'static str {
    match weight {
        VulkanWeight::Iq4Xs { .. } => "iq4_xs",
        VulkanWeight::Q4K { .. } => "q4_k",
        VulkanWeight::Q5K { .. } => "q5_k",
        VulkanWeight::Q6K { .. } => "q6_k",
        VulkanWeight::Q8_0 { .. } => "q8_0",
        VulkanWeight::W8A16 { .. } => "w8a16",
        VulkanWeight::F32 { .. } => "f32",
    }
}
