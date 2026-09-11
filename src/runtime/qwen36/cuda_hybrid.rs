//! Qwen3.8 GGUF × CPU/CUDA 连续分层执行。
//!
//! CPU 执行前缀层，CUDA 执行显存容许的连续后缀层并持有输出头。KV cache
//! 与 GDN state 始终跟随负责该层的设备；每个 chunk/token 只跨设备一次 hidden。

use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::sync_channel,
    },
    time::{Duration, Instant},
};

use crate::{
    attention::{
        gated_delta_net::GatedDeltaNetState,
        hybrid::{HybridAttentionOptions, HybridTokenMixer},
        rope::RopeTable,
    },
    backend::{
        Backend,
        cpu::{CpuContext, CpuGatedDeltaNetStorage, CpuKvCache},
        cuda::{CudaContext, CudaGatedDeltaNetStorage, CudaKvCache, CudaTensor},
    },
    kernel::cpu::CpuTensor,
    runtime::{
        qwen36::{Qwen36Config, Qwen36Runtime, Qwen36RuntimeLayer, chat_prompt, prepare_qwen36_gguf_layer, prepare_qwen36_gguf_output},
        session::GenerationOutput,
    },
    weight::container::gguf::{GgufReader, GgufTensorInfo},
};

const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;
const MIN_WORKSPACE_BYTES: usize = 256 * MIB;

#[derive(Clone, Copy, Debug)]
pub struct HybridCudaOptions {
    /// None 按当前空闲显存与实际 KV 格式准入预算自动选择；Some(n) 类比 llama.cpp `-ngl n`。
    pub gpu_layers: Option<usize>,
    pub max_sequence_length: usize,
    pub prefill_chunk_size: usize,
    pub decode_cpu_threads: Option<usize>,
    pub cpu_packed_resident: bool,
    pub vram_reserve_bytes: usize,
    pub precise_gqa_prefill: bool,
    pub kv_f16: bool,
}

#[derive(Clone, Debug)]
pub struct HybridCudaPlan {
    pub gpu_layer_start: usize,
    pub gpu_layers: usize,
    pub full_attention_layers: usize,
    pub estimated_weight_bytes: usize,
    pub estimated_kv_bytes: usize,
    pub estimated_state_bytes: usize,
    pub estimated_workspace_bytes: usize,
    pub estimated_total_bytes: usize,
}

fn checked_add(total: usize, value: usize, what: &str) -> Result<usize, String> {
    total.checked_add(value).ok_or_else(|| format!("Qwen3.8 {what} 字节数溢出"))
}

fn cuda_tensor_bytes(tensor: &GgufTensorInfo) -> Result<usize, String> {
    match tensor.tensor_type.0 {
        12..=14 | 8 => Ok(tensor.bytes),
        0 | 1 | 30 => tensor.elements().checked_mul(2).ok_or_else(|| format!("{} dense tensor 大小溢出", tensor.name)),
        other => Err(format!("CUDA GPU 层不支持 GGUF {}(type={other}) tensor {}；当前要求 Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16/F32", tensor.tensor_type.name(), tensor.name)),
    }
}

fn tensors_bytes<'a>(mut tensors: impl Iterator<Item = &'a GgufTensorInfo>) -> Result<usize, String> {
    tensors.try_fold(0usize, |total, tensor| checked_add(total, cuda_tensor_bytes(tensor)?, "resident weight"))
}

fn layer_weight_bytes(reader: &GgufReader, layer: usize) -> Result<usize, String> {
    let prefix = format!("blk.{layer}.");
    tensors_bytes(reader.tensors().iter().filter(|tensor| tensor.name.starts_with(&prefix)))
}

fn output_weight_bytes(reader: &GgufReader) -> Result<usize, String> {
    let output = if reader.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let tensors = ["output_norm.weight", output].into_iter().map(|name| reader.tensor(name).ok_or_else(|| format!("GGUF 缺少 {name}"))).collect::<Result<Vec<_>, _>>()?;
    tensors_bytes(tensors.into_iter())
}

fn layer_state_bytes(cfg: &Qwen36Config, layer: usize) -> Result<usize, String> {
    if (layer + 1).is_multiple_of(cfg.full_attention_interval) {
        return Ok(0);
    }
    let spec = cfg.gated_delta_net_spec();
    spec.recurrent_elements().checked_add(spec.conv_state_elements()).and_then(|elements| elements.checked_mul(4)).ok_or_else(|| "GDN state 大小溢出".to_owned())
}

fn layer_kv_bytes(cfg: &Qwen36Config, layer: usize, max_sequence_length: usize, kv_f16: bool) -> Result<usize, String> {
    if !(layer + 1).is_multiple_of(cfg.full_attention_interval) {
        return Ok(0);
    }
    let columns = cfg.num_kv_heads.checked_mul(cfg.head_dim).ok_or("KV columns 大小溢出")?;
    if kv_f16 {
        return max_sequence_length.checked_mul(columns).and_then(|elements| elements.checked_mul(4)).ok_or_else(|| "F16 K/V 大小溢出".to_owned());
    }
    // Q8G64：K/V 各含一字节 code 与每 64 元素一个 F32 scale。
    let codes = max_sequence_length.checked_mul(columns).ok_or("KV codes 大小溢出")?;
    let scales = max_sequence_length.checked_mul(columns / 64).and_then(|value| value.checked_mul(4)).ok_or("KV scales 大小溢出")?;
    codes.checked_add(scales).and_then(|one| one.checked_mul(2)).ok_or_else(|| "Q8G64 KV 大小溢出".to_owned())
}

fn workspace_bytes(cfg: &Qwen36Config, prefill_chunk_size: usize) -> Result<usize, String> {
    // 量化矩阵原地计算，不需要 dequant workspace；这里只预算 activation 与 CUDA runtime。
    let activation = prefill_chunk_size.checked_mul(cfg.intermediate_size).and_then(|value| value.checked_mul(4)).ok_or("activation workspace 大小溢出")?;
    Ok(activation.max(MIN_WORKSPACE_BYTES))
}

pub fn plan_hybrid_layers(reader: &GgufReader, cfg: &Qwen36Config, options: HybridCudaOptions, free_bytes: usize) -> Result<HybridCudaPlan, String> {
    crate::runtime::validate_max_sequence_length("Qwen3.6", options.max_sequence_length, cfg.max_position_embeddings)?;
    if options.prefill_chunk_size == 0 || options.prefill_chunk_size > options.max_sequence_length {
        return Err(format!("prefill_chunk_size={} 必须位于 1..={}", options.prefill_chunk_size, options.max_sequence_length));
    }
    if options.gpu_layers.is_some_and(|layers| layers > cfg.num_layers) {
        return Err(format!("gpu_layers 超过模型层数 {}", cfg.num_layers));
    }
    let output_bytes = output_weight_bytes(reader)?;
    let workspace = workspace_bytes(cfg, options.prefill_chunk_size)?;
    let budget = free_bytes.checked_sub(options.vram_reserve_bytes).ok_or_else(|| format!("空闲显存 {:.2}GiB 小于保留 {:.2}GiB", free_bytes as f64 / GIB as f64, options.vram_reserve_bytes as f64 / GIB as f64))?;
    if checked_add(output_bytes, workspace, "output/workspace")? > budget {
        return Err("输出头与 CUDA workspace 已超过显存预算".to_owned());
    }

    let wanted = options.gpu_layers.unwrap_or(cfg.num_layers);
    let mut weights = output_bytes;
    let mut kv = 0usize;
    let mut state = 0usize;
    let mut accepted = 0usize;
    let mut full_attention_layers = 0usize;
    for layer in (cfg.num_layers - wanted..cfg.num_layers).rev() {
        let candidate_weights = checked_add(weights, layer_weight_bytes(reader, layer)?, "weight")?;
        let layer_kv = layer_kv_bytes(cfg, layer, options.max_sequence_length, options.kv_f16)?;
        let candidate_kv = checked_add(kv, layer_kv, "KV")?;
        let candidate_state = checked_add(state, layer_state_bytes(cfg, layer)?, "state")?;
        let candidate = [candidate_weights, candidate_kv, candidate_state, workspace].into_iter().try_fold(0usize, |total, value| checked_add(total, value, "plan"))?;
        if options.gpu_layers.is_none() && candidate > budget {
            break;
        }
        weights = candidate_weights;
        kv = candidate_kv;
        state = candidate_state;
        accepted += 1;
        full_attention_layers += usize::from(layer_kv != 0);
    }
    let total = [weights, kv, state, workspace].into_iter().try_fold(0usize, |sum, value| checked_add(sum, value, "plan total"))?;
    if accepted != wanted && options.gpu_layers.is_some() || total > budget {
        return Err(format!("gpu_layers={accepted} 的准入预计 {:.2}GiB，扣除保留后预算 {:.2}GiB", total as f64 / GIB as f64, budget as f64 / GIB as f64));
    }
    Ok(HybridCudaPlan {
        gpu_layer_start: cfg.num_layers - accepted,
        gpu_layers: accepted,
        full_attention_layers,
        estimated_weight_bytes: weights,
        estimated_kv_bytes: kv,
        estimated_state_bytes: state,
        estimated_workspace_bytes: workspace,
        estimated_total_bytes: total,
    })
}

fn run_cpu_layers(
    runtime: &Qwen36Runtime<'_, CpuContext>,
    layers: &[Qwen36RuntimeLayer<crate::backend::cpu::CpuWeight>],
    cache: &mut CpuKvCache,
    recurrent: &mut GatedDeltaNetState<CpuGatedDeltaNetStorage>,
    position: usize,
    mut hidden: CpuTensor,
) -> Result<CpuTensor, String> {
    for (layer, weights) in layers.iter().enumerate() {
        hidden = runtime.prepared_layer(cache, recurrent, position, layer, weights, hidden).map_err(|error| format!("CPU L{layer}: {error:?}"))?;
    }
    Ok(hidden)
}

fn make_cpu_layer_resident(layer: &mut Qwen36RuntimeLayer<crate::backend::cpu::CpuWeight>) -> Result<usize, String> {
    let mut total = 0usize;
    for weight in [&mut layer.input_norm, &mut layer.post_attention_norm, &mut layer.mlp.gate, &mut layer.mlp.up, &mut layer.mlp.down] {
        total = checked_add(total, weight.make_gguf_resident()?, "CPU resident weight")?;
    }
    match &mut layer.token_mixer {
        HybridTokenMixer::FullAttention(weights) => {
            for weight in [&mut weights.query_gate, &mut weights.query_norm, &mut weights.key, &mut weights.key_norm, &mut weights.value, &mut weights.output] {
                total = checked_add(total, weight.make_gguf_resident()?, "CPU resident attention")?;
            }
        }
        HybridTokenMixer::DeltaNet(weights) => {
            for weight in [&mut weights.qkv, &mut weights.z, &mut weights.alpha, &mut weights.beta, &mut weights.conv, &mut weights.a_log, &mut weights.dt_bias, &mut weights.norm, &mut weights.output] {
                total = checked_add(total, weight.make_gguf_resident()?, "CPU resident DeltaNet")?;
            }
        }
    }
    Ok(total)
}

fn run_gpu_layers(
    runtime: &Qwen36Runtime<'_, CudaContext>,
    first_layer: usize,
    layers: &[Qwen36RuntimeLayer<crate::backend::cuda::CudaWeight>],
    cache: &mut CudaKvCache,
    recurrent: &mut GatedDeltaNetState<CudaGatedDeltaNetStorage>,
    position: usize,
    mut hidden: CudaTensor,
) -> Result<CudaTensor, String> {
    for (offset, weights) in layers.iter().enumerate() {
        let layer = first_layer + offset;
        hidden = runtime.prepared_layer(cache, recurrent, position, layer, weights, hidden).map_err(|error| format!("CUDA L{layer}: {error:?}"))?;
    }
    Ok(hidden)
}

struct PrefillChunk {
    position: usize,
    rows: usize,
    hidden: CpuTensor,
    cpu_wall: Duration,
}

#[allow(clippy::too_many_arguments)]
pub fn generate(
    backend: &CudaContext,
    model_path: &Path,
    rendered_prompt: &str,
    max_tokens: usize,
    options: HybridCudaOptions,
    stops: &[String],
    cancellation: &AtomicBool,
    on_token: &mut dyn FnMut(u32, String) -> bool,
) -> Result<crate::runtime::session::GenerationSummary, Box<dyn std::error::Error>> {
    let cfg = Qwen36Config::standard_27b();
    let located = GgufReader::locate(model_path)?;
    let reader = GgufReader::open(&located)?;
    reader.expect_metadata_str("general.architecture", "qwen35")?;
    reader.expect_metadata_u64("qwen35.embedding_length", cfg.hidden_size as u64)?;
    let blocks = reader.metadata_u64("qwen35.block_count")? as usize;
    if blocks != cfg.num_layers && blocks != cfg.num_layers + cfg.mtp_layers {
        return Err(format!("qwen35.block_count={blocks}，期望 {} 或 {}", cfg.num_layers, cfg.num_layers + cfg.mtp_layers).into());
    }
    let tokenizer = reader.bpe_tokenizer()?;
    let detokenizer = reader.bpe_detokenizer()?;
    let tokens = tokenizer.tokenize(rendered_prompt.as_bytes());
    if tokens.is_empty() || tokens.len() >= options.max_sequence_length {
        return Err(format!("prompt tokens={}，max_sequence_length={}", tokens.len(), options.max_sequence_length).into());
    }

    let (free, total) = backend.device().mem_get_info()?;
    let plan = plan_hybrid_layers(&reader, &cfg, options, free)?;
    eprintln!(
        "[qwen38-hybrid-plan] device={} vram={:.2}/{:.2}GiB cpu_layers=0..{} cuda_layers={}..{} full_attention_cuda={} estimate: weights={:.2}GiB kv_{}={:.2}GiB state={:.2}GiB workspace={:.2}GiB total={:.2}GiB reserve={:.2}GiB",
        backend.device_name(),
        free as f64 / GIB as f64,
        total as f64 / GIB as f64,
        plan.gpu_layer_start,
        plan.gpu_layer_start,
        cfg.num_layers,
        plan.full_attention_layers,
        plan.estimated_weight_bytes as f64 / GIB as f64,
        if options.kv_f16 { "f16" } else { "q8g64" },
        plan.estimated_kv_bytes as f64 / GIB as f64,
        plan.estimated_state_bytes as f64 / GIB as f64,
        plan.estimated_workspace_bytes as f64 / GIB as f64,
        plan.estimated_total_bytes as f64 / GIB as f64,
        options.vram_reserve_bytes as f64 / GIB as f64,
    );

    let cpu = CpuContext;
    let prepare_started = Instant::now();
    let mut cpu_layers = (0..plan.gpu_layer_start).map(|layer| prepare_qwen36_gguf_layer(&cpu, &reader, &cfg, layer).map_err(|error| format!("准备 CPU L{layer}: {error:?}"))).collect::<Result<Vec<_>, _>>()?;
    if options.cpu_packed_resident {
        let resident_bytes = cpu_layers.iter_mut().try_fold(0usize, |total, layer| make_cpu_layer_resident(layer).and_then(|bytes| checked_add(total, bytes, "CPU resident total")))?;
        eprintln!("[qwen38-hybrid-load] cpu_packed_resident={:.2}GiB", resident_bytes as f64 / GIB as f64);
    }
    let mut gpu_layers = Vec::with_capacity(plan.gpu_layers);
    for layer in plan.gpu_layer_start..cfg.num_layers {
        gpu_layers.push(prepare_qwen36_gguf_layer(backend, &reader, &cfg, layer).map_err(|error| format!("准备 CUDA L{layer}: {error:?}"))?);
        eprintln!("[qwen38-hybrid-load] cuda_layer={}/{}", layer + 1, cfg.num_layers);
    }
    let (final_norm, output_head) = prepare_qwen36_gguf_output(backend, &reader).map_err(|error| format!("准备 CUDA output: {error:?}"))?;
    backend.synchronize()?;
    let (free_after_load, _) = backend.device().mem_get_info()?;
    eprintln!("[qwen38-hybrid-load] wall={:.3}s vram_weights_used={:.2}GiB free={:.2}GiB", prepare_started.elapsed().as_secs_f64(), free.saturating_sub(free_after_load) as f64 / GIB as f64, free_after_load as f64 / GIB as f64);

    let rope = RopeTable::precompute(options.max_sequence_length, cfg.rope_dim, cfg.rope_theta);
    let attention = HybridAttentionOptions { precise_prefill: options.precise_gqa_prefill };
    let kv_columns = cfg.num_kv_heads.checked_mul(cfg.head_dim).ok_or("KV columns 大小溢出")?;
    let mut gpu_cache = if options.kv_f16 { CudaKvCache::new(cfg.num_layers, options.max_sequence_length, kv_columns) } else { CudaKvCache::new_q8g64(cfg.num_layers, options.max_sequence_length, kv_columns)? };
    let mut gpu_recurrent = GatedDeltaNetState::<CudaGatedDeltaNetStorage>::new(cfg.num_layers, cfg.gated_delta_net_spec()).map_err(|error| format!("创建 CUDA GDN state: {error:?}"))?;
    let gpu_runtime = Qwen36Runtime::new(backend, &cfg, &gpu_layers, &rope, attention);

    // CPU 前缀与前一 chunk 的 CUDA 后缀流水重叠；channel=1 限制 activation 生命周期。
    let prefill_started = Instant::now();
    let ((mut cpu_cache, mut cpu_recurrent), final_hidden) = std::thread::scope(|scope| -> Result<_, String> {
        let (sender, receiver) = sync_channel::<Result<PrefillChunk, String>>(1);
        let cfg_ref = &cfg;
        let layers_ref = &cpu_layers;
        let reader_ref = &reader;
        let rope_ref = &rope;
        let tokens_ref = &tokens;
        let chunk_size = options.prefill_chunk_size;
        let producer = scope.spawn(move || -> Result<(CpuKvCache, GatedDeltaNetState<CpuGatedDeltaNetStorage>), String> {
            let runtime = Qwen36Runtime::new(&cpu, cfg_ref, layers_ref, rope_ref, attention);
            let mut cache = CpuKvCache::new(cfg_ref.num_layers);
            let mut recurrent = GatedDeltaNetState::<CpuGatedDeltaNetStorage>::new(cfg_ref.num_layers, cfg_ref.gated_delta_net_spec()).map_err(|error| format!("创建 CPU GDN state: {error:?}"))?;
            for (chunk, chunk_tokens) in tokens_ref.chunks(chunk_size).enumerate() {
                let position = chunk * chunk_size;
                let started = Instant::now();
                let embedding = reader_ref.embedding_rows("token_embd.weight", chunk_tokens, cfg_ref.hidden_size, cfg_ref.vocab_size)?;
                let input = CpuTensor { data: embedding, rows: chunk_tokens.len(), cols: cfg_ref.hidden_size };
                let hidden = run_cpu_layers(&runtime, layers_ref, &mut cache, &mut recurrent, position, input)?;
                sender.send(Ok(PrefillChunk { position, rows: chunk_tokens.len(), hidden, cpu_wall: started.elapsed() })).map_err(|_| "CUDA prefill 已提前结束".to_owned())?;
            }
            Ok((cache, recurrent))
        });

        let mut last = None;
        let mut gpu_error = None;
        while let Ok(item) = receiver.recv() {
            let chunk = match item {
                Ok(chunk) => chunk,
                Err(error) => {
                    gpu_error = Some(error);
                    break;
                }
            };
            let started = Instant::now();
            let input = backend.tensor_from_f32(&chunk.hidden.data, chunk.rows, cfg.hidden_size)?;
            let hidden = run_gpu_layers(&gpu_runtime, plan.gpu_layer_start, &gpu_layers, &mut gpu_cache, &mut gpu_recurrent, chunk.position, input)?;
            backend.synchronize().map_err(|error| format!("CUDA prefill synchronize: {error:?}"))?;
            eprintln!("[qwen38-hybrid-prefill] position={} rows={} cpu={:.3}s cuda={:.3}s", chunk.position, chunk.rows, chunk.cpu_wall.as_secs_f64(), started.elapsed().as_secs_f64());
            last = Some(hidden);
        }
        drop(receiver);
        let cpu_state = producer.join().map_err(|_| "CPU prefill 线程 panic".to_owned())??;
        if let Some(error) = gpu_error {
            return Err(error);
        }
        Ok((cpu_state, last.ok_or_else(|| "prefill 没有产出 hidden".to_owned())?))
    })?;
    eprintln!("[qwen38-hybrid-prefill] tokens={} chunks={} wall={:.3}s", tokens.len(), tokens.len().div_ceil(options.prefill_chunk_size), prefill_started.elapsed().as_secs_f64());

    let last_row = (tokens.len() - 1) % options.prefill_chunk_size;
    let mut hidden = backend.select_row(&final_hidden, last_row).map_err(|error| format!("选择 prefill 末行: {error:?}"))?;
    let cpu_runtime = Qwen36Runtime::new(&cpu, &cfg, &cpu_layers, &rope, attention);
    let decode_pool = options.decode_cpu_threads.map(|threads| rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|error| format!("创建 decode CPU 线程池: {error}"))).transpose()?;
    let decode_started = Instant::now();
    let mut output = GenerationOutput::new(stops);
    for step in 0..max_tokens {
        if cancellation.load(Ordering::Acquire) {
            output.cancel();
            break;
        }
        let round_started = Instant::now();
        let normalized = backend.gemma_rmsnorm(&hidden, &final_norm, cfg.rms_norm_eps).map_err(|error| format!("output norm: {error:?}"))?;
        let logits = backend.linear(&normalized, &output_head).map_err(|error| format!("output head: {error:?}"))?;
        let token = backend.argmax_excluding(&logits, &[cfg.image_token_id, cfg.video_token_id]).map_err(|error| format!("argmax: {error:?}"))?;
        if cfg.eos_token_ids.contains(&token) {
            output.stop();
            break;
        }
        let bytes = crate::runtime::tool::decode_output_token(&detokenizer, token)?;
        if !output.push(&bytes, |chunk| on_token(token, chunk)) {
            break;
        }
        let sample_wall = round_started.elapsed();
        if output.completion_tokens() == max_tokens {
            eprintln!("[qwen38-hybrid-decode] step={} position={} token={} sample={:.3}s cpu=0.000s cuda=0.000s round={:.3}s", step + 1, tokens.len() + step, token, sample_wall.as_secs_f64(), round_started.elapsed().as_secs_f64());
            break;
        }
        let position = tokens.len() + step;
        if position >= options.max_sequence_length {
            eprintln!("[qwen38-hybrid-decode] step={} position={} token={} sample={:.3}s cpu=0.000s cuda=0.000s round={:.3}s", step + 1, tokens.len() + step, token, sample_wall.as_secs_f64(), round_started.elapsed().as_secs_f64());
            break;
        }
        let cpu_started = Instant::now();
        let embedding = reader.embedding_rows("token_embd.weight", &[token], cfg.hidden_size, cfg.vocab_size)?;
        let input = CpuTensor { data: embedding, rows: 1, cols: cfg.hidden_size };
        let cpu_hidden = if let Some(pool) = &decode_pool {
            pool.install(|| run_cpu_layers(&cpu_runtime, &cpu_layers, &mut cpu_cache, &mut cpu_recurrent, position, input))?
        } else {
            run_cpu_layers(&cpu_runtime, &cpu_layers, &mut cpu_cache, &mut cpu_recurrent, position, input)?
        };
        let cpu_wall = cpu_started.elapsed();
        let cuda_started = Instant::now();
        let gpu_input = backend.tensor_from_f32(&cpu_hidden.data, 1, cfg.hidden_size)?;
        hidden = run_gpu_layers(&gpu_runtime, plan.gpu_layer_start, &gpu_layers, &mut gpu_cache, &mut gpu_recurrent, position, gpu_input)?;
        backend.synchronize().map_err(|error| format!("CUDA decode synchronize: {error:?}"))?;
        let cuda_wall = cuda_started.elapsed();
        eprintln!(
            "[qwen38-hybrid-decode] step={} position={} token={} sample={:.3}s cpu={:.3}s cuda={:.3}s round={:.3}s",
            step + 1,
            tokens.len() + step,
            token,
            sample_wall.as_secs_f64(),
            cpu_wall.as_secs_f64(),
            cuda_wall.as_secs_f64(),
            round_started.elapsed().as_secs_f64(),
        );
    }
    output.finish(|chunk| on_token(0, chunk));
    println!();
    let elapsed = decode_started.elapsed().as_secs_f64();
    eprintln!(
        "[qwen38-hybrid-summary] generated={} wall={elapsed:.3}s throughput={:.3} tok/s boundary_bytes/token={} cpu_state={:.2}MiB cuda_state={:.2}MiB cuda_kv={} {:.2}MiB",
        output.completion_tokens(),
        output.completion_tokens() as f64 / elapsed.max(f64::EPSILON),
        cfg.hidden_size * 4,
        cpu_recurrent.allocated_bytes() as f64 / MIB as f64,
        gpu_recurrent.allocated_bytes() as f64 / MIB as f64,
        gpu_cache.format(),
        gpu_cache.allocated_bytes() as f64 / MIB as f64,
    );
    Ok(output.summary(tokens.len()))
}

pub fn run(backend: &CudaContext, model_path: &Path, prompt: &str, max_tokens: usize, options: HybridCudaOptions) -> Result<(), Box<dyn std::error::Error>> {
    let cancelled = AtomicBool::new(false);
    let rendered_prompt = chat_prompt(prompt);
    generate(backend, model_path, &rendered_prompt, max_tokens, options, &[], &cancelled, &mut |_, text| {
        print!("{text}");
        true
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_64k_each_gqa_layer_uses_q8g64_budget() {
        let cfg = Qwen36Config::standard_27b();
        assert_eq!(layer_kv_bytes(&cfg, 3, 65_536, false).unwrap(), 136 * MIB);
        assert_eq!(layer_kv_bytes(&cfg, 3, 65_536, true).unwrap(), 256 * MIB);
        assert_eq!(layer_kv_bytes(&cfg, 2, 65_536, false).unwrap(), 0);
    }
}
