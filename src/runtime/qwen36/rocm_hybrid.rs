//! Qwen3.8 GGUF × CPU/ROCm 连续分层执行。
//!
//! CPU 执行前缀层，ROCm 执行尽可能长的后缀层并持有输出头。每个 prefill
//! chunk / decode token 只在层边界上传一次 hidden；两侧 KV 与 GDN state
//! 跟随各自负责的层，不在 token 热路径换入换出。

use std::{
    path::Path,
    sync::mpsc::sync_channel,
    time::{Duration, Instant},
};

use crate::{
    attention::{gated_delta_net::GatedDeltaNetState, hybrid::HybridAttentionOptions, rope::RopeTable},
    backend::{
        Backend, BackendResources, StageExecutionBackend,
        cpu::{CpuContext, CpuGatedDeltaNetStorage, CpuKvCache},
        rocm::{RocmContext, RocmGatedDeltaNetStorage, RocmKvCache, RocmTensor},
    },
    kernel::cpu::CpuTensor,
    runtime::qwen36::{Qwen36Config, Qwen36Runtime, Qwen36RuntimeLayer, chat_prompt, prepare_qwen36_gguf_layer, prepare_qwen36_gguf_output},
    weight::container::gguf::{GgufReader, GgufTensorInfo},
};

const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;
const MIN_WORKSPACE_BYTES: usize = 512 * MIB;

#[derive(Clone, Copy, Debug)]
pub struct HybridRocmOptions {
    /// None 按当前空闲显存和 64K KV 准入预算自动选择；Some(n) 类比 llama.cpp -ngl n。
    pub gpu_layers: Option<usize>,
    pub max_sequence_length: usize,
    pub prefill_chunk_size: usize,
    pub vram_reserve_bytes: usize,
    pub precise_gqa_prefill: bool,
}

#[derive(Clone, Debug)]
pub struct HybridRocmPlan {
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

/// ROCm 对 K-quant 保留原始 GGUF block，kernel 就地解码 scale/min。
fn rocm_tensor_bytes(tensor: &GgufTensorInfo) -> Result<usize, String> {
    let bytes = match tensor.tensor_type.0 {
        12 | 13 | 14 => tensor.bytes,
        0 => tensor.elements().checked_mul(4).ok_or("F32 tensor 大小溢出")?,
        1 | 30 => tensor.elements().checked_mul(2).ok_or("F16/BF16 tensor 大小溢出")?,
        8 => tensor.bytes,
        other => return Err(format!("ROCm GPU 层不支持 GGUF {}(type={other}) tensor {}；Qwen3.8 混合路径当前要求 Q4_K/Q5_K/Q6_K/Q8_0/F16/F32", tensor.tensor_type.name(), tensor.name)),
    };
    Ok(bytes)
}

fn tensors_bytes<'a>(mut tensors: impl Iterator<Item = &'a GgufTensorInfo>) -> Result<usize, String> {
    tensors.try_fold(0usize, |total, tensor| checked_add(total, rocm_tensor_bytes(tensor)?, "resident weight"))
}

fn layer_weight_bytes(reader: &GgufReader, layer: usize) -> Result<usize, String> {
    let prefix = format!("blk.{layer}.");
    tensors_bytes(reader.tensors().iter().filter(|tensor| tensor.name.starts_with(&prefix)))
}

fn output_weight_bytes(reader: &GgufReader) -> Result<usize, String> {
    let output = if reader.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    tensors_bytes(["output_norm.weight", output].into_iter().map(|name| reader.tensor(name).ok_or_else(|| format!("GGUF 缺少 {name}"))).collect::<Result<Vec<_>, _>>()?.into_iter())
}

fn layer_state_bytes(cfg: &Qwen36Config, layer: usize) -> Result<usize, String> {
    if (layer + 1).is_multiple_of(cfg.full_attention_interval) {
        return Ok(0);
    }
    let spec = cfg.gated_delta_net_spec();
    spec.recurrent_elements().checked_add(spec.conv_state_elements()).and_then(|elements| elements.checked_mul(4)).ok_or_else(|| "GDN state 大小溢出".to_owned())
}

fn layer_kv_bytes(cfg: &Qwen36Config, layer: usize, max_sequence_length: usize) -> Result<usize, String> {
    if !(layer + 1).is_multiple_of(cfg.full_attention_interval) {
        return Ok(0);
    }
    max_sequence_length.checked_mul(cfg.num_kv_heads).and_then(|value| value.checked_mul(cfg.head_dim)).and_then(|value| value.checked_mul(2)).and_then(|value| value.checked_mul(4)).ok_or_else(|| "GQA KV 大小溢出".to_owned())
}

fn workspace_bytes(cfg: &Qwen36Config, prefill_chunk_size: usize) -> Result<usize, String> {
    // gate/up + attention/output 的并发临时量；另保留固定下限吸收 HIPRTC、pool 与 logits。
    let activation = prefill_chunk_size.checked_mul(cfg.intermediate_size).and_then(|value| value.checked_mul(3)).and_then(|value| value.checked_mul(4)).ok_or_else(|| "prefill workspace 大小溢出".to_owned())?;
    Ok(activation.max(MIN_WORKSPACE_BYTES))
}

pub fn plan_hybrid_layers(reader: &GgufReader, cfg: &Qwen36Config, options: HybridRocmOptions, free_bytes: usize) -> Result<HybridRocmPlan, String> {
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
    let base = checked_add(output_bytes, workspace, "output/workspace")?;
    if base > budget {
        return Err(format!("输出头+workspace 预计 {:.2}GiB，扣除保留后仅 {:.2}GiB", base as f64 / GIB as f64, budget as f64 / GIB as f64));
    }

    let wanted = options.gpu_layers.unwrap_or(cfg.num_layers);
    let mut weights = output_bytes;
    let mut kv = 0usize;
    let mut state = 0usize;
    let mut accepted = 0usize;
    let mut full_attention_layers = 0usize;
    for layer in (cfg.num_layers - wanted..cfg.num_layers).rev() {
        let layer_weight = layer_weight_bytes(reader, layer)?;
        let layer_kv = layer_kv_bytes(cfg, layer, options.max_sequence_length)?;
        let layer_state = layer_state_bytes(cfg, layer)?;
        let candidate_weights = checked_add(weights, layer_weight, "weight")?;
        let candidate_kv = checked_add(kv, layer_kv, "KV")?;
        let candidate_state = checked_add(state, layer_state, "state")?;
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
    if options.gpu_layers.is_some() && accepted != wanted {
        return Err("显式 gpu_layers 规划不完整".to_owned());
    }
    let total = [weights, kv, state, workspace].into_iter().try_fold(0usize, |total, value| checked_add(total, value, "plan total"))?;
    if total > budget {
        return Err(format!("gpu_layers={accepted} 的 64K 准入预计 {:.2}GiB，扣除保留后预算 {:.2}GiB；减少 gpu_layers 或 vram_reserve", total as f64 / GIB as f64, budget as f64 / GIB as f64));
    }
    Ok(HybridRocmPlan {
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

fn run_gpu_layers(
    runtime: &Qwen36Runtime<'_, RocmContext>,
    first_layer: usize,
    layers: &[Qwen36RuntimeLayer<crate::backend::rocm::RocmWeight>],
    cache: &mut RocmKvCache,
    recurrent: &mut GatedDeltaNetState<RocmGatedDeltaNetStorage>,
    position: usize,
    mut hidden: RocmTensor,
) -> Result<RocmTensor, String> {
    for (offset, weights) in layers.iter().enumerate() {
        let layer = first_layer + offset;
        hidden = runtime.prepared_layer(cache, recurrent, position, layer, weights, hidden).map_err(|error| format!("ROCm L{layer}: {error:?}"))?;
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
pub fn run(backend: &RocmContext, model_path: &Path, prompt: &str, max_tokens: usize, options: HybridRocmOptions) -> Result<(), Box<dyn std::error::Error>> {
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
    let tokens = tokenizer.tokenize(chat_prompt(prompt).as_bytes());
    if tokens.is_empty() || tokens.len() >= options.max_sequence_length {
        return Err(format!("prompt tokens={}，max_sequence_length={}", tokens.len(), options.max_sequence_length).into());
    }

    let (free, total) = (backend.stage_available_bytes()?, backend.stage_total_bytes()?);
    let plan = plan_hybrid_layers(&reader, &cfg, options, free)?;
    eprintln!(
        "[qwen38-hybrid-plan] device={} vram={:.2}/{:.2}GiB cpu_layers=0..{} gpu_layers={}..{} full_attention_gpu={} estimate: weights={:.2}GiB kv64k={:.2}GiB state={:.2}GiB workspace={:.2}GiB total={:.2}GiB reserve={:.2}GiB",
        crate::backend::rocm::device_name(backend),
        free as f64 / GIB as f64,
        total as f64 / GIB as f64,
        plan.gpu_layer_start,
        plan.gpu_layer_start,
        cfg.num_layers,
        plan.full_attention_layers,
        plan.estimated_weight_bytes as f64 / GIB as f64,
        plan.estimated_kv_bytes as f64 / GIB as f64,
        plan.estimated_state_bytes as f64 / GIB as f64,
        plan.estimated_workspace_bytes as f64 / GIB as f64,
        plan.estimated_total_bytes as f64 / GIB as f64,
        options.vram_reserve_bytes as f64 / GIB as f64,
    );

    let cpu = CpuContext;
    let prepare_started = Instant::now();
    let cpu_layers = (0..plan.gpu_layer_start).map(|layer| prepare_qwen36_gguf_layer(&cpu, &reader, &cfg, layer).map_err(|error| format!("准备 CPU L{layer}: {error:?}"))).collect::<Result<Vec<_>, _>>()?;
    let mut gpu_layers = Vec::with_capacity(plan.gpu_layers);
    for layer in plan.gpu_layer_start..cfg.num_layers {
        gpu_layers.push(prepare_qwen36_gguf_layer(backend, &reader, &cfg, layer).map_err(|error| format!("准备 ROCm L{layer}: {error:?}"))?);
        eprintln!("[qwen38-hybrid-load] gpu_layer={}/{}", layer + 1, cfg.num_layers);
    }
    let (final_norm, output_head) = prepare_qwen36_gguf_output(backend, &reader).map_err(|error| format!("准备 ROCm output: {error:?}"))?;
    backend.synchronize()?;
    let (free_after_load, _) = (backend.stage_available_bytes()?, backend.stage_total_bytes()?);
    eprintln!("[qwen38-hybrid-load] wall={:.3}s vram_weights_used={:.2}GiB free={:.2}GiB", prepare_started.elapsed().as_secs_f64(), (free - free_after_load) as f64 / GIB as f64, free_after_load as f64 / GIB as f64);

    let rope = RopeTable::precompute(options.max_sequence_length, cfg.rope_dim, cfg.rope_theta);
    let attention = HybridAttentionOptions { precise_prefill: options.precise_gqa_prefill };
    let mut gpu_cache = RocmKvCache::with_capacity(cfg.num_layers, options.max_sequence_length);
    let mut gpu_recurrent = GatedDeltaNetState::<RocmGatedDeltaNetStorage>::new(cfg.num_layers, cfg.gated_delta_net_spec()).map_err(|error| format!("创建 ROCm GDN state: {error:?}"))?;
    let gpu_runtime = Qwen36Runtime::new(backend, &cfg, &gpu_layers, &rope, attention);

    // CPU 前缀与前一 chunk 的 GPU 后缀流水重叠；channel=1 限制 activation 生命周期。
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
                sender.send(Ok(PrefillChunk { position, rows: chunk_tokens.len(), hidden, cpu_wall: started.elapsed() })).map_err(|_| "ROCm prefill 已提前结束".to_owned())?;
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
            let input = backend.tensor_from_f32(chunk.hidden.data, chunk.rows, cfg.hidden_size)?;
            let hidden = run_gpu_layers(&gpu_runtime, plan.gpu_layer_start, &gpu_layers, &mut gpu_cache, &mut gpu_recurrent, chunk.position, input)?;
            backend.synchronize().map_err(|error| format!("ROCm prefill synchronize: {error:?}"))?;
            eprintln!("[qwen38-hybrid-prefill] position={} rows={} cpu={:.3}s gpu={:.3}s", chunk.position, chunk.rows, chunk.cpu_wall.as_secs_f64(), started.elapsed().as_secs_f64());
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
    let decode_started = Instant::now();
    let mut generated = 0usize;
    for step in 0..max_tokens {
        let round_started = Instant::now();
        let normalized = backend.gemma_rmsnorm_f32(&hidden, &final_norm, cfg.rms_norm_eps).map_err(|error| format!("output norm: {error:?}"))?;
        let logits = backend.linear(&normalized, &output_head).map_err(|error| format!("output head: {error:?}"))?;
        let token = backend.argmax_excluding(&logits, &[cfg.image_token_id, cfg.video_token_id]).map_err(|error| format!("argmax: {error:?}"))?;
        crate::runtime::generation::write_token(&detokenizer, token, true)?;
        generated += 1;
        eprintln!("[qwen38-hybrid-decode] step={} position={} wall={:.3}s", step + 1, tokens.len() + step, round_started.elapsed().as_secs_f64());
        if cfg.eos_token_ids.contains(&token) || step + 1 == max_tokens {
            break;
        }
        let position = tokens.len() + step;
        if position >= options.max_sequence_length {
            break;
        }
        let embedding = reader.embedding_rows("token_embd.weight", &[token], cfg.hidden_size, cfg.vocab_size)?;
        let input = CpuTensor { data: embedding, rows: 1, cols: cfg.hidden_size };
        let cpu_hidden = run_cpu_layers(&cpu_runtime, &cpu_layers, &mut cpu_cache, &mut cpu_recurrent, position, input)?;
        let gpu_input = backend.tensor_from_f32(cpu_hidden.data, 1, cfg.hidden_size)?;
        hidden = run_gpu_layers(&gpu_runtime, plan.gpu_layer_start, &gpu_layers, &mut gpu_cache, &mut gpu_recurrent, position, gpu_input)?;
    }
    println!();
    let elapsed = decode_started.elapsed().as_secs_f64();
    eprintln!(
        "[qwen38-hybrid-summary] generated={} wall={elapsed:.3}s throughput={:.3} tok/s boundary_bytes/token={} cpu_state={:.2}MiB gpu_state={:.2}MiB",
        generated,
        generated as f64 / elapsed.max(f64::EPSILON),
        cfg.hidden_size * 4,
        cpu_recurrent.allocated_bytes() as f64 / MIB as f64,
        gpu_recurrent.allocated_bytes() as f64 / MIB as f64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_64k_each_gqa_layer_uses_half_gib_f32_kv() {
        let cfg = Qwen36Config::standard_27b();
        assert_eq!(layer_kv_bytes(&cfg, 3, 65_536).unwrap(), 512 * MIB);
        assert_eq!(layer_kv_bytes(&cfg, 2, 65_536).unwrap(), 0);
    }

    #[test]
    fn qwen38_gdn_state_budget_matches_layer_shape() {
        let cfg = Qwen36Config::standard_27b();
        let spec = cfg.gated_delta_net_spec();
        assert_eq!(layer_state_bytes(&cfg, 0).unwrap(), (spec.recurrent_elements() + spec.conv_state_elements()) * 4);
    }
}
