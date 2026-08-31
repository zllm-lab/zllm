//! GLM-5.2 × ROCm standalone/stage 组合。

use crate::runtime::{generation, pipeline, stage_artifact};

use crate::runtime::glm52::rocm_swap::{Glm52CacheIdentity, Glm52CacheSnapshot, Glm52MtpCache, Glm52SwapStore, download_glm52_session, upload_glm52_session};

use std::{
    collections::{HashMap, VecDeque},
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use crate::{
    attention::rope::RopeTable,
    attention::{AttentionSpec, mla::MlaSpec},
    backend::rocm::{RocmContext, RocmDsaSelection, RocmDsaState, RocmKvCache, RocmPrefillExperts, RocmTensor, RocmWeight},
    backend::{Backend, BackendResources, LinearWeight, SegmentedTensorBackend, StageExecutionBackend},
    config::{BackendConfig, Glm52StageDiagnosticsConfig, KvCacheFormat, RuntimeProcessConfig, StageModelConfig, StageTransportConfig},
    moe::{
        UncachedMoeState,
        expert_predictor::{ExpertPredictorConfig, ExpertPredictorWeights},
    },
    runtime::{
        Model,
        glm52::{Glm52, Glm52Config},
    },
    runtime::{
        expert_pipeline::ExpertDecodePipeline,
        generation_guard::{GenerationGuard, TokenFenceProgram},
        glm52::dspark_rocm::attach_dspark_projections,
        glm52::stage::{
            Glm52PrefillLayer as RuntimeGlm52PrefillLayer, Glm52StageState as RuntimeGlm52StageState, build_glm52_stage_states, build_pp_prefill_chunks, drive_glm52_stream_stage_pipeline_stateful, last_token_row as last_bf16_row,
            prepare_prefill_layers, run_glm52_stage_pipeline_stateful, run_glm52_stream_stage_pipeline_stateful,
        },
        glm52::tool::Glm52ToolFence,
        glm52::{
            Glm52DecodeLayer, Glm52Mtp, Glm52OutputHead, Glm52PrefillSegment, glm52_decode_layers, glm52_dense_prefill_layer, glm52_moe_prefill_layer, glm52_mtp_cache_segmented, glm52_mtp_decode, glm52_mtp_prefill_segmented,
            glm52_mtp_token_ids_fenced, glm52_mtp_token_output, glm52_prefill_stage, glm52_sampled_token_ids, glm52_sampled_token_ids_fenced, glm52_token_output, prepare_glm52_decode_layers, prepare_glm52_mtp_ct, prepare_glm52_mtp_gguf,
            prepare_glm52_output_head, prepare_glm52_output_head_quantized, print_expert_cache, print_token,
        },
        output::{DraftHead, SamplingConfig, SamplingState, load_draft_vocabulary, normalized_draft_token_ids_fenced, prepare_draft_head},
        prefill::{StageSchedulerOutput, run_token_chunk_stage_pipeline},
        speculative::verify_samples,
    },
    server::iroh::IrohConfig,
    server::stage_transport::{RequestId, StageDeviceMemory, StageMessage, StageTransport},
    tokenizer::{Detokenizer, Tokenizer},
    weight::{Glm52Weights, LmHeadQuantization, ResidentWeightQuantization, expert_source::Glm52ExpertSources},
};

use pipeline::{PIPELINE_ALPN, PipelineMessage, PipelineMessageKind, read_pipeline_message, write_pipeline_message};
use stage_artifact::{StageArtifactHeader, StageArtifactReader, StageArtifactWriter};
type Glm52StageState = RuntimeGlm52StageState<RocmContext>;
type PrefillLayer = RuntimeGlm52PrefillLayer<RocmWeight>;

pub(super) fn prepare_glm52_rope_resident(contexts: &[RocmContext], rope: &RopeTable) -> Result<(), String> {
    let started = Instant::now();
    let row_width = rope.rotary_dim / 2;
    for context in contexts {
        let device_started = Instant::now();
        context.activate()?;
        let _ = crate::kernel::rocm::hip::resident_rope_tables(context.device_id(), &rope.cos, &rope.sin, row_width, std::iter::once(0))?;
        eprintln!("[glm52-rope-resident] device={} rows={} wall={:.3}s", context.device_id(), rope.seq_len, device_started.elapsed().as_secs_f64());
    }
    eprintln!("[glm52-rope-resident] devices={} rows={} total={:.3}s", contexts.len(), rope.seq_len, started.elapsed().as_secs_f64());
    Ok(())
}

pub(super) struct RocmMtpRuntime {
    pub(super) backend: RocmContext,
    pub(super) weights: Glm52Mtp<RocmWeight>,
    pub(super) experts: RocmPrefillExperts,
    pub(super) draft_head: Option<DraftHead<RocmWeight>>,
}

pub(super) struct RocmMtpSession {
    pub(super) cache: RocmKvCache,
    pub(super) dsa: RocmDsaState,
    /// 已提交到 MTP cache 的逻辑行数，恒等于 target position - 1。
    pub(super) position: usize,
    /// target 最后一行真实 hidden，下一次移位 catch-up/draft 使用。
    pub(super) pending_hidden: Option<RocmTensor>,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) max_decode: usize,
    pub(super) draft_tokens: usize,
    pub(super) verify_inputs: Vec<u32>,
    /// resident state 可以跨请求保存，但只有收到本轮 MtpContext 后才参与计算。
    pub(super) active: bool,
}

impl RocmMtpSession {
    pub(super) fn fresh(cfg: &Glm52Config, max_seq_len: usize) -> Result<Self, String> {
        let layers = cfg.layer_count + cfg.mtp_layer_count;
        Ok(Self {
            cache: RocmKvCache::with_capacity(layers, max_seq_len),
            dsa: RocmDsaState::new(layers, max_seq_len, cfg.index_head_dim, cfg.index_top_k)?,
            position: 0,
            pending_hidden: None,
            prompt_tokens: Vec::new(),
            max_decode: 0,
            draft_tokens: 0,
            verify_inputs: Vec::new(),
            active: false,
        })
    }

    pub(super) fn begin_request(&mut self, target_position: usize, prompt_tokens: Vec<u32>, max_decode: usize, draft_tokens: usize) -> Result<(), String> {
        if max_decode == 0 || draft_tokens == 0 || target_position > prompt_tokens.len() {
            return Err(format!("MTP context 非法: target_position={target_position} prompt={} max_decode={max_decode} drafts={draft_tokens}", prompt_tokens.len()));
        }
        let expected = target_position.saturating_sub(1);
        if self.position != expected || (target_position > 0 && self.pending_hidden.is_none()) {
            return Err(format!("MTP resident 状态错位: position={} expected={expected} pending_hidden={}", self.position, self.pending_hidden.is_some()));
        }
        if !self.prompt_tokens.is_empty() && !prompt_tokens.starts_with(&self.prompt_tokens) {
            return Err(format!("MTP cache token 前缀不匹配: cached={} prompt={}", self.prompt_tokens.len(), prompt_tokens.len()));
        }
        self.prompt_tokens = prompt_tokens;
        self.max_decode = max_decode;
        self.draft_tokens = draft_tokens;
        self.verify_inputs.clear();
        self.active = true;
        Ok(())
    }

    pub(super) fn deactivate(&mut self) {
        self.max_decode = 0;
        self.draft_tokens = 0;
        self.verify_inputs.clear();
        self.active = false;
    }

    pub(super) fn truncate_to_target(&mut self, target_position: usize) -> Result<(), crate::backend::BackendError> {
        let rows = target_position.saturating_sub(1);
        self.cache.truncate_rows(rows)?;
        self.dsa.truncate_rows(rows)?;
        self.position = rows;
        Ok(())
    }

    /// 只允许在 terminal 边界保存；本轮 draft/verify 临时态由下一请求重建。
    pub(super) fn snapshot(&self, context: &RocmContext) -> Result<Glm52MtpCache, String> {
        if self.active {
            return Err("MTP active session 不能持久化".to_owned());
        }
        let pending_hidden = self.pending_hidden.as_ref().ok_or("MTP terminal state 缺少 pending hidden")?;
        context.activate().map_err(|error| format!("激活 MTP ROCm device {}: {error}", context.device_id()))?;
        Ok(Glm52MtpCache {
            position: self.position,
            pending_hidden: context.tensor_to_bf16_bits(pending_hidden).map_err(|error| format!("下载 MTP pending hidden: {error:?}"))?,
            prompt_tokens: self.prompt_tokens.clone(),
            kv: self.cache.download_layers().map_err(|error| format!("下载 MTP KV: {error:?}"))?,
            dsa: self.dsa.download_layers().map_err(|error| format!("下载 MTP DSA: {error:?}"))?,
        })
    }

    pub(super) fn restore(snapshot: Glm52MtpCache, context: &RocmContext, cfg: &Glm52Config, max_seq_len: usize, reserved_rows: usize) -> Result<Self, String> {
        let layers = cfg.layer_count + cfg.mtp_layer_count;
        if snapshot.pending_hidden.len() != cfg.hidden_size || snapshot.kv.len() > layers || snapshot.dsa.len() > layers {
            return Err(format!("MTP snapshot shape 非法: hidden={}/{} kv={}/{} dsa={}/{}", snapshot.pending_hidden.len(), cfg.hidden_size, snapshot.kv.len(), layers, snapshot.dsa.len(), layers));
        }
        context.activate().map_err(|error| format!("激活 MTP ROCm device {}: {error}", context.device_id()))?;
        let mut session = Self::fresh(cfg, max_seq_len)?;
        session.cache.upload_layers(context, &snapshot.kv, reserved_rows).map_err(|error| format!("恢复 MTP KV: {error:?}"))?;
        session.dsa.upload_layers(context, &snapshot.dsa, reserved_rows).map_err(|error| format!("恢复 MTP DSA: {error:?}"))?;
        session.position = snapshot.position;
        session.pending_hidden = Some(context.tensor_from_bf16_bits(snapshot.pending_hidden, 1, cfg.hidden_size).map_err(|error| format!("恢复 MTP pending hidden: {error:?}"))?);
        session.prompt_tokens = snapshot.prompt_tokens;
        session.deactivate();
        Ok(session)
    }
}

/// tail stage 的在途 session;states 被 pipeline 取走时为空 Vec。
#[path = "rocm_tail.rs"]
mod rocm_tail;
use rocm_tail::*;
pub(super) use rocm_tail::{RocmMtpCatchUp, RocmMtpDraftBatch, mtp_catch_up_batch, mtp_draft_batch};
enum StageLink {
    Listen(IrohConfig),
    Connect { ticket: String, iroh: IrohConfig },
}

struct RocmEntry {
    args: Args,
    stage_start: usize,
    stage_end: usize,
    stage_transport: Option<StageLink>,
    pipeline_chunk_size: usize,
    preload_experts: bool,
    cooperative_expert_pairs: bool,
    preload_layers_per_device: Option<usize>,
    max_concurrency: usize,
    diagnostics: Glm52StageDiagnosticsConfig,
    kv_cache_format: KvCacheFormat,
    mtp_enabled: bool,
    mtp_draft_tokens: usize,
    mtp_draft_vocabulary: Option<PathBuf>,
    dspark_directory: Option<PathBuf>,
    dspark_weight_quantization: ResidentWeightQuantization,
    lm_head_quantization: LmHeadQuantization,
    scheduling: crate::config::Glm52SchedulingConfig,
}

impl RocmEntry {
    fn from_config(config: RuntimeProcessConfig) -> Result<(crate::config::RocmBackendConfig, Self), Box<dyn std::error::Error>> {
        match config {
            RuntimeProcessConfig::Node(_) | RuntimeProcessConfig::Standalone(_) => Err("kind: node/standalone 由 main 分支处理,不应进入 from_config".into()),
            RuntimeProcessConfig::Stage(config) => {
                let BackendConfig::Rocm(backend) = config.backend else {
                    return Err("zllm-rt-rocm 只接受 kind: rocm backend".into());
                };
                let StageModelConfig::Glm52(model) = config.model else { unreachable!("入口已按模型分发") };
                let generation = model.generation.unwrap_or_else(|| crate::config::StageGenerationConfig { prompt: String::new(), decode_steps: 0 });
                let transport = match config.transport {
                    StageTransportConfig::Listen { iroh } => StageLink::Listen(iroh.runtime()?),
                    StageTransportConfig::Connect { ticket, iroh } => StageLink::Connect { ticket, iroh: iroh.runtime()? },
                };
                let lm_head_quantization = model.lm_head_quantization;
                let execution = model.execution;
                let args = Args {
                    model_dir: model.weights_directory,
                    prompt: generation.prompt,
                    max_seq_len: model.max_sequence_length,
                    decode_steps: generation.decode_steps,
                    expert_cache_gib: None,
                    expert_prefetch_count: None,
                    nvfp4_root: model.nvfp4_directory,
                    ct_root: model.compressed_tensors_directory,
                    gguf_root: model.gguf_directory,
                    prefill_devices: Some(backend.devices.clone()),
                    prefill_layer_ends: Some(model.layers.device_layer_ends),
                    start_layer: None,
                    end_layer: None,
                    next_node: None,
                    decode_node: None,
                    cache_dir: model.cache_directory,
                    pipeline_iroh: None,
                };
                Ok((
                    backend,
                    Self {
                        args,
                        stage_start: model.layers.start,
                        stage_end: model.layers.end,
                        stage_transport: Some(transport),
                        pipeline_chunk_size: execution.prefill_chunk_size,
                        preload_experts: execution.preload_experts,
                        cooperative_expert_pairs: execution.cooperative_expert_pairs,
                        preload_layers_per_device: execution.preload_layers_per_device,
                        max_concurrency: execution.max_concurrency,
                        diagnostics: execution.diagnostics,
                        kv_cache_format: execution.kv_cache_format,
                        mtp_enabled: execution.mtp,
                        mtp_draft_tokens: execution.mtp_draft_tokens,
                        mtp_draft_vocabulary: execution.mtp_draft_vocabulary,
                        dspark_directory: execution.dspark_directory,
                        dspark_weight_quantization: execution.dspark_weight_quantization,
                        lm_head_quantization,
                        scheduling: execution.scheduling,
                    },
                ))
            }
        }
    }
}

pub fn run(ctx: &RocmContext, config: RuntimeProcessConfig) -> Result<(), Box<dyn std::error::Error>> {
    let (_, entry) = RocmEntry::from_config(config)?;
    run_glm52_entry(ctx, entry)
}

fn run_glm52_entry(ctx: &RocmContext, entry: RocmEntry) -> Result<(), Box<dyn std::error::Error>> {
    let RocmEntry {
        args: parsed,
        stage_start,
        stage_end,
        stage_transport,
        pipeline_chunk_size,
        preload_experts,
        cooperative_expert_pairs,
        preload_layers_per_device,
        max_concurrency,
        diagnostics,
        kv_cache_format,
        mtp_enabled,
        mtp_draft_tokens,
        mtp_draft_vocabulary,
        dspark_directory,
        dspark_weight_quantization,
        lm_head_quantization,
        scheduling,
        ..
    } = entry;
    let args = parsed;
    let scheduler_policy = crate::runtime::glm52::stage::Glm52SchedulerPolicy {
        execution_slots: scheduling.execution_slots,
        decode_execution_slots: scheduling.decode_execution_slots,
        pipeline_work_window: scheduling.pipeline_work_window,
        prefill_admission_burst: scheduling.prefill_admission_burst,
        decode_batch_limit: scheduling.decode_batch_limit,
        prefill_batch_limit: scheduling.prefill_batch_limit,
        profile_completion: scheduling.profile_completion,
    };
    let profile_completion = scheduling.profile_completion;
    if diagnostics.trace_stage_events {
        crate::runtime::prefill_scheduler::enable_stage_event_trace();
    }
    if profile_completion {
        crate::kernel::rocm::hip::enable_device_profile();
    }
    let chunk_policy = crate::runtime::prefill::AdaptiveChunkPolicy {
        initial_chunk_size: pipeline_chunk_size,
        append_chunk_size: scheduling.append_prefill_chunk_size,
        long_context_threshold_tokens: scheduling.long_prefill_threshold_tokens,
        long_context_chunk_size: scheduling.long_prefill_chunk_size,
    };
    let cfg = Glm52Config::standard();
    if stage_start >= stage_end || stage_end > cfg.layer_count {
        return Err(format!("GLM stage 必须满足 0 <= start < end <= {}，实际 {stage_start}..{stage_end}", cfg.layer_count).into());
    }
    let stage_listen = matches!(stage_transport, Some(StageLink::Listen(_)));
    let stage_ticket = match &stage_transport {
        Some(StageLink::Connect { ticket, .. }) => Some(ticket.as_str()),
        _ => None,
    };
    let distributed = stage_listen || stage_ticket.is_some();
    if stage_start > 0 && !stage_listen {
        return Err("非首段必须设置 model.transport.listen".into());
    }
    if stage_end < cfg.layer_count && stage_ticket.is_none() {
        return Err("非末段必须设置 model.transport.connect".into());
    }
    if stage_end < cfg.layer_count && args.decode_steps != 0 && !distributed {
        return Err("GLM-5.2 分段执行当前先支持 prefill，请设置 --decode-steps 0".into());
    }
    let tokenizer_path = crate::runtime::glm52::tokenizer_path(&args.model_dir, args.ct_root.as_deref(), args.nvfp4_root.as_deref());
    let tokenizer = Tokenizer::new(&tokenizer_path)?;
    let tokens = if stage_start == 0 {
        let tokens = tokenizer.tokenize(args.prompt.as_bytes());
        if tokens.is_empty() {
            return Err("prompt 不能为空".into());
        }
        if tokens.len() > args.max_seq_len {
            return Err(format!("prompt {} tokens 超过 max_seq_len {}", tokens.len(), args.max_seq_len).into());
        }
        tokens
    } else {
        Vec::new()
    };

    let weights = crate::runtime::glm52::open_weights(&cfg, &args.model_dir, args.ct_root.as_deref(), args.nvfp4_root.as_deref(), args.gguf_root.as_deref())?;
    let model = Glm52::standard();
    let mla = match &model.layer_spec(0)?.attention {
        AttentionSpec::Mla(spec) => spec.clone(),
        _ => return Err("GLM-5.2 dense 层不是 MLA".into()),
    };
    let experts_dir = args.model_dir.join("experts");
    let nvfp4_source = weights.nvfp4_experts().map(|source| source.with_archive_dir(&experts_dir));
    let backend = ctx;
    if let Some(start_layer) = args.start_layer {
        return run_glm52_pipeline_node(ctx, &args, &cfg, &mla, &weights, start_layer);
    }
    if let Some(path) = diagnostics.input_artifact.as_deref() {
        return run_glm52_stage_input(backend, &args, &cfg, &mla, &weights, &tokenizer_path, path);
    }
    let (prefill_contexts, cooperative_peer_contexts, prefill_layer_ends) = match (&args.prefill_devices, &args.prefill_layer_ends) {
        (None, None) if cooperative_expert_pairs => return Err("cooperative_expert_pairs 要求显式配置物理 devices 和每个双卡组的 layer_ends".into()),
        (None, None) => (vec![*ctx], Vec::new(), vec![stage_end - 1]),
        (Some(devices), Some(ends)) => {
            if ends.last().copied() != Some(stage_end - 1) {
                return Err(format!("--prefill-layer-ends 的最后边界必须是 {}", stage_end - 1).into());
            }
            if ends.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err("--prefill-layer-ends 必须严格递增".into());
            }
            if ends.first().is_some_and(|end| *end < stage_start) {
                return Err("--prefill-layer-ends 的首个边界早于当前 stage_start".into());
            }
            if cooperative_expert_pairs {
                if devices.len() < 2 || devices.len() % 2 != 0 || ends.len() != devices.len() / 2 {
                    return Err(format!("cooperative_expert_pairs 要求每两张物理卡对应一个 layer_ends：devices={} layer_ends={}", devices.len(), ends.len()).into());
                }
                let owners = devices.iter().step_by(2).map(|&device| ctx.for_device(device).map_err(|error| format!("ROCm owner device {device} 初始化失败: {error}"))).collect::<Result<Vec<_>, _>>()?;
                let peers = devices.iter().skip(1).step_by(2).map(|&device| ctx.for_device(device).map_err(|error| format!("ROCm cooperative peer device {device} 初始化失败: {error}"))).collect::<Result<Vec<_>, _>>()?;
                (owners, peers, ends.clone())
            } else {
                if devices.len() != ends.len() {
                    return Err("--prefill-devices 与 --prefill-layer-ends 数量必须相等".into());
                }
                let contexts = devices.iter().map(|&device| ctx.for_device(device).map_err(|error| format!("ROCm device {device} 初始化失败: {error}"))).collect::<Result<Vec<_>, _>>()?;
                (contexts, Vec::new(), ends.clone())
            }
        }
        _ => return Err("--prefill-devices 与 --prefill-layer-ends 必须同时提供".into()),
    };
    for pair in prefill_contexts.windows(2) {
        pair[1].enable_peer_access_from(pair[0].device_id()).map_err(|error| format!("初始化 ROCm P2P {} -> {} 失败: {error}", pair[0].device_id(), pair[1].device_id()))?;
    }
    if distributed && pipeline_chunk_size == 0 {
        return Err("distributed stage 的 model.execution.prefill_chunk_size 必须大于 0".into());
    }
    let pipeline_enabled = pipeline_chunk_size > 0 && prefill_contexts.len() > 1 && (args.decode_steps == 0 || distributed);
    // 先发布 ticket，让两端权重预热并行；真正计算仍在连接建立后开始。
    let stage_listener = match &stage_transport {
        Some(StageLink::Listen(iroh)) => Some(StageTransport::bind(iroh.clone()).map_err(|error| format!("启动 stage listener: {error}"))?),
        _ => None,
    };
    crate::kernel::rocm::hip::enable_device_buffer_reuse();
    let resident_started = Instant::now();
    let prefill_layers = prepare_prefill_layers(&prefill_contexts, &prefill_layer_ends, stage_start, stage_end, &cfg, &mla, &weights).map_err(|error| format!("准备 prefill layers: {error:?}"))?;
    eprintln!("[prefill-resident] layers={} wall={:.3}s", prefill_layers.len(), resident_started.elapsed().as_secs_f64(),);
    let rope = RopeTable::precompute(args.max_seq_len, mla.qk_rope_head_dim, mla.rope_theta);
    prepare_glm52_rope_resident(&prefill_contexts, &rope)?;
    let new_prefill_experts = || -> Result<RocmPrefillExperts, Box<dyn std::error::Error>> {
        Ok(if let Some(source) = &nvfp4_source {
            RocmPrefillExperts::nvfp4(source.clone())
        } else if weights.source_is_ct() {
            RocmPrefillExperts::ct(weights.ct_source()?)
        } else if weights.source_is_gguf() {
            RocmPrefillExperts::gguf(weights.gguf_source()?)
        } else {
            RocmPrefillExperts::fp8(&args.model_dir, cfg.expert_intermediate_size, cfg.hidden_size, cfg.expert_count)?
        })
    };
    let expert_state_count = if pipeline_enabled || distributed { prefill_contexts.len() } else { 1 };
    let mut prefill_experts = (0..expert_state_count).map(|_| new_prefill_experts()).collect::<Result<Vec<_>, _>>()?;
    if cooperative_expert_pairs {
        if !distributed || !preload_experts || preload_layers_per_device.is_some() || !weights.source_is_ct() || prefill_contexts.len() != cooperative_peer_contexts.len() {
            return Err("cooperative_expert_pairs 要求 distributed CT stage、preload_experts=true、完整预载且每个逻辑 stage 有一张 peer 卡".into());
        }
        let mut layer_start = stage_start;
        let counts = prefill_layer_ends
            .iter()
            .map(|&layer_end| {
                let count = layer_end + 1 - layer_start;
                layer_start = layer_end + 1;
                count
            })
            .collect::<Vec<_>>();
        for &count in &counts {
            if count > 12 {
                return Err(format!("cooperative expert 双卡组层数 {count} 超过 12 层预算").into());
            }
        }
        for (stage, (experts, &peer)) in prefill_experts.iter_mut().zip(&cooperative_peer_contexts).enumerate() {
            experts.enable_cooperative_peer(peer, 0).map_err(|error| format!("配置 ROCm cooperative expert stage={stage}: {error:?}"))?;
        }
        eprintln!(
            "[glm52-cooperative-experts] physical_devices={} pairs={} logical_stages={} mode=prefill+decode partition=gate-up-row/down-k-half device-route",
            prefill_contexts.len() + cooperative_peer_contexts.len(),
            prefill_contexts.len(),
            prefill_contexts.len(),
        );
    }
    // 分布式 decode 必须直接命中 GPU 常驻专家；否则 prefill 命中集合会让多数层退回 host route。
    if preload_experts && (weights.source_is_ct() || weights.source_is_gguf()) {
        let started = Instant::now();
        let layers_per_device = preload_layers_per_device.unwrap_or(usize::MAX);
        let mut device_layers = vec![0_usize; prefill_contexts.len()];
        let mut resident_layers = 0;
        for (layer_offset, resident) in prefill_layers.iter().enumerate() {
            let layer = stage_start + layer_offset;
            if !matches!(resident, PrefillLayer::Moe(_)) {
                continue;
            }
            let placement = prefill_layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| format!("L{layer} 没有 prefill expert device"))?;
            if device_layers[placement] >= layers_per_device {
                continue;
            }
            let layer_backend = &prefill_contexts[placement];
            layer_backend.activate()?;
            let expert_state = if pipeline_enabled || distributed { placement } else { 0 };
            prefill_experts[expert_state].preload_layer(layer_backend, layer, cfg.expert_count).map_err(|error| format!("预加载 ROCm L{layer} experts 失败: {error:?}"))?;
            device_layers[placement] += 1;
            resident_layers += 1;
        }
        eprintln!("[prefill-expert-resident] layers={resident_layers} experts={} wall={:.3}s", cfg.expert_count, started.elapsed().as_secs_f64(),);
    }
    if distributed {
        let mut states = build_glm52_stage_states(&prefill_contexts, &prefill_layer_ends, stage_start, &cfg, prefill_layers, prefill_experts, args.max_seq_len).map_err(|error| format!("构造 GLM stage states: {error:?}"))?;
        if let Some(root) = dspark_directory.as_deref() {
            attach_dspark_projections(&mut states, root, cfg.layer_count, dspark_weight_quantization).map_err(|error| format!("加载 stage DSpark projections: {error:?}"))?;
        }
        if stage_start == 0 {
            let mut link = if let Some(listener) = stage_listener {
                listener.accept().map_err(|error| format!("接受 stage peer: {error}"))?
            } else if let Some(StageLink::Connect { ticket, iroh }) = &stage_transport {
                StageTransport::connect(ticket, iroh.clone()).map_err(|error| format!("连接 stage peer: {error}"))?
            } else {
                return Err("distributed stage 缺少 transport".into());
            };
            let memory = link.recv()?;
            let StageMessage::DeviceMemory { devices, .. } = memory.message else {
                return Err("front stage 连接后未收到下游 device memory".into());
            };
            eprintln!("[stage-device-memory] downstream_devices={}", devices.len());
            let request_id = match diagnostics.request_id.as_deref() {
                Some(value) => RequestId::parse(value)?,
                None if diagnostics.allow_generated_request_id => {
                    let request_id = RequestId::generate_for_test();
                    eprintln!("[stage-request-test-only] 本地生成 request_id={request_id}");
                    request_id
                }
                None => return Err("首段 Stage 必须在 diagnostics.request_id 中配置 scheduler 下发的 request id".into()),
            };
            eprintln!("[stage-request] request_id={request_id} layers={stage_start}..{stage_end}");
            link.send_open(request_id, None, 0, false, SamplingConfig::greedy(0), false)?;
            let ready = link.recv()?;
            if ready.request_id != request_id || !matches!(ready.message, StageMessage::Ready { cached_tokens: 0 }) {
                return Err("front stage 等待后继 Open Ready 时收到非法 frame".into());
            }
            let chunks = build_pp_prefill_chunks(&prefill_contexts[0], &weights, &cfg, &tokens, 0, chunk_policy)
                .map_err(|error| format!("准备 GLM prefill chunks: {error:?}"))?
                .into_iter()
                .map(|(position, hidden)| Ok::<_, crate::backend::BackendError>((position, hidden, None)));
            let output_context = *prefill_contexts.last().ok_or("distributed stage 没有输出 device")?;
            let prefill_started = Instant::now();
            states = run_glm52_stage_pipeline_stateful(chunks, states, &cfg, &mla, &rope, scheduler_policy, |position, hidden, selection| {
                let rows = hidden.rows;
                let values = output_context.tensor_to_bf16_bits(&hidden)?;
                let selection = selection.map(|selection| selection.to_host()).transpose()?;
                link.send_prefill(request_id, position, rows, cfg.hidden_size, &values, selection.as_deref().unwrap_or(&[])).map_err(|msg| crate::backend::BackendError::Compute { msg })
            })
            .map_err(|error| format!("ROCm distributed front prefill: {error:?}"))?;
            link.send_prefill_done(request_id, tokens.len())?;
            eprintln!("[stage-prefill-sent] request_id={request_id} layers=0..{stage_end} tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64(),);
            crate::kernel::rocm::hip::enable_device_buffer_reuse();
            if args.decode_steps == 0 {
                link.send_delete(request_id)?;
                return Ok(());
            }
            let detokenizer = Detokenizer::load(&tokenizer_path)?;
            for step in 0..args.decode_steps {
                let frame = match link.recv() {
                    Ok(frame) => frame,
                    Err(error) if stage_listen => {
                        eprintln!("[stage-reconnect] 上游连接已断开，保留 resident cache 等待重连: {error}");
                        link.accept_reconnect().map_err(|reconnect| format!("stage 上游重连失败: {reconnect}"))?;
                        continue;
                    }
                    Err(error) => return Err(error.into()),
                };
                if frame.request_id != request_id {
                    return Err(format!("收到其他任务 token: expected={request_id} actual={}", frame.request_id).into());
                }
                let StageMessage::Token { token, eos } = frame.message else {
                    return Err("front stage 等待 token 时收到非 token frame".into());
                };
                print_token(&detokenizer, token, true)?;
                if eos || cfg.eos_token_ids.contains(&token) {
                    break;
                }
                let position = tokens.len() + step;
                if step + 1 == args.decode_steps || position == args.max_seq_len {
                    break;
                }
                let input = prefill_contexts[0].tensor_from_f32(weights.embedding_rows(&[token])?, 1, cfg.hidden_size)?;
                let mut output = None;
                let mut output_selection: Option<RocmDsaSelection> = None;
                let decode_started = Instant::now();
                states = run_glm52_stage_pipeline_stateful(vec![Ok((position, input, None))], states, &cfg, &mla, &rope, scheduler_policy, |_, hidden, selection| {
                    output = Some(hidden);
                    output_selection = selection;
                    Ok(())
                })
                .map_err(|error| format!("ROCm distributed front decode: {error:?}"))?;
                let hidden = output.take().ok_or("front decode 没有 stage output")?;
                let values = output_context.tensor_to_bf16_bits(&hidden).map_err(|error| format!("下载 front decode BF16 失败: {error:?}"))?;
                let selection = output_selection.map(|selection| selection.to_host()).transpose().map_err(|error| format!("下载 DSA selection 失败: {error:?}"))?;
                link.send_decode(request_id, position, cfg.hidden_size, &values, selection.as_deref().unwrap_or(&[]))?;
                eprintln!("\n[stage-decode-sent] request_id={request_id} position={position} wall={:.3}s", decode_started.elapsed().as_secs_f64(),);
            }
            link.send_delete(request_id)?;
            println!();
            return Ok(());
        }

        let input_context = prefill_contexts[0];
        let output_context = *prefill_contexts.last().ok_or("tail stage 没有输出 device")?;
        crate::kernel::rocm::hip::enable_device_buffer_reuse();
        // MTP 的 LM head/L78 必须在上报显存前固定驻留 B7；否则 A 会基于虚高的
        // free memory 放大 KV 准入，运行后再加载权重会形成隐性超卖。
        let (mut output_head, mut mtp_runtime) = if mtp_enabled {
            let (head, mtp) = prepare_tail_output_runtime(&output_context, &weights, &cfg, &mla, true, mtp_draft_tokens, mtp_draft_vocabulary.as_deref(), lm_head_quantization)?;
            (Some(head), mtp)
        } else {
            (None, None)
        };
        let mut device_start = stage_start;
        let device_count = prefill_contexts.len();
        let device_memory = prefill_contexts
            .iter()
            .zip(&prefill_layer_ends)
            .enumerate()
            .map(|(device_index, (context, &layer_end))| {
                let model_units = layer_end + 1 - device_start + usize::from(mtp_enabled && device_index + 1 == device_count) * cfg.mtp_layer_count;
                device_start = layer_end + 1;
                Ok(StageDeviceMemory {
                    device: context.device_id(),
                    model_units,
                    available_bytes: u64::try_from(context.stage_available_bytes().map_err(|error| format!("查询 tail ROCm device {} 可用显存: {error:?}", context.device_id()))?).map_err(|_| "stage available bytes 超过 u64".to_owned())?,
                    total_bytes: u64::try_from(context.stage_total_bytes().map_err(|error| format!("查询 tail ROCm device {} 总显存: {error:?}", context.device_id()))?).map_err(|_| "stage total bytes 超过 u64".to_owned())?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let cache_dir = args.cache_dir.as_ref().ok_or("GLM decode 后继必须设置 model.cache_directory")?;
        let cache_identity = Glm52CacheIdentity::new(&weights, kv_cache_format, stage_start, stage_end, mtp_enabled, mtp_enabled, args.max_seq_len, dspark_directory.as_deref());
        let swap = Arc::new(Glm52SwapStore::open(cache_dir.join("glm52").join(format!("stage-{stage_start}-{stage_end}")), &cache_identity)?);
        let mut link = if let Some(listener) = stage_listener {
            listener.accept().map_err(|error| format!("接受 stage peer: {error}"))?
        } else if let Some(StageLink::Connect { ticket, iroh }) = &stage_transport {
            StageTransport::connect(ticket, iroh.clone()).map_err(|error| format!("连接 stage peer: {error}"))?
        } else {
            return Err("distributed stage 缺少 transport".into());
        };
        link.send_device_memory(&device_memory, Some(max_concurrency))?;
        let templates = states;
        let mut active = HashMap::<RequestId, TailSession>::new();
        let mut resident = HashMap::<RequestId, TailResident>::new();
        let mut pending_opens = Vec::<TailPendingOpen>::new();
        loop {
            let frame = loop {
                let received = if pending_opens.is_empty() { link.recv().map(Some) } else { link.try_recv() };
                match received {
                    Ok(Some(frame)) => break frame,
                    Ok(None) => {
                        if !poll_tail_stage_opens(&mut link, &mut active, &mut resident, &mut pending_opens, &swap, &output_context, &templates, &cfg, args.max_seq_len)? {
                            std::thread::sleep(Duration::from_micros(50));
                        }
                    }
                    Err(error) => {
                        pending_opens.clear();
                        eprintln!("[stage-reconnect] 上游连接已断开，保留 {} active / {} resident session: {error}", active.len(), resident.len());
                        while let Err(reconnect) = link.accept_reconnect() {
                            eprintln!("[stage-reconnect-wait] 上游尚未就绪，继续等待: {reconnect}");
                        }
                        link.send_device_memory(&device_memory, Some(max_concurrency))?;
                    }
                }
            };
            let request_id = frame.request_id;
            if output_head.is_none() && matches!(&frame.message, StageMessage::Stream { .. } | StageMessage::MtpContext { .. } | StageMessage::PrefillDone { .. } | StageMessage::Decode { .. }) {
                let (head, mtp) = prepare_tail_output_runtime(&output_context, &weights, &cfg, &mla, mtp_enabled, mtp_draft_tokens, mtp_draft_vocabulary.as_deref(), lm_head_quantization)?;
                output_head = Some(head);
                mtp_runtime = mtp;
            }
            match frame.message {
                StageMessage::ContinuousStream { requests } => {
                    if requests.is_empty() {
                        return Err("tail 在 continuous stream 外收到结束帧".into());
                    }
                    let session_capacity = max_concurrency.max(requests.len());
                    let mut slot_requests = std::iter::repeat_with(|| None).take(session_capacity).collect::<Vec<Option<RequestId>>>();
                    let mut request_sessions = HashMap::new();
                    let mut accepted_positions = HashMap::new();
                    let mut session_states = Vec::with_capacity(requests.len());
                    for (session, &stream_request_id) in requests.iter().enumerate() {
                        let tail = active.get_mut(&stream_request_id).ok_or_else(|| format!("tail continuous request={stream_request_id} 尚未 Open"))?;
                        slot_requests[session] = Some(stream_request_id);
                        request_sessions.insert(stream_request_id, session);
                        accepted_positions.insert(stream_request_id, tail.position);
                        session_states.push(std::mem::take(&mut tail.states));
                    }
                    enum TailControl {
                        Cache { next_request_id: RequestId, tokens: usize },
                        Delete,
                    }
                    let mut pending_controls = HashMap::<RequestId, TailControl>::new();
                    let mut prefill_done = HashMap::<RequestId, usize>::new();
                    let mut prefill_emitted = HashMap::<RequestId, bool>::new();
                    let mut pending_tail_heads = HashMap::<u64, TailHeadCohort>::new();
                    let mut tail_head_requests = HashMap::<RequestId, (u64, usize)>::new();
                    let mut stream_ended = false;
                    let mut stream_control_id = None;
                    let profile_boundaries = diagnostics.profile_boundaries;
                    let mut recv_profile = [0_u128; 3];
                    let mut recv_profile_count = 0_usize;
                    let mut decode_started = std::iter::repeat_with(VecDeque::<Instant>::new).take(session_capacity).collect::<Vec<_>>();
                    let mut chain_profile_micros = 0_u128;
                    let mut chain_profile_count = 0_usize;
                    let mut head_profile_micros = 0_u128;
                    let mut head_profile_count = 0_usize;
                    let mut connection_lost = false;
                    let run = drive_glm52_stream_stage_pipeline_stateful(session_states, session_capacity, &cfg, &mla, &rope, scheduler_policy, |pipeline| {
                        let backend_error = |msg: String| crate::backend::BackendError::Compute { msg };
                        loop {
                            let mut progressed = false;
                            while let Some(output) = pipeline.try_recv()? {
                                progressed = true;
                                match output {
                                    StageSchedulerOutput::Work { cohort, session, mut position, value } => {
                                        let stream_request_id = slot_requests.get(session).and_then(|request| *request).ok_or_else(|| backend_error(format!("tail continuous 输出落到空 session={session}")))?;
                                        let tail = active.get_mut(&stream_request_id).ok_or_else(|| backend_error(format!("tail continuous request={stream_request_id} 完成时消失")))?;
                                        if (value.decode || value.verify)
                                            && let Some(started) = decode_started[session].pop_front()
                                        {
                                            chain_profile_micros += started.elapsed().as_micros();
                                            chain_profile_count += 1;
                                            if chain_profile_count == 32 {
                                                eprintln!("[glm52-boundary-b-chain] tokens=32 chain_ms={:.3}", chain_profile_micros as f64 / 1000.0);
                                                chain_profile_micros = 0;
                                                chain_profile_count = 0;
                                            }
                                        }
                                        let mut rows = value.hidden.rows;
                                        let mut next_position = position.saturating_add(rows);
                                        let expected_position = if value.verify { tail.verify_position.map(|base| base + tail.verify_hiddens.len()).unwrap_or(tail.position) } else { tail.position };
                                        if position != expected_position {
                                            return Err(backend_error(format!("tail continuous 输出边界错误: request={stream_request_id} position={position} expected={expected_position}",)));
                                        }
                                        if !value.decode
                                            && !value.verify
                                            && let (Some(runtime), Some(mtp)) = (mtp_runtime.as_mut(), tail.mtp.as_mut())
                                            && mtp.active
                                        {
                                            let inputs = mtp.prompt_tokens.get(position..next_position).ok_or_else(|| backend_error(format!("MTP prompt tokens 越界: [{position},{next_position})/{}", mtp.prompt_tokens.len())))?.to_vec();
                                            let mut batch = [RocmMtpCatchUp { session: mtp, target_position: position, target_inputs: inputs, target_hidden: value.hidden.clone() }];
                                            mtp_catch_up_batch(runtime, &mut batch, &weights, &cfg, &mla, &rope)?;
                                        }
                                        if value.decode
                                            && let (Some(runtime), Some(mtp)) = (mtp_runtime.as_mut(), tail.mtp.as_mut())
                                            && mtp.active
                                            && mtp.verify_inputs.len() == 1
                                        {
                                            let inputs = std::mem::take(&mut mtp.verify_inputs);
                                            mtp.truncate_to_target(position)?;
                                            let mut batch = [RocmMtpCatchUp { session: mtp, target_position: position, target_inputs: inputs, target_hidden: value.hidden.clone() }];
                                            mtp_catch_up_batch(runtime, &mut batch, &weights, &cfg, &mla, &rope)?;
                                        }
                                        if value.verify && tail.tail_sampling {
                                            let verify_rows = tail.mtp.as_ref().filter(|mtp| mtp.active).ok_or_else(|| backend_error(format!("request={stream_request_id} verify output 缺少 active MTP session")))?.verify_inputs.len();
                                            let verify_hidden = if rows == 1 && verify_rows > 1 {
                                                if tail.verify_position.is_none() {
                                                    tail.verify_position = Some(position);
                                                }
                                                tail.verify_hiddens.push(value.hidden.clone());
                                                if tail.verify_hiddens.len() < verify_rows {
                                                    continue;
                                                }
                                                if tail.verify_hiddens.len() != verify_rows {
                                                    return Err(backend_error(format!("B7 MTP split verify outputs={} expected={verify_rows}", tail.verify_hiddens.len())));
                                                }
                                                position = tail.verify_position.take().expect("split verify 已记录起点");
                                                rows = verify_rows;
                                                next_position = position.saturating_add(rows);
                                                let hiddens = std::mem::take(&mut tail.verify_hiddens);
                                                let refs = hiddens.iter().collect::<Vec<_>>();
                                                output_context.concat_token_rows(&refs)?
                                            } else {
                                                if tail.verify_position.is_some() || !tail.verify_hiddens.is_empty() {
                                                    return Err(backend_error(format!("B7 MTP verify 混入未完成 split outputs={}", tail.verify_hiddens.len())));
                                                }
                                                value.hidden.clone()
                                            };
                                            if pending_controls.contains_key(&stream_request_id) {
                                                continue;
                                            }
                                            let runtime = mtp_runtime.as_mut().ok_or_else(|| backend_error("收到 verify output 但 B7 MTP runtime 未启用".to_owned()))?;
                                            let mtp = tail.mtp.as_mut().filter(|mtp| mtp.active).ok_or_else(|| backend_error(format!("request={stream_request_id} verify output 缺少 active MTP session")))?;
                                            if mtp.verify_inputs.len() != rows {
                                                return Err(backend_error(format!("B7 MTP verify inputs={} rows={rows}", mtp.verify_inputs.len())));
                                            }
                                            if tail.pending_fences.len() < rows {
                                                return Err(backend_error(format!("B7 MTP verify fences={} rows={rows}", tail.pending_fences.len())));
                                            }
                                            let fences = tail.pending_fences.drain(..rows).collect::<Vec<_>>();
                                            let mut speculative_sampling = tail.sampling;
                                            let sampling = (0..rows).map(|_| speculative_sampling.next()).collect::<Vec<_>>();
                                            let target_tokens =
                                                glm52_sampled_token_ids_fenced(&output_context, &cfg, output_head.as_ref().ok_or_else(|| backend_error("B7 MTP verify 缺少 output head".to_owned()))?, &verify_hidden, &sampling, &fences)?;
                                            let outcome = verify_samples(&target_tokens, &mtp.verify_inputs[1..], &cfg.eos_token_ids)?;
                                            for _ in 0..outcome.retained_rows {
                                                tail.sampling.next();
                                            }
                                            let retained_hidden = output_context.slice_token_rows(&verify_hidden, 0, outcome.retained_rows)?;
                                            let hidden = output_context.slice_token_rows(&verify_hidden, outcome.retained_rows - 1, 1)?;
                                            let inputs = mtp.verify_inputs[..outcome.retained_rows].to_vec();
                                            mtp.truncate_to_target(position)?;
                                            let mut catch_up = [RocmMtpCatchUp { session: mtp, target_position: position, target_inputs: inputs, target_hidden: retained_hidden.clone() }];
                                            mtp_catch_up_batch(runtime, &mut catch_up, &weights, &cfg, &mla, &rope)?;
                                            tail.completion_tokens += outcome.tokens.len();
                                            let latest = *outcome.tokens.last().expect("verify 至少输出一个 token");
                                            let remaining = mtp.max_decode.saturating_sub(tail.completion_tokens);
                                            let count = if outcome.eos { 0 } else { mtp.draft_tokens.min(remaining.saturating_sub(1)) };
                                            let draft_hidden = mtp.pending_hidden.clone().ok_or_else(|| backend_error("B7 MTP verify 后缺少 target hidden".to_owned()))?;
                                            let mut drafts = [RocmMtpDraftBatch { session: mtp, token: latest, hidden: draft_hidden, count, drafts: Vec::new(), fence: GenerationGuard::new(Glm52ToolFence::new(false), false, []) }];
                                            mtp_draft_batch(runtime, &mut drafts, &weights, output_head.as_ref().ok_or_else(|| backend_error("B7 MTP draft 缺少 output head".to_owned()))?, &cfg, &mla, &rope)?;
                                            let next_drafts = std::mem::take(&mut drafts[0].drafts);
                                            drafts[0].session.verify_inputs = std::iter::once(latest).chain(next_drafts.iter().copied()).collect();
                                            let committed_position = position + outcome.retained_rows;
                                            tail.position = committed_position;
                                            tail.hidden = Some(hidden);
                                            tail.completed_hidden = Some((position, retained_hidden));
                                            accepted_positions.insert(stream_request_id, committed_position);
                                            if diagnostics.trace_stage_events || diagnostics.profile_boundaries {
                                                eprintln!(
                                                    "[glm52-mtp-verify] request_id={stream_request_id} drafts={} accepted={}/{} head_pass={} retained={}/{} output_tokens={} eos={}",
                                                    rows - 1,
                                                    outcome.accepted_drafts,
                                                    rows - 1,
                                                    rows,
                                                    outcome.retained_rows,
                                                    rows,
                                                    outcome.tokens.len(),
                                                    outcome.eos
                                                );
                                            }
                                            link.send_speculative(stream_request_id, &outcome.tokens, outcome.retained_rows, &next_drafts, outcome.eos).map_err(backend_error)?;
                                            continue;
                                        }
                                        let hidden = last_bf16_row(&output_context, value.hidden.clone()).map_err(|error| backend_error(format!("{error:?}")))?;
                                        if value.decode && diagnostics.trace_stage_output {
                                            let values = output_context.tensor_to_bf16_bits(&hidden).map_err(|error| backend_error(format!("下载 tail decode BF16: {error:?}")))?;
                                            let hash = values.iter().fold(0xcbf29ce484222325_u64, |hash, &value| (hash ^ u64::from(value)).wrapping_mul(0x100000001b3));
                                            eprintln!("[glm52-tail-output] request_id={stream_request_id} session={session} position={position} hash={hash:016x}");
                                        }
                                        tail.position = next_position;
                                        let emit_prefill = !value.decode && prefill_done.get(&stream_request_id).copied() == Some(next_position) && !prefill_emitted.get(&stream_request_id).copied().unwrap_or(false);
                                        if emit_prefill {
                                            prefill_emitted.insert(stream_request_id, true);
                                            tail.prompt_hidden = Some(hidden.clone());
                                            tail.prompt_position = Some(next_position);
                                        }
                                        // 同一轮 drain 中 Work 后面可能紧跟 Closed；先提交 terminal
                                        // hidden，保证 Close/Cache 能看到最终状态。output head 只额外持有
                                        // device buffer 的 Arc，不延迟 session 生命周期提交。
                                        tail.hidden = Some(hidden);
                                        // A0 才能决定 DSpark 接受长度。split verify 在 B7 逐行完成时
                                        // 保留整段输出，终局没有下一轮 decode 触发 truncate_to 时，Cache
                                        // 仍能取到接受边界的 terminal hidden 并正确回滚。
                                        tail.completed_hidden = Some(if value.verify && !tail.tail_sampling {
                                            match tail.completed_hidden.take() {
                                                Some((start, completed)) if start.saturating_add(completed.rows) == position => (start, output_context.concat_token_rows(&[&completed, &value.hidden])?),
                                                _ => (position, value.hidden.clone()),
                                            }
                                        } else {
                                            (position, value.hidden.clone())
                                        });
                                        let tail_sampling = tail.tail_sampling;
                                        let terminal_hidden = tail.hidden.as_ref().expect("刚提交 tail hidden").clone();
                                        let _ = tail;
                                        let (cohort, cohort_size) = cohort.unwrap_or((0, 1));
                                        if value.decode && tail_sampling && pending_controls.contains_key(&stream_request_id) {
                                            continue;
                                        }
                                        if value.decode && tail_sampling {
                                            let work = TailHeadWork { request_id: stream_request_id, cohort, cohort_size, position, hidden: terminal_hidden };
                                            if let Some(works) = push_tail_head_work(&mut pending_tail_heads, &mut tail_head_requests, work).map_err(backend_error)? {
                                                let work_count = works.len();
                                                let head_started = profile_boundaries.then(Instant::now);
                                                tail_sample_batch(
                                                    &mut link,
                                                    &mut active,
                                                    &output_context,
                                                    &mut mtp_runtime,
                                                    &weights,
                                                    &cfg,
                                                    &mla,
                                                    &rope,
                                                    output_head.as_ref().ok_or_else(|| backend_error("tail_sampling 缺少 output head".to_owned()))?,
                                                    works,
                                                )
                                                .map_err(backend_error)?;
                                                if let Some(started) = head_started {
                                                    head_profile_micros += started.elapsed().as_micros();
                                                    head_profile_count += work_count;
                                                    if head_profile_count >= 32 {
                                                        eprintln!("[glm52-boundary-b-head] tokens={head_profile_count} head_sample_send_ms={:.3}", head_profile_micros as f64 / 1000.0);
                                                        head_profile_micros = 0;
                                                        head_profile_count = 0;
                                                    }
                                                }
                                            }
                                            continue;
                                        }
                                        let values = output_context.completed_tensor_to_bf16_bits(&value.hidden).map_err(|error| backend_error(format!("下载 tail continuous BF16: {error:?}")))?;
                                        let aux_values = value
                                            .aux_hidden
                                            .as_ref()
                                            .map(|hidden| output_context.completed_tensor_to_bf16_bits(hidden))
                                            .transpose()
                                            .map_err(|error| backend_error(format!("下载 tail continuous aux hidden: {error:?}")))?
                                            .unwrap_or_default();
                                        let trace_started = (diagnostics.trace_stage_events && (value.decode || value.verify)).then(|| (Instant::now(), crate::runtime::prefill_scheduler::stage_trace_timestamp_us()));
                                        if value.verify {
                                            link.send_verify_cohort_aux(stream_request_id, cohort, cohort_size, position, rows, cfg.hidden_size, &values, &[], &aux_values, value.aux_taps).map_err(backend_error)?;
                                        } else if value.decode {
                                            link.send_decode_cohort_aux(stream_request_id, cohort, cohort_size, position, cfg.hidden_size, &values, &[], &aux_values, value.aux_taps).map_err(backend_error)?;
                                        } else {
                                            link.send_prefill_aux(stream_request_id, position, rows, cfg.hidden_size, &values, &[], &aux_values, value.aux_taps).map_err(backend_error)?;
                                            if emit_prefill && tail_sampling {
                                                tail_sample_batch(
                                                    &mut link,
                                                    &mut active,
                                                    &output_context,
                                                    &mut mtp_runtime,
                                                    &weights,
                                                    &cfg,
                                                    &mla,
                                                    &rope,
                                                    output_head.as_ref().ok_or_else(|| backend_error("tail_sampling 缺少 output head".to_owned()))?,
                                                    vec![TailHeadWork { request_id: stream_request_id, cohort: 0, cohort_size: 1, position: next_position.saturating_sub(1), hidden: terminal_hidden }],
                                                )
                                                .map_err(backend_error)?;
                                            }
                                        }
                                        if let Some((started, begin_us)) = trace_started {
                                            let kind = if value.verify { "Verify" } else { "Decode" };
                                            crate::runtime::prefill_scheduler::record_stage_trace(format!(
                                                "[stage-boundary-trace] ts_us={begin_us} phase=send direction=B_to_A request_id={stream_request_id} lane={session}@{position} kind={kind} rows={rows} complete_us={} duration_us={}",
                                                crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                                                started.elapsed().as_micros()
                                            ));
                                        }
                                    }
                                    StageSchedulerOutput::Opened { .. } => {}
                                    StageSchedulerOutput::Closed { session, states } => {
                                        let stream_request_id = slot_requests.get_mut(session).and_then(Option::take).ok_or_else(|| backend_error(format!("tail continuous Close 落到空 session={session}")))?;
                                        let tail = active.get_mut(&stream_request_id).ok_or_else(|| backend_error(format!("tail continuous Close request={stream_request_id} 时消失")))?;
                                        tail.states = states;
                                        request_sessions.remove(&stream_request_id);
                                        accepted_positions.remove(&stream_request_id);
                                        prefill_done.remove(&stream_request_id);
                                        prefill_emitted.remove(&stream_request_id);
                                        match pending_controls.remove(&stream_request_id) {
                                            Some(TailControl::Cache { next_request_id, tokens }) => {
                                                let tail = active.remove(&stream_request_id).expect("刚检查 active");
                                                tail_stage_cache_commit(&mut link, &mut resident, &output_context, tail, stream_request_id, next_request_id, tokens, true).map_err(backend_error)?;
                                                eprintln!("[stage-continuous-cache] request_id={stream_request_id} cache_id={next_request_id} tokens={tokens} resident={}", resident.len(),);
                                            }
                                            Some(TailControl::Delete) => {
                                                let position = tail_stage_delete(&mut active, &mut resident, &swap, stream_request_id).map_err(backend_error)?;
                                                link.send_ready(stream_request_id, position).map_err(backend_error)?;
                                                eprintln!("[stage-continuous-delete] request_id={stream_request_id} active={} resident={}", active.len(), resident.len(),);
                                            }
                                            None => {
                                                return Err(backend_error(format!("tail continuous Close request={stream_request_id} 缺少控制命令",)));
                                            }
                                        }
                                    }
                                }
                            }

                            loop {
                                let frame = match link.try_recv() {
                                    Ok(Some(frame)) => frame,
                                    Ok(None) => break,
                                    // 连接断开（A 重启等）：不 crash，放弃当前 continuous batch 回到外层 reconnect。
                                    Err(message) => {
                                        connection_lost = true;
                                        eprintln!("[stage-warn] tail continuous stream 连接断开，等待上游重连: {message}");
                                        return Ok(());
                                    }
                                };
                                progressed = true;
                                let frame_request_id = frame.request_id;
                                match frame.message {
                                    StageMessage::ContinuousStream { requests } if requests.is_empty() => {
                                        stream_ended = true;
                                        stream_control_id = Some(frame_request_id);
                                    }
                                    StageMessage::ContinuousStream { requests } => {
                                        // A 在 continuous stream 中途重发起始帧（通常是客户端断开后 batch 调度重新触发）。
                                        // 跳过而非 crash，避免整个 B 节点退出。
                                        eprintln!("[stage-warn] tail continuous stream 中收到重复起始帧 requests={}，已跳过", requests.len());
                                    }
                                    StageMessage::StreamAssign { session } => {
                                        if session >= session_capacity || slot_requests[session].is_some() {
                                            return Err(backend_error(format!("tail continuous Assign session={session}/{} 非空或越界", session_capacity,)));
                                        }
                                        let tail = active.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail continuous Assign request={frame_request_id} 尚未 Open")))?;
                                        if tail.states.is_empty() {
                                            return Err(backend_error(format!("tail continuous Assign request={frame_request_id} 已在 pipeline")));
                                        }
                                        let states = std::mem::take(&mut tail.states);
                                        pipeline.open(session, states)?;
                                        slot_requests[session] = Some(frame_request_id);
                                        request_sessions.insert(frame_request_id, session);
                                        accepted_positions.insert(frame_request_id, tail.position);
                                    }
                                    StageMessage::Open { cache_request_id, cached_tokens, reserved_rows, cache_hit, sampling, tail_sampling } => {
                                        if tail_sampling && output_head.is_none() {
                                            let (head, mtp) = prepare_tail_output_runtime(&output_context, &weights, &cfg, &mla, false, mtp_draft_tokens, None, lm_head_quantization).map_err(backend_error)?;
                                            output_head = Some(head);
                                            mtp_runtime = mtp;
                                        }
                                        begin_tail_stage_open(
                                            &mut link,
                                            &mut active,
                                            &mut resident,
                                            &mut pending_opens,
                                            &swap,
                                            &output_context,
                                            &templates,
                                            &cfg,
                                            args.max_seq_len,
                                            TailOpenRequest { request_id: frame_request_id, cache_request_id, cached_tokens, reserved_rows, cache_hit, sampling, tail_sampling },
                                        )
                                        .map_err(backend_error)?;
                                    }
                                    StageMessage::MtpContext { prompt_tokens, max_decode, draft_tokens } => {
                                        let tail = active.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail MTP context request={frame_request_id} 尚未 Open")))?;
                                        tail_stage_mtp_context(tail, mtp_runtime.is_some(), mtp_draft_tokens, &cfg, args.max_seq_len, frame_request_id, prompt_tokens, max_decode, draft_tokens).map_err(backend_error)?;
                                    }
                                    StageMessage::Prefill { position, rows, cols, values, selection, aux_values, aux_taps } => {
                                        let session = request_sessions.get(&frame_request_id).copied().ok_or_else(|| backend_error(format!("tail continuous prefill request={frame_request_id} 尚未 Assign")))?;
                                        let expected = accepted_positions.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail continuous prefill request={frame_request_id} 缺少 position")))?;
                                        if position != *expected || cols != cfg.hidden_size {
                                            return Err(backend_error(format!("tail continuous prefill 边界错误: request={frame_request_id} position={position} expected={} shape=[{rows},{cols}]", *expected,)));
                                        }
                                        let selection = if selection.is_empty() {
                                            None
                                        } else {
                                            Some(RocmDsaSelection::from_host(&input_context, rows, position, &selection).map_err(|error| backend_error(format!("上传 continuous prefill DSA selection: {error:?}")))?)
                                        };
                                        let hidden = input_context.tensor_from_bf16_bits_ordered(values, rows, cols).map_err(|error| backend_error(format!("上传 tail continuous prefill BF16: {error:?}")))?;
                                        let aux_hidden = (!aux_values.is_empty())
                                            .then(|| input_context.tensor_from_bf16_bits_ordered(aux_values, rows, cols))
                                            .transpose()
                                            .map_err(|error| backend_error(format!("上传 tail continuous prefill aux hidden: {error:?}")))?;
                                        *expected = position.saturating_add(rows);
                                        pipeline.submit_glm52_aux(session, position, false, hidden, selection, aux_hidden, aux_taps)?;
                                    }
                                    StageMessage::Decode { cohort, cohort_size, position, cols, values, selection, aux_values, aux_taps } => {
                                        let session = request_sessions.get(&frame_request_id).copied().ok_or_else(|| backend_error(format!("tail continuous decode request={frame_request_id} 尚未 Assign")))?;
                                        if diagnostics.trace_stage_events {
                                            crate::runtime::prefill_scheduler::record_stage_trace(format!(
                                                "[stage-boundary-trace] ts_us={} phase=recv direction=A_to_B request_id={frame_request_id} lane={session}@{position} kind=Decode rows=1 cohort={cohort}/{cohort_size}",
                                                crate::runtime::prefill_scheduler::stage_trace_timestamp_us()
                                            ));
                                        }
                                        let expected = accepted_positions.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail continuous decode request={frame_request_id} 缺少 position")))?;
                                        if position > *expected || cols != cfg.hidden_size || position >= args.max_seq_len {
                                            return Err(backend_error(format!("tail continuous decode 边界错误: request={frame_request_id} position={position} expected={} cols={cols}", *expected,)));
                                        }
                                        let tail = active.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail continuous decode request={frame_request_id} 尚未 Open")))?;
                                        if position < tail.position {
                                            tail.position = position;
                                        }
                                        let boundary_started = profile_boundaries.then(Instant::now);
                                        let selection = if selection.is_empty() {
                                            None
                                        } else {
                                            Some(RocmDsaSelection::from_host(&input_context, 1, position, &selection).map_err(|error| backend_error(format!("上传 continuous decode DSA selection: {error:?}")))?)
                                        };
                                        let selection_micros = boundary_started.map(|started| started.elapsed().as_micros()).unwrap_or(0);
                                        if diagnostics.trace_stage_output {
                                            let hash = values.iter().fold(0xcbf29ce484222325_u64, |hash, &value| (hash ^ u64::from(value)).wrapping_mul(0x100000001b3));
                                            eprintln!("[glm52-tail-input] request_id={frame_request_id} session={session} position={position} hash={hash:016x}");
                                        }
                                        let hidden = input_context.tensor_from_bf16_bits_ordered(values, 1, cols).map_err(|error| backend_error(format!("上传 tail continuous decode BF16: {error:?}")))?;
                                        let aux_hidden = (!aux_values.is_empty())
                                            .then(|| input_context.tensor_from_bf16_bits_ordered(aux_values, 1, cols))
                                            .transpose()
                                            .map_err(|error| backend_error(format!("上传 tail continuous decode aux hidden: {error:?}")))?;
                                        let hidden_micros = boundary_started.map(|started| started.elapsed().as_micros()).unwrap_or(0);
                                        *expected = position.saturating_add(1);
                                        if tail.tail_sampling && cohort != 0 && cohort_size > 1 && tail_head_requests.insert(frame_request_id, (cohort, cohort_size)).is_some() {
                                            return Err(backend_error(format!("tail output request={frame_request_id} 重复加入 cohort")));
                                        }
                                        if cohort == 0 {
                                            pipeline.submit_glm52_aux(session, position, true, hidden, selection, aux_hidden, aux_taps)?;
                                        } else {
                                            pipeline.submit_glm52_wave_aux(cohort, cohort_size, session, position, hidden, selection, aux_hidden, aux_taps)?;
                                        }
                                        if let Some(started) = boundary_started {
                                            decode_started[session].push_back(started);
                                        }
                                        if let Some(started) = boundary_started {
                                            let total_micros = started.elapsed().as_micros();
                                            recv_profile[0] += selection_micros;
                                            recv_profile[1] += hidden_micros.saturating_sub(selection_micros);
                                            recv_profile[2] += total_micros.saturating_sub(hidden_micros);
                                            recv_profile_count += 1;
                                            if recv_profile_count == 32 {
                                                eprintln!(
                                                    "[glm52-boundary-b-recv] tokens=32 selection_ms={:.3} hidden_ms={:.3} submit_ms={:.3}",
                                                    recv_profile[0] as f64 / 1000.0,
                                                    recv_profile[1] as f64 / 1000.0,
                                                    recv_profile[2] as f64 / 1000.0
                                                );
                                                recv_profile = [0; 3];
                                                recv_profile_count = 0;
                                            }
                                        }
                                    }
                                    StageMessage::Verify { cohort, cohort_size, position, rows, cols, values, selection, aux_values, aux_taps } => {
                                        let session = request_sessions.get(&frame_request_id).copied().ok_or_else(|| backend_error(format!("tail continuous verify request={frame_request_id} 尚未 Assign")))?;
                                        if diagnostics.trace_stage_events {
                                            crate::runtime::prefill_scheduler::record_stage_trace(format!(
                                                "[stage-boundary-trace] ts_us={} phase=recv direction=A_to_B request_id={frame_request_id} lane={session}@{position} kind=Verify rows={rows} cohort={cohort}/{cohort_size}",
                                                crate::runtime::prefill_scheduler::stage_trace_timestamp_us()
                                            ));
                                        }
                                        let expected = accepted_positions.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail continuous verify request={frame_request_id} 缺少 position")))?;
                                        if rows == 0 || position > *expected || cols != cfg.hidden_size || position.saturating_add(rows) > args.max_seq_len {
                                            return Err(backend_error(format!("tail continuous verify 边界错误: request={frame_request_id} position={position} expected={} shape=[{rows},{cols}]", *expected)));
                                        }
                                        let tail = active.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail continuous verify request={frame_request_id} 尚未 Open")))?;
                                        if position < tail.position {
                                            tail.position = position;
                                        }
                                        let selection = if selection.is_empty() {
                                            None
                                        } else {
                                            Some(RocmDsaSelection::from_host(&input_context, rows, position, &selection).map_err(|error| backend_error(format!("上传 continuous verify DSA selection: {error:?}")))?)
                                        };
                                        let hidden = input_context.tensor_from_bf16_bits_ordered(values, rows, cols).map_err(|error| backend_error(format!("上传 tail continuous verify BF16: {error:?}")))?;
                                        let aux_hidden = (!aux_values.is_empty())
                                            .then(|| input_context.tensor_from_bf16_bits_ordered(aux_values, rows, cols))
                                            .transpose()
                                            .map_err(|error| backend_error(format!("上传 tail continuous verify aux hidden: {error:?}")))?;
                                        *expected = position.saturating_add(rows);
                                        if cohort == 0 {
                                            pipeline.submit_glm52_verify_aux(session, position, hidden, selection, aux_hidden, aux_taps)?;
                                        } else {
                                            pipeline.submit_glm52_verify_wave_aux(cohort, cohort_size, session, position, hidden, selection, aux_hidden, aux_taps)?;
                                        }
                                        if profile_boundaries {
                                            decode_started[session].push_back(Instant::now());
                                        }
                                    }
                                    StageMessage::PrefillDone { tokens } => {
                                        let Some(expected) = accepted_positions.get(&frame_request_id).copied() else {
                                            // 完整 prompt 命中 resident cache 时没有需要提交的 prefill，
                                            // A 会直接发送 PrefillDone；此时无需先占用 pipeline slot。
                                            let tail = active.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail continuous PrefillDone request={frame_request_id} 尚未 Open")))?;
                                            if tail.states.is_empty() || tail.position != tokens {
                                                return Err(backend_error(format!("tail continuous PrefillDone request={frame_request_id} 未 Assign且边界不匹配: tokens={tokens} position={} states={}", tail.position, tail.states.len())));
                                            }
                                            if !prefill_emitted.get(&frame_request_id).copied().unwrap_or(false) {
                                                let hidden = tail.hidden.clone().ok_or_else(|| backend_error(format!("tail continuous PrefillDone request={frame_request_id} 缺少缓存 hidden")))?;
                                                tail.prompt_hidden = Some(hidden.clone());
                                                tail.prompt_position = Some(tokens);
                                                prefill_emitted.insert(frame_request_id, true);
                                                if tail.tail_sampling {
                                                    let _ = tail;
                                                    tail_sample_batch(
                                                        &mut link,
                                                        &mut active,
                                                        &output_context,
                                                        &mut mtp_runtime,
                                                        &weights,
                                                        &cfg,
                                                        &mla,
                                                        &rope,
                                                        output_head.as_ref().ok_or_else(|| backend_error("tail_sampling 缺少 output head".to_owned()))?,
                                                        vec![TailHeadWork { request_id: frame_request_id, cohort: 0, cohort_size: 1, position: tokens.saturating_sub(1), hidden }],
                                                    )
                                                    .map_err(backend_error)?;
                                                }
                                            }
                                            eprintln!("[stage-continuous-prefill-done-unassigned] request_id={frame_request_id} tokens={tokens}");
                                            continue;
                                        };
                                        if tokens != expected {
                                            return Err(backend_error(format!("tail continuous PrefillDone request={frame_request_id} tokens={tokens} expected={expected}",)));
                                        }
                                        prefill_done.insert(frame_request_id, tokens);
                                        let tail = active.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail continuous PrefillDone request={frame_request_id} 尚未 Open")))?;
                                        if tail.position == tokens && !prefill_emitted.get(&frame_request_id).copied().unwrap_or(false) {
                                            let hidden = tail.hidden.clone().ok_or_else(|| backend_error(format!("tail continuous PrefillDone request={frame_request_id} 缺少完成 hidden")))?;
                                            tail.prompt_hidden = Some(hidden.clone());
                                            tail.prompt_position = Some(tokens);
                                            prefill_emitted.insert(frame_request_id, true);
                                            if tail.tail_sampling {
                                                let _ = tail;
                                                tail_sample_batch(
                                                    &mut link,
                                                    &mut active,
                                                    &output_context,
                                                    &mut mtp_runtime,
                                                    &weights,
                                                    &cfg,
                                                    &mla,
                                                    &rope,
                                                    output_head.as_ref().ok_or_else(|| backend_error("tail_sampling 缺少 output head".to_owned()))?,
                                                    vec![TailHeadWork { request_id: frame_request_id, cohort: 0, cohort_size: 1, position: tokens.saturating_sub(1), hidden }],
                                                )
                                                .map_err(backend_error)?;
                                            }
                                        }
                                    }
                                    StageMessage::Sample { excluded } => {
                                        let tail = active.get_mut(&frame_request_id).ok_or_else(|| backend_error(format!("tail Sample request={frame_request_id} 尚未 Open")))?;
                                        if !tail.tail_sampling || tail.pending_fences.len() >= mtp_draft_tokens.saturating_add(1) {
                                            return Err(backend_error(format!("tail Sample request={frame_request_id} 状态非法: enabled={} pending={}", tail.tail_sampling, tail.pending_fences.len())));
                                        }
                                        tail.pending_fences.push_back(crate::backend::TokenFence::excluding(excluded));
                                    }
                                    StageMessage::Cache { next_request_id, tokens } => {
                                        if let Some(works) = cancel_tail_head_work(&mut pending_tail_heads, &mut tail_head_requests, frame_request_id).map_err(backend_error)? {
                                            tail_sample_batch(
                                                &mut link,
                                                &mut active,
                                                &output_context,
                                                &mut mtp_runtime,
                                                &weights,
                                                &cfg,
                                                &mla,
                                                &rope,
                                                output_head.as_ref().ok_or_else(|| backend_error("tail_sampling 缺少 output head".to_owned()))?,
                                                works,
                                            )
                                            .map_err(backend_error)?;
                                        }
                                        if let Some(session) = request_sessions.get(&frame_request_id).copied() {
                                            if pending_controls.insert(frame_request_id, TailControl::Cache { next_request_id, tokens }).is_some() {
                                                return Err(backend_error(format!("tail continuous request={frame_request_id} 重复控制")));
                                            }
                                            pipeline.close(session)?;
                                        } else {
                                            // Open 后客户端可能在 StreamAssign 前取消；此时状态仍在 active，
                                            // 直接提交并 ACK，不能让单个请求击穿整个 B 进程。
                                            let tail = active.remove(&frame_request_id).ok_or_else(|| backend_error(format!("tail continuous Cache request={frame_request_id} 尚未 Open")))?;
                                            if tail.states.is_empty() {
                                                return Err(backend_error(format!("tail continuous Cache request={frame_request_id} 未 Assign 但 states 为空")));
                                            }
                                            tail_stage_cache_commit(&mut link, &mut resident, &output_context, tail, frame_request_id, next_request_id, tokens, true).map_err(backend_error)?;
                                            eprintln!("[stage-continuous-cache-unassigned] request_id={frame_request_id} cache_id={next_request_id} tokens={tokens} resident={}", resident.len());
                                        }
                                    }
                                    StageMessage::Delete => {
                                        if let Some(works) = cancel_tail_head_work(&mut pending_tail_heads, &mut tail_head_requests, frame_request_id).map_err(backend_error)? {
                                            tail_sample_batch(
                                                &mut link,
                                                &mut active,
                                                &output_context,
                                                &mut mtp_runtime,
                                                &weights,
                                                &cfg,
                                                &mla,
                                                &rope,
                                                output_head.as_ref().ok_or_else(|| backend_error("tail_sampling 缺少 output head".to_owned()))?,
                                                works,
                                            )
                                            .map_err(backend_error)?;
                                        }
                                        if let Some(&session) = request_sessions.get(&frame_request_id) {
                                            if pending_controls.insert(frame_request_id, TailControl::Delete).is_some() {
                                                return Err(backend_error(format!("tail continuous request={frame_request_id} 重复控制")));
                                            }
                                            pipeline.close(session)?;
                                        } else {
                                            let position = tail_stage_delete(&mut active, &mut resident, &swap, frame_request_id).map_err(backend_error)?;
                                            link.send_ready(frame_request_id, position).map_err(backend_error)?;
                                            eprintln!("[stage-continuous-delete-unassigned] request_id={frame_request_id} tokens={position} active={} resident={}", active.len(), resident.len());
                                        }
                                    }
                                    StageMessage::SwapOut => {
                                        tail_stage_swap_out(&mut link, &mut resident, &swap, &output_context, frame_request_id).map_err(backend_error)?;
                                    }
                                    StageMessage::Persist => {
                                        tail_stage_persist(&mut resident, &swap, &output_context, frame_request_id).map_err(backend_error)?;
                                    }
                                    StageMessage::Stream { .. }
                                    | StageMessage::Shutdown { .. }
                                    | StageMessage::Token { .. }
                                    | StageMessage::Sampled { .. }
                                    | StageMessage::Speculative { .. }
                                    | StageMessage::Ready { .. }
                                    | StageMessage::DeviceMemory { .. } => {
                                        return Err(backend_error(format!("tail continuous request={frame_request_id} 收到非法 frame={:?}", frame.message,)));
                                    }
                                }
                            }
                            // 先排空上游控制帧，让同批 Open 都启动 host 读；之后才把
                            // 已完成 snapshot 顺序上传 GPU，避免第一路 H2D 挡住后续读盘。
                            if poll_tail_stage_opens(&mut link, &mut active, &mut resident, &mut pending_opens, &swap, &output_context, &templates, &cfg, args.max_seq_len).map_err(backend_error)? {
                                progressed = true;
                            }
                            if stream_ended && request_sessions.is_empty() {
                                let control_id = stream_control_id.ok_or_else(|| backend_error("tail continuous stream 结束但缺少 control id".to_owned()))?;
                                link.send_ready(control_id, 0).map_err(backend_error)?;
                                break;
                            }
                            if !progressed {
                                std::thread::sleep(std::time::Duration::from_micros(50));
                            }
                        }
                        Ok(())
                    })
                    .map_err(|error| format!("ROCm distributed tail continuous stream: {error:?}"))?;
                    if crate::kernel::rocm::hip::options().kernel_profile || profile_completion {
                        crate::kernel::rocm::hip::report_device_profiles(0, request_sessions.len());
                    }
                    if connection_lost {
                        let dropped = active.len();
                        active.clear();
                        eprintln!("[stage-upstream-reset] 已取消旧上游的 {dropped} 个 active session，保留 {} 个 resident cache", resident.len());
                    } else if run.1.iter().any(Option::is_some) {
                        eprintln!("[stage-warn] tail continuous stream 结束时仍有 resident pipeline session，已忽略");
                    }
                }
                StageMessage::Stream { requests } => {
                    if diagnostics.trace_stage_events {
                        eprintln!("[stage-stream-tail] requests={}", requests.iter().map(ToString::to_string).collect::<Vec<_>>().join(","));
                    }
                    let stream_started = Instant::now();
                    let request_count = requests.len();
                    let mut positions = Vec::with_capacity(request_count);
                    let mut accepted_positions = Vec::with_capacity(request_count);
                    let mut prefill_done = Vec::with_capacity(request_count);
                    let mut prefill_emitted = Vec::with_capacity(request_count);
                    let mut session_states = Vec::with_capacity(requests.len());
                    let mut sampling_states = Vec::with_capacity(requests.len());
                    for &request_id in &requests {
                        let session = active.get_mut(&request_id).ok_or_else(|| format!("tail stream request={request_id} 尚未 Open"))?;
                        positions.push(AtomicUsize::new(session.position));
                        accepted_positions.push(session.position);
                        prefill_done.push(AtomicUsize::new(usize::MAX));
                        prefill_emitted.push(AtomicUsize::new(0));
                        session_states.push(std::mem::take(&mut session.states));
                        sampling_states.push(session.sampling);
                    }
                    let request_sessions = requests.iter().copied().enumerate().map(|(session, request_id)| (request_id, session)).collect::<HashMap<_, _>>();
                    let mut saw_decode = false;
                    let mut stream_closed = false;
                    let outputs = Mutex::new((0..request_count).map(|_| None).collect::<Vec<Option<RocmTensor>>>());
                    let sampling_states = Mutex::new(sampling_states);
                    let stream_link = &mut link;
                    let mut connection_lost = false;
                    let (outbound, outbound_rx) = std::sync::mpsc::channel::<(usize, u32, bool)>();
                    // 输入闭包迁移到专用 feeder 线程后,Receiver(!Sync)按引用捕获会
                    // 阻止闭包跨线程;包 Mutex 仅为此,排空语义与单线程时一致。
                    let outbound_rx = std::sync::Mutex::new(outbound_rx);
                    use crate::runtime::prefill::TokenStreamBatchPoll;
                    let chunks = || {
                        while let Ok((session, token, eos)) = outbound_rx.lock().expect("tail outbound 锁中毒").try_recv() {
                            if let Err(msg) = stream_link.send_token(requests[session], token, eos) {
                                connection_lost = true;
                                eprintln!("[stage-warn] tail stream 发送失败，等待上游重连: {msg}");
                                return TokenStreamBatchPoll::Closed;
                            }
                        }
                        if stream_closed {
                            return TokenStreamBatchPoll::Closed;
                        }
                        loop {
                            let frame = match stream_link.try_recv() {
                                Ok(Some(frame)) => frame,
                                Ok(None) => return TokenStreamBatchPoll::Pending,
                                Err(msg) => {
                                    connection_lost = true;
                                    eprintln!("[stage-warn] tail stream 连接断开，等待上游重连: {msg}");
                                    return TokenStreamBatchPoll::Closed;
                                }
                            };
                            if let StageMessage::Stream { requests } = &frame.message {
                                if requests.is_empty() {
                                    stream_closed = true;
                                    return TokenStreamBatchPoll::Closed;
                                }
                                return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute { msg: format!("tail stream 中收到非法 Stream frame: requests={}", requests.len()) }));
                            }
                            let Some(&session_index) = request_sessions.get(&frame.request_id) else {
                                return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute { msg: format!("tail stream 收到未声明 request={}", frame.request_id) }));
                            };
                            let (position, rows, cols, values, selection, decode) = match frame.message {
                                StageMessage::Prefill { position, rows, cols, values, selection, .. } => (position, rows, cols, values, selection, false),
                                StageMessage::Decode { position, cols, values, selection, .. } => (position, 1, cols, values, selection, true),
                                StageMessage::PrefillDone { tokens } => {
                                    let expected = accepted_positions[session_index];
                                    if tokens != expected {
                                        return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute {
                                            msg: format!("tail stream PrefillDone 边界错误: request={} tokens={tokens} expected={expected}", frame.request_id)
                                        }));
                                    }
                                    prefill_done[session_index].store(tokens, Ordering::Release);
                                    if positions[session_index].load(Ordering::Acquire) == tokens && prefill_emitted[session_index].compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                                        let outputs = outputs.lock().unwrap();
                                        let Some(hidden) = outputs[session_index].as_ref() else {
                                            return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute { msg: format!("tail stream PrefillDone request={} 缺少完成 hidden", frame.request_id) }));
                                        };
                                        let sampling = {
                                            let mut states = sampling_states.lock().unwrap();
                                            [states[session_index].next()]
                                        };
                                        let token = match glm52_sampled_token_ids(&output_context, &cfg, output_head.as_ref().expect("legacy output head 已准备"), hidden, &sampling) {
                                            Ok(output) => match output.into_iter().next() {
                                                Some(token) => token,
                                                None => return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute { msg: format!("tail ROCm GLM stream prefill output 缺少 token: sampling={} 条", sampling.len()) })),
                                            },
                                            Err(error) => return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute { msg: format!("tail ROCm GLM stream prefill output: {error:?}") })),
                                        };
                                        if outbound.send((session_index, token, cfg.eos_token_ids.contains(&token))).is_err() {
                                            return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute { msg: "tail stream prefill token 队列已关闭".to_owned() }));
                                        }
                                    }
                                    continue;
                                }
                                message => {
                                    return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute {
                                        msg: format!("tail stream request={} 收到非 Prefill/Decode/PrefillDone frame={message:?} current={}", frame.request_id, requests.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")),
                                    }));
                                }
                            };
                            saw_decode |= decode;
                            let expected = accepted_positions[session_index];
                            let Some(next_position) = position.checked_add(rows) else {
                                return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute { msg: format!("tail stream position 溢出: request={} position={position} rows={rows}", frame.request_id) }));
                            };
                            if rows == 0 || position != expected || cols != cfg.hidden_size || next_position > args.max_seq_len {
                                return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute {
                                    msg: format!("tail stream 边界错误: request={} position={position} expected={expected} shape=[{rows},{cols}] decode={decode} max_seq_len={}", frame.request_id, args.max_seq_len),
                                }));
                            }
                            accepted_positions[session_index] = next_position;
                            let selection = if selection.is_empty() {
                                None
                            } else {
                                match RocmDsaSelection::from_host(&input_context, rows, position, &selection) {
                                    Ok(selection) => Some(selection),
                                    Err(error) => return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute { msg: format!("上传 stream DSA selection: {error:?}") })),
                                }
                            };
                            let hidden = match input_context.tensor_from_bf16_bits_ordered(values, rows, cols) {
                                Ok(hidden) => hidden,
                                Err(error) => return TokenStreamBatchPoll::Ready(Err(crate::backend::BackendError::Compute { msg: format!("上传 tail stream BF16: {error:?}") })),
                            };
                            return TokenStreamBatchPoll::Ready(Ok((session_index, position, decode, hidden, selection)));
                        }
                    };
                    let run = run_glm52_stream_stage_pipeline_stateful(chunks, session_states, &cfg, &mla, &rope, scheduler_policy, |session, position, decode, hidden, _| {
                        let rows = hidden.rows;
                        let hidden = last_bf16_row(&output_context, hidden)?;
                        let next_position = position.saturating_add(rows);
                        // 先保存输出，再发布完成位置。PrefillDone 线程用 Acquire 读取位置后，
                        // 必须已经能看到对应 hidden，不能观察到一半完成的 session。
                        outputs.lock().unwrap()[session] = Some(hidden);
                        positions[session].store(next_position, Ordering::Release);
                        let prefill_complete = !decode && prefill_done[session].load(Ordering::Acquire) == next_position && prefill_emitted[session].compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok();
                        if decode || prefill_complete {
                            let outputs = outputs.lock().unwrap();
                            let hidden = outputs[session].as_ref().expect("完成位置发布前已保存 hidden");
                            let sampling = {
                                let mut states = sampling_states.lock().unwrap();
                                [states[session].next()]
                            };
                            let token = glm52_sampled_token_ids(&output_context, &cfg, output_head.as_ref().expect("legacy output head 已准备"), hidden, &sampling)
                                .map_err(|error| crate::backend::BackendError::Compute { msg: format!("tail ROCm GLM stream output: {error:?}") })?
                                .into_iter()
                                .next()
                                .ok_or_else(|| crate::backend::BackendError::Compute { msg: format!("tail ROCm GLM stream output 缺少 token: session={session} sampling={} 条", sampling.len()) })?;
                            outbound.send((session, token, cfg.eos_token_ids.contains(&token))).map_err(|_| crate::backend::BackendError::Compute { msg: "tail stream token 队列已关闭".to_owned() })?;
                        }
                        Ok(())
                    });
                    let session_states = run.map_err(|error| format!("ROCm distributed tail stream: {error:?}"))?;
                    if connection_lost {
                        let dropped = active.len();
                        active.clear();
                        eprintln!("[stage-upstream-reset] 已取消旧上游的 {dropped} 个 active session，保留 {} 个 resident cache", resident.len());
                        continue;
                    }
                    let mut outputs = outputs.into_inner().unwrap();
                    let sampling_states = sampling_states.into_inner().unwrap();
                    for (session_index, (request_id, states)) in requests.into_iter().zip(session_states).enumerate() {
                        let hidden = outputs[session_index].take().ok_or_else(|| format!("tail stream request={request_id} 没有 stage output"))?;
                        let session = active.get_mut(&request_id).ok_or_else(|| format!("tail stream request={request_id} 完成时 session 消失"))?;
                        session.states = states;
                        session.hidden = Some(hidden);
                        session.position = positions[session_index].load(Ordering::Acquire);
                        session.sampling = sampling_states[session_index];
                    }
                    if saw_decode {
                        eprintln!("[stage-decode-batch] sessions={request_count} wall={:.3}s active={}", stream_started.elapsed().as_secs_f64(), active.len());
                    }
                }
                StageMessage::Open { cache_request_id, cached_tokens, reserved_rows, cache_hit, sampling, tail_sampling } => {
                    if tail_sampling && output_head.is_none() {
                        let (head, mtp) = prepare_tail_output_runtime(&output_context, &weights, &cfg, &mla, false, mtp_draft_tokens, None, lm_head_quantization)?;
                        output_head = Some(head);
                        mtp_runtime = mtp;
                    }
                    begin_tail_stage_open(
                        &mut link,
                        &mut active,
                        &mut resident,
                        &mut pending_opens,
                        &swap,
                        &output_context,
                        &templates,
                        &cfg,
                        args.max_seq_len,
                        TailOpenRequest { request_id, cache_request_id, cached_tokens, reserved_rows, cache_hit, sampling, tail_sampling },
                    )?;
                }
                StageMessage::MtpContext { prompt_tokens, max_decode, draft_tokens } => {
                    let session = active.get_mut(&request_id).ok_or_else(|| format!("tail MTP context request={request_id} 尚未 Open"))?;
                    tail_stage_mtp_context(session, mtp_runtime.is_some(), mtp_draft_tokens, &cfg, args.max_seq_len, request_id, prompt_tokens, max_decode, draft_tokens)?;
                }
                StageMessage::Prefill { position, rows, cols, values, selection, .. } => {
                    let session = active.get_mut(&request_id).ok_or_else(|| format!("tail prefill request={request_id} 尚未 Open"))?;
                    if position != session.position || cols != cfg.hidden_size {
                        return Err(format!("tail prefill 边界错误: request={request_id} position={position} expected={} shape=[{rows},{cols}]", session.position).into());
                    }
                    let selection = if selection.is_empty() { None } else { Some(RocmDsaSelection::from_host(&input_context, rows, position, &selection).map_err(|error| format!("上传 prefill DSA selection: {error:?}"))?) };
                    let hidden = input_context.tensor_from_bf16_bits_ordered(values, rows, cols).map_err(|error| format!("上传 tail prefill BF16: {error:?}"))?;
                    let mut output = None;
                    let states = std::mem::take(&mut session.states);
                    session.states = run_glm52_stage_pipeline_stateful(vec![Ok((position, hidden, selection))], states, &cfg, &mla, &rope, scheduler_policy, |_, result, _| {
                        output = Some(result);
                        Ok(())
                    })
                    .map_err(|error| format!("ROCm distributed tail prefill: {error:?}"))?;
                    session.hidden = output;
                    session.position += rows;
                }
                StageMessage::PrefillDone { tokens } => {
                    let session = active.get_mut(&request_id).ok_or_else(|| format!("tail PrefillDone request={request_id} 尚未 Open"))?;
                    if tokens != session.position {
                        return Err(format!("tail prefill tokens={tokens}，已接收={}", session.position).into());
                    }
                    let hidden = session.hidden.take().ok_or_else(|| if session.resumed { "tail cache 命中但没有 last hidden".to_owned() } else { "tail 新请求没有 prefill hidden".to_owned() })?;
                    let hidden = last_bf16_row(&output_context, hidden).map_err(|error| format!("取 tail prefill 最后一行: {error:?}"))?;
                    session.prompt_hidden = Some(hidden.clone());
                    session.prompt_position = Some(tokens);
                    let sampling = [session.sampling.next()];
                    let token = glm52_sampled_token_ids(&output_context, &cfg, output_head.as_ref().expect("legacy output head 已准备"), &hidden, &sampling)
                        .map_err(|error| format!("tail ROCm GLM output: {error:?}"))?
                        .into_iter()
                        .next()
                        .ok_or_else(|| format!("tail ROCm GLM output 缺少 token: request={request_id} sampling={} 条", sampling.len()))?;
                    let eos = cfg.eos_token_ids.contains(&token);
                    if let Some(mtp) = session.mtp.as_mut().filter(|mtp| mtp.active) {
                        mtp.verify_inputs.clear();
                        mtp.verify_inputs.push(token);
                    }
                    session.hidden = Some(hidden);
                    link.send_token(request_id, token, eos)?;
                    eprintln!("[stage-prefill-complete] request_id={request_id} resumed={} layers={stage_start}..{stage_end} tokens={tokens} wall={:.3}s", session.resumed, session.started.elapsed().as_secs_f64());
                }
                StageMessage::Decode { position, cols, values, selection, .. } => {
                    let session = active.get_mut(&request_id).ok_or_else(|| format!("tail decode request={request_id} 尚未 Open"))?;
                    if cols != cfg.hidden_size || position != session.position || position >= args.max_seq_len {
                        return Err(format!("tail decode 边界错误: request={request_id} position={position} expected={} cols={cols}", session.position).into());
                    }
                    let input = input_context.tensor_from_bf16_bits_ordered(values, 1, cols).map_err(|error| format!("上传 tail decode BF16: {error:?}"))?;
                    let selection = if selection.is_empty() { None } else { Some(RocmDsaSelection::from_host(&input_context, 1, position, &selection).map_err(|error| format!("上传 decode DSA selection: {error:?}"))?) };
                    let mut output = None;
                    let started = Instant::now();
                    let states = std::mem::take(&mut session.states);
                    session.states = run_glm52_stage_pipeline_stateful(vec![Ok((position, input, selection))], states, &cfg, &mla, &rope, scheduler_policy, |_, result, _| {
                        output = Some(result);
                        Ok(())
                    })
                    .map_err(|error| format!("ROCm distributed tail decode: {error:?}"))?;
                    let hidden = last_bf16_row(&output_context, output.ok_or("tail decode 没有 stage output")?).map_err(|error| format!("取 tail decode 最后一行: {error:?}"))?;
                    session.position += 1;
                    let sampling = [session.sampling.next()];
                    let token = glm52_sampled_token_ids(&output_context, &cfg, output_head.as_ref().expect("legacy output head 已准备"), &hidden, &sampling)
                        .map_err(|error| format!("tail ROCm GLM output: {error:?}"))?
                        .into_iter()
                        .next()
                        .ok_or_else(|| format!("tail ROCm GLM output 缺少 token: request={request_id} sampling={} 条", sampling.len()))?;
                    let eos = cfg.eos_token_ids.contains(&token);
                    session.hidden = Some(hidden);
                    link.send_token(request_id, token, eos)?;
                    eprintln!("[stage-decode-complete] request_id={request_id} position={position} wall={:.3}s active={}", started.elapsed().as_secs_f64(), active.len());
                }
                StageMessage::Cache { next_request_id, tokens } => {
                    let session = active.remove(&request_id).ok_or_else(|| format!("tail Cache request={request_id} 尚未 Open"))?;
                    tail_stage_cache_commit(&mut link, &mut resident, &output_context, session, request_id, next_request_id, tokens, false)?;
                    eprintln!("[stage-cache] request_id={request_id} cache_id={next_request_id} tokens={tokens} resident={}", resident.len());
                }
                StageMessage::SwapOut => {
                    tail_stage_swap_out(&mut link, &mut resident, &swap, &output_context, request_id)?;
                }
                StageMessage::Persist => {
                    tail_stage_persist(&mut resident, &swap, &output_context, request_id)?;
                }
                StageMessage::Shutdown { persist } => {
                    if !active.is_empty() {
                        eprintln!("[stage-shutdown-warn] 优雅退出时仍有 {} 个 active session，将由进程退出释放", active.len());
                    }
                    let persisted = if persist { tail_stage_persist_all(&mut resident, &swap, &output_context)? } else { 0 };
                    eprintln!("[stage-shutdown] persisted={persisted} resident={} active={}", resident.len(), active.len());
                    link.send_ready(request_id, 0)?;
                    // stage 进程没有其他服务职责。ACK 已 flush 后直接退出，
                    // 避免数百 GB resident ROCm 对象逐个析构卡在 HSA ioctl。
                    std::process::exit(0);
                }
                StageMessage::Delete => {
                    // 非 continuous 模式 Delete 不需要 ACK,丢弃返回的 position。
                    let _ = tail_stage_delete(&mut active, &mut resident, &swap, request_id)?;
                    eprintln!("[stage-delete] request_id={request_id} active={} resident={}", active.len(), resident.len());
                }
                StageMessage::StreamAssign { .. }
                | StageMessage::Verify { .. }
                | StageMessage::Token { .. }
                | StageMessage::Sample { .. }
                | StageMessage::Sampled { .. }
                | StageMessage::Speculative { .. }
                | StageMessage::Ready { .. }
                | StageMessage::DeviceMemory { .. } => {
                    return Err("tail stage 收到 stream 外控制或反向 frame".into());
                }
            }
        }
    }
    if pipeline_enabled {
        struct StageState {
            context: RocmContext,
            layer_start: usize,
            layers: Vec<PrefillLayer>,
            experts: RocmPrefillExperts,
            cache: RocmKvCache,
            dsa: RocmDsaState,
        }

        let mut layer_groups = (0..prefill_contexts.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (layer, resident) in prefill_layers.into_iter().enumerate() {
            let placement = prefill_layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| format!("L{layer} 没有 prefill pipeline stage"))?;
            layer_groups[placement].push(resident);
        }
        let mut expert_states = prefill_experts.into_iter();
        let mut layer_start = 0;
        let mut states = Vec::with_capacity(prefill_contexts.len());
        for (context, layers) in prefill_contexts.iter().copied().zip(layer_groups) {
            let current_start = layer_start;
            layer_start += layers.len();
            states.push(StageState {
                context,
                layer_start: current_start,
                layers,
                experts: expert_states.next().ok_or("prefill pipeline 缺少 expert state")?,
                cache: RocmKvCache::with_capacity(cfg.layer_count, args.max_seq_len),
                dsa: RocmDsaState::new(cfg.layer_count, args.max_seq_len, cfg.index_head_dim, cfg.index_top_k)?,
            });
        }
        let mut chunks = Vec::with_capacity(tokens.len().div_ceil(pipeline_chunk_size));
        for position in (0..tokens.len()).step_by(pipeline_chunk_size) {
            let end = (position + pipeline_chunk_size).min(tokens.len());
            let hidden = prefill_contexts[0].tensor_from_f32(weights.embedding_rows(&tokens[position..end])?, end - position, cfg.hidden_size)?;
            chunks.push((position, hidden));
        }
        let prefill_started = Instant::now();
        let outputs = run_token_chunk_stage_pipeline(chunks, states, |state, stage, position, hidden| {
            let started = Instant::now();
            let token_count = hidden.rows;
            let context = state.context;
            let layer_start = state.layer_start;
            let layer_end = layer_start + state.layers.len();
            context.activate().map_err(|msg| crate::backend::BackendError::Compute { msg: format!("prefill stage {stage} 激活 ROCm device 失败: {msg}") })?;
            let hidden = context.tensor_on_device(hidden)?;
            let layers = &state.layers;
            let cache = &mut state.cache;
            let dsa = &mut state.dsa;
            let hidden = glm52_prefill_stage(
                &context,
                &cfg,
                token_count,
                layer_start,
                layer_end,
                hidden,
                &mut state.experts,
                |_, _, _| Ok(()),
                |experts, layer, _kind, hidden| {
                    let hidden = context.tensor_as_f32(hidden)?;
                    let output = match &layers[layer - layer_start] {
                        PrefillLayer::Dense(resident) => glm52_dense_prefill_layer(&context, &cfg, &mla, resident, layer, Some(dsa), &hidden, &rope, Some(cache), position),
                        PrefillLayer::Moe(resident) => glm52_moe_prefill_layer(&context, &cfg, &mla, resident, layer, experts, None, Some(dsa), &hidden, &rope, Some(cache), position),
                    }?;
                    context.tensor_as_bf16(output)
                },
            )?;
            eprintln!("[prefill-chunk] stage={stage} device={} layers={layer_start}..{layer_end} position={position} rows={token_count} wall={:.3}s", context.device_id(), started.elapsed().as_secs_f64(),);
            Ok(hidden)
        })
        .map_err(|error| format!("ROCm chunk pipeline prefill: {error:?}"))?;
        let output_count = outputs.len();
        eprintln!("[prefill] layers=0..{stage_end} tokens={} chunks={} wall={:.3}s", tokens.len(), output_count, prefill_started.elapsed().as_secs_f64(),);
        if let Some(path) = diagnostics.output_artifact.as_deref() {
            let artifact_started = Instant::now();
            let header = StageArtifactHeader::new("glm52", 0, stage_end, tokens.len(), cfg.hidden_size)?;
            let mut artifact = StageArtifactWriter::create(path, header)?;
            let output_context = prefill_contexts.last().ok_or("prefill pipeline 没有输出设备")?;
            for (position, hidden) in outputs {
                let rows = hidden.rows;
                let values = output_context.tensor_to_bf16_bits(&hidden).map_err(|error| format!("下载 stage output position={position}: {error:?}"))?;
                artifact.write_bf16_chunk(position, rows, &values)?;
            }
            let path = artifact.finish()?;
            eprintln!("[stage-output] path={} tokens={} hidden={} bytes={} wall={:.3}s", path.display(), tokens.len(), cfg.hidden_size, tokens.len() * cfg.hidden_size * size_of::<u16>(), artifact_started.elapsed().as_secs_f64(),);
        }
        return Ok(());
    }

    let mut cache = RocmKvCache::with_capacity(cfg.layer_count, args.max_seq_len);
    let mut dsa_state = RocmDsaState::new(cfg.layer_count, args.max_seq_len, cfg.index_head_dim, cfg.index_top_k)?;
    let mut hidden = prefill_contexts[0].tensor_from_f32(weights.embedding_rows(&tokens)?, tokens.len(), cfg.hidden_size)?;
    let prefill_started = Instant::now();
    hidden = glm52_prefill_stage(
        backend,
        &cfg,
        tokens.len(),
        0,
        stage_end,
        hidden,
        &mut prefill_experts,
        |_, _, _| Ok(()),
        |experts, layer, _kind, hidden| {
            let layer_started = Instant::now();
            let placement = prefill_layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| crate::backend::BackendError::Compute { msg: format!("L{layer} 没有 prefill device") })?;
            let layer_backend = &prefill_contexts[placement];
            layer_backend.activate().map_err(|msg| crate::backend::BackendError::Compute { msg: format!("L{layer} 激活 ROCm device 失败: {msg}") })?;
            let hidden = layer_backend.tensor_as_f32(layer_backend.tensor_on_device(hidden)?)?;
            let output = match &prefill_layers[layer] {
                PrefillLayer::Dense(resident) => glm52_dense_prefill_layer(layer_backend, &cfg, &mla, &resident, layer, Some(&mut dsa_state), &hidden, &rope, Some(&mut cache), 0)?,
                PrefillLayer::Moe(resident) => glm52_moe_prefill_layer(layer_backend, &cfg, &mla, &resident, layer, &mut experts[0], None, Some(&mut dsa_state), &hidden, &rope, Some(&mut cache), 0)?,
            };
            let output = layer_backend.tensor_as_bf16(output)?;
            eprintln!("[prefill-layer] device={} layer={layer} rows={} wall={:.3}s", layer_backend.device_id(), output.rows, layer_started.elapsed().as_secs_f64());
            Ok(output)
        },
    )
    .map_err(|error| format!("ROCm stage prefill 0..{stage_end}: {error:?}"))?;
    eprintln!("[prefill] layers=0..{stage_end} tokens={} wall={:.3}s", tokens.len(), prefill_started.elapsed().as_secs_f64());

    if args.decode_steps == 0 {
        return Ok(());
    }
    let output_backend = prefill_contexts.last().ok_or("GLM prefill 没有输出 backend")?;
    let hidden_data = if mtp_enabled && args.decode_steps > 1 { Some(output_backend.tensor_to_f32(&hidden).map_err(|error| format!("下载 GLM MTP prompt hidden 失败: {error:?}"))?) } else { None };
    hidden = backend.tensor_on_device(output_backend.select_row(&hidden, tokens.len() - 1).map_err(|error| format!("选择 GLM prefill 末行: {error:?}"))?).map_err(|error| format!("迁移 GLM decode hidden: {error:?}"))?;
    let decode_expert_sources = super::decode_expert_sources(&weights, nvfp4_source.as_ref(), &args.model_dir, &cfg)?;

    let prepare_started = Instant::now();
    let layers = if args.decode_steps > 1 { prepare_glm52_decode_layers(backend, &cfg, &mla, &weights).map_err(|error| format!("准备 ROCm decode 层: {error:?}"))? } else { Vec::new() };
    eprintln!("[decode-prepare] layers={} wall={:.3}s", layers.len(), prepare_started.elapsed().as_secs_f64());
    if args.expert_cache_gib.is_some() {
        eprintln!("[expert] ROCm 当前按需读取 expert，忽略 --expert-cache-gib");
    }
    let expert_prefetch_count = args.expert_prefetch_count.unwrap_or(crate::runtime::DEFAULT_DECODE_PREFETCH_COUNT);
    let backend_state = UncachedMoeState::default();
    let mut moe_state = ExpertDecodePipeline::new(
        backend_state,
        ExpertPredictorConfig {
            first_layer: cfg.dense_layer_count,
            layer_count: cfg.layer_count - cfg.dense_layer_count,
            expert_count: cfg.expert_count,
            routed_top_k: cfg.expert_top_k,
            prefetch_count: expert_prefetch_count,
            weights: ExpertPredictorWeights::default(),
        },
    )?;
    eprintln!("[expert] policy=uncached prefetch={expert_prefetch_count}");

    let final_norm = weights.final_norm()?;
    let lm_head = weights.lm_head_bf16_bytes()?;
    let output_head = prepare_glm52_output_head(backend, &cfg, &final_norm, LinearWeight::Bf16Bytes(&lm_head)).map_err(|error| format!("准备 ROCm GLM output head: {error:?}"))?;
    let mtp_session = if mtp_enabled && args.decode_steps > 1 {
        let started = Instant::now();
        let mtp_weights = if weights.source_is_ct() {
            let layer = weights.ct_source()?.load_mtp_layer(cfg.layer_count)?;
            prepare_glm52_mtp_ct(backend, &cfg, &mla, &layer).map_err(|error| format!("准备 ROCm GLM MTP layer: {error:?}"))?
        } else if weights.source_is_gguf() {
            let layer = weights.load_mtp_layer_gguf()?;
            prepare_glm52_mtp_gguf(backend, &cfg, &mla, &layer).map_err(|error| format!("准备 ROCm GLM MTP layer: {error:?}"))?
        } else {
            return Err("GLM-5.2 MTP 当前仅支持 compressed-tensors/GGUF 权重".into());
        };
        let target_norm = backend.prepare_weight(LinearWeight::F32(&final_norm), 1, cfg.hidden_size).map_err(|error| format!("准备 ROCm GLM MTP target norm: {error:?}"))?;
        let mtp_layer_count = cfg.layer_count + cfg.mtp_layer_count;
        let mut mtp_cache = RocmKvCache::with_capacity(mtp_layer_count, args.max_seq_len);
        let mut mtp_dsa = RocmDsaState::new(mtp_layer_count, args.max_seq_len, cfg.index_head_dim, cfg.index_top_k)?;
        let mtp_backend_state = UncachedMoeState::default();
        let mut mtp_expert_state = ExpertDecodePipeline::new(
            mtp_backend_state,
            ExpertPredictorConfig { first_layer: cfg.layer_count, layer_count: cfg.mtp_layer_count, expert_count: cfg.expert_count, routed_top_k: cfg.expert_top_k, prefetch_count: 0, weights: ExpertPredictorWeights::default() },
        )?;
        let hidden_data = hidden_data.as_ref().expect("启用 MTP 时已下载 prompt hidden");
        for row in 0..tokens.len().saturating_sub(1) {
            let token_embedding = backend.tensor_from_f32(weights.embedding_rows(&[tokens[row + 1]])?, 1, cfg.hidden_size)?;
            let target_hidden = backend.tensor_from_f32(hidden_data[row * cfg.hidden_size..(row + 1) * cfg.hidden_size].to_vec(), 1, cfg.hidden_size)?;
            let target_hidden = backend.rmsnorm(&target_hidden, &target_norm, cfg.rms_eps).map_err(|error| format!("GLM MTP prompt target norm row={row}: {error:?}"))?;
            glm52_mtp_decode(backend, &cfg, &mla, &mtp_weights, &decode_expert_sources, &mut mtp_expert_state, &mut mtp_dsa, &mut mtp_cache, &token_embedding, &target_hidden, &rope, row)
                .map_err(|error| format!("GLM MTP prompt position={row}: {error:?}"))?;
        }
        eprintln!("[glm52-mtp-prepare] prompt_cache={} wall={:.3}s", tokens.len().saturating_sub(1), started.elapsed().as_secs_f64(),);
        Some((mtp_weights, mtp_cache, mtp_dsa, mtp_expert_state))
    } else {
        None
    };
    let detokenizer = Detokenizer::load(&tokenizer_path)?;
    if let Some((mtp_weights, mut mtp_cache, mut mtp_dsa, mut mtp_expert_state)) = mtp_session {
        let decode_started = Instant::now();
        let mut generated = 0_usize;
        let mut drafted = 0_usize;
        let mut accepted = 0_usize;
        let mut target_rounds = 0_usize;
        let mut target_position = tokens.len();
        let mut mtp_position = tokens.len().saturating_sub(1);
        while generated < args.decode_steps {
            let target_output = glm52_token_output(backend, &cfg, &output_head, &hidden).map_err(|error| format!("ROCm GLM output: {error:?}"))?;
            let token = target_output.token_id;
            print_token(&detokenizer, token, true)?;
            generated += 1;
            if cfg.eos_token_ids.contains(&token) || generated >= args.decode_steps || target_position >= args.max_seq_len {
                break;
            }

            let token_embedding = backend.tensor_from_f32(weights.embedding_rows(&[token])?, 1, cfg.hidden_size)?;
            let draft_hidden = glm52_mtp_decode(backend, &cfg, &mla, &mtp_weights, &decode_expert_sources, &mut mtp_expert_state, &mut mtp_dsa, &mut mtp_cache, &token_embedding, &target_output.input, &rope, mtp_position)
                .map_err(|error| format!("GLM MTP decode position={mtp_position}: {error:?}"))?;
            mtp_position += 1;
            let draft = glm52_mtp_token_output(backend, &output_head, &draft_hidden).map_err(|error| format!("GLM MTP output: {error:?}"))?;
            drafted += 1;

            let first_hidden = forward_token(token, target_position, backend, &cfg, &mla, &layers, &weights, &decode_expert_sources, &rope, &mut cache, &mut dsa_state, &mut moe_state)?;
            target_rounds += 1;
            if target_position + 2 > args.max_seq_len {
                hidden = first_hidden;
                target_position += 1;
                continue;
            }
            let target_token = glm52_token_output(backend, &cfg, &output_head, &first_hidden).map_err(|error| format!("GLM MTP target verify output: {error:?}"))?.token_id;
            eprintln!(
                "
[glm52-mtp-verify] next_step={} draft={draft} target={target_token} accepted={}",
                generated,
                target_token == draft,
            );
            if target_token == draft {
                hidden = forward_token(draft, target_position + 1, backend, &cfg, &mla, &layers, &weights, &decode_expert_sources, &rope, &mut cache, &mut dsa_state, &mut moe_state)?;
                target_rounds += 1;
                target_position += 2;
                accepted += 1;
                print_token(&detokenizer, draft, true)?;
                generated += 1;
                if cfg.eos_token_ids.contains(&draft) || generated >= args.decode_steps || target_position >= args.max_seq_len {
                    break;
                }
                let draft_embedding = backend.tensor_from_f32(weights.embedding_rows(&[draft])?, 1, cfg.hidden_size)?;
                glm52_mtp_decode(backend, &cfg, &mla, &mtp_weights, &decode_expert_sources, &mut mtp_expert_state, &mut mtp_dsa, &mut mtp_cache, &draft_embedding, &draft_hidden, &rope, mtp_position)
                    .map_err(|error| format!("GLM MTP advance position={mtp_position}: {error:?}"))?;
                mtp_position += 1;
            } else {
                hidden = first_hidden;
                target_position += 1;
            }
        }
        println!();
        let seconds = decode_started.elapsed().as_secs_f64();
        eprintln!(
            "[glm52-mtp-decode] tokens={generated} drafts={drafted} accepted={accepted} acceptance={:.1}% target_rounds={target_rounds} wall={seconds:.3}s throughput={:.3} tok/s",
            accepted as f64 * 100.0 / drafted.max(1) as f64,
            generated as f64 / seconds.max(f64::EPSILON),
        );
        print_expert_cache(&moe_state);
        return Ok(());
    }
    generation::run_generation(
        &mut hidden,
        generation::GenerationLimits { prompt_tokens: tokens.len(), max_tokens: args.decode_steps, max_sequence_length: args.max_seq_len, eos_tokens: &cfg.eos_token_ids },
        |hidden, _| Ok::<_, Box<dyn std::error::Error>>(glm52_token_output(backend, &cfg, &output_head, hidden).map_err(|error| format!("ROCm GLM output: {error:?}"))?.token_id),
        |token, _| generation::write_token(&detokenizer, token, true),
        |hidden, token, position, step| {
            let decode_started = Instant::now();
            *hidden = forward_token(token, position, backend, &cfg, &mla, &layers, &weights, &decode_expert_sources, &rope, &mut cache, &mut dsa_state, &mut moe_state)?;
            eprintln!("\n[decode {}] position={} wall={:.3}s", step + 1, position, decode_started.elapsed().as_secs_f64());
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    println!();
    print_expert_cache(&moe_state);
    Ok(())
}

#[path = "rocm_pipeline.rs"]
mod rocm_pipeline;
use rocm_pipeline::run_glm52_pipeline_node;
fn run_glm52_stage_input(backend: &RocmContext, args: &Args, cfg: &Glm52Config, mla: &MlaSpec, weights: &Glm52Weights, tokenizer_path: &Path, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if args.decode_steps > 1 {
        return Err("stage hidden 只包含 prefill 激活，不含 L0-L39 KV cache；独立终端阶段当前只能输出首个 decode token".into());
    }
    if !weights.source_is_ct() && !weights.source_is_gguf() {
        return Err("GLM stage-input 当前仅支持 compressed-tensors/GGUF 权重".into());
    }
    let mut artifact = StageArtifactReader::open(path)?;
    // 切分点由 artifact header 自带：前序 stage 覆盖了 0..layer_end，本终端从 layer_end 继续，
    // 只要求它是从 L0 开始的前缀且 hidden 与模型一致，不写死具体层号。
    let stage_start = artifact.header.layer_end;
    if artifact.header.layer_start != 0 || artifact.header.hidden_size != cfg.hidden_size || stage_start == 0 || stage_start >= cfg.layer_count {
        return Err(format!(
            "stage artifact 与模型前缀不匹配: layers={}..{} tokens={} hidden={}，期望 layers=0..1..{} hidden={}",
            artifact.header.layer_start, artifact.header.layer_end, artifact.header.token_count, artifact.header.hidden_size, cfg.layer_count, cfg.hidden_size,
        )
        .into());
    }
    let devices = args.prefill_devices.clone().unwrap_or_else(|| vec![backend.device_id()]);
    let layer_ends = args.prefill_layer_ends.clone().unwrap_or_else(|| vec![cfg.layer_count - 1]);
    if devices.len() != layer_ends.len() || layer_ends.last().copied() != Some(cfg.layer_count - 1) || layer_ends.windows(2).any(|pair| pair[0] >= pair[1]) || layer_ends.first().copied().unwrap_or(0) < stage_start {
        return Err("stage-input 的 --prefill-devices/--prefill-layer-ends 配置无效".into());
    }
    let contexts = devices.iter().map(|&device| backend.for_device(device).map_err(|error| format!("ROCm stage device {device} 初始化失败: {error}"))).collect::<Result<Vec<_>, _>>()?;
    let token_count = artifact.header.token_count;
    let capacity = token_count.checked_add(args.decode_steps).ok_or("stage cache capacity 溢出")?;
    let load_started = Instant::now();
    let values = artifact.read_all_bf16()?;
    let data = values.into_iter().map(|bits| half::bf16::from_bits(bits).to_f32()).collect::<Vec<_>>();
    let mut hidden = contexts[0].tensor_from_f32(data, token_count, cfg.hidden_size)?;
    eprintln!("[stage-input] path={} layers={stage_start}..{} tokens={token_count} hidden={} wall={:.3}s", path.display(), cfg.layer_count, cfg.hidden_size, load_started.elapsed().as_secs_f64(),);

    let rope = RopeTable::precompute(capacity, mla.qk_rope_head_dim, mla.rope_theta);
    let mut caches = (0..contexts.len()).map(|_| RocmKvCache::with_capacity(cfg.layer_count, capacity)).collect::<Vec<_>>();
    let mut dsa_states = (0..contexts.len()).map(|_| RocmDsaState::new(cfg.layer_count, capacity, cfg.index_head_dim, cfg.index_top_k)).collect::<Result<Vec<_>, _>>()?;
    let mut resident_layers = Vec::with_capacity(cfg.layer_count - stage_start);
    for layer in stage_start..cfg.layer_count {
        let placement = layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| format!("L{layer} 没有 stage device"))?;
        let layer_backend = &contexts[placement];
        layer_backend.activate()?;
        let layer_weights = crate::runtime::glm52::load_prepare_moe_prefill_layer(layer_backend, cfg, mla, weights, layer, false).map_err(|error| format!("准备 ROCm stage L{layer}: {error:?}"))?;
        resident_layers.push((placement, layer_weights));
    }
    let mut experts = contexts
        .iter()
        .map(|_| -> Result<RocmPrefillExperts, Box<dyn std::error::Error>> {
            if weights.source_is_gguf() { Ok(RocmPrefillExperts::gguf(weights.gguf_source()?)) } else { Ok(RocmPrefillExperts::ct(weights.ct_source()?)) }
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (offset, (placement, _)) in resident_layers.iter().enumerate() {
        let layer = stage_start + offset;
        let layer_backend = &contexts[*placement];
        layer_backend.activate()?;
        experts[*placement].preload_layer(layer_backend, layer, cfg.expert_count).map_err(|error| format!("常驻 ROCm stage L{layer} experts: {error:?}"))?;
    }
    let prefill_started = Instant::now();
    for (offset, (placement, resident)) in resident_layers.iter().enumerate() {
        let layer = stage_start + offset;
        let layer_started = Instant::now();
        let layer_backend = &contexts[*placement];
        layer_backend.activate()?;
        let dsa_state = &mut dsa_states[*placement];
        let cache = &mut caches[*placement];
        hidden = layer_backend.tensor_on_device(hidden).and_then(|hidden| layer_backend.tensor_as_f32(hidden)).map_err(|error| format!("迁移 ROCm stage L{layer} hidden: {error:?}"))?;
        hidden = glm52_moe_prefill_layer(layer_backend, cfg, mla, resident, layer, &mut experts[*placement], None, Some(dsa_state), &hidden, &rope, Some(cache), 0).map_err(|error| format!("ROCm stage prefill L{layer}: {error:?}"))?;
        hidden = layer_backend.tensor_as_bf16(hidden).map_err(|error| format!("ROCm stage L{layer} 转 BF16: {error:?}"))?;
        eprintln!("[stage-prefill-layer] layer={layer} rows={token_count} wall={:.3}s", layer_started.elapsed().as_secs_f64(),);
    }
    eprintln!("[stage-prefill] layers={stage_start}..{} tokens={token_count} wall={:.3}s", cfg.layer_count, prefill_started.elapsed().as_secs_f64(),);
    if args.decode_steps == 0 {
        return Ok(());
    }

    let output_backend = contexts.last().ok_or("stage-input 没有输出设备")?;
    output_backend.activate()?;
    let hidden = output_backend.select_row(&hidden, token_count - 1).map_err(|error| format!("ROCm stage 选择末行: {error:?}"))?;
    let final_norm = weights.final_norm()?;
    let lm_head = weights.lm_head_bf16_bytes()?;
    let output_head = prepare_glm52_output_head(output_backend, cfg, &final_norm, LinearWeight::Bf16Bytes(&lm_head)).map_err(|error| format!("准备 ROCm stage output head: {error:?}"))?;
    let token = glm52_token_output(output_backend, cfg, &output_head, &hidden).map_err(|error| format!("ROCm stage first token: {error:?}"))?.token_id;
    let detokenizer = Detokenizer::load(tokenizer_path)?;
    print_token(&detokenizer, token, true)?;
    println!();
    eprintln!("[stage-decode] token={token} position={token_count}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_token(
    token: u32,
    position: usize,
    backend: &RocmContext,
    cfg: &Glm52Config,
    mla: &MlaSpec,
    layers: &[Glm52DecodeLayer<RocmWeight>],
    weights: &Glm52Weights,
    expert_sources: &Glm52ExpertSources,
    rope: &RopeTable,
    cache: &mut RocmKvCache,
    dsa_state: &mut RocmDsaState,
    moe_state: &mut ExpertDecodePipeline<UncachedMoeState>,
) -> Result<RocmTensor, Box<dyn std::error::Error>> {
    let hidden = backend.tensor_from_f32(weights.embedding_rows(&[token])?, 1, cfg.hidden_size)?;
    glm52_decode_layers(backend, cfg, mla, layers, expert_sources, moe_state, dsa_state, hidden, rope, cache, position).map_err(|error| format!("ROCm decode position={position}: {error:?}").into())
}

struct Args {
    model_dir: PathBuf,
    prompt: String,
    max_seq_len: usize,
    decode_steps: usize,
    expert_cache_gib: Option<usize>,
    expert_prefetch_count: Option<usize>,
    nvfp4_root: Option<PathBuf>,
    ct_root: Option<PathBuf>,
    gguf_root: Option<PathBuf>,
    prefill_devices: Option<Vec<i32>>,
    prefill_layer_ends: Option<Vec<usize>>,
    start_layer: Option<usize>,
    end_layer: Option<usize>,
    next_node: Option<String>,
    decode_node: Option<String>,
    cache_dir: Option<PathBuf>,
    pipeline_iroh: Option<IrohConfig>,
}
