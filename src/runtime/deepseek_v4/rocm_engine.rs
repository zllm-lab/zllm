//! DeepSeek-V4 ROCm 八卡常驻执行体，CLI 与正式 node 共用。

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    attention::compressed_sparse::CompressedSparseKernel,
    backend::{
        BackendError, SegmentedTensorBackend, TokenFence,
        rocm::{RocmContext, RocmPrefillExperts, RocmTensor, RocmWeight},
    },
    runtime::deepseek_v4::{
        DeepSeekV4, DeepSeekV4Config, DeepSeekV4LayerCache, DeepSeekV4OutputHead, DeepSeekV4RopeTables, deepseek_v4_token_output, deepseek_v4_token_outputs,
        dspark_rocm::{RocmDeepSeekV4Dspark, RocmDeepSeekV4DsparkCache, RocmDeepSeekV4DsparkSession},
        prepare_deepseek_v4_gguf_layer, prepare_deepseek_v4_layer,
        rocm_stage::{DeepSeekV4ReadyForward, DeepSeekV4StageCache, DeepSeekV4StageEvent, DeepSeekV4StagePipeline, DeepSeekV4StageState, DeepSeekV4StageTemplate, allocate_stage_caches, drive_deepseek_v4_stage_pipeline, open_stage_session},
        rocm_swap::{DeepSeekV4CacheSnapshot, DeepSeekV4SwapStore},
        vision::{DeepseekV41SpanOverlays, DeepseekV41VisionInput, deepseek_v41_vision_encode, span_rows_f32},
    },
    runtime::rocm_chain,
    runtime::session::KvCacheDeviceCapacity,
    runtime::{
        generation_guard::{GenerationGuard, LoopKind, TokenFenceProgram},
        json_fence::JsonTokenTable,
        multiplex::{BatchScheduler, RequestPhase},
        prefill::{OpportunisticPrefillAdmission, prefill_token_segments},
        tool::{DsmlToolFence, DsmlToolSpec},
    },
    tokenizer::{Detokenizer, Tokenizer},
    vision::ImageTokenRange,
    weight::{
        expert_source::GgufExpertSource,
        model::deepseek_v4::{DeepSeekV4Gguf, DeepSeekV4Weights},
    },
};

/// GGUF 与官方 safetensors 只在装配处不同，执行路径完全共享。
enum Source {
    Gguf(DeepSeekV4Gguf),
    Official(DeepSeekV4Weights),
}

impl Source {
    fn open(root: &Path, config: DeepSeekV4Config) -> Result<Self, String> {
        let official = std::fs::read_dir(root).map(|entries| entries.filter_map(Result::ok).any(|entry| entry.path().extension().is_some_and(|extension| extension == "safetensors"))).unwrap_or(false);
        if official { Ok(Self::Official(DeepSeekV4Weights::open(root, config)?)) } else { Ok(Self::Gguf(DeepSeekV4Gguf::open(root, config)?)) }
    }

    /// safetensors 权重的引用(engram 装配需要);GGUF 源为 None。
    fn official_weights(&self) -> Option<&DeepSeekV4Weights> {
        match self {
            Self::Official(weights) => Some(weights),
            Self::Gguf(_) => None,
        }
    }

    fn tokenizer(&self, root: &Path) -> Result<(Tokenizer, Detokenizer), String> {
        match self {
            Self::Gguf(source) => Ok((source.tokenizer()?, source.detokenizer()?)),
            Self::Official(_) => {
                let path = root.join("tokenizer.json");
                Ok((Tokenizer::new(&path).map_err(|error| format!("加载 tokenizer: {error}"))?, Detokenizer::load(&path).map_err(|error| format!("加载 detokenizer: {error}"))?))
            }
        }
    }

    fn embedding_rows(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        match self {
            Self::Gguf(source) => source.embedding_rows(token_ids),
            Self::Official(source) => Ok(source.embedding_rows_bf16(token_ids)?.chunks_exact(2).map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect()),
        }
    }

    fn layer_storage_bytes(&self, layer: usize) -> Result<usize, String> {
        match self {
            Self::Gguf(source) => source.layer_storage_bytes(layer),
            Self::Official(source) => source.layer_storage_bytes(layer),
        }
    }

    fn expert_source(&self) -> Result<Arc<dyn crate::weight::expert_source::Mxfp4ExpertSource>, String> {
        match self {
            Self::Gguf(_) => Err("GGUF 来源不提供 MXFP4 专家".to_owned()),
            Self::Official(source) => Ok(Arc::new(source.expert_source())),
        }
    }

    fn output_head(&self, backend: &RocmContext) -> Result<DeepSeekV4OutputHead<RocmWeight>, BackendError> {
        match self {
            Self::Gguf(source) => crate::runtime::deepseek_v4::prepare_deepseek_v4_gguf_output_head(backend, source),
            Self::Official(source) => crate::runtime::deepseek_v4::prepare_deepseek_v4_official_output_head(backend, source),
        }
    }
}

/// 按权重目录内容探测 V4/V4.1:safetensors 看 config.json 的 model_type,GGUF 看 architecture。
fn detect_flash_model(root: &Path) -> Result<DeepSeekV4, String> {
    let config_json = root.join("config.json");
    if config_json.exists() {
        let text = std::fs::read_to_string(&config_json).map_err(|error| format!("读取 {config_json:?}: {error}"))?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|error| format!("解析 {config_json:?}: {error}"))?;
        let is_v41 = value.get("model_type").and_then(serde_json::Value::as_str) == Some("deepseek_v41");
        eprintln!("[load] DeepSeek-V4 权重探测: model_type={} → {}", value.get("model_type").and_then(serde_json::Value::as_str).unwrap_or("?"), if is_v41 { "V4.1-Flash" } else { "V4-Flash" });
        return Ok(if is_v41 { DeepSeekV4::flash_v41() } else { DeepSeekV4::flash() });
    }
    if let Ok(path) = crate::weight::container::gguf::GgufReader::locate(root) {
        let reader = crate::weight::container::gguf::GgufReader::open(&path)?;
        if let Some(arch) = reader.metadata("general.architecture").and_then(crate::weight::container::gguf::GgufValue::as_str) {
            let is_v41 = arch == "deepseek41";
            eprintln!("[load] DeepSeek-V4 权重探测: architecture={arch} → {}", if is_v41 { "V4.1-Flash" } else { "V4-Flash" });
            return Ok(if is_v41 { DeepSeekV4::flash_v41() } else { DeepSeekV4::flash() });
        }
    }
    Ok(DeepSeekV4::flash())
}

#[derive(Clone)]
pub struct RocmDeepSeekV4Options {
    pub weights_directory: PathBuf,
    pub cache_directory: PathBuf,
    pub devices: Vec<i32>,
    pub layer_ends: Vec<usize>,
    pub max_sequence_length: usize,
    pub core_cache_gib: usize,
    pub prefill_chunk_size: usize,
    pub decode_priority_prefill_chunk_size: usize,
    pub decode_priority_prefill_chunk_ceiling: usize,
    pub device_pool_gib: usize,
    pub device_arena_gib: usize,
    pub kv_reservation_page_tokens: usize,
    pub memory_reserve_bytes: usize,
    pub long_prefill_threshold_tokens: Option<usize>,
    pub long_prefill_chunk_size: Option<usize>,
    pub dspark: bool,
    pub dspark_draft_tokens: Option<usize>,
    pub dspark_min_sessions: usize,
    pub dspark_confidence_threshold: Option<f32>,
    pub decode_batch_limit: usize,
    pub terminal_cache_global_entries: usize,
    pub terminal_cache_prefix_rounds: usize,
    pub persist_kv_cache: bool,
    pub score_expert_top_k: Option<usize>,
    pub profile: bool,
    /// V4.1 engram CPU 注入开关;false 时跳过(二分定位/对照用)。
    pub engram_enabled: bool,
}

pub struct RocmGeneration {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
}

pub const DEEPSEEK_V4_MAX_CONCURRENCY: usize = 8;

pub struct RocmBatchRequest {
    pub request_id: String,
    pub prompt: String,
    pub requested_tokens: usize,
    pub cancellation: Arc<AtomicBool>,
    pub cache_id: Option<String>,
    pub resume_suffix: Option<String>,
    /// 瘦身 resume 请求(messages 只有 [边界 assistant, ...增量]):命中仅走
    /// cache_id 精确恢复,残缺 prompt 不参与前缀复用,miss 哨兵上报。
    pub slim_resume: bool,
    pub cache_namespace: Option<String>,
    pub tool_fence: Option<DsmlToolSpec>,
    pub repeat_loop_breaker: bool,
    /// V4.1 多模态:prompt 内已含 image 占位符的解码图像;engine 侧展开
    /// span、跑视觉塔并拼接 embedding。None 走纯文本。
    pub vision_images: Option<Vec<crate::vision::RgbImage>>,
}

pub struct RocmBatchResult {
    pub request_id: String,
    pub result: Result<RocmGeneration, String>,
}

struct BatchTask {
    request_id: String,
    cancellation: Arc<AtomicBool>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    chunk_policy: crate::runtime::prefill::AdaptiveChunkPolicy,
    next_prefill: usize,
    last_hidden: Option<RocmTensor>,
    generated: Vec<u32>,
    finish_reason: String,
    stats: crate::runtime::speculative::SpeculativeStats,
    cache_id: Option<String>,
    resume_suffix_tokens: Option<Vec<u32>>,
    slim_resume: bool,
    cache_namespace: Option<String>,
    cached_tokens: Vec<u32>,
    resumed_cache_id: Option<String>,
    resumed_cache_round: Option<usize>,
    token_fence: GenerationGuard<Option<DsmlToolFence>>,
    vision: Option<Arc<DeepseekV41SpanOverlays>>,
}

struct RocmSessionState {
    stages: Vec<DeepSeekV4StageState>,
    dspark: Option<RocmDeepSeekV4DsparkSession>,
}

impl RocmSessionState {
    fn fork_session(&self) -> Result<Self, String> {
        // engram 统一 fork 一次,全部 stage 共享(此前各 stage fork_session 只是 clone 占位)
        let engram = self
            .stages
            .first()
            .and_then(|stage| stage.engram.clone())
            .map(|engram| {
                let forked = engram.lock().map_err(|_| "engram 锁中毒".to_owned())?.fork();
                Ok::<_, String>(Arc::new(Mutex::new(forked)))
            })
            .transpose()?;
        let v41_caches = self.stages.first().map(DeepSeekV4StageState::fork_v41_caches).transpose().map_err(backend_error)?.flatten();
        let stages = self
            .stages
            .iter()
            .map(|stage| {
                let mut stage = stage.fork_session_with_v41(v41_caches.clone()).map_err(backend_error)?;
                stage.engram = engram.clone();
                Ok::<_, String>(stage)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { stages, dspark: self.dspark.as_ref().map(RocmDeepSeekV4DsparkSession::fork_session).transpose().map_err(backend_error)? })
    }

    fn reset_session(&mut self) {
        for stage in &mut self.stages {
            stage.reset_session();
        }
        if let Some(dspark) = &mut self.dspark {
            dspark.reset_session();
        }
    }

    fn allocated_bytes(&self) -> u64 {
        self.stages.iter().map(DeepSeekV4StageState::cache_allocated_bytes).sum::<u64>().saturating_add(self.dspark.as_ref().map_or(0, RocmDeepSeekV4DsparkSession::allocated_bytes))
    }

    fn download_cache(&self) -> Result<(Vec<DeepSeekV4StageCache>, Option<RocmDeepSeekV4DsparkCache>), String> {
        let stages = self.stages.iter().map(DeepSeekV4StageState::download_cache).collect::<Result<_, _>>().map_err(backend_error)?;
        let dspark = self.dspark.as_ref().map(RocmDeepSeekV4DsparkSession::download_cache).transpose().map_err(backend_error)?;
        Ok((stages, dspark))
    }
}

struct RocmTerminalState {
    session: RocmSessionState,
    cache_namespace: Option<String>,
    bytes: u64,
    modified_unix: u64,
}

struct RocmMaterializedSession {
    session: RocmSessionState,
    cache_namespace: Option<String>,
}

impl crate::kv_cache::terminal_cache::SharedBlockGraph for RocmTerminalState {
    type Session = RocmMaterializedSession;
    type Error = String;

    fn materialize(&self) -> Result<Self::Session, Self::Error> {
        Ok(RocmMaterializedSession { session: self.session.fork_session()?, cache_namespace: self.cache_namespace.clone() })
    }
}

pub struct RocmTerminalCacheInfo {
    pub cache_id: String,
    pub prompt_tokens: usize,
    pub bytes: u64,
    pub modified_unix: u64,
}

struct GpuTier {
    context: RocmContext,
    layer_cache: DeepSeekV4LayerCache<RocmWeight>,
    experts: RocmPrefillExperts,
    caches: Vec<crate::backend::rocm::RocmCompressedKvStorage>,
}

pub struct RocmDeepSeekV4Engine {
    options: RocmDeepSeekV4Options,
    tokenizer: Arc<Tokenizer>,
    detokenizer: Detokenizer,
    json_tokens: Arc<JsonTokenTable>,
    output_head: DeepSeekV4OutputHead<RocmWeight>,
    last_context: RocmContext,
    entry_context: RocmContext,
    model: Arc<DeepSeekV4>,
    source: Arc<Source>,
    stage_templates: Vec<DeepSeekV4StageTemplate>,
    reusable_sessions: Vec<RocmSessionState>,
    terminal_sessions: crate::kv_cache::terminal_cache::SharedBlockCache<RocmTerminalState>,
    /// load 时快照的每卡 KV token 容量；权重与常驻状态不再变化，无需重估。
    kv_cache_devices: Vec<KvCacheDeviceCapacity>,
    swap: Option<DeepSeekV4SwapStore>,
    pending_sessions: HashMap<String, (Vec<u32>, RocmSessionState, Option<String>, Option<(String, usize)>)>,
    dspark: Option<Arc<Mutex<RocmDeepSeekV4Dspark>>>,
    dspark_window: Option<usize>,
}

impl RocmDeepSeekV4Engine {
    pub fn load(options: RocmDeepSeekV4Options) -> Result<Self, String> {
        if options.devices.is_empty() || options.max_sequence_length == 0 || options.prefill_chunk_size == 0 {
            return Err("DeepSeek-V4 ROCm node 配置为空".to_owned());
        }
        let root = options.weights_directory.as_path();
        let mut model = detect_flash_model(root)?;
        if let Some(top_k) = options.score_expert_top_k {
            model = model.with_score_expert_top_k(top_k).map_err(|error| format!("DeepSeek-V4 score expert top-k: {error}"))?;
        }
        let model = Arc::new(model);
        let source = Arc::new(Source::open(root, model.config().clone())?);
        let (tokenizer, detokenizer) = source.tokenizer(root)?;
        let tokenizer = Arc::new(tokenizer);
        let json_tokens = JsonTokenTable::new(&detokenizer, model.config().vocab_size)?;
        let layer_count = model.layer_count();
        if options.layer_ends.len() != options.devices.len() || options.layer_ends.last() != Some(&layer_count) || options.layer_ends.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(format!("layer_ends={:?} 与 devices={:?} 不匹配", options.layer_ends, options.devices));
        }
        let dspark_config = options.dspark.then(|| crate::weight::model::deepseek_v4_dspark::DeepSeekV4DsparkConfig::read(root, model.config())).transpose()?;
        let capture_plan = dspark_config.as_ref().map(|config| config.capture_plan(model.config())).transpose()?.map(Arc::new);
        let gib = 1024usize * 1024 * 1024;
        crate::kernel::rocm::hip::set_device_buffer_pool_limit(options.device_pool_gib.checked_mul(gib).ok_or("device_pool_gib 溢出")?)?;
        crate::kernel::rocm::hip::set_arena_segment_bytes(options.device_arena_gib.checked_mul(gib).ok_or("device_arena_gib 溢出")?);
        crate::kernel::rocm::hip::enable_device_buffer_reuse();
        for target in &options.devices {
            let context = RocmContext::new(*target).map_err(|error| format!("初始化 ROCm 设备 {target}: {error}"))?;
            for source_device in &options.devices {
                if source_device != target {
                    context.enable_peer_access_from(*source_device).map_err(|error| format!("P2P {source_device}->{target}: {error}"))?;
                }
            }
        }
        let rope_extra = dspark_config.as_ref().map_or(0, |config| config.dspark_block_size);
        let rope = Arc::new(DeepSeekV4RopeTables::new(&model, options.max_sequence_length.checked_add(rope_extra).ok_or("sequence capacity 溢出")?).map_err(backend_error)?);
        let started = Instant::now();
        let mut handles = Vec::with_capacity(options.devices.len());
        for (tier_index, &device) in options.devices.iter().enumerate() {
            let layer_start = if tier_index == 0 { 0 } else { options.layer_ends[tier_index - 1] };
            let layer_end = options.layer_ends[tier_index];
            let source = source.clone();
            let model = model.clone();
            let core_cache_gib = options.core_cache_gib;
            handles.push(std::thread::spawn(move || -> Result<(usize, GpuTier), String> {
                let context = RocmContext::new(device).map_err(|error| format!("初始化 ROCm 设备 {device}: {error}"))?;
                let mut layer_bytes = vec![usize::MAX; layer_count];
                for layer in layer_start..layer_end {
                    layer_bytes[layer] = source.layer_storage_bytes(layer)?;
                }
                let mut layer_cache = DeepSeekV4LayerCache::with_layer_bytes(layer_bytes, core_cache_gib.saturating_mul(gib));
                // V4.1 的 CSA cache 在会话级全局表(见下方 v41_table),stage 不再重复分配
                let caches = if model.is_v41() { Vec::new() } else { allocate_stage_caches(&context, &model, layer_start, layer_end).map_err(backend_error)? };
                let experts = build_experts(&source, &context, layer_start, layer_end, model.config())?;
                for layer in layer_start..layer_end {
                    prepare_layer(&source, &context, layer, &mut layer_cache).map_err(backend_error)?;
                }
                eprintln!("[tier {tier_index}] device={device} layers={layer_start}..{layer_end} resident_layers={}", layer_cache.planned_layers());
                Ok((tier_index, GpuTier { context, layer_cache, experts, caches }))
            }));
        }
        let mut tiers = handles.into_iter().map(|handle| handle.join().map_err(|_| "ROCm tier 装配线程 panic".to_owned())?).collect::<Result<Vec<_>, String>>()?;
        tiers.sort_by_key(|(tier_index, _)| *tier_index);
        let tiers = tiers.into_iter().map(|(_, tier)| tier).collect::<Vec<_>>();
        eprintln!("[load] DeepSeek-V4 全部层与专家驻留完成 {:.1}s", started.elapsed().as_secs_f64());
        let last_context = tiers.last().expect("tiers 非空").context;
        let output_head = source.output_head(&last_context).map_err(backend_error)?;
        let dspark = if options.dspark {
            let checkpoint = crate::weight::model::deepseek_v4_dspark::DeepSeekV4DsparkCheckpoint::open(root, model.config().clone())?;
            let mut candidates = options
                .devices
                .iter()
                .enumerate()
                .map(|(index, &device)| {
                    let start = if index == 0 { 0 } else { options.layer_ends[index - 1] };
                    (options.layer_ends[index] - start, index, device)
                })
                .collect::<Vec<_>>();
            // 末 stage 还承担输出头与采样；即使少一层，也不要优先叠加辅助模型。
            candidates.sort_by_key(|&(layers, index, _)| (index == 0 || index + 1 == options.devices.len(), layers, index));
            let dspark_devices = (0..checkpoint.config.dspark_target_layer_ids.len()).map(|index| candidates[index % candidates.len()].2).collect::<Vec<_>>();
            Some(Arc::new(Mutex::new(
                RocmDeepSeekV4Dspark::load(last_context, checkpoint, &output_head, rope.clone(), &dspark_devices, options.dspark_draft_tokens, options.dspark_confidence_threshold, options.profile).map_err(backend_error)?,
            )))
        } else {
            None
        };
        let dspark_window = dspark.as_ref().map(|runtime| runtime.lock().map_err(|_| "DSpark runtime mutex poisoned".to_owned())?.target_window_size().map_err(backend_error)).transpose()?;
        let entry_context = tiers[0].context;
        // V4.1 engram CPU 母版:wkv BF16 驻内存(算法与数据都不进显存,用户决策);
        // 会话 fork 时 Arc 共享权重、hash 状态重置。
        let engram = if model.is_v41() && options.engram_enabled {
            let weights = source.official_weights().ok_or("V4.1 engram 需要 safetensors 权重源(GGUF 不支持)")?;
            let token_map = root.join("engram_token_map.bin");
            let cpu = crate::runtime::deepseek_v4::engram_cpu::DeepSeekV4EngramCpu::load(&token_map, weights.clone(), model.config().hyper_connection_copies, model.config().hidden_size, model.config().rms_eps)?;
            eprintln!("[load] engram CPU 算子就绪(2 层 wkv BF16 驻内存,token_map={token_map:?})");
            Some(Arc::new(Mutex::new(cpu)))
        } else {
            None
        };
        // V4.1 会话级全层 CSA cache 母版表:40 层各归属其 stage 的 device,
        // 会话 fork 时逐层 fork_session(compressed 前缀 Arc 共享、recent 私有化)。
        let v41_table = if model.is_v41() {
            let mut table = Vec::with_capacity(layer_count);
            let mut start = 0;
            for (tier_index, &end) in options.layer_ends.iter().enumerate() {
                for layer in start..end {
                    let spec = model.layer_spec(layer).map_err(|error| format!("V4.1 layer {layer} spec: {error}"))?;
                    table.push(Mutex::new(tiers[tier_index].context.allocate_compressed_kv(&spec.attention).map_err(backend_error)?));
                }
                start = end;
            }
            if table.len() != layer_count {
                return Err(format!("V4.1 cache 表 {} 层与 layer_count={layer_count} 不符", table.len()));
            }
            eprintln!("[load] V4.1 会话级 CSA cache 表 {} 层建立(P2P 共享)", table.len());
            Some(Arc::new(table))
        } else {
            None
        };
        let mut layer_start = 0usize;
        let mut stage_contexts = Vec::with_capacity(tiers.len());
        let resident_states: Vec<DeepSeekV4StageState> = tiers
            .into_iter()
            .map(|tier| {
                stage_contexts.push(tier.context);
                let state = DeepSeekV4StageState::new(
                    tier.context,
                    model.clone(),
                    rope.clone(),
                    layer_start,
                    options.layer_ends[stage_contexts.len() - 1] - layer_start,
                    tier.layer_cache,
                    tier.experts,
                    tier.caches,
                    v41_table.clone(),
                    engram.clone(),
                    capture_plan.clone(),
                    options.profile,
                );
                layer_start += state.layer_count();
                state
            })
            .collect();
        let stage_templates = resident_states.iter().map(DeepSeekV4StageState::template).collect();
        let reusable_sessions = vec![RocmSessionState { stages: resident_states, dspark: None }];
        let terminal_sessions = crate::kv_cache::terminal_cache::SharedBlockCache::new(options.terminal_cache_global_entries, options.terminal_cache_prefix_rounds);
        let swap = options
            .persist_kv_cache
            .then(|| {
                let metadata = vec![
                    ("schema_version".to_owned(), "2".to_owned()),
                    ("architecture".to_owned(), "deepseek-v4-flash".to_owned()),
                    ("weights".to_owned(), options.weights_directory.to_string_lossy().into_owned()),
                    ("max_sequence_length".to_owned(), options.max_sequence_length.to_string()),
                    ("layer_ends".to_owned(), format!("{:?}", options.layer_ends)),
                    ("dspark".to_owned(), options.dspark.to_string()),
                ];
                DeepSeekV4SwapStore::open(options.cache_directory.join("deepseek-v4-terminal"), &metadata)
            })
            .transpose()?;
        // arena 是固定预留段；在 KV 容量快照前建立，避免先高估容量、首轮
        // prefill 再突然吃掉 arena 段。
        for &device in &options.devices {
            crate::kernel::rocm::hip::reserve_device_arena(device).map_err(|error| format!("DeepSeek-V4 device {device} arena: {error}"))?;
        }
        // 权重、输出头、DSpark、首个 session 与 arena 全部驻留后再查空闲显存，逐卡按
        // CSA 线性 bytes/token 折算 KV token 容量；调度器据此对准入按 token
        // 记账，而不是只数请求数。卡内各层压缩比不同，先对层求和再调用
        // （layers=1），保持与 rocm_chain 统一的安全水位与 total/2 封顶口径。
        let mut layer_start = 0usize;
        let mut kv_cache_devices = Vec::with_capacity(stage_contexts.len());
        for (context, &layer_end) in stage_contexts.iter().zip(&options.layer_ends) {
            let bytes_per_token = (layer_start..layer_end).map(|layer| model.kv_bytes_per_layer_token(layer)).sum::<usize>().max(1);
            layer_start = layer_end;
            let device = rocm_chain::kv_device_capacity(context, format!("rocm-device-{}", context.device_id()), 1, bytes_per_token, options.memory_reserve_bytes, 0).map_err(|error| format!("DeepSeek-V4 KV 容量估算: {error}"))?;
            eprintln!("[deepseek-v4] device {} kv_budget={:.2}GiB bytes/token={} token_capacity={}", device.device, device.available_bytes as f64 / (1_u64 << 30) as f64, device.bytes_per_token, device.token_capacity);
            kv_cache_devices.push(device);
        }
        Ok(Self {
            options,
            tokenizer,
            detokenizer,
            json_tokens,
            output_head,
            last_context,
            entry_context,
            model,
            source,
            stage_templates,
            reusable_sessions,
            terminal_sessions,
            swap,
            kv_cache_devices,
            pending_sessions: HashMap::new(),
            dspark,
            dspark_window,
        })
    }

    /// terminal cache 的模型隔离键:V4.1 权重与 V4 不同,cache 不可混用。
    pub fn cache_model_key(&self) -> &'static str {
        if self.model.is_v41() { "deepseek-v41-flash" } else { "deepseek-v4-flash" }
    }

    pub fn layer_count(&self) -> usize {
        self.model.layer_count()
    }

    pub fn max_sequence_length(&self) -> usize {
        self.options.max_sequence_length
    }
    pub fn dspark_enabled(&self) -> bool {
        self.options.dspark
    }
    pub fn kv_cache_devices(&self) -> &[KvCacheDeviceCapacity] {
        &self.kv_cache_devices
    }
    pub fn kv_reservation_page_tokens(&self) -> usize {
        self.options.kv_reservation_page_tokens
    }

    fn decode_admission_blocker(&self) -> Result<Option<(i32, usize)>, (i32, String)> {
        for &device in &self.options.devices {
            match crate::kernel::rocm::hip::device_admission_available_bytes(device) {
                Ok(available) if available >= self.options.memory_reserve_bytes => {}
                Ok(available) => return Ok(Some((device, available))),
                Err(error) => return Err((device, error)),
            }
        }
        Ok(None)
    }

    pub fn terminal_cache_infos(&self) -> Vec<RocmTerminalCacheInfo> {
        let mut infos = self
            .terminal_sessions
            .entries()
            .map(|(cache_id, tokens, state)| (cache_id.to_owned(), RocmTerminalCacheInfo { cache_id: cache_id.to_owned(), prompt_tokens: tokens.len(), bytes: state.bytes, modified_unix: state.modified_unix }))
            .collect::<HashMap<_, _>>();
        if let Some(swap) = &self.swap {
            for swapped in swap.infos() {
                infos.entry(swapped.cache_id.clone()).or_insert(RocmTerminalCacheInfo { cache_id: swapped.cache_id, prompt_tokens: swapped.token_count, bytes: swapped.resident_bytes, modified_unix: swapped.modified_unix });
            }
        }
        infos.into_values().collect()
    }

    pub fn commit_cache(&mut self, request_id: &str, cache_id: String) -> Option<RocmTerminalCacheInfo> {
        let (tokens, session, cache_namespace, parent) = self.pending_sessions.remove(request_id)?;
        // 会话显存分解:同量级上下文的会话字节数差异可达数倍,定位固定开销
        // (recent/compressed 环与 DSpark block/target cache)与线性 KV 的占比。
        {
            let dspark_bytes = session.dspark.as_ref().map_or(0, RocmDeepSeekV4DsparkSession::allocated_bytes);
            let stage_bytes = session.stages.iter().map(DeepSeekV4StageState::cache_allocated_bytes).sum::<u64>();
            let batch_bytes = session.stages.iter().map(DeepSeekV4StageState::cache_batch_allocated_bytes).sum::<u64>();
            let kv_bytes = stage_bytes.saturating_sub(batch_bytes);
            let total = stage_bytes.saturating_add(dspark_bytes);
            let gib = |bytes: u64| bytes as f64 / (1_u64 << 30) as f64;
            eprintln!(
                "[deepseek-v4-cache] 会话显存分解 tokens={} 总={:.3}GiB KV={:.3}GiB batch_scratch={:.3}GiB dspark={:.3}GiB 线性口径={:.3}GiB",
                tokens.len(),
                gib(total),
                gib(kv_bytes),
                gib(batch_bytes),
                gib(dspark_bytes),
                gib(tokens.len() as u64 * 5560)
            );
        }
        let info = RocmTerminalCacheInfo { cache_id: cache_id.clone(), prompt_tokens: tokens.len(), bytes: session.allocated_bytes(), modified_unix: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() };
        if let Some(swap) = &self.swap
            && let Some((parent_id, round)) = &parent
        {
            let result = if parent_id == &cache_id || *round > self.options.terminal_cache_prefix_rounds { swap.delete(parent_id) } else { swap.set_head(parent_id, false) };
            if let Err(error) = result {
                eprintln!("[deepseek-v4-cache] 更新 SSD parent 失败 cache_id={parent_id}: {error}");
            }
        }
        let removed = self.terminal_sessions.insert(cache_id, tokens, parent, RocmTerminalState { session, cache_namespace, bytes: info.bytes, modified_unix: info.modified_unix });
        for removed in removed {
            if removed.persist
                && let Some(swap) = &self.swap
            {
                match removed.graph.session.download_cache().and_then(|(stages, dspark)| {
                    swap.put(&DeepSeekV4CacheSnapshot {
                        cache_id: removed.cache_id.clone(),
                        tokens: removed.tokens.clone(),
                        cache_namespace: removed.graph.cache_namespace.clone(),
                        round: removed.round,
                        head: removed.head,
                        stages,
                        dspark,
                        resident_bytes: removed.graph.bytes,
                        modified_unix: removed.graph.modified_unix,
                    })
                    .map(|_| ())
                }) {
                    Ok(()) => eprintln!("[deepseek-v4-cache] 已换出 SSD cache_id={} tokens={}", removed.cache_id, removed.tokens.len()),
                    Err(error) => eprintln!("[deepseek-v4-cache] 换出 SSD 失败 cache_id={}: {error}", removed.cache_id),
                }
            }
            if let Ok(mut graph) = Arc::try_unwrap(removed.graph) {
                graph.session.reset_session();
                self.reusable_sessions.push(graph.session);
            }
        }
        Some(info)
    }

    pub fn discard_pending(&mut self, request_id: &str) {
        if let Some((_, mut session, _, _)) = self.pending_sessions.remove(request_id) {
            session.reset_session();
            self.reusable_sessions.push(session);
        }
    }

    fn recycle_failed_session(&mut self, task: &BatchTask, mut session: RocmSessionState) {
        let _ = task;
        session.reset_session();
        self.reusable_sessions.push(session);
    }

    fn acquire_session(&mut self, task: &mut BatchTask) -> Result<RocmSessionState, String> {
        let mut resident = None;
        if let Some(cache_id) = task.cache_id.as_deref()
            && let Some(cached) = self.terminal_sessions.cached_tokens(cache_id).map(<[u32]>::to_vec)
        {
            if task.prompt_tokens.starts_with(&cached) && cached.len() < task.prompt_tokens.len() {
                if let Some(materialized) = self.terminal_sessions.materialize(cache_id, &task.prompt_tokens) {
                    let (cached, round, state) = materialized?;
                    if state.cache_namespace == task.cache_namespace {
                        resident = Some((cache_id.to_owned(), cached, round, state));
                    }
                }
            } else if let Some(suffix) = task.resume_suffix_tokens.take() {
                let mut resumed = cached.clone();
                resumed.extend(suffix);
                if cached.len() < resumed.len()
                    && let Some(materialized) = self.terminal_sessions.materialize(cache_id, &resumed)
                {
                    let (cached, round, state) = materialized?;
                    if state.cache_namespace == task.cache_namespace {
                        task.prompt_tokens = resumed;
                        resident = Some((cache_id.to_owned(), cached, round, state));
                    }
                }
            } else {
                // 请求携带的精确 cache_id 对应快照与前缀不符:链条断点最直接的信号。
                let common = cached.iter().zip(&task.prompt_tokens).take_while(|(a, b)| a == b).count();
                eprintln!("[deepseek-v4-cache] 精确 cache_id 前缀分叉 cache_id={cache_id} 公共前缀={common} 快照长度={} prompt 长度={}", cached.len(), task.prompt_tokens.len());
            }
        }
        // 瘦身请求的 prompt_tokens 只有增量,内容前缀复用会错误匹配其他
        // cache 的短前缀;命中只允许走上面的 cache_id 精确恢复路径。
        if resident.is_none()
            && !task.slim_resume
            && let Some(materialized) = self.terminal_sessions.materialize_longest_prefix(&task.prompt_tokens, |state| state.cache_namespace == task.cache_namespace)
        {
            let (cache_id, cached, round, state) = materialized?;
            if cached.len() < task.prompt_tokens.len() {
                resident = Some((cache_id, cached, round, state));
            }
        }
        if resident.is_none()
            && let Some(swap) = &self.swap
        {
            let mut snapshot = task.cache_id.as_deref().map(|cache_id| swap.get(cache_id)).transpose()?.flatten().filter(|snapshot| snapshot.cache_namespace == task.cache_namespace);
            if let Some(exact) = &snapshot
                && !task.prompt_tokens.starts_with(&exact.tokens)
                && let Some(suffix) = task.resume_suffix_tokens.take()
            {
                let mut resumed = exact.tokens.clone();
                resumed.extend(suffix);
                task.prompt_tokens = resumed;
            }
            snapshot = snapshot.filter(|snapshot| snapshot.tokens.len() < task.prompt_tokens.len() && task.prompt_tokens.starts_with(&snapshot.tokens));
            if snapshot.is_none()
                && !task.slim_resume
                && let Some(cache_id) = swap.longest_prefix_cache_id(&task.prompt_tokens)?
            {
                snapshot = swap.get(&cache_id)?.filter(|snapshot| snapshot.tokens.len() < task.prompt_tokens.len());
            }
            if let Some(snapshot) = snapshot {
                let cache_id = snapshot.cache_id.clone();
                let cached = snapshot.tokens.clone();
                let round = snapshot.round;
                let cache_namespace = snapshot.cache_namespace.clone();
                if self.stage_templates.len() != snapshot.stages.len() {
                    return Err(format!("DeepSeek-V4 SSD stage 数量={}，当前={}", snapshot.stages.len(), self.stage_templates.len()));
                }
                // SSD prefix 每次恢复都重新分配 session，会让已经换出的旧 session
                // 永久堆在 reusable pool。优先覆盖其中一份，使设备 buffer 真正复用。
                let mut session = self.reusable_sessions.pop().map(Ok).unwrap_or_else(|| open_stage_session(&self.stage_templates).map(|stages| RocmSessionState { stages, dspark: None })).map_err(backend_error)?;
                for (stage, cached_stage) in session.stages.iter_mut().zip(snapshot.stages) {
                    stage.upload_cache(cached_stage).map_err(backend_error)?;
                }
                session.dspark = match snapshot.dspark {
                    Some(snapshot) => {
                        Some(self.dspark.as_ref().ok_or("DeepSeek-V4 SSD 快照含 DSpark，但当前未启用".to_owned())?.lock().map_err(|_| "DSpark runtime mutex poisoned".to_owned())?.restore_session(snapshot).map_err(backend_error)?)
                    }
                    None => None,
                };
                eprintln!("[deepseek-v4-cache] 已从 SSD 恢复 cache_id={cache_id} cached_tokens={} prompt_tokens={} round={round}", cached.len(), task.prompt_tokens.len());
                resident = Some((cache_id, cached, round, RocmMaterializedSession { session, cache_namespace }));
            }
        }
        if let Some((cache_id, cached, round, state)) = resident {
            task.next_prefill = cached.len();
            task.cached_tokens = cached;
            task.resumed_cache_id = Some(cache_id);
            task.resumed_cache_round = Some(round);
            return Ok(state.session);
        }
        // 瘦身请求没有完整 messages,miss 不能退回残缺 prompt 的全量 prefill。
        if task.slim_resume {
            let cache_id = task.cache_id.as_deref().unwrap_or_default();
            return Err(format!("{} {cache_id}", crate::runtime::session::TERMINAL_RESUME_MISS));
        }
        let mut session = self.reusable_sessions.pop().map(Ok).unwrap_or_else(|| open_stage_session(&self.stage_templates).map(|stages| RocmSessionState { stages, dspark: None })).map_err(backend_error)?;
        session.reset_session();
        Ok(session)
    }

    fn prepare_batch_task(&self, request: RocmBatchRequest, prompt_tokens: Vec<u32>) -> Result<BatchTask, String> {
        if prompt_tokens.is_empty() || prompt_tokens.len() >= self.options.max_sequence_length {
            return Err(format!("DeepSeek-V4 prompt_tokens={} 超出 max_sequence_length={}", prompt_tokens.len(), self.options.max_sequence_length));
        }
        if request.cancellation.load(Ordering::Acquire) {
            return Err("请求已取消".to_owned());
        }
        let target_chunks = self.options.devices.len().saturating_mul(7).max(1);
        let aligned = prompt_tokens.len().div_ceil(target_chunks).div_ceil(256).saturating_mul(256).max(256);
        let prefill_chunk = self.options.prefill_chunk_size.min(aligned);
        let long_prefill = match (self.options.long_prefill_threshold_tokens, self.options.long_prefill_chunk_size) {
            (Some(threshold), Some(size)) => (threshold, size),
            (None, None) => (usize::MAX, prefill_chunk),
            _ => return Err("long prefill threshold/chunk 必须同时配置".to_owned()),
        };
        let long_chunk = long_prefill.1.min(prefill_chunk);
        // 整个请求已经属于长上下文时从首块就收紧，不能先用大块把显存顶满。
        let long_threshold = if prompt_tokens.len() >= long_prefill.0 {
            0
        } else if prefill_chunk >= long_chunk.saturating_mul(2) {
            long_prefill.0.min(prompt_tokens.len().div_ceil(2))
        } else {
            long_prefill.0
        };
        let max_tokens = request.requested_tokens.min(self.options.max_sequence_length - prompt_tokens.len()).max(1);
        let tool_fence = request.tool_fence.map(|spec| DsmlToolFence::new(&self.tokenizer, self.json_tokens.clone(), self.model.config().vocab_size, &self.model.config().eos_token_ids, spec)).transpose()?;
        let token_fence = GenerationGuard::new(tool_fence, request.repeat_loop_breaker, self.model.config().eos_token_ids.iter().copied());
        let resume_suffix_tokens = request.resume_suffix.as_deref().map(|suffix| self.tokenizer.tokenize(suffix.as_bytes()));
        // 多模态:占位符 token 展开为完整 span,并在入口设备上完成视觉编码。
        let (prompt_tokens, vision) = match request.vision_images.as_ref() {
            Some(images) if !images.is_empty() => {
                let input = self.expand_vision_input(images, &prompt_tokens)?;
                let overlays = self.build_vision_overlays(&input)?;
                if input.tokens.len() >= self.options.max_sequence_length {
                    return Err(format!("DeepSeek-V4.1 图文展开后 prompt_tokens={} 超出 max_sequence_length={}", input.tokens.len(), self.options.max_sequence_length));
                }
                (input.tokens, Some(Arc::new(overlays)))
            }
            _ => (prompt_tokens, None),
        };
        Ok(BatchTask {
            request_id: request.request_id,
            cancellation: request.cancellation,
            prompt_tokens,
            max_tokens,
            chunk_policy: crate::runtime::prefill::AdaptiveChunkPolicy { initial_chunk_size: prefill_chunk, append_chunk_size: prefill_chunk, long_context_threshold_tokens: long_threshold, long_context_chunk_size: long_chunk },
            next_prefill: 0,
            last_hidden: None,
            generated: Vec::new(),
            finish_reason: "length".to_owned(),
            stats: crate::runtime::speculative::SpeculativeStats::default(),
            cache_id: request.cache_id,
            resume_suffix_tokens,
            slim_resume: request.slim_resume,
            cache_namespace: request.cache_namespace,
            cached_tokens: Vec::new(),
            resumed_cache_id: None,
            resumed_cache_round: None,
            token_fence,
            vision,
        })
    }

    /// 把 prompt 中的 image 占位符展开为完整 span(预处理图像 → patch 张量)。
    fn expand_vision_input(&self, images: &[crate::vision::RgbImage], prompt_tokens: &[u32]) -> Result<DeepseekV41VisionInput, String> {
        let image_token_id = self.model.config().image_token_id.ok_or("当前 DeepSeek-V4 配置无 image token,不支持图像输入")?;
        let vision_config = crate::model_spec::deepseek_v4::DeepseekV41VisionConfig::standard();
        let processor = super::vision::DeepseekV41ImageProcessor::new(vision_config.clone())?;
        let tensors = images.iter().map(|image| crate::vision::ImageProcessor::preprocess(&processor, image)).collect::<Result<Vec<_>, _>>()?;
        let (tokens, spans) = super::vision::expand_image_spans(prompt_tokens, image_token_id, &tensors, vision_config.downsample_ratio)?;
        Ok(DeepseekV41VisionInput { tokens, images: tensors, spans })
    }

    /// 多模态请求的视觉编码:入口设备上跑视觉塔,把 aligner 行与首尾/换行
    /// 学习向量组装成 span 覆盖(chunked prefill 的 embedding 上传拼接用)。
    fn build_vision_overlays(&self, input: &DeepseekV41VisionInput) -> Result<DeepseekV41SpanOverlays, String> {
        let weights = self.source.official_weights().ok_or("DeepSeek-V4.1 多模态需要官方 safetensors checkpoint")?;
        if !weights.has_vision_tower() {
            return Err("DeepSeek-V4.1 checkpoint 缺少视觉塔,不能处理图像输入".to_owned());
        }
        let config = crate::model_spec::deepseek_v4::DeepseekV41VisionConfig::standard();
        let span_embeddings = weights.load_image_span_embeddings()?;
        let hidden = self.model.config().hidden_size;
        let context = self.entry_context;
        context.activate().map_err(|error| format!("入口设备: {error}"))?;
        let mut ranges = Vec::with_capacity(input.spans.len());
        let mut rows = Vec::with_capacity(input.spans.len());
        for (index, span) in input.spans.iter().enumerate() {
            let image = input.images.get(index).ok_or_else(|| format!("缺少图像 {index} 的 patch 张量"))?;
            let encoded = deepseek_v41_vision_encode(&context, &config, weights, image).map_err(|error| format!("图像 {index} 视觉编码: {error:?}"))?;
            let aligner_rows = context.tensor_to_f32(&encoded).map_err(|error| format!("图像 {index} 视觉输出下载: {error}"))?;
            rows.push(span_rows_f32(span, &aligner_rows, hidden, &span_embeddings)?);
            ranges.push(ImageTokenRange { image_index: index, tokens: span.start..span.start + span.token_count() });
        }
        Ok(DeepseekV41SpanOverlays { ranges, rows })
    }

    /// checkpoint 是否携带视觉塔(node 能力上报与图像请求接受判定)。
    pub fn vision_available(&self) -> bool {
        self.source.official_weights().is_some_and(|weights| weights.has_vision_tower())
    }

    /// V4.1 image span 共用 token id(node 输出侧拦截生成)。
    pub fn image_token_id(&self) -> Option<u32> {
        self.model.config().image_token_id
    }

    pub fn generate(&mut self, prompt: &str, requested_tokens: usize, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<RocmGeneration, String> {
        let request = RocmBatchRequest {
            request_id: "single".to_owned(),
            prompt: prompt.to_owned(),
            requested_tokens,
            cancellation: Arc::new(AtomicBool::new(cancellation.load(Ordering::Acquire))),
            cache_id: None,
            resume_suffix: None,
            slim_resume: false,
            cache_namespace: None,
            tool_fence: None,
            repeat_loop_breaker: true,
            vision_images: None,
        };
        let mut completed = Vec::new();
        let mut results = self.generate_batch(vec![request], &mut |_| Vec::new(), &mut |_, token, text| on_token(token, text), &mut |_, result| completed.push(result));
        results.append(&mut completed);
        let result = results.pop().expect("单路 batch 必须返回一个结果");
        self.discard_pending(&result.request_id);
        result.result
    }

    pub fn generate_batch(
        &mut self,
        requests: Vec<RocmBatchRequest>,
        intake: &mut dyn FnMut(usize) -> Vec<RocmBatchRequest>,
        on_token: &mut dyn FnMut(&str, u32, String) -> bool,
        on_result: &mut dyn FnMut(&mut Self, RocmBatchResult),
    ) -> Vec<RocmBatchResult> {
        let mut results = Vec::new();
        let mut tasks = Vec::new();
        for request in requests.into_iter().take(DEEPSEEK_V4_MAX_CONCURRENCY) {
            let request_id = request.request_id.clone();
            let prompt_tokens = self.tokenizer.tokenize(request.prompt.as_bytes());
            match self.prepare_batch_task(request, prompt_tokens) {
                Ok(task) => tasks.push(task),
                Err(message) => results.push(RocmBatchResult { request_id, result: Err(message) }),
            }
        }
        if tasks.is_empty() {
            return results;
        }
        let mut reported = vec![false; tasks.len()];

        let mut active_sessions = Vec::with_capacity(tasks.len());
        for task in &mut tasks {
            match self.acquire_session(task) {
                Ok(session) => active_sessions.push(session),
                Err(error) => {
                    for (index, session) in active_sessions.into_iter().enumerate() {
                        self.recycle_failed_session(&tasks[index], session);
                    }
                    results.extend(tasks.drain(..).map(|task| RocmBatchResult { request_id: task.request_id, result: Err(error.clone()) }));
                    return results;
                }
            }
        }
        let mut opened_dspark = 0usize;
        for (session, state) in active_sessions.iter_mut().enumerate() {
            if let Some(dspark) = self.dspark.as_ref() {
                if let Err(error) = dspark.lock().map_err(|_| "DSpark runtime mutex poisoned".to_owned()).and_then(|mut dspark| dspark.open_session_with(session, state.dspark.take()).map_err(backend_error)) {
                    if let Ok(mut dspark) = dspark.lock() {
                        for previous in 0..active_sessions.len() {
                            if previous < opened_dspark {
                                active_sessions[previous].dspark = dspark.take_session(previous).ok();
                            } else {
                                dspark.close_session(previous);
                            }
                        }
                    }
                    for (index, state) in active_sessions.into_iter().enumerate() {
                        self.recycle_failed_session(&tasks[index], state);
                    }
                    results.extend(tasks.drain(..).map(|task| RocmBatchResult { request_id: task.request_id, result: Err(error.clone()) }));
                    return results;
                }
                opened_dspark += 1;
            }
        }
        let sessions = active_sessions.into_iter().map(|state| state.stages).collect::<Vec<_>>();
        let capture_count = self.dspark.as_ref().map_or(0, |_| 3);
        let config = self.model.config().clone();
        // V4.1 的大批量 mHC/CSA scratch 在一个 8-stage 波次内已经覆盖全部设备；
        // 继续预灌三波只会延长临时 buffer 生命周期并把显存峰值推到 OOM。
        let pipeline_work_window = if self.model.is_v41() { self.options.devices.len() } else { self.options.devices.len().saturating_mul(3) };
        let driven = drive_deepseek_v4_stage_pipeline(sessions, DEEPSEEK_V4_MAX_CONCURRENCY, &config, pipeline_work_window, self.options.decode_batch_limit, self.options.profile, |scheduler| {
            let pipeline = DeepSeekV4StagePipeline::new(scheduler, capture_count);
            let entry_context = self.entry_context;
            let source = self.source.clone();
            let hidden_size = self.model.config().hidden_size;
            let pack_input = move |tokens: &[u32], start: usize, vision: Option<&Arc<DeepseekV41SpanOverlays>>| -> Result<RocmTensor, String> {
                entry_context.activate().map_err(|error| format!("入口设备: {error}"))?;
                let mut rows = source.embedding_rows(tokens)?;
                if let Some(overlay) = vision {
                    crate::vision::splice_image_embeddings_f32(&mut rows, hidden_size, start, &overlay.ranges, &overlay.rows).map_err(|error| format!("视觉 embedding 拼接: {error}"))?;
                }
                // 调度线程没有 stage compute stream；独立上传在专用 H2D stream
                // 完成后返回，并把输入 buffer 纳入显式 device pool。
                entry_context.tensor_from_f32_independent(rows, tokens.len(), hidden_size).map_err(|error| format!("输入上传: {error:?}"))
            };
            let run = (|| -> Result<(), String> {
                let prefill_started = Instant::now();
                let work_window = pipeline_work_window.max(1);
                let mut in_flight = 0usize;
                let mut cursor = 0usize;
                let mut work_id = 0usize;
                loop {
                    for request in intake(DEEPSEEK_V4_MAX_CONCURRENCY.saturating_sub(tasks.len())) {
                        let request_id = request.request_id.clone();
                        let prompt_tokens = self.tokenizer.tokenize(request.prompt.as_bytes());
                        match self.prepare_batch_task(request, prompt_tokens) {
                            Ok(task) => {
                                let session = tasks.len();
                                let mut task = task;
                                let mut state = self.acquire_session(&mut task)?;
                                scheduler.open(session, state.stages).map_err(backend_error)?;
                                if let Some(dspark) = self.dspark.as_ref() {
                                    dspark.lock().map_err(|_| "DSpark runtime mutex poisoned".to_owned())?.open_session_with(session, state.dspark.take()).map_err(backend_error)?;
                                }
                                reported.push(false);
                                tasks.push(task);
                            }
                            Err(message) => results.push(RocmBatchResult { request_id, result: Err(message) }),
                        }
                    }
                    for task in &mut tasks {
                        if task.finish_reason == "length" && task.cancellation.load(Ordering::Acquire) {
                            task.finish_reason = "cancelled".to_owned();
                            task.next_prefill = task.prompt_tokens.len();
                        }
                    }
                    let mut submitted = false;
                    while in_flight < work_window {
                        let Some(session) = (0..tasks.len()).map(|offset| (cursor + offset) % tasks.len()).find(|&session| tasks[session].next_prefill < tasks[session].prompt_tokens.len()) else { break };
                        let task = &mut tasks[session];
                        let start = task.next_prefill;
                        let end = start.saturating_add(task.chunk_policy.chunk_size(0, start)).min(task.prompt_tokens.len());
                        let tokens = task.prompt_tokens[start..end].to_vec();
                        let input = pack_input(&tokens, start, task.vision.as_ref())?;
                        let dspark_prefill_from = self.dspark_window.and_then(|window| {
                            let tail_start = task.prompt_tokens.len().saturating_sub(window);
                            (end > tail_start).then_some(tail_start.saturating_sub(start))
                        });
                        pipeline.push(session, work_id, start, tokens, input, dspark_prefill_from)?;
                        task.next_prefill = end;
                        work_id += 1;
                        in_flight += 1;
                        cursor = (session + 1) % tasks.len();
                        submitted = true;
                    }
                    if in_flight == 0 {
                        if !submitted {
                            break;
                        }
                        continue;
                    }
                    let output = pipeline.pull()?;
                    in_flight -= 1;
                    if let Some(from_row) = output.dspark_prefill_from {
                        let positions = (output.position..output.position + output.tokens.len()).collect::<Vec<_>>();
                        self.dspark
                            .as_ref()
                            .ok_or("DSpark capture 缺少 runtime")?
                            .lock()
                            .map_err(|_| "DSpark runtime mutex poisoned".to_owned())?
                            .prefill_target_suffix_session(output.session, &output.captures, &positions, from_row)
                            .map_err(backend_error)?;
                    }
                    tasks[output.session].last_hidden = Some(output.hidden);
                }
                eprintln!("[prefill-batch] sessions={} tokens={} wall={:.3}s", tasks.len(), tasks.iter().map(|task| task.prompt_tokens.len()).sum::<usize>(), prefill_started.elapsed().as_secs_f64());

                for (session, task) in tasks.iter_mut().enumerate() {
                    if task.finish_reason == "length" && task.cancellation.load(Ordering::Acquire) {
                        task.finish_reason = "cancelled".to_owned();
                    }
                    if task.finish_reason != "length" {
                        continue;
                    }
                    let first = fenced_task_token(&self.last_context, self.model.config(), &self.output_head, task, task.last_hidden.as_ref().ok_or_else(|| format!("session={session} prefill 缺少输出"))?).map_err(backend_error)?;
                    task.generated.push(first);
                    advance_task_fence(task, first);
                    if self.model.config().eos_token_ids.contains(&first) {
                        task.finish_reason = "stop".to_owned();
                    } else if !on_token(&task.request_id, first, token_text(&self.detokenizer, first)?) {
                        task.finish_reason = "cancelled".to_owned();
                    }
                }

                // 单路 target decode 在已验证的 8 卡主机上比 DSpark 更快；按本次 engine batch
                // 的初始会话数固定模式，避免同一会话因后续请求到达而中途切换语义。
                if let Some(dspark) = self.dspark.clone().filter(|_| tasks.len() >= self.options.dspark_min_sessions) {
                    struct DraftWork {
                        session: usize,
                        draft: crate::runtime::speculative::SpeculativeBlock,
                        inputs: Vec<u32>,
                        position: usize,
                        transactional: bool,
                    }
                    struct Verified {
                        session: usize,
                        captures: Vec<RocmTensor>,
                        positions: Vec<usize>,
                        retained_rows: usize,
                        transactional: bool,
                        emitted: Vec<u32>,
                        hard_loop: Option<LoopKind>,
                    }
                    struct DraftFlight {
                        work: DraftWork,
                        remaining_segments: usize,
                        next_row: usize,
                        target_tokens: Vec<u32>,
                        capture_parts: Vec<Vec<RocmTensor>>,
                    }
                    enum Flight {
                        Forward(DraftFlight),
                        Commit,
                    }

                    let (tokenized_sender, tokenized_receiver) = mpsc::channel::<(usize, RocmBatchRequest, Vec<u32>)>();
                    let stream_started = Instant::now();
                    let mut completed_rounds = 0usize;
                    let mut emitted_tokens = 0usize;
                    let mut flights = std::iter::repeat_with(|| None).take(tasks.len()).collect::<Vec<Option<Flight>>>();
                    let mut prefill_flights = vec![0_usize; tasks.len()];
                    let mut pending_commits = vec![None; tasks.len()];
                    let mut closing = vec![false; tasks.len()];
                    let mut tokenizing = vec![false; tasks.len()];
                    let mut appending_tokenizing = false;
                    let mut prefill_admission = OpportunisticPrefillAdmission::default();
                    let mut decode_admission_blocked = false;
                    let mut next_decode_admission_check = Instant::now();
                    let mut decode_scheduler = BatchScheduler::new(1, DEEPSEEK_V4_MAX_CONCURRENCY)?;
                    decode_scheduler.align_initial_batch(tasks.len());
                    loop {
                        let mut active = false;
                        let mut decode_phases = vec![RequestPhase::Finished; tasks.len()];
                        let decode_active = tasks
                            .iter()
                            .enumerate()
                            .any(|(session, task)| !reported[session] && prefill_flights[session] == 0 && task.next_prefill >= task.prompt_tokens.len() && task.finish_reason == "length" && task.generated.len() < task.max_tokens);
                        let (prefill_target, prefill_work_window, prefill_chunk_limit) = prefill_admission.limits(
                            scheduler.stage_flow_snapshot(),
                            scheduler.pipeline_work_window(),
                            scheduler.stage_count(),
                            decode_active,
                            self.options.decode_priority_prefill_chunk_size,
                            self.options.decode_priority_prefill_chunk_ceiling,
                        );
                        for (session, task) in tasks.iter_mut().enumerate() {
                            if reported[session] {
                                continue;
                            }
                            if closing[session] {
                                active = true;
                                continue;
                            }
                            if prefill_flights[session] != 0 {
                                decode_phases[session] = RequestPhase::Pending;
                                active = true;
                                continue;
                            }
                            if flights[session].is_some() {
                                decode_phases[session] = RequestPhase::Pending;
                                active = true;
                                continue;
                            }
                            if task.finish_reason == "length" && task.generated.len() < task.max_tokens && task.cancellation.load(Ordering::Acquire) {
                                task.finish_reason = "cancelled".to_owned();
                            }
                            if task.next_prefill < task.prompt_tokens.len() {
                                decode_phases[session] = RequestPhase::Pending;
                                active = true;
                                continue;
                            }
                            if task.finish_reason != "length" || task.generated.len() >= task.max_tokens {
                                if let Some(rows) = pending_commits[session].take() {
                                    pipeline.submit_commit_speculative(session, rows)?;
                                    flights[session] = Some(Flight::Commit);
                                    active = true;
                                } else {
                                    pipeline.close(session)?;
                                    closing[session] = true;
                                    active = true;
                                }
                                continue;
                            }
                            active = true;
                            let position = task.prompt_tokens.len() + task.generated.len() - 1;
                            decode_phases[session] = RequestPhase::DecodeReady { position };
                        }
                        let mut draft_batch = Vec::new();
                        for session in decode_scheduler.next(&decode_phases).map_or_else(Vec::new, |plan| plan.decode) {
                            let task = &tasks[session];
                            let anchor = *task.generated.last().expect("generated 非空");
                            let position = task.prompt_tokens.len() + task.generated.len() - 1;
                            draft_batch.push((session, anchor, position, pending_commits[session].take()));
                        }
                        let draft_requests = draft_batch.iter().map(|&(session, anchor, position, _)| (session, anchor, position)).collect::<Vec<_>>();
                        let mut draft_fences = draft_batch.iter().map(|&(session, _, _, _)| tasks[session].token_fence.clone()).collect::<Vec<_>>();
                        let mut draft_fence_refs = draft_fences.iter_mut().map(|fence| Some(fence as &mut dyn TokenFenceProgram)).collect::<Vec<_>>();
                        let drafts = dspark.lock().map_err(|_| "DSpark runtime mutex poisoned".to_owned())?.draft_batch(&draft_requests, &mut draft_fence_refs).map_err(backend_error)?;
                        if drafts.len() != draft_batch.len() {
                            return Err(format!("DSpark batch results={} 期望={}", drafts.len(), draft_batch.len()));
                        }
                        let mut forward_wave = Vec::with_capacity(draft_batch.len());
                        for ((session, anchor, position, commit_speculative), draft) in draft_batch.into_iter().zip(drafts) {
                            let mut inputs = Vec::with_capacity(draft.drafts.len() + 1);
                            inputs.push(anchor);
                            inputs.extend(draft.drafts.iter().copied());
                            let transactional = !draft.drafts.is_empty();
                            let work = DraftWork { session, draft, inputs, position, transactional };
                            forward_wave.push((work, commit_speculative));
                        }
                        let mut ready_forwards = Vec::new();
                        for (work, commit_speculative) in forward_wave {
                            let session = work.session;
                            // 官方 K=5 的完整 verify block 只有一份 work，不足以填满八级流水线；
                            // 按 token 切成独立 work，让 target verify 覆盖更多 pipeline stage。
                            let segment_count = if work.transactional { work.inputs.len() } else { 1 };
                            let segments = prefill_token_segments(0, work.inputs.len(), segment_count);
                            for (index, segment) in segments.iter().enumerate() {
                                let tokens = work.inputs[segment.clone()].to_vec();
                                // DSpark verify 段是生成 token,image overlay 区间不可能命中。
                                let hidden = pack_input(&tokens, work.position + segment.start, tasks[session].vision.as_ref())?;
                                ready_forwards.push(DeepSeekV4ReadyForward {
                                    session,
                                    chunk_index: work_id,
                                    position: work.position + segment.start,
                                    tokens,
                                    hidden,
                                    speculative: work.transactional,
                                    begin_speculative: work.transactional && index == 0,
                                    commit_speculative: (index == 0).then_some(commit_speculative).flatten(),
                                });
                                work_id += 1;
                            }
                            flights[session] = Some(Flight::Forward(DraftFlight { work, remaining_segments: segments.len(), next_row: 0, target_tokens: Vec::new(), capture_parts: Vec::new() }));
                        }
                        if !ready_forwards.is_empty() {
                            pipeline.push_dspark_ready(ready_forwards)?;
                        }
                        let mut prefill_work_in_flight = prefill_flights.iter().sum::<usize>();
                        let mut prefill_admissions_in_flight = if decode_active { prefill_flights.iter().filter(|&&count| count != 0).count() } else { prefill_work_in_flight };
                        while prefill_admissions_in_flight < prefill_target && prefill_work_in_flight < prefill_work_window {
                            let remaining_admissions = prefill_target - prefill_admissions_in_flight;
                            let eligible_sessions = tasks
                                .iter()
                                .enumerate()
                                .filter(|(session, task)| {
                                    !reported[*session]
                                        && !closing[*session]
                                        && flights[*session].is_none()
                                        && (!decode_active || prefill_flights[*session] == 0)
                                        && !task.cancellation.load(Ordering::Acquire)
                                        && task.next_prefill < task.prompt_tokens.len()
                                })
                                .count()
                                .min(remaining_admissions)
                                .max(1);
                            let Some(session) = scheduler.next_prefill_session(prefill_admission.cursor_mut(), tasks.len(), |session| {
                                !reported[session]
                                    && !closing[session]
                                    && flights[session].is_none()
                                    && (!decode_active || prefill_flights[session] == 0)
                                    && !tasks[session].cancellation.load(Ordering::Acquire)
                                    && tasks[session].next_prefill < tasks[session].prompt_tokens.len()
                            }) else {
                                break;
                            };
                            let task = &mut tasks[session];
                            let start = task.next_prefill;
                            let configured = task.chunk_policy.chunk_size(0, start);
                            let chunk_size = prefill_chunk_limit.map_or(configured, |limit| configured.min(limit));
                            let end = start.saturating_add(chunk_size).min(task.prompt_tokens.len());
                            let segment_budget = if decode_active { (prefill_work_window - prefill_work_in_flight).div_ceil(eligible_sessions) } else { 1 };
                            let segments = prefill_token_segments(start, end - start, segment_budget);
                            for segment in &segments {
                                let tokens = task.prompt_tokens[segment.clone()].to_vec();
                                let input = pack_input(&tokens, segment.start, task.vision.as_ref())?;
                                let dspark_prefill_from = self.dspark_window.and_then(|window| {
                                    let tail_start = task.prompt_tokens.len().saturating_sub(window);
                                    (segment.end > tail_start).then_some(tail_start.saturating_sub(segment.start))
                                });
                                pipeline.push(session, work_id, segment.start, tokens, input, dspark_prefill_from)?;
                                work_id += 1;
                            }
                            task.next_prefill = end;
                            prefill_flights[session] += segments.len();
                            prefill_work_in_flight += segments.len();
                            prefill_admissions_in_flight += 1;
                            active = true;
                            scheduler.commit_prefill_submission(prefill_admission.cursor_mut(), session, end == task.prompt_tokens.len());
                            prefill_admission.commit_work(end - start, segments.len());
                        }
                        // Close 与下一条请求到达之间可能存在竞态；空槽必须持续做非阻塞
                        // admission，不能只在 Closed 事件到达的瞬间尝试一次。decode 下
                        // 每次只引入一条 prefill，并限频查询显存，避免探测本身拖慢热路径。
                        let prefill_pending = tasks.iter().enumerate().any(|(session, task)| !reported[session] && task.next_prefill < task.prompt_tokens.len());
                        let has_admission_slot = !prefill_pending && !appending_tokenizing && !tokenizing.iter().any(|&pending| pending) && (reported.iter().any(|&done| done) || tasks.len() < DEEPSEEK_V4_MAX_CONCURRENCY);
                        let now = Instant::now();
                        let admission = (decode_active && has_admission_slot && now >= next_decode_admission_check).then(|| {
                            next_decode_admission_check = now + std::time::Duration::from_millis(250);
                            self.decode_admission_blocker()
                        });
                        let mut can_admit = has_admission_slot && (!decode_active || matches!(&admission, Some(Ok(None))));
                        if let Some(admission) = &admission
                            && !matches!(admission, Ok(None))
                            && !decode_admission_blocked
                        {
                            match admission {
                                Ok(Some((device, available))) => eprintln!(
                                    "[deepseek-v4-admission] 暂停 decode 期间接单 device={device} available={:.2}GiB reserve={:.2}GiB",
                                    *available as f64 / (1_u64 << 30) as f64,
                                    self.options.memory_reserve_bytes as f64 / (1_u64 << 30) as f64,
                                ),
                                Err((device, error)) => eprintln!("[deepseek-v4-admission] 暂停 decode 期间接单 device={device} 显存查询失败: {error}"),
                                Ok(None) => {}
                            }
                            decode_admission_blocked = true;
                        } else if matches!(&admission, Some(Ok(None))) && decode_admission_blocked {
                            eprintln!("[deepseek-v4-admission] 显存恢复，继续接单");
                            decode_admission_blocked = false;
                        }
                        let mut admitted = false;
                        for session in 0..tasks.len() {
                            if !can_admit {
                                break;
                            }
                            if !reported[session] || tokenizing[session] {
                                continue;
                            }
                            let Some(request) = intake(1).into_iter().next() else {
                                continue;
                            };
                            tokenizing[session] = true;
                            let tokenizer = self.tokenizer.clone();
                            let sender = tokenized_sender.clone();
                            rayon::spawn(move || {
                                let prompt_tokens = tokenizer.tokenize(request.prompt.as_bytes());
                                let _ = sender.send((session, request, prompt_tokens));
                            });
                            can_admit = false;
                            break;
                        }
                        if can_admit
                            && !appending_tokenizing
                            && tasks.len() < DEEPSEEK_V4_MAX_CONCURRENCY
                            && let Some(request) = intake(1).into_iter().next()
                        {
                            let session = tasks.len();
                            appending_tokenizing = true;
                            let tokenizer = self.tokenizer.clone();
                            let sender = tokenized_sender.clone();
                            rayon::spawn(move || {
                                let prompt_tokens = tokenizer.tokenize(request.prompt.as_bytes());
                                let _ = sender.send((session, request, prompt_tokens));
                            });
                        }
                        while let Ok((session, request, prompt_tokens)) = tokenized_receiver.try_recv() {
                            let appending = session == tasks.len();
                            if appending {
                                appending_tokenizing = false;
                            } else if let Some(tokenizing) = tokenizing.get_mut(session) {
                                *tokenizing = false;
                            } else {
                                return Err(format!("tokenize session={session} 越界，现有 slots={}", tasks.len()));
                            }
                            let request_id = request.request_id.clone();
                            let mut task = match self.prepare_batch_task(request, prompt_tokens) {
                                Ok(task) => task,
                                Err(message) => {
                                    on_result(self, RocmBatchResult { request_id, result: Err(message) });
                                    continue;
                                }
                            };
                            let mut state = match self.acquire_session(&mut task) {
                                Ok(state) => state,
                                Err(message) => {
                                    on_result(self, RocmBatchResult { request_id, result: Err(message) });
                                    continue;
                                }
                            };
                            scheduler.open(session, state.stages).map_err(backend_error)?;
                            dspark.lock().map_err(|_| "DSpark runtime mutex poisoned".to_owned())?.open_session_with(session, state.dspark.take()).map_err(backend_error)?;
                            if appending {
                                tasks.push(task);
                                reported.push(false);
                                flights.push(None);
                                prefill_flights.push(0);
                                pending_commits.push(None);
                                closing.push(false);
                                tokenizing.push(false);
                            } else {
                                tasks[session] = task;
                                reported[session] = false;
                            }
                            admitted = true;
                        }
                        if !active && !admitted {
                            if appending_tokenizing || tokenizing.iter().any(|&pending| pending) {
                                std::thread::yield_now();
                                continue;
                            }
                            break;
                        }

                        for event in pipeline.try_pull_ready_events()? {
                            let mut completed = None;
                            match event {
                                DeepSeekV4StageEvent::Forward(verified) => {
                                    let session = verified.session;
                                    if prefill_flights[session] != 0 {
                                        prefill_flights[session] -= 1;
                                        if let Some(from_row) = verified.dspark_prefill_from {
                                            let positions = (verified.position..verified.position + verified.tokens.len()).collect::<Vec<_>>();
                                            dspark.lock().map_err(|_| "DSpark runtime mutex poisoned".to_owned())?.prefill_target_suffix_session(session, &verified.captures, &positions, from_row).map_err(backend_error)?;
                                        }
                                        let task = &mut tasks[session];
                                        task.last_hidden = Some(verified.hidden);
                                        if verified.position + verified.tokens.len() == task.prompt_tokens.len() {
                                            let first = match task_fence(task).forced() {
                                                Some(token) => token,
                                                None => {
                                                    deepseek_v4_token_output(&self.last_context, self.model.config(), &self.output_head, task.last_hidden.as_ref().ok_or_else(|| format!("session={session} 补位 prefill 缺少输出"))?)
                                                        .map_err(backend_error)?
                                                        .token_id
                                                }
                                            };
                                            task.generated.push(first);
                                            advance_task_fence(task, first);
                                            if self.model.config().eos_token_ids.contains(&first) {
                                                task.finish_reason = "stop".to_owned();
                                            } else if !on_token(&task.request_id, first, token_text(&self.detokenizer, first)?) {
                                                task.finish_reason = "cancelled".to_owned();
                                            }
                                        }
                                        continue;
                                    }
                                    let Some(Flight::Forward(flight)) = flights.get_mut(session).and_then(Option::as_mut) else {
                                        return Err(format!("DSpark session={session} Forward/Commit 状态错位"));
                                    };
                                    let work = &flight.work;
                                    let row_start = verified.position.checked_sub(work.position).ok_or_else(|| format!("DSpark session={session} segment position={} 早于 round={}", verified.position, work.position))?;
                                    if row_start != flight.next_row || verified.tokens.is_empty() || row_start + verified.tokens.len() > work.inputs.len() {
                                        return Err(format!("DSpark session={session} segment rows={row_start}..{} 期望从 {} 开始，总行数={}", row_start + verified.tokens.len(), flight.next_row, work.inputs.len()));
                                    }
                                    let mut probe = tasks[session].token_fence.clone();
                                    for token in work.inputs.iter().skip(1).take(row_start) {
                                        probe.advance(*token);
                                    }
                                    let fences = (0..verified.tokens.len())
                                        .map(|row| {
                                            let fence = probe.fence();
                                            if let Some(token) = work.inputs.get(row_start + row + 1) {
                                                probe.advance(*token);
                                            }
                                            fence
                                        })
                                        .collect::<Vec<_>>();
                                    let target_tokens = if fences.iter().all(|fence| fence.forced().is_some()) {
                                        fences.iter().map(|fence| fence.forced().expect("刚确认每行 forced")).collect()
                                    } else {
                                        if self.options.profile {
                                            self.last_context.profile_scope_begin("target_head").map_err(backend_error)?;
                                        }
                                        let result = deepseek_v4_token_outputs(&self.last_context, self.model.config(), &self.output_head, &verified.hidden).and_then(|output| self.last_context.argmax_rows_fenced(&output.logits, &fences));
                                        if self.options.profile {
                                            self.last_context.profile_scope_end().map_err(backend_error)?;
                                        }
                                        result.map_err(backend_error)?
                                    };
                                    flight.target_tokens.extend(target_tokens);
                                    if flight.capture_parts.is_empty() {
                                        flight.capture_parts.resize_with(verified.captures.len(), Vec::new);
                                    }
                                    if flight.capture_parts.len() != verified.captures.len() {
                                        return Err(format!("DSpark session={session} segment captures={} 期望={}", verified.captures.len(), flight.capture_parts.len()));
                                    }
                                    for (parts, capture) in flight.capture_parts.iter_mut().zip(verified.captures) {
                                        parts.push(capture);
                                    }
                                    flight.next_row += verified.tokens.len();
                                    flight.remaining_segments = flight.remaining_segments.checked_sub(1).ok_or_else(|| format!("DSpark session={session} segment 计数下溢"))?;
                                    if flight.remaining_segments != 0 {
                                        continue;
                                    }
                                    let Some(Flight::Forward(flight)) = flights.get_mut(session).and_then(Option::take) else { unreachable!("刚确认 Forward flight") };
                                    let work = flight.work;
                                    let captures = flight
                                        .capture_parts
                                        .into_iter()
                                        .map(|mut parts| {
                                            if parts.len() == 1 {
                                                Ok(parts.pop().expect("capture part 非空"))
                                            } else {
                                                let refs = parts.iter().collect::<Vec<_>>();
                                                self.last_context.concat_token_rows(&refs)
                                            }
                                        })
                                        .collect::<Result<Vec<_>, BackendError>>()
                                        .map_err(backend_error)?;
                                    let verification = crate::runtime::speculative::verify_samples(&flight.target_tokens, &work.draft.drafts, &self.model.config().eos_token_ids).map_err(backend_error)?;
                                    tasks[session].stats.record(work.draft.drafts.len(), &verification);
                                    let task = &mut tasks[session];
                                    let room = task.max_tokens - task.generated.len();
                                    let mut emitted = verification.tokens.iter().take(room).copied().collect::<Vec<_>>();
                                    if emitted.is_empty() {
                                        return Err(format!("DSpark session={session} verifier 未产生 token"));
                                    }
                                    let loop_recovery = task.token_fence.recover(&mut emitted, None);
                                    let retained_rows = loop_recovery.map_or(verification.retained_rows, |recovery| verification.retained_rows.min(recovery.retained_rows));
                                    let hard_loop = loop_recovery.map(|recovery| recovery.kind);
                                    let positions = (work.position..work.position + work.inputs.len()).collect::<Vec<_>>();
                                    let verified = Verified { session: work.session, captures, positions, retained_rows, transactional: work.transactional, emitted, hard_loop };
                                    if verified.transactional {
                                        if pending_commits[session].replace(crate::backend::SpeculativeCacheCommit::retaining(verified.retained_rows)).is_some() {
                                            return Err(format!("DSpark session={session} 上一事务尚未提交"));
                                        }
                                    }
                                    completed = Some(verified);
                                }
                                DeepSeekV4StageEvent::CommittedSpeculative { session } => {
                                    let Some(Flight::Commit) = flights.get_mut(session).and_then(Option::take) else {
                                        return Err(format!("DSpark session={session} CommitSpeculative 状态错位"));
                                    };
                                }
                                DeepSeekV4StageEvent::Closed { session, states } => {
                                    if !closing.get(session).copied().unwrap_or(false) || reported.get(session).copied().unwrap_or(true) {
                                        return Err(format!("DSpark session={session} Closed 状态错位"));
                                    }
                                    let dspark_state = dspark.lock().map_err(|_| "DSpark runtime mutex poisoned".to_owned())?.take_session(session).map_err(backend_error)?;
                                    let task = &tasks[session];
                                    rocm_chain::release_request_workspaces(&self.options.devices).map_err(|error| format!("DeepSeek-V4 terminal workspace 回收失败: {error}"))?;
                                    eprintln!("[rocm-workspace] model=deepseek-v4 request_id={} finish_reason={} released=true", task.request_id, task.finish_reason);
                                    let mut cached = task.prompt_tokens.clone();
                                    cached.extend(task.generated.iter().take(task.generated.len().saturating_sub(1)).copied());
                                    let parent = task.resumed_cache_id.clone().zip(task.resumed_cache_round);
                                    self.pending_sessions.insert(task.request_id.clone(), (cached, RocmSessionState { stages: states, dspark: Some(dspark_state) }, task.cache_namespace.clone(), parent));
                                    eprintln!(
                                        "[dspark-stats] request_id={} rounds={} proposed={} accepted={} emitted={} target_forwards={}",
                                        task.request_id, task.stats.rounds, task.stats.proposed, task.stats.accepted, task.stats.emitted, task.stats.target_forwards
                                    );
                                    if self.options.profile {
                                        let mut hash = blake3::Hasher::new();
                                        for token in &task.generated {
                                            hash.update(&token.to_le_bytes());
                                        }
                                        let trace = if task.generated.len() <= 512 { format!("{:?}", task.generated) } else { format!("head={:?} tail={:?}", &task.generated[..16], &task.generated[task.generated.len() - 16..]) };
                                        eprintln!(
                                            "[request-lifecycle] phase=engine_terminal request_id={} finish_reason={} prompt_tokens={} completion_tokens={} eos={} token_hash={} tokens={}",
                                            task.request_id,
                                            task.finish_reason,
                                            task.prompt_tokens.len(),
                                            task.generated.len(),
                                            task.generated.last().is_some_and(|token| self.model.config().eos_token_ids.contains(token)),
                                            hash.finalize().to_hex(),
                                            trace,
                                        );
                                    }
                                    let result = RocmBatchResult {
                                        request_id: task.request_id.clone(),
                                        result: Ok(RocmGeneration { prompt_tokens: task.prompt_tokens.len(), completion_tokens: task.generated.len(), finish_reason: task.finish_reason.clone() }),
                                    };
                                    reported[session] = true;
                                    closing[session] = false;
                                    on_result(self, result);
                                }
                            }
                            if let Some(verified) = completed {
                                if verified.transactional && verified.retained_rows != 0 {
                                    dspark
                                        .lock()
                                        .map_err(|_| "DSpark runtime mutex poisoned".to_owned())?
                                        .prefill_target_prefix_session(verified.session, &verified.captures, &verified.positions, verified.retained_rows)
                                        .map_err(backend_error)?;
                                } else if !verified.transactional {
                                    dspark.lock().map_err(|_| "DSpark runtime mutex poisoned".to_owned())?.prefill_target_suffix_session(verified.session, &verified.captures, &verified.positions, 0).map_err(backend_error)?;
                                }
                                emitted_tokens += verified.emitted.len();
                                completed_rounds += 1;
                                let task = &mut tasks[verified.session];
                                for token in verified.emitted {
                                    // speculative block 只能逐 token 成为已完成输出；否则中途 EOS/
                                    // 下游取消会把尚未发送的尾部计入 usage，并污染 terminal cache。
                                    task.generated.push(token);
                                    advance_task_fence(task, token);
                                    if self.model.config().eos_token_ids.contains(&token) {
                                        task.finish_reason = "stop".to_owned();
                                        break;
                                    }
                                    if !on_token(&task.request_id, token, token_text(&self.detokenizer, token)?) {
                                        task.finish_reason = "cancelled".to_owned();
                                        break;
                                    }
                                }
                                if let Some(kind) = verified.hard_loop {
                                    task.finish_reason = "repetition".to_owned();
                                    eprintln!("[deepseek-v4-loop-guard] request_id={} kind={} action=stop", task.request_id, kind);
                                }
                                if completed_rounds % 64 == 0 {
                                    let active_sessions = tasks.iter().filter(|task| task.finish_reason == "length" && task.generated.len() < task.max_tokens).count();
                                    eprintln!("[dspark-stream] rounds={completed_rounds} emitted={emitted_tokens} active={active_sessions} wall={:.3}s", stream_started.elapsed().as_secs_f64());
                                    if self.options.profile {
                                        crate::kernel::rocm::hip::report_device_profiles(completed_rounds, active_sessions);
                                    }
                                }
                            }
                        }
                    }
                    eprintln!("[dspark-stream-summary] rounds={completed_rounds} emitted={emitted_tokens} wall={:.3}s", stream_started.elapsed().as_secs_f64());
                    if self.options.profile {
                        crate::kernel::rocm::hip::report_device_profiles(completed_rounds, 0);
                    }
                } else {
                    let mut completed_rounds = 0usize;
                    while tasks.iter().any(|task| task.finish_reason == "length" && task.generated.len() < task.max_tokens) {
                        let active = tasks
                            .iter_mut()
                            .enumerate()
                            .filter_map(|(session, task)| {
                                if task.finish_reason != "length" || task.generated.len() >= task.max_tokens {
                                    return None;
                                }
                                if task.cancellation.load(Ordering::Acquire) {
                                    task.finish_reason = "cancelled".to_owned();
                                    return None;
                                }
                                Some(session)
                            })
                            .collect::<Vec<_>>();
                        for &session in &active {
                            let token = *tasks[session].generated.last().expect("generated 非空");
                            let position = tasks[session].prompt_tokens.len() + tasks[session].generated.len() - 1;
                            let input = pack_input(&[token], position, None)?;
                            pipeline.push(session, work_id, position, vec![token], input, None)?;
                            work_id += 1;
                        }
                        for _ in 0..active.len() {
                            let output = pipeline.pull()?;
                            let task = &mut tasks[output.session];
                            if self.options.profile {
                                self.last_context.profile_scope_begin("target_head").map_err(backend_error)?;
                            }
                            let token = fenced_task_token(&self.last_context, self.model.config(), &self.output_head, task, &output.hidden);
                            if self.options.profile {
                                self.last_context.profile_scope_end().map_err(backend_error)?;
                            }
                            let token = token.map_err(backend_error)?;
                            task.generated.push(token);
                            advance_task_fence(task, token);
                            if self.model.config().eos_token_ids.contains(&token) {
                                task.finish_reason = "stop".to_owned();
                            } else if !on_token(&task.request_id, token, token_text(&self.detokenizer, token)?) {
                                task.finish_reason = "cancelled".to_owned();
                            }
                            completed_rounds += 1;
                            if self.options.profile && completed_rounds % 64 == 0 {
                                let active_sessions = tasks.iter().filter(|task| task.finish_reason == "length" && task.generated.len() < task.max_tokens).count();
                                crate::kernel::rocm::hip::report_device_profiles(completed_rounds, active_sessions);
                            }
                        }
                    }
                    if self.options.profile {
                        crate::kernel::rocm::hip::report_device_profiles(completed_rounds, 0);
                    }
                }
                Ok(())
            })();

            run.map_err(|message| BackendError::Compute { msg: message })
        });
        let (run, mut sessions) = match driven {
            Ok(driven) => driven,
            Err(error) => {
                if let Some(dspark) = &self.dspark
                    && let Ok(mut dspark) = dspark.lock()
                {
                    for session in 0..tasks.len() {
                        dspark.close_session(session);
                    }
                }
                let error = backend_error(error);
                results.extend(tasks.into_iter().map(|task| RocmBatchResult { request_id: task.request_id, result: Err(error.clone()) }));
                return results;
            }
        };
        let mut run = run.map_err(backend_error);
        if let Err(error) = rocm_chain::release_request_workspaces(&self.options.devices) {
            let release_error = format!("DeepSeek-V4 terminal workspace 回收失败: {error}");
            run = Err(match run {
                Ok(()) => release_error,
                Err(run_error) => format!("{run_error}; {release_error}"),
            });
        }
        let mut recovered = Vec::with_capacity(tasks.len());
        let mut recovery_error = None;
        for session in 0..tasks.len() {
            if reported[session] {
                if let Some(stages) = sessions.get_mut(session).and_then(Option::take) {
                    recovery_error.get_or_insert_with(|| format!("DeepSeek session={session} 已 Closed 但 state 再次归还"));
                    let mut state = RocmSessionState { stages, dspark: None };
                    state.reset_session();
                    self.reusable_sessions.push(state);
                }
                recovered.push(None);
                continue;
            }
            let stages = sessions.get_mut(session).and_then(Option::take).ok_or_else(|| BackendError::Compute { msg: format!("DeepSeek session={session} state 未归还") });
            let dspark = self.dspark.as_ref().map(|dspark| dspark.lock().map_err(|_| BackendError::Compute { msg: "DSpark runtime mutex poisoned".to_owned() })?.take_session(session)).transpose();
            match (stages, dspark) {
                (Ok(stages), Ok(dspark)) => recovered.push(Some(RocmSessionState { stages, dspark })),
                (Ok(stages), Err(error)) => {
                    recovery_error.get_or_insert_with(|| backend_error(error));
                    recovered.push(Some(RocmSessionState { stages, dspark: None }));
                }
                (Err(error), Ok(_)) => {
                    recovery_error.get_or_insert_with(|| backend_error(error));
                    recovered.push(None);
                }
                (Err(error), Err(_)) => {
                    recovery_error.get_or_insert_with(|| backend_error(error));
                    recovered.push(None);
                }
            }
        }
        if let Some(recovery_error) = recovery_error {
            for (index, session) in recovered.into_iter().enumerate() {
                if !reported[index]
                    && let Some(session) = session
                {
                    self.recycle_failed_session(&tasks[index], session);
                }
            }
            let error = match &run {
                Ok(()) => recovery_error,
                Err(run_error) => format!("{run_error}; session state 回收失败: {recovery_error}"),
            };
            results.extend(tasks.into_iter().enumerate().filter(|(index, _)| !reported[*index]).map(|(_, task)| RocmBatchResult { request_id: task.request_id, result: Err(error.clone()) }));
            return results;
        }
        if run.is_ok() {
            for (index, (task, session)) in tasks.iter().zip(recovered.iter_mut()).enumerate() {
                if reported[index] {
                    continue;
                }
                let session = session.take().expect("未上报 session 必须归还 state");
                let mut cached = task.prompt_tokens.clone();
                cached.extend(task.generated.iter().take(task.generated.len().saturating_sub(1)).copied());
                let parent = task.resumed_cache_id.clone().zip(task.resumed_cache_round);
                self.pending_sessions.insert(task.request_id.clone(), (cached, session, task.cache_namespace.clone(), parent));
            }
        } else {
            for (index, (task, session)) in tasks.iter().zip(recovered.into_iter()).enumerate() {
                if !reported[index]
                    && let Some(session) = session
                {
                    self.recycle_failed_session(task, session);
                }
            }
        }
        match run {
            Ok(()) => results.extend(tasks.into_iter().enumerate().filter(|(index, _)| !reported[*index]).map(|(_, task)| {
                eprintln!(
                    "[dspark-stats] request_id={} rounds={} proposed={} accepted={} emitted={} target_forwards={}",
                    task.request_id, task.stats.rounds, task.stats.proposed, task.stats.accepted, task.stats.emitted, task.stats.target_forwards
                );
                if self.options.profile {
                    let mut hash = blake3::Hasher::new();
                    for token in &task.generated {
                        hash.update(&token.to_le_bytes());
                    }
                    let trace = if task.generated.len() <= 512 { format!("{:?}", task.generated) } else { format!("head={:?} tail={:?}", &task.generated[..16], &task.generated[task.generated.len() - 16..]) };
                    eprintln!(
                        "[request-lifecycle] phase=engine_terminal request_id={} finish_reason={} prompt_tokens={} completion_tokens={} eos={} token_hash={} tokens={}",
                        task.request_id,
                        task.finish_reason,
                        task.prompt_tokens.len(),
                        task.generated.len(),
                        task.generated.last().is_some_and(|token| self.model.config().eos_token_ids.contains(token)),
                        hash.finalize().to_hex(),
                        trace,
                    );
                }
                RocmBatchResult { request_id: task.request_id, result: Ok(RocmGeneration { prompt_tokens: task.prompt_tokens.len(), completion_tokens: task.generated.len(), finish_reason: task.finish_reason }) }
            })),
            Err(error) => results.extend(tasks.into_iter().enumerate().filter(|(index, _)| !reported[*index]).map(|(_, task)| RocmBatchResult { request_id: task.request_id, result: Err(error.clone()) })),
        }
        results
    }

    pub fn shutdown(&mut self) {
        let entries = self.terminal_sessions.drain_entries();
        if let Some(swap) = &self.swap {
            for entry in &entries {
                match entry.graph.session.download_cache().and_then(|(stages, dspark)| {
                    swap.put(&DeepSeekV4CacheSnapshot {
                        cache_id: entry.cache_id.clone(),
                        tokens: entry.tokens.clone(),
                        cache_namespace: entry.graph.cache_namespace.clone(),
                        round: entry.round,
                        head: entry.head,
                        stages,
                        dspark,
                        resident_bytes: entry.graph.bytes,
                        modified_unix: entry.graph.modified_unix,
                    })
                    .map(|_| ())
                }) {
                    Ok(()) => eprintln!("[deepseek-v4-cache] 优雅退出已持久化 cache_id={} tokens={}", entry.cache_id, entry.tokens.len()),
                    Err(error) => eprintln!("[deepseek-v4-cache] 优雅退出持久化失败 cache_id={}: {error}", entry.cache_id),
                }
            }
        } else {
            eprintln!("[deepseek-v4-cache] 持久化已关闭，退出时跳过 SSD 快照");
        }
        for session in self.reusable_sessions.drain(..) {
            std::mem::forget(session);
        }
        for entry in entries {
            std::mem::forget(entry.graph);
        }
        for (_, (_, state, _, _)) in self.pending_sessions.drain() {
            std::mem::forget(state);
        }
    }
}

fn token_text(detokenizer: &Detokenizer, token: u32) -> Result<String, String> {
    crate::runtime::tool::decode_output_token(&detokenizer, token).map(|bytes| String::from_utf8_lossy(&bytes).into_owned()).map_err(|error| format!("detokenize {token}: {error}"))
}

fn task_fence(task: &BatchTask) -> TokenFence {
    task.token_fence.fence()
}

fn fenced_task_token(context: &RocmContext, config: &DeepSeekV4Config, head: &DeepSeekV4OutputHead<RocmWeight>, task: &BatchTask, hidden: &RocmTensor) -> Result<u32, BackendError> {
    let output = deepseek_v4_token_output(context, config, head, hidden)?;
    let fence = task_fence(task);
    if fence.is_open() { Ok(output.token_id) } else { Ok(context.argmax_rows_fenced(&output.logits, &[fence])?[0]) }
}

fn advance_task_fence(task: &mut BatchTask, token: u32) {
    task.token_fence.advance(token);
}

fn backend_error(error: BackendError) -> String {
    format!("{error:?}")
}

fn build_experts(source: &Source, context: &RocmContext, layer_start: usize, layer_end: usize, config: &crate::model_spec::deepseek_v4::DeepSeekV4Config) -> Result<RocmPrefillExperts, String> {
    let mut experts = match source {
        Source::Gguf(gguf) => {
            let expert_source: Arc<dyn GgufExpertSource> = Arc::new(gguf.clone());
            RocmPrefillExperts::gguf(expert_source)
        }
        Source::Official(_) => RocmPrefillExperts::mxfp4(source.expert_source()?),
    };
    match source {
        Source::Gguf(_) => {
            for layer in layer_start..layer_end {
                experts.preload_layer(context, layer, config.expert_count).map_err(|error| format!("预载 L{layer} 专家: {error:?}"))?;
            }
        }
        Source::Official(_) => {
            // MXFP4 的 grouped/preshuffle 布局同时覆盖 decode 与 prefill；不要再预载一套逐专家 resident 权重。
            let spec = crate::moe::topk_moe::TopkMoeSpec {
                num_experts: config.expert_count,
                top_k: config.expert_top_k,
                num_shared_experts: config.shared_expert_count,
                scoring_func: crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias,
                normalize_selected: true,
                routed_scaling_factor: config.routed_scaling_factor,
                intermediate_size: config.expert_intermediate_size,
                shared_intermediate_size: config.expert_intermediate_size,
                activation: crate::moe::Activation::SiluClamped { limit: config.swiglu_limit },
            };
            for layer in layer_start..layer_end {
                experts.mxfp4_grouped(context, layer, &spec).map_err(|error| format!("装配 L{layer} grouped: {error:?}"))?;
            }
        }
    }
    Ok(experts)
}

fn prepare_layer(source: &Source, context: &RocmContext, layer: usize, layer_cache: &mut DeepSeekV4LayerCache<RocmWeight>) -> Result<(), BackendError> {
    let prepared = match source {
        Source::Gguf(gguf) => prepare_deepseek_v4_gguf_layer(context, gguf, layer)?,
        Source::Official(official) => prepare_deepseek_v4_layer(context, official, layer)?,
    };
    let _ = layer_cache.put(layer, prepared);
    Ok(())
}
