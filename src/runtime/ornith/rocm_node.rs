//! Ornith × ROCm 多卡节点引擎：单机设备链分层执行。
//!
//! 结构对齐 GLM-5.2 rocm_node 的分层模式：每卡一个 RocmContext，持有自己的层区间、
//! 常驻 experts 与各会话 KV/DeltaNet 状态；prefill chunk 与 decode step 通过
//! `runtime::prefill::run_single_stage_chain` 沿设备链推进，跨卡张量走
//! `move_tensor_to_stage_ordered`。本文件只保留 Ornith 特有的状态类型与请求编排；
//! 设备链、KV 容量与能力上报复用 `runtime::rocm_chain`，请求层协议复用
//! `runtime::ornith::protocol`。
//!
//! decode 与 GLM 同策略：单 token 作为 1 行 chunk 走 prefill 路径，复用常驻 experts。

#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::attention::gated_delta_net::GatedDeltaNetState;
use crate::attention::rope::RopeTable;
use crate::backend::Backend;
use crate::backend::StageTensorBackend;
use crate::backend::rocm::{RocmContext, RocmGatedDeltaNetStorage, RocmKvCache, RocmPrefillExperts, RocmTensor, RocmWeight};
use crate::config::{OrnithNodeModelConfig, RocmBackendConfig};
use crate::kv_cache::terminal_cache::TerminalCache;
use crate::kv_cache::terminal_cache::TerminalInfo as CacheInfo;
use crate::runtime::ornith::{self, OrnithConfig, OrnithGguf, OrnithLayer, OrnithOutputHead, OrnithRuntime};
use crate::runtime::rocm_chain::{DynError, RocmDeviceChain, node_capabilities};
use crate::runtime::session::{AtomicCounterU64, BatchTokenGuard, GenerationOutput, GenerationSummary, NodeCapabilities, RuntimeStatus as NodeRuntime, TerminalResume, parse_stops, request_terminal_resume, terminal_cache_id};
use crate::runtime::tool::{RequestToolCallStream, ToolDialect, emit_request_tool_chunk, finish_request_tool_stream};
use crate::server::node::{NodeConfig, NodeEngine, run_node};
use crate::tokenizer::{Detokenizer, Tokenizer};

use super::options::OrnithOptions;
use super::protocol::{chat_prompt, chat_prompt_suffix};

/// prefill 分块上限：控制 CPU reference GQA 每次回传 host 的 K/V 规模与中间激活显存。
const PREFILL_CHUNK_TOKENS: usize = 2048;

pub async fn run(model: OrnithNodeModelConfig, backend: RocmBackendConfig, config: NodeConfig) -> Result<(), DynError> {
    let weights_path: PathBuf = model.weights_directory.clone();
    let max_seq_len = model.max_sequence_length;
    let lm_head_quantization = model.lm_head_quantization;
    let layer_ends = model.layer_ends.clone();
    let devices = backend.devices.clone();
    let allow_cpu_reference_fallback = backend.allow_cpu_reference_fallback;
    let options = OrnithOptions::from(model.execution);
    let factory = Box::new(move |runtime, compute_steps| {
        OrnithRocmEngine::load(&weights_path, devices, allow_cpu_reference_fallback, max_seq_len, layer_ends, options, lm_head_quantization, runtime, compute_steps).map(|engine| Box::new(engine) as Box<dyn NodeEngine>)
    });
    run_node(config, factory).await
}

/// 一个设备上的会话状态。cache 与 recurrent 都按全模型层 id 索引，
/// 未触达的层保持惰性空槽，因此直接用 layer_count 建槽不浪费设备内存。
struct OrnithRocmStage {
    backend: RocmContext,
    first_layer: usize,
    layers: Arc<[OrnithLayer<RocmWeight>]>,
    experts: Arc<Mutex<RocmPrefillExperts>>,
    cache: RocmKvCache,
    recurrent: GatedDeltaNetState<RocmGatedDeltaNetStorage>,
}

impl OrnithRocmStage {
    fn allocated_bytes(&self) -> u64 {
        self.cache.allocated_bytes().saturating_add(self.recurrent.allocated_bytes() as u64)
    }
}

struct OrnithRocmSequence {
    stages: Vec<OrnithRocmStage>,
    hidden: RocmTensor,
    tokens: Vec<u32>,
}

impl OrnithRocmSequence {
    fn token_count(&self) -> usize {
        self.tokens.len()
    }

    fn allocated_bytes(&self) -> u64 {
        self.stages.iter().map(OrnithRocmStage::allocated_bytes).sum()
    }
}

struct OrnithTerminalState {
    sequence: OrnithRocmSequence,
    pending_tokens: Vec<u32>,
    info: CacheInfo,
}

pub struct OrnithRocmEngine {
    cfg: OrnithConfig,
    chain: RocmDeviceChain,
    /// 每设备的层区间起点（与 chain.layer_ends 对应）。
    layer_starts: Vec<usize>,
    layer_stash: Vec<Arc<[OrnithLayer<RocmWeight>]>>,
    experts: Vec<Arc<Mutex<RocmPrefillExperts>>>,
    output_head: OrnithOutputHead<RocmWeight>,
    weights: Arc<OrnithGguf>,
    rope: RopeTable,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    max_seq_len: usize,
    options: OrnithOptions,
    capabilities: NodeCapabilities,
    terminal_states: TerminalCache<OrnithTerminalState>,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
    resident_expert_bytes: u64,
}

/// 未显式配置 layer_ends 时按设备数均分（余数给靠前的卡）。
fn balanced_layer_ends(layer_count: usize, devices: usize) -> Vec<usize> {
    let base = layer_count / devices;
    let remainder = layer_count % devices;
    (0..devices)
        .scan(0usize, |end, index| {
            *end += base + usize::from(index < remainder);
            Some(*end - 1)
        })
        .collect()
}

impl OrnithRocmEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights_path: &std::path::Path,
        devices: Vec<i32>,
        allow_cpu_reference_fallback: bool,
        max_seq_len: usize,
        layer_ends: Option<Vec<usize>>,
        options: OrnithOptions,
        lm_head_quantization: crate::weight::LmHeadQuantization,
        runtime: Arc<Mutex<NodeRuntime>>,
        compute_steps: Arc<AtomicCounterU64>,
    ) -> Result<Self, DynError> {
        let weights = Arc::new(OrnithGguf::open(weights_path)?);
        let cfg = weights.config().clone();
        ornith::ensure_supported(&cfg).map_err(|error| -> DynError { format!("Ornith runtime 不支持: {error:?}").into() })?;
        if devices.is_empty() {
            return Err("Ornith ROCm 至少需要一个 device".into());
        }
        let layer_ends = layer_ends.unwrap_or_else(|| balanced_layer_ends(cfg.layer_count, devices.len()));
        if layer_ends.len() > 1 && layer_ends.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(format!("Ornith layer_ends 必须严格递增: {layer_ends:?}").into());
        }
        let chain = RocmDeviceChain::new(&devices, layer_ends, cfg.layer_count - 1, allow_cpu_reference_fallback).map_err(|error| -> DynError { format!("Ornith ROCm 设备链: {error}").into() })?;
        let tokenizer = weights.tokenizer()?;
        let detokenizer = weights.detokenizer()?;
        let rope = RopeTable::precompute(max_seq_len, cfg.rope_dim, cfg.rope_theta);

        let mut layer_starts = Vec::with_capacity(chain.contexts.len());
        let mut layer_stash = Vec::with_capacity(chain.contexts.len());
        let mut experts = Vec::with_capacity(chain.contexts.len());
        let mut resident_expert_bytes = 0u64;
        for (device, context) in chain.contexts.iter().enumerate() {
            let (start, end) = chain.layer_range(device)?;
            context.activate().map_err(|error| -> DynError { format!("激活 ROCm device {}: {error}", context.device_id()).into() })?;
            let started = Instant::now();
            let layers: Arc<[OrnithLayer<RocmWeight>]> =
                (start..=end).map(|layer| ornith::prepare_ornith_layer(context, weights.as_ref(), layer).map_err(|error| -> DynError { format!("准备 Ornith ROCm L{layer}: {error:?}").into() })).collect::<Result<Vec<_>, _>>()?.into();
            let mut device_experts = RocmPrefillExperts::gguf(weights.clone());
            if !options.lazy_experts {
                for layer in start..=end {
                    device_experts.preload_layer(context, layer, cfg.num_experts).map_err(|error| -> DynError { format!("常驻 Ornith ROCm L{layer} experts: {error:?}").into() })?;
                }
            }
            // 常驻 expert 字节按 GGUF 路由专家总量折算（RocmPrefillExperts 不暴露内部账目）。
            let per_layer_expert_bytes = {
                let inventory = weights.inventory();
                inventory.routed_experts.bytes / (cfg.layer_count + usize::from(weights.has_mtp())).max(1) as u64
            };
            let bytes = if options.lazy_experts { 0 } else { per_layer_expert_bytes * (end - start + 1) as u64 };
            resident_expert_bytes += bytes;
            eprintln!("[ornith-rocm] device={} layers=[{},{}] expert_bytes={:.2}GiB wall={:.3}s", context.device_id(), start, end, bytes as f64 / (1_u64 << 30) as f64, started.elapsed().as_secs_f64());
            layer_starts.push(start);
            layer_stash.push(layers);
            experts.push(Arc::new(Mutex::new(device_experts)));
        }
        let tail = chain.contexts.last().ok_or("Ornith ROCm 设备链为空")?;
        let output_head = ornith::prepare_ornith_output_head_quantized(tail, weights.as_ref(), lm_head_quantization).map_err(|error| -> DynError { format!("准备 Ornith ROCm 输出头: {error:?}").into() })?;
        // RocmKvCache 的 GQA K/V 当前是 F32 device buffer；在 backend 真正实现
        // F16/Q8 存储前不得按请求配置伪报 cache format。
        let capabilities = node_capabilities(&chain.contexts, max_seq_len, "f32", "ornith-gguf-rocm".to_owned(), weights.reader().file_len());
        Ok(Self {
            cfg,
            chain,
            layer_starts,
            layer_stash,
            experts,
            output_head,
            weights,
            rope,
            tokenizer,
            detokenizer,
            max_seq_len,
            options,
            capabilities,
            terminal_states: TerminalCache::new(options.terminal_cache_entries),
            runtime,
            compute_steps,
            resident_expert_bytes,
        })
    }

    fn new_stages(&self) -> Result<Vec<OrnithRocmStage>, String> {
        self.chain
            .contexts
            .iter()
            .enumerate()
            .map(|(device, context)| {
                let recurrent = GatedDeltaNetState::new(self.cfg.layer_count, self.cfg.gated_delta_net_spec()).map_err(|error| format!("创建 Ornith ROCm DeltaNet state: {error:?}"))?;
                Ok(OrnithRocmStage {
                    backend: context.clone(),
                    first_layer: self.layer_starts[device],
                    layers: self.layer_stash[device].clone(),
                    experts: self.experts[device].clone(),
                    cache: RocmKvCache::with_capacity(self.cfg.layer_count, self.max_seq_len),
                    recurrent,
                })
            })
            .collect()
    }

    /// 把一段 token 从 position 起穿过整条设备链；返回尾设备上的末行 hidden。
    /// decode 复用同一路径：tokens 长度为 1 即单步 decode。
    fn forward_tokens(&self, stages: Vec<OrnithRocmStage>, tokens: &[u32], position: usize) -> Result<(RocmTensor, Vec<OrnithRocmStage>), String> {
        if tokens.is_empty() {
            return Err("Ornith forward token 数为 0".to_owned());
        }
        let backends: Vec<RocmContext> = stages.iter().map(|stage| stage.backend.clone()).collect();
        let mut stages = stages;
        let mut hidden = None;
        let mut final_chunk_len = 0usize;
        for (chunk_index, chunk) in tokens.chunks(PREFILL_CHUNK_TOKENS).enumerate() {
            let chunk_position = position + chunk_index * PREFILL_CHUNK_TOKENS;
            final_chunk_len = chunk.len();
            let embedding = self.weights.embedding_rows(chunk).map_err(|error| format!("Ornith embedding: {error}"))?;
            let input = backends[0].tensor_from_f32(embedding, chunk.len(), self.cfg.hidden_size).map_err(|error| format!("上传 Ornith embedding: {error:?}"))?;
            let taken = std::mem::take(&mut stages);
            let runtime_options = self.options.runtime;
            let (_position, output, returned) = crate::runtime::prefill::run_single_stage_chain(&backends, taken, chunk_position, input, |backend, slots, _stage, mut batch| {
                let (_, position, hidden) = batch.pop().ok_or_else(|| crate::runtime::compute_error("Ornith stage batch 为空"))?;
                let mut state = slots[0].take().ok_or_else(|| crate::runtime::compute_error("Ornith stage state 缺失"))?;
                backend.activate_stage().map_err(|error| crate::runtime::compute_error(format!("激活 Ornith stage: {error:?}")))?;
                let hidden = backend.move_tensor_to_stage_ordered(hidden).map_err(|error| crate::runtime::compute_error(format!("Ornith 跨卡搬运: {error:?}")))?;
                // runtime 对 state.layers 的借用限制在本块内，结束后才能归还 state 槽位。
                let output = {
                    let runtime = OrnithRuntime::new(backend, &self.cfg, &state.layers, state.first_layer, &self.rope, runtime_options);
                    let mut experts = state.experts.lock().map_err(|_| crate::runtime::compute_error("Ornith expert 锁中毒"))?;
                    runtime.at(&mut state.cache, &mut state.recurrent, position).prefill(&mut experts, hidden).map_err(|error| crate::runtime::compute_error(format!("Ornith ROCm 分层执行: {error:?}")))
                }?;
                slots[0] = Some(state);
                Ok(vec![(0, position, output)])
            })
            .map_err(|error| format!("Ornith ROCm stage 链: {error:?}"))?;
            stages = returned;
            hidden = Some(output);
        }
        let hidden = hidden.ok_or("Ornith forward 没有产出")?;
        let tail = backends.last().expect("设备链非空");
        let last_row = tail.select_row(&hidden, final_chunk_len - 1).map_err(|error| format!("选择 Ornith 末行: {error:?}"))?;
        Ok((last_row, stages))
    }

    fn next_token(&self, hidden: &RocmTensor) -> Result<u32, String> {
        let tail = self.chain.contexts.last().expect("设备链非空");
        ornith::ornith_token_output(tail, &self.cfg, &self.output_head, hidden).map(|output| output.token_id).map_err(|error| format!("Ornith output: {error:?}"))
    }
}

impl NodeEngine for OrnithRocmEngine {
    fn model_key(&self) -> &'static str {
        "ornith"
    }
    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        let resident = self.resident_expert_bytes;
        (self.capabilities.clone(), Arc::new(move || resident))
    }
    fn refresh_runtime(&self) {
        if let Ok(mut runtime) = self.runtime.lock() {
            runtime.resident_expert_bytes = self.resident_expert_bytes;
            crate::runtime::session::refresh_cache_runtime(&mut runtime, self.terminal_states.states().map(|state| &state.info), self.max_seq_len);
        }
    }
    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        self.terminal_states.states().map(|state| state.info.clone()).collect()
    }
    fn max_concurrency(&self) -> usize {
        1
    }

    fn generate_one(&mut self, request_id: &str, request: &serde_json::Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        self.generate(request_id, request, cancellation, on_token)
    }
}

impl OrnithRocmEngine {
    /// 与 Metal 引擎同构的请求循环；区别只在序列操作走多卡设备链。
    fn generate(&mut self, request_id: &str, request: &serde_json::Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        let prompt = chat_prompt(request)?;
        let tokens = self.tokenizer.tokenize(prompt.as_bytes());
        let prompt_tokens = tokens.len();
        let requested_tokens = crate::runtime::session::requested_completion_tokens(request);
        if requested_tokens == 0 {
            return Err("max_tokens 必须是大于 0 的整数".to_owned());
        }
        if tokens.is_empty() || tokens.len() >= self.max_seq_len {
            return Err(format!("Ornith prompt {} tokens 超出 max_seq_len {}", tokens.len(), self.max_seq_len));
        }
        let stops = parse_stops(request.get("stop"))?;
        let resumed = match request_terminal_resume(request)? {
            TerminalResume::Match { cache_id, assistant } => self.terminal_states.take(&cache_id).map(|state| (assistant, state)),
            TerminalResume::Mismatch { requested, expected } => {
                eprintln!("[zllm-node] terminal cache hash 不匹配 id={requested} expected={expected}");
                None
            }
            TerminalResume::None => None,
        };
        let batch_tokens = resumed.as_ref().map_or(tokens.len(), |(_, (cached, _))| tokens.len().saturating_sub(cached.len()));
        let _batch_guard = BatchTokenGuard::new(&self.runtime, batch_tokens);
        let mut sequence = if let Some((assistant, (cached_tokens, state))) = resumed {
            let OrnithTerminalState { mut sequence, pending_tokens, .. } = state;
            let mut suffix = pending_tokens;
            suffix.extend(self.tokenizer.tokenize(chat_prompt_suffix(request, assistant)?.as_bytes()));
            let stages = std::mem::take(&mut sequence.stages);
            let (hidden, stages) = self.forward_tokens(stages, &suffix, sequence.tokens.len())?;
            sequence.stages = stages;
            sequence.hidden = hidden;
            sequence.tokens.extend_from_slice(&suffix);
            eprintln!("[zllm-node] terminal cache 命中 cached_tokens={} new_tokens={}", cached_tokens.len(), suffix.len());
            sequence
        } else {
            let stages = self.new_stages()?;
            let (hidden, stages) = self.forward_tokens(stages, &tokens, 0)?;
            OrnithRocmSequence { stages, hidden, tokens }
        };
        if sequence.token_count() >= self.max_seq_len {
            return Err(format!("Ornith 会话状态 {} tokens 超过 max_seq_len {}", sequence.token_count(), self.max_seq_len));
        }
        let max_tokens = requested_tokens.min(self.max_seq_len - sequence.token_count());
        if max_tokens == 0 {
            return Err("max_tokens 必须大于 0".to_owned());
        }
        let mut response_text = String::new();
        let mut tool_stream = RequestToolCallStream::new(request, request_id, ToolDialect::ChatmlJson);
        let mut pending_token = None;
        let result: Result<GenerationSummary, String> = (|| {
            let mut output = GenerationOutput::new(&stops);
            for step in 0..max_tokens {
                if cancellation.load(Ordering::Acquire) {
                    output.cancel();
                    break;
                }
                let token = self.next_token(&sequence.hidden)?;
                if self.cfg.eos_token_ids.contains(&token) {
                    output.stop();
                    break;
                }
                pending_token = Some(token);
                let bytes = crate::runtime::tool::decode_output_token(&self.detokenizer, token).map_err(|error| format!("Ornith detokenize {token}: {error}"))?;
                if !output.push(&bytes, |chunk| emit_request_tool_chunk(&mut tool_stream, token, &chunk, &mut response_text, on_token)) {
                    if output.finish_reason() == "stop" {
                        pending_token = None;
                    }
                    break;
                }
                if step + 1 == max_tokens {
                    break;
                }
                let position = sequence.tokens.len();
                let stages = std::mem::take(&mut sequence.stages);
                let (hidden, stages) = self.forward_tokens(stages, std::slice::from_ref(&token), position)?;
                sequence.stages = stages;
                sequence.hidden = hidden;
                sequence.tokens.push(token);
                self.compute_steps.fetch_add(1);
                pending_token = None;
            }
            output.finish(|chunk| emit_request_tool_chunk(&mut tool_stream, 0, &chunk, &mut response_text, on_token));
            if !output.is_cancelled() && !finish_request_tool_stream(&mut tool_stream, 0, &mut response_text, on_token) {
                output.cancel();
            }
            if !output.is_cancelled() && !tool_stream.calls.is_empty() {
                output.mark_tool_calls();
            }
            Ok(output.summary(prompt_tokens))
        })();
        let mut summary = result?;
        if summary.finish_reason != "cancelled" && stops.is_empty() {
            let cache_id = terminal_cache_id(request, &response_text, &tool_stream.calls)?;
            let info = CacheInfo {
                cache_id,
                model_key: "ornith".to_owned(),
                cache_format: "ornith-rocm-terminal-v1".to_owned(),
                last_layer: self.cfg.layer_count.saturating_sub(1),
                prompt_tokens: sequence.token_count(),
                bytes: sequence.allocated_bytes(),
                modified_unix: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            };
            let terminal_tokens = sequence.tokens.clone();
            let state = OrnithTerminalState { sequence, pending_tokens: pending_token.into_iter().collect(), info: info.clone() };
            if self.terminal_states.insert(info.cache_id.clone(), terminal_tokens, state) {
                summary.cache = Some(info);
            }
        }
        summary.tool_calls = std::mem::take(&mut tool_stream.calls);
        Ok(summary)
    }
}
