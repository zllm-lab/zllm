//! GLM-5.2 × ROCm 节点引擎：链头 PP stage，连接 scheduler 与下游 StageTransport。
//!
//! 设计（见 plan `abundant-popping-fairy`）：
//! - **链头** = `NodeEngine`，注册 scheduler，对 axum 可见；下游 stage 是独立
//!   `zllm-rt-rocm` 进程，走 StageTransport，对 axum 透明。
//! - **跨请求 KV 复用**：`TerminalCache<Glm52HeadState>`，线性会话只留最长。
//! - **PP 编排**：A 跑 L0..L38，B 跑 L39..L77；B 只回传最终 hidden。
//! - **输出环**：默认由 A0 执行 final norm、LM head、sampling 与 MTP L78；
//!   输出/采样/MTP 由本机首卡 A0 执行,decode 回传 hidden。
//!
//! ⚠️ PP 编排时序（send_prefill/recv token/cancel 解锁）需 16 卡 + GLM-5.2 真权重
//! 端到端验证；本文件编译通过 + 结构对齐 rocm front stage（`zllm-rt-rocm` 已跑通）。

#![cfg(target_os = "linux")]

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use super::protocol::*;
use super::rocm_swap::{Glm52CacheIdentity, Glm52CacheSnapshot, Glm52DsparkAuxCache, Glm52DsparkTargetCache, Glm52DsparkTargetLayerCache, Glm52DsparkTargetTensor, Glm52SwapStore, download_glm52_session, upload_glm52_session};
use crate::attention::rope::RopeTable;
use crate::attention::{AttentionSpec, mla::MlaSpec};
use crate::backend::{BackendResources, LinearWeight, SegmentedTensorBackend, StageExecutionBackend};
use crate::backend::{
    cpu::CpuContext,
    rocm::{RocmContext, RocmPrefillExperts, RocmTensor, RocmWeight},
};
use crate::config::{Glm52DsparkExecutionBackend, Glm52NodeExecutionConfig, KvCacheFormat};
use crate::kernel::cpu::CpuTensor;
use crate::kv_cache::terminal_cache::TerminalInfo as CacheInfo;
use crate::kv_cache::terminal_cache::{ResidencyBudget, ResidencyReservation, TerminalCache};
use crate::runtime::glm52::stage::{self as glm52_stage, Glm52StageState as RuntimeGlm52StageState, Glm52StageValue, build_glm52_stage_states, build_pp_prefill_chunk, drive_glm52_stream_stage_pipeline_stateful, prepare_prefill_layers};
use crate::runtime::glm52::tool::Glm52ToolFence;
use crate::runtime::multiplex::{BatchScheduler, RequestPhase};
use crate::runtime::prefill::{AdaptiveChunkPolicy, OpportunisticPrefillAdmission, StageSchedulerHandle, StageSchedulerOutput, prefill_token_segments};
use crate::runtime::session::{AtomicCounterU64, BatchTokenGuard, GenerationSummary, KvCacheDeviceCapacity, NodeCapabilities, RuntimeStatus as NodeRuntime, TerminalResume, ToolCallDelta, request_terminal_resume, terminal_cache_id};
use crate::runtime::{
    Model,
    generation_guard::{GenerationGuard, LoopKind, TokenFenceProgram, build_token_signatures},
    glm52::{Glm52, Glm52Config, Glm52OutputHead, glm52_sampled_token_ids_fenced},
    output::{SamplingConfig, SamplingState},
    rocm_chain,
    speculative::{verify_samples, verify_samples_prefix},
};
use crate::server::iroh::IrohConfig;
use crate::server::node::{NodeBatchRequest, NodeBatchResult, NodeEngine};
use crate::server::stage_transport::{RequestId, StageDeviceMemory, StageMessage, StageTransport};
use crate::tokenizer::{Detokenizer, Tokenizer, Utf8StreamDecoder};
use crate::weight::Glm52Weights;

use super::dspark_cpu::{CpuDsparkExecutor, CpuDsparkJob, CpuDsparkRuntime};
use super::dspark_rocm::{RocmDsparkDraftBatch, RocmDsparkRuntime, attach_dspark_projections};
use super::rocm::{RocmMtpCatchUp, RocmMtpDraftBatch, RocmMtpRuntime, RocmMtpSession, gather_embedding_rows, load_resident_embedding, mtp_catch_up_batch, mtp_draft_batch, prepare_glm52_rope_resident, prepare_head_output_runtime};
use crate::runtime::dspark::{DsparkTargetCache, DsparkTargetCacheSnapshot, DsparkTargetLayerSnapshot};

type DynError = Box<dyn std::error::Error + Send + Sync>;
type Glm52StageState = RuntimeGlm52StageState<RocmContext>;

#[path = "terminal.rs"]
mod terminal;
use terminal::Glm52HeadState;
#[path = "rocm_single.rs"]
mod rocm_single;
use rocm_single::Glm52SingleEngine;

pub async fn run(model: crate::config::Glm52NodeModelConfig, backend: crate::config::RocmBackendConfig, config: crate::server::node::NodeConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = model.weights_directory;
    let compressed_tensors_directory = model.compressed_tensors_directory;
    let nvfp4_directory = model.nvfp4_directory;
    let gguf_directory = model.gguf_directory;
    let tokenizer_path = model.tokenizer.expect("配置加载已补全 tokenizer");
    let max_seq_len = model.max_sequence_length;
    let lm_head_quantization = model.lm_head_quantization;
    let execution = model.execution;
    let stage_end = model.head.stage_end;
    let layer_ends = model.head.layer_ends;
    let downstream_ticket = model.head.downstream.ticket;
    let downstream_iroh = (stage_end < crate::runtime::glm52::Glm52Config::standard().layer_count).then(|| model.head.downstream.iroh.runtime()).transpose()?;
    let devices = backend.devices;
    let cache_dir = config.cache_dir.clone();
    let persist_kv_cache = config.persist_kv_cache;
    let factory = Box::new(move |runtime, compute_steps| {
        let weights = std::sync::Arc::new(
            crate::runtime::glm52::open_weights(&crate::runtime::glm52::Glm52Config::standard(), &model_path, compressed_tensors_directory.as_deref(), nvfp4_directory.as_deref(), gguf_directory.as_deref())
                .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> { error.into() })?,
        );
        if stage_end == crate::runtime::glm52::Glm52Config::standard().layer_count {
            Glm52SingleEngine::load(weights, devices.clone(), layer_ends.clone(), &tokenizer_path, max_seq_len, execution, lm_head_quantization, runtime, compute_steps)
                .map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
        } else {
            Glm52Engine::load(
                weights,
                &cache_dir,
                persist_kv_cache,
                devices.clone(),
                layer_ends.clone(),
                stage_end,
                &downstream_ticket,
                downstream_iroh.clone().expect("分段模式已解析 downstream iroh"),
                &tokenizer_path,
                max_seq_len,
                execution,
                lm_head_quantization,
                runtime,
                compute_steps,
            )
            .map(|engine| Box::new(engine) as Box<dyn crate::server::node::NodeEngine>)
        }
    });
    crate::server::node::run_node(config, factory).await
}

#[path = "batch.rs"]
mod batch;
use batch::*;
pub struct Glm52Engine {
    cfg: Glm52Config,
    mla: MlaSpec,
    /// 每个物理双卡组的 owner；完整 layer stage、Attention/KV/DSA 都只放这里。
    contexts: Vec<RocmContext>,
    /// cooperative MoE 的另一半计算卡，与 `contexts` 一一对应，不拥有独立 stage。
    cooperative_peer_contexts: Vec<RocmContext>,
    layer_ends: Vec<usize>,
    stage_end: usize,
    weights: Arc<Glm52Weights>,
    rope: Arc<RopeTable>,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    token_signatures: Arc<[u64]>,
    thinking_end_token: Option<u32>,
    max_seq_len: usize,
    pipeline_chunk_size: usize,
    link: StageTransport,
    downstream_memory: Vec<StageDeviceMemory>,
    downstream_session_capacity: Option<usize>,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
    output_head: Option<Glm52OutputHead<RocmWeight>>,
    /// 首卡常驻 BF16 embedding 表;decode/verify/draft/prefill 的行 gather 全部设备侧完成。
    embedding_table: Option<std::sync::Arc<crate::kernel::rocm::hip::DeviceBuffer>>,
    mtp_runtime: Option<RocmMtpRuntime>,
    dspark_runtime: Option<RocmDsparkRuntime>,
    cpu_dspark_executor: Option<CpuDsparkExecutor>,
    next_cpu_dspark_job: u64,
    cpu_dspark_submit_profile: [u128; 3],
    accept_phase_profile: [u128; 7],
    resident_states: Option<Vec<Glm52StageState>>,
    terminal_states: TerminalCache<Glm52HeadState>,
    swap: Option<Arc<Glm52SwapStore>>,
    persist_kv_cache: bool,
    pending_messages: HashMap<RequestId, VecDeque<StageMessage>>,
    intake_capacity: usize,
    kv_budget: ResidencyBudget,
    kv_reservation_page_tokens: usize,
    options: Glm52NodeExecutionConfig,
}

impl Glm52Engine {
    fn has_finishable_short_prefill(slots: &[Option<Glm52BatchTask>], submitted: &[usize], in_flight_by_session: &[usize], ceiling: usize) -> bool {
        slots.iter().enumerate().any(|(session, task)| {
            task.as_ref().is_some_and(|task| {
                let remaining = task.tokens.len().saturating_sub(submitted[session]);
                in_flight_by_session[session] == 0 && remaining != 0 && remaining <= ceiling && remaining <= task.prefill_policy.chunk_size(task.prefill_suffix_start, submitted[session])
            })
        })
    }

    /// 构造下一份 decode/verify 工作。只做容量预留与 embedding 上传，不动任务的
    /// 逻辑位置；`cached_tokens` / `pending_verify_rows` 由调用方在 submit 成功后
    /// 用返回的 inputs 提交，保证失败路径上任务状态与实际执行保持一致。
    fn prepare_next_work(&mut self, session: usize, task: &mut Glm52BatchTask, next: Glm52NextWork, weights: &Glm52Weights) -> Result<Option<(Vec<(usize, usize, Glm52StageValue<RocmContext>)>, Vec<u32>)>, crate::backend::BackendError> {
        let backend_error = |msg: String| crate::backend::BackendError::Compute { msg };
        let (inputs, verify) = match next {
            Glm52NextWork::Decode(token) => (vec![token], false),
            Glm52NextWork::Verify(tokens) => (tokens, true),
            Glm52NextWork::Finish => return Ok(None),
        };
        let rows = inputs.len();
        if rows != 1 && (!task.dspark_verify_hidden.is_empty() || !task.dspark_verify_aux.is_empty()) {
            return Err(backend_error("DSpark 上一轮 split verify 尚未回收完成".to_owned()));
        }
        if rows != 1 {
            task.dspark_verify_hidden.reserve(rows);
            task.dspark_verify_aux.reserve(rows);
        }
        let required_tokens = task.cached_tokens.len().saturating_add(rows);
        self.grow_kv_reservation(&mut task.kv_reservation, required_tokens).map_err(backend_error)?;
        let position = task.cached_tokens.len();
        let embedding = match self.embedding_table.as_ref() {
            Some(table) => gather_embedding_rows(&self.contexts[0], table, &inputs, self.cfg.hidden_size).map_err(backend_error)?,
            None => self.contexts[0]
                .tensor_from_f32(weights.embedding_rows(&inputs).map_err(|error| backend_error(format!("embedding: {error:?}")))?, rows, self.cfg.hidden_size)
                .map_err(|error| backend_error(format!("上传 continuous decode embedding: {error:?}")))?,
        };
        let works = if verify {
            // verify 行按 verify_group_rows 分组提交:组间保持逐组流水重叠
            // (row0 组进入下一 stage 时本 stage 立即推进下一组),组内共享一次
            // GEMV/attention 的权重读取。1 退化为逐行流水;整批(与行数相等)
            // 已实测因失去重叠而回退。不能把全部行组成一个 work,ready 侧按
            // 多行 hidden 返回,scheduler 拒绝同 session 重复。
            let group = self.options.dspark_verify_group_rows.clamp(1, rows);
            if group >= rows {
                vec![(session, position, Glm52StageValue { decode: false, verify: true, truncate_to: Some(position), hidden: embedding, selection: None, aux_hidden: None, aux_taps: 0 })]
            } else {
                (0..rows)
                    .step_by(group)
                    .map(|start| {
                        let count = group.min(rows - start);
                        let group_position = position + start;
                        let hidden = self.contexts[0].slice_token_rows(&embedding, start, count)?;
                        Ok((session, group_position, Glm52StageValue { decode: false, verify: true, truncate_to: Some(group_position), hidden, selection: None, aux_hidden: None, aux_taps: 0 }))
                    })
                    .collect::<Result<Vec<_>, crate::backend::BackendError>>()?
            }
        } else {
            vec![(session, position, Glm52StageValue { decode: true, verify: false, truncate_to: Some(position), hidden: embedding, selection: None, aux_hidden: None, aux_taps: 0 })]
        };
        Ok(Some((works, inputs)))
    }

    /// 提交成功后才推进任务逻辑位置：verify 行数与 token 缓存必须与真正进入流水线的工作对齐。
    fn commit_submitted_work(task: &mut Glm52BatchTask, inputs: &[u32]) {
        if inputs.len() != 1 || task.dspark_cpu_anchor_in_flight {
            task.pending_verify_rows = inputs.len();
        }
        if task.dspark_cpu_flight.is_some() {
            task.dspark_verify_inputs = inputs.to_vec();
        }
        if task.pending_token.is_some_and(|pending| inputs.first() == Some(&pending)) {
            task.pending_token = None;
        }
        task.cached_tokens.extend_from_slice(inputs);
    }

    /// CPU proposal 返回时，anchor verifier 已经在途；这里只构造尚未提交的
    /// draft suffix。每行继续独立穿过 16-stage，不建立多行 stage barrier。
    fn prepare_cpu_verify_suffix(&mut self, session: usize, task: &mut Glm52BatchTask, drafts: &[u32], weights: &Glm52Weights) -> Result<Vec<(usize, usize, Glm52StageValue<RocmContext>)>, crate::backend::BackendError> {
        if drafts.is_empty() {
            return Ok(Vec::new());
        }
        let backend_error = |msg: String| crate::backend::BackendError::Compute { msg };
        if !task.dspark_cpu_anchor_in_flight || task.pending_verify_rows != 1 || !task.dspark_verify_hidden.is_empty() || !task.dspark_verify_aux.is_empty() {
            return Err(backend_error(
                format!("CPU DSpark suffix 状态非法: anchor={} pending={} hidden={} aux={}", task.dspark_cpu_anchor_in_flight, task.pending_verify_rows, task.dspark_verify_hidden.len(), task.dspark_verify_aux.len(),),
            ));
        }
        let required_tokens = task.cached_tokens.len().saturating_add(drafts.len());
        self.grow_kv_reservation(&mut task.kv_reservation, required_tokens).map_err(backend_error)?;
        let position = task.cached_tokens.len();
        let embedding = self.contexts[0]
            .tensor_from_f32(weights.embedding_rows(drafts).map_err(|error| backend_error(format!("embedding: {error:?}")))?, drafts.len(), self.cfg.hidden_size)
            .map_err(|error| backend_error(format!("上传 CPU DSpark verify suffix embedding: {error:?}")))?;
        let group = self.options.dspark_verify_group_rows.clamp(1, drafts.len());
        if group >= drafts.len() {
            return Ok(vec![(session, position, Glm52StageValue { decode: false, verify: true, truncate_to: Some(position), hidden: embedding, selection: None, aux_hidden: None, aux_taps: 0 })]);
        }
        (0..drafts.len())
            .step_by(group)
            .map(|start| {
                let rows = group.min(drafts.len() - start);
                let row_position = position + start;
                let hidden = self.contexts[0].slice_token_rows(&embedding, start, rows)?;
                Ok((session, row_position, Glm52StageValue { decode: false, verify: true, truncate_to: Some(row_position), hidden, selection: None, aux_hidden: None, aux_taps: 0 }))
            })
            .collect()
    }

    fn append_dspark_aux_history(context: &RocmContext, task: &mut Glm52BatchTask, position: usize, hidden: RocmTensor, window: Option<usize>) -> Result<(), String> {
        if hidden.rows == 0 {
            return Err("DSpark aux history 禁止追加空张量".to_owned());
        }
        let mut start = task.dspark_aux_history_start;
        let mut history = match task.dspark_aux_history.take() {
            Some(history) => {
                let end = start.checked_add(history.rows).ok_or("DSpark aux history 位置溢出")?;
                if end != position {
                    return Err(format!("DSpark aux history 不连续: range=[{start},{end}) append={position}"));
                }
                context.concat_token_rows(&[&history, &hidden]).map_err(|error| format!("拼接 DSpark aux history: {error:?}"))?
            }
            None => {
                start = position;
                hidden
            }
        };
        if let Some(window) = window
            && history.rows > window
        {
            let dropped = history.rows - window;
            history = context.slice_token_rows(&history, dropped, window).map_err(|error| format!("裁剪 DSpark aux history: {error:?}"))?;
            start = start.checked_add(dropped).ok_or("DSpark aux history start 溢出")?;
        }
        task.dspark_aux_history_start = start;
        task.dspark_aux_history = Some(history);
        Ok(())
    }

    /// terminal state 必须同时拥有与 aux history 完全同区间的 target K/V。
    /// cache 落后时只投影缺失后缀；cache 含 verify 尾部时先截到接受边界。
    fn align_dspark_target_cache(&self, task: &mut Glm52BatchTask) -> Result<(), String> {
        let Some(runtime) = &self.dspark_runtime else {
            if task.dspark_aux_history.is_some() || task.dspark_target_cache.warmed_layers() != 0 {
                return Err("未启用 DSpark 但任务持有 DSpark cache".to_owned());
            }
            return Ok(());
        };
        let context = self.output_context();
        let history = task.dspark_aux_history.as_ref().ok_or("提交 DSpark cache 时缺少 aux history")?;
        let start = task.dspark_aux_history_start;
        let end = start.checked_add(history.rows).ok_or("提交 DSpark cache 时 aux range 溢出")?;

        if let Some(range) = task.dspark_target_cache.covered_range(&context)
            && range.start == start
            && range.end > end
        {
            task.dspark_target_cache.truncate_end(&context, end).map_err(|error| format!("截断 DSpark target cache: {error:?}"))?;
        }
        match task.dspark_target_cache.covered_range(&context) {
            Some(range) if range == (start..end) => {}
            Some(range) if runtime.target_history_window().is_none() && range.start == start && range.end < end => {
                let suffix = context.slice_token_rows(history, range.end - start, end - range.end).map_err(|error| format!("切分 DSpark target cache 后缀: {error:?}"))?;
                runtime.warm_target_cache(&context, &mut task.dspark_target_cache, &suffix, range.end).map_err(|error| format!("补齐 DSpark target cache: {error:?}"))?;
            }
            _ if runtime.target_history_window().is_some() => {
                // sliding drafter 每轮都会右移窗口；保留旧 cache 与新 history 的
                // 重叠区，只投影新接受后缀，不能 reset 后重算整个窗口。
                runtime.warm_target_cache(&context, &mut task.dspark_target_cache, history, start).map_err(|error| format!("滚动 DSpark target cache: {error:?}"))?;
            }
            _ => {
                task.dspark_target_cache.reset_session();
                runtime.warm_target_cache(&context, &mut task.dspark_target_cache, history, start).map_err(|error| format!("重建 DSpark target cache: {error:?}"))?;
            }
        }

        let range = task.dspark_target_cache.covered_range(&context).ok_or("提交 DSpark target cache 不完整")?;
        let shapes = task.dspark_target_cache.try_snapshot(|tensor| Ok::<_, String>((tensor.rows, tensor.cols)))?.ok_or("提交 DSpark target cache 尚未预热全部层")?;
        let expected_columns = runtime.target_columns();
        if range != (start..end) || shapes.layers.len() != runtime.target_layer_count() || shapes.layers.iter().any(|layer| layer.key != (history.rows, expected_columns) || layer.value != (history.rows, expected_columns)) {
            return Err(format!("提交 DSpark target cache 元数据不匹配: range={range:?}/[{start},{end}) layers={}/{} columns={expected_columns}", shapes.layers.len(), runtime.target_layer_count()));
        }
        Ok(())
    }

    /// CPU proposal 复用 GPU 已维护的 normalized aux history；首次从权威 GPU
    /// target cache 精确复制，后续只把 cache 尚未覆盖的新 suffix 交给 worker。
    fn take_cpu_dspark_state(&self, task: &mut Glm52BatchTask) -> Result<(CpuTensor, DsparkTargetCache<CpuTensor>, usize, usize), String> {
        if self.cpu_dspark_executor.is_none() {
            return Err("CPU DSpark executor 未加载".to_owned());
        }
        if task.dspark_cpu_pending.is_some() {
            return Err("CPU DSpark 上一轮 job 尚未返回".to_owned());
        }
        let runtime = self.dspark_runtime.as_ref().ok_or("CPU DSpark 缺少 GPU runtime")?;
        let gpu_history = task.dspark_aux_history.as_ref().ok_or("CPU DSpark 缺少 aux history")?;
        let start = task.dspark_aux_history_start;
        let end = start.checked_add(gpu_history.rows).ok_or("CPU DSpark aux range 溢出")?;
        let backend = CpuContext;
        let mut cache = std::mem::replace(&mut task.dspark_cpu_target_cache, DsparkTargetCache::new());
        let mut range = cache.covered_range(&backend);
        // sliding attention 只读取末尾 window，CPU cache 可暂留更早的行。配合
        // ownership-consuming append，绝大多数轮只写 suffix；达到 2×window
        // 才压回当前窗口，避免每轮搬移约 192MiB K/V。
        let compact = range.as_ref().is_some_and(|range| {
            let suffix_rows = end.saturating_sub(range.end.min(end));
            runtime.target_history_window().is_some_and(|window| range.start < start && range.end.saturating_sub(range.start).saturating_add(suffix_rows) > window.saturating_mul(2))
        });
        if compact {
            cache.truncate_start(&backend, start).map_err(|error| format!("裁剪 CPU DSpark target cache 头部: {error:?}"))?;
            range = cache.covered_range(&backend);
        }
        if range != Some(start..end) {
            let can_roll = range.as_ref().is_some_and(|range| start >= range.start && start < range.end && range.end <= end);
            if !can_roll {
                let snapshot = task
                    .dspark_target_cache
                    .try_snapshot(|tensor| {
                        let data = self.output_context().tensor_to_f32(tensor).map_err(|error| format!("下载 CPU DSpark target cache: {error:?}"))?;
                        Ok::<_, String>(CpuTensor { data, rows: tensor.rows, cols: tensor.cols })
                    })?
                    .ok_or("CPU DSpark 首轮缺少完整 GPU target cache")?;
                cache = DsparkTargetCache::from_snapshot(snapshot);
                eprintln!("[glm52-dspark-cpu-cache] source=gpu old={range:?} range=[{start},{end}) layers={}", cache.warmed_layers());
            }
        }
        let actual = cache.covered_range(&backend);
        let can_roll = actual.as_ref().is_some_and(|range| start >= range.start && start < range.end && range.end <= end);
        if actual != Some(start..end) && !can_roll {
            return Err(format!("CPU DSpark target cache range={actual:?}，无法推进到 [{start},{end})"));
        }
        // CPU cache 已持有旧前缀时只下载尚未投影的 aux suffix。把完整 history
        // 每轮交给 backbone 会重复扫描并重投影全部历史，延迟会随 decode 长度增长。
        let suffix_start = actual.map_or(start, |range| range.end.min(end));
        let suffix_rows = end - suffix_start;
        let history = if suffix_rows == 0 {
            CpuTensor { data: Vec::new(), rows: 0, cols: gpu_history.cols }
        } else {
            let suffix = self.output_context().slice_token_rows(gpu_history, suffix_start - start, suffix_rows).map_err(|error| format!("切分 CPU DSpark aux suffix: {error:?}"))?;
            CpuTensor { data: self.output_context().tensor_to_f32(&suffix).map_err(|error| format!("下载 CPU DSpark aux suffix: {error:?}"))?, rows: suffix_rows, cols: gpu_history.cols }
        };
        Ok((history, cache, suffix_start, end))
    }

    fn dspark_cache_valid(&self, aux: Option<&RocmTensor>, start: usize, target: &DsparkTargetCache<RocmTensor>, token_count: usize) -> bool {
        let Some(runtime) = &self.dspark_runtime else { return aux.is_none() && target.warmed_layers() == 0 };
        let Some(aux) = aux.filter(|aux| aux.rows > 0 && aux.cols == self.cfg.hidden_size) else { return false };
        let Some(end) = start.checked_add(aux.rows) else { return false };
        let context = self.output_context();
        if end != token_count || target.covered_range(&context) != Some(start..end) {
            return false;
        }
        let Ok(Some(snapshot)) = target.try_snapshot(|tensor| Ok::<_, ()>((tensor.rows, tensor.cols))) else { return false };
        let columns = runtime.target_columns();
        snapshot.layers.len() == runtime.target_layer_count() && snapshot.layers.iter().all(|layer| layer.key == (aux.rows, columns) && layer.value == (aux.rows, columns))
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_prefill_window(
        pipeline: &StageSchedulerHandle<Glm52StageValue<RocmContext>, Glm52StageState>,
        slots: &[Option<Glm52BatchTask>],
        target: usize,
        work_window: usize,
        chunk_limit: Option<usize>,
        chunk_ceiling: usize,
        short_only: bool,
        submitted: &mut [usize],
        in_flight: &mut usize,
        in_flight_by_session: &mut [usize],
        admission: &mut OpportunisticPrefillAdmission,
        input_context: &RocmContext,
        weights: &Glm52Weights,
        cfg: &Glm52Config,
        embedding: Option<&std::sync::Arc<crate::kernel::rocm::hip::DeviceBuffer>>,
    ) -> Result<(), crate::backend::BackendError> {
        if slots.is_empty() {
            return Ok(());
        }
        let decode_limited = chunk_limit.is_some();
        let mut admitted = if decode_limited { in_flight_by_session.iter().filter(|&&count| count != 0).count() } else { *in_flight };
        while admitted < target && *in_flight < work_window {
            let remaining_admissions = target - admitted;
            let eligible = |session: usize| {
                let Some(task) = slots[session].as_ref() else { return false };
                if task.cancellation.load(Ordering::Acquire) || submitted[session] >= task.tokens.len() || decode_limited && in_flight_by_session[session] != 0 {
                    return false;
                }
                let remaining = task.tokens.len() - submitted[session];
                !short_only || remaining <= chunk_ceiling && remaining <= task.prefill_policy.chunk_size(task.prefill_suffix_start, submitted[session])
            };
            let eligible_sessions = slots.iter().enumerate().filter(|(session, _)| eligible(*session)).count().min(remaining_admissions).max(1);
            let Some(session) = pipeline.next_prefill_session(admission.cursor_mut(), slots.len(), eligible) else {
                break;
            };
            let task = slots[session].as_ref().expect("admission 只选择有工作的 session");
            let position = submitted[session];
            let chunk_size = task.prefill_policy.chunk_size(task.prefill_suffix_start, position);
            let remaining = task.tokens.len() - position;
            let chunk_size = chunk_limit.map_or(chunk_size, |limit| admission.chunk_size(chunk_size, limit, remaining));
            let end = position.saturating_add(chunk_size).min(task.tokens.len());
            let segment_budget = if decode_limited { (work_window - *in_flight).div_ceil(eligible_sessions) } else { 1 };
            let segments = prefill_token_segments(position, end - position, segment_budget);
            for segment in &segments {
                let hidden = match embedding {
                    Some(table) => {
                        gather_embedding_rows(input_context, table, &task.tokens[segment.clone()], cfg.hidden_size).map_err(|error| crate::backend::BackendError::Compute { msg: format!("准备 continuous prefill segment: {error}") })?
                    }
                    None => build_pp_prefill_chunk(input_context, weights, cfg, &task.tokens[segment.clone()], segment.start)
                        .map_err(|error| crate::backend::BackendError::Compute { msg: format!("准备 continuous prefill segment: {error:?}") })?,
                };
                pipeline.submit_glm52(session, segment.start, false, hidden, None)?;
            }
            submitted[session] = end;
            *in_flight += segments.len();
            in_flight_by_session[session] += segments.len();
            admitted += 1;
            pipeline.commit_prefill_submission(admission.cursor_mut(), session, end == task.tokens.len());
            admission.commit_work(end - position, segments.len());
        }
        Ok(())
    }

    /// 链头构造：`devices` 是 head 负责的卡，`layer_ends` 是每卡层边界（最后 = stage_end-1），
    /// `downstream_ticket` 是下游 `zllm-rt-rocm` 的 StageTransport ticket。
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: Arc<Glm52Weights>,
        cache_dir: &Path,
        persist_kv_cache: bool,
        devices: Vec<i32>,
        layer_ends: Vec<usize>,
        stage_end: usize,
        downstream_ticket: &str,
        downstream_iroh: IrohConfig,
        tokenizer_path: &Path,
        max_seq_len: usize,
        options: Glm52NodeExecutionConfig,
        lm_head_quantization: crate::weight::LmHeadQuantization,
        runtime: Arc<Mutex<NodeRuntime>>,
        compute_steps: Arc<AtomicCounterU64>,
    ) -> Result<Self, DynError> {
        let cfg = Glm52Config::standard();
        let model = Glm52::standard();
        let layer_spec = model.layer_spec(0)?;
        let mla = match &layer_spec.attention {
            AttentionSpec::Mla(spec) => spec.clone(),
            _ => return Err("GLM-5.2 第 0 层不是 MLA".into()),
        };
        if stage_end == 0 || stage_end >= cfg.layer_count {
            return Err(format!("Glm52Engine stage_end 必须在 1..{}，实际 {stage_end}", cfg.layer_count).into());
        }
        let (contexts, cooperative_peer_contexts) = if options.cooperative_expert_pairs {
            if devices.len() < 2 || devices.len() % 2 != 0 || layer_ends.len() != devices.len() / 2 {
                return Err(format!("cooperative_expert_pairs 要求每两张物理卡对应一个 layer_ends：devices={} layer_ends={}", devices.len(), layer_ends.len()).into());
            }
            let owner_devices = devices.iter().step_by(2).copied().collect::<Vec<_>>();
            let peer_devices = devices.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();
            let contexts = crate::runtime::rocm_chain::RocmDeviceChain::new(&owner_devices, layer_ends.clone(), stage_end - 1, false).map_err(|error| -> DynError { format!("GLM-5.2 双卡组 owner 设备链: {error}").into() })?.contexts;
            let peers =
                peer_devices.iter().map(|&device| RocmContext::configured(device, false).map_err(|error| -> DynError { format!("ROCm cooperative peer device {device} 初始化失败: {error}").into() })).collect::<Result<Vec<_>, _>>()?;
            (contexts, peers)
        } else {
            (crate::runtime::rocm_chain::RocmDeviceChain::new(&devices, layer_ends.clone(), stage_end - 1, false).map_err(|error| -> DynError { format!("GLM-5.2 设备链: {error}").into() })?.contexts, Vec::new())
        };
        let output_context = contexts[0];
        let embedding_table = load_resident_embedding(&output_context, &weights, &cfg).map_err(|error| -> DynError { error.into() })?;
        // LM head、采样与 MTP 全部驻留 A0;输出运行时由 rocm_tail 的统一装配入口
        // 提供(含 FR-Spec draft head 与 lm_head 量化),不再有尾端采样形态。
        let (output_head, mtp_runtime) =
            prepare_head_output_runtime(&output_context, &weights, &cfg, &mla, options.mtp, options.mtp_draft_tokens, None, lm_head_quantization, embedding_table.clone()).map_err(|error| -> DynError { error.into() })?;
        let dspark_runtime = options
            .dspark_directory
            .as_deref()
            .map(|root| {
                if options.dspark_backend == Glm52DsparkExecutionBackend::Cpu {
                    RocmDsparkRuntime::load_cache(&output_context, root, max_seq_len, options.dspark_weight_quantization).map_err(|error| -> DynError { format!("准备 A0 ROCm DSpark cache runtime: {error:?}").into() })
                } else {
                    RocmDsparkRuntime::load(&output_context, root, max_seq_len, options.dspark_draft_tokens, options.dspark_confidence_threshold, lm_head_quantization, options.dspark_weight_quantization)
                        .map_err(|error| -> DynError { format!("准备 A0 ROCm DSpark: {error:?}").into() })
                }
            })
            .transpose()?;
        let cpu_dspark_executor = if options.dspark_backend == Glm52DsparkExecutionBackend::Cpu {
            let root = options.dspark_directory.as_deref().ok_or("dspark_backend=cpu 要求配置 dspark_directory")?;
            if options.dspark_weight_quantization != crate::weight::ResidentWeightQuantization::Q8g128 {
                return Err("dspark_backend=cpu 当前要求 dspark_weight_quantization=q8g128".into());
            }
            let started = Instant::now();
            let load = || CpuDsparkRuntime::load(&CpuContext, root, max_seq_len, options.dspark_draft_tokens, options.dspark_confidence_threshold, lm_head_quantization, options.dspark_weight_quantization);
            let runtime = match options.dspark_cpu_affinity.as_deref() {
                Some(cpu_list) => crate::kernel::cpu::with_current_thread_affinity(cpu_list, load).map_err(|error| -> DynError { format!("准备 CPU DSpark NUMA 放置: {error}").into() })?,
                None => load(),
            }
            .map_err(|error| -> DynError { format!("准备 CPU DSpark: {error:?}").into() })?;
            eprintln!("[glm52-dspark-cpu-resident] drafts={} wall={:.3}s", options.dspark_draft_tokens, started.elapsed().as_secs_f64());
            Some(CpuDsparkExecutor::new(runtime, options.dspark_cpu_affinity.as_deref()).map_err(|error| -> DynError { error.into() })?)
        } else {
            None
        };
        let tokenizer = if tokenizer_path.is_file() {
            Tokenizer::new(tokenizer_path).map_err(|error| -> DynError { format!("加载 tokenizer: {error}").into() })?
        } else if let Ok(source) = weights.gguf_source() {
            source.tokenizer().map_err(|error| -> DynError { format!("从 GGUF 构造 tokenizer: {error}").into() })?
        } else {
            return Err(format!("tokenizer 文件不存在: {}", tokenizer_path.display()).into());
        };
        let thinking_end_token = if options.diagnostics.official_chat_template {
            let tokens = tokenizer.tokenize(b"</think>");
            match tokens.as_slice() {
                [token] => Some(*token),
                _ => return Err(format!("GLM-5.2 tokenizer 必须把 </think> 编码为单 token，实际 {tokens:?}").into()),
            }
        } else {
            None
        };
        let detokenizer = if tokenizer_path.is_file() {
            Detokenizer::load(tokenizer_path).map_err(|error| -> DynError { format!("加载 detokenizer: {error}").into() })?
        } else {
            weights.gguf_source().map_err(|error| -> DynError { error.into() })?.detokenizer().map_err(|error| -> DynError { format!("从 GGUF 构造 detokenizer: {error}").into() })?
        };
        let token_signatures = build_token_signatures(&detokenizer, cfg.vocab_size);
        let rope = Arc::new(RopeTable::precompute(max_seq_len, mla.qk_rope_head_dim, mla.rope_theta));
        prepare_glm52_rope_resident(&contexts, &rope)?;
        let pipeline_chunk_size = options.prefill_chunk_size;
        let mut link = StageTransport::connect(downstream_ticket, downstream_iroh).map_err(|error| -> DynError { format!("连接下游 stage: {error}").into() })?;
        let (downstream_memory, downstream_session_capacity) = match link.recv().map_err(|error| -> DynError { format!("接收下游设备资源: {error}").into() })?.message {
            StageMessage::DeviceMemory { devices, session_capacity } if !devices.is_empty() => (devices, session_capacity),
            message => return Err(format!("下游连接后未报告设备资源: {message:?}").into()),
        };
        let kv_f16 = options.kv_cache_format == KvCacheFormat::F16;
        let capability_contexts = contexts.iter().chain(&cooperative_peer_contexts).copied().collect::<Vec<_>>();
        let capabilities = crate::runtime::rocm_chain::node_capabilities(&capability_contexts, max_seq_len, if kv_f16 { "f16" } else { "q8g64" }, "glm52-rocm".to_owned(), 0);
        // resident terminal cache 由 KV token budget 淘汰，不再设固定条目上限。
        let terminal_limit = options.terminal_cache_entries;
        let kv_reservation_page_tokens = options.kv_reservation_page_tokens;
        let cache_identity = Glm52CacheIdentity::new(&weights, options.kv_cache_format, 0, stage_end, options.mtp, max_seq_len, options.dspark_directory.as_deref());
        let swap = persist_kv_cache.then(|| Glm52SwapStore::open(cache_dir.join("glm52").join(format!("stage-0-{stage_end}")), &cache_identity).map(Arc::new)).transpose()?;
        let mut engine = Self {
            cfg,
            mla,
            contexts,
            cooperative_peer_contexts,
            layer_ends,
            stage_end,
            weights,
            rope,
            tokenizer,
            detokenizer,
            token_signatures,
            thinking_end_token,
            max_seq_len,
            pipeline_chunk_size,
            link,
            downstream_memory,
            downstream_session_capacity,
            capabilities,
            runtime,
            compute_steps,
            output_head: Some(output_head),
            embedding_table,
            mtp_runtime,
            dspark_runtime,
            cpu_dspark_executor,
            next_cpu_dspark_job: 0,
            cpu_dspark_submit_profile: [0; 3],
            accept_phase_profile: [0; 7],
            resident_states: None,
            terminal_states: TerminalCache::new(terminal_limit),
            swap,
            persist_kv_cache,
            pending_messages: HashMap::new(),
            intake_capacity: 1,
            kv_budget: ResidencyBudget::new(1),
            kv_reservation_page_tokens,
            options,
        };
        // 分布式 decode 必须在注册调度器前准备全部 CT experts；否则首个 prompt
        // 只会按实际路由填充少量 experts，后续 decode 会长期退回 host route。
        engine.ensure_resident_states().map_err(|error| -> DynError { error.into() })?;
        let (intake_capacity, kv_token_capacity, kv_cache_devices) = engine.memory_capacity()?;
        engine.intake_capacity = intake_capacity;
        engine.kv_budget = ResidencyBudget::new(kv_token_capacity);
        engine.capabilities.kv_cache_devices = kv_cache_devices;
        engine.capabilities.kv_reservation_page_tokens = engine.kv_reservation_page_tokens;
        eprintln!(
            "[glm52-head] KV admission token_budget={} page_tokens={} max_active@1M={} intake_window={}",
            engine.kv_budget.capacity(),
            engine.kv_reservation_page_tokens,
            engine.kv_budget.capacity() / 1_048_576,
            engine.intake_capacity,
        );
        Ok(engine)
    }

    /// 初始化跨会话共享的 layer 和 expert 权重；每个 session 只复制 cache/DSA 状态。
    fn ensure_resident_states(&mut self) -> Result<(), String> {
        if self.resident_states.is_none() {
            let mut layers = prepare_prefill_layers(&self.contexts, &self.layer_ends, 0, self.stage_end, &self.cfg, &self.mla, &self.weights, crate::kernel::rocm::hip::options().prefill_attention_cpu)
                .map_err(|error| format!("准备 prefill layers: {error:?}"))?;
            let mut experts = (0..self.contexts.len()).map(|_| self.new_experts()).collect::<Result<Vec<_>, _>>()?;
            let preload_experts = self.options.preload_experts;
            if self.options.cooperative_expert_pairs {
                if !preload_experts || self.options.preload_layers_per_device.is_some() || !(self.weights.source_is_ct() || self.weights.source_is_gguf()) || self.contexts.len() != self.cooperative_peer_contexts.len() {
                    return Err("cooperative_expert_pairs 要求 CT/GGUF 权重、preload_experts=true、完整预载且每个逻辑 stage 有一张 peer 卡".to_owned());
                }
                let mut layer_start = 0usize;
                let counts = self
                    .layer_ends
                    .iter()
                    .map(|&layer_end| {
                        let count = layer_end + 1 - layer_start;
                        layer_start = layer_end + 1;
                        count
                    })
                    .collect::<Vec<_>>();
                for &count in &counts {
                    if count > 12 {
                        return Err(format!("cooperative expert 双卡组层数 {count} 超过 12 层预算"));
                    }
                }
                for (stage, (expert, &peer)) in experts.iter_mut().zip(&self.cooperative_peer_contexts).enumerate() {
                    expert.enable_cooperative_peer(peer, 0).map_err(|error| format!("配置 ROCm cooperative expert stage={stage}: {error:?}"))?;
                }
                let mla_started = Instant::now();
                let ct_source = self.weights.source_is_ct().then(|| self.weights.ct_source()).transpose()?;
                let mut mla_layers = 0usize;
                for (layer, resident) in layers.iter_mut().enumerate() {
                    let placement = self.layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| format!("L{layer} 没有 head cooperative MLA device"))?;
                    let dense = matches!(resident, glm52_stage::Glm52PrefillLayer::Dense(_));
                    let owner_weights = match &*resident {
                        glm52_stage::Glm52PrefillLayer::Dense(weights) => (weights.q_b_proj.clone(), weights.kv_b_proj.clone()),
                        glm52_stage::Glm52PrefillLayer::Moe(weights) => (weights.q_b_proj.clone(), weights.kv_b_proj.clone()),
                    };
                    let owner_o = match (dense, ct_source.as_ref()) {
                        (true, Some(source)) => {
                            let weights = source.load_dense_layer(layer).map_err(|error| format!("加载 L{layer} dense cooperative MLA 权重: {error}"))?;
                            super::rocm::prepare_cooperative_mla_dense_layer_ct(
                                &self.contexts[placement],
                                &self.cooperative_peer_contexts[placement],
                                &mut experts[placement],
                                &self.cfg,
                                &self.mla,
                                layer,
                                weights,
                                (&owner_weights.0, &owner_weights.1),
                            )?
                        }
                        (false, Some(source)) => {
                            let weights = source.load_moe_layer(layer).map_err(|error| format!("加载 L{layer} MoE cooperative MLA 权重: {error}"))?;
                            super::rocm::prepare_cooperative_mla_layer_ct(
                                &self.contexts[placement],
                                &self.cooperative_peer_contexts[placement],
                                &mut experts[placement],
                                &self.cfg,
                                &self.mla,
                                layer,
                                weights,
                                (&owner_weights.0, &owner_weights.1),
                            )?
                        }
                        (true, None) => {
                            let weights = self.weights.load_dense_layer_gguf(layer).map_err(|error| format!("加载 L{layer} dense GGUF cooperative MLA 权重: {error}"))?;
                            super::rocm::prepare_cooperative_mla_dense_layer_gguf(
                                &self.contexts[placement],
                                &self.cooperative_peer_contexts[placement],
                                &mut experts[placement],
                                &self.cfg,
                                &self.mla,
                                layer,
                                weights,
                                (&owner_weights.0, &owner_weights.1),
                            )?
                        }
                        (false, None) => {
                            let weights = self.weights.load_moe_layer_gguf(layer).map_err(|error| format!("加载 L{layer} MoE GGUF cooperative MLA 权重: {error}"))?;
                            super::rocm::prepare_cooperative_mla_layer_gguf(
                                &self.contexts[placement],
                                &self.cooperative_peer_contexts[placement],
                                &mut experts[placement],
                                &self.cfg,
                                &self.mla,
                                layer,
                                weights,
                                (&owner_weights.0, &owner_weights.1),
                            )?
                        }
                    };
                    match resident {
                        glm52_stage::Glm52PrefillLayer::Dense(weights) => weights.o_proj = owner_o,
                        glm52_stage::Glm52PrefillLayer::Moe(weights) => weights.o_proj = owner_o,
                    }
                    mla_layers += 1;
                }
                eprintln!(
                    "[glm52-cooperative-experts] physical_devices={} pairs={} logical_stages={} mode=attention-sequence+moe partition=kv-block-parity+gate-up-row/down-k-half device-route",
                    self.contexts.len() + self.cooperative_peer_contexts.len(),
                    self.contexts.len(),
                    self.contexts.len(),
                );
                eprintln!("[glm52-cooperative-mla-resident] layers={mla_layers} wall={:.3}s", mla_started.elapsed().as_secs_f64());
            }
            if preload_experts && (self.weights.source_is_ct() || self.weights.source_is_gguf()) {
                let started = Instant::now();
                let layers_per_device = self.options.preload_layers_per_device.unwrap_or(usize::MAX);
                let mut device_layers = vec![0_usize; self.contexts.len()];
                let mut resident_layers = 0_usize;
                for (layer, resident) in layers.iter().enumerate() {
                    if !matches!(resident, glm52_stage::Glm52PrefillLayer::Moe(_)) {
                        continue;
                    }
                    let placement = self.layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| format!("L{layer} 没有 head expert device"))?;
                    if device_layers[placement] >= layers_per_device {
                        continue;
                    }
                    let backend = &self.contexts[placement];
                    backend.activate().map_err(|error| format!("激活 ROCm device {}: {error}", backend.device_id()))?;
                    experts[placement].preload_layer(backend, layer, self.cfg.expert_count).map_err(|error| format!("预加载 ROCm L{layer} experts: {error:?}"))?;
                    device_layers[placement] += 1;
                    resident_layers += 1;
                }
                eprintln!("[glm52-head-expert-resident] layers={resident_layers} experts={} wall={:.3}s", self.cfg.expert_count, started.elapsed().as_secs_f64());
            }
            let mut states = build_glm52_stage_states(&self.contexts, &self.layer_ends, 0, &self.cfg, layers, experts, self.max_seq_len).map_err(|error| format!("build stage states: {error:?}"))?;
            if let Some(root) = self.options.dspark_directory.as_deref() {
                attach_dspark_projections(&mut states, root, self.cfg.layer_count, self.options.dspark_weight_quantization).map_err(|error| format!("加载 head DSpark projections: {error:?}"))?;
            }
            self.resident_states = Some(states);
        }
        Ok(())
    }

    /// 构造一组全新的 session stage states（cache miss 时使用）。
    fn build_fresh_states(&mut self) -> Result<Vec<Glm52StageState>, String> {
        self.ensure_resident_states()?;
        self.resident_states.as_ref().expect("resident stage 初始化后必须存在").iter().map(|state| state.fresh_session(&self.cfg, self.max_seq_len).map_err(|error| format!("创建 stage session: {error:?}"))).collect()
    }

    fn new_experts(&self) -> Result<RocmPrefillExperts, String> {
        if self.weights.source_is_nvfp4() {
            let source = self.weights.nvfp4_experts().ok_or("GLM-5.2 nvfp4 expert source 缺失")?;
            Ok(RocmPrefillExperts::nvfp4(source))
        } else if self.weights.source_is_ct() {
            Ok(RocmPrefillExperts::ct(self.weights.ct_source()?))
        } else if self.weights.source_is_gguf() {
            Ok(RocmPrefillExperts::gguf(self.weights.gguf_source()?))
        } else if self.weights.source_is_official() {
            RocmPrefillExperts::fp8(self.weights.cache_source_path(), self.cfg.expert_intermediate_size, self.cfg.hidden_size, self.cfg.expert_count)
        } else {
            Err("Glm52Engine 不支持当前 expert 权重源".to_owned())
        }
    }

    fn output_context(&self) -> RocmContext {
        // 输出/采样/MTP 固定驻留 A0 执行。
        self.contexts[0]
    }

    fn boundary_context(&self) -> RocmContext {
        // A 的最后一张卡产生传给 B 的 hidden。
        *self.contexts.last().unwrap()
    }

    fn wait_ready(&mut self, request_id: RequestId, expected_tokens: usize) -> Result<(), String> {
        match self.recv_stage(request_id)? {
            StageMessage::Ready { cached_tokens } if cached_tokens == expected_tokens => Ok(()),
            StageMessage::Ready { cached_tokens } => Err(format!("下游 cache tokens={cached_tokens}，期望 {expected_tokens}")),
            _ => Err("等待下游 cache ready 时收到非 Ready frame".to_owned()),
        }
    }

    fn wait_ready_after_cancel(&mut self, request_id: RequestId, expected_tokens: usize) -> Result<(), String> {
        loop {
            match self.recv_stage(request_id)? {
                StageMessage::Prefill { .. } | StageMessage::Decode { .. } | StageMessage::Verify { .. } | StageMessage::Token { .. } | StageMessage::Sampled { .. } | StageMessage::Speculative { .. } => continue,
                StageMessage::Ready { cached_tokens } if cached_tokens == expected_tokens => return Ok(()),
                StageMessage::Ready { cached_tokens } => return Err(format!("下游 cancel cache tokens={cached_tokens}，期望 {expected_tokens}")),
                _ => return Err("等待下游 cancel cache ready 时收到非法 frame".to_owned()),
            }
        }
    }

    fn recv_stage(&mut self, request_id: RequestId) -> Result<StageMessage, String> {
        if let Some(message) = self.pending_messages.get_mut(&request_id).and_then(VecDeque::pop_front) {
            return Ok(message);
        }
        loop {
            let frame = self.link.recv().map_err(|error| format!("等待下游 frame: {error}"))?;
            if frame.request_id == request_id {
                return Ok(frame.message);
            }
            self.pending_messages.entry(frame.request_id).or_default().push_back(frame.message);
        }
    }

    fn poll_stage(&mut self) -> Result<(), String> {
        while let Some((frame, queued)) = self.link.try_recv_timed().map_err(|error| format!("轮询下游 frame: {error}"))? {
            if self.options.diagnostics.trace_stage_events {
                let boundary = match &frame.message {
                    StageMessage::Decode { position, .. } => Some((*position, "Decode", 1)),
                    StageMessage::Verify { position, rows, .. } => Some((*position, "Verify", *rows)),
                    _ => None,
                };
                if let Some((position, kind, rows)) = boundary {
                    crate::runtime::prefill_scheduler::record_stage_trace(format!(
                        "[stage-boundary-trace] ts_us={} phase=poll direction=B_to_A request_id={} position={position} kind={kind} rows={rows} queue_us={}",
                        crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                        frame.request_id,
                        queued.as_micros(),
                    ));
                }
            }
            self.pending_messages.entry(frame.request_id).or_default().push_back(frame.message);
        }
        Ok(())
    }

    fn stage_hidden_ready(&self, request_id: RequestId) -> bool {
        self.pending_messages.get(&request_id).is_some_and(|messages| {
            messages.iter().any(|message| matches!(message, StageMessage::Prefill { .. } | StageMessage::Decode { .. } | StageMessage::Verify { .. } | StageMessage::Sampled { .. } | StageMessage::Speculative { .. }))
        })
    }

    fn snapshot(&self, cache_id: String, tokens: Vec<u32>, state: &Glm52HeadState) -> Result<Glm52CacheSnapshot, String> {
        let last_hidden = self.output_context().tensor_to_bf16_bits(&state.last_hidden).map_err(|error| format!("下载 head terminal hidden: {error:?}"))?;
        let dspark_aux = state
            .dspark_aux_history
            .as_ref()
            .map(|hidden| {
                let values = self.output_context().tensor_to_bf16_bits(hidden).map_err(|error| format!("下载 DSpark aux history: {error:?}"))?;
                Ok::<_, String>(Glm52DsparkAuxCache { start_position: state.dspark_aux_history_start, rows: hidden.rows, columns: hidden.cols, values })
            })
            .transpose()?;
        let dspark_target = match (&state.dspark_aux_history, &self.dspark_runtime) {
            (Some(_), Some(_)) => {
                let snapshot = state
                    .dspark_target_cache
                    .try_snapshot(|tensor| {
                        let values = self.output_context().tensor_to_f32(tensor).map_err(|error| format!("下载 DSpark target cache: {error:?}"))?;
                        Ok::<_, String>(Glm52DsparkTargetTensor { rows: tensor.rows, columns: tensor.cols, values })
                    })?
                    .ok_or("持久化 DSpark cache 时 target cache 未完整预热")?;
                Some(Glm52DsparkTargetCache { layers: snapshot.layers.into_iter().map(|layer| Glm52DsparkTargetLayerCache { start_position: layer.start_position, key: layer.key, value: layer.value }).collect() })
            }
            (None, None) => None,
            _ => return Err("持久化 DSpark cache 时 runtime/aux 状态不一致".to_owned()),
        };
        Ok(Glm52CacheSnapshot {
            cache_id,
            cache_namespace: state.cache_namespace.clone(),
            token_count: tokens.len(),
            tokens,
            pending_tokens: state.pending_tokens.clone(),
            last_hidden,
            stages: download_glm52_session(&state.states)?,
            mtp: state.mtp.as_ref().map(|mtp| mtp.snapshot(&self.output_context())).transpose()?,
            dspark_aux,
            dspark_target,
        })
    }

    /// A/B 必须按同一个 cache_id 一起换出；成功后返回可复用的 GPU session。
    fn evict_oldest_terminal(&mut self) -> Result<Option<Vec<Glm52StageState>>, String> {
        let Some((cache_id, tokens, state)) = self.terminal_states.take_oldest() else { return Ok(None) };
        let request_id = RequestId::from_cache_id(&cache_id);
        let prompt_tokens = state.info.prompt_tokens;
        if !self.persist_kv_cache {
            if let Err(error) = self.link.send_delete(request_id).map_err(|error| format!("命令下游释放 cache {cache_id}: {error}")).and_then(|()| self.wait_ready(request_id, prompt_tokens)) {
                self.terminal_states.insert(cache_id, tokens, state);
                return Err(error);
            }
            eprintln!("[glm52-head] A/B cache 已释放且未写入 SSD cache_id={cache_id} tokens={prompt_tokens}");
            return Ok(Some(state.states));
        }
        let snapshot = match self.snapshot(cache_id.clone(), tokens.clone(), &state) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.terminal_states.insert(cache_id, tokens, state);
                return Err(format!("准备 A cache 换出失败: {error}"));
            }
        };
        if let Err(error) = self.swap.as_ref().expect("启用 KV cache 持久化时 swap 必须存在").put(&snapshot) {
            self.terminal_states.insert(cache_id.clone(), tokens, state);
            return Err(format!("A cache 写入 Fjall 失败 cache_id={cache_id}: {error}"));
        }
        if let Err(error) = self.link.send_swap_out(request_id).map_err(|error| format!("命令下游换出 cache {cache_id}: {error}")).and_then(|()| self.wait_ready(request_id, prompt_tokens)) {
            let _ = self.swap.as_ref().expect("启用 KV cache 持久化时 swap 必须存在").delete(&cache_id);
            self.terminal_states.insert(cache_id.clone(), tokens, state);
            return Err(error);
        }
        eprintln!("[glm52-head] A/B cache 已换出 SSD cache_id={cache_id} tokens={prompt_tokens}");
        Ok(Some(state.states))
    }

    fn ensure_terminal_slot(&mut self) -> Result<(), String> {
        if self.terminal_states.is_full() && self.evict_oldest_terminal()?.is_none() {
            return Err("terminal cache 已满但没有可换出状态".to_owned());
        }
        Ok(())
    }

    fn memory_capacity(&self) -> Result<(usize, usize, Vec<KvCacheDeviceCapacity>), String> {
        let safety_bytes = self.options.memory_reserve_bytes;
        // 链头不知道下游的物理分层，部署可以用整条流水线最大单卡层数覆盖。
        let admission_layers = self.options.kv_admission_layers_per_device;
        let latent_bytes = if self.options.kv_cache_format == KvCacheFormat::F16 { self.mla.kv_lora_rank * 2 } else { self.mla.kv_lora_rank + self.mla.kv_lora_rank / crate::kv_cache::DEFAULT_GROUP_SIZE * 2 };
        let mla_bytes = latent_bytes + self.mla.qk_rope_head_dim * 2;
        let index_bytes = self.cfg.index_head_dim * 2;
        // pair 卡按固定 block parity 各持有一半 MLA KV；DSA 为按 query rows
        // 并行扫描完整历史，Indexer cache 仍各保留一份。
        let kv_bytes_per_layer_token = if self.options.cooperative_expert_pairs { mla_bytes.div_ceil(2) + index_bytes } else { mla_bytes + index_bytes };
        let mut layer_start = 0usize;
        let mut token_capacity = usize::MAX;
        let mut devices = Vec::with_capacity(self.contexts.len() + self.downstream_memory.len());
        for (device_index, (context, &layer_end)) in self.contexts.iter().zip(&self.layer_ends).enumerate() {
            let layers = layer_end + 1 - layer_start + usize::from(device_index == 0 && self.options.mtp) * self.cfg.mtp_layer_count;
            layer_start = layer_end + 1;
            let free = context.stage_available_bytes().map_err(|error| format!("查询 ROCm device {} 可用显存: {error:?}", context.device_id()))?;
            let total = context.stage_total_bytes().map_err(|error| format!("查询 ROCm device {} 总显存: {error:?}", context.device_id()))?;
            let device = crate::runtime::rocm_chain::kv_capacity_from_free(format!("local/rocm-device-{}", context.device_id()), free, total, layers, kv_bytes_per_layer_token, safety_bytes, admission_layers)?;
            token_capacity = token_capacity.min(device.token_capacity);
            eprintln!("[glm52-head] {} kv_budget={:.2}GiB layers={} admission_layers={} kv_bytes/token={}", device.device, device.available_bytes as f64 / (1_u64 << 30) as f64, layers, layers.max(admission_layers), device.bytes_per_token,);
            devices.push(device);
        }
        for report in &self.downstream_memory {
            let free = usize::try_from(report.available_bytes).map_err(|_| format!("下游 device {} 可用显存超过 usize", report.device))?;
            let total = usize::try_from(report.total_bytes).map_err(|_| format!("下游 device {} 总显存超过 usize", report.device))?;
            let device = crate::runtime::rocm_chain::kv_capacity_from_free(format!("downstream/rocm-device-{}", report.device), free, total, report.model_units, kv_bytes_per_layer_token, safety_bytes, admission_layers)?;
            token_capacity = token_capacity.min(device.token_capacity);
            eprintln!(
                "[glm52-head] {} kv_budget={:.2}GiB layers={} admission_layers={} kv_bytes/token={}",
                device.device,
                device.available_bytes as f64 / (1_u64 << 30) as f64,
                report.model_units,
                report.model_units.max(admission_layers),
                device.bytes_per_token,
            );
            devices.push(device);
        }
        let token_capacity = token_capacity.max(1);
        let memory_capacity = (token_capacity / self.kv_reservation_page_tokens).max(1);
        let intake_capacity = self.downstream_session_capacity.map_or(memory_capacity, |capacity| memory_capacity.min(capacity));
        if let Some(capacity) = self.downstream_session_capacity {
            eprintln!("[glm52-head] downstream_session_capacity={capacity} memory_session_capacity={memory_capacity} intake_capacity={intake_capacity}");
        }
        Ok((intake_capacity, token_capacity, devices))
    }

    fn reservation_tokens(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.kv_reservation_page_tokens).saturating_mul(self.kv_reservation_page_tokens)
    }

    fn request_reservation_tokens(&self, prompt_tokens: usize, max_tokens: usize) -> usize {
        let max_decode = max_tokens.min(self.max_seq_len.saturating_sub(prompt_tokens));
        self.reservation_tokens(prompt_tokens.saturating_add(max_decode)).min(self.max_seq_len)
    }

    fn resident_kv_cost(&self) -> usize {
        let page_tokens = self.kv_reservation_page_tokens;
        self.terminal_states.resident_cost(|tokens, _| tokens.len().div_ceil(page_tokens).saturating_mul(page_tokens))
    }

    fn grow_kv_reservation(&mut self, reservation: &mut ResidencyReservation, required_tokens: usize) -> Result<(), String> {
        let target = self.reservation_tokens(required_tokens);
        while reservation.cost() < target {
            let additional = target - reservation.cost();
            while self.kv_budget.used().saturating_add(self.resident_kv_cost()).saturating_add(additional) > self.kv_budget.capacity() {
                if self.evict_oldest_terminal()?.is_none() {
                    return Err(format!("GLM-5.2 KV 增长需要 {additional} tokens，当前可用 {} tokens", self.kv_budget.available()));
                }
            }
            if !reservation.try_grow(additional) {
                return Err(format!("GLM-5.2 KV 增长竞争: request={additional} available={}", self.kv_budget.available()));
            }
        }
        Ok(())
    }

    fn send_open_with_reconnect(&mut self, request_id: RequestId, cache_request_id: Option<RequestId>, cached_tokens: usize, reserved_rows: usize, cache_hit: bool, sampling: SamplingConfig) -> Result<(), String> {
        let first_error = match self.link.send_open_reserved(request_id, cache_request_id, cached_tokens, reserved_rows, cache_hit, sampling, false) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        eprintln!("[stage-downstream-reconnect] Open 前连接失效，开始重连: {first_error}");
        self.link.reconnect_downstream().map_err(|error| format!("下游 stage 重连失败（原错误: {first_error}）: {error}"))?;
        let (devices, session_capacity) = match self.link.recv().map_err(|error| format!("下游 stage 重连后接收设备资源失败: {error}"))?.message {
            StageMessage::DeviceMemory { devices, session_capacity } if !devices.is_empty() => (devices, session_capacity),
            message => return Err(format!("下游 stage 重连后未报告设备资源: {message:?}")),
        };
        if session_capacity != self.downstream_session_capacity {
            return Err(format!("下游 stage 重连后 session capacity 改变: {:?} -> {:?}", self.downstream_session_capacity, session_capacity));
        }
        self.downstream_memory = devices;
        self.link.send_open_reserved(request_id, cache_request_id, cached_tokens, reserved_rows, cache_hit, sampling, false).map_err(|error| format!("下游 stage 重连后再次 Open 失败: {error}"))
    }
}

impl NodeEngine for Glm52Engine {
    fn model_key(&self) -> &'static str {
        "glm-5.2"
    }

    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        // ROCm 当前显存占用：暂报 0（scheduler 不依赖精确值；精确统计可后续接 hip mem pool）。
        let allocated = Arc::new(|| 0u64);
        (self.capabilities.clone(), allocated)
    }

    fn refresh_runtime(&self) {
        if let Ok(mut runtime) = self.runtime.lock() {
            crate::runtime::session::refresh_cache_runtime(&mut runtime, self.terminal_states.states().map(|state| &state.info), self.max_seq_len);
            runtime.memory_cache_ids = self.terminal_states.states().map(|state| state.info.cache_id.clone()).collect();
            runtime.memory_cache_ids.sort();
            runtime.ssd_cache_ids = self.swap.as_ref().map_or_else(Vec::new, |swap| swap.infos().into_iter().map(|info| info.cache_id).collect());
            runtime.ssd_cache_ids.sort();
        }
    }

    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        let mut infos = self.terminal_states.states().map(|state| (state.info.cache_id.clone(), state.info.clone())).collect::<HashMap<_, _>>();
        if let Some(swap) = &self.swap {
            for swapped in swap.infos() {
                infos.entry(swapped.cache_id.clone()).or_insert(CacheInfo {
                    cache_id: swapped.cache_id,
                    model_key: "glm-5.2".to_owned(),
                    cache_format: "glm52-rocm-mla-fjall-v3".to_owned(),
                    last_layer: self.cfg.layer_count.saturating_sub(1),
                    prompt_tokens: swapped.token_count,
                    bytes: swapped.resident_bytes,
                    modified_unix: swapped.modified_unix,
                });
            }
        }
        infos.into_values().collect()
    }

    fn shutdown(&mut self) -> Result<(), String> {
        let shutdown_id = RequestId::from_cache_id("zllm-stage-shutdown:v1");
        let persist = self.swap.is_some();
        self.link.send_shutdown(shutdown_id, persist).map_err(|error| format!("通知下游持久化并退出: {error}"))?;
        let mut local_error = None;
        while let Some((cache_id, tokens, state)) = self.terminal_states.take_oldest() {
            if let Some(swap) = &self.swap {
                let result = self.snapshot(cache_id.clone(), tokens, &state).and_then(|snapshot| swap.put(&snapshot));
                if let Err(error) = result {
                    eprintln!("[glm52-head] 优雅退出持久化 A cache 失败 cache_id={cache_id}: {error}");
                    if local_error.is_none() {
                        local_error = Some(error);
                    }
                } else {
                    eprintln!("[glm52-head] 优雅退出，A 已持久化 cache_id={cache_id}");
                }
            }
        }
        let downstream = self.wait_ready(shutdown_id, 0).map_err(|error| format!("等待下游持久化退出: {error}"));
        self.refresh_runtime();
        downstream?;
        if let Some(error) = local_error {
            return Err(format!("A cache 持久化失败: {error}"));
        }
        eprintln!("[glm52-head] 已收到 B 持久化退出 ACK，A 现在退出");
        Ok(())
    }

    fn max_concurrency(&self) -> usize {
        self.intake_capacity
    }

    fn generate_batch(
        &mut self,
        requests: Vec<NodeBatchRequest>,
        intake: &mut dyn FnMut(usize) -> Vec<NodeBatchRequest>,
        on_token: &mut dyn FnMut(&str, u32, String) -> bool,
        on_tool_call_delta: &mut dyn FnMut(&str, ToolCallDelta) -> bool,
        on_runtime_changed: &mut dyn FnMut(),
        on_result: &mut dyn FnMut(NodeBatchResult),
    ) -> Vec<NodeBatchResult> {
        self.generate_glm52_batch(requests, intake, on_token, on_tool_call_delta, on_runtime_changed, on_result)
    }
}

impl Glm52Engine {
    fn generate_glm52_batch(
        &mut self,
        requests: Vec<NodeBatchRequest>,
        intake: &mut dyn FnMut(usize) -> Vec<NodeBatchRequest>,
        on_token: &mut dyn FnMut(&str, u32, String) -> bool,
        on_tool_call_delta: &mut dyn FnMut(&str, ToolCallDelta) -> bool,
        on_runtime_changed: &mut dyn FnMut(),
        on_result: &mut dyn FnMut(NodeBatchResult),
    ) -> Vec<NodeBatchResult> {
        let capacity = self.max_concurrency();
        let mut pending = VecDeque::new();
        for request in requests {
            let request_id = request.request_id.clone();
            match self.prepare_batch_task(request) {
                Ok(task) => pending.push_back(task),
                Err(message) => on_result(NodeBatchResult { request_id, result: Err(message) }),
            }
        }
        for request in intake(capacity.saturating_sub(pending.len())) {
            let request_id = request.request_id.clone();
            match self.prepare_batch_task(request) {
                Ok(task) => pending.push_back(task),
                Err(message) => on_result(NodeBatchResult { request_id, result: Err(message) }),
            }
        }

        let mut slots = std::iter::repeat_with(|| None).take(capacity).collect::<Vec<Option<Glm52BatchTask>>>();
        while slots.iter().any(Option::is_none) {
            let Some(pending_task) = pending.pop_front() else { break };
            // 不能先用 available() 拒绝；open_batch_task 会按总预算换出可驱逐的
            // terminal cache，再为新 active session 做最终 reservation。
            let session = slots.iter().position(Option::is_none).expect("初始 session 数不超过 capacity");
            let request_id = pending_task.input.request_id.clone();
            match self.open_batch_task(pending_task) {
                // 先把整批 Open 发到 B，才可能让两端 SSD restore 真正并行。
                Ok(task) => slots[session] = Some(task),
                Err(message) => on_result(NodeBatchResult { request_id, result: Err(message) }),
            }
        }
        let mut opened = Vec::with_capacity(capacity);
        for mut task in slots.iter_mut().filter_map(Option::take) {
            let request_id = task.request_id.clone();
            match self.finish_batch_open(&mut task) {
                Ok(()) => opened.push(task),
                Err(message) => {
                    let _ = self.link.send_delete(task.stage_id);
                    on_result(NodeBatchResult { request_id, result: Err(message) });
                }
            }
        }
        for (slot, task) in slots.iter_mut().zip(opened) {
            *slot = Some(task);
        }
        if slots.iter().all(Option::is_none) {
            while let Some(task) = pending.pop_front() {
                on_result(NodeBatchResult { request_id: task.input.request_id, result: Err(format!("GLM-5.2 请求需要预留 {} tokens，超过单机 KV 预算 {} tokens", task.reserved_tokens, self.kv_budget.capacity(),)) });
            }
            return Vec::new();
        }

        if crate::kernel::rocm::hip::options().kernel_profile {
            crate::kernel::rocm::hip::hip_api_stats::report_phase("glm52-post-open");
        }

        let initial_count = slots.iter().take_while(|task| task.is_some()).count();
        let initial_request_ids = slots[..initial_count].iter().map(|task| task.as_ref().expect("初始 slot 连续").stage_id).collect::<Vec<_>>();
        // stream 控制帧不能复用业务 request_id。异常清理时 Delete 的 Ready 可能和
        // stream-end ACK 同时在途；独立 id 才能无歧义地等待 B 完成整批回收。
        let stream_control_id = RequestId::from_cache_id(&format!("zllm-continuous-stream:{first}", first = initial_request_ids[0]));
        if let Err(message) = self.link.send_continuous_stream(&initial_request_ids) {
            for task in slots.iter_mut().filter_map(Option::take) {
                let _ = self.link.send_delete(task.stage_id);
                on_result(NodeBatchResult { request_id: task.request_id, result: Err(message.clone()) });
            }
            return Vec::new();
        }
        for task in slots[..initial_count].iter_mut().filter_map(Option::as_mut).filter(|task| task.cached_output_ready) {
            if let Err(message) = self.link.send_prefill_done(task.stage_id, task.cached_tokens.len()) {
                on_result(NodeBatchResult { request_id: task.request_id.clone(), result: Err(message) });
                return Vec::new();
            }
        }
        let initial_states = slots[..initial_count].iter_mut().map(|task| std::mem::take(&mut task.as_mut().expect("初始 slot 连续").states)).collect::<Vec<_>>();
        let cfg = self.cfg.clone();
        let mla = self.mla.clone();
        let rope = self.rope.clone();
        let output_context = self.boundary_context();
        let hidden_size = self.cfg.hidden_size;
        let index_top_k = self.cfg.index_top_k;
        let scheduler_policy = crate::runtime::glm52::stage::Glm52SchedulerPolicy {
            execution_slots: self.options.scheduling.execution_slots,
            decode_execution_slots: self.options.scheduling.decode_execution_slots,
            pipeline_work_window: self.options.scheduling.pipeline_work_window,
            prefill_admission_burst: self.options.scheduling.prefill_admission_burst,
            decode_batch_limit: self.options.scheduling.decode_batch_limit,
            prefill_batch_limit: self.options.scheduling.prefill_batch_limit,
            profile_completion: self.options.scheduling.profile_completion,
        };
        let profile_completion = self.options.scheduling.profile_completion;
        if self.options.diagnostics.trace_stage_events {
            crate::runtime::prefill_scheduler::enable_stage_event_trace();
        }
        if profile_completion {
            crate::kernel::rocm::hip::enable_device_profile();
        }
        let diagnostics = self.options.diagnostics;
        let weights = Arc::clone(&self.weights);
        let decode_pipeline_stage_count = self.contexts.len().saturating_add(self.downstream_memory.len());
        let run = drive_glm52_stream_stage_pipeline_stateful(initial_states, capacity, &cfg, &mla, &rope, scheduler_policy, |pipeline| {
            let backend_error = |msg: String| crate::backend::BackendError::Compute { msg };
            let mut request_sessions = slots.iter().enumerate().filter_map(|(session, task)| task.as_ref().map(|task| (task.stage_id, session))).collect::<HashMap<_, _>>();
            let mut closing = vec![false; capacity];
            let profile_boundaries = diagnostics.profile_boundaries;
            let mut send_profile = [0_u128; 5];
            let mut send_profile_count = 0_usize;
            let mut token_profile = [0_u128; 4];
            let mut token_profile_count = 0_usize;
            // 只累计调度数量，不同步设备；用来区分 head 没组成 wave 与
            // stage 收到 wave 后又拆成 singleton 两类完全不同的问题。
            let mut decode_wave_profile = [0_usize; 5];
            let mut decode_started = std::iter::repeat_with(VecDeque::<Instant>::new).take(capacity).collect::<Vec<_>>();
            let mut round_started = std::iter::repeat_with(|| None).take(capacity).collect::<Vec<Option<(Instant, usize)>>>();
            let mut allocation_profile_round = 0_usize;
            let mut round_profile_count = 0_usize;
            let mut round_profile_rows = 0_usize;
            let mut round_profile_micros = 0_u128;
            let mut round_profile_max_micros = 0_u128;
            let mut round_profile_lane_count = vec![0_usize; capacity];
            let mut round_profile_lane_micros = vec![0_u128; capacity];
            let mut dspark_supply_profile = [0_usize; 7];
            let mut submitted_prefill = slots.iter().map(|task| task.as_ref().map_or(0, |task| task.prefill_position)).collect::<Vec<_>>();
            let mut prefill_in_flight = 0_usize;
            let mut prefill_in_flight_by_session = vec![0_usize; capacity];
            let mut prefill_admission = OpportunisticPrefillAdmission::default();
            let mut ready_backlog = VecDeque::<Glm52TailReady>::with_capacity(capacity * 2);
            let mut coordinator_idle_trace = None::<(Instant, String)>;
            let mut decode_scheduler = BatchScheduler::new(1, self.options.scheduling.decode_batch_limit.min(capacity)).map_err(backend_error)?;
            decode_scheduler.align_initial_batch(initial_count);
            let decode_active = slots.iter().flatten().any(|task| task.prefill_position >= task.tokens.len());
            let (mut prefill_target, prefill_work_window, prefill_chunk_limit) = prefill_admission.limits(
                pipeline.stage_flow_snapshot(),
                pipeline.pipeline_work_window(),
                pipeline.stage_count(),
                decode_active,
                self.options.scheduling.decode_priority_prefill_chunk_size,
                self.options.scheduling.decode_priority_prefill_chunk_ceiling,
            );
            let short_only =
                prefill_target == 0 && prefill_admission.short_request_ready() && Self::has_finishable_short_prefill(&slots, &submitted_prefill, &prefill_in_flight_by_session, self.options.scheduling.decode_priority_prefill_chunk_ceiling);
            prefill_target += usize::from(short_only);
            Self::fill_prefill_window(
                pipeline,
                &slots,
                prefill_target,
                prefill_work_window,
                prefill_chunk_limit,
                self.options.scheduling.decode_priority_prefill_chunk_ceiling,
                short_only,
                &mut submitted_prefill,
                &mut prefill_in_flight,
                &mut prefill_in_flight_by_session,
                &mut prefill_admission,
                &self.contexts[0],
                &weights,
                &cfg,
                self.embedding_table.as_ref(),
            )?;

            let mut reported_load = None;
            loop {
                let load = slots.iter().flatten().fold((0usize, 0usize, 0usize), |mut load, task| {
                    if task.prefill_position >= task.tokens.len() {
                        load.2 += 1;
                    } else if task.prefill_suffix_start == 0 {
                        load.0 += 1;
                    } else {
                        load.1 += 1;
                    }
                    load
                });
                let dspark = crate::runtime::session::effective_dspark_draft_tokens(self.options.dspark_draft_tokens, load.2);
                let mut memory_cache_ids = self.terminal_states.states().map(|state| state.info.cache_id.clone()).collect::<Vec<_>>();
                memory_cache_ids.sort();
                let mut ssd_cache_ids = self.swap.as_ref().map_or_else(Vec::new, |swap| swap.infos().into_iter().map(|info| info.cache_id).collect());
                ssd_cache_ids.sort();
                let current = (load.0, load.1, load.2, dspark, memory_cache_ids.clone(), ssd_cache_ids.clone());
                if reported_load.as_ref() != Some(&current) {
                    if let Ok(mut runtime) = self.runtime.lock() {
                        runtime.new_prefill = load.0;
                        runtime.append_prefill = load.1;
                        runtime.decode = load.2;
                        runtime.dspark_draft_tokens = dspark;
                        runtime.memory_cache_ids = memory_cache_ids;
                        runtime.ssd_cache_ids = ssd_cache_ids;
                    }
                    reported_load = Some(current);
                    on_runtime_changed();
                }
                let mut progressed = false;
                while let Some(output) = pipeline.try_recv()? {
                    progressed = true;
                    match output {
                        StageSchedulerOutput::Work { cohort, session, position, value } => {
                            let task = slots.get_mut(session).and_then(Option::as_mut).ok_or_else(|| backend_error(format!("continuous 输出落到空 session={session}")))?;
                            let rows = value.hidden.rows;
                            let critical = value.decode || value.verify;
                            let chain_micros = if critical { decode_started[session].pop_front().map(|started| started.elapsed().as_micros()).unwrap_or(0) } else { 0 };
                            let boundary_started = (profile_boundaries && critical).then(Instant::now);
                            let values = output_context.completed_tensor_to_bf16_bits(&value.hidden).map_err(|error| backend_error(format!("下载 continuous BF16: {error:?}")))?;
                            let aux_values =
                                value.aux_hidden.as_ref().map(|hidden| output_context.completed_tensor_to_bf16_bits(hidden)).transpose().map_err(|error| backend_error(format!("下载 continuous aux hidden: {error:?}")))?.unwrap_or_default();
                            let d2h_micros = boundary_started.map(|started| started.elapsed().as_micros()).unwrap_or(0);
                            if critical && diagnostics.trace_stage_output {
                                let hash = values.iter().fold(0xcbf29ce484222325_u64, |hash, &value| (hash ^ u64::from(value)).wrapping_mul(0x100000001b3));
                                // verify 的行级数值摘要：容差内比较（位级 hash 会被
                                // kernel 路径差异的 1-ulp 淹没），前 4 值 + L2 范数。
                                let row_digest = (0..rows)
                                    .map(|row| {
                                        let start = row * hidden_size;
                                        let row_values = &values[start..start + hidden_size];
                                        let head4 = row_values.iter().take(4).map(|&bits| half::bf16::from_bits(bits).to_f32()).collect::<Vec<_>>();
                                        let norm = row_values.iter().map(|&bits| f64::from(half::bf16::from_bits(bits).to_f32())).map(|value| value * value).sum::<f64>().sqrt();
                                        format!("[{head4:?} |{norm:.3}]")
                                    })
                                    .collect::<Vec<_>>();
                                let kind = if value.verify { "Verify" } else { "Decode" };
                                eprintln!("[glm52-stage-output] kind={kind} session={session} position={position} rows={rows} hash={hash:016x} digest={row_digest:?}");
                            }
                            let selection = if critical || position.saturating_add(rows) > index_top_k { value.selection.map(|selection| selection.to_host_completed()).transpose()?.unwrap_or_default() } else { Vec::new() };
                            let selection_micros = boundary_started.map(|started| started.elapsed().as_micros()).unwrap_or(0);
                            if value.verify {
                                let (cohort, cohort_size) = cohort.unwrap_or((0, 1));
                                let trace_started = diagnostics.trace_stage_events.then(|| (Instant::now(), crate::runtime::prefill_scheduler::stage_trace_timestamp_us()));
                                self.link.send_verify_cohort_aux(task.stage_id, cohort, cohort_size, position, rows, hidden_size, &values, &selection, &aux_values, value.aux_taps).map_err(backend_error)?;
                                if let Some((started, begin_us)) = trace_started {
                                    crate::runtime::prefill_scheduler::record_stage_trace(format!(
                                        "[stage-boundary-trace] ts_us={begin_us} phase=send direction=A_to_B request_id={} lane={session}@{position} kind=Verify rows={rows} complete_us={} duration_us={}",
                                        task.stage_id,
                                        crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                                        started.elapsed().as_micros()
                                    ));
                                }
                                continue;
                            }
                            let last_hidden_micros = boundary_started.map(|started| started.elapsed().as_micros()).unwrap_or(0);
                            if value.decode {
                                let (cohort, cohort_size) = cohort.unwrap_or((0, 1));
                                let trace_started = diagnostics.trace_stage_events.then(|| (Instant::now(), crate::runtime::prefill_scheduler::stage_trace_timestamp_us()));
                                self.link.send_decode_cohort_aux(task.stage_id, cohort, cohort_size, position, hidden_size, &values, &selection, &aux_values, value.aux_taps).map_err(backend_error)?;
                                if let Some((started, begin_us)) = trace_started {
                                    crate::runtime::prefill_scheduler::record_stage_trace(format!(
                                        "[stage-boundary-trace] ts_us={begin_us} phase=send direction=A_to_B request_id={} lane={session}@{position} kind=Decode rows={rows} complete_us={} duration_us={}",
                                        task.stage_id,
                                        crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                                        started.elapsed().as_micros()
                                    ));
                                }
                                if let Some(started) = boundary_started {
                                    let total_micros = started.elapsed().as_micros();
                                    send_profile[0] += d2h_micros;
                                    send_profile[1] += selection_micros.saturating_sub(d2h_micros);
                                    send_profile[2] += last_hidden_micros.saturating_sub(selection_micros);
                                    send_profile[3] += total_micros.saturating_sub(last_hidden_micros);
                                    send_profile[4] += chain_micros;
                                    send_profile_count += 1;
                                    if send_profile_count == 32 {
                                        eprintln!(
                                            "[glm52-boundary-a-send] tokens=32 chain_ms={:.3} d2h_ms={:.3} selection_ms={:.3} last_hidden_ms={:.3} transport_ms={:.3}",
                                            send_profile[4] as f64 / 1000.0,
                                            send_profile[0] as f64 / 1000.0,
                                            send_profile[1] as f64 / 1000.0,
                                            send_profile[2] as f64 / 1000.0,
                                            send_profile[3] as f64 / 1000.0
                                        );
                                        send_profile = [0; 5];
                                        send_profile_count = 0;
                                    }
                                }
                                continue;
                            }
                            self.link.send_prefill_aux(task.stage_id, position, rows, hidden_size, &values, &selection, &aux_values, value.aux_taps).map_err(backend_error)?;
                            let end = position.saturating_add(rows);
                            if position != task.prefill_position || end > task.tokens.len() {
                                return Err(backend_error(format!("continuous prefill 边界错误: request={} position={position} expected={} rows={rows} tokens={}", task.stage_id, task.prefill_position, task.tokens.len(),)));
                            }
                            prefill_in_flight = prefill_in_flight.checked_sub(1).ok_or_else(|| backend_error("continuous prefill 在途计数下溢".to_owned()))?;
                            prefill_in_flight_by_session[session] = prefill_in_flight_by_session[session].checked_sub(1).ok_or_else(|| backend_error(format!("continuous prefill session={session} 在途计数下溢")))?;
                            task.cached_tokens.extend_from_slice(&task.tokens[position..end]);
                            task.prefill_position = end;
                            if end == task.tokens.len() {
                                self.link.send_prefill_done(task.stage_id, end).map_err(backend_error)?;
                            }
                        }
                        StageSchedulerOutput::Opened { .. } => {}
                        StageSchedulerOutput::Closed { session, states } => {
                            round_started[session] = None;
                            prefill_in_flight = prefill_in_flight.checked_sub(prefill_in_flight_by_session[session]).ok_or_else(|| backend_error(format!("continuous Close session={session} 在途计数下溢")))?;
                            prefill_in_flight_by_session[session] = 0;
                            let mut task = slots.get_mut(session).and_then(Option::take).ok_or_else(|| backend_error(format!("continuous Close 落到空 session={session}")))?;
                            task.states = states;
                            request_sessions.remove(&task.stage_id);
                            let request_id = task.request_id.clone();
                            let finish_reason = task.finish_reason.clone();
                            let mut result = self.finish_batch_task(task, on_token, on_tool_call_delta);
                            let devices = self.contexts.iter().chain(&self.cooperative_peer_contexts).map(RocmContext::device_id).collect::<Vec<_>>();
                            if let Err(error) = rocm_chain::release_request_workspaces(&devices) {
                                let release_error = format!("GLM-5.2 terminal workspace 回收失败: {error}");
                                result = Err(match result {
                                    Ok(_) => release_error,
                                    Err(run_error) => format!("{run_error}; {release_error}"),
                                });
                            } else {
                                eprintln!("[rocm-workspace] model=glm-5.2 request_id={request_id} finish_reason={finish_reason} released=true");
                            }
                            on_result(NodeBatchResult { request_id, result });
                            closing[session] = false;
                        }
                    }
                }

                for session in 0..capacity {
                    if closing[session] {
                        continue;
                    }
                    let Some(task) = slots[session].as_mut() else { continue };
                    if !task.cancellation.load(Ordering::Acquire) {
                        continue;
                    }
                    progressed = true;
                    ready_backlog.retain(|item| item.session != session);
                    task.finish_reason = "cancelled".to_owned();
                    pipeline.cancel(session)?;
                    pipeline.close(session)?;
                    closing[session] = true;
                }

                let cpu_results = self.cpu_dspark_executor.as_ref().map(CpuDsparkExecutor::drain_ready).unwrap_or_default();
                if !cpu_results.is_empty() {
                    progressed = true;
                    for result in cpu_results {
                        if result.session >= slots.len() || closing[result.session] {
                            continue;
                        }
                        let Some(task) = slots[result.session].as_mut() else { continue };
                        if task.dspark_cpu_pending != Some(result.id) {
                            continue;
                        }
                        task.dspark_cpu_pending = None;
                        task.dspark_cpu_target_cache = result.cache;
                        let anchor_position = task.cached_tokens.len().saturating_sub(1);
                        if !task.dspark_cpu_anchor_in_flight || task.pending_verify_rows != 1 || task.cached_tokens.get(anchor_position) != Some(&result.anchor) {
                            return Err(backend_error(format!(
                                "CPU DSpark job={} late verify 非法: anchor={} position={} token={:?} in_flight={} pending={}",
                                result.id,
                                result.anchor,
                                anchor_position,
                                task.cached_tokens.get(anchor_position),
                                task.dspark_cpu_anchor_in_flight,
                                task.pending_verify_rows,
                            )));
                        }
                        let drafts = result.drafts.map_err(backend_error)?;
                        let flight = Glm52DsparkCpuFlight::new(result.anchor, drafts, task.dspark_cpu_window);
                        let inputs = flight.inputs();
                        let suffix = &inputs[1..];
                        let result_us = diagnostics.trace_stage_events.then(crate::runtime::prefill_scheduler::stage_trace_timestamp_us);
                        task.dspark_verify_inputs = inputs.clone();
                        let suffix_started = diagnostics.trace_stage_events.then(Instant::now);
                        let works = self.prepare_cpu_verify_suffix(result.session, task, suffix, &weights)?;
                        let suffix_us = suffix_started.map(|started| started.elapsed().as_micros()).unwrap_or(0);
                        let fence_started = diagnostics.trace_stage_events.then(Instant::now);
                        let fence_us = fence_started.map(|started| started.elapsed().as_micros()).unwrap_or(0);
                        if profile_boundaries {
                            decode_started[result.session].extend(std::iter::repeat_n(Instant::now(), works.len()));
                        }
                        if !works.is_empty() {
                            if diagnostics.trace_stage_events {
                                let lanes = works.iter().map(|(session, position, _)| format!("{session}@{position}")).collect::<Vec<_>>().join(",");
                                crate::runtime::prefill_scheduler::record_stage_trace(format!(
                                    "[dspark-work-trace] ts_us={} phase=result lane={}@{} job={} anchor={} drafts={} suffix_submit_us={} lanes={lanes} prepare_us={suffix_us} fence_us={fence_us}",
                                    result_us.unwrap_or_default(),
                                    result.session,
                                    anchor_position,
                                    result.id,
                                    result.anchor,
                                    suffix.len(),
                                    crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                                ));
                            }
                            pipeline.submit_many(works)?;
                        }
                        task.pending_verify_rows = inputs.len();
                        task.dspark_verify_hidden.reserve(suffix.len());
                        task.dspark_verify_aux.reserve(suffix.len());
                        task.cached_tokens.extend_from_slice(suffix);
                        task.dspark_cpu_flight = Some(flight);
                    }
                }

                self.poll_stage().map_err(backend_error)?;
                for session in 0..capacity {
                    if closing[session] || ready_backlog.iter().any(|item| item.session == session) {
                        continue;
                    }
                    let Some((stage_id, cached_output_ready)) = slots[session].as_ref().map(|task| (task.stage_id, task.cached_output_ready)) else { continue };
                    if !cached_output_ready && !self.stage_hidden_ready(stage_id) {
                        continue;
                    }
                    progressed = true;
                    let task = slots[session].as_mut().expect("slot 已检查");
                    ready_backlog.push_back(self.take_tail_ready(session, task).map_err(backend_error)?);
                }
                let phases = slots
                    .iter()
                    .enumerate()
                    .map(|(session, task)| {
                        if task.is_none() || closing[session] {
                            RequestPhase::Finished
                        } else if let Some(item) = ready_backlog.iter().find(|item| item.session == session)
                            && task.as_ref().is_some_and(|task| task.dspark_cpu_pending.is_none())
                        {
                            RequestPhase::DecodeReady { position: item.position }
                        } else if task.as_ref().is_some_and(|task| task.prefill_position >= task.tokens.len()) {
                            // 空闲 wave 的第一项直接发射；已有 target 占用流水线时，
                            // ready 只在 completion 边界聚合，禁止用定时器等待凑批。
                            RequestPhase::DecodeInFlight { position: task.as_ref().map_or(0, |task| task.cached_tokens.len()) }
                        } else {
                            RequestPhase::Pending
                        }
                    })
                    .collect::<Vec<_>>();
                let selected_sessions = decode_scheduler.next(&phases).map_or_else(Vec::new, |plan| plan.decode);
                let mut selected = vec![false; capacity];
                for session in selected_sessions {
                    selected[session] = true;
                }
                let mut tail_ready = Vec::new();
                let mut remaining_ready = VecDeque::with_capacity(ready_backlog.len());
                for item in ready_backlog.drain(..) {
                    if selected[item.session] {
                        tail_ready.push(item);
                    } else {
                        remaining_ready.push_back(item);
                    }
                }
                ready_backlog = remaining_ready;
                if !tail_ready.is_empty() {
                    progressed = true;
                    if crate::kernel::rocm::hip::options().kernel_profile && allocation_profile_round < 2 {
                        crate::kernel::rocm::hip::hip_api_stats::report_phase(&format!("glm52-round-{allocation_profile_round}-pre-accept"));
                    }
                    let boundary_started = profile_boundaries.then(Instant::now);
                    let active_decode = slots.iter().enumerate().filter(|(session, task)| !closing[*session] && task.as_ref().is_some_and(|task| task.prefill_position >= task.tokens.len())).count();
                    let maximum_dspark_drafts = crate::runtime::session::effective_dspark_draft_tokens(self.options.dspark_draft_tokens, active_decode);
                    let minimum_dspark_drafts = active_decode.checked_sub(1).map_or(0, |_| decode_pipeline_stage_count.div_ceil(active_decode).saturating_sub(1)).min(maximum_dspark_drafts);
                    let ready_count = tail_ready.len();
                    let decisions = self.accept_tail_batch(&mut slots, tail_ready, minimum_dspark_drafts, maximum_dspark_drafts, profile_completion, on_token, on_tool_call_delta).map_err(backend_error)?;
                    if crate::kernel::rocm::hip::options().kernel_profile && allocation_profile_round < 2 {
                        crate::kernel::rocm::hip::hip_api_stats::report_phase(&format!("glm52-round-{allocation_profile_round}-post-accept"));
                        allocation_profile_round += 1;
                    }
                    if profile_completion {
                        dspark_supply_profile[0] += 1;
                        dspark_supply_profile[1] += active_decode;
                        dspark_supply_profile[2] += minimum_dspark_drafts;
                        dspark_supply_profile[3] += ready_count;
                        for (_, decision) in &decisions {
                            match decision.next.as_ref() {
                                Some(Glm52NextWork::Verify(inputs)) => {
                                    dspark_supply_profile[4] += 1;
                                    dspark_supply_profile[6] += inputs.len();
                                }
                                Some(Glm52NextWork::Decode(_)) => {
                                    dspark_supply_profile[5] += 1;
                                    dspark_supply_profile[6] += 1;
                                }
                                Some(Glm52NextWork::Finish) | None => {}
                            }
                        }
                        if dspark_supply_profile[0] == 32 {
                            eprintln!(
                                "[glm52-dspark-supply] waves=32 active_avg={:.2} minimum_drafts_avg={:.2} ready_avg={:.2} verify={} decode={} rows={}",
                                dspark_supply_profile[1] as f64 / 32.0,
                                dspark_supply_profile[2] as f64 / 32.0,
                                dspark_supply_profile[3] as f64 / 32.0,
                                dspark_supply_profile[4],
                                dspark_supply_profile[5],
                                dspark_supply_profile[6],
                            );
                            dspark_supply_profile.fill(0);
                        }
                    }
                    let accept_micros = boundary_started.map(|started| started.elapsed().as_micros()).unwrap_or(0);
                    if profile_completion {
                        for (session, decision) in &decisions {
                            if decision.next.is_none() {
                                continue;
                            }
                            let Some((started, rows)) = round_started[*session].take() else { continue };
                            let elapsed = started.elapsed().as_micros();
                            if elapsed >= 500_000 {
                                let position = slots[*session].as_ref().map_or(0, |task| task.cached_tokens.len());
                                eprintln!("[glm52-round-slow] session={session} position={position} rows={rows} wall_ms={:.3} transport={}", elapsed as f64 / 1000.0, self.link.connection_diagnostics(),);
                            }
                            round_profile_count += 1;
                            round_profile_rows += rows;
                            round_profile_micros += elapsed;
                            round_profile_max_micros = round_profile_max_micros.max(elapsed);
                            round_profile_lane_count[*session] += 1;
                            round_profile_lane_micros[*session] += elapsed;
                        }
                        if round_profile_count >= 32 {
                            let lanes = round_profile_lane_count
                                .iter()
                                .zip(&round_profile_lane_micros)
                                .enumerate()
                                .filter_map(|(session, (count, micros))| (*count != 0).then(|| format!("{session}:{count}/{:.3}", *micros as f64 / *count as f64 / 1000.0)))
                                .collect::<Vec<_>>()
                                .join(",");
                            eprintln!(
                                "[glm52-round-latency] rounds={} rows={} avg_rows={:.3} avg_ms={:.3} max_ms={:.3} lanes={lanes}",
                                round_profile_count,
                                round_profile_rows,
                                round_profile_rows as f64 / round_profile_count as f64,
                                round_profile_micros as f64 / round_profile_count as f64 / 1000.0,
                                round_profile_max_micros as f64 / 1000.0,
                            );
                            round_profile_count = 0;
                            round_profile_rows = 0;
                            round_profile_micros = 0;
                            round_profile_max_micros = 0;
                            round_profile_lane_count.fill(0);
                            round_profile_lane_micros.fill(0);
                        }
                    }
                    let decision_count = decisions.iter().filter(|(_, decision)| decision.next.is_some()).count();
                    let mut split_verify = Vec::<(Vec<(usize, usize, Glm52StageValue<RocmContext>)>, usize, Vec<u32>)>::new();
                    let mut groups = Vec::<((bool, usize), (Vec<(usize, usize, Glm52StageValue<RocmContext>)>, Vec<(usize, Vec<u32>)>))>::new();
                    for (session, decision) in decisions {
                        let Some(next) = decision.next else { continue };
                        let task = slots[session].as_mut().ok_or_else(|| backend_error(format!("A0 decision 落到空 session={session}")))?;
                        let Some((mut works, inputs)) = self.prepare_next_work(session, task, next, &weights)? else {
                            pipeline.close(session)?;
                            closing[session] = true;
                            continue;
                        };
                        if works.len() > 1 {
                            split_verify.push((works, session, inputs));
                            continue;
                        }
                        let work = works.pop().expect("非 split 工作应恰有一项");
                        let key = (work.2.verify, work.2.hidden.rows);
                        let commit = (work.0, inputs);
                        if let Some((_, group)) = groups.iter_mut().find(|(candidate, _)| *candidate == key) {
                            group.0.push(work);
                            group.1.push(commit);
                        } else {
                            groups.push((key, (vec![work], vec![commit])));
                        }
                    }
                    if profile_completion {
                        let split_works = split_verify.iter().map(|(works, _, _)| works.len()).sum::<usize>();
                        let grouped_sessions = groups.iter().map(|(_, (_, commits))| commits.len()).sum::<usize>();
                        decode_wave_profile[0] = decode_wave_profile[0].saturating_add(1);
                        decode_wave_profile[1] = decode_wave_profile[1].saturating_add(decision_count);
                        decode_wave_profile[2] = decode_wave_profile[2].saturating_add(split_verify.len());
                        decode_wave_profile[3] = decode_wave_profile[3].saturating_add(split_works);
                        decode_wave_profile[4] = decode_wave_profile[4].saturating_add(grouped_sessions);
                        if decode_wave_profile[0] == 32 {
                            eprintln!("[glm52-decode-wave-summary] waves=32 decisions={} split_sessions={} split_works={} grouped_sessions={}", decode_wave_profile[1], decode_wave_profile[2], decode_wave_profile[3], decode_wave_profile[4],);
                            decode_wave_profile = [0; 5];
                        }
                    }
                    if !split_verify.is_empty() {
                        // 同一批 ready session 按 verify 深度交错成一个输入波次：每个
                        // stage 都先看到不同 session 的同深度单行，因此能独立合批并
                        // 共享权重读取；不携带 cohort barrier，慢 session 不会阻塞
                        // 后续 stage。session 内顺序由通用 scheduler 的 completion 门控。
                        let mut pending = Vec::with_capacity(split_verify.len());
                        let mut work_count = 0usize;
                        for (works, session, inputs) in split_verify {
                            slots[session].as_ref().ok_or_else(|| backend_error(format!("MTP split verify session={session} 消失")))?;
                            if let Some(started) = boundary_started {
                                decode_started[session].extend(std::iter::repeat_n(started, works.len()));
                            }
                            work_count = work_count.saturating_add(works.len());
                            pending.push((VecDeque::from(works), session, inputs));
                        }
                        let mut ordered = Vec::with_capacity(work_count);
                        while ordered.len() < work_count {
                            for (works, _, _) in &mut pending {
                                if let Some(work) = works.pop_front() {
                                    ordered.push(work);
                                }
                            }
                        }
                        let submitted_at = profile_completion.then(Instant::now);
                        pipeline.submit_many(ordered)?;
                        for (_, session, inputs) in pending {
                            let task = slots[session].as_mut().ok_or_else(|| backend_error(format!("MTP split verify 提交后 session={session} 消失")))?;
                            round_started[session] = submitted_at.map(|started| (started, inputs.len()));
                            Self::commit_submitted_work(task, &inputs);
                        }
                    }
                    for (_, (works, commits)) in groups {
                        // 与 DeepSeek 一致：ready session 形成一个有序输入波次，但每份
                        // work 保持独立 completion。stage 可从自然 backlog 合批读取
                        // 权重，慢 session 不会形成跨卡 cohort barrier。
                        for (session, _) in &commits {
                            slots[*session].as_ref().ok_or_else(|| backend_error(format!("cohort decode session={} 消失", *session)))?;
                            if let Some(started) = boundary_started {
                                decode_started[*session].push_back(started);
                            }
                        }
                        let submitted_at = profile_completion.then(Instant::now);
                        pipeline.submit_many(works)?;
                        for (session, inputs) in commits {
                            let task = slots[session].as_mut().ok_or_else(|| backend_error(format!("A0 提交后 session={session} 消失")))?;
                            round_started[session] = submitted_at.map(|started| (started, inputs.len()));
                            Self::commit_submitted_work(task, &inputs);
                        }
                    }
                    if let Some(started) = boundary_started {
                        token_profile[0] += accept_micros;
                        token_profile[3] += started.elapsed().as_micros().saturating_sub(accept_micros);
                        token_profile_count += decision_count;
                        if token_profile_count >= 32 {
                            eprintln!("[glm52-boundary-a-token] sessions={token_profile_count} a0_phases_ms={:.3} submit_ms={:.3}", token_profile[0] as f64 / 1000.0, token_profile[3] as f64 / 1000.0);
                            token_profile = [0; 4];
                            token_profile_count = 0;
                        }
                    }
                }
                let active = slots.iter().filter(|task| task.is_some()).count();
                for request in intake(capacity.saturating_sub(active.saturating_add(pending.len()))) {
                    progressed = true;
                    let request_id = request.request_id.clone();
                    match self.prepare_batch_task(request) {
                        Ok(task) => pending.push_back(task),
                        Err(message) => on_result(NodeBatchResult { request_id, result: Err(message) }),
                    }
                }
                while let Some(session) = slots.iter().enumerate().find_map(|(session, slot)| (slot.is_none() && (decode_scheduler.initial_batch_released() || session >= initial_count)).then_some(session)) {
                    // 同一内容 hash 只允许一个 writer。断线重试先等旧 writer 把已完成
                    // 前缀发布为 cache，再从该水位继续，不能在另一个空 slot 从零重算。
                    let Some(pending_index) = pending.iter_mut().position(|candidate| {
                        if !candidate.swap_prefetch.poll_ready() {
                            return false;
                        }
                        let cache_id = candidate.input.request.get("cache_id").and_then(Value::as_str);
                        cache_id.is_none_or(|cache_id| !slots.iter().filter_map(Option::as_ref).any(|task| task.request.get("cache_id").and_then(Value::as_str) == Some(cache_id)))
                    }) else {
                        break;
                    };
                    // 与初始批一致，准入必须让统一 open 路径先驱逐 terminal cache。
                    progressed = true;
                    let pending_task = pending.remove(pending_index).expect("pending index 已检查");
                    let request_id = pending_task.input.request_id.clone();
                    let mut task = match self.open_batch_task(pending_task) {
                        Ok(task) => task,
                        Err(message) => {
                            on_result(NodeBatchResult { request_id, result: Err(message) });
                            continue;
                        }
                    };
                    if let Err(message) = self.finish_batch_open(&mut task) {
                        let _ = self.link.send_delete(task.stage_id);
                        on_result(NodeBatchResult { request_id, result: Err(message) });
                        continue;
                    }
                    if let Err(message) = self.link.send_stream_assign(task.stage_id, session) {
                        let _ = self.link.send_delete(task.stage_id);
                        on_result(NodeBatchResult { request_id, result: Err(message) });
                        continue;
                    }
                    if task.cached_output_ready {
                        if let Err(message) = self.link.send_prefill_done(task.stage_id, task.cached_tokens.len()) {
                            let _ = self.link.send_delete(task.stage_id);
                            on_result(NodeBatchResult { request_id, result: Err(message) });
                            continue;
                        }
                    }
                    let states = std::mem::take(&mut task.states);
                    let stage_id = task.stage_id;
                    // stream assign 已生效，open 失败必须 send_delete 回收下游 session，
                    // 否则下游泄漏且该请求被静默丢弃；与初始 open 路径同样通知客户端。
                    if let Err(message) = pipeline.open(session, states) {
                        let _ = self.link.send_delete(stage_id);
                        on_result(NodeBatchResult { request_id, result: Err(format!("打开 continuous pipeline session={session}: {message:?}")) });
                        continue;
                    }
                    request_sessions.insert(stage_id, session);
                    submitted_prefill[session] = task.prefill_position;
                    slots[session] = Some(task);
                    decode_scheduler.extend_initial_batch(session + 1);
                }
                let decode_active = slots.iter().flatten().any(|task| task.prefill_position >= task.tokens.len());
                let (mut prefill_target, prefill_work_window, prefill_chunk_limit) = prefill_admission.limits(
                    pipeline.stage_flow_snapshot(),
                    pipeline.pipeline_work_window(),
                    pipeline.stage_count(),
                    decode_active,
                    self.options.scheduling.decode_priority_prefill_chunk_size,
                    self.options.scheduling.decode_priority_prefill_chunk_ceiling,
                );
                let short_only = prefill_target == 0
                    && prefill_admission.short_request_ready()
                    && Self::has_finishable_short_prefill(&slots, &submitted_prefill, &prefill_in_flight_by_session, self.options.scheduling.decode_priority_prefill_chunk_ceiling);
                prefill_target += usize::from(short_only);
                Self::fill_prefill_window(
                    pipeline,
                    &slots,
                    prefill_target,
                    prefill_work_window,
                    prefill_chunk_limit,
                    self.options.scheduling.decode_priority_prefill_chunk_ceiling,
                    short_only,
                    &mut submitted_prefill,
                    &mut prefill_in_flight,
                    &mut prefill_in_flight_by_session,
                    &mut prefill_admission,
                    &self.contexts[0],
                    &weights,
                    &cfg,
                    self.embedding_table.as_ref(),
                )?;

                if slots.iter().all(Option::is_none) && pending.is_empty() {
                    let incoming = intake(capacity);
                    if incoming.is_empty() {
                        self.link.send_continuous_stream_end(stream_control_id).map_err(backend_error)?;
                        self.wait_ready(stream_control_id, 0).map_err(backend_error)?;
                        break;
                    }
                    for request in incoming {
                        let request_id = request.request_id.clone();
                        match self.prepare_batch_task(request) {
                            Ok(task) => pending.push_back(task),
                            Err(message) => on_result(NodeBatchResult { request_id, result: Err(message) }),
                        }
                    }
                }
                if progressed {
                    if let Some((started, reason)) = coordinator_idle_trace.take() {
                        let duration_us = started.elapsed().as_micros();
                        if duration_us >= 20_000 {
                            let end_us = crate::runtime::prefill_scheduler::stage_trace_timestamp_us();
                            crate::runtime::prefill_scheduler::record_stage_trace(format!("[coordinator-block-trace] ts_us={} phase=interval reason={reason} duration_us={duration_us}", end_us.saturating_sub(duration_us)));
                        }
                    }
                } else {
                    if diagnostics.trace_stage_events && coordinator_idle_trace.is_none() {
                        let active = slots.iter().flatten().count();
                        let cpu_pending = slots.iter().flatten().filter(|task| task.dspark_cpu_pending.is_some()).count();
                        let target_pending = slots.iter().flatten().filter(|task| task.pending_verify_rows != 0).count();
                        let reason = format!(
                            "no_progress active={active} cpu_pending={cpu_pending} target_pending={target_pending} ready_backlog={} prefill_in_flight={prefill_in_flight} closing={} intake_pending={}",
                            ready_backlog.len(),
                            closing.iter().filter(|&&closing| closing).count(),
                            pending.len(),
                        );
                        coordinator_idle_trace = Some((Instant::now(), reason));
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            }
            Ok(())
        });

        if crate::kernel::rocm::hip::options().kernel_profile || profile_completion {
            crate::kernel::rocm::hip::report_device_profiles(0, slots.iter().flatten().count());
        }

        if let Err(error) = run {
            let failed = slots.iter_mut().filter_map(Option::take).map(|task| (task.request_id, task.stage_id)).collect::<Vec<_>>();
            for &(_, stage_id) in &failed {
                let _ = self.link.send_delete(stage_id);
            }
            let cleanup = self.link.send_continuous_stream_end(stream_control_id).and_then(|()| self.wait_ready(stream_control_id, 0));
            let mut message = format!("GLM-5.2 continuous stream: {error:?}");
            if let Err(cleanup) = cleanup {
                message.push_str(&format!("; 清理 stream 失败: {cleanup}"));
            }
            eprintln!("[glm52-continuous-error] {message}");
            for (request_id, stage_id) in failed {
                self.pending_messages.remove(&stage_id);
                on_result(NodeBatchResult { request_id, result: Err(message.clone()) });
            }
        }
        if let Ok(mut runtime) = self.runtime.lock() {
            runtime.new_prefill = 0;
            runtime.append_prefill = 0;
            runtime.decode = 0;
            runtime.dspark_draft_tokens = 0;
        }
        on_runtime_changed();
        Vec::new()
    }

    fn finish_batch_open(&mut self, task: &mut Glm52BatchTask) -> Result<(), String> {
        let ready = task.open_ready_tokens.take().expect("新打开的 task 必须等待 Ready");
        self.wait_ready(task.stage_id, ready)?;
        // B 的 SSD restore 完成并建立 active session 后再下发依赖该 session 的消息。
        // 这样一批 Open 可以先全部到达 B，host 读盘才能并行。
        // MTP context 由本机 A0 持有,不再向尾端下发。
        Ok(())
    }

    fn load_swap_snapshot(swap: &Glm52SwapStore, resume: &TerminalResume, requested_cache_id: Option<&str>, cache_namespace: Option<&str>, tokens: &[u32]) -> Glm52SwapPrefetchResult {
        if let TerminalResume::Match { cache_id, assistant } = resume
            && let Some(snapshot) = swap.get(cache_id)?
            && snapshot.cache_namespace.as_deref() == cache_namespace
            && snapshot.pending_tokens.is_some()
        {
            return Ok(Some((snapshot, Some(*assistant))));
        }
        if let Some(cache_id) = requested_cache_id
            && let Some(snapshot) = swap.get(cache_id)?
            && !snapshot.tokens.is_empty()
            && tokens.starts_with(&snapshot.tokens)
        {
            return Ok(Some((snapshot, None)));
        }
        let Some(cache_id) = swap.longest_prefix_cache_id(cache_namespace, tokens)? else { return Ok(None) };
        Ok(swap.get(&cache_id)?.map(|snapshot| (snapshot, None)))
    }

    fn start_swap_prefetch(&self, input: &NodeBatchRequest, tokens: &[u32]) -> Result<Glm52SwapPrefetch, String> {
        let Some(swap) = self.swap.as_ref().map(Arc::clone) else { return Ok(Glm52SwapPrefetch::Disabled) };
        let resume = request_terminal_resume(&input.request)?;
        let requested_cache_id = input.request.get("cache_id").and_then(Value::as_str).map(str::to_owned);
        let cache_namespace = input.request.get("_zllm_cache_namespace").and_then(Value::as_str).map(str::to_owned);
        let tokens = tokens.to_vec();
        let request_id = input.request_id.clone();
        let thread_name = format!("glm52-swap-{}", request_id.chars().take(12).collect::<String>());
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let started = Instant::now();
                let result = Self::load_swap_snapshot(&swap, &resume, requested_cache_id.as_deref(), cache_namespace.as_deref(), &tokens);
                let matched = result.as_ref().ok().and_then(|snapshot| snapshot.as_ref()).map(|(snapshot, _)| snapshot.cache_id.as_str()).unwrap_or("miss");
                eprintln!("[glm52-swap-prefetch] side=head request_id={request_id} cache_id={matched} read_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                let _ = sender.send(result);
            })
            .map_err(|error| format!("启动 GLM-5.2 SSD prefetch: {error}"))?;
        Ok(Glm52SwapPrefetch::Loading(receiver))
    }

    fn prepare_batch_task(&self, input: NodeBatchRequest) -> Result<Glm52PendingTask, String> {
        let reasoning_effort = request_reasoning_effort(&input.request, self.options.reasoning_effort)?;
        let prompt = chat_prompt_glm52_with_template(&input.request, self.options.diagnostics.official_chat_template, reasoning_effort)?;
        let tokens = self.tokenizer.tokenize(prompt.as_bytes());
        if tokens.is_empty() {
            return Err("GLM-5.2 prompt 不能为空".to_owned());
        }
        if tokens.len() > self.max_seq_len {
            return Err(format!("GLM-5.2 prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.max_seq_len));
        }
        let max_tokens = crate::runtime::session::requested_completion_tokens(&input.request);
        if max_tokens == 0 || tokens.len() == self.max_seq_len {
            return Err("GLM-5.2 max_tokens 必须大于 0，且 prompt 后必须保留生成空间".to_owned());
        }
        let reserved_tokens = self.request_reservation_tokens(tokens.len(), max_tokens);
        // 未显式指定 temperature 时保持既有 greedy；显式参数必须透传到 tail。
        let temperature = input.request.get("temperature").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let top_p = input.request.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        // 显式 seed 让 full/FR-Spec 等配置可以逐 token 复现；未提供时保持
        // 原有 request_id 隔离，避免并发请求意外共用随机序列。
        let seed = input.request.get("seed").and_then(Value::as_u64).unwrap_or_else(|| {
            let hash = blake3::hash(input.request_id.as_bytes());
            u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
        });
        let sampling = SamplingConfig { temperature, top_p, seed }.validate()?;
        let thinking_token_budget = if self.thinking_end_token.is_some() { request_thinking_token_budget(&input.request, self.options.thinking_token_budget)? } else { None };
        let swap_prefetch = self.start_swap_prefetch(&input, &tokens)?;
        Ok(Glm52PendingTask { input, tokens, max_tokens, reserved_tokens, sampling, thinking_end_token: self.thinking_end_token, thinking_token_budget, swap_prefetch })
    }

    fn open_batch_task(&mut self, pending: Glm52PendingTask) -> Result<Glm52BatchTask, String> {
        let Glm52PendingTask { input, mut tokens, max_tokens, reserved_tokens: _, sampling, thinking_end_token, thinking_token_budget, swap_prefetch } = pending;
        let requested_cache_id = input.request.get("cache_id").and_then(Value::as_str).map(str::to_owned);
        let cache_namespace = input.request.get("_zllm_cache_namespace").and_then(Value::as_str).map(str::to_owned);
        let stage_id = RequestId::from_cache_id(&input.request_id);
        let resume = request_terminal_resume(&input.request)?;
        let mut resident = None;
        match &resume {
            TerminalResume::Match { cache_id, assistant } => {
                if let Some((cached, state)) = self.terminal_states.take(cache_id) {
                    let boundary_ready = state.pending_tokens.is_some();
                    if state.cache_namespace == cache_namespace && (boundary_ready || tokens.starts_with(&cached)) {
                        resident = Some((cache_id.clone(), cached, state, boundary_ready.then_some(*assistant)));
                    } else {
                        self.terminal_states.insert(cache_id.clone(), cached, state);
                    }
                }
            }
            TerminalResume::Mismatch { requested, expected } => {
                eprintln!("[glm52-cache-mismatch] request_cache_id={requested} expected={expected}");
            }
            TerminalResume::None => {}
        }
        if resident.is_none() {
            resident = requested_cache_id
                .as_deref()
                .filter(|cache_id| self.terminal_states.cached_tokens(cache_id).is_some_and(|cached| tokens.starts_with(cached)))
                .and_then(|cache_id| self.terminal_states.take(cache_id).map(|(cached, state)| (cache_id.to_owned(), cached, state, None)))
                .or_else(|| self.terminal_states.take_longest_prefix(&tokens, |state| state.cache_namespace == cache_namespace).map(|(cache_id, cached, state)| (cache_id, cached, state, None)));
        }
        let mut matched_cache_id = resident.as_ref().map(|(cache_id, _, _, _)| cache_id.clone());
        let mut reusable = None;
        while self.kv_budget.used().saturating_add(self.resident_kv_cost()) > self.kv_budget.capacity() {
            let Some(states) = self.evict_oldest_terminal()? else { break };
            if reusable.is_none() {
                reusable = Some(states);
            }
        }
        let mut cache_hit = false;
        let head_mtp_enabled = self.options.mtp;
        let dspark_enabled = self.dspark_runtime.is_some();
        let (mut states, last_hidden, cached_tokens, prefill_position, mut mtp, mut dspark_aux_history, mut dspark_aux_history_start, mut dspark_target_cache) = if let Some((matched_id, cached_tokens, state, resume_assistant)) = resident {
            if let Some(assistant) = resume_assistant {
                let mut resumed = cached_tokens.clone();
                resumed.extend(state.pending_tokens.as_ref().expect("boundary resume 已检查 pending token"));
                // 边界后没有新消息时这是"继续上一轮"请求:不构造 suffix,直接从终点
                // 状态续写。此前抛 "resume 边界之后没有新消息" 会以 stream error 杀掉
                // Codex 会话,而 server 侧预检覆盖不到 previous_response_id miss 降级
                // 后自动附挂 cache_id 的路径。
                let suffix = match chat_prompt_suffix_glm52(&input.request, assistant, self.options.diagnostics.official_chat_template) {
                    Ok(suffix) => suffix,
                    Err(_error) if !boundary_has_followup(&input.request, assistant) => {
                        eprintln!("[glm52-resume-continue] request_id={} cache_id={matched_id} 边界后无新消息,从终点状态续写", input.request_id);
                        String::new()
                    }
                    Err(error) => {
                        self.terminal_states.insert(matched_id.clone(), cached_tokens, state);
                        return Err(error);
                    }
                };
                resumed.extend(self.tokenizer.tokenize(suffix.as_bytes()));
                if resumed.len() >= self.max_seq_len {
                    self.terminal_states.insert(matched_id.clone(), cached_tokens, state);
                    return Err(format!("GLM-5.2 resume 后 prompt {} tokens 超过 max_seq_len {}", resumed.len(), self.max_seq_len));
                }
                tokens = resumed;
            }
            let Glm52HeadState { states, last_hidden, pending_tokens: _, mtp, dspark_aux_history, dspark_aux_history_start, dspark_target_cache, .. } = state;
            let dspark_valid = self.dspark_cache_valid(dspark_aux_history.as_ref(), dspark_aux_history_start, &dspark_target_cache, cached_tokens.len());
            if head_mtp_enabled && mtp.is_none() || dspark_enabled && !dspark_valid {
                matched_cache_id = None;
                eprintln!("[glm52-cache-reject] cache_id={matched_id} resident state 缺少 {}，重新 prefill", if head_mtp_enabled && mtp.is_none() { "MTP" } else { "完整 DSpark aux/target cache" });
                (states, None, Vec::new(), 0, None, None, 0, DsparkTargetCache::new())
            } else {
                let position = cached_tokens.len();
                cache_hit = true;
                eprintln!(
                    "[glm52-cache-hit] cache_id={matched_id} request_cache_id={} source=resident tokens={position} suffix_tokens={} dspark_aux_rows={} dspark_target_layers={}",
                    requested_cache_id.as_deref().unwrap_or_default(),
                    tokens.len().saturating_sub(position),
                    dspark_aux_history.as_ref().map_or(0, |hidden| hidden.rows),
                    dspark_target_cache.warmed_layers()
                );
                (states, Some(last_hidden), cached_tokens, position, mtp, dspark_aux_history, dspark_aux_history_start, dspark_target_cache)
            }
        } else {
            let swapped = match swap_prefetch.finish() {
                Some(result) => result?,
                None => self.swap.as_deref().map(|swap| Self::load_swap_snapshot(swap, &resume, requested_cache_id.as_deref(), cache_namespace.as_deref(), &tokens)).transpose()?.flatten(),
            };
            match swapped {
                Some((snapshot, resume_assistant)) => {
                    let matched_id = snapshot.cache_id.clone();
                    matched_cache_id = Some(matched_id.clone());
                    let dspark_valid = match (&self.dspark_runtime, &snapshot.dspark_aux, &snapshot.dspark_target) {
                        (Some(runtime), Some(aux), Some(target)) => {
                            let columns = runtime.target_columns();
                            aux.columns == self.cfg.hidden_size
                                && aux.start_position.checked_add(aux.rows) == Some(snapshot.tokens.len())
                                && target.layers.len() == runtime.target_layer_count()
                                && target.layers.iter().all(|layer| layer.start_position == aux.start_position && layer.key.rows == aux.rows && layer.value.rows == aux.rows && layer.key.columns == columns && layer.value.columns == columns)
                        }
                        (None, None, None) => true,
                        _ => false,
                    };
                    if head_mtp_enabled && snapshot.mtp.is_none() || dspark_enabled && !dspark_valid {
                        matched_cache_id = None;
                        eprintln!("[glm52-cache-reject] cache_id={matched_id} SSD state 缺少 {}，重新 prefill", if head_mtp_enabled && snapshot.mtp.is_none() { "MTP" } else { "完整 DSpark aux/target cache" });
                        let states = match reusable {
                            Some(states) => states,
                            None => self.build_fresh_states()?,
                        };
                        (states, None, Vec::new(), 0, None, None, 0, DsparkTargetCache::new())
                    } else {
                        if let Some(assistant) = resume_assistant {
                            let mut resumed = snapshot.tokens.clone();
                            resumed.extend(snapshot.pending_tokens.as_ref().expect("boundary resume 已检查 pending token"));
                            let suffix = match chat_prompt_suffix_glm52(&input.request, assistant, self.options.diagnostics.official_chat_template) {
                                Ok(suffix) => suffix,
                                Err(_error) if !boundary_has_followup(&input.request, assistant) => {
                                    eprintln!("[glm52-resume-continue] request_id={} cache_id={matched_id} 边界后无新消息,从终点状态续写(SSD)", input.request_id);
                                    String::new()
                                }
                                Err(error) => return Err(error),
                            };
                            resumed.extend(self.tokenizer.tokenize(suffix.as_bytes()));
                            tokens = resumed;
                        }
                        let mut states = match reusable.take() {
                            Some(states) => states,
                            None => self.build_fresh_states()?,
                        };
                        let restored_rows = self.request_reservation_tokens(tokens.len(), max_tokens);
                        upload_glm52_session(&mut states, &snapshot.stages, &self.cfg, self.max_seq_len, restored_rows)?;
                        let hidden = self.output_context().tensor_from_bf16_bits(snapshot.last_hidden, 1, self.cfg.hidden_size).map_err(|error| format!("恢复 head terminal hidden: {error:?}"))?;
                        let position = snapshot.tokens.len();
                        let mtp = snapshot.mtp.map(|mtp| RocmMtpSession::restore(mtp, &self.output_context(), &self.cfg, self.max_seq_len, restored_rows)).transpose()?;
                        let (dspark_aux_history, dspark_aux_history_start) = match snapshot.dspark_aux {
                            Some(aux) => {
                                let start = aux.start_position;
                                let hidden = self
                                    .output_context()
                                    .tensor_from_bf16_bits(aux.values, aux.rows, aux.columns)
                                    .and_then(|hidden| self.output_context().tensor_as_f32(hidden))
                                    .map_err(|error| format!("恢复 DSpark aux history: {error:?}"))?;
                                (Some(hidden), start)
                            }
                            None => (None, 0),
                        };
                        let dspark_target_cache = match snapshot.dspark_target {
                            Some(target) => {
                                let layers = target
                                    .layers
                                    .into_iter()
                                    .map(|layer| {
                                        let key = self.output_context().tensor_from_f32(layer.key.values, layer.key.rows, layer.key.columns).map_err(|error| format!("恢复 DSpark target key: {error}"))?;
                                        let value = self.output_context().tensor_from_f32(layer.value.values, layer.value.rows, layer.value.columns).map_err(|error| format!("恢复 DSpark target value: {error}"))?;
                                        Ok::<_, String>(DsparkTargetLayerSnapshot { start_position: layer.start_position, key, value })
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                DsparkTargetCache::from_snapshot(DsparkTargetCacheSnapshot { layers })
                            }
                            None => DsparkTargetCache::new(),
                        };
                        if let Some(mtp) = &mtp {
                            eprintln!("[glm52-mtp-swap-in] cache_id={matched_id} position={}", mtp.position);
                        }
                        cache_hit = true;
                        eprintln!(
                            "[glm52-cache-hit] cache_id={matched_id} request_cache_id={} source=ssd tokens={position} suffix_tokens={} dspark_aux_rows={} dspark_target_layers={}",
                            requested_cache_id.as_deref().unwrap_or_default(),
                            tokens.len().saturating_sub(position),
                            dspark_aux_history.as_ref().map_or(0, |hidden| hidden.rows),
                            dspark_target_cache.warmed_layers()
                        );
                        (states, Some(hidden), snapshot.tokens, position, mtp, dspark_aux_history, dspark_aux_history_start, dspark_target_cache)
                    }
                }
                _ => {
                    let states = match reusable {
                        Some(states) => states,
                        None => self.build_fresh_states()?,
                    };
                    (states, None, Vec::new(), 0, None, None, 0, DsparkTargetCache::new())
                }
            }
        };
        if !dspark_enabled {
            // DSpark 是 target KV checkpoint 之上的可选加速状态。关闭 drafter
            // 时仍可复用同一份 target KV，但不应把旧 aux/cache 带入本轮终态。
            dspark_aux_history = None;
            dspark_aux_history_start = 0;
            dspark_target_cache = DsparkTargetCache::new();
        }
        if tokens.len() >= self.max_seq_len {
            return Err(format!("GLM-5.2 resume 后 prompt {} tokens 超过 max_seq_len {}", tokens.len(), self.max_seq_len));
        }
        let max_decode = max_tokens.min(self.max_seq_len - tokens.len());
        let reserved_tokens = self.request_reservation_tokens(tokens.len(), max_tokens);
        while self.kv_budget.used().saturating_add(self.resident_kv_cost()).saturating_add(reserved_tokens) > self.kv_budget.capacity() {
            if self.evict_oldest_terminal()?.is_none() {
                return Err(format!("GLM-5.2 KV 预算不足: request={reserved_tokens} available={}", self.kv_budget.available()));
            }
        }
        let kv_reservation = self.kv_budget.try_reserve(reserved_tokens).ok_or_else(|| format!("GLM-5.2 KV 预算竞争: request={reserved_tokens} available={}", self.kv_budget.available()))?;
        if !cache_hit {
            for state in &mut states {
                state.reset_session(&self.cfg, self.max_seq_len).map_err(|error| format!("重置 batch stage session: {error:?}"))?;
            }
        }
        let tools_enabled = input.request.get("tool_choice").and_then(Value::as_str) != Some("none") && input.request.get("tools").and_then(Value::as_array).is_some_and(|tools| !tools.is_empty());
        let repeat_loop_breaker = match input.request.get("repeat_loop_breaker") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(enabled)) => *enabled,
            Some(_) => return Err("repeat_loop_breaker 必须是 bool".to_owned()),
        };
        let protected = self.cfg.eos_token_ids.iter().copied().chain(thinking_end_token);
        // 只接受 tokenizer 能编码成单个 token 的推理重启词，避免把空格/换行等
        // 公共 token 错当成 marker。达到阈值后只禁 marker，不改写或格式化输出。
        let restart_markers = [
            "Wait",
            " Wait",
            "wait",
            " wait",
            "Actually",
            " Actually",
            "actually",
            " actually",
            "reconsider",
            " reconsider",
            "Reconsider",
            " Reconsider",
            // tokenizer 会把 “Let's” 拆成 Let + 's，因此用首 token 统计该重启句式。
            "Let",
            " Let",
            "let",
            " let",
            "Maybe",
            " Maybe",
            "maybe",
            " maybe",
            "Perhaps",
            " Perhaps",
            "perhaps",
            " perhaps",
            // 真实 MMLU-Pro 长尾会用 “What if ...” 连续改写题设；累计到阈值后
            // 禁止继续以 What 重启，但不影响前面的正常假设检验。
            "What",
            " What",
            "what",
            " what",
            "Hmm",
            " Hmm",
            "hmm",
            " hmm",
        ]
        .into_iter()
        .filter_map(|text| {
            let tokens = self.tokenizer.tokenize(text.as_bytes());
            (tokens.len() == 1).then_some(tokens[0])
        });
        let semantic_restart_sequences =
            ["If there", " If there", "Could it", " Could it", "Could this", " Could this", "Is there any chance", " Is there any chance", "Suppose instead", " Suppose instead", "Alternatively", " Alternatively", "What if", " What if"]
                .into_iter()
                .map(|text| self.tokenizer.tokenize(text.as_bytes()));
        let token_fence = GenerationGuard::new(Glm52ToolFence::new(tools_enabled), repeat_loop_breaker, protected)
            .with_restart_markers(restart_markers)
            .with_semantic_restart_sequences(semantic_restart_sequences)
            .with_token_signatures(self.token_signatures.clone())
            .with_segment_end_tokens(thinking_end_token);
        let mtp_enabled = head_mtp_enabled;
        if mtp_enabled && mtp.is_none() && prefill_position == 0 {
            mtp = Some(RocmMtpSession::fresh(&self.cfg, self.max_seq_len)?);
        }
        if let Some(mtp) = mtp.as_mut() {
            mtp.begin_request(prefill_position, tokens.clone(), max_decode, self.options.mtp_draft_tokens)?;
        } else if mtp_enabled && prefill_position > 0 {
            eprintln!("[glm52-mtp-fallback] request_id={} SSD cache 没有 A0 MTP resident state，本轮退回普通 decode", input.request_id);
        }
        let sampling_state = SamplingState::new(sampling)?;
        let batch_guard = BatchTokenGuard::new(&self.runtime, tokens.len().saturating_sub(prefill_position));
        let prefill_policy = AdaptiveChunkPolicy {
            initial_chunk_size: self.pipeline_chunk_size,
            append_chunk_size: self.options.scheduling.append_prefill_chunk_size,
            long_context_threshold_tokens: self.options.scheduling.long_prefill_threshold_tokens,
            long_context_chunk_size: self.options.scheduling.long_prefill_chunk_size,
        };
        let cache_request_id = matched_cache_id.as_deref().filter(|_| cache_hit).map(RequestId::from_cache_id);
        self.send_open_with_reconnect(stage_id, cache_request_id, cached_tokens.len(), reserved_tokens.min(self.max_seq_len), cache_hit, sampling).map_err(|error| format!("命令下游打开 cache: {error}"))?;
        let open_ready_tokens = Some(cached_tokens.len());
        let prompt_last_hidden = (prefill_position == tokens.len()).then(|| last_hidden.clone()).flatten();
        let prompt_dspark_aux_history = (prefill_position == tokens.len()).then(|| dspark_aux_history.clone()).flatten();
        let prompt_dspark_aux_history_start = dspark_aux_history_start;
        let cached_output_ready = prefill_position == tokens.len();
        let tool_stream = GlmToolCallStream::new(&input.request, &input.request_id);
        Ok(Glm52BatchTask {
            request_id: input.request_id,
            stage_id,
            tool_stream,
            request: input.request,
            cancellation: input.cancellation,
            _batch_guard: batch_guard,
            tokens,
            max_decode,
            kv_reservation,
            states,
            last_hidden,
            prompt_last_hidden,
            cached_tokens,
            prefill_position,
            tail_prefill_position: prefill_position,
            prefill_suffix_start: prefill_position,
            prefill_policy,
            open_ready_tokens,
            response_text: String::new(),
            utf8: Utf8StreamDecoder::default(),
            think_filter: ThinkTagFilter::default(),
            completion_tokens: 0,
            pending_token: None,
            thinking_tokens: 0,
            thinking_end_token,
            thinking_token_budget,
            finish_reason: "length".to_owned(),
            sampling: sampling_state,
            token_fence,
            mtp,
            cached_output_ready,
            pending_verify_rows: 0,
            mtp_verify_rows: Vec::new(),
            dspark_aux_history,
            dspark_aux_history_start,
            prompt_dspark_aux_history,
            prompt_dspark_aux_history_start,
            dspark_target_cache,
            dspark_cpu_target_cache: DsparkTargetCache::new(),
            dspark_cpu_pending: None,
            dspark_cpu_anchor_in_flight: false,
            dspark_cpu_window: 1,
            dspark_cpu_flight: None,
            dspark_verify_inputs: Vec::new(),
            dspark_verify_hidden: Vec::new(),
            dspark_verify_aux: Vec::new(),
            dspark_target_rounds: 0,
            dspark_verify_rounds: 0,
            dspark_verified_drafts: 0,
            dspark_accepted_drafts: 0,
            // 容量取 max(dspark, mtp)：统计字段由 DSpark 与 MTP(nextn) 共用。
            dspark_verified_by_depth: vec![0; self.options.dspark_draft_tokens.max(self.options.mtp_draft_tokens)],
            dspark_accepted_by_depth: vec![0; self.options.dspark_draft_tokens.max(self.options.mtp_draft_tokens)],
        })
    }

    fn take_tail_ready(&mut self, session: usize, task: &mut Glm52BatchTask) -> Result<Glm52TailReady, String> {
        let output_context = self.output_context();
        if task.cached_output_ready {
            task.cached_output_ready = false;
            return Ok(Glm52TailReady {
                session,
                position: task.cached_tokens.len().saturating_sub(1),
                hidden: task.last_hidden.clone().ok_or("命中完整 cache 但缺少 A0 terminal hidden")?,
                aux_hidden: None,
                aux_taps: 0,
                decode: false,
                verify: false,
                cached: true,
                sampled_token: None,
                speculative: None,
            });
        }
        match self.recv_stage(task.stage_id)? {
            StageMessage::Prefill { position, rows, cols, values, selection, aux_values, aux_taps } => {
                if rows == 0 || cols != self.cfg.hidden_size || !selection.is_empty() {
                    return Err(format!("B→A prefill hidden 非法: position={position} shape=[{rows},{cols}] selection={}", selection.len()));
                }
                let hidden = output_context.tensor_from_bf16_bits_ordered(values, rows, cols).map_err(|error| format!("上传 A0 prefill hidden: {error:?}"))?;
                let aux_hidden = (!aux_values.is_empty()).then(|| output_context.tensor_from_bf16_bits_ordered(aux_values, rows, cols)).transpose().map_err(|error| format!("上传 A0 prefill aux hidden: {error:?}"))?;
                Ok(Glm52TailReady { session, position, hidden, aux_hidden, aux_taps, decode: false, verify: false, cached: false, sampled_token: None, speculative: None })
            }
            StageMessage::Decode { position, cols, values, selection, aux_values, aux_taps, .. } => {
                if self.options.diagnostics.trace_stage_events {
                    crate::runtime::prefill_scheduler::record_stage_trace(format!(
                        "[stage-boundary-trace] ts_us={} phase=recv direction=B_to_A request_id={} lane={session}@{position} kind=Decode rows=1",
                        crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                        task.stage_id
                    ));
                }
                if cols != self.cfg.hidden_size || !selection.is_empty() {
                    return Err(format!("B→A decode hidden 非法: position={position} cols={cols} selection={}", selection.len()));
                }
                let hidden = output_context.tensor_from_bf16_bits_ordered(values, 1, cols).map_err(|error| format!("上传 A0 decode hidden: {error:?}"))?;
                let aux_hidden = (!aux_values.is_empty()).then(|| output_context.tensor_from_bf16_bits_ordered(aux_values, 1, cols)).transpose().map_err(|error| format!("上传 A0 decode aux hidden: {error:?}"))?;
                Ok(Glm52TailReady { session, position, hidden, aux_hidden, aux_taps, decode: true, verify: false, cached: false, sampled_token: None, speculative: None })
            }
            StageMessage::Verify { position, rows, cols, values, selection, aux_values, aux_taps, .. } => {
                if self.options.diagnostics.trace_stage_events {
                    crate::runtime::prefill_scheduler::record_stage_trace(format!(
                        "[stage-boundary-trace] ts_us={} phase=recv direction=B_to_A request_id={} lane={session}@{position} kind=Verify rows={rows}",
                        crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                        task.stage_id
                    ));
                }
                if rows == 0 || cols != self.cfg.hidden_size || !selection.is_empty() {
                    return Err(format!("B→A verify hidden 非法: position={position} shape=[{rows},{cols}] selection={}", selection.len()));
                }
                let hidden = output_context.tensor_from_bf16_bits_ordered(values, rows, cols).map_err(|error| format!("上传 A0 verify hidden: {error:?}"))?;
                let aux_hidden = (!aux_values.is_empty()).then(|| output_context.tensor_from_bf16_bits_ordered(aux_values, rows, cols)).transpose().map_err(|error| format!("上传 A0 verify aux hidden: {error:?}"))?;
                Ok(Glm52TailReady { session, position, hidden, aux_hidden, aux_taps, decode: false, verify: true, cached: false, sampled_token: None, speculative: None })
            }
            StageMessage::Sampled { position, token, eos, .. } => Ok(Glm52TailReady {
                session,
                position,
                hidden: RocmTensor { data: Vec::new(), rows: 1, cols: 0, dtype: crate::backend::rocm::RocmTensorDType::F32, layout: crate::backend::rocm::RocmTensorLayout::RowMajor, device: None },
                aux_hidden: None,
                aux_taps: 0,
                decode: true,
                verify: false,
                cached: false,
                sampled_token: Some((token, eos)),
                speculative: None,
            }),
            StageMessage::Speculative { tokens, retained_rows, drafts, eos } => Ok(Glm52TailReady {
                session,
                position: task.cached_tokens.len().saturating_sub(task.pending_verify_rows),
                hidden: RocmTensor { data: Vec::new(), rows: 1, cols: 0, dtype: crate::backend::rocm::RocmTensorDType::F32, layout: crate::backend::rocm::RocmTensorLayout::RowMajor, device: None },
                aux_hidden: None,
                aux_taps: 0,
                decode: true,
                verify: task.pending_verify_rows != 0,
                cached: false,
                sampled_token: None,
                speculative: Some((tokens, retained_rows, drafts, eos)),
            }),
            _ => Err("等待 B→A hidden 时收到非法 frame".to_owned()),
        }
    }

    /// B→A hidden 不进入 A0 stage scheduler，而是直接由输出头/MTP 消费；
    /// 因此在真实消费 phase 才把 deferred H2D 排到 A0 stream。
    fn enqueue_tail_ready(output_context: &RocmContext, item: &Glm52TailReady) -> Result<(), String> {
        output_context.activate()?;
        if let Some(buffer) = item.hidden.device.as_ref() {
            buffer.enqueue_deferred_upload()?;
        }
        if let Some(buffer) = item.aux_hidden.as_ref().and_then(|hidden| hidden.device.as_ref()) {
            buffer.enqueue_deferred_upload()?;
        }
        Ok(())
    }

    /// 取消时 A 已经送出的 prefill 必须等 B 返回 terminal hidden/DSpark aux，
    /// 再按同一个完成水位提交 cache。单会话在途受 prefill_admission_burst 限制，
    /// 因此这里不会退化为等待整份未完成 prompt。
    fn finish_cancelled_prefill(&mut self, task: &mut Glm52BatchTask) -> Result<(), String> {
        if task.completion_tokens != 0 {
            return Ok(());
        }
        let output_context = self.output_context();
        while task.tail_prefill_position < task.prefill_position {
            let item = self.take_tail_ready(0, task)?;
            Self::enqueue_tail_ready(&output_context, &item)?;
            if item.cached || item.decode || item.verify || item.sampled_token.is_some() || item.speculative.is_some() {
                return Err(format!("取消 prefill 等待 B 水位时收到非 prefill frame: position={} decode={} verify={} cached={}", item.position, item.decode, item.verify, item.cached));
            }
            if item.position != task.tail_prefill_position {
                return Err(format!("取消 prefill 的 B 水位不连续: position={} expected={}", item.position, task.tail_prefill_position));
            }
            let end = item.position.checked_add(item.hidden.rows).ok_or("取消 prefill 的 B 水位溢出")?;
            if end > task.prefill_position {
                return Err(format!("取消 prefill 的 B 水位={end} 超过 A 水位={}", task.prefill_position));
            }
            task.last_hidden = Some(output_context.slice_token_rows(&item.hidden, item.hidden.rows - 1, 1).map_err(|error| format!("取消 prefill 取 terminal hidden: {error:?}"))?);
            if let Some(runtime) = self.dspark_runtime.as_ref() {
                let aux = item.aux_hidden.as_ref().ok_or("取消 prefill 等待 B 时缺少 DSpark aux hidden")?;
                if item.aux_taps != runtime.capture_count() || aux.rows != item.hidden.rows {
                    return Err(format!("取消 prefill DSpark aux 非法: taps={}/{} rows={}/{}", item.aux_taps, runtime.capture_count(), aux.rows, item.hidden.rows));
                }
                let normalized = runtime.normalize_aux_hidden(&output_context, aux).map_err(|error| format!("取消 prefill normalize DSpark aux: {error:?}"))?;
                Self::append_dspark_aux_history(&output_context, task, item.position, normalized, runtime.target_history_window())?;
            }
            task.tail_prefill_position = end;
        }
        if task.cached_tokens.len() < task.tail_prefill_position {
            return Err(format!("取消 prefill cache={} 落后 B 水位={}", task.cached_tokens.len(), task.tail_prefill_position));
        }
        task.cached_tokens.truncate(task.tail_prefill_position);
        if task.tail_prefill_position == task.tokens.len() {
            task.prompt_last_hidden = task.last_hidden.clone();
            task.prompt_dspark_aux_history = task.dspark_aux_history.clone();
            task.prompt_dspark_aux_history_start = task.dspark_aux_history_start;
        }
        Ok(())
    }

    /// A0 后处理按 phase 执行：合并 MTP catch-up 与 target head，verify 接受后
    /// 再批量补齐，最后按 draft depth 合并 L78 + LM head。
    fn accept_tail_batch(
        &mut self,
        slots: &mut [Option<Glm52BatchTask>],
        ready: Vec<Glm52TailReady>,
        minimum_dspark_drafts: usize,
        maximum_dspark_drafts: usize,
        profile_completion: bool,
        on_token: &mut dyn FnMut(&str, u32, String) -> bool,
        on_tool_call_delta: &mut dyn FnMut(&str, ToolCallDelta) -> bool,
    ) -> Result<Vec<(usize, Glm52BatchDecision)>, String> {
        let mut completion_phase_started = profile_completion.then(Instant::now);
        let mut completion_phase_micros = [0_u128; 6];
        let output_context = self.output_context();
        for item in &ready {
            Self::enqueue_tail_ready(&output_context, item)?;
        }
        let cfg = self.cfg.clone();
        let mla = self.mla.clone();
        let weights = Arc::clone(&self.weights);
        let mut ready_by_session = std::iter::repeat_with(|| None).take(slots.len()).collect::<Vec<Option<Glm52TailReady>>>();
        let dspark_capture_count = self.dspark_runtime.as_ref().map(RocmDsparkRuntime::capture_count);
        for mut item in ready {
            if item.session >= slots.len() {
                return Err(format!("A0 ready session={} 越界 {}", item.session, slots.len()));
            }
            if item.verify && self.dspark_runtime.is_none() && slots[item.session].as_ref().is_some_and(|task| task.mtp.as_ref().is_some_and(|mtp| mtp.active)) {
                // MTP(nextn) 与 DSpark split verify 同构：K+1 行各自穿 16-stage
                // 回到 A0（每行一个单行 frame）。按到达顺序聚齐 concat 成整段
                // （position 回退到 anchor 位）后再进入 accept 两遍式处理。
                let task = slots[item.session].as_mut().ok_or_else(|| format!("MTP verify 落到空 session={}", item.session))?;
                let expected = task.pending_verify_rows;
                if expected == 0 {
                    return Err(format!("MTP session={} 收到无 pending 的 verify", item.session));
                }
                if expected > item.hidden.rows || !task.mtp_verify_rows.is_empty() {
                    let received = task.mtp_verify_rows.iter().map(|(_, hidden)| hidden.rows).sum::<usize>();
                    let base = task.cached_tokens.len().checked_sub(expected).ok_or("MTP split verify cached_tokens 下溢")?;
                    if item.position != base + received {
                        return Err(format!("MTP split verify session={} position={}，期望 {}", item.session, item.position, base + received));
                    }
                    let total = received.saturating_add(item.hidden.rows);
                    if total > expected {
                        return Err(format!("MTP split verify session={} accumulated rows={total}，期望 {expected}", item.session));
                    }
                    task.mtp_verify_rows.push((item.position, item.hidden.clone()));
                    if total < expected {
                        continue;
                    }
                    item.position = base;
                    let rows = std::mem::take(&mut task.mtp_verify_rows);
                    let hidden_refs = rows.iter().map(|(_, hidden)| hidden).collect::<Vec<_>>();
                    item.hidden = output_context.concat_token_rows(&hidden_refs).map_err(|error| format!("汇合 MTP verify hidden: {error:?}"))?;
                }
            }
            if let Some(capture_count) = dspark_capture_count
                && item.verify
            {
                let task = slots[item.session].as_mut().ok_or_else(|| format!("DSpark verify 落到空 session={}", item.session))?;
                let expected = task.pending_verify_rows;
                if expected == 0 {
                    return Err(format!("DSpark session={} 收到无 pending 的 verify", item.session));
                }
                if expected > item.hidden.rows || !task.dspark_verify_hidden.is_empty() {
                    let aux = item.aux_hidden.as_ref().ok_or_else(|| format!("DSpark split verify session={} 缺少 aux hidden", item.session))?;
                    if item.aux_taps != capture_count {
                        return Err(format!("DSpark split verify session={} capture taps={}，期望 {capture_count}", item.session, item.aux_taps));
                    }
                    if aux.rows != item.hidden.rows {
                        return Err(format!("DSpark split verify session={} hidden rows={}，aux rows={}", item.session, item.hidden.rows, aux.rows));
                    }
                    let received = task.dspark_verify_hidden.iter().map(|hidden| hidden.rows).sum::<usize>();
                    let base = task.cached_tokens.len().checked_sub(expected).ok_or("DSpark split verify cached_tokens 下溢")?;
                    if item.position != base + received {
                        return Err(format!("DSpark split verify session={} position={}，期望 {}", item.session, item.position, base + received));
                    }
                    let total = received.saturating_add(item.hidden.rows);
                    if total > expected {
                        return Err(format!("DSpark split verify session={} accumulated rows={total}，期望 {expected}", item.session));
                    }
                    task.dspark_verify_hidden.push(item.hidden.clone());
                    task.dspark_verify_aux.push(aux.clone());
                    if total < expected {
                        continue;
                    }
                    item.position = base;
                    let hidden = std::mem::take(&mut task.dspark_verify_hidden);
                    let hidden_refs = hidden.iter().collect::<Vec<_>>();
                    item.hidden = output_context.concat_token_rows(&hidden_refs).map_err(|error| format!("汇合 DSpark verify hidden: {error:?}"))?;
                    let aux = std::mem::take(&mut task.dspark_verify_aux);
                    let aux_refs = aux.iter().collect::<Vec<_>>();
                    item.aux_hidden = Some(output_context.concat_token_rows(&aux_refs).map_err(|error| format!("汇合 DSpark verify aux: {error:?}"))?);
                }
            }
            if ready_by_session[item.session].replace(item).is_some() {
                return Err("同一 A0 cohort 出现重复 session".to_owned());
            }
        }
        if let Some(runtime) = self.dspark_runtime.as_ref() {
            for (session, item) in ready_by_session.iter().enumerate().filter_map(|(session, item)| item.as_ref().map(|item| (session, item))) {
                let Some(aux_hidden) = item.aux_hidden.as_ref() else {
                    if !item.cached && item.sampled_token.is_none() && item.speculative.is_none() {
                        return Err(format!("DSpark session={session} 缺少聚合 hidden"));
                    }
                    continue;
                };
                if item.aux_taps != runtime.capture_count() {
                    return Err(format!("DSpark session={session} capture taps={}，期望 {}", item.aux_taps, runtime.capture_count()));
                }
                if item.verify {
                    continue;
                }
                let normalized = runtime.normalize_aux_hidden(&output_context, aux_hidden).map_err(|error| format!("DSpark normalize aux hidden: {error:?}"))?;
                let task = slots[session].as_mut().ok_or_else(|| format!("DSpark ready 落到空 session={session}"))?;
                Self::append_dspark_aux_history(&output_context, task, item.position, normalized, runtime.target_history_window())?;
            }
        }
        let profile_mtp = crate::kernel::rocm::hip::options().kernel_profile && ready_by_session.iter().flatten().any(|item| item.verify);
        let mut mtp_phase_micros = [0_u128; 5];

        // prefill/decode 的真实 target hidden 先补齐 MTP；verify 必须等 target
        // 接受长度确定后再补齐。
        let mut catch_up = std::iter::repeat_with(|| None).take(slots.len()).collect::<Vec<Option<(usize, Vec<u32>, RocmTensor)>>>();
        for (session, item) in ready_by_session.iter().enumerate().filter_map(|(session, item)| item.as_ref().map(|item| (session, item))) {
            let task = slots[session].as_mut().ok_or_else(|| format!("A0 ready 落到空 session={session}"))?;
            if item.cached || item.verify || item.sampled_token.is_some() || item.speculative.is_some() {
                continue;
            }
            let rows = item.hidden.rows;
            let inputs = if item.decode {
                vec![*task.cached_tokens.get(item.position).ok_or_else(|| format!("A0 decode token position={} 越界 {}", item.position, task.cached_tokens.len()))?]
            } else {
                let end = item.position.saturating_add(rows);
                task.tokens.get(item.position..end).ok_or_else(|| format!("A0 prefill tokens 越界: [{},{end})/{}", item.position, task.tokens.len()))?.to_vec()
            };
            if task.mtp.as_ref().is_some_and(|mtp| mtp.active) {
                if item.decode {
                    task.mtp.as_mut().unwrap().truncate_to_target(item.position).map_err(|error| format!("A0 MTP decode truncate: {error:?}"))?;
                }
                catch_up[session] = Some((item.position, inputs, item.hidden.clone()));
            }
        }
        if catch_up.iter().any(Option::is_some) {
            let runtime = self.mtp_runtime.as_mut().ok_or("A0 active MTP session 缺少 runtime")?;
            let mut batch = slots
                .iter_mut()
                .enumerate()
                .filter_map(|(session, slot)| {
                    let (position, inputs, hidden) = catch_up[session].take()?;
                    let mtp = slot.as_mut()?.mtp.as_mut()?;
                    Some(RocmMtpCatchUp { session: mtp, target_position: position, target_inputs: inputs, target_hidden: hidden })
                })
                .collect::<Vec<_>>();
            mtp_catch_up_batch(runtime, &mut batch, &weights, &cfg, &mla, &self.rope).map_err(|error| format!("A0 MTP cohort catch-up: {error:?}"))?;
        }
        if let Some(started) = completion_phase_started.as_mut() {
            completion_phase_micros[0] = started.elapsed().as_micros();
            *started = Instant::now();
        }

        let phase_started = profile_mtp.then(Instant::now);
        let mut head_hiddens = Vec::new();
        let mut head_ranges = vec![None; slots.len()];
        let mut sampling = Vec::new();
        let mut fences = Vec::new();
        for (session, item) in ready_by_session.iter().enumerate().filter_map(|(session, item)| item.as_ref().map(|item| (session, item))) {
            let task = slots[session].as_mut().expect("ready session 已检查");
            if item.sampled_token.is_some() || item.speculative.is_some() {
                continue;
            }
            if !item.cached && !item.decode && !item.verify {
                let last = output_context.slice_token_rows(&item.hidden, item.hidden.rows - 1, 1).map_err(|error| format!("取 A0 prefill terminal hidden: {error:?}"))?;
                task.last_hidden = Some(last.clone());
                let end = item.position + item.hidden.rows;
                if item.position != task.tail_prefill_position {
                    return Err(format!("A0 prefill 的 B 水位不连续: position={} expected={}", item.position, task.tail_prefill_position));
                }
                task.tail_prefill_position = end;
                if end != task.tokens.len() {
                    continue;
                }
                task.prompt_last_hidden = Some(last.clone());
                if self.dspark_runtime.is_some() {
                    let history = task.dspark_aux_history.as_ref().ok_or("prompt 完成时缺少 DSpark aux history")?;
                    let history_end = task.dspark_aux_history_start.checked_add(history.rows).ok_or("DSpark aux history 位置溢出")?;
                    if history_end != end {
                        return Err(format!("prompt DSpark aux history=[{},{history_end})，prompt={end}", task.dspark_aux_history_start));
                    }
                    task.prompt_dspark_aux_history = Some(history.clone());
                    task.prompt_dspark_aux_history_start = task.dspark_aux_history_start;
                }
                head_ranges[session] = Some((sampling.len(), 1));
                sampling.push(task.sampling.next());
                fences.push(task.token_fence.fence());
                head_hiddens.push(last);
            } else if item.verify {
                if task.pending_verify_rows != item.hidden.rows {
                    return Err(format!("MTP verify rows={}，期望 {}", item.hidden.rows, task.pending_verify_rows));
                }
                let mut speculative = task.sampling;
                head_ranges[session] = Some((sampling.len(), item.hidden.rows));
                sampling.extend((0..item.hidden.rows).map(|_| speculative.next()));
                let mut speculative_fence = task.token_fence.clone();
                let verify_inputs = if let Some(mtp) = task.mtp.as_ref().filter(|mtp| mtp.active) { mtp.verify_inputs.clone() } else { task.dspark_verify_inputs.clone() };
                if verify_inputs.len() != item.hidden.rows {
                    return Err(format!("speculative verify inputs={} rows={}", verify_inputs.len(), item.hidden.rows));
                }
                for row in 0..item.hidden.rows {
                    fences.push(speculative_fence.fence());
                    if let Some(&draft) = verify_inputs.get(row + 1) {
                        speculative_fence.advance(draft);
                    }
                }
                head_hiddens.push(item.hidden.clone());
            } else {
                if item.decode {
                    task.last_hidden = Some(item.hidden.clone());
                }
                let hidden = task.last_hidden.clone().ok_or("A0 target head 缺少 terminal hidden")?;
                head_ranges[session] = Some((sampling.len(), 1));
                sampling.push(task.sampling.next());
                fences.push(task.token_fence.fence());
                head_hiddens.push(hidden);
            }
        }
        let target_tokens = if head_hiddens.is_empty() {
            Vec::new()
        } else {
            let refs = head_hiddens.iter().collect::<Vec<_>>();
            let hidden = output_context.concat_token_rows(&refs).map_err(|error| format!("拼接 A0 target head cohort: {error:?}"))?;
            glm52_sampled_token_ids_fenced(&output_context, &cfg, self.output_head.as_ref().ok_or("A0 target head 未加载")?, &hidden, &sampling, &fences).map_err(|error| format!("A0 target head cohort: {error:?}"))?
        };
        mtp_phase_micros[0] = phase_started.map_or(0, |started| started.elapsed().as_micros());
        if let Some(started) = completion_phase_started.as_mut() {
            completion_phase_micros[1] = started.elapsed().as_micros();
            *started = Instant::now();
        }

        let phase_started = profile_mtp.then(Instant::now);
        let mut outcomes = std::iter::repeat_with(|| None).take(slots.len()).collect::<Vec<Option<Glm52TailOutcome>>>();
        let mut verify_catch_up = std::iter::repeat_with(|| None).take(slots.len()).collect::<Vec<Option<(usize, Vec<u32>, RocmTensor)>>>();
        for (session, item) in ready_by_session.iter().enumerate().filter_map(|(session, item)| item.as_ref().map(|item| (session, item))) {
            if let Some((frame_tokens, retained_rows, frame_drafts, frame_eos)) = &item.speculative {
                let task = slots[session].as_mut().expect("speculative session 已检查");
                if frame_tokens.is_empty() {
                    return Err("B7 MTP speculative token 不能为空".to_owned());
                }
                if task.pending_verify_rows == 0 {
                    if *retained_rows != 0 {
                        return Err(format!("B7 MTP 初始 retained_rows={retained_rows}，期望 0"));
                    }
                } else {
                    if *retained_rows == 0 || *retained_rows > task.pending_verify_rows {
                        return Err(format!("B7 MTP retained_rows={retained_rows} 超出 verify rows={}", task.pending_verify_rows));
                    }
                    let base = task.cached_tokens.len().checked_sub(task.pending_verify_rows).ok_or("B7 MTP verify cached_tokens 下溢")?;
                    task.cached_tokens.truncate(base + retained_rows);
                    task.pending_verify_rows = 0;
                }
                let mut tokens = frame_tokens.clone();
                let mut drafts = frame_drafts.clone();
                let forced_budget = if let (Some(end_token), Some(budget)) = (task.thinking_end_token, task.thinking_token_budget)
                    && let Some(forced_rows) = enforce_thinking_token_budget(&mut tokens, task.thinking_tokens, budget, end_token)
                {
                    if *retained_rows != 0 {
                        let base = task.cached_tokens.len().saturating_sub(*retained_rows);
                        task.cached_tokens.truncate(base + forced_rows);
                    }
                    drafts.clear();
                    Some(budget)
                } else {
                    None
                };
                if let Some(budget) = forced_budget {
                    eprintln!("[glm52-thinking-budget] request_id={} budget={} inserted=</think>", task.stage_id, budget);
                }
                let loop_recovery = forced_budget.is_none().then(|| task.token_fence.recover(&mut tokens, task.thinking_end_token)).flatten();
                if let Some(recovery) = loop_recovery {
                    if *retained_rows != 0 {
                        let base = task.cached_tokens.len().saturating_sub(*retained_rows);
                        task.cached_tokens.truncate(base + recovery.retained_rows);
                    }
                    drafts.clear();
                    eprintln!("[glm52-loop-guard] request_id={} kind={} action={}", task.stage_id, recovery.kind, if recovery.forced_end { "close_thinking" } else { "stop" });
                }
                let hard_loop = loop_recovery.filter(|recovery| !recovery.forced_end).map(|recovery| recovery.kind);
                outcomes[session] = Some(Glm52TailOutcome { tokens, drafts, eos: forced_budget.is_none() && loop_recovery.is_none() && *frame_eos, hard_loop, continue_dspark: false });
                continue;
            }
            if let Some((token, sampled_eos)) = item.sampled_token {
                let task = slots[session].as_mut().expect("sampled session 已检查");
                let mut tokens = vec![token];
                let forced_budget = if let (Some(end_token), Some(budget)) = (task.thinking_end_token, task.thinking_token_budget)
                    && enforce_thinking_token_budget(&mut tokens, task.thinking_tokens, budget, end_token).is_some()
                {
                    Some(budget)
                } else {
                    None
                };
                if let Some(budget) = forced_budget {
                    eprintln!("[glm52-thinking-budget] request_id={} budget={} inserted=</think>", task.stage_id, budget);
                }
                let loop_recovery = forced_budget.is_none().then(|| task.token_fence.recover(&mut tokens, task.thinking_end_token)).flatten();
                if let Some(recovery) = loop_recovery {
                    eprintln!("[glm52-loop-guard] request_id={} kind={} action={}", task.stage_id, recovery.kind, if recovery.forced_end { "close_thinking" } else { "stop" });
                }
                let hard_loop = loop_recovery.filter(|recovery| !recovery.forced_end).map(|recovery| recovery.kind);
                outcomes[session] = Some(Glm52TailOutcome { tokens, drafts: Vec::new(), eos: forced_budget.is_none() && loop_recovery.is_none() && sampled_eos, hard_loop, continue_dspark: false });
                continue;
            }
            let Some((offset, count)) = head_ranges[session] else { continue };
            let task = slots[session].as_mut().expect("ready session 已检查");
            let dspark_active = self.dspark_runtime.is_some() && task.mtp.as_ref().is_none_or(|mtp| !mtp.active);
            // MTP(nextn) verify 与 DSpark 共用计数器，命中率统计两条路径都覆盖。
            let mtp_active = task.mtp.as_ref().is_some_and(|mtp| mtp.active);
            if dspark_active || mtp_active {
                task.dspark_target_rounds += 1;
            }
            if item.verify {
                let verify_inputs = if let Some(mtp) = task.mtp.as_ref().filter(|mtp| mtp.active) { mtp.verify_inputs.clone() } else { task.dspark_verify_inputs.clone() };
                if verify_inputs.len() != count {
                    return Err(format!("A0 speculative verify inputs={} rows={count}", verify_inputs.len()));
                }
                let (mut outcome, verified_depth, verified_drafts, terminal_chunk) = if let Some(flight) = task.dspark_cpu_flight.as_ref() {
                    let expected_inputs = flight.inputs();
                    if verify_inputs != expected_inputs {
                        return Err(format!("CPU DSpark verify window 失配: inputs={verify_inputs:?} expected={expected_inputs:?}"));
                    }
                    let expected = flight.expected(count);
                    let terminal = flight.terminal(count);
                    let outcome = if terminal { verify_samples(&target_tokens[offset..offset + count], &expected, &cfg.eos_token_ids) } else { verify_samples_prefix(&target_tokens[offset..offset + count], &expected, &cfg.eos_token_ids) }
                        .map_err(|error| format!("A0 CPU DSpark verify: {error:?}"))?;
                    (outcome, flight.depth(), expected.len(), terminal)
                } else {
                    let outcome = verify_samples(&target_tokens[offset..offset + count], &verify_inputs[1..], &cfg.eos_token_ids).map_err(|error| format!("A0 MTP verify: {error:?}"))?;
                    (outcome, 0, count - 1, true)
                };
                let model_accepted_drafts = outcome.accepted_drafts;
                let forced_budget = if let (Some(end_token), Some(budget)) = (task.thinking_end_token, task.thinking_token_budget)
                    && let Some(retained_rows) = enforce_thinking_token_budget(&mut outcome.tokens, task.thinking_tokens, budget, end_token)
                {
                    outcome.accepted_drafts = retained_rows.saturating_sub(1);
                    outcome.retained_rows = retained_rows;
                    outcome.eos = false;
                    eprintln!("[glm52-thinking-budget] request_id={} budget={} inserted=</think>", task.stage_id, budget);
                    true
                } else {
                    false
                };
                let loop_recovery = (!forced_budget).then(|| task.token_fence.recover(&mut outcome.tokens, task.thinking_end_token)).flatten();
                if let Some(recovery) = loop_recovery {
                    outcome.accepted_drafts = recovery.retained_rows.saturating_sub(1);
                    outcome.retained_rows = recovery.retained_rows;
                    outcome.eos = false;
                    eprintln!("[glm52-loop-guard] request_id={} kind={} action={}", task.stage_id, recovery.kind, if recovery.forced_end { "close_thinking" } else { "stop" });
                }
                let continue_dspark = task.dspark_cpu_flight.is_some() && !terminal_chunk && model_accepted_drafts == verified_drafts && outcome.retained_rows == count && !outcome.eos && !forced_budget && loop_recovery.is_none();
                for _ in 0..outcome.retained_rows {
                    task.sampling.next();
                }
                let retained_hidden = output_context.slice_token_rows(&item.hidden, 0, outcome.retained_rows).map_err(|error| format!("取 A0 MTP retained hidden: {error:?}"))?;
                let last = output_context.slice_token_rows(&item.hidden, outcome.retained_rows - 1, 1).map_err(|error| format!("取 A0 MTP terminal hidden: {error:?}"))?;
                if let Some(mtp) = task.mtp.as_mut().filter(|mtp| mtp.active) {
                    mtp.truncate_to_target(item.position).map_err(|error| format!("A0 MTP rollback: {error:?}"))?;
                    verify_catch_up[session] = Some((item.position, verify_inputs[..outcome.retained_rows].to_vec(), retained_hidden));
                } else if self.dspark_runtime.is_some() {
                    let aux = item.aux_hidden.as_ref().ok_or("DSpark verify 缺少 aux hidden")?;
                    let normalized = self.dspark_runtime.as_ref().unwrap().normalize_aux_hidden(&output_context, aux).map_err(|error| format!("DSpark verify normalize: {error:?}"))?;
                    let retained = output_context.slice_token_rows(&normalized, 0, outcome.retained_rows).map_err(|error| format!("取 DSpark retained aux: {error:?}"))?;
                    let history = task.dspark_aux_history.take().ok_or("DSpark verify 缺少 aux history")?;
                    let start = task.dspark_aux_history_start;
                    let end = start.checked_add(history.rows).ok_or("DSpark aux history 位置溢出")?;
                    if item.position < start || item.position > end {
                        return Err(format!("DSpark verify rollback={}; history=[{start},{end})", item.position));
                    }
                    let prefix_rows = item.position - start;
                    task.dspark_aux_history = if prefix_rows == 0 { None } else { Some(output_context.slice_token_rows(&history, 0, prefix_rows).map_err(|error| format!("截断 DSpark aux history: {error:?}"))?) };
                    Self::append_dspark_aux_history(&output_context, task, item.position, retained, self.dspark_runtime.as_ref().unwrap().target_history_window())?;
                    task.dspark_verify_inputs.clear();
                }
                task.cached_tokens.truncate(item.position + outcome.retained_rows);
                task.pending_verify_rows = 0;
                if continue_dspark {
                    task.dspark_cpu_flight.as_mut().expect("CPU DSpark continuation 已检查").advance(count);
                } else {
                    task.dspark_cpu_flight = None;
                    task.dspark_cpu_anchor_in_flight = false;
                }
                task.last_hidden = Some(last);
                if dspark_active || mtp_active {
                    task.dspark_verify_rounds += 1;
                    task.dspark_verified_drafts += verified_drafts;
                    task.dspark_accepted_drafts += model_accepted_drafts;
                    for depth in verified_depth..verified_depth.saturating_add(verified_drafts) {
                        if let Some(verified) = task.dspark_verified_by_depth.get_mut(depth) {
                            *verified += 1;
                        }
                    }
                    for depth in verified_depth..verified_depth.saturating_add(model_accepted_drafts) {
                        if let Some(accepted) = task.dspark_accepted_by_depth.get_mut(depth) {
                            *accepted += 1;
                        }
                    }
                }
                if self.options.diagnostics.trace_stage_events || self.options.diagnostics.profile_boundaries {
                    eprintln!(
                        "[glm52-mtp-verify] request_id={} drafts={} accepted={}/{} depth={} terminal={} continue={} retained={}/{} output_tokens={} eos={}",
                        task.stage_id,
                        verified_drafts,
                        model_accepted_drafts,
                        verified_drafts,
                        verified_depth,
                        terminal_chunk,
                        continue_dspark,
                        outcome.retained_rows,
                        count,
                        outcome.tokens.len(),
                        outcome.eos
                    );
                }
                let hard_loop = loop_recovery.filter(|recovery| !recovery.forced_end).map(|recovery| recovery.kind);
                outcomes[session] = Some(Glm52TailOutcome { tokens: outcome.tokens, drafts: Vec::new(), eos: outcome.eos, hard_loop, continue_dspark });
            } else {
                let token = target_tokens[offset];
                let mut tokens = vec![token];
                let forced_budget = if let (Some(end_token), Some(budget)) = (task.thinking_end_token, task.thinking_token_budget)
                    && enforce_thinking_token_budget(&mut tokens, task.thinking_tokens, budget, end_token).is_some()
                {
                    Some(budget)
                } else {
                    None
                };
                if let Some(budget) = forced_budget {
                    eprintln!("[glm52-thinking-budget] request_id={} budget={} inserted=</think>", task.stage_id, budget);
                }
                let loop_recovery = forced_budget.is_none().then(|| task.token_fence.recover(&mut tokens, task.thinking_end_token)).flatten();
                if let Some(recovery) = loop_recovery {
                    eprintln!("[glm52-loop-guard] request_id={} kind={} action={}", task.stage_id, recovery.kind, if recovery.forced_end { "close_thinking" } else { "stop" });
                }
                let hard_loop = loop_recovery.filter(|recovery| !recovery.forced_end).map(|recovery| recovery.kind);
                outcomes[session] = Some(Glm52TailOutcome { tokens, drafts: Vec::new(), eos: forced_budget.is_none() && loop_recovery.is_none() && cfg.eos_token_ids.contains(&token), hard_loop, continue_dspark: false });
            }
        }
        if profile_mtp {
            output_context.synchronize().map_err(|error| format!("同步 A0 MTP verify accept: {error:?}"))?;
        }
        mtp_phase_micros[1] = phase_started.map_or(0, |started| started.elapsed().as_micros());
        if let Some(started) = completion_phase_started.as_mut() {
            completion_phase_micros[2] = started.elapsed().as_micros();
            *started = Instant::now();
        }
        let phase_started = profile_mtp.then(Instant::now);
        if verify_catch_up.iter().any(Option::is_some) {
            let runtime = self.mtp_runtime.as_mut().ok_or("A0 verify MTP session 缺少 runtime")?;
            let mut batch = slots
                .iter_mut()
                .enumerate()
                .filter_map(|(session, slot)| {
                    let (position, inputs, hidden) = verify_catch_up[session].take()?;
                    let mtp = slot.as_mut()?.mtp.as_mut()?;
                    Some(RocmMtpCatchUp { session: mtp, target_position: position, target_inputs: inputs, target_hidden: hidden })
                })
                .collect::<Vec<_>>();
            mtp_catch_up_batch(runtime, &mut batch, &weights, &cfg, &mla, &self.rope).map_err(|error| format!("A0 MTP verify cohort catch-up: {error:?}"))?;
            if profile_mtp {
                runtime.backend.synchronize().map_err(|error| format!("同步 A0 MTP verify catch-up: {error:?}"))?;
            }
        }
        mtp_phase_micros[2] = phase_started.map_or(0, |started| started.elapsed().as_micros());
        if let Some(started) = completion_phase_started.as_mut() {
            completion_phase_micros[3] = started.elapsed().as_micros();
            *started = Instant::now();
        }

        let phase_started = profile_mtp.then(Instant::now);
        if self.mtp_runtime.is_some() {
            let mut draft_sessions = Vec::new();
            let mut draft_batch = slots
                .iter_mut()
                .enumerate()
                .filter_map(|(session, slot)| {
                    let outcome = outcomes[session].as_ref()?;
                    if outcome.eos || outcome.hard_loop.is_some() {
                        return None;
                    }
                    let task = slot.as_mut()?;
                    let mtp = task.mtp.as_mut().filter(|mtp| mtp.active)?;
                    let latest = *outcome.tokens.last()?;
                    let remaining = mtp.max_decode.saturating_sub(task.completion_tokens.saturating_add(outcome.tokens.len()));
                    let count = mtp.draft_tokens.min(remaining.saturating_sub(1));
                    let hidden = mtp.pending_hidden.clone()?;
                    let mut fence = task.token_fence.clone();
                    fence.advance(latest);
                    draft_sessions.push(session);
                    Some(RocmMtpDraftBatch { session: mtp, token: latest, hidden, count, drafts: Vec::new(), fence })
                })
                .collect::<Vec<_>>();
            if !draft_batch.is_empty() {
                mtp_draft_batch(self.mtp_runtime.as_mut().unwrap(), &mut draft_batch, &weights, self.output_head.as_ref().ok_or("A0 MTP output head 未加载")?, &cfg, &mla, &self.rope)
                    .map_err(|error| format!("A0 MTP draft cohort: {error:?}"))?;
                for (session, item) in draft_sessions.into_iter().zip(draft_batch.iter_mut()) {
                    let outcome = outcomes[session].as_mut().expect("draft outcome 已检查");
                    outcome.drafts = std::mem::take(&mut item.drafts);
                    item.session.verify_inputs = std::iter::once(*outcome.tokens.last().unwrap()).chain(outcome.drafts.iter().copied()).collect();
                }
            }
        }
        let mut cpu_dispatched = vec![false; slots.len()];
        if self.cpu_dspark_executor.is_some() {
            let mut cpu_jobs = Vec::new();
            for (session, slot) in slots.iter_mut().enumerate() {
                let Some(outcome) = outcomes[session].as_ref() else { continue };
                if outcome.eos || outcome.hard_loop.is_some() || outcome.continue_dspark {
                    continue;
                }
                let task = slot.as_mut().expect("CPU DSpark outcome session 已检查");
                let align_started = Instant::now();
                self.align_dspark_target_cache(task)?;
                let align_micros = align_started.elapsed().as_micros();
                let Some(&anchor) = outcome.tokens.last() else { continue };
                let remaining = task.max_decode.saturating_sub(task.completion_tokens.saturating_add(outcome.tokens.len()));
                if remaining <= 1 {
                    continue;
                }
                let transfer_started = Instant::now();
                let (history, cache, target_position, block_position) = self.take_cpu_dspark_state(task)?;
                let transfer_micros = transfer_started.elapsed().as_micros();
                let id = self.next_cpu_dspark_job;
                self.next_cpu_dspark_job = self.next_cpu_dspark_job.wrapping_add(1);
                if self.options.diagnostics.trace_stage_events {
                    crate::runtime::prefill_scheduler::record_stage_trace(format!(
                        "[dspark-work-trace] ts_us={} phase=submit lane={session}@{} job={id} anchor={anchor} target_position={target_position} block_position={block_position} align_us={align_micros} transfer_us={transfer_micros}",
                        crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
                        task.cached_tokens.len().saturating_sub(1),
                    ));
                }
                cpu_jobs.push(CpuDsparkJob { session, id, anchor, max_drafts: (remaining - 1).min(maximum_dspark_drafts), cache, history, target_position, block_position, minimum_drafts: minimum_dspark_drafts });
                task.dspark_cpu_pending = Some(id);
                task.dspark_cpu_anchor_in_flight = true;
                task.dspark_cpu_window = minimum_dspark_drafts.saturating_add(1);
                cpu_dispatched[session] = true;
                self.cpu_dspark_submit_profile[0] += 1;
                self.cpu_dspark_submit_profile[1] += align_micros;
                self.cpu_dspark_submit_profile[2] += transfer_micros;
                if self.cpu_dspark_submit_profile[0] == 32 {
                    eprintln!("[glm52-dspark-cpu-submit] jobs=32 align_ms={:.3} transfer_ms={:.3}", self.cpu_dspark_submit_profile[1] as f64 / 32_000.0, self.cpu_dspark_submit_profile[2] as f64 / 32_000.0,);
                    self.cpu_dspark_submit_profile.fill(0);
                }
            }
            self.cpu_dspark_executor.as_ref().unwrap().submit_batch(cpu_jobs)?;
        } else if let Some(runtime) = self.dspark_runtime.as_ref() {
            // 只合并本轮已经到达 A0 的 draft，绝不等待尚未就绪的请求。
            let mut draft_sessions = Vec::new();
            let mut draft_batch = Vec::new();
            for (session, slot) in slots.iter_mut().enumerate() {
                let Some(outcome) = outcomes[session].as_ref() else { continue };
                if outcome.eos || outcome.hard_loop.is_some() {
                    continue;
                }
                let task = slot.as_mut().expect("DSpark outcome session 已检查");
                // verify 后 cache 可能包含未接受尾部或落后于新接受前缀；先按
                // terminal 语义截断/补后缀，draft backbone 才能只跑 query block。
                self.align_dspark_target_cache(task)?;
                let Some(history) = task.dspark_aux_history.as_ref() else { continue };
                let Some(&anchor) = outcome.tokens.last() else { continue };
                let remaining = task.max_decode.saturating_sub(task.completion_tokens.saturating_add(outcome.tokens.len()));
                if remaining <= 1 {
                    continue;
                }
                draft_sessions.push((session, remaining, anchor));
                draft_batch.push(RocmDsparkDraftBatch {
                    cache: &mut task.dspark_target_cache,
                    anchor,
                    target_hidden: history,
                    target_position: task.dspark_aux_history_start,
                    block_position: task.dspark_aux_history_start + history.rows,
                    minimum_drafts: minimum_dspark_drafts,
                    drafts: Vec::new(),
                });
            }
            runtime.draft_batch(&output_context, &mut draft_batch).map_err(|error| format!("A0 DSpark draft batch: {error:?}"))?;
            let drafted = draft_sessions.into_iter().zip(draft_batch).map(|((session, remaining, anchor), item)| (session, remaining, anchor, item.drafts)).collect::<Vec<_>>();
            for (session, remaining, anchor, drafts) in drafted {
                let outcome = outcomes[session].as_mut().expect("DSpark draft outcome 已检查");
                let task = slots[session].as_mut().expect("DSpark draft session 已检查");
                outcome.drafts = drafts.into_iter().take((remaining - 1).min(maximum_dspark_drafts)).collect();
                task.dspark_verify_inputs = std::iter::once(anchor).chain(outcome.drafts.iter().copied()).collect();
            }
        }
        mtp_phase_micros[3] = phase_started.map_or(0, |started| started.elapsed().as_micros());
        if let Some(started) = completion_phase_started.as_mut() {
            completion_phase_micros[4] = started.elapsed().as_micros();
            *started = Instant::now();
        }

        let phase_started = profile_mtp.then(Instant::now);
        let mut decisions = Vec::new();
        for (session, outcome) in outcomes.into_iter().enumerate().filter_map(|(session, outcome)| outcome.map(|outcome| (session, outcome))) {
            let task = slots[session].as_mut().expect("output session 已检查");
            let request_id = task.request_id.clone();
            for (index, &token) in outcome.tokens.iter().enumerate() {
                task.token_fence.advance(token);
                let text = self.detokenizer.decode_bytes(&[token], true).map_err(|error| format!("detokenize {token}: {error}"))?;
                let text = task.utf8.push(&text);
                // 边界 </think> 自己必须原样通过(server 靠它切分 reasoning);
                // 状态翻转在其后发生,因此它之后的答案文本才会被剥离。
                let text = task.think_filter.push(&text, task.thinking_end_token.is_some());
                if !emit_glm_tool_aware_chunk(&mut task.tool_stream, token, &text, &mut task.response_text, &mut |token, text| on_token(&request_id, token, text), &mut |delta| on_tool_call_delta(&request_id, delta)) {
                    task.finish_reason = "cancelled".to_owned();
                    break;
                }
                if task.thinking_end_token == Some(token) {
                    task.thinking_end_token = None;
                } else if task.thinking_end_token.is_some() {
                    task.thinking_tokens += 1;
                }
                task.completion_tokens += 1;
                self.compute_steps.fetch_add(1);
                if cfg.eos_token_ids.contains(&token) || outcome.eos && index + 1 == outcome.tokens.len() {
                    task.pending_token = None;
                    task.finish_reason = "stop".to_owned();
                    break;
                }
                task.pending_token = Some(token);
                if outcome.hard_loop.is_some() && index + 1 == outcome.tokens.len() {
                    task.finish_reason = "repetition".to_owned();
                    break;
                }
                if task.completion_tokens >= task.max_decode {
                    break;
                }
            }
            let next = if matches!(task.finish_reason.as_str(), "cancelled" | "stop" | "repetition") || task.completion_tokens >= task.max_decode {
                Some(Glm52NextWork::Finish)
            } else if outcome.continue_dspark {
                let inputs = task.dspark_cpu_flight.as_ref().ok_or("CPU DSpark continuation 缺少 flight")?.inputs();
                Some(Glm52NextWork::Verify(inputs))
            } else if cpu_dispatched[session] {
                let token = *outcome.tokens.last().expect("已检查非空");
                Some(Glm52NextWork::Verify(vec![token]))
            } else {
                let token = *outcome.tokens.last().expect("已检查非空");
                Some(if outcome.drafts.is_empty() { Glm52NextWork::Decode(token) } else { Glm52NextWork::Verify(std::iter::once(token).chain(outcome.drafts).collect()) })
            };
            decisions.push((session, Glm52BatchDecision { next }));
        }
        for (session, item) in ready_by_session.into_iter().enumerate() {
            if item.as_ref().is_some_and(|item| item.sampled_token.is_none() && item.speculative.is_none()) && head_ranges[session].is_none() {
                decisions.push((session, Glm52BatchDecision { next: None }));
            }
        }
        mtp_phase_micros[4] = phase_started.map_or(0, |started| started.elapsed().as_micros());
        if let Some(started) = completion_phase_started {
            completion_phase_micros[5] = started.elapsed().as_micros();
            let total_micros = completion_phase_micros.iter().sum::<u128>();
            self.accept_phase_profile[0] += 1;
            for (total, phase) in self.accept_phase_profile[1..].iter_mut().zip(completion_phase_micros) {
                *total += phase;
            }
            if self.accept_phase_profile[0] == 32 {
                eprintln!(
                    "[glm52-accept-phases] calls=32 prepare_ms={:.3} target_head_ms={:.3} accept_ms={:.3} catch_up_ms={:.3} draft_submit_ms={:.3} emit_ms={:.3}",
                    self.accept_phase_profile[1] as f64 / 32_000.0,
                    self.accept_phase_profile[2] as f64 / 32_000.0,
                    self.accept_phase_profile[3] as f64 / 32_000.0,
                    self.accept_phase_profile[4] as f64 / 32_000.0,
                    self.accept_phase_profile[5] as f64 / 32_000.0,
                    self.accept_phase_profile[6] as f64 / 32_000.0,
                );
                self.accept_phase_profile.fill(0);
            }
            if total_micros >= 500_000 {
                eprintln!(
                    "[glm52-accept-slow] total_ms={:.3} prepare_ms={:.3} target_head_ms={:.3} accept_ms={:.3} catch_up_ms={:.3} draft_ms={:.3} emit_ms={:.3}",
                    total_micros as f64 / 1000.0,
                    completion_phase_micros[0] as f64 / 1000.0,
                    completion_phase_micros[1] as f64 / 1000.0,
                    completion_phase_micros[2] as f64 / 1000.0,
                    completion_phase_micros[3] as f64 / 1000.0,
                    completion_phase_micros[4] as f64 / 1000.0,
                    completion_phase_micros[5] as f64 / 1000.0,
                );
            }
        }
        if profile_mtp {
            eprintln!(
                "[glm52-mtp-phases] target_head_ms={:.3} accept_ms={:.3} catch_up_ms={:.3} draft_ms={:.3} emit_ms={:.3}",
                mtp_phase_micros[0] as f64 / 1000.0,
                mtp_phase_micros[1] as f64 / 1000.0,
                mtp_phase_micros[2] as f64 / 1000.0,
                mtp_phase_micros[3] as f64 / 1000.0,
                mtp_phase_micros[4] as f64 / 1000.0,
            );
        }
        Ok(decisions)
    }

    fn finish_batch_task(&mut self, mut task: Glm52BatchTask, on_token: &mut dyn FnMut(&str, u32, String) -> bool, on_tool_call_delta: &mut dyn FnMut(&str, ToolCallDelta) -> bool) -> Result<GenerationSummary, String> {
        let request_id = task.request_id.clone();
        let mut trailing = task.utf8.finish();
        trailing = task.think_filter.push(&trailing, task.thinking_end_token.is_some());
        trailing.push_str(&task.think_filter.finish());
        if task.finish_reason != "cancelled"
            && !trailing.is_empty()
            && !emit_glm_tool_aware_chunk(&mut task.tool_stream, 0, &trailing, &mut task.response_text, &mut |token, text| on_token(&request_id, token, text), &mut |delta| on_tool_call_delta(&request_id, delta))
        {
            task.finish_reason = "cancelled".to_owned();
        }
        if task.finish_reason != "cancelled" && !finish_glm_tool_aware_stream(&mut task.tool_stream, &mut task.response_text, &mut |token, text| on_token(&task.request_id, token, text)) {
            task.finish_reason = "cancelled".to_owned();
        }
        if task.finish_reason != "cancelled" && !task.tool_stream.calls.is_empty() {
            task.finish_reason = "tool_calls".to_owned();
        }
        let cancelled = task.finish_reason == "cancelled";
        if cancelled {
            self.finish_cancelled_prefill(&mut task)?;
        }
        if cancelled && task.completion_tokens > 0 {
            let prompt_tokens = task.tokens.len();
            if task.cached_tokens.len() < prompt_tokens {
                return Err(format!("取消后恢复 prompt cache 失败: cached={} prompt={prompt_tokens}", task.cached_tokens.len()));
            }
            for state in &mut task.states {
                state.cache.truncate_rows(prompt_tokens).map_err(|error| format!("回退 prompt KV cache: {error:?}"))?;
                state.dsa.truncate_rows(prompt_tokens).map_err(|error| format!("回退 prompt DSA cache: {error:?}"))?;
            }
            task.cached_tokens.truncate(prompt_tokens);
            task.pending_token = None;
            task.last_hidden = Some(task.prompt_last_hidden.take().ok_or("取消后恢复 prompt cache 时缺少 prompt hidden")?);
            if self.dspark_runtime.is_some() {
                task.dspark_aux_history = Some(task.prompt_dspark_aux_history.take().ok_or("取消后恢复 prompt cache 时缺少 DSpark aux history")?);
                task.dspark_aux_history_start = task.prompt_dspark_aux_history_start;
            }
            if let Some(mtp) = task.mtp.as_mut()
                && mtp.active
            {
                mtp.truncate_to_target(prompt_tokens).map_err(|error| format!("取消后回退 A0 MTP cache: {error:?}"))?;
                mtp.pending_hidden = task.last_hidden.clone();
            }
        }
        if self.dspark_runtime.is_some() || task.dspark_verify_rounds > 0 {
            let acceptance = task.dspark_accepted_drafts as f64 * 100.0 / task.dspark_verified_drafts.max(1) as f64;
            eprintln!(
                "[glm52-dspark-summary] request_id={} completion={} target_rounds={} verify_rounds={} verified_drafts={} accepted_drafts={} acceptance={acceptance:.2}% verified_by_depth={:?} accepted_by_depth={:?}",
                task.stage_id, task.completion_tokens, task.dspark_target_rounds, task.dspark_verify_rounds, task.dspark_verified_drafts, task.dspark_accepted_drafts, task.dspark_verified_by_depth, task.dspark_accepted_by_depth,
            );
        }
        let cache_id = if task.finish_reason != "cancelled" {
            Some(terminal_cache_id(&task.request, &task.response_text, &task.tool_stream.calls)?)
        } else if !task.cached_tokens.is_empty() {
            // API 已按当前内容生成并按 API key namespace 隔离 cache_id。取消时只发布
            // 真正走完本机 pipeline 的 token；下游 Cache ACK 再保证尾段也提交到同一位置。
            task.request.get("cache_id").and_then(Value::as_str).map(str::to_owned)
        } else {
            None
        };
        let preserve_existing =
            cancelled && cache_id.as_deref().is_some_and(|cache_id| self.terminal_states.cached_tokens(cache_id).is_some_and(|existing| existing.len() >= task.cached_tokens.len() && existing.starts_with(&task.cached_tokens)));
        let cache = if preserve_existing {
            let cache_id = cache_id.as_deref().unwrap_or_default();
            let tokens = task.cached_tokens.len();
            self.link.send_delete(task.stage_id).and_then(|()| self.wait_ready_after_cancel(task.stage_id, tokens))?;
            eprintln!("[glm52-prefill-checkpoint-preserve] cache_id={cache_id} rejected_tokens={tokens} existing_tokens={}", self.terminal_states.cached_tokens(cache_id).map_or(0, |existing| existing.len()));
            None
        } else if let Some(cache_id) = cache_id {
            self.align_dspark_target_cache(&mut task)?;
            let info = CacheInfo {
                cache_id: cache_id.clone(),
                model_key: "glm-5.2".to_owned(),
                cache_format: "glm52-rocm-mla-v1".to_owned(),
                last_layer: self.cfg.layer_count.saturating_sub(1),
                prompt_tokens: task.cached_tokens.len(),
                bytes: task.states.iter().map(Glm52StageState::cache_allocated_bytes).sum::<u64>()
                    + task.mtp.as_ref().map_or(0, |mtp| mtp.cache.allocated_bytes().saturating_add(mtp.dsa.allocated_bytes()))
                    + task.dspark_aux_history.as_ref().map_or(0, |hidden| self.output_context().tensor_allocated_bytes(hidden))
                    + task.dspark_target_cache.allocated_bytes(&self.output_context()),
                modified_unix: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            };
            // 最后一轮 verify 可能已经把未接受的 draft 写入各层；没有下一轮时
            // truncate_to 不会再经过 pipeline，terminal cache 必须在这里收口。
            for state in &mut task.states {
                state.cache.truncate_rows(info.prompt_tokens).map_err(|error| format!("提交 A0 terminal KV cache: {error:?}"))?;
                state.dsa.truncate_rows(info.prompt_tokens).map_err(|error| format!("提交 A0 terminal DSA cache: {error:?}"))?;
            }
            let next_request_id = RequestId::from_cache_id(&cache_id);
            let last_hidden = task.last_hidden.ok_or("提交 batch cache 时缺少 head hidden")?;
            let mtp = match task.mtp.take() {
                Some(mut mtp) if mtp.active => {
                    mtp.truncate_to_target(info.prompt_tokens).map_err(|error| format!("提交 A0 MTP cache: {error:?}"))?;
                    mtp.pending_hidden = Some(last_hidden.clone());
                    mtp.deactivate();
                    Some(mtp)
                }
                _ => None,
            };
            let (dspark_aux_history, dspark_aux_history_start) = match task.dspark_aux_history.take() {
                Some(history) => {
                    let end = task.dspark_aux_history_start.checked_add(history.rows).ok_or("提交 DSpark aux history 位置溢出")?;
                    if history.cols != self.cfg.hidden_size || end != info.prompt_tokens {
                        return Err(format!("提交 DSpark aux history=[{},{end}) shape=[{},{}]，terminal={}", task.dspark_aux_history_start, history.rows, history.cols, info.prompt_tokens));
                    }
                    (Some(history), task.dspark_aux_history_start)
                }
                None if self.dspark_runtime.is_some() => return Err("提交 DSpark cache 时缺少 aux history".to_owned()),
                None => (None, 0),
            };
            self.ensure_terminal_slot()?;
            let cache_namespace = task.request.get("_zllm_cache_namespace").and_then(Value::as_str).map(str::to_owned);
            let inserted = self.terminal_states.insert(
                cache_id.clone(),
                task.cached_tokens,
                Glm52HeadState {
                    states: task.states,
                    last_hidden,
                    pending_tokens: Some(task.pending_token.into_iter().collect()),
                    mtp,
                    dspark_aux_history,
                    dspark_aux_history_start,
                    dspark_target_cache: task.dspark_target_cache,
                    cache_namespace,
                    info: info.clone(),
                },
            );
            if inserted {
                let committed = self
                    .link
                    .send_cache(task.stage_id, next_request_id, info.prompt_tokens)
                    .and_then(|()| if cancelled { self.wait_ready_after_cancel(task.stage_id, info.prompt_tokens) } else { self.wait_ready(task.stage_id, info.prompt_tokens) });
                if let Err(error) = committed {
                    let _ = self.terminal_states.take(&cache_id);
                    let _ = self.link.send_delete(next_request_id);
                    return Err(format!("提交下游 batch cache 失败: {error}"));
                }
                if cancelled {
                    eprintln!("[glm52-prefill-checkpoint] cache_id={cache_id} tokens={} request_id={}", info.prompt_tokens, task.request_id);
                }
                Some(info)
            } else {
                self.link.send_delete(task.stage_id).and_then(|()| self.wait_ready(task.stage_id, info.prompt_tokens))?;
                None
            }
        } else {
            let tokens = task.cached_tokens.len();
            self.link.send_delete(task.stage_id).and_then(|()| if cancelled { self.wait_ready_after_cancel(task.stage_id, tokens) } else { self.wait_ready(task.stage_id, tokens) })?;
            None
        };
        Ok(GenerationSummary { finish_reason: task.finish_reason, prompt_tokens: task.tokens.len(), completion_tokens: task.completion_tokens, cache, tool_calls: std::mem::take(&mut task.tool_stream.calls) })
    }
}
