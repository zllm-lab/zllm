//! GLM-5.2 批请求生命周期数据。执行与资源所有权仍由 rocm_node 持有。

use super::*;

pub(super) struct Glm52BatchTask {
    pub(super) request_id: String,
    pub(super) stage_id: RequestId,
    pub(super) request: Value,
    pub(super) cancellation: Arc<AtomicBool>,
    pub(super) _batch_guard: BatchTokenGuard,
    pub(super) tokens: Vec<u32>,
    pub(super) max_decode: usize,
    pub(super) kv_reservation: ResidencyReservation,
    pub(super) states: Vec<Glm52StageState>,
    pub(super) last_hidden: Option<RocmTensor>,
    pub(super) prompt_last_hidden: Option<RocmTensor>,
    pub(super) cached_tokens: Vec<u32>,
    pub(super) prefill_position: usize,
    pub(super) tail_prefill_position: usize,
    pub(super) prefill_suffix_start: usize,
    pub(super) prefill_policy: AdaptiveChunkPolicy,
    pub(super) open_ready_tokens: Option<usize>,
    pub(super) response_text: String,
    pub(super) utf8: Utf8StreamDecoder,
    pub(super) think_filter: ThinkTagFilter,
    pub(super) tool_stream: GlmToolCallStream,
    pub(super) completion_tokens: usize,
    pub(super) pending_token: Option<u32>,
    pub(super) thinking_tokens: usize,
    pub(super) thinking_end_token: Option<u32>,
    pub(super) thinking_token_budget: Option<usize>,
    pub(super) finish_reason: String,
    pub(super) sampling: SamplingState,
    pub(super) token_fence: GenerationGuard<Glm52ToolFence>,
    pub(super) mtp: Option<RocmMtpSession>,
    pub(super) cached_output_ready: bool,
    pub(super) pending_verify_rows: usize,
    pub(super) dspark_aux_history: Option<RocmTensor>,
    pub(super) dspark_aux_history_start: usize,
    pub(super) prompt_dspark_aux_history: Option<RocmTensor>,
    pub(super) prompt_dspark_aux_history_start: usize,
    pub(super) dspark_target_cache: DsparkTargetCache<RocmTensor>,
    pub(super) dspark_cpu_target_cache: DsparkTargetCache<CpuTensor>,
    pub(super) dspark_cpu_pending: Option<u64>,
    /// CPU proposal 计算期间，anchor verifier 已经先进入 target 流水线。
    /// proposal 返回后只追加 draft suffix，尾端仍按完整 verify 一次验收。
    pub(super) dspark_cpu_anchor_in_flight: bool,
    pub(super) dspark_verify_inputs: Vec<u32>,
    pub(super) dspark_verify_hidden: Vec<RocmTensor>,
    pub(super) dspark_verify_aux: Vec<RocmTensor>,
    pub(super) dspark_target_rounds: usize,
    pub(super) dspark_verify_rounds: usize,
    pub(super) dspark_verified_drafts: usize,
    pub(super) dspark_accepted_drafts: usize,
    pub(super) dspark_verified_by_depth: Vec<usize>,
    pub(super) dspark_accepted_by_depth: Vec<usize>,
}

pub(super) enum Glm52NextWork {
    Decode(u32),
    Verify(Vec<u32>),
    Finish,
}

pub(super) struct Glm52BatchDecision {
    pub(super) next: Option<Glm52NextWork>,
}

pub(super) struct Glm52TailReady {
    pub(super) session: usize,
    pub(super) position: usize,
    pub(super) hidden: RocmTensor,
    pub(super) aux_hidden: Option<RocmTensor>,
    pub(super) aux_taps: usize,
    pub(super) decode: bool,
    pub(super) verify: bool,
    pub(super) cached: bool,
    pub(super) sampled_token: Option<(u32, bool)>,
    pub(super) speculative: Option<(Vec<u32>, usize, Vec<u32>, bool)>,
}

pub(super) struct Glm52TailOutcome {
    pub(super) tokens: Vec<u32>,
    pub(super) drafts: Vec<u32>,
    pub(super) eos: bool,
    pub(super) hard_loop: Option<LoopKind>,
}

pub(super) struct Glm52PendingTask {
    pub(super) input: NodeBatchRequest,
    pub(super) tokens: Vec<u32>,
    pub(super) max_tokens: usize,
    pub(super) reserved_tokens: usize,
    pub(super) sampling: SamplingConfig,
    pub(super) thinking_end_token: Option<u32>,
    pub(super) thinking_token_budget: Option<usize>,
    pub(super) swap_prefetch: Glm52SwapPrefetch,
}

pub(super) type Glm52SwapPrefetchResult = Result<Option<(Glm52CacheSnapshot, Option<usize>)>, String>;

/// SSD 只在 host 线程并行读取；GPU session 的创建和 H2D 恢复仍由引擎线程顺序执行。
/// `Ready` 缓存 try_recv 的结果，避免动态 intake 为检查就绪状态而丢失 channel 值。
pub(super) enum Glm52SwapPrefetch {
    Disabled,
    Loading(std::sync::mpsc::Receiver<Glm52SwapPrefetchResult>),
    Ready(Glm52SwapPrefetchResult),
}

impl Glm52SwapPrefetch {
    pub(super) fn poll_ready(&mut self) -> bool {
        let Self::Loading(receiver) = self else { return true };
        match receiver.try_recv() {
            Ok(result) => {
                *self = Self::Ready(result);
                true
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                *self = Self::Ready(Err("GLM-5.2 SSD prefetch 线程提前退出".to_owned()));
                true
            }
        }
    }

    pub(super) fn finish(self) -> Option<Glm52SwapPrefetchResult> {
        match self {
            Self::Disabled => None,
            Self::Ready(result) => Some(result),
            Self::Loading(receiver) => Some(receiver.recv().unwrap_or_else(|_| Err("GLM-5.2 SSD prefetch 线程提前退出".to_owned()))),
        }
    }
}
