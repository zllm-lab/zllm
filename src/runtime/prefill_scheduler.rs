//! 模型无关的多 stage 队列、completion 与 session 生命周期调度。

use std::{
    collections::VecDeque,
    sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(all(target_os = "linux", feature = "with-rocm"))]
use std::{
    fs::File,
    io::{BufWriter, Write},
};

use crate::backend::{BackendError, StageExecutionBackend, StageSubmissionKind};

#[cfg(test)]
use super::prefill::{AdaptiveChunkPolicy, run_chunked_prefill};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageWorkKind {
    Decode,
    Prefill,
}

static TRACE_STAGE_EVENTS: AtomicBool = AtomicBool::new(false);
static TRACE_STAGE_SENDER: OnceLock<mpsc::SyncSender<String>> = OnceLock::new();
static TRACE_STAGE_DROPPED: AtomicU64 = AtomicU64::new(0);

/// 显式启用一次进程级 stage 事件追踪。配置只在诊断进程使用，进程重启即恢复关闭。
#[cfg(all(target_os = "linux", feature = "with-rocm"))]
pub(crate) fn enable_stage_event_trace() {
    TRACE_STAGE_SENDER.get_or_init(|| {
        let path = format!("/tmp/zllm-stage-trace-{}.log", std::process::id());
        let (sender, receiver) = mpsc::sync_channel::<String>(65_536);
        std::thread::Builder::new()
            .name("zllm-stage-trace".to_owned())
            .spawn(move || {
                let Ok(file) = File::create(path) else {
                    return;
                };
                let mut writer = BufWriter::with_capacity(1 << 20, file);
                while let Ok(line) = receiver.recv() {
                    let _ = writeln!(writer, "{line}");
                    while let Ok(line) = receiver.try_recv() {
                        let _ = writeln!(writer, "{line}");
                    }
                    let _ = writer.flush();
                }
            })
            .expect("启动 stage trace writer");
        eprintln!("[stage-trace] path=/tmp/zllm-stage-trace-{}.log", std::process::id());
        sender
    });
    TRACE_STAGE_EVENTS.store(true, Ordering::Release);
}

pub(crate) fn stage_event_trace_enabled() -> bool {
    TRACE_STAGE_EVENTS.load(Ordering::Acquire)
}

pub(crate) fn stage_trace_timestamp_us() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_micros()
}

/// 诊断事件只允许无等待投递；磁盘慢或消费者落后时丢 trace，不能反向阻塞计算线程。
pub(crate) fn record_stage_trace(line: String) {
    let Some(sender) = TRACE_STAGE_SENDER.get() else {
        return;
    };
    if sender.try_send(line).is_err() {
        TRACE_STAGE_DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// stage scheduler 的模型无关设备时间快照。时间来自 backend completion event
/// 从提交到完成的区间；空泡是同一 stage 无在途工作到下一次提交之间的区间。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageFlowSnapshot {
    pub decode_micros: u64,
    pub prefill_micros: u64,
    pub prefill_work_units: u64,
    pub prefill_batches: u64,
    pub idle_micros: u64,
    pub prefill_idle_micros: u64,
    pub prefill_active: u64,
    pub prefill_idle_pending: u64,
}

#[derive(Default)]
struct StageFlowMetrics {
    decode_micros: std::sync::atomic::AtomicU64,
    prefill_micros: std::sync::atomic::AtomicU64,
    prefill_work_units: std::sync::atomic::AtomicU64,
    prefill_batches: std::sync::atomic::AtomicU64,
    idle_micros: std::sync::atomic::AtomicU64,
    prefill_idle_micros: std::sync::atomic::AtomicU64,
    prefill_active: std::sync::atomic::AtomicU64,
    prefill_epoch: std::sync::atomic::AtomicU64,
    prefill_idle_pending: std::sync::atomic::AtomicU64,
}

impl StageFlowMetrics {
    fn new(_stage_count: usize) -> Self {
        Self::default()
    }

    fn add(target: &std::sync::atomic::AtomicU64, value: u64) {
        let _ = target.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |current| Some(current.saturating_add(value)));
    }

    fn record_work(&self, kind: StageWorkKind, work_units: usize, micros: u64, decode_uncontended: bool) {
        match kind {
            StageWorkKind::Decode if decode_uncontended => Self::add(&self.decode_micros, micros),
            StageWorkKind::Decode => {}
            StageWorkKind::Prefill => {
                Self::add(&self.prefill_micros, micros);
                if work_units != 0 {
                    Self::add(&self.prefill_work_units, u64::try_from(work_units).unwrap_or(u64::MAX));
                    Self::add(&self.prefill_batches, 1);
                }
            }
        }
    }

    fn begin_prefill(&self) {
        let _ = self.prefill_epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self::add(&self.prefill_active, 1);
    }

    fn finish_prefill(&self) {
        let _ = self.prefill_active.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |active| Some(active.saturating_sub(1)));
    }

    fn prefill_active(&self) -> bool {
        self.prefill_active.load(std::sync::atomic::Ordering::Relaxed) != 0
    }

    fn prefill_epoch(&self) -> u64 {
        self.prefill_epoch.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn record_idle(&self, micros: u64) {
        Self::add(&self.idle_micros, micros);
    }

    fn record_prefill_idle(&self, micros: u64) {
        Self::add(&self.prefill_idle_micros, micros);
    }

    fn enter_idle(&self, contaminated: bool) {
        if contaminated {
            Self::add(&self.prefill_idle_pending, 1);
        }
    }

    fn leave_idle(&self, contaminated: bool) {
        if contaminated {
            let _ = self.prefill_idle_pending.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |count| Some(count.saturating_sub(1)));
        }
    }

    fn snapshot(&self) -> StageFlowSnapshot {
        let ordering = std::sync::atomic::Ordering::Relaxed;
        StageFlowSnapshot {
            decode_micros: self.decode_micros.load(ordering),
            prefill_micros: self.prefill_micros.load(ordering),
            prefill_work_units: self.prefill_work_units.load(ordering),
            prefill_batches: self.prefill_batches.load(ordering),
            // 只记录所有 stage 同时空闲的交集；错峰 idle 不能承载完整 pipeline work。
            idle_micros: self.idle_micros.load(ordering),
            prefill_idle_micros: self.prefill_idle_micros.load(ordering),
            prefill_active: self.prefill_active.load(ordering),
            prefill_idle_pending: self.prefill_idle_pending.load(ordering),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StageSchedulerConfig {
    pub session_capacity: usize,
    pub batch_work_limit: usize,
    pub execution_slots: usize,
    pub decode_execution_slots: usize,
    pub pipeline_work_window: usize,
    pub prefill_admission_burst: usize,
    pub decode_batch_limit: usize,
    pub prefill_batch_limit: usize,
    pub profile_completion: bool,
}

impl StageSchedulerConfig {
    fn validate(self, session_count: usize, stage_count: usize) -> Result<Self, BackendError> {
        if session_count == 0
            || stage_count == 0
            || session_count > self.session_capacity
            || self.batch_work_limit == 0
            || self.execution_slots == 0
            || self.decode_execution_slots == 0
            || self.pipeline_work_window == 0
            || self.prefill_admission_burst == 0
            || self.decode_batch_limit == 0
            || self.prefill_batch_limit == 0
        {
            return Err(BackendError::Compute {
                msg: format!(
                    "stage scheduler 配置非法: sessions={session_count}/{} stages={stage_count} work_limit={} slots={}/{} window={} prefill_burst={} decode_batch={} prefill_batch={}",
                    self.session_capacity, self.batch_work_limit, self.execution_slots, self.decode_execution_slots, self.pipeline_work_window, self.prefill_admission_burst, self.decode_batch_limit, self.prefill_batch_limit,
                ),
            });
        }
        Ok(self)
    }

    fn batch_limit(self, kind: StageWorkKind) -> usize {
        match kind {
            StageWorkKind::Decode => self.decode_batch_limit,
            StageWorkKind::Prefill => self.prefill_batch_limit,
        }
        .min(self.session_capacity)
    }

    fn execution_slots(self, kind: StageWorkKind) -> usize {
        match kind {
            StageWorkKind::Decode => self.decode_execution_slots,
            StageWorkKind::Prefill => self.execution_slots,
        }
    }
}

pub enum StageSchedulerOutput<T, S> {
    Work { cohort: Option<(u64, usize)>, session: usize, position: usize, value: T },
    Opened { session: usize },
    Closed { session: usize, states: Vec<S> },
}

enum StageSchedulerMessage<T, S> {
    Work { queued_at: Instant, cohort: Option<(u64, usize)>, wave: Option<(u64, usize)>, items: Vec<(usize, usize, T)> },
    Open { session: usize, states: VecDeque<S> },
    Cancel { session: usize },
    Close { session: usize, states: Vec<S> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueuedGroup {
    id: u64,
    size: usize,
    batch: bool,
}

struct StageSessionControl<S> {
    occupied: bool,
    deferred: bool,
    in_flight: bool,
    cancelled: bool,
    pending_work: usize,
    pending_close: Option<Vec<S>>,
    order: VecDeque<StageWorkKind>,
}

struct StageSubmissionTiming {
    sessions: usize,
    work_units: usize,
    position_min: usize,
    position_max: usize,
    started: Instant,
    submit_micros: u64,
    prefill_epoch: u64,
    trace_lanes: Option<String>,
    trace_dispatch_us: Option<u128>,
    trace_queue_micros: u64,
    trace_run_micros: u64,
    trace_completion_record_micros: u64,
    trace_queue_after: usize,
}

struct StageInFlight<C, T> {
    completion: C,
    output: Option<(Option<QueuedGroup>, Vec<(usize, usize, T)>)>,
    sessions: Vec<usize>,
    kind: StageWorkKind,
    timing: Option<StageSubmissionTiming>,
}

fn ready_stage_completion<B, T>(backend: &B, in_flight: &VecDeque<StageInFlight<B::Completion, T>>, concurrent_submissions: bool) -> Result<Option<usize>, BackendError>
where
    B: StageExecutionBackend,
{
    if !concurrent_submissions {
        return in_flight.front().map(|item| backend.stage_completion_ready(&item.completion).map(|ready| ready.then_some(0))).transpose().map(Option::flatten);
    }
    // latency completion 即使后提交也先退休；同一 session 仍由 control.in_flight
    // 保证只有一份工作在途，所以这里只改变不同 session 之间的完成顺序。
    for kind in [StageWorkKind::Decode, StageWorkKind::Prefill] {
        for (index, item) in in_flight.iter().enumerate().filter(|(_, item)| item.kind == kind) {
            if backend.stage_completion_ready(&item.completion)? {
                return Ok(Some(index));
            }
        }
    }
    Ok(None)
}

fn stage_completion_to_wait<C, T>(in_flight: &VecDeque<StageInFlight<C, T>>, concurrent_submissions: bool) -> Option<&C> {
    if concurrent_submissions { in_flight.iter().find(|item| item.kind == StageWorkKind::Decode).or_else(|| in_flight.front()).map(|item| &item.completion) } else { in_flight.front().map(|item| &item.completion) }
}

fn should_wait_for_stage_completion(concurrent_submissions: bool, in_flight: usize, execution_window: usize, decode_in_flight: usize) -> bool {
    !concurrent_submissions || in_flight >= execution_window || decode_in_flight != 0
}

fn concurrent_stage_window(supported: bool, prefill_queued: bool, prefill_in_flight: usize) -> bool {
    supported && (prefill_queued || prefill_in_flight != 0)
}

impl<S> Default for StageSessionControl<S> {
    fn default() -> Self {
        Self { occupied: false, deferred: false, in_flight: false, cancelled: false, pending_work: 0, pending_close: None, order: VecDeque::new() }
    }
}

impl QueuedGroup {
    fn message_parts(self) -> (Option<(u64, usize)>, Option<(u64, usize)>) {
        let group = Some((self.id, self.size));
        if self.batch { (group, None) } else { (None, group) }
    }
}

fn recover_scheduler_state<S>(recovered: &mut [Vec<Option<S>>], recovery_error: &mut Option<BackendError>, session: usize, stage: usize, state: S) {
    let Some(slot) = recovered.get_mut(session).and_then(|states| states.get_mut(stage)) else {
        recovery_error.get_or_insert_with(|| BackendError::Compute { msg: format!("stage scheduler 回收 state 越界: session={session} stage={stage}") });
        return;
    };
    if slot.is_some() {
        recovery_error.get_or_insert_with(|| BackendError::Compute { msg: format!("stage scheduler 重复回收 state: session={session} stage={stage}") });
        return;
    }
    *slot = Some(state);
}

fn recover_scheduler_control<T, S>(recovered: &mut [Vec<Option<S>>], recovery_error: &mut Option<BackendError>, next_stage: usize, control: StageSchedulerMessage<T, S>) {
    match control {
        StageSchedulerMessage::Open { session, states } => {
            for (offset, state) in states.into_iter().enumerate() {
                recover_scheduler_state(recovered, recovery_error, session, next_stage + offset, state);
            }
        }
        StageSchedulerMessage::Close { session, states } => {
            for (stage, state) in states.into_iter().enumerate() {
                recover_scheduler_state(recovered, recovery_error, session, stage, state);
            }
        }
        StageSchedulerMessage::Work { .. } | StageSchedulerMessage::Cancel { .. } => {}
    }
}

fn is_scheduler_transport_exit(error: &BackendError) -> bool {
    matches!(
        error,
        BackendError::Compute { msg }
            if msg == "stage scheduler 输入 stage 已退出" || msg == "stage scheduler 输出 stage 已退出"
    )
}

/// 调度器的 session 槽位表：按下标取 session，`None` = 已关闭，
/// `Some` = 该 session 的各 stage state。
pub type SchedulerSessions<S> = Vec<Option<Vec<S>>>;

/// 驱动端只提交真实工作；调度器不会等待凑 batch。
pub struct StageSchedulerHandle<T, S> {
    input: std::sync::mpsc::Sender<Result<StageSchedulerMessage<T, S>, BackendError>>,
    output: std::sync::mpsc::Receiver<Result<StageSchedulerMessage<T, S>, BackendError>>,
    pending: std::sync::Mutex<VecDeque<StageSchedulerOutput<T, S>>>,
    next_ready_cohort: std::sync::atomic::AtomicU64,
    session_capacity: usize,
    stage_count: usize,
    pipeline_work_window: usize,
    prefill_admission_burst: usize,
    decode_batch_limit: usize,
    stage_flow: std::sync::Arc<StageFlowMetrics>,
}

pub use super::prefill_admission::{OpportunisticPrefillAdmission, PrefillAdmissionCursor};

/// 把一段连续 token 均衡切成不超过 `max_segments` 个有序 work item。
/// 调用方仍决定 admission 大小；这里不等待、不补齐，也不改变 token 总量。
pub fn prefill_token_segments(start: usize, len: usize, max_segments: usize) -> Vec<std::ops::Range<usize>> {
    if len == 0 {
        return Vec::new();
    }
    let count = max_segments.max(1).min(len);
    let width = len / count;
    let extra = len % count;
    let mut cursor = start;
    (0..count)
        .map(|segment| {
            let next = cursor + width + usize::from(segment < extra);
            let range = cursor..next;
            cursor = next;
            range
        })
        .collect()
}

fn choose_stage_work_kind(decode_available: bool, prefill_available: bool, decode_dispatches: usize, prefill_max_wait: usize) -> Option<StageWorkKind> {
    if prefill_available && (!decode_available || decode_dispatches >= prefill_max_wait.max(1)) {
        Some(StageWorkKind::Prefill)
    } else if decode_available {
        Some(StageWorkKind::Decode)
    } else if prefill_available {
        Some(StageWorkKind::Prefill)
    } else {
        None
    }
}

impl<T, S> StageSchedulerHandle<T, S> {
    /// 填满全部 stage execution slot 所需的模型无关工作窗口。
    pub fn pipeline_work_window(&self) -> usize {
        self.pipeline_work_window
    }

    pub fn stage_count(&self) -> usize {
        self.stage_count
    }

    /// 同一会话连续 admission 与在途 prefill chunk 的上限；不改变 stage batch。
    pub fn prefill_admission_burst(&self) -> usize {
        self.prefill_admission_burst
    }

    pub fn stage_flow_snapshot(&self) -> StageFlowSnapshot {
        self.stage_flow.snapshot()
    }

    /// 从最老会话开始选择下一份 prefill；连续 burst 到达上限或会话完成后再轮转。
    pub fn next_prefill_session<F>(&self, cursor: &mut PrefillAdmissionCursor, session_count: usize, mut has_work: F) -> Option<usize>
    where
        F: FnMut(usize) -> bool,
    {
        if session_count == 0 {
            return None;
        }
        for _ in 0..session_count {
            let session = cursor.session % session_count;
            if has_work(session) {
                return Some(session);
            }
            cursor.session = (session + 1) % session_count;
            cursor.submitted = 0;
        }
        None
    }

    pub fn commit_prefill_submission(&self, cursor: &mut PrefillAdmissionCursor, session: usize, session_complete: bool) {
        cursor.submitted += 1;
        if session_complete || cursor.submitted >= self.prefill_admission_burst {
            cursor.session = (session + 1) % self.session_capacity;
            cursor.submitted = 0;
        }
    }

    pub fn submit(&self, session: usize, position: usize, value: T) -> Result<(), BackendError> {
        self.submit_many(std::iter::once((session, position, value)))
    }

    /// 原子入队一组已经同时就绪的真实工作，不等待后续工作凑批。
    pub fn submit_many<I>(&self, work: I) -> Result<(), BackendError>
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        let work = work.into_iter().collect::<Vec<_>>();
        if let Some((session, _, _)) = work.iter().find(|(session, _, _)| *session >= self.session_capacity) {
            return Err(BackendError::Compute { msg: format!("stage scheduler work session={session} 越界，capacity={}", self.session_capacity) });
        }
        if work.is_empty() {
            return Ok(());
        }
        self.input.send(Ok(StageSchedulerMessage::Work { queued_at: Instant::now(), cohort: None, wave: None, items: work })).map_err(|_| BackendError::Compute { msg: "stage scheduler 输入 stage 已退出".to_owned() })
    }

    /// 把调用方此刻已经就绪的 decode 工作作为一个原子 wave 穿过全部 stage。
    /// 不等待后续工作凑批；高位 id 空间与跨进程显式 cohort 分离。
    pub fn submit_ready_cohort<I>(&self, work: I) -> Result<(u64, usize), BackendError>
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        let work = work.into_iter().collect::<Vec<_>>();
        if work.is_empty() {
            return Err(BackendError::Compute { msg: "stage scheduler ready cohort 不能为空".to_owned() });
        }
        let cohort_size = work.len();
        let cohort = self.next_ready_cohort.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.submit_cohort(cohort, cohort_size, work)?;
        Ok((cohort, cohort_size))
    }

    /// 把此刻已经就绪的 decode 行作为有序 wave 穿过全部 stage。wave 只保持
    /// 成员邻接和尾部聚合标识，每个成员仍以单行 submission 立即交给下一 stage；
    /// 因而不会切换到多行 prefill kernel，也不会用整批 barrier 排空流水线。
    pub fn submit_ready_wave<I>(&self, work: I) -> Result<(u64, usize), BackendError>
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        let work = work.into_iter().collect::<Vec<_>>();
        if work.is_empty() || work.len() > self.decode_batch_limit {
            return Err(BackendError::Compute { msg: format!("stage scheduler ready wave 大小非法: {}/{}", work.len(), self.decode_batch_limit) });
        }
        if let Some((session, _, _)) = work.iter().find(|(session, _, _)| *session >= self.session_capacity) {
            return Err(BackendError::Compute { msg: format!("stage scheduler ready wave session={session} 越界，capacity={}", self.session_capacity) });
        }
        let size = work.len();
        let id = self.next_ready_cohort.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.input.send(Ok(StageSchedulerMessage::Work { queued_at: Instant::now(), cohort: None, wave: Some((id, size)), items: work })).map_err(|_| BackendError::Compute { msg: "stage scheduler 输入 stage 已退出".to_owned() })?;
        Ok((id, size))
    }

    /// 接收跨进程 wave 的一个成员。成员无需等待同 wave 的其余行到齐，传入顺序
    /// 就是 stage 顺序；尾部仍可用同一 id/size 收口共享 output head。
    pub fn submit_wave_member(&self, wave: u64, wave_size: usize, session: usize, position: usize, value: T) -> Result<(), BackendError> {
        if wave == 0 || wave_size == 0 || wave_size > self.decode_batch_limit || session >= self.session_capacity {
            return Err(BackendError::Compute { msg: format!("stage scheduler wave 参数非法: id={wave} size={wave_size}/{} session={session}/{}", self.decode_batch_limit, self.session_capacity) });
        }
        self.input
            .send(Ok(StageSchedulerMessage::Work { queued_at: Instant::now(), cohort: None, wave: Some((wave, wave_size)), items: vec![(session, position, value)] }))
            .map_err(|_| BackendError::Compute { msg: "stage scheduler 输入 stage 已退出".to_owned() })
    }

    /// 显式 cohort 在全部 stage 保持原子分组；编号 0 留给不要求保持分组的工作。
    pub fn submit_cohort<I>(&self, cohort: u64, cohort_size: usize, work: I) -> Result<(), BackendError>
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        if cohort == 0 || cohort_size == 0 || cohort_size > self.decode_batch_limit {
            return Err(BackendError::Compute { msg: format!("stage scheduler cohort 参数非法: id={cohort} size={cohort_size}/{}", self.decode_batch_limit) });
        }
        let work = work.into_iter().collect::<Vec<_>>();
        if let Some((session, _, _)) = work.iter().find(|(session, _, _)| *session >= self.session_capacity) {
            return Err(BackendError::Compute { msg: format!("stage scheduler cohort={cohort} session={session} 越界，capacity={}", self.session_capacity) });
        }
        if work.is_empty() {
            return Ok(());
        }
        self.input
            .send(Ok(StageSchedulerMessage::Work { queued_at: Instant::now(), cohort: Some((cohort, cohort_size)), wave: None, items: work }))
            .map_err(|_| BackendError::Compute { msg: "stage scheduler 输入 stage 已退出".to_owned() })
    }

    pub fn open(&self, session: usize, states: Vec<S>) -> Result<(), BackendError> {
        if session >= self.session_capacity || states.len() != self.stage_count {
            return Err(BackendError::Compute { msg: format!("stage scheduler Open 参数非法: session={session}/{} states={}/{}", self.session_capacity, states.len(), self.stage_count,) });
        }
        self.input.send(Ok(StageSchedulerMessage::Open { session, states: states.into() })).map_err(|_| BackendError::Compute { msg: "stage scheduler 输入 stage 已退出".to_owned() })
    }

    pub fn close(&self, session: usize) -> Result<(), BackendError> {
        if session >= self.session_capacity {
            return Err(BackendError::Compute { msg: format!("stage scheduler Close session={session} 越界，capacity={}", self.session_capacity) });
        }
        self.input.send(Ok(StageSchedulerMessage::Close { session, states: Vec::with_capacity(self.stage_count) })).map_err(|_| BackendError::Compute { msg: "stage scheduler 输入 stage 已退出".to_owned() })
    }

    /// 撤销该 session 尚未发射的工作；已经提交到 backend 的工作仍由 completion
    /// 正常回收。调用方随后 `close`，即可在最后一份在途工作退休后安全取回 state。
    pub fn cancel(&self, session: usize) -> Result<(), BackendError> {
        if session >= self.session_capacity {
            return Err(BackendError::Compute { msg: format!("stage scheduler Cancel session={session} 越界，capacity={}", self.session_capacity) });
        }
        self.input.send(Ok(StageSchedulerMessage::Cancel { session })).map_err(|_| BackendError::Compute { msg: "stage scheduler 输入 stage 已退出".to_owned() })
    }

    pub fn try_recv(&self) -> Result<Option<StageSchedulerOutput<T, S>>, BackendError> {
        // guard 必须在 pop_front 后立即释放:后续 match 分支会再次加锁
        // pending,长期持锁曾在同线程重入时死锁。
        if let Some(output) = { self.pending.lock().map_err(|_| BackendError::Compute { msg: "stage scheduler pending 锁中毒".to_owned() })?.pop_front() } {
            return Ok(Some(output));
        }
        let message = match self.output.try_recv() {
            Ok(message) => message?,
            Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(BackendError::Compute { msg: "stage scheduler 输出 stage 已退出".to_owned() });
            }
        };
        match message {
            StageSchedulerMessage::Work { cohort, wave, items, .. } if !items.is_empty() => {
                let mut items = items.into_iter();
                let (session, position, value) = items.next().expect("已检查非空");
                let group = wave.or(cohort);
                self.pending.lock().map_err(|_| BackendError::Compute { msg: "stage scheduler pending 锁中毒".to_owned() })?.extend(items.map(|(session, position, value)| StageSchedulerOutput::Work {
                    cohort: group,
                    session,
                    position,
                    value,
                }));
                Ok(Some(StageSchedulerOutput::Work { cohort: group, session, position, value }))
            }
            StageSchedulerMessage::Open { session, states } if states.is_empty() => Ok(Some(StageSchedulerOutput::Opened { session })),
            StageSchedulerMessage::Close { session, states } if states.len() == self.stage_count => Ok(Some(StageSchedulerOutput::Closed { session, states })),
            StageSchedulerMessage::Cancel { session } => Err(BackendError::Compute { msg: format!("stage scheduler Cancel session={session} 不应到达输出端") }),
            StageSchedulerMessage::Work { .. } => Err(BackendError::Compute { msg: "stage scheduler 输出 work 为空".to_owned() }),
            StageSchedulerMessage::Open { session, states } => Err(BackendError::Compute { msg: format!("stage scheduler Open session={session} 遗留 {} 个 state", states.len()) }),
            StageSchedulerMessage::Close { session, states } => Err(BackendError::Compute { msg: format!("stage scheduler Close session={session} 只返回 {}/{} 个 state", states.len(), self.stage_count,) }),
        }
    }

    /// 返回当前已经到达输出端的全部结果；第一项未就绪时立即返回空集。
    pub fn try_recv_ready(&self) -> Result<Vec<StageSchedulerOutput<T, S>>, BackendError> {
        let mut ready = Vec::new();
        loop {
            match self.try_recv() {
                Ok(Some(output)) => ready.push(output),
                Ok(None) => return Ok(ready),
                Err(error) => {
                    // Closed 携带该 session 唯一一份 state；后续输出报错时必须把
                    // 已取出的结果放回 handle，交给 recoverable 入口统一回收。
                    let mut pending = self.pending.lock().map_err(|_| BackendError::Compute { msg: "stage scheduler pending 锁中毒".to_owned() })?;
                    while let Some(output) = ready.pop() {
                        pending.push_front(output);
                    }
                    return Err(error);
                }
            }
        }
    }

    /// 只等待第一项真实完成，随后立即排空当前 ready 输出，不等待更多结果。
    pub fn recv_ready(&self) -> Result<Vec<StageSchedulerOutput<T, S>>, BackendError> {
        loop {
            let ready = self.try_recv_ready()?;
            if !ready.is_empty() {
                return Ok(ready);
            }
            std::thread::yield_now();
        }
    }

    /// 测试和 recoverable 协调路径等待首个错误；期间已经到达的 Close/Open/Work
    /// 不转移所有权，错误出现时统一放回 pending，供外层回收。
    #[cfg(test)]
    fn recv_until_error(&self) -> BackendError {
        let mut received = Vec::new();
        loop {
            match self.try_recv_ready() {
                Ok(mut ready) => {
                    received.append(&mut ready);
                    std::thread::yield_now();
                }
                Err(error) => {
                    let mut pending = self.pending.lock().expect("stage scheduler pending 锁中毒");
                    while let Some(output) = received.pop() {
                        pending.push_front(output);
                    }
                    return error;
                }
            }
        }
    }
}

/// 忙等退避：先让出 CPU 若干次，持续无进展时升级为短睡眠（50µs 起步、上限
/// 200µs），避免 worker 在队列空转或执行槽满时占满一个核。有进展时必须 `reset`，
/// 让下一次等待从低延迟的 yield 重新开始。
struct SpinBackoff {
    idle_rounds: u32,
}

impl SpinBackoff {
    const YIELD_ROUNDS: u32 = 16;

    fn new() -> Self {
        Self { idle_rounds: 0 }
    }

    fn reset(&mut self) {
        self.idle_rounds = 0;
    }

    fn wait(&mut self) {
        self.idle_rounds = self.idle_rounds.saturating_add(1);
        if self.idle_rounds <= Self::YIELD_ROUNDS {
            std::thread::yield_now();
            return;
        }
        // 睡眠时长按空转轮次翻倍并封顶，长尾等待让出 CPU，短等待仍保持低延迟。
        let micros = 50_u64.checked_shl((self.idle_rounds - Self::YIELD_ROUNDS - 1).min(2)).unwrap_or(200).min(200);
        std::thread::sleep(Duration::from_micros(micros));
    }
}

/// 单份工作没有 backlog，不需要为每个 stage 创建线程和队列。
///
/// 各 stage 仍先通过 backend 的 ordered handoff 提交完整依赖链，最后只等待链尾
/// completion；链尾完成已经蕴含此前 stage 完成。连续多路和真实 backlog 继续由
/// `drive_stage_scheduler` 管理。
pub fn run_single_stage_chain<T, S, B, F>(backends: &[B], states: Vec<S>, position: usize, value: T, run_batch: F) -> Result<(usize, T, Vec<S>), BackendError>
where
    B: StageExecutionBackend,
    F: Fn(&B, &mut [Option<S>], usize, Vec<(usize, usize, T)>) -> Result<Vec<(usize, usize, T)>, BackendError>,
{
    if backends.is_empty() || backends.len() != states.len() {
        return Err(BackendError::Compute { msg: format!("single stage chain 参数非法: backends={} states={}", backends.len(), states.len()) });
    }
    let mut states = states.into_iter().map(|state| vec![Some(state)]).collect::<Vec<_>>();
    let mut batch = vec![(0, position, value)];
    let mut completions = Vec::with_capacity(backends.len());
    for (stage, backend) in backends.iter().enumerate() {
        backend.activate_stage_submission(StageSubmissionKind::Latency)?;
        backend.begin_stage_submission()?;
        batch = match run_batch(backend, &mut states[stage], stage, batch) {
            Ok(batch) => batch,
            Err(error) => {
                let _ = backend.abort_stage_submission();
                return Err(error);
            }
        };
        match backend.record_stage_completion() {
            Ok(completion) => completions.push(completion),
            Err(error) => {
                let _ = backend.abort_stage_submission();
                return Err(error);
            }
        }
    }
    let backend = backends.last().expect("已检查非空");
    let completion = completions.last().expect("每个 stage 均记录 completion");
    let mut backoff = SpinBackoff::new();
    while !backend.stage_completion_ready(completion)? {
        backoff.wait();
    }
    for (backend, completion) in backends.iter().zip(&completions) {
        backend.retire_ordered_stage_completion(completion)?;
    }
    let (session, position, value) = batch.pop().ok_or_else(|| BackendError::Compute { msg: "single stage chain 没有输出".to_owned() })?;
    if session != 0 || !batch.is_empty() {
        return Err(BackendError::Compute { msg: format!("single stage chain 输出非法: session={session} remaining={}", batch.len()) });
    }
    let states = states.into_iter().enumerate().map(|(stage, mut states)| states.pop().flatten().ok_or_else(|| BackendError::Compute { msg: format!("single stage chain stage={stage} 丢失 state") })).collect::<Result<Vec<_>, _>>()?;
    Ok((position, value, states))
}

/// 运行常驻的事件驱动 stage scheduler。
///
/// run_batch 只负责模型计算；调度器在它返回后通过对应 backend 记录 completion。
/// drive 可随时 Open/Close session，或立即提交 decode/prefill 工作。
#[allow(clippy::too_many_arguments)]
pub fn drive_stage_scheduler<T, S, B, F, D, R, W, K>(backends: Vec<B>, sessions: Vec<Vec<S>>, config: StageSchedulerConfig, work_size: W, work_kind: K, run_batch: F, drive: D) -> Result<(R, SchedulerSessions<S>), BackendError>
where
    T: Send,
    S: Send,
    B: StageExecutionBackend + Sync,
    F: Fn(&B, &mut [Option<S>], usize, Vec<(usize, usize, T)>) -> Result<Vec<(usize, usize, T)>, BackendError> + Sync,
    D: FnOnce(&StageSchedulerHandle<T, S>) -> Result<R, BackendError>,
    W: Fn(&T) -> usize + Sync,
    K: Fn(&T) -> StageWorkKind + Sync,
{
    drive_stage_scheduler_with_batch_class(backends, sessions, config, work_size, work_kind, |_| 0, run_batch, drive)
}

/// `batch_class` 相同的 work 才能进入同一个物理 batch；0 表示没有额外约束。
/// 模型组合层用它表达 dtype/layout/kernel 几何兼容性，scheduler 不解释取值。
#[allow(clippy::too_many_arguments)]
pub fn drive_stage_scheduler_with_batch_class<T, S, B, F, D, R, W, K, C>(
    backends: Vec<B>,
    sessions: Vec<Vec<S>>,
    config: StageSchedulerConfig,
    work_size: W,
    work_kind: K,
    batch_class: C,
    run_batch: F,
    drive: D,
) -> Result<(R, SchedulerSessions<S>), BackendError>
where
    T: Send,
    S: Send,
    B: StageExecutionBackend + Sync,
    F: Fn(&B, &mut [Option<S>], usize, Vec<(usize, usize, T)>) -> Result<Vec<(usize, usize, T)>, BackendError> + Sync,
    D: FnOnce(&StageSchedulerHandle<T, S>) -> Result<R, BackendError>,
    W: Fn(&T) -> usize + Sync,
    K: Fn(&T) -> StageWorkKind + Sync,
    C: Fn(&T) -> usize + Sync,
{
    let (result, sessions) = drive_stage_scheduler_recoverable_with_batch_class(backends, sessions, config, work_size, work_kind, batch_class, run_batch, drive)?;
    result.map(|result| (result, sessions))
}

/// 与普通入口执行完全相同，但算子/驱动错误作为内层结果返回，确保调用方仍能
/// 取回各 stage 的 session state；只有装配错误或 worker panic 才丢失所有权。
pub fn drive_stage_scheduler_recoverable<T, S, B, F, D, R, W, K>(
    backends: Vec<B>,
    sessions: Vec<Vec<S>>,
    config: StageSchedulerConfig,
    work_size: W,
    work_kind: K,
    run_batch: F,
    drive: D,
) -> Result<(Result<R, BackendError>, SchedulerSessions<S>), BackendError>
where
    T: Send,
    S: Send,
    B: StageExecutionBackend + Sync,
    F: Fn(&B, &mut [Option<S>], usize, Vec<(usize, usize, T)>) -> Result<Vec<(usize, usize, T)>, BackendError> + Sync,
    D: FnOnce(&StageSchedulerHandle<T, S>) -> Result<R, BackendError>,
    W: Fn(&T) -> usize + Sync,
    K: Fn(&T) -> StageWorkKind + Sync,
{
    drive_stage_scheduler_recoverable_with_batch_class(backends, sessions, config, work_size, work_kind, |_| 0, run_batch, drive)
}

#[allow(clippy::too_many_arguments)]
pub fn drive_stage_scheduler_recoverable_with_batch_class<T, S, B, F, D, R, W, K, C>(
    backends: Vec<B>,
    sessions: Vec<Vec<S>>,
    config: StageSchedulerConfig,
    work_size: W,
    work_kind: K,
    batch_class: C,
    run_batch: F,
    drive: D,
) -> Result<(Result<R, BackendError>, SchedulerSessions<S>), BackendError>
where
    T: Send,
    S: Send,
    B: StageExecutionBackend + Sync,
    F: Fn(&B, &mut [Option<S>], usize, Vec<(usize, usize, T)>) -> Result<Vec<(usize, usize, T)>, BackendError> + Sync,
    D: FnOnce(&StageSchedulerHandle<T, S>) -> Result<R, BackendError>,
    W: Fn(&T) -> usize + Sync,
    K: Fn(&T) -> StageWorkKind + Sync,
    C: Fn(&T) -> usize + Sync,
{
    let session_count = sessions.len();
    let stage_count = backends.len();
    let config = config.validate(session_count, stage_count)?;
    if sessions.iter().any(|states| states.len() != stage_count) {
        return Err(BackendError::Compute { msg: format!("stage scheduler 的 session stage 数不一致: sessions={session_count} stages={stage_count}") });
    }

    let mut stage_states = (0..stage_count).map(|_| std::iter::repeat_with(|| None).take(config.session_capacity).collect::<Vec<Option<S>>>()).collect::<Vec<_>>();
    for (session, states) in sessions.into_iter().enumerate() {
        for (stage, state) in states.into_iter().enumerate() {
            stage_states[stage][session] = Some(state);
        }
    }

    std::thread::scope(|scope| {
        let (input, mut receiver) = std::sync::mpsc::channel::<Result<StageSchedulerMessage<T, S>, BackendError>>();
        let mut workers = Vec::with_capacity(stage_count);
        let stage_flow = std::sync::Arc::new(StageFlowMetrics::new(stage_count));
        for (stage, (backend, mut states)) in backends.iter().zip(stage_states).enumerate() {
            let (sender, next_receiver) = std::sync::mpsc::channel();
            let stage_receiver = receiver;
            let run_batch = &run_batch;
            let work_size = &work_size;
            let work_kind = &work_kind;
            let batch_class = &batch_class;
            let stage_flow_worker = std::sync::Arc::clone(&stage_flow);
            workers.push(scope.spawn(move || {
                let mut decode_queue = VecDeque::<(Option<QueuedGroup>, usize, usize, T, Instant)>::new();
                let mut prefill_queue = VecDeque::<(Option<QueuedGroup>, usize, usize, T, Instant)>::new();
                let mut active_sessions = states.iter().filter(|state| state.is_some()).count();
                let mut direct_submissions = [0_usize; 2];
                let concurrent_submissions = backend.supports_concurrent_stage_submissions();
                let decode_submission_slots = config.execution_slots(StageWorkKind::Decode).min(backend.max_queued_latency_submissions().max(1));
                let submission_slots = |kind| match kind {
                    StageWorkKind::Decode => decode_submission_slots,
                    StageWorkKind::Prefill => config.execution_slots(StageWorkKind::Prefill),
                };
                let execution_window = |concurrent| {
                    if concurrent {
                        submission_slots(StageWorkKind::Decode).saturating_add(submission_slots(StageWorkKind::Prefill))
                    } else {
                        submission_slots(StageWorkKind::Decode).max(submission_slots(StageWorkKind::Prefill))
                    }
                };
                let mut in_flight = VecDeque::<StageInFlight<B::Completion, T>>::new();
                let mut in_flight_by_kind = [0_usize; 2];
                let mut controls = std::iter::repeat_with(StageSessionControl::default).take(config.session_capacity).collect::<Vec<_>>();
                // decode/prefill 使用独立优先级队列，但同一 session 的 cache/state
                // 仍是单一有序状态机。显式保存入队顺序，并在 completion 前占住
                // session，防止后来的 decode 越过尚未退休的 prefill/verify。
                let mut cohort_sizes = std::collections::HashMap::<u64, (usize, usize, usize)>::new();
                let mut recovered_controls = Vec::<(usize, StageSchedulerMessage<T, S>)>::new();
                let mut input_closed = false;
                let mut failed = false;
                let mut filling_slots = false;
                let mut idle_started = None::<(Instant, bool, u64)>;
                let mut backoff = SpinBackoff::new();
                let profile_completion = config.profile_completion;
                let trace_events = stage_event_trace_enabled();
                let mut profile_batches = [0_usize; 2];
                let mut profile_sessions = [0_usize; 2];
                let mut profile_submit_micros = [0_u64; 2];
                let mut profile_total_micros = [0_u64; 2];
                // index 0 保留；其余项按本次实际 decode batch 行数累计
                // [batches, submit_us, total_us]，避免只看平均 sessions/batch
                // 掩盖某个多行 kernel 的近线性退化。
                let mut profile_decode_batch_buckets = vec![[0_u64; 3]; config.decode_batch_limit.saturating_add(1)];
                let mut profile_decode_idle_micros = 0_u64;
                let mut profile_decode_dispatches = 0_u64;
                let mut profile_decode_queue_after = 0_u64;
                let mut profile_decode_empty_after = 0_u64;
                let mut profile_decode_handoff_micros = 0_u64;
                let mut profile_decode_handoff_max_micros = 0_u64;
                let mut profile_decode_dispatch_gap_micros = 0_u64;
                let mut profile_decode_dispatch_gap_max_micros = 0_u64;
                let mut profile_decode_dispatch_gaps = 0_u64;
                let mut profile_decode_input_waits = 0_u64;
                let mut profile_decode_input_wait_micros = 0_u64;
                let mut profile_decode_completion_waits = 0_u64;
                let mut profile_decode_completion_wait_micros = 0_u64;
                let mut profile_last_decode_dispatch = None::<Instant>;
                let mut decode_cursor = 0_usize;
                let mut decode_dispatches_since_prefill = 0_usize;
                // admission window 表示允许在流水线中积累的工作量，burst 表示
                // 单会话可连续送入的 prefill chunk。两者的商给出 decode-first
                // 周期：默认仍饿死 prefill；积压达到一整个公平份额后只放一块。
                let prefill_max_wait = config.pipeline_work_window.div_ceil(config.prefill_admission_burst).max(1);
                let enqueue = |queued_at: Instant,
                               cohort: Option<(u64, usize)>,
                               wave: Option<(u64, usize)>,
                               items: Vec<(usize, usize, T)>,
                               decode_queue: &mut VecDeque<_>,
                               prefill_queue: &mut VecDeque<_>,
                               cohort_sizes: &mut std::collections::HashMap<u64, (usize, usize, usize)>,
                               controls: &mut [StageSessionControl<S>]|
                 -> Result<(), BackendError> {
                    if cohort.is_some() && wave.is_some() {
                        return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} work 同时携带 cohort/wave") });
                    }
                    let class = items.first().map_or(0, |item| batch_class(&item.2));
                    if cohort.is_some() && items.iter().any(|item| batch_class(&item.2) != class) {
                        return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} cohort 内兼容类不一致") });
                    }
                    let group = if let Some((id, original)) = cohort {
                        let size = match cohort_sizes.entry(id) {
                            std::collections::hash_map::Entry::Vacant(entry) => entry.insert((original, original, class)).1,
                            std::collections::hash_map::Entry::Occupied(entry) if entry.get().0 == original && entry.get().2 == class => entry.get().1,
                            std::collections::hash_map::Entry::Occupied(entry) => {
                                return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} cohort={id} 大小/兼容类冲突: {original}/{class} vs {}/{}", entry.get().0, entry.get().2) });
                            }
                        };
                        Some(QueuedGroup { id, size, batch: true })
                    } else if let Some((id, size)) = wave {
                        Some(QueuedGroup { id, size, batch: false })
                    } else {
                        None
                    };
                    for item in items {
                        let (session, position, value) = item;
                        let Some(control) = controls.get_mut(session) else {
                            return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} Work session={session} 越界") });
                        };
                        if control.cancelled {
                            continue;
                        }
                        if control.pending_close.is_some() {
                            return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} session={session} 已请求 Close，拒绝新 Work") });
                        }
                        control.pending_work = control.pending_work.checked_add(1).ok_or_else(|| BackendError::Compute { msg: format!("stage scheduler stage={stage} session={session} pending work 溢出") })?;
                        let kind = work_kind(&value);
                        control.order.push_back(kind);
                        match kind {
                            StageWorkKind::Decode => decode_queue.push_back((group, session, position, value, queued_at)),
                            StageWorkKind::Prefill => prefill_queue.push_back((group, session, position, value, queued_at)),
                        }
                    }
                    Ok(())
                };
                let cancel_session = |session: usize,
                                      states: &[Option<S>],
                                      decode_queue: &mut VecDeque<(Option<QueuedGroup>, usize, usize, T, Instant)>,
                                      prefill_queue: &mut VecDeque<(Option<QueuedGroup>, usize, usize, T, Instant)>,
                                      cohort_sizes: &mut std::collections::HashMap<u64, (usize, usize, usize)>,
                                      controls: &mut [StageSessionControl<S>]|
                 -> Result<(), BackendError> {
                    if session >= config.session_capacity || states[session].is_none() {
                        return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} Cancel session={session} 为空或越界") });
                    }
                    controls[session].cancelled = true;
                    let mut affected = decode_queue.iter().chain(prefill_queue.iter()).filter_map(|item| (item.1 == session).then_some(item.0.filter(|group| group.batch).map(|group| group.id)).flatten()).collect::<Vec<_>>();
                    affected.sort_unstable();
                    affected.dedup();
                    let before = decode_queue.len() + prefill_queue.len();
                    decode_queue.retain(|item| item.1 != session);
                    prefill_queue.retain(|item| item.1 != session);
                    let removed = before - decode_queue.len() - prefill_queue.len();
                    controls[session].pending_work = controls[session].pending_work.checked_sub(removed).ok_or_else(|| BackendError::Compute { msg: format!("stage scheduler stage={stage} Cancel session={session} pending work 下溢") })?;
                    controls[session].order.clear();
                    for cohort in affected {
                        let (_, size, _) = cohort_sizes.get_mut(&cohort).ok_or_else(|| BackendError::Compute { msg: format!("stage scheduler stage={stage} cohort={cohort} 缺少大小") })?;
                        *size = size.checked_sub(1).ok_or_else(|| BackendError::Compute { msg: format!("stage scheduler stage={stage} cohort={cohort} 取消后大小下溢") })?;
                        for item in decode_queue.iter_mut().chain(prefill_queue.iter_mut()).filter(|item| item.0.is_some_and(|candidate| candidate.batch && candidate.id == cohort)) {
                            item.0 = Some(QueuedGroup { id: cohort, size: *size, batch: true });
                        }
                        if *size == 0 {
                            cohort_sizes.remove(&cohort);
                        }
                    }
                    Ok(())
                };
                let finish_close = |session: usize, states: &mut [Option<S>], controls: &mut [StageSessionControl<S>], active_sessions: &mut usize| -> Result<Option<StageSchedulerMessage<T, S>>, BackendError> {
                    if controls[session].pending_work != 0 {
                        return Ok(None);
                    }
                    let Some(mut outgoing) = controls[session].pending_close.take() else {
                        return Ok(None);
                    };
                    backend.finish_stage_session()?;
                    let state = states[session].take().ok_or_else(|| BackendError::Compute { msg: format!("stage scheduler stage={stage} Close session={session} 已为空") })?;
                    *active_sessions = active_sessions.checked_sub(1).ok_or_else(|| BackendError::Compute { msg: format!("stage scheduler stage={stage} active session 下溢") })?;
                    outgoing.push(state);
                    Ok(Some(StageSchedulerMessage::Close { session, states: outgoing }))
                };
                let handle_control = |message: StageSchedulerMessage<T, S>, states: &mut [Option<S>], active_sessions: &mut usize, controls: &mut [StageSessionControl<S>]| match message {
                    StageSchedulerMessage::Open { session, states: mut incoming } => {
                        if session >= config.session_capacity || states[session].is_some() || controls[session].pending_close.is_some() {
                            return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} Open session={session} 非空或越界") });
                        }
                        let Some(state) = incoming.pop_front() else {
                            return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} Open session={session} 缺少 state") });
                        };
                        states[session] = Some(state);
                        controls[session].cancelled = false;
                        *active_sessions = active_sessions.checked_add(1).ok_or_else(|| BackendError::Compute { msg: format!("stage scheduler stage={stage} active session 溢出") })?;
                        Ok(Some(StageSchedulerMessage::Open { session, states: incoming }))
                    }
                    StageSchedulerMessage::Close { session, states: outgoing } => {
                        let Some(slot) = states.get(session) else {
                            return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} Close session={session} 越界") });
                        };
                        if slot.is_none() {
                            return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} Close session={session} 已为空") });
                        }
                        if controls[session].pending_close.is_some() {
                            return Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} Close session={session} 已在等待") });
                        }
                        controls[session].pending_close = Some(outgoing);
                        finish_close(session, states, controls, active_sessions)
                    }
                    StageSchedulerMessage::Work { .. } | StageSchedulerMessage::Cancel { .. } => {
                        unreachable!("控制路径不处理 work")
                    }
                };
                // 阻塞 recv 分支与非阻塞 drain 循环处理同一组输入消息；收口到一处，
                // 避免两份逐字重复的分支各自漂移。返回 Err 表示该 stage 必须终止，
                // 调用方负责把 Err 转发到输出端（通道已关闭时静默失败即可）。
                #[allow(clippy::too_many_arguments)]
                let handle_stage_message = |message: Result<StageSchedulerMessage<T, S>, BackendError>,
                                            states: &mut [Option<S>],
                                            decode_queue: &mut VecDeque<(Option<QueuedGroup>, usize, usize, T, Instant)>,
                                            prefill_queue: &mut VecDeque<(Option<QueuedGroup>, usize, usize, T, Instant)>,
                                            cohort_sizes: &mut std::collections::HashMap<u64, (usize, usize, usize)>,
                                            active_sessions: &mut usize,
                                            controls: &mut [StageSessionControl<S>],
                                            recovered_controls: &mut Vec<(usize, StageSchedulerMessage<T, S>)>| {
                    match message {
                        Ok(StageSchedulerMessage::Work { queued_at, cohort, wave, items }) => enqueue(queued_at, cohort, wave, items, decode_queue, prefill_queue, cohort_sizes, controls),
                        Ok(StageSchedulerMessage::Cancel { session }) => cancel_session(session, states, decode_queue, prefill_queue, cohort_sizes, controls),
                        Ok(control) => match handle_control(control, states, active_sessions, controls)? {
                            Some(control) => match sender.send(Ok(control)) {
                                Ok(()) => Ok(()),
                                Err(std::sync::mpsc::SendError(Ok(control))) => {
                                    recovered_controls.push((stage + 1, control));
                                    Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} 控制输出已关闭") })
                                }
                                Err(std::sync::mpsc::SendError(Err(error))) => Err(error),
                            },
                            None => Ok(()),
                        },
                        Err(error) => Err(error),
                    }
                };

                loop {
                    let mut made_progress = false;
                    if !filling_slots {
                        loop {
                            let concurrent_window = concurrent_stage_window(concurrent_submissions, !prefill_queue.is_empty(), in_flight_by_kind[1]);
                            let ready = match ready_stage_completion(backend, &in_flight, concurrent_window) {
                                Ok(ready) => ready,
                                Err(error) => {
                                    let _ = sender.send(Err(error));
                                    failed = true;
                                    break;
                                }
                            };
                            let Some(ready) = ready else {
                                break;
                            };
                        let StageInFlight { output, sessions: completed_sessions, kind, timing, .. } = in_flight.remove(ready).expect("已检查完成队列索引");
                        if kind == StageWorkKind::Prefill {
                            stage_flow_worker.finish_prefill();
                        }
                        let kind_index = usize::from(kind == StageWorkKind::Prefill);
                        let Some(remaining) = in_flight_by_kind[kind_index].checked_sub(1) else {
                            let _ = sender.send(Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} {kind:?} 在途计数下溢") }));
                            failed = true;
                            break;
                        };
                        in_flight_by_kind[kind_index] = remaining;
                        let completion_prefill_epoch = timing.as_ref().map(|timing| timing.prefill_epoch);
                        if let Some(timing) = timing {
                            let total_micros = timing.started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                            let decode_uncontended = !stage_flow_worker.prefill_active() && timing.prefill_epoch == stage_flow_worker.prefill_epoch();
                            if let (Some(lanes), Some(dispatch_us)) = (timing.trace_lanes.as_deref(), timing.trace_dispatch_us) {
                                record_stage_trace(format!(
                                    "[stage-work-trace] ts_us={dispatch_us} phase=work stage={stage} kind={kind:?} lanes={lanes} complete_us={} queue_us={} submit_us={} run_us={} completion_record_us={} device_us={total_micros} queue_after={} in_flight_after={remaining}",
                                    stage_trace_timestamp_us(),
                                    timing.trace_queue_micros,
                                    timing.submit_micros,
                                    timing.trace_run_micros,
                                    timing.trace_completion_record_micros,
                                    timing.trace_queue_after,
                                ));
                            }
                            if profile_completion && total_micros >= 500_000 {
                                eprintln!(
                                    "[stage-completion-slow] stage={stage} kind={kind:?} sessions={} work_units={} positions={}..={} submit_ms={:.3} total_ms={:.3}",
                                    timing.sessions,
                                    timing.work_units,
                                    timing.position_min,
                                    timing.position_max,
                                    timing.submit_micros as f64 / 1000.0,
                                    total_micros as f64 / 1000.0,
                                );
                            }
                            // GPU 时间汇总全部 stage，token units 只在链尾计一次，
                            // 否则会把一块 prefill 的成本低估为 stage_count 分之一。
                            stage_flow_worker.record_work(kind, if stage + 1 == stage_count { timing.work_units } else { 0 }, total_micros, decode_uncontended);
                            if profile_completion {
                                let profile = kind_index;
                                profile_batches[profile] += 1;
                                profile_sessions[profile] += timing.sessions;
                                profile_submit_micros[profile] = profile_submit_micros[profile].saturating_add(timing.submit_micros);
                                profile_total_micros[profile] = profile_total_micros[profile].saturating_add(total_micros);
                                if kind == StageWorkKind::Decode {
                                    let bucket = timing.sessions.min(profile_decode_batch_buckets.len() - 1);
                                    profile_decode_batch_buckets[bucket][0] += 1;
                                    profile_decode_batch_buckets[bucket][1] = profile_decode_batch_buckets[bucket][1].saturating_add(timing.submit_micros);
                                    profile_decode_batch_buckets[bucket][2] = profile_decode_batch_buckets[bucket][2].saturating_add(total_micros);
                                }
                                if profile_batches[profile] == 32 {
                                    eprintln!(
                                        "[stage-completion-summary] stage={stage} kind={kind:?} batches=32 sessions={} submit_ms={:.3} total_ms={:.3}",
                                        profile_sessions[profile],
                                        profile_submit_micros[profile] as f64 / 1000.0,
                                        profile_total_micros[profile] as f64 / 1000.0,
                                    );
                                    if kind == StageWorkKind::Decode {
                                        eprintln!("[stage-decode-batch-profile] stage={stage} bucket=count/submit_us/total_us {:?}", profile_decode_batch_buckets);
                                        eprintln!(
                                            "[stage-decode-supply] stage={stage} dispatches={} idle_ms={:.3} handoff_avg_us={:.3} handoff_max_us={} dispatch_gap_avg_us={:.3} dispatch_gap_max_us={} input_waits={} input_wait_ms={:.3} completion_waits={} completion_wait_ms={:.3} queue_after_avg={:.3} empty_after={}",
                                            profile_decode_dispatches,
                                            profile_decode_idle_micros as f64 / 1000.0,
                                            profile_decode_handoff_micros as f64 / profile_decode_dispatches.max(1) as f64,
                                            profile_decode_handoff_max_micros,
                                            profile_decode_dispatch_gap_micros as f64 / profile_decode_dispatch_gaps.max(1) as f64,
                                            profile_decode_dispatch_gap_max_micros,
                                            profile_decode_input_waits,
                                            profile_decode_input_wait_micros as f64 / 1000.0,
                                            profile_decode_completion_waits,
                                            profile_decode_completion_wait_micros as f64 / 1000.0,
                                            profile_decode_queue_after as f64 / profile_decode_dispatches.max(1) as f64,
                                            profile_decode_empty_after,
                                        );
                                        profile_decode_batch_buckets.fill([0_u64; 3]);
                                        profile_decode_idle_micros = 0;
                                        profile_decode_dispatches = 0;
                                        profile_decode_queue_after = 0;
                                        profile_decode_empty_after = 0;
                                        profile_decode_handoff_micros = 0;
                                        profile_decode_handoff_max_micros = 0;
                                        profile_decode_dispatch_gap_micros = 0;
                                        profile_decode_dispatch_gap_max_micros = 0;
                                        profile_decode_dispatch_gaps = 0;
                                        profile_decode_input_waits = 0;
                                        profile_decode_input_wait_micros = 0;
                                        profile_decode_completion_waits = 0;
                                        profile_decode_completion_wait_micros = 0;
                                    }
                                    profile_batches[profile] = 0;
                                    profile_sessions[profile] = 0;
                                    profile_submit_micros[profile] = 0;
                                    profile_total_micros[profile] = 0;
                                }
                            }
                        }
                        if in_flight.is_empty() {
                            let current_epoch = stage_flow_worker.prefill_epoch();
                            let contaminated = kind == StageWorkKind::Prefill || stage_flow_worker.prefill_active() || completion_prefill_epoch.is_some_and(|epoch| epoch != current_epoch);
                            idle_started = Some((Instant::now(), contaminated, current_epoch));
                            stage_flow_worker.enter_idle(contaminated);
                        }
                        if let Some((group, output)) = output
                            && {
                                let (cohort, wave) = group.map_or((None, None), QueuedGroup::message_parts);
                                sender.send(Ok(StageSchedulerMessage::Work { queued_at: Instant::now(), cohort, wave, items: output })).is_err()
                            }
                        {
                            failed = true;
                            break;
                        }
                        for session in completed_sessions {
                            let Some(control) = controls.get_mut(session) else {
                                let _ = sender.send(Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} completion session={session} 越界") }));
                                failed = true;
                                break;
                            };
                            if !std::mem::replace(&mut control.in_flight, false) {
                                let _ = sender.send(Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} session={session} completion 缺少在途标记") }));
                                failed = true;
                                break;
                            }
                            let Some(next) = control.pending_work.checked_sub(1) else {
                                let _ = sender.send(Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} session={session} pending work 下溢") }));
                                failed = true;
                                break;
                            };
                            control.pending_work = next;
                            match finish_close(session, &mut states, &mut controls, &mut active_sessions) {
                                Ok(Some(control)) => {
                                    if let Err(std::sync::mpsc::SendError(message)) = sender.send(Ok(control)) {
                                        if let Ok(control) = message {
                                            recovered_controls.push((stage + 1, control));
                                        }
                                        failed = true;
                                        break;
                                    }
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    let _ = sender.send(Err(error));
                                    failed = true;
                                    break;
                                }
                            }
                        }
                        if failed {
                            break;
                        }
                            made_progress = true;
                            backoff.reset();
                        }
                    }
                    if failed {
                        break;
                    }

                    let mut waited = false;
                    if !filling_slots && decode_queue.is_empty() && prefill_queue.is_empty() && in_flight.is_empty() && !input_closed {
                        let message = match stage_receiver.try_recv() {
                            Ok(message) => Some(message),
                            Err(std::sync::mpsc::TryRecvError::Empty) => {
                                waited = true;
                                let started = profile_completion.then(Instant::now);
                                let trace_started = trace_events.then(|| (Instant::now(), stage_trace_timestamp_us()));
                                let message = stage_receiver.recv().ok();
                                if let Some((started, begin_us)) = trace_started {
                                    let duration_us = started.elapsed().as_micros();
                                    if duration_us >= 20_000 {
                                        record_stage_trace(format!(
                                            "[stage-block-trace] ts_us={begin_us} phase=interval stage={stage} reason=input_wait duration_us={duration_us} decode_queue=0 prefill_queue=0 in_flight=0",
                                        ));
                                    }
                                }
                                if let Some(started) = started {
                                    profile_decode_input_waits += 1;
                                    profile_decode_input_wait_micros = profile_decode_input_wait_micros.saturating_add(started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
                                }
                                message
                            }
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
                        };
                        match message {
                            Some(message) => {
                                if let Err(error) = handle_stage_message(message, &mut states, &mut decode_queue, &mut prefill_queue, &mut cohort_sizes, &mut active_sessions, &mut controls, &mut recovered_controls) {
                                    let _ = sender.send(Err(error));
                                    break;
                                }
                            }
                            None => input_closed = true,
                        }
                    }
                    if !filling_slots && !waited && !input_closed {
                        loop {
                            match stage_receiver.try_recv() {
                                Ok(message) => {
                                    if let Err(error) = handle_stage_message(message, &mut states, &mut decode_queue, &mut prefill_queue, &mut cohort_sizes, &mut active_sessions, &mut controls, &mut recovered_controls) {
                                        let _ = sender.send(Err(error));
                                        failed = true;
                                        break;
                                    }
                                }
                                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                    input_closed = true;
                                    break;
                                }
                            }
                        }
                    }
                    if failed {
                        break;
                    }
                    if decode_queue.is_empty() && prefill_queue.is_empty() {
                        if input_closed && in_flight.is_empty() {
                            break;
                        }
                        if !made_progress {
                            let concurrent_window = concurrent_stage_window(concurrent_submissions, false, in_flight_by_kind[1]);
                            if should_wait_for_stage_completion(concurrent_window, in_flight.len(), execution_window(concurrent_window), in_flight_by_kind[0])
                                && let Some(completion) = stage_completion_to_wait(&in_flight, concurrent_window)
                            {
                                let started = profile_completion.then(Instant::now);
                                if let Err(error) = backend.wait_stage_completion(completion) {
                                    let _ = sender.send(Err(error));
                                    break;
                                }
                                if let Some(started) = started {
                                    profile_decode_completion_waits += 1;
                                    profile_decode_completion_wait_micros = profile_decode_completion_wait_micros.saturating_add(started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
                                }
                            } else {
                                backoff.wait();
                            }
                        }
                        continue;
                    }
                    let concurrent_window = concurrent_stage_window(concurrent_submissions, !prefill_queue.is_empty(), in_flight_by_kind[1]);
                    let shared_slot_available = in_flight.len() < execution_window(concurrent_window);
                    let dispatchable = |controls: &[StageSessionControl<S>], session: usize, kind: StageWorkKind| session < config.session_capacity && !controls[session].in_flight && controls[session].order.front() == Some(&kind);
                    let decode_first = if decode_queue.front().is_some_and(|item| item.0.is_none()) {
                        // 设备忙时同一 session 可能先积累出多行。completion 后按
                        // session 轮转选择首行，禁止 K 行 verify 长串独占本 stage；
                        // 显式 cohort/wave 仍保持调用方给出的边界与顺序。
                        decode_queue
                            .iter()
                            .enumerate()
                            .take_while(|(_, item)| item.0.is_none())
                            .filter(|(_, item)| dispatchable(&controls, item.1, StageWorkKind::Decode))
                            .min_by_key(|(_, item)| (item.1 + config.session_capacity - decode_cursor) % config.session_capacity)
                            .map(|(index, _)| index)
                    } else {
                        decode_queue.iter().position(|item| dispatchable(&controls, item.1, StageWorkKind::Decode))
                    };
                    let prefill_first = prefill_queue.iter().position(|item| dispatchable(&controls, item.1, StageWorkKind::Prefill));
                    let decode_available = shared_slot_available && decode_first.is_some() && in_flight_by_kind[0] < submission_slots(StageWorkKind::Decode);
                    let prefill_available = shared_slot_available && prefill_first.is_some() && in_flight_by_kind[1] < submission_slots(StageWorkKind::Prefill);
                    let Some(kind) = choose_stage_work_kind(decode_available, prefill_available, decode_dispatches_since_prefill, prefill_max_wait) else {
                        filling_slots = false;
                        // 独立后台队列只允许纯 prefill 留在后台等待；只要 latency
                        // submission 在途就直接等它的 event。B 侧常按 just-in-time
                        // 供给、队列暂时为空，若只在有 backlog 时等待，仍会在每个
                        // stage 退化成 50-200us 轮询空泡。
                        if should_wait_for_stage_completion(concurrent_window, in_flight.len(), execution_window(concurrent_window), in_flight_by_kind[0])
                            && let Some(completion) = stage_completion_to_wait(&in_flight, concurrent_window)
                        {
                            let started = profile_completion.then(Instant::now);
                            if let Err(error) = backend.wait_stage_completion(completion) {
                                let _ = sender.send(Err(error));
                                break;
                            }
                            if let Some(started) = started {
                                profile_decode_completion_waits += 1;
                                profile_decode_completion_wait_micros = profile_decode_completion_wait_micros.saturating_add(started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
                            }
                        } else {
                            backoff.wait();
                        }
                        continue;
                    };
                    let queue_index = usize::from(kind == StageWorkKind::Prefill);
                    let execution_slots = submission_slots(kind);
                    if kind == StageWorkKind::Prefill && active_sessions <= execution_slots {
                        direct_submissions[queue_index] = 0;
                    }
                    let queue = match kind {
                        StageWorkKind::Decode => &mut decode_queue,
                        StageWorkKind::Prefill => &mut prefill_queue,
                    };
                    let first_index = match kind {
                        StageWorkKind::Decode => decode_first,
                        StageWorkKind::Prefill => prefill_first,
                    }
                    .expect("work kind 只从可提交队列选择");
                    let first_class = batch_class(&queue[first_index].3);
                    let mut decode_batch_size = 1_usize;
                    if kind == StageWorkKind::Decode {
                        let first_group = queue.get(first_index).map(|item| item.0).expect("decode work 已定位");
                        let required = first_group.map_or(config.batch_limit(kind), |group| if group.batch { group.size } else { 1 });
                        controls.iter_mut().for_each(|control| control.occupied = false);
                        let ready_sessions = queue
                            .iter()
                            .filter(|item| item.0 == first_group)
                            .filter(|item| batch_class(&item.3) == first_class)
                            .filter(|item| {
                                let fresh = dispatchable(&controls, item.1, kind) && !controls[item.1].occupied;
                                controls[item.1].occupied = true;
                                fresh
                            })
                            .take(config.batch_limit(kind))
                            .count();
                        if required > 1 && ready_sessions < required {
                            if first_group.is_some_and(|group| group.batch) {
                                // 显式 cohort 由上游保证完整发送；禁止超时后拆批。
                                filling_slots = false;
                                backoff.wait();
                                continue;
                            } else if first_group.is_none() {
                                decode_batch_size = ready_sessions.max(1);
                            }
                        } else if first_group.is_none() {
                            // 不等待凑满上限：CU 忙期间自然积累了几路，就在本次
                            // completion 后批几路；只有真实 singleton 才继续单行。
                            decode_batch_size = ready_sessions.max(1);
                        }
                    }
                    let (group, first_session, first_position, first_value, queued_at) = queue.remove(first_index).expect("work 已定位");
                    if first_session >= config.session_capacity || states[first_session].is_none() {
                        let _ = sender.send(Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} 收到未打开 session={first_session}") }));
                        break;
                    }
                    let mut batch_work = work_size(&first_value).max(1);
                    controls.iter_mut().for_each(|control| {
                        control.occupied = false;
                        control.deferred = false;
                    });
                    controls[first_session].occupied = true;
                    let mut batch = vec![(first_session, first_position, first_value)];
                    let batching = match kind {
                        StageWorkKind::Decode => group.is_some_and(|group| group.batch) || (group.is_none() && decode_batch_size > 1),
                        StageWorkKind::Prefill => active_sessions > execution_slots && direct_submissions[queue_index] >= execution_slots,
                    };
                    let batch_limit = if group.is_none() && kind == StageWorkKind::Decode {
                        decode_batch_size
                    } else if batching {
                        config.batch_limit(kind)
                    } else {
                        1
                    };
                    let mut candidate = 0;
                    while batch.len() < batch_limit && candidate < queue.len() {
                        let (next_cohort, next_session, next_work, next_class) = {
                            let next = &queue[candidate];
                            (next.0, next.1, work_size(&next.3).max(1), batch_class(&next.3))
                        };
                        if next_session >= config.session_capacity || states[next_session].is_none() {
                            let _ = sender.send(Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} 收到未打开 session={next_session}") }));
                            failed = true;
                            break;
                        }
                        if next_cohort != group
                            || next_class != first_class
                            || controls[next_session].occupied
                            || controls[next_session].deferred
                            || !dispatchable(&controls, next_session, kind)
                            || batch_work.saturating_add(next_work) > config.batch_work_limit
                        {
                            // 稳定扫描队列，不移动被跳过的 work；这样同一 session
                            // 即使跨 kind，也不会因凑批旋转到后来 work 之后。
                            controls[next_session].deferred = true;
                            candidate += 1;
                            continue;
                        }
                        let (_, next_session, next_position, next_value, _) = queue.remove(candidate).expect("candidate 已定位");
                        controls[next_session].occupied = true;
                        batch_work = batch_work.saturating_add(next_work);
                        batch.push((next_session, next_position, next_value));
                    }
                    if failed {
                        break;
                    }
                    if let Some(group) = group.filter(|group| group.batch) {
                        if batch.len() != group.size {
                            let _ = sender.send(Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} cohort={} 被拆批: batch={} expected={}", group.id, batch.len(), group.size) }));
                            break;
                        }
                        cohort_sizes.remove(&group.id);
                    }
                    if !batching {
                        direct_submissions[queue_index] = direct_submissions[queue_index].saturating_add(1).min(execution_slots);
                    }
                    let session_ids = batch.iter().map(|item| item.0).collect::<Vec<_>>();
                    if kind == StageWorkKind::Decode {
                        decode_cursor = (session_ids.last().copied().unwrap_or(first_session) + 1) % config.session_capacity;
                    }
                    for &session in &session_ids {
                        let Some(queued) = controls[session].order.pop_front() else {
                            let _ = sender.send(Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} session={session} 缺少入队顺序") }));
                            failed = true;
                            break;
                        };
                        if queued != kind || std::mem::replace(&mut controls[session].in_flight, true) {
                            let _ = sender.send(Err(BackendError::Compute { msg: format!("stage scheduler stage={stage} session={session} 提交顺序冲突: queued={queued:?} dispatch={kind:?}") }));
                            failed = true;
                            break;
                        }
                    }
                    if failed {
                        break;
                    }
                    let sessions = session_ids.len();
                    let position_min = batch.iter().map(|(_, position, _)| *position).min().unwrap_or(first_position);
                    let position_max = batch.iter().map(|(_, position, _)| *position).max().unwrap_or(first_position);
                    let trace_lanes = trace_events.then(|| batch.iter().map(|(session, position, _)| format!("{session}@{position}")).collect::<Vec<_>>().join(","));
                    let trace_dispatch_us = trace_events.then(stage_trace_timestamp_us);
                    let trace_queue_micros = queued_at.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                    let trace_queue_after = queue.len();
                    if profile_completion && kind == StageWorkKind::Decode {
                        let now = Instant::now();
                        let handoff_micros = queued_at.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                        profile_decode_dispatches += 1;
                        profile_decode_queue_after = profile_decode_queue_after.saturating_add(u64::try_from(queue.len()).unwrap_or(u64::MAX));
                        profile_decode_empty_after += u64::from(queue.is_empty());
                        profile_decode_handoff_micros = profile_decode_handoff_micros.saturating_add(handoff_micros);
                        profile_decode_handoff_max_micros = profile_decode_handoff_max_micros.max(handoff_micros);
                        if let Some(previous) = profile_last_decode_dispatch.replace(now) {
                            let gap_micros = now.duration_since(previous).as_micros().min(u128::from(u64::MAX)) as u64;
                            profile_decode_dispatch_gap_micros = profile_decode_dispatch_gap_micros.saturating_add(gap_micros);
                            profile_decode_dispatch_gap_max_micros = profile_decode_dispatch_gap_max_micros.max(gap_micros);
                            profile_decode_dispatch_gaps += 1;
                        }
                    }
                    if in_flight.is_empty()
                        && let Some((started, contaminated, epoch)) = idle_started.take()
                    {
                        stage_flow_worker.leave_idle(contaminated);
                        let micros = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                        if contaminated || stage_flow_worker.prefill_active() || epoch != stage_flow_worker.prefill_epoch() {
                            stage_flow_worker.record_prefill_idle(micros);
                        } else {
                            // decode/prefill 都按 stage 累计，空泡也使用同一量纲；
                            // 只统计全 stage 同时空闲会漏掉流水线中的错峰空泡。
                            stage_flow_worker.record_idle(micros);
                            if profile_completion {
                                profile_decode_idle_micros = profile_decode_idle_micros.saturating_add(micros);
                            }
                        }
                    }
                    if profile_completion && kind == StageWorkKind::Decode {
                        let queued_micros = queued_at.elapsed().as_micros();
                        if queued_micros >= 500_000 {
                            eprintln!("[stage-queue-slow] stage={stage} sessions={sessions} position={position_min}..={position_max} queued_ms={:.3}", queued_micros as f64 / 1000.0,);
                        }
                    }
                    let started = Instant::now();
                    let prefill_epoch = stage_flow_worker.prefill_epoch();
                    let submission_kind = match kind {
                        StageWorkKind::Decode => StageSubmissionKind::Latency,
                        StageWorkKind::Prefill => StageSubmissionKind::Background,
                    };
                    if let Err(error) = backend.activate_stage_submission(submission_kind) {
                        let _ = sender.send(Err(error));
                        break;
                    }
                    if let Err(error) = backend.begin_stage_submission() {
                        let _ = sender.send(Err(error));
                        break;
                    }
                    if let Some(lanes) = trace_lanes.as_deref()
                        && let Err(error) = backend.trace_stage_work_begin(&format!("stage={stage} kind={kind:?} lanes={lanes}"))
                    {
                        let _ = backend.abort_stage_submission();
                        let _ = sender.send(Err(error));
                        break;
                    }
                    let begin_micros = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                    let run_started = Instant::now();
                    let output = match run_batch(backend, &mut states, stage, batch) {
                        Ok(output) => output,
                        Err(error) => {
                            if trace_lanes.is_some() {
                                let _ = backend.trace_stage_work_end();
                            }
                            let _ = backend.abort_stage_submission();
                            let _ = sender.send(Err(error));
                            break;
                        }
                    };
                    let run_micros = run_started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                    let submit_micros = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                    let completion_started = Instant::now();
                    let completion = match backend.record_stage_completion() {
                        Ok(completion) => completion,
                        Err(error) => {
                            if trace_lanes.is_some() {
                                let _ = backend.trace_stage_work_end();
                            }
                            let _ = backend.abort_stage_submission();
                            let _ = sender.send(Err(error));
                            break;
                        }
                    };
                    let completion_micros = completion_started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                    if trace_lanes.is_some()
                        && let Err(error) = backend.trace_stage_work_end()
                    {
                        let _ = sender.send(Err(error));
                        break;
                    }
                    if profile_completion && kind == StageWorkKind::Decode && (submit_micros >= 100_000 || completion_micros >= 100_000) {
                        eprintln!(
                            "[stage-submit-slow] stage={stage} sessions={sessions} position={position_min}..={position_max} begin_ms={:.3} run_ms={:.3} completion_record_ms={:.3}",
                            begin_micros as f64 / 1000.0,
                            run_micros as f64 / 1000.0,
                            completion_micros as f64 / 1000.0,
                        );
                    }
                    // 中间 stage 立即把有序交接工作送到下一 stage，由 backend
                    // completion capability 串起源计算、搬运和目标计算；最终 stage
                    // 必须等完成后再把 tensor 暴露给 host/采样回调。
                    let output = if stage + 1 < stage_count {
                        let (cohort, wave) = group.map_or((None, None), QueuedGroup::message_parts);
                        if sender.send(Ok(StageSchedulerMessage::Work { queued_at: Instant::now(), cohort, wave, items: output })).is_err() {
                            break;
                        }
                        None
                    } else {
                        Some((group, output))
                    };
                    let timing = Some(StageSubmissionTiming {
                        sessions,
                        work_units: batch_work,
                        position_min,
                        position_max,
                        started,
                        submit_micros,
                        prefill_epoch,
                        trace_lanes,
                        trace_dispatch_us,
                        trace_queue_micros,
                        trace_run_micros: run_micros,
                        trace_completion_record_micros: completion_micros,
                        trace_queue_after,
                    });
                    if kind == StageWorkKind::Prefill {
                        stage_flow_worker.begin_prefill();
                        decode_dispatches_since_prefill = 0;
                    } else {
                        decode_dispatches_since_prefill = decode_dispatches_since_prefill.saturating_add(1);
                    }
                    in_flight_by_kind[queue_index] += 1;
                    in_flight.push_back(StageInFlight { completion, output, sessions: session_ids, kind, timing });
                    backoff.reset();
                    // admission 只限制可入队的 prefill 总量；stage 上有 decode
                    // backlog 时不消耗其 CU，直到本 stage 没有可提交 decode。
                    let concurrent_window = concurrent_stage_window(concurrent_submissions, !prefill_queue.is_empty(), in_flight_by_kind[1]);
                    let shared_slot_available = in_flight.len() < execution_window(concurrent_window);
                    filling_slots = shared_slot_available
                        && ((!decode_queue.is_empty() && in_flight_by_kind[0] < submission_slots(StageWorkKind::Decode)) || (!prefill_queue.is_empty() && in_flight_by_kind[1] < submission_slots(StageWorkKind::Prefill)));
                }
                if profile_completion {
                    for (profile, kind) in [StageWorkKind::Decode, StageWorkKind::Prefill].into_iter().enumerate() {
                        if profile_batches[profile] != 0 {
                            eprintln!(
                                "[stage-completion-summary] stage={stage} kind={kind:?} batches={} sessions={} submit_ms={:.3} total_ms={:.3}",
                                profile_batches[profile],
                                profile_sessions[profile],
                                profile_submit_micros[profile] as f64 / 1000.0,
                                profile_total_micros[profile] as f64 / 1000.0,
                            );
                            if kind == StageWorkKind::Decode {
                                eprintln!("[stage-decode-batch-profile] stage={stage} bucket=count/submit_us/total_us {:?}", profile_decode_batch_buckets);
                                eprintln!(
                                    "[stage-decode-supply] stage={stage} dispatches={} idle_ms={:.3} handoff_avg_us={:.3} handoff_max_us={} dispatch_gap_avg_us={:.3} dispatch_gap_max_us={} input_waits={} input_wait_ms={:.3} completion_waits={} completion_wait_ms={:.3} queue_after_avg={:.3} empty_after={}",
                                    profile_decode_dispatches,
                                    profile_decode_idle_micros as f64 / 1000.0,
                                    profile_decode_handoff_micros as f64 / profile_decode_dispatches.max(1) as f64,
                                    profile_decode_handoff_max_micros,
                                    profile_decode_dispatch_gap_micros as f64 / profile_decode_dispatch_gaps.max(1) as f64,
                                    profile_decode_dispatch_gap_max_micros,
                                    profile_decode_input_waits,
                                    profile_decode_input_wait_micros as f64 / 1000.0,
                                    profile_decode_completion_waits,
                                    profile_decode_completion_wait_micros as f64 / 1000.0,
                                    profile_decode_queue_after as f64 / profile_decode_dispatches.max(1) as f64,
                                    profile_decode_empty_after,
                                );
                            }
                        }
                    }
                }
                recovered_controls.extend(stage_receiver.try_iter().filter_map(Result::ok).map(|control| (stage, control)));
                (stage, states, controls, recovered_controls)
            }));
            receiver = next_receiver;
        }

        let scheduler = StageSchedulerHandle {
            input,
            output: receiver,
            pending: std::sync::Mutex::new(VecDeque::new()),
            next_ready_cohort: std::sync::atomic::AtomicU64::new(1_u64 << 63),
            session_capacity: config.session_capacity,
            stage_count,
            pipeline_work_window: config.pipeline_work_window,
            prefill_admission_burst: config.prefill_admission_burst,
            decode_batch_limit: config.decode_batch_limit,
            stage_flow,
        };
        let mut drive_result = drive(&scheduler);
        // 先关闭输入、保留最终输出 receiver，再等待全部 worker 退出。这样已经发送
        // 成功的 Closed 不会在 drop handle 时丢失；中间通道发送失败的控制消息由
        // 对应 worker 带回。
        let StageSchedulerHandle { input, output, pending, .. } = scheduler;
        drop(input);
        let pending = pending.into_inner().map_err(|_| BackendError::Compute { msg: "stage scheduler pending 锁中毒".to_owned() })?;
        let worker_states = workers.into_iter().map(|worker| worker.join().map_err(|_| BackendError::Compute { msg: "stage scheduler worker panic".to_owned() })).collect::<Result<Vec<_>, _>>()?;
        let mut recovered = (0..config.session_capacity).map(|_| (0..stage_count).map(|_| None).collect::<Vec<Option<S>>>()).collect::<Vec<_>>();
        let mut recovery_error = None;
        for (stage, states, session_controls, controls) in worker_states {
            for (session, state) in states.into_iter().enumerate() {
                if let Some(state) = state {
                    recover_scheduler_state(&mut recovered, &mut recovery_error, session, stage, state);
                }
            }
            for (session, control) in session_controls.into_iter().enumerate() {
                if let Some(states) = control.pending_close {
                    for (stage, state) in states.into_iter().enumerate() {
                        recover_scheduler_state(&mut recovered, &mut recovery_error, session, stage, state);
                    }
                }
            }
            for (next_stage, control) in controls {
                recover_scheduler_control(&mut recovered, &mut recovery_error, next_stage, control);
            }
        }
        for ready in pending {
            if let StageSchedulerOutput::Closed { session, states } = ready {
                for (stage, state) in states.into_iter().enumerate() {
                    recover_scheduler_state(&mut recovered, &mut recovery_error, session, stage, state);
                }
            }
        }
        for message in output {
            match message {
                Ok(control) => recover_scheduler_control(&mut recovered, &mut recovery_error, stage_count, control),
                Err(error) => {
                    // backend 首错沿 stage 链传播可能慢于输入 receiver 退出。驱动端
                    // 此时只能观察到传输层错误；回收时必须用真实首错替换它。
                    let replace = match &drive_result {
                        Ok(_) => true,
                        Err(error) => is_scheduler_transport_exit(error),
                    };
                    if replace {
                        drive_result = Err(error);
                    }
                }
            }
        }
        let sessions = recovered
            .into_iter()
            .enumerate()
            .map(|(session, states)| {
                let count = states.iter().filter(|state| state.is_some()).count();
                if count == 0 {
                    None
                } else if count == stage_count {
                    Some(states.into_iter().map(|state| state.expect("已确认完整")).collect())
                } else {
                    recovery_error.get_or_insert_with(|| BackendError::Compute { msg: format!("stage scheduler session={session} 仅回收 {count}/{stage_count} 个 state") });
                    None
                }
            })
            .collect();
        if let Some(error) = recovery_error {
            return Err(error);
        }
        Ok((drive_result, sessions))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };

    use super::*;

    #[test]
    fn 独立后台窗口不把decode完成降级为轮询() {
        assert!(!concurrent_stage_window(true, false, 0), "纯 decode 必须保持单队列快路径");
        assert!(concurrent_stage_window(true, true, 0), "prefill 入队后才打开独立窗口");
        assert!(concurrent_stage_window(true, false, 1), "prefill 在途时必须允许后来 decode 插队");
        assert!(!should_wait_for_stage_completion(true, 1, 2, 0), "只有 background prefill 在途时必须继续接收新 decode");
        assert!(should_wait_for_stage_completion(true, 1, 2, 1), "即使下游供给暂时为空也必须等待 latency completion");
        assert!(should_wait_for_stage_completion(true, 2, 2, 0), "共享窗口占满时必须等待任一优先 completion");
        assert!(should_wait_for_stage_completion(false, 1, 1, 0), "单队列 backend 保持原有阻塞等待语义");
    }

    #[test]
    fn opportunistic_prefill只消费空泡与decode公平预算() {
        let mut admission = OpportunisticPrefillAdmission::default();
        assert_eq!(admission.limits(StageFlowSnapshot::default(), 8, 8, true, 32, 256), (1, 1, Some(32)));
        admission.commit_work(32, 1);
        assert_eq!(admission.limits(StageFlowSnapshot::default(), 8, 8, true, 32, 256), (0, 1, Some(32)));

        let first = StageFlowSnapshot { decode_micros: 1_600, prefill_micros: 3_200, prefill_work_units: 32, prefill_batches: 1, idle_micros: 25_600, prefill_idle_micros: 0, prefill_active: 0, prefill_idle_pending: 0 };
        assert_eq!(admission.limits(first, 8, 8, true, 32, 256), (1, 1, Some(32)));
        admission.commit_work(32, 1);
        let bubble = StageFlowSnapshot { prefill_micros: 6_400, prefill_work_units: 64, prefill_batches: 2, idle_micros: 128_000, ..first };
        assert_eq!(admission.limits(bubble, 8, 8, true, 32, 256), (1, 1, Some(128)));
        assert_eq!(admission.limits(bubble, 8, 8, false, 32, 256), (8, 8, None));
    }

    #[test]
    fn opportunistic_prefill短请求按quantum串行续发() {
        let mut admission = OpportunisticPrefillAdmission::default();
        assert_eq!(admission.limits(StageFlowSnapshot::default(), 8, 8, true, 32, 256), (1, 1, Some(32)));
        assert_eq!(admission.chunk_size(4096, 32, 179), 32);
        assert_eq!(admission.chunk_size(4096, 32, 257), 32);
        assert_eq!(admission.chunk_size(16, 32, 179), 16);
    }

    #[test]
    fn opportunistic_prefill短请求必须等上一块全stage退休() {
        let mut admission = OpportunisticPrefillAdmission::default();
        assert_eq!(admission.limits(StageFlowSnapshot::default(), 8, 8, true, 32, 256), (1, 1, Some(32)));
        assert!(admission.short_request_ready());
        admission.commit_work(128, 1);
        assert!(!admission.short_request_ready());
        let completed = StageFlowSnapshot { prefill_micros: 8_000, prefill_work_units: 128, prefill_batches: 1, prefill_active: 0, prefill_idle_pending: 0, ..StageFlowSnapshot::default() };
        assert_eq!(admission.limits(completed, 8, 8, true, 32, 256), (0, 1, Some(32)));
        assert!(admission.short_request_ready());
    }

    #[test]
    fn adaptive_chunk按请求类型与当前位置调整() {
        let policy = AdaptiveChunkPolicy { initial_chunk_size: 4096, append_chunk_size: 2048, long_context_threshold_tokens: 128 * 1024, long_context_chunk_size: 2048 };
        assert_eq!(policy.chunk_size(0, 0), 4096);
        assert_eq!(policy.chunk_size(0, 4096), 4096);
        assert_eq!(policy.chunk_size(4096, 4096), 2048);
        assert_eq!(policy.chunk_size(0, 128 * 1024), 2048);
    }

    #[test]
    fn adaptive_chunk把零值规范为最小工作量() {
        let policy = AdaptiveChunkPolicy { initial_chunk_size: 0, append_chunk_size: 4096, long_context_threshold_tokens: 128 * 1024, long_context_chunk_size: 0 };
        assert_eq!(policy.chunk_size(0, 0), 1);
        assert_eq!(policy.chunk_size(0, 128 * 1024), 1);
        assert_eq!(policy.chunk_size(4096, 128 * 1024), 1);
    }

    #[test]
    fn chunked_prefill按块覆盖区间并携带绝对位置() {
        let mut visited = Vec::new();
        run_chunked_prefill(8, 10, 4, |range, position| {
            assert_eq!(position, range.start + 8);
            visited.push((range.start, range.end));
            Ok::<_, ()>(())
        })
        .unwrap();
        // offset=8 len=10 chunk=4: [8,12) [12,16) [16,18)，相对区间 [0,4) [4,8) [8,10)
        assert_eq!(visited, vec![(0, 4), (4, 8), (8, 10)]);
    }

    #[test]
    fn chunked_prefill空区间与错误短路() {
        let mut visited = 0;
        run_chunked_prefill(0, 0, 4, |_, _| {
            visited += 1;
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(visited, 0);
        let error = run_chunked_prefill(0, 10, 4, |range, _| if range.start >= 4 { Err("stop") } else { Ok(()) });
        assert!(error.is_err());
    }

    #[derive(Clone)]
    struct TestBackend {
        ready: Arc<AtomicBool>,
        poll_allowed: bool,
    }

    struct TestCompletion {
        ready: Arc<AtomicBool>,
        poll_allowed: bool,
    }

    impl StageExecutionBackend for TestBackend {
        type Completion = TestCompletion;

        fn stage_available_bytes(&self) -> Result<usize, BackendError> {
            Ok(usize::MAX)
        }

        fn record_stage_completion(&self) -> Result<Self::Completion, BackendError> {
            Ok(TestCompletion { ready: self.ready.clone(), poll_allowed: self.poll_allowed })
        }

        fn stage_completion_ready(&self, completion: &Self::Completion) -> Result<bool, BackendError> {
            if !completion.poll_allowed {
                return Err(BackendError::Compute { msg: "不应轮询有序链的中间 completion".to_owned() });
            }
            Ok(completion.ready.load(Ordering::Acquire))
        }
    }

    #[derive(Clone)]
    struct OrderedLatencyTestBackend(TestBackend);

    impl StageExecutionBackend for OrderedLatencyTestBackend {
        type Completion = TestCompletion;

        fn stage_available_bytes(&self) -> Result<usize, BackendError> {
            self.0.stage_available_bytes()
        }

        fn max_queued_latency_submissions(&self) -> usize {
            1
        }

        fn record_stage_completion(&self) -> Result<Self::Completion, BackendError> {
            self.0.record_stage_completion()
        }

        fn stage_completion_ready(&self, completion: &Self::Completion) -> Result<bool, BackendError> {
            self.0.stage_completion_ready(completion)
        }
    }

    #[derive(Clone)]
    struct ConcurrentTestBackend {
        ready: Arc<Mutex<Vec<bool>>>,
        activations: Arc<Mutex<Vec<StageSubmissionKind>>>,
    }

    impl ConcurrentTestBackend {
        fn new() -> Self {
            Self { ready: Arc::new(Mutex::new(Vec::new())), activations: Arc::new(Mutex::new(Vec::new())) }
        }

        fn recorded(&self) -> usize {
            self.ready.lock().unwrap().len()
        }

        fn complete(&self, submission: usize) {
            self.ready.lock().unwrap()[submission] = true;
        }
    }

    impl StageExecutionBackend for ConcurrentTestBackend {
        type Completion = usize;

        fn stage_available_bytes(&self) -> Result<usize, BackendError> {
            Ok(usize::MAX)
        }

        fn activate_stage_submission(&self, kind: StageSubmissionKind) -> Result<(), BackendError> {
            self.activations.lock().unwrap().push(kind);
            Ok(())
        }

        fn supports_concurrent_stage_submissions(&self) -> bool {
            true
        }

        fn record_stage_completion(&self) -> Result<Self::Completion, BackendError> {
            let mut ready = self.ready.lock().unwrap();
            let submission = ready.len();
            ready.push(false);
            Ok(submission)
        }

        fn stage_completion_ready(&self, completion: &Self::Completion) -> Result<bool, BackendError> {
            Ok(self.ready.lock().unwrap()[*completion])
        }
    }

    #[derive(Clone)]
    struct ClosingBackend(Arc<Mutex<usize>>);

    impl StageExecutionBackend for ClosingBackend {
        type Completion = ();

        fn stage_available_bytes(&self) -> Result<usize, BackendError> {
            Ok(usize::MAX)
        }

        fn record_stage_completion(&self) -> Result<Self::Completion, BackendError> {
            Ok(())
        }

        fn stage_completion_ready(&self, _completion: &Self::Completion) -> Result<bool, BackendError> {
            Ok(true)
        }

        fn finish_stage_session(&self) -> Result<(), BackendError> {
            *self.0.lock().unwrap() += 1;
            Ok(())
        }
    }

    #[test]
    fn close在每个stage归还状态前释放临时资源() {
        let releases = Arc::new(Mutex::new(0));
        let config = StageSchedulerConfig {
            session_capacity: 1,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 1,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![ClosingBackend(releases.clone()), ClosingBackend(releases.clone())],
            vec![vec![0_i32, 1_i32]],
            config,
            |_| 1,
            |_: &()| StageWorkKind::Decode,
            |_, _, _, batch| Ok(batch),
            |scheduler| {
                scheduler.close(0)?;
                loop {
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Closed { states, .. }) => {
                            assert_eq!(states, vec![0, 1]);
                            break Ok(());
                        }
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Work { .. }) | None => std::thread::yield_now(),
                    }
                }
            },
        )
        .unwrap();
        assert_eq!(*releases.lock().unwrap(), 2);
        assert_eq!(remaining, vec![None]);
    }

    #[test]
    fn recoverable_scheduler保留首个stage错误() {
        let ready = Arc::new(AtomicBool::new(true));
        let config = StageSchedulerConfig {
            session_capacity: 1,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 1,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let (result, remaining) = drive_stage_scheduler_recoverable(
            vec![TestBackend { ready, poll_allowed: true }],
            vec![vec![0_i32]],
            config,
            |_| 1,
            |_: &()| StageWorkKind::Decode,
            |_, _, stage, _| Err(BackendError::Compute { msg: format!("stage={stage} 首个错误") }),
            |scheduler| -> Result<(), BackendError> {
                scheduler.submit(0, 0, ())?;
                loop {
                    let _ = scheduler.try_recv()?;
                    std::thread::yield_now();
                }
            },
        )
        .unwrap();
        assert!(matches!(result, Err(BackendError::Compute { msg }) if msg == "stage=0 首个错误"));
        assert_eq!(remaining, vec![Some(vec![0])]);
    }

    #[test]
    fn 输入通道退出不遮蔽首个stage错误() {
        let ready = Arc::new(AtomicBool::new(true));
        let failed = Arc::new(AtomicBool::new(false));
        let config = StageSchedulerConfig {
            session_capacity: 1,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 1,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let (result, remaining) = drive_stage_scheduler_recoverable(
            vec![TestBackend { ready, poll_allowed: true }],
            vec![vec![0_i32]],
            config,
            |_| 1,
            |_: &()| StageWorkKind::Decode,
            |_, _, stage, _| {
                failed.store(true, Ordering::Release);
                Err(BackendError::Compute { msg: format!("stage={stage} 真实错误") })
            },
            |scheduler| -> Result<(), BackendError> {
                scheduler.submit(0, 0, ())?;
                while !failed.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                loop {
                    match scheduler.submit(0, 0, ()) {
                        Ok(()) => std::thread::yield_now(),
                        Err(error) => return Err(error),
                    }
                }
            },
        )
        .unwrap();
        assert!(matches!(result, Err(BackendError::Compute { msg }) if msg == "stage=0 真实错误"));
        assert_eq!(remaining, vec![Some(vec![0])]);
    }

    #[test]
    fn recoverable_scheduler错误时不丢已完成close的state() {
        let ready = Arc::new(AtomicBool::new(true));
        let config = StageSchedulerConfig {
            session_capacity: 2,
            batch_work_limit: 2,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 1,
            prefill_admission_burst: 1,
            decode_batch_limit: 2,
            prefill_batch_limit: 2,
            profile_completion: false,
        };
        let stages = 8;
        let sessions = vec![(0..stages).collect::<Vec<_>>(), (100..100 + stages).collect::<Vec<_>>()];
        let (result, remaining) = drive_stage_scheduler_recoverable(
            (0..stages).map(|_| TestBackend { ready: ready.clone(), poll_allowed: true }).collect(),
            sessions.clone(),
            config,
            |_| 1,
            |_: &()| StageWorkKind::Decode,
            |_, _, stage, batch| {
                if stage == 0 && batch.iter().any(|(session, _, _)| *session == 1) { Err(BackendError::Compute { msg: "close 后的 stage 错误".to_owned() }) } else { Ok(batch) }
            },
            |scheduler| -> Result<(), BackendError> {
                scheduler.close(0)?;
                scheduler.submit(1, 0, ())?;
                Err(scheduler.recv_until_error())
            },
        )
        .unwrap();
        assert!(matches!(result, Err(BackendError::Compute { msg }) if msg == "close 后的 stage 错误"));
        assert_eq!(remaining, sessions.into_iter().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn single_stage_chain_submits_without_worker_queues() {
        let ready = Arc::new(AtomicBool::new(true));
        let dispatches = Mutex::new(Vec::new());
        let (position, value, states) = run_single_stage_chain(&[TestBackend { ready: ready.clone(), poll_allowed: false }, TestBackend { ready, poll_allowed: true }], vec![0_i32, 100_i32], 7, 9_i32, |_, states, stage, batch| {
            dispatches.lock().unwrap().push(stage);
            *states[0].as_mut().unwrap() += 1;
            Ok(batch)
        })
        .unwrap();
        assert_eq!((position, value), (7, 9));
        assert_eq!(states, vec![1, 101]);
        assert_eq!(*dispatches.lock().unwrap(), vec![0, 1]);
    }

    #[test]
    fn intermediate_stage_handoff_precedes_host_completion_poll() {
        let first_ready = Arc::new(AtomicBool::new(false));
        let second_ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<usize>::new());
        let config = StageSchedulerConfig {
            session_capacity: 1,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 2,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: first_ready.clone(), poll_allowed: true }, TestBackend { ready: second_ready.clone(), poll_allowed: true }],
            vec![vec![0_i32, 100_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, stage, batch| {
                dispatches.lock().unwrap().push(stage);
                *states[0].as_mut().unwrap() += 1;
                Ok(batch)
            },
            |scheduler| {
                assert_eq!(scheduler.pipeline_work_window(), 2);
                assert_eq!(scheduler.prefill_admission_burst(), 1);
                scheduler.submit(0, 0, (StageWorkKind::Decode, 7))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().len() < 2 {
                    assert!(std::time::Instant::now() < deadline, "中间 stage 等待了 host completion 轮询");
                    std::thread::yield_now();
                }
                assert!(scheduler.try_recv()?.is_none(), "最终 stage completion 前不得暴露输出");
                first_ready.store(true, Ordering::Release);
                second_ready.store(true, Ordering::Release);
                loop {
                    assert!(std::time::Instant::now() < deadline, "等待最终 stage completion 超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { session, position, value, .. }) => {
                            assert_eq!((session, position, value.1), (0, 0, 7));
                            break;
                        }
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![Some(vec![1, 101])]);
        assert_eq!(*dispatches.lock().unwrap(), vec![0, 1]);
    }

    #[test]
    fn 独立后台队列允许跨会话decode越过prefill完成() {
        let backend = ConcurrentTestBackend::new();
        let dispatches = Mutex::new(Vec::<(StageWorkKind, usize)>::new());
        let config = StageSchedulerConfig {
            session_capacity: 2,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 2,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![backend.clone()],
            vec![vec![0_i32], vec![100_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, usize)| value.0,
            |_, states, _, batch| {
                for (session, _, value) in &batch {
                    dispatches.lock().unwrap().push((value.0, *session));
                    *states[*session].as_mut().unwrap() += 1;
                }
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit(0, 0, (StageWorkKind::Prefill, 7))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while backend.recorded() < 1 {
                    assert!(std::time::Instant::now() < deadline, "prefill 未提交到后台队列");
                    std::thread::yield_now();
                }
                scheduler.submit(1, 0, (StageWorkKind::Decode, 9))?;
                while backend.recorded() < 2 {
                    assert!(std::time::Instant::now() < deadline, "未完成 prefill 阻塞了新到达 decode");
                    std::thread::yield_now();
                }
                assert_eq!(*backend.activations.lock().unwrap(), [StageSubmissionKind::Background, StageSubmissionKind::Latency]);
                assert_eq!(*dispatches.lock().unwrap(), [(StageWorkKind::Prefill, 0), (StageWorkKind::Decode, 1)]);

                backend.complete(1);
                loop {
                    assert!(std::time::Instant::now() < deadline, "等待 decode 跨序退休超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { session, value, .. }) => {
                            assert_eq!((session, value), (1, (StageWorkKind::Decode, 9)));
                            break;
                        }
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                assert!(scheduler.try_recv()?.is_none(), "后台 prefill 未完成前不得暴露输出");

                backend.complete(0);
                loop {
                    assert!(std::time::Instant::now() < deadline, "等待 prefill 退休超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { session, value, .. }) => {
                            assert_eq!((session, value), (0, (StageWorkKind::Prefill, 7)));
                            break;
                        }
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![Some(vec![1]), Some(vec![101])]);
    }

    #[test]
    fn 后台prefill通过device_event直接跨stage交接() {
        let first = ConcurrentTestBackend::new();
        let second = ConcurrentTestBackend::new();
        let dispatches = Mutex::new(Vec::<usize>::new());
        let config = StageSchedulerConfig {
            session_capacity: 1,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 2,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![first.clone(), second.clone()],
            vec![vec![0_i32, 100_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, usize)| value.0,
            |_, states, stage, batch| {
                dispatches.lock().unwrap().push(stage);
                *states[0].as_mut().unwrap() += 1;
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit(0, 0, (StageWorkKind::Prefill, 7))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while first.recorded() < 1 {
                    assert!(std::time::Instant::now() < deadline, "首 stage prefill 未提交");
                    std::thread::yield_now();
                }
                while dispatches.lock().unwrap().len() < 2 {
                    assert!(std::time::Instant::now() < deadline, "后台 prefill 等待了源 completion，未直接交接下一 stage");
                    std::thread::yield_now();
                }
                assert_eq!(*dispatches.lock().unwrap(), [0, 1]);
                first.complete(0);
                while second.recorded() < 1 {
                    assert!(std::time::Instant::now() < deadline, "下一 stage 未记录 completion");
                    std::thread::yield_now();
                }
                second.complete(0);
                loop {
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { session, value, .. }) => {
                            assert_eq!((session, value), (0, (StageWorkKind::Prefill, 7)));
                            break;
                        }
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => {
                            assert!(std::time::Instant::now() < deadline, "等待最终 stage 输出超时");
                            std::thread::yield_now();
                        }
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![Some(vec![1, 101])]);
        assert_eq!(*dispatches.lock().unwrap(), [0, 1]);
    }

    #[test]
    fn 同会话多份decode跨stage流水但每个stage等待前项completion() {
        let first_ready = Arc::new(AtomicBool::new(false));
        let second_ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<(usize, usize)>::new());
        let config = StageSchedulerConfig {
            session_capacity: 1,
            batch_work_limit: 1,
            execution_slots: 3,
            decode_execution_slots: 3,
            pipeline_work_window: 3,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: first_ready.clone(), poll_allowed: true }, TestBackend { ready: second_ready.clone(), poll_allowed: true }],
            vec![vec![0_i32, 100_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, usize)| value.0,
            |_, states, stage, batch| {
                for (_, position, _) in &batch {
                    dispatches.lock().unwrap().push((stage, *position));
                }
                *states[0].as_mut().unwrap() += batch.len() as i32;
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit_many((0..3).map(|position| (0, position, (StageWorkKind::Decode, position))))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().len() < 2 {
                    assert!(std::time::Instant::now() < deadline, "首份同会话 decode 未穿过后续 stage");
                    std::thread::yield_now();
                }
                assert_eq!(*dispatches.lock().unwrap(), vec![(0, 0), (1, 0)], "同一 stage 不得在前项 completion 前修改同一 session 状态");
                assert!(scheduler.try_recv()?.is_none(), "最终 stage completion 前不得暴露输出");
                first_ready.store(true, Ordering::Release);
                second_ready.store(true, Ordering::Release);
                let mut positions = Vec::new();
                while positions.len() < 3 {
                    assert!(std::time::Instant::now() < deadline, "等待同会话 decode 输出超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { position, .. }) => positions.push(position),
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                assert_eq!(positions, [0, 1, 2]);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![Some(vec![3, 103])]);
        let dispatches = dispatches.lock().unwrap();
        assert_eq!(dispatches.iter().filter_map(|(stage, position)| (*stage == 0).then_some(*position)).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(dispatches.iter().filter_map(|(stage, position)| (*stage == 1).then_some(*position)).collect::<Vec<_>>(), [0, 1, 2]);
    }

    #[test]
    fn 同会话decode不能越过较早prefill() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<(usize, StageWorkKind, usize)>::new());
        let config = StageSchedulerConfig {
            session_capacity: 2,
            batch_work_limit: 2,
            execution_slots: 2,
            decode_execution_slots: 2,
            pipeline_work_window: 2,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            vec![vec![0_i32], vec![0_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, usize)| value.0,
            |_, _, _, batch| {
                dispatches.lock().unwrap().extend(batch.iter().map(|(session, position, value)| (*session, value.0, *position)));
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit_many([(0, 0, (StageWorkKind::Prefill, 0)), (0, 1, (StageWorkKind::Decode, 1)), (1, 0, (StageWorkKind::Decode, 0))])?;
                let deadline = Instant::now() + Duration::from_secs(1);
                while dispatches.lock().unwrap().len() < 2 {
                    assert!(Instant::now() < deadline, "等待独立 session decode 与较早 prefill 提交超时");
                    std::thread::yield_now();
                }
                let before_completion = dispatches.lock().unwrap().clone();
                assert!(before_completion.contains(&(1, StageWorkKind::Decode, 0)));
                assert!(before_completion.contains(&(0, StageWorkKind::Prefill, 0)));
                assert!(!before_completion.contains(&(0, StageWorkKind::Decode, 1)));
                ready.store(true, Ordering::Release);
                let mut work = 0;
                while work < 3 {
                    assert!(Instant::now() < deadline, "等待跨 kind 有序工作完成超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        let dispatches = dispatches.lock().unwrap();
        let prefill = dispatches.iter().position(|item| *item == (0, StageWorkKind::Prefill, 0)).unwrap();
        let decode = dispatches.iter().position(|item| *item == (0, StageWorkKind::Decode, 1)).unwrap();
        assert!(prefill < decode, "同一 session 的跨 kind 顺序必须按入队顺序退休: {dispatches:?}");
    }

    #[test]
    fn decode优先且只在公平份额耗尽后放行prefill() {
        assert!(matches!(choose_stage_work_kind(true, true, 0, 15), Some(StageWorkKind::Decode)));
        assert!(matches!(choose_stage_work_kind(true, true, 14, 15), Some(StageWorkKind::Decode)));
        assert!(matches!(choose_stage_work_kind(true, true, 15, 15), Some(StageWorkKind::Prefill)));
        assert!(matches!(choose_stage_work_kind(true, false, 15, 15), Some(StageWorkKind::Decode)));
        assert!(matches!(choose_stage_work_kind(false, true, 0, 15), Some(StageWorkKind::Prefill)));
        assert!(choose_stage_work_kind(false, false, 0, 15).is_none());
    }

    #[test]
    fn prefill_admission_burst_finishes_current_session_before_rotation() {
        let handle = StageSchedulerHandle::<(), ()> {
            input: std::sync::mpsc::channel().0,
            output: std::sync::mpsc::channel().1,
            pending: std::sync::Mutex::new(VecDeque::new()),
            next_ready_cohort: std::sync::atomic::AtomicU64::new(1_u64 << 63),
            session_capacity: 3,
            stage_count: 1,
            pipeline_work_window: 8,
            prefill_admission_burst: 3,
            decode_batch_limit: 3,
            stage_flow: Arc::new(StageFlowMetrics::new(1)),
        };
        let mut cursor = PrefillAdmissionCursor::default();
        let mut remaining = [3_usize, 2, 1];
        let mut order = Vec::new();
        while remaining.iter().any(|&count| count != 0) {
            let session = handle.next_prefill_session(&mut cursor, remaining.len(), |session| remaining[session] != 0).unwrap();
            remaining[session] -= 1;
            order.push(session);
            handle.commit_prefill_submission(&mut cursor, session, remaining[session] == 0);
        }
        assert_eq!(order, vec![0, 0, 0, 1, 1, 2]);
    }

    #[test]
    fn busy_stage_prioritizes_decode_before_prefill() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<Vec<i32>>::new());
        let config = StageSchedulerConfig {
            session_capacity: 8,
            batch_work_limit: 3,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 3,
            prefill_admission_burst: 1,
            decode_batch_limit: 2,
            prefill_batch_limit: 3,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            vec![vec![0_i32], vec![10_i32], vec![20_i32], vec![30_i32], vec![40_i32], vec![50_i32], vec![60_i32], vec![70_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, _, batch| {
                dispatches.lock().unwrap().push(batch.iter().map(|(_, _, value)| value.1).collect());
                let mut output = Vec::with_capacity(batch.len());
                for (session, position, value) in batch {
                    *states[session].as_mut().unwrap() += value.1 + 1;
                    output.push((session, position, value));
                }
                Ok(output)
            },
            |scheduler| {
                scheduler.submit(0, 0, (StageWorkKind::Prefill, 0))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().is_empty() {
                    assert!(std::time::Instant::now() < deadline, "首份工作没有立即执行");
                    std::thread::yield_now();
                }
                for session in 1..7 {
                    scheduler.submit(session, 0, (StageWorkKind::Prefill, session as i32))?;
                }
                scheduler.submit(7, 0, (StageWorkKind::Decode, 7))?;
                ready.store(true, Ordering::Release);

                let mut work = 0;
                let mut closed = Vec::new();
                while work < 8 || closed.len() < 8 {
                    assert!(std::time::Instant::now() < deadline, "等待 stage scheduler 输出超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => {
                            work += 1;
                            if work == 8 {
                                for session in 0..8 {
                                    scheduler.close(session)?;
                                }
                            }
                        }
                        Some(StageSchedulerOutput::Closed { session, states }) => closed.push((session, states[0])),
                        Some(StageSchedulerOutput::Opened { .. }) | None => std::thread::yield_now(),
                    }
                }
                closed.sort_by_key(|(session, _)| *session);
                assert_eq!(closed, vec![(0, 1), (1, 12), (2, 23), (3, 34), (4, 45), (5, 56), (6, 67), (7, 78),]);
                Ok(())
            },
        )
        .unwrap();
        assert!(remaining.into_iter().all(|states| states.is_none()));
        assert_eq!(*dispatches.lock().unwrap(), vec![vec![0], vec![7], vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn close等待同一会话的排队和执行工作完成() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<i32>::new());
        let config = StageSchedulerConfig {
            session_capacity: 1,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 1,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            vec![vec![0_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, _, batch| {
                let value = batch[0].2.1;
                dispatches.lock().unwrap().push(value);
                *states[0].as_mut().unwrap() += value;
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit(0, 0, (StageWorkKind::Decode, 1))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().is_empty() {
                    assert!(std::time::Instant::now() < deadline, "首份工作没有提交");
                    std::thread::yield_now();
                }
                scheduler.submit(0, 1, (StageWorkKind::Decode, 2))?;
                scheduler.close(0)?;
                for _ in 0..10_000 {
                    assert!(scheduler.try_recv()?.is_none(), "设备 completion 前不得返回 Work 或 Closed");
                    std::thread::yield_now();
                }

                ready.store(true, Ordering::Release);
                let mut work = 0;
                let mut closed = None;
                while work < 2 || closed.is_none() {
                    assert!(std::time::Instant::now() < deadline, "等待排队工作和 Close 完成超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Closed { session, states }) => {
                            assert_eq!(session, 0);
                            closed = Some(states[0]);
                        }
                        Some(StageSchedulerOutput::Opened { .. }) | None => std::thread::yield_now(),
                    }
                }
                assert_eq!(closed, Some(3));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![None]);
        assert_eq!(*dispatches.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn cancel丢弃未发射工作并等待在途completion后关闭() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<i32>::new());
        let config = StageSchedulerConfig {
            session_capacity: 1,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 3,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            vec![vec![0_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, _, batch| {
                let value = batch[0].2.1;
                dispatches.lock().unwrap().push(value);
                *states[0].as_mut().unwrap() += value;
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit(0, 0, (StageWorkKind::Prefill, 1))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().is_empty() {
                    assert!(std::time::Instant::now() < deadline, "首份工作没有提交");
                    std::thread::yield_now();
                }
                scheduler.submit(0, 1, (StageWorkKind::Prefill, 2))?;
                scheduler.submit(0, 2, (StageWorkKind::Prefill, 4))?;
                scheduler.cancel(0)?;
                scheduler.close(0)?;
                ready.store(true, Ordering::Release);

                let mut completed = 0;
                loop {
                    assert!(std::time::Instant::now() < deadline, "等待取消后的 Close 超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Closed { session, states }) => {
                            assert_eq!(session, 0);
                            assert_eq!(states, vec![1]);
                            assert_eq!(completed, 1, "已发射工作必须完成，才能形成一致 checkpoint");
                            break;
                        }
                        Some(StageSchedulerOutput::Work { .. }) => completed += 1,
                        Some(StageSchedulerOutput::Opened { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![None]);
        assert_eq!(*dispatches.lock().unwrap(), vec![1]);
    }

    #[test]
    fn cancel让已发射工作走完整条stage链() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<(usize, i32)>::new());
        let config = StageSchedulerConfig {
            session_capacity: 1,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 3,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }, TestBackend { ready: ready.clone(), poll_allowed: true }],
            vec![vec![0_i32, 100_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, stage, batch| {
                let value = batch[0].2.1;
                dispatches.lock().unwrap().push((stage, value));
                *states[0].as_mut().unwrap() += value;
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit(0, 0, (StageWorkKind::Prefill, 1))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().len() < 2 {
                    assert!(std::time::Instant::now() < deadline, "首份工作没有进入完整 stage 链");
                    std::thread::yield_now();
                }
                scheduler.submit(0, 1, (StageWorkKind::Prefill, 2))?;
                scheduler.submit(0, 2, (StageWorkKind::Prefill, 4))?;
                scheduler.cancel(0)?;
                scheduler.close(0)?;
                ready.store(true, Ordering::Release);

                let mut completed = 0;
                loop {
                    assert!(std::time::Instant::now() < deadline, "等待跨 stage 取消关闭超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => completed += 1,
                        Some(StageSchedulerOutput::Closed { session, states }) => {
                            assert_eq!(session, 0);
                            assert_eq!(states, vec![1, 101]);
                            assert_eq!(completed, 1);
                            break;
                        }
                        Some(StageSchedulerOutput::Opened { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![None]);
        assert_eq!(*dispatches.lock().unwrap(), vec![(0, 1), (1, 1)]);
    }

    #[test]
    fn idle_slots_dispatch_multiple_prefill_batches() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<Vec<i32>>::new());
        let config = StageSchedulerConfig {
            session_capacity: 4,
            batch_work_limit: 1,
            execution_slots: 4,
            decode_execution_slots: 4,
            pipeline_work_window: 4,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            vec![vec![0_i32], vec![10_i32], vec![20_i32], vec![30_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, _, batch| {
                dispatches.lock().unwrap().push(batch.iter().map(|(_, _, value)| value.1).collect());
                for (session, _, value) in &batch {
                    *states[*session].as_mut().unwrap() += value.1 + 1;
                }
                Ok(batch)
            },
            |scheduler| {
                for session in 0..4 {
                    scheduler.submit(session, 0, (StageWorkKind::Prefill, session as i32))?;
                }
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().len() < 4 {
                    assert!(std::time::Instant::now() < deadline, "空闲 execution slot 没有继续提交 prefill");
                    std::thread::yield_now();
                }
                ready.store(true, Ordering::Release);
                let mut work = 0;
                while work < 4 {
                    assert!(std::time::Instant::now() < deadline, "等待 prefill completion 超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                for session in 0..4 {
                    scheduler.close(session)?;
                }
                let mut closed = 0;
                while closed < 4 {
                    assert!(std::time::Instant::now() < deadline, "等待关闭 session 超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Closed { .. }) => closed += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Work { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert!(remaining.into_iter().all(|states| states.is_none()));
        assert_eq!(*dispatches.lock().unwrap(), vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn decode_slots_can_exceed_prefill_slots() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<i32>::new());
        let config = StageSchedulerConfig {
            session_capacity: 4,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 4,
            pipeline_work_window: 4,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            vec![vec![0_i32], vec![10_i32], vec![20_i32], vec![30_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, _, batch| {
                let (session, _, value) = batch[0];
                dispatches.lock().unwrap().push(value.1);
                *states[session].as_mut().unwrap() += value.1;
                Ok(batch)
            },
            |scheduler| {
                for session in 0..4 {
                    scheduler.submit(session, 0, (StageWorkKind::Decode, session as i32 + 1))?;
                }
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().len() < 4 {
                    assert!(std::time::Instant::now() < deadline, "decode completion 未就绪时没有继续提交轻量工作");
                    std::thread::yield_now();
                }
                ready.store(true, Ordering::Release);
                let mut work = 0;
                while work < 4 {
                    assert!(std::time::Instant::now() < deadline, "等待 decode completion 超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![Some(vec![1]), Some(vec![12]), Some(vec![23]), Some(vec![34])]);
        assert_eq!(*dispatches.lock().unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn backend_latency上限把设备在途保持为一份() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<i32>::new());
        let config = StageSchedulerConfig {
            session_capacity: 2,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 4,
            pipeline_work_window: 2,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![OrderedLatencyTestBackend(TestBackend { ready: ready.clone(), poll_allowed: true })],
            vec![vec![0_i32], vec![10_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, _, batch| {
                let (session, _, value) = batch[0];
                dispatches.lock().unwrap().push(value.1);
                *states[session].as_mut().unwrap() += value.1;
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit(0, 0, (StageWorkKind::Decode, 1))?;
                scheduler.submit(1, 0, (StageWorkKind::Decode, 2))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().is_empty() {
                    assert!(std::time::Instant::now() < deadline, "首份 decode 没有提交");
                    std::thread::yield_now();
                }
                for _ in 0..10_000 {
                    assert_eq!(dispatches.lock().unwrap().len(), 1, "前一 completion 未完成时不应继续排入 latency stream");
                    std::thread::yield_now();
                }
                ready.store(true, Ordering::Release);
                let mut work = 0;
                while work < 2 {
                    assert!(std::time::Instant::now() < deadline, "等待 decode completion 超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![Some(vec![1]), Some(vec![12])]);
        assert_eq!(*dispatches.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn decode占满共享窗口时prefill等待completion() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<(StageWorkKind, i32)>::new());
        let config = StageSchedulerConfig {
            session_capacity: 3,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 2,
            pipeline_work_window: 3,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            vec![vec![0_i32], vec![10_i32], vec![20_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, _, batch| {
                let (session, _, value) = batch[0];
                dispatches.lock().unwrap().push(value);
                *states[session].as_mut().unwrap() += value.1;
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit(0, 0, (StageWorkKind::Decode, 1))?;
                scheduler.submit(1, 0, (StageWorkKind::Decode, 2))?;
                scheduler.submit(2, 0, (StageWorkKind::Prefill, 3))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().len() < 2 {
                    assert!(std::time::Instant::now() < deadline, "decode 没有填满共享执行窗口");
                    std::thread::yield_now();
                }
                for _ in 0..10_000 {
                    assert_eq!(dispatches.lock().unwrap().len(), 2, "decode 未完成时不应把 prefill 塞入共享执行窗口");
                    std::thread::yield_now();
                }
                ready.store(true, Ordering::Release);
                let mut work = 0;
                while work < 3 {
                    assert!(std::time::Instant::now() < deadline, "等待 decode/prefill completion 超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(remaining, vec![Some(vec![1]), Some(vec![12]), Some(vec![23])]);
        assert_eq!(*dispatches.lock().unwrap(), vec![(StageWorkKind::Decode, 1), (StageWorkKind::Decode, 2), (StageWorkKind::Prefill, 3)]);
    }

    #[test]
    fn completion即时就绪时按公平窗口放行一块prefill() {
        let ready = Arc::new(AtomicBool::new(true));
        let release_first = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<(StageWorkKind, i32)>::new());
        let config = StageSchedulerConfig {
            session_capacity: 6,
            batch_work_limit: 1,
            execution_slots: 1,
            decode_execution_slots: 2,
            pipeline_work_window: 3,
            prefill_admission_burst: 1,
            decode_batch_limit: 1,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready, poll_allowed: true }],
            (0..6).map(|_| vec![0_i32]).collect(),
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, _, _, batch| {
                let value = batch[0].2;
                let first = {
                    let mut dispatches = dispatches.lock().unwrap();
                    dispatches.push(value);
                    dispatches.len() == 1
                };
                while first && !release_first.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                Ok(batch)
            },
            |scheduler| {
                for session in 0..5 {
                    scheduler.submit(session, 0, (StageWorkKind::Decode, session as i32 + 1))?;
                }
                scheduler.submit(5, 0, (StageWorkKind::Prefill, 9))?;
                release_first.store(true, Ordering::Release);
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                let mut work = 0;
                while work < 6 {
                    assert!(std::time::Instant::now() < deadline, "等待即时 completion 工作完成超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert!(remaining.into_iter().all(|states| states.is_some()));
        let dispatches = dispatches.lock().unwrap();
        let prefill_position = dispatches.iter().position(|value| *value == (StageWorkKind::Prefill, 9)).unwrap();
        assert_eq!(prefill_position, 3, "三个 decode batch 后应只放行一块 prefill: {dispatches:?}");
        assert_eq!(dispatches.iter().filter_map(|(kind, value)| (*kind == StageWorkKind::Decode).then_some(*value)).collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn 同会话队头不阻挡其他会话成批() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<Vec<(usize, i32)>>::new());
        let config = StageSchedulerConfig {
            session_capacity: 4,
            batch_work_limit: 3,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 4,
            prefill_admission_burst: 1,
            decode_batch_limit: 3,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            (0..4).map(|_| vec![0_i32]).collect(),
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, states, _, batch| {
                dispatches.lock().unwrap().push(batch.iter().map(|(session, _, value)| (*session, value.1)).collect());
                for (session, _, value) in &batch {
                    let state = states[*session].as_mut().unwrap();
                    *state = *state * 10 + value.1;
                }
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit(3, 0, (StageWorkKind::Decode, 9))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().is_empty() {
                    assert!(std::time::Instant::now() < deadline, "等待首份直接提交超时");
                    std::thread::yield_now();
                }
                scheduler.submit_many([(0, 1, (StageWorkKind::Decode, 1)), (0, 2, (StageWorkKind::Decode, 2)), (1, 1, (StageWorkKind::Decode, 3)), (2, 1, (StageWorkKind::Decode, 4))])?;
                ready.store(true, Ordering::Release);
                let mut work = 0;
                while work < 5 {
                    assert!(std::time::Instant::now() < deadline, "等待跨 session backlog 完成超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*dispatches.lock().unwrap(), vec![vec![(3, 9)], vec![(0, 1), (1, 3), (2, 4)], vec![(0, 2)]]);
        assert_eq!(remaining, vec![Some(vec![12]), Some(vec![3]), Some(vec![4]), Some(vec![9])]);
    }

    #[test]
    fn 首项立即发射且只在cu_completion后loop() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<Vec<i32>>::new());
        let config = StageSchedulerConfig {
            session_capacity: 4,
            batch_work_limit: 4,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 4,
            prefill_admission_burst: 1,
            decode_batch_limit: 4,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            (0..4).map(|_| vec![0_i32]).collect(),
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, _, _, batch| {
                dispatches.lock().unwrap().push(batch.iter().map(|(_, _, value)| value.1).collect());
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit(0, 0, (StageWorkKind::Decode, 0))?;
                let deadline = Instant::now() + Duration::from_secs(1);
                while dispatches.lock().unwrap().is_empty() {
                    assert!(Instant::now() < deadline, "空闲 CU 没有立即发射首项");
                    std::thread::yield_now();
                }
                scheduler.submit_many((1..4).map(|session| (session, 0, (StageWorkKind::Decode, session as i32))))?;
                for _ in 0..100 {
                    std::thread::yield_now();
                }
                assert_eq!(*dispatches.lock().unwrap(), vec![vec![0]], "CU 忙时只能积累队列，不得预提交下一轮");
                ready.store(true, Ordering::Release);
                let mut work = 0;
                while work < 4 {
                    assert!(Instant::now() < deadline, "completion 后 loop 没有立即消费 backlog");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*dispatches.lock().unwrap(), vec![vec![0], vec![1, 2, 3]]);
    }

    #[test]
    fn stage0对设备忙时自然积累的decode立即部分合批() {
        let ready = Arc::new(AtomicBool::new(false));
        let dispatches = Mutex::new(Vec::<Vec<i32>>::new());
        let config = StageSchedulerConfig {
            session_capacity: 5,
            batch_work_limit: 5,
            execution_slots: 1,
            decode_execution_slots: 2,
            pipeline_work_window: 5,
            prefill_admission_burst: 1,
            decode_batch_limit: 3,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }],
            vec![vec![0_i32], vec![10_i32], vec![20_i32], vec![30_i32], vec![40_i32]],
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, _, _, batch| {
                dispatches.lock().unwrap().push(batch.iter().map(|(_, _, value)| value.1).collect());
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit_many((0..5).map(|session| (session, 0, (StageWorkKind::Decode, session as i32))))?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                while dispatches.lock().unwrap().len() < 2 {
                    assert!(std::time::Instant::now() < deadline, "decode cohort 没有按上限提交");
                    std::thread::yield_now();
                }
                assert_eq!(*dispatches.lock().unwrap(), vec![vec![0, 1, 2], vec![3, 4]]);
                ready.store(true, Ordering::Release);
                let mut work = 0;
                while work < 5 {
                    assert!(std::time::Instant::now() < deadline, "等待 backlog batch 超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert!(remaining.into_iter().all(|states| states.is_some()));
        assert_eq!(*dispatches.lock().unwrap(), vec![vec![0, 1, 2], vec![3, 4]]);
    }

    #[test]
    fn stage只合并相同兼容类的work() {
        let ready = Arc::new(AtomicBool::new(true));
        let dispatches = Mutex::new(Vec::<Vec<(usize, i32)>>::new());
        let config = StageSchedulerConfig {
            session_capacity: 4,
            batch_work_limit: 4,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 4,
            prefill_admission_burst: 1,
            decode_batch_limit: 4,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        drive_stage_scheduler_with_batch_class(
            vec![TestBackend { ready, poll_allowed: true }],
            (0..4).map(|_| vec![0_i32]).collect(),
            config,
            |_| 1,
            |_: &(StageWorkKind, usize, i32)| StageWorkKind::Decode,
            |value| value.1,
            |_, _, _, batch| {
                let class = batch[0].2.1;
                assert!(batch.iter().all(|item| item.2.1 == class));
                let mut current = batch.iter().map(|(session, _, value)| (*session, value.2)).collect::<Vec<_>>();
                current.sort_unstable();
                dispatches.lock().unwrap().push(current);
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit_many([(0, 0, (StageWorkKind::Decode, 0, 0)), (1, 0, (StageWorkKind::Decode, 1, 1)), (2, 0, (StageWorkKind::Decode, 0, 2)), (3, 0, (StageWorkKind::Decode, 1, 3))])?;
                let deadline = Instant::now() + Duration::from_secs(1);
                let mut work = 0;
                while work < 4 {
                    assert!(Instant::now() < deadline, "等待兼容类 batch 完成超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*dispatches.lock().unwrap(), vec![vec![(0, 0), (2, 2)], vec![(1, 1), (3, 3)]]);
    }

    #[test]
    fn stage0低压decode不等待并消费已到达的小批() {
        let ready = Arc::new(AtomicBool::new(true));
        let dispatches = Mutex::new(Vec::<Vec<i32>>::new());
        let config = StageSchedulerConfig {
            session_capacity: 4,
            batch_work_limit: 4,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 3,
            prefill_admission_burst: 1,
            decode_batch_limit: 4,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        drive_stage_scheduler(
            vec![TestBackend { ready, poll_allowed: true }],
            (0..3).map(|_| vec![0_i32]).collect(),
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, _, _, batch| {
                dispatches.lock().unwrap().push(batch.iter().map(|(_, _, value)| value.1).collect());
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit_many((0..3).map(|session| (session, 0, (StageWorkKind::Decode, session as i32))))?;
                let deadline = Instant::now() + Duration::from_secs(1);
                let mut work = 0;
                while work < 3 {
                    assert!(Instant::now() < deadline, "低压 decode 没有立即完成");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { .. }) => work += 1,
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*dispatches.lock().unwrap(), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn 显式cohort穿过全部stage不拆批() {
        let ready = Arc::new(AtomicBool::new(true));
        let dispatches = Mutex::new(Vec::<(usize, Vec<i32>)>::new());
        let config = StageSchedulerConfig {
            session_capacity: 4,
            batch_work_limit: 16,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 4,
            prefill_admission_burst: 1,
            decode_batch_limit: 4,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }, TestBackend { ready, poll_allowed: true }],
            (0..4).map(|_| vec![0_i32, 0_i32]).collect(),
            config,
            |value: &(StageWorkKind, i32, usize)| value.2,
            |value| value.0,
            |_, _, stage, batch| {
                dispatches.lock().unwrap().push((stage, batch.iter().map(|(_, _, value)| value.1).collect()));
                Ok(batch)
            },
            |scheduler| {
                for session in 0..4 {
                    scheduler.submit_cohort(7, 4, [(session, 0, (StageWorkKind::Decode, session as i32, 4))])?;
                }
                let deadline = Instant::now() + Duration::from_secs(1);
                let mut output = Vec::new();
                while output.len() < 4 {
                    assert!(Instant::now() < deadline, "等待显式 cohort 完成超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { cohort, session, .. }) => {
                            assert_eq!(cohort, Some((7, 4)));
                            output.push(session);
                        }
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                output.sort_unstable();
                assert_eq!(output, vec![0, 1, 2, 3]);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*dispatches.lock().unwrap(), vec![(0, vec![0, 1, 2, 3]), (1, vec![0, 1, 2, 3])]);
        assert!(remaining.into_iter().all(|states| states.is_some()));
    }

    #[test]
    fn 有序wave逐行穿过全部stage并保留尾部聚合标识() {
        let ready = Arc::new(AtomicBool::new(true));
        let dispatches = Mutex::new(Vec::<(usize, Vec<i32>)>::new());
        let config = StageSchedulerConfig {
            session_capacity: 4,
            batch_work_limit: 16,
            execution_slots: 1,
            decode_execution_slots: 4,
            pipeline_work_window: 8,
            prefill_admission_burst: 1,
            decode_batch_limit: 4,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        let ((), remaining) = drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }, TestBackend { ready, poll_allowed: true }],
            (0..4).map(|_| vec![0_i32, 0_i32]).collect(),
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, _, stage, batch| {
                dispatches.lock().unwrap().push((stage, batch.iter().map(|(_, _, value)| value.1).collect()));
                Ok(batch)
            },
            |scheduler| {
                let wave = scheduler.submit_ready_wave((0..4).map(|session| (session, 0, (StageWorkKind::Decode, session as i32))))?;
                let deadline = Instant::now() + Duration::from_secs(1);
                let mut output = Vec::new();
                while output.len() < 4 {
                    assert!(Instant::now() < deadline, "等待有序 wave 完成超时");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { cohort, session, .. }) => {
                            assert_eq!(cohort, Some(wave));
                            output.push(session);
                        }
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                assert_eq!(output, vec![0, 1, 2, 3]);
                Ok(())
            },
        )
        .unwrap();
        let dispatches = dispatches.lock().unwrap();
        for stage in 0..2 {
            let rows = dispatches.iter().filter(|(candidate, _)| *candidate == stage).map(|(_, batch)| batch.clone()).collect::<Vec<_>>();
            assert_eq!(rows, vec![vec![0], vec![1], vec![2], vec![3]], "stage={stage} 应保持单行有序执行");
        }
        assert!(remaining.into_iter().all(|states| states.is_some()));
    }

    #[test]
    fn 取消排队会话后cohort缩小但不拆批() {
        let ready = Arc::new(AtomicBool::new(true));
        let dispatches = Mutex::new(Vec::<(usize, Vec<usize>)>::new());
        let config = StageSchedulerConfig {
            session_capacity: 4,
            batch_work_limit: 4,
            execution_slots: 1,
            decode_execution_slots: 1,
            pipeline_work_window: 4,
            prefill_admission_burst: 1,
            decode_batch_limit: 4,
            prefill_batch_limit: 1,
            profile_completion: false,
        };
        drive_stage_scheduler(
            vec![TestBackend { ready: ready.clone(), poll_allowed: true }, TestBackend { ready, poll_allowed: true }],
            (0..4).map(|_| vec![0_i32, 0_i32]).collect(),
            config,
            |_| 1,
            |value: &(StageWorkKind, i32)| value.0,
            |_, _, stage, batch| {
                dispatches.lock().unwrap().push((stage, batch.iter().map(|(session, _, _)| *session).collect()));
                Ok(batch)
            },
            |scheduler| {
                scheduler.submit_cohort(9, 4, [(0, 0, (StageWorkKind::Decode, 0))])?;
                scheduler.submit_cohort(9, 4, [(1, 0, (StageWorkKind::Decode, 1))])?;
                scheduler.cancel(0)?;
                scheduler.submit_cohort(9, 4, [(2, 0, (StageWorkKind::Decode, 2))])?;
                scheduler.submit_cohort(9, 4, [(3, 0, (StageWorkKind::Decode, 3))])?;
                let deadline = Instant::now() + Duration::from_secs(1);
                let mut outputs = Vec::new();
                while outputs.len() < 3 {
                    assert!(Instant::now() < deadline, "取消后 cohort 没有继续完成");
                    match scheduler.try_recv()? {
                        Some(StageSchedulerOutput::Work { cohort, session, .. }) => {
                            assert_eq!(cohort, Some((9, 3)));
                            outputs.push(session);
                        }
                        Some(StageSchedulerOutput::Opened { .. } | StageSchedulerOutput::Closed { .. }) | None => std::thread::yield_now(),
                    }
                }
                outputs.sort_unstable();
                assert_eq!(outputs, vec![1, 2, 3]);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*dispatches.lock().unwrap(), vec![(0, vec![1, 2, 3]), (1, vec![1, 2, 3])]);
    }
}
