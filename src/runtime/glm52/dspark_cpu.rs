//! GLM-5.2 DSpark 的 CPU 张量搬运组合。

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};

use super::dspark::{Glm52DsparkBackend, Glm52DsparkDraftBatch, Glm52DsparkRuntime};
use crate::{
    backend::{BackendError, cpu::CpuContext},
    kernel::cpu::CpuTensor,
    runtime::dspark::DsparkTargetCache,
};

pub type CpuDsparkRuntime = Glm52DsparkRuntime<CpuContext>;
pub type CpuDsparkDraftBatch<'a> = Glm52DsparkDraftBatch<'a, CpuContext>;

/// A0 把同一 ready wave 中已经到齐的会话一次投进队列；单路立即执行，
/// 多路共享一次权重扫描，不等待未来会话，也不引入人为凑批延迟。
pub(super) struct CpuDsparkJob {
    pub(super) session: usize,
    pub(super) id: u64,
    pub(super) anchor: u32,
    pub(super) max_drafts: usize,
    pub(super) cache: DsparkTargetCache<CpuTensor>,
    pub(super) history: CpuTensor,
    pub(super) target_position: usize,
    pub(super) block_position: usize,
    pub(super) minimum_drafts: usize,
}

pub(super) struct CpuDsparkResult {
    pub(super) session: usize,
    pub(super) id: u64,
    pub(super) anchor: u32,
    pub(super) cache: DsparkTargetCache<CpuTensor>,
    pub(super) drafts: Result<Vec<u32>, String>,
}

const CPU_DSPARK_WORKERS: usize = 3;
const CPU_DSPARK_MAX_QUEUED_BATCH: usize = 4;

enum CpuDsparkEvent {
    Draft(Vec<CpuDsparkJob>),
    Completed(CpuDsparkCompletion),
    Shutdown,
}

enum CpuDsparkWorkerCommand {
    Draft(Vec<CpuDsparkQueuedJob>),
    Shutdown,
}

struct CpuDsparkQueuedJob {
    job: CpuDsparkJob,
    queued_at: Instant,
}

struct CpuDsparkCompletion {
    worker: usize,
    results: Vec<CpuDsparkResult>,
    jobs: usize,
    compute_micros: u128,
    queue_micros: u128,
}

/// 固定 worker 的无等待分派状态：有空闲 worker 时单路立即发出；只有所有
/// worker 都忙时才进入 queued，完成者只合并已经存在的队列，不启动计时器。
struct CpuDsparkDispatch<T> {
    idle: VecDeque<usize>,
    queued: VecDeque<T>,
}

impl<T> CpuDsparkDispatch<T> {
    fn new(workers: usize) -> Self {
        Self { idle: (0..workers).collect(), queued: VecDeque::new() }
    }

    fn submit_many(&mut self, mut items: Vec<T>, max_batch: usize) -> Option<(usize, Vec<T>)> {
        if items.is_empty() {
            return None;
        }
        match self.idle.pop_front() {
            Some(worker) => {
                let count = max_batch.max(1).min(items.len());
                let batch = items.drain(..count).collect();
                self.queued.extend(items);
                Some((worker, batch))
            }
            None => {
                self.queued.extend(items);
                None
            }
        }
    }

    fn complete(&mut self, worker: usize, max_batch: usize) -> Option<(usize, Vec<T>)> {
        if self.queued.is_empty() {
            self.idle.push_back(worker);
            return None;
        }
        let count = max_batch.max(1).min(self.queued.len());
        Some((worker, self.queued.drain(..count).collect()))
    }
}

pub(super) struct CpuDsparkExecutor {
    sender: mpsc::Sender<CpuDsparkEvent>,
    receiver: mpsc::Receiver<CpuDsparkResult>,
    coordinator: Option<thread::JoinHandle<()>>,
}

impl CpuDsparkExecutor {
    pub(super) fn new(runtime: CpuDsparkRuntime) -> Result<Self, String> {
        let runtime = Arc::new(runtime);
        let (event_tx, event_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let mut worker_senders: Vec<mpsc::Sender<CpuDsparkWorkerCommand>> = Vec::with_capacity(CPU_DSPARK_WORKERS);
        let mut workers: Vec<thread::JoinHandle<()>> = Vec::with_capacity(CPU_DSPARK_WORKERS);
        for worker in 0..CPU_DSPARK_WORKERS {
            let (worker_tx, worker_rx) = mpsc::channel();
            let runtime = Arc::clone(&runtime);
            let event_tx = event_tx.clone();
            let handle = match thread::Builder::new().name(format!("glm52-dspark-cpu-{worker}")).spawn(move || cpu_dspark_worker(worker, runtime, worker_rx, event_tx)) {
                Ok(handle) => handle,
                Err(error) => {
                    for sender in &worker_senders {
                        let _ = sender.send(CpuDsparkWorkerCommand::Shutdown);
                    }
                    for handle in workers {
                        let _ = handle.join();
                    }
                    return Err(format!("启动 CPU DSpark worker {worker}: {error}"));
                }
            };
            worker_senders.push(worker_tx);
            workers.push(handle);
        }
        let coordinator = match thread::Builder::new().name("glm52-dspark-cpu-scheduler".to_owned()).spawn(move || cpu_dspark_coordinator(event_rx, result_tx, worker_senders, workers)) {
            Ok(handle) => handle,
            Err(error) => return Err(format!("启动 CPU DSpark coordinator: {error}")),
        };
        Ok(Self { sender: event_tx, receiver: result_rx, coordinator: Some(coordinator) })
    }

    pub(super) fn submit_batch(&self, jobs: Vec<CpuDsparkJob>) -> Result<(), String> {
        if jobs.is_empty() {
            return Ok(());
        }
        self.sender.send(CpuDsparkEvent::Draft(jobs)).map_err(|_| "CPU DSpark coordinator 已退出".to_owned())
    }

    pub(super) fn drain_ready(&self) -> Vec<CpuDsparkResult> {
        self.receiver.try_iter().collect()
    }
}

impl Drop for CpuDsparkExecutor {
    fn drop(&mut self) {
        let _ = self.sender.send(CpuDsparkEvent::Shutdown);
        if let Some(coordinator) = self.coordinator.take() {
            let _ = coordinator.join();
        }
    }
}

fn dispatch_cpu_dspark_batch(senders: &[mpsc::Sender<CpuDsparkWorkerCommand>], worker: usize, jobs: Vec<CpuDsparkQueuedJob>) -> bool {
    senders.get(worker).is_some_and(|sender| sender.send(CpuDsparkWorkerCommand::Draft(jobs)).is_ok())
}

fn cpu_dspark_coordinator(receiver: mpsc::Receiver<CpuDsparkEvent>, result_sender: mpsc::Sender<CpuDsparkResult>, worker_senders: Vec<mpsc::Sender<CpuDsparkWorkerCommand>>, workers: Vec<thread::JoinHandle<()>>) {
    let mut dispatch = CpuDsparkDispatch::new(worker_senders.len());
    let mut profile_batches = 0usize;
    let mut profile_jobs = 0usize;
    let mut profile_micros = 0u128;
    let mut profile_queue_micros = 0u128;
    loop {
        match receiver.recv() {
            Ok(CpuDsparkEvent::Draft(jobs)) => {
                let queued_at = Instant::now();
                let queued = jobs.into_iter().map(|job| CpuDsparkQueuedJob { job, queued_at }).collect();
                if let Some((worker, jobs)) = dispatch.submit_many(queued, CPU_DSPARK_MAX_QUEUED_BATCH)
                    && !dispatch_cpu_dspark_batch(&worker_senders, worker, jobs)
                {
                    break;
                }
            }
            Ok(CpuDsparkEvent::Completed(completion)) => {
                profile_batches += 1;
                profile_jobs += completion.jobs;
                profile_micros += completion.compute_micros;
                profile_queue_micros += completion.queue_micros;
                for result in completion.results {
                    if result_sender.send(result).is_err() {
                        return;
                    }
                }
                if profile_jobs >= 32 {
                    eprintln!(
                        "[glm52-dspark-cpu-batch] workers={} batches={} jobs={} batch_avg={:.2} batch_ms={:.3} job_ms={:.3} queue_ms={:.3}",
                        worker_senders.len(),
                        profile_batches,
                        profile_jobs,
                        profile_jobs as f64 / profile_batches as f64,
                        profile_micros as f64 / profile_batches as f64 / 1000.0,
                        profile_micros as f64 / profile_jobs as f64 / 1000.0,
                        profile_queue_micros as f64 / profile_jobs as f64 / 1000.0,
                    );
                    profile_batches = 0;
                    profile_jobs = 0;
                    profile_micros = 0;
                    profile_queue_micros = 0;
                }
                if let Some((worker, jobs)) = dispatch.complete(completion.worker, CPU_DSPARK_MAX_QUEUED_BATCH)
                    && !dispatch_cpu_dspark_batch(&worker_senders, worker, jobs)
                {
                    break;
                }
            }
            Ok(CpuDsparkEvent::Shutdown) | Err(_) => break,
        }
    }
    for sender in &worker_senders {
        let _ = sender.send(CpuDsparkWorkerCommand::Shutdown);
    }
    for worker in workers {
        let _ = worker.join();
    }
}

fn cpu_dspark_worker(worker: usize, runtime: Arc<CpuDsparkRuntime>, receiver: mpsc::Receiver<CpuDsparkWorkerCommand>, sender: mpsc::Sender<CpuDsparkEvent>) {
    while let Ok(command) = receiver.recv() {
        let CpuDsparkWorkerCommand::Draft(mut queued) = command else {
            break;
        };
        let trace_lanes = crate::runtime::prefill_scheduler::stage_event_trace_enabled().then(|| queued.iter().map(|queued| format!("{}#{}@{}", queued.job.session, queued.job.id, queued.job.target_position)).collect::<Vec<_>>().join(","));
        let queue_micros = queued.iter().map(|job| job.queued_at.elapsed().as_micros()).sum();
        let mut jobs = queued.drain(..).map(|queued| queued.job).collect::<Vec<_>>();
        let batch_size = jobs.len();
        let batch_started = Instant::now();
        let drafted = {
            let mut batch = jobs
                .iter_mut()
                .map(|job| CpuDsparkDraftBatch {
                    cache: &mut job.cache,
                    anchor: job.anchor,
                    target_hidden: &job.history,
                    target_position: job.target_position,
                    block_position: job.block_position,
                    minimum_drafts: job.minimum_drafts,
                    drafts: Vec::new(),
                })
                .collect::<Vec<_>>();
            match runtime.draft_batch(&CpuContext, &mut batch) {
                Ok(()) => Ok(batch.into_iter().map(|item| item.drafts).collect::<Vec<_>>()),
                Err(error) => Err(format!("CPU DSpark draft batch: {error:?}")),
            }
        };
        let compute_micros = batch_started.elapsed().as_micros();
        if let Some(lanes) = trace_lanes.as_deref() {
            crate::runtime::prefill_scheduler::record_stage_trace(format!(
                "[dspark-work-trace] ts_us={} phase=worker worker={worker} lanes={lanes} jobs={batch_size} queue_us={queue_micros} compute_us={compute_micros}",
                crate::runtime::prefill_scheduler::stage_trace_timestamp_us(),
            ));
        }
        let mut results = Vec::with_capacity(batch_size);
        match drafted {
            Ok(drafts) => {
                for (job, mut drafts) in jobs.into_iter().zip(drafts) {
                    drafts.truncate(job.max_drafts);
                    results.push(CpuDsparkResult { session: job.session, id: job.id, anchor: job.anchor, cache: job.cache, drafts: Ok(drafts) });
                }
            }
            Err(error) => {
                for job in jobs {
                    results.push(CpuDsparkResult { session: job.session, id: job.id, anchor: job.anchor, cache: job.cache, drafts: Err(error.clone()) });
                }
            }
        }
        if sender.send(CpuDsparkEvent::Completed(CpuDsparkCompletion { worker, results, jobs: batch_size, compute_micros, queue_micros })).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::CpuDsparkDispatch;

    #[test]
    fn idle_workers_receive_single_jobs_without_batch_wait() {
        let mut dispatch = CpuDsparkDispatch::new(2);
        assert_eq!(dispatch.submit_many(vec![10], 4), Some((0, vec![10])));
        assert_eq!(dispatch.submit_many(vec![11], 4), Some((1, vec![11])));
        assert_eq!(dispatch.submit_many(vec![12], 4), None);
        assert_eq!(dispatch.complete(0, 4), Some((0, vec![12])));
    }

    #[test]
    fn only_already_queued_jobs_form_a_batch() {
        let mut dispatch = CpuDsparkDispatch::new(2);
        assert_eq!(dispatch.submit_many(vec![10], 4), Some((0, vec![10])));
        assert_eq!(dispatch.submit_many(vec![11], 4), Some((1, vec![11])));
        assert_eq!(dispatch.submit_many(vec![12], 4), None);
        assert_eq!(dispatch.submit_many(vec![13], 4), None);
        assert_eq!(dispatch.complete(0, 4), Some((0, vec![12, 13])));
        assert_eq!(dispatch.complete(1, 4), None);
        assert_eq!(dispatch.submit_many(vec![14], 4), Some((1, vec![14])));
    }

    #[test]
    fn already_ready_wave_uses_one_weight_scan() {
        let mut dispatch = CpuDsparkDispatch::new(3);
        assert_eq!(dispatch.submit_many(vec![10, 11, 12], 4), Some((0, vec![10, 11, 12])));
        assert_eq!(dispatch.submit_many(vec![13], 4), Some((1, vec![13])));
    }
}

static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);

/// 外部 benchmark 可按需打开 phase 计时；生产默认关闭，不引入环境变量分支。
pub fn set_profile_enabled(enabled: bool) {
    PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

impl Glm52DsparkBackend for CpuContext {
    fn dspark_profile_enabled(&self) -> bool {
        PROFILE_ENABLED.load(Ordering::Relaxed)
    }

    fn dspark_tensor_from_bf16_bits(&self, values: Vec<u16>, rows: usize, cols: usize) -> Result<Self::Tensor, BackendError> {
        let expected = rows.checked_mul(cols).ok_or_else(|| compute("CPU DSpark tensor 大小溢出"))?;
        if values.len() != expected {
            return Err(compute(format!("CPU DSpark tensor=[{rows},{cols}]，实际元素={}", values.len())));
        }
        Ok(CpuTensor { data: values.into_iter().map(|value| half::bf16::from_bits(value).to_f32()).collect(), rows, cols })
    }

    fn dspark_tensor_as_f32(&self, tensor: Self::Tensor) -> Result<Self::Tensor, BackendError> {
        Ok(tensor)
    }

    fn dspark_tensor_to_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, BackendError> {
        Ok(tensor.data.clone())
    }

    fn dspark_argmax_add_rows(&self, logits: &Self::Tensor, rows: &[u32], bias: &Self::Tensor) -> Result<Vec<u32>, BackendError> {
        if rows.len() != bias.rows || logits.cols != bias.cols || rows.iter().any(|&row| row as usize >= logits.rows) {
            return Err(compute(format!("CPU DSpark add+argmax logits=[{},{}] rows={rows:?} bias=[{},{}]", logits.rows, logits.cols, bias.rows, bias.cols)));
        }
        // vocab 级加法+argmax 是热路径;AVX-512 两遍实现(求最大,再找最先命中),
        // 与标量实现的并列取最先下标语义一致。
        #[cfg(target_arch = "x86_64")]
        if logits.cols.is_multiple_of(16) && std::arch::is_x86_feature_detected!("avx512f") {
            return (0..rows.len())
                .map(|index| {
                    let row = rows[index] as usize;
                    let logits = &logits.data[row * logits.cols..(row + 1) * logits.cols];
                    let bias = &bias.data[index * bias.cols..(index + 1) * bias.cols];
                    argmax_add_avx512(logits, bias).map_err(|message| compute(format!("CPU DSpark add+argmax row={row}: {message}")))
                })
                .collect();
        }
        rows.iter()
            .enumerate()
            .map(|(index, &row)| {
                let logits = &logits.data[row as usize * logits.cols..(row as usize + 1) * logits.cols];
                let bias = &bias.data[index * bias.cols..(index + 1) * bias.cols];
                let (token, _) = logits
                    .iter()
                    .zip(bias)
                    .enumerate()
                    .map(|(token, (logit, bias))| (token, logit + bias))
                    .max_by(|left, right| left.1.total_cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
                    .ok_or_else(|| compute("CPU DSpark add+argmax 输入为空"))?;
                u32::try_from(token).map_err(|_| compute("CPU DSpark token 超出 u32"))
            })
            .collect()
    }
}

/// `argmax(logits + bias)`,并列值取最先下标(与标量 max_by 语义一致)。
#[cfg(target_arch = "x86_64")]
fn argmax_add_avx512(logits: &[f32], bias: &[f32]) -> Result<u32, String> {
    use std::arch::x86_64::*;
    if logits.is_empty() {
        return Err("加法+argmax 输入为空".to_owned());
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn scan_maximum(logits: &[f32], bias: &[f32]) -> f32 {
        unsafe {
            let mut maximum = _mm512_set1_ps(f32::NEG_INFINITY);
            for (logit, bias) in logits.chunks_exact(16).zip(bias.chunks_exact(16)) {
                let value = _mm512_add_ps(_mm512_loadu_ps(logit.as_ptr()), _mm512_loadu_ps(bias.as_ptr()));
                maximum = _mm512_max_ps(maximum, value);
            }
            _mm512_reduce_max_ps(maximum)
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn first_equal(logits: &[f32], bias: &[f32], maximum: f32) -> Option<usize> {
        unsafe {
            let target = _mm512_set1_ps(maximum);
            for (offset, (logit, bias)) in logits.chunks_exact(16).zip(bias.chunks_exact(16)).enumerate() {
                let value = _mm512_add_ps(_mm512_loadu_ps(logit.as_ptr()), _mm512_loadu_ps(bias.as_ptr()));
                let mask = _mm512_cmp_ps_mask(value, target, _CMP_EQ_OQ);
                if mask != 0 {
                    return Some(offset * 16 + mask.trailing_zeros() as usize);
                }
            }
            None
        }
    }

    unsafe {
        let mut maximum = scan_maximum(logits, bias);
        let tail = logits.len() / 16 * 16;
        for index in tail..logits.len() {
            maximum = maximum.max(logits[index] + bias[index]);
        }
        let index = first_equal(logits, bias, maximum).or_else(|| (tail..logits.len()).find(|&index| logits[index] + bias[index] == maximum)).unwrap_or(0);
        u32::try_from(index).map_err(|_| "argmax 下标超出 u32".to_owned())
    }
}

fn compute(msg: impl Into<String>) -> BackendError {
    BackendError::Compute { msg: msg.into() }
}
