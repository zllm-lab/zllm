use super::*;

pub struct DeviceBuffer {
    pub(super) device_id: i32,
    pub(super) pointer: *mut c_void,
    pub(super) bytes: usize,
    pub(super) capacity_bytes: usize,
    pub(super) recyclable: bool,
    /// 双机边界输入由消费 stage 的 completion 保持最后一个 Arc；只有该
    /// completion 退休后，DeviceBuffer 的 Drop 才能把槽放回 cache。
    pub(super) retain_until_stage_completion: bool,
    pub(super) stage_completion_ready: std::sync::atomic::AtomicBool,
    deferred_host: Option<CachedPinnedHostBuffer>,
    deferred_upload_enqueued: std::sync::atomic::AtomicBool,
    pub(super) async_allocated: bool,
    pub(super) owner: Option<std::sync::Arc<DeviceBuffer>>,
}

pub(super) const DEVICE_POOL_MAX_BUFFER_BYTES: usize = 384 * 1024 * 1024;
static DEVICE_POOL_MAX_BYTES_PER_DEVICE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(3 * 1024 * 1024 * 1024);
pub(super) const DEVICE_ASYNC_MAX_BUFFER_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// stage completion 批量回收只适合短工作。长 prefill 若把所有层的临时 tensor
/// 都留到 stage 尾部，会在逻辑引用已经释放后仍占满整卡显存。超过水位后退回逐
/// buffer event，后续同 shape allocation 可在同一 stage 内安全复用。
pub(super) const STAGE_DEFERRED_BUFFER_MAX_BYTES: usize = 512 * 1024 * 1024;

pub(super) struct PendingDeviceBuffer {
    pointer: usize,
    event: usize,
}

struct PendingP2pSource {
    sources: Vec<std::sync::Arc<DeviceBuffer>>,
    event: usize,
}

#[derive(Default)]
pub(super) struct DeviceBufferPool {
    buffers: HashMap<(i32, usize), Vec<usize>>,
    pending: HashMap<(i32, usize), Vec<PendingDeviceBuffer>>,
    bytes: HashMap<i32, usize>,
    available_events: Vec<usize>,
}

pub(super) static DEVICE_BUFFER_POOLS: OnceLock<Vec<Mutex<DeviceBufferPool>>> = OnceLock::new();
pub(super) static DEVICE_BUFFER_REUSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[derive(Default)]
struct PoolMissProfile {
    count: u64,
    micros: u64,
}

static POOL_MISS_PROFILES: OnceLock<Mutex<HashMap<(i32, usize, &'static str, u32), PoolMissProfile>>> = OnceLock::new();

fn record_pool_miss(device_id: i32, bytes: usize, caller: &'static std::panic::Location<'static>, started: Option<std::time::Instant>) {
    let Some(started) = started else { return };
    let elapsed = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    let Ok(mut profiles) = POOL_MISS_PROFILES.get_or_init(|| Mutex::new(HashMap::new())).lock() else { return };
    let profile = profiles.entry((device_id, bytes, caller.file(), caller.line())).or_default();
    profile.count = profile.count.saturating_add(1);
    profile.micros = profile.micros.saturating_add(elapsed);
}

fn report_pool_misses(label: Option<&str>) {
    let Some(profiles) = POOL_MISS_PROFILES.get() else { return };
    let Ok(mut profiles) = profiles.lock() else { return };
    if profiles.is_empty() {
        return;
    }
    let mut entries = std::mem::take(&mut *profiles).into_iter().collect::<Vec<_>>();
    drop(profiles);
    entries.sort_unstable_by_key(|((_, bytes, _, _), profile)| std::cmp::Reverse(bytes.saturating_mul(profile.count as usize)));
    let total_count = entries.iter().map(|(_, profile)| profile.count).sum::<u64>();
    let total_bytes = entries.iter().map(|((_, bytes, _, _), profile)| bytes.saturating_mul(profile.count as usize)).sum::<usize>();
    let total_micros = entries.iter().map(|(_, profile)| profile.micros).sum::<u64>();
    let top = entries
        .iter()
        .take(32)
        .map(|((device_id, bytes, file, line), profile)| format!("d{device_id}:{}x{}/{:.1}ms@{}:{line}", bytes, profile.count, profile.micros as f64 / 1000.0, file.rsplit('/').next().unwrap_or(file)))
        .collect::<Vec<_>>()
        .join(",");
    let label = label.map_or(String::new(), |label| format!(" label={label}"));
    eprintln!("[hip-pool-miss]{label} count={total_count} bytes={total_bytes} wall_ms={:.1} top={top}", total_micros as f64 / 1000.0);
}

thread_local! {
    static HOST_TRANSFER_BYTES: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// 调用方已等待目标 device 的 ordered P2P stream 后，源 allocation 可以退休。
/// 通用 stage scheduler 由 `DeviceCompletion` 持有这些引用；H3 的严格 collective
/// barrier 不创建 stage completion，因此在全 rank 同步后显式调用本函数。
pub(crate) fn retire_pending_p2p_sources(device_id: i32) {
    let sources = PENDING_P2P_SOURCES.with(|sources| sources.borrow_mut().remove(&device_id).unwrap_or_default());
    recycle_pending_p2p_sources(sources);
}

fn recycle_pending_p2p_sources(sources: Vec<PendingP2pSource>) {
    for source in sources {
        let Some(source_device_id) = source.sources.first().map(|source| source.device_id) else { continue };
        if let Some(pool) = device_buffer_pool(source_device_id)
            && let Ok(mut pool) = pool.lock()
        {
            pool.available_events.push(source.event);
            continue;
        }
        if set_device(source_device_id).is_err() {
            continue;
        }
        let Ok(runtime) = RocmRuntime::open() else { continue };
        let Ok(destroy) = runtime.event_destroy() else { continue };
        let stats_started = hip_api_stats::start();
        let _ = unsafe { destroy(source.event as HipEvent) };
        hip_api_stats::counted(hip_api_stats::EVENT_DESTROY, stats_started);
    }
}

/// 进程内显式 host/device 复制累计值。runtime 用差分快照守住 resident 热路径；
/// D2D/P2P 不计入，因为它们从未经过 host。
pub(crate) fn host_transfer_bytes() -> (u64, u64) {
    HOST_TRANSFER_BYTES.with(std::cell::Cell::get)
}

fn record_host_transfer(host_to_device: bool, bytes: usize) {
    let bytes = bytes.min(u64::MAX as usize) as u64;
    HOST_TRANSFER_BYTES.with(|totals| {
        let (h2d, d2h) = totals.get();
        totals.set(if host_to_device { (h2d.saturating_add(bytes), d2h) } else { (h2d, d2h.saturating_add(bytes)) });
    });
}

/// kernel_profile 门控的 HIP API 聚合计时。decode 单路时间几乎全部落在
/// host 侧 HIP 调用上,按调用类别计数与计时可以把 stage 链耗时归因到具体 API。
pub(crate) mod hip_api_stats {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Instant;

    pub(crate) const KINDS: &[&str] = &[
        "launch",
        "malloc_async",
        "free_async",
        "malloc_sync",
        "memcpy_sync",
        "memcpy_async",
        "memcpy_peer_async",
        "event_create",
        "event_destroy",
        "event_record",
        "event_query",
        "event_synchronize",
        "stream_wait_event",
        "pool_take_hit",
        "pool_take_miss",
    ];
    static COUNTS: [AtomicU64; KINDS.len()] = [const { AtomicU64::new(0) }; KINDS.len()];
    static MICROS: [AtomicU64; KINDS.len()] = [const { AtomicU64::new(0) }; KINDS.len()];
    static REPORTING: AtomicBool = AtomicBool::new(false);

    pub(crate) const LAUNCH: usize = 0;
    pub(crate) const MALLOC_ASYNC: usize = 1;
    pub(crate) const FREE_ASYNC: usize = 2;
    pub(crate) const MALLOC_SYNC: usize = 3;
    pub(crate) const MEMCPY_SYNC: usize = 4;
    pub(crate) const MEMCPY_ASYNC: usize = 5;
    pub(crate) const MEMCPY_PEER_ASYNC: usize = 6;
    pub(crate) const EVENT_CREATE: usize = 7;
    pub(crate) const EVENT_DESTROY: usize = 8;
    pub(crate) const EVENT_RECORD: usize = 9;
    pub(crate) const EVENT_QUERY: usize = 10;
    pub(crate) const EVENT_SYNCHRONIZE: usize = 11;
    pub(crate) const STREAM_WAIT_EVENT: usize = 12;
    pub(crate) const POOL_TAKE_HIT: usize = 13;
    pub(crate) const POOL_TAKE_MISS: usize = 14;

    /// 记录一次调用;launch 是最高频类别,每 65536 次打一行快照。
    pub(crate) fn record(kind: usize, started: Instant) {
        let elapsed = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        COUNTS[kind].fetch_add(1, Ordering::Relaxed);
        MICROS[kind].fetch_add(elapsed, Ordering::Relaxed);
        if kind == LAUNCH && COUNTS[LAUNCH].load(Ordering::Relaxed) % 65536 == 0 && !REPORTING.swap(true, Ordering::AcqRel) {
            let mut line = String::new();
            for (index, name) in KINDS.iter().enumerate() {
                let count = COUNTS[index].swap(0, Ordering::Relaxed);
                if count == 0 {
                    continue;
                }
                let micros = MICROS[index].swap(0, Ordering::Relaxed);
                line.push_str(&format!(" {name}={count}/{:.1}ms", micros as f64 / 1000.0));
            }
            eprintln!("[hip-api]{line}");
            super::report_pool_misses(None);
            REPORTING.store(false, Ordering::Release);
        }
    }

    /// 诊断开关：`kernel_profile` 会附带各相位设备同步，纯主机侧 API 统计只需要
    /// 本开关（ZLLM_HIP_API_STATS=1），不引入任何同步点。
    fn env_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("ZLLM_HIP_API_STATS").is_ok_and(|value| value == "1"))
    }

    pub(crate) fn start() -> Option<Instant> {
        (crate::kernel::rocm::hip::options().kernel_profile || env_enabled()).then(Instant::now)
    }

    pub(crate) fn counted(kind: usize, started: Option<Instant>) {
        if let Some(started) = started {
            record(kind, started);
        }
    }

    /// profile 周期结束时输出并清零累计值，短 decode 不必等到固定 launch 阈值。
    pub(crate) fn report() {
        report_labeled(None);
    }

    /// 在模型边界强制切片；只在 kernel_profile 下生效，不同步设备。
    pub(crate) fn report_phase(label: &str) {
        report_labeled(Some(label));
    }

    fn report_labeled(label: Option<&str>) {
        if !(crate::kernel::rocm::hip::options().kernel_profile || env_enabled()) || REPORTING.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut line = String::new();
        for (index, name) in KINDS.iter().enumerate() {
            let count = COUNTS[index].swap(0, Ordering::Relaxed);
            if count == 0 {
                continue;
            }
            let micros = MICROS[index].swap(0, Ordering::Relaxed);
            line.push_str(&format!(" {name}={count}/{:.1}ms", micros as f64 / 1000.0));
        }
        if !line.is_empty() {
            let label = label.map_or(String::new(), |label| format!(" label={label}"));
            eprintln!("[hip-api]{label}{line}");
        }
        super::report_pool_misses(label);
        REPORTING.store(false, Ordering::Release);
    }
}

thread_local! {
    /// 异步 P2P 的源 buffer 与 source event 由目标 stage completion 持有，
    /// 复制结束前都不能回池复用。
    static PENDING_P2P_SOURCES: RefCell<HashMap<i32, Vec<PendingP2pSource>>> = RefCell::new(HashMap::new());
    /// 边界输入的对象生命周期由消费 stage completion 直接持有，不能只在
    /// tensor Drop 时转交裸 pointer。
    static PENDING_STAGE_INPUTS: RefCell<HashMap<i32, Vec<std::sync::Arc<DeviceBuffer>>>> = RefCell::new(HashMap::new());
    /// 同一 stage 的临时 buffer 共用 completion event，避免每个 Drop 单独录制事件。
    static STAGE_BUFFER_RECYCLES: RefCell<HashMap<i32, StageBufferRecycleBatch>> = RefCell::new(HashMap::new());
}

#[derive(Default)]
pub(super) struct StageBufferRecycleBatch {
    active: bool,
    bytes: usize,
    buffers: Vec<(usize, usize)>,
}

pub(crate) fn begin_stage_buffer_recycle(device_id: i32) -> Result<(), String> {
    if !options().memory_pool {
        return Ok(());
    }
    STAGE_BUFFER_RECYCLES.with(|batches| {
        let mut batches = batches.borrow_mut();
        let batch = batches.entry(device_id).or_default();
        if batch.active {
            return Err(format!("ROCm device {device_id} stage buffer 回收批次重复开始"));
        }
        batch.active = true;
        Ok(())
    })
}

pub(super) fn defer_stage_buffer_recycle(device_id: i32, pointer: *mut c_void, bytes: usize) -> bool {
    if !options().memory_pool || pointer.is_null() {
        return false;
    }
    STAGE_BUFFER_RECYCLES
        .try_with(|batches| {
            let mut batches = batches.borrow_mut();
            let Some(batch) = batches.get_mut(&device_id) else { return false };
            if !batch.active || batch.bytes.saturating_add(bytes) > STAGE_DEFERRED_BUFFER_MAX_BYTES {
                return false;
            }
            batch.bytes += bytes;
            batch.buffers.push((pointer as usize, bytes));
            true
        })
        .unwrap_or(false)
}

pub(super) fn take_stage_buffer_recycles(device_id: i32) -> Vec<(usize, usize)> {
    STAGE_BUFFER_RECYCLES.with(|batches| {
        let mut batches = batches.borrow_mut();
        let Some(batch) = batches.get_mut(&device_id) else { return Vec::new() };
        batch.active = false;
        batch.bytes = 0;
        std::mem::take(&mut batch.buffers)
    })
}

/// 显式池超限时优先释放大块 prefill scratch，给高频的小块 decode shape 留空间。
/// `pool.bytes` 同时包含 pending，不能触碰尚未完成 event 的 pointer。
fn enforce_device_buffer_pool_limit(pool: &mut DeviceBufferPool, device_id: i32) -> Vec<usize> {
    let limit = DEVICE_POOL_MAX_BYTES_PER_DEVICE.load(std::sync::atomic::Ordering::Relaxed);
    let mut used = pool.bytes.get(&device_id).copied().unwrap_or(0);
    let mut keys = pool.buffers.keys().copied().filter(|&(buffer_device, _)| buffer_device == device_id).collect::<Vec<_>>();
    keys.sort_unstable_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
    let mut pointers_to_free = Vec::new();
    for key @ (_, bytes) in keys {
        let Some(pointers) = pool.buffers.remove(&key) else { continue };
        let mut reusable = Vec::new();
        for pointer in pointers {
            if bytes > DEVICE_POOL_MAX_BUFFER_BYTES || used > limit {
                used = used.saturating_sub(bytes);
                pointers_to_free.push(pointer);
            } else {
                reusable.push(pointer);
            }
        }
        if !reusable.is_empty() {
            pool.buffers.insert(key, reusable);
        }
    }
    if used == 0 {
        pool.bytes.remove(&device_id);
    } else {
        pool.bytes.insert(device_id, used);
    }
    pointers_to_free
}

pub(super) fn recycle_completed_stage_buffers(device_id: i32, buffers: Vec<(usize, usize)>) {
    if buffers.is_empty() {
        return;
    }
    let Some(pool) = device_buffer_pool(device_id) else { return };
    let Ok(mut pool) = pool.lock() else { return };
    let mut used = pool.bytes.get(&device_id).copied().unwrap_or(0);
    for (pointer, bytes) in buffers {
        pool.buffers.entry((device_id, bytes)).or_default().push(pointer);
        used = used.saturating_add(bytes);
    }
    pool.bytes.insert(device_id, used);
    let pointers_to_free = enforce_device_buffer_pool_limit(&mut pool, device_id);
    drop(pool);
    if pointers_to_free.is_empty() || set_device(device_id).is_err() {
        return;
    }
    let Ok(runtime) = RocmRuntime::open() else { return };
    let Ok(free) = runtime.free() else { return };
    let free_count = pointers_to_free.len();
    let free_started = std::time::Instant::now();
    let free_start_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
    for pointer in pointers_to_free {
        let _ = unsafe { free(pointer as *mut c_void) };
    }
    let free_micros = free_started.elapsed().as_micros();
    if free_micros >= 20_000 {
        let complete_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
        eprintln!("[rocm-buffer-pool-trim-slow] ts_us={free_start_us} device={device_id} path=stage-retire buffers={free_count} duration_us={free_micros} complete_us={complete_us}");
    }
}

pub(crate) fn abort_stage_buffer_recycle(device_id: i32) -> Result<(), String> {
    synchronize_device(device_id, "abort stage buffer recycle")?;
    PENDING_STAGE_INPUTS.with(|inputs| {
        if let Some(inputs) = inputs.borrow_mut().remove(&device_id) {
            for input in inputs {
                input.stage_completion_ready.store(true, std::sync::atomic::Ordering::Release);
            }
        }
    });
    recycle_completed_stage_buffers(device_id, take_stage_buffer_recycles(device_id));
    promote_device_buffers(device_id);
    Ok(())
}

/// 当前 submission stream 上一次提交的完成标记。只在源 device 上轮询，避免跨卡 event wait。
pub(crate) struct DeviceCompletion {
    device_id: i32,
    event: usize,
    p2p_sources: Mutex<Vec<PendingP2pSource>>,
    stage_inputs: Vec<std::sync::Arc<DeviceBuffer>>,
    recycled: Mutex<Vec<(usize, usize)>>,
    retired: std::sync::atomic::AtomicBool,
}

impl DeviceCompletion {
    pub(crate) fn record(device_id: i32) -> Result<Self, String> {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let create = runtime.event_create()?;
        let record = runtime.event_record()?;
        let destroy = runtime.event_destroy()?;
        let mut event = device_buffer_pool(device_id).and_then(|pool| pool.lock().ok()?.available_events.pop()).map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
        if event.is_null() {
            let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventCreateWithFlags stage completion"));
            }
        }
        let stats_started = hip_api_stats::start();
        let status = unsafe { record(event, crate::kernel::rocm::hip::active_compute_stream()) };
        hip_api_stats::counted(hip_api_stats::EVENT_RECORD, stats_started);
        if status != HIP_SUCCESS {
            let stats_started = hip_api_stats::start();
            let _ = unsafe { destroy(event) };
            hip_api_stats::counted(hip_api_stats::EVENT_DESTROY, stats_started);
            return Err(runtime.hip_error(status, "hipEventRecord stage completion"));
        }
        let p2p_sources = PENDING_P2P_SOURCES.with(|sources| sources.borrow_mut().remove(&device_id).unwrap_or_default());
        let stage_inputs = PENDING_STAGE_INPUTS.with(|inputs| inputs.borrow_mut().remove(&device_id).unwrap_or_default());
        let recycled = take_stage_buffer_recycles(device_id);
        Ok(Self { device_id, event: event as usize, p2p_sources: Mutex::new(p2p_sources), stage_inputs, recycled: Mutex::new(recycled), retired: std::sync::atomic::AtomicBool::new(false) })
    }

    fn finish_stage_inputs(&self) {
        for input in &self.stage_inputs {
            input.stage_completion_ready.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    fn recycle_p2p_sources(&self) {
        let Ok(mut sources) = self.p2p_sources.lock() else { return };
        recycle_pending_p2p_sources(std::mem::take(&mut *sources));
    }

    pub(super) fn recycle_buffers(&self) {
        let Ok(mut buffers) = self.recycled.lock() else { return };
        recycle_completed_stage_buffers(self.device_id, std::mem::take(&mut *buffers));
    }

    pub(crate) fn is_complete(&self) -> Result<bool, String> {
        if self.retired.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(true);
        }
        set_device(self.device_id)?;
        let runtime = RocmRuntime::open()?;
        let query = runtime.event_query()?;
        let stats_started = hip_api_stats::start();
        let query_status = unsafe { query(self.event as HipEvent) };
        hip_api_stats::counted(hip_api_stats::EVENT_QUERY, stats_started);
        match query_status {
            HIP_SUCCESS => {
                self.recycle_p2p_sources();
                self.finish_stage_inputs();
                self.recycle_buffers();
                // stage event 覆盖当前 submission stream 的所有临时 buffer，及时归还已完成 allocation，
                // 避免取消整卡同步后 pending pool 只增不减。
                promote_device_buffers(self.device_id);
                self.retired.store(true, std::sync::atomic::Ordering::Release);
                Ok(true)
            }
            HIP_ERROR_NOT_READY => Ok(false),
            status => Err(runtime.hip_error(status, "hipEventQuery stage completion")),
        }
    }

    pub(crate) fn wait(&self) -> Result<(), String> {
        if self.retired.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(());
        }
        set_device(self.device_id)?;
        let runtime = RocmRuntime::open()?;
        let synchronize = runtime.event_synchronize()?;
        let stats_started = hip_api_stats::start();
        let status = unsafe { synchronize(self.event as HipEvent) };
        hip_api_stats::counted(hip_api_stats::EVENT_SYNCHRONIZE, stats_started);
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipEventSynchronize stage completion"));
        }
        self.recycle_p2p_sources();
        self.finish_stage_inputs();
        self.recycle_buffers();
        promote_device_buffers(self.device_id);
        self.retired.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// ordered 链尾 completion 已完成时，前序同一 stream 也必然完成。
    pub(crate) fn retire_ordered(&self) {
        self.recycle_p2p_sources();
        self.finish_stage_inputs();
        self.recycle_buffers();
        self.retired.store(true, std::sync::atomic::Ordering::Release);
    }
}

impl Drop for DeviceCompletion {
    fn drop(&mut self) {
        if self.event == 0 {
            return;
        }
        if self.retired.load(std::sync::atomic::Ordering::Acquire)
            && let Some(pool) = device_buffer_pool(self.device_id)
            && let Ok(mut pool) = pool.lock()
        {
            pool.available_events.push(self.event);
            return;
        }
        if set_device(self.device_id).is_err() {
            return;
        }
        let Ok(runtime) = RocmRuntime::open() else { return };
        let Ok(query) = runtime.event_query() else { return };
        let complete = if unsafe { query(self.event as HipEvent) } == HIP_SUCCESS {
            true
        } else if let Ok(synchronize) = runtime.event_synchronize() {
            (unsafe { synchronize(self.event as HipEvent) }) == HIP_SUCCESS
        } else {
            false
        };
        if complete {
            self.recycle_p2p_sources();
            self.finish_stage_inputs();
            self.recycle_buffers();
            promote_device_buffers(self.device_id);
        }
        if complete
            && let Some(pool) = device_buffer_pool(self.device_id)
            && let Ok(mut pool) = pool.lock()
        {
            pool.available_events.push(self.event);
            return;
        }
        if let Ok(destroy) = runtime.event_destroy() {
            let stats_started = hip_api_stats::start();
            let _ = unsafe { destroy(self.event as HipEvent) };
            hip_api_stats::counted(hip_api_stats::EVENT_DESTROY, stats_started);
        }
    }
}

pub(super) fn device_buffer_pool(device_id: i32) -> Option<&'static Mutex<DeviceBufferPool>> {
    let count = usize::try_from(*HIP_DEVICE_COUNT.get()?.as_ref().ok()?).ok()?;
    let pools = DEVICE_BUFFER_POOLS.get_or_init(|| (0..count).map(|_| Mutex::new(DeviceBufferPool::default())).collect());
    pools.get(usize::try_from(device_id).ok()?)
}

fn take_device_buffer_bounded(device_id: i32, bytes: usize, max_capacity: usize) -> Option<(*mut c_void, usize)> {
    if !options().memory_pool {
        return None;
    }
    let stats_started = hip_api_stats::start();
    let outcome = take_device_buffer_inner(device_id, bytes, max_capacity);
    hip_api_stats::counted(if outcome.is_some() { hip_api_stats::POOL_TAKE_HIT } else { hip_api_stats::POOL_TAKE_MISS }, stats_started);
    outcome
}

pub(super) fn take_device_buffer_inner(device_id: i32, bytes: usize, max_capacity: usize) -> Option<(*mut c_void, usize)> {
    let pool = device_buffer_pool(device_id)?;
    let mut pool = pool.lock().ok()?;
    let key = pool.buffers.iter().filter(|(key, pointers)| key.0 == device_id && key.1 >= bytes && key.1 <= max_capacity && !pointers.is_empty()).map(|(key, _)| *key).min_by_key(|&(_, capacity)| capacity);
    if let Some(key @ (_, capacity)) = key {
        let pointer = pool.buffers.get_mut(&key).and_then(Vec::pop).expect("best-fit key 已检查非空");
        let used = pool.bytes.entry(device_id).or_default();
        debug_assert!(*used >= capacity, "HIP 池计账下溢: device={device_id} capacity={capacity} used={used}");
        *used = used.saturating_sub(capacity);
        return Some((pointer as *mut c_void, capacity));
    }

    let mut pending_keys = pool.pending.keys().copied().filter(|&(buffer_device, capacity)| buffer_device == device_id && capacity >= bytes && capacity <= max_capacity).collect::<Vec<_>>();
    pending_keys.sort_unstable_by_key(|&(_, capacity)| capacity);
    if pending_keys.is_empty() {
        return None;
    }
    let runtime = RocmRuntime::open().ok()?;
    let query = runtime.event_query().ok()?;
    let mut completed = None;
    for key @ (_, capacity) in pending_keys {
        let found = {
            let pending = pool.pending.get_mut(&key).expect("pending key 来自当前 map");
            pending.iter().rposition(|buffer| unsafe { query(buffer.event as HipEvent) } == HIP_SUCCESS).map(|index| (pending.swap_remove(index), pending.is_empty()))
        };
        if let Some((buffer, remove_key)) = found {
            if remove_key {
                pool.pending.remove(&key);
            }
            completed = Some((buffer, capacity));
            break;
        }
    }
    let (buffer, capacity) = completed?;
    let used = pool.bytes.entry(device_id).or_default();
    debug_assert!(*used >= capacity, "HIP 池计账下溢: device={device_id} capacity={capacity} used={used}");
    *used = used.saturating_sub(capacity);
    pool.available_events.push(buffer.event);
    Some((buffer.pointer as *mut c_void, capacity))
}

pub(super) fn recycle_device_buffer(device_id: i32, pointer: *mut c_void, bytes: usize) -> bool {
    if !options().memory_pool || pointer.is_null() || set_device(device_id).is_err() {
        return false;
    }
    let Ok(runtime) = RocmRuntime::open() else { return false };
    let Ok(create) = runtime.event_create() else {
        return false;
    };
    let Ok(record) = runtime.event_record() else {
        return false;
    };
    let Ok(destroy) = runtime.event_destroy() else {
        return false;
    };
    let mut event = device_buffer_pool(device_id).and_then(|pool| pool.lock().ok()?.available_events.pop()).map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
    if event.is_null() && unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) } != HIP_SUCCESS {
        return false;
    }
    if unsafe { record(event, crate::kernel::rocm::hip::active_compute_stream()) } != HIP_SUCCESS {
        let stats_started = hip_api_stats::start();
        let _ = unsafe { destroy(event) };
        hip_api_stats::counted(hip_api_stats::EVENT_DESTROY, stats_started);
        return false;
    }

    let Some(pool) = device_buffer_pool(device_id) else { return false };
    let Ok(mut pool) = pool.lock() else {
        let stats_started = hip_api_stats::start();
        let _ = unsafe { destroy(event) };
        hip_api_stats::counted(hip_api_stats::EVENT_DESTROY, stats_started);
        return false;
    };
    let used = pool.bytes.get(&device_id).copied().unwrap_or(0);
    pool.pending.entry((device_id, bytes)).or_default().push(PendingDeviceBuffer { pointer: pointer as usize, event: event as usize });
    pool.bytes.insert(device_id, used.saturating_add(bytes));
    true
}

pub(super) fn promote_device_buffers(device_id: i32) {
    if set_device(device_id).is_err() {
        return;
    }
    let Some(pool) = device_buffer_pool(device_id) else { return };
    let Ok(runtime) = RocmRuntime::open() else { return };
    let Ok(query) = runtime.event_query() else { return };
    let Ok(free) = runtime.free() else { return };
    let pointers_to_free = {
        let Ok(mut pool) = pool.lock() else { return };
        let pending_keys = pool.pending.keys().copied().filter(|&(buffer_device, _)| buffer_device == device_id).collect::<Vec<_>>();
        for key in pending_keys {
            let Some(pending) = pool.pending.remove(&key) else { continue };
            let mut incomplete = Vec::new();
            for buffer in pending {
                if unsafe { query(buffer.event as HipEvent) } == HIP_SUCCESS {
                    pool.buffers.entry(key).or_default().push(buffer.pointer);
                    pool.available_events.push(buffer.event);
                } else {
                    incomplete.push(buffer);
                }
            }
            if !incomplete.is_empty() {
                pool.pending.insert(key, incomplete);
            }
        }

        enforce_device_buffer_pool_limit(&mut pool, device_id)
    };
    let free_count = pointers_to_free.len();
    let free_started = std::time::Instant::now();
    let free_start_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
    for pointer in pointers_to_free {
        let _ = unsafe { free(pointer as *mut c_void) };
    }
    let free_micros = free_started.elapsed().as_micros();
    if free_micros >= 20_000 {
        let complete_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
        eprintln!("[rocm-buffer-pool-trim-slow] ts_us={free_start_us} device={device_id} path=promote buffers={free_count} duration_us={free_micros} complete_us={complete_us}");
    }
}

/// 新 shape 在显存紧张时不能复用旧 shape；OOM 后只释放已经完成并进入
/// available 池的 buffer，pending buffer 仍由各自 event 保护。
pub(super) fn release_available_device_buffers(device_id: i32) -> usize {
    if set_device(device_id).is_err() {
        return 0;
    }
    let Some(pool) = device_buffer_pool(device_id) else { return 0 };
    let Ok(runtime) = RocmRuntime::open() else { return 0 };
    let Ok(free) = runtime.free() else { return 0 };
    let (pointers, released) = {
        let Ok(mut pool) = pool.lock() else { return 0 };
        let keys = pool.buffers.keys().copied().filter(|&(buffer_device, _)| buffer_device == device_id).collect::<Vec<_>>();
        let mut pointers = Vec::new();
        let mut released = 0usize;
        for key @ (_, bytes) in keys {
            let Some(buffers) = pool.buffers.remove(&key) else { continue };
            released = released.saturating_add(bytes.saturating_mul(buffers.len()));
            pointers.extend(buffers);
        }
        let remaining = pool.bytes.get(&device_id).copied().unwrap_or(0).saturating_sub(released);
        if remaining == 0 {
            pool.bytes.remove(&device_id);
        } else {
            pool.bytes.insert(device_id, remaining);
        }
        (pointers, released)
    };
    for pointer in pointers {
        let _ = unsafe { free(pointer as *mut c_void) };
    }
    released
}

pub(super) fn explicit_device_pool_enabled(_reusable: bool) -> bool {
    options().legacy_pool || DEVICE_BUFFER_REUSE.load(std::sync::atomic::Ordering::Acquire)
}

pub fn enable_device_buffer_reuse() {
    DEVICE_BUFFER_REUSE.store(true, std::sync::atomic::Ordering::Release);
}

/// 显式池只缓存已完成的临时 allocation；OOM 路径仍可全部驱逐，为 KV 等
/// 长生命周期状态让路。调用方必须在创建 context 前设置本机软水位。
pub fn set_device_buffer_pool_limit(bytes: usize) -> Result<(), String> {
    if bytes == 0 {
        return Err("ROCm device buffer pool limit 需大于 0".to_owned());
    }
    DEVICE_POOL_MAX_BYTES_PER_DEVICE.store(bytes, std::sync::atomic::Ordering::Release);
    Ok(())
}

pub(super) fn release_device_buffer_pool(device_id: i32) {
    if set_device(device_id).is_err() {
        return;
    }
    let Some(pool) = device_buffer_pool(device_id) else { return };
    let (pointers, events) = {
        let Ok(mut pool) = pool.lock() else { return };
        let mut pointers = Vec::new();
        let mut events = std::mem::take(&mut pool.available_events);
        pool.buffers.retain(|&(buffer_device, _), buffers| {
            if buffer_device == device_id {
                pointers.extend(buffers.drain(..));
                false
            } else {
                true
            }
        });
        pool.pending.retain(|&(buffer_device, _), buffers| {
            if buffer_device == device_id {
                for buffer in buffers.drain(..) {
                    pointers.push(buffer.pointer);
                    events.push(buffer.event);
                }
                false
            } else {
                true
            }
        });
        pool.bytes.remove(&device_id);
        (pointers, events)
    };
    let Ok(runtime) = RocmRuntime::open() else { return };
    let Ok(free) = runtime.free() else { return };
    if let Ok(destroy) = runtime.event_destroy() {
        for event in events {
            let stats_started = hip_api_stats::start();
            let _ = unsafe { destroy(event as HipEvent) };
            hip_api_stats::counted(hip_api_stats::EVENT_DESTROY, stats_started);
        }
    }
    for pointer in pointers {
        let _ = unsafe { free(pointer as *mut c_void) };
    }
}

pub(super) struct PinnedHostBuffer {
    pointer: *mut c_void,
    bytes: usize,
}

/// 在 compute stream 上提交 D2H 后，把等待移动到消费线程。
/// staging 与 event 常驻复用；分段接口避免额外的 device pack buffer。
pub(crate) struct AsyncHostDownload {
    device_id: i32,
    event: HipEvent,
    staging: PinnedHostBuffer,
    bytes: usize,
    pending: bool,
}

unsafe impl Send for AsyncHostDownload {}

impl AsyncHostDownload {
    pub(crate) fn new(device_id: i32, bytes: usize) -> Result<Self, String> {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let create = runtime.event_create()?;
        let destroy = runtime.event_destroy()?;
        let mut event = ptr::null_mut();
        let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipEventCreateWithFlags async D2H"));
        }
        match PinnedHostBuffer::allocate(bytes.max(1)) {
            Ok(staging) => Ok(Self { device_id, event, staging, bytes: 0, pending: false }),
            Err(error) => {
                let _ = unsafe { destroy(event) };
                Err(error)
            }
        }
    }

    pub(crate) fn enqueue(&mut self, source: &DeviceBuffer, bytes: usize) -> Result<(), String> {
        self.enqueue_segments(&[(source, 0, bytes)])
    }

    /// 在同一 producer 边界后把多个 device 片段直接拼入 pinned staging，
    /// 避免先在 compute stream 上做小块 D2D pack。
    pub(crate) fn enqueue_segments(&mut self, sources: &[(&DeviceBuffer, usize, usize)]) -> Result<(), String> {
        if self.pending {
            return Err("async D2H 上一份传输尚未完成".to_owned());
        }
        if sources.is_empty() {
            return Err("async D2H segments 不能为空".to_owned());
        }
        let mut bytes = 0usize;
        for &(source, offset, length) in sources {
            if source.device_id != self.device_id {
                return Err(format!("async D2H device 不一致: transfer={} source={}", self.device_id, source.device_id));
            }
            if offset.checked_add(length).is_none_or(|end| end > source.bytes) {
                return Err(format!("async D2H segment 越界: offset={offset} bytes={length} source={}", source.bytes));
            }
            bytes = bytes.checked_add(length).ok_or("async D2H segments 大小溢出")?;
        }
        if bytes > self.staging.bytes {
            self.staging = PinnedHostBuffer::allocate(bytes)?;
        }
        set_device(self.device_id)?;
        let runtime = RocmRuntime::open()?;
        let copy = runtime.memcpy_async()?;
        let record = runtime.event_record()?;
        let stream = crate::kernel::rocm::hip::active_compute_stream();
        let mut destination_offset = 0usize;
        for &(source, source_offset, length) in sources {
            let stats_started = hip_api_stats::start();
            let status = unsafe { copy(self.staging.pointer.cast::<u8>().add(destination_offset).cast(), source.pointer.cast::<u8>().add(source_offset).cast(), length, HIP_MEMORY_COPY_DEVICE_TO_HOST, stream) };
            hip_api_stats::counted(hip_api_stats::MEMCPY_ASYNC, stats_started);
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipMemcpyAsync segmented D2H"));
            }
            destination_offset += length;
        }
        let stats_started = hip_api_stats::start();
        let status = unsafe { record(self.event, stream) };
        hip_api_stats::counted(hip_api_stats::EVENT_RECORD, stats_started);
        if status != HIP_SUCCESS {
            // copy 已经进入 stream，返回前必须等待，不能释放它仍在写入的 staging。
            if let Ok(synchronize) = runtime.stream_synchronize() {
                let _ = unsafe { synchronize(stream) };
            }
            return Err(runtime.hip_error(status, "hipEventRecord async D2H"));
        }
        self.bytes = bytes;
        self.pending = true;
        record_host_transfer(false, bytes);
        Ok(())
    }

    pub(crate) fn wait(&mut self) -> Result<&[u8], String> {
        if self.pending {
            set_device(self.device_id)?;
            let runtime = RocmRuntime::open()?;
            let synchronize = runtime.event_synchronize()?;
            let stats_started = hip_api_stats::start();
            let status = unsafe { synchronize(self.event) };
            hip_api_stats::counted(hip_api_stats::EVENT_SYNCHRONIZE, stats_started);
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventSynchronize async D2H"));
            }
            self.pending = false;
        }
        Ok(unsafe { std::slice::from_raw_parts(self.staging.pointer.cast(), self.bytes) })
    }
}

impl Drop for AsyncHostDownload {
    fn drop(&mut self) {
        if set_device(self.device_id).is_err() {
            return;
        }
        let Ok(runtime) = RocmRuntime::open() else { return };
        if self.pending
            && let Ok(synchronize) = runtime.event_synchronize()
        {
            let _ = unsafe { synchronize(self.event) };
        }
        if let Ok(destroy) = runtime.event_destroy() {
            let _ = unsafe { destroy(self.event) };
        }
    }
}

/// CPU prefill 输出的单槽 pinned H2D。下一次上传前只等待上一份 H2D event，
/// staging 可以跨层复用，不必把每层 host buffer 保留到 stage completion。
pub(crate) struct AsyncHostUpload {
    device_id: i32,
    event: HipEvent,
    staging: PinnedHostBuffer,
    pending: bool,
}

unsafe impl Send for AsyncHostUpload {}

impl AsyncHostUpload {
    pub(crate) fn new(device_id: i32, bytes: usize) -> Result<Self, String> {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let create = runtime.event_create()?;
        let destroy = runtime.event_destroy()?;
        let mut event = ptr::null_mut();
        let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipEventCreateWithFlags async H2D"));
        }
        match PinnedHostBuffer::allocate(bytes.max(1)) {
            Ok(staging) => Ok(Self { device_id, event, staging, pending: false }),
            Err(error) => {
                let _ = unsafe { destroy(event) };
                Err(error)
            }
        }
    }

    pub(crate) fn upload(&mut self, input: &[u8]) -> Result<DeviceBuffer, String> {
        set_device(self.device_id)?;
        let runtime = RocmRuntime::open()?;
        if self.pending {
            let synchronize = runtime.event_synchronize()?;
            let status = unsafe { synchronize(self.event) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventSynchronize async H2D"));
            }
            self.pending = false;
        }
        if input.len() > self.staging.bytes {
            self.staging = PinnedHostBuffer::allocate(input.len())?;
        }
        unsafe {
            ptr::copy_nonoverlapping(input.as_ptr(), self.staging.pointer.cast(), input.len());
        }
        // attention 输出只活到紧随其后的 o_proj；走 stream-ordered 临时分配，
        // 显式复用池 miss 会调用同步 hipMalloc，把同卡其它 pipeline stream 一起卡住。
        let output = DeviceBuffer::allocate(self.device_id, input.len())?;
        let copy = runtime.memcpy_async()?;
        let record = runtime.event_record()?;
        let stream = crate::kernel::rocm::hip::active_compute_stream();
        let stats_started = hip_api_stats::start();
        let status = unsafe { copy(output.pointer, self.staging.pointer, input.len(), HIP_MEMORY_COPY_HOST_TO_DEVICE, stream) };
        hip_api_stats::counted(hip_api_stats::MEMCPY_ASYNC, stats_started);
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipMemcpyAsync CPU prefill H2D"));
        }
        let stats_started = hip_api_stats::start();
        let status = unsafe { record(self.event, stream) };
        hip_api_stats::counted(hip_api_stats::EVENT_RECORD, stats_started);
        if status != HIP_SUCCESS {
            if let Ok(synchronize) = runtime.stream_synchronize() {
                let _ = unsafe { synchronize(stream) };
            }
            return Err(runtime.hip_error(status, "hipEventRecord CPU prefill H2D"));
        }
        self.pending = true;
        record_host_transfer(true, input.len());
        Ok(output)
    }
}

impl Drop for AsyncHostUpload {
    fn drop(&mut self) {
        if set_device(self.device_id).is_err() {
            return;
        }
        let Ok(runtime) = RocmRuntime::open() else { return };
        if self.pending
            && let Ok(synchronize) = runtime.event_synchronize()
        {
            let _ = unsafe { synchronize(self.event) };
        }
        if let Ok(destroy) = runtime.event_destroy() {
            let _ = unsafe { destroy(self.event) };
        }
    }
}

/// 双机边界的小块 pinned host cache。槽与 device buffer 一起被 stage
/// completion 持有；Drop 只归还本进程 cache，不调用 hipHostFree。
struct CachedPinnedHostBuffer {
    pointer: *mut c_void,
    bytes: usize,
}

unsafe impl Send for CachedPinnedHostBuffer {}
unsafe impl Sync for CachedPinnedHostBuffer {}

static CACHED_PINNED_HOST_BUFFERS: OnceLock<Mutex<HashMap<usize, Vec<usize>>>> = OnceLock::new();

impl CachedPinnedHostBuffer {
    fn allocate(bytes: usize) -> Result<Self, String> {
        let pool = CACHED_PINNED_HOST_BUFFERS.get_or_init(|| Mutex::new(HashMap::new()));
        if let Ok(mut pool) = pool.lock()
            && let Some(capacity) = pool.keys().copied().filter(|&capacity| capacity >= bytes && pool.get(&capacity).is_some_and(|buffers| !buffers.is_empty())).min()
        {
            let pointer = pool.get_mut(&capacity).and_then(Vec::pop).expect("pinned host cache 已检查非空");
            return Ok(Self { pointer: pointer as *mut c_void, bytes: capacity });
        }
        let pinned = std::mem::ManuallyDrop::new(PinnedHostBuffer::allocate(bytes.max(1))?);
        Ok(Self { pointer: pinned.pointer, bytes: pinned.bytes })
    }
}

impl Drop for CachedPinnedHostBuffer {
    fn drop(&mut self) {
        if self.pointer.is_null() {
            return;
        }
        if let Ok(mut pool) = CACHED_PINNED_HOST_BUFFERS.get_or_init(|| Mutex::new(HashMap::new())).lock() {
            pool.entry(self.bytes).or_default().push(self.pointer as usize);
            self.pointer = ptr::null_mut();
        }
    }
}

impl PinnedHostBuffer {
    pub(super) fn allocate(bytes: usize) -> Result<Self, String> {
        let runtime = RocmRuntime::open()?;
        let malloc: Symbol<HipHostMalloc> = runtime.symbol(&runtime.hip, b"hipHostMalloc\0")?;
        let mut pointer = ptr::null_mut();
        let status = unsafe { malloc(&mut pointer, bytes, HIP_HOST_MALLOC_DEFAULT) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipHostMalloc staging"));
        }
        Ok(Self { pointer, bytes })
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        let Ok(runtime) = RocmRuntime::open() else { return };
        let Ok(free) = runtime.symbol::<HipHostFree>(&runtime.hip, b"hipHostFree\0") else { return };
        unsafe { free(self.pointer) };
    }
}

pub(super) struct CompletedHostTransfer {
    device_id: i32,
    stream: *mut c_void,
    staging: PinnedHostBuffer,
}

impl CompletedHostTransfer {
    pub(super) fn new(device_id: i32, bytes: usize) -> Result<Self, String> {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let create: Symbol<HipStreamCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipStreamCreateWithFlags\0")?;
        let destroy: Symbol<HipStreamDestroy> = runtime.symbol(&runtime.hip, b"hipStreamDestroy\0")?;
        let mut stream = ptr::null_mut();
        let status = unsafe { create(&mut stream, HIP_STREAM_NON_BLOCKING) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipStreamCreateWithFlags completed D2H"));
        }
        match PinnedHostBuffer::allocate(bytes.max(1)) {
            Ok(staging) => Ok(Self { device_id, stream, staging }),
            Err(error) => {
                let _ = unsafe { destroy(stream) };
                Err(error)
            }
        }
    }

    pub(super) fn copy(&mut self, source: *const c_void, output: &mut [u8]) -> Result<(), String> {
        set_device(self.device_id)?;
        if output.len() > self.staging.bytes {
            self.staging = PinnedHostBuffer::allocate(output.len())?;
        }
        let runtime = RocmRuntime::open()?;
        let copy = runtime.memcpy_async()?;
        let synchronize = runtime.stream_synchronize()?;
        let status = unsafe { copy(self.staging.pointer, source, output.len(), HIP_MEMORY_COPY_DEVICE_TO_HOST, self.stream) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipMemcpyAsync completed D2H"));
        }
        let status = unsafe { synchronize(self.stream) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipStreamSynchronize completed D2H"));
        }
        unsafe {
            ptr::copy_nonoverlapping(self.staging.pointer.cast::<u8>(), output.as_mut_ptr(), output.len());
        }
        Ok(())
    }
}

impl Drop for CompletedHostTransfer {
    fn drop(&mut self) {
        if self.stream.is_null() || set_device(self.device_id).is_err() {
            return;
        }
        let Ok(runtime) = RocmRuntime::open() else { return };
        let Ok(destroy) = runtime.symbol::<HipStreamDestroy>(&runtime.hip, b"hipStreamDestroy\0") else { return };
        let _ = unsafe { destroy(self.stream) };
    }
}

/// 网络/调度线程的小块 H2D 不得同步 compute stream，否则新到工作
/// 只能在当前 stage 用完 CU 后才进队，自然 backlog 会被意外串行化。
pub(super) struct IndependentHostUpload {
    device_id: i32,
    stream: *mut c_void,
    staging: PinnedHostBuffer,
}

impl IndependentHostUpload {
    pub(super) fn new(device_id: i32, bytes: usize) -> Result<Self, String> {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let create: Symbol<HipStreamCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipStreamCreateWithFlags\0")?;
        let destroy: Symbol<HipStreamDestroy> = runtime.symbol(&runtime.hip, b"hipStreamDestroy\0")?;
        let mut stream = ptr::null_mut();
        let status = unsafe { create(&mut stream, HIP_STREAM_NON_BLOCKING) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipStreamCreateWithFlags independent H2D"));
        }
        match PinnedHostBuffer::allocate(bytes.max(1)) {
            Ok(staging) => Ok(Self { device_id, stream, staging }),
            Err(error) => {
                let _ = unsafe { destroy(stream) };
                Err(error)
            }
        }
    }

    pub(super) fn copy(&mut self, input: &[u8], destination: *mut c_void) -> Result<(), String> {
        set_device(self.device_id)?;
        if input.len() > self.staging.bytes {
            self.staging = PinnedHostBuffer::allocate(input.len())?;
        }
        unsafe {
            ptr::copy_nonoverlapping(input.as_ptr(), self.staging.pointer.cast(), input.len());
        }
        let runtime = RocmRuntime::open()?;
        let copy = runtime.memcpy_async()?;
        let synchronize = runtime.stream_synchronize()?;
        let stats_started = hip_api_stats::start();
        let status = unsafe { copy(destination, self.staging.pointer, input.len(), HIP_MEMORY_COPY_HOST_TO_DEVICE, self.stream) };
        hip_api_stats::counted(hip_api_stats::MEMCPY_ASYNC, stats_started);
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipMemcpyAsync independent H2D"));
        }
        let status = unsafe { synchronize(self.stream) };
        if status == HIP_SUCCESS { Ok(()) } else { Err(runtime.hip_error(status, "hipStreamSynchronize independent H2D")) }
    }
}

impl Drop for IndependentHostUpload {
    fn drop(&mut self) {
        if self.stream.is_null() || set_device(self.device_id).is_err() {
            return;
        }
        let Ok(runtime) = RocmRuntime::open() else { return };
        let Ok(destroy) = runtime.symbol::<HipStreamDestroy>(&runtime.hip, b"hipStreamDestroy\0") else { return };
        let _ = unsafe { destroy(self.stream) };
    }
}

thread_local! {
    static H2D_STAGING: std::cell::RefCell<Option<PinnedHostBuffer>> = const { std::cell::RefCell::new(None) };
    static INDEPENDENT_H2D: RefCell<HashMap<i32, IndependentHostUpload>> = RefCell::new(HashMap::new());
    static COMPLETED_D2H: RefCell<HashMap<i32, CompletedHostTransfer>> = RefCell::new(HashMap::new());
    pub(super) static CURRENT_HIP_DEVICE: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
}

pub(crate) struct CtGroupedWeightRef<'a> {
    pub packed: &'a DeviceBuffer,
    pub scales: &'a DeviceBuffer,
    pub scale_dtype: u32,
    pub group_size: usize,
    /// 0=compressed-tensors W4/W8，1=official E4M3 + F32 block scale。
    pub format: u32,
}

/// GGUF grouped decode experts 的 meta(与 kernel 侧 GgufKqMeta 布局一致)。
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgufGroupedExpertMeta {
    pub gate: u64,
    pub up: u64,
    pub down: u64,
    pub gate_type: u32,
    pub up_type: u32,
    pub down_type: u32,
}

pub(crate) struct CtGroupedExpertRef<'a> {
    pub gate: CtGroupedWeightRef<'a>,
    pub up: CtGroupedWeightRef<'a>,
    pub down: CtGroupedWeightRef<'a>,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DeviceBuffer").field("device_id", &self.device_id).field("bytes", &self.bytes).finish_non_exhaustive()
    }
}

impl DeviceBuffer {
    #[track_caller]
    pub fn upload(device_id: i32, bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() {
            return Err("ROCm device buffer 大小不能为 0".to_owned());
        }
        // 常驻权重优先独立分配；阶段权重在同步池不足时复用异步临时池。
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        if !crate::kernel::rocm::hip::active_compute_stream().is_null() {
            let buffer = Self::allocate(device_id, bytes.len())?;
            buffer.copy_from_host(bytes)?;
            return Ok(buffer);
        }
        let malloc = runtime.malloc()?;
        let mut pointer = ptr::null_mut();
        let status = unsafe { malloc(&mut pointer, bytes.len()) };
        let buffer = if status == HIP_SUCCESS {
            Self {
                device_id,
                pointer,
                bytes: bytes.len(),
                capacity_bytes: bytes.len(),
                recyclable: false,
                retain_until_stage_completion: false,
                stage_completion_ready: std::sync::atomic::AtomicBool::new(false),
                deferred_host: None,
                deferred_upload_enqueued: std::sync::atomic::AtomicBool::new(false),
                async_allocated: false,
                owner: None,
            }
        } else {
            Self::allocate(device_id, bytes.len()).map_err(|fallback| format!("{}；异步池回退失败: {fallback}", runtime.hip_error(status, "hipMalloc uploaded device buffer")))?
        };
        buffer.copy_from_host(bytes)?;
        Ok(buffer)
    }

    /// 返回时 H2D 已完成，但不会等待该卡的 compute stream。用于网络
    /// 边界和调度线程；常驻权重仍走 `upload`。
    #[track_caller]
    pub fn upload_independent(device_id: i32, bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() {
            return Err("ROCm device buffer 大小不能为 0".to_owned());
        }
        let buffer = Self::allocate_peer(device_id, bytes.len())?;
        INDEPENDENT_H2D.with(|uploads| {
            let mut uploads = uploads.borrow_mut();
            if !uploads.contains_key(&device_id) {
                uploads.insert(device_id, IndependentHostUpload::new(device_id, bytes.len())?);
            }
            uploads.get_mut(&device_id).expect("已插入 independent H2D").copy(bytes, buffer.pointer)
        })?;
        Ok(buffer)
    }

    /// 双机边界的稳态上传：网络线程只取得 device/pinned host cache 槽，
    /// H2D 延迟到 stage0 的真实 consumer stream 上提交。
    #[track_caller]
    pub fn upload_ordered(device_id: i32, bytes: &[u8]) -> Result<Self, String> {
        Self::upload_ordered_with(device_id, bytes.len(), |staging| staging.copy_from_slice(bytes))
    }

    /// 直接在 pinned staging 中构造有序上传内容，避免调用方先写 pageable Vec、
    /// 再完整复制一次。闭包返回后 staging 由 stage completion 保活。
    #[track_caller]
    pub(crate) fn upload_ordered_with(device_id: i32, bytes: usize, fill: impl FnOnce(&mut [u8])) -> Result<Self, String> {
        if bytes == 0 {
            return Err("ROCm device buffer 大小不能为 0".to_owned());
        }
        let mut buffer = Self::allocate_peer(device_id, bytes)?;
        buffer.retain_until_stage_completion = true;
        let staging = CachedPinnedHostBuffer::allocate(bytes)?;
        let staging_bytes = unsafe { std::slice::from_raw_parts_mut(staging.pointer.cast(), bytes) };
        fill(staging_bytes);
        buffer.deferred_host = Some(staging);
        Ok(buffer)
    }

    /// 上传 f32 切片。
    ///
    /// 调用方直接对字面量数组 `from_raw_parts(as_ptr().cast(), n)` 构造字节切片时,
    /// 若数组从未被安全代码读取,LLVM 可合法地复用其栈槽(实测 dev 构建下 FFI 读到
    /// 被覆盖的栈内容);经过本函数的 `&[f32]` 边界后借用真实逃逸,栈内容必然物化。
    #[track_caller]
    pub fn upload_f32(device_id: i32, values: &[f32]) -> Result<Self, String> {
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) };
        Self::upload(device_id, bytes)
    }

    #[track_caller]
    pub fn allocate(device_id: i32, bytes: usize) -> Result<Self, String> {
        Self::allocate_inner(device_id, bytes, false, false, usize::MAX, std::panic::Location::caller())
    }

    /// 长期 cache 只接收精确尺寸的池块，避免小会话持有上一条长会话的大 allocation。
    #[track_caller]
    pub(crate) fn allocate_cache(device_id: i32, bytes: usize) -> Result<Self, String> {
        Self::allocate_inner(device_id, bytes, false, false, bytes, std::panic::Location::caller())
    }

    #[track_caller]
    pub fn allocate_reusable(device_id: i32, bytes: usize) -> Result<Self, String> {
        Self::allocate_inner(device_id, bytes, true, false, usize::MAX, std::panic::Location::caller())
    }

    #[track_caller]
    pub(super) fn allocate_peer(device_id: i32, bytes: usize) -> Result<Self, String> {
        Self::allocate_inner(device_id, bytes, true, true, usize::MAX, std::panic::Location::caller())
    }

    pub(super) fn allocate_inner(device_id: i32, bytes: usize, reusable: bool, force_explicit_pool: bool, max_reuse_capacity: usize, caller: &'static std::panic::Location<'static>) -> Result<Self, String> {
        if bytes == 0 {
            return Err("ROCm device buffer 大小不能为 0".to_owned());
        }
        let request_started = std::time::Instant::now();
        let request_start_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
        let log_slow_request = |path: &str| {
            let elapsed = request_started.elapsed();
            if elapsed >= std::time::Duration::from_millis(5) {
                let complete_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
                eprintln!(
                    "[rocm-buffer-allocate-slow] ts_us={request_start_us} device={device_id} bytes={bytes} path={path} wall_ms={:.3} caller={}:{} complete_us={complete_us}",
                    elapsed.as_secs_f64() * 1000.0,
                    caller.file().rsplit('/').next().unwrap_or(caller.file()),
                    caller.line(),
                );
            }
        };
        set_device(device_id)?;
        let explicit_pool = force_explicit_pool || explicit_device_pool_enabled(reusable);
        if explicit_pool {
            if let Some((pointer, capacity_bytes)) = take_device_buffer_bounded(device_id, bytes, max_reuse_capacity) {
                log_slow_request("explicit-pool-hit");
                return Ok(Self {
                    device_id,
                    pointer,
                    bytes,
                    capacity_bytes,
                    recyclable: explicit_pool,
                    retain_until_stage_completion: false,
                    stage_completion_ready: std::sync::atomic::AtomicBool::new(false),
                    deferred_host: None,
                    deferred_upload_enqueued: std::sync::atomic::AtomicBool::new(false),
                    async_allocated: false,
                    owner: None,
                });
            }
        }
        let runtime = RocmRuntime::open()?;
        // 超大 activation 在 ROCm 默认异步池中容易形成无法及时 trim 的碎片；层边界用同步释放保证复用空间。
        if !explicit_pool && bytes <= DEVICE_ASYNC_MAX_BUFFER_BYTES {
            if let Ok(malloc_async) = runtime.malloc_async() {
                let mut pointer = ptr::null_mut();
                let stats_started = hip_api_stats::start();
                let status = unsafe { malloc_async(&mut pointer, bytes, crate::kernel::rocm::hip::active_compute_stream()) };
                hip_api_stats::counted(hip_api_stats::MALLOC_ASYNC, stats_started);
                if status == HIP_SUCCESS {
                    log_slow_request("hipMallocAsync");
                    return Ok(Self {
                        device_id,
                        pointer,
                        bytes,
                        capacity_bytes: bytes,
                        recyclable: false,
                        retain_until_stage_completion: false,
                        stage_completion_ready: std::sync::atomic::AtomicBool::new(false),
                        deferred_host: None,
                        deferred_upload_enqueued: std::sync::atomic::AtomicBool::new(false),
                        async_allocated: true,
                        owner: None,
                    });
                }
            }
        }
        let malloc = runtime.malloc()?;
        let mut pointer = ptr::null_mut();
        let stats_started = hip_api_stats::start();
        let miss_started = (explicit_pool && options().kernel_profile).then(std::time::Instant::now);
        let allocation_started = std::time::Instant::now();
        let allocation_start_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
        let mut status = unsafe { malloc(&mut pointer, bytes) };
        let allocation_micros = allocation_started.elapsed().as_micros();
        if allocation_micros >= 5_000 {
            let complete_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
            eprintln!(
                "[rocm-slow-allocation] ts_us={allocation_start_us} device={device_id} bytes={bytes} wall_ms={:.3} caller={}:{} complete_us={complete_us}",
                allocation_micros as f64 / 1000.0,
                caller.file().rsplit('/').next().unwrap_or(caller.file()),
                caller.line(),
            );
        }
        record_pool_miss(device_id, bytes, caller, miss_started);
        hip_api_stats::counted(hip_api_stats::MALLOC_SYNC, stats_started);
        if status != HIP_SUCCESS && options().memory_pool {
            // 精确 shape 池无法满足放大的 prefill microbatch 时，先回收已完成的
            // 旧 shape，再清理默认异步池后重试。只有这条无同步慢路径仍然 OOM
            // 时才等待设备，把 pending 和当前 stage 已完成的 buffer 一并释放。
            promote_device_buffers(device_id);
            let mut released = release_available_device_buffers(device_id);
            let _ = trim_device_memory_pool(device_id);
            pointer = ptr::null_mut();
            status = unsafe { malloc(&mut pointer, bytes) };
            if status != HIP_SUCCESS {
                synchronize_device(device_id, "hipDeviceSynchronize OOM pending buffer")?;
                recycle_completed_stage_buffers(device_id, take_stage_buffer_recycles(device_id));
                // 同步后逐 buffer event 也已完成；先晋升 pending，否则下面只能
                // 释放 available，4 GiB 软池会被误当成不可回收显存。
                promote_device_buffers(device_id);
                released = released.saturating_add(release_available_device_buffers(device_id));
                let _ = trim_device_memory_pool(device_id);
                pointer = ptr::null_mut();
                status = unsafe { malloc(&mut pointer, bytes) };
            }
            if status != HIP_SUCCESS {
                return Err(format!("{} (申请 {} bytes，OOM 同步回收 {} bytes 后重试失败)", runtime.hip_error(status, "hipMalloc device buffer"), bytes, released));
            }
        } else if status != HIP_SUCCESS {
            return Err(format!("{} (申请 {} bytes)", runtime.hip_error(status, "hipMalloc device buffer"), bytes));
        }
        log_slow_request("hipMalloc");
        Ok(Self {
            device_id,
            pointer,
            bytes,
            capacity_bytes: bytes,
            recyclable: explicit_pool,
            retain_until_stage_completion: false,
            stage_completion_ready: std::sync::atomic::AtomicBool::new(false),
            deferred_host: None,
            deferred_upload_enqueued: std::sync::atomic::AtomicBool::new(false),
            async_allocated: false,
            owner: None,
        })
    }

    pub(crate) fn copy_from_host(&self, input: &[u8]) -> Result<(), String> {
        if input.len() > self.bytes {
            return Err(format!("ROCm H2D 字节数 {}，buffer 容量 {}", input.len(), self.bytes));
        }
        set_device(self.device_id)?;
        let runtime = RocmRuntime::open()?;
        let stream = crate::kernel::rocm::hip::active_compute_stream();
        if !stream.is_null() {
            let copy = runtime.memcpy_async()?;
            let synchronize = runtime.stream_synchronize()?;
            let status = unsafe { copy(self.pointer, input.as_ptr().cast(), input.len(), HIP_MEMORY_COPY_HOST_TO_DEVICE, stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipMemcpyAsync input H2D"));
            }
            let status = unsafe { synchronize(stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipStreamSynchronize input H2D"));
            }
            record_host_transfer(true, input.len());
            return Ok(());
        }
        let copy = runtime.memcpy()?;
        let mode = H2D_UPLOAD_MODE.load(std::sync::atomic::Ordering::Relaxed);
        if mode != H2D_MODE_PINNED {
            let stats_started = hip_api_stats::start();
            let status = unsafe { copy(self.pointer, input.as_ptr().cast(), input.len(), HIP_MEMORY_COPY_HOST_TO_DEVICE) };
            hip_api_stats::counted(hip_api_stats::MEMCPY_SYNC, stats_started);
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipMemcpy input H2D"));
            }
            // 部分 ROCm 组合的 pageable 小块上传会退化 20 倍；只用首个真实大块选择路径。
            if mode == H2D_MODE_AUTO && input.len() >= H2D_PROBE_BYTES {
                let started = std::time::Instant::now();
                let status = unsafe { copy(self.pointer, input.as_ptr().cast(), input.len(), HIP_MEMORY_COPY_HOST_TO_DEVICE) };
                if status != HIP_SUCCESS {
                    return Err(runtime.hip_error(status, "hipMemcpy H2D probe"));
                }
                let seconds = started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
                let gib_per_second = input.len() as f64 / (1_u64 << 30) as f64 / seconds;
                H2D_UPLOAD_MODE.store(if gib_per_second >= H2D_DIRECT_MIN_GIB_PER_SECOND { H2D_MODE_DIRECT } else { H2D_MODE_PINNED }, std::sync::atomic::Ordering::Relaxed);
            }
            record_host_transfer(true, input.len());
            return Ok(());
        }
        H2D_STAGING.with(|staging| {
            let mut staging = staging.borrow_mut();
            if staging.is_none() {
                *staging = Some(PinnedHostBuffer::allocate(H2D_STAGING_BYTES)?);
            }
            let staging = staging.as_ref().expect("H2D staging 已初始化");
            for (offset, chunk) in input.chunks(H2D_STAGING_BYTES).enumerate() {
                let byte_offset = offset * H2D_STAGING_BYTES;
                unsafe {
                    ptr::copy_nonoverlapping(chunk.as_ptr(), staging.pointer.cast(), chunk.len());
                }
                let status = unsafe { copy(self.pointer.cast::<u8>().add(byte_offset).cast(), staging.pointer, chunk.len(), HIP_MEMORY_COPY_HOST_TO_DEVICE) };
                if status != HIP_SUCCESS {
                    return Err(runtime.hip_error(status, "hipMemcpy pinned input H2D"));
                }
            }
            Ok(())
        })?;
        record_host_transfer(true, input.len());
        Ok(())
    }

    pub(crate) fn copy_to_host(&self, output: &mut [u8]) -> Result<(), String> {
        if output.len() > self.bytes {
            return Err(format!("ROCm D2H 字节数 {}，buffer 容量 {}", output.len(), self.bytes));
        }
        set_device(self.device_id)?;
        let runtime = RocmRuntime::open()?;
        let stream = crate::kernel::rocm::hip::active_compute_stream();
        if !stream.is_null() {
            let copy = runtime.memcpy_async()?;
            let synchronize = runtime.stream_synchronize()?;
            let status = unsafe { copy(output.as_mut_ptr().cast(), self.pointer, output.len(), HIP_MEMORY_COPY_DEVICE_TO_HOST, stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipMemcpyAsync output D2H"));
            }
            let status = unsafe { synchronize(stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipStreamSynchronize output D2H"));
            }
            record_host_transfer(false, output.len());
            return Ok(());
        }
        let copy = runtime.memcpy()?;
        let stats_started = hip_api_stats::start();
        let status = unsafe { copy(output.as_mut_ptr().cast(), self.pointer, output.len(), HIP_MEMORY_COPY_DEVICE_TO_HOST) };
        hip_api_stats::counted(hip_api_stats::MEMCPY_SYNC, stats_started);
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipMemcpy output D2H"));
        }
        record_host_transfer(false, output.len());
        Ok(())
    }

    /// 调用方已经观察到来源 completion 后，用独立 non-blocking stream 下载。
    /// 这样不会排到默认 stream 中随后提交的 decode 批次之后。
    pub(crate) fn copy_completed_to_host(&self, output: &mut [u8]) -> Result<(), String> {
        if output.len() > self.bytes {
            return Err(format!("ROCm completed D2H 字节数 {}，buffer 容量 {}", output.len(), self.bytes));
        }
        COMPLETED_D2H.with(|transfers| {
            let mut transfers = transfers.borrow_mut();
            if !transfers.contains_key(&self.device_id) {
                transfers.insert(self.device_id, CompletedHostTransfer::new(self.device_id, output.len())?);
            }
            transfers.get_mut(&self.device_id).expect("刚插入 completed D2H stream").copy(self.pointer, output)
        })?;
        record_host_transfer(false, output.len());
        Ok(())
    }

    /// 同一 device 默认 stream 内有序复制 resident 数据；cache 扩容和 ring 写入不经过 host。
    ///
    /// 这里不能用同步 hipMemcpy：V4 decode 每层会写 recent Q8 K/V、scale 与
    /// index key，host 等待这些小复制会把整条提交链切成五段。来源 activation
    /// 的释放同样排在默认 stream，显式池来源则由 stage completion event 持有，
    /// 因而异步返回不会缩短来源 allocation 的有效期。
    pub(crate) fn copy_from_device(&self, destination_offset: usize, source: &Self, source_offset: usize, bytes: usize) -> Result<(), String> {
        super::graph::graph_foreign_stream_op();
        if self.device_id != source.device_id {
            return Err(format!("ROCm D2D device 不一致: destination={} source={}", self.device_id, source.device_id));
        }
        if destination_offset.checked_add(bytes).is_none_or(|end| end > self.bytes) || source_offset.checked_add(bytes).is_none_or(|end| end > source.bytes) {
            return Err(format!("ROCm D2D 越界: destination={destination_offset}+{bytes}/{} source={source_offset}+{bytes}/{}", self.bytes, source.bytes,));
        }
        if bytes == 0 {
            return Ok(());
        }
        set_device(self.device_id)?;
        let runtime = RocmRuntime::open()?;
        let copy = runtime.memcpy_async()?;
        let stats_started = hip_api_stats::start();
        let status =
            unsafe { copy(self.pointer.cast::<u8>().add(destination_offset).cast(), source.pointer.cast::<u8>().add(source_offset).cast(), bytes, HIP_MEMORY_COPY_DEVICE_TO_DEVICE, crate::kernel::rocm::hip::active_compute_stream()) };
        hip_api_stats::counted(hip_api_stats::MEMCPY_ASYNC, stats_started);
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipMemcpyAsync resident D2D"));
        }
        Ok(())
    }

    /// 把 per-thread async allocation 稳定到当前线程的显式设备池，供跨线程/P2P 使用。
    pub(crate) fn copy_to_stable(&self) -> Result<Self, String> {
        synchronize_device(self.device_id, "stable copy source synchronize")?;
        let output = Self::allocate_peer(self.device_id, self.bytes)?;
        output.copy_from_device(0, self, 0, self.bytes)?;
        synchronize_device(self.device_id, "stable copy destination synchronize")?;
        Ok(output)
    }

    /// 源 buffer 已完成且不是 async allocation 时，提交 P2P 复制。
    /// 调用方必须在独立 compute stream 消费前完成目标 rank barrier；需要
    /// overlap 的调用方应走显式 event 串依赖的 ordered 路径。
    pub(crate) fn copy_stable_to_device_ready(&self, device_id: i32) -> Result<Self, String> {
        if self.async_allocated {
            return Err("ready P2P source 不能是 async allocation".to_owned());
        }
        let output = Self::allocate_peer(device_id, self.bytes)?;
        let runtime = RocmRuntime::open()?;
        enable_peer_access(device_id, self.device_id)?;
        let copy: Symbol<HipMemcpyPeer> = runtime.symbol(&runtime.hip, b"hipMemcpyPeer\0")?;
        let status = unsafe { copy(output.pointer, device_id, self.pointer, self.device_id, self.bytes) };
        if status != HIP_SUCCESS {
            return Err(format!("{}: source_device={} destination_device={device_id} bytes={}", runtime.hip_error(status, "hipMemcpyPeer ready activation"), self.device_id, self.bytes));
        }
        Ok(output)
    }

    /// 不等待 host 轮询源 completion，直接用跨设备 event 串起源 submission
    /// stream、P2P copy 与目标 submission stream。目标 completion 持有源
    /// buffer 到复制完成。
    pub(crate) fn copy_stable_to_device_ordered_async(self: &std::sync::Arc<Self>, device_id: i32) -> Result<Self, String> {
        self.copy_stable_to_device_ordered_async_retained_by(device_id, device_id)
    }

    /// 与 ordered P2P 相同，但允许把源 buffer 挂到另一张卡的 stage completion。
    /// 双卡 MoE 的 owner completion 已经依赖 peer 结果回传，因此它也覆盖
    /// owner→peer activation 的读取生命周期，无需为 peer 单独同步或建 completion。
    pub(crate) fn copy_stable_to_device_ordered_async_retained_by(self: &std::sync::Arc<Self>, device_id: i32, completion_device_id: i32) -> Result<Self, String> {
        self.copy_stable_to_device_ordered_async_retained_by_inner(device_id, completion_device_id, false, None)
    }

    /// 只建立跨设备 producer→consumer event 依赖，不复制数据。consumer
    /// kernel 通过 peer BAR 只读一次来源 buffer 时，这比先复制再读取少一次
    /// 大块显存写入；来源仍由 consumer completion 保活。
    pub(crate) fn wait_stable_on_device_ordered_on_streams_retained_by(self: &std::sync::Arc<Self>, device_id: i32, completion_device_id: i32, source_stream: usize, destination_stream: usize) -> Result<(), String> {
        if self.async_allocated {
            return Err("ordered peer wait source 不能是 async allocation".to_owned());
        }
        let runtime = RocmRuntime::open()?;
        enable_peer_access(device_id, self.device_id)?;
        let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0")?;
        let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0")?;
        let wait: Symbol<HipStreamWaitEvent> = runtime.symbol(&runtime.hip, b"hipStreamWaitEvent\0")?;
        let destroy: Symbol<HipEventDestroy> = runtime.symbol(&runtime.hip, b"hipEventDestroy\0")?;
        let mut event = device_buffer_pool(self.device_id).and_then(|pool| pool.lock().ok()?.available_events.pop()).map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
        set_device(self.device_id)?;
        if event.is_null() {
            let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventCreateWithFlags ordered peer wait"));
            }
        }
        let result = (|| {
            let status = unsafe { record(event, source_stream as *mut c_void) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventRecord ordered peer wait source"));
            }
            set_device(device_id)?;
            let status = unsafe { wait(destination_stream as *mut c_void, event, 0) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipStreamWaitEvent ordered peer wait destination"));
            }
            PENDING_P2P_SOURCES.with(|sources| {
                sources.borrow_mut().entry(completion_device_id).or_default().push(PendingP2pSource { sources: vec![self.clone()], event: event as usize });
            });
            Ok(())
        })();
        if result.is_err() {
            let _ = set_device(self.device_id).and_then(|()| {
                let status = unsafe { destroy(event) };
                if status == HIP_SUCCESS { Ok(()) } else { Err(runtime.hip_error(status, "hipEventDestroy ordered peer wait")) }
            });
        }
        result?;
        set_device(device_id)
    }

    /// 只等待源 producer event；P2P copy 与 consumer 仍在目标 stream 异步执行。
    pub(crate) fn copy_stable_to_device_after_event_retained_by(self: &std::sync::Arc<Self>, device_id: i32, completion_device_id: i32) -> Result<Self, String> {
        self.copy_stable_to_device_ordered_async_retained_by_inner(device_id, completion_device_id, true, None)
    }

    /// cooperative peer 没有 pipeline stage，因此它的 producer 可以合法位于
    /// null stream。显式传入两端 stream，避免把“peer default”误判成“缺少
    /// 与 owner 对应的 stage-prefill stream”。
    pub(crate) fn copy_stable_to_device_after_event_on_streams_retained_by(self: &std::sync::Arc<Self>, device_id: i32, completion_device_id: i32, source_stream: usize, destination_stream: usize) -> Result<Self, String> {
        self.copy_stable_to_device_ordered_async_retained_by_inner(device_id, completion_device_id, true, Some((source_stream, destination_stream)))
    }

    fn copy_stable_to_device_ordered_async_retained_by_inner(self: &std::sync::Arc<Self>, device_id: i32, completion_device_id: i32, synchronize_source_event: bool, explicit_streams: Option<(usize, usize)>) -> Result<Self, String> {
        if self.async_allocated {
            return Err("ordered P2P source 不能是 async allocation".to_owned());
        }
        let output = Self::allocate_peer(device_id, self.bytes)?;
        let runtime = RocmRuntime::open()?;
        enable_peer_access(device_id, self.device_id)?;
        let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0")?;
        let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0")?;
        let wait: Symbol<HipStreamWaitEvent> = runtime.symbol(&runtime.hip, b"hipStreamWaitEvent\0")?;
        let synchronize = runtime.event_synchronize()?;
        let copy: Symbol<HipMemcpyPeerAsync> = runtime.symbol(&runtime.hip, b"hipMemcpyPeerAsync\0")?;
        let copy_ready: Symbol<HipMemcpyPeer> = runtime.symbol(&runtime.hip, b"hipMemcpyPeer\0")?;
        let destroy: Symbol<HipEventDestroy> = runtime.symbol(&runtime.hip, b"hipEventDestroy\0")?;
        let mut event = device_buffer_pool(self.device_id).and_then(|pool| pool.lock().ok()?.available_events.pop()).map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
        let destination_stream = explicit_streams.map_or_else(|| compute_stream_for(device_id), |(_, stream)| stream as *mut c_void);
        // background prefill 的 stage 各在线程中提交，目标线程的 TLS 没有源卡
        // stream 映射。若目标正使用 stage-prefill stream，显式取同类源 stream；
        // 否则沿用 latency/default 或调用方已经绑定的 source stream。
        let mapped_source_stream = compute_stream_for(self.device_id);
        let source_stream = if let Some((stream, _)) = explicit_streams {
            stream as *mut c_void
        } else if !mapped_source_stream.is_null() {
            mapped_source_stream
        } else if !destination_stream.is_null() && initialized_background_stage_stream(device_id) == Some(destination_stream as usize) {
            initialized_background_stage_stream(self.device_id).ok_or_else(|| format!("ROCm ordered P2P 缺少 source stage-prefill stream: source={} destination={device_id}", self.device_id))? as *mut c_void
        } else {
            ptr::null_mut()
        };
        set_device(self.device_id)?;
        if event.is_null() {
            let stats_started = hip_api_stats::start();
            let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
            hip_api_stats::counted(hip_api_stats::EVENT_CREATE, stats_started);
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventCreateWithFlags ordered P2P"));
            }
        }
        let result = (|| {
            let stats_started = hip_api_stats::start();
            let status = unsafe { record(event, source_stream) };
            hip_api_stats::counted(hip_api_stats::EVENT_RECORD, stats_started);
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventRecord ordered P2P source"));
            }
            if synchronize_source_event {
                let status = unsafe { synchronize(event) };
                if status != HIP_SUCCESS {
                    return Err(runtime.hip_error(status, "hipEventSynchronize ordered P2P source"));
                }
                set_device(device_id)?;
            } else {
                set_device(device_id)?;
                let stats_started = hip_api_stats::start();
                let status = unsafe { wait(destination_stream, event, 0) };
                hip_api_stats::counted(hip_api_stats::STREAM_WAIT_EVENT, stats_started);
                if status != HIP_SUCCESS {
                    return Err(runtime.hip_error(status, "hipStreamWaitEvent ordered P2P destination"));
                }
            }
            // 目标 stream 上这个 marker 排在跨卡 wait 之后、copy 之前，因此可把
            // 上游依赖等待与真正的 PCIe 传输分开。
            super::device_profile_operator(device_id, "handoff_copy")?;
            // W7900D 的 hipMemcpyPeerAsync copy-engine 路径会在连续多层提交时退化；
            // peer access 已启用，改用目标卡上的 float4 拷贝 kernel 经 PCIe BAR
            // 直读源卡显存。Amd-4 完整 19-step 为 154.316s，对照全 DMA
            // 201.295s、仅大块 DMA 181.090s。
            if synchronize_source_event {
                let status = unsafe { copy_ready(output.pointer, device_id, self.pointer, self.device_id, self.bytes) };
                if status != HIP_SUCCESS {
                    return Err(format!("{}: source_device={} destination_device={device_id} bytes={}", runtime.hip_error(status, "hipMemcpyPeer event-ready activation"), self.device_id, self.bytes));
                }
            } else if self.bytes % 16 == 0 && crate::kernel::rocm::hip::active_compute_stream() == destination_stream {
                super::peer_copy::try_peer_copy_kernel_ordered(device_id, output.pointer, self.pointer, self.bytes)?;
            } else {
                let stats_started = hip_api_stats::start();
                let status = unsafe { copy(output.pointer, device_id, self.pointer, self.device_id, self.bytes, destination_stream) };
                hip_api_stats::counted(hip_api_stats::MEMCPY_PEER_ASYNC, stats_started);
                if status != HIP_SUCCESS {
                    return Err(format!("{}: source_device={} destination_device={device_id} bytes={}", runtime.hip_error(status, "hipMemcpyPeerAsync ordered activation"), self.device_id, self.bytes));
                }
            }
            PENDING_P2P_SOURCES.with(|sources| {
                sources.borrow_mut().entry(completion_device_id).or_default().push(PendingP2pSource { sources: vec![self.clone()], event: event as usize });
            });
            Ok(output)
        })();
        if result.is_err() {
            let _ = set_device(self.device_id).and_then(|()| {
                let stats_started = hip_api_stats::start();
                let status = unsafe { destroy(event) };
                hip_api_stats::counted(hip_api_stats::EVENT_DESTROY, stats_started);
                if status == HIP_SUCCESS { Ok(()) } else { Err(runtime.hip_error(status, "hipEventDestroy ordered P2P")) }
            });
        }
        let output = result?;
        // 成功路径仍停在目标设备；source event 由目标 completion 退休后回池。
        set_device(device_id)?;
        Ok(output)
    }

    /// 用一个源 stream event 把一组稳定 buffer 一次性交给目标 stream。
    /// route ids、route weights 与 activation 共享同一 producer 边界，逐个
    /// 建 event 既增加 host/API 开销，也容易在切卡后误取 stream。
    pub(crate) fn copy_stable_group_to_device_ordered_async_retained_by(sources: &[std::sync::Arc<Self>], device_id: i32, completion_device_id: i32) -> Result<Vec<Self>, String> {
        let source_device_id = sources.first().ok_or("ordered P2P buffer 组不能为空")?.device_id;
        if sources.iter().any(|source| source.device_id != source_device_id || source.async_allocated) {
            return Err("ordered P2P buffer 组必须来自同一 device 的显式 allocation".to_owned());
        }
        let outputs = sources.iter().map(|source| Self::allocate_peer(device_id, source.bytes)).collect::<Result<Vec<_>, _>>()?;
        let runtime = RocmRuntime::open()?;
        enable_peer_access(device_id, source_device_id)?;
        let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0")?;
        let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0")?;
        let wait: Symbol<HipStreamWaitEvent> = runtime.symbol(&runtime.hip, b"hipStreamWaitEvent\0")?;
        let copy: Symbol<HipMemcpyPeerAsync> = runtime.symbol(&runtime.hip, b"hipMemcpyPeerAsync\0")?;
        let destroy: Symbol<HipEventDestroy> = runtime.symbol(&runtime.hip, b"hipEventDestroy\0")?;
        let source_stream = compute_stream_for(source_device_id);
        let destination_stream = compute_stream_for(device_id);
        let mut event = device_buffer_pool(source_device_id).and_then(|pool| pool.lock().ok()?.available_events.pop()).map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
        set_device(source_device_id)?;
        if event.is_null() {
            let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventCreateWithFlags grouped ordered P2P"));
            }
        }
        let result = (|| {
            let status = unsafe { record(event, source_stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventRecord grouped ordered P2P source"));
            }
            set_device(device_id)?;
            let status = unsafe { wait(destination_stream, event, 0) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipStreamWaitEvent grouped ordered P2P destination"));
            }
            super::device_profile_operator(device_id, "handoff_copy")?;
            for (source, output) in sources.iter().zip(&outputs) {
                if source.bytes % 16 == 0 && crate::kernel::rocm::hip::active_compute_stream() == destination_stream {
                    super::peer_copy::try_peer_copy_kernel_ordered(device_id, output.pointer, source.pointer, source.bytes)?;
                } else {
                    let status = unsafe { copy(output.pointer, device_id, source.pointer, source_device_id, source.bytes, destination_stream) };
                    if status != HIP_SUCCESS {
                        return Err(format!("{}: source_device={source_device_id} destination_device={device_id} bytes={}", runtime.hip_error(status, "hipMemcpyPeerAsync grouped ordered P2P"), source.bytes));
                    }
                }
            }
            PENDING_P2P_SOURCES.with(|pending| {
                pending.borrow_mut().entry(completion_device_id).or_default().push(PendingP2pSource { sources: sources.to_vec(), event: event as usize });
            });
            Ok(())
        })();
        if result.is_err() {
            let _ = set_device(source_device_id).and_then(|()| {
                let status = unsafe { destroy(event) };
                if status == HIP_SUCCESS { Ok(()) } else { Err(runtime.hip_error(status, "hipEventDestroy grouped ordered P2P")) }
            });
        }
        result?;
        set_device(device_id)?;
        Ok(outputs)
    }

    /// 双向 P2P 交换先在两边 producer stream 都记录 ready event，
    /// 再分别向对端排队。若连续调用两次单向 copy，第二边的
    /// producer event 会落在第一边的入向 copy 之后，导致本可并行的两个传输串行。
    pub(crate) fn exchange_stable_groups_ordered_async_retained_by(left: &[std::sync::Arc<Self>], right: &[std::sync::Arc<Self>], completion_device_id: i32) -> Result<(Vec<Self>, Vec<Self>), String> {
        let left_device_id = left.first().ok_or("ordered P2P left buffer 组不能为空")?.device_id;
        let right_device_id = right.first().ok_or("ordered P2P right buffer 组不能为空")?.device_id;
        if left_device_id == right_device_id || left.iter().any(|source| source.device_id != left_device_id || source.async_allocated) || right.iter().any(|source| source.device_id != right_device_id || source.async_allocated) {
            return Err("ordered P2P 双向 buffer 组必须来自不同 device 的显式 allocation".to_owned());
        }
        let left_on_right = left.iter().map(|source| Self::allocate_peer(right_device_id, source.bytes)).collect::<Result<Vec<_>, _>>()?;
        let right_on_left = right.iter().map(|source| Self::allocate_peer(left_device_id, source.bytes)).collect::<Result<Vec<_>, _>>()?;
        let runtime = RocmRuntime::open()?;
        enable_peer_access(right_device_id, left_device_id)?;
        enable_peer_access(left_device_id, right_device_id)?;
        let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0")?;
        let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0")?;
        let wait: Symbol<HipStreamWaitEvent> = runtime.symbol(&runtime.hip, b"hipStreamWaitEvent\0")?;
        let copy: Symbol<HipMemcpyPeerAsync> = runtime.symbol(&runtime.hip, b"hipMemcpyPeerAsync\0")?;
        let destroy: Symbol<HipEventDestroy> = runtime.symbol(&runtime.hip, b"hipEventDestroy\0")?;
        let left_stream = compute_stream_for(left_device_id);
        let right_stream = compute_stream_for(right_device_id);
        let mut left_event = device_buffer_pool(left_device_id).and_then(|pool| pool.lock().ok()?.available_events.pop()).map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
        let create_event = |device_id: i32, event: &mut HipEvent| -> Result<(), String> {
            set_device(device_id)?;
            if event.is_null() {
                let status = unsafe { create(event, HIP_EVENT_DISABLE_TIMING) };
                if status != HIP_SUCCESS {
                    return Err(runtime.hip_error(status, "hipEventCreateWithFlags bidirectional ordered P2P"));
                }
            }
            Ok(())
        };
        create_event(left_device_id, &mut left_event)?;
        let mut right_event = device_buffer_pool(right_device_id).and_then(|pool| pool.lock().ok()?.available_events.pop()).map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
        if let Err(error) = create_event(right_device_id, &mut right_event) {
            set_device(left_device_id)?;
            let _ = unsafe { destroy(left_event) };
            return Err(error);
        }
        let result = (|| {
            set_device(left_device_id)?;
            let status = unsafe { record(left_event, left_stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventRecord bidirectional P2P left"));
            }
            set_device(right_device_id)?;
            let status = unsafe { record(right_event, right_stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventRecord bidirectional P2P right"));
            }

            // 两个 ready marker 都已入队；此后的两个目标 stream 不再
            // 依赖对向 copy，可以在两张卡上同时读取对端 BAR。
            let copy_group = |sources: &[std::sync::Arc<Self>], outputs: &[Self], source_device_id: i32, destination_device_id: i32, destination_stream: *mut c_void, event: HipEvent| -> Result<(), String> {
                set_device(destination_device_id)?;
                let status = unsafe { wait(destination_stream, event, 0) };
                if status != HIP_SUCCESS {
                    return Err(runtime.hip_error(status, "hipStreamWaitEvent bidirectional P2P destination"));
                }
                super::device_profile_operator(destination_device_id, "handoff_copy")?;
                for (source, output) in sources.iter().zip(outputs) {
                    if source.bytes % 16 == 0 && crate::kernel::rocm::hip::active_compute_stream() == destination_stream {
                        super::peer_copy::try_peer_copy_kernel_ordered(destination_device_id, output.pointer, source.pointer, source.bytes)?;
                    } else {
                        let status = unsafe { copy(output.pointer, destination_device_id, source.pointer, source_device_id, source.bytes, destination_stream) };
                        if status != HIP_SUCCESS {
                            return Err(format!("{}: source_device={source_device_id} destination_device={destination_device_id} bytes={}", runtime.hip_error(status, "hipMemcpyPeerAsync bidirectional ordered P2P"), source.bytes,));
                        }
                    }
                }
                Ok(())
            };
            copy_group(left, &left_on_right, left_device_id, right_device_id, right_stream, left_event)?;
            copy_group(right, &right_on_left, right_device_id, left_device_id, left_stream, right_event)?;
            PENDING_P2P_SOURCES.with(|pending| {
                let mut pending = pending.borrow_mut();
                let entries = pending.entry(completion_device_id).or_default();
                entries.push(PendingP2pSource { sources: left.to_vec(), event: left_event as usize });
                entries.push(PendingP2pSource { sources: right.to_vec(), event: right_event as usize });
            });
            Ok(())
        })();
        if result.is_err() {
            for (device_id, event) in [(left_device_id, left_event), (right_device_id, right_event)] {
                let _ = set_device(device_id).and_then(|()| {
                    let status = unsafe { destroy(event) };
                    if status == HIP_SUCCESS { Ok(()) } else { Err(runtime.hip_error(status, "hipEventDestroy bidirectional ordered P2P")) }
                });
            }
        }
        result?;
        set_device(completion_device_id)?;
        Ok((left_on_right, right_on_left))
    }

    /// 与 grouped ordered P2P 相同，但直接写入调用方持有的固定目标地址。
    /// pair-native cache 用它把增量落到 peer cache offset，省掉临时 P2P
    /// allocation 与随后一次同卡 D2D。
    pub(crate) fn copy_stable_group_into_device_ordered_async_retained_by(sources: &[std::sync::Arc<Self>], destinations: &[(&Self, usize)], device_id: i32, completion_device_id: i32) -> Result<(), String> {
        let source_device_id = sources.first().ok_or("ordered P2P buffer 组不能为空")?.device_id;
        if sources.len() != destinations.len()
            || sources.iter().any(|source| source.device_id != source_device_id || source.async_allocated)
            || destinations.iter().zip(sources).any(|((destination, offset), source)| destination.device_id != device_id || offset.checked_add(source.bytes).is_none_or(|end| end > destination.bytes))
        {
            return Err("ordered P2P buffer 组来源或固定目标非法".to_owned());
        }
        let runtime = RocmRuntime::open()?;
        enable_peer_access(device_id, source_device_id)?;
        let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0")?;
        let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0")?;
        let wait: Symbol<HipStreamWaitEvent> = runtime.symbol(&runtime.hip, b"hipStreamWaitEvent\0")?;
        let copy: Symbol<HipMemcpyPeerAsync> = runtime.symbol(&runtime.hip, b"hipMemcpyPeerAsync\0")?;
        let destroy: Symbol<HipEventDestroy> = runtime.symbol(&runtime.hip, b"hipEventDestroy\0")?;
        let source_stream = compute_stream_for(source_device_id);
        let destination_stream = compute_stream_for(device_id);
        let mut event = device_buffer_pool(source_device_id).and_then(|pool| pool.lock().ok()?.available_events.pop()).map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
        set_device(source_device_id)?;
        if event.is_null() {
            let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventCreateWithFlags grouped ordered P2P"));
            }
        }
        let result = (|| {
            let status = unsafe { record(event, source_stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventRecord grouped ordered P2P source"));
            }
            set_device(device_id)?;
            let status = unsafe { wait(destination_stream, event, 0) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipStreamWaitEvent grouped ordered P2P destination"));
            }
            super::device_profile_operator(device_id, "handoff_copy")?;
            for (source, (destination, offset)) in sources.iter().zip(destinations) {
                let destination_pointer = unsafe { destination.pointer.cast::<u8>().add(*offset).cast() };
                if source.bytes % 16 == 0 && crate::kernel::rocm::hip::active_compute_stream() == destination_stream {
                    super::peer_copy::try_peer_copy_kernel_ordered(device_id, destination_pointer, source.pointer, source.bytes)?;
                } else {
                    let status = unsafe { copy(destination_pointer, device_id, source.pointer, source_device_id, source.bytes, destination_stream) };
                    if status != HIP_SUCCESS {
                        return Err(format!("{}: source_device={source_device_id} destination_device={device_id} bytes={}", runtime.hip_error(status, "hipMemcpyPeerAsync grouped ordered P2P"), source.bytes));
                    }
                }
            }
            PENDING_P2P_SOURCES.with(|pending| {
                pending.borrow_mut().entry(completion_device_id).or_default().push(PendingP2pSource { sources: sources.to_vec(), event: event as usize });
            });
            Ok(())
        })();
        if result.is_err() {
            let _ = set_device(source_device_id).and_then(|()| {
                let status = unsafe { destroy(event) };
                if status == HIP_SUCCESS { Ok(()) } else { Err(runtime.hip_error(status, "hipEventDestroy grouped ordered P2P")) }
            });
        }
        result?;
        set_device(device_id)?;
        Ok(())
    }

    /// 在当前 submission stream 尾部把阶段结果稳定到显式池；后续 completion
    /// event 覆盖这次复制。
    pub(crate) fn copy_to_stable_deferred(&self) -> Result<Self, String> {
        set_device(self.device_id)?;
        let output = Self::allocate_peer(self.device_id, self.bytes)?;
        let runtime = RocmRuntime::open()?;
        let copy = runtime.memcpy_async()?;
        let stats_started = hip_api_stats::start();
        let status = unsafe { copy(output.pointer, self.pointer, self.bytes, HIP_MEMORY_COPY_DEVICE_TO_DEVICE, crate::kernel::rocm::hip::active_compute_stream()) };
        hip_api_stats::counted(hip_api_stats::MEMCPY_ASYNC, stats_started);
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipMemcpyAsync stable stage output"));
        }
        Ok(output)
    }

    pub(crate) fn copy_to_device(&self, device_id: i32) -> Result<Self, String> {
        let source_stream = compute_stream_for(self.device_id);
        let destination_stream = compute_stream_for(device_id);
        if !source_stream.is_null() && !destination_stream.is_null() {
            return self.copy_to_device_stream_ordered(device_id, source_stream, destination_stream);
        }
        synchronize_device(self.device_id, "hipMemcpyPeer source synchronize")?;
        // ROCm 7.2 的 hipMemcpyPeer 不能可靠读取 stream-ordered allocation；先稳定到显式池。
        let stable_source = if self.async_allocated {
            let stable = Self::allocate_peer(self.device_id, self.bytes)?;
            stable.copy_from_device(0, self, 0, self.bytes)?;
            // 无显式 compute stream 时，下面的同步 peer copy 不会替这次
            // async D2D 建立依赖；先完成稳定化，避免目标卡读到未写入的池块。
            synchronize_device(self.device_id, "hipMemcpyPeer stable source synchronize")?;
            Some(stable)
        } else {
            None
        };
        let source = stable_source.as_ref().unwrap_or(self);
        let output = Self::allocate_peer(device_id, source.bytes)?;
        let runtime = RocmRuntime::open()?;
        enable_peer_access(device_id, source.device_id)?;
        let copy: Symbol<HipMemcpyPeer> = runtime.symbol(&runtime.hip, b"hipMemcpyPeer\0")?;
        let status = unsafe { copy(output.pointer, device_id, source.pointer, source.device_id, source.bytes) };
        if status != HIP_SUCCESS {
            return Err(format!("{}: source_device={} destination_device={device_id} bytes={}", runtime.hip_error(status, "hipMemcpyPeer activation"), source.device_id, source.bytes));
        }
        synchronize_device(device_id, "hipMemcpyPeer destination synchronize")?;
        Ok(output)
    }

    /// DSpark 跨卡迁移只在自己的 compute stream 间建立依赖并等待复制完成，
    /// 不同步同卡 target 默认 stream。等待仅覆盖很小的 hidden P2P，保证来源
    /// activation 返回后可立即释放，同时保留两条模型计算链的设备级重叠。
    fn copy_to_device_stream_ordered(&self, device_id: i32, source_stream: *mut c_void, destination_stream: *mut c_void) -> Result<Self, String> {
        let runtime = RocmRuntime::open()?;
        enable_peer_access(device_id, self.device_id)?;
        let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0")?;
        let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0")?;
        let wait: Symbol<HipStreamWaitEvent> = runtime.symbol(&runtime.hip, b"hipStreamWaitEvent\0")?;
        let copy: Symbol<HipMemcpyPeerAsync> = runtime.symbol(&runtime.hip, b"hipMemcpyPeerAsync\0")?;
        let synchronize: Symbol<HipStreamSynchronize> = runtime.symbol(&runtime.hip, b"hipStreamSynchronize\0")?;
        let destroy: Symbol<HipEventDestroy> = runtime.symbol(&runtime.hip, b"hipEventDestroy\0")?;

        set_device(self.device_id)?;
        let stable_source = if self.async_allocated {
            let stable = Self::allocate_peer(self.device_id, self.bytes)?;
            stable.copy_from_device(0, self, 0, self.bytes)?;
            Some(stable)
        } else {
            None
        };
        let source = stable_source.as_ref().unwrap_or(self);
        let mut event = ptr::null_mut();
        let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipEventCreateWithFlags DSpark P2P"));
        }
        let result = (|| {
            let status = unsafe { record(event, source_stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipEventRecord DSpark P2P source"));
            }
            let output = Self::allocate_peer(device_id, source.bytes)?;
            set_device(device_id)?;
            let status = unsafe { wait(destination_stream, event, 0) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipStreamWaitEvent DSpark P2P destination"));
            }
            let status = unsafe { copy(output.pointer, device_id, source.pointer, source.device_id, source.bytes, destination_stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipMemcpyPeerAsync DSpark activation"));
            }
            let status = unsafe { synchronize(destination_stream) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipStreamSynchronize DSpark P2P"));
            }
            Ok(output)
        })();
        let _ = set_device(self.device_id).and_then(|()| {
            let status = unsafe { destroy(event) };
            if status == HIP_SUCCESS { Ok(()) } else { Err(runtime.hip_error(status, "hipEventDestroy DSpark P2P")) }
        });
        result
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// 底层 pointer 的真实 allocation 大小；显存归属统计不能用逻辑 view/请求大小。
    pub(crate) fn allocation_bytes(&self) -> usize {
        self.owner.as_ref().map_or(self.capacity_bytes, |owner| owner.capacity_bytes)
    }

    /// 设备地址(kernels 组装 meta 时使用；仅 crate 内部)。
    pub(crate) fn device_pointer(&self) -> usize {
        self.pointer as usize
    }

    /// reserved tensor 用覆盖 allocation 起点的 view 表示当前有效前缀；只有这种
    /// view 才能安全在 owner 尾部追加，带 offset 的滑窗 view 必须重新整理。
    pub(crate) fn prefix_capacity_owner(&self) -> Option<std::sync::Arc<Self>> {
        let owner = self.owner.as_ref()?;
        (self.pointer == owner.pointer).then(|| owner.clone())
    }

    pub(crate) fn is_async_allocated(&self) -> bool {
        self.async_allocated
    }

    pub(crate) fn retain_for_active_stage(self: &std::sync::Arc<Self>) {
        if self.retain_until_stage_completion {
            PENDING_STAGE_INPUTS.with(|inputs| inputs.borrow_mut().entry(self.device_id).or_default().push(self.clone()));
        }
    }

    /// 只在 stage0 已激活真实 submission stream 后调用。pinned host 槽由
    /// completion 持有的 DeviceBuffer Arc 保活，因此不需要 host 同步。
    pub(crate) fn enqueue_deferred_upload(&self) -> Result<(), String> {
        let Some(staging) = self.deferred_host.as_ref() else { return Ok(()) };
        if self.deferred_upload_enqueued.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return Ok(());
        }
        set_device(self.device_id)?;
        let runtime = RocmRuntime::open()?;
        let copy = runtime.memcpy_async()?;
        let stats_started = hip_api_stats::start();
        let status = unsafe { copy(self.pointer, staging.pointer, self.bytes, HIP_MEMORY_COPY_HOST_TO_DEVICE, crate::kernel::rocm::hip::active_compute_stream()) };
        hip_api_stats::counted(hip_api_stats::MEMCPY_ASYNC, stats_started);
        if status != HIP_SUCCESS {
            self.deferred_upload_enqueued.store(false, std::sync::atomic::Ordering::Release);
            return Err(runtime.hip_error(status, "hipMemcpyAsync deferred boundary H2D"));
        }
        record_host_transfer(true, self.bytes);
        Ok(())
    }

    pub(crate) fn view(owner: std::sync::Arc<Self>, offset: usize, bytes: usize) -> Result<Self, String> {
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|end| end > owner.bytes) {
            return Err(format!("ROCm device view 越界: offset={offset} bytes={bytes} capacity={}", owner.bytes));
        }
        let pointer = unsafe { owner.pointer.cast::<u8>().add(offset).cast() };
        Ok(Self {
            device_id: owner.device_id,
            pointer,
            bytes,
            capacity_bytes: bytes,
            recyclable: false,
            retain_until_stage_completion: false,
            stage_completion_ready: std::sync::atomic::AtomicBool::new(false),
            deferred_host: None,
            deferred_upload_enqueued: std::sync::atomic::AtomicBool::new(false),
            // view 仍指向 owner 的 allocation；保留来源，跨卡复制才能稳定化 hipMallocAsync。
            async_allocated: owner.async_allocated,
            owner: Some(owner),
        })
    }

    pub fn download_f32(&self, elements: usize) -> Result<Vec<f32>, String> {
        let bytes = elements.checked_mul(std::mem::size_of::<f32>()).ok_or("ROCm tensor 下载大小溢出")?;
        let mut output = vec![0.0_f32; elements];
        self.copy_to_host(unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), bytes) })?;
        Ok(output)
    }

    pub fn download_u16(&self, elements: usize) -> Result<Vec<u16>, String> {
        let bytes = elements.checked_mul(std::mem::size_of::<u16>()).ok_or("ROCm BF16 tensor 下载大小溢出")?;
        let mut output = vec![0_u16; elements];
        self.copy_to_host(unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), bytes) })?;
        Ok(output)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.owner.is_some() {
            return;
        }
        if self.pointer.is_null() || set_device(self.device_id).is_err() {
            return;
        }
        if self.async_allocated {
            if let Ok(runtime) = RocmRuntime::open() {
                if let Ok(free) = runtime.free_async() {
                    let stats_started = hip_api_stats::start();
                    let status = unsafe { free(self.pointer, crate::kernel::rocm::hip::active_compute_stream()) };
                    hip_api_stats::counted(hip_api_stats::FREE_ASYNC, stats_started);
                    if status == HIP_SUCCESS {
                        return;
                    }
                    if options().log_memory {
                        eprintln!("[rocm-memory] hipFreeAsync device={} bytes={} status={}，回退 hipFree", self.device_id, self.bytes, status);
                    }
                }
            }
        }
        if self.recyclable && self.retain_until_stage_completion && self.stage_completion_ready.load(std::sync::atomic::Ordering::Acquire) && options().memory_pool {
            recycle_completed_stage_buffers(self.device_id, vec![(self.pointer as usize, self.capacity_bytes)]);
            return;
        }
        if self.recyclable && (defer_stage_buffer_recycle(self.device_id, self.pointer, self.capacity_bytes) || recycle_device_buffer(self.device_id, self.pointer, self.capacity_bytes)) {
            return;
        }
        if let Ok(runtime) = RocmRuntime::open() {
            if let Ok(free) = runtime.free() {
                let started = std::time::Instant::now();
                let start_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
                let _ = unsafe { free(self.pointer) };
                let duration_us = started.elapsed().as_micros();
                if duration_us >= 5_000 {
                    let complete_us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
                    eprintln!("[rocm-buffer-retire-slow] ts_us={start_us} device={} bytes={} path=hipFree duration_us={duration_us} complete_us={complete_us}", self.device_id, self.capacity_bytes);
                }
            }
        }
    }
}

pub(crate) fn release_tensor_workspace(device_id: i32) -> Result<(), String> {
    let before = device_memory_info(device_id);
    let mut released = synchronize_device(device_id, "HIP phase workspace pre-release");
    if released.is_ok() {
        TENSOR_WORKSPACES.with(|workspaces| {
            workspaces.borrow_mut().retain(|&(workspace_device, _), _| workspace_device != device_id);
        });
        ATTENTION_WORKSPACES.with(|workspaces| {
            workspaces.borrow_mut().retain(|&(workspace_device, _), _| workspace_device != device_id);
        });
        tensor::release_deferred_tensor_workspace(device_id);
        CT_QUANTIZED_WORKSPACES.with(|workspaces| {
            workspaces.borrow_mut().retain(|&(workspace_device, _), _| workspace_device != device_id);
        });
        release_gguf_fused_workspace(device_id);
        release_paged_mla_workspaces(device_id);
        release_device_buffer_pool(device_id);
        released = synchronize_device(device_id, "HIP phase workspace release").and_then(|()| trim_device_memory_pool(device_id));
    }
    if options().log_memory {
        let format = |info: Result<(usize, usize), String>| match info {
            Ok((free, total)) => format!("free={:.2}GiB used={:.2}GiB", free as f64 / (1_u64 << 30) as f64, (total - free) as f64 / (1_u64 << 30) as f64),
            Err(error) => format!("error={error}"),
        };
        eprintln!("[rocm-memory] device={device_id} before={} after={} release={}", format(before), format(device_memory_info(device_id)), released.as_ref().map(|()| "ok").unwrap_or_else(|error| error));
    }
    released
}

pub fn device_memory_info(device_id: i32) -> Result<(usize, usize), String> {
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let get_info: Symbol<HipMemGetInfo> = runtime.symbol(&runtime.hip, b"hipMemGetInfo\0")?;
    let mut free = 0;
    let mut total = 0;
    let status = unsafe { get_info(&mut free, &mut total) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipMemGetInfo"));
    }
    Ok((free, total))
}

/// decode 期间的新请求准入不能把可驱逐软池误算成会话常驻显存。这里只晋升
/// 已完成 event 并统计 available buffer，不释放指针，避免容量探测破坏热池。
pub fn device_admission_available_bytes(device_id: i32) -> Result<usize, String> {
    promote_device_buffers(device_id);
    let reclaimable = device_buffer_pool(device_id)
        .and_then(|pool| pool.lock().ok())
        .map(|pool| pool.buffers.iter().filter(|((buffer_device, _), _)| *buffer_device == device_id).map(|((_, bytes), pointers)| bytes.saturating_mul(pointers.len())).fold(0usize, usize::saturating_add))
        .unwrap_or(0);
    let (free, total) = device_memory_info(device_id)?;
    Ok(free.saturating_add(reclaimable).min(total))
}

/// 异步释放只把显存交还默认池；阶段结束时裁剪池，供后续同步分配复用。
pub(super) fn trim_device_memory_pool(device_id: i32) -> Result<(), String> {
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let get_pool: Symbol<HipDeviceGetDefaultMemPool> = runtime.symbol(&runtime.hip, b"hipDeviceGetDefaultMemPool\0")?;
    let trim: Symbol<HipMemPoolTrimTo> = runtime.symbol(&runtime.hip, b"hipMemPoolTrimTo\0")?;
    let mut pool = ptr::null_mut();
    let status = unsafe { get_pool(&mut pool, device_id) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipDeviceGetDefaultMemPool"));
    }
    let status = unsafe { trim(pool, 0) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipMemPoolTrimTo"));
    }
    Ok(())
}
