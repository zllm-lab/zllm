use super::*;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

const DETAIL_SAMPLE_STRIDE: u64 = 16;

#[derive(Clone, Copy, Default)]
struct TimingStat {
    count: u64,
    total_ms: f64,
    max_ms: f32,
}

impl TimingStat {
    fn add(&mut self, milliseconds: f32) {
        self.count += 1;
        self.total_ms += f64::from(milliseconds);
        self.max_ms = self.max_ms.max(milliseconds);
    }

    fn delta(self, previous: Self) -> Self {
        Self { count: self.count.saturating_sub(previous.count), total_ms: (self.total_ms - previous.total_ms).max(0.0), max_ms: self.max_ms }
    }

    fn average_ms(self) -> f64 {
        if self.count == 0 { 0.0 } else { self.total_ms / self.count as f64 }
    }
}

struct TimingMarker {
    event: usize,
    label: &'static str,
}

#[derive(Default)]
struct DeviceTimeline {
    stage_batches: u64,
    stage_ends: u64,
    reported_stage_ends: u64,
    detailed: bool,
    markers: VecDeque<TimingMarker>,
    available_events: Vec<usize>,
    stats: HashMap<&'static str, TimingStat>,
    reported: HashMap<&'static str, TimingStat>,
}

static ENABLED: AtomicBool = AtomicBool::new(false);
// 诊断 profile 需要把长 prefill 与 decode 的累计 event 分开。第一个
// latency submission 负责 flush，其他 stage 等它完成后再提交 decode，
// 避免边界两侧的 scope event 交叉计入。
static DECODE_BOUNDARY: AtomicU8 = AtomicU8::new(0);
static TIMELINES: OnceLock<Mutex<HashMap<i32, DeviceTimeline>>> = OnceLock::new();
static SCOPE_TIMELINES: OnceLock<Mutex<HashMap<i32, DeviceTimeline>>> = OnceLock::new();

fn timelines() -> &'static Mutex<HashMap<i32, DeviceTimeline>> {
    TIMELINES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn scope_timelines() -> &'static Mutex<HashMap<i32, DeviceTimeline>> {
    SCOPE_TIMELINES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn enable_device_profile() {
    DECODE_BOUNDARY.store(0, Ordering::Release);
    ENABLED.store(true, Ordering::Release);
}

pub(crate) fn device_profile_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

pub(crate) fn device_profile_decode_boundary() {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    match DECODE_BOUNDARY.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => {
            report_device_profiles(0, 0);
            eprintln!("[device-profile-boundary] phase=decode");
            DECODE_BOUNDARY.store(2, Ordering::Release);
        }
        Err(1) => {
            while DECODE_BOUNDARY.load(Ordering::Acquire) == 1 {
                std::hint::spin_loop();
            }
        }
        Err(_) => {}
    }
}

fn drain_completed(device_id: i32, timeline: &mut DeviceTimeline, runtime: &RocmRuntime) -> Result<(), String> {
    let query: Symbol<HipEventQuery> = runtime.symbol(&runtime.hip, b"hipEventQuery\0")?;
    let elapsed: Symbol<HipEventElapsedTime> = runtime.symbol(&runtime.hip, b"hipEventElapsedTime\0")?;
    while timeline.markers.len() >= 2 {
        let end = timeline.markers.get(1).expect("timing marker pair").event as HipEvent;
        match unsafe { query(end) } {
            HIP_SUCCESS => {}
            HIP_ERROR_NOT_READY => break,
            status => return Err(runtime.hip_error(status, "hipEventQuery device profile")),
        }
        let start = timeline.markers.front().expect("timing marker start");
        let mut milliseconds = 0.0f32;
        let status = unsafe { elapsed(&mut milliseconds, start.event as HipEvent, end) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipEventElapsedTime device profile"));
        }
        if milliseconds >= 500.0 {
            eprintln!("[device-profile-slow] device={device_id} label={} device_ms={milliseconds:.3}", start.label);
        }
        timeline.stats.entry(start.label).or_default().add(milliseconds);
        let completed = timeline.markers.pop_front().expect("timing marker complete");
        timeline.available_events.push(completed.event);
    }
    let _ = device_id;
    Ok(())
}

fn record_locked(device_id: i32, timeline: &mut DeviceTimeline, label: &'static str) -> Result<(), String> {
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0")?;
    let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0")?;
    let destroy: Symbol<HipEventDestroy> = runtime.symbol(&runtime.hip, b"hipEventDestroy\0")?;
    let mut event = timeline.available_events.pop().map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
    if event.is_null() {
        let status = unsafe { create(&mut event, 0) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipEventCreateWithFlags device profile"));
        }
    }
    let status = unsafe { record(event, crate::kernel::rocm::hip::active_compute_stream()) };
    if status != HIP_SUCCESS {
        let _ = unsafe { destroy(event) };
        return Err(runtime.hip_error(status, "hipEventRecord device profile"));
    }
    timeline.markers.push_back(TimingMarker { event: event as usize, label });
    drain_completed(device_id, timeline, runtime)
}

pub(crate) fn device_profile_stage_begin(device_id: i32, detailed_eligible: bool) -> Result<(), String> {
    ENABLED.store(true, Ordering::Release);
    let mut timelines = timelines().lock().map_err(|_| "ROCm device profile mutex 已损坏".to_owned())?;
    let timeline = timelines.entry(device_id).or_default();
    timeline.stage_batches += 1;
    timeline.detailed = detailed_eligible && timeline.stage_batches % DETAIL_SAMPLE_STRIDE == 1;
    let label = if timeline.detailed { "handoff" } else { "stage_total" };
    record_locked(device_id, timeline, label)
}

pub(crate) fn device_profile_operator(device_id: i32, label: &'static str) -> Result<(), String> {
    if !ENABLED.load(Ordering::Acquire) {
        return Ok(());
    }
    let mut timelines = timelines().lock().map_err(|_| "ROCm device profile mutex 已损坏".to_owned())?;
    let Some(timeline) = timelines.get_mut(&device_id) else {
        return Ok(());
    };
    if !timeline.detailed {
        return Ok(());
    }
    record_locked(device_id, timeline, label)
}

pub(crate) fn device_profile_stage_end(device_id: i32) -> Result<(), String> {
    let mut timelines = timelines().lock().map_err(|_| "ROCm device profile mutex 已损坏".to_owned())?;
    let timeline = timelines.get_mut(&device_id).ok_or_else(|| format!("ROCm device={device_id} profile stage 尚未开始"))?;
    record_locked(device_id, timeline, "pipeline_idle")?;
    timeline.detailed = false;
    timeline.stage_ends += 1;
    Ok(())
}

/// 记录模型组合层的独立设备区间，不混入常驻 stage 的 busy/idle 统计。
pub(crate) fn device_profile_scope_begin(device_id: i32, label: &'static str) -> Result<(), String> {
    ENABLED.store(true, Ordering::Release);
    let mut timelines = scope_timelines().lock().map_err(|_| "ROCm device scope profile mutex 已损坏".to_owned())?;
    let timeline = timelines.entry(device_id).or_default();
    if timeline.detailed {
        return Err(format!("ROCm device={device_id} profile scope 不允许嵌套"));
    }
    timeline.detailed = true;
    timeline.stage_batches += 1;
    record_locked(device_id, timeline, label)
}

pub(crate) fn device_profile_scope_end(device_id: i32) -> Result<(), String> {
    let mut timelines = scope_timelines().lock().map_err(|_| "ROCm device scope profile mutex 已损坏".to_owned())?;
    let timeline = timelines.get_mut(&device_id).ok_or_else(|| format!("ROCm device={device_id} profile scope 尚未开始"))?;
    if !timeline.detailed {
        return Err(format!("ROCm device={device_id} profile scope 已结束"));
    }
    record_locked(device_id, timeline, "scope_idle")?;
    timeline.detailed = false;
    timeline.stage_ends += 1;
    Ok(())
}

fn stat_delta(timeline: &DeviceTimeline, label: &'static str) -> TimingStat {
    timeline.stats.get(label).copied().unwrap_or_default().delta(timeline.reported.get(label).copied().unwrap_or_default())
}

pub(crate) fn report_device_profiles(rounds: usize, active: usize) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    let Ok(mut timelines) = timelines().lock() else {
        eprintln!("[device-profile-error] mutex 已损坏");
        return;
    };
    let mut devices = timelines.keys().copied().collect::<Vec<_>>();
    devices.sort_unstable();
    for device_id in devices {
        let timeline = timelines.get_mut(&device_id).expect("device profile 存在");
        let result = set_device(device_id).and_then(|()| RocmRuntime::open().and_then(|runtime| drain_completed(device_id, timeline, runtime)));
        if let Err(error) = result {
            eprintln!("[device-profile-error] device={device_id} error={error}");
            continue;
        }
        // 普通 stage 用 stage_total 单段计时；每 16 个 stage 的详细样本则由
        // handoff/operator/stabilize 连续区间覆盖，二者相加才是完整 busy 时间。
        let detailed_busy_ms = timeline.stats.keys().filter(|&&label| label != "stage_total" && label != "pipeline_idle").map(|&label| stat_delta(timeline, label).total_ms).sum::<f64>();
        let busy_ms = stat_delta(timeline, "stage_total").total_ms + detailed_busy_ms;
        let idle = stat_delta(timeline, "pipeline_idle");
        let total_ms = busy_ms + idle.total_ms;
        let bubble_pct = if total_ms == 0.0 { 0.0 } else { idle.total_ms * 100.0 / total_ms };
        let stages = timeline.stage_ends.saturating_sub(timeline.reported_stage_ends);
        let avg = |label| stat_delta(timeline, label).average_ms();
        eprintln!(
            "[device-profile] rounds={rounds} active={active} device={device_id} stages={stages} busy_ms={busy_ms:.3} idle_ms={:.3} bubble_pct={bubble_pct:.2} stage_avg_ms={:.3} handoff_ms={:.3} handoff_copy_ms={:.3} input_expand_ms={:.3} attention_hc_ms={:.3} attn_qkv_ms={:.3} attn_compress_ms={:.3} attn_index_ms={:.3} attn_csa_ms={:.3} attn_out_ms={:.3} attention_merge_ms={:.3} ffn_hc_ms={:.3} moe_hash_ms={:.3} moe_score_ms={:.3} ffn_merge_hash_ms={:.3} ffn_merge_score_ms={:.3} layer_compact_ms={:.3} stage_stabilize_ms={:.3} stage_captures_ms={:.3} stage_tail_ms={:.3} glm_attention_ms={:.3} glm_attn_query_ms={:.3} glm_index_key_ms={:.3} glm_index_query_ms={:.3} glm_index_select_ms={:.3} glm_attn_kv_ms={:.3} glm_attn_mla_ms={:.3} glm_attn_out_ms={:.3} glm_ffn_ms={:.3} glm_dense_gate_up_ms={:.3} glm_dense_down_ms={:.3} glm_dense_residual_ms={:.3} moe_shared_ms={:.3} moe_routed_ms={:.3} moe_experts_ms={:.3} moe_epilogue_ms={:.3} glm_layer_tail_ms={:.3}",
            idle.total_ms,
            if stages == 0 { 0.0 } else { busy_ms / stages as f64 },
            avg("handoff"),
            avg("handoff_copy"),
            avg("input_expand"),
            avg("attention_hc"),
            avg("attn_qkv"),
            avg("attn_compress"),
            avg("attn_index"),
            avg("attn_csa"),
            avg("attn_out"),
            avg("attention_merge"),
            avg("ffn_hc"),
            avg("moe_hash"),
            avg("moe_score"),
            avg("ffn_merge_hash"),
            avg("ffn_merge_score"),
            avg("layer_compact"),
            avg("stage_stabilize"),
            avg("stage_captures"),
            avg("stage_tail"),
            avg("glm_attention"),
            avg("glm_attn_query"),
            avg("glm_index_key"),
            avg("glm_index_query"),
            avg("glm_index_select"),
            avg("glm_attn_kv"),
            avg("glm_attn_mla"),
            avg("glm_attn_out"),
            avg("glm_ffn"),
            avg("glm_dense_gate_up"),
            avg("glm_dense_down"),
            avg("glm_dense_residual"),
            avg("moe_shared"),
            avg("moe_routed"),
            avg("moe_experts"),
            avg("moe_epilogue"),
            avg("glm_layer_tail"),
        );
        timeline.reported = timeline.stats.clone();
        timeline.reported_stage_ends = timeline.stage_ends;
    }
    drop(timelines);
    report_scope_profiles(rounds, active);
    super::hip_api_stats::report();
}

fn report_scope_profiles(rounds: usize, active: usize) {
    let Ok(mut timelines) = scope_timelines().lock() else {
        eprintln!("[device-scope-profile-error] mutex 已损坏");
        return;
    };
    let mut devices = timelines.keys().copied().collect::<Vec<_>>();
    devices.sort_unstable();
    for device_id in devices {
        let timeline = timelines.get_mut(&device_id).expect("device scope profile 存在");
        let result = set_device(device_id).and_then(|()| RocmRuntime::open().and_then(|runtime| drain_completed(device_id, timeline, runtime)));
        if let Err(error) = result {
            eprintln!("[device-scope-profile-error] device={device_id} error={error}");
            continue;
        }
        let mut labels = timeline.stats.keys().copied().filter(|label| *label != "scope_idle").collect::<Vec<_>>();
        labels.sort_unstable();
        for label in labels {
            let stat = stat_delta(timeline, label);
            if stat.count == 0 {
                continue;
            }
            eprintln!("[device-scope-profile] rounds={rounds} active={active} device={device_id} label={label} count={} total_ms={:.3} avg_ms={:.3} max_ms={:.3}", stat.count, stat.total_ms, stat.average_ms(), stat.max_ms,);
        }
        timeline.reported = timeline.stats.clone();
        timeline.reported_stage_ends = timeline.stage_ends;
    }
}
