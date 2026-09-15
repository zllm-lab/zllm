//! per-device TLSF arena 管理：多段按需扩、pointer→node 映射、淘汰吸收。
//!
//! 两级架构的 L2（docs/rocm-mempool-arena-design-20260914.md §4）：
//! 精确桶（device_buffer.rs）miss 时在此供给，软池淘汰/OOM 回收的
//! arena 出身块在此合并归还。内核是 backend 无关的
//! [`crate::mempool::arena::Arena`]，本文件只做 HIP 侧段管理与映射。
//! 段按请求尺寸的 2 的幂动态扩展（受配置上界与空闲显存约束），
//! 不预分配大段。

use super::ffi::HIP_DEVICE_COUNT;
use super::*;
use crate::mempool::Completion;
use crate::mempool::arena::{Arena, NodeIndex};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

/// node 直接挂的 HIP event 完成证明。
pub(super) struct RocmCompletion {
    device_id: i32,
    event: usize,
}

impl Completion for RocmCompletion {
    fn is_complete(&self) -> bool {
        let Ok(runtime) = RocmRuntime::open() else { return false };
        let Ok(query) = runtime.event_query() else { return false };
        let status = unsafe { query(self.event as HipEvent) };
        status == HIP_SUCCESS
    }
}

impl Drop for RocmCompletion {
    fn drop(&mut self) {
        // Arena::promote 在 arena 锁内销毁 completion。这里不能再获取 L1 pool
        // 锁，否则会与 pool 淘汰时的 pool→arena 顺序形成锁反转。
        if super::set_device(self.device_id).is_err() {
            return;
        }
        if let Ok(runtime) = RocmRuntime::open()
            && let Ok(destroy) = runtime.event_destroy()
        {
            let _ = unsafe { destroy(self.event as HipEvent) };
        }
    }
}

/// 单个物理段：一次 hipMalloc 的连续区，内含独立 TLSF。coalesce 只在
/// 段内有意义（跨段物理不连续），因此每段一个内核实例是正确结构。
struct ArenaSegment {
    base: usize,
    bytes: usize,
    arena: Arena<RocmCompletion>,
}

pub(super) struct DeviceArena {
    segments: Vec<ArenaSegment>,
    total_bytes: usize,
    allocations: HashMap<usize, (u32, NodeIndex)>,
}

static DEVICE_ARENAS: OnceLock<Vec<Mutex<Option<DeviceArena>>>> = OnceLock::new();

/// arena 总上界（字节）。默认 0 = 关闭；由 yaml `device_arena_gib` 经
/// [`set_arena_bound_bytes`] 设置（行为配置一律 yaml，不做 env 开关）。
/// 段不预分配：按请求尺寸的 2 的幂动态扩展，总占用不超过上界。
static ARENA_BOUND_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_arena_bound_bytes(bytes: usize) {
    ARENA_BOUND_BYTES.store(bytes, std::sync::atomic::Ordering::Release);
}

fn arena_bound_bytes() -> usize {
    ARENA_BOUND_BYTES.load(std::sync::atomic::Ordering::Acquire)
}

fn device_arenas() -> Option<&'static Vec<Mutex<Option<DeviceArena>>>> {
    if arena_bound_bytes() == 0 {
        return None;
    }
    let count = usize::try_from(*HIP_DEVICE_COUNT.get()?.as_ref().ok()?).ok()?;
    Some(DEVICE_ARENAS.get_or_init(|| (0..count).map(|_| Mutex::new(None)).collect()))
}

/// 新段尺寸 = 请求的 2 的幂，受剩余上界与当前空闲显存（半数，防止一次
/// 勒死设备——2026-09-14 H3 15s 1MP U8 曾被固定 16 GiB 段 + 权重 + 在途
/// 瞬时挤爆 48G 卡）双重约束；约束后放不下请求则不建段。
fn expand_segment(device_id: i32, request: usize, total_bytes: usize) -> Option<ArenaSegment> {
    let mut bytes = request.next_power_of_two();
    if let Some(remaining) = arena_bound_bytes().checked_sub(total_bytes) {
        bytes = bytes.min(remaining);
    } else {
        return None;
    }
    if let Ok((free, _total)) = super::device_buffer::device_memory_info(device_id)
        && free > 0
        && free / 2 < bytes
    {
        bytes = free / 2;
    }
    if bytes < request {
        return None;
    }
    set_device(device_id).ok()?;
    let runtime = RocmRuntime::open().ok()?;
    let malloc = runtime.malloc().ok()?;
    let mut base = ptr::null_mut();
    if unsafe { malloc(&mut base, bytes) } != HIP_SUCCESS {
        eprintln!("[rocm-arena] device={device_id} 扩段 {bytes} MiB 失败，本轮回退驱动分配");
        return None;
    }
    eprintln!("[rocm-arena] device={device_id} segment={} MiB 建立 base={base:p} total={} MiB", bytes >> 20, (total_bytes + bytes) >> 20);
    Some(ArenaSegment { base: base as usize, bytes, arena: Arena::new(bytes as u64, 65536) })
}

/// L1 精确桶 miss 后的 L2 供给：先试已有段（老段优先，减少碎片分散），
/// 全 miss 则按请求扩段。arena 关闭/上界或显存不足时返回 None，调用方
/// 回退驱动分配路径。
pub(super) fn allocate(device_id: i32, bytes: usize) -> Option<(*mut c_void, usize)> {
    let arenas = device_arenas()?;
    let slot = arenas.get(usize::try_from(device_id).ok()?)?;
    let mut guard = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.is_none() {
        *guard = Some(DeviceArena { segments: Vec::new(), total_bytes: 0, allocations: HashMap::new() });
    }
    let state = guard.as_mut()?;
    // 晋升 release_pending 挂的已完成 completion（懒惰式，与池时序一致）。
    for segment in &mut state.segments {
        segment.arena.promote();
    }
    let started = super::device_buffer::hip_api_stats::start();
    for (index, segment) in state.segments.iter_mut().enumerate() {
        if let Some(allocation) = segment.arena.allocate(bytes as u64) {
            super::device_buffer::hip_api_stats::counted(super::device_buffer::hip_api_stats::ARENA_ALLOC, started);
            let pointer = (segment.base + allocation.offset_bytes as usize) as *mut c_void;
            state.allocations.insert(pointer as usize, (index as u32, allocation.node));
            mem_trace("AA", device_id, bytes, pointer);
            return Some((pointer, allocation.bytes as usize));
        }
    }
    let total_bytes = state.total_bytes;
    let Some(mut segment) = expand_segment(device_id, bytes, total_bytes) else {
        super::device_buffer::hip_api_stats::counted(super::device_buffer::hip_api_stats::ARENA_FULL, started);
        return None;
    };
    let index = state.segments.len() as u32;
    let base = segment.base;
    let allocation = segment.arena.allocate(bytes as u64);
    state.total_bytes += segment.bytes;
    state.segments.push(segment);
    super::device_buffer::hip_api_stats::counted(super::device_buffer::hip_api_stats::ARENA_ALLOC, started);
    let Some(allocation) = allocation else {
        // 新段刚建立必然能容纳请求；此处为防御分支，段仍保留供后续使用。
        return None;
    };
    let pointer = (base + allocation.offset_bytes as usize) as *mut c_void;
    state.allocations.insert(pointer as usize, (index, allocation.node));
    mem_trace("AA", device_id, bytes, pointer);
    Some((pointer, allocation.bytes as usize))
}

/// 淘汰/OOM 释放分派：arena 出身块还回 arena（free_ready——只作用于已
/// promote 完成的块，无需 event），返回 true；驱动出身块返回 false，由
/// 调用方继续 hipFree。
pub(super) fn release(device_id: i32, pointer: *mut c_void) -> bool {
    let Some(arenas) = device_arenas() else { return false };
    let Ok(device_index) = usize::try_from(device_id) else { return false };
    let Some(slot) = arenas.get(device_index) else { return false };
    let mut guard = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(state) = guard.as_mut() else { return false };
    let Some((index, node)) = state.allocations.remove(&(pointer as usize)) else { return false };
    let Some(segment) = state.segments.get_mut(index as usize) else { return false };
    segment.arena.free_ready(node);
    super::device_buffer::hip_api_stats::counted(super::device_buffer::hip_api_stats::ARENA_RELEASE, super::device_buffer::hip_api_stats::start());
    mem_trace("AR", device_id, 0, pointer);
    true
}

/// Drop 兜底/池关闭路径的归还分派：arena 出身块挂 event 进 pending
/// （设备完成前不复用）。**返回 false 仅表示"非 arena 块"（hipFree 安全）；
/// arena 块在 event 录制失败等异常下主动泄漏并返回 true——hipFree 段内
/// offset 指针会静默释放整个段（2026-09-14 探针实证），泄漏远好过炸段。**
pub(super) fn release_pending(device_id: i32, pointer: *mut c_void) -> bool {
    let Some(arenas) = device_arenas() else { return false };
    let Ok(device_index) = usize::try_from(device_id) else { return false };
    let Some(slot) = arenas.get(device_index) else { return false };
    {
        let guard = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(state) = guard.as_ref() else { return false };
        if !state.allocations.contains_key(&(pointer as usize)) {
            return false;
        }
    }
    // 已知是 arena 块：以下任何失败都走"泄漏防炸"分支（return true）。
    let leak = |reason: &str| {
        super::device_buffer::hip_api_stats::counted(super::device_buffer::hip_api_stats::ARENA_LEAK, super::device_buffer::hip_api_stats::start());
        eprintln!("[rocm-arena] device={device_id} ptr={pointer:p} 归还失败（{reason}），主动泄漏防炸段");
        true
    };
    if set_device(device_id).is_err() {
        return leak("set_device");
    }
    let Ok(runtime) = RocmRuntime::open() else {
        return leak("runtime");
    };
    let Ok(create) = runtime.event_create() else {
        return leak("event_create");
    };
    let Ok(record) = runtime.event_record() else {
        return leak("event_record");
    };
    let mut event = super::device_buffer::device_buffer_pool(device_id).and_then(|pool| pool.lock().ok()?.available_events.pop()).map(|event| event as HipEvent).unwrap_or(ptr::null_mut());
    if event.is_null() && unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) } != HIP_SUCCESS {
        return leak("event_create_with_flags");
    }
    if unsafe { record(event, active_compute_stream()) } != HIP_SUCCESS {
        if let Ok(destroy) = runtime.event_destroy() {
            let _ = unsafe { destroy(event) };
        }
        return leak("event_record_call");
    }
    let mut guard = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(state) = guard.as_mut() else {
        if let Ok(destroy) = runtime.event_destroy() {
            let _ = unsafe { destroy(event) };
        }
        return leak("arena_vacant");
    };
    let Some((index, node)) = state.allocations.remove(&(pointer as usize)) else {
        // 并发下已被其他路径摘除：按已处理论，销毁 event 后返回 true。
        if let Ok(destroy) = runtime.event_destroy() {
            let _ = unsafe { destroy(event) };
        }
        return true;
    };
    let Some(segment) = state.segments.get_mut(index as usize) else {
        if let Ok(destroy) = runtime.event_destroy() {
            let _ = unsafe { destroy(event) };
        }
        return leak("segment_missing");
    };
    segment.arena.free(node, RocmCompletion { device_id, event: event as usize });
    super::device_buffer::hip_api_stats::counted(super::device_buffer::hip_api_stats::ARENA_PENDING, super::device_buffer::hip_api_stats::start());
    mem_trace("AP", device_id, 0, pointer);
    true
}

/// 碎片水位诊断（largestFreeRegion 等），供 memory_diagnostics 调用。
/// 多段版本：汇总各段的 ready/pending/used，largest_free 取各段最大值。
#[allow(dead_code)]
pub(super) fn storage_report(device_id: i32) -> Option<String> {
    let arenas = device_arenas()?;
    let slot = arenas.get(usize::try_from(device_id).ok()?)?;
    let guard = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let state = guard.as_ref()?;
    let mut ready = 0;
    let mut pending = 0;
    let mut used = 0;
    let mut largest_free = 0;
    let mut regions = 0;
    let mut allocate_calls = 0;
    let mut splits = 0;
    let mut coalesces = 0;
    let mut promotes = 0;
    for segment in &state.segments {
        let report = segment.arena.storage_report();
        let stats = segment.arena.stats();
        ready += report.ready_bytes;
        pending += report.pending_bytes;
        used += report.used_bytes;
        largest_free = largest_free.max(report.largest_free_region_bytes);
        regions += report.ready_regions;
        allocate_calls += stats.allocate_calls;
        splits += stats.splits;
        coalesces += stats.coalesces;
        promotes += stats.promotes;
    }
    Some(format!(
        "arena(segments={} total={}MiB ready={}MiB pending={}MiB used={}MiB largest_free={}MiB regions={} alloc={} splits={} coalesce={} promotes={})",
        state.segments.len(),
        state.total_bytes >> 20,
        ready >> 20,
        pending >> 20,
        used >> 20,
        largest_free >> 20,
        regions,
        allocate_calls,
        splits,
        coalesces,
        promotes,
    ))
}
