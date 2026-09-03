//! 固定地址 HIP Graph：只在录制期拦截 kernel launch，重放期只提交一次 graph。
//!
//! 这里故意不提供逐节点 `SetParams`。调用方必须持有 graph 内所有输入、输出、
//! workspace 与参数 buffer，并保证它们在 `StaticHipGraph` 析构前地址不变。
//! 动态数据应在 graph 外写入固定 buffer，或由 kernel 从固定 device parameter
//! buffer 读取。

use super::*;
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    /// 只在当前 stage 提交线程录制；重放不进入 TLS，也不会重新执行 Rust 调度。
    static GRAPH_RECORDING: RefCell<Option<GraphRecording>> = const { RefCell::new(None) };
}

struct GraphRecording {
    graph: *mut c_void,
    prev: *mut c_void,
    device_id: i32,
    stream: *mut c_void,
    nodes: usize,
    invalid_reason: Option<String>,
    launch_error: Option<HipError>,
}

/// 录制 guard。未显式 `finish` 时自动销毁模板，避免错误路径把 TLS 留脏。
pub(crate) struct StaticGraphRecorder {
    active: bool,
    /// HIP current device 与 active stream 都是线程局部状态，录制 guard 不可跨线程。
    _thread_bound: PhantomData<Rc<()>>,
}

/// 已实例化的固定地址 graph。它不保存 node 句柄，也不存在逐节点参数更新路径。
pub(crate) struct StaticHipGraph {
    exec: *mut c_void,
    device_id: i32,
    stream: *mut c_void,
    nodes: usize,
}

unsafe impl Send for StaticHipGraph {}

impl StaticHipGraph {
    pub(crate) fn node_count(&self) -> usize {
        self.nodes
    }

    /// 重放只允许原 device/original stream，参数与 tensor 地址由 owner 保证固定。
    pub(crate) fn launch(&self, device_id: i32, stream: *mut c_void) -> Result<(), String> {
        if device_id != self.device_id || stream != self.stream {
            return Err(format!("HIP graph owner 不匹配: device={device_id}/{} stream={stream:p}/{:p}", self.device_id, self.stream));
        }
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let launch = runtime.graph_launch()?;
        let started = super::hip_api_stats::start();
        let status = unsafe { launch(self.exec, stream) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipGraphLaunch fixed-address graph"));
        }
        Ok(())
    }
}

impl Drop for StaticHipGraph {
    fn drop(&mut self) {
        if self.exec.is_null() || set_device(self.device_id).is_err() {
            return;
        }
        if let Ok(runtime) = RocmRuntime::open()
            && let Ok(destroy) = runtime.graph_exec_destroy()
        {
            let _ = unsafe { destroy(self.exec) };
        }
        self.exec = ptr::null_mut();
    }
}

impl StaticGraphRecorder {
    pub(crate) fn begin(device_id: i32, stream: *mut c_void) -> Result<Self, String> {
        set_device(device_id)?;
        let nested = GRAPH_RECORDING.with(|recording| recording.borrow().is_some());
        if nested {
            return Err("HIP graph 不允许嵌套录制".to_owned());
        }
        let runtime = RocmRuntime::open()?;
        let create = runtime.graph_create()?;
        let mut graph = ptr::null_mut();
        let status = unsafe { create(&mut graph, 0) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipGraphCreate fixed-address graph"));
        }
        GRAPH_RECORDING.with(|recording| {
            *recording.borrow_mut() = Some(GraphRecording { graph, prev: ptr::null_mut(), device_id, stream, nodes: 0, invalid_reason: None, launch_error: None });
        });
        Ok(Self { active: true, _thread_bound: PhantomData })
    }

    /// `Ok(None)` 只表示片段没有 kernel 节点；非法 stream 操作直接返回错误。
    pub(crate) fn finish(mut self) -> Result<Option<StaticHipGraph>, String> {
        let recording = take_recording()?;
        self.active = false;
        let runtime = RocmRuntime::open()?;
        let destroy = runtime.graph_destroy()?;
        if let Some(reason) = recording.invalid_reason {
            let _ = unsafe { destroy(recording.graph) };
            return Err(format!("HIP graph 录制失效: {reason}"));
        }
        if let Some(status) = recording.launch_error {
            let _ = unsafe { destroy(recording.graph) };
            return Err(runtime.hip_error(status, "hipGraphAddKernelNode fixed-address graph"));
        }
        if recording.nodes == 0 {
            let _ = unsafe { destroy(recording.graph) };
            return Ok(None);
        }
        let instantiate = runtime.graph_instantiate_with_flags()?;
        let mut exec = ptr::null_mut();
        let status = unsafe { instantiate(&mut exec, recording.graph, 0) };
        let _ = unsafe { destroy(recording.graph) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipGraphInstantiateWithFlags fixed-address graph"));
        }
        Ok(Some(StaticHipGraph { exec, device_id: recording.device_id, stream: recording.stream, nodes: recording.nodes }))
    }
}

impl Drop for StaticGraphRecorder {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let recording = GRAPH_RECORDING.with(|recording| recording.borrow_mut().take());
        if let Some(recording) = recording
            && let Ok(runtime) = RocmRuntime::open()
            && let Ok(destroy) = runtime.graph_destroy()
        {
            let _ = unsafe { destroy(recording.graph) };
        }
    }
}

fn take_recording() -> Result<GraphRecording, String> {
    GRAPH_RECORDING.with(|recording| recording.borrow_mut().take().ok_or_else(|| "HIP graph 当前没有进行中的录制".to_owned()))
}

/// 录制段若出现 memcpy 或其他 stream 数据操作，调用方必须在操作前调用本函数。
/// 该轮模板会被放弃；数据操作仍可正常执行，随后由调用方 eager 重跑 kernel 片段。
#[track_caller]
pub(crate) fn graph_foreign_stream_op() {
    let caller = std::panic::Location::caller();
    GRAPH_RECORDING.with(|recording| {
        if let Some(recording) = recording.borrow_mut().as_mut() {
            recording.invalid_reason.get_or_insert_with(|| format!("stream 数据操作 {}:{}", caller.file(), caller.line()));
        }
    });
}

/// kernel launch 的统一入口：录制期构造严格线性依赖，其他时间直接 eager launch。
pub(crate) unsafe extern "C" fn kernel_launch_trampoline(
    function: *mut c_void,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared: u32,
    stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> HipError {
    const LAUNCH_FAILED: HipError = 1;
    let building = GRAPH_RECORDING.with(|recording| recording.borrow().is_some());
    let runtime = match RocmRuntime::open() {
        Ok(runtime) => runtime,
        Err(_) => return LAUNCH_FAILED,
    };
    if !building {
        let launch = match runtime.module_launch() {
            Ok(launch) => launch,
            Err(_) => return LAUNCH_FAILED,
        };
        let started = std::time::Instant::now();
        let status = unsafe { launch(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared, stream, kernel_params, extra) };
        let elapsed = started.elapsed();
        if elapsed >= std::time::Duration::from_millis(100) {
            eprintln!(
                "[rocm-kernel-launch-slow] device={} function={function:p} grid={grid_x}x{grid_y}x{grid_z} block={block_x}x{block_y}x{block_z} shared={shared} stream={stream:p} wall_ms={:.3}",
                ACTIVE_DEVICE.get(),
                elapsed.as_secs_f64() * 1000.0,
            );
        }
        return status;
    }

    let params = HipKernelNodeParams { func: function, grid_dim_x: grid_x, grid_dim_y: grid_y, grid_dim_z: grid_z, block_dim_x: block_x, block_dim_y: block_y, block_dim_z: block_z, shared_mem_bytes: shared, kernel_params, extra };
    let (graph, prev, expected_stream) = GRAPH_RECORDING.with(|recording| {
        let recording = recording.borrow();
        let recording = recording.as_ref().expect("building 已检查");
        (recording.graph, recording.prev, recording.stream)
    });
    if stream != expected_stream {
        GRAPH_RECORDING.with(|recording| {
            recording.borrow_mut().as_mut().expect("building 已检查").invalid_reason.get_or_insert_with(|| format!("kernel stream={stream:p} 与 owner stream={expected_stream:p} 不一致"));
        });
        return HIP_SUCCESS;
    }
    let add_node = match runtime.graph_add_kernel_node() {
        Ok(add_node) => add_node,
        Err(_) => return LAUNCH_FAILED,
    };
    let mut node = ptr::null_mut();
    let deps = if prev.is_null() { ptr::null() } else { &prev };
    let dep_count = usize::from(!prev.is_null());
    let status = unsafe { add_node(&mut node, graph, deps, dep_count, &params) };
    GRAPH_RECORDING.with(|recording| {
        let mut recording = recording.borrow_mut();
        let recording = recording.as_mut().expect("building 已检查");
        if status == HIP_SUCCESS {
            recording.prev = node;
            recording.nodes += 1;
        } else {
            recording.launch_error.get_or_insert(status);
        }
    });
    // 把建图错误延迟到 finish，保证上层能走统一 eager fallback/报错路径。
    HIP_SUCCESS
}

#[cfg(test)]
mod tests {
    /// go/no-go 探针：decode 单行层链的 eager 连续 launch vs 静态 graph replay
    /// 在设备时间线上的间隙差。运行：
    /// `cargo test --release --features with-rocm static_graph_decode_chain_gap_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn static_graph_decode_chain_gap_probe() {
        use super::super::tensor::try_add_resident_f32;
        use super::super::*;
        use super::StaticGraphRecorder;
        let device_id = 0;
        set_device(device_id).expect("set device");
        let elements = 64 * 1024;
        let left = DeviceBuffer::upload_f32(device_id, &vec![1.0f32; elements]).expect("upload left");
        let right = DeviceBuffer::upload_f32(device_id, &vec![2.0f32; elements]).expect("upload right");
        const CHAIN: usize = 64;
        const REPLAYS: usize = 200;
        let elapsed_ms = |label: &str, run: &mut dyn FnMut()| {
            // 用流上 event 对测纯设备时间线时长（含 kernel 间隙）。
            let runtime = RocmRuntime::open().expect("runtime");
            let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0").expect("event create symbol");
            let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0").expect("event record symbol");
            let sync: Symbol<HipEventSynchronize> = runtime.symbol(&runtime.hip, b"hipEventSynchronize\0").expect("event sync symbol");
            let elapsed: Symbol<HipEventElapsedTime> = runtime.symbol(&runtime.hip, b"hipEventElapsedTime\0").expect("event elapsed symbol");
            let mut begin = ptr::null_mut();
            let mut end = ptr::null_mut();
            unsafe {
                assert_eq!(create(&mut begin, 0), HIP_SUCCESS);
                assert_eq!(create(&mut end, 0), HIP_SUCCESS);
                assert_eq!(record(begin, active_compute_stream()), HIP_SUCCESS);
            }
            run();
            unsafe {
                assert_eq!(record(end, active_compute_stream()), HIP_SUCCESS);
                assert_eq!(sync(end), HIP_SUCCESS);
                let mut ms = 0.0f32;
                assert_eq!(elapsed(&mut ms, begin, end), HIP_SUCCESS);
                eprintln!("[graph-gap-probe] {label} device_ms={ms:.3}");
            }
        };
        // 预热（含 hipMalloc 池与模块加载）
        let mut out = try_add_resident_f32(device_id, &left, &right, elements, 1.0).expect("warm");
        let mut eager_total = 0.0f32;
        for _ in 0..3 {
            elapsed_ms("eager x64", &mut || {
                for _ in 0..CHAIN {
                    out = try_add_resident_f32(device_id, &left, &right, elements, 1.0).expect("eager add");
                }
            });
            eager_total += 1.0;
        }
        drop(out);
        let _ = eager_total;
        let stream = active_compute_stream();
        let graph = {
            let recorder = StaticGraphRecorder::begin(device_id, stream).expect("begin recording");
            let mut out = try_add_resident_f32(device_id, &left, &right, elements, 1.0).expect("record add");
            for _ in 1..CHAIN {
                out = try_add_resident_f32(device_id, &out, &right, elements, 1.0).expect("record add");
            }
            let graph = recorder.finish().expect("finish").expect("graph 非空");
            std::mem::forget(out);
            graph
        };
        eprintln!("[graph-gap-probe] nodes={}", graph.node_count());
        for _ in 0..3 {
            elapsed_ms("graph x64", &mut || {
                graph.launch(device_id, stream).expect("graph replay");
            });
        }
        elapsed_ms("graph x64 x200 replays", &mut || {
            for _ in 0..REPLAYS {
                graph.launch(device_id, stream).expect("graph replay loop");
            }
        });
    }
}

#[cfg(test)]
mod p2p_tests {
    /// P2P 握手延迟探针：pair decode 每层 4-6 次跨卡握手的单次成本标定。
    /// `cargo test --release --features with-rocm p2p_handshake_latency_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn p2p_handshake_latency_probe() {
        use super::super::*;
        use std::time::Instant;
        let runtime = RocmRuntime::open().expect("runtime");
        let create: Symbol<HipEventCreateWithFlags> = runtime.symbol(&runtime.hip, b"hipEventCreateWithFlags\0").expect("create");
        let record: Symbol<HipEventRecord> = runtime.symbol(&runtime.hip, b"hipEventRecord\0").expect("record");
        let wait: Symbol<HipStreamWaitEvent> = runtime.symbol(&runtime.hip, b"hipStreamWaitEvent\0").expect("wait");
        let sync: Symbol<HipEventSynchronize> = runtime.symbol(&runtime.hip, b"hipEventSynchronize\0").expect("sync");
        let elapsed: Symbol<HipEventElapsedTime> = runtime.symbol(&runtime.hip, b"hipEventElapsedTime\0").expect("elapsed");
        let copy: Symbol<HipMemcpyPeerAsync> = runtime.symbol(&runtime.hip, b"hipMemcpyPeerAsync\0").expect("peer copy");
        let legacy = ptr::null_mut::<c_void>();
        for (owner, peer) in [(0, 1), (2, 3), (4, 5), (6, 7)] {
            enable_peer_access(peer, owner).expect("peer access owner->peer");
            enable_peer_access(owner, peer).expect("peer access peer->owner");
            let src = DeviceBuffer::allocate(owner, 1024 * 1024).expect("src");
            let dst = DeviceBuffer::allocate(peer, 1024 * 1024).expect("dst");
            let mut begin = ptr::null_mut();
            let mut end = ptr::null_mut();
            let mut owner_ready = ptr::null_mut();
            let mut peer_done = ptr::null_mut();
            unsafe {
                set_device(owner).expect("dev0");
                assert_eq!(create(&mut begin, 0), HIP_SUCCESS);
                assert_eq!(create(&mut end, 0), HIP_SUCCESS);
                assert_eq!(create(&mut owner_ready, HIP_EVENT_DISABLE_TIMING), HIP_SUCCESS);
                set_device(peer).expect("peer device");
                assert_eq!(create(&mut peer_done, HIP_EVENT_DISABLE_TIMING), HIP_SUCCESS);
                for &(bytes, label) in &[(4 * 1024usize, "4KB"), (132 * 1024, "132KB"), (1024 * 1024, "1MB")] {
                    // 两个 device 各自 record 本地 event；跨卡只 wait，不能跨 device
                    // 重新 record 同一个 event。
                    let started = Instant::now();
                    const ROUNDS: usize = 200;
                    set_device(owner).expect("owner device");
                    assert_eq!(record(begin, legacy), HIP_SUCCESS);
                    for _ in 0..ROUNDS {
                        assert_eq!(record(owner_ready, legacy), HIP_SUCCESS);
                        set_device(peer).expect("peer device");
                        assert_eq!(wait(legacy, owner_ready, 0), HIP_SUCCESS);
                        assert_eq!(copy(dst.pointer, peer, src.pointer, owner, bytes, legacy), HIP_SUCCESS);
                        assert_eq!(record(peer_done, legacy), HIP_SUCCESS);
                        set_device(owner).expect("owner device");
                        assert_eq!(wait(legacy, peer_done, 0), HIP_SUCCESS);
                    }
                    assert_eq!(record(end, legacy), HIP_SUCCESS);
                    assert_eq!(sync(end), HIP_SUCCESS);
                    let mut ms = 0.0f32;
                    assert_eq!(elapsed(&mut ms, begin, end), HIP_SUCCESS);
                    eprintln!("[p2p-probe] pair={owner}/{peer} {label} roundtrip avg_us={:.1} wall_us={:.1}", ms * 1000.0 / ROUNDS as f32, started.elapsed().as_secs_f64() * 1e6 / ROUNDS as f64);
                }
            }
        }
    }
}
