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
