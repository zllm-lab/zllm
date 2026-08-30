//! HIP graph 模板重放:单行 decode 热路径的 launch 整段重放。
//!
//! Record 阶段把 `kernel_launch` 吞下的每个 launch 当场转成
//! `hipGraphAddKernelNode` 建严格线性链(参数在调用栈存活期内被 HIP 深拷贝);
//! Replay 阶段 Rust 代码照常构造参数,每个 launch 用
//! `hipGraphExecKernelNodeSetParams` 原位更新参数与网格,末尾单次
//! `hipGraphLaunch`。收益来自把每 stage 上百次 launch 的 host 提交与
//! kernel 间 gap 压成一次 graph launch。
//!
//! 一致性前提(由调用方与检测共同保证):
//! - 同一形状 key 的 launch 序列确定(相同 kernel、相同数量、相同顺序);
//! - 段内不得出现 memcpy 等数据面 stream 操作(检测到即作废本 session);
//! - 中间 tensor 地址经复用池在每次 replay 时返回相同地址序列。
//! Record 遍不执行任何 kernel,调用方必须回退 KV/DSA 行与 aux 状态后重放。

use super::*;

thread_local! {
    /// 当前线程的 graph 执行段状态。Record/Replay 只在 stage 执行线程内
    /// 开启,trampoline 与 foreign-op 检测都读它。
    static GRAPH_RUN: RefCell<Option<GraphRun>> = const { RefCell::new(None) };
}

enum GraphRun {
    Building {
        graph: *mut c_void,
        prev: *mut c_void,
        nodes: usize,
        foreign: bool,
    },
    Replaying {
        exec: *mut c_void,
        nodes: Vec<*mut c_void>,
        /// 下一个待更新位点;usize::MAX 表示未开始本轮 replay。
        next: usize,
        stream: *mut c_void,
        foreign: bool,
    },
}

/// Record/Replay 期间出现数据面 stream 操作(memcpy 等)时置脏。资源与
/// 同步类操作(malloc/event)不破坏 kernel 参数一致性,不算 foreign。
pub(crate) fn graph_foreign_stream_op() {
    GRAPH_RUN.with(|run| {
        if let Some(run) = run.borrow_mut().as_mut() {
            match run {
                GraphRun::Building { foreign, .. } => *foreign = true,
                GraphRun::Replaying { foreign, .. } => *foreign = true,
            }
        }
    });
}

/// 开始 Record:创建空图并进入 Building。
pub(crate) fn graph_begin_record() -> Result<(), String> {
    let runtime = RocmRuntime::open()?;
    let create = runtime.graph_create()?;
    let mut graph = ptr::null_mut();
    // flags=0,与 cudaGraphCreate 语义一致。
    let status = unsafe { create(&mut graph, 0) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipGraphCreate"));
    }
    GRAPH_RUN.with(|run| {
        *run.borrow_mut() = Some(GraphRun::Building { graph, prev: ptr::null_mut(), nodes: 0, foreign: false });
    });
    Ok(())
}

/// 结束 Record:实例化 graph exec。段内出现过数据面 stream 操作时放弃
/// (返回 Ok(false),调用方按 eager 语义回退状态重跑)。
pub(crate) fn graph_finish_record(stream: *mut c_void) -> Result<bool, String> {
    let building = GRAPH_RUN.with(|run| match run.borrow_mut().take() {
        Some(run @ GraphRun::Building { .. }) => Some(run),
        other => {
            *run.borrow_mut() = other;
            None
        }
    });
    let Some(GraphRun::Building { graph, nodes, foreign, .. }) = building else {
        return Err("graph_finish_record: 当前没有进行中的 Record".to_owned());
    };
    let runtime = RocmRuntime::open()?;
    if foreign || nodes == 0 {
        let destroy = runtime.graph_destroy()?;
        let _ = unsafe { destroy(graph) };
        return Ok(false);
    }
    let instantiate = runtime.graph_instantiate_with_flags()?;
    let mut exec = ptr::null_mut();
    let status = unsafe { instantiate(&mut exec, graph, 0) };
    if status != HIP_SUCCESS {
        let destroy = runtime.graph_destroy()?;
        let _ = unsafe { destroy(graph) };
        return Err(runtime.hip_error(status, "hipGraphInstantiateWithFlags"));
    }
    // 线性依赖链把节点序固定为提交序;取回句柄供 replay 逐位点 SetParams。
    let get_nodes = runtime.graph_get_nodes()?;
    let mut handles = vec![ptr::null_mut(); nodes];
    let mut written = 0usize;
    let status = unsafe { get_nodes(graph, handles.as_mut_ptr(), &mut written) };
    let get_nodes_failed = status != HIP_SUCCESS || written != nodes;
    if get_nodes_failed {
        let destroy_exec = runtime.graph_exec_destroy()?;
        let _ = unsafe { destroy_exec(exec) };
    }
    if status != HIP_SUCCESS {
        let destroy = runtime.graph_destroy()?;
        let _ = unsafe { destroy(graph) };
        return Err(runtime.hip_error(status, "hipGraphGetNodes"));
    }
    if written != nodes {
        let destroy = runtime.graph_destroy()?;
        let _ = unsafe { destroy(graph) };
        return Err(format!("graph 节点数不匹配: 写出 {written} / 记录 {nodes}"));
    }
    // exec 独立持有执行状态,原始 graph 句柄可以释放。
    let destroy = runtime.graph_destroy()?;
    let _ = unsafe { destroy(graph) };
    GRAPH_RUN.with(|run| {
        *run.borrow_mut() = Some(GraphRun::Replaying { exec, nodes: handles, next: usize::MAX, stream, foreign: false });
    });
    Ok(true)
}

/// 开始一次 Replay:重置位点计数。
pub(crate) fn graph_begin_replay() -> Result<(), String> {
    GRAPH_RUN.with(|run| {
        if let Some(GraphRun::Replaying { next, foreign, .. }) = run.borrow_mut().as_mut() {
            *next = 0;
            *foreign = false;
            Ok(())
        } else {
            Err("graph_begin_replay: 没有已实例化的 session".to_owned())
        }
    })
}

/// 结束一次 Replay:校验位点数一致后单次 graphLaunch。
/// 位点数不匹配或段内出现数据面 stream 操作时返回 Err,调用方必须回退
/// 状态并按 eager 重跑本 token。
pub(crate) fn graph_finish_replay() -> Result<(), String> {
    let runtime = RocmRuntime::open()?;
    GRAPH_RUN.with(|run| {
        let mut current = run.borrow_mut();
        let Some(GraphRun::Replaying { exec, nodes, next, stream, foreign }) = current.as_mut() else {
            return Err("graph_finish_replay: 没有进行中的 Replay".to_owned());
        };
        if *foreign {
            return Err("graph replay 段内出现数据面 stream 操作".to_owned());
        }
        if *next != nodes.len() {
            return Err(format!("graph replay 位点不匹配: 提交 {} / 捕获 {}", next, nodes.len()));
        }
        let launch = runtime.graph_launch()?;
        let status = unsafe { launch(*exec, *stream) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipGraphLaunch"));
        }
        *next = usize::MAX;
        Ok(())
    })
}

/// 丢弃当前 session,之后所有 launch 回到 eager。
pub(crate) fn graph_abandon() {
    GRAPH_RUN.with(|run| {
        if let Some(GraphRun::Replaying { exec, .. }) = run.borrow_mut().take() {
            if let Ok(runtime) = RocmRuntime::open() {
                if let Ok(destroy_exec) = runtime.graph_exec_destroy() {
                    let _ = unsafe { destroy_exec(exec) };
                }
            }
        }
    });
}

/// 当前线程是否已具备可 replay 的 session。
pub(crate) fn graph_ready() -> bool {
    GRAPH_RUN.with(|run| matches!(run.borrow().as_ref(), Some(GraphRun::Replaying { .. })))
}

/// kernel launch 的统一入口:graph 段内改写为建节点/参数更新,其余直发。
/// 签名与 `HipModuleLaunchKernel` 完全一致,现有 kernel 代码把
/// `runtime.module_launch()` 的绑定换成本 trampoline 即可。
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
    // hipErrorInvalidValue;仅在未来到不可恢复状态时出现。
    const LAUNCH_FAILED: HipError = 1;
    enum Phase {
        Eager,
        Building,
        Replaying,
    }
    let phase = GRAPH_RUN.with(|run| match run.borrow().as_ref() {
        None => Phase::Eager,
        Some(GraphRun::Building { .. }) => Phase::Building,
        Some(GraphRun::Replaying { .. }) => Phase::Replaying,
    });
    let runtime = match RocmRuntime::open() {
        Ok(runtime) => runtime,
        Err(_) => return LAUNCH_FAILED,
    };
    let params = HipKernelNodeParams { func: function, grid_dim_x: grid_x, grid_dim_y: grid_y, grid_dim_z: grid_z, block_dim_x: block_x, block_dim_y: block_y, block_dim_z: block_z, shared_mem_bytes: shared, kernel_params, extra };
    match phase {
        Phase::Eager => {
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
            status
        }
        Phase::Building => {
            let add_node = match runtime.graph_add_kernel_node() {
                Ok(symbol) => symbol,
                Err(_) => return LAUNCH_FAILED,
            };
            // 先取出 prev 再调 HIP,避免跨 FFI 持有 RefCell 借用。
            let (graph, prev) = GRAPH_RUN.with(|run| {
                let current = run.borrow();
                let Some(GraphRun::Building { graph, prev, .. }) = current.as_ref() else { unreachable!() };
                (*graph, *prev)
            });
            let mut node = ptr::null_mut();
            let deps = if prev.is_null() { ptr::null() } else { &prev };
            let dep_count = if prev.is_null() { 0 } else { 1 };
            // HIP 在本调用内按 func 元数据深拷贝 kernelParams 指向的参数值,
            // 栈上参数此刻仍然存活。
            let status = unsafe { add_node(&mut node, graph, deps, dep_count, &params) };
            if status != HIP_SUCCESS {
                return status;
            }
            GRAPH_RUN.with(|run| {
                if let Some(GraphRun::Building { prev, nodes, .. }) = run.borrow_mut().as_mut() {
                    *prev = node;
                    *nodes += 1;
                }
            });
            HIP_SUCCESS
        }
        Phase::Replaying => {
            let node = GRAPH_RUN.with(|run| {
                let mut current = run.borrow_mut();
                let Some(GraphRun::Replaying { nodes, next, .. }) = current.as_mut() else { unreachable!() };
                let slot = *next;
                if slot < nodes.len() {
                    *next += 1;
                    Some(nodes[slot])
                } else {
                    None
                }
            });
            let Some(node) = node else {
                // 本轮提交数超过捕获数:序列已分叉。位点未推进到末尾,
                // finish_replay 会因不匹配拒绝 launch。
                return LAUNCH_FAILED;
            };
            let set_params = match runtime.graph_exec_kernel_node_set_params() {
                Ok(symbol) => symbol,
                Err(_) => return LAUNCH_FAILED,
            };
            let exec = GRAPH_RUN.with(|run| {
                let current = run.borrow();
                let Some(GraphRun::Replaying { exec, .. }) = current.as_ref() else { unreachable!() };
                *exec
            });
            unsafe { set_params(exec, node, &params) }
        }
    }
}
