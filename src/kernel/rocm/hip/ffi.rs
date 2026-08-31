use super::*;

use libloading::{Library, Symbol};

pub(super) const HIP_LIBRARIES: &[&str] = &["libamdhip64.so", "libamdhip64.so.5", "libamdhip64.so.6"];
pub(super) const HIPRTC_LIBRARIES: &[&str] = &["libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so.5"];

pub(super) const HIP_SUCCESS: i32 = 0;
pub(super) const HIP_ERROR_NOT_READY: i32 = 600;
pub(super) const HIP_EVENT_DISABLE_TIMING: u32 = 0x2;
pub(super) const HIP_STREAM_NON_BLOCKING: u32 = 0x1;
pub(super) const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: i32 = 704;
pub(super) const HIP_MEMORY_COPY_HOST_TO_DEVICE: i32 = 1;
pub(super) const HIP_MEMORY_COPY_DEVICE_TO_HOST: i32 = 2;
pub(super) const HIP_MEMORY_COPY_DEVICE_TO_DEVICE: i32 = 3;
pub(super) const HIP_HOST_MALLOC_DEFAULT: u32 = 0;
pub(super) const H2D_STAGING_BYTES: usize = 16 * 1024 * 1024;
pub(super) const H2D_PROBE_BYTES: usize = 1024 * 1024;
pub(super) const H2D_DIRECT_MIN_GIB_PER_SECOND: f64 = 10.0;
pub(super) const H2D_MODE_AUTO: u8 = 0;
pub(super) const H2D_MODE_DIRECT: u8 = 1;
pub(super) const H2D_MODE_PINNED: u8 = 2;

pub(super) static H2D_UPLOAD_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(H2D_MODE_AUTO);
pub(super) static HIP_DEVICE_COUNT: OnceLock<Result<i32, String>> = OnceLock::new();
pub(super) static ENABLED_PEER_ACCESS: OnceLock<Mutex<HashMap<(i32, i32), ()>>> = OnceLock::new();

#[allow(clippy::upper_case_acronyms)]
pub(super) type HipError = i32;

pub(super) type HipInit = unsafe extern "C" fn(u32) -> HipError;
pub(super) type HipGetDeviceCount = unsafe extern "C" fn(*mut i32) -> HipError;
pub(super) type HipGetDevice = unsafe extern "C" fn(*mut i32) -> HipError;
pub(super) type HipSetDevice = unsafe extern "C" fn(i32) -> HipError;
pub(super) type HipGetErrorString = unsafe extern "C" fn(HipError) -> *const c_char;
pub(super) type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> HipError;
pub(super) type HipFree = unsafe extern "C" fn(*mut c_void) -> HipError;
pub(super) type HipMallocAsync = unsafe extern "C" fn(*mut *mut c_void, usize, *mut c_void) -> HipError;
pub(super) type HipFreeAsync = unsafe extern "C" fn(*mut c_void, *mut c_void) -> HipError;
pub(super) type HipMemPool = *mut c_void;
pub(super) type HipDeviceGetDefaultMemPool = unsafe extern "C" fn(*mut HipMemPool, i32) -> HipError;
pub(super) type HipMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> HipError;
pub(super) type HipMemPoolTrimTo = unsafe extern "C" fn(HipMemPool, usize) -> HipError;
pub(super) type HipHostMalloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> HipError;
pub(super) type HipHostFree = unsafe extern "C" fn(*mut c_void) -> HipError;
pub(super) type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> HipError;
pub(super) type HipMemcpyAsync = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, *mut c_void) -> HipError;
pub(super) type HipMemcpyPeer = unsafe extern "C" fn(*mut c_void, i32, *const c_void, i32, usize) -> HipError;
pub(super) type HipMemcpyPeerAsync = unsafe extern "C" fn(*mut c_void, i32, *const c_void, i32, usize, *mut c_void) -> HipError;
pub(super) type HipStreamCreateWithFlags = unsafe extern "C" fn(*mut *mut c_void, u32) -> HipError;
pub(super) type HipStreamCreateWithPriority = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> HipError;
pub(super) type HipDeviceGetStreamPriorityRange = unsafe extern "C" fn(*mut i32, *mut i32) -> HipError;
pub(super) type HipStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> HipError;
pub(super) type HipStreamDestroy = unsafe extern "C" fn(*mut c_void) -> HipError;
pub(super) type HipDeviceCanAccessPeer = unsafe extern "C" fn(*mut i32, i32, i32) -> HipError;
pub(super) type HipDeviceEnablePeerAccess = unsafe extern "C" fn(i32, u32) -> HipError;
pub(super) type HipDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> HipError;
pub(super) type HipDeviceSynchronize = unsafe extern "C" fn() -> HipError;
pub(super) type HipEvent = *mut c_void;
pub(super) type HipEventCreateWithFlags = unsafe extern "C" fn(*mut HipEvent, u32) -> HipError;
pub(super) type HipEventRecord = unsafe extern "C" fn(HipEvent, *mut c_void) -> HipError;
pub(super) type HipStreamWaitEvent = unsafe extern "C" fn(*mut c_void, HipEvent, u32) -> HipError;
pub(super) type HipEventQuery = unsafe extern "C" fn(HipEvent) -> HipError;
pub(super) type HipEventSynchronize = unsafe extern "C" fn(HipEvent) -> HipError;
pub(super) type HipEventElapsedTime = unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> HipError;
pub(super) type HipEventDestroy = unsafe extern "C" fn(HipEvent) -> HipError;
pub(super) type HipModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> HipError;
pub(super) type HipModuleGetFunction = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> HipError;
pub(super) type HipModuleLaunchKernel = unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> HipError;
pub(super) type HipModuleUnload = unsafe extern "C" fn(*mut c_void) -> HipError;

// HIP graph 显式建图与模板重放。graph 热路径符号与 kernel launch 同级,
// 必须进程级缓存,避免每节点一次 dlsym。
pub(super) type HipGraph = *mut c_void;
pub(super) type HipGraphExec = *mut c_void;
pub(super) type HipGraphNode = *mut c_void;
pub(super) type HipGraphCreate = unsafe extern "C" fn(*mut HipGraph, u32) -> HipError;
pub(super) type HipGraphAddKernelNode = unsafe extern "C" fn(*mut HipGraphNode, HipGraph, *const HipGraphNode, usize, *const HipKernelNodeParams) -> HipError;
pub(super) type HipGraphInstantiateWithFlags = unsafe extern "C" fn(*mut HipGraphExec, HipGraph, u32) -> HipError;
pub(super) type HipGraphLaunch = unsafe extern "C" fn(HipGraphExec, *mut c_void) -> HipError;
pub(super) type HipGraphDestroy = unsafe extern "C" fn(HipGraph) -> HipError;
pub(super) type HipGraphExecDestroy = unsafe extern "C" fn(HipGraphExec) -> HipError;

/// 与 HIP runtime 的 hipKernelNodeParams 布局一致。
#[repr(C)]
pub(super) struct HipKernelNodeParams {
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    pub extra: *mut *mut c_void,
    pub func: *mut c_void,
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub kernel_params: *mut *mut c_void,
    pub shared_mem_bytes: u32,
}

pub(super) type HiprtcProgram = *mut c_void;
pub(super) type HiprtcResult = i32;
pub(super) type HiprtcCreateProgram = unsafe extern "C" fn(*mut HiprtcProgram, *const c_char, *const c_char, i32, *const *const c_char, *const *const c_char) -> HiprtcResult;
pub(super) type HiprtcCompileProgram = unsafe extern "C" fn(HiprtcProgram, i32, *const *const c_char) -> HiprtcResult;
pub(super) type HiprtcGetProgramLogSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult;
pub(super) type HiprtcGetProgramLog = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult;
pub(super) type HiprtcGetCodeSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult;
pub(super) type HiprtcGetCode = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult;
pub(super) type HiprtcDestroyProgram = unsafe extern "C" fn(*mut HiprtcProgram) -> HiprtcResult;

pub(super) const HIPRTC_SUCCESS: HiprtcResult = 0;

#[cfg(test)]
mod tests {
    use super::HipKernelNodeParams;

    #[test]
    fn hip_kernel_node_params_matches_rocm_10_abi() {
        assert_eq!(std::mem::size_of::<HipKernelNodeParams>(), 64);
        assert_eq!(std::mem::offset_of!(HipKernelNodeParams, block_dim_x), 0);
        assert_eq!(std::mem::offset_of!(HipKernelNodeParams, extra), 16);
        assert_eq!(std::mem::offset_of!(HipKernelNodeParams, func), 24);
        assert_eq!(std::mem::offset_of!(HipKernelNodeParams, grid_dim_x), 32);
        assert_eq!(std::mem::offset_of!(HipKernelNodeParams, kernel_params), 48);
        assert_eq!(std::mem::offset_of!(HipKernelNodeParams, shared_mem_bytes), 56);
    }
}

pub(super) struct RocmRuntime {
    pub(super) hip: Library,
}

static INDEPENDENT_COMPUTE_STREAMS: OnceLock<Mutex<HashMap<i32, usize>>> = OnceLock::new();
static BACKGROUND_STAGE_STREAMS: OnceLock<Mutex<HashMap<i32, usize>>> = OnceLock::new();
static COOPERATIVE_PEER_STREAMS: OnceLock<Mutex<HashMap<i32, usize>>> = OnceLock::new();

fn low_priority_compute_stream(device_id: i32, registry: &'static OnceLock<Mutex<HashMap<i32, usize>>>, label: &'static str) -> Result<usize, String> {
    let mut streams = registry.get_or_init(|| Mutex::new(HashMap::new())).lock().map_err(|_| format!("ROCm {label} stream 注册表已损坏"))?;
    if let Some(&stream) = streams.get(&device_id) {
        return Ok(stream);
    }
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let priority_range: Symbol<HipDeviceGetStreamPriorityRange> = runtime.symbol(&runtime.hip, b"hipDeviceGetStreamPriorityRange\0")?;
    let create: Symbol<HipStreamCreateWithPriority> = runtime.symbol(&runtime.hip, b"hipStreamCreateWithPriority\0")?;
    let mut least_priority = 0;
    let mut greatest_priority = 0;
    let status = unsafe { priority_range(&mut least_priority, &mut greatest_priority) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, &format!("hipDeviceGetStreamPriorityRange {label}")));
    }
    let mut stream = ptr::null_mut();
    // target decode 的 default stream 保持中等优先级；后台工作只在已退休
    // thread block 释放 CU 后补位，不能抢占正在执行的 kernel。
    let status = unsafe { create(&mut stream, HIP_STREAM_NON_BLOCKING, least_priority) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, &format!("hipStreamCreateWithPriority {label}")));
    }
    eprintln!("[rocm-compute-stream] device={device_id} kind={label} priority={least_priority} greatest={greatest_priority}");
    streams.insert(device_id, stream as usize);
    Ok(stream as usize)
}

/// DSpark runtime 在同一进程内只需要每 device 一条独立 stream。注册表按 device
/// 复用 handle，生命周期与 HIP runtime 相同，保持 `RocmContext` 为轻量 Copy 值。
pub(crate) fn independent_compute_stream(device_id: i32) -> Result<usize, String> {
    low_priority_compute_stream(device_id, &INDEPENDENT_COMPUTE_STREAMS, "dspark")
}

/// target prefill 与 DSpark 必须使用不同队列，否则两种后台工作的依赖会互相串行。
pub(crate) fn background_stage_stream(device_id: i32) -> Result<usize, String> {
    low_priority_compute_stream(device_id, &BACKGROUND_STAGE_STREAMS, "stage-prefill")
}

/// 同一张卡自己的 pipeline stage 与替邻卡执行的 MoE 不能共用 stream：双向
/// event handoff 会形成环，也会把本应并发的两份计算重新串行。
pub(crate) fn cooperative_peer_stream(device_id: i32) -> Result<usize, String> {
    low_priority_compute_stream(device_id, &COOPERATIVE_PEER_STREAMS, "cooperative-peer")
}

pub(crate) fn initialized_background_stage_stream(device_id: i32) -> Option<usize> {
    BACKGROUND_STAGE_STREAMS.get()?.lock().ok()?.get(&device_id).copied()
}

pub(crate) fn order_stream_after(device_id: i32, source_stream: usize, destination_stream: usize) -> Result<(), String> {
    if source_stream == destination_stream {
        return set_device(device_id);
    }
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let create = runtime.event_create()?;
    let record = runtime.event_record()?;
    let wait: Symbol<HipStreamWaitEvent> = runtime.symbol(&runtime.hip, b"hipStreamWaitEvent\0")?;
    let destroy = runtime.event_destroy()?;
    let mut event = ptr::null_mut();
    let status = unsafe { create(&mut event, HIP_EVENT_DISABLE_TIMING) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipEventCreateWithFlags stream handoff"));
    }
    let result = (|| {
        let status = unsafe { record(event, source_stream as *mut c_void) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipEventRecord stream handoff source"));
        }
        let status = unsafe { wait(destination_stream as *mut c_void, event, 0) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipStreamWaitEvent stream handoff destination"));
        }
        Ok(())
    })();
    let _ = unsafe { destroy(event) };
    result
}

/// 热路径 HIP 符号的进程级缓存函数指针,写法与 `module_launch()` 相同:
/// 每次 `symbol()` 等于一次 dlsym 并争 loader 锁;DeviceBuffer 的 malloc/
/// free/memcpy/event 轮询每个 decode step 都会走到,统一缓存后消除这部分
/// host 开销。失败结果同样缓存,避免缺符号的平台反复 dlsym。
macro_rules! cached_hip_symbol {
    ($method:ident, $ty:ty, $name:literal) => {
        pub(super) fn $method(&self) -> Result<$ty, String> {
            static CACHED: OnceLock<Result<$ty, String>> = OnceLock::new();
            CACHED
                .get_or_init(|| {
                    let symbol = self.symbol::<$ty>(&self.hip, concat!($name, "\0").as_bytes())?;
                    Ok::<$ty, String>(*symbol)
                })
                .clone()
        }
    };
}

/// 指定 ROCm device 上的一段独占显存。权重通过 `Arc<DeviceBuffer>` 共享，不重复上传。
impl RocmRuntime {
    pub(super) fn open() -> Result<&'static Self, String> {
        static RUNTIME: OnceLock<Result<RocmRuntime, String>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| {
                // HIP 常驻 allocation 会持有 dmabuf FD；首次初始化时统一扩展进程容量。
                raise_rocm_open_file_limit()?;
                let hip = resolve_library(HIP_LIBRARIES).ok_or_else(|| "未找到 libamdhip64.so".to_string())?;
                Ok(Self { hip })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn symbol<'a, T>(&'a self, lib: &'a Library, name: &[u8]) -> Result<Symbol<'a, T>, String> {
        unsafe { lib.get(name).map_err(|error| format!("加载 {name:?} 失败: {error}")) }
    }

    /// hipModuleLaunchKernel 的进程级缓存函数指针。每次 kernel launch 走
    /// `symbol()` 等于每 launch 一次 dlsym,8 个流水线程并发时还要争 loader
    /// 锁;decode 实测 kernel 间 gap 中位数 11µs 的主要 host 成分即在此。
    pub(super) fn module_launch(&self) -> Result<HipModuleLaunchKernel, String> {
        static LAUNCH: OnceLock<Result<HipModuleLaunchKernel, String>> = OnceLock::new();
        LAUNCH
            .get_or_init(|| {
                let symbol = self.symbol::<HipModuleLaunchKernel>(&self.hip, b"hipModuleLaunchKernel\0")?;
                Ok::<HipModuleLaunchKernel, String>(*symbol)
            })
            .clone()
    }

    // 热路径符号缓存由模块级 `cached_hip_symbol!` 宏生成。
    cached_hip_symbol!(malloc, HipMalloc, "hipMalloc");
    cached_hip_symbol!(malloc_async, HipMallocAsync, "hipMallocAsync");
    cached_hip_symbol!(free, HipFree, "hipFree");
    cached_hip_symbol!(free_async, HipFreeAsync, "hipFreeAsync");
    cached_hip_symbol!(memcpy, HipMemcpy, "hipMemcpy");
    cached_hip_symbol!(memcpy_async, HipMemcpyAsync, "hipMemcpyAsync");
    cached_hip_symbol!(stream_synchronize, HipStreamSynchronize, "hipStreamSynchronize");
    cached_hip_symbol!(event_create, HipEventCreateWithFlags, "hipEventCreateWithFlags");
    cached_hip_symbol!(event_record, HipEventRecord, "hipEventRecord");
    cached_hip_symbol!(event_query, HipEventQuery, "hipEventQuery");
    cached_hip_symbol!(event_synchronize, HipEventSynchronize, "hipEventSynchronize");
    cached_hip_symbol!(event_destroy, HipEventDestroy, "hipEventDestroy");
    cached_hip_symbol!(graph_create, HipGraphCreate, "hipGraphCreate");
    cached_hip_symbol!(graph_add_kernel_node, HipGraphAddKernelNode, "hipGraphAddKernelNode");
    cached_hip_symbol!(graph_instantiate_with_flags, HipGraphInstantiateWithFlags, "hipGraphInstantiateWithFlags");
    cached_hip_symbol!(graph_launch, HipGraphLaunch, "hipGraphLaunch");
    cached_hip_symbol!(graph_destroy, HipGraphDestroy, "hipGraphDestroy");
    cached_hip_symbol!(graph_exec_destroy, HipGraphExecDestroy, "hipGraphExecDestroy");

    pub(super) fn hip_error(&self, code: HipError, action: &str) -> String {
        if code == HIP_SUCCESS {
            return String::new();
        }
        let message = if let Ok(get_error) = self.symbol::<HipGetErrorString>(&self.hip, b"hipGetErrorString\0") {
            unsafe {
                let raw = get_error(code);
                if raw.is_null() {
                    "<未返回错误文本>".to_owned()
                } else {
                    let c_str = std::ffi::CStr::from_ptr(raw);
                    c_str.to_string_lossy().into_owned()
                }
            }
        } else {
            format!("code={code}")
        };
        format!("{action} 失败: {message}")
    }
}

/// 是否能加载 HIP 运行时。
pub fn is_hip_available() -> bool {
    RocmRuntime::open().is_ok()
}

/// device 的 wavefront size(32/64),按 device 缓存。
/// 个别 kernel 按 wave32 硬编码 lane 划分(`threadIdx.x & 31`),launch 前用它
/// 显式校验,避免在 CDNA(wave64)上静默算错。
pub(crate) fn device_wavefront_size(device_id: i32) -> Result<u32, String> {
    static SIZES: OnceLock<Mutex<HashMap<i32, Result<u32, String>>>> = OnceLock::new();
    let sizes = SIZES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut sizes = sizes.lock().map_err(|_| "ROCm wavefront size cache mutex 已损坏".to_owned())?;
    if let Some(result) = sizes.get(&device_id) {
        return result.clone();
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let device_get_attribute: Symbol<HipDeviceGetAttribute> = runtime.symbol(&runtime.hip, b"hipDeviceGetAttribute\0")?;
        let mut wavefront_size = 0i32;
        // 87 = hipDeviceAttributeWarpSize,与 attention/linear 已有的查询一致。
        let status = unsafe { device_get_attribute(&mut wavefront_size, 87, device_id) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipDeviceGetAttribute wavefront size"));
        }
        u32::try_from(wavefront_size).map_err(|_| format!("ROCm device={device_id} wavefront size={wavefront_size} 非法"))
    })();
    sizes.insert(device_id, result.clone());
    result
}

/// 运行时版本参与缓存 key:ROCm 升级后旧 code object ABI 不再匹配,
/// 不混入版本会造成跨版本复用旧 .co(hipModuleLoadData 失败或加载错对象)。
pub(crate) fn enable_peer_access(destination_device_id: i32, source_device_id: i32) -> Result<(), String> {
    if destination_device_id == source_device_id {
        return Ok(());
    }
    let mut enabled = ENABLED_PEER_ACCESS.get_or_init(|| Mutex::new(HashMap::new())).lock().map_err(|_| "ROCm peer access 状态锁中毒".to_owned())?;
    if enabled.contains_key(&(destination_device_id, source_device_id)) {
        return Ok(());
    }
    set_device(destination_device_id)?;
    let runtime = RocmRuntime::open()?;
    let can_access: Symbol<HipDeviceCanAccessPeer> = runtime.symbol(&runtime.hip, b"hipDeviceCanAccessPeer\0")?;
    let mut supported = 0;
    let status = unsafe { can_access(&mut supported, destination_device_id, source_device_id) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipDeviceCanAccessPeer activation"));
    }
    if supported == 0 {
        return Err(format!("ROCm P2P 不支持: source_device={source_device_id} destination_device={destination_device_id}"));
    }
    let enable: Symbol<HipDeviceEnablePeerAccess> = runtime.symbol(&runtime.hip, b"hipDeviceEnablePeerAccess\0")?;
    let status = unsafe { enable(source_device_id, 0) };
    if status != HIP_SUCCESS && status != HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED {
        return Err(runtime.hip_error(status, "hipDeviceEnablePeerAccess activation"));
    }
    enabled.insert((destination_device_id, source_device_id), ());
    Ok(())
}

pub(crate) fn synchronize_device(device_id: i32, action: &str) -> Result<(), String> {
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let synchronize: Symbol<HipDeviceSynchronize> = runtime.symbol(&runtime.hip, b"hipDeviceSynchronize\0")?;
    let status = unsafe { synchronize() };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, action));
    }
    promote_device_buffers(device_id);
    Ok(())
}

/// 只等待当前线程为该 device 激活的 compute stream；用于调用方已把生产、
/// P2P 与消费严格排在同一 stream 上的边界，避免把无关 stream 纳入 barrier。
pub(crate) fn synchronize_compute_stream(device_id: i32, action: &str) -> Result<(), String> {
    set_device(device_id)?;
    let runtime = RocmRuntime::open()?;
    let synchronize = runtime.stream_synchronize()?;
    let status = unsafe { synchronize(compute_stream_for(device_id)) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, action));
    }
    promote_device_buffers(device_id);
    Ok(())
}

pub fn set_device(device_id: i32) -> Result<(), String> {
    // workspace TLS 退出时会析构 DeviceBuffer；此时设备缓存 TLS 可能已经先销毁。
    // 析构路径仍需设置 HIP device，但不能再用 LocalKey::with 触发二次 panic。
    if CURRENT_HIP_DEVICE.try_with(|current| current.get() == Some(device_id)).unwrap_or(false) {
        remember_active_device(device_id);
        return Ok(());
    }
    let runtime = RocmRuntime::open()?;
    let count = *HIP_DEVICE_COUNT
        .get_or_init(|| {
            let hip_init: Symbol<HipInit> = runtime.symbol(&runtime.hip, b"hipInit\0")?;
            let hip_get_device_count: Symbol<HipGetDeviceCount> = runtime.symbol(&runtime.hip, b"hipGetDeviceCount\0")?;
            let init_status = unsafe { hip_init(0) };
            if init_status != HIP_SUCCESS {
                return Err(runtime.hip_error(init_status, "hipInit"));
            }
            let mut count = 0;
            let count_status = unsafe { hip_get_device_count(&mut count) };
            if count_status != HIP_SUCCESS {
                return Err(runtime.hip_error(count_status, "hipGetDeviceCount"));
            }
            Ok(count)
        })
        .as_ref()
        .map_err(Clone::clone)?;
    if device_id < 0 || device_id >= count {
        return Err(format!("HIP device {device_id} 越界，可用设备数={count}"));
    }
    let hip_set_device: Symbol<HipSetDevice> = runtime.symbol(&runtime.hip, b"hipSetDevice\0")?;
    let set_status = unsafe { hip_set_device(device_id) };
    if set_status != HIP_SUCCESS {
        return Err(runtime.hip_error(set_status, &format!("hipSetDevice({device_id})")));
    }
    let _ = CURRENT_HIP_DEVICE.try_with(|current| current.set(Some(device_id)));
    remember_active_device(device_id);
    Ok(())
}
