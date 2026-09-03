//! 跨卡 peer 拷贝 kernel。
//!
//! W7900D 上 `hipMemcpyPeerAsync` 的 copy-engine 路径会把大块传输拆成
//! 128KiB 分块的 rocclr copyBuffer kernel,每块 ~2ms(等效 ~61MB/s),
//! 120 MiB 的层间 hidden 单边界要 ~1.9s。peer access 已启用时,目标卡上的
//! 普通内核可以直接读源卡显存,一个 float4 向量化拷贝 kernel 就能以
//! PCIe 整段速度(~6-12GB/s)完成同一传输。

use super::*;

const PEER_COPY_SOURCE: &str = include_str!("peer_copy/source.hip");

#[derive(Clone, Copy)]
struct PeerCopyFunctions {
    copy: usize,
}

fn peer_copy_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(PEER_COPY_SOURCE, "zllm_rocm_peer_copy.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

fn peer_copy_functions(device_id: i32) -> Result<PeerCopyFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Result<(usize, PeerCopyFunctions), String>>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "ROCm peer copy kernel cache mutex 已损坏".to_owned())?;
    if let Some(result) = functions.get(&device_id) {
        return result.as_ref().map(|(_, functions)| *functions).map_err(Clone::clone);
    }
    let result = (|| {
        set_device(device_id)?;
        let runtime = RocmRuntime::open()?;
        let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
        let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
        let code = peer_copy_code()?;
        let mut module = ptr::null_mut();
        let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
        if load_status != HIP_SUCCESS {
            return Err(runtime.hip_error(load_status, "hipModuleLoadData peer copy"));
        }
        let name = CString::new("zllm_peer_copy_f32x4").unwrap();
        let mut function = ptr::null_mut();
        let status = unsafe { module_get_function(&mut function, module, name.as_ptr()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleGetFunction peer copy"));
        }
        Ok((module as usize, PeerCopyFunctions { copy: function as usize }))
    })();
    functions.insert(device_id, result.clone());
    result.map(|(_, functions)| functions)
}

/// 在目标卡 stream 上执行 peer 拷贝(整段 kernel,替换 hipMemcpyPeerAsync)。
/// 要求 `bytes` 是 16 的倍数——层间 hidden 等大张量天然满足;同卡路径不走这里。
pub(crate) fn try_peer_copy_kernel_ordered(device_id: i32, destination: *mut c_void, source: *mut c_void, bytes: usize) -> Result<(), String> {
    if bytes % 16 != 0 {
        return Err(format!("peer copy kernel 要求 16 字节对齐大小,实际 {bytes}"));
    }
    let functions = peer_copy_functions(device_id)?;
    let vectors = bytes / 16;
    let vectors_u32 = u32::try_from(vectors).map_err(|_| "peer copy 向量数超过 u32".to_owned())?;
    let mut d_source = source;
    let mut d_target = destination;
    let mut vectors_arg = vectors_u32;
    let mut arguments = [(&mut d_source as *mut *mut c_void).cast(), (&mut d_target as *mut *mut c_void).cast(), (&mut vectors_arg as *mut u32).cast()];
    let block = 256u32;
    // 大 hidden 保持足够在途负载，但 route ids/weights 只有几十 KiB；固定
    // 2048 blocks 会为小 handoff 启动五十多万个线程，反而占满 peer 队列。
    // 按实际向量数缩小 grid，上限仍保留已验证的大块 BAR 吞吐配置。
    let grid = vectors_u32.div_ceil(block).clamp(1, 2048);
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let __hip_launch_result = unsafe { launch(functions.copy as *mut c_void, grid, 1, 1, block, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel peer copy"));
    }
    if let Some(started) = profile_started {
        synchronize_device(device_id, "peer copy profile")?;
        eprintln!("[rocm-kernel] peer-copy device={device_id} bytes={bytes} grid={grid} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(())
}
