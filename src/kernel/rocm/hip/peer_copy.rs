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
    copy3: usize,
    join_residual: usize,
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
        let mut handles = [ptr::null_mut(); 3];
        for (index, name) in ["zllm_peer_copy_f32x4", "zllm_peer_copy3_f32x4", "zllm_peer_join_residual_f32x4"].into_iter().enumerate() {
            let name = CString::new(name).unwrap();
            let status = unsafe { module_get_function(&mut handles[index], module, name.as_ptr()) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleGetFunction peer copy"));
            }
        }
        Ok((module as usize, PeerCopyFunctions { copy: handles[0] as usize, copy3: handles[1] as usize, join_residual: handles[2] as usize }))
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

/// 三段小数据共享同一个 producer 边界，直接传指针，避免另上传 descriptor。
pub(crate) fn try_peer_copy3_kernel_ordered(device_id: i32, mut sources: [*mut c_void; 3], mut destinations: [*mut c_void; 3], bytes: [usize; 3]) -> Result<(), String> {
    if sources.iter().chain(&destinations).any(|pointer| !(*pointer as usize).is_multiple_of(16)) || bytes.iter().any(|bytes| !bytes.is_multiple_of(16) || *bytes > 4096) || bytes.iter().sum::<usize>() > 4096 {
        return Err(format!("peer copy3 要求三段地址和大小 16 字节对齐且总量不超过 4096，实际 {bytes:?}"));
    }
    let mut vectors = bytes.map(|bytes| (bytes / 16) as u32);
    let functions = peer_copy_functions(device_id)?;
    let mut arguments = [
        (&mut sources[0] as *mut *mut c_void).cast(),
        (&mut sources[1] as *mut *mut c_void).cast(),
        (&mut sources[2] as *mut *mut c_void).cast(),
        (&mut destinations[0] as *mut *mut c_void).cast(),
        (&mut destinations[1] as *mut *mut c_void).cast(),
        (&mut destinations[2] as *mut *mut c_void).cast(),
        (&mut vectors[0] as *mut u32).cast(),
        (&mut vectors[1] as *mut u32).cast(),
        (&mut vectors[2] as *mut u32).cast(),
    ];
    let started = super::hip_api_stats::start();
    let status = unsafe { super::kernel_launch_trampoline(functions.copy3 as *mut c_void, 1, 1, 1, 256, 1, 1, 0, super::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, started);
    if status != HIP_SUCCESS {
        return Err(RocmRuntime::open()?.hip_error(status, "hipModuleLaunchKernel peer copy3"));
    }
    Ok(())
}

/// 在目标卡直接读取对端 F32 partial，并与本地 partial/residual 融合。
/// 调用方负责先在目标 stream 排入对端 producer event wait。
pub(crate) fn try_peer_join_residual_f32(device_id: i32, local_partial: &DeviceBuffer, peer_partial: &DeviceBuffer, residual: &DeviceBuffer, elements: usize) -> Result<DeviceBuffer, String> {
    if elements == 0 || !elements.is_multiple_of(4) {
        return Err(format!("peer partial join 元素数必须是非零 4 倍数，实际 {elements}"));
    }
    let bytes = elements.checked_mul(std::mem::size_of::<f32>()).ok_or("peer partial join 大小溢出")?;
    validate_resident(local_partial, device_id, bytes, "peer join local partial")?;
    validate_resident(residual, device_id, bytes, "peer join residual")?;
    if peer_partial.bytes < bytes {
        return Err(format!("peer join remote partial 大小 {} 小于 {bytes}", peer_partial.bytes));
    }
    if peer_partial.device_id != device_id {
        enable_peer_access(device_id, peer_partial.device_id)?;
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, bytes)?;
    let functions = peer_copy_functions(device_id)?;
    let mut local_pointer = local_partial.pointer;
    let mut peer_pointer = peer_partial.pointer;
    let mut residual_pointer = residual.pointer;
    let mut output_pointer = output.pointer;
    let mut vectors = u32::try_from(elements / 4).map_err(|_| "peer partial join 向量数超过 u32".to_owned())?;
    let mut arguments = [
        (&mut local_pointer as *mut *mut c_void).cast(),
        (&mut peer_pointer as *mut *mut c_void).cast(),
        (&mut residual_pointer as *mut *mut c_void).cast(),
        (&mut output_pointer as *mut *mut c_void).cast(),
        (&mut vectors as *mut u32).cast(),
    ];
    let block = 256u32;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let status = unsafe { launch(functions.join_residual as *mut c_void, vectors.div_ceil(block), 1, 1, block, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel peer partial join"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_copy3_preserves_bytes_and_destination_offsets() {
        if !is_hip_available() {
            return;
        }
        for sizes in [[16_usize, 32, 48], [1024, 32, 256], [2048, 64, 512], [2032, 16, 2048]] {
            let inputs = sizes.map(|bytes| (0..bytes).map(|index| (index.wrapping_mul(97) >> 3) as u8).collect::<Vec<_>>());
            let sources = inputs.each_ref().map(|input| DeviceBuffer::upload(0, input).unwrap());
            let destinations = sizes.map(|bytes| DeviceBuffer::upload(0, &vec![0xa5; bytes + 32]).unwrap());
            let pointers: [*mut c_void; 3] = destinations.each_ref().map(|buffer| unsafe { buffer.pointer.cast::<u8>().add(16).cast() });
            let mut unaligned = pointers;
            unaligned[1] = unsafe { unaligned[1].cast::<u8>().add(4).cast() };
            assert!(try_peer_copy3_kernel_ordered(0, sources.each_ref().map(|buffer| buffer.pointer), unaligned, sizes).is_err());
            try_peer_copy3_kernel_ordered(0, sources.each_ref().map(|buffer| buffer.pointer), pointers, sizes).unwrap();
            for ((destination, input), bytes) in destinations.iter().zip(&inputs).zip(sizes) {
                let mut actual = vec![0; bytes + 32];
                destination.copy_to_host(&mut actual).unwrap();
                assert_eq!(&actual[..16], &[0xa5; 16]);
                assert_eq!(&actual[16..16 + bytes], input);
                assert_eq!(&actual[16 + bytes..], &[0xa5; 16]);
            }
        }
    }

    #[test]
    fn peer_join_residual_matches_two_adds() {
        if !is_hip_available() {
            return;
        }
        let local_values = [1.0_f32, -2.0, 3.5, 4.0, 5.0, 6.0, -7.0, 8.0];
        let peer_values = [0.5_f32, 3.0, -1.5, 2.0, -4.0, 1.0, 2.0, -3.0];
        let residual_values = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let local = DeviceBuffer::upload_f32(0, &local_values).unwrap();
        let peer = DeviceBuffer::upload_f32(0, &peer_values).unwrap();
        let residual = DeviceBuffer::upload_f32(0, &residual_values).unwrap();
        let output = try_peer_join_residual_f32(0, &local, &peer, &residual, local_values.len()).unwrap();
        let actual = output.download_f32(local_values.len()).unwrap();
        let expected = local_values.into_iter().zip(peer_values).zip(residual_values).map(|((local, peer), residual)| residual + (local + peer)).collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
