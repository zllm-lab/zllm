use super::*;

const QK_MATMUL_SOURCE: &str = include_str!("qk_matmul/source.hip");

fn qk_matmul_code() -> Result<&'static [u8], String> {
    static CODE: OnceLock<Result<Vec<u8>, String>> = OnceLock::new();
    CODE.get_or_init(|| compile_hip_source(QK_MATMUL_SOURCE, "zllm_rocm_qk.hip")).as_ref().map(Vec::as_slice).map_err(Clone::clone)
}

pub fn try_qk_matmul_f32(device_id: i32, tensor_type: u32, input: &[f32], weight: &[u8], input_rows: usize, columns: usize, output_rows: usize, output: &mut [f32]) -> Result<(), String> {
    const QK_K: usize = 256;
    let block_bytes = match tensor_type {
        12 => 144,
        13 => 176,
        other => return Err(format!("ROCm QK matmul 不支持 GGUF type {other}")),
    };
    if input_rows == 0 || columns == 0 || output_rows == 0 || !columns.is_multiple_of(QK_K) {
        return Err(format!("QK matmul shape 非法: [{input_rows},{columns}] x [{output_rows},{columns}]"));
    }
    let input_elements = input_rows.checked_mul(columns).ok_or("QK input 大小溢出")?;
    let output_elements = input_rows.checked_mul(output_rows).ok_or("QK output 大小溢出")?;
    let weight_bytes = output_rows.checked_mul(columns / QK_K).and_then(|blocks| blocks.checked_mul(block_bytes)).ok_or("QK weight 大小溢出")?;
    if input.len() != input_elements || output.len() != output_elements || weight.len() != weight_bytes {
        return Err(format!("QK matmul 数据大小非法: type={tensor_type} input={}/{input_elements} weight={}/{weight_bytes} output={}/{output_elements}", input.len(), weight.len(), output.len()));
    }

    let runtime = RocmRuntime::open()?;
    let hip_init: Symbol<HipInit> = runtime.symbol(&runtime.hip, b"hipInit\0")?;
    let hip_set_device: Symbol<HipSetDevice> = runtime.symbol(&runtime.hip, b"hipSetDevice\0")?;
    let hip_malloc: Symbol<HipMalloc> = runtime.symbol(&runtime.hip, b"hipMalloc\0")?;
    let hip_free: Symbol<HipFree> = runtime.symbol(&runtime.hip, b"hipFree\0")?;
    let hip_memcpy: Symbol<HipMemcpy> = runtime.symbol(&runtime.hip, b"hipMemcpy\0")?;
    let hip_synchronize: Symbol<HipDeviceSynchronize> = runtime.symbol(&runtime.hip, b"hipDeviceSynchronize\0")?;
    let module_launch = crate::kernel::rocm::hip::kernel_launch_trampoline;

    let init_status = unsafe { hip_init(0) };
    if init_status != HIP_SUCCESS {
        return Err(runtime.hip_error(init_status, "hipInit Q4_K"));
    }
    let set_device_status = unsafe { hip_set_device(device_id) };
    if set_device_status != HIP_SUCCESS {
        return Err(runtime.hip_error(set_device_status, "hipSetDevice Q4_K"));
    }

    let mut d_input = ptr::null_mut();
    let mut d_weight = ptr::null_mut();
    let mut d_output = ptr::null_mut();
    let result = (|| {
        let input_bytes = std::mem::size_of_val(input);
        let output_bytes = std::mem::size_of_val(output);
        for (target, bytes, label) in [(&mut d_input, input_bytes, "input"), (&mut d_weight, weight_bytes, "weight"), (&mut d_output, output_bytes, "output")] {
            let status = unsafe { hip_malloc(target, bytes) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, &format!("hipMalloc Q4_K {label}")));
            }
        }
        for (target, source, bytes, label) in [(d_input, input.as_ptr().cast(), input_bytes, "input"), (d_weight, weight.as_ptr().cast(), weight_bytes, "weight")] {
            let status = unsafe { hip_memcpy(target, source, bytes, HIP_MEMORY_COPY_HOST_TO_DEVICE) };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, &format!("hipMemcpy Q4_K {label}")));
            }
        }

        // 复用按设备的 resident kernel 缓存;原进程级 OnceLock 会在多卡下
        // 把 device 0 加载的句柄错误地用于其他设备。
        let (function, _) = qk_matmul_resident_functions(device_id, runtime)?;

        let mut input_rows = u32::try_from(input_rows).map_err(|_| "Q4_K input_rows 超过 u32")?;
        let mut columns = u32::try_from(columns).map_err(|_| "Q4_K columns 超过 u32")?;
        let mut output_rows = u32::try_from(output_rows).map_err(|_| "Q4_K output_rows 超过 u32")?;
        let mut tensor_type = tensor_type;
        let mut input_is_bf16 = 0_u32;
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_weight as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut input_rows as *mut u32).cast(),
            (&mut columns as *mut u32).cast(),
            (&mut output_rows as *mut u32).cast(),
            (&mut tensor_type as *mut u32).cast(),
            (&mut input_is_bf16 as *mut u32).cast(),
        ];
        let token_groups = input_rows.div_ceil(8);
        let launch_status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let __hip_launch_result = unsafe { module_launch(function, output_rows, token_groups, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if launch_status != HIP_SUCCESS {
            return Err(runtime.hip_error(launch_status, "hipModuleLaunchKernel Q4_K"));
        }
        let sync_status = unsafe { hip_synchronize() };
        if sync_status != HIP_SUCCESS {
            return Err(runtime.hip_error(sync_status, "hipDeviceSynchronize Q4_K"));
        }
        let copy_status = unsafe { hip_memcpy(output.as_mut_ptr().cast(), d_output, output_bytes, HIP_MEMORY_COPY_DEVICE_TO_HOST) };
        if copy_status != HIP_SUCCESS {
            return Err(runtime.hip_error(copy_status, "hipMemcpy Q4_K output"));
        }
        Ok(())
    })();

    for pointer in [d_output, d_weight, d_input] {
        if !pointer.is_null() {
            let _ = unsafe { hip_free(pointer) };
        }
    }
    result
}

/// [`try_qk_matmul_f32`] 的常驻变体：Q4_K/Q5_K 权重已驻留设备，输入从 host 上传，
/// 输出留在设备。与非常驻版本共用同一 `qk_matmul_f32` kernel。
/// qk/q6 matmul 的常驻执行：Q4_K/Q5_K/Q6_K 权重已驻留设备。
/// 输入优先复用设备侧 buffer(F32 或 BF16，kernel 内联转换)；host 输入走池化上传。
#[allow(clippy::too_many_arguments)]
pub fn try_qk_matmul_resident_f32(device_id: i32, tensor_type: u32, input: &[f32], input_device: Option<&DeviceBuffer>, weight: &DeviceBuffer, input_rows: usize, columns: usize, output_rows: usize) -> Result<DeviceBuffer, String> {
    const QK_K: usize = 256;
    let block_bytes = match tensor_type {
        12 => 144,
        13 => 176,
        14 => 210,
        other => return Err(format!("ROCm QK resident matmul 不支持 GGUF type {other}")),
    };
    if input_rows == 0 || columns == 0 || output_rows == 0 || !columns.is_multiple_of(QK_K) {
        return Err(format!("QK resident matmul shape 非法: [{input_rows},{columns}] x [{output_rows},{columns}]"));
    }
    if weight.device_id != device_id {
        return Err("QK resident weight 与执行 device 不一致".to_owned());
    }
    let input_elements = input_rows.checked_mul(columns).ok_or("QK resident input 大小溢出")?;
    let output_bytes = input_rows.checked_mul(output_rows).and_then(|elements| elements.checked_mul(4)).ok_or("QK resident output 大小溢出")?;
    let weight_bytes = output_rows.checked_mul(columns / QK_K).and_then(|blocks| blocks.checked_mul(block_bytes)).ok_or("QK resident weight 大小溢出")?;
    if weight.bytes() != weight_bytes {
        return Err(format!("QK resident weight 字节 {}，期望 {weight_bytes}", weight.bytes()));
    }
    let input_bytes_f32 = input_elements.checked_mul(4).ok_or("QK resident input 字节溢出")?;
    let input_bytes_bf16 = input_elements.checked_mul(2).ok_or("QK resident input 字节溢出")?;
    let uploaded_input;
    let input_pointer: *mut c_void;
    let input_is_bf16: u32;
    match input_device {
        Some(buffer) if buffer.device_id == device_id && buffer.bytes == input_bytes_bf16 => {
            input_pointer = buffer.pointer;
            input_is_bf16 = 1;
        }
        Some(buffer) if buffer.device_id == device_id && buffer.bytes == input_bytes_f32 => {
            input_pointer = buffer.pointer;
            input_is_bf16 = 0;
        }
        _ => {
            if input.len() != input_elements {
                return Err(format!("QK resident matmul host input={}/{}，且无匹配设备 buffer", input.len(), input_elements));
            }
            uploaded_input = DeviceBuffer::allocate_reusable(device_id, input_bytes_f32)?;
            uploaded_input.copy_from_host(unsafe { std::slice::from_raw_parts(input.as_ptr().cast(), input_bytes_f32) })?;
            input_pointer = uploaded_input.pointer;
            input_is_bf16 = 0;
        }
    }
    let runtime = RocmRuntime::open()?;
    let hip_init: Symbol<HipInit> = runtime.symbol(&runtime.hip, b"hipInit\0")?;
    let module_launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let init_status = unsafe { hip_init(0) };
    if init_status != HIP_SUCCESS {
        return Err(runtime.hip_error(init_status, "hipInit QK resident"));
    }
    let d_output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;
    let (qk_function, q6_function) = qk_matmul_resident_functions(device_id, &runtime)?;
    let function = if tensor_type == 14 { q6_function } else { qk_function };
    let mut rows = u32::try_from(input_rows).map_err(|_| "QK resident input_rows 超过 u32")?;
    let mut cols = u32::try_from(columns).map_err(|_| "QK resident columns 超过 u32")?;
    let mut out_rows = u32::try_from(output_rows).map_err(|_| "QK resident output_rows 超过 u32")?;
    let mut tensor_type = tensor_type;
    let mut is_bf16 = input_is_bf16;
    let mut input_pointer = input_pointer;
    let mut arguments = [
        (&mut input_pointer as *mut *mut c_void).cast(),
        (&weight.pointer as *const _ as *mut c_void).cast(),
        (&d_output.pointer as *const _ as *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut cols as *mut u32).cast(),
        (&mut out_rows as *mut u32).cast(),
        (&mut tensor_type as *mut u32).cast(),
        (&mut is_bf16 as *mut u32).cast(),
    ];
    let token_groups = input_rows.div_ceil(8);
    let launch_status = unsafe {
        module_launch(
            function,
            u32::try_from(output_rows).map_err(|_| "QK resident grid 超过 u32")?,
            u32::try_from(token_groups).map_err(|_| "QK resident grid 超过 u32")?,
            1,
            256,
            1,
            1,
            0,
            crate::kernel::rocm::hip::active_compute_stream(),
            arguments.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    if launch_status != HIP_SUCCESS {
        return Err(runtime.hip_error(launch_status, "hipModuleLaunchKernel QK resident"));
    }
    // DeviceBuffer 的池化回收已由当前 stream 的 HIP event 保护。默认保持整条
    // linear 链异步，避免 decode 每层多次整卡同步；调试时仍可显式强制同步。
    if options().kernel_sync {
        synchronize_device(device_id, "QK resident")?;
    }
    Ok(d_output)
}

/// qk/q6 matmul kernel 的按设备加载：hipModuleLoadData 绑定加载时的设备上下文，
/// 多卡场景必须每个 device 各加载一份(参照 bf16_gemv_functions 的缓存模式)。
fn qk_matmul_resident_functions(device_id: i32, runtime: &RocmRuntime) -> Result<(*mut c_void, *mut c_void), String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, (usize, usize)>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "QK resident kernel cache 已损坏".to_owned())?;
    if let Some(&(qk, q6)) = functions.get(&device_id) {
        return Ok((qk as *mut c_void, q6 as *mut c_void));
    }
    set_device(device_id)?;
    let module_load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
    let module_get_function: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
    let code = qk_matmul_code()?;
    let mut module = ptr::null_mut();
    let load_status = unsafe { module_load(&mut module, code.as_ptr().cast()) };
    if load_status != HIP_SUCCESS {
        return Err(runtime.hip_error(load_status, "hipModuleLoadData QK resident"));
    }
    let mut loaded = [ptr::null_mut(); 2];
    for (slot, name) in ["qk_matmul_f32", "q6_matmul_f32"].into_iter().enumerate() {
        let name = CString::new(name).unwrap();
        let status = unsafe { module_get_function(&mut loaded[slot], module, name.as_ptr()) };
        if status != HIP_SUCCESS {
            let module_unload: Symbol<HipModuleUnload> = runtime.symbol(&runtime.hip, b"hipModuleUnload\0")?;
            let _ = unsafe { module_unload(module) };
            return Err(runtime.hip_error(status, "hipModuleGetFunction QK resident"));
        }
    }
    // module 常驻不卸载；仅保存 function 句柄。
    functions.insert(device_id, (loaded[0] as usize, loaded[1] as usize));
    Ok((loaded[0], loaded[1]))
}
