use super::*;

#[cfg(test)]
use super::ct_grouped::{CtGroupedExpertMeta, CtGroupedWeightMeta, launch_ct_decode_experts_bf16, upload_pod};

const BF16_GEMV_SOURCE: &str = include_str!("bf16_gemv/source.hip");

#[derive(Clone, Copy)]
struct Bf16GemvFunctions {
    _module: usize,
    gemv: usize,
    gemv_bfdot: usize,
    f32_gemv: usize,
    f32_gemv_small_n: usize,
    f32_gemv_splitk: usize,
    f32_gemv_splitk_reduce: usize,
    f32_dual_gemv: usize,
    bf16_dual_gemv: usize,
    gemv_rows2: usize,
    gemv_rows3: usize,
    gemv_rows4: usize,
    gemv_rows8: usize,
    gemv_rows8_bfdot: usize,
}

fn bf16_gemv_functions(device_id: i32, runtime: &RocmRuntime) -> Result<Bf16GemvFunctions, String> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<i32, Bf16GemvFunctions>>> = OnceLock::new();
    let functions = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut functions = functions.lock().map_err(|_| "BF16 GEMV kernel cache 已损坏".to_owned())?;
    if let Some(&function) = functions.get(&device_id) {
        return Ok(function);
    }
    set_device(device_id)?;
    let code = compile_hip_source(&[super::super::hiprtc::DEVICE_CONVERSIONS_PREAMBLE, BF16_GEMV_SOURCE].concat(), "zllm_bf16_gemv.hip")?;
    let load: Symbol<HipModuleLoadData> = runtime.symbol(&runtime.hip, b"hipModuleLoadData\0")?;
    let get: Symbol<HipModuleGetFunction> = runtime.symbol(&runtime.hip, b"hipModuleGetFunction\0")?;
    let mut module = ptr::null_mut();
    let status = unsafe { load(&mut module, code.as_ptr().cast()) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLoadData BF16 GEMV"));
    }
    let mut gemv: *mut c_void = ptr::null_mut();
    let mut gemv_bfdot: *mut c_void = ptr::null_mut();
    let mut f32_gemv: *mut c_void = ptr::null_mut();
    let mut f32_gemv_small_n: *mut c_void = ptr::null_mut();
    let mut f32_gemv_splitk: *mut c_void = ptr::null_mut();
    let mut f32_gemv_splitk_reduce: *mut c_void = ptr::null_mut();
    let mut f32_dual_gemv: *mut c_void = ptr::null_mut();
    let mut bf16_dual_gemv: *mut c_void = ptr::null_mut();
    let mut gemv_rows2: *mut c_void = ptr::null_mut();
    let mut gemv_rows3: *mut c_void = ptr::null_mut();
    let mut gemv_rows4: *mut c_void = ptr::null_mut();
    let mut gemv_rows8: *mut c_void = ptr::null_mut();
    let mut gemv_rows8_bfdot: *mut c_void = ptr::null_mut();
    for (name, target) in [
        ("bf16_gemv_f32", &mut gemv),
        ("bf16_gemv_bfdot_f32", &mut gemv_bfdot),
        ("f32_gemv_f32", &mut f32_gemv),
        ("f32_gemv_small_n_f32", &mut f32_gemv_small_n),
        ("f32_gemv_splitk_f32", &mut f32_gemv_splitk),
        ("f32_gemv_splitk_reduce_f32", &mut f32_gemv_splitk_reduce),
        ("f32_dual_gemv_f32", &mut f32_dual_gemv),
        ("bf16_dual_gemv_f32", &mut bf16_dual_gemv),
        ("bf16_gemv_f32_rows2", &mut gemv_rows2),
        ("bf16_gemv_f32_rows3", &mut gemv_rows3),
        ("bf16_gemv_f32_rows4", &mut gemv_rows4),
        ("bf16_gemv_f32_rows8", &mut gemv_rows8),
        ("bf16_gemv_bfdot_f32_rows8", &mut gemv_rows8_bfdot),
    ] {
        let cname = CString::new(name).unwrap();
        let status = unsafe { get(target, module, cname.as_ptr()) };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleGetFunction BF16 GEMV"));
        }
    }
    let function = Bf16GemvFunctions {
        _module: module as usize,
        gemv: gemv as usize,
        gemv_bfdot: gemv_bfdot as usize,
        f32_gemv: f32_gemv as usize,
        f32_gemv_small_n: f32_gemv_small_n as usize,
        f32_gemv_splitk: f32_gemv_splitk as usize,
        f32_gemv_splitk_reduce: f32_gemv_splitk_reduce as usize,
        f32_dual_gemv: f32_dual_gemv as usize,
        bf16_dual_gemv: bf16_dual_gemv as usize,
        gemv_rows2: gemv_rows2 as usize,
        gemv_rows3: gemv_rows3 as usize,
        gemv_rows4: gemv_rows4 as usize,
        gemv_rows8: gemv_rows8 as usize,
        gemv_rows8_bfdot: gemv_rows8_bfdot as usize,
    };
    functions.insert(device_id, function);
    Ok(function)
}

pub fn try_f32_gemv_resident_f32(device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, input_rows: usize, columns: usize, output_rows: usize) -> Result<DeviceBuffer, String> {
    if input.device_id != device_id || weight.device_id != device_id || input_rows == 0 || columns == 0 || output_rows == 0 {
        return Err("ROCm F32 GEMV shape 为空或 device 不一致".to_owned());
    }
    if device_wavefront_size(device_id)? != 32 {
        return Err(format!("ROCm F32 GEMV kernel 仅支持 wave32,device={device_id}"));
    }
    let input_bytes = input_rows.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("F32 GEMV input 大小溢出")?;
    let weight_bytes = output_rows.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("F32 GEMV weight 大小溢出")?;
    if input.bytes < input_bytes || weight.bytes < weight_bytes {
        return Err(format!("ROCm F32 GEMV buffer 过小: input={}/{} weight={}/{}", input.bytes, input_bytes, weight.bytes, weight_bytes));
    }
    let output = DeviceBuffer::allocate_reusable(device_id, input_rows.checked_mul(output_rows).and_then(|n| n.checked_mul(4)).ok_or("F32 GEMV output 大小溢出")?)?;
    let runtime = RocmRuntime::open()?;
    let functions = bf16_gemv_functions(device_id, runtime)?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let input_row_bytes = columns.checked_mul(4).ok_or("F32 GEMV input 行大小溢出")?;
    let output_row_bytes = output_rows.checked_mul(4).ok_or("F32 GEMV output 行大小溢出")?;
    let mut input_pointer = input.pointer;
    let mut weight_pointer = weight.pointer;
    let mut output_pointer = output.pointer;
    let mut columns = u32::try_from(columns).map_err(|_| "F32 GEMV columns 超过 u32")?;
    let mut rows = u32::try_from(output_rows).map_err(|_| "F32 GEMV rows 超过 u32")?;
    let input_rows_u32 = u32::try_from(input_rows).map_err(|_| "F32 GEMV input rows 超过 u32")?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    if input_rows == 1 {
        // 小输出行 + 长列(mHC function 投影 [~24, 16384])单行 grid 只有 rows
        // 个 wave,切列段并行;段对齐 128(wave32 × float4),两段 launch 后合并。
        if rows <= 32 && columns >= 2048 {
            let column_count = columns as usize;
            let segment = column_count.div_ceil(512usize.div_ceil(rows as usize).max(1)).next_multiple_of(128).max(128).min(column_count.next_multiple_of(128));
            let splits = column_count.div_ceil(segment).max(1);
            let mut splits_u32 = u32::try_from(splits).map_err(|_| "F32 GEMV splits 超过 u32")?;
            let mut segment_u32 = u32::try_from(segment).map_err(|_| "F32 GEMV segment 超过 u32")?;
            let partial = DeviceBuffer::allocate_reusable(device_id, splits.checked_mul(output_rows).and_then(|n| n.checked_mul(4)).ok_or("F32 GEMV split partial 大小溢出")?)?;
            let mut partial_pointer = partial.pointer;
            let mut split_arguments = [
                (&mut input_pointer as *mut *mut c_void).cast(),
                (&mut weight_pointer as *mut *mut c_void).cast(),
                (&mut partial_pointer as *mut *mut c_void).cast(),
                (&mut columns as *mut u32).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut splits_u32 as *mut u32).cast(),
                (&mut segment_u32 as *mut u32).cast(),
            ];
            let stats_started = super::hip_api_stats::start();
            let status =
                unsafe { launch(functions.f32_gemv_splitk as *mut c_void, (rows as usize * splits).div_ceil(8) as u32, 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), split_arguments.as_mut_ptr(), ptr::null_mut()) };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, stats_started);
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "F32 split-K GEMV launch"));
            }
            let mut reduce_arguments = [(&mut partial_pointer as *mut *mut c_void).cast(), (&mut output_pointer as *mut *mut c_void).cast(), (&mut rows as *mut u32).cast(), (&mut splits_u32 as *mut u32).cast()];
            let stats_started = super::hip_api_stats::start();
            let status = unsafe { launch(functions.f32_gemv_splitk_reduce as *mut c_void, rows.div_ceil(8), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), reduce_arguments.as_mut_ptr(), ptr::null_mut()) };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, stats_started);
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "F32 split-K GEMV reduce launch"));
            }
        } else {
            let mut arguments =
                [(&mut input_pointer as *mut *mut c_void).cast(), (&mut weight_pointer as *mut *mut c_void).cast(), (&mut output_pointer as *mut *mut c_void).cast(), (&mut columns as *mut u32).cast(), (&mut rows as *mut u32).cast()];
            let stats_started = super::hip_api_stats::start();
            let status = unsafe { launch(functions.f32_gemv as *mut c_void, rows.div_ceil(8), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, stats_started);
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "F32 GEMV launch"));
            }
        }
    } else {
        for token_base in (0..input_rows).step_by(8) {
            let chunk_rows = (input_rows - token_base).min(8);
            input_pointer = unsafe { input.pointer.cast::<u8>().add(token_base * input_row_bytes).cast() };
            output_pointer = unsafe { output.pointer.cast::<u8>().add(token_base * output_row_bytes).cast() };
            let mut chunk_rows = u32::try_from(chunk_rows).expect("F32 GEMV chunk rows <= 8");
            let mut arguments = [
                (&mut input_pointer as *mut *mut c_void).cast(),
                (&mut weight_pointer as *mut *mut c_void).cast(),
                (&mut output_pointer as *mut *mut c_void).cast(),
                (&mut columns as *mut u32).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut chunk_rows as *mut u32).cast(),
            ];
            let stats_started = super::hip_api_stats::start();
            let status = unsafe { launch(functions.f32_gemv_small_n as *mut c_void, rows.div_ceil(8), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, stats_started);
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "F32 small-N GEMV launch"));
            }
        }
    }
    if profile_started.is_some() {
        synchronize_device(device_id, &format!("F32 GEMV synchronize m={columns} k={rows}"))?;
    }
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] f32-gemv device={device_id} n={input_rows_u32} m={columns} k={rows} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
}

/// 单 token、同输入的两路 resident F32 GEMV 合并成一次提交。
pub fn try_f32_dual_gemv_resident_f32(device_id: i32, input: &DeviceBuffer, first_weight: &DeviceBuffer, first_rows: usize, second_weight: &DeviceBuffer, second_rows: usize, columns: usize) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    if columns == 0 || first_rows == 0 || second_rows == 0 || input.device_id != device_id || first_weight.device_id != device_id || second_weight.device_id != device_id {
        return Err("ROCm dual F32 GEMV shape 为空或 device 不一致".to_owned());
    }
    if device_wavefront_size(device_id)? != 32 {
        return Err(format!("ROCm dual F32 GEMV kernel 仅支持 wave32,device={device_id}"));
    }
    let input_bytes = columns.checked_mul(4).ok_or("dual F32 GEMV input 大小溢出")?;
    let matrix_bytes = |rows: usize| rows.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or_else(|| "dual F32 GEMV weight 大小溢出".to_owned());
    let first_weight_bytes = matrix_bytes(first_rows)?;
    let second_weight_bytes = matrix_bytes(second_rows)?;
    if input.bytes < input_bytes || first_weight.bytes < first_weight_bytes || second_weight.bytes < second_weight_bytes {
        return Err(format!("ROCm dual F32 GEMV buffer 过小: input={}/{} weight={}/{}+{}/{}", input.bytes, input_bytes, first_weight.bytes, first_weight_bytes, second_weight.bytes, second_weight_bytes));
    }
    let first_output_bytes = first_rows.checked_mul(4).ok_or("dual F32 GEMV first output 大小溢出")?;
    let second_output_bytes = second_rows.checked_mul(4).ok_or("dual F32 GEMV second output 大小溢出")?;
    let owner = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, first_output_bytes.checked_add(second_output_bytes).ok_or("dual F32 GEMV output 大小溢出")?)?);
    let runtime = RocmRuntime::open()?;
    let functions = bf16_gemv_functions(device_id, runtime)?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let mut input_pointer = input.pointer;
    let mut first_weight_pointer = first_weight.pointer;
    let mut second_weight_pointer = second_weight.pointer;
    let mut output_pointer = owner.pointer;
    let mut columns = u32::try_from(columns).map_err(|_| "dual F32 GEMV columns 超过 u32")?;
    let mut first_rows_u32 = u32::try_from(first_rows).map_err(|_| "dual F32 GEMV first rows 超过 u32")?;
    let mut second_rows_u32 = u32::try_from(second_rows).map_err(|_| "dual F32 GEMV second rows 超过 u32")?;
    let combined_rows = first_rows_u32.checked_add(second_rows_u32).ok_or("dual F32 GEMV rows 超过 u32")?;
    let mut arguments = [
        (&mut input_pointer as *mut *mut c_void).cast(),
        (&mut first_weight_pointer as *mut *mut c_void).cast(),
        (&mut second_weight_pointer as *mut *mut c_void).cast(),
        (&mut output_pointer as *mut *mut c_void).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut first_rows_u32 as *mut u32).cast(),
        (&mut second_rows_u32 as *mut u32).cast(),
    ];
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let stats_started = super::hip_api_stats::start();
    let status = unsafe { launch(functions.f32_dual_gemv as *mut c_void, combined_rows.div_ceil(8), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, stats_started);
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "dual F32 GEMV launch"));
    }
    if profile_started.is_some() {
        synchronize_device(device_id, &format!("dual F32 GEMV synchronize m={columns} k={first_rows_u32}/{second_rows_u32}"))?;
    }
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] f32-dual-gemv device={device_id} m={columns} k={first_rows_u32}/{second_rows_u32} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    let first = DeviceBuffer::view(owner.clone(), 0, first_output_bytes)?;
    let second = DeviceBuffer::view(owner, first_output_bytes, second_output_bytes)?;
    Ok((first, second))
}

/// 单 token、同输入的两路 resident BF16 权重 GEMV 合并成一次提交。
pub fn try_bf16_dual_gemv_resident_f32(device_id: i32, input: &DeviceBuffer, first_weight: &DeviceBuffer, first_rows: usize, second_weight: &DeviceBuffer, second_rows: usize, columns: usize) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    if columns == 0 || first_rows == 0 || second_rows == 0 || input.device_id != device_id || first_weight.device_id != device_id || second_weight.device_id != device_id {
        return Err("ROCm dual BF16 GEMV shape 为空或 device 不一致".to_owned());
    }
    if device_wavefront_size(device_id)? != 32 {
        return Err(format!("ROCm dual BF16 GEMV kernel 仅支持 wave32,device={device_id}"));
    }
    if !bf16_gemv_columns_supported(columns) {
        return Err(format!("ROCm dual BF16 GEMV columns={columns} 的尾列无法完整覆盖"));
    }
    let input_bytes = columns.checked_mul(4).ok_or("dual BF16 GEMV input 大小溢出")?;
    if input.bytes < input_bytes {
        return Err(format!("ROCm dual BF16 GEMV input buffer 过小: {}/{}", input.bytes, input_bytes));
    }
    let matrix_bytes = |rows: usize| rows.checked_mul(columns).and_then(|n| n.checked_mul(2)).ok_or_else(|| "dual BF16 GEMV weight 大小溢出".to_owned());
    let first_weight_bytes = matrix_bytes(first_rows)?;
    let second_weight_bytes = matrix_bytes(second_rows)?;
    if first_weight.bytes < first_weight_bytes || second_weight.bytes < second_weight_bytes {
        return Err(format!("ROCm dual BF16 GEMV weight buffer 过小: {}/{}+{}/{}", first_weight.bytes, first_weight_bytes, second_weight.bytes, second_weight_bytes));
    }
    let first_output_bytes = first_rows.checked_mul(4).ok_or("dual BF16 GEMV first output 大小溢出")?;
    let second_output_bytes = second_rows.checked_mul(4).ok_or("dual BF16 GEMV second output 大小溢出")?;
    let owner = std::sync::Arc::new(DeviceBuffer::allocate_reusable(device_id, first_output_bytes.checked_add(second_output_bytes).ok_or("dual BF16 GEMV output 大小溢出")?)?);
    let runtime = RocmRuntime::open()?;
    let functions = bf16_gemv_functions(device_id, runtime)?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let mut input_pointer = input.pointer;
    let mut first_weight_pointer = first_weight.pointer;
    let mut second_weight_pointer = second_weight.pointer;
    let mut output_pointer = owner.pointer;
    let mut columns = u32::try_from(columns).map_err(|_| "dual BF16 GEMV columns 超过 u32")?;
    let mut first_rows_u32 = u32::try_from(first_rows).map_err(|_| "dual BF16 GEMV first rows 超过 u32")?;
    let mut second_rows_u32 = u32::try_from(second_rows).map_err(|_| "dual BF16 GEMV second rows 超过 u32")?;
    let combined_rows = first_rows_u32.checked_add(second_rows_u32).ok_or("dual BF16 GEMV rows 超过 u32")?;
    let mut arguments = [
        (&mut input_pointer as *mut *mut c_void).cast(),
        (&mut first_weight_pointer as *mut *mut c_void).cast(),
        (&mut second_weight_pointer as *mut *mut c_void).cast(),
        (&mut output_pointer as *mut *mut c_void).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut first_rows_u32 as *mut u32).cast(),
        (&mut second_rows_u32 as *mut u32).cast(),
    ];
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let stats_started = super::hip_api_stats::start();
    let status = unsafe { launch(functions.bf16_dual_gemv as *mut c_void, combined_rows.div_ceil(8), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
    super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, stats_started);
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "dual BF16 GEMV launch"));
    }
    if profile_started.is_some() {
        synchronize_device(device_id, &format!("dual BF16 GEMV synchronize m={columns} k={first_rows_u32}/{second_rows_u32}"))?;
    }
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] bf16-dual-gemv device={device_id} m={columns} k={first_rows_u32}/{second_rows_u32} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    let first = DeviceBuffer::view(owner.clone(), 0, first_output_bytes)?;
    let second = DeviceBuffer::view(owner, first_output_bytes, second_output_bytes)?;
    Ok((first, second))
}

fn bf16_gemv_columns_supported(columns: usize) -> bool {
    columns > 0 && columns % 4 <= 1
}

pub fn try_bf16_gemv_resident_f32(device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, input_rows: usize, columns: usize, output_rows: usize, approximate_bfdot: bool) -> Result<DeviceBuffer, String> {
    if input_rows == 0 || input.device_id != device_id || weight.device_id != device_id {
        return Err("ROCm BF16 GEMV 输入行为空或 device 不一致".to_owned());
    }
    // 主循环按 float4 向量化；尾循环能覆盖唯一余列，但余 2/3 列时会漏列。
    // columns=1 是 warmup 的合法标量矩阵，不能把它与真实的不完整尾部一起拒绝。
    if !bf16_gemv_columns_supported(columns) {
        return Err(format!("ROCm BF16 GEMV columns={columns} 的尾列无法完整覆盖，要求 columns % 4 为 0 或 1"));
    }
    // bf16_gemv_f32 按 wave32 硬编码 lane/wave 划分(threadIdx.x & 31,block 256 = 8 wave),
    // CDNA(wave64)上会算错,launch 前显式拒绝。
    let wavefront_size = device_wavefront_size(device_id)?;
    if wavefront_size != 32 {
        return Err(format!("ROCm BF16 GEMV kernel 仅支持 wave32,device={device_id} wavefront size={wavefront_size}"));
    }
    let input_elements = input_rows.checked_mul(columns).ok_or("BF16 GEMV input 大小溢出")?;
    let input_f32_bytes = input_elements.checked_mul(4).ok_or("BF16 GEMV F32 input 大小溢出")?;
    let input_bf16_bytes = input_elements.checked_mul(2).ok_or("BF16 GEMV BF16 input 大小溢出")?;
    let weight_bytes = output_rows.checked_mul(columns).and_then(|n| n.checked_mul(2)).ok_or("BF16 GEMV weight 大小溢出")?;
    let mut input_is_bf16 = match input.bytes {
        bytes if bytes == input_bf16_bytes => 1u32,
        bytes if bytes >= input_f32_bytes => 0u32,
        bytes => return Err(format!("ROCm BF16 GEMV buffer 过小: input={bytes} weight={}，期望 BF16/F32={input_bf16_bytes}/{input_f32_bytes} weight={weight_bytes}", weight.bytes)),
    };
    if weight.bytes < weight_bytes {
        return Err(format!("ROCm BF16 GEMV weight buffer 过小: actual={} expected={weight_bytes}", weight.bytes));
    }
    let output = DeviceBuffer::allocate_reusable(device_id, input_rows.checked_mul(output_rows).and_then(|n| n.checked_mul(4)).ok_or("BF16 GEMV output 大小溢出")?)?;
    let runtime = RocmRuntime::open()?;
    let functions = bf16_gemv_functions(device_id, runtime)?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let mut input_pointer = input.pointer;
    let mut weight_pointer = weight.pointer;
    let mut output_pointer = output.pointer;
    let mut columns = u32::try_from(columns).map_err(|_| "BF16 GEMV columns 超过 u32".to_owned())?;
    let mut rows = u32::try_from(output_rows).map_err(|_| "BF16 GEMV rows 超过 u32".to_owned())?;
    let mut input_rows = u32::try_from(input_rows).map_err(|_| "BF16 GEMV input rows 超过 u32".to_owned())?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let status = if input_rows == 1 {
        let mut arguments = [
            (&mut input_pointer as *mut *mut c_void).cast(),
            (&mut weight_pointer as *mut *mut c_void).cast(),
            (&mut output_pointer as *mut *mut c_void).cast(),
            (&mut columns as *mut u32).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut input_is_bf16 as *mut u32).cast(),
        ];
        let __hip_stats_started = super::hip_api_stats::start();
        let function = if approximate_bfdot && input_is_bf16 == 0 { functions.gemv_bfdot } else { functions.gemv };
        let __hip_launch_result = unsafe { launch(function as *mut c_void, rows.div_ceil(8), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    } else {
        let mut arguments = [
            (&mut input_pointer as *mut *mut c_void).cast(),
            (&mut weight_pointer as *mut *mut c_void).cast(),
            (&mut output_pointer as *mut *mut c_void).cast(),
            (&mut columns as *mut u32).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut input_rows as *mut u32).cast(),
            (&mut input_is_bf16 as *mut u32).cast(),
        ];
        let __hip_stats_started = super::hip_api_stats::start();
        let (function, row_group) = if input_rows == 2 {
            (functions.gemv_rows2, 2)
        } else if input_rows == 3 {
            (functions.gemv_rows3, 3)
        } else if input_rows <= 4 {
            (functions.gemv_rows4, 4)
        } else {
            (if approximate_bfdot && input_is_bf16 == 0 { functions.gemv_rows8_bfdot } else { functions.gemv_rows8 }, 8)
        };
        let __hip_launch_result = unsafe { launch(function as *mut c_void, rows.div_ceil(8), input_rows.div_ceil(row_group), 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "BF16 GEMV launch"));
    }
    if options().kernel_sync || profile_started.is_some() {
        synchronize_device(device_id, &format!("BF16 GEMV synchronize n={input_rows} m={columns} k={rows}"))?;
    }
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] bf16-gemv device={device_id} n={input_rows} m={columns} k={rows} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
}

#[cfg(test)]
mod dense_tile_tests {
    use super::*;

    #[test]
    fn bf16_gemv_column_tail_contract() {
        for columns in [1, 4, 5, 128, 129] {
            assert!(bf16_gemv_columns_supported(columns), "columns={columns}");
        }
        for columns in [0, 2, 3, 6, 7, 130, 131] {
            assert!(!bf16_gemv_columns_supported(columns), "columns={columns}");
        }
    }

    fn f32_bytes(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn bf16_value(value: f32) -> f32 {
        f32::from_bits((bf16(value) as u32) << 16)
    }

    #[test]
    fn rocm_f32_gemv_matches_cpu_oracle() {
        if !super::super::is_hip_available() {
            eprintln!("[f32-gemv] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let columns = 257usize;
        let output_rows = 37usize;
        let input_rows = 5usize;
        let input = (0..input_rows * columns).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
        let weight = (0..output_rows * columns).map(|index| (index as f32 * 0.019).cos()).collect::<Vec<_>>();
        let actual = try_f32_gemv_resident_f32(0, &DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload F32 GEMV input"), &DeviceBuffer::upload(0, f32_bytes(&weight)).expect("upload F32 GEMV weight"), input_rows, columns, output_rows)
            .expect("F32 GEMV")
            .download_f32(input_rows * output_rows)
            .expect("download F32 GEMV");
        for token in 0..input_rows {
            for row in 0..output_rows {
                let expected = input[token * columns..(token + 1) * columns].iter().zip(&weight[row * columns..(row + 1) * columns]).map(|(&x, &w)| x * w).sum::<f32>();
                let actual = actual[token * output_rows + row];
                assert!((actual - expected).abs() <= expected.abs() * 1.0e-4 + 1.0e-4, "token={token} row={row} actual={actual} expected={expected}");
            }
        }
    }

    /// rows<=32 且 columns>=2048 走 split-K 路径;两组 shape 覆盖段边界尾循环。
    #[test]
    fn rocm_f32_gemv_splitk_matches_cpu_oracle() {
        if !super::super::is_hip_available() {
            eprintln!("[f32-gemv-splitk] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        for (columns, output_rows) in [(16384usize, 24usize), (2053usize, 7usize)] {
            let input = (0..columns).map(|index| (index as f32 * 0.011).sin()).collect::<Vec<_>>();
            let weight = (0..output_rows * columns).map(|index| (index as f32 * 0.017).cos()).collect::<Vec<_>>();
            let actual =
                try_f32_gemv_resident_f32(0, &DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload F32 split-K input"), &DeviceBuffer::upload(0, f32_bytes(&weight)).expect("upload F32 split-K weight"), 1, columns, output_rows)
                    .expect("F32 split-K GEMV")
                    .download_f32(output_rows)
                    .expect("download F32 split-K GEMV");
            for row in 0..output_rows {
                let expected = input.iter().zip(&weight[row * columns..(row + 1) * columns]).map(|(&x, &w)| x * w).sum::<f32>();
                let actual = actual[row];
                assert!((actual - expected).abs() <= expected.abs() * 1.0e-4 + 1.0e-4, "columns={columns} row={row} actual={actual} expected={expected}");
            }
        }
    }

    #[test]
    fn rocm_f32_dual_gemv_matches_single_outputs_bitwise() {
        if !super::super::is_hip_available() {
            eprintln!("[f32-dual-gemv] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let columns = 257usize;
        let first_rows = 37usize;
        let second_rows = 19usize;
        let input = (0..columns).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
        let first_weight = (0..first_rows * columns).map(|index| (index as f32 * 0.019).cos()).collect::<Vec<_>>();
        let second_weight = (0..second_rows * columns).map(|index| (index as f32 * 0.023).sin()).collect::<Vec<_>>();
        let input_device = DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload dual F32 GEMV input");
        let first_weight_device = DeviceBuffer::upload(0, f32_bytes(&first_weight)).expect("upload first F32 GEMV weight");
        let second_weight_device = DeviceBuffer::upload(0, f32_bytes(&second_weight)).expect("upload second F32 GEMV weight");
        let first_expected = try_f32_gemv_resident_f32(0, &input_device, &first_weight_device, 1, columns, first_rows).expect("first F32 GEMV").download_f32(first_rows).expect("download first F32 GEMV");
        let second_expected = try_f32_gemv_resident_f32(0, &input_device, &second_weight_device, 1, columns, second_rows).expect("second F32 GEMV").download_f32(second_rows).expect("download second F32 GEMV");
        let (first, second) = try_f32_dual_gemv_resident_f32(0, &input_device, &first_weight_device, first_rows, &second_weight_device, second_rows, columns).expect("dual F32 GEMV");
        assert_eq!(first.download_f32(first_rows).expect("download dual first F32 GEMV"), first_expected);
        assert_eq!(second.download_f32(second_rows).expect("download dual second F32 GEMV"), second_expected);
    }

    #[test]
    fn rocm_bf16_dual_gemv_matches_single_outputs_bitwise() {
        if !super::super::is_hip_available() {
            eprintln!("[bf16-dual-gemv] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let columns = 257usize;
        let first_rows = 37usize;
        let second_rows = 19usize;
        let input = (0..columns).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
        let first_weight = (0..first_rows * columns).flat_map(|index| bf16((index as f32 * 0.019).cos()).to_ne_bytes()).collect::<Vec<_>>();
        let second_weight = (0..second_rows * columns).flat_map(|index| bf16((index as f32 * 0.023).sin()).to_ne_bytes()).collect::<Vec<_>>();
        let input_device = DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload dual BF16 GEMV input");
        let first_weight_device = DeviceBuffer::upload(0, &first_weight).expect("upload first BF16 GEMV weight");
        let second_weight_device = DeviceBuffer::upload(0, &second_weight).expect("upload second BF16 GEMV weight");
        let first_expected = try_bf16_gemv_resident_f32(0, &input_device, &first_weight_device, 1, columns, first_rows, false).expect("first BF16 GEMV").download_f32(first_rows).expect("download first BF16 GEMV");
        let second_expected = try_bf16_gemv_resident_f32(0, &input_device, &second_weight_device, 1, columns, second_rows, false).expect("second BF16 GEMV").download_f32(second_rows).expect("download second BF16 GEMV");
        let (first, second) = try_bf16_dual_gemv_resident_f32(0, &input_device, &first_weight_device, first_rows, &second_weight_device, second_rows, columns).expect("dual BF16 GEMV");
        assert_eq!(first.download_f32(first_rows).expect("download dual first BF16 GEMV"), first_expected);
        assert_eq!(second.download_f32(second_rows).expect("download dual second BF16 GEMV"), second_expected);
    }

    #[test]
    fn rocm_bf16_batched_gemv_matches_single_rows_bitwise() {
        let columns = 256usize;
        let output_rows = 96usize;
        let weight = (0..output_rows * columns).flat_map(|index| bf16((index as f32 * 0.019).cos()).to_ne_bytes()).collect::<Vec<_>>();
        let weight_device = DeviceBuffer::upload(0, &weight).expect("upload batched GEMV weight");
        for rows in [2usize, 3, 4, 5, 8, 9, 16] {
            let input = (0..rows * columns).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
            let input_device = std::sync::Arc::new(DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload batched GEMV input"));
            let actual = try_bf16_gemv_resident_f32(0, &input_device, &weight_device, rows, columns, output_rows, false).expect("batched GEMV").download_f32(rows * output_rows).expect("download batched GEMV");
            for row in 0..rows {
                let input_row = DeviceBuffer::view(input_device.clone(), row * columns * 4, columns * 4).expect("view GEMV row");
                let expected = try_bf16_gemv_resident_f32(0, &input_row, &weight_device, 1, columns, output_rows, false).expect("single GEMV").download_f32(output_rows).expect("download single GEMV");
                assert_eq!(&actual[row * output_rows..(row + 1) * output_rows], expected, "rows={rows} row={row}");
            }

            let quantized_input = input.iter().map(|&value| bf16_value(value)).collect::<Vec<_>>();
            let input_bf16 = input.iter().flat_map(|&value| bf16(value).to_ne_bytes()).collect::<Vec<_>>();
            let expected = try_bf16_gemv_resident_f32(0, &DeviceBuffer::upload(0, f32_bytes(&quantized_input)).expect("upload F32 GEMV input"), &weight_device, rows, columns, output_rows, false)
                .expect("F32 GEMV")
                .download_f32(rows * output_rows)
                .expect("download F32 GEMV");
            let actual = try_bf16_gemv_resident_f32(0, &DeviceBuffer::upload(0, &input_bf16).expect("upload BF16 GEMV input"), &weight_device, rows, columns, output_rows, false)
                .expect("BF16 input GEMV")
                .download_f32(rows * output_rows)
                .expect("download BF16 input GEMV");
            assert_eq!(actual, expected, "rows={rows}");
        }
    }

    #[test]
    fn rocm_w8_g128_rows2_matches_single_rows_bitwise() {
        if !super::super::is_hip_available() {
            eprintln!("[w8-rows2] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let rows = 2usize;
        let columns = 6_144usize;
        let output_rows = 96usize;
        let input = (0..rows * columns).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
        let packed = (0..output_rows * columns).map(|index| ((index * 29 + 17) % 255 + 1) as u8).collect::<Vec<_>>();
        let scales = (0..output_rows * (columns / 128)).flat_map(|index| bf16(2.0f32.powi(-8 + (index % 3) as i32)).to_ne_bytes()).collect::<Vec<_>>();
        let packed = DeviceBuffer::upload(0, &packed).expect("upload W8 rows2 packed");
        let scales = DeviceBuffer::upload(0, &scales).expect("upload W8 rows2 scales");
        let actual = try_ct_quantized_matmul_bf16(0, 8, &input, None, &packed, &scales, 0, 128, rows, columns, output_rows).expect("W8 rows2").download_f32(rows * output_rows).expect("download W8 rows2");
        for row in 0..rows {
            let expected =
                try_ct_quantized_matmul_bf16(0, 8, &input[row * columns..(row + 1) * columns], None, &packed, &scales, 0, 128, 1, columns, output_rows).expect("W8 single row").download_f32(output_rows).expect("download W8 single row");
            assert_eq!(&actual[row * output_rows..(row + 1) * output_rows], expected, "row={row}");
        }
    }

    #[test]
    fn rocm_w8_g128_shared_rows_match_single_rows_bitwise() {
        if !super::super::is_hip_available() {
            eprintln!("[w8-shared-rows] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let columns = 6_144usize;
        let output_rows = 96usize;
        let packed = (0..output_rows * columns).map(|index| ((index * 29 + 17) % 255 + 1) as u8).collect::<Vec<_>>();
        let scales = (0..output_rows * (columns / 128)).flat_map(|index| bf16(2.0f32.powi(-8 + (index % 3) as i32)).to_ne_bytes()).collect::<Vec<_>>();
        let packed = DeviceBuffer::upload(0, &packed).expect("upload W8 shared rows packed");
        let scales = DeviceBuffer::upload(0, &scales).expect("upload W8 shared rows scales");
        for rows in 3..=8 {
            let input = (0..rows * columns).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
            let actual = try_ct_quantized_matmul_bf16(0, 8, &input, None, &packed, &scales, 0, 128, rows, columns, output_rows).expect("W8 shared rows").download_f32(rows * output_rows).expect("download W8 shared rows");
            for row in 0..rows {
                let expected =
                    try_ct_quantized_matmul_bf16(0, 8, &input[row * columns..(row + 1) * columns], None, &packed, &scales, 0, 128, 1, columns, output_rows).expect("W8 single row").download_f32(output_rows).expect("download W8 single row");
                assert_eq!(&actual[row * output_rows..(row + 1) * output_rows], expected, "rows={rows} row={row}");
            }
        }
    }

    #[test]
    fn rocm_w4_g128_single_row_matches_cpu_oracle() {
        if !super::super::is_hip_available() {
            eprintln!("[w4-g128] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let columns = 512usize;
        let output_rows = 37usize;
        let groups = columns / 128;
        let input = (0..columns).map(|index| (index as f32 * 0.013).sin() * 0.125).collect::<Vec<_>>();
        let code = |index: usize| ((index * 7 + 3) % 16) as u8;
        let packed = (0..output_rows * columns / 2).map(|index| code(index * 2) | (code(index * 2 + 1) << 4)).collect::<Vec<_>>();
        let scales = (0..output_rows * groups).flat_map(|index| bf16(2.0f32.powi(-7 + (index % 4) as i32)).to_ne_bytes()).collect::<Vec<_>>();
        let packed = DeviceBuffer::upload(0, &packed).expect("upload W4 packed");
        let scales = DeviceBuffer::upload(0, &scales).expect("upload W4 scales");
        let actual = try_ct_quantized_matmul_bf16(0, 4, &input, None, &packed, &scales, 0, 128, 1, columns, output_rows).expect("ROCm W4G128 GEMV").download_f32(output_rows).expect("download W4G128 GEMV");
        for (row, &actual) in actual.iter().enumerate() {
            let expected = (0..columns)
                .map(|column| {
                    let scale = 2.0f32.powi(-7 + ((row * groups + column / 128) % 4) as i32);
                    bf16_value(input[column]) * f32::from(i16::from(code(row * columns + column)) - 8) * scale
                })
                .sum::<f32>();
            assert!((actual - expected).abs() <= expected.abs() * 2.0e-4 + 2.0e-4, "row={row} actual={actual} expected={expected}");
        }
    }

    #[test]
    fn rocm_w4_g128_rows8_matches_single_rows_bitwise() {
        if !super::super::is_hip_available() {
            eprintln!("[w4-rows8] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let columns = 6_144usize;
        let output_rows = 96usize;
        for rows in [2usize, 4, 8] {
            let input = (0..rows * columns).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
            let code = |index: usize| ((index * 7 + 3) % 16) as u8;
            let packed = (0..output_rows * columns / 2).map(|index| code(index * 2) | (code(index * 2 + 1) << 4)).collect::<Vec<_>>();
            let scales = (0..output_rows * (columns / 128)).flat_map(|index| bf16(2.0f32.powi(-8 + (index % 3) as i32)).to_ne_bytes()).collect::<Vec<_>>();
            let packed = DeviceBuffer::upload(0, &packed).expect("upload W4 rows8 packed");
            let scales = DeviceBuffer::upload(0, &scales).expect("upload W4 rows8 scales");
            let actual = try_ct_quantized_matmul_bf16(0, 4, &input, None, &packed, &scales, 0, 128, rows, columns, output_rows).expect("W4 rows8").download_f32(rows * output_rows).expect("download W4 rows8");
            for row in 0..rows {
                let expected =
                    try_ct_quantized_matmul_bf16(0, 4, &input[row * columns..(row + 1) * columns], None, &packed, &scales, 0, 128, 1, columns, output_rows).expect("W4 single row").download_f32(output_rows).expect("download W4 single row");
                assert_eq!(&actual[row * output_rows..(row + 1) * output_rows], expected, "rows={rows} row={row}");
            }
        }
    }

    #[test]
    fn rocm_w4_g128_dual_matches_single_outputs_bitwise() {
        if !super::super::is_hip_available() {
            eprintln!("[w4-dual] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let columns = 512usize;
        let first_rows = 37usize;
        let second_rows = 19usize;
        let groups = columns / 128;
        let input = (0..columns).flat_map(|index| bf16((index as f32 * 0.013).sin()).to_ne_bytes()).collect::<Vec<_>>();
        let packed = |rows: usize, seed: usize| (0..rows * columns / 2).map(|index| (((index * 2 + seed) % 16) | (((index * 2 + seed + 5) % 16) << 4)) as u8).collect::<Vec<_>>();
        let scales = |rows: usize, seed: usize| (0..rows * groups).flat_map(|index| bf16(2.0f32.powi(-7 + ((index + seed) % 4) as i32)).to_ne_bytes()).collect::<Vec<_>>();
        let input = DeviceBuffer::upload(0, &input).expect("upload dual W4 BF16 input");
        let first_packed = DeviceBuffer::upload(0, &packed(first_rows, 3)).expect("upload first W4 packed");
        let first_scales = DeviceBuffer::upload(0, &scales(first_rows, 1)).expect("upload first W4 scales");
        let second_packed = DeviceBuffer::upload(0, &packed(second_rows, 7)).expect("upload second W4 packed");
        let second_scales = DeviceBuffer::upload(0, &scales(second_rows, 2)).expect("upload second W4 scales");
        let first_expected = try_ct_quantized_matmul_bf16(0, 4, &[], Some(&input), &first_packed, &first_scales, 0, 128, 1, columns, first_rows).expect("first W4 GEMV").download_f32(first_rows).expect("download first W4 GEMV");
        let second_expected = try_ct_quantized_matmul_bf16(0, 4, &[], Some(&input), &second_packed, &second_scales, 0, 128, 1, columns, second_rows).expect("second W4 GEMV").download_f32(second_rows).expect("download second W4 GEMV");
        let (first, second) = try_ct_dual_gemv_bf16(0, 4, &[], Some(&input), columns, 1, &first_packed, &first_scales, 0, 128, first_rows, &second_packed, &second_scales, 0, 128, second_rows).expect("dual W4 GEMV");
        assert_eq!(first.download_f32(first_rows).expect("download dual first W4"), first_expected);
        assert_eq!(second.download_f32(second_rows).expect("download dual second W4"), second_expected);

        let input_rows = 3usize;
        let input = (0..input_rows * columns).flat_map(|index| bf16((index as f32 * 0.009).cos()).to_ne_bytes()).collect::<Vec<_>>();
        let input = DeviceBuffer::upload(0, &input).expect("upload dual W4 B3 input");
        let first_expected =
            try_ct_quantized_matmul_bf16(0, 4, &[], Some(&input), &first_packed, &first_scales, 0, 128, input_rows, columns, first_rows).expect("first W4 B3 GEMV").download_f32(input_rows * first_rows).expect("download first W4 B3 GEMV");
        let second_expected = try_ct_quantized_matmul_bf16(0, 4, &[], Some(&input), &second_packed, &second_scales, 0, 128, input_rows, columns, second_rows)
            .expect("second W4 B3 GEMV")
            .download_f32(input_rows * second_rows)
            .expect("download second W4 B3 GEMV");
        let (first, second) = try_ct_dual_gemv_bf16(0, 4, &[], Some(&input), columns, input_rows, &first_packed, &first_scales, 0, 128, first_rows, &second_packed, &second_scales, 0, 128, second_rows).expect("dual W4 B3 GEMV");
        assert_eq!(first.download_f32(input_rows * first_rows).expect("download dual first W4 B3"), first_expected);
        assert_eq!(second.download_f32(input_rows * second_rows).expect("download dual second W4 B3"), second_expected);
    }

    #[test]
    fn rocm_dual_w8_accepts_bf16_input() {
        let columns = 128usize;
        let output_rows = 32usize;
        let input = vec![bf16(1.0).to_ne_bytes(); columns].into_iter().flatten().collect::<Vec<_>>();
        let packed = vec![128_u8; output_rows * columns];
        let scales = vec![bf16(1.0).to_ne_bytes(); output_rows].into_iter().flatten().collect::<Vec<_>>();
        let input = DeviceBuffer::upload(0, &input).expect("upload dual W8 BF16 input");
        let packed = DeviceBuffer::upload(0, &packed).expect("upload dual W8 packed");
        let scales = DeviceBuffer::upload(0, &scales).expect("upload dual W8 scales");
        let (first, second) = try_ct_dual_gemv_bf16(0, 8, &[], Some(&input), columns, 1, &packed, &scales, 0, 128, output_rows, &packed, &scales, 0, 128, output_rows).expect("dual W8 BF16 input");

        assert_eq!(first.download_f32(output_rows).unwrap(), vec![0.0; output_rows]);
        assert_eq!(second.download_f32(output_rows).unwrap(), vec![0.0; output_rows]);
        let scales64 = vec![bf16(1.0).to_ne_bytes(); output_rows * 2].into_iter().flatten().collect::<Vec<_>>();
        let scales64 = DeviceBuffer::upload(0, &scales64).expect("upload dual W8 G64 scales");
        let (first, second) = try_ct_dual_gemv_bf16(0, 8, &[], Some(&input), columns, 1, &packed, &scales, 0, 128, output_rows, &packed, &scales64, 0, 64, output_rows).expect("dual W8 fallback BF16 input");
        assert_eq!(first.download_f32(output_rows).unwrap(), vec![0.0; output_rows]);
        assert_eq!(second.download_f32(output_rows).unwrap(), vec![0.0; output_rows]);
    }

    #[test]
    fn rocm_decode_moe_fused_epilogue_matches_unfused() {
        if !super::super::is_hip_available() {
            eprintln!("[decode-moe-epilogue] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let input_rows = 4usize;
        let hidden_size = 64usize;
        let intermediate_size = 64usize;
        let expert_count = 8usize;
        let top_k = 2usize;
        let route_count = input_rows * top_k;
        let input = (0..input_rows * hidden_size).map(|index| (index as f32 * 0.001).sin() * 0.01).collect::<Vec<_>>();
        let packed = vec![0x99999999_u32; intermediate_size * hidden_size / 8];
        let scales = vec![bf16(0.01).to_ne_bytes(); intermediate_size].into_iter().flatten().collect::<Vec<_>>();
        let packed = upload_pod(0, &packed).expect("upload decode MoE W4");
        let scales = DeviceBuffer::upload(0, &scales).expect("upload decode MoE scale");
        let weight = CtGroupedWeightMeta { packed: packed.pointer as usize as u64, scales: scales.pointer as usize as u64, group_size: hidden_size as u32, scale_dtype: 0, format: 0 };
        let metas = vec![CtGroupedExpertMeta { gate: weight, up: weight, down: weight }; expert_count];
        let metas = upload_pod(0, &metas).expect("upload decode MoE metas");
        let route_ids = upload_pod(0, &(0..route_count as u32).collect::<Vec<_>>()).expect("upload decode MoE routes");
        let route_weights = upload_pod(0, &vec![0.5_f32; route_count]).expect("upload decode MoE route weights");
        let input_bf16_values = input.iter().flat_map(|&value| bf16(value).to_ne_bytes()).collect::<Vec<_>>();
        let input = DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload decode MoE input");
        let input_bf16 = DeviceBuffer::upload(0, &input_bf16_values).expect("upload decode MoE BF16 input");
        let input_workspace = DeviceBuffer::allocate(0, input_rows * hidden_size * 2).expect("allocate decode MoE BF16 workspace");
        let activated = DeviceBuffer::allocate(0, route_count * intermediate_size * 2).expect("allocate decode MoE activated");
        let plain = launch_ct_decode_experts_bf16(0, &input, input_rows, hidden_size, intermediate_size, &route_ids, &route_weights, route_count, expert_count, &metas, &input_workspace, &activated, None, None, false)
            .expect("decode MoE without epilogue")
            .download_f32(input_rows * hidden_size)
            .expect("download decode MoE output");
        let bf16_plain = launch_ct_decode_experts_bf16(0, &input_bf16, input_rows, hidden_size, intermediate_size, &route_ids, &route_weights, route_count, expert_count, &metas, &input_workspace, &activated, None, None, false)
            .expect("decode MoE with BF16 input")
            .download_f32(input_rows * hidden_size)
            .expect("download BF16-input decode MoE output");
        assert_eq!(bf16_plain, plain);

        let shared = (0..input_rows * hidden_size).map(|index| index as f32 * 0.0001).collect::<Vec<_>>();
        let residual = (0..input_rows * hidden_size).map(|index| index as f32 * -0.0002).collect::<Vec<_>>();
        let shared_device = DeviceBuffer::upload(0, f32_bytes(&shared)).expect("upload decode MoE shared");
        let residual_device = DeviceBuffer::upload(0, f32_bytes(&residual)).expect("upload decode MoE residual");
        let fused = launch_ct_decode_experts_bf16(
            0,
            &input_bf16,
            input_rows,
            hidden_size,
            intermediate_size,
            &route_ids,
            &route_weights,
            route_count,
            expert_count,
            &metas,
            &input_workspace,
            &activated,
            Some((&shared_device, &residual_device)),
            None,
            false,
        )
        .expect("decode MoE fused epilogue")
        .download_f32(input_rows * hidden_size)
        .expect("download decode MoE fused output");
        for (index, ((&fused, &plain), (&shared, &residual))) in fused.iter().zip(&plain).zip(shared.iter().zip(&residual)).enumerate() {
            assert_eq!(fused.to_bits(), (residual + (shared + plain)).to_bits(), "index={index}");
        }

        let shared_routes = upload_pod(0, &vec![0_u32; input_rows]).expect("upload shared expert routes");
        let shared_weights = upload_pod(0, &vec![1.0_f32; input_rows]).expect("upload shared expert weights");
        let shared_metas = upload_pod(0, &[CtGroupedExpertMeta { gate: weight, up: weight, down: weight }]).expect("upload shared expert meta");
        let shared_activated = DeviceBuffer::allocate(0, input_rows * intermediate_size * 2).expect("allocate shared expert activated");
        let shared_plain = launch_ct_decode_experts_bf16(0, &input_bf16, input_rows, hidden_size, intermediate_size, &shared_routes, &shared_weights, input_rows, 1, &shared_metas, &input_workspace, &shared_activated, None, None, false)
            .expect("decode standalone shared expert")
            .download_f32(input_rows * hidden_size)
            .expect("download standalone shared expert");
        let integrated_metas = upload_pod(0, &vec![CtGroupedExpertMeta { gate: weight, up: weight, down: weight }; expert_count + 1]).expect("upload integrated expert metas");
        let integrated_activated = DeviceBuffer::allocate(0, (route_count + input_rows) * intermediate_size * 2).expect("allocate integrated expert activated");
        let integrated = launch_ct_decode_experts_bf16(
            0,
            &input_bf16,
            input_rows,
            hidden_size,
            intermediate_size,
            &route_ids,
            &route_weights,
            route_count,
            expert_count,
            &integrated_metas,
            &input_workspace,
            &integrated_activated,
            None,
            Some((expert_count, &residual_device)),
            false,
        )
        .expect("decode integrated shared expert")
        .download_f32(input_rows * hidden_size)
        .expect("download integrated shared expert");
        for (index, ((&integrated, &routed), (&shared, &residual))) in integrated.iter().zip(&plain).zip(shared_plain.iter().zip(&residual)).enumerate() {
            assert_eq!(integrated.to_bits(), (residual + (shared + routed)).to_bits(), "integrated index={index}");
        }
    }

    #[test]
    fn rocm_decode_w8_experts_matches_cpu() {
        if !super::super::is_hip_available() {
            eprintln!("[decode-w8-moe] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        struct Weight {
            codes: Vec<i8>,
            scale: f32,
            packed: DeviceBuffer,
            scales: DeviceBuffer,
        }
        impl Weight {
            fn new(rows: usize, columns: usize, seed: usize) -> Self {
                let codes = (0..rows * columns).map(|index| ((index * 13 + seed * 7) % 7) as i8 - 3).collect::<Vec<_>>();
                let scale = 0.015625;
                let packed = DeviceBuffer::upload(0, &codes.iter().map(|&value| (i16::from(value) + 128) as u8).collect::<Vec<_>>()).expect("upload W8 expert codes");
                let scales = DeviceBuffer::upload(0, &vec![bf16(scale).to_ne_bytes(); rows].into_iter().flatten().collect::<Vec<_>>()).expect("upload W8 expert scales");
                Self { codes, scale, packed, scales }
            }

            fn reference(&self, input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
                (0..rows).map(|row| input.iter().enumerate().map(|(column, &value)| value * self.codes[row * columns + column] as f32 * self.scale).sum()).collect()
            }

            fn grouped(&self, group_size: usize) -> CtGroupedWeightRef<'_> {
                CtGroupedWeightRef { packed: &self.packed, scales: &self.scales, scale_dtype: 0, group_size, format: 0 }
            }
        }
        struct Expert {
            gate: Weight,
            up: Weight,
            down: Weight,
        }
        let hidden = 64usize;
        let intermediate = 32usize;
        let experts = (0..2).map(|expert| Expert { gate: Weight::new(intermediate, hidden, expert * 3), up: Weight::new(intermediate, hidden, expert * 3 + 1), down: Weight::new(hidden, intermediate, expert * 3 + 2) }).collect::<Vec<_>>();
        let input = (0..hidden).map(|index| bf16_value((index as f32 * 0.071).sin() * 0.25)).collect::<Vec<_>>();
        let input_bf16 = input.iter().flat_map(|&value| bf16(value).to_ne_bytes()).collect::<Vec<_>>();
        let input_device = DeviceBuffer::upload(0, &input_bf16).expect("upload W8 expert input");
        let routes = [1_u32, 0];
        let route_weights = [0.25_f32, 0.75];
        let route_ids_device = upload_pod(0, &routes).expect("upload W8 expert routes");
        let route_weights_device = upload_pod(0, &route_weights).expect("upload W8 expert route weights");
        let grouped = experts.iter().map(|expert| CtGroupedExpertRef { gate: expert.gate.grouped(hidden), up: expert.up.grouped(hidden), down: expert.down.grouped(intermediate) }).collect::<Vec<_>>();
        let actual =
            try_ct_w8_decode_experts_bf16(0, &input_device, hidden, intermediate, &route_ids_device, &route_weights_device, routes.len(), &grouped).expect("W8 decode experts").download_f32(hidden).expect("download W8 decode experts");
        let mut expected = vec![0.0_f32; hidden];
        for (&expert, &route_weight) in routes.iter().zip(&route_weights) {
            let expert = &experts[expert as usize];
            let gate = expert.gate.reference(&input, intermediate, hidden);
            let up = expert.up.reference(&input, intermediate, hidden);
            let activated = gate.iter().zip(up).map(|(&gate, up)| bf16_value(gate / (1.0 + (-gate).exp()) * up)).collect::<Vec<_>>();
            let down = expert.down.reference(&activated, hidden, intermediate);
            for (expected, down) in expected.iter_mut().zip(down) {
                *expected += down * route_weight;
            }
        }
        for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() <= 2.0e-5, "index={index} actual={actual} expected={expected}");
        }
    }

    #[test]
    fn rocm_grouped_official_fp8_matches_cpu_with_288_experts() {
        if !super::super::is_hip_available() {
            eprintln!("[grouped-official-fp8] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        struct Weight {
            decoded: Vec<f32>,
            codes: DeviceBuffer,
            scales: DeviceBuffer,
        }
        impl Weight {
            fn new(rows: usize, columns: usize, seed: usize) -> Self {
                let values = [0x00_u8, 0x20, 0x28, 0x30, 0x38, 0x40, 0xa8, 0xb0, 0xb8];
                let codes = (0..rows * columns).map(|index| values[(index * 5 + seed) % values.len()]).collect::<Vec<_>>();
                // 二次幂 scale 使 E4M3 解码值可由 BF16 精确表示，down 的
                // scalar/WMMA 调度选择不会改变 oracle 语义。
                let scale_values = vec![0.015625_f32; rows.div_ceil(128) * columns.div_ceil(128)];
                let scale_bytes = scale_values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
                let mut decoded = vec![0.0_f32; rows * columns];
                crate::weight::codec::fp8::decode_fp8_matrix(&codes, &scale_bytes, rows, columns, &mut decoded);
                let codes_device = DeviceBuffer::upload(0, &codes).expect("upload official FP8 codes");
                let scales = DeviceBuffer::upload(0, &scale_bytes).expect("upload official FP8 scales");
                Self { decoded, codes: codes_device, scales }
            }

            fn grouped(&self) -> CtGroupedWeightRef<'_> {
                CtGroupedWeightRef { packed: &self.codes, scales: &self.scales, scale_dtype: 2, group_size: 128, format: 1 }
            }

            fn project(&self, input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
                (0..rows).map(|row| (0..columns).map(|column| input[column] * self.decoded[row * columns + column]).sum()).collect()
            }
        }

        // expert 0 超过一个 128-row WMMA tile，其余 route 分散为 singleton，
        // 同时覆盖 dynamic persistent tile 和紧凑 small-expert 补算路径。
        let (rows, hidden, intermediate, expert_count, top_k) = (144usize, 128usize, 128usize, 288usize, 2usize);
        let gate = Weight::new(intermediate, hidden, 0);
        let up = Weight::new(intermediate, hidden, 1);
        let down = Weight::new(hidden, intermediate, 2);
        let grouped = (0..expert_count).map(|_| CtGroupedExpertRef { gate: gate.grouped(), up: up.grouped(), down: down.grouped() }).collect::<Vec<_>>();
        let input = (0..rows * hidden).map(|index| (index as f32 * 0.031).sin() * 0.125).collect::<Vec<_>>();
        let route_ids = (0..rows * top_k).map(|route| if route % top_k == 0 { 0_u32 } else { 1 + (route / top_k % 287) as u32 }).collect::<Vec<_>>();
        let route_weights = (0..rows * top_k).map(|route| if route % top_k == 0 { 0.25_f32 } else { 0.75_f32 }).collect::<Vec<_>>();
        let input_device = DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload official FP8 input");
        let route_ids_device = upload_pod(0, &route_ids).expect("upload official FP8 route ids");
        let route_weights_device = upload_pod(0, &route_weights).expect("upload official FP8 route weights");
        let actual = try_ct_grouped_experts_bf16(0, &input_device, rows, hidden, intermediate, &[], &[], &[], &grouped, Some((&route_ids_device, &route_weights_device, route_ids.len())), None, None)
            .expect("official FP8 grouped experts")
            .download_f32(rows * hidden)
            .expect("download official FP8 grouped output");

        let mut expected = vec![0.0_f32; rows * hidden];
        for route in 0..rows * top_k {
            let token = route / top_k;
            let input = input[token * hidden..(token + 1) * hidden].iter().map(|&value| bf16_value(value)).collect::<Vec<_>>();
            let gate_output = gate.project(&input, intermediate, hidden);
            let up_output = up.project(&input, intermediate, hidden);
            let activated = gate_output.iter().zip(up_output).map(|(&gate, up)| bf16_value(gate / (1.0 + (-gate).exp()) * up)).collect::<Vec<_>>();
            let projected = down.project(&activated, hidden, intermediate);
            for column in 0..hidden {
                expected[token * hidden + column] += projected[column] * route_weights[route];
            }
        }
        for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() <= 3.0e-4 + expected.abs() * 3.0e-3, "index={index} actual={actual} expected={expected}");
        }
    }

    #[test]
    fn rocm_dense_tile_matches_cpu() {
        let rows = 17usize;
        let columns = 32usize;
        let output_rows = 48usize;
        let input = (0..rows * columns).map(|index| (index as f32 * 0.031).sin()).collect::<Vec<_>>();
        let weight = (0..output_rows * columns).map(|index| (index as f32 * 0.017).cos()).collect::<Vec<_>>();
        let weight_bf16 = weight.iter().flat_map(|&value| bf16(value).to_ne_bytes()).collect::<Vec<_>>();
        let actual = try_dense_matmul_bf16_f32(0, &DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload dense input"), &DeviceBuffer::upload(0, &weight_bf16).expect("upload dense weight"), rows, columns, output_rows)
            .expect("ROCm dense")
            .download_f32(rows * output_rows)
            .expect("download dense");
        for row in 0..rows {
            for output_row in 0..output_rows {
                let expected = (0..columns).map(|column| bf16_value(input[row * columns + column]) * bf16_value(weight[output_row * columns + column])).sum::<f32>();
                let value = actual[row * output_rows + output_row];
                assert!((value - expected).abs() < 0.08, "row={row} output={output_row} actual={value} expected={expected}");
            }
        }
    }

    #[test]
    fn rocm_dense_aligned_tile_matches_cpu() {
        let rows = 128usize;
        let columns = 64usize;
        let output_rows = 128usize;
        let input = vec![1.0f32; rows * columns];
        let weight = vec![bf16(1.0).to_ne_bytes(); output_rows * columns].into_iter().flatten().collect::<Vec<_>>();
        let actual = try_dense_matmul_bf16_f32(0, &DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload aligned dense input"), &DeviceBuffer::upload(0, &weight).expect("upload aligned dense weight"), rows, columns, output_rows)
            .expect("ROCm aligned dense")
            .download_f32(rows * output_rows)
            .expect("download aligned dense");
        for value in actual {
            assert!((value - 64.0).abs() < 0.01, "aligned dense value={value}");
        }
    }

    #[test]
    fn rocm_w8_g128_register_fragment_maps_every_k_column_and_input_tile() {
        if !super::super::is_hip_available() {
            eprintln!("[w8-register-fragment] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let rows = 129usize;
        let columns = 128usize;
        let output_rows = 32usize;
        let mut input = vec![0.0f32; rows * columns];
        for row in 0..rows {
            input[row * columns + row % columns] = 1.0;
        }
        let packed = (0..output_rows * columns).map(|index| ((index * 29 + 17) % 256) as u8).collect::<Vec<_>>();
        let scales = (0..output_rows).flat_map(|row| bf16(2.0f32.powi(-6 + (row % 3) as i32)).to_ne_bytes()).collect::<Vec<_>>();
        let packed_device = DeviceBuffer::upload(0, &packed).expect("upload W8 packed");
        let scales_device = DeviceBuffer::upload(0, &scales).expect("upload W8 scales");
        let actual =
            try_ct_quantized_matmul_bf16(0, 8, &input, None, &packed_device, &scales_device, 0, 128, rows, columns, output_rows).expect("ROCm W8 register fragment").download_f32(rows * output_rows).expect("download W8 register fragment");
        let residual = (0..rows * output_rows).map(|index| bf16_value((index as f32 * 0.007).sin())).collect::<Vec<_>>();
        let residual_bf16 = residual.iter().flat_map(|value| bf16(*value).to_ne_bytes()).collect::<Vec<_>>();
        let residual_device = DeviceBuffer::upload(0, &residual_bf16).expect("upload W8 residual");
        let fused = try_ct_quantized_matmul_bf16_add(0, 8, &input, None, &packed_device, &scales_device, 0, 128, rows, columns, output_rows, &residual_device)
            .expect("ROCm W8 residual epilogue")
            .download_f32(rows * output_rows)
            .expect("download W8 residual epilogue");
        assert!(fused.iter().zip(&actual).zip(&residual).all(|((&fused, &actual), &residual)| fused.to_bits() == (residual + actual).to_bits()));
        let decode = try_ct_quantized_matmul_bf16(0, 8, &input[..columns], None, &packed_device, &scales_device, 0, 128, 1, columns, output_rows).expect("ROCm W8 decode").download_f32(output_rows).expect("download W8 decode");
        let decode_residual = (0..output_rows).map(|index| (index as f32 * 0.013).cos()).collect::<Vec<_>>();
        let decode_residual_device = DeviceBuffer::upload(0, f32_bytes(&decode_residual)).expect("upload W8 decode residual");
        let decode_fused = try_ct_quantized_matmul_bf16_add(0, 8, &input[..columns], None, &packed_device, &scales_device, 0, 128, 1, columns, output_rows, &decode_residual_device)
            .expect("ROCm W8 decode residual epilogue")
            .download_f32(output_rows)
            .expect("download W8 decode residual epilogue");
        assert!(decode_fused.iter().zip(&decode).zip(&decode_residual).all(|((&fused, &projected), &residual)| fused.to_bits() == (residual + projected).to_bits()));
        let mut max_ulp = 0;
        let mut worst = (0, 0, 0.0, 0.0);
        for row in 0..rows {
            for output_row in 0..output_rows {
                let scale = bf16_value(2.0f32.powi(-6 + (output_row % 3) as i32));
                let code = packed[output_row * columns + row % columns];
                let expected = bf16_value((i32::from(code) - 128) as f32 * scale);
                let value = actual[row * output_rows + output_row];
                let ulp = value.to_bits().abs_diff(expected.to_bits());
                if ulp > max_ulp {
                    max_ulp = ulp;
                    worst = (row, output_row, value, expected);
                }
            }
        }
        assert!(max_ulp <= 1, "max_ulp={max_ulp} row={} output={} actual={} expected={}", worst.0, worst.1, worst.2, worst.3);
    }

    #[test]
    fn rocm_bf16_rows8_packed_dot_matches_cpu() {
        if !super::super::is_hip_available() {
            eprintln!("[bf16-rows8-packed-dot] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let columns = 128usize;
        let output_rows = 37usize;
        let weight = (0..output_rows * columns).map(|index| (index as f32 * 0.017).cos() * 0.125).collect::<Vec<_>>();
        let weight_bf16 = weight.iter().flat_map(|&value| bf16(value).to_ne_bytes()).collect::<Vec<_>>();
        let weight_device = DeviceBuffer::upload(0, &weight_bf16).expect("upload BF16 rows8 weight");

        for input_rows in [5usize, 8] {
            let input = (0..input_rows * columns).map(|index| (index as f32 * 0.031).sin() * 0.125).collect::<Vec<_>>();
            let input_device = DeviceBuffer::upload(0, f32_bytes(&input)).expect("upload BF16 rows8 input");
            let actual = try_bf16_gemv_resident_f32(0, &input_device, &weight_device, input_rows, columns, output_rows, true).expect("BF16 rows8 packed-dot").download_f32(input_rows * output_rows).expect("download BF16 rows8 packed-dot");
            for row in 0..input_rows {
                for output_row in 0..output_rows {
                    let expected = (0..columns).map(|column| bf16_value(input[row * columns + column]) * bf16_value(weight[output_row * columns + column])).sum::<f32>();
                    let value = actual[row * output_rows + output_row];
                    assert!((value - expected).abs() <= 2.0e-3, "rows={input_rows} row={row} output={output_row} actual={value} expected={expected}");
                }
            }
        }
    }
}
