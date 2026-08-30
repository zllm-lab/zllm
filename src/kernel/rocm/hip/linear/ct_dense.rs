pub(super) const SOURCE: &str = include_str!("ct_dense/source.hip");

use super::*;

pub fn try_dense_matmul_bf16_f32(device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, input_rows: usize, input_columns: usize, output_rows: usize) -> Result<DeviceBuffer, String> {
    if input_rows == 0 || input_columns == 0 || output_rows == 0 || !input_columns.is_multiple_of(16) || !output_rows.is_multiple_of(16) {
        return Err(format!("ROCm dense BF16 WMMA shape rows={input_rows} cols={input_columns} out={output_rows} 必须非零且 cols/out 为 16 的倍数",));
    }
    let input_elements = input_rows.checked_mul(input_columns).ok_or("ROCm dense input 大小溢出")?;
    let weight_elements = output_rows.checked_mul(input_columns).ok_or("ROCm dense weight 大小溢出")?;
    let output_elements = input_rows.checked_mul(output_rows).ok_or("ROCm dense output 大小溢出")?;
    let input_f32_bytes = input_elements.checked_mul(4).ok_or("ROCm dense F32 input 字节溢出")?;
    let input_bf16_bytes = input_elements.checked_mul(2).ok_or("ROCm dense BF16 input 字节溢出")?;
    if input.device_id != device_id || (input.bytes != input_f32_bytes && input.bytes != input_bf16_bytes) {
        return Err(format!("ROCm dense input bytes={}，期望 F32 {input_f32_bytes} 或 BF16 {input_bf16_bytes}", input.bytes));
    }
    validate_resident(weight, device_id, weight_elements.checked_mul(2).ok_or("ROCm dense weight 字节溢出")?, "dense weight")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements.checked_mul(4).ok_or("ROCm dense output 字节溢出")?)?;
    let functions = ct_quantized_functions(device_id)?;
    let aligned_tiles = input_columns.is_multiple_of(64) && output_rows.is_multiple_of(128);
    let converted_input;
    let input_bf16 = if input.bytes == input_bf16_bytes {
        input
    } else {
        converted_input = DeviceBuffer::allocate(device_id, input_bf16_bytes)?;
        let mut cast_input = input.pointer;
        let mut cast_output = converted_input.pointer;
        let mut cast_elements = u32::try_from(input_elements).map_err(|_| "ROCm dense cast elements 超过 u32")?;
        let mut cast_arguments = [(&mut cast_input as *mut *mut c_void).cast(), (&mut cast_output as *mut *mut c_void).cast(), (&mut cast_elements as *mut u32).cast()];
        launch_tensor_kernel(functions.cast, cast_elements.div_ceil(256), 256, &mut cast_arguments, "HIP dense F32 to BF16")?;
        &converted_input
    };
    if options().dense_profile {
        eprintln!("[rocm-dense] device={device_id} n={input_rows} m={input_columns} k={output_rows}");
    }
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let mut d_input = input_bf16.pointer;
    let mut d_weight = weight.pointer;
    let mut d_output = output.pointer;
    let mut input_rows_u32 = u32::try_from(input_rows).map_err(|_| "ROCm dense input rows 超过 u32")?;
    let mut input_columns_u32 = u32::try_from(input_columns).map_err(|_| "ROCm dense input columns 超过 u32")?;
    let mut output_rows_u32 = u32::try_from(output_rows).map_err(|_| "ROCm dense output rows 超过 u32")?;
    let mut input_row_block_offset_u32 = 0_u32;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_weight as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut input_rows_u32 as *mut u32).cast(),
        (&mut input_columns_u32 as *mut u32).cast(),
        (&mut output_rows_u32 as *mut u32).cast(),
        (&mut input_row_block_offset_u32 as *mut u32).cast(),
    ];
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let grid_x = u32::try_from(output_rows.div_ceil(128)).map_err(|_| "ROCm dense grid x 超过 u32")?;
    let full_row_blocks = input_rows / 128;
    if aligned_tiles && full_row_blocks != 0 {
        let status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let __hip_launch_result = unsafe {
                launch(
                    functions.dense_aligned as *mut c_void,
                    grid_x,
                    u32::try_from(full_row_blocks).map_err(|_| "ROCm dense aligned grid y 超过 u32")?,
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
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLaunchKernel dense BF16 WMMA aligned"));
        }
    }
    if !aligned_tiles || !input_rows.is_multiple_of(128) {
        let mut tail_input_row_block_offset_u32 = if aligned_tiles { u32::try_from(full_row_blocks).map_err(|_| "ROCm dense tail row offset 超过 u32")? } else { 0 };
        let mut tail_arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_weight as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut input_rows_u32 as *mut u32).cast(),
            (&mut input_columns_u32 as *mut u32).cast(),
            (&mut output_rows_u32 as *mut u32).cast(),
            (&mut tail_input_row_block_offset_u32 as *mut u32).cast(),
        ];
        let grid_y = if aligned_tiles { 1 } else { input_rows.div_ceil(128) };
        let status = {
            let __hip_stats_started = super::hip_api_stats::start();
            let __hip_launch_result = unsafe {
                launch(
                    functions.dense as *mut c_void,
                    grid_x,
                    u32::try_from(grid_y).map_err(|_| "ROCm dense tail grid y 超过 u32")?,
                    1,
                    256,
                    1,
                    1,
                    0,
                    crate::kernel::rocm::hip::active_compute_stream(),
                    tail_arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
            __hip_launch_result
        };
        if status != HIP_SUCCESS {
            return Err(runtime.hip_error(status, "hipModuleLaunchKernel dense BF16 WMMA tail"));
        }
    }
    if options().kernel_sync || profile_started.is_some() {
        synchronize_device(device_id, &format!("dense BF16 WMMA synchronize n={input_rows} m={input_columns} k={output_rows}",))?;
    }
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] dense device={device_id} n={input_rows} m={input_columns} k={output_rows} wall={:.6}s", started.elapsed().as_secs_f64(),);
    }
    Ok(output)
}

pub fn try_cast_f32_to_bf16_resident(device_id: i32, input: &DeviceBuffer, elements: usize) -> Result<DeviceBuffer, String> {
    validate_resident(input, device_id, elements.checked_mul(4).ok_or("F32 to BF16 input 大小溢出")?, "F32 to BF16 input")?;
    let output = DeviceBuffer::allocate_reusable(device_id, elements.checked_mul(2).ok_or("F32 to BF16 output 大小溢出")?)?;
    let functions = ct_quantized_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "F32 to BF16 元素数超过 u32")?;
    let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.cast, elements.div_ceil(256), 256, &mut arguments, "HIP F32 to BF16")?;
    Ok(output)
}

pub fn try_cast_bf16_to_f32_resident(device_id: i32, input: &DeviceBuffer, elements: usize) -> Result<DeviceBuffer, String> {
    validate_resident(input, device_id, elements.checked_mul(2).ok_or("BF16 to F32 input 大小溢出")?, "BF16 to F32 input")?;
    let output = DeviceBuffer::allocate(device_id, elements.checked_mul(4).ok_or("BF16 to F32 output 大小溢出")?)?;
    let functions = ct_quantized_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "BF16 to F32 元素数超过 u32")?;
    let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.expand, elements.div_ceil(256), 256, &mut arguments, "HIP BF16 to F32")?;
    Ok(output)
}

pub fn warmup_ct_quantized(device_id: i32) -> Result<(), String> {
    ct_quantized_functions(device_id)?;
    let input = DeviceBuffer::upload(device_id, &0.0f32.to_le_bytes())?;
    let weight = DeviceBuffer::upload(device_id, &[0u8; 2])?;
    let _output = try_bf16_gemv_resident_f32(device_id, &input, &weight, 1, 1, 1, false)?;
    synchronize_device(device_id, "BF16 GEMV warmup")?;
    tensor_functions(device_id)?;
    moe_prefill_functions(device_id).map(|_| ())
}

/// 压缩权重保持 resident；activation 在设备内转 BF16 后直接执行 W4/W8 GEMM。
#[allow(clippy::too_many_arguments)]
pub fn try_ct_quantized_matmul_bf16(
    device_id: i32,
    bits: u32,
    input: &[f32],
    input_device: Option<&DeviceBuffer>,
    packed: &DeviceBuffer,
    scales: &DeviceBuffer,
    scale_dtype: u32,
    group_size: usize,
    input_rows: usize,
    input_columns: usize,
    output_rows: usize,
) -> Result<DeviceBuffer, String> {
    try_ct_quantized_matmul_bf16_epilogue(device_id, bits, input, input_device, packed, scales, scale_dtype, group_size, input_rows, input_columns, output_rows, None)
}

#[allow(clippy::too_many_arguments)]
pub fn try_ct_quantized_matmul_bf16_add(
    device_id: i32,
    bits: u32,
    input: &[f32],
    input_device: Option<&DeviceBuffer>,
    packed: &DeviceBuffer,
    scales: &DeviceBuffer,
    scale_dtype: u32,
    group_size: usize,
    input_rows: usize,
    input_columns: usize,
    output_rows: usize,
    residual: &DeviceBuffer,
) -> Result<DeviceBuffer, String> {
    try_ct_quantized_matmul_bf16_epilogue(device_id, bits, input, input_device, packed, scales, scale_dtype, group_size, input_rows, input_columns, output_rows, Some(residual))
}

#[allow(clippy::too_many_arguments)]
fn try_ct_quantized_matmul_bf16_epilogue(
    device_id: i32,
    bits: u32,
    input: &[f32],
    input_device: Option<&DeviceBuffer>,
    packed: &DeviceBuffer,
    scales: &DeviceBuffer,
    scale_dtype: u32,
    group_size: usize,
    input_rows: usize,
    input_columns: usize,
    output_rows: usize,
    residual: Option<&DeviceBuffer>,
) -> Result<DeviceBuffer, String> {
    if !matches!(bits, 4 | 8) {
        return Err(format!("ROCm CT quantized bits={bits} 不受支持"));
    }
    if packed.device_id != device_id || scales.device_id != device_id {
        return Err("ROCm quantized weight 与执行 device 不一致".to_owned());
    }
    let input_elements = input_rows.checked_mul(input_columns).ok_or("ROCm quantized input 大小溢出")?;
    let output_elements = input_rows.checked_mul(output_rows).ok_or("ROCm quantized output 大小溢出")?;
    if (input_device.is_none() && input.len() != input_elements)
        || input_device.is_some_and(|buffer| buffer.device_id != device_id || (buffer.bytes < input_elements * 4 && buffer.bytes != input_elements * 2))
        || group_size == 0
        || !input_columns.is_multiple_of(group_size)
    {
        return Err(format!("ROCm CT quantized shape 无效: input={} output={} rows={input_rows} cols={input_columns} out={output_rows} group={group_size}", input.len(), output_elements,));
    }
    let values_per_word = 32usize / bits as usize;
    let expected_packed = output_rows.checked_mul(input_columns.div_ceil(values_per_word)).and_then(|words| words.checked_mul(4)).ok_or("ROCm CT packed 字节数溢出")?;
    let scale_bytes = match scale_dtype {
        0 | 1 => 2usize,
        2 => 4usize,
        _ => return Err(format!("ROCm CT scale dtype={scale_dtype} 无效")),
    };
    let expected_scales = output_rows.checked_mul(input_columns / group_size).and_then(|values| values.checked_mul(scale_bytes)).ok_or("ROCm CT scales 字节数溢出")?;
    if packed.bytes < expected_packed || scales.bytes < expected_scales {
        return Err(format!("ROCm CT resident 权重过小: packed={} expected={expected_packed}, scales={} expected={expected_scales}", packed.bytes, scales.bytes,));
    }
    set_device(device_id)?;
    let input_bytes = input_elements.checked_mul(4).ok_or("ROCm quantized input 字节溢出")?;
    let output_bytes = output_elements.checked_mul(4).ok_or("ROCm quantized output 字节溢出")?;
    let bf16_bytes = input_elements.checked_mul(2).ok_or("ROCm BF16 input 大小溢出")?;
    let input_is_bf16 = input_device.is_some_and(|buffer| buffer.bytes == bf16_bytes);
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let functions = ct_quantized_functions(device_id)?;
    let use_wmma = input_rows >= 16;
    let use_w8_g128_wmma = use_wmma && bits == 8 && group_size == 128 && scale_dtype == 0 && functions.wavefront_size == 32;
    let use_w8 = !use_wmma && bits == 8 && group_size % 4 == 0 && input_columns % 4 == 0;
    let use_w8_rows2 = use_w8 && input_rows == 2 && group_size == 128 && functions.wavefront_size == 32;
    let use_w4_rows_shared = !use_wmma && bits == 4 && (2..=8).contains(&input_rows) && group_size == 128 && functions.wavefront_size == 32;
    if let Some(residual) = residual {
        if !use_w8_g128_wmma && !(use_w8 && input_rows == 1) {
            return Err(format!("ROCm CT residual epilogue 仅支持 W8G128 prefill 或单行 W8 decode，实际 bits={bits} group={group_size} scale={scale_dtype} rows={input_rows}"));
        }
        let element_bytes = if input_rows == 1 { 4 } else { 2 };
        let residual_bytes = output_elements.checked_mul(element_bytes).ok_or("ROCm quantized residual 字节溢出")?;
        validate_resident(residual, device_id, residual_bytes, "ROCm quantized residual")?;
    }
    if options().log_memory && output_bytes >= 64 * 1024 * 1024 {
        if let Ok((free, total)) = device_memory_info(device_id) {
            eprintln!("[rocm-ct-memory] device={device_id} input=[{input_rows},{input_columns}] output_rows={output_rows} free={:.2}GiB used={:.2}GiB", free as f64 / (1_u64 << 30) as f64, (total - free) as f64 / (1_u64 << 30) as f64);
        }
    }
    let d_result = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;
    CT_QUANTIZED_WORKSPACES.with(|workspaces| {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        workspace.ensure(device_id, if input_device.is_none() { input_bytes.max(1) } else { 0 }, bf16_bytes, output_bytes)?;
        let d_input_f32 = workspace.input_f32.as_ref();
        let d_input_bf16 = workspace.input_bf16.as_ref().unwrap();
        if input_device.is_none() {
            d_input_f32.expect("host input staging 已分配").copy_from_host(unsafe { std::slice::from_raw_parts(input.as_ptr().cast(), input_bytes) })?;
        }

        let mut quantized_input = if input_is_bf16 {
            input_device.expect("BF16 input 已验证").pointer
        } else {
            let mut cast_input = input_device.map_or_else(|| d_input_f32.expect("host input staging 已分配").pointer, |buffer| buffer.pointer);
            let mut cast_output = d_input_bf16.pointer;
            let mut elements = u32::try_from(input_elements).map_err(|_| "ROCm activation 元素数超过 u32".to_owned())?;
            let mut cast_args = [(&mut cast_input as *mut *mut c_void).cast(), (&mut cast_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
            let cast_status = {
                let __hip_stats_started = super::hip_api_stats::start();
                let __hip_launch_result = unsafe { launch(functions.cast as *mut c_void, elements.div_ceil(256), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), cast_args.as_mut_ptr(), ptr::null_mut()) };
                super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                __hip_launch_result
            };
            if cast_status != HIP_SUCCESS {
                return Err(runtime.hip_error(cast_status, "hipModuleLaunchKernel f32_to_bf16"));
            }
            d_input_bf16.pointer
        };
        let mut quantized_packed = packed.pointer;
        let mut quantized_scales = scales.pointer;
        let mut quantized_output = d_result.pointer;
        let mut input_rows = u32::try_from(input_rows).map_err(|_| "ROCm input_rows 超过 u32".to_owned())?;
        let mut input_columns = u32::try_from(input_columns).map_err(|_| "ROCm input_columns 超过 u32".to_owned())?;
        let mut output_rows = u32::try_from(output_rows).map_err(|_| "ROCm output_rows 超过 u32".to_owned())?;
        let mut group_size = u32::try_from(group_size).map_err(|_| "ROCm group_size 超过 u32".to_owned())?;
        let mut scale_dtype = scale_dtype;
        let mut bits = bits;
        let mut quantized_residual = residual.map_or(ptr::null_mut(), |buffer| buffer.pointer);
        let mut quantized_args = [
            (&mut quantized_input as *mut *mut c_void).cast(),
            (&mut quantized_packed as *mut *mut c_void).cast(),
            (&mut quantized_scales as *mut *mut c_void).cast(),
            (&mut quantized_output as *mut *mut c_void).cast(),
            (&mut input_rows as *mut u32).cast(),
            (&mut input_columns as *mut u32).cast(),
            (&mut output_rows as *mut u32).cast(),
            (&mut group_size as *mut u32).cast(),
            (&mut scale_dtype as *mut u32).cast(),
            (&mut bits as *mut u32).cast(),
            (&mut quantized_residual as *mut *mut c_void).cast(),
        ];
        // 多行 verify(draft+anchor)走共享装载路径:权重只读一次,避免超过 L2 的
        // 大矩阵(如 154880 输出的 lm head)被 grid.y 逐行重读 6-8 遍。
        // 仅覆盖 wave32 + group128 的快路径形状,其他形状回退逐行 scalar。
        let use_w8_rows_shared = use_w8 && matches!(input_rows, 6 | 8) && group_size == 128 && input_columns % 128 == 0 && input_columns / group_size >= 32 && functions.wavefront_size == 32;
        let wmma_profile_started = (use_wmma && options().kernel_profile).then(std::time::Instant::now);
        let gemv_profile_started = (!use_wmma && options().kernel_profile).then(std::time::Instant::now);
        let w8_profile_started = (use_w8 && options().w8_profile).then(std::time::Instant::now);
        let scalar_threads = if use_w4_rows_shared {
            128
        } else if use_w8 && output_rows >= 12_288 {
            512
        } else if use_w8 && output_rows <= 1024 {
            128
        } else {
            256
        };
        let scalar_subgroup = if use_w8 && group_size == 128 && (input_columns == 2048 || input_columns / group_size >= 32) { 32 } else { 16 };
        // 大输出矩阵用 16 waves 复用 input tile；小矩阵保留 8 waves，避免空 lane 和驻留回退。
        let wmma_waves = if output_rows >= 12_288 || (output_rows >= 6_144 && input_rows >= 512) { 16 } else { 8 };
        let wmma_threads = functions.wavefront_size * wmma_waves;
        let quantized_status = unsafe {
            launch(
                if use_wmma {
                    if use_w8_g128_wmma { functions.wmma_w8_g128 } else { functions.wmma }
                } else if use_w8_rows_shared {
                    if input_rows == 6 { functions.w8_rows6 } else { functions.w8_rows8 }
                } else if use_w8_rows2 {
                    functions.w8_rows2
                } else if use_w4_rows_shared {
                    functions.w4_rows8
                } else if use_w8 {
                    functions.w8_scalar
                } else {
                    functions.scalar
                } as *mut c_void,
                if use_wmma {
                    output_rows.div_ceil(wmma_waves * 16)
                } else if use_w8_rows2 || use_w8_rows_shared {
                    output_rows.div_ceil(scalar_threads / 32)
                } else if use_w4_rows_shared {
                    output_rows.div_ceil(scalar_threads / 16)
                } else {
                    output_rows.div_ceil(scalar_threads / scalar_subgroup)
                },
                if use_wmma {
                    input_rows.div_ceil(128)
                } else if use_w8_rows2 || use_w8_rows_shared || use_w4_rows_shared {
                    1
                } else {
                    input_rows
                },
                1,
                if use_wmma { wmma_threads } else { scalar_threads },
                1,
                1,
                0,
                crate::kernel::rocm::hip::active_compute_stream(),
                quantized_args.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        if quantized_status != HIP_SUCCESS {
            return Err(runtime.hip_error(quantized_status, "hipModuleLaunchKernel CT quantized"));
        }
        if options().kernel_sync || wmma_profile_started.is_some() || gemv_profile_started.is_some() {
            synchronize_device(device_id, "CT quantized synchronize")?;
        }
        if let Some(started) = wmma_profile_started {
            eprintln!(
                "[rocm-kernel] ct-wmma device={device_id} bits={bits} input_rows={input_rows} input_columns={input_columns} output_rows={output_rows} group_size={group_size} waves={wmma_waves} register_b={use_w8_g128_wmma} wall={:.6}s",
                started.elapsed().as_secs_f64(),
            );
        }
        if let Some(started) = gemv_profile_started {
            eprintln!(
                "[rocm-kernel] ct-gemv device={device_id} bits={bits} input_rows={input_rows} input_columns={input_columns} output_rows={output_rows} group_size={group_size} shared_rows2={use_w8_rows2} wall={:.6}s",
                started.elapsed().as_secs_f64(),
            );
        }
        if let Some(started) = w8_profile_started {
            synchronize_device(device_id, "W8 GEMV profile")?;
            eprintln!("[rocm-w8] input_rows={input_rows} input_columns={input_columns} output_rows={output_rows} group_size={group_size} wall={:.6}s", started.elapsed().as_secs_f64(),);
        }
        Ok(())
    })?;
    Ok(d_result)
}

/// Decode 的两个同格式 compressed-tensors 投影共享一次 activation BF16 转换。
#[allow(clippy::too_many_arguments)]
pub fn try_ct_dual_gemv_bf16(
    device_id: i32,
    bits: u32,
    input: &[f32],
    input_device: Option<&DeviceBuffer>,
    input_columns: usize,
    input_rows: usize,
    first_packed: &DeviceBuffer,
    first_scales: &DeviceBuffer,
    first_scale_dtype: u32,
    first_group_size: usize,
    first_output_rows: usize,
    second_packed: &DeviceBuffer,
    second_scales: &DeviceBuffer,
    second_scale_dtype: u32,
    second_group_size: usize,
    second_output_rows: usize,
) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    if !matches!(bits, 4 | 8) {
        return Err(format!("ROCm dual CT bits={bits} 不受支持"));
    }
    if input_rows == 0 || input_rows > 8 {
        return Err(format!("ROCm dual CT input_rows={input_rows} 超出 1-8"));
    }
    if bits == 4 && (first_group_size != 128 || second_group_size != 128) {
        return Err(format!("ROCm dual W4 仅支持 group128，实际 first={first_group_size} second={second_group_size}"));
    }
    if first_packed.device_id != device_id || first_scales.device_id != device_id || second_packed.device_id != device_id || second_scales.device_id != device_id {
        return Err("ROCm dual CT weight 与执行 device 不一致".to_owned());
    }
    let input_elements = input_rows.checked_mul(input_columns).ok_or("ROCm dual CT input 大小溢出")?;
    let input_bytes = input_elements.checked_mul(4).ok_or("ROCm dual CT input 字节溢出")?;
    let bf16_bytes = input_elements.checked_mul(2).ok_or("ROCm dual CT BF16 input 字节溢出")?;
    if (input_device.is_none() && input.len() != input_elements)
        || input_device.is_some_and(|buffer| buffer.device_id != device_id || buffer.bytes < input_bytes && buffer.bytes != bf16_bytes)
        || first_group_size == 0
        || second_group_size == 0
        || !input_columns.is_multiple_of(first_group_size)
        || !input_columns.is_multiple_of(second_group_size)
    {
        return Err(format!("ROCm dual CT shape 无效: input={} rows={input_rows} cols={input_columns} first=[{first_output_rows},{first_group_size}] second=[{second_output_rows},{second_group_size}]", input.len(),));
    }

    set_device(device_id)?;
    let input_is_bf16 = input_device.is_some_and(|buffer| buffer.bytes == bf16_bytes);
    let first_output_bytes = input_rows.checked_mul(first_output_rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm dual CT first output 字节溢出")?;
    let second_output_bytes = input_rows.checked_mul(second_output_rows).and_then(|n| n.checked_mul(4)).ok_or("ROCm dual CT second output 字节溢出")?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let functions = ct_quantized_functions(device_id)?;
    let first_result = DeviceBuffer::allocate_reusable(device_id, first_output_bytes)?;
    let second_result = DeviceBuffer::allocate_reusable(device_id, second_output_bytes)?;

    CT_QUANTIZED_WORKSPACES.with(|workspaces| {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        workspace.ensure(device_id, if input_device.is_none() { input_bytes.max(1) } else { 0 }, bf16_bytes, first_output_bytes.max(second_output_bytes))?;
        let d_input_f32 = workspace.input_f32.as_ref();
        let d_input_bf16 = workspace.input_bf16.as_ref().unwrap();
        if input_device.is_none() {
            d_input_f32.expect("host input staging 已分配").copy_from_host(unsafe { std::slice::from_raw_parts(input.as_ptr().cast(), input_bytes) })?;
        }

        if !input_is_bf16 {
            let mut cast_input = input_device.map_or_else(|| d_input_f32.expect("host input staging 已分配").pointer, |buffer| buffer.pointer);
            let mut cast_output = d_input_bf16.pointer;
            let mut elements = u32::try_from(input_elements).map_err(|_| "ROCm dual CT activation 元素数超过 u32".to_owned())?;
            let mut cast_args = [(&mut cast_input as *mut *mut c_void).cast(), (&mut cast_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
            let status = {
                let __hip_stats_started = super::hip_api_stats::start();
                let __hip_launch_result = unsafe { launch(functions.cast as *mut c_void, elements.div_ceil(256), 1, 1, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), cast_args.as_mut_ptr(), ptr::null_mut()) };
                super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                __hip_launch_result
            };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel dual CT f32_to_bf16"));
            }
        }
        let input_bf16 = input_device.filter(|_| input_is_bf16).map_or(d_input_bf16.pointer, |buffer| buffer.pointer);

        // 单投影 W8 scalar GEMV；多行或非 128 group 时按行回退，每行数值独立不变。
        let launch_w8 = |row: usize, packed: &DeviceBuffer, scales: &DeviceBuffer, output: &DeviceBuffer, scale_dtype: u32, group_size: usize, output_rows: usize| -> Result<(), String> {
            let mut d_input = unsafe { input_bf16.cast::<u8>().add(row * input_columns * 2).cast() };
            let mut d_packed = packed.pointer;
            let mut d_scales = scales.pointer;
            let mut d_output = unsafe { output.pointer.cast::<u8>().add(row * output_rows * 4).cast() };
            let mut kernel_input_rows = 1u32;
            let mut input_columns = u32::try_from(input_columns).map_err(|_| "ROCm dual CT input_columns 超过 u32".to_owned())?;
            let mut output_rows = u32::try_from(output_rows).map_err(|_| "ROCm dual CT output_rows 超过 u32".to_owned())?;
            let mut group_size = u32::try_from(group_size).map_err(|_| "ROCm dual CT group_size 超过 u32".to_owned())?;
            let mut scale_dtype = scale_dtype;
            let mut bits = bits;
            let mut d_residual: *mut c_void = ptr::null_mut();
            let mut arguments = [
                (&mut d_input as *mut *mut c_void).cast(),
                (&mut d_packed as *mut *mut c_void).cast(),
                (&mut d_scales as *mut *mut c_void).cast(),
                (&mut d_output as *mut *mut c_void).cast(),
                (&mut kernel_input_rows as *mut u32).cast(),
                (&mut input_columns as *mut u32).cast(),
                (&mut output_rows as *mut u32).cast(),
                (&mut group_size as *mut u32).cast(),
                (&mut scale_dtype as *mut u32).cast(),
                (&mut bits as *mut u32).cast(),
                (&mut d_residual as *mut *mut c_void).cast(),
            ];
            let threads = if output_rows <= 1024 { 128 } else { 256 };
            let subgroup = if group_size == 128 && input_columns / group_size >= 32 { 32 } else { 16 };
            let profile_started = (options().kernel_profile || options().w8_profile).then(std::time::Instant::now);
            let status = {
                let __hip_stats_started = super::hip_api_stats::start();
                let __hip_launch_result =
                    unsafe { launch(functions.w8_scalar as *mut c_void, output_rows.div_ceil(threads / subgroup), 1, 1, threads, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) };
                super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                __hip_launch_result
            };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel dual CT GEMV"));
            }
            if let Some(started) = profile_started {
                synchronize_device(device_id, "dual CT GEMV profile")?;
                eprintln!("[rocm-kernel] ct-dual-gemv-part device={device_id} bits={bits} input_rows={input_rows} row={row} input_columns={input_columns} output_rows={output_rows} group_size={group_size} wall={:.6}s", started.elapsed().as_secs_f64(),);
            }
            Ok(())
        };

        if first_group_size == 128 && second_group_size == 128 && (input_rows == 1 || bits == 4) {
            let mut d_input = input_bf16;
            let mut d_first_packed = first_packed.pointer;
            let mut d_first_scales = first_scales.pointer;
            let mut d_first_output = first_result.pointer;
            let mut first_output_rows = u32::try_from(first_output_rows).map_err(|_| "ROCm dual CT first output_rows 超过 u32".to_owned())?;
            let mut first_scale_dtype = first_scale_dtype;
            let mut d_second_packed = second_packed.pointer;
            let mut d_second_scales = second_scales.pointer;
            let mut d_second_output = second_result.pointer;
            let mut second_output_rows = u32::try_from(second_output_rows).map_err(|_| "ROCm dual CT second output_rows 超过 u32".to_owned())?;
            let mut second_scale_dtype = second_scale_dtype;
            let mut input_columns = u32::try_from(input_columns).map_err(|_| "ROCm dual CT input_columns 超过 u32".to_owned())?;
            let mut input_rows_u32 = u32::try_from(input_rows).map_err(|_| "ROCm dual CT input_rows 超过 u32".to_owned())?;
            let max_output_rows = first_output_rows.max(second_output_rows);
            let started = (options().kernel_profile || options().w8_profile).then(std::time::Instant::now);
            let status = {
                let __hip_stats_started = super::hip_api_stats::start();
                let __hip_launch_result = if bits == 4 && input_rows > 1 {
                    // 2-8 行共用一次 dispatch 与 packed word/scale 装载，grid.z 选择权重。
                    let mut arguments = [
                        (&mut d_input as *mut *mut c_void).cast(),
                        (&mut d_first_packed as *mut *mut c_void).cast(),
                        (&mut d_first_scales as *mut *mut c_void).cast(),
                        (&mut d_first_output as *mut *mut c_void).cast(),
                        (&mut first_output_rows as *mut u32).cast(),
                        (&mut first_scale_dtype as *mut u32).cast(),
                        (&mut d_second_packed as *mut *mut c_void).cast(),
                        (&mut d_second_scales as *mut *mut c_void).cast(),
                        (&mut d_second_output as *mut *mut c_void).cast(),
                        (&mut second_output_rows as *mut u32).cast(),
                        (&mut second_scale_dtype as *mut u32).cast(),
                        (&mut input_columns as *mut u32).cast(),
                        (&mut input_rows_u32 as *mut u32).cast(),
                    ];
                    unsafe { launch(functions.w4_dual_rows8 as *mut c_void, max_output_rows.div_ceil(16), 1, 2, 256, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) }
                } else {
                    let mut arguments = [
                        (&mut d_input as *mut *mut c_void).cast(),
                        (&mut d_first_packed as *mut *mut c_void).cast(),
                        (&mut d_first_scales as *mut *mut c_void).cast(),
                        (&mut d_first_output as *mut *mut c_void).cast(),
                        (&mut first_output_rows as *mut u32).cast(),
                        (&mut first_scale_dtype as *mut u32).cast(),
                        (&mut d_second_packed as *mut *mut c_void).cast(),
                        (&mut d_second_scales as *mut *mut c_void).cast(),
                        (&mut d_second_output as *mut *mut c_void).cast(),
                        (&mut second_output_rows as *mut u32).cast(),
                        (&mut second_scale_dtype as *mut u32).cast(),
                        (&mut input_columns as *mut u32).cast(),
                    ];
                    let threads = if bits == 8 && max_output_rows <= 1024 { 128 } else { 256 };
                    let subgroup = if bits == 4 { 16 } else { 32 };
                    let function = if bits == 4 { functions.w4_dual } else { functions.w8_dual };
                    unsafe { launch(function as *mut c_void, max_output_rows.div_ceil(threads / subgroup), 1, 2, threads, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), arguments.as_mut_ptr(), ptr::null_mut()) }
                };
                super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
                __hip_launch_result
            };
            if status != HIP_SUCCESS {
                return Err(runtime.hip_error(status, "hipModuleLaunchKernel fused dual CT GEMV"));
            }
            if let Some(started) = started {
                synchronize_device(device_id, "fused dual CT GEMV profile")?;
                eprintln!("[rocm-kernel] ct-dual-gemv device={device_id} bits={bits} input_rows={input_rows_u32} input_columns={input_columns} first_output_rows={first_output_rows} second_output_rows={second_output_rows} first_group_size={first_group_size} second_group_size={second_group_size} wall={:.6}s", started.elapsed().as_secs_f64(),);
            }
            return Ok(());
        }

        for row in 0..input_rows {
            launch_w8(row, first_packed, first_scales, &first_result, first_scale_dtype, first_group_size, first_output_rows)?;
            launch_w8(row, second_packed, second_scales, &second_result, second_scale_dtype, second_group_size, second_output_rows)?;
        }
        Ok(())
    })?;
    Ok((first_result, second_result))
}
