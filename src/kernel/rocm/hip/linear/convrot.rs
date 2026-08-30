pub(super) const SOURCE: &str = include_str!("convrot/source.hip");

use super::*;

/// Comfy INT8 ConvRot：在线 Hadamard + per-token INT8 量化，再用 rocWMMA i8×i8→i32。
#[allow(clippy::too_many_arguments)]
pub fn try_convrot_int8_matmul_f32(
    device_id: i32,
    input: &[f32],
    input_device: Option<&DeviceBuffer>,
    packed: &DeviceBuffer,
    scales: &DeviceBuffer,
    group_size: usize,
    input_rows: usize,
    input_columns: usize,
    output_rows: usize,
) -> Result<DeviceBuffer, String> {
    if group_size != 256 || input_rows == 0 || input_columns == 0 || output_rows == 0 || !input_columns.is_multiple_of(group_size) || !output_rows.is_multiple_of(16) {
        return Err(format!("ROCm INT8 ConvRot shape rows={input_rows} cols={input_columns} out={output_rows} group={group_size} 无效"));
    }
    let input_elements = input_rows.checked_mul(input_columns).ok_or("ROCm ConvRot input 大小溢出")?;
    let output_elements = input_rows.checked_mul(output_rows).ok_or("ROCm ConvRot output 大小溢出")?;
    let expected_weight = output_rows.checked_mul(input_columns).ok_or("ROCm ConvRot weight 大小溢出")?;
    if packed.device_id != device_id || scales.device_id != device_id || packed.bytes < expected_weight || scales.bytes < output_rows * 4 {
        return Err("ROCm INT8 ConvRot resident 权重不完整或 device 不一致".to_owned());
    }
    let input_bf16_bytes = input_elements.checked_mul(2).ok_or("ROCm ConvRot BF16 input 大小溢出")?;
    let input_f32_bytes = input_elements.checked_mul(4).ok_or("ROCm ConvRot F32 input 大小溢出")?;
    if input_device.is_none() && input.len() != input_elements {
        return Err(format!("ROCm INT8 ConvRot host input={}，期望 {input_elements}", input.len()));
    }
    if let Some(buffer) = input_device {
        if buffer.device_id != device_id || (buffer.bytes != input_bf16_bytes && buffer.bytes < input_f32_bytes) {
            return Err(format!("ROCm INT8 ConvRot device input bytes={} 无效", buffer.bytes));
        }
    }

    set_device(device_id)?;
    let uploaded = if input_device.is_none() {
        let bytes = unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<u8>(), input_f32_bytes) };
        Some(DeviceBuffer::upload(device_id, bytes)?)
    } else {
        None
    };
    let source = input_device.or(uploaded.as_ref()).expect("ConvRot input 已验证");
    let input_bf16 = u32::from(source.bytes == input_bf16_bytes);
    let padded_rows = input_rows.div_ceil(256) * 256;
    let quantized_elements = padded_rows.checked_mul(input_columns).ok_or("ROCm ConvRot activation 大小溢出")?;
    let quantized = DeviceBuffer::allocate_reusable(device_id, quantized_elements)?;
    let activation_scales = DeviceBuffer::allocate_reusable(device_id, padded_rows.checked_mul(4).ok_or("ROCm ConvRot scale 大小溢出")?)?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_elements.checked_mul(4).ok_or("ROCm ConvRot output bytes 溢出")?)?;
    let functions = ct_quantized_functions(device_id)?;
    let runtime = RocmRuntime::open()?;
    let launch = crate::kernel::rocm::hip::kernel_launch_trampoline;
    let profile = options().kernel_profile;

    let mut d_input = source.pointer;
    let mut d_quantized = quantized.pointer;
    let mut d_activation_scales = activation_scales.pointer;
    let mut logical_rows = u32::try_from(input_rows).map_err(|_| "ROCm ConvRot rows 超过 u32")?;
    let mut padded_rows_u32 = u32::try_from(padded_rows).map_err(|_| "ROCm ConvRot padded rows 超过 u32")?;
    let mut input_columns_u32 = u32::try_from(input_columns).map_err(|_| "ROCm ConvRot columns 超过 u32")?;
    let mut input_bf16 = input_bf16;
    let mut quantize_arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_quantized as *mut *mut c_void).cast(),
        (&mut d_activation_scales as *mut *mut c_void).cast(),
        (&mut logical_rows as *mut u32).cast(),
        (&mut input_columns_u32 as *mut u32).cast(),
        (&mut input_bf16 as *mut u32).cast(),
    ];
    let cached_quantize = options().convrot_cached_quant && (input_columns <= 7168 || options().convrot_force_cached_quant) && input_columns.checked_mul(4).is_some_and(|bytes| bytes <= 60 * 1024 - 256 * 4);
    let quantize_started = std::time::Instant::now();
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let __hip_launch_result = unsafe {
            launch(
                (if cached_quantize { functions.convrot_quantize_cached } else { functions.convrot_quantize }) as *mut c_void,
                padded_rows_u32,
                1,
                1,
                256,
                1,
                1,
                if cached_quantize { input_columns_u32 * 4 } else { 0 },
                crate::kernel::rocm::hip::active_compute_stream(),
                quantize_arguments.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel ConvRot quantize"));
    }
    if profile {
        synchronize_device(device_id, "ConvRot quantize profile")?;
        eprintln!("[rocm-convrot] stage=quantize rows={input_rows} cols={input_columns} cached={cached_quantize} wall={:.6}s", quantize_started.elapsed().as_secs_f64());
    }

    let mut d_weight = packed.pointer;
    let mut d_weight_scales = scales.pointer;
    let mut d_output = output.pointer;
    let mut output_rows_u32 = u32::try_from(output_rows).map_err(|_| "ROCm ConvRot output rows 超过 u32")?;
    let mut gemm_arguments = [
        (&mut d_quantized as *mut *mut c_void).cast(),
        (&mut d_weight as *mut *mut c_void).cast(),
        (&mut d_activation_scales as *mut *mut c_void).cast(),
        (&mut d_weight_scales as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut logical_rows as *mut u32).cast(),
        (&mut padded_rows_u32 as *mut u32).cast(),
        (&mut input_columns_u32 as *mut u32).cast(),
        (&mut output_rows_u32 as *mut u32).cast(),
    ];
    let baseline = options().convrot_baseline;
    let tiled = input_columns % 64 == 0 && !baseline && options().convrot_tiled && (input_rows >= 96 || options().convrot_force_tiled);
    let optimized = input_columns % 64 == 0 && !baseline;
    let row16 = optimized && output_rows > input_columns;
    let (kernel, grid_x, grid_y, block_threads, mode) = if tiled {
        let tall = input_rows >= 256 || input_columns > output_rows;
        let block_rows = if tall { 256usize } else { 128usize };
        (
            if tall { functions.convrot_wmma_tiled256 } else { functions.convrot_wmma_tiled128 },
            u32::try_from(output_rows.div_ceil(128)).map_err(|_| "ROCm ConvRot tiled grid.x 超过 u32")?,
            u32::try_from(input_rows.div_ceil(block_rows)).map_err(|_| "ROCm ConvRot tiled grid.y 超过 u32")?,
            functions.wavefront_size * 8,
            if tall { "tiled256x128x64" } else { "tiled128x128x64" },
        )
    } else {
        let waves = if row16 { 16u32 } else { 8u32 };
        let tile_rows = padded_rows / if row16 { 256 } else { 16 };
        let output_tiles = output_rows / 16;
        let tiles = tile_rows.checked_mul(output_tiles).ok_or("ROCm ConvRot tile 数溢出")?;
        (
            if optimized { if row16 { functions.convrot_wmma_row16 } else { functions.convrot_wmma_split4 } } else { functions.convrot_wmma },
            u32::try_from(tiles.div_ceil(waves as usize)).map_err(|_| "ROCm ConvRot grid 超过 u32")?,
            1,
            functions.wavefront_size * waves,
            if row16 {
                "row16"
            } else if optimized {
                "split4"
            } else {
                "baseline"
            },
        )
    };
    let gemm_started = std::time::Instant::now();
    let status = {
        let __hip_stats_started = super::hip_api_stats::start();
        let __hip_launch_result = unsafe { launch(kernel as *mut c_void, grid_x, grid_y, 1, block_threads, 1, 1, 0, crate::kernel::rocm::hip::active_compute_stream(), gemm_arguments.as_mut_ptr(), ptr::null_mut()) };
        super::hip_api_stats::counted(super::hip_api_stats::LAUNCH, __hip_stats_started);
        __hip_launch_result
    };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipModuleLaunchKernel ConvRot INT8 WMMA"));
    }
    if profile {
        synchronize_device(device_id, "ConvRot GEMM profile")?;
        eprintln!("[rocm-convrot] stage=gemm rows={input_rows} cols={input_columns} out={output_rows} mode={mode} wall={:.6}s", gemm_started.elapsed().as_secs_f64());
    }
    Ok(output)
}
