pub(super) const PREFIX: &str = include_str!("elementwise/source.hip");

pub(super) const SELECTION: &str = include_str!("elementwise/selection.hip");

pub(super) const DIFFUSION: &str = include_str!("elementwise/diffusion.hip");

pub(super) const VAE: &str = include_str!("elementwise/vae.hip");

use super::*;

pub fn try_add_resident_f32(device_id: i32, left: &DeviceBuffer, right: &DeviceBuffer, elements: usize, scale: f32) -> Result<DeviceBuffer, String> {
    let bytes = elements.checked_mul(4).ok_or("resident add 大小溢出")?;
    validate_resident(left, device_id, bytes, "add left")?;
    validate_resident(right, device_id, bytes, "add right")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_left = left.pointer;
    let mut d_right = right.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "resident add 元素数超过 u32".to_owned())?;
    let mut scale = scale;
    let mut arguments = [(&mut d_left as *mut *mut c_void).cast(), (&mut d_right as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast(), (&mut scale as *mut f32).cast()];
    launch_tensor_kernel(functions.add_scaled, elements.div_ceil(256), 256, &mut arguments, "HIP resident add")?;
    Ok(output)
}

pub fn try_subtract_resident_f32(device_id: i32, left: &DeviceBuffer, right: &DeviceBuffer, elements: usize) -> Result<DeviceBuffer, String> {
    let bytes = elements.checked_mul(4).ok_or("resident subtract 大小溢出")?;
    validate_resident(left, device_id, bytes, "subtract left")?;
    validate_resident(right, device_id, bytes, "subtract right")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_left = left.pointer;
    let mut d_right = right.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "resident subtract 元素数超过 u32".to_owned())?;
    let mut arguments = [(&mut d_left as *mut *mut c_void).cast(), (&mut d_right as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.subtract, elements.div_ceil(256), 256, &mut arguments, "HIP resident subtract")?;
    Ok(output)
}

/// 只下载每个 block 的两个 partial sum，避免为 cache 判据回读整块 activation。
pub fn try_relative_l1_delta_resident_f32(device_id: i32, output: &DeviceBuffer, input: &DeviceBuffer, previous: &DeviceBuffer, elements: usize) -> Result<(f64, f64), String> {
    let bytes = elements.checked_mul(4).ok_or("resident relative L1 大小溢出")?;
    validate_resident(output, device_id, bytes, "relative L1 output")?;
    validate_resident(input, device_id, bytes, "relative L1 input")?;
    validate_resident(previous, device_id, bytes, "relative L1 previous")?;
    set_device(device_id)?;
    let blocks = elements.div_ceil(256).min(1024);
    let output_elements = blocks.checked_mul(2).ok_or("relative L1 partial 大小溢出")?;
    let partial_output = DeviceBuffer::allocate_reusable(device_id, output_elements.checked_mul(4).ok_or("relative L1 partial 字节溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_delta_output = output.pointer;
    let mut d_delta_input = input.pointer;
    let mut d_previous = previous.pointer;
    let mut d_partial = partial_output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "relative L1 元素数超过 u32".to_owned())?;
    let mut arguments =
        [(&mut d_delta_output as *mut *mut c_void).cast(), (&mut d_delta_input as *mut *mut c_void).cast(), (&mut d_previous as *mut *mut c_void).cast(), (&mut d_partial as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.relative_l1_delta_partial, u32::try_from(blocks).map_err(|_| "relative L1 blocks 超过 u32")?, 256, &mut arguments, "HIP relative L1 delta partial")?;
    let partial = partial_output.download_f32(output_elements)?;
    let numerator = partial.iter().step_by(2).map(|&value| f64::from(value)).sum();
    let denominator = partial.iter().skip(1).step_by(2).map(|&value| f64::from(value)).sum();
    Ok((numerator, denominator))
}

pub fn try_add_resident_bf16_f32(device_id: i32, left: &DeviceBuffer, right: &DeviceBuffer, elements: usize) -> Result<DeviceBuffer, String> {
    let left_bytes = elements.checked_mul(2).ok_or("resident BF16+F32 add left 大小溢出")?;
    let right_bytes = elements.checked_mul(4).ok_or("resident BF16+F32 add right 大小溢出")?;
    validate_resident(left, device_id, left_bytes, "BF16+F32 add left")?;
    validate_resident(right, device_id, right_bytes, "BF16+F32 add right")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, right_bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_left = left.pointer;
    let mut d_right = right.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "resident BF16+F32 add 元素数超过 u32".to_owned())?;
    let mut arguments = [(&mut d_left as *mut *mut c_void).cast(), (&mut d_right as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.add_bf16_f32, elements.div_ceil(256), 256, &mut arguments, "HIP resident BF16+F32 add")?;
    Ok(output)
}

pub fn try_zeros_resident_f32(device_id: i32, elements: usize) -> Result<DeviceBuffer, String> {
    if elements == 0 {
        return Err("HIP zeros elements 为 0".to_owned());
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, elements.checked_mul(4).ok_or("HIP zeros bytes 溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP zeros elements 超过 u32".to_owned())?;
    let mut arguments = [(&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.fill_zero, elements.div_ceil(256), 256, &mut arguments, "HIP zeros")?;
    Ok(output)
}

/// 用 kernel 参数生成常量 u32 buffer，避免为小控制量引入同步 H2D 边界。
pub fn try_fill_resident_u32(device_id: i32, value: u32, elements: usize) -> Result<DeviceBuffer, String> {
    if elements == 0 {
        return Err("HIP u32 fill elements 为 0".to_owned());
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, elements.checked_mul(4).ok_or("HIP u32 fill bytes 溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_output = output.pointer;
    let mut value = value;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP u32 fill elements 超过 u32".to_owned())?;
    let mut arguments = [(&mut d_output as *mut *mut c_void).cast(), (&mut value as *mut u32).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.fill_u32, elements.div_ceil(256), 256, &mut arguments, "HIP u32 fill")?;
    Ok(output)
}

pub fn try_concat_rows_resident_f32(device_id: i32, left: &DeviceBuffer, left_elements: usize, right: &DeviceBuffer, right_elements: usize) -> Result<DeviceBuffer, String> {
    let left_bytes = left_elements.checked_mul(4).ok_or("resident row concat left 大小溢出")?;
    let right_bytes = right_elements.checked_mul(4).ok_or("resident row concat right 大小溢出")?;
    let output_bytes = left_bytes.checked_add(right_bytes).ok_or("resident row concat output 大小溢出")?;
    validate_resident(left, device_id, left_bytes, "row concat left")?;
    validate_resident(right, device_id, right_bytes, "row concat right")?;
    set_device(device_id)?;
    // concat 属于当前 compute stream 的提交链。同步 hipMemcpy 会跑到默认流，
    // 无法可靠等待 hipMallocAsync 返回的 allocation，也会切断上游 kernel 依赖。
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;
    output.copy_from_device(0, left, 0, left_bytes)?;
    output.copy_from_device(left_bytes, right, 0, right_bytes)?;
    Ok(output)
}

pub fn try_concat_weight_rows_batched_resident_f32(device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, batch: usize, input_rows: usize, weight_rows: usize, columns: usize) -> Result<DeviceBuffer, String> {
    let input_elements = batch.checked_mul(input_rows).and_then(|v| v.checked_mul(columns)).ok_or("batched row concat input 大小溢出")?;
    let weight_elements = weight_rows.checked_mul(columns).ok_or("batched row concat weight 大小溢出")?;
    let output_elements = batch.checked_mul(input_rows + weight_rows).and_then(|v| v.checked_mul(columns)).ok_or("batched row concat output 大小溢出")?;
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("batched row concat input 字节溢出")?, "batched row concat input")?;
    validate_resident(weight, device_id, weight_elements.checked_mul(4).ok_or("batched row concat weight 字节溢出")?, "batched row concat weight")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements.checked_mul(4).ok_or("batched row concat output 字节溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_weight = weight.pointer;
    let mut d_output = output.pointer;
    let mut batch = u32::try_from(batch).map_err(|_| "batched row concat batch 超过 u32")?;
    let mut input_rows = u32::try_from(input_rows).map_err(|_| "batched row concat input rows 超过 u32")?;
    let mut weight_rows = u32::try_from(weight_rows).map_err(|_| "batched row concat weight rows 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "batched row concat columns 超过 u32")?;
    let mut args = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_weight as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut batch as *mut u32).cast(),
        (&mut input_rows as *mut u32).cast(),
        (&mut weight_rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
    ];
    let grid = u32::try_from(output_elements.div_ceil(256)).map_err(|_| "batched row concat grid 超过 u32")?;
    launch_tensor_kernel(functions.concat_weight_rows_batched, grid, 256, &mut args, "HIP batched weight row concat")?;
    Ok(output)
}

/// 在设备内按行拼接列块，避免 resident tensor 回落 host。
pub fn try_concat_columns_resident_f32(device_id: i32, left: &DeviceBuffer, right: &DeviceBuffer, rows: usize, left_columns: usize, right_columns: usize) -> Result<DeviceBuffer, String> {
    let left_bytes = rows.checked_mul(left_columns).and_then(|elements| elements.checked_mul(4)).ok_or("resident column concat left 大小溢出")?;
    let right_bytes = rows.checked_mul(right_columns).and_then(|elements| elements.checked_mul(4)).ok_or("resident column concat right 大小溢出")?;
    validate_resident(left, device_id, left_bytes, "column concat left")?;
    validate_resident(right, device_id, right_bytes, "column concat right")?;
    let columns = left_columns.checked_add(right_columns).ok_or("resident column concat columns 溢出")?;
    let elements = rows.checked_mul(columns).ok_or("resident column concat elements 溢出")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, elements.checked_mul(4).ok_or("resident column concat output 大小溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_left = left.pointer;
    let mut d_right = right.pointer;
    let mut d_output = output.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "resident column concat rows 超过 u32".to_owned())?;
    let mut left_columns = u32::try_from(left_columns).map_err(|_| "resident column concat left columns 超过 u32".to_owned())?;
    let mut right_columns = u32::try_from(right_columns).map_err(|_| "resident column concat right columns 超过 u32".to_owned())?;
    let mut arguments = [
        (&mut d_left as *mut *mut c_void).cast(),
        (&mut d_right as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut left_columns as *mut u32).cast(),
        (&mut right_columns as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.concat_columns, u32::try_from(elements.div_ceil(256)).map_err(|_| "resident column concat grid 超过 u32".to_owned())?, 256, &mut arguments, "HIP resident concat_columns")?;
    Ok(output)
}

/// 把原始 Q/K/V 的连续 head 范围压紧，减少跨卡传输。
pub fn try_compact_qkv_head_range_resident_f32(device_id: i32, input: &DeviceBuffer, rows: usize, total_head_count: usize, head_start: usize, head_count: usize, head_dim: usize) -> Result<DeviceBuffer, String> {
    if rows == 0 || total_head_count == 0 || head_count == 0 || head_start.checked_add(head_count).is_none_or(|end| end > total_head_count) || head_dim == 0 {
        return Err(format!("compact QKV rows={rows} total_heads={total_head_count} range={head_start}..{} dim={head_dim} 非法", head_start.saturating_add(head_count)));
    }
    let total_columns = total_head_count.checked_mul(head_dim).ok_or("compact QKV total columns 溢出")?;
    let local_columns = head_count.checked_mul(head_dim).ok_or("compact QKV local columns 溢出")?;
    let input_elements = rows.checked_mul(total_columns).and_then(|value| value.checked_mul(3)).ok_or("compact QKV input elements 溢出")?;
    let output_elements = rows.checked_mul(local_columns).and_then(|value| value.checked_mul(3)).ok_or("compact QKV output elements 溢出")?;
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("compact QKV input bytes 溢出")?, "compact QKV input")?;
    set_device(device_id)?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let output = DeviceBuffer::allocate_peer(device_id, output_elements.checked_mul(4).ok_or("compact QKV output bytes 溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "compact QKV rows 超过 u32".to_owned())?;
    let mut total_head_count = u32::try_from(total_head_count).map_err(|_| "compact QKV total heads 超过 u32".to_owned())?;
    let mut head_start = u32::try_from(head_start).map_err(|_| "compact QKV head start 超过 u32".to_owned())?;
    let mut head_count = u32::try_from(head_count).map_err(|_| "compact QKV heads 超过 u32".to_owned())?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "compact QKV head dim 超过 u32".to_owned())?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut total_head_count as *mut u32).cast(),
        (&mut head_start as *mut u32).cast(),
        (&mut head_count as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
    ];
    let grid = u32::try_from(output_elements.div_ceil(256)).map_err(|_| "compact QKV grid 超过 u32".to_owned())?;
    launch_tensor_kernel(functions.compact_qkv_head_range, grid, 256, &mut arguments, "HIP compact QKV head range")?;
    if let Some(started) = profile_started {
        synchronize_device(device_id, "compact QKV profile")?;
        eprintln!("[rocm-kernel] compact-qkv device={device_id} rows={rows} total_heads={total_head_count} heads={head_count} dim={head_dim} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
}

pub fn try_prefix_rows_resident_f32(device_id: i32, input: &DeviceBuffer, rows: usize, columns: usize) -> Result<DeviceBuffer, String> {
    let bytes = rows.checked_mul(columns).and_then(|elements| elements.checked_mul(4)).ok_or("resident prefix rows 大小溢出")?;
    validate_resident(input, device_id, bytes, "prefix rows input")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let runtime = RocmRuntime::open()?;
    let copy: Symbol<HipMemcpy> = runtime.symbol(&runtime.hip, b"hipMemcpy\0")?;
    let status = unsafe { copy(output.pointer, input.pointer, bytes, HIP_MEMORY_COPY_DEVICE_TO_DEVICE) };
    if status != HIP_SUCCESS {
        return Err(runtime.hip_error(status, "hipMemcpy prefix rows D2D"));
    }
    Ok(output)
}

pub fn try_take_rows_batched_resident_f32(device_id: i32, input: &DeviceBuffer, batch: usize, input_rows: usize, output_rows: usize, columns: usize) -> Result<DeviceBuffer, String> {
    let input_elements = batch.checked_mul(input_rows).and_then(|v| v.checked_mul(columns)).ok_or("batched take rows input 大小溢出")?;
    let output_elements = batch.checked_mul(output_rows).and_then(|v| v.checked_mul(columns)).ok_or("batched take rows output 大小溢出")?;
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("batched take rows input 字节溢出")?, "batched take rows input")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements.checked_mul(4).ok_or("batched take rows output 字节溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut batch = u32::try_from(batch).map_err(|_| "batched take rows batch 超过 u32")?;
    let mut input_rows = u32::try_from(input_rows).map_err(|_| "batched take rows input rows 超过 u32")?;
    let mut output_rows = u32::try_from(output_rows).map_err(|_| "batched take rows output rows 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "batched take rows columns 超过 u32")?;
    let mut args = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut batch as *mut u32).cast(),
        (&mut input_rows as *mut u32).cast(),
        (&mut output_rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
    ];
    let grid = u32::try_from(output_elements.div_ceil(256)).map_err(|_| "batched take rows grid 超过 u32")?;
    launch_tensor_kernel(functions.take_rows_batched, grid, 256, &mut args, "HIP batched take rows")?;
    Ok(output)
}

pub fn try_adaln_modulate_resident_f32(device_id: i32, input: &DeviceBuffer, shift: &DeviceBuffer, scale: &DeviceBuffer, rows: usize, columns: usize, modulation_rows: usize) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(columns).ok_or("resident AdaLN 大小溢出")?;
    let modulation_elements = modulation_rows.checked_mul(columns).ok_or("resident AdaLN modulation 大小溢出")?;
    let bytes = elements.checked_mul(4).ok_or("resident AdaLN 字节数溢出")?;
    let modulation_bytes = modulation_elements.checked_mul(4).ok_or("resident AdaLN modulation 字节数溢出")?;
    validate_resident(input, device_id, bytes, "AdaLN input")?;
    validate_resident(shift, device_id, modulation_bytes, "AdaLN shift")?;
    validate_resident(scale, device_id, modulation_bytes, "AdaLN scale")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_shift = shift.pointer;
    let mut d_scale = scale.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "resident AdaLN 元素数超过 u32".to_owned())?;
    let mut columns = u32::try_from(columns).map_err(|_| "resident AdaLN columns 超过 u32".to_owned())?;
    let mut modulation_rows = u32::try_from(modulation_rows).map_err(|_| "resident AdaLN rows 超过 u32".to_owned())?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_shift as *mut *mut c_void).cast(),
        (&mut d_scale as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut elements as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut modulation_rows as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.adaln_modulate, elements.div_ceil(256), 256, &mut arguments, "HIP resident AdaLN")?;
    Ok(output)
}

pub fn try_adaln_modulate_segmented_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    shift: &DeviceBuffer,
    scale: &DeviceBuffer,
    rows: usize,
    columns: usize,
    modulation_rows: usize,
    segments: &[(usize, usize, usize)],
) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(columns).ok_or("resident segmented AdaLN 大小溢出")?;
    let modulation_elements = modulation_rows.checked_mul(columns).ok_or("resident segmented AdaLN modulation 大小溢出")?;
    let bytes = elements.checked_mul(4).ok_or("resident segmented AdaLN 字节溢出")?;
    let modulation_bytes = modulation_elements.checked_mul(4).ok_or("resident segmented AdaLN modulation 字节溢出")?;
    validate_resident(input, device_id, bytes, "segmented AdaLN input")?;
    validate_resident(shift, device_id, modulation_bytes, "segmented AdaLN shift")?;
    validate_resident(scale, device_id, modulation_bytes, "segmented AdaLN scale")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    for &(start, end, modulation) in segments {
        if start >= end || end > rows || modulation >= modulation_rows {
            return Err(format!("resident segmented AdaLN segment=({start},{end},{modulation}) 非法"));
        }
        let mut d_input = input.pointer;
        let mut d_shift = shift.pointer;
        let mut d_scale = scale.pointer;
        let mut d_output = output.pointer;
        let mut start = u32::try_from(start).map_err(|_| "segmented AdaLN start 超过 u32")?;
        let mut row_count = u32::try_from(end - start as usize).map_err(|_| "segmented AdaLN rows 超过 u32")?;
        let mut columns = u32::try_from(columns).map_err(|_| "segmented AdaLN columns 超过 u32")?;
        let mut modulation = u32::try_from(modulation).map_err(|_| "segmented AdaLN modulation row 超过 u32")?;
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_shift as *mut *mut c_void).cast(),
            (&mut d_scale as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut start as *mut u32).cast(),
            (&mut row_count as *mut u32).cast(),
            (&mut columns as *mut u32).cast(),
            (&mut modulation as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.adaln_modulate_segment, (row_count * columns).div_ceil(256), 256, &mut arguments, "HIP segmented AdaLN")?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_rmsnorm_adaln_segmented_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    shift: &DeviceBuffer,
    scale: &DeviceBuffer,
    rows: usize,
    columns: usize,
    modulation_rows: usize,
    eps: f32,
    segments: &[(usize, usize, usize)],
) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(columns).ok_or("resident fused RMSNorm AdaLN 大小溢出")?;
    let f32_bytes = elements.checked_mul(4).ok_or("resident fused RMSNorm AdaLN F32 字节溢出")?;
    let bf16_bytes = elements.checked_mul(2).ok_or("resident fused RMSNorm AdaLN BF16 字节溢出")?;
    let input_is_bf16 = match input.bytes() {
        bytes if bytes == f32_bytes => false,
        bytes if bytes == bf16_bytes => true,
        bytes => return Err(format!("resident fused RMSNorm AdaLN input bytes={bytes}，期望 BF16={bf16_bytes} 或 F32={f32_bytes}")),
    };
    let modulation_bytes = modulation_rows.checked_mul(columns).and_then(|value| value.checked_mul(4)).ok_or("resident fused RMSNorm AdaLN modulation 字节溢出")?;
    validate_resident(input, device_id, input.bytes(), "fused RMSNorm AdaLN input")?;
    validate_resident(weight, device_id, columns.checked_mul(4).ok_or("resident fused RMSNorm AdaLN weight 字节溢出")?, "fused RMSNorm AdaLN weight")?;
    validate_resident(shift, device_id, modulation_bytes, "fused RMSNorm AdaLN shift")?;
    validate_resident(scale, device_id, modulation_bytes, "fused RMSNorm AdaLN scale")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, f32_bytes)?;
    let functions = tensor_functions(device_id)?;
    for &(start, end, modulation) in segments {
        if start >= end || end > rows || modulation >= modulation_rows {
            return Err(format!("resident fused RMSNorm AdaLN segment=({start},{end},{modulation}) 非法"));
        }
        let mut d_input = input.pointer;
        let mut d_weight = weight.pointer;
        let mut d_shift = shift.pointer;
        let mut d_scale = scale.pointer;
        let mut d_output = output.pointer;
        let mut start = u32::try_from(start).map_err(|_| "fused RMSNorm AdaLN start 超过 u32")?;
        let mut row_count = u32::try_from(end - start as usize).map_err(|_| "fused RMSNorm AdaLN rows 超过 u32")?;
        let mut columns = u32::try_from(columns).map_err(|_| "fused RMSNorm AdaLN columns 超过 u32")?;
        let mut modulation = u32::try_from(modulation).map_err(|_| "fused RMSNorm AdaLN modulation 超过 u32")?;
        let mut eps = eps;
        let mut input_is_bf16 = u32::from(input_is_bf16);
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_weight as *mut *mut c_void).cast(),
            (&mut d_shift as *mut *mut c_void).cast(),
            (&mut d_scale as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut start as *mut u32).cast(),
            (&mut row_count as *mut u32).cast(),
            (&mut columns as *mut u32).cast(),
            (&mut modulation as *mut u32).cast(),
            (&mut eps as *mut f32).cast(),
            (&mut input_is_bf16 as *mut u32).cast(),
        ];
        let threads = if columns >= 2048 { 512 } else { 256 };
        launch_tensor_kernel(functions.rmsnorm_adaln_segment, row_count, threads, &mut arguments, "HIP fused RMSNorm segmented AdaLN")?;
    }
    Ok(output)
}

pub fn try_gated_residual_segmented_resident_f32(
    device_id: i32,
    residual: &DeviceBuffer,
    update: &DeviceBuffer,
    gate: &DeviceBuffer,
    rows: usize,
    columns: usize,
    modulation_rows: usize,
    segments: &[(usize, usize, usize)],
) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(columns).ok_or("resident segmented gate 大小溢出")?;
    let gate_elements = modulation_rows.checked_mul(columns).ok_or("resident segmented gate modulation 大小溢出")?;
    let bytes = elements.checked_mul(4).ok_or("resident segmented gate 字节溢出")?;
    validate_resident(residual, device_id, bytes, "segmented gate residual")?;
    validate_resident(update, device_id, bytes, "segmented gate update")?;
    validate_resident(gate, device_id, gate_elements.checked_mul(4).ok_or("resident segmented gate 字节溢出")?, "segmented gate")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    for &(start, end, modulation) in segments {
        if start >= end || end > rows || modulation >= modulation_rows {
            return Err(format!("resident segmented gate segment=({start},{end},{modulation}) 非法"));
        }
        let mut d_residual = residual.pointer;
        let mut d_update = update.pointer;
        let mut d_gate = gate.pointer;
        let mut d_output = output.pointer;
        let mut start = u32::try_from(start).map_err(|_| "segmented gate start 超过 u32")?;
        let mut row_count = u32::try_from(end - start as usize).map_err(|_| "segmented gate rows 超过 u32")?;
        let mut columns = u32::try_from(columns).map_err(|_| "segmented gate columns 超过 u32")?;
        let mut modulation = u32::try_from(modulation).map_err(|_| "segmented gate modulation row 超过 u32")?;
        let mut arguments = [
            (&mut d_residual as *mut *mut c_void).cast(),
            (&mut d_update as *mut *mut c_void).cast(),
            (&mut d_gate as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut start as *mut u32).cast(),
            (&mut row_count as *mut u32).cast(),
            (&mut columns as *mut u32).cast(),
            (&mut modulation as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.gated_residual_segment, (row_count * columns).div_ceil(256), 256, &mut arguments, "HIP segmented gated residual")?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_silu_resident_f32(device_id: i32, input: &DeviceBuffer, elements: usize) -> Result<DeviceBuffer, String> {
    let bytes = elements.checked_mul(4).ok_or("resident SiLU 字节溢出")?;
    validate_resident(input, device_id, bytes, "SiLU input")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "SiLU elements 超过 u32")?;
    let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.silu, elements.div_ceil(256), 256, &mut arguments, "HIP resident SiLU")?;
    Ok(output)
}

pub fn try_add_row_bias_resident_f32(device_id: i32, input: &DeviceBuffer, bias: &DeviceBuffer, rows: usize, columns: usize) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(columns).ok_or("resident row bias elements 溢出")?;
    let bytes = elements.checked_mul(4).ok_or("resident row bias 字节溢出")?;
    validate_resident(input, device_id, bytes, "row bias input")?;
    validate_resident(bias, device_id, columns.checked_mul(4).ok_or("row bias 大小溢出")?, "row bias")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_bias = bias.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "row bias elements 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "row bias columns 超过 u32")?;
    let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_bias as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast(), (&mut columns as *mut u32).cast()];
    launch_tensor_kernel(functions.add_row_bias, elements.div_ceil(256), 256, &mut arguments, "HIP resident row bias")?;
    Ok(output)
}

pub fn try_modulation_chunks_resident_f32(device_id: i32, input: &DeviceBuffer, rows: usize, modalities: usize, chunks: usize, hidden: usize) -> Result<Vec<DeviceBuffer>, String> {
    if rows == 0 || modalities == 0 || chunks == 0 || hidden == 0 {
        return Err("resident modulation chunks shape 含 0".to_owned());
    }
    let input_elements = rows.checked_mul(modalities).and_then(|value| value.checked_mul(chunks)).and_then(|value| value.checked_mul(hidden)).ok_or("resident modulation input 溢出")?;
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("resident modulation input 字节溢出")?, "modulation input")?;
    let output_elements = rows.checked_mul(modalities).and_then(|value| value.checked_mul(hidden)).ok_or("resident modulation output 溢出")?;
    let output_bytes = output_elements.checked_mul(4).ok_or("resident modulation output 字节溢出")?;
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let mut output = Vec::with_capacity(chunks);
    for chunk in 0..chunks {
        let buffer = DeviceBuffer::allocate(device_id, output_bytes)?;
        let mut d_input = input.pointer;
        let mut d_output = buffer.pointer;
        let mut elements = u32::try_from(output_elements).map_err(|_| "modulation output elements 超过 u32")?;
        let mut modalities = u32::try_from(modalities).map_err(|_| "modulation modalities 超过 u32")?;
        let mut chunks = u32::try_from(chunks).map_err(|_| "modulation chunks 超过 u32")?;
        let mut hidden = u32::try_from(hidden).map_err(|_| "modulation hidden 超过 u32")?;
        let mut chunk = u32::try_from(chunk).map_err(|_| "modulation chunk 超过 u32")?;
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut elements as *mut u32).cast(),
            (&mut modalities as *mut u32).cast(),
            (&mut chunks as *mut u32).cast(),
            (&mut hidden as *mut u32).cast(),
            (&mut chunk as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.modulation_chunk, elements.div_ceil(256), 256, &mut arguments, "HIP resident modulation chunk")?;
        output.push(buffer);
    }
    Ok(output)
}
pub fn try_timestep_embedding_resident_f32(device_id: i32, timesteps: &[f32], dim: usize) -> Result<DeviceBuffer, String> {
    if timesteps.is_empty() || dim == 0 || !dim.is_multiple_of(2) {
        return Err(format!("HIP timestep batch={} dim={dim} 非法", timesteps.len()));
    }
    set_device(device_id)?;
    let timestep_bytes = std::mem::size_of_val(timesteps);
    let input = DeviceBuffer::upload(device_id, unsafe { std::slice::from_raw_parts(timesteps.as_ptr().cast::<u8>(), timestep_bytes) })?;
    let elements = timesteps.len().checked_mul(dim).ok_or("HIP timestep 输出大小溢出")?;
    let output = DeviceBuffer::allocate(device_id, elements.checked_mul(4).ok_or("HIP timestep 输出字节溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut batch = u32::try_from(timesteps.len()).map_err(|_| "HIP timestep batch 超过 u32".to_owned())?;
    let mut dim = u32::try_from(dim).map_err(|_| "HIP timestep dim 超过 u32".to_owned())?;
    let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut batch as *mut u32).cast(), (&mut dim as *mut u32).cast()];
    launch_tensor_kernel(functions.timestep_embedding, (batch * dim / 2).div_ceil(256), 256, &mut arguments, "HIP timestep embedding")?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_group_norm_resident_f32(device_id: i32, input: &DeviceBuffer, scale: &DeviceBuffer, bias: &DeviceBuffer, channels: usize, spatial: usize, groups: usize, eps: f32) -> Result<DeviceBuffer, String> {
    if groups == 0 || !channels.is_multiple_of(groups) {
        return Err(format!("HIP GroupNorm channels={channels} groups={groups} 非法"));
    }
    let elements = channels.checked_mul(spatial).ok_or("HIP GroupNorm 大小溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP GroupNorm 字节数溢出")?;
    let parameter_bytes = channels.checked_mul(4).ok_or("HIP GroupNorm 参数字节数溢出")?;
    validate_resident(input, device_id, bytes, "GroupNorm input")?;
    validate_resident(scale, device_id, parameter_bytes, "GroupNorm scale")?;
    validate_resident(bias, device_id, parameter_bytes, "GroupNorm bias")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_scale = scale.pointer;
    let mut d_bias = bias.pointer;
    let mut d_output = output.pointer;
    let mut channels = u32::try_from(channels).map_err(|_| "HIP GroupNorm channels 超过 u32".to_owned())?;
    let mut spatial = u32::try_from(spatial).map_err(|_| "HIP GroupNorm spatial 超过 u32".to_owned())?;
    let mut groups = u32::try_from(groups).map_err(|_| "HIP GroupNorm groups 超过 u32".to_owned())?;
    let mut eps = eps;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_scale as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut channels as *mut u32).cast(),
        (&mut spatial as *mut u32).cast(),
        (&mut groups as *mut u32).cast(),
        (&mut eps as *mut f32).cast(),
    ];
    launch_tensor_kernel(functions.group_norm, groups, 256, &mut arguments, "HIP GroupNorm")?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_group_norm_time_isolated_resident_f32(device_id: i32, input: &DeviceBuffer, scale: &DeviceBuffer, bias: &DeviceBuffer, channels: usize, time: usize, spatial: usize, groups: usize, eps: f32) -> Result<DeviceBuffer, String> {
    if groups == 0 || !channels.is_multiple_of(groups) || time == 0 || spatial == 0 {
        return Err(format!("HIP time-isolated GroupNorm channels={channels} time={time} spatial={spatial} groups={groups} 非法"));
    }
    let elements = channels.checked_mul(time).and_then(|value| value.checked_mul(spatial)).ok_or("HIP time-isolated GroupNorm 大小溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP time-isolated GroupNorm 字节数溢出")?;
    let parameter_bytes = channels.checked_mul(4).ok_or("HIP time-isolated GroupNorm 参数字节数溢出")?;
    validate_resident(input, device_id, bytes, "time-isolated GroupNorm input")?;
    validate_resident(scale, device_id, parameter_bytes, "time-isolated GroupNorm scale")?;
    validate_resident(bias, device_id, parameter_bytes, "time-isolated GroupNorm bias")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_scale = scale.pointer;
    let mut d_bias = bias.pointer;
    let mut d_output = output.pointer;
    let mut channels = u32::try_from(channels).map_err(|_| "HIP time-isolated GroupNorm channels 超过 u32".to_owned())?;
    let mut time = u32::try_from(time).map_err(|_| "HIP time-isolated GroupNorm time 超过 u32".to_owned())?;
    let mut spatial = u32::try_from(spatial).map_err(|_| "HIP time-isolated GroupNorm spatial 超过 u32".to_owned())?;
    let mut groups = u32::try_from(groups).map_err(|_| "HIP time-isolated GroupNorm groups 超过 u32".to_owned())?;
    let mut eps = eps;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_scale as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut channels as *mut u32).cast(),
        (&mut time as *mut u32).cast(),
        (&mut spatial as *mut u32).cast(),
        (&mut groups as *mut u32).cast(),
        (&mut eps as *mut f32).cast(),
    ];
    launch_tensor_kernel(functions.group_norm_time_isolated, time.checked_mul(groups).ok_or("HIP time-isolated GroupNorm grid 溢出")?, 256, &mut arguments, "HIP time-isolated GroupNorm")?;
    Ok(output)
}

pub fn try_snake_resident_f32(device_id: i32, input: &DeviceBuffer, alpha: &DeviceBuffer, channels: usize, columns: usize) -> Result<DeviceBuffer, String> {
    let elements = channels.checked_mul(columns).ok_or("HIP Snake 元素数溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP Snake 字节数溢出")?;
    validate_resident(input, device_id, bytes, "Snake input")?;
    validate_resident(alpha, device_id, channels.checked_mul(4).ok_or("HIP Snake alpha 字节数溢出")?, "Snake alpha")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_alpha = alpha.pointer;
    let mut d_output = output.pointer;
    let mut channels = u32::try_from(channels).map_err(|_| "HIP Snake channels 超过 u32".to_owned())?;
    let mut columns = u32::try_from(columns).map_err(|_| "HIP Snake columns 超过 u32".to_owned())?;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP Snake elements 超过 u32".to_owned())?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_alpha as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut channels as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut elements as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.snake, elements.div_ceil(256), 256, &mut arguments, "HIP Snake")?;
    Ok(output)
}

pub fn try_gelu_tanh_resident_f32(device_id: i32, input: &DeviceBuffer, elements: usize) -> Result<DeviceBuffer, String> {
    let bytes = elements.checked_mul(4).ok_or("HIP GELU 字节数溢出")?;
    validate_resident(input, device_id, bytes, "GELU input")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP GELU elements 超过 u32".to_owned())?;
    let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.gelu_tanh, elements.div_ceil(256), 256, &mut arguments, "HIP GELU")?;
    Ok(output)
}

pub fn try_channels_to_time_resident_f32(device_id: i32, input: &DeviceBuffer, batch: usize, channels: usize, time: usize) -> Result<DeviceBuffer, String> {
    if batch == 0 || channels == 0 || time == 0 {
        return Err(format!("HIP channels-to-time batch={batch} channels={channels} time={time} 非法"));
    }
    let elements = batch.checked_mul(channels).and_then(|value| value.checked_mul(time)).ok_or("HIP channels-to-time elements 溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP channels-to-time bytes 溢出")?;
    validate_resident(input, device_id, bytes, "channels-to-time input")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut batch = u32::try_from(batch).map_err(|_| "HIP channels-to-time batch 超过 u32".to_owned())?;
    let mut channels = u32::try_from(channels).map_err(|_| "HIP channels-to-time channels 超过 u32".to_owned())?;
    let mut time = u32::try_from(time).map_err(|_| "HIP channels-to-time time 超过 u32".to_owned())?;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP channels-to-time elements 超过 u32".to_owned())?;
    let mut arguments =
        [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut batch as *mut u32).cast(), (&mut channels as *mut u32).cast(), (&mut time as *mut u32).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.channels_to_time, elements.div_ceil(256), 256, &mut arguments, "HIP channels-to-time")?;
    Ok(output)
}

pub fn try_argmax_excluding_resident_f32(device_id: i32, input: &DeviceBuffer, elements: usize, excluded: &[u32]) -> Result<u32, String> {
    let mut output = try_argmax_rows_excluding_resident_f32(device_id, input, 1, elements, excluded)?;
    Ok(output.remove(0))
}

pub fn try_argmax_rows_excluding_resident_f32(device_id: i32, input: &DeviceBuffer, rows: usize, columns: usize, excluded: &[u32]) -> Result<Vec<u32>, String> {
    if rows == 0 || columns == 0 {
        return Err("HIP argmax rows 输入为空".to_owned());
    }
    let elements = rows.checked_mul(columns).ok_or("HIP argmax rows 输入大小溢出")?;
    validate_resident(input, device_id, elements.checked_mul(4).ok_or("HIP argmax rows 输入大小溢出")?, "argmax rows input")?;
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let excluded_bytes = std::mem::size_of_val(excluded);
    let output_bytes = rows.checked_mul(std::mem::size_of::<u32>()).ok_or("HIP argmax rows 输出大小溢出")?;
    let grid = u32::try_from(rows).map_err(|_| "HIP argmax rows 行数超过 u32".to_owned())?;
    let partial_count = if columns >= 4096 { columns.div_ceil(1024).min(128) } else { 1 };
    let partial_elements = rows.checked_mul(partial_count).ok_or("HIP argmax partial 元素数溢出")?;
    let partial_value_bytes = partial_elements.checked_mul(std::mem::size_of::<f32>()).ok_or("HIP argmax partial value 大小溢出")?;
    let partial_index_bytes = partial_elements.checked_mul(std::mem::size_of::<u32>()).ok_or("HIP argmax partial index 大小溢出")?;
    with_tensor_workspace(device_id, &[excluded_bytes, output_bytes, partial_value_bytes, partial_index_bytes], |workspace| {
        if !excluded.is_empty() {
            workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(excluded.as_ptr().cast(), excluded_bytes) })?;
        }
        let mut d_input = input.pointer;
        let mut d_excluded = if excluded.is_empty() { ptr::null_mut() } else { workspace.buffer(0).pointer };
        let mut d_output = workspace.buffer(1).pointer;
        let mut columns = u32::try_from(columns).map_err(|_| "HIP argmax rows columns 超过 u32".to_owned())?;
        let mut excluded_count = u32::try_from(excluded.len()).map_err(|_| "HIP argmax 排除项超过 u32".to_owned())?;
        if partial_count == 1 {
            let mut arguments =
                [(&mut d_input as *mut *mut c_void).cast(), (&mut columns as *mut u32).cast(), (&mut d_excluded as *mut *mut c_void).cast(), (&mut excluded_count as *mut u32).cast(), (&mut d_output as *mut *mut c_void).cast()];
            launch_tensor_kernel(functions.argmax_rows_excluding, grid, 256, &mut arguments, "HIP resident row argmax")?;
        } else {
            let mut d_partial_values = workspace.buffer(2).pointer;
            let mut d_partial_indices = workspace.buffer(3).pointer;
            let mut partial_count = u32::try_from(partial_count).map_err(|_| "HIP argmax partial 数量超过 u32")?;
            let partial_grid = grid.checked_mul(partial_count).ok_or("HIP argmax partial grid 溢出")?;
            let mut partial_arguments = [
                (&mut d_input as *mut *mut c_void).cast(),
                (&mut columns as *mut u32).cast(),
                (&mut d_excluded as *mut *mut c_void).cast(),
                (&mut excluded_count as *mut u32).cast(),
                (&mut d_partial_values as *mut *mut c_void).cast(),
                (&mut d_partial_indices as *mut *mut c_void).cast(),
                (&mut partial_count as *mut u32).cast(),
            ];
            launch_tensor_kernel(functions.argmax_rows_excluding_partial, partial_grid, 256, &mut partial_arguments, "HIP resident sharded row argmax")?;
            let mut merge_arguments = [(&mut d_partial_values as *mut *mut c_void).cast(), (&mut d_partial_indices as *mut *mut c_void).cast(), (&mut partial_count as *mut u32).cast(), (&mut d_output as *mut *mut c_void).cast()];
            launch_tensor_kernel(functions.argmax_rows_merge_partial, grid, 256, &mut merge_arguments, "HIP resident merge row argmax")?;
        }
        let mut output = vec![u32::MAX; rows];
        workspace.buffer(1).copy_to_host(unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), output_bytes) })?;
        if output.contains(&u32::MAX) {
            return Err("HIP argmax rows 存在没有有限可选 token 的行".to_owned());
        }
        Ok(output)
    })
}

pub fn try_argmax_add_rows_resident_f32(device_id: i32, logits: &DeviceBuffer, logits_rows: usize, columns: usize, selected_rows: &[u32], bias: &DeviceBuffer) -> Result<Vec<u32>, String> {
    if selected_rows.is_empty() || columns == 0 {
        return Err("HIP add rows argmax 输入为空".to_owned());
    }
    if selected_rows.iter().any(|&row| row as usize >= logits_rows) {
        return Err(format!("HIP add rows argmax row 超过 logits_rows={logits_rows}"));
    }
    let logits_elements = logits_rows.checked_mul(columns).ok_or("HIP add rows argmax logits 大小溢出")?;
    let bias_elements = selected_rows.len().checked_mul(columns).ok_or("HIP add rows argmax bias 大小溢出")?;
    validate_resident(logits, device_id, logits_elements.checked_mul(4).ok_or("HIP add rows argmax logits bytes 溢出")?, "add rows argmax logits")?;
    validate_resident(bias, device_id, bias_elements.checked_mul(4).ok_or("HIP add rows argmax bias bytes 溢出")?, "add rows argmax bias")?;
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let rows_bytes = std::mem::size_of_val(selected_rows);
    let output_bytes = rows_bytes;
    let grid = u32::try_from(selected_rows.len()).map_err(|_| "HIP add rows argmax rows 超过 u32")?;
    let partial_count = if columns >= 4096 { columns.div_ceil(1024).min(128) } else { 1 };
    let partial_elements = selected_rows.len().checked_mul(partial_count).ok_or("HIP add rows argmax partial 元素数溢出")?;
    let partial_value_bytes = partial_elements.checked_mul(std::mem::size_of::<f32>()).ok_or("HIP add rows argmax partial value 大小溢出")?;
    let partial_index_bytes = partial_elements.checked_mul(std::mem::size_of::<u32>()).ok_or("HIP add rows argmax partial index 大小溢出")?;
    with_tensor_workspace(device_id, &[rows_bytes, output_bytes, partial_value_bytes, partial_index_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(selected_rows.as_ptr().cast(), rows_bytes) })?;
        let mut d_logits = logits.pointer;
        let mut d_rows = workspace.buffer(0).pointer;
        let mut d_bias = bias.pointer;
        let mut d_output = workspace.buffer(1).pointer;
        let mut d_partial_values = workspace.buffer(2).pointer;
        let mut d_partial_indices = workspace.buffer(3).pointer;
        let mut columns = u32::try_from(columns).map_err(|_| "HIP add rows argmax columns 超过 u32")?;
        let mut partial_count = u32::try_from(partial_count).map_err(|_| "HIP add rows argmax partial 数量超过 u32")?;
        let partial_grid = grid.checked_mul(partial_count).ok_or("HIP add rows argmax partial grid 溢出")?;
        let mut partial_arguments = [
            (&mut d_logits as *mut *mut c_void).cast(),
            (&mut d_rows as *mut *mut c_void).cast(),
            (&mut d_bias as *mut *mut c_void).cast(),
            (&mut columns as *mut u32).cast(),
            (&mut d_partial_values as *mut *mut c_void).cast(),
            (&mut d_partial_indices as *mut *mut c_void).cast(),
            (&mut partial_count as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.argmax_add_rows_partial, partial_grid, 256, &mut partial_arguments, "HIP resident add rows argmax")?;
        let mut merge_arguments = [(&mut d_partial_values as *mut *mut c_void).cast(), (&mut d_partial_indices as *mut *mut c_void).cast(), (&mut partial_count as *mut u32).cast(), (&mut d_output as *mut *mut c_void).cast()];
        launch_tensor_kernel(functions.argmax_rows_merge_partial, grid, 256, &mut merge_arguments, "HIP resident merge add rows argmax")?;
        let mut output = vec![u32::MAX; selected_rows.len()];
        workspace.buffer(1).copy_to_host(unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), output_bytes) })?;
        if output.contains(&u32::MAX) {
            return Err("HIP add rows argmax 存在没有有限可选 token 的行".to_owned());
        }
        Ok(output)
    })
}

pub fn try_sample_top_p_rows_excluding_resident_f32(device_id: i32, input: &DeviceBuffer, rows: usize, columns: usize, sampling: &[crate::backend::TokenSampling], excluded: &[u32]) -> Result<Vec<u32>, String> {
    if rows == 0 || columns == 0 || rows != sampling.len() {
        return Err(format!("HIP sample rows shape 非法: rows={rows} columns={columns} sampling={}", sampling.len()));
    }
    for sample in sampling {
        if !sample.temperature.is_finite() || !(0.0..=2.0).contains(&sample.temperature) || !sample.top_p.is_finite() || !(0.0..=1.0).contains(&sample.top_p) || !sample.random.is_finite() || !(0.0..1.0).contains(&sample.random) {
            return Err(format!("HIP sample rows 参数非法: {sample:?}"));
        }
    }
    let elements = rows.checked_mul(columns).ok_or("HIP sample rows 输入大小溢出")?;
    validate_resident(input, device_id, elements.checked_mul(4).ok_or("HIP sample rows 输入大小溢出")?, "sample rows input")?;
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let sampling_bytes = std::mem::size_of_val(sampling);
    let excluded_bytes = std::mem::size_of_val(excluded);
    let probability_bytes = elements.checked_mul(std::mem::size_of::<f32>()).ok_or("HIP sample rows 概率缓冲大小溢出")?;
    let output_bytes = rows.checked_mul(std::mem::size_of::<u32>()).ok_or("HIP sample rows 输出大小溢出")?;
    let grid = u32::try_from(rows).map_err(|_| "HIP sample rows 行数超过 u32".to_owned())?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let output = with_tensor_workspace(device_id, &[sampling_bytes, excluded_bytes, probability_bytes, output_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(sampling.as_ptr().cast(), sampling_bytes) })?;
        if !excluded.is_empty() {
            workspace.buffer(1).copy_from_host(unsafe { std::slice::from_raw_parts(excluded.as_ptr().cast(), excluded_bytes) })?;
        }
        let mut d_input = input.pointer;
        let mut d_probabilities = workspace.buffer(2).pointer;
        let mut d_sampling = workspace.buffer(0).pointer;
        let mut d_excluded = if excluded.is_empty() { ptr::null_mut() } else { workspace.buffer(1).pointer };
        let mut d_output = workspace.buffer(3).pointer;
        let mut columns = u32::try_from(columns).map_err(|_| "HIP sample rows columns 超过 u32".to_owned())?;
        let mut excluded_count = u32::try_from(excluded.len()).map_err(|_| "HIP sample rows 排除项超过 u32".to_owned())?;
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_probabilities as *mut *mut c_void).cast(),
            (&mut columns as *mut u32).cast(),
            (&mut d_sampling as *mut *mut c_void).cast(),
            (&mut d_excluded as *mut *mut c_void).cast(),
            (&mut excluded_count as *mut u32).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
        ];
        launch_tensor_kernel(functions.sample_top_p_rows_excluding, grid, 1024, &mut arguments, "HIP resident row top-p")?;
        let mut output = vec![u32::MAX; rows];
        workspace.buffer(3).copy_to_host(unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), output_bytes) })?;
        if output.contains(&u32::MAX) {
            return Err("HIP sample rows 存在没有可选 token 的行".to_owned());
        }
        Ok(output)
    })?;
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] top-p device={device_id} rows={rows} columns={columns} excluded={} wall={:.6}s", excluded.len(), started.elapsed().as_secs_f64());
    }
    Ok(output)
}

pub fn try_select_rows_resident_f32(device_id: i32, input: &DeviceBuffer, rows: usize, cols: usize, selected: &[u32]) -> Result<DeviceBuffer, String> {
    if rows == 0 || cols == 0 || selected.is_empty() {
        return Err(format!("HIP resident select_rows shape=[{rows},{cols}] selected={} 非法", selected.len()));
    }
    if selected.iter().any(|&row| row as usize >= rows) {
        return Err(format!("HIP resident select_rows 行越界，input_rows={rows} rows={selected:?}"));
    }
    let input_elements = rows.checked_mul(cols).ok_or("HIP resident select_rows input 大小溢出")?;
    let f32_bytes = input_elements.checked_mul(4).ok_or("HIP resident select_rows F32 字节溢出")?;
    let bf16_bytes = input_elements.checked_mul(2).ok_or("HIP resident select_rows BF16 字节溢出")?;
    let input_is_bf16 = match input.bytes() {
        bytes if bytes == f32_bytes => false,
        bytes if bytes == bf16_bytes => true,
        bytes => return Err(format!("HIP resident select_rows input 字节={bytes}，期望 BF16={bf16_bytes} 或 F32={f32_bytes}")),
    };
    validate_resident(input, device_id, input.bytes(), "select_rows input")?;
    let output_elements = selected.len().checked_mul(cols).ok_or("HIP resident select_rows output 大小溢出")?;
    let output_bytes = output_elements.checked_mul(4).ok_or("HIP resident select_rows output 字节溢出")?;
    set_device(device_id)?;
    let selected_bytes = std::mem::size_of_val(selected);
    let indices = DeviceBuffer::upload(device_id, unsafe { std::slice::from_raw_parts(selected.as_ptr().cast(), selected_bytes) })?;
    let output = DeviceBuffer::allocate(device_id, output_bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_selected = indices.pointer;
    let mut d_output = output.pointer;
    let mut selected_count = u32::try_from(selected.len()).map_err(|_| "HIP resident select_rows selected_count 超过 u32")?;
    let mut cols = u32::try_from(cols).map_err(|_| "HIP resident select_rows cols 超过 u32")?;
    let mut input_is_bf16 = u32::from(input_is_bf16);
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_selected as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut selected_count as *mut u32).cast(),
        (&mut cols as *mut u32).cast(),
        (&mut input_is_bf16 as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.select_rows, u32::try_from(output_elements).map_err(|_| "HIP resident select_rows output elements 超过 u32")?.div_ceil(256), 256, &mut arguments, "HIP resident select_rows")?;
    Ok(output)
}

pub fn try_rmsnorm_resident_to_f32(device_id: i32, input: &DeviceBuffer, weight: &[f32], rows: usize, cols: usize, eps: f32, gemma: bool) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(cols).ok_or("resident RMSNorm 大小溢出")?;
    let output_bytes = elements.checked_mul(4).ok_or("resident RMSNorm 字节溢出")?;
    let bf16_bytes = elements.checked_mul(2).ok_or("resident RMSNorm BF16 字节溢出")?;
    let input_is_bf16 = match input.bytes() {
        bytes if bytes == output_bytes => false,
        bytes if bytes == bf16_bytes => true,
        bytes => return Err(format!("resident RMSNorm input 字节={bytes}，期望 BF16={bf16_bytes} 或 F32={output_bytes}")),
    };
    validate_resident(input, device_id, input.bytes(), "RMSNorm input")?;
    if weight.len() != cols {
        return Err(format!("resident RMSNorm weight={}，期望 {cols}", weight.len()));
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;
    let functions = tensor_functions(device_id)?;
    let weight_bytes = std::mem::size_of_val(weight);
    with_tensor_workspace(device_id, &[weight_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(weight.as_ptr().cast(), weight_bytes) })?;
        let mut d_input = input.pointer;
        let mut d_weight = workspace.buffer(0).pointer;
        let mut d_output = output.pointer;
        let mut d_quantized_output = ptr::null_mut();
        let mut rows = u32::try_from(rows).map_err(|_| "resident RMSNorm rows 超过 u32".to_owned())?;
        let mut cols = u32::try_from(cols).map_err(|_| "resident RMSNorm cols 超过 u32".to_owned())?;
        let mut eps = eps;
        let mut gemma = u32::from(gemma);
        let mut input_is_bf16 = u32::from(input_is_bf16);
        let mut output_is_bf16 = 0u32;
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_weight as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut d_quantized_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut cols as *mut u32).cast(),
            (&mut eps as *mut f32).cast(),
            (&mut gemma as *mut u32).cast(),
            (&mut input_is_bf16 as *mut u32).cast(),
            (&mut output_is_bf16 as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.rmsnorm, rows, 256, &mut arguments, "HIP resident rmsnorm")
    })?;
    Ok(output)
}

pub fn try_rmsnorm_resident_weight_to_f32(device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, rows: usize, cols: usize, eps: f32, gemma: bool) -> Result<DeviceBuffer, String> {
    Ok(try_rmsnorm_resident_weight(device_id, input, weight, rows, cols, eps, gemma, RmsnormOutput::F32)?.0)
}

pub fn try_rmsnorm_resident_weight_to_bf16(device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, rows: usize, cols: usize, eps: f32, gemma: bool) -> Result<DeviceBuffer, String> {
    Ok(try_rmsnorm_resident_weight(device_id, input, weight, rows, cols, eps, gemma, RmsnormOutput::Bf16)?.0)
}

#[derive(Clone, Copy)]
enum RmsnormOutput {
    F32,
    Bf16,
    F32AndBf16,
}

fn try_rmsnorm_resident_weight(device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, rows: usize, cols: usize, eps: f32, gemma: bool, output_kind: RmsnormOutput) -> Result<(DeviceBuffer, Option<DeviceBuffer>), String> {
    let elements = rows.checked_mul(cols).ok_or("resident weight RMSNorm 大小溢出")?;
    let f32_bytes = elements.checked_mul(4).ok_or("resident weight RMSNorm F32 字节溢出")?;
    let bf16_bytes = elements.checked_mul(2).ok_or("resident weight RMSNorm BF16 字节溢出")?;
    let output_is_bf16 = matches!(output_kind, RmsnormOutput::Bf16);
    let output_bytes = if output_is_bf16 { bf16_bytes } else { f32_bytes };
    let input_is_bf16 = match input.bytes() {
        bytes if bytes == f32_bytes => false,
        bytes if bytes == bf16_bytes => true,
        bytes => return Err(format!("resident weight RMSNorm input 字节={bytes}，期望 BF16={bf16_bytes} 或 F32={f32_bytes}")),
    };
    validate_resident(input, device_id, input.bytes(), "resident weight RMSNorm input")?;
    validate_resident(weight, device_id, cols.checked_mul(4).ok_or("resident weight RMSNorm 参数字节溢出")?, "resident weight RMSNorm weight")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;
    // dual RMSNorm 的 BF16 分支会被 cooperative MoE 直接跨卡消费；从源头
    // 放入显式池，避免 async allocation 再 deferred D2D 的跨 stream 可见性窗口。
    let quantized = matches!(output_kind, RmsnormOutput::F32AndBf16).then(|| DeviceBuffer::allocate_peer(device_id, bf16_bytes)).transpose()?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_weight = weight.pointer;
    let mut d_output = output.pointer;
    let mut d_quantized_output = quantized.as_ref().map_or(ptr::null_mut(), |buffer| buffer.pointer);
    let mut rows = u32::try_from(rows).map_err(|_| "resident weight RMSNorm rows 超过 u32".to_owned())?;
    let mut cols = u32::try_from(cols).map_err(|_| "resident weight RMSNorm cols 超过 u32".to_owned())?;
    let mut eps = eps;
    let mut gemma = u32::from(gemma);
    let mut input_is_bf16 = u32::from(input_is_bf16);
    let mut output_is_bf16 = u32::from(output_is_bf16);
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_weight as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut d_quantized_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut cols as *mut u32).cast(),
        (&mut eps as *mut f32).cast(),
        (&mut gemma as *mut u32).cast(),
        (&mut input_is_bf16 as *mut u32).cast(),
        (&mut output_is_bf16 as *mut u32).cast(),
    ];
    // 同一行不能因为邻接行数量改变归约树，否则连续批处理会改变解码结果。
    let threads = if cols >= 2048 { 512 } else { 256 };
    launch_tensor_kernel(functions.rmsnorm, rows, threads, &mut arguments, "HIP resident weight rmsnorm")?;
    Ok((output, quantized))
}

pub fn try_rmsnorm_resident_weight_to_f32_bf16(device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, rows: usize, cols: usize, eps: f32, gemma: bool) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    let (output, quantized) = try_rmsnorm_resident_weight(device_id, input, weight, rows, cols, eps, gemma, RmsnormOutput::F32AndBf16)?;
    Ok((output, quantized.ok_or("resident weight RMSNorm dual 缺少 BF16 输出")?))
}

pub fn try_layernorm_bias_resident_f32(device_id: i32, input: &DeviceBuffer, weight: &DeviceBuffer, bias: &DeviceBuffer, rows: usize, cols: usize, eps: f32) -> Result<DeviceBuffer, String> {
    let bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(4)).ok_or("resident LayerNorm 大小溢出")?;
    validate_resident(input, device_id, bytes, "LayerNorm input")?;
    let parameter_bytes = cols.checked_mul(4).ok_or("resident LayerNorm 参数大小溢出")?;
    validate_resident(weight, device_id, parameter_bytes, "LayerNorm weight")?;
    validate_resident(bias, device_id, parameter_bytes, "LayerNorm bias")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    {
        let mut d_input = input.pointer;
        let mut d_weight = weight.pointer;
        let mut d_bias = bias.pointer;
        let mut d_output = output.pointer;
        let mut rows = u32::try_from(rows).map_err(|_| "resident LayerNorm rows 超过 u32".to_owned())?;
        let mut cols = u32::try_from(cols).map_err(|_| "resident LayerNorm cols 超过 u32".to_owned())?;
        let mut eps = eps;
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_weight as *mut *mut c_void).cast(),
            (&mut d_bias as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut cols as *mut u32).cast(),
            (&mut eps as *mut f32).cast(),
        ];
        launch_tensor_kernel(functions.layernorm_bias, rows, 256, &mut arguments, "HIP resident layernorm")?;
    }
    Ok(output)
}

pub fn try_rmsnorm_heads_unit_resident_f32(device_id: i32, input: &DeviceBuffer, rows: usize, heads: usize, head_dim: usize, eps: f32) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(heads).and_then(|n| n.checked_mul(head_dim)).ok_or("resident head RMSNorm 大小溢出")?;
    let bytes = elements.checked_mul(4).ok_or("resident head RMSNorm 字节数溢出")?;
    validate_resident(input, device_id, bytes, "head RMSNorm input")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let threads = u32::try_from(head_dim.next_power_of_two().clamp(32, 256)).map_err(|_| "resident head RMSNorm block size 超过 u32")?;
    let mut d_output = output.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "resident head RMSNorm rows 超过 u32")?;
    let mut heads = u32::try_from(heads).map_err(|_| "resident head RMSNorm heads 超过 u32")?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "resident head RMSNorm head_dim 超过 u32")?;
    let mut eps = eps;
    let mut arguments =
        [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut rows as *mut u32).cast(), (&mut heads as *mut u32).cast(), (&mut head_dim as *mut u32).cast(), (&mut eps as *mut f32).cast()];
    launch_tensor_kernel(functions.rmsnorm_heads_unit, rows.checked_mul(heads).ok_or("resident head RMSNorm grid 溢出")?, threads, &mut arguments, "HIP resident unit head RMSNorm")?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_rmsnorm_rope_pair_unit_resident_f32(
    device_id: i32,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    rows: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
    cosine: &[f32],
    sine: &[f32],
) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    if rows == 0 || heads == 0 || head_dim == 0 || head_dim > 256 || rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!("resident QK norm+RoPE rows={rows} heads={heads} head_dim={head_dim} rotary_dim={rotary_dim} 非法"));
    }
    let elements = rows.checked_mul(heads).and_then(|n| n.checked_mul(head_dim)).ok_or("resident QK norm+RoPE 大小溢出")?;
    let bytes = elements.checked_mul(4).ok_or("resident QK norm+RoPE 字节数溢出")?;
    let table_elements = rows.checked_mul(rotary_dim / 2).ok_or("resident QK norm+RoPE table 大小溢出")?;
    if cosine.len() != table_elements || sine.len() != table_elements {
        return Err(format!("resident QK norm+RoPE table cosine={} sine={}，期望 {table_elements}", cosine.len(), sine.len()));
    }
    validate_resident(query, device_id, bytes, "QK norm+RoPE query")?;
    validate_resident(key, device_id, bytes, "QK norm+RoPE key")?;
    set_device(device_id)?;
    let query_output = DeviceBuffer::allocate(device_id, bytes)?;
    let key_output = DeviceBuffer::allocate(device_id, bytes)?;
    let (cosine, sine) = resident_rope_tables(device_id, cosine, sine, rotary_dim / 2, 0..rows)?;
    let functions = tensor_functions(device_id)?;
    let threads = u32::try_from(head_dim.next_power_of_two().clamp(32, 256)).map_err(|_| "resident QK norm+RoPE block size 超过 u32")?;
    let mut d_query = query.pointer;
    let mut d_key = key.pointer;
    let mut d_cosine = cosine.pointer;
    let mut d_sine = sine.pointer;
    let mut d_query_output = query_output.pointer;
    let mut d_key_output = key_output.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "resident QK norm+RoPE rows 超过 u32")?;
    let mut heads_u32 = u32::try_from(heads).map_err(|_| "resident QK norm+RoPE heads 超过 u32")?;
    let mut head_dim_u32 = u32::try_from(head_dim).map_err(|_| "resident QK norm+RoPE head_dim 超过 u32")?;
    let mut rotary_dim_u32 = u32::try_from(rotary_dim).map_err(|_| "resident QK norm+RoPE rotary_dim 超过 u32")?;
    let mut eps = eps;
    let mut arguments = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_cosine as *mut *mut c_void).cast(),
        (&mut d_sine as *mut *mut c_void).cast(),
        (&mut d_query_output as *mut *mut c_void).cast(),
        (&mut d_key_output as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut heads_u32 as *mut u32).cast(),
        (&mut head_dim_u32 as *mut u32).cast(),
        (&mut rotary_dim_u32 as *mut u32).cast(),
        (&mut eps as *mut f32).cast(),
    ];
    let grid = rows_u32.checked_mul(heads_u32).ok_or("resident QK norm+RoPE grid 溢出")?;
    launch_tensor_kernel(functions.rmsnorm_rope_pair_unit, grid, threads, &mut arguments, "HIP resident QK norm+RoPE")?;
    Ok((query_output, key_output))
}

pub fn try_scaled_residual_columns_resident_f32(device_id: i32, input: &DeviceBuffer, update: &DeviceBuffer, bias: Option<&DeviceBuffer>, scale: &DeviceBuffer, elements: usize, columns: usize) -> Result<DeviceBuffer, String> {
    let bytes = elements.checked_mul(4).ok_or("resident scaled residual 大小溢出")?;
    let scale_bytes = columns.checked_mul(4).ok_or("resident scaled residual scale 溢出")?;
    validate_resident(input, device_id, bytes, "scaled residual input")?;
    validate_resident(update, device_id, bytes, "scaled residual update")?;
    if let Some(bias) = bias {
        validate_resident(bias, device_id, scale_bytes, "scaled residual bias")?;
    }
    validate_resident(scale, device_id, scale_bytes, "scaled residual scale")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_update = update.pointer;
    let mut d_bias = bias.map_or(ptr::null_mut(), |bias| bias.pointer);
    let mut d_scale = scale.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "resident scaled residual elements 超过 u32")?;
    let mut columns = u32::try_from(columns).map_err(|_| "resident scaled residual columns 超过 u32")?;
    let mut has_bias = u32::from(bias.is_some());
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_update as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_scale as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut elements as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut has_bias as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.scaled_residual_columns, elements.div_ceil(256), 256, &mut arguments, "HIP resident scaled residual")?;
    Ok(output)
}

pub fn try_unpatch_affine_resident_f32(device_id: i32, input: &DeviceBuffer, scale: &DeviceBuffer, bias: &DeviceBuffer, shape: [usize; 3], patch: [usize; 3], channels: usize) -> Result<DeviceBuffer, String> {
    if shape.contains(&0) || patch.contains(&0) || channels == 0 || (0..3).any(|axis| shape[axis] % patch[axis] != 0) {
        return Err("resident unpatch affine 参数非法".to_owned());
    }
    let elements = shape.into_iter().product::<usize>().checked_mul(channels).ok_or("resident unpatch affine output 溢出")?;
    let bytes = elements.checked_mul(4).ok_or("resident unpatch affine 字节溢出")?;
    let parameter_bytes = channels.checked_mul(4).ok_or("resident unpatch affine 参数字节溢出")?;
    validate_resident(input, device_id, bytes, "unpatch affine input")?;
    validate_resident(scale, device_id, parameter_bytes, "unpatch affine scale")?;
    validate_resident(bias, device_id, parameter_bytes, "unpatch affine bias")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_scale = scale.pointer;
    let mut d_bias = bias.pointer;
    let mut d_output = output.pointer;
    let mut channels = u32::try_from(channels).map_err(|_| "resident unpatch affine channels 超过 u32")?;
    let mut time = u32::try_from(shape[0]).map_err(|_| "resident unpatch affine time 超过 u32")?;
    let mut height = u32::try_from(shape[1]).map_err(|_| "resident unpatch affine height 超过 u32")?;
    let mut width = u32::try_from(shape[2]).map_err(|_| "resident unpatch affine width 超过 u32")?;
    let mut patch_t = u32::try_from(patch[0]).map_err(|_| "resident unpatch affine patch_t 超过 u32")?;
    let mut patch_h = u32::try_from(patch[1]).map_err(|_| "resident unpatch affine patch_h 超过 u32")?;
    let mut patch_w = u32::try_from(patch[2]).map_err(|_| "resident unpatch affine patch_w 超过 u32")?;
    let mut elements = u32::try_from(elements).map_err(|_| "resident unpatch affine elements 超过 u32")?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_scale as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut channels as *mut u32).cast(),
        (&mut time as *mut u32).cast(),
        (&mut height as *mut u32).cast(),
        (&mut width as *mut u32).cast(),
        (&mut patch_t as *mut u32).cast(),
        (&mut patch_h as *mut u32).cast(),
        (&mut patch_w as *mut u32).cast(),
        (&mut elements as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.unpatch_affine, elements.div_ceil(256), 256, &mut arguments, "HIP resident unpatch affine")?;
    Ok(output)
}

fn activation_parameters(activation: &Activation) -> (u32, f32, f32, f32, f32, u32) {
    match activation {
        Activation::Silu => (0, 0.0, 0.0, 0.0, 0.0, 0),
        Activation::SiluClamped { limit } => (1, 0.0, 0.0, *limit, 0.0, 0),
        Activation::Situ { beta, linear_beta } => (2, 0.0, *beta, 0.0, linear_beta.unwrap_or(0.0), u32::from(linear_beta.is_some())),
        Activation::SwigluOai { alpha, limit } => (3, *alpha, 0.0, *limit, 0.0, 0),
        Activation::GeluTanh => (4, 0.0, 0.0, 0.0, 0.0, 0),
    }
}

pub fn try_gated_activation_resident_f32(device_id: i32, gate: &DeviceBuffer, up: &DeviceBuffer, elements: usize, activation: &Activation) -> Result<DeviceBuffer, String> {
    let bytes = elements.checked_mul(4).ok_or("resident gated activation 大小溢出")?;
    validate_resident(gate, device_id, bytes, "gated gate")?;
    validate_resident(up, device_id, bytes, "gated up")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate_reusable(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let (kind, alpha, beta, limit, linear_beta, has_linear_beta) = activation_parameters(activation);
    let mut d_gate = gate.pointer;
    let mut d_up = up.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "resident gated 元素数超过 u32".to_owned())?;
    let mut kind = kind;
    let mut alpha = alpha;
    let mut beta = beta;
    let mut limit = limit;
    let mut linear_beta = linear_beta;
    let mut has_linear_beta = has_linear_beta;
    let mut arguments = [
        (&mut d_gate as *mut *mut c_void).cast(),
        (&mut d_up as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut elements as *mut u32).cast(),
        (&mut kind as *mut u32).cast(),
        (&mut alpha as *mut f32).cast(),
        (&mut beta as *mut f32).cast(),
        (&mut limit as *mut f32).cast(),
        (&mut linear_beta as *mut f32).cast(),
        (&mut has_linear_beta as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.gated_activation, elements.div_ceil(256), 256, &mut arguments, "HIP resident gated")?;
    Ok(output)
}

pub fn try_gated_activation_resident_bf16(device_id: i32, gate: &DeviceBuffer, up: &DeviceBuffer, elements: usize, activation: &Activation) -> Result<DeviceBuffer, String> {
    let input_bytes = elements.checked_mul(4).ok_or("resident gated activation 输入大小溢出")?;
    validate_resident(gate, device_id, input_bytes, "gated gate")?;
    validate_resident(up, device_id, input_bytes, "gated up")?;
    set_device(device_id)?;
    let output_bytes = elements.checked_mul(2).ok_or("resident gated activation BF16 输出大小溢出")?;
    let output = DeviceBuffer::allocate_reusable(device_id, output_bytes)?;
    let functions = tensor_functions(device_id)?;
    let (kind, alpha, beta, limit, linear_beta, has_linear_beta) = activation_parameters(activation);
    let mut d_gate = gate.pointer;
    let mut d_up = up.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "resident gated BF16 元素数超过 u32".to_owned())?;
    let mut kind = kind;
    let mut alpha = alpha;
    let mut beta = beta;
    let mut limit = limit;
    let mut linear_beta = linear_beta;
    let mut has_linear_beta = has_linear_beta;
    let mut arguments = [
        (&mut d_gate as *mut *mut c_void).cast(),
        (&mut d_up as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut elements as *mut u32).cast(),
        (&mut kind as *mut u32).cast(),
        (&mut alpha as *mut f32).cast(),
        (&mut beta as *mut f32).cast(),
        (&mut limit as *mut f32).cast(),
        (&mut linear_beta as *mut f32).cast(),
        (&mut has_linear_beta as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.gated_activation_bf16, elements.div_ceil(256), 256, &mut arguments, "HIP resident gated BF16")?;
    Ok(output)
}

pub fn try_sigmoid_gate_resident_f32(device_id: i32, input: &DeviceBuffer, gate: &DeviceBuffer, rows: usize, cols: usize) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(cols).ok_or("HIP sigmoid gate 元素数溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP sigmoid gate 字节数溢出")?;
    validate_resident(input, device_id, bytes, "sigmoid gate input")?;
    let row_bytes = rows.checked_mul(4).ok_or("HIP sigmoid gate 行参数字节数溢出")?;
    let gate_columns = if gate.bytes == row_bytes {
        1
    } else if gate.bytes == bytes {
        cols
    } else {
        return Err(format!("HIP sigmoid gate bytes={}，期望 {row_bytes} 或 {bytes}", gate.bytes));
    };
    validate_resident(gate, device_id, gate.bytes, "sigmoid gate")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_gate = gate.pointer;
    let mut d_output = output.pointer;
    let mut cols = u32::try_from(cols).map_err(|_| "HIP sigmoid gate cols 超过 u32".to_owned())?;
    let mut gate_columns = u32::try_from(gate_columns).map_err(|_| "HIP sigmoid gate gate cols 超过 u32".to_owned())?;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP sigmoid gate 元素数超过 u32".to_owned())?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_gate as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut cols as *mut u32).cast(),
        (&mut gate_columns as *mut u32).cast(),
        (&mut elements as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.sigmoid_gate, elements.div_ceil(256), 256, &mut arguments, "HIP resident sigmoid gate")?;
    Ok(output)
}

pub fn try_split_interleaved_columns_resident_f32(device_id: i32, input: &DeviceBuffer, rows: usize, cols: usize, block_columns: usize) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    let pair_columns = block_columns.checked_mul(2).ok_or("HIP split interleaved block 溢出")?;
    if block_columns == 0 || !cols.is_multiple_of(pair_columns) {
        return Err(format!("HIP split interleaved cols={cols} block={block_columns} 非法"));
    }
    let input_elements = rows.checked_mul(cols).ok_or("HIP split interleaved 输入元素数溢出")?;
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("HIP split interleaved 输入字节数溢出")?, "split interleaved input")?;
    let output_elements = input_elements / 2;
    let output_bytes = output_elements.checked_mul(4).ok_or("HIP split interleaved 输出字节数溢出")?;
    set_device(device_id)?;
    let left = DeviceBuffer::allocate(device_id, output_bytes)?;
    let right = DeviceBuffer::allocate(device_id, output_bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_left = left.pointer;
    let mut d_right = right.pointer;
    let mut cols = u32::try_from(cols).map_err(|_| "HIP split interleaved cols 超过 u32".to_owned())?;
    let mut block_columns = u32::try_from(block_columns).map_err(|_| "HIP split interleaved block 超过 u32".to_owned())?;
    let mut output_elements = u32::try_from(output_elements).map_err(|_| "HIP split interleaved 元素数超过 u32".to_owned())?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_left as *mut *mut c_void).cast(),
        (&mut d_right as *mut *mut c_void).cast(),
        (&mut cols as *mut u32).cast(),
        (&mut block_columns as *mut u32).cast(),
        (&mut output_elements as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.split_interleaved_columns, output_elements.div_ceil(256), 256, &mut arguments, "HIP resident split interleaved")?;
    Ok((left, right))
}

pub fn try_split_gated_activation_owned_bf16(device_id: i32, input: DeviceBuffer, bias: Option<&DeviceBuffer>, rows: usize, columns: usize, activation: &Activation) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(columns).ok_or("resident packed gated activation 元素数溢出")?;
    let input_bytes = elements.checked_mul(2).and_then(|n| n.checked_mul(4)).ok_or("resident packed gated activation 输入大小溢出")?;
    let bias_bytes = columns.checked_mul(2).and_then(|n| n.checked_mul(4)).ok_or("resident packed gated activation bias 大小溢出")?;
    let output_bytes = elements.checked_mul(2).ok_or("resident packed gated activation 输出大小溢出")?;
    validate_resident(&input, device_id, input_bytes, "packed gated input")?;
    if let Some(bias) = bias {
        validate_resident(bias, device_id, bias_bytes, "packed gated bias")?;
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_bytes)?;
    let functions = tensor_functions(device_id)?;
    let (kind, alpha, beta, limit, linear_beta, has_linear_beta) = activation_parameters(activation);
    let mut d_input = input.pointer;
    let mut d_bias = bias.map_or(ptr::null_mut(), |bias| bias.pointer);
    let mut d_output = output.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "resident packed gated rows 超过 u32".to_owned())?;
    let mut columns = u32::try_from(columns).map_err(|_| "resident packed gated columns 超过 u32".to_owned())?;
    let mut has_bias = u32::from(bias.is_some());
    let mut kind = kind;
    let mut alpha = alpha;
    let mut beta = beta;
    let mut limit = limit;
    let mut linear_beta = linear_beta;
    let mut has_linear_beta = has_linear_beta;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut columns as *mut u32).cast(),
        (&mut has_bias as *mut u32).cast(),
        (&mut kind as *mut u32).cast(),
        (&mut alpha as *mut f32).cast(),
        (&mut beta as *mut f32).cast(),
        (&mut limit as *mut f32).cast(),
        (&mut linear_beta as *mut f32).cast(),
        (&mut has_linear_beta as *mut u32).cast(),
    ];
    let elements = rows.checked_mul(columns).ok_or("resident packed gated elements 溢出")?;
    launch_tensor_kernel(functions.split_gated_activation, elements.div_ceil(256), 256, &mut arguments, "HIP resident packed gated")?;
    drop(input);
    Ok(output)
}

pub fn try_split_columns_resident_f32(device_id: i32, input: &DeviceBuffer, rows: usize, cols: usize, left_columns: usize) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    try_split_columns_resident_f32_impl(device_id, input, None, rows, cols, left_columns)
}

pub fn try_split_columns_bias_resident_f32(device_id: i32, input: &DeviceBuffer, bias: &DeviceBuffer, rows: usize, cols: usize, left_columns: usize) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    try_split_columns_resident_f32_impl(device_id, input, Some(bias), rows, cols, left_columns)
}

fn try_split_columns_resident_f32_impl(device_id: i32, input: &DeviceBuffer, bias: Option<&DeviceBuffer>, rows: usize, cols: usize, left_columns: usize) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    let input_bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(4)).ok_or("resident split 大小溢出")?;
    validate_resident(input, device_id, input_bytes, "split input")?;
    if let Some(bias) = bias {
        validate_resident(bias, device_id, cols.checked_mul(4).ok_or("resident split bias 大小溢出")?, "split bias")?;
    }
    if left_columns == 0 || left_columns >= cols {
        return Err(format!("resident split left={left_columns} cols={cols} 非法"));
    }
    set_device(device_id)?;
    let right_columns = cols - left_columns;
    let left = DeviceBuffer::allocate(device_id, rows * left_columns * 4)?;
    let right = DeviceBuffer::allocate(device_id, rows * right_columns * 4)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_bias = bias.map_or(ptr::null_mut(), |bias| bias.pointer);
    let mut d_left = left.pointer;
    let mut d_middle = ptr::null_mut();
    let mut d_right = right.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "resident split rows 超过 u32".to_owned())?;
    let mut cols = u32::try_from(cols).map_err(|_| "resident split cols 超过 u32".to_owned())?;
    let mut left_columns = u32::try_from(left_columns).map_err(|_| "resident split left 超过 u32".to_owned())?;
    let mut middle_columns = 0_u32;
    let mut has_bias = u32::from(bias.is_some());
    let mut has_middle = 0_u32;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_left as *mut *mut c_void).cast(),
        (&mut d_middle as *mut *mut c_void).cast(),
        (&mut d_right as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut cols as *mut u32).cast(),
        (&mut left_columns as *mut u32).cast(),
        (&mut middle_columns as *mut u32).cast(),
        (&mut has_bias as *mut u32).cast(),
        (&mut has_middle as *mut u32).cast(),
    ];
    let elements = rows.checked_mul(cols).ok_or("resident split elements 溢出")?;
    launch_tensor_kernel(functions.split_columns, elements.div_ceil(256), 256, &mut arguments, "HIP resident split")?;
    Ok((left, right))
}

pub fn try_split_three_columns_bias_resident_f32(device_id: i32, input: &DeviceBuffer, bias: &DeviceBuffer, rows: usize, cols: usize, columns: usize) -> Result<(DeviceBuffer, DeviceBuffer, DeviceBuffer), String> {
    if columns == 0 || cols != columns.checked_mul(3).ok_or("resident split3 列数溢出")? {
        return Err(format!("resident split3 cols={cols}，期望 3x{columns}"));
    }
    let input_bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(4)).ok_or("resident split3 input 大小溢出")?;
    let bias_bytes = cols.checked_mul(4).ok_or("resident split3 bias 大小溢出")?;
    let output_bytes = rows.checked_mul(columns).and_then(|n| n.checked_mul(4)).ok_or("resident split3 output 大小溢出")?;
    validate_resident(input, device_id, input_bytes, "split3 input")?;
    validate_resident(bias, device_id, bias_bytes, "split3 bias")?;
    set_device(device_id)?;
    let first = DeviceBuffer::allocate(device_id, output_bytes)?;
    let second = DeviceBuffer::allocate(device_id, output_bytes)?;
    let third = DeviceBuffer::allocate(device_id, output_bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_bias = bias.pointer;
    let mut d_first = first.pointer;
    let mut d_second = second.pointer;
    let mut d_third = third.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "resident split3 rows 超过 u32")?;
    let mut cols_u32 = u32::try_from(cols).map_err(|_| "resident split3 cols 超过 u32")?;
    let mut columns_u32 = u32::try_from(columns).map_err(|_| "resident split3 columns 超过 u32")?;
    let mut has_bias = 1_u32;
    let mut has_middle = 1_u32;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_first as *mut *mut c_void).cast(),
        (&mut d_second as *mut *mut c_void).cast(),
        (&mut d_third as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut cols_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut columns_u32 as *mut u32).cast(),
        (&mut has_bias as *mut u32).cast(),
        (&mut has_middle as *mut u32).cast(),
    ];
    let elements = rows_u32.checked_mul(cols_u32).ok_or("resident split3 elements 溢出")?;
    launch_tensor_kernel(functions.split_columns, elements.div_ceil(256), 256, &mut arguments, "HIP resident split3+bias")?;
    Ok((first, second, third))
}

pub fn try_scale_tensor_resident_f32(device_id: i32, input: &DeviceBuffer, elements: usize, scale: f32) -> Result<DeviceBuffer, String> {
    let bytes = elements.checked_mul(4).ok_or("HIP tensor scale 字节数溢出")?;
    validate_resident(input, device_id, bytes, "tensor scale input")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP tensor scale 元素数超过 u32".to_owned())?;
    let mut scale = scale;
    let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast(), (&mut scale as *mut f32).cast()];
    launch_tensor_kernel(functions.scale_tensor, elements.div_ceil(256), 256, &mut arguments, "HIP tensor scale")?;
    Ok(output)
}

pub fn try_tanh_tensor_resident_f32(device_id: i32, input: &DeviceBuffer, elements: usize) -> Result<DeviceBuffer, String> {
    let bytes = elements.checked_mul(4).ok_or("HIP tensor tanh 字节数溢出")?;
    validate_resident(input, device_id, bytes, "tensor tanh input")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP tensor tanh 元素数超过 u32".to_owned())?;
    let mut arguments = [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast()];
    launch_tensor_kernel(functions.tanh_tensor, elements.div_ceil(256), 256, &mut arguments, "HIP tensor tanh")?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_u32_matches_host_scalar() {
        const DEVICE_ID: i32 = 0;
        if set_device(DEVICE_ID).is_err() {
            return;
        }
        let output = try_fill_resident_u32(DEVICE_ID, 0x1234_abcd, 19).unwrap();
        synchronize_device(DEVICE_ID, "HIP u32 fill oracle").unwrap();
        let mut actual = [0_u32; 19];
        let bytes = unsafe { std::slice::from_raw_parts_mut(actual.as_mut_ptr().cast(), std::mem::size_of_val(&actual)) };
        output.copy_to_host(bytes).unwrap();
        assert_eq!(actual, [0x1234_abcd; 19]);
    }
}
