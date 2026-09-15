pub(super) const SOURCE: &str = include_str!("conv3d_vision/source.hip");

use super::*;

pub fn try_causal_attention_resident_f32(device_id: i32, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, batch: usize, time: usize, heads: usize, head_dim: usize, scale: f32) -> Result<DeviceBuffer, String> {
    if batch == 0 || time == 0 || heads == 0 || head_dim == 0 || head_dim > 256 {
        return Err(format!("HIP causal attention time={time} heads={heads} head_dim={head_dim} 非法"));
    }
    let elements = batch.checked_mul(time).and_then(|value| value.checked_mul(heads)).and_then(|value| value.checked_mul(head_dim)).ok_or("HIP causal attention 元素数溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP causal attention 字节数溢出")?;
    validate_resident(q, device_id, bytes, "causal attention q")?;
    validate_resident(k, device_id, bytes, "causal attention k")?;
    validate_resident(v, device_id, bytes, "causal attention v")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_q = q.pointer;
    let mut d_k = k.pointer;
    let mut d_v = v.pointer;
    let mut d_output = output.pointer;
    let mut batch = u32::try_from(batch).map_err(|_| "HIP causal attention batch 超过 u32".to_owned())?;
    let mut time = u32::try_from(time).map_err(|_| "HIP causal attention time 超过 u32".to_owned())?;
    let mut heads = u32::try_from(heads).map_err(|_| "HIP causal attention heads 超过 u32".to_owned())?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "HIP causal attention head_dim 超过 u32".to_owned())?;
    let mut scale = scale;
    let mut arguments = [
        (&mut d_q as *mut *mut c_void).cast(),
        (&mut d_k as *mut *mut c_void).cast(),
        (&mut d_v as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut batch as *mut u32).cast(),
        (&mut time as *mut u32).cast(),
        (&mut heads as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
        (&mut scale as *mut f32).cast(),
    ];
    launch_tensor_kernel(functions.causal_attention, batch.checked_mul(time).and_then(|value| value.checked_mul(heads)).ok_or("HIP causal attention grid 溢出")?, 256, &mut arguments, "HIP causal attention")?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_conv3d_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    input_channels: usize,
    output_channels: usize,
    input_shape: [usize; 3],
    kernel: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    causal: bool,
    output_shape: [usize; 3],
) -> Result<DeviceBuffer, String> {
    let input_elements = input_channels * input_shape.into_iter().product::<usize>();
    let weight_elements = output_channels * input_channels * kernel.into_iter().product::<usize>();
    let output_elements = output_channels * output_shape.into_iter().product::<usize>();
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("HIP Conv3D input 字节溢出")?, "Conv3D input")?;
    validate_resident(weight, device_id, weight_elements.checked_mul(4).ok_or("HIP Conv3D weight 字节溢出")?, "Conv3D weight")?;
    if let Some(bias) = bias {
        validate_resident(bias, device_id, output_channels.checked_mul(4).ok_or("HIP Conv3D bias 字节溢出")?, "Conv3D bias")?;
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements.checked_mul(4).ok_or("HIP Conv3D output 字节溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_weight = weight.pointer;
    let mut d_bias = bias.map_or(input.pointer, |bias| bias.pointer);
    let mut d_output = output.pointer;
    let mut dims = [
        input_channels,
        output_channels,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        kernel[0],
        kernel[1],
        kernel[2],
        stride[0],
        stride[1],
        stride[2],
        usize::from(bias.is_some()),
        padding[0],
        padding[1],
        padding[2],
        0,
        0,
        usize::from(causal),
        0,
        output_shape[0],
        output_shape[1],
        output_shape[2],
    ]
    .map(|value| u32::try_from(value).map_err(|_| "HIP Conv3D 维度超过 u32".to_owned()))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    let mut arguments: Vec<*mut c_void> = vec![(&mut d_input as *mut *mut c_void).cast(), (&mut d_weight as *mut *mut c_void).cast(), (&mut d_bias as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast()];
    arguments.extend(dims.iter_mut().map(|value| (value as *mut u32).cast()));
    let output_elements = u32::try_from(output_elements).map_err(|_| "HIP Conv3D output 元素数超过 u32".to_owned())?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let (function, grid) = conv3d_function_grid(&functions, input_channels, output_channels, output_shape, output_elements)?;
    launch_tensor_kernel(function, grid, 256, &mut arguments, "HIP Conv3D")?;
    if let Some(started) = profile_started {
        synchronize_device(device_id, "HIP Conv3D profile")?;
        eprintln!("[rocm-vae] conv3d in={input_channels} out={output_channels} input={input_shape:?} kernel={kernel:?} stride={stride:?} output={output_shape:?} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
}

// 先完整完成展开/GEMM再决定是否回退；异常输入重算F32，不延长history生命周期。
#[allow(clippy::too_many_arguments)]
fn try_conv3d_columns(
    device_id: i32,
    input: &DeviceBuffer,
    history: Option<&DeviceBuffer>,
    weight: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    output: &DeviceBuffer,
    spec: &crate::vae::Conv3dSpec,
    head: usize,
    spatial: usize,
) -> Result<bool, String> {
    let functions = tensor_functions(device_id)?;
    let pointwise = spec.kernel == [1; 3];
    let pack = if pointwise { functions.conv3d_pointwise_columns } else { functions.conv3d_history_columns };
    let (Some(pack), Some(gemm)) = (pack, functions.conv3d_columns_gemm) else { return Ok(false) };
    let tile = if pointwise {
        16384usize
    } else {
        match spec.output_channels {
            128 => 8192usize,
            256 => 16384,
            _ => 2048,
        }
    };
    let k = spec.input_channels.checked_mul(if pointwise { 1 } else { 27 }).ok_or("Conv3D columns K 溢出")?;
    let scratch_bytes = k.checked_mul(tile).and_then(|n| n.checked_mul(2)).ok_or("Conv3D columns 工作区溢出")?;
    let columns = DeviceBuffer::allocate_reusable(device_id, scratch_bytes)?;
    let flag = DeviceBuffer::upload(device_id, &[0; 4])?;
    let mut d_input = input.pointer;
    let mut d_history = history.map_or(ptr::null_mut(), |h| h.pointer);
    let mut d_columns = columns.pointer;
    let mut d_flag = flag.pointer;
    let mut d_weight = weight.pointer;
    let mut d_bias = bias.map_or(ptr::null_mut(), |b| b.pointer);
    let mut d_output = output.pointer;
    let mut depth = u32::try_from(spec.input_shape[0]).map_err(|_| "Conv3D columns depth 超过 u32")?;
    let mut height = u32::try_from(spec.input_shape[1]).map_err(|_| "Conv3D columns height 超过 u32")?;
    let mut width = u32::try_from(spec.input_shape[2]).map_err(|_| "Conv3D columns width 超过 u32")?;
    let mut channels = u32::try_from(spec.input_channels).map_err(|_| "Conv3D columns channels 超过 u32")?;
    let mut head = u32::try_from(head).map_err(|_| "Conv3D columns head 超过 u32")?;
    let mut co = u32::try_from(spec.output_channels).map_err(|_| "Conv3D columns output channels 超过 u32")?;
    let mut reduction = u32::try_from(k).map_err(|_| "Conv3D columns K 超过 u32")?;
    let mut total = spatial as u64;
    let mut has_bias = u32::from(bias.is_some());
    for start in (0..spatial).step_by(tile) {
        let mut start = start as u64;
        let mut n = u32::try_from(tile.min(spatial - start as usize)).map_err(|_| "Conv3D columns tile 超过 u32")?;
        let mut pack_args = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_history as *mut *mut c_void).cast(),
            (&mut d_columns as *mut *mut c_void).cast(),
            (&mut d_flag as *mut *mut c_void).cast(),
            (&mut depth as *mut u32).cast(),
            (&mut height as *mut u32).cast(),
            (&mut width as *mut u32).cast(),
            (&mut channels as *mut u32).cast(),
            (&mut head as *mut u32).cast(),
            (&mut start as *mut u64).cast(),
            (&mut n as *mut u32).cast(),
        ];
        let pack_grid = u32::try_from((k * n as usize).div_ceil(256)).map_err(|_| "Conv3D columns grid 超过 u32")?;
        if pointwise {
            let mut pointwise_args = [
                (&mut d_input as *mut *mut c_void).cast(),
                (&mut d_history as *mut *mut c_void).cast(),
                (&mut d_columns as *mut *mut c_void).cast(),
                (&mut d_flag as *mut *mut c_void).cast(),
                (&mut channels as *mut u32).cast(),
                (&mut total as *mut u64).cast(),
                (&mut start as *mut u64).cast(),
                (&mut n as *mut u32).cast(),
            ];
            launch_tensor_kernel(pack, pack_grid, 256, &mut pointwise_args, "HIP Conv3D checked pointwise columns")?;
        } else {
            launch_tensor_kernel(pack, pack_grid, 256, &mut pack_args, "HIP Conv3D checked columns")?;
        }
        let mut gemm_args = [
            (&mut d_columns as *mut *mut c_void).cast(),
            (&mut d_weight as *mut *mut c_void).cast(),
            (&mut d_bias as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut co as *mut u32).cast(),
            (&mut reduction as *mut u32).cast(),
            (&mut n as *mut u32).cast(),
            (&mut start as *mut u64).cast(),
            (&mut total as *mut u64).cast(),
            (&mut has_bias as *mut u32).cast(),
        ];
        let gemm_grid = n.div_ceil(64).checked_mul(co.div_ceil(64)).ok_or("Conv3D columns GEMM grid 溢出")?;
        launch_tensor_kernel(gemm, gemm_grid, 256, &mut gemm_args, "HIP Conv3D columns GEMM")?;
    }
    let mut invalid = [0u8; 4];
    flag.copy_to_host(&mut invalid)?;
    Ok(invalid == [0; 4])
}

/// 双输入时间窗口与紧凑尾部；F16片段路径需由调用方确认权重无损可表示F16。
#[allow(clippy::too_many_arguments)]
pub fn try_conv3d_with_history_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    mut history: Option<std::sync::Arc<DeviceBuffer>>,
    weight: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    spec: &crate::vae::Conv3dSpec,
    history_frames: Option<usize>,
    spatial_pad_after: [usize; 2],
    allow_f16_wmma: bool,
) -> Result<(DeviceBuffer, Option<DeviceBuffer>), String> {
    let (head, keep, output_shape) = crate::vae::conv3d_history_layout(spec, history_frames, spatial_pad_after)?;
    if history.is_some() != history_frames.is_some() {
        return Err("HIP Conv3D history buffer 与帧数不匹配".to_owned());
    }
    let plane = spec.input_shape[1].checked_mul(spec.input_shape[2]).ok_or("HIP Conv3D history plane 溢出")?;
    let input_elements = spec.input_channels.checked_mul(spec.input_spatial()?).ok_or("HIP Conv3D history input 溢出")?;
    let k_count = spec.kernel.into_iter().try_fold(spec.input_channels, |n, d| n.checked_mul(d).ok_or("HIP Conv3D history K 溢出"))?;
    let weight_elements = spec.output_channels.checked_mul(k_count).ok_or("HIP Conv3D history weight 溢出")?;
    let spatial = output_shape.into_iter().try_fold(1usize, |n, d| n.checked_mul(d).ok_or("HIP Conv3D history output spatial 溢出"))?;
    let elements = spec.output_channels.checked_mul(spatial).ok_or("HIP Conv3D history output elements 溢出")?;
    let tail_elements = spec.input_channels.checked_mul(keep).and_then(|n| n.checked_mul(plane)).ok_or("HIP Conv3D history tail 溢出")?;
    let bytes = |n: usize| n.checked_mul(4).ok_or("HIP Conv3D history 字节溢出");
    validate_resident(input, device_id, bytes(input_elements)?, "Conv3D history input")?;
    validate_resident(weight, device_id, bytes(weight_elements)?, "Conv3D history weight")?;
    if let Some(history) = history.as_deref() {
        validate_resident(history, device_id, bytes(tail_elements)?, "Conv3D history tail")?;
    }
    if let Some(bias) = bias {
        validate_resident(bias, device_id, bytes(spec.output_channels)?, "Conv3D history bias")?;
    }
    // 复用既有标量/tiled的32位维度与K迭代，所有allocation/地址字节另以usize/64位校验。
    u32::try_from(k_count).map_err(|_| "HIP Conv3D history K 超过 u32")?;
    let output_elements = u32::try_from(elements).map_err(|_| "HIP Conv3D history output 元素数超过 u32")?;
    let depth = spec.input_shape[0].checked_add(head).ok_or("HIP Conv3D history depth 溢出")?;
    let mut dims = [
        spec.input_channels,
        spec.output_channels,
        depth,
        spec.input_shape[1],
        spec.input_shape[2],
        spec.kernel[0],
        spec.kernel[1],
        spec.kernel[2],
        spec.stride[0],
        spec.stride[1],
        spec.stride[2],
        usize::from(bias.is_some()),
        0,
        spec.padding[1],
        spec.padding[2],
        spatial_pad_after[0],
        spatial_pad_after[1],
        0,
        0,
        output_shape[0],
        output_shape[1],
        output_shape[2],
    ]
    .map(|v| u32::try_from(v).map_err(|_| "HIP Conv3D history 维度超过 u32"))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let (_, mut grid) = conv3d_function_grid(&functions, spec.input_channels, spec.output_channels, output_shape, output_elements)?;
    let mut function = if spec.input_channels >= 16 && spec.output_channels >= 16 && spatial >= 16 { functions.conv3d_history_tiled } else { functions.conv3d_history };
    if allow_f16_wmma
        && spec.input_channels != 512
        && let Some(wmma) = functions.conv3d_history_wmma
    {
        grid = u32::try_from(spatial.div_ceil(16).checked_mul(spec.output_channels.div_ceil(128)).ok_or("HIP Conv3D history WMMA grid 溢出")?).map_err(|_| "HIP Conv3D history WMMA grid 超过 u32")?;
        function = wmma;
    }
    let output = DeviceBuffer::allocate(device_id, bytes(elements)?)?;
    // try_unwrap原子地取得唯一所有权；view即使外层Arc唯一也仍可能与owner别名。
    let reusable_history = if history.as_ref().is_some_and(|h| h.owner.is_none()) {
        match std::sync::Arc::try_unwrap(history.take().expect("已检查history")) {
            Ok(buffer) => Some(buffer),
            Err(shared) => {
                history = Some(shared);
                None
            }
        }
    } else {
        None
    };
    let mut d_input = input.pointer;
    let mut d_history = reusable_history.as_ref().or(history.as_deref()).map_or(ptr::null_mut(), |h| h.pointer);
    let mut d_weight = weight.pointer;
    let mut d_bias = bias.map_or(ptr::null_mut(), |b| b.pointer);
    let mut d_output = output.pointer;
    let mut head_u32 = u32::try_from(head).map_err(|_| "HIP Conv3D history head 超过 u32")?;
    let mut args: Vec<*mut c_void> = vec![(&mut d_input as *mut *mut c_void).cast(), (&mut d_weight as *mut *mut c_void).cast(), (&mut d_bias as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast()];
    args.extend(dims.iter_mut().map(|v| (v as *mut u32).cast()));
    args.extend([(&mut d_history as *mut *mut c_void).cast(), (&mut head_u32 as *mut u32).cast()]);
    let profile_started = if options().kernel_profile {
        synchronize_device(device_id, "HIP Conv3D history profile begin")?;
        Some(std::time::Instant::now())
    } else {
        None
    };
    let columns_done = allow_f16_wmma
        && matches!(spec.input_channels, 128 | 256 | 512)
        && ((spec.kernel == [3; 3] && spec.padding == [0, 1, 1]) || (spec.kernel == [1; 3] && spec.padding == [0; 3] && head == 0))
        && spec.stride == [1; 3]
        && !spec.causal
        && spatial_pad_after == [0; 2]
        && try_conv3d_columns(device_id, input, reusable_history.as_ref().or(history.as_deref()), weight, bias, &output, spec, head, spatial)?;
    if !columns_done {
        launch_tensor_kernel(function, grid, 256, &mut args, if Some(function) == functions.conv3d_history_wmma { "HIP Conv3D history F16 WMMA" } else { "HIP Conv3D history F32" })?;
    }
    if let Some(started) = profile_started {
        synchronize_device(device_id, "HIP Conv3D history profile end")?;
        eprintln!(
            "[rocm-vae] history-conv3d device={device_id} columns={columns_done} in={} out={} input={:?} kernel={:?} wall={:.6}s",
            spec.input_channels,
            spec.output_channels,
            spec.input_shape,
            spec.kernel,
            started.elapsed().as_secs_f64()
        );
    }
    let tail = if keep == 0 {
        None
    } else {
        let tail = if let Some(history) = reusable_history { history } else { DeviceBuffer::allocate(device_id, bytes(tail_elements)?)? };
        let mut d_tail = tail.pointer;
        let mut current_depth = u32::try_from(spec.input_shape[0]).map_err(|_| "HIP Conv3D history current depth 超过 u32")?;
        let mut keep_u32 = u32::try_from(keep).map_err(|_| "HIP Conv3D history keep 超过 u32")?;
        let mut plane_u64 = plane as u64;
        let mut elements_u64 = tail_elements as u64;
        let mut args = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_history as *mut *mut c_void).cast(),
            (&mut d_tail as *mut *mut c_void).cast(),
            (&mut current_depth as *mut u32).cast(),
            (&mut head_u32 as *mut u32).cast(),
            (&mut keep_u32 as *mut u32).cast(),
            (&mut plane_u64 as *mut u64).cast(),
            (&mut elements_u64 as *mut u64).cast(),
        ];
        launch_tensor_kernel(functions.conv3d_history_tail, u32::try_from((tail_elements / keep).div_ceil(256)).map_err(|_| "HIP Conv3D history tail grid 超过 u32")?, 256, &mut args, "HIP Conv3D compact history tail")?;
        Some(tail)
    };
    Ok((output, tail))
}

#[allow(clippy::too_many_arguments)]
pub fn try_encoder_conv3d_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    input_channels: usize,
    output_channels: usize,
    input_shape: [usize; 3],
    kernel: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    padding_after: [usize; 2],
    causal: bool,
    reflect_spatial: bool,
    output_shape: [usize; 3],
) -> Result<DeviceBuffer, String> {
    let input_elements = input_channels * input_shape.into_iter().product::<usize>();
    let weight_elements = output_channels * input_channels * kernel.into_iter().product::<usize>();
    let output_elements = output_channels * output_shape.into_iter().product::<usize>();
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("HIP encoder Conv3D input 字节溢出")?, "encoder Conv3D input")?;
    validate_resident(weight, device_id, weight_elements.checked_mul(4).ok_or("HIP encoder Conv3D weight 字节溢出")?, "encoder Conv3D weight")?;
    if let Some(bias) = bias {
        validate_resident(bias, device_id, output_channels.checked_mul(4).ok_or("HIP encoder Conv3D bias 字节溢出")?, "encoder Conv3D bias")?;
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements.checked_mul(4).ok_or("HIP encoder Conv3D output 字节溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let use_single_frame_wmma = input_shape[0] == 1 && output_shape[0] == 1 && padding[0] < kernel[0];
    let packed_weight = if use_single_frame_wmma {
        let effective_k = input_channels.checked_mul(kernel[1]).and_then(|value| value.checked_mul(kernel[2])).ok_or("HIP encoder packed weight K 溢出")?;
        let elements = effective_k.checked_mul(output_channels).ok_or("HIP encoder packed weight elements 溢出")?;
        let packed = DeviceBuffer::allocate(device_id, elements.checked_mul(2).ok_or("HIP encoder packed weight bytes 溢出")?)?;
        let mut d_source = weight.pointer;
        let mut d_packed = packed.pointer;
        let mut pack_dims = [input_channels, output_channels, kernel[0], kernel[1], kernel[2], padding[0], elements]
            .map(|value| u32::try_from(value).map_err(|_| "HIP encoder packed weight 维度超过 u32".to_owned()))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let mut pack_arguments: Vec<*mut c_void> = vec![(&mut d_source as *mut *mut c_void).cast(), (&mut d_packed as *mut *mut c_void).cast()];
        pack_arguments.extend(pack_dims.iter_mut().map(|value| (value as *mut u32).cast()));
        let pack_elements = u32::try_from(elements).map_err(|_| "HIP encoder packed weight elements 超过 u32")?;
        launch_tensor_kernel(functions.pack_conv3d_single_frame_weight, pack_elements.div_ceil(256), 256, &mut pack_arguments, "HIP encoder pack single-frame Conv3D weight")?;
        Some(packed)
    } else {
        None
    };
    let mut d_input = input.pointer;
    let mut d_weight = packed_weight.as_ref().map_or(weight.pointer, |packed| packed.pointer);
    let mut d_bias = bias.map_or(input.pointer, |bias| bias.pointer);
    let mut d_output = output.pointer;
    let mut dims = [
        input_channels,
        output_channels,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        kernel[0],
        kernel[1],
        kernel[2],
        stride[0],
        stride[1],
        stride[2],
        usize::from(bias.is_some()),
        padding[0],
        padding[1],
        padding[2],
        padding_after[0],
        padding_after[1],
        usize::from(causal),
        usize::from(reflect_spatial),
        output_shape[0],
        output_shape[1],
        output_shape[2],
    ]
    .map(|value| u32::try_from(value).map_err(|_| "HIP encoder Conv3D 维度超过 u32".to_owned()))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    let mut arguments: Vec<*mut c_void> = vec![(&mut d_input as *mut *mut c_void).cast(), (&mut d_weight as *mut *mut c_void).cast(), (&mut d_bias as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast()];
    arguments.extend(dims.iter_mut().map(|value| (value as *mut u32).cast()));
    let output_elements = u32::try_from(output_elements).map_err(|_| "HIP encoder Conv3D output 元素数超过 u32".to_owned())?;
    let (function, grid, label) = if use_single_frame_wmma {
        let spatial = output_shape[1].checked_mul(output_shape[2]).ok_or("HIP encoder Conv3D spatial 溢出")?;
        let position_tiles = u32::try_from(spatial.div_ceil(16)).map_err(|_| "HIP encoder Conv3D position tiles 超过 u32")?;
        let output_groups = u32::try_from(output_channels.div_ceil(128)).map_err(|_| "HIP encoder Conv3D output groups 超过 u32")?;
        (functions.conv3d_single_frame_wmma, position_tiles.checked_mul(output_groups).ok_or("HIP encoder Conv3D WMMA grid 溢出")?, "HIP encoder single-frame Conv3D WMMA")
    } else {
        let (function, grid) = conv3d_function_grid(&functions, input_channels, output_channels, output_shape, output_elements)?;
        (function, grid, "HIP encoder Conv3D")
    };
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    launch_tensor_kernel(function, grid, 256, &mut arguments, label)?;
    if let Some(started) = profile_started {
        synchronize_device(device_id, "HIP encoder Conv3D profile")?;
        eprintln!(
            "[rocm-vae] encoder-conv3d wmma={use_single_frame_wmma} in={input_channels} out={output_channels} input={input_shape:?} kernel={kernel:?} stride={stride:?} output={output_shape:?} wall={:.6}s",
            started.elapsed().as_secs_f64()
        );
    }
    Ok(output)
}

fn conv3d_function_grid(functions: &TensorFunctions, input_channels: usize, output_channels: usize, output_shape: [usize; 3], elements: u32) -> Result<(usize, u32), String> {
    let spatial = output_shape.into_iter().try_fold(1usize, |n, d| n.checked_mul(d).ok_or("HIP Conv3D spatial溢出"))?;
    if input_channels >= 16 && output_channels >= 16 && spatial >= 16 {
        let tiles = spatial.div_ceil(16).checked_mul(output_channels.div_ceil(64)).ok_or("HIP Conv3D tiled grid溢出")?;
        Ok((functions.conv3d_tiled, u32::try_from(tiles).map_err(|_| "HIP Conv3D tiled grid超过u32")?))
    } else {
        Ok((functions.conv3d, elements.div_ceil(256)))
    }
}

pub fn try_vision_rope_resident_f32(
    device_id: i32,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    cosine: &DeviceBuffer,
    sine: &DeviceBuffer,
    rows: usize,
    columns: usize,
    heads: usize,
    rotary_dim: usize,
) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    if rows == 0 || heads == 0 || !columns.is_multiple_of(heads) || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > columns / heads {
        return Err(format!("HIP vision RoPE rows={rows} columns={columns} heads={heads} rotary_dim={rotary_dim} 非法"));
    }
    let elements = rows.checked_mul(columns).ok_or("HIP vision RoPE elements 溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP vision RoPE bytes 溢出")?;
    let rope_bytes = rows.checked_mul(rotary_dim).and_then(|value| value.checked_mul(4)).ok_or("HIP vision RoPE table bytes 溢出")?;
    validate_resident(query, device_id, bytes, "vision RoPE query")?;
    validate_resident(key, device_id, bytes, "vision RoPE key")?;
    validate_resident(cosine, device_id, rope_bytes, "vision RoPE cosine")?;
    validate_resident(sine, device_id, rope_bytes, "vision RoPE sine")?;
    set_device(device_id)?;
    let query_output = DeviceBuffer::allocate(device_id, bytes)?;
    let key_output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_query = query.pointer;
    let mut d_key = key.pointer;
    let mut d_cosine = cosine.pointer;
    let mut d_sine = sine.pointer;
    let mut d_query_output = query_output.pointer;
    let mut d_key_output = key_output.pointer;
    let head_dim = columns / heads;
    let mut heads = u32::try_from(heads).map_err(|_| "HIP vision RoPE heads 超过 u32".to_owned())?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "HIP vision RoPE head_dim 超过 u32".to_owned())?;
    let mut rotary_dim = u32::try_from(rotary_dim).map_err(|_| "HIP vision RoPE rotary_dim 超过 u32".to_owned())?;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP vision RoPE elements 超过 u32".to_owned())?;
    let mut arguments = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_cosine as *mut *mut c_void).cast(),
        (&mut d_sine as *mut *mut c_void).cast(),
        (&mut d_query_output as *mut *mut c_void).cast(),
        (&mut d_key_output as *mut *mut c_void).cast(),
        (&mut heads as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
        (&mut rotary_dim as *mut u32).cast(),
        (&mut elements as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.vision_rope, elements.div_ceil(256), 256, &mut arguments, "HIP vision RoPE")?;
    Ok((query_output, key_output))
}

pub fn try_vision_attention_resident_f32(
    device_id: i32,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    cosine: &DeviceBuffer,
    sine: &DeviceBuffer,
    rows: usize,
    columns: usize,
    heads: usize,
    rotary_dim: usize,
) -> Result<DeviceBuffer, String> {
    if rows == 0 || heads == 0 || !columns.is_multiple_of(heads) || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > columns / heads || columns / heads > 256 {
        return Err(format!("HIP fused vision attention rows={rows} columns={columns} heads={heads} rotary_dim={rotary_dim} 非法"));
    }
    let elements = rows.checked_mul(columns).ok_or("HIP fused vision attention elements 溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP fused vision attention bytes 溢出")?;
    let rope_bytes = rows.checked_mul(rotary_dim).and_then(|value| value.checked_mul(4)).ok_or("HIP fused vision attention RoPE bytes 溢出")?;
    validate_resident(query, device_id, bytes, "fused vision attention query")?;
    validate_resident(key, device_id, bytes, "fused vision attention key")?;
    validate_resident(value, device_id, bytes, "fused vision attention value")?;
    validate_resident(cosine, device_id, rope_bytes, "fused vision attention cosine")?;
    validate_resident(sine, device_id, rope_bytes, "fused vision attention sine")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_query = query.pointer;
    let mut d_key = key.pointer;
    let mut d_value = value.pointer;
    let mut d_cosine = cosine.pointer;
    let mut d_sine = sine.pointer;
    let mut d_output = output.pointer;
    let head_dim = columns / heads;
    let mut rows = u32::try_from(rows).map_err(|_| "HIP fused vision attention rows 超过 u32".to_owned())?;
    let mut heads = u32::try_from(heads).map_err(|_| "HIP fused vision attention heads 超过 u32".to_owned())?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "HIP fused vision attention head_dim 超过 u32".to_owned())?;
    let mut rotary_dim = u32::try_from(rotary_dim).map_err(|_| "HIP fused vision attention rotary_dim 超过 u32".to_owned())?;
    let mut arguments = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_value as *mut *mut c_void).cast(),
        (&mut d_cosine as *mut *mut c_void).cast(),
        (&mut d_sine as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut heads as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
        (&mut rotary_dim as *mut u32).cast(),
    ];
    let query_tiles = rows.div_ceil(4);
    launch_tensor_kernel(functions.vision_attention, query_tiles.checked_mul(heads).ok_or("HIP fused vision attention grid 溢出")?, 256, &mut arguments, "HIP fused vision attention")?;
    Ok(output)
}

pub fn try_scatter_rows_resident_f32(device_id: i32, destination: &DeviceBuffer, destination_rows: usize, source: &DeviceBuffer, source_rows: usize, columns: usize, start_row: usize) -> Result<(), String> {
    if start_row.checked_add(source_rows).is_none_or(|end| end > destination_rows) {
        return Err(format!("HIP scatter rows destination={destination_rows} start={start_row} source={source_rows} 非法"));
    }
    let source_elements = source_rows.checked_mul(columns).ok_or("HIP scatter rows source elements 溢出")?;
    validate_resident(source, device_id, source_elements.checked_mul(4).ok_or("HIP scatter rows source bytes 溢出")?, "scatter rows source")?;
    validate_resident(destination, device_id, destination_rows.checked_mul(columns).and_then(|value| value.checked_mul(4)).ok_or("HIP scatter rows destination bytes 溢出")?, "scatter rows destination")?;
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let mut d_source = source.pointer;
    let mut d_destination = destination.pointer;
    let mut start_row = u32::try_from(start_row).map_err(|_| "HIP scatter rows start 超过 u32".to_owned())?;
    let mut columns = u32::try_from(columns).map_err(|_| "HIP scatter rows columns 超过 u32".to_owned())?;
    let mut count = u32::try_from(source_elements).map_err(|_| "HIP scatter rows count 超过 u32".to_owned())?;
    let mut arguments = [(&mut d_source as *mut *mut c_void).cast(), (&mut d_destination as *mut *mut c_void).cast(), (&mut start_row as *mut u32).cast(), (&mut columns as *mut u32).cast(), (&mut count as *mut u32).cast()];
    launch_tensor_kernel(functions.scatter_rows, count.div_ceil(256), 256, &mut arguments, "HIP scatter rows")
}

pub fn try_pixel_shuffle_resident_f32(device_id: i32, input: &DeviceBuffer, channels: usize, height: usize, width: usize, upscale: usize) -> Result<DeviceBuffer, String> {
    let input_elements = channels * upscale * upscale * height * width;
    let output_elements = channels * height * upscale * width * upscale;
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("HIP pixel shuffle input 字节溢出")?, "pixel shuffle input")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements.checked_mul(4).ok_or("HIP pixel shuffle output 字节溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_output = output.pointer;
    let mut channels = u32::try_from(channels).map_err(|_| "HIP pixel shuffle channels 超过 u32".to_owned())?;
    let mut height = u32::try_from(height).map_err(|_| "HIP pixel shuffle height 超过 u32".to_owned())?;
    let mut width = u32::try_from(width).map_err(|_| "HIP pixel shuffle width 超过 u32".to_owned())?;
    let mut upscale = u32::try_from(upscale).map_err(|_| "HIP pixel shuffle upscale 超过 u32".to_owned())?;
    let mut arguments =
        [(&mut d_input as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut channels as *mut u32).cast(), (&mut height as *mut u32).cast(), (&mut width as *mut u32).cast(), (&mut upscale as *mut u32).cast()];
    let output_elements = u32::try_from(output_elements).map_err(|_| "HIP pixel shuffle output 元素数超过 u32".to_owned())?;
    launch_tensor_kernel(functions.pixel_shuffle, output_elements.div_ceil(256), 256, &mut arguments, "HIP pixel shuffle")?;
    Ok(output)
}
