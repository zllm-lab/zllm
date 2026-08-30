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
    launch_tensor_kernel(functions.conv3d, output_elements.div_ceil(256), 256, &mut arguments, "HIP Conv3D")?;
    if let Some(started) = profile_started {
        synchronize_device(device_id, "HIP Conv3D profile")?;
        eprintln!("[rocm-vae] conv3d in={input_channels} out={output_channels} input={input_shape:?} kernel={kernel:?} stride={stride:?} output={output_shape:?} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
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
        (functions.conv3d, output_elements.div_ceil(256), "HIP encoder Conv3D")
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
