//! ROCm 音频 VAE 算子。

use super::*;

pub fn try_audio_unpack_affine_resident_f32(device_id: i32, input: &DeviceBuffer, scale: &DeviceBuffer, bias: &DeviceBuffer, batch: usize, time: usize, channels: usize) -> Result<DeviceBuffer, String> {
    let elements = batch.checked_mul(time).and_then(|value| value.checked_mul(channels)).ok_or("HIP audio unpack 大小溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP audio unpack 字节数溢出")?;
    let parameter_bytes = channels.checked_mul(4).ok_or("HIP audio unpack 参数字节数溢出")?;
    validate_resident(input, device_id, bytes, "audio unpack input")?;
    validate_resident(scale, device_id, parameter_bytes, "audio unpack scale")?;
    validate_resident(bias, device_id, parameter_bytes, "audio unpack bias")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_scale = scale.pointer;
    let mut d_bias = bias.pointer;
    let mut d_output = output.pointer;
    let mut batch = u32::try_from(batch).map_err(|_| "HIP audio unpack batch 超过 u32".to_owned())?;
    let mut time = u32::try_from(time).map_err(|_| "HIP audio unpack time 超过 u32".to_owned())?;
    let mut channels = u32::try_from(channels).map_err(|_| "HIP audio unpack channels 超过 u32".to_owned())?;
    let mut elements = u32::try_from(elements).map_err(|_| "HIP audio unpack 元素数超过 u32".to_owned())?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_scale as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut batch as *mut u32).cast(),
        (&mut time as *mut u32).cast(),
        (&mut channels as *mut u32).cast(),
        (&mut elements as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.audio_unpack_affine, elements.div_ceil(256), 256, &mut arguments, "HIP audio unpack affine")?;
    Ok(output)
}

fn try_audio_weight_norm_resident_f32(device_id: i32, weight_g: &DeviceBuffer, weight_v: &DeviceBuffer, rows: usize, columns: usize) -> Result<DeviceBuffer, String> {
    let elements = rows.checked_mul(columns).ok_or("HIP audio weight norm 大小溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP audio weight norm 字节数溢出")?;
    let scale_bytes = rows.checked_mul(4).ok_or("HIP audio weight norm scale 字节数溢出")?;
    validate_resident(weight_g, device_id, scale_bytes, "audio weight_g")?;
    validate_resident(weight_v, device_id, bytes, "audio weight_v")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_weight_g = weight_g.pointer;
    let mut d_weight_v = weight_v.pointer;
    let mut d_output = output.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "HIP audio weight norm rows 超过 u32".to_owned())?;
    let mut columns = u32::try_from(columns).map_err(|_| "HIP audio weight norm columns 超过 u32".to_owned())?;
    let mut arguments = [(&mut d_weight_g as *mut *mut c_void).cast(), (&mut d_weight_v as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut rows as *mut u32).cast(), (&mut columns as *mut u32).cast()];
    launch_tensor_kernel(functions.audio_weight_norm, rows, 256, &mut arguments, "HIP audio weight norm")?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_audio_conv1d_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight_g: Option<&DeviceBuffer>,
    weight_v: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    input_length: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
) -> Result<(DeviceBuffer, usize), String> {
    try_audio_conv1d_strided_resident_f32(device_id, input, weight_g, weight_v, bias, batch, input_channels, output_channels, input_length, kernel, dilation, padding, 1)
}

#[allow(clippy::too_many_arguments)]
pub fn try_audio_conv1d_strided_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight_g: Option<&DeviceBuffer>,
    weight_v: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    input_length: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
    stride: usize,
) -> Result<(DeviceBuffer, usize), String> {
    if kernel == 0 || dilation == 0 || stride == 0 {
        return Err("HIP audio Conv1D kernel/dilation/stride 必须非零".to_owned());
    }
    let effective = dilation.checked_mul(kernel - 1).and_then(|value| value.checked_add(1)).ok_or("HIP audio Conv1D effective kernel 溢出")?;
    let padded = input_length.checked_add(padding.checked_mul(2).ok_or("HIP audio Conv1D padding 溢出")?).ok_or("HIP audio Conv1D output length 溢出")?;
    let output_length = padded.checked_sub(effective).ok_or("HIP audio Conv1D effective kernel 超过输入")? / stride + 1;
    let input_elements = batch.checked_mul(input_channels).and_then(|value| value.checked_mul(input_length)).ok_or("HIP audio Conv1D input 大小溢出")?;
    let weight_columns = input_channels.checked_mul(kernel).ok_or("HIP audio Conv1D weight columns 溢出")?;
    let weight_elements = output_channels.checked_mul(weight_columns).ok_or("HIP audio Conv1D weight 大小溢出")?;
    let output_elements = batch.checked_mul(output_channels).and_then(|value| value.checked_mul(output_length)).ok_or("HIP audio Conv1D output 大小溢出")?;
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("HIP audio Conv1D input 字节溢出")?, "audio Conv1D input")?;
    validate_resident(weight_v, device_id, weight_elements.checked_mul(4).ok_or("HIP audio Conv1D weight 字节溢出")?, "audio Conv1D weight_v")?;
    if let Some(bias) = bias {
        validate_resident(bias, device_id, output_channels.checked_mul(4).ok_or("HIP audio Conv1D bias 字节溢出")?, "audio Conv1D bias")?;
    }
    let normalized = weight_g.map(|weight_g| try_audio_weight_norm_resident_f32(device_id, weight_g, weight_v, output_channels, weight_columns)).transpose()?;
    let weight = normalized.as_ref().unwrap_or(weight_v);
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements.checked_mul(4).ok_or("HIP audio Conv1D output 字节溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_weight = weight.pointer;
    let mut d_bias = bias.map_or(input.pointer, |bias| bias.pointer);
    let mut d_output = output.pointer;
    let [mut batch, mut input_channels, mut output_channels, mut input_length, mut output_length_u32, mut kernel, mut dilation, mut padding, mut stride, mut has_bias, mut elements] =
        [batch, input_channels, output_channels, input_length, output_length, kernel, dilation, padding, stride, usize::from(bias.is_some()), output_elements]
            .into_iter()
            .map(|value| u32::try_from(value).map_err(|_| "HIP audio Conv1D 维度超过 u32".to_owned()))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "HIP audio Conv1D 参数转换失败".to_owned())?;
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_weight as *mut *mut c_void).cast(),
        (&mut d_bias as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut batch as *mut u32).cast(),
        (&mut input_channels as *mut u32).cast(),
        (&mut output_channels as *mut u32).cast(),
        (&mut input_length as *mut u32).cast(),
        (&mut output_length_u32 as *mut u32).cast(),
        (&mut kernel as *mut u32).cast(),
        (&mut dilation as *mut u32).cast(),
        (&mut padding as *mut u32).cast(),
        (&mut stride as *mut u32).cast(),
        (&mut has_bias as *mut u32).cast(),
        (&mut elements as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.audio_conv1d, elements.div_ceil(256), 256, &mut arguments, "HIP audio Conv1D")?;
    Ok((output, output_length))
}

#[allow(clippy::too_many_arguments)]
pub fn try_audio_conv_transpose1d_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    weight_g: &DeviceBuffer,
    weight_v: &DeviceBuffer,
    bias: &DeviceBuffer,
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    input_length: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<(DeviceBuffer, usize), String> {
    if input_length == 0 || kernel == 0 || stride == 0 {
        return Err("HIP audio ConvTranspose1D input/kernel/stride 必须非零".to_owned());
    }
    let expanded = (input_length - 1).checked_mul(stride).and_then(|value| value.checked_add(kernel)).ok_or("HIP audio ConvTranspose1D output length 溢出")?;
    let output_length = expanded.checked_sub(padding.checked_mul(2).ok_or("HIP audio ConvTranspose1D padding 溢出")?).ok_or("HIP audio ConvTranspose1D padding 超过输出")?;
    let input_elements = batch.checked_mul(input_channels).and_then(|value| value.checked_mul(input_length)).ok_or("HIP audio ConvTranspose1D input 大小溢出")?;
    let weight_columns = output_channels.checked_mul(kernel).ok_or("HIP audio ConvTranspose1D weight columns 溢出")?;
    let weight_elements = input_channels.checked_mul(weight_columns).ok_or("HIP audio ConvTranspose1D weight 大小溢出")?;
    let output_elements = batch.checked_mul(output_channels).and_then(|value| value.checked_mul(output_length)).ok_or("HIP audio ConvTranspose1D output 大小溢出")?;
    validate_resident(input, device_id, input_elements.checked_mul(4).ok_or("HIP audio ConvTranspose1D input 字节溢出")?, "audio ConvTranspose1D input")?;
    validate_resident(weight_v, device_id, weight_elements.checked_mul(4).ok_or("HIP audio ConvTranspose1D weight 字节溢出")?, "audio ConvTranspose1D weight_v")?;
    validate_resident(bias, device_id, output_channels.checked_mul(4).ok_or("HIP audio ConvTranspose1D bias 字节溢出")?, "audio ConvTranspose1D bias")?;
    let normalized = try_audio_weight_norm_resident_f32(device_id, weight_g, weight_v, input_channels, weight_columns)?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, output_elements.checked_mul(4).ok_or("HIP audio ConvTranspose1D output 字节溢出")?)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_weight = normalized.pointer;
    let mut d_bias = bias.pointer;
    let mut d_output = output.pointer;
    let values = [batch, input_channels, output_channels, input_length, output_length, kernel, stride, padding, output_elements];
    let mut dims = values.into_iter().map(|value| u32::try_from(value).map_err(|_| "HIP audio ConvTranspose1D 维度超过 u32".to_owned())).collect::<Result<Vec<_>, _>>()?;
    let mut arguments: Vec<*mut c_void> = vec![(&mut d_input as *mut *mut c_void).cast(), (&mut d_weight as *mut *mut c_void).cast(), (&mut d_bias as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast()];
    arguments.extend(dims.iter_mut().map(|value| (value as *mut u32).cast()));
    let elements = u32::try_from(output_elements).map_err(|_| "HIP audio ConvTranspose1D 元素数超过 u32".to_owned())?;
    launch_tensor_kernel(functions.audio_conv_transpose1d, elements.div_ceil(256), 256, &mut arguments, "HIP audio ConvTranspose1D")?;
    Ok((output, output_length))
}

#[allow(clippy::too_many_arguments)]
pub fn try_audio_snake_beta_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    alpha: &DeviceBuffer,
    beta: &DeviceBuffer,
    up_filter: &DeviceBuffer,
    down_filter: &DeviceBuffer,
    batch: usize,
    channels: usize,
    length: usize,
    filter_kernel: usize,
) -> Result<DeviceBuffer, String> {
    if length == 0 || filter_kernel < 2 {
        return Err("HIP audio SnakeBeta length 必须非零且 filter kernel 至少为 2".to_owned());
    }
    let elements = batch.checked_mul(channels).and_then(|value| value.checked_mul(length)).ok_or("HIP audio SnakeBeta 大小溢出")?;
    let high_elements = elements.checked_mul(2).ok_or("HIP audio SnakeBeta 上采样大小溢出")?;
    let high_length = length.checked_mul(2).ok_or("HIP audio SnakeBeta 上采样长度溢出")?;
    let bytes = elements.checked_mul(4).ok_or("HIP audio SnakeBeta 字节数溢出")?;
    let high_bytes = high_elements.checked_mul(4).ok_or("HIP audio SnakeBeta 上采样字节数溢出")?;
    let parameter_bytes = channels.checked_mul(4).ok_or("HIP audio SnakeBeta 参数字节数溢出")?;
    let filter_bytes = filter_kernel.checked_mul(4).ok_or("HIP audio SnakeBeta filter 字节数溢出")?;
    validate_resident(input, device_id, bytes, "audio SnakeBeta input")?;
    validate_resident(alpha, device_id, parameter_bytes, "audio SnakeBeta alpha")?;
    validate_resident(beta, device_id, parameter_bytes, "audio SnakeBeta beta")?;
    validate_resident(up_filter, device_id, filter_bytes, "audio SnakeBeta up filter")?;
    validate_resident(down_filter, device_id, filter_bytes, "audio SnakeBeta down filter")?;
    set_device(device_id)?;
    let upsampled = DeviceBuffer::allocate(device_id, high_bytes)?;
    let activated = DeviceBuffer::allocate(device_id, high_bytes)?;
    let output = DeviceBuffer::allocate(device_id, bytes)?;
    let functions = tensor_functions(device_id)?;

    let mut d_input = input.pointer;
    let mut d_up_filter = up_filter.pointer;
    let mut d_upsampled = upsampled.pointer;
    let mut batch_u32 = u32::try_from(batch).map_err(|_| "HIP audio SnakeBeta batch 超过 u32".to_owned())?;
    let mut channels_u32 = u32::try_from(channels).map_err(|_| "HIP audio SnakeBeta channels 超过 u32".to_owned())?;
    let mut length_u32 = u32::try_from(length).map_err(|_| "HIP audio SnakeBeta length 超过 u32".to_owned())?;
    let mut filter_u32 = u32::try_from(filter_kernel).map_err(|_| "HIP audio SnakeBeta filter 超过 u32".to_owned())?;
    let mut high_elements_u32 = u32::try_from(high_elements).map_err(|_| "HIP audio SnakeBeta 上采样元素数超过 u32".to_owned())?;
    let mut up_arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_up_filter as *mut *mut c_void).cast(),
        (&mut d_upsampled as *mut *mut c_void).cast(),
        (&mut batch_u32 as *mut u32).cast(),
        (&mut channels_u32 as *mut u32).cast(),
        (&mut length_u32 as *mut u32).cast(),
        (&mut filter_u32 as *mut u32).cast(),
        (&mut high_elements_u32 as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.alias_upsample2, high_elements_u32.div_ceil(256), 256, &mut up_arguments, "HIP audio alias upsample")?;

    let mut d_alpha = alpha.pointer;
    let mut d_beta = beta.pointer;
    let mut d_activated = activated.pointer;
    let mut high_length_u32 = u32::try_from(high_length).map_err(|_| "HIP audio SnakeBeta 上采样长度超过 u32".to_owned())?;
    let mut snake_arguments = [
        (&mut d_upsampled as *mut *mut c_void).cast(),
        (&mut d_alpha as *mut *mut c_void).cast(),
        (&mut d_beta as *mut *mut c_void).cast(),
        (&mut d_activated as *mut *mut c_void).cast(),
        (&mut channels_u32 as *mut u32).cast(),
        (&mut high_length_u32 as *mut u32).cast(),
        (&mut high_elements_u32 as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.snake_beta, high_elements_u32.div_ceil(256), 256, &mut snake_arguments, "HIP audio SnakeBeta")?;

    let mut d_down_filter = down_filter.pointer;
    let mut d_output = output.pointer;
    let mut elements_u32 = u32::try_from(elements).map_err(|_| "HIP audio SnakeBeta output 元素数超过 u32".to_owned())?;
    let mut down_arguments = [
        (&mut d_activated as *mut *mut c_void).cast(),
        (&mut d_down_filter as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut length_u32 as *mut u32).cast(),
        (&mut filter_u32 as *mut u32).cast(),
        (&mut elements_u32 as *mut u32).cast(),
    ];
    launch_tensor_kernel(functions.alias_downsample2, elements_u32.div_ceil(256), 256, &mut down_arguments, "HIP audio alias downsample")?;
    Ok(output)
}
pub(super) const TENSOR_SOURCE: &str = include_str!("audio/source.hip");
