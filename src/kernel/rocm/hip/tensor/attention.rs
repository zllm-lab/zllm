pub(super) const SOURCE: &str = include_str!("attention/source.hip");

use super::*;

pub fn try_full_attention_qkv_resident_f32(
    device_id: i32,
    qkv: DeviceBuffer,
    query_weight: &[f32],
    key_weight: &[f32],
    rows: usize,
    head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
    cosine: &[f32],
    sine: &[f32],
    score_scale: f32,
) -> Result<DeviceBuffer, String> {
    try_full_attention_qkv_range_resident_f32(device_id, qkv, query_weight, key_weight, rows, head_count, 0, head_count, head_dim, rotary_dim, eps, cosine, sine, score_scale)
}

#[derive(Default)]
struct AttentionRopeTables {
    cosine_host: Vec<f32>,
    sine_host: Vec<f32>,
    cosine_device: Option<DeviceBuffer>,
    sine_device: Option<DeviceBuffer>,
}

thread_local! {
    static ATTENTION_ROPE_TABLES: std::cell::RefCell<std::collections::HashMap<(i32, usize), AttentionRopeTables>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

fn with_attention_rope_tables<R>(device_id: i32, cosine: &[f32], sine: &[f32], run: impl FnOnce(&DeviceBuffer, &DeviceBuffer) -> Result<R, String>) -> Result<R, String> {
    ATTENTION_ROPE_TABLES.with(|tables| {
        let mut tables = tables.borrow_mut();
        let tables = tables.entry(crate::kernel::rocm::hip::compute_workspace_key(device_id)).or_default();
        let cosine_bytes = cosine.len().checked_mul(4).ok_or("attention cosine table bytes 溢出")?;
        let sine_bytes = sine.len().checked_mul(4).ok_or("attention sine table bytes 溢出")?;
        if tables.cosine_device.as_ref().is_none_or(|buffer| buffer.bytes < cosine_bytes) {
            tables.cosine_device = Some(DeviceBuffer::allocate(device_id, cosine_bytes)?);
            tables.cosine_host.clear();
        }
        if tables.sine_device.as_ref().is_none_or(|buffer| buffer.bytes < sine_bytes) {
            tables.sine_device = Some(DeviceBuffer::allocate(device_id, sine_bytes)?);
            tables.sine_host.clear();
        }
        if tables.cosine_host != cosine {
            tables.cosine_device.as_ref().unwrap().copy_from_host(unsafe { std::slice::from_raw_parts(cosine.as_ptr().cast(), cosine_bytes) })?;
            tables.cosine_host.clear();
            tables.cosine_host.extend_from_slice(cosine);
        }
        if tables.sine_host != sine {
            tables.sine_device.as_ref().unwrap().copy_from_host(unsafe { std::slice::from_raw_parts(sine.as_ptr().cast(), sine_bytes) })?;
            tables.sine_host.clear();
            tables.sine_host.extend_from_slice(sine);
        }
        run(tables.cosine_device.as_ref().unwrap(), tables.sine_device.as_ref().unwrap())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn try_full_attention_qkv_range_resident_f32(
    device_id: i32,
    qkv: DeviceBuffer,
    query_weight: &[f32],
    key_weight: &[f32],
    rows: usize,
    total_head_count: usize,
    head_start: usize,
    head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
    cosine: &[f32],
    sine: &[f32],
    score_scale: f32,
) -> Result<DeviceBuffer, String> {
    if query_weight.len() != head_dim || key_weight.len() != head_dim {
        return Err(format!("resident fused QKV attention norm weight={}/{}，期望 {head_dim}", query_weight.len(), key_weight.len()));
    }
    let query_weight = DeviceBuffer::upload(device_id, unsafe { std::slice::from_raw_parts(query_weight.as_ptr().cast(), query_weight.len() * 4) })?;
    let key_weight = DeviceBuffer::upload(device_id, unsafe { std::slice::from_raw_parts(key_weight.as_ptr().cast(), key_weight.len() * 4) })?;
    try_full_attention_qkv_range_resident_weights_f32(device_id, qkv, &query_weight, &key_weight, rows, total_head_count, head_start, head_count, head_dim, rotary_dim, eps, cosine, sine, score_scale)
}

#[allow(clippy::too_many_arguments)]
pub fn try_full_attention_qkv_range_resident_weights_f32(
    device_id: i32,
    qkv: DeviceBuffer,
    query_weight: &DeviceBuffer,
    key_weight: &DeviceBuffer,
    rows: usize,
    total_head_count: usize,
    head_start: usize,
    head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
    cosine: &[f32],
    sine: &[f32],
    score_scale: f32,
) -> Result<DeviceBuffer, String> {
    if rows == 0
        || total_head_count == 0
        || head_count == 0
        || head_start.checked_add(head_count).is_none_or(|end| end > total_head_count)
        || head_dim == 0
        || head_dim > 128
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || !eps.is_finite()
        || eps < 0.0
        || !score_scale.is_finite()
        || score_scale <= 0.0
    {
        return Err(format!(
            "resident fused QKV attention rows={rows} total_heads={total_head_count} head_range={head_start}..{} head_dim={head_dim} rotary_dim={rotary_dim} eps={eps} scale={score_scale} 非法",
            head_start.saturating_add(head_count)
        ));
    }
    let total_columns = total_head_count.checked_mul(head_dim).ok_or("resident fused QKV attention total columns 溢出")?;
    let columns = head_count.checked_mul(head_dim).ok_or("resident fused QKV attention columns 溢出")?;
    let qkv_elements = rows.checked_mul(total_columns).and_then(|value| value.checked_mul(3)).ok_or("resident fused QKV attention elements 溢出")?;
    validate_resident(&qkv, device_id, qkv_elements.checked_mul(4).ok_or("resident fused QKV attention input bytes 溢出")?, "fused QKV attention input")?;
    let half = rotary_dim / 2;
    let table_elements = rows.checked_mul(half).ok_or("resident fused QKV attention table 溢出")?;
    if cosine.len() != table_elements || sine.len() != table_elements {
        return Err(format!("resident fused QKV attention table cos={} sin={} expected={table_elements}", cosine.len(), sine.len()));
    }
    let packed_rows = rows.div_ceil(128).checked_mul(128).ok_or("resident fused QKV attention padded rows 溢出")?;
    let packed_elements = packed_rows.checked_mul(columns).ok_or("resident fused QKV attention packed elements 溢出")?;
    let packed_bytes = packed_elements.checked_mul(2).ok_or("resident fused QKV attention packed bytes 溢出")?;
    let output_elements = rows.checked_mul(columns).ok_or("resident fused QKV attention output elements 溢出")?;
    let output_bytes = output_elements.checked_mul(4).ok_or("resident fused QKV attention output bytes 溢出")?;
    set_device(device_id)?;
    let packed_query = DeviceBuffer::allocate(device_id, packed_bytes)?;
    let packed_key = DeviceBuffer::allocate(device_id, packed_bytes)?;
    let packed_value = DeviceBuffer::allocate(device_id, packed_bytes)?;
    let functions = tensor_functions(device_id)?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let weight_bytes = head_dim.checked_mul(4).ok_or("resident fused QKV attention weight bytes 溢出")?;
    validate_resident(query_weight, device_id, weight_bytes, "fused QKV query norm weight")?;
    validate_resident(key_weight, device_id, weight_bytes, "fused QKV key norm weight")?;
    with_attention_rope_tables(device_id, cosine, sine, |cosine_device, sine_device| {
        let mut d_qkv = qkv.pointer;
        let mut d_query_weight = query_weight.pointer;
        let mut d_key_weight = key_weight.pointer;
        let mut d_cosine = cosine_device.pointer;
        let mut d_sine = sine_device.pointer;
        let mut d_query = packed_query.pointer;
        let mut d_key = packed_key.pointer;
        let mut d_value = packed_value.pointer;
        let mut rows = u32::try_from(rows).map_err(|_| "resident fused QKV attention rows 超过 u32")?;
        let mut packed_rows = u32::try_from(packed_rows).map_err(|_| "resident fused QKV attention packed rows 超过 u32")?;
        let mut total_head_count = u32::try_from(total_head_count).map_err(|_| "resident fused QKV attention total heads 超过 u32")?;
        let mut head_start = u32::try_from(head_start).map_err(|_| "resident fused QKV attention head start 超过 u32")?;
        let mut head_count = u32::try_from(head_count).map_err(|_| "resident fused QKV attention heads 超过 u32")?;
        let mut head_dim = u32::try_from(head_dim).map_err(|_| "resident fused QKV attention head_dim 超过 u32")?;
        let mut rotary_dim = u32::try_from(rotary_dim).map_err(|_| "resident fused QKV attention rotary_dim 超过 u32")?;
        let mut eps = eps;
        let mut arguments = [
            (&mut d_qkv as *mut *mut c_void).cast(),
            (&mut d_query_weight as *mut *mut c_void).cast(),
            (&mut d_key_weight as *mut *mut c_void).cast(),
            (&mut d_cosine as *mut *mut c_void).cast(),
            (&mut d_sine as *mut *mut c_void).cast(),
            (&mut d_query as *mut *mut c_void).cast(),
            (&mut d_key as *mut *mut c_void).cast(),
            (&mut d_value as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut packed_rows as *mut u32).cast(),
            (&mut total_head_count as *mut u32).cast(),
            (&mut head_start as *mut u32).cast(),
            (&mut head_count as *mut u32).cast(),
            (&mut head_dim as *mut u32).cast(),
            (&mut rotary_dim as *mut u32).cast(),
            (&mut eps as *mut f32).cast(),
        ];
        let grid = packed_rows.checked_mul(head_count).ok_or("resident fused QKV attention prepare grid 溢出")?;
        launch_tensor_kernel(functions.prepare_attention_qkv_bf16, grid, 128, &mut arguments, "HIP prepare attention QKV BF16")?;
        Ok(())
    })?;

    let generic_attention = options().generic_full_attention;
    let native_kv = head_dim == 128 && !generic_attention && options().native_full_attention_kv;
    if native_kv {
        let mut d_key = packed_key.pointer;
        let mut d_value = packed_value.pointer;
        let mut packed_rows = u32::try_from(packed_rows).map_err(|_| "resident fused QKV attention packed rows 超过 u32")?;
        let mut head_count = u32::try_from(head_count).map_err(|_| "resident fused QKV attention heads 超过 u32")?;
        let mut arguments = [(&mut d_key as *mut *mut c_void).cast(), (&mut d_value as *mut *mut c_void).cast(), (&mut packed_rows as *mut u32).cast(), (&mut head_count as *mut u32).cast()];
        let grid = packed_rows.checked_div(16).and_then(|tiles| tiles.checked_mul(head_count)).ok_or("resident fused QKV native K/V grid 溢出")?;
        launch_tensor_kernel(functions.pack_attention_kv_native_bf16, grid, 256, &mut arguments, "HIP pack native attention K/V")?;
    }

    let output = DeviceBuffer::allocate(device_id, output_bytes)?;
    let blocks_per_head = rows.div_ceil(128);
    let grid = blocks_per_head.checked_mul(head_count).ok_or("resident fused QKV attention grid 溢出")?;
    let mut d_query = packed_query.pointer;
    let mut d_key = packed_key.pointer;
    let mut d_value = packed_value.pointer;
    let mut d_output = output.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "resident fused QKV attention rows 超过 u32")?;
    let mut head_count = u32::try_from(head_count).map_err(|_| "resident fused QKV attention heads 超过 u32")?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "resident fused QKV attention head_dim 超过 u32")?;
    let mut score_scale = score_scale;
    let mut arguments = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_value as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut head_count as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
        (&mut score_scale as *mut f32).cast(),
    ];
    let attention_kernel = if head_dim == 128 && !generic_attention { if native_kv { functions.full_attention_128_native_kv } else { functions.full_attention_128 } } else { functions.full_attention };
    launch_tensor_kernel(attention_kernel, u32::try_from(grid).map_err(|_| "resident fused QKV attention grid 超过 u32")?, 256, &mut arguments, "HIP fused QKV full attention")?;
    if profile_started.is_some() || options().kernel_sync {
        synchronize_device(device_id, "fused QKV full attention")?;
    }
    drop(qkv);
    drop(packed_query);
    drop(packed_key);
    drop(packed_value);
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] fused-qkv-attention device={device_id} rows={rows} heads={head_count} dim={head_dim} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_full_attention_padded_heads_resident_f32(
    device_id: i32,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    rows: usize,
    heads: usize,
    head_dim: usize,
    padded_head_dim: usize,
    score_scale: f32,
) -> Result<DeviceBuffer, String> {
    if rows == 0 || heads == 0 || head_dim == 0 || padded_head_dim < head_dim || padded_head_dim > 128 || !padded_head_dim.is_multiple_of(16) {
        return Err(format!("padded attention rows={rows} heads={heads} dim={head_dim}/{padded_head_dim} 非法"));
    }
    let elements = rows.checked_mul(heads).and_then(|v| v.checked_mul(head_dim)).ok_or("padded attention elements 溢出")?;
    let padded_elements = rows.checked_mul(heads).and_then(|v| v.checked_mul(padded_head_dim)).ok_or("padded attention packed elements 溢出")?;
    let input_bytes = elements.checked_mul(4).ok_or("padded attention input bytes 溢出")?;
    let packed_bytes = padded_elements.checked_mul(2).ok_or("padded attention packed bytes 溢出")?;
    let padded_output_bytes = padded_elements.checked_mul(4).ok_or("padded attention output bytes 溢出")?;
    validate_resident(query, device_id, input_bytes, "padded attention query")?;
    validate_resident(key, device_id, input_bytes, "padded attention key")?;
    validate_resident(value, device_id, input_bytes, "padded attention value")?;
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let output = with_attention_workspace(device_id, &[packed_bytes, packed_bytes, packed_bytes, padded_output_bytes], |workspace| {
        let mut d_query = query.pointer;
        let mut d_key = key.pointer;
        let mut d_value = value.pointer;
        let mut d_packed_query = workspace.buffer(0).pointer;
        let mut d_packed_key = workspace.buffer(1).pointer;
        let mut d_packed_value = workspace.buffer(2).pointer;
        let mut rows_u32 = u32::try_from(rows).map_err(|_| "padded attention rows 超过 u32")?;
        let mut heads_u32 = u32::try_from(heads).map_err(|_| "padded attention heads 超过 u32")?;
        let mut head_dim_u32 = u32::try_from(head_dim).map_err(|_| "padded attention head dim 超过 u32")?;
        let mut padded_head_dim_u32 = u32::try_from(padded_head_dim).map_err(|_| "padded attention padded dim 超过 u32")?;
        let mut pack_args = [
            (&mut d_query as *mut *mut c_void).cast(),
            (&mut d_key as *mut *mut c_void).cast(),
            (&mut d_value as *mut *mut c_void).cast(),
            (&mut d_packed_query as *mut *mut c_void).cast(),
            (&mut d_packed_key as *mut *mut c_void).cast(),
            (&mut d_packed_value as *mut *mut c_void).cast(),
            (&mut rows_u32 as *mut u32).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut head_dim_u32 as *mut u32).cast(),
            (&mut padded_head_dim_u32 as *mut u32).cast(),
        ];
        let padded_elements_u32 = u32::try_from(padded_elements).map_err(|_| "padded attention packed elements 超过 u32")?;
        launch_tensor_kernel(functions.pack_attention_heads_bf16, padded_elements_u32.div_ceil(256), 256, &mut pack_args, "HIP pack padded attention")?;

        let mut d_padded_output = workspace.buffer(3).pointer;
        let mut score_scale = score_scale;
        let mut attention_args = [
            (&mut d_packed_query as *mut *mut c_void).cast(),
            (&mut d_packed_key as *mut *mut c_void).cast(),
            (&mut d_packed_value as *mut *mut c_void).cast(),
            (&mut d_padded_output as *mut *mut c_void).cast(),
            (&mut rows_u32 as *mut u32).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut padded_head_dim_u32 as *mut u32).cast(),
            (&mut score_scale as *mut f32).cast(),
        ];
        let grid = rows_u32.div_ceil(128).checked_mul(heads_u32).ok_or("padded attention grid 溢出")?;
        let function = if padded_head_dim == 128 { functions.full_attention_128 } else { functions.full_attention };
        launch_tensor_kernel(function, grid, 256, &mut attention_args, "HIP padded full attention")?;

        let output = DeviceBuffer::allocate(device_id, input_bytes)?;
        let mut d_output = output.pointer;
        let mut unpack_args = [
            (&mut d_padded_output as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows_u32 as *mut u32).cast(),
            (&mut heads_u32 as *mut u32).cast(),
            (&mut head_dim_u32 as *mut u32).cast(),
            (&mut padded_head_dim_u32 as *mut u32).cast(),
        ];
        let elements_u32 = u32::try_from(elements).map_err(|_| "padded attention output elements 超过 u32")?;
        launch_tensor_kernel(functions.unpack_attention_heads_f32, elements_u32.div_ceil(256), 256, &mut unpack_args, "HIP unpack padded attention")?;
        if profile_started.is_some() || options().kernel_sync {
            synchronize_device(device_id, "padded full attention")?;
        }
        Ok(output)
    })?;
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] attention-padded rows={rows} heads={heads} dim={head_dim}/{padded_head_dim} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_gqa_prefill_wmma_resident_f32(
    device_id: i32,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    rows: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
) -> Result<DeviceBuffer, String> {
    if rows == 0 || head_dim != 128 || head_count == 0 || kv_head_count == 0 || !head_count.is_multiple_of(kv_head_count) {
        return Err(format!("WMMA GQA rows={rows} heads={head_count}/{kv_head_count} dim={head_dim} 非法"));
    }
    let query_columns = head_count.checked_mul(head_dim).ok_or("WMMA GQA query columns 溢出")?;
    let kv_columns = kv_head_count.checked_mul(head_dim).ok_or("WMMA GQA KV columns 溢出")?;
    let query_elements = rows.checked_mul(query_columns).ok_or("WMMA GQA query elements 溢出")?;
    let kv_elements = rows.checked_mul(kv_columns).ok_or("WMMA GQA KV elements 溢出")?;
    validate_resident(query, device_id, query_elements.checked_mul(4).ok_or("WMMA GQA query bytes 溢出")?, "WMMA GQA query")?;
    validate_resident(key, device_id, kv_elements.checked_mul(4).ok_or("WMMA GQA key bytes 溢出")?, "WMMA GQA key")?;
    validate_resident(value, device_id, kv_elements.checked_mul(4).ok_or("WMMA GQA value bytes 溢出")?, "WMMA GQA value")?;
    let packed_query_elements = rows.div_ceil(128).checked_mul(128).and_then(|v| v.checked_mul(query_columns)).ok_or("WMMA GQA packed query 溢出")?;
    let query_bytes = packed_query_elements.checked_mul(2).ok_or("WMMA GQA packed query bytes 溢出")?;
    let kv_bytes = kv_elements.checked_mul(2).ok_or("WMMA GQA packed KV bytes 溢出")?;
    let output_bytes = query_elements.checked_mul(4).ok_or("WMMA GQA output bytes 溢出")?;
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let output = with_attention_workspace(device_id, &[query_bytes, kv_bytes, kv_bytes], |workspace| {
        let mut d_query = query.pointer;
        let mut d_key = key.pointer;
        let mut d_value = value.pointer;
        let mut d_packed_query = workspace.buffer(0).pointer;
        let mut d_packed_key = workspace.buffer(1).pointer;
        let mut d_packed_value = workspace.buffer(2).pointer;
        let mut query_elements = u32::try_from(query_elements).map_err(|_| "WMMA GQA query elements 超过 u32")?;
        let mut packed_query_elements = u32::try_from(packed_query_elements).map_err(|_| "WMMA GQA packed query elements 超过 u32")?;
        let mut kv_elements = u32::try_from(kv_elements).map_err(|_| "WMMA GQA KV elements 超过 u32")?;
        let mut packed_elements = packed_query_elements.max(kv_elements);
        let mut pack_args = [
            (&mut d_query as *mut *mut c_void).cast(),
            (&mut d_key as *mut *mut c_void).cast(),
            (&mut d_value as *mut *mut c_void).cast(),
            (&mut d_packed_query as *mut *mut c_void).cast(),
            (&mut d_packed_key as *mut *mut c_void).cast(),
            (&mut d_packed_value as *mut *mut c_void).cast(),
            (&mut query_elements as *mut u32).cast(),
            (&mut packed_query_elements as *mut u32).cast(),
            (&mut kv_elements as *mut u32).cast(),
            (&mut packed_elements as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.pack_gqa_bf16, packed_elements.div_ceil(256), 256, &mut pack_args, "HIP pack GQA BF16")?;
        let output = DeviceBuffer::allocate(device_id, output_bytes)?;
        let mut d_output = output.pointer;
        let mut rows = u32::try_from(rows).map_err(|_| "WMMA GQA rows 超过 u32")?;
        let mut head_count = u32::try_from(head_count).map_err(|_| "WMMA GQA heads 超过 u32")?;
        let mut kv_head_count = u32::try_from(kv_head_count).map_err(|_| "WMMA GQA KV heads 超过 u32")?;
        let mut head_dim = u32::try_from(head_dim).map_err(|_| "WMMA GQA head dim 超过 u32")?;
        let mut score_scale = score_scale;
        let mut args = [
            (&mut d_packed_query as *mut *mut c_void).cast(),
            (&mut d_packed_key as *mut *mut c_void).cast(),
            (&mut d_packed_value as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut head_count as *mut u32).cast(),
            (&mut kv_head_count as *mut u32).cast(),
            (&mut head_dim as *mut u32).cast(),
            (&mut score_scale as *mut f32).cast(),
        ];
        let grid = rows.div_ceil(128).checked_mul(head_count).ok_or("WMMA GQA grid 溢出")?;
        launch_tensor_kernel(functions.gqa_attention_128, grid, 256, &mut args, "HIP WMMA GQA prefill")?;
        if profile_started.is_some() || options().kernel_sync {
            synchronize_device(device_id, "WMMA GQA prefill")?;
        }
        Ok(output)
    })?;
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] gqa-wmma rows={rows} heads={head_count}/{kv_head_count} dim={head_dim} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_full_attention_resident_f32(device_id: i32, query: DeviceBuffer, key: DeviceBuffer, value: DeviceBuffer, rows: usize, head_count: usize, head_dim: usize, score_scale: f32) -> Result<DeviceBuffer, String> {
    try_full_attention_resident_f32_inner(device_id, query, key, value, 1, rows, head_count, head_dim, score_scale)
}

#[allow(clippy::too_many_arguments)]
pub fn try_block_attention_resident_f32(
    device_id: i32,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    visible_ranges: &[u32],
    query_rows: usize,
    kv_rows: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
) -> Result<DeviceBuffer, String> {
    if query_rows == 0
        || kv_rows == 0
        || head_count == 0
        || kv_head_count == 0
        || head_dim == 0
        || head_dim > 256
        || !head_count.is_multiple_of(kv_head_count)
        || visible_ranges.len() != query_rows * 2
        || !score_scale.is_finite()
        || score_scale <= 0.0
    {
        return Err(format!("resident block attention 参数非法: Q={query_rows} KV={kv_rows} heads={head_count}/{kv_head_count} dim={head_dim} visible={}", visible_ranges.len()));
    }
    let query_columns = head_count.checked_mul(head_dim).ok_or("block attention query columns 溢出")?;
    let kv_columns = kv_head_count.checked_mul(head_dim).ok_or("block attention KV columns 溢出")?;
    let query_bytes = query_rows.checked_mul(query_columns).and_then(|v| v.checked_mul(4)).ok_or("block attention query bytes 溢出")?;
    let kv_bytes = kv_rows.checked_mul(kv_columns).and_then(|v| v.checked_mul(4)).ok_or("block attention KV bytes 溢出")?;
    validate_resident(query, device_id, query_bytes, "block attention query")?;
    validate_resident(key, device_id, kv_bytes, "block attention key")?;
    validate_resident(value, device_id, kv_bytes, "block attention value")?;
    for (query_row, range) in visible_ranges.chunks_exact(2).enumerate() {
        if range[0] >= range[1] || range[1] as usize > kv_rows {
            return Err(format!("block attention query={query_row} visible={}..{} 越界", range[0], range[1]));
        }
    }
    set_device(device_id)?;
    let ranges = DeviceBuffer::upload(device_id, unsafe { std::slice::from_raw_parts(visible_ranges.as_ptr().cast(), std::mem::size_of_val(visible_ranges)) })?;
    let output = DeviceBuffer::allocate_reusable(device_id, query_bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_query = query.pointer;
    let mut d_key = key.pointer;
    let mut d_value = value.pointer;
    let mut d_ranges = ranges.pointer;
    let mut d_output = output.pointer;
    let mut query_rows = u32::try_from(query_rows).map_err(|_| "block attention query rows 超过 u32")?;
    let mut kv_rows = u32::try_from(kv_rows).map_err(|_| "block attention KV rows 超过 u32")?;
    let mut head_count = u32::try_from(head_count).map_err(|_| "block attention heads 超过 u32")?;
    let mut kv_head_count = u32::try_from(kv_head_count).map_err(|_| "block attention KV heads 超过 u32")?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "block attention dim 超过 u32")?;
    let mut score_scale = score_scale;
    let mut arguments = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_value as *mut *mut c_void).cast(),
        (&mut d_ranges as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut query_rows as *mut u32).cast(),
        (&mut kv_rows as *mut u32).cast(),
        (&mut head_count as *mut u32).cast(),
        (&mut kv_head_count as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
        (&mut score_scale as *mut f32).cast(),
    ];
    let grid = query_rows.checked_mul(head_count).ok_or("block attention grid 溢出")?;
    let threads = head_dim.next_power_of_two().max(32);
    launch_tensor_kernel(functions.block_attention, grid, threads, &mut arguments, "HIP resident block attention")?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_block_attention_prefix_suffix_resident_f32(
    device_id: i32,
    query: &DeviceBuffer,
    prefix_key: &DeviceBuffer,
    prefix_value: &DeviceBuffer,
    suffix_key: &DeviceBuffer,
    suffix_value: &DeviceBuffer,
    visible_ranges: &[u32],
    query_rows: usize,
    prefix_rows: usize,
    suffix_rows: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    score_scale: f32,
) -> Result<DeviceBuffer, String> {
    let kv_rows = prefix_rows.checked_add(suffix_rows).ok_or("split block attention KV rows 溢出")?;
    if query_rows == 0
        || prefix_rows == 0
        || suffix_rows == 0
        || head_count == 0
        || kv_head_count == 0
        || head_dim == 0
        || head_dim > 256
        || !head_count.is_multiple_of(kv_head_count)
        || visible_ranges.len() != query_rows * 2
        || !score_scale.is_finite()
        || score_scale <= 0.0
    {
        return Err(format!("resident split block attention 参数非法: Q={query_rows} KV={prefix_rows}+{suffix_rows} heads={head_count}/{kv_head_count} dim={head_dim} visible={}", visible_ranges.len()));
    }
    let query_columns = head_count.checked_mul(head_dim).ok_or("split block attention query columns 溢出")?;
    let kv_columns = kv_head_count.checked_mul(head_dim).ok_or("split block attention KV columns 溢出")?;
    let query_bytes = query_rows.checked_mul(query_columns).and_then(|v| v.checked_mul(4)).ok_or("split block attention query bytes 溢出")?;
    let prefix_bytes = prefix_rows.checked_mul(kv_columns).and_then(|v| v.checked_mul(4)).ok_or("split block attention prefix bytes 溢出")?;
    let suffix_bytes = suffix_rows.checked_mul(kv_columns).and_then(|v| v.checked_mul(4)).ok_or("split block attention suffix bytes 溢出")?;
    validate_resident(query, device_id, query_bytes, "split block attention query")?;
    validate_resident(prefix_key, device_id, prefix_bytes, "split block attention prefix key")?;
    validate_resident(prefix_value, device_id, prefix_bytes, "split block attention prefix value")?;
    validate_resident(suffix_key, device_id, suffix_bytes, "split block attention suffix key")?;
    validate_resident(suffix_value, device_id, suffix_bytes, "split block attention suffix value")?;
    for (query_row, range) in visible_ranges.chunks_exact(2).enumerate() {
        if range[0] >= range[1] || range[1] as usize > kv_rows {
            return Err(format!("split block attention query={query_row} visible={}..{} 越界", range[0], range[1]));
        }
    }
    set_device(device_id)?;
    let ranges = DeviceBuffer::upload(device_id, unsafe { std::slice::from_raw_parts(visible_ranges.as_ptr().cast(), std::mem::size_of_val(visible_ranges)) })?;
    let output = DeviceBuffer::allocate_reusable(device_id, query_bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_query = query.pointer;
    let mut d_prefix_key = prefix_key.pointer;
    let mut d_prefix_value = prefix_value.pointer;
    let mut d_suffix_key = suffix_key.pointer;
    let mut d_suffix_value = suffix_value.pointer;
    let mut d_ranges = ranges.pointer;
    let mut d_output = output.pointer;
    let mut query_rows = u32::try_from(query_rows).map_err(|_| "split block attention query rows 超过 u32")?;
    let mut prefix_rows = u32::try_from(prefix_rows).map_err(|_| "split block attention prefix rows 超过 u32")?;
    let mut suffix_rows = u32::try_from(suffix_rows).map_err(|_| "split block attention suffix rows 超过 u32")?;
    let mut head_count = u32::try_from(head_count).map_err(|_| "split block attention heads 超过 u32")?;
    let mut kv_head_count = u32::try_from(kv_head_count).map_err(|_| "split block attention KV heads 超过 u32")?;
    let mut head_dim = u32::try_from(head_dim).map_err(|_| "split block attention dim 超过 u32")?;
    let mut score_scale = score_scale;
    let mut arguments = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_prefix_key as *mut *mut c_void).cast(),
        (&mut d_prefix_value as *mut *mut c_void).cast(),
        (&mut d_suffix_key as *mut *mut c_void).cast(),
        (&mut d_suffix_value as *mut *mut c_void).cast(),
        (&mut d_ranges as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut query_rows as *mut u32).cast(),
        (&mut prefix_rows as *mut u32).cast(),
        (&mut suffix_rows as *mut u32).cast(),
        (&mut head_count as *mut u32).cast(),
        (&mut kv_head_count as *mut u32).cast(),
        (&mut head_dim as *mut u32).cast(),
        (&mut score_scale as *mut f32).cast(),
    ];
    let grid = query_rows.checked_mul(head_count).ok_or("split block attention grid 溢出")?;
    let threads = head_dim.next_power_of_two().max(32);
    launch_tensor_kernel(functions.block_attention_prefix_suffix, grid, threads, &mut arguments, "HIP resident split block attention")?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn try_full_attention_batched_resident_f32(device_id: i32, query: DeviceBuffer, key: DeviceBuffer, value: DeviceBuffer, batch: usize, rows: usize, head_count: usize, head_dim: usize, score_scale: f32) -> Result<DeviceBuffer, String> {
    try_full_attention_resident_f32_inner(device_id, query, key, value, batch, rows, head_count, head_dim, score_scale)
}

#[allow(clippy::too_many_arguments)]
fn try_full_attention_resident_f32_inner(device_id: i32, query: DeviceBuffer, key: DeviceBuffer, value: DeviceBuffer, batch: usize, rows: usize, head_count: usize, head_dim: usize, score_scale: f32) -> Result<DeviceBuffer, String> {
    if batch == 0 || rows == 0 || head_count == 0 || head_dim == 0 || head_dim > 128 || !head_dim.is_multiple_of(16) || !score_scale.is_finite() || score_scale <= 0.0 {
        return Err(format!("resident full attention rows={rows} heads={head_count} head_dim={head_dim} scale={score_scale} 非法"));
    }
    let columns = head_count.checked_mul(head_dim).ok_or("resident full attention columns 溢出")?;
    let total_rows = batch.checked_mul(rows).ok_or("resident full attention total rows 溢出")?;
    let elements = total_rows.checked_mul(columns).ok_or("resident full attention elements 溢出")?;
    let bytes = elements.checked_mul(4).ok_or("resident full attention 字节溢出")?;
    validate_resident(&query, device_id, bytes, "full attention query")?;
    validate_resident(&key, device_id, bytes, "full attention key")?;
    validate_resident(&value, device_id, bytes, "full attention value")?;
    let dim64_batched = batch != 1 && head_dim == 64;
    let query_block_rows = if dim64_batched { 192 } else { 128 };
    let packed_rows = if dim64_batched { total_rows.checked_add(query_block_rows - 1).ok_or("full attention padded rows 溢出")? } else { total_rows.div_ceil(128).checked_mul(128).ok_or("full attention padded rows 溢出")? };
    let packed_elements = packed_rows.checked_mul(columns).ok_or("full attention packed elements 溢出")?;
    let packed_bytes = packed_elements.checked_mul(2).ok_or("full attention packed 字节溢出")?;
    let grid = batch.checked_mul(rows.div_ceil(query_block_rows)).and_then(|v| v.checked_mul(head_count)).ok_or("full attention grid 溢出")?;
    let grid = u32::try_from(grid).map_err(|_| "full attention grid 超过 u32")?;
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let output = with_attention_workspace(device_id, &[packed_bytes; 3], move |workspace| {
        let mut d_query = query.pointer;
        let mut d_key = key.pointer;
        let mut d_value = value.pointer;
        let mut d_packed_query = workspace.buffer(0).pointer;
        let mut d_packed_key = workspace.buffer(1).pointer;
        let mut d_packed_value = workspace.buffer(2).pointer;
        let mut elements = u32::try_from(elements).map_err(|_| "full attention elements 超过 u32")?;
        let mut packed_elements = u32::try_from(packed_elements).map_err(|_| "full attention packed elements 超过 u32")?;
        let mut pack_arguments = [
            (&mut d_query as *mut *mut c_void).cast(),
            (&mut d_key as *mut *mut c_void).cast(),
            (&mut d_value as *mut *mut c_void).cast(),
            (&mut d_packed_query as *mut *mut c_void).cast(),
            (&mut d_packed_key as *mut *mut c_void).cast(),
            (&mut d_packed_value as *mut *mut c_void).cast(),
            (&mut elements as *mut u32).cast(),
            (&mut packed_elements as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.pack_qkv_bf16, packed_elements.div_ceil(256), 256, &mut pack_arguments, "HIP pack QKV BF16")?;
        synchronize_device(device_id, "full attention pack QKV before releasing F32")?;
        drop(query);
        drop(key);
        drop(value);

        let output = DeviceBuffer::allocate(device_id, bytes)?;
        let mut d_output = output.pointer;
        let mut rows = u32::try_from(rows).map_err(|_| "full attention rows 超过 u32")?;
        let mut head_count = u32::try_from(head_count).map_err(|_| "full attention heads 超过 u32")?;
        let mut head_dim = u32::try_from(head_dim).map_err(|_| "full attention head_dim 超过 u32")?;
        let mut score_scale = score_scale;
        let mut arguments = [
            (&mut d_packed_query as *mut *mut c_void).cast(),
            (&mut d_packed_key as *mut *mut c_void).cast(),
            (&mut d_packed_value as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut head_count as *mut u32).cast(),
            (&mut head_dim as *mut u32).cast(),
            (&mut score_scale as *mut f32).cast(),
        ];
        let function = if batch != 1 && head_dim == 64 {
            functions.full_attention_batched_64
        } else {
            match (batch == 1, head_dim == 128) {
                (true, true) => functions.full_attention_128,
                (true, false) => functions.full_attention,
                (false, true) => functions.full_attention_batched_128,
                (false, false) => functions.full_attention_batched,
            }
        };
        launch_tensor_kernel(function, grid, if dim64_batched { 384 } else { 256 }, &mut arguments, "HIP resident full attention")?;
        if options().kernel_sync || profile_started.is_some() {
            synchronize_device(device_id, &format!("full attention synchronize rows={rows} heads={head_count} dim={head_dim}"))?;
        }
        Ok(output)
    })?;
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] attention device={device_id} batch={batch} rows={rows} heads={head_count} dim={head_dim} wall={:.6}s", started.elapsed().as_secs_f64(),);
    }
    Ok(output)
}

/// 把 f32 K/V 行追加到常驻设备 GQA 缓冲（纯 D2D 拷贝，`offset_elements` 为目标缓冲内偏移）。
pub fn try_gqa_cache_append_f32(device_id: i32, source: &DeviceBuffer, target: &DeviceBuffer, offset_elements: usize, elements: usize) -> Result<(), String> {
    if elements == 0 {
        return Ok(());
    }
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let mut d_source = source.pointer;
    let mut d_target = target.pointer;
    let mut offset = u64::try_from(offset_elements).map_err(|_| "GQA cache append offset 超过 u64")?;
    let mut count = u64::try_from(elements).map_err(|_| "GQA cache append elements 超过 u64")?;
    let mut args = [(&mut d_source as *mut *mut c_void).cast(), (&mut d_target as *mut *mut c_void).cast(), (&mut offset as *mut u64).cast(), (&mut count as *mut u64).cast()];
    let grid = (count as u32).div_ceil(256);
    launch_tensor_kernel(functions.gqa_cache_append, grid, 256, &mut args, "HIP GQA cache append")?;
    if options().kernel_sync {
        synchronize_device(device_id, "GQA cache append")?;
    }
    Ok(())
}

/// 单 query 行对常驻设备 K/V 前缀的 GQA decode attention（flash-decode split-K 两阶段）。
/// 返回 [1, head_count*head_dim] 的 f32 设备输出；K/V 布局为 [rows][kv_heads*head_dim]。
#[allow(clippy::too_many_arguments)]
pub fn try_gqa_decode_cached_f32(device_id: i32, query: &DeviceBuffer, key: &DeviceBuffer, value: &DeviceBuffer, rows: usize, head_count: usize, kv_head_count: usize, head_dim: usize, score_scale: f32) -> Result<DeviceBuffer, String> {
    if rows == 0 || head_count == 0 || kv_head_count == 0 || !head_count.is_multiple_of(kv_head_count) || head_dim == 0 || head_dim > 256 {
        return Err(format!("GQA decode rows={rows} heads={head_count}/{kv_head_count} dim={head_dim} 非法"));
    }
    let kv_cols = kv_head_count.checked_mul(head_dim).ok_or("GQA decode KV columns 溢出")?;
    let query_bytes = head_count.checked_mul(head_dim).and_then(|n| n.checked_mul(4)).ok_or("GQA decode query bytes 溢出")?;
    let kv_bytes = rows.checked_mul(kv_cols).and_then(|n| n.checked_mul(4)).ok_or("GQA decode KV bytes 溢出")?;
    validate_resident(query, device_id, query_bytes, "GQA decode query")?;
    validate_resident(key, device_id, kv_bytes, "GQA decode key")?;
    validate_resident(value, device_id, kv_bytes, "GQA decode value")?;
    let tiles = rows.div_ceil(128);
    set_device(device_id)?;
    let functions = tensor_functions(device_id)?;
    let profile_started = options().kernel_profile.then(std::time::Instant::now);
    let stats = DeviceBuffer::allocate(device_id, head_count.checked_mul(tiles).and_then(|n| n.checked_mul(8)).ok_or("GQA decode stats bytes 溢出")?)?;
    let partial = DeviceBuffer::allocate(device_id, head_count.checked_mul(tiles).and_then(|n| n.checked_mul(head_dim)).and_then(|n| n.checked_mul(4)).ok_or("GQA decode partial bytes 溢出")?)?;
    let output = DeviceBuffer::allocate(device_id, query_bytes)?;
    let mut d_query = query.pointer;
    let mut d_key = key.pointer;
    let mut d_value = value.pointer;
    let mut d_partial = partial.pointer;
    let mut d_stats = stats.pointer;
    let mut rows_u32 = u32::try_from(rows).map_err(|_| "GQA decode rows 超过 u32")?;
    let mut kv_cols_u32 = u32::try_from(kv_cols).map_err(|_| "GQA decode KV columns 超过 u32")?;
    let mut head_dim_u32 = u32::try_from(head_dim).map_err(|_| "GQA decode head dim 超过 u32")?;
    let mut group_u32 = u32::try_from(head_count / kv_head_count).map_err(|_| "GQA decode group 超过 u32")?;
    let mut tiles_u32 = u32::try_from(tiles).map_err(|_| "GQA decode tiles 超过 u32")?;
    let mut score_scale_var = score_scale;
    let mut partial_args = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_value as *mut *mut c_void).cast(),
        (&mut d_partial as *mut *mut c_void).cast(),
        (&mut d_stats as *mut *mut c_void).cast(),
        (&mut rows_u32 as *mut u32).cast(),
        (&mut kv_cols_u32 as *mut u32).cast(),
        (&mut head_dim_u32 as *mut u32).cast(),
        (&mut group_u32 as *mut u32).cast(),
        (&mut score_scale_var as *mut f32).cast(),
        (&mut tiles_u32 as *mut u32).cast(),
    ];
    let grid = tiles_u32.checked_mul(u32::try_from(head_count).map_err(|_| "GQA decode heads 超过 u32")?).ok_or("GQA decode grid 溢出")?;
    launch_tensor_kernel(functions.gqa_decode_partial, grid, 256, &mut partial_args, "HIP GQA decode partial")?;
    let mut d_output = output.pointer;
    let mut merge_args = [(&mut d_partial as *mut *mut c_void).cast(), (&mut d_stats as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut tiles_u32 as *mut u32).cast(), (&mut head_dim_u32 as *mut u32).cast()];
    launch_tensor_kernel(functions.gqa_decode_merge, u32::try_from(head_count).map_err(|_| "GQA decode heads 超过 u32")?, 256, &mut merge_args, "HIP GQA decode merge")?;
    if profile_started.is_some() || options().kernel_sync {
        synchronize_device(device_id, "GQA decode")?;
    }
    if let Some(started) = profile_started {
        eprintln!("[rocm-kernel] gqa-decode rows={rows} heads={head_count}/{kv_head_count} dim={head_dim} tiles={tiles} wall={:.6}s", started.elapsed().as_secs_f64());
    }
    Ok(output)
}
