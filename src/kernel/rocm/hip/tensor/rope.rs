pub(super) const SOURCE: &str = include_str!("rope/source.hip");

use super::*;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct ResidentRopeTableKey {
    device_id: i32,
    cosine_ptr: usize,
    cosine_len: usize,
    sine_ptr: usize,
    sine_len: usize,
    row_width: usize,
}

struct ResidentRopeTables {
    cosine: std::sync::Arc<DeviceBuffer>,
    sine: std::sync::Arc<DeviceBuffer>,
    row_hashes: Vec<[u8; 32]>,
}

static RESIDENT_ROPE_TABLES: std::sync::OnceLock<std::sync::Mutex<HashMap<ResidentRopeTableKey, ResidentRopeTables>>> = std::sync::OnceLock::new();

pub(crate) fn resident_rope_tables(device_id: i32, cosine: &[f32], sine: &[f32], row_width: usize, rows: impl IntoIterator<Item = usize>) -> Result<(std::sync::Arc<DeviceBuffer>, std::sync::Arc<DeviceBuffer>), String> {
    if row_width == 0 || cosine.len() != sine.len() || !cosine.len().is_multiple_of(row_width) {
        return Err(format!("resident RoPE table shape 非法: cosine={} sine={} row_width={row_width}", cosine.len(), sine.len()));
    }
    let requested_rows = rows.into_iter().collect::<Vec<_>>();
    let table_rows = cosine.len() / row_width;
    if requested_rows.iter().any(|&row| row >= table_rows) {
        return Err(format!("resident RoPE 请求行越界: rows={requested_rows:?} table_rows={table_rows}"));
    }
    let hash_row = |row: usize| {
        let begin = row * row_width;
        let end = begin + row_width;
        let mut hash = blake3::Hasher::new();
        hash.update(unsafe { std::slice::from_raw_parts(cosine[begin..end].as_ptr().cast(), row_width * std::mem::size_of::<f32>()) });
        hash.update(unsafe { std::slice::from_raw_parts(sine[begin..end].as_ptr().cast(), row_width * std::mem::size_of::<f32>()) });
        *hash.finalize().as_bytes()
    };
    let key = ResidentRopeTableKey { device_id, cosine_ptr: cosine.as_ptr() as usize, cosine_len: cosine.len(), sine_ptr: sine.as_ptr() as usize, sine_len: sine.len(), row_width };
    let mut tables = RESIDENT_ROPE_TABLES.get_or_init(|| std::sync::Mutex::new(HashMap::new())).lock().map_err(|_| "resident RoPE table cache 已损坏".to_owned())?;
    if let Some(resident) = tables.get(&key)
        && requested_rows.iter().all(|&row| resident.row_hashes[row] == hash_row(row))
    {
        return Ok((resident.cosine.clone(), resident.sine.clone()));
    }
    let cosine_bytes = unsafe { std::slice::from_raw_parts(cosine.as_ptr().cast(), std::mem::size_of_val(cosine)) };
    let sine_bytes = unsafe { std::slice::from_raw_parts(sine.as_ptr().cast(), std::mem::size_of_val(sine)) };
    let resident =
        ResidentRopeTables { cosine: std::sync::Arc::new(DeviceBuffer::upload(device_id, cosine_bytes)?), sine: std::sync::Arc::new(DeviceBuffer::upload(device_id, sine_bytes)?), row_hashes: (0..table_rows).map(hash_row).collect() };
    let result = (resident.cosine.clone(), resident.sine.clone());
    tables.insert(key, resident);
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub fn try_rope_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    rows: usize,
    cols: usize,
    head_count: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    position: usize,
    cos: &[f32],
    sin: &[f32],
    prefix: bool,
) -> Result<DeviceBuffer, String> {
    let half = rotary_dim / 2;
    let end = position.checked_add(rows).and_then(|n| n.checked_mul(half)).ok_or("resident RoPE table 大小溢出")?;
    if end > cos.len() || end > sin.len() {
        return Err("resident RoPE table 太短".to_owned());
    }
    // cos/sin 对所有层和 token 相同；整表常驻后由 kernel 用 position 直接定位。
    let (cosine, sine) = resident_rope_tables(device_id, cos, sin, half, position..position + rows)?;
    try_rope_with_resident_tables_f32(device_id, input, rows, cols, head_count, rotary_dim, layout, position, &cosine, &sine, prefix)
}

/// 已取得常驻 RoPE 表时直接提交 kernel。双卡 operator worker 不能借用调用
/// 线程上的 host slice，因此显式传递同设备的 resident buffer。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_rope_with_resident_tables_f32(
    device_id: i32,
    input: &DeviceBuffer,
    rows: usize,
    cols: usize,
    head_count: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    position: usize,
    cosine: &DeviceBuffer,
    sine: &DeviceBuffer,
    prefix: bool,
) -> Result<DeviceBuffer, String> {
    let input_bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(4)).ok_or("resident RoPE 大小溢出")?;
    validate_resident(input, device_id, input_bytes, "RoPE input")?;
    let half = rotary_dim / 2;
    let required_table_bytes = position.checked_add(rows).and_then(|n| n.checked_mul(half)).and_then(|n| n.checked_mul(4)).ok_or("resident RoPE table 大小溢出")?;
    validate_resident(cosine, device_id, required_table_bytes, "RoPE cosine")?;
    validate_resident(sine, device_id, required_table_bytes, "RoPE sine")?;
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, input_bytes)?;
    let functions = tensor_functions(device_id)?;
    let mut d_input = input.pointer;
    let mut d_cosine = cosine.pointer;
    let mut d_sine = sine.pointer;
    let mut d_output = output.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "resident RoPE rows 超过 u32".to_owned())?;
    let mut cols = u32::try_from(cols).map_err(|_| "resident RoPE cols 超过 u32".to_owned())?;
    let mut head_count = u32::try_from(head_count).map_err(|_| "resident RoPE heads 超过 u32".to_owned())?;
    let mut rotary_dim = u32::try_from(rotary_dim).map_err(|_| "resident RoPE dim 超过 u32".to_owned())?;
    let mut table_position = u32::try_from(position).map_err(|_| "resident RoPE position 超过 u32".to_owned())?;
    let mut split_half = u32::from(layout == RotaryLayout::SplitHalf);
    let mut prefix = u32::from(prefix);
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_cosine as *mut *mut c_void).cast(),
        (&mut d_sine as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut cols as *mut u32).cast(),
        (&mut head_count as *mut u32).cast(),
        (&mut rotary_dim as *mut u32).cast(),
        (&mut table_position as *mut u32).cast(),
        (&mut split_half as *mut u32).cast(),
        (&mut prefix as *mut u32).cast(),
    ];
    let elements = rows.checked_mul(cols).ok_or("resident RoPE elements 溢出")?;
    launch_tensor_kernel(functions.rope, elements.div_ceil(256), 256, &mut arguments, "HIP resident rope")?;
    Ok(output)
}

/// graph 兼容变体：position 从 device buffer 读取，replay 期参数地址不变。
/// `position_buffer` 由调用方持有并每 token 更新（graph 外 H2D）。
#[allow(clippy::too_many_arguments)]
pub fn try_rope_indirect_resident_f32(
    device_id: i32,
    input: &DeviceBuffer,
    rows: usize,
    cols: usize,
    head_count: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    position_buffer: &DeviceBuffer,
    cos: &[f32],
    sin: &[f32],
    prefix: bool,
) -> Result<DeviceBuffer, String> {
    let input_bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(4)).ok_or("resident RoPE 大小溢出")?;
    validate_resident(input, device_id, input_bytes, "RoPE input")?;
    validate_resident(position_buffer, device_id, 4, "RoPE indirect position")?;
    let half = rotary_dim / 2;
    if cos.len() < half || sin.len() < half {
        return Err("resident RoPE table 太短".to_owned());
    }
    set_device(device_id)?;
    let output = DeviceBuffer::allocate(device_id, input_bytes)?;
    let functions = tensor_functions(device_id)?;
    // graph 模式下 table 必须整表常驻：录制后无法再按 position 切片。
    let table_rows = cos.len() / half;
    let (cosine, sine) = resident_rope_tables(device_id, cos, sin, half, 0..table_rows)?;
    let mut d_input = input.pointer;
    let mut d_cosine = cosine.pointer;
    let mut d_sine = sine.pointer;
    let mut d_output = output.pointer;
    let mut d_position = position_buffer.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "resident RoPE rows 超过 u32".to_owned())?;
    let mut cols = u32::try_from(cols).map_err(|_| "resident RoPE cols 超过 u32".to_owned())?;
    let mut head_count = u32::try_from(head_count).map_err(|_| "resident RoPE heads 超过 u32".to_owned())?;
    let mut rotary_dim = u32::try_from(rotary_dim).map_err(|_| "resident RoPE dim 超过 u32".to_owned())?;
    let mut split_half = u32::from(layout == RotaryLayout::SplitHalf);
    let mut prefix = u32::from(prefix);
    let mut arguments = [
        (&mut d_input as *mut *mut c_void).cast(),
        (&mut d_cosine as *mut *mut c_void).cast(),
        (&mut d_sine as *mut *mut c_void).cast(),
        (&mut d_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut cols as *mut u32).cast(),
        (&mut head_count as *mut u32).cast(),
        (&mut rotary_dim as *mut u32).cast(),
        (&mut d_position as *mut *mut c_void).cast(),
        (&mut split_half as *mut u32).cast(),
        (&mut prefix as *mut u32).cast(),
    ];
    let elements = rows.checked_mul(cols).ok_or("resident RoPE elements 溢出")?;
    launch_tensor_kernel(functions.rope_indirect, elements.div_ceil(256), 256, &mut arguments, "HIP resident indirect rope")?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn try_rope_segmented_pair_resident_f32(
    device_id: i32,
    query: &DeviceBuffer,
    rows: usize,
    query_cols: usize,
    query_head_count: usize,
    key: &DeviceBuffer,
    key_cols: usize,
    key_head_count: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    row_positions: &[u32],
    cos: &[f32],
    sin: &[f32],
) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    if rows == 0 || query_head_count == 0 || key_head_count == 0 || rotary_dim == 0 || rotary_dim % 2 != 0 {
        return Err("resident segmented RoPE shape 无效".to_owned());
    }
    if query_cols % query_head_count != 0 || key_cols % key_head_count != 0 || rotary_dim > query_cols / query_head_count || rotary_dim > key_cols / key_head_count {
        return Err("resident segmented RoPE head shape 无效".to_owned());
    }
    if row_positions.len() != rows {
        return Err(format!("resident segmented RoPE positions={}，rows={rows}", row_positions.len()));
    }
    let query_bytes = rows.checked_mul(query_cols).and_then(|n| n.checked_mul(4)).ok_or("resident segmented RoPE query 大小溢出")?;
    let key_bytes = rows.checked_mul(key_cols).and_then(|n| n.checked_mul(4)).ok_or("resident segmented RoPE key 大小溢出")?;
    validate_resident(query, device_id, query_bytes, "segmented RoPE query")?;
    validate_resident(key, device_id, key_bytes, "segmented RoPE key")?;
    let half = rotary_dim / 2;
    let table_rows = row_positions.iter().copied().max().ok_or("resident segmented RoPE positions 为空")? as usize + 1;
    let table_end = table_rows.checked_mul(half).ok_or("resident segmented RoPE table 大小溢出")?;
    if table_end > cos.len() || table_end > sin.len() {
        return Err("resident segmented RoPE table 太短".to_owned());
    }
    set_device(device_id)?;
    let query_output = DeviceBuffer::allocate(device_id, query_bytes)?;
    let key_output = DeviceBuffer::allocate(device_id, key_bytes)?;
    let position_bytes = unsafe { std::slice::from_raw_parts(row_positions.as_ptr().cast(), std::mem::size_of_val(row_positions)) };
    let positions = DeviceBuffer::upload(device_id, position_bytes)?;
    let (cosine, sine) = resident_rope_tables(device_id, cos, sin, half, row_positions.iter().map(|&position| position as usize))?;
    let functions = tensor_functions(device_id)?;
    let mut d_query = query.pointer;
    let mut d_key = key.pointer;
    let mut d_cosine = cosine.pointer;
    let mut d_sine = sine.pointer;
    let mut d_positions = positions.pointer;
    let mut d_query_output = query_output.pointer;
    let mut d_key_output = key_output.pointer;
    let mut rows = u32::try_from(rows).map_err(|_| "resident segmented RoPE rows 超过 u32".to_owned())?;
    let mut query_cols = u32::try_from(query_cols).map_err(|_| "resident segmented RoPE query cols 超过 u32".to_owned())?;
    let mut query_head_count = u32::try_from(query_head_count).map_err(|_| "resident segmented RoPE query heads 超过 u32".to_owned())?;
    let mut key_cols = u32::try_from(key_cols).map_err(|_| "resident segmented RoPE key cols 超过 u32".to_owned())?;
    let mut key_head_count = u32::try_from(key_head_count).map_err(|_| "resident segmented RoPE key heads 超过 u32".to_owned())?;
    let mut rotary_dim = u32::try_from(rotary_dim).map_err(|_| "resident segmented RoPE dim 超过 u32".to_owned())?;
    let mut split_half = u32::from(layout == RotaryLayout::SplitHalf);
    let mut arguments = [
        (&mut d_query as *mut *mut c_void).cast(),
        (&mut d_key as *mut *mut c_void).cast(),
        (&mut d_cosine as *mut *mut c_void).cast(),
        (&mut d_sine as *mut *mut c_void).cast(),
        (&mut d_positions as *mut *mut c_void).cast(),
        (&mut d_query_output as *mut *mut c_void).cast(),
        (&mut d_key_output as *mut *mut c_void).cast(),
        (&mut rows as *mut u32).cast(),
        (&mut query_cols as *mut u32).cast(),
        (&mut query_head_count as *mut u32).cast(),
        (&mut key_cols as *mut u32).cast(),
        (&mut key_head_count as *mut u32).cast(),
        (&mut rotary_dim as *mut u32).cast(),
        (&mut split_half as *mut u32).cast(),
    ];
    let columns = query_cols.checked_add(key_cols).ok_or("resident segmented RoPE columns 溢出")?;
    let elements = rows.checked_mul(columns).ok_or("resident segmented RoPE elements 溢出")?;
    launch_tensor_kernel(functions.rope_segmented_pair, elements.div_ceil(256), 256, &mut arguments, "HIP resident segmented RoPE pair")?;
    Ok((query_output, key_output))
}

#[allow(clippy::too_many_arguments)]
pub fn try_rope_pair_resident_f32(
    device_id: i32,
    query: DeviceBuffer,
    key: DeviceBuffer,
    rows: usize,
    cols: usize,
    head_count: usize,
    rotary_dim: usize,
    layout: RotaryLayout,
    position: usize,
    cos: &[f32],
    sin: &[f32],
    prefix: bool,
) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    let input_bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(4)).ok_or("resident paired RoPE 大小溢出")?;
    validate_resident(&query, device_id, input_bytes, "paired RoPE query")?;
    validate_resident(&key, device_id, input_bytes, "paired RoPE key")?;
    let half = rotary_dim / 2;
    let begin = position.checked_mul(half).ok_or("resident paired RoPE table offset 溢出")?;
    let end = position.checked_add(rows).and_then(|n| n.checked_mul(half)).ok_or("resident paired RoPE table 大小溢出")?;
    if end > cos.len() || end > sin.len() {
        return Err("resident paired RoPE table 太短".to_owned());
    }
    set_device(device_id)?;
    let query_output = DeviceBuffer::allocate(device_id, input_bytes)?;
    let key_output = DeviceBuffer::allocate(device_id, input_bytes)?;
    let functions = tensor_functions(device_id)?;
    let cosine = &cos[begin..end];
    let sine = &sin[begin..end];
    let table_bytes = std::mem::size_of_val(cosine);
    with_tensor_workspace(device_id, &[table_bytes, table_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(cosine.as_ptr().cast(), table_bytes) })?;
        workspace.buffer(1).copy_from_host(unsafe { std::slice::from_raw_parts(sine.as_ptr().cast(), table_bytes) })?;
        let rows = u32::try_from(rows).map_err(|_| "resident paired RoPE rows 超过 u32".to_owned())?;
        let cols = u32::try_from(cols).map_err(|_| "resident paired RoPE cols 超过 u32".to_owned())?;
        let head_count = u32::try_from(head_count).map_err(|_| "resident paired RoPE heads 超过 u32".to_owned())?;
        let rotary_dim = u32::try_from(rotary_dim).map_err(|_| "resident paired RoPE dim 超过 u32".to_owned())?;
        let elements = rows.checked_mul(cols).ok_or("resident paired RoPE elements 溢出")?;
        let launch_rope = |input: &DeviceBuffer, output: &DeviceBuffer, action: &str| {
            let mut d_input = input.pointer;
            let mut d_cosine = workspace.buffer(0).pointer;
            let mut d_sine = workspace.buffer(1).pointer;
            let mut d_output = output.pointer;
            let mut rows = rows;
            let mut cols = cols;
            let mut head_count = head_count;
            let mut rotary_dim = rotary_dim;
            let mut table_position = 0_u32;
            let mut split_half = u32::from(layout == RotaryLayout::SplitHalf);
            let mut prefix = u32::from(prefix);
            let mut arguments = [
                (&mut d_input as *mut *mut c_void).cast(),
                (&mut d_cosine as *mut *mut c_void).cast(),
                (&mut d_sine as *mut *mut c_void).cast(),
                (&mut d_output as *mut *mut c_void).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut cols as *mut u32).cast(),
                (&mut head_count as *mut u32).cast(),
                (&mut rotary_dim as *mut u32).cast(),
                (&mut table_position as *mut u32).cast(),
                (&mut split_half as *mut u32).cast(),
                (&mut prefix as *mut u32).cast(),
            ];
            launch_tensor_kernel(functions.rope, elements.div_ceil(256), 256, &mut arguments, action)
        };
        launch_rope(&query, &query_output, "HIP resident paired query rope")?;
        launch_rope(&key, &key_output, "HIP resident paired key rope")?;
        synchronize_device(device_id, "paired Q/K RoPE before releasing inputs")
    })?;
    drop(query);
    drop(key);
    Ok((query_output, key_output))
}
